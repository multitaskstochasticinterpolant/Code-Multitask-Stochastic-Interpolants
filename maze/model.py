import torch
import torch.nn as nn
from torch.func import vmap
import numpy as np
import math
import torch.nn.functional as F

Tensor = type(torch.tensor([]))

class Interpolant:
    """
    Base Class for interpolant. 
    """
    def _single_xt(self, x0, x1, alpha):
        return alpha*x0 + (1 - alpha)*x1
        
    def alpha(self, alpha):
        return alpha
        
    def dotalpha(self, alpha):
        return -1.0 + 0*alpha
        
    def beta(self, alpha):
        return 1 - alpha
        
    def dotbeta(self, alpha):
        return 1.0 + 0*alpha
        
    def _single_dtxt(self, x0, x1):
        return x0 - x1
    
    def xt(self, x0, x1, alpha):
        #Repeat along the two coordinates
        if len(alpha.shape) == 2:
            alpha = alpha[...,np.newaxis].repeat(1, 1, 2)
        # elif len(alpha.shape) == 3:
        #     alpha = alpha[...,np.newaxis].repeat(1, 1, 2)
        return vmap(self._single_xt, in_dims=(0, 0, 0))(x0,x1,alpha) #Vectorize along each axis
    
    def dtxt(self, x0, x1):
        return vmap(self._single_dtxt, in_dims=(0, 0))(x0,x1)


# This is the utils file
def zero_out(layer):
    """
    Zero-out all parameters in layer.
    """
    for p in layer.parameters():
        p.detach().zero_() #To manually change the values of the tensor entries, you need to detach the tensor.
    return layer

# AdaGN according to paper "Diffusion Models Beat GANs on Image Synthesis"
class AdaNorm(nn.Module):
    def __init__(self, num_channel: int):
        super().__init__()
        num_group = int(num_channel/16) # According to group norm paper, 16 channels per group produces the best result
        self.gnorm = nn.GroupNorm(num_group, num_channel, affine=False)

    def forward(self, tensor: torch.Tensor, emb: torch.Tensor):
        """
        Apply an affine transormation to tensor by chunking embedding into two equally sized pieces.
        """
        scale, shift = torch.chunk(emb, 2, dim=1)
        tensor = self.gnorm(tensor)
        tensor = tensor * (1 + scale) + shift
        return tensor

class MyGroupNorm(nn.Module):
    def __init__(self, num_channel: int):
        super().__init__()
        num_group = int(num_channel/16) # According to group norm paper, 16 channels per group produces the best result
        self.gnorm = nn.GroupNorm(num_group, num_channel, affine=False)

    def forward(self, tensor: torch.Tensor):
        return self.gnorm(tensor)
        
class ResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, emb_dim: int, up: bool = False, down: bool = False):
        super().__init__()
        assert not (up and down), "You can not have both UP and DOWN set to True"
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2*out_channel))
        if up:
            self.change_size = nn.Upsample(scale_factor=(2, 1), mode='nearest') #Upsampling in one dimension. For 
        elif down:
            self.change_size = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)) #Downsampling in one dimension
        else:
            self.change_size = nn.Identity() #For first and mid-blocks.

        # Normalization
        self.gnorm1 = MyGroupNorm(in_channel)
        self.gnorm2 = AdaNorm(out_channel)

        # Convolution
        
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(5,3), padding=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(5,3), padding=(2, 1))


        if in_channel != out_channel:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size = 1)
        else:
            self.conv3 = nn.Identity()

        # Combine input stage
        self.input = nn.Sequential(
            self.gnorm1,
            nn.SiLU(),
            self.change_size,
            self.conv1
        )

        # Combine output stage
        self.output = nn.Sequential(
            nn.SiLU(),
            zero_out(self.conv2)
        )

        # Skip connection
        self.skip_connection = nn.Sequential(
            self.change_size,
            self.conv3
        )

        # Embedding
        self.embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_channel)
        )

    def forward(self, tensor: torch.Tensor, emb: torch.Tensor):
        emb = self.embed(emb).view(tensor.shape[0], -1, 1, 1)

        h = self.input(tensor)
        h = self.gnorm2(h, emb)
        h = self.output(h)
        x = self.skip_connection(tensor)

        return x + h

class SelfAttention(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.num_head = int(channel/32)
        
    def forward(self, tensor: torch.Tensor):
        batch, channel, length = tensor.shape
        ch = channel // 3 // self.num_head
        q, k, v = tensor.chunk(3, dim = 1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # The code below is from Diffusion Model Beat GANs on Image Synthesis paper code
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(batch * self.num_head, ch, length),
            (k * scale).view(batch * self.num_head, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(batch * self.num_head, ch, length))
        return a.reshape(batch, -1, length)

class Attention(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.gnorm = MyGroupNorm(channel)
        self.qkv = nn.Conv1d(channel, channel * 3, 1) #key-query-vector neural network
        self.attention = SelfAttention(channel) #Self attention neural network.
        self.output = zero_out(nn.Conv1d(channel, channel, 1))
        
    #Leak
    def forward(self, tensor: torch.Tensor):
        # Perform self attention
        batch, channel, width, height = tensor.shape
        tensor = tensor.reshape(batch, channel, -1)
        # Skip connection
        tensor_skip = tensor
        tensor = self.gnorm(tensor)
        tensor = self.qkv(tensor)
        tensor = self.attention(tensor)
        tensor = self.output(tensor)

        # Adding the skip connection tensor back to the current tensor
        tensor = tensor + tensor_skip

        tensor = tensor.reshape(batch, channel, width, height)
        return tensor

class UNet(nn.Module):
    def __init__(self, emb_dim: int, TS_channel: int = 1, depth: int = 2):
        super().__init__()

        # Create model architecture
        # channels = [32, 64, 128, 256]
        # channels = [48, 80, 160, 256] #Architecture for L50
        channels = [80, 160, 256, 512] #Architecture for L100
        # channels = [32, 64, 80, 128]
        attention_channel = channels[-2:]
        self.encoder = nn.ModuleList([nn.ModuleList([nn.Conv2d(in_channels=TS_channel, out_channels=channels[0], kernel_size=(5,3), padding=(2, 1))])])
        self.decoder = nn.ModuleList()

        skip_channel = [channels[0]]

        # Encoder
        for i in range(len(channels)):
            for _ in range(depth):
                layer = nn.ModuleList()
                layer.append(ResBlock(channels[i], channels[i], emb_dim = emb_dim))
                if channels[i] in attention_channel:
                    layer.append(Attention(channels[i]))
                self.encoder.append(layer)
                skip_channel.append(channels[i])

            if i != len(channels)-1:
                layer = nn.ModuleList()
                layer.append(ResBlock(channels[i], channels[i + 1], down=True, emb_dim = emb_dim)) #Add a final residual block
                self.encoder.append(layer)
                skip_channel.append(channels[i+1])

        # Bottleneck
        self.bottle_neck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], emb_dim = emb_dim),
            Attention(channels[-1]),
            ResBlock(channels[-1], channels[-1], emb_dim = emb_dim),
        ])

        # Decoder
        for i in range(len(channels)-1, -1, -1):
            for block in range(depth+1):
                layer = nn.ModuleList()
                layer.append(ResBlock(channels[i] + skip_channel.pop(), channels[i], emb_dim = emb_dim))
                if channels[i] in attention_channel:
                    layer.append(Attention(channels[i]))

                if i != 0 and block == depth:
                    layer.append(ResBlock(channels[i], channels[i - 1], up=True, emb_dim = emb_dim))

                self.decoder.append(layer)

        #Not necessary
        # # Create time embedding
        # self.time_embedding = nn.Sequential(
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(emb_dim, emb_dim)
        # )

        # Output kernels to change back to image channel
        self.out = nn.Sequential(
            MyGroupNorm(channels[0]),
            nn.SiLU(),
            zero_out(nn.Conv2d(channels[0], TS_channel, 3, padding=1)),
        )
    
    def forward(self, tensor: torch.Tensor, alpha: torch.Tensor):
        # Creating embedding
        # embedding = self.time_embedding(time_embedding)
        embedding = alpha
        skip_connection = []
        
        # Encoder-Memory leak
        for layer in self.encoder:
            for module in layer:
                if(isinstance(module, ResBlock)):
                    tensor = module(tensor, embedding)
                else:
                    tensor = module(tensor)

            skip_connection.append(tensor)

        # Bottleneck-Memory leak
        for module in self.bottle_neck:
            if(isinstance(module, ResBlock)):
                tensor = module(tensor, embedding)
            else:
                tensor = module(tensor)

        # Decoder-Memory leak
        for layer in self.decoder:
            P = skip_connection.pop()
            if P.shape[-2] == tensor.shape[-2] + 1:
                tensor= F.pad(tensor, (0, 0, 1, 0), "constant", 0) 
            tensor = torch.concatenate((tensor, P), dim = 1)
            for module in layer:
                if(isinstance(module, ResBlock)):
                    tensor = module(tensor, embedding)
                else:
                    tensor = module(tensor)

        tensor = self.out(tensor)

        return tensor

class VelocityFieldTS(nn.Module):
    def __init__(self, init_features, device='cpu'):
        super(VelocityFieldTS, self).__init__()
        # Increment the input channels by 1 to accommodate the alpha dimension
        self.unet = UNet(TS_channel = 1, depth = 2, emb_dim = init_features).to(device)

    def forward(self, x, alpha):
        """
        Important Note: alpha should be a linear vector, and not a matrix.
        """

        #Memory leak
        genn = self.unet(x, alpha)

        return genn