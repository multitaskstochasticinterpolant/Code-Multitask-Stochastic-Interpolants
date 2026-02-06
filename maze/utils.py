import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.func import vmap
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import os
import sys
from dotenv import load_dotenv
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from functools import wraps
import gc

Tensor = type(torch.tensor([]))

from model import VelocityFieldTS, Interpolant

inter = 6

length = 300

criterion = MSELoss(reduction="sum")

device = "cuda" if torch.cuda.is_available() else "cpu" 

def generate_points(goal_point: torch.Tensor, dataset: torch.Tensor, n: int = 10, name: str = None):
    """
    Data comes in the order: position, velocity of the start point, then the position of the end point.  dataset and goal_point should be normalized.
    """
    points = dataset[np.random.randint(low=0, high=len(dataset), size = (n,))]
    points = np.concatenate((points[:, np.newaxis,:2], np.repeat(goal_point[:, np.newaxis, :], repeats=n, axis=0)), axis = 1)
    if name is None:
        return points
    with open(name, "wb") as f:
        pkl.dump(points, f)


def get_keys(h5file):
    """
    Return all keys within h5file.
    """
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def loss_fn(b: VelocityFieldTS, interpolant: Interpolant, x0: Tensor, x1: Tensor, alpha: Tensor):
    """
    Interpolant loss function for a single datapoint of (x0, x1, alpha). This function is aimed to be vectorized.
    Args:
        b (VelocityFieldsTS). Generator.
        interpolant (Interpolant). Interpolation function.
        x0 (torch.Tensor). Sample from base distribution.
        x1 (torch.Tensor). Sample from target distribution.
        alpha (torch.Tensor). Interpolation parameter.
    Returns:
        The loss between the ground truth and the model prediction.
    """
    It   = interpolant.xt(x0, x1, alpha)
    dtIt = interpolant.dtxt(x0, x1)
    bt          = b.forward(It, alpha)
    loss        = criterion(bt.squeeze(), dtIt.squeeze())/len(x0)
    return loss

def linear_interpolation(points, inter):
    """
    Should return an array of inter points. The first point of the output is points[0]. The output does not include points[1] as it is already included in the next call to linear_interpolation.
    The interpolation shall be linear, and the points equally spaced.
    Args:
        points (torch.Tensor). Couple of points to be interpolated.
        inter (int). Number of points to add in linear interpolation.
    Returns:
        An array of points linearly interpolated between points[0] (included) and points[1] (not included).
    """
    point_1 = points[0]
    point_2 = points[1]
    slope = point_2 - point_1
    return np.array([point_1 + i/inter*slope for i in range(inter)])

def group_pair(obs: np.ndarray):
    """
    Pair together two consecutives points.
    Args:
        obs (numpy array). Array of 2D points.
    Returns:
        An array of pair 2D points.
    """
    _, c, _ = obs.shape
    grouped = np.empty((c, 2, 2))
    for i in range(c - 1):
        grouped[i] = np.array([obs[0,i], obs[0,i+1]])
    grouped[-1] = np.array([obs[0, -2], obs[0, -1]], dtype=float)
    return grouped

def interpolate(obs: np.ndarray, length=300, inter=6):
    """
    Add new points by interpolation between two consecutive points in obs.
    """
    pair = group_pair(obs)
    vect_inter = np.vectorize(linear_interpolation, signature="(m, n)->(k, n)", excluded=('inter',1), otypes=[float])
    pair_interp = vect_inter(pair, inter=inter)
    res = np.empty((length, 2), dtype=float)
    for i in range(0, length, inter):
        res[i: i+inter] = pair_interp[i//inter]
    return res

def setup_data(filename:str, scaler=StandardScaler()):
    """
    Setup the data by loading and then scaling it.
    Data is a dictionary made out of several keys, among them 'observations', and 'actions'. 'observations' is composed of four columns. The two first ones are the x and y position's coordinate, and the two lasts are the point's velocity components.
    Args:
        filename (str): Path where data is stored.
        scaler: Scaler to apply to normalize data.
    Returns:
        A fitted scaler, a dataloader, observations and actions array.
    """
    file = h5py.File("maze2d-medium-sparse-v1.hdf5", "r")
    print(get_keys(file))
    if scaler is not None:
        observations=scaler.fit_transform(file["observations"][:, :2])
        vel = scaler.transform(file["observations"][:, 2:])
    else:
        observations=file["observations"][:, :2]
        vel = file["observations"][:, 2:]
    observations = np.concatenate((observations, vel), axis = -1)
    Actions = file["actions"]
    print(observations.shape)
    Path_Loader = DataLoader(dataset=observations, batch_size=5000, shuffle=True)
    return scaler, Path_Loader, observations, Actions

    
def wrapper_memory(func):
    """
    Wrapper to check if a function has GPU memory leak.
    """
    @wraps(func)
    def memory_leak(*args, **kwargs):
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        a = mem.free
        x = func(*args, **kwargs)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"{(mem.free - a)/1024**2:5.2f}MB")
        return x
    return memory_leak

def setup():
    """
    Add necessary paths to python PATH environment variable to make all import of module possible.
    Args:
        None
    Returns:
        None
    """
    load_dotenv() #To load environment variables.
    
    #Add path to mujoco libraries.
    path1=os.path.join(os.getcwd(), "lib/python3.10/site-packages/.mujoco/mujoco210/bin")
    path2=os.path.join(os.getcwd(), "lib/python3.10/site-packages/.mujoco/mujoco210")
    path3=os.path.join(os.getcwd(), "lib/python3.10/site-packages/setuptools")
    
    if path1 not in sys.path:
        sys.path.append(path1)
    if path2 not in sys.path:
        sys.path.append(path2)
    if path3 not in sys.path:
        sys.path.append(path3)


def test_Unet(array, n=10, length=300, inter=6):
    """
    Test function to check everything is well-shaped to feed-forward the U-Net neural network.
    n and dim are respectively the batch size and the dimension of the data we deal with.

    Args:
        n(int): Batch size.
        length(int): Path length
        inter(int): space between each point in the path
    """
    from model import Interpolant, VelocityFieldTS
    from distributions import BaseDistribution, target

    b =  VelocityFieldTS(init_features=length//inter, device=device).to(device)
    interpolant = Interpolant()
    mean=torch.zeros(1)
    cov=torch.eye(1)
    base = BaseDistribution(mean=mean, cov=cov, device=device)

    x0s = base.sample(n=n, l=length//inter)
    x1 = target(n=n, l=length, inter=inter, array=array, device=device)
    alpha  = torch.rand(n, length // inter).to(device)
    l = loss_fn(b, interpolant, x0s, x1, alpha)
    return(l.item())

def test_Transformer(array, n=10, length=300, inter=6):
    """
    Test function to check everything is well-shaped to feed-forward the Transformer neural network.
    n and dim are respectively the batch size and the dimension of the data we deal with.

    Args:
        n(int): Batch size.
        length(int): Path length
        inter(int): space between each point in the path

    """
    from model import Interpolant
    from transformer import Transformer
    from distributions import BaseDistribution, target

    b = Transformer(x_dim=2, external_cond_dim=0, size=128, num_layers=12, nhead=4, dim_feedforward=512, dropout=0.0).to(device)
    interpolant = Interpolant()
    mean=torch.zeros(1)
    cov=torch.eye(1)
    base = BaseDistribution(mean=mean, cov=cov, device=device)

    x0s = base.sample(n=n, l=length//inter).squeeze()
    x1 = target(n=n, l=length, inter=inter, array=array, device=device).squeeze()
    
    #Switch batch and sequence axis
    x0s=torch.transpose(x0s, 0, 1)
    x1=torch.transpose(x1, 0, 1)
    
    alpha=torch.rand(n, length // inter).to(device)
    alpha=torch.transpose(alpha, 0, 1)[...,None]
    print(x0s.shape, x1.shape, alpha.shape)
    l = loss_fn_Transformer(b, interpolant, x0s, x1, alpha)
    return(l.item())
