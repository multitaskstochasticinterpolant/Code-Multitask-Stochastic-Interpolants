## Multitask learning with Stochastic Interpolants

This repository contains the code to reproduce some of the numerical experiments displayed in the paper [multitask learning with Stochastic Interpolants](https://arxiv.org/abs/2508.04605). More specifically, you can find the notebooks written for the MNIST, $\varphi^4$ and maze experiments. Feel free to check out the [website](https://multitaskstochasticinterpolant.github.io/) for visual illustrations!

### Overview

This paper introduces an operator-based paradigm for generating data with stochastic interpolants. While trivially incorporating the previous formulations, it offers more flexibility and multitask inference **with one single trained model**. Our experiments focuse on: *Inpainting*, *Posterior Sampling* and *motion planning*. Each are detailed below on their dedicated section.


![Overview Figure](./image/multitask_fig.jpg)



### MNIST

We performed on the **MNIST dataset**. Adapting it to higher-dimensional image is no challenge though. To demonstrate the variety of inpainting task our model can perform, we use a large range of mask. Notice how a figure can mutate (e.g. a $6$ transform into a $3$) when you apply the mask and make the underlying value ambiguous.

### $\varphi^4$

The $\varphi^4$ model stem from statistical field theory, similar to the spin model but where the field can take continuous values at the lattice's nodes. An paramount issue in physics is to compute expectation over all field's configuration weighted by Boltzmann distribution. Unfortunately, in the high-dimensional setting and near phase transition, sampling a configuration is computationnaly challenging. In this context, inpainting and posterior sampling can be of great help. Inpainting lets you sample, by masking some nodes, a configuration close to the one at hand. Posterior sampling

### Maze

The idea is to apply stochastic interpolants on a maze, similarly to what is done in [this paper](https://arxiv.org/abs/2407.01392). We wanted to compare our method with theirs, based on the figure displayed on Table 2. However, it was not successful, we suspect the numbers shown are not correct. Despite my numerous attempts and my questions to an original author, I could not reproduce them.

#### Maze planning with d4rl

This section only concerns a small part of the maze notebook. It only comes after the path generation to smooth it and extract a score out of it. If you're only interested in path generation, no need to bother with this. 

It heavily uses the gym environment [Point maze](https://robotics.farama.org/envs/maze/point_maze/), which runs upon complicated dependancies, such as the deprecated [d4rl](https://github.com/Farama-Foundation/D4RL) library. The code I use is a adapted replica of the `interact` function you can find [here](https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/df_planning.py#L261).

Also, be careful, you need Python 3.10, no newer.


## Citations

If you find this work useful for your research, please cite us.

```
@misc{negrel2025multitasklearningstochasticinterpolants,
      title={Multitask Learning with Stochastic Interpolants}, 
      author={Hugo Negrel and Florentin Coeurdoux and Michael S. Albergo and Eric Vanden-Eijnden},
      year={2025},
      eprint={2508.04605},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.04605}, 
}
```


