# Multitask learning with Stochastic Interpolants

This repository contains the code to reproduce some of the numerical experiments displayed in the paper [multitask learning with Stochastic Interpolants](https://arxiv.org/abs/2508.04605). More specifically, you can find the notebooks written for the MNIST, $\varphi^4$ and maze experiments. Feel free to check out the [website](https://multitaskstochasticinterpolant.github.io/) for visual illustrations!

The weights of no models are made public; you need to train the neural networks yourself.

**TO DO: Modify the arxiv paper to add the repo's link**

### Overview

This paper introduces an operator-based paradigm for generating data with stochastic interpolants. While trivially incorporating the previous formulations, it offers more flexibility and multitask inference **with one single trained model**. Our experiments focuse on: *Inpainting*, *Posterior Sampling* and *motion planning*. Each are detailed below on their dedicated section.

![Overview Figure](./image/multitask_fig.jpg)


### MNIST

We performed on the **MNIST dataset**. Adapting it to higher-dimensional image is no tough challenge though. To demonstrate the variety of inpainting task our model can perform, we use a large range of mask. Notice how a figure can mutate (e.g. a $6$ transform into a $3$) when you apply the mask and make the underlying value ambiguous. Running the notebook should be straightforward.

### $\varphi^4$ model of spins

The $\varphi^4$ model stem from statistical field theory, similar to the spin model but where the field can take continuous values on the nodes of a discrete lattice $\Lambda$. An challenge of paramount importance in physics is the computation of expectation over all field's configuration. The probability of measuring a given field configuration with energy $E$ is given by Boltzmann distribution: $p(E)=Z^{-1}e^{-\beta E}$. Unfortunately, in the high-dimensional setting and near phase transition, sampling a configuration is computationnaly very challenging, it can introduce bias or is very time consuming. In this context, inpainting and posterior sampling can be of great help.

Inpainting enables you to sample a configuration close to the one at hand by masking some nodes of the discrete. This is ideal for sampling new lattices conditioned on the values of the unmasked nodes.

On the other hand, posterior sampling considers the linearly shifted Boltzmann distribution: $p(E) ∝ e^{(–\beta E + (h, \varphi))} = e^{(–\beta E + h \sum\limits_{i \in \Lambda}\varphi\_i)}$. Physically speaking, h plays the role of a magnetic field and shifts the spins' mean value. The quadratic case has also been theoretically investigated; some numerical experiments will be conducted in the coming months.

You should train the neural network from scratch here.

### Maze

The idea is to apply stochastic interpolation to a maze in a similar way to that described in [this paper](https://arxiv.org/abs/2407.01392). We wanted to compare our method with theirs based on Figure 2. However, this was unsuccessful; we suspect the numbers shown are incorrect. Despite numerous attempts and questions to one of the original authors, I could not reproduce the results. 

The way we proceed is mucher simpler than what is proposed in the paper cited above. In this environment, the original path is 600 points long, but we have decided to generate only one point every six. It drastically reduces the dimension of data space to 50 points, making learning easier. All missing points are recovered at the end by a linear interpolation between every pairs.

The dataset used is free to download [here](https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5)

If you want the weights of the numerical experiment, please ask me directly and I can provide it for you.

#### Maze planning with d4rl

This section only covers a small part of the maze notebook. It comes after path generation to smooth it out and extract a score. If you are only interested in raw path generation, there is no need to bother with this. The exact depedencies required are listed in `requirements.txt` and `requirement_extra.txt` files.

It heavily uses the gym environment [Point maze](https://robotics.farama.org/envs/maze/point_maze/), which runs upon complicated dependancies, such as the deprecated [d4rl](https://github.com/Farama-Foundation/D4RL) library. The code I use is an adapted replica of the `interact` function you can find [here](https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/df_planning.py#L261).

Also, be careful, you need `python==3.10`, `pip==21` and `setuptools==65.5.0`, no newer. I heavily recommend creating a conda virtual environment:
```
conda create -n maze-env python=3.10 pip=21 setuptools=65.5.0
```

Do **not** change the environment name please.

Then, type in the CLI:
```
pip install -r requirements.txt extra_requirements.txt
```

To install Mujoco, download the archive [here](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco) and follow the instructions. The `.mujoco` folder is certainly at `$HOME/miniconda3/envs/maze-env/lib/python3.10/site-packages/.mujoco` if you use miniconda, or `$HOME/anaconda3/envs/maze-env/lib/python3.10/site-packages/.mujoco` if you use anacond.

When first imported, `d4rl` set up `mujoco\_py` by compiling a Cython file. It is a complicated pre-processing step, I encountered a lot of issues when I tried to import `d4rl` on the first time. You should follow the next instructions carefully:

Install `libosmesa6-dev` and `patchelf` package.
```
sudo apt-get install libosmesa6-dev patchelf
```

If you use `miniconda3` to manage your virtual environments, you should use the `.env` file I provide in the `maze` folder. Also, you need to load the environment variable `LD\_LIBRARY\_PATH` **before** the jupyter kernel starts, it is important. The only solution is to directly specify its value in a configuration file. If your IDE is VS Code, search and open `settings.json` in the search bar activated by `CTRL+SHIFT+P`. Then, you simply add the following line inside:

```
{
  "python.envFile": "${workspaceFolder}/maze/.env",
}
```
to specify where to look for `.env` file when loading the environment variable.

Finally, I had to make a small modifications in two of the files called by `env.reset()`. The native `gym` library would not allow me to specify the path starting point. To add this feature, you should replace the files `mujoco_env.py` and `maze_model.py` at `$HOME/miniconda3/envs/maze-env/lib/python3.10/site-packages/gym/envs/mujoco/` and `$HOME/miniconda3/envs/maze-env/lib/python3.10/site-packages/d4rl/pointmaze` by the ones I provide.

## Contact

If you have any trouble running the notebooks or have any question, don't hesitate to directly contact me at hugonegrel13@gmail.com.

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


