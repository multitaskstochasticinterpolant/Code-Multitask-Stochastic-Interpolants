from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
import os
Tensor = type(torch.tensor([]))

def get_maze_grid():
    """
    Return a string representation of the maze. '#' is a wall case, 'O' is a free case and 'G' is the goal case.
    """
    maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


class Maze2dOfflineRLDataset(torch.utils.data.Dataset):
    """
    Wrapping class of the Maze2d offline dataset.
    """
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.save_dir = './'
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        self.gamma = 1
        self.n_frames = 600 + 1
        self.total_steps = len(self.dataset["observations"])
        self.dataset["values"] = self.compute_value(self.dataset["rewards"]) * (1 - self.gamma) * 4 - 1

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            value[i] += self.gamma * value[i + 1]
        return value

    def __len__(self):
        return self.total_steps - self.n_frames + 1

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.dataset["observations"][idx : idx + self.n_frames]).float()
        action = torch.from_numpy(self.dataset["actions"][idx : idx + self.n_frames]).float()
        reward = torch.from_numpy(self.dataset["rewards"][idx : idx + self.n_frames]).float()
        value = torch.from_numpy(self.dataset["values"][idx : idx + self.n_frames]).float()

        done = np.zeros(self.n_frames, dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)

        goal = torch.zeros((self.n_frames, 0))

        return observation, action, reward, nonterminal

    def get_dataset(self):
        h5path = os.path.join(os.getcwd(), "maze2d-medium-sparse-v1.hdf5")
        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in get_keys(dataset_file):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]
    
        N_samples = data_dict["observations"].shape[0]
    
        if data_dict["rewards"].shape == (N_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]
    
        if data_dict["terminals"].shape == (N_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]
    
        return data_dict

def in_cube(point, xy, side_size):
    """
    Check if 'point' is in a cube with bottom-left coordinate xy and of size side_size.
    """
    if point[0] > xy[0] and point[0] < xy[0] + side_size:
        if point[1] > xy[1] and point[1] < xy[1] + side_size:
            return True
    return False

def in_wall(point: np.ndarray):
    """
    Return a boolean stating if a point lie inside a wall box.
    """
    maze_grid = get_maze_grid()
    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#" and in_cube(point, (i+0.5, j+.5), 1):
                return True
    return False

def count_in_wall(pathway, scaler=None):
    """
    Count the number of point in pathway within a wall box.
    """
    if scaler is not None:
        pathway = scaler.inverse_transform(pathway)
    in_wall_vect = np.vectorize(in_wall, signature="(n)->()")
    return np.count_nonzero(in_wall_vect(pathway))

def draw_star(center, radius, num_points=5, color="black", ax=None):
    """
    Draw a star centered around 'center' and with radius 'radius'.
    """
    angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
    inner_radius = radius / 2.0

    points = []
    for angle in angles:
        points.extend(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                center[1] + inner_radius * np.sin(angle + np.pi / num_points),
            ]
        )

    star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
    if ax is None:
        plt.gca().add_patch(star)
    else:
        ax.add_patch(star)

def plot_maze(obs: np.ndarray, scaler = None, path: str = None, title: str = None, fig_ax=None, bar: bool=True, star: bool=True):
    """
    Plot the pathway with the maze as background.
    Args:
        obs (np.ndarray): array data to plot in the maze. If normalized, you must provide a scaler.
        scaler: Optional. To unnormalize obs. If set to None, the data are supposed to be already unnormalized.
        path (str): Optional. Indicate where to save the plot. If set to None, the plot is not saved.
        title (str): Optional. The plot title.
        fig_ax: Optional. The matplotlib axis on which you want to plot.
        bar (bool). Optional. Whether to display a bar or not.
    Returns:
        A plot with the pathway traced on it with the maze as a background.
    """
    if len(obs.shape) == 2: #I use it when I want to plot only endpoints.
        c, _ = obs.shape #c is the number of points in the path
        if scaler is not None:
            inv_transform = scaler.inverse_transform
            obs = inv_transform(obs)
        obs = obs[np.newaxis, ...]
    elif len(obs.shape) == 3:
        _, c, _ = obs.shape #c is the number of points in the path
        if scaler is not None:
            for i in range(len(obs)):
                obs[i] = scaler.inverse_transform(obs[i])

    if fig_ax is None:
        fig=plt.gcf()
        ax=plt.gca()
    else:
        fig, ax=fig_ax

    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    maze_grid = get_maze_grid()

    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                ax.add_patch(square)

    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("lightgray")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(1, len(maze_grid), 0.5), minor=True)
    ax.set_yticks(np.arange(1, len(maze_grid[0]), 0.5), minor=True)
    ax.set_xlim([0.5, len(maze_grid) + 0.5])
    ax.set_ylim([0.5, len(maze_grid[0]) + 0.5])

    ax.grid(True, color="white", which="minor", linewidth=4)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)

    if title:
        ax.set_title(title)

    for i in range(len(obs)): #Number of batch
        scatter = ax.scatter(obs[i, :, 0], obs[i, :, 1], c=np.arange(c), cmap="Reds")
    
        start_x, start_y = obs[i, 0, :2] #Start point
        start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
        ax.add_patch(start_circle)
        inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
        ax.add_patch(inner_circle)
    
        goal_x, goal_y = obs[i, -1, :2] # End points
        if star:
            goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
            ax.add_patch(goal_circle) #Add a circle patch
            draw_star((goal_x, goal_y), radius=0.08, ax=ax) #Add the star at the end of pathway

    if bar:
        cbar = plt.colorbar(scatter, shrink=1)
        cbar.ax.text(.7, 1.02, 'end', ha='center', va='bottom', transform=cbar.ax.transAxes, size = "large")
        cbar.ax.text(.7, -0.05, 'start', ha='center', va='top', transform=cbar.ax.transAxes, size = "large")
        cbar.set_ticks([])
    print("Done.")
    if path is not None:
        print("Plot saved at", path)
        fig.savefig(path)
    
    return ax

env_id = "maze2d-medium-v1"

def plot_maze_tmp(obs, pos_const = None, fig_ax = None, title=None, save_path: str=None, bar: bool=False):
    """
    The data is assumed not normalized. There is no scaler as argument.s
    Add a supplementary pin to locate the constraint in the maze.
    """
    if fig_ax is None:
        fig=plt.gcf()
        ax=plt.gca()
    else:
        fig, ax=fig_ax
    scatter=ax.scatter(obs[:, 0], obs[:, 1], c=np.arange(len(obs)), cmap="Reds")
    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]

    # maze_string = gym.make(env_id).str_maze_spec
    grid = get_maze_grid()

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                ax.add_patch(square)

    start_x, start_y = obs[..., 0, :2]
    start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_circle)
    inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(inner_circle)

    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    def draw_rectangle(center, radius, num_points=5, color="black"):
        """
        Function to add an additional pin with a rectangular pattern to visualize the constraint enforced on the path. 
        """
        points = []
        inner_radius=radius/2
        points.extend(
            [
                center[0] + radius * 1/2,
                center[1] + radius * 1/2,
                center[0] + inner_radius * 1/2,
                center[1] + inner_radius * 1/2,

                center[0] - radius * 1/2,
                center[1] - radius * 1/2,
                center[0] - inner_radius * 1/2,
                center[1] - inner_radius * 1/2,
            ]
        )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)


    goal_x, goal_y = obs[..., -1, :2]
    if pos_const is not None:
        const_x, const_y=pos_const
        const_circle=plt.Circle((const_x, const_y), 0.16, facecolor="white", edgecolor="black")
        ax.add_patch(const_circle)
        draw_rectangle((const_x, const_y), radius=0.08)

    goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_circle)
    draw_star((goal_x, goal_y), radius=0.08)

    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("lightgray")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    ax.set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    ax.set_xlim([0.5, len(grid) + 0.5])
    ax.set_ylim([0.5, len(grid[0]) + 0.5])
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    ax.grid(True, color="white", which="minor", linewidth=4)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)

    if bar:
        plt.subplots_adjust(right=0.9, wspace=0.3)
        cbar_ax = fig.add_axes([.92, .32, 0.02, 0.35])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.ax.text(.7, 1.02, 'end', ha='center', va='bottom', transform=cbar.ax.transAxes, size = "large")
        cbar.ax.text(.7, -0.05, 'start', ha='center', va='top', transform=cbar.ax.transAxes, size = "large")
        cbar.set_ticks([])

    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
    print("Done.")


def plot_maze_smooth(obs: np.ndarray, Start_point, End_point, path: str = None, title: str = None, fig_ax=None):
    """
    Plot the pathway with the maze as background. The path data is assumed to be normalized, no scaler argument is proposed. No colorbar.
    This function should be solely used to plot the maze with 
    Args:
        obs (np.ndarray): array data to plot in the maze.
        save (bool): Indicate whether save plot or not.
        path (str): Indicate where to save the plot.
        title (str): The plot title.
        ax: The matplotlib axis on which you want to plot.
    """
    
    if fig_ax is None:
        fig=plt.gcf()
        ax=plt.gca()
    else:
        fig, ax = fig_ax
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    maze_grid = get_maze_grid()

    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                ax.add_patch(square)

    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("lightgray")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(1, len(maze_grid), 0.5), minor=True)
    ax.set_yticks(np.arange(1, len(maze_grid[0]), 0.5), minor=True)
    ax.set_xlim([0.5, len(maze_grid) + 0.5])
    ax.set_ylim([0.5, len(maze_grid[0]) + 0.5])

    ax.grid(True, color="white", which="minor", linewidth=4)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)

    if title:
        ax.set_title(title)
    
    c = len(obs)

    ax.scatter(obs[:,0], obs[:, 1], c=np.arange(c), cmap="Reds", linewidths=5)

    start_x, start_y = Start_point[0].item(), Start_point[1].item() #Start point
    start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_circle)
    inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(inner_circle)

    goal_x, goal_y = End_point[0].item(), End_point[-1].item() # End points
    goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_circle) #Add a circle patch
    draw_star((goal_x, goal_y), radius=0.08, ax=ax) #Add the star

    print("Done.")
    if path:
        fig.savefig(path)


def find_length(start_point: np.ndarray, end_point: np.ndarray, device, scaler=None):
    """
    Find out the most appropriate path length. This function is NOT vectorized.
    Args:
        start_point: (numpy array). Starting point of the pathway.
        end_point: (numpy array). End point of the pathway.
        device: (str). Device on which to run the tensors' computation.
        scaler. If set to None, start_point and end_point are already unnormalized.
    Returns:
        The appropriate path length to connect start_point and end_point.
    """
    Time = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    
    if scaler is not None:
        start_point=scaler.transform(start_point)
        end_point=scaler.transform(end_point)
    
    #Convert to tensor
    start_point, end_point = torch.tensor(start_point, device=device), torch.tensor(end_point, device=device)

    #Path length
    length=300
    #Number of path points between each generated points
    inter=6
    
    for idx in Time:
                    
        Alpha = torch.stack((torch.linspace(1.0,0.0, 80 + 1, device = device),)*((length//inter)), axis = 1)
        Alpha[:,idx] = 0
        Alpha[:,0] = 0
        
        mask = torch.ones((length//inter), device = device)
        mask[idx] = 0
        mask[0] = 0
        mask = torch.concatenate((mask[:,np.newaxis], mask[:,np.newaxis]), axis = 1)
        
        sde  = SDE(b, interpolant, n_step = 80, device = device, eps = .7, Alpha=Alpha, mask=mask)
        n_bs = 1 #Only one sample, no need to do more.
        
        x0s = base.sample(n=n_bs, l=length//inter)
        x0s = x0s.to(device)
        x_init = x0s.clone().detach()  # Assume this is your forward-solved endpoint
        x_init[:, 0, idx], x_init[:, 0, 0] = end_point, start_point
        
        # x_init = ode.solve(x_init)
        x_final = sde.solve(x_init).squeeze(2) #Remove the channel axis
        x1s = x_final[-1].cpu().detach().numpy() #Keep the last time step
        x1s = x1s[:, :idx+1]
        n = 600//idx
        x1s = interpolate(x1s, length=idx*n, inter=n)
        counter = count_in_wall(x1s, scaler=scaler)
        if not counter:
            return Time[idx]
            break
        print(counter)
        