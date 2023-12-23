import os
from functools import lru_cache

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyvoro
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import *

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 10


def save_trajectories(
    positions,
    mass,
    centroid,
    dcdt,
    dcdp,
    cost,
    t,
    dt,
    T,
    algorithm,
    env,
    save_params,
    phi,
):
    """
    Plots complete trajectories nicely for paper submission
    """
    path = "data/" + algorithm + save_params + "/"
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    num_frames = round(T / dt)
    # One plot with trajectories
    colors = cm.viridis(np.linspace(0, 1, positions.shape[1]))
    dpi = 600
    fs = (3.5, 3.5)
    msize = 4
    fig = plt.figure(figsize=fs, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.xaxis._axinfo["grid"].update(
        {"linewidth": 0.5, "linestyle": ":", "dash_capstyle": "round"}
    )
    ax.yaxis._axinfo["grid"].update(
        {"linewidth": 0.5, "linestyle": ":", "dash_capstyle": "round"}
    )
    ax.zaxis._axinfo["grid"].update(
        {"linewidth": 0.5, "linestyle": ":", "dash_capstyle": "round"}
    )
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.axes.set_xlim3d(left=env[0][0], right=env[0][1])
    ax.axes.set_ylim3d(bottom=env[1][0], top=env[1][1])
    ax.axes.set_zlim3d(bottom=env[2][0], top=env[2][1])
    fig.tight_layout()

    frame = num_frames - 2
    vor = pyvoro.compute_voronoi(positions[-1], env, dispersion=10)
    visualize_3d_voronoi(
        positions[-1],
        None,
        None,
        env,
        vor,
        frame,
        T,
        phi,
        cost[frame],
        path,
        algorithm,
        save=False,
        ax=ax,
        fig=fig,
    )

    def plot_stems():
        ax.plot(
            [positions[j, i, 0], positions[j, i, 0]],
            [positions[j, i, 1], positions[j, i, 1]],
            [env[2][0], positions[j, i, 2]],
            "--",
            linewidth=0.5,
            color="k",
            alpha=0.25,
            zorder=1,
        )

    def plot_density():
        def func(t):
            return [r * np.cos(t / tau), r * np.sin(t / tau), 0 * t]

        t_vec = np.linspace(0, T, num_frames)
        tau = 20
        r = 4
        trajectory = func(t_vec)
        levels = 8
        cmap = colormap(levels)
        c = cmap(0.5)
        order = 2
        # Trajectories
        ax.plot(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            "--",
            linewidth=1,
            color=c,
            zorder=order,
            label="Density\nCenter",
        )
        ax.plot(
            trajectory[0],
            trajectory[1],
            env[2][0],
            "--",
            linewidth=0.5,
            color=c,
            zorder=order,
        )
        # Markers
        msize = 2
        ax.plot(
            trajectory[0][-1],
            trajectory[1][-1],
            env[2][0],
            marker="o",
            markersize=msize / 2,
            color=c,
            zorder=order,
            alpha=0.5,
        )
        ax.plot(
            trajectory[0][0],
            trajectory[1][0],
            env[2][0],
            marker="s",
            markersize=msize / 2,
            color=c,
            zorder=order,
            alpha=0.5,
        )

    plot_density()

    for i in range(positions.shape[1]):
        for j in range(num_frames):
            n = 3
            if num_frames >= n:
                if (j % (round(num_frames / n)) == 0) or (j == (num_frames - 1)):
                    plot_stems()
            else:
                plot_stems()
        # projections
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            env[2][0],
            "-",
            color=colors[i],
            alpha=0.5,
            linewidth=0.7,
        )
        # agent trajectories
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            positions[:, i, 2],
            "-",
            color=colors[i],
            label=(r"$A_{{{:1d}}}$".format(i + 1)),
        )
        # markers
        order = 30
        ax.plot(
            positions[0, i, 0],
            positions[0, i, 1],
            positions[0, i, 2],
            marker="s",
            markersize=msize,
            markerfacecolor=colors[i],
            zorder=order,
            markeredgecolor="black",
        )
        ax.plot(
            positions[-1, i, 0],
            positions[-1, i, 1],
            positions[-1, i, 2],
            marker="o",
            markersize=msize,
            markerfacecolor=colors[i],
            zorder=order,
            markeredgecolor="black",
        )
        ax.plot(
            positions[0, i, 0],
            positions[0, i, 1],
            env[2][0],
            marker="s",
            markersize=msize / 2,
            color=colors[i],
            zorder=order,
            alpha=0.5,
        )
        ax.plot(
            positions[-1, i, 0],
            positions[-1, i, 1],
            env[2][0],
            marker="o",
            markersize=msize / 2,
            color=colors[i],
            zorder=order,
            alpha=0.5,
        )
    if positions.shape[1] <= 5:
        ax.legend(
            loc="upper left",
            ncol=positions.shape[1] + 1,
            bbox_to_anchor=(0.0, 1, 1.0, 0.05),
            borderaxespad=0,
        )
    plt.savefig(path + "trajectory" + ".png", dpi=600, bbox_inches="tight")
    plt.close()


def visualize_3d_voronoi(
    position,
    mu,
    sigma,
    env,
    vor,
    frame,
    t,
    phi,
    cost,
    name,
    algorithm,
    save=True,
    ax=None,
    fig=None,
):
    """
    Plots 3D voronoi diagrams
    """
    # Visualize partitions and starting positions
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")
    if save is True:
        ax.scatter3D(position[:, 0], position[:, 1], position[:, 2], marker="o", s=40)
        cost_val = np.array2string(cost, formatter={"float_kind": lambda x: "%.2f" % x})
        ax.text2D(0.3, 0.95, "Instantaneous Cost = " + cost_val, transform=ax.transAxes)
        ax.set_title(algorithm + " Coverage at time = " + str(t))
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)

    r = 4
    tau = 20
    center = np.array([r * np.cos((t) / tau), r * np.sin((t) / tau)])
    sigma = 2
    offset = 1.4 * sigma
    bound = [center - offset, center + offset]
    pts = 300
    X = np.linspace(bound[0][0], bound[1][0], num=pts)
    Y = np.linspace(bound[0][1], bound[1][1], num=pts)

    PHI = np.zeros([len(X), len(Y)])
    for i in range(len(X)):
        for j in range(len(Y)):
            PHI[j, i] = phi(X[i], Y[j], 0, 0, sigma, t)
    dr = X[1] - X[0]
    for i in range(pts):
        for j in range(pts):
            r = np.sqrt((X[i] - center[0]) ** 2 + (Y[j] - center[1]) ** 2)
            if (r - dr / 2) > offset:
                PHI[j, i] = "nan"
    levels = 8
    map = colormap(levels)
    CS = ax.contourf(
        X,
        Y,
        PHI,
        100,
        zdir="z",
        offset=0,
        cmap=map,
        levels=levels,
        alpha=0.5,
        zorder=1,
        antialiased=True,
    )

    CS2 = ax.contour(
        X,
        Y,
        PHI,
        8,
        colors="k",
        zdir="z",
        offset=0,
        linewidths=0.5,
        alpha=0.1,
        zorder=2,
        antialiased=True,
    )
    axins = inset_axes(ax, width="5%", height="80%", loc="center right", borderpad=-5)
    cbar = fig.colorbar(CS, cax=axins)
    cbar.ax.set_ylabel("Sliced Density", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)

    # Plot Voronoi regions
    plot_voronoi_regions(position, vor, ax=ax)

    # Plot settings for debugging
    if save:
        plt.show(block=False)
        plt.savefig(name + str(frame), dpi=600)
        plt.close()


@lru_cache(1)
def colormap(levels):
    """
    Helper function for contour maps
    """
    ncolors = 256
    color_array = plt.get_cmap("afmhot_r")(range(ncolors))
    map_object = LinearSegmentedColormap.from_list(
        name="Reds_alpha", colors=color_array
    )
    plt.register_cmap(cmap=map_object)
    map = plt.cm.get_cmap("Reds_alpha", levels + 1)
    map.set_under((1, 1, 1, 0))
    return map


def plot_cost(cost, algorithm, save_params, T):
    path = "data/" + algorithm + save_params + "/"
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    cost_val = np.array2string(
        np.array(sum(cost)), formatter={"float_kind": lambda x: "%.2f" % x}
    )
    dpi = 600
    fs = (3.5, 3.5)
    fig = plt.figure(figsize=fs, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(cost)
    fig.suptitle(str(algorithm) + " Total Cost = " + cost_val, fontsize=10)
    plt.ylabel("Cost H", fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.xlim((0, T))
    plt.ylim(bottom=0)
    plt.savefig(path + "Cost" + str(algorithm) + ".png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_figures(
    positions,
    mass,
    centroid,
    dcdt,
    dcdp,
    cost,
    t,
    dt,
    T,
    env,
    algorithm,
    save_params,
    phi,
):
    dim = 3
    if np.all(positions[0][-1]) == 0:
        dim = 2
    plot_cost(cost, algorithm, save_params, T)

    if dim == 3:
        save_trajectories(
            positions,
            mass,
            centroid,
            dcdt,
            dcdp,
            cost,
            t,
            dt,
            T,
            algorithm,
            env,
            save_params,
            phi,
        )
    else:
        raise ValueError("dim should be 3")


def plot_voronoi_regions(position, vor, ax=None):
    """
    Used for debugging
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    # Plot Voronoi Regions
    vertices = []
    for partition in vor:
        vertices.append(np.array(partition["vertices"]))
    for i, cube in enumerate(vertices):
        hull = ConvexHull(cube)
        for s in hull.simplices:
            tri = Poly3DCollection(
                [cube[s]],
            )
            tri.set_color("k")
            tri.set_alpha(0.03)
            tri.set_edgecolor("none")

            ax.add_collection3d(tri)