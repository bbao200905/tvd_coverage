import csv
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pyvoro
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from scipy.spatial import ConvexHull
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

@lru_cache(1)
def colormap(levels):
    """
    Helper function for contour maps
    """
    ncolors = 256
    color_array = plt.get_cmap("afmhot_r")(range(ncolors))
    map_object = LinearSegmentedColormap.from_list(
        name="Reds_alpha", colors=color_array[: -int(ncolors / 4)]
    )
    plt.register_cmap(cmap=map_object)
    map = plt.cm.get_cmap("Reds_alpha", levels + 1)
    map.set_under((1, 1, 1, 0))
    return map


def plot_voronoi_regions(
    position=None, ax=None, zorder=20, vor=None, linewidth=1, pos=None
):
    """
    Used for debugging
    """

    vertices = []
    if vor is None and position is not None:
        vor = pyvoro.compute_2d_voronoi(position, [[-10, 10], [-10, 10]], dispersion=4)
    elif vor is None and pos is not None:
        try:
            vor = pyvoro.compute_2d_voronoi(pos, [[-1.6, 1.6], [-1, 1]], dispersion=10)
        except:
            print("Agent left, skipping trial")
            return
    lines = []
    for partition in vor:
        vertices.append(np.array(partition["vertices"]))
    lines = []
    for cube in vertices:
        hull = ConvexHull(cube)
        for s in hull.simplices:
            a = [np.append(lists, 0) for lists in cube[s]]
            lines.append(a)
    return lines


def plot_cost(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    cost = []
    time = []
    for row in data:
        time.append(row[0])
        cost.append(row[1])
    dpi = 600
    fs = (3.5, 3.5)
    cost_val = sum(cost)
    fig = plt.figure(figsize=fs, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(cost)
    fig.suptitle("TVD-SP-hybrid"+" Total Cost "+f"{cost_val:.2f}", fontsize=10)
    plt.ylabel("Cost H", fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.xlim((0, len(time)))
    plt.ylim(bottom=0)
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image  
    axins = inset_axes(ax, width=2, height=1.5, loc='lower center')
    # axins = zoomed_inset_axes(ax, zoom=2, loc='lower center')
    axins.plot(cost[:200])
    # fix the number of ticks on the inset axes
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    axins.tick_params(labelleft=False, labelbottom=False)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    plt.savefig('Robotarium TVD-SP-hybrid'+ "Cost" +
                ".png", bbox_inches='tight', dpi=600)
    exit()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, cost, label="Cost")
    ax.legend(loc="best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost H")
    config = f"Algorithm: {CoverageControl.algorithm} Total Cost: {sum(cost):.2f}, {n_agents} agents, {num_points:.0f} samples, {len(time)} iterations"
    ax.set_title("TVD-SP-Cost ")
    plt.savefig(
        "pycoverage_robotarium\\CostTVD-SP-hybrid.png"
    )

def main():
    filename = 'pycoverage_robotarium\\TVD-SP-hybrid_cost.txt'
    plot_cost(filename)

if __name__ == '__main__':
    main()