"""
TVD-SP paper robotarium sim
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
from coveragecontrol_2d import TVD, TVD_K, TVD_SP, Agent, Lloyd
from generate_plots import colormap, plot_voronoi_regions
from utils_2d import DoubleEllipse, Ellipse

import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
from rps.utilities.graph import *
from rps.utilities.misc import *
from rps.utilities.transformations import *


def getAlgorithm(
    algorithm_num, params, eps, deta, trajectory, Agents, k, algorithm_list
):
    algorithms = dict(zip(range(1, len(algorithm_list) + 1), algorithm_list))
    algorithm = algorithms[algorithm_num]

    if algorithm_num == 4:
        CoverageControl = Lloyd(params, trajectory, Agents)
    elif algorithm_num == 5:
        CoverageControl = TVD(params, trajectory, Agents)
    elif algorithm_num == 6:
        CoverageControl = TVD_K(params, k, trajectory, Agents)
    elif algorithm_num == 2:
        CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    elif algorithm_num == 1:
        CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    elif algorithm_num == 3:
        CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    else:
        raise ValueError("Pick num 1-6 according to the algorithm_list")

    return CoverageControl


def run_sim(num_points=1e4, n_agents=5, algorithm=4, T=1000, tau=2000):
    """
    Simulation Settings:

    position: Initial agent position. Supports many agents.
    env: Environment bounds, limited to rectangular prism shape.
    mu_a: Initial start position of target agent.
    mu_b: Final position in simulation. The generator will make linear path between the points.
    sigma_a, sigma_b: Covariance at start and end. Generator linearly interpolates value at each frame.
    num_frames: Number of frames for sim. Will affect red agent speed.
    """

    # Parameters for coverage control
    dim = 2
    env = [
        [-1.6, 1.6],
        [-1.0, 1.0],
    ]  # meters, robotarium is 3.2 by 2 m

    n_agents = n_agents
    seed = 0

    np.random.seed(seed)
    # Initial Conditions to Avoid Barrier Use in the Beginning. (x,y pos for each)
    if n_agents == 5:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28], [0.8, -0.3, -0.75, 0.1, 0.34], [0, 0, 0, 0, 0]]
        ).T
    elif n_agents == 6:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28, 0.85], [0.8, -0.3, -0.75, 0.1, 0.34], [0, 0, 0, 0, 0, 0]]
        )
    elif n_agents == 7:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28, 0.85, -1], [0.8, -0.3, -0.75, 0.1, 0.34, -0.7, 0.7], [0, 0, 0, 0, 0, 0, 0]]
        ).T
    elif n_agents == 8:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28, 0.85, -1, -0.65], [0.8, -0.3, -0.75, 0.1, 0.34, -0.7, 0.7, -0.2], [0, 0, 0, 0, 0, 0, 0, 0]]
        ).T
    elif n_agents == 9:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28, 0.85, -1, -0.65, -0.8], [0.8, -0.3, -0.75, 0.1, 0.34, -0.7, 0.7, -0.2, 0.2], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).T
    elif n_agents == 10:
        agent_pos_init = np.array(
            [[1, 0.5, -0.5, -0.25, 0.28, 0.85, -1, -0.65,-0.8, 1], [0.8, -0.3, -0.75, 0.1, 0.34, -0.7, 0.7, -0.2, 0.2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).T
    else:
        raise NotImplementedError
    agent_pos = agent_pos_init[:, :2]
    Agents = []
    for n in range(n_agents):
        Agents.append(Agent(agent_pos[n, :]))
    Agents = np.array(Agents)
    trim_vel = True

    # Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
    # Total time T and time step dt
    T = T
    dt = 1  # needs to be 1 for sim. 1 will be 0.033 seconds
    t = 0
    # 0.4 m/s max speed
    num_frames = int(round(T / dt))
    # Proportional term to drive agents to centroid
    kappa = 1

    # Coverage control sim config
    algorithm_list = [
        "TVD-SP-hybrid",
        "TVD-SP",
        "TVD-SP-delayed",
        "Lloyd",
        "TVD-C",
        "TVD-K",
    ]
    algorithm = algorithm
    max_vel = 0.15
    max_vel_alg = 0.15
    mass_eps = 0# 3e-3  # condition of no mass in a Voronoi cell, 3 sigma
    num_points = num_points  # Monte Carlo Sampling Points
    # TVD Parameters
    # Number of hops for distributed TVD-K algorithm
    k = 1

    # Singular Perturbation time scaling parameter, must be small ie: (1e-2 to 1e-6). For larger eps, try TVD_SSP. deta> 1/L where L is the largest
    # |eigval| of the Hessian of the A matrix. A = np.eye(np.size(blue_pos)) -dcdp.
    eps = 1e-1
    deta = 1e-3

    # Trajectory
    mu_a = [10.0, 10.0]
    mu_b = [0.0, 0.0]


    sigma_a = [0.75, 0.75]
    sigma_b = [0.75, 0.75]
    radius = 0.6
    center = np.array([0, 0])
    rotations = T / tau  # L: tau
    a = np.array([1, 0])
    b = np.array([0, 1])
    c = 0.6
    d = 0.6
    trajectory = Ellipse(
        mu_a,
        mu_b,
        sigma_a,
        sigma_b,
        num_frames,
        dim,
        radius,
        center,
        a,
        b,
        c,
        d,
        rotations,
    )
    sigma_a = np.array([[0.3, 0.3], [0.6, 0.6]])
    sigma_b = np.array([[0.3, 0.3], [0.6, 0.6]])
    radius = np.array([0.3, 0.3])
    center = np.array([[-1.6/3., 0],[1.6/3,0]])
    rotations = T / tau  # L: tau
    a = np.array([[1, 0], [1, 0]])
    b = np.array([[0, 1], [0, 1]])
    c = np.array([0.3, 0.3])
    d = np.array([0.3, 0.3])
    trajectory = DoubleEllipse(
        mu_a,
        mu_b,
        sigma_a,
        sigma_b,
        num_frames,
        dim,
        radius,
        center,
        a,
        b,
        c,
        d,
        rotations,
    )

    params = {
        "env": env,
        "trim_vel": trim_vel,
        "seed": seed,
        "T": T,
        "dt": dt,
        "t": t,
        "num_frames": num_frames,
        "kappa": kappa,
        "max_vel": max_vel_alg,
        "mass_eps": mass_eps,
        "num_points": num_points,
        "tau": tau,
    }
    save_params = (
        "agents"
        + str(n_agents)
        + "eps"
        + str(eps)
        + "deta"
        + str(deta)
        + "max_u"
        + str(max_vel)
        + "T"
        + str(T)
        + "dt"
        + str(dt)
        + "samples"
        + str(num_points)
    )
    if dim == 2:
        save_params += "dim" + str(dim)

    # robotarium stuff
    # For computational/memory reasons, initialize the velocity vector
    dxi = np.zeros((2, n_agents))

    CoverageControl = getAlgorithm(
        algorithm, params, eps, deta, trajectory, Agents, k, algorithm_list
    )

    # Instantiate the Robotarium object with these parameters
    r = robotarium.Robotarium(
        number_of_robots=n_agents,
        show_figure=True,
        initial_conditions=agent_pos_init.T,
        sim_in_real_time=False,
    )  # pos transposed compared to coverage control

    # Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
    # Single-integrator -> unicycle dynamics mapping
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion(
        angular_velocity_limit=np.pi
    )
    # Single-integrator barrier certificates
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(
        safety_radius=0.15
    )
    # Single-integrator position controller
    coverage_controller = create_si_position_controller(velocity_magnitude_limit=max_vel)

    # define x initially
    x = r.get_poses()
    xi = uni_to_si_states(x)
    # Algorithm initially
    x_coverage = copy.deepcopy(x[:2, :])

    velocity = CoverageControl.calculateVelocities(x_coverage)
    waypoints = xi + velocity.T[:2, :]

    # Plotting Parameters
    CM = np.random.rand(n_agents, 3)  # Random Colors
    # TODO add cmap for density, plot 4 circles
    cmap = colormap(5)(range(6))
    goal_marker_size_m = 0.05
    robot_marker_size_m = 0.15
    center_marker_size_m = 0.03
    max_sigma1 = 1.4 * sigma_a[0][0]
    max_sigma2 = 1.4 * sigma_a[1][0]
    srange = np.linspace(0.25, 1, 4,True)
    sigma_marker_size_m1 = np.flip(max_sigma1*srange)
    sigma_marker_size_m2 = np.flip(max_sigma2*srange)
    marker_size_goal = determine_marker_size(r, goal_marker_size_m)
    marker_size_robot = determine_marker_size(r, robot_marker_size_m)
    marker_size_center = determine_marker_size(r, center_marker_size_m)
    marker_size_sigma1 = [determine_marker_size(r, size) for size in sigma_marker_size_m1]
    marker_size_sigma2 = [determine_marker_size(r, size) for size in sigma_marker_size_m2]
    font_size_m = 0.1
    font_size = determine_font_size(r, font_size_m)
    line_width = 4

    # Create Goal Point Markers
    goal_markers = [
        r.axes.scatter(
            waypoints[0, ii],
            waypoints[1, ii],
            s=marker_size_goal,
            marker=".",
            facecolors="none",
            edgecolors=CM[ii, :],
            linewidth=line_width,
            zorder=1,
        )
        for ii in range(waypoints.shape[1])
    ]
    robot_markers = [
        r.axes.scatter(
            x[0, ii],
            x[1, ii],
            s=marker_size_robot,
            marker="o",
            facecolors="none",
            edgecolors=CM[ii, :],
            linewidth=line_width,
        )
        for ii in range(n_agents)
    ]
    robot_text = np.empty((n_agents, 0))
    for jj in range(1, n_agents + 1):
        robot_text = np.append(robot_text, "{0}".format(jj))
    robot_labels = [
        r.axes.text(
            x[0, kk],
            x[1, kk] + 0.15,
            robot_text[kk],
            fontsize=font_size,
            color=CM[kk, :],
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            zorder=1,
        )
        for kk in range(n_agents)
    ]
    lines = plot_voronoi_regions(pos=x_coverage.T, ax=r.axes, linewidth=line_width)
    voronoi = [
        r.axes.plot(
            [line[0][0], line[1][0]],
            [line[0][1], line[1][1]],
            "k-",
            zorder=5,
            linewidth=line_width,
        )
        for line in lines
    ]
    mu, sigma = trajectory.getValuesAtT(1)

    center1 = r.axes.scatter(
        mu[0][0],
        mu[0][1],
        s=marker_size_center,
        marker=".",
        facecolors="none",
        edgecolors=cmap[-1],
        linewidth=line_width,
        zorder=0
    )
    circle1 = [r.axes.scatter(
        mu[0][0],
        mu[0][1],
        s=size,
        marker="o",
        facecolors="none",
        edgecolors=cmap[i+1],
        linewidth=line_width,
        zorder=0
    ) for i, size in enumerate(marker_size_sigma1)]
    
    center2 = r.axes.scatter(
        mu[0][0],
        mu[0][1],
        s=marker_size_center,
        marker=".",
        facecolors="none",
        edgecolors=cmap[-1],
        linewidth=line_width,
        zorder=0
    )
    circle2 = [r.axes.scatter(
        mu[0][0],
        mu[0][1],
        s=size,
        marker="o",
        facecolors="none",
        edgecolors=cmap[i+1],
        linewidth=line_width,
        zorder=0
    ) for i, size in enumerate(marker_size_sigma2)]

    r.step()

    for t in range(2, num_frames):
        # Get the most recent pose information from the Robotarium. The time delay is
        # approximately 0.033s
        x = r.get_poses()
        # print('x: ',x)
        xi = uni_to_si_states(x)

        # Algorithm
        x_coverage = copy.deepcopy(x[:2, :])

        velocity = CoverageControl.calculateVelocities(x_coverage)
        waypoints = xi + velocity.T[:2, :]

        # Update Plot
        # Update Robot Marker Plotted Visualization
        for i in range(x.shape[1]):
            robot_markers[i].set_offsets(x[:2, i].T)
            # This updates the marker sizes if the figure window size is changed.
            robot_labels[i].set_position([xi[0, i], xi[1, i] + 0.15])
            robot_labels[i].set_fontsize(determine_font_size(r, font_size_m))

        for j in range(waypoints.shape[1]):
            goal_markers[j].set_offsets(waypoints[:2, j].T)

        lines = plot_voronoi_regions(
            pos=x_coverage.T, ax=r.axes, zorder=-1, linewidth=line_width
        )
        try:
            for k in range(len(lines)):
                line = lines[k]
                try:
                    voronoi[k][0].set_data(
                        [line[0][0], line[1][0]], [line[0][1], line[1][1]]
                    )  # TODO set while dcdp does't work
                except:
                    pass
        except:
            pass
        mu, sigma = trajectory.getValuesAtT(t)
        center1.set_offsets(mu[0].T[0].T)
        for j in range(len(marker_size_sigma1)):
            circle1[j].set_offsets(mu[0].T[0].T)
        center2.set_offsets(mu[0].T[1].T)
        for j in range(len(marker_size_sigma1)):
            circle2[j].set_offsets(mu[0].T[1].T)

        dxi = coverage_controller(xi, waypoints[:2][:])

        # Keep single integrator control vectors under specified magnitude
        # Threshold control inputs
        # norms = np.linalg.norm(dxi, 2, 0)
        # idxs_to_normalize = norms > max_vel
        # dxi[:, idxs_to_normalize] *= max_vel / norms[idxs_to_normalize]

        # # Use barriers and convert single-integrator to unicycle commands
        # # dxi = si_barrier_cert(dxi, x[:2, :])
        dxi = si_barrier_cert(dxi, xi)
        dxu = si_to_uni_dyn(dxi, x)
        # Set the velocities of agents 1,...,n_agents to dxu
        r.set_velocities(np.arange(n_agents), dxu)
        # Iterate the simulation
        r.step()

    # Call at end of script to print debug information and for your script to run on the Robotarium server properly
    cost = CoverageControl.cost_vec
    time = CoverageControl.t_vec
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(time, cost, label='Cost')
    ax.legend(loc='best')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cost')
    config = f"Algorithm: {CoverageControl.algorithm} Total Cost: {sum(cost):.2f}, {n_agents} agents, {num_points:.0f} samples, {len(time)} iterations"
    ax.set_title(config)
    plt.savefig(f"Algorithm{CoverageControl.algorithm}_TotalCost{sum(cost):.2f}_agents{n_agents}_samples{num_points:.0f}_iterations{len(time)}.png")
    print(config)
    np.savetxt(f"{CoverageControl.algorithm}_cost.txt", np.array(list(zip(time, cost))), delimiter=',', header="time, cost", comments="")

    r.call_at_scripts_end()


def main():
    num_points = 1e4
    n_agents_opt = [3, 4, 5, 10]
    n_agents_opt = [8]
    """
    algorithm_list = [
        "TVD-SP-hybrid",
        "TVD-SP",
        "TVD-SP-delayed",
        "Lloyd",
        "TVD-C",
        "TVD-K",
    ]
    """
    algorithm_opt = [4, 2, 1, 3, 5, 6]
    T = 2500
    tau = 1500
    for algorithm in algorithm_opt:
        for n_agents in n_agents_opt:
            run_sim(num_points, n_agents, algorithm, T=T, tau=tau)


if __name__ == "__main__":
    main()
