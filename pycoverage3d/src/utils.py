import cProfile
import functools
import os
import pickle
import pstats

import numpy as np


def profile(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            retval = func(*args, **kwargs)
        finally:
            filename = "profile.prof"  # You can change this if needed
            profiler.dump_stats(filename)
            profiler.disable()
            with open("profile.txt", "w") as profile_file:
                sortby = "cumulative"
                stats = pstats.Stats(profiler, stream=profile_file).sort_stats(sortby)
                stats.print_stats()
        return retval

    return inner


def save_data(dir, seed, model, algorithm, params, data):
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = dir + algorithm + params + "phi_" + str(model) + "_seed" + str(seed) + ".pkl"
    with open(name, "wb") as f:
        pickle.dump(data, f)


def load_data(dir, seed, model, algorithm, params):
    name = dir + algorithm + params + "phi_" + str(model) + "_seed" + str(seed) + ".pkl"
    with open(name, "rb") as f:
        data = pickle.load(f)

    return data


class Trajectory:
    """
    Create target agent trajectories. One agent has a position mu and covariance sigma, start and end points, and the trajectory runs for a num_frames
    """

    def __init__(self, mu_a, mu_b, sigma_a, sigma_b, num_frames, dim):
        # self.trajectory = type # target trajectory
        # if self.trajectory != "ellipse" and self.trajectory != "diagonal":
        #     raise ValueError('trajectory type must be "ellipse" or "diagonal"')
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.num_frames = num_frames
        self.x = None
        self.y = None
        self.z = None
        self.sigma_xt = None
        self.sigma_yt = None
        self.sigma_zt = None
        self.dim = dim
        self.sigmaGenerator()

    def sigmaGenerator(self):
        self.sigma_xt = []
        self.sigma_yt = []
        self.sigma_zt = []
        for t in range(self.num_frames):
            self.sigma_xt.append(
                self.sigma_a[0] * (1 - float(t) / self.num_frames)
                + self.sigma_b[0] * (float(t) / self.num_frames)
            )

            self.sigma_yt.append(
                self.sigma_a[1] * (1 - float(t) / self.num_frames)
                + self.sigma_b[1] * (float(t) / self.num_frames)
            )

            self.sigma_zt.append(
                self.sigma_a[2] * (1 - float(t) / self.num_frames)
                + self.sigma_b[2] * (float(t) / self.num_frames)
            )

    def getValuesAtT(self, i):
        if i < 1:
            mu = np.array(
                [
                    [
                        self.x[0],
                        self.y[0],
                        self.z[0],
                    ],
                    [
                        self.x[0],
                        self.y[0],
                        self.z[0],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        self.sigma_xt[0],
                        self.sigma_yt[0],
                        self.sigma_zt[0],
                    ],
                    [
                        self.sigma_xt[0],
                        self.sigma_yt[0],
                        self.sigma_zt[0],
                    ],
                ]
            )
        else:
            mu = np.array(
                [
                    [
                        self.x[i],
                        self.y[i],
                        self.z[i],
                    ],
                    [
                        self.x[i - 1],
                        self.y[i - 1],
                        self.z[i - 1],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        self.sigma_xt[i],
                        self.sigma_yt[i],
                        self.sigma_zt[i],
                    ],
                    [
                        self.sigma_xt[i - 1],
                        self.sigma_yt[i - 1],
                        self.sigma_zt[i - 1],
                    ],
                ]
            )
        return mu, sigma


class Ellipse(Trajectory):
    def __init__(
        self,
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
    ):
        super().__init__(mu_a, mu_b, sigma_a, sigma_b, num_frames, dim)
        self.radius = radius
        self.center = center
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.rotations = rotations
        self.calculateEllipsePoints()

    def calculateEllipsePoints(self):
        """
        Will produce points for an circle centedensity at center, with orthonormal vectors a and b defining basis coordinates for the ellipse. c and d scale the respective axes.
        """
        t = np.linspace(0, 2 * np.pi * self.rotations, self.num_frames, endpoint=False)
        self.x = list(
            self.center[0]
            + self.c * np.cos(t) * self.a[0]
            + self.d * np.sin(t) * self.b[0]
        )
        self.y = list(
            self.center[1]
            + self.c * np.cos(t) * self.a[1]
            + self.d * np.sin(t) * self.b[1]
        )
        self.z = list(
            self.center[2]
            + self.radius
            * (np.cos(t) * self.a[2] / self.c + np.sin(t) * self.b[2] / self.d)
        )
        if self.dim == 2:
            self.z = list(0 * t)


def trajectory_generator(mu_a, mu_b, sigma_a, sigma_b, num_frames, type):
    """
    Create density trajectories. Currently supports diagonal and circle
    """

    def ellipse_points(radius, num_frames, center, a, b, c, d, rotations):
        """
        Will produce points for an circle centedensity at center, with orthonormal vectors a and b defining basis coordinates for the ellipse. c and d scale the respective axes.
        """
        t = np.linspace(0, 2 * np.pi * rotations, num_frames, endpoint=False)
        x = center[0] + c * np.cos(t) * a[0] + d * np.sin(t) * b[0]
        y = center[1] + c * np.cos(t) * a[1] + d * np.sin(t) * b[1]
        z = center[2] + radius * (np.cos(t) * a[2] / c + np.sin(t) * b[2] / d)
        # z = np.linspace(1, 9, num_points, endpoint=False)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(x, y, z, 'gray')
        # ax.set_xlim3d(0,10)
        # ax.set_ylim3d(0,10)
        # ax.set_zlim3d(0,10)
        # plt.show()

        return x, y, z

    if type == "diagonal":
        # mu_a[0] * (float)((1 - ((float)(t) / num_frames))) + 0 * float((float)(float(t) / 10.0))
        def mu_xt_generator(t):
            return mu_a[0] * (1 - float(t) / num_frames) + mu_b[0] * (
                float(t) / num_frames
            )

        def mu_yt_generator(t):
            return mu_a[1] * (1 - float(t) / num_frames) + mu_b[1] * (
                float(t) / num_frames
            )

        def mu_zt_generator(t):
            return mu_a[2] * (1 - float(t) / num_frames) + mu_b[2] * (
                float(t) / num_frames
            )

        def sigma_xt_generator(t):
            return sigma_a[0] * (1 - float(t) / num_frames) + sigma_b[0] * (
                float(t) / num_frames
            )

        def sigma_yt_generator(t):
            return sigma_a[1] * (1 - float(t) / num_frames) + sigma_b[1] * (
                float(t) / num_frames
            )

        def sigma_zt_generator(t):
            return sigma_a[2] * (1 - float(t) / num_frames) + sigma_b[2] * (
                float(t) / num_frames
            )

        mu_xt = [mu_xt_generator(k) for k in range(num_frames)]
        mu_yt = [mu_yt_generator(k) for k in range(num_frames)]
        mu_zt = [mu_zt_generator(k) for k in range(num_frames)]
        sigma_xt = [sigma_xt_generator(k) for k in range(num_frames)]
        sigma_yt = [sigma_yt_generator(k) for k in range(num_frames)]
        sigma_zt = [sigma_zt_generator(k) for k in range(num_frames)]
    elif type == "ellipse":
        radius = 5
        center = np.array([0, 0, 0])
        rotations = 10

        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = 1
        d = 1

        [mu_xt_generator, mu_yt_generator, mu_zt_generator] = ellipse_points(
            radius, num_frames, center, a, b, c, d, rotations
        )

        def sigma_xt_generator(t):
            return sigma_a[0] * (1 - float(t) / num_frames) + sigma_b[0] * (
                float(t) / num_frames
            )

        def sigma_yt_generator(t):
            return sigma_a[1] * (1 - float(t) / num_frames) + sigma_b[1] * (
                float(t) / num_frames
            )

        def sigma_zt_generator(t):
            return sigma_a[2] * (1 - float(t) / num_frames) + sigma_b[2] * (
                float(t) / num_frames
            )

        mu_xt = list(mu_xt_generator)
        mu_yt = list(mu_yt_generator)
        mu_zt = list(mu_zt_generator)
        sigma_xt = [sigma_xt_generator(k) for k in range(num_frames)]
        sigma_yt = [sigma_yt_generator(k) for k in range(num_frames)]
        sigma_zt = [sigma_zt_generator(k) for k in range(num_frames)]
    else:
        raise ValueError("not a valid trajectory type")

    density_params = {
        "mu_xt": mu_xt,
        "mu_yt": mu_yt,
        "mu_zt": mu_zt,
        "sigma_xt": sigma_xt,
        "sigma_yt": sigma_yt,
        "sigma_zt": sigma_zt,
    }
    return density_params
