""" 
Python file containing utilities for basic Voronoi operations in 2D
"""

import numpy as np
from scipy.spatial.qhull import Delaunay

__author__ = "Brandon Bao"
__version__ = "1.0"
__maintainer__ = "Brandon Bao"
__email__ = "bjbao@ucsd.edu"


def random_points_stratified(chull, num_points, random_seed=None):
    """
    Generate random points in convex hull using stratified sampling. Delaunay tetras will be created from the vertices, and will be uniformly randomly sampled based on the weighted area of the tetra
    """
    hull_vertices = np.array(chull.points)
    dim = int(np.size(chull.points[0]))
    triangulation = Delaunay(hull_vertices)

    # Plot Simplices
    # def plot_tri(ax, points, tri):
    #     """
    #     https://stackoverflow.com/questions/20025784/how-to-visualize-2D-delaunay-triangulation-in-python
    #     """
    #     edges = collect_edges(tri)
    #     x = np.array([])
    #     y = np.array([])
    #     z = np.array([])
    #     for (i,j) in edges:
    #         x = np.append(x, [points[i, 0], points[j, 0], np.nan])
    #         y = np.append(y, [points[i, 1], points[j, 1], np.nan])
    #         z = np.append(z, [points[i, 2], points[j, 2], np.nan])
    #     ax.plot2D(x, y, z, color='k', lw='0.5')

    #     ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

    # def collect_edges(tri):
    #     edges = set()

    #     def sorted_tuple(a,b):
    #         return (a,b) if a < b else (b,a)
    #     # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    #     for (i0, i1, i2, i3) in tri.simplices:
    #         edges.add(sorted_tuple(i0,i1))
    #         edges.add(sorted_tuple(i0,i2))
    #         edges.add(sorted_tuple(i0,i3))
    #         edges.add(sorted_tuple(i1,i2))
    #         edges.add(sorted_tuple(i1,i3))
    #         edges.add(sorted_tuple(i2,i3))
    #     return edges

    # fig = plt.figure()
    # ax = plt.axes(projection='2D')
    # plot_tri(ax, hull_vertices, triangulation)
    # plt.show()

    simplices = triangulation.simplices
    points = triangulation.points
    vertices = points[simplices, :]
    # print(vertices)
    # plt.triplot(points[:,0], points[:,1], simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

    # shoelace method for area or area
    area = [abs(np.linalg.det(np.c_[v, np.ones(int(dim + 1))])) / 6 for v in vertices]
    totalv = sum(area)
    weights = area / totalv
    decimals = int(np.log10(num_points))
    # Want exactly num_points so need to minimally modify expected weights that are used to determine the number of points needed

    def round_series_retain_integer_sum(xs):
        """
        https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal
        """
        N = sum(xs)
        Rs = [int(x) for x in xs]
        K = N - sum(Rs)
        K = round(K)
        # assert K == int(K)
        fs = [x - int(x) for x in xs]
        indices = [
            i
            for order, (e, i) in enumerate(
                reversed(sorted((e, i) for i, e in enumerate(fs)))
            )
            if order < K
        ]
        ys = [R + 1 if i in indices else R for i, R in enumerate(Rs)]
        return ys

    counts_full = np.array(round_series_retain_integer_sum(num_points * weights))
    # counts_full = (num_points*rounded_weights).astype(int)
    # counts_full = (float(num_points)*rounded_weights)
    # print(math.fsum(counts_full))
    # counts_full = [int(round(x)) for x in counts_full]
    # print(math.fsum(counts_full))
    sampled_points = []
    num_sampled = []
    sampled_points_append = sampled_points.append
    num_sampled_append = num_sampled.append
    for idx, simplex in enumerate(triangulation.simplices):
        if counts_full[idx] == 0:
            num_sampled_append(0)
            continue
        # r, s, t, u = -np.log(np.random.uniform(size=counts_full[idx])), -np.log(np.random.uniform(size=counts_full[idx])), -np.log(
        #     np.random.uniform(size=counts_full[idx])), -np.log(np.random.uniform(size=counts_full[idx]))
        # var_sum = r + s + t + u

        test_samples = -np.log(np.random.uniform(size=(counts_full[idx], int(dim + 1))))
        test_sum = np.sum(test_samples, axis=1)
        test_points = test_samples / test_sum[:, np.newaxis]

        # xi_a, xi_b, xi_c, xi_d = (r/var_sum)[:, None], (s/var_sum)[:, None], (t/var_sum)[
        #     :, None], (u/var_sum)[:, None]  # vector of length num_points
        simplex_vertices = points[simplex, :]
        simplex_points = np.dot(test_points, simplex_vertices)

        # simplex_points = xi_a * simplex_vertices[0, :] + xi_b * simplex_vertices[1,
        #                                                                          :] + xi_c * simplex_vertices[2, :] + xi_d * simplex_vertices[3, :]

        # Plot points in individual simplex
        # fig = plt.figure()
        # ax = plt.axes(projection='2D')
        # ax.scatter(simplex_points[:,0], simplex_points[:,1], simplex_points[:,2], color='r',s=1)
        # plt.show()
        num_sampled_append(len(simplex_points))
        sampled_points_append(simplex_points)

    # used in chi_square_stratified function
    num_sampled = np.vstack(num_sampled)
    sampled_points = np.vstack((sampled_points))

    # fig = plt.figure()
    # ax = plt.axes(projection='2D')
    # ax.scatter(sampled_points[:,0], sampled_points[:,1], sampled_points[:,2], color='r',s=1)
    # plt.show()

    return sampled_points, num_sampled.flatten()


def phi(x, y, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Density functions for robotarium
    """
    if not (isinstance(sigma, float) or isinstance(sigma, int)):
        if sigma.size == 2:
            # Phi 6
            tau = 1500 / (2 * np.pi)  # 20 # L: tau
            r = 0.3  # 5
            s1 = sigma[0]  # 1
            s2 = sigma[1]
            offset = 1.6 / 3.0
            f1 = np.exp(
                -1
                * (
                    ((x - r * np.cos(t / tau) + offset) / s1) ** 2
                    + ((y - r * np.sin(t / tau)) / s1) ** 2
                )
            )
            f2 = np.exp(
                -1
                * (
                    ((x - r * np.cos(-t / tau) - offset) / s2) ** 2
                    + ((y - r * np.sin(-t / tau)) / s2) ** 2
                )
            )
            return f1 + f2
    # Phi 5 Density Function
    tau = 1000 / (2 * np.pi)  # 20 # L: tau
    r = 0.6  # 5, 0.6 for robotatrium
    s = sigma  # 1
    f = np.exp(
        -1
        * (((x - r * np.cos(t / tau)) / s) ** 2 + ((y - r * np.sin(t / tau)) / s) ** 2)
    )

    return f


def phidot(x, y, mu, sigma, mu_past, sigma_past, t, dt, pi=None, sig_dig=4):
    """
    Used to calculate dc/dt
    """

    if sigma.size == 2:
        # Phi 6
        tau = 1500 / (2 * np.pi)  # 20 # L: tau
        r = 0.3  # 5
        s1 = sigma[0]  # 1
        s2 = sigma[1]
        offset = 1.6 / 3.0
        f1 = np.exp(
            -1
            * (
                ((x - r * np.cos(t / tau) + offset) / s1) ** 2
                + ((y - r * np.sin(t / tau)) / s1) ** 2
            )
        )
        f2 = np.exp(
            -1
            * (
                ((x - r * np.cos(-t / tau) - offset) / s2) ** 2
                + ((y - r * np.sin(-t / tau)) / s2) ** 2
            )
        )
        f = (
            f1
            * (
                -2
                * r
                / (tau * s1**2)
                * (x - r * np.cos(t / tau) + offset)
                * -1.0
                * np.sin(t / tau)
                + 2
                * r
                / (tau * s1**2)
                * (y - r * np.sin(t / tau))
                * -1.0
                * np.cos(t / tau)
            )
        ) + (
            f2
            * (
                -2
                * r
                / (tau * s2**2)
                * (x - r * np.cos(t / tau) - offset)
                * -1.0
                * np.sin(-t / tau)
                + 2
                * r
                / (tau * s2**2)
                * (y - r * np.sin(t / tau))
                * -1.0
                * np.cos(-t / tau)
            )
        )
        return f
    # Phi 5 Density Function
    tau = 1000 / (2 * np.pi)  # L: tau
    r = 0.6  # 5, 0.6 for robotatrium
    s = sigma
    f = phi(x, y, mu, sigma, t) * (
        -2 * r / (tau * s**2) * (x - r * np.cos(t / tau)) * np.sin(t / tau)
        + 2 * r / (tau * s**2) * (y - r * np.sin(t / tau)) * np.cos(t / tau)
    )
    return f


def weighted_phi(x, y, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Weighted Gaussian used to find the centroid
    """
    # return np.around(
    #     np.transpose(np.multiply(np.array([x, y, z]) , phi(x,y,z,mu,sigma,sig_dig=4))), sig_dig
    # )
    return np.multiply(np.array([x, y]), phi(x, y, mu, sigma, t, sig_dig=4)).T


def dcdt_function(x, y, mu, sigma, t, dt, c, pi=None, sig_dig=4):
    """
    Calclate dc/dt for Time Varying Densities
    """
    # return np.around(np.multiply((np.array([x, y, z]).T - c).T, phidot(x,y,np.array([mu[0]]),np.array([sigma[0]]),np.array([mu[1]]),np.array([sigma[1]]),dt,sig_dig=sig_dig)), sig_dig).T
    return np.multiply(
        (np.array([x, y]).T - c).T,
        phidot(
            x,
            y,
            np.array([mu[0]]),
            sigma,
            np.array([mu[1]]),
            sigma,  # TODO keep sigma as size 2, so that the phidot can be appoximated
            t,
            dt,
            sig_dig=sig_dig,
        ),
    ).T


def cost_function(x, y, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Calculate locational cost function H to compare different coverage control algorithms
    """
    try:
        qq = np.sqrt((x - pi[0]) ** 2 + (y - pi[1]) ** 2)
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
    return qq * phi(x, y, mu, sigma, t, sig_dig=sig_dig)


def monte_carlo_integrate(
    points_all, area, function, mu, sigma, t, dt, c, pos, num_points=None
):
    """
    Used by lloyd_single_step to find mass and centroid using sampled points. Takes in sampled points, a function such as phi/weighted_phi, and the parameters of the mu location. The area normalizes the calculation
    """
    integral = []
    # dim = np.shape(points_all)[2]
    # if dim == 2: # add z points so that function eval works
    #     points_all = [np.c_[points, np.zeros(shape=(len(points),))] for points in points_all]
    # Go one voronoi partition at a time
    for idx, points in enumerate(points_all):
        # Evaluate function on points
        function_result = function(
            points[:, 0],
            points[:, 1],
            mu,
            sigma,
            t,
            dt,
            c[idx],
            pos[idx],
            sig_dig=4,
        )
        # Plot Monte Carlo Function Evaluation Result
        # visualize_monte_carlo(points,sigma,function_result)
        integral_result = np.multiply(
            area[idx] / num_points, np.sum(function_result, axis=0)
        )
        integral.append(integral_result)

    return np.array(integral)  # in same order as inputted through vor


def massless_centroid(pos, mu_pos, vor, partitions):
    """
    Find centroid when mass is zero

    The centroid must remain in the Voronoi cell and we want it to be as close as possible to the red position
    We take a ray which is from the agent position to the mu position and test if it intersects a face
    Then we find the point where it intersects.
    TODO: Make sure it works in edge cases
    """
    # print("Line 679 Position after Massless: ", pos)
    dim = 2
    # if pos[-1] == 0.:
    #     dim = 2
    vertices = vor["vertices"]
    if check_membership(vertices, mu_pos):
        centroid = mu_pos
        return centroid
    norm_dist = np.linalg.norm(mu_pos - pos)
    rayDirection = (mu_pos - pos) / norm_dist
    rayPoint = pos
    faces = vor["faces"]
    centroid = [0, 0, 0]
    centroids = []
    prevdist = norm_dist
    for face in faces:
        verticesIdx = face["vertices"]
        A = vertices[verticesIdx[0]]
        B = vertices[verticesIdx[1]]
        # C = vertices[verticesIdx[2]]
        # planeNormal = np.cross(B-A,C-A)
        # t = (planeNormal[0]*(A[0]-pos[0]) + planeNormal[1]*(A[1]-pos[1]) + planeNormal[2]*(A[2]-pos[2]))/(np.dot(A,A))
        # proj_pi = np.array([pos[0]+t*planeNormal[0],pos[1]+t*planeNormal[1],pos[2]+t*planeNormal[2]])
        # orientation = np.dot(proj_pi-pos,planeNormal)
        # if orientation < 0:
        #     planeNormal *= -1
        # planePoint = A
        if dim == 2:
            intersect = point_line_intersect(A[0:2], B[0:2], mu_pos, pos)
            # print(intersect)
            if len(intersect) != 0:
                # intersect =  np.r_[intersect, 0.]
                dist = np.linalg.norm(mu_pos - intersect)
            else:
                dist = prevdist
        # else:
        #     intersect = point_plane_intersect(planeNormal,planePoint,rayDirection,rayPoint)
        #     if intersect.size:
        #         dist = np.linalg.norm(mu_pos-intersect)
        #         # print(intersect, intersect.size)
        #     else:
        #         print("Edge case, unexpected Voronoi cell configuration")
        # check if there exists a valid intersection
        centroids.append(intersect)
        # print(dist,prevdist)
        if dist < prevdist:
            centroid = intersect
            prevdist = dist

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='2D')
    # ax.scatter(centroid[0], centroid[1] , centroid[2],  color='green')
    # ax.scatter(pos[0], pos[1] , pos[2],  color='blue')
    # ax.scatter(mu_pos[0], mu_pos[1] , mu_pos[2],  color='red')

    # generate_plots.plot_voronoi_regions(pos,partitions,ax=ax)
    # plt.show()
    # print(centroid)
    return centroid
