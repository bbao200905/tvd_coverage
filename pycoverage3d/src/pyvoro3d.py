""" Python file containing utilities for basic Voronoi operations in 3D.

"""

import scipy.stats
import scipy.optimize
import scipy.spatial
import numpy as np
from scipy.spatial.qhull import Delaunay

__author__ = "Brandon Bao, Simon Hu"
__version__ = "1.0"
__maintainer__ = "Brandon Bao"
__email__ = "bjbao@ucsd.edu"



def check_membership(hull_points, point):
    """Determines if ``point`` is contained in the convex polygon determined by ``hull_points``.

    Membership is checked by determining whether the point can be written as a convex combination of the points that define the convex hull. This is equivalent to solving the linear problem...finish the problem later.

    Parameters
    ----------
    hull_points : array-like
        Array-like containing n-dimensional coordinates of the points that define the convex hull.
    point : array-like
        Array-like containing n-dimensional coordinates of the point for which membership is checked.

    Returns
    -------
    success : bool
        Boolean determining whether ``point`` is contained in the convex polygon determined by ``hull_points``.

    Notes
    -----
    The input to this function is not restricted to points in 3D. In fact, this algorithm generalizes to n-dimensions.

    Note that by the definition of a convex hull, it is not neccessary that hull_points define a convex hull. The algorithm also works on any set of points, since a subset of those points will be guaranteed to define a convex hull.

    Examples
    --------
    >>> # Test if point (0.0, 0.0, 1.01) is in the unit cube. Expecting False.
    >>> unit_cube = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]
    >>> point = [0.0, 0.0, 1.01]
    >>> pyvoro.check_membership(unit_cube, point)
    False

    >>> # Test if point (0.0, 0.0, 1.0) is in the unit cube. Expecting True.
    >>> point = [0.0, 0.0, 1.0]
    >>> pyvoro.check_membership(unit_cube, point)
    True

    See Also
    --------

    """
    # Convert point-to-check to numpy.ndarray for consistency.
    try:
        point = np.array(point)
    except TypeError:
        print(
            "Input must be an array-like object like a list or numpy ndarray. See documentation for example usuage"
        )

    try:
        hull_points = np.array(hull_points)
    except TypeError:
        print(
            "Input must be an array-like object like a list or numpy ndarray. See documentation for example usage."
        )

    # Set up the linear program.
    num_points = len(hull_points)
    num_dim = len(point)
    c = np.zeros(num_points)
    A = np.vstack((hull_points.T, np.ones((1, num_points))))
    b = np.hstack((point, np.ones(1)))

    # Solve the linear program to determine membership.
    lp = scipy.optimize.linprog(c, A_eq=A, b_eq=b)
    success = lp.success

    return success


def random_points_stratified(chull, num_points, random_seed=None):
    """
    Generate random points in convex hull using stratified sampling. Delaunay tetras will be created from the vertices, and will be uniformly randomly sampled based on the weighted area of the tetra
    """
    hull_vertices = np.array(chull.points)
    dim = int(np.size(chull.points[0]))
    triangulation = Delaunay(hull_vertices)
    simplices = triangulation.simplices
    points = triangulation.points
    vertices = points[simplices, :]

    # shoelace method for volume or area
    volume = [abs(np.linalg.det(np.c_[v, np.ones(int(dim + 1))])) / 6 for v in vertices]
    totalv = sum(volume)
    weights = volume / totalv

    def round_series_retain_integer_sum(xs):
        """
        https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal
        """
        N = sum(xs)
        Rs = [int(x) for x in xs]
        K = N - sum(Rs)
        K = round(K)
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
    sampled_points = []
    num_sampled = []
    sampled_points_append = sampled_points.append
    num_sampled_append = num_sampled.append
    for idx, simplex in enumerate(triangulation.simplices):
        if counts_full[idx] == 0:
            num_sampled_append(0)
            continue
        test_samples = -np.log(np.random.uniform(size=(counts_full[idx], int(dim + 1))))
        test_sum = np.sum(test_samples, axis=1)
        test_points = test_samples / test_sum[:, np.newaxis]
        simplex_vertices = points[simplex, :]
        simplex_points = np.dot(test_points, simplex_vertices)
        num_sampled_append(len(simplex_points))
        sampled_points_append(simplex_points)

    # used in chi_square_stratified function
    num_sampled = np.vstack(num_sampled)
    sampled_points = np.vstack((sampled_points))

    return sampled_points, num_sampled.flatten()


def point_plane_intersect(planeNormal, planePoint, rayDirection, rayPoint):
    """
    Returns a point where a ray intersects a plane.  Returns error if there is no inteersection, or if the line is within the plane.

    Plane: planeNormal, planePoint
    Ray: rayDirection, rayPoint
    """
    epsilon = 1e-6

    ndotu = planeNormal.dot(rayDirection)

    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")
        return []

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint

    return Psi


def point_line_intersect(vertexA, vertexB, target, pos):
    """
    Return a point where a ray intersects a line. Projects down to 2D, where a given z is set to zero. Returns error if there is no line segment intersection.

    If t and u are in [0, 1], there is an interesection between the two segments
    """
    x1 = vertexA[0]
    y1 = vertexA[1]
    x2 = vertexB[0]
    y2 = vertexB[1]
    x3 = target[0]
    y3 = target[1]
    x4 = pos[0]
    y4 = pos[1]
    intersect = []
    if abs(((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))) < 1e-9:
        return intersect
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    if (t <= 1 and t >= 0) and (u <= 1 and u >= 0):
        intersect = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return intersect

def phi(x, y, z, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Gaussian model in 3D space
    sigma and mu are (1,3) numpy arrays
    """

    # Phi 4
    tau = 20
    r = 4
    s = 2
    f = np.exp(
        -1
        * (
            ((x - r * np.cos(t / tau)) / s) ** 2
            + ((y - r * np.sin(t / tau)) / s) ** 2
            + (z / (s / 2)) ** 2
        )
    )

    return f


def phidot(x, y, z, mu, sigma, mu_past, sigma_past, t, dt, pi=None, sig_dig=4):
    """
    Used to calculate dc/dt
    """
    # Phi 4
    tau = 20
    r = 4
    s = 2
    f = phi(x, y, z, mu, sigma, t) * (
        -2 * r / (tau * s**2) * (x - r * np.cos(t / tau)) * np.sin(t / tau)
        + 2 * r / (tau * s**2) * (y - r * np.sin(t / tau)) * np.cos(t / tau)
    )
    return f


def weighted_phi(x, y, z, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Weighted Gaussian used to find the centroid
    """
    return np.multiply(np.array([x, y, z]), phi(x, y, z, mu, sigma, t, sig_dig=4)).T


def dcdt_function(x, y, z, mu, sigma, t, dt, c, pi=None, sig_dig=4):
    """
    Calclate dc/dt for Time Varying Densities
    """
    return np.multiply(
        (np.array([x, y, z]).T - c).T,
        phidot(
            x,
            y,
            z,
            np.array([mu[0]]),
            np.array([sigma[0]]),
            np.array([mu[1]]),
            np.array([sigma[1]]),
            t,
            dt,
            sig_dig=sig_dig,
        ),
    ).T


def cost_function(x, y, z, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4):
    """
    Calculate locational cost function H to compare different coverage control algorithms
    """
    try:
        dim = np.size(pi)
        if dim == 2:
            qq = np.sqrt((x - pi[0]) ** 2 + (y - pi[1]) ** 2)
        elif dim == 3:
            qq = np.sqrt((x - pi[0]) ** 2 + (y - pi[1]) ** 2 + (z - pi[2]) ** 2)

    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
    return qq * phi(x, y, z, mu, sigma, t, sig_dig=sig_dig)


def monte_carlo_integrate(
    points_all, volume, function, mu, sigma, t, dt, c, position, num_points=None
):
    """
    Used by lloyd_single_step to find mass and centroid using sampled points. Takes in sampled points, a function such as phi/weighted_phi, and the parameters of the density location. The volume normalizes the calculation
    """
    integral = []
    # Go one voronoi partition at a time
    for idx, points in enumerate(points_all):
        # Evaluate function on points
        function_result = function(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            mu,
            sigma,
            t,
            dt,
            c[idx],
            position[idx],
            sig_dig=4,
        )
        integral_result = np.multiply(
            volume[idx] / num_points, np.sum(function_result, axis=0)
        )
        integral.append(integral_result)

    return np.array(integral)  # in same order as inputted through vor


def massless_centroid(pos, target_pos, vor, partitions):
    """
    Find centroid when mass is zero

    The centroid must remain in the Voronoi cell and we want it to be as close as possible to the density position
    We take a ray which is from the blue position to the density position and test if it intersects a face
    Then we find the point where it intersects.
    TODO: Make sure it works in edge cases
    """
    print("Line 679 Position after Massless: ", pos)
    dim = 3
    if pos[-1] == 0.0:
        dim = 2
    vertices = vor["vertices"]
    if check_membership(vertices, target_pos):
        centroid = target_pos
        return centroid
    norm_dist = np.linalg.norm(target_pos - pos)
    rayDirection = (target_pos - pos) / norm_dist
    rayPoint = pos
    faces = vor["faces"]
    centroid = [0, 0, 0]
    centroids = []
    prevdist = norm_dist
    for face in faces:
        verticesIdx = face["vertices"]
        A = vertices[verticesIdx[0]]
        B = vertices[verticesIdx[1]]
        C = vertices[verticesIdx[2]]
        planeNormal = np.cross(B - A, C - A)
        t = (
            planeNormal[0] * (A[0] - pos[0])
            + planeNormal[1] * (A[1] - pos[1])
            + planeNormal[2] * (A[2] - pos[2])
        ) / (np.dot(A, A))
        proj_pi = np.array(
            [
                pos[0] + t * planeNormal[0],
                pos[1] + t * planeNormal[1],
                pos[2] + t * planeNormal[2],
            ]
        )
        orientation = np.dot(proj_pi - pos, planeNormal)
        if orientation < 0:
            planeNormal *= -1
        planePoint = A
        if dim == 2:
            intersect = point_line_intersect(A[0:2], B[0:2], target_pos, pos)
            if len(intersect) != 0:
                intersect = np.r_[intersect, 0.0]
                dist = np.linalg.norm(target_pos - intersect)
            else:
                dist = prevdist
        else:
            intersect = point_plane_intersect(
                planeNormal, planePoint, rayDirection, rayPoint
            )
            if intersect.size:
                dist = np.linalg.norm(target_pos - intersect)
            else:
                print("Edge case, unexpected Voronoi cell configuration")
        # check if there exists a valid intersection
        centroids.append(intersect)
        if dist < prevdist:
            centroid = intersect
            prevdist = dist
    return centroid

