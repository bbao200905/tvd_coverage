import matplotlib.pyplot as plt
import numpy as np


def return_neighbors(vor, voridx, env):
    """
    Provides Neighbor Information given a pyvoro voronoi group. Meant for 3D computations of dc/dp

    Arguments:
        vor: all of the voronoi cells
        voridx: which voronoi cell you want neighbors of
        env: environment boundary

    Return:
        neighbors: list of faces which gives cell id and indices of vertices
        vertices: list of x,y,z  points
        p_neigh: neighbor x,y,z location
    """
    cell = vor[voridx]
    vertices = cell["vertices"]
    faces = cell["faces"]
    if [x["adjacent_cell"] for x in faces]:
        neighbors = list(filter(lambda e: e["adjacent_cell"] >= 0, faces))
    else:
        print("Voronoi faces dict does not contain adjacent_cell key")
    p_neigh = []
    for i, p in enumerate(neighbors):
        id = p["adjacent_cell"]
        n = vor[id]
        p_neigh.append(n["original"])
    return neighbors, vertices, p_neigh


def return_neighbors2(vor, voridx, env, neighbors1):
    """
    return 2-hop neighbors for hybrid TVD-SP algorithm

    Arguments:
        vor: all of the voronoi cells
        voridx: which voronoi cell you want neighbors of
        env: environment boundary
        neighbors1: list of idx of 1-hop neighbors for agent with voridx

    Return:
        neighbors2_1: idx of 2-hop neighbors that are also 1-hop neighbors
        neighbors2_2: idx of exclusive 2-hop neighbors
    """
    neighbors2_1 = {}
    neighbors2_2 = {}
    neighbors2_temp = []
    for neighbor in neighbors1:
        neighbors_temp1, _, _ = return_neighbors(vor, neighbor, env)
        for neighbor2 in neighbors_temp1:
            id = neighbor2["adjacent_cell"]
            if id not in neighbors2_temp:
                neighbors2_temp.append(id)

    neighbors1_temp = neighbors1
    # print('idx',voridx,'2-hop', neighbors2_temp,'1-hop',neighbors1_temp)

    neighbors2_1 = [val for val in neighbors2_temp if val in neighbors1_temp]
    neighbors2_2 = [
        val
        for val in neighbors2_temp
        if (val not in neighbors1_temp) and (val != voridx)
    ]
    # print('idx',voridx,'2-hop', neighbors2_2,'1-hop',neighbors2_1)

    return (neighbors2_1, neighbors2_2)


# def interesting_point(pk, pkk, norm, mu):
#     """
#     Help integrator by telling it where most of the density will be. Find the projection of the expected value of the gaussian distribution onto the line defined by the start and stop points of integration. If it is not on the segment, then put the interesting point as 0 or 1. Doesn't work right now
#     """
#     t = max(0, min(1, np.sum((mu-pk)*(pkk-pk))/norm))
#     projection = pk + t*(pkk-pk)
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')

#     ax.scatter3D(pk[0], pk[1], pk[2], color='red')
#     ax.scatter3D(pkk[0], pkk[1], pkk[2], color='black')
#     ax.scatter3D(projection[0], projection[1], projection[2], color='blue')
#     ax.scatter3D(mu[0], mu[1], mu[2], color='green')
#     plt.show()
#     # projDist = np.linalg.norm(projection-pk)
#     # test = projDist/norm
#     return t


def checkOrientation(neighbor, vertices, pi, p_neigh=None):
    """
    Get positively oriented vertices of the neighbor, from neighbor, vertices, pi
    """
    n_vertices = [vertices[x] for x in neighbor["vertices"]]
    # n_vertices.append(n_vertices[0])
    # dim = np.size(n_vertices[0])
    if len(n_vertices) > 1:
        # Make sure vertices are positively oriented
        A = np.append(pi, 0)
        B = np.append(n_vertices[0], 0)
        C = np.append(n_vertices[1], 0)

        # print(A)
        # a = np.vstack((A,B,C))
        # b = np.array([[1],[1],[1]])
        # # print(a,b)
        # O = np.hstack((b, a))
        # orientation = np.linalg.det(O)
        # print(f"{O=}, {orientation=}")

        AB = B - A
        AC = C - A
        orientation = np.cross(AB, AC)
        # Test orientation of vertices
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter(pi[0],pi[1],c='black')
        # # ax.scatter(proj_pi[0],proj_pi[1],proj_pi[2],c='red')
        # if p_neigh is not None:
        #     for i, p in enumerate(p_neigh):
        #         ax.scatter(p[0], p[1], color='k')
        #         ax.text(p[0], p[1], f"N {i}")
        # # for i, p in enumerate(vertices):
        # #     ax.scatter(p[0], p[1], color='k')
        # #     ax.text(p[0], p[1], f"  \n {i}")
        # for i, p in enumerate([B,C]):
        #     print(p)
        #     ax.scatter(p[0],p[1])
        #     ax.text(p[0],p[1],'%s' % (str(i)), size=10, zorder=1,color='k')
        # plt.show()
        # print(f"Be{n_vertices=}")

        if orientation[2] < 0.0:
            n_vertices = np.flipud(n_vertices)
        # print(f"Af{n_vertices}=")
        # print(f"{orientation[2]=}")
    else:
        print("1 or less vertices error, not enough vertices")
    return n_vertices


def dcdp_integration(
    pk, pkk, norm1, norm, c, pi, p_neigh, phi, mu, sigma, t, point=None, numPoints=150
):
    integral_ii = np.zeros([2, 2])
    integral_ij = np.zeros([2, 2])
    pk = pk[..., np.newaxis].T
    pkk = pkk[..., np.newaxis].T

    s = np.linspace(0, 1, numPoints, endpoint=False)
    s = s[np.newaxis, ...].T
    q = s * pk + (1 - s) * pkk
    # qq = [np.dstack(i) for i in q]
    # f = rv.pdf(x=q) # There is some discrepency. Check which phi is being used for all calculations
    f = phi(q[:, 0], q[:, 1], mu, sigma, t)

    normRatio = norm1 / norm / numPoints / 2
    qmc = q - c
    for i, q_value in enumerate(q):
        if i == 0 or i == len(q):
            integral_ii += 0.5 * np.outer(qmc[i], q_value - pi) * f[i]
            integral_ij += -0.5 * np.outer(qmc[i], q_value - p_neigh) * f[i]
        else:
            integral_ii += np.outer(qmc[i], q_value - pi) * f[i]
            integral_ij += -np.outer(qmc[i], q_value - p_neigh) * f[i]

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111)
    # ax.plot(s, f)
    # # ax.vlines(x=point, ymin=0, ymax=max(f), colors='r')

    # plt.show()

    return integral_ii * normRatio, integral_ij * normRatio


def dcdp(
    phi,
    pi,
    m,
    c,
    neighbors,
    vertices,
    p_neigh,
    mu,
    sigma,
    t,
    eps_rel=1e-3,
    show_plot=True,
):
    """
    Calculating dc/dp with surface integrals using boundary line integrals

    Arguments:
        poly: polyhedron representing a voronoi partition. It contains the neighbors and vertices. Computed using pyvoro.
        function: phi representing a gaussian that takes
        pi: agent (x,y,z) point
        m: mass in cell, directional(x,y,z)
        c: centroid of cell (x,y,z)
        neighbor: dict with neighbor idx in poly and vertices, from return_neighbors function
    """

    deriv = np.zeros([len(neighbors), 2, 2])
    deriv1 = np.zeros([len(neighbors), 2, 2])
    integral1 = np.zeros([2, 2])
    integral11 = np.zeros([2, 2])

    idx_vec = []
    for idx, neighbor in enumerate(neighbors):
        n_vertices = checkOrientation(neighbor, vertices, pi, p_neigh)
        norm = m * np.linalg.norm(pi - p_neigh[idx])

        if show_plot:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(c[0], c[1], c="green")
            ax.scatter(pi[0], pi[1], c="black")
            for i, p in enumerate(n_vertices):
                ax.scatter(p[0], p[1])
                ax.text(p[0], p[1], "%s" % (str(i)), size=10, zorder=1, color="k")
            plt.show()

        for k in range(len(n_vertices) - 1):  # Do surface integral in 2D
            pk = n_vertices[k]
            pkk = n_vertices[k + 1]
            norm1 = np.linalg.norm(pkk - pk)
            integrationPoints = 10

            [integralii, integralij] = dcdp_integration(
                pk,
                pkk,
                norm1,
                norm,
                c,
                pi,
                p_neigh[idx],
                phi,
                mu,
                sigma,
                t,
                numPoints=integrationPoints,
            )

            integral11 += integralii
            deriv1[idx, :, :] += integralij
        integral1 += integral11
        deriv[idx, :, :] = deriv1[idx, :, :]
        idx_vec.append(idx)

    return integral1, deriv, idx_vec
