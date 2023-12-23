import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def neumann(A, b, k):
    """
    Neumann approximation for the TVD-C algorithm
    Taken from Melcior's code

    Depreciated: sent into coverage control TVD_K class
    """
    B = np.eye(len(A)) - A
    bb = b.copy()
    u = b.copy()  # order 0 aproximation
    for j in range(k):
        bb = np.dot(B, bb)
        u = u + bb

    return u


def sing_perturbation(A, b, u0, dt, deta, eps, trim_u=False, max_u=3, debug_mode=False):
    """
    Calculates gradient direction based on Singular Perturbation
    Taken from Melcior's code

    Depreciated: sent into coverage control TVD_SP class
    """

    # A*u = b --> f(u) = 1/2||Au-b||**2
    # eps * du/dt = -grad(f)
    def grad(u):
        w = np.dot(A, u) - b
        return np.dot(A.T, w)

    u = u0.copy()

    if debug_mode:
        # SANITY CHECK THAT GRADIENT DESCEND WILL CONVERGE
        Hess = np.dot(A.T, A)
        [w, v] = np.linalg.eig(Hess)
        L = np.max(np.abs(w))
        index = np.argmax(np.abs(w))
        [w2, v2] = np.linalg.eig(A)
        index2 = np.argmax(np.abs(w2))
        print(w2[index2])
        if deta > 1 / L:
            print(
                "deta too big, singular perturbation may not converge, we need deta < 1/L"
            )
            print("1/L: ", 1 / L)
            print("deta: ", deta)

    t_int = dt / eps
    # ITERATE GRADIENT DESCEND
    n_steps = int(np.ceil(t_int / deta))
    print("Perturbation Steps:", n_steps)
    for step in range(n_steps):
        u = u - deta * grad(u)
        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    return u


def sing_perturbation_delayed(
    A, b, u0, u_past, dt, deta, eps, trim_u=False, max_u=3, debug_mode=False, d=None
):
    """
    Calculates gradient direction based on Singular Perturbation
    Taken from Melcior's code
    Depreciated: sent into coverage control TVD_SP class
    """
    # TVD-C computed u
    u_tvd = np.linalg.inv(A).dot(b)

    # A*u = b --> f(u) = 1/2||Au-b||**2
    # eps * du/dt = -grad(f)
    def grad(u):
        w = np.dot(A, u) - b
        return np.dot(A.T, w)

    u = u0.copy()
    t_int = dt / eps
    interval = 1
    if debug_mode:
        # SANITY CHECK THAT GRADIENT DESCEND WILL CONVERGE
        Hess = np.dot(A.T, A)
        [w, v] = np.linalg.eig(Hess)
        L = np.max(np.abs(w))
        index = np.argmax(np.abs(w))
        [w2, v2] = np.linalg.eig(A)
        index2 = np.argmax(np.abs(w2))
        print("cond num of Hess: ", np.linalg.cond(Hess))
        print("1/L: ", 1 / L)
        print("deta: ", deta)
        peta = 1 / (40 * L)
        psteps = int(np.ceil(t_int / peta))
        n_steps = int(np.ceil(t_int / deta))
        print("proposed eta: ", peta)
        print("proposed steps: ", psteps)
        if deta > 1 / L:
            print(
                "deta too big, singular perturbation may not converge, we need deta < 1/L"
            )
            print("1/L: ", 1 / L)
            print("deta: ", deta)
        u_p_vec = []
        m = 30  # push the rate of convergence
        for step in range(n_steps):
            u_next = u - m * peta * grad(u_past)
            u_past = u
            u = u_next
            if trim_u:
                max_grad = max_u * np.array([np.ones(len(u))]).T
                u = np.minimum(u, max_grad)
                u = np.maximum(u, -max_grad)
            if step % interval == 0:
                u_p_vec.append(u)
        t_p = np.arange(1, len(u_p_vec) + 1)
        u_p_vec = np.squeeze(np.array(u_p_vec))

    # ITERATE GRADIENT DESCEND
    n_steps = int(np.ceil(t_int / deta))
    print("Perturbation Steps:", n_steps)
    u_vec = []  # 1 step delayed gradient descent
    u_ndelay_vec = (
        []
    )  # Non delayed gradient descentcommunication needs for a distributed algorithm

    u = u0.copy()
    for step in range(n_steps):
        u_next = u - deta * grad(u_past)
        u_past = u
        u = u_next
        if step % interval == 0:
            u_vec.append(u)  # delayed

        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    u = u0.copy()
    for step in range(n_steps):
        u = u - deta * grad(u)
        if step % interval == 0:
            u_ndelay_vec.append(u)  # no delay
        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    u_vec = np.squeeze(np.array(u_vec))
    u_ndelay_vec = np.squeeze(np.array(u_ndelay_vec))

    if debug_mode:
        return u

    print("done")
    plt.close("all")
    return u


def sing_perturbation_hybrid(
    A,
    b,
    u0,
    dt,
    deta,
    eps,
    neighbors1,
    neighbors2,
    trim_u=False,
    max_u=3,
    debug_mode=False,
    d=None,
):
    """
    Calculates gradient direction based on Singular Perturbation
    Taken from Melcior's code
    A : num_agent
    Depreciated: sent into coverage control TVD_SP class
    """
    # TVD-C computed u
    # A*u = b --> f(u) = 1/2||Au-b||**2
    # eps * du/dt = -grad(f)
    def grad(u):
        w = np.dot(A, u) - b
        return np.dot(A.T, w)

    A_shape = A.shape  # gives tuple
    num_agents = int(len(A) / d)

    def hybrid_terms_1(A):
        """
        Computes the hybrid gradient update for each agent. There are better ways to compute this, but this is meant to do the calculation so that we have concrete simulation for the paper. A_1 in paper
        grad = A_bar*u + A_hat*u + b
        """
        A_bar = np.zeros(shape=A_shape)
        A_hat = np.zeros(shape=A_shape)
        b_bar = np.zeros(shape=(len(A), 1))
        for i in range(num_agents):
            temp_bar_ii = np.zeros(shape=(d, d))
            temp_bar_ij1 = np.zeros(shape=(d, d))
            temp_bar_ij2 = np.zeros(shape=(d, d))
            temp_b = np.zeros(shape=(d, 1))
            ii = A[i * d : (i + 1) * d, i * d : (i + 1) * d]
            for j in neighbors1[i]:
                ij = A[i * d : (i + 1) * d, j * d : (j + 1) * d]
                ji = A[j * d : (j + 1) * d, i * d : (i + 1) * d]
                jj = A[j * d : (j + 1) * d, j * d : (j + 1) * d]
                temp_bar_ii = ji @ ji
                temp_bar_ij1 = ii @ ij + ji @ jj
                temp_b += ji @ b[j * d : (j + 1) * d]
                if j in neighbors2[i][0]:
                    ki = A[j * d : (j + 1) * d, i * d : (i + 1) * d]
                    kj = A[j * d : (j + 1) * d, j * d : (j + 1) * d]
                    temp_bar_ij2 = ki @ kj
                A_bar[i * d : (i + 1) * d, j * d : (j + 1) * d] = (
                    -temp_bar_ij1 + temp_bar_ij2
                )

            for k in neighbors2[i][1]:
                # only 2 hop
                ki = A[k * d : (k + 1) * d, i * d : (i + 1) * d]
                kj = A[k * d : (k + 1) * d, j * d : (j + 1) * d]
                A_hat[i * d : (i + 1) * d, k * d : (k + 1) * d] = ki @ kj

            A_bar[i * d : (i + 1) * d, i * d : (i + 1) * d] = ii @ ii + temp_bar_ii

            b_bar[i * d : (i + 1) * d] = ii @ b[i * d : (i + 1) * d] - temp_b

        return A_bar, A_hat, b_bar

    def hybrid_terms(A):
        """
        Computes the hybrid gradient update for each agent. There are better ways to compute this, but this is meant to do the calculation so that we have concrete simulation for the paper
        grad = A_bar*u + A_hat*u + b. A in paper, not A_1.
        """
        A_bar = np.zeros(shape=A_shape)
        A_hat = np.zeros(shape=A_shape)
        b_bar = np.zeros(shape=(len(A), 1))
        for i in range(num_agents):
            temp_bar_ii = np.zeros(shape=(d, d))
            temp_bar_ij1 = np.zeros(shape=(d, d))
            temp_bar_ij2 = np.zeros(shape=(d, d))
            temp_b = np.zeros(shape=(d, 1))
            ii = A[i * d : (i + 1) * d, i * d : (i + 1) * d]
            for j in neighbors1[i]:
                ij = A[i * d : (i + 1) * d, j * d : (j + 1) * d]
                ji = A[j * d : (j + 1) * d, i * d : (i + 1) * d]
                jj = A[j * d : (j + 1) * d, j * d : (j + 1) * d]
                temp_bar_ii = ji @ ji
                temp_bar_ij1 = ii @ ij + ji @ jj
                temp_b += ji @ b[j * d : (j + 1) * d]
                if j in neighbors2[i][0]:
                    ki = A[j * d : (j + 1) * d, i * d : (i + 1) * d]
                    kj = A[j * d : (j + 1) * d, j * d : (j + 1) * d]
                    temp_bar_ij2 = ki @ kj
                A_bar[i * d : (i + 1) * d, j * d : (j + 1) * d] = (
                    -temp_bar_ij1 + temp_bar_ij2
                )

            for k in neighbors2[i][1]:
                # only 2 hop
                ki = A[k * d : (k + 1) * d, i * d : (i + 1) * d]
                kj = A[k * d : (k + 1) * d, j * d : (j + 1) * d]
                A_hat[i * d : (i + 1) * d, k * d : (k + 1) * d] = ki @ kj

            A_bar[i * d : (i + 1) * d, i * d : (i + 1) * d] = ii @ ii + temp_bar_ii

            b_bar[i * d : (i + 1) * d] = ii @ b[i * d : (i + 1) * d] - temp_b

        return A_bar, A_hat, b_bar

    def debugEig(A, name):
        """
        Calculates eigenvalues for a square matrix
        """
        [w, v] = np.linalg.eig(A)
        print(name, "Max Val: ", np.max(A), "Min Val: ", np.min(A))
        print("eigs: ", w)
        print("min eig: ", np.min(np.real(w)))
        print("max eig: ", np.max(np.real(w)))

    u = u0.copy()
    u_past = u
    t_int = dt / eps
    interval = 1
    # ITERATE GRADIENT DESCEND
    n_steps = int(np.ceil(t_int / deta))
    print("Perturbation Steps:", n_steps)
    if debug_mode:
        # SANITY CHECK THAT GRADIENT DESCEND WILL CONVERGE
        Hess = np.dot(A.T, A)
        [w, v] = np.linalg.eig(Hess)
        L = np.max(np.abs(w))
        index = np.argmax(np.abs(w))
        [w2, v2] = np.linalg.eig(A)
        index2 = np.argmax(np.abs(w2))
        print("cond num of Hess: ", np.linalg.cond(Hess))
        print("1/L: ", 1 / L)
        print("deta: ", deta)
        peta = 1 / (80 * L)
        psteps = int(np.ceil(t_int / peta))
        n_steps = int(np.ceil(t_int / deta))
        print("proposed eta: ", peta)
        print("proposed steps: ", psteps)
        if deta > 1 / L:
            print(
                "deta too big, singular perturbation may not converge, we need deta < 1/L"
            )
            print("1/L: ", 1 / L)
            print("deta: ", deta)
        u_p_vec = []
        m = 1  # push the rate of convergence
        for step in range(n_steps):
            u_next = u - m * peta * grad(u_past)
            u_past = u
            u = u_next
            if step % interval == 0:
                u_p_vec.append(u)
        t_p = np.arange(1, len(u_p_vec) + 1)
        u_p_vec = np.squeeze(np.array(u_p_vec))

    u_vec = []  # 1 step delayed gradient descent
    u_ndelay_vec = (
        []
    )  # Non delayed gradient descent communication needs a distributed algorithm
    u_hybrid_vec = []  # Hybrid

    # Non-Delayed
    u = u0.copy()
    for step in range(n_steps):
        u_next = u - deta * grad(u)
        u = u_next
        if step % interval == 0:
            u_vec.append(u)

        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    # Delayed
    u = u0.copy()
    for step in range(n_steps):
        u_next = u - deta * grad(u_past)
        u_past = u
        u = u_next
        if step % interval == 0:
            u_ndelay_vec.append(u)
        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    # Hybrid delay
    u = u0.copy()
    A_bar, A_hat, b_bar = hybrid_terms_1(np.dot(A.T, A))
    debugEig(A_bar, "A_bar")
    debugEig(A_hat, "A_hat")

    for step in range(n_steps):
        grad_temp = np.dot(A_bar, u) + np.dot(A_hat, u_past) - b_bar
        u_next = u - deta * grad_temp
        u_past = u
        u = u_next
        if step % interval == 0:
            u_hybrid_vec.append(u)  # hybrid
        if trim_u:
            max_grad = max_u * np.array([np.ones(len(u))]).T
            u = np.minimum(u, max_grad)
            u = np.maximum(u, -max_grad)

    u_vec = np.squeeze(np.array(u_vec))
    u_ndelay_vec = np.squeeze(np.array(u_ndelay_vec))
    u_hybrid_vec = np.squeeze(np.array(u_hybrid_vec))

    if debug_mode:
        return u
    return u


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
    neighbors2_1 = [val for val in neighbors2_temp if val in neighbors1_temp]
    neighbors2_2 = [
        val
        for val in neighbors2_temp
        if (val not in neighbors1_temp) and (val != voridx)
    ]
    return (neighbors2_1, neighbors2_2)


def interesting_point(pk, pkk, norm, mu):
    """
    Help integrator by telling it where most of the density will be. Find the projection of the expected value of the gaussian distribution onto the line defined by the start and stop points of integration. If it is not on the segment, then put the interesting point as 0 or 1. Doesn't work right now
    """
    t = max(0, min(1, np.sum((mu - pk) * (pkk - pk)) / norm))
    projection = pk + t * (pkk - pk)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(pk[0], pk[1], pk[2], color="red")
    ax.scatter3D(pkk[0], pkk[1], pkk[2], color="black")
    ax.scatter3D(projection[0], projection[1], projection[2], color="blue")
    ax.scatter3D(mu[0], mu[1], mu[2], color="green")
    plt.show()
    return t


def checkOrientation(neighbor, vertices, pi):
    """
    Get positively oriented vertices of the neighbor, from neighbor, vertices, pi
    """
    n_vertices = [vertices[x] for x in neighbor["vertices"]]
    n_vertices.append(n_vertices[0])
    dim = np.size(n_vertices[0])
    if len(n_vertices) > 2:
        # Make sure vertices are positively oriented
        A = n_vertices[0]
        B = n_vertices[1]
        C = n_vertices[2]
        normal = np.cross(B - A, C - A)
        if dim == 3:
            t = (
                normal[0] * (A[0] - pi[0])
                + normal[1] * (A[1] - pi[1])
                + normal[2] * (A[2] - pi[2])
            ) / (np.dot(A, A))
            proj_pi = np.array(
                [pi[0] + t * normal[0], pi[1] + t * normal[1], pi[2] + t * normal[2]]
            )
        elif dim == 2:
            t = (normal[0] * (A[0] - pi[0]) + normal[1] * (A[1] - pi[1])) / (
                np.dot(A, A)
            )
            proj_pi = np.array([pi[0] + t * normal[0], pi[1] + t * normal[1]])
        orientation = np.dot(proj_pi - pi, normal)
        if orientation > 0:
            n_vertices = np.flipud(n_vertices)
    else:
        print("2 or less vertices error, not enough vertices")
    return n_vertices


def dcdp_integration(
    pk,
    pkk,
    norm1,
    norm,
    c,
    pi,
    p_neigh,
    phi,
    mu,
    sigma,
    t,
    point=None,
    numPoints=150,
):
    integral_ii = np.zeros([3, 3])
    integral_ij = np.zeros([3, 3])
    pk = pk[..., np.newaxis].T
    pkk = pkk[..., np.newaxis].T

    s = np.linspace(0, 1, numPoints, endpoint=False)
    s = s[np.newaxis, ...].T
    q = s * pk + (1 - s) * pkk
    f = phi(q[:,0], q[:,1], q[:,2], mu, sigma, t)

    normRatio = norm1 / norm / numPoints / 2
    qmc = q - c
    for i, q_value in enumerate(q):
        if i == 0 or i == len(q):
            integral_ii += 0.5*np.outer(qmc[i], q_value - pi) * f[i]    
            integral_ij += -0.5*np.outer(qmc[i], q_value - p_neigh) * f[i]
        else:
            integral_ii += np.outer(qmc[i], q_value - pi) * f[i]
            integral_ij += -np.outer(qmc[i], q_value - p_neigh) * f[i]
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

    deriv = np.zeros([len(neighbors), 3, 3])
    deriv1 = deriv
    integral1 = np.zeros([3, 3])
    integral11 = np.zeros([3, 3])
    idx_vec = []
    for idx, neighbor in enumerate(neighbors):
        n_vertices = checkOrientation(neighbor, vertices, pi)
        norm = m * np.linalg.norm(pi - p_neigh[idx])
        if show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(c[0], c[1], c[2], c="green")
            ax.scatter(pi[0], pi[1], pi[2], c="black")
            for i, p in enumerate(n_vertices):
                ax.scatter(p[0], p[1], p[2])
                ax.text(p[0], p[1], p[2], "%s" % (str(i)), size=10, zorder=1, color="k")
            plt.show()

        for k in range(len(n_vertices) - 1):  # Do surface integral in 3D
            pk = n_vertices[k]
            pkk = n_vertices[k + 1]
            norm1 = np.linalg.norm(pkk - pk)
            integrationPoints = 100

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
