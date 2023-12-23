"""
Simulations for Orbit Logic and Singular Perturbation Paper
"""
import os

import generate_plots
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pyvoro
import pyvoro3d
import tvd_calc as tvd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from utils import Ellipse, load_data, save_data

__author__ = "Brandon Bao"
__version__ = "1.0"
__maintainer__ = "Brandon Bao"
__email__ = "bjbao@ucsd.edu"

"""
Implementation:
# For each robot...
    for i in range(N):
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        j = topological_neighbors(L, i)
        # Compute the consensus algorithm
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)
"""
# Global
PLOTTING = False

class Agent:
    def __init__(self, position=None):
        if position is None:
            raise ValueError("Include agent position in initialization")
        if np.size(position) == 2 or position[-1] == 0.:
            self.dim = 2
            self.x = position[0]
            self.y = position[1]
            self.position = np.array(position)
            self.vx = None
            self.vy = None
        elif np.size(position) == 3:
            self.dim = 3
            self.x = position[0]
            self.y = position[1]
            self.z = position[2]
            self.position = np.array(position)
            self.vx = None
            self.vy = None
            self.vz = None
        else:
            raise ValueError("positions must be a np array of size 2 or 3")
        
        self.Voronoi = None
        self.neighbors = None 
        self.mass = None
        self.centroid = None

        self.dcdt = None
        self.dcdp = None
        self.t = 0

        self.mass_vec = []
        self.centroid_vec = []
        self.dcdt_vec = []
        self.dcdp_vec = []
        self.t_vec = []

class CoverageControl:
    """
    Class initializes with agent and target data, and algorithm parameters
    """
    def __init__(self, params=None, trajectory=None, agents=None):
        if params is None:
            self.env = [
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-10.0, 10.0]
            ]
            self.env_size = abs(np.array(self.env)[:,0]-np.array(self.env)[:,1])
            self.n_agents = len(agents)
            self.dim = agents[0].dim
            self.agents = agents
            self.trim_vel = True

            # Total time T and time step dt
            self.T = 100
            self.dt = 1
            self.t = 0
            self.seed = 0
            self.num_frames = int(round(self.T/self.dt))
            self.frame = 0
            # Proportional term to drive agents to centroid
            self.kappa = 1

            # If the agents move a lot in one time step, the control input is trimmed to max_vel
            self.max_vel = 1
            self.mass_eps = 1e-5 # condition of no mass for a Voronoi cell
            self.num_points = 2e6 # Monte Carlo Sampling Points
        else:
            self.env = params["env"]
            self.env_size = abs(np.array(self.env)[:,0]-np.array(self.env)[:,1])
            self.n_agents = len(agents)
            self.dim = agents[0].dim
            self.agents = agents
            self.trim_vel = params["trim_vel"]
            self.T = params["T"]
            self.dt = params["dt"]
            self.t = params["t"]
            self.seed = params["seed"]
            self.num_frames = params["num_frames"]
            self.frame = 0
            self.kappa = params["kappa"]
            self.max_vel = params["max_vel"]
            self.mass_eps = params["mass_eps"]
            self.num_points = params["num_points"]
        self.algorithm = None   
        if trajectory is None:
            raise ValueError('Not a valid Trajectory object')
        if agents is None:
            raise ValueError('Agents should be a list of Agent objects')
        self.agents = agents
        self.position = self.__getAgentsPosition()
        self.position_vec = [self.position] # initial positions, only positions will have positions for T=0
        self.velocity = None
        self.velocity_vec = []
        self.trajectory = trajectory
        self.mass_vec = []
        self.centroid_vec = []
        self.dcdt_vec = []
        self.dcdp_vec = []
        self.cost_vec = []
        self.t_vec = []
        
    def calculateVelocities(self):
        """
        Call after init
        """
        raise NotImplementedError()
    
    def saveData(self, dir, seed, model, save_params):
        data = [np.array(self.position_vec), np.array(self.mass_vec), np.array(self.centroid_vec), np.array(self.dcdt_vec), np.array(self.dcdp_vec), np.array(self.cost_vec), np.array(self.t_vec), self.dt, self.T, self.env, pyvoro3d.phi]
        self.name = dir + self.algorithm + save_params + 'phi_' + \
        str(model)+'_seed' + str(seed)
        save_data(dir, seed, model, self.algorithm, save_params, data)

    def getEssentialTerms(self):
        """
        Calculate Voronoi Cells, Neighbors, Mass, Centroid for all agents
        """
        self.__calculateVoronoi()
        self.__samplePoints()
        self.__calculateMassAndCentroid()
        self.__getNeighbors()

    def __calculateVoronoi(self):
        if self.dim == 3:
            self.vor = pyvoro.compute_voronoi(self.position, self.env, dispersion=10)
            
        elif self.dim == 2:
            self.vor = pyvoro.compute_voronoi(self.position, self.env, dispersion=10)
        for agent, cell in enumerate(self.vor):
                self.agents[agent].Voronoi = cell
        
    def __samplePoints(self):
        self.points_all = []
        for partition in self.vor:
            chull = ConvexHull(partition["vertices"])
            points, _ = pyvoro3d.random_points_stratified(chull, self.num_points, random_seed=self.seed) # stratified sampling of Polyhedra
            if self.dim == 2:
                points[:,2] = 0.
            self.points_all.append(points)

    def __getNeighbors(self):
        self.neighbors1 = {} # 1 hop neighbors
        self.neighbors2 = {} # 2 hop neighbors that are and are not 1 hop neighbors, 2 lists
        for i, partition  in enumerate(self.vor):
            [neighbors, vertices, p_neigh] = tvd.return_neighbors(self.vor, i,self.env)
            temp_neighbors = []
            for k, neighbor in enumerate(neighbors):
                j = neighbor['adjacent_cell'] # agent idx in dcdp
                temp_neighbors.append(j) # 1-hop
               
            self.neighbors1[i] = temp_neighbors
            self.neighbors2[i] = tvd.return_neighbors2(self.vor, i, self.env, temp_neighbors)
            self.agents[i].neighbors = (self.neighbors1, self.neighbors2)
    
    def __calculateMassAndCentroid(self):
        self.c = np.zeros(np.size(self.position))
        self.volume = np.array([x["volume"] for x in self.vor])
        [self.mu, self.sigma] = self.trajectory.getValuesAtT(self.t)
        self.mass = pyvoro3d.monte_carlo_integrate(self.points_all, self.volume, pyvoro3d.phi, np.array([self.mu[0]]), np.array([self.sigma[0]]), self.t, self.dt, self.c, self.position, num_points=self.num_points)
        self.zero_mass = np.where(self.mass < self.mass_eps)[0]
        self.centroid = pyvoro3d.monte_carlo_integrate(self.points_all, self.volume, pyvoro3d.weighted_phi, np.array([self.mu[0]]), np.array([self.sigma[0]]), self.t, self.dt, self.c, self.position, num_points=self.num_points)
        self.centroid = np.divide(self.centroid.T,self.mass).T
        if len(self.zero_mass):
            for i in self.zero_mass:
                # Move toward center of 3 std ellipse
                self.centroid[i] = pyvoro3d.massless_centroid(self.position[i], self.mu[0], self.vor[i], self.vor)
        for i, agent in enumerate(self.agents):
            agent.mass = self.mass[self.dim*i:self.dim*(i+1)] # check the size
            agent.mass_vec.append(agent.mass)
            agent.centroid = self.centroid[self.dim*i:self.dim*(i+1)]
            agent.centroid_vec.append(agent.centroid)
        self.mass_vec.append(self.mass)
        self.centroid_vec.append(self.centroid)

    def __getAgentsPosition(self):
        """
        Get position from all agents
        """
        position = []
        for agent in self.agents:
            if self.dim == 3:
                position.append(np.array([agent.x, agent.y, agent.z]))
            elif self.dim == 2: 
                position.append(np.array([agent.x, agent.y, 0.]))
            else:
                raise ValueError("Agents position not valid")
        return np.array(position)
    
    def __trimVelocities(self):
        """
        Trims algorithm output velocity to norm(vel) = max_vel
        """
        try:
            len(self.velocity)
        except ValueError:
            print("Velocities have not been calculated yet")
        if self.trim_vel:
            grad = self.velocity
            normed_vel = np.linalg.norm(self.velocity,axis=1)
            unit_vel = [velocity/normed_vel[i] for i,velocity in enumerate(self.velocity)]
            vel_ratios = self.max_vel/normed_vel
            for i, vel_ratio in enumerate(vel_ratios):
                if vel_ratio < 1.:
                    grad[i] = vel_ratio*self.velocity[i]
            
            self.velocity = grad
        else:
            pass

    def __calculateCost(self):
        """
        Cost for a step
        """
        self.cost = sum(pyvoro3d.monte_carlo_integrate(self.points_all, self.volume, pyvoro3d.cost_function, np.array([self.mu[0]]), np.array([self.sigma[0]]), self.t, self.dt, self.c, self.position, num_points=self.num_points))
        try:
            self.cost_vec.append(self.cost)
        except ValueError:
            print("Check the size of the cost array\n","cost: ", np.size(self.cost))

    def __updatePosVelTime(self):
        for i, agent in enumerate(self.agents):
            agent.vx = self.velocity[i,0]
            agent.vy = self.velocity[i,1]
            agent.x = self.new_pos[i,0]
            agent.y = self.new_pos[i,1]
            if self.dim == 2:
                agent.position = np.array([self.new_pos[i,0], self.new_pos[i,1]])
            elif self.dim == 3:
                agent.vz = self.velocity[i,2]
                agent.z = self.new_pos[i,2]
                agent.position = np.array([self.new_pos[i,0], self.new_pos[i,1], self.new_pos[i,2]])
            agent.t += self.dt
            agent.t_vec.append(agent.t)
        self.position = self.new_pos
        self.position_vec.append(self.new_pos)
        self.velocity_vec.append(self.velocity)
        self.t_vec.append(self.t)

    def velocityCleanup(self):
        """
        Makes sure agents don't leave cells. Updates agent position, velocity, time, cost. Calculates final velocity.
        """
        self.velocity = self.velocity.reshape(len(self.position),3)
        self.__trimVelocities() # make sure velocity is not greater than self.max_vel
        self.new_pos = self.position + self.velocity
        # Don't allow agent to leave environment
        if self.dim == 3:
            max_bound = np.array([self.env[0][1],self.env[1][1],self.env[2][1]])
            min_bound = np.array([self.env[0][0],self.env[1][0],self.env[2][0]])
            max_pos = max_bound*np.ones([len(self.position),self.dim])-1e-6*np.random.rand(self.dim)
            min_pos = min_bound*np.ones([len(self.position),self.dim])+1e-6*np.random.rand(self.dim)
            self.new_pos = np.minimum(self.new_pos, max_pos)
            self.new_pos = np.maximum(self.new_pos, min_pos)
            print("\nNew Position:", self.new_pos)
            self.velocity = self.new_pos - self.position
            print("\nVelocity:", self.velocity) 
        elif self.dim == 2:
            max_bound = np.array([self.env[0][1],self.env[1][1], 0.])
            min_bound = np.array([self.env[0][0],self.env[1][0], 0.])
            max_pos = max_bound*np.ones([len(self.position), 3])-1e-6*np.random.rand(3)
            min_pos = min_bound*np.ones([len(self.position), 3])+1e-6*np.random.rand(3)
            self.new_pos = np.minimum(self.new_pos, max_pos)
            self.new_pos = np.maximum(self.new_pos, min_pos)
            # Enforce z = 0
            self.new_pos[:,-1] = 0
            print("\nNew Position:", self.new_pos)
            self.velocity = self.new_pos - self.position
            # Enforce z = 0
            self.velocity[:,-1] = 0
            print("\nVelocity:", self.velocity) 
        self.__updatePosVelTime()
        self.__calculateCost()

class Lloyd(CoverageControl):
    def __init__(self, params=None, trajectory=None, agents=None):
        super().__init__(params, trajectory, agents)
        self.algorithm = 'Lloyd'

    def calculateVelocities(self):
        """
        Call after init
        """
        self.frame += 1
        self.t = self.dt*self.frame
        if np.shape(self.position)[1] == 2:
            self.position = np.c_[self.position, np.zeros(shape=(3,))]
        self.getEssentialTerms()
        self.velocity  = -self.kappa*(self.position-self.centroid)
        super().velocityCleanup()
        return self.velocity

class TVD(CoverageControl):
    """
    Implements TVD-C algorithms
    """
    def __init__(self, params=None, trajectory=None, agents=None):
        super().__init__(params, trajectory, agents)
        # Number of hops for distributed TVD-K algorithm
        self.max_eig_norm_vec = []
        self.min_eig_norm_vec = []
        self.min_eig_vec = []
        self.algorithm = 'TVD-C'   
        self.A_min_eigs = []
        self.A_max_eigs = []

    def calculateVelocities(self):
        """
        Call after init
        """
        self.t += self.dt # will this update t for super()?
        self.frame += 1
        self.getEssentialTerms()
        self.calculateTVDTerms()
        grad1 = np.linalg.inv(self.A).dot(self.b)
        self.velocity = grad1
        self.velocityCleanup()
        return self.velocity
    
    def calculateTVDTerms(self):
        self.fix_dim = False
        if self.dim == 2:
            self.fix_dim = True
            self.dim = 3
        self.dcdt = pyvoro3d.monte_carlo_integrate(self.points_all, self.volume, pyvoro3d.dcdt_function, self.mu, self.sigma, self.t, self.dt, c=self.centroid, position=self.position, num_points=self.num_points)
        self.dcdt = np.divide(self.dcdt.T,self.mass).T
        self.dcdp = np.zeros([self.n_agents*self.dim, self.n_agents*self.dim]) 
        for i, partition in enumerate(self.vor):
            [neighbors, vertices, p_neigh] = tvd.return_neighbors(self.vor,i,self.env)
            [integral1, deriv, _] = tvd.dcdp(pyvoro3d.phi, self.position[i], self.mass[i], self.centroid[i], neighbors, vertices, p_neigh, np.array([self.mu[0]]), np.array([self.sigma[0]]), self.t, eps_rel=1e-3, show_plot=False) #output 2 by 2 for self.dim = 2
            self.dcdp[self.dim * i:self.dim * (i + 1), self.dim * i:self.dim * (i + 1)] = integral1

            # Add neighbor tuple to return list
            for k, neighbor in enumerate(neighbors):
                j = neighbor['adjacent_cell'] # agent idx in dcdp
                self.dcdp[self.dim * i:self.dim * (i + 1), self.dim * j:self.dim * (j + 1)] = deriv[k]
        if len(self.zero_mass):
            for i in self.zero_mass:
                # Move toward center of 3 std ellipse
                self.dcdt[i] = 0
                self.dcdp[i*(self.dim):i*(self.dim)+self.dim,:] = 0
        try:
            for i, paritition in enumerate(self.vor):
                self.agents[i].dcdt = self.dcdt[self.dim*i:self.dim*(i+1)]
                self.agents[i].dcdp = self.dcdp[self.dim*i:self.dim*(i+1),:]
                self.agents[i].dcdt_vec.append(self.agents[i].dcdt)
                self.agents[i].dcdp_vec.append(self.agents[i].dcdp)
        except ValueError:
            print("Messed up the dimensions")

        I = np.eye(np.size(self.position)) # check size of self.position
        
        self.A = I - self.dcdp
        self.debugEig(self.A, "A", self.A_min_eigs, self.A_max_eigs)
        self.b = -self.kappa * (self.position.reshape(self.dim*len(self.position),1) - self.centroid.reshape(self.dim*len(self.position),1)) + self.dcdt.reshape(self.dim*len(self.position),1)
        if self.fix_dim:
            self.dim = 2

    def debugEig(self, A, name, min_eig_vec, max_eig_vec):
        """
        Calculates eigenvalues for a square matrix, saves min and
        """
        [w, v] = np.linalg.eig(A)
        min_eig = np.min(np.real(w))
        max_eig = np.max(np.real(w))
        min_eig_vec.append(min_eig)
        max_eig_vec.append(max_eig)
    
    def saveEig(self, name, min_eig_vec, max_eig_vec):
        """
        Plot Eigs:
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(3.5, 3.5),
            sharex=True
        )
        ax1.plot(self.t_vec, min_eig_vec)
        ax2.plot(self.t_vec, max_eig_vec)
        # ax1.annotate(
        #     "Min: {:5.2f}\nFinal: {:5.4f}".format(min(min_eig_vec), min_eig_vec[-1]),
        #     xy=(0.3, 0.8),
        #     ha="right",
        #     va="bottom",
        #     xycoords="figure fraction",
        #     xytext=(0.99, 0.01),
        #     textcoords="axes fraction",
        # )
        ax1.set(ylabel="Eigenvalue")
        # ax2.annotate(
        #     "Max: {:5.2f}\nFinal: {:5.4f}".format(max(max_eig_vec), max_eig_vec[-1]),
        #     xy=(0.3, 0.8),
        #     ha="right",
        #     va="bottom",
        #     xycoords="figure fraction",
        #     xytext=(0.99, 0.01),
        #     textcoords="axes fraction",
        # )
        ax2.set(xlabel="Iteration", ylabel="Eigenvalue")
        ax1.title.set_text("Min Eigenvalues")
        ax2.title.set_text("Max Eigenvalues")
        l_lim = -1.1
        u_lim = 1.1
        tick = 0.1
        if name == 'A':
            l_lim = 0.4
            u_lim = 1.1
            tick = 0.2
        elif name == 'A Hat':
            l_lim = -0.35
            u_lim = 0.35
            tick = 0.1
        elif name == 'A Bar':
            l_lim = 0.4
            u_lim = 1.1
            tick = 0.2
        ax1.set_ylim([l_lim,u_lim])
        ax2.set_ylim([l_lim,u_lim])
        loc = plticker.MultipleLocator(base=tick) # this locator puts ticks at regular intervals
        ax1.yaxis.set_major_locator(loc)
        ax2.yaxis.set_major_locator(loc)

        path = "data/" + self.algorithm + self.save_params + "/"
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        plt.savefig(path + name + ".png", dpi=600, bbox_inches="tight")
        plt.show
        plt.close

    def saveData(self, dir, seed, model, save_params):
        self.save_params = save_params
        super().saveData(dir, seed, model, save_params)
        self.saveEig("A", self.A_min_eigs, self.A_max_eigs)        

class TVD_K(TVD):
    def __init__(self, params=None, k=None, trajectory=None, agents=None):
        super().__init__(params, trajectory, agents)
        self.k = k # number of hops
        self.algorithm = 'TVD-K'

    def calculateVelocities(self):
        """
        Call after init
        """
        self.t += self.dt # will this update t for super()?
        self.frame += 1
        self.getEssentialTerms()
        self.calculateTVDTerms()
        grad = self.__neumann()
        self.velocity = grad
        self.velocityCleanup()
        return self.velocity
    
    def __neumann(self):
        """
        Neumann approximation for the TVD-C algorithm
        Taken from Melcior's code
        """
        B = np.eye(len(self.A)) - self.A
        bb = self.b.copy()
        u = self.b.copy()  # order 0 aproximation
        for j in range(self.k):
            bb = np.dot(B, bb)
            u = u + bb
        return u

class TVD_SP(TVD):
    def __init__(self, params=None, eps=None, deta=None, algorithm=None, trajectory=None, agents=None):
        super().__init__(params, trajectory, agents)
        self.algorithm = algorithm
        # Singular Perturbation time scaling parameter, must be small ie: (1e-2 to 1e-6). For larger eps, try TVD_SSP. deta> 1/L where L is the largest 
        # |eigval| of the Hessian of the A matrix. A = np.eye(np.size(position)) -dcdp.
        # t_scaling = dt/eps, steps = int(t_scaling/deta)
        self.eps = eps #1e-1
        # N = dt/(eps*deta) = # of time steps for du/deta dynamics. 
        self.deta = deta #1e-3
        self.A_1_bar_min_eigs = []
        self.A_1_bar_max_eigs = []
        self.A_1_hat_min_eigs = []
        self.A_1_hat_max_eigs = []
        self.A_bar_min_eigs = []
        self.A_bar_max_eigs = []
        self.A_hat_min_eigs = []
        self.A_hat_max_eigs = []
    
    def calculateVelocities(self):
        """
        Call after init
        """
        self.t += self.dt
        self.frame += 1
        self.getEssentialTerms()
        self.calculateTVDTerms()
        self.velocity = self.__singularPerturbation()
        self.velocityCleanup()
        return self.velocity
    
    def __singularPerturbation(self):
        """
        Calculate velocity using Singular Perturbation Theory
        """
        if self.algorithm == 'TVD-SP':
            if self.frame == 1:
                self.grad0 = self.b
            grad = self.sing_perturbation(self.A, self.b, self.grad0, self.dt, self.deta, self.eps, trim_u=True, max_u=self.max_vel, debug_mode = True)
            self.grad0 = grad
        elif self.algorithm == 'TVD-SSP':
            self.grad0 = self.b
            grad = self.sing_perturbation(self.A, self.b, self.grad0, self.dt, self.deta, self.eps, trim_u=True, max_u=self.max_vel, debug_mode = False)
            self.grad0 = grad
        elif self.algorithm == 'TVD-SP-hybrid':
            self.grad0 = self.b
            grad = self.sing_perturbation_hybrid(self.A, self.b, self.grad0, self.dt, self.deta, self.eps, self.neighbors1, self.neighbors2, trim_u=True, max_u=self.max_vel, debug_mode = False, d = self.dim)
            self.grad0 = grad    
        elif self.algorithm == 'TVD-SP-delayed':
            self.grad0 = self.b
            grad = self.sing_perturbation_delayed(self.A, self.b, self.grad0, self.grad0, self.dt, self.deta, self.eps, trim_u=True, max_u=3, debug_mode = False, d = self.dim)
            self.grad0 = grad
        else:
            raise ValueError("Algorithm choices are 'TVD-SP', 'TVD-SSP', 'TVD-SP-hybrid', 'TVD-SP-delayed'")
        return self.grad0
    
    def __checkConvergence(self):
        #SANITY CHECK THAT GRADIENT DESCEND WILL CONVERGE
        Hess = np.dot(self.A.T, self.A)
        [w, v] = np.linalg.eig(Hess)
        L = np.max(np.abs(w))
        index = np.argmax(np.abs(w))
        [w2, v2] = np.linalg.eig(self.A)
        index2 = np.argmax(np.abs(w2))
        print(w2[index2])
        self.m_deta = self.deta
        
        if self.deta > 1/L:
            print(
                'deta too big, singular perturbation may not converge, we need deta < 1/L')
            print('1/L: ', 1/L)
            print('deta: ', self.deta)
            self.m_deta = 1/(2*L*self.eps)
        
    def saveData(self, dir, seed, model, save_params):
        super().saveData(dir, seed, model, save_params)
        if self.algorithm == 'TVD-SP-hybrid':
            super().saveEig("A Bar", self.A_bar_min_eigs, self.A_bar_max_eigs)
            super().saveEig("A Hat", self.A_hat_min_eigs, self.A_hat_max_eigs)
        
    def __gradient(A, b, u):
        """
        Calculates gradient used in Singular Perturbation Algorithms

        A*u = b --> f(u) = 1/2||Au-b||**2
        eps * du/dt = -grad(f)
        grad(f) = A.T(A*u-b)
        """
        w = np.dot(A, u) - b
        return np.dot(A.T, w)

    def sing_perturbation(self, A, b, u0, dt, deta, eps, trim_u=False, max_u=3, debug_mode=False):
        """
        Calculates gradient direction based on Singular Perturbation
        Taken from Melcior's code
        """
        # A*u = b --> f(u) = 1/2||Au-b||**2
        # eps * du/dt = -grad(f)
        def gradient(A, b, u):
            """
            Calculates gradient used in Singular Perturbation Algorithms

            A*u = b --> f(u) = 1/2||Au-b||**2
            eps * du/dt = -grad(f)
            grad(f) = A.T(A*u-b)
            """
            w = np.dot(A, u) - b
            return np.dot(A.T, w)

        u = u0.copy()

        if debug_mode:
            self.__checkConvergence()

        t_int = dt/eps
        # ITERATE GRADIENT DESCEND
        n_steps = int(np.ceil(t_int/deta))
        print('Perturbation Steps:', n_steps)
        for step in range(n_steps):
            u = u - deta * gradient(A, b, u)
            if trim_u:
                max_grad = max_u * np.array([np.ones(len(u))]).T
                u = np.minimum(u, max_grad)
                u = np.maximum(u, -max_grad)

        return u
    
    def sing_perturbation_delayed(self, A, b, u0, u_past, dt, deta, eps, trim_u=False, max_u=3, debug_mode=False, d=None):
        """
        Calculates 1 time step delayed gradient descent
        """
        def gradient(A, b, u):
            """
            Calculates gradient used in Singular Perturbation Algorithms

            A*u = b --> f(u) = 1/2||Au-b||**2
            eps * du/dt = -grad(f)
            grad(f) = A.T(A*u-b)
            """
            w = np.dot(A, u) - b
            return np.dot(A.T, w)
        u = u0.copy()
        t_int = dt/eps
        interval = 1
        if debug_mode:
            raise NotImplementedError("See tvd_calc.py for plots, debugging, and comparison to normal sing_perturbation")

        # ITERATE GRADIENT DESCEND
        n_steps = int(np.ceil(t_int/deta))
        print('Perturbation Steps:', n_steps)
        u_vec = [] # 1 step delayed gradient descent
        u_ndelay_vec = [] # Non delayed gradient descent communication needs for a distributed algorithm
        
        u = u0.copy() 
        for step in range(n_steps):
            u_next = u - deta * gradient(A, b, u_past)
            u_past = u
            u = u_next
            if step % interval == 0:
                u_vec.append(u) # delayed

            if trim_u:
                max_grad = max_u * np.array([np.ones(len(u))]).T
                u = np.minimum(u, max_grad)
                u = np.maximum(u, -max_grad)
        
        u_vec = np.squeeze(np.array(u_vec))

        return u

    def sing_perturbation_hybrid(self, A, b, u0, dt, deta, eps, neighbors1, neighbors2, trim_u=False, max_u=3, debug_mode=False, d=None):
        """
        Calculates gradient direction based on Singular Perturbation
        Taken from Melcior's code
        A : num_agent
        """
        
        A_shape = A.shape # gives tuple
        num_agents = self.n_agents

        def hybrid_terms_1(A):
            """
            Computes the hybrid gradient update for each agent. There are better ways to compute this, but this is meant to do the calculation so that we have concrete simulation for the paper. A_1 in paper
            grad = A_bar*u + A_hat*u + b
            """
            A_bar = np.zeros(shape=A_shape)
            A_hat = np.zeros(shape=A_shape)
            b_bar = np.zeros(shape=(len(A),1))
            for i in range(num_agents):
                temp_bar_ii = np.zeros(shape=(d,d))
                temp_bar_ij1 = np.zeros(shape=(d,d))
                temp_bar_ij2 = np.zeros(shape=(d,d))
                temp_b = np.zeros(shape=(d,1))
                ii = A[i*d:(i+1)*d,i*d:(i+1)*d]
                for j in neighbors1[i]:
                    ij = A[i*d:(i+1)*d,j*d:(j+1)*d]
                    ji = A[j*d:(j+1)*d,i*d:(i+1)*d]
                    jj = A[j*d:(j+1)*d,j*d:(j+1)*d]
                    temp_bar_ii = ji@ji
                    temp_bar_ij1 = ii@ij+ji@jj
                    temp_b += ji@b[j*d:(j+1)*d]
                    if j in neighbors2[i][0]:
                        ki = A[j*d:(j+1)*d,i*d:(i+1)*d]
                        kj = A[j*d:(j+1)*d,j*d:(j+1)*d]
                        temp_bar_ij2 = ki@kj
                    A_bar[i*d:(i+1)*d,j*d:(j+1)*d] = -temp_bar_ij1 + temp_bar_ij2

                for k in neighbors2[i][1]:
                    # only 2 hop
                    ki = A[k*d:(k+1)*d,i*d:(i+1)*d]
                    kj = A[k*d:(k+1)*d,j*d:(j+1)*d]
                    A_hat[i*d:(i+1)*d,k*d:(k+1)*d] = ki@kj

                A_bar[i*d:(i+1)*d,i*d:(i+1)*d] = ii@ii + temp_bar_ii
                
                b_bar[i*d:(i+1)*d] = ii@b[i*d:(i+1)*d] - temp_b
            
            return A_bar, A_hat, b_bar

        def hybrid_terms(A):
            """
            Computes the hybrid gradient update for each agent. There are better ways to compute this, but this is meant to do the calculation so that we have concrete simulation for the paper
            grad = A_bar*u + A_hat*u + b. Alternative A in paper, not A_1.
            """
            A_bar = np.zeros(shape=A_shape)
            A_hat = np.zeros(shape=A_shape)
            b_bar = np.zeros(shape=(len(A),1))
            for i in range(num_agents):
                temp_bar_ii = np.zeros(shape=(d,d))
                temp_bar_ij1 = np.zeros(shape=(d,d))
                temp_bar_ij2 = np.zeros(shape=(d,d))
                temp_b = np.zeros(shape=(d,1))
                ii = A[i*d:(i+1)*d,i*d:(i+1)*d]
                for j in neighbors1[i]:
                    ij = A[i*d:(i+1)*d,j*d:(j+1)*d]
                    ji = A[j*d:(j+1)*d,i*d:(i+1)*d]
                    jj = A[j*d:(j+1)*d,j*d:(j+1)*d]
                    temp_bar_ii = ji@ji
                    temp_bar_ij1 = ii@ij+ji@jj
                    temp_b += ji@b[j*d:(j+1)*d]
                    A_bar[i*d:(i+1)*d,j*d:(j+1)*d] = -temp_bar_ij1

                for k in neighbors2[i][0]:
                    # only 2 hop
                    ki = A[k*d:(k+1)*d,i*d:(i+1)*d]
                    kj = A[k*d:(k+1)*d,j*d:(j+1)*d]
                    A_hat[i*d:(i+1)*d,k*d:(k+1)*d] = ki@kj

                A_bar[i*d:(i+1)*d,i*d:(i+1)*d] = ii@ii + temp_bar_ii
                
                b_bar[i*d:(i+1)*d] = ii@b[i*d:(i+1)*d] - temp_b
            
            return A_bar, A_hat, b_bar
            
        u = u0.copy()
        u_past = u
        t_int = dt/eps
        interval = 1
        # ITERATE GRADIENT DESCEND
        n_steps = int(np.ceil(t_int/deta))
        print('Perturbation Steps:', n_steps)
        if debug_mode:
            raise NotImplementedError("Use tvd_calc.py to see eigenvalues")
        u_hybrid_vec = [] # Hybrid
        
        # Hybrid delay
        u = u0.copy() 
        # A_bar, A_hat, b_bar = hybrid_terms(np.dot(A.T,A)) # A alt in paper
        A_bar, A_hat, b_bar = hybrid_terms_1(np.dot(A.T,A)) # A_1 in paper
        self.debugEig(A_bar, 'A_bar', self.A_bar_min_eigs, self.A_bar_max_eigs)
        self.debugEig(A_hat, 'A_hat', self.A_hat_min_eigs, self.A_hat_max_eigs)

        for step in range(n_steps):
            grad_temp = np.dot(A_bar, u) + np.dot(A_hat, u_past) - b_bar
            u_next = u - deta * grad_temp
            u_past = u
            u = u_next
            if step % interval == 0:
                u_hybrid_vec.append(u) # hybrid
            if trim_u:
                max_grad = max_u * np.array([np.ones(len(u))]).T
                u = np.minimum(u, max_grad)
                u = np.maximum(u, -max_grad)
        
        u_hybrid_vec = np.squeeze(np.array(u_hybrid_vec))
        
        return u
            

def main():
    """
    Simulation Settings:

    position: Initial agent position. Supports many agents.
    env: Environment bounds, limited to rectangular prism shape.
    mu_a: Initial start position of target agent.
    mu_b: Final position in simulation. The generator will make linear path between the points.
    sigma_a, sigma_b: Covariance at start and end. Generator linearly interpolates value at each frame.
    num_frames: Number of frames for sim. Will affect density speed.
    """
    # Parameters
    dim = 3
    env = [
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0]
    ]
    env_size = abs(np.array(env)[:,0]-np.array(env)[:,1])
        
    n_agents = 3
    seed = 0
    np.random.seed(seed)
    agent_pos = np.random.rand(n_agents,3)*env_size + np.array(env)[:,0]
    if dim == 2:
        agent_pos[:,2] = 0.
    Agents = []
    for n in range(n_agents):
        Agents.append(Agent(agent_pos[n,:]))
    Agents = np.array(Agents)
    trim_vel = True
    
    # Total time T and time step dt
    T = 100
    dt = 1
    t = 0
    # 0.4 m/s max speed
    num_frames = int(round(T/dt))
    frame = 0
    # Proportional term to drive agents to centroid
    kappa = 1

    # If the agents move a lot in one time step, the control input is trimmed to max_vel
    # density_speed = np.sqrt(300)/num_frames # distance over time, speed for each time step
    max_vel = 1.
    mass_eps = 1e-5 # condition of no mass in a Voronoi cell

    num_points = 1e4 # Monte Carlo Sampling Points
    # TVD Parameters
    # Number of hops for distributed TVD-K algorithm
    k = 1

    # Singular Perturbation time scaling parameter, must be small ie: (1e-2 to 1e-6). For larger eps, try TVD_SSP. deta> 1/L where L is the largest 
    # |eigval| of the Hessian of the A matrix. A = np.eye(np.size(position)) -dcdp.
    # t_scaling = dt/eps, steps = int(t_scaling/deta)
    eps = 1e-1
    # eps = 5e-2
    # N = dt/(eps*deta) = # of time steps for du/deta dynamics. 
    deta = 1e-3


    # Trajectory
    mu_a = [10, 10, 10]
    mu_b = [0, 0, 0]

    sigma_a = [2, 2, 2]
    sigma_b = [2, 2, 2]
    radius = 4
    center = np.array([0, 0, 0])
    rotations = 5
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = 1
    d = 1
    trajectory = Ellipse(mu_a, mu_b, sigma_a, sigma_b, num_frames, dim, radius, center, a, b, c, d, rotations)
    params = {
        "env" : env,
        "trim_vel" : trim_vel,
        "seed" : seed,
        "T" : T,
        "dt" : dt,
        "t" : t,
        "num_frames" : num_frames,
        "kappa" : kappa,
        "max_vel" : max_vel,
        "mass_eps" : mass_eps,
        "num_points" : num_points,
    }
    dir = "data/"
    model = 2
    save_params = "agents" + str(n_agents)+'eps' + str(eps) + 'deta' + str(deta) + "max_u" + str(max_vel) + "T" + str(T) + "dt" + str(dt) + "samples"+ str(num_points)
    if dim == 2:
        save_params += "dim" + str(dim)
    # Run coveragecontrol here
    CoverageControl = Lloyd(params, trajectory, Agents)
    algorithm = 'Lloyd'
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

    CoverageControl = TVD(params, trajectory, Agents)
    algorithm = 'TVD-C'
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

    CoverageControl = TVD_K(params, k, trajectory, Agents)
    algorithm = 'TVD-K'
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

    algorithm = 'TVD-SP'
    CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

    algorithm = 'TVD-SP-hybrid'
    CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

    algorithm = 'TVD-SP-delayed'
    CoverageControl = TVD_SP(params, eps, deta, algorithm, trajectory, Agents)
    if not PLOTTING:
        for frame in tqdm(range(1,num_frames)):
            print("\nStarting Frame = ",frame,"\n")
            velocity = CoverageControl.calculateVelocities()
        CoverageControl.saveData(dir, seed, model, save_params)
    [positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env, phi_func] = load_data(dir, seed, model, algorithm, save_params)
    generate_plots.plot_figures(positions, mass, centroid, dcdt, dcdp, cost, t, dt, T, env,  algorithm, save_params, phi_func)

if __name__ == "__main__":
    main()