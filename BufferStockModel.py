import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from InterpolationFunctions import *

class BufferStockModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20 # time periods
        
        # preferences
        par.beta = 0.99 # discount factor
        par.rho = 1.5 # CRRA coefficient

        # income
        par.G = 1.02 # income growth level
        par.sigma_trans = 0.1 # transitory income shock standard deviation
        par.sigma_perm = 0.1 # permanent income shock standard deviation

        # saving
        par.r = 0.03 # interest rate

        # grid
        par.max_m = 5.0 # maximum point in resource grid
        par.num_m = 50 # number of grid points in resource grid    

        par.Nxi = 5 # number of points in transitory income shock expectaion
        par.Npsi = 5  # number of points in permanent income shock expectaion

        # EGM
        par.max_A_pd = 5.0

        # iEGM
        par.num_C = 30 # number of points in pre-computation grid
        par.max_C = 10.0 # maximum point in pre-computation grid
        par.unequal_C = 1.1

        par.interp_method = 'linear' # numerical, linear, Bspline
        par.interp_inverse = False # True: interpolate inverse consumption
        par.interp_degree = 8

        # simulation
        par.seed = 9210
        par.simN = 10_000 # number of consumers simulated

        # solution method
        par.method = 'vfi' # vfi, egm, or iegm.


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.m_grid = nonlinspace(0.00001,par.max_m,par.num_m,1.1) # always have a bit of resources
        par.a_grid = nonlinspace(0.00001,par.max_A_pd,par.num_m,1.1) # always have a bit of resources

        # b. income shock grids
        par.xi_grid,par.xi_weight = log_normal_gauss_hermite(par.sigma_trans,par.Nxi)
        par.psi_grid,par.psi_weight = log_normal_gauss_hermite(par.sigma_perm,par.Npsi)

        # b. solution arrays
        shape = (par.T,par.num_m)
        sol.c = np.nan + np.zeros(shape)
        sol.m = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # c. simulation arrays
        shape = (par.simN,par.T)
        sim.c = np.nan + np.zeros(shape)
        sim.m = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.C = np.nan + np.zeros(shape)
        sim.M = np.nan + np.zeros(shape)
        sim.A = np.nan + np.zeros(shape)
        sim.P = np.nan + np.zeros(shape)
        sim.Y = np.nan + np.zeros(shape)

        sim.util = np.nan + np.zeros(shape)
        sim.mean_lifetime_util = np.nan
        sim.euler = np.nan + np.zeros(shape)
        sim.mean_log10_euler = np.nan
        
        # d. random log-normal mean one income shocks
        self.allocate_draws()

        # e. initialization
        sim.a_init = np.linspace(0.0,par.max_m*0.5,par.simN)
        sim.P_init = np.ones(par.simN)

        # f. pre-computation grids
        par.grid_C = nonlinspace(0.1,par.max_C,par.num_C,par.unequal_C)
        par.grid_marg_U = np.nan + np.zeros(par.num_C)

        par.grid_C_flip = np.nan + np.zeros(par.num_C) 
        par.grid_marg_U_flip = np.nan + np.zeros(par.num_C)    
        
    def allocate_draws(self):
        par = self.par
        sim = self.sim

        np.random.seed(par.seed)
        shape = (par.simN,par.T)
        sim.xi = np.exp(par.sigma_trans*np.random.normal(size=shape) - 0.5*par.sigma_trans**2)
        sim.psi = np.exp(par.sigma_perm*np.random.normal(size=shape) - 0.5*par.sigma_perm**2)

        


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period (consume everything)
        t = par.T-1
        sol.c[t,:] = par.m_grid
        sol.m[t,:] = par.m_grid
        sol.V[t,:] = self.util(sol.c[t,:])

        # c. solve all previous periods
        if par.method == 'vfi':
            self.solve_vfi()
        
        elif par.method == 'egm':
            self.solve_egm()
        
        elif par.method == 'iegm':
            self.precompute_C()
            self.solve_egm()

        else:
            raise Exception('Unknown method')
        
    def solve_egm(self):
        # a. unpack
        par = self.par
        sol = self.sol

        for t in reversed(range(par.T-1)):

            # add credit constraint
            m_interp_next =  np.concatenate((np.array([0.0]),sol.m[t+1]))
            c_interp_next =  np.concatenate((np.array([0.0]),sol.c[t+1]))

            # b. loop over end-of-period wealth
            for ia,assets in enumerate(par.a_grid): # same dimension as m_grid
                
                # i. loop over income shocks to get expected marginal utility
                EmargV_next = 0.0
                for i_xi,xi in enumerate(par.xi_grid):
                    for i_psi,psi in enumerate(par.psi_grid):
                        fac = par.G*psi # normalization factor

                        # interpolate next period value function for this combination of transitory and permanent income shocks
                        m_next = (1.0+par.r)*assets/fac + xi
                        C_next_interp = interp_1d(m_interp_next,c_interp_next,m_next)
                        margV_next_interp = self.marg_util(fac*C_next_interp)

                        # weight the interpolated value with the likelihood
                        EmargV_next += margV_next_interp*par.xi_weight[i_xi]*par.psi_weight[i_psi]
                
                # ii. invert marginal utility to get consumption (interpolate pre-computed consumption if iEGM)
                EmargV_next = par.beta*(1.0+par.r)*EmargV_next
                if par.method=='egm':
                    sol.c[t,ia] = self.inv_marg_util(EmargV_next)
                    
                elif par.method=='iegm':

                    if par.interp_method == 'linear':
                        sol.c[t,ia] = interp_1d(par.grid_marg_U_flip,par.grid_C_flip, EmargV_next)  
                    elif par.interp_method == 'regression':
                        sol.c[t,ia] = regression_interp(EmargV_next,par.interp_coefs)
                    elif par.interp_method == 'Bspline':
                        sol.c[t,ia] = interp_Bspline(EmargV_next,par)
                    elif par.interp_method == 'numerical':
                        sol.c[t,ia] = numerical_inverse(self.marg_util,EmargV_next,par.max_C)
                    else:
                        Warning(f'interpolation method "{par.interp_method}" not implemented!')

                    if par.interp_inverse:
                        sol.c[t,ia] = 1.0/sol.c[t,ia] # inverse consumption has be interpolated

                # iii. endogenous level of resources (value function not stored since not needed here)
                sol.m[t,ia] = assets + sol.c[t,ia]


    def solve_vfi(self):
        # a. unpack
        par = self.par
        sol = self.sol

        for t in reversed(range(par.T-1)):

            # i. loop over state varible: resources in beginning of period
            for im,resources in enumerate(par.m_grid):

                # ii. find optimal consumption at this level of resources in this period t.
                obj = lambda c: - self.value_of_choice(c[0],resources,t)  

                # bounds on consumption
                lb = 0.000001 # avoid dividing with zero
                ub = resources

                # call optimizer
                c_init = sol.c[t+1,im] # initial guess on optimal consumption: last period's optimal consumption
                res = minimize(obj,c_init,bounds=((lb,ub),),method='SLSQP')
                
                # store results
                sol.c[t,im] = res.x[0]
                sol.m[t,im] = resources
                sol.V[t,im] = -res.fun


    def value_of_choice(self,cons,resources,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. utility from consumption
        util = self.util(cons)
        
        # c. expected continuation value from savings
        V_next = sol.V[t+1]
        assets = resources - cons
        
        # loop over income shocks 
        EV_next = 0.0
        for i_xi,xi in enumerate(par.xi_grid):
            for i_psi,psi in enumerate(par.psi_grid):
                fac = par.G*psi # normalization factor

                # interpolate next period value function for this combination of transitory and permanent income shocks
                m_next = (1.0+par.r)*assets/fac + xi
                V_next_interp = interp_1d(par.m_grid,V_next,m_next)
                V_next_interp = (fac**(1.0-par.rho)) * V_next_interp # normalization factor

                # weight the interpolated value with the likelihood
                EV_next += V_next_interp*par.xi_weight[i_xi]*par.psi_weight[i_psi]

        # d. return value of choice
        return util + par.beta*EV_next


    def util(self,c):
        par = self.par
        return (c)**(1.0-par.rho) / (1.0-par.rho)

    def marg_util(self,c):
        par = self.par
        return c**(-par.rho)
    
    def inv_marg_util(self,mu):
        par = self.par
        return mu**(-1.0/par.rho)
    
    def precompute_C(self):
        """ precompute consumption function on a grid """

        # a. unpack
        par = self.par

        # b. loop over consumption grid and store marginal utility
        par.grid_marg_U = self.marg_util(par.grid_C)

        # c. flip grids such that marginal utility is increasing (for interpolation)
        par.grid_marg_U_flip = np.flip(par.grid_marg_U)
        par.grid_C_flip = np.flip(par.grid_C)

        # d. inverse if wanted
        if par.interp_inverse:
            par.grid_C_flip = 1.0/par.grid_C_flip # inverse consumption is interpolated

        # e. regression
        if par.interp_method == 'regression':
            par.interp_coefs = regression_coefs(par.grid_marg_U_flip,par.grid_C_flip,par.interp_degree)
        
        elif par.interp_method == 'Bspline':
            setup_Bspline(par.grid_marg_U_flip,par.grid_C_flip,par)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        # i. initialize permanent income and normalized assets 
        t = 0
        sim.P[:,t] = sim.P_init[:]
        sim.Y[:,t] = sim.P[:,t]*sim.xi[:,t]

        sim.a[:,t] = sim.a_init[:]
        sim.A[:,t] = sim.a[:,t]*sim.P[:,t]
        
        # ii. resources (normalized)
        sim.M[:,t] = (1.0+par.r)*sim.A[:,t] + sim.Y[:,t]
        sim.m[:,t] = sim.M[:,t]/sim.P[:,t]

        for t in range(par.T):
            # add credit constraint
            m_interp =  np.concatenate((np.array([0.0]),sol.m[t]))
            c_interp =  np.concatenate((np.array([0.0]),sol.c[t]))
            
            if t<par.T: # check that simulation does not go further than solution                 

                # iii. interpolate optimal consumption (normalized)
                interp_1d_vec(m_interp,c_interp,sim.m[:,t],sim.c[:,t])

                # iv. Update next-period states
                if t<par.T-1:
                    sim.P[:,t+1] = par.G*sim.P[:,t]*sim.psi[:,t+1]
                    sim.Y[:,t+1] = sim.P[:,t+1]*sim.xi[:,t+1]

                    sim.a[:,t+1] = sim.m[:,t] - sim.c[:,t]
                    sim.A[:,t+1] = sim.a[:,t+1]*sim.P[:,t+1]

                    sim.M[:,t+1] = (1.0+par.r)*sim.A[:,t+1] + sim.Y[:,t+1]
                    sim.m[:,t+1] = sim.M[:,t+1]/sim.P[:,t+1]

                # v. discounted utility
                sim.util[:,t] = (par.beta**t)*self.util(sim.c[:,t])

                # vi. Euler error
                if t<par.T-1:
                    m_interp_next =  np.concatenate((np.array([0.0]),sol.m[t+1]))
                    c_interp_next =  np.concatenate((np.array([0.0]),sol.c[t+1]))

                    I = (sim.a[:,t+1]>=0.001)
                    num = np.int_(np.sum(I))
                    EmargV_next = np.zeros(num)
                    c_next = np.zeros(num)
                    for i_xi,xi in enumerate(par.xi_grid):
                        for i_psi,psi in enumerate(par.psi_grid):
                            fac = par.G*psi # normalization factor

                            # interpolate next period value function for this combination of transitory and permanent income shocks
                            m_next = (1.0+par.r)*sim.a[I,t+1]/fac + xi
                            interp_1d_vec(m_interp_next,c_interp_next,m_next,c_next)
                            margV_next = self.marg_util(fac*c_next)

                            # weight the interpolated value with the likelihood
                            EmargV_next += margV_next*par.xi_weight[i_xi]*par.psi_weight[i_psi]
                    
                    EmargV_next = par.beta*(1.0+par.r)*EmargV_next
                    sim.euler[I,t] = sim.c[I,t] - self.inv_marg_util(EmargV_next)
                    sim.euler[~I,t] = np.nan

        sim.mean_lifetime_util = np.mean(np.sum(sim.util,axis=1))
        sim.mean_log10_euler = np.nanmean(np.log10( abs(sim.euler/sim.c) + 1.0e-16));
            



