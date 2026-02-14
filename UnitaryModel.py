import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from InterpolationFunctions import *

class UnitaryModelClass(EconModelClass):

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
        
        par.rho_w = 1.5 # CRRA coefficient, women
        par.rho_m = 1.5 # CRRA coefficient, men
        
        par.phi_w = 0.5 # private and public consumption complementarity, women
        par.phi_m = 0.5 # private and public consumption complementarity, men
        
        par.alpha = 0.5 # weight on private consumption

        par.mu = 0.5 # weight on women's utility
        
        # income
        par.Yw = 1.0 # income level, women
        par.Ym = 1.0 # income level, men
        
        par.sigma_w = 0.1
        par.sigma_m = 0.1
        
        par.NYw = 5 # number of points in women's income shock expectaion
        par.NYm = 5  # number of points in men's income shock expectaion


        # saving
        par.r = 0.03 # interest rate

        # grid
        par.max_m = 5.0 # maximum point in resource grid
        par.num_m = 50 # number of grid points in resource grid   

        # EGM
        par.max_A_pd = 5.0

        # iEGM
        par.num_C = 30 # number of points in pre-computation grid
        par.max_C = 10.0 # maximum point in pre-computation grid
        par.unequal_C = 1.1

        par.interp_method = 'linear' # numerical, linear, Bspline
        par.interp_inverse = False # True: interpolate inverse consumption
        par.interp_degree = 8
        
        par.precompute_intra = False

        # simulation
        par.seed = 9210
        par.simN = 10_000 # number of consumers simulated

        # solution method
        par.method = 'vfi' # vfi, egm, or iegm.
        par.restricted_model = False
        par.scale_w = 1.0 # updated if restricted model
        par.scale_m = 1.0 # updated if restricted model
        par.scale = 1.0 # updated if restricted model


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.M_grid = nonlinspace(0.0001,par.max_m,par.num_m,1.1) # always have a bit of resources
        par.a_grid = nonlinspace(0.0001,par.max_A_pd,par.num_m,1.1) # always have a bit of resources

        par.Yw_grid,par.Yw_weight = log_normal_gauss_hermite(par.sigma_w,par.NYw)
        par.Ym_grid,par.Ym_weight = log_normal_gauss_hermite(par.sigma_m,par.NYm)
        
        
        # b. solution arrays
        shape = (par.T,par.num_m)
        sol.C = np.nan + np.zeros(shape)
        sol.Cw = np.nan + np.zeros(shape)
        sol.Cm = np.nan + np.zeros(shape)
        sol.Cpub = np.nan + np.zeros(shape)
        sol.M = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # c. simulation arrays
        shape = (par.simN,par.T)
        sim.C = np.nan + np.zeros(shape)
        sim.Cw = np.nan + np.zeros(shape)
        sim.Cm = np.nan + np.zeros(shape)
        sim.Cpub = np.nan + np.zeros(shape)
        sim.M = np.nan + np.zeros(shape)
        sim.A = np.nan + np.zeros(shape)

        sim.util = np.nan + np.zeros(shape)
        sim.mean_lifetime_util = np.nan
        sim.euler = np.nan + np.zeros(shape)
        sim.mean_log10_euler = np.nan
    
        # e. initialization
        sim.a_init = np.linspace(0.0,par.max_m*0.5,par.simN)

        # f. pre-computation grids
        par.grid_C = nonlinspace(0.1,par.max_C,par.num_C,par.unequal_C)
        par.grid_marg_U = np.nan + np.zeros(par.num_C)

        par.grid_C_flip = np.nan + np.zeros(par.num_C) 
        par.grid_marg_U_flip = np.nan + np.zeros(par.num_C)    
        
        par.grid_Cw = np.nan + np.ones(par.num_C)
        par.grid_Cm = np.nan + np.ones(par.num_C)
    
    def restrict_par(self):
        par = self.par
        
        if par.restricted_model:
            par.rho_m = par.rho_w
            par.phi_m = par.phi_w
            
            par.alpha=1.0 # no public consumption
            
            # scales
            _scale = 0.5*(par.mu/(1.0-par.mu))**(1.0/par.rho_w)
            par.scale_w = (_scale)
            par.scale_m = (1.0-_scale)
            par.scale = par.scale_w**(1.0-par.rho_w) * par.mu + par.scale_m**(1.0-par.rho_w) * (1.0-par.mu)
            


    ############
    # Solution #
    def solve(self):
        
        # restrict parameters if restricted model
        self.restrict_par();
        
        # a. unpack
        par = self.par
        sol = self.sol
        
        # precomputations
        if par.method == 'iegm':
            self.precompute_C()
            
        if par.precompute_intra:
            self.precompute_intra()
        
        # b. solve last period (consume everything)
        t = par.T-1
        sol.C[t,:] = par.M_grid
        for iC,Ctot in enumerate(sol.C[t,:]):
            sol.Cw[t,iC], sol.Cm[t,iC], sol.Cpub[t,iC] = self.solve_intratemporal_allocation(Ctot,par.precompute_intra)
            sol.V[t,iC] = self.util(sol.Cw[t,iC],sol.Cm[t,iC],sol.Cpub[t,iC])
        
        sol.M[t,:] = par.M_grid

        # c. solve all previous periods
        if par.method == 'vfi':
            self.solve_vfi()
        
        elif 'egm' in par.method:
            self.solve_egm()
        
        else:
            raise Exception('Unknown method')
        
    def solve_egm(self):
        # a. unpack
        par = self.par
        sol = self.sol

        for t in reversed(range(par.T-1)):

            # add credit constraint
            m_interp_next =  np.concatenate((np.array([0.0]),sol.M[t+1]))
            c_interp_next =  np.concatenate((np.array([0.0]),sol.C[t+1]))

            # b. loop over end-of-period wealth
            for ia,assets in enumerate(par.a_grid): # same dimension as m_grid
                
                # i. interpoalte consumption
                # m_next = self.trans_m(assets)
                # C_next_interp = interp_1d(m_interp_next,c_interp_next,m_next)
                
                # ii. discounted marginal value of wealth, W
                EmargV_next = 0.0
                for i_Yw,Yw in enumerate(par.Yw_grid):
                    for i_Ym,Ym in enumerate(par.Ym_grid):
                        
                        # interpolate next period value function for this combination of transitory and permanent income shocks
                        m_next = self.trans_m(assets,Yw,Ym)
                        C_next_interp = interp_1d(m_interp_next,c_interp_next,m_next)
                        margV_next_interp = self.marg_HH_util(C_next_interp)

                        # weight the interpolated value with the likelihood
                        EmargV_next += margV_next_interp*par.Yw_weight[i_Yw]*par.Ym_weight[i_Ym]
                
                EmargV_next = par.beta*(1.0+par.r)*EmargV_next

                if par.method=='egm':
                    assert par.restricted_model , f'EGM is only applicable in the restricted model!'
                    sol.C[t,ia] = self.inv_marg_HH_util(EmargV_next)
                    
                elif par.method=='iegm':

                    if par.interp_method == 'linear':
                        sol.C[t,ia] = interp_1d(par.grid_marg_U_flip,par.grid_C_flip, EmargV_next)  
                    elif par.interp_method == 'regression':
                        sol.C[t,ia] = regression_interp(EmargV_next,par.interp_coefs)
                    elif par.interp_method == 'Bspline':
                        sol.C[t,ia] = interp_Bspline(EmargV_next,par)
                    elif par.interp_method == 'numerical':
                        sol.C[t,ia] = numerical_inverse(self.marg_HH_util,EmargV_next,par.max_C)
                    else:
                        Warning(f'interpolation method "{par.interp_method}" not implemented!')

                    if par.interp_inverse:
                        sol.C[t,ia] = 1.0/sol.C[t,ia] # inverse consumption has be interpolated

                # iii. endogenous level of resources (value function not stored since not needed here)
                sol.M[t,ia] = assets + sol.C[t,ia]
                
                # iv. intra-temporal allocation
                sol.Cw[t,ia], sol.Cm[t,ia], sol.Cpub[t,ia] = self.solve_intratemporal_allocation(sol.C[t,ia],par.precompute_intra)


    def solve_vfi(self):
        # a. unpack
        par = self.par
        sol = self.sol

        for t in reversed(range(par.T-1)):

            # i. loop over state varible: resources in beginning of period
            for im,resources in enumerate(par.M_grid):

                # ii. find optimal consumption at this level of resources in this period t.
                obj = lambda Ctot: - self.intertemporal_value_of_choice(Ctot[0],resources,t)  

                # bounds on consumption
                lb = 0.0000001 # avoid dividing with zero
                ub = resources

                # call optimizer
                c_init = sol.C[t+1,im] # initial guess on optimal consumption: last period's optimal consumption
                res = minimize(obj,c_init,bounds=((lb,ub),),method='SLSQP')
                
                # store results
                sol.C[t,im] = res.x[0]
                sol.Cw[t,im], sol.Cm[t,im], sol.Cpub[t,im] = self.solve_intratemporal_allocation(sol.C[t,im],par.precompute_intra)
                
                sol.M[t,im] = resources
                sol.V[t,im] = -res.fun


    def intertemporal_value_of_choice(self,Ctot,resources,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. household utility from consumption
        util = self.HH_util(Ctot)
        
        # c. expected continuation value from savings
        V_next = sol.V[t+1]
        assets = resources - Ctot
        
        EV_next = 0.0
        for i_Yw,Yw in enumerate(par.Yw_grid):
            for i_Ym,Ym in enumerate(par.Ym_grid):
                m_next = self.trans_m(assets,Yw,Ym)
                V_next_interp = interp_1d(par.M_grid,V_next,m_next)
                
                EV_next += V_next_interp*par.Yw_weight[i_Yw]*par.Ym_weight[i_Ym]
                
        # d. return value of choice
        return util + par.beta*EV_next

    def solve_intratemporal_allocation(self,Ctot,precomputed):
        par = self.par
        
        if par.restricted_model: 
            # closed form 
            Cw = par.scale_w * Ctot
            Cm = Ctot - Cw
            Cpub = 0.0
            
        # elif par.restricted_model: # Cpub always zero
        #     if precomputed:
        #         Cw = interp_1d(par.grid_C,par.grid_Cw,Ctot)       
            
        #     else:
        #         obj = lambda Cvec: - self.util(Cvec[0],Ctot-Cvec[0],0.0)
                
        #         lb = 0.00001
        #         ub = Ctot
                
        #         init = np.array([0.5*Ctot])
        #         res = minimize(obj,init,bounds=((lb,ub),),method='SLSQP')
                
        #         Cw = res.x[0]
            
        #     Cm = Ctot - Cw
        #     Cpub = 0.0
            
        else:
            if precomputed:
                Cw = interp_1d(par.grid_C,par.grid_Cw,Ctot) 
                Cm = interp_1d(par.grid_C,par.grid_Cm,Ctot)       
            
            else:
                def obj(Cshare):
                    Cw = Ctot*Cshare[0]
                    Cm = (Ctot-Cw)*Cshare[1]
                    Cpub = Ctot-Cw-Cm
                    
                    return - self.util(Cw,Cm,Cpub)
                
                # find shares, ensuring total consumption 
                lb = 0.0001
                ub = 0.99999
                bounds=((lb,ub),(lb,ub))
                
                init = np.array([0.3,0.3])
                res = minimize(obj,init,bounds=bounds,method='SLSQP')
                
                Cw = Ctot*res.x[0]
                Cm = (Ctot-Cw)*res.x[1]
            
            Cpub = Ctot - Cw - Cm
             
        return Cw,Cm,Cpub

    def util_i(self,Cpriv,Cpub,sex):
        par = self.par
        
        rho = par.rho_w if sex=='woman' else par.rho_m
        phi = par.phi_w if sex=='woman' else par.phi_m
        
        CES = (par.alpha*Cpriv**phi + (1.0-par.alpha)*Cpub**phi)**(1.0/phi)
        
        return (CES)**(1.0-rho) / (1.0-rho)
    
    def util(self,Cw,Cm,Cpub,verbose=True):
        par = self.par
        
        # ensure that all consumption elements are non-negative
        penalty = 0.0
        if Cw<0.0:
            penalty += Cw**2 * 1_000.0
            Cw = 0.00001
        if Cm<0.0:
            penalty += Cm**2 * 1_000.0
            Cm = 0.00001
        if Cpub<0.0:
            penalty += Cpub**2 * 1_000.0
            Cpub = 0.00001
            
        if verbose & (penalty>0.0): print(f'WARNING: consumption components negative. \nMight not be a problem for final allocation.')
        
        util = par.mu*self.util_i(Cw,Cpub,'woman') + (1.0-par.mu)*self.util_i(Cm,Cpub,'man')
        
        return util + penalty

    
    def HH_util(self,Ctot):
        par = self.par
        
        # solve intratemporal problem
        Cw,Cm,Cpub = self.solve_intratemporal_allocation(Ctot,par.precompute_intra)
        
        # return household level utility for this allocation
        return self.util(Cw,Cm,Cpub)
        
    
    def marg_HH_util(self,Ctot):
        par = self.par
        
        if par.restricted_model: # then marginal utility is known in closed form
            rho = par.rho_w 
            return par.scale*Ctot**(-rho)
        
        else:
            # something is weird with these interpolations.. using this code messes with another interpolation..
            # if ((par.method=='iegm') & (par.interp_method!='numerical')):
            #     # interpolate pre-computed marginal utility
            #     return interp_1d(par.grid_C,par.grid_marg_U,Ctot)
            
            # else:
                #numerical gradient of HH_util
            step = 1.0e-4
            forward = self.HH_util(Ctot+step)
            backward = self.HH_util(Ctot-step)
            
            return (forward - backward)/(2*step)
            
    
    def inv_marg_HH_util(self,W):
        '''This function returns the inversa marginal utility. 
        Only works for the restricted model'''
        par = self.par
        rho = par.rho_w
        scale = par.scale**(1.0/rho)
        return scale*W**(-1.0/rho)
    
    def trans_m(self,assets,Yw,Ym):
        par = self.par
        return (1+par.r)*assets + par.Yw*Yw + par.Ym*Ym
    
    def precompute_C(self):
        """ precompute consumption function on a grid """

        # a. unpack
        par = self.par
        
        precompute_intra = par.precompute_intra
        par.precompute_intra = False

        # b. loop over consumption grid and store marginal utility
        if par.restricted_model:
            par.grid_marg_U = self.marg_HH_util(par.grid_C)
        else:
            for iC,C in enumerate(par.grid_C):
                par.grid_marg_U[iC] = self.marg_HH_util(C)

        # c. flip grids such that marginal utility is increasing (for interpolation)
        par.grid_marg_U_flip = np.flip(par.grid_marg_U.copy())
        par.grid_C_flip = np.flip(par.grid_C.copy())

        # d. inverse if wanted
        if par.interp_inverse:
            par.grid_C_flip = 1.0/par.grid_C_flip # inverse consumption is interpolated

        # e. regression
        if par.interp_method == 'regression':
            par.interp_coefs = regression_coefs(par.grid_marg_U_flip,par.grid_C_flip,par.interp_degree)
        
        elif par.interp_method == 'Bspline':
            setup_Bspline(par.grid_marg_U_flip,par.grid_C_flip,par)
        
        par.precompute_intra = precompute_intra
    
    def precompute_intra(self):
        par = self.par
        
        # precompute intra-temporal allocation
        if par.restricted_model:
            par.grid_Cw,par.grid_Cm,_ = self.solve_intratemporal_allocation(par.grid_C,precomputed=False)
        else:
            for iC,C in enumerate(par.grid_C):
                par.grid_Cw[iC],par.grid_Cm[iC],_ = self.solve_intratemporal_allocation(C,precomputed=False)
        

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        # i. assets 
        t = 0

        sim.A[:,t] = sim.a_init[:]
        
        # ii. resources
        sim.M[:,t] = self.trans_m(sim.A[:,t])

        for t in range(par.T):
            # add credit constraint
            m_interp =  np.concatenate((np.array([0.0]),sol.M[t]))
            c_interp =  np.concatenate((np.array([0.0]),sol.C[t]))
            
            if t<par.T: # check that simulation does not go further than solution                 

                # iii. interpolate optimal total consumption
                interp_1d_vec(m_interp,c_interp,sim.m[:,t],sim.C[:,t])
                
                # iv. intra-temporal allocation (not really needed for our purpose)

                # v. Update next-period states
                if t<par.T-1:

                    sim.A[:,t+1] = sim.M[:,t] - sim.C[:,t]
                    sim.M[:,t+1] = self.trans_m(sim.A[:,t+1])
            



