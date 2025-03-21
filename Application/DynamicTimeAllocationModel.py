import numpy as np
import numba as nb
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d
from consav import quadrature
import scipy.stats as stats

import time

# set gender indication as globals
woman = 1
man = 2

class HouseholdModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.beta = 1.0/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife
        par.div_cost = 0.0

        par.Day = 1.0 # time in day
        
        # a. income
        par.mu_w = 0.5              # level
        par.gamma_w = 0.1           # return to human capital

        par.mu_m = 0.5              # level
        par.gamma_m = 0.1           # return to human capital   

        # a.1. human capital
        par.delta = 0.05                   # depreciation
        par.sigma_epsilon_w = 0.08         # std of shock to human capital
        par.sigma_epsilon_m = 0.08         # std of shock to human capital


        # b. utility: gender-specific parameters
        par.rho_w = 2.0        # CRRA
        par.rho_m = 2.0        # CRRA
        
        par.phi_w = 0.05        # weight on labor supply
        par.phi_m = 0.05        # weight on labor supply  

        par.eta_w = 0.5         # curvature on labor supply
        par.eta_m = 0.5         # curvature on labor supply

        par.lambda_w = 0.5      # weight on public good
        par.lambda_m = 0.5      # weight on public good

        # c. Home production
        par.alpha = 1.0         # output elasticity of hw relative to hm in housework aggregator
        par.zeta = 0.5             # Substitution paremeter between hw and hm in housework aggregator
        par.omega = 0.5         # weight on market purchased good in home production function      
        

        # c. state variables
        par.T = 10
        
        # c.1 wealth
        par.num_A = 50
        par.max_A = 15.0

        # c.2. human capital
        par.num_K = 50
        par.max_K = par.T*1.5
        
        # c.3 bargaining power
        par.num_power = 21

        # c.4 love/match quality
        par.num_love = 41
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5

        # d. re-partnering
        par.p_meet = 0.1
        par.prob_partner_A_w = np.array([[np.nan]]) # if not set here, defaults to np.eye(par.num_A) in setup_grids
        par.prob_partner_A_m = np.array([[np.nan]])

        # e. discrete choices
        par.grid_l = np.array([0.0, 0.5, 0.75])
        par.num_l = len(par.grid_l)


        # f. pre-computation
        par.interp_inverse = False # True: interpolate inverse consumption
        par.precompute_intratemporal = True # if True, precompute intratemporal allocation, else re-solve every time

        # f.1. intratemporal precomputation
        par.num_Ctot = 20
        par.max_Ctot = par.max_A*2
        
        # f.2. intertemporal precomputation (iEGM)
        par.do_egm = False
        par.num_A_pd = par.num_A * 2
        par.max_A_pd = par.max_A
        par.num_marg_u = 20

        # g. simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50

        # h. misc
        par.threads = 8
        par.interp_method = 'linear'
        par.centered_gradient = True

        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        self.setup_grids()
        
        # a. singles
        shape_single = (par.T,par.num_A)                        # single states: T, human capital, assets

        # a.1. single to single
        sol.Vw_single_to_single = np.ones(shape_single) + np.nan                # value
        sol.Vm_single_to_single = np.ones(shape_single) + np.nan

        sol.Cw_priv_single_to_single = np.ones(shape_single) + np.nan           # private consumption, single
        sol.Cm_priv_single_to_single = np.ones(shape_single) + np.nan
        sol.Cw_inter_single_to_single = np.ones(shape_single) + np.nan            # intermediate good, single
        sol.Cm_inter_single_to_single = np.ones(shape_single) + np.nan
        sol.Qw_single_to_single = np.ones(shape_single) + np.nan                # home produced good, single
        sol.Qm_single_to_single = np.ones(shape_single) + np.nan
        sol.lw_single_to_single = np.ones(shape_single, dtype = int) + np.nan   # labor supply, single
        sol.lm_single_to_single = np.ones(shape_single, dtype = int) + np.nan
        sol.hw_single_to_single = np.ones(shape_single) + np.nan                # housework, single
        sol.hm_single_to_single = np.ones(shape_single) + np.nan

        ### a.1.1. post-decision grids (EGM)
        # TBD: nedenstående udkommenterede kode er den gamle version
        # sol.EmargUw_single_to_single_pd = np.zeros(shape_single)           # Expected marginal utility post-decision, woman single
        # sol.C_totw_single_to_single_pd = np.zeros(par.num_A_pd)            # C for EGM, woman single 
        # sol.Mw_single_to_single_pd = np.zeros(par.num_A_pd)                # Endogenous grid, woman single
        # sol.Vw_single_to_single_pd = np.zeros(par.num_A_pd)                # Value of being single, post-decision

        # sol.EmargUm_single_to_single_pd = np.zeros(shape_single)          # Expected marginal utility post-decision, man single
        # sol.C_totm_single_to_single_pd = np.zeros(par.num_A_pd)           # C for EGM, man single
        # sol.Mm_single_to_single_pd = np.zeros(par.num_A_pd)               # Endogenous grid, man single
        # sol.Vm_single_to_single_pd = np.zeros(par.num_A_pd)               # Value of being single, post-decision

        ## a.2. couple to single
        sol.Vw_couple_to_single = np.nan + np.ones(shape_single)        # Value marriage -> single
        sol.Vm_couple_to_single = np.nan + np.ones(shape_single)

        sol.Cw_priv_couple_to_single = np.nan + np.ones(shape_single)   # Private consumption marriage -> single
        sol.Cm_priv_couple_to_single = np.nan + np.ones(shape_single)
        sol.Cw_inter_couple_to_single = np.nan + np.ones(shape_single)    # intermediate consumption marriage -> single 
        sol.Cm_inter_couple_to_single = np.nan + np.ones(shape_single)    # Not used
        sol.Cw_tot_couple_to_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_couple_to_single = np.nan + np.ones(shape_single)

        ## a.3. start as single
        # TBD: nedenstående udkommenterede kode er den gamle version
        # sol.EVw_start_as_single = np.nan + np.ones(shape_single)
        # sol.EVm_start_as_single = np.nan + np.ones(shape_single)  
        # sol.EmargVw_start_as_single = np.nan + np.ones(shape_single)
        # sol.EmargVm_start_as_single = np.nan + np.ones(shape_single)  

        # sol.EVw_cond_meet_partner = np.nan + np.ones(shape_single)
        # sol.EVm_cond_meet_partner = np.nan + np.ones(shape_single)


        # b. couples
        shape_couple = (par.T, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
            # couple states: T, power, love, human capital w, human capital w, assets

        # b.1. couple to couple
        sol.Vw_couple_to_couple = np.ones(shape_couple) + np.nan                # value
        sol.Vm_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.V_couple_to_couple = np.ones(shape_couple) + np.nan                 # couple objective function

        sol.Cw_priv_couple_to_couple = np.ones(shape_couple) + np.nan           # private consumption, couple
        sol.Cm_priv_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.C_inter_couple_to_couple = np.ones(shape_couple) + np.nan             # intermediate good, couple
        sol.Q_couple_to_couple = np.ones(shape_couple) + np.nan                 # home produced good, couple
        sol.lw_couple_to_couple = np.ones(shape_couple, dtype = int) + np.nan   # labor supply, couple
        sol.lm_couple_to_couple = np.ones(shape_couple, dtype = int) + np.nan
        sol.hw_couple_to_couple = np.ones(shape_couple) + np.nan                # housework, couple
        sol.hm_couple_to_couple = np.ones(shape_couple) + np.nan

        sol.Sw = np.ones(par.num_power) + np.nan                                 # surplus of marriage
        sol.Sm = np.ones(par.num_power) + np.nan

        sol.power_idx = np.zeros(shape_couple, dtype = int)                     # index of bargaining weight (approx)
        sol.power = np.zeros(shape_couple) + np.nan                             # bargainng weight (interpolated)

        ### b.1.1. post-decision grids (EGM)
        # TBD: nedenstående udkommenterede kode er den gamle version
        # shape_egm = (par.T, par.num_power,par.num_love,par.num_A_pd)
        # sol.EmargU_pd = np.zeros(shape_egm)                     # Expected marginal utility post-decision
        # sol.C_tot_pd = np.zeros(shape_egm)                      # C for EGM
        # sol.M_pd = np.zeros(shape_egm)                          # Endogenous grid
        # sol.V_couple_to_couple_pd = np.zeros(shape_egm)         # Value of being couple, post-decision

        ## b.2. single to couple
        # TBD: nedensteånede udkommenterede kode er den gamle version
        # sol.Vw_single_to_couple = np.nan + np.ones(shape_couple)           # value single -> marriage
        # sol.Vm_single_to_couple = np.nan + np.ones(shape_couple)
        # sol.V_single_to_couple = -np.inf + np.ones(shape_couple)           
                                                                        
        # sol.Cw_priv_single_to_couple = np.nan + np.ones(shape_couple)      
        # sol.Cm_priv_single_to_couple = np.nan + np.ones(shape_couple)      
        # sol.C_inter_single_to_couple = np.nan + np.ones(shape_couple)        
        # sol.Cw_tot_single_to_couple = np.nan + np.ones(shape_couple)   
        # sol.Cm_tot_single_to_couple = np.nan + np.ones(shape_couple) 
  
        # shape_power =(par.T,par.num_love,par.num_A,par.num_A)          
        # sol.initial_power = np.nan + np.zeros(shape_power)
        # sol.initial_power_idx = np.zeros(shape_power,dtype=np.int_)

        ## b.3. start as couple
        sol.Vw_start_as_couple = np.ones(shape_couple) + np.nan                 # value
        sol.Vm_start_as_couple = np.ones(shape_couple) + np.nan
        sol.margV_start_as_couple = np.ones(shape_couple) + np.nan              # marginal value

        sol.EVw_start_as_couple = np.ones(shape_couple) + np.nan                # expected value
        sol.EVm_start_as_couple = np.ones(shape_couple) + np.nan
        sol.EmargV_start_as_couple = np.ones(shape_couple) + np.nan             # expected marginal value

        sol.Cw_priv_start_as_couple = np.ones(shape_couple) + np.nan            # private consumption
        sol.Cm_priv_start_as_couple = np.ones(shape_couple) + np.nan
        sol.C_inter_start_as_couple = np.ones(shape_couple) + np.nan              # intermediate good
        sol.Q_start_as_couple = np.ones(shape_couple) + np.nan                  # home produced good
        sol.lw_start_as_couple = np.ones(shape_couple, dtype = int) + np.nan    # labor supply
        sol.lm_start_as_couple = np.ones(shape_couple, dtype = int) + np.nan
        sol.hw_start_as_couple = np.ones(shape_couple) + np.nan                 # housework
        sol.hm_start_as_couple = np.ones(shape_couple) + np.nan
        

        # c. Precomputed intratemporal solution
        # c.1. couple
        shape_pre = (par.num_l, par.num_l, par.num_Ctot, par.num_power)
        sol.pre_Cw_priv_couple = np.ones(shape_pre) + np.nan
        sol.pre_Cm_priv_couple = np.ones(shape_pre) + np.nan
        sol.pre_C_inter_couple = np.ones(shape_pre) + np.nan
        sol.pre_Q_couple = np.ones(shape_pre) + np.nan
        sol.pre_hw_couple = np.ones(shape_pre) + np.nan
        sol.pre_hm_couple = np.ones(shape_pre) + np.nan

        # c.2. single
        shape_pre_single = (par.num_l, par.num_Ctot)
        sol.pre_Cw_priv_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cm_priv_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cw_inter_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cm_inter_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Qw_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Qm_single = np.ones(shape_pre_single) + np.nan
        sol.pre_hw_single = np.ones(shape_pre_single) + np.nan
        sol.pre_hm_single = np.ones(shape_pre_single) + np.nan


        # d. Precomputed intertemporal solution
        # d.1. couple
        shape_pre_couple = (par.num_l, par.num_l, par.num_marg_u, par.num_power)
        par.grid_marg_u_couple = np.ones(shape_pre_couple) + np.nan
        par.grid_marg_u_couple_for_inv = np.ones(shape_pre_couple) + np.nan

        # d.2. single
        shape_pre_single = (par.num_l, par.num_marg_u)
        par.grid_marg_u_single_w = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_m = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_w_for_inv = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_m_for_inv = np.ones(shape_pre_single) + np.nan

        # d.3 Common
        par.grid_C_for_marg_u = nonlinspace(1.0e-5,par.max_Ctot,par.num_marg_u,1.1)    # Consumption interpolator grid
    

        # d. simulation
        # TBD: nedenstående udkommenterede kode er den gamle version
        # # NB: all arrays not containing "init" or "draw" in name are wiped before each simulation
        # shape_sim = (par.simN,par.simT)
        # sim.Cw_priv = np.nan + np.ones(shape_sim)               
        # sim.Cm_priv = np.nan + np.ones(shape_sim)
        # sim.Cw_pub = np.nan + np.ones(shape_sim)
        # sim.Cm_pub = np.nan + np.ones(shape_sim)
        # sim.Cw_tot = np.nan + np.ones(shape_sim)
        # sim.Cm_tot = np.nan + np.ones(shape_sim)
        # sim.C_tot = np.nan + np.ones(shape_sim)
        
        # sim.A = np.nan + np.ones(shape_sim)
        # sim.Aw = np.nan + np.ones(shape_sim)
        # sim.Am = np.nan + np.ones(shape_sim)
        # sim.couple = np.nan + np.ones(shape_sim)
        # sim.power = np.nan + np.ones(shape_sim)
        # sim.love = np.nan + np.ones(shape_sim)

        # # lifetime utility
        # sim.util = np.nan + np.ones((par.simN, par.simT))
        # sim.mean_lifetime_util = np.array([np.nan])

        # # containers for verifying simulaton
        # sim.A_own = np.nan + np.ones(shape_sim)
        # sim.A_partner = np.nan + np.ones(shape_sim)

        # ## d.2. shocks
        # self.allocate_draws()

        # ## d.3. initial distribution
        # sim.init_A = np.linspace(0.0,par.max_A*0.5,par.simN) 
        # sim.init_Aw = sim.init_A * par.div_A_share
        # sim.init_Am = sim.init_A * (1.0 - par.div_A_share)
        # sim.init_couple = np.ones(par.simN,dtype=np.bool_)
        # sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        # sim.init_love = np.zeros(par.simN)
        
        # e. timing
        sol.solution_time = np.array([0.0])

    # def allocate_draws(self):
    #     # GAMMEL VERSION
    #     par = self.par
    #     sim = self.sim
    #     shape_sim = (par.simN,par.simT)

    #     np.random.seed(par.seed)
    #     sim.draw_love = np.random.normal(size=shape_sim)
    #     sim.draw_meet = np.random.uniform(size=shape_sim) # for meeting a partner

    #     sim.draw_uniform_partner_Aw = np.random.uniform(size=shape_sim) # for inverse cdf transformation of partner wealth
    #     sim.draw_uniform_partner_Am = np.random.uniform(size=shape_sim) # for inverse cdf tranformation of partner wealth

    #     sim.draw_repartner_love = par.sigma_love*np.random.normal(0.0,1.0,size=shape_sim) #np.random.choice(par.num_love, p=par.prob_partner_love, size=shape_sim) # Love index when repartnering

        
    def setup_grids(self):
        par = self.par
        
        # a. state variables
        # a.1. wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)       # asset grid

        par.grid_Aw = par.div_A_share * par.grid_A                  # asset grid in case of divorce
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # a.2. human capital
        par.grid_Kw = nonlinspace(0.0, par.max_K, par.num_K, 1.1)
        par.grid_Km = nonlinspace(0.0, par.max_K, par.num_K, 1.1)

        # a.3 power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(1e-6,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(1e-6,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)
        par.grid_power_flip = np.flip(par.grid_power) # flip for men

        # a.4 love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)
        else:
            par.grid_love = np.array([0.0])

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # b. Precomputation
        # b.1. Intratemporal
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)   

        # EGM
        # TBD: nedenstående udkommenterede kode er den gamle version
        # par.grid_marg_u = np.nan + np.ones((par.num_power,par.num_marg_u))
        # par.grid_marg_u_for_inv = np.nan + np.ones((par.num_power,par.num_marg_u))

        # par.grid_C_for_marg_u = nonlinspace(1.0e-6,par.max_Ctot,par.num_marg_u,1.1)

        # par.grid_inv_marg_u = np.flip(par.grid_C_for_marg_u) # Flipped to make interpolation possible ## AMO: invert
        # if par.interp_inverse:
        #     par.grid_inv_marg_u = 1.0/par.grid_inv_marg_u

        # par.grid_marg_u_single_w = np.nan + np.ones((par.num_marg_u))
        # par.grid_marg_u_single_w_for_inv = np.nan + np.ones((par.num_marg_u))

        # par.grid_marg_u_single_m = np.nan + np.ones((par.num_marg_u))
        # par.grid_marg_u_single_m_for_inv = np.nan + np.ones((par.num_marg_u))


        # re-partering probabilities
        # TBD: nedenstående udkommenterede kode er den gamle version
        # par.prob_repartner = par.p_meet*np.ones(par.T) # likelihood of meeting a partner

        # if np.isnan(par.prob_partner_A_w[0,0]):
        #     par.prob_partner_A_w = np.eye(par.num_A) #np.ones((par.num_A,par.num_A))/par.num_A # likelihood of meeting a partner with a particular level of wealth, conditional on own
    
        # if np.isnan(par.prob_partner_A_m[0,0]):
        #     par.prob_partner_A_m = np.eye(par.num_A) #np.ones((par.num_A,par.num_A))/par.num_A # likelihood of meeting a partner with a particular level of wealth, conditional on own
       
        # # Norm distributed initial love - note: Probability mass between points (approximation of continuous distribution)
        # if par.sigma_love<=1.0e-6:
        #     love_cdf = np.where(par.grid_love>=0.0,1.0,0.0)
        # else:
        #     love_cdf = stats.norm.cdf(par.grid_love,0.0,par.sigma_love)
        # par.prob_partner_love = np.diff(love_cdf,1)
        # par.prob_partner_love = np.append(par.prob_partner_love,0.0) # lost last point in diff
        # # par.prob_partner_love = np.ones(par.num_love)/par.num_love # uniform

        # par.cdf_partner_Aw = np.cumsum(par.prob_partner_A_w,axis=1) # cumulative distribution to be used in simulation
        # par.cdf_partner_Am = np.cumsum(par.prob_partner_A_m,axis=1)




    def solve(self):

        sol = self.sol
        par = self.par 

        # re-allocate to ensure new solution
        # TODO: find alternative to re-allocate every time model is solved
        self.allocate()

        self.cpp.solve(sol,par)


    # def simulate(self):
    #     # GAMMEL VERSION
    #     sol = self.sol
    #     sim = self.sim
    #     par = self.par

    #     # clear simulation
    #     for key, val in sim.__dict__.items():
    #         if 'init' in key or 'draw' in key: continue
    #         setattr(sim, key, np.zeros(val.shape)+np.nan)

    #     self.cpp.simulate(sim,sol,par)

    #     sim.mean_lifetime_util[0] = np.mean(np.sum(sim.util,axis=1))

        

        