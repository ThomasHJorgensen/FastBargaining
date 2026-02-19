import numpy as np
import numba as nb
import scipy.optimize as optimize
import polars as pl
from collections import OrderedDict


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

        par.available_hours = 1.0 #7.0*16.0 # assuming 5*16*52=4160 annual hours)   
        par.tax_rate = 0.25 # flat tax rate on labor income     
        
        # a. income
        # par.inc_w = 1.0
        par.mu = 0.5              # level
        par.sigma_mu = 0.1        # std of mu over types
        par.gamma = 0.1           # return to human capital
        par.gamma2 = 0.00         # quadratic return to human capital

        # par.inc_m = 1.0
        par.mu_mult = 1.0              # level
        par.sigma_mu_mult = 1.0        # std of mu over types
        par.gamma_mult = 1.0           # return to human capital   
        par.gamma2_mult = 1.0           # return to human capital   

        # a.1. human capital
        par.delta = -1.0 # depreciation  
        par.phi_k = 0.4 # accumulation efficiency
        par.sigma_epsilon = 0.08         # std of shock to human capital (women baseline)
        par.sigma_epsilon_mult = 1.0


        # b. utility: gender-specific parameters
        par.rho = 2.0        # CRRA (women baseline)
        par.rho_mult = 1.0

        par.phi = 0.05        # weight on labor supply (women baseline)
        par.phi_mult = 1.0

        par.eta = 0.5         # curvature on labor supply (women baseline)
        par.eta_mult = 1.0

        par.lambda_ = 0.5      # weight on public good (women baseline)
        par.lambda_mult = 1.0

        # c. Home production
        par.alpha = 1.0         # output elasticity of hw relative to hm in housework aggregator
        par.zeta = 0.5             # Substitution paremeter between hw and hm in housework aggregator
        par.omega = 0.5         # substitution between home produced good and market purchased good
        par.pi = 0.5            # weight on marked purchased goods in home production

        # c. state variables
        par.T = 10
        
        # c.1 wealth
        par.num_A = 50
        par.max_A = 15.0

        # c.2. human capital
        par.num_K = 16
        par.max_K = par.T*1.5
        
        par.num_shock_K = 5
        par.sigma_K = 0.1
        par.sigma_K_mult = 1.0
        
        # c.3 bargaining power
        par.num_power = 21

        # c.4 love/match quality
        par.num_love = 11
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5 #can not be 1, because of interpolation things in code

        # d. re-partnering
        par.p_meet = 0.0
        par.prob_partner_Kw = np.array([[np.nan]]) # if not set here, defaults to np.eye(par.num_A) in setup_grids
        par.prob_partner_Km = np.array([[np.nan]])
        par.prob_partner_A_w = np.array([[np.nan]]) # if not set here, defaults to np.eye(par.num_A) in setup_grids
        par.prob_partner_A_m = np.array([[np.nan]])
        par.prob_partner_type_w = np.array([[np.nan]]) # if not set here, defaults to np.eye(par.num_types) in setup_grids
        par.prob_partner_type_m = np.array([[np.nan]])

        # e. discrete choices
        # par.grid_l = np.array([0.00, 0.4, 0.6])
        par.full_time_hours = 0.35


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
        par.num_marg_u = 200

        # g. simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50
        par.init_couple_share = 1.0

        # h. misc
        par.threads = 8
        par.num_multistart = 1
        par.interp_method = 'linear'
        par.centered_gradient = True
        
        # types
        par.num_types = 3 # number of types

        
    def setup_gender_parameters(self):
        par = self.par
        
        # set
        par.mu_w = par.mu
        par.mu_m = par.mu * par.mu_mult
        
        par.gamma_w = par.gamma
        par.gamma_m = par.gamma * par.gamma_mult
        
        par.gamma2_w = par.gamma2
        par.gamma2_m = par.gamma2 * par.gamma2_mult
        
        # other scalar gendered parameters derived from baseline + multiplier
        par.sigma_epsilon_w = par.sigma_epsilon
        par.sigma_epsilon_m = par.sigma_epsilon * par.sigma_epsilon_mult

        par.rho_w = par.rho
        par.rho_m = par.rho * par.rho_mult

        par.phi_w = par.phi
        par.phi_m = par.phi * par.phi_mult

        par.eta_w = par.eta
        par.eta_m = par.eta * par.eta_mult

        # keep python keyword safe name `lambda_` as baseline, derive `lambda_w`/`lambda_m`
        par.lambda_w = par.lambda_
        par.lambda_m = par.lambda_ * par.lambda_mult

        par.sigma_Kw = par.sigma_K
        par.sigma_Km = par.sigma_K * par.sigma_K_mult
        

        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # derive gender-specific parameters
        self.setup_gender_parameters()

        # setup grids
        self.setup_grids()
        
        # a. singles
        shape_single = (par.T, par.num_types, par.num_K, par.num_A)                        # single states: T, human capital, assets
        shape_single_d = (par.T, par.num_l, par.num_types, par.num_K, par.num_A)                        # single states: T, human capital, assets
        shape_single_egm = (par.T, par.num_l, par.num_types, par.num_K, par.num_A_pd)

        # a.1. single to single
        sol.Vwd_single_to_single = np.ones(shape_single_d) - np.inf                
        sol.Vmd_single_to_single = np.ones(shape_single_d) - np.inf

        sol.Cwd_tot_single_to_single = np.ones(shape_single_d) + np.nan           # private consumption, single
        sol.Cmd_tot_single_to_single = np.ones(shape_single_d) + np.nan
        sol.Cwd_priv_single_to_single = np.ones(shape_single_d) + np.nan           # private consumption, single
        sol.Cmd_priv_single_to_single = np.ones(shape_single_d) + np.nan
        sol.Cwd_inter_single_to_single = np.ones(shape_single_d) + np.nan            # intermediate good, single
        sol.Cmd_inter_single_to_single = np.ones(shape_single_d) + np.nan
        sol.Qwd_single_to_single = np.ones(shape_single_d) + np.nan                # home produced good, single
        sol.Qmd_single_to_single = np.ones(shape_single_d) + np.nan
        sol.lwd_single_to_single = np.ones(shape_single_d) + np.nan   # labor supply, single
        sol.lmd_single_to_single = np.ones(shape_single_d) + np.nan
        sol.hwd_single_to_single = np.ones(shape_single_d) + np.nan                # housework, single
        sol.hmd_single_to_single = np.ones(shape_single_d) + np.nan

        ### a.1.1. post-decision grids (EGM)
        sol.EmargUwd_single_to_single_pd = np.zeros(shape_single_egm)           # Expected marginal utility post-decision, woman single
        sol.Cwd_tot_single_to_single_pd = np.zeros(shape_single_egm)            # C for EGM, woman single 
        sol.Mwd_single_to_single_pd = np.zeros(shape_single_egm)                # Endogenous grid, woman single
        sol.Vwd_single_to_single_pd = np.zeros(shape_single_egm)                # Value of being single, post-decision

        sol.EmargUmd_single_to_single_pd = np.zeros(shape_single_egm)          # Expected marginal utility post-decision, man single
        sol.Cmd_totm_single_to_single_pd = np.zeros(shape_single_egm)           # C for EGM, man single
        sol.Mmd_single_to_single_pd = np.zeros(shape_single_egm)               # Endogenous grid, man single
        sol.Vmd_single_to_single_pd = np.zeros(shape_single_egm)               # Value of being single, post-decision

        ## a.2. couple to single
        sol.Vw_couple_to_single = np.nan + np.ones(shape_single)        # Value marriage -> single
        sol.Vm_couple_to_single = np.nan + np.ones(shape_single)
        sol.lw_couple_to_single = np.nan + np.ones(shape_single)   # labor supply marriage -> single
        sol.lm_couple_to_single = np.nan + np.ones(shape_single)

        sol.Cw_priv_couple_to_single = np.nan + np.ones(shape_single)   # Private consumption marriage -> single
        sol.Cm_priv_couple_to_single = np.nan + np.ones(shape_single)
        sol.Cw_inter_couple_to_single = np.nan + np.ones(shape_single)    # intermediate consumption marriage -> single 
        sol.Cm_inter_couple_to_single = np.nan + np.ones(shape_single)    # Not used
        sol.Cw_tot_couple_to_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_couple_to_single = np.nan + np.ones(shape_single)
        sol.hw_couple_to_single = np.nan + np.ones(shape_single)                # housework,
        sol.hm_couple_to_single = np.nan + np.ones(shape_single)
        sol.Qw_couple_to_single = np.nan + np.ones(shape_single)        # home produced good, marriage -> single
        sol.Qm_couple_to_single = np.nan + np.ones(shape_single)        # home produced good, marriage -> single

        ## a.3. start as single
        sol.EVw_start_as_single = -np.inf + np.ones(shape_single)
        sol.EVm_start_as_single = -np.inf + np.ones(shape_single)  
        sol.EmargVw_start_as_single = np.nan + np.ones(shape_single)
        sol.EmargVm_start_as_single = np.nan + np.ones(shape_single)  

        sol.EVw_cond_meet_partner = np.nan + np.ones(shape_single)
        sol.EVm_cond_meet_partner = np.nan + np.ones(shape_single)
        sol.EVw_uncond_meet_partner = np.nan + np.ones(shape_single)
        sol.EVm_uncond_meet_partner = np.nan + np.ones(shape_single)


        # b. couples
        shape_couple = (par.T, par.num_power, par.num_love, par.num_types, par.num_types, par.num_K, par.num_K, par.num_A)
        shape_couple_d = (par.T, par.num_l, par.num_l, par.num_power, par.num_love, par.num_types, par.num_types, par.num_K, par.num_K, par.num_A)
        shape_couple_egm = (par.T, par.num_l, par.num_l, par.num_power,par.num_love, par.num_types, par.num_types, par.num_K, par.num_K,par.num_A_pd)
        # shape_couple_d = (par.T, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
            # couple states: T, power, love, human capital w, human capital w, assets

        # b.1. couple to couple
        sol.Vwd_couple_to_couple = np.ones(shape_couple_d) + np.nan                # value
        sol.Vmd_couple_to_couple = np.ones(shape_couple_d) + np.nan
        sol.Vd_couple_to_couple = np.ones(shape_couple_d) - np.inf                 # couple objective function

        sol.Cwd_priv_couple_to_couple = np.ones(shape_couple_d) + np.nan           # private consumption, couple
        sol.Cmd_priv_couple_to_couple = np.ones(shape_couple_d) + np.nan
        sol.Cd_inter_couple_to_couple = np.ones(shape_couple_d) + np.nan             # intermediate good, couple
        sol.Qd_couple_to_couple = np.ones(shape_couple_d) + np.nan                 # home produced good, couple
        sol.lwd_couple_to_couple = np.ones(shape_couple_d) + np.nan   # labor supply, couple
        sol.lmd_couple_to_couple = np.ones(shape_couple_d) + np.nan
        sol.hwd_couple_to_couple = np.ones(shape_couple_d) + np.nan                # housework, couple
        sol.hmd_couple_to_couple = np.ones(shape_couple_d) + np.nan
        sol.Cd_tot_couple_to_couple = np.ones(shape_couple_d) + np.nan

        sol.type_w = np.ones(par.num_types) + np.nan                                 # surplus of marriage
        sol.type_m = np.ones(par.num_types) + np.nan                                 # surplus of marriage

        sol.power_idx = np.zeros(shape_couple, dtype = np.int_)                     # index of bargaining weight (approx)
        sol.power = np.zeros(shape_couple) + np.nan                             # bargainng weight (interpolated)

        ### b.1.1. post-decision grids (EGM)
        sol.EmargUd_pd = np.zeros(shape_couple_egm)                     # Expected marginal utility post-decision
        sol.Cd_tot_pd = np.zeros(shape_couple_egm)                      # C for EGM
        sol.Md_pd = np.zeros(shape_couple_egm)                          # Endogenous grid
        sol.Vd_couple_to_couple_pd = np.zeros(shape_couple_egm)         # Value of being couple, post-decision

        ## b.2. single to couple
        sol.Vw_single_to_couple = np.nan + np.ones(shape_couple)           # value single -> marriage
        sol.Vm_single_to_couple = np.nan + np.ones(shape_couple)
        sol.V_single_to_couple = -np.inf + np.ones(shape_couple) 
        sol.lw_single_to_couple = np.nan + np.ones(shape_couple)   # labor supply single -> marriage
        sol.lm_single_to_couple = np.nan + np.ones(shape_couple)   # labor supply single -> marriage

        sol.Cw_priv_single_to_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_single_to_couple = np.nan + np.ones(shape_couple)
        sol.hw_single_to_couple = np.nan + np.ones(shape_couple)
        sol.hm_single_to_couple = np.nan + np.ones(shape_couple)
        sol.C_inter_single_to_couple = np.nan + np.ones(shape_couple)        
        sol.Q_single_to_couple = np.nan + np.ones(shape_couple)        
        sol.Cw_tot_single_to_couple = np.nan + np.ones(shape_couple)   
        sol.Cm_tot_single_to_couple = np.nan + np.ones(shape_couple) 
  
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

        sol.C_tot_start_as_couple = np.ones(shape_couple) + np.nan            # private consumption
        sol.Cw_priv_start_as_couple = np.ones(shape_couple) + np.nan            # private consumption
        sol.Cm_priv_start_as_couple = np.ones(shape_couple) + np.nan
        sol.C_inter_start_as_couple = np.ones(shape_couple) + np.nan              # intermediate good
        sol.Q_start_as_couple = np.ones(shape_couple) + np.nan                  # home produced good
        sol.lw_start_as_couple = np.ones(shape_couple) + np.nan    # labor supply
        sol.lm_start_as_couple = np.ones(shape_couple) + np.nan
        sol.hw_start_as_couple = np.ones(shape_couple) + np.nan                 # housework
        sol.hm_start_as_couple = np.ones(shape_couple) + np.nan
        

        # c. Precomputed intratemporal solution
        # c.1. couple
        shape_pre = (par.num_l, par.num_l, par.num_power, par.num_Ctot)
        sol.pre_Cwd_priv_couple = np.ones(shape_pre) + np.nan
        sol.pre_Cmd_priv_couple = np.ones(shape_pre) + np.nan
        sol.pre_Cd_inter_couple = np.ones(shape_pre) + np.nan
        sol.pre_Qd_couple = np.ones(shape_pre) + np.nan
        sol.pre_hwd_couple = np.ones(shape_pre) + np.nan
        sol.pre_hmd_couple = np.ones(shape_pre) + np.nan

        # c.2. single
        shape_pre_single = (par.num_l, par.num_K, par.num_Ctot)
        sol.pre_Cwd_priv_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cmd_priv_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cwd_inter_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Cmd_inter_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Qwd_single = np.ones(shape_pre_single) + np.nan
        sol.pre_Qmd_single = np.ones(shape_pre_single) + np.nan
        sol.pre_hwd_single = np.ones(shape_pre_single) + np.nan
        sol.pre_hmd_single = np.ones(shape_pre_single) + np.nan
    

        # d. simulation
        # NB: all arrays not containing "init" or "draw" in name are wiped before each simulation
        shape_sim = (par.simN,par.simT)
        sim.lw = np.nan + np.ones(shape_sim)
        sim.lm = np.nan + np.ones(shape_sim)
        sim.Cw_priv = np.nan + np.ones(shape_sim)               
        sim.Cm_priv = np.nan + np.ones(shape_sim)
        sim.hw = np.nan + np.ones(shape_sim)
        sim.hm = np.nan + np.ones(shape_sim)
        sim.Cw_inter = np.nan + np.ones(shape_sim)
        sim.Cm_inter = np.nan + np.ones(shape_sim)
        sim.Qw = np.nan + np.ones(shape_sim)
        sim.Qm = np.nan + np.ones(shape_sim)
        sim.Cw_tot = np.nan + np.ones(shape_sim)
        sim.Cm_tot = np.nan + np.ones(shape_sim)
        sim.C_tot = np.nan + np.ones(shape_sim)
        
        sim.Kw = np.nan + np.ones(shape_sim)
        sim.Km = np.nan + np.ones(shape_sim)
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)
        sim.type_w = np.zeros(shape_sim,dtype=np.int_)
        sim.type_m = np.zeros(shape_sim,dtype=np.int_)

        # for simulated moments
        sim.wage_w = np.nan + np.ones(shape_sim)
        sim.wage_m = np.nan + np.ones(shape_sim)
        sim.leisure_w = np.nan + np.ones(shape_sim)
        sim.leisure_m = np.nan + np.ones(shape_sim)
        
        # lifetime utility
        sim.util = np.nan + np.ones((par.simN, par.simT))
        sim.mean_lifetime_util = np.array([np.nan])

        # # containers for verifying simulaton
        # sim.A_own = np.nan + np.ones(shape_sim)
        # sim.A_partner = np.nan + np.ones(shape_sim)

        ## d.2. shocks
        self.allocate_draws()

        ## d.3. initial distribution
        # sim.init_A = np.linspace(0.0,par.max_A*0.2,par.simN) 
        sim.init_A = np.zeros(par.simN)
        # sim.init_Kw = np.linspace(0.0,par.max_K*0.2,par.simN) 
        sim.init_Kw = np.zeros(par.simN)
        # sim.init_Km = np.linspace(0.0,par.max_K*0.2,par.simN) 
        sim.init_Km = np.zeros(par.simN)
        sim.init_Aw = sim.init_A * par.div_A_share
        sim.init_Am = sim.init_A * (1.0 - par.div_A_share)
        sim.init_couple = np.random.choice([True, False], par.simN, p=[par.init_couple_share, 1 - par.init_couple_share])
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        sim.init_type_w = np.random.choice(par.num_types, par.simN, p=par.type_w_share)
        sim.init_type_m = np.random.choice(par.num_types, par.simN, p=par.type_m_share)
        
        # e. timing
        sol.solution_time = np.array([0.0])
        
        # f. optimal choices over discrete choices
        ## a. single
        sol.Vw_single_to_single = np.ones(shape_single) - np.inf
        sol.Vm_single_to_single = np.ones(shape_single) - np.inf
        sol.Cw_tot_single_to_single = np.ones(shape_single) + np.nan
        sol.Cm_tot_single_to_single = np.ones(shape_single) + np.nan
        sol.lw_single_to_single = np.ones(shape_single) + np.nan
        sol.lm_single_to_single = np.ones(shape_single) + np.nan
        # sol.Cw_priv_single_to_single = np.ones(shape_single) + np.nan
        # sol.Cm_priv_single_to_single = np.ones(shape_single) + np.nan
        # sol.hw_single_to_single = np.ones(shape_single) + np.nan
        # sol.hm_single_to_single = np.ones(shape_single) + np.nan
        # sol.Cw_inter_single_to_single = np.ones(shape_single) + np.nan
        # sol.Cm_inter_single_to_single = np.ones(shape_single) + np.nan
        # sol.Qw_single_to_single = np.ones(shape_single) + np.nan        
        # sol.Qm_single_to_single = np.ones(shape_single) + np.nan        

        # b. couple
        sol.V_couple_to_couple = np.ones(shape_couple) - np.inf
        sol.Vw_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.Vm_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.C_tot_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.lw_couple_to_couple = np.ones(shape_couple) + np.nan
        sol.lm_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.Cw_priv_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.Cm_priv_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.hw_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.hm_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.C_inter_couple_to_couple = np.ones(shape_couple) + np.nan
        # sol.Q_couple_to_couple = np.ones(shape_couple) + np.nan

    def allocate_draws(self):
        par = self.par
        sim = self.sim
        shape_sim = (par.simN,par.simT)

        np.random.seed(par.seed)
        # draw K shocks
        sim.draw_shock_Kw = np.random.lognormal(size=shape_sim, mean=-0.5*par.sigma_Kw**2, sigma=par.sigma_Kw)
        sim.draw_shock_Km = np.random.lognormal(size=shape_sim, mean=-0.5*par.sigma_Km**2, sigma=par.sigma_Km)
        sim.draw_love = np.random.normal(size=shape_sim)
        sim.draw_meet = np.random.uniform(size=shape_sim) # for meeting a partner

        sim.draw_uniform_partner_Kw = np.random.uniform(size=shape_sim) # for inverse cdf transformation of partner human capital
        sim.draw_uniform_partner_Km = np.random.uniform(size=shape_sim) # for inverse cdf tranformation of partner human capital
        sim.draw_uniform_partner_Aw = np.random.uniform(size=shape_sim) # for inverse cdf transformation of partner wealth
        sim.draw_uniform_partner_Am = np.random.uniform(size=shape_sim) # for inverse cdf tranformation of partner wealth
        sim.draw_uniform_partner_type_w = np.random.uniform(size=shape_sim) # for discrete draw of partner type
        sim.draw_uniform_partner_type_m = np.random.uniform(size=shape_sim) # for discrete draw of partner type

        sim.draw_repartner_love = par.sigma_love*np.random.normal(0.0,1.0,size=shape_sim) #np.random.choice(par.num_love, p=par.prob_partner_love, size=shape_sim) # Love index when repartnering

        
    def setup_grids(self):
        par = self.par
        
        par.grid_type = np.arange(par.num_types, dtype=np.float64)        
        par.grid_mu_w, par.type_w_share = quadrature.normal_gauss_hermite(par.sigma_mu, par.num_types, mu=par.mu)
        par.grid_mu_m, par.type_m_share = quadrature.normal_gauss_hermite(par.sigma_mu * par.sigma_mu_mult, par.num_types, mu=par.mu * par.mu_mult)

        
        par.grid_l = np.array([0.0, 0.75, 1])  * par.full_time_hours # labor supply choices (in hours)
        par.num_l = len(par.grid_l)
        
        # 0. time
        par.grid_t = np.arange(par.T) # time grid
        
        # a. state variables
        # a.1. wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)       # asset grid

        par.grid_Aw = par.div_A_share * par.grid_A                  # asset grid in case of divorce
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # a.2. human capital
        par.grid_Kw = nonlinspace(0.0, par.max_K, par.num_K, 1.1)
        par.grid_Km = nonlinspace(0.0, par.max_K, par.num_K, 1.1)
        
        par.grid_shock_Kw, par.grid_weight_Kw = quadrature.log_normal_gauss_hermite(par.sigma_Kw, par.num_shock_K)
        par.grid_shock_Km, par.grid_weight_Km = quadrature.log_normal_gauss_hermite(par.sigma_Km, par.num_shock_K)

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

        # b. Precomputation of intertemporal allocation
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)  

        # b.1. couple
        shape_pre_couple = (par.num_l, par.num_l, par.num_power, par.num_marg_u, )
        par.grid_marg_u_couple = np.ones(shape_pre_couple) + np.nan
        par.grid_marg_u_couple_for_inv = np.ones(shape_pre_couple) + np.nan

        # b.2. single
        shape_pre_single = (par.num_l, par.num_marg_u)
        par.grid_marg_u_single_w = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_m = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_w_for_inv = np.ones(shape_pre_single) + np.nan
        par.grid_marg_u_single_m_for_inv = np.ones(shape_pre_single) + np.nan

        # b.3 Common
        par.grid_C_for_marg_u = nonlinspace(1.0e-5,par.max_Ctot,par.num_marg_u,1.1)    # Consumption interpolator grid 

        # EGM
        par.grid_inv_marg_u = np.flip(par.grid_C_for_marg_u) # Flipped to make interpolation possible ## AMO: invert
        if par.interp_inverse:
            par.grid_inv_marg_u = 1.0/par.grid_inv_marg_u

        par.grid_A_pd = nonlinspace(0.0,par.max_A_pd,par.num_A_pd,1.1)
        par.grid_Aw_pd = par.div_A_share * par.grid_A_pd
        par.grid_Am_pd = (1.0 - par.div_A_share) * par.grid_A_pd


        # re-partering probabilities
        par.prob_repartner = par.p_meet*np.ones(par.T) # likelihood of meeting a partner
        
        if np.isnan(par.prob_partner_Kw[0,0]):
            par.prob_partner_Kw = np.eye(par.num_K)
    
        if np.isnan(par.prob_partner_Km[0,0]):
            par.prob_partner_Km = np.eye(par.num_K)
       
        if np.isnan(par.prob_partner_A_w[0,0]):
            par.prob_partner_A_w = np.eye(par.num_A) #np.ones((par.num_A,par.num_A))/par.num_A # likelihood of meeting a partner with a particular level of wealth, conditional on own
    
        if np.isnan(par.prob_partner_A_m[0,0]):
            par.prob_partner_A_m = np.eye(par.num_A) #np.ones((par.num_A,par.num_A))/par.num_A # likelihood of meeting a partner with a particular level of wealth, conditional on own
       
        if np.isnan(par.prob_partner_type_w[0,0]):
            par.prob_partner_type_w = np.eye(par.num_types) #np.ones((par.num_types,par.num_types))/par.num_types # likelihood of meeting a partner with a particular type, conditional on own
        
        if np.isnan(par.prob_partner_type_m[0,0]):
            par.prob_partner_type_m = np.eye(par.num_types) #np.ones((par.num_types,par.num_types))/par.num_types # likelihood of meeting a partner with a particular type, conditional on own
           
           
        # Norm distributed initial love - note: Probability mass between points (approximation of continuous distribution)
        if par.sigma_love<=1.0e-6:
            love_cdf = np.where(par.grid_love>=0.0,1.0,0.0)
        else:
            love_cdf = stats.norm.cdf(par.grid_love,0.0,par.sigma_love)
        par.prob_partner_love = np.diff(love_cdf,1)
        par.prob_partner_love = np.append(par.prob_partner_love,0.0) # lost last point in diff
        # par.prob_partner_love = np.ones(par.num_love)/par.num_love # uniform

        par.cdf_partner_Kw = np.cumsum(par.prob_partner_Kw,axis=1) # cumulative distribution to be used in simulation
        par.cdf_partner_Km = np.cumsum(par.prob_partner_Km,axis=1)
        par.cdf_partner_Aw = np.cumsum(par.prob_partner_A_w,axis=1) # cumulative distribution to be used in simulation
        par.cdf_partner_Am = np.cumsum(par.prob_partner_A_m,axis=1)
        par.cdf_partner_type_w = np.cumsum(par.prob_partner_type_w,axis=1)
        par.cdf_partner_type_m = np.cumsum(par.prob_partner_type_m,axis=1)


    def solve(self):

        sol = self.sol
        par = self.par 

        # re-allocate to ensure new solution
        # TODO: find alternative to re-allocate every time model is solved
        self.allocate()

        self.cpp.solve(sol,par)


    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        # clear simulation
        for key, val in sim.__dict__.items():
            if 'init' in key or 'draw' in key: continue
            setattr(sim, key, np.zeros(val.shape)+np.nan)

        self.cpp.simulate(sim,sol,par)

        sim.mean_lifetime_util[0] = np.mean(np.sum(sim.util,axis=1))
        
    # Make a function that takes sim and makes it into a polars dataframe
    
    # Estimation
    def obj_func(self,theta,estpar,datamoms,weights=None,do_print=False):
        
        # update parameters, impose bounds and return penalty
        penalty = self.update_par(theta,estpar)
        
        # a. solve model
        self.solve()

        # b. simulate
        self.simulate()

        # c. calculate moments
        moms = self.calc_moments()
        
        # d. go through all potential moments and calculate squared difference
        sqdiff = 0.0
        for mom_name in datamoms.keys():
            diff = moms[mom_name] - datamoms[mom_name]
            
            if weights is not None:
                if mom_name in weights:
                    diff *= weights[mom_name]
            
            sqdiff += diff*diff
            
        # e. add additional penalty if employment share is too high
        cutoff = 93.0
        penalty_moms = 0.0
        for mom_name in ('employment_rate_w_35_44', 'employment_rate_m_35_44'):
            if mom_name in moms:
                if moms[mom_name] > cutoff:
                    penalty_moms += 10.0 * (moms[mom_name] - cutoff) ** 2

        objective = sqdiff + penalty + penalty_moms
        
        if do_print:
            print('Parameters:')
            for i in range(theta.size):
                name = list(estpar.keys())[i]
                print(f'  {name:<15}: {getattr(self.par,name):.4f} (init: {estpar[name]["guess"]:.4f})') 
            print('Moments:')
            for mom_name in datamoms.keys():
                print(f'  {str(mom_name):<25}: sim: {moms[mom_name]:.4f}, data: {datamoms[mom_name]:.4f}')
            print(f'Objective function value: {objective:.4f} (penalty: {penalty:.4f}, {penalty_moms:.4f})')
            print('-------------------------------------')

        return objective


    def calc_moments(self):
        # calculate all potential moments and store in ordered dict
        
        # a) setup
        par = self.par
        sim = self.sim
        moms = OrderedDict()
        money_metric = 1 #_000
        hours_metric = 7*16  # full time hours per week
        
        # b) samples
        ## age groups
        t_array = np.tile(np.arange(par.simT, dtype=int), (par.simN, 1))
        age_25 = 0
        age_25_to_34_mask = (t_array >= age_25) & (t_array <= age_25 + 9)
        age_35_to_44_mask = (t_array >= age_25 + 10) & (t_array <= age_25 + 19)
        age_25_to_41_mask = (t_array >= age_25) & (t_array <= age_25 + 16)
        
        ## full time
        full_time = par.grid_l[-1]
        full_time_w_mask = (sim.lw >= full_time)
        full_time_m_mask = (sim.lm >= full_time)
        
        # unemployment
        unemployed = par.grid_l[0]
        unemployed_w_mask = (sim.lw <= unemployed)
        unemployed_m_mask = (sim.lm <= unemployed)
        
        ## couple
        couple_mask = (sim.couple == 1)
        ever_couple_mask = np.repeat(np.any(sim.couple == 1, axis=1, keepdims=True), par.simT, axis=1)

        # c) moments
        # wages (should be stored in simulation. Now just use labor supply, e.g. lw, for illustration)
        moms['wage_level_w_25_34'] = np.nanmean(sim.wage_w[age_25_to_34_mask & couple_mask & full_time_w_mask]) * money_metric 
        moms['wage_level_m_25_34'] = np.nanmean(sim.wage_m[age_25_to_34_mask & couple_mask & full_time_m_mask]) * money_metric
        moms['wage_level_w_35_44'] = np.nanmean(sim.wage_w[age_35_to_44_mask & couple_mask & full_time_w_mask]) * money_metric
        moms['wage_level_m_35_44'] = np.nanmean(sim.wage_m[age_35_to_44_mask & couple_mask & full_time_m_mask]) * money_metric
        # moms['wage_level_w_25_34'] = np.nanmean(sim.wage_w[age_25_to_34_mask & couple_mask]) * 1.0/(1.0 - par.tax_rate) * money_metric 
        # moms['wage_level_m_25_34'] = np.nanmean(sim.wage_m[age_25_to_34_mask & couple_mask]) * 1.0/(1.0 - par.tax_rate) * money_metric
        # moms['wage_level_w_35_44'] = np.nanmean(sim.wage_w[age_35_to_44_mask & couple_mask]) * 1.0/(1.0 - par.tax_rate) * money_metric
        # moms['wage_level_m_35_44'] = np.nanmean(sim.wage_m[age_35_to_44_mask & couple_mask]) * 1.0/(1.0 - par.tax_rate) * money_metric

        # employment rates
        moms['employment_rate_w_35_44'] = np.nanmean(sim.lw[age_35_to_44_mask & couple_mask] > unemployed) * 100.0
        moms['employment_rate_m_35_44'] = np.nanmean(sim.lm[age_35_to_44_mask & couple_mask] > unemployed) * 100.0
        moms['work_hours_w'] = np.nanmean(sim.lw[age_25_to_41_mask & (~unemployed_w_mask)]) * hours_metric
        moms['work_hours_m'] = np.nanmean(sim.lm[age_25_to_41_mask & (~unemployed_m_mask)]) * hours_metric
        
        # home production
        moms['home_prod_w'] = np.nanmean(sim.hw[age_25_to_41_mask]) * hours_metric
        moms['home_prod_m'] = np.nanmean(sim.hm[age_25_to_41_mask]) * hours_metric
        
        # leisure
        moms['leisure_w'] = np.nanmean(sim.leisure_w) * hours_metric
        moms['leisure_m'] = np.nanmean(sim.leisure_m) * hours_metric
        
        # consumption
        moms['consumption'] = np.nanmean(sim.C_tot)  * money_metric
        
        # marriage
        moms['marriage_rate_35_44'] = np.nanmean(sim.couple[age_35_to_44_mask]) * 100.0
        moms['divorce_rate_35_44'] = np.nanmean(sim.couple[age_35_to_44_mask & ever_couple_mask]==0) * 100.0
        
        # t_level = 0
        # for dt in (5,10,15):
        #     Iw = (~np.isnan(sim.wage_w[:,t_level+dt])) & (~np.isnan(sim.wage_w[:,t_level+dt-1]))
        #     Im = (~np.isnan(sim.wage_m[:,t_level+dt])) & (~np.isnan(sim.wage_m[:,t_level+dt-1]))
        #     moms[('wage_growth_w',dt)] = np.mean(np.log(sim.wage_w[Iw,t_level+dt])-np.log(sim.wage_w[Iw,t_level+dt-1]))
        #     moms[('wage_growth_m',dt)] = np.mean(np.log(sim.wage_m[Im,t_level+dt])-np.log(sim.wage_m[Im,t_level+dt-1]))

        # # Time allocation
        # moms['time_work_w'] = np.nanmean(sim.lw.ravel()) * hours_metric
        # moms['time_work_m'] = np.nanmean(sim.lm.ravel()) * hours_metric

        # moms['time_leisure_w'] = np.nanmean(sim.leisure_w.ravel()) * hours_metric 
        # moms['time_leisure_m'] = np.nanmean(sim.leisure_m.ravel()) * hours_metric
        


        return moms
    

    def update_par(self,theta,estpar):
        """ update model parameters and impose bounds with penalty """

        if theta is None: return 0
        
        names = list(estpar.keys())
        
        # a. clip and penalty
        penalty = 0
        for i in range(theta.size):
            
            # i. clip
            lower = estpar[names[i]]['lower']
            upper = estpar[names[i]]['upper']
            if (lower != None) or (upper != None):
                theta_clipped = np.clip(theta[i],lower,upper)
            else:
                theta_clipped = theta[i]

            # ii. penalty
            penalty += 10_000.0*(theta[i]-theta_clipped)**2

            ## iii. set clipped value
            setattr(self.par,names[i],theta_clipped)

        return penalty
    
    def global_search(self,estpars, datamoms,weights,num_points,num_guess=1,folder_name='calibrate',draw_method='halton',do_print=False):
        from calibrate import draw
        
        num_params = len(estpars)
    
        # draw numbers in [0,1] in all dimensions
        w = draw.generate_initial(num_params,num_points,method=draw_method)
        
        # construct associated parameter values from bounds and unit interval numbers
        lower = np.array([value['lower'] for value in estpars.values()])
        upper = np.array([value['upper'] for value in estpars.values()])
        names = list(estpars.keys())
        
        param_guess_mat = w*lower + (1-w)*upper #(num_points,num_params)
        
        # setup log-file
        with open(f'{folder_name}/parameter_search.txt', 'w') as f:
                pass
            
        # loop through all parameter vectors
        objs = np.nan + np.ones(num_points)
        obj_min = 1e10
        i_min = -1
        for i in range(num_points):
            theta = param_guess_mat[i,:]
            objs[i] = self.obj_func(theta,estpars, datamoms,weights, do_print=do_print)

            if objs[i] < obj_min:
                obj_min = objs[i]   
                i_min = i
                
            with open(f'{folder_name}/parameter_search.txt', 'a+') as f:
                f.write(f'\nguess {i} of {num_points}\n')
                for x,name in zip(theta,names): f.write(f' {name:22s} = {x:8.4f}\n')
                f.write(f' obj. = {objs[i]:.8f} (best :{obj_min:.6f})\n')
                
        Imin = np.argsort(objs) 
        return param_guess_mat[Imin[0:num_guess],:]
