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

        # -------- core / accounting --------
        par.R = 1.03
        par.beta = 1.0 / par.R  # Discount factor

        par.div_A_share = 0.5  # divorce share of wealth to wife
        par.div_cost = 0.0

        par.available_hours = 1.0
        par.tax_rate = 0.25

        # -------- preferences (baseline + multipliers) --------
        par.rho = 2.0
        par.rho_mult = 1.0

        par.phi = 0.05
        par.phi_mult = 1.0

        par.eta = 0.5
        par.eta_mult = 1.0

        par.lambda_ = 0.5
        par.lambda_mult = 1.0

        # -------- home production --------
        par.alpha = 1.0
        par.zeta = 0.5
        par.omega = 0.5
        par.pi = 0.5

        # -------- income / wages (baseline + multipliers) --------
        par.mu = 0.5
        par.sigma_mu = 0.1
        par.gamma = 0.1
        par.gamma2 = 0.00

        par.mu_mult = 1.0
        par.sigma_mu_mult = 1.0
        par.gamma_mult = 1.0
        par.gamma2_mult = 1.0

        # -------- human capital process --------
        par.delta = -1.0
        par.phi_k = 0.4
        par.sigma_epsilon = 0.08
        par.sigma_epsilon_mult = 1.0

        par.num_shock_K = 5
        par.sigma_K = 0.1
        par.sigma_K_mult = 1.0

        # -------- model horizon / state space sizes --------
        par.T = 10

        # wealth
        par.num_A = 50
        par.max_A = 15.0

        # human capital
        par.num_K = 16
        par.max_K = par.T * 1.5

        # bargaining power
        par.num_power = 21

        # love / match quality
        par.num_love = 11
        par.max_love = 100.0
        par.sigma_love = 0.1
        par.num_shock_love = 5  # cannot be 1 due to interpolation

        # types
        par.num_types = 3
        
        # EGM
        par.num_A_pd = par.num_A * 2
        par.max_A_pd = par.max_A
        
        # precomputation of intratemporal solution (for iEGM)
        par.num_marg_u = 200

        # -------- (re-)partnering --------
        par.p_meet = 0.0
        par.prob_partner_Kw = np.array([[np.nan]])
        par.prob_partner_Km = np.array([[np.nan]])
        par.prob_partner_A_w = np.array([[np.nan]])
        par.prob_partner_A_m = np.array([[np.nan]])
        par.prob_partner_type_w = np.array([[np.nan]])
        par.prob_partner_type_m = np.array([[np.nan]])

        # -------- discrete choices --------
        par.full_time_hours = 0.35

        # -------- precomputation controls --------
        par.interp_inverse = False
        par.precompute_intratemporal = True

        # intratemporal precomputation
        par.num_Ctot = 20
        par.max_Ctot = par.max_A * 2

        # EGM
        par.do_egm = False

        # -------- simulation --------
        par.seed = 9210
        par.simT = par.T
        par.simN = 50
        par.init_couple_share = 1.0

        # -------- misc --------
        par.threads = 8
        par.num_multistart = 1
        par.interp_method = "linear"
        par.centered_gradient = True

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
        
    def fast_unravel_indices(self, shape, dtype=np.int64):
        """Helper function to unravel indices for a given shape (for unindexing grids).
        This is a faster implementation of np.unravel_index for large arrays, as it avoids creating large intermediate arrays.
        INPUT:
        - shape: tuple, the shape of the array to unravel indices for
        - dtype: data type of the output indices (default: np.int32)
        
        OUTPUT:
        - tuple of arrays, where each array contains the indices for that dimension
        """
        size = np.product(shape)
        idx = np.arange(size, dtype=dtype)
        out = []
        for dim in reversed(shape):
            out.append(idx % dim)
            idx //= dim
        return tuple(reversed(out)) 
   
    def setup_grids(self):
        par = self.par

        # ---------- 0) time ----------
        par.grid_t = np.arange(par.T)

        # ---------- 1) types ----------
        par.grid_type = np.arange(par.num_types, dtype=np.float64)
        par.grid_mu_w, par.type_w_share = quadrature.normal_gauss_hermite(par.sigma_mu, par.num_types, mu=par.mu)
        par.grid_mu_m, par.type_m_share = quadrature.normal_gauss_hermite(
            par.sigma_mu * par.sigma_mu_mult, par.num_types, mu=par.mu * par.mu_mult
        )

        # ---------- 2) discrete choices ----------
        par.grid_l = np.array([0.0, 0.75, 1.0]) * par.full_time_hours
        par.num_l = len(par.grid_l)

        # ---------- 3) state grids ----------
        # 3.1 assets
        par.grid_A = nonlinspace(0.0, par.max_A, par.num_A, 1.1)
        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # 3.2 human capital
        par.grid_Kw = nonlinspace(0.0, par.max_K, par.num_K, 1.1)
        par.grid_Km = nonlinspace(0.0, par.max_K, par.num_K, 1.1)

        # 3.3 bargaining power (more mass in tails)
        odd_num = np.mod(par.num_power, 2)
        first_part = nonlinspace(1e-6, 0.5, (par.num_power + odd_num) // 2, 1.3)
        last_part = np.flip(1.0 - nonlinspace(1e-6, 0.5, (par.num_power - odd_num) // 2 + 1, 1.3))[1:]
        par.grid_power = np.append(first_part, last_part)
        par.grid_power_flip = np.flip(par.grid_power)

        # 3.4 love
        par.grid_love = np.linspace(-par.max_love, par.max_love, par.num_love) if par.num_love > 1 else np.array([0.0])

        # ---------- 4) shocks ----------
        par.grid_shock_Kw, par.grid_weight_Kw = quadrature.log_normal_gauss_hermite(par.sigma_Kw, par.num_shock_K)
        par.grid_shock_Km, par.grid_weight_Km = quadrature.log_normal_gauss_hermite(par.sigma_Km, par.num_shock_K)

        if par.sigma_love <= 1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love, par.grid_weight_love = np.array([0.0]), np.array([1.0])
        else:
            par.grid_shock_love, par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love, par.num_shock_love)

        # ---------- 5) EGM / iEGM grids ----------
        # post-decision assets
        par.grid_A_pd = nonlinspace(0.0, par.max_A_pd, par.num_A_pd, 1.1)
        par.grid_Aw_pd = par.div_A_share * par.grid_A_pd
        par.grid_Am_pd = (1.0 - par.div_A_share) * par.grid_A_pd

        # total consumption grid + marginal utility grid
        par.grid_Ctot = nonlinspace(1.0e-6, par.max_Ctot, par.num_Ctot, 1.1)
        par.grid_C_for_marg_u = nonlinspace(1.0e-5, par.max_Ctot, par.num_marg_u, 1.1)

        par.grid_inv_marg_u = np.flip(par.grid_C_for_marg_u)
        if par.interp_inverse:
            par.grid_inv_marg_u = 1.0/par.grid_inv_marg_u
        
        
        
        # ---------- 6) repartnering ----------
        
        # re-partering probabilities
        par.prob_repartner = par.p_meet*np.ones(par.T) # likelihood of meeting a partner
        
        def _use_eye_if_nan(arr, n):
            return np.eye(n) if np.isnan(arr[0, 0]) else arr
            
        par.prob_partner_Kw = _use_eye_if_nan(par.prob_partner_Kw, par.num_K)
        par.prob_partner_Km = _use_eye_if_nan(par.prob_partner_Km, par.num_K)
        par.prob_partner_A_w = _use_eye_if_nan(par.prob_partner_A_w, par.num_A)
        par.prob_partner_A_m = _use_eye_if_nan(par.prob_partner_A_m, par.num_A)
        par.prob_partner_type_w = _use_eye_if_nan(par.prob_partner_type_w, par.num_types)
        par.prob_partner_type_m = _use_eye_if_nan(par.prob_partner_type_m, par.num_types)
          
        par.cdf_partner_Kw = np.cumsum(par.prob_partner_Kw,axis=1) # cumulative distribution to be used in simulation
        par.cdf_partner_Km = np.cumsum(par.prob_partner_Km,axis=1)
        par.cdf_partner_Aw = np.cumsum(par.prob_partner_A_w,axis=1) # cumulative distribution to be used in simulation
        par.cdf_partner_Am = np.cumsum(par.prob_partner_A_m,axis=1)
        par.cdf_partner_type_w = np.cumsum(par.prob_partner_type_w,axis=1)
        par.cdf_partner_type_m = np.cumsum(par.prob_partner_type_m,axis=1)
        
        # Norm distributed initial love - note: Probability mass between points (approximation of continuous distribution)
        if par.sigma_love <= 1.0e-6:
            love_cdf = np.where(par.grid_love >= 0.0,1.0,0.0)
        else:
            love_cdf = stats.norm.cdf(par.grid_love,0.0,par.sigma_love)
        par.prob_partner_love = np.append(np.diff(love_cdf, 1), 0.0)
        
        
        # ---------- 7) unindex grids --------
        # a. singles
        shape_single = (par.T, par.num_types, par.num_K, par.num_A)
        shape_single_d = (par.T, par.num_types, par.num_l, par.num_K, par.num_A)
        shape_single_egm = (par.T, par.num_types, par.num_l, par.num_K, par.num_A_pd)

        # b. couples
        shape_couple = (par.T, par.num_types, par.num_types, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
        shape_couple_d = (par.T, par.num_types, par.num_types, par.num_l, par.num_l, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
        shape_couple_egm = (par.T, par.num_types, par.num_types, par.num_l, par.num_l, par.num_power,par.num_love, par.num_K, par.num_K,par.num_A_pd)
 
        # c. precomputations
        shape_pre_single = (par.num_l, par.num_K, par.num_Ctot)
        shape_pre_couple = (par.num_l, par.num_l, par.num_power, par.num_Ctot)
        
        # flatten shape indices 
        # par.idx_single_T, par.idx_single_type, par.idx_single_K, par.idx_single_A = self.fast_unravel_indices(shape_single)
        # par.idx_single_d_T, par.idx_single_d_type, par.idx_single_d_l, par.idx_single_d_K, par.idx_single_d_A = self.fast_unravel_indices(shape_single_d)
        # par.idx_single_egm_T, par.idx_single_egm_type, par.idx_single_egm_l, par.idx_single_egm_K, par.idx_single_egm_A = self.fast_unravel_indices(shape_single_egm)
        # par.idx_couple_T, par.idx_couple_type_w, par.idx_couple_type_m, par.idx_couple_power, par.idx_couple_love, par.idx_couple_Kw, par.idx_couple_Km, par.idx_couple_A = self.fast_unravel_indices(shape_couple)
        # par.idx_couple_d_T, par.idx_couple_d_type_w, par.idx_couple_d_type_m, par.idx_couple_d_lw, par.idx_couple_d_lm, par.idx_couple_d_power, par.idx_couple_d_love, par.idx_couple_d_Kw, par.idx_couple_d_Km, par.idx_couple_d_A = self.fast_unravel_indices(shape_couple_d)
        # par.idx_couple_egm_T, par.idx_couple_egm_type_w, par.idx_couple_egm_type_m, par.idx_couple_egm_lw, par.idx_couple_egm_lm, par.idx_couple_egm_power, par.idx_couple_egm_love, par.idx_couple_egm_Kw, par.idx_couple_egm_Km, par.idx_couple_egm_A = self.fast_unravel_indices(shape_couple_egm)
        # par.idx_pre_single_l, par.idx_pre_single_K, par.idx_pre_single_Ctot = self.fast_unravel_indices(shape_pre_single)
        # par.idx_pre_couple_lw, par.idx_pre_couple_lm, par.idx_pre_couple_power, par.idx_pre_couple_Ctot = self.fast_unravel_indices(shape_pre_couple)
        # OBS: shape_pre_single should not be used for iEGM where i_u_marg is important and not iC.
        
    def allocate(self):
        """Allocate model storage (memory) and initialize all values."""
        # derive gender-specific parameters + grids (needed for sizes)
        self.setup_gender_parameters()
        self.setup_grids()

        self.allocate_memory()
        self.fill_allocations()

    def allocate_memory(self):
        """Allocate arrays only (no filling/initialization)."""
        par = self.par
        sol = self.sol
        sim = self.sim

        # a. singles
        shape_single = (par.T, par.num_types, par.num_K, par.num_A)
        shape_single_d = (par.T, par.num_types, par.num_l, par.num_K, par.num_A)
        shape_single_egm = (par.T, par.num_types, par.num_l, par.num_K, par.num_A_pd)

        # b. couples
        shape_couple = (par.T, par.num_types, par.num_types, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
        shape_couple_d = (par.T, par.num_types, par.num_types, par.num_l, par.num_l, par.num_power, par.num_love, par.num_K, par.num_K, par.num_A)
        shape_couple_egm = (par.T, par.num_types, par.num_types, par.num_l, par.num_l, par.num_power,par.num_love, par.num_K, par.num_K,par.num_A_pd)
 
        # c. precomputations
        shape_pre_single = (par.num_l, par.num_Ctot)
        shape_pre_couple = (par.num_l, par.num_l, par.num_power, par.num_Ctot)

        # d. simulation
        shape_sim = (par.simN, par.simT)
        
        
        # helper
        def _alloc(obj, name, shape, dtype=np.float64):
            setattr(obj, name, np.empty(shape, dtype=dtype))

        # --- a.1. single to single (discrete l choices) ---
        for name in (
            "Vwd_single_to_single", "Vmd_single_to_single",
            "Cwd_tot_single_to_single", "Cmd_tot_single_to_single",
            "Cwd_priv_single_to_single", "Cmd_priv_single_to_single",
            "Cwd_inter_single_to_single", "Cmd_inter_single_to_single",
            "Qwd_single_to_single", "Qmd_single_to_single",
            "lwd_single_to_single", "lmd_single_to_single",
            "hwd_single_to_single", "hmd_single_to_single",
        ):
            _alloc(sol, name, shape_single_d)

        # --- single to single (EGM post-decision) ---
        for name in (
            "EmargUwd_single_to_single_pd", "Cwd_tot_single_to_single_pd",
            "Mwd_single_to_single_pd", "Vwd_single_to_single_pd",
            "EmargUmd_single_to_single_pd", "Cmd_totm_single_to_single_pd",
            "Mmd_single_to_single_pd", "Vmd_single_to_single_pd",
        ):
            _alloc(sol, name, shape_single_egm)

        # --- a.2. couple to single ---
        for name in (
            "Vw_couple_to_single", "Vm_couple_to_single",
            "lw_couple_to_single", "lm_couple_to_single",
            "Cw_priv_couple_to_single", "Cm_priv_couple_to_single",
            "Cw_inter_couple_to_single", "Cm_inter_couple_to_single",
            "Cw_tot_couple_to_single", "Cm_tot_couple_to_single",
            "hw_couple_to_single", "hm_couple_to_single",
            "Qw_couple_to_single", "Qm_couple_to_single",
        ):
            _alloc(sol, name, shape_single)

        # --- a.3. start as single ---
        for name in (
            "EVw_start_as_single", "EVm_start_as_single",
            "EmargVw_start_as_single", "EmargVm_start_as_single",
            "EVw_cond_meet_partner", "EVm_cond_meet_partner",
            "EVw_uncond_meet_partner", "EVm_uncond_meet_partner",
        ):
            _alloc(sol, name, shape_single)
            
        # --- a.4 optimal discrete choices single ---
        for name in (
            "Vw_single_to_single", "Vm_single_to_single",
            "Cw_tot_single_to_single", "Cm_tot_single_to_single",
            "lw_single_to_single", "lm_single_to_single",
        ):
            _alloc(sol, name, shape_single)

        # --- b.1. couple to couple ---
        for name in (
            "Vwd_couple_to_couple", "Vmd_couple_to_couple", "Vd_couple_to_couple",
            "Cwd_priv_couple_to_couple", "Cmd_priv_couple_to_couple",
            "Cd_inter_couple_to_couple", "Qd_couple_to_couple",
            "lwd_couple_to_couple", "lmd_couple_to_couple",
            "hwd_couple_to_couple", "hmd_couple_to_couple",
            "Cd_tot_couple_to_couple",
        ):
            _alloc(sol, name, shape_couple_d)

        # state-level power objects (must be shape_couple)
        _alloc(sol, "power_idx", shape_couple, dtype=np.int_)
        _alloc(sol, "power", shape_couple)

        # --- couple to couple (EGM post-decision) ---
        for name in (
            "EmargUd_pd", "Cd_tot_pd", "Md_pd", "Vd_couple_to_couple_pd"
        ):
            _alloc(sol, name, shape_couple_egm)

        # --- b.2. single to couple ---
        for name in (
            "Vw_single_to_couple", "Vm_single_to_couple", "V_single_to_couple",
            "lw_single_to_couple", "lm_single_to_couple",
            "Cw_priv_single_to_couple", "Cm_priv_single_to_couple",
            "hw_single_to_couple", "hm_single_to_couple",
            "C_inter_single_to_couple", "Q_single_to_couple",
            "Cw_tot_single_to_couple", "Cm_tot_single_to_couple",
        ):
            _alloc(sol, name, shape_couple)
            
        # --- b.3. start as couple ---
        for name in (
            "Vw_start_as_couple", "Vm_start_as_couple", "margV_start_as_couple",
            "EVw_start_as_couple", "EVm_start_as_couple", "EmargV_start_as_couple",
            "C_tot_start_as_couple", "Cw_priv_start_as_couple", "Cm_priv_start_as_couple",
            "C_inter_start_as_couple", "Q_start_as_couple",
            "lw_start_as_couple", "lm_start_as_couple",
            "hw_start_as_couple", "hm_start_as_couple",
        ):
            _alloc(sol, name, shape_couple)
            
        # --- b.4 optimal discrete choices couple ---
        for name in (
            "V_couple_to_couple", "Vw_couple_to_couple", "Vm_couple_to_couple",
            "C_tot_couple_to_couple", "lw_couple_to_couple", "lm_couple_to_couple",
        ):
            _alloc(sol, name, shape_couple)

        # --- c. precomputed intratemporal solution ---
        for name in (
            "pre_Cwd_priv_single", "pre_Cmd_priv_single",
            "pre_Cwd_inter_single", "pre_Cmd_inter_single",
            "pre_Qwd_single", "pre_Qmd_single",
            "pre_hwd_single", "pre_hmd_single",
            "grid_marg_u_single_w", "grid_marg_u_single_m",
            "grid_marg_u_single_w_for_inv", "grid_marg_u_single_m_for_inv",
        ):
            _alloc(sol, name, shape_pre_single)

        for name in (
            "pre_Cwd_priv_couple", "pre_Cmd_priv_couple",
            "pre_Cd_inter_couple", "pre_Qd_couple",
            "pre_hwd_couple", "pre_hmd_couple",
            "pre_Cwd_priv_single", "pre_Cmd_priv_single",
            "pre_Cwd_inter_single", "pre_Cmd_inter_single",
            "pre_Qwd_single", "pre_Qmd_single",
            "pre_hwd_single", "pre_hmd_single",
            "grid_marg_u_couple", "grid_marg_u_couple_for_inv",
        ):
            _alloc(sol, name, shape_pre_couple)


        # --- d.1 simulation variables ---
        shape_sim = (par.simN, par.simT)
        for name in (
            "lw", "lm", "Cw_priv", "Cm_priv", "hw", "hm",
            "Cw_inter", "Cm_inter", "Qw", "Qm", "Cw_tot", "Cm_tot", "C_tot",
            "Kw", "Km", "A", "Aw", "Am", "couple", "power", "love",
            "wage_w", "wage_m", "leisure_w", "leisure_m",
        ):
            _alloc(sim, name, shape_sim)

        # ints (do NOT allocate as float)
        _alloc(sim, "type_w", shape_sim, dtype=np.int_)
        _alloc(sim, "type_m", shape_sim, dtype=np.int_)

        _alloc(sim, "util", (par.simN, par.simT))
        _alloc(sim, "mean_lifetime_util", (1,))

        # --- d.2 shocks ---
        for name in (
            "draw_shock_Kw", "draw_shock_Km", "draw_love", "draw_meet",
            "draw_uniform_partner_Kw", "draw_uniform_partner_Km",
            "draw_uniform_partner_Aw", "draw_uniform_partner_Am",
            "draw_uniform_partner_type_w", "draw_uniform_partner_type_m",
            "draw_repartner_love",
        ):
            _alloc(sim, name, shape_sim)


        # --- d.3 initial distribution ---
        _alloc(sim, "init_type_w", (par.simN,), dtype=np.int_)
        _alloc(sim, "init_type_m", (par.simN,), dtype=np.int_)
        _alloc(sim, "init_love", (par.simN,))
        _alloc(sim, "init_Kw", (par.simN,))
        _alloc(sim, "init_Km", (par.simN,))
        _alloc(sim, "init_A", (par.simN,))
        _alloc(sim, "init_Aw", (par.simN,))
        _alloc(sim, "init_Am", (par.simN,))
        _alloc(sim, "init_couple", (par.simN,), dtype=np.bool_)
        _alloc(sim, "init_power_idx", (par.simN,), dtype=np.int_)

        # --- e. other
        # timing
        _alloc(sol, "solution_time", (1,))

    def fill_allocations(self):
        """Fill all allocated arrays with their initial values (nan/inf/zeros) and draws/init states."""
        par = self.par
        sol = self.sol
        sim = self.sim

        def _fill(obj, names, value):
            for n in names:
                getattr(obj, n)[...] = value

        # ========= a. singles =========
        # a.1 single -> single (discrete)
        _fill(sol, ("Vwd_single_to_single", "Vmd_single_to_single"), -np.inf)
        _fill(sol, (
            "Cwd_tot_single_to_single", "Cmd_tot_single_to_single",
            "Cwd_priv_single_to_single", "Cmd_priv_single_to_single",
            "Cwd_inter_single_to_single", "Cmd_inter_single_to_single",
            "Qwd_single_to_single", "Qmd_single_to_single",
            "lwd_single_to_single", "lmd_single_to_single",
            "hwd_single_to_single", "hmd_single_to_single",
        ), np.nan)

        # a.1 EGM post-decision (singles)
        _fill(sol, (
            "EmargUwd_single_to_single_pd", "Cwd_tot_single_to_single_pd",
            "Mwd_single_to_single_pd", "Vwd_single_to_single_pd",
            "EmargUmd_single_to_single_pd", "Cmd_totm_single_to_single_pd",
            "Mmd_single_to_single_pd", "Vmd_single_to_single_pd",
        ), 0.0)

        # a.2 couple -> single
        _fill(sol, (
            "Vw_couple_to_single", "Vm_couple_to_single",
            "lw_couple_to_single", "lm_couple_to_single",
            "Cw_priv_couple_to_single", "Cm_priv_couple_to_single",
            "Cw_inter_couple_to_single", "Cm_inter_couple_to_single",
            "Cw_tot_couple_to_single", "Cm_tot_couple_to_single",
            "hw_couple_to_single", "hm_couple_to_single",
            "Qw_couple_to_single", "Qm_couple_to_single",
        ), np.nan)

        # a.3 start as single
        _fill(sol, ("EVw_start_as_single", "EVm_start_as_single"), -np.inf)
        _fill(sol, (
            "EmargVw_start_as_single", "EmargVm_start_as_single",
            "EVw_cond_meet_partner", "EVm_cond_meet_partner",
            "EVw_uncond_meet_partner", "EVm_uncond_meet_partner",
        ), np.nan)

        # a.4 optimal discrete choices (single)
        _fill(sol, ("Vw_single_to_single", "Vm_single_to_single"), -np.inf)
        _fill(sol, (
            "Cw_tot_single_to_single", "Cm_tot_single_to_single",
            "lw_single_to_single", "lm_single_to_single",
        ), np.nan)

        # ========= b. couples =========
        # b.1 couple -> couple (discrete)
        _fill(sol, ("Vwd_couple_to_couple", "Vmd_couple_to_couple"), np.nan)
        sol.Vd_couple_to_couple[...] = -np.inf
        _fill(sol, (
            "Cwd_priv_couple_to_couple", "Cmd_priv_couple_to_couple",
            "Cd_inter_couple_to_couple", "Qd_couple_to_couple",
            "lwd_couple_to_couple", "lmd_couple_to_couple",
            "hwd_couple_to_couple", "hmd_couple_to_couple",
            "Cd_tot_couple_to_couple",
        ), np.nan)

        sol.power_idx[...] = 0
        sol.power[...] = np.nan

        # b.1 EGM post-decision (couples)
        _fill(sol, ("EmargUd_pd", "Cd_tot_pd", "Md_pd", "Vd_couple_to_couple_pd"), 0.0)

        # b.2 single -> couple
        _fill(sol, ("Vw_single_to_couple", "Vm_single_to_couple"), np.nan)
        sol.V_single_to_couple[...] = -np.inf
        _fill(sol, (
            "lw_single_to_couple", "lm_single_to_couple",
            "Cw_priv_single_to_couple", "Cm_priv_single_to_couple",
            "hw_single_to_couple", "hm_single_to_couple",
            "C_inter_single_to_couple", "Q_single_to_couple",
            "Cw_tot_single_to_couple", "Cm_tot_single_to_couple",
        ), np.nan)

        # b.3 start as couple
        _fill(sol, (
            "Vw_start_as_couple", "Vm_start_as_couple", "margV_start_as_couple",
            "EVw_start_as_couple", "EVm_start_as_couple", "EmargV_start_as_couple",
            "C_tot_start_as_couple", "Cw_priv_start_as_couple", "Cm_priv_start_as_couple",
            "C_inter_start_as_couple", "Q_start_as_couple",
            "lw_start_as_couple", "lm_start_as_couple",
            "hw_start_as_couple", "hm_start_as_couple",
        ), np.nan)

        # b.4 optimal discrete choices (couple)
        sol.V_couple_to_couple[...] = -np.inf
        _fill(sol, (
            "Vw_couple_to_couple", "Vm_couple_to_couple",
            "C_tot_couple_to_couple",
            "lw_couple_to_couple", "lm_couple_to_couple",
        ), np.nan)

        # ========= c. precomputed intratemporal solution =========
        _fill(sol, (
            "pre_Cwd_priv_couple", "pre_Cmd_priv_couple",
            "pre_Cd_inter_couple", "pre_Qd_couple",
            "pre_hwd_couple", "pre_hmd_couple",
            "pre_Cwd_priv_single", "pre_Cmd_priv_single",
            "pre_Cwd_inter_single", "pre_Cmd_inter_single",
            "pre_Qwd_single", "pre_Qmd_single",
            "pre_hwd_single", "pre_hmd_single",
        ), np.nan)

        # ========= d. simulation =========
        # d.1 simulated outcomes (floats -> nan, ints -> 0)
        _fill(sim, (
            "lw", "lm", "Cw_priv", "Cm_priv", "hw", "hm",
            "Cw_inter", "Cm_inter", "Qw", "Qm",
            "Cw_tot", "Cm_tot", "C_tot",
            "Kw", "Km", "A", "Aw", "Am",
            "couple", "power", "love",
            "wage_w", "wage_m", "leisure_w", "leisure_m",
            "util",
        ), np.nan)

        sim.type_w[...] = -1000
        sim.type_m[...] = -1000
        sim.mean_lifetime_util[...] = np.nan

        # d.2 shocks (seed -> draws)
        np.random.seed(par.seed)
        shape_sim = (par.simN, par.simT)
        
        sim.draw_shock_Kw[...] = np.random.lognormal(size=shape_sim, mean=-0.5 * par.sigma_Kw**2, sigma=par.sigma_Kw)
        sim.draw_shock_Km[...] = np.random.lognormal(size=shape_sim, mean=-0.5 * par.sigma_Km**2, sigma=par.sigma_Km)
        sim.draw_love[...] = np.random.normal(size=shape_sim)
        sim.draw_meet[...] = np.random.uniform(size=shape_sim)

        sim.draw_uniform_partner_Kw[...] = np.random.uniform(size=shape_sim)
        sim.draw_uniform_partner_Km[...] = np.random.uniform(size=shape_sim)
        sim.draw_uniform_partner_Aw[...] = np.random.uniform(size=shape_sim)
        sim.draw_uniform_partner_Am[...] = np.random.uniform(size=shape_sim)
        sim.draw_uniform_partner_type_w[...] = np.random.uniform(size=shape_sim)
        sim.draw_uniform_partner_type_m[...] = np.random.uniform(size=shape_sim)

        sim.draw_repartner_love[...] = par.sigma_love * np.random.normal(0.0, 1.0, size=shape_sim)


        # d.3 initial distribution
        sim.init_A[...] = 0.0
        sim.init_Kw[...] = 0.0
        sim.init_Km[...] = 0.0
        sim.init_Aw[...] = sim.init_A * par.div_A_share
        sim.init_Am[...] = sim.init_A * (1.0 - par.div_A_share)
        sim.init_couple[...] = np.random.choice([True, False], par.simN, p=[par.init_couple_share, 1 - par.init_couple_share])
        sim.init_power_idx[...] = (par.num_power // 2)
        sim.init_love[...] = 0.0
        sim.init_type_w[...] = np.random.choice(par.num_types, par.simN, p=par.type_w_share)
        sim.init_type_m[...] = np.random.choice(par.num_types, par.simN, p=par.type_m_share)

        # ========= e. timing =========
        sol.solution_time[...] = 0.0

    def solve(self):

        sol = self.sol
        par = self.par

        # always ensure sizes/grids are up-to-date (needed for shape checks)
        self.setup_gender_parameters()
        self.setup_grids()

        # allocate once (or when shapes changed), otherwise just reset values
        shape_single_d = (par.T, par.num_types, par.num_l, par.num_K, par.num_A)
        if (not hasattr(sol, "Vwd_single_to_single")) or (sol.Vwd_single_to_single.shape != shape_single_d):
            self.allocate()
        else:
            self.fill_allocations()

        self.cpp.solve(sol, par)


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
            
    # Estimation
    def obj_func(self,theta,estpar,datamoms,weights=None,do_print=False):
        
        # update parameters, impose bounds and return penalty
        penalty = self.update_par(theta,estpar)
        self.setup_gender_parameters()
        self.setup_grids()
        
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
