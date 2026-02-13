#ifndef MAIN
#define SINGLE
#include "myheader.cpp"
#endif

namespace single {

    // Data passed to the single-agent objective solver
    struct SolverSingleData {
        int t;
        int il;
        int iS;      // NEW: type index
        double K;
        double M;
        double* EV_next;
        int gender;
        par_struct* par;
        sol_struct* sol;
    };

    // Small constant to avoid exact-zero resources
    static constexpr double RESOURCES_EPS = 1.0e-4;

    // Multistart factor in first multistart run
    static constexpr double MULTISTART_FACTOR = 0.5;

    // helper: run 1D multistart optimization given optimizer and bounds
    double run_multistart_optimizer_1d(nlopt_opt optimizer_handle, double initial_guess,
                            const double* lower_bounds, const double* upper_bounds,
                            int num_starts) {
        
        // first run from provided initial guess
        double x[1] = { initial_guess };
        double minf_global = 0.0;
        nlopt_optimize(optimizer_handle, x, &minf_global);
        double best_Ctot = x[0];

        // setup RNG for additional random starts if needed
        if (num_starts > 1) {
        static const int seed_rng = []() {
            std::srand(123456789u);
            return 0;
        }();
        (void)seed_rng;
        }

        // setup placeholder for local minimum
        double minf_local = 0.0;

        // additional starts
        for (int s = 0; s < num_starts; ++s) {
        if (s == 0) {
            x[0] = x[0] * MULTISTART_FACTOR; // try a lower starting value
        } else {
            double u_rand = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            x[0] = lower_bounds[0] + u_rand * (upper_bounds[0] - lower_bounds[0]);
        }

        nlopt_optimize(optimizer_handle, x, &minf_local);
        if (minf_local < minf_global) {
            best_Ctot = x[0];
            minf_global = minf_local;
        }
        }

        return best_Ctot;
    };

    double resources_single(double labor, double K, double A, int gender, par_struct* par) {
        if (labor == 0.0) return par->R * A + RESOURCES_EPS;

        double w = utils::wage(K, gender, par);
        return par->R * A + w * labor * par->available_hours * (1.0 - par->tax_rate);
    }

    double value_of_choice_single_to_single(double* C_priv, double* h, double* C_inter, double* Q,
                                            double C_tot, int t, int il, double K, double M, int gender,
                                            double* V_next, par_struct* par, sol_struct* sol) {
        const double labor = par->grid_l[il];

        // intraperiod allocation: fills C_priv, h, C_inter, Q
        precompute::intraperiod_allocation_single(C_priv, h, C_inter, Q, C_tot, il, gender, par, sol,
                                                par->precompute_intratemporal);

        double total_hours = *h + labor;
        double util = utils::util(*C_priv, total_hours, *Q, gender, par, 0.0);

        double continuation = 0.0;
        if (t < (par->T - 1)) {
            double* grid_A = (gender == man) ? par->grid_Am : par->grid_Aw;
            double* grid_K = (gender == man) ? par->grid_Km : par->grid_Kw;
            double A_next = M - C_tot;
            double K_next = utils::human_capital_transition(K, labor, par);
            continuation = tools::interp_2d(grid_K, grid_A, par->num_K, par->num_A, V_next, K_next, A_next);
        }

        return util + par->beta * continuation;
    }

    double objfunc_single_to_single(unsigned /*n*/, const double* x, double* /*grad*/, void* solver_data_in) {
        auto* data = static_cast<SolverSingleData*>(solver_data_in);

        double C_tot = x[0];
        int t = data->t;
        int il = data->il;
        int iS = data->iS;
        int gender = data->gender;
        double K = data->K;
        double M = data->M;
        double* EV_next = data->EV_next;
        par_struct* par = data->par;
        sol_struct* sol = data->sol;

        // intraperiod outputs
        double C_priv = 0.0, h = 0.0, C_inter = 0.0, Q = 0.0;

        // Clamp C_tot to (tiny, M)
        constexpr double C_TINY = 1.0e-12;
        if (C_tot < C_TINY) C_tot = C_TINY;
        if (C_tot > M) C_tot = M;

        // optimizer minimizes -> return negative value
        return -value_of_choice_single_to_single(&C_priv, &h, &C_inter, &Q,
                                                C_tot, t, il, K, M, gender,
                                                EV_next, par, sol);
    }

    void solve_single_to_single_step(
        double* Cd_priv, double* hd, double* Cd_inter, double* Qd, double* Vd,
        double M_resources, int t, int il, int iS, double K,
        double* EV_next,
        double starting_val, int gender, sol_struct* sol, par_struct* par
    ) {
        double Cd_tot = M_resources;

        if (t < (par->T - 1)) {
            // Setup solver data and optimizer
            SolverSingleData* solver_data = new SolverSingleData{t, il, iS, K, M_resources, EV_next, gender, par, sol};

            constexpr int dim = 1;
            double lb[dim], ub[dim];

            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);

            nlopt_set_min_objective(opt, objfunc_single_to_single, solver_data);

            // bounds
            lb[0] = 1.0e-8;
            ub[0] = solver_data->M;
            if (ub[0] < lb[0]) ub[0] = lb[0];
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            // clamp starting value
            if (starting_val < lb[0]) starting_val = lb[0];
            if (starting_val > ub[0]) starting_val = ub[0];

            Cd_tot = run_multistart_optimizer_1d(opt, starting_val, lb, ub, par->num_multistart);
            
            // cleanup
            nlopt_destroy(opt);
            delete solver_data;
        }

        // compute implied allocation and value
        *Vd = value_of_choice_single_to_single(Cd_priv, hd, Cd_inter, Qd, Cd_tot,
                                            t, il, K, M_resources, gender, EV_next, par, sol);
    }

    void solve_single_to_single_Agrid_vfi(int t, int il, int iS, int iK, double* EV_next, int gender, sol_struct* sol, par_struct* par) {
        const double labor = par->grid_l[il];

        // get index
        auto idx_d_A = index::single_d(t, il, iS, iK, 0, par);

        // pointers to storage (gender-specific)
        double* Cd_tot = &sol->Cwd_tot_single_to_single[idx_d_A];
        double* Cd_priv = &sol->Cwd_priv_single_to_single[idx_d_A];
        double* hd = &sol->hwd_single_to_single[idx_d_A];
        double* Cd_inter = &sol->Cwd_inter_single_to_single[idx_d_A];
        double* Qd = &sol->Qwd_single_to_single[idx_d_A];
        double* Vd = &sol->Vwd_single_to_single[idx_d_A];
        double* grid_A = par->grid_Aw;
        double* grid_K = par->grid_Kw;

        if (gender == man) {
            Cd_tot = &sol->Cmd_tot_single_to_single[idx_d_A];
            Cd_priv = &sol->Cmd_priv_single_to_single[idx_d_A];
            hd = &sol->hmd_single_to_single[idx_d_A];
            Cd_inter = &sol->Cmd_inter_single_to_single[idx_d_A];
            Qd = &sol->Qmd_single_to_single[idx_d_A];
            Vd = &sol->Vmd_single_to_single[idx_d_A];
            grid_A = par->grid_Am;
            grid_K = par->grid_Km;
        }

        for (int iA = 0; iA < par->num_A; iA++) {
            // resources depend on K and A
            const double K = grid_K[iK];
            const double A = grid_A[iA];
            double M_resources = resources_single(labor, K, A, gender, par);

            // starting value: previous solution or fraction of resources
            double starting_val = (iA > 0) ? Cd_tot[iA - 1] : (M_resources * 0.8);

            solve_single_to_single_step(&Cd_priv[iA], &hd[iA], &Cd_inter[iA], &Qd[iA], &Vd[iA],
                                    M_resources, t, il, iS, K, EV_next, starting_val, gender, sol, par);

            Cd_tot[iA] = Cd_priv[iA] + Cd_inter[iA];
        } // iA

    }

        //////////////////// EGM ////////////////////////
        ////// numerical EGM //////
    double marg_util_C_single(double C_tot, int il, int gender, par_struct* par, sol_struct* sol) {
        // closed form (precomputed) utility at C_tot and a small forward perturbation
        double util = precompute::util_C_single(C_tot, il, gender, par, sol, par->precompute_intratemporal);
        constexpr double delta = 1.0e-4;
        double util_delta = precompute::util_C_single(C_tot + delta, il, gender, par, sol, par->precompute_intratemporal);
        return (util_delta - util) / delta;
    }

    struct SolverInvData {
        int il;
        double margU;
        int gender;
        par_struct* par;
        sol_struct* sol;
    };

    double obj_inv_marg_util_single(unsigned /*n*/, const double* x, double* /*grad*/, void* solver_data_in) {
        auto* d = static_cast<SolverInvData*>(solver_data_in);

        double C_tot = x[0];
        int il = d->il;
        double margU = d->margU;
        int gender = d->gender;
        par_struct* par = d->par;
        sol_struct* sol = d->sol;

        // clip and penalize non-positive consumption
        double penalty = 0.0;
        if (C_tot <= 0.0) {
            penalty += 1000.0 * C_tot * C_tot;
            C_tot = 1.0e-6;
        }

        double diff = marg_util_C_single(C_tot, il, gender, par, sol) - margU;
        return diff * diff + penalty;
    }

    double inv_marg_util_single(double margU, int il, int gender, par_struct* par, sol_struct* sol,
                                double guess_C_tot) {
        SolverInvData* data = new SolverInvData{il, margU, gender, par, sol};

        constexpr int dim = 1;
        double lb[dim], ub[dim], x[dim];

        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);
        double minf = 0.0;

        nlopt_set_min_objective(opt, obj_inv_marg_util_single, data);
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, 1.0e-6);
        nlopt_set_xtol_rel(opt, 1.0e-5);

        lb[0] = 0.0;
        ub[0] = 2.0 * par->max_Ctot;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        x[0] = guess_C_tot;
        nlopt_optimize(opt, x, &minf);
        nlopt_destroy(opt);

        double result = x[0];
        delete data;
        return result;
    }


        /////// iEGM //////
    void interpolate_to_exogenous_grid_single(
        int t, int il, int iS,int iK, int gender,
        double* m_vec, double* c_vec, double* v_vec,
        double* C_tot, double* C_priv, double* h, double* C_inter, double* Q, double* V,
        double* EV_next, sol_struct* sol, par_struct* par
    ) {
        double labor = par->grid_l[il];
        double* grid_A = (gender == man) ? par->grid_Am : par->grid_Aw;
        double* grid_A_pd = (gender == man) ? par->grid_Am_pd : par->grid_Aw_pd;
        double* grid_K = (gender == man) ? par->grid_Km : par->grid_Kw;

        const double K = grid_K[iK];

        // Loop over the common (exogenous) asset grid
        for (int iA = 0; iA < par->num_A; iA++) {
            double M_now = resources_single(labor, K, grid_A[iA], gender, par);

            // If liquidity constraint binds, consume all resources
            if (M_now < m_vec[0]) {
                C_tot[iA] = M_now;
                V[iA] = value_of_choice_single_to_single(&C_priv[iA], &h[iA], &C_inter[iA], &Q[iA],
                                                        C_tot[iA], t, il, K, M_now, gender, EV_next, par, sol);
                continue;
            }

            // Otherwise, search for the correct interval in the endogenous grid
            for (int iA_pd = 0; iA_pd < par->num_A_pd - 1; iA_pd++) {
                double m_low = m_vec[iA_pd];
                double m_high = m_vec[iA_pd + 1];

                bool in_interval = (M_now >= m_low && M_now <= m_high);
                bool extrapolate_above = (iA_pd == par->num_A_pd - 2 && M_now > m_vec[par->num_A_pd - 1]);

                if (in_interval || extrapolate_above) {
                    // Endogenous asset grid points
                    double A_low = grid_A_pd[iA_pd];
                    double A_high = grid_A_pd[iA_pd + 1];

                    // Value and consumption at interval endpoints
                    double V_low = v_vec[iA_pd];
                    double V_high = v_vec[iA_pd + 1];
                    double c_low = c_vec[iA_pd];
                    double c_high = c_vec[iA_pd + 1];

                    // Slopes for interpolation
                    double v_slope = (V_high - V_low) / (A_high - A_low);
                    double c_slope = (c_high - c_low) / (m_high - m_low);

                    // Interpolate consumption and assets
                    double c_guess = c_low + c_slope * (M_now - m_low);
                    double a_guess = M_now - c_guess;
                    double V_guess = V_low + v_slope * (a_guess - A_low);

                    // upper envelope
                    if (V_guess > V[iA]) {
                        C_tot[iA] = c_guess;
                        V[iA] = value_of_choice_single_to_single(&C_priv[iA], &h[iA], &C_inter[iA], &Q[iA],
                                                                C_tot[iA], t, il, K, M_now, gender, EV_next, par, sol);
                    } // if upper envelope
                } // if in_interval
            } // iA_pd
        } // iA
    }


    void solve_single_to_single_Agrid_egm(int t, int il, int iS, int iK, int gender, sol_struct* sol, par_struct* par){
        // get index
        auto idx_A_pd  = index::single_pd(t, il, iS, iK, 0, par);
        auto idx_d_A   = index::single_d(t, il, iS, iK, 0, par);
        auto idx_next  = index::single(t + 1, iS, 0, 0, par);
        auto idx_interp = index::index2(il, 0, par->num_l, par->num_marg_u); // OBS: index::index3(il, iS, 0, par->num_l, par->num_marg_u);
        
        // 1. Setup: gender-specific pointers
        double* grid_A = par->grid_Aw;
        double* grid_A_pd = par->grid_Aw_pd;
        double* grid_K = par->grid_Kw;
        double* grid_marg_u_single_for_inv = par->grid_marg_u_single_w_for_inv;
        double* V = sol->Vwd_single_to_single;
        double* EV = sol->EVw_start_as_single;
        double* margV = sol->EmargVw_start_as_single;
        double* C_tot = sol->Cwd_tot_single_to_single;
        double* C_priv = sol->Cwd_priv_single_to_single;
        double* h = sol->hwd_single_to_single;
        double* C_inter = sol->Cwd_inter_single_to_single;
        double* Q = sol->Qwd_single_to_single;
        double* EmargU_pd = &sol->EmargUwd_single_to_single_pd[idx_A_pd];
        double* C_tot_pd = &sol->Cwd_tot_single_to_single_pd[idx_A_pd];
        double* M_pd = &sol->Mwd_single_to_single_pd[idx_A_pd];
        double* V_pd = &sol->Vwd_single_to_single_pd[idx_A_pd];

        if (gender == man) {
            grid_A = par->grid_Am;
            grid_A_pd = par->grid_Am_pd;
            grid_K = par->grid_Km;
            grid_marg_u_single_for_inv = par->grid_marg_u_single_m_for_inv;
            V = sol->Vmd_single_to_single;
            EV = sol->EVm_start_as_single;
            margV = sol->EmargVm_start_as_single;
            C_tot = sol->Cmd_tot_single_to_single;
            C_priv = sol->Cmd_priv_single_to_single;
            h = sol->hmd_single_to_single;
            C_inter = sol->Cmd_inter_single_to_single;
            Q = sol->Qmd_single_to_single;
            EmargU_pd = &sol->EmargUmd_single_to_single_pd[idx_A_pd];
            C_tot_pd = &sol->Cmd_totm_single_to_single_pd[idx_A_pd];
            M_pd = &sol->Mmd_single_to_single_pd[idx_A_pd];
            V_pd = &sol->Vmd_single_to_single_pd[idx_A_pd];
        }

        // placeholders 
        double C_priv_ph = 0;
        double h_ph = 0;
        double C_inter_ph = 0;
        double Q_ph = 0;


        // 2. EGM step
        int min_point_A = 0;
        double K = grid_K[iK];
        double K_next = utils::human_capital_transition(K, par->grid_l[il], par);
        int min_point_K = tools::binary_search(0, par->num_K, grid_K, K_next);

        for (int iA_pd = 0; iA_pd < par->num_A_pd; iA_pd++) {
            double A_next = grid_A_pd[iA_pd];

            // expected marginal utility
            min_point_A = tools::binary_search(min_point_A, par->num_A, grid_A, A_next);
            EmargU_pd[iA_pd] = par->beta * tools::_interp_2d(grid_K, grid_A, par->num_K, par->num_A, &margV[idx_next], K_next, A_next, min_point_K, min_point_A);

            // invert marginal utility
            if (strcmp(par->interp_method, "numerical") == 0) {
                double guess_C_tot = 3.0;
                if (iA_pd > 0) guess_C_tot = C_tot_pd[iA_pd - 1];
                C_tot_pd[iA_pd] = inv_marg_util_single(EmargU_pd[iA_pd], il, gender, par, sol, guess_C_tot);
            } else {
                if (strcmp(par->interp_method, "linear") == 0) {
                    C_tot_pd[iA_pd] = tools::interp_1d(&grid_marg_u_single_for_inv[idx_interp], par->num_marg_u, par->grid_inv_marg_u, EmargU_pd[iA_pd]);
                }
                if (par->interp_inverse) C_tot_pd[iA_pd] = 1.0 / C_tot_pd[iA_pd];
            }

            // endogenous grid over resources
            M_pd[iA_pd] = C_tot_pd[iA_pd] + A_next;

            // value at endogenous point
            V_pd[iA_pd] = value_of_choice_single_to_single(&C_priv_ph, &h_ph, &C_inter_ph, &Q_ph, C_tot_pd[iA_pd], t, il, K, M_pd[iA_pd], gender, &EV[idx_next], par, sol);
        }

        // 3. Apply liquidity constraint and upper envelope while interpolating onto common grid
        interpolate_to_exogenous_grid_single(
            t, il, iS, iK, gender,
            M_pd, C_tot_pd, V_pd,
            &C_tot[idx_d_A], &C_priv[idx_d_A], &h[idx_d_A], &C_inter[idx_d_A], &Q[idx_d_A], &V[idx_d_A],
            &EV[idx_next], sol, par
        );
    }

    int find_interpolated_labor_index_single(int t, int iS, double K, double A, int gender, sol_struct* sol, par_struct* par){

        //--- Set variables based on gender ---
        double* grid_A = (gender == woman) ? par->grid_Aw : par->grid_Am;
        double* grid_K = (gender == woman) ? par->grid_Kw : par->grid_Km;
        double* Vd_single_to_single = (gender == woman) ? sol->Vwd_single_to_single : sol->Vmd_single_to_single;

        // find nearest index once
        int iK = tools::binary_search(0, par->num_K, grid_K, K);
        int iA = tools::binary_search(0, par->num_A, grid_A, A);

        double maxV = -std::numeric_limits<double>::infinity();
        int labor_index = 0;

        //--- Loop over labor choices ---
        for (int il = 0; il < par->num_l; il++) {
            auto idx_d_A = index::single_d(t, il, iS, 0, 0, par);
            double V_now = tools::_interp_2d(grid_K, grid_A, par->num_K, par->num_A, &Vd_single_to_single[idx_d_A], K, A, iK, iA);
            if (V_now > maxV) {
                maxV = V_now;
                labor_index = il;
            }
        }

        return labor_index;
    }

    void calc_marginal_value_single_Agrid(double* V, double* margV, int gender, sol_struct* sol, par_struct* par)
    {
        double* grid_A = (gender == woman) ? par->grid_Aw : par->grid_Am;

        if (par->centered_gradient) {
            for (int iA = 1; iA <= par->num_A - 2; ++iA) {
                int iA_plus = iA + 1;
                int iA_minus = iA - 1;
                double denom = 1.0 / (grid_A[iA_plus] - grid_A[iA_minus]);
                margV[iA] = V[iA_plus] * denom - V[iA_minus] * denom;
                if (iA == par->num_A - 2) margV[iA_plus] = margV[iA];
            }

            margV[0] = (margV[2] - margV[1]) / (grid_A[2] - grid_A[1]) * (grid_A[0] - grid_A[1]) + margV[1];
            int i = par->num_A - 1;
            margV[i] = (margV[i - 2] - margV[i - 1]) / (grid_A[i - 2] - grid_A[i - 1]) * (grid_A[i] - grid_A[i - 1]) + margV[i - 1];
        } else {
            for (int iA = 0; iA <= par->num_A - 2; ++iA) {
                int iA_plus = iA + 1;
                double denom = 1.0 / (grid_A[iA_plus] - grid_A[iA]);
                margV[iA] = V[iA_plus] * denom - V[iA] * denom;
                if (iA == par->num_A - 2) margV[iA_plus] = margV[iA];
            }
        }
    }
    
    // void calc_marginal_value_single_Agrid_old(int t, int iS, int iK, int gender, sol_struct* sol, par_struct* par){

    //     // unpack
    //     int const &num_A = par->num_A;
    //     double* grid_K = (gender == woman) ? par->grid_Kw : par->grid_Km;
    //     double K = grid_K[iK];

    //     // set index
    //     auto idx_A = index::single(t, iS, iK, 0, par); // FIX: include iS

    //     // gender specific variables
    //     double* grid_A = par->grid_Aw;
    //     double* margV = &sol->EmargVw_start_as_single[idx_A];
    //     double* V      = &sol->Vwd_single_to_single[0];

    //     if (gender == man){
    //         grid_A = par->grid_Am;
    //         margV = &sol->EmargVm_start_as_single[idx_A];
    //         V      = &sol->Vmd_single_to_single[0];
    //     }


    //     // approximate marginal value by finite differences
    //     if (par->centered_gradient) {
    //         for (int iA = 1; iA < num_A - 1; iA++) {
    //             int iA_plus = iA + 1;
    //             int iA_minus = iA - 1;

    //             // find il
    //             int il = find_interpolated_labor_index_single(t, iS, K, grid_A[iA], gender, sol, par);
    //             auto idx_d       = index::single_d(t, il, iS, iK, iA, par);       // CHANGED
    //             auto idx_d_plus  = index::single_d(t, il, iS, iK, iA_plus, par);  // CHANGED
    //             auto idx_d_minus = index::single_d(t, il, iS, iK, iA_minus, par); // CHANGED
    //             double denom = 1.0 / (grid_A[iA_plus] - grid_A[iA_minus]);

    //             // Calculate finite difference
    //             margV[iA] = V[idx_d_plus] * denom - V[idx_d_minus] * denom;
    //         }
    //          // Extrapolate gradient in end points
    //         int i=0;
    //         margV[i] = (margV[i+2] - margV[i+1]) / (grid_A[i+2] - grid_A[i+1]) * (grid_A[i] - grid_A[i+1]) + margV[i+1];
    //         i = par->num_A-1;
    //         margV[i] = (margV[i-2] - margV[i-1]) / (grid_A[i-2] - grid_A[i-1]) * (grid_A[i] - grid_A[i-1]) + margV[i-1];
            
    //     } 
    //     else {
    //         for (int iA=0; iA<num_A-1; iA++){
    //             // Setup indices
    //             int iA_plus = iA + 1;

    //             // find il
    //             int il = find_interpolated_labor_index_single(t, iS, K, grid_A[iA], gender, sol, par);

    //             auto idx_d_A = index::single_d(t, il, iS, iK, 0, par); // already OK
    //             double* Vd = &V[idx_d_A];

    //             auto idx_d = index::single_d(t, il, iS, iK, iA, par);  // FIX: include iS
    //             (void)idx_d; // idx_d not used below; keep if you want consistency checks

    //             double delta = 1.0e-6;

    //             double denom = 1/delta;

    //             // calculate V_now
    //             double V_now = Vd[iA];

    //             // caluclate V_delta by interpolating V at A + delta
    //             double V_delta = tools::interp_1d_index(grid_A, par->num_A, Vd, grid_A[iA] + delta, iA);

    //             // Calculate finite difference
    //             margV[iA] = V_delta*denom - V_now* denom;

    //             // Extrapolate gradient in last point
    //             if (iA == num_A-2){
    //                 margV[iA_plus] = margV[iA];
    //             }
    //         }
    //     }
    // }

    void update_optimal_discrete_solution_single_Agrid(int t, int il, int iS, int iK, int gender, sol_struct* sol, par_struct* par){

        // get index
        auto idx_A = index::single(t, iS, iK, 0, par);
        auto idx_d_A = index::single_d(t, il, iS, iK, 0, par);

        // get variables
        double* V = &sol->Vw_single_to_single[idx_A];
        double* Vd = &sol->Vwd_single_to_single[idx_d_A];
        double* labor = &sol->lw_single_to_single[idx_A];
        if (gender == man) {
            V = &sol->Vm_single_to_single[idx_A];
            Vd = &sol->Vmd_single_to_single[idx_d_A];
            labor = &sol->lm_single_to_single[idx_A];
        }

        // Find maximum value over all labor choices
        for (int iA=0; iA<par->num_A; iA++){
            if (V[iA] < Vd[iA]) {
                V[iA] = Vd[iA];
                labor[iA] = par->grid_l[il];
            }
        }
    }


    

    void solve_choice_specific_single_to_single(int t, int il, int iS, int iK, int gender, sol_struct *sol, par_struct *par) {

        // Terminal period: no continuation value
        if (t == (par->T - 1)) {
            solve_single_to_single_Agrid_vfi(t, il, iS, iK, nullptr, gender, sol, par);
        } else {
            // next period expected value (used by VFI)
            const auto idx_next = index::single(t + 1, iS, 0, 0, par);
            double* const EV_next = (gender == man) ? &sol->EVm_start_as_single[idx_next] : &sol->EVw_start_as_single[idx_next];

            // Choose EGM or VFI method
            if (par->do_egm) {
                solve_single_to_single_Agrid_egm(t, il, iS, iK, gender, sol, par);
            } else {
                solve_single_to_single_Agrid_vfi(t, il, iS, iK, EV_next, gender, sol, par);
            }
        }

        // Update solution with optimal discrete labor choice
        update_optimal_discrete_solution_single_Agrid(t, il, iS, iK, gender, sol, par);
    }


    void solve_single_to_single(int t, sol_struct *sol,par_struct *par){
        // 1. solve choice specific
        #pragma omp parallel for collapse(3) num_threads(par->threads) // CHANGED collapse(2)->collapse(3)
        for (int iS = 0; iS < par->num_S; iS++) {                     // NEW loop
            for (int iK = 0; iK < par->num_K; iK++) {
                for (int sex = 0; sex < 2; sex++) {
                    for (int il = 0; il < par->num_l; il++) {
                        const int gender = (sex == 0) ? woman : man;
                        solve_choice_specific_single_to_single(t, il, iS, iK, gender, sol, par); // CHANGED
                    }
                }
            }
        }
    }

     double repartner_surplus(double power, index::state_couple_struct* state_couple, index::state_single_struct* state_single, int gender, par_struct* par, sol_struct* sol){
        // unpack index
        int t = state_single->t;
        int iS = state_single->iS;
        double A = state_single->A;
        double K = state_single->K;
        double love = state_couple->love;
        int iSw = state_couple->iSw;
        int iSm = state_couple->iSm;
        double Kw = state_couple->Kw;
        double Km = state_couple->Km;
        double A_tot = state_couple->A; 

        // gender specific
        double* V_single_to_single = sol->Vw_single_to_single;
        double* V_single_to_couple = sol->Vw_single_to_couple;
        double* grid_A_single = par->grid_Aw;
        double* grid_K_single = par->grid_Kw;
        if (gender == man){
            V_single_to_single = sol->Vm_single_to_single;
            V_single_to_couple = sol->Vm_single_to_couple;
            grid_A_single = par->grid_Am;
            grid_K_single = par->grid_Km;
        }
        
        // Get indices
        int iA_single = state_single->iA;
        int iK_single = state_single->iK; // OBS: return to this. Probably need to interpolate over K as well in single interpolation.
        int iL_couple = state_couple->iL;
        int iKw_couple = state_couple->iKw;
        int iKm_couple = state_couple->iKm;
        int iA_couple = state_couple->iA;
        int iP = tools::binary_search(0, par->num_power, par->grid_power, power);

        // compute missing indices lazily
        if (iL_couple == -1) iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        if (iKw_couple == -1) iKw_couple = tools::binary_search(0, par->num_K, par->grid_Kw, Kw);
        if (iKm_couple == -1) iKm_couple = tools::binary_search(0, par->num_K, par->grid_Km, Km);
        if (iA_couple == -1) iA_couple = tools::binary_search(0, par->num_A, par->grid_A, A_tot);
        if (iA_single == -1) iA_single = tools::binary_search(0, par->num_A, grid_A_single, A);

        //interpolate V_single_to_single
        auto idx_interp_single = index::single(t, iS, iK_single, 0, par);
        double Vsts = tools::interp_1d_index(grid_A_single, par->num_A, &V_single_to_single[idx_interp_single], A, iA_single);

        // interpolate couple V_single_to_couple
        auto idx_interp_couple = index::couple(t, 0, 0, 0, 0, 0, 0, 0, par); // OBS: can we do something else than interpolating over all dimensions here? Does this even work with S?
        double Sw = par->grid_S[iSw]; // OBS: I didn't mean to include S in the interpolation, but this is a quick fix for now. We can revisit how to handle S in the future.
        double Sm = par->grid_S[iSm];
        auto idx_interp_Sw = tools::binary_search(0, par->num_S, par->grid_S, Sw);
        auto idx_interp_Sm = tools::binary_search(0, par->num_S, par->grid_S, Sm);
        double Vstc = tools::_interp_7d_index(par->grid_power, par->grid_love, par->grid_S, par->grid_S, par->grid_Kw, par->grid_Km, par->grid_A,
            par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
            &V_single_to_couple[idx_interp_couple], power, love, Sw, Sm, Kw, Km, A_tot,
            iP, iL_couple, idx_interp_Sw, idx_interp_Sm, iKw_couple, iKm_couple, iA_couple);

        // surplus
        return Vstc - Vsts;
    }

    double calc_initial_bargaining_weight(int t, double love, int iSw, int iSm, double Kw, double Km, double Aw, double Am, sol_struct* sol, par_struct* par, int iL_couple=-1){
        // state structs
        index::state_couple_struct* state_couple = new index::state_couple_struct();
        index::state_single_struct* state_single_w = new index::state_single_struct();
        index::state_single_struct* state_single_m = new index::state_single_struct();

        // couple
        state_couple->t = t;
        state_couple->love = love;
        state_couple->Kw = Kw;
        state_couple->Km = Km;
        state_couple->A = Aw + Am;
        state_couple->iKw = tools::binary_search(0, par->num_K, par->grid_Kw, Kw);
        state_couple->iKm = tools::binary_search(0, par->num_K, par->grid_Km, Km);
        state_couple->iA = tools::binary_search(0, par->num_A, par->grid_A, Aw + Am);
        if (iL_couple == -1) iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        state_couple->iL = iL_couple;

        // single woman
        state_single_w->t = t;
        state_single_w->iS = iSw;
        state_single_w->K = Kw;
        state_single_w->A = Aw;
        state_single_w->iA = tools::binary_search(0, par->num_A, par->grid_Aw, Aw);

        // single man
        state_single_m->t = t;
        state_single_m->iS = iSm;
        state_single_m->K = Km;
        state_single_m->A = Am;
        state_single_m->iA = tools::binary_search(0, par->num_A, par->grid_Am, Am);

        // solver input
        bargaining::nash_solver_struct* nash_struct = new bargaining::nash_solver_struct();
        nash_struct->surplus_func = repartner_surplus;
        nash_struct->state_couple = state_couple;
        nash_struct->state_single_w = state_single_w;
        nash_struct->state_single_m = state_single_m;
        nash_struct->sol = sol;
        nash_struct->par = par;

        // solve
        double init_mu = bargaining::nash_bargain(nash_struct);

        // clean up
        delete state_couple;
        delete state_single_w;
        delete state_single_m;
        delete nash_struct;

        return init_mu;
    }
    
    
    double expected_value_cond_not_meet_partner(int t, int iS, int iK, int iA, int gender, sol_struct* sol, par_struct* par){ // NEW iS
        auto idx = index::single(t, iS, iK, iA, par); // CHANGED
        double* V_single_to_single = (gender == man) ? sol->Vm_single_to_single : sol->Vw_single_to_single;
        return V_single_to_single[idx];
    
    }
    
        double expected_value_cond_meet_partner(int t, int iS, int iK, int iA, int gender, sol_struct* sol, par_struct* par){ // NEW iS
        // unpack
        double* V_single_to_single = sol->Vw_single_to_single;
        double* V_single_to_couple = sol->Vw_single_to_couple;
        double* prob_partner_A = par->prob_partner_A_w;
        double* prob_partner_K = par->prob_partner_Kw;
        double* grid_A = par->grid_Aw;
        double* grid_K = par->grid_Kw;
        if (gender == man){
            V_single_to_single = sol->Vm_single_to_single;
            V_single_to_couple = sol->Vm_single_to_couple;
            prob_partner_A = par->prob_partner_A_m;
            prob_partner_K = par->prob_partner_Km;
            grid_A = par->grid_Am;
            grid_K = par->grid_Km;
        }
        // // value of remaining single
        auto idx_single = index::single(t, iS, iK, iA, par); // CHANGED

        // loop over potential partners conditional on meeting a partner
        double Ev_cond = 0.0;
        for (int iL = 0; iL < par->num_love; iL++) {
            const double prob_love = par->prob_partner_love[iL];
            if (prob_love <= 0.0) continue;

            for (int iSp = 0; iSp < par->num_S; iSp++) { // partner's type
                // OBS: insert things here later

                for (int iKp = 0; iKp < par->num_K; iKp++) { // partner's capital
                    auto idx_Kgrid = index::index2(iK, iKp, par->num_K, par->num_K);
                    const double prob_K = prob_partner_K[idx_Kgrid];
                    if (prob_K <= 0.0) continue;

                    const double love = par->grid_love[iL];
                    for (int iAp = 0; iAp < par->num_A; iAp++) { // partner's wealth
                        auto idx_Agrid = index::index2(iA, iAp, par->num_A, par->num_A);
                        const double prob_A = prob_partner_A[idx_Agrid];
                        if (prob_A <= 0.0) continue;

                        const double prob = prob_A * prob_K * prob_love;

                        // only calculate if match has positive probability of happening
                        if (prob>0.0) {
                            int iAw = iA;
                            int iAm = iAp;
                            int iKw = iK;
                            int iKm = iKp;
                            int iSw = iS;
                            int iSm = iSp;
                            if (gender == man) {
                                iAw = iAp;
                                iAm = iA;
                                iKw = iKp;
                                iKm = iK;
                                iSw = iSp;
                                iSm = iS;
                            }

                            // meet person with same level of wealth and human capital
                            const double Aw = grid_A[iAw];
                            const double Am = grid_A[iAm];
                            const double Kw = grid_K[iKw]; 
                            const double Km = grid_K[iKm];

                            double power = calc_initial_bargaining_weight(t, love, iSw, iSm, Kw, Km, Aw, Am, sol, par, iL);

                            double val;
                            if (power >= 0.0) {
                                double A_tot = Aw + Am;
                                auto idx_interp_couple = index::couple(t, 0, 0, 0, 0, 0, 0, 0, par); // OBS: Does interpolation work with iS?
                                double Sw = par->grid_S[iSw]; // OBS: I didn't mean to include S in the interpolation, but this is a quick fix for now. We can revisit how to handle S in the future.
                                double Sm = par->grid_S[iSm];
                                val = tools::_interp_7d(par->grid_power, par->grid_love, par->grid_S, par->grid_S, par->grid_Kw, par->grid_Km, par->grid_A,
                                                    par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
                                                    &V_single_to_couple[idx_interp_couple], power, love, Sw, Sm, Kw, Km, A_tot);
                                // OBS: actually onlu interpolation in power and A_tot is needed here
                            } else {
                                val = V_single_to_single[idx_single];
                            }

                            Ev_cond += prob * val;
                        } // if prob>0
                    } // iAp
                } // iKp
            } // iSp
        } // iL
        return Ev_cond;
    }


    void expected_value_start_single_Agrid(int t, int iS, int iK, int gender, sol_struct* sol,par_struct* par){
        auto idx_A = index::single(t, iS, iK, 0, par);
        // get variables
        double* EV_start_as_single = (gender == man) ? &sol->EVm_start_as_single[idx_A] : &sol->EVw_start_as_single[idx_A];
        double* EV_cond_meet_partner = (gender == man) ? &sol->EVm_cond_meet_partner[idx_A] : &sol->EVw_cond_meet_partner[idx_A];
        double* V_single_to_single = (gender == man) ? &sol->Vm_single_to_single[idx_A] : &sol->Vw_single_to_single[idx_A];
        double* EmargV_start_as_single = (gender == man) ? &sol->EmargVm_start_as_single[idx_A] : &sol->EmargVw_start_as_single[idx_A];

        const double p_meet = par->prob_repartner[t];
        const bool no_meet = (par->p_meet == 0.0);
        for (int iA = 0; iA < par->num_A; iA++) {
            if (no_meet) {
                // expected value of starting single is just value of remaining single
                EV_start_as_single[iA] = V_single_to_single[iA];
                EV_cond_meet_partner[iA] = V_single_to_single[iA];
                continue;
            }

            // Value conditional on meeting partner
            double EV_cond = expected_value_cond_meet_partner(t, iS, iK, iA, gender, sol, par);

            // expected value of starting single
            EV_start_as_single[iA] = p_meet * EV_cond + (1.0 - p_meet) * V_single_to_single[iA];
            EV_cond_meet_partner[iA] = EV_cond;
        }

        
    }


    
    void calc_expected_value_single(int t, int iS, int iK, int iA, int gender, double* V, double* EV, sol_struct* /*sol*/, par_struct* par)
    {

        auto idx = index::single(t, iS, iK, iA, par); // CHANGED

        double* grid_K = (gender == woman) ? par->grid_Kw : par->grid_Km;
        double* grid_shock_K = (gender == woman) ? par->grid_shock_Kw : par->grid_shock_Km;
        double* grid_weight_K = (gender == woman) ? par->grid_weight_Kw : par->grid_weight_Km;

        double Eval = 0.0;
        double delta_K = index::single(t, iS, 1, iA, par) - index::single(t, iS, 0, iA, par); // CHANGED

        for (int iK_shock = 0; iK_shock < par->num_shock_K; ++iK_shock) {
            double K_shock = grid_shock_K[iK_shock] * grid_K[iK];
            double weight_K = grid_weight_K[iK_shock];
            auto idx_K = tools::binary_search(0, par->num_K, grid_K, K_shock);
                
            auto idx_interp = index::single(t, iS, 0, iA, par); // CHANGED
            double V_now = tools::interp_1d_index_delta(grid_K, par->num_K, &V[idx_interp], K_shock, idx_K, delta_K);
            
            double weight = weight_K;
            Eval += weight * V_now;
        }

        EV[idx] = Eval;
    }


    void expected_value_start_single(int t, sol_struct* sol, par_struct* par){

        for (int sex = 0; sex < 2; sex++) {
            int gender = (sex == 0) ? woman : man;
            
            // get variables
            double* EV_start_as_single = (gender == man) ? sol->EVm_start_as_single : sol->EVw_start_as_single;
            double* EV_cond_meet_partner = (gender == man) ? sol->EVm_cond_meet_partner : sol->EVw_cond_meet_partner;
            double* EV_uncond_meet_partner = (gender == man) ? sol->EVm_uncond_meet_partner : sol->EVw_uncond_meet_partner;
            double* V_single_to_single = (gender == man) ? sol->Vm_single_to_single : sol->Vw_single_to_single;
            double* EmargV_start_as_single = (gender == man) ? sol->EmargVm_start_as_single : sol->EmargVw_start_as_single;
            
            const double p_meet = par->prob_repartner[t];
            const bool repartnering = (par->p_meet > 0.0);
            double EV_uncondtitional;
            
            #pragma omp parallel for collapse(3) num_threads(par->threads) // CHANGED
            for (int iS = 0; iS < par->num_S; iS++) {                     // NEW
                for (int iK = 0; iK < par->num_K; iK++) {
                    for (int iA = 0; iA < par->num_A; iA++) {
                        auto idx = index::single(t, iS, iK, iA, par); // CHANGED

                        double EV_cond_not_meet = expected_value_cond_not_meet_partner(t, iS, iK, iA, gender, sol, par); // CHANGED

                        if (repartnering) {
                            double EV_cond_meet = expected_value_cond_meet_partner(t, iS, iK, iA, gender, sol, par); // CHANGED
                            EV_cond_meet_partner[idx] = EV_cond_meet;
                            EV_uncond_meet_partner[idx] = p_meet * EV_cond_meet + (1.0 - p_meet) * EV_cond_not_meet;
                        } else {
                            EV_uncond_meet_partner[idx] = EV_cond_not_meet;
                        }
                    }
                }
            }

            // apply quadrature weights + marginal values (type-specific slices)
            for (int iS = 0; iS < par->num_S; iS++) { // NEW
                for (int iK = 0; iK < par->num_K; iK++) {
                    for (int iA = 0; iA < par->num_A; iA++) {
                        calc_expected_value_single(t, iS, iK, iA, gender, EV_uncond_meet_partner, EV_start_as_single, sol, par); // CHANGED
                    }

                    if (par->do_egm){
                        auto idx_A = index::single(t, iS, iK, 0, par); // CHANGED
                        calc_marginal_value_single_Agrid(&EV_start_as_single[idx_A], &EmargV_start_as_single[idx_A], gender, sol, par);
                    }
                }
            }
        }
    }

    void solve_couple_to_single(int t, sol_struct *sol, par_struct *par) {
        const double div_cost = par->div_cost;

        for (int iS = 0; iS < par->num_S; iS++) {          // NEW
            for (int iK = 0; iK < par->num_K; iK++) {
                auto idx_A = index::single(t, iS, iK, 0, par); // CHANGED

                double* Vw_couple_to_single = &sol->Vw_couple_to_single[idx_A];
                double* Vm_couple_to_single = &sol->Vm_couple_to_single[idx_A];
                double* Vw_single_to_single = &sol->Vw_single_to_single[idx_A];
                double* Vm_single_to_single = &sol->Vm_single_to_single[idx_A];
                double* lw_couple_to_single = &sol->lw_couple_to_single[idx_A];
                double* lm_couple_to_single = &sol->lm_couple_to_single[idx_A];
                double* lw_single_to_single = &sol->lw_single_to_single[idx_A];
                double* lm_single_to_single = &sol->lm_single_to_single[idx_A];

                for (int iA = 0; iA < par->num_A; iA++) {
                    Vw_couple_to_single[iA] = Vw_single_to_single[iA] - div_cost;
                    Vm_couple_to_single[iA] = Vm_single_to_single[iA] - div_cost;
                    lw_couple_to_single[iA] = lw_single_to_single[iA];
                    lm_couple_to_single[iA] = lm_single_to_single[iA];
                }
            }
        }
    }

} // namespace single
