#ifndef MAIN
#define SINGLE
#include "myheader.cpp"
#endif

namespace single {

    // Data passed to the single-agent objective solver
    struct SolverSingleData {
        int t;
        int il;
        double M;
        double* EV_next;
        int gender;
        par_struct* par;
        sol_struct* sol;
    };

    // Small constant to avoid exact-zero resources
    static constexpr double RESOURCES_EPS = 1.0e-4;

    double resources_single(double labor, double A, int gender, par_struct* par) {
        if (labor == 0.0) return par->R * A + RESOURCES_EPS;

        const double K = 5.0; // wage calibration constant used by utils::wage
        double w = utils::wage(K, gender, par);
        return par->R * A + w * labor;
    }

    double value_of_choice_single_to_single(double* C_priv, double* h, double* C_inter, double* Q,
                                            double C_tot, int t, int il, double M, int gender,
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
            double A = M - C_tot;
            continuation = tools::interp_1d(grid_A, par->num_A, V_next, A);
        }

        return util + par->beta * continuation;
    }

    double objfunc_single_to_single(unsigned /*n*/, const double* x, double* /*grad*/, void* solver_data_in) {
        auto* data = static_cast<SolverSingleData*>(solver_data_in);

        double C_tot = x[0];
        int t = data->t;
        int il = data->il;
        int gender = data->gender;
        double M = data->M;
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
                                                C_tot, t, il, M, gender,
                                                data->EV_next, par, sol);
    }

    void solve_single_to_single_step(
        double* Cd_priv, double* hd, double* Cd_inter, double* Qd, double* Vd,
        double M_resources, int t, int il,
        double* EV_next,
        double starting_val, int gender, sol_struct* sol, par_struct* par
    ) {
        double Cd_tot = M_resources;

        if (t < (par->T - 1)) {
            // pack solver data
            SolverSingleData* solver_data = new SolverSingleData{t, il, M_resources, EV_next, gender, par, sol};

            constexpr int dim = 1;
            double lb[dim], ub[dim], x[dim];

            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);
            double minf = 0.0;

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
            x[0] = starting_val;

            nlopt_optimize(opt, x, &minf);
            nlopt_destroy(opt);

            // unpack results
            Cd_tot = x[0];

            delete solver_data;
        }

        // compute implied allocation and value
        *Vd = value_of_choice_single_to_single(Cd_priv, hd, Cd_inter, Qd, Cd_tot,
                                            t, il, M_resources, gender, EV_next, par, sol);
    }

    void solve_single_to_single_Agrid_vfi(int t, int il, double* EV_next, int gender, sol_struct* sol, par_struct* par) {
        const double labor = par->grid_l[il];

        // get index
        auto idx_d_A = index::single_d(t, il, 0, par);

        // pointers to storage (gender-specific)
        double* Cd_tot = &sol->Cwd_tot_single_to_single[idx_d_A];
        double* Cd_priv = &sol->Cwd_priv_single_to_single[idx_d_A];
        double* hd = &sol->hwd_single_to_single[idx_d_A];
        double* Cd_inter = &sol->Cwd_inter_single_to_single[idx_d_A];
        double* Qd = &sol->Qwd_single_to_single[idx_d_A];
        double* Vd = &sol->Vwd_single_to_single[idx_d_A];
        double* grid_A = par->grid_Aw;

        if (gender == man) {
            Cd_tot = &sol->Cmd_tot_single_to_single[idx_d_A];
            Cd_priv = &sol->Cmd_priv_single_to_single[idx_d_A];
            hd = &sol->hmd_single_to_single[idx_d_A];
            Cd_inter = &sol->Cmd_inter_single_to_single[idx_d_A];
            Qd = &sol->Qmd_single_to_single[idx_d_A];
            Vd = &sol->Vmd_single_to_single[idx_d_A];
            grid_A = par->grid_Am;
        }

        for (int iA = 0; iA < par->num_A; iA++) {
            double M_resources = resources_single(labor, grid_A[iA], gender, par);

            // starting value: previous solution or fraction of resources
            double starting_val = (iA > 0) ? Cd_tot[iA - 1] : (M_resources * 0.8);

            solve_single_to_single_step(&Cd_priv[iA], &hd[iA], &Cd_inter[iA], &Qd[iA], &Vd[iA],
                                    M_resources, t, il, EV_next, starting_val, gender, sol, par);

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
        int t, int il, int gender,
        double* m_vec, double* c_vec, double* v_vec,
        double* C_tot, double* C_priv, double* h, double* C_inter, double* Q, double* V,
        double* EV_next, sol_struct* sol, par_struct* par
    ) {
        double labor = par->grid_l[il];
        double* grid_A = (gender == man) ? par->grid_Am : par->grid_Aw;
        double* grid_A_pd = (gender == man) ? par->grid_Am_pd : par->grid_Aw_pd;

        // Loop over the common (exogenous) asset grid
        for (int iA = 0; iA < par->num_A; iA++) {
            double M_now = resources_single(labor, grid_A[iA], gender, par);

            // If liquidity constraint binds, consume all resources
            if (M_now < m_vec[0]) {
                C_tot[iA] = M_now;
                V[iA] = value_of_choice_single_to_single(&C_priv[iA], &h[iA], &C_inter[iA], &Q[iA],
                                                        C_tot[iA], t, il, M_now, gender, EV_next, par, sol);
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
                                                                C_tot[iA], t, il, M_now, gender, EV_next, par, sol);
                    } // if upper envelope
                } // if in_interval
            } // iA_pd
        } // iA
    }


    void solve_single_to_single_Agrid_egm(int t, int il, int gender, sol_struct* sol, par_struct* par){
        // 1. Setup: gender-specific pointers
        double* grid_A = par->grid_Aw;
        double* grid_A_pd = par->grid_Aw_pd;
        double* grid_marg_u_single_for_inv = par->grid_marg_u_single_w_for_inv;
        double* V = sol->Vwd_single_to_single;
        double* EV = sol->EVw_start_as_single;
        double* margV = sol->EmargVw_start_as_single;
        double* C_tot = sol->Cwd_tot_single_to_single;
        double* C_priv = sol->Cwd_priv_single_to_single;
        double* h = sol->hwd_single_to_single;
        double* C_inter = sol->Cwd_inter_single_to_single;
        double* Q = sol->Qwd_single_to_single;
        double* EmargU_pd = sol->EmargUwd_single_to_single_pd;
        double* C_tot_pd = sol->Cwd_tot_single_to_single_pd;
        double* M_pd = sol->Mwd_single_to_single_pd;
        double* V_pd = sol->Vwd_single_to_single_pd;

        if (gender == man) {
            grid_A = par->grid_Am;
            grid_A_pd = par->grid_Am_pd;
            grid_marg_u_single_for_inv = par->grid_marg_u_single_m_for_inv;
            V = sol->Vmd_single_to_single;
            EV = sol->EVm_start_as_single;
            margV = sol->EmargVm_start_as_single;
            C_tot = sol->Cmd_tot_single_to_single;
            C_priv = sol->Cmd_priv_single_to_single;
            h = sol->hmd_single_to_single;
            C_inter = sol->Cmd_inter_single_to_single;
            Q = sol->Qmd_single_to_single;
            EmargU_pd = sol->EmargUmd_single_to_single_pd;
            C_tot_pd = sol->Cmd_totm_single_to_single_pd;
            M_pd = sol->Mmd_single_to_single_pd;
            V_pd = sol->Vmd_single_to_single_pd;
        }

        // placeholders 
        double C_priv_ph = 0;
        double h_ph = 0;
        double C_inter_ph = 0;
        double Q_ph = 0;


        // 2. EGM step
        auto idx_d_A = index::single_d(t, il, 0, par);
        auto idx_A_next = index::single(t + 1, 0, par);
        auto idx_interp = index::index2(il, 0, par->num_l, par->num_marg_u);
        int min_point_A = 0;

        for (int iA_pd = 0; iA_pd < par->num_A_pd; iA_pd++) {
            double A_next = grid_A_pd[iA_pd];

            // expected marginal utility
            min_point_A = tools::binary_search(min_point_A, par->num_A, grid_A, A_next);
            EmargU_pd[iA_pd] = par->beta * tools::interp_1d_index(grid_A, par->num_A, &margV[idx_A_next], A_next, min_point_A);

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
            V_pd[iA_pd] = value_of_choice_single_to_single(&C_priv_ph, &h_ph, &C_inter_ph, &Q_ph, C_tot_pd[iA_pd], t, il, M_pd[iA_pd], gender, &EV[idx_A_next], par, sol);
        }

        // 3. Apply liquidity constraint and upper envelope while interpolating onto common grid
        interpolate_to_exogenous_grid_single(t, il, gender, M_pd, C_tot_pd, V_pd, &C_tot[idx_d_A], &C_priv[idx_d_A], &h[idx_d_A], &C_inter[idx_d_A], &Q[idx_d_A], &V[idx_d_A], &EV[idx_A_next], sol, par);
    }


    void calc_marginal_value_single_Agrid(int t, int gender, sol_struct* sol, par_struct* par){

        // unpack
        int const &num_A = par->num_A;

        // set index
        auto idx_A = index::single(t,0,par);

        // gender specific variables
        double* grid_A = par->grid_Aw;
        double* margV = &sol->EmargVw_start_as_single[idx_A];
        double* V      = &sol->EVw_start_as_single[idx_A];

        if (gender == man){
            grid_A = par->grid_Am;
            margV = &sol->EmargVm_start_as_single[idx_A];
            V      = &sol->EVm_start_as_single[idx_A];
        }

        // approximate marginal value by finite differences
        if (par->centered_gradient) {
            for (int iA = 1; iA < num_A - 1; iA++) {
                int iA_plus = iA + 1;
                int iA_minus = iA - 1;

                double denom = 1.0 / (grid_A[iA_plus] - grid_A[iA_minus]);

                // Calculate finite difference
                margV[iA] = V[iA_plus] * denom - V[iA_minus] * denom;
            }
             // Extrapolate gradient in end points
            int i=0;
            margV[i] = (margV[i+2] - margV[i+1]) / (grid_A[i+2] - grid_A[i+1]) * (grid_A[i] - grid_A[i+1]) + margV[i+1];
            i = par->num_A-1;
            margV[i] = (margV[i-2] - margV[i-1]) / (grid_A[i-2] - grid_A[i-1]) * (grid_A[i] - grid_A[i-1]) + margV[i-1];
            
        } 
        else {
            for (int iA=0; iA<num_A-1; iA++){
                // Setup indices
                int iA_plus = iA + 1;

                double denom = 1/(grid_A[iA_plus] - grid_A[iA]);

                // Calculate finite difference
                margV[iA] = V[iA_plus]*denom - V[iA]* denom; 

                // Extrapolate gradient in last point
                if (iA == num_A-2){
                    margV[iA_plus] = margV[iA];
                }
            }
        }
    }

    void update_optimal_discrete_solution_single_Agrid(int t, int il, int gender, sol_struct* sol, par_struct* par){

        // get index
        auto idx_A = index::single(t,0,par);
        auto idx_d_A = index::single_d(t,il,0,par);

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


    

    void solve_choice_specific_single_to_single(int t, int il, int gender, sol_struct *sol, par_struct *par) {

        // Terminal period: no continuation value
        if (t == (par->T - 1)) {
            solve_single_to_single_Agrid_vfi(t, il, nullptr, gender, sol, par);
        } else {
            // next period expected value (used by VFI)
            const auto idx_A_next = index::single(t + 1, 0, par);
            double* const EV_next = (gender == man) ? &sol->EVm_start_as_single[idx_A_next] : &sol->EVw_start_as_single[idx_A_next];

            // Choose EGM or VFI method
            if (par->do_egm) {
                solve_single_to_single_Agrid_egm(t, il, gender, sol, par);
            } else {
                solve_single_to_single_Agrid_vfi(t, il, EV_next, gender, sol, par);
            }
        }

        // Update solution with optimal discrete labor choice
        update_optimal_discrete_solution_single_Agrid(t, il, gender, sol, par);
    }


    void solve_single_to_single(int t, sol_struct *sol,par_struct *par){
        // 1. solve choice specific
        #pragma omp parallel for collapse(2) num_threads(par->threads)
        for (int sex = 0; sex < 2; sex++) {
            for (int il = 0; il < par->num_l; il++) {
                const int gender = (sex == 0) ? woman : man;
                solve_choice_specific_single_to_single(t, il, gender, sol, par);
            }
        }
    }

     double repartner_surplus(double power, index::state_couple_struct* state_couple, index::state_single_struct* state_single, int gender, par_struct* par, sol_struct* sol){
        // unpack index
        int t = state_single->t;
        double A = state_single->A;
        double love = state_couple->love;
        double A_tot = state_couple->A; 

        // gender specific
        double* V_single_to_single = sol->Vw_single_to_single;
        double* V_single_to_couple = sol->Vw_single_to_couple;
        double* grid_A_single = par->grid_Aw;
        if (gender == man){
            V_single_to_single = sol->Vm_single_to_single;
            V_single_to_couple = sol->Vm_single_to_couple;
            grid_A_single = par->grid_Am;
        }
        
        // Get indices
        int iA_single = state_single->iA;
        int iL_couple = state_couple->iL;
        int iA_couple = state_couple->iA;
        int iP = tools::binary_search(0, par->num_power, par->grid_power, power);

        // compute missing indices lazily
        if (iL_couple == -1) iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        if (iA_couple == -1) iA_couple = tools::binary_search(0, par->num_A, par->grid_A, A_tot);
        if (iA_single == -1) iA_single = tools::binary_search(0, par->num_A, grid_A_single, A);

        //interpolate V_single_to_single
        auto idx_interp_single = index::single(t,0,par);
        double Vsts = tools::interp_1d_index(grid_A_single, par->num_A, &V_single_to_single[idx_interp_single], A, iA_single);

        // interpolate couple V_single_to_couple  
        auto idx_interp_couple = index::couple(t,0,0,0,par);
        double Vstc = tools::_interp_3d(par->grid_power, par->grid_love, par->grid_A,
                        par->num_power, par->num_love, par->num_A,
                        &V_single_to_couple[idx_interp_couple], power, love, A_tot,
                        iP, iL_couple, iA_couple);

        // surplus
        return Vstc - Vsts;
    }

    double calc_initial_bargaining_weight(int t, double love, double Aw, double Am, sol_struct* sol, par_struct* par, int iL_couple=-1){
        // state structs
        index::state_couple_struct* state_couple = new index::state_couple_struct();
        index::state_single_struct* state_single_w = new index::state_single_struct();
        index::state_single_struct* state_single_m = new index::state_single_struct();

        // couple
        state_couple->t = t;
        state_couple->love = love;
        state_couple->A = Aw + Am;
        state_couple->iA = tools::binary_search(0, par->num_A, par->grid_A, Aw + Am);
        if (iL_couple == -1) iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        state_couple->iL = iL_couple;

        // single woman
        state_single_w->t = t;
        state_single_w->A = Aw;
        state_single_w->iA = tools::binary_search(0, par->num_A, par->grid_Aw, Aw);

        // single man
        state_single_m->t = t;
        state_single_m->A = Am;
        state_single_m->iA = tools::binary_search(0, par->num_A, par->grid_Am, Am);
        // Note: We don't know whether we are on the woman or man asset grid, so we need to search both.
        // We could pass gender to calc_initial_bargaining_weight to infer which grid we are on, and avoid binary search for that gender

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
    
    
    double expected_value_cond_meet_partner(int t, int iA, int gender, sol_struct* sol, par_struct* par){
        // unpack
        double* V_single_to_single = sol->Vw_single_to_single;
        double* V_single_to_couple = sol->Vw_single_to_couple;
        double* prob_partner_A = par->prob_partner_A_w;
        double* grid_A = par->grid_Aw;
        if (gender == man){
            V_single_to_single = sol->Vm_single_to_single;
            V_single_to_couple = sol->Vm_single_to_couple;
            prob_partner_A = par->prob_partner_A_m;
            grid_A = par->grid_Am;
        }
        // // value of remaining single
        auto idx_single = index::single(t, iA, par);

        // loop over potential partners conditional on meeting a partner
        double Ev_cond = 0.0;
        for (int iL = 0; iL < par->num_love; iL++) {
            const double prob_love = par->prob_partner_love[iL];
            if (prob_love <= 0.0) continue;

            const double love = par->grid_love[iL];
            for (int iAp = 0; iAp < par->num_A; iAp++) { // partner's wealth
                auto idx_Agrid = index::index2(iA, iAp, par->num_A, par->num_A);
                const double prob_A = prob_partner_A[idx_Agrid];
                if (prob_A <= 0.0) continue;

                const double prob = prob_A * prob_love;

                // only calculate if match has positive probability of happening
                if (prob>0.0) {
                    int iAw = iA;
                    int iAm = iAp;
                    if (gender == man) {
                        iAw = iAp;
                        iAm = iA;
                    }

                    const double Aw = grid_A[iAw];
                    const double Am = grid_A[iAm];

                    double power = calc_initial_bargaining_weight(t, love, Aw, Am, sol, par, iL);

                    double val;
                    if (power >= 0.0) {
                        double A_tot = Aw + Am;
                        auto idx_interp_couple = index::couple(t, 0, 0, 0, par);
                        val = tools::interp_3d(par->grid_power, par->grid_love, par->grid_A,
                                            par->num_power, par->num_love, par->num_A,
                                            &V_single_to_couple[idx_interp_couple], power, love, A_tot);
                    } else {
                        val = V_single_to_single[idx_single];
                    }

                    Ev_cond += prob * val;
                } // if prob>0
            } // iAp
        } // iL
        return Ev_cond;
    }


    void expected_value_start_single_Agrid(int t, int gender, sol_struct* sol,par_struct* par){

        // get index
        auto idx_A = index::single(t,0,par);

        // get variables
        double* EV_start_as_single = (gender == man) ? &sol->EVm_start_as_single[idx_A] : &sol->EVw_start_as_single[idx_A];
        double* EV_cond_meet_partner = (gender == man) ? &sol->EVm_cond_meet_partner[idx_A] : &sol->EVw_cond_meet_partner[idx_A];
        double* V_single_to_single = (gender == man) ? &sol->Vm_single_to_single[idx_A] : &sol->Vw_single_to_single[idx_A];

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
            double EV_cond = expected_value_cond_meet_partner(t, iA, gender, sol, par);

            // expected value of starting single
            EV_start_as_single[iA] = p_meet * EV_cond + (1.0 - p_meet) * V_single_to_single[iA];
            EV_cond_meet_partner[iA] = EV_cond;
        }

        if (par->do_egm){
            calc_marginal_value_single_Agrid(t, gender, sol, par);
        }
    }

    void expected_value_start_single(int t, sol_struct* sol, par_struct* par){

        #pragma omp parallel for collapse(1) num_threads(par->threads)
        for (int sex = 0; sex < 2; sex++) {
            int gender = (sex == 0) ? woman : man;
            expected_value_start_single_Agrid(t, gender, sol, par);
        }
    }

    void solve_couple_to_single(int t, sol_struct *sol, par_struct *par) {
        // get index
        auto idx_A = index::single(t,0,par);
        // get variables
        double* Vw_couple_to_single = &sol->Vw_couple_to_single[idx_A];
        double* Vm_couple_to_single = &sol->Vm_couple_to_single[idx_A];
        double* Vw_single_to_single = &sol->Vw_single_to_single[idx_A];
        double* Vm_single_to_single = &sol->Vm_single_to_single[idx_A];
        double* lw_couple_to_single = &sol->lw_couple_to_single[idx_A];
        double* lm_couple_to_single = &sol->lm_couple_to_single[idx_A];
        double* lw_single_to_single = &sol->lw_single_to_single[idx_A];
        double* lm_single_to_single = &sol->lm_single_to_single[idx_A];

        const double div_cost = par->div_cost;
        for (int iA = 0; iA < par->num_A; iA++) {
            Vw_couple_to_single[iA] = Vw_single_to_single[iA] - div_cost;
            Vm_couple_to_single[iA] = Vm_single_to_single[iA] - div_cost;
            lw_couple_to_single[iA] = lw_single_to_single[iA];
            lm_couple_to_single[iA] = lm_single_to_single[iA];
        }
    }

    int find_interpolated_labor_index_single(int t, double A, int gender, sol_struct* sol, par_struct* par){

        //--- Set variables based on gender ---
        double* grid_A = (gender == woman) ? par->grid_Aw : par->grid_Am;
        double* Vd_single_to_single = (gender == woman) ? sol->Vwd_single_to_single : sol->Vmd_single_to_single;

        // find nearest asset index once
        int iA = tools::binary_search(0, par->num_A, grid_A, A);

        double maxV = -std::numeric_limits<double>::infinity();
        int labor_index = 0;

        //--- Loop over labor choices ---
        for (int il = 0; il < par->num_l; il++) {
            auto idx_d_A = index::single_d(t, il, 0, par);
            double V_now = tools::interp_1d_index(grid_A, par->num_A, &Vd_single_to_single[idx_d_A], A, iA);
            if (V_now > maxV) {
                maxV = V_now;
                labor_index = il;
            }
        }

        return labor_index;
    }

}
