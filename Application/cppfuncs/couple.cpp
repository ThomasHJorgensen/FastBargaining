// Functions for solving the model for couples.
#ifndef MAIN
#define COUPLE
#include "myheader.cpp"
#endif

namespace couple {

    struct SolverData {
        int t;
        int ilw;
        int ilm;
        int iL;
        int iP;
        double Kw;
        double Km;
        double M;
        double* EVw_next;
        double* EVm_next;
        sol_struct* sol;
        par_struct* par;
    };

    // Multistart factor in first multistart run
    static constexpr double MULTISTART_FACTOR = 0.5;


    double resources_couple(double labor_w, double labor_m, double Kw, double Km, double A, par_struct* par) {
        // If no labor income, return asset income plus small epsilon to avoid zero
        if ((labor_w == 0.0) && (labor_m == 0.0)) {
            return par->R * A + 1.0e-4;
        }

        double wage_w = utils::wage(Kw, woman, par);
        double wage_m = utils::wage(Km, man, par);

        return par->R * A + wage_w * labor_w * par->available_hours * (1.0 - par->tax_rate) + wage_m * labor_m * par->available_hours * (1.0 - par->tax_rate);
    }

    double value_of_choice_couple_to_couple(double* Cw_priv, double* Cm_priv, double* hw, double* hm,
        double* C_inter, double* Q, int ilw, int ilm, double C_tot, double Kw, double Km, double M_resources,
        int t, int iL, int iP, double* Vw, double* Vm, double* EVw_next, double* EVm_next,
        par_struct* par, sol_struct* sol)
    {
        double love = par->grid_love[iL];
        double power = par->grid_power[iP];

        precompute::intraperiod_allocation_couple(Cw_priv, Cm_priv, hw, hm, C_inter, Q,
            ilw, ilm, iP, power, C_tot, par, sol, par->precompute_intratemporal, true /*use_power_index*/);

        Vw[0] = utils::util(*Cw_priv, *hw + par->grid_l[ilw], *Q, woman, par, love);
        Vm[0] = utils::util(*Cm_priv, *hm + par->grid_l[ilm], *Q, man, par, love);

        if (t < (par->T - 1)) {
            double A_next = M_resources - C_tot;
            double Kw_next = utils::human_capital_transition(Kw, par->grid_l[ilw], par);
            double Km_next = utils::human_capital_transition(Km, par->grid_l[ilm], par);
            // obs: maybe use _interp_3d_2out here
            double EVw_plus = tools::interp_3d(par->grid_Kw, par->grid_Km, par->grid_A, par->num_K, par->num_K, par->num_A, EVw_next, Kw_next, Km_next, A_next);
            double EVm_plus = tools::interp_3d(par->grid_Kw, par->grid_Km, par->grid_A, par->num_K, par->num_K, par->num_A, EVm_next, Kw_next, Km_next, A_next);
            Vw[0] += par->beta * EVw_plus;
            Vm[0] += par->beta * EVm_plus;
        }

        return power * Vw[0] + (1.0 - power) * Vm[0];
    }

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

    //////////////////
    // VFI solution
    double objfunc_couple_to_couple(unsigned /*n*/, const double* x, double* /*grad*/, void* solver_data_in) {
        auto* d = static_cast<SolverData*>(solver_data_in);
        double C_tot = x[0];

        double Cw_priv, Cm_priv, hw, hm, C_inter, Q, Vw, Vm;
        return -value_of_choice_couple_to_couple(&Cw_priv, &Cm_priv, &hw, &hm, &C_inter, &Q,
            d->ilw, d->ilm, C_tot, d->Kw, d->Km,d->M, d->t, d->iL, d->iP,
            &Vw, &Vm, d->EVw_next, d->EVm_next, d->par, d->sol);
    }

    void solve_couple_to_couple_step(double* Cw_priv, double* Cm_priv, double* hw, double* hm,
        double* C_inter, double* Q, double* Vw, double* Vm, double Kw, double Km, double M_resources,
        int t, int ilw, int ilm, int iL, int iP, double* EVw_next, double* EVm_next,
        double starting_val, sol_struct* sol, par_struct* par)
    {
        double C_tot = M_resources;

        if (t < (par->T - 1)) {
            const int dim = 1;
            double lb[dim], ub[dim];

            nlopt_opt opt = nlopt_create(NLOPT_LN_BOBYQA, dim);

            auto* data = new SolverData();
            data->t = t;
            data->ilw = ilw;
            data->ilm = ilm;
            data->iL = iL;
            data->iP = iP;
            data->Kw = Kw;
            data->Km = Km;
            data->M = M_resources;
            data->EVw_next = EVw_next;
            data->EVm_next = EVm_next;
            data->sol = sol;
            data->par = par;
            
            nlopt_set_min_objective(opt, objfunc_couple_to_couple, data);

            lb[0] = 1.0e-6;
            ub[0] = data->M - 1.0e-6;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            // first optimization run
            double x[dim] = { starting_val };
            double minf_global = 0.0;
            nlopt_optimize(opt, x, &minf_global);
            C_tot = x[0];
            
            // multistart: try a lower starting consumption as well
            C_tot = run_multistart_optimizer_1d(opt, starting_val, lb, ub, par->num_multistart);
            
            // cleanup
            nlopt_destroy(opt);
            delete data;
        }

        // Recompute implied allocation for chosen C_tot
        double _ = value_of_choice_couple_to_couple(Cw_priv, Cm_priv, hw, hm, C_inter, Q,
            ilw, ilm, C_tot, Kw, Km, M_resources, t, iL, iP, Vw, Vm, EVw_next, EVm_next, par, sol);
    }

    void solve_couple_to_couple_Agrid_vfi(
        int t, int ilw, int ilm, int iP, int iL, int iSw, int iSm, int iKw, int iKm,
        double* EVw_next, double* EVm_next, sol_struct* sol, par_struct* par)
    {
        double labor_w = par->grid_l[ilw];
        double labor_m = par->grid_l[ilm];
        double Kw = par->grid_Kw[iKw];
        double Km = par->grid_Km[iKm];

        auto idx_d_A = index::couple_d(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, 0, par);

        double* Cwd_priv = &sol->Cwd_priv_couple_to_couple[idx_d_A];
        double* Cmd_priv = &sol->Cmd_priv_couple_to_couple[idx_d_A];
        double* hwd = &sol->hwd_couple_to_couple[idx_d_A];
        double* hmd = &sol->hmd_couple_to_couple[idx_d_A];
        double* Cd_inter = &sol->Cd_inter_couple_to_couple[idx_d_A];
        double* Qd = &sol->Qd_couple_to_couple[idx_d_A];
        double* Vwd = &sol->Vwd_couple_to_couple[idx_d_A];
        double* Vmd = &sol->Vmd_couple_to_couple[idx_d_A];
        double* Cd_tot = &sol->Cd_tot_couple_to_couple[idx_d_A];

        for (int iA = 0; iA < par->num_A; iA++) {
            double M_resources = resources_couple(labor_w, labor_m, par->grid_Kw[iKw], par->grid_Km[iKm], par->grid_A[iA], par);

            // starting values
            double starting_val = M_resources * 0.8;
            if (iA > 0) {
                starting_val = Cwd_priv[iA - 1] + Cmd_priv[iA - 1] + Cd_inter[iA - 1];
            }

                solve_couple_to_couple_step(&Cwd_priv[iA], &Cmd_priv[iA], &hwd[iA], &hmd[iA], &Cd_inter[iA], &Qd[iA],
                &Vwd[iA], &Vmd[iA], Kw, Km, M_resources, t, ilw, ilm, iL, iP, EVw_next, EVm_next, starting_val, sol, par);

            Cd_tot[iA] = Cwd_priv[iA] + Cmd_priv[iA] + Cd_inter[iA];
        }
    }

    ////////////////////////////// EGM numerical solution //////////////////////////////
    double marg_util_C_couple(double C_tot, int ilw, int ilm, int iP, par_struct* par, sol_struct* sol,
        double /*guess_Cw_priv*/, double /*guess_Cm_priv*/)
    {
        // Use forward difference to approximate marginal utility
        const double delta = 1.0e-4;
        double util = precompute::util_C_couple(C_tot, ilw, ilm, iP, 0.0, par, sol, par->precompute_intratemporal);
        double util_delta = precompute::util_C_couple(C_tot + delta, ilw, ilm, iP, 0.0, par, sol, par->precompute_intratemporal);
        return (util_delta - util) / delta;
    }

    // numerical inverse marginal utility
    struct SolverInvData {
        int ilw;
        int ilm;
        double margU;
        int iP;
        par_struct* par;
        sol_struct* sol;
        bool do_print;
        double guess_Cw_priv;
        double guess_Cm_priv;
    };

    double obj_inv_marg_util_couple(unsigned /*n*/, const double* x, double* /*grad*/, void* solver_data_in) {
        auto* d = static_cast<SolverInvData*>(solver_data_in);
        double C_tot = x[0];

        double penalty = 0.0;
        if (C_tot <= 0.0) {
            penalty = 1000.0 * C_tot * C_tot;
            C_tot = 1.0e-6;
        }

        double diff = marg_util_C_couple(C_tot, d->ilm, d->ilw, d->iP, d->par, d->sol, d->guess_Cw_priv, d->guess_Cm_priv) - d->margU;

        if (d->do_print) {
            logs::write("inverse_log.txt", 1, "C_tot: %f, diff: %f, penalty: %f\n", C_tot, diff, penalty);
        }
        return diff * diff + penalty;
    }

    double inv_marg_util_couple(double margU, int ilw, int ilm, int iP, par_struct* par, sol_struct* sol,
        double guess_Ctot, double guess_Cw_priv, double guess_Cm_priv, bool do_print = false)
    {
        auto* data = new SolverInvData();
        data->ilw = ilw; data->ilm = ilm; data->margU = margU; data->iP = iP;
        data->par = par; data->sol = sol; data->do_print = do_print;
        data->guess_Cw_priv = guess_Cw_priv; data->guess_Cm_priv = guess_Cm_priv;

        const int dim = 1;
        double lb[dim], ub[dim];

        nlopt_opt opt = nlopt_create(NLOPT_LN_BOBYQA, dim);

        if (do_print) logs::write("inverse_log.txt", 0, "margU: %f\n", margU);

        nlopt_set_min_objective(opt, obj_inv_marg_util_couple, data);
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, 1.0e-6);
        nlopt_set_xtol_rel(opt, 1.0e-5);

        lb[0] = 0.0;
        ub[0] = 2.0 * par->max_Ctot;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        // call the helper (multistart count kept larger than original)
        double C_tot = run_multistart_optimizer_1d(opt, guess_Ctot, lb, ub, par->num_multistart);

        nlopt_destroy(opt);
        delete data;

        return C_tot;
    }

    //////////////////
    // EGM solution
    void interpolate_to_exogenous_grid_couple(
        int t, int ilw, int ilm, int iP, int iL, int iSw, int iSm, int iKw, int iKm,
        double* m_vec, double* c_vec, double* v_vec, double* EmargUd_pd,
        double* C_tot, double* Cw_priv, double* Cm_priv, double* hw, double* hm,
        double* C_inter, double* Q, double* Vw, double* Vm,
        double* EVw_next, double* EVm_next, double* V,
        sol_struct* sol, par_struct* par)
    {
        double Kw = par->grid_Kw[iKw];
        double Km = par->grid_Km[iKm];

        // Loop over exogenous asset grid
        for (int iA = 0; iA < par->num_A; iA++) {
            double M_now = resources_couple(par->grid_l[ilw], par->grid_l[ilm], par->grid_Kw[iKw], par->grid_Km[iKm], par->grid_A[iA], par);

            // If liquidity constraint binds, consume all resources
            if (M_now < m_vec[0]) {
                C_tot[iA] = M_now;
                value_of_choice_couple_to_couple(&Cw_priv[iA], &Cm_priv[iA], &hw[iA], &hm[iA], &C_inter[iA], &Q[iA],
                    ilw, ilm, C_tot[iA], Kw, Km, M_now, t, iL, iP, &Vw[iA], &Vm[iA], EVw_next, EVm_next, par, sol);

                double power = par->grid_power[iP];
                V[iA] = power * Vw[iA] + (1.0 - power) * Vm[iA];
                continue;
            }

            // Otherwise, search for the correct interval in the endogenous grid
            for (int iA_pd = 0; iA_pd < par->num_A_pd - 1; iA_pd++) {
                double m_low = m_vec[iA_pd];
                double m_high = m_vec[iA_pd + 1];

                bool in_interval = (M_now >= m_low) && (M_now <= m_high);
                bool extrap_above = (iA_pd == par->num_A_pd - 2) && (M_now > m_vec[par->num_A_pd - 1]);

                if (in_interval || extrap_above) {
                    // Endogenous grid points
                    double A_low = par->grid_A_pd[iA_pd];
                    double A_high = par->grid_A_pd[iA_pd + 1];

                    double V_low = v_vec[iA_pd];
                    double V_high = v_vec[iA_pd + 1];

                    double c_low = c_vec[iA_pd];
                    double c_high = c_vec[iA_pd + 1];

                    // Linear interpolation slopes
                    double v_slope = (V_high - V_low) / (A_high - A_low);
                    double c_slope = (c_high - c_low) / (m_high - m_low);

                    // Interpolated guesses
                    double c_guess = c_low + c_slope * (M_now - m_low);
                    double a_guess = M_now - c_guess;
                    double V_guess = V_low + v_slope * (a_guess - A_low);

                    // Upper envelope: only update if value is higher
                    if (V_guess > V[iA]) {
                        C_tot[iA] = c_guess;

                        value_of_choice_couple_to_couple(
                            &Cw_priv[iA], &Cm_priv[iA], &hw[iA], &hm[iA], &C_inter[iA], &Q[iA],
                            ilw, ilm, C_tot[iA], Kw, Km, M_now, t, iL, iP,
                            &Vw[iA], &Vm[iA], EVw_next, EVm_next, par, sol
                        );

                        double power = par->grid_power[iP];
                        V[iA] = power * Vw[iA] + (1.0 - power) * Vm[iA];
                    }
                }
            }
        }
    }

    void solve_couple_to_couple_Agrid_egm(
        int t, int ilw, int ilm, int iP, int iL, int iSw, int iSm, int iKw, int iKm,
        double* EVw_next, double* EVm_next, double* EmargV_next, sol_struct* sol, par_struct* par)
    {
        double Cw_priv = 0.0, Cm_priv = 0.0, hw = 0.0, hm = 0.0, C_inter = 0.0, Q = 0.0, Vw = 0.0, Vm = 0.0;

        auto idx_A_pd      = index::couple_pd(t,     ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, 0, par);
        auto idx_A_pd_next = index::couple_pd(t + 1, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, 0, par);

        double* EmargUd_pd = &sol->EmargUd_pd[idx_A_pd];
        double* Cd_tot_pd = &sol->Cd_tot_pd[idx_A_pd];
        double* Cd_tot_pd_next = &sol->Cd_tot_pd[idx_A_pd_next];
        double* Md_pd = &sol->Md_pd[idx_A_pd];
        double* Vd_couple_to_couple_pd = &sol->Vd_couple_to_couple_pd[idx_A_pd];

        double Kw = par->grid_Kw[iKw];
        double Km = par->grid_Km[iKm];
        double Kw_next = utils::human_capital_transition(Kw, par->grid_l[ilw], par);
        double Km_next = utils::human_capital_transition(Km, par->grid_l[ilm], par);

        // Loop over endogenous asset grid
        for (int iA_pd = 0; iA_pd < par->num_A_pd; ++iA_pd) {
            double A_next = par->grid_A_pd[iA_pd];

            EmargUd_pd[iA_pd] = par->beta * tools::interp_3d(par->grid_Kw, par->grid_Km, par->grid_A, par->num_K, par->num_K, par->num_A, EmargV_next, Kw_next, Km_next, A_next);

            if (strcmp(par->interp_method, "numerical") == 0) {

                // starting values
                double guess_Ctot = 3.0;
                double guess_Cw_priv = guess_Ctot / 3.0;
                double guess_Cm_priv = guess_Ctot / 3.0;

                if (iA_pd > 0) {
                    // last found solution
                    guess_Ctot = Cd_tot_pd[iA_pd - 1];
                    guess_Cw_priv = Cw_priv;
                    guess_Cm_priv = Cm_priv;

                } else if (t < (par->T - 2)) {
                    guess_Ctot = Cd_tot_pd_next[iA_pd]; // if not first period, use next period solution as starting value
                }

                Cd_tot_pd[iA_pd] = inv_marg_util_couple(EmargUd_pd[iA_pd], ilw, ilm, iP, par, sol,
                    guess_Ctot, guess_Cw_priv, guess_Cm_priv);
            } else {
                if (strcmp(par->interp_method, "linear") == 0) {
                    auto idx_interp = index::index4(ilw, ilm, iP, 0, par->num_l, par->num_l, par->num_power, par->num_marg_u);
                    Cd_tot_pd[iA_pd] = tools::interp_1d(&par->grid_marg_u_couple_for_inv[idx_interp], par->num_marg_u, par->grid_inv_marg_u, EmargUd_pd[iA_pd]);
                }

                if (par->interp_inverse) Cd_tot_pd[iA_pd] = 1.0 / Cd_tot_pd[iA_pd];
            }

            // Get endogenous grid points
            Md_pd[iA_pd] = A_next + Cd_tot_pd[iA_pd];

            // v. Get post-choice value (also updates the intra-period allocation)
            Vd_couple_to_couple_pd[iA_pd] = value_of_choice_couple_to_couple(
                &Cw_priv, &Cm_priv, &hw, &hm, &C_inter, &Q,
                ilw, ilm, Cd_tot_pd[iA_pd], Kw, Km, Md_pd[iA_pd], t, iL, iP,
                &Vw, &Vm, EVw_next, EVm_next,
                par, sol
            );
        }

        // Apply liquidity constraint and upper envelope while interpolating onto common grid
        auto idx_d_A = index::couple_d(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, 0, par);
        interpolate_to_exogenous_grid_couple(
            t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm,
            &sol->Md_pd[idx_A_pd], &sol->Cd_tot_pd[idx_A_pd], &sol->Vd_couple_to_couple_pd[idx_A_pd],
            &sol->EmargUd_pd[idx_A_pd], &sol->Cd_tot_couple_to_couple[idx_d_A],
            &sol->Cwd_priv_couple_to_couple[idx_d_A], &sol->Cmd_priv_couple_to_couple[idx_d_A],
            &sol->hwd_couple_to_couple[idx_d_A], &sol->hmd_couple_to_couple[idx_d_A],
            &sol->Cd_inter_couple_to_couple[idx_d_A], &sol->Qd_couple_to_couple[idx_d_A],
            &sol->Vwd_couple_to_couple[idx_d_A], &sol->Vmd_couple_to_couple[idx_d_A],
            EVw_next, EVm_next, &sol->Vd_couple_to_couple[idx_d_A],
            sol, par
        );
    }

    void calc_marginal_value_couple_Agrid(double power, double* Vw, double* Vm, double* margV, sol_struct* sol, par_struct* par)
    {
        if (par->centered_gradient) {
            for (int iA = 1; iA <= par->num_A - 2; ++iA) {
                int iA_plus = iA + 1;
                int iA_minus = iA - 1;
                double denom = 1.0 / (par->grid_A[iA_plus] - par->grid_A[iA_minus]);
                double margVw = Vw[iA_plus] * denom - Vw[iA_minus] * denom;
                double margVm = Vm[iA_plus] * denom - Vm[iA_minus] * denom;
                margV[iA] = power * margVw + (1.0 - power) * margVm;
                if (iA == par->num_A - 2) margV[iA_plus] = margV[iA];
            }

            margV[0] = (margV[2] - margV[1]) / (par->grid_A[2] - par->grid_A[1]) * (par->grid_A[0] - par->grid_A[1]) + margV[1];
            int i = par->num_A - 1;
            margV[i] = (margV[i - 2] - margV[i - 1]) / (par->grid_A[i - 2] - par->grid_A[i - 1]) * (par->grid_A[i] - par->grid_A[i - 1]) + margV[i - 1];
        } else {
            for (int iA = 0; iA <= par->num_A - 2; ++iA) {
                int iA_plus = iA + 1;
                double denom = 1.0 / (par->grid_A[iA_plus] - par->grid_A[iA]);
                double margVw = Vw[iA_plus] * denom - Vw[iA] * denom;
                double margVm = Vm[iA_plus] * denom - Vm[iA] * denom;
                margV[iA] = power * margVw + (1.0 - power) * margVm;
                if (iA == par->num_A - 2) margV[iA_plus] = margV[iA];
            }
        }
    }

    void update_optimal_discrete_solution_couple_Agrid(
        int t, int ilw, int ilm, int iP, int iL, int iSw, int iSm, int iKw, int iKm,
        sol_struct* sol, par_struct* par)
    {
        auto idx_A   = index::couple(t, iP, iL, iSw, iSm, iKw, iKm, 0, par);
        auto idx_d_A = index::couple_d(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, 0, par);

        double power = par->grid_power[iP];
        double* Vwd_couple_to_couple = &sol->Vwd_couple_to_couple[idx_d_A];
        double* Vmd_couple_to_couple = &sol->Vmd_couple_to_couple[idx_d_A];
        double* V_couple_to_couple = &sol->V_couple_to_couple[idx_A];
        double* Vw_couple_to_couple = &sol->Vw_couple_to_couple[idx_A];
        double* Vm_couple_to_couple = &sol->Vm_couple_to_couple[idx_A];
        double* lw_couple_to_couple = &sol->lw_couple_to_couple[idx_A];
        double* lm_couple_to_couple = &sol->lm_couple_to_couple[idx_A];

        for (int iA = 0; iA < par->num_A; ++iA) {
            double Vd_now = power * Vwd_couple_to_couple[iA] + (1.0 - power) * Vmd_couple_to_couple[iA];
            if (V_couple_to_couple[iA] < Vd_now) {
                V_couple_to_couple[iA] = Vd_now;
                Vw_couple_to_couple[iA] = Vwd_couple_to_couple[iA];
                Vm_couple_to_couple[iA] = Vmd_couple_to_couple[iA];
                lw_couple_to_couple[iA] = par->grid_l[ilw];
                lm_couple_to_couple[iA] = par->grid_l[ilm];
            }
        }
    }

    void solve_choice_specific_couple_to_couple(
        int t, int iP, int iL, int iSw, int iSm, int iKw, int iKm, int ilw, int ilm,
        sol_struct* sol, par_struct* par)
    {
        if (t == (par->T - 1)) {
            solve_couple_to_couple_Agrid_vfi(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, nullptr, nullptr, sol, par);
        } else {
            auto idx_next = index::couple(t + 1, iP, iL, iSw, iSm, 0, 0, 0, par);
            double* EVw_next    = &sol->EVw_start_as_couple[idx_next];
            double* EVm_next    = &sol->EVm_start_as_couple[idx_next];
            double* EmargV_next = &sol->EmargV_start_as_couple[idx_next];

            if (par->do_egm) {
                solve_couple_to_couple_Agrid_egm(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, EVw_next, EVm_next, EmargV_next, sol, par);
            } else {
                solve_couple_to_couple_Agrid_vfi(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, EVw_next, EVm_next, sol, par);
            }
        }

        update_optimal_discrete_solution_couple_Agrid(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, sol, par);
    }

    void solve_couple(int t, sol_struct* sol, par_struct* par)
    {
        #pragma omp parallel for collapse(6) num_threads(par->threads)
        for (int iP = 0; iP < par->num_power; ++iP) {
            for (int iL = 0; iL < par->num_love; ++iL) {
                for (int iSw = 0; iSw < par->num_S; ++iSw) {
                    for (int iSm = 0; iSm < par->num_S; ++iSm) {
                        for (int iKw = 0; iKw < par->num_K; ++iKw) {
                            for (int iKm = 0; iKm < par->num_K; ++iKm) {
                                // Note: important to have discrete choice as inner loop
                                //       to allow parallelization over outer loops while
                                //       making the optimal choice of discrete choice
                                //       thread-safe
                                for (int ilw = 0; ilw < par->num_l; ++ilw) {
                                    for (int ilm = 0; ilm < par->num_l; ++ilm) {
                                        solve_choice_specific_couple_to_couple(t, iP, iL, iSw, iSm, iKw, iKm, ilw, ilm, sol, par);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double calc_marital_surplus(double Vd_couple_to_couple, double V_couple_to_single) {
        return Vd_couple_to_couple - V_couple_to_single;
    }

    void solve_start_as_couple_powergrid(
        int t, int iL, int iSw, int iSm, int iKw, int iKm, int iA,
        sol_struct* sol, par_struct* par)
    {
        const int num = 2;
        double** list_start_as_couple = new double*[num];
        double** list_couple_to_couple = new double*[num];
        double* list_couple_to_single = new double[num];

        double* surplus_w = new double[par->num_power];
        double* surplus_m = new double[par->num_power];

        auto* idx_couple_fct = new index::index_couple_struct;

        auto idx_single_w = index::single(t, iSw, iKw, iA, par);
        auto idx_single_m = index::single(t, iSm, iKm, iA, par);

        idx_couple_fct->t = t;
        idx_couple_fct->iSw = iSw;
        idx_couple_fct->iSm = iSm;
        idx_couple_fct->iL = iL;
        idx_couple_fct->iKw = iKw;
        idx_couple_fct->iKm = iKm;
        idx_couple_fct->iA = iA;
        idx_couple_fct->par = par;

        for (int iP = 0; iP < par->num_power; ++iP) {
            auto idx_couple = index::couple(t, iP, iL, iSw, iSm, iKw, iKm, iA, par);
            surplus_w[iP] = calc_marital_surplus(sol->Vw_couple_to_couple[idx_couple], sol->Vw_couple_to_single[idx_single_w]);
            surplus_m[iP] = calc_marital_surplus(sol->Vm_couple_to_couple[idx_couple], sol->Vm_couple_to_single[idx_single_m]);
        }

        // logs::write("surplus_log.txt", 1, "\n - surplus_w: %f, surplus_m: %f\n", surplus_w[0], surplus_m[0]);
        list_start_as_couple[0] = sol->Vw_start_as_couple;
        list_start_as_couple[1] = sol->Vm_start_as_couple;
        list_couple_to_couple[0] = sol->Vw_couple_to_couple;
        list_couple_to_couple[1] = sol->Vm_couple_to_couple;
        list_couple_to_single[0] = sol->Vw_couple_to_single[idx_single_w];
        list_couple_to_single[1] = sol->Vm_couple_to_single[idx_single_m];

        // update solutions in list_start_as_couple
        // logs::write("barg_log.txt", 1, "\n \n indices: t: %d, iL: %d, iSw: %d, iSm: %d, iKw: %d, iKm: %d, iA: %d\n", t, iL, iSw, iSm, iKw, iKm, iA);

        bargaining::check_participation_constraints(sol->power_idx, sol->power, surplus_w, surplus_m, idx_couple_fct,
            list_start_as_couple, list_couple_to_couple, list_couple_to_single, num, par);

        delete[] list_start_as_couple;
        delete[] list_couple_to_couple;
        delete[] list_couple_to_single;
        delete[] surplus_w;
        delete[] surplus_m;
        delete idx_couple_fct;
    }

    void calc_expected_value_couple(
        int t, int iP, int iL, int iSw, int iSm, int iKw, int iKm, int iA,
        double* Vw, double* Vm, double* EVw, double* EVm,
        sol_struct* /*sol*/, par_struct* par)
    {
        double love = par->grid_love[iL];
        double Kw = par->grid_Kw[iKw];
        double Km = par->grid_Km[iKm];
        auto idx = index::couple(t, iP, iL, iSw, iSm, iKw, iKm, iA, par);

        double Eval_w = 0.0;
        double Eval_m = 0.0;

        // OBS: This interpolation thing and especially its indices needs to be handled properly
        // currently it does not allow S to be less than two, becayse then interpolation goes out of bounds
        auto idx_A = (iA < par->num_A -1) ? iA : (par->num_A - 2); // if A index is out of bounds, use the second to last index for interpolation
        auto idx_Sw = (iSw < par->num_S - 1) ? iSw : (par->num_S - 2);
        auto idx_Sm = (iSm < par->num_S - 1) ? iSm : (par->num_S - 2);

        for (int i_love_shock = 0; i_love_shock < par->num_shock_love; ++i_love_shock) {
            double love_shock = love + par->grid_shock_love[i_love_shock];
            double weight_love = par->grid_weight_love[i_love_shock];
            auto idx_love = tools::binary_search(0, par->num_love, par->grid_love, love_shock);
            for (int iKw_shock = 0; iKw_shock < par->num_shock_K; ++iKw_shock) {
                double Kw_shock = par->grid_shock_Kw[iKw_shock] * Kw;
                double weight_Kw = par->grid_weight_Kw[iKw_shock];
                auto idx_Kw = tools::binary_search(0, par->num_K, par->grid_Kw, Kw_shock);
                for (int iKm_shock = 0; iKm_shock < par->num_shock_K; ++iKm_shock) {
                    double Km_shock = par->grid_shock_Km[iKm_shock] * Km;
                    double weight_Km = par->grid_weight_Km[iKm_shock];
                    auto idx_Km = tools::binary_search(0, par->num_K, par->grid_Km, Km_shock);
                    
                    
                    auto idx_interp = index::couple(t, iP, 0, 0, 0, 0, 0, 0, par); // OBS: does interpolation over S go well?
                    double Sw = par->grid_S[iSw];
                    double Sm = par->grid_S[iSm];
                    double Vw_now = tools::_interp_6d_index(
                        par->grid_love, par->grid_S, par->grid_S, par->grid_Kw,par->grid_Km,par->grid_A,
                        par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
                        &Vw[idx_interp],
                        love_shock, Sw, Sm, Kw_shock, Km_shock, par->grid_A[iA],
                        idx_love, idx_Sw, idx_Sm, idx_Kw, idx_Km, idx_A
                    );
                    double Vm_now = tools::_interp_6d_index(
                        par->grid_love, par->grid_S, par->grid_S, par->grid_Kw,par->grid_Km,par->grid_A,
                        par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
                        &Vm[idx_interp],
                        love_shock, Sw, Sm, Kw_shock, Km_shock, par->grid_A[iA],
                        idx_love, idx_Sw, idx_Sm, idx_Kw, idx_Km, idx_A
                    );

                    double weight = weight_love * weight_Kw * weight_Km;
                    Eval_w += weight * Vw_now;
                    Eval_m += weight * Vm_now;
                }
            }
        }

        EVw[idx] = Eval_w;
        EVm[idx] = Eval_m;
    }

    void expected_value_start_couple(int t, sol_struct* sol, par_struct* par)
    {
        // logs::write("barg_log.txt", 0, "start");
        // #pragma omp parallel for collapse(4) num_threads(par->threads)
        for (int iL = 0; iL < par->num_love; ++iL) {
            for (int iSw = 0; iSw < par->num_S; ++iSw) {
                for (int iSm = 0; iSm < par->num_S; ++iSm) {
                    for (int iKw = 0; iKw < par->num_K; ++iKw) {
                        for (int iKm = 0; iKm < par->num_K; ++iKm) {
                            for (int iA = 0; iA < par->num_A; ++iA) {
                                solve_start_as_couple_powergrid(t, iL, iSw, iSm, iKw, iKm, iA, sol, par);
                            }
                        }
                    }
                }
            }
        }

        #pragma omp parallel for collapse(6) num_threads(par->threads)
        for (int iP = 0; iP < par->num_power; ++iP) {
            for (int iL = 0; iL < par->num_love; ++iL) {
                for (int iSw = 0; iSw < par->num_S; ++iSw) {
                    for (int iSm = 0; iSm < par->num_S; ++iSm) {
                        for (int iKw = 0; iKw < par->num_K; ++iKw) {
                            for (int iKm = 0; iKm < par->num_K; ++iKm) {
                                for (int iA = 0; iA < par->num_A; ++iA) {
                                    calc_expected_value_couple(
                                        t, iP, iL, iSw, iSm, iKw, iKm, iA,
                                        sol->Vw_start_as_couple, sol->Vm_start_as_couple,
                                        sol->EVw_start_as_couple, sol->EVm_start_as_couple,
                                        sol, par
                                    );
                                }

                                if (par->do_egm) {
                                    auto idx_A = index::couple(t, iP, iL, iSw, iSm, iKw, iKm, 0, par);
                                    double power = par->grid_power[iP];
                                    calc_marginal_value_couple_Agrid(
                                        power,
                                        &sol->EVw_start_as_couple[idx_A], &sol->EVm_start_as_couple[idx_A],
                                        &sol->EmargV_start_as_couple[idx_A],
                                        sol, par
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void solve_single_to_couple(int t, sol_struct* sol, par_struct* par) {
        for (int iP = 0; iP < par->num_power; ++iP) {
            for (int iL = 0; iL < par->num_love; ++iL) {
                for (int iSw = 0; iSw < par->num_S; ++iSw) {
                    for (int iSm = 0; iSm < par->num_S; ++iSm) {
                        for (int iKw = 0; iKw < par->num_K; ++iKw) {
                            for (int iKm = 0; iKm < par->num_K; ++iKm) {
                                auto idx_A = index::couple(t, iP, iL, iSw, iSm, iKw, iKm, 0, par);

                                double* Vw_single_to_couple = &sol->Vw_single_to_couple[idx_A];
                                double* Vm_single_to_couple = &sol->Vm_single_to_couple[idx_A];
                                double* Vw_couple_to_couple = &sol->Vw_couple_to_couple[idx_A];
                                double* Vm_couple_to_couple = &sol->Vm_couple_to_couple[idx_A];
                                double* lw_single_to_couple = &sol->lw_single_to_couple[idx_A];
                                double* lm_single_to_couple = &sol->lm_single_to_couple[idx_A];
                                double* lw_couple_to_couple = &sol->lw_couple_to_couple[idx_A];
                                double* lm_couple_to_couple = &sol->lm_couple_to_couple[idx_A];

                                for (int iA = 0; iA < par->num_A; ++iA) {
                                    Vw_single_to_couple[iA] = Vw_couple_to_couple[iA];
                                    Vm_single_to_couple[iA] = Vm_couple_to_couple[iA];
                                    lw_single_to_couple[iA] = lw_couple_to_couple[iA];
                                    lm_single_to_couple[iA] = lm_couple_to_couple[iA];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void find_interpolated_labor_index_couple(
        int t, double power, double love, int iSw, int iSm, double Kw, double Km, double A,
        int* ilw_out, int* ilm_out,
        sol_struct* sol, par_struct* par)
    {
        int iP = tools::binary_search(0, par->num_power, par->grid_power, power);
        int iL = tools::binary_search(0, par->num_love, par->grid_love, love);
        int idx_Sw = tools::binary_search(0, par->num_S, par->grid_S, par->grid_S[iSw]);
        int idx_Sm = tools::binary_search(0, par->num_S, par->grid_S, par->grid_S[iSm]);
        int iKw = tools::binary_search(0, par->num_K, par->grid_Kw, Kw);
        int iKm = tools::binary_search(0, par->num_K, par->grid_Km, Km);
        int iA = tools::binary_search(0, par->num_A, par->grid_A, A);


        double maxV = -std::numeric_limits<double>::infinity();
        int labor_index_w = 0;
        int labor_index_m = 0;

        for (int ilw = 0; ilw < par->num_l; ++ilw) {
            for (int ilm = 0; ilm < par->num_l; ++ilm) {
                double Sw = par->grid_S[iSw];
                double Sm = par->grid_S[iSm];
                auto idx_interp = index::couple_d(t, ilw, ilm, 0, 0, 0, 0, 0, 0, 0, par);

                double Vw_now = tools::_interp_7d_index(
                    par->grid_power, par->grid_love, par->grid_S, par->grid_S, par->grid_Kw, par->grid_Km, par->grid_A,
                    par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
                    &sol->Vwd_couple_to_couple[idx_interp],
                    power, love, Sw, Sm, Kw, Km, A,
                    iP, iL, idx_Sw, idx_Sm, iKw, iKm, iA
                );
                double Vm_now = tools::_interp_7d_index(
                    par->grid_power, par->grid_love, par->grid_S, par->grid_S, par->grid_Kw, par->grid_Km, par->grid_A,
                    par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A,
                    &sol->Vmd_couple_to_couple[idx_interp],
                    power, love, Sw, Sm, Kw, Km, A,
                    iP, iL, idx_Sw, idx_Sm, iKw, iKm, iA
                );

                double V_now = power * Vw_now + (1.0 - power) * Vm_now;
                if (maxV < V_now) {
                    maxV = V_now;
                    labor_index_w = ilw;
                    labor_index_m = ilm;
                }
            }
        }

        //--- Return optimal labor choice ---
        *ilw_out = labor_index_w;
        *ilm_out = labor_index_m;
    }

}
