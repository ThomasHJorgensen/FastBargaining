#ifndef MAIN
#define SINGLE
#include "myheader.cpp"
#endif

namespace single {
    typedef struct {
        int t;
        int il;     
                
        double M;             
        double *EV_next;      
        int gender;

        par_struct *par;      
        sol_struct *sol;      

    } solver_single_struct;

    double resources_single(double labor, double A,int gender,par_struct* par) {
        if (labor == 0.0) {
            // no labor income, just resources from assets
            return par->R*A + 1.0e-4; // add a small amount to avoid errors with zero
        }
        
        double K = 5.0;

        double w = utils::wage(K, woman, par);
        if (gender == man) {
            w = utils::wage(K, man, par);
        }
        return par->R*A + w*labor;
    }

    double value_of_choice_single_to_single(double* C_priv, double* h, double* C_inter, double* Q, double C_tot, int t, int il, double M, int gender, double* V_next, par_struct* par, sol_struct* sol){
        double love = 0.0; // no love for singles
        double labor = par->grid_l[il];

        // intraperiod allocation
        precompute::intraperiod_allocation_single(C_priv, h, C_inter, Q, C_tot, il, gender, par, sol, par->precompute_intratemporal);
        
        // current utility from consumption allocation
        double lh = *h + labor;
        double Util = utils::util(*C_priv, lh, *Q, gender, par, love);
        double V_next_value = 0.0;

        if (t < (par->T-1)) {
            // continuation value
            double *grid_A = par->grid_Aw; 
            if (gender==man){
                grid_A = par->grid_Am;
            }
            double A = M - C_tot;

            V_next_value = tools::interp_1d(grid_A,par->num_A,V_next,A);
        }
        
        // return discounted sum
        return Util + par->beta*V_next_value;
    }

    double objfunc_single_to_single(unsigned n, const double *x, double *grad, void *solver_data_in){
        double love = 0.0;

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double C_tot = x[0];
        int t = solver_data->t;
        int il = solver_data->il;
        int gender = solver_data->gender;
        double M = solver_data->M;
        par_struct *par = solver_data->par;
        sol_struct *sol = solver_data->sol;

        double C_priv, h, C_inter, Q;

        return - value_of_choice_single_to_single(
            &C_priv,
            &h,
            &C_inter,
            &Q,
            C_tot,
            t,
            il,
            M,
            gender,
            solver_data->EV_next,
            par,
            sol
        );
    }

    void solve_single_to_single_step(
        double* Cd_priv, double* hd, double* Cd_inter, double* Qd, double* Vd,
        double M_resources, int t, int il, 
        double* EV_next,
        double starting_val, int gender, sol_struct *sol,par_struct *par
    ){
        double Cd_tot = M_resources;

        if (t < (par->T-1)){

            // 1. allocate objects for solver
            solver_single_struct* solver_data = new solver_single_struct;
            
            int const dim = 1;
            double lb[dim],ub[dim],x[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
            double minf=0.0;

            // search over optimal total consumption, C
            // WOMEN
            // settings
            solver_data->t = t;
            solver_data->il = il;
            solver_data->M = M_resources;
            solver_data->EV_next = EV_next; // sol->EVw_start_single;
            solver_data->gender = gender;
            solver_data->par = par;
            solver_data->sol = sol;
            nlopt_set_min_objective(opt, objfunc_single_to_single, solver_data); 

            // bounds
            lb[0] = 1.0e-8;
            ub[0] = solver_data->M;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            // optimize
            x[0] = starting_val;  // start_value
            nlopt_optimize(opt, x, &minf); 
            nlopt_destroy(opt);

            // unpack results
            Cd_tot = x[0];
            // Vd[iA] = -minf;

            delete solver_data;
        }

        // implied consumption allocation (re-calculation)
        *Vd = value_of_choice_single_to_single(Cd_priv, hd, Cd_inter, Qd, Cd_tot, t, il, M_resources, gender, EV_next, par, sol);
    }

    void solve_single_to_single_Agrid_vfi(int t, int il, double* EV_next, int gender,sol_struct* sol, par_struct* par){
        double love = 0.0; // no love for singles 
        double labor = par->grid_l[il];

        // get index
        auto idx_d_A = index::single_d(t,il,0,par);
        auto idx_A_next = index::single(t+1,0,par);


        // get variables
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

        for (int iA=0; iA<par->num_A;iA++){
            double M_resources = resources_single(labor, grid_A[iA], gender, par);

            // starting values
            double starting_val = M_resources * 0.8;
            if (iA>0){
                starting_val = Cd_tot[iA-1];
            }

            solve_single_to_single_step(&Cd_priv[iA], &hd[iA], &Cd_inter[iA], &Qd[iA], &Vd[iA], M_resources, t, il, EV_next, starting_val, gender, sol, par);
            Cd_tot[iA] = Cd_priv[iA] + Cd_inter[iA];
        } // iA

    }

    //////////////////// EGM ////////////////////////
    ////// numerical EGM //////
    double marg_util_C_single(double C_tot, int il, int gender, par_struct* par, sol_struct* sol, double guess_C_priv = 3.0, double guess_h = 3.0){

        // OBS: Implement start values for C_priv and h
        double C_priv = guess_C_priv;
        double h = guess_h;

        // closed form solution for intra-period problem of single
        double util = precompute::util_C_single(C_tot, il, gender, par, sol, par->precompute_intratemporal);

        // forward difference
        double delta = 0.0001;
        double util_delta = precompute::util_C_single(C_tot + delta, il, gender, par, sol, par->precompute_intratemporal);
        return (util_delta - util)/delta;
    }

    typedef struct { 
        int il;
        double margU;
        int gender;
        par_struct *par;
        sol_struct *sol;
        bool do_print;

        double guess_C_priv;
        double guess_h;
    } solver_inv_struct_single;

    double obj_inv_marg_util_single(unsigned n, const double *x, double *grad, void *solver_data_in){
         // unpack
        solver_inv_struct_single *solver_data = (solver_inv_struct_single *) solver_data_in; 
        
        double C_tot = x[0];
        double il = solver_data->il;
        double margU = solver_data->margU;
        int gender = solver_data->gender;
        bool do_print = solver_data->do_print;
        par_struct *par = solver_data->par;
        sol_struct *sol = solver_data->sol;
        
        double guess_C_priv = solver_data->guess_C_priv; // OBS: starting values not used
        double guess_h = solver_data->guess_h; // OBS: starting values not used

        // clip
        double penalty = 0.0;
        if (C_tot <= 0.0) {
            penalty += 1000.0*C_tot*C_tot;
            C_tot = 1.0e-6;
        }

        // return squared difference (using analytical marginal utility)
        double diff = marg_util_C_single(C_tot, il, gender, par, sol, guess_C_priv, guess_h) - margU;

        if (do_print){
            logs::write("inverse_log.txt",1,"C_tot: %f, diff: %f, penalty: %f\n",C_tot,diff,penalty);
        }
        return diff*diff + penalty;

    }

    double inv_marg_util_single(double margU, int il, int gender, par_struct* par, sol_struct* sol, double guess_C_tot, double guess_C_priv, double guess_h, bool do_print=false){
        // setup numerical solver
        solver_inv_struct_single* solver_data = new solver_inv_struct_single;  
                
        int const dim = 1;
        double lb[dim],ub[dim],x[dim];   
        
        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT NLOPT_LN_BOBYQA
        double minf=0.0;

        // search over optimal total consumption, C
        // settings
        solver_data->il = il;  
        solver_data->margU = margU;  
        solver_data->gender = gender;         
        solver_data->par = par;
        solver_data->sol = sol;
        solver_data->do_print = do_print;   
        solver_data->guess_C_priv = guess_C_priv;     
        solver_data->guess_h = guess_h;

        if (do_print){
            logs::write("inverse_log.txt",0,"margU: %f\n",margU);
        }

        nlopt_set_min_objective(opt, obj_inv_marg_util_single, solver_data);   
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, 1.0e-6);
        nlopt_set_xtol_rel(opt, 1.0e-5);

        // bounds
        lb[0] = 0.0;  
        ub[0] = 2.0*par->max_Ctot;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        // optimize
        x[0] = guess_C_tot; 
        nlopt_optimize(opt, x, &minf);          
        nlopt_destroy(opt);                 
        
        delete solver_data;

        // return consumption value
        return x[0];
        
    }


    /////// iEGM //////
    void interpolate_to_exogenous_grid_single(
        int t, int il, int gender,
        double* m_vec, double* c_vec, double* v_vec,
        double* C_tot, double* C_priv, double* h, double* C_inter, double* Q, double* V,
        double* EV_next, sol_struct* sol, par_struct* par
    ) {
        // Select asset and endogenous asset grids based on gender
        double labor = par->grid_l[il];
        double* grid_A = (gender == man) ? par->grid_Am : par->grid_Aw;
        double* grid_A_pd = (gender == man) ? par->grid_Am_pd : par->grid_Aw_pd;

        // Loop over the common (exogenous) asset grid
        for (int iA = 0; iA < par->num_A; iA++) {
            double M_now = resources_single(labor, grid_A[iA], gender, par);

            // If liquidity constraint binds, consume all resources
            if (M_now < m_vec[0]) {
                C_tot[iA] = M_now;
                V[iA] = value_of_choice_single_to_single(
                    &C_priv[iA], &h[iA], &C_inter[iA], &Q[iA],
                    C_tot[iA], t, il, M_now, gender, EV_next, par, sol
                );
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

                    // Upper envelope: keep the highest value
                    if (V_guess > V[iA]) {
                        C_tot[iA] = c_guess;
                        V[iA] = value_of_choice_single_to_single(
                            &C_priv[iA], &h[iA], &C_inter[iA], &Q[iA],
                            C_tot[iA], t, il, M_now, gender, EV_next, par, sol
                        );
                    }
                }
            }
        }
    }


    void solve_single_to_single_Agrid_egm(int t, int il, int gender, sol_struct* sol, par_struct* par){
        // 1. Setup
        /// a. unpack
        double* const &grid_inv_marg_u {par->grid_inv_marg_u};

        /// b. gender specific variables
        //// o. woman
        double* grid_A {par->grid_Aw};
        double* grid_A_pd {par->grid_Aw_pd};
        double* grid_marg_u_single_for_inv {par->grid_marg_u_single_w_for_inv};
        double* V {sol->Vwd_single_to_single};
        double* EV {sol->EVw_start_as_single};
        double* margV {sol->EmargVw_start_as_single};
        double* C_tot {sol->Cwd_tot_single_to_single};
        double* C_priv {sol->Cwd_priv_single_to_single};
        double* h {sol->hwd_single_to_single};
        double* C_inter {sol->Cwd_inter_single_to_single};
        double* Q {sol->Qwd_single_to_single};
        double* EmargU_pd {sol->EmargUwd_single_to_single_pd};
        double* C_tot_pd {sol->Cwd_tot_single_to_single_pd};
        double* M_pd {sol->Mwd_single_to_single_pd};
        double* V_pd {sol->Vwd_single_to_single_pd};
        //// oo. man
        if (gender == man){
            grid_A = par->grid_Am;
            grid_A_pd = par->grid_Am_pd;
            grid_marg_u_single_for_inv = par->grid_marg_u_single_m_for_inv;
            V = sol->Vmd_single_to_single;
            EV = sol->EVm_start_as_single;
            margV = sol->EmargVm_start_as_single;
            C_tot = sol->Cmd_tot_single_to_single;
            C_priv = sol->Cmd_priv_single_to_single;
            h = {sol->hmd_single_to_single};
            C_inter = {sol->Cmd_inter_single_to_single};
            Q = {sol->Qmd_single_to_single};
            EmargU_pd = sol->EmargUmd_single_to_single_pd;
            C_tot_pd = sol->Cmd_totm_single_to_single_pd;
            M_pd = sol->Mmd_single_to_single_pd;
            V_pd = sol->Vmd_single_to_single_pd;
        }

        /// c. Allocate memory
        // double* EmargU_pd {new double[par->num_A_pd]};
        // double* C_tot_pd {new double[par->num_A_pd]};
        // double* M_pd {new double[par->num_A_pd]};

        // 2. EGM step
        /// setup
        auto idx = index::single_d(t,il,0, par);
        auto idx_next = index::single(t+1,0, par);
        auto idx_interp = index::index2(il,0,par->num_l,par->num_marg_u);
        int min_point_A = 0;

        for (int iA_pd=0; iA_pd<par->num_A_pd; iA_pd++){

            /// a. get next period assets
            double A_next = grid_A_pd[iA_pd];

            /// b. calculate expected marginal utility
            min_point_A = tools::binary_search(min_point_A, par->num_A, grid_A, A_next);
            EmargU_pd[iA_pd] = par->beta*tools::interp_1d_index(grid_A, par->num_A, &margV[idx_next],A_next, min_point_A);

            /// c. invert marginal utility by interpolation from pre-computed grid
            if(strcmp(par->interp_method,"numerical")==0){
                // starting values
                    double guess_C_tot = 3.0;
                    double guess_C_priv = guess_C_tot/3.0;
                    double guess_h = (par->Day - par->grid_l[il])/2.0; // OBS: Return to this when implementing labor choice (I think it should be part of x or something we max over afterwards)
                    if(iA_pd>0){
                        // last found solution
                        guess_C_tot = C_tot_pd[iA_pd-1];
                        // guess_C_priv = C_priv;
                        // guess_C_priv = guess_h;
                    }
                    // OBS: starting values are not used.
                    C_tot_pd[iA_pd] = inv_marg_util_single(EmargU_pd[iA_pd],il, gender, par,sol, guess_C_tot, guess_C_priv, guess_h); // numerical inverse
            } else {
                if(strcmp(par->interp_method,"linear")==0){
                    C_tot_pd[iA_pd] = tools::interp_1d(&grid_marg_u_single_for_inv[idx_interp],par->num_marg_u,par->grid_inv_marg_u, EmargU_pd[iA_pd]);
                }
                if (par->interp_inverse){
                    C_tot_pd[iA_pd] = 1.0/C_tot_pd[iA_pd];
                }
            }

            /// d. endogenous grid over resources
            M_pd[iA_pd] = C_tot_pd[iA_pd] + A_next;

            /// e. value
            V_pd[iA_pd] = value_of_choice_single_to_single(&C_priv[idx], &h[idx], &C_inter[idx], &Q[idx], C_tot_pd[iA_pd], t, il, M_pd[iA_pd], gender, &EV[idx_next], par, sol);
        }

        // 3. Apply liquidity constraint and upper envelope while interpolating onto common grid
        interpolate_to_exogenous_grid_single(t, il, gender, M_pd, C_tot_pd, V_pd, &C_tot[idx], &C_priv[idx], &h[idx], &C_inter[idx], &Q[idx], &V[idx], &EV[idx_next], sol, par);


        // 4. clean up
        // delete[] EmargU_pd;
        // delete[] C_tot_pd;
        // delete[] M_pd;
        EmargU_pd = nullptr;
        C_tot_pd = nullptr;
        M_pd = nullptr;
    }


    void calc_marginal_value_single_Agrid(int t, int gender, sol_struct* sol, par_struct* par){

        // unpack
        int const &num_A = par->num_A;

        // set index
        auto idx = index::single(t,0,par);

        // gender specific variables
        double* grid_A = par->grid_Aw;
        double* margV = &sol->EmargVw_start_as_single[idx];
        double* V      = &sol->EVw_start_as_single[idx];

        if (gender == man){
            grid_A = par->grid_Am;
            margV = &sol->EmargVm_start_as_single[idx];
            V      = &sol->EVm_start_as_single[idx];
        }

        // approximate marginal value by finite diff
        if (par->centered_gradient){
            for (int iA=1; iA<num_A-1; iA++){
                // Setup indices
                int iA_plus = iA + 1;
                int iA_minus = iA - 1;

                double denom = 1/(grid_A[iA_plus] - grid_A[iA_minus]);

                // Calculate finite difference
                margV[iA] = V[iA_plus]*denom - V[iA_minus]* denom; 
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
        auto idx_A_d = index::single_d(t,il,0,par);

        // get variables
        double* V = &sol->Vw_single_to_single[idx_A];
        double* Vd = &sol->Vwd_single_to_single[idx_A_d];
        double* labor = &sol->lw_single_to_single[idx_A];
        if (gender == man) {
            V = &sol->Vm_single_to_single[idx_A];
            Vd = &sol->Vmd_single_to_single[idx_A_d];
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
            // Set pointer to next period's expected value
            auto idx_next = index::single(t+1, 0, par);
            double* EV_next = (gender == man) ? &sol->EVm_start_as_single[idx_next] : &sol->EVw_start_as_single[idx_next];

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
            for (int il=0; il<par->num_l; il++) {
                int gender = sex == 0 ? woman : man;
                solve_choice_specific_single_to_single(t, il, gender, sol, par);
            }
        }
    }

     double repartner_surplus(double power, index::state_couple_struct* state_couple, index::state_single_struct* state_single, int gender, par_struct* par, sol_struct* sol){ //TODO: add index
        // unpack
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

        // Get indices if not provided
        if (iL_couple == -1){
            iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        }
        if (iA_couple == -1){
            iA_couple = tools::binary_search(0, par->num_A, par->grid_A, A_tot);
        }
        if (iA_single == -1){
            iA_single = tools::binary_search(0, par->num_A, grid_A_single, A);
        }

        //interpolate V_single_to_single
        auto idx_single = index::single(t,0,par);
        double Vsts = tools::interp_1d_index(grid_A_single, par->num_A, &V_single_to_single[idx_single], A, iA_single); 

        // interpolate couple V_single_to_couple  
        auto idx_couple = index::couple(t,0,0,0,par);
        double Vstc = tools::_interp_3d(par->grid_power, par->grid_love, par->grid_A, 
                                       par->num_power, par->num_love, par->num_A, 
                                       &V_single_to_couple[idx_couple], power, love, A_tot,
                                       iP, iL_couple, iA_couple);

        // surplus
        return Vstc - Vsts;
    }

    double calc_initial_bargaining_weight(int t, double love, double Aw, double Am, sol_struct* sol, par_struct* par, int iL_couple=-1){ //TODO: add index
        // state structs
        index::state_couple_struct* state_couple = new index::state_couple_struct;
        index::state_single_struct* state_single_w = new index::state_single_struct;
        index::state_single_struct* state_single_m = new index::state_single_struct;

        // couple
        state_couple->t = t;
        state_couple->love = love;
        state_couple->A = Aw+Am;
        state_couple->iA = tools::binary_search(0, par->num_A, par->grid_A, Aw+Am);
        if (iL_couple == -1){
            iL_couple = tools::binary_search(0, par->num_love, par->grid_love, love);
        }
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

        //solver input
        bargaining::nash_solver_struct* nash_struct = new bargaining::nash_solver_struct;
        nash_struct->surplus_func = repartner_surplus;
        nash_struct->state_couple = state_couple;
        nash_struct->state_single_w = state_single_w;
        nash_struct->state_single_m = state_single_m;
        nash_struct->sol = sol;
        nash_struct->par = par;

        // solve
        double init_mu =  bargaining::nash_bargain(nash_struct);

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
        auto idx_single = index::single(t,iA,par);

        // // b.1. loop over potential partners conditional on meeting a partner
        double Ev_cond = 0.0;
        double val = 0.0;
        for(int iL=0;iL<par->num_love;iL++){
            for(int iAp=0;iAp<par->num_A;iAp++){ // partner's wealth 

                // b.1.1. probability of meeting a specific type of partner
                auto idx_A = index::index2(iA,iAp,par->num_A,par->num_A);
                double prob_A = prob_partner_A[idx_A]; 
                double prob_love = par->prob_partner_love[iL]; 
                double prob = prob_A*prob_love;

                // only calculate if match has positive probability of happening
                if (prob>0.0) {
                    // Figure out gender
                    int iAw = iA;
                    int iAm = iAp;
                    if (gender==man) {
                        int iAw = iAp;
                        int iAm = iA;
                    }

                    // // b.1.2. bargain over consumption
                    double love = par->grid_love[iL];
                    double Aw = grid_A[iAw];
                    double Am = grid_A[iAm];
                    double power = calc_initial_bargaining_weight(t, love, Aw, Am, sol, par, iL);
                    
                    // b.1.3 Value conditional on meeting partner
                    if (power>=0.0){
                        double A_tot = Aw + Am;
                        auto idx_interp = index::couple(t, 0, 0, 0, par);
                        val = tools::interp_3d(par->grid_power, par->grid_love, par->grid_A, 
                                       par->num_power, par->num_love, par->num_A, 
                                       &V_single_to_couple[idx_interp], power, love, A_tot); //TODO: reuse index
                    } else {
                        val = V_single_to_single[idx_single];
                    }

                    // expected value conditional on meeting a partner
                    Ev_cond += prob*val;
                } // if
            } // iAp
        } // love 
        return Ev_cond;
    }


    void expected_value_start_single_Agrid(int t, int gender, sol_struct* sol,par_struct* par){

        // get index
        auto idx_A = index::single(t,0,par);

        // get variables
        double* EV_start_as_single = (gender == man) ? &sol->EVm_start_as_single[idx_A] : &sol->EVw_start_as_single[idx_A];
        double* EV_cond_meet_partner = (gender == man) ? &sol->EVm_cond_meet_partner[idx_A] : &sol->EVw_cond_meet_partner[idx_A];
        double* V_single_to_single = (gender == man) ? &sol->Vm_single_to_single[idx_A] : &sol->Vw_single_to_single[idx_A];

        for (int iA=0; iA<par->num_A;iA++){
            if (par->p_meet == 0.0) {
                // expected value of starting single is just value of remaining single
                EV_start_as_single[iA] = V_single_to_single[iA];
            } else {
                // a.1 Value conditional on meeting partner
                double EV_cond = expected_value_cond_meet_partner(t,iA,gender,sol,par);

                // a.2. expected value of starting single
                double p_meet = par->prob_repartner[t];
                auto idx_single = index::single(t,iA,par);
                EV_start_as_single[iA] = p_meet*EV_cond + (1.0-p_meet)*V_single_to_single[iA];
                EV_cond_meet_partner[iA] = EV_cond;
            } // if p_meet
        } // iA

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
        double *Vw_couple_to_single = &sol->Vw_couple_to_single[idx_A];
        double *Vm_couple_to_single = &sol->Vm_couple_to_single[idx_A];
        double *Vw_single_to_single = &sol->Vw_single_to_single[idx_A];
        double *Vm_single_to_single = &sol->Vm_single_to_single[idx_A];
        double *lw_couple_to_single = &sol->lw_couple_to_single[idx_A];
        double *lm_couple_to_single = &sol->lm_couple_to_single[idx_A];
        double *lw_single_to_single = &sol->lw_single_to_single[idx_A];
        double *lm_single_to_single = &sol->lm_single_to_single[idx_A];


        for (int iA=0; iA<par->num_A;iA++){
            //--- Indices and value update ---
            Vw_couple_to_single[iA] = Vw_single_to_single[iA] - par->div_cost;
            Vm_couple_to_single[iA] = Vm_single_to_single[iA] - par->div_cost;
            lw_couple_to_single[iA] = lw_single_to_single[iA];
            lm_couple_to_single[iA] = lm_single_to_single[iA];
        }
    }

    int find_interpolated_labor_index_single(int t, double A, int gender, sol_struct* sol, par_struct* par){

        //--- Set variables based on gender ---
        double* grid_A = (gender == woman) ? par->grid_Aw : par->grid_Am;
        double* Vd_single_to_single = (gender == woman) ? sol->Vwd_single_to_single : sol->Vmd_single_to_single;

        //--- Find asset index ---
        int iA = tools::binary_search(0, par->num_A, grid_A, A);

        //--- Initialize variables ---
        double maxV = -std::numeric_limits<double>::infinity();
        int labor_index = 0;

        //--- Loop over labor choices ---
        for (int il = 0; il < par->num_l; il++) {
            auto idx_interp = index::single_d(t, il, 0, par);
            double V_now = tools::interp_1d_index(grid_A, par->num_A, &Vd_single_to_single[idx_interp], A, iA);

            //--- Update maximum value and labor choice ---
            if (maxV < V_now) {
                maxV = V_now;
                labor_index = il;
            }
        }

        //--- Return optimal labor choice ---
        return labor_index;
    }

}
