#ifndef MAIN
#define SINGLE
#include "myheader.cpp"
#endif

namespace single {
    typedef struct {
        
        int il;             
        double M;             
        double *V_next;      
        int gender;          
        par_struct *par;      
        sol_struct *sol;      

    } solver_single_struct;

    double resources(double labor, double A,int gender,par_struct* par) {
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

    double value_of_choice_single_to_single(double* C_priv, double* h, double* C_inter, double* Q, double C_tot, int il, double M, int gender, double* V_next, par_struct* par, sol_struct* sol){
        double love = 0.0; // no love for singles
        double labor = par->grid_l[il];

        // intraperiod allocation
        precompute::intraperiod_allocation_single(C_priv, h, C_inter, Q, C_tot, il, gender, par, sol, par->precompute_intratemporal);
        
        // current utility from consumption allocation
        double lh = *h + labor;
        double Util = utils::util(*C_priv, lh, *Q, gender, par, love);

        // continuation value
        double *grid_A = par->grid_Aw; 
        if (gender==man){
            grid_A = par->grid_Am;
        }
        double A = M - C_tot;

        double Vnext = tools::interp_1d(grid_A,par->num_A,V_next,A);
        
        // return discounted sum
        return Util + par->beta*Vnext;
    }

    double objfunc_single_to_single(unsigned n, const double *x, double *grad, void *solver_data_in){
        double love = 0.0;

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double C_tot = x[0];
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
            il,
            M,
            gender,
            solver_data->V_next,
            par,
            sol
        );
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
    void handle_liquidity_constraint_single_to_single(int t, int il, int gender, double* m_vec, double* C_tot, double* C_priv, double* h, double* C_inter, double* Q, double* V, double* EV_next, sol_struct* sol, par_struct* par){
        // 1. Check if liquidity constraint binds
        // constraint: binding if common m is smaller than smallest m in endogenous grid
        double labor = par->grid_l[il];
        double* grid_A = par->grid_Aw;
        if (gender == man) {
            grid_A = par->grid_Am;
        }
        for (int iA = 0; iA < par->num_A; iA++){
            double M_now = resources(labor, grid_A[iA],gender,par);

            if (M_now < m_vec[0]){

                // a. Set total consumption equal to resources (consume all)
                C_tot[iA] = M_now;

                // c. Calculate value
                V[iA] = value_of_choice_single_to_single(&C_priv[iA], &h[iA], &C_inter[iA], &Q[iA], C_tot[iA], il, M_now, gender, EV_next, par, sol);

            }
        }
    }

    void do_upper_envelope_single_to_single(int t, int il, int gender, double* m_vec, double* c_vec, double* v_vec, double* C_tot, double* C_priv, double* h, double* C_inter, double* Q, double* V, double* EV_next, sol_struct* sol, par_struct* par){
        // Unpack
        double labor = par->grid_l[il];
        double* grid_A = par->grid_Aw;
        double* grid_A_pd = {par->grid_Aw_pd};
        if (gender == man){
            grid_A = par->grid_Am;
            grid_A_pd = par->grid_Am_pd;
        }

        // Loop through unsorted endogenous grid
        for (int iA_pd = 0; iA_pd<par->num_A_pd; iA_pd++){

            // 1. Unpack intervals
            double A_low = grid_A_pd[iA_pd];
            double A_high = grid_A_pd[iA_pd+1];
            
            double V_low = v_vec[iA_pd];
            double V_high = v_vec[iA_pd+1];

            double m_low = m_vec[iA_pd];
            double m_high = m_vec[iA_pd+1];

            double c_low = c_vec[iA_pd];
            double c_high = c_vec[iA_pd+1];

            // 2. Calculate slopes
            double v_slope = (V_high - V_low)/(A_high - A_low);
            double c_slope = (c_high - c_low)/(A_high - A_low);

            // 3. Loop through common grid
            for (int iA = 0; iA<par->num_A; iA++){

                // i. Check if resources from common grid are in current interval of endogenous grid
                double M_now = resources(labor, grid_A[iA], gender, par);
                bool interp = ((M_now >= m_low) & (M_now <= m_high));
                bool extrap_above = ((iA_pd == par->num_A_pd-2) & (M_now > m_vec[par->num_A_pd-1])); // extrapolate above last point in endogenous grid

                if (interp | extrap_above){

                    // ii. Interpolate consumption and value
                    double c_guess = c_low + c_slope*(M_now - m_low);
                    double a_guess = M_now - c_guess;
                    double V_guess = V_low + v_slope*(a_guess - A_low);

                    // iii. Update sol if v is higher than previous guess (upper envelope)
                    if (V_guess > V[iA]){
                        // o. Update total consumption
                        C_tot[iA] = c_guess;

                        // ooo. Update  intra-period allocation and value
                        V[iA] = value_of_choice_single_to_single(&C_priv[iA], &h[iA], &C_inter[iA], &Q[iA], C_tot[iA], il, M_now, gender, EV_next, par, sol);
                    }
                }
            }
        }
    }


    void EGM_single_to_single(int t, int il, int gender, sol_struct* sol, par_struct* par){
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
            V_pd[iA_pd] = value_of_choice_single_to_single(&C_priv[idx], &h[idx], &C_inter[idx], &Q[idx], C_tot_pd[iA_pd], il, M_pd[iA_pd], gender, &EV[idx_next], par, sol);
        }

        // 3. liquidity constraint
        handle_liquidity_constraint_single_to_single(t, il, gender, M_pd, &C_tot[idx], &C_priv[idx], &h[idx], &C_inter[idx], &Q[idx], &V[idx], &EV[idx_next], sol, par);

        // 4. upper envelope
        do_upper_envelope_single_to_single(t, il, gender, M_pd, C_tot_pd, V_pd, &C_tot[idx], &C_priv[idx], &h[idx], &C_inter[idx], &Q[idx], &V[idx], &EV[idx_next], sol, par);


        // 4. clean up
        // delete[] EmargU_pd;
        // delete[] C_tot_pd;
        // delete[] M_pd;
        EmargU_pd = nullptr;
        C_tot_pd = nullptr;
        M_pd = nullptr;
    }


    void calc_marginal_value_single(int t, int gender, sol_struct* sol, par_struct* par){

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



    void solve_choice_specific_single_to_single(int t, int il, sol_struct *sol,par_struct *par){
        double love = 0.0; // no love for singles 
        double labor = par->grid_l[il];
        
        // terminal period
        if (t == (par->T-1)){
            for (int iA=0; iA<par->num_A;iA++){
                auto idx = index::single_d(t,il,iA,par);

                double Aw = par->grid_Aw[iA];
                double Am = par->grid_Am[iA];

                sol->Cwd_tot_single_to_single[idx] = resources(labor, Aw,woman,par); 
                sol->Cmd_tot_single_to_single[idx] = resources(labor, Am,man,par); 
                // sol->lwd_single_to_single[idx] = ;  // OBS: Return to this when implementing labor choice
                // sol->lmd_single_to_single[idx] = ;  // OBS: Return to this when implementing labor choice

                // intraperiod allocation
                precompute::intraperiod_allocation_single(
                    &sol->Cwd_priv_single_to_single[idx],
                    &sol->hwd_single_to_single[idx],
                    &sol->Cwd_inter_single_to_single[idx],
                    &sol->Qwd_single_to_single[idx],
                    sol->Cwd_tot_single_to_single[idx],
                    il,
                    woman,
                    par,
                    sol,
                    par->precompute_intratemporal
                );

                precompute::intraperiod_allocation_single(
                    &sol->Cmd_priv_single_to_single[idx],
                    &sol->hmd_single_to_single[idx],
                    &sol->Cmd_inter_single_to_single[idx],
                    &sol->Qmd_single_to_single[idx],
                    sol->Cmd_tot_single_to_single[idx],
                    il,
                    man,
                    par,
                    sol,
                    par->precompute_intratemporal
                );

                // Calculate value
                sol->Vwd_single_to_single[idx] = utils::util(
                    sol->Cwd_priv_single_to_single[idx], 
                    sol->hmd_single_to_single[idx] + par->grid_l[il], 
                    sol->Qwd_single_to_single[idx],
                    woman, 
                    par, 
                    love);

                sol->Vmd_single_to_single[idx] = utils::util(
                    sol->Cmd_priv_single_to_single[idx], 
                    sol->hmd_single_to_single[idx] + par->grid_l[il], 
                    sol->Qmd_single_to_single[idx],
                    man, 
                    par, 
                    love);
            } // iA
        } else {
            if (par->do_egm) {
                EGM_single_to_single(t, il, woman, sol, par);
                EGM_single_to_single(t, il, man, sol, par);
            }
            else {
                #pragma omp parallel num_threads(par->threads)
                {

                    // 1. allocate objects for solver
                    solver_single_struct* solver_data = new solver_single_struct;
                    
                    int const dim = 1;
                    double lb[dim],ub[dim],x[dim];
                    
                    auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
                    double minf=0.0;

                    // 2. loop over assets
                    #pragma omp for
                    for (int iA=0; iA<par->num_A;iA++){
                        auto idx = index::single_d(t,il,iA,par);
                        auto idx_interp = index::single(t+1,0,par);
                        
                        // resources
                        double Aw = par->grid_Aw[iA];
                        double Am = par->grid_Am[iA];
                        
                        double Mw = resources(labor, Aw,woman,par); 
                        double Mm = resources(labor, Am,man,par); 
                        
                        // search over optimal total consumption, C
                        // WOMEN
                        // settings
                        solver_data->il = il;
                        solver_data->M = Mw;
                        solver_data->V_next = &sol->EVw_start_as_single[idx_interp]; // sol->EVw_start_single;
                        solver_data->gender = woman;
                        solver_data->par = par;
                        solver_data->sol = sol;
                        nlopt_set_min_objective(opt, objfunc_single_to_single, solver_data); 

                        // bounds
                        lb[0] = 1.0e-8;
                        ub[0] = solver_data->M;
                        nlopt_set_lower_bounds(opt, lb);
                        nlopt_set_upper_bounds(opt, ub);

                        // optimize
                        x[0] = solver_data->M/2.0;  // start_value
                        nlopt_optimize(opt, x, &minf); 

                        // unpack results
                        sol->Cwd_tot_single_to_single[idx] = x[0];
                        sol->Vwd_single_to_single[idx] = -minf;
                        precompute::intraperiod_allocation_single(
                            &sol->Cwd_priv_single_to_single[idx],
                            &sol->hwd_single_to_single[idx],
                            &sol->Cwd_inter_single_to_single[idx],
                            &sol->Qwd_single_to_single[idx],
                            sol->Cwd_tot_single_to_single[idx],
                            il,
                            woman,
                            par,
                            sol,
                            par->precompute_intratemporal
                        );

                        // MEN
                        // settings
                        solver_data->il = il;
                        solver_data->M = Mm;
                        solver_data->V_next = &sol->EVm_start_as_single[idx_interp]; // sol->EVm_start_singl;
                        solver_data->gender = man;
                        solver_data->par = par;
                        solver_data->sol = sol;
                        nlopt_set_min_objective(opt, objfunc_single_to_single, solver_data);
                            
                        // bounds
                        lb[0] = 1.0e-8;
                        ub[0] = solver_data->M;
                        nlopt_set_lower_bounds(opt, lb);
                        nlopt_set_upper_bounds(opt, ub);

                        // optimize
                        x[0] = solver_data->M/2.0; // start_value
                        nlopt_optimize(opt, x, &minf);

                        // unpack results
                        sol->Cmd_tot_single_to_single[idx] = x[0];
                        sol->Vmd_single_to_single[idx] = -minf;       
                        
                        // intraperiod allocation
                        precompute::intraperiod_allocation_single(
                            &sol->Cmd_priv_single_to_single[idx],
                            &sol->hmd_single_to_single[idx],
                            &sol->Cmd_inter_single_to_single[idx],
                            &sol->Qmd_single_to_single[idx],
                            sol->Cmd_tot_single_to_single[idx],
                            il,
                            man,
                            par,
                            sol,
                            par->precompute_intratemporal
                        );
                        
                    } // iA

                    // 4. destroy optimizer
                    nlopt_destroy(opt);
                    delete solver_data;

                } // pragma
            } // VFI /EGM
        }   // t
        
    }

    void find_unconditional_single_solution(int t, sol_struct* sol, par_struct* par){

        // Loop over states
        for (int iP=0; iP<par->num_power; iP++){
            for (int iL=0; iL<par->num_love; iL++){
                for (int iA=0; iA<par->num_A;iA++){
                    // get index
                    auto idx_single = index::index2(t,iA,par->T,par->num_A);

                    // Find maximum value over all labor choices
                    for (int il = 0; il < par->num_l; il++) {
                        auto idx_single_d = index::single_d(t,il,iA,par);

                        if (sol->Vw_single_to_single[idx_single] < sol->Vwd_single_to_single[idx_single_d]) {
                            sol->Vw_single_to_single[idx_single] = sol->Vwd_single_to_single[idx_single_d];
                            sol->lw_single_to_single[idx_single] = par->grid_l[il];
                        }
                        if (sol->Vm_single_to_single[idx_single] < sol->Vmd_single_to_single[idx_single_d]) {
                            sol->Vm_single_to_single[idx_single] = sol->Vmd_single_to_single[idx_single_d];
                            sol->lm_single_to_single[idx_single] = par->grid_l[il];
                        }
                    }
                    // // find optimal choices
                    // for (int il = 0; il < par->num_l; il++) {
                    //     auto index_choice =  index::single_d(t,il,iA,par);

                    //     if (list_choice_specific_values_w[il] == maxVw) {
                    //         // sol->Vw_single_to_single[idx] = sol->Vw_single_to_single[index_choice];
                    //         sol->Cw_tot_single_to_single[idx] = sol->Cw_tot_single_to_single[index_choice];
                    //         sol->lw_single_to_single[idx] = par->grid_l[il];
                    //         // sol->Cw_priv_single_to_single[idx] = sol->Cw_priv_single_to_single[index_choice];
                    //         // sol->hw_single_to_single[idx] = sol->hw_single_to_single[index_choice];
                    //         // sol->Cw_inter_single_to_single[idx] = sol->Cw_inter_single_to_single[index_choice];
                    //         // sol->Qw_single_to_single[idx] = sol->Qw_single_to_single[index_choice];
                    //     }
                    //     if (list_choice_specific_values_m[il] == maxVm) {
                    //         // sol->Vm_single_to_single[idx] = sol->Vm_single_to_single[index_choice];
                    //         sol->Cm_tot_single_to_single[idx] = sol->Cm_tot_single_to_single[index_choice];
                    //         sol->lm_single_to_single[idx] = par->grid_l[il];
                    //         // sol->Cm_priv_single_to_single[idx] = sol->Cm_priv_single_to_single[index_choice];
                    //         // sol->hm_single_to_single[idx] = sol->hm_single_to_single[index_choice];
                    //         // sol->Cm_inter_single_to_single[idx] = sol->Cm_inter_single_to_single[index_choice];
                    //         // sol->Qm_single_to_single[idx] = sol->Qm_single_to_single[index_choice];
                    //     }
                    // }
                
                } // wealth
            } // iP
        } // iL
    }

    void solve_single_to_single(int t, sol_struct *sol,par_struct *par){
        // 1. solve choice specific
        for (int il=0; il<par->num_l;il++){
            solve_choice_specific_single_to_single(t, il, sol, par);
        }
        find_unconditional_single_solution(t, sol, par);
    }


    void solve_couple_to_single(int t, sol_struct *sol, par_struct *par) {
        #pragma omp parallel num_threads(par->threads)
        {
            #pragma omp for
            for (int iA=0; iA<par->num_A;iA++){
                //--- Indices and value update ---
                auto idx = index::single(t,iA,par);

                sol->Vw_couple_to_single[idx] = sol->Vw_single_to_single[idx] - par->div_cost;
                sol->Vm_couple_to_single[idx] = sol->Vm_single_to_single[idx] - par->div_cost;
                sol->lw_couple_to_single[idx] = sol->lw_single_to_single[idx];
                sol->lm_couple_to_single[idx] = sol->lm_single_to_single[idx];

                //--- Find labor indices for woman and man ---
                int ilw = tools::binary_search(0, par->num_l, par->grid_l, sol->lw_couple_to_single[idx]);
                int ilm = tools::binary_search(0, par->num_l, par->grid_l, sol->lm_couple_to_single[idx]);
                auto idx_wd = index::single_d(t, ilw, iA, par);
                auto idx_md = index::single_d(t, ilm, iA, par);

                //--- Copy optimal choices from single solution ---
                sol->Cw_priv_couple_to_single[idx] = sol->Cwd_priv_single_to_single[idx_wd];
                sol->Cm_priv_couple_to_single[idx] = sol->Cmd_priv_single_to_single[idx_md];
                sol->hw_couple_to_single[idx] = sol->hwd_single_to_single[idx_wd]; 
                sol->hm_couple_to_single[idx] = sol->hmd_single_to_single[idx_md]; 
                sol->Cw_inter_couple_to_single[idx] = sol->Cwd_inter_single_to_single[idx_wd]; 
                sol->Cm_inter_couple_to_single[idx] = sol->Cmd_inter_single_to_single[idx_md]; 
                sol->Qw_couple_to_single[idx] = sol->Qwd_single_to_single[idx_wd]; 
                sol->Qm_couple_to_single[idx] = sol->Qmd_single_to_single[idx_md]; 
                sol->Cw_tot_couple_to_single[idx] = sol->Cwd_tot_single_to_single[idx_wd]; 
                sol->Cm_tot_couple_to_single[idx] = sol->Cmd_tot_single_to_single[idx_md]; 
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


    void expected_value_start_single(int t, sol_struct* sol,par_struct* par){

        // a. Loop over states
        for (int iA=0; iA<par->num_A;iA++){
            if (par->p_meet == 0.0) {
                // get index
                auto idx_single = index::single(t,iA,par);

                // expected value of starting single is just value of remaining single
                sol->EVw_start_as_single[idx_single] = sol->Vw_single_to_single[idx_single];
                sol->EVm_start_as_single[idx_single] = sol->Vm_single_to_single[idx_single];
                
            } else {
                // a.1 Value conditional on meeting partner
                double EVw_cond = expected_value_cond_meet_partner(t,iA,woman,sol,par);
                double EVm_cond = expected_value_cond_meet_partner(t,iA,man,sol,par);

                // a.2. expected value of starting single
                double p_meet = par->prob_repartner[t];
                auto idx_single = index::single(t,iA,par);
                sol->EVw_start_as_single[idx_single] = p_meet*EVw_cond + (1.0-p_meet)*sol->Vw_single_to_single[idx_single];
                sol->EVm_start_as_single[idx_single] = p_meet*EVm_cond + (1.0-p_meet)*sol->Vm_single_to_single[idx_single];

                sol->EVw_cond_meet_partner[idx_single] = EVw_cond;
                sol->EVm_cond_meet_partner[idx_single] = EVm_cond;
            } // if p_meet
        } // iA

        if (par->do_egm){
            calc_marginal_value_single(t, woman, sol, par);
            calc_marginal_value_single(t, man, sol, par);
        }
    }


}
