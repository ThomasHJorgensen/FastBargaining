
// functions for solving model for couples.
#ifndef MAIN
#define COUPLE
#include "myheader.cpp"
#endif

namespace couple {
    
    typedef struct {
        int t;              
        int ilw;              
        int ilm;              
        int iL;             
        int iP;             
        double M;           
        double *EVw_next;    
        double *EVm_next;    

        sol_struct *sol;
        par_struct *par;

    } solver_couple_struct;

    double calc_marital_surplus(double Vd_couple_to_couple,double V_couple_to_single,par_struct* par){
        return Vd_couple_to_couple - V_couple_to_single;
    }

    double resources(double labor_w, double labor_m, double A,par_struct* par) {
        if ((labor_w == 0.0) && (labor_m == 0.0)){
            // no labor income, just resources from assets
            return par->R*A + 1.0e-4; // add a small amount to avoid errors with zero
        }
        
        double K_w = 5.0;
        double K_m = 5.0;

        double wage_w = utils::wage(K_w, woman, par);
        double wage_m = utils::wage(K_m, man, par);

        return par->R*A + wage_w*labor_w + wage_m*labor_m;
    }

    double value_of_choice_couple_to_couple(double* Cw_priv, double* Cm_priv, double* hw, double* hm, double* C_inter, double* Q,
        int ilw, int ilm, double C_tot, double M_resources, int t,int iL, int iP, double* Vw,double* Vm, double* EVw_next,double* EVm_next ,par_struct* par, sol_struct* sol){
            // double* Cw_priv,double* Cm_priv,double* C_pub,double* Vw,double* Vm,  double C_tot,int t,double M_resources,int iL,int iP,double* EVw_next,double* EVm_next,sol_struct *sol, par_struct *par){
        
        double love = par->grid_love[iL];
        double power = par->grid_power[iP];

        // current utility from consumption allocation
        precompute::intraperiod_allocation_couple(Cw_priv, Cm_priv, hw, hm, C_inter, Q, 
            ilw, ilm,C_tot,power, 
            par, sol,
            par->precompute_intratemporal
        );
        // Note Vw and Vm are not a vector, just a pointer to one value
        Vw[0] = utils::util(*Cw_priv, *hw + par->grid_l[ilw], *Q, woman, par, love); 
        Vm[0] = utils::util(*Cm_priv, *hm + par->grid_l[ilm], *Q, man, par, love); 

        // add continuation value
        if (t < (par->T-1)){
            double savings = M_resources - C_tot ;
            double EVw_plus = 0.0;
            double EVm_plus = 0.0;

            EVw_plus = tools::interp_1d(par->grid_A, par->num_A, EVw_next, savings);
            EVm_plus = tools::interp_1d(par->grid_A, par->num_A, EVm_next, savings);

            Vw[0] += par->beta*EVw_plus;
            Vm[0] += par->beta*EVm_plus;
        }

        
        // return
        return power*Vw[0] + (1.0-power)*Vm[0];
    }

    //////////////////
    // VFI solution //
    double objfunc_couple_to_couple(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        solver_couple_struct *solver_data = (solver_couple_struct *) solver_data_in;

        double C_tot = x[0];

        int t = solver_data->t;
        int ilw = solver_data->ilw;
        int ilm = solver_data->ilm;
        int iL = solver_data->iL;
        int iP = solver_data->iP;
        double M = solver_data->M;
        double *EVw_next = solver_data->EVw_next;
        double *EVm_next = solver_data->EVm_next;

        sol_struct *sol = solver_data->sol;
        par_struct *par = solver_data->par;

        // return negative of value
        double Cw_priv,Cm_priv,hw, hm, C_inter, Q ,Vw,Vm;
        return - value_of_choice_couple_to_couple(
            &Cw_priv, &Cm_priv, &hw, &hm, &C_inter, &Q,
            ilw, ilm, C_tot, M, t, iL, iP, 
            &Vw, &Vm, EVw_next, EVm_next,
            par, sol
        );
    }

    void solve_couple_to_couple(
        double* Cw_priv,double* Cm_priv, double* hw, double* hm,double* C_inter, double* Q, double* Vw,double* Vm,
        double M_resources, int t, int ilw, int ilm, int iL, int iP, 
        double* EVw_next,double* EVm_next,
        double starting_val,sol_struct *sol,par_struct *par){
        // double* Vw,double* Vm , int t,double M_resources,int iL,int iP,double* EVw_next,double* EVm_next,double starting_val,sol_struct *sol,par_struct *par){


        double C_tot = M_resources;
        
        if (t<(par->T-1)){ 
            // objective function
            int const dim = 1;
            double lb[dim],ub[dim],x[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
            double minf=0.0;

            solver_couple_struct* solver_data = new solver_couple_struct;
            solver_data->t = t;
            solver_data->ilw = ilw;
            solver_data->ilm = ilm;
            solver_data->iL = iL;
            solver_data->iP = iP;
            solver_data->M = M_resources;
            solver_data->EVw_next = EVw_next;
            solver_data->EVm_next = EVm_next;

            solver_data->sol = sol;
            solver_data->par = par;
            nlopt_set_min_objective(opt, objfunc_couple_to_couple, solver_data);
                
            // bounds
            lb[0] = 1.0e-6;
            ub[0] = solver_data->M - 1.0e-6;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            // optimize
            x[0] = starting_val;
            nlopt_optimize(opt, x, &minf);
            nlopt_destroy(opt);

            C_tot = x[0];
            
            delete solver_data;
        }

        // implied consumption allocation (re-calculation)
        double _ = value_of_choice_couple_to_couple(
            Cw_priv, Cm_priv, hw, hm, C_inter, Q,
            ilw, ilm, C_tot, 
            M_resources, t,iL, iP, 
            Vw,Vm, EVw_next,EVm_next,
            par, sol
        );
    }
    
    void solve_couple_to_couple_Agrid_vfi(int t, int ilw, int ilm, int iP, int iL, double* EVw_next, double* EVm_next,sol_struct* sol, par_struct* par){
        double labor_w = par->grid_l[ilw];
        double labor_m = par->grid_l[ilm];
        
        for (int iA=0; iA<par->num_A;iA++){
            auto idx = index::couple_d(t,ilw,ilm,iP,iL,iA,par);

            double M_resources = resources(labor_w, labor_m, par->grid_A[iA],par); 

            // starting values
            double starting_val = M_resources * 0.8;
            if (iA>0){ 
                auto idx_last = index::couple_d(t,ilw,ilm,iP,iL,iA-1,par);
                starting_val = sol->Cwd_priv_couple_to_couple[idx_last] + sol->Cmd_priv_couple_to_couple[idx_last] + sol->Cd_inter_couple_to_couple[idx_last];
            }

            // solve unconstrained problem
            solve_couple_to_couple(
                &sol->Cwd_priv_couple_to_couple[idx], &sol->Cmd_priv_couple_to_couple[idx], 
                &sol->hwd_couple_to_couple[idx], &sol->hmd_couple_to_couple[idx], 
                &sol->Cd_inter_couple_to_couple[idx], &sol->Qd_couple_to_couple[idx],
                &sol->Vwd_couple_to_couple[idx], &sol->Vmd_couple_to_couple[idx],
                M_resources, t, ilw, ilm, iL, iP, 
                EVw_next, EVm_next, starting_val,
                sol, par
            );
            sol->Cd_tot_couple_to_couple[idx] = sol->Cwd_priv_couple_to_couple[idx] + sol->Cmd_priv_couple_to_couple[idx] + sol->Cd_inter_couple_to_couple[idx];

        } // wealth   
    }

    ////////////////////////////// EGM numerical solution //////////////////////////////
    double marg_util_C_couple(double C_tot, int ilw, int ilm, int iP, par_struct* par, sol_struct* sol, double guess_Cw_priv, double guess_Cm_priv){
        // baseline utility (could be passed as argument to avoid recomputation of utility at C_tot)

        double power = par->grid_power[iP];
        double love = 0.0; // does not matter for the marginal utility

        // OBS: Implement start values for Cw_priv and Cm_priv as well as hw, hm (and C_inter?)

        double util = precompute::util_C_couple(C_tot,ilw, ilm, power, love, par, sol, par->precompute_intratemporal);

        // forward difference
        double delta = 0.0001;
        double util_delta = precompute::util_C_couple(C_tot + delta,ilw, ilm, power, love, par, sol, par->precompute_intratemporal);
        return (util_delta - util)/delta;
    }
    
    // numerical inverse marginal utility
    typedef struct { 
        int ilw;
        int ilm;
        double margU;
        int iP;
        int gender;
        par_struct *par;
        sol_struct *sol;
        bool do_print;

        double guess_Cw_priv;
        double guess_Cm_priv;
    } solver_inv_struct_couple;

    double obj_inv_marg_util_couple(unsigned n, const double *x, double *grad, void *solver_data_in){
         // unpack
        solver_inv_struct_couple *solver_data = (solver_inv_struct_couple *) solver_data_in; 
        
        double C_tot = x[0];
        double ilw = solver_data->ilw;
        double ilm = solver_data->ilm;
        double margU = solver_data->margU;
        int iP = solver_data->iP;
        double start_Cw_priv = solver_data->guess_Cw_priv;//C_tot/3.0;
        double start_Cm_priv = solver_data->guess_Cm_priv;//C_tot/3.0;
        bool do_print = solver_data->do_print;
        par_struct *par = solver_data->par;
        sol_struct *sol = solver_data->sol;

        // clip
        double penalty = 0.0;
        if (C_tot <= 0.0) {
            penalty += 1000.0*C_tot*C_tot;
            C_tot = 1.0e-6;
        }

        // return squared difference
        double diff = marg_util_C_couple(C_tot,ilm, ilw, iP,par,sol, start_Cw_priv, start_Cm_priv) - margU;

        if (do_print){
            logs::write("inverse_log.txt",1,"C_tot: %f, diff: %f, penalty: %f\n",C_tot,diff,penalty);
        }
        return diff*diff + penalty;

    }

    double inv_marg_util_couple(double margU, int ilw, int ilm, int iP,par_struct* par, sol_struct* sol, double guess_Ctot, double guess_Cw_priv, double guess_Cm_priv,bool do_print=false ){
        // setup numerical solver
        solver_inv_struct_couple* solver_data = new solver_inv_struct_couple;  
                
        int const dim = 1;
        double lb[dim],ub[dim],x[dim];   
        
        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT NLOPT_LN_BOBYQA
        double minf=0.0;

        // search over optimal total consumption, C
        // settings
        solver_data->ilw = ilw;
        solver_data->ilm = ilm;
        solver_data->margU = margU;         
        solver_data->iP = iP;
        solver_data->par = par;
        solver_data->sol = sol;
        solver_data->do_print = do_print;    

        solver_data->guess_Cw_priv = guess_Cw_priv;
        solver_data->guess_Cm_priv = guess_Cm_priv;


        if (do_print){
            logs::write("inverse_log.txt",0,"margU: %f\n",margU);
        }

        nlopt_set_min_objective(opt, obj_inv_marg_util_couple, solver_data);   
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, 1.0e-6);
        nlopt_set_xtol_rel(opt, 1.0e-5);

        // bounds
        lb[0] = 0.0;  
        ub[0] = 2.0*par->max_Ctot;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        // optimize
        x[0] = guess_Ctot; 
        nlopt_optimize(opt, x, &minf);          
        nlopt_destroy(opt);   

        delete solver_data;              
        
        // return consumption value
        return x[0];
        
    }

    //////////////////
    // EGM solution //

    void handle_liquidity_constraint_couple_to_couple(int t, int ilw, int ilm, int iP, int iL, double* m_vec, double* EmargUd_pd, double* C_tot, double* Cw_priv, double* Cm_priv, double* hw, double* hm, double* C_inter, double* Q, double* Vw,double* Vm, double* EVw_next, double* EVm_next, double* V, sol_struct* sol, par_struct* par){
        // 1. Check if liquidity constraint binds
        // constraint: binding if common m is smaller than smallest m in endogenous grid 
        for (int iA=0; iA < par->num_A; iA++){
            double M_now = resources(par->grid_l[ilw], par->grid_l[ilm], par->grid_A[iA],par);

            if (M_now < m_vec[0]){

                // a. Set total consumption equal to resources (consume all)
                C_tot[iA] = M_now;

                // b. Calculate intra-period allocation
                double _ = value_of_choice_couple_to_couple(
                    &Cw_priv[iA], &Cm_priv[iA], &hw[iA], &hm[iA], &C_inter[iA], &Q[iA],
                    ilw, ilm, C_tot[iA], 
                    M_now, t,iL, iP, 
                    &Vw[iA], &Vm[iA], EVw_next, EVm_next,
                    par, sol
                );

                // c. Calculate value
                double power = par->grid_power[iP];
                V[iA] = power*Vw[iA] + (1-power)*Vm[iA];
            }
        }
    }

    void do_upper_envelope_couple_to_couple(int t, int ilw, int ilm, int iP, int iL, double* m_vec, double* c_vec, double* v_vec, double* EmargUd_pd, double* C_tot, double* Cw_priv, double* Cm_priv, double* hw, double* hm, double* C_inter, double* Q, double* Vw,double* Vm, double* EVw_next, double* EVm_next, double* V, sol_struct* sol, par_struct* par){

        // Loop through unsorted endogenous grid
        for (int iA_pd = 0; iA_pd<par->num_A_pd-1;iA_pd++){

            // 1. Unpack intervals
            double A_low = par->grid_A_pd[iA_pd];
            double A_high = par->grid_A_pd[iA_pd+1];

            double V_low = v_vec[iA_pd];
            double V_high = v_vec[iA_pd+1];

            double m_low = m_vec[iA_pd];
            double m_high = m_vec[iA_pd+1];

            double c_low = c_vec[iA_pd];
            double c_high = c_vec[iA_pd+1];

            // 2. Calculate slopes
            double v_slope = (V_high - V_low)/(A_high - A_low);
            double c_slope = (c_high - c_low)/(m_high - m_low);

            // 3. Loop through common grid
            for (int iA = 0; iA<par->num_A; iA++){

                // i. Check if resources from common grid are in current interval of endogenous grid
                double M_now = resources(par->grid_l[ilw], par->grid_l[ilm], par->grid_A[iA],par);
                bool interp = (M_now >= m_low) && (M_now <= m_high); 
                bool extrap_above = (iA_pd == par->num_A_pd-2) && (M_now>m_vec[par->num_A_pd-1]); // extrapolate above last point in endogenous grid
                if (interp || extrap_above){

                    // ii. Interpolate consumption and value
                    double c_guess = c_low + c_slope*(M_now - m_low);
                    double a_guess = M_now - c_guess;
                    double V_guess = V_low + v_slope*(a_guess - A_low);

                    // iii. Update sol if V is higher than previous guess (upper envelope)
                    if (V_guess > V[iA]){

                        // o. Update total consumption
                        C_tot[iA] = c_guess;
                        
                        // oo. Update intra-period allocation
                        double _ = value_of_choice_couple_to_couple(
                            &Cw_priv[iA], &Cm_priv[iA], &hw[iA], &hm[iA], &C_inter[iA], &Q[iA],
                            ilw, ilm, C_tot[iA], 
                            M_now, t,iL, iP, 
                            &Vw[iA], &Vm[iA], EVw_next, EVm_next,
                            par, sol
                        );
                        // ooo. Update value
                        V[iA] = par->grid_power[iP]*Vw[iA] + (1-par->grid_power[iP])*Vm[iA];
                    }
                }
            }
        }
    }

    void solve_couple_to_couple_Agrid_egm(int t, int ilw, int ilm, int iP, int iL, double* EVw_next, double* EVm_next, double* EmargV_next,sol_struct* sol, par_struct* par){
        
        // 1. Solve terminal period with VFI
        if(t==(par->T-1)){
            solve_couple_to_couple_Agrid_vfi(t,ilw, ilm, iP,iL,EVw_next,EVm_next,sol,par);

        // 2. Solve remaining periods with EGM
        } else {
            // Solve on endogenous grid
            double Cw_priv = 0.0;
            double Cm_priv = 0.0;
            double hw = 0.0;
            double hm = 0.0;
            double C_inter = 0.0;
            double Q = 0.0;
            double Vw = 0.0;
            double Vm = 0.0;

            for (int iA_pd=0; iA_pd<par->num_A_pd;iA_pd++){

                // i. Unpack
                double A_next = par->grid_A_pd[iA_pd]; // assets next period
                auto idx_pd = index::couple_pd(t,ilw, ilm, iP,iL,iA_pd,par);
                auto idx_interp = index::index4(ilw, ilm, iP,0, par->num_l, par->num_l, par->num_power,par->num_marg_u);

                // ii. interpolate marginal utility
                sol->EmargUd_pd[idx_pd] = par->beta * tools::interp_1d(par->grid_A, par->num_A, EmargV_next, A_next);

                // iii. Get total consumption by interpolation of pre-computed inverse marginal utility (coming from Euler)
                if (strcmp(par->interp_method,"numerical")==0){
                    
                    // starting values
                    double guess_Ctot = 3.0;
                    double guess_Cw_priv = guess_Ctot/3.0;
                    double guess_Cm_priv = guess_Ctot/3.0;
                    if(iA_pd>0){
                        // last found solution
                        guess_Ctot = sol->Cd_tot_pd[index::couple_pd(t,ilw,ilm,iP,iL,iA_pd-1,par)];
                        guess_Cw_priv = Cw_priv;
                        guess_Cm_priv = Cm_priv;

                    } else if (t<(par->T-2)) {
                        guess_Ctot = sol->Cd_tot_pd[index::couple_pd(t+1,ilw,ilm,iP,iL,iA_pd,par)];
                    }
                    // OBS: starting values are not used. Also, there should be starting values for hw, hm, and C_inter as well
                    sol->Cd_tot_pd[idx_pd] = inv_marg_util_couple(sol->EmargUd_pd[idx_pd],ilw, ilm, iP,par,sol, guess_Ctot, guess_Cw_priv, guess_Cm_priv); // numerical inverse

                } else {
                    if(strcmp(par->interp_method,"linear")==0){
                        sol->Cd_tot_pd[idx_pd] = tools::interp_1d(&par->grid_marg_u_couple_for_inv[idx_interp],par->num_marg_u,par->grid_inv_marg_u,sol->EmargUd_pd[idx_pd]);
                    }

                    if (par->interp_inverse){
                        sol->Cd_tot_pd[idx_pd] = 1.0/sol->Cd_tot_pd[idx_pd];
                    }
                }
                
                // iv. Get endogenous grid points
                sol->Md_pd[idx_pd] = A_next + sol->Cd_tot_pd[idx_pd];

                // v. Get post-choice value (also updates the intra-period allocation)
                sol->Vd_couple_to_couple_pd[idx_pd] = value_of_choice_couple_to_couple(
                        &Cw_priv, &Cm_priv, &hw, &hm, &C_inter, &Q,
                        ilw, ilm, sol->Cd_tot_pd[idx_pd], sol->Md_pd[idx_pd], t, iL, iP, 
                        &Vw, &Vm, EVw_next, EVm_next,
                        par, sol
                    );
            }

            // 3. Apply upper envelope and interpolate onto common grid
            auto idx_interp_pd = index::couple_pd(t,ilw,ilm,iP,iL,0,par);
            auto idx_interp = index::couple_d(t,ilw,ilm,iP,iL,0,par);

            handle_liquidity_constraint_couple_to_couple(t, ilw, ilm, iP, iL, &sol->Md_pd[idx_interp_pd], &sol->EmargUd_pd[idx_interp_pd], 
                                                         &sol->Cd_tot_couple_to_couple[idx_interp], 
                                                         &sol->Cwd_priv_couple_to_couple[idx_interp], &sol->Cmd_priv_couple_to_couple[idx_interp], 
                                                         &sol->hwd_couple_to_couple[idx_interp], &sol->hmd_couple_to_couple[idx_interp], 
                                                         &sol->Cd_inter_couple_to_couple[idx_interp], &sol->Qd_couple_to_couple[idx_interp],
                                                         &sol->Vwd_couple_to_couple[idx_interp], &sol->Vmd_couple_to_couple[idx_interp], 
                                                         EVw_next, EVm_next, &sol->Vd_couple_to_couple[idx_interp], sol, par);
            do_upper_envelope_couple_to_couple( t, ilw, ilm, iP, iL, 
                                                &sol->Md_pd[idx_interp_pd], &sol->Cd_tot_pd[idx_interp_pd], &sol->Vd_couple_to_couple_pd[idx_interp_pd], 
                                                &sol->EmargUd_pd[idx_interp_pd], &sol->Cd_tot_couple_to_couple[idx_interp], 
                                                &sol->Cwd_priv_couple_to_couple[idx_interp] ,&sol->Cmd_priv_couple_to_couple[idx_interp],
                                                &sol->hwd_couple_to_couple[idx_interp], &sol->hmd_couple_to_couple[idx_interp], 
                                                &sol->Cd_inter_couple_to_couple[idx_interp], &sol->Qd_couple_to_couple[idx_interp],
                                                &sol->Vwd_couple_to_couple[idx_interp], &sol->Vmd_couple_to_couple[idx_interp], 
                                                EVw_next, EVm_next, &sol->Vd_couple_to_couple[idx_interp], 
                                                sol, par);

        } // period check        
    }


    void calc_expected_value_couple(int t, int iP, int iL, int iA, double* Vw, double* Vm, double* EVw, double* EVm, sol_struct* sol, par_struct* par){
                
        double love = par->grid_love[iL];
        double power = par->grid_power[iP];
        auto idx = index::couple(t,iP,iL,iA,par);
        auto delta_love = index::couple(t,iP,1,iA,par) - index::couple(t,iP,0,iA,par);
        double Eval_w = 0;
        double Eval_m = 0;        
        double Vw_now = 0;
        double Vm_now = 0;        
        for (int i_love_shock=0; i_love_shock<par->num_shock_love; i_love_shock++){
            double love_next = love + par->grid_shock_love[i_love_shock];
            double weight = par->grid_weight_love[i_love_shock];
            auto idx_love = tools::binary_search(0,par->num_love,par->grid_love,love_next);
            double maxV = -std::numeric_limits<double>::infinity();
            double maxVw = -std::numeric_limits<double>::infinity();
            double maxVm = -std::numeric_limits<double>::infinity();
            

            auto idx_interp = index::couple(t,iP,0,iA,par);
            
            Vw_now = tools::interp_1d_index_delta(par->grid_love, par->num_love, &Vw[idx_interp], love_next, idx_love, delta_love);
            Vm_now = tools::interp_1d_index_delta(par->grid_love, par->num_love, &Vm[idx_interp], love_next, idx_love, delta_love);


            // add to expected value
            Eval_w += weight*Vw_now;
            Eval_m += weight*Vm_now;
        }

        EVw[idx] = Eval_w;
        EVm[idx] = Eval_m;
    }

    void calc_marginal_value_couple(double power, double* Vw, double* Vm, double* margV, sol_struct* sol, par_struct* par){

        // approximate marginal value of marriage by finite diff
        if (par->centered_gradient){
            for (int iA=1; iA<=par->num_A-2;iA++){
                // Setup indices
                int iA_plus = iA + 1;
                int iA_minus = iA - 1;

                // Calculate finite difference
                double margVw {0};
                double margVm {0};
                double denom = 1.0/(par->grid_A[iA_plus] - par->grid_A[iA_minus]);
                margVw = Vw[iA_plus]*denom - Vw[iA_minus]*denom;
                margVm = Vm[iA_plus]*denom - Vm[iA_minus]*denom;

                // Update solution
                margV[iA] = power*margVw + (1.0-power)*margVm;

                // Extrapolate gradient in last point
                if (iA == par->num_A-2){
                    margV[iA_plus] = margV[iA];
                }
            }

            // Extrapolate gradient in end points
            int i=0;
            margV[i] = (margV[i+2] - margV[i+1]) / (par->grid_A[i+2] - par->grid_A[i+1]) * (par->grid_A[i] - par->grid_A[i+1]) + margV[i+1];
            i = par->num_A-1;
            margV[i] = (margV[i-2] - margV[i-1]) / (par->grid_A[i-2] - par->grid_A[i-1]) * (par->grid_A[i] - par->grid_A[i-1]) + margV[i-1];
        }
        else {
            for (int iA=0; iA<=par->num_A-2;iA++){
                // Setup indices
                int iA_plus = iA + 1;

                // Calculate finite difference
                double margVw {0};
                double margVm {0};
                double denom = 1.0/(par->grid_A[iA_plus] - par->grid_A[iA]);
                margVw = Vw[iA_plus]*denom - Vw[iA]*denom;
                margVm = Vm[iA_plus]*denom - Vm[iA]*denom;

                // Update solution
                margV[iA] = power*margVw + (1.0-power)*margVm;

                // Extrapolate gradient in last point
                if (iA == par->num_A-2){
                    margV[iA_plus] = margV[iA];
                }
            }
        }
    }

    void solve_choice_specific_couple(int t, int ilw, int ilm, sol_struct *sol,par_struct *par){
        
        #pragma omp parallel num_threads(par->threads)
        { 
            // 2. solve for values of remaining a couple
            #pragma omp for
            for (int iP=0; iP<par->num_power; iP++){

                // Get next period continuation values
                double *EVw_next = nullptr;  
                double *EVm_next = nullptr;
                double *EmargV_next = nullptr;
                // solve
                for (int iL=0; iL<par->num_love; iL++){
                    if (t<(par->T-1)){
                        auto idx_next = index::couple(t+1,iP,iL,0,par);
                        EVw_next = &sol->EVw_start_as_couple[idx_next];  
                        EVm_next = &sol->EVm_start_as_couple[idx_next];
                        EmargV_next = &sol->EmargV_start_as_couple[idx_next];
                    }

                    if (par->do_egm){
                        solve_couple_to_couple_Agrid_egm(t,ilw,ilm,iP,iL,EVw_next,EVm_next, EmargV_next,sol,par); 

                    } else {
                        solve_couple_to_couple_Agrid_vfi(t,ilw,ilm,iP,iL,EVw_next,EVm_next,sol,par); 

                    }
                } // love
            } // power
        } // omp
    }

    void find_interpolated_labor_index_couple(int t, double power, double love, double A, double* ilw_out, double* ilm_out, sol_struct* sol, par_struct* par){

        //--- Find index ---
        int iP = tools::binary_search(0, par->num_power, par->grid_power, power);
        int iL = tools::binary_search(0, par->num_love, par->grid_love, love);
        int iA = tools::binary_search(0, par->num_A, par->grid_A, A);

        //--- Initialize variables ---
        double maxV = -std::numeric_limits<double>::infinity();
        int labor_index_w = 0;
        int labor_index_m = 0;

        //--- Loop over labor choices ---
        for (int ilw = 0; ilw < par->num_l; ilw++) {
            for (int ilm = 0; ilm < par->num_l; ilm++) {

                // get index for choice specific value
                auto idx_couple_d = index::couple_d(t,ilw,ilm,iP,iL,iA,par); 


                //--- Interpolate value ---
                auto idx_interp = index::couple_d(t, ilw, ilm, 0,0,0, par);
                double Vw_now = tools::_interp_3d(
                    par->grid_power, par->grid_love, par->grid_A,
                    par->num_power, par->num_love, par->num_A,
                    &sol->Vwd_couple_to_couple[idx_interp],
                    power, love, A,
                    iP, iL, iA
                );
                double Vm_now = tools::_interp_3d(
                    par->grid_power, par->grid_love, par->grid_A,
                    par->num_power, par->num_love, par->num_A,
                    &sol->Vmd_couple_to_couple[idx_interp],
                    power, love, A,
                    iP, iL, iA
                );

                // max V over labor choices 
                double V_now = power*Vw_now + (1.0-power)*Vm_now;

                //--- Update maximum value and labor choice ---
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

    void find_unconditional_couple_solution(int t, sol_struct* sol, par_struct* par){
                
        for (int iP=0; iP<par->num_power; iP++){
            double power = par->grid_power[iP];
            for (int iL=0; iL<par->num_love; iL++){
                for (int iA=0; iA<par->num_A;iA++){
                    // get index
                    auto idx_couple = index::couple(t,iP,iL,iA,par);
                    
                    // loop over labor choices
                    for (int ilw = 0; ilw < par->num_l; ilw++) {
                        for (int ilm = 0; ilm < par->num_l; ilm++) {
                            // get index for choice specific value
                            auto idx_couple_d = index::couple_d(t,ilw,ilm,iP,iL,iA,par); 

                            // max V over labor choices
                            double maxV_d = power*sol->Vwd_couple_to_couple[idx_couple_d] + (1.0-power)*sol->Vmd_couple_to_couple[idx_couple_d];
                            if (sol->V_couple_to_couple[idx_couple] < maxV_d) {
                                sol->V_couple_to_couple[idx_couple] = maxV_d;
                                sol->Vw_couple_to_couple[idx_couple] = sol->Vwd_couple_to_couple[idx_couple_d];
                                sol->Vm_couple_to_couple[idx_couple] = sol->Vmd_couple_to_couple[idx_couple_d];
                                sol->lw_couple_to_couple[idx_couple] = par->grid_l[ilw];
                                sol->lm_couple_to_couple[idx_couple] = par->grid_l[ilm];
                            }
                        }
                    }

                    // // update solution
                    // for (int ilw = 0; ilw < par->num_l; ilw++) {
                    //     for (int ilm = 0; ilm < par->num_l; ilm++) {
                    //         auto i = index::index2(ilw, ilm, par->num_l, par->num_l);
                    //         auto idx_choice = index::couple_d(t,ilw,ilm,iP,iL,iA,par);

                    //         if (list_choice_specific_values[i] == maxV) {
                    //             // sol->Vw_couple_to_couple[idx] = sol->Vw_couple_to_couple[idx_choice];
                    //             // sol->Vm_couple_to_couple[idx] = sol->Vm_couple_to_couple[idx_choice];
                    //             sol->C_tot_couple_to_couple[idx] = sol->Cd_tot_couple_to_couple[idx_choice];
                    //             sol->lw_couple_to_couple[idx] = par->grid_l[ilw];
                    //             sol->lm_couple_to_couple[idx] = par->grid_l[ilm];
                    //             // sol->Cw_priv_couple_to_couple[idx] = sol->Cw_priv_couple_to_couple[idx_choice];
                    //             // sol->Cm_priv_couple_to_couple[idx] = sol->Cm_priv_couple_to_couple[idx_choice];
                    //             // sol->hw_couple_to_couple[idx] = sol->hw_couple_to_couple[idx_choice];
                    //             // sol->hm_couple_to_couple[idx] = sol->hm_couple_to_couple[idx_choice];
                    //             // sol->C_inter_couple_to_couple[idx] = sol->C_inter_couple_to_couple[idx_choice];
                    //             // sol->Q_couple_to_couple[idx] = sol->Q_couple_to_couple[idx_choice];
                    //         }
                    //     } // ilm
                    // } // ilw
                } // iA
            } // iL
        } // iP
    }

    void solve_start_as_couple(int t, sol_struct *sol,par_struct *par){

        // #pragma omp parallel num_threads(par->threads)
        // {
            // 1. Setup
            /// a. lists
            int num = 2;
            double** list_start_as_couple = new double*[num]; 
            double** list_couple_to_couple = new double*[num];
            double* list_couple_to_single = new double[num];             

            // b. temporary arrays
            double* Sw = new double[par->num_power];
            double* Sm = new double[par->num_power];

            // c. index struct to pass to bargaining algorithm
            index::index_couple_struct* idx_couple_fct = new index::index_couple_struct;
            
            // Solve for values of starting as couple (check participation constraints)
            // #pragma omp for
            for (int iL=0; iL<par->num_love; iL++){    
                for (int iA=0; iA<par->num_A;iA++){
                    // i. Get indices
                    auto idx_single = index::single(t,iA,par);
                    idx_couple_fct->t = t;
                    idx_couple_fct->iL = iL;
                    idx_couple_fct->iA = iA;
                    idx_couple_fct->par = par;

                    // ii Calculate marital surplus
                    for (int iP=0; iP<par->num_power; iP++){
                        auto idx_couple = index::couple(t,iP,iL,iA,par);
                        Sw[iP] = calc_marital_surplus(sol->Vw_couple_to_couple[idx_couple],sol->Vw_couple_to_single[idx_single],par);
                        Sm[iP] = calc_marital_surplus(sol->Vm_couple_to_couple[idx_couple],sol->Vm_couple_to_single[idx_single],par);
                    }

                    // iii. setup relevant lists 
                    int i = 0;
                    list_start_as_couple[i] = sol->Vw_start_as_couple; i++;
                    list_start_as_couple[i] = sol->Vm_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->Cw_priv_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->Cm_priv_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->hw_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->hm_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->C_inter_start_as_couple; i++;
                    // list_start_as_couple[i] = sol->Q_start_as_couple; i++; // OBS: Maybe Q should be calculated from hw, hw and C_inter
                    i = 0;
                    list_couple_to_couple[i] = sol->Vw_couple_to_couple; i++;
                    list_couple_to_couple[i] = sol->Vm_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->Cwd_priv_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->Cmd_priv_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->hwd_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->hmd_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->Cd_inter_couple_to_couple; i++;
                    // list_couple_to_couple[i] = sol->Qd_couple_to_couple; i++; // OBS: Maybe Q should be calculated from hw, hw and C_inter
                    i = 0;
                    list_couple_to_single[i] = sol->Vw_couple_to_single[idx_single]; i++;
                    list_couple_to_single[i] = sol->Vm_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->Cw_priv_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->Cm_priv_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->hw_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->hm_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->Cw_inter_couple_to_single[idx_single]; i++;
                    // list_couple_to_single[i] = sol->Qw_couple_to_single[idx_single]; i++;

                    // iv. Update solution
                    // Update solutions in list_start_as_couple
                    bargaining::check_participation_constraints(sol->power_idx, sol->power, Sw, Sm, idx_couple_fct, list_start_as_couple, list_couple_to_couple, list_couple_to_single, num, par);
                    
                    // update C_tot_couple (Note: C__start_as_couple is nan when they divorce)
                    // for (int iP=0; iP<par->num_power; iP++){
                    //     auto idx = index::couple(t,iP,iL,iA,par);
                    //     if (sol->power[idx] >= 0.0){
                    //         sol->C_tot_start_as_couple[idx] = sol->Cw_priv_start_as_couple[idx] + sol->Cm_priv_start_as_couple[idx] + sol->C_inter_start_as_couple[idx];
                    //     }
                    // }

                } // wealth
            } // love
            
            // delete pointers
            // for (int i=0; i<num;i++){
            //     delete[] list_start_as_couple[i];
            //     delete[] list_couple_to_couple[i];
            // }

            delete[] list_start_as_couple;
            delete[] list_couple_to_couple;
            delete[] list_couple_to_single;
            delete[] Sw;
            delete[] Sm;
            list_start_as_couple = nullptr;
            list_couple_to_couple = nullptr;
            list_couple_to_single = nullptr;
            Sw = nullptr;
            Sm = nullptr;

            delete idx_couple_fct;
        // } // pragma
    }

    void solve_couple(int t, sol_struct *sol, par_struct *par){

        // #pragma omp parallel num_threads(par->threads) 
        // {

            // 1. Solve choice specific values
            for (int ilw=0; ilw<par->num_l; ilw++){
                for (int ilm=0; ilm<par->num_l; ilm++){
                    // solve choice specific
                    solve_choice_specific_couple(t,ilw,ilm,sol,par);
                } // ilm
            } // ilw

            // 2. Solve unconditional values
            find_unconditional_couple_solution(t,sol,par);

            // 3. Solve starting as couple
            solve_start_as_couple(t,sol,par);

            // 4. Find expected value
            // #pragma omp for
            for (int iP=0; iP<par->num_power; iP++){
                for (int iL=0; iL<par->num_love; iL++){
                    for (int iA=0; iA<par->num_A;iA++){
                        // Update expected value
                        calc_expected_value_couple(t, iP, iL, iA, sol->Vw_start_as_couple, sol->Vm_start_as_couple, sol->EVw_start_as_couple, sol->EVm_start_as_couple, sol, par);

                        // vi. Update marginal value
                        if (par->do_egm){
                            auto idx_interp = index::couple(t,iP,iL,0,par);
                            double power = par->grid_power[iP];
                            // calc_marginal_value_couple(power, &sol->Vw_start_as_couple[idx_interp], &sol->Vm_start_as_couple[idx_interp], &sol->margV_start_as_couple[idx_interp], sol, par);
                            calc_marginal_value_couple(power, &sol->EVw_start_as_couple[idx_interp], &sol->EVm_start_as_couple[idx_interp], &sol->EmargV_start_as_couple[idx_interp], sol, par);
                        }
                    } // wealth
                } // iP
            } // iL
        // } // pragma
    }


    void solve_single_to_couple(int t, sol_struct *sol, par_struct *par){
        #pragma omp parallel num_threads(par->threads)
        {
            #pragma omp for

            for (int iP=0; iP<par->num_power; iP++){
                for (int iL=0; iL< par->num_love; iL++){
                    for (int iA=0; iA<par->num_A;iA++){
                        auto idx = index::couple(t,iP,iL,iA,par);

                        sol->Vw_single_to_couple[idx] = sol->Vw_couple_to_couple[idx];
                        sol->Vm_single_to_couple[idx] = sol->Vm_couple_to_couple[idx];
                        sol->lw_single_to_couple[idx] = sol->lw_couple_to_couple[idx];
                        sol->lm_single_to_couple[idx] = sol->lm_couple_to_couple[idx];

                        //--- Find labor indices for woman and man ---
                        int ilw = tools::binary_search(0, par->num_l, par->grid_l, sol->lw_single_to_couple[idx]);
                        int ilm = tools::binary_search(0, par->num_l, par->grid_l, sol->lm_single_to_couple[idx]);
                        auto idx_d = index::couple_d(t, ilw, ilm, iP, iL, iA, par);

                        sol->Cw_priv_single_to_couple[idx] = sol->Cwd_priv_couple_to_couple[idx_d];
                        sol->Cm_priv_single_to_couple[idx] = sol->Cmd_priv_couple_to_couple[idx_d];
                        sol->hw_single_to_couple[idx] = sol->hwd_couple_to_couple[idx_d];
                        sol->hm_single_to_couple[idx] = sol->hmd_couple_to_couple[idx_d];
                        sol->C_inter_single_to_couple[idx] = sol->Cd_inter_couple_to_couple[idx_d];  
                        sol->Q_single_to_couple[idx] = sol->Qd_couple_to_couple[idx_d];  

                        sol->Cw_tot_single_to_couple[idx] = sol->Cw_priv_single_to_couple[idx] + sol->C_inter_single_to_couple[idx];
                        sol->Cm_tot_single_to_couple[idx] = sol->Cm_priv_single_to_couple[idx] + sol->C_inter_single_to_couple[idx];
                    } // wealth
                } // love
            } // power
        } // pragma
    }


        
}
