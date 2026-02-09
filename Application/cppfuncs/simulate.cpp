
// functions for solving model for singles.
#ifndef MAIN
#define SIMULATE
#include "myheader.cpp"
#endif

namespace sim {

    double update_power(int t, double power_lag, double love, double Kw_lag, double Km_lag, double A_lag,double Aw_lag,double Am_lag,sim_struct* sim, sol_struct* sol, par_struct* par){
        
        // a. value of remaining a couple at current power
        double power = 1000.0; // nonsense value
        auto idx_sol = index::couple(t,0,0,0,0,0,par); 
        double Vw_couple_to_couple=0.0;
        double Vm_couple_to_couple=0.0;
        tools::interp_5d_2out(
            par->grid_power,par->grid_love, par->grid_Kw, par->grid_Km, par->grid_A, 
            par->num_power,par->num_love, par->num_K, par->num_K, par->num_A, 
            &sol->Vw_couple_to_couple[idx_sol],&sol->Vm_couple_to_couple[idx_sol], 
            power_lag, love, Kw_lag, Km_lag, A_lag, 
            &Vw_couple_to_couple, &Vm_couple_to_couple);

        // b. value of transitioning into singlehood
        auto idx_single = index::single(t,0,0,par);
        double Vw_couple_to_single = tools::interp_2d(par->grid_Kw,par->grid_Aw,par->num_K,par->num_A,&sol->Vw_couple_to_single[idx_single],Kw_lag,Aw_lag);
        double Vm_couple_to_single = tools::interp_2d(par->grid_Km,par->grid_Am,par->num_K,par->num_A,&sol->Vm_couple_to_single[idx_single],Km_lag,Am_lag);
        
        // c. check participation constraints
        if ((Vw_couple_to_couple>=Vw_couple_to_single) & (Vm_couple_to_couple>=Vm_couple_to_single)){
            power = power_lag;

        } else if ((Vw_couple_to_couple<Vw_couple_to_single) & (Vm_couple_to_couple<Vm_couple_to_single)){
            power = -1.0;

        } else {
          
            // i. determine which partner is unsatisfied
            double* V_power_vec = new double[par->num_power];
            double* V_couple_to_couple = nullptr;
            double* V_couple_to_couple_partner = nullptr;
            double V_couple_to_single = 0.0;
            double V_couple_to_single_partner = 0.0;
            double* grid_power = nullptr;
            bool flip = false;
            if ((Vw_couple_to_couple<Vw_couple_to_single)){ // woman wants to leave
                V_couple_to_single = Vw_couple_to_single;
                V_couple_to_single_partner = Vm_couple_to_single;

                V_couple_to_couple = sol->Vw_couple_to_couple;
                V_couple_to_couple_partner = sol->Vm_couple_to_couple;  

                flip = false;
                grid_power = par->grid_power;                

            } else { // man wants to leave
                V_couple_to_single = Vm_couple_to_single;
                V_couple_to_single_partner = Vw_couple_to_single;

                V_couple_to_couple = sol->Vm_couple_to_couple;
                V_couple_to_couple_partner = sol->Vw_couple_to_couple;

                flip = true;
                grid_power = par->grid_power_flip;
            }

            // ii. find indifference point of unsatisfied partner:
            int j_love = tools::binary_search(0,par->num_love,par->grid_love,love); 
            int j_Kw = tools::binary_search(0,par->num_K,par->grid_Kw,Kw_lag); 
            int j_Km = tools::binary_search(0,par->num_K,par->grid_Km,Km_lag); 
            int j_A = tools::binary_search(0,par->num_A,par->grid_A,A_lag); 
            for (int iP=0; iP<par->num_power; iP++){ 
                auto idx = 0;
                if(flip){
                    idx = index::couple(t,par->num_power-1 - iP,0,0,0,0,par); // flipped for men
                } else {
                    idx = index::couple(t,iP,0,0,0,0,par); 
                }
                V_power_vec[iP] = tools::_interp_4d_index(par->grid_love,par->grid_Kw,par->grid_Km,par->grid_A,par->num_love,par->num_K,par->num_K,par->num_A,&V_couple_to_couple[idx],love,Kw_lag,Km_lag,A_lag,j_love,j_Kw,j_Km,j_A);
            }
            
            // iii. interpolate the power based on the value of single to find indifference-point. (flip the axis)
            power = tools::interp_1d(V_power_vec, par->num_power, grid_power, V_couple_to_single);
            delete[] V_power_vec;
            V_power_vec = nullptr;

            if((power<0.0)|(power>1.0)){ // divorce
                power = -1.0;
            }
            else{
                // iv. find marital surplus of partner at this new power allocation
                int j_power = tools::binary_search(0,par->num_power,par->grid_power,power);
                double V_power_partner = tools::_interp_5d_index(par->grid_power,par->grid_love,par->grid_Kw,par->grid_Km,par->grid_A, par->num_power,par->num_love,par->num_K,par->num_K,par->num_A, &V_couple_to_couple_partner[idx_sol], power,love,Kw_lag,Km_lag,A_lag,j_power,j_love,j_Kw,j_Km,j_A);
                double S_partner = couple::calc_marital_surplus(V_power_partner,V_couple_to_single_partner);
                
                // v. check if partner is happy. If not divorce
                if(S_partner<0.0){
                    power = -1.0; 
                }
            } 
        }

        return power;
    } // update_power


    double draw_partner_assets_cont(double A, int gender, int i, int t, sim_struct *sim, par_struct *par){
        // unpack
        double* cdf_partner_A = par->cdf_partner_Aw;
        double* grid_A = par->grid_Aw;
        double* uniform_partner_A = sim->draw_uniform_partner_Aw;
        if (gender == man){
            cdf_partner_A = par->cdf_partner_Am;
            grid_A = par->grid_Am;
            uniform_partner_A = sim->draw_uniform_partner_Am;
        }

        double* cdf_Ap_cond = new double[par->num_A];
        int index_iA = tools::binary_search(0,par->num_A,grid_A,A);

        // a. find cdf of partner assets
        for (int iAp=0; iAp<par->num_A; iAp++){
            cdf_Ap_cond[iAp] = tools::interp_1d_index_delta(grid_A,par->num_A,cdf_partner_A,A, index_iA,par->num_A, iAp,1,0);
        }

        // b. find inverted cdf of random uniform draw
        int index_sim = index::index2(i,t,par->simN,par->simT);
        double random = uniform_partner_A[index_sim];
        double A_sim = tools::interp_1d(cdf_Ap_cond,par->num_A,grid_A,random);

        delete[] cdf_Ap_cond;
        cdf_Ap_cond = nullptr;

        if (A_sim<0.0){ // WATCH OUT FOR EXTRAPOLATION OR FLAT CDF'S!!!
            A_sim = 0.0;
        }

        return A_sim;
    }

    double draw_partner_assets(double A, int gender, int i, int t, sim_struct *sim, par_struct *par){
        // unpack
        double* cdf_partner_A = par->cdf_partner_Aw;
        double* grid_A = par->grid_Aw;
        double* uniform_partner_A = sim->draw_uniform_partner_Aw;
        if (gender == man){
            cdf_partner_A = par->cdf_partner_Am;
            grid_A = par->grid_Am;
            uniform_partner_A = sim->draw_uniform_partner_Am;
        }

        // a. random uniform number
        int index_sim = index::index2(i,t,par->simN,par->simT);
        double random = uniform_partner_A[index_sim];

        // b. find first index in asset cdf above uniform draw.
        int index_iA = tools::binary_search(0,par->num_A,grid_A,A);
        for (int iAp=0; iAp<par->num_A; iAp++){
            double cdf_Ap_cond = tools::interp_1d_index_delta(grid_A,par->num_A,cdf_partner_A,A, index_iA,par->num_A, iAp,1,0);
            if(cdf_Ap_cond >= random){
                return grid_A[iAp];
            }
        }

        // c. return asset value
        return grid_A[par->num_A-1];

    }

    double draw_partner_human_capital(double K, int gender, int i, int t, sim_struct *sim, par_struct *par){
        // unpack
        double* cdf_partner_K = par->cdf_partner_Kw;
        double* uniform_partner_K = sim->draw_uniform_partner_Kw;
        double* grid_K = par->grid_Kw;
        double* grid_Kp = par->grid_Km;
        if (gender == man){
            cdf_partner_K = par->cdf_partner_Km;
            uniform_partner_K = sim->draw_uniform_partner_Km;
            grid_K = par->grid_Km;
            grid_Kp = par->grid_Kw;
        }

        // a. random uniform number
        int index_sim = index::index2(i,t,par->simN,par->simT);
        double random = uniform_partner_K[index_sim];

        // b. find first index in human capital cdf above uniform draw.
        int index_iK = tools::binary_search(0,par->num_K,grid_K,K);
        for (int iKp=0; iKp<par->num_K; iKp++){
            double cdf_Kp_cond = tools::interp_1d_index_delta(grid_K,par->num_K,cdf_partner_K,K, index_iK,par->num_K, iKp,1,0); // OBS: Make sure this is correct
            if(cdf_Kp_cond >= random){
                return grid_Kp[iKp];
            }
        }

        // c. return human capital value
        return grid_Kp[par->num_K-1]; // OBS: Is this right? Returning highest value if not found?

    }


    void model(sim_struct *sim, sol_struct *sol, par_struct *par){
    
        // pre-compute intra-temporal optimalallocation
        #pragma omp parallel num_threads(par->threads)
        {
            #pragma omp for
            for (int i=0; i<par->simN; i++){
                for (int t=0; t < par->simT; t++){
                    int it = index::index2(i,t,par->simN,par->simT);

                    // state variables
                    double A_lag = 0.0;
                    double Aw_lag = 0.0;
                    double Am_lag = 0.0;
                    double Kw_lag = 0.0;
                    double Km_lag = 0.0;
                    bool   couple_lag = false;
                    double power_lag = 0.0;
                    double love = 0.0;
                    if (t==0){
                        Kw_lag = sim->init_Kw[i];
                        Km_lag = sim->init_Km[i];
                        A_lag = sim->init_A[i];
                        Aw_lag = sim->init_Aw[i];
                        Am_lag = sim->init_Am[i];
                        couple_lag = sim->init_couple[i];
                        power_lag = par->grid_power[sim->init_power_idx[i]];
                        love = sim->init_love[i];
                        sim->love[it] = love;
                    } else {
                        int it_1 = index::index2(i,t-1,par->simN,par->simT);
                        Kw_lag = sim->Kw[it_1];
                        Km_lag = sim->Km[it_1];
                        A_lag = sim->A[it_1];
                        Aw_lag = sim->Aw[it_1];
                        Am_lag = sim->Am[it_1];
                        couple_lag = sim->couple[it_1];
                        power_lag = sim->power[it_1];
                        love = sim->love[it];
                    } 
                    
                    // a) Find transitions in couple/single status and calculate power 
                    double power = 1000.0; // nonsense value
                    if (couple_lag) { // if start as couple

                        power = update_power(t,power_lag,love,Kw_lag,Km_lag,A_lag,Aw_lag,Am_lag,sim,sol,par);
        
                        if (power < 0.0) { // divorce is coded as -1
                            sim->couple[it] = false;
                        } else {
                            sim->couple[it] = true;
                        }

                    } else { // if start as single - follow woman only
                        bool meet = (sim->draw_meet[it] < par->prob_repartner[t]);
                        if (meet){ // if meet a potential partner
                            double Kp = draw_partner_human_capital(Kw_lag, woman, i,t, sim, par);
                            double Ap = draw_partner_assets(Aw_lag, woman, i,t, sim, par);
                            love = sim->draw_repartner_love[it]; // note: love draws on grid.

                            power = single::calc_initial_bargaining_weight(t, love, Kw_lag, Kp, Aw_lag, Ap, sol, par);

                            if ((0.0 <= power) & (power <= 1.0)) { // if meet and agree to couple
                                sim->couple[it] = true;

                                // set beginning-of-period couple states
                                A_lag = Aw_lag + Ap;
                                Kw_lag = Kw_lag;
                                Km_lag = Kp;
                                sim->love[it] = love;
                            } else { // if meet but do not agree to couple
                                power = -1.0;
                                sim->couple[it] = false;
                            }
                            
                        } else { // if do not meet
                            power = -1.0;
                            sim->couple[it] = false;
                        }
                    }

                    // b) Find choices and update states
                    if (sim->couple[it]){

                        // Find labor choice
                        int ilw = -1;
                        int ilm = -1;
                        couple::find_interpolated_labor_index_couple(t, power, love, Kw_lag, Km_lag, A_lag, &ilw, &ilm, sol, par);
                        double labor_w = par->grid_l[ilw];
                        double labor_m = par->grid_l[ilm];
                        sim->lw[it] = labor_w;
                        sim->lm[it] = labor_m;


                        // total consumption
                        auto idx_sol = index::couple_d(t,ilw,ilm,0,0,0,0,0, par);
                        double C_tot = tools::_interp_5d(
                            par->grid_power, par->grid_love, par->grid_Kw, par->grid_Km, par->grid_A,
                            par->num_power,par->num_love,par->num_K, par->num_K, par->num_A,
                            &sol->Cd_tot_couple_to_couple[idx_sol],
                            power,love,Kw_lag,Km_lag,A_lag);

                        double M_resources = couple::resources_couple(labor_w,labor_m,Kw_lag,Km_lag,A_lag,par); // enforce ressource constraint (may be slightly broken due to approximation error)
                        if (C_tot > M_resources){ 
                            C_tot = M_resources;
                        }
                        sim->C_tot[it] = C_tot;

                        // consumpton allocation
                        double C_inter = 0.0; // placeholder for public consumption
                        double Q = 0.0; // placeholder for public goods
                        precompute::intraperiod_allocation_couple(&sim->Cw_priv[it], &sim->Cm_priv[it], &sim->hw[it], &sim->hm[it], &C_inter, &Q, ilw, ilm, -1000, power, C_tot, par, sol, 
                            true, // interpolate 
                            false // do not use power index
                        );
                        sim->Cw_inter[it] = C_inter;
                        sim->Cm_inter[it] = C_inter;
                        sim->Qw[it] = Q;
                        sim->Qm[it] = Q;

                        // update end-of-period states
                        sim->Kw[it] = utils::human_capital_transition(Kw_lag, labor_w, par) * sim->draw_shock_Kw[it];
                        sim->Km[it] = utils::human_capital_transition(Km_lag, labor_m, par) * sim->draw_shock_Km[it];
                        sim->A[it] = M_resources - sim->Cw_priv[it] - sim->Cm_priv[it] - C_inter;
                        sim->Aw[it] = par->div_A_share * sim->A[it];
                        sim->Am[it] = (1.0-par->div_A_share) * sim->A[it];
                        sim->power[it] = power;
                        if(t<par->simT-1){
                            int it1 = index::index2(i,t+1,par->simN,par->simT);
                            sim->love[it1] = love + par->sigma_love*sim->draw_love[it1];
                        }


                    } else { // single
                        // find labor choice
                        int ilw = single::find_interpolated_labor_index_single(t, Kw_lag, Aw_lag, woman, sol, par);
                        int ilm = single::find_interpolated_labor_index_single(t, Km_lag, Am_lag, man, sol, par);
                        double labor_w = par->grid_l[ilw]; 
                        double labor_m = par->grid_l[ilm];
                        sim->lw[it] = labor_w;
                        sim->lm[it] = labor_m; 

                        auto idx_sol_single_w = index::single_d(t,ilw,0,0,par);
                        auto idx_sol_single_m = index::single_d(t,ilm,0,0,par);
                        double *sol_single_w = &sol->Cwd_tot_single_to_single[idx_sol_single_w];
                        double *sol_single_m = &sol->Cmd_tot_single_to_single[idx_sol_single_m];

                        // total consumption
                        double Cw_tot = tools::interp_2d(par->grid_Kw,par->grid_Aw,par->num_K,par->num_A,sol_single_w,Kw_lag,Aw_lag);
                        double Cm_tot = tools::interp_2d(par->grid_Km,par->grid_Am,par->num_K,par->num_A,sol_single_m,Km_lag,Am_lag);
                        double Mw = single::resources_single(labor_w, Kw_lag, Aw_lag, woman, par); // enforce ressource constraint (may be slightly broken due to approximation error)
                        double Mm = single::resources_single(labor_m, Km_lag, Am_lag, man, par);
                        if (Cw_tot > Mw){
                            Cw_tot = Mw;
                        }
                        if (Cm_tot > Mm){
                            Cm_tot = Mm;
                        }
                        sim->Cw_tot[it] = Cw_tot;
                        sim->Cm_tot[it] = Cm_tot;

                        // consumption allocation
                        precompute::intraperiod_allocation_single(&sim->Cw_priv[it],&sim->hw[it], &sim->Cw_inter[it], &sim->Qw[it], Cw_tot, ilw, woman,par, sol);
                        precompute::intraperiod_allocation_single(&sim->Cm_priv[it],&sim->hm[it], &sim->Cm_inter[it], &sim->Qm[it], Cm_tot, ilm, man,par, sol);

                        // update end-of-period states  
                        sim->Kw[it] = utils::human_capital_transition(Kw_lag, labor_w, par) * sim->draw_shock_Kw[it];
                        sim->Km[it] = utils::human_capital_transition(Km_lag, labor_m, par) * sim->draw_shock_Km[it];
                        sim->Aw[it] = Mw - sim->Cw_priv[it] - sim->Cw_inter[it];
                        sim->Am[it] = Mm - sim->Cm_priv[it] - sim->Cm_inter[it];
                        sim->power[it] = -1.0;

                    }

                    // c) variables for moment simulation
                    // i) wages
                    sim->wage_w[it] = utils::wage(Kw_lag, woman, par);
                    sim->wage_m[it] = utils::wage(Km_lag, man, par);

                    // ii) leisure
                    sim->leisure_w[it] = (1.0 - sim->hw[it] - sim->lw[it]);
                    sim->leisure_m[it] = (1.0 - sim->hm[it] - sim->lm[it]);

                    // iii) utility of women
                    double love_now = 0.0;
                    if (sim->couple[it]){
                        love_now = sim->love[it];
                    }
                    double lh_w = (sim->lw[it] + sim->hw[it]);
                    sim->util[it] = pow(par->beta , t) * utils::util(sim->Cw_priv[it], lh_w, sim->Qw[it],woman,par,love_now);

                } // t
            } // i

        } // pragma

    } // simulate
}
