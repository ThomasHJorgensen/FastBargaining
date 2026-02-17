#ifndef MAIN
#define PRECOMPUTE
#include "myheader.cpp"
#endif

namespace precompute{

    typedef struct { 
    double lw;
    double lm; 
    double power;
    double C_tot;

    par_struct *par;
    } solver_precompute_couple_struct;

    typedef struct {
        double C_tot;
        double l;
        int gender;

        par_struct *par;
    } solver_precompute_single_struct;

    ////////////////////////////// Single precompute functions //////////////////////////////
    double objfunc_precompute_single(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        solver_precompute_single_struct *solver_data = (solver_precompute_single_struct *) solver_data_in;

        double C_tot = solver_data->C_tot;
        double l = solver_data->l;
        int gender = solver_data->gender;
        par_struct *par = solver_data->par;

        double c_priv = x[0];
        double h = x[1];
        double c = C_tot - c_priv;

        double Q = 0.0;
        // if(gender==woman){
        //     Q = utils::Q(c, h, 0, solver_data->par);
        // } else {
        //     Q = utils::Q(c, 0, h, solver_data->par);
        // }
        Q = utils::Q_single(c, h, gender, solver_data->par);

        // clip and penalty
        double penalty = 0.0;
        if (c_priv < 1.0e-8) {
            penalty += 1000.0*(c_priv*c_priv);
            c_priv = 1.0e-6;
        }
        if (h < 1.0e-8) {
            penalty += 1000.0*(h*h);
            h = 1.0e-6;
        }

        // utility of choice
        double love = 0.0;
        double val = utils::util(c_priv, l+h, Q, gender, par, love);

        // return negative of value
        return - val + penalty;
    }

    void solve_intraperiod_single(double* C_priv, double* h, double* C_inter, double* Q, double C_tot, double labor, double* start_c_priv, double* start_h, int gender, par_struct *par) {
        // setup numerical solver
        solver_precompute_single_struct* solver_data = new solver_precompute_single_struct;

        int const dim = 2;
        double lb[dim], ub[dim], x[dim];

        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);
        double minf = 0.0;

        // conditional on total consumption C_tot and labor supply l
        solver_data->C_tot = C_tot;
        solver_data->l = labor;
        solver_data->par = par;
        solver_data->gender;
        nlopt_set_min_objective(opt, objfunc_precompute_single, solver_data);
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, 1.0e-6);
        nlopt_set_xtol_rel(opt, 1.0e-5);

        // bounds
        lb[0] = 1.0e-6;
        lb[1] = 1.0e-6;
        ub[0] = solver_data->C_tot;
        ub[1] = 1 - labor;

        // may adjust bounds if not working properly
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        // initial guess
        x[0] = *start_c_priv;
        x[1] = *start_h;


        nlopt_optimize(opt, x, &minf);

        // unpack
        *C_priv = x[0];
        *h = x[1];
        *C_inter = C_tot - *C_priv;
        *Q = 0.0;
        // if(gender==woman){
        //     *Q = utils::Q(*C_inter, *h, 0, par);
        // } else {
        //     *Q = utils::Q(*C_inter, 0, *h, par);
        // }
        *Q = utils::Q_single(*C_inter, *h, gender, par);

        nlopt_destroy(opt);
        delete solver_data;
    }

    void intraperiod_allocation_single(double* C_priv, double* h, double* C_inter, double* Q, double C_tot, int il, int gender, par_struct *par, sol_struct *sol, bool interpolate = true){

        double l = par->grid_l[il];
        
        if(interpolate){ // interpolate pre-computed solution
            auto idx = index::index2(il,0,par->num_l,par->num_Ctot);

            double* C_priv_grid = sol->pre_Cmd_priv_single;
            double* h_grid = sol->pre_hmd_single;
            double* C_inter_grid = sol->pre_Cmd_inter_single;
            if(gender == woman) {
                C_priv_grid = sol->pre_Cwd_priv_single;
                h_grid = sol->pre_hwd_single;
                C_inter_grid = sol->pre_Cwd_inter_single;
            }

            int iC = tools::binary_search(0, par->num_Ctot, par->grid_Ctot, C_tot);

            *C_priv = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &C_priv_grid[idx], C_tot, iC);
            *h = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &h_grid[idx], C_tot, iC);
            *C_inter = C_tot - *C_priv;
            // if(gender==man){
            //     *Q = utils::Q(*C_inter, *h, 0, par);
            // } 
            // else {
            //     *Q = utils::Q(*C_inter, 0, *h, par);
            // }
            *Q = utils::Q_single(*C_inter, *h, gender, par);
        } 
        else { // solve intraperiod problem for single numerically
            double start_c_priv = C_tot/2.0;
            double start_h = (1 - l)/2.0;
            solve_intraperiod_single(C_priv, h, C_inter, Q, C_tot, l, &start_c_priv, &start_h, gender, par);
        }
    }

    double util_C_single(double C_tot, int il, int gender, par_struct *par, sol_struct *sol, bool interpolate = true){ //
        // closed form solution for intra-period problem of single
        double l = par->grid_l[il];
        double C_priv = 0.0;
        double h = 0.0;
        double C_inter = 0.0;
        double Q = 0.0;
        intraperiod_allocation_single(&C_priv, &h, &C_inter, &Q, C_tot, il, gender, par, sol, interpolate);

        return utils::util(C_priv, l+h, Q, gender, par, 0.0); // love = 0.0
    }


    void precompute_cons_interp_single(int i_marg_u, int il, int gender, par_struct *par, sol_struct *sol, bool interpolate = true){ 
        // get baseline utility
        double* grid_marg_u_single = par->grid_marg_u_single_w;
        double* grid_marg_u_single_for_inv = par->grid_marg_u_single_w_for_inv;
        if(gender==man){
            grid_marg_u_single = par->grid_marg_u_single_m;
            grid_marg_u_single_for_inv = par->grid_marg_u_single_m_for_inv;
        }

        double delta = 0.0001;
        double util = util_C_single(par->grid_C_for_marg_u[i_marg_u], il, gender, par, sol, interpolate);
        double util_delta = util_C_single(par->grid_C_for_marg_u[i_marg_u] + delta, il, gender, par, sol, interpolate);

        auto idx = index::index2(il, i_marg_u, par->num_l, par->num_marg_u);
        grid_marg_u_single[idx] = (util_delta - util)/delta;

        auto idx_inv = index::index2(il, par->num_marg_u-1 - i_marg_u, par->num_l, par->num_marg_u);
        grid_marg_u_single_for_inv[idx_inv] = grid_marg_u_single[i_marg_u];
    }

    ////////////////////////////// Couple precompute functions //////////////////////////////

    double objfunc_precompute_couple(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        solver_precompute_couple_struct *solver_data = (solver_precompute_couple_struct *) solver_data_in;
        double C_tot = solver_data->C_tot;
        double power = solver_data->power;
        double lw = solver_data->lw;
        double lm = solver_data->lm;
        par_struct *par = solver_data->par;

        // four dimensions! let's go
        double Cw_priv = x[0];
        double Cm_priv = x[1];
        double hw = x[2];
        double hm = x[3];

        // total hours
        double hlw = hw + lw;
        double hlm = hm + lm;

        // home production
        double C_inter = C_tot - Cw_priv - Cm_priv;
        // double Q = utils::Q(C_inter, hw, hm, par);
        double Q = utils::Q_couple(C_inter, hw, hm, par);

        // clip and penalty
        double penalty = 0.0;
        if(Cw_priv < 1.0e-8){
            penalty += 1000.0*(Cw_priv*Cw_priv);
            Cw_priv = 1.0e-6;
        }
        if(Cm_priv < 1.0e-8){
            penalty += 1000.0*(Cm_priv*Cm_priv);
            Cm_priv = 1.0e-6;
        }
        if(hw < 1.0e-8){
            penalty += 1000.0*(hw*hw);
            hw = 1.0e-6;
        }
        if(hm < 1.0e-8){
            penalty += 1000.0*(hm*hm);
            hm = 1.0e-6;
        }

        // time constraint
        if(hlw >= 1){
            penalty += 1000.0*(hlw - 1)*(hlw - 1);
            hw = 1 - lw - 1.0e-6;
        }
        if(hlm >= 1){
            penalty += 1000.0*(hlm - 1)*(hlm - 1);
            hm = 1 - lm - 1.0e-6;
        }

        // consumption constraint
        if(C_inter < 1.0e-8){
            penalty += 1000.0*(C_inter*C_inter);
            C_inter = 1.0e-6;
        }

        // utility of choice
        double uw = utils::util(Cw_priv, hlw, Q, woman, par, 0.0); // love not important for intratemporal allocation
        double um = utils::util(Cm_priv, hlm, Q, man, par, 0.0); 
        double val = power*uw + (1.0-power)*um;

        // return negative of value
        return - val + penalty;

    }

    EXPORT void solve_intraperiod_couple(double* Cw_priv, double* Cm_priv, double* hw, double* hm, double* C_inter, double* Q,
                double C_tot, double lw, double lm, double power, par_struct* par, 
                double start_Cw_priv, double start_Cm_priv, double start_hw, double start_hm,
                double ftol = 1.0e-6, double xtol = 1.0e-5){
        // setup numerical solver
        solver_precompute_couple_struct* solver_data = new solver_precompute_couple_struct;

        int const dim = 4;
        double lb[dim], ub[dim], x[dim];

        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);
        double minf = 0.0;

        // settings
        solver_data->C_tot = C_tot;
        solver_data->lw = lw;
        solver_data->lm = lm;
        solver_data->power = power;
        solver_data->par = par;

        nlopt_set_min_objective(opt, objfunc_precompute_couple, solver_data);
        nlopt_set_maxeval(opt, 2000);
        nlopt_set_ftol_rel(opt, ftol);
        nlopt_set_xtol_rel(opt, xtol);

        // bounds
        lb[0] = 1.0e-6; // Cw_priv
        ub[0] = C_tot;

        lb[1] = 1.0e-6; // Cm_priv
        ub[1] = C_tot; 

        lb[2] = 1.0e-6; // hw
        ub[2] = 1 - lw;

        lb[3] = 1.0e-6; // hm
        ub[3] = 1 - lm;

        // may need to adjust in the cases where l = 1

        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        // initial guess
        x[0] = start_Cw_priv;
        x[1] = start_Cm_priv;
        x[2] = start_hw;
        x[3] = start_hm;

        // run optimizer!
        nlopt_optimize(opt, x, &minf);

        // unpack
        *Cw_priv = x[0];
        *Cm_priv = x[1];
        *hw = x[2];
        *hm = x[3];

        *C_inter = C_tot - *Cw_priv - *Cm_priv;
        // *Q = utils::Q(*C_inter, *hw, *hm, par);
        *Q = utils::Q_couple(*C_inter, *hw, *hm, par);

        // free memory
        nlopt_destroy(opt);
        delete solver_data;
    }

    // Helper for couple allocation: handles both power index and value
    void intraperiod_allocation_couple(
        double* Cw_priv, double* Cm_priv, double* hw, double* hm, double* C_inter, double* Q,
        int ilw, int ilm, int iP, double power, double C_tot,
        par_struct* par, sol_struct* sol, bool interpolate, bool use_power_index)
    {
        if(interpolate){
            if(use_power_index) {
                // Use power index for 1D interpolation
                auto idx = index::index4(ilw, ilm, iP, 0, par->num_l, par->num_l, par->num_power, par->num_Ctot);
                int iC = tools::binary_search(0, par->num_Ctot, par->grid_Ctot, C_tot);

                *Cw_priv = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &sol->pre_Cwd_priv_couple[idx], C_tot, iC);
                *Cm_priv = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &sol->pre_Cmd_priv_couple[idx], C_tot, iC);
                *hw = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &sol->pre_hwd_couple[idx], C_tot, iC);
                *hm = tools::interp_1d_index(par->grid_Ctot, par->num_Ctot, &sol->pre_hmd_couple[idx], C_tot, iC);
            } else {
                // Use power value for 2D interpolation
                auto idx = index::index4(ilw, ilm, 0, 0, par->num_l, par->num_l, par->num_power, par->num_Ctot);
                int iP_ = tools::binary_search(0, par->num_power, par->grid_power, power);
                int iC = tools::binary_search(0, par->num_Ctot, par->grid_Ctot, C_tot);

                *Cw_priv = tools::_interp_2d(par->grid_power, par->grid_Ctot, par->num_power, par->num_Ctot, &sol->pre_Cwd_priv_couple[idx], power, C_tot, iP_, iC);
                *Cm_priv = tools::_interp_2d(par->grid_power, par->grid_Ctot, par->num_power, par->num_Ctot, &sol->pre_Cmd_priv_couple[idx], power, C_tot, iP_, iC);
                *hw = tools::_interp_2d(par->grid_power, par->grid_Ctot, par->num_power, par->num_Ctot, &sol->pre_hwd_couple[idx], power, C_tot, iP_, iC);
                *hm = tools::_interp_2d(par->grid_power, par->grid_Ctot, par->num_power, par->num_Ctot, &sol->pre_hmd_couple[idx], power, C_tot, iP_, iC);
            }
            *C_inter = C_tot - *Cw_priv - *Cm_priv;
            // *Q = utils::Q(*C_inter, *hw, *hm, par);
            *Q = utils::Q_couple(*C_inter, *hw, *hm, par);
        } else {
            double lw = par->grid_l[ilw];
            double lm = par->grid_l[ilm];
            double power_val = use_power_index ? par->grid_power[iP] : power;

            double start_Cw_priv = C_tot/3.0;
            double start_Cm_priv = C_tot/3.0;
            double start_hw = (1 - (lw-1e-6))/2.0;
            double start_hm = (1 - (lm-1e-6))/2.0;

            solve_intraperiod_couple(Cw_priv, Cm_priv, hw, hm, C_inter, Q, C_tot, lw, lm, power_val, par,
                start_Cw_priv, start_Cm_priv, start_hw, start_hm);
        }
    }

    EXPORT double util_C_couple(double C_tot, int ilw, int ilm, int iP, double love, 
        par_struct *par, sol_struct *sol, bool interpolate = true){

        double lw = par->grid_l[ilw];
        double lm = par->grid_l[ilm];
        double power = par->grid_power[iP];
        double Cw_priv = 0.0;
        double Cm_priv = 0.0;
        double hw = 0.0;
        double hm = 0.0;

        double C_inter = 0.0; // C_tot - Cw_priv - Cm_priv;
        double Q = 0.0; // utils::Q(C_inter, hw, hm, par);

        intraperiod_allocation_couple(&Cw_priv, &Cm_priv, &hw, &hm, &C_inter, &Q, ilw, ilm, iP, power, C_tot, par, sol, interpolate, true);


        double uw = utils::util(Cw_priv, lw+hw, Q, woman, par, love);
        double um = utils::util(Cm_priv, lm+hm, Q, man, par, love);

        return power*uw + (1.0-power)*um;
    }

    void precompute_cons_interp_couple(int i_marg_u, int iP, int ilw, int ilm, par_struct *par, sol_struct *sol, bool interpolate = true){

        double delta = 0.0001;
        double util = util_C_couple(par->grid_C_for_marg_u[i_marg_u], ilw, ilm, iP, 0.0, par, sol, interpolate);
        double util_delta = util_C_couple(par->grid_C_for_marg_u[i_marg_u] + delta, ilw, ilm, iP, 0.0, par, sol, interpolate);

        auto idx = index::index4(ilw, ilm, iP, i_marg_u, par->num_l, par->num_l, par->num_power, par->num_marg_u);
        par->grid_marg_u_couple[idx] = (util_delta - util)/delta;

        auto idx_inv = index::index4(ilw, ilm, iP, par->num_marg_u-1 - i_marg_u, par->num_l, par->num_l, par->num_power, par->num_marg_u);
        par->grid_marg_u_couple_for_inv[idx_inv] = par->grid_marg_u_couple[idx];
    }
    
    ////////////////////////////// Precomputation //////////////////////////////
    void precompute(sol_struct* sol, par_struct* par){
        // pre-compute optimal allocation for single
        # pragma omp parallel for num_threads(par->threads)
        for (int il=0; il<par->num_l; il++){
            
            double l = par->grid_l[il];
            
            double start_hw = (1 - (l-1e-6))/2.0;
            double start_hm = (1 - (l-1e-6))/2.0;
            
            for (int iC=par->num_Ctot - 1; iC>=0; iC--){  //solve in descending order to have correct starting values for h in first grid point
                double C_tot = par->grid_Ctot[iC];
                auto idx = index::index2(il,iC,par->num_l,par->num_Ctot);

                double start_C_priv = C_tot/2.0;

                solve_intraperiod_single(&sol->pre_Cmd_priv_single[idx], &sol->pre_hmd_single[idx], &sol->pre_Cmd_inter_single[idx], &sol->pre_Qmd_single[idx], C_tot, l, &start_C_priv, &start_hm, man, par);
                solve_intraperiod_single(&sol->pre_Cwd_priv_single[idx], &sol->pre_hwd_single[idx], &sol->pre_Cwd_inter_single[idx], &sol->pre_Qwd_single[idx], C_tot, l, &start_C_priv, &start_hw, woman, par);

                // start_hw = sol->pre_hmd_single[idx]; //update starting values for h
                // start_hm = sol->pre_hmd_single[idx];

            } //C_tot

            if(strcmp(par->interp_method,"numerical")!=0){
                for (int i_marg_u=0; i_marg_u<par->num_marg_u; i_marg_u++){ 
                    bool interpolate = true;
                    precompute_cons_interp_single(i_marg_u, il, woman, par, sol, interpolate);
                    precompute_cons_interp_single(i_marg_u, il, man, par, sol, interpolate);
                } // marg_u
            } // interp method
        } // labor supply

        // precompute optimal allocation for couples
        const int nL = par->num_l;
        const int nP = par->num_power;

        // total number of iterations
        const long long total = (long long)nL * nL * nP;

        #pragma omp parallel for num_threads(par->threads)
        for (long long idx = 0; idx < total; ++idx) {

            long long tmp = idx;

            const int iP  = tmp % nP;
            tmp /= nP;

            const int ilm = tmp % nL;
            tmp /= nL;

            const int ilw = tmp;
                    
            double lw = par->grid_l[ilw];
            double lm = par->grid_l[ilm];
            double power = par->grid_power[iP];
            
            double start_hw = (1 - (lw-1e-6))/2.0;
            double start_hm = (1 - (lm-1e-6))/2.0;
            
            for(int iC=par->num_Ctot - 1; iC>=0; iC--){ //solve in descending order to have correct starting values for h in first grid point
                
                double C_tot = par->grid_Ctot[iC];
                double start_Cw_priv = C_tot/3.0;
                double start_Cm_priv = C_tot/3.0;


                if(C_tot<1.0){ // reuse staring values when Ctot is low
                    start_hw = sol->pre_hwd_couple[index::index4(ilw, ilm, iP, iC+1, par->num_l, par->num_l, par->num_power, par->num_Ctot)];
                    start_hm = sol->pre_hmd_couple[index::index4(ilw, ilm, iP, iC+1, par->num_l, par->num_l, par->num_power, par->num_Ctot)];
                }

                auto idx = index::index4(ilw, ilm, iP, iC, par->num_l, par->num_l, par->num_power, par->num_Ctot);
                solve_intraperiod_couple(&sol->pre_Cwd_priv_couple[idx], &sol->pre_Cmd_priv_couple[idx], &sol->pre_hwd_couple[idx], &sol->pre_hmd_couple[idx], 
                    &sol->pre_Cd_inter_couple[idx], &sol->pre_Qd_couple[idx],
                    C_tot, lw, lm, power, par,
                    start_Cw_priv, start_Cm_priv, start_hw, start_hm,
                    1.0e-8, 1.0e-7);
            } // iC

            if(strcmp(par->interp_method,"numerical")!=0){
                for(int i_marg_u=0; i_marg_u<par->num_marg_u; i_marg_u++){
                    bool interpolate = true; // this is sometimes a bit unstable when resolve numerically
                    precompute_cons_interp_couple(i_marg_u, iP, ilw, ilm, par, sol, interpolate);
                } // i_marg_u
            } // interp method
        } // idx
    } // precompute

}

