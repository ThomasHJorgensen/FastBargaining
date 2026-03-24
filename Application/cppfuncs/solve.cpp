#define MAIN
#include "myheader.h"

// include these again here to ensure that they are automatically compiled by consav
// #ifndef MAIN
// #include "precompute.cpp"
// #endif

/////////////
// 5. MAIN //
/////////////

EXPORT void solve(sol_struct *sol, par_struct *par){
    
    #pragma omp parallel num_threads(par->threads)
    {
        // // pre-compute intra-temporal optimal allocation
        precompute::precompute(sol,par);

        // TJ:
        // couple::precompute_couple(sol,par);

        // loop backwards
        for (int t = par->T-1; t >= 0; t--){
            single::solve_single_to_single(t,sol,par); 
            single::solve_couple_to_single(t,sol,par); 
            couple::solve_couple(t,sol,par);
            couple::solve_single_to_couple(t,sol,par);
            single::expected_value_start_single(t,sol,par);
            couple::expected_value_start_couple(t,sol,par);
        }
    }
}


EXPORT void simulate(sim_struct *sim, sol_struct *sol, par_struct *par){
    #pragma omp parallel num_threads(par->threads)
    {
        sim::model(sim,sol,par);
    }

}


EXPORT void compute_margEV(sol_struct* sol, par_struct* par){
    // for (int t = 0; t < par->T; t++){
    //     single::calc_marginal_value_single(t, woman, sol, par);
    //     single::calc_marginal_value_single(t, man, sol, par);

    //     for (int iP=0; iP<par->num_power; iP++){
    //         for (int iL=0; iL<par->num_love; iL++){
    //             auto idx = index::couple(t,iP,iL,0,par);
    //             double* EVw = &sol->EVw_start_as_couple[idx];
    //             double* EVm = &sol->EVm_start_as_couple[idx];
    //             double* EmargV = &sol->EmargV_start_as_couple[idx];
    //             couple::calc_marginal_value_couple(t, iP, iL, EVw, EVm, EmargV, sol, par);
    //         }
    //     }
    // }
    ;
}


EXPORT double calc_init_mu(int t, double love, double Aw, double Am, sol_struct* sol, par_struct* par){
    // logs::write("barg_log.txt", 0, "calc_init_mu\n");
    // double power =  single::calc_initial_bargaining_weight(t, love, Aw, Am, sol, par);
    // logs::write("barg_log.txt", 1, "poewr: %f\n", power);
    // return power;
    return 0.0;
}


EXPORT void random_C_points(double* labor_w, double* labor_m, double* power_diff, double* consumption, int num_P, int num_love, int num_Kw, int num_Km, int num_A, par_struct* par, sol_struct* sol){

    int t = 0;

    int num_total = par->num_types * par->num_types * par->num_l * par->num_l * num_P * num_love * num_Kw * num_Km * num_A;

    for (int type_w = 0; type_w < par->num_types; type_w++){
        for (int type_m = 0; type_m < par->num_types; type_m++){
            for (int iP = 0; iP < num_P; iP++){
                double power = par->grid_power[0] + (par->grid_power[par->num_power-1] - par->grid_power[0]) * (iP+1) / (num_P+1);
                int iP_left = tools::binary_search(0, par->num_power, par->grid_power, power);
                for (int iL = 0; iL < num_love; iL++){
                    double love = par->grid_love[0] + (par->grid_love[par->num_love-1] - par->grid_love[0]) * (iL+1) / (num_love+1);
                    int iL_left = tools::binary_search(0, par->num_love, par->grid_love, love);
                    for (int iKw = 0; iKw < num_Kw; iKw++){
                        double Kw = par->grid_Kw[0] + (par->grid_Kw[par->num_K-1] - par->grid_Kw[0]) * (iKw+1) / (num_Kw+1);
                        int iKw_left = tools::binary_search(0, par->num_K, par->grid_Kw, Kw);
                        for (int iKm = 0; iKm < num_Km; iKm++){
                            double Km = par->grid_Km[0] + (par->grid_Km[par->num_K-1] - par->grid_Km[0]) * (iKm+1) / (num_Km+1);
                            int iKm_left = tools::binary_search(0, par->num_K, par->grid_Km, Km);
                            for (int iA = 0; iA < num_A; iA++){
                                double A = par->grid_A[0] + (par->grid_A[par->num_A-1] - par->grid_A[0]) * (iA+1) / (num_A+1);
                                int iA_left = tools::binary_search(0, par->num_A, par->grid_A, A);
                                
                                auto idx = index::index7(
                                    type_w, type_m, iP, iL, iKw, iKm, iA, 
                                    par->num_types, par->num_types, num_P, num_love, num_Kw, num_Km, num_A
                                );

                                // labor points
                                int ilw_update = -1;
                                int ilm_update = -1;
                                couple::find_interpolated_labor_index_couple(t, type_w, type_m, power, love, Kw, Km, A, &ilw_update, &ilm_update, sol, par);
                                labor_w[idx] = ilw_update;
                                labor_m[idx] = ilm_update;
                                
                                // power
                                double Aw = par->div_A_share * A;
                                double Am = (1.0 - par->div_A_share) * A;
                                double power_update = sim::update_power(t,type_w, type_m, power, love, Kw, Km, A, Aw, Am, sol, par);
                                // if (power_update < 0.0) {
                                //     divorce[idx] = 1.0;
                                // } else {
                                //     divorce[idx] = 0.0;
                                // }
                                power_diff[idx] = power_update - power;

                                // consumption points
                                for (int ilw = 0; ilw < par->num_l; ilw++){
                                    for (int ilm = 0; ilm < par->num_l; ilm++){
                                        auto idx_interp = index::couple_d(t, type_w, type_m, ilw, ilm, 0, 0, 0, 0, 0, par);
                                        double C = tools::_interp_5d_index(
                                            par->grid_power, par->grid_love, par->grid_Kw, par->grid_Km, par->grid_A,
                                            par->num_power, par->num_love, par->num_K, par->num_K, par->num_A,
                                            &sol->Cd_tot_couple_to_couple[idx_interp],
                                            power, love, Kw, Km, A,
                                            iP_left, iL_left, iKw_left, iKm_left, iA_left
                                        );
                                        auto idx_d = index::index9(
                                            type_w, type_m, ilw, ilm, iP, iL, iKw, iKm, iA,
                                            par->num_types, par->num_types, par->num_l, par->num_l, num_P, num_love, num_Kw, num_Km, num_A
                                        );
                                        consumption[idx_d] = C;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}