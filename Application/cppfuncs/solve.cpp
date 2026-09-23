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


EXPORT void accuracy_measures(double* labor_w, double* labor_m, double* power_update, double* power_diff, double* consumption, par_struct* par, sol_struct* sol){

    // evaluate on the accuracy grids (of length num_acc) in par
    int num_P = par->num_acc;
    int num_love = par->num_acc;
    int num_Kw = par->num_acc;
    int num_Km = par->num_acc;
    int num_A = par->num_acc;

    #pragma omp parallel for num_threads(par->threads) schedule(dynamic)
    for (int t = 0; t < par->T; t++){
        for (int type_w = 0; type_w < par->num_types; type_w++){
            for (int type_m = 0; type_m < par->num_types; type_m++){
                for (int iP = 0; iP < num_P; iP++){
                    double power = par->grid_power_acc[iP];
                    int iP_left = tools::binary_search(0, par->num_power, par->grid_power, power);
                    for (int iL = 0; iL < num_love; iL++){
                        double love = par->grid_love_acc[iL];
                        int iL_left = tools::binary_search(0, par->num_love, par->grid_love, love);
                        for (int iKw = 0; iKw < num_Kw; iKw++){
                            double Kw = par->grid_Kw_acc[iKw];
                            int iKw_left = tools::binary_search(0, par->num_K, par->grid_Kw, Kw);
                            for (int iKm = 0; iKm < num_Km; iKm++){
                                double Km = par->grid_Km_acc[iKm];
                                int iKm_left = tools::binary_search(0, par->num_K, par->grid_Km, Km);
                                for (int iA = 0; iA < num_A; iA++){
                                    double A = par->grid_A_acc[iA];
                                    int iA_left = tools::binary_search(0, par->num_A, par->grid_A, A);
                                
                                    auto idx = index::index8(
                                        t, type_w, type_m, iP, iL, iKw, iKm, iA, 
                                        par->T, par->num_types, par->num_types, num_P, num_love, num_Kw, num_Km, num_A
                                    );

                                    // labor points
                                    int ilw_update = -1;
                                    int ilm_update = -1;
                                    sim::find_interpolated_labor_index_couple(t, type_w, type_m, power, love, Kw, Km, A, &ilw_update, &ilm_update, sol, par);
                                    labor_w[idx] = ilw_update;
                                    labor_m[idx] = ilm_update;
                                
                                    // power
                                    double Aw = par->div_A_share * A;
                                    double Am = (1.0 - par->div_A_share) * A;
                                    power_update[idx] = sim::update_power(t,type_w, type_m, power, love, Kw, Km, A, Aw, Am, sol, par);
                                    power_diff[idx] = power_update[idx] - power;

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
                                            auto idx_d = index::index10(
                                                t, type_w, type_m, ilw, ilm, iP, iL, iKw, iKm, iA,
                                                par->T, par->num_types, par->num_types, par->num_l, par->num_l, num_P, num_love, num_Kw, num_Km, num_A
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
    } // t

}