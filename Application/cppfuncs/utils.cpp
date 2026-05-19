// functions related to utility and environment.
#ifndef MAIN
#define UTILS
#include "myheader.cpp"
#endif

namespace utils {
    double util(double C_priv, double lh, double Q, int gender, par_struct *par, double love){
        double rho = par->rho_w;
        double phi = par->phi_w;
        double eta = par->eta_w;
        double lambda = par->lambda_w;
        if (gender == man) {
            rho = par->rho_m;
            phi = par->phi_m;
            eta = par->eta_m;
            lambda = par->lambda_m;
        }

        // note: the log(Q + 0.01) is to avoid log(0) in case Q=0
        double leisure = (1.0 - lh) * par->available_hours;
        return pow(C_priv, 1.0-rho)/(1.0-rho) + phi*pow(leisure, 1.0-eta)/(1.0 - eta) + lambda*log(Q + 0.5) + love;
    }

    double CES_household(double C, double h_agg, par_struct *par){
        double agg = (1.0 - par->pi)*pow(C , par->omega) + par->pi*pow(h_agg, par->omega);
        return pow(agg, 1.0/par->omega);
    
    }
    double Q_single(double C_inter, double h,int gender, par_struct *par){
        double weight = par->alpha;
        if (gender == man) {
            weight = 2.0 - par->alpha;
        }

        double h_agg = weight * h * par->available_hours;
        
        return CES_household(C_inter, h_agg, par);
    }

    double Q_couple(double C_inter, double hw, double hm, par_struct *par){
        double weight_w = par->alpha;
        double weight_m = 2.0 - par->alpha;
        double h_agg = pow(weight_w * pow(hw*par->available_hours, par->zeta) + weight_m * pow(hm*par->available_hours, par->zeta), 1.0/par->zeta);
        
        return CES_household(C_inter, h_agg, par);
    }

    double wage(int type, double K, int gender, par_struct* par) {
        double log_wage = par->grid_mu_w[type] + par->gamma_w*K;
        if (gender == man) {
            log_wage = par->grid_mu_m[type] + par->gamma_m*K;
        }
        double full_time = par->grid_l[par->num_l-1];
        return exp(log_wage); 
    }

    double human_capital_transition(double K, double labor, par_struct* par) {
        double K_next = (1.0 - par->delta) * K + par->phi_k * labor;
        return tools::max(0.0, tools::min(K_next, par->max_K));
    }

}