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

        // note: the log(Q + 0.00001) is to avoid log(0) in case Q=0
        return pow(C_priv, 1-rho)/(1-rho) - phi*pow(lh, 1+1/eta)/(1 + 1/eta) + lambda*log(Q + 0.00001) + love;
    }

    double util_couple(double Cw_priv, double Cm_priv, double lhw, double lhm, double Q, double power, int iL,par_struct* par){
        double love = par->grid_love[iL];

        double Uw = util(Cw_priv,lhw,Q,woman,par,love);
        double Um = util(Cm_priv,lhm,Q,man,par,love);

        return power*Uw + (1.0-power)*Um;
    }

    double Q(double C_inter, double hw, double hm, par_struct *par){

        double h_agg = pow(par->alpha*pow(hw, par->zeta) + pow(hm, par->zeta), 1/par->zeta);
        return pow(C_inter + 1.0e-4, par->omega) * pow(h_agg, 1.0 - par->omega);

        // // alternative formulation - useful for testing, but should remain commented out
        // double sub = 0.5;
        // double inner = (pow(h_agg, sub) + pow(C_inter, sub));
        // return pow(inner, 1.0/sub);
    }

    double Q_single(double C_inter, double h,int gender, par_struct *par){
        double weight = par->alpha;
        if (gender == man) {
            weight = 1.0 - par->alpha;
        }

        double h_agg = weight * h;
        
        return pow(C_inter + 1.0e-4, par->omega) * pow(h_agg, 1.0 - par->omega);
    }

    double wage(double K, int gender, par_struct* par) {
        double log_wage = par->mu_w + par->gamma_w*K;
        if (gender == man) {
            log_wage = par->mu_m + par->gamma_m*K;
        }
        return exp(log_wage);
    }

    double human_capital_transition(double K, double labor, par_struct* par) {
        return ((1-par->delta) * K + par->phi_k * labor);
    }

    // double cons_priv_single(double C_tot,int gender,par_struct *par){
    //     // closed form solution for intra-period problem of single.
    //     double rho = par->rho_w;
    //     double phi = par->phi_w;
    //     double alpha1 = par->alpha1_w;
    //     double alpha2 = par->alpha2_w;
    //     if (gender == man) {
    //         rho = par->rho_m;
    //         phi = par->phi_m;
    //         alpha1 = par->alpha1_m;
    //         alpha2 = par->alpha2_m;
    //     }  
        
    //     return C_tot/(1.0 + pow(alpha2/alpha1,1.0/(1.0-phi) ));
    // }

    // double util_C_single(double C_tot, int gender, par_struct* par){
    //     double love = 0.0;
        
    //     // flow-utility
    //     double C_priv = cons_priv_single(C_tot,gender,par);
    //     double C_pub = C_tot - C_priv;
        
    //     return util(C_priv,C_pub,gender,par,love);
    // }

    // double marg_util_C(double C_tot, int gender, par_struct* par){
    //     double rho = par->rho_w;
    //     double phi = par->phi_w;
    //     double alpha1 = par->alpha1_w;
    //     double alpha2 = par->alpha2_w;
    //     if (gender == man) {
    //         rho = par->rho_m;
    //         phi = par->phi_m;
    //         alpha1 = par->alpha1_m;
    //         alpha2 = par->alpha2_m;
    //     }  
        
    //     double share = 1.0/(1.0 + pow(alpha2/alpha1,1.0/(1.0-phi) ));
    //     double constant = alpha1*pow(share,phi) + alpha2*pow(1.0-share,phi);
    //     return phi * pow(C_tot,(1.0-rho)*phi -1.0 ) * pow(constant,1.0 - rho);
    // }

    // double inv_marg_util_C(double marg_U, int gender, par_struct* par){
    //     double rho = par->rho_w;
    //     double phi = par->phi_w;
    //     double alpha1 = par->alpha1_w;
    //     double alpha2 = par->alpha2_w;
    //     if (gender == man) {
    //         rho = par->rho_m;
    //         phi = par->phi_m;
    //         alpha1 = par->alpha1_m;
    //         alpha2 = par->alpha2_m;
    //     }  
        
    //     double share = 1.0/(1.0 + pow(alpha2/alpha1,1.0/(1.0-phi) ));
    //     double constant = alpha1*pow(share,phi) + alpha2*pow(1.0-share,phi);
    //     return pow(marg_U / (phi * pow(constant,(1.0-rho))), 1 / ((1-rho)*phi - 1.0));
    // }
}