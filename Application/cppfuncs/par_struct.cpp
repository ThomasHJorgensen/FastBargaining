typedef struct par_struct
{
 double R;
 double beta;
 double div_A_share;
 double div_cost;
 double available_hours;
 double rho;
 double rho_mult;
 double phi;
 double phi_mult;
 double eta;
 double eta_mult;
 double lambda_;
 double lambda_mult;
 double alpha;
 double zeta;
 double omega;
 double pi;
 double mu;
 double sigma_mu;
 double gamma;
 double gamma2;
 double mu_mult;
 double sigma_mu_mult;
 double gamma_mult;
 double gamma2_mult;
 double phi_k;
 double delta;
 double part_time;
 double full_time_hours;
 int T;
 int num_A;
 double max_A;
 int num_K;
 double max_K;
 double sigma_K;
 double sigma_K_mult;
 int num_shock_K;
 int num_power;
 int num_love;
 double max_love;
 double sigma_love;
 double mean_love;
 int num_shock_love;
 int num_types;
 double type_corr;
 int num_A_pd;
 double max_A_pd;
 int num_marg_u;
 double p_meet;
 double* prob_partner_Kw;
 double* prob_partner_Km;
 double* prob_partner_A_w;
 double* prob_partner_A_m;
 double* prob_partner_type_w;
 double* prob_partner_type_m;
 bool interp_inverse;
 bool precompute_intratemporal;
 int num_Ctot;
 double max_Ctot;
 bool do_egm;
 int seed;
 int simT;
 int simN;
 double init_couple_share;
 bool init_nash_bargaining;
 int threads;
 bool do_multistart;
 char* interp_method;
 bool centered_gradient;
 char* bargaining;
 double mu_shock_low;
 double mu_shock_high;
 double mu_w;
 double mu_m;
 double gamma_w;
 double gamma_m;
 double gamma2_w;
 double gamma2_m;
 double rho_w;
 double rho_m;
 double phi_w;
 double phi_m;
 double eta_w;
 double eta_m;
 double lambda_w;
 double lambda_m;
 double sigma_Kw;
 double sigma_Km;
 long long* grid_t;
 double* grid_type;
 double* grid_mu_w;
 double* type_w_share;
 double* grid_mu_m;
 double* type_m_share;
 double* grid_l;
 int num_l;
 double* grid_A;
 double* grid_Aw;
 double* grid_Am;
 double* grid_Kw;
 double* grid_Km;
 double* grid_power;
 double* grid_power_flip;
 double* grid_love;
 double* grid_shock_Kw;
 double* grid_weight_Kw;
 double* grid_shock_Km;
 double* grid_weight_Km;
 double* grid_shock_love;
 double* grid_weight_love;
 double* grid_A_pd;
 double* grid_Aw_pd;
 double* grid_Am_pd;
 double* grid_Ctot;
 double* grid_C_for_marg_u;
 double* grid_inv_marg_u;
 double* grid_Wpre;
 double* prob_repartner;
 double* cdf_partner_Kw;
 double* cdf_partner_Km;
 double* cdf_partner_Aw;
 double* cdf_partner_Am;
 double* cdf_partner_type_w;
 double* cdf_partner_type_m;
 double* prob_partner_love;
 long long* idx_single_type;
 long long* idx_single_K;
 long long* idx_couple_type_w;
 long long* idx_couple_type_m;
 long long* idx_couple_power;
 long long* idx_couple_love;
 long long* idx_couple_Kw;
 long long* idx_couple_Km;
 long long* idx_couple_barg_type_w;
 long long* idx_couple_barg_type_m;
 long long* idx_couple_barg_love;
 long long* idx_couple_barg_Kw;
 long long* idx_couple_barg_Km;
 long long* idx_pre_couple_lw;
 long long* idx_pre_couple_lm;
 long long* idx_pre_couple_power;
} par_struct;

double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"beta") == 0 ){ return x->beta; }
 else if( strcmp(name,"div_A_share") == 0 ){ return x->div_A_share; }
 else if( strcmp(name,"div_cost") == 0 ){ return x->div_cost; }
 else if( strcmp(name,"available_hours") == 0 ){ return x->available_hours; }
 else if( strcmp(name,"rho") == 0 ){ return x->rho; }
 else if( strcmp(name,"rho_mult") == 0 ){ return x->rho_mult; }
 else if( strcmp(name,"phi") == 0 ){ return x->phi; }
 else if( strcmp(name,"phi_mult") == 0 ){ return x->phi_mult; }
 else if( strcmp(name,"eta") == 0 ){ return x->eta; }
 else if( strcmp(name,"eta_mult") == 0 ){ return x->eta_mult; }
 else if( strcmp(name,"lambda_") == 0 ){ return x->lambda_; }
 else if( strcmp(name,"lambda_mult") == 0 ){ return x->lambda_mult; }
 else if( strcmp(name,"alpha") == 0 ){ return x->alpha; }
 else if( strcmp(name,"zeta") == 0 ){ return x->zeta; }
 else if( strcmp(name,"omega") == 0 ){ return x->omega; }
 else if( strcmp(name,"pi") == 0 ){ return x->pi; }
 else if( strcmp(name,"mu") == 0 ){ return x->mu; }
 else if( strcmp(name,"sigma_mu") == 0 ){ return x->sigma_mu; }
 else if( strcmp(name,"gamma") == 0 ){ return x->gamma; }
 else if( strcmp(name,"gamma2") == 0 ){ return x->gamma2; }
 else if( strcmp(name,"mu_mult") == 0 ){ return x->mu_mult; }
 else if( strcmp(name,"sigma_mu_mult") == 0 ){ return x->sigma_mu_mult; }
 else if( strcmp(name,"gamma_mult") == 0 ){ return x->gamma_mult; }
 else if( strcmp(name,"gamma2_mult") == 0 ){ return x->gamma2_mult; }
 else if( strcmp(name,"phi_k") == 0 ){ return x->phi_k; }
 else if( strcmp(name,"delta") == 0 ){ return x->delta; }
 else if( strcmp(name,"part_time") == 0 ){ return x->part_time; }
 else if( strcmp(name,"full_time_hours") == 0 ){ return x->full_time_hours; }
 else if( strcmp(name,"max_A") == 0 ){ return x->max_A; }
 else if( strcmp(name,"max_K") == 0 ){ return x->max_K; }
 else if( strcmp(name,"sigma_K") == 0 ){ return x->sigma_K; }
 else if( strcmp(name,"sigma_K_mult") == 0 ){ return x->sigma_K_mult; }
 else if( strcmp(name,"max_love") == 0 ){ return x->max_love; }
 else if( strcmp(name,"sigma_love") == 0 ){ return x->sigma_love; }
 else if( strcmp(name,"mean_love") == 0 ){ return x->mean_love; }
 else if( strcmp(name,"type_corr") == 0 ){ return x->type_corr; }
 else if( strcmp(name,"max_A_pd") == 0 ){ return x->max_A_pd; }
 else if( strcmp(name,"p_meet") == 0 ){ return x->p_meet; }
 else if( strcmp(name,"max_Ctot") == 0 ){ return x->max_Ctot; }
 else if( strcmp(name,"init_couple_share") == 0 ){ return x->init_couple_share; }
 else if( strcmp(name,"mu_shock_low") == 0 ){ return x->mu_shock_low; }
 else if( strcmp(name,"mu_shock_high") == 0 ){ return x->mu_shock_high; }
 else if( strcmp(name,"mu_w") == 0 ){ return x->mu_w; }
 else if( strcmp(name,"mu_m") == 0 ){ return x->mu_m; }
 else if( strcmp(name,"gamma_w") == 0 ){ return x->gamma_w; }
 else if( strcmp(name,"gamma_m") == 0 ){ return x->gamma_m; }
 else if( strcmp(name,"gamma2_w") == 0 ){ return x->gamma2_w; }
 else if( strcmp(name,"gamma2_m") == 0 ){ return x->gamma2_m; }
 else if( strcmp(name,"rho_w") == 0 ){ return x->rho_w; }
 else if( strcmp(name,"rho_m") == 0 ){ return x->rho_m; }
 else if( strcmp(name,"phi_w") == 0 ){ return x->phi_w; }
 else if( strcmp(name,"phi_m") == 0 ){ return x->phi_m; }
 else if( strcmp(name,"eta_w") == 0 ){ return x->eta_w; }
 else if( strcmp(name,"eta_m") == 0 ){ return x->eta_m; }
 else if( strcmp(name,"lambda_w") == 0 ){ return x->lambda_w; }
 else if( strcmp(name,"lambda_m") == 0 ){ return x->lambda_m; }
 else if( strcmp(name,"sigma_Kw") == 0 ){ return x->sigma_Kw; }
 else if( strcmp(name,"sigma_Km") == 0 ){ return x->sigma_Km; }
 else {return NAN;}

}


int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"num_A") == 0 ){ return x->num_A; }
 else if( strcmp(name,"num_K") == 0 ){ return x->num_K; }
 else if( strcmp(name,"num_shock_K") == 0 ){ return x->num_shock_K; }
 else if( strcmp(name,"num_power") == 0 ){ return x->num_power; }
 else if( strcmp(name,"num_love") == 0 ){ return x->num_love; }
 else if( strcmp(name,"num_shock_love") == 0 ){ return x->num_shock_love; }
 else if( strcmp(name,"num_types") == 0 ){ return x->num_types; }
 else if( strcmp(name,"num_A_pd") == 0 ){ return x->num_A_pd; }
 else if( strcmp(name,"num_marg_u") == 0 ){ return x->num_marg_u; }
 else if( strcmp(name,"num_Ctot") == 0 ){ return x->num_Ctot; }
 else if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"simT") == 0 ){ return x->simT; }
 else if( strcmp(name,"simN") == 0 ){ return x->simN; }
 else if( strcmp(name,"threads") == 0 ){ return x->threads; }
 else if( strcmp(name,"num_l") == 0 ){ return x->num_l; }
 else {return -9999;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"prob_partner_Kw") == 0 ){ return x->prob_partner_Kw; }
 else if( strcmp(name,"prob_partner_Km") == 0 ){ return x->prob_partner_Km; }
 else if( strcmp(name,"prob_partner_A_w") == 0 ){ return x->prob_partner_A_w; }
 else if( strcmp(name,"prob_partner_A_m") == 0 ){ return x->prob_partner_A_m; }
 else if( strcmp(name,"prob_partner_type_w") == 0 ){ return x->prob_partner_type_w; }
 else if( strcmp(name,"prob_partner_type_m") == 0 ){ return x->prob_partner_type_m; }
 else if( strcmp(name,"grid_type") == 0 ){ return x->grid_type; }
 else if( strcmp(name,"grid_mu_w") == 0 ){ return x->grid_mu_w; }
 else if( strcmp(name,"type_w_share") == 0 ){ return x->type_w_share; }
 else if( strcmp(name,"grid_mu_m") == 0 ){ return x->grid_mu_m; }
 else if( strcmp(name,"type_m_share") == 0 ){ return x->type_m_share; }
 else if( strcmp(name,"grid_l") == 0 ){ return x->grid_l; }
 else if( strcmp(name,"grid_A") == 0 ){ return x->grid_A; }
 else if( strcmp(name,"grid_Aw") == 0 ){ return x->grid_Aw; }
 else if( strcmp(name,"grid_Am") == 0 ){ return x->grid_Am; }
 else if( strcmp(name,"grid_Kw") == 0 ){ return x->grid_Kw; }
 else if( strcmp(name,"grid_Km") == 0 ){ return x->grid_Km; }
 else if( strcmp(name,"grid_power") == 0 ){ return x->grid_power; }
 else if( strcmp(name,"grid_power_flip") == 0 ){ return x->grid_power_flip; }
 else if( strcmp(name,"grid_love") == 0 ){ return x->grid_love; }
 else if( strcmp(name,"grid_shock_Kw") == 0 ){ return x->grid_shock_Kw; }
 else if( strcmp(name,"grid_weight_Kw") == 0 ){ return x->grid_weight_Kw; }
 else if( strcmp(name,"grid_shock_Km") == 0 ){ return x->grid_shock_Km; }
 else if( strcmp(name,"grid_weight_Km") == 0 ){ return x->grid_weight_Km; }
 else if( strcmp(name,"grid_shock_love") == 0 ){ return x->grid_shock_love; }
 else if( strcmp(name,"grid_weight_love") == 0 ){ return x->grid_weight_love; }
 else if( strcmp(name,"grid_A_pd") == 0 ){ return x->grid_A_pd; }
 else if( strcmp(name,"grid_Aw_pd") == 0 ){ return x->grid_Aw_pd; }
 else if( strcmp(name,"grid_Am_pd") == 0 ){ return x->grid_Am_pd; }
 else if( strcmp(name,"grid_Ctot") == 0 ){ return x->grid_Ctot; }
 else if( strcmp(name,"grid_C_for_marg_u") == 0 ){ return x->grid_C_for_marg_u; }
 else if( strcmp(name,"grid_inv_marg_u") == 0 ){ return x->grid_inv_marg_u; }
 else if( strcmp(name,"grid_Wpre") == 0 ){ return x->grid_Wpre; }
 else if( strcmp(name,"prob_repartner") == 0 ){ return x->prob_repartner; }
 else if( strcmp(name,"cdf_partner_Kw") == 0 ){ return x->cdf_partner_Kw; }
 else if( strcmp(name,"cdf_partner_Km") == 0 ){ return x->cdf_partner_Km; }
 else if( strcmp(name,"cdf_partner_Aw") == 0 ){ return x->cdf_partner_Aw; }
 else if( strcmp(name,"cdf_partner_Am") == 0 ){ return x->cdf_partner_Am; }
 else if( strcmp(name,"cdf_partner_type_w") == 0 ){ return x->cdf_partner_type_w; }
 else if( strcmp(name,"cdf_partner_type_m") == 0 ){ return x->cdf_partner_type_m; }
 else if( strcmp(name,"prob_partner_love") == 0 ){ return x->prob_partner_love; }
 else {return NULL;}

}


bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"interp_inverse") == 0 ){ return x->interp_inverse; }
 else if( strcmp(name,"precompute_intratemporal") == 0 ){ return x->precompute_intratemporal; }
 else if( strcmp(name,"do_egm") == 0 ){ return x->do_egm; }
 else if( strcmp(name,"init_nash_bargaining") == 0 ){ return x->init_nash_bargaining; }
 else if( strcmp(name,"do_multistart") == 0 ){ return x->do_multistart; }
 else if( strcmp(name,"centered_gradient") == 0 ){ return x->centered_gradient; }
 else {return false;}

}


char* get_char_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"interp_method") == 0 ){ return x->interp_method; }
 else if( strcmp(name,"bargaining") == 0 ){ return x->bargaining; }
 else {return NULL;}

}


long long* get_long_long_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"grid_t") == 0 ){ return x->grid_t; }
 else if( strcmp(name,"idx_single_type") == 0 ){ return x->idx_single_type; }
 else if( strcmp(name,"idx_single_K") == 0 ){ return x->idx_single_K; }
 else if( strcmp(name,"idx_couple_type_w") == 0 ){ return x->idx_couple_type_w; }
 else if( strcmp(name,"idx_couple_type_m") == 0 ){ return x->idx_couple_type_m; }
 else if( strcmp(name,"idx_couple_power") == 0 ){ return x->idx_couple_power; }
 else if( strcmp(name,"idx_couple_love") == 0 ){ return x->idx_couple_love; }
 else if( strcmp(name,"idx_couple_Kw") == 0 ){ return x->idx_couple_Kw; }
 else if( strcmp(name,"idx_couple_Km") == 0 ){ return x->idx_couple_Km; }
 else if( strcmp(name,"idx_couple_barg_type_w") == 0 ){ return x->idx_couple_barg_type_w; }
 else if( strcmp(name,"idx_couple_barg_type_m") == 0 ){ return x->idx_couple_barg_type_m; }
 else if( strcmp(name,"idx_couple_barg_love") == 0 ){ return x->idx_couple_barg_love; }
 else if( strcmp(name,"idx_couple_barg_Kw") == 0 ){ return x->idx_couple_barg_Kw; }
 else if( strcmp(name,"idx_couple_barg_Km") == 0 ){ return x->idx_couple_barg_Km; }
 else if( strcmp(name,"idx_pre_couple_lw") == 0 ){ return x->idx_pre_couple_lw; }
 else if( strcmp(name,"idx_pre_couple_lm") == 0 ){ return x->idx_pre_couple_lm; }
 else if( strcmp(name,"idx_pre_couple_power") == 0 ){ return x->idx_pre_couple_power; }
 else {return NULL;}

}


