typedef struct sim_struct
{
 double* lw;
 double* lm;
 double* Cw_priv;
 double* Cm_priv;
 double* hw;
 double* hm;
 double* Cw_inter;
 double* Cm_inter;
 double* Qw;
 double* Qm;
 double* Cw_tot;
 double* Cm_tot;
 double* C_tot;
 double* Kw;
 double* Km;
 double* A;
 double* Aw;
 double* Am;
 double* couple;
 double* power;
 double* love;
 double* wage_inc_w;
 double* wage_inc_m;
 double* after_tax_inc_w;
 double* after_tax_inc_m;
 double* leisure_w;
 double* leisure_m;
 double* util_w;
 double* util_m;
 double* divorces;
 double* mean_lifetime_util;
 double* C_ineq_90_10;
 double* draw_shock_Kw;
 double* draw_shock_Km;
 double* draw_love;
 double* draw_meet;
 double* draw_uniform_partner_Kw;
 double* draw_uniform_partner_Km;
 double* draw_uniform_partner_Aw;
 double* draw_uniform_partner_Am;
 double* draw_uniform_partner_type_w;
 double* draw_uniform_partner_type_m;
 double* draw_repartner_love;
 double* init_love;
 double* init_Kw;
 double* init_Km;
 double* init_A;
 double* init_Aw;
 double* init_Am;
 double* init_divorces;
 int* init_type_w;
 int* init_type_m;
 int* init_power_idx;
 bool* init_couple;
} sim_struct;

double* get_double_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"lw") == 0 ){ return x->lw; }
 else if( strcmp(name,"lm") == 0 ){ return x->lm; }
 else if( strcmp(name,"Cw_priv") == 0 ){ return x->Cw_priv; }
 else if( strcmp(name,"Cm_priv") == 0 ){ return x->Cm_priv; }
 else if( strcmp(name,"hw") == 0 ){ return x->hw; }
 else if( strcmp(name,"hm") == 0 ){ return x->hm; }
 else if( strcmp(name,"Cw_inter") == 0 ){ return x->Cw_inter; }
 else if( strcmp(name,"Cm_inter") == 0 ){ return x->Cm_inter; }
 else if( strcmp(name,"Qw") == 0 ){ return x->Qw; }
 else if( strcmp(name,"Qm") == 0 ){ return x->Qm; }
 else if( strcmp(name,"Cw_tot") == 0 ){ return x->Cw_tot; }
 else if( strcmp(name,"Cm_tot") == 0 ){ return x->Cm_tot; }
 else if( strcmp(name,"C_tot") == 0 ){ return x->C_tot; }
 else if( strcmp(name,"Kw") == 0 ){ return x->Kw; }
 else if( strcmp(name,"Km") == 0 ){ return x->Km; }
 else if( strcmp(name,"A") == 0 ){ return x->A; }
 else if( strcmp(name,"Aw") == 0 ){ return x->Aw; }
 else if( strcmp(name,"Am") == 0 ){ return x->Am; }
 else if( strcmp(name,"couple") == 0 ){ return x->couple; }
 else if( strcmp(name,"power") == 0 ){ return x->power; }
 else if( strcmp(name,"love") == 0 ){ return x->love; }
 else if( strcmp(name,"wage_inc_w") == 0 ){ return x->wage_inc_w; }
 else if( strcmp(name,"wage_inc_m") == 0 ){ return x->wage_inc_m; }
 else if( strcmp(name,"after_tax_inc_w") == 0 ){ return x->after_tax_inc_w; }
 else if( strcmp(name,"after_tax_inc_m") == 0 ){ return x->after_tax_inc_m; }
 else if( strcmp(name,"leisure_w") == 0 ){ return x->leisure_w; }
 else if( strcmp(name,"leisure_m") == 0 ){ return x->leisure_m; }
 else if( strcmp(name,"util_w") == 0 ){ return x->util_w; }
 else if( strcmp(name,"util_m") == 0 ){ return x->util_m; }
 else if( strcmp(name,"divorces") == 0 ){ return x->divorces; }
 else if( strcmp(name,"mean_lifetime_util") == 0 ){ return x->mean_lifetime_util; }
 else if( strcmp(name,"C_ineq_90_10") == 0 ){ return x->C_ineq_90_10; }
 else if( strcmp(name,"draw_shock_Kw") == 0 ){ return x->draw_shock_Kw; }
 else if( strcmp(name,"draw_shock_Km") == 0 ){ return x->draw_shock_Km; }
 else if( strcmp(name,"draw_love") == 0 ){ return x->draw_love; }
 else if( strcmp(name,"draw_meet") == 0 ){ return x->draw_meet; }
 else if( strcmp(name,"draw_uniform_partner_Kw") == 0 ){ return x->draw_uniform_partner_Kw; }
 else if( strcmp(name,"draw_uniform_partner_Km") == 0 ){ return x->draw_uniform_partner_Km; }
 else if( strcmp(name,"draw_uniform_partner_Aw") == 0 ){ return x->draw_uniform_partner_Aw; }
 else if( strcmp(name,"draw_uniform_partner_Am") == 0 ){ return x->draw_uniform_partner_Am; }
 else if( strcmp(name,"draw_uniform_partner_type_w") == 0 ){ return x->draw_uniform_partner_type_w; }
 else if( strcmp(name,"draw_uniform_partner_type_m") == 0 ){ return x->draw_uniform_partner_type_m; }
 else if( strcmp(name,"draw_repartner_love") == 0 ){ return x->draw_repartner_love; }
 else if( strcmp(name,"init_love") == 0 ){ return x->init_love; }
 else if( strcmp(name,"init_Kw") == 0 ){ return x->init_Kw; }
 else if( strcmp(name,"init_Km") == 0 ){ return x->init_Km; }
 else if( strcmp(name,"init_A") == 0 ){ return x->init_A; }
 else if( strcmp(name,"init_Aw") == 0 ){ return x->init_Aw; }
 else if( strcmp(name,"init_Am") == 0 ){ return x->init_Am; }
 else if( strcmp(name,"init_divorces") == 0 ){ return x->init_divorces; }
 else {return NULL;}

}


int* get_int_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"init_type_w") == 0 ){ return x->init_type_w; }
 else if( strcmp(name,"init_type_m") == 0 ){ return x->init_type_m; }
 else if( strcmp(name,"init_power_idx") == 0 ){ return x->init_power_idx; }
 else {return NULL;}

}


bool* get_bool_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"init_couple") == 0 ){ return x->init_couple; }
 else {return NULL;}

}


