typedef struct sol_struct
{
 double* Vwd_single_to_single;
 double* Vmd_single_to_single;
 double* Cwd_tot_single_to_single;
 double* Cmd_tot_single_to_single;
 double* Cwd_priv_single_to_single;
 double* Cmd_priv_single_to_single;
 double* Cwd_inter_single_to_single;
 double* Cmd_inter_single_to_single;
 double* Qwd_single_to_single;
 double* Qmd_single_to_single;
 double* lwd_single_to_single;
 double* lmd_single_to_single;
 double* hwd_single_to_single;
 double* hmd_single_to_single;
 double* EmargUwd_single_to_single_pd;
 double* Cwd_tot_single_to_single_pd;
 double* Mwd_single_to_single_pd;
 double* Vwd_single_to_single_pd;
 double* EmargUmd_single_to_single_pd;
 double* Cmd_totm_single_to_single_pd;
 double* Mmd_single_to_single_pd;
 double* Vmd_single_to_single_pd;
 double* Vw_couple_to_single;
 double* Vm_couple_to_single;
 double* EVw_start_as_single;
 double* EVm_start_as_single;
 double* EmargVw_start_as_single;
 double* EmargVm_start_as_single;
 double* EVw_cond_meet_partner;
 double* EVm_cond_meet_partner;
 double* EVw_uncond_meet_partner;
 double* EVm_uncond_meet_partner;
 double* Vw_single_to_single;
 double* Vm_single_to_single;
 double* Cw_tot_single_to_single;
 double* Cm_tot_single_to_single;
 double* lw_single_to_single;
 double* lm_single_to_single;
 double* Vd_couple_to_couple;
 double* Vwd_couple_to_couple;
 double* Vmd_couple_to_couple;
 double* Cwd_priv_couple_to_couple;
 double* Cmd_priv_couple_to_couple;
 double* Cd_inter_couple_to_couple;
 double* Qd_couple_to_couple;
 double* lwd_couple_to_couple;
 double* lmd_couple_to_couple;
 double* hwd_couple_to_couple;
 double* hmd_couple_to_couple;
 double* Cd_tot_couple_to_couple;
 int* power_idx;
 double* power;
 double* EmargUd_pd;
 double* Cd_tot_pd;
 double* Md_pd;
 double* Vd_couple_to_couple_pd;
 double* V_single_to_couple;
 double* Vw_single_to_couple;
 double* Vm_single_to_couple;
 double* Vw_start_as_couple;
 double* Vm_start_as_couple;
 double* margV_start_as_couple;
 double* EVw_start_as_couple;
 double* EVm_start_as_couple;
 double* EmargV_start_as_couple;
 double* V_couple_to_couple;
 double* Vw_couple_to_couple;
 double* Vm_couple_to_couple;
 double* C_tot_couple_to_couple;
 double* lw_couple_to_couple;
 double* lm_couple_to_couple;
 double* pre_Cwd_priv_single;
 double* pre_Cmd_priv_single;
 double* pre_Cwd_inter_single;
 double* pre_Cmd_inter_single;
 double* pre_Qwd_single;
 double* pre_Qmd_single;
 double* pre_hwd_single;
 double* pre_hmd_single;
 double* grid_marg_u_single_w;
 double* grid_marg_u_single_m;
 double* grid_marg_u_single_w_for_inv;
 double* grid_marg_u_single_m_for_inv;
 double* pre_Cwd_priv_couple;
 double* pre_Cmd_priv_couple;
 double* pre_Cd_inter_couple;
 double* pre_Qd_couple;
 double* pre_hwd_couple;
 double* pre_hmd_couple;
 double* grid_marg_u_couple;
 double* grid_marg_u_couple_for_inv;
 double* grid_Cinterp_couple;
 double* solution_time;
} sol_struct;

double* get_double_p_sol_struct(sol_struct* x, char* name){

 if( strcmp(name,"Vwd_single_to_single") == 0 ){ return x->Vwd_single_to_single; }
 else if( strcmp(name,"Vmd_single_to_single") == 0 ){ return x->Vmd_single_to_single; }
 else if( strcmp(name,"Cwd_tot_single_to_single") == 0 ){ return x->Cwd_tot_single_to_single; }
 else if( strcmp(name,"Cmd_tot_single_to_single") == 0 ){ return x->Cmd_tot_single_to_single; }
 else if( strcmp(name,"Cwd_priv_single_to_single") == 0 ){ return x->Cwd_priv_single_to_single; }
 else if( strcmp(name,"Cmd_priv_single_to_single") == 0 ){ return x->Cmd_priv_single_to_single; }
 else if( strcmp(name,"Cwd_inter_single_to_single") == 0 ){ return x->Cwd_inter_single_to_single; }
 else if( strcmp(name,"Cmd_inter_single_to_single") == 0 ){ return x->Cmd_inter_single_to_single; }
 else if( strcmp(name,"Qwd_single_to_single") == 0 ){ return x->Qwd_single_to_single; }
 else if( strcmp(name,"Qmd_single_to_single") == 0 ){ return x->Qmd_single_to_single; }
 else if( strcmp(name,"lwd_single_to_single") == 0 ){ return x->lwd_single_to_single; }
 else if( strcmp(name,"lmd_single_to_single") == 0 ){ return x->lmd_single_to_single; }
 else if( strcmp(name,"hwd_single_to_single") == 0 ){ return x->hwd_single_to_single; }
 else if( strcmp(name,"hmd_single_to_single") == 0 ){ return x->hmd_single_to_single; }
 else if( strcmp(name,"EmargUwd_single_to_single_pd") == 0 ){ return x->EmargUwd_single_to_single_pd; }
 else if( strcmp(name,"Cwd_tot_single_to_single_pd") == 0 ){ return x->Cwd_tot_single_to_single_pd; }
 else if( strcmp(name,"Mwd_single_to_single_pd") == 0 ){ return x->Mwd_single_to_single_pd; }
 else if( strcmp(name,"Vwd_single_to_single_pd") == 0 ){ return x->Vwd_single_to_single_pd; }
 else if( strcmp(name,"EmargUmd_single_to_single_pd") == 0 ){ return x->EmargUmd_single_to_single_pd; }
 else if( strcmp(name,"Cmd_totm_single_to_single_pd") == 0 ){ return x->Cmd_totm_single_to_single_pd; }
 else if( strcmp(name,"Mmd_single_to_single_pd") == 0 ){ return x->Mmd_single_to_single_pd; }
 else if( strcmp(name,"Vmd_single_to_single_pd") == 0 ){ return x->Vmd_single_to_single_pd; }
 else if( strcmp(name,"Vw_couple_to_single") == 0 ){ return x->Vw_couple_to_single; }
 else if( strcmp(name,"Vm_couple_to_single") == 0 ){ return x->Vm_couple_to_single; }
 else if( strcmp(name,"EVw_start_as_single") == 0 ){ return x->EVw_start_as_single; }
 else if( strcmp(name,"EVm_start_as_single") == 0 ){ return x->EVm_start_as_single; }
 else if( strcmp(name,"EmargVw_start_as_single") == 0 ){ return x->EmargVw_start_as_single; }
 else if( strcmp(name,"EmargVm_start_as_single") == 0 ){ return x->EmargVm_start_as_single; }
 else if( strcmp(name,"EVw_cond_meet_partner") == 0 ){ return x->EVw_cond_meet_partner; }
 else if( strcmp(name,"EVm_cond_meet_partner") == 0 ){ return x->EVm_cond_meet_partner; }
 else if( strcmp(name,"EVw_uncond_meet_partner") == 0 ){ return x->EVw_uncond_meet_partner; }
 else if( strcmp(name,"EVm_uncond_meet_partner") == 0 ){ return x->EVm_uncond_meet_partner; }
 else if( strcmp(name,"Vw_single_to_single") == 0 ){ return x->Vw_single_to_single; }
 else if( strcmp(name,"Vm_single_to_single") == 0 ){ return x->Vm_single_to_single; }
 else if( strcmp(name,"Cw_tot_single_to_single") == 0 ){ return x->Cw_tot_single_to_single; }
 else if( strcmp(name,"Cm_tot_single_to_single") == 0 ){ return x->Cm_tot_single_to_single; }
 else if( strcmp(name,"lw_single_to_single") == 0 ){ return x->lw_single_to_single; }
 else if( strcmp(name,"lm_single_to_single") == 0 ){ return x->lm_single_to_single; }
 else if( strcmp(name,"Vd_couple_to_couple") == 0 ){ return x->Vd_couple_to_couple; }
 else if( strcmp(name,"Vwd_couple_to_couple") == 0 ){ return x->Vwd_couple_to_couple; }
 else if( strcmp(name,"Vmd_couple_to_couple") == 0 ){ return x->Vmd_couple_to_couple; }
 else if( strcmp(name,"Cwd_priv_couple_to_couple") == 0 ){ return x->Cwd_priv_couple_to_couple; }
 else if( strcmp(name,"Cmd_priv_couple_to_couple") == 0 ){ return x->Cmd_priv_couple_to_couple; }
 else if( strcmp(name,"Cd_inter_couple_to_couple") == 0 ){ return x->Cd_inter_couple_to_couple; }
 else if( strcmp(name,"Qd_couple_to_couple") == 0 ){ return x->Qd_couple_to_couple; }
 else if( strcmp(name,"lwd_couple_to_couple") == 0 ){ return x->lwd_couple_to_couple; }
 else if( strcmp(name,"lmd_couple_to_couple") == 0 ){ return x->lmd_couple_to_couple; }
 else if( strcmp(name,"hwd_couple_to_couple") == 0 ){ return x->hwd_couple_to_couple; }
 else if( strcmp(name,"hmd_couple_to_couple") == 0 ){ return x->hmd_couple_to_couple; }
 else if( strcmp(name,"Cd_tot_couple_to_couple") == 0 ){ return x->Cd_tot_couple_to_couple; }
 else if( strcmp(name,"power") == 0 ){ return x->power; }
 else if( strcmp(name,"EmargUd_pd") == 0 ){ return x->EmargUd_pd; }
 else if( strcmp(name,"Cd_tot_pd") == 0 ){ return x->Cd_tot_pd; }
 else if( strcmp(name,"Md_pd") == 0 ){ return x->Md_pd; }
 else if( strcmp(name,"Vd_couple_to_couple_pd") == 0 ){ return x->Vd_couple_to_couple_pd; }
 else if( strcmp(name,"V_single_to_couple") == 0 ){ return x->V_single_to_couple; }
 else if( strcmp(name,"Vw_single_to_couple") == 0 ){ return x->Vw_single_to_couple; }
 else if( strcmp(name,"Vm_single_to_couple") == 0 ){ return x->Vm_single_to_couple; }
 else if( strcmp(name,"Vw_start_as_couple") == 0 ){ return x->Vw_start_as_couple; }
 else if( strcmp(name,"Vm_start_as_couple") == 0 ){ return x->Vm_start_as_couple; }
 else if( strcmp(name,"margV_start_as_couple") == 0 ){ return x->margV_start_as_couple; }
 else if( strcmp(name,"EVw_start_as_couple") == 0 ){ return x->EVw_start_as_couple; }
 else if( strcmp(name,"EVm_start_as_couple") == 0 ){ return x->EVm_start_as_couple; }
 else if( strcmp(name,"EmargV_start_as_couple") == 0 ){ return x->EmargV_start_as_couple; }
 else if( strcmp(name,"V_couple_to_couple") == 0 ){ return x->V_couple_to_couple; }
 else if( strcmp(name,"Vw_couple_to_couple") == 0 ){ return x->Vw_couple_to_couple; }
 else if( strcmp(name,"Vm_couple_to_couple") == 0 ){ return x->Vm_couple_to_couple; }
 else if( strcmp(name,"C_tot_couple_to_couple") == 0 ){ return x->C_tot_couple_to_couple; }
 else if( strcmp(name,"lw_couple_to_couple") == 0 ){ return x->lw_couple_to_couple; }
 else if( strcmp(name,"lm_couple_to_couple") == 0 ){ return x->lm_couple_to_couple; }
 else if( strcmp(name,"pre_Cwd_priv_single") == 0 ){ return x->pre_Cwd_priv_single; }
 else if( strcmp(name,"pre_Cmd_priv_single") == 0 ){ return x->pre_Cmd_priv_single; }
 else if( strcmp(name,"pre_Cwd_inter_single") == 0 ){ return x->pre_Cwd_inter_single; }
 else if( strcmp(name,"pre_Cmd_inter_single") == 0 ){ return x->pre_Cmd_inter_single; }
 else if( strcmp(name,"pre_Qwd_single") == 0 ){ return x->pre_Qwd_single; }
 else if( strcmp(name,"pre_Qmd_single") == 0 ){ return x->pre_Qmd_single; }
 else if( strcmp(name,"pre_hwd_single") == 0 ){ return x->pre_hwd_single; }
 else if( strcmp(name,"pre_hmd_single") == 0 ){ return x->pre_hmd_single; }
 else if( strcmp(name,"grid_marg_u_single_w") == 0 ){ return x->grid_marg_u_single_w; }
 else if( strcmp(name,"grid_marg_u_single_m") == 0 ){ return x->grid_marg_u_single_m; }
 else if( strcmp(name,"grid_marg_u_single_w_for_inv") == 0 ){ return x->grid_marg_u_single_w_for_inv; }
 else if( strcmp(name,"grid_marg_u_single_m_for_inv") == 0 ){ return x->grid_marg_u_single_m_for_inv; }
 else if( strcmp(name,"pre_Cwd_priv_couple") == 0 ){ return x->pre_Cwd_priv_couple; }
 else if( strcmp(name,"pre_Cmd_priv_couple") == 0 ){ return x->pre_Cmd_priv_couple; }
 else if( strcmp(name,"pre_Cd_inter_couple") == 0 ){ return x->pre_Cd_inter_couple; }
 else if( strcmp(name,"pre_Qd_couple") == 0 ){ return x->pre_Qd_couple; }
 else if( strcmp(name,"pre_hwd_couple") == 0 ){ return x->pre_hwd_couple; }
 else if( strcmp(name,"pre_hmd_couple") == 0 ){ return x->pre_hmd_couple; }
 else if( strcmp(name,"grid_marg_u_couple") == 0 ){ return x->grid_marg_u_couple; }
 else if( strcmp(name,"grid_marg_u_couple_for_inv") == 0 ){ return x->grid_marg_u_couple_for_inv; }
 else if( strcmp(name,"grid_Cinterp_couple") == 0 ){ return x->grid_Cinterp_couple; }
 else if( strcmp(name,"solution_time") == 0 ){ return x->solution_time; }
 else {return NULL;}

}


int* get_int_p_sol_struct(sol_struct* x, char* name){

 if( strcmp(name,"power_idx") == 0 ){ return x->power_idx; }
 else {return NULL;}

}


