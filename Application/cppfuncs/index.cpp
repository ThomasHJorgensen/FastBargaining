// functions for calculating linear indices.
#ifndef MAIN
#define INDEX
#include "myheader.cpp"
#endif

namespace index {
    long long int index2(long long int i1,long long int i2,long long int N1,long long int N2){
        return i2 + i1*N2;
    }
    long long int index3(long long int i1,long long int i2,long long int i3,long long int N1,long long int N2, long long int N3){
        return i3 + i2*N3 + i1*N2*N3;
    }
    long long int index4(long long int i1,long long int i2,long long int i3,long long int i4,long long int N1,long long int N2, long long int N3, long long int N4){
        return i4 + (i3 + (i2 + i1*N2)*N3)*N4;
    }

    long long int index5(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int N1,long long int N2, long long int N3, long long int N4, long long int N5){
        return i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5;
    }

    long long int index6(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int i6,long long int N1,long long int N2, long long int N3, long long int N4, long long int N5, long long int N6){
        return i6 + (i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5)*N6;
    }

    long long int couple(long long int t, long long int iP,long long int iL,long long int iA,par_struct* par){
        return index4(t,iP,iL,iA , par->T,par->num_power,par->num_love,par->num_A); 
    }
    long long int couple_d(long long int t, long long int ilw, long long int ilm, long long int iP,long long int iL,long long int iA,par_struct* par){
        return index6(t,ilw, ilm, iP,iL,iA , par->T,par->num_l,par->num_l,par->num_power,par->num_love,par->num_A); 
    }
    long long int couple_pd(long long int t, long long int ilw, long long int ilm, long long int iP,long long int iL,long long int iA_pd,par_struct* par){
        return index6(t,ilw,ilm,iP,iL,iA_pd , par->T,par->num_l,par->num_l,par->num_power,par->num_love,par->num_A_pd); 
    }
    // long long int single_to_couple(long long int t,long long int iL,long long int iA,par_struct* par){
    //     return index3(t,iL,iA , par->T,par->num_love,par->num_A); 
    // }
    // Single-state indexing now uses (t, il, iK, iA)
    inline long long int single(long long int t, long long int iK, long long int iA, par_struct* par){
        return index3(t, iK, iA, par->T, par->num_K, par->num_A);
    }
    inline long long int single_d(long long int t, long long int il, long long int iK, long long int iA, par_struct* par){
        return index4(t, il, iK, iA, par->T, par->num_l, par->num_K, par->num_A);
    }

    struct index_couple_struct {
            int t;
            int iL;
            int iA;
            par_struct *par; 
            long long int idx(long long int iP){
                    return index::couple(t,iP,iL,iA,par); 
            }
    };

    typedef struct{
            // state levels
            int t;
            double love;
            double A;
            double power;

            // indices
            int iL;
            int iA;

            // model content
            par_struct *par;
            sol_struct *sol;
    } state_couple_struct;

    typedef struct{
            // state levels
            int t;
            double K;
            double A;

            // indices
            int iK;
            int iA;

            // model content
            par_struct *par;
            sol_struct *sol;
    } state_single_struct;
}