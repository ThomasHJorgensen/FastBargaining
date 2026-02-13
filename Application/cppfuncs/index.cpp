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
    long long int index7(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int i6,long long int i7,long long int N1,long long int N2, long long int N3, long long int N4, long long int N5, long long int N6, long long int N7){
        return i7 + (i6 + (i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5)*N6)*N7;
    }
    long long int index8(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int i6,long long int i7,long long int i8,
        long long int N1,long long int N2, long long int N3, long long int N4, long long int N5, long long int N6, long long int N7, long long int N8){
        return i8 + (i7 + (i6 + (i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5)*N6)*N7)*N8;
    }

    long long int index9(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int i6,long long int i7,long long int i8, long long int i9,
        long long int N1,long long int N2, long long int N3, long long int N4, long long int N5, long long int N6, long long int N7, long long int N8, long long int N9){
        return i9 + (i8 + (i7 + (i6 + (i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5)*N6)*N7)*N8)*N9;
    }

    long long int index10(long long int i1,long long int i2,long long int i3,long long int i4,long long int i5,long long int i6,long long int i7,long long int i8, long long int i9, long long int i10,
        long long int N1,long long int N2, long long int N3, long long int N4, long long int N5, long long int N6, long long int N7, long long int N8, long long int N9, long long int N10){
        return i10 + (i9 + (i8 + (i7 + (i6 + (i5 + (i4 + (i3 + (i2 + i1*N2)*N3)*N4)*N5)*N6)*N7)*N8)*N9)*N10;
    }

    // New canonical couple indexing with two K indices: (t, iP, iL, iKw, iKm, iA)
    long long int couple(long long int t, long long int iP, long long int iL, long long int iSw, long long int iSm, long long int iKw, long long int iKm, long long int iA, par_struct* par){
        return index8(t, iP, iL, iSw, iSm, iKw, iKm, iA, par->T, par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A);
    }
    // Discrete couple index with two K indices: (t, ilw, ilm, iP, iL, iKw, iKm, iA)
    long long int couple_d(long long int t, long long int ilw, long long int ilm, long long int iP, long long int iL, long long int iSw, long long int iSm, long long int iKw, long long int iKm, long long int iA, par_struct* par){
        return index10(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, iA, par->T, par->num_l, par->num_l, par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A);
    }
    // Endogenous asset grid variant with two K indices: (t, ilw, ilm, iP, iL, iKw, iKm, iA_pd)
    long long int couple_pd(long long int t, long long int ilw, long long int ilm, long long int iP, long long int iL, long long int iSw, long long int iSm, long long int iKw, long long int iKm, long long int iA_pd, par_struct* par){
        return index10(t, ilw, ilm, iP, iL, iSw, iSm, iKw, iKm, iA_pd, par->T, par->num_l, par->num_l, par->num_power, par->num_love, par->num_S, par->num_S, par->num_K, par->num_K, par->num_A_pd);
    }

    // Single-state indexing now uses (t, il, iK, iA)
    long long int single(long long int t, long long int iS, long long int iK, long long int iA, par_struct* par){
        return index4(t, iS, iK, iA, par->T, par->num_S, par->num_K, par->num_A);
    }
    long long int single_d(long long int t, long long int il, long long int iS, long long int iK, long long int iA, par_struct* par){
        return index5(t, il, iS, iK, iA, par->T, par->num_l, par->num_S, par->num_K, par->num_A);
    }
    long long int single_pd(long long int t, long long int il, long long int iS, long long int iK, long long int iA_pd, par_struct* par){
        return index5(t, il, iS, iK, iA_pd, par->T, par->num_l, par->num_S, par->num_K, par->num_A_pd);
    }

    struct index_couple_struct {
        int t;
        int iL;
        int iSw;
        int iSm;
        int iKw;
        int iKm;
        int iA;
        par_struct *par; 
        long long int idx(long long int iP){
            return index::couple(t,iP,iL,iSw,iSm,iKw,iKm,iA,par); 
        }
    };

    typedef struct{
            // state levels
            int t;
            double love;
            int iSw;
            int iSm;
            double Kw;
            double Km;
            double A;
            double power;

            // indices
            int iL;
            int iKw;
            int iKm;
            int iA;

            // model content
            par_struct *par;
            sol_struct *sol;
    } state_couple_struct;

    typedef struct{
            // state levels
            int t;
            int iS;
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