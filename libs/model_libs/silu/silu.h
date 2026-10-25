#ifndef silu_h
#define silu_h
#include<cmath>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
void silu(float* x,float* y,int n){
    if(cpu_can.fma3)
        #include"silu_fma.txt"
    else if(cpu_can.avx)
        #include"silu_avx.txt"
    else{
        par_for(0,n,n>=8192,[&](int64_t i0,int64_t i1){
            for(int64_t i=i0;i<i1;i++) y[i]=x[i]/(1.0f+expf(-x[i]));
        });
    }
}
#endif