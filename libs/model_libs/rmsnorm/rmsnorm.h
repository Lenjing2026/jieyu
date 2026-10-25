#ifndef rmsnorm_h
#define rmsnorm_h
#include<cmath>
#include<immintrin.h>
#include "../../date_libs/date.h"
#include "../../date_libs/check_cpu.h"
void rmsnorm(float* x,float* y,const float* weight,int n,float eps){
    if(cpu_can.fma3)
        #include"rmsnorm_fma.txt"
    else if(cpu_can.avx)
        #include"rmsnorm_avx.txt"
    else{
        float ss=0.0f;
        for(int i=0;i<n;i++) ss+=x[i]*x[i];
        const float den=ss/n+eps;
        if(den<=0.0f){
            for(int i=0;i<n;i++) y[i]=0.0f;
        }else{
            const float scale=1.0f/sqrtf(den);
            for(int i=0;i<n;i++) y[i]=x[i]*scale*weight[i];
        }
    }
}
#endif
