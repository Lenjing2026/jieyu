#ifndef MATMUL_H
#define MATMUL_H
#include<cstring>
#include<immintrin.h>
#include"../date_libs/check_cpu.h"
#include"../date_libs/parallel.h"
void matmul(const int m,const int n,const int k,const float* A,const float* B,float* C){
    memset(C,0,sizeof(float)*m*n);
    if(cpu_can.fma3)
        #include"matmul_fma.txt"
    else if(cpu_can.avx)
        #include"matmul_avx.txt"
    else{
        par_for(0,m,m>=4,[&](int64_t i0,int64_t i1){
            for(int64_t i=i0;i<i1;i++){
                float* crow=C+(size_t)i*n;
                for(int j=0;j<n;j++) crow[j]=0.0f;
                for(int t=0;t<k;t++){
                    const float aval=A[(size_t)i*k+t];
                    const float* brow=B+(size_t)t*n;
                    for(int j=0;j<n;j++) crow[j]+=aval*brow[j];
                }
            }
        });
    }
}
#endif