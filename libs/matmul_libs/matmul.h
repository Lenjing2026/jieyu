#ifndef MATMUL_H
#define MATMUL_H
#include <omp.h>
#include "matmul_avx.h"
#include "matmul_fma.h"
#include "matmul_avx_fma.h"
#include "../date_libs/check_cpu.h"
void matmul_classic(const int m,const int n,const int k,const float* A,const float* B,float* C) {
    #pragma omp parallel for if(m>=4)
    for (int i=0;i<m;i++) {
        float* crow=C+i*n;
        for (int j=0;j<n;j++) crow[j]=0.0f;
        for (int t=0;t<k;t++){
            const float aval=A[i*k+t];
            const float* brow=B+t*n;
            for (int j=0;j<n;j++) crow[j]+=aval*brow[j];
        }
    }
}
void matmul(const int m,const int n,const int k,const float* A,const float* B,float* C) {
    memset(C,0,sizeof(float)*m*n);
    if(cpu_can.avx && cpu_can.fma3)
        matmul_avx_fma(A,B,C,m,n,k);
    else if(cpu_can.fma3)
        matmul_fma(A,B,C,m,n,k);
    else if(cpu_can.avx)
        matmul_avx(A,B,C,m,n,k);
    else
        matmul_classic(m,n,k,A,B,C);
}
#endif  // MATMUL_H