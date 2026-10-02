#include "matmul_avx.h"
#include "matmul_fma.h"
#include "matmul_avx_fma.h"
#include "../date_libs/check_cpu.h"
void matmul(const int m,const int n,const int k,const float* A,const float* B,float* C) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++){
            float sum=0.0f;
            for (int t=0;t<k;t++) {
                sum+=A[i*k+t]*B[t*n+j];
            }
            C[i*n+j]=sum;
        }
    }
}
void matmul_choose(const int m,const int n,const int k,const float* A,const float* B,float* C) {
    memset(C,0,sizeof(float)*m*n);
    if(cpu_can.avx && cpu_can.fma3)
        matmul_avx_fma(A,B,C,m,n,k);
    else if(cpu_can.fma3)
        matmul_fma(A,B,C,m,n,k);
    else if(cpu_can.avx)
        matmul_avx(A,B,C,m,n,k);
    else
        matmul(m,n,k,A,B,C);
}
//地狱绘图