#ifndef silu_h
#define silu_h
#include<cmath>
void silu(float* x,float*y,int n){
    #pragma omp parallel for if(n>=8192)
    for(int i=0;i<n;i++)
        y[i]=x[i]/(1.0f+expf(-x[i]));
}
#endif