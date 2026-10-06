#ifndef RMSNORM_AVX_H
#define RMSNORM_AVX_H
#include<cmath>
#include<immintrin.h>
__attribute__((target("avx")))
void rmsnorm_avx(float* x,float* y,const float* weight,int n,float eps){
    float ss=0.0f;
    __m256 s1=_mm256_setzero_ps();
    __m256 s2=_mm256_setzero_ps();
    int i=0;
    for(;i+16<=n;i+=16){
        __m256 x0=_mm256_loadu_ps(x+i);
        __m256 x1=_mm256_loadu_ps(x+i+8);
        s1=_mm256_add_ps(s1,_mm256_mul_ps(x0,x0));
        s2=_mm256_add_ps(s2,_mm256_mul_ps(x1,x1));
    }
    s1=_mm256_add_ps(s1,s2);
    __m128 lo=_mm256_castps256_ps128(s1);
    __m128 hi=_mm256_extractf128_ps(s1, 1);
    __m128 s=_mm_add_ps(lo, hi);
    s=_mm_hadd_ps(s,s);
    s=_mm_hadd_ps(s,s);
    ss=_mm_cvtss_f32(s);
    for(;i<n;i++) ss+=x[i]*x[i];
    float den=ss/n+eps;
    if(den<=0.0f){
        for(int i=0;i<n;i++) y[i]=0.0f;
        return;
    }
    float scale=1.0f / sqrtf(den);
    __m256 vscale=_mm256_set1_ps(scale);
    i=0;
    for (; i+8<=n;i+=8){
        __m256 xv=_mm256_loadu_ps(x+i);
        __m256 wv=_mm256_loadu_ps(weight+i);
        __m256 yv=_mm256_mul_ps(_mm256_mul_ps(xv,vscale),wv);
        _mm256_storeu_ps(y+i,yv);
    }
    for (;i<n;i++) {
        y[i]=x[i]*scale*weight[i];
    }
}
#endif  // RMSNORM_AVX_H
