#ifndef SILU_FMA_H
#define SILU_FMA_H
#include<cmath>
#include<immintrin.h>
__attribute__((target("avx,fma")))
void silu_fma(float* x,float* y,int n){
    const __m256 one=_mm256_set1_ps(1.0f);
    const __m256 lg2e=_mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2h=_mm256_set1_ps(0.693359375f);
    const __m256 ln2l=_mm256_set1_ps(-2.12194440e-4f);
    const __m256 elo=_mm256_set1_ps(-87.0f);
    const __m256 ehi=_mm256_set1_ps(88.0f);
    const __m256 q1=_mm256_set1_ps(1.0f/720.0f);
    const __m256 q2=_mm256_set1_ps(1.0f/120.0f);
    const __m256 q3=_mm256_set1_ps(1.0f/24.0f);
    const __m256 q4=_mm256_set1_ps(1.0f/6.0f);
    const __m256 q5=_mm256_set1_ps(0.5f);
    const int nv=n/8*8;
    #pragma omp parallel for if(n>=8192)
    for(int i=0;i<nv;i+=8){
        const __m256 vx=_mm256_loadu_ps(x+i);
        const __m256 xa=_mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(),vx),elo),ehi);
        const __m256 t=_mm256_mul_ps(xa,lg2e);
        const __m256 nf=_mm256_round_ps(t,0x08);
        const __m256 r=_mm256_sub_ps(_mm256_sub_ps(xa,_mm256_mul_ps(nf,ln2h)),_mm256_mul_ps(nf,ln2l));
        __m256 p=_mm256_fmadd_ps(q1,r,q2);
        p=_mm256_fmadd_ps(p,r,q3);
        p=_mm256_fmadd_ps(p,r,q4);
        p=_mm256_fmadd_ps(p,r,q5);
        p=_mm256_fmadd_ps(p,r,one);
        p=_mm256_fmadd_ps(p,r,one);
        const __m128i nel=_mm_cvtps_epi32(_mm256_castps256_ps128(nf));
        const __m128i neh=_mm_cvtps_epi32(_mm256_extractf128_ps(nf,1));
        const __m128i pl=_mm_slli_epi32(_mm_add_epi32(nel,_mm_set1_epi32(127)),23);
        const __m128i ph=_mm_slli_epi32(_mm_add_epi32(neh,_mm_set1_epi32(127)),23);
        const __m256 pw=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(pl)),_mm_castsi128_ps(ph),1);
        const __m256 e=_mm256_mul_ps(p,pw);
        _mm256_storeu_ps(y+i,_mm256_div_ps(vx,_mm256_add_ps(one,e)));
    }
    for(int i=nv;i<n;i++)
        y[i]=x[i]/(1.0f+expf(-x[i]));
}
#endif
