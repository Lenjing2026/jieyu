#ifndef rmsnorm_h
#define rmsnorm_h
#include<cmath>
#include<immintrin.h>
#include "../../date_libs/date.h"
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/info.h"
void rmsnorm(float* x,float* y,const float* weight,int n,float eps){
    if(cpu_can.avx2){
        #define SIMD_K __m256
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_ZERO() _mm256_setzero_ps()
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_HSUM(v) ({__m256 _t=(v);__m128 _a=_mm_add_ps(_mm256_castps256_ps128(_t),_mm256_extractf128_ps(_t,1));__m128 _b=_mm_hadd_ps(_a,_a);_mm_cvtss_f32(_mm_hadd_ps(_b,_b));})
        #define LONG 8
        #define IN_SET "avx2"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
    else if(cpu_can.avx){
        #define SIMD_K __m256
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_ZERO() _mm256_setzero_ps()
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_HSUM(v) ({__m256 _t=(v);__m128 _a=_mm_add_ps(_mm256_castps256_ps128(_t),_mm256_extractf128_ps(_t,1));__m128 _b=_mm_hadd_ps(_a,_a);_mm_cvtss_f32(_mm_hadd_ps(_b,_b));})
        #define LONG 8
        #define IN_SET "avx"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
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
