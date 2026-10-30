#ifndef silu_h
#define silu_h
#include<cmath>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
void silu(float* x,float* y,int n){
    if(cpu_can.avx2){
        #define SIMD_K __m256
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_ZERO() _mm256_setzero_ps()
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_SUB _mm256_sub_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_DIV _mm256_div_ps
        #define SIMD_MIN _mm256_min_ps
        #define SIMD_MAX _mm256_max_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_ROUND(v) _mm256_round_ps(v,0x08)
        #define SIMD_SCALE2N(v) ({__m256 _n=(v);__m128i _l=_mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_castps256_ps128(_n)),_mm_set1_epi32(127)),23);__m128i _h=_mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_extractf128_ps(_n,1)),_mm_set1_epi32(127)),23);_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(_l)),_mm_castsi128_ps(_h),1);})
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
        #define SIMD_SUB _mm256_sub_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_DIV _mm256_div_ps
        #define SIMD_MIN _mm256_min_ps
        #define SIMD_MAX _mm256_max_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_ROUND(v) _mm256_round_ps(v,0x08)
        #define SIMD_SCALE2N(v) ({__m256 _n=(v);__m128i _l=_mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_castps256_ps128(_n)),_mm_set1_epi32(127)),23);__m128i _h=_mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_extractf128_ps(_n,1)),_mm_set1_epi32(127)),23);_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(_l)),_mm_castsi128_ps(_h),1);})
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
        par_for(0,n,n>=8192,[&](int64_t i0,int64_t i1){
            for(int64_t i=i0;i<i1;i++) y[i]=x[i]/(1.0f+expf(-x[i]));
        });
    }
}
#endif