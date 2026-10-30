#ifndef MATMUL_H
#define MATMUL_H
#include<cstring>
#include<immintrin.h>
#include"../date_libs/check_cpu.h"
#include"../date_libs/parallel.h"
void matmul(const int m,const int n,const int k,const float* A,const float* B,float* C){
    memset(C,0,sizeof(float)*m*n);
    if(cpu_can.avx2){
        #define SIMD_K __m256
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
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
        #include"undef.h"
    }
    else if(cpu_can.avx){
        #define SIMD_K __m256
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
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
        #include"undef.h"
    }
    else{
        par_for(0,m,m>=4,[&](int64_t i0,int64_t i1){
            for(int64_t i=i0;i<i1;i++){
                float* crow=C+(size_t)i*n;
                const float* arow=A+(size_t)i*k;
                for(int p=0;p<k;p++){
                    const float aval=arow[p];
                    const float* brow=B+(size_t)p*n;
                    for(int j=0;j<n;j++) crow[j]+=aval*brow[j];
                }
            }
        });
    }
}
#endif
