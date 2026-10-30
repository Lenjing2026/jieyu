#ifndef Rope_h
#define Rope_h
#include<cstddef>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
void rope_at(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    if(cpu_can.avx2){
        #define SIMD_K __m256
        #define SIMD_V128 __m128
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_LOAD128 _mm_loadu_ps
        #define SIMD_STORE128 _mm_storeu_ps
        #define SIMD_SHUFFLE_PS _mm256_shuffle_ps
        #define SIMD_SHUFFLE128 _mm_shuffle_ps
        #define SIMD_UNPACKLO _mm256_unpacklo_ps
        #define SIMD_UNPACKHI _mm256_unpackhi_ps
        #define SIMD_LO128 _mm256_castps256_ps128
        #define SIMD_HI128(v) _mm256_extractf128_ps(v,1)
        #define SIMD_MK(lo,hi) _mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1)
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
        #define SIMD_FMSUB(a,b,c) _mm256_sub_ps(_mm256_mul_ps(a,b),c)
        #define LONG 8
        #define IN_SET "avx2"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef SIMD_FMSUB
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define SIMD_FMSUB _mm256_fmsub_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
    else if(cpu_can.avx){
        #define SIMD_K __m256
        #define SIMD_V128 __m128
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_LOAD128 _mm_loadu_ps
        #define SIMD_STORE128 _mm_storeu_ps
        #define SIMD_SHUFFLE_PS _mm256_shuffle_ps
        #define SIMD_SHUFFLE128 _mm_shuffle_ps
        #define SIMD_UNPACKLO _mm256_unpacklo_ps
        #define SIMD_UNPACKHI _mm256_unpackhi_ps
        #define SIMD_LO128 _mm256_castps256_ps128
        #define SIMD_HI128(v) _mm256_extractf128_ps(v,1)
        #define SIMD_MK(lo,hi) _mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1)
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_FMA(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
        #define SIMD_FMSUB(a,b,c) _mm256_sub_ps(_mm256_mul_ps(a,b),c)
        #define LONG 8
        #define IN_SET "avx"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef SIMD_FMSUB
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define SIMD_FMSUB _mm256_fmsub_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
    else{
        const int half=head_dim/2;
        par_for(0,seq,seq>=32,[&](int64_t p0,int64_t p1){
        for (int64_t pos=p0;pos<p1;pos++) {
            const float* c=cos_table+(size_t)(start_pos+pos)*half;
            const float* s=sin_table+(size_t)(start_pos+pos)*half;
            for (int h=0;h<num_heads;h++) {
                float* qh=q+((size_t)pos*num_heads+h)*head_dim;
                for (int i=0;i<half;i++){
                    const float x0=qh[2*i];
                    const float x1=qh[2*i+1];
                    qh[2*i]=x0*c[i]-x1*s[i];
                    qh[2*i+1]=x0*s[i]+x1*c[i];
                }
            }
        }
        });
    }
}
void rope(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,0,num_heads,head_dim,cos_table,sin_table);
}
void rope_pos(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
}
#endif