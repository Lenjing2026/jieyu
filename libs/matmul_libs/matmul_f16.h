#ifndef MATMUL_F16_H
#define MATMUL_F16_H
#include<cstdint>
#include<immintrin.h>
#include"../date_libs/wmat.h"
#include"../date_libs/parallel.h"

void matmul_f16_classic(const float* a,const uint16_t* b,float* c,int m,int n,int k){
    par_for(0,m,m>=4,[&](int64_t i0,int64_t i1){
        for(int64_t i=i0;i<i1;i++)
            for(int j=0;j<n;j++){
                float sum=0.0f;
                for(int t=0;t<k;t++) sum+=a[(size_t)i*k+t]*wmat::to_f32(b[(size_t)t*n+j]);
                c[(size_t)i*n+j]=sum;
            }
    });
}

__attribute__((target("avx2,fma,f16c")))
void matmul_f16_avx_fma(const float* a,const uint16_t* b,float* c,int m,int n,int k){
    par_for(0,m,m>=4,[&](int64_t i0,int64_t i1) __attribute__((target("avx2,fma,f16c"))){
    for(int64_t i=i0;i<i1;i++){
        float* crow=c+(size_t)i*n;
        const float* arow=a+(size_t)i*k;
        for(int p=0;p<k;p++){
            const float aval=arow[p];
            __m256 va=_mm256_set1_ps(aval);
            const uint16_t* brow=b+(size_t)p*n;
            int j=0;
            for(;j<=n-8;j+=8){
                const __m128i hb=_mm_loadu_si128((const __m128i*)(brow+j));
                const __m256 vb=_mm256_cvtph_ps(hb);
                __m256 vc=_mm256_loadu_ps(crow+j);
                vc=_mm256_fmadd_ps(va,vb,vc);
                _mm256_storeu_ps(crow+j,vc);
            }
            for(;j<n;j++) crow[j]+=aval*wmat::to_f32(brow[j]);
        }
    }
    });
}

#endif
