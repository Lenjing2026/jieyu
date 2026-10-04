#ifndef MATMUL_AVX_H
#define MATMUL_AVX_H

#include <immintrin.h>
__attribute__((target("avx")))
void matmul_avx(const float *a, const float *b, float *c, int m, int n, int k){
    for(int i=0;i<m;i++){
        float* crow=c+i*n;
        const float* arow=a+i*k;
        for(int p=0;p<k;p++){
            float aval=arow[p];
            __m256 va=_mm256_set1_ps(aval);
            const float* brow=b+p*n;
            int j=0;
            for(;j<=n-8;j+=8){
                __m256 vb=_mm256_loadu_ps(brow+j);
                __m256 vc=_mm256_loadu_ps(crow+j);
                vc=_mm256_add_ps(vc,_mm256_mul_ps(va,vb));
                _mm256_storeu_ps(crow+j,vc);
            }
            for(;j<n;j++){
                crow[j]+=aval*brow[j];
            }
        }
    }
}
#endif  // MATMUL_AVX_H