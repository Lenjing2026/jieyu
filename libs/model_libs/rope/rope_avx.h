#ifndef ROPE_AVX_H
#define ROPE_AVX_H
#include<immintrin.h>
__attribute__((target("avx")))
void rope_avx_at(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    int half=head_dim/2;
    #pragma omp parallel for if(seq>=32)
    for (int pos=0;pos<seq;pos++) {
        const float* c=cos_table+(size_t)(start_pos+pos)*half;
        const float* s=sin_table+(size_t)(start_pos+pos)*half;
        for (int h=0;h<num_heads;h++) {
            float* qh=q+((size_t)pos*num_heads+h)*head_dim;
            int i=0;
            for (;i+8<=half;i+=8){
                const __m256 lo=_mm256_loadu_ps(qh+2*i);
                const __m256 hi=_mm256_loadu_ps(qh+2*i+8);
                const __m256 x0=_mm256_shuffle_ps(lo,hi,0x88);
                const __m256 x1=_mm256_shuffle_ps(lo,hi,0xDD);
                const __m128 cl=_mm_loadu_ps(c+i);
                const __m128 ch=_mm_loadu_ps(c+i+4);
                const __m128 sl=_mm_loadu_ps(s+i);
                const __m128 sh=_mm_loadu_ps(s+i+4);
                const __m256 cv=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(cl,ch,0x44)),_mm_shuffle_ps(cl,ch,0xEE),1);
                const __m256 sv=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(sl,sh,0x44)),_mm_shuffle_ps(sl,sh,0xEE),1);
                const __m256 r0=_mm256_sub_ps(_mm256_mul_ps(x0,cv),_mm256_mul_ps(x1,sv));
                const __m256 r1=_mm256_add_ps(_mm256_mul_ps(x0,sv),_mm256_mul_ps(x1,cv));
                const __m256 e=_mm256_unpacklo_ps(r0,r1);
                const __m256 o=_mm256_unpackhi_ps(r0,r1);
                _mm_storeu_ps(qh+2*i,_mm256_castps256_ps128(e));
                _mm_storeu_ps(qh+2*i+4,_mm256_castps256_ps128(o));
                _mm_storeu_ps(qh+2*i+8,_mm256_extractf128_ps(e,1));
                _mm_storeu_ps(qh+2*i+12,_mm256_extractf128_ps(o,1));
            }
            for (;i<half;i++){
                float x0=qh[2*i];
                float x1=qh[2*i+1];
                qh[2*i]=x0*c[i]-x1*s[i];
                qh[2*i+1]=x0*s[i]+x1*c[i];
            }
        }
    }
}
#endif
