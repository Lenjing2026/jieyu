#ifndef Q4_AVX_H
#define Q4_AVX_H
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "../../date_libs/wmat.h"
__attribute__((target("avx")))
void q4_avx(const unsigned char* b,float* dst){
    auto hv=[](const unsigned char* p){ uint16_t v; std::memcpy(&v,p,2); return wmat::to_f32(v); };
    auto sm=[](int j,const unsigned char* q,unsigned char& sc,unsigned char& mn){
        if(j<4){ sc=(unsigned char)(q[j]&63); mn=(unsigned char)(q[j+4]&63); }
        else{ sc=(unsigned char)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); mn=(unsigned char)((q[j+4]>>4)|((q[j]>>6)<<4)); }
    };
    const float d=hv(b),dmin=hv(b+2);
    const unsigned char* sc=b+4;
    const unsigned char* qs=b+16;
    int is=0;
    for(int g=0;g<4;g++){
        unsigned char s0,m0,s1,m1;
        sm(is,sc,s0,m0);
        sm(is+1,sc,s1,m1);
        const __m256 dd1=_mm256_set1_ps(d*(float)s0),mm1=_mm256_set1_ps(-dmin*(float)m0);
        const __m256 dd2=_mm256_set1_ps(d*(float)s1),mm2=_mm256_set1_ps(-dmin*(float)m1);
        auto put=[&](float* p,const __m128i x,const int top) __attribute__((target("avx"))){
            const __m256 f=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(x))),
                                                _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(x,4))),1);
            _mm256_storeu_ps(p,_mm256_add_ps(_mm256_mul_ps(f,top?dd2:dd1),top?mm2:mm1));
        };
        for(int l=0;l<32;l+=16){
            const __m128i v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(qs+l));
            const __m128i lo=_mm_and_si128(v,_mm_set1_epi8(0x0F));
            const __m128i hg=_mm_and_si128(_mm_srli_epi16(v,4),_mm_set1_epi8(0x0F));
            put(dst+g*64+l,lo,0);
            put(dst+g*64+l+8,_mm_srli_si128(lo,8),0);
            put(dst+g*64+32+l,hg,1);
            put(dst+g*64+32+l+8,_mm_srli_si128(hg,8),1);
        }
        qs+=32;
        is+=2;
    }
}
#endif  // Q4_AVX_H
