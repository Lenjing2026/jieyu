#ifndef Q4_FMA_H
#define Q4_FMA_H
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "../../date_libs/wmat.h"
__attribute__((target("avx2,fma")))
void q4_fma(const unsigned char* b, float* dst) {
    auto hv = [](const unsigned char* p) {
        uint16_t v;
        std::memcpy(&v, p, 2);
        return wmat::to_f32(v);
    };
    auto sm = [](int j, const unsigned char* q, unsigned char& sc, unsigned char& mn) {
        if (j < 4) {
            sc = (unsigned char)(q[j] & 63);
            mn = (unsigned char)(q[j + 4] & 63);
        } else {
            sc = (unsigned char)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            mn = (unsigned char)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    };
    const float d = hv(b), dmin = hv(b + 2);
    const unsigned char* sc = b + 4;
    const unsigned char* qs = b + 16;
    int is = 0;
    for (int g = 0; g < 4; g++) {
        unsigned char s0, m0, s1, m1;
        sm(is, sc, s0, m0);
        sm(is + 1, sc, s1, m1);
        const __m256 dd1 = _mm256_set1_ps(d * (float)s0), mm1 = _mm256_set1_ps(-dmin * (float)m0);
        const __m256 dd2 = _mm256_set1_ps(d * (float)s1), mm2 = _mm256_set1_ps(-dmin * (float)m1);
        for (int l = 0; l < 32; l += 16) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + l));
            const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
            const __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
            const __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
            const __m256 h0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
            const __m256 h1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
            _mm256_storeu_ps(dst + g * 64 + l, _mm256_fmadd_ps(l0, dd1, mm1));
            _mm256_storeu_ps(dst + g * 64 + l + 8, _mm256_fmadd_ps(l1, dd1, mm1));
            _mm256_storeu_ps(dst + g * 64 + 32 + l, _mm256_fmadd_ps(h0, dd2, mm2));
            _mm256_storeu_ps(dst + g * 64 + 32 + l + 8, _mm256_fmadd_ps(h1, dd2, mm2));
        }
        qs += 32;
        is += 2;
    }
}

#endif  // Q4_FMA_H
