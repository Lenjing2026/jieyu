#ifndef MATMUL_Q_H
#define MATMUL_Q_H
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <immintrin.h>
#include "../date_libs/wmat.h"
#include "../date_libs/check_cpu.h"

namespace qmat {

enum : uint8_t { Q4K = 0, Q5K = 1, Q6K = 2, Q8_0 = 3 };

inline size_t block_elems(uint8_t t) { return t == Q8_0 ? 32 : 256; }

inline size_t block_bytes(uint8_t t) {
    switch (t) {
        case Q4K: return 144;
        case Q5K: return 176;
        case Q6K: return 210;
        case Q8_0: return 34;
        default: throw std::runtime_error("qmat: 不认识的量化类型");
    }
}

inline size_t row_bytes(uint8_t t, int width) {
    const size_t be = block_elems(t);
    if (width <= 0 || (size_t)width % be != 0) throw std::runtime_error("qmat: 行长不是量化块的整数倍");
    return (size_t)width / be * block_bytes(t);
}

inline uint16_t u16(const unsigned char* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline float f16(const unsigned char* p) { return wmat::to_f32(u16(p)); }

inline void get_scale_min_k4(int j, const unsigned char* q, unsigned char& sc, unsigned char& mn) {
    if (j < 4) {
        sc = (unsigned char)(q[j] & 63);
        mn = (unsigned char)(q[j + 4] & 63);
    } else {
        sc = (unsigned char)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = (unsigned char)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__attribute__((target("avx2,fma")))
inline void dequant_q4k_avx2(const unsigned char* b, float* dst) {
    const float d = f16(b), dmin = f16(b + 2);
    const unsigned char* sc = b + 4;
    const unsigned char* qs = b + 16;
    int is = 0;
    for (int g = 0; g < 4; g++) {
        unsigned char s0, m0, s1, m1;
        get_scale_min_k4(is, sc, s0, m0);
        get_scale_min_k4(is + 1, sc, s1, m1);
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

inline void dequant_block(uint8_t t, const void* src, float* dst) {
    const unsigned char* b = static_cast<const unsigned char*>(src);
    switch (t) {
        case Q4K: {
            if (cpu_can.avx2 && cpu_can.fma3) { dequant_q4k_avx2(b, dst); return; }
            const float d = f16(b), dmin = f16(b + 2);
            const unsigned char* sc = b + 4;
            const unsigned char* qs = b + 16;
            int is = 0;
            for (int g = 0; g < 4; g++) {
                unsigned char s0, m0, s1, m1;
                get_scale_min_k4(is, sc, s0, m0);
                get_scale_min_k4(is + 1, sc, s1, m1);
                const float d1 = d * (float)s0, n1 = dmin * (float)m0;
                const float d2 = d * (float)s1, n2 = dmin * (float)m1;
                for (int l = 0; l < 32; l++) dst[g * 64 + l] = d1 * (float)(qs[l] & 0xF) - n1;
                for (int l = 0; l < 32; l++) dst[g * 64 + 32 + l] = d2 * (float)(qs[l] >> 4) - n2;
                qs += 32;
                is += 2;
            }
            return;
        }
        case Q5K: {
            const float d = f16(b), dmin = f16(b + 2);
            const unsigned char* sc = b + 4;
            const unsigned char* qh = b + 16;
            const unsigned char* qs = b + 48;
            int is = 0;
            unsigned char u1 = 1, u2 = 2;
            for (int g = 0; g < 4; g++) {
                unsigned char s0, m0, s1, m1;
                get_scale_min_k4(is, sc, s0, m0);
                get_scale_min_k4(is + 1, sc, s1, m1);
                const float d1 = d * (float)s0, n1 = dmin * (float)m0;
                const float d2 = d * (float)s1, n2 = dmin * (float)m1;
                for (int l = 0; l < 32; l++)
                    dst[g * 64 + l] = d1 * (float)((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - n1;
                for (int l = 0; l < 32; l++)
                    dst[g * 64 + 32 + l] = d2 * (float)((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - n2;
                qs += 32;
                is += 2;
                u1 = (unsigned char)(u1 << 2);
                u2 = (unsigned char)(u2 << 2);
            }
            return;
        }
        case Q6K: {
            const unsigned char* ql = b;
            const unsigned char* qh = b + 128;
            const signed char* sc = (const signed char*)(b + 192);
            const float d = f16(b + 208);
            for (int nn = 0; nn < 2; nn++) {
                for (int l = 0; l < 32; l++) {
                    const int is = l / 16;
                    const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    dst[nn * 128 + l] = d * (float)sc[is] * (float)q1;
                    dst[nn * 128 + l + 32] = d * (float)sc[is + 2] * (float)q2;
                    dst[nn * 128 + l + 64] = d * (float)sc[is + 4] * (float)q3;
                    dst[nn * 128 + l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
            return;
        }
        case Q8_0: {
            const float d = f16(b);
            for (int l = 0; l < 32; l++) dst[l] = (float)(int8_t)b[2 + l] * d;
            return;
        }
        default:
            throw std::runtime_error("qmat: 不认识的量化类型");
    }
}

__attribute__((target("avx2,fma")))
inline float dot_row_avx2(const float* a,const unsigned char* row,uint8_t t,int k){
    const size_t be=block_elems(t),bb=block_bytes(t),nblk=(size_t)k/be;
    alignas(32) float blk[256];
    __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
    size_t off=0;
    for(size_t bl=0;bl<nblk;bl++,off+=be){
        dequant_block(t,row+bl*bb,blk);
        const float* av=a+off;
        for(size_t l=0;l<be;l+=32){
            s0=_mm256_fmadd_ps(_mm256_loadu_ps(av+l),_mm256_loadu_ps(blk+l),s0);
            s1=_mm256_fmadd_ps(_mm256_loadu_ps(av+l+8),_mm256_loadu_ps(blk+l+8),s1);
            s2=_mm256_fmadd_ps(_mm256_loadu_ps(av+l+16),_mm256_loadu_ps(blk+l+16),s2);
            s3=_mm256_fmadd_ps(_mm256_loadu_ps(av+l+24),_mm256_loadu_ps(blk+l+24),s3);
        }
    }
    const __m256 s=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
    __m128 r=_mm_add_ps(_mm256_castps256_ps128(s),_mm256_extractf128_ps(s,1));
    r=_mm_hadd_ps(r,r);
    r=_mm_hadd_ps(r,r);
    return _mm_cvtss_f32(r);
}

inline void matmul_q(const float* a, const void* b, float* c, int m, int n, int k, uint8_t t) {
    const size_t be = block_elems(t), bb = block_bytes(t);
    const size_t rbytes = (size_t)k / be * bb;
    const size_t nblk = (size_t)k / be;
    const unsigned char* base = static_cast<const unsigned char*>(b);
    if (m == 1 && cpu_can.avx2 && cpu_can.fma3) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) c[j] = dot_row_avx2(a, base + (size_t)j * rbytes, t, k);
        return;
    }
    #pragma omp parallel
    {
        std::vector<float> acc((size_t)m);
        float blk[256];
        #pragma omp for
        for (int j = 0; j < n; j++) {
            const unsigned char* row = base + (size_t)j * rbytes;
            for (int i = 0; i < m; i++) acc[(size_t)i] = 0.0f;
            for (size_t bl = 0; bl < nblk; bl++) {
                dequant_block(t, row + bl * bb, blk);
                const size_t off = bl * be;
                for (size_t l = 0; l < be; l++) {
                    const float bv = blk[l];
                    for (int i = 0; i < m; i++) acc[(size_t)i] += a[(size_t)i * k + off + l] * bv;
                }
            }
            for (int i = 0; i < m; i++) c[(size_t)i * n + j] = acc[(size_t)i];
        }
    }
}

}  // namespace qmat

#endif  // MATMUL_Q_H
