#ifndef GGUF_H
#define GGUF_H
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gguf {
enum : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7,
    T_Q8_0 = 8, T_Q8_1 = 9, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13,
    T_Q6_K = 14, T_Q8_K = 15, T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18,
    T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23,
    T_I8 = 24, T_I16 = 25, T_I32 = 26, T_I64 = 27, T_F64 = 28, T_IQ1_M = 29,
    T_BF16 = 30
};

enum : uint32_t {
    M_U8 = 0, M_I8 = 1, M_U16 = 2, M_I16 = 3, M_U32 = 4, M_I32 = 5,
    M_F32 = 6, M_BOOL = 7, M_STR = 8, M_ARR = 9, M_U64 = 10, M_I64 = 11, M_F64 = 12
};

inline const char* type_name(uint32_t t) {
    switch (t) {
        case T_F32: return "f32";
        case T_F16: return "f16";
        case T_BF16: return "bf16";
        case T_Q4_0: return "q4_0";
        case T_Q4_1: return "q4_1";
        case T_Q5_0: return "q5_0";
        case T_Q5_1: return "q5_1";
        case T_Q8_0: return "q8_0";
        case T_Q8_1: return "q8_1";
        case T_Q2_K: return "q2_K";
        case T_Q3_K: return "q3_K";
        case T_Q4_K: return "q4_K";
        case T_Q5_K: return "q5_K";
        case T_Q6_K: return "q6_K";
        case T_IQ4_NL: return "iq4_nl";
        case T_IQ4_XS: return "iq4_xs";
        case T_I8: return "i8";
        case T_I16: return "i16";
        case T_I32: return "i32";
        case T_I64: return "i64";
        case T_F64: return "f64";
        default: return "?";
    }
}

inline size_t block_elems(uint32_t t) {
    switch (t) {
        case T_F32: case T_F16: case T_BF16: case T_I8: case T_I16:
        case T_I32: case T_I64: case T_F64: return 1;
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_Q8_1: case T_IQ4_NL: return 32;
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_Q8_K: case T_IQ4_XS: return 256;
        default: return 0;
    }
}

inline size_t block_bytes(uint32_t t) {
    switch (t) {
        case T_F32: case T_I32: return 4;
        case T_F16: case T_BF16: case T_I16: return 2;
        case T_F64: case T_I64: return 8;
        case T_I8: return 1;
        case T_Q4_0: return 18;
        case T_Q4_1: return 20;
        case T_Q5_0: return 22;
        case T_Q5_1: return 24;
        case T_Q8_0: return 34;
        case T_Q8_1: return 36;
        case T_Q2_K: return 84;
        case T_Q3_K: return 110;
        case T_Q4_K: return 144;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_Q8_K: return 292;
        case T_IQ4_NL: return 18;
        case T_IQ4_XS: return 136;
        default: return 0;
    }
}

inline uint64_t type_bytes(uint32_t t, uint64_t nelem) {
    const size_t be = block_elems(t), bb = block_bytes(t);
    if (be == 0) throw std::runtime_error(std::string("gguf: 不认识的量化类型 ") + type_name(t));
    if (nelem % be != 0) throw std::runtime_error("gguf: 元素个数不是块的整数倍");
    return (nelem / be) * bb;
}

inline float half_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) & 1u;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s << 31;
        } else {
            uint32_t mm = m;
            int k = 0;
            while ((mm & 0x400u) == 0) { mm <<= 1; k++; }
            bits = (s << 31) | ((uint32_t)(113 - k) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        bits = (s << 31) | ((e + 112u) << 23) | (m << 13);
    }
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline uint16_t u16_at(const unsigned char* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline float f16_at(const unsigned char* p) { return half_to_f32(u16_at(p)); }

inline void get_scale_min_k4(int j, const unsigned char* q, unsigned char& sc, unsigned char& mn) {
    if (j < 4) {
        sc = (unsigned char)(q[j] & 63);
        mn = (unsigned char)(q[j + 4] & 63);
    } else {
        sc = (unsigned char)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = (unsigned char)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

inline void dequant(uint32_t type, const unsigned char* src, float* dst, uint64_t n) {
    switch (type) {
        case T_F32:
            std::memcpy(dst, src, (size_t)n * 4);
            return;
        case T_F16:
            for (uint64_t i = 0; i < n; i++) dst[i] = f16_at(src + i * 2);
            return;
        case T_BF16:
            for (uint64_t i = 0; i < n; i++) dst[i] = bf16_to_f32(u16_at(src + i * 2));
            return;
        case T_Q8_0: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 34;
                const float d = f16_at(b);
                for (int j = 0; j < 32; j++) dst[i * 32 + j] = (float)(int8_t)b[2 + j] * d;
            }
            return;
        }
        case T_Q4_0: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 18;
                const float d = f16_at(b);
                const unsigned char* q = b + 2;
                for (int j = 0; j < 16; j++) {
                    dst[i * 32 + j] = (float)((int)(q[j] & 0xF) - 8) * d;
                    dst[i * 32 + j + 16] = (float)((int)(q[j] >> 4) - 8) * d;
                }
            }
            return;
        }
        case T_Q4_1: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 20;
                const float d = f16_at(b);
                const float m = f16_at(b + 2);
                const unsigned char* q = b + 4;
                for (int j = 0; j < 16; j++) {
                    dst[i * 32 + j] = (float)(q[j] & 0xF) * d + m;
                    dst[i * 32 + j + 16] = (float)(q[j] >> 4) * d + m;
                }
            }
            return;
        }
        case T_Q5_0: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 22;
                const float d = f16_at(b);
                uint32_t qh;
                std::memcpy(&qh, b + 2, 4);
                const unsigned char* q = b + 6;
                for (int j = 0; j < 16; j++) {
                    const int x0 = (q[j] & 0xF) | (int)(((qh >> j) & 1u) << 4);
                    const int x1 = (q[j] >> 4) | (int)(((qh >> (j + 16)) & 1u) << 4);
                    dst[i * 32 + j] = (float)(x0 - 16) * d;
                    dst[i * 32 + j + 16] = (float)(x1 - 16) * d;
                }
            }
            return;
        }
        case T_Q5_1: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 24;
                const float d = f16_at(b);
                const float m = f16_at(b + 2);
                uint32_t qh;
                std::memcpy(&qh, b + 4, 4);
                const unsigned char* q = b + 8;
                for (int j = 0; j < 16; j++) {
                    const int x0 = (q[j] & 0xF) | (int)(((qh >> j) & 1u) << 4);
                    const int x1 = (q[j] >> 4) | (int)(((qh >> (j + 16)) & 1u) << 4);
                    dst[i * 32 + j] = (float)x0 * d + m;
                    dst[i * 32 + j + 16] = (float)x1 * d + m;
                }
            }
            return;
        }
        case T_Q4_K: {
            const uint64_t nb = n / 256;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 144;
                const float d = f16_at(b);
                const float dmin = f16_at(b + 2);
                const unsigned char* sc = b + 4;
                const unsigned char* q = b + 16;
                float* y = dst + i * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    unsigned char s0, m0, s1, m1;
                    get_scale_min_k4(is, sc, s0, m0);
                    get_scale_min_k4(is + 1, sc, s1, m1);
                    const float d1 = d * (float)s0, n1 = dmin * (float)m0;
                    const float d2 = d * (float)s1, n2 = dmin * (float)m1;
                    for (int l = 0; l < 32; l++) y[l] = d1 * (float)(q[l] & 0xF) - n1;
                    for (int l = 0; l < 32; l++) y[l + 32] = d2 * (float)(q[l] >> 4) - n2;
                    q += 32;
                    y += 64;
                    is += 2;
                }
            }
            return;
        }
        case T_Q5_K: {
            const uint64_t nb = n / 256;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 176;
                const float d = f16_at(b);
                const float dmin = f16_at(b + 2);
                const unsigned char* sc = b + 4;
                const unsigned char* qh = b + 16;
                const unsigned char* q = b + 48;
                float* y = dst + i * 256;
                int is = 0;
                unsigned char u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    unsigned char s0, m0, s1, m1;
                    get_scale_min_k4(is, sc, s0, m0);
                    get_scale_min_k4(is + 1, sc, s1, m1);
                    const float d1 = d * (float)s0, n1 = dmin * (float)m0;
                    const float d2 = d * (float)s1, n2 = dmin * (float)m1;
                    for (int l = 0; l < 32; l++)
                        y[l] = d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - n1;
                    for (int l = 0; l < 32; l++)
                        y[l + 32] = d2 * (float)((q[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - n2;
                    q += 32;
                    y += 64;
                    is += 2;
                    u1 = (unsigned char)(u1 << 2);
                    u2 = (unsigned char)(u2 << 2);
                }
            }
            return;
        }
        case T_Q6_K: {
            const uint64_t nb = n / 256;
            for (uint64_t i = 0; i < nb; i++) {
                const unsigned char* b = src + i * 210;
                const unsigned char* ql = b;
                const unsigned char* qh = b + 128;
                const signed char* sc = (const signed char*)(b + 192);
                const float d = f16_at(b + 208);
                float* y = dst + i * 256;
                for (int nn = 0; nn < 256; nn += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l] = d * (float)sc[is] * (float)q1;
                        y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                        y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                        y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return;
        }
        default:
            throw std::runtime_error(std::string("gguf: 还没写这种量化的反量化 ") + type_name(type));
    }
}

struct Meta {
    uint32_t type = 0;
    uint32_t elem_type = 0;
    uint64_t n = 0;
    long long i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<long long> ints;
    std::vector<double> floats;
    std::vector<std::string> strs;
    long long as_int(long long d = 0) const {
        if (type != M_ARR) return i;
        return ints.empty() ? d : ints[0];
    }
    double as_float(double d = 0) const {
        if (type != M_ARR) return f;
        return floats.empty() ? d : floats[0];
    }
};

struct Tensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> shape;
    uint64_t offset = 0;
    uint64_t n_elem = 0;
    uint64_t nbytes = 0;
    uint64_t ne(int i) const { return shape[(size_t)i]; }
};

namespace rd {

inline void bytes(std::istream& f, void* p, size_t n) {
    if (n == 0) return;
    f.read(reinterpret_cast<char*>(p), (std::streamsize)n);
    if (!f) throw std::runtime_error("gguf: 文件被截断");
}

inline uint8_t u8(std::istream& f) {
    unsigned char b[1];
    bytes(f, b, 1);
    return b[0];
}

inline uint16_t u16(std::istream& f) {
    unsigned char b[2];
    bytes(f, b, 2);
    return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}

inline uint32_t u32(std::istream& f) {
    unsigned char b[4];
    bytes(f, b, 4);
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)b[i] << (8 * i);
    return v;
}

inline uint64_t u64(std::istream& f) {
    unsigned char b[8];
    bytes(f, b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

inline float f32(std::istream& f) {
    const uint32_t v = u32(f);
    float x;
    std::memcpy(&x, &v, 4);
    return x;
}

inline double f64(std::istream& f) {
    const uint64_t v = u64(f);
    double x;
    std::memcpy(&x, &v, 8);
    return x;
}

inline std::string str(std::istream& f, bool v3) {
    const uint64_t n = v3 ? u64(f) : (uint64_t)u32(f);
    std::string s;
    s.resize((size_t)n);
    if (n != 0) bytes(f, &s[0], (size_t)n);
    return s;
}

inline void scalar(std::istream& f, uint32_t t, Meta& m) {
    switch (t) {
        case M_U8: m.i = (long long)u8(f); break;
        case M_I8: m.i = (long long)(int8_t)u8(f); break;
        case M_U16: m.i = (long long)u16(f); break;
        case M_I16: m.i = (long long)(int16_t)u16(f); break;
        case M_U32: m.i = (long long)u32(f); break;
        case M_I32: m.i = (long long)(int32_t)u32(f); break;
        case M_F32: m.f = (double)f32(f); break;
        case M_BOOL: m.b = u8(f) != 0; m.i = m.b ? 1 : 0; break;
        case M_U64: m.i = (long long)u64(f); break;
        case M_I64: m.i = (long long)u64(f); break;
        case M_F64: m.f = f64(f); break;
        default: throw std::runtime_error("gguf: 不认识的元数据类型 " + std::to_string(t));
    }
}

inline void meta_value(std::istream& f, bool v3, Meta& m) {
    m.type = u32(f);
    if (m.type == M_STR) {
        m.s = str(f, v3);
        return;
    }
    if (m.type != M_ARR) {
        scalar(f, m.type, m);
        return;
    }
    m.elem_type = u32(f);
    m.n = v3 ? u64(f) : (uint64_t)u32(f);
    if (m.elem_type == M_STR) {
        m.strs.resize((size_t)m.n);
        for (uint64_t i = 0; i < m.n; i++) m.strs[(size_t)i] = str(f, v3);
        return;
    }
    if (m.elem_type == M_ARR) throw std::runtime_error("gguf: 不支持嵌套数组");
    if (m.elem_type == M_F32 || m.elem_type == M_F64) {
        m.floats.resize((size_t)m.n);
        for (uint64_t i = 0; i < m.n; i++) {
            Meta e;
            scalar(f, m.elem_type, e);
            m.floats[(size_t)i] = e.f;
        }
        return;
    }
    m.ints.resize((size_t)m.n);
    for (uint64_t i = 0; i < m.n; i++) {
        Meta e;
        scalar(f, m.elem_type, e);
        m.ints[(size_t)i] = e.i;
    }
}

}  // namespace rd

struct File {
    std::string path;
    uint32_t version = 0;
    uint64_t align = 32;
    uint64_t data_off = 0;
    uint64_t file_size = 0;
    std::vector<std::pair<std::string, Meta> > kvs;
    std::vector<Tensor> tensors;

    const Meta* meta(const std::string& k) const {
        for (size_t i = 0; i < kvs.size(); i++)
            if (kvs[i].first == k) return &kvs[i].second;
        return nullptr;
    }
    const Tensor* find(const std::string& n) const {
        for (size_t i = 0; i < tensors.size(); i++)
            if (tensors[i].name == n) return &tensors[i];
        return nullptr;
    }
    std::vector<unsigned char> raw(const Tensor& t) const {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("gguf: 打不开 " + path);
        f.seekg((std::streamoff)(data_off + t.offset));
        std::vector<unsigned char> buf((size_t)t.nbytes);
        if (t.nbytes != 0) rd::bytes(f, buf.data(), (size_t)t.nbytes);
        return buf;
    }
    std::vector<float> f32(const Tensor& t) const {
        const std::vector<unsigned char> b = raw(t);
        std::vector<float> v((size_t)t.n_elem);
        dequant(t.type, b.data(), v.data(), t.n_elem);
        return v;
    }
};

inline File open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("gguf: 打不开 " + path);
    File g;
    g.path = path;
    char magic[4];
    rd::bytes(f, magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("gguf: magic 不是 GGUF");
    g.version = rd::u32(f);
    if (g.version < 2 || g.version > 3)
        throw std::runtime_error("gguf: 只支持 v2/v3，遇到 v" + std::to_string(g.version));
    const bool v3 = (g.version == 3);
    const uint64_t nt = v3 ? rd::u64(f) : (uint64_t)rd::u32(f);
    const uint64_t nk = v3 ? rd::u64(f) : (uint64_t)rd::u32(f);
    for (uint64_t i = 0; i < nk; i++) {
        std::string key = rd::str(f, v3);
        Meta m;
        rd::meta_value(f, v3, m);
        g.kvs.push_back(std::make_pair(key, m));
    }
    g.tensors.resize((size_t)nt);
    for (uint64_t i = 0; i < nt; i++) {
        Tensor& t = g.tensors[(size_t)i];
        t.name = rd::str(f, v3);
        const uint32_t nd = rd::u32(f);
        if (nd == 0 || nd > 4) throw std::runtime_error("gguf: 维度数不对 " + t.name);
        t.shape.resize(nd);
        for (uint32_t d = 0; d < nd; d++) t.shape[d] = rd::u64(f);
        t.type = rd::u32(f);
        t.offset = rd::u64(f);
        t.n_elem = 1;
        for (uint32_t d = 0; d < nd; d++) t.n_elem *= t.shape[d];
        t.nbytes = type_bytes(t.type, t.n_elem);
    }
    const Meta* al = g.meta("general.alignment");
    if (al != nullptr) {
        const long long a = al->as_int(32);
        if (a > 0) g.align = (uint64_t)a;
    }
    const std::streamoff here = f.tellg();
    g.data_off = (uint64_t)((here + (std::streamoff)g.align - 1) / (std::streamoff)g.align * (std::streamoff)g.align);
    f.seekg(0, std::ios::end);
    g.file_size = (uint64_t)f.tellg();
    return g;
}

}  // namespace gguf

#endif  // GGUF_H
