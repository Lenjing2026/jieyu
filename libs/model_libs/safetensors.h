#ifndef SAFETENSORS_H
#define SAFETENSORS_H

// ============================================================
//  safetensors.h —— 读 HuggingFace safetensors
//
//  文件布局：
//    [0..7]        header 字节数 N（小端 u64）
//    [8..8+N)      header：一段 JSON
//                  {"张量名":{"dtype":"F32","shape":[..],"data_offsets":[begin,end]}, ...}
//    [8+N..文件尾)  各张量的原始数据
//
//  header 直接交给 libs/json_libs 的递归下降解析器（json_detail::Parser）
// ============================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../json_libs/json_parser.h"
#include "bfile.h"

namespace safetensors {

struct Info {
    std::string name;
    std::string src_path;  // 来自哪个 safetensors（分片时用得上）
    uint8_t dtype = bfile::F32;
    std::vector<uint32_t> shape;
    uint64_t begin = 0;   // 数据在文件里的起点（绝对偏移）
    uint64_t nbytes = 0;  // 元素个数 * 单元素字节数
    uint32_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1u; }
};

inline uint64_t elems(const std::vector<uint32_t>& shape) {
    uint64_t n = 1;
    for (uint32_t s : shape) n *= s;
    return n;
}

struct File {
    std::string path;
    uint64_t size = 0;
    uint64_t data_base = 0;
    std::vector<Info> tensors;

    const Info* find(const std::string& name) const {
        for (const Info& t : tensors) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
};

inline uint8_t dtype_code(const std::string& text) {
    const int code = bfile::dtype_of(text);
    if (code < 0) {
        throw std::runtime_error("safetensors: 不认识的 dtype: " + text
                                 + "（当前支持 F32/F16/BF16/I8/I32）");
    }
    return static_cast<uint8_t>(code);
}

inline uint64_t parse_u64(const std::string& text) {
    return std::strtoull(text.c_str(), nullptr, 10);
}

// f16 -> f32（含次正规数）
inline float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                shift += 1;
            }
            man &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(113 - shift) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, 4);
    return out;
}

// bf16 就是 f32 的高 16 位
inline float bf16_to_float(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline float f16_to_float_bytes(const unsigned char* p) {
    uint16_t h = static_cast<uint16_t>(p[0] | (p[1] << 8));
    return half_to_float(h);
}

inline float bf16_to_float_bytes(const unsigned char* p) {
    uint16_t v = static_cast<uint16_t>(p[0] | (p[1] << 8));
    return bf16_to_float(v);
}

// 把任意支持的 dtype 的原始字节转成 f32
inline void to_f32(const unsigned char* src, uint64_t count, uint8_t dtype, float* dst) {
    switch (dtype) {
        case bfile::F32:
            std::memcpy(dst, src, static_cast<size_t>(count) * 4);
            break;
        case bfile::F16:
            for (uint64_t i = 0; i < count; i++) dst[i] = f16_to_float_bytes(src + i * 2);
            break;
        case bfile::BF16:
            for (uint64_t i = 0; i < count; i++) dst[i] = bf16_to_float_bytes(src + i * 2);
            break;
        case bfile::I8:
            for (uint64_t i = 0; i < count; i++) dst[i] = static_cast<float>(static_cast<int8_t>(src[i]));
            break;
        case bfile::I32:
            for (uint64_t i = 0; i < count; i++) {
                int32_t v = 0;
                std::memcpy(&v, src + i * 4, 4);
                dst[i] = static_cast<float>(v);
            }
            break;
        default:
            throw std::runtime_error("safetensors: 不支持的 dtype");
    }
}

// 读一段原始字节
inline void read_range(const std::string& path, uint64_t begin, uint64_t nbytes,
                       std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: 打不开 " + path);
    f.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
    out.resize(static_cast<size_t>(nbytes));
    if (nbytes > 0) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nbytes));
        if (static_cast<uint64_t>(f.gcount()) != nbytes) {
            throw std::runtime_error("safetensors: 数据被截断 " + path);
        }
    }
}

inline void open(const std::string& path, File& out) {
    out = File();
    out.path = path;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: 打不开 " + path);
    f.seekg(0, std::ios::end);
    out.size = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    if (out.size < 8) throw std::runtime_error("safetensors: 文件太短 " + path);

    unsigned char length_buf[8] = {};
    f.read(reinterpret_cast<char*>(length_buf), 8);
    const uint64_t head_len = bfile::read_u64(length_buf);
    if (head_len == 0 || 8 + head_len > out.size) {
        throw std::runtime_error("safetensors: header 长度不合法 " + path);
    }
    std::string head(static_cast<size_t>(head_len), '\0');
    f.read(head.data(), static_cast<std::streamsize>(head_len));
    if (static_cast<uint64_t>(f.gcount()) != head_len) {
        throw std::runtime_error("safetensors: header 被截断 " + path);
    }
    out.data_base = 8 + head_len;

    json_libs::json_detail::Value root;
    json_libs::json_detail::Parser parser(head);
    if (!parser.parse(root)) {
        throw std::runtime_error("safetensors: header 解析失败: " + parser.error());
    }
    if (root.type != json_libs::json_detail::Value::Type::Object) {
        throw std::runtime_error("safetensors: header 不是 JSON 对象");
    }

    for (const auto& member : root.members) {
        if (member.first == "__metadata__") continue;
        const json_libs::json_detail::Value& node = member.second;
        if (node.type != json_libs::json_detail::Value::Type::Object) continue;
        Info info;
        info.name = member.first;
        info.src_path = path;
        bool has_dtype = false, has_shape = false, has_offsets = false;
        for (const auto& field : node.members) {
            if (field.first == "dtype") {
                info.dtype = dtype_code(field.second.text);
                has_dtype = true;
            } else if (field.first == "shape") {
                for (const json_libs::json_detail::Value& item : field.second.items) {
                    info.shape.push_back(static_cast<uint32_t>(parse_u64(item.raw)));
                }
                has_shape = true;
            } else if (field.first == "data_offsets") {
                if (field.second.items.size() != 2) {
                    throw std::runtime_error("safetensors: data_offsets 不是 2 个元素: " + info.name);
                }
                const uint64_t begin = parse_u64(field.second.items[0].raw);
                const uint64_t end = parse_u64(field.second.items[1].raw);
                if (end < begin) throw std::runtime_error("safetensors: data_offsets 反了: " + info.name);
                info.begin = out.data_base + begin;
                info.nbytes = end - begin;
                has_offsets = true;
            }
        }
        if (!has_dtype || !has_shape || !has_offsets) {
            throw std::runtime_error("safetensors: 张量缺少字段: " + info.name);
        }
        const uint64_t want = elems(info.shape) * bfile::dtype_bytes(info.dtype);
        if (want != info.nbytes) {
            throw std::runtime_error("safetensors: shape 与 data_offsets 不符: " + info.name);
        }
        if (info.begin + info.nbytes > out.size) {
            throw std::runtime_error("safetensors: 数据越界: " + info.name);
        }
        out.tensors.push_back(info);
    }
    if (out.tensors.empty()) {
        throw std::runtime_error("safetensors: 里面一个张量都没有 " + path);
    }
}

// 写一份 safetensors（造测试模型用）
struct RawTensor {
    std::string name;
    uint8_t dtype = bfile::F32;
    std::vector<uint32_t> shape;
    std::vector<unsigned char> data;
};

// f32 的便捷版本
inline RawTensor make_f32(const std::string& name, std::vector<uint32_t> shape,
                          const std::vector<float>& values) {
    RawTensor t;
    t.name = name;
    t.dtype = bfile::F32;
    t.shape = shape;
    t.data.resize(values.size() * 4);
    std::memcpy(t.data.data(), values.data(), t.data.size());
    return t;
}

inline void write(const std::string& path, const std::vector<RawTensor>& tensors) {
    std::string head = "{";
    std::vector<unsigned char> body;
    uint64_t cursor = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const RawTensor& t = tensors[i];
        const uint64_t count = elems(t.shape);
        if (count * bfile::dtype_bytes(t.dtype) != t.data.size()) {
            throw std::runtime_error("safetensors: shape 与数据长度不符: " + t.name);
        }
        if (i > 0) head += ",";
        head += "\"" + t.name + "\":{\"dtype\":\"" + bfile::dtype_name(t.dtype) + "\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); d++) {
            if (d > 0) head += ",";
            head += std::to_string(t.shape[d]);
        }
        head += "],\"data_offsets\":[" + std::to_string(cursor) + ","
              + std::to_string(cursor + t.data.size()) + "]}";
        cursor += t.data.size();
        body.insert(body.end(), t.data.begin(), t.data.end());
        const uint64_t padded = bfile::align_up(body.size(), 8);
        body.resize(static_cast<size_t>(padded), 0);
        cursor = padded;
    }
    head += "}";

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("safetensors: 写不了 " + path);
    unsigned char length_buf[8] = {};
    bfile::write_u64(length_buf, head.size());
    f.write(reinterpret_cast<const char*>(length_buf), 8);
    f.write(head.data(), static_cast<std::streamsize>(head.size()));
    if (!body.empty()) {
        f.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(body.size()));
    }
    f.flush();
    if (!f) throw std::runtime_error("safetensors: 写入失败 " + path);
}

}  // namespace safetensors

#endif  // SAFETENSORS_H
