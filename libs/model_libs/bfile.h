#ifndef BFILE_H
#define BFILE_H

// ============================================================
//  bfile.h —— model.bf v2 格式（读写都在这）
//
//  [0..5]    魔数 "JYAIBF"
//  [6..13]   hidden_size       u64
//  [14..21]  layer_count       u64
//  [22]      mode              u8
//  [23]      num_heads         u8
//  [24]      num_kv_heads      u8
//  [25]      head_dim          u8
//  [26..29]  intermediate_size u32
//  [30..37]  vocab_size        u64
//  [38..41]  version           u32（v2 起为 2，老文件是 0）
//  [42..45]  tensor_count      u32
//  [46..49]  data_off          u32（数据区起点，64 对齐）
//  [50..53]  entry_size        u32（目录项字节数 = 104）
//  [54..63]  0
//  [64 .. data_off)      张量目录：tensor_count 个 104 字节项
//  [data_off .. 文件尾)  数据区：每个张量 64 字节对齐
//
//  目录项（104 字节，小端）：
//    [0..63]   名字（nul 结尾，最长 63 字符）
//    [64]      dtype（0=f32 1=f16 2=bf16 3=i8 4=i32）
//    [65]      ndim（最多 4）
//    [66..67]  0
//    [68..83]  shape[4] u32
//    [84..87]  0
//    [88..95]  offset  u64（相对数据区起点）
//    [96..103] nbytes  u64
// ============================================================

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfile {

inline constexpr char kMagic[6] = {'J', 'Y', 'A', 'I', 'B', 'F'};
inline constexpr uint32_t kVersion = 2;
inline constexpr size_t kBaseHeader = 38;
inline constexpr size_t kHeaderSize = 64;
inline constexpr size_t kEntrySize = 104;
inline constexpr size_t kAlign = 64;
inline constexpr uint32_t kMaxDim = 4;
inline constexpr size_t kMaxName = 63;

enum dtype_t : uint8_t { F32 = 0, F16 = 1, BF16 = 2, I8 = 3, I32 = 4, Q4K = 5, Q5K = 6, Q6K = 7, Q8_0 = 8 };

inline size_t dtype_bytes(uint8_t d) {
    switch (d) {
        case F32: case I32: return 4;
        case F16: case BF16: return 2;
        case I8: return 1;
        default: return 0;
    }
}

// 块量化：一个块几个元素 / 几字节；不是块量化就返回 0
inline size_t block_elems(uint8_t d) {
    switch (d) {
        case Q4K: case Q5K: case Q6K: return 256;
        case Q8_0: return 32;
        default: return 0;
    }
}

inline size_t block_bytes(uint8_t d) {
    switch (d) {
        case Q4K: return 144;
        case Q5K: return 176;
        case Q6K: return 210;
        case Q8_0: return 34;
        default: return 0;
    }
}

inline bool dtype_known(uint8_t d) { return dtype_bytes(d) != 0 || block_elems(d) != 0; }

// 一张张量占多少字节；组合不合法返回 0
inline size_t tensor_bytes(uint8_t d, uint64_t elems) {
    const size_t be = block_elems(d);
    if (be == 0) return elems * dtype_bytes(d);
    if (elems % be != 0) return 0;
    return (elems / be) * block_bytes(d);
}

inline const char* dtype_name(uint8_t d) {
    switch (d) {
        case F32: return "f32";
        case F16: return "f16";
        case BF16: return "bf16";
        case I8: return "i8";
        case I32: return "i32";
        case Q4K: return "q4_K";
        case Q5K: return "q5_K";
        case Q6K: return "q6_K";
        case Q8_0: return "q8_0";
        default: return "?";
    }
}

// 识别 dtype 字符串：HF 叫 F32/F16/BF16，本工程也接受 f32/float32 之类写法
inline int dtype_of(const std::string& s) {
    std::string t;
    for (char c : s) t.push_back(static_cast<char>(::toupper(static_cast<unsigned char>(c))));
    if (t == "F32" || t == "FLOAT32" || t == "FLOAT" || t == "I32" || t == "INT32") {
        return (t[0] == 'I') ? I32 : F32;
    }
    if (t == "F16" || t == "FLOAT16" || t == "HALF") return F16;
    if (t == "BF16" || t == "BFLOAT16") return BF16;
    if (t == "I8" || t == "INT8") return I8;
    if (t == "Q4K" || t == "Q4_K") return Q4K;
    if (t == "Q5K" || t == "Q5_K") return Q5K;
    if (t == "Q6K" || t == "Q6_K") return Q6K;
    if (t == "Q8_0" || t == "Q80") return Q8_0;
    return -1;
}

inline size_t align_up(size_t v, size_t a = kAlign) {
    if (a == 0) return v;
    return (v + a - 1) / a * a;
}

inline uint32_t read_u32(const unsigned char* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= static_cast<uint32_t>(p[i]) << (8 * i);
    return v;
}

inline uint64_t read_u64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

inline void write_u32(unsigned char* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
}

inline void write_u64(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
}

#pragma pack(push, 1)
struct Entry {
    char name[64];
    uint8_t dtype;
    uint8_t ndim;
    uint16_t reserved;
    uint32_t shape[kMaxDim];
    uint32_t reserved2;
    uint64_t offset;
    uint64_t nbytes;
};
#pragma pack(pop)

static_assert(sizeof(Entry) == kEntrySize, "Entry 必须是 104 字节");

inline uint64_t entry_elems(const Entry& e) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < e.ndim && i < kMaxDim; i++) n *= e.shape[i];
    return n;
}

// 只读视图：直接指向 mmap 出来的字节，不拷贝
struct View {
    const unsigned char* base = nullptr;
    size_t size = 0;
    uint32_t version = 0;
    uint32_t tensor_count = 0;
    uint64_t data_off = 0;
    bool ok = false;

    void reset() {
        base = nullptr;
        size = 0;
        version = 0;
        tensor_count = 0;
        data_off = 0;
        ok = false;
    }

    bool open(const void* map, size_t bytes) {
        reset();
        base = static_cast<const unsigned char*>(map);
        size = bytes;
        if (!base || bytes < kHeaderSize) return false;
        if (std::memcmp(base, kMagic, 6) != 0) return false;
        version = read_u32(base + 38);
        if (version < kVersion) return false;
        tensor_count = read_u32(base + 42);
        data_off = read_u32(base + 46);
        const uint32_t entry_size = read_u32(base + 50);
        if (entry_size != kEntrySize) return false;
        if (data_off < kHeaderSize || data_off > bytes) return false;
        if (kHeaderSize + static_cast<uint64_t>(tensor_count) * kEntrySize > data_off) return false;
        for (uint32_t i = 0; i < tensor_count; i++) {
            const Entry& e = *entry(i);
            if (e.ndim > kMaxDim) return false;
            if (!dtype_known(e.dtype)) return false;
            if (tensor_bytes(e.dtype, entry_elems(e)) != e.nbytes) return false;
            if (e.offset % kAlign != 0) return false;
            if (data_off + e.offset + e.nbytes > bytes) return false;
        }
        ok = true;
        return true;
    }

    const Entry* entry(uint32_t i) const {
        return reinterpret_cast<const Entry*>(base + kHeaderSize + static_cast<size_t>(i) * kEntrySize);
    }

    const Entry* find(const char* name) const {
        if (!ok || name == nullptr) return nullptr;
        for (uint32_t i = 0; i < tensor_count; i++) {
            const Entry* e = entry(i);
            if (std::strncmp(e->name, name, 64) == 0) return e;
        }
        return nullptr;
    }

    const void* data(const Entry* e) const {
        return base + data_off + e->offset;
    }

    const float* f32(const Entry* e) const {
        return reinterpret_cast<const float*>(data(e));
    }
};

// 当前装载的那一份（由 install_model 填），load_weights.h 靠它按名字取张量
inline View& current() {
    static View view;
    return view;
}

// 写文件用的一份张量（data 已经是小端原始字节）
struct Tensor {
    std::string name;
    uint8_t dtype = F32;
    std::vector<uint32_t> shape;
    std::vector<unsigned char> data;
};

inline Tensor make_f32(const std::string& name, const std::vector<uint32_t>& shape,
                       const std::vector<float>& v) {
    Tensor t;
    t.name = name;
    t.dtype = F32;
    t.shape = shape;
    t.data.resize(v.size() * sizeof(float));
    std::memcpy(t.data.data(), v.data(), t.data.size());
    return t;
}

inline Tensor make_i32(const std::string& name, const std::vector<uint32_t>& shape,
                       const std::vector<int32_t>& v) {
    Tensor t;
    t.name = name;
    t.dtype = I32;
    t.shape = shape;
    t.data.resize(v.size() * sizeof(int32_t));
    std::memcpy(t.data.data(), v.data(), t.data.size());
    return t;
}

// 把 f32 按 [rows, cols] 转置成 [cols, rows]（HF 的 nn.Linear 权重 -> 本工程布局）
inline std::vector<float> transpose(const std::vector<float>& src, int rows, int cols) {
    std::vector<float> dst(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[static_cast<size_t>(j) * rows + i] = src[static_cast<size_t>(i) * cols + j];
        }
    }
    return dst;
}

struct Header {
    uint64_t hidden = 0;
    uint64_t layers = 0;
    uint8_t mode = 1;
    uint8_t num_heads = 0;
    uint8_t num_kv_heads = 0;
    uint8_t head_dim = 0;
    uint32_t intermediate = 0;
    uint64_t vocab_size = 0;
};

// 流式写：数据由 emit 现场产出（转换大模型时不用整份塞内存）
struct StreamTensor {
    std::string name;
    uint8_t dtype = F32;
    std::vector<uint32_t> shape;
    uint64_t nbytes = 0;
    std::function<bool(std::ostream&)> emit;
};

inline void check_tensor_meta(const std::string& name, uint8_t dtype,
                              const std::vector<uint32_t>& shape, uint64_t nbytes) {
    if (name.empty() || name.size() > kMaxName) {
        throw std::runtime_error("bfile: 张量名长度必须 1..63: " + name);
    }
    if (shape.size() > kMaxDim) {
        throw std::runtime_error("bfile: 维度超过 4: " + name);
    }
    if (!dtype_known(dtype)) {
        throw std::runtime_error("bfile: 不认识的 dtype: " + name);
    }
    uint64_t elems = 1;
    for (uint32_t s : shape) elems *= s;
    if (tensor_bytes(dtype, elems) != nbytes) {
        throw std::runtime_error("bfile: shape 与数据长度不符: " + name);
    }
}

inline void write_bfile_stream(const std::string& path, const Header& h,
                               const std::vector<StreamTensor>& ts) {
    const size_t dir_bytes = ts.size() * kEntrySize;
    const size_t header_size = align_up(kHeaderSize + dir_bytes);
    if (header_size > 0xFFFFFFFFull) throw std::runtime_error("bfile: 张量太多");

    std::vector<uint64_t> offs(ts.size());
    uint64_t cursor = 0;
    for (size_t i = 0; i < ts.size(); i++) {
        check_tensor_meta(ts[i].name, ts[i].dtype, ts[i].shape, ts[i].nbytes);
        offs[i] = cursor;
        cursor = align_up(cursor + ts[i].nbytes);
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("bfile: 无法写入 " + path);

    unsigned char head[kHeaderSize] = {};
    std::memcpy(head, kMagic, 6);
    write_u64(head + 6, h.hidden);
    write_u64(head + 14, h.layers);
    head[22] = h.mode;
    head[23] = h.num_heads;
    head[24] = h.num_kv_heads;
    head[25] = h.head_dim;
    write_u32(head + 26, h.intermediate);
    write_u64(head + 30, h.vocab_size);
    write_u32(head + 38, kVersion);
    write_u32(head + 42, static_cast<uint32_t>(ts.size()));
    write_u32(head + 46, static_cast<uint32_t>(header_size));
    write_u32(head + 50, static_cast<uint32_t>(kEntrySize));
    f.write(reinterpret_cast<const char*>(head), kHeaderSize);

    for (size_t i = 0; i < ts.size(); i++) {
        Entry e = {};
        std::memcpy(e.name, ts[i].name.data(), ts[i].name.size());
        e.name[ts[i].name.size()] = '\0';
        e.dtype = ts[i].dtype;
        e.ndim = static_cast<uint8_t>(ts[i].shape.size());
        for (size_t d = 0; d < ts[i].shape.size(); d++) e.shape[d] = ts[i].shape[d];
        e.offset = offs[i];
        e.nbytes = ts[i].nbytes;
        f.write(reinterpret_cast<const char*>(&e), kEntrySize);
    }

    const size_t pad = header_size - kHeaderSize - dir_bytes;
    if (pad > 0) {
        const std::vector<char> zeros(pad, 0);
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }

    uint64_t written = 0;
    for (size_t i = 0; i < ts.size(); i++) {
        if (offs[i] > written) {
            const std::vector<char> zeros(static_cast<size_t>(offs[i] - written), 0);
            f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
            written = offs[i];
        }
        if (ts[i].emit && !ts[i].emit(f)) {
            throw std::runtime_error("bfile: 写张量失败: " + ts[i].name);
        }
        written += ts[i].nbytes;
    }
    f.flush();
    if (!f) throw std::runtime_error("bfile: 写入失败 " + path);
}

// 只读头部 + 目录（不 mmap 整个文件，转换器校验/查看目录用这个）
inline void read_header_file(const std::string& path, Header& h, std::vector<Entry>& dir,
                             uint64_t& file_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("bfile: 打不开 " + path);
    f.seekg(0, std::ios::end);
    file_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    if (file_size < kHeaderSize) throw std::runtime_error("bfile: 文件太短: " + path);

    unsigned char head[kHeaderSize] = {};
    f.read(reinterpret_cast<char*>(head), kHeaderSize);
    if (std::memcmp(head, kMagic, 6) != 0) throw std::runtime_error("model.bf magic not match");
    h.hidden = read_u64(head + 6);
    h.layers = read_u64(head + 14);
    h.mode = head[22];
    h.num_heads = head[23];
    h.num_kv_heads = head[24];
    h.head_dim = head[25];
    h.intermediate = read_u32(head + 26);
    h.vocab_size = read_u64(head + 30);

    const uint32_t version = read_u32(head + 38);
    const uint32_t count = read_u32(head + 42);
    const uint32_t data_off = read_u32(head + 46);
    const uint32_t entry_size = read_u32(head + 50);
    if (version < kVersion || entry_size != kEntrySize) {
        throw std::runtime_error("bfile: 不是 v2 格式（没有张量目录）: " + path);
    }
    if (data_off < kHeaderSize || data_off > file_size) {
        throw std::runtime_error("bfile: data_off 越界: " + path);
    }
    if (kHeaderSize + static_cast<uint64_t>(count) * kEntrySize > data_off) {
        throw std::runtime_error("bfile: 目录区放不下这么多张量: " + path);
    }

    dir.resize(count);
    if (count > 0) {
        const std::streamsize want = static_cast<std::streamsize>(count * kEntrySize);
        f.read(reinterpret_cast<char*>(dir.data()), want);
        if (f.gcount() != want) throw std::runtime_error("bfile: 目录被截断: " + path);
    }
    for (const Entry& e : dir) {
        if (e.ndim > kMaxDim || !dtype_known(e.dtype)) {
            throw std::runtime_error("bfile: 目录项不合法: " + std::string(e.name));
        }
        if (tensor_bytes(e.dtype, entry_elems(e)) != e.nbytes) {
            throw std::runtime_error("bfile: 目录项 shape 与 nbytes 不符: " + std::string(e.name));
        }
        if (e.offset % kAlign != 0) {
            throw std::runtime_error("bfile: 目录项 offset 没有 64 对齐: " + std::string(e.name));
        }
        if (data_off + e.offset + e.nbytes > file_size) {
            throw std::runtime_error("bfile: 目录项数据越界: " + std::string(e.name));
        }
    }
}

// 内存版：数据已经在手上，直接写
inline void write_bfile(const std::string& path, const Header& h, const std::vector<Tensor>& ts) {
    std::vector<StreamTensor> sinks;
    sinks.reserve(ts.size());
    for (const Tensor& t : ts) {
        StreamTensor s;
        s.name = t.name;
        s.dtype = t.dtype;
        s.shape = t.shape;
        s.nbytes = t.data.size();
        const unsigned char* ptr = t.data.empty() ? nullptr : t.data.data();
        const std::streamsize size = static_cast<std::streamsize>(t.data.size());
        s.emit = [ptr, size](std::ostream& f) -> bool {
            if (ptr != nullptr && size > 0) f.write(reinterpret_cast<const char*>(ptr), size);
            return static_cast<bool>(f);
        };
        sinks.push_back(s);
    }
    write_bfile_stream(path, h, sinks);
}

}  // namespace bfile

#endif  // BFILE_H
