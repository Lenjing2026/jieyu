#ifndef JIEYU_TEXT_TEXT_H
#define JIEYU_TEXT_TEXT_H

// ============================================================
//  libs/text/text.h —— 自测函数集合
//    check_matmul()     matmul 系列（朴素 / AVX / FMA / AVX+FMA）正确性
//    check_bfile()      model.bf 二进制格式（date_libs/date.h + model_libs/creater_model.h）
//    check_ffn()        FFN(SwiGLU) 前向 + silu（model_libs/ffn/ffn.h）
//    check_rope()       RoPE 旋转位置编码（model_libs/rope/rope.h）
//    check_rmsnorm()    RMSNorm（model_libs/rmsnorm/rmsnorm.h）
//    check_attention()  因果 GQA 注意力（model_libs/attention/attention.h）
//    check_transformer_layer()  整层前向端到端（model_libs/transformer_layer.h）
//    check_generate()   自回归生成（model_libs/generate.h）
//    check_tensor_dir() model.bf v2 张量目录（model_libs/bfile.h）
//    check_convert()    safetensors -> model.bf 转换器（model_libs/converter.h）
//    check_bind_weights() 按名字填 layers[]（model_libs/load_weights.h）
//    check_pipeline()   文本 -> token -> 生成 -> 文本（model_libs/pipeline.h）
// ============================================================

// 必须在包含 windows.h 之前定义，避免 min/max 宏污染全局
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "libs.h"
#include "date_libs/date.h"
#include "model_libs/ffn/ffn.h"
#include "model_libs/rope/rope.h"
#include "model_libs/rmsnorm/rmsnorm.h"
#include "model_libs/attention/attention.h"
#include "model_libs/transformer_layer.h"
#include "model_libs/generate.h"
#include "model_libs/bfile.h"
#include "model_libs/safetensors.h"
#include "model_libs/converter.h"
#include "model_libs/load_weights.h"
#include "model_libs/pipeline.h"
#include "tokenizer_libs/include/byte_vocab.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <system_error>
#include <filesystem>

bool check_matmul() {
    struct { int m, n, k; float eps; } cases[] = {
        {1, 1, 1, 1e-4f}, {2, 2, 2, 1e-4f}, {4, 4, 4, 1e-4f},
        {8, 8, 8, 1e-4f}, {16, 16, 16, 1e-4f}, {32, 32, 32, 1e-4f},
        {64, 64, 64, 1e-4f}, {1, 100, 50, 1e-4f}, {100, 1, 50, 1e-4f},
        {100, 50, 1, 1e-4f}, {3, 5, 7, 1e-4f}, {7, 13, 17, 1e-4f},
        {17, 31, 63, 1e-4f}, {33, 65, 129, 1e-4f}, {128, 128, 128, 1e-3f},
        {256, 256, 256, 1e-3f}, {512, 512, 512, 1e-3f}, {1, 4096, 1, 1e-4f},
        {4096, 1, 1, 1e-4f}, {1, 1, 4096, 1e-4f},
    };
    int total = sizeof(cases) / sizeof(cases[0]);
    int passed = 0;

    for (int ci = 0; ci < total; ci++) {
        int m = cases[ci].m, n = cases[ci].n, k = cases[ci].k;
        float eps = cases[ci].eps;
        size_t sA = (size_t)m * k, sB = (size_t)k * n, sC = (size_t)m * n;

        float* A = (float*)malloc(sA * sizeof(float));
        float* B = (float*)malloc(sB * sizeof(float));
        float* C_ref = (float*)malloc(sC * sizeof(float));
        float* C_got = (float*)malloc(sC * sizeof(float));

        if (!A || !B || !C_ref || !C_got) {
            free(A); free(B); free(C_ref); free(C_got);
            continue;
        }

        srand(42 + ci);
        for (size_t i = 0; i < sA; i++) A[i] = (rand() % 2000 - 1000) / 1000.0f;
        for (size_t i = 0; i < sB; i++) B[i] = (rand() % 2000 - 1000) / 1000.0f;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int t = 0; t < k; t++)
                    s += (double)A[i*k+t] * (double)B[t*n+j];
                C_ref[i*n+j] = (float)s;
            }

        // matmul 现在是“分发版”：自己 memset 后按 cpu_can 选 AVX+FMA / FMA / AVX / 朴素
        matmul(m, n, k, A, B, C_got);

        bool ok = true;
        for (size_t i = 0; i < sC; i++) {
            if (std::fabs(C_ref[i] - C_got[i]) > eps) {
                printf("FAIL: m=%d n=%d k=%d idx=%zu ref=%.6f got=%.6f\n",
                       m, n, k, i, C_ref[i], C_got[i]);
                ok = false;
                break;
            }
        }
        if (ok) passed++;

        free(A); free(B); free(C_ref); free(C_got);
    }

    printf("check_matmul: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_bfile() —— model.bf 新功能测试
//
//  被测对象：
//    libs/model_libs/creater_model.h  create_model() / create_bfile()
//    libs/model_libs/install_model.h  install_model()
//    libs/mapp_libs/Windows_InMapp.h  load_model_map() / free_model_map()
//    libs/date_libs/date.h            ModelInfo 结构体
//
//  model.bf 布局 v2（小端，详见 libs/model_libs/bfile.h）：
//    [0..5]   魔数 "JYAIBF"
//    [6..13]  hidden_size       (uint64)
//    [14..21] layer_count       (uint64)
//    [22]     mode              (uint8)
//    [23]     num_heads         (uint8)
//    [24]     num_kv_heads      (uint8)
//    [25]     head_dim          (uint8)
//    [26..29] intermediate_size (uint32)
//    [30..37] vocab_size        (uint64)
//    [38..41] version           (uint32, v2 = 2)
//    [42..45] tensor_count      (uint32)
//    [46..49] data_off          (uint32, 64 对齐)
//    [50..53] entry_size        (uint32 = 104)
//    [54..63] 零填充
//    [64..data_off)     张量目录（tensor_count × 104 字节）
//    [data_off..文件尾)  数据区（每个张量 64 字节对齐）
//    没有张量时就是 64 字节的头
// ============================================================

namespace text_detail {

// 把整个文件读成字节串（二进制，不做任何转换）
bool read_bytes(const std::string& file, std::string& out) {
    std::ifstream in(file, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

// 按小端读出 8 字节无符号整数
uint64_t read_u64_le(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

// 按小端读出 4 字节无符号整数
uint32_t read_u32_le(const unsigned char* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= static_cast<uint32_t>(p[i]) << (8 * i);
    }
    return v;
}

// 头部有效字节数 / 文件对齐长度
inline constexpr size_t kBfHeaderSize = 38;
inline constexpr size_t kBfFileSize = 64;

// v2 新增字段的位置与取值
inline constexpr size_t kBfVersionOff = 38;
inline constexpr size_t kBfTensorCountOff = 42;
inline constexpr size_t kBfDataOffOff = 46;
inline constexpr size_t kBfEntrySizeOff = 50;
inline constexpr uint32_t kBfVersion2 = 2;
inline constexpr uint32_t kBfEntrySize2 = 104;

// 一份模型超参（默认值就是测试用的期望值）
struct model_params {
    int hidden;
    int layers;
    int mode;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    model_params()
        : hidden(512), layers(16), mode(1), num_heads(8), num_kv_heads(2),
          head_dim(64), intermediate_size(2048), vocab_size(32000) {}
};

// model.bf 头部解析结果
struct bf_head {
    bool ok;                 // 文件足够长、解析成功
    size_t size;             // 文件字节数
    bool magic_ok;           // 前 6 字节 == "JYAIBF"
    bool pad_zero_ok;        // 38 字节之后的补齐是否全为 0
    uint64_t h;              // hidden_size
    uint64_t l;              // layer_count
    uint8_t mode;            // mode
    uint8_t num_heads;       // 注意力头数
    uint8_t num_kv_heads;    // KV 头数
    uint8_t head_dim;        // 单头维度
    uint32_t intermediate_size;  // FFN 中间层宽度
    uint64_t vocab_size;     // 词表大小
    uint32_t version;        // 格式版本（老文件是 0）
    uint32_t tensor_count;   // 张量个数
    uint32_t data_off;       // 数据区起点
    uint32_t entry_size;     // 目录项字节数
    bf_head()
        : ok(false), size(0), magic_ok(false), pad_zero_ok(false), h(0), l(0),
          mode(0), num_heads(0), num_kv_heads(0), head_dim(0),
          intermediate_size(0), vocab_size(0), version(0), tensor_count(0),
          data_off(0), entry_size(0) {}

    // 是不是 v2 头（带张量目录）
    bool is_v2() const {
        return ok && version >= kBfVersion2 && entry_size == kBfEntrySize2
            && data_off >= kBfFileSize;
    }

    // 与期望值逐字段比较
    bool matches(const model_params& p) const {
        return ok
            && h == static_cast<uint64_t>(p.hidden)
            && l == static_cast<uint64_t>(p.layers)
            && mode == static_cast<uint8_t>(p.mode)
            && num_heads == static_cast<uint8_t>(p.num_heads)
            && num_kv_heads == static_cast<uint8_t>(p.num_kv_heads)
            && head_dim == static_cast<uint8_t>(p.head_dim)
            && intermediate_size == static_cast<uint32_t>(p.intermediate_size)
            && vocab_size == static_cast<uint64_t>(p.vocab_size);
    }
};

// 读出 dir/model.bf 的头部
bf_head read_bf_head(const std::filesystem::path& dir) {
    bf_head head;
    std::string bytes;
    if (!read_bytes((dir / "model.bf").string(), bytes)) {
        return head;
    }
    head.size = bytes.size();
    if (bytes.size() < kBfHeaderSize) {
        return head;
    }
    const unsigned char* p = reinterpret_cast<const unsigned char*>(bytes.data());
    head.magic_ok = (std::memcmp(p, "JYAIBF", 6) == 0);
    head.h = read_u64_le(p + 6);
    head.l = read_u64_le(p + 14);
    head.mode = p[22];
    head.num_heads = p[23];
    head.num_kv_heads = p[24];
    head.head_dim = p[25];
    head.intermediate_size = read_u32_le(p + 26);
    head.vocab_size = read_u64_le(p + 30);
    if (bytes.size() >= 54) {
        head.version = read_u32_le(p + kBfVersionOff);
        head.tensor_count = read_u32_le(p + kBfTensorCountOff);
        head.data_off = read_u32_le(p + kBfDataOffOff);
        head.entry_size = read_u32_le(p + kBfEntrySizeOff);
    }

    head.pad_zero_ok = true;
    for (size_t i = kBfEntrySizeOff + 4; i < bytes.size() && i < kBfFileSize; i++) {
        if (p[i] != 0) {
            head.pad_zero_ok = false;
            break;
        }
    }
    head.ok = true;
    return head;
}

// 用工程的 json_lib 写一份 model_info.json
bool write_model_info(const std::filesystem::path& dir, const model_params& p) {
    json_lib info;
    info.SetJsonWay((dir / "model_info.json").string());
    bool ok = info.WriteJsonKey("hidden_size", std::to_string(p.hidden));
    ok = info.WriteJsonKey("layer_count", std::to_string(p.layers)) && ok;
    ok = info.WriteJsonKey("mode", std::to_string(p.mode)) && ok;
    ok = info.WriteJsonKey("num_heads", std::to_string(p.num_heads)) && ok;
    ok = info.WriteJsonKey("num_kv_heads", std::to_string(p.num_kv_heads)) && ok;
    ok = info.WriteJsonKey("head_dim", std::to_string(p.head_dim)) && ok;
    ok = info.WriteJsonKey("intermediate_size", std::to_string(p.intermediate_size)) && ok;
    ok = info.WriteJsonKey("vocab_size", std::to_string(p.vocab_size)) && ok;
    return ok;
}

// create_bfile 的调用结果（它用 throw "字符串" 报错）
struct call_result {
    bool threw;           // 是否抛了异常
    std::string message;  // 异常内容
    call_result() : threw(false) {}
};

call_result call_create_bfile(const std::filesystem::path& dir) {
    call_result r;
    try {
        create_bfile(dir);
    } catch (const char* msg) {
        r.threw = true;
        r.message = msg;
    } catch (const std::string& msg) {
        r.threw = true;
        r.message = msg;
    } catch (const std::exception& e) {
        r.threw = true;
        r.message = e.what();
    } catch (...) {
        r.threw = true;
        r.message = "(未知异常类型)";
    }
    return r;
}

}  // namespace text_detail

bool check_bfile() {
    namespace fs = std::filesystem;
    using text_detail::bf_head;
    using text_detail::call_create_bfile;
    using text_detail::read_bf_head;
    using text_detail::write_model_info;

    const int total = 18;
    int passed = 0;

    const text_detail::model_params def;   // 期望值：512/16/1/8/2/64/2048/32000

    printf("==== model.bf 新功能测试 ====\n");

    const fs::path root         = fs::path("_file") / "bfile_test";
    const fs::path dir_ok       = root / "model_ok";
    const fs::path dir_nojson   = root / "model_nojson";
    const fs::path dir_badlayer = root / "model_badlayer";
    const fs::path dir_edge     = root / "model_edge";
    const fs::path dir_diag     = root / "model_diag";
    const fs::path dir_nobf     = root / "model_nobf";
    const fs::path dir_badmagic = root / "model_badmagic";

    std::error_code ec;
    fs::remove_all(root, ec);       // 清掉上次残留，保证可重复运行
    fs::create_directories(root, ec);

    // ---------- 1. create_model 建目录 ----------
    {
        const bool created = create_model(dir_ok);
        const bool got = created && fs::exists(dir_ok);
        printf("[ 1/%d] create_model 建目录                     %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
    }

    // ---------- 2~10. 正常写入 + 逐字段校验 ----------
    {
        write_model_info(dir_ok, def);
        const text_detail::call_result r = call_create_bfile(dir_ok);
        const bf_head head = read_bf_head(dir_ok);

        if (r.threw) {
            printf("       ! create_bfile 抛异常: %s\n", r.message.c_str());
        }

        bool got = head.ok && head.size == 64;
        printf("[ 2/%d] 文件大小 %zu 字节 (v2 头，0 个张量)      %s\n", total, head.size, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.is_v2() && head.version == 2 && head.tensor_count == 0
            && head.data_off == 64 && head.entry_size == 104;
        printf("[18/%d] v2 头字段 version=2 count=0 off=64 size=104  %s\n", total, got ? "OK" : "FAIL");
        if (!got) {
            printf("       ! 实际 version=%u count=%u data_off=%u entry=%u\n",
                   (unsigned)head.version, (unsigned)head.tensor_count,
                   (unsigned)head.data_off, (unsigned)head.entry_size);
        }
        if (got) {
            passed++;
        }

        got = head.ok && head.magic_ok;
        printf("[ 3/%d] 魔数 \"JYAIBF\"                          %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.h == 512;
        printf("[ 4/%d] hidden_size 512 (读回 %llu)              %s\n", total, (unsigned long long)head.h, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.l == 16;
        printf("[ 5/%d] layer_count 16 (读回 %llu)               %s\n", total, (unsigned long long)head.l, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.mode == 1;
        printf("[ 6/%d] mode 1 (读回 %u)                         %s\n", total, (unsigned)head.mode, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.num_heads == 8 && head.num_kv_heads == 2 && head.head_dim == 64;
        printf("[ 7/%d] num_heads/kv/dim = %u/%u/%u (期望 8/2/64)  %s\n", total,
               (unsigned)head.num_heads, (unsigned)head.num_kv_heads, (unsigned)head.head_dim,
               got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.intermediate_size == 2048;
        printf("[ 8/%d] intermediate_size 2048 (读回 %u)          %s\n", total,
               (unsigned)head.intermediate_size, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.vocab_size == 32000;
        printf("[ 9/%d] vocab_size 32000 (读回 %llu)              %s\n", total,
               (unsigned long long)head.vocab_size, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.pad_zero_ok;
        printf("[10/%d] 补齐字节 54..63 全为 0                    %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
    }

    // ---------- 11. 内存映射按偏移直接读回全部字段 ----------
    {
        const fs::path bf = dir_ok / "model.bf";
        HANDLE file_handle = CreateFileW(bf.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        HANDLE map_handle = nullptr;
        LPVOID view = nullptr;
        bool got = false;

        if (file_handle != INVALID_HANDLE_VALUE) {
            map_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (map_handle != nullptr) {
                view = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
                if (view != nullptr) {
                    const unsigned char* p = static_cast<const unsigned char*>(view);
                    got = (std::memcmp(p, "JYAIBF", 6) == 0)
                        && text_detail::read_u64_le(p + 6) == 512
                        && text_detail::read_u64_le(p + 14) == 16
                        && p[22] == 1 && p[23] == 8 && p[24] == 2 && p[25] == 64
                        && text_detail::read_u32_le(p + 26) == 2048
                        && text_detail::read_u64_le(p + 30) == 32000;
                }
            }
        }

        printf("[11/%d] 内存映射按偏移读回全部字段              %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        if (view != nullptr) {
            UnmapViewOfFile(view);
        }
        if (map_handle != nullptr) {
            CloseHandle(map_handle);
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
    }

    // ---------- 12. install_model() 正常装载 ----------
    {
        free_model_map();
        bool got = false;
        try {
            install_model(dir_ok);
            got = (model.h == 512 && model.l == 16 && model.mode == 1
                   && model.num_heads == 8 && model.num_kv_heads == 2 && model.head_dim == 64
                   && model.intermediate_size == 2048 && model.vocab_size == 32000
                   && model.model_map != nullptr);
            if (!got) {
                printf("       读回 h=%llu l=%llu mode=%u nh=%u kv=%u dim=%u is=%u vs=%llu\n",
                       (unsigned long long)model.h, (unsigned long long)model.l, (unsigned)model.mode,
                       (unsigned)model.num_heads, (unsigned)model.num_kv_heads, (unsigned)model.head_dim,
                       (unsigned)model.intermediate_size, (unsigned long long)model.vocab_size);
            }
        } catch (const std::exception& e) {
            printf("       ! install_model 抛异常: %s\n", e.what());
        }
        printf("[12/%d] install_model 装载后字段与模型一致      %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
        free_model_map();
    }

    // ---------- 13. install_model() 缺 model.bf ----------
    {
        fs::create_directories(dir_nobf, ec);
        write_model_info(dir_nobf, def);
        std::string msg;
        bool threw = false;
        try {
            install_model(dir_nobf);
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        } catch (...) {
            threw = true;
            msg = "(非标准异常)";
        }
        const bool got = threw && msg == "model.bf not found";
        printf("[13/%d] install_model 缺 model.bf -> 抛错       %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", threw ? msg.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
        free_model_map();
    }

    // ---------- 14. install_model() 魔数不对 ----------
    {
        fs::create_directories(dir_badmagic, ec);
        write_model_info(dir_badmagic, def);
        {
            std::ofstream out((dir_badmagic / "model.bf").string(), std::ios::binary);
            out.write("XXXXXX", 6);
            const char zeros[58] = {};
            out.write(zeros, sizeof(zeros));
        }
        std::string msg;
        bool threw = false;
        try {
            install_model(dir_badmagic);
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        } catch (...) {
            threw = true;
            msg = "(非标准异常)";
        }
        const bool got = threw && msg == "model.bf magic not match";
        printf("[14/%d] install_model 魔数错 -> 抛错            %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", threw ? msg.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
        free_model_map();
    }

    // ---------- 15. 没有 model_info.json ----------
    {
        fs::create_directories(dir_nojson, ec);
        const text_detail::call_result r = call_create_bfile(dir_nojson);
        const bool got = r.threw && r.message == "NO_JSONFILE";
        printf("[15/%d] 缺 model_info.json -> 抛 NO_JSONFILE    %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", r.threw ? r.message.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
        if (fs::exists(dir_nojson / "model.bf")) {
            printf("       ! 失败路径仍留下了 model.bf（%llu 字节）\n",
                   (unsigned long long)fs::file_size(dir_nojson / "model.bf", ec));
        }
    }

    // ---------- 16. layer_count 不是 16 的倍数 ----------
    {
        fs::create_directories(dir_badlayer, ec);
        text_detail::model_params p;
        p.layers = 12;
        write_model_info(dir_badlayer, p);
        const text_detail::call_result r = call_create_bfile(dir_badlayer);
        const bool got = r.threw && r.message == "NO16_LAYER";
        printf("[16/%d] layer_count=12 -> 抛 NO16_LAYER        %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", r.threw ? r.message.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
    }

    // ---------- 17. 边界值：所有字段一起换一轮 ----------
    {
        fs::create_directories(dir_edge, ec);
        text_detail::model_params p;
        p.hidden = 4096;
        p.layers = 32;
        p.mode = 255;
        p.num_heads = 32;
        p.num_kv_heads = 8;
        p.head_dim = 128;
        p.intermediate_size = 11008;
        p.vocab_size = 128256;
        write_model_info(dir_edge, p);
        call_create_bfile(dir_edge);
        const bf_head head = read_bf_head(dir_edge);
        const bool got = head.matches(p);
        printf("[17/%d] 边界值 4096/32/255/32/8/128/11008/128256 %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
    }

    // ---------- 诊断：越界值会被静默截断 ----------
    {
        fs::create_directories(dir_diag, ec);
        text_detail::model_params p;
        p.hidden = -1;
        p.mode = 300;
        write_model_info(dir_diag, p);
        call_create_bfile(dir_diag);
        const bf_head head = read_bf_head(dir_diag);
        printf("诊断: mode=300 写回 %u；hidden_size=-1 写回 %llu\n",
               (unsigned)head.mode, (unsigned long long)head.h);
    }

    // ---------- 诊断：model_info.json 存在但字段缺失 ----------
    {
        const fs::path dir_missing = root / "model_missingkey";
        fs::create_directories(dir_missing, ec);
        // 只写一个无关的键，不写 hidden_size / layer_count / mode
        json_lib bad;
        bad.SetJsonWay((dir_missing / "model_info.json").string());
        bad.WriteJsonKey("aaa", "1");
        const text_detail::call_result r = call_create_bfile(dir_missing);
        const bf_head head = read_bf_head(dir_missing);
        printf("诊断: 字段缺失时 -> 抛异常=%s，写出的 h=%llu l=%llu mode=%u\n",
               r.threw ? r.message.c_str() : "否",
               (unsigned long long)head.h, (unsigned long long)head.l, (unsigned)head.mode);
    }

    // ---------- 证据：model.bf 前 54 字节的十六进制 ----------
    {
        std::string bytes;
        if (text_detail::read_bytes((dir_ok / "model.bf").string(), bytes)) {
            printf("证据: model_ok/model.bf 前 54 字节 =");
            for (size_t i = 0; i < bytes.size() && i < 54; i++) {
                printf(" %02X", (unsigned char)bytes[i]);
            }
            printf("\n");
        }
    }

    printf("check_bfile: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_ffn() —— FFN(SwiGLU) 前向 + silu 测试
//
//  被测对象：
//    libs/model_libs/ffn/ffn.h     ffn()
//    libs/model_libs/silu/silu.h   silu()
//    libs/matmul_libs/matmul.h     matmul()（按 CPU 指令集分发）
//
//  形状约定（行主序，B 不转置）：
//    gate = x @ w1        w1: [hidden][intermediate]
//    up   = x @ w3        w3: [hidden][intermediate]
//    out  = (silu(gate) * up) @ w2   w2: [intermediate][hidden]
//    x/out: [seq][hidden]
//
//  注：ffn 内部的 matmul() 依赖 check_cpu.h 的全局 cpu_can；
//      没调过 GetCpuCan() 时它的成员（静态存储期）全为 false，会走朴素分支。
// ============================================================

namespace ffn_detail {

// 确定性伪随机，保证每次结果一致
inline float pseudo(int i) {
    return static_cast<float>((i * 37) % 19 - 9) / 8.0f;
}

// silu 的 double 参考实现
inline double silu_ref(double v) {
    return v / (1.0 + std::exp(-v));
}

// 手写 double 版 FFN，作为参考
inline void reference(const std::vector<float>& x,
                      const std::vector<float>& w1,
                      const std::vector<float>& w2,
                      const std::vector<float>& w3,
                      int seq, int hidden, int intermediate,
                      std::vector<float>& out) {
    std::vector<double> acc(static_cast<size_t>(seq) * hidden, 0.0);
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < intermediate; j++) {
            double g = 0.0;
            double u = 0.0;
            for (int k = 0; k < hidden; k++) {
                const double xv = x[static_cast<size_t>(i) * hidden + k];
                g += xv * w1[static_cast<size_t>(k) * intermediate + j];
                u += xv * w3[static_cast<size_t>(k) * intermediate + j];
            }
            const double h = silu_ref(g) * u;
            for (int o = 0; o < hidden; o++) {
                acc[static_cast<size_t>(i) * hidden + o] +=
                    h * w2[static_cast<size_t>(j) * hidden + o];
            }
        }
    }
    out.assign(static_cast<size_t>(seq) * hidden, 0.0f);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = static_cast<float>(acc[i]);
    }
}

// 相对误差（用 1+|want| 归一化，大小数值都合适）
inline float rel_err(float got, float want) {
    return std::fabs(got - want) / (1.0f + std::fabs(want));
}

// 造输入、跑一次 ffn，返回与参考的最大相对误差
inline float run_case(int seq, int hidden, int intermediate) {
    const size_t sx = static_cast<size_t>(seq) * hidden;
    const size_t s1 = static_cast<size_t>(hidden) * intermediate;
    const size_t s2 = static_cast<size_t>(intermediate) * hidden;

    std::vector<float> x(sx), w1(s1), w3(s1), w2(s2), got(sx, 0.0f);
    for (size_t i = 0; i < sx; i++) x[i] = pseudo(static_cast<int>(i));
    for (size_t i = 0; i < s1; i++) w1[i] = pseudo(static_cast<int>(i) + 3);
    for (size_t i = 0; i < s1; i++) w3[i] = pseudo(static_cast<int>(i) + 5);
    for (size_t i = 0; i < s2; i++) w2[i] = pseudo(static_cast<int>(i) + 7);

    std::vector<float> want;
    reference(x, w1, w2, w3, seq, hidden, intermediate, want);

    ffn(x.data(), w1.data(), w2.data(), w3.data(), got.data(), seq, hidden, intermediate);

    float worst = 0.0f;
    for (size_t i = 0; i < want.size() && i < got.size(); i++) {
        const float e = rel_err(got[i], want[i]);
        if (e > worst) worst = e;
    }
    return worst;
}

}  // namespace ffn_detail

bool check_ffn() {
    using ffn_detail::pseudo;
    using ffn_detail::rel_err;
    using ffn_detail::run_case;
    using ffn_detail::silu_ref;

    const int total = 8;
    int passed = 0;

    printf("==== FFN(SwiGLU) / silu 新功能测试 ====\n");

    // ffn 内部的 matmul() 按 cpu_can 分发，先确保探测过
    cpu_can.GetCpuCan();

    // ---------- 1~4. 与 double 参考比对（含退化形状） ----------
    struct shape_case {
        const char* name;
        int seq;
        int hidden;
        int inter;
    };
    const shape_case shapes[4] = {
        {"常规 seq=3 hidden=8 inter=16", 3, 8, 16},
        {"退化 seq=1", 1, 16, 24},
        {"退化 intermediate=1", 4, 8, 1},
        {"退化 hidden=1", 4, 1, 8},
    };
    for (int i = 0; i < 4; i++) {
        const float worst = run_case(shapes[i].seq, shapes[i].hidden, shapes[i].inter);
        const bool ok = (worst <= 1e-4f);
        printf("[%d/%d] %s  ->  最大相对误差 %.3g  %s\n",
               i + 1, total, shapes[i].name, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. silu 数值 ----------
    {
        float in[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float out[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        silu(in, out, 5);
        bool ok = true;
        for (int i = 0; i < 5; i++) {
            if (rel_err(out[i], static_cast<float>(silu_ref(in[i]))) > 1e-6f) {
                ok = false;
                break;
            }
        }
        printf("[5/%d] silu 数值 = x/(1+exp(-x))            %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 6. silu 原地调用（ffn 就是这么用的） ----------
    {
        float inplace[4] = {-1.5f, 0.25f, 2.0f, -0.75f};
        float src[4] = {-1.5f, 0.25f, 2.0f, -0.75f};
        float sep[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        silu(src, sep, 4);          // 分离缓冲
        silu(inplace, inplace, 4);  // 原地（ffn 依赖这种用法）
        bool ok = true;
        for (int i = 0; i < 4; i++) {
            if (inplace[i] != sep[i]) {
                ok = false;
                break;
            }
        }
        printf("[6/%d] silu 原地调用与分离调用等价          %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 7. out 缓冲区之外不被改写（前后哨兵） ----------
    {
        const int seq = 3, hidden = 8, inter = 16;
        const size_t sx = static_cast<size_t>(seq) * hidden;
        const size_t s1 = static_cast<size_t>(hidden) * inter;
        const size_t s2 = static_cast<size_t>(inter) * hidden;
        const float guard = -12345.0f;

        std::vector<float> x(sx), w1(s1), w3(s1), w2(s2);
        for (size_t i = 0; i < sx; i++) x[i] = pseudo(static_cast<int>(i));
        for (size_t i = 0; i < s1; i++) w1[i] = pseudo(static_cast<int>(i) + 3);
        for (size_t i = 0; i < s1; i++) w3[i] = pseudo(static_cast<int>(i) + 5);
        for (size_t i = 0; i < s2; i++) w2[i] = pseudo(static_cast<int>(i) + 7);

        std::vector<float> buf(sx + 8, guard);
        float* out = buf.data() + 4;
        ffn(x.data(), w1.data(), w2.data(), w3.data(), out, seq, hidden, inter);

        bool ok = true;
        for (size_t i = 0; i < 4; i++) {
            if (buf[i] != guard) {
                ok = false;
            }
            if (buf[4 + sx + i] != guard) {
                ok = false;
            }
        }
        printf("[7/%d] out 前后哨兵未被越界改写            %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 8. 朴素分支 vs 指令集分发分支 结果一致 ----------
    {
        const int seq = 4, hidden = 8, inter = 12;
        const size_t sx = static_cast<size_t>(seq) * hidden;
        const size_t s1 = static_cast<size_t>(hidden) * inter;
        const size_t s2 = static_cast<size_t>(inter) * hidden;

        std::vector<float> x(sx), w1(s1), w3(s1), w2(s2);
        std::vector<float> a(sx, 0.0f), b(sx, 0.0f);
        for (size_t i = 0; i < sx; i++) x[i] = pseudo(static_cast<int>(i));
        for (size_t i = 0; i < s1; i++) w1[i] = pseudo(static_cast<int>(i) + 3);
        for (size_t i = 0; i < s1; i++) w3[i] = pseudo(static_cast<int>(i) + 5);
        for (size_t i = 0; i < s2; i++) w2[i] = pseudo(static_cast<int>(i) + 7);

        const bool has_avx = cpu_can.avx;
        const bool has_fma = cpu_can.fma3;

        cpu_can.avx = false;    // 强制走 matmul_classic
        cpu_can.fma3 = false;
        ffn(x.data(), w1.data(), w2.data(), w3.data(), a.data(), seq, hidden, inter);

        cpu_can.GetCpuCan();    // 恢复真实探测结果
        ffn(x.data(), w1.data(), w2.data(), w3.data(), b.data(), seq, hidden, inter);

        float worst = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            const float e = rel_err(a[i], b[i]);
            if (e > worst) {
                worst = e;
            }
        }
        const char* backend = (has_avx && has_fma) ? "AVX+FMA"
                            : (has_fma ? "FMA" : (has_avx ? "AVX" : "朴素"));
        const bool ok = (worst <= 1e-4f);
        printf("[8/%d] 朴素 vs %s 结果一致 (误差 %.3g)   %s\n",
               total, backend, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    printf("check_ffn: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_rope() —— RoPE 旋转位置编码测试
//
//  被测对象：libs/model_libs/rope/rope.h  rope()（目前直接转发 rope_classic）
//
//  约定：
//    q          : [seq][num_heads][head_dim]
//    cos/sin 表 : [seq][head_dim/2]（第 pos 行的起点是 table + pos*half）
//    旋转对     : 相邻对 (2i, 2i+1) —— GPT-J 风格（不是 NeoX 的“前后半分开”风格）
// ============================================================

namespace rope_detail {

inline void make_q(std::vector<float>& q, int seq, int num_heads, int head_dim) {
    const size_t n = static_cast<size_t>(seq) * num_heads * head_dim;
    q.assign(n, 0.0f);
    for (size_t i = 0; i < n; i++) {
        q[i] = ffn_detail::pseudo(static_cast<int>(i));
    }
}

// 标准 RoPE 表：theta = pos * 10000^(-2i/d)
inline void make_tables(int seq, int head_dim,
                        std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int half = head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * half, 0.0f);
    sin_t.assign(static_cast<size_t>(seq) * half, 0.0f);
    for (int pos = 0; pos < seq; pos++) {
        for (int i = 0; i < half; i++) {
            const double theta = static_cast<double>(pos) *
                                 std::pow(10000.0, -2.0 * i / head_dim);
            cos_t[static_cast<size_t>(pos) * half + i] = static_cast<float>(std::cos(theta));
            sin_t[static_cast<size_t>(pos) * half + i] = static_cast<float>(std::sin(theta));
        }
    }
}

// 所有位置都用同一个 (cos, sin)——方便做手算验证
inline void make_const_tables(int seq, int head_dim, double c, double s,
                              std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int half = head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * half, static_cast<float>(c));
    sin_t.assign(static_cast<size_t>(seq) * half, static_cast<float>(s));
}

// double 参考实现
inline std::vector<float> reference(const std::vector<float>& q, int seq, int num_heads,
                                    int head_dim, const std::vector<float>& cos_t,
                                    const std::vector<float>& sin_t) {
    const int half = head_dim / 2;
    std::vector<float> out = q;
    for (int pos = 0; pos < seq; pos++) {
        for (int h = 0; h < num_heads; h++) {
            const size_t base = (static_cast<size_t>(pos) * num_heads + h) * head_dim;
            for (int i = 0; i < half; i++) {
                const double x0 = q[base + 2 * i];
                const double x1 = q[base + 2 * i + 1];
                const double c = cos_t[static_cast<size_t>(pos) * half + i];
                const double s = sin_t[static_cast<size_t>(pos) * half + i];
                out[base + 2 * i] = static_cast<float>(x0 * c - x1 * s);
                out[base + 2 * i + 1] = static_cast<float>(x0 * s + x1 * c);
            }
        }
    }
    return out;
}

inline float worst_err(const std::vector<float>& got, const std::vector<float>& want) {
    float worst = 0.0f;
    for (size_t i = 0; i < want.size() && i < got.size(); i++) {
        const float e = ffn_detail::rel_err(got[i], want[i]);
        if (e > worst) {
            worst = e;
        }
    }
    return worst;
}

}  // namespace rope_detail

bool check_rope() {
    using namespace rope_detail;

    const int total = 6;
    int passed = 0;

    printf("==== RoPE 新功能测试 ====\n");

    // ---------- 1. 常规形状 vs double 参考 ----------
    {
        const int seq = 3, heads = 2, dim = 8;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_tables(seq, dim, cos_t, sin_t);
        const std::vector<float> want = reference(q, seq, heads, dim, cos_t, sin_t);

        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-5f);
        printf("[1/%d] seq=3 heads=2 dim=8  ->  最大相对误差 %.3g   %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 2. 每个 head 的 2-范数守恒（旋转是正交变换，独立不变量） ----------
    {
        const int seq = 5, heads = 3, dim = 6;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_tables(seq, dim, cos_t, sin_t);
        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());

        float worst = 0.0f;
        for (int pos = 0; pos < seq; pos++) {
            for (int h = 0; h < heads; h++) {
                const size_t base = (static_cast<size_t>(pos) * heads + h) * dim;
                double n0 = 0.0, n1 = 0.0;
                for (int d = 0; d < dim; d++) {
                    n0 += static_cast<double>(q[base + d]) * q[base + d];
                    n1 += static_cast<double>(got[base + d]) * got[base + d];
                }
                const double r0 = std::sqrt(n0);
                const double r1 = std::sqrt(n1);
                const float e = static_cast<float>(std::fabs(r1 - r0) / (1e-9 + r0));
                if (e > worst) {
                    worst = e;
                }
            }
        }
        const bool ok = (worst <= 1e-5f);
        printf("[2/%d] 每个 head 的 2-范数守恒 (最大偏差 %.3g)   %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 3. cos=1 sin=0 表 -> q 完全不变 ----------
    {
        const int seq = 4, heads = 2, dim = 4;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_const_tables(seq, dim, 1.0, 0.0, cos_t, sin_t);
        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());

        bool ok = true;
        for (size_t i = 0; i < q.size(); i++) {
            if (got[i] != q[i]) {
                ok = false;
                break;
            }
        }
        printf("[3/%d] cos=1 sin=0 -> q 不变                   %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 4. cos=0 sin=1 表 -> (x0,x1) 应变成 (-x1,x0) ----------
    {
        const int seq = 2, heads = 2, dim = 4;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_const_tables(seq, dim, 0.0, 1.0, cos_t, sin_t);
        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());

        bool ok = true;
        for (int pos = 0; pos < seq && ok; pos++) {
            for (int h = 0; h < heads && ok; h++) {
                const size_t base = (static_cast<size_t>(pos) * heads + h) * dim;
                for (int i = 0; i < dim / 2; i++) {
                    const float want0 = -q[base + 2 * i + 1];
                    const float want1 = q[base + 2 * i];
                    if (got[base + 2 * i] != want0 || got[base + 2 * i + 1] != want1) {
                        ok = false;
                    }
                }
            }
        }
        printf("[4/%d] cos=0 sin=1 -> (x0,x1) 变 (-x1,x0)        %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. 哨兵：只改 q 自己的范围，不越界 ----------
    {
        const int seq = 3, heads = 2, dim = 4;
        const size_t n = static_cast<size_t>(seq) * heads * dim;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_tables(seq, dim, cos_t, sin_t);

        const float guard = -12345.0f;
        std::vector<float> buf(n + 8, guard);
        for (size_t i = 0; i < n; i++) {
            buf[4 + i] = q[i];
        }
        float* ptr = buf.data() + 4;
        rope(ptr, seq, heads, dim, cos_t.data(), sin_t.data());

        bool ok = true;
        for (size_t i = 0; i < 4; i++) {
            if (buf[i] != guard || buf[4 + n + i] != guard) {
                ok = false;
            }
        }
        printf("[5/%d] 哨兵未被越界改写                     %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 6. 最小形状：seq=1 heads=1 dim=2 ----------
    {
        const int seq = 1, heads = 1, dim = 2;
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_tables(seq, dim, cos_t, sin_t);
        const std::vector<float> want = reference(q, seq, heads, dim, cos_t, sin_t);
        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-5f);
        printf("[6/%d] 最小形状 seq=1 heads=1 dim=2 (误差 %.3g)   %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 诊断：奇数 head_dim ----------
    {
        const int seq = 1, heads = 1, dim = 3;   // half = 1，第 3 个元素不参与旋转
        std::vector<float> q, cos_t, sin_t;
        make_q(q, seq, heads, dim);
        make_const_tables(seq, dim, 0.0, 1.0, cos_t, sin_t);
        std::vector<float> got = q;
        rope(got.data(), seq, heads, dim, cos_t.data(), sin_t.data());
        printf("诊断: head_dim=3（奇数）时 下标2 旋转前=%.3f 旋转后=%.3f（未参与旋转）\n",
               static_cast<double>(q[2]), static_cast<double>(got[2]));
    }

    printf("check_rope: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_rmsnorm() —— RMSNorm 测试
//
//  被测对象：libs/model_libs/rmsnorm/rmsnorm.h  rmsnorm()
//  参考实现：rmsnorm_classic.h 里的 rmsnorm_classic()
//  公式：y[i] = x[i] * (1/sqrt(mean(x^2) + eps)) * weight[i]
//  注：eps 的形参类型是 int，不是 float（下面有诊断）
// ============================================================

namespace rms_detail {

inline std::vector<float> reference(const std::vector<float>& x, const std::vector<float>& w,
                                    int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        ss += static_cast<double>(x[i]) * x[i];
    }
    const double scale = 1.0 / std::sqrt(ss / n + static_cast<double>(eps));
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; i++) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * scale * w[i]);
    }
    return out;
}

inline float worst_err(const std::vector<float>& got, const std::vector<float>& want) {
    float worst = 0.0f;
    for (size_t i = 0; i < want.size() && i < got.size(); i++) {
        const float e = ffn_detail::rel_err(got[i], want[i]);
        if (e > worst) {
            worst = e;
        }
    }
    return worst;
}

}  // namespace rms_detail

bool check_rmsnorm() {
    using namespace rms_detail;

    const int total = 6;
    int passed = 0;

    printf("==== RMSNorm 新功能测试 ====\n");

    // ---------- 1. 常规 n=8 eps=0 vs double 参考 ----------
    {
        const int n = 8;
        std::vector<float> x(n), w(n), got(n, 0.0f);
        for (int i = 0; i < n; i++) {
            x[i] = ffn_detail::pseudo(i) * 3.0f;
            w[i] = ffn_detail::pseudo(i + 11) + 1.5f;
        }
        const std::vector<float> want = reference(x, w, n, 0.0f);
        rmsnorm(x.data(), got.data(), w.data(), n, 0.0f);

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-5f);
        printf("[1/%d] n=8 eps=0 与参考一致 (误差 %.3g)        %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 2. w=1 eps=0 时输出 RMS 应恰好为 1（独立不变量） ----------
    {
        const int n = 16;
        std::vector<float> x(n), w(n, 1.0f), got(n, 0.0f);
        for (int i = 0; i < n; i++) {
            x[i] = ffn_detail::pseudo(i) * 5.0f + 0.3f;
        }
        rmsnorm(x.data(), got.data(), w.data(), n, 0.0f);

        double ss = 0.0;
        for (int i = 0; i < n; i++) {
            ss += static_cast<double>(got[i]) * got[i];
        }
        const double rms = std::sqrt(ss / n);
        const bool ok = (std::fabs(rms - 1.0) <= 1e-4);
        printf("[2/%d] w=1 eps=0 -> 输出 RMS = %.6f (期望 1)   %s\n",
               total, rms, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 3. 退化 n=1 ----------
    {
        const int n = 1;
        std::vector<float> x(1), w(1), got(1, 0.0f);
        x[0] = -0.75f;
        w[0] = 2.0f;
        const std::vector<float> want = reference(x, w, n, 0.0f);
        rmsnorm(x.data(), got.data(), w.data(), n, 0.0f);

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-5f);
        printf("[3/%d] 退化 n=1 (值 %.4f, 期望 %.4f)           %s\n",
               total, static_cast<double>(got[0]), static_cast<double>(want[0]),
               ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 4. 哨兵：y 之外不被改写 ----------
    {
        const int n = 6;
        std::vector<float> x(n), w(n);
        for (int i = 0; i < n; i++) {
            x[i] = ffn_detail::pseudo(i) + 0.5f;
            w[i] = 1.0f;
        }
        const float guard = -12345.0f;
        std::vector<float> buf(n + 8, guard);
        rmsnorm(x.data(), buf.data() + 4, w.data(), n, 0.0f);

        bool ok = true;
        for (int i = 0; i < 4; i++) {
            if (buf[i] != guard || buf[4 + n + i] != guard) {
                ok = false;
            }
        }
        printf("[4/%d] 哨兵未被越界改写                     %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. 大 n + 大数值：与 double 参考的一致性 ----------
    {
        const int n = 1024;
        std::vector<float> x(n), w(n), got(n, 0.0f);
        for (int i = 0; i < n; i++) {
            x[i] = ffn_detail::pseudo(i) * 1000.0f;
            w[i] = 1.0f;
        }
        const std::vector<float> want = reference(x, w, n, 0.0f);
        rmsnorm(x.data(), got.data(), w.data(), n, 0.0f);

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-3f);
        printf("[5/%d] n=1024 大数值与 double 参考一致 (误差 %.3g)   %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 6. eps 改成 float 后，标准 eps=1e-5 能用了 ----------
    {
        const int n = 8;
        std::vector<float> x(n), w(n), got(n, 0.0f);
        for (int i = 0; i < n; i++) {
            x[i] = ffn_detail::pseudo(i) * 2.0f;
            w[i] = 1.0f;
        }
        const float eps = 1e-5f;
        const std::vector<float> want = reference(x, w, n, eps);
        rmsnorm(x.data(), got.data(), w.data(), n, eps);

        const float worst = worst_err(got, want);
        const bool ok = (worst <= 1e-5f);
        printf("[6/%d] eps=1e-5（float 参数）与参考一致 (误差 %.3g)   %s\n",
               total, static_cast<double>(worst), ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 诊断：全零输入不再出 nan ----------
    {
        const int n = 4;
        std::vector<float> x(n, 0.0f), w(n, 1.0f), got(n, 0.0f);
        rmsnorm(x.data(), got.data(), w.data(), n, 0.0f);
        const bool bad = std::isnan(got[0]) || std::isinf(got[0]);
        printf("诊断: x 全零且 eps=0 -> y[0]=%.3f（%s）\n",
               static_cast<double>(got[0]), bad ? "仍出 nan/inf" : "已兜住，不再出 nan");
    }

    printf("check_rmsnorm: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_attention() —— 因果 GQA 注意力测试
//
//  被测对象：libs/model_libs/attention/attention.h  attention()
//
//  形状约定：
//    Q   : [seq][num_heads][head_dim]
//    K/V : [seq][num_kv_heads][head_dim]
//    out : [seq][num_heads][head_dim]
//    因果（j <= i）、GQA（num_heads / num_kv_heads 分组）
// ============================================================

namespace attn_detail {

inline void fill_random(std::vector<float>& v, int seed) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ffn_detail::pseudo(static_cast<int>(i) + seed);
    }
}

// double 参考实现（同样因果 + GQA + 减最大值 softmax）
inline std::vector<float> reference(const std::vector<float>& Q, const std::vector<float>& K,
                                    const std::vector<float>& V, int seq, int num_heads,
                                    int num_kv_heads, int head_dim) {
    const int group = num_heads / num_kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<float> out(static_cast<size_t>(seq) * num_heads * head_dim, 0.0f);
    std::vector<double> sc(static_cast<size_t>(seq), 0.0);
    for (int h = 0; h < num_heads; h++) {
        const int kv_h = h / group;
        for (int i = 0; i < seq; i++) {
            const size_t qb = (static_cast<size_t>(i) * num_heads + h) * head_dim;
            double mx = -1e300;
            for (int j = 0; j <= i; j++) {
                const size_t kb = (static_cast<size_t>(j) * num_kv_heads + kv_h) * head_dim;
                double dt = 0.0;
                for (int d = 0; d < head_dim; d++) {
                    dt += static_cast<double>(Q[qb + d]) * K[kb + d];
                }
                dt *= scale;
                sc[j] = dt;
                if (dt > mx) {
                    mx = dt;
                }
            }
            double sum = 0.0;
            for (int j = 0; j <= i; j++) {
                sc[j] = std::exp(sc[j] - mx);
                sum += sc[j];
            }
            const size_t ob = (static_cast<size_t>(i) * num_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (int j = 0; j <= i; j++) {
                    const size_t vb = (static_cast<size_t>(j) * num_kv_heads + kv_h) * head_dim;
                    acc += sc[j] / sum * V[vb + d];
                }
                out[ob + d] = static_cast<float>(acc);
            }
        }
    }
    return out;
}

inline float worst_err(const std::vector<float>& got, const std::vector<float>& want) {
    float worst = 0.0f;
    for (size_t i = 0; i < want.size() && i < got.size(); i++) {
        const float e = ffn_detail::rel_err(got[i], want[i]);
        if (e > worst) {
            worst = e;
        }
    }
    return worst;
}

// 跑一次 attention，返回与 double 参考的最大相对误差
inline float run_case(int seq, int num_heads, int num_kv_heads, int head_dim) {
    const size_t sq = static_cast<size_t>(seq) * num_heads * head_dim;
    const size_t skv = static_cast<size_t>(seq) * num_kv_heads * head_dim;
    std::vector<float> Q(sq), K(skv), V(skv), got(sq, 0.0f);
    fill_random(Q, 1);
    fill_random(K, 2);
    fill_random(V, 3);
    const std::vector<float> want = reference(Q, K, V, seq, num_heads, num_kv_heads, head_dim);
    attention(Q.data(), K.data(), V.data(), got.data(), seq, num_heads, num_kv_heads, head_dim);
    return worst_err(got, want);
}

}  // namespace attn_detail

bool check_attention() {
    using namespace attn_detail;

    const int total = 7;
    int passed = 0;

    printf("==== Attention 新功能测试 ====\n");

    // ---------- 1~3. MHA / GQA / MQA 与 double 参考对比 ----------
    {
        struct cfg { const char* name; int heads; int kv; };
        const cfg cfgs[3] = {
            {"MHA heads=4 kv=4", 4, 4},
            {"GQA heads=4 kv=2", 4, 2},
            {"MQA heads=4 kv=1", 4, 1},
        };
        for (int c = 0; c < 3; c++) {
            const float worst = run_case(5, cfgs[c].heads, cfgs[c].kv, 8);
            const bool ok = (worst <= 1e-4f);
            printf("[%d/%d] %s (误差 %.3g)   %s\n",
                   c + 1, total, cfgs[c].name, static_cast<double>(worst), ok ? "OK" : "FAIL");
            if (ok) {
                passed++;
            }
        }
    }

    // ---------- 4. seq=1：只有一个 key，输出必须等于 V[0] ----------
    {
        const int seq = 1, heads = 2, kv = 2, dim = 4;
        const size_t sq = static_cast<size_t>(seq) * heads * dim;
        const size_t skv = static_cast<size_t>(seq) * kv * dim;
        std::vector<float> Q(sq), K(skv), V(skv), got(sq, 0.0f);
        fill_random(Q, 1);
        fill_random(K, 2);
        fill_random(V, 3);
        attention(Q.data(), K.data(), V.data(), got.data(), seq, heads, kv, dim);

        bool ok = true;
        for (int h = 0; h < heads; h++) {
            for (int d = 0; d < dim; d++) {
                if (got[h * dim + d] != V[h * dim + d]) {
                    ok = false;
                }
            }
        }
        printf("[4/%d] seq=1 -> out 等于 V[0]                 %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. 因果性：改后面的 V 行不影响 out[0] ----------
    {
        const int seq = 4, heads = 2, kv = 2, dim = 4;
        const size_t sq = static_cast<size_t>(seq) * heads * dim;
        const size_t skv = static_cast<size_t>(seq) * kv * dim;
        std::vector<float> Q(sq), K(skv), V(skv), a(sq, 0.0f), b(sq, 0.0f);
        fill_random(Q, 1);
        fill_random(K, 2);
        fill_random(V, 3);
        attention(Q.data(), K.data(), V.data(), a.data(), seq, heads, kv, dim);

        for (size_t i = skv / seq; i < skv; i++) {   // 把 j>=1 的 V 全改掉
            V[i] += 100.0f;
        }
        attention(Q.data(), K.data(), V.data(), b.data(), seq, heads, kv, dim);

        bool ok = true;
        for (int h = 0; h < heads; h++) {
            for (int d = 0; d < dim; d++) {
                if (a[h * dim + d] != b[h * dim + d]) {
                    ok = false;
                }
            }
        }
        printf("[5/%d] 因果性：改 j>=1 的 V 不影响 out[0]     %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 6. 所有 V 行相同 -> out 必须等于该值（权重和为 1） ----------
    {
        const int seq = 3, heads = 2, kv = 1, dim = 4;
        const size_t sq = static_cast<size_t>(seq) * heads * dim;
        const size_t skv = static_cast<size_t>(seq) * kv * dim;
        std::vector<float> Q(sq), K(skv), V(skv), got(sq, 0.0f);
        fill_random(Q, 1);
        fill_random(K, 2);
        const float c0 = 0.75f, c1 = -1.25f;
        for (int j = 0; j < seq; j++) {
            V[j * dim + 0] = c0;
            V[j * dim + 1] = c1;
            V[j * dim + 2] = c0;
            V[j * dim + 3] = c1;
        }
        attention(Q.data(), K.data(), V.data(), got.data(), seq, heads, kv, dim);

        bool ok = true;
        for (size_t i = 0; i < got.size(); i++) {
            const float want = (i % 2 == 0) ? c0 : c1;
            if (std::fabs(got[i] - want) > 1e-5f) {
                ok = false;
            }
        }
        printf("[6/%d] V 全同 -> out 等于该值（权重和=1）      %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 7. 数值稳定：logits 极大时不出 nan/inf ----------
    {
        const int seq = 3, heads = 1, kv = 1, dim = 16;
        const size_t sq = static_cast<size_t>(seq) * heads * dim;
        const size_t skv = static_cast<size_t>(seq) * kv * dim;
        std::vector<float> Q(sq), K(skv), V(skv), got(sq, 0.0f);
        for (size_t i = 0; i < sq; i++) Q[i] = 5000.0f + static_cast<float>(i % 7);
        for (size_t i = 0; i < skv; i++) K[i] = 5000.0f + static_cast<float>(i % 5);
        for (size_t i = 0; i < skv; i++) V[i] = 1.0f;
        attention(Q.data(), K.data(), V.data(), got.data(), seq, heads, kv, dim);

        bool ok = true;
        for (size_t i = 0; i < got.size(); i++) {
            if (std::isnan(got[i]) || std::isinf(got[i])) {
                ok = false;
            }
        }
        printf("[7/%d] logits≈1e8 时无 nan/inf（有减最大值）    %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 诊断：num_heads 不能被 num_kv_heads 整除 ----------
    {
        const int seq = 1, heads = 3, kv = 2, dim = 2;
        std::vector<float> Q(static_cast<size_t>(seq) * heads * dim, 1.0f);
        std::vector<float> K(static_cast<size_t>(seq) * kv * dim, 1.0f);
        std::vector<float> V(static_cast<size_t>(seq) * kv * dim, 1.0f);
        std::vector<float> got(static_cast<size_t>(seq) * heads * dim, 0.0f);
        try {
            attention(Q.data(), K.data(), V.data(), got.data(), seq, heads, kv, dim);
            printf("诊断: heads=3 kv=2 -> 没拊异常，out[0]=%.1f\n",
                   static_cast<double>(got[0]));
        } catch (const std::exception& e) {
            printf("诊断: heads=3 kv=2 -> 已拦截：%s\n", e.what());
        }
    }

    printf("check_attention: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_transformer_layer() —— 整层前向端到端测试
//
//  被测对象：libs/model_libs/transformer_layer.h  transformer_layer()
//
//  流程：RMSNorm -> QKV 投影 -> RoPE(Q,K) -> Attention
//        -> 输出投影 -> 残差 -> RMSNorm -> FFN -> 残差
//
//  形状（行主序）：
//    X          [seq][hidden]
//    Wq         [hidden][num_heads*head_dim]
//    Wk / Wv    [hidden][num_kv_heads*head_dim]
//    Wo         [num_heads*head_dim][hidden]
//    W1 / W3    [hidden][intermediate]
//    W2         [intermediate][hidden]
//     cos/sin 表 [seq][head_dim/2]
//
//  参考实现由各模块自己的 double 参考“拼”成（逐步之间用 float 承载），
//  所以它验证的是**接线**（形状 / 顺序 / 残差位置），
//  模块自身的数值精度已由各自的测试覆盖。
// ============================================================

namespace layer_detail {

// double 版矩阵乘：C[m][n] = A[m][k] * B[k][n]
inline void dmatmul(int m, int n, int k, const std::vector<float>& A,
                    const std::vector<float>& B, std::vector<float>& C) {
    C.assign(static_cast<size_t>(m) * n, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int t = 0; t < k; t++) {
                s += static_cast<double>(A[static_cast<size_t>(i) * k + t]) *
                     B[static_cast<size_t>(t) * n + j];
            }
            C[static_cast<size_t>(i) * n + j] = static_cast<float>(s);
        }
    }
}

// 逐行 RMSNorm（用 rms_detail 的 double 参考）
inline void rms_rows(const std::vector<float>& src, const std::vector<float>& w,
                     int seq, int n, float eps, std::vector<float>& dst) {
    dst.assign(static_cast<size_t>(seq) * n, 0.0f);
    for (int i = 0; i < seq; i++) {
        std::vector<float> row(src.begin() + static_cast<size_t>(i) * n,
                               src.begin() + static_cast<size_t>(i + 1) * n);
        const std::vector<float> out = rms_detail::reference(row, w, n, eps);
        for (int d = 0; d < n; d++) {
            dst[static_cast<size_t>(i) * n + d] = out[d];
        }
    }
}

// 整层 double 参考
inline void reference(const std::vector<float>& X0,
                      const std::vector<float>& rms1,
                      const std::vector<float>& Wq, const std::vector<float>& Wk,
                      const std::vector<float>& Wv, const std::vector<float>& Wo,
                      const std::vector<float>& cos_t, const std::vector<float>& sin_t,
                      const std::vector<float>& rms2,
                      const std::vector<float>& W1, const std::vector<float>& W2,
                      const std::vector<float>& W3,
                      int seq, int hidden, int num_heads, int num_kv_heads,
                      int head_dim, int intermediate, float eps,
                      std::vector<float>& out) {
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    std::vector<float> X(X0.begin(), X0.begin() + static_cast<size_t>(seq) * hidden);
    std::vector<float> xn;

    rms_rows(X, rms1, seq, hidden, eps, xn);                 // 1
    std::vector<float> Q, K, V;
    dmatmul(seq, q_dim, hidden, xn, Wq, Q);                  // 2
    dmatmul(seq, kv_dim, hidden, xn, Wk, K);
    dmatmul(seq, kv_dim, hidden, xn, Wv, V);
    Q = rope_detail::reference(Q, seq, num_heads, head_dim, cos_t, sin_t);       // 3
    K = rope_detail::reference(K, seq, num_kv_heads, head_dim, cos_t, sin_t);
    const std::vector<float> attn =
        attn_detail::reference(Q, K, V, seq, num_heads, num_kv_heads, head_dim); // 4
    std::vector<float> proj;
    dmatmul(seq, hidden, q_dim, attn, Wo, proj);             // 5
    for (size_t i = 0; i < X.size(); i++) {
        X[i] += proj[i];                                     // 6
    }
    rms_rows(X, rms2, seq, hidden, eps, xn);                 // 7
    std::vector<float> ffn_out;
    ffn_detail::reference(xn, W1, W2, W3, seq, hidden, intermediate, ffn_out);  // 8
    for (size_t i = 0; i < X.size(); i++) {
        X[i] += ffn_out[i];                                  // 9
    }
    out = X;
}

// 重量级造数据：填一层需要的全部权重
struct layer_weights {
    int seq, hidden, num_heads, num_kv_heads, head_dim, intermediate;
    std::vector<float> X, rms1, Wq, Wk, Wv, Wo, cos_t, sin_t, rms2, W1, W2, W3;

    void build(int s, int h, int nh, int nkv, int hd, int inter, bool random_attn = true) {
        seq = s; hidden = h; num_heads = nh; num_kv_heads = nkv;
        head_dim = hd; intermediate = inter;
        const int q_dim = nh * hd;
        const int kv_dim = nkv * hd;
        X.assign(static_cast<size_t>(s) * h, 0.0f);
        rms1.assign(h, 1.0f);
        rms2.assign(h, 1.0f);
        Wq.assign(static_cast<size_t>(h) * q_dim, 0.0f);
        Wk.assign(static_cast<size_t>(h) * kv_dim, 0.0f);
        Wv.assign(static_cast<size_t>(h) * kv_dim, 0.0f);
        Wo.assign(static_cast<size_t>(q_dim) * h, 0.0f);
        W1.assign(static_cast<size_t>(h) * inter, 0.0f);
        W2.assign(static_cast<size_t>(inter) * h, 0.0f);
        W3.assign(static_cast<size_t>(h) * inter, 0.0f);
        for (size_t i = 0; i < X.size(); i++) X[i] = ffn_detail::pseudo(static_cast<int>(i)) * 2.0f;
        for (int i = 0; i < h; i++) {
            rms1[i] = 1.0f + ffn_detail::pseudo(i + 1) * 0.25f;
            rms2[i] = 1.0f + ffn_detail::pseudo(i + 2) * 0.25f;
        }
        if (random_attn) {
            for (size_t i = 0; i < Wq.size(); i++) Wq[i] = ffn_detail::pseudo(static_cast<int>(i) + 3) * 0.5f;
            for (size_t i = 0; i < Wk.size(); i++) Wk[i] = ffn_detail::pseudo(static_cast<int>(i) + 5) * 0.5f;
            for (size_t i = 0; i < Wv.size(); i++) Wv[i] = ffn_detail::pseudo(static_cast<int>(i) + 7) * 0.5f;
            for (size_t i = 0; i < Wo.size(); i++) Wo[i] = ffn_detail::pseudo(static_cast<int>(i) + 11) * 0.5f;
        }
        for (size_t i = 0; i < W1.size(); i++) W1[i] = ffn_detail::pseudo(static_cast<int>(i) + 13) * 0.5f;
        for (size_t i = 0; i < W2.size(); i++) W2[i] = ffn_detail::pseudo(static_cast<int>(i) + 17) * 0.5f;
        for (size_t i = 0; i < W3.size(); i++) W3[i] = ffn_detail::pseudo(static_cast<int>(i) + 19) * 0.5f;
        rope_detail::make_tables(s, hd, cos_t, sin_t);
    }

    void run(float* x, float eps = 1e-5f) const {
        transformer_layer(x, rms1.data(), Wq.data(), Wk.data(), Wv.data(), Wo.data(),
                          cos_t.data(), sin_t.data(), rms2.data(), W1.data(), W2.data(),
                          W3.data(), seq, hidden, num_heads, num_kv_heads, head_dim,
                          intermediate, eps);
    }
};

// 跑一层，返回与参考的最大相对误差
inline float run_case(int seq, int hidden, int num_heads, int num_kv_heads,
                      int head_dim, int intermediate, float eps) {
    layer_weights w;
    w.build(seq, hidden, num_heads, num_kv_heads, head_dim, intermediate);

    std::vector<float> want;
    reference(w.X, w.rms1, w.Wq, w.Wk, w.Wv, w.Wo, w.cos_t, w.sin_t, w.rms2,
              w.W1, w.W2, w.W3, seq, hidden, num_heads, num_kv_heads, head_dim,
              intermediate, eps, want);

    std::vector<float> got = w.X;
    w.run(got.data(), eps);
    return rms_detail::worst_err(got, want);
}

}  // namespace layer_detail

bool check_transformer_layer() {
    using namespace layer_detail;

    const int total = 7;
    int passed = 0;

    printf("==== Transformer 整层测试 ====\n");

    // ---------- 1~3. 端到端 vs double 参考（MHA / GQA / seq=1） ----------
    {
        struct cfg { const char* name; int seq, hidden, heads, kv, dim, inter; };
        const cfg cases[3] = {
            {"MHA seq=3 hidden=16 heads=4 kv=4", 3, 16, 4, 4, 4, 32},
            {"GQA seq=3 hidden=16 heads=4 kv=2", 3, 16, 4, 2, 4, 32},
            {"seq=1 退化", 1, 8, 2, 2, 4, 16},
        };
        for (int c = 0; c < 3; c++) {
            const float e = run_case(cases[c].seq, cases[c].hidden, cases[c].heads,
                                     cases[c].kv, cases[c].dim, cases[c].inter, 1e-5f);
            const bool ok = (e <= 1e-3f);
            printf("[%d/%d] %s (误差 %.3g)   %s\n",
                   c + 1, total, cases[c].name, static_cast<double>(e), ok ? "OK" : "FAIL");
            if (ok) {
                passed++;
            }
        }
    }

    // ---------- 4. 全零权重 -> X 完全不变（残差恒等，且不会出 nan） ----------
    {
        layer_weights w;
        w.build(3, 8, 2, 2, 4, 16);
        std::fill(w.Wq.begin(), w.Wq.end(), 0.0f);
        std::fill(w.Wk.begin(), w.Wk.end(), 0.0f);
        std::fill(w.Wv.begin(), w.Wv.end(), 0.0f);
        std::fill(w.Wo.begin(), w.Wo.end(), 0.0f);
        std::fill(w.W1.begin(), w.W1.end(), 0.0f);
        std::fill(w.W2.begin(), w.W2.end(), 0.0f);
        std::fill(w.W3.begin(), w.W3.end(), 0.0f);
        std::vector<float> got = w.X;
        w.run(got.data());

        bool ok = true;
        for (size_t i = 0; i < got.size(); i++) {
            if (got[i] != w.X[i]) {
                ok = false;
            }
        }
        printf("[4/%d] 全零权重 -> X 完全不变                  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. 只留 FFN 分支：输出应恰好是 X + ffn(rmsnorm(X)) ----------
    {
        const float eps = 1e-5f;
        layer_weights w;
        w.build(2, 8, 2, 1, 4, 16);
        // QKV/Wo 全 0 -> 注意力分支贡献 0，只剩残差 + FFN
        std::fill(w.Wq.begin(), w.Wq.end(), 0.0f);
        std::fill(w.Wk.begin(), w.Wk.end(), 0.0f);
        std::fill(w.Wv.begin(), w.Wv.end(), 0.0f);
        std::fill(w.Wo.begin(), w.Wo.end(), 0.0f);

        std::vector<float> got = w.X;
        w.run(got.data(), eps);

        std::vector<float> norm(got.size(), 0.0f);
        for (int i = 0; i < w.seq; i++) {
            rmsnorm(w.X.data() + static_cast<size_t>(i) * w.hidden,
                    norm.data() + static_cast<size_t>(i) * w.hidden,
                    w.rms2.data(), w.hidden, eps);
        }
        std::vector<float> ffn_out;
        ffn_detail::reference(norm, w.W1, w.W2, w.W3, w.seq, w.hidden, w.intermediate, ffn_out);

        bool ok = true;
        for (size_t i = 0; i < got.size(); i++) {
            if (ffn_detail::rel_err(got[i], w.X[i] + ffn_out[i]) > 1e-4f) {
                ok = false;
            }
        }
        printf("[5/%d] 只留 FFN 分支 -> X + ffn(rmsnorm(X))    %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 6. 哨兵：X 之外不被越界改写 ----------
    {
        layer_weights w;
        w.build(3, 8, 2, 2, 4, 16);
        const float guard = -12345.0f;
        const size_t n = static_cast<size_t>(w.seq) * w.hidden;
        std::vector<float> buf(n + 8, guard);
        float* xp = buf.data() + 4;
        for (size_t i = 0; i < n; i++) {
            xp[i] = w.X[i];
        }
        w.run(xp);

        bool ok = true;
        for (size_t i = 0; i < 4; i++) {
            if (buf[i] != guard || buf[4 + n + i] != guard) {
                ok = false;
            }
        }
        printf("[6/%d] 哨兵未被越界改写                     %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 7. 确实做了计算（防空实现）+ 输出有限 ----------
    {
        layer_weights w;
        w.build(3, 16, 4, 2, 4, 32);
        std::vector<float> got = w.X;
        w.run(got.data());

        bool changed = false;
        bool finite = true;
        for (size_t i = 0; i < got.size(); i++) {
            if (std::isnan(got[i]) || std::isinf(got[i])) {
                finite = false;
            }
            if (got[i] != w.X[i]) {
                changed = true;
            }
        }
        const bool ok = changed && finite;
        printf("[7/%d] 输出有限且与输入不同（非空实现）        %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    printf("check_transformer_layer: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_generate() —— 自回归生成
//
//  被测对象：libs/model_libs/generate.h  generate()
//
//  每步：embed 查表 -> 整层 forward（含最终 RMSNorm）-> 最后一行 * lm_head
//        -> argmax / 温度采样 -> 追加 token
//
//  参考实现：用 layer_detail / rms_detail 的 double 参考把同样一步重做一遍
// ============================================================

namespace gen_detail {

struct gen_weights {
    int vocab, hidden, num_heads, num_kv_heads, head_dim, intermediate, num_layers, max_seq;
    std::vector<float> embed, lm_head, final_rms, cos_t, sin_t;
    std::vector<std::vector<float>> rms1, Wq, Wk, Wv, Wo, rms2, W1, W2, W3;
    std::vector<LayerWeights> layers;

    void build(int v, int h, int nh, int nkv, int hd, int inter, int nl, int mseq, bool zero_lm) {
        vocab = v; hidden = h; num_heads = nh; num_kv_heads = nkv; head_dim = hd;
        intermediate = inter; num_layers = nl; max_seq = mseq;
        const int qdim = nh * hd;
        const int kvdim = nkv * hd;
        rms1.clear(); rms2.clear(); Wq.clear(); Wk.clear(); Wv.clear(); Wo.clear();
        W1.clear(); W2.clear(); W3.clear(); layers.clear();
        for (int l = 0; l < nl; l++) {
            rms1.push_back(std::vector<float>(h, 1.0f));
            rms2.push_back(std::vector<float>(h, 1.0f));
            Wq.push_back(std::vector<float>(static_cast<size_t>(h) * qdim));
            Wk.push_back(std::vector<float>(static_cast<size_t>(h) * kvdim));
            Wv.push_back(std::vector<float>(static_cast<size_t>(h) * kvdim));
            Wo.push_back(std::vector<float>(static_cast<size_t>(qdim) * h));
            W1.push_back(std::vector<float>(static_cast<size_t>(h) * inter));
            W2.push_back(std::vector<float>(static_cast<size_t>(inter) * h));
            W3.push_back(std::vector<float>(static_cast<size_t>(h) * inter));
        }
        for (int l = 0; l < nl; l++) {
            for (size_t i = 0; i < rms1[l].size(); i++) rms1[l][i] = 1.0f + ffn_detail::pseudo(static_cast<int>(i) + 3 * l) * 0.25f;
            for (size_t i = 0; i < rms2[l].size(); i++) rms2[l][i] = 1.0f + ffn_detail::pseudo(static_cast<int>(i) + 5 * l) * 0.25f;
            for (size_t i = 0; i < Wq[l].size(); i++) Wq[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 7 * l) * 0.5f;
            for (size_t i = 0; i < Wk[l].size(); i++) Wk[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 11 * l) * 0.5f;
            for (size_t i = 0; i < Wv[l].size(); i++) Wv[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 13 * l) * 0.5f;
            for (size_t i = 0; i < Wo[l].size(); i++) Wo[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 17 * l) * 0.5f;
            for (size_t i = 0; i < W1[l].size(); i++) W1[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 19 * l) * 0.5f;
            for (size_t i = 0; i < W2[l].size(); i++) W2[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 23 * l) * 0.5f;
            for (size_t i = 0; i < W3[l].size(); i++) W3[l][i] = ffn_detail::pseudo(static_cast<int>(i) + 29 * l) * 0.5f;
        }
        embed.assign(static_cast<size_t>(v) * h, 0.0f);
        for (size_t i = 0; i < embed.size(); i++) embed[i] = ffn_detail::pseudo(static_cast<int>(i) + 101) * 0.5f;
        lm_head.assign(static_cast<size_t>(h) * v, 0.0f);
        if (!zero_lm) {
            for (size_t i = 0; i < lm_head.size(); i++) lm_head[i] = ffn_detail::pseudo(static_cast<int>(i) + 211) * 0.5f;
        }
        final_rms.assign(h, 1.0f);
        for (size_t i = 0; i < final_rms.size(); i++) final_rms[i] = 1.0f + ffn_detail::pseudo(static_cast<int>(i) + 307) * 0.25f;
        rope_detail::make_tables(mseq, hd, cos_t, sin_t);
        layers.resize(nl);
        for (int l = 0; l < nl; l++) {
            layers[l].rms1_weight = rms1[l].data();
            layers[l].Wq = Wq[l].data();
            layers[l].Wk = Wk[l].data();
            layers[l].Wv = Wv[l].data();
            layers[l].Wo = Wo[l].data();
            layers[l].rms2_weight = rms2[l].data();
            layers[l].W1 = W1[l].data();
            layers[l].W2 = W2[l].data();
            layers[l].W3 = W3[l].data();
        }
    }

    int run(const int* prompt, int prompt_len, int n_new, float temp, int* out) const {
        return generate(prompt, prompt_len, nullptr, 0, embed.data(), lm_head.data(),
                        layers.data(), num_layers, final_rms.data(), cos_t.data(),
                        sin_t.data(), max_seq, n_new, temp, vocab, hidden,
                        num_heads, num_kv_heads, head_dim, intermediate, out);
    }
};

inline int reference_ids(const gen_weights& w, const int* prompt, int prompt_len,
                         int n_new, int* out) {
    const float eps = 1e-5f;
    for (int i = 0; i < prompt_len; i++) {
        out[i] = prompt[i];
    }
    int len = prompt_len;
    std::vector<float> X(static_cast<size_t>(w.max_seq) * w.hidden);
    for (int step = 0; step < n_new && len < w.max_seq; step++) {
        for (int i = 0; i < len; i++) {
            const float* row = w.embed.data() + static_cast<size_t>(out[i]) * w.hidden;
            for (int d = 0; d < w.hidden; d++) {
                X[static_cast<size_t>(i) * w.hidden + d] = row[d];
            }
        }
        std::vector<float> hs;
        for (int l = 0; l < w.num_layers; l++) {
            const std::vector<float>& src = (l == 0) ? X : hs;
            std::vector<float> tmp;
            layer_detail::reference(src, w.rms1[l], w.Wq[l], w.Wk[l], w.Wv[l], w.Wo[l],
                                    w.cos_t, w.sin_t, w.rms2[l], w.W1[l], w.W2[l],
                                    w.W3[l], len, w.hidden, w.num_heads, w.num_kv_heads,
                                    w.head_dim, w.intermediate, eps, tmp);
            hs = tmp;
        }
        for (int i = 0; i < len; i++) {
            std::vector<float> row(hs.begin() + static_cast<size_t>(i) * w.hidden,
                                   hs.begin() + static_cast<size_t>(i + 1) * w.hidden);
            const std::vector<float> nr = rms_detail::reference(row, w.final_rms, w.hidden, eps);
            for (int d = 0; d < w.hidden; d++) {
                hs[static_cast<size_t>(i) * w.hidden + d] = nr[d];
            }
        }
        const std::vector<float> last(hs.begin() + static_cast<size_t>(len - 1) * w.hidden,
                                      hs.begin() + static_cast<size_t>(len) * w.hidden);
        std::vector<float> logits;
        layer_detail::dmatmul(1, w.vocab, w.hidden, last, w.lm_head, logits);
        int best = 0;
        for (int v = 1; v < w.vocab; v++) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        out[len++] = best;
    }
    return len;
}

}  // namespace gen_detail

bool check_generate() {
    using namespace gen_detail;

    const int total = 5;
    int passed = 0;

    printf("==== 自回归生成测试 ====\n");

    const int prompt[3] = {1, 5, 9};
    const int n_new = 3;

    gen_weights w;
    w.build(16, 8, 2, 1, 4, 16, 2, 8, false);

    std::vector<int> got(3 + n_new, 0);
    std::vector<int> want(3 + n_new, 0);
    const int len_got = w.run(prompt, 3, n_new, 0.0f, got.data());

    // ---------- 1. 与 double 参考逐 token 一致 ----------
    {
        const int len_want = reference_ids(w, prompt, 3, n_new, want.data());
        bool ok = (len_got == len_want);
        for (int i = 0; i < len_got && ok; i++) {
            if (got[i] != want[i]) {
                ok = false;
            }
        }
        printf("[1/%d] 与 double 参考逐 token 一致 (prompt+%d)\n", total, n_new);
        printf("       实际 %d %d %d   期望 %d %d %d   %s\n",
               got[3], got[4], got[5], want[3], want[4], want[5], ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 2. 前缀保持 + 长度正确 ----------
    {
        bool ok = (len_got == 3 + n_new);
        for (int i = 0; i < 3; i++) {
            if (got[i] != prompt[i]) {
                ok = false;
            }
        }
        printf("[2/%d] 前缀保持不变、长度 = %d   %s\n", total, 3 + n_new, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 3. 全零 lm_head -> 所有 logits 相等 -> 恒取下标 0 ----------
    {
        gen_weights z;
        z.build(16, 8, 2, 1, 4, 16, 2, 8, true);
        std::vector<int> o(3 + n_new, -1);
        z.run(prompt, 3, n_new, 0.0f, o.data());
        bool ok = true;
        for (int i = 3; i < 3 + n_new; i++) {
            if (o[i] != 0) {
                ok = false;
            }
        }
        printf("[3/%d] 全零 lm_head -> 生成 token 全为 0      %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 4. 确定性：跑两次完全一致 ----------
    {
        std::vector<int> again(3 + n_new, -1);
        w.run(prompt, 3, n_new, 0.0f, again.data());
        bool ok = true;
        for (int i = 0; i < 3 + n_new; i++) {
            if (again[i] != got[i]) {
                ok = false;
            }
        }
        printf("[4/%d] 贪心生成可复现（跑两次一致）          %s\n", total, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 5. EOS 提前停止 ----------
    {
        const int eos = got[3];
        std::vector<int> o(3 + n_new, -1);
        const int len = generate(prompt, 3, &eos, 1, w.embed.data(), w.lm_head.data(),
                                 w.layers.data(), w.num_layers, w.final_rms.data(),
                                 w.cos_t.data(), w.sin_t.data(), w.max_seq, n_new, 0.0f,
                                 w.vocab, w.hidden, w.num_heads, w.num_kv_heads, w.head_dim,
                                 w.intermediate, o.data());
        const bool ok = (len == 4 && o[3] == eos);
        printf("[5/%d] 命中 EOS 立即停止 (长度 %d)           %s\n", total, len, ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    printf("check_generate: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_tensor_dir() —— model.bf v2：张量目录 + 数据区
//
//  被测对象：libs/model_libs/bfile.h 的 write_bfile / View / read_header_file
//  重点：目录项布局、数据区 64 字节对齐、按名字取数、损坏文件被拦住
// ============================================================
bool check_tensor_dir() {
    namespace fs = std::filesystem;
    using text_detail::read_bf_head;
    using text_detail::read_bytes;
    using text_detail::write_model_info;

    const int total = 11;
    int passed = 0;
    printf("==== model.bf 张量目录测试 ====\n");

    const fs::path dir = fs::path("_file") / "bfile_test" / "tensor_dir";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const fs::path bf_path = dir / "model.bf";

    text_detail::model_params p;
    p.hidden = 8;
    p.layers = 2;
    p.mode = 1;
    p.num_heads = 2;
    p.num_kv_heads = 1;
    p.head_dim = 4;
    p.intermediate_size = 16;
    p.vocab_size = 259;
    write_model_info(dir, p);

    std::vector<float> norm(8), q(64), odd(123);
    for (int i = 0; i < 8; i++) norm[i] = 1.0f + 0.125f * i;
    for (int i = 0; i < 64; i++) q[i] = ffn_detail::pseudo(i + 5) * 0.5f;
    for (int i = 0; i < 123; i++) odd[i] = ffn_detail::pseudo(i + 77) * 0.25f;

    bfile::Header head;
    head.hidden = 8;
    head.layers = 2;
    head.mode = 1;
    head.num_heads = 2;
    head.num_kv_heads = 1;
    head.head_dim = 4;
    head.intermediate = 16;
    head.vocab_size = 259;

    std::vector<bfile::Tensor> ts;
    ts.push_back(bfile::make_f32("model.norm.weight", {8}, norm));
    ts.push_back(bfile::make_f32("model.layers.0.self_attn.q_proj.weight", {8, 8}, q));
    ts.push_back(bfile::make_f32("model.odd.weight", {123}, odd));

    // 目录 3*104=312 字节，64+312=376 -> 数据区从 384 开始
    // 数据区：norm 32 字节，q 对齐到 64 再放 256 字节，odd 对齐到 320 再放 492 字节
    const size_t want_data_off = bfile::align_up(bfile::kHeaderSize + ts.size() * bfile::kEntrySize);
    const uint64_t want_off0 = 0;
    const uint64_t want_off1 = bfile::align_up(32);
    const uint64_t want_off2 = bfile::align_up(want_off1 + 256);
    const size_t want_size = want_data_off + static_cast<size_t>(want_off2) + 492;

    try {
        bfile::write_bfile(bf_path.string(), head, ts);
    } catch (const std::exception& e) {
        printf("       ! write_bfile 抛异常: %s\n", e.what());
    }

    // ---------- 1. 头部字段 ----------
    {
        const text_detail::bf_head h = read_bf_head(dir);
        const bool ok = h.is_v2() && h.tensor_count == 3
            && h.data_off == static_cast<uint32_t>(want_data_off) && h.entry_size == 104;
        printf("[ 1/%d] 头部 version=2 count=3 data_off=%zu entry=104  %s\n",
               total, want_data_off, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 2. 文件大小 ----------
    {
        const uint64_t size = fs::file_size(bf_path, ec);
        const bool ok = (size == want_size);
        printf("[ 2/%d] 文件大小 %llu（期望 %llu）               %s\n", total,
               (unsigned long long)size, (unsigned long long)want_size, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 3. 目录项：offset 64 对齐、nbytes 与 shape 相符 ----------
    {
        bfile::Header read_head;
        std::vector<bfile::Entry> entries;
        uint64_t size = 0;
        bfile::read_header_file(bf_path.string(), read_head, entries, size);
        bool ok = (entries.size() == 3);
        if (ok) ok = entries[0].offset == want_off0 && entries[1].offset == want_off1
                     && entries[2].offset == want_off2;
        if (ok) ok = entries[0].nbytes == 32 && entries[1].nbytes == 256 && entries[2].nbytes == 492;
        printf("[ 3/%d] 数据区偏移 0/64/320，长度 32/256/492     %s\n", total, ok ? "OK" : "FAIL");
        if (!ok && entries.size() == 3) {
            printf("       ! 实际 %llu/%llu/%llu 长度 %llu/%llu/%llu\n",
                   (unsigned long long)entries[0].offset, (unsigned long long)entries[1].offset,
                   (unsigned long long)entries[2].offset, (unsigned long long)entries[0].nbytes,
                   (unsigned long long)entries[1].nbytes, (unsigned long long)entries[2].nbytes);
        }
        if (ok) passed++;
    }

    // ---------- 4. 目录项 shape / dtype ----------
    {
        bfile::Header read_head;
        std::vector<bfile::Entry> entries;
        uint64_t size = 0;
        bfile::read_header_file(bf_path.string(), read_head, entries, size);
        bool ok = entries.size() == 3;
        if (ok) ok = entries[1].ndim == 2 && entries[1].shape[0] == 8 && entries[1].shape[1] == 8;
        if (ok) ok = entries[2].ndim == 1 && entries[2].shape[0] == 123;
        if (ok) ok = entries[0].dtype == bfile::F32 && entries[1].dtype == bfile::F32;
        printf("[ 4/%d] 目录项 shape / dtype 正确               %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 5. install_model + 张量表挂上 ----------
    {
        free_model_map();
        bool ok = false;
        try {
            install_model(dir);
            const bfile::View& v = bfile::current();
            ok = v.ok && v.tensor_count == 3 && v.data_off == want_data_off;
        } catch (const std::exception& e) {
            printf("       ! install_model 抛异常: %s\n", e.what());
        }
        printf("[ 5/%d] install_model 后张量表就绪（%u 个）    %s\n", total,
               (unsigned)bfile::current().tensor_count, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 6. 按名字取指针 ----------
    {
        const bfile::Entry* e0 = find_tensor("model.norm.weight");
        const bfile::Entry* e1 = find_tensor("model.layers.0.self_attn.q_proj.weight");
        const bfile::Entry* e2 = find_tensor("model.odd.weight");
        const bool ok = e0 != nullptr && e1 != nullptr && e2 != nullptr && e0->ndim == 1;
        printf("[ 6/%d] 三个张量都能按名字找到                  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 7. 数据逐字节一致（一维） ----------
    {
        const float* got = tensor_f32("model.norm.weight");
        bool ok = (got != nullptr);
        if (ok) ok = std::memcmp(got, norm.data(), norm.size() * 4) == 0;
        printf("[ 7/%d] norm 权重逐字节一致                     %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 8. 数据逐字节一致（二维） ----------
    {
        const float* got = tensor_f32("model.layers.0.self_attn.q_proj.weight");
        bool ok = (got != nullptr);
        if (ok) ok = std::memcmp(got, q.data(), q.size() * 4) == 0;
        printf("[ 8/%d] q_proj 权重逐字节一致                   %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 9. 长度不是 64 倍数的张量也能读对 ----------
    {
        const bfile::Entry* e = find_tensor("model.odd.weight");
        const float* got = tensor_f32("model.odd.weight");
        bool ok = (e != nullptr && got != nullptr && e->nbytes == 492);
        if (ok) ok = std::memcmp(got, odd.data(), odd.size() * 4) == 0;
        printf("[ 9/%d] 492 字节（非对齐长度）张量读取正确       %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 10. 查不到的名字返回空 ----------
    {
        const bool ok = (find_tensor("model.not.exist") == nullptr)
            && (tensor_f32("model.not.exist") == nullptr);
        printf("[10/%d] 查不存在的张量返回空                    %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 11. 损坏的 data_off 被拦住 ----------
    {
        std::string bytes;
        read_bytes(bf_path.string(), bytes);
        bool ok = false;
        if (bytes.size() >= 64) {
            unsigned char broken[64];
            std::memcpy(broken, bytes.data(), 64);
            bfile::write_u32(broken + 46, 0xFFFFFFF0u);   // data_off 指到文件外
            bfile::View v;
            ok = !v.open(broken, bytes.size());
            // 顺便确认正常文件是能被接受的
            bfile::View good;
            ok = ok && good.open(bytes.data(), bytes.size()) && good.tensor_count == 3;
        }
        printf("[11/%d] 数据区起点越界 -> 检查不通过           %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    free_model_map();
    printf("check_tensor_dir: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_convert() —— safetensors -> model.bf（纯 C++ 转换器）
//
//  被测对象：libs/model_libs/safetensors.h + converter.h
//  做法：造一份演示模型（层权重全 0，lm_head 输出“下一个字节”），
//        再塞两个 f16/bf16 张量，转成 model.bf 后逐项核对
// ============================================================
namespace conv_detail {

inline void make_probe_tensors(std::vector<safetensors::RawTensor>& out) {
    const float values[4] = {0.5f, -0.25f, 1.5f, -3.0f};

    safetensors::RawTensor f16;
    f16.name = "probe.f16.weight";
    f16.dtype = bfile::F16;
    f16.shape = {4};
    f16.data.resize(8);
    for (int i = 0; i < 4; i++) {
        const uint16_t h = converter::float_to_half(values[i]);
        f16.data[static_cast<size_t>(i) * 2] = static_cast<unsigned char>(h & 0xFF);
        f16.data[static_cast<size_t>(i) * 2 + 1] = static_cast<unsigned char>(h >> 8);
    }
    out.push_back(f16);

    safetensors::RawTensor bf16;
    bf16.name = "probe.bf16.weight";
    bf16.dtype = bfile::BF16;
    bf16.shape = {4};
    bf16.data.resize(8);
    for (int i = 0; i < 4; i++) {
        const uint16_t h = converter::float_to_bf16(values[i]);
        bf16.data[static_cast<size_t>(i) * 2] = static_cast<unsigned char>(h & 0xFF);
        bf16.data[static_cast<size_t>(i) * 2 + 1] = static_cast<unsigned char>(h >> 8);
    }
    out.push_back(bf16);
}

}  // namespace conv_detail

bool check_convert() {
    namespace fs = std::filesystem;
    using text_detail::read_bf_head;

    const int total = 12;
    int passed = 0;
    printf("==== safetensors -> model.bf 转换器测试 ====\n");

    const fs::path root = fs::path("_file") / "conv_test";
    const fs::path src = root / "src";
    const fs::path out_dir = root / "model";
    std::error_code ec;
    fs::remove_all(root, ec);

    converter::DemoSpec spec;
    const int hidden = spec.hidden;
    const int heads = spec.heads;
    const int head_dim = spec.head_dim;
    const int kv_heads = spec.kv_heads;
    const int intermediate = spec.intermediate;
    const int layers = spec.layers;
    const int vocab = spec.vocab;

    converter::Result result;
    bool converted = false;
    try {
        converter::make_demo_model(src.string(), spec);
        std::vector<safetensors::RawTensor> probe;
        conv_detail::make_probe_tensors(probe);
        safetensors::write((src / "probe.safetensors").string(), probe);

        converter::Options opt;
        opt.sources.push_back((src / "model.safetensors").string());
        opt.sources.push_back((src / "probe.safetensors").string());
        opt.out_dir = out_dir.string();
        opt.config_path = (src / "config.json").string();
        opt.tokenizer_dir = src.string();
        opt.verbose = false;
        result = converter::convert(opt);
        converted = true;
    } catch (const std::exception& e) {
        printf("       ! 转换抛异常: %s\n", e.what());
    }

    // 预期目录项：1 embed + layers*9 + 1 norm + 1 lm_head + 2 probe
    const size_t want_count = 1 + static_cast<size_t>(layers) * 9 + 1 + 1 + 2;
    const size_t want_data_off = bfile::align_up(bfile::kHeaderSize + want_count * bfile::kEntrySize);

    // ---------- 1. 转换成功 + 文件存在 ----------
    {
        const bool ok = converted && fs::exists(out_dir / "model.bf", ec)
            && fs::exists(out_dir / "model_info.json", ec);
        printf("[ 1/%d] 转换完成，model.bf / model_info.json 都在  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 2. 超参 ----------
    {
        const bool ok = converted && result.head.hidden == static_cast<uint64_t>(hidden)
            && result.head.layers == static_cast<uint64_t>(layers)
            && result.head.num_heads == heads && result.head.num_kv_heads == kv_heads
            && result.head.head_dim == head_dim && result.head.intermediate == static_cast<uint32_t>(intermediate)
            && result.head.vocab_size == static_cast<uint64_t>(vocab);
        printf("[ 2/%d] 超参 hidden=%d layers=%d heads=%d kv=%d dim=%d inter=%d vocab=%d  %s\n",
               total, hidden, layers, heads, kv_heads, head_dim, intermediate, vocab, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 3. 张量个数与数据区起点 ----------
    {
        const text_detail::bf_head h = read_bf_head(out_dir);
        const bool ok = h.is_v2() && h.tensor_count == want_count
            && h.data_off == static_cast<uint32_t>(want_data_off);
        printf("[ 3/%d] 张量 %zu 个（2 个分片合并），数据区起点 %zu  %s\n",
               total, want_count, want_data_off, ok ? "OK" : "FAIL");
        if (!ok) {
            printf("       ! 实际 count=%u data_off=%u\n",
                   (unsigned)h.tensor_count, (unsigned)h.data_off);
        }
        if (ok) passed++;
    }

    // ---------- 4. 每个张量 offset 64 对齐、不越界 ----------
    {
        bool ok = converted;
        try {
            bfile::Header head;
            std::vector<bfile::Entry> dir;
            uint64_t size = 0;
            bfile::read_header_file((out_dir / "model.bf").string(), head, dir, size);
            // read_header_file 内部就会校验对齐与越界，能跑完就说明没问题
            ok = ok && (size > 0) && !dir.empty();
            ok = ok && (head.hidden == static_cast<uint64_t>(hidden));
        } catch (const std::exception& e) {
            printf("       ! read_header_file 抛异常: %s\n", e.what());
            ok = false;
        }
        printf("[ 4/%d] 全部张量 offset 64 对齐、数据不越界      %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    free_model_map();
    bool loaded = false;
    try {
        install_model(out_dir);
        loaded = bfile::current().ok;
    } catch (const std::exception& e) {
        printf("       ! install_model 抛异常: %s\n", e.what());
    }

    // ---------- 5. 全部张量都是 f32、名字都能找到 ----------
    {
        bool ok = loaded;
        const bfile::View& v = bfile::current();
        for (uint32_t i = 0; i < v.tensor_count && ok; i++) {
            if (v.entry(i)->dtype != bfile::F32) ok = false;
        }
        static const char* names[] = {
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight", "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight", "probe.f16.weight", "probe.bf16.weight"};
        for (const char* name : names) {
            if (find_tensor(name) == nullptr) {
                ok = false;
                printf("       ! 少了 %s\n", name);
            }
        }
        printf("[ 5/%d] 全部张量 dtype=f32 且都装上了            %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 6. embed 是恒等映射（转置时不该动它） ----------
    {
        const float* embed = tensor_f32("model.embed_tokens.weight");
        bool ok = (embed != nullptr);
        if (ok) ok = (embed[0] == 1.0f) && (embed[1] == 0.0f)
                     && (embed[static_cast<size_t>(5) * hidden + 5] == 1.0f)
                     && (embed[static_cast<size_t>(5) * hidden + 6] == 0.0f)
                     && (embed[static_cast<size_t>(vocab - 1) * hidden] == 0.0f);
        printf("[ 6/%d] embed_tokens 原样搬过来（没被转置）      %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 7. lm_head 被转置：bf[d][t]=1 当且仅当 t=d+1 ----------
    {
        const float* lm = tensor_f32("lm_head.weight");
        const size_t n = static_cast<size_t>(vocab);
        bool ok = (lm != nullptr);
        if (ok) ok = (lm[0 * n + 1] == 1.0f) && (lm[0 * n + 0] == 0.0f)
                     && (lm[10 * n + 11] == 1.0f) && (lm[10 * n + 10] == 0.0f)
                     && (lm[254 * n + 255] == 1.0f);
        printf("[ 7/%d] lm_head 已转置（[hidden,vocab]）          %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 8. f16 -> f32 ----------
    {
        const float* got = tensor_f32("probe.f16.weight");
        const float want[4] = {0.5f, -0.25f, 1.5f, -3.0f};
        bool ok = (got != nullptr);
        for (int i = 0; ok && i < 4; i++) {
            if (got[i] != want[i]) ok = false;
        }
        printf("[ 8/%d] f16 张量转成 f32 数值精确               %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 9. bf16 -> f32 ----------
    {
        const float* got = tensor_f32("probe.bf16.weight");
        const float want[4] = {0.5f, -0.25f, 1.5f, -3.0f};
        bool ok = (got != nullptr);
        for (int i = 0; ok && i < 4; i++) {
            if (got[i] != want[i]) ok = false;
        }
        printf("[ 9/%d] bf16 张量转成 f32 数值精确              %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 10. bind_model：形状检查全过 ----------
    {
        pipeline::ModelRefs refs;
        std::string err;
        const bool ok = pipeline::bind_model(refs, 64, &err);
        if (!ok) printf("       ! %s\n", err.c_str());
        printf("[10/%d] bind_model 形状校验全过                 %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 11. 词表文件也一起复制过来了 ----------
    {
        const bool ok = fs::exists(out_dir / "vocab.json", ec) && fs::exists(out_dir / "merges.txt", ec);
        printf("[11/%d] 分词器文件已复制到输出目录              %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 12. 权重共享：没有 lm_head 时自动补转置副本 ----------
    {
        bool ok = false;
        try {
            const fs::path src2 = root / "src_tied";
            const fs::path out2 = root / "model_tied";
            converter::make_demo_model(src2.string(), spec, false);   // 不写 lm_head
            converter::Options opt;
            opt.sources.push_back((src2 / "model.safetensors").string());
            opt.out_dir = out2.string();
            opt.config_path = (src2 / "config.json").string();
            opt.verbose = false;
            const converter::Result r2 = converter::convert(opt);
            // 重新装一遍这个目录，检查补出来的 lm_head
            free_model_map();
            install_model(out2);
            const float* lm = tensor_f32("lm_head.weight");
            const bfile::Entry* e = find_tensor("lm_head.weight");
            ok = r2.tied_lm_head && lm != nullptr && e != nullptr
                 && e->shape[0] == static_cast<uint32_t>(hidden)
                 && e->shape[1] == static_cast<uint32_t>(vocab)
                 && (lm[0 * static_cast<size_t>(vocab) + 0] == 1.0f)      // embed 第 0 行第 0 列
                 && (lm[3 * static_cast<size_t>(vocab) + 3] == 1.0f);
        } catch (const std::exception& e) {
            printf("       ! 权重共享路径抛异常: %s\n", e.what());
        }
        printf("[12/%d] 缺 lm_head 时补一份转置副本              %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    free_model_map();
    printf("check_convert: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_bind_weights() —— 按名字拿指针，填进 layers[] 跑生成
//
//  被测对象：libs/model_libs/load_weights.h + pipeline.h 的 bind_model
//  重点：指针是否指到正确的张量、形状不对要能拦住、缺张量要能报出名字
// ============================================================
namespace pipe_detail {

// 造一套权重（lm_head 放大一点，让 argmax 稳稳分开，浮点误差不会翻转结果）
inline void make_weights(gen_detail::gen_weights& w, int vocab, int hidden, int heads, int kv,
                         int head_dim, int inter, int layers, int max_seq) {
    w.build(vocab, hidden, heads, kv, head_dim, inter, layers, max_seq, false);
    for (size_t i = 0; i < w.lm_head.size(); i++) {
        w.lm_head[i] = ffn_detail::pseudo(static_cast<int>(i) + 211) * 2.0f;
    }
}

// 把一套权重写成 model.bf（按引擎要的 [k,n] 布局，不需要转置）
inline void write_model_dir(const std::filesystem::path& dir, const gen_detail::gen_weights& w,
                            bool wrong_q = false, bool drop_final_norm = false) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const uint32_t h = static_cast<uint32_t>(w.hidden);
    const uint32_t qd = static_cast<uint32_t>(w.num_heads * w.head_dim);
    const uint32_t kvd = static_cast<uint32_t>(w.num_kv_heads * w.head_dim);
    const uint32_t inter = static_cast<uint32_t>(w.intermediate);
    const uint32_t vocab = static_cast<uint32_t>(w.vocab);

    bfile::Header head;
    head.hidden = w.hidden;
    head.layers = w.num_layers;
    head.mode = 1;
    head.num_heads = static_cast<uint8_t>(w.num_heads);
    head.num_kv_heads = static_cast<uint8_t>(w.num_kv_heads);
    head.head_dim = static_cast<uint8_t>(w.head_dim);
    head.intermediate = inter;
    head.vocab_size = vocab;

    std::vector<bfile::Tensor> ts;
    ts.push_back(bfile::make_f32("model.embed_tokens.weight", {vocab, h}, w.embed));
    if (!drop_final_norm) ts.push_back(bfile::make_f32("model.norm.weight", {h}, w.final_rms));
    for (int l = 0; l < w.num_layers; l++) {
        const std::vector<uint32_t> q_shape = wrong_q
            ? std::vector<uint32_t>{qd, h}      // 故意写成没转置的形状
            : std::vector<uint32_t>{h, qd};
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "input_layernorm.weight"), {h}, w.rms1[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "self_attn.q_proj.weight"), q_shape, w.Wq[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "self_attn.k_proj.weight"), {h, kvd}, w.Wk[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "self_attn.v_proj.weight"), {h, kvd}, w.Wv[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "self_attn.o_proj.weight"), {qd, h}, w.Wo[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "post_attention_layernorm.weight"), {h}, w.rms2[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "mlp.gate_proj.weight"), {h, inter}, w.W1[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "mlp.down_proj.weight"), {inter, h}, w.W2[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l, "mlp.up_proj.weight"), {h, inter}, w.W3[l]));
    }
    ts.push_back(bfile::make_f32("lm_head.weight", {h, vocab}, w.lm_head));
    bfile::write_bfile((dir / "model.bf").string(), head, ts);

    json_lib info;
    info.SetJsonWay((dir / "model_info.json").string());
    info.WriteJsonKey("hidden_size", std::to_string(w.hidden));
    info.WriteJsonKey("layer_count", std::to_string(w.num_layers));
    info.WriteJsonKey("num_heads", std::to_string(w.num_heads));
    info.WriteJsonKey("num_kv_heads", std::to_string(w.num_kv_heads));
    info.WriteJsonKey("head_dim", std::to_string(w.head_dim));
    info.WriteJsonKey("intermediate_size", std::to_string(w.intermediate));
    info.WriteJsonKey("vocab_size", std::to_string(w.vocab));
    info.WriteJsonKey("rope_theta", "10000");
    info.WriteJsonKey("max_position_embeddings", std::to_string(w.max_seq));
}

// 从已经装好的 model.bf 里把权重抄成 gen_detail::gen_weights（走的是另一条路径，
// 于是参考实现用的是文件里的值，引擎用的是 mmap 指针，两边对不上就会暴露）
inline bool load_gen_weights(gen_detail::gen_weights& w, int max_seq) {
    if (!bfile::current().ok) return false;
    w.vocab = static_cast<int>(model.vocab_size);
    w.hidden = static_cast<int>(model.h);
    w.num_heads = model.num_heads;
    w.num_kv_heads = model.num_kv_heads;
    w.head_dim = model.head_dim;
    w.intermediate = static_cast<int>(model.intermediate_size);
    w.num_layers = static_cast<int>(model.l);
    w.max_seq = max_seq;

    const size_t h = static_cast<size_t>(w.hidden);
    const size_t qd = static_cast<size_t>(w.num_heads) * w.head_dim;
    const size_t kvd = static_cast<size_t>(w.num_kv_heads) * w.head_dim;
    const size_t inter = static_cast<size_t>(w.intermediate);

    auto grab = [](const std::string& name, std::vector<float>& dst, size_t count) -> bool {
        const float* p = tensor_f32(name);
        if (p == nullptr) return false;
        dst.assign(p, p + count);
        return true;
    };

    if (!grab("model.embed_tokens.weight", w.embed, h * static_cast<size_t>(w.vocab))) return false;
    if (!grab("model.norm.weight", w.final_rms, h)) return false;
    const float* lm = bind_lm_head(w.hidden, w.vocab);
    if (lm == nullptr) return false;
    w.lm_head.assign(lm, lm + h * static_cast<size_t>(w.vocab));

    w.rms1.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.rms2.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.Wq.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.Wk.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.Wv.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.Wo.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.W1.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.W2.assign(static_cast<size_t>(w.num_layers), std::vector<float>());
    w.W3.assign(static_cast<size_t>(w.num_layers), std::vector<float>());

    for (int l = 0; l < w.num_layers; l++) {
        if (!grab(layer_tensor_name(l, "input_layernorm.weight"), w.rms1[l], h)) return false;
        if (!grab(layer_tensor_name(l, "self_attn.q_proj.weight"), w.Wq[l], h * qd)) return false;
        if (!grab(layer_tensor_name(l, "self_attn.k_proj.weight"), w.Wk[l], h * kvd)) return false;
        if (!grab(layer_tensor_name(l, "self_attn.v_proj.weight"), w.Wv[l], h * kvd)) return false;
        if (!grab(layer_tensor_name(l, "self_attn.o_proj.weight"), w.Wo[l], qd * h)) return false;
        if (!grab(layer_tensor_name(l, "post_attention_layernorm.weight"), w.rms2[l], h)) return false;
        if (!grab(layer_tensor_name(l, "mlp.gate_proj.weight"), w.W1[l], h * inter)) return false;
        if (!grab(layer_tensor_name(l, "mlp.down_proj.weight"), w.W2[l], inter * h)) return false;
        if (!grab(layer_tensor_name(l, "mlp.up_proj.weight"), w.W3[l], h * inter)) return false;
    }

    w.layers.assign(static_cast<size_t>(w.num_layers), LayerWeights{});
    for (int l = 0; l < w.num_layers; l++) {
        w.layers[l].rms1_weight = w.rms1[l].data();
        w.layers[l].Wq = w.Wq[l].data();
        w.layers[l].Wk = w.Wk[l].data();
        w.layers[l].Wv = w.Wv[l].data();
        w.layers[l].Wo = w.Wo[l].data();
        w.layers[l].rms2_weight = w.rms2[l].data();
        w.layers[l].W1 = w.W1[l].data();
        w.layers[l].W2 = w.W2[l].data();
        w.layers[l].W3 = w.W3[l].data();
    }
    rope_detail::make_tables(max_seq, w.head_dim, w.cos_t, w.sin_t);
    return true;
}

}  // namespace pipe_detail

bool check_bind_weights() {
    namespace fs = std::filesystem;
    using pipe_detail::load_gen_weights;
    using pipe_detail::make_weights;
    using pipe_detail::write_model_dir;

    const int total = 9;
    int passed = 0;
    printf("==== 按名字装载权重测试 ====\n");

    const fs::path root = fs::path("_file") / "bind_test";
    const fs::path dir = root / "model";
    std::error_code ec;
    fs::remove_all(root, ec);

    const int vocab = 24, hidden = 8, heads = 2, kv = 1, head_dim = 4;
    const int inter = 16, layers = 2, max_seq = 8;

    gen_detail::gen_weights w;
    make_weights(w, vocab, hidden, heads, kv, head_dim, inter, layers, max_seq);
    write_model_dir(dir, w);

    free_model_map();
    bool loaded = false;
    try {
        install_model(dir);
        loaded = bfile::current().ok;
    } catch (const std::exception& e) {
        printf("       ! install_model 抛异常: %s\n", e.what());
    }

    // ---------- 1. 装载 + 张量个数 ----------
    {
        const size_t want = 1 + static_cast<size_t>(layers) * 9 + 1 + 1;
        const bool ok = loaded && bfile::current().tensor_count == want;
        printf("[ 1/%d] install_model 成功，张量 %zu 个          %s\n", total, want, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 2. bind_layers 填的指针与 find_tensor 一致 ----------
    {
        std::vector<LayerWeights> got(static_cast<size_t>(layers));
        std::string missing;
        const bool bound = bind_layers(got.data(), layers, &missing);
        bool ok = bound;
        if (ok) ok = got[1].Wq == tensor_f32(layer_tensor_name(1, "self_attn.q_proj.weight"));
        if (ok) ok = got[0].W2 == tensor_f32(layer_tensor_name(0, "mlp.down_proj.weight"));
        if (ok) ok = got[1].rms2_weight == tensor_f32(layer_tensor_name(1, "post_attention_layernorm.weight"));
        if (ok) ok = got[0].Wq != got[1].Wq;   // 两层不能指到同一块
        if (!bound) printf("       ! bind_layers 失败于 %s\n", missing.c_str());
        printf("[ 2/%d] 9 个权重指针都指到对应张量              %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 3. 从文件读回的数值与写入的一致 ----------
    {
        const float* embed = tensor_f32("model.embed_tokens.weight");
        const float* wq = tensor_f32(layer_tensor_name(1, "self_attn.q_proj.weight"));
        bool ok = embed != nullptr && wq != nullptr;
        if (ok) ok = std::memcmp(embed, w.embed.data(), w.embed.size() * 4) == 0;
        if (ok) ok = std::memcmp(wq, w.Wq[1].data(), w.Wq[1].size() * 4) == 0;
        printf("[ 3/%d] embed / 第1层 Wq 数值逐字节一致          %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 4. bind_model 形状校验全过 ----------
    pipeline::ModelRefs refs;
    std::string err;
    bool bound_model = false;
    {
        bound_model = pipeline::bind_model(refs, max_seq, &err);
        if (!bound_model) printf("       ! %s\n", err.c_str());
        printf("[ 4/%d] bind_model 形状校验全过                  %s\n", total, bound_model ? "OK" : "FAIL");
        if (bound_model) passed++;
    }

    // ---------- 5. 参考实现也能从文件拿到权重 ----------
    gen_detail::gen_weights ref;
    bool ref_ok = false;
    {
        ref_ok = bound_model && load_gen_weights(ref, max_seq);
        printf("[ 5/%d] 参考实现从同一份文件读出权重            %s\n", total, ref_ok ? "OK" : "FAIL");
        if (ref_ok) passed++;
    }

    // ---------- 6. 引擎生成 vs double 参考 ----------
    {
        const int prompt[3] = {1, 2, 3};
        std::vector<int> got(6, -1), want(6, -1);
        int len_got = -1, len_want = -1;
        if (ref_ok) {
            len_got = generate(prompt, 3, nullptr, 0, refs.embed, refs.lm_head, refs.layers.data(),
                               refs.num_layers, refs.final_norm, refs.rope.cos.data(),
                               refs.rope.sin.data(), refs.max_seq, 3, 0.0f, refs.vocab_size,
                               refs.hidden, refs.num_heads, refs.num_kv_heads, refs.head_dim,
                               refs.intermediate, got.data());
            len_want = gen_detail::reference_ids(ref, prompt, 3, 3, want.data());
        }
        bool ok = ref_ok && len_got == len_want;
        if (ok) {
            for (int i = 0; i < len_got; i++) {
                if (got[i] != want[i]) ok = false;
            }
        }
        printf("[ 6/%d] 生成结果与 double 参考一致 (实际", total);
        for (int i = 3; i < len_got && i < 6; i++) printf(" %d", got[i]);
        printf(" 期望");
        for (int i = 3; i < len_want && i < 6; i++) printf(" %d", want[i]);
        printf(")  %s\n", ok ? "OK" : "FAIL");
        if (ok) {
            passed++;
        }
    }

    // ---------- 7. 形状写反（没转置）要能被拦住 ----------
    {
        const fs::path bad_dir = root / "wrong_shape";
        gen_detail::gen_weights w6;
        make_weights(w6, 24, 8, 2, 1, 6, 16, 1, max_seq);   // q_dim=12 != hidden=8
        write_model_dir(bad_dir, w6, true, false);
        free_model_map();
        bool ok = false;
        try {
            install_model(bad_dir);
            pipeline::ModelRefs bad_refs;
            std::string bad_err;
            const bool bound_bad = pipeline::bind_model(bad_refs, max_seq, &bad_err);
            ok = !bound_bad && bad_err.find("q_proj") != std::string::npos;
            if (!ok) printf("       ! 竟然通过了，错误信息: %s\n", bad_err.c_str());
        } catch (const std::exception& e) {
            printf("       ! 抛异常: %s\n", e.what());
        }
        printf("[ 7/%d] q_proj 形状写反 -> 报错并指出张量名      %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 8. 缺张量要报出缺哪个 ----------
    {
        const fs::path bad_dir = root / "missing";
        gen_detail::gen_weights w8;
        make_weights(w8, 24, 8, 2, 1, 4, 16, 1, max_seq);
        write_model_dir(bad_dir, w8, false, true);   // 去掉 model.norm.weight
        free_model_map();
        bool ok = false;
        try {
            install_model(bad_dir);
            pipeline::ModelRefs bad_refs;
            std::string bad_err;
            const bool bound_bad = pipeline::bind_model(bad_refs, max_seq, &bad_err);
            ok = !bound_bad && bad_err.find("model.norm.weight") != std::string::npos;
            if (!ok) printf("       ! 竟然通过了，错误信息: %s\n", bad_err.c_str());
        } catch (const std::exception& e) {
            printf("       ! 抛异常: %s\n", e.what());
        }
        printf("[ 8/%d] 缺 model.norm.weight -> 报出名字          %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 9. 贪心生成可复现 ----------
    {
        free_model_map();
        install_model(dir);
        pipeline::ModelRefs r2;
        std::string e2;
        bool ok = pipeline::bind_model(r2, max_seq, &e2);
        const int prompt[3] = {5, 6, 7};
        std::vector<int> a(6, -1), b(6, -1);
        if (ok) {
            const int la = generate(prompt, 3, nullptr, 0, r2.embed, r2.lm_head, r2.layers.data(),
                                    r2.num_layers, r2.final_norm, r2.rope.cos.data(),
                                    r2.rope.sin.data(), r2.max_seq, 3, 0.0f, r2.vocab_size,
                                    r2.hidden, r2.num_heads, r2.num_kv_heads, r2.head_dim,
                                    r2.intermediate, a.data());
            const int lb = generate(prompt, 3, nullptr, 0, r2.embed, r2.lm_head, r2.layers.data(),
                                    r2.num_layers, r2.final_norm, r2.rope.cos.data(),
                                    r2.rope.sin.data(), r2.max_seq, 3, 0.0f, r2.vocab_size,
                                    r2.hidden, r2.num_heads, r2.num_kv_heads, r2.head_dim,
                                    r2.intermediate, b.data());
            ok = (la == lb);
            for (int i = 0; i < la && ok; i++) {
                if (a[i] != b[i]) ok = false;
            }
        }
        printf("[ 9/%d] 同一权重跑两次结果一致                  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    free_model_map();
    printf("check_bind_weights: %d/%d passed\n", passed, total);
    return passed == total;
}

// ============================================================
//  check_pipeline() —— 文本进 -> 文本出
//
//  演示模型：层权重全 0（残差把 embed 原样带过去），lm_head 输出“上一个字节 + 1”，
//  所以输入 abcdefgh 必然得到 ijklmnopqrstuvwxy —— 可读、可断言
// ============================================================
bool check_pipeline() {
    namespace fs = std::filesystem;

    const int total = 10;
    int passed = 0;
    printf("==== 端到端（文本进 -> 文本出）测试 ====\n");

    const fs::path root = fs::path("_file") / "pipe_test";
    const fs::path src = root / "src";
    const fs::path dir = root / "model";
    std::error_code ec;
    fs::remove_all(root, ec);

    converter::DemoSpec spec;
    bool made = false;
    try {
        converter::make_demo_model(src.string(), spec);
        converter::Options opt;
        opt.sources.push_back((src / "model.safetensors").string());
        opt.out_dir = dir.string();
        opt.config_path = (src / "config.json").string();
        opt.tokenizer_dir = src.string();
        opt.verbose = false;
        converter::convert(opt);
        made = true;
    } catch (const std::exception& e) {
        printf("       ! 造/转模型失败: %s\n", e.what());
    }

    free_model_map();
    pipeline::LoadedModel m;
    std::string err;
    const bool loaded = made && pipeline::load_model_dir(m, dir.string(), 64, &err);
    if (!loaded) printf("       ! %s\n", err.empty() ? "装载失败" : err.c_str());

    // ---------- 1. 装载成功 ----------
    {
        const bool ok = loaded && m.refs.num_layers == spec.layers
            && m.refs.hidden == spec.hidden && m.refs.vocab_size == spec.vocab;
        printf("[ 1/%d] 装载模型目录成功                        %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 2. 词表大小 = 256 字节 + bos/eos/unk ----------
    {
        const bool ok = loaded && m.tokenizer.vocab_size() == 259;
        printf("[ 2/%d] 词表 %zu 个 token（256 字节 + 3 特殊）  %s\n", total,
               loaded ? m.tokenizer.vocab_size() : 0, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 3. 英文往返 ----------
    {
        const std::string text = "Hello, JieYu AI!";
        const std::string back = loaded ? m.tokenizer.decode(m.tokenizer.encode(text)) : "";
        const bool ok = loaded && back == text;
        printf("[ 3/%d] decode(encode(英文)) 还原原文            %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 4. 中文往返 ----------
    {
        const std::string text = "你好，世界！昆仑分词器。";
        const std::string back = loaded ? m.tokenizer.decode(m.tokenizer.encode(text)) : "";
        const bool ok = loaded && back == text;
        printf("[ 4/%d] decode(encode(中文)) 还原原文            %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 5. encode 出的是字节 id ----------
    {
        const std::vector<int32_t> ids = loaded ? m.tokenizer.encode("abc") : std::vector<int32_t>();
        const bool ok = loaded && ids.size() == 3 && ids[0] == 97 && ids[1] == 98 && ids[2] == 99;
        printf("[ 5/%d] encode(\"abc\") = {97,98,99}              %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 6. 端到端：abcdefgh -> ijklmnopqrstuvwx ----------
    {
        std::string out;
        const int produced = loaded ? pipeline::generate_text(m.tokenizer, "abcdefgh", m.refs, 16, 0.0f, out)
                                    : -1;
        const bool ok = loaded && out == "ijklmnopqrstuvwx" && produced == 16;
        printf("[ 6/%d] abcdefgh -> %s (%d 个 token)   %s\n", total, out.c_str(), produced,
               ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 7. 生成长度 = max_new ----------
    {
        std::string out;
        const int produced = loaded ? pipeline::generate_text(m.tokenizer, "abc", m.refs, 5, 0.0f, out)
                                    : -1;
        const bool ok = loaded && produced == 5 && out == "defgh";
        printf("[ 7/%d] max_new=5 就正好生成 5 个 (得到 %s)      %s\n", total, out.c_str(),
               ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 8. 两次结果一致 ----------
    {
        std::string a, b;
        if (loaded) {
            pipeline::generate_text(m.tokenizer, "xyz", m.refs, 8, 0.0f, a);
            pipeline::generate_text(m.tokenizer, "xyz", m.refs, 8, 0.0f, b);
        }
        const bool ok = loaded && !a.empty() && a == b;
        printf("[ 8/%d] 同一提示词两次结果一致 (得到 %s)        %s\n", total, a.c_str(),
               ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 9. 加不加 BOS 结果一样（BOS 行是零向量） ----------
    {
        std::string with_bos, without_bos;
        if (loaded) {
            pipeline::generate_text(m.tokenizer, "abc", m.refs, 4, 0.0f, with_bos, true);
            pipeline::generate_text(m.tokenizer, "abc", m.refs, 4, 0.0f, without_bos, false);
        }
        const bool ok = loaded && !with_bos.empty() && with_bos == without_bos;
        printf("[ 9/%d] 加不加 BOS 都不影响结果                  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    // ---------- 10. 超长提示词被 max_seq 截住，不越界不崩 ----------
    {
        std::string out;
        const std::string huge(200, 'k');
        const int produced = loaded ? pipeline::generate_text(m.tokenizer, huge, m.refs, 16, 0.0f, out)
                                    : -1;
        const bool ok = loaded && produced == 0 && out.empty();
        printf("[10/%d] 提示词 200 字符 > max_seq 64 -> 安全停下  %s\n", total, ok ? "OK" : "FAIL");
        if (ok) passed++;
    }

    m.free();
    printf("check_pipeline: %d/%d passed\n", passed, total);
    return passed == total;
}

#endif  // JIEYU_TEXT_TEXT_H