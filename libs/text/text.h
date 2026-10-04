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
//  model.bf 布局（小端）：
//    [0..5]   魔数 "JYAIBF"
//    [6..13]  hidden_size       (uint64)
//    [14..21] layer_count       (uint64)
//    [22]     mode              (uint8)
//    [23]     num_heads         (uint8)
//    [24]     num_kv_heads      (uint8)
//    [25]     head_dim          (uint8)
//    [26..29] intermediate_size (uint32)
//    [30..37] vocab_size        (uint64)
//    [38..63] 零填充，补齐到 64 字节
//    合计 64 字节
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
    bf_head()
        : ok(false), size(0), magic_ok(false), pad_zero_ok(false), h(0), l(0),
          mode(0), num_heads(0), num_kv_heads(0), head_dim(0),
          intermediate_size(0), vocab_size(0) {}

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

    head.pad_zero_ok = true;
    for (size_t i = kBfHeaderSize; i < bytes.size(); i++) {
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

    const int total = 17;
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
        printf("[ 2/%d] 文件大小 %zu 字节 (38 有效 + 补齐到 64)  %s\n", total, head.size, got ? "OK" : "FAIL");
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
        printf("[10/%d] 补齐字节 38..63 全为 0                    %s\n", total, got ? "OK" : "FAIL");
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

    // ---------- 证据：model.bf 前 38 字节的十六进制 ----------
    {
        std::string bytes;
        if (text_detail::read_bytes((dir_ok / "model.bf").string(), bytes)) {
            printf("证据: model_ok/model.bf 前 38 字节 =");
            for (size_t i = 0; i < bytes.size() && i < 38; i++) {
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

    std::vector<float> X = X0;
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

#endif  // JIEYU_TEXT_TEXT_H