#ifndef JIEYU_TEXT_TEXT_H
#define JIEYU_TEXT_TEXT_H

// ============================================================
//  libs/text/text.h —— 自测函数集合
//    check_matmul()  matmul 系列（朴素 / AVX / FMA / AVX+FMA）正确性
//    check_bfile()   model.bf 二进制格式（date_libs/date.h + model_libs/creater_model.h）
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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <iterator>
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

        matmul_choose(m, n, k, A, B, C_got);

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
//    libs/date_libs/date.h            __model 结构体（h / l / mode / 路径 / 映射句柄）
//    libs/model_libs/creater_model.h  create_modle() / create_bfile()
//
//  model.bf 布局（小端）：
//    [0..5]   魔数 "JYAIBF"
//    [6..13]  hidden_size  (uint64)
//    [14..21] layer_count  (uint64)
//    [22]     mode         (uint8)
//    合计 23 字节
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

// model.bf 头部解析结果
struct bf_head {
    bool ok;          // 文件足够长、解析成功
    size_t size;      // 文件字节数
    bool magic_ok;    // 前 6 字节 == "JYAIBF"
    uint64_t h;       // hidden_size
    uint64_t l;       // layer_count
    uint8_t mode;     // mode
    bf_head() : ok(false), size(0), magic_ok(false), h(0), l(0), mode(0) {}
};

// 读出 dir/model.bf 的头部
bf_head read_bf_head(const std::filesystem::path& dir) {
    bf_head head;
    std::string bytes;
    if (!read_bytes((dir / "model.bf").string(), bytes)) {
        return head;
    }
    head.size = bytes.size();
    if (bytes.size() < 23) {
        return head;
    }
    const unsigned char* p = reinterpret_cast<const unsigned char*>(bytes.data());
    head.magic_ok = (std::memcmp(p, "JYAIBF", 6) == 0);
    head.h = read_u64_le(p + 6);
    head.l = read_u64_le(p + 14);
    head.mode = p[22];
    head.ok = true;
    return head;
}

// 用工程的 json_lib 写一份 model_info.json
bool write_model_info(const std::filesystem::path& dir, int hidden, int layers, int mode) {
    json_lib info;
    info.SetJsonWay((dir / "model_info.json").string());
    bool ok = info.WriteJsonKey("hidden_size", std::to_string(hidden));
    ok = info.WriteJsonKey("layer_count", std::to_string(layers)) && ok;
    ok = info.WriteJsonKey("mode", std::to_string(mode)) && ok;
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

    const int total = 10;
    int passed = 0;

    printf("==== model.bf 新功能测试 ====\n");

    const fs::path root         = fs::path("_file") / "bfile_test";
    const fs::path dir_ok       = root / "model_ok";
    const fs::path dir_nojson   = root / "model_nojson";
    const fs::path dir_badlayer = root / "model_badlayer";
    const fs::path dir_edge     = root / "model_edge";
    const fs::path dir_diag     = root / "model_diag";

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

    // ---------- 2~6. 正常写入 + 逐字段校验 ----------
    {
        write_model_info(dir_ok, 512, 16, 1);
        const text_detail::call_result r = call_create_bfile(dir_ok);
        const bf_head head = read_bf_head(dir_ok);

        if (r.threw) {
            printf("       ! create_bfile 抛异常: %s\n", r.message.c_str());
        }

        bool got = head.ok && head.size == 23;
        printf("[ 2/%d] 文件大小 %zu 字节 (期望 23)             %s\n", total, head.size, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.magic_ok;
        printf("[ 3/%d] 魔数 \"JYAIBF\"                         %s\n", total, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.h == 512;
        printf("[ 4/%d] hidden_size 512 (读回 %llu)             %s\n", total, (unsigned long long)head.h, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.l == 16;
        printf("[ 5/%d] layer_count 16 (读回 %llu)              %s\n", total, (unsigned long long)head.l, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        got = head.ok && head.mode == 1;
        printf("[ 6/%d] mode 1 (读回 %u)                        %s\n", total, (unsigned)head.mode, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
    }

    // ---------- 7. 没有 model_info.json ----------
    {
        fs::create_directories(dir_nojson, ec);
        const text_detail::call_result r = call_create_bfile(dir_nojson);
        const bool got = r.threw && r.message == "NO_JSONFILE";
        printf("[ 7/%d] 缺 model_info.json -> 抛 NO_JSONFILE    %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", r.threw ? r.message.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
        if (fs::exists(dir_nojson / "model.bf")) {
            printf("       ! 失败路径仍留下了 model.bf（%llu 字节）\n",
                   (unsigned long long)fs::file_size(dir_nojson / "model.bf", ec));
        }
    }

    // ---------- 8. layer_count 不是 16 的倍数 ----------
    {
        fs::create_directories(dir_badlayer, ec);
        write_model_info(dir_badlayer, 512, 12, 1);
        const text_detail::call_result r = call_create_bfile(dir_badlayer);
        const bool got = r.threw && r.message == "NO16_LAYER";
        printf("[ 8/%d] layer_count=12 -> 抛 NO16_LAYER        %s   [实际: %s]\n",
               total, got ? "OK" : "FAIL", r.threw ? r.message.c_str() : "没抛异常");
        if (got) {
            passed++;
        }
    }

    // ---------- 9. 边界值 h=4096 / l=32 / mode=255 ----------
    {
        fs::create_directories(dir_edge, ec);
        write_model_info(dir_edge, 4096, 32, 255);
        call_create_bfile(dir_edge);
        const bf_head head = read_bf_head(dir_edge);
        const bool got = head.ok && head.h == 4096 && head.l == 32 && head.mode == 255;
        printf("[ 9/%d] 边界 4096/32/255 (读回 %llu/%llu/%u)    %s\n",
               total, (unsigned long long)head.h, (unsigned long long)head.l,
               (unsigned)head.mode, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }
    }

    // ---------- 10. 内存映射读回（__model 的 model_handle / model_map 用途） ----------
    {
        const fs::path bf = dir_edge / "model.bf";
        HANDLE file_handle = CreateFileW(bf.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        HANDLE map_handle = nullptr;
        LPVOID view = nullptr;
        bool got = false;
        uint64_t map_h = 0;
        uint64_t map_l = 0;
        uint8_t map_mode = 0;

        if (file_handle != INVALID_HANDLE_VALUE) {
            map_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (map_handle != nullptr) {
                view = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
                if (view != nullptr) {
                    const unsigned char* p = static_cast<const unsigned char*>(view);
                    map_h = text_detail::read_u64_le(p + 6);
                    map_l = text_detail::read_u64_le(p + 14);
                    map_mode = p[22];

                    // 顺便验证 __model 的各字段能装下这些数据
                    model.h = static_cast<long long>(map_h);
                    model.l = static_cast<long long>(map_l);
                    model.mode = map_mode;
                    model.model_info = dir_edge / "model_info.json";
                    model.model_vector = bf;
                    model.model_handle = map_handle;
                    model.model_map = view;

                    got = (model.h == 4096 && model.l == 32 && model.mode == 255);
                }
            }
        }

        printf("[10/%d] 内存映射读回 %llu/%llu/%u               %s\n",
               total, (unsigned long long)map_h, (unsigned long long)map_l,
               (unsigned)map_mode, got ? "OK" : "FAIL");
        if (got) {
            passed++;
        }

        // 测试收尾：解除映射，不留悬挂指针
        if (view != nullptr) {
            UnmapViewOfFile(view);
            model.model_map = nullptr;
        }
        if (map_handle != nullptr) {
            CloseHandle(map_handle);
            model.model_handle = nullptr;
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
    }

    // ---------- 诊断：越界值会被静默截断 ----------
    {
        fs::create_directories(dir_diag, ec);
        write_model_info(dir_diag, -1, 16, 300);
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

    // ---------- 证据：model.bf 前 23 字节的十六进制 ----------
    {
        std::string bytes;
        if (text_detail::read_bytes((dir_ok / "model.bf").string(), bytes)) {
            printf("证据: model_ok/model.bf 前 23 字节 =");
            for (size_t i = 0; i < bytes.size() && i < 23; i++) {
                printf(" %02X", (unsigned char)bytes[i]);
            }
            printf("\n");
        }
    }

    printf("check_bfile: %d/%d passed\n", passed, total);
    return passed == total;
}

#endif  // JIEYU_TEXT_TEXT_H