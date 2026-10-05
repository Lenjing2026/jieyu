#ifndef CONVERTER_H
#define CONVERTER_H

// ============================================================
//  converter.h —— safetensors -> model.bf（纯 C++，不需要 Python）
//
//  流程：
//    1. 读每个分片的 header，收集张量清单（名字 / dtype / shape / 偏移）
//    2. 超参优先取 config.json，缺了按张量形状推
//    3. 2D 权重转置（matmul 要 [k,n]，HF 给的是 [out,in]），一维和 embed 不动
//    4. 边读边转边写（bfile::write_bfile_stream），大张量分块流式，不整份占内存
//    5. 顺手写 model_info.json，并把分词器文件复制过去
// ============================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../json_libs/json.h"
#include "../tokenizer_libs/include/byte_vocab.h"
#include "bfile.h"
#include "safetensors.h"

namespace converter {

struct Options {
    std::vector<std::string> sources;   // 一个或多个 safetensors 分片
    std::string out_dir;                // 输出目录
    std::string config_path;            // config.json（可空）
    std::string tokenizer_dir;          // 分词器文件目录（可空）
    uint8_t out_dtype = bfile::F32;
    bool copy_tokenizer = true;
    bool verbose = true;
};

struct Result {
    bfile::Header head;
    int rope_theta = 10000;
    int max_position = 0;
    uint64_t tensor_count = 0;
    uint64_t data_off = 0;
    uint64_t file_size = 0;
    int transposed = 0;
    bool tied_lm_head = false;
};

inline int dtype_from_name(const std::string& name) {
    if (name == "f32") return bfile::F32;
    if (name == "f16") return bfile::F16;
    if (name == "bf16") return bfile::BF16;
    return -1;
}

// embed_tokens 是查表用的 [vocab, hidden]，不能转置；其余 2D 权重都要转
inline bool need_transpose(const std::string& name, const std::vector<uint32_t>& shape) {
    if (shape.size() != 2) return false;
    if (name.find("embed_tokens.weight") != std::string::npos) return false;
    return true;
}

inline uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 31) & 1u;
    const uint32_t exp = (bits >> 23) & 0xFFu;
    uint32_t man = bits & 0x7FFFFFu;
    if (exp == 0xFFu) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00u | (man != 0 ? 0x200u : 0u));
    }
    int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        man = (man | 0x800000u) >> static_cast<unsigned>(1 - e);
        man = (man + 0x1000u) >> 13;
        return static_cast<uint16_t>((sign << 15) | man);
    }
    uint32_t half_man = (man + 0x1000u) >> 13;
    if (half_man == 0x400u) {
        half_man = 0;
        e += 1;
        if (e >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(e) << 10) | half_man);
}

inline uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    if ((bits & 0x7F800000u) != 0x7F800000u) bits += 0x8000u;  // 四舍五入
    return static_cast<uint16_t>(bits >> 16);
}

// 把 f32 数据按目标 dtype 写出去
inline void write_as(std::ostream& f, const float* values, uint64_t count, uint8_t dtype) {
    if (dtype == bfile::F32) {
        f.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count * 4));
        return;
    }
    std::vector<unsigned char> out(static_cast<size_t>(count) * 2);
    for (uint64_t i = 0; i < count; i++) {
        const uint16_t h = (dtype == bfile::F16) ? float_to_half(values[i]) : float_to_bf16(values[i]);
        out[i * 2] = static_cast<unsigned char>(h & 0xFF);
        out[i * 2 + 1] = static_cast<unsigned char>(h >> 8);
    }
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

// 一个源张量 -> 一段输出（转置 / dtype 转换都在这里做）
inline bfile::StreamTensor make_sink(const safetensors::Info& src, bool transpose, uint8_t out_dtype) {
    bfile::StreamTensor sink;
    sink.name = src.name;
    sink.dtype = out_dtype;
    sink.shape = src.shape;
    if (transpose) sink.shape = {src.shape[1], src.shape[0]};
    sink.nbytes = safetensors::elems(sink.shape) * bfile::dtype_bytes(out_dtype);

    const safetensors::Info info = src;
    sink.emit = [info, transpose, out_dtype](std::ostream& f) -> bool {
        const uint64_t elem_bytes = bfile::dtype_bytes(info.dtype);
        std::ifstream in(info.src_path, std::ios::binary);
        if (!in) return false;

        if (!transpose) {
            // 不转置：分块读，内存只占一块
            const uint64_t chunk = (1u << 22) / elem_bytes;  // 约 4MB 一块
            std::vector<unsigned char> raw;
            std::vector<float> buf;
            uint64_t done = 0;
            const uint64_t total = safetensors::elems(info.shape);
            while (done < total) {
                const uint64_t n = std::min(chunk, total - done);
                const uint64_t bytes = n * elem_bytes;
                raw.resize(static_cast<size_t>(bytes));
                in.seekg(static_cast<std::streamoff>(info.begin + done * elem_bytes), std::ios::beg);
                in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
                if (static_cast<uint64_t>(in.gcount()) != bytes) return false;
                buf.resize(static_cast<size_t>(n));
                safetensors::to_f32(raw.data(), n, info.dtype, buf.data());
                write_as(f, buf.data(), n, out_dtype);
                done += n;
            }
            return static_cast<bool>(f);
        }

        const uint64_t rows = info.shape[0];
        const uint64_t cols = info.shape[1];
        std::vector<unsigned char> raw(static_cast<size_t>(rows * cols * elem_bytes));
        in.seekg(static_cast<std::streamoff>(info.begin), std::ios::beg);
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (static_cast<uint64_t>(in.gcount()) != raw.size()) return false;

        std::vector<float> src_f32(static_cast<size_t>(rows * cols));
        safetensors::to_f32(raw.data(), rows * cols, info.dtype, src_f32.data());
        std::vector<unsigned char>().swap(raw);
        std::vector<float> dst(static_cast<size_t>(rows * cols));
        const uint64_t ib = 32;
        for (uint64_t i0 = 0; i0 < rows; i0 += ib) {
            const uint64_t i1 = std::min(rows, i0 + ib);
            for (uint64_t j = 0; j < cols; j++) {
                const float* col = src_f32.data() + j;
                for (uint64_t i = i0; i < i1; i++) {
                    dst[static_cast<size_t>(j * rows + i)] = col[static_cast<size_t>(i) * cols];
                }
            }
        }
        write_as(f, dst.data(), rows * cols, out_dtype);
        return static_cast<bool>(f);
    };
    return sink;
}

// 二维权重跟 out_dtype，一维（norm / bias）永远 f32：它们太小，而且引擎按 f32 直接取
inline uint8_t sink_dtype(const std::vector<uint32_t>& shape,uint8_t want){
    return shape.size()==2?want:(uint8_t)bfile::F32;
}

// 从张量名字里数层数（model.layers.N.xxx）
inline int count_layers(const std::vector<safetensors::Info>& list) {
    int layers = 0;
    for (const safetensors::Info& t : list) {
        const std::string key = "model.layers.";
        if (t.name.compare(0, key.size(), key) != 0) continue;
        const size_t dot = t.name.find('.', key.size());
        if (dot == std::string::npos) continue;
        const int index = std::atoi(t.name.substr(key.size(), dot - key.size()).c_str());
        layers = std::max(layers, index + 1);
    }
    return layers;
}

inline void derive_head(const std::vector<safetensors::Info>& list, const json_libs::json_lib& cfg,
                        bfile::Header& head, int& rope_theta, int& max_position) {
    const safetensors::Info* embed = nullptr;
    for (const safetensors::Info& t : list) {
        if (t.name.find("embed_tokens.weight") != std::string::npos) {
            embed = &t;
            break;
        }
    }
    if (embed == nullptr || embed->shape.size() != 2) {
        throw std::runtime_error("converter: 找不到 embed_tokens 权重");
    }
    const auto layer0 = [&list](const std::string& suffix) -> const safetensors::Info* {
        const std::string name = "model.layers.0." + suffix;
        for (const safetensors::Info& t : list) {
            if (t.name == name) return &t;
        }
        return nullptr;
    };
    const safetensors::Info* q = layer0("self_attn.q_proj.weight");
    const safetensors::Info* k = layer0("self_attn.k_proj.weight");
    const safetensors::Info* gate = layer0("mlp.gate_proj.weight");

    const int hidden = cfg.GetJsonInt("hidden_size", static_cast<int>(embed->shape[1]));
    const int vocab_size = cfg.GetJsonInt("vocab_size", static_cast<int>(embed->shape[0]));
    const int layer_count = cfg.GetJsonInt("num_hidden_layers", count_layers(list));
    const int num_heads = cfg.GetJsonInt("num_attention_heads", 0);
    const int num_kv_heads = cfg.GetJsonInt("num_key_value_heads", num_heads);
    const int head_dim_cfg = cfg.GetJsonInt("head_dim", 0);
    const int intermediate = cfg.GetJsonInt("intermediate_size", gate ? static_cast<int>(gate->shape[0]) : 0);
    rope_theta = cfg.GetJsonInt("rope_theta", 10000);
    max_position = cfg.GetJsonInt("max_position_embeddings", 0);

    const int q_dim = q ? static_cast<int>(q->shape[0]) : 0;
    const int kv_dim = k ? static_cast<int>(k->shape[0]) : 0;
    int nh = num_heads;
    int nkv = num_kv_heads;
    int hd = head_dim_cfg;
    if (hd <= 0 && nh > 0 && q_dim > 0) hd = q_dim / nh;
    if (hd <= 0 && q_dim > 0) {
        const int candidates[] = {128, 64, 96, 80, 256, 32, 48, 112, 192};
        for (int c : candidates) {
            if (q_dim % c == 0) {
                hd = c;
                break;
            }
        }
        if (hd <= 0) hd = q_dim;
    }
    if (nh <= 0 && q_dim > 0 && hd > 0) nh = q_dim / hd;
    if (nkv <= 0 && kv_dim > 0 && hd > 0) nkv = kv_dim / hd;
    if (nkv <= 0) nkv = nh;

    if (hidden <= 0) throw std::runtime_error("converter: hidden_size 不合法");
    if (vocab_size <= 0) throw std::runtime_error("converter: vocab_size 不合法");
    if (intermediate <= 0) throw std::runtime_error("converter: 推不出 intermediate_size");
    if (layer_count <= 0) throw std::runtime_error("converter: 数不出层数（张量名字不是 HF 那套？）");
    if (nh <= 0 || nh > 255 || nkv <= 0 || nkv > 255 || hd <= 0 || hd > 255) {
        throw std::runtime_error("converter: num_heads / num_kv_heads / head_dim 必须落在 1..255（model.bf 里是 u8）");
    }
    if (nh % nkv != 0) {
        throw std::runtime_error("converter: num_heads 必须能被 num_kv_heads 整除");
    }
    if (hidden != nh * hd) {
        std::cout << "[警告] hidden_size(" << hidden << ") != num_heads*head_dim("
                  << nh * hd << ")，本工程的 attention 是按这个假设写的\n";
    }

    head.hidden = static_cast<uint64_t>(hidden);
    head.vocab_size = static_cast<uint64_t>(vocab_size);
    head.layers = static_cast<uint64_t>(layer_count);
    head.num_heads = static_cast<uint8_t>(nh);
    head.num_kv_heads = static_cast<uint8_t>(nkv);
    head.head_dim = static_cast<uint8_t>(hd);
    head.intermediate = static_cast<uint32_t>(intermediate);
    head.mode = cfg.GetJsonInt("mode", 1);
}

inline void copy_tokenizer_files(const std::string& from_dir, const std::string& to_dir,
                                 std::vector<std::string>& copied) {
    namespace fs = std::filesystem;
    const char* names[] = {"vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                           "special_tokens_map.json", "generation_config.json"};
    std::error_code ec;
    for (const char* name : names) {
        const fs::path src = fs::path(from_dir) / name;
        if (!fs::exists(src, ec)) continue;
        fs::copy_file(src, fs::path(to_dir) / name, fs::copy_options::overwrite_existing, ec);
        if (!ec) copied.push_back(name);
    }
}

inline Result convert(const Options& opt) {
    namespace fs = std::filesystem;
    if (opt.sources.empty()) throw std::runtime_error("converter: 没有输入文件");
    if (opt.out_dir.empty()) throw std::runtime_error("converter: 没有输出目录");

    std::vector<safetensors::Info> list;
    uint64_t source_bytes = 0;
    for (const std::string& path : opt.sources) {
        safetensors::File file;
        safetensors::open(path, file);
        source_bytes += file.size;
        for (const safetensors::Info& t : file.tensors) {
            for (const safetensors::Info& other : list) {
                if (other.name == t.name) {
                    throw std::runtime_error("converter: 多个分片里有同名张量: " + t.name);
                }
            }
            list.push_back(t);
        }
    }
    std::sort(list.begin(), list.end(),
              [](const safetensors::Info& a, const safetensors::Info& b) { return a.name < b.name; });

    json_libs::json_lib cfg;
    if (!opt.config_path.empty()) {
        if (!cfg.SetJsonWay(opt.config_path)) {
            std::cout << "[警告] config.json 读不了（" << cfg.LastError() << "），改按张量形状推超参\n";
        }
    }

    Result result;
    derive_head(list, cfg, result.head, result.rope_theta, result.max_position);

    std::vector<bfile::StreamTensor> sinks;
    sinks.reserve(list.size() + 1);
    for (const safetensors::Info& t : list) {
        sinks.push_back(make_sink(t, need_transpose(t.name, t.shape), sink_dtype(t.shape, opt.out_dtype)));
        if (need_transpose(t.name, t.shape)) result.transposed += 1;
    }
    std::sort(sinks.begin(), sinks.end(),
              [](const bfile::StreamTensor& a, const bfile::StreamTensor& b) { return a.name < b.name; });

    // 权重共享：没有 lm_head.weight 就补一份转置过的 embed
    if (std::find_if(sinks.begin(), sinks.end(), [](const bfile::StreamTensor& s) {
            return s.name == "lm_head.weight";
        }) == sinks.end()) {
        const safetensors::Info* embed = nullptr;
        for (const safetensors::Info& t : list) {
            if (t.name.find("embed_tokens.weight") != std::string::npos) {
                embed = &t;
                break;
            }
        }
        if (embed == nullptr || embed->shape.size() != 2) {
            throw std::runtime_error("converter: embed_tokens 不是 2D，补不出 lm_head");
        }
        // 转置后 embed 就是 [hidden, vocab]，正是 lm_head 要的形状
        bfile::StreamTensor sink = make_sink(*embed, true, sink_dtype(embed->shape, opt.out_dtype));
        sink.name = "lm_head.weight";
        sinks.push_back(sink);
        result.transposed += 1;
        result.tied_lm_head = true;
        std::sort(sinks.begin(), sinks.end(),
                  [](const bfile::StreamTensor& a, const bfile::StreamTensor& b) { return a.name < b.name; });
    }

    std::error_code ec;
    fs::create_directories(opt.out_dir, ec);
    if (ec) throw std::runtime_error("converter: 建不了输出目录 " + opt.out_dir);

    const std::string bf_path = (fs::path(opt.out_dir) / "model.bf").string();
    bfile::write_bfile_stream(bf_path, result.head, sinks);
    result.tensor_count = sinks.size();
    result.file_size = fs::file_size(bf_path, ec);

    // 写 model_info.json（复用工程自己的 JSON 写入器）
    {
        json_libs::json_lib info;
        const std::string info_path = (fs::path(opt.out_dir) / "model_info.json").string();
        info.SetJsonWay(info_path);
        info.WriteJsonKey("hidden_size", std::to_string(result.head.hidden));
        info.WriteJsonKey("layer_count", std::to_string(result.head.layers));
        info.WriteJsonKey("mode", std::to_string(result.head.mode));
        info.WriteJsonKey("num_heads", std::to_string(result.head.num_heads));
        info.WriteJsonKey("num_kv_heads", std::to_string(result.head.num_kv_heads));
        info.WriteJsonKey("head_dim", std::to_string(result.head.head_dim));
        info.WriteJsonKey("intermediate_size", std::to_string(result.head.intermediate));
        info.WriteJsonKey("vocab_size", std::to_string(result.head.vocab_size));
        info.WriteJsonKey("rope_theta", std::to_string(result.rope_theta));
        info.WriteJsonKey("max_position_embeddings", std::to_string(result.max_position));
        info.WriteJsonKey("dtype", bfile::dtype_name(opt.out_dtype));
    }

    std::vector<std::string> copied;
    if (opt.copy_tokenizer && !opt.tokenizer_dir.empty()) {
        copy_tokenizer_files(opt.tokenizer_dir, opt.out_dir, copied);
    }

    // 写完自己校一遍（不 mmap 大文件，只读头部与目录）
    bfile::Header check_head;
    std::vector<bfile::Entry> dir;
    uint64_t check_size = 0;
    bfile::read_header_file(bf_path, check_head, dir, check_size);
    if (dir.size() != sinks.size()) {
        throw std::runtime_error("converter: 写完了但目录项数量对不上");
    }
    {
        std::ifstream f(bf_path, std::ios::binary);
        unsigned char head[bfile::kHeaderSize] = {};
        f.read(reinterpret_cast<char*>(head), bfile::kHeaderSize);
        result.data_off = bfile::read_u32(head + 46);
    }

    if (opt.verbose) {
        std::cout << "已写出 " << bf_path << "\n";
        std::cout << "  源文件            " << opt.sources.size() << " 个，共 "
                  << (static_cast<double>(source_bytes) / 1048576.0) << " MB\n";
        std::cout << "  张量              " << result.tensor_count << " 个（其中 "
                  << result.transposed << " 个 2D 权重转置）\n";
        std::cout << "  数据区起点        " << result.data_off << " 字节\n";
        std::cout << "  文件大小          " << (static_cast<double>(result.file_size) / 1048576.0) << " MB\n";
        std::cout << "  超参              hidden=" << result.head.hidden
                  << " layers=" << result.head.layers
                  << " heads=" << static_cast<int>(result.head.num_heads)
                  << " kv=" << static_cast<int>(result.head.num_kv_heads)
                  << " head_dim=" << static_cast<int>(result.head.head_dim)
                  << " inter=" << result.head.intermediate
                  << " vocab=" << result.head.vocab_size << "\n";
        std::cout << "  rope_theta        " << result.rope_theta << "\n";
        std::cout << "  输出 dtype        " << bfile::dtype_name(opt.out_dtype) << "\n";
        if (result.tied_lm_head) {
            std::cout << "  权重共享          没有 lm_head.weight，已补一份转置副本\n";
        }
        if (!copied.empty()) {
            std::cout << "  已复制分词器文件  ";
            for (size_t i = 0; i < copied.size(); i++) {
                std::cout << (i ? ", " : "") << copied[i];
            }
            std::cout << "\n";
        }
        if (opt.out_dtype != bfile::F32) {
            std::cout << "  [注意] 当前引擎只读 f32，"
                      << bfile::dtype_name(opt.out_dtype) << " 还得等 matmul 支持\n";
        }
    }
    return result;
}

// ------------------------------------------------------------
//  造一个演示用小模型：层权重全 0，于是残差把 embed 原样带到底，
//  lm_head 设计成“输出上一个 token + 1”，所以
//      输入 abcdefgh -> 输出 ijklmnopqrstuvwxy（人能看懂）
// ------------------------------------------------------------
struct DemoSpec {
    int hidden = 256;
    int heads = 8;
    int head_dim = 32;
    int kv_heads = 2;
    int intermediate = 64;
    int layers = 2;
    int vocab = 259;   // 256 个字节 + bos/eos/unk
};

inline void make_demo_model(const std::string& src_dir, DemoSpec spec = DemoSpec(),
                            bool with_lm_head = true) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(src_dir, ec);
    if (ec) throw std::runtime_error("converter: 建不了目录 " + src_dir);

    const int q_dim = spec.heads * spec.head_dim;
    const int kv_dim = spec.kv_heads * spec.head_dim;
    std::vector<safetensors::RawTensor> tensors;

    const auto zeros = [](size_t n) { return std::vector<float>(n, 0.0f); };
    const auto ones = [](size_t n) { return std::vector<float>(n, 1.0f); };

    // embed：v < 256 时第 v 维为 1，其余（bos/eos/unk）整行 0
    std::vector<float> embed(static_cast<size_t>(spec.vocab) * spec.hidden, 0.0f);
    for (int v = 0; v < 256 && v < spec.vocab; v++) {
        embed[static_cast<size_t>(v) * spec.hidden + v] = 1.0f;
    }
    tensors.push_back(safetensors::make_f32("model.embed_tokens.weight",
                                            {static_cast<uint32_t>(spec.vocab),
                                             static_cast<uint32_t>(spec.hidden)}, embed));

    for (int l = 0; l < spec.layers; l++) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        tensors.push_back(safetensors::make_f32(prefix + "input_layernorm.weight",
                                                {static_cast<uint32_t>(spec.hidden)}, ones(spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "self_attn.q_proj.weight",
                                                {static_cast<uint32_t>(q_dim),
                                                 static_cast<uint32_t>(spec.hidden)}, zeros(static_cast<size_t>(q_dim) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "self_attn.k_proj.weight",
                                                {static_cast<uint32_t>(kv_dim),
                                                 static_cast<uint32_t>(spec.hidden)}, zeros(static_cast<size_t>(kv_dim) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "self_attn.v_proj.weight",
                                                {static_cast<uint32_t>(kv_dim),
                                                 static_cast<uint32_t>(spec.hidden)}, zeros(static_cast<size_t>(kv_dim) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "self_attn.o_proj.weight",
                                                {static_cast<uint32_t>(spec.hidden),
                                                 static_cast<uint32_t>(q_dim)}, zeros(static_cast<size_t>(q_dim) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "post_attention_layernorm.weight",
                                                {static_cast<uint32_t>(spec.hidden)}, ones(spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "mlp.gate_proj.weight",
                                                {static_cast<uint32_t>(spec.intermediate),
                                                 static_cast<uint32_t>(spec.hidden)}, zeros(static_cast<size_t>(spec.intermediate) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "mlp.up_proj.weight",
                                                {static_cast<uint32_t>(spec.intermediate),
                                                 static_cast<uint32_t>(spec.hidden)}, zeros(static_cast<size_t>(spec.intermediate) * spec.hidden)));
        tensors.push_back(safetensors::make_f32(prefix + "mlp.down_proj.weight",
                                                {static_cast<uint32_t>(spec.hidden),
                                                 static_cast<uint32_t>(spec.intermediate)}, zeros(static_cast<size_t>(spec.intermediate) * spec.hidden)));
    }

    tensors.push_back(safetensors::make_f32("model.norm.weight",
                                            {static_cast<uint32_t>(spec.hidden)}, ones(spec.hidden)));

    if (with_lm_head) {
        // row v 里第 v-1 列为 1（转置后就是“下一个字节”）
        std::vector<float> lm(static_cast<size_t>(spec.vocab) * spec.hidden, 0.0f);
        for (int v = 1; v < 256 && v < spec.vocab; v++) {
            lm[static_cast<size_t>(v) * spec.hidden + (v - 1)] = 1.0f;
        }
        tensors.push_back(safetensors::make_f32("lm_head.weight",
                                                {static_cast<uint32_t>(spec.vocab),
                                                 static_cast<uint32_t>(spec.hidden)}, lm));
    }

    safetensors::write((fs::path(src_dir) / "model.safetensors").string(), tensors);

    json_libs::json_lib cfg;
    cfg.SetJsonWay((fs::path(src_dir) / "config.json").string());
    cfg.WriteJsonKey("model_type", "jieyu-demo");
    cfg.WriteJsonKey("hidden_size", std::to_string(spec.hidden));
    cfg.WriteJsonKey("num_hidden_layers", std::to_string(spec.layers));
    cfg.WriteJsonKey("num_attention_heads", std::to_string(spec.heads));
    cfg.WriteJsonKey("num_key_value_heads", std::to_string(spec.kv_heads));
    cfg.WriteJsonKey("head_dim", std::to_string(spec.head_dim));
    cfg.WriteJsonKey("intermediate_size", std::to_string(spec.intermediate));
    cfg.WriteJsonKey("vocab_size", std::to_string(spec.vocab));
    cfg.WriteJsonKey("rope_theta", "10000");
    cfg.WriteJsonKey("max_position_embeddings", "1024");

    json_libs::write_byte_level_vocab((fs::path(src_dir) / "vocab.json").string(),
                                      (fs::path(src_dir) / "merges.txt").string());
}

inline void show(const std::string& path) {
    bfile::Header head;
    std::vector<bfile::Entry> dir;
    uint64_t size = 0;
    bfile::read_header_file(path, head, dir, size);
    std::printf("文件      %s (%.2f MB)\n", path.c_str(), static_cast<double>(size) / 1048576.0);
    std::printf("张量      %zu 个\n", dir.size());
    std::printf("超参      hidden=%llu layers=%llu heads=%u kv=%u head_dim=%u inter=%u vocab=%llu\n",
                static_cast<unsigned long long>(head.hidden),
                static_cast<unsigned long long>(head.layers),
                static_cast<unsigned>(head.num_heads), static_cast<unsigned>(head.num_kv_heads),
                static_cast<unsigned>(head.head_dim), head.intermediate,
                static_cast<unsigned long long>(head.vocab_size));
    std::printf("%-40s %-4s %-13s %-8s %-8s\n", "名字", "类型", "形状", "偏移", "字节");
    for (const bfile::Entry& e : dir) {
        std::string shape;
        for (uint32_t d = 0; d < e.ndim; d++) {
            shape += std::to_string(e.shape[d]);
            if (d + 1 < e.ndim) shape += "x";
        }
        std::printf("%-40s %-4s %-13s %-8llu %-8llu\n", e.name, bfile::dtype_name(e.dtype),
                    shape.c_str(), static_cast<unsigned long long>(e.offset),
                    static_cast<unsigned long long>(e.nbytes));
    }
}

}  // namespace converter

#endif  // CONVERTER_H
