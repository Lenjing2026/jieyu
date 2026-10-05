#ifndef PIPELINE_H
#define PIPELINE_H

// ============================================================
//  pipeline.h —— 端到端：文本 -> token -> 自回归生成 -> 文本
//
//  1. install_model(dir)           装载 model.bf（拿到 mmap 与张量表）
//  2. bind_model(refs, max_seq)    按名字把权重指针填进 layers[]
//  3. tokenizer.load(vocab,merges) 加载词表
//  4. generate_text(...)           encode -> generate -> decode
// ============================================================

#if defined(_WIN32)
// windows.h 必须排在任何 "using namespace std;" 之前，否则 C++17 的 std::byte
// 会和 rpcndr.h 的 byte 撞名
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include "../date_libs/date.h"
#include "../json_libs/json.h"
#include "../tokenizer_libs/include/tokenizer.h"
#include "generate.h"
#include "install_model.h"
#include "load_weights.h"

namespace pipeline {

struct ModelRefs {
    WMat embed;
    WMat lm_head;
    const float* final_norm = nullptr;
    std::vector<LayerWeights> layers;
    RoPETable rope;
    int num_layers = 0, hidden = 0, num_heads = 0, num_kv_heads = 0;
    int head_dim = 0, intermediate = 0, vocab_size = 0, max_seq = 0;
    bool qkv_bias = false;
    bool add_bos = true;
    int q_dim() const { return num_heads * head_dim; }
    int kv_dim() const { return num_kv_heads * head_dim; }
    void free() {
        layers.clear();
        rope.free();
    }
};

inline int read_info_int(const std::filesystem::path& info_path, const char* key, int fallback) {
    json_libs::json_lib j;
    if (!j.SetJsonWay(info_path.string())) return fallback;
    return j.GetJsonInt(key, fallback);
}

inline float read_rope_theta(const std::filesystem::path& info_path) {
    return static_cast<float>(read_info_int(info_path, "rope_theta", 10000));
}

inline bool bind_model(ModelRefs& m, int max_seq = 1024, std::string* error = nullptr) {
    auto fail = [&](const std::string& msg) {
        if (error != nullptr) *error = msg;
        return false;
    };
    if (!bfile::current().ok) {
        return fail("model.bf 没有张量目录（v2），请先用 jieyu --convert 把 safetensors 转过来");
    }
    m.num_layers = static_cast<int>(model.l);
    m.hidden = static_cast<int>(model.h);
    m.num_heads = model.num_heads;
    m.num_kv_heads = model.num_kv_heads;
    m.head_dim = model.head_dim;
    m.intermediate = static_cast<int>(model.intermediate_size);
    m.vocab_size = static_cast<int>(model.vocab_size);
    if (m.num_layers <= 0 || m.hidden <= 0 || m.head_dim <= 0 || m.num_heads <= 0) {
        return fail("model.bf 头部超参不合法");
    }
    m.max_seq = (max_seq > 0) ? max_seq : 1024;

    m.layers.assign(static_cast<size_t>(m.num_layers), LayerWeights{});
    std::string missing;
    if (!bind_layers(m.layers.data(), m.num_layers, &missing)) {
        return fail("缺少张量: " + missing);
    }
    m.qkv_bias = bind_biases(m.layers.data(), m.num_layers);
    m.embed = bind_embed();
    if (m.embed.empty()) return fail("缺少张量: model.embed_tokens.weight");
    m.lm_head = bind_lm_head(m.hidden, m.vocab_size);
    if (m.lm_head.empty()) {
        return fail("lm_head.weight 缺失或形状不是 [hidden, vocab]（权重共享时需要转换器补一份转置副本）");
    }
    m.final_norm = bind_final_norm();
    if (m.final_norm == nullptr) return fail("缺少张量: model.norm.weight");

    const int q = m.q_dim();
    const int kv = m.kv_dim();
    struct Expect {
        std::string name;
        std::vector<int64_t> shape;
    };
    // 普通浮点权重的内存是 [in,out]（转换时转置过）；块量化是 [out,in]（行就是输出）
    auto want_w = [&](const std::string& name, int in, int out) -> std::vector<int64_t> {
        const bfile::Entry* t = find_tensor(name);
        if (t != nullptr && bfile::block_elems(t->dtype) != 0) return {out, in};
        return {in, out};
    };
    const Expect checks[] = {
        {"model.embed_tokens.weight", {m.vocab_size, m.hidden}},
        {"model.norm.weight", {m.hidden}},
        {layer_tensor_name(0, "input_layernorm.weight"), {m.hidden}},
        {layer_tensor_name(0, "self_attn.q_proj.weight"), want_w(layer_tensor_name(0, "self_attn.q_proj.weight"), m.hidden, q)},
        {layer_tensor_name(0, "self_attn.k_proj.weight"), want_w(layer_tensor_name(0, "self_attn.k_proj.weight"), m.hidden, kv)},
        {layer_tensor_name(0, "self_attn.v_proj.weight"), want_w(layer_tensor_name(0, "self_attn.v_proj.weight"), m.hidden, kv)},
        {layer_tensor_name(0, "self_attn.o_proj.weight"), want_w(layer_tensor_name(0, "self_attn.o_proj.weight"), q, m.hidden)},
        {layer_tensor_name(0, "mlp.gate_proj.weight"), want_w(layer_tensor_name(0, "mlp.gate_proj.weight"), m.hidden, m.intermediate)},
        {layer_tensor_name(0, "mlp.up_proj.weight"), want_w(layer_tensor_name(0, "mlp.up_proj.weight"), m.hidden, m.intermediate)},
        {layer_tensor_name(0, "mlp.down_proj.weight"), want_w(layer_tensor_name(0, "mlp.down_proj.weight"), m.intermediate, m.hidden)},
    };
    for (const Expect& e : checks) {
        if (shape_is(e.name, e.shape)) continue;
        const bfile::Entry* t = find_tensor(e.name);
        if (t == nullptr) return fail("缺少张量: " + e.name);
        std::string got = "[";
        for (uint32_t d = 0; d < t->ndim; d++) {
            got += std::to_string(t->shape[d]);
            if (d + 1 < t->ndim) got += ",";
        }
        got += "]";
        std::string want = "[";
        for (size_t d = 0; d < e.shape.size(); d++) {
            want += std::to_string(e.shape[d]);
            if (d + 1 < e.shape.size()) want += ",";
        }
        want += "]";
        return fail(e.name + " 形状是 " + got + "，期望 " + want + "（2D 权重需要转置）");
    }

    const Expect bias_checks[] = {
        {layer_tensor_name(0, "self_attn.q_proj.bias"), {q}},
        {layer_tensor_name(0, "self_attn.k_proj.bias"), {kv}},
        {layer_tensor_name(0, "self_attn.v_proj.bias"), {kv}},
    };
    for (const Expect& e : bias_checks) {
        if (find_tensor(e.name) == nullptr) continue;
        if (!shape_is(e.name, e.shape)) return fail("张量形状不对: " + e.name);
    }

    m.add_bos = read_info_int(model.model_info, "add_bos_token", 1) != 0;
    m.rope.init(m.max_seq, m.head_dim, read_rope_theta(model.model_info));
    return true;
}

// s 末尾是否正好停在一个完整 UTF-8 字符上；返回可以安全输出的字节数
inline size_t utf8_whole_bytes(const std::string& s) {
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t need = 1;
        if (c >= 0xF0u) need = 4;
        else if (c >= 0xE0u) need = 3;
        else if (c >= 0xC0u) need = 2;
        if (i + need > n) return i;
        i += need;
    }
    return n;
}

// 流式输出用：每来一个 token 就解码、拼 out、能凑成完整 UTF-8 就立刻吐给 echo
struct EchoSink {
    const json_libs::Tokenizer* tok;
    std::string* out;
    std::ostream* echo;
    std::string pending;
};
inline void echo_token(int id, void* ud) {
    EchoSink* s = static_cast<EchoSink*>(ud);
    const std::vector<int32_t> one(1, id);
    const std::string piece = s->tok->decode(one);
    *s->out += piece;
    if (s->echo == nullptr) return;
    s->pending += piece;
    const size_t whole = utf8_whole_bytes(s->pending);
    if (whole > 0) {
        *s->echo << s->pending.substr(0, whole) << std::flush;
        s->pending.erase(0, whole);
    }
}

// 返回新生成的 token 个数；出错返回 -1
inline int generate_text(const json_libs::Tokenizer& tok, const std::string& prompt, const ModelRefs& m,
                         int max_new, float temperature, std::string& out,
                         bool add_bos = true, std::ostream* echo = nullptr,
                         float repetition_penalty = 1.0f) {
    std::vector<int32_t> ids = tok.encode(prompt);
    const int32_t bos = tok.bos_id();
    if (add_bos && bos >= 0 && bos < m.vocab_size) ids.insert(ids.begin(), bos);
    if (ids.empty() || max_new <= 0) {
        out.clear();
        return 0;
    }
    for (int32_t id : ids) {
        if (id < 0 || id >= m.vocab_size) {
            if (echo != nullptr) {
                *echo << "[pipeline] token id " << id << " 超出词表 " << m.vocab_size
                      << "（词表与模型不匹配）" << std::endl;
            }
            return -1;
        }
    }

    int prompt_len = static_cast<int>(ids.size());
    if (prompt_len > m.max_seq) {
        prompt_len = m.max_seq;
        ids.resize(static_cast<size_t>(prompt_len));
    }
    if (prompt_len + max_new > m.max_seq) max_new = m.max_seq - prompt_len;
    if (max_new <= 0) {
        out.clear();
        return 0;
    }

    std::vector<int> in_ids(ids.begin(), ids.end());
    std::vector<int> out_ids(static_cast<size_t>(m.max_seq));
    const int32_t eos = tok.eos_id();
    const int32_t eos_list[1] = {eos};
    const int32_t* eos_ptr = (eos >= 0 && eos < m.vocab_size) ? eos_list : nullptr;
    out.clear();
    EchoSink sink{&tok, &out, echo, std::string()};
    const int len = generate(in_ids.data(), prompt_len, eos_ptr, (eos_ptr != nullptr) ? 1 : 0,
                             m.embed, m.lm_head, m.layers.data(), m.num_layers, m.final_norm,
                             m.rope.cos.data(), m.rope.sin.data(), m.max_seq, max_new,
                             temperature, m.vocab_size, m.hidden, m.num_heads, m.num_kv_heads,
                             m.head_dim, m.intermediate, out_ids.data(), repetition_penalty,
                             echo_token, &sink);
    if (echo != nullptr) {
        if (!sink.pending.empty()) *echo << sink.pending;
        *echo << std::endl;
    }
    return len - prompt_len;
}

// ------------------------------------------------------------
//  从一个模型目录把整套东西装好（model.bf + 分词器）
// ------------------------------------------------------------
struct LoadedModel {
    ModelRefs refs;
    json_libs::Tokenizer tokenizer;
    std::string dir;

    void free() {
        refs.free();
        free_model_map();
    }
};

inline bool load_model_dir(LoadedModel& out, const std::string& dir, int max_seq = 1024,
                           std::string* error = nullptr) {
    auto fail = [&](const std::string& msg) {
        if (error != nullptr) *error = msg;
        return false;
    };
    try {
        out.dir = dir;
        install_model(std::filesystem::path(dir));
    } catch (const std::exception& e) {
        return fail(std::string("装载 model.bf 失败: ") + e.what());
    }
    if (!bind_model(out.refs, max_seq, error)) {
        return false;
    }
    const std::filesystem::path base(dir);
    const std::string vocab = (base / "vocab.json").string();
    const std::string merges = (base / "merges.txt").string();
    if (!out.tokenizer.load(vocab, merges)) {
        return fail("分词器加载失败: " + vocab + " / " + merges);
    }
    if (static_cast<int>(out.tokenizer.vocab_size()) > out.refs.vocab_size) {
        std::cerr << "[pipeline] 警告：词表有 " << out.tokenizer.vocab_size()
                  << " 个 token，而模型只有 " << out.refs.vocab_size
                  << " 行 embedding，超出的会报错" << std::endl;
    }
    return true;
}

// ------------------------------------------------------------
//  Windows 控制台编码：命令行参数/标准输入是 ANSI(GBK)，转成 UTF-8
// ------------------------------------------------------------
inline std::string acp_to_utf8(const std::string& text) {
#if defined(_WIN32)
    if (text.empty()) return text;
    const int wide_len = MultiByteToWideChar(CP_ACP, 0, text.c_str(),
                                             static_cast<int>(text.size()), nullptr, 0);
    if (wide_len <= 0) return text;
    std::wstring wide(static_cast<size_t>(wide_len), L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.c_str(), static_cast<int>(text.size()),
                        wide.data(), wide_len);
    const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wide_len,
                                             nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 0) return text;
    std::string utf8(static_cast<size_t>(utf8_len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wide_len, utf8.data(), utf8_len,
                        nullptr, nullptr);
    return utf8;
#else
    return text;
#endif
}

// ------------------------------------------------------------
//  交互式：一行一行读，生成一段就打印一段
// ------------------------------------------------------------
inline int run_chat(LoadedModel& m, int max_new, float temperature, float repetition_penalty = 1.1f) {
    std::cout << "模型已装载：" << m.dir << "\n";
    std::cout << "  hidden=" << m.refs.hidden << " layers=" << m.refs.num_layers
              << " heads=" << m.refs.num_heads << " kv=" << m.refs.num_kv_heads
              << " head_dim=" << m.refs.head_dim << " vocab=" << m.refs.vocab_size << "\n";
    std::cout << "  词表 " << m.tokenizer.vocab_size() << " 个 token，最大长度 "
              << m.refs.max_seq << "，每次最多生成 " << max_new << " 个 token\n";
    std::cout << "输入提示词回车生成，输入 :q 退出\n" << std::flush;

    std::string line;
    while (true) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        const std::string prompt = acp_to_utf8(line);
        if (prompt == ":q" || prompt == ":quit" || prompt == ":exit") break;
        if (prompt.empty()) continue;
        std::string out;
        const int produced = generate_text(m.tokenizer, prompt, m.refs, max_new, temperature,
                                           out, m.refs.add_bos, &std::cout, repetition_penalty);
        if (produced < 0) {
            std::cout << "(生成失败)" << std::endl;
        }
    }
    std::cout << std::endl;
    return 0;
}

// 一次性：给一句提示词，生成完打印
inline int run_once(LoadedModel& m, const std::string& prompt, int max_new, float temperature,
                    float repetition_penalty = 1.1f) {
    std::string out;
    const int produced = generate_text(m.tokenizer, prompt, m.refs, max_new, temperature,
                                       out, m.refs.add_bos, &std::cout, repetition_penalty);
    if (produced <= 0) {
        std::cout << "(没有生成任何 token)" << std::endl;
        return produced;
    }
    return 0;
}

}  // namespace pipeline

#endif  // PIPELINE_H
