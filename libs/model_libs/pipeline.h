#ifndef PIPELINE_H
#define PIPELINE_H


#if defined(_WIN32)
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
    std::string s_start, s_end, s_text;
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

inline std::string read_info_str(const std::string& path,const std::string& key,const std::string& fallback){
    json_libs::json_lib j;
    if(!j.SetJsonWay(path)) return fallback;
    const std::string v=j.GetJsonData(key);
    return v.empty()?fallback:v;
}

inline uint64_t register_special(json_libs::Tokenizer& tok,const std::string& piece){
    if(piece.empty()) return 0;
    const int32_t id=tok.add_special_token(piece);
    return (id<0)?(uint64_t)0:(uint64_t)id;
}

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

inline size_t hold_back(const std::string& pending) {
    const size_t lt = pending.rfind('<');
    if (lt == std::string::npos) return 0;
    const size_t n = pending.size() - lt;
    return (n <= 24) ? n : 0;
}
inline size_t utf8_flushable(const std::string& s, size_t hold) {
    const size_t whole = utf8_whole_bytes(s);
    if (whole <= hold) return 0;
    size_t cut = whole - hold;
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0u) == 0x80u) cut--;
    return cut;
}

inline bool is_marker_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '-';
}

inline size_t marker_at(const std::string& s, const std::string* stops, int nstop) {
    size_t best = std::string::npos;
    for (int i = 0; i < nstop; i++) {
        if (stops[i].empty()) continue;
        const size_t at = s.find(stops[i]);
        if (at != std::string::npos && (best == std::string::npos || at < best)) best = at;
    }
    for (size_t lt = s.find("<|"); lt != std::string::npos; lt = s.find("<|", lt + 1)) {
        size_t n = 0;
        while (lt + 2 + n < s.size() && n < 24 &&
               is_marker_char(static_cast<unsigned char>(s[lt + 2 + n]))) {
            n++;
        }
        if (n >= 1 && (best == std::string::npos || lt < best)) best = lt;
    }
    return best;
}

struct EchoSink {
    const json_libs::Tokenizer* tok;
    std::string* out;
    std::ostream* echo;
    std::string pending;
    int32_t sp[3];
    int nsp;
    const std::string* stops;
    int nstop;
};
inline bool echo_token(int id, void* ud) {
    EchoSink* s = static_cast<EchoSink*>(ud);
    for (int i = 0; i < s->nsp; i++) {
        if (id == s->sp[i]) return true;
    }
    const std::vector<int32_t> one(1, id);
    const std::string piece = s->tok->decode(one);
    *s->out += piece;
    if (s->echo != nullptr) s->pending += piece;
    const size_t at = marker_at(*s->out, s->stops, s->nstop);
    if (at != std::string::npos) {
        s->out->erase(at);
        const size_t pm = marker_at(s->pending, s->stops, s->nstop);
        if (pm != std::string::npos) s->pending.erase(pm);
        return true;
    }
    if (s->echo == nullptr) return false;
    const size_t cut = utf8_flushable(s->pending, hold_back(s->pending));
    if (cut > 0) {
        *s->echo << s->pending.substr(0, cut) << std::flush;
        s->pending.erase(0, cut);
    }
    return false;
}

inline int generate_text(const json_libs::Tokenizer& tok, const std::string& prompt, const ModelRefs& m,
                         int max_new, float temperature, std::string& out,
                         bool add_bos = true, std::ostream* echo = nullptr,
                         float repetition_penalty = 1.0f) {
    std::vector<int32_t> ids = tok.encode(prompt);
    const int32_t bos = tok.bos_id();
    if (add_bos && bos >= 0 && bos < m.vocab_size) ids.insert(ids.begin(), bos);
    if (ids.empty()) {
        out.clear();
        return 0;
    }
    if (max_new <= 0) max_new = m.max_seq;
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
    int32_t eos_list[3];
    int num_eos = 0;
    auto want_eos = [&](long long id) {
        if (id < 0 || id >= m.vocab_size) return;
        for (int i = 0; i < num_eos; i++) {
            if (eos_list[i] == (int32_t)id) return;
        }
        if (num_eos < 3) eos_list[num_eos++] = (int32_t)id;
    };
    want_eos(tok.eos_id());
    want_eos((long long)model.s_token.text_end);
    want_eos((long long)model.s_token.meesage_end);
    const int32_t* eos_ptr = (num_eos > 0) ? eos_list : nullptr;
    out.clear();
    const std::string stops[3] = {m.s_start, m.s_end, m.s_text};
    EchoSink sink{&tok, &out, echo, std::string(), {0, 0, 0}, 0, stops, 3};
    const uint64_t spec[3] = {model.s_token.message_start, model.s_token.meesage_end,
                              model.s_token.text_end};
    for (int i = 0; i < 3; i++) {
        if (spec[i] != 0) sink.sp[sink.nsp++] = (int32_t)spec[i];
    }
    const int len = generate(in_ids.data(), prompt_len, eos_ptr, num_eos,
                             m.embed, m.lm_head, m.layers.data(), m.num_layers, m.final_norm,
                             m.rope.cos.data(), m.rope.sin.data(), m.max_seq, max_new,
                             temperature, m.vocab_size, m.hidden, m.num_heads, m.num_kv_heads,
                             m.head_dim, m.intermediate, out_ids.data(), repetition_penalty,
                             echo_token, &sink);
    if (echo != nullptr) {
        const size_t at = marker_at(*sink.out, stops, 3);
        if (at != std::string::npos) {
            sink.out->erase(at);
            const size_t pm = marker_at(sink.pending, stops, 3);
            if (pm != std::string::npos) sink.pending.erase(pm);
        }
        if (!sink.pending.empty()) *echo << sink.pending;
        *echo << std::endl;
    }
    return len - prompt_len;
}

struct LoadedModel {
    ModelRefs refs;
    json_libs::Tokenizer tokenizer;
    std::string dir;
    std::string s_start, s_end;
    std::string s_system;
    std::string s_name;

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
    const std::string info = model.model_info.string();
    out.s_start = read_info_str(info, "message_start", "");
    out.s_end = read_info_str(info, "message_end", "");
    out.refs.s_start = out.s_start;
    out.refs.s_end = out.s_end;
    out.refs.s_text = read_info_str(info, "text_end", "");
    out.s_system = read_info_str(info, "system", "");
    out.s_name = read_info_str(info, "source_name", "");
    model.s_token.message_start = register_special(out.tokenizer, out.s_start);
    model.s_token.meesage_end = register_special(out.tokenizer, out.s_end);
    model.s_token.text_end = register_special(out.tokenizer, read_info_str(info, "text_end", ""));
    return true;
}

inline bool utf8_valid(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t need = 1;
        if (c < 0x80u) need = 1;
        else if ((c & 0xE0u) == 0xC0u) need = 2;
        else if ((c & 0xF0u) == 0xE0u) need = 3;
        else if ((c & 0xF8u) == 0xF0u) need = 4;
        else return false;
        if (i + need > s.size()) return false;
        for (size_t k = 1; k < need; k++) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) return false;
        }
        i += need;
    }
    return true;
}

inline std::string acp_to_utf8(const std::string& text) {
#if defined(_WIN32)
    if (text.empty() || utf8_valid(text)) return text;
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

inline int run_chat(LoadedModel& m, int max_new, float temperature, float repetition_penalty = 1.1f) {
    const std::string& p_start = m.s_start;
    const std::string& p_end = m.s_end;
    auto has_base=[](const std::string& s)->bool{
        for(size_t i=0;i+4<=s.size();i++){
            const char c0=(char)((s[i]>='A'&&s[i]<='Z')?s[i]+32:s[i]);
            if(c0!='b') continue;
            const char c1=(char)((s[i+1]>='A'&&s[i+1]<='Z')?s[i+1]+32:s[i+1]);
            const char c2=(char)((s[i+2]>='A'&&s[i+2]<='Z')?s[i+2]+32:s[i+2]);
            const char c3=(char)((s[i+3]>='A'&&s[i+3]<='Z')?s[i+3]+32:s[i+3]);
            if(c1=='a'&&c2=='s'&&c3=='e') return true;
        }
        return false;
    };
    const bool looks_base = has_base(m.s_name) || has_base(m.dir);
    const bool templated = !looks_base && (p_start == "<|im_start|>" && p_end == "<|im_end|>");
    std::string sys = m.s_system;
    if (templated && sys.empty()) sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    const std::string sys_turn = templated ? (p_start + "system\n" + sys + p_end + "\n") : std::string();

    std::cout << "模型已装载：" << m.dir << "\n";
    std::cout << "  hidden=" << m.refs.hidden << " layers=" << m.refs.num_layers
              << " heads=" << m.refs.num_heads << " kv=" << m.refs.num_kv_heads
              << " head_dim=" << m.refs.head_dim << " vocab=" << m.refs.vocab_size << "\n";
    std::cout << "  词表 " << m.tokenizer.vocab_size() << " 个 token，最大长度 "
              << m.refs.max_seq;
    if (max_new > 0) {
        std::cout << "，每次最多生成 " << max_new << " 个 token";
    } else {
        std::cout << "，不限（生成到 <|im_end|> / 上下文满为止）";
    }
    std::cout << "\n";
    std::cout << "  对话模板 " << (templated ? "开（<|im_start|>…<|im_end|>）" : "关（纯续写）");
    if (looks_base) {
        std::cout << "\n  [注意] 这看着是 base 模型"
                  << (m.s_name.empty() ? std::string() : ("（" + m.s_name + "）"))
                  << "，它不会对话，只会续写文档 —— 想聊天请用 instruct 版的目录";
    }
    if (model.s_token.text_end != 0) {
        std::cout << "，碰到 text_end(" << model.s_token.text_end << ") / message_end("
                  << model.s_token.meesage_end << ") 会停";
    }
    std::cout << "\n输入提示词回车生成，输入 :q 退出\n" << std::flush;

    std::vector<std::string> turns;
    std::string line;
    while (true) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        std::string prompt = acp_to_utf8(line);
        while (!prompt.empty() && (prompt.back() == '\r' || prompt.back() == '\n')) {
            prompt.pop_back();
        }
        if (prompt == ":q" || prompt == ":quit" || prompt == ":exit") break;
        if (prompt.empty()) continue;
        std::string turn = prompt;
        if (templated) turn = p_start + "user\n" + prompt + p_end + "\n";
        std::string feed = sys_turn;
        for (const std::string& t : turns) feed += t;
        feed += turn;
        if (templated) feed += p_start + "assistant\n";
        std::cout << "  [给模型看 " << m.tokenizer.encode(feed).size() << " token，历史 "
                  << turns.size() << " 轮]" << std::endl;
        std::string out;
        const int produced = generate_text(m.tokenizer, feed, m.refs, max_new, temperature,
                                           out, m.refs.add_bos, &std::cout, repetition_penalty);
        if (produced < 0) {
            std::cout << "(生成失败)" << std::endl;
            continue;
        }
        if (templated && produced > 0) {
            turns.push_back(turn + p_start + "assistant\n" + out + p_end + "\n");
            while (turns.size() > 8) turns.erase(turns.begin());
        }
    }
    std::cout << std::endl;
    return 0;
}

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
