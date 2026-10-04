#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

// ============================================================
//  load_weights.h —— 按名字从 model.bf 里取权重指针
//
//  命名沿用 HF / Qwen / Llama 那一套：
//    model.embed_tokens.weight
//    model.layers.{i}.input_layernorm.weight
//    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//    model.layers.{i}.post_attention_layernorm.weight
//    model.layers.{i}.mlp.{gate,up,down}_proj.weight
//    model.norm.weight
//    lm_head.weight
//
//  注意：本工程的 matmul 是 C[m,n]=A[m,k]*B[k,n]，B 取 [k,n]；
//  而 HF 的 nn.Linear 权重是 [out,in]=[n,k]，所以转换器要把 2D 权重转置。
//  一维的 norm 权重不用转置，embed_tokens 也不用。
// ============================================================

#include <string>
#include <vector>
#include "bfile.h"
#include "../date_libs/date.h"

inline const bfile::Entry* find_tensor(const std::string& name) {
    return bfile::current().find(name.c_str());
}

inline bool has_tensor(const std::string& name) {
    return find_tensor(name) != nullptr;
}

inline const float* tensor_f32(const std::string& name) {
    const bfile::Entry* e = find_tensor(name);
    if (e == nullptr || e->dtype != bfile::F32) return nullptr;
    return bfile::current().f32(e);
}

inline std::string layer_tensor_name(int layer, const char* suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

// 形状校验：want 里 -1 表示该维不检查
inline bool shape_is(const std::string& name, std::vector<int64_t> want) {
    const bfile::Entry* e = find_tensor(name);
    if (e == nullptr) return false;
    if (want.size() != e->ndim) return false;
    for (size_t i = 0; i < want.size(); i++) {
        if (want[i] >= 0 && static_cast<uint64_t>(want[i]) != e->shape[i]) return false;
    }
    return true;
}

// 9 个权重按 LayerWeights 的顺序排好
inline const char* layer_suffix(int index) {
    switch (index) {
        case 0: return "input_layernorm.weight";
        case 1: return "self_attn.q_proj.weight";
        case 2: return "self_attn.k_proj.weight";
        case 3: return "self_attn.v_proj.weight";
        case 4: return "self_attn.o_proj.weight";
        case 5: return "post_attention_layernorm.weight";
        case 6: return "mlp.gate_proj.weight";
        case 7: return "mlp.down_proj.weight";
        case 8: return "mlp.up_proj.weight";
        default: return nullptr;
    }
}

inline void set_layer_slot(LayerWeights& w, int index, const float* p) {
    switch (index) {
        case 0: w.rms1_weight = p; break;
        case 1: w.Wq = p; break;
        case 2: w.Wk = p; break;
        case 3: w.Wv = p; break;
        case 4: w.Wo = p; break;
        case 5: w.rms2_weight = p; break;
        case 6: w.W1 = p; break;
        case 7: w.W2 = p; break;
        case 8: w.W3 = p; break;
        default: break;
    }
}

// 把 layers[0..num_layers) 全部填上指针；缺哪个就返回 false 并记在 missing
inline bool bind_layers(LayerWeights* layers, int num_layers, std::string* missing = nullptr) {
    if (layers == nullptr || num_layers <= 0) return false;
    for (int l = 0; l < num_layers; l++) {
        LayerWeights w = {};
        for (int k = 0; k < 9; k++) {
            const std::string name = layer_tensor_name(l, layer_suffix(k));
            const float* p = tensor_f32(name);
            if (p == nullptr) {
                if (missing != nullptr) *missing = name;
                return false;
            }
            set_layer_slot(w, k, p);
        }
        layers[l] = w;
    }
    return true;
}

inline const float* bind_embed() {
    return tensor_f32("model.embed_tokens.weight");
}

inline const float* bind_final_norm() {
    return tensor_f32("model.norm.weight");
}

// lm_head 缺失时退回 embed（只当形状正好是 [hidden, vocab] 才收，免得静默算错）
inline const float* bind_lm_head(int hidden, int vocab) {
    const bfile::Entry* e = find_tensor("lm_head.weight");
    if (e == nullptr) e = find_tensor("model.embed_tokens.weight");
    if (e == nullptr || e->dtype != bfile::F32) return nullptr;
    if (e->ndim != 2) return nullptr;
    if (e->shape[0] != static_cast<uint64_t>(hidden)) return nullptr;
    if (e->shape[1] != static_cast<uint64_t>(vocab)) return nullptr;
    return bfile::current().f32(e);
}

#endif  // LOAD_WEIGHTS_H
