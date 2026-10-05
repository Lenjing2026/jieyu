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

inline int wmat_qt(uint8_t d) {
    switch (d) {
        case bfile::Q4K: return qmat::Q4K;
        case bfile::Q5K: return qmat::Q5K;
        case bfile::Q6K: return qmat::Q6K;
        case bfile::Q8_0: return qmat::Q8_0;
        default: return -1;
    }
}

// f32 / f16 / 块量化 都能取；取不到就是空的 WMat
inline WMat tensor_wmat(const std::string& name) {
    const bfile::Entry* e = find_tensor(name);
    if (e == nullptr) return WMat();
    if (e->dtype == bfile::F32) return WMat(bfile::current().f32(e));
    if (e->dtype == bfile::F16) return WMat((const uint16_t*)bfile::current().data(e));
    const int qt = wmat_qt(e->dtype);
    if (qt >= 0) return WMat(bfile::current().data(e), (uint8_t)qt);
    return WMat();
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

inline void set_layer_wmat(LayerWeights& w, int index, const WMat& m) {
    switch (index) {
        case 1: w.Wq = m; break;
        case 2: w.Wk = m; break;
        case 3: w.Wv = m; break;
        case 4: w.Wo = m; break;
        case 6: w.W1 = m; break;
        case 7: w.W2 = m; break;
        case 8: w.W3 = m; break;
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
            if (k == 0 || k == 5) {
                const float* p = tensor_f32(name);
                if (p == nullptr) {
                    if (missing != nullptr) *missing = name;
                    return false;
                }
                set_layer_slot(w, k, p);
            } else {
                const WMat m = tensor_wmat(name);
                if (m.empty()) {
                    if (missing != nullptr) *missing = name;
                    return false;
                }
                set_layer_wmat(w, k, m);
            }
        }
        layers[l] = w;
    }
    return true;
}

inline WMat bind_embed() {
    return tensor_wmat("model.embed_tokens.weight");
}

inline bool bind_biases(LayerWeights* layers, int num_layers) {
    if (layers == nullptr || num_layers <= 0) return false;
    for (int l = 0; l < num_layers; l++) {
        layers[l].bq = tensor_f32(layer_tensor_name(l, "self_attn.q_proj.bias"));
        layers[l].bk = tensor_f32(layer_tensor_name(l, "self_attn.k_proj.bias"));
        layers[l].bv = tensor_f32(layer_tensor_name(l, "self_attn.v_proj.bias"));
    }
    return layers[0].bq != nullptr;
}

inline const float* bind_final_norm() {
    return tensor_f32("model.norm.weight");
}

// lm_head 缺失时退回 embed。普通浮点是转置过的 [hidden,vocab]；
// 量化权重是“行=输出”的 [vocab,hidden]，embed 正好就是
inline WMat bind_lm_head(int hidden, int vocab) {
    const bfile::Entry* e = find_tensor("lm_head.weight");
    if (e == nullptr) e = find_tensor("model.embed_tokens.weight");
    if (e == nullptr || e->ndim != 2) return WMat();
    if (bfile::block_elems(e->dtype) != 0) {
        if (e->shape[0] != static_cast<uint64_t>(vocab)) return WMat();
        if (e->shape[1] != static_cast<uint64_t>(hidden)) return WMat();
    } else {
        if (e->dtype != bfile::F32 && e->dtype != bfile::F16) return WMat();
        if (e->shape[0] != static_cast<uint64_t>(hidden)) return WMat();
        if (e->shape[1] != static_cast<uint64_t>(vocab)) return WMat();
    }
    return tensor_wmat(e->name);
}

#endif  // LOAD_WEIGHTS_H
