#ifndef forward_h
#define forward_h
#include <cstring>
#include <stdexcept>
#include "../date_libs/date.h"
#include "prefetch.h"
#include"transformer_layer.h"
void forward(
    float* X,
    const LayerWeights* layers,
    int num_layers,
    const float* final_rms_weight,
    const float* cos_table,
    const float* sin_table,
    int seq, int hidden,
    int num_heads, int num_kv_heads,
    int head_dim, int intermediate
){
    const size_t q_dim=(size_t)num_heads*head_dim,kv_dim=(size_t)num_kv_heads*head_dim;
    prefetch::layer(layers[0],(size_t)hidden,q_dim,kv_dim,(size_t)intermediate);
    for (int l=0;l<num_layers;l++){
        if(l+1<num_layers)
            prefetch::layer(layers[l+1],(size_t)hidden,q_dim,kv_dim,(size_t)intermediate);
        const LayerWeights& w=layers[l];
        transformer_layer(
            X,
            w.rms1_weight,
            w.Wq, w.Wk, w.Wv, w.Wo,
            w.bq, w.bk, w.bv,
            cos_table, sin_table,
            w.rms2_weight,
            w.W1, w.W2, w.W3,
            seq, hidden, num_heads, num_kv_heads,
            head_dim, intermediate
        );
    }
    std::vector<float> xn((size_t)seq*hidden);
    for (int i=0;i<seq;i++)
        rmsnorm(
            X+(size_t)i*hidden,
            xn.data()+(size_t)i*hidden,
            final_rms_weight,
            hidden,1e-5f
        );
    memcpy(X,xn.data(),(size_t)seq*hidden*sizeof(float));
}
void forward_cached(
    float* X,
    const LayerWeights* layers,
    int num_layers,
    const float* final_rms_weight,
    const float* cos_table,
    const float* sin_table,
    int seq, int start_pos,
    __KVcache& cache,
    int hidden,
    int num_heads, int num_kv_heads,
    int head_dim, int intermediate
){
    if(cache.num_layers<num_layers||cache.num_kv_heads!=num_kv_heads||cache.head_dim!=head_dim)
        throw std::runtime_error("forward: KV cache 尺寸与模型不符");
    if(start_pos<0) throw std::runtime_error("forward: start_pos 不能为负");
    if((size_t)start_pos+(size_t)seq>(size_t)cache.max_seq)
        throw std::runtime_error("forward: KV cache 容量不足");
    const size_t q_dim=(size_t)num_heads*head_dim,kv_dim=(size_t)num_kv_heads*head_dim;
    prefetch::layer(layers[0],(size_t)hidden,q_dim,kv_dim,(size_t)intermediate);
    for (int l=0;l<num_layers;l++){
        if(l+1<num_layers)
            prefetch::layer(layers[l+1],(size_t)hidden,q_dim,kv_dim,(size_t)intermediate);
        const LayerWeights& w=layers[l];
        transformer_layer(
            X,
            w.rms1_weight,
            w.Wq, w.Wk, w.Wv, w.Wo,
            w.bq, w.bk, w.bv,
            cos_table, sin_table,
            w.rms2_weight,
            w.W1, w.W2, w.W3,
            seq, hidden, num_heads, num_kv_heads,
            head_dim, intermediate,
            start_pos, cache.k_ptr(l,0), cache.v_ptr(l,0), cache.max_seq
        );
    }
    std::vector<float> xn((size_t)seq*hidden);
    for (int i=0;i<seq;i++)
        rmsnorm(
            X+(size_t)i*hidden,
            xn.data()+(size_t)i*hidden,
            final_rms_weight,
            hidden,1e-5f
        );
    memcpy(X,xn.data(),(size_t)seq*hidden*sizeof(float));
    cache.current_len=start_pos+seq;
}
#endif