#ifndef forward_h
#define forward_h
#include <cstring>
#include "../date_libs/date.h"
#include"transformer_layer.h"
void forward(
    float* X,                          // [seq, hidden]，输入输出
    const LayerWeights* layers,        // [num_layers]
    int num_layers,
    const float* final_rms_weight,     // [hidden]
    const float* cos_table,
    const float* sin_table,
    int seq, int hidden,
    int num_heads, int num_kv_heads,
    int head_dim, int intermediate
){
    for (int l=0;l<num_layers;l++){
        const LayerWeights& w=layers[l];
        transformer_layer(
            X,
            w.rms1_weight,
            w.Wq, w.Wk, w.Wv, w.Wo,
            cos_table, sin_table,
            w.rms2_weight,
            w.W1, w.W2, w.W3,
            seq, hidden, num_heads, num_kv_heads,
            head_dim, intermediate
        );
    }
    // 最终 RMSNorm
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
#endif