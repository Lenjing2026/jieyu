#ifndef transformer_layer_h
#define transformer_layer_h
#include <vector>
#include "ffn/ffn.h"
#include "rmsnorm/rmsnorm.h"
#include "attention/attention.h"
#include "rope/rope.h"
#include "silu/silu.h"
void transformer_layer(
    float* X,                    // [seq, hidden]，输入输出都在这里
    const float* rms1_weight,    // [hidden]
    const float* Wq,             // QKV + 输出投影
    const float* Wk,
    const float* Wv,
    const float* Wo,
    const float* cos_table,
    const float* sin_table,
    const float* rms2_weight,    // [hidden]
    const float* W1,             // FFN
    const float* W2,
    const float* W3,
    int seq, int hidden, int num_heads, int num_kv_heads,
    int head_dim, int intermediate,
    float eps=1e-5f
) {
    int q_dim=num_heads*head_dim;
    int kv_dim=num_kv_heads*head_dim;
    std::vector<float> xn((size_t)seq*hidden);
    std::vector<float> Q((size_t)seq*q_dim);
    std::vector<float> K((size_t)seq*kv_dim);
    std::vector<float> V((size_t)seq*kv_dim);
    std::vector<float> attn((size_t)seq*q_dim);
    std::vector<float> proj((size_t)seq*hidden);
    std::vector<float> ffn_out((size_t)seq*hidden);

    // 1. RMSNorm
    for(int i=0;i<seq;i++)
        rmsnorm(X+(size_t)i*hidden,xn.data()+(size_t)i*hidden,rms1_weight,hidden,eps);

    // 2. QKV 投影
    matmul(seq,q_dim,hidden,xn.data(),Wq,Q.data());
    matmul(seq,kv_dim,hidden,xn.data(),Wk,K.data());
    matmul(seq,kv_dim,hidden,xn.data(),Wv,V.data());

    // 3. RoPE
    rope(Q.data(),seq,num_heads,head_dim,cos_table,sin_table);
    rope(K.data(),seq,num_kv_heads,head_dim,cos_table,sin_table);

    // 4. Attention
    attention(Q.data(),K.data(),V.data(),attn.data(),seq,num_heads,num_kv_heads,head_dim);

    // 5. 输出投影
    matmul(seq,hidden,q_dim,attn.data(),Wo,proj.data());

    // 6. 残差
    for(size_t i=0;i<(size_t)seq*hidden;i++)
        X[i]+=proj[i];

    // 7. RMSNorm
    for(int i=0;i<seq;i++)
        rmsnorm(X+(size_t)i*hidden,xn.data()+(size_t)i*hidden,rms2_weight,hidden,eps);

    // 8. FFN
    ffn(xn.data(),W1,W2,W3,ffn_out.data(),seq,hidden,intermediate);

    // 9. 残差
    for(size_t i=0;i<(size_t)seq*hidden;i++)
        X[i]+=ffn_out[i];
}
#endif