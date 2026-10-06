#ifndef transformer_layer_h
#define transformer_layer_h
#include <vector>
#include "ffn/ffn.h"
#include "rmsnorm/rmsnorm.h"
#include "attention/attention.h"
#include "rope/rope.h"
#include "silu/silu.h"
void transformer_layer(
    float* X,
    const float* rms1_weight,
    const WMat& Wq,
    const WMat& Wk,
    const WMat& Wv,
    const WMat& Wo,
    const float* bq,
    const float* bk,
    const float* bv,
    const float* cos_table,
    const float* sin_table,
    const float* rms2_weight,
    const WMat& W1,
    const WMat& W2,
    const WMat& W3,
    int seq, int hidden, int num_heads, int num_kv_heads,
    int head_dim, int intermediate,
    int start_pos, float* cache_k, float* cache_v, int kv_cap,
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

    #pragma omp parallel for if(seq>=32)
    for(int i=0;i<seq;i++)
        rmsnorm(X+(size_t)i*hidden,xn.data()+(size_t)i*hidden,rms1_weight,hidden,eps);

    matmul_w(Wq,seq,q_dim,hidden,xn.data(),Q.data());
    matmul_w(Wk,seq,kv_dim,hidden,xn.data(),K.data());
    matmul_w(Wv,seq,kv_dim,hidden,xn.data(),V.data());

    if(bq!=nullptr){
        #pragma omp parallel for if(seq>=32)
        for(int i=0;i<seq;i++) for(int j=0;j<q_dim;j++) Q[(size_t)i*q_dim+j]+=bq[j];
    }
    if(bk!=nullptr){
        #pragma omp parallel for if(seq>=32)
        for(int i=0;i<seq;i++) for(int j=0;j<kv_dim;j++) K[(size_t)i*kv_dim+j]+=bk[j];
    }
    if(bv!=nullptr){
        #pragma omp parallel for if(seq>=32)
        for(int i=0;i<seq;i++) for(int j=0;j<kv_dim;j++) V[(size_t)i*kv_dim+j]+=bv[j];
    }

    rope_pos(Q.data(),seq,start_pos,num_heads,head_dim,cos_table,sin_table);
    rope_pos(K.data(),seq,start_pos,num_kv_heads,head_dim,cos_table,sin_table);

    attention_kv(Q.data(),K.data(),V.data(),attn.data(),seq,start_pos,
                 num_heads,num_kv_heads,head_dim,cache_k,cache_v,kv_cap);

    matmul_w(Wo,seq,hidden,q_dim,attn.data(),proj.data());

    #pragma omp parallel for if(seq>=32)
    for(size_t i=0;i<(size_t)seq*hidden;i++)
        X[i]+=proj[i];

    #pragma omp parallel for if(seq>=32)
    for(int i=0;i<seq;i++)
        rmsnorm(X+(size_t)i*hidden,xn.data()+(size_t)i*hidden,rms2_weight,hidden,eps);

    ffn(xn.data(),W1,W2,W3,ffn_out.data(),seq,hidden,intermediate);

    #pragma omp parallel for if(seq>=32)
    for(size_t i=0;i<(size_t)seq*hidden;i++)
        X[i]+=ffn_out[i];
}

void transformer_layer(
    float* X,
    const float* rms1_weight,
    const WMat& Wq,
    const WMat& Wk,
    const WMat& Wv,
    const WMat& Wo,
    const float* cos_table,
    const float* sin_table,
    const float* rms2_weight,
    const WMat& W1,
    const WMat& W2,
    const WMat& W3,
    int seq, int hidden, int num_heads, int num_kv_heads,
    int head_dim, int intermediate,
    float eps=1e-5f
) {
    transformer_layer(X,rms1_weight,Wq,Wk,Wv,Wo,nullptr,nullptr,nullptr,
                      cos_table,sin_table,rms2_weight,W1,W2,W3,
                      seq,hidden,num_heads,num_kv_heads,head_dim,intermediate,
                      0,nullptr,nullptr,0,eps);
}

void transformer_layer(
    float* X,
    const float* rms1_weight,
    const WMat& Wq,
    const WMat& Wk,
    const WMat& Wv,
    const WMat& Wo,
    const float* bq,
    const float* bk,
    const float* bv,
    const float* cos_table,
    const float* sin_table,
    const float* rms2_weight,
    const WMat& W1,
    const WMat& W2,
    const WMat& W3,
    int seq, int hidden, int num_heads, int num_kv_heads,
    int head_dim, int intermediate,
    float eps=1e-5f
) {
    transformer_layer(X,rms1_weight,Wq,Wk,Wv,Wo,bq,bk,bv,cos_table,sin_table,rms2_weight,
                      W1,W2,W3,seq,hidden,num_heads,num_kv_heads,head_dim,intermediate,
                      0,nullptr,nullptr,0,eps);
}
#endif