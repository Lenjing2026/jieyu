#ifndef DATE_LIBS_DATE_H
#define DATE_LIBS_DATE_H
#include<bits/stdc++.h>
#include<windows.h>
#include"wmat.h"
#include"../matmul_libs/matmul.h"
using namespace std;
using namespace std::filesystem;
struct ModelInfo{
    uint64_t h; 
    uint64_t l;
    uint8_t mode;
    uint8_t num_heads;
    uint8_t num_kv_heads;
    uint8_t head_dim;
    uint32_t intermediate_size;
    uint64_t vocab_size;
    uint64_t file_size;
    path model_vector;
    path model_info;
    path model_tokenizer;
    HANDLE model_handle;
    LPVOID model_map;
} model;
struct RoPETable{
    int max_seq,
        head_dim;
    vector<float> cos;
    vector<float> sin;
    void init(int seq,int dim,float t_base){
        max_seq=seq;
        head_dim=dim;
        int helf=dim/2;
        cos.resize((size_t)seq*helf);
        sin.resize((size_t)seq*helf);
        for(int i=0;i<max_seq;i++){
            for(int j=0;j<helf;j++){
                float freq=1.0f/powf(t_base,2.0f*j/head_dim);
                float there=i*freq;
                cos[(size_t)i*helf+j]=cosf(there);
                sin[(size_t)i*helf+j]=sinf(there);
            }
        }
    }
    void free(){
        cos.clear();
        sin.clear();
        cos.shrink_to_fit();
        sin.shrink_to_fit();
        max_seq=head_dim=0;
    }
} trif;
struct LayerWeights{
    const float* rms1_weight;
    WMat Wq;
    WMat Wk;
    WMat Wv;
    WMat Wo;
    const float* bq;
    const float* bk;
    const float* bv;
    const float* rms2_weight;
    WMat W1;
    WMat W2;
    WMat W3;
};
struct __KVcache{
    int num_layers,
        max_seq,
        num_kv_heads,
        head_dim,
        current_len;
    std::vector<float> k;
    std::vector<float> v;
    void init(int l,int max_s,int nkh,int hd){
        num_layers=l;
        max_seq=max_s;
        num_kv_heads=nkh;
        head_dim=hd;
        current_len=0;
        size_t size=(size_t)num_layers*max_seq*num_kv_heads*head_dim;
        k.assign(size,0.0f);
        v.assign(size,0.0f);
    }
    void free(){current_len=0;} //我免费了！！！！！！！！！！！！！！
    float* k_ptr(int l,int p){
        size_t o=((size_t)l*max_seq+p)*num_kv_heads*head_dim;
        return k.data()+o;
    }
    float* v_ptr(int l,int p){
        size_t o=((size_t)l*max_seq+p)*num_kv_heads*head_dim;
        return v.data()+o;
    }
};
#endif  // DATE_LIBS_DATE_H
