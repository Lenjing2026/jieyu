#ifndef attention_h
#define attention_h
#include<cmath>
#include<cstring>
#include<vector>
#include<stdexcept>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
void attention(float* Q,float* K,float* V,float* out,int seq,int start_pos,
               int num_heads,int num_kv_heads,int head_dim,
               float* ck=nullptr,float* cv=nullptr,int kv_cap=0){
    if(cpu_can.fma3)
        #include"attention_fma.txt"
    else if(cpu_can.avx)
        #include"attention_avx.txt"
    else{
        if(num_kv_heads<=0||num_heads%num_kv_heads!=0)
            throw std::runtime_error("attention: num_heads 必须能被 num_kv_heads 整除");
        if(seq<=0) throw std::runtime_error("attention: seq 必须为正");
        if(start_pos<0) throw std::runtime_error("attention: start_pos 不能为负");
        const int kv_dim=num_kv_heads*head_dim;
        if(ck!=nullptr||cv!=nullptr){
            if(ck==nullptr||cv==nullptr)
                throw std::runtime_error("attention: KV cache 两个指针必须成对给出");
            if((size_t)start_pos+(size_t)seq>(size_t)kv_cap)
                throw std::runtime_error("attention: KV cache 容量不足");
            memcpy(ck+(size_t)start_pos*kv_dim,K,(size_t)seq*kv_dim*sizeof(float));
            memcpy(cv+(size_t)start_pos*kv_dim,V,(size_t)seq*kv_dim*sizeof(float));
        }else{
            start_pos=0;
            ck=K;
            cv=V;
        }
        const int group=num_heads/num_kv_heads;
        const float scale=1.0f/sqrtf((float)head_dim);
        par_for(0,num_heads,(size_t)num_heads*(size_t)(start_pos+seq)>=1536,[&](int64_t h0,int64_t h1){
        for (int64_t h=h0;h<h1;h++) {
            std::vector<float> scores((size_t)start_pos+seq);
            const int kv_h=(int)(h/group);
            for (int i=0;i<seq;i++) {
                const int p=start_pos+i;
                const float* q=Q+((size_t)i*num_heads+h)*head_dim;
                float mxs=-1e30f;
                for (int j=0;j<=p;j++) {
                    const float* k=ck+((size_t)j*num_kv_heads+kv_h)*head_dim;
                    float dt=0.0f;
                    for (int d=0;d<head_dim;d++)
                        dt+=q[d]*k[d];
                    dt*=scale;
                    scores[j]=dt;
                    if (dt>mxs) mxs=dt;
                }
                float sum=0.0f;
                for (int j=0;j<=p;j++) {
                    scores[j]=expf(scores[j]-mxs);
                    sum+=scores[j];
                }
                const float inv_sum=1.0f/sum;
                for (int j=0;j<=p;j++) scores[j]*=inv_sum;
                float* o=out+((size_t)i*num_heads+h)*head_dim;
                memset(o,0,head_dim*sizeof(float));
                for (int j=0;j<=p;j++) {
                    const float* v=cv+((size_t)j*num_kv_heads+kv_h)*head_dim;
                    for (int d=0;d<head_dim;d++)
                        o[d]+=scores[j]*v[d];
                }
            }
        }
        });
    }
}
#endif