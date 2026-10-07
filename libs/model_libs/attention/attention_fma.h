#ifndef ATTENTION_FMA_H
#define ATTENTION_FMA_H
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>
using namespace std;
__attribute__((target("avx,fma")))
void attention_fma(float* Q,float* K,float* V,float* out,int seq,int start_pos,
                       int num_heads,int num_kv_heads,int head_dim,
                       float* ck=nullptr,float* cv=nullptr,int kv_cap=0){
    if(num_kv_heads<=0||num_heads%num_kv_heads!=0)
        throw std::runtime_error("attention: num_heads 必须能被 num_kv_heads 整除");
    if(seq<=0) throw std::runtime_error("attention: seq 必须为正");
    if(start_pos<0) throw std::runtime_error("attention: start_pos 不能为负");
    int kv_dim=num_kv_heads*head_dim;
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
    int group = num_heads / num_kv_heads;
    float scale=1.0f/sqrtf((float)head_dim);
    #pragma omp parallel for if((size_t)num_heads*(size_t)(start_pos+seq)>=1536)
    for (int h=0;h<num_heads;h++) {
        std::vector<float> scores((size_t)start_pos+seq);
        int kv_h=h/group;
        for (int i=0;i<seq;i++) {
            const int p=start_pos+i;
            const float* q=Q+((size_t)i*num_heads+h)*head_dim;
            float mxs=-1e30f;
            for (int j=0;j<=p;j++) {
                const float* k=ck+((size_t)j*num_kv_heads+kv_h)*head_dim;
                int d=0;
                __m256 a0=_mm256_setzero_ps();
                __m256 a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps();
                __m256 a3=_mm256_setzero_ps();
                for (;d+32<=head_dim;d+=32) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(q+d),_mm256_loadu_ps(k+d),a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(q+d+8),_mm256_loadu_ps(k+d+8),a1);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(q+d+16),_mm256_loadu_ps(k+d+16),a2);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(q+d+24),_mm256_loadu_ps(k+d+24),a3);
                }
                for (;d+8<=head_dim;d+=8)
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(q+d),_mm256_loadu_ps(k+d),a0);
                a0=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
                __m128 lo=_mm256_castps256_ps128(a0);
                __m128 hi=_mm256_extractf128_ps(a0,1);
                __m128 s4=_mm_hadd_ps(_mm_add_ps(lo,hi),_mm_add_ps(lo,hi));
                s4=_mm_hadd_ps(s4,s4);
                float dt=_mm_cvtss_f32(s4);
                for (;d<head_dim;d++) dt+=q[d]*k[d];
                dt*=scale;
                scores[j]=dt;
                if (dt>mxs) mxs=dt;
            }
            float sum = 0.0f;
            for (int j=0;j<=p;j++) {
                scores[j]=expf(scores[j]-mxs);
                sum+=scores[j];
            }
            float inv_sum=1.0f/sum;
            for (int j=0;j<=p;j++) scores[j]*=inv_sum;
            float* o=out+((size_t)i*num_heads + h) * head_dim;
            memset(o,0,head_dim*sizeof(float));
            for (int j=0;j<=p;j++) {
                const float* v=cv+((size_t)j*num_kv_heads+kv_h)*head_dim;
                const float prob=scores[j];
                const __m256 vprob=_mm256_set1_ps(prob);
                int d=0;
                for (;d+8<=head_dim;d+=8) {
                    __m256 ov=_mm256_loadu_ps(o+d);
                    ov=_mm256_fmadd_ps(vprob,_mm256_loadu_ps(v+d),ov);
                    _mm256_storeu_ps(o+d,ov);
                }
                for (;d<head_dim;d++) o[d]+=prob*v[d];
            }
        }
    }
}
#endif