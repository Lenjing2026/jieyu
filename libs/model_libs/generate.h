#ifndef generate_h
#define generate_h
#include <vector>
#include <stdexcept>
#include "../date_libs/date.h"
#include "forward.h"
#include "prefetch.h"
// 每产出一个 token 就回调一次（流式输出用）；回调返回 true 表示该停了（碰到结束记号）
typedef bool (*token_cb)(int id,void* ud);
int generate_cached(
    const int* prompt_ids,int prompt_len,
    const int* eos_ids,int num_eos,
    const WMat& embed,
    const WMat& lm_head,
    const LayerWeights* layers,int num_layers,
    const float* final_rms_weight,
    const float* cos_table,const float* sin_table,int max_seq,
    int max_new_tokens,float temperature,
    int vocab_size,int hidden,
    int num_heads,int num_kv_heads,int head_dim,int intermediate,
    __KVcache& cache,
    int* out_ids,
    float repetition_penalty=1.0f,
    token_cb cb=nullptr,void* cb_ud=nullptr
){
    if(prompt_len<=0||prompt_len>max_seq) return 0;
    if(cache.num_layers<num_layers||cache.max_seq<max_seq
       ||cache.num_kv_heads!=num_kv_heads||cache.head_dim!=head_dim)
        throw std::runtime_error("generate: KV cache 尺寸与模型不符");
    for(int i=0;i<prompt_len;i++)
        out_ids[i]=prompt_ids[i];
    cache.current_len=0;
    if(max_new_tokens<=0) return prompt_len;

    std::vector<float> X((size_t)max_seq*hidden);
    std::vector<float> logits((size_t)vocab_size,0.0f);
    unsigned rng=12345u;
    int len=prompt_len;
    int seq=prompt_len;
    for(int step=0;step<max_new_tokens&&len<max_seq;step++){
        const int start_pos=len-seq;
        prefetch::range(prefetch::data_of(lm_head),
                        prefetch::bytes_of(lm_head,(size_t)vocab_size,(size_t)hidden));
        for(int i=0;i<seq;i++)
            wmat_row(embed,(size_t)out_ids[start_pos+i],hidden,X.data()+(size_t)i*hidden);
        forward_cached(X.data(),layers,num_layers,final_rms_weight,
                       cos_table,sin_table,seq,start_pos,cache,hidden,
                       num_heads,num_kv_heads,head_dim,intermediate);
        const float* last=X.data()+(size_t)(seq-1)*hidden;
        matmul_w(lm_head,1,vocab_size,hidden,last,logits.data());
        // 重复惩罚只压"这一轮自己生成的"——带上提示词的话，用户刚说的词会被压低，
        // 贪心就会把 hello 挤成 hetoo 这种
        if(repetition_penalty!=1.0f)
            for(int i=prompt_len;i<len;i++){
                const int id=out_ids[i];
                if(id>=0&&id<vocab_size)
                    logits[id]=(logits[id]>0.0f)?logits[id]/repetition_penalty
                                                 :logits[id]*repetition_penalty;
            }
        int next=0;
        if(temperature<=0.0f){
            for(int v=1;v<vocab_size;v++)
                if(logits[v]>logits[next]) next=v;
        }else{
            float mx=logits[0];
            for(int v=1;v<vocab_size;v++)
                if(logits[v]>mx) mx=logits[v];
            double sum=0.0;
            for(int v=0;v<vocab_size;v++){
                logits[v]=expf((logits[v]-mx)/temperature);
                sum+=logits[v];
            }
            rng=rng*1664525u+1013904223u;
            double r=(double)(rng>>8)/16777216.0*sum;
            next=vocab_size-1;
            for(int v=0;v<vocab_size;v++){
                r-=logits[v];
                if(r<=0.0){ next=v; break; }
            }
        }
        out_ids[len++]=next;
        if(cb!=nullptr&&cb(next,cb_ud)) break;
        seq=1;

        bool stop=false;
        if(eos_ids!=nullptr)
            for(int e=0;e<num_eos;e++)
                if(next==eos_ids[e]) stop=true;
        if(stop) break;
    }
    return len;
}
int generate(
    const int* prompt_ids,int prompt_len,
    const int* eos_ids,int num_eos,
    const WMat& embed,
    const WMat& lm_head,
    const LayerWeights* layers,int num_layers,
    const float* final_rms_weight,
    const float* cos_table,const float* sin_table,int max_seq,
    int max_new_tokens,float temperature,
    int vocab_size,int hidden,
    int num_heads,int num_kv_heads,int head_dim,int intermediate,
    int* out_ids,
    float repetition_penalty=1.0f,
    token_cb cb=nullptr,void* cb_ud=nullptr
){
    __KVcache cache;
    cache.init(num_layers,max_seq,num_kv_heads,head_dim);
    return generate_cached(prompt_ids,prompt_len,eos_ids,num_eos,embed,lm_head,
                           layers,num_layers,final_rms_weight,cos_table,sin_table,max_seq,
                           max_new_tokens,temperature,vocab_size,hidden,
                           num_heads,num_kv_heads,head_dim,intermediate,cache,out_ids,
                           repetition_penalty,cb,cb_ud);
}
#endif