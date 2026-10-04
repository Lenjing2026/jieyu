#ifndef generate_h
#define generate_h
#include <vector>
#include "../date_libs/date.h"
#include "forward.h"
int generate(
    const int* prompt_ids,int prompt_len,
    const int* eos_ids,int num_eos,
    const float* embed,
    const float* lm_head,
    const LayerWeights* layers,int num_layers,
    const float* final_rms_weight,
    const float* cos_table,const float* sin_table,int max_seq,
    int max_new_tokens,float temperature,
    int vocab_size,int hidden,
    int num_heads,int num_kv_heads,int head_dim,int intermediate,
    int* out_ids
){
    if(prompt_len<=0||prompt_len>max_seq) return 0;
    for(int i=0;i<prompt_len;i++)
        out_ids[i]=prompt_ids[i];
    if(max_new_tokens<=0) return prompt_len;

    std::vector<float> X((size_t)max_seq*hidden);
    std::vector<float> logits((size_t)vocab_size,0.0f);
    unsigned rng=12345u;
    int len=prompt_len;
    for(int step=0;step<max_new_tokens&&len<max_seq;step++){
        for(int i=0;i<len;i++){
            const float* row=embed+(size_t)out_ids[i]*hidden;
            float* dst=X.data()+(size_t)i*hidden;
            for(int d=0;d<hidden;d++)
                dst[d]=row[d];
        }
        forward(X.data(),layers,num_layers,final_rms_weight,
                cos_table,sin_table,len,hidden,num_heads,num_kv_heads,
                head_dim,intermediate);
        const float* last=X.data()+(size_t)(len-1)*hidden;
        matmul(1,vocab_size,hidden,last,lm_head,logits.data());
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

        bool stop=false;
        for(int e=0;e<num_eos;e++)
            if(next==eos_ids[e]) stop=true;
        if(stop) break;
    }
    return len;
}
#endif