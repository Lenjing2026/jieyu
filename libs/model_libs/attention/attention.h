#ifndef attention_h
#define attention_h
#include"attention_classic.h"
void attention(float* Q,float* K,float* V,float* out,int seq,int num_heads,int num_kv_heads,int head_dim){
    attention_classic(Q,K,V,out,seq,num_heads,num_kv_heads,head_dim);
}
void attention_kv(float* Q,float* K,float* V,float* out,int seq,int start_pos,
                  int num_heads,int num_kv_heads,int head_dim,
                  float* ck,float* cv,int kv_cap){
    attention_classic_kv(Q,K,V,out,seq,start_pos,num_heads,num_kv_heads,head_dim,ck,cv,kv_cap);
}
#endif