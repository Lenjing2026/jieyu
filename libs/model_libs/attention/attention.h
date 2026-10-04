#ifndef attention_h
#define attention_h
#include"attention_classic.h"
void attention(float* Q,float* K,float* V,float* out,int seq,int num_heads,int num_kv_heads,int head_dim){
    attention_classic(Q,K,V,out,seq,num_heads,num_kv_heads,head_dim);
}
#endif