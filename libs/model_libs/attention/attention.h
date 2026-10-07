#ifndef attention_h
#define attention_h
#include"attention_classic.h"
#include"attention_avx.h"
#include"attention_fma.h"
#include "../../date_libs/check_cpu.h"
void attention(float* Q,float* K,float* V,float* out,int seq,int start_pos,
               int num_heads,int num_kv_heads,int head_dim,
               float* ck=nullptr,float* cv=nullptr,int kv_cap=0){
    if(cpu_can.fma3)
        attention_fma(Q,K,V,out,seq,start_pos,num_heads,num_kv_heads,head_dim,ck,cv,kv_cap);
    else if(cpu_can.avx)
        attention_avx(Q,K,V,out,seq,start_pos,num_heads,num_kv_heads,head_dim,ck,cv,kv_cap);
    else
        attention_classic(Q,K,V,out,seq,start_pos,num_heads,num_kv_heads,head_dim,ck,cv,kv_cap);
}
#endif