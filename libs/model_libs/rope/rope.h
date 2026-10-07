#ifndef Rope_h
#define Rope_h
#include "rope_classic.h"
#include "rope_avx.h"
#include "rope_fma.h"
#include "../../date_libs/check_cpu.h"
void rope_at(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    if(cpu_can.fma3)
        rope_fma_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
    else if(cpu_can.avx)
        rope_avx_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
    else
        rope_classic_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
}
void rope(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,0,num_heads,head_dim,cos_table,sin_table);
}
void rope_pos(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
}
#endif