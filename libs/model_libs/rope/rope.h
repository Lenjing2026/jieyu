#ifndef Rope_h
#define Rope_h
#include "rope_classic.h"
void rope(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_classic(q,seq,num_heads,head_dim,cos_table,sin_table);
}
void rope_pos(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_classic_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
}
#endif