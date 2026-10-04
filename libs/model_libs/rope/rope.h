#ifndef Rope_h
#define Rope_h
#include "rope_classic.h"
void rope(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_classic(q,seq,num_heads,head_dim,cos_table,sin_table);
}
#endif