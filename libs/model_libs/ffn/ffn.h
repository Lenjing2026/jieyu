#ifndef _HPP_PNG
#define _HPP_PNG
#include "../../matmul_libs/matmul.h"
#include "../silu/silu.h"
#include<vector>
void ffn(float* x,const float* w1,const float* w2,const float* w3,
         float* out, int seq,int hidden,int intermediate){
    std::vector<float> gate(seq*intermediate);
    std::vector<float> up(seq*intermediate);
    matmul(seq,intermediate,hidden,x,w1,gate.data());
    matmul(seq,intermediate,hidden,x,w3,up.data());
    silu(gate.data(),gate.data(),seq*intermediate);
    for(int i=0;i<seq*intermediate;i++)
        gate[i]=gate[i]*up[i];
    matmul(seq,hidden,intermediate,gate.data(),w2,out);       
}
#endif