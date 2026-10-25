#ifndef _HPP_PNG
#define _HPP_PNG
#include "../wmat_kernel.h"
#include "../silu/silu.h"
#include "../../date_libs/parallel.h"
#include<vector>
void ffn(float* x,const WMat& w1,const WMat& w2,const WMat& w3,
         float* out, int seq,int hidden,int intermediate){
    std::vector<float> gate(seq*intermediate);
    std::vector<float> up(seq*intermediate);
    matmul_w(w1,seq,intermediate,hidden,x,gate.data());
    matmul_w(w3,seq,intermediate,hidden,x,up.data());
    silu(gate.data(),gate.data(),seq*intermediate);
    const int n_sw=(int)((size_t)seq*intermediate);
    par_for(0,n_sw,n_sw>=8192,[&](int64_t i0,int64_t i1){
        for(int64_t i=i0;i<i1;i++) gate[i]=gate[i]*up[i];
    });
    matmul_w(w2,seq,hidden,intermediate,gate.data(),out);
}
#endif