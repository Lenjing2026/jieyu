#ifndef LM_HEAD_H
#define LM_HEAD_H
#include "wmat_kernel.h"
#include <vector>
inline void lm_head(const float* X,const WMat& W_lm,float* logits,
                    int seq,int hidden,int vocab_size){
    matmul_w(W_lm,seq,vocab_size,hidden,X,logits);
}
#endif