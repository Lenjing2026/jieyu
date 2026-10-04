#ifndef LM_HEAD_H
#define LM_HEAD_H
#include "matmul.h"
#include <vector>
inline void lm_head(const float* X,const float* W_lm,float* logits,
                    int seq,int hidden,int vocab_size){
    matmul(seq,vocab_size,hidden,X,W_lm,logits);
}
#endif