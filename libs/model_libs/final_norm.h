#ifndef FINAL_NORM_H
#define FINAL_NORM_H
#include "rmsnorm/rmsnorm.h"
#include <cstring>
inline void final_norm(float* X,float* weight,
                       int seq,int hidden,float eps=1e-5f){
    for (int i=0;i<seq;i++)
        rmsnorm(X+(size_t)i*hidden,
                weight,
                X+(size_t)i*hidden,
                hidden, eps);
}
#endif