#ifndef rmsnorm_h
#define rmsnorm_h
#include "rmsnorm_classic.h"
void rmsnorm(float* x,float* y,const float* weight,int n,float eps){
    rmsnorm_classic(x,y,weight,n,eps);
}
#endif
