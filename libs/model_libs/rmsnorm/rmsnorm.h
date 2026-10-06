#ifndef rmsnorm_h
#define rmsnorm_h
#include "rmsnorm_classic.h"
#include "rmsnorm_avx.h"
#include "rmsnorm_fma.h"
#include "../../date_libs/date.h"
#include "../../date_libs/check_cpu.h"
void rmsnorm(float* x,float* y,const float* weight,int n,float eps){
    if(cpu_can.fma3)      rmsnorm_fma(x,y,weight,n,eps);
    else if(cpu_can.avx)  rmsnorm_avx(x,y,weight,n,eps);
    else                  rmsnorm_classic(x,y,weight,n,eps);
}
#endif
