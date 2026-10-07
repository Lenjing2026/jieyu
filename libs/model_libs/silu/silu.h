#ifndef silu_h
#define silu_h
#include "silu_classic.h"
#include "silu_avx.h"
#include "silu_fma.h"
#include "../../date_libs/check_cpu.h"
void silu(float* x,float* y,int n){
    if(cpu_can.fma3)
        silu_fma(x,y,n);
    else if(cpu_can.avx)
        silu_avx(x,y,n);
    else
        silu_classic(x,y,n);
}
#endif