#ifndef Q4_H
#define Q4_H
#include"q4_classic.h"
#include"q4_avx.h"
#include"q4_fma.h"
#include "../../date_libs/check_cpu.h"
void dequant_q4k(const unsigned char* b, float* dst) {
    if(cpu_can.avx2 && cpu_can.fma3) q4_fma(b, dst);
    else if(cpu_can.avx) q4_avx(b, dst);
    else q4_classic(b, dst);
}
#endif  // Q4_H
