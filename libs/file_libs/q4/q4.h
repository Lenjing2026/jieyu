#ifndef Q4_H
#define Q4_H
#include<cstdint>
#include<cstring>
#include<immintrin.h>
#include"../../date_libs/wmat.h"
#include"../../date_libs/check_cpu.h"
void dequant_q4k(const unsigned char* b,float* dst){
    if(cpu_can.avx2&&cpu_can.fma3)
        #include"q4_fma.txt"
    else if(cpu_can.avx)
        #include"q4_avx.txt"
    else{
        auto hv=[](const unsigned char* p){ uint16_t v; std::memcpy(&v,p,2); return wmat::to_f32(v); };
        auto sm=[](int j,const unsigned char* q,unsigned char& sc,unsigned char& mn){
            if(j<4){ sc=(unsigned char)(q[j]&63); mn=(unsigned char)(q[j+4]&63); }
            else{ sc=(unsigned char)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); mn=(unsigned char)((q[j+4]>>4)|((q[j]>>6)<<4)); }
        };
        const float d=hv(b),dmin=hv(b+2);
        const unsigned char* scp=b+4;
        const unsigned char* qs=b+16;
        int is=0;
        for(int g=0;g<4;g++){
            unsigned char s0,m0,s1,m1;
            sm(is,scp,s0,m0);
            sm(is+1,scp,s1,m1);
            const float d1=d*(float)s0,n1=dmin*(float)m0;
            const float d2=d*(float)s1,n2=dmin*(float)m1;
            for(int l=0;l<32;l++) dst[g*64+l]=d1*(float)(qs[l]&0xF)-n1;
            for(int l=0;l<32;l++) dst[g*64+32+l]=d2*(float)(qs[l]>>4)-n2;
            qs+=32;
            is+=2;
        }
    }
}
#endif
