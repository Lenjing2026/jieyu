#ifndef Rope_h
#define Rope_h
#include<cstddef>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
void rope_at(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    if(cpu_can.fma3)
        #include"rope_fma.txt"
    else if(cpu_can.avx)
        #include"rope_avx.txt"
    else{
        const int half=head_dim/2;
        par_for(0,seq,seq>=32,[&](int64_t p0,int64_t p1){
        for (int64_t pos=p0;pos<p1;pos++) {
            const float* c=cos_table+(size_t)(start_pos+pos)*half;
            const float* s=sin_table+(size_t)(start_pos+pos)*half;
            for (int h=0;h<num_heads;h++) {
                float* qh=q+((size_t)pos*num_heads+h)*head_dim;
                for (int i=0;i<half;i++){
                    const float x0=qh[2*i];
                    const float x1=qh[2*i+1];
                    qh[2*i]=x0*c[i]-x1*s[i];
                    qh[2*i+1]=x0*s[i]+x1*c[i];
                }
            }
        }
        });
    }
}
void rope(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,0,num_heads,head_dim,cos_table,sin_table);
}
void rope_pos(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_at(q,seq,start_pos,num_heads,head_dim,cos_table,sin_table);
}
#endif