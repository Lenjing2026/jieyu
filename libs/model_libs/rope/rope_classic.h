#ifndef ROPE_CLASSIC_H
#define ROPE_CLASSIC_H
#include<bits/stdc++.h>
void rope_classic_at(float* q,int seq,int start_pos,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    int half=head_dim/2;
    #pragma omp parallel for if(seq>=32)
    for (int pos=0;pos<seq;pos++) {
        const float* c=cos_table+(size_t)(start_pos+pos)*half;
        const float* s=sin_table+(size_t)(start_pos+pos)*half;
        for (int h=0;h<num_heads;h++) {
            float* qh=q+((size_t)pos*num_heads+h)*head_dim;
            for (int i=0;i<half;i++){
                float x0=qh[2*i];
                float x1=qh[2*i+1];
                qh[2*i]=x0*c[i]-x1*s[i];
                qh[2*i+1]=x0*s[i]+x1*c[i];
            }
        }
    }
}
void rope_classic(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    rope_classic_at(q,seq,0,num_heads,head_dim,cos_table,sin_table);
}
#endif