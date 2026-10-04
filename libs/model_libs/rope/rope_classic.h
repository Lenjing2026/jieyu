#include<bits/stdc++.h>
void rope_classic(float* q,int seq,int num_heads,int head_dim,const float* cos_table,const float* sin_table){
    int half=head_dim/2;
    for (int pos=0;pos<seq;pos++) {
        const float* c=cos_table+(size_t)pos*half;
        const float* s=sin_table+(size_t)pos*half;
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