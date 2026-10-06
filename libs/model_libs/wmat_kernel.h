#ifndef WMAT_KERNEL_H
#define WMAT_KERNEL_H
#include <cstring>
#include "../date_libs/wmat.h"
#include "../matmul_libs/matmul.h"
#include "../matmul_libs/matmul_f16.h"
#include "../matmul_libs/matmul_q.h"

inline void matmul_w(const WMat& B,const int m,const int n,const int k,const float* A,float* C){
    if(B.f!=nullptr){ matmul(m,n,k,A,B.f,C); return; }
    if(B.q!=nullptr){ qmat::matmul_q(A,B.q,C,m,n,k,B.qt); return; }
    memset(C,0,sizeof(float)*m*n);
    if(B.h==nullptr) return;
    if(cpu_can.f16c&&cpu_can.avx2&&cpu_can.fma3) matmul_f16_avx_fma(A,B.h,C,m,n,k);
    else matmul_f16_classic(A,B.h,C,m,n,k);
}

inline void wmat_row(const WMat& w,size_t id,int width,float* dst){
    if(w.f!=nullptr){ memcpy(dst,w.f+id*(size_t)width,(size_t)width*sizeof(float)); return; }
    if(w.h!=nullptr){ wmat::row_to_f32(w.h+id*(size_t)width,dst,width); return; }
    if(w.q!=nullptr){
        const size_t be=qmat::block_elems(w.qt),bb=qmat::block_bytes(w.qt);
        const unsigned char* row=(const unsigned char*)w.q+id*((size_t)width/be)*bb;
        for(size_t bl=0,nb=(size_t)width/be;bl<nb;bl++)
            qmat::dequant_block(w.qt,row+bl*bb,dst+bl*be);
        return;
    }
    memset(dst,0,(size_t)width*sizeof(float));
}
#endif  // WMAT_KERNEL_H
