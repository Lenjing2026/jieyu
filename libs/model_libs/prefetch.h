#ifndef PREFETCH_H
#define PREFETCH_H
#include<cstddef>
#include<cstdint>
#include<string>
#if !defined(_WIN32)
#include<sys/mman.h>
#include<unistd.h>
#endif
#include"../date_libs/wmat.h"
#include"../matmul_libs/matmul_q.h"
#include"../date_libs/date.h"
namespace prefetch {

#ifdef _WIN32
inline HANDLE open_model_bf(){
    const std::string path=model.model_vector.string();
    if(path.empty()) return INVALID_HANDLE_VALUE;
    return CreateFileA(path.c_str(),GENERIC_READ,
                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
}
inline HANDLE get_handle(){
    static HANDLE h=INVALID_HANDLE_VALUE;
    static std::string cur;
    const std::string want=model.model_vector.string();
    if(h!=INVALID_HANDLE_VALUE&&want==cur) return h;
    if(h!=INVALID_HANDLE_VALUE) CloseHandle(h);
    h=open_model_bf();
    cur=want;
    return h;
}
inline void* get_scratch(){
    static void* p=VirtualAlloc(nullptr,1u << 22,MEM_COMMIT | MEM_RESERVE,PAGE_READWRITE);
    return p;
}
#endif

inline void range(const void* p,size_t bytes){
#ifdef _WIN32
    if(p==nullptr||bytes==0) return;
    const uintptr_t base=(uintptr_t)model.model_map;
    const uintptr_t q=(uintptr_t)p;
    if(base==0||q < base) return;
    const unsigned long long off=(unsigned long long)(q - base);
    if(model.file_size!=0&&off + bytes > model.file_size) return;
    const HANDLE h=get_handle();
    void* buf=get_scratch();
    if(h==INVALID_HANDLE_VALUE||buf==nullptr) return;
    const size_t chunk=(size_t)1 << 22;
    unsigned long long pos=off,left=bytes;
    while(left > 0){
        const DWORD want=(DWORD)((left < chunk)?left:chunk);
        DWORD got=0;
        OVERLAPPED ov={};
        ov.Offset=(DWORD)(pos & 0xFFFFFFFFull);
        ov.OffsetHigh=(DWORD)(pos >> 32);
        if(!ReadFile(h,buf,want,&got,&ov)||got==0) return;
        pos+=got;
        left-=got;
    }
#else
    if(p==nullptr||bytes==0) return;
    const uintptr_t base=(uintptr_t)model.model_map;
    const uintptr_t q=(uintptr_t)p;
    if(base==0||q < base) return;
    const size_t off=(size_t)(q - base);
    if(model.file_size!=0&&off + bytes > model.file_size) return;
    const long ps=sysconf(_SC_PAGESIZE);
    const size_t pg=ps > 0?(size_t)ps:4096;
    const size_t start=off - off % pg;
    size_t len=bytes + (off - start);
    len=(len + pg - 1) / pg * pg;
    madvise((void*)(base + start),len,MADV_WILLNEED);
#endif
}

inline const void* data_of(const WMat& w){
    if(w.f!=nullptr) return w.f;
    if(w.h!=nullptr) return w.h;
    return w.q;
}

inline size_t bytes_of(const WMat& w,size_t rows,size_t cols){
    if(data_of(w)==nullptr) return 0;
    if(w.q!=nullptr){
        const size_t be=qmat::block_elems(w.qt),bb=qmat::block_bytes(w.qt);
        return rows * (cols / be) * bb;
    }
    return rows * cols * ((w.h!=nullptr)?2u:4u);
}

inline void layer(const LayerWeights& w,size_t hidden,size_t q_dim,size_t kv_dim,size_t inter){
    range(data_of(w.Wq),bytes_of(w.Wq,q_dim,hidden));
    range(data_of(w.Wk),bytes_of(w.Wk,kv_dim,hidden));
    range(data_of(w.Wv),bytes_of(w.Wv,kv_dim,hidden));
    range(data_of(w.Wo),bytes_of(w.Wo,hidden,q_dim));
    range(data_of(w.W1),bytes_of(w.W1,inter,hidden));
    range(data_of(w.W3),bytes_of(w.W3,inter,hidden));
    range(data_of(w.W2),bytes_of(w.W2,hidden,inter));
}

}
#endif
