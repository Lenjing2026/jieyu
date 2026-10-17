#ifndef DATE_LIBS_WMAT_H
#define DATE_LIBS_WMAT_H
#include<cstdint>
#include<cstring>

namespace wmat {

inline float to_f32(uint16_t h){
    const uint32_t s=(uint32_t)(h>>15)&1u,e=(uint32_t)(h>>10)&0x1Fu,m=(uint32_t)h&0x3FFu;
    uint32_t bits;
    if(e==0){
        if(m==0) bits=s<<31;
        else{
            uint32_t mm=m;
            int k=0;
            while((mm&0x400u)==0){ mm<<=1; k++; }
            bits=(s<<31)|((uint32_t)(113-k)<<23)|((mm&0x3FFu)<<13);
        }
    }else if(e==31) bits=(s<<31)|0x7F800000u|(m<<13);
    else bits=(s<<31)|((e+112u)<<23)|(m<<13);
    float v;
    std::memcpy(&v,&bits,4);
    return v;
}

inline void row_to_f32(const uint16_t* src,float* dst,int n){
    for(int i=0;i<n;i++) dst[i]=to_f32(src[i]);
}

}

struct WMat{
    const float* f=nullptr;
    const uint16_t* h=nullptr;
    const void* q=nullptr;
    uint8_t qt=0;
    WMat(){}
    WMat(const float* p):f(p){}
    WMat(const uint16_t* p):h(p){}
    WMat(const void* p,uint8_t t):q(p),qt(t){}
    bool empty() const{ return f==nullptr&&h==nullptr&&q==nullptr; }
    bool is_f16() const{ return f==nullptr&&h!=nullptr; }
    bool is_quant() const{ return q!=nullptr; }
};

#endif
