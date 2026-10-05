#ifndef DATE_LIBS_WMAT_H
#define DATE_LIBS_WMAT_H
#include <cstdint>
#include <cstring>

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

inline uint16_t to_f16(float v){
    uint32_t bits;
    std::memcpy(&bits,&v,4);
    const uint32_t s=(bits>>16)&0x8000u;
    const int32_t e=(int32_t)((bits>>23)&0xFFu)-127+15;
    const uint32_t m=bits&0x7FFFFFu;
    if(((bits>>23)&0xFFu)==0xFFu) return (uint16_t)(s|0x7C00u|(m?0x200u:0));
    if(e>=31) return (uint16_t)(s|0x7C00u);
    if(e<=0){
        if(e<-10) return (uint16_t)s;
        const uint32_t mm=(m|0x800000u)>>(uint32_t)(1-e);
        return (uint16_t)(s|(mm>>13));
    }
    return (uint16_t)(s|((uint32_t)e<<10)|(m>>13));
}

inline void row_to_f32(const uint16_t* src,float* dst,int n){
    for(int i=0;i<n;i++) dst[i]=to_f32(src[i]);
}

inline void f32_row_to_f16(const float* src,uint16_t* dst,int n){
    for(int i=0;i<n;i++) dst[i]=to_f16(src[i]);
}

}  // namespace wmat

// 一个权重矩阵：f32 / f16 / 块量化，三选一
//   f / h：普通浮点，内存是 [k,n]（n 连续）
//   q    ：块量化，内存是 [n,k]（k 连续，块沿 k），行号就是输出号；qt 是 qmat 的类型代号
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

#endif  // DATE_LIBS_WMAT_H
