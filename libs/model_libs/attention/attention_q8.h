#ifndef attention_q8_h
#define attention_q8_h
#include<cmath>
#include<cstdint>
#include<cstring>
#include<vector>
#include<stdexcept>
#include<immintrin.h>
#include "../../date_libs/check_cpu.h"
#include "../../date_libs/parallel.h"
#include "../../date_libs/info.h"
struct __KVq{
    int8_t* k;
    float* ks;
    int8_t* v;
    float* vs;
    int cap;
    bool ok() const{return k!=nullptr&&v!=nullptr&&ks!=nullptr&&vs!=nullptr&&cap>0;}
};
void kv_pack_q8(const float* src,int8_t* dst,float* sc,int heads,int hd,int n){
    for(int i=0;i<n;i++)for(int h=0;h<heads;h++){
        const float* s=src+((size_t)i*heads+h)*hd;
        int8_t* d=dst+((size_t)i*heads+h)*hd;
        float amax=0.0f;
        for(int k=0;k<hd;k++){
            const float a=fabsf(s[k]);
            if(a>amax) amax=a;
        }
        const float inv=amax>1e-8f?127.0f/amax:0.0f;
        sc[(size_t)i*heads+h]=amax>1e-8f?amax/127.0f:0.0f;
        for(int k=0;k<hd;k++){
            float x=s[k]*inv;
            x=x>127.0f?127.0f:(x<-127.0f?-127.0f:x);
            d[k]=(int8_t)lrintf(x);
        }
    }
}
void attention_q8(float* Q,float* K,float* V,float* out,int seq,int start_pos,
                  int num_heads,int num_kv_heads,int head_dim,const __KVq& kvq){
    if(!kvq.ok()){
        const int kv_dim=num_kv_heads*head_dim;
        std::vector<int8_t> tk((size_t)seq*kv_dim),tv((size_t)seq*kv_dim);
        std::vector<float> tks((size_t)seq*num_kv_heads),tvs((size_t)seq*num_kv_heads);
        attention_q8(Q,K,V,out,seq,0,num_heads,num_kv_heads,head_dim,
                     __KVq{tk.data(),tks.data(),tv.data(),tvs.data(),seq});
        return;
    }
    if(cpu_can.avxvnni){
        #define SIMD_K __m256
        #define SIMD_INT __m256i
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_LOADI(p) _mm256_loadu_si256((const __m256i*)(p))
        #define SIMD_ZEROI() _mm256_setzero_si256()
        #define SIMD_ABS8 _mm256_abs_epi8
        #define SIMD_SIGN8 _mm256_sign_epi8
        #define SIMD_DP4A(acc,u,s) _mm256_dpbusd_epi32(acc,u,s)
        #define SIMD_I8TOI32(p) _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p)))
        #define SIMD_I32TOPS _mm256_cvtepi32_ps
        #define SIMD_HSUM32(v) ({__m256i _t=(v);__m128i _a=_mm_add_epi32(_mm256_castsi256_si128(_t),_mm256_extracti128_si256(_t,1));_a=_mm_add_epi32(_a,_mm_shuffle_epi32(_a,_MM_SHUFFLE(1,0,3,2)));_a=_mm_add_epi32(_a,_mm_shuffle_epi32(_a,_MM_SHUFFLE(2,3,0,1)));_mm_cvtsi128_si32(_a);})
        #define LONG 8
        #define IN_SET "avx2,avxvnni"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
    else if(cpu_can.avx2){
        #define SIMD_K __m256
        #define SIMD_INT __m256i
        #define SIMD_LOAD _mm256_loadu_ps
        #define SIMD_STORE _mm256_storeu_ps
        #define SIMD_SET1 _mm256_set1_ps
        #define SIMD_MUL _mm256_mul_ps
        #define SIMD_ADD _mm256_add_ps
        #define SIMD_FMA(a,b,c) SIMD_ADD(SIMD_MUL(a,b),c)
        #define SIMD_LOADI(p) _mm256_loadu_si256((const __m256i*)(p))
        #define SIMD_ZEROI() _mm256_setzero_si256()
        #define SIMD_SET1_16(x) _mm256_set1_epi16(x)
        #define SIMD_ADD32 _mm256_add_epi32
        #define SIMD_ABS8 _mm256_abs_epi8
        #define SIMD_SIGN8 _mm256_sign_epi8
        #define SIMD_MADDUBS _mm256_maddubs_epi16
        #define SIMD_MADD16 _mm256_madd_epi16
        #define SIMD_DP4A(acc,u,s) SIMD_ADD32(acc,SIMD_MADD16(SIMD_MADDUBS(u,s),SIMD_SET1_16(1)))
        #define SIMD_I8TOI32(p) _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(p)))
        #define SIMD_I32TOPS _mm256_cvtepi32_ps
        #define SIMD_HSUM32(v) ({__m256i _t=(v);__m128i _a=_mm_add_epi32(_mm256_castsi256_si128(_t),_mm256_extracti128_si256(_t,1));_a=_mm_add_epi32(_a,_mm_shuffle_epi32(_a,_MM_SHUFFLE(1,0,3,2)));_a=_mm_add_epi32(_a,_mm_shuffle_epi32(_a,_MM_SHUFFLE(2,3,0,1)));_mm_cvtsi128_si32(_a);})
        #define LONG 8
        #define IN_SET "avx2"
        #define IN_EXT ""
        if(cpu_can.fma3){
            #undef SIMD_FMA
            #undef IN_EXT
            #define FMA_
            #define SIMD_FMA _mm256_fmadd_ps
            #define IN_EXT ",fma"
        }
        #include"x86/x86_classic.txt"
        #include"../../matmul_libs/undef.h"
    }
    else{
        if((size_t)start_pos+(size_t)seq>(size_t)kvq.cap)
            error_exit("KV_OOM",EXIT_KV_OOM_ERROR);
        const int kv_dim=num_kv_heads*head_dim;
        kv_pack_q8(K,kvq.k+(size_t)start_pos*kv_dim,kvq.ks+(size_t)start_pos*num_kv_heads,
                   num_kv_heads,head_dim,seq);
        kv_pack_q8(V,kvq.v+(size_t)start_pos*kv_dim,kvq.vs+(size_t)start_pos*num_kv_heads,
                   num_kv_heads,head_dim,seq);
        const int group=num_heads/num_kv_heads;
        const float scale=1.0f/sqrtf((float)head_dim);
        par_for(0,num_heads,(size_t)num_heads*(size_t)(start_pos+seq)>=1536,[&](int64_t h0,int64_t h1){
        for (int64_t h=h0;h<h1;h++) {
            std::vector<float> scores((size_t)start_pos+seq);
            std::vector<int8_t> q8((size_t)head_dim);
            const int kv_h=(int)(h/group);
            for (int i=0;i<seq;i++) {
                const int p=start_pos+i;
                const float* q=Q+((size_t)i*num_heads+h)*head_dim;
                float amax=0.0f;
                for (int d=0;d<head_dim;d++) {
                    const float a=fabsf(q[d]);
                    if (a>amax) amax=a;
                }
                const float qinv=amax>1e-8f?127.0f/amax:0.0f;
                const float qsc=amax>1e-8f?amax/127.0f:0.0f;
                for (int d=0;d<head_dim;d++) {
                    float x=q[d]*qinv;
                    x=x>127.0f?127.0f:(x<-127.0f?-127.0f:x);
                    q8[d]=(int8_t)lrintf(x);
                }
                float mxs=-1e30f;
                for (int j=0;j<=p;j++) {
                    const int8_t* k=kvq.k+((size_t)j*num_kv_heads+kv_h)*head_dim;
                    const float ksc=kvq.ks[(size_t)j*num_kv_heads+kv_h]*qsc*scale;
                    int32_t s=0;
                    for (int d=0;d<head_dim;d++)
                        s+=(int32_t)q8[d]*(int32_t)k[d];
                    const float dt=(float)s*ksc;
                    scores[j]=dt;
                    if (dt>mxs) mxs=dt;
                }
                float sum=0.0f;
                for (int j=0;j<=p;j++) {
                    scores[j]=expf(scores[j]-mxs);
                    sum+=scores[j];
                }
                const float inv_sum=1.0f/sum;
                for (int j=0;j<=p;j++) scores[j]*=inv_sum;
                float* o=out+((size_t)i*num_heads+h)*head_dim;
                memset(o,0,head_dim*sizeof(float));
                for (int j=0;j<=p;j++) {
                    const int8_t* v=kvq.v+((size_t)j*num_kv_heads+kv_h)*head_dim;
                    const float pr=scores[j]*kvq.vs[(size_t)j*num_kv_heads+kv_h];
                    for (int d=0;d<head_dim;d++)
                        o[d]+=pr*(float)v[d];
                }
            }
        }
        });
    }
}
#endif
