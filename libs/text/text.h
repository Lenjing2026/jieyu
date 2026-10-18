#ifndef JIEYU_TEXT_TEXT_H
#define JIEYU_TEXT_TEXT_H


#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#if !defined(_WIN32)
#include<fcntl.h>
#include<sys/mman.h>
#include<sys/stat.h>
#include<unistd.h>
#endif

#include"libs.h"
#include"date_libs/date.h"
#include"model_libs/ffn/ffn.h"
#include"model_libs/rope/rope.h"
#include"model_libs/rmsnorm/rmsnorm.h"
#include"model_libs/attention/attention.h"
#include"model_libs/transformer_layer.h"
#include"model_libs/generate.h"
#include"model_libs/bfile.h"
#include"model_libs/safetensors.h"
#include"model_libs/converter.h"
#include"model_libs/load_weights.h"
#include"model_libs/pipeline.h"
#include"tokenizer_libs/include/byte_vocab.h"

#include<algorithm>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstdint>
#include<cstring>
#include<string>
#include<fstream>
#include<iterator>
#include<vector>
#include<system_error>
#include<filesystem>

int self_passed=0,self_total=0;

bool check_matmul(){
    struct { int m,n,k; float eps; } cases[]={
        {1,1,1,1e-4f},{2,2,2,1e-4f},{4,4,4,1e-4f},
        {8,8,8,1e-4f},{16,16,16,1e-4f},{32,32,32,1e-4f},
        {64,64,64,1e-4f},{1,100,50,1e-4f},{100,1,50,1e-4f},
        {100,50,1,1e-4f},{3,5,7,1e-4f},{7,13,17,1e-4f},
        {17,31,63,1e-4f},{33,65,129,1e-4f},{128,128,128,1e-3f},
        {256,256,256,1e-3f},{512,512,512,1e-3f},{1,4096,1,1e-4f},
        {4096,1,1,1e-4f},{1,1,4096,1e-4f},
    };
    int total=sizeof(cases) / sizeof(cases[0]);
    int passed=0;

    for(int ci=0;ci < total;ci++){
        int m=cases[ci].m,n=cases[ci].n,k=cases[ci].k;
        float eps=cases[ci].eps;
        size_t sA=(size_t)m * k,sB=(size_t)k * n,sC=(size_t)m * n;

        float* A=(float*)malloc(sA * sizeof(float));
        float* B=(float*)malloc(sB * sizeof(float));
        float* C_ref=(float*)malloc(sC * sizeof(float));
        float* C_got=(float*)malloc(sC * sizeof(float));

        if(!A||!B||!C_ref||!C_got){
            free(A); free(B); free(C_ref); free(C_got);
            continue;
        }

        srand(42 + ci);
        for(size_t i=0;i < sA;i++) A[i]=(rand() % 2000 - 1000) / 1000.0f;
        for(size_t i=0;i < sB;i++) B[i]=(rand() % 2000 - 1000) / 1000.0f;

        for(int i=0;i < m;i++)
            for(int j=0;j < n;j++){
                double s=0.0;
                for(int t=0;t < k;t++)
                    s+=(double)A[i*k+t] * (double)B[t*n+j];
                C_ref[i*n+j]=(float)s;
            }

        matmul(m,n,k,A,B,C_got);

        bool ok=true;
        for(size_t i=0;i < sC;i++){
            if(std::fabs(C_ref[i] - C_got[i]) > eps){
                printf("FAIL: m=%d n=%d k=%d idx=%zu ref=%.6f got=%.6f\n",
                       m,n,k,i,C_ref[i],C_got[i]);
                ok=false;
                break;
            }
        }
        if(ok) passed++;

        free(A); free(B); free(C_ref); free(C_got);
    }

    printf("check_matmul: %d/%d passed\n",passed,total);
    self_passed=passed;self_total=total;
    return passed==total;
}



namespace ffn_detail {

inline float pseudo(int i){
    return static_cast<float>((i * 37) % 19 - 9) / 8.0f;
}

inline double silu_ref(double v){
    return v / (1.0 + std::exp(-v));
}

inline void reference(const std::vector<float>& x,
                      const std::vector<float>& w1,
                      const std::vector<float>& w2,
                      const std::vector<float>& w3,
                      int seq,int hidden,int intermediate,
                      std::vector<float>& out){
    std::vector<double> acc(static_cast<size_t>(seq) * hidden,0.0);
    for(int i=0;i < seq;i++){
        for(int j=0;j < intermediate;j++){
            double g=0.0;
            double u=0.0;
            for(int k=0;k < hidden;k++){
                const double xv=x[static_cast<size_t>(i) * hidden + k];
                g+=xv * w1[static_cast<size_t>(k) * intermediate + j];
                u+=xv * w3[static_cast<size_t>(k) * intermediate + j];
            }
            const double h=silu_ref(g) * u;
            for(int o=0;o < hidden;o++){
                acc[static_cast<size_t>(i) * hidden + o]+=
                    h * w2[static_cast<size_t>(j) * hidden + o];
            }
        }
    }
    out.assign(static_cast<size_t>(seq) * hidden,0.0f);
    for(size_t i=0; i < out.size(); i++){
        out[i]=static_cast<float>(acc[i]);
    }
}

inline float rel_err(float got,float want){
    return std::fabs(got - want) / (1.0f + std::fabs(want));
}

inline float run_case(int seq,int hidden,int intermediate){
    const size_t sx=static_cast<size_t>(seq) * hidden;
    const size_t s1=static_cast<size_t>(hidden) * intermediate;
    const size_t s2=static_cast<size_t>(intermediate) * hidden;

    std::vector<float> x(sx),w1(s1),w3(s1),w2(s2),got(sx,0.0f);
    for(size_t i=0;i < sx;i++) x[i]=pseudo(static_cast<int>(i));
    for(size_t i=0;i < s1;i++) w1[i]=pseudo(static_cast<int>(i) + 3);
    for(size_t i=0;i < s1;i++) w3[i]=pseudo(static_cast<int>(i) + 5);
    for(size_t i=0;i < s2;i++) w2[i]=pseudo(static_cast<int>(i) + 7);

    std::vector<float> want;
    reference(x,w1,w2,w3,seq,hidden,intermediate,want);

    ffn(x.data(),w1.data(),w2.data(),w3.data(),got.data(),seq,hidden,intermediate);

    float worst=0.0f;
    for(size_t i=0; i < want.size()&&i < got.size(); i++){
        const float e=rel_err(got[i],want[i]);
        if(e > worst) worst=e;
    }
    return worst;
}

}

namespace rope_detail {

inline void make_q(std::vector<float>& q,int seq,int num_heads,int head_dim){
    const size_t n=static_cast<size_t>(seq) * num_heads * head_dim;
    q.assign(n,0.0f);
    for(size_t i=0;i < n;i++){
        q[i]=ffn_detail::pseudo(static_cast<int>(i));
    }
}

inline void make_tables(int seq,int head_dim,
                        std::vector<float>& cos_t,std::vector<float>& sin_t){
    const int half=head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * half,0.0f);
    sin_t.assign(static_cast<size_t>(seq) * half,0.0f);
    for(int pos=0;pos < seq;pos++){
        for(int i=0;i < half;i++){
            const double theta=static_cast<double>(pos) *
                                 std::pow(10000.0,-2.0 * i / head_dim);
            cos_t[static_cast<size_t>(pos) * half + i]=static_cast<float>(std::cos(theta));
            sin_t[static_cast<size_t>(pos) * half + i]=static_cast<float>(std::sin(theta));
        }
    }
}

inline void make_const_tables(int seq,int head_dim,double c,double s,
                              std::vector<float>& cos_t,std::vector<float>& sin_t){
    const int half=head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * half,static_cast<float>(c));
    sin_t.assign(static_cast<size_t>(seq) * half,static_cast<float>(s));
}

inline std::vector<float> reference(const std::vector<float>& q,int seq,int num_heads,
                                    int head_dim,const std::vector<float>& cos_t,
                                    const std::vector<float>& sin_t){
    const int half=head_dim / 2;
    std::vector<float> out=q;
    for(int pos=0;pos < seq;pos++){
        for(int h=0;h < num_heads;h++){
            const size_t base=(static_cast<size_t>(pos) * num_heads + h) * head_dim;
            for(int i=0;i < half;i++){
                const double x0=q[base + 2 * i];
                const double x1=q[base + 2 * i + 1];
                const double c=cos_t[static_cast<size_t>(pos) * half + i];
                const double s=sin_t[static_cast<size_t>(pos) * half + i];
                out[base + 2 * i]=static_cast<float>(x0 * c - x1 * s);
                out[base + 2 * i + 1]=static_cast<float>(x0 * s + x1 * c);
            }
        }
    }
    return out;
}

inline float worst_err(const std::vector<float>& got,const std::vector<float>& want){
    float worst=0.0f;
    for(size_t i=0; i < want.size()&&i < got.size(); i++){
        const float e=ffn_detail::rel_err(got[i],want[i]);
        if(e > worst){
            worst=e;
        }
    }
    return worst;
}

}

namespace rms_detail {

inline std::vector<float> reference(const std::vector<float>& x,const std::vector<float>& w,
                                    int n,float eps){
    double ss=0.0;
    for(int i=0;i < n;i++){
        ss+=static_cast<double>(x[i]) * x[i];
    }
    const double scale=1.0 / std::sqrt(ss / n + static_cast<double>(eps));
    std::vector<float> out(static_cast<size_t>(n),0.0f);
    for(int i=0;i < n;i++){
        out[i]=static_cast<float>(static_cast<double>(x[i]) * scale * w[i]);
    }
    return out;
}

inline float worst_err(const std::vector<float>& got,const std::vector<float>& want){
    float worst=0.0f;
    for(size_t i=0; i < want.size()&&i < got.size(); i++){
        const float e=ffn_detail::rel_err(got[i],want[i]);
        if(e > worst){
            worst=e;
        }
    }
    return worst;
}

}

namespace attn_detail {

inline void fill_random(std::vector<float>& v,int seed){
    for(size_t i=0; i < v.size(); i++){
        v[i]=ffn_detail::pseudo(static_cast<int>(i) + seed);
    }
}

inline std::vector<float> reference(const std::vector<float>& Q,const std::vector<float>& K,
                                    const std::vector<float>& V,int seq,int num_heads,
                                    int num_kv_heads,int head_dim){
    const int group=num_heads / num_kv_heads;
    const double scale=1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<float> out(static_cast<size_t>(seq) * num_heads * head_dim,0.0f);
    std::vector<double> sc(static_cast<size_t>(seq),0.0);
    for(int h=0;h < num_heads;h++){
        const int kv_h=h / group;
        for(int i=0;i < seq;i++){
            const size_t qb=(static_cast<size_t>(i) * num_heads + h) * head_dim;
            double mx=-1e300;
            for(int j=0;j<=i;j++){
                const size_t kb=(static_cast<size_t>(j) * num_kv_heads + kv_h) * head_dim;
                double dt=0.0;
                for(int d=0;d < head_dim;d++){
                    dt+=static_cast<double>(Q[qb + d]) * K[kb + d];
                }
                dt*=scale;
                sc[j]=dt;
                if(dt > mx){
                    mx=dt;
                }
            }
            double sum=0.0;
            for(int j=0;j<=i;j++){
                sc[j]=std::exp(sc[j] - mx);
                sum+=sc[j];
            }
            const size_t ob=(static_cast<size_t>(i) * num_heads + h) * head_dim;
            for(int d=0;d < head_dim;d++){
                double acc=0.0;
                for(int j=0;j<=i;j++){
                    const size_t vb=(static_cast<size_t>(j) * num_kv_heads + kv_h) * head_dim;
                    acc+=sc[j] / sum * V[vb + d];
                }
                out[ob + d]=static_cast<float>(acc);
            }
        }
    }
    return out;
}

inline float worst_err(const std::vector<float>& got,const std::vector<float>& want){
    float worst=0.0f;
    for(size_t i=0; i < want.size()&&i < got.size(); i++){
        const float e=ffn_detail::rel_err(got[i],want[i]);
        if(e > worst){
            worst=e;
        }
    }
    return worst;
}

inline float run_case(int seq,int num_heads,int num_kv_heads,int head_dim){
    const size_t sq=static_cast<size_t>(seq) * num_heads * head_dim;
    const size_t skv=static_cast<size_t>(seq) * num_kv_heads * head_dim;
    std::vector<float> Q(sq),K(skv),V(skv),got(sq,0.0f);
    fill_random(Q,1);
    fill_random(K,2);
    fill_random(V,3);
    const std::vector<float> want=reference(Q,K,V,seq,num_heads,num_kv_heads,head_dim);
    attention(Q.data(),K.data(),V.data(),got.data(),seq,0,num_heads,num_kv_heads,head_dim);
    return worst_err(got,want);
}

}

namespace layer_detail {

inline void dmatmul(int m,int n,int k,const std::vector<float>& A,
                    const std::vector<float>& B,std::vector<float>& C){
    C.assign(static_cast<size_t>(m) * n,0.0f);
    for(int i=0;i < m;i++){
        for(int j=0;j < n;j++){
            double s=0.0;
            for(int t=0;t < k;t++){
                s+=static_cast<double>(A[static_cast<size_t>(i) * k + t]) *
                     B[static_cast<size_t>(t) * n + j];
            }
            C[static_cast<size_t>(i) * n + j]=static_cast<float>(s);
        }
    }
}

inline void rms_rows(const std::vector<float>& src,const std::vector<float>& w,
                     int seq,int n,float eps,std::vector<float>& dst){
    dst.assign(static_cast<size_t>(seq) * n,0.0f);
    for(int i=0;i < seq;i++){
        std::vector<float> row(src.begin() + static_cast<size_t>(i) * n,
                               src.begin() + static_cast<size_t>(i + 1) * n);
        const std::vector<float> out=rms_detail::reference(row,w,n,eps);
        for(int d=0;d < n;d++){
            dst[static_cast<size_t>(i) * n + d]=out[d];
        }
    }
}

inline void reference(const std::vector<float>& X0,
                      const std::vector<float>& rms1,
                      const std::vector<float>& Wq,const std::vector<float>& Wk,
                      const std::vector<float>& Wv,const std::vector<float>& Wo,
                      const std::vector<float>& cos_t,const std::vector<float>& sin_t,
                      const std::vector<float>& rms2,
                      const std::vector<float>& W1,const std::vector<float>& W2,
                      const std::vector<float>& W3,
                      int seq,int hidden,int num_heads,int num_kv_heads,
                      int head_dim,int intermediate,float eps,
                      std::vector<float>& out){
    const int q_dim=num_heads * head_dim;
    const int kv_dim=num_kv_heads * head_dim;

    std::vector<float> X(X0.begin(),X0.begin() + static_cast<size_t>(seq) * hidden);
    std::vector<float> xn;

    rms_rows(X,rms1,seq,hidden,eps,xn);
    std::vector<float> Q,K,V;
    dmatmul(seq,q_dim,hidden,xn,Wq,Q);
    dmatmul(seq,kv_dim,hidden,xn,Wk,K);
    dmatmul(seq,kv_dim,hidden,xn,Wv,V);
    Q=rope_detail::reference(Q,seq,num_heads,head_dim,cos_t,sin_t);
    K=rope_detail::reference(K,seq,num_kv_heads,head_dim,cos_t,sin_t);
    const std::vector<float> attn=
        attn_detail::reference(Q,K,V,seq,num_heads,num_kv_heads,head_dim);
    std::vector<float> proj;
    dmatmul(seq,hidden,q_dim,attn,Wo,proj);
    for(size_t i=0; i < X.size(); i++){
        X[i]+=proj[i];
    }
    rms_rows(X,rms2,seq,hidden,eps,xn);
    std::vector<float> ffn_out;
    ffn_detail::reference(xn,W1,W2,W3,seq,hidden,intermediate,ffn_out);
    for(size_t i=0; i < X.size(); i++){
        X[i]+=ffn_out[i];
    }
    out=X;
}

struct layer_weights {
    int seq,hidden,num_heads,num_kv_heads,head_dim,intermediate;
    std::vector<float> X,rms1,Wq,Wk,Wv,Wo,cos_t,sin_t,rms2,W1,W2,W3;

    void build(int s,int h,int nh,int nkv,int hd,int inter,bool random_attn=true){
        seq=s; hidden=h; num_heads=nh; num_kv_heads=nkv;
        head_dim=hd; intermediate=inter;
        const int q_dim=nh * hd;
        const int kv_dim=nkv * hd;
        X.assign(static_cast<size_t>(s) * h,0.0f);
        rms1.assign(h,1.0f);
        rms2.assign(h,1.0f);
        Wq.assign(static_cast<size_t>(h) * q_dim,0.0f);
        Wk.assign(static_cast<size_t>(h) * kv_dim,0.0f);
        Wv.assign(static_cast<size_t>(h) * kv_dim,0.0f);
        Wo.assign(static_cast<size_t>(q_dim) * h,0.0f);
        W1.assign(static_cast<size_t>(h) * inter,0.0f);
        W2.assign(static_cast<size_t>(inter) * h,0.0f);
        W3.assign(static_cast<size_t>(h) * inter,0.0f);
        for(size_t i=0; i < X.size(); i++) X[i]=ffn_detail::pseudo(static_cast<int>(i)) * 2.0f;
        for(int i=0;i < h;i++){
            rms1[i]=1.0f + ffn_detail::pseudo(i + 1) * 0.25f;
            rms2[i]=1.0f + ffn_detail::pseudo(i + 2) * 0.25f;
        }
        if(random_attn){
            for(size_t i=0; i < Wq.size(); i++) Wq[i]=ffn_detail::pseudo(static_cast<int>(i) + 3) * 0.5f;
            for(size_t i=0; i < Wk.size(); i++) Wk[i]=ffn_detail::pseudo(static_cast<int>(i) + 5) * 0.5f;
            for(size_t i=0; i < Wv.size(); i++) Wv[i]=ffn_detail::pseudo(static_cast<int>(i) + 7) * 0.5f;
            for(size_t i=0; i < Wo.size(); i++) Wo[i]=ffn_detail::pseudo(static_cast<int>(i) + 11) * 0.5f;
        }
        for(size_t i=0; i < W1.size(); i++) W1[i]=ffn_detail::pseudo(static_cast<int>(i) + 13) * 0.5f;
        for(size_t i=0; i < W2.size(); i++) W2[i]=ffn_detail::pseudo(static_cast<int>(i) + 17) * 0.5f;
        for(size_t i=0; i < W3.size(); i++) W3[i]=ffn_detail::pseudo(static_cast<int>(i) + 19) * 0.5f;
        rope_detail::make_tables(s,hd,cos_t,sin_t);
    }

    void run(float* x,float eps=1e-5f) const {
        transformer_layer(x,rms1.data(),Wq.data(),Wk.data(),Wv.data(),Wo.data(),
                          cos_t.data(),sin_t.data(),rms2.data(),W1.data(),W2.data(),
                          W3.data(),seq,hidden,num_heads,num_kv_heads,head_dim,
                          intermediate,eps);
    }
};

inline float run_case(int seq,int hidden,int num_heads,int num_kv_heads,
                      int head_dim,int intermediate,float eps){
    layer_weights w;
    w.build(seq,hidden,num_heads,num_kv_heads,head_dim,intermediate);

    std::vector<float> want;
    reference(w.X,w.rms1,w.Wq,w.Wk,w.Wv,w.Wo,w.cos_t,w.sin_t,w.rms2,
              w.W1,w.W2,w.W3,seq,hidden,num_heads,num_kv_heads,head_dim,
              intermediate,eps,want);

    std::vector<float> got=w.X;
    w.run(got.data(),eps);
    return rms_detail::worst_err(got,want);
}

}

namespace gen_detail {

struct gen_weights {
    int vocab,hidden,num_heads,num_kv_heads,head_dim,intermediate,num_layers,max_seq;
    std::vector<float> embed,lm_head,final_rms,cos_t,sin_t;
    std::vector<std::vector<float>> rms1,Wq,Wk,Wv,Wo,rms2,W1,W2,W3;
    std::vector<LayerWeights> layers;

    void build(int v,int h,int nh,int nkv,int hd,int inter,int nl,int mseq,bool zero_lm){
        vocab=v; hidden=h; num_heads=nh; num_kv_heads=nkv; head_dim=hd;
        intermediate=inter; num_layers=nl; max_seq=mseq;
        const int qdim=nh * hd;
        const int kvdim=nkv * hd;
        rms1.clear(); rms2.clear(); Wq.clear(); Wk.clear(); Wv.clear(); Wo.clear();
        W1.clear(); W2.clear(); W3.clear(); layers.clear();
        for(int l=0;l < nl;l++){
            rms1.push_back(std::vector<float>(h,1.0f));
            rms2.push_back(std::vector<float>(h,1.0f));
            Wq.push_back(std::vector<float>(static_cast<size_t>(h) * qdim));
            Wk.push_back(std::vector<float>(static_cast<size_t>(h) * kvdim));
            Wv.push_back(std::vector<float>(static_cast<size_t>(h) * kvdim));
            Wo.push_back(std::vector<float>(static_cast<size_t>(qdim) * h));
            W1.push_back(std::vector<float>(static_cast<size_t>(h) * inter));
            W2.push_back(std::vector<float>(static_cast<size_t>(inter) * h));
            W3.push_back(std::vector<float>(static_cast<size_t>(h) * inter));
        }
        for(int l=0;l < nl;l++){
            for(size_t i=0; i < rms1[l].size(); i++) rms1[l][i]=1.0f + ffn_detail::pseudo(static_cast<int>(i) + 3 * l) * 0.25f;
            for(size_t i=0; i < rms2[l].size(); i++) rms2[l][i]=1.0f + ffn_detail::pseudo(static_cast<int>(i) + 5 * l) * 0.25f;
            for(size_t i=0; i < Wq[l].size(); i++) Wq[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 7 * l) * 0.5f;
            for(size_t i=0; i < Wk[l].size(); i++) Wk[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 11 * l) * 0.5f;
            for(size_t i=0; i < Wv[l].size(); i++) Wv[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 13 * l) * 0.5f;
            for(size_t i=0; i < Wo[l].size(); i++) Wo[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 17 * l) * 0.5f;
            for(size_t i=0; i < W1[l].size(); i++) W1[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 19 * l) * 0.5f;
            for(size_t i=0; i < W2[l].size(); i++) W2[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 23 * l) * 0.5f;
            for(size_t i=0; i < W3[l].size(); i++) W3[l][i]=ffn_detail::pseudo(static_cast<int>(i) + 29 * l) * 0.5f;
        }
        embed.assign(static_cast<size_t>(v) * h,0.0f);
        for(size_t i=0; i < embed.size(); i++) embed[i]=ffn_detail::pseudo(static_cast<int>(i) + 101) * 0.5f;
        lm_head.assign(static_cast<size_t>(h) * v,0.0f);
        if(!zero_lm){
            for(size_t i=0; i < lm_head.size(); i++) lm_head[i]=ffn_detail::pseudo(static_cast<int>(i) + 211) * 0.5f;
        }
        final_rms.assign(h,1.0f);
        for(size_t i=0; i < final_rms.size(); i++) final_rms[i]=1.0f + ffn_detail::pseudo(static_cast<int>(i) + 307) * 0.25f;
        rope_detail::make_tables(mseq,hd,cos_t,sin_t);
        layers.resize(nl);
        for(int l=0;l < nl;l++){
            layers[l].rms1_weight=rms1[l].data();
            layers[l].Wq=Wq[l].data();
            layers[l].Wk=Wk[l].data();
            layers[l].Wv=Wv[l].data();
            layers[l].Wo=Wo[l].data();
            layers[l].rms2_weight=rms2[l].data();
            layers[l].W1=W1[l].data();
            layers[l].W2=W2[l].data();
            layers[l].W3=W3[l].data();
        }
    }

    int run(const int* prompt,int prompt_len,int n_new,float temp,int* out) const {
        return generate(prompt,prompt_len,nullptr,0,embed.data(),lm_head.data(),
                        layers.data(),num_layers,final_rms.data(),cos_t.data(),
                        sin_t.data(),max_seq,n_new,temp,vocab,hidden,
                        num_heads,num_kv_heads,head_dim,intermediate,out);
    }
};

inline int reference_ids(const gen_weights& w,const int* prompt,int prompt_len,
                         int n_new,int* out){
    const float eps=1e-5f;
    for(int i=0;i < prompt_len;i++){
        out[i]=prompt[i];
    }
    int len=prompt_len;
    std::vector<float> X(static_cast<size_t>(w.max_seq) * w.hidden);
    for(int step=0;step < n_new&&len < w.max_seq;step++){
        for(int i=0;i < len;i++){
            const float* row=w.embed.data() + static_cast<size_t>(out[i]) * w.hidden;
            for(int d=0;d < w.hidden;d++){
                X[static_cast<size_t>(i) * w.hidden + d]=row[d];
            }
        }
        std::vector<float> hs;
        for(int l=0;l < w.num_layers;l++){
            const std::vector<float>& src=(l==0)?X:hs;
            std::vector<float> tmp;
            layer_detail::reference(src,w.rms1[l],w.Wq[l],w.Wk[l],w.Wv[l],w.Wo[l],
                                    w.cos_t,w.sin_t,w.rms2[l],w.W1[l],w.W2[l],
                                    w.W3[l],len,w.hidden,w.num_heads,w.num_kv_heads,
                                    w.head_dim,w.intermediate,eps,tmp);
            hs=tmp;
        }
        for(int i=0;i < len;i++){
            std::vector<float> row(hs.begin() + static_cast<size_t>(i) * w.hidden,
                                   hs.begin() + static_cast<size_t>(i + 1) * w.hidden);
            const std::vector<float> nr=rms_detail::reference(row,w.final_rms,w.hidden,eps);
            for(int d=0;d < w.hidden;d++){
                hs[static_cast<size_t>(i) * w.hidden + d]=nr[d];
            }
        }
        const std::vector<float> last(hs.begin() + static_cast<size_t>(len - 1) * w.hidden,
                                      hs.begin() + static_cast<size_t>(len) * w.hidden);
        std::vector<float> logits;
        layer_detail::dmatmul(1,w.vocab,w.hidden,last,w.lm_head,logits);
        int best=0;
        for(int v=1;v < w.vocab;v++){
            if(logits[v] > logits[best]){
                best=v;
            }
        }
        out[len++]=best;
    }
    return len;
}

}


namespace pipe_detail {

inline void make_weights(gen_detail::gen_weights& w,int vocab,int hidden,int heads,int kv,
                         int head_dim,int inter,int layers,int max_seq){
    w.build(vocab,hidden,heads,kv,head_dim,inter,layers,max_seq,false);
    for(size_t i=0; i < w.lm_head.size(); i++){
        w.lm_head[i]=ffn_detail::pseudo(static_cast<int>(i) + 211) * 2.0f;
    }
}

inline void write_model_dir(const std::filesystem::path& dir,const gen_detail::gen_weights& w,
                            bool wrong_q=false,bool drop_final_norm=false){
    namespace fs=std::filesystem;
    std::error_code ec;
    fs::create_directories(dir,ec);
    const uint32_t h=static_cast<uint32_t>(w.hidden);
    const uint32_t qd=static_cast<uint32_t>(w.num_heads * w.head_dim);
    const uint32_t kvd=static_cast<uint32_t>(w.num_kv_heads * w.head_dim);
    const uint32_t inter=static_cast<uint32_t>(w.intermediate);
    const uint32_t vocab=static_cast<uint32_t>(w.vocab);

    bfile::Header head;
    head.hidden=w.hidden;
    head.layers=w.num_layers;
    head.mode=1;
    head.num_heads=static_cast<uint8_t>(w.num_heads);
    head.num_kv_heads=static_cast<uint8_t>(w.num_kv_heads);
    head.head_dim=static_cast<uint8_t>(w.head_dim);
    head.intermediate=inter;
    head.vocab_size=vocab;

    std::vector<bfile::Tensor> ts;
    ts.push_back(bfile::make_f32("model.embed_tokens.weight",{vocab,h},w.embed));
    if(!drop_final_norm) ts.push_back(bfile::make_f32("model.norm.weight",{h},w.final_rms));
    for(int l=0;l < w.num_layers;l++){
        const std::vector<uint32_t> q_shape=wrong_q
            ?std::vector<uint32_t>{qd,h}
            :std::vector<uint32_t>{h,qd};
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"input_layernorm.weight"),{h},w.rms1[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"self_attn.q_proj.weight"),q_shape,w.Wq[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"self_attn.k_proj.weight"),{h,kvd},w.Wk[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"self_attn.v_proj.weight"),{h,kvd},w.Wv[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"self_attn.o_proj.weight"),{qd,h},w.Wo[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"post_attention_layernorm.weight"),{h},w.rms2[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"mlp.gate_proj.weight"),{h,inter},w.W1[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"mlp.down_proj.weight"),{inter,h},w.W2[l]));
        ts.push_back(bfile::make_f32(layer_tensor_name(l,"mlp.up_proj.weight"),{h,inter},w.W3[l]));
    }
    ts.push_back(bfile::make_f32("lm_head.weight",{h,vocab},w.lm_head));
    bfile::write_bfile((dir / "model.bf").string(),head,ts);

    json_lib info;
    info.SetJsonWay((dir / "model_info.json").string());
    info.WriteJsonKey("hidden_size",std::to_string(w.hidden));
    info.WriteJsonKey("layer_count",std::to_string(w.num_layers));
    info.WriteJsonKey("num_heads",std::to_string(w.num_heads));
    info.WriteJsonKey("num_kv_heads",std::to_string(w.num_kv_heads));
    info.WriteJsonKey("head_dim",std::to_string(w.head_dim));
    info.WriteJsonKey("intermediate_size",std::to_string(w.intermediate));
    info.WriteJsonKey("vocab_size",std::to_string(w.vocab));
    info.WriteJsonKey("rope_theta","10000");
    info.WriteJsonKey("max_position_embeddings",std::to_string(w.max_seq));
}

inline bool load_gen_weights(gen_detail::gen_weights& w,int max_seq){
    if(!bfile::current().ok) return false;
    w.vocab=static_cast<int>(model.vocab_size);
    w.hidden=static_cast<int>(model.h);
    w.num_heads=model.num_heads;
    w.num_kv_heads=model.num_kv_heads;
    w.head_dim=model.head_dim;
    w.intermediate=static_cast<int>(model.intermediate_size);
    w.num_layers=static_cast<int>(model.l);
    w.max_seq=max_seq;

    const size_t h=static_cast<size_t>(w.hidden);
    const size_t qd=static_cast<size_t>(w.num_heads) * w.head_dim;
    const size_t kvd=static_cast<size_t>(w.num_kv_heads) * w.head_dim;
    const size_t inter=static_cast<size_t>(w.intermediate);

    auto grab=[](const std::string& name,std::vector<float>& dst,size_t count)->bool {
        const float* p=tensor_f32(name);
        if(p==nullptr) return false;
        dst.assign(p,p + count);
        return true;
    };

    if(!grab("model.embed_tokens.weight",w.embed,h * static_cast<size_t>(w.vocab))) return false;
    if(!grab("model.norm.weight",w.final_rms,h)) return false;
    const WMat lm=bind_lm_head(w.hidden,w.vocab);
    if(lm.empty()) return false;
    w.lm_head.assign(static_cast<size_t>(h) * w.vocab,0.0f);
    for(int r=0; r < (int)h; r++)
        wmat_row(lm,static_cast<size_t>(r),w.vocab,w.lm_head.data() + static_cast<size_t>(r) * w.vocab);

    w.rms1.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.rms2.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.Wq.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.Wk.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.Wv.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.Wo.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.W1.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.W2.assign(static_cast<size_t>(w.num_layers),std::vector<float>());
    w.W3.assign(static_cast<size_t>(w.num_layers),std::vector<float>());

    for(int l=0;l < w.num_layers;l++){
        if(!grab(layer_tensor_name(l,"input_layernorm.weight"),w.rms1[l],h)) return false;
        if(!grab(layer_tensor_name(l,"self_attn.q_proj.weight"),w.Wq[l],h * qd)) return false;
        if(!grab(layer_tensor_name(l,"self_attn.k_proj.weight"),w.Wk[l],h * kvd)) return false;
        if(!grab(layer_tensor_name(l,"self_attn.v_proj.weight"),w.Wv[l],h * kvd)) return false;
        if(!grab(layer_tensor_name(l,"self_attn.o_proj.weight"),w.Wo[l],qd * h)) return false;
        if(!grab(layer_tensor_name(l,"post_attention_layernorm.weight"),w.rms2[l],h)) return false;
        if(!grab(layer_tensor_name(l,"mlp.gate_proj.weight"),w.W1[l],h * inter)) return false;
        if(!grab(layer_tensor_name(l,"mlp.down_proj.weight"),w.W2[l],inter * h)) return false;
        if(!grab(layer_tensor_name(l,"mlp.up_proj.weight"),w.W3[l],h * inter)) return false;
    }

    w.layers.assign(static_cast<size_t>(w.num_layers),LayerWeights{});
    for(int l=0;l < w.num_layers;l++){
        w.layers[l].rms1_weight=w.rms1[l].data();
        w.layers[l].Wq=w.Wq[l].data();
        w.layers[l].Wk=w.Wk[l].data();
        w.layers[l].Wv=w.Wv[l].data();
        w.layers[l].Wo=w.Wo[l].data();
        w.layers[l].rms2_weight=w.rms2[l].data();
        w.layers[l].W1=w.W1[l].data();
        w.layers[l].W2=w.W2[l].data();
        w.layers[l].W3=w.W3[l].data();
    }
    rope_detail::make_tables(max_seq,w.head_dim,w.cos_t,w.sin_t);
    return true;
}

}

bool check_bind_weights(){
    namespace fs=std::filesystem;
    using pipe_detail::load_gen_weights;
    using pipe_detail::make_weights;
    using pipe_detail::write_model_dir;

    const int total=9;
    int passed=0;
    printf("==== 按名字装载权重测试 ====\n");

    const fs::path root=fs::path("_file") / "bind_test";
    const fs::path dir=root / "model";
    std::error_code ec;
    fs::remove_all(root,ec);

    const int vocab=24,hidden=8,heads=2,kv=1,head_dim=4;
    const int inter=16,layers=2,max_seq=8;

    gen_detail::gen_weights w;
    make_weights(w,vocab,hidden,heads,kv,head_dim,inter,layers,max_seq);
    write_model_dir(dir,w);

    free_model_map();
    bool loaded=false;
    try {
        install_model(dir);
        loaded=bfile::current().ok;
    } catch(const std::exception& e){
        printf("       ! install_model 抛异常: %s\n",e.what());
    }

    {
        const size_t want=1 + static_cast<size_t>(layers) * 9 + 1 + 1;
        const bool ok=loaded&&bfile::current().tensor_count==want;
        printf("[ 1/%d] install_model 成功，张量 %zu 个          %s\n",total,want,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::vector<LayerWeights> got(static_cast<size_t>(layers));
        std::string missing;
        const bool bound=bind_layers(got.data(),layers,&missing);
        bool ok=bound;
        if(ok) ok=got[1].Wq.f==tensor_f32(layer_tensor_name(1,"self_attn.q_proj.weight"));
        if(ok) ok=got[0].W2.f==tensor_f32(layer_tensor_name(0,"mlp.down_proj.weight"));
        if(ok) ok=got[1].rms2_weight==tensor_f32(layer_tensor_name(1,"post_attention_layernorm.weight"));
        if(ok) ok=got[0].Wq.f!=got[1].Wq.f;
        if(!bound) printf("       ! bind_layers 失败于 %s\n",missing.c_str());
        printf("[ 2/%d] 9 个权重指针都指到对应张量              %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const float* embed=tensor_f32("model.embed_tokens.weight");
        const float* wq=tensor_f32(layer_tensor_name(1,"self_attn.q_proj.weight"));
        bool ok=embed!=nullptr&&wq!=nullptr;
        if(ok) ok=std::memcmp(embed,w.embed.data(),w.embed.size() * 4)==0;
        if(ok) ok=std::memcmp(wq,w.Wq[1].data(),w.Wq[1].size() * 4)==0;
        printf("[ 3/%d] embed / 第1层 Wq 数值逐字节一致          %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    pipeline::ModelRefs refs;
    std::string err;
    bool bound_model=false;
    {
        bound_model=pipeline::bind_model(refs,max_seq,&err);
        if(!bound_model) printf("       ! %s\n",err.c_str());
        printf("[ 4/%d] bind_model 形状校验全过                  %s\n",total,bound_model?"OK":"FAIL");
        if(bound_model) passed++;
    }

    gen_detail::gen_weights ref;
    bool ref_ok=false;
    {
        ref_ok=bound_model&&load_gen_weights(ref,max_seq);
        printf("[ 5/%d] 参考实现从同一份文件读出权重            %s\n",total,ref_ok?"OK":"FAIL");
        if(ref_ok) passed++;
    }

    {
        const int prompt[3]={1,2,3};
        std::vector<int> got(6,-1),want(6,-1);
        int len_got=-1,len_want=-1;
        if(ref_ok){
            len_got=generate(prompt,3,nullptr,0,refs.embed,refs.lm_head,refs.layers.data(),
                               refs.num_layers,refs.final_norm,refs.rope.cos.data(),
                               refs.rope.sin.data(),refs.max_seq,3,0.0f,refs.vocab_size,
                               refs.hidden,refs.num_heads,refs.num_kv_heads,refs.head_dim,
                               refs.intermediate,got.data());
            len_want=gen_detail::reference_ids(ref,prompt,3,3,want.data());
        }
        bool ok=ref_ok&&len_got==len_want;
        if(ok){
            for(int i=0;i < len_got;i++){
                if(got[i]!=want[i]) ok=false;
            }
        }
        printf("[ 6/%d] 生成结果与 double 参考一致 (实际",total);
        for(int i=3;i < len_got&&i < 6;i++) printf(" %d",got[i]);
        printf(" 期望");
        for(int i=3;i < len_want&&i < 6;i++) printf(" %d",want[i]);
        printf(")  %s\n",ok?"OK":"FAIL");
        if(ok){
            passed++;
        }
    }

    {
        const fs::path bad_dir=root / "wrong_shape";
        gen_detail::gen_weights w6;
        make_weights(w6,24,8,2,1,6,16,1,max_seq);
        write_model_dir(bad_dir,w6,true,false);
        free_model_map();
        bool ok=false;
        try {
            install_model(bad_dir);
            pipeline::ModelRefs bad_refs;
            std::string bad_err;
            const bool bound_bad=pipeline::bind_model(bad_refs,max_seq,&bad_err);
            ok=!bound_bad&&bad_err.find("q_proj")!=std::string::npos;
            if(!ok) printf("       ! 竟然通过了，错误信息: %s\n",bad_err.c_str());
        } catch(const std::exception& e){
            printf("       ! 抛异常: %s\n",e.what());
        }
        printf("[ 7/%d] q_proj 形状写反 -> 报错并指出张量名      %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const fs::path bad_dir=root / "missing";
        gen_detail::gen_weights w8;
        make_weights(w8,24,8,2,1,4,16,1,max_seq);
        write_model_dir(bad_dir,w8,false,true);
        free_model_map();
        bool ok=false;
        try {
            install_model(bad_dir);
            pipeline::ModelRefs bad_refs;
            std::string bad_err;
            const bool bound_bad=pipeline::bind_model(bad_refs,max_seq,&bad_err);
            ok=!bound_bad&&bad_err.find("model.norm.weight")!=std::string::npos;
            if(!ok) printf("       ! 竟然通过了，错误信息: %s\n",bad_err.c_str());
        } catch(const std::exception& e){
            printf("       ! 抛异常: %s\n",e.what());
        }
        printf("[ 8/%d] 缺 model.norm.weight -> 报出名字          %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        free_model_map();
        install_model(dir);
        pipeline::ModelRefs r2;
        std::string e2;
        bool ok=pipeline::bind_model(r2,max_seq,&e2);
        const int prompt[3]={5,6,7};
        std::vector<int> a(6,-1),b(6,-1);
        if(ok){
            const int la=generate(prompt,3,nullptr,0,r2.embed,r2.lm_head,r2.layers.data(),
                                    r2.num_layers,r2.final_norm,r2.rope.cos.data(),
                                    r2.rope.sin.data(),r2.max_seq,3,0.0f,r2.vocab_size,
                                    r2.hidden,r2.num_heads,r2.num_kv_heads,r2.head_dim,
                                    r2.intermediate,a.data());
            const int lb=generate(prompt,3,nullptr,0,r2.embed,r2.lm_head,r2.layers.data(),
                                    r2.num_layers,r2.final_norm,r2.rope.cos.data(),
                                    r2.rope.sin.data(),r2.max_seq,3,0.0f,r2.vocab_size,
                                    r2.hidden,r2.num_heads,r2.num_kv_heads,r2.head_dim,
                                    r2.intermediate,b.data());
            ok=(la==lb);
            for(int i=0;i < la&&ok;i++){
                if(a[i]!=b[i]) ok=false;
            }
        }
        printf("[ 9/%d] 同一权重跑两次结果一致                  %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    free_model_map();
    printf("check_bind_weights: %d/%d passed\n",passed,total);
    self_passed=passed;self_total=total;
    return passed==total;
}

bool check_pipeline(){
    namespace fs=std::filesystem;

    const int total=10;
    int passed=0;
    printf("==== 端到端（文本进 -> 文本出）测试 ====\n");

    const fs::path root=fs::path("_file") / "pipe_test";
    const fs::path src=root / "src";
    const fs::path dir=root / "model";
    std::error_code ec;
    fs::remove_all(root,ec);

    converter::DemoSpec spec;
    bool made=false;
    try {
        converter::make_demo_model(src.string(),spec);
        converter::Options opt;
        opt.sources.push_back((src / "model.safetensors").string());
        opt.out_dir=dir.string();
        opt.config_path=(src / "config.json").string();
        opt.tokenizer_dir=src.string();
        opt.verbose=false;
        converter::convert(opt);
        made=true;
    } catch(const std::exception& e){
        printf("       ! 造/转模型失败: %s\n",e.what());
    }

    free_model_map();
    pipeline::LoadedModel m;
    std::string err;
    const bool loaded=made&&pipeline::load_model_dir(m,dir.string(),64,&err);
    if(!loaded) printf("       ! %s\n",err.empty()?"装载失败":err.c_str());

    {
        const bool ok=loaded&&m.refs.num_layers==spec.layers
            &&m.refs.hidden==spec.hidden&&m.refs.vocab_size==spec.vocab;
        printf("[ 1/%d] 装载模型目录成功                        %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const bool ok=loaded&&m.tokenizer.vocab_size()==259;
        printf("[ 2/%d] 词表 %zu 个 token（256 字节 + 3 特殊）  %s\n",total,
               loaded?m.tokenizer.vocab_size():0,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const std::string text="Hello, JieYu AI!";
        const std::string back=loaded?m.tokenizer.decode(m.tokenizer.encode(text)):"";
        const bool ok=loaded&&back==text;
        printf("[ 3/%d] decode(encode(英文)) 还原原文            %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const std::string text="你好，世界！昆仑分词器。";
        const std::string back=loaded?m.tokenizer.decode(m.tokenizer.encode(text)):"";
        const bool ok=loaded&&back==text;
        printf("[ 4/%d] decode(encode(中文)) 还原原文            %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        const std::vector<int32_t> ids=loaded?m.tokenizer.encode("abc"):std::vector<int32_t>();
        const bool ok=loaded&&ids.size()==3&&ids[0]==97&&ids[1]==98&&ids[2]==99;
        printf("[ 5/%d] encode(\"abc\") = {97,98,99}              %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::string out;
        const int produced=loaded?pipeline::generate_text(m.tokenizer,"abcdefgh",m.refs,16,0.0f,out)
            :-1;
        const bool ok=loaded&&out=="ijklmnopqrstuvwx"&&produced==16;
        printf("[ 6/%d] abcdefgh -> %s (%d 个 token)   %s\n",total,out.c_str(),produced,
               ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::string out;
        const int produced=loaded?pipeline::generate_text(m.tokenizer,"abc",m.refs,5,0.0f,out)
            :-1;
        const bool ok=loaded&&produced==5&&out=="defgh";
        printf("[ 7/%d] max_new=5 就正好生成 5 个 (得到 %s)      %s\n",total,out.c_str(),
               ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::string a,b;
        if(loaded){
            pipeline::generate_text(m.tokenizer,"xyz",m.refs,8,0.0f,a);
            pipeline::generate_text(m.tokenizer,"xyz",m.refs,8,0.0f,b);
        }
        const bool ok=loaded&&!a.empty()&&a==b;
        printf("[ 8/%d] 同一提示词两次结果一致 (得到 %s)        %s\n",total,a.c_str(),
               ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::string with_bos,without_bos;
        if(loaded){
            pipeline::generate_text(m.tokenizer,"abc",m.refs,4,0.0f,with_bos,true);
            pipeline::generate_text(m.tokenizer,"abc",m.refs,4,0.0f,without_bos,false);
        }
        const bool ok=loaded&&!with_bos.empty()&&with_bos==without_bos;
        printf("[ 9/%d] 加不加 BOS 都不影响结果                  %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    {
        std::string out;
        const std::string huge(200,'k');
        const int produced=loaded?pipeline::generate_text(m.tokenizer,huge,m.refs,16,0.0f,out)
            :-1;
        const bool ok=loaded&&produced==0&&out.empty();
        printf("[10/%d] 提示词 200 字符 > max_seq 64 -> 安全停下  %s\n",total,ok?"OK":"FAIL");
        if(ok) passed++;
    }

    m.free();
    printf("check_pipeline: %d/%d passed\n",passed,total);
    self_passed=passed;self_total=total;
    return passed==total;
}

#endif