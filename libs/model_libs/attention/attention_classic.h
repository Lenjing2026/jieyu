#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
using namespace std;
void attention_classic(float* Q,float* K,float* V,float* out,int seq,int num_heads,int num_kv_heads,int head_dim){
    if(num_kv_heads<=0||num_heads%num_kv_heads!=0)
        throw std::runtime_error("attention: num_heads 必须能被 num_kv_heads 整除");
    int group = num_heads / num_kv_heads;
    float scale=1.0f/sqrtf((float)head_dim);
    std::vector<float> scores(seq);
    for (int h=0;h<num_heads;h++) {
        int kv_h=h/group;
        for (int i=0;i<seq;i++) {
            float* q=Q+((size_t)i*num_heads+h)*head_dim;
            float mxs=-1e30f;
            for (int j=0;j<=i;j++) {
                float* k=K+((size_t)j*num_kv_heads+kv_h)*head_dim;
                float dt=0.0f;
                for (int d=0;d<head_dim;d++)
                    dt+=q[d]*k[d];
                dt*=scale;
                scores[j]=dt;
                if (dt>mxs) mxs=dt;
            }
            float sum = 0.0f;
            for (int j=0;j<=i;j++) {
                scores[j]=expf(scores[j]-mxs);
                sum+=scores[j];
            }
            float inv_sum=1.0f/sum;
            for (int j=0;j<=i;j++) scores[j]*=inv_sum;
            float* o=out+((size_t)i*num_heads + h) * head_dim;
            memset(o,0,head_dim*sizeof(float));
            for (int j=0;j<=i;j++) {
                const float* v=V+((size_t)j*num_kv_heads+kv_h)*head_dim;
                float p=scores[j];
                for (int d=0;d<head_dim;d++)
                    o[d]+=p*v[d];
            }
        }
    }
}