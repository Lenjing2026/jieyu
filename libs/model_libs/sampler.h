#ifndef SAMPLER_H
#define SAMPLER_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
// 贪心：选概率最大的 token
int argmax_sample(const float* l,int vb_s){
    int b=0;
    float be_v=l[0];
    for (int i=1;i<vb_s;i++)
        if (l[i]>be_v){
            be_v=l[i];
            b=i;
        }
    return b;
}
// top-k 采样
int top_k_sample(const float* l,int vb_s,
                        int k,float t,std::mt19937& rng) {
    // 1. 温度缩放 + 找 top-k
    std::vector<std::pair<float,int>> c(vb_s);
    for (int i=0;i<vb_s;i++)
        c[i]={l[i]/t,i};
    k = std::min(k, vb_s);
    std::partial_sort(c.begin(),c.begin()+k,
                      c.end(),
                      [](const auto& a,const auto& b){
                          return a.first>b.first;
                      });
    // 2. softmax
    float max_val=c[0].first;
    float sum=0.0f;
    std::vector<float> probs(k);
    for (int i=0;i<k;i++) {
        probs[i]=expf(c[i].first-max_val);
        sum+=probs[i];
    }
    for (int i=0;i<k;i++) probs[i]/=sum;
    // 3. 按概率采样
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    float r=d(rng);
    float acc=0.0f;
    for(int i=0;i<k;i++) {
        acc+=probs[i];
        if(r<acc) return c[i].second;
    }
    return c[k-1].second;
}
#endif