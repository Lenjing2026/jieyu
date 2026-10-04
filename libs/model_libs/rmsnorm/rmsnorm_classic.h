
#include <cmath>
void rmsnorm_classic(float* x,float* y,const float* weight,int n,float eps){
    float ss=0.0f;
    for(int i=0;i<n;i++)
        ss+=x[i]*x[i];
    float den=ss/n+eps;
    if(den<=0.0f){
        for(int i=0;i<n;i++)
            y[i]=0.0f;
        return;
    }
    float scale=1.0f/sqrtf(den);
    for(int i=0;i<n;i++)
        y[i]=x[i]*scale*weight[i];
}
