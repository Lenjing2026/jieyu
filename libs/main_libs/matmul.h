void matmul(const int m,const int n,const int k,const float* A,const float* B,float* C) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++){
            float sum=0.0f;
            for (int t=0;t<k;t++) {
                sum+=A[i*k+t]*B[t*n+j];
            }
            C[i*n+j]=sum;
        }
    }
}