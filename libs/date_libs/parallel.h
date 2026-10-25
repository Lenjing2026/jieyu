#ifndef JIEYU_PARALLEL_H
#define JIEYU_PARALLEL_H


#include<cstdint>
#include<functional>
#include<mutex>
#include<condition_variable>
#include<thread>
#include<vector>

#if defined(_OPENMP)
#include<omp.h>
#endif

inline int par_threads(){
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    const unsigned n=std::thread::hardware_concurrency();
    return (n==0)?1:(int)n;
#endif
}

#if !defined(_OPENMP)
class par_pool {
public:
    explicit par_pool(int threads){
        for(int i=1;i<threads;i++) workers_.emplace_back([this]{ worker(); });
    }
    ~par_pool(){
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_=true;
            ++gen_;
        }
        cv_.notify_all();
        for(std::thread& t:workers_){
            if(t.joinable()) t.join();
        }
    }
    par_pool(const par_pool&)=delete;
    par_pool& operator=(const par_pool&)=delete;

    void run(int chunks,const std::function<void(int)>& job){
        if(chunks<=1||workers_.empty()){
            job(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            job_=&job;
            chunks_=chunks;
            next_=1;
            pending_=chunks;
            ++gen_;
        }
        cv_.notify_all();
        job(0);
        std::unique_lock<std::mutex> lk(m_);
        if(--pending_==0) cv_done_.notify_all();
        cv_done_.wait(lk,[this]{ return pending_==0; });
        job_=nullptr;
    }

private:
    void worker(){
        unsigned seen=0;
        for(;;){
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk,[this,&seen]{ return stop_||gen_!=seen; });
                if(stop_) return;
                seen=gen_;
            }
            for(;;){
                int idx=-1;
                const std::function<void(int)>* job=nullptr;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if(next_>=chunks_) break;
                    idx=next_++;
                    job=job_;
                }
                (*job)(idx);
                std::lock_guard<std::mutex> lk(m_);
                if(--pending_==0) cv_done_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_,cv_done_;
    const std::function<void(int)>* job_=nullptr;
    int chunks_=0,next_=0,pending_=0;
    unsigned gen_=0;
    bool stop_=false;
};
#endif

template<class F>
inline void par_for(int64_t lo,int64_t hi,bool on,F body){
    const int64_t total=hi-lo;
    if(total<=0) return;
    int n=on?par_threads():1;
    if((int64_t)n>total) n=(int)total;
    if(n<=1){
        body(lo,hi);
        return;
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n)
    for(int t=0;t<n;t++){
        const int64_t a=lo+total*t/n;
        const int64_t b=lo+total*(t+1)/n;
        if(a<b) body(a,b);
    }
#else
    static par_pool pool(par_threads());
    pool.run(n,[&body,lo,total,n](int t){
        const int64_t a=lo+total*t/n;
        const int64_t b=lo+total*(t+1)/n;
        if(a<b) body(a,b);
    });
#endif
}

#endif
