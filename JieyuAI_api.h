/*
 * ========================================================================
 * Copyright (c) 2026 Lenjing2026
 *
 * This file is part of JieYuAI.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * ========================================================================
*/
#ifndef JIEYUAI_API_H
#define JIEYUAI_API_H
#ifndef JIEYU_API
#ifdef _WIN32
    #ifdef MAIN_EXPORTS
        #define JIEYU_API __declspec(dllexport)
    #elif defined(JIEYU_USE_DLL)
        #define JIEYU_API __declspec(dllimport)
    #else
        #define JIEYU_API
    #endif
#else
    #define JIEYU_API __attribute__((visibility("default")))
#endif
#endif
#define JIEYU_PATH_MAX 512
#define JIEYU_ERR_MAX  256
/*
  句柄内部数据：只有 char 数据项，跨 DLL 边界不出现 std::string
    dir      模型目录（句柄靠它认是哪份模型）
    error    这个句柄上一次的错误（空串 = 没出错）
    max_seq  上下文长度上限
  句柄是不透明指针，这里只是为了 C 侧能给 jieyu_model 起名字
*/
struct jieyu_ctx {
    char dir[JIEYU_PATH_MAX]={0};
    char error[JIEYU_ERR_MAX]={0};
    int max_seq=0;
};
#ifdef __cplusplus
extern "C" {
    typedef struct jieyu_ctx* jieyu_model;
    JIEYU_API jieyu_model jieyu_load(const char* dir,int max_seq=0);
    JIEYU_API void jieyu_free(jieyu_model h);
    JIEYU_API const char* jieyu_last_error(jieyu_model h);
    JIEYU_API const char* jieyu_version();
    JIEYU_API int jieyu_context_length(jieyu_model h);
    JIEYU_API const char* jieyu_model_dir(jieyu_model h);
    JIEYU_API int jieyu_generate(jieyu_model h,const char* prompt,
                             char* out,int cap,
                             int max_new,
                             float temperature,
                             float repetition_penalty);
    JIEYU_API int jieyu_generate_jsonl(jieyu_model h,const char* jsonl,
                                   const char* qb,const char* qe,
                                   char* out,int cap,
                                   int max_new=256,
                                   float temperature=0.8f,
                                   float repetition_penalty=1.1f);
}
#include<cstddef>
#include<stdexcept>
#include<string>
#include<string_view>
#include<utility>
#include<vector>
class jieyu{
public:
    class error:public std::runtime_error{
    public:
        explicit error(const std::string& msg):std::runtime_error(msg){}
    };
    struct options{
        int max_new=256;
        float temperature=0.7f;
        float repetition_penalty=1.1f;
    };
    jieyu() noexcept=default;
    explicit jieyu(std::string_view dir,int max_seq=0){ open(dir,max_seq); }
    ~jieyu(){ close(); }
    jieyu(const jieyu&)=delete;
    jieyu& operator=(const jieyu&)=delete;
    jieyu(jieyu&& other) noexcept:h_(std::exchange(other.h_,nullptr)){}
    jieyu& operator=(jieyu&& other) noexcept{
        if(this!=&other){
            close();
            h_=std::exchange(other.h_,nullptr);
        }
        return *this;
    }
    void open(std::string_view dir,int max_seq=0){
        close();
        std::string d(dir);
        h_=jieyu_load(d.c_str(),max_seq);
        if(!h_) throw error(std::string("装载失败: ")+jieyu_last_error(nullptr));
    }
    void close() noexcept{
        if(h_){
            jieyu_free(h_);
            h_=nullptr;
        }
    }
    bool is_open() const noexcept{ return h_!=nullptr; }
    explicit operator bool() const noexcept{ return h_!=nullptr; }
    int context_length() const noexcept{ return h_?jieyu_context_length(h_):0; }
    std::string dir() const{ return h_?std::string(jieyu_model_dir(h_)):std::string(); }
    std::string last_error() const{ return std::string(jieyu_last_error(h_)); }
    std::string generate(std::string_view prompt){ return generate(prompt,options{}); }
    std::string generate(std::string_view prompt,const options& opt){
        check();
        std::string p(prompt);
        std::vector<char> buf((std::size_t)(opt.max_new>0?opt.max_new:4096)*16u+256u,0);
        if(jieyu_generate(h_,p.c_str(),buf.data(),(int)buf.size(),opt.max_new,opt.temperature,opt.repetition_penalty)<0)
            throw error(jieyu_last_error(h_));
        return std::string(buf.data());
    }
    std::string generate_jsonl(std::string_view jsonl,std::string_view qb={},std::string_view qe={}){
        return generate_jsonl(jsonl,qb,qe,options{});
    }
    std::string generate_jsonl(std::string_view jsonl,std::string_view qb,std::string_view qe,const options& opt){
        check();
        std::string j(jsonl),b(qb),e(qe);
        std::vector<char> buf((std::size_t)(opt.max_new>0?opt.max_new:4096)*16u+256u,0);
        if(jieyu_generate_jsonl(h_,j.c_str(),b.c_str(),e.c_str(),buf.data(),(int)buf.size(),
                                opt.max_new,opt.temperature,opt.repetition_penalty)<0)
            throw error(jieyu_last_error(h_));
        return std::string(buf.data());
    }
    static std::string version(){ return std::string(jieyu_version()); }
private:
    void check() const{
        if(!h_) throw error("模型未装载（先 open）");
    }
    jieyu_model h_=nullptr;
};
#endif
#endif
