#ifndef MAIN_HPP
#define MAIN_HPP
#include<libs.h>
#include<date_libs/consten_pool.h>
#include<cstring>
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

struct jieyu_ctx {
    pipeline::LoadedModel model;
    std::string error;
};

inline int jieyu_copy_out(const std::string& text,char* out,int cap){
    if(out==nullptr||cap<=0) return (int)text.size();
    const size_t room=(size_t)cap-1;
    const size_t n=text.size()<room?text.size():room;
    memcpy(out,text.data(),n);
    out[n]='\0';
    return (int)text.size();
}

extern "C" {

typedef struct jieyu_ctx* jieyu_model;

JIEYU_API jieyu_model jieyu_load(const char* dir,int max_seq){
    if(dir==nullptr) return nullptr;
    static bool cpu_ready=false;
    if(!cpu_ready){
        cpu_can.GetCpuCan();
        cpu_ready=true;
    }
    jieyu_ctx* ctx=new(std::nothrow) jieyu_ctx();
    if(ctx==nullptr) return nullptr;
    if(!pipeline::load_model_dir(ctx->model,dir,max_seq>0?max_seq:1024,&ctx->error)){
        delete ctx;
        return nullptr;
    }
    return ctx;
}

JIEYU_API void jieyu_free(jieyu_model h){
    if(h==nullptr) return;
    h->model.free();
    delete h;
}

JIEYU_API const char* jieyu_last_error(jieyu_model h){
    return (h==nullptr)?"句柄为空":h->error.c_str();
}

JIEYU_API const char* jieyu_version(){
    return consten_pool.version.charv.c_str();
}

JIEYU_API int jieyu_generate(jieyu_model h,const char* prompt,
                             char* out,int cap,
                             int max_new,float temperature,float repetition_penalty){
    if(h==nullptr||prompt==nullptr) return -1;
    std::string text;
    const int produced=pipeline::generate_text(h->model.tokenizer,prompt,h->model.refs,
                                               max_new,temperature,text,
                                               h->model.refs.add_bos,nullptr,
                                               repetition_penalty,false);
    if(produced<0){
        h->error="生成失败（词表与模型不匹配？）";
        return -1;
    }
    jieyu_copy_out(text,out,cap);
    return produced;
}

JIEYU_API int jieyu_generate_jsonl(jieyu_model h,const char* jsonl,
                                   const char* qb,const char* qe,
                                   char* out,int cap,
                                   int max_new,float temperature,float repetition_penalty){
    if(h==nullptr||jsonl==nullptr) return -1;
    json_libs::json_file way;
    if(!way.SetJsonWay(jsonl)){
        h->error=way.LastError();
        return -1;
    }
    json_libs::jsonl_lib lines(way);
    std::string feed;
    const int n=jsonl_len(way);
    for(int i=0;i<n;i++){
        const std::string tn=lines.jsonl_sget(i,"name");
        const std::string tm=lines.jsonl_sget(i,"message");
        if(tn.empty()&&tm.empty()) continue;
        feed+=(qb==nullptr?h->model.s_start:qb);
        feed+=tn;
        feed+='\n';
        feed+=tm;
        feed+=(qe==nullptr?h->model.s_end:qe);
        feed+='\n';
    }
    if(feed.empty()){
        h->error="jsonl 里没有可用记录";
        return -1;
    }
    if(!h->model.s_asst.empty()) feed+=h->model.s_asst;
    else if(h->model.s_start=="<|im_start|>") feed+=h->model.s_start+"assistant\n";
    std::string text;
    const int produced=pipeline::generate_text(h->model.tokenizer,feed,h->model.refs,
                                               max_new,temperature,text,
                                               h->model.refs.add_bos,nullptr,
                                               repetition_penalty,false);
    if(produced<0){
        h->error="生成失败（词表与模型不匹配？）";
        return -1;
    }
    jieyu_copy_out(text,out,cap);
    return produced;
}

}

#endif
