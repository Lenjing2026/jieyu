#ifndef MAIN_HPP
#define MAIN_HPP
#define JIEYU_HPP
#include<libs.h>
#include<date_libs/consten_pool.h>
#include<cstring>
#include<algorithm>
#include"JieyuAI_api.h"

/*
  进程内的模型数据（权重 / 分词器 / 超参）
  引擎只有一套全局 mmap 与词表，所以模型只能挂一份；句柄只记路径，
  多个句柄共享同一份，最后一个 free 掉时才卸载
*/
inline pipeline::LoadedModel& jieyu_model_data(){
    static pipeline::LoadedModel data;
    return data;
}

inline int& jieyu_handle_count(){
    static int n=0;
    return n;
}

inline bool& jieyu_model_ready(){
    static bool ready=false;
    return ready;
}

inline void jieyu_init_cpu(){
    static const bool once=(cpu_can.GetCpuCan(),true);
    (void)once;
}

inline void jieyu_set_text(char* dst,size_t cap,const char* src){
    if(dst==nullptr||cap==0) return;
    const size_t n=(src==nullptr)?0:strlen(src);
    const size_t k=n<cap-1?n:cap-1;
    if(k>0) memcpy(dst,src,k);
    dst[k]='\0';
}

inline void jieyu_set_error(jieyu_ctx* h,const char* msg){
    if(h==nullptr) return;
    jieyu_set_text(h->error,JIEYU_ERR_MAX,msg);
}

/*
  装载阶段的错误：失败时句柄会被 delete，没地方挂错误，
  所以单独留一个线程内的 char 缓冲（同样不用 std::string）
*/
inline char* jieyu_load_error(){
    static thread_local char text[JIEYU_ERR_MAX]={0};
    return text;
}

inline void jieyu_set_load_error(const std::string& msg){
    jieyu_set_text(jieyu_load_error(),JIEYU_ERR_MAX,msg.c_str());
}

/*
  自动上下文长度：读 model_info.json 的 max_position_embeddings，夹到 [64, 4096]
*/
inline int jieyu_auto_max_seq(const std::filesystem::path& dir){
    const int want=pipeline::read_info_int(dir/"model_info.json","max_position_embeddings",1024);
    return (std::max)(64,(std::min)(want,4096));
}

/*
  把生成结果拷进调用方缓冲区
    text  生成的纯文本
    out   目标缓冲区；给 nullptr 就是只问长度（先问长度再分配）
    cap   out 的字节数（含结尾 '\0'）
  返回：text 的总字节数（不含 '\0'），可能大于 cap-1（说明被截断）
*/
inline int jieyu_copy_out(const std::string& text,char* out,int cap){
    if(out==nullptr||cap<=0) return (int)text.size();
    const size_t room=(size_t)cap-1;
    const size_t n=text.size()<room?text.size():room;
    memcpy(out,text.data(),n);
    out[n]='\0';
    return (int)text.size();
}

extern "C" {

/*
  装载模型：dir 是模型目录，max_seq 给 0 就按 model_info.json 自动算
*/
JIEYU_API jieyu_model jieyu_load(const char* dir,int max_seq){
    if(dir==nullptr){
        jieyu_set_load_error("dir 为空");
        return nullptr;
    }
    jieyu_init_cpu();
    const std::string dir_utf8=pipeline::acp_to_utf8(dir);
    const std::filesystem::path model_dir=dir_utf8;
    if(jieyu_model_ready()&&jieyu_model_data().dir!=model_dir.string()){
        jieyu_model_data().free();
        free_model_map();
        jieyu_model_ready()=false;
    }
    jieyu_ctx* ctx=new(std::nothrow) jieyu_ctx();
    if(ctx==nullptr){
        jieyu_set_load_error("内存不足");
        return nullptr;
    }
    jieyu_set_text(ctx->dir,JIEYU_PATH_MAX,dir_utf8.c_str());
    ctx->max_seq=max_seq>0?max_seq:jieyu_auto_max_seq(model_dir);
    if(!jieyu_model_ready()){
        std::string err;
        if(!pipeline::load_model_dir(jieyu_model_data(),model_dir.string(),ctx->max_seq,&err)){
            jieyu_set_load_error(err);
            delete ctx;
            return nullptr;
        }
        jieyu_model_ready()=true;
    }
    jieyu_handle_count()++;
    jieyu_set_load_error("");
    return ctx;
}

JIEYU_API void jieyu_free(jieyu_model h){
    if(h==nullptr) return;
    if(jieyu_handle_count()>0&&--jieyu_handle_count()<=0){
        jieyu_model_data().free();
        free_model_map();
        jieyu_model_ready()=false;
    }
    delete h;
}

/*
  最后一次错误
    h 给句柄 ：返回这个句柄上一次的错误（空字符串表示没出错）
    h 给 nullptr：返回装载阶段的错误（jieyu_load 失败的原因）
*/
JIEYU_API const char* jieyu_last_error(jieyu_model h){
    return h!=nullptr?h->error:jieyu_load_error();
}

JIEYU_API const char* jieyu_version(){
    return consten_pool.version.charv.c_str();
}
JIEYU_API int jieyu_context_length(jieyu_model h){
    return h==nullptr?0:h->max_seq;
}

JIEYU_API const char* jieyu_model_dir(jieyu_model h){
    return h!=nullptr?h->dir:"";
}
/*
  纯文本生成：out 给 nullptr 时只问所需长度，返回新生成的 token 数
*/
JIEYU_API int jieyu_generate(jieyu_model h,const char* prompt,
                             char* out,int cap,
                             int max_new,float temperature,float repetition_penalty){
    if(h==nullptr){
        jieyu_set_load_error("句柄为空");
        return -1;
    }
    if(prompt==nullptr){
        jieyu_set_error(h,"prompt 为空");
        return -1;
    }
    if(!jieyu_model_ready()){
        jieyu_set_error(h,"模型未装载（先 jieyu_load）");
        return -1;
    }
    jieyu_set_error(h,"");
    pipeline::LoadedModel& m=jieyu_model_data();
    m.refs.max_new=max_new;
    m.refs.repetition_penalty=repetition_penalty;
    std::string text;
    const int produced=pipeline::generate_text(m.tokenizer,prompt,m.refs,text,
                                               temperature,nullptr);
    if(produced<0){
        jieyu_set_error(h,"生成失败（提示词里有超出词表的 token）");
        return -1;
    }
    if(produced==0) jieyu_set_error(h,"没有产出（提示词为空或上下文已满）");
    jieyu_copy_out(text,out,cap);
    return produced;
}

/*
  jsonl 对话生成：逐行读 name/message 拼成提示词再生成一次
*/
JIEYU_API int jieyu_generate_jsonl(jieyu_model h,const char* jsonl,
                                   const char* qb,const char* qe,
                                   char* out,int cap,
                                   int max_new,float temperature,float repetition_penalty){
    if(h==nullptr){
        jieyu_set_load_error("句柄为空");
        return -1;
    }
    if(jsonl==nullptr){
        jieyu_set_error(h,"jsonl 路径为空");
        return -1;
    }
    if(!jieyu_model_ready()){
        jieyu_set_error(h,"模型未装载（先 jieyu_load）");
        return -1;
    }
    jieyu_set_error(h,"");
    pipeline::LoadedModel& m=jieyu_model_data();
    m.refs.max_new=max_new;
    m.refs.repetition_penalty=repetition_penalty;
    std::string text;
    const int produced=json_ochat_text(jsonl,m,text,
                                       qb==nullptr?"":qb,qe==nullptr?"":qe,
                                       temperature);
    if(produced==-2){
        jieyu_set_error(h,"jsonl_no_srch（文件里没有可用的 name/message）");
        return -1;
    }
    if(produced<0){
        jieyu_set_error(h,"生成失败（提示词里有超出词表的 token）");
        return -1;
    }
    if(produced==0) jieyu_set_error(h,"没有产出（提示词为空或上下文已满）");
    jieyu_copy_out(text,out,cap);
    return produced;
}

}

#endif
