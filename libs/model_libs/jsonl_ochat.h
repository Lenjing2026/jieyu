#ifndef JIEYU_JSONL_OCHAT_H
#define JIEYU_JSONL_OCHAT_H
#include<libs.h>

string jsonl_oc_feed="";

inline std::string json_ochat_build(const path jsonl_p,pipeline::LoadedModel& m,
                                    const string qb="",const string qe=""){
    jsonl_oc_feed.clear();
    json_libs::json_file temp;
    temp.SetJsonWay(jsonl_p.string());
    json_libs::jsonl_lib joc_jsonl(temp);
    const int jl=jsonl_len(temp);
    const string pre=qb.empty()?m.s_start:qb;
    const string suf=qe.empty()?m.s_end:qe;
    for(int i=0;i<jl;i++){
        const string tn=joc_jsonl.jsonl_sget(i,"name"),tm=joc_jsonl.jsonl_sget(i,"message");
        if(tn.empty()&&tm.empty()) continue;
        jsonl_oc_feed+=pre+tn+'\n';
        jsonl_oc_feed+=tm+suf+'\n';
    }
    if(!jsonl_oc_feed.empty()){
        if(!m.s_asst.empty()) jsonl_oc_feed+=m.s_asst;
        else if(m.s_start=="<|im_start|>") jsonl_oc_feed+=m.s_start+"assistant\n";
    }
    return jsonl_oc_feed;
}

inline int json_ochat_text(const path jsonl_p,pipeline::LoadedModel& m,std::string& out,
                           const string qb="",const string qe="",float temperature=0.8f){
    out.clear();
    const std::string feed=json_ochat_build(jsonl_p,m,qb,qe);
    if(feed.empty()) return -2;
    return pipeline::generate_text(m.tokenizer,feed,m.refs,out,temperature);
}

inline bool json_ochat(const path jsonl_p,pipeline::LoadedModel& m,
                       const string qb="",const string qe="",
                       float temperature=0.8f,FILE* echo=stdout){
    const std::string feed=json_ochat_build(jsonl_p,m,qb,qe);
    if(feed.empty()) return false;
    std::string out;
    const int produced=pipeline::generate_text(m.tokenizer,feed,m.refs,out,temperature,echo);
    return produced>0;
}

#endif
