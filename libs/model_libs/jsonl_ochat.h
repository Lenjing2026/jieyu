#ifndef JIEYU_JSONL_OCHAT_H
#define JIEYU_JSONL_OCHAT_H
#include<libs.h>
string jsonl_oc_feed="";
bool json_ochat(const path jsonl_p,const string qb,const string qe,pipeline::LoadedModel& m,
                int max_new=256,float temperature=0.8f,float repetition_penalty=1.1f,
                FILE* echo=stdout,bool show_speed=false){
    jsonl_oc_feed.clear();
    json_libs::json_file temp;
    temp.SetJsonWay(jsonl_p.string());
    json_libs::jsonl_lib joc_jsonl(temp);
    const int jl=jsonl_len(temp);
    for(int i=0;i<jl;i++){
        const string tn=joc_jsonl.jsonl_sget(i,"name"),tm=joc_jsonl.jsonl_sget(i,"message");
        if(tn.empty()&&tm.empty()) continue;
        jsonl_oc_feed+=qb+tn+'\n';
        jsonl_oc_feed+=tm+qe+'\n';
    }
    if(jsonl_oc_feed.empty()) return false;
    if(!m.s_asst.empty()) jsonl_oc_feed+=m.s_asst;
    else if(m.s_start=="<|im_start|>") jsonl_oc_feed+=m.s_start+"assistant\n";
    string out;
    const int produced=pipeline::generate_text(m.tokenizer,jsonl_oc_feed,m.refs,max_new,temperature,
                                               out,m.refs.add_bos,echo,repetition_penalty,show_speed);
    return produced>0;
}
#endif