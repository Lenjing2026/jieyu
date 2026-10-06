#ifndef SETUP_H
#define SETUP_H
#include<filesystem>
#include<fstream>
#include"json_libs/json.h"
using namespace std;
using namespace std::filesystem;
using namespace json_libs;

string langrage_dir(){ return "langrage"; }
string config_path(){ return current_path().string()+"/config.json"; }

bool write_language_file(const string& name){
    create_directories(path(langrage_dir()));
    json_lib j;
    j.SetJsonWay(langrage_dir()+"/"+name);
    if(name=="zh_cn.json"){
        j.WriteJsonKey("model_writer","作者");
        j.WriteJsonKey("model_description","描述");
    }else if(name=="en_us.json"){
        j.WriteJsonKey("model_writer","Author");
        j.WriteJsonKey("model_description","Description");
    }else{
        return false;
    }
    return true;
}
bool chinese_json(){ return write_language_file("zh_cn.json"); }

bool create_config_file(){
    json_lib config;
    config.SetJsonWay(config_path());
    config.WriteJsonKey("langrage","zh_cn.json");
    return true;
}
bool setup_way(){
    create_directories(path("models"));
    return true;
}
bool setup_all(){
    create_config_file();
    setup_way();
    create_directories(path("_file"));
    write_language_file("zh_cn.json");
    write_language_file("en_us.json");
    return true;
}

string current_langrage(){
    json_lib c;
    if(!c.SetJsonWay(config_path())) return "zh_cn.json";
    const string v=c.GetJsonData("langrage");
    return v.empty()?string("zh_cn.json"):v;
}
bool set_langrage(const string& value){
    json_lib c;
    c.SetJsonWay(config_path());
    c.WriteJsonKey("langrage",value);
    return true;
}
string tr(const string& key,const string& fallback=""){
    json_lib j;
    if(j.SetJsonWay(langrage_dir()+"/"+current_langrage())){
        const string v=j.GetJsonData(key);
        if(!v.empty()) return v;
    }
    return fallback.empty()?key:fallback;
}
#endif  // SETUP_H
