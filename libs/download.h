#include "libs.h"
#include "date_libs/consten_pool.h"
#include <cstdlib>
#include <string>
using namespace std;
using namespace std::filesystem;
using namespace json_libs;
bool download_json(){
    const string url=consten_pool.downloand_mj;
    if(system(("curl -L -f --retry 2 -o download.json \""+url+"\"").c_str())==0) return true;
    if(system(("wget -q -O download.json \""+url+"\"").c_str())==0) return true;
#if defined(_WIN32)
    return system(("powershell -NoProfile -Command \"$ProgressPreference='SilentlyContinue';Invoke-WebRequest -UseBasicParsing -Uri '"+url+"' -OutFile 'download.json'\"").c_str())==0;
#else
    return false;
#endif
}
bool download(string name){
    json_lib json;
    if(!json.SetJsonWay("download.json")&&!(download_json()&&json.SetJsonWay("download.json"))) return false;
    string body=json.GetJsonData(name);
    if(body.empty()&&download_json()&&json.SetJsonWay("download.json")) body=json.GetJsonData(name);
    if(body.empty()) return false;
    json_detail::Value sub;
    if(!json_detail::Parser(body).parse(sub)) return false;
    string begin;
    for(const auto& m:sub.members) if(m.first=="begin") begin=m.second.text;
    if(begin.empty()) return false;
    const path dir=path("models")/name;
    create_directories(dir);
    auto fetch=[&](const string& url,const string& out){
        if(system(("curl -L -f --retry 2 -o \""+out+"\" \""+url+"\"").c_str())==0) return true;
        if(system(("wget -q -O \""+out+"\" \""+url+"\"").c_str())==0) return true;
#if defined(_WIN32)
        return 
            system(
                ("powershell -NoProfile -Command \"$ProgressPreference='SilentlyContinue';Invoke-WebRequest -UseBasicParsing -Uri '"+url+"' -OutFile '"+out+"'\"").c_str()
            )==0;
#else
        return false;
#endif
    };
    for(const auto& m:sub.members){
        if(m.first=="begin") continue;
        if(!fetch(begin+m.second.text,(dir/m.second.text).string())) return false;
    }
    return true;
}