#ifndef JSON_LIBS_JSON_H
#define JSON_LIBS_JSON_H


#include<cstddef>
#include<functional>
#include<string>
#include<vector>

#include"file_util.h"
#include"json_parser.h"
#include"json_value.h"

namespace json_libs {

class json_lib {
public:
    json_lib():loaded_(false){}

    bool SetJsonWay(const std::string& path){
        path_=path;
        return Reload();
    }

    bool Reload(){
        loaded_=false;
        root_=json_detail::Value();
        error_.clear();
        if(path_.empty()){
            error_="还没有设置路径，请先调用 SetJsonWay(路径)";
            return false;
        }
        const std::string resolved=file_util::resolve_relative_path(path_);
        std::string content;
        if(!file_util::read_file(resolved,content)){
            error_="无法读取文件: " + resolved;
            return false;
        }
        json_detail::Parser parser(content);
        if(!parser.parse(root_)){
            error_="解析失败: " + parser.error();
            root_=json_detail::Value();
            return false;
        }
        loaded_=true;
        return true;
    }

    const std::string& GetJsonWay() const {
        return path_;
    }

    bool IsLoaded() const {
        return loaded_;
    }

    std::string GetJsonData(const std::string& key) const {
        const json_detail::Value* value=find(key);
        if(value==nullptr){
            return std::string();
        }
        return value_to_string(*value);
    }

    std::string GetJsonData(const std::string& key,const std::string& default_value) const {
        const json_detail::Value* value=find(key);
        if(value==nullptr){
            return default_value;
        }
        return value_to_string(*value);
    }

    int GetJsonInt(const std::string& key,int default_value=0) const {
        const std::string text=GetJsonData(key);
        if(text.empty()){
            return default_value;
        }
        try {
            return std::stoi(text);
        } catch(...){
            return default_value;
        }
    }

    double GetJsonDouble(const std::string& key,double default_value=0.0) const {
        const std::string text=GetJsonData(key);
        if(text.empty()){
            return default_value;
        }
        try {
            return std::stod(text);
        } catch(...){
            return default_value;
        }
    }

    bool GetJsonBool(const std::string& key,bool default_value=false) const {
        const std::string text=GetJsonData(key);
        if(text=="true"||text=="1"){
            return true;
        }
        if(text=="false"||text=="0"){
            return false;
        }
        return default_value;
    }

    bool HasKey(const std::string& key) const {
        return find(key)!=nullptr;
    }

    size_t KeyCount() const {
        if(!loaded_||root_.type!=json_detail::Value::Type::Object){
            return 0;
        }
        return root_.members.size();
    }

    std::vector<std::string> Keys() const {
        std::vector<std::string> keys;
        if(loaded_&&root_.type==json_detail::Value::Type::Object){
            keys.reserve(root_.members.size());
            for(const auto& member:root_.members){
                keys.push_back(member.first);
            }
        }
        return keys;
    }

    bool ForEachKey(const std::function<bool(const std::string& key,const std::string& value)>& visit) const {
        if(!loaded_||root_.type!=json_detail::Value::Type::Object){
            return false;
        }
        for(const auto& member:root_.members){
            if(!visit(member.first,value_to_string(member.second))){
                return false;
            }
        }
        return true;
    }

    const std::string& LastError() const {
        return error_;
    }

    bool WriteJsonKey(const std::string& key,const std::string& value){
        if(key.empty()){
            error_="键不能为空";
            return false;
        }
        if(path_.empty()){
            error_="还没有设置路径，请先调用 SetJsonWay(路径)";
            return false;
        }
        if(!ensure_writable()){
            return false;
        }
        const json_detail::Value new_value=json_detail::make_value(value);
        if(!set_member(root_,key,new_value)){
            return false;
        }
        return SaveJsonFile();
    }

    bool SaveJsonFile(){
        if(!loaded_){
            error_="还没有可保存的内容";
            return false;
        }
        const std::string resolved=file_util::resolve_relative_path(path_);
        std::string text;
        json_detail::write_value(text,root_,0);
        text.push_back('\n');
        if(!file_util::write_file(resolved,text)){
            error_="写入失败: " + resolved;
            return false;
        }
        error_.clear();
        return true;
    }

    std::string ToJsonText() const {
        if(!loaded_){
            return std::string();
        }
        std::string text;
        json_detail::write_value(text,root_,0);
        text.push_back('\n');
        return text;
    }

private:
    static std::string value_to_string(const json_detail::Value& value){
        if(value.type==json_detail::Value::Type::String){
            return value.text;
        }
        if(!value.raw.empty()){
            return value.raw;
        }
        std::string text;
        json_detail::write_value(text,value,0);
        return text;
    }

    const json_detail::Value* find(const std::string& key) const {
        if(!loaded_||root_.type!=json_detail::Value::Type::Object){
            return nullptr;
        }
        for(const auto& member:root_.members){
            if(member.first==key){
                return &member.second;
            }
        }
        if(key.find('.')==std::string::npos){
            return nullptr;
        }
        const json_detail::Value* current=&root_;
        size_t start=0;
        while(true){
            const size_t dot=key.find('.',start);
            const std::string part=(dot==std::string::npos)
                ?key.substr(start)
                :key.substr(start,dot - start);
            if(part.empty()||current->type!=json_detail::Value::Type::Object){
                return nullptr;
            }
            const json_detail::Value* next=nullptr;
            for(const auto& member:current->members){
                if(member.first==part){
                    next=&member.second;
                    break;
                }
            }
            if(next==nullptr){
                return nullptr;
            }
            current=next;
            if(dot==std::string::npos){
                break;
            }
            start=dot + 1;
        }
        return current;
    }

    bool ensure_writable(){
        if(loaded_){
            return true;
        }
        const std::string resolved=file_util::resolve_relative_path(path_);
        std::string content;
        if(file_util::read_file(resolved,content)
            &&content.find_first_not_of(" \t\r\n")!=std::string::npos){
            if(error_.empty()){
                error_="文件已存在但解析失败，为避免覆盖已拒绝写入: " + resolved;
            }
            return false;
        }
        root_=json_detail::Value();
        root_.type=json_detail::Value::Type::Object;
        loaded_=true;
        error_.clear();
        return true;
    }

    bool set_member(json_detail::Value& root,const std::string& key,const json_detail::Value& value){
        for(auto& member:root.members){
            if(member.first==key){
                member.second=value;
                return true;
            }
        }
        if(key.find('.')==std::string::npos){
            root.members.emplace_back(key,value);
            return true;
        }
        json_detail::Value* current=&root;
        size_t start=0;
        while(true){
            const size_t dot=key.find('.',start);
            const std::string part=(dot==std::string::npos)
                ?key.substr(start)
                :key.substr(start,dot - start);
            if(part.empty()){
                error_="键路径中有空的层级: " + key;
                return false;
            }
            const bool is_last=(dot==std::string::npos);
            json_detail::Value* next=nullptr;
            for(auto& member:current->members){
                if(member.first==part){
                    next=&member.second;
                    break;
                }
            }
            if(is_last){
                if(next!=nullptr){
                    *next=value;
                }else {
                    current->members.emplace_back(part,value);
                }
                return true;
            }
            if(next==nullptr){
                json_detail::Value created;
                created.type=json_detail::Value::Type::Object;
                current->members.emplace_back(part,created);
                next=&current->members.back().second;
            }else if(next->type!=json_detail::Value::Type::Object){
                error_="键路径中的 \"" + part + "\" 不是对象，无法写入";
                return false;
            }
            current=next;
            start=dot + 1;
        }
    }

    std::string path_;
    std::string error_;
    json_detail::Value root_;
    bool loaded_;
};

}

#endif