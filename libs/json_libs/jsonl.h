#ifndef JSON_LIBS_JSONL_H
#define JSON_LIBS_JSONL_H


#include<cstddef>
#include<string>

#include"file_util.h"
#include"json.h"
#include"json_parser.h"
#include"json_value.h"

namespace json_libs {

inline int jsonl_len(const json_file& json){
    std::string content;
    if(!file_util::read_file(file_util::resolve_relative_path(json.GetJsonWay()),content)){
        return 0;
    }
    int lines=0;
    size_t start=0;
    while(start<content.size()){
        lines++;
        const size_t end=content.find('\n',start);
        if(end==std::string::npos){
            break;
        }
        start=end + 1;
    }
    return lines;
}

class jsonl_lib {
public:
    explicit jsonl_lib(const json_file& json):json_(json){}

    std::string jsonl_sget(int line,const std::string& parameter) const {
        return jsonl_sget(line,parameter,std::string());
    }

    std::string jsonl_sget(int line,const std::string& parameter,const std::string& default_value) const {
        if(line<0){
            return default_value;
        }
        std::string content;
        if(!file_util::read_file(file_util::resolve_relative_path(json_.GetJsonWay()),content)){
            return default_value;
        }
        size_t start=0;
        for(int i=0;;i++){
            const size_t end=content.find('\n',start);
            const size_t stop=(end==std::string::npos)?content.size():end;
            if(i==line){
                json_detail::Value value;
                json_detail::Parser parser(content.substr(start,stop - start));
                if(!parser.parse(value)){
                    return default_value;
                }
                const json_detail::Value* hit=find_in(value,parameter);
                if(hit==nullptr){
                    return default_value;
                }
                return value_to_string(*hit);
            }
            if(end==std::string::npos){
                return default_value;
            }
            start=end + 1;
        }
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

    static const json_detail::Value* find_in(const json_detail::Value& root,const std::string& key){
        if(root.type!=json_detail::Value::Type::Object){
            return nullptr;
        }
        for(const auto& member:root.members){
            if(member.first==key){
                return &member.second;
            }
        }
        if(key.find('.')==std::string::npos){
            return nullptr;
        }
        const json_detail::Value* current=&root;
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

    const json_file& json_;
};

}

#endif
