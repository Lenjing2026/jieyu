#ifndef JSON_LIBS_JSON_VALUE_H
#define JSON_LIBS_JSON_VALUE_H


#include<cstddef>
#include<cstdint>
#include<cstdio>
#include<string>
#include<utility>
#include<vector>

namespace json_libs {
namespace json_detail {

struct Value {
    enum class Type { Null,Bool,Number,String,Array,Object };

    Type type=Type::Null;
    std::string text;
    std::string raw;
    std::vector<Value> items;
    std::vector<std::pair<std::string,Value>> members;
};

inline void escape_string(std::string& out,const std::string& text){
    out.push_back('"');
    for(const unsigned char c:text){
        switch(c){
            case '"':out+="\\\""; break;
            case '\\':out+="\\\\"; break;
            case '\b':out+="\\b"; break;
            case '\f':out+="\\f"; break;
            case '\n':out+="\\n"; break;
            case '\r':out+="\\r"; break;
            case '\t':out+="\\t"; break;
            default:
                if(c < 0x20u){
                    char buffer[8];
                    std::snprintf(buffer,sizeof(buffer),"\\u%04x",static_cast<unsigned int>(c));
                    out+=buffer;
                }else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

inline void write_indent(std::string& out,int level){
    out.append(static_cast<size_t>(level) * 2,' ');
}

inline void write_value(std::string& out,const Value& value,int level){
    switch(value.type){
        case Value::Type::Null:
            out+="null";
            return;
        case Value::Type::Bool:
        case Value::Type::Number:
            out+=value.raw.empty()?std::string("null"):value.raw;
            return;
        case Value::Type::String:
            escape_string(out,value.text);
            return;
        case Value::Type::Array:{
            if(value.items.empty()){
                out+="[]";
                return;
            }
            out+="[\n";
            for(size_t i=0; i < value.items.size(); ++i){
                write_indent(out,level + 1);
                write_value(out,value.items[i],level + 1);
                if(i + 1 < value.items.size()){
                    out.push_back(',');
                }
                out.push_back('\n');
            }
            write_indent(out,level);
            out.push_back(']');
            return;
        }
        case Value::Type::Object:{
            if(value.members.empty()){
                out+="{}";
                return;
            }
            out+="{\n";
            for(size_t i=0; i < value.members.size(); ++i){
                write_indent(out,level + 1);
                escape_string(out,value.members[i].first);
                out+=": ";
                write_value(out,value.members[i].second,level + 1);
                if(i + 1 < value.members.size()){
                    out.push_back(',');
                }
                out.push_back('\n');
            }
            write_indent(out,level);
            out.push_back('}');
            return;
        }
    }
}

inline bool is_json_number(const std::string& text){
    if(text.empty()){
        return false;
    }
    size_t i=0;
    if(text[i]=='-'){
        i+=1;
    }
    if(i>=text.size()){
        return false;
    }
    if(text[i]=='0'){
        i+=1;
        if(i < text.size()&&text[i]>='0'&&text[i]<='9'){
            return false;
        }
    }else if(text[i]>='1'&&text[i]<='9'){
        while(i < text.size()&&text[i]>='0'&&text[i]<='9'){
            i+=1;
        }
    }else {
        return false;
    }
    if(i < text.size()&&text[i]=='.'){
        i+=1;
        const size_t digits=i;
        while(i < text.size()&&text[i]>='0'&&text[i]<='9'){
            i+=1;
        }
        if(i==digits){
            return false;
        }
    }
    if(i < text.size()&&(text[i]=='e'||text[i]=='E')){
        i+=1;
        if(i < text.size()&&(text[i]=='+'||text[i]=='-')){
            i+=1;
        }
        const size_t digits=i;
        while(i < text.size()&&text[i]>='0'&&text[i]<='9'){
            i+=1;
        }
        if(i==digits){
            return false;
        }
    }
    return i==text.size();
}

}
}

#endif
