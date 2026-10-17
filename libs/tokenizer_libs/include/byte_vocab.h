#ifndef JSON_LIBS_BYTE_VOCAB_H
#define JSON_LIBS_BYTE_VOCAB_H


#include<array>
#include<fstream>
#include<stdexcept>
#include<string>
#include"tokenizer.h"

namespace json_libs {

inline constexpr size_t kByteVocabSize=256;

inline void write_byte_level_vocab(const std::string& vocab_path,const std::string& merges_path){
    const std::array<std::string,256>& table=detail::byte_level_table();
    std::ofstream vocab(vocab_path,std::ios::binary | std::ios::trunc);
    if(!vocab) throw std::runtime_error("byte_vocab: 写不了 " + vocab_path);
    vocab << "{";
    for(size_t b=0; b < table.size(); b++){
        if(b!=0) vocab << ",";
        vocab << "\"";
        for(char c:table[b]){
            if(c=='"'||c=='\\') vocab << "\\";
            vocab << c;
        }
        vocab << "\":" << b;
    }
    vocab << "}";
    vocab.flush();
    if(!vocab) throw std::runtime_error("byte_vocab: 写入失败 " + vocab_path);

    std::ofstream merges(merges_path,std::ios::binary | std::ios::trunc);
    if(!merges) throw std::runtime_error("byte_vocab: 写不了 " + merges_path);
    merges << "#version: 0.2\n";
    merges.flush();
    if(!merges) throw std::runtime_error("byte_vocab: 写入失败 " + merges_path);
}

}

#endif
