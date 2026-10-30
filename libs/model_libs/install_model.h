#ifndef INSTALL_MODEL_H
#define INSTALL_MODEL_H
#include<cstdint>
#include<cstring>
#include<filesystem>
#include<fstream>
#include<stdexcept>
#include<string>
#include<vector>
#include"bfile.h"
#include"../json_libs/json.h"
#include"../mapp_libs/InMapp.h"
#include"../date_libs/date.h"
using namespace json_libs;
using namespace std;
using namespace std::filesystem;
void install_model(path model_path){
    model.set_model(model_path);
    char magic[7];
    magic[6]='\0';
    if(!exists(model.model_info))
        throw std::runtime_error("model_info.json not found");
    if(!exists(model.model_vector))
        throw std::runtime_error("model.bf not found");
    load_model_map(model.model_vector.parent_path());
    std::ifstream info_file(model.model_info);
    if(!info_file.is_open())
        throw std::runtime_error("model_info.json open failed");
    memcpy(magic,model.model_map,6);
    magic[6]='\0';
    if(strcmp(magic,"JYAIBF")!=0)
        throw std::runtime_error("model.bf magic not match");
    model.h=*reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 6);
    model.l=*reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 14);
    model.mode=*reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 22);
    model.num_heads=*reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 23);
    model.num_kv_heads=*reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 24);
    model.head_dim=*reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 25);
    model.intermediate_size=*reinterpret_cast<uint32_t*>(static_cast<unsigned char*>(model.model_map) + 26);
    model.vocab_size=*reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 30);
    info_file.close();
    bfile::current().reset();
    if(model.file_size>=bfile::kHeaderSize
        &&bfile::read_u32(static_cast<const unsigned char*>(model.model_map) + 38)>=bfile::kVersion){
        if(!bfile::current().open(model.model_map,static_cast<size_t>(model.file_size)))
            throw std::runtime_error("model.bf tensor table broken");
    }
}

#endif