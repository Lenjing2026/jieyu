#include<filesystem>
#include<string>
#include<fstream>
#include<iostream>
#include"libs.h"
#include"../mapp_libs/Windows_InMapp.h"
using namespace std;
using namespace std::filesystem;
using namespace json_libs;
bool create_model(path temp_path){
    return create_directories(temp_path);
}
void create_bfile(path temp_path) {
    std::ofstream file(temp_path.string() + "/model.bf", std::ios::binary);
    path model_info_json = temp_path / "model_info.json";
    if(!exists(model_info_json)) throw "NO_JSONFILE";
    json_lib jtemp;
    jtemp.SetJsonWay(model_info_json.string());
    if(jtemp.GetJsonInt("layer_count")%16!=0) throw "NO16_LAYER";
    file.write("JYAIBF", 6);
    uint64_t h=jtemp.GetJsonInt("hidden_size");
    uint64_t l=jtemp.GetJsonInt("layer_count");
    uint8_t mode=jtemp.GetJsonInt("mode");
    uint8_t num_heads=jtemp.GetJsonInt("num_heads");
    uint8_t num_kv_heads=jtemp.GetJsonInt("num_kv_heads");
    uint8_t head_dim=jtemp.GetJsonInt("head_dim");
    uint32_t intermediate_size=jtemp.GetJsonInt("intermediate_size");
    uint64_t vocab_size=jtemp.GetJsonInt("vocab_size");
    file.write(reinterpret_cast<const char*>(&h), sizeof(h));
    file.write(reinterpret_cast<const char*>(&l), sizeof(l));
    file.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    file.write(reinterpret_cast<const char*>(&num_kv_heads), sizeof(num_kv_heads));
    file.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));
    file.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    size_t pos=file.tellp();
    size_t pad=(64-pos%64)%64;
    for (size_t i=0;i<pad;i++) file.put('\0');

    file.close();
}