#ifndef creater_model_h
#define creater_model_h
#include<filesystem>
#include<string>
#include<fstream>
#include<iostream>
#include"bfile.h"
#include"../libs.h"
#include"../mapp_libs/Windows_InMapp.h"
using namespace std;
using namespace std::filesystem;
using namespace json_libs;
bool create_model(path temp_path){
    return create_directories(temp_path);
}
void create_bfile(path temp_path) {
    path model_info_json = temp_path / "model_info.json";
    if(!exists(model_info_json)) throw "NO_JSONFILE";
    json_lib jtemp;
    jtemp.SetJsonWay(model_info_json.string());
    if(jtemp.GetJsonInt("layer_count")%16!=0) throw "NO16_LAYER";
    bfile::Header head;
    head.hidden=jtemp.GetJsonInt("hidden_size");
    head.layers=jtemp.GetJsonInt("layer_count");
    head.mode=jtemp.GetJsonInt("mode");
    head.num_heads=jtemp.GetJsonInt("num_heads");
    head.num_kv_heads=jtemp.GetJsonInt("num_kv_heads");
    head.head_dim=jtemp.GetJsonInt("head_dim");
    head.intermediate=jtemp.GetJsonInt("intermediate_size");
    head.vocab_size=jtemp.GetJsonInt("vocab_size");
    bfile::write_bfile((temp_path / "model.bf").string(),head,{});
}
#endif  // creater_model_h