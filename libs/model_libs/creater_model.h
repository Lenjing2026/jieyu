#include<filesystem>
#include<string>
#include<fstream>
#include<iostream>
#include"libs.h"
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
    file.write(reinterpret_cast<const char*>(&h), sizeof(h));
    file.write(reinterpret_cast<const char*>(&l), sizeof(l));
    file.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
    file.close();
}