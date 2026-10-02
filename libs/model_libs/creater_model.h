#include<filesystem>
#include<string>
#include<fstream>
#include<iostream>
#include"libs.h"
using namespace std;
using namespace std::filesystem;
using namespace json_libs;
bool create_modle(path temp_path) {
    return create_directories(temp_path);
}
void create_bfile(path temp_path) {
    std::ofstream file(temp_path.string() + "/model.bf", std::ios::binary);
    path modle_info_json = temp_path / "model_info.json";
    if(!exists(modle_info_json)) throw "NO_JSONFILE";
    json_lib jtemp;
    jtemp.SetJsonWay(modle_info_json.string());
    file.write("JYAIBF", 6);
    long long h=jtemp.GetJsonInt("hidden_size");
    long long l=jtemp.GetJsonInt("layer_count");
    file.write(reinterpret_cast<const char*>(&h), sizeof(h));
    file.write(reinterpret_cast<const char*>(&l), sizeof(l));
    file.close();
}