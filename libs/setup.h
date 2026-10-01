#include<filesystem>
#include<fstream>
#include"json_libs/json.h"
using namespace std;
using namespace std::filesystem;
using namespace json_libs;
bool create_config_file(){
    std::ofstream temp_file(current_path().string() + "/config.json");
    json_lib config;
    config.SetJsonWay(current_path().string() + "/config.json");
    config.WriteJsonKey("aaa","114514");
    config.WriteJsonKey("bbb","1919810");
    return true;
}
bool setup_way(){
    path temp_path="models";
    return create_directories(temp_path);
}
