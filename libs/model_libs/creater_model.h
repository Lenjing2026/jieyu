#include<filesystem>
#include<string>
using namespace std;
using namespace std::filesystem;
bool create_modle(char modle_name[]){
    path temp_path="models/" + string(modle_name);
    return create_directories(temp_path);
}