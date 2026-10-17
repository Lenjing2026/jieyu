#include<string>
#include<filesystem>
using namespace std;
using namespace std::filesystem;
struct __consten_pool{
    struct version{
        string charv;
        int a,b,c;
    } version;
    string downloand_mj;
    path cp;
};
const __consten_pool consten_pool={
    .version={"0.2.3",0,2,3},
    .downloand_mj="https://github.com/Lenjing2026/Jieyu_modle_downlond/releases/download/download_json/download.json",
    .cp=filesystem::current_path()
};