#include<bits/stdc++.h>
#include"libs/libs.h"
#include"libs/text/text.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include<windows.h>
#endif
using namespace std;
using namespace json_libs;
int main(){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    create_config_file();
    setup_way();
    cout<<"稍等，我们正在获取你的cpu信息\n";
    cpu_can.GetCpuCan();
    OutCPUInfo(cpu_can);
    cout << "初始化完成：已生成 config.json 与 models 目录" << endl;
    if (check_matmul()) {
        cout << "所有测试通过！" << endl;
    } else {
        cout << "部分测试失败！" << endl;
    }
    return 0;
}