#include<bits/stdc++.h>
#include"libs/libs.h"
#include"libs/text/text.h"

#ifdef _WIN32
// bits/stdc++.h 在 MinGW 下已经定义过这两个宏，这里加保护避免重定义警告
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
    // 控制台默认代码页是 936(GBK)，而本程序输出的是 UTF-8 字节，中文会显示成乱码；
    // 这里把控制台输出代码页也切到 UTF-8，cmd / PowerShell 里就正常了
    SetConsoleOutputCP(CP_UTF8);
#endif
    create_config_file();
    setup_way();
    cout << "初始化完成：已生成 config.json 与 models 目录" << endl;
    if (check_matmul()) {
        cout << "所有测试通过！" << endl;
    } else {
        cout << "部分测试失败！" << endl;
    }
    return 0;
}