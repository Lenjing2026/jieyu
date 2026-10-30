#ifndef INFO_H
#define INFO_H
#include<stdlib.h>
#include<string>
#include<cstdio>
#define EXIT_UNKNOW_ERROR 1
#define EXIT_KV_OOM_ERROR 2
#define EXIT_VCOBE_ERROR 3
void error_exit(string error_info="unknow_error",int exit_id=0,bool bexit=true){
    printf("[ERROR]%s %d",error_info.c_str(),exit_id);
    if(bexit) exit(exit_id);
}
#endif