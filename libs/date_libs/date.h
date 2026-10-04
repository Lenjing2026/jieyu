#ifndef DATE_LIBS_DATE_H
#define DATE_LIBS_DATE_H
#include<bits/stdc++.h>
#include<windows.h>
using namespace std;
using namespace std::filesystem;
struct __model{
    uint64_t h; 
    uint64_t l;
    uint8_t mode;
    path model_vector;
    path model_info;
    path model_tokenizer;
    HANDLE model_handle;
    LPVOID model_map;
} model;
#endif  // DATE_LIBS_DATE_H
