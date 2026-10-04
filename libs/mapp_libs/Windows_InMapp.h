#ifndef winmpp_h
#define winmpp_h
#include<windows.h>
#include<filesystem>
#include"date.h"
using namespace std;
using namespace std::filesystem;
inline void load_model_map(const path& dir) {
    path file_path=dir / "model.bf";
    HANDLE hFile=CreateFileA(
        file_path.string().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        throw runtime_error("CreateFile_Error");
    }
    HANDLE hMap = CreateFileMappingA(
        hFile,NULL,
        PAGE_READONLY,
        0,0,
        NULL
    );
    if (hMap == NULL) {
        CloseHandle(hFile);
        throw runtime_error("CreateFileMapping_Error");
    }
    LPVOID base=MapViewOfFile(hMap,FILE_MAP_READ,0,0,0);
    if (base == NULL){
        CloseHandle(hMap);
        CloseHandle(hFile);
        throw runtime_error("MapViewOfFile_Error");
    }
    CloseHandle(hFile);
    model.model_handle=hMap;
    model.model_map=base;
}
inline void free_model_map(){
    if (model.model_map){
        UnmapViewOfFile(model.model_map);
        model.model_map=nullptr;
    }
    if (model.model_handle){
        CloseHandle(model.model_handle);
        model.model_handle=NULL;
    }
}
#endif  // WINMPP_H
