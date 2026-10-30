#ifndef LINLUX_INMAPP_H
#define LINLUX_INMAPP_H
#if !defined(_WIN32)
#include<fcntl.h>
#include<sys/mman.h>
#include<sys/stat.h>
#include<unistd.h>
#include<filesystem>
#include"../date_libs/date.h"
#include"../model_libs/bfile.h"
using namespace std;
using namespace std::filesystem;
inline void load_model_map(const path& dir){
    const string p=(dir/"model.bf").string();
    const int fd=open(p.c_str(),O_RDONLY);
    if(fd<0) throw runtime_error("open_Error");
    struct stat st={};
    if(fstat(fd,&st)!=0){
        close(fd);
        throw runtime_error("fstat_Error");
    }
    void* base=mmap(nullptr,(::size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){
        close(fd);
        throw runtime_error("mmap_Error");
    }
    model.file_size=(uint64_t)st.st_size;
    model.model_handle=fd;
    model.model_map=base;
}
inline void free_model_map(){
    bfile::current().reset();
    if(model.model_map){
        munmap(model.model_map,(::size_t)model.file_size);
        model.model_map=nullptr;
    }
    if(model.model_handle>=0){
        close(model.model_handle);
        model.model_handle=-1;
    }
}
#endif // !defined(_WIN32)
#endif
