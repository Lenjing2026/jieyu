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
namespace cli {

inline int read_json_int(const std::string& path,const std::string& key,int fallback){
    json_lib j;
    if(!j.SetJsonWay(path)) return fallback;
    return j.GetJsonInt(key,fallback);
}

inline void usage(){
    cout<<"用法：\n"
        <<"  jieyu                                跑全部自测\n"
        <<"  jieyu --demo [目录]                  造一个演示小模型：转换 + 文本进文本出\n"
        <<"  jieyu --convert <safetensors...> -o <目录> [--config config.json] [--tokenizer <目录>]\n"
        <<"  jieyu --show <model.bf>              打印 model.bf 的张量目录\n"
        <<"  jieyu --gen <模型目录> <提示词> [最多生成] [温度]\n"
        <<"  jieyu --chat <模型目录> [最多生成] [温度]\n";
}

inline int demo(string root){
    cout<<"==== 端到端演示 ====\n";
    cout<<"[1/3] 造演示模型（层权重全 0，lm_head 设计成输出下一个字节）\n";
    converter::make_demo_model(root+"/src");
    cout<<"[2/3] 转换 safetensors -> model.bf\n";
    converter::Options opt;
    opt.sources.push_back(root+"/src/model.safetensors");
    opt.out_dir=root+"/model";
    opt.config_path=root+"/src/config.json";
    opt.tokenizer_dir=root+"/src";
    converter::convert(opt);
    cout<<"[3/3] 文本进 -> 文本出\n";
    pipeline::LoadedModel m;
    string err;
    if(!pipeline::load_model_dir(m,root+"/model",1024,&err)){
        cout<<"装载失败: "<<err<<"\n";
        return 1;
    }
    const string prompt="abcdefgh";
    cout<<"\n提示词  "<<prompt<<"\n生成    "<<flush;
    pipeline::run_once(m,prompt,16,0.0f);
    cout<<"（演示模型是特意设计的，只会把字节一个一个往后加，不是真会说话）\n";
    m.free();
    return 0;
}

inline int convert(vector<string> args){
    converter::Options opt;
    for(size_t i=0;i<args.size();i++){
        const string& a=args[i];
        if(a=="-o"||a=="--out"){ if(i+1<args.size()) opt.out_dir=args[++i]; }
        else if(a=="--config"){ if(i+1<args.size()) opt.config_path=args[++i]; }
        else if(a=="--tokenizer"){ if(i+1<args.size()) opt.tokenizer_dir=args[++i]; }
        else if(a=="--dtype"){ if(i+1<args.size()){ const int d=converter::dtype_from_name(args[++i]); if(d>=0) opt.out_dtype=(uint8_t)d; } }
        else if(a=="--no-tokenizer") opt.copy_tokenizer=false;
        else if(!a.empty()&&a[0]=='-'){ cout<<"不认识的参数: "<<a<<"\n"; return 1; }
        else{
            namespace fs=std::filesystem;
            if(fs::is_directory(a)){
                for(const auto& e:fs::directory_iterator(a)){
                    if(e.path().extension()==".safetensors") opt.sources.push_back(e.path().string());
                }
                if(opt.tokenizer_dir.empty()) opt.tokenizer_dir=a;
                if(opt.config_path.empty()) opt.config_path=(fs::path(a)/"config.json").string();
            }else opt.sources.push_back(a);
        }
    }
    if(opt.sources.empty()){ usage(); return 1; }
    if(opt.out_dir.empty()){
        opt.out_dir=(std::filesystem::path(opt.sources[0]).parent_path()/"jieyu_bf").string();
    }
    if(opt.config_path.empty()){
        const std::filesystem::path guess=std::filesystem::path(opt.sources[0]).parent_path()/"config.json";
        std::error_code ec;
        if(std::filesystem::exists(guess,ec)) opt.config_path=guess.string();
    }
    if(opt.tokenizer_dir.empty()) opt.tokenizer_dir=std::filesystem::path(opt.sources[0]).parent_path().string();
    try{
        converter::convert(opt);
    }catch(const std::exception& e){
        cout<<"转换失败: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}

inline int run_model(const string& dir,const string& prompt,int max_new,float temperature,bool chat){
    pipeline::LoadedModel m;
    string err;
    const int want=read_json_int(dir+"/model_info.json","max_position_embeddings",1024);
    const int max_seq=(std::max)(64,(std::min)(want,4096));
    if(!pipeline::load_model_dir(m,dir,max_seq,&err)){
        cout<<"装载失败: "<<err<<"\n";
        return 1;
    }
    cout<<"模型   "<<dir<<"\n";
    cout<<"超参   hidden="<<m.refs.hidden<<" layers="<<m.refs.num_layers
        <<" heads="<<m.refs.num_heads<<" kv="<<m.refs.num_kv_heads
        <<" head_dim="<<m.refs.head_dim<<" inter="<<m.refs.intermediate
        <<" vocab="<<m.refs.vocab_size<<"\n";
    cout<<"词表   "<<m.tokenizer.vocab_size()<<" 个 token，最大长度 "<<m.refs.max_seq<<"\n";
    cout<<"说明   现在没有 KV cache，每生成一个 token 都要重算整段前缀，越长越慢\n";
    int code=0;
    if(chat) code=pipeline::run_chat(m,max_new,temperature);
    else code=pipeline::run_once(m,prompt,max_new,temperature);
    m.free();
    return code;
}

inline int main_impl(int argc,char** argv){
    const string mode=argv[1];
    if(mode=="--demo"){
        const string root=(argc>2)?argv[2]:"_file/demo";
        return demo(root);
    }
    if(mode=="--show"){
        if(argc<3){ usage(); return 1; }
        try{ converter::show(argv[2]); }
        catch(const std::exception& e){ cout<<e.what()<<"\n"; return 1; }
        return 0;
    }
    if(mode=="--convert"){
        vector<string> rest;
        for(int i=2;i<argc;i++) rest.push_back(pipeline::acp_to_utf8(argv[i]));
        return convert(rest);
    }
    if(mode=="--gen"||mode=="--chat"){
        if(argc<3){ usage(); return 1; }
        const string dir=pipeline::acp_to_utf8(argv[2]);
        const bool chat=(mode=="--chat");
        string prompt;
        int max_new=64;
        float temperature=0.0f;
        int next=3;
        if(!chat){
            if(argc>3) prompt=pipeline::acp_to_utf8(argv[3]);
            next=4;
        }
        if(argc>next) max_new=(std::max)(1,std::atoi(argv[next]));
        if(argc>next+1) temperature=(float)std::atof(argv[next+1]);
        return run_model(dir,prompt,max_new,temperature,chat);
    }
    usage();
    return 1;
}

}  // namespace cli

int main(int argc,char** argv){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if(argc>1) return cli::main_impl(argc,argv);
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
    if (check_bfile()) {
        cout << "model.bf 测试通过！" << endl;
    } else {
        cout << "model.bf 测试失败！" << endl;
    }
    if (check_tensor_dir()) {
        cout << "model.bf 张量目录测试通过！" << endl;
    } else {
        cout << "model.bf 张量目录测试失败！" << endl;
    }
    if (check_ffn()) {
        cout << "FFN 测试通过！" << endl;
    } else {
        cout << "FFN 测试失败！" << endl;
    }
    if (check_rope()) {
        cout << "RoPE 测试通过！" << endl;
    } else {
        cout << "RoPE 测试失败！" << endl;
    }
    if (check_rmsnorm()) {
        cout << "RMSNorm 测试通过！" << endl;
    } else {
        cout << "RMSNorm 测试失败！" << endl;
    }
    if (check_attention()) {
        cout << "Attention 测试通过！" << endl;
    } else {
        cout << "Attention 测试失败！" << endl;
    }
    if (check_transformer_layer()) {
        cout << "Transformer 整层测试通过！" << endl;
    } else {
        cout << "Transformer 整层测试失败！" << endl;
    }
    if (check_generate()) {
        cout << "自回归生成测试通过！" << endl;
    } else {
        cout << "自回归生成测试失败！" << endl;
    }
    if (check_convert()) {
        cout << "safetensors 转换器测试通过！" << endl;
    } else {
        cout << "safetensors 转换器测试失败！" << endl;
    }
    if (check_bind_weights()) {
        cout << "权重装载测试通过！" << endl;
    } else {
        cout << "权重装载测试失败！" << endl;
    }
    if (check_pipeline()) {
        cout << "端到端（文本进文本出）测试通过！" << endl;
    } else {
        cout << "端到端（文本进文本出）测试失败！" << endl;
    }
    return 0;
}