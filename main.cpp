#include"libs/libs.h"
#include"libs/text/text.h"
#include"libs/date_libs/consten_pool.h"
#include<cstdio>
#if defined(_WIN32)
#include<shellapi.h>
#endif
using namespace std;
using namespace json_libs;
namespace cli {

inline void usage(){
    printf("用法：\n"
           "  jieyu -run <名字> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]    装载 models/<名字> 并交互生成\n"
           "  jieyu -list                                    同上\n"
           "  jieyu -info <名字>                             看 models/<名字>/model_info.json 里的作者/描述\n"
           "  jieyu -setup                                   建目录/文件：config.json、langrage/、models/、_file/\n"
           "  jieyu -langrage [名字|{json}]                  切语言（写进 config.json），不带参数就看当前和可用语言\n"
           "  jieyu -convert <safetensors...|xxx.gguf> -o <目录> [-config f] [-tokenizer d] [-dtype quant|f32|f16]\n"
           "  jieyu -chat <模型目录> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]\n"
           "  jieyu -?                                       看这份用法\n"
           "    （[最多生成] 省了或给 0 就是不限，生成到 <|im_end|> / 上下文满为止）\n"
           "    （-tokenspeed 跟在 -run / -gen / -chat 后面：每次生成完分开打印预填充与解码两段速度，位置随便放）\n");
}

inline int read_json_int(const string& path,const string& key,int fallback){
    json_lib j;
    if(!j.SetJsonWay(path)) return fallback;
    return j.GetJsonInt(key,fallback);
}

inline bool looks_gguf(const string& p){
    if(path(p).extension()==".gguf") return true;
    ifstream f(p,ios::binary);
    if(!f) return false;
    char m[4]={0,0,0,0};
    f.read(m,4);
    return f.gcount()==4&&memcmp(m,"GGUF",4)==0;
}

inline string find_model_dir(const string& name){
    if(is_directory(name)) return name;
    const path p=path("models")/name;
    if(is_directory(p)) return p.string();
    return string();
}

inline int list_models(){
    printf("models/ 下可以跑的模型：\n");
    int n=0;
    if(is_directory("models")){
        for(const auto& e:directory_iterator("models")){
            if(!is_directory(e.path())) continue;
            if(!exists(e.path()/"model.bf")) continue;
            const string d=e.path().string();
            std::error_code ec;
            const double gb=(double)file_size(e.path()/"model.bf",ec)/(1<<30);
            printf("  %-22s hidden=%-6d layers=%-3d heads=%-3d head_dim=%-4d vocab=%-7d %.2f GB\n",
                   e.path().filename().string().c_str(),
                   read_json_int(d+"/model_info.json","hidden_size",0),
                   read_json_int(d+"/model_info.json","layer_count",0),
                   read_json_int(d+"/model_info.json","num_heads",0),
                   read_json_int(d+"/model_info.json","head_dim",0),
                   read_json_int(d+"/model_info.json","vocab_size",0),gb);
            n++;
        }
    }
    if(n==0) printf("  （空的，先 jieyu -convert <xxx.gguf> -o models/<名字>）\n");
    return n==0?1:0;
}

inline int convert(vector<string> args){
    converter::Options opt;
    string gguf_src;
    bool quant=true;
    for(size_t i=0;i<args.size();i++){
        const string& a=args[i];
        string key;
        if(!a.empty()&&a[0]=='-'){ key=a; while(!key.empty()&&key[0]=='-') key.erase(0,1); }
        if(key=="o"||key=="out"){ if(i+1<args.size()) opt.out_dir=args[++i]; }
        else if(key=="config"){ if(i+1<args.size()) opt.config_path=args[++i]; }
        else if(key=="tokenizer"){ if(i+1<args.size()) opt.tokenizer_dir=args[++i]; }
        else if(key=="dtype"){
            if(i+1<args.size()){
                const string v=args[++i];
                const int d=converter::dtype_from_name(v);
                if(d>=0){ opt.out_dtype=(uint8_t)d; quant=false; }
                else quant=true;
            }
        }
        else if(key=="no-tokenizer") opt.copy_tokenizer=false;
        else if(!a.empty()&&a[0]=='-'){ printf("不认识的参数: %s\n",a.c_str()); return 1; }
        else if(is_directory(a)){
            for(const auto& e:directory_iterator(a))
                if(e.path().extension()==".safetensors") opt.sources.push_back(e.path().string());
            if(opt.tokenizer_dir.empty()) opt.tokenizer_dir=a;
            if(opt.config_path.empty()) opt.config_path=(path(a)/"config.json").string();
        }
        else if(looks_gguf(a)) gguf_src=a;
        else opt.sources.push_back(a);
    }
    if(!gguf_src.empty()){
        ggufc::Options g;
        g.src=gguf_src;
        g.out_dir=opt.out_dir.empty()?(path(gguf_src).parent_path()/path(gguf_src).stem()).string():opt.out_dir;
        g.out_dtype=opt.out_dtype;
        g.quant=quant;
        try{
            const ggufc::Result r=ggufc::convert(g);
            printf("转换完成  %s\n"
                   "  超参   hidden=%llu layers=%llu heads=%u kv=%u head_dim=%u inter=%u vocab=%llu\n"
                   "  张量   %llu 个（2D 转置 %d，q/k 置换 %d，原样搬量化块 %d%s）\n"
                   "  RoPE   theta=%lld  max_position=%lld  eos=%d\n"
                   "  文件   %llu 字节\n",
                   g.out_dir.c_str(),(unsigned long long)r.head.hidden,(unsigned long long)r.head.layers,
                   (unsigned)r.head.num_heads,(unsigned)r.head.num_kv_heads,(unsigned)r.head.head_dim,
                   (unsigned)r.head.intermediate,(unsigned long long)r.head.vocab_size,
                   (unsigned long long)r.tensor_count,r.transposed,r.permuted,r.quant,
                   r.tied_lm_head?",lm_head 用 embed":"",
                   (long long)r.rope_theta,(long long)r.max_position,r.eos_id,
                   (unsigned long long)r.file_size);
        }catch(const std::exception& e){
            printf("转换失败: %s\n",e.what());
            return 1;
        }
        return 0;
    }
    if(opt.sources.empty()){ usage(); return 1; }
    if(opt.out_dir.empty()) opt.out_dir=(path(opt.sources[0]).parent_path()/"jieyu_bf").string();
    if(opt.config_path.empty()){
        const path guess=path(opt.sources[0]).parent_path()/"config.json";
        if(exists(guess)) opt.config_path=guess.string();
    }
    if(opt.tokenizer_dir.empty()) opt.tokenizer_dir=path(opt.sources[0]).parent_path().string();
    try{
        converter::convert(opt);
    }catch(const std::exception& e){
        printf("转换失败: %s\n",e.what());
        return 1;
    }
    return 0;
}

inline bool collect_pos(int argc,char** argv,int from,vector<string>& pos){
    bool show_speed=false;
    for(int i=from;i<argc;i++){
        string a=argv[i];
        while(!a.empty()&&(a[0]=='-'||a[0]=='/')) a.erase(0,1);
        if(a=="tokenspeed"||a=="speed"){ show_speed=true; continue; }
        pos.push_back(pipeline::acp_to_utf8(argv[i]));
    }
    return show_speed;
}

inline int run_model(const string& dir,const string& prompt,int max_new,float temperature,float repetition_penalty,bool chat,int max_seq_want,bool show_speed=false){
    pipeline::LoadedModel m;
    string err;
    const int want=read_json_int(dir+"/model_info.json","max_position_embeddings",1024);
    int max_seq=(std::max)(64,(std::min)(want,4096));
    if(max_seq_want>0) max_seq=(std::max)(8,max_seq_want);
    if(!pipeline::load_model_dir(m,dir,max_seq,&err)){
        printf("装载失败: %s\n",err.c_str());
        return 1;
    }
    printf("模型   %s\n",dir.c_str());
    const int code=chat?pipeline::run_chat(m,max_new,temperature,repetition_penalty,show_speed)
        :pipeline::run_once(m,prompt,max_new,temperature,repetition_penalty,show_speed);
    m.free();
    return code;
}

inline int selftest(){
    setup_all();
    printf("稍等，我们正在获取你的cpu信息\n");
    cpu_can.GetCpuCan();
    OutCPUInfo(cpu_can);
    struct Case{ const char* name; bool (*fn)(); };
    const Case cases[]={
        {"matmul",check_matmul},
        {"model.bf",check_bfile},
        {"model.bf 张量目录",check_tensor_dir},
        {"FFN",check_ffn},
        {"RoPE",check_rope},
        {"RMSNorm",check_rmsnorm},
        {"Attention",check_attention},
        {"Transformer 整层",check_transformer_layer},
        {"自回归生成",check_generate},
        {"KV cache",check_kv_cache},
        {"safetensors 转换器",check_convert},
        {"权重装载",check_bind_weights},
        {"端到端（文本进文本出）",check_pipeline},
    };
    int bad=0;
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
        if(cases[i].fn()) printf("%s 测试通过！\n",cases[i].name);
        else{ printf("%s 测试失败！\n",cases[i].name); bad++; }
    }
    if(bad==0) printf("全部测试通过！\n");
    else printf("有 %d 组测试失败！\n",bad);
    return bad==0?0:1;
}

inline int main_impl(int argc,char** argv){
    cpu_can.GetCpuCan();
    if(argc<2) return selftest();
    string mode=argv[1];
    while(!mode.empty()&&(mode[0]=='-'||mode[0]=='/')) mode.erase(0,1);
    if(mode=="?"){ usage(); return 0; }
    if(mode=="h"||mode=="help"||mode=="test"||mode=="text") return selftest();
    if(mode=="v") printf("%s",consten_pool.version.charv.c_str());
    if(mode=="run"){
        if(argc<3) return list_models();
        const string dir=find_model_dir(pipeline::acp_to_utf8(argv[2]));
        if(dir.empty()){ printf("找不到模型 %s\n",argv[2]); return list_models(); }
        vector<string> pos;
        const bool show_speed=collect_pos(argc,argv,3,pos);
        int max_new=0,max_seq_want=0;
        float temperature=0.7f,rep_penalty=1.1f;
        if(pos.size()>0) max_new=(std::max)(0,std::atoi(pos[0].c_str()));
        if(pos.size()>1) temperature=(float)std::atof(pos[1].c_str());
        if(pos.size()>2) max_seq_want=(std::max)(8,std::atoi(pos[2].c_str()));
        if(pos.size()>3) rep_penalty=(float)std::atof(pos[3].c_str());
        return run_model(dir,"",max_new,temperature,rep_penalty,true,max_seq_want,show_speed);
    }
    if(mode=="show"){
        if(argc<3){ usage(); return 1; }
        try{ converter::show(argv[2]); }
        catch(const std::exception& e){ printf("%s\n",e.what()); return 1; }
        return 0;
    }
    if(mode=="convert"){
        vector<string> rest;
        for(int i=2;i<argc;i++) rest.push_back(pipeline::acp_to_utf8(argv[i]));
        return convert(rest);
    }
    if(mode=="chat"){
        if(argc<3){ usage(); return 1; }
        const string dir=pipeline::acp_to_utf8(argv[2]);
        const string found=find_model_dir(dir);
        const bool chat=(mode=="chat");
        vector<string> pos;
        const bool show_speed=collect_pos(argc,argv,3,pos);
        string prompt;
        int max_new=0,max_seq_want=0;
        float temperature=chat?0.7f:0.0f,rep_penalty=1.1f;
        size_t next=0;
        if(!chat){ if(pos.size()>next) prompt=pos[next]; next=1; }
        if(pos.size()>next) max_new=(std::max)(0,std::atoi(pos[next].c_str()));
        if(pos.size()>next+1) temperature=(float)std::atof(pos[next+1].c_str());
        if(pos.size()>next+2) max_seq_want=(std::max)(8,std::atoi(pos[next+2].c_str()));
        if(pos.size()>next+3) rep_penalty=(float)std::atof(pos[next+3].c_str());
        return run_model(found.empty()?dir:found,prompt,max_new,temperature,rep_penalty,chat,max_seq_want,show_speed);
    }
    if(mode=="setup"){
        setup_all();
        const char* items[]={"config.json","langrage/zh_cn.json","langrage/en_us.json"};
        for(const char* p:items) printf("%s%s\n",exists(path(p))?"  OK   ":"  FAIL ",p);
        printf("%s%s\n",exists(path("models"))?"  OK   ":"  FAIL ","models/");
        printf("%s%s\n",exists(path("_file"))?"  OK   ":"  FAIL ","_file/");
        return 0;
    }
    if(mode=="langrage"||mode=="language"){
        if(argc<3){
            printf("当前语言文件：%s\n可用的：\n",current_langrage().c_str());
            if(exists(path(langrage_dir())))
                for(const auto& e:directory_iterator(path(langrage_dir())))
                    if(e.path().extension()==".json") printf("  %s\n",e.path().filename().string().c_str());
            printf("用法：jieyu -langrage <名字|{json}>   例如 zh_cn.json 或 '{\"langrage\":\"en_us.json\"}'\n");
            return 0;
        }
        string arg=pipeline::acp_to_utf8(argv[2]);
        if(!arg.empty()&&arg[0]=='{'){
            create_directories(path("_file"));
            const string tmp="_file/.langrage_arg.json";
            { ofstream f(tmp,ios::binary|ios::trunc); f<<arg; }
            json_libs::json_lib src;
            if(!src.SetJsonWay(tmp)){
                printf("这段 json 解析不了：%s\n%s\n（PowerShell 会把双引号吃掉，用 cmd 或写成文件再传）\n",arg.c_str(),src.LastError().c_str());
                return 1;
            }
            src.ForEachKey([](const string& k,const string& v)->bool{
                json_libs::json_lib c;
                c.SetJsonWay(config_path());
                c.WriteJsonKey(k,v);
                return true;
            });
            std::filesystem::remove(path(tmp));
            printf("已合进 config.json，当前语言文件：%s\n",current_langrage().c_str());
            return 0;
        }
        if(arg.size()<5||arg.compare(arg.size()-5,5,".json")!=0) arg+=".json";
        if(!exists(path(langrage_dir())/arg)){
            write_language_file(arg);
            if(!exists(path(langrage_dir())/arg)){ printf("没有这个语言文件：%s\n",arg.c_str()); return 1; }
        }
        set_langrage(arg);
        printf("语言已设为 %s（写进 config.json）\n",arg.c_str());
        return 0;
    }
    if(mode=="list"){
        return list_models();
    }
    if(mode=="info"&&argc==3){
        const string name=pipeline::acp_to_utf8(argv[2]);
        json_libs::json_lib temp;
        temp.SetJsonWay("models/"+name+"/model_info.json");
        if(!temp.IsLoaded()) temp.SetJsonWay("models/"+name+".json");
        printf("%s：%s\n",tr("model_writer","作者").c_str(),temp.GetJsonData("writer").c_str());
        printf("%s：%s\n",tr("model_description","描述").c_str(),temp.GetJsonData("say").c_str());
        return 0;
    }
    usage();
    return 1;
}

}

#ifdef _WIN32
string wide_to_utf8(const wchar_t* w){
    if(w==nullptr) return string();
    const int n=WideCharToMultiByte(CP_UTF8,0,w,-1,nullptr,0,nullptr,nullptr);
    if(n<=1) return string();
    string s((size_t)(n-1),'\0');
    WideCharToMultiByte(CP_UTF8,0,w,-1,s.data(),n,nullptr,nullptr);
    return s;
}
#endif

int main(int argc,char** argv){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    int wc=0;
    LPWSTR* wa=CommandLineToArgvW(GetCommandLineW(),&wc);
    vector<string> args;
    vector<char*> av;
    if(wa!=nullptr){
        args.reserve((size_t)wc);
        for(int i=0;i<wc;i++) args.push_back(wide_to_utf8(wa[i]));
        LocalFree(wa);
        for(string& s:args) av.push_back(s.data());
        argc=(int)av.size();
        argv=av.data();
    }
#endif
    return cli::main_impl(argc,argv);
}