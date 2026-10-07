#include"libs/libs.h"
#include"libs/text/text.h"
#include<shellapi.h>
using namespace std;
using namespace json_libs;
namespace cli {

inline void usage(){
    cout<<"用法：\n"
        <<"  jieyu                                          跑全部自测（= jieyu -test / jieyu -text）\n"
        <<"  jieyu -run <名字> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]    装载 models/<名字> 并交互生成\n"
        <<"  jieyu -run                                     列出 models/ 下能跑的模型\n"
        <<"  jieyu -list                                    同上\n"
        <<"  jieyu -info <名字>                             看 models/<名字>/model_info.json 里的作者/描述\n"
        <<"  jieyu -setup                                   建目录/文件：config.json、langrage/、models/、_file/\n"
        <<"  jieyu -langrage [名字|{json}]                  切语言（写进 config.json），不带参数就看当前和可用语言\n"
        <<"  jieyu -demo [目录]                             造演示小模型，文本进文本出\n"
        <<"  jieyu -convert <safetensors...|xxx.gguf> -o <目录> [-config f] [-tokenizer d] [-dtype quant|f32|f16]\n"
        <<"  jieyu -gguf <xxx.gguf>                         看 GGUF：元数据 / 词表 / 量化分布 / 张量清单\n"
        <<"  jieyu -show <model.bf>                         打印 model.bf 的张量目录\n"
        <<"  jieyu -gen <模型目录> <提示词> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]\n"
        <<"  jieyu -chat <模型目录> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]\n"
        <<"  jieyu -?                                       看这份用法\n"
        <<"    （[最多生成] 省了或给 0 就是不限，生成到 <|im_end|> / 上下文满为止）\n"
        <<"    （-tokenspeed 跟在 -run / -gen / -chat 后面：每次生成完分开打印预填充与解码两段速度，位置随便放）\n";
}

inline int read_json_int(const string& path,const string& key,int fallback){
    json_lib j;
    if(!j.SetJsonWay(path)) return fallback;
    return j.GetJsonInt(key,fallback);
}

inline bool utf8_ok(const string& s){
    size_t i=0;
    while(i<s.size()){
        const unsigned char c=(unsigned char)s[i];
        size_t need=1;
        if(c>=0xF0u) need=4;
        else if(c>=0xE0u) need=3;
        else if(c>=0xC0u) need=2;
        else if(c>=0x80u) return false;
        if(i+need>s.size()) return false;
        for(size_t k=1;k<need;k++)
            if(((unsigned char)s[i+k]&0xC0u)!=0x80u) return false;
        i+=need;
    }
    return true;
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
    cout<<"models/ 下可以跑的模型：\n";
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
    if(n==0) cout<<"  （空的，先 jieyu -convert <xxx.gguf> -o models/<名字>）\n";
    return n==0?1:0;
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
        else if(!a.empty()&&a[0]=='-'){ cout<<"不认识的参数: "<<a<<"\n"; return 1; }
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
            cout<<"转换完成  "<<g.out_dir<<"\n"
                <<"  超参   hidden="<<r.head.hidden<<" layers="<<r.head.layers
                <<" heads="<<(unsigned)r.head.num_heads<<" kv="<<(unsigned)r.head.num_kv_heads
                <<" head_dim="<<(unsigned)r.head.head_dim<<" inter="<<r.head.intermediate
                <<" vocab="<<r.head.vocab_size<<"\n"
                <<"  张量   "<<r.tensor_count<<" 个（2D 转置 "<<r.transposed
                <<"，q/k 置换 "<<r.permuted<<"，原样搬量化块 "<<r.quant
                <<(r.tied_lm_head?"，lm_head 用 embed":"")<<"）\n"
                <<"  RoPE   theta="<<r.rope_theta<<"  max_position="<<r.max_position<<"  eos="<<r.eos_id<<"\n"
                <<"  文件   "<<r.file_size<<" 字节\n";
        }catch(const std::exception& e){
            cout<<"转换失败: "<<e.what()<<"\n";
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
        cout<<"转换失败: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}

inline int gguf_info(const string& file){
    const gguf::File g=gguf::open(file);
    cout<<"GGUF v"<<g.version<<"  张量 "<<g.tensors.size()<<"  kv "<<g.kvs.size()
        <<"  对齐 "<<g.align<<"  数据区 "<<g.data_off<<"  文件 "<<g.file_size<<"\n";
    cout<<"---- 元数据 ----\n";
    for(size_t i=0;i<g.kvs.size();i++){
        const gguf::Meta& m=g.kvs[i].second;
        cout<<"  "<<g.kvs[i].first<<" = ";
        if(m.type==gguf::M_STR){
            cout<<"\""<<(m.s.size()>60?m.s.substr(0,60)+"...":m.s)<<"\"";
        }else if(m.type==gguf::M_ARR){
            const size_t show=(size_t)(m.n<6?m.n:6);
            cout<<"["<<m.n<<"]";
            if(m.elem_type==gguf::M_STR){ for(size_t j=0;j<show;j++) cout<<" \""<<m.strs[j]<<"\""; }
            else if(!m.floats.empty()){ for(size_t j=0;j<show;j++) cout<<" "<<m.floats[j]; }
            else{ for(size_t j=0;j<show;j++) cout<<" "<<m.ints[j]; }
            if(m.n>show) cout<<" ...";
        }else if(m.type==gguf::M_F32||m.type==gguf::M_F64){
            cout<<m.f;
        }else if(m.type==gguf::M_BOOL){
            cout<<(m.b?"true":"false");
        }else{
            cout<<m.i;
        }
        cout<<"\n";
    }
    const gguf::Meta* tk=g.meta("tokenizer.ggml.tokens");
    if(tk!=nullptr&&!tk->strs.empty()){
        size_t ctrl=0,badutf8=0,esc=0,ascii=0;
        int id_space=-1,id_gpt2_space=-1;
        for(size_t i=0;i<tk->strs.size();i++){
            const string& s=tk->strs[i];
            bool c=false,e=false;
            for(size_t j=0;j<s.size();j++){
                const unsigned char b=(unsigned char)s[j];
                if(b<0x20) c=true;
                if(b==0xC4&&j+1<s.size()&&(unsigned char)s[j+1]>=0x80) e=true;
                if(b==0xC5&&j+1<s.size()&&(unsigned char)s[j+1]<=0x83) e=true;
            }
            if(c) ctrl++;
            if(e) esc++;
            if(!utf8_ok(s)) badutf8++;
            if(s.size()==1&&(unsigned char)s[0]<0x80) ascii++;
            if(s==" ") id_space=(int)i;
            if(s=="\xC4\xA0") id_gpt2_space=(int)i;
        }
        cout<<"---- 词表体检 ----\n"
            <<"  共 "<<tk->strs.size()<<"  含控制字节 "<<ctrl<<"  非法UTF-8 "<<badutf8
            <<"  单字节ASCII "<<ascii<<"  含U+0100..U+0143 "<<esc<<"\n"
            <<"  id(空格字节)= "<<id_space<<"   id(GPT2的Ġ)= "<<id_gpt2_space
            <<"   ← Ġ 有值而空格字节没值 = GPT-2 字节转义，和本工程分词器一致\n";
    }
    cout<<"---- 量化类型分布 ----\n";
    map<string,int> hist;
    for(size_t i=0;i<g.tensors.size();i++) hist[gguf::type_name(g.tensors[i].type)]++;
    for(map<string,int>::iterator it=hist.begin();it!=hist.end();++it)
        cout<<"  "<<it->first<<" : "<<it->second<<"\n";
    cout<<"---- 张量清单（ggml 维度顺序，shape[0] 是最内层连续维）----\n";
    for(size_t i=0;i<g.tensors.size();i++){
        const gguf::Tensor& t=g.tensors[i];
        cout<<"  "<<t.name<<"  "<<gguf::type_name(t.type)<<"  [";
        for(size_t d=0;d<t.shape.size();d++){
            cout<<t.shape[d];
            if(d+1<t.shape.size()) cout<<",";
        }
        cout<<"]\n";
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
        cout<<"装载失败: "<<err<<"\n";
        return 1;
    }
    cout<<"模型   "<<dir<<"\n"
        <<"超参   hidden="<<m.refs.hidden<<" layers="<<m.refs.num_layers
        <<" heads="<<(unsigned)m.refs.num_heads<<" kv="<<(unsigned)m.refs.num_kv_heads
        <<" head_dim="<<(unsigned)m.refs.head_dim<<" inter="<<m.refs.intermediate
        <<" vocab="<<m.refs.vocab_size<<"\n"
        <<"词表   "<<m.tokenizer.vocab_size()<<" 个 token，最大长度 "<<m.refs.max_seq
        <<"，q/k/v bias "<<(m.refs.qkv_bias?"有":"无")
        <<"，add_bos "<<(m.refs.add_bos?"开":"关")
        <<"，重复惩罚 "<<repetition_penalty
        <<"，特殊 token "<<model.s_token.message_start<<"/"<<model.s_token.meesage_end
        <<"/"<<model.s_token.text_end<<"\n";
    const int code=chat?pipeline::run_chat(m,max_new,temperature,repetition_penalty,show_speed)
                     :pipeline::run_once(m,prompt,max_new,temperature,repetition_penalty,show_speed);
    m.free();
    return code;
}

inline int selftest(){
    setup_all();
    cout<<"稍等，我们正在获取你的cpu信息\n";
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
        if(cases[i].fn()) cout<<cases[i].name<<" 测试通过！\n";
        else{ cout<<cases[i].name<<" 测试失败！\n"; bad++; }
    }
    cout<<(bad==0?"全部测试通过！":"有 "+to_string(bad)+" 组测试失败！")<<"\n";
    return bad==0?0:1;
}

inline int main_impl(int argc,char** argv){
    cpu_can.GetCpuCan();
    if(argc<2) return selftest();
    string mode=argv[1];
    while(!mode.empty()&&(mode[0]=='-'||mode[0]=='/')) mode.erase(0,1);
    if(mode=="?"){ usage(); return 0; }
    if(mode=="h"||mode=="help"||mode=="test"||mode=="text") return selftest();
    if(mode=="run"){
        if(argc<3) return list_models();
        const string dir=find_model_dir(pipeline::acp_to_utf8(argv[2]));
        if(dir.empty()){ cout<<"找不到模型 "<<argv[2]<<"\n"; return list_models(); }
        vector<string> pos;
        const bool show_speed=collect_pos(argc,argv,3,pos);
        int max_new=0,max_seq_want=0;
        float temperature=0.7f,rep_penalty=1.1f;
        if(pos.size()>0) max_new=(std::max)(0,std::atoi(pos[0].c_str()));
        if(pos.size()>1) temperature=(float)std::atof(pos[1].c_str());
        if(pos.size()>2) max_seq_want=(std::max)(8,std::atoi(pos[2].c_str()));
        if(pos.size()>3) rep_penalty=(float)std::atof(pos[3].c_str());
        return run_model(dir,"",max_new,temperature,rep_penalty,true,max_seq_want,show_speed);
        cout<<"用法：jieyu -run <名字> [最多生成] [温度] [最大长度]\n";
    }
    if(mode=="demo") return demo((argc>2)?argv[2]:"_file/demo");
    if(mode=="show"){
        if(argc<3){ usage(); return 1; }
        try{ converter::show(argv[2]); }
        catch(const std::exception& e){ cout<<e.what()<<"\n"; return 1; }
        return 0;
    }
    if(mode=="gguf"){
        if(argc<3){ usage(); return 1; }
        try{ return gguf_info(pipeline::acp_to_utf8(argv[2])); }
        catch(const std::exception& e){ cout<<e.what()<<"\n"; return 1; }
    }
    if(mode=="convert"){
        vector<string> rest;
        for(int i=2;i<argc;i++) rest.push_back(pipeline::acp_to_utf8(argv[i]));
        return convert(rest);
    }
    if(mode=="gen"||mode=="chat"){
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
        for(const char* p:items) cout<<(exists(path(p))?"  OK   ":"  FAIL ")<<p<<"\n";
        cout<<(exists(path("models"))?"  OK   ":"  FAIL ")<<"models/\n";
        cout<<(exists(path("_file"))?"  OK   ":"  FAIL ")<<"_file/\n";
        return 0;
    }
    if(mode=="langrage"||mode=="language"){
        if(argc<3){
            cout<<"当前语言文件："<<current_langrage()<<"\n可用的：\n";
            if(exists(path(langrage_dir())))
                for(const auto& e:directory_iterator(path(langrage_dir())))
                    if(e.path().extension()==".json") cout<<"  "<<e.path().filename().string()<<"\n";
            cout<<"用法：jieyu -langrage <名字|{json}>   例如 zh_cn.json 或 '{\"langrage\":\"en_us.json\"}'\n";
            return 0;
        }
        string arg=pipeline::acp_to_utf8(argv[2]);
        if(!arg.empty()&&arg[0]=='{'){
            create_directories(path("_file"));
            const string tmp="_file/.langrage_arg.json";
            { ofstream f(tmp,ios::binary|ios::trunc); f<<arg; }
            json_libs::json_lib src;
            if(!src.SetJsonWay(tmp)){
                cout<<"这段 json 解析不了："<<arg<<"\n"<<src.LastError()<<"\n"
                    <<"（PowerShell 会把双引号吃掉，用 cmd 或写成文件再传）\n";
                return 1;
            }
            src.ForEachKey([](const string& k,const string& v)->bool{
                json_libs::json_lib c;
                c.SetJsonWay(config_path());
                c.WriteJsonKey(k,v);
                return true;
            });
            std::filesystem::remove(path(tmp));
            cout<<"已合进 config.json，当前语言文件："<<current_langrage()<<"\n";
            return 0;
        }
        if(arg.size()<5||arg.compare(arg.size()-5,5,".json")!=0) arg+=".json";
        if(!exists(path(langrage_dir())/arg)){
            write_language_file(arg);
            if(!exists(path(langrage_dir())/arg)){ cout<<"没有这个语言文件："<<arg<<"\n"; return 1; }
        }
        set_langrage(arg);
        cout<<"语言已设为 "<<arg<<"（写进 config.json）\n";
        return 0;
    }
    if(mode=="list"){
        return list_models();
    }
    if(mode=="info" && argc==3){
        const string name=pipeline::acp_to_utf8(argv[2]);
        json_libs::json_lib temp;
        temp.SetJsonWay("models/"+name+"/model_info.json");
        if(!temp.IsLoaded()) temp.SetJsonWay("models/"+name+".json");
        cout<<tr("model_writer","作者")<<"："<<temp.GetJsonData("writer")<<"\n";
        cout<<tr("model_description","描述")<<"："<<temp.GetJsonData("say")<<"\n";
        return 0;
    }
    usage();
    return 1;
}

}  // namespace cli

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
