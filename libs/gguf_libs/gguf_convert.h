#ifndef GGUF_CONVERT_H
#define GGUF_CONVERT_H
#include<algorithm>
#include<filesystem>
#include<fstream>
#include<iostream>
#include<string>
#include<vector>
#include"../model_libs/bfile.h"
#include"../model_libs/converter.h"
#include"gguf.h"

namespace ggufc {
using namespace std;
namespace fs=std::filesystem;

struct Options{
    string src,out_dir;
    uint8_t out_dtype=bfile::F32;
    bool quant=true;      // 把 GGUF 里的 q4_K/q6_K 块原样搬过去（不反量化，最省）
    bool verbose=true;
};

struct Result{
    bfile::Header head;
    long long rope_theta=10000,max_position=32768;
    size_t tensor_count=0;
    uint64_t file_size=0;
    int transposed=0,permuted=0,quant=0;
    bool tied_lm_head=false;
    int eos_id=-1;
    string arch;
};

// 能原样搬的量化类型 → bfile 的 dtype 代号；不能就 -1
inline int quant_code(uint32_t t){
    switch(t){
        case gguf::T_Q4_K: return bfile::Q4K;
        case gguf::T_Q5_K: return bfile::Q5K;
        case gguf::T_Q6_K: return bfile::Q6K;
        case gguf::T_Q8_0: return bfile::Q8_0;
        default: return -1;
    }
}

inline long long meta_int(const gguf::File& g,const string& k,long long d){
    const gguf::Meta* m=g.meta(k);
    return m==nullptr?d:m->as_int(d);
}

inline double meta_float(const gguf::File& g,const string& k,double d){
    const gguf::Meta* m=g.meta(k);
    return m==nullptr?d:m->as_float(d);
}

inline string jesc(const string& s){
    string o;
    for(size_t i=0;i<s.size();i++){
        const char c=s[i];
        if(c=='"') o+="\\\"";
        else if(c=='\\') o+="\\\\";
        else o+=c;
    }
    return o;
}

inline int perm_row(int r,int hd){
    const int h=r/hd,k=r%hd;
    return h*hd+((k&1)?(hd/2+k/2):(k/2));
}

inline const char* suffix_map(const string& s){
    if(s=="attn_norm.weight") return "input_layernorm.weight";
    if(s=="attn_q.weight") return "self_attn.q_proj.weight";
    if(s=="attn_q.bias") return "self_attn.q_proj.bias";
    if(s=="attn_k.weight") return "self_attn.k_proj.weight";
    if(s=="attn_k.bias") return "self_attn.k_proj.bias";
    if(s=="attn_v.weight") return "self_attn.v_proj.weight";
    if(s=="attn_v.bias") return "self_attn.v_proj.bias";
    if(s=="attn_output.weight") return "self_attn.o_proj.weight";
    if(s=="ffn_norm.weight") return "post_attention_layernorm.weight";
    if(s=="ffn_gate.weight") return "mlp.gate_proj.weight";
    if(s=="ffn_down.weight") return "mlp.down_proj.weight";
    if(s=="ffn_up.weight") return "mlp.up_proj.weight";
    return nullptr;
}

inline string hf_name(const string& n){
    if(n=="token_embd.weight") return "model.embed_tokens.weight";
    if(n=="output_norm.weight") return "model.norm.weight";
    if(n=="output.weight") return "lm_head.weight";
    if(n.size()>4&&n.compare(0,4,"blk.")==0){
        const size_t b=n.find('.',4);
        if(b==string::npos) return string();
        const string idx=n.substr(4,b-4);
        if(idx.empty()) return string();
        for(size_t i=0;i<idx.size();i++) if(idx[i]<'0'||idx[i]>'9') return string();
        const char* sfx=suffix_map(n.substr(b+1));
        if(sfx==nullptr) return string();
        return "model.layers."+idx+"."+sfx;
    }
    return string();
}

inline bool is_qk(const string& n){
    return n.find(".attn_q.")!=string::npos||n.find(".attn_k.")!=string::npos;
}

inline uint64_t elems_of(const vector<uint32_t>& sh){
    uint64_t e=1;
    for(size_t i=0;i<sh.size();i++) e*=sh[i];
    return e;
}

inline bfile::StreamTensor make_sink(const gguf::File& g,const gguf::Tensor& t,
                                     const string& name,bool transpose,bool permute,
                                     int head_dim,uint8_t out_dtype){
    const uint64_t in_dim=t.shape[0],out_dim=(t.shape.size()>1)?t.shape[1]:0;
    bfile::StreamTensor s;
    s.name=name;
    s.dtype=out_dtype;
    if(t.shape.size()==1) s.shape={(uint32_t)t.shape[0]};
    else if(transpose) s.shape={(uint32_t)in_dim,(uint32_t)out_dim};
    else s.shape={(uint32_t)out_dim,(uint32_t)in_dim};
    s.nbytes=elems_of(s.shape)*bfile::dtype_bytes(out_dtype);

    const string path=g.path;
    const uint64_t base=g.data_off+t.offset;
    const uint32_t type=t.type;
    const size_t be=gguf::block_elems(type),bb=gguf::block_bytes(type);
    const uint64_t total=t.n_elem,nbytes=t.nbytes;
    s.emit=[path,base,type,be,bb,total,nbytes,in_dim,out_dim,transpose,permute,head_dim,out_dtype]
           (ostream& f)->bool{
        ifstream in(path,ios::binary);
        if(!in) return false;
        vector<unsigned char> raw;
        vector<float> tmp;
        if(permute&&!transpose){
            raw.resize((size_t)nbytes);
            in.seekg((streamoff)base);
            in.read((char*)raw.data(),(streamsize)raw.size());
            if((uint64_t)in.gcount()!=raw.size()) return false;
            tmp.resize((size_t)total);
            gguf::dequant(type,raw.data(),tmp.data(),total);
            vector<float> o((size_t)total);
            for(uint64_t i=0;i<total;i++) o[(size_t)i]=tmp[(size_t)perm_row((int)i,head_dim)];
            converter::write_as(f,o.data(),total,out_dtype);
            return (bool)f;
        }
        if(!transpose){
            const uint64_t chunk=((uint64_t)(4u<<20)/bb)*be;
            uint64_t done=0;
            while(done<total){
                const uint64_t n=(std::min)(chunk,total-done);
                const uint64_t bytes=(n/be)*bb;
                raw.resize((size_t)bytes);
                in.seekg((streamoff)(base+(done/be)*bb));
                in.read((char*)raw.data(),(streamsize)bytes);
                if((uint64_t)in.gcount()!=bytes) return false;
                tmp.resize((size_t)n);
                gguf::dequant(type,raw.data(),tmp.data(),n);
                converter::write_as(f,tmp.data(),n,out_dtype);
                done+=n;
            }
            return (bool)f;
        }
        uint64_t tile=(128u<<20)/(4*out_dim);
        if(tile<1) tile=1;
        if(tile>in_dim) tile=in_dim;
        uint64_t rb=(8u<<20)/(4*in_dim);
        rb=rb/(uint64_t)head_dim*(uint64_t)head_dim;
        if(rb<(uint64_t)head_dim) rb=(uint64_t)head_dim;
        if(rb>out_dim) rb=out_dim;
        vector<float> dst;
        for(uint64_t t0=0;t0<in_dim;t0+=tile){
            const uint64_t t1=(std::min)(in_dim,t0+tile);
            dst.assign((size_t)((t1-t0)*out_dim),0.0f);
            for(uint64_t j0=0;j0<out_dim;j0+=rb){
                const uint64_t j1=(std::min)(out_dim,j0+rb);
                const uint64_t n=(j1-j0)*in_dim;
                const uint64_t bytes=(n/be)*bb;
                raw.resize((size_t)bytes);
                in.seekg((streamoff)(base+((j0*in_dim)/be)*bb));
                in.read((char*)raw.data(),(streamsize)bytes);
                if((uint64_t)in.gcount()!=bytes) return false;
                tmp.resize((size_t)n);
                gguf::dequant(type,raw.data(),tmp.data(),n);
                for(uint64_t j=j0;j<j1;j++){
                    const uint64_t r2=permute?(uint64_t)perm_row((int)j,head_dim):j;
                    const float* srow=tmp.data()+(size_t)(r2-j0)*in_dim;
                    for(uint64_t t=t0;t<t1;t++) dst[(size_t)((t-t0)*out_dim+j)]=srow[t];
                }
            }
            converter::write_as(f,dst.data(),dst.size(),out_dtype);
        }
        return (bool)f;
    };
    return s;
}

// 量化块原样搬：不反量化、不转置，只把行按需重排（q/k 置换）
// 行 = 输出（embed 就是词），块沿 k
inline bfile::StreamTensor make_raw_sink(const gguf::File& g,const gguf::Tensor& t,
                                         const string& name,bool permute,int head_dim,
                                         uint8_t dtype){
    const uint64_t k=t.shape[0],n=t.shape[1];
    const size_t be=gguf::block_elems(t.type),bb=gguf::block_bytes(t.type);
    if(bfile::block_elems(dtype)!=be||bfile::block_bytes(dtype)!=bb)
        throw runtime_error("gguf: 量化块尺寸对不上 "+t.name);
    bfile::StreamTensor s;
    s.name=name;
    s.dtype=dtype;
    s.shape={(uint32_t)n,(uint32_t)k};
    s.nbytes=(size_t)n*(k/be)*bb;
    const string path=g.path;
    const uint64_t base=g.data_off+t.offset,rbytes=(k/be)*bb;
    s.emit=[path,base,rbytes,n,permute,head_dim](ostream& f)->bool{
        ifstream in(path,ios::binary);
        if(!in) return false;
        vector<unsigned char> row((size_t)rbytes);
        for(uint64_t j=0;j<n;j++){
            const uint64_t src=permute?(uint64_t)perm_row((int)j,head_dim):j;
            in.seekg((streamoff)(base+src*rbytes));
            in.read((char*)row.data(),(streamsize)rbytes);
            if((uint64_t)in.gcount()!=rbytes) return false;
            f.write((const char*)row.data(),(streamsize)rbytes);
        }
        return (bool)f;
    };
    return s;
}

inline void write_tokenizer(const gguf::File& g,const string& out_dir,Result& r,bool verbose){
    const gguf::Meta* tk=g.meta("tokenizer.ggml.tokens");
    if(tk==nullptr||tk->strs.empty()) throw runtime_error("gguf: 没有 tokenizer.ggml.tokens，抽不出词表");
    r.eos_id=(int)meta_int(g,"tokenizer.ggml.eos_token_id",-1);
    if(r.eos_id<0) r.eos_id=(int)meta_int(g,"tokenizer.ggml.bos_token_id",-1);
    {
        ofstream v((fs::path(out_dir)/"vocab.json").string(),ios::binary|ios::trunc);
        if(!v) throw runtime_error("gguf: 写不了 vocab.json");
        v<<"{\n";
        for(size_t i=0;i<tk->strs.size();i++){
            if(i!=0) v<<",\n";
            v<<"  \""<<jesc(tk->strs[i])<<"\": "<<i;
        }
        v<<"\n}\n";
    }
    const gguf::Meta* mg=g.meta("tokenizer.ggml.merges");
    {
        ofstream m((fs::path(out_dir)/"merges.txt").string(),ios::binary|ios::trunc);
        if(!m) throw runtime_error("gguf: 写不了 merges.txt");
        m<<"#version: 0.2\n";
        if(mg!=nullptr)
            for(size_t i=0;i<mg->strs.size();i++) m<<mg->strs[i]<<"\n";
    }
    if(verbose)
        cout<<"  词表 "<<tk->strs.size()<<" 个 token，合并规则 "
            <<(mg==nullptr?0:mg->strs.size())<<" 条，eos="<<r.eos_id<<"\n";
}

inline Result convert(const Options& opt){
    gguf::File g=gguf::open(opt.src);
    Result r;
    const gguf::Meta* am=g.meta("general.architecture");
    r.arch=(am!=nullptr&&!am->s.empty())?am->s:string("llama");
    const string pre=r.arch+".";
    if(opt.verbose)
        cout<<"GGUF v"<<g.version<<"  "<<r.arch<<"  张量 "<<g.tensors.size()<<"\n";

    long long hidden=meta_int(g,pre+"embedding_length",0);
    long long nl=meta_int(g,pre+"block_count",0);
    long long inter=meta_int(g,pre+"feed_forward_length",0);
    long long nh=meta_int(g,pre+"attention.head_count",0);
    long long nkv=meta_int(g,pre+"attention.head_count_kv",nh);
    long long hd=meta_int(g,pre+"attention.key_length",0);
    const double rope=meta_float(g,pre+"rope.freq_base",10000.0);
    long long ctx=meta_int(g,pre+"context_length",1024);

    const gguf::Tensor* emb=g.find("token_embd.weight");
    long long vocab=0;
    if(emb!=nullptr){
        if(hidden<=0) hidden=(long long)emb->shape[0];
        vocab=(long long)emb->shape[1];
    }
    if(vocab<=0){
        const gguf::Meta* tk=g.meta("tokenizer.ggml.tokens");
        if(tk!=nullptr) vocab=(long long)tk->n;
    }
    if(nl<=0){
        for(size_t i=0;i<g.tensors.size();i++){
            const string& n=g.tensors[i].name;
            if(n.size()>4&&n.compare(0,4,"blk.")==0){
                const long long id=atoll(n.c_str()+4);
                if(id+1>nl) nl=id+1;
            }
        }
    }
    if(hidden<=0||vocab<=0||nl<=0||inter<=0)
        throw runtime_error("gguf: 超参不全（hidden / layers / intermediate / vocab）");
    if(nh<=0) throw runtime_error("gguf: 没有 attention.head_count");
    if(hd<=0) hd=hidden/nh;
    if(nh>255||nkv>255||hd>255) throw runtime_error("gguf: num_heads / num_kv_heads / head_dim 必须落在 1..255");
    if(nh%nkv!=0) throw runtime_error("gguf: num_heads 必须能被 num_kv_heads 整除");
    if(hidden!=nh*hd)
        cout<<"[警告] hidden("<<hidden<<") != num_heads*head_dim("<<nh*hd<<")\n";

    for(size_t i=0;i<g.tensors.size();i++){
        const gguf::Tensor& t=g.tensors[i];
        if(t.shape.size()==2&&t.shape[0]%gguf::block_elems(t.type)!=0)
            throw runtime_error("gguf: "+t.name+" 的行长不是量化块的整数倍");
    }

    vector<bfile::StreamTensor> sinks;
    vector<string> skipped;
    for(size_t i=0;i<g.tensors.size();i++){
        const gguf::Tensor& t=g.tensors[i];
        const string hf=hf_name(t.name);
        if(hf.empty()){ skipped.push_back(t.name); continue; }
        const bool is_emb=(t.name=="token_embd.weight");
        const bool tr=(t.shape.size()==2&&!is_emb);
        const bool pm=is_qk(t.name);
        const int qc=(t.shape.size()==2)?quant_code(t.type):-1;
        if(opt.quant&&qc>=0){
            sinks.push_back(make_raw_sink(g,t,hf,pm,(int)hd,(uint8_t)qc));
            r.quant++;
        }else{
            const uint8_t dt=(t.shape.size()==2)?opt.out_dtype:(uint8_t)bfile::F32;
            sinks.push_back(make_sink(g,t,hf,tr,pm,(int)hd,dt));
            if(tr) r.transposed++;
        }
        if(pm) r.permuted++;
    }
    if(g.find("output.weight")==nullptr){
        if(emb==nullptr) throw runtime_error("gguf: 既没有 output.weight 也没有 token_embd.weight");
        r.tied_lm_head=true;
        // 量化的 embed 本身就是“行=输出”，直接当 lm_head 用，不用再复制一份
        if(!(opt.quant&&quant_code(emb->type)>=0)){
            sinks.push_back(make_sink(g,*emb,"lm_head.weight",true,false,(int)hd,opt.out_dtype));
            r.transposed++;
        }
    }
    if(skipped.size()!=0&&opt.verbose){
        cout<<"  跳过 "<<skipped.size()<<" 个不认识的张量：";
        for(size_t i=0;i<skipped.size()&&i<6;i++) cout<<skipped[i]<<" ";
        if(skipped.size()>6) cout<<"...";
        cout<<"\n";
    }

    r.head.hidden=(uint64_t)hidden;
    r.head.layers=(uint64_t)nl;
    r.head.mode=1;
    r.head.num_heads=(uint8_t)nh;
    r.head.num_kv_heads=(uint8_t)nkv;
    r.head.head_dim=(uint8_t)hd;
    r.head.intermediate=(uint32_t)inter;
    r.head.vocab_size=(uint64_t)vocab;
    r.rope_theta=(long long)(rope<0?-rope:rope);
    r.max_position=ctx;

    fs::create_directories(opt.out_dir);
    const string bf=(fs::path(opt.out_dir)/"model.bf").string();
    if(opt.verbose)
        cout<<"  写 model.bf（"<<sinks.size()<<" 个张量，其中 "<<r.transposed<<" 个转置、"
            <<r.permuted<<" 个 q/k 置换、"<<r.quant<<" 个原样搬量化块）..."<<flush;
    bfile::write_bfile_stream(bf,r.head,sinks);
    r.tensor_count=sinks.size();
    r.file_size=fs::file_size(bf);
    if(opt.verbose) cout<<"完成\n";

    {
        json_libs::json_lib info;
        const string p=(fs::path(opt.out_dir)/"model_info.json").string();
        info.SetJsonWay(p);
        info.WriteJsonKey("hidden_size",to_string(r.head.hidden));
        info.WriteJsonKey("layer_count",to_string(r.head.layers));
        info.WriteJsonKey("mode",to_string(r.head.mode));
        info.WriteJsonKey("num_heads",to_string(r.head.num_heads));
        info.WriteJsonKey("num_kv_heads",to_string(r.head.num_kv_heads));
        info.WriteJsonKey("head_dim",to_string(r.head.head_dim));
        info.WriteJsonKey("intermediate_size",to_string(r.head.intermediate));
        info.WriteJsonKey("vocab_size",to_string(r.head.vocab_size));
        info.WriteJsonKey("rope_theta",to_string(r.rope_theta));
        info.WriteJsonKey("max_position_embeddings",to_string(r.max_position));
        info.WriteJsonKey("dtype",opt.quant?string("quant"):string(bfile::dtype_name(opt.out_dtype)));
        info.WriteJsonKey("add_bos_token",string("0"));
        info.WriteJsonKey("rope_style",string("adjacent_after_permute"));
        info.WriteJsonKey("source_arch",r.arch);
        info.WriteJsonKey("source_file",fs::path(opt.src).filename().string());
        // 把 GGUF 里的 general.name 记下来：run 时要靠它认 base/instruct
        if(const gguf::Meta* nm=g.meta("general.name"))
            if(!nm->s.empty()) info.WriteJsonKey("source_name",nm->s);
        // 特殊 token：从自带的词表里挑（Qwen 的 <|im_start|> 这类），挑到哪个写哪个
        if(const gguf::Meta* tk=g.meta("tokenizer.ggml.tokens")){
            auto pick=[&](const char* key,std::initializer_list<const char*> cands)->bool{
                for(const char* c:cands)
                    for(const string& t:tk->strs)
                        if(t==c){ info.WriteJsonKey(key,string(c)); return true; }
                return false;
            };
            pick("message_start",{"<|im_start|>","<|start_header_id|>"});
            pick("message_end",{"<|im_end|>","<|eot_id|>","<|end_of_turn|>"});
            pick("text_end",{"<|endoftext|>","<|end_of_text|>","</s>"});
        }
        r.eos_id=(int)meta_int(g,"tokenizer.ggml.eos_token_id",r.eos_id);
    }
    write_tokenizer(g,opt.out_dir,r,opt.verbose);
    return r;
}

}  // namespace ggufc
#endif  // GGUF_CONVERT_H
