#ifndef JSON_LIBS_TOKENIZER_H
#define JSON_LIBS_TOKENIZER_H


#include<algorithm>
#include<array>
#include<cstddef>
#include<cstdint>
#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<functional>
#include<queue>
#include<sstream>
#include<string>
#include<unordered_map>
#include<utility>
#include<vector>

#include"../../json_libs/file_util.h"
#include"../../json_libs/json.h"

namespace json_libs {

inline constexpr char kBosToken[]="<" "\xEF\xBD\x9C" "begin" "\xE2\x96\x81" "of" "\xE2\x96\x81" "sentence" "\xEF\xBD\x9C" ">";
inline constexpr char kEosToken[]="<" "\xEF\xBD\x9C" "end" "\xE2\x96\x81" "of" "\xE2\x96\x81" "sentence" "\xEF\xBD\x9C" ">";
inline constexpr char kUnkToken[]="<unk>";

inline constexpr char kWordEndToken[]="</w>";

namespace detail {


inline uint64_t pair_key(int32_t left,int32_t right){
    const uint64_t left_bits=static_cast<uint64_t>(static_cast<uint32_t>(left));
    const uint64_t right_bits=static_cast<uint64_t>(static_cast<uint32_t>(right));
    return (left_bits << 32) | right_bits;
}

inline int32_t key_left(uint64_t key){
    return static_cast<int32_t>(key >> 32);
}

inline int32_t key_right(uint64_t key){
    return static_cast<int32_t>(key & 0xFFFFFFFFull);
}

using json_libs::file_util::utf8_encode;
using json_libs::file_util::utf8_decode;

using json_libs::file_util::read_file;
using json_libs::file_util::write_file;
using json_libs::file_util::split_lines;


inline std::string& configured_vocab_path(){
    static std::string path;
    return path;
}

inline std::string& configured_merges_path(){
    static std::string path;
    return path;
}

using json_libs::file_util::file_readable;
using json_libs::file_util::resolve_relative_path;

struct ByteLevelMap {
    std::array<std::string,256> to_char;
    std::array<int32_t,68> to_byte;
    std::array<bool,256> printable;

    ByteLevelMap(){
        printable.fill(false);
        for(int b=33;b<=126;++b){
            printable[static_cast<size_t>(b)]=true;
        }
        for(int b=161;b<=172;++b){
            printable[static_cast<size_t>(b)]=true;
        }
        for(int b=174;b<=255;++b){
            printable[static_cast<size_t>(b)]=true;
        }
        to_byte.fill(-1);
        uint32_t next_codepoint=256;
        int32_t slot=0;
        for(int b=0;b < 256;++b){
            const size_t index=static_cast<size_t>(b);
            if(printable[index]){
                to_char[index]=utf8_encode(static_cast<uint32_t>(b));
            }else {
                to_char[index]=utf8_encode(next_codepoint);
                to_byte[static_cast<size_t>(slot)]=b;
                slot+=1;
                next_codepoint+=1;
            }
        }
    }
};

inline const ByteLevelMap& byte_level_map(){
    static const ByteLevelMap instance;
    return instance;
}

inline const std::array<std::string,256>& byte_level_table(){
    return byte_level_map().to_char;
}

inline int32_t byte_level_decode(uint32_t codepoint){
    const ByteLevelMap& map=byte_level_map();
    if(codepoint < 256u){
        if(map.printable[static_cast<size_t>(codepoint)]){
            return static_cast<int32_t>(codepoint);
        }
        return -1;
    }
    const uint32_t slot=codepoint - 256u;
    if(slot < map.to_byte.size()){
        return map.to_byte[static_cast<size_t>(slot)];
    }
    return -1;
}


inline bool is_space_codepoint(uint32_t codepoint){
    return codepoint==0x20u
        ||codepoint==0x09u
        ||codepoint==0x0Au
        ||codepoint==0x0Bu
        ||codepoint==0x0Cu
        ||codepoint==0x0Du
        ||codepoint==0x3000u;
}

inline bool is_punct_codepoint(uint32_t codepoint){
    if(codepoint>=0x21u&&codepoint<=0x2Fu) return true;
    if(codepoint>=0x3Au&&codepoint<=0x40u) return true;
    if(codepoint>=0x5Bu&&codepoint<=0x60u) return true;
    if(codepoint>=0x7Bu&&codepoint<=0x7Eu) return true;
    if(codepoint>=0x2010u&&codepoint<=0x2027u) return true;
    if(codepoint>=0x2030u&&codepoint<=0x205Eu) return true;
    if(codepoint>=0x3001u&&codepoint<=0x303Fu) return true;
    if(codepoint>=0xFF01u&&codepoint<=0xFF0Fu) return true;
    if(codepoint>=0xFF1Au&&codepoint<=0xFF20u) return true;
    if(codepoint>=0xFF3Bu&&codepoint<=0xFF40u) return true;
    if(codepoint>=0xFF5Bu&&codepoint<=0xFF65u) return true;
    return false;
}

inline std::vector<std::pair<size_t,size_t>> split_word_ranges(const std::string& text){
    std::vector<std::pair<size_t,size_t>> ranges;
    const size_t size=text.size();
    size_t index=0;
    while(index < size){
        uint32_t codepoint=0;
        size_t length=1;
        utf8_decode(text.data() + index,size - index,codepoint,length);

        if(is_space_codepoint(codepoint)){
            size_t next=index;
            while(next < size){
                uint32_t inner=0;
                size_t inner_length=1;
                utf8_decode(text.data() + next,size - next,inner,inner_length);
                if(!is_space_codepoint(inner)){
                    break;
                }
                next+=inner_length;
            }
            const size_t space_bytes=next - index;
            if(space_bytes==1&&text[index]==' '&&next < size){
                size_t stop=next;
                uint32_t after=0;
                size_t after_length=1;
                utf8_decode(text.data() + stop,size - stop,after,after_length);
                if(is_punct_codepoint(after)){
                    stop+=after_length;
                }else {
                    while(stop < size){
                        uint32_t inner=0;
                        size_t inner_length=1;
                        utf8_decode(text.data() + stop,size - stop,inner,inner_length);
                        if(is_space_codepoint(inner)||is_punct_codepoint(inner)){
                            break;
                        }
                        stop+=inner_length;
                    }
                }
                ranges.emplace_back(index,stop);
                index=stop;
            }else {
                ranges.emplace_back(index,next);
                index=next;
            }
            continue;
        }

        if(is_punct_codepoint(codepoint)){
            ranges.emplace_back(index,index + length);
            index+=length;
            continue;
        }

        size_t stop=index;
        while(stop < size){
            uint32_t inner=0;
            size_t inner_length=1;
            utf8_decode(text.data() + stop,size - stop,inner,inner_length);
            if(is_space_codepoint(inner)||is_punct_codepoint(inner)){
                break;
            }
            stop+=inner_length;
        }
        ranges.emplace_back(index,stop);
        index=stop;
    }
    return ranges;
}


}

inline void tokenizer_way(const std::string& vocab_path,const std::string& merges_path){
    detail::configured_vocab_path()=vocab_path;
    detail::configured_merges_path()=merges_path;
}

inline void tokenizer_way(const std::string& directory_path){
    if(directory_path.empty()){
        tokenizer_way(std::string(),std::string());
        return;
    }
    std::string directory=directory_path;
    if(directory.back()!='/'&&directory.back()!='\\'){
        directory.push_back('/');
    }
    tokenizer_way(directory + "vocab.json",directory + "merges.txt");
}

class Tokenizer {
public:
    Tokenizer();

    bool load(const std::string& vocab_path,const std::string& merges_path);

    bool load();

    std::vector<int32_t> encode(const std::string& text) const;

    std::string decode(const std::vector<int32_t>& ids) const;

    std::string token_to_piece(int32_t id) const;

    size_t vocab_size() const;

    int32_t bos_id() const;
    int32_t eos_id() const;
    int32_t unk_id() const;
    int32_t word_end_id() const;

    bool contains_token(const std::string& piece) const;

    int32_t add_special_token(const std::string& piece);

private:
    std::vector<int32_t> merge_word(const std::vector<int32_t>& symbols) const;

    int32_t find_special_token(const std::string& text,size_t from,size_t& position,size_t& length) const;

    int32_t piece_id(const std::string& piece) const;

    std::unordered_map<std::string,int32_t> piece_to_id_;
    std::vector<std::string> id_to_piece_;

    std::unordered_map<uint64_t,int32_t> merge_rank_;
    std::vector<int32_t> merge_result_id_;

    std::array<int32_t,256> byte_id_;

    int32_t word_end_id_;
    int32_t bos_id_;
    int32_t eos_id_;
    int32_t unk_id_;

    std::vector<std::string> special_pieces_;
    std::vector<int32_t> special_ids_;
};


inline Tokenizer::Tokenizer()
    :word_end_id_(-1),
      bos_id_(-1),
      eos_id_(-1),
      unk_id_(-1){
    byte_id_.fill(-1);
}

inline int32_t Tokenizer::piece_id(const std::string& piece) const {
    const auto it=piece_to_id_.find(piece);
    if(it==piece_to_id_.end()){
        return -1;
    }
    return it->second;
}

inline bool Tokenizer::contains_token(const std::string& piece) const {
    return piece_to_id_.find(piece)!=piece_to_id_.end();
}

inline int32_t Tokenizer::add_special_token(const std::string& piece){
    if(piece.empty()){
        return -1;
    }
    const int32_t id=piece_id(piece);
    if(id < 0){
        return -1;
    }
    for(const std::string& have:special_pieces_){
        if(have==piece){
            return id;
        }
    }
    special_pieces_.push_back(piece);
    special_ids_.push_back(id);
    return id;
}

inline bool Tokenizer::load(){
    const std::string& vocab_path=detail::configured_vocab_path();
    const std::string& merges_path=detail::configured_merges_path();
    if(vocab_path.empty()||merges_path.empty()){
        std::fprintf(stderr,"[tokenizer] 还没有规定路径，请先调用 tokenizer_way(路径)\n");
        return false;
    }
    return load(vocab_path,merges_path);
}

inline bool Tokenizer::load(const std::string& vocab_path,const std::string& merges_path){
    const std::string resolved_vocab=detail::resolve_relative_path(vocab_path);
    const std::string resolved_merges=detail::resolve_relative_path(merges_path);

    piece_to_id_.clear();
    id_to_piece_.clear();
    merge_rank_.clear();
    merge_result_id_.clear();
    special_pieces_.clear();
    special_ids_.clear();
    byte_id_.fill(-1);
    word_end_id_=-1;
    bos_id_=-1;
    eos_id_=-1;
    unk_id_=-1;

    std::string merges_text;
    if(!detail::read_file(resolved_merges,merges_text)){
        std::fprintf(stderr,"[tokenizer] 无法读取合并文件: %s\n",resolved_merges.c_str());
        return false;
    }

    json_libs::json_lib vocab;
    if(!vocab.SetJsonWay(resolved_vocab)){
        std::fprintf(stderr,"[tokenizer] 词表解析失败: %s\n",vocab.LastError().c_str());
        return false;
    }
    size_t entry_count=0;
    vocab.ForEachKey([&](const std::string& piece,const std::string& value)->bool {
        char* end=nullptr;
        const long id=std::strtol(value.c_str(),&end,10);
        if(end==value.c_str()||id < 0){
            return true;
        }
        const size_t index=static_cast<size_t>(id);
        if(index>=id_to_piece_.size()){
            id_to_piece_.resize(index + 1);
        }
        id_to_piece_[index]=piece;
        piece_to_id_.emplace(piece,static_cast<int32_t>(id));
        entry_count+=1;
        return true;
    });
    if(entry_count==0){
        std::fprintf(stderr,"[tokenizer] 词表为空\n");
        return false;
    }

    auto ensure_token=[this](const std::string& piece)->int32_t {
        const int32_t existing=piece_id(piece);
        if(existing>=0){
            return existing;
        }
        const int32_t id=static_cast<int32_t>(id_to_piece_.size());
        id_to_piece_.push_back(piece);
        piece_to_id_.emplace(piece,id);
        return id;
    };

    bos_id_=ensure_token(kBosToken);
    eos_id_=ensure_token(kEosToken);
    unk_id_=ensure_token(kUnkToken);
    word_end_id_=piece_id(kWordEndToken);

    const std::array<std::string,256>& table=detail::byte_level_table();
    for(size_t b=0; b < table.size(); ++b){
        const int32_t id=piece_id(table[b]);
        byte_id_[b]=(id>=0)?id:unk_id_;
    }

    const std::vector<std::string> lines=detail::split_lines(merges_text);
    for(const std::string& raw_line:lines){
        std::string line=raw_line;
        if(!line.empty()&&line.back()=='\r'){
            line.pop_back();
        }
        if(line.empty()||line[0]=='#'){
            continue;
        }
        const size_t space=line.find(' ');
        if(space==std::string::npos){
            continue;
        }
        const std::string left=line.substr(0,space);
        std::string right=line.substr(space + 1);
        if(right.empty()){
            continue;
        }
        const int32_t left_id=ensure_token(left);
        const int32_t right_id=ensure_token(right);
        const uint64_t key=detail::pair_key(left_id,right_id);
        if(merge_rank_.find(key)!=merge_rank_.end()){
            continue;
        }
        const int32_t merged_id=ensure_token(left + right);
        merge_rank_.emplace(key,static_cast<int32_t>(merge_result_id_.size()));
        merge_result_id_.push_back(merged_id);
    }

    special_pieces_.push_back(kBosToken);
    special_pieces_.push_back(kEosToken);
    special_pieces_.push_back(kUnkToken);
    std::sort(special_pieces_.begin(),special_pieces_.end(),
              [](const std::string& a,const std::string& b){
                  if(a.size()!=b.size()){
                      return a.size() > b.size();
                  }
                  return a < b;
              });
    special_ids_.clear();
    for(const std::string& piece:special_pieces_){
        special_ids_.push_back(ensure_token(piece));
    }

    return true;
}

inline int32_t Tokenizer::find_special_token(const std::string& text,size_t from,
                                             size_t& position,size_t& length) const {
    int32_t found=-1;
    size_t best_position=std::string::npos;
    size_t best_length=0;
    for(size_t i=0; i < special_pieces_.size(); ++i){
        const size_t at=text.find(special_pieces_[i],from);
        if(at==std::string::npos){
            continue;
        }
        const bool better=(found < 0)
            ||(at < best_position)
            ||(at==best_position&&special_pieces_[i].size() > best_length);
        if(better){
            found=static_cast<int32_t>(i);
            best_position=at;
            best_length=special_pieces_[i].size();
        }
    }
    if(found>=0){
        position=best_position;
        length=best_length;
    }
    return found;
}

inline std::vector<int32_t> Tokenizer::merge_word(const std::vector<int32_t>& symbols) const {
    const size_t size=symbols.size();
    if(size < 2||merge_rank_.empty()){
        return symbols;
    }

    struct Node {
        int32_t symbol;
        int32_t prev;
        int32_t next;
        uint32_t version;
        bool alive;
    };
    struct Candidate {
        int32_t rank;
        int32_t pos;
        uint32_t version;

        bool operator>(const Candidate& other) const {
            if(rank!=other.rank){
                return rank > other.rank;
            }
            return pos > other.pos;
        }
    };

    std::vector<Node> nodes(size);
    for(size_t i=0;i < size;++i){
        nodes[i].symbol=symbols[i];
        nodes[i].prev=static_cast<int32_t>(i) - 1;
        nodes[i].next=(i + 1 < size)?static_cast<int32_t>(i + 1):-1;
        nodes[i].version=0;
        nodes[i].alive=true;
    }

    typedef std::priority_queue<Candidate,std::vector<Candidate>,std::greater<Candidate>> CandidateQueue;
    CandidateQueue candidates;

    auto try_push=[&](int32_t pos){
        if(pos < 0||static_cast<size_t>(pos)>=size){
            return;
        }
        const int32_t next=nodes[static_cast<size_t>(pos)].next;
        if(next < 0){
            return;
        }
        const int32_t left=nodes[static_cast<size_t>(pos)].symbol;
        const int32_t right=nodes[static_cast<size_t>(next)].symbol;
        const auto it=merge_rank_.find(detail::pair_key(left,right));
        if(it==merge_rank_.end()){
            return;
        }
        Candidate candidate;
        candidate.rank=it->second;
        candidate.pos=pos;
        candidate.version=nodes[static_cast<size_t>(pos)].version;
        candidates.push(candidate);
    };

    for(size_t i=0;i + 1 < size;++i){
        try_push(static_cast<int32_t>(i));
    }

    while(!candidates.empty()){
        const Candidate top=candidates.top();
        candidates.pop();

        const size_t pos=static_cast<size_t>(top.pos);
        if(top.pos < 0||pos>=size){
            continue;
        }
        if(!nodes[pos].alive){
            continue;
        }
        if(nodes[pos].version!=top.version){
            continue;
        }
        const int32_t next=nodes[pos].next;
        if(next < 0){
            continue;
        }
        const auto it=merge_rank_.find(detail::pair_key(nodes[pos].symbol,nodes[static_cast<size_t>(next)].symbol));
        if(it==merge_rank_.end()||it->second!=top.rank){
            continue;
        }
        const int32_t merged_id=merge_result_id_[static_cast<size_t>(top.rank)];
        if(merged_id < 0){
            continue;
        }

        nodes[pos].symbol=merged_id;
        const size_t next_index=static_cast<size_t>(next);
        const int32_t after=nodes[next_index].next;
        nodes[next_index].alive=false;
        nodes[pos].next=after;
        if(after>=0){
            nodes[static_cast<size_t>(after)].prev=top.pos;
        }

        nodes[pos].version+=1;
        const int32_t prev=nodes[pos].prev;
        if(prev>=0){
            nodes[static_cast<size_t>(prev)].version+=1;
        }
        try_push(prev);
        try_push(top.pos);
    }

    std::vector<int32_t> result;
    result.reserve(size);
    for(int32_t index=0; index>=0&&static_cast<size_t>(index) < size;
         index=nodes[static_cast<size_t>(index)].next){
        if(nodes[static_cast<size_t>(index)].alive){
            result.push_back(nodes[static_cast<size_t>(index)].symbol);
        }
    }
    return result;
}

inline std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    if(text.empty()){
        return ids;
    }

    std::vector<int32_t> symbols;
    size_t cursor=0;
    while(cursor < text.size()){
        size_t special_position=0;
        size_t special_length=0;
        const int32_t special_index=find_special_token(text,cursor,special_position,special_length);
        const size_t segment_end=(special_index>=0)?special_position:text.size();

        if(segment_end > cursor){
            const std::string segment=text.substr(cursor,segment_end - cursor);
            const std::vector<std::pair<size_t,size_t>> ranges=detail::split_word_ranges(segment);
            for(const auto& range:ranges){
                if(range.second<=range.first){
                    continue;
                }
                symbols.clear();
                symbols.reserve(range.second - range.first + 1);
                for(size_t i=range.first;i < range.second;++i){
                    symbols.push_back(byte_id_[static_cast<unsigned char>(segment[i])]);
                }
                if(word_end_id_>=0){
                    symbols.push_back(word_end_id_);
                }
                const std::vector<int32_t> merged=merge_word(symbols);
                ids.insert(ids.end(),merged.begin(),merged.end());
            }
        }

        if(special_index < 0){
            break;
        }
        ids.push_back(special_ids_[static_cast<size_t>(special_index)]);
        cursor=special_position + special_length;
    }
    return ids;
}

inline std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string raw;
    for(int32_t id:ids){
        if(id < 0||static_cast<size_t>(id)>=id_to_piece_.size()){
            continue;
        }
        if(id==bos_id_||id==eos_id_||id==unk_id_){
            raw+=id_to_piece_[static_cast<size_t>(id)];
            continue;
        }
        std::string body=id_to_piece_[static_cast<size_t>(id)];
        const size_t marker_length=4;
        if(body.size()>=marker_length
            &&body.compare(body.size() - marker_length,marker_length,kWordEndToken)==0){
            body.resize(body.size() - marker_length);
        }
        size_t index=0;
        while(index < body.size()){
            uint32_t codepoint=0;
            size_t length=1;
            detail::utf8_decode(body.data() + index,body.size() - index,codepoint,length);
            const int32_t byte=detail::byte_level_decode(codepoint);
            if(byte>=0){
                raw.push_back(static_cast<char>(byte));
            }
            index+=length;
        }
    }
    return raw;
}

inline std::string Tokenizer::token_to_piece(int32_t id) const {
    if(id < 0||static_cast<size_t>(id)>=id_to_piece_.size()){
        return std::string();
    }
    return id_to_piece_[static_cast<size_t>(id)];
}

inline size_t Tokenizer::vocab_size() const {
    return piece_to_id_.size();
}

inline int32_t Tokenizer::bos_id() const {
    return bos_id_;
}

inline int32_t Tokenizer::eos_id() const {
    return eos_id_;
}

inline int32_t Tokenizer::unk_id() const {
    return unk_id_;
}

inline int32_t Tokenizer::word_end_id() const {
    return word_end_id_;
}

}

#endif