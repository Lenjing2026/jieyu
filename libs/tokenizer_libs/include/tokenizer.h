#ifndef JSON_LIBS_TOKENIZER_H
#define JSON_LIBS_TOKENIZER_H

// ============================================================
//  昆仑 字节级 BPE 分词器 —— 单头文件实现
//
//  * 纯 C++17，只依赖标准库，不做任何第三方依赖
//  * 全部函数均为 inline，可直接 #include 使用，无需链接静态库
//  * 词表文件格式与 HuggingFace 保持兼容：
//      vocab.json  形如 {"token": id, ...}
//      merges.txt  第一行 "#version: 0.2"，其后每行一条合并规则
//
//  编码算法：双向链表 + 优先队列 + 版本号惰性失效，复杂度 O(n log n)
//  JSON 解析：复用 libs/json_libs 的手写递归下降解析器（无第三方依赖）
// ============================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// 读 vocab.json 需要 JSON 解析：直接复用 libs/json_libs 下的公用头，
// 这里不再自带一份 JSON 解析器（file_util 提供文件读写与相对路径解析）
#include "../../json_libs/file_util.h"
#include "../../json_libs/json.h"

namespace json_libs {

// ------------------------------------------------------------
// 特殊 token 文本（UTF-8，用十六进制转义拼接，避免源码编码差异）
//   U+FF5C 全角竖线 = EF BD 9C
//   U+2581 下划线   = E2 96 81
// ------------------------------------------------------------
inline constexpr char kBosToken[] = "<" "\xEF\xBD\x9C" "begin" "\xE2\x96\x81" "of" "\xE2\x96\x81" "sentence" "\xEF\xBD\x9C" ">";
inline constexpr char kEosToken[] = "<" "\xEF\xBD\x9C" "end" "\xE2\x96\x81" "of" "\xE2\x96\x81" "sentence" "\xEF\xBD\x9C" ">";
inline constexpr char kUnkToken[] = "<unk>";

// 词尾标记（subword-nmt / GPT-2 风格）
inline constexpr char kWordEndToken[] = "</w>";

// ============================================================
//  detail：UTF-8 编解码、文件读写、字节级映射、预分词、简化 JSON
// ============================================================
namespace detail {

// ------------------------------------------------------------
// 符号对打包工具
// ------------------------------------------------------------

// 把 (左符号 id, 右符号 id) 打包成一个 64 位 key
inline uint64_t pair_key(int32_t left, int32_t right) {
    const uint64_t left_bits = static_cast<uint64_t>(static_cast<uint32_t>(left));
    const uint64_t right_bits = static_cast<uint64_t>(static_cast<uint32_t>(right));
    return (left_bits << 32) | right_bits;
}

// 从 key 中取回左符号 id
inline int32_t key_left(uint64_t key) {
    return static_cast<int32_t>(key >> 32);
}

// 从 key 中取回右符号 id
inline int32_t key_right(uint64_t key) {
    return static_cast<int32_t>(key & 0xFFFFFFFFull);
}

// ------------------------------------------------------------
// UTF-8：直接复用 json_libs::file_util 的实现（见 libs/json_libs/file_util.h）
// ------------------------------------------------------------
using json_libs::file_util::utf8_encode;
using json_libs::file_util::utf8_decode;

// ------------------------------------------------------------
// 文件读写：直接复用 json_libs::file_util 的实现
// ------------------------------------------------------------
using json_libs::file_util::read_file;
using json_libs::file_util::write_file;
using json_libs::file_util::split_lines;

// ------------------------------------------------------------
// 路径配置（tokenizer_way 的落地存储）
// ------------------------------------------------------------

// 分词器数据路径（inline 函数的局部静态变量在所有 TU 中共用同一份）
inline std::string& configured_vocab_path() {
    static std::string path;
    return path;
}

inline std::string& configured_merges_path() {
    static std::string path;
    return path;
}

// 文件是否可打开 / 相对路径解析：直接复用 json_libs::file_util 的实现
using json_libs::file_util::file_readable;
using json_libs::file_util::resolve_relative_path;

// ------------------------------------------------------------
// 字节级映射表（GPT-2 byte-level 规则）
//   可见字节直接映射为自身码点，其余字节按顺序映射到 256 起的码点
// ------------------------------------------------------------
struct ByteLevelMap {
    std::array<std::string, 256> to_char;  // 字节 -> 字符
    std::array<int32_t, 68> to_byte;       // (码点 - 256) -> 字节
    std::array<bool, 256> printable;       // 该字节是否直接映射为自身码点

    ByteLevelMap() {
        printable.fill(false);
        for (int b = 33; b <= 126; ++b) {
            printable[static_cast<size_t>(b)] = true;
        }
        for (int b = 161; b <= 172; ++b) {
            printable[static_cast<size_t>(b)] = true;
        }
        for (int b = 174; b <= 255; ++b) {
            printable[static_cast<size_t>(b)] = true;
        }
        to_byte.fill(-1);
        uint32_t next_codepoint = 256;
        int32_t slot = 0;
        for (int b = 0; b < 256; ++b) {
            const size_t index = static_cast<size_t>(b);
            if (printable[index]) {
                to_char[index] = utf8_encode(static_cast<uint32_t>(b));
            } else {
                to_char[index] = utf8_encode(next_codepoint);
                to_byte[static_cast<size_t>(slot)] = b;
                slot += 1;
                next_codepoint += 1;
            }
        }
    }
};

// 全局唯一的映射表实例（inline 函数的局部静态变量在所有 TU 中共享）
inline const ByteLevelMap& byte_level_map() {
    static const ByteLevelMap instance;
    return instance;
}

// 字节级映射表：256 个字节 -> “可打印”的 UTF-8 字符
inline const std::array<std::string, 256>& byte_level_table() {
    return byte_level_map().to_char;
}

// 反向映射：把一个 Unicode 码点还原成字节，失败返回 -1
inline int32_t byte_level_decode(uint32_t codepoint) {
    const ByteLevelMap& map = byte_level_map();
    if (codepoint < 256u) {
        // 注意：这里必须查可打印表，不能用码点的 UTF-8 字节长度判断
        // （字节 161..172 / 174..255 直接映射为自身码点，但其 UTF-8 编码是 2 字节）
        if (map.printable[static_cast<size_t>(codepoint)]) {
            return static_cast<int32_t>(codepoint);
        }
        return -1;
    }
    const uint32_t slot = codepoint - 256u;
    if (slot < map.to_byte.size()) {
        return map.to_byte[static_cast<size_t>(slot)];
    }
    return -1;
}

// ------------------------------------------------------------
// 字符分类（按 Unicode 码点）
// ------------------------------------------------------------

// 是否为空白字符
inline bool is_space_codepoint(uint32_t codepoint) {
    return codepoint == 0x20u    // 空格
        || codepoint == 0x09u    // 制表符
        || codepoint == 0x0Au    // 换行
        || codepoint == 0x0Bu    // 垂直制表
        || codepoint == 0x0Cu    // 换页
        || codepoint == 0x0Du    // 回车
        || codepoint == 0x3000u;  // 全角空格
}

// 是否为标点字符（覆盖 ASCII 与常用中英文标点）
inline bool is_punct_codepoint(uint32_t codepoint) {
    if (codepoint >= 0x21u && codepoint <= 0x2Fu) return true;       // !"#$%&'()*+,-./
    if (codepoint >= 0x3Au && codepoint <= 0x40u) return true;       // :;<=>?@
    if (codepoint >= 0x5Bu && codepoint <= 0x60u) return true;       // [\]^_`
    if (codepoint >= 0x7Bu && codepoint <= 0x7Eu) return true;       // {|}~
    if (codepoint >= 0x2010u && codepoint <= 0x2027u) return true;   // 常用西文标点
    if (codepoint >= 0x2030u && codepoint <= 0x205Eu) return true;   // 常用西文标点
    if (codepoint >= 0x3001u && codepoint <= 0x303Fu) return true;   // 中文标点
    if (codepoint >= 0xFF01u && codepoint <= 0xFF0Fu) return true;   // 全角标点
    if (codepoint >= 0xFF1Au && codepoint <= 0xFF20u) return true;   // 全角标点
    if (codepoint >= 0xFF3Bu && codepoint <= 0xFF40u) return true;   // 全角标点
    if (codepoint >= 0xFF5Bu && codepoint <= 0xFF65u) return true;   // 全角标点
    return false;
}

// ------------------------------------------------------------
// 预分词
//   1. 连续空白整体成词；若恰好是 1 个空格且其后还有内容，则该空格并入后面的词
//   2. 每个标点字符单独成词
//   3. 其余字符（字母、数字、汉字等）连续成词
//   返回每个词的字节区间 [begin, end)
// ------------------------------------------------------------
inline std::vector<std::pair<size_t, size_t>> split_word_ranges(const std::string& text) {
    std::vector<std::pair<size_t, size_t>> ranges;
    const size_t size = text.size();
    size_t index = 0;
    while (index < size) {
        uint32_t codepoint = 0;
        size_t length = 1;
        utf8_decode(text.data() + index, size - index, codepoint, length);

        if (is_space_codepoint(codepoint)) {
            size_t next = index;
            while (next < size) {
                uint32_t inner = 0;
                size_t inner_length = 1;
                utf8_decode(text.data() + next, size - next, inner, inner_length);
                if (!is_space_codepoint(inner)) {
                    break;
                }
                next += inner_length;
            }
            const size_t space_bytes = next - index;
            if (space_bytes == 1 && text[index] == ' ' && next < size) {
                // 单个空格：并入紧随其后的那个词
                size_t stop = next;
                uint32_t after = 0;
                size_t after_length = 1;
                utf8_decode(text.data() + stop, size - stop, after, after_length);
                if (is_punct_codepoint(after)) {
                    stop += after_length;
                } else {
                    while (stop < size) {
                        uint32_t inner = 0;
                        size_t inner_length = 1;
                        utf8_decode(text.data() + stop, size - stop, inner, inner_length);
                        if (is_space_codepoint(inner) || is_punct_codepoint(inner)) {
                            break;
                        }
                        stop += inner_length;
                    }
                }
                ranges.emplace_back(index, stop);
                index = stop;
            } else {
                ranges.emplace_back(index, next);
                index = next;
            }
            continue;
        }

        if (is_punct_codepoint(codepoint)) {
            ranges.emplace_back(index, index + length);
            index += length;
            continue;
        }

        size_t stop = index;
        while (stop < size) {
            uint32_t inner = 0;
            size_t inner_length = 1;
            utf8_decode(text.data() + stop, size - stop, inner, inner_length);
            if (is_space_codepoint(inner) || is_punct_codepoint(inner)) {
                break;
            }
            stop += inner_length;
        }
        ranges.emplace_back(index, stop);
        index = stop;
    }
    return ranges;
}

// 说明：本文件原先自带的 JSON 解析器（JsonValue / JsonParser / parse_vocab_json）
// 已删除，统一复用 libs/json_libs/json.h；Tokenizer::load 通过 json_libs::json_lib 读取词表。

}  // namespace detail

// ------------------------------------------------------------
// 分词器数据路径接口：tokenizer_way(路径)
//
//   tokenizer_way("_file/BPE");                        // 目录：自动补 vocab.json / merges.txt
//   tokenizer_way("_file/BPE/vocab.json",
//                 "_file/BPE/merges.txt");             // 显式指定两个文件
//   tokenizer_way("");                                 // 清空
//
//   路径允许写成相对路径，查找顺序为：
//     原样 -> ../ -> ../../ -> ../../../
//   所以无论从仓库根目录、build/ 还是 libs/tokenizer/build/ 运行，
//   都能定位到同一份数据。
//
//   规定好之后，用 tokenizer.load()（无参）加载即可。
// ------------------------------------------------------------
inline void tokenizer_way(const std::string& vocab_path, const std::string& merges_path) {
    detail::configured_vocab_path() = vocab_path;
    detail::configured_merges_path() = merges_path;
}

// 目录形式：自动拼接 vocab.json 与 merges.txt
inline void tokenizer_way(const std::string& directory_path) {
    if (directory_path.empty()) {
        tokenizer_way(std::string(), std::string());
        return;
    }
    std::string directory = directory_path;
    if (directory.back() != '/' && directory.back() != '\\') {
        directory.push_back('/');
    }
    tokenizer_way(directory + "vocab.json", directory + "merges.txt");
}

// 查询当前规定的词表路径（未规定时为空串）
inline const std::string& tokenizer_vocab_path() {
    return detail::configured_vocab_path();
}

// 查询当前规定的合并规则路径（未规定时为空串）
inline const std::string& tokenizer_merges_path() {
    return detail::configured_merges_path();
}

// ============================================================
//  Tokenizer
// ============================================================
class Tokenizer {
public:
    Tokenizer();

    // 从 vocab.json 与 merges.txt 加载词表与合并规则（支持相对路径）
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // 使用 tokenizer_way() 规定的路径加载
    bool load();

    // 编码：文本 -> token id 序列（文本中出现的特殊 token 会被直接识别）
    std::vector<int32_t> encode(const std::string& text) const;

    // 解码：token id 序列 -> 文本
    std::string decode(const std::vector<int32_t>& ids) const;

    // token id -> 原始 piece（HF 字节级风格，空格映射为 U+0120，词尾带 </w>）
    std::string token_to_piece(int32_t id) const;

    // 词表大小
    size_t vocab_size() const;

    // 特殊 token 的 id，不存在时返回 -1
    int32_t bos_id() const;
    int32_t eos_id() const;
    int32_t unk_id() const;
    int32_t word_end_id() const;

    // 查询某个 piece 是否在词表中
    bool contains_token(const std::string& piece) const;

    // 把模型自带的特殊 token（<|im_start|> 这种）登记进识别表，以后 encode 会直接认出它
    // 返回它的 id；词表里没有这个 piece 就返回 -1
    int32_t add_special_token(const std::string& piece);

private:
    // 在一个词内做 BPE 合并：双向链表 + 优先队列，复杂度 O(n log n)
    std::vector<int32_t> merge_word(const std::vector<int32_t>& symbols) const;

    // 在 text 的 [from, 末尾) 中寻找最早出现的特殊 token
    // 返回特殊 token 在 special_pieces_ 中的下标，未找到返回 -1
    int32_t find_special_token(const std::string& text, size_t from, size_t& position, size_t& length) const;

    // 查询 piece 的 id，不存在返回 -1
    int32_t piece_id(const std::string& piece) const;

    std::unordered_map<std::string, int32_t> piece_to_id_;  // piece -> id
    std::vector<std::string> id_to_piece_;                  // id -> piece

    std::unordered_map<uint64_t, int32_t> merge_rank_;  // 符号对 -> 合并优先级
    std::vector<int32_t> merge_result_id_;              // 合并优先级 -> 合并后的 id

    std::array<int32_t, 256> byte_id_;  // 单个字节字符对应的 id

    int32_t word_end_id_;
    int32_t bos_id_;
    int32_t eos_id_;
    int32_t unk_id_;

    std::vector<std::string> special_pieces_;  // 需要特殊识别的 token（按长度降序）
    std::vector<int32_t> special_ids_;
};

// ============================================================
//  Tokenizer 实现
// ============================================================

inline Tokenizer::Tokenizer()
    : word_end_id_(-1),
      bos_id_(-1),
      eos_id_(-1),
      unk_id_(-1) {
    byte_id_.fill(-1);
}

inline int32_t Tokenizer::piece_id(const std::string& piece) const {
    const auto it = piece_to_id_.find(piece);
    if (it == piece_to_id_.end()) {
        return -1;
    }
    return it->second;
}

inline bool Tokenizer::contains_token(const std::string& piece) const {
    return piece_to_id_.find(piece) != piece_to_id_.end();
}

inline int32_t Tokenizer::add_special_token(const std::string& piece) {
    if (piece.empty()) {
        return -1;
    }
    const int32_t id = piece_id(piece);
    if (id < 0) {
        return -1;  // 词表里没这个 piece，认不出来
    }
    for (const std::string& have : special_pieces_) {
        if (have == piece) {
            return id;  // 已经登记过
        }
    }
    special_pieces_.push_back(piece);
    special_ids_.push_back(id);
    return id;
}

// 使用 tokenizer_way() 规定的路径加载
inline bool Tokenizer::load() {
    const std::string& vocab_path = detail::configured_vocab_path();
    const std::string& merges_path = detail::configured_merges_path();
    if (vocab_path.empty() || merges_path.empty()) {
        std::fprintf(stderr, "[tokenizer] 还没有规定路径，请先调用 tokenizer_way(路径)\n");
        return false;
    }
    return load(vocab_path, merges_path);
}

inline bool Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // 支持相对路径：原样找不到时自动逐级向上查找（../、../../、../../../）
    const std::string resolved_vocab = detail::resolve_relative_path(vocab_path);
    const std::string resolved_merges = detail::resolve_relative_path(merges_path);

    // 先重置全部状态
    piece_to_id_.clear();
    id_to_piece_.clear();
    merge_rank_.clear();
    merge_result_id_.clear();
    special_pieces_.clear();
    special_ids_.clear();
    byte_id_.fill(-1);
    word_end_id_ = -1;
    bos_id_ = -1;
    eos_id_ = -1;
    unk_id_ = -1;

    std::string merges_text;
    if (!detail::read_file(resolved_merges, merges_text)) {
        std::fprintf(stderr, "[tokenizer] 无法读取合并文件: %s\n", resolved_merges.c_str());
        return false;
    }

    // ---------- 1. 加载 vocab.json（复用 libs/json_libs 的 json_libs::json_lib 类） ----------
    json_libs::json_lib vocab;
    if (!vocab.SetJsonWay(resolved_vocab)) {
        std::fprintf(stderr, "[tokenizer] 词表解析失败: %s\n", vocab.LastError().c_str());
        return false;
    }
    size_t entry_count = 0;
    vocab.ForEachKey([&](const std::string& piece, const std::string& value) -> bool {
        char* end = nullptr;
        const long id = std::strtol(value.c_str(), &end, 10);
        if (end == value.c_str() || id < 0) {
            return true;  // 值不是合法 id，跳过
        }
        const size_t index = static_cast<size_t>(id);
        if (index >= id_to_piece_.size()) {
            id_to_piece_.resize(index + 1);
        }
        id_to_piece_[index] = piece;
        piece_to_id_.emplace(piece, static_cast<int32_t>(id));
        entry_count += 1;
        return true;
    });
    if (entry_count == 0) {
        std::fprintf(stderr, "[tokenizer] 词表为空\n");
        return false;
    }

    // ---------- 2. 补齐特殊 token 与词尾标记 ----------
    auto ensure_token = [this](const std::string& piece) -> int32_t {
        const int32_t existing = piece_id(piece);
        if (existing >= 0) {
            return existing;
        }
        const int32_t id = static_cast<int32_t>(id_to_piece_.size());
        id_to_piece_.push_back(piece);
        piece_to_id_.emplace(piece, id);
        return id;
    };

    bos_id_ = ensure_token(kBosToken);
    eos_id_ = ensure_token(kEosToken);
    unk_id_ = ensure_token(kUnkToken);
    word_end_id_ = piece_id(kWordEndToken);

    // ---------- 3. 建立字节 -> id 的快查表 ----------
    const std::array<std::string, 256>& table = detail::byte_level_table();
    for (size_t b = 0; b < table.size(); ++b) {
        const int32_t id = piece_id(table[b]);
        byte_id_[b] = (id >= 0) ? id : unk_id_;
    }

    // ---------- 4. 加载 merges.txt ----------
    const std::vector<std::string> lines = detail::split_lines(merges_text);
    for (const std::string& raw_line : lines) {
        std::string line = raw_line;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;  // 跳过空行与 "#version: 0.2" 头
        }
        const size_t space = line.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const std::string left = line.substr(0, space);
        std::string right = line.substr(space + 1);
        if (right.empty()) {
            continue;
        }
        const int32_t left_id = ensure_token(left);
        const int32_t right_id = ensure_token(right);
        const uint64_t key = detail::pair_key(left_id, right_id);
        if (merge_rank_.find(key) != merge_rank_.end()) {
            continue;  // 保留首次出现（优先级最高）
        }
        const int32_t merged_id = ensure_token(left + right);
        merge_rank_.emplace(key, static_cast<int32_t>(merge_result_id_.size()));
        merge_result_id_.push_back(merged_id);
    }

    // ---------- 5. 特殊 token 识别表（按长度降序，长优先） ----------
    special_pieces_.push_back(kBosToken);
    special_pieces_.push_back(kEosToken);
    special_pieces_.push_back(kUnkToken);
    std::sort(special_pieces_.begin(), special_pieces_.end(),
              [](const std::string& a, const std::string& b) {
                  if (a.size() != b.size()) {
                      return a.size() > b.size();
                  }
                  return a < b;
              });
    special_ids_.clear();
    for (const std::string& piece : special_pieces_) {
        special_ids_.push_back(ensure_token(piece));
    }

    return true;
}

inline int32_t Tokenizer::find_special_token(const std::string& text, size_t from,
                                             size_t& position, size_t& length) const {
    int32_t found = -1;
    size_t best_position = std::string::npos;
    size_t best_length = 0;
    for (size_t i = 0; i < special_pieces_.size(); ++i) {
        const size_t at = text.find(special_pieces_[i], from);
        if (at == std::string::npos) {
            continue;
        }
        const bool better = (found < 0)
            || (at < best_position)
            || (at == best_position && special_pieces_[i].size() > best_length);
        if (better) {
            found = static_cast<int32_t>(i);
            best_position = at;
            best_length = special_pieces_[i].size();
        }
    }
    if (found >= 0) {
        position = best_position;
        length = best_length;
    }
    return found;
}

inline std::vector<int32_t> Tokenizer::merge_word(const std::vector<int32_t>& symbols) const {
    const size_t size = symbols.size();
    if (size < 2 || merge_rank_.empty()) {
        return symbols;
    }

    // 双向链表节点（用数组模拟，避免指针开销）
    struct Node {
        int32_t symbol;
        int32_t prev;
        int32_t next;
        uint32_t version;  // 惰性失效版本号
        bool alive;
    };
    // 优先队列候选：按 (合并优先级, 位置) 升序出队
    struct Candidate {
        int32_t rank;
        int32_t pos;
        uint32_t version;

        bool operator>(const Candidate& other) const {
            if (rank != other.rank) {
                return rank > other.rank;
            }
            return pos > other.pos;
        }
    };

    std::vector<Node> nodes(size);
    for (size_t i = 0; i < size; ++i) {
        nodes[i].symbol = symbols[i];
        nodes[i].prev = static_cast<int32_t>(i) - 1;
        nodes[i].next = (i + 1 < size) ? static_cast<int32_t>(i + 1) : -1;
        nodes[i].version = 0;
        nodes[i].alive = true;
    }

    typedef std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> CandidateQueue;
    CandidateQueue candidates;

    auto try_push = [&](int32_t pos) {
        if (pos < 0 || static_cast<size_t>(pos) >= size) {
            return;
        }
        const int32_t next = nodes[static_cast<size_t>(pos)].next;
        if (next < 0) {
            return;
        }
        const int32_t left = nodes[static_cast<size_t>(pos)].symbol;
        const int32_t right = nodes[static_cast<size_t>(next)].symbol;
        const auto it = merge_rank_.find(detail::pair_key(left, right));
        if (it == merge_rank_.end()) {
            return;
        }
        Candidate candidate;
        candidate.rank = it->second;
        candidate.pos = pos;
        candidate.version = nodes[static_cast<size_t>(pos)].version;
        candidates.push(candidate);
    };

    for (size_t i = 0; i + 1 < size; ++i) {
        try_push(static_cast<int32_t>(i));
    }

    while (!candidates.empty()) {
        const Candidate top = candidates.top();
        candidates.pop();

        const size_t pos = static_cast<size_t>(top.pos);
        if (top.pos < 0 || pos >= size) {
            continue;
        }
        if (!nodes[pos].alive) {
            continue;
        }
        if (nodes[pos].version != top.version) {
            continue;  // 已被其它合并作废
        }
        const int32_t next = nodes[pos].next;
        if (next < 0) {
            continue;
        }
        const auto it = merge_rank_.find(detail::pair_key(nodes[pos].symbol, nodes[static_cast<size_t>(next)].symbol));
        if (it == merge_rank_.end() || it->second != top.rank) {
            continue;  // 当前已不是该符号对
        }
        const int32_t merged_id = merge_result_id_[static_cast<size_t>(top.rank)];
        if (merged_id < 0) {
            continue;  // 词表中没有合并结果，跳过
        }

        // 执行合并：把 next 节点并入 pos 节点
        nodes[pos].symbol = merged_id;
        const size_t next_index = static_cast<size_t>(next);
        const int32_t after = nodes[next_index].next;
        nodes[next_index].alive = false;
        nodes[pos].next = after;
        if (after >= 0) {
            nodes[static_cast<size_t>(after)].prev = top.pos;
        }

        // 使 pos 与 pos 左邻居的旧候选失效
        nodes[pos].version += 1;
        const int32_t prev = nodes[pos].prev;
        if (prev >= 0) {
            nodes[static_cast<size_t>(prev)].version += 1;
        }
        try_push(prev);
        try_push(top.pos);
    }

    std::vector<int32_t> result;
    result.reserve(size);
    for (int32_t index = 0; index >= 0 && static_cast<size_t>(index) < size;
         index = nodes[static_cast<size_t>(index)].next) {
        if (nodes[static_cast<size_t>(index)].alive) {
            result.push_back(nodes[static_cast<size_t>(index)].symbol);
        }
    }
    return result;
}

inline std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    if (text.empty()) {
        return ids;
    }

    std::vector<int32_t> symbols;
    size_t cursor = 0;
    while (cursor < text.size()) {
        // 先找出最早的（最长的）特殊 token
        size_t special_position = 0;
        size_t special_length = 0;
        const int32_t special_index = find_special_token(text, cursor, special_position, special_length);
        const size_t segment_end = (special_index >= 0) ? special_position : text.size();

        if (segment_end > cursor) {
            const std::string segment = text.substr(cursor, segment_end - cursor);
            const std::vector<std::pair<size_t, size_t>> ranges = detail::split_word_ranges(segment);
            for (const auto& range : ranges) {
                if (range.second <= range.first) {
                    continue;
                }
                symbols.clear();
                symbols.reserve(range.second - range.first + 1);
                for (size_t i = range.first; i < range.second; ++i) {
                    symbols.push_back(byte_id_[static_cast<unsigned char>(segment[i])]);
                }
                if (word_end_id_ >= 0) {
                    symbols.push_back(word_end_id_);
                }
                const std::vector<int32_t> merged = merge_word(symbols);
                ids.insert(ids.end(), merged.begin(), merged.end());
            }
        }

        if (special_index < 0) {
            break;
        }
        ids.push_back(special_ids_[static_cast<size_t>(special_index)]);
        cursor = special_position + special_length;
    }
    return ids;
}

inline std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string raw;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_piece_.size()) {
            continue;
        }
        // 特殊 token 直接输出原文
        if (id == bos_id_ || id == eos_id_ || id == unk_id_) {
            raw += id_to_piece_[static_cast<size_t>(id)];
            continue;
        }
        std::string body = id_to_piece_[static_cast<size_t>(id)];
        // 去掉词尾标记
        const size_t marker_length = 4;  // "</w>"
        if (body.size() >= marker_length
            && body.compare(body.size() - marker_length, marker_length, kWordEndToken) == 0) {
            body.resize(body.size() - marker_length);
        }
        // 逐个码点还原成原始字节
        size_t index = 0;
        while (index < body.size()) {
            uint32_t codepoint = 0;
            size_t length = 1;
            detail::utf8_decode(body.data() + index, body.size() - index, codepoint, length);
            const int32_t byte = detail::byte_level_decode(codepoint);
            if (byte >= 0) {
                raw.push_back(static_cast<char>(byte));
            }
            index += length;
        }
    }
    return raw;
}

inline std::string Tokenizer::token_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_piece_.size()) {
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

}  // namespace json_libs

#endif  // JSON_LIBS_TOKENIZER_H
