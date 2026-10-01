#ifndef JSON_LIBS_TOKENIZER_TRAINER_H
#define JSON_LIBS_TOKENIZER_TRAINER_H

// ============================================================
//  昆仑 BPE 训练器 —— 单头文件实现
//
//  * 纯 C++17，只依赖标准库，全部函数 inline
//  * 已包含 tokenizer.h（只需 #include "trainer.h" 即可同时拿到分词器）
//
//  训练流程：
//    1. 读取语料（每行一句）
//    2. 按 空白 + 标点 切词，词尾追加 </w>
//    3. 迭代合并最高频的符号对，直到词表满或最高频 < 2
//    4. 输出 HuggingFace 兼容的 vocab.json 与 merges.txt
//
//  词表 id 布局：
//    0   .. 255 : 256 个字节字符
//    256        : 词尾标记 </w>
//    257        : BOS
//    258        : EOS
//    259        : UNK
//    260  ..    : 迭代合并产生的新 token
// ============================================================

#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace json_libs {

namespace detail {

// 基础词表布局常量
inline constexpr int32_t kBaseByteVocabSize = 256;  // 字节字符数量
inline constexpr int32_t kBaseWordEndId = 256;      // 词尾标记 </w>
inline constexpr int32_t kBaseBosId = 257;          // 句首
inline constexpr int32_t kBaseEosId = 258;          // 句尾
inline constexpr int32_t kBaseUnkId = 259;          // 未知
inline constexpr size_t kBaseVocabSize = 260;       // 基础词表大小（含特殊 token）

static_assert(kBaseWordEndId == kBaseByteVocabSize, "词尾标记紧跟字节字符之后");
static_assert(kBaseBosId == 257 && kBaseEosId == 258 && kBaseUnkId == 259, "特殊 token id 布局错误");
static_assert(kBaseVocabSize == 260, "基础词表大小应为 260");

// JSON 字符串转义
inline std::string json_escape(const std::string& text) {
    std::string out;
    out.push_back('"');
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20u) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(c));
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

}  // namespace detail

// ============================================================
//  BPETrainer
// ============================================================
class BPETrainer {
public:
    explicit BPETrainer(int target_vocab_size = 50000);

    // 读取语料文件（每行一句，UTF-8）
    bool load_corpus(const std::string& corpus_path);

    // 训练：构建基础词表并迭代合并
    void train();

    // 保存 vocab.json 与 merges.txt 到目录（目录不存在会自动创建）
    bool save(const std::string& output_dir) const;

    // 词表大小
    size_t vocab_size() const;

    // 合并规则条数
    size_t merge_count() const;

    // 调试用：id -> piece
    const std::vector<std::string>& pieces() const;

private:
    // 一个唯一词：符号序列 + 出现次数
    struct WordEntry {
        std::vector<int32_t> symbols;
        uint64_t count;
    };

    // 优先队列候选（大顶堆：频次高的先出队）
    struct PairCandidate {
        uint64_t count;
        uint64_t key;
        uint64_t version;

        bool operator<(const PairCandidate& other) const {
            if (count != other.count) {
                return count < other.count;
            }
            return key < other.key;
        }
    };

    void build_base_vocab();
    void build_word_entries();
    void merge_pair(uint64_t key, int32_t left, int32_t right);
    void add_pair_count(int64_t delta, uint64_t key);

    int32_t ensure_token(const std::string& piece);

    static void collect_pairs(const std::vector<int32_t>& symbols, std::vector<uint64_t>& out);
    static bool contains_pair(const std::vector<int32_t>& symbols, int32_t left, int32_t right);

    std::string make_vocab_json() const;
    std::string make_merges_text() const;

    int target_vocab_size_;
    bool trained_;

    std::vector<std::string> corpus_lines_;
    std::unordered_map<std::string, uint64_t> word_counts_;  // 原始字节串 -> 词频
    std::vector<WordEntry> words_;                           // 唯一词表

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;

    std::unordered_map<uint64_t, uint64_t> pair_counts_;              // 符号对 -> 频次
    std::unordered_map<uint64_t, uint64_t> pair_versions_;            // 惰性失效版本号
    std::unordered_map<uint64_t, std::vector<uint32_t>> pair_words_;  // 倒排：符号对 -> 包含它的词
    std::priority_queue<PairCandidate> heap_;
};

// ============================================================
//  BPETrainer 实现
// ============================================================

inline BPETrainer::BPETrainer(int target_vocab_size)
    : target_vocab_size_(target_vocab_size < static_cast<int>(detail::kBaseVocabSize)
                             ? static_cast<int>(detail::kBaseVocabSize)
                             : target_vocab_size),
      trained_(false) {
}

inline bool BPETrainer::load_corpus(const std::string& corpus_path) {
    std::string content;
    if (!detail::read_file(corpus_path, content)) {
        std::fprintf(stderr, "[trainer] 无法读取语料文件: %s\n", corpus_path.c_str());
        return false;
    }
    corpus_lines_ = detail::split_lines(content);
    return true;
}

inline int32_t BPETrainer::ensure_token(const std::string& piece) {
    const auto it = piece_to_id_.find(piece);
    if (it != piece_to_id_.end()) {
        return it->second;
    }
    const int32_t id = static_cast<int32_t>(id_to_piece_.size());
    id_to_piece_.push_back(piece);
    piece_to_id_.emplace(piece, id);
    return id;
}

inline void BPETrainer::build_base_vocab() {
    id_to_piece_.clear();
    piece_to_id_.clear();
    merges_.clear();
    pair_counts_.clear();
    pair_versions_.clear();
    pair_words_.clear();
    heap_ = std::priority_queue<PairCandidate>();

    // 256 个字节字符，id 就等于字节值
    const std::array<std::string, 256>& table = detail::byte_level_table();
    id_to_piece_.reserve(detail::kBaseVocabSize);
    for (int32_t b = 0; b < detail::kBaseByteVocabSize; ++b) {
        id_to_piece_.push_back(table[static_cast<size_t>(b)]);
        piece_to_id_.emplace(table[static_cast<size_t>(b)], b);
    }

    // 词尾标记与特殊 token
    id_to_piece_.push_back(kWordEndToken);
    piece_to_id_.emplace(kWordEndToken, detail::kBaseWordEndId);

    ensure_token(kBosToken);
    ensure_token(kEosToken);
    ensure_token(kUnkToken);
}

inline void BPETrainer::build_word_entries() {
    word_counts_.clear();
    for (const std::string& line : corpus_lines_) {
        const std::vector<std::pair<size_t, size_t>> ranges = detail::split_word_ranges(line);
        for (const auto& range : ranges) {
            if (range.second <= range.first) {
                continue;
            }
            word_counts_[line.substr(range.first, range.second - range.first)] += 1;
        }
    }

    words_.clear();
    words_.reserve(word_counts_.size());
    for (const auto& entry : word_counts_) {
        WordEntry word;
        word.count = entry.second;
        word.symbols.reserve(entry.first.size() + 1);
        for (size_t i = 0; i < entry.first.size(); ++i) {
            word.symbols.push_back(static_cast<int32_t>(static_cast<unsigned char>(entry.first[i])));
        }
        word.symbols.push_back(detail::kBaseWordEndId);  // 词尾加 </w>
        words_.push_back(std::move(word));
    }

    // 初始化符号对频次与倒排索引
    std::vector<uint64_t> pairs;
    for (size_t index = 0; index < words_.size(); ++index) {
        const WordEntry& word = words_[index];
        collect_pairs(word.symbols, pairs);
        for (const uint64_t key : pairs) {
            add_pair_count(static_cast<int64_t>(word.count), key);
            pair_words_[key].push_back(static_cast<uint32_t>(index));
        }
    }
}

inline void BPETrainer::collect_pairs(const std::vector<int32_t>& symbols, std::vector<uint64_t>& out) {
    out.clear();
    if (symbols.size() < 2) {
        return;
    }
    out.reserve(symbols.size() - 1);
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        out.push_back(detail::pair_key(symbols[i], symbols[i + 1]));
    }
}

inline bool BPETrainer::contains_pair(const std::vector<int32_t>& symbols, int32_t left, int32_t right) {
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        if (symbols[i] == left && symbols[i + 1] == right) {
            return true;
        }
    }
    return false;
}

inline void BPETrainer::add_pair_count(int64_t delta, uint64_t key) {
    uint64_t& count = pair_counts_[key];
    if (delta >= 0) {
        count += static_cast<uint64_t>(delta);
    } else {
        const uint64_t amount = static_cast<uint64_t>(-delta);
        count = (count <= amount) ? 0 : (count - amount);
    }
    const uint64_t version = ++pair_versions_[key];
    if (count > 0) {
        PairCandidate candidate;
        candidate.count = count;
        candidate.key = key;
        candidate.version = version;
        heap_.push(candidate);
    }
}

inline void BPETrainer::merge_pair(uint64_t key, int32_t left, int32_t right) {
    // 这里必须拷贝：ensure_token 可能引起 id_to_piece_ 扩容
    const std::string left_piece = id_to_piece_[static_cast<size_t>(left)];
    const std::string right_piece = id_to_piece_[static_cast<size_t>(right)];
    merges_.emplace_back(left_piece, right_piece);
    const int32_t merged_id = ensure_token(left_piece + right_piece);

    const auto it = pair_words_.find(key);
    if (it == pair_words_.end()) {
        return;
    }
    // 拷贝一份再清空：被合并掉的符号对不可能再次出现，索引无需保留
    const std::vector<uint32_t> targets = it->second;
    it->second.clear();

    std::vector<uint64_t> pairs;
    for (const uint32_t index : targets) {
        if (static_cast<size_t>(index) >= words_.size()) {
            continue;
        }
        WordEntry& word = words_[static_cast<size_t>(index)];
        if (!contains_pair(word.symbols, left, right)) {
            continue;  // 倒排索引中的过期记录
        }

        // 1) 先扣掉旧符号对的计数
        collect_pairs(word.symbols, pairs);
        for (const uint64_t pair : pairs) {
            add_pair_count(-static_cast<int64_t>(word.count), pair);
        }

        // 2) 就地重写符号序列
        std::vector<int32_t> merged;
        merged.reserve(word.symbols.size());
        for (size_t i = 0; i < word.symbols.size();) {
            if (i + 1 < word.symbols.size() && word.symbols[i] == left && word.symbols[i + 1] == right) {
                merged.push_back(merged_id);
                i += 2;
            } else {
                merged.push_back(word.symbols[i]);
                i += 1;
            }
        }
        word.symbols.swap(merged);

        // 3) 登记新的符号对计数与倒排索引
        collect_pairs(word.symbols, pairs);
        for (const uint64_t pair : pairs) {
            add_pair_count(static_cast<int64_t>(word.count), pair);
            pair_words_[pair].push_back(index);
        }
    }
}

inline void BPETrainer::train() {
    build_base_vocab();
    build_word_entries();

    while (static_cast<int>(id_to_piece_.size()) < target_vocab_size_ && !heap_.empty()) {
        const PairCandidate top = heap_.top();
        heap_.pop();

        // 惰性失效校验
        const auto version_it = pair_versions_.find(top.key);
        if (version_it == pair_versions_.end() || version_it->second != top.version) {
            continue;
        }
        const auto count_it = pair_counts_.find(top.key);
        if (count_it == pair_counts_.end() || count_it->second != top.count) {
            continue;
        }
        // 大顶堆，最高频都小于 2 就可以停止了
        if (top.count < 2) {
            break;
        }
        merge_pair(top.key, detail::key_left(top.key), detail::key_right(top.key));
    }

    trained_ = true;
}

inline std::string BPETrainer::make_vocab_json() const {
    std::vector<std::pair<std::string, int32_t>> entries;
    entries.reserve(id_to_piece_.size());
    for (size_t id = 0; id < id_to_piece_.size(); ++id) {
        if (id_to_piece_[id].empty()) {
            continue;
        }
        entries.emplace_back(id_to_piece_[id], static_cast<int32_t>(id));
    }
    // 按 id 升序输出，保证结果可复现
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<std::string, int32_t>& a, const std::pair<std::string, int32_t>& b) {
                  return a.second < b.second;
              });

    std::string out;
    out += "{\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        out += "  ";
        out += detail::json_escape(entries[i].first);
        out += ": ";
        out += std::to_string(entries[i].second);
        if (i + 1 < entries.size()) {
            out += ",";
        }
        out += "\n";
    }
    out += "}\n";
    return out;
}

inline std::string BPETrainer::make_merges_text() const {
    std::string out = "#version: 0.2\n";
    for (const auto& merge : merges_) {
        out += merge.first;
        out += ' ';
        out += merge.second;
        out += '\n';
    }
    return out;
}

inline bool BPETrainer::save(const std::string& output_dir) const {
    if (id_to_piece_.empty()) {
        std::fprintf(stderr, "[trainer] 还没有训练结果，无法保存\n");
        return false;
    }
    const std::filesystem::path directory(output_dir);
    if (!directory.empty()) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            std::fprintf(stderr, "[trainer] 无法创建目录: %s\n", output_dir.c_str());
            return false;
        }
    }
    const std::filesystem::path vocab_path = directory / "vocab.json";
    const std::filesystem::path merges_path = directory / "merges.txt";
    if (!detail::write_file(vocab_path.string(), make_vocab_json())) {
        std::fprintf(stderr, "[trainer] 写入失败: %s\n", vocab_path.string().c_str());
        return false;
    }
    if (!detail::write_file(merges_path.string(), make_merges_text())) {
        std::fprintf(stderr, "[trainer] 写入失败: %s\n", merges_path.string().c_str());
        return false;
    }
    return true;
}

inline size_t BPETrainer::vocab_size() const {
    return id_to_piece_.size();
}

inline size_t BPETrainer::merge_count() const {
    return merges_.size();
}

inline const std::vector<std::string>& BPETrainer::pieces() const {
    return id_to_piece_;
}

}  // namespace json_libs

#endif  // JSON_LIBS_TOKENIZER_TRAINER_H
