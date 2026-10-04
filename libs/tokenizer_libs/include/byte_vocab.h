#ifndef JSON_LIBS_BYTE_VOCAB_H
#define JSON_LIBS_BYTE_VOCAB_H

// ============================================================
//  byte_vocab.h —— 生成“纯字节级”词表（GPT-2 byte-level 那 256 个）
//
//  用途：造小模型 / 自测 / 演示时，不需要 HuggingFace 的 vocab.json，
//        直接用分词器内置的字节映射表写一份最小可用的词表。
//
//    vocab.json   {"!":0, "\"":1, ...}  共 256 项
//    merges.txt   只有一行 "#version: 0.2"（没有合并规则）
//
//  加载后 Tokenizer::encode 会把文本按 UTF-8 字节切开，
//  decode(encode(text)) == text 恒成立。
// ============================================================

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include "tokenizer.h"

namespace json_libs {

// 纯字节级词表的 token 个数
inline constexpr size_t kByteVocabSize = 256;

inline void write_byte_level_vocab(const std::string& vocab_path, const std::string& merges_path) {
    const std::array<std::string, 256>& table = detail::byte_level_table();
    std::ofstream vocab(vocab_path, std::ios::binary | std::ios::trunc);
    if (!vocab) throw std::runtime_error("byte_vocab: 写不了 " + vocab_path);
    vocab << "{";
    for (size_t b = 0; b < table.size(); b++) {
        if (b != 0) vocab << ",";
        vocab << "\"";
        for (char c : table[b]) {
            if (c == '"' || c == '\\') vocab << "\\";
            vocab << c;
        }
        vocab << "\":" << b;
    }
    vocab << "}";
    vocab.flush();
    if (!vocab) throw std::runtime_error("byte_vocab: 写入失败 " + vocab_path);

    std::ofstream merges(merges_path, std::ios::binary | std::ios::trunc);
    if (!merges) throw std::runtime_error("byte_vocab: 写不了 " + merges_path);
    merges << "#version: 0.2\n";
    merges.flush();
    if (!merges) throw std::runtime_error("byte_vocab: 写入失败 " + merges_path);
}

}  // namespace json_libs

#endif  // JSON_LIBS_BYTE_VOCAB_H
