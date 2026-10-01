#ifndef JSON_LIBS_FILE_UTIL_H
#define JSON_LIBS_FILE_UTIL_H

// ============================================================
//  libs/json_libs/file_util.h —— 通用文本 / 文件小工具
//
//  这些工具被 json.h 与 tokenizer 共同使用，所以单独拆成一个头，
//  保证同一份实现只存在一处（避免多份拷贝各自演化）：
//    * read_file / write_file / file_readable   文件读写
//    * resolve_relative_path                    相对路径逐级向上查找
//    * split_lines                              按 '\n' 切分文本
//    * utf8_encode / utf8_decode                UTF-8 编解码
//
//  纯 C++17，只依赖标准库，全部 inline
// ============================================================

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace json_libs {
namespace file_util {

// ------------------------------------------------------------
// UTF-8
// ------------------------------------------------------------

// 把码点编码成 UTF-8 字符串
inline std::string utf8_encode(uint32_t codepoint) {
    std::string out;
    if (codepoint <= 0x7Fu) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    } else if (codepoint <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
    }
    return out;
}

// 解码一个 UTF-8 序列，输出码点与字节长度；非法序列按单字节处理
inline void utf8_decode(const char* data, size_t size, uint32_t& codepoint, size_t& length) {
    const unsigned char b0 = static_cast<unsigned char>(data[0]);
    codepoint = b0;
    length = 1;
    if (b0 < 0x80u) {
        return;
    }
    if ((b0 & 0xE0u) == 0xC0u && size >= 2) {
        const unsigned char b1 = static_cast<unsigned char>(data[1]);
        if ((b1 & 0xC0u) == 0x80u) {
            codepoint = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            length = 2;
        }
        return;
    }
    if ((b0 & 0xF0u) == 0xE0u && size >= 3) {
        const unsigned char b1 = static_cast<unsigned char>(data[1]);
        const unsigned char b2 = static_cast<unsigned char>(data[2]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
            codepoint = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            length = 3;
        }
        return;
    }
    if ((b0 & 0xF8u) == 0xF0u && size >= 4) {
        const unsigned char b1 = static_cast<unsigned char>(data[1]);
        const unsigned char b2 = static_cast<unsigned char>(data[2]);
        const unsigned char b3 = static_cast<unsigned char>(data[3]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u) {
            codepoint = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            length = 4;
        }
        return;
    }
    // 非法 UTF-8 序列：按单个字节处理，保证编码与解码可逆
    codepoint = b0;
    length = 1;
}

// ------------------------------------------------------------
// 文件读写
// ------------------------------------------------------------

// 一次性读入整个文件（二进制模式，不改动换行）
inline bool read_file(const std::string& path, std::string& out) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    out = buffer.str();
    return true;
}

// 一次性写出整个文件（二进制模式）
inline bool write_file(const std::string& path, const std::string& content) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    return output.good();
}

// 文件是否可打开
inline bool file_readable(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return static_cast<bool>(input);
}

// 把可能是相对路径的路径解析成实际可读的路径，查找顺序：
//   1) 原样（相对进程当前工作目录）
//   2) ../      3) ../../      4) ../../../
// 全部失败时原样返回（让调用方按原路径报错）
inline std::string resolve_relative_path(const std::string& path) {
    if (path.empty() || file_readable(path)) {
        return path;
    }
    std::string prefix;
    for (int level = 0; level < 3; ++level) {
        prefix += "../";
        const std::string candidate = prefix + path;
        if (file_readable(candidate)) {
            return candidate;
        }
    }
    return path;
}

// ------------------------------------------------------------
// 文本切分
// ------------------------------------------------------------

// 按 '\n' 切分文本（保留 '\r'，由调用方决定是否去掉）
inline std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    if (text.empty()) {
        return lines;
    }
    size_t index = 0;
    while (index < text.size()) {
        const size_t next = text.find('\n', index);
        if (next == std::string::npos) {
            lines.push_back(text.substr(index));
            break;
        }
        lines.push_back(text.substr(index, next - index));
        index = next + 1;
    }
    return lines;
}

}  // namespace file_util
}  // namespace json_libs

#endif  // JSON_LIBS_FILE_UTIL_H
