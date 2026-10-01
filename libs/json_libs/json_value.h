#ifndef JSON_LIBS_JSON_VALUE_H
#define JSON_LIBS_JSON_VALUE_H

// ============================================================
//  libs/json_libs/json_value.h —— JSON 值模型 + 序列化
//
//  * 只负责“数据模型 + 文本输出”，解析器见 json_parser.h
//  * write_value 统一按 2 空格缩进输出（自动格式化）
//  * 纯 C++17，只依赖标准库，全部 inline
// ============================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace json_libs {
namespace json_detail {

// 一个 JSON 值
struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    std::string text;                                     // 字符串内容（已反转义）
    std::string raw;                                      // 该值在源文本中的原始片段
    std::vector<Value> items;                             // 数组元素
    std::vector<std::pair<std::string, Value>> members;   // 对象成员
};

// 把字符串转义后写进 JSON 文本（含首尾引号）
inline void escape_string(std::string& out, const std::string& text) {
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
}

// 缩进：每层 2 个空格
inline void write_indent(std::string& out, int level) {
    out.append(static_cast<size_t>(level) * 2, ' ');
}

// 递归序列化：统一 2 空格缩进（自动格式化）
inline void write_value(std::string& out, const Value& value, int level) {
    switch (value.type) {
        case Value::Type::Null:
            out += "null";
            return;
        case Value::Type::Bool:
        case Value::Type::Number:
            out += value.raw.empty() ? std::string("null") : value.raw;
            return;
        case Value::Type::String:
            escape_string(out, value.text);
            return;
        case Value::Type::Array: {
            if (value.items.empty()) {
                out += "[]";
                return;
            }
            out += "[\n";
            for (size_t i = 0; i < value.items.size(); ++i) {
                write_indent(out, level + 1);
                write_value(out, value.items[i], level + 1);
                if (i + 1 < value.items.size()) {
                    out.push_back(',');
                }
                out.push_back('\n');
            }
            write_indent(out, level);
            out.push_back(']');
            return;
        }
        case Value::Type::Object: {
            if (value.members.empty()) {
                out += "{}";
                return;
            }
            out += "{\n";
            for (size_t i = 0; i < value.members.size(); ++i) {
                write_indent(out, level + 1);
                escape_string(out, value.members[i].first);
                out += ": ";
                write_value(out, value.members[i].second, level + 1);
                if (i + 1 < value.members.size()) {
                    out.push_back(',');
                }
                out.push_back('\n');
            }
            write_indent(out, level);
            out.push_back('}');
            return;
        }
    }
}

// 是否是合法的 JSON 数字字面量（42 / -3.5 / 1e9，不允许前导 0）
inline bool is_json_number(const std::string& text) {
    if (text.empty()) {
        return false;
    }
    size_t i = 0;
    if (text[i] == '-') {
        i += 1;
    }
    if (i >= text.size()) {
        return false;
    }
    if (text[i] == '0') {
        i += 1;
        if (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            return false;
        }
    } else if (text[i] >= '1' && text[i] <= '9') {
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            i += 1;
        }
    } else {
        return false;
    }
    if (i < text.size() && text[i] == '.') {
        i += 1;
        const size_t digits = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            i += 1;
        }
        if (i == digits) {
            return false;
        }
    }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        i += 1;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
            i += 1;
        }
        const size_t digits = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            i += 1;
        }
        if (i == digits) {
            return false;
        }
    }
    return i == text.size();
}

}  // namespace json_detail
}  // namespace json_libs

#endif  // JSON_LIBS_JSON_VALUE_H
