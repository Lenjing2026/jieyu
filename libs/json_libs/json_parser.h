#ifndef JSON_LIBS_JSON_PARSER_H
#define JSON_LIBS_JSON_PARSER_H

// ============================================================
//  libs/json_libs/json_parser.h —— 递归下降 JSON 解析器
//
//  * 支持 对象 / 数组 / 字符串 / 数字 / 布尔 / null
//  * 字符串支持 \" \\ \/ \b \f \n \r \t \uXXXX 以及 UTF-16 代理对
//  * 每个值同时记录 raw（源文本片段），便于原样输出
//  * make_value 把字符串转成值，并自动识别类型
//  * 纯 C++17，只依赖标准库，全部 inline
// ============================================================

#include "file_util.h"
#include "json_value.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace json_libs {
namespace json_detail {

class Parser {
public:
    explicit Parser(const std::string& source) : src_(source), pos_(0) {}

    bool parse(Value& out) {
        skip_ws();
        if (!parse_value(out)) {
            return false;
        }
        skip_ws();
        if (pos_ != src_.size()) {
            return fail("JSON 末尾存在多余内容");
        }
        return true;
    }

    const std::string& error() const {
        return error_;
    }

private:
    void skip_ws() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_ += 1;
                continue;
            }
            break;
        }
    }

    bool fail(const std::string& message) {
        if (error_.empty()) {
            error_ = message;
        }
        return false;
    }

    bool match(char expected) {
        if (pos_ < src_.size() && src_[pos_] == expected) {
            pos_ += 1;
            return true;
        }
        return false;
    }

    bool expect(char expected) {
        if (match(expected)) {
            return true;
        }
        std::string message = "期望字符 ";
        message.push_back(expected);
        return fail(message);
    }

    bool parse_value(Value& out) {
        skip_ws();
        if (pos_ >= src_.size()) {
            return fail("JSON 提前结束");
        }
        const size_t begin = pos_;
        const char c = src_[pos_];
        bool ok = false;
        if (c == '{') {
            ok = parse_object(out);
        } else if (c == '[') {
            ok = parse_array(out);
        } else if (c == '"') {
            out.type = Value::Type::String;
            ok = parse_string(out.text);
        } else if (c == 't' || c == 'f') {
            ok = parse_bool(out);
        } else if (c == 'n') {
            ok = parse_null(out);
        } else {
            ok = parse_number(out);
        }
        if (!ok) {
            return false;
        }
        out.raw = src_.substr(begin, pos_ - begin);
        return true;
    }

    bool parse_object(Value& out) {
        out.type = Value::Type::Object;
        if (!expect('{')) {
            return false;
        }
        skip_ws();
        if (match('}')) {
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) {
                return false;
            }
            skip_ws();
            if (!expect(':')) {
                return false;
            }
            Value value;
            if (!parse_value(value)) {
                return false;
            }
            out.members.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (match(',')) {
                continue;
            }
            if (match('}')) {
                return true;
            }
            return fail("对象缺少 , 或 }");
        }
    }

    bool parse_array(Value& out) {
        out.type = Value::Type::Array;
        if (!expect('[')) {
            return false;
        }
        skip_ws();
        if (match(']')) {
            return true;
        }
        while (true) {
            Value value;
            if (!parse_value(value)) {
                return false;
            }
            out.items.push_back(std::move(value));
            skip_ws();
            if (match(',')) {
                continue;
            }
            if (match(']')) {
                return true;
            }
            return fail("数组缺少 , 或 ]");
        }
    }

    bool parse_string(std::string& out) {
        if (!expect('"')) {
            return false;
        }
        out.clear();
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            pos_ += 1;
            if (c == '"') {
                return true;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= src_.size()) {
                break;
            }
            const char escape = src_[pos_];
            pos_ += 1;
            switch (escape) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if (!parse_hex4(codepoint)) {
                        return false;
                    }
                    // UTF-16 代理对
                    if (codepoint >= 0xD800u && codepoint <= 0xDBFFu
                        && pos_ + 1 < src_.size()
                        && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                        pos_ += 2;
                        uint32_t low = 0;
                        if (!parse_hex4(low)) {
                            return false;
                        }
                        if (low >= 0xDC00u && low <= 0xDFFFu) {
                            codepoint = 0x10000u + ((codepoint - 0xD800u) << 10) + (low - 0xDC00u);
                        }
                    }
                    out += file_util::utf8_encode(codepoint);
                    break;
                }
                default:
                    return fail("非法的转义字符");
            }
        }
        return fail("字符串未闭合");
    }

    bool parse_hex4(uint32_t& out) {
        if (pos_ + 4 > src_.size()) {
            return fail("\\u 转义不完整");
        }
        uint32_t value = 0;
        for (size_t i = 0; i < 4; ++i) {
            const char c = src_[pos_ + i];
            uint32_t digit = 0;
            if (c >= '0' && c <= '9') {
                digit = static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                digit = static_cast<uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                digit = static_cast<uint32_t>(c - 'A' + 10);
            } else {
                return fail("\\u 转义包含非十六进制字符");
            }
            value = (value << 4) | digit;
        }
        pos_ += 4;
        out = value;
        return true;
    }

    bool parse_number(Value& out) {
        const size_t start = pos_;
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            const bool accepted = (c >= '0' && c <= '9')
                || c == '-' || c == '+' || c == '.'
                || c == 'e' || c == 'E';
            if (!accepted) {
                break;
            }
            pos_ += 1;
        }
        if (pos_ == start) {
            return fail("非法的数字");
        }
        out.type = Value::Type::Number;
        return true;
    }

    bool parse_bool(Value& out) {
        if (pos_ + 4 <= src_.size() && src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out.type = Value::Type::Bool;
            return true;
        }
        if (pos_ + 5 <= src_.size() && src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out.type = Value::Type::Bool;
            return true;
        }
        return fail("非法的布尔字面量");
    }

    bool parse_null(Value& out) {
        if (pos_ + 4 <= src_.size() && src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out.type = Value::Type::Null;
            return true;
        }
        return fail("非法的 null 字面量");
    }

    const std::string& src_;
    size_t pos_;
    std::string error_;
};

// 由字符串构造 JSON 值（写键值时的类型自动识别，依赖上面的 Parser）
inline Value make_value(const std::string& text) {
    Value value;
    if (is_json_number(text)) {
        value.type = Value::Type::Number;
        value.raw = text;
        return value;
    }
    if (text == "true" || text == "false") {
        value.type = Value::Type::Bool;
        value.raw = text;
        return value;
    }
    if (text == "null") {
        value.type = Value::Type::Null;
        value.raw = "null";
        return value;
    }
    if (!text.empty() && (text[0] == '{' || text[0] == '[')) {
        // 看起来是对象 / 数组，能解析就原样嵌入
        Parser parser(text);
        Value parsed;
        if (parser.parse(parsed)) {
            return parsed;
        }
    }
    value.type = Value::Type::String;
    value.text = text;
    return value;
}

}  // namespace json_detail
}  // namespace json_libs

#endif  // JSON_LIBS_JSON_PARSER_H
