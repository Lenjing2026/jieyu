#ifndef JSON_LIBS_JSON_H
#define JSON_LIBS_JSON_H

// ============================================================
//  libs/json_libs/json.h —— 极简 JSON 读写类
//
//  读：SetJsonWay(路径) -> GetJsonDate(键) / HasKey / ForEachKey / Keys
//  写：WriteJsonKey(键, 值)（存在则改、不存在则建，写完自动格式化）
//
//  本文件只是“聚合 + 业务类”，底层能力拆在三个头里：
//    file_util.h   文件读写 / 相对路径 / 按行切分 / UTF-8 编解码
//    json_value.h  值模型 Value + 序列化（2 空格自动格式化）
//    json_parser.h 递归下降解析器 + 类型自动识别
//
//  用法：
//    json_libs::json_lib json;
//    if (json.SetJsonWay("_file/BPE/vocab.json")) {
//        std::string id  = json.GetJsonDate("!");          // 顶层键
//        std::string val = json.GetJsonDate("model.type"); // 点号路径取嵌套键
//        json.WriteJsonKey("model.type", "bpe");           // 写入（自动格式化）
//    }
// ============================================================

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "file_util.h"
#include "json_parser.h"
#include "json_value.h"

namespace json_libs {

class json_lib {
public:
    json_lib() : loaded_(false) {}

    // 设置 JSON 文件路径并立即解析，返回是否成功
    // 允许相对路径：原样找不到时会依次在 ../ ../../ ../../../ 中查找
    bool SetJsonWay(const std::string& path) {
        path_ = path;
        return Reload();
    }

    // 用当前已设置的路径重新解析
    bool Reload() {
        loaded_ = false;
        root_ = json_detail::Value();
        error_.clear();
        if (path_.empty()) {
            error_ = "还没有设置路径，请先调用 SetJsonWay(路径)";
            return false;
        }
        const std::string resolved = file_util::resolve_relative_path(path_);
        std::string content;
        if (!file_util::read_file(resolved, content)) {
            error_ = "无法读取文件: " + resolved;
            return false;
        }
        json_detail::Parser parser(content);
        if (!parser.parse(root_)) {
            error_ = "解析失败: " + parser.error();
            root_ = json_detail::Value();
            return false;
        }
        loaded_ = true;
        return true;
    }

    // 当前路径 / 是否已成功加载
    const std::string& GetJsonWay() const {
        return path_;
    }

    bool IsLoaded() const {
        return loaded_;
    }

    // 按键取值，返回字符串；键不存在时返回空串
    //   * 字符串      -> 去掉引号并反转义后的内容
    //   * 数字/布尔   -> 原始字面量文本，如 "33" / "true"
    //   * null        -> "null"
    //   * 对象/数组   -> 该值在文件中的原始文本（含花括号/方括号）
    // key 支持 "a.b.c" 形式的嵌套路径；若顶层存在完全同名的键，则优先按整串匹配
    std::string GetJsonDate(const std::string& key) const {
        const json_detail::Value* value = find(key);
        if (value == nullptr) {
            return std::string();
        }
        return value_to_string(*value);
    }

    // 带默认值的版本：键不存在时返回 default_value
    std::string GetJsonDate(const std::string& key, const std::string& default_value) const {
        const json_detail::Value* value = find(key);
        if (value == nullptr) {
            return default_value;
        }
        return value_to_string(*value);
    }

    // 键是否存在
    bool HasKey(const std::string& key) const {
        return find(key) != nullptr;
    }

    // 顶层键个数（非对象或未加载时为 0）
    size_t KeyCount() const {
        if (!loaded_ || root_.type != json_detail::Value::Type::Object) {
            return 0;
        }
        return root_.members.size();
    }

    // 顶层全部键名
    std::vector<std::string> Keys() const {
        std::vector<std::string> keys;
        if (loaded_ && root_.type == json_detail::Value::Type::Object) {
            keys.reserve(root_.members.size());
            for (const auto& member : root_.members) {
                keys.push_back(member.first);
            }
        }
        return keys;
    }

    // 遍历顶层键值（O(n)，适合大文件，例如 5 万条词表）
    // 值统一以字符串给出（规则同 GetJsonDate）；visit 返回 false 可提前结束
    // 返回 false 表示未加载 / 顶层不是对象（visit 主动中断也返回 false）
    bool ForEachKey(const std::function<bool(const std::string& key, const std::string& value)>& visit) const {
        if (!loaded_ || root_.type != json_detail::Value::Type::Object) {
            return false;
        }
        for (const auto& member : root_.members) {
            if (!visit(member.first, value_to_string(member.second))) {
                return false;
            }
        }
        return true;
    }

    // 最近一次失败的原因（成功时为空串）
    const std::string& LastError() const {
        return error_;
    }

    // 写入键值：存在则修改，不存在则创建，写完自动格式化并保存到文件
    // value 自动识别类型：
    //   合法 JSON 数字（42 / -3.5 / 1e9）  -> 写成数字
    //   true / false / null               -> 写成对应字面量
    //   以 { 或 [ 开头且能解析成 JSON      -> 原样嵌入为对象 / 数组
    //   其余                              -> 写成 JSON 字符串
    // key 支持 "a.b.c"：顶层同名键优先整体匹配，否则按 . 逐层下钻（缺的层级自动建对象）
    bool WriteJsonKey(const std::string& key, const std::string& value) {
        if (key.empty()) {
            error_ = "键不能为空";
            return false;
        }
        if (path_.empty()) {
            error_ = "还没有设置路径，请先调用 SetJsonWay(路径)";
            return false;
        }
        if (!ensure_writable()) {
            return false;
        }
        const json_detail::Value new_value = json_detail::make_value(value);
        if (!set_member(root_, key, new_value)) {
            return false;
        }
        return SaveJsonFile();
    }

    // 把当前内存中的 JSON 按 2 空格缩进写回文件
    bool SaveJsonFile() {
        if (!loaded_) {
            error_ = "还没有可保存的内容";
            return false;
        }
        const std::string resolved = file_util::resolve_relative_path(path_);
        std::string text;
        json_detail::write_value(text, root_, 0);
        text.push_back('\n');
        if (!file_util::write_file(resolved, text)) {
            error_ = "写入失败: " + resolved;
            return false;
        }
        error_.clear();
        return true;
    }

    // 当前内存中的 JSON 文本（2 空格缩进，未加载时返回空串）
    std::string ToJsonText() const {
        if (!loaded_) {
            return std::string();
        }
        std::string text;
        json_detail::write_value(text, root_, 0);
        text.push_back('\n');
        return text;
    }

private:
    // 值 -> 字符串
    //   * 字符串      -> 取内容（已反转义）
    //   * 对象 / 数组 -> 解析得来的用原文；程序写出来的实时序列化
    //   * 其它        -> 取原始字面量（如 "20" / "true" / "null"）
    static std::string value_to_string(const json_detail::Value& value) {
        if (value.type == json_detail::Value::Type::String) {
            return value.text;
        }
        if (!value.raw.empty()) {
            return value.raw;
        }
        std::string text;
        json_detail::write_value(text, value, 0);
        return text;
    }

    // 查找键：先按整串匹配顶层键，再按 '.' 拆成路径逐层下钻
    const json_detail::Value* find(const std::string& key) const {
        if (!loaded_ || root_.type != json_detail::Value::Type::Object) {
            return nullptr;
        }
        for (const auto& member : root_.members) {
            if (member.first == key) {
                return &member.second;
            }
        }
        if (key.find('.') == std::string::npos) {
            return nullptr;
        }
        const json_detail::Value* current = &root_;
        size_t start = 0;
        while (true) {
            const size_t dot = key.find('.', start);
            const std::string part = (dot == std::string::npos)
                ? key.substr(start)
                : key.substr(start, dot - start);
            if (part.empty() || current->type != json_detail::Value::Type::Object) {
                return nullptr;
            }
            const json_detail::Value* next = nullptr;
            for (const auto& member : current->members) {
                if (member.first == part) {
                    next = &member.second;
                    break;
                }
            }
            if (next == nullptr) {
                return nullptr;
            }
            current = next;
            if (dot == std::string::npos) {
                break;
            }
            start = dot + 1;
        }
        return current;
    }

    // 确保有可写的根对象：
    //   * 已成功加载 -> 直接用
    //   * 文件不存在或为空 -> 从空对象开始（相当于新建文件）
    //   * 文件有内容但解析失败 -> 拒绝写入，避免覆盖
    bool ensure_writable() {
        if (loaded_) {
            return true;
        }
        const std::string resolved = file_util::resolve_relative_path(path_);
        std::string content;
        if (file_util::read_file(resolved, content)
            && content.find_first_not_of(" \t\r\n") != std::string::npos) {
            if (error_.empty()) {
                error_ = "文件已存在但解析失败，为避免覆盖已拒绝写入: " + resolved;
            }
            return false;
        }
        root_ = json_detail::Value();
        root_.type = json_detail::Value::Type::Object;
        loaded_ = true;
        error_.clear();
        return true;
    }

    // 写入成员：顶层同名键直接替换，否则按 '.' 路径逐层下钻并创建缺失的对象
    bool set_member(json_detail::Value& root, const std::string& key, const json_detail::Value& value) {
        for (auto& member : root.members) {
            if (member.first == key) {
                member.second = value;
                return true;
            }
        }
        if (key.find('.') == std::string::npos) {
            root.members.emplace_back(key, value);
            return true;
        }
        json_detail::Value* current = &root;
        size_t start = 0;
        while (true) {
            const size_t dot = key.find('.', start);
            const std::string part = (dot == std::string::npos)
                ? key.substr(start)
                : key.substr(start, dot - start);
            if (part.empty()) {
                error_ = "键路径中有空的层级: " + key;
                return false;
            }
            const bool is_last = (dot == std::string::npos);
            json_detail::Value* next = nullptr;
            for (auto& member : current->members) {
                if (member.first == part) {
                    next = &member.second;
                    break;
                }
            }
            if (is_last) {
                if (next != nullptr) {
                    *next = value;
                } else {
                    current->members.emplace_back(part, value);
                }
                return true;
            }
            if (next == nullptr) {
                json_detail::Value created;
                created.type = json_detail::Value::Type::Object;
                current->members.emplace_back(part, created);
                next = &current->members.back().second;
            } else if (next->type != json_detail::Value::Type::Object) {
                error_ = "键路径中的 \"" + part + "\" 不是对象，无法写入";
                return false;
            }
            current = next;
            start = dot + 1;
        }
    }

    std::string path_;
    std::string error_;
    json_detail::Value root_;
    bool loaded_;
};

}  // namespace json_libs

#endif  // JSON_LIBS_JSON_H
