# JSON 模块使用文档

`libs/json_libs/` 下的一套 **header-only JSON 读写组件**：纯 C++17，只依赖标准库，全部函数 `inline`，无需编译任何 `.cpp`。

- 读：`SetJsonWay(路径)` → `GetJsonDate(键)` / `HasKey` / `ForEachKey`
- 写：`WriteJsonKey(键, 值)`（存在则改、不存在则建，写完自动 2 空格缩进格式化并落盘）

---

## 1. 文件拆分

| 文件 | 命名空间 | 职责 |
| --- | --- | --- |
| `file_util.h` | `json_libs::file_util` | 文件读写、相对路径解析、按行切分、UTF-8 编解码 |
| `json_value.h` | `json_libs::json_detail` | `Value` 值模型、转义、序列化（2 空格自动格式化）、数字合法性校验 |
| `json_parser.h` | `json_libs::json_detail` | 递归下降解析器 `Parser`、`make_value`（写入时的类型自动识别） |
| `json.h` | `json_libs` | `json_lib` 类（对外主力接口，聚合上面三个头） |

> 一般只用 `#include "json.h"` 即可；需要文件工具时可以单独 `#include "file_util.h"`。

---

## 2. 快速开始

编译（把 `libs/json_libs` 加进头文件搜索路径）：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Ilibs/json_libs main.cpp -o app
```

最小示例：

```cpp
#include "json.h"
#include <iostream>

int main() {
    json_libs::json_lib json;

    // 读
    if (json.SetJsonWay("_file/BPE/vocab.json")) {
        std::cout << json.GetJsonDate("!") << "\n";          // 顶层键
        std::cout << json.GetJsonDate("model.type") << "\n"; // 嵌套键（点号路径）
    } else {
        std::cerr << json.LastError() << "\n";
    }

    // 写（存在则改、不存在则建，自动格式化）
    json.SetJsonWay("_file/BPE/config.json");
    json.WriteJsonKey("model.type", "bpe");
    json.WriteJsonKey("model.size", "50000");
}
```

---

## 3. `json_libs::json_lib` 接口一览

| 函数 | 参数 | 返回 | 说明 |
| --- | --- | --- | --- |
| `SetJsonWay` | `const std::string& path` | `bool` | 设置文件路径并立即解析；支持相对路径 |
| `Reload` | — | `bool` | 用当前路径重新解析 |
| `GetJsonWay` | — | `const std::string&` | 当前路径 |
| `IsLoaded` | — | `bool` | 是否已成功加载 |
| `GetJsonDate` | `const std::string& key` | `std::string` | 取值；键不存在返回空串 |
| `GetJsonDate` | `key`、`const std::string& default_value` | `std::string` | 键不存在返回默认值 |
| `HasKey` | `key` | `bool` | 键是否存在 |
| `KeyCount` | — | `size_t` | 顶层键个数（非对象 / 未加载为 0） |
| `Keys` | — | `std::vector<std::string>` | 顶层全部键名（顺序 = 文件中的顺序） |
| `ForEachKey` | `const std::function<bool(const std::string& key, const std::string& value)>& visit` | `bool` | 遍历顶层键值，O(n)；`visit` 返回 `false` 可提前结束 |
| `LastError` | — | `const std::string&` | 最近一次失败原因（成功时为空串） |
| `WriteJsonKey` | `const std::string& key`、`const std::string& value` | `bool` | 写入键值，写完自动格式化保存 |
| `SaveJsonFile` | — | `bool` | 把内存中的 JSON 写回文件（一般不必手动调用） |
| `ToJsonText` | — | `std::string` | 预览当前 JSON 文本（2 空格缩进，未加载返回空串） |

---

## 4. 读取

### 4.1 取值规则（`GetJsonDate`）

| JSON 里的类型 | 返回的字符串 |
| --- | --- |
| 字符串 `"abc"` | `abc`（已反转义，不含引号） |
| 数字 `20` / `3.5` | `20` / `3.5`（原始字面量文本） |
| 布尔 `true` | `true` |
| `null` | `null` |
| 对象 / 数组 | 该值的 JSON 文本（解析得来的用原文；程序写出来的实时序列化） |
| 键不存在 | `""`（用两参版本则返回默认值） |

### 4.2 键路径规则

1. **先按整串匹配顶层键**：如果顶层真的存在 `"a.b"` 这样一个键，就取它；
2. 没有再按 `.` 拆分逐层下钻：`GetJsonDate("model.type")`。

这样既能取嵌套字段，又不会把词表里带 `.` 的 token 误拆。

### 4.3 遍历大文件

`ForEachKey` 是 O(n) 的一趟遍历，适合几万条的大文件（tokenizer 就是这么读 `vocab.json` 的）：

```cpp
json_libs::json_lib json;
json.SetJsonWay("_file/BPE/vocab.json");
size_t count = 0;
json.ForEachKey([&](const std::string& key, const std::string& value) -> bool {
    const long id = std::strtol(value.c_str(), nullptr, 10);
    std::cout << key << " -> " << id << "\n";
    count += 1;
    return true;   // 返回 false 可提前结束
});
```

---

## 5. 写入（`WriteJsonKey`）

### 5.1 值类型自动识别

| 传入的字符串 | 写进 JSON 的形式 |
| --- | --- |
| `"42"`、`"-3.5"`、`"1e9"` | 数字 `42` / `-3.5` / `1e9` |
| `"true"`、`"false"` | 布尔 |
| `"null"` | `null` |
| `"{\"x\":1}"`、`"[1,2,3]"` | 解析后原样嵌入为对象 / 数组 |
| 其它（含 `"0123"` 这种非法数字格式） | JSON 字符串（自动加引号并转义） |

### 5.2 键路径与顺序

- `WriteJsonKey("model.type", "bpe")`、`WriteJsonKey("a.b.c", "1")` 会自动创建缺失的中间对象；
- 顶层同名键优先整体匹配（与读取一致）；
- **已有键保持原位置**，新键追加到末尾；
- 写完把整棵树按 2 空格缩进重新序列化后落盘。

### 5.3 安全行为

| 情况 | 行为 |
| --- | --- |
| 文件不存在 / 内容为空 | 从空对象开始，正常创建并写入 |
| 文件有内容但 JSON 语法错误 | **拒绝写入**（不覆盖原文件），`LastError()` 说明原因 |
| 键为空串 | 返回 `false`，`LastError()` 为「键不能为空」 |
| 键路径中间某层不是对象 | 返回 `false`，`LastError()` 指出是哪一层 |

---

## 6. 相对路径规则（读和写都适用）

路径解析顺序：**原样 → `../` → `../../` → `../../../`**。

因此无论从仓库根目录、`build/` 还是更深的子目录运行，都能定位到同一份数据。实现见 `file_util::resolve_relative_path`（用 `ifstream` 探测可读性，纯标准库、无平台代码）。

---

## 7. 底层接口（需要时可直接用）

### 7.1 `json_libs::file_util`（`file_util.h`）

| 函数 | 参数 | 返回 |
| --- | --- | --- |
| `read_file` | `const std::string& path`、`std::string& out` | `bool`（二进制模式整文件读入） |
| `write_file` | `path`、`const std::string& content` | `bool` |
| `file_readable` | `path` | `bool` |
| `resolve_relative_path` | `path` | `std::string`（解析结果，找不到时原样返回） |
| `split_lines` | `const std::string& text` | `std::vector<std::string>`（按 `\n` 切，保留 `\r`） |
| `utf8_encode` | `uint32_t codepoint` | `std::string` |
| `utf8_decode` | `const char* data`、`size_t size`、`uint32_t& codepoint`、`size_t& length` | `void`（非法序列按单字节处理） |

### 7.2 `json_libs::json_detail`

| 名称 | 说明 |
| --- | --- |
| `Value` | 值模型：`type` / `text`（字符串内容）/ `raw`（原始文本）/ `items`（数组）/ `members`（对象） |
| `Parser` | 递归下降解析器：`Parser(src)` → `parse(Value&)` → `error()` |
| `make_value` | 字符串 → `Value`（类型自动识别） |
| `write_value` | `Value` → 2 空格缩进的 JSON 文本 |
| `is_json_number` | 是否合法 JSON 数字字面量 |
| `escape_string` | 转义并写出带引号的字符串 |

---

## 8. 与 tokenizer 的关系

- `libs/tokenizer/include/tokenizer.h` 里原本自带一份 JSON 解析器，**现已删除**，改为：

  ```cpp
  #include "../../json_libs/file_util.h"
  #include "../../json_libs/json.h"
  ```

- `Tokenizer::load` 用 `json_libs::json_lib` + `ForEachKey` 一趟读完 `vocab.json`；
- `utf8_*` / `read_file` / `write_file` / `split_lines` / `resolve_relative_path` 也都通过 `using file_util::xxx;` 复用，**全局只有一份实现**；
- 因此单独编译 tokenizer 时不需要额外的 `-I`（相对路径包含），而要用 `json_lib` 类本身的程序加 `-Ilibs/json_libs` 即可。

---

## 9. 完整示例（实测输出）

```cpp
#include "json.h"
#include <iostream>

int main(int argc, char** argv) {
    json_libs::json_lib j;
    j.SetJsonWay(argv[1]);          // 文件不存在也行，写的时候会创建

    j.WriteJsonKey("b", "2");       // 数字
    j.WriteJsonKey("a", "1");
    j.WriteJsonKey("b", "20");      // 已存在 -> 修改，位置不变
    j.WriteJsonKey("c.d", "3");     // 自动创建中间对象 c
    j.WriteJsonKey("flag", "true"); // 布尔
    j.WriteJsonKey("nil", "null");  // null
    j.WriteJsonKey("name", "昆仑");  // 字符串

    std::cout << j.ToJsonText();
    std::cout << j.GetJsonDate("b") << " " << j.GetJsonDate("c.d") << "\n";
    std::cout << j.KeyCount() << " " << j.HasKey("c.d") << "\n";
    for (const std::string& k : j.Keys()) { std::cout << k << " "; }
    std::cout << "\n";
    return 0;
}
```

生成的 `doc_demo.json`：

```json
{
  "b": 20,
  "a": 1,
  "c": {
    "d": 3
  },
  "flag": true,
  "nil": null,
  "name": "昆仑"
}
```

程序输出：

```
--- GetJsonDate ---
b=[20] c.d=[3] c=[{
  "d": 3
}] x=[] xdef=[-]
keycount=6 has(c)=1 has(c.d)=1
keys: b a c flag nil name
  foreach b = 20
  foreach a = 1
  foreach c = {
  "d": 3
}
  foreach flag = true
  foreach nil = null
  foreach name = 昆仑
lastError=[]
```

要点：修改 `b` 后它仍在第一位；`c.d` 自动建了 `c` 对象；`"20"` 写成数字、`"true"` 写成布尔、`"昆仑"` 写成字符串。

---

## 10. 注意事项与限制

- **字符串转义**：解析支持 `\" \\ \/ \b \f \n \r \t \uXXXX` 以及 UTF-16 代理对；写出时非 ASCII 字符按 UTF-8 原样输出（不转成 `\uXXXX`），引号、反斜杠、控制字符会转义；
- **数字保持原文**：`1E2`、`3.50` 这类写法写回时保持原样，不做数值规范化；
- **标准 JSON 子集**：不支持注释、单引号字符串、尾随逗号等 JSON5 扩展；
- **整体载入内存**：读是整文件读入、整棵树解析，超大 JSON 要注意内存；写入是整棵树重新序列化后覆盖写；
- **格式化**：写出统一 2 空格缩进；键顺序保持（新键追加末尾），但缩进会按统一规则重排（原本单行紧凑的文件会变成多行）；
- **无全局状态**：`json_lib` 是普通对象，多个实例互不影响；`SetJsonWay` 只作用于该实例；
- **线程安全**：同一实例不要多线程并发读写（不同实例可以）。
