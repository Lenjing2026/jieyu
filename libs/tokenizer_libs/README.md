# tokenizer —— 字节级 BPE 分词器（namespace json_libs）

纯 C++17 实现的**字节级 BPE 分词器 + 训练器**，不依赖任何第三方库（只用 C++ 标准库）。
输出格式与 HuggingFace（GPT-2 风格）兼容，可直接被 `transformers` 的 `PreTrainedTokenizerFast` 之类的工具读取。

---

## 目录结构

```
libs/tokenizer/
├── include/
│   ├── tokenizer.h      # 分词器：声明 + 全部 inline 实现（JSON 复用 libs/json_libs）
│   └── trainer.h        # 训练器：声明 + 全部 inline 实现（内部 include tokenizer.h）
└── README.md
```

> **header-only**：没有 `src/*.cpp`，实现全部内联在 `include/` 的两个头文件里。
> 主类（或任何使用方）只需 `#include "tokenizer.h"` 或 `#include "trainer.h"` 即可调用。

---

## 编译

不需要 CMake：本子模块是 header-only，编译使用方时把 `include/` 加入搜索路径即可。

```bash
# 在 JieYuAI 根目录执行（编译顶层主类）
g++ -std=c++17 -O2 -Wall -Wextra -Ilibs/tokenizer/include main.cpp -o tokenizer_cli
```

- Windows（MinGW）生成 `tokenizer_cli.exe`
- 已验证工具链：g++ 16.2.0（WinLibs MinGW-w64 UCRT）
- 本子模块自身不产出 `.a`，也没有独立可执行文件 / 测试程序

使用方只需要包含头文件：

```cpp
#include "tokenizer.h"   // 只要分词器
#include "trainer.h"     // 训练器（内部会引入 tokenizer.h）
```

---

## 使用

以下命令由顶层 `main.cpp` 编译出的 `tokenizer_cli` 提供（Windows 下为 `.\\tokenizer_cli.exe`）：

```bash
# 1. 训练：语料每行一句，输出 vocab.json + merges.txt
./tokenizer_cli train _file/BPE/corpus.txt _file/BPE 50000

# 2. 规定（相对）路径并试加载
./tokenizer_cli way ../../_file/BPE

# 3. 编码 / 解码 / 查看词表信息
./tokenizer_cli encode _file/BPE/vocab.json _file/BPE/merges.txt "Byte pair encoding 是一个分词算法！"
./tokenizer_cli decode _file/BPE/vocab.json _file/BPE/merges.txt 260 715 1234
./tokenizer_cli info   _file/BPE/vocab.json _file/BPE/merges.txt
```

---

## 接口

```cpp
namespace json_libs {

class Tokenizer {
public:
    bool load(const std::string& vocab_path, const std::string& merges_path);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string token_to_piece(int32_t id) const;
    size_t vocab_size() const;
    int32_t bos_id() const;         // <｜begin▁of▁sentence｜>
    int32_t eos_id() const;         // <｜end▁of▁sentence｜>
    int32_t unk_id() const;         // <unk>
    int32_t word_end_id() const;    // </w>
};

class BPETrainer {
public:
    explicit BPETrainer(int target_vocab_size = 50000);
    bool load_corpus(const std::string& corpus_path);
    void train();
    bool save(const std::string& output_dir) const;
    size_t vocab_size() const;
    size_t merge_count() const;
};

}
```

> 特殊 token 使用全角竖线（U+FF5C）与 U+2581，与 DeepSeek 系列模型的写法一致；
> 源码中以十六进制转义书写，避免不同编辑器编码差异。

### 函数与参数速查

主类（使用方）只要包含头文件即可：

```cpp
#include "trainer.h"   // 训练器 + 分词器（trainer.h 内部会 include tokenizer.h）
#include "tokenizer.h" // 只要分词器时
```

**Tokenizer** —— `include/tokenizer.h`


| 函数             | 参数                                                                                      | 返回                   | 说明                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `Tokenizer()`    | —                                                                                        | —                     | 构造空分词器，各特殊 token id 初始为 -1                                           |
| `load`           | `const std::string& vocab_path` 词表路径<br>`const std::string& merges_path` 合并规则路径 | `bool`                 | 读入 HF 风格的`vocab.json` + `merges.txt`；支持相对路径；失败会向 stderr 打印原因 |
| `load`           | —（使用`tokenizer_way()` 规定的路径）                                                    | `bool`                 | 无参重载；没有规定过路径时返回 false                                              |
| `encode`         | `const std::string& text` UTF-8 文本                                                      | `std::vector<int32_t>` | 文本 -> token id；文本中的特殊 token 会被直接识别；不会自动添加 BOS/EOS           |
| `decode`         | `const std::vector<int32_t>& ids`                                                         | `std::string`          | token id -> UTF-8 文本；去掉词尾标记并把字节字符还原为原始字节                    |
| `token_to_piece` | `int32_t id`                                                                              | `std::string`          | 返回词表中的原始 piece（空格为 U+0120，词尾带`</w>`）；越界返回空串               |
| `vocab_size`     | —                                                                                        | `size_t`               | 词表条目数                                                                        |
| `bos_id`         | —                                                                                        | `int32_t`              | 句首 token 的 id，不存在返回 -1                                                   |
| `eos_id`         | —                                                                                        | `int32_t`              | 句尾 token 的 id                                                                  |
| `unk_id`         | —                                                                                        | `int32_t`              | `<unk>` 的 id                                                                     |
| `word_end_id`    | —                                                                                        | `int32_t`              | 词尾标记`</w>` 的 id                                                              |
| `contains_token` | `const std::string& piece`                                                                | `bool`                 | 该 piece 是否在词表中                                                             |

**路径配置 `tokenizer_way`** —— `include/tokenizer.h`（自由函数，`namespace json_libs`）


| 函数                    | 参数                                                              | 返回                 | 说明                                                                       |
| ----------------------- | ----------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------- |
| `tokenizer_way`         | `const std::string& directory_path` 数据目录                      | `void`               | 规定目录，自动拼成`<dir>/vocab.json` 与 `<dir>/merges.txt`；传空串表示清空 |
| `tokenizer_way`         | `const std::string& vocab_path`、`const std::string& merges_path` | `void`               | 显式规定两个文件                                                           |
| `tokenizer_vocab_path`  | —                                                                | `const std::string&` | 查询已规定的词表路径                                                       |
| `tokenizer_merges_path` | —                                                                | `const std::string&` | 查询已规定的合并规则路径                                                   |

**相对路径规则**：路径允许写成相对形式，查找顺序为
`原样` → `../` → `../../` → `../../../`，因此无论从仓库根目录、`build/` 还是
`libs/tokenizer/build/` 运行，都能定位到同一份数据。

```cpp
// 主类里这样用：先规定相对路径，再无参加载
json_libs::tokenizer_way("../../_file/BPE");   // 目录形式，也可以用两个文件的版本

json_libs::Tokenizer tokenizer;
if (!tokenizer.load()) {                     // 无参 load()：使用上面规定的路径
    return 1;
}
```

**BPETrainer** —— `include/trainer.h`


| 函数          | 参数                                                         | 返回                              | 说明                                                          |
| ------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------- |
| `BPETrainer`  | `int target_vocab_size = 50000` 目标词表大小                 | —                                | 小于 260 会被抬到 260（基础词表大小）                         |
| `load_corpus` | `const std::string& corpus_path` 语料路径（每行一句，UTF-8） | `bool`                            | 把语料读入内存并按行切开                                      |
| `train`       | —                                                           | `void`                            | 构建基础词表 + 迭代合并，直到词表满或最高频 pair 出现次数 < 2 |
| `save`        | `const std::string& output_dir` 输出目录                     | `bool`                            | 写出`vocab.json` 与 `merges.txt`；目录不存在会自动创建        |
| `vocab_size`  | —                                                           | `size_t`                          | 当前词表大小                                                  |
| `merge_count` | —                                                           | `size_t`                          | 合并规则条数                                                  |
| `pieces`      | —                                                           | `const std::vector<std::string>&` | 调试用：id -> piece                                           |

**常量**（`namespace json_libs`）


| 常量            | 含义                                          |
| --------------- | --------------------------------------------- |
| `kBosToken`     | 句首特殊 token（`<｜begin▁of▁sentence｜>`） |
| `kEosToken`     | 句尾特殊 token（`<｜end▁of▁sentence｜>`）   |
| `kUnkToken`     | 未知 token`<unk>`                             |
| `kWordEndToken` | 词尾标记`</w>`                                |

**典型调用顺序**

```cpp
// 1) 训练（离线一次）
json_libs::BPETrainer trainer(50000);
trainer.load_corpus("_file/BPE/corpus.txt");
trainer.train();
trainer.save("_file/BPE");           // 产出 vocab.json + merges.txt

// 2) 主类里使用
json_libs::Tokenizer tokenizer;
if (!tokenizer.load("_file/BPE/vocab.json", "_file/BPE/merges.txt")) {
    return 1;
}
std::vector<int32_t> ids = tokenizer.encode("你好，世界！");
std::string back = tokenizer.decode(ids);   // back == 原文本
std::cout << tokenizer.vocab_size() << " " << tokenizer.bos_id() << std::endl;
```

---

## 算法要点

### 1. 编码：双向链表 + 优先队列（O(n log n)）

对每个词，把符号序列建成一条**双向链表**（数组模拟），把所有可合并的相邻符号对按
`(合并优先级, 位置)` 塞进**小顶堆**：

1. 弹出优先级最高的候选；
2. 若该位置已被合并（`alive == false`）或版本号过期（惰性失效），直接丢弃；
3. 否则把右节点并入左节点，更新链表指针，并使左节点及其左邻居的旧候选失效；
4. 把新产生的两个候选重新入堆。

每个候选最多入堆常数次、出堆一次，因此整体是 **O(n log n)**，而不是朴素实现的 O(n²)。

### 2. 训练：优先队列 + 倒排索引

- 预分词后统计唯一词频，符号对频次用哈希表维护；
- 用大顶堆取全局最高频符号对，配合**版本号惰性失效**避免惰性删除的开销；
- 维护「符号对 -> 包含它的词」倒排索引，只更新受影响的词；
- 停止条件：词表达到目标大小，或最高频符号对出现次数 `< 2`。

### 3. 字节级 + 词尾标记

- 输入按 UTF-8 解码，但每一步都落回**原始字节**（256 个字节字符，按 GPT-2 byte-level 规则映射成可打印字符）；
- 因此**任何字节序列都能无损往返**，不会出现 OOV；
- 每个词的符号序列末尾追加 `</w>` 作为词尾标记，解码时去掉。

### 4. 预分词规则

1. 连续空白整体成词；若恰好是 1 个空格且其后还有内容，则该空格并入后面的词（GPT-2 风格 `Ġ`）；
2. 每个标点字符单独成词（覆盖 ASCII 标点、常用西文标点、中文标点、全角标点）；
3. 其余字符（字母、数字、汉字等）连续成词。

### 5. JSON 复用

分词器不再自带 JSON 解析器，改为复用 `libs/json_libs/`（`json_libs::json_lib`）：

- `Tokenizer::load` 用 `json_lib` + `ForEachKey` 一趟读完 `vocab.json`（O(n)）；
- 文件读写、相对路径解析、UTF-8 编解码统一来自 `libs/json_libs/file_util.h`；
- 因此整个工程只有一份 JSON 解析实现，也没有 nlohmann 依赖。

---

## 文件格式

### vocab.json

```json
{
  "!": 33,
  "the</w>": 271,
  "<｜begin▁of▁sentence｜>": 257
}
```

- 顶层是 `{"token": id}` 形式的对象；
- 前 256 个 id 是 256 个字节字符，256 是 `</w>`，257/258/259 是特殊 token，其后是合并产生的新 token。

### merges.txt

```
#version: 0.2
t h
th e</w>
...
```

每行一条合并规则，按训练时的合并顺序排列（越靠前优先级越高）。

---

## 已知限制

- 预分词规则为简化版，与 GPT-2 原版正则不完全等价（不处理 `'s`、数字切分等特例）；
- 若正文中真的出现字面量 `</w>`（4 个 ASCII 字符），解码时可能与词尾标记混淆——这与
  subword-nmt 的机制一致，实际语料基本不会遇到；
- 训练器把整个语料读入内存，超大数据集需要改用流式统计。


当然BPE是AI写的，因为我没时间，我要做优化
