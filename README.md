# JieYuAI

AI 基础组件实验工程（纯 C++17，无第三方依赖）。

## 目录结构

```
JieYuAI/
├── main.cpp                主类（入口程序）
├── README.md               本文件
├── libs/
│   ├── json_libs/          JSON 读写（header-only，拆成 4 个头）
│   │   ├── file_util.h     文件读写 / 相对路径解析 / UTF-8 / 按行切分
│   │   ├── json_value.h    JSON 值模型 + 序列化（2 空格自动格式化）
│   │   ├── json_parser.h   递归下降解析器 + 类型自动识别
│   │   ├── json.h          json_libs::json_lib 类（SetJsonWay / GetJsonDate / WriteJsonKey）
│   │   └── json文档.md      JSON 模块使用文档（接口 / 取值 / 写入 / 示例）
│   └── tokenizer/          字节级 BPE 分词器（header-only）
│       ├── include/        tokenizer.h / trainer.h（复用 json_libs，无重复实现）
│       └── README.md       分词器详细文档（含函数与参数速查）
└── _file/
    └── BPE/                训练数据目录
        ├── corpus.txt      输入语料（每行一句）
        ├── vocab.json      输出词表
        └── merges.txt      输出合并规则
```

> 分词器是 **header-only**：没有 `src/*.cpp`，主类只要
> `#include "tokenizer.h"` 或 `#include "trainer.h"` 就能直接调用。

## 编译

不用 CMake，一条 g++ 命令即可（全部 header-only）：

```bash
# 在 JieYuAI 根目录执行
g++ -std=c++17 -O2 -Wall -Wextra -I. -Ilibs/tokenizer/include -Ilibs/json_libs main.cpp -o app
```

- `-I.`：让 main.cpp 可以写 `#include "libs/json_libs/json.h"`
- `-Ilibs/json_libs`：也可以直接写 `#include "json.h"`
- `-Ilibs/tokenizer/include`：`#include "tokenizer.h"` / `#include "trainer.h"`
- tokenizer.h 内部用相对路径包含 json_libs，所以它自己不需要额外的 `-I`
- 已验证工具链：g++ 16.2.0（WinLibs MinGW-w64 UCRT）；调试用 `-g` 替换 `-O2`

## 使用

```bash
# 1. 训练：语料每行一句（Windows 下为 .\tokenizer_cli.exe）
./tokenizer_cli train _file/BPE/corpus.txt _file/BPE 50000

# 2. 查看词表信息
./tokenizer_cli info _file/BPE/vocab.json _file/BPE/merges.txt

# 3. 规定（相对）路径并试加载
./tokenizer_cli way ../../_file/BPE

# 4. 编码 / 解码
./tokenizer_cli encode _file/BPE/vocab.json _file/BPE/merges.txt "Byte pair encoding 是一个分词算法！"
./tokenizer_cli decode _file/BPE/vocab.json _file/BPE/merges.txt 260 715 1234
```

## 核心特性

- **字节级 BPE**：UTF-8 按字节拆分并映射为可打印字符，任何字节序列都能无损往返，不存在 OOV。
- **O(n log n) 编码**：双向链表 + 优先队列 + 版本号惰性失效，避免朴素实现的 O(n²)。
- **手写 JSON**：`libs/json_libs` 内含简化递归下降 JSON 解析器（无第三方依赖），分词器直接复用。
- **HuggingFace 兼容**：输出 `vocab.json`（`{"token": id}`）与 `merges.txt`（首行 `#version: 0.2`）。
- **特殊 token**：`<｜begin▁of▁sentence｜>`、`<｜end▁of▁sentence｜>`、`<unk>`，并在编码时自动识别。

详细设计请见 [`libs/tokenizer/README.md`](libs/tokenizer/README.md)。
