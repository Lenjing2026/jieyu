# JieYuAI

从零手写的 AI 推理框架（纯 C++17，header-only，无第三方依赖）。

从字节级 BPE 分词、JSON、二进制模型文件，到 Transformer 各算子（RMSNorm / RoPE / GQA / SwiGLU）
和整层前向，全部自己实现。

## 目录结构

```
JieYuAI/
├── main.cpp                    入口：初始化 + 跑全部自测
├── CMakeLists.txt
├── jieyu.exe                   构建产物（直接产出到根目录）
├── libs/
│   ├── libs.h                  一站式入口头（include 这个就够了）
│   ├── setup.h                 工程初始化（create_config_file / setup_way）
│   ├── json_libs/              JSON 读写（手写递归下降解析器，拆成 4 个头）
│   │   ├── file_util.h         文件读写 / 相对路径解析 / UTF-8 / 按行切分
│   │   ├── json_value.h        JSON 值模型 + 序列化（2 空格自动格式化）
│   │   ├── json_parser.h       递归下降解析器 + 类型自动识别
│   │   ├── json.h              json_libs::json_lib 类
│   │   └── json文档.md          JSON 模块使用文档
│   ├── tokenizer_libs/         字节级 BPE 分词器 + 训练器
│   │   ├── include/            tokenizer.h / trainer.h（复用 json_libs）
│   │   └── README.md           分词器详细文档
│   ├── matmul_libs/            matmul（朴素 / AVX / FMA / AVX+FMA 运行时分发）
│   ├── mapp_libs/              Windows 内存映射（load_model_map / free_model_map）
│   ├── date_libs/              date.h（ModelInfo / LayerWeights / RoPETable）、check_cpu.h
│   ├── model_libs/             模型层
│   │   ├── silu/               SiLU 激活
│   │   ├── ffn/                SwiGLU 前向
│   │   ├── rmsnorm/            RMSNorm
│   │   ├── rope/               RoPE 旋转位置编码
│   │   ├── attention/          因果 GQA 注意力
│   │   ├── transformer_layer.h 单个 block 前向（9 步）
│   │   ├── creater_model.h     生成 model.bf
│   │   ├── install_model.h     装载 model.bf（mmap + 校验 + 读头部）
│   │   └── forward.h           多层串联 + 最终 RMSNorm
│   └── text/text.h             全部自测函数
└── _file/                      数据目录（BPE 语料 / 模型文件 / 测试产物）
    └── BPE/                    corpus.txt 输入，vocab.json + merges.txt 输出
```

## 构建

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build -j
```

- 本机没装 VS，**必须**显式指定 MinGW 生成器
- `cmake --build build -j` 才是编译；只写 `cmake build` 只是重新配置，不会编译
- 产物 `jieyu.exe` 直接落到工程根目录（`RUNTIME_OUTPUT_DIRECTORY`）
- 全部 header-only，不用 CMake 时一条 g++ 命令也行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I. -Ilibs -Ilibs/json_libs -Ilibs/tokenizer_libs/include main.cpp -o jieyu
```

已验证工具链：g++ 16.2.0（WinLibs MinGW-w64 UCRT）+ CMake 4.4.3。

## 运行自测

```bash
./jieyu.exe
```

运行即执行全部自测（每个模块都拿 double 精度参考实现逐元素比对）：

| 测试 | 项数 | 覆盖内容 |
| --- | --- | --- |
| `check_matmul` | 20 | 4 种 kernel 正确性、指令集分发一致性 |
| `check_bfile` | 17 | `model.bf` 写入 / 异常分支 / mmap 读回 / `install_model` |
| `check_ffn` | 8 | SwiGLU（double 参考 + 退化形状 + 哨兵 + 分发对比） |
| `check_rope` | 6 | RoPE（double 参考 + 范数守恒 + 90° 手算） |
| `check_rmsnorm` | 6 | RMSNorm（double 参考 + 输出 RMS=1 不变量） |
| `check_attention` | 7 | MHA / GQA / MQA + 因果性 + 数值稳定 |
| `check_transformer_layer` | 7 | 整层端到端（double 参考 + 残差恒等） |
| **合计** | **71** | 全部通过时退出码为 0 |

## model.bf 文件格式

小端，共 38 字节头部 + 零填充对齐到 64 字节：

| 偏移 | 字段 | 类型 |
| --- | --- | --- |
| 0..5 | 魔数 `JYAIBF` | char[6] |
| 6..13 | hidden_size | uint64 |
| 14..21 | layer_count | uint64 |
| 22 | mode | uint8 |
| 23 | num_heads | uint8 |
| 24 | num_kv_heads | uint8 |
| 25 | head_dim | uint8 |
| 26..29 | intermediate_size | uint32 |
| 30..37 | vocab_size | uint64 |
| 38..63 | 零填充 | — |

配套 `model_info.json` 提供上述字段；`layer_count` 必须是 16 的倍数，否则生成时抛 `NO16_LAYER`。

## 核心实现

- **字节级 BPE**：UTF-8 按字节拆分并映射为可打印字符，任何字节序列都能无损往返，不存在 OOV；编码用双向链表 + 优先队列 + 版本号惰性失效做到 O(n log n)
- **手写 JSON**：简化递归下降解析器，支持 `\uXXXX` 与代理对；写入时 2 空格自动格式化
- **matmul 指令集分发**：运行时按 CPU 能力选 AVX+FMA / FMA / AVX / 朴素（`__builtin_cpu_supports` 探测）
- **完整 Transformer 算子**：RMSNorm、RoPE、因果 GQA 注意力（减最大值 softmax）、SwiGLU FFN、整层与多层串联
- **模型装载**：`model.bf` 走 Windows 内存映射，头部校验后按偏移读出各超参
- **header-only**：没有 `src/*.cpp`，全部实现写在头文件里

## 已知限制

- **RoPE 用的是相邻对风格**（`(2i, 2i+1)`，GPT-J / 原始 RoPE）；Llama / Qwen 权重用的是 NeoX 半分裂风格。两者不匹配时不会报错，只会静默算错，接入外部权重前必须确认
- **没有 KV cache**：注意力每步都重算整段序列，只能做全量前向，还不支持逐 token 增量解码
- **RoPE / RMSNorm / Attention 目前只有经典实现**，没有 AVX 版本（matmul / ffn 已有）
- 每层前向都会分配临时缓冲区，多层推理时开销明显

详细的分词器文档见 [`libs/tokenizer_libs/README.md`](libs/tokenizer_libs/README.md)，
JSON 模块见 [`libs/json_libs/json文档.md`](libs/json_libs/json文档.md)。
