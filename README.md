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
│   │   ├── include/            tokenizer.h / trainer.h / byte_vocab.h（复用 json_libs）
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
│   │   ├── bfile.h            model.bf v2 格式：头部 / 张量目录 / 读写（含流式写）
│   │   ├── creater_model.h     生成空的 model.bf（只有头部）
│   │   ├── install_model.h     装载 model.bf（mmap + 校验 + 挂上张量表）
│   │   ├── safetensors.h       读 HuggingFace safetensors（复用 json_libs 解析 header）
│   │   ├── converter.h         safetensors -> model.bf（含造演示模型）
│   │   ├── load_weights.h      按名字取权重指针，填进 layers[]
│   │   ├── forward.h           多层串联 + 最终 RMSNorm
│   │   ├── generate.h          自回归生成（采样 + EOS 停止）
│   │   ├── pipeline.h          端到端：文本 -> token -> 生成 -> 文本
│   │   ├── final_norm.h        逐行 RMSNorm（单独抽出）
│   │   ├── lm_head.h           最后一行 * unembedding
│   │   └── sampler.h           argmax / top-k 采样
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
| `check_bfile` | 18 | `model.bf` v2 头部写入 / 异常分支 / mmap 读回 / `install_model` |
| `check_tensor_dir` | 11 | 张量目录：偏移 64 对齐、按名字取数、越界/损坏文件被拦住 |
| `check_ffn` | 8 | SwiGLU（double 参考 + 退化形状 + 哨兵 + 分发对比） |
| `check_rope` | 6 | RoPE（double 参考 + 范数守恒 + 90° 手算） |
| `check_rmsnorm` | 6 | RMSNorm（double 参考 + 输出 RMS=1 不变量） |
| `check_attention` | 7 | MHA / GQA / MQA + 因果性 + 数值稳定 |
| `check_transformer_layer` | 7 | 整层端到端（double 参考 + 残差恒等） |
| `check_generate` | 5 | 自回归生成（逐 token 对比 double 参考 + EOS 停止） |
| `check_convert` | 12 | safetensors → model.bf：转置、f16/bf16 转换、分片合并、权重共享 |
| `check_bind_weights` | 9 | 按名字填 `layers[]`、生成对齐 double 参考、形状写反能被拦住 |
| `check_pipeline` | 10 | 文本进文本出：分词往返（中/英）+ 端到端生成结果断言 |
| **合计** | **119** | 全部通过时退出码为 0 |

## 命令行用法

```bash
./jieyu.exe                                # 跑全部自测
./jieyu.exe --demo                         # 造演示小模型：转换 + 文本进文本出
./jieyu.exe --convert model.safetensors -o models/qwen2-0.5b
./jieyu.exe --convert <HuggingFace 模型目录> -o models/qwen2-0.5b
./jieyu.exe --show models/qwen2-0.5b/model.bf     # 打印张量目录
./jieyu.exe --gen models/qwen2-0.5b "你好" 64 0.8 # 提示词 / 最多生成 / 温度
./jieyu.exe --chat models/qwen2-0.5b 64 0.8       # 交互式，:q 退出
```

`--convert` 会自动找同目录的 `config.json` 和 `vocab.json` / `merges.txt`，也可用
`--config` / `--tokenizer` 显式指定。分片模型（`model-00001-of-00003.safetensors`）
可以一次传多个文件，名字冲突会报错。

`--demo` 不需要任何模型：它现场造一个小模型（层权重全 0，`lm_head` 设计成输出「上一个字节 + 1」），
走完 **safetensors → model.bf → 分词 → 生成 → 解码** 全流程，输出可读、可断言：

```
提示词  abcdefgh
生成    ijklmnopqrstuvwx
```

## model.bf 文件格式（v2）

小端；头部 64 字节，后面是张量目录区，最后是 64 字节对齐的数据区。

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
| 38..41 | version（v2 = 2；老文件是 0） | uint32 |
| 42..45 | tensor_count | uint32 |
| 46..49 | data_off（数据区起点，64 对齐） | uint32 |
| 50..53 | entry_size（= 104） | uint32 |
| 54..63 | 零填充 | — |
| 64..data_off | 张量目录：`tensor_count` 项，每项 104 字节 | — |
| data_off..文件尾 | 数据区：每个张量 64 字节对齐 | — |

目录项 104 字节：

| 偏移 | 字段 | 类型 |
| --- | --- | --- |
| 0..63 | 张量名（nul 结尾，最长 63 字符） | char[64] |
| 64 | dtype（0=f32 1=f16 2=bf16 3=i8 4=i32） | uint8 |
| 65 | ndim（最多 4） | uint8 |
| 68..83 | shape[4] | uint32 ×4 |
| 88..95 | offset（相对数据区） | uint64 |
| 96..103 | nbytes | uint64 |

- 没有张量的空模型就是 64 字节的头；老格式（version = 0）也能被认出，只是没有目录
- 装载时 `install_model` 会把 mmap 指针包成 `bfile::View` 挂在张量表上，
  之后用 `find_tensor("名字")` 就能拿到指针，全程零拷贝
- 配套 `model_info.json` 提供同样字段，另外还有 `rope_theta` 给 RoPE 用

## 权重布局（很容易踩的坑）

本工程的 `matmul` 语义是 $C_{[m,n]} = A_{[m,k]} \times B_{[k,n]}$，也就是 **B 要按 `[k, n]` 存**；
而 HuggingFace 的 `nn.Linear` 权重是 `[out_features, in_features] = [n, k]`。

所以**转换器会把所有二维权重转置**：

| 张量 | HF 形状 | model.bf 里的形状 |
| --- | --- | --- |
| `model.embed_tokens.weight` | `[vocab, hidden]` | 不动（查表用） |
| `self_attn.{q,k,v}_proj.weight` | `[q_dim, hidden]` | `[hidden, q_dim]` |
| `self_attn.o_proj.weight` | `[hidden, q_dim]` | `[q_dim, hidden]` |
| `mlp.gate_proj.weight` / `up_proj` | `[inter, hidden]` | `[hidden, inter]` |
| `mlp.down_proj.weight` | `[hidden, inter]` | `[inter, hidden]` |
| `lm_head.weight` | `[vocab, hidden]` | `[hidden, vocab]` |
| `*layernorm.weight` / `model.norm.weight` | `[hidden]` | 不动 |

`bind_model()` 会按这张表逐项校验形状，转置漏了会直接报错并打印实际/期望形状，不会静默算错。

## 核心实现

- **字节级 BPE**：UTF-8 按字节拆分并映射为可打印字符，任何字节序列都能无损往返，不存在 OOV；编码用双向链表 + 优先队列 + 版本号惰性失效做到 O(n log n)
- **手写 JSON**：简化递归下降解析器，支持 `\uXXXX` 与代理对；写入时 2 空格自动格式化
- **matmul 指令集分发**：运行时按 CPU 能力选 AVX+FMA / FMA / AVX / 朴素（`__builtin_cpu_supports` 探测）
- **完整 Transformer 算子**：RMSNorm、RoPE、因果 GQA 注意力（减最大值 softmax）、SwiGLU FFN、整层与多层串联
- **模型装载**：`model.bf` 走 Windows 内存映射，头部校验后按偏移读出各超参；v2 张量目录让权重可以直接按名字查，零拷贝
- **safetensors 转换器**：纯 C++（不用 Python），自己解析 safetensors 的 JSON 头；转置、dtype 转换（f16/bf16 → f32）都在转换时做掉，不转置的张量分块流式读写，内存只占一块缓冲区
- **自回归生成**：embed 查表 → 整层前向 → `lm_head` → argmax / 温度采样 → 追加 token，支持 EOS 提前停止
- **端到端管道**：`encode -> generate -> decode`，逐 token 解码并处理跨 token 的 UTF-8 多字节字符，不会被截断成乱码
- **header-only**：没有 `src/*.cpp`，全部实现写在头文件里

## 已知限制

- **RoPE 用的是相邻对风格**（`(2i, 2i+1)`，GPT-J / 原始 RoPE）；Llama / Qwen 权重用的是 NeoX 半分裂风格。两者不匹配时不会报错，只会静默算错，接入真实权重前必须确认
- **没有 KV cache**：注意力每步都重算整段序列，生成第 n 个 token 的代价是 $O(n^2)$，只能跑短上下文
- **引擎只读 f32**：转换器支持写出 f16/bf16（格式里也有 dtype 字段），但 matmul 还没写对应分支
- **转置要占内存**：单个二维权重转置时需要 `4 × 元素数` 的临时内存，转一个 0.5B 模型的 `lm_head` 大约 500 MB
- **RoPE / RMSNorm / Attention 目前只有经典实现**，没有 AVX 版本（matmul / ffn 已有）
- 每层前向都会分配临时缓冲区，多层推理时开销明显

详细的分词器文档见 [`libs/tokenizer_libs/README.md`](libs/tokenizer_libs/README.md)，
JSON 模块见 [`libs/json_libs/json文档.md`](libs/json_libs/json文档.md)。
