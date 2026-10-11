# 解语

从零手写的 AI 推理框架（纯 C++17，header-only，无第三方依赖）。

从字节级 BPE 分词、JSON、二进制模型文件，到 Transformer 各算子（RMSNorm / RoPE / GQA / SwiGLU）
和整层前向，全部自己实现。  
使用前，该执行以下指令
```bash
jieyu.exe -setup
```

## 运行自测

```bash
jieyu.exe -test
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
| `check_kv_cache` | 9 | 分段算 == 整段算（逐元素）；cache 内容、current_len、越界与层数不符拦截 |
| `check_convert` | 12 | safetensors → model.bf：转置、f16/bf16 转换、分片合并、权重共享 |
| `check_bind_weights` | 9 | 按名字填 `layers[]`、生成对齐 double 参考、形状写反能被拦住 |
| `check_pipeline` | 10 | 文本进文本出：分词往返（中/英）+ 端到端生成结果断言 |
| **合计** | **128** | 全部通过时退出码为 0 |

## 命令行用法

```txt
  jieyu -run <名字> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]    装载 models/<名字> 并交互生成
  jieyu -run                                     列出 models/ 下能跑的模型
  jieyu -list                                    同上
  jieyu -info <名字>                             看 models/<名字>/model_info.json 里的作者/描述
  jieyu -download                                列出 download.json 里能下的模型
  jieyu -download <名字>                         下载到 models/<名字>/（curl / wget / PowerShell 自动挑）
  jieyu -setup                                   建目录/文件：config.json、langrage/、models/、_file/
  jieyu -langrage [名字|{json}]                  切语言（写进 config.json），不带参数就看当前和可用语言
  jieyu -demo [目录]                             造演示小模型，文本进文本出
  jieyu -convert <safetensors...|xxx.gguf> -o <目录> [-config f] [-tokenizer d] [-dtype quant|f32|f16]
  jieyu -gguf <xxx.gguf>                         看 GGUF：元数据 / 词表 / 量化分布 / 张量清单
  jieyu -show <model.bf>                         打印 model.bf 的张量目录
  jieyu -gen <模型目录> <提示词> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]
  jieyu -chat <模型目录> [最多生成] [温度] [最大长度] [重复惩罚] [-tokenspeed]
  jieyu -?                                       看这份用法
    （[最多生成] 省了或给 0 就是不限，生成到 <|im_end|> / 上下文满为止）
    （-tokenspeed 跟在 -run / -gen / -chat 后面：每次生成完分开打印预填充与解码两段速度，位置随便放）
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

## 已知限制

- **RoPE 用的是相邻对风格**（`(2i, 2i+1)`，GPT-J / 原始 RoPE）；Llama / Qwen 权重用的是 NeoX 半分裂风格。两者不匹配时不会报错，只会静默算错，接入真实权重前必须确认
- **没有 KV cache 之外的缓存策略**：不支持滑动窗口 / 分页注意力，`max_seq` 开多大就占多少内存（$2 \times L \times S \times H_{kv} \times D \times 4$ 字节，0.5B 模型 4K 上下文约 100 MB）
- **引擎只读 f32**：转换器支持写出 f16/bf16（格式里也有 dtype 字段），但 matmul 还没写对应分支
- **转置要占内存**：单个二维权重转置时需要 `4 × 元素数` 的临时内存，转一个 0.5B 模型的 `lm_head` 大约 500 MB
- **RoPE / RMSNorm / Attention 目前只有经典实现**，没有 AVX 版本（matmul / ffn 已有）
- 每层前向都会分配临时缓冲区（带 cache 后每步只有 1 行，开销很小）  
- 其他可见[Wiki](https://github.com/Lenjing2026/jieyu/wiki)
