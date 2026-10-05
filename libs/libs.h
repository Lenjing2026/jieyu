#ifndef JIEYU_LIBS_HI
#define JIEYU_LIBS_HI

// ============================================================
//  libs/libs.h —— 一站式入口头
//
//  主类里只写一行就能拿到 libs/ 下的全部能力：
//      #include "libs/libs.h"
//
//  基础
//      libs/setup.h                        工程/目录初始化（create_config_file / setup_way）
//      libs/date_libs/date.h               ModelInfo / LayerWeights / RoPETable / __KVcache
//      libs/date_libs/check_cpu.h          cpu_can / OutCPUInfo
//      libs/mapp_libs/Windows_InMapp.h     load_model_map / free_model_map
//      libs/matmul_libs/matmul.h           matmul（按 CPU 指令集分发）
//      libs/matmul_libs/matmul_f16.h       f16 权重 kernel
//      libs/matmul_libs/matmul_q.h         q4_K / q5_K / q6_K / q8_0 权重 kernel（qmat）
//      libs/json_libs/json.h               json_libs::json_lib
//      libs/tokenizer_libs/include/tokenizer.h    json_libs::Tokenizer / tokenizer_way
//      libs/tokenizer_libs/include/trainer.h      json_libs::BPETrainer
//      libs/tokenizer_libs/include/byte_vocab.h   写 256 项字节级词表
//
//  模型（model.bf v2）
//      libs/model_libs/bfile.h             格式：写文件 / 张量目录 / View
//      libs/model_libs/creater_model.h     建模型目录、建空 model.bf
//      libs/model_libs/install_model.h     install_model（mmap + 校验 + 挂张量表）
//      libs/model_libs/safetensors.h       读 HuggingFace safetensors
//      libs/model_libs/converter.h         safetensors -> model.bf
//      libs/model_libs/load_weights.h      按名字取权重指针（bind_layers / bind_biases ...）
//      libs/model_libs/wmat_kernel.h       matmul_w（WMat 三路分派）/ wmat_row（取一行）
//      libs/model_libs/prefetch.h          权重预读（PrefetchVirtualMemory，让盘和算并行）
//      libs/model_libs/pipeline.h          端到端：文本进文本出（namespace pipeline）
//
//  前向 / 生成
//      libs/model_libs/silu|rmsnorm|rope|attention|ffn
//      libs/model_libs/transformer_layer.h / forward.h / final_norm.h / lm_head.h
//      libs/model_libs/sampler.h / generate.h
//
//  GGUF
//      libs/gguf_libs/gguf.h               读 GGUF v2/v3 + 反量化（q4_K / q5_K / q6_K ...）
//      libs/gguf_libs/gguf_convert.h       GGUF -> model.bf（含 q/k 置换、词表抽取）
//
//  自测（不在这里，单独加）
//      libs/text/text.h                    check_matmul / check_kv_cache / ...
//
//  注意：windows.h 必须排在任何 "using namespace std;" 之前，
//        否则 C++17 的 std::byte 会与 Windows rpcndr.h 的 byte 撞名，
//        报 "reference to 'byte' is ambiguous"。
// ============================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// 这些标准头是为了让本文件不依赖调用方先 include 别的：matmul.h 用了 memset，
// gguf.h / bfile.h 用了 cstdint、string、vector。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "setup.h"

#include "date_libs/date.h"
#include "date_libs/check_cpu.h"
#include "mapp_libs/Windows_InMapp.h"
#include "matmul_libs/matmul.h"
#include "matmul_libs/matmul_f16.h"
#include "matmul_libs/matmul_q.h"

#include "json_libs/json.h"
#include "tokenizer_libs/include/tokenizer.h"
#include "tokenizer_libs/include/trainer.h"
#include "tokenizer_libs/include/byte_vocab.h"

#include "model_libs/bfile.h"
#include "model_libs/creater_model.h"
#include "model_libs/install_model.h"
#include "model_libs/safetensors.h"
#include "model_libs/converter.h"
#include "model_libs/load_weights.h"
#include "model_libs/wmat_kernel.h"
#include "model_libs/prefetch.h"

#include "model_libs/silu/silu.h"
#include "model_libs/rmsnorm/rmsnorm.h"
#include "model_libs/rope/rope.h"
#include "model_libs/attention/attention.h"
#include "model_libs/ffn/ffn.h"
#include "model_libs/transformer_layer.h"
#include "model_libs/forward.h"
#include "model_libs/final_norm.h"
#include "model_libs/lm_head.h"
#include "model_libs/sampler.h"
#include "model_libs/generate.h"
#include "model_libs/pipeline.h"

#include "gguf_libs/gguf.h"
#include "gguf_libs/gguf_convert.h"
#endif  // JIEYU_LIBS_HI
