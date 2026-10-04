#ifndef JIEYU_LIBS_HI
#define JIEYU_LIBS_HI

// ============================================================
//  libs/libs.h —— 一站式入口头
//
//  主类里只写一行就能拿到 libs/ 下的全部能力：
//      #include "libs/libs.h"
//
//  引入内容：
//      libs/setup.h                      工程/目录初始化（create_config_file / setup_way）
//      libs/model_libs/creater_model.h   模型目录创建（create_model / create_bfile）
//      libs/model_libs/install_model.h   模型装载（install_model）
//      libs/json_libs/json.h             JSON 读写（json_libs::json_lib）
//      libs/tokenizer_libs/include/tokenizer.h
//                                       BPE 分词器（json_libs::Tokenizer / tokenizer_way）
//      libs/tokenizer_libs/include/trainer.h
//                                       BPE 训练器（json_libs::BPETrainer）
//      libs/matmul_libs/matmul.h         matmul / matmul_choose
//      libs/mapp_libs/Windows_InMapp.h   内存映射（load_model_map / free_model_map）
//      libs/date_libs/date.h             ModelInfo 结构体
//      libs/date_libs/check_cpu.h        CPU 指令集检测（cpu_can / OutCPUInfo）
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

#include "setup.h"
#include "model_libs/creater_model.h"
#include "model_libs/install_model.h"
#include "json_libs/json.h"
#include "tokenizer_libs/include/tokenizer.h"
#include "tokenizer_libs/include/trainer.h"
#include "matmul_libs/matmul.h"
#include "mapp_libs/Windows_InMapp.h"
#include "date_libs/check_cpu.h"
#endif  // JIEYU_LIBS_HI
