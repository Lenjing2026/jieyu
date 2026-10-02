#ifndef JIEYU_LIBS_H
#define JIEYU_LIBS_H

// ============================================================
//  libs/libs.h —— 一站式入口头
//
//  主类里只写一行就能拿到 libs/ 下的全部能力：
//      #include "libs/libs.h"
//
//  引入内容：
//      libs/setup.h                      工程/目录初始化（create_config_file / setup_way）
//      libs/model_libs/creater_model.h   模型目录创建（create_modle）
//      libs/json_libs/json.h             JSON 读写（json_libs::json_lib）
//      libs/tokenizer/include/tokenizer.h
//                                       BPE 分词器（json_libs::Tokenizer / tokenizer_way）
//      libs/tokenizer/include/trainer.h
//                                       BPE 训练器（json_libs::BPETrainer）
// ============================================================

#include "setup.h"
#include "model_libs/creater_model.h"
#include "json_libs/json.h"
#include "tokenizer_libs/include/tokenizer.h"
#include "tokenizer_libs/include/trainer.h"
#include "matmul_libs/matmul.h"
#include "date_libs/check_cpu.h"
#endif  // JIEYU_LIBS_H
