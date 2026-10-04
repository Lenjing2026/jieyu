#ifndef INSTALL_MODEL_H
#define INSTALL_MODEL_H
#include <bits/stdc++.h>
#include <filesystem>
#include "../json_libs/json.h"
#include "../mapp_libs/Windows_InMapp.h"
#include "../date_libs/date.h"
using namespace json_libs;
using namespace std;
using namespace std::filesystem;
void install_model(path model_path) {
    path dir = model_path;
    path bf_path = dir / "model.bf";
    path info_path = dir / "model_info.json";
    char magic[7];
    magic[6] = '\0';
    if (!exists(info_path))
        throw std::runtime_error("model_info.json not found");
    if (!exists(bf_path))
        throw std::runtime_error("model.bf not found");
    load_model_map(dir);
    model.model_vector=bf_path;
    model.model_info=info_path;
    model.model_tokenizer=dir / "vocab.json";
    std::ifstream info_file(info_path);
    if (!info_file.is_open())
        throw std::runtime_error("model_info.json open failed");
    memcpy(magic, model.model_map, 6);
    magic[6] = '\0';
    if (strcmp(magic, "JYAIBF") != 0)
        throw std::runtime_error("model.bf magic not match");
    model.h = *reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 6);
    model.l = *reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 14);
    model.mode = *reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 22);
    model.num_heads = *reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 23);
    model.num_kv_heads = *reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 24);
    model.head_dim = *reinterpret_cast<uint8_t*>(static_cast<unsigned char*>(model.model_map) + 25);
    model.intermediate_size = *reinterpret_cast<uint32_t*>(static_cast<unsigned char*>(model.model_map) + 26);
    model.vocab_size = *reinterpret_cast<uint64_t*>(static_cast<unsigned char*>(model.model_map) + 30);
    info_file.close();
}

#endif  // INSTALL_MODEL_H