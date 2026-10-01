# ============================================================
#  JieYuAI 顶层构建脚本（CMake >= 3.15, C++17）
#
#  构建（Windows / MinGW，本机没装 VS，必须显式指定生成器）：
#    cmake -S . -B build -G "MinGW Makefiles"
#    cmake --build build -j
#
#  目标：
#    jieyu                main.cpp 编出的可执行文件（直接产出到工程根目录 jieyu.exe）
#    json_libs            头文件库（libs/json_libs，INTERFACE）
#    tokenizer_headers    头文件库（libs/tokenizer/include，INTERFACE）
#    jieyu_headers        自有头（libs/setup.h、libs/model_libs/*）
#
#  说明：所有库都是 header-only，INTERFACE 目标只负责向下游传递 include 路径
# ============================================================
cmake_minimum_required(VERSION 3.15)

project(JieYuAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# ------------------------------------------------------------
# 头文件库
# ------------------------------------------------------------

# JSON 读写：libs/json_libs/*.h（namespace json_libs）
add_library(json_libs INTERFACE)
target_include_directories(json_libs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/json_libs
)

# BPE 分词器：libs/tokenizer/include/*.h（内部用相对路径包含 json_libs）
add_library(tokenizer_headers INTERFACE)
target_include_directories(tokenizer_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/tokenizer/include
)
target_link_libraries(tokenizer_headers INTERFACE json_libs)

# 自有头文件：libs/setup.h、libs/model_libs/creater_model.h
#   暴露两个路径，使 #include "setup.h" 与 #include "libs/setup.h" 都能用
add_library(jieyu_headers INTERFACE)
target_include_directories(jieyu_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)
target_link_libraries(jieyu_headers INTERFACE json_libs tokenizer_headers)

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
add_executable(jieyu main.cpp)

# 让可执行文件直接产出到工程根目录，而不是藏在 build/ 里
# （RUNTIME_OUTPUT_DIRECTORY 用于单配置生成器，带 _XXX 的用于 Visual Studio 这类多配置生成器）
set_target_properties(jieyu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(jieyu PRIVATE
    jieyu_headers
    json_libs
    tokenizer_headers
)

if(MSVC)
    target_compile_options(jieyu PRIVATE /utf-8 /W4)
else()
    target_compile_options(jieyu PRIVATE -Wall -Wextra)
endif()

# GCC 8 及更早版本使用 std::filesystem 需要显式链接 stdc++fs
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(jieyu PRIVATE stdc++fs)
endif()
