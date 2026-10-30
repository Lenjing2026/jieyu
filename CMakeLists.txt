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

option(JIEYU_NO_O3 "不要强加 -O3" OFF)

# 所有产物统一放到工程根目录的 resut/ 里
set(JIEYU_OUT "${CMAKE_CURRENT_SOURCE_DIR}/resut")
file(MAKE_DIRECTORY "${JIEYU_OUT}")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/tokenizer_libs/include
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
add_executable(jieyu main.cpp
    libs/model_libs/silu/silu.h
    libs/model_libs/ffn/ffn.h
    libs/model_libs/forward.h
    libs/model_libs/final_norm.h
    libs/model_libs/lm_head.h
    libs/model_libs/generate.h)

# 让可执行文件直接产出到 resut/，而不是藏在 build/ 里
# （RUNTIME_OUTPUT_DIRECTORY 用于单配置生成器，带 _XXX 的用于 Visual Studio 这类多配置生成器）
set_target_properties(jieyu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${JIEYU_OUT}"
)

# 额外在工程根目录留一份 jieyu.exe（resut/ 里的保留，方便直接在根目录敲 ./jieyu）
if(WIN32)
    add_custom_command(TARGET jieyu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:jieyu>"
                "${CMAKE_CURRENT_SOURCE_DIR}/jieyu.exe"
        COMMENT "复制 jieyu.exe 到工程根目录"
    )
endif()

target_link_libraries(jieyu PRIVATE
    jieyu_headers
    json_libs
    tokenizer_headers
)

if(MSVC)
    target_compile_options(jieyu PRIVATE /utf-8 /W4 /openmp)
else()
    target_compile_options(jieyu PRIVATE -Wall -Wextra -fopenmp)
    target_link_options(jieyu PRIVATE -fopenmp)
    if(NOT JIEYU_NO_O3)
        target_compile_options(jieyu PRIVATE -O3 -DNDEBUG)
    endif()
endif()

# GCC 8 及更早版本使用 std::filesystem 需要显式链接 stdc++fs
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(jieyu PRIVATE stdc++fs)
endif()

# ------------------------------------------------------------
#  动态库（对外 C 接口，声明见 main.hpp）
#    main.hpp 直接当源文件编（CMake 里给它 LANGUAGE CXX，等价于 -x c++）
#    cmake --build build --target jieyu_shared
#  Windows/MinGW：jieyu.dll + libjieyu.dll.a
#  Linux：libjieyu.so
#  导出符号：jieyu_load / jieyu_free / jieyu_generate / jieyu_generate_jsonl
#            / jieyu_last_error / jieyu_version
# ------------------------------------------------------------
set_source_files_properties(main.hpp PROPERTIES LANGUAGE CXX)
if(MSVC)
    set_source_files_properties(main.hpp PROPERTIES COMPILE_OPTIONS "/TP")
else()
    set_source_files_properties(main.hpp PROPERTIES COMPILE_OPTIONS "-x;c++")
endif()
add_library(jieyu_shared SHARED main.hpp)
target_link_libraries(jieyu_shared PRIVATE
    jieyu_headers
    json_libs
    tokenizer_headers
)
target_compile_definitions(jieyu_shared PRIVATE MAIN_EXPORTS)

if(MSVC)
    target_compile_options(jieyu_shared PRIVATE /utf-8 /W4 /openmp)
else()
    target_compile_options(jieyu_shared PRIVATE -Wall -Wextra -fopenmp -fvisibility=hidden)
    target_link_options(jieyu_shared PRIVATE -fopenmp)
    if(NOT JIEYU_NO_O3)
        target_compile_options(jieyu_shared PRIVATE -O3 -DNDEBUG)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(jieyu_shared PRIVATE stdc++fs)
endif()

set_target_properties(jieyu_shared PROPERTIES
    OUTPUT_NAME "jieyu"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${JIEYU_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${JIEYU_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${JIEYU_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${JIEYU_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${JIEYU_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${JIEYU_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY "${JIEYU_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${JIEYU_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${JIEYU_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${JIEYU_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${JIEYU_OUT}"
)

# ------------------------------------------------------------
#  交叉编译 Linux 版（Zig + musl，静态链接，Windows 上也能直接出 ELF）
#    cmake --build build --target jieyu_linux          可执行
#    cmake --build build --target jieyu_linux_so       动态库
#  Zig 不在 PATH 里时：-DZIG_EXECUTABLE=D:/path/to/zig.exe
#  产物：resut/ 下的 jieyu（静态 ELF）与 jieyu.so（C 接口动态库）
# ------------------------------------------------------------
if(NOT ZIG_EXECUTABLE)
    find_program(ZIG_EXECUTABLE NAMES zig
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/Trush/zig")
endif()
if(ZIG_EXECUTABLE)
    add_custom_target(jieyu_linux ALL
        COMMAND "${ZIG_EXECUTABLE}" c++
                -target x86_64-linux-musl
                -O3 -DNDEBUG -s
                -std=c++17 -pthread
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/libs
                ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                -o ${JIEYU_OUT}/jieyu
        COMMENT "交叉编译 jieyu（x86_64-linux-musl，静态，std::thread 线程池）"
        VERBATIM)

    add_custom_target(jieyu_linux_so ALL
        COMMAND "${ZIG_EXECUTABLE}" c++
                -target x86_64-linux-musl
                -O3 -DNDEBUG -s
                -std=c++17 -pthread -fPIC -shared
                -fvisibility=hidden
                -DMAIN_EXPORTS
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/libs
                -x c++ ${CMAKE_CURRENT_SOURCE_DIR}/main.hpp
                -o ${JIEYU_OUT}/jieyu.so
        COMMENT "交叉编译 jieyu.so（x86_64-linux-musl，C 接口）"
        VERBATIM)
else()
    message(STATUS "没找到 Zig：目标 jieyu_linux / jieyu_linux_so 不可用（把 zig.exe 放到 Trush/zig 或传 -DZIG_EXECUTABLE=...）")
endif()
