#ifndef CHECK_CPU_H
#define CHECK_CPU_H
#include <cstdio>
using namespace std;
struct cpu_dat{
    bool
    mmx,sse,sse2,sse3,
    ssse3,sse4_1,sse4_2,avx,
    avx2,fma3,f16c,avx512f,avx512bw,
    avx512dq,avx512vnni,avx512vl,
    avx512bf16,amx,avxvnni,avx10;
    void GetCpuCan(){
        mmx=__builtin_cpu_supports("mmx");
        sse=__builtin_cpu_supports("sse");
        sse2=__builtin_cpu_supports("sse2");
        sse3=__builtin_cpu_supports("sse3");
        ssse3=__builtin_cpu_supports("ssse3");
        sse4_1=__builtin_cpu_supports("sse4.1");
        sse4_2=__builtin_cpu_supports("sse4.2");
        avx=__builtin_cpu_supports("avx");
        avx2=__builtin_cpu_supports("avx2");
        fma3=__builtin_cpu_supports("fma");
        f16c=__builtin_cpu_supports("f16c");
        avx512f=__builtin_cpu_supports("avx512f");
        avx512bw=__builtin_cpu_supports("avx512bw");
        avx512dq=__builtin_cpu_supports("avx512dq");
        avx512vnni=__builtin_cpu_supports("avx512vnni");
        avx512vl=__builtin_cpu_supports("avx512vl");
        avx512bf16=__builtin_cpu_supports("avx512bf16");
        avxvnni=__builtin_cpu_supports("avxvnni");
    }
} cpu_can;
void OutCPUInfo(cpu_dat cpu){
    printf("CPU支持的指令集:\n");
    printf("MMX: %s\n", cpu.mmx ? "支持" : "不支持");
    printf("SSE: %s\n", cpu.sse ? "支持" : "不支持");
    printf("SSE2: %s\n", cpu.sse2 ? "支持" : "不支持");
    printf("SSE3: %s\n", cpu.sse3 ? "支持" : "不支持");
    printf("SSSE3: %s\n", cpu.ssse3 ? "支持" : "不支持");
    printf("SSE4.1: %s\n", cpu.sse4_1 ? "支持" : "不支持");
    printf("SSE4.2: %s\n", cpu.sse4_2 ? "支持" : "不支持");
    printf("AVX: %s\n", cpu.avx ? "支持" : "不支持");
    printf("AVX2: %s\n", cpu.avx2 ? "支持" : "不支持");
    printf("FMA3: %s\n", cpu.fma3 ? "支持" : "不支持");
    printf("F16C: %s\n", cpu.f16c ? "支持" : "不支持");
    printf("AVX-512F: %s\n", cpu.avx512f ? "支持" : "不支持");
    printf("AVX-512BW: %s\n", cpu.avx512bw ? "支持" : "不支持");
    printf("AVX-512DQ: %s\n", cpu.avx512dq ? "支持" : "不支持");
    printf("AVX-512VNNI: %s\n", cpu.avx512vnni ? "支持" : "不支持");
    printf("AVX-512VL: %s\n", cpu.avx512vl ? "支持" : "不支持");
    printf("AVX-512BF16: %s\n", cpu.avx512bf16 ? "支持" : "不支持");
    printf("AMX:无法检查的指令集\n");
    printf("AVX-VNNI: %s\n", cpu.avxvnni ? "支持" : "不支持");
    printf("AVX10:无法检查的指令集\n");
    //我的gcc版本是13.2.0，无法检查AMX和AVX10指令集，可能是因为这些指令集较新，gcc还没有更新支持。
}
#endif
//手废了，啊啊啊啊啊啊啊啊啊啊妈咪啊妈咪妈妈吗咪咪咪