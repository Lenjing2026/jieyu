#ifndef CHECK_CPU_H
#define CHECK_CPU_H
#include<cstdio>
#include<cpuid.h>
using namespace std;
struct cpu_dat{
    bool
    mmx,sse,sse2,sse3,
    ssse3,sse4_1,sse4_2,avx,
    avx2,fma3,f16c,avx512f,avx512bw,
    avx512dq,avx512vnni,avx512vl,
    avx512bf16,amx,avxvnni,avx10;
    void GetCpuCan(){
        unsigned a=0,b=0,c=0,d=0;
        __cpuid(0,a,b,c,d);
        const unsigned max_leaf=a;
        __cpuid(1,a,b,c,d);
        mmx=(d>>23)&1u;
        sse=(d>>25)&1u;
        sse2=(d>>26)&1u;
        sse3=c&1u;
        ssse3=(c>>9)&1u;
        sse4_1=(c>>19)&1u;
        sse4_2=(c>>20)&1u;
        fma3=(c>>12)&1u;
        f16c=(c>>29)&1u;
        avx=(c>>28)&1u;
        const bool osxsave=((c>>27)&1u)!=0;
        unsigned a7=0,b7=0,c7=0,d7=0;
        if(max_leaf>=7) __cpuid_count(7,0,a7,b7,c7,d7);
        avx2=(b7>>5)&1u;
        avx512f=(b7>>16)&1u;
        avx512dq=(b7>>17)&1u;
        avx512bw=(b7>>30)&1u;
        avx512vl=(b7>>31)&1u;
        avx512vnni=(c7>>11)&1u;
        unsigned a71=0,b71=0,c71=0,d71=0;
        if(max_leaf>=7) __cpuid_count(7,1,a71,b71,c71,d71);
        avxvnni=(a71>>4)&1u;
        avx512bf16=(a71>>5)&1u;
        amx=(d71>>22)&1u;
        avx10=false;
        unsigned x0=0,x1=0;
        if(osxsave) __asm__ __volatile__("xgetbv":"=a"(x0),"=d"(x1):"c"(0));
        if(!osxsave||(x0&6u)!=6u){
            avx=false;avx2=false;fma3=false;f16c=false;
        }
        if(!osxsave||(x0&0xE6u)!=0xE6u){
            avx512f=false;avx512bw=false;avx512dq=false;
            avx512vnni=false;avx512vl=false;avx512bf16=false;
        }
    }
} cpu_can;
void OutCPUInfo(cpu_dat cpu){
    printf("CPU支持的指令集:\n");
    printf("MMX: %s\n",cpu.mmx?"支持":"不支持");
    printf("SSE: %s\n",cpu.sse?"支持":"不支持");
    printf("SSE2: %s\n",cpu.sse2?"支持":"不支持");
    printf("SSE3: %s\n",cpu.sse3?"支持":"不支持");
    printf("SSSE3: %s\n",cpu.ssse3?"支持":"不支持");
    printf("SSE4.1: %s\n",cpu.sse4_1?"支持":"不支持");
    printf("SSE4.2: %s\n",cpu.sse4_2?"支持":"不支持");
    printf("AVX: %s\n",cpu.avx?"支持":"不支持");
    printf("AVX2: %s\n",cpu.avx2?"支持":"不支持");
    printf("FMA3: %s\n",cpu.fma3?"支持":"不支持");
    printf("F16C: %s\n",cpu.f16c?"支持":"不支持");
    printf("AVX-512F: %s\n",cpu.avx512f?"支持":"不支持");
    printf("AVX-512BW: %s\n",cpu.avx512bw?"支持":"不支持");
    printf("AVX-512DQ: %s\n",cpu.avx512dq?"支持":"不支持");
    printf("AVX-512VNNI: %s\n",cpu.avx512vnni?"支持":"不支持");
    printf("AVX-512VL: %s\n",cpu.avx512vl?"支持":"不支持");
    printf("AVX-512BF16: %s\n",cpu.avx512bf16?"支持":"不支持");
    printf("AMX:无法检查的指令集\n");
    printf("AVX-VNNI: %s\n",cpu.avxvnni?"支持":"不支持");
    printf("AVX10:无法检查的指令集\n");
}
#endif