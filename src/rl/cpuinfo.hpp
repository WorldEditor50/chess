#ifndef RL_CPUINFO_HPP
#define RL_CPUINFO_HPP
#include <string>
#include "simd_ops.hpp"

/*
 * RL::cpuinfo - 运行时 CPU 指令集探测。
 *
 * 为什么需要它: SIMD 内核是**编译期**选择的 (由 __AVX2__ / __SSE2__ 宏决定, 见
 * simd_ops.hpp), 而 MSVC 在 x64 下默认只定义 __SSE2__ —— 想用 AVX2 内核就必须给
 * 相关翻译单元加 /arch:AVX2。代价是: 这样编出来的二进制**要求** CPU 支持 AVX2,
 * 拿到不支持的机器上会直接非法指令崩溃。所以两边都要能回答"系统到底支持什么":
 *
 *   * 配置阶段: CMakeLists.txt 用 try_run 跑一次 CPUID 探测, 系统不支持就自动把
 *     CHESS_ENABLE_AVX2 关掉并给出提示 (避免构出会崩的二进制);
 *   * 运行阶段: 本文件提供 avx2Supported() 与 describe(), 让程序能报告"编译进来的
 *     是 AVX2 还是 SSE2、这台机器支不支持 AVX2", 而不是靠猜。
 *
 * 探测逻辑遵循 Intel 的手册顺序: 先看 CPUID 是否报告 AVX2, 再确认操作系统已经通过
 * XCR0 打开 YMM 状态 (OSXSAVE + AVX), 否则 AVX2 指令虽然"支持"但会 #UD。
 */
namespace RL {
namespace cpuinfo {

#if defined(_MSC_VER)
#include <intrin.h>
inline bool detectAvx2()
{
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return false;
    }
    /* leaf 1: OSXSAVE (ECX bit 27) 与 AVX (ECX bit 28) */
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }
    /* XCR0 bit1 (SSE) 与 bit2 (YMM) 必须都已置位 */
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL) {
        return false;
    }
    /* leaf 7 subleaf 0: EBX bit 5 = AVX2 */
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
}
#elif defined(__GNUC__) || defined(__clang__)
inline bool detectAvx2()
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}
#else
inline bool detectAvx2()
{
    return false;
}
#endif

/* 这台机器 (含操作系统对 YMM 状态的许可) 能不能跑 AVX2 */
inline bool avx2Supported()
{
    static const bool value = detectAvx2();
    return value;
}

/* SSE2 是所有 x86-64 的基线, 32 位 x86 上才需要探测 */
inline bool sse2Supported()
{
#if defined(_M_X64) || defined(__x86_64__)
    return true;
#elif defined(_M_IX86) || defined(__i386__)
    #if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    return (regs[3] & (1 << 26)) != 0;   /* EDX bit 26 = SSE2 */
    #else
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2") != 0;
    #endif
#else
    return false;
#endif
}

/* 本翻译单元实际编译进来的指令集 (由 simd_ops.hpp 的 #if 决定) */
inline const char *compiledSet()
{
    return simdops::instructionSet();
}

/* 编译进来的指令集与运行时支持情况是否自洽 */
inline bool activeIsUsable()
{
    const std::string s = compiledSet();
    if (s == "AVX2") {
        return avx2Supported();
    }
    if (s == "SSE2") {
        return sse2Supported();
    }
    return true;
}

/*
 * 一行人类可读的说明, 例如
 *   "AVX2 kernels active (CPU supports AVX2)"
 *   "AVX2 kernels active (!! CPU reports NO AVX2 - rebuild with -DCHESS_ENABLE_AVX2=OFF)"
 *   "SSE2 kernels active (CPU supports AVX2)"
 *
 * 注意 "编译进来的指令集" 是**按翻译单元**决定的: 目标没链 RL_CORE、也没有
 * /arch:AVX2 时, 它自己那套内联算符就是 SSE2/标量的 (例如 test_rules 只测棋规,
 * 根本不碰 Tensor)。所以这行字描述的是**当前这个二进制**, 不是整台机器。
 */
inline std::string describe()
{
    const std::string set = compiledSet();
    std::string out = set + " kernels active";
    if (set == "AVX2") {
        out += avx2Supported()
             ? " (CPU supports AVX2)"
             : " (!! CPU reports NO AVX2 - rebuild with -DCHESS_ENABLE_AVX2=OFF)";
    } else if (set == "SSE2") {
        out += avx2Supported() ? " (CPU supports AVX2)" : " (CPU has no AVX2)";
    } else {
        out += " (no SIMD compiled in)";
    }
    return out;
}

} /* namespace cpuinfo */
} /* namespace RL */

#endif // RL_CPUINFO_HPP
