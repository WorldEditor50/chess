/*
 * probe_alloc_main.cpp — [2026-10 P3 验收 1] `Tensor` 的**堆分配次数**探针
 * ============================================================================
 * 为什么要有它: 方案 Part 4 的 P3 验收第 1 条是
 *
 *     "探针量到『构造 3 次分配 → 1 次』"
 *
 * 而 A11（形状 SSO / 定长存储）到底省掉几次分配、省的是哪两次，必须先有**本机读数**
 * 才能谈"改完没有"。方案 §1.2 记的是旧机器上的读数（"(19,1) 构造 ≈ 947~1036 ns、
 * 恰好 3 次堆分配: shape 8 B + sizes 8 B + 32 B 对齐数据"），换机后 §5.1 已声明
 * 绝对读数不可比 —— 而**"几次分配"是结构事实，不是计时**，跨机可比。
 *
 * 口径（刻意只数次数，不数时间）:
 *   * 用 MSVC 的 `_CrtSetAllocHook` 钩住 **CRT 堆**：`std::vector<int>`（shape/sizes）
 *     走的 `::operator new` 与 `AlignedAllocator::allocate` 走的 `_aligned_malloc`
 *     都会经过它 ⇒ "总数"是完整的，不需要往 `src/rl` 里塞计数器；
 *   * ⚠ 钩子只在 **Debug CRT (`/MDd`)** 里存在 ⇒ 本探针必须在 **Debug 构建目录**里跑
 *     （`cmake -S . -B build/p3dbg -DCMAKE_BUILD_TYPE=Debug`）。Release 下钩子不生效，
 *     探针会**响亮退出 (exit 2)** 而不是打印一堆 0 骗人；
 *   * ⚠⚠ **Debug 构建必须带 `-D_ITERATOR_DEBUG_LEVEL=0`**（`build/p3dbg` 的配置命令里
 *     已经写死）：默认的 2 会让**每个 `std::vector` 多分配一块 16 B 的 `_Container_proxy`**，
 *     于是"一次 `Tensor(r,c)` 构造 = 3 次分配"会被读成 8 次 —— 那是 Debug 仪器的开销，
 *     不是被测对象的结构（Release 的 `_ITERATOR_DEBUG_LEVEL` 本来就是 0）。
 *     这条是**本探针第一次跑就踩到的坑**，记在这里免得下次又对着 8 发愣；
 *   * 每个用例都打印"分配次数 / 释放次数 / 字节数 / 前几次分配的字节序列" ——
 *     字节序列用来分辨"哪一次是 shape/sizes（8 B）、哪一次是数据（对齐后的缓冲）"。
 *
 * 运行:
 *   cmake -S . -B build/p3dbg -G Ninja -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 \
 *         -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="/D_ITERATOR_DEBUG_LEVEL=0"
 *   cmake --build build/p3dbg --target probe_alloc
 *   build/p3dbg/probe_alloc.exe
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rl/tensor.hpp"

#ifdef _DEBUG
#  include <crtdbg.h>
#endif

namespace {

/* ---- 计数状态 (只在一个"窗口"内累计) ---- */
bool        g_on = false;
long        g_allocs = 0;
long        g_frees = 0;
std::size_t g_bytes = 0;
std::size_t g_sizes[16];
int         g_sizeCount = 0;

#ifdef _DEBUG
int __cdecl allocHook(int allocType, void *, std::size_t size, int, long,
                      const unsigned char *, int)
{
    if (g_on) {
        if (allocType == _HOOK_ALLOC) {
            ++g_allocs;
            g_bytes += size;
            if (g_sizeCount < 16) { g_sizes[g_sizeCount++] = size; }
        } else if (allocType == _HOOK_FREE) {
            ++g_frees;
        }
    }
    return 1;   /* 1 = 让这次分配照常进行 */
}
#endif

void beginCount()
{
    g_allocs = 0;
    g_frees = 0;
    g_bytes = 0;
    g_sizeCount = 0;
    g_on = true;
}

struct Count { long allocs; long frees; std::size_t bytes; std::string sizes; };

Count endCount()
{
    g_on = false;
    Count c;
    c.allocs = g_allocs;
    c.frees = g_frees;
    c.bytes = g_bytes;
    for (int i = 0; i < g_sizeCount && i < 8; i++) {
        if (i) { c.sizes += " + "; }
        c.sizes += std::to_string(g_sizes[i]);
    }
    if (g_sizeCount > 8) { c.sizes += " + ..."; }
    return c;
}

int g_checks = 0;
int g_failed = 0;

void report(const char *what, const Count &c, const char *note = "")
{
    ++g_checks;
    std::printf("    %-34s 分配 %2ld 次 / 释放 %2ld 次 / %6zu B   前几次分配: [%s]%s\n",
                what, c.allocs, c.frees, c.bytes, c.sizes.c_str(),
                (note[0] ? ("   " + std::string(note)).c_str() : ""));
}

/* 期望值断言: 与"读数"分开印, 免得把期望当成实测 */
void expectAllocs(const char *what, const Count &c, long want)
{
    ++g_checks;
    if (c.allocs == want) {
        std::printf("      ✓ %s: %ld 次分配 (期望 %ld)\n", what, c.allocs, want);
    } else {
        ++g_failed;
        std::printf("      ✗ %s: %ld 次分配 (期望 %ld)\n", what, c.allocs, want);
    }
}

template <typename Fn>
Count measure(Fn fn)
{
    beginCount();
    fn();
    return endCount();
}

} /* namespace */

int main()
{
#ifdef _DEBUG
    _CrtSetAllocHook(allocHook);
    std::printf("[info] Debug CRT: _CrtSetAllocHook 已装上 (/_MDd)\n\n");
#else
    std::printf("[FATAL] 这是 Release 构建: `_CrtSetAllocHook` 不存在, 计数会全 0。\n"
                "        请在 Debug 构建目录里跑本探针 (见文件头注释)。\n");
    return 2;
#endif

    /* ============================================================
     *  [A] 钩子自检: 证明"三条分配路径"都被数到了
     * ============================================================ */
    std::printf("[A] 钩子自检 (三条路径各一次分配)\n");
    {
        Count c1 = measure([] { void *p = ::operator new(64); ::operator delete(p); });
        report("::operator new(64)", c1);
        expectAllocs("裸 operator new", c1, 1);

        Count c2 = measure([] { void *p = _aligned_malloc(128, 32); _aligned_free(p); });
        report("_aligned_malloc(128,32)  ← 数据缓冲走这条", c2);
        expectAllocs("_aligned_malloc", c2, 1);

        Count c3 = measure([] { std::vector<int> v(10, 0); (void)v; });
        report("std::vector<int>(10)      ← shape/sizes 走这条", c3);
        expectAllocs("std::vector<int>", c3, 1);    }
    std::printf("\n");

    /* ============================================================
     *  [B] 构造路径
     *  A11（形状 SSO）**之前**的基线是方案 §1.2 那句"恰好 3 次":
     *      shape 8 B + sizes 8 B + 对齐数据（秩 ≤ 4 时也是 3 次 —— 这正是 A11 要消灭的）
     *  A11 **之后**应当是 1 次: 只有数据缓冲（形状/步长内联在对象里）
     * ============================================================ */
    std::printf("[B] 构造: A11 前 = 3 次 (shape + sizes + 数据), A11 后应为 1 次 (只有数据)\n");
    {
        Count c = measure([] { RL::Tensor t(19, 1); (void)t; });
        report("Tensor(19,1)  [变参构造]", c);
        expectAllocs("Tensor(19,1)", c, 1);

        Count c2 = measure([] { RL::Tensor t(90, 64); (void)t; });
        report("Tensor(90,64)", c2);
        expectAllocs("Tensor(90,64)", c2, 1);

        std::vector<int> shape = {90, 64};
        Count c3 = measure([&] { RL::Tensor t(shape); (void)t; });
        report("Tensor(shape_vector)  [2 维]", c3);
        expectAllocs("Tensor(shape_vector) 2 维", c3, 1);

        std::vector<int> shape3 = {2, 90, 64};
        Count c4 = measure([&] { RL::Tensor t(shape3); (void)t; });
        report("Tensor(shape_vector)  [3 维]", c4);
        expectAllocs("Tensor(shape_vector) 3 维", c4, 1);

        RL::Tensor src(90, 64);
        Count c5 = measure([&] { RL::Tensor t(src); (void)t; });
        report("拷贝构造 Tensor(src)", c5);
        expectAllocs("拷贝构造", c5, 1);

        RL::Tensor mov(90, 64);
        Count c6 = measure([&] { RL::Tensor t(std::move(mov)); (void)t; });
        report("move 构造 (noexcept)", c6);
        expectAllocs("move 构造", c6, 0);

        Count c7 = measure([] { RL::Tensor t; (void)t; });
        report("默认构造 Tensor()", c7);
        expectAllocs("默认构造", c7, 0);

        RL::Tensor null(1, 1);
        Count c8 = measure([&] { RL::Tensor t(0, 5); (void)t; });
        report("Tensor(0,5)  [退化形状]", c8);
        expectAllocs("Tensor(0,5)", c8, 0);
        (void)null;
    }
    std::printf("\n");

    /* ============================================================
     *  [C] 赋值与形状操作
     * ============================================================ */
    std::printf("[C] 赋值 / reshape (A9 之后 totalSize 与 shape 必须永远同步)\n");
    {
        RL::Tensor a(90, 64);
        RL::Tensor b(90, 64);
        Count c = measure([&] { b = a; });
        report("赋值 b = a  (容量已够)", c);
        expectAllocs("赋值 (复用容量)", c, 0);

        Count c2 = measure([&] { a.reshape(64, 90); });
        report("reshape(64,90)  [同元素数]", c2);
        expectAllocs("reshape (形状内联)", c2, 0);

        Count c3 = measure([&] { a.reshape(5760, 1); });
        report("reshape(5760,1) [同元素数]", c3);
        expectAllocs("reshape (形状内联)", c3, 0);
    }
    std::printf("\n");

    /* ============================================================
     *  [D] 形状规模对分配次数的影响 (A11 形状 SSO 的直接判据)
     *      秩 ≤ RL_SHAPE_INLINE(4) ⇒ 形状/步长零分配; 秩 > 4 才回退到堆
     *      （A10 删掉 `sizes` 成员之后, 超容量的秩 5 从 3 次降到 2 次: 只剩 shape 那一次）
     * ============================================================ */
    std::printf("[D] 形状维数 vs 分配次数 (内联容量 = %zu)\n",
                (std::size_t)RL::SmallShapeVec<4>::inlineCapacity());
    {
        struct Case { const char *name; long want; };
        const Case cases[] = {
            { "Tensor(90,1)        [秩 2]", 1 },
            { "Tensor(90,64)       [秩 2]", 1 },
            { "Tensor(2,90,64)     [秩 3]", 1 },
            { "Tensor(2,3,4,5,6)   [秩 5 > 内联容量]", 2 },
        };
        for (int k = 0; k < 4; k++) {
            Count c;
            if (k == 0) { c = measure([] { RL::Tensor t(90, 1); (void)t; }); }
            if (k == 1) { c = measure([] { RL::Tensor t(90, 64); (void)t; }); }
            if (k == 2) { c = measure([] { RL::Tensor t(2, 90, 64); (void)t; }); }
            if (k == 3) { c = measure([] { RL::Tensor t(2, 3, 4, 5, 6); (void)t; }); }
            report(cases[k].name, c);
            expectAllocs(cases[k].name, c, cases[k].want);
        }
    }
    std::printf("\n");

    /* ============================================================
     *  [E] 对象体积: 三个阶段一起看 (A10 把 `sizes` 成员删掉之后**比原始还小**)
     * ============================================================ */
    std::printf("[E] 对象体积: 形状/长度存储改了三轮, 这里把几个阶段并排印出来\n");
    std::printf("    原始 (shape/sizes 都是 std::vector<int>) : 80 B = totalSize 8 + val 24 + sizes 24 + shape 24\n");
    std::printf("    A11 之后 (两者都换成内联 4 个 int)      : 112 B = 8 + 24 + (16+24) + (16+24)\n");
    std::printf("    A10 之后 (sizes 成员删掉, 步长按需算)   :  72 B = 8 + 24 + (16+24)\n");
    std::printf("    A9  之后 (totalSize 成员也删掉)         : %zu B = 24 + (16+24)  ← 长度由 `val` 自己带着\n",
                sizeof(RL::Tensor));
    std::printf("    ⇒ 相对**原始**净变化 = %+d B\n", (int)sizeof(RL::Tensor) - 80);
    std::printf("    换来的是「每次构造少 2 次 malloc」(%s)、步长/长度各自只剩一个来源。\n\n",
                "§1.2 实测单次小分配 230~262 ns");

    std::printf("=== 探针自查: %d 项, 失败 %d 项 ===\n", g_checks, g_failed);
    std::printf("(分配次数是**结构事实**, 跨机器可比; 计时不在这里量 —— 口径见方案 §5.1)\n");
    return g_failed == 0 ? 0 : 1;
}
