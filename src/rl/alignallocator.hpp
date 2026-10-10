#ifndef RL_ALIGNALLOCATOR_HPP
#define RL_ALIGNALLOCATOR_HPP

/*
 * alignallocator.hpp — 对齐分配器（2026-10 从 N-spirits 的 `basic/alignallocator.hpp` 迁移）
 * ============================================================================
 * 为什么需要它: `Tensor_` 的存储是 `std::vector<T, Alloc<T>>`, 而默认的 `std::allocator`
 * 只保证 `alignof(std::max_align_t)`（MSVC x64 上是 **16 B**）。这带来两件事:
 *
 *   1. **正确性（潜在）**: `simd/avx2func.hpp` 里有一个内核用了**要求对齐**的指令
 *      （`_mm256_store_ps` 要 32 B、`_mm_load_ps` 要 16 B）。只要有人对**真矩阵**调用它，
 *      16 B 对齐的缓冲就会触发 #GP（不是"慢一点"，是崩）。本文件把"碰运气"变成"有保证"。
 *   2. **性能（有限、需实测）**: 32 B 对齐让一条 32 B 的 AVX 访问**永不跨 64 B cache line**。
 *      ⚠ 本仓库的实测口径（见 docs/seq_transformer_design.md §8.4）: 只有"同一进程同一轮"
 *      的成对比值可信。到目前为止的读数表明**对齐不是主要收益来源**（`MM::ikjk` 的循环写法
 *      是 12~25×，整批 GEMM 只有 1.00~1.13×），所以不要把"换分配器"当成提速手段。
 *
 * 与上游的差异（三处，都是**刻意**的）:
 *   * 放进 `namespace RL`（上游在全局名字空间）—— 本仓库其余部分都在 `RL` 里;
 *   * **修掉 POSIX 分支的 UB**: C11 的 `aligned_alloc` 要求 `size` 是 `alignment` 的
 *     整数倍，否则行为未定义。上游直接传 `n*sizeof(T)`（例如 N=32、float、n=90 → 360 B ✗）。
 *     这里按 `⌈size/N⌉·N` 向上取整。MSVC 分支的 `_aligned_malloc` 无此要求，但两条分支
 *     行为一致更好;
 *   * 补上 C++11 之后 `std::vector` 会用到的 traits（`is_always_equal` /
 *     `propagate_on_container_*`）与一个**对齐自检**入口 `alignmentOf()`
 *     （本工程的惯例: 这种"看不见的契约"要能读出来）。
 *
 * 它是**无状态**分配器（`operator==` 恒真）⇒ 不同实例之间可以互相 deallocate，
 * 容器可以自由拷贝/移动。`_aligned_malloc` 必须配 `_aligned_free`（MSVC 下混用 CRT `free`
 * 会坏堆），这里配对正确。
 *
 * 用法（这就是它存在的意义）: `Tensor_` 的第二个模板参数就是分配器槽位 ——
 *     `RL::Tensor_<float, RL::AlignAllocator32>`
 * 本仓库在 `tensor.hpp` 里把它设成了**默认值**，所以 `Tensor` / `Tensorf` 全域都是 32 B 对齐。
 */
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

#ifndef RL_ALIGN_DEFAULT
#define RL_ALIGN_DEFAULT 32      /* 默认对齐 = AVX2 的向量宽度 (ymm) */
#endif

namespace RL {

/* 对齐向上取整（`aligned_alloc` 的 size 必须是 alignment 的整数倍） */
inline std::size_t alignUp(std::size_t n, std::size_t a)
{
    return ((n + a - 1) / a) * a;
}

/* 这个指针是否是 `a` 字节对齐的（给 SIMD 派发/自检用） */
inline bool isAlignedTo(const void *p, std::size_t a)
{
    return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}

template <typename T, std::size_t N>
class AlignedAllocator
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    /* C++11 之后 std::vector 会读这几个 traits: 无状态 ⇒ 恒等; 不随容器传播 (不需要) */
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;

    static constexpr std::size_t alignment() { return N; }

public:
    AlignedAllocator() noexcept {}
    template <typename T2>
    AlignedAllocator(const AlignedAllocator<T2, N> &) noexcept {}
    ~AlignedAllocator() {}

    pointer address(reference r) const { return &r; }
    const_pointer address(const_reference r) const { return &r; }

    pointer allocate(size_type n)
    {
        if (n == 0) { return nullptr; }
        const std::size_t bytes = n * sizeof(value_type);
#ifdef _MSC_VER
        void *p = _aligned_malloc(alignUp(bytes, N), N);
#else
        void *p = std::aligned_alloc(N, alignUp(bytes, N));   /* ← size 必须取整, 见文件头 */
#endif
        if (p == nullptr) {
            /*
               这里抛异常而不是 abort: 分配失败是**资源**问题, 由调用方决定怎么办
               (std::vector 会把它原样传播出去)。本工程"响亮失败"的是**结构/契约**错误
               (例如 dIn 不被 SeqLen 整除), 那类才 abort。
            */
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }
    void deallocate(pointer p, size_type) noexcept
    {
        if (p == nullptr) { return; }
#ifdef _MSC_VER
        _aligned_free(p);          /* 必须与 _aligned_malloc 配对 */
#else
        std::free(p);
#endif
    }

    /* 上游保留了 C++03 风格的 construct/destroy/adress —— 这里只留 construct/destroy */
    template <typename U, typename... Args>
    void construct(U *p, Args &&... args) { ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...); }
    template <typename U>
    void destroy(U *p) { p->~U(); }

    size_type max_size() const noexcept { return size_type(-1) / sizeof(value_type); }

    template <typename T2>
    struct rebind { using other = AlignedAllocator<T2, N>; };

    bool operator!=(const AlignedAllocator<T, N> &) const { return false; }
    bool operator==(const AlignedAllocator<T, N> &) const { return true; }
};

/* 上游就提供这三个单参数别名 —— 它们存在的意义正是当 `template<typename> class Alloc`
   实参用（`AlignedAllocator` 本身是**两个**模板参数, 不能直接填进那个槽位）。 */
template <typename T>
using AlignAllocator16 = AlignedAllocator<T, 16>;

template <typename T>
using AlignAllocator32 = AlignedAllocator<T, RL_ALIGN_DEFAULT>;

template <typename T>
using AlignAllocator64 = AlignedAllocator<T, 64>;

} /* namespace RL */

#endif // RL_ALIGNALLOCATOR_HPP
