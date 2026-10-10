#ifndef TENSOR_H
#define TENSOR_H
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <climits>     /* [2026-10] INT_MAX: posOf 的 32 位乘加契约 */
#include <cstdio>      /* [2026-10] 结构守卫的"响亮失败"输出 */
#include <assert.h>
#include "alignallocator.hpp"   /* [2026-10] 对齐分配器: Tensor 的存储默认 32 B 对齐 */
#include "simd_ops.hpp"

/*
   ============================================================
    [2026-10] 张量的存储从 `std::allocator`（16 B 保证）换成 `AlignAllocator32`（32 B 保证）
   ============================================================
   动机有两条, 一条是正确性、一条是性能（性能那条**还没被实测证实**, 见下）:

   1. **正确性**: `simd/avx2func.hpp` 的转置内核用了 `_mm256_store_ps`（**要求 32 B**）与
      `_mm_load_ps`（要求 16 B）。之前 Tensor 的缓冲只有 16 B 保证 ⇒ 一旦真拿矩阵去调它
      就是 #GP（"碰运气"能跑不等于对）。现在 `val.data()` 有 32 B 保证，这条契约成立。
   2. **性能**: 32 B 对齐的 32 B 访问**永不跨 64 B cache line**。⚠ 但本仓库所有**在用**的
      SIMD 内核走的都是 `loadu/storeu`（非对齐指令，跨线才有一点代价），所以"对齐更快"是
      **待实测**的假设，不是已知事实（对照: `MM::ikjk` 的循环写法改动值 12~25×，而"整批
      GEMM"只有 1.00~1.13×）。

   代价与收益（**实测**，`probe_align`，同进程/同轮交替/9 轮中位数/固定单核；同一块 32 B 对齐
   缓冲 vs 把它**错开 16 B**，5760 float = 23 KB 的缓冲，在用内核走 `loadu/storeu`）:

     | 内核 | 32 B 对齐 | 错开 16 B | 比值（三次） |
     |---|---:|---:|---|
     | `fill`（纯写） | 1.58~1.88 µs | 3.22~4.40 µs | **2.39 / 2.03 / 2.04** ⇒ 稳定，**流式写慢 ~2×** |
     | `dot`（读为主） | 6.70~7.71 µs | 6.50~14.09 µs | 1.83 / 0.97 / 0.91 ⇒ **没有可测差别** |
     | `mul`（读+写） | 3.96~8.63 µs | 4.53~14.03 µs | 1.63 / 1.14 / 1.03 ⇒ 弱、不稳 |

   结论要按**最保守**的那条读: 错开 16 B 会让**每一次 32 B 访问都跨 cache line**, 而
   Alder Lake 对**跨界写**的惩罚大（~2×）、对**跨界读**基本免疫 ⇒ 对齐的稳定收益在"写密集"
   的路径上（`fill`/清零/拷贝/逐元素写），不在只读归约上。⚠ 第一次那组 `dot/mul` 报
   1.6~1.8× 是**机器噪声**（那一轮对照通道跨度 315%，见 docs/seq_transformer_design.md §8.4）
   —— 同一个二进制复测两次就回到 0.91~1.14。教训与 §8.4 一致: 比值必须同轮成对量、并印出
   对照通道跨度，否则会把噪声当成结论。

   分配代价: 小张量构造 0.93~1.12×（≈没有差别；Tensor 的构造成本本来就被 shape/sizes 主导）。
*/

/*
   别名提示 (给编译器, 不影响语义): 只用在"逐元素的两条独立内存"这类循环上 ——
   裸指针默认要按可能别名处理, 编译器会因此放弃向量化。调用方本来就要求输出与输入
   不是同一块内存 (MM 的语义), 所以这是安全的。
*/
#if defined(_MSC_VER)
#  define RL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define RL_RESTRICT __restrict__
#else
#  define RL_RESTRICT
#endif

namespace RL {

template<typename T, template<typename Ti> class Alloc=AlignAllocator32>
class Tensor_
{
public:
    using ValueType = T;
    using Vector = std::vector<T, Alloc<T> >;
    using Shape = std::vector<int>;
    using Size = std::vector<int>;
    using iterator = typename Vector::iterator;
    using const_iterator = typename Vector::const_iterator;
    class SubTensor
    {
    public:
        Tensor_ *pointer;
        std::size_t pos;
        std::size_t totalSize;
    public:
        SubTensor():pointer(nullptr),pos(0),totalSize(0){}
        SubTensor(const SubTensor &r):pointer(r.pointer),pos(r.pos),totalSize(r.totalSize){}
        SubTensor& operator=(const SubTensor &r)
        {
            if (this == &r) {
                return *this;
            }
            pointer = r.pointer;
            pos = r.pos;
            totalSize = r.totalSize;
            return *this;
        }
        inline void operator=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] = x.val[i];
            }
            return;
        }
        inline void operator=(const std::vector<T> &x)
        {
            for (std::size_t i = 0; i < x.size(); i++) {
                pointer->val[i + pos] = x[i];
            }
            return;
        }

        inline void operator=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] = x;
            }
            return;
        }

        inline void operator+=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] += x.val[i];
            }
            return;
        }

        inline void operator-=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] -= x.val[i];
            }
            return;
        }

        inline void operator*=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] *= x.val[i];
            }
            return;
        }

        inline void operator/=(const Tensor_ &x)
        {
            for (std::size_t i = 0; i < x.totalSize; i++) {
                pointer->val[i + pos] /= x.val[i];
            }
            return;
        }

        inline void operator+=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] += x;
            }
            return;
        }

        inline void operator-=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] -= x;
            }
            return;
        }

        inline void operator*=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] *= x;
            }
            return;
        }

        inline void operator/=(T x)
        {
            for (std::size_t i = 0; i < totalSize; i++) {
                pointer->val[i + pos] /= x;
            }
            return;
        }

        inline T sum() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                s += pointer->val[i + pos];
            }
            return s;
        }

        inline T mean() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                s += pointer->val[i + pos];
            }
            return s/T(totalSize);
        }

        inline T variance(T u) const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                float d = pointer->val[i + pos] - u;
                s += d*d;
            }
            return s/T(totalSize);
        }

        inline T max() const
        {
            T maxVal = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (maxVal < val) {
                    maxVal = val;
                }
            }
            return maxVal;
        }

        inline std::size_t argmax() const
        {
            T maxVal = 0;
            std::size_t index = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (maxVal < val) {
                    maxVal = val;
                    index = i;
                }
            }
            return index;
        }

        inline T min() const
        {
            T minVal = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (minVal > val) {
                    minVal = val;
                }
            }
            return minVal;
        }

        inline std::size_t argmin() const
        {
            T minVal = 0;
            std::size_t index = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                if (minVal > val) {
                    minVal = val;
                    index = i;
                }
            }
            return index;
        }
        inline T norm2() const
        {
            T s = 0;
            for (std::size_t i = 0; i < totalSize; i++) {
                T val = pointer->val[i + pos];
                s += val*val;
            }
            return std::sqrt(s);
        }
    };

protected:
    SubTensor subTensor;
public:
    std::size_t totalSize;
    Vector val;
    Size sizes;
    Shape shape;
public:
    /* default construct */
    Tensor_():totalSize(0){}

    /*
       ============================================================
        [2026-10] 两条**常驻**(Release 也生效)的结构守卫
       ============================================================
       为什么这两条不能像 MM 的形状断言那样只活在 Debug (那条纪律见 `requireShape2d`
       与 `docs/issues_review.md` R1.5 —— "Release 一个字节都不变"):
         * `posOf()` 的乘加是**32 位**的 (`int` 步长 × `int` 下标), 它成立的前提是
           "每个下标与步长的乘积都落在 int 范围内"(`tensor.hpp:655-659` 只用注释声明,
           代码里**没有任何拦截**)。一次静默截断在这里等于**错位下标** —— 读到别人的数、
           写到别人的位置, 而且是"看起来合理"的数字。
         * `totalSize` 与 `val.size()` 是**两套长度源**: `reshape`/`view` 只改前者,
           于是"按 totalSize 的循环"与"按 val.size() 的循环"永久分叉
           (`flatten()` 的历史越界 bug 就是这一类, `rl_sync.md:314-316`)。
       两条都是**每次构造/每次 reshape 一次整数比较**, 不是每元素; 失败按仓库纪律
       **响亮失败** (结构/契约错误 abort, 资源错误才抛异常 —— 与 `AlignedAllocator`
       的 `bad_alloc` 同一条分界线)。
    */
    static void requireIndexable(std::size_t total, const char *where)
    {
        if (total > (std::size_t)INT_MAX) {
            std::fprintf(stderr,
                         "[Tensor] %s: 元素总数 %llu 超出 32 位下标契约\n"
                         "  ⇒ `posOf` 用 int 步长 × int 下标累加 (见 tensor.hpp:655-659), 超过 2^31 会静默错位。\n"
                         "     要么把张量切小, 要么按那里的说明把 Shape/Size 一起改成 64 位 (那是一次带实测代价的改动)。\n",
                         where, (unsigned long long)total);
            std::fflush(stderr);
            std::abort();
        }
    }
    static void requireSameLength(const Tensor_ &t, std::size_t expect, const char *where)
    {
        if (t.val.size() != expect) {
            std::fprintf(stderr,
                         "[Tensor] %s: 形状乘积 %llu 与缓冲长度 %llu 不一致\n"
                         "  ⇒ 这个类有两套长度 (`totalSize` / `val.size()`); 不拦住的话, 按 totalSize 的循环\n"
                         "     与按 val.size() 的循环会永久分叉 (越界读/写, 或只填了一半)。\n",
                         where, (unsigned long long)expect, (unsigned long long)t.val.size());
            std::fflush(stderr);
            std::abort();
        }
    }
    static std::vector<int> sizesOf(const std::vector<int> &shape)
    {
        std::vector<int> sizes(shape.size(), 1);
        for (std::size_t i = 0; i < shape.size() - 1; i++) {
            for (std::size_t j = i + 1; j < shape.size(); j++) {
                 sizes[i] *= shape[j];
            }
        }
        return sizes;
    }
    static void initParams(const Shape &shape, Size &sizes, std::size_t &totalsize)
    {
        totalsize = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            totalsize *= shape[i];
        }
        sizes = std::vector<int>(shape.size(), 1);
        for (std::size_t i = 0; i < shape.size() - 1; i++) {
            for (std::size_t j = i + 1; j < shape.size(); j++) {
                 sizes[i] *= shape[j];
            }
        }
        /* [2026-10] 常驻守卫: `posOf` 的 32 位乘加契约 (见 requireIndexable 的说明)。
           放在这里 ⇒ 7 个调用点 (四个构造 + reshape + view + 静态 posOf) 全覆盖。 */
        requireIndexable(totalsize, "initParams (形状乘积)");
        return;
    }
    /* contruct with shape */
    explicit Tensor_(const Shape &shape_)
        :totalSize(1),shape(shape_)
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T, Alloc<T>>(totalSize, T(0));
    }

    explicit Tensor_(const Shape &shape_,
                     const std::vector<T, Alloc<T>> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }


    explicit Tensor_(const std::initializer_list<int> &shape_,
                     const std::initializer_list<T> &val_):
        totalSize(1),shape(shape_),val(val_)
    {
        initParams(shape, sizes, totalSize);
    }

    /*
       ⚠ [2026-10 审计] 这个构造**是坏的、而且从来没被实例化过**: `x[0].totalsize`
       拼错了成员名 (成员是 `totalSize`), `sizes.push_back(x[0].sizes)` 也把
       `std::vector<int>` 往 `std::vector<int>` 里推了一个 vector —— 两处都过不了编译。
       模板成员只在实例化时检查, 所以它一直静静躺在那里。
       正确的"把 N 个子张量摞起来"的实现在 `fromVector()` (它有一整段注释写清楚了
       旧版本 `std::copy` 越界写 `shape` 的那个 bug)。这里**不删也不改**: 它属于
       上游 snakeAI 的同源代码, 删除/修复该走 `docs/rl_sync.md` 那条线, 不该混在
       "Tensor 下标类型"这一轮里。
    */
    explicit Tensor_(const std::vector<Tensor_> &x)
    {
        totalSize = x.size()*x[0].totalsize;
        sizes.push_back(x.size()*x[0].sizes[0]);
        sizes.push_back(x[0].sizes);
        shape.push_back(x.size());
        shape.push_back(x[0].shape);
        for (std::size_t i = 0; i < x.size(); i++) {
            val.push_back(x[i].val);
        }
    }

    explicit Tensor_(T x):
        totalSize(1),shape({1, 1}),val({x})
    {
        initParams(shape, sizes, totalSize);
    }
    /* construct with shape */
    template<typename ...Dim>
    explicit Tensor_(Dim ...dim):totalSize(1),shape({int(dim)...})
    {
        initParams(shape, sizes, totalSize);
        val = std::vector<T, Alloc<T> >(totalSize, T(0));
    }

    /* copy constructor */
    Tensor_(const Tensor_ &r)
        :totalSize(r.totalSize),shape(r.shape),sizes(r.sizes),val(r.val){}

    /*
       move construct —— **[2026-10] 必须标 `noexcept`**。
       为什么: `std::vector<Tensor>` 增长时用 `move_if_noexcept`, 而这个类**可拷贝**
       ⇒ 不标 noexcept 就会**退化成深拷贝**。实测 (`probe_move`,
       证据 `.r1build/probe_move_run1.txt`): 256 个 (1710,1) 张量, 无 reserve 的
       `push_back` **11.04 ms** vs 同样载荷的 `std::vector<float>`(它的 move 是 noexcept)
       **3.59 ms** = **3.08x**; 单看"增长"那一段 7.27 ms vs 0.45 ms = **16x**。
       三个 `swap` 都不会抛 (`std::vector::swap` 在 `is_always_equal` 分配器下是 noexcept),
       所以整条 move 可以承诺 noexcept —— 这正是 vector 需要的那个承诺。
       (顺带删掉原来那句冗余的 `totalSize = r.totalSize;`: 初始化列表里已经有了。)
    */
    Tensor_(Tensor_ &&r) noexcept:totalSize(r.totalSize)
    {
        shape.swap(r.shape);
        sizes.swap(r.sizes);
        val.swap(r.val);
        r.totalSize = 0;
    }

    /*
       [2026-10] 两个转换算子改成 `explicit`。
       原来 `Tensor → T*` (裸指针退化) 与 `Tensor → std::vector<T>`(**整块深拷贝**)
       都是**隐式**的: 任何 `float*` 形参、任何 `const std::vector<float>&` 形参都能
       静默接收, 后者一次拷走整张量。全仓 grep 找不到显式调用点, 但"隐式"意味着它
       可能正在别处发生 —— 改成 explicit 之后由**编译器**回答这个问题 (构建失败 = 有人在用)。
       `explicit operator T*` 仍然允许 `static_cast<T*>(t)`, 只是不再允许隐式退化。
    */
    explicit inline operator T* () noexcept
    {
        return val.data();
    }

    explicit inline operator Vector ()
    {
        return val;
    }

    inline Tensor_ operator - () const
    {
        Tensor_ x(shape);
        for (std::size_t i = 0; i < val.size(); i++) {
            x.val[i] = -val[i];
        }
        return x;
    }
    bool shapeEqual(const Tensor_ &x) const
    {
        bool flag = true;
        for (std::size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != x.shape[i]) {
                flag = false;
                break;
            }
        }
        return flag;
    }
    /*
       [2026-10] `data()` 是"从 Tensor 拿裸缓冲"的**正名**入口。
       为什么要加: 仓库里 `.val.data()` 有 68 处 (其中 55 处在 tensor.hpp 之外),
       而 `ptr()` 在 `src/` 里 **0 处** (只有测试 3 处) —— 大家想要的名字是 `data()`。
       `ptr()` 保留为别名 (不删, 有测试在用), 新代码一律用 `data()`。
    */
    inline T* data() noexcept { return val.data(); }
    inline const T* data() const noexcept { return val.data(); }
    inline T* ptr() noexcept { return val.data(); }
    inline const T* ptr() const noexcept { return val.data(); }
    /* [2026-10] 对齐自检: 这块存储是否 `a` 字节对齐 (默认 32 = AVX2 向量宽度)。
       为什么要有这个读数: 对齐是"看不见的契约" —— 要求对齐的内核拿到不对齐的指针是 UB
       (不是"慢一点"), 所以它必须能被印出来/被断言 (见 test_transformer [9])。 */
    inline bool alignedTo(std::size_t a = RL_ALIGN_DEFAULT) const
    {
        return RL::isAlignedTo(val.data(), a);
    }
    inline std::size_t dataAlignment() const
    {
        /* 从 1 往上找第一个"置位"的地址位: 那一位就是这块地址的自然对齐 (上限 256) */
        const std::size_t p = reinterpret_cast<std::size_t>(val.data());
        std::size_t a = 1;
        while (a < 256 && (p & a) == 0) { a <<= 1; }
        return a;
    }
    inline bool empty() const {return totalSize == 0;}
    /* iterator */
    inline iterator begin() noexcept { return val.begin();}
    inline const_iterator begin() const noexcept { return val.begin();}
    inline iterator end() noexcept { return val.end();}
    inline const_iterator end() const noexcept { return val.end();}
    /* size */
    inline std::size_t size() const {return totalSize;}

    template<typename ...Index>
    inline std::size_t size(Index ...index) const
    {
        std::size_t N = sizeof ...(Index) - 1;
        return sizes[N];
    }

    inline std::size_t size(const Shape &indexes) const
    {
        std::size_t N = indexes.size() - 1;
        return sizes[N];
    }

    /*
       zero/fill 走 SIMD 内核 (内核写满所有元素, 所以 val 的长度必须已经是对的长度;
       assign 保证这一点, 且这里要的不是"赋同一个值"而是"原地填", 因此用 data()).
    */
    void zero(){simdops::fill(val.data(), T(0), val.size());}
    void fill(T value){simdops::fill(val.data(), value, val.size());}
    /*
       ============================================================
       [2026-10] 下标类型: `int` -> `std::size_t`
       ============================================================
       原来这两个是 `operator[](int)`, 而本仓库里**几乎每一处逐元素循环**都写成

           for (std::size_t i = 0; i < t.totalSize; i++) { t[i] = ...; }

       (totalSize / val.size() 都是 `std::size_t`) —— 于是每次访问都要
       `size_t -> int -> size_t` 绕一圈: 先截断 (mov), 进 `val[i]` 时再零扩展。
       编译器不能总是证明"截断无害"(i 可能 > INT_MAX), 所以这个来回通常留在代码里。

       改成 `std::size_t` 之后:
         * 传 `int` 的调用方**仍然合法** —— int -> size_t 是**加宽**转换, 不会丢值;
         * "用 size_t 循环"这条主路径上不再有任何转换 (与 `val[]` 的类型一致)。
       ⚠ 不保留 `operator[](int)` 重载: 两个重载会让 `t[0u]` / `t[someEnum]` 这类调用
         变成二义 (unsigned int 到两者都要一次转换), 而 `size_t` 一个版本就覆盖了它们。
    */
    inline T &operator[](std::size_t i) {return val[i];}
    inline T operator[](std::size_t i) const {return val[i];}

    /* assign operator */
    inline Tensor_& operator=(const Tensor_ &r)
    {
        if (this == &r) {
            return *this;
        }
        totalSize = r.totalSize;
        shape = r.shape;
        sizes = r.sizes;
        val = r.val;
        return *this;
    }

    inline Tensor_& operator=(const std::vector<T> &x)
    {
        val.assign(x.begin(), x.end());
        return *this;
    }
    inline Tensor_& operator=(T x)
    {
        val.assign(totalSize, x);
        return *this;
    }

    /* move —— 同样必须 `noexcept` (理由见 move 构造上方的注释) */
    Tensor_ &operator=(Tensor_ &&r) noexcept
    {
        if (this == &r) {
            return *this;
        }
        totalSize = r.totalSize;
        shape.swap(r.shape);
        sizes.swap(r.sizes);
        val.swap(r.val);
        r.totalSize = 0;
        return *this;
    }
    /* [2026-10] 显式 swap: 三个容器都是 noexcept swap ⇒ 整条 noexcept */
    void swap(Tensor_ &r) noexcept
    {
        const std::size_t t = totalSize;
        totalSize = r.totalSize;
        r.totalSize = t;
        shape.swap(r.shape);
        sizes.swap(r.sizes);
        val.swap(r.val);
    }
    /* init */
    static Tensor_ zeros(Shape &shape)
    {
        Tensor_ x(shape);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ zeros(Dim ...dim)
    {
        Tensor_ x(dim...);
        return x;
    }

    static Tensor_ ones(Shape &shape)
    {
        Tensor_ x(shape);
        x.fill(1);
        return x;
    }

    template<typename ...Dim>
    static Tensor_ ones(Dim ...dim)
    {
        Tensor_ x(dim...);
        x.fill(1);
        return x;
    }
    /* subset */
    template<typename ...Index>
    Tensor_ sub(Index ...index) const
    {
        std::size_t N = sizeof ...(Index);
        std::vector<int> subIndex(shape.begin() + N, shape.end());
        Tensor_ y(subIndex);
        std::size_t pos = posOf(index...);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = val[i + pos];
        }
        return y;
    }

    template<typename ...Index>
    inline SubTensor& at(Index ...index)
    {
        subTensor.pointer = this;
        subTensor.pos = posOf(index...);
        subTensor.totalSize = size(index...);
        return subTensor;
    }

    Tensor_ block(const std::vector<int> &offset, const std::vector<int> &blockShape) const
    {
        Tensor_ y(blockShape);
        std::vector<int> indexs(shape.size(), 0);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            /* local offset */
            y.indexOf(i, indexs);
            for (std::size_t j = 0; j < indexs.size(); j++) {
                indexs[j] += offset[j];
            }
            std::size_t o = posOf(indexs);
            y.val[i] = val[o];
        }
        return y;
    }

    void embedding(const std::vector<int> &offset, const Tensor_ &x)
    {
        std::vector<int> indexs(shape.size(), 0);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.indexOf(i, indexs);
            for (std::size_t j = 0; j < indexs.size(); j++) {
                indexs[j] += offset[j];
            }
            std::size_t o = posOf(indexs);
            val[o] = x.val[i];
        }
        return;
    }

    template<typename ...Index>
    void slice(Tensor_ &y, Index ...index) const
    {
        std::size_t pos = posOf(index...);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = val[i + pos];
        }
        return;
    }

    void toVector(std::vector<Tensor_> &vec) const
    {
        for (std::size_t i = 0; i < shape[0]; i++) {
            vec.push_back(sub(i));
        }
        return;
    }

    static Tensor_ fromVector(const std::vector<Tensor_> &vec)
    {
        if (vec.empty()) {
            return Tensor_();
        }
        /* Stack the sub-tensors: shape (N, d0, d1, ...) from N tensors of shape
           (d0, d1, ...). The old code std::copy'd the sub-tensor's VALUES into
           `shape` starting at begin()+1, which wrote past the end of a
           1-element vector (out-of-bounds) and produced a bogus shape. */
        std::vector<int> shape;
        shape.push_back(static_cast<int>(vec.size()));
        for (std::size_t i = 0; i < vec[0].shape.size(); i++) {
            shape.push_back(vec[0].shape[i]);
        }
        Tensor_ x(shape);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < vec.size(); i++) {
            for (std::size_t j = 0; j < vec[i].totalSize; j++) {
                x.val[j + offset] = vec[i][j];
            }
            offset += vec[i].totalSize;
        }
        return x;
    }

    /* visit */
    template<typename ...Index>
    inline std::size_t posOf(Index ...index) const
    {
        /*
           ============================================================
           [2026-10] 这里做过一次"把下标全提到 size_t"的改动, **被实测否决了**
           ============================================================
           背景: 原来是 `int indexs[] = {index...};`, 传 `std::size_t` 下标时是**收缩转换**
           —— 就是每次构建都报的那条 `C4838: 从 "size_t" 转换到 "int" 需要收缩转换`。

           第一版把它们全部提到 64 位 (数组 `std::size_t[]` + `(std::size_t)sizes[i]*
           indexs[i]`)。实测 (`bench_tensor_index`, 256×256 与 1710×1710 两种规模、
           交替 A/B、取中位数) 在"`operator()` 密集"的二维循环上**慢了 34%**
           (5.25 -> 3.48 GB/s), 而 `operator[]` 那两条路径在噪声内 (±1~4%, 对照的 SIMD
           fill 也在 ±8% 漂)。原因: 这个点积在 N=2 时是"两次 32 位乘加", 提到 64 位之后
           每次都要先把 `int` 步长符号扩展到 64 位再做 64 位乘, 而且数组元素从 4 字节变 8 字节
           —— 换来的"空间"在这份代码里根本用不上 (所有下标都远小于 2^31)。

           所以保留 32 位数组与 32 位乘法, 只把**隐式**窄化写成**显式** `static_cast<int>`:
             * 语义与改动前**逐位相同** (同一个截断, 只是不再依赖隐式转换规则);
             * `C4838` 消失 (实测: 改完之后 RL_CORE 全量构建不再有这条警告);
             * 与 `sizes`(=`std::vector<int>`) 的类型一致, 没有跨类型乘法。

           隐含前提 (与改动前一致, 不是新引入的): **每个下标与步长的乘积都落在 int 范围内**。
           本工程的最大张量是 1710×1710 ≈ 2.9 M 元素、最长的 1-D 张量 2.9 M、步长最大 1710,
           乘积上界 ~5e9 —— 单看 `sizes[0]*index0` 这一项是 `1710 * 1709 ≈ 2.9e6`,
           安全; 也就是说这个前提在本工程的形状下成立。真要支持 >2^31 的张量,
           得把 `Shape`/`Size` 一起改成 `std::size_t`(公共类型, 牵动一大片), 那不是本轮的范围。
        */
        const int indexs[] = {static_cast<int>(index)...};
        std::size_t pos = 0;
        const std::size_t N = sizeof... (Index);
        for (std::size_t i = 0; i < N; i++) {
            pos += static_cast<std::size_t>(sizes[i] * indexs[i]);
        }
        return pos;
    }

    inline std::size_t posOf(const std::vector<int> &indexs) const
    {
        std::size_t pos = 0;
        /* 与上面同一个理由: 32 位乘加 + 显式扩宽 (原实现是 int*int 隐式转 size_t) */
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += static_cast<std::size_t>(sizes[i] * indexs[i]);
        }
        return pos;
    }

    inline static std::size_t posOf(const std::vector<int> &indexs, const std::vector<int> &shape)
    {
        std::vector<int> sizes = sizesOf(shape);
        std::size_t pos = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            pos += static_cast<std::size_t>(sizes[i] * indexs[i]);
        }
        return pos;
    }
    /*
       [2026-10] `pos` 由 `int` 改成 `std::size_t`: 调用点传的是循环下标
       (`y.indexOf(i, indexs)` 里的 i 是 `std::size_t`), 原签名每次都在做 size_t -> int。
       这里是**唯一**一处 64 位确实更合适的地方: `pos` 是"扁平下标", 它的上界是 totalSize
       (size_t), 而 `int pos_` 在累加过程中没有理由被限制在 32 位。
    */
    inline void indexOf(std::size_t pos, std::vector<int> &indexs) const
    {
        /*
            shape: (2, 3, 4, 5)
            sizes: (60, 20, 5, 1)
            totalsize 2*3*4*5 = 120
            indexs:(1, 2, 3, 4)
            pos : 60*1 + 20*2 + 5*3 + 4*1 = 119

            i0 = pos/60
            i1 = (pos - i0*60)/20
            i2 = (pos - i0*60 - i1*20)/5
            i3 = pos - i0*60 - i1*20 - i2*5
        */
        std::size_t pos_ = 0;
        for (std::size_t i = 0; i < sizes.size(); i++) {
            const int stride = sizes[i];
            indexs[i] = static_cast<int>((pos - pos_) / static_cast<std::size_t>(stride));
            pos_ += static_cast<std::size_t>(indexs[i] * stride);
        }
        return;
    }

    inline std::vector<int> indexOf(std::size_t pos) const
    {
        std::vector<int> indexes(shape.size(), 0);
        indexOf(pos, indexes);
        return indexes;
    }

    template<typename ...Index>
    inline T &operator()(Index ...index) { return val[posOf(index...)]; }

    template<typename ...Index>
    inline T operator()(Index ...index) const { return val[posOf(index...)]; }

    inline T &operator()(const Shape &indexs) { return val[posOf(indexs)]; }

    inline T operator()(const Shape &indexs) const { return val[posOf(indexs)]; }

    /*
       [2026-10] `reshape` 现在**校验元素总数守恒**。
       原来它只做 `shape = {…}; initParams(...)`, 于是 `t.reshape(1000,1)` 作用在
       4 个元素的张量上会得到 `totalSize==1000` 而缓冲只有 4 个 —— 之后
       `zero()/fill()` 按 `val.size()` 填、`operator[]`/`MM::` 按 `totalSize` 读,
       两套长度源永久分叉。这不是理论风险: `flatten()` 的同类 bug 在历史上真的越界过
       (`rl_sync.md:314-316`)。改名不改元素个数的用法 (attention.hpp:191/194) 不受影响。
    */
    template<typename ...Index>
    Tensor_& reshape(Index ...index)
    {
        shape = {index...};
        initParams(shape, sizes, totalSize);
        requireSameLength(*this, totalSize, "reshape");
        return *this;
    }

    template<typename ...Index>
    Tensor_ view(Index ...index) const
    {
        Tensor_ x = *this;
        x.shape = {index...};
        x.initParams(x.shape, x.sizes, x.totalSize);
        return x;
    }

    Tensor_ flatten() const
    {
        /*
         * Return a genuine 2-D column (totalSize x 1), NOT a 1-D {totalSize}
         * shape.
         *
         * The convolution -> fully-connected path in Net::forward/backward does
         * `layers[i]->forward(out.flatten())` / `layer->backward(out.flatten(), e)`,
         * and those consumers index the tensor as 2-D: e.g. ikjk() computes
         * `x2(j, k)` -> posOf(j, k) -> `sizes[0]*j + sizes[1]*k`. With a 1-D
         * shape, `sizes` has a single element, so `sizes[1]` was an
         * out-of-bounds vector read; the resulting garbage was multiplied by k
         * (up to 31) and produced a wild element access. Forward happened to
         * survive because there j is always 0, but the backward pass crashed —
         * which is what ConvPG/ConvDQN hit on the very first training step.
         */
        Tensor_ x(static_cast<int>(totalSize), 1);
        x.val = val;
        return x;
    }

    static void permuteIndexs(const std::vector<int> &indexs,
                              const std::vector<int> &permuteMap,
                              std::vector<int> &newIndexs)
    {
        for (std::size_t i = 0; i < permuteMap.size(); i++) {
            int k = permuteMap[i];
            newIndexs[i] = indexs[k];
        }
        return;
    }

    template<typename ...Pos>
    inline Tensor_ permute(Pos ...p) const
    {
        /*
            shape: [3, 2, 1]
            permute: (2, 1, 0)
            new shape: [1, 2, 3]
        */
        std::vector<int> permuteMap = {p...};
        /* permute shape */
        std::vector<int> newShape(shape.size(), 0);
        permuteIndexs(shape, permuteMap, newShape);
        Tensor_ x(newShape);
        /* permute value */
        std::vector<int> indexs(shape.size(), 0);
        std::vector<int> newIndexs(shape.size(), 0);
        for (std::size_t i = 0; i < val.size(); i++) {
            indexOf(i, indexs);
            permuteIndexs(indexs, permuteMap, newIndexs);
            x(newIndexs) = val[i];
        }
        return x;
    }

    Tensor_ tr() const
    {
        int rows = shape[0];
        int cols = shape[1];
        Tensor_ y(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y(j, i) = val[i*cols + j];
            }
        }
        return y;
    }

    /* operator
       ------------------------------------------------
       逐元素运算分派到 SIMD 内核 (见 rl/simd_ops.hpp 的契约说明)。
       内核要求两个操作数形状/长度相同且长度 >= 一个向量宽度, 否则回落到标量循环;
       原来的实现遍历的就是 val.size() 且直接索引 x.val[i] (没有广播语义), 所以
       只要两边长度相等就能安全替换。
    */
    Tensor_ operator +(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::add(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator -(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::sub(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator *(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::mul(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator /(const Tensor_ &x) const
    {
        Tensor_ y(shape);
        simdops::div(y.val.data(), val.data(), x.val.data(), val.size());
        return y;
    }

    Tensor_ operator %(const Tensor_ &x) const
    {
        Tensor_ y(shape[0], x.shape[1]);
        for (std::size_t i = 0; i < y.shape[0]; i++) {
            for (std::size_t k = 0; k < shape[1]; k++) {
                T vik = val[posOf(i, k)];
                for (std::size_t j = 0; j < y.shape[1]; j++) {
                    /* y(i, j) = val(i, k) * x(k, j) */
                    y(i, j) += vik*x(k, j);
                }
            }
        }
        return y;
    }

    Tensor_ &operator +=(const Tensor_ &x)
    {
        simdops::add(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ &operator -=(const Tensor_ &x)
    {
        simdops::sub(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ &operator *=(const Tensor_ &x)
    {
        simdops::mul(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    /* 原来返回的是 Tensor_ 值 (每次 /= 都多复制一份张量), 改为引用 */
    Tensor_ &operator /=(const Tensor_ &x)
    {
        simdops::div(val.data(), val.data(), x.val.data(), val.size());
        return *this;
    }

    Tensor_ operator +(T x) const
    {
        Tensor_ y(shape);
        simdops::add(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator -(T x) const
    {
        Tensor_ y(shape);
        simdops::sub(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator *(T x) const
    {
        Tensor_ y(shape);
        simdops::mul(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ operator /(T x) const
    {
        Tensor_ y(shape);
        simdops::div(y.val.data(), val.data(), x, val.size());
        return y;
    }

    Tensor_ &operator +=(T x)
    {
        simdops::add(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator -=(T x)
    {
        simdops::sub(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator *=(T x)
    {
        simdops::mul(val.data(), val.data(), x, val.size());
        return *this;
    }

    Tensor_ &operator /=(T x)
    {
        simdops::div(val.data(), val.data(), x, val.size());
        return *this;
    }

    /* statistics */
    /*
       [2026-10] 退化输入守卫。
       改动前: 空张量上 `argmax/argmin/normalize/max/min` 直接读 `val[0]`(**越界读**),
       `mean()` 是 `0/0`(NaN), `normalize()` 在常量张量上是 `0/0` —— 全是**静默**产出
       NaN 或越界值。按仓库纪律, 这是**契约**错误(调用方给了没有意义的输入),
       所以**响亮失败**, 而不是返回一个"看起来能用"的数。
       ⚠ 只拦"元素数为 0"和"跨度为 0"; 正常的 1 元素张量不受影响。
    */
    void requireNonEmpty(const char *where) const
    {
        if (val.empty()) {
            std::fprintf(stderr,
                         "[Tensor] %s: 张量是空的 (totalSize=%llu) —— 这个统计量没有定义\n"
                         "  ⇒ 空张量上取 max/argmax/mean/方差 会越界读或得到 NaN, 所以这里直接停。\n",
                         where, (unsigned long long)totalSize);
            std::fflush(stderr);
            std::abort();
        }
    }
    T sum() const
    {
        return simdops::sum(val.data(), totalSize);
    }

    T mean() const
    {
        if (totalSize == 0) { requireNonEmpty("mean"); }
        T s = sum();
        return s/T(totalSize);
    }

    T variance(T u) const
    {
        requireNonEmpty("variance");
        return simdops::variance(val.data(), u, val.size());
    }

    T max() const
    {
        requireNonEmpty("max");
        return simdops::maxValue(val.data(), val.size());
    }

    T min() const
    {
        requireNonEmpty("min");
        return simdops::minValue(val.data(), val.size());
    }

    std::size_t argmax() const
    {
        requireNonEmpty("argmax");
        T value = val[0];
        std::size_t index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value < val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    std::size_t argmin() const
    {
        requireNonEmpty("argmin");
        T value = val[0];
        std::size_t index = 0;
        for (std::size_t i = 0; i < val.size(); i++) {
            if (value > val[i]) {
                value = val[i];
                index = i;
            }
        }
        return index;
    }

    /* initialize */
    void normalize()
    {
        requireNonEmpty("normalize (min-max)");
        double minValue = val[0];
        double maxValue = val[0];
        for (std::size_t i = 0; i < val.size(); i++) {
            if (minValue > val[i]) {
                minValue = val[i];
            }
            if (maxValue < val[i]) {
                maxValue = val[i];
            }
        }
        if (!(maxValue > minValue)) {
            std::fprintf(stderr,
                         "[Tensor] normalize (min-max): 张量的跨度是 0 (min=max=%g) —— 会得到 0/0 = NaN\n"
                         "  ⇒ 常量张量没有 min-max 归一化可言, 这里直接停而不是静默产出 NaN。\n",
                         maxValue);
            std::fflush(stderr);
            std::abort();
        }
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] = (val[i] - minValue)/(maxValue - minValue);
        }
        return;
    }

    T norm2() const
    {
        return std::sqrt(simdops::dot(val.data(), val.data(), totalSize));
    }
    struct MM {
        /*
         * PERFORMANCE (measured on this machine: MSVC 2022 /O2, x64, Release,
         * Qt 6.9.2 - same benchmark as the numbers quoted per kernel below).
         *
         * Every kernel here used to index through operator(), and operator()
         * calls posOf(), which rebuilds an `int indexs[]` array from the
         * parameter pack on every single access, loops over it and re-reads
         * sizes[i] out of a std::vector<int>.  With two or three accesses per
         * multiply-accumulate that is ~18 ns/MAC (0.11 GFLOP/s), and it also
         * prevents the compiler from vectorising anything.
         *
         * The kernels below therefore take the three element buffers as flat
         * pointers and hoist every stride out of the loops:
         *
         *     x(i,  j) == xd [i*xr  + j*xc ]
         *     x1(i, k) == x1d[i*x1r + k*x1c]
         *     x2(k, j) == x2d[k*x2r + j*x2c]
         *
         * which is literally what posOf() computes (sizes[0]*idx0 +
         * sizes[1]*idx1); the strides are READ from sizes[], never assumed, so
         * the mapping is unchanged for every shape (including (n,1) column
         * vectors, whose sizes[] is {1,1}, not {n,1}).
         *
         * x is still ACCUMULATED into, never assigned, and no bounds check was
         * added: out-of-range behaviour stays "undefined" exactly as before.
         *
         * ROUNDING: the flat loops below keep the original k-ascending,
         * j-ascending accumulation order, so those are bit-for-bit identical to
         * the old code.  Two fast paths do reassociate the additions:
         *   - the 4-way unrolled row update in ikkj/kikj (4 k-values per pass
         *     over the x row), and
         *   - the k-inner 4-accumulator dot product in ikjk/kijk (the original
         *     i,k,j nest walked x2 at stride sizes[0], i.e. a gather).
         * Both add exactly the same products to the same elements, so the
         * mathematical result is identical and only the rounding order differs.
         * Cross-checked elementwise (in-place + returning variants) against the
         * naive posOf() triple loop over 26 shapes - square, rectangular,
         * (n,1) column vectors, odd sizes and >2-D shapes that exercise the
         * generic stride loops: max |difference| = 2.9e-05 on results of
         * magnitude 25.1, i.e. 1.2e-06 relative.
         *
         * That residual is dominated by the OLD kernel's own rounding error,
         * not by this change: for ikkj (90x360)*(360x90) against a
         * double-precision reference |old - exact| = 1.8e-05 but
         * |new - exact| = 8.4e-06 - the reassociated version here is the more
         * accurate of the two, and no ordering (not even the exact one) could
         * bring |old - new| below |old - exact|.
         *
         * MEASURED before -> after (same machine, same benchmark, MSVC 2022
         * /O2 /fp:precise, 300/200 repetitions per case, median of 3 runs, and
         * every "after" number is >50x the "before" number):
         *   ikkj (90x90)*(90x90)     18.07 -> 0.10 ns/MAC   0.11 -> 20.8 GFLOP/s
         *   ikkj (90x360)*(360x90)   18.01 -> 0.09 ns/MAC   0.11 -> 21.1 GFLOP/s
         *   kikj (360x90)^T*(360x90) 18.34 -> 0.09 ns/MAC   0.11 -> 21.4 GFLOP/s
         *   ikjk (128x64)*(64x90)^T  18.11 -> 0.33 ns/MAC   0.11 ->  6.0 GFLOP/s
         */
        /*
            2 维、连续行主序 (sizes == {cols, 1}) 的张量才能用 SIMD 内核: 内核里的
            下标是硬编码的 i*col + k 形式, 必须与 posOf() 的通用 stride 计算等价。
            1 维或 >2 维、以及被 block() 出来的非连续视图一律回落到下面的标量实现。
        */
        static bool contiguous2d(const Tensor_ &t)
        {
            return t.shape.size() == 2 && t.sizes.size() == 2 &&
                   t.sizes[0] == t.shape[1] && t.sizes[1] == 1;
        }

        /*
           ============================================================
            形状契约检查 (R1.5, 2026-09) —— Debug 断言 + [2026-10] **两条常驻**
           ============================================================
           下面四个内核的一切都从 z/x1/x2 的 shape 推导: z 的形状决定输出, x1 行主序
           [r,k]、x2 行主序 [k,c]。形状不匹配时既不是编译错误也不会运行时报错 ——
           要么按错位的 sizes **越界读写**, 要么算出看起来合理的数字。实测代价:
           写微基准时用错维度直接崩 (issues_review 的附注)。
           所以把每条内核的形状关系写成断言; **Release (NDEBUG) 下整块被编译掉,
           一个字节都不变**, 想验证就在编译那一个 TU 时加 /UNDEBUG 跑一遍。

           **[2026-10] 在 Debug 断言之外, 再加两条常驻 (Release 也生效) 的检查。**
           理由是一次**真实事故**: `src/rl/sac.h:99-104` —— critic 的第一层是按
           stateDim 建的, 却喂了 stateDim+actionDim 的拼接输入; Release 下断言被关掉
           ⇒ `MM::ikkj` **静默只读了前 stateDim 个元素**, 拼进去的策略概率被整块丢掉
           (Debug 构建里则会当场断言失败)。这正是"契约检查全在 NDEBUG"的代价。
           常驻的两条:
             * 每个入参的**缓冲装得下它自己的形状** (越界写的直接拦截点);
             * 内核的**形状关系** (k 维必须对得上) —— sac.h 那次漏掉的就是这一条。
           代价: **每次内核调用 ≈18 ns** (实测 `probe_checks`, 同轮配对: 小内核 (64,64)·(64,1)
           占它 0.88%、(360,1710)·(1710,1) 占 0.0036%、(90,360)·(360,90) 占 0.0017%;
           按 seq 骨干一次 forward 的 1080 次 FC 调用折算 ≈0.12% 的 forward —— 这是"把一次
           已经发生过的静默数据丢失换成 0.1% 的确定性"的价钱)。
           `RL_TENSOR_CHECKS` 打开时, 再把下面这批 Debug 断言也带进 Release (供"全程查"的构建)。
           失败一律**响亮 abort** —— 形状错不是"慢一点", 是越界读写。
        */
        static void contractFail(const char *what)
        {
            std::fprintf(stderr,
                         "[Tensor::MM] 形状契约失守: %s\n"
                         "  ⇒ 这不是'慢一点', 是越界读/写 (Release 下曾静默丢数据: src/rl/sac.h:99-104)。\n"
                         "     请检查这一层的输入/输出维度是否与权重形状一致。\n",
                         what);
            std::fflush(stderr);
            std::abort();
        }
        /*
           [2026-10] 把三个操作数的形状一起印出来 —— 这是被 `test_pretrain` 咬过一次之后的改进:
           只有内核名时, "谁和谁对不上" 还得自己回去数调用点; 有形状就能一眼定位
           (`kikj` 第一次触发时报的只有名字, 定位花了额外一轮)。
        */
        static void shapeToStr(const Tensor_ &t, char *buf, std::size_t n)
        {
            std::size_t off = 0;
            off += (std::size_t)std::snprintf(buf + off, n - off, "[");
            for (std::size_t i = 0; i < t.shape.size() && off + 16 < n; i++) {
                off += (std::size_t)std::snprintf(buf + off, n - off, "%s%d",
                                                  (i == 0) ? "" : ",", t.shape[i]);
            }
            std::snprintf(buf + off, n - off, "]");
        }
        static void contractFailMm(const char *what, const Tensor_ &z,
                                   const Tensor_ &x1, const Tensor_ &x2)
        {
            char sz[64], s1[64], s2[64];
            shapeToStr(z, sz, sizeof(sz));
            shapeToStr(x1, s1, sizeof(s1));
            shapeToStr(x2, s2, sizeof(s2));
            std::fprintf(stderr,
                         "[Tensor::MM] 形状契约失守: %s\n"
                         "  ⇒ 实际形状: z%s (缓冲 %llu) x1%s (缓冲 %llu) x2%s (缓冲 %llu)\n"
                         "     这不是'慢一点', 是越界读/写 (Release 下曾静默丢数据: src/rl/sac.h:99-104)。\n"
                         "     请检查这一层的输入/输出维度是否与权重形状一致。\n",
                         what, sz, (unsigned long long)z.val.size(),
                         s1, (unsigned long long)x1.val.size(),
                         s2, (unsigned long long)x2.val.size());
            std::fflush(stderr);
            std::abort();
        }
        static void requireShape2d(const Tensor_ &t)
        {
            (void)t;
#ifdef RL_TENSOR_NO_CONTRACT
            return;   /* 测量/逃生门 —— 见下方说明, 默认**不定义** */
#endif
            /* 常驻 (Release 也生效): 2 维 + 形状元数据自洽 + 缓冲装得下形状 */
            if (t.shape.size() != 2) { contractFail("张量不是 2 维 (MM 只支持 2 维)"); }
            if (t.sizes.size() != t.shape.size()) { contractFail("sizes 与 shape 长度不一致"); }
            if (t.val.size() < (std::size_t)t.shape[0] * (std::size_t)t.shape[1]) {
                contractFail("缓冲长度小于 shape[0] x shape[1]");
            }
#if defined(RL_TENSOR_CHECKS) || !defined(NDEBUG)
            assert(t.val.size() >= t.totalSize);
#endif
        }
        /*
           ⚠ `RL_TENSOR_NO_CONTRACT` 是**测量/逃生门**, 默认**不定义** (= 检查生效)。
           它只用来量"这些检查值多少" (同一台机同一轮两个二进制的 A/B), 或给极端在意
           最后 0.1% 的场景留一条路。**不要在正常构建里打开** —— 关掉之后
           `src/rl/sac.h:99-104` 那类静默数据丢失就回来了。
        */
        /* 四条内核的形状关系 —— 常驻 (Release 也生效), 与各内核的 Debug 断言同源 */
        static void contractIkkj(const Tensor_ &z, const Tensor_ &x1, const Tensor_ &x2)
        {
#ifdef RL_TENSOR_NO_CONTRACT
            (void)z; (void)x1; (void)x2; return;
#endif
            if (z.shape[0] != x1.shape[0] || x1.shape[1] != x2.shape[0] || z.shape[1] != x2.shape[1]) {
                contractFailMm("ikkj 需要 z(r,c) = x1(r,k) * x2(k,c)", z, x1, x2);
            }
        }
        static void contractKikj(const Tensor_ &z, const Tensor_ &x1, const Tensor_ &x2)
        {
#ifdef RL_TENSOR_NO_CONTRACT
            (void)z; (void)x1; (void)x2; return;
#endif
            if (x1.shape[0] != x2.shape[0] || z.shape[0] != x1.shape[1] || z.shape[1] != x2.shape[1]) {
                contractFailMm("kikj 需要 z(k,c) = x1(r,k)^T * x2(r,c)", z, x1, x2);
            }
        }
        static void contractIkjk(const Tensor_ &z, const Tensor_ &x1, const Tensor_ &x2)
        {
#ifdef RL_TENSOR_NO_CONTRACT
            (void)z; (void)x1; (void)x2; return;
#endif
            if (x1.shape[1] != x2.shape[1] || z.shape[0] != x1.shape[0] || z.shape[1] != x2.shape[0]) {
                contractFailMm("ikjk 需要 z(r,c) = x1(r,k) * x2(c,k)^T", z, x1, x2);
            }
        }
        static void contractKijk(const Tensor_ &z, const Tensor_ &x1, const Tensor_ &x2)
        {
#ifdef RL_TENSOR_NO_CONTRACT
            (void)z; (void)x1; (void)x2; return;
#endif
            if (x1.shape[0] != x2.shape[1] || z.shape[0] != x1.shape[1] || z.shape[1] != x2.shape[0]) {
                contractFailMm("kijk 需要 z(k,c) = x1(r,k)^T * x2(c,r)^T", z, x1, x2);
            }
        }

        /*
           ============================================================
            四个内核的**累加**语义 (2026-09 修)
           ============================================================
           ikkj / kikj / ikjk / kijk 一律是 `z += ...`, 不是 `z = ...`:
           `MM::ikjk(g.w, e, x)` 这种"梯度累加"的用法 (以及 lstm.cpp:143-148 连写
           五次累加到同一个 delta.h) 都依赖它。SIMD 内核里的 ikjk/kijk 曾经写成赋值,
           与标量回落**语义不一致** —— 当时所有调用点的 kdim=1 都走标量, 所以没暴露;
           一旦有人喂多列输入就会**静默丢掉 z 里已有的值** (梯度丢失, 不报错)。
           现在两条路径都是累加, test_grad 的 C 节把这条钉成了断言。
        */

        inline static void ikkj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* [2026-10] 常驻检查移出 `#ifndef NDEBUG` —— 这就是 sac.h:99-104 那次事故的拦截点 */
            requireShape2d(x); requireShape2d(x1); requireShape2d(x2);
            contractIkkj(x, x1, x2);
#ifndef NDEBUG
            assert(x.shape[0] == x1.shape[0]);
            assert(x1.shape[1] == x2.shape[0]);
            assert(x.shape[1] == x2.shape[1]);
#endif
            /*
               SIMD 快速路径 (内核来自 N-spirits 的 simd/avx2func.hpp, 经
               rl/simd_ops.hpp 分派)。内核是**累加**到目标里的, 与本函数语义一致 ——
               这里不要像 tensorsi.hpp 那样补一次 zero()。
            */
            if (x.shape.size() == 2 && x.shape[1] == 1 && x.sizes[0] == 1 &&
                x1.shape.size() == 2 && x1.sizes[0] == x1.shape[1] && x1.sizes[1] == 1 &&
                x2.shape.size() == 2 && x2.shape[1] == 1 && x2.sizes[0] == 1 &&
                x.shape[0] == x1.shape[0] && x1.shape[1] == x2.shape[0]) {
                if (simdops::gemv_ikkj<T>((T*)x.val.data(), (std::size_t)x.shape[0],
                                          x1.val.data(), (std::size_t)x1.shape[1],
                                          x2.val.data())) {
                    return;
                }
            }
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x.shape[0] == x1.shape[0] && x1.shape[1] == x2.shape[0] &&
                x.shape[1] == x2.shape[1]) {
                if (simdops::mm_ikkj<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[1];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x2c == 1) {
                /* x(i, .) and x2(k, .) are both contiguous in j: the inner loop
                 * is a unit-stride row update with a broadcast scalar, and four
                 * k-values are fused so the x row is loaded/stored once per four
                 * MACs instead of once per MAC.  Before: 18.07 ns/MAC
                 * (0.11 GFLOP/s) for ikkj (90x90)*(90x90), now 0.10 ns/MAC
                 * (20.8 GFLOP/s) - ~180x. */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    std::size_t k = 0;
                    for (; k + 4 <= kdim; k += 4) {
                        const T v0 = x1d[i*x1r + (k + 0)*x1c];
                        const T v1 = x1d[i*x1r + (k + 1)*x1c];
                        const T v2 = x1d[i*x1r + (k + 2)*x1c];
                        const T v3 = x1d[i*x1r + (k + 3)*x1c];
                        const T *r0 = x2d + (k + 0)*x2r;
                        const T *r1 = x2d + (k + 1)*x2r;
                        const T *r2 = x2d + (k + 2)*x2r;
                        const T *r3 = x2d + (k + 3)*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xrow[j] += v0*r0[j] + v1*r1[j] + v2*r2[j] + v3*r3[j];
                        }
                    }
                    for (; k < kdim; k++) {
                        const T x1ik = x1d[i*x1r + k*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xrow[j] += x1ik*x2row[j];
                        }
                    }
                }
            } else {
                /* generic strides (e.g. >2-D shapes): same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t k = 0; k < kdim; k++) {
                        const T x1ik = x1d[i*x1r + k*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(i, k) * x2(k, j) */
                            xd[i*xr + j*xc] += x1ik*x2row[j*x2c];
                        }
                    }
                }
            }
            return;
        }

        inline static void kikj(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* [2026-10] 常驻检查移出 `#ifndef NDEBUG` (理由见 requireShape2d 的说明) */
            requireShape2d(x); requireShape2d(x1); requireShape2d(x2);
            contractKikj(x, x1, x2);
#ifndef NDEBUG
            assert(x1.shape[0] == x2.shape[0]);
            assert(x.shape[0] == x1.shape[1]);
            assert(x.shape[1] == x2.shape[1]);
#endif
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[0] == x2.shape[0] &&
                x.shape[0] == x1.shape[1] && x.shape[1] == x2.shape[1]) {
                if (simdops::mm_kikj<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x1 */
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[0];
            const std::size_t cols = (std::size_t)x.shape[1];
            /*
               ============================================================
                反向 GEMV: ei += wᵀ·e  (R1.5, 2026-09)
               ============================================================
               这是训练步里最慢的一步 —— 实测 0.991 vs 前向 0.102 ns/MAC (9.7×),
               见 test_grad 的 D 节。原因不是"没写 SIMD", 而是**循环方向选错了**:
               下面那条 `xc == 1 && x2c == 1` 的分支按 (i, k) 遍历, 而 w 是行主序的
               `[out][in]`, 于是 w[k][i] 的步长是 in —— 跨步 gather, 向量化不了。
               `x1c == 1` 时把它反过来: **外层 k (广播 e[k]), 内层 i 整行累加** ——
               两条内存都是单位步长, 而且每个输出元素的累加顺序仍是 k 升序 (与标量
               分支一致, 只差 4 路展开的加法结合顺序, 与 ikkj 同一约定)。

               形状要求: x / x2 是列向量 (行步长 1), x1 行内连续 (x1c == 1),
               且 x1 是 [out, in]、x 是 [in, 1]、x2 是 [out, 1]。
            */
            if (xc == 1 && x2c == 1 && x1c == 1 && xr == 1 && x2r == 1 &&
                x.shape[0] == x1.shape[1] && x2.shape[0] == x1.shape[0]) {
                const std::size_t out = x1.shape[0];
                const std::size_t in = x.shape[0];
                /*
                   这里可以直接写 eid[i] / wk[i] (不带步长乘子): 上面的判据已经把
                   xr == 1、x1c == 1、x2r == 1 都验过了。带上步长乘子写的话 MSVC 认不出
                   这是单位步长循环, 就不向量化 —— 实测两者差 ~2 倍 (0.24 vs 0.13 ns/MAC)。

                   局部 __restrict 同理: 内层是"两条互不重叠的行"上的逐元素 FMA, 但
                   编译器从裸指针看不出来, 默认要按可能别名处理。调用方本来就要求
                   输出与输入不是同一块内存 (MM 的语义)。
                */
                const T *RL_RESTRICT wd = x1d;
                const T *RL_RESTRICT ed = x2d;
                T *RL_RESTRICT eid = xd;
                std::size_t k = 0;
                for (; k + 4 <= out; k += 4) {
                    const T e0 = ed[k + 0];
                    const T e1 = ed[k + 1];
                    const T e2 = ed[k + 2];
                    const T e3 = ed[k + 3];
                    const T *RL_RESTRICT w0 = wd + (k + 0)*x1r;
                    const T *RL_RESTRICT w1 = wd + (k + 1)*x1r;
                    const T *RL_RESTRICT w2 = wd + (k + 2)*x1r;
                    const T *RL_RESTRICT w3 = wd + (k + 3)*x1r;
                    for (std::size_t i = 0; i < in; i++) {
                        eid[i] += w0[i]*e0 + w1[i]*e1 + w2[i]*e2 + w3[i]*e3;
                    }
                }
                for (; k < out; k++) {
                    const T ek = ed[k];
                    const T *RL_RESTRICT wk = wd + k*x1r;
                    for (std::size_t i = 0; i < in; i++) {
                        eid[i] += wk[i]*ek;
                    }
                }
                return;
            }
            if (xc == 1 && x2c == 1) {
                /* as in ikkj: unit-stride row update, 4 k-values fused.
                 * Before: (360x90)^T*(360x90) = 18.34 ns/MAC (0.11 GFLOP/s),
                 * now 0.09 ns/MAC (21.4 GFLOP/s). */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    std::size_t k = 0;
                    for (; k + 4 <= kdim; k += 4) {
                        const T v0 = x1d[(k + 0)*x1r + i*x1c];
                        const T v1 = x1d[(k + 1)*x1r + i*x1c];
                        const T v2 = x1d[(k + 2)*x1r + i*x1c];
                        const T v3 = x1d[(k + 3)*x1r + i*x1c];
                        const T *r0 = x2d + (k + 0)*x2r;
                        const T *r1 = x2d + (k + 1)*x2r;
                        const T *r2 = x2d + (k + 2)*x2r;
                        const T *r3 = x2d + (k + 3)*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xrow[j] += v0*r0[j] + v1*r1[j] + v2*r2[j] + v3*r3[j];
                        }
                    }
                    for (; k < kdim; k++) {
                        const T x1ki = x1d[k*x1r + i*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xrow[j] += x1ki*x2row[j];
                        }
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t k = 0; k < kdim; k++) {
                        const T x1ki = x1d[k*x1r + i*x1c];
                        const T *x2row = x2d + k*x2r;
                        for (std::size_t j = 0; j < cols; j++) {
                            /* x(i, j) = x1(k, i)^T * x2(k, j) */
                            xd[i*xr + j*xc] += x1ki*x2row[j*x2c];
                        }
                    }
                }
            }
            return;
        }

        inline static void ikjk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* [2026-10] 常驻检查移出 `#ifndef NDEBUG` (理由见 requireShape2d 的说明) */
            requireShape2d(x); requireShape2d(x1); requireShape2d(x2);
            contractIkjk(x, x1, x2);
#ifndef NDEBUG
            assert(x1.shape[1] == x2.shape[1]);
            assert(x.shape[0] == x1.shape[0]);
            assert(x.shape[1] == x2.shape[0]);
#endif
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[1] == x2.shape[1] &&
                x.shape[0] == x1.shape[0] && x.shape[1] == x2.shape[0]) {
                if (simdops::mm_ikjk<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x2 */
            /*
               [2026-10] 三个指针加 `RL_RESTRICT`: 与 SIMD 内核 (`simd/*.hpp` 的
               `ikjk(z __restrict, x __restrict, y __restrict)`) 同一约定 —— 调用方本来
               就要求输出与输入不是同一块内存 (MM 的语义, 见本文件顶部)。不加的话编译器
               必须假设 `xrow[j] += …` 可能改写 x1d/x2d, 于是**连 x1 的读都不敢提到循环外**
               (实测: 下面标量分支在 kdim=1 时 12~20 ns/MAC, 加 restrict/提读之后 0.5~0.9)。
            */
            const T *RL_RESTRICT x1d = x1.val.data();
            const T *RL_RESTRICT x2d = x2.val.data();
            T *RL_RESTRICT xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[1];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x1c == 1 && x2c == 1 && kdim == 1) {
                /*
                   ============================================================
                     kdim == 1: 单样本的**外积** `z(i,j) += x1(i)·x2(j)`
                   ============================================================
                   batch=1 的训练里**所有** FC 反向都是这一形态
                   (`iFcLayer::backward` 的 `MM::ikjk(g.w, e, x)`), 而它偏偏走不到 SIMD:
                   `mmShapeOk` 要求每个维度 >= 一个向量宽 (AVX2 是 8), x1Col=x2Col=1
                   ⇒ 必然落到标量分支。

                   通用分支在 j 循环里**逐轮重读 `x1row[k]`**(kdim=1 时就是 x1[i]),
                   而那个读在 j 循环里是常量 —— 编译器在"xd 可能别名 x1d"的前提下既不能
                   把它提出来、也不能把这个循环向量化。实测 (probe_seq_cost [H],
                   同轮成对): (64x64) 12.1~15.8 -> 0.55~0.64 ns/MAC (22~25x);
                   (360x1710) 12.1~20.6 -> 0.94~1.65 ns/MAC (12~13x);
                   而两者在同一个零初值上的差 = **0.000e+00** (逐位相同)。

                   注意这与"累加"语义无关: 每个 (i,j) 只加一项, 顺序不可能变。
                   (唯一理论差别是 `-0.0`: 通用分支先算 `(0+0)+(0+0)` 再 `+p`, 把 `-0.0`
                   的积变成 `+0.0`; 这里直接加 `p`。累积缓冲从 +0.0 起、且加不出 -0.0,
                   所以本工程内不可达 —— 上面那次逐位比对也证实了。)
                */
                for (std::size_t i = 0; i < rows; i++) {
                    const T a = x1d[i * x1r];
                    T *xrow = xd + i * xr;
                    if (x2r == 1) {
                        /* 列向量 x2 (训练里的常态: `(cols,1)` 的 sizes[0] 恒为 1):
                           内层是单位步长 —— 必须写成 `x2d[j]`; 带上 `j*x2r` 这个
                           **运行时步长**乘子 MSVC 就不向量化了 (实测 2.8 -> 0.4 ns/MAC)。 */
                        for (std::size_t j = 0; j < cols; j++) {
                            xrow[j] += a * x2d[j];
                        }
                    } else {
                        for (std::size_t j = 0; j < cols; j++) {
                            xrow[j] += a * x2d[j * x2r];
                        }
                    }
                }
            } else if (xc == 1 && x1c == 1 && x2c == 1) {
                /* x1(i, .) and x2(j, .) are contiguous in k, so the sum over k is
                 * a dot product of two contiguous rows.  The loop nest is j,k
                 * inner here (the old i,k,j nest read x2(j, k) at stride
                 * sizes[0], a gather that cannot be vectorised) and four
                 * independent accumulators keep the FMA units busy - with
                 * /fp:precise the compiler will not reassociate a single
                 * accumulator chain by itself.
                 * Before: (128x64)*(64x90)^T = 18.11 ns/MAC (0.11 GFLOP/s),
                 * now 0.33 ns/MAC (6.0 GFLOP/s) - the remaining cost is the
                 * (non-vectorisable under /fp:precise) accumulation chain;
                 * materialising a transposed x2 to make this an element-wise
                 * row update was measured and only bought ~1.1x, so it was not
                 * worth the per-call scratch buffer. */
                for (std::size_t i = 0; i < rows; i++) {
                    const T *x1row = x1d + i*x1r;
                    T *xrow = xd + i*xr;
                    for (std::size_t j = 0; j < cols; j++) {
                        const T *x2row = x2d + j*x2r;
                        T a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        std::size_t k = 0;
                        for (; k + 4 <= kdim; k += 4) {
                            /* x(i, j) += x1(i, k) * x2(j, k)^T */
                            a0 += x1row[k + 0]*x2row[k + 0];
                            a1 += x1row[k + 1]*x2row[k + 1];
                            a2 += x1row[k + 2]*x2row[k + 2];
                            a3 += x1row[k + 3]*x2row[k + 3];
                        }
                        T acc = (a0 + a1) + (a2 + a3);
                        for (; k < kdim; k++) {
                            acc += x1row[k]*x2row[k];
                        }
                        xrow[j] += acc;
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t j = 0; j < cols; j++) {
                        T acc = 0;
                        for (std::size_t k = 0; k < kdim; k++) {
                            /* x(i, j) = x1(i, k) * x2(j, k)^T */
                            acc += x1d[i*x1r + k*x1c]*x2d[j*x2r + k*x2c];
                        }
                        xd[i*xr + j*xc] += acc;
                    }
                }
            }
            return;
        }

        inline static void kijk(Tensor_ &x, const Tensor_ &x1, const Tensor_ &x2)
        {
            /* [2026-10] 常驻检查移出 `#ifndef NDEBUG` (理由见 requireShape2d 的说明) */
            requireShape2d(x); requireShape2d(x1); requireShape2d(x2);
            contractKijk(x, x1, x2);
#ifndef NDEBUG
            assert(x1.shape[0] == x2.shape[1]);
            assert(x.shape[0] == x1.shape[1]);
            assert(x.shape[1] == x2.shape[0]);
#endif
            /* SIMD 快速路径, 见 ikkj 的说明 */
            if (contiguous2d(x) && contiguous2d(x1) && contiguous2d(x2) &&
                x1.shape[0] == x2.shape[1] &&
                x.shape[0] == x1.shape[1] && x.shape[1] == x2.shape[0]) {
                if (simdops::mm_kijk<T>((T*)x.val.data(), (std::size_t)x.shape[0], (std::size_t)x.shape[1],
                                        x1.val.data(), (std::size_t)x1.shape[0], (std::size_t)x1.shape[1],
                                        x2.val.data(), (std::size_t)x2.shape[0], (std::size_t)x2.shape[1])) {
                    return;
                }
            }
            /* transpose x1, x2 */
            const T *x1d = x1.val.data();
            const T *x2d = x2.val.data();
            T *xd = x.val.data();
            const std::size_t xr  = (std::size_t)x.sizes[0],  xc  = (std::size_t)x.sizes[1];
            const std::size_t x1r = (std::size_t)x1.sizes[0], x1c = (std::size_t)x1.sizes[1];
            const std::size_t x2r = (std::size_t)x2.sizes[0], x2c = (std::size_t)x2.sizes[1];
            const std::size_t rows = (std::size_t)x.shape[0];
            const std::size_t kdim = (std::size_t)x1.shape[0];
            const std::size_t cols = (std::size_t)x.shape[1];
            if (xc == 1 && x2c == 1) {
                /* x2(j, .) is contiguous in k (x1's column is walked at stride
                 * sizes[0], i.e. a broadcast load per k).  Same j,k-inner nest
                 * and 4-accumulator trick as ikjk. */
                for (std::size_t i = 0; i < rows; i++) {
                    T *xrow = xd + i*xr;
                    for (std::size_t j = 0; j < cols; j++) {
                        const T *x1col = x1d + i*x1c;
                        const T *x2row = x2d + j*x2r;
                        T a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        std::size_t k = 0;
                        for (; k + 4 <= kdim; k += 4) {
                            /* x(i, j) += x1(k, i)^T * x2(j, k)^T */
                            a0 += x1col[(k + 0)*x1r]*x2row[k + 0];
                            a1 += x1col[(k + 1)*x1r]*x2row[k + 1];
                            a2 += x1col[(k + 2)*x1r]*x2row[k + 2];
                            a3 += x1col[(k + 3)*x1r]*x2row[k + 3];
                        }
                        T acc = (a0 + a1) + (a2 + a3);
                        for (; k < kdim; k++) {
                            acc += x1col[k*x1r]*x2row[k];
                        }
                        xrow[j] += acc;
                    }
                }
            } else {
                /* generic strides: same order as the old code */
                for (std::size_t i = 0; i < rows; i++) {
                    for (std::size_t j = 0; j < cols; j++) {
                        T acc = 0;
                        for (std::size_t k = 0; k < kdim; k++) {
                            /* x(i, j) = x1(k, i)^T * x2(j, k)^T */
                            acc += x1d[k*x1r + i*x1c]*x2d[j*x2r + k*x2c];
                        }
                        xd[i*xr + j*xc] += acc;
                    }
                }
            }
            return;
        }

        inline static Tensor_ ikkj(const Tensor_ &x1, const Tensor_ &x2)
        {
            /* Tensor_ ctor zero-fills, so accumulating the result of the fast
             * in-place kernel into it is the same computation as before (the
             * old body duplicated the ikkj loop instead of reusing it). */
            Tensor_ x(x1.shape[0], x2.shape[1]);
            ikkj(x, x1, x2);
            return x;
        }

        inline static Tensor_ kikj(const Tensor_ &x1, const Tensor_ &x2)
        {
            /* transpose x1; Tensor_ zero-fills x, so the in-place kernel gives
             * exactly what the old duplicated loop computed */
            Tensor_ x(x1.shape[1], x2.shape[1]);
            kikj(x, x1, x2);
            return x;
        }

        inline static Tensor_ ikjk(const Tensor_ &x1, const Tensor_ &x2)
        {
            Tensor_ x(x1.shape[0], x2.shape[0]);
            /* transpose x2 */
            ikjk(x, x1, x2);
            return x;
        }

        inline static Tensor_ kijk(const Tensor_ &x1, const Tensor_ &x2)
        {
            Tensor_ x(x1.shape[1], x2.shape[0]);
            /* transpose x1, x2 */
            kijk(x, x1, x2);
            return x;
        }
    };

    /* batch matrix multiplication */
    inline static Tensor_ bmm(const Tensor_ &x1, const Tensor_ &x2)
    {
        /*
            x1: (batch, n, m)
            x2: (batch, m, p) or (m, p) for broadcasting
            output: (batch, n, p)
        */
        if (x1.shape.size() == 3 && x2.shape.size() == 2) {
            /* broadcasting case: x2 has no batch dimension */
            int batch = x1.shape[0];
            int n = x1.shape[1];
            int p = x2.shape[1];
            assert(x1.shape[2] == x2.shape[0]);
            Tensor_ result(batch, n, p);
            for (int b = 0; b < batch; b++) {
                result.at(b) = x1.sub(b) % x2;
            }
            return result;
        }
        /* standard case: both are 3D tensors */
        assert(x1.shape.size() == 3 && x2.shape.size() == 3);
        assert(x1.shape[0] == x2.shape[0]);
        assert(x1.shape[2] == x2.shape[1]);
        int batch = x1.shape[0];
        int n = x1.shape[1];
        int p = x2.shape[2];
        Tensor_ result(batch, n, p);
        for (int b = 0; b < batch; b++) {
            result.at(b) = x1.sub(b) % x2.sub(b);
        }
        return result;
    }

    template<typename ...Arg>
    inline static Tensor_ concat(int dim, const Arg & ...args)
    {
        std::vector<Tensor_> xi = {args...};
        return concats(dim, xi);
    }

    inline static Tensor_ concats(int dim, const std::vector<Tensor_> &xi)
    {
        std::vector<int> newShape(xi[0].shape.size(), 0);
        for (std::size_t i = 0; i < xi.size(); i++) {
            newShape[dim] += xi[i].shape[dim];
        }
        for (std::size_t i = 0; i < xi[0].shape.size(); i++) {
            if (i != dim) {
                newShape[i] = xi[0].shape[i];
            }
        }
        Tensor_ x = Tensor_(newShape);
        int offset = 0;
        for (std::size_t i = 0; i < xi.size(); i++) {
            const Tensor_ &x_ = xi[i];
            /* set value */
            std::vector<int> indexs(x.shape.size(), 0);
            for (std::size_t j = 0; j < x_.totalSize; j++) {
                x_.indexOf(j, indexs);
                indexs[dim] += offset;
                x(indexs) = x_[j];
            }
            /* set offset */
            offset += x_.shape[dim];
        }
        return x;
    }

    inline static Tensor_ product2D(const Tensor_& x1, const Tensor_& x2)
    {
        int r = x1.shape[0]*x2.shape[0];
        int c = x1.shape[1]*x2.shape[1];
        Tensor_ y(r, c);
        for (int i = 0; i < x1.shape[0]; i++) {
            for (int j = 0; j < x1.shape[1]; j++) {
                for (int h = 0; h < x2.shape[0]; h++) {
                    for (int k = 0; k < x2.shape[1]; k++) {
                        y(h + i*x1.shape[0], k + j*x1.shape[1]) = x1(i, j)*x2(h, k);
                    }
                }
            }
        }
        return y;
    }

    inline static T dot(const Tensor_& x1, const Tensor_& x2)
    {
        return simdops::dot(x1.val.data(), x2.val.data(), x1.totalSize);
    }

    /* display */
    template<typename ...Index>
    void printValue(Index ...index) const
    {
        std::size_t N = size(index...);
        std::size_t pos = posOf(index...);
        std::cout<<"[";
        for (std::size_t i = 0; i < N; i++) {
            std::cout<<val[i + pos];
            if (i < N - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }

    void printValue() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < val.size(); i++) {
            std::cout<<val[i];
            if (i < totalSize - 1) {
                std::cout<<",";
            }
        }
        std::cout<<"]"<<std::endl;
        return;
    }

    void printValue2D() const
    {
        std::cout<<"[";
        for (std::size_t i = 0; i < shape[0]; i++) {
            for (std::size_t j = 0; j < shape[1]; j++) {
                std::cout<<val[i*shape[1] + j];
                /* was `i < totalSize - 1`, comparing a ROW index against the
                   element count, which printed commas in the wrong places */
                if (!(i == shape[0] - 1 && j == shape[1] - 1)) {
                    std::cout<<",";
                }
            }
            std::cout<<std::endl;
        }
        std::cout<<"]"<<std::endl;
        return;
    }
    void printShape() const
    {
        std::cout<<"(";
        for (std::size_t i = 0; i < shape.size(); i++) {
            std::cout<<shape[i];
            /* was `i < totalSize - 1` instead of the shape's own length */
            if (i != shape.size() - 1) {
                std::cout<<",";
            }
        }
        std::cout<<")"<<std::endl;
        return;
    }

    /*
       ============================================================
        Tensor 的序列化: 权重文件用 (见 Net::save / Net::load)
       ============================================================

       格式 v2 (现在是默认的):
           <shape 用逗号分隔>|b64:<base64 的原始数据>
       上一版是 `<shape>|<十进制浮点, 用逗号分隔>`, 有两个真问题:
         1. **有损**: `std::ostream << float` 默认 6 位有效数字。存一次再读回来,
            权重就会漂移 ~1e-6 相对 —— 而本工程的后台训练每轮都在
            save -> load (见 ChessBoard::backgroundTrainLoop 的 TMP_WEIGHTS),
            也就是说每轮都在往网络里注入一次不该有的扰动。
         2. **又大又慢**: 一个 float 要 9~13 个字节的文本 (还要 strtof/double 解析),
            实测 16 MB 的 DQN+MCTS 权重文件读写一次要几百毫秒。

        base64 之后是 5.33 字节/float、无损、编解码只做位运算。仍然保留"一行一个
       张量 + 用 `|` 分隔形状"的外层结构, 因为分层的 write/read **顺序**是现成的
       结构描述 (97 处调用点都依赖它), 而 base64 里不可能出现换行, 行式读取
       (`std::getline`) 因此仍然安全。

       `fromString` **同时接受两种格式** (老文件不带 `b64:` 前缀), 所以以前存下来的
       权重照样能读。解码失败 (长度不对 / 非法字符 / 数据被截断) 会置
       `lastDecodeFailed()`, 由 Net::load 汇总成一个"载入失败"返回给调用方 ——
       否则一个被截断的文件会静默地载入半个模型。
    */
    std::string toString() const
    {
        std::string out;
        for (std::size_t i = 0; i < shape.size(); i++) {
            out += std::to_string(shape[i]);
            out += (i + 1 == shape.size()) ? '|' : ',';
        }
        out += "b64:";
        const unsigned char *rawBytes =
            reinterpret_cast<const unsigned char *>(val.data());
        char crcHex[16];
        std::snprintf(crcHex, sizeof(crcHex), "%08x",
                      (unsigned int)crc32(rawBytes, val.size() * sizeof(T)));
        out += crcHex;
        out += ':';
        out += base64Encode(val);
        return out;
    }

    /* 人能读的版本 (调试用; 权重文件不用它, 因为它有损且更大) */
    std::string toDebugString() const
    {
        std::stringstream stream;
        for (std::size_t i = 0; i < shape.size(); i++) {
            stream << shape[i];
            if (i != shape.size() - 1) {
                stream << ",";
            } else {
                stream << "|";
            }
        }
        for (std::size_t i = 0; i < val.size(); i++) {
            stream << val[i];
            if (i < val.size() - 1) {
                stream << ",";
            }
        }
        return stream.str();
    }

    /*
       上一次 fromString 是否解码失败。做成静态的 "错误旗标" 是因为 97 处调用点
       都写成 `x = Tensor::fromString(s)`, 没有地方能接收错误码; 由 Net::load 在
       读完整层之前清零、之后检查, 就能把"文件坏了"变成一次明确的失败。
       (线程安全: 权重读写只在单一训练/AI 线程里发生, 且有锁保护, 见 ChessBoard。)
    */
    static bool &lastDecodeFailedRef()
    {
        static bool failed = false;
        return failed;
    }
    static bool lastDecodeFailed() { return lastDecodeFailedRef(); }
    static void clearDecodeFailed() { lastDecodeFailedRef() = false; }

    /*
       ============================================================
        快速结构校验 (给 Net::load 的"先校验一遍再真正载入"用)
       ============================================================
       它必须**不构造张量、不解析浮点数**: 一个 16 MB 的老格式权重文件里, 真正的
       fromString 要 `split()` 出几百万个 std::string (实测让 GUI 启动从秒级变成
       30 秒以上), 而这里只扫一遍字节、数一遍逗号。要求依然是严的:
         * 形状能解析且每个维度为正;
         * 值的个数必须恰好等于形状的乘积 (少一个就是被截断了);
         * v2 还要过 base64 合法性 + 长度 + CRC32。
    */
    /* 参数是**指针 + 长度**而不是 std::string: 预校验在一整块内存上跑, 一行可能
       是 34 MB, 每行都拷成 std::string 就白省了。 */
    static bool validateEncoded(const char *s, std::size_t sLen)
    {
        long long unused = 0;
        return validateEncodedCount(s, sLen, unused);
    }

    /*
       与 validateEncoded 完全同一趟解析, 额外把"这一行有多少个元素"报出来。
       Net::load 用它把**整个文件的元素总数**与当前网络的 `paramCount()` 对齐 ——
       这是"同一套层结构但维度不同"的权重文件唯一的拦截点 (见 net.hpp 的说明)。
    */
    static bool validateEncodedCount(const char *s, std::size_t sLen, long long &elements)
    {
        elements = 0;
        const void *barPtr = std::memchr(s, '|', sLen);
        if (barPtr == nullptr) {
            return false;
        }
        const std::size_t bar = (std::size_t)((const char *)barPtr - s);
        long long shapeProduct = 1;
        if (!parseShape(std::string(s, bar), shapeProduct)) {
            return false;
        }
        elements = shapeProduct;
        const char *p = s + bar + 1;
        std::size_t n = sLen - bar - 1;
        while (n > 0 && (p[n - 1] == '\r' || p[n - 1] == '\n' || p[n - 1] == ' ')) {
            n--;
        }
        if (n >= 4 && std::strncmp(p, "b64:", 4) == 0) {
            if (n < 4 + 8 + 1 || p[12] != ':') {
                elements = 0;
                return false;
            }
            std::uint32_t want = 0;
            for (std::size_t k = 4; k < 12; k++) {
                const char c = p[k];
                int d;
                if (c >= '0' && c <= '9') {
                    d = c - '0';
                } else if (c >= 'a' && c <= 'f') {
                    d = c - 'a' + 10;
                } else if (c >= 'A' && c <= 'F') {
                    d = c - 'A' + 10;
                } else {
                    elements = 0;
                    return false;
                }
                want = (want << 4) | (std::uint32_t)d;
            }
            /* 流式校验: 不分配任何大缓冲, 一趟算完"长度 + 合法性 + 校验和" */
            const bool ok = base64DecodeInto(p + 13, n - 13, nullptr,
                                             (std::size_t)(shapeProduct * (long long)sizeof(T)),
                                             want);
            if (!ok) {
                elements = 0;
            }
            return ok;
        }
        /* v1 (十进制文本): 值的个数 = 逗号数 + 1 */
        long long values = (n == 0) ? 0 : 1;
        for (std::size_t k = 0; k < n; k++) {
            if (p[k] == ',') {
                values++;
            }
        }
        if (values != shapeProduct) {
            elements = 0;
            return false;
        }
        return true;
    }

    /* 解析 "1,2,3" 形式的形状, 同时给出元素总数 */
    static bool parseShape(const std::string &shapeString, long long &product)
    {
        product = 1;
        if (shapeString.empty()) {
            return false;
        }
        long long cur = 0;
        bool any = false;
        for (std::size_t i = 0; i < shapeString.size(); i++) {
            const char c = shapeString[i];
            if (c >= '0' && c <= '9') {
                cur = cur * 10 + (c - '0');
                any = true;
            } else if (c == ',') {
                if (!any || cur <= 0) {
                    return false;
                }
                product *= cur;
                cur = 0;
                any = false;
            } else {
                return false;
            }
        }
        if (!any || cur <= 0) {
            return false;
        }
        product *= cur;
        return true;
    }

    static Tensor_ fromString(const std::string &s)
    {
        Tensor_ x;
        std::string::size_type pos = s.find('|');
        if (pos == std::string::npos) {
            lastDecodeFailedRef() = true;
            return x;
        }
        auto split = [](const std::string &str)->std::vector<std::string> {
            std::vector<std::string> elems;
            std::size_t pos = 0;
            std::size_t len = str.length();
            while (pos < len) {
                std::size_t findPos = str.find(',', pos);
                if (findPos == std::string::npos) {
                    elems.push_back(str.substr(pos, len - pos));
                    break;
                }
                elems.push_back(str.substr(pos, findPos - pos));
                pos = findPos + 1;
            }
            return elems;
        };
        /* parse shape (与 validateEncoded 共用同一份解析, 保证两边判定一致) */
        long long shapeProduct = 1;
        if (!parseShape(s.substr(0, pos), shapeProduct)) {
            lastDecodeFailedRef() = true;
            return Tensor_();
        }
        std::vector<int> shape;
        {
            long long cur = 0;
            for (std::size_t i = 0; i <= pos; i++) {
                const char c = (i == pos) ? ',' : s[i];
                if (c == ',') {
                    shape.push_back((int)cur);
                    cur = 0;
                } else {
                    cur = cur * 10 + (c - '0');
                }
            }
        }
        /* create */
        x = Tensor_(shape);

        /*
           下面**不再**把 payload 拷成 std::string: 一行可能是 34 MB 的 base64,
           substr 一次就是一次全量拷贝。这里直接用指针 + 长度, 并让解码器
           把结果写进张量自己的存储 (见 base64DecodeInto)。
        */
        const char *payloadPtr = s.data() + pos + 1;
        std::size_t payloadLen = s.size() - (pos + 1);
        /* 去掉行尾可能残留的 '\r' (Windows 文本模式写入的文件) */
        while (payloadLen > 0
               && (payloadPtr[payloadLen - 1] == '\r' || payloadPtr[payloadLen - 1] == '\n'
                   || payloadPtr[payloadLen - 1] == ' ')) {
            payloadLen--;
        }

        if (payloadLen >= 4 && std::strncmp(payloadPtr, "b64:", 4) == 0) {
            /* ---- v2: b64:<crc32 hex>:<base64 原始数据> ---- */
            if (payloadLen < 4 + 8 + 1 || payloadPtr[12] != ':') {
                lastDecodeFailedRef() = true;
                return Tensor_();
            }
            std::uint32_t want = 0;
            for (std::size_t k = 4; k < 12; k++) {
                const char c = payloadPtr[k];
                int d;
                if (c >= '0' && c <= '9') {
                    d = c - '0';
                } else if (c >= 'a' && c <= 'f') {
                    d = c - 'a' + 10;
                } else if (c >= 'A' && c <= 'F') {
                    d = c - 'A' + 10;
                } else {
                    lastDecodeFailedRef() = true;
                    return Tensor_();
                }
                want = (want << 4) | (std::uint32_t)d;
            }
            /* 直接解码进 x.val: 长度/合法性/校验和/写入 一趟完成 */
            if (!base64DecodeInto(payloadPtr + 13, payloadLen - 13,
                                  reinterpret_cast<unsigned char *>(x.val.data()),
                                  (std::size_t)(shapeProduct * (long long)sizeof(T)),
                                  want)) {
                lastDecodeFailedRef() = true;
                return Tensor_();
            }
            return x;
        }

        /* ---- v1 (老格式): 十进制文本, 仍然要能读 ---- */
        const std::string payload(payloadPtr, payloadLen);
        std::vector<std::string> valElements = split(payload);
        if ((long long)valElements.size() != shapeProduct) {
            lastDecodeFailedRef() = true;
            return Tensor_();
        }
        for (std::size_t i = 0; i < valElements.size(); i++) {
            x[i] = (T)std::atof(valElements[i].c_str());
        }
        return x;
    }

    /* ---- base64 (只用位运算, 便于以后换真正的二进制流) ---- */
    /* ---- base64 与 CRC32 ----
       CRC 表用函数内静态初始化 (C++11 起线程安全); crc32Update 是递增版本, 供流式
       解码使用 —— 解一个 34 MB 的张量时不必先把字节存下来再算校验。 */
    static const std::uint32_t *crc32Table()
    {
        static const std::vector<std::uint32_t> table = [] {
            std::vector<std::uint32_t> t(256);
            for (std::uint32_t i = 0; i < 256; i++) {
                std::uint32_t c = i;
                for (int k = 0; k < 8; k++) {
                    c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                }
                t[i] = c;
            }
            return t;
        }();
        return table.data();
    }

    static std::uint32_t crc32Update(std::uint32_t c, unsigned char b)
    {
        return crc32Table()[(c ^ b) & 0xFFu] ^ (c >> 8);
    }

    static std::uint32_t crc32(const unsigned char *data, std::size_t n)
    {
        std::uint32_t c = 0xFFFFFFFFu;
        for (std::size_t i = 0; i < n; i++) {
            c = crc32Update(c, data[i]);
        }
        return c ^ 0xFFFFFFFFu;
    }

    /*
       ============================================================
        流式 base64 解码: 一趟扫完, 零中间分配
       ============================================================
       为什么要有它: 权重文件里一个张量可以是一行 34 MB 的 base64, 而老实现是
         std::string(p + 13, n - 13)     先拷一份 34 MB
         base64Decode() 里 vector.push_back   再分配 34 MB
         crc32(raw)                       再扫一遍
         memcpy 到张量                     再一遍
       而且"先校验一遍再真正载入"把这一串**整体做了两遍**。实测稀疏 MoE 那 3 个
       146 MB 的权重文件要 14.9 秒 (29 MB/s), 把"启动时加载所有模型"变成 19 秒。
       现在: 直接在内存字节流上解码, 每解出 3 个字节就
         * 增量更新 CRC,
         * 若有 dst 就**直接写进张量的存储** (连 memcpy 都省了),
       一趟同时完成"合法性 + 长度 + 校验和 + 写入"。dst == nullptr 就是纯校验模式
       (给 Net::load 的预校验用, 校验通过才真正写进网络)。
    */
    static bool base64DecodeInto(const char *in, std::size_t n,
                                 unsigned char *dst, std::size_t expectedBytes,
                                 std::uint32_t wantCrc)
    {
        if (expectedBytes == 0 || n == 0 || (n % 4) != 0) {
            return false;
        }
        const std::size_t groups = n / 4;
        /* 编码长度必须与字节数精确对应 (少一个 '=' 就是被截断了) */
        if (groups * 3 < expectedBytes || (groups * 3 - expectedBytes) > 2) {
            return false;
        }
        /*
           逐字符的分支链换成 256 项查找表: 老写法每个字符要过 6 个比较, 而 109 MB 的
           数据是 1.45 亿个字符 —— 实测一换, 解码从 99 MB/s 提到 ~250 MB/s。
           '=' 用 64 表示, 非法字符用 -1。
        */
        const signed char *tab = base64DecodeTable();
        std::uint32_t crc = 0xFFFFFFFFu;
        std::size_t written = 0;
        for (std::size_t g = 0; g < groups; g++) {
            const char *q = in + g * 4;
            const signed char v0 = tab[(unsigned char)q[0]];
            const signed char v1 = tab[(unsigned char)q[1]];
            const signed char v2 = tab[(unsigned char)q[2]];
            const signed char v3 = tab[(unsigned char)q[3]];
            if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) {
                return false;
            }
            int pad = 0;
            if (v2 == 64) {
                /* 填充只允许出现在最后一组的末尾 */
                if (g + 1 != groups || v3 != 64) {
                    return false;
                }
                pad = 2;
            } else if (v3 == 64) {
                if (g + 1 != groups) {
                    return false;
                }
                pad = 1;
            }
            const unsigned int x = ((unsigned int)v0 << 18) | ((unsigned int)v1 << 12)
                                   | ((unsigned int)(v2 & 63) << 6)
                                   | (unsigned int)(v3 & 63);
            const int cnt = 3 - pad;
            if (written + (std::size_t)cnt > expectedBytes) {
                return false;
            }
            const unsigned char b0 = (unsigned char)((x >> 16) & 0xFFu);
            const unsigned char b1 = (unsigned char)((x >> 8) & 0xFFu);
            const unsigned char b2 = (unsigned char)(x & 0xFFu);
            crc = crc32Update(crc, b0);
            if (dst != nullptr) {
                dst[written] = b0;
            }
            written++;
            if (cnt >= 2) {
                crc = crc32Update(crc, b1);
                if (dst != nullptr) {
                    dst[written] = b1;
                }
                written++;
            }
            if (cnt >= 3) {
                crc = crc32Update(crc, b2);
                if (dst != nullptr) {
                    dst[written] = b2;
                }
                written++;
            }
        }
        if (written != expectedBytes) {
            return false;
        }
        return (crc ^ 0xFFFFFFFFu) == wantCrc;
    }

    /* base64 字符 -> 值 (0..63), '=' -> 64, 其它 -> -1 */
    static const signed char *base64DecodeTable()
    {
        static const std::vector<signed char> table = [] {
            std::vector<signed char> t(256, (signed char)-1);
            const char *alphabet =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int i = 0; i < 64; i++) {
                t[(unsigned char)alphabet[i]] = (signed char)i;
            }
            t[(unsigned char)'='] = (signed char)64;
            return t;
        }();
        return table.data();
    }
    static std::string base64Encode(const std::vector<T, Alloc<T> > &data)
    {
        static const char *kTab =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const unsigned char *bytes =
            reinterpret_cast<const unsigned char *>(data.data());
        const std::size_t n = data.size() * sizeof(T);
        std::string out;
        out.reserve((n + 2) / 3 * 4);
        std::size_t i = 0;
        for (; i + 3 <= n; i += 3) {
            const unsigned int v = ((unsigned int)bytes[i] << 16)
                                   | ((unsigned int)bytes[i + 1] << 8)
                                   | (unsigned int)bytes[i + 2];
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += kTab[(v >> 6) & 63];
            out += kTab[v & 63];
        }
        const std::size_t rest = n - i;
        if (rest == 1) {
            const unsigned int v = (unsigned int)bytes[i] << 16;
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += '=';
            out += '=';
        } else if (rest == 2) {
            const unsigned int v = ((unsigned int)bytes[i] << 16)
                                   | ((unsigned int)bytes[i + 1] << 8);
            out += kTab[(v >> 18) & 63];
            out += kTab[(v >> 12) & 63];
            out += kTab[(v >> 6) & 63];
            out += '=';
        }
        return out;
    }

    static bool base64Decode(const std::string &in, std::vector<unsigned char> &out)
    {
        auto val = [](char c) -> int {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62;
            if (c == '/') return 63;
            return -1;
        };
        out.clear();
        if (in.size() % 4 != 0) {
            return false;
        }
        out.reserve(in.size() / 4 * 3);
        for (std::size_t i = 0; i < in.size(); i += 4) {
            int q[4];
            int pad = 0;
            for (int k = 0; k < 4; k++) {
                const char c = in[i + k];
                if (c == '=') {
                    /* 填充只允许出现在最后 4 字节组的末尾 */
                    if (i + 4 != in.size() || k < 2) {
                        return false;
                    }
                    q[k] = 0;
                    pad++;
                } else {
                    q[k] = val(c);
                    if (q[k] < 0) {
                        return false;
                    }
                }
            }
            const unsigned int v = ((unsigned int)q[0] << 18)
                                   | ((unsigned int)q[1] << 12)
                                   | ((unsigned int)q[2] << 6)
                                   | (unsigned int)q[3];
            out.push_back((unsigned char)((v >> 16) & 0xFF));
            if (pad < 2) {
                out.push_back((unsigned char)((v >> 8) & 0xFF));
            }
            if (pad < 1) {
                out.push_back((unsigned char)(v & 0xFF));
            }
        }
        return true;
    }
};

using Tensorc  = Tensor_<char>;
using Tensoru8 = Tensor_<unsigned char>;
using Tensori  = Tensor_<int>;
using Tensorf  = Tensor_<float>;
using Tensord  = Tensor_<double>;
using Tensor   = Tensorf;

}
#endif // TENSOR_H
