#ifndef RL_SIMD_OPS_HPP
#define RL_SIMD_OPS_HPP
#include <cstddef>
#include <cmath>
#include <type_traits>
#include "simd/sse2func.hpp"
#include "simd/avx2func.hpp"

/*
 * RL::simdops - SIMD dispatch layer for RL::Tensor_.
 *
 * 这一层是从 N-spirits 的 basic/simd/tensorsi.hpp 搬过来的思路: 把"能不能用 SIMD"
 * 的判据集中在一个地方, 内核来自 simd/sse2func.hpp 与 simd/avx2func.hpp (同样是
 * N-spirits 的 basic/simd/ 下的那两个文件, 原样复制过来, 只把它们对
 * ../basic_def.h 的依赖换成本地的宏定义)。
 *
 * 与 Tensorsi_ 的区别: Tensorsi_ 是另一个**类型** (继承自 N-spirits 的 Tensor_,
 * 用 Buffer/mempool 做存储), 而这里保持 RL::Tensor_ 这一个类型不变, 只把它的
 * 热点运算分派到 SIMD 内核上。这样整个 RL 库 (layer.h / net.hpp / attention.hpp /
 * moe.hpp / 各 agent) 的 API 与语义都不需要动 —— 换类型的代价是所有调用点都要
 * 重新验证一遍语义, 而收益只多一点点。
 *
 * 内核契约 (与 tensorsi.hpp 中的说明一致):
 *   * 逐元素内核要求两个操作数**形状相同**; 长度不足一个向量时不可使用
 *     (有些内核会无条件加载一整个向量)。
 *   * MatMul 内核**累加**到目标里, 调用方需要自己先清零 (与原 Tensor_ 语义一致;
 *     注意 Tensorsi_ 在调用前额外做了一次 zero, 那是因为 N-spirits 的
 *     Tensor_::MM 自身会清零, 而 RL 的不会 —— 所以这里绝不能补那次 zero)。
 *   * 目标不得与任一操作数重叠。
 *   * 归约内核 (sum/max/min/dot) 的**加法顺序**与标量版本不同, 浮点结果会有
 *     ~1e-7 量级的相对差异, 这是可接受的 (见 docs/issues_review.md B19 的交叉验证)。
 */
namespace RL {
namespace simdops {

/*
 * 本翻译单元实际启用的指令集。
 *
 * 注意 SSE2 的判据: MSVC 在 x64 下**不定义** __SSE2__ (它只定义 _M_X64 /
 * _M_IX86_FP), 所以只判 __SSE2__ 会让 MSVC 构建直接掉到标量路径 —— SSE2 是所有
 * x86-64 的基线, 上游 tensorsi.hpp 也是按"没有 AVX2 就用 SSE2"来选的。
 */
#if defined(__AVX2__)
using Instruct = ::simd::AVX2;
inline const char *instructionSet() { return "AVX2"; }
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
      (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
using Instruct = ::simd::SSE2;
inline const char *instructionSet() { return "SSE2"; }
#else
using Instruct = void;
inline const char *instructionSet() { return "scalar"; }
#endif

/*
 * 编译期是否有可用的内核: 与上面的选择保持一致 (指令集可用 + T 是 float/double)。
 * 单独抽出来是因为 hasKernel<T>() 要在 if constexpr 里当常量表达式用。
 */
#if defined(__AVX2__) || defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define RL_SIMD_HAVE_INSTRUCT 1
#else
#define RL_SIMD_HAVE_INSTRUCT 0
#endif

/* 只有 float / double 有 SIMD 内核特化 */
template<typename T>
struct IsSimdType : std::false_type {};
template<>
struct IsSimdType<float> : std::true_type {};
template<>
struct IsSimdType<double> : std::true_type {};

/* 一个向量指令处理的元素个数 (非 SIMD 类型为 0) */
template<typename T>
constexpr std::size_t step()
{
    return ::simd::Step<T>::value;
}

/* 长度是否够用: 内核可能无条件加载一整个向量, 所以 n 必须 >= step */
template<typename T>
inline bool ok(std::size_t n)
{
    return step<T>() > 0 && n >= step<T>();
}

/* 编译期是否有可用的内核 (T 受支持且本 TU 启用了指令集) */
template<typename T>
constexpr bool hasKernel()
{
#if RL_SIMD_HAVE_INSTRUCT
    return IsSimdType<T>::value;
#else
    (void)sizeof(T);
    return false;
#endif
}

/* ============================================================
 *  逐元素内核: z = y (op) x,  n >= step 时走 SIMD
 * ============================================================ */
template<typename T>
inline void fill(T *x, T value, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::fill(x, value, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        x[i] = value;
    }
}

template<typename T>
inline void add(T *z, const T *y, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::add(z, y, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i] + x[i];
    }
}

template<typename T>
inline void sub(T *z, const T *y, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            /* 内核契约: z = y - x */
            Instruct::sub(z, y, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i] - x[i];
    }
}

template<typename T>
inline void mul(T *z, const T *y, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::mul(z, y, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i]*x[i];
    }
}

template<typename T>
inline void div(T *z, const T *y, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            /* 内核契约: z = y / x */
            Instruct::div(z, y, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i]/x[i];
    }
}

/* 与标量运算的逐元素版本: z = y (op) value */
template<typename T>
inline void add(T *z, const T *y, T value, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::add(z, y, value, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i] + value;
    }
}

template<typename T>
inline void sub(T *z, const T *y, T value, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::sub(z, y, value, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i] - value;
    }
}

template<typename T>
inline void mul(T *z, const T *y, T value, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::mul(z, y, value, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i]*value;
    }
}

template<typename T>
inline void div(T *z, const T *y, T value, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::div(z, y, value, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = y[i]/value;
    }
}

/* ============================================================
 *  归约内核
 * ============================================================ */
template<typename T>
inline T sum(const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            return (T)Instruct::sum(x, n);
        }
    }
    T s = 0;
    for (std::size_t i = 0; i < n; i++) {
        s += x[i];
    }
    return s;
}

/*
 * 注意: 内核返回的是**均方偏差** s/N, 与 Tensor_::variance(u) 的语义一致
 * (已核对 avx2func.hpp 的 `return s/float(N);`)。
 */
template<typename T>
inline T variance(const T *x, T u, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            return (T)Instruct::variance(x, u, n);
        }
    }
    T s = 0;
    for (std::size_t i = 0; i < n; i++) {
        T d = x[i] - u;
        s += d*d;
    }
    return s/T(n);
}

template<typename T>
inline T maxValue(const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            return (T)Instruct::max(x, n);
        }
    }
    T v = x[0];
    for (std::size_t i = 1; i < n; i++) {
        if (v < x[i]) {
            v = x[i];
        }
    }
    return v;
}

template<typename T>
inline T minValue(const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            return (T)Instruct::min(x, n);
        }
    }
    T v = x[0];
    for (std::size_t i = 1; i < n; i++) {
        if (v > x[i]) {
            v = x[i];
        }
    }
    return v;
}

template<typename T>
inline T dot(const T *x1, const T *x2, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            return (T)Instruct::dot(x1, x2, n);
        }
    }
    T s = 0;
    for (std::size_t i = 0; i < n; i++) {
        s += x1[i]*x2[i];
    }
    return s;
}

/* ============================================================
 *  单目内核
 * ============================================================ */
template<typename T>
inline void sqrt(T *z, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::sqrt(z, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = std::sqrt(x[i]);
    }
}

template<typename T>
inline void abs(T *z, const T *x, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::abs(z, x, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        z[i] = x[i] < T(0) ? -x[i] : x[i];
    }
}

template<typename T>
inline void clip(T *z, const T *x, T lo, T hi, std::size_t n)
{
    if constexpr (hasKernel<T>()) {
        if (ok<T>(n)) {
            Instruct::clip(z, x, lo, hi, n);
            return;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        T v = x[i];
        z[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

/* ============================================================
 *  矩阵乘内核
 *
 *  返回 true 表示已经用 SIMD 完成, false 表示调用方应当走自己的标量实现
 *  (RL::Tensor_::MM 里那套已经优化过的扁平指针循环)。
 *
 *  只在"两个操作数都是 2 维、都是连续行主序 (sizes == {cols, 1})、且每一维都不
 *  小于一个向量宽度"时才启用 —— 内核的下标是 i*col + k 这种硬编码的行主序算术,
 *  必须与 posOf() 的通用 stride 计算等价才行。
 * ============================================================ */

/*
 * 分块矩阵乘内核 (tensorsi.hpp 里的 MatMulBlocked): 每次内层迭代把 step 行 z
 * (因此 2*step 个寄存器) 一次算完, 在大矩阵上比普通 ikkj 快。
 * N-spirits 的测量是 256^3 附近打平, 512^3 时 AVX2 快 1.25x; 本机复测
 * (90x360)*(360x90) 也就是 2.9M MAC: 普通内核 0.265 ms, 分块内核 0.162 ms。
 */
template<typename Instruct>
struct MatMulBlocked
{
    static constexpr bool available = false;
    static void ikkj(float *z, std::size_t zRow, std::size_t zCol,
                     const float *x, std::size_t xRow, std::size_t xCol,
                     const float *y, std::size_t yRow, std::size_t yCol)
    {
        (void)z; (void)zRow; (void)zCol; (void)x; (void)xRow; (void)xCol;
        (void)y; (void)yRow; (void)yCol;
    }
};

template<>
struct MatMulBlocked<::simd::SSE2>
{
    static constexpr bool available = true;
    static void ikkj(float *z, std::size_t zRow, std::size_t zCol,
                     const float *x, std::size_t xRow, std::size_t xCol,
                     const float *y, std::size_t yRow, std::size_t yCol)
    {
        ::simd::SSE2::matMul32(z, zRow, zCol, x, xRow, xCol, y, yRow, yCol);
    }
};

#if defined(__AVX2__)
template<>
struct MatMulBlocked<::simd::AVX2>
{
    static constexpr bool available = true;
    static void ikkj(float *z, std::size_t zRow, std::size_t zCol,
                     const float *x, std::size_t xRow, std::size_t xCol,
                     const float *y, std::size_t yRow, std::size_t yCol)
    {
        ::simd::AVX2::matMul64(z, zRow, zCol, x, xRow, xCol, y, yRow, yCol);
    }
};
#endif

/* 分块内核只在运算量足够大时才划算 (与 tensorsi.hpp 取同一个阈值) */
inline bool useBlocked(std::size_t zRow, std::size_t zCol, std::size_t xCol)
{
    return zRow*zCol*xCol >= 128*128*128;
}

template<typename T>
inline bool mmShapeOk(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3)
{
    if constexpr (hasKernel<T>()) {
        const std::size_t s = step<T>();
        return s > 0 && n0 >= s && n1 >= s && n2 >= s && n3 >= s;
    }
    (void)n0; (void)n1; (void)n2; (void)n3;
    return false;
}

template<typename T>
inline bool mm_ikkj(T *z, std::size_t zRow, std::size_t zCol,
                    const T *x1, std::size_t x1Row, std::size_t x1Col,
                    const T *x2, std::size_t x2Row, std::size_t x2Col)
{
    (void)x1Row; (void)x2Row;
    if constexpr (hasKernel<T>()) {
        if (mmShapeOk<T>(zRow, zCol, x1Col, x2Col)) {
            /* 分块内核只有 float 版本 */
            if constexpr (std::is_same<T, float>::value) {
                if (MatMulBlocked<Instruct>::available &&
                    useBlocked(zRow, zCol, x1Col)) {
                    MatMulBlocked<Instruct>::ikkj(z, zRow, zCol, x1, x1Row, x1Col,
                                                  x2, x2Row, x2Col);
                    return true;
                }
            }
            Instruct::MatMul::ikkj(z, zRow, zCol, x1, x1Row, x1Col,
                                   x2, x2Row, x2Col);
            return true;
        }
    }
    return false;
}

/*
 * GEMV: z(i,0) += sum_k x1(i,k) * x2(k,0), 即"矩阵乘列向量"。
 *
 * 这是本库里的绝对热路径 —— Layer<Fn>::forward 的 o = w*x、attention 的
 * q = wq*x / k = wk*x / v = wv*x 全是这个形状, 而它的结果是**一列**, 所以上面那些
 * SIMD MatMul 内核用不上 (它们沿 z 的列方向向量化, z 只有一列时向量循环长度为 0,
 * tensorsi.hpp 的 longEnough() 也是因此要求每一维都 >= step)。
 * 这里换成"按行做点积": 每行与 x2 的点积正好是 Instruct::dot 的用途。
 *
 * 要求 x1、x2 连续 (sizes == {cols,1} / {1,1}), 由调用方检查。
 * 仍然是**累加**语义。
 */
template<typename T>
inline bool gemv_ikkj(T *z, std::size_t rows,
                      const T *x1, std::size_t kdim,
                      const T *x2)
{
    if constexpr (hasKernel<T>()) {
        if (kdim >= step<T>()) {
            for (std::size_t i = 0; i < rows; i++) {
                z[i] += (T)Instruct::dot(x1 + i*kdim, x2, kdim);
            }
            return true;
        }
    }
    (void)z; (void)rows; (void)x1; (void)kdim; (void)x2;
    return false;
}

template<typename T>
inline bool mm_kikj(T *z, std::size_t zRow, std::size_t zCol,
                    const T *x1, std::size_t x1Row, std::size_t x1Col,
                    const T *x2, std::size_t x2Row, std::size_t x2Col)
{
    (void)x1Row; (void)x2Row;
    if constexpr (hasKernel<T>()) {
        if (mmShapeOk<T>(zRow, zCol, x1Row, x2Col)) {
            Instruct::MatMul::kikj(z, zRow, zCol, x1, x1Row, x1Col,
                                   x2, x2Row, x2Col);
            return true;
        }
    }
    return false;
}

template<typename T>
inline bool mm_ikjk(T *z, std::size_t zRow, std::size_t zCol,
                    const T *x1, std::size_t x1Row, std::size_t x1Col,
                    const T *x2, std::size_t x2Row, std::size_t x2Col)
{
    (void)x1Row; (void)x2Row;
    if constexpr (hasKernel<T>()) {
        if (mmShapeOk<T>(zRow, zCol, x1Col, x2Col)) {
            Instruct::MatMul::ikjk(z, zRow, zCol, x1, x1Row, x1Col,
                                   x2, x2Row, x2Col);
            return true;
        }
    }
    return false;
}

template<typename T>
inline bool mm_kijk(T *z, std::size_t zRow, std::size_t zCol,
                    const T *x1, std::size_t x1Row, std::size_t x1Col,
                    const T *x2, std::size_t x2Row, std::size_t x2Col)
{
    (void)x1Row; (void)x2Row;
    if constexpr (hasKernel<T>()) {
        if (mmShapeOk<T>(zRow, zCol, x1Row, x2Row)) {
            Instruct::MatMul::kijk(z, zRow, zCol, x1, x1Row, x1Col,
                                   x2, x2Row, x2Col);
            return true;
        }
    }
    return false;
}

} /* namespace simdops */
} /* namespace RL */

#endif // RL_SIMD_OPS_HPP
