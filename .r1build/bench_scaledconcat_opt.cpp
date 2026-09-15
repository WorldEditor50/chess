/*
 * bench_scaledconcat_opt.cpp — 一次性 scratch: ScaledConcat 到底慢在哪 + 打包版原型
 * (不进构建, 不进 ctest; 与 bench_concat_vs_moe.cpp 同一套编译开关)
 *
 * 两件事:
 *   [A] 阶段拆解: 照抄 ScaledConcat::forward / ::backward 的每一步分别计时,
 *       看 3.56 us 的前向 / 36 us 的反向到底花在哪 (注意: 各阶段是**单独**计时,
 *       跑某个阶段时其余状态的"新鲜度"与真实一次调用不同 —— 只用来定代价量级)。
 *   [B] 打包版原型 PackedScaledConcat: N 个子层的权重打包成一张 (N*u x d) 矩阵,
 *       砍掉 16 次 embedding / block 的逐元素通用索引, 预分配反向 scratch,
 *       sigmoid+softmax 融合成单趟; 并与原版做**逐元素等价性**核对 (前向输出 +
 *       全部梯度: w1/w2/b + 每个子层的 w/b + 传给输入的 ei)。
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <random>
#include <vector>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/concat.hpp"
#include "rl/moe.hpp"
#include "rl/activate.h"
#include "rl/net.hpp"

using namespace RL;

static float urand(float lo, float hi)
{
    static std::mt19937 g(20240915u);
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

static int g_repsFwd = 20000;
static int g_repsBwd = 5000;

static double nowUs()
{
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

template<typename F>
static double perCall(F &&f, int reps)
{
    f();
    const double t0 = nowUs();
    for (int i = 0; i < reps; i++) {
        f();
    }
    return (nowUs() - t0) / reps;
}

static float maxAbsDiff(const Tensor &a, const Tensor &b)
{
    if (a.totalSize != b.totalSize) {
        return 1e30f;
    }
    float m = 0;
    for (std::size_t i = 0; i < a.totalSize; i++) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static float maxAbs(const Tensor &a)
{
    float m = 0;
    for (std::size_t i = 0; i < a.totalSize; i++) {
        const float d = std::fabs(a[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

/* ============================================================================
 * [A] 原版 ScaledConcat 的阶段拆解
 * ==========================================================================*/
template<int N, int U>
static void profileOriginal(int D)
{
    using SC = ScaledConcat<Layer<Sigmoid>, N>;
    auto sc = SC::_(Layer<Sigmoid>(D, U, true, true), D, U, true);
    const int out = U * N;

    Tensor x(D, 1);
    Random::uniform(x, -1.0f, 1.0f);
    sc->forward(x);

    std::printf("\n---- 阶段拆解: ScaledConcat<Sigmoid,%d> u=%d @ D=%d (out=%d) ----\n",
                N, U, D, out);

    /* ---- 前向 ---- */
    const double s_sub   = perCall([&]{ for (int i = 0; i < N; i++) sc->layers[i].forward(x); }, g_repsFwd);
    const double s_embed = perCall([&]{ for (int i = 0; i < N; i++) sc->a.embedding({U*i, 0}, sc->layers[i].o); }, g_repsFwd);
    const double s_smax  = perCall([&]{ softmax(sc->a); }, g_repsFwd);
    const double s_w1    = perCall([&]{ sc->o.zero(); Tensor::MM::ikkj(sc->o, sc->w1, sc->a); }, g_repsFwd);
    const double s_w2    = perCall([&]{ sc->o.zero(); Tensor::MM::ikkj(sc->o, sc->w2, x); }, g_repsFwd);
    const double s_bias  = perCall([&]{ sc->o += sc->b; }, g_repsFwd);
    const double s_tanh  = perCall([&]{ for (int i = 0; i < out; i++) sc->o[i] = Tanh::f(sc->o[i]); }, g_repsFwd);
    const double s_full  = perCall([&]{ sc->forward(x); }, g_repsFwd);

    std::printf("  前向:\n");
    std::printf("    %2d 个子层 forward        %9.3f us\n", N, s_sub);
    std::printf("    %2d 次 embedding          %9.3f us\n", N, s_embed);
    std::printf("    softmax(a)               %9.3f us\n", s_smax);
    std::printf("    gemv w1 (o=W1*a)         %9.3f us\n", s_w1);
    std::printf("    gemv w2 (o=W2*x)         %9.3f us\n", s_w2);
    std::printf("    o += b                   %9.3f us\n", s_bias);
    std::printf("    tanh(%d)                  %9.3f us\n", out, s_tanh);
    std::printf("    小计                     %9.3f us\n",
                s_sub + s_embed + s_smax + s_w1 + s_w2 + s_bias + s_tanh);
    std::printf("    **完整一次 forward**     %9.3f us\n", s_full);

    /* ---- 反向 ---- */
    Tensor err(out, 1);
    Random::uniform(err, -0.5f, 0.5f);
    sc->forward(x);
    sc->e = err;

    Tensor dz(out, 1), dzSoft(out, 1), da(out, 1);
    Tensor ei(D, 1);

    const double b_alloc = perCall([&]{ Tensor a1(out, 1); Tensor a2(out, 1); Tensor a3(out, 1); }, g_repsBwd);
    const double b_dz    = perCall([&]{ for (int i = 0; i < out; i++) dz[i] = Tanh::df(sc->o[i]) * sc->e[i]; }, g_repsBwd);
    const double b_dzs   = perCall([&]{ dzSoft.zero(); Tensor::MM::kikj(dzSoft, sc->w1, dz); }, g_repsBwd);
    const double b_jvp   = perCall([&]{ Softmax::jacobian_transpose_mul(sc->a, dzSoft, da); }, g_repsBwd);
    const double b_slice = perCall([&]{ for (int i = 0; i < N; i++) sc->layers[i].e = da.block({U*i, 0}, {U, 1}); }, g_repsBwd);
    const double b_ei    = perCall([&]{ ei.zero(); Tensor::MM::kikj(ei, sc->w2, dz); }, g_repsBwd);
    const double b_ik1   = perCall([&]{ Tensor::MM::ikjk(sc->g.w1, dz, sc->a); }, g_repsBwd);
    const double b_ik2   = perCall([&]{ Tensor::MM::ikjk(sc->g.w2, dz, x); }, g_repsBwd);
    /* 同一份数学, 手写连续指针版 —— 用来定位 ikjk 的回退代价 */
    Tensor tmpG(out, out), tmpG2(out, D);
    const double b_ik1raw = perCall([&]{
        float *gd = tmpG.val.data();
        const float *ud = dz.val.data();
        const float *vd = sc->a.val.data();
        for (int i = 0; i < out; i++) {
            const float ui = ud[i];
            float *row = gd + (std::size_t)i * out;
            for (int j = 0; j < out; j++) {
                row[j] += ui * vd[j];
            }
        }
    }, g_repsBwd);
    const double b_ik2raw = perCall([&]{
        float *gd = tmpG2.val.data();
        const float *ud = dz.val.data();
        const float *vd = x.val.data();
        for (int i = 0; i < out; i++) {
            const float ui = ud[i];
            float *row = gd + (std::size_t)i * D;
            for (int j = 0; j < D; j++) {
                row[j] += ui * vd[j];
            }
        }
    }, g_repsBwd);
    const double b_gb    = perCall([&]{ sc->g.b += dz; }, g_repsBwd);
    const double b_sub   = perCall([&]{
        for (int i = 0; i < N; i++) {
            sc->layers[i].e = da.block({U*i, 0}, {U, 1});
            sc->layers[i].backward(x, ei);
        }
    }, g_repsBwd);
    const double b_full  = perCall([&]{
        sc->forward(x);
        sc->e = err;
        sc->backward(x, ei);
    }, g_repsBwd);

    std::printf("  反向:\n");
    std::printf("    3 个 Tensor 分配         %9.3f us\n", b_alloc);
    std::printf("    dz = tanh'(o)*e          %9.3f us\n", b_dz);
    std::printf("    gemv w1^T dz             %9.3f us\n", b_dzs);
    std::printf("    softmax J^T (O(n))       %9.3f us\n", b_jvp);
    std::printf("    %2d 次 block 切片         %9.3f us\n", N, b_slice);
    std::printf("    gemv w2^T dz -> ei       %9.3f us\n", b_ei);
    std::printf("    ikjk g.w1               %9.3f us   [手写外积 %.3f us]\n", b_ik1, b_ik1raw);
    std::printf("    ikjk g.w2               %9.3f us   [手写外积 %.3f us]\n", b_ik2, b_ik2raw);
    std::printf("    g.b += dz               %9.3f us\n", b_gb);
    std::printf("    %2d 个子层 backward       %9.3f us\n", N, b_sub);
    std::printf("    小计                     %9.3f us\n",
                b_alloc + b_dz + b_dzs + b_jvp + b_slice + b_ei + b_ik1 + b_ik2 + b_gb + b_sub);
    std::printf("    **完整一次 fwd+bwd**     %9.3f us  (其中 fwd %.3f us -> bwd %.3f us)\n",
                b_full, s_full, b_full - s_full);
}

/* ============================================================================
 * [B] 打包版原型
 * ==========================================================================*/
class PackedScaledConcat
{
public:
    int D, U, N, out;
    Tensor W, B;            /* 打包: (out x D) 与 (out x 1) —— N 个子层 */
    Tensor W1, W2, b;
    Tensor sig, a, o, e;    /* sig=sigmoid 输出, a=softmax(sig) */
    Tensor dz, dzSoft, da, dpre;
    Tensor gW, gB, gW1, gW2, gb;
    /* 变体 B2: 把 [W1|W2] 拼成一张, 输入 [a;x] 拼成一条 */
    Tensor Wcat, xcat;

    PackedScaledConcat(int D_, int U_, int N_)
        :D(D_), U(U_), N(N_), out(U_*N_)
    {
        W  = Tensor(out, D);
        B  = Tensor(out, 1);
        W1 = Tensor(out, out);
        W2 = Tensor(out, D);
        b  = Tensor(out, 1);
        Random::uniform(W, -1, 1);
        Random::uniform(B, -1, 1);
        Random::uniform(W1, -1, 1);
        Random::uniform(W2, -1, 1);
        Random::uniform(b, -1, 1);
        sig = Tensor(out, 1);
        a   = Tensor(out, 1);
        o   = Tensor(out, 1);
        e   = Tensor(out, 1);
        dz     = Tensor(out, 1);
        dzSoft = Tensor(out, 1);
        da     = Tensor(out, 1);
        dpre   = Tensor(out, 1);
        gW  = Tensor(out, D);
        gB  = Tensor(out, 1);
        gW1 = Tensor(out, out);
        gW2 = Tensor(out, D);
        gb  = Tensor(out, 1);
        Wcat = Tensor(out, out + D);
        xcat = Tensor(out + D, 1);
    }

    Tensor &forward(const Tensor &x)
    {
        /* N 个子层的 pre-activation 一次 gemv 算完 (原来 N 次 gemv + N 次 embedding) */
        a.zero();
        Tensor::MM::ikkj(a, W, x);
        a += B;
        /* sigmoid + softmax: 单趟融合, 不再走 4 趟 Tensor 级运算 + 两次 max/sum 归约 */
        float m = -1e30f;
        for (int i = 0; i < out; i++) {
            const float s = 1.0f / (1.0f + std::exp(-1.702f * a[i]));
            sig[i] = s;
            if (s > m) {
                m = s;
            }
        }
        float sum = 0;
        for (int i = 0; i < out; i++) {
            const float v = std::exp(sig[i] - m);
            a[i] = v;
            sum += v;
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < out; i++) {
            a[i] *= inv;
        }
        /* 偏置折进累加器初值: o = b; o += W1*a; o += W2*x (省掉一趟 o.zero() + 一趟 o+=b) */
        for (int i = 0; i < out; i++) {
            o[i] = b[i];
        }
        Tensor::MM::ikkj(o, W1, a);
        Tensor::MM::ikkj(o, W2, x);
        for (int i = 0; i < out; i++) {
            o[i] = std::tanh(o[i]);
        }
        return o;
    }

    /* 变体 B2: 一次 gemv 取代两次 (需要把 [a;x] 拼起来) */
    Tensor &forwardCat(const Tensor &x)
    {
        a.zero();
        Tensor::MM::ikkj(a, W, x);
        a += B;
        float m = -1e30f;
        for (int i = 0; i < out; i++) {
            const float s = 1.0f / (1.0f + std::exp(-1.702f * a[i]));
            sig[i] = s;
            if (s > m) {
                m = s;
            }
        }
        float sum = 0;
        for (int i = 0; i < out; i++) {
            const float v = std::exp(sig[i] - m);
            a[i] = v;
            sum += v;
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < out; i++) {
            a[i] *= inv;
        }
        for (int i = 0; i < out; i++) {
            xcat[i] = a[i];
            xcat[out + i] = x[i];
        }
        for (int i = 0; i < out; i++) {
            o[i] = b[i];
        }
        Tensor::MM::ikkj(o, Wcat, xcat);
        for (int i = 0; i < out; i++) {
            o[i] = std::tanh(o[i]);
        }
        return o;
    }

    /*
        g += u·v^T (列向量外积), 手写连续指针版。
        注意 Tensor::MM::ikjk 在这三个形状上会掉进**通用 stride 回退**:
        实测 g.w1 (64x64) 的 ikjk 要 7.15 us (4096 MAC -> 1.7 ns/MAC),
        g.w2 (64x90) 要 10.0 us —— 而同一份数学用连续指针写只要零点几 us。
    */
    static void outerAcc(Tensor &g, int rows, int cols,
                         const Tensor &u, const Tensor &v)
    {
        float *gd = g.val.data();
        const float *ud = u.val.data();
        const float *vd = v.val.data();
        for (int i = 0; i < rows; i++) {
            const float ui = ud[i];
            float *row = gd + (std::size_t)i * cols;
            for (int j = 0; j < cols; j++) {
                row[j] += ui * vd[j];
            }
        }
    }

    void backward(const Tensor &x, Tensor &ei)
    {
        for (int i = 0; i < out; i++) {
            dz[i] = (1.0f - o[i]*o[i]) * e[i];
        }
        outerAcc(gW1, out, out, dz, a);
        outerAcc(gW2, out, D, dz, x);
        gb += dz;
        dzSoft.zero();
        Tensor::MM::kikj(dzSoft, W1, dz);
        Softmax::jacobian_transpose_mul(a, dzSoft, da);
        for (int i = 0; i < out; i++) {
            dpre[i] = 1.702f * sig[i] * (1.0f - sig[i]) * da[i];
        }
        outerAcc(gW, out, D, dpre, x);      /* 原来 N 次 rank-1 更新 */
        gB += dpre;
        Tensor::MM::kikj(ei, W, dpre);      /* 原来 N 次 gemv —— 一次顶掉 */
        Tensor::MM::kikj(ei, W2, dz);       /* x 通路 (第一版漏了, 被等价性检查抓到) */
        o.zero();
        e.zero();
    }

    void zeroGrad()
    {
        gW.zero(); gB.zero(); gW1.zero(); gW2.zero(); gb.zero();
    }
};

/* 把原版的权重搬到打包结构里 (逐元素等价性检查用) */
template<int N, int U>
static void copyWeights(const std::shared_ptr<ScaledConcat<Layer<Sigmoid>, N>> &src,
                        PackedScaledConcat &dst)
{
    for (int i = 0; i < N; i++) {
        const Tensor &w = src->layers[i].w;
        const Tensor &bb = src->layers[i].b;
        for (int r = 0; r < (int)w.shape[0]; r++) {
            for (int c = 0; c < (int)w.shape[1]; c++) {
                dst.W(U*i + r, c) = w(r, c);
            }
        }
        for (int r = 0; r < (int)bb.totalSize; r++) {
            dst.B[U*i + r] = bb[r];
        }
    }
    dst.W1 = src->w1;
    dst.W2 = src->w2;
    dst.b  = src->b;
    /* [W1|W2] 拼接 */
    for (int r = 0; r < dst.out; r++) {
        for (int c = 0; c < dst.out; c++) {
            dst.Wcat(r, c) = dst.W1(r, c);
        }
        for (int c = 0; c < dst.D; c++) {
            dst.Wcat(r, dst.out + c) = dst.W2(r, c);
        }
    }
}

/* ============================================================================
 * 等价性 + 计时
 * ==========================================================================*/
template<int N, int U>
static void compareAndTime(int D)
{
    using SC = ScaledConcat<Layer<Sigmoid>, N>;
    auto sc = SC::_(Layer<Sigmoid>(D, U, true, true), D, U, true);
    PackedScaledConcat pk(D, U, N);
    const int out = U * N;

    /* 两边同权重 */
    copyWeights<N, U>(sc, pk);
    /* 原版的 w1/w2/b 也搬成打包版的 (保证两边完全一致) */
    pk.W1 = sc->w1;
    pk.W2 = sc->w2;
    pk.b  = sc->b;

    Tensor x(D, 1);
    Random::uniform(x, -1.0f, 1.0f);
    Tensor err(out, 1);
    Random::uniform(err, -0.5f, 0.5f);

    /* ---- 前向等价 ---- */
    sc->forward(x);
    Tensor yRef = sc->o;
    pk.forward(x);
    Tensor yPk = pk.o;
    const float dFwd = maxAbsDiff(yRef, yPk);

    /* ---- 反向等价 ---- */
    for (int i = 0; i < N; i++) {
        sc->layers[i].g.zero();
    }
    sc->g.zero();
    pk.zeroGrad();

    sc->forward(x);
    sc->e = err;
    Tensor eiRef(D, 1);
    eiRef.zero();
    sc->backward(x, eiRef);

    pk.forward(x);
    pk.e = err;
    Tensor eiPk(D, 1);
    eiPk.zero();
    pk.backward(x, eiPk);

    const float dEi = maxAbsDiff(eiRef, eiPk);
    const float dW1 = maxAbsDiff(sc->g.w1, pk.gW1);
    const float dW2 = maxAbsDiff(sc->g.w2, pk.gW2);
    const float dB  = maxAbsDiff(sc->g.b, pk.gb);

    /* 子层梯度: 原版每个子层的 g.w/g.b 对打包版第 i*U 行起的那一段 */
    float dSubW = 0, dSubB = 0;
    for (int i = 0; i < N; i++) {
        const Tensor &gw = sc->layers[i].g.w;
        const Tensor &gbb = sc->layers[i].g.b;
        for (int r = 0; r < (int)gw.shape[0]; r++) {
            for (int c = 0; c < (int)gw.shape[1]; c++) {
                dSubW = std::fmax(dSubW, std::fabs(gw(r, c) - pk.gW(U*i + r, c)));
            }
        }
        for (int r = 0; r < (int)gbb.totalSize; r++) {
            dSubB = std::fmax(dSubB, std::fabs(gbb[r] - pk.gB[U*i + r]));
        }
    }

    std::printf("\n---- 等价性 (ScaledConcat<Sigmoid,%d> u=%d @ D=%d) ----\n", N, U, D);
    std::printf("  前向输出   y     : |y|=%.4f  最大差 %.3e\n", maxAbs(yRef), dFwd);
    std::printf("  输入梯度   ei    : |ei_ref|=%.4f |ei_pack|=%.4f  最大差 %.3e\n",
                maxAbs(eiRef), maxAbs(eiPk), dEi);
    std::printf("  g.w1             : |g|=%.4f  最大差 %.3e\n", maxAbs(sc->g.w1), dW1);
    std::printf("  g.w2             : |g|=%.4f  最大差 %.3e\n", maxAbs(sc->g.w2), dW2);
    std::printf("  g.b              : |g|=%.4f  最大差 %.3e\n", maxAbs(sc->g.b), dB);
    std::printf("  子层 g.w -> 打包 gW : (逐元素)   最大差 %.3e\n", dSubW);
    std::printf("  子层 g.b -> 打包 gB : (逐元素)   最大差 %.3e\n", dSubB);
    std::printf("  (参考: 原版 g.w1 里 |g| 若为 0 则上面是\"拿 0 比 0\"的假通过)\n");

    /* ---- 计时 ---- */
    for (int i = 0; i < N; i++) {
        sc->layers[i].g.zero();
    }
    sc->g.zero();
    pk.zeroGrad();
    sc->forward(x);
    sc->e = err;
    pk.forward(x);
    pk.e = err;

    const double f_ref  = perCall([&]{ sc->forward(x); }, g_repsFwd);
    const double f_pk   = perCall([&]{ pk.forward(x); }, g_repsFwd);
    const double f_cat  = perCall([&]{ pk.forwardCat(x); }, g_repsFwd);
    const double fb_ref = perCall([&]{ sc->forward(x); sc->e = err; sc->backward(x, eiRef); }, g_repsBwd);

    Tensor dummy(D, 1);
    const double fb_pk  = perCall([&]{ pk.forward(x); pk.e = err; pk.backward(x, dummy); }, g_repsBwd);
    /* 打包版 backward 单独计时 (不进 forward), 用来核对阶段拆解的小计 */
    const double b_pk    = perCall([&]{ pk.e = err; pk.backward(x, dummy); }, g_repsBwd);
    /* 原版 backward 单独计时: forward 放在循环外, 但 backward 会清 o/e, 所以每轮补一次
       o/e 的代价(几十个 float)可以忽略, 用来和打包版同口径比较 */
    const double b_ref   = perCall([&]{
        for (int i = 0; i < out; i++) sc->o[i] = yRef[i];
        sc->e = err;
        sc->backward(x, eiRef);
    }, g_repsBwd);

    std::printf("\n---- 计时 (D=%d, out=%d) ----\n", D, out);
    std::printf("  原版      forward              %9.3f us\n", f_ref);
    std::printf("  打包      forward              %9.3f us   (%.2fx)\n", f_pk, f_ref / f_pk);
    std::printf("  打包+拼接 forward (一次 gemv)   %9.3f us   (%.2fx)\n", f_cat, f_ref / f_cat);
    std::printf("  原版      forward+backward     %9.3f us  (bwd %.3f us)\n", fb_ref, fb_ref - f_ref);
    std::printf("  原版      backward 单独        %9.3f us\n", b_ref);
    std::printf("  打包      backward 单独        %9.3f us   (%.2fx)\n", b_pk, b_ref / b_pk);
    std::printf("  打包      forward+backward     %9.3f us  (bwd %.3f us)  (%.2fx)\n",
                fb_pk, fb_pk - f_pk, fb_ref / fb_pk);
    std::printf("  (梯度累积量 |pk.gW1|=%.1f, |pk.gW2|=%.1f —— 非 0 说明循环真跑了)\n",
                maxAbs(pk.gW1), maxAbs(pk.gW2));

    /* 同一上下文里, 同一个数学的两种写法: MM::ikjk vs 手写连续指针外积 */
    Tensor tG(out, out), tG2(out, D);
    const double mm1  = perCall([&]{ Tensor::MM::ikjk(pk.gW1, pk.dz, pk.a); }, g_repsBwd);
    const double raw1 = perCall([&]{ PackedScaledConcat::outerAcc(tG, out, out, pk.dz, pk.a); }, g_repsBwd);
    const double mm2  = perCall([&]{ Tensor::MM::ikjk(pk.gW2, pk.dz, x); }, g_repsBwd);
    const double raw2 = perCall([&]{ PackedScaledConcat::outerAcc(tG2, out, D, pk.dz, x); }, g_repsBwd);
    std::printf("  g.w1 += dz*a^T %dx%d : ikjk %8.3f us  vs 手写外积 %8.3f us  (%.1fx)\n",
                out, out, mm1, raw1, mm1 / raw1);
    std::printf("  g.w2 += dz*x^T %dx%d : ikjk %8.3f us  vs 手写外积 %8.3f us  (%.1fx)\n",
                out, D, mm2, raw2, mm2 / raw2);
}

/* ============================================================================
 * [C] 这个慢路径不只是 ScaledConcat 的事:
 *     单样本下每个 FC 层的 backward 都是 `ikjk(g.w, e, x)`, e/x 都是列向量
 *     -> kdim == 1 的 rank-1 更新, 同样落进 ikjk 的标量回退。
 * ==========================================================================*/
static void fcProbe(int D, int OUT)
{
    auto lin = Layer<Linear>::_(D, OUT, true, true);   /* w: (OUT x D) */
    Tensor x(D, 1), e(OUT, 1), ei(D, 1);
    Random::uniform(x, -1.0f, 1.0f);
    Random::uniform(e, -1.0f, 1.0f);
    lin->forward(x);

    const double f    = perCall([&]{ lin->forward(x); }, g_repsFwd);
    const double b    = perCall([&]{ lin->e = e; ei.zero(); lin->backward(x, ei); }, g_repsBwd);
    const double braw = perCall([&]{
        lin->e = e;
        ei.zero();
        Tensor::MM::kikj(ei, lin->w, lin->e);          /* 这一路是 gemv, 有快速路径 */
        float *gd = lin->g.w.val.data();
        const float *ed = lin->e.val.data();
        const float *xd = x.val.data();
        for (int i = 0; i < OUT; i++) {                /* 只有这一路(i,k,j 三循环) 被换掉 */
            const float ui = ed[i];
            float *row = gd + (std::size_t)i * D;
            for (int j = 0; j < D; j++) {
                row[j] += ui * xd[j];
            }
        }
        lin->g.b += lin->e;
        lin->e.zero();
        lin->o.zero();
    }, g_repsBwd);

    std::printf("  FC %4d->%-5d : forward %8.3f us | backward(现版) %9.3f us (%.1fx fwd) "
                "| 换成外积 %8.3f us  (%.1fx)\n",
                D, OUT, f, b, b / f, braw, b / braw);
}

/* ============================================================================
 * [D] 结构诊断: 这个"混合"在数学上能有多"选择性" + 初始化的量级
 *
 * ScaledConcat 的门控是 softmax(sigmoid(pre)):
 *   Sigmoid::f(x) = 1/(1+exp(-1.702x)) ∈ (0, 1)
 *   -> 送进 softmax 的 logits 全部落在 (0,1), 跨度 < 1
 *   -> gate_max/gate_min = exp(a_max - a_min) < e = 2.718, 与训练/参数量无关
 * 这里把这个上界、以及"子层若是 Linear 会怎样"实测出来。
 * ==========================================================================*/
static void gateStats(const std::vector<float> &logit, const char *tag)
{
    const int n = (int)logit.size();
    float mx = logit[0], mn = logit[0];
    for (float v : logit) {
        mx = std::fmax(mx, v);
        mn = std::fmin(mn, v);
    }
    float s = 0;
    std::vector<float> g(n);
    for (int i = 0; i < n; i++) {
        g[i] = std::exp(logit[i] - mx);
        s += g[i];
    }
    float gmax = 0, gmin = 1e30f, H = 0;
    for (int i = 0; i < n; i++) {
        g[i] /= s;
        gmax = std::fmax(gmax, g[i]);
        gmin = std::fmin(gmin, g[i]);
        H -= g[i] * std::log(g[i] + 1e-30f);
    }
    std::printf("    %-30s logit 跨度 %.3f -> gate max/min %6.2fx | 熵 %.3f (均匀 %.3f)"
                " | 有效路数 %.1f/%d\n",
                tag, mx - mn, gmax / gmin, H, std::log((float)n), std::exp(H), n);
}

template<int N, int U>
static void structureProbe(int D, bool chessLike)
{
    using SC = ScaledConcat<Layer<Sigmoid>, N>;
    auto sc = SC::_(Layer<Sigmoid>(D, U, true, true), D, U, true);
    const int out = U * N;

    /* 真实的象棋状态是 90 维、稀疏、非零值 = ±(1..5)/7 (见 dqnagent.cpp:58) */
    Tensor x(D, 1);
    x.zero();
    if (chessLike) {
        const float PV[6] = { 0.0f, 1.0f/7, 2.0f/7, 2.0f/7, 3.0f/7, 5.0f/7 };
        for (int i = 0; i < D; i++) {
            if (urand(0.0f, 1.0f) < 0.36f) {
                const float v = PV[1 + (int)(urand(0.0f, 4.999f))];
                x[i] = (urand(0.0f, 1.0f) < 0.5f) ? -v : v;
            }
        }
    } else {
        Random::uniform(x, -1.0f, 1.0f);
    }
    int nz = 0;
    float ax = 0;
    for (int i = 0; i < D; i++) {
        if (x[i] != 0) nz++;
        ax += std::fabs(x[i]);
    }
    std::printf("\n---- [D] 结构诊断 (D=%d, N=%d, u=%d, out=%d) x: 非零 %d/%d, 平均|x|=%.3f\n",
                D, N, U, out, nz, D, ax / D);

    /* 子层 pre-activation (w x + b) 与 sigmoid 之后的值 —— 全部按公开成员重算 */
    std::vector<float> pre(out), sig(out);
    for (int i = 0; i < N; i++) {
        for (int r = 0; r < U; r++) {
            float acc = sc->layers[i].b[r];
            for (int c = 0; c < D; c++) {
                acc += sc->layers[i].w(r, c) * x[c];
            }
            pre[i*U + r] = acc;
            sig[i*U + r] = Sigmoid::f(acc);
        }
    }
    gateStats(sig, "现状 softmax(sigmoid(pre))");
    gateStats(pre, "若子层是 Linear: softmax(pre)");
    {
        float dsig = 0;
        for (int i = 0; i < out; i++) {
            dsig += Sigmoid::df(sig[i]);
        }
        std::printf("    门控通路的梯度因子 mean sigmoid'(y) = %.5f  (y∈{0,1} 时 -> 0; 子层"
                    "自己的梯度也走这条路)\n", dsig / out);
    }
    std::printf("    理论: logits ⊂ (0,1) => gate max/min < e^1 = 2.718 (与训练/规模无关)\n");

    /* 最终 tanh 的 pre-activation: z = W1·g + W2·x + b */
    auto finalStats = [&](const std::vector<float> &g, const char *tag) {
        float zmax = 0, sat = 0, dtanh = 0;
        for (int i = 0; i < out; i++) {
            float z = sc->b[i];
            for (int j = 0; j < out; j++) {
                z += sc->w1(i, j) * g[j];
            }
            for (int c = 0; c < D; c++) {
                z += sc->w2(i, c) * x[c];
            }
            zmax = std::fmax(zmax, std::fabs(z));
            if (std::fabs(z) > 3.0f) sat += 1.0f;
            const float t = Tanh::f(z);
            dtanh += Tanh::df(t);          /* Tanh::df 的自变量是 tanh 的**输出**, 不是 pre-activation */
        }
        std::printf("    %-30s max|z|=%.2f, |z|>3 的比例 %.0f%%, 平均 tanh'(o)=%.4f\n",
                    tag, zmax, 100.0f*sat/out, dtanh/out);
    };
    std::vector<float> gn(out);
    {
        float s = 0, mx = sig[0];
        for (float v : sig) mx = std::fmax(mx, v);
        for (int i = 0; i < out; i++) { gn[i] = std::exp(sig[i]-mx); s += gn[i]; }
        for (int i = 0; i < out; i++) gn[i] /= s;
    }
    finalStats(gn, "现状初始化");

    /* 对照: fan-in 缩放初始化 (w /= sqrt(fan_in)) —— 教科书做法 */
    for (int i = 0; i < N; i++) {
        const float s = 1.0f/std::sqrt((float)D);
        for (int r = 0; r < U; r++) {
            for (int c = 0; c < D; c++) {
                sc->layers[i].w(r, c) *= s;
            }
            sc->layers[i].b[r] *= s;
        }
    }
    for (int i = 0; i < out; i++) {
        for (int c = 0; c < D; c++) sc->w2(i, c) /= std::sqrt((float)D);
        for (int j = 0; j < out; j++) sc->w1(i, j) /= std::sqrt((float)out);
        sc->b[i] /= std::sqrt((float)out);
    }
    std::vector<float> pre2(out), sig2(out);
    for (int i = 0; i < N; i++) {
        for (int r = 0; r < U; r++) {
            float acc = sc->layers[i].b[r];
            for (int c = 0; c < D; c++) {
                acc += sc->layers[i].w(r, c) * x[c];
            }
            pre2[i*U + r] = acc;
            sig2[i*U + r] = Sigmoid::f(acc);
        }
    }
    gateStats(sig2, "fan-in 缩放后的门控");
    std::vector<float> g2(out);
    {
        float s = 0, mx = sig2[0];
        for (float v : sig2) mx = std::fmax(mx, v);
        for (int i = 0; i < out; i++) { g2[i] = std::exp(sig2[i]-mx); s += g2[i]; }
        for (int i = 0; i < out; i++) g2[i] /= s;
    }
    finalStats(g2, "fan-in 缩放初始化");
}

/* 顺带看一眼**现役**骨干的门控是不是同一回事 */
template<int E, int H>
static void moeGateProbe(int D, bool chessLike)
{
    auto m = MOE<E, H>::_(D, false);
    Tensor x(D, 1);
    x.zero();
    if (chessLike) {
        const float PV[6] = { 0.0f, 1.0f/7, 2.0f/7, 2.0f/7, 3.0f/7, 5.0f/7 };
        for (int i = 0; i < D; i++) {
            if (urand(0.0f, 1.0f) < 0.36f) {
                const float v = PV[1 + (int)(urand(0.0f, 4.999f))];
                x[i] = (urand(0.0f, 1.0f) < 0.5f) ? -v : v;
            }
        }
    } else {
        Random::uniform(x, -1.0f, 1.0f);
    }
    std::vector<float> logits(E);
    for (int i = 0; i < E; i++) {
        float acc = m->b[i];
        for (int c = 0; c < D; c++) {
            acc += m->wg(i, c) * x[c];
        }
        logits[i] = acc;
    }
    std::printf("\n---- MOE<%d,%d> @D=%d, x %s ----\n", E, H, D, chessLike ? "稀疏(象棋)" : "稠密");
    gateStats(logits, "softmax(Wg x + b), Wg~U(-0.1,0.1)");
}

int main(int argc, char **argv)
{
    if (argc > 1) g_repsFwd = std::atoi(argv[1]);
    if (argc > 2) g_repsBwd = std::atoi(argv[2]);

    std::printf("ScaledConcat 阶段拆解 + 打包版原型  (repsFwd=%d repsBwd=%d)\n", g_repsFwd, g_repsBwd);

    if (argc > 3 && std::string(argv[3]) == "struct") {
        for (int k = 0; k < 3; k++) {
            structureProbe<16, 4>(90, true);
            structureProbe<16, 4>(90, false);
            moeGateProbe<16, 16>(90, true);
            moeGateProbe<16, 16>(90, false);
            moeGateProbe<8, 4>(90, true);
        }
        return 0;
    }
    profileOriginal<16, 4>(90);
    compareAndTime<16, 4>(90);
    compareAndTime<16, 4>(1260);
    compareAndTime<16, 8>(1260);

    std::printf("\n---- [C] 单样本 FC 层 (每个 agent 的每次训练都走这条路) ----\n");
    fcProbe(64, 8100);     /* SACAZ/PPO 的策略头 */
    fcProbe(1260, 64);
    fcProbe(90, 90);
    fcProbe(1440, 64);

    structureProbe<16, 4>(90, true);
    structureProbe<16, 4>(90, false);
    return 0;
}
