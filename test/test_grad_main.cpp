/*
 * test_grad_main.cpp - 用有限差分检验"梯度传播在使用 SIMD 内核之后是否仍然正确"
 *
 * 为什么需要这个测试
 * ------------------
 * SIMD 优化改的是**热点运算的实现**, 不改数学。但"改实现不改数学"这句话只有在
 * 两个前提下才成立, 而这两个前提都不能靠读代码确认:
 *
 *   1. 每个内核的语义与它替换掉的标量代码**逐条一致**。这里最容易出问题的是
 *      "累加 vs 覆盖": 标量 MM::ikjk 是 `x(i,j) += ...` (累加), 而 SIMD 内核是
 *      赋值。只要有一个调用点依赖累加 (比如同一批里连调两次、或跨 mini-batch 累积),
 *      结果就静默地错掉。
 *   2. 走不走 SIMD 是由**形状**决定的 (每个维度都要 >= 一个向量宽度), 所以同一个
 *      表达式在小张量上走标量、大张量上走 SIMD。梯度检查必须覆盖真实的形状。
 *
 * 检查方法: 解析梯度 (Net::backward 填的 g.w) vs 中心差分
 *   (L(w+eps) - L(w-eps)) / (2*eps), 其中 L = Σ MSE 分量, 与其梯度 MSE::df 一致。
 *
 * 三个部分:
 *   A. 普通 MLP: 全部参数都查 (w + b, 逐元素)
 *   B. 生产配置 (DQN 的 MOE<16,16> + TransformerBlock<16> + TanhNorm + Sigmoid):
 *      抽样查 (逐元素太慢), 覆盖 GEMV / LayerNorm 的 mean-variance / 门控 / FFN
 *   C. MM 内核语义探针: 直接把"累加 vs 覆盖"这件事量出来, 并判断真实形状走哪条路
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/loss.h"
#include "rl/moe.hpp"
#include "rl/ppo.h"
#include "rl/transformer.hpp"
#include "rl/cpuinfo.hpp"
#include "rl/simd_ops.hpp"

using namespace RL;

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

/* L = 所有 MSE 分量之和; 它的梯度正好是 Loss::MSE::df */
static double lossOf(Net &net, const Tensor &x, const Tensor &t)
{
    Tensor &o = net.forward(x, true);
    Tensor l = Loss::MSE::f(o, t);
    double s = 0;
    for (std::size_t i = 0; i < l.size(); i++) {
        s += (double)l[i];
    }
    return s;
}

static Tensor randTensor(std::size_t r, std::size_t c, unsigned seed)
{
    Tensor t(r, c);
    unsigned s = seed;
    for (std::size_t i = 0; i < t.size(); i++) {
        s = s*1664525u + 1013904223u;
        t[i] = (float)((double)(s >> 8) / (double)(1u << 24) * 2.0 - 1.0) * 0.5f;
    }
    return t;
}

/*
 * 对 w 的每个 (抽样) 元素做中心差分, 与解析梯度 gw 比较。
 * stride: 抽样步长 (1 = 全部元素)。
 * 返回最大相对误差; 同时打印最差的那个元素。
 */
static double checkParam(Net &net, const char *tag, Tensor &w, Tensor &gw,
                         const Tensor &x, const Tensor &t, float eps, int stride)
{
    double worst = 0;
    double maxNum = 0, maxAna = 0;
    int wi = -1;
    double wnum = 0, wana = 0;
    const std::size_t n = w.size();
    for (std::size_t i = 0; i < n; i += (std::size_t)stride) {
        const float save = w[i];
        w[i] = save + eps;
        const double lp = lossOf(net, x, t);
        w[i] = save - eps;
        const double lm = lossOf(net, x, t);
        w[i] = save;

        const double num = (lp - lm) / (2.0*(double)eps);
        const double ana = (double)gw[i];
        maxNum = std::fmax(maxNum, std::fabs(num));
        maxAna = std::fmax(maxAna, std::fabs(ana));
        const double denom = std::fmax(1.0, std::fmax(std::fabs(num), std::fabs(ana)));
        const double rel = std::fabs(num - ana)/denom;
        if (rel > worst) {
            worst = rel; wi = (int)i; wnum = num; wana = ana;
        }
    }
    /*
       max|num| 与 max|ana| 分成两列:
        - 两者都是 0  -> 这个参数对 loss 根本没影响 (前向没用到它, 例如未被选中的专家)
        - num 非 0 而 ana 为 0 -> **反向没有把梯度填进来** (真正的 bug)
    */
    const char *verdict = "";
    if (maxNum == 0 && maxAna == 0) {
        verdict = "  [前向未使用/未被选中]";
    } else if (maxAna == 0) {
        verdict = "  **[反向没填梯度]**";
    }
    std::printf("    %-34s n=%-5zu max_rel=%.2e  |num|max=%.3e |ana|max=%.3e%s",
                tag, n / (std::size_t)stride, worst, maxNum, maxAna, verdict);
    std::printf("\n");

    /*
       对最差的那个元素换几个 eps 再算一遍中心差分。
         - num 随 eps 变小而收敛到 ana    -> 一致, 之前的差是差分本身的误差
         - num 在小 eps 下稳定地偏离 ana  -> 解析梯度确实不对
       (浮点下 eps 太小会被舍入淹没, 所以往下取到 1e-4 就够了。)
    */
    if (wi >= 0 && maxNum > 0 && worst > 1e-4 && eps > 1e-4f) {
        std::printf("      %-32s eps sweep @i=%d: ana=%.6f |", tag, wi, wana);
        for (float e2 = eps; e2 >= 1e-4f; e2 *= 0.1f) {
            const float save = w[wi];
            w[wi] = save + e2;
            const double lp = lossOf(net, x, t);
            w[wi] = save - e2;
            const double lm = lossOf(net, x, t);
            w[wi] = save;
            std::printf(" eps=%.0e->%.4f", e2, (lp - lm) / (2.0*(double)e2));
        }
        std::printf("\n");
    }
    return worst;
}

/* 对一个 (iFcLayer 家族) 层查 w 与 b */
static double checkFcLayer(Net &net, const char *tag, iFcLayer &ly,
                           const Tensor &x, const Tensor &t, float eps, int stride)
{
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s.w", tag);
    double worst = checkParam(net, buf, ly.w, ly.g.w, x, t, eps, stride);
    std::snprintf(buf, sizeof(buf), "%s.b", tag);
    worst = std::fmax(worst, checkParam(net, buf, ly.b, ly.g.b, x, t, eps, stride));
    return worst;
}

/* ============================================================
 *  A. 普通 MLP: 逐元素全查
 * ============================================================ */
static double mlpWorst(Net &net, Layer<Tanh> &l1, Layer<Tanh> &l2, Layer<Sigmoid> &l3,
                       const Tensor &x, const Tensor &t, float eps)
{
    double worst = 0;
    worst = std::fmax(worst, checkFcLayer(net, "l1", l1, x, t, eps, 1));
    worst = std::fmax(worst, checkFcLayer(net, "l2", l2, x, t, eps, 1));
    worst = std::fmax(worst, checkFcLayer(net, "l3", l3, x, t, eps, 1));
    return worst;
}

static void partA()
{
    std::printf("\n=== A. 普通 MLP (Tanh 64-32-16, Sigmoid 输出), 全部参数逐个查 ===\n");

    const std::size_t in = 64, h1 = 32, h2 = 16, out = 8;
    auto l1 = Layer<Tanh>::_(in, h1, true, true);
    auto l2 = Layer<Tanh>::_(h1, h2, true, true);
    auto l3 = Layer<Sigmoid>::_(h2, out, true, true);
    Net net(l1, l2, l3);

    Tensor x = randTensor(in, 1, 12345);
    Tensor t = randTensor(out, 1, 999);

    /* 解析梯度 */
    Tensor &o = net.forward(x, false);
    net.backward(x, Loss::MSE::df(o, t));

    std::printf("  eps = 1e-2 (全部 %zu 个权重/偏置):\n", l1->w.size() + l1->b.size()
                + l2->w.size() + l2->b.size() + l3->w.size() + l3->b.size());
    const double coarse = mlpWorst(net, *l1, *l2, *l3, x, t, 1e-2f);
    std::printf("  eps = 1e-3:\n");
    const double fine = mlpWorst(net, *l1, *l2, *l3, x, t, 1e-3f);
    std::printf("  -> eps=1e-2: %.3e, eps=1e-3: %.3e\n", coarse, fine);

    /*
       MLP 的激活很光滑, eps=1e-2 的截断误差已经可以忽略, 此时两个 eps 的误差
       都在 1e-4 量级 —— 差异主要是 float 舍入 (最差那个元素的解析梯度只有
       1e-4 量级, 相对误差自然被放大)。所以这里只断言"两个 eps 都在容差内",
       不去断言单调性 (单调性在复合层上才有意义, 见 partB)。
    */
    CHECK(fine < 5e-3, "MLP 全参数解析梯度与数值梯度一致 (eps=1e-3, 相对误差 < 5e-3)");
    CHECK(coarse < 5e-3, "MLP 在 eps=1e-2 下同样一致");
}

/* ============================================================
 *  B. 生产配置: DQN 的 MOE<16,16> + TransformerBlock<16> + TanhNorm + Sigmoid
 * ============================================================ */
static double prodWorst(Net &net, MOE<16, 16> &moe, TransformerBlock<16> &tb,
                        TanhNorm<Sigmoid> &tn, Layer<Sigmoid> &outl,
                        const Tensor &x, const Tensor &t, float eps, int stride)
{
    double worst = 0;
    worst = std::fmax(worst, checkParam(net, "moe.wg", moe.wg, moe.g.wg, x, t, eps, stride));
    worst = std::fmax(worst, checkParam(net, "moe.b", moe.b, moe.g.b, x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "moe.exp0.ffn_up", moe.experts[0].ffn_up,
                                          x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "moe.exp0.ffn_down", moe.experts[0].ffn_down,
                                          x, t, eps, stride));
    worst = std::fmax(worst, checkParam(net, "moe.exp0.gamma1", moe.experts[0].gamma1,
                                        moe.experts[0].g1.gamma, x, t, eps, stride));
    worst = std::fmax(worst, checkParam(net, "moe.exp0.beta1", moe.experts[0].beta1,
                                        moe.experts[0].g1.beta, x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "tb.ffn_up", tb.ffn_up, x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "tb.ffn_down", tb.ffn_down, x, t, eps, stride));
    worst = std::fmax(worst, checkParam(net, "tb.gamma1", tb.gamma1, tb.g1.gamma,
                                        x, t, eps, stride));
    worst = std::fmax(worst, checkParam(net, "tb.gamma2", tb.gamma2, tb.g2.gamma,
                                        x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "tanhnorm", tn, x, t, eps, stride));
    worst = std::fmax(worst, checkFcLayer(net, "out", outl, x, t, eps, stride));
    return worst;
}

static void partB()
{
    std::printf("\n=== B. 生产配置 (MOE<16,16>+TransformerBlock<16>+TanhNorm+Sigmoid), 抽样 ===\n");

    const std::size_t stateDim = 90, hidden = 64, actionDim = 8;
    auto moe = MOE<16, 16>::_(stateDim, true);
    auto tb  = TransformerBlock<16>::_(stateDim, true);
    auto tn  = TanhNorm<Sigmoid>::_(stateDim, hidden, true, true);
    auto outl = Layer<Sigmoid>::_(hidden, actionDim, true, true);
    Net net(moe, tb, tn, outl);

    Tensor x = randTensor(stateDim, 1, 777);
    Tensor t = randTensor(actionDim, 1, 4242);

    Tensor &o = net.forward(x, false);
    net.backward(x, Loss::MSE::df(o, t));

    const int stride = 23;      /* 抽样: 逐元素太慢 */
    std::printf("  eps = 1e-2:\n");
    const double coarse = prodWorst(net, *moe, *tb, *tn, *outl, x, t, 1e-2f, stride);
    std::printf("  eps = 1e-3:\n");
    const double fine = prodWorst(net, *moe, *tb, *tn, *outl, x, t, 1e-3f, stride);
    std::printf("  -> eps=1e-2: %.3e, eps=1e-3: %.3e\n", coarse, fine);

    /*
       判据说明: 复合层 (MOE 的门控 softmax、LayerNorm、GeLU) 的三阶导很大,
       eps=1e-2 的中心差分截断误差能到 10% 量级 —— 那是**差分**的误差, 不是梯度错。
       所以这里断言两件事:
         1. 缩小 eps 之后误差显著变小 (说明解析值就是差分收敛到的极限);
         2. 在 eps=1e-3 下误差已经很小。
    */
    CHECK(fine < 5e-2, "生产配置解析梯度与数值梯度一致 (eps=1e-3, 抽样)");
    CHECK(fine < coarse, "缩小 eps 误差变小 -> 之前的差是差分截断误差, 不是梯度错");
}

/* ============================================================
 *  C. MM 内核语义探针: 累加 vs 覆盖
 *
 *  标量实现四个 MM 都是 `z += ...` (累加)。SIMD 内核里 ikkj/kikj 也是累加
 *  (fmadd 到已加载的 z), 但 ikjk/kijk 是**赋值**。这里:
 *    C1. 用 k=1 的形状 (库里真实的"列向量"用法) 调两次, 看是否累加;
 *    C2. 用 k>=8 的形状 (会走 SIMD) 调两次, 看是否累加;
 *    C3. 直接问 dispatch: 真实形状下 mm_ikjk 到底走不走 SIMD。
 * ============================================================ */
static double naiveIkjkAcc(const Tensor &a, const Tensor &b, std::size_t r,
                           std::size_t c, std::size_t k)
{
    /* 期望语义: z(i,j) += Σ_k a(i,k)*b(j,k) */
    double s = 0;
    for (std::size_t i = 0; i < r; i++) {
        for (std::size_t j = 0; j < c; j++) {
            for (std::size_t kk = 0; kk < k; kk++) {
                s += (double)a[i*k + kk]*(double)b[j*k + kk];
            }
        }
    }
    return s;
}

static void probeIkjk(const char *tag, std::size_t r, std::size_t c, std::size_t k)
{
    Tensor a = randTensor(r, k, 11);
    Tensor b = randTensor(c, k, 22);
    Tensor z(r, c);
    z.zero();

    /* 两次 ikjk: 期望 = 2 × A·Bᵀ (标量语义是累加) */
    Tensor::MM::ikjk(z, a, b);
    Tensor::MM::ikjk(z, a, b);

    double got = 0;
    for (std::size_t i = 0; i < z.size(); i++) {
        got += (double)z[i];
    }
    const double once = naiveIkjkAcc(a, b, r, c, k);
    const double expect = 2.0*once;
    const double rel = std::fabs(got - expect)/std::fmax(1.0, std::fabs(expect));
    const bool accum = (rel < 1e-5);
    std::printf("    ikjk %-24s (r=%2zu c=%2zu k=%2zu): got/expect = %.6f  %s\n",
                tag, r, c, k, got/expect,
                accum ? "累加 -> 与标量语义一致" : "**覆盖 -> 只保留最后一次**");
    /*
       2026-09: 这条从"只打印"升级成断言。SIMD 内核里的 ikjk/kijk 原来是赋值, 与标量
       回落不一致; 当时所有调用点 kdim=1 走标量所以没暴露, 但多列输入会静默丢梯度
       (见 issues_review P1-1)。修完之后两条路径都是累加, 这里就不许再变回去。
    */
    CHECK(accum, "ikjk 是累加语义 (SIMD 内核与标量回落一致)");
}

/*
 * kijk 的同款探针: z(i,j) += Σ_k x1(k,i) * x2(j,k), k in [0, x1.shape[0])
 * (lstm.cpp:143-148 连写五次累加到同一个 delta.h 就是靠这条语义)
 */
static void probeKijk(const char *tag, std::size_t r, std::size_t c, std::size_t k)
{
    /* x1 是 (k, r), x2 是 (c, k), z 是 (r, c) */
    Tensor a = randTensor(k, r, 33);
    Tensor b = randTensor(c, k, 44);
    Tensor z(r, c);
    z.zero();

    Tensor::MM::kijk(z, a, b);
    Tensor::MM::kijk(z, a, b);

    double got = 0;
    for (std::size_t i = 0; i < z.size(); i++) {
        got += (double)z[i];
    }
    double once = 0;
    for (std::size_t i = 0; i < r; i++) {
        for (std::size_t j = 0; j < c; j++) {
            for (std::size_t kk = 0; kk < k; kk++) {
                once += (double)a[kk*r + i]*(double)b[j*k + kk];
            }
        }
    }
    const double expect = 2.0*once;
    const double rel = std::fabs(got - expect)/std::fmax(1.0, std::fabs(expect));
    const bool accum = (rel < 1e-5);
    std::printf("    kijk %-24s (r=%2zu c=%2zu k=%2zu): got/expect = %.6f  %s\n",
                tag, r, c, k, got/expect,
                accum ? "累加 -> 与标量语义一致" : "**覆盖 -> 只保留最后一次**");
    CHECK(accum, "kijk 是累加语义 (SIMD 内核与标量回落一致)");
}

static void partC()
{
    std::printf("\n=== C. MM 内核语义探针 (累加 vs 覆盖) ===\n");

    /* C1: 库里真实的列向量形状 —— kdim = 1 */
    probeIkjk("k=1 (列向量, 真实用法)", 90, 90, 1);
    /* C2: kdim >= 8 的形状 —— 会命中 SIMD 内核 */
    probeIkjk("k=32 (命中 SIMD)", 90, 90, 32);
    probeKijk("k=32 (命中 SIMD)", 90, 90, 32);
    probeKijk("k=1 (列向量)", 90, 90, 1);

    /*
       C4: (曾经想在这里钉"列向量输入下 kijk 与 kikj 等价", 用来给 lstm.cpp 的内核替换
       做背书 —— 写不出来, 因为 kijk 的**形状契约**(x1.shape[0] == x2.shape[1]) 在这种
       形状下本来就不成立, Debug 断言 (见 tensor.hpp 的 requireShape2d) 会直接拒绝这次
       调用。这恰好说明了原调用点是**越约**用法: 它靠 "列向量下 x2(j,k) 与 x2(k,j) 都
       落到 x2[k]" 碰巧算对, 换多列输入就是越界。所以那边改成了 kikj ——
       kikj 的正确性由 D 节逐元素对着朴素实现查。
    */

    /*
       C3: 直接问 dispatch 的判据。库里 ikjk/kijk 的 kdim 永远是 1
       (MM::ikjk(g.w, e, x), e 是 ∂L/∂o 列向量), 所以:
     */
    const bool simdForColumn = simdops::mmShapeOk<float>(90, 90, 1, 1);
    const bool simdForK32 = simdops::mmShapeOk<float>(90, 90, 32, 32);
    std::printf("    mmShapeOk(zRow=90, zCol=90, kdim=1 ) = %d   <- 库里的真实形状\n",
                (int)simdForColumn);
    std::printf("    mmShapeOk(zRow=90, zCol=90, kdim=32) = %d\n", (int)simdForK32);
    CHECK(!simdForColumn,
          "库里 ikjk/kijk 的真实形状 (kdim=1) 不满足 SIMD 判据 -> 走标量路径");
    CHECK(simdForK32,
          "kdim>=8 的形状会命中 SIMD 内核 (所以这个语义差异是可达的, 只是当前调用点没触发)");
}

/* ============================================================
 *  D. 前向 GEMV 与反向 GEMV 的代价对比
 *
 *  gemv_ikkj (o = w·x) 一直有专门的 SIMD 内核; kikj (ei = wᵀ·e) **没有** ——
 *  它的 mm 内核要求每一维都 >= 一个向量宽度, 而 ei 是一列 (zCol = 1), 判据不成立,
 *  于是掉回标量循环。也就是说 SIMD 只加速了前向那一半。
 *
 *  2026-09 补了反向 GEMV 分支 (循环方向从"每个 i 跨步 gather"改成"外层 k 广播、
 *  内层 i 整行累加"), 这里把两侧量出来: 修之前反向是前向的 **9.7x**, 修之后 ~2.7x。
 *  (每侧取 3 轮最小值, 见 benchIkkj/benchKikj 的说明。)
 * ============================================================ */
static void partD()
{
    std::printf("\n=== D. 前向 GEMV (ikkj, 有 SIMD) vs 反向 GEMV (kikj, 标量) ===\n");

    const std::size_t rows = 360, kdim = 90;
    Tensor w = randTensor(rows, kdim, 5);
    Tensor x = randTensor(kdim, 1, 6);
    Tensor e = randTensor(rows, 1, 7);
    Tensor o(rows, 1);
    Tensor ei(kdim, 1);

    const int iters = 20000;
    /*
       这台机器上后台进程很多 (Adobe/Edge/VMware…), 单次微基准的抖动实测有 2 倍
       (同一份代码 0.09~0.19 ns/MAC), 所以每一侧跑 3 轮**取最小值** —— 微基准里
       min 才是"这段代码能多快"的稳健估计, 均值会被抢占污染。
    */
    auto benchIkkj = [&]() {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            const auto a = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) {
                o.zero();
                Tensor::MM::ikkj(o, w, x);
            }
            const auto b = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()
                              / (double)iters;
            best = std::min(best, ns);
        }
        return best;
    };
    auto benchKikj = [&]() {
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            const auto a = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) {
                ei.zero();
                Tensor::MM::kikj(ei, w, e);
            }
            const auto b = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()
                              / (double)iters;
            best = std::min(best, ns);
        }
        return best;
    };
    const double fwdNs = benchIkkj();
    const double bwdNs = benchKikj();
    const double macs = (double)rows*(double)kdim;
    std::printf("    ikkj (o = w·x,  %zux%zu): %8.0f ns/次  = %.3f ns/MAC\n",
                rows, kdim, fwdNs, fwdNs/macs);
    std::printf("    kikj (ei = wᵀ·e, %zux%zu): %8.0f ns/次  = %.3f ns/MAC\n",
                rows, kdim, bwdNs, bwdNs/macs);
    std::printf("    -> 反向是前向的 %.1fx\n", bwdNs/fwdNs);

    /*
       正确性: 新的反向 GEMV 分支 (R1.5) 换了循环方向 (外层 k 广播、内层 i 整行累加),
       必须逐元素等于朴素定义 `ei[i] += Σ_k w[k][i]·e[k]`。同时钉住"累加到已有值上"
       这条语义 (先塞一个非零初值)。
    */
    Tensor ref(kdim, 1);
    for (std::size_t i = 0; i < kdim; i++) {
        ref[i] = 0.25f;      /* 非零初值: 覆盖型实现会把它抹掉 */
    }
    ei = ref;
    Tensor::MM::kikj(ei, w, e);
    double worst = 0.0;
    for (std::size_t i = 0; i < kdim; i++) {
        double acc = (double)ref[i];
        for (std::size_t k = 0; k < rows; k++) {
            acc += (double)w[k*kdim + i]*(double)e[k];
        }
        worst = std::fmax(worst, std::fabs(acc - (double)ei[i]));
    }
    std::printf("    反向 GEMV 与朴素实现的最大偏差 = %.3e\n", worst);
    CHECK(worst < 1e-4, "kikj (反向 GEMV) 逐元素等于 ei += wᵀ·e, 且是累加 (非零初值被保留)");
}

/* ============================================================
 *  E. R2: 训练侧只算合法列 —— 稀疏训练头的解析梯度 vs 中心差分
 *
 *  R2 的前向/反向都换了实现 (头只算合法列 + 头自己的 backward 手工做完 + 骨干走
 *  `Net::backwardFrom`), 而梯度错了**不会报错**, 只会让训练慢慢跑偏。所以这里对着
 *  中心差分挨个查: 头的权重、头的偏置、以及**骨干**(Tanh 层) 的权重 —— 后者是
 *  `backwardFrom` 那条新路径唯一的凭据。
 *
 *  损失用与 accumulateGradSparse 完全相同的公式独立算一遍 (forwardTrunk + sparseLogits
 *  + 合法集上 softmax + CE), 不调用它内部的东西 —— 否则就是自己证明自己。
 * ============================================================ */
static void partE()
{
    printf("\n=== E. R2 \u7a00\u758f\u8bad\u7ec3\u5934: \u89e3\u6790\u68af\u5ea6 vs \u4e2d\u5fc3\u5dee\u5206 ===\n");

    const std::size_t ACT = 8100;      /* = PPOMCTSAgent::ACTION_DIM (这里不引 agent 头) */
    const std::size_t SDIM = 64;
    const std::size_t HID = 32;
    /* moeAuxCoef = 0: 关掉辅助损失, 免得它混进梯度 (它由 applyGradients 注入, 这里也不调) */
    PPO ppo(SDIM, HID, ACT, 16, 0.0f, true);
    /*
       熵奖励必须关掉 (2026-09): 本节的有限差分复算的是**纯交叉熵**那一项
       (`maskedCE`), 而 `accumulateGradSparse` 在 entropyCoef>0 时还会加一项熵奖励的
       梯度。两者不同目标时差分当然对不上 —— 这不是梯度算错, 是"用错了目标函数"。
       熵项自己的正确性由 `test_ppomcts` 的 [13d] 那一节管 (它比的是"给了 p_old /
       收紧裁剪之后位移怎么变")。
    */
    ppo.entropyCoef = 0.0f;
    ppo.clipEps = 0.0f;    /* 同理: 本节不喂 p_old, 走的就是纯 CE 那一支 */

    const int legalArr[] = { 11, 222, 3333, 7000, 8099, 5, 77, 1234 };
    std::vector<int> legalIdx(legalArr, legalArr + 8);
    const int tgtArr[] = { 222, 3333 };
    std::vector<int> tgtIdx(tgtArr, tgtArr + 2);
    const float tgtProbArr[] = { 0.7f, 0.3f };
    std::vector<float> tgtProb(tgtProbArr, tgtProbArr + 2);

    Tensor state = randTensor(SDIM, 1, 4242);

    /* 独立复算损失: 与 accumulateGradSparse 同一公式 */
    auto maskedCE = [&]() -> double {
        Tensor &h = ppo.actorP.forwardTrunk(state);
        std::vector<float> logits;
        if (!ppo.actorP.sparseLogits(h, legalIdx, logits)) {
            return 1e30;
        }
        float m = logits[0];
        for (std::size_t i = 1; i < logits.size(); i++) {
            if (logits[i] > m) { m = logits[i]; }
        }
        double s = 0.0;
        std::vector<double> p(logits.size(), 0.0);
        for (std::size_t i = 0; i < logits.size(); i++) {
            p[i] = std::exp((double)logits[i] - (double)m);
            s += p[i];
        }
        double ce = 0.0;
        for (std::size_t k = 0; k < tgtIdx.size(); k++) {
            for (std::size_t i = 0; i < legalIdx.size(); i++) {
                if (legalIdx[i] == tgtIdx[k]) {
                    ce -= (double)tgtProb[k] * std::log(p[i] / s + 1e-8);
                    break;
                }
            }
        }
        return ce;
    };

    /* ---- 解析梯度: 清梯度 (RMSProp lr=0 只清 g, 不动权重) 再累积一条 ----
       注意: 熵奖励与裁剪都必须关掉 (上面已设 entropyCoef=0 / clipEps=0) —— 本节
       复算的是**纯交叉熵**那一项; 熵/裁剪两条目标由 test_ppomcts 的 [13d] 管。
       这两条一起构成"解析梯度 = 中心差分"的完整证据链 (2026-09 我在这里写过一版
       错公式: 把 dL/dπ 当成 dL/dlogit 用, 整条梯度被缩小 p_i 倍, 就是这条断言抓到的)。 */
    ppo.actorP.RMSProp(0.0f, 0.9f, 0.0f, false);
    ppo.accumulateGradSparse(state, legalIdx, tgtIdx, tgtProb, 0.0f);

    iFcLayer *head = dynamic_cast<iFcLayer *>(ppo.actorP[ppo.actorP.size() - 1]);
    iFcLayer *trunk = dynamic_cast<iFcLayer *>(ppo.actorP[ppo.actorP.size() - 2]);
    if (head == nullptr || trunk == nullptr) {
        printf("  **\u7ed3\u6784\u4e0d\u7b26: \u6700\u540e\u4e24\u5c42\u4e0d\u662f iFcLayer**\n");
        CHECK(false, "R2: 头/骨干都是全连接层");
        return;
    }

    /*
       步长取 1e-3 而不是 1e-4: 头对 w 是**线性**的 (h 固定 ⇒ logit 线性), 所以差分本身
       精确, 唯一误差来自 float32 的量化噪声 —— 步长越大信噪比越好。
       骨干那层不是线性的 (过 Tanh), 但 1e-3 的二阶项 (eps²/6·|f'''|) 仍远小于这里的容差。
       实测教训: 用 1e-4 时骨干那几个抽样的相对误差能到 0.49 —— 那不是梯度错了, 是
       差分被 float32 噪声吃了 (梯度绝对量级只有 1e-5)。
    */
    /*
       判据: **相对误差** 或 **绝对误差到达 float32 差分噪声地板**。
       噪声地板怎么来的: 损失是 double, 但 logit 是 float 累加出来的, 量化噪声 ~1e-8;
       中心差分除以 2*eps=2e-3 ⇒ 差分本身的噪声 ~5e-6。所以梯度只有 1e-5 量级时,
       "相对误差 3%" 是噪声不是错。为了不让判据变成空话, **特意挑梯度最大的那些点**
       来查 (它们远高于噪声地板, 相对误差才有意义)。
    */
    const double eps = 1e-3;
    const double kAbsFloor = 5e-6;
    auto fdOk = [](double analytic, double numeric, double absFloor) {
        const double absErr = std::fabs(analytic - numeric);
        const double rel = absErr / std::fmax(1e-6, std::fabs(numeric));
        return (absErr < absFloor) || (rel < 5e-3);
    };
    double worstHead = 0.0, worstTrunk = 0.0;

    /* ---- 头: 抽 6 个权重 (含被目标命中的行与没命中的行) ---- */
    {
        const std::size_t in = head->inputDim;
        const std::size_t rows[6] = { 222, 3333, 11, 8099, 7000, 5 };
        const std::size_t cols[6] = { 0, 1, 7, 13, 29, 31 };
        bool allOk = true;
        for (int t = 0; t < 6; t++) {
            const std::size_t a = rows[t], k = cols[t];
            const std::size_t idx = a * in + k;
            const double analytic = (double)head->g.w[idx];
            const float saved = head->w[idx];
            head->w[idx] = saved + (float)eps;
            const double up = maskedCE();
            head->w[idx] = saved - (float)eps;
            const double dn = maskedCE();
            head->w[idx] = saved;
            const double numeric = (up - dn) / (2.0 * eps);
            const double absErr = std::fabs(analytic - numeric);
            const double rel = absErr / std::fmax(1e-6, std::fabs(numeric));
            allOk = allOk && fdOk(analytic, numeric, kAbsFloor);
            worstHead = std::fmax(worstHead, rel);
            if (t < 3) {
                printf("    head.w[%4zu,%2zu]: \u89e3\u6790 %+.6e  \u5dee\u5206 %+.6e  "
                       "\u7edd\u5bf9 %.2e \u76f8\u5bf9 %.2e\n",
                       a, k, analytic, numeric, absErr, rel);
            }
        }
        printf("    \u5934\u6743\u91cd 6 \u4e2a\u62bd\u6837\u70b9: \u6700\u5927\u76f8\u5bf9\u8bef\u5dee %.3e ("
               "\u6216\u7edd\u5bf9\u8bef\u5dee < %.0e)\n", worstHead, kAbsFloor);
        CHECK(allOk, "R2: 稀疏头的权重梯度 == 中心差分 (含未被目标命中的合法行)");

        /* 头的偏置: 6 行都该有梯度 (dlogit = p - t) */
        const double biasAnalytic = (double)head->g.b[8099];
        const float savedB = head->b[8099];
        head->b[8099] = savedB + (float)eps;
        const double up = maskedCE();
        head->b[8099] = savedB - (float)eps;
        const double dn = maskedCE();
        head->b[8099] = savedB;
        const double biasNumeric = (up - dn) / (2.0 * eps);
        const double biasErr = std::fabs(biasAnalytic - biasNumeric) /
                               std::fmax(1e-6, std::fabs(biasNumeric));
        printf("    head.b[8099]: \u89e3\u6790 %+.6e  \u5dee\u5206 %+.6e  \u76f8\u5bf9\u8bef\u5dee %.2e\n",
               biasAnalytic, biasNumeric, biasErr);
        CHECK(fdOk(biasAnalytic, biasNumeric, kAbsFloor), "R2: 稀疏头的偏置梯度正确");
    }

    /* ---- 骨干 (Tanh 层): 验证 Net::backwardFrom 那条路径 ----
       抽样点取**梯度绝对值最大**的那几个 (见上面噪声地板的说明) —— 小梯度的相对
       误差在 float32 差分里没有意义。 */
    {
        const std::size_t in = trunk->inputDim;
        std::size_t bestIdx[4] = { 0, 0, 0, 0 };
        double bestAbs[4] = { -1.0, -1.0, -1.0, -1.0 };
        for (std::size_t i = 0; i < trunk->g.w.size(); i++) {
            const double g = std::fabs((double)trunk->g.w[i]);
            for (int s = 0; s < 4; s++) {
                if (g > bestAbs[s]) {
                    for (int t = 3; t > s; t--) {
                        bestAbs[t] = bestAbs[t - 1];
                        bestIdx[t] = bestIdx[t - 1];
                    }
                    bestAbs[s] = g;
                    bestIdx[s] = i;
                    break;
                }
            }
        }
        bool allOk = true;
        for (int t = 0; t < 4; t++) {
            const std::size_t idx = bestIdx[t];
            const std::size_t a = idx / in, k = idx % in;
            const double analytic = (double)trunk->g.w[idx];
            const float saved = trunk->w[idx];
            trunk->w[idx] = saved + (float)eps;
            const double up = maskedCE();
            trunk->w[idx] = saved - (float)eps;
            const double dn = maskedCE();
            trunk->w[idx] = saved;
            const double numeric = (up - dn) / (2.0 * eps);
            const double absErr = std::fabs(analytic - numeric);
            const double rel = absErr / std::fmax(1e-6, std::fabs(numeric));
            allOk = allOk && fdOk(analytic, numeric, kAbsFloor);
            worstTrunk = std::fmax(worstTrunk, rel);
            printf("    trunk.w[%2zu,%2zu]: \u89e3\u6790 %+.6e  \u5dee\u5206 %+.6e  "
                   "\u7edd\u5bf9 %.2e \u76f8\u5bf9 %.2e\n",
                   a, k, analytic, numeric, absErr, rel);
        }
        printf("    \u9aa8\u5e72\u6743\u91cd 4 \u4e2a(\u68af\u5ea6\u6700\u5927\u7684)\u62bd\u6837\u70b9: \u6700\u5927\u76f8\u5bf9\u8bef\u5dee %.3e\n",
               worstTrunk);
        CHECK(allOk, "R2: 骨干梯度 (Net::backwardFrom) == 中心差分");
    }

    /*
       不变量: 合法集上的 softmax+CE 梯度**在合法集上求和为 0** (Σp = Σt = 1)。
       全量口径下则是对全部 8100 个槽位求和为 0 —— 两条口径的这个差别正是"换学习问题"。
    */
    {
        double sumD = 0.0;
        std::vector<float> logits;
        Tensor &h = ppo.actorP.forwardTrunk(state);
        ppo.actorP.sparseLogits(h, legalIdx, logits);
        float m = logits[0];
        for (std::size_t i = 1; i < logits.size(); i++) { m = std::max(m, logits[i]); }
        double s = 0.0;
        std::vector<double> p(logits.size(), 0.0);
        for (std::size_t i = 0; i < logits.size(); i++) {
            p[i] = std::exp((double)logits[i] - (double)m);
            s += p[i];
        }
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            double t = 0.0;
            for (std::size_t k = 0; k < tgtIdx.size(); k++) {
                if (tgtIdx[k] == legalIdx[i]) { t = (double)tgtProb[k]; break; }
            }
            sumD += p[i] / s - t;
        }
        printf("    \u5408\u6cd5\u96c6\u4e0a \u03a3(p - t) = %.3e (\u5e94\u4e3a 0, \u4e24\u8fb9\u90fd\u5f52\u4e00)\n", sumD);
        CHECK(std::fabs(sumD) < 1e-5, "R2: 梯度在合法集上求和为 0 (p、t 都在合法集上归一)");
    }
}

int main(int /*argc*/, char * /*argv*/[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    /* 固定随机种子: 这个测试必须可复现 */
    Random::setSeed(20240613);

    std::printf("=== 梯度传播正确性检查 (SIMD 之后) ===\n");
    std::printf("SIMD 内核: %s\n", simdops::instructionSet());

    partA();
    partB();
    partC();
    partD();
    partE();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
