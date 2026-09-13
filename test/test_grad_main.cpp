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
    std::printf("    ikjk %-24s (r=%2zu c=%2zu k=%2zu): got/expect = %.6f  %s\n",
                tag, r, c, k, got/expect,
                (rel < 1e-5) ? "累加 -> 与标量语义一致" : "**覆盖 -> 只保留最后一次**");
}

static void partC()
{
    std::printf("\n=== C. MM 内核语义探针 (累加 vs 覆盖) ===\n");

    /* C1: 库里真实的列向量形状 —— kdim = 1 */
    probeIkjk("k=1 (列向量, 真实用法)", 90, 90, 1);
    /* C2: kdim >= 8 的形状 —— 会命中 SIMD 内核 */
    probeIkjk("k=32 (命中 SIMD)", 90, 90, 32);

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
 *  gemv_ikkj (o = w·x) 有专门的 SIMD 内核; kikj (ei = wᵀ·e) **没有** ——
 *  它的 mm 内核要求每一维都 >= 一个向量宽度, 而 ei 是一列 (zCol = 1), 判据不成立,
 *  于是掉回标量循环。也就是说 SIMD 只加速了前向那一半。
 *  这里把两者量出来 (纯信息, 不做断言)。
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
    /* 前向: o = w·x  -> gemv_ikkj */
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        o.zero();
        Tensor::MM::ikkj(o, w, x);
    }
    auto t1 = std::chrono::steady_clock::now();
    /* 反向: ei = wᵀ·e -> kikj (无 GEMV 内核) */
    for (int i = 0; i < iters; i++) {
        ei.zero();
        Tensor::MM::kikj(ei, w, e);
    }
    auto t2 = std::chrono::steady_clock::now();

    const double fwdNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                         / (double)iters;
    const double bwdNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
                         / (double)iters;
    const double macs = (double)rows*(double)kdim;
    std::printf("    ikkj (o = w·x,  %zux%zu): %8.0f ns/次  = %.3f ns/MAC\n",
                rows, kdim, fwdNs, fwdNs/macs);
    std::printf("    kikj (ei = wᵀ·e, %zux%zu): %8.0f ns/次  = %.3f ns/MAC\n",
                rows, kdim, bwdNs, bwdNs/macs);
    std::printf("    -> 反向是前向的 %.1fx  (SIMD 只覆盖了前向那一半)\n", bwdNs/fwdNs);
}

int main(int /*argc*/, char * /*argv*/[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    /* 固定随机种子: 这个测试必须可复现 */
    Random::engine.seed(20240613);

    std::printf("=== 梯度传播正确性检查 (SIMD 之后) ===\n");
    std::printf("SIMD 内核: %s\n", simdops::instructionSet());

    partA();
    partB();
    partC();
    partD();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
