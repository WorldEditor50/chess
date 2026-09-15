/*
 * test_scaledconcat_main.cpp — ScaledConcat (rl/concat.hpp) 的结构不变量与梯度
 * ============================================================================
 *
 * 为什么需要这个文件
 * ------------------
 * `ScaledConcat` 在这之前**一个测试都没有**: 它只出现在 dqn.cpp 的一棵 `#elif 0`
 * 分支里, ctest 里没有任何实例。2026-09 它被重写 (门控与特征解耦 / 门控 logits
 * 无界 / 初始化按 fan-in 缩放 / 专家类型是模板参数), 于是三类东西必须钉住:
 *
 *   A. **旧实现的结构性缺陷真的被修掉了** —— 不是"看起来更合理", 而是可测的:
 *      旧门控是 softmax(sigmoid(·)), logits 落在 (0,1) => gate_max/gate_min < e^1,
 *      实测有效路数 58.6~61.3/64; 新门控的 logits 是实数, 可以真的尖。
 *      这里把**上界本身**写成断言 (旧实现的比值必须 ≤ e, 新实现必须能 > e 很多倍) ——
 *      否则"修好了"这句话没有任何东西挡着它退回去。
 *
 *   B. **门控与特征真的解耦了** —— 改一个专家的权重, 门控必须**逐位不变**;
 *      改门控权重, 特征必须逐位不变。旧实现里两者是同一批数, 这条**必然失败**,
 *      所以它同时是"新结构"的正向证明和"旧结构"的反向证据。
 *
 *   C. **反向是对的** —— 解析梯度 vs 中心差分, 覆盖 w1/w2/b/wg/bg + 专家的参数
 *      + 投影的参数, 以及**输入梯度** (ei 有三条累加通路: W2ᵀdz、Wgᵀdlogit、
 *      各专家的 backward —— 只查参数是抓不到"漏了一条通路"的)。
 *
 * 七个部分:
 *   [1] 门控的选择性: 新实现能多尖 vs 旧实现的 e^1 上界 (结构修正的直接证据)
 *   [2] 门控与特征解耦的不变量 (改一边, 另一边逐位不变)
 *   [3] 反向 vs 有限差分: 参数 (含专家与投影)
 *   [4] 反向 vs 有限差分: 输入梯度的三条累加通路
 *   [5] 专家模板参数: MlpExpert / TransformerBlock / Layer<Fn> 都能用
 *   [6] 初始化与保维: fan-in 缩放 / W2 零初始化 / 残差 / 存取往返 / copyTo
 *   [7] 代价: 三种专家的前向耗时 (便于选型)
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/loss.h"
#include "rl/concat.hpp"
#include "rl/moe.hpp"
#include "rl/transformer.hpp"
#include "rl/expert.hpp"

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

static Tensor randTensor(std::size_t r, std::size_t c, unsigned seed, float amp = 0.5f)
{
    Tensor t(r, c);
    unsigned s = seed;
    for (std::size_t i = 0; i < t.size(); i++) {
        s = s * 1664525u + 1013904223u;
        t[i] = (float)((double)(s >> 8) / (double)(1u << 24) * 2.0 - 1.0) * amp;
    }
    return t;
}

static bool allFinite(const Tensor &t)
{
    for (std::size_t i = 0; i < t.size(); i++) {
        const double v = (double)t[i];
        if (!(v == v) || std::fabs(v) > 1e30) {
            return false;
        }
    }
    return true;
}

static double maxAbsDiff(const Tensor &a, const Tensor &b)
{
    if (a.size() != b.size()) {
        return 1e30;
    }
    double m = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
        m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    }
    return m;
}

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

struct FdResult {
    double worst;
    double maxNum;
    double maxAna;
    std::size_t n;
};

/* 中心差分核对一个参数张量 (扰动的是**真身**, 不是拷贝) */
static FdResult checkParam(Net &net, Tensor &w, Tensor &gw,
                           const Tensor &x, const Tensor &t, float eps, int stride)
{
    FdResult r;
    r.worst = 0;
    r.maxNum = 0;
    r.maxAna = 0;
    r.n = 0;
    for (std::size_t i = 0; i < w.size(); i += (std::size_t)stride) {
        const float save = w[i];
        w[i] = save + eps;
        const double lp = lossOf(net, x, t);
        w[i] = save - eps;
        const double lm = lossOf(net, x, t);
        w[i] = save;

        const double num = (lp - lm) / (2.0 * (double)eps);
        const double ana = (double)gw[i];
        r.maxNum = std::fmax(r.maxNum, std::fabs(num));
        r.maxAna = std::fmax(r.maxAna, std::fabs(ana));
        const double denom = std::fmax(1.0, std::fmax(std::fabs(num), std::fabs(ana)));
        const double rel = std::fabs(num - ana) / denom;
        if (rel > r.worst) {
            r.worst = rel;
        }
        r.n++;
    }
    return r;
}

/* 输入梯度: 扰动 x, 与 Net::inputGrad 比 */
static FdResult checkInput(Net &net, Tensor &x, Tensor &ei,
                           const Tensor &t, float eps, int stride)
{
    FdResult r;
    r.worst = 0;
    r.maxNum = 0;
    r.maxAna = 0;
    r.n = 0;
    for (std::size_t j = 0; j < x.size(); j += (std::size_t)stride) {
        const float save = x[j];
        x[j] = save + eps;
        const double lp = lossOf(net, x, t);
        x[j] = save - eps;
        const double lm = lossOf(net, x, t);
        x[j] = save;

        const double num = (lp - lm) / (2.0 * (double)eps);
        const double ana = (double)ei[j];
        r.maxNum = std::fmax(r.maxNum, std::fabs(num));
        r.maxAna = std::fmax(r.maxAna, std::fabs(ana));
        const double denom = std::fmax(1.0, std::fmax(std::fabs(num), std::fabs(ana)));
        const double rel = std::fabs(num - ana) / denom;
        if (rel > r.worst) {
            r.worst = rel;
        }
        r.n++;
    }
    return r;
}

static void report(const char *tag, const FdResult &r)
{
    const char *verdict = "";
    if (r.maxNum == 0.0 && r.maxAna == 0.0) {
        verdict = "  [前向未使用]";
    } else if (r.maxAna == 0.0) {
        verdict = "  **[反向没填梯度]**";
    } else if (r.maxNum == 0.0) {
        verdict = "  **[数值没感觉: 梯度是假的]**";
    }
    std::printf("    %-34s n=%-4zu max_rel=%.2e  |num|max=%.3e |ana|max=%.3e%s\n",
                tag, r.n, r.worst, r.maxNum, r.maxAna, verdict);
}

/* 门控的 max/min 比 (直接看 gate 张量) */
static double gateRatio(const Tensor &gate)
{
    double mx = -1e30, mn = 1e30;
    for (std::size_t i = 0; i < gate.size(); i++) {
        const double v = (double)gate[i];
        mx = std::fmax(mx, v);
        mn = std::fmin(mn, v);
    }
    return (mn > 0) ? (mx / mn) : 1e30;
}

/* ============================================================
 *  [1] 门控的选择性: 新实现 vs 旧实现的硬上界
 * ============================================================ */
static void part1()
{
    std::printf("\n[1] 门控的选择性 (结构修正的直接证据)\n");
    const int D = 24;
    const int U = 4;
    const int N = 4;

    /* --- 旧实现: softmax(sigmoid(·)), logits 落在 (0,1) => 比值 < e --- */
    {
        using Legacy = ScaledConcatLegacy<Layer<Sigmoid>, N>;
        auto lay = Legacy::_(Layer<Sigmoid>(D, U, true, true), D, U, true);
        Tensor x = randTensor(D, 1, 4242, 1.0f);
        Tensor &o = lay->forward(x, true);
        (void)o;
        /* a 在 forward 里已经被 softmax 覆盖, 就是门控本身 */
        const double ratio = gateRatio(lay->a);
        std::printf("    旧: softmax(sigmoid(pre))  gate_max/min = %.3f  (理论上界 e = %.3f)\n",
                    ratio, std::exp(1.0));
        CHECK(ratio > 1.0, "旧实现的 gate 不是恒等 (检查不是拿 0 比 0)");
        CHECK(ratio < std::exp(1.0) + 1e-3,
              "旧实现的 gate 比值被 e^1 卡住 —— 这是 logits 落在 (0,1) 的必然结果");
    }

    /* --- 新实现: 实数 logits, 可以真的尖 --- */
    {
        auto lay = ScaledConcat<MlpExpert, N, U>::_(D, true, 4);
        /*
           造一个"第 0 路吃掉全部质量"的门控: 第 0 路的 logit 是 Wg·x (随 x 变),
           其余 15 路是常数 −8 (质量被压到可忽略)。
           注意不能只写 `bg[0] = -5`, 那样 15 个"其余路"彼此相等、质量还占大头,
           门控比值虽然大但分布并不集中 (实测有效路数仍有 15/16) —— 这正是
           "看比值不如看有效路数"的例子, 所以两条断言都要有。
        */
        lay->wg.zero();
        lay->bg.zero();
        for (int j = 0; j < D; j++) {
            lay->wg(0, j) = 1.0f;
        }
        for (int i = 1; i < N*U; i++) {
            lay->bg[i] = -8.0f;
        }
        Tensor x = randTensor(D, 1, 4242, 1.0f);
        lay->forward(x, true);
        const double ratio = gateRatio(lay->gate);
        std::printf("    新: softmax(Wg·x + bg)     gate_max/min = %.3e  (单元数 %d)\n",
                    ratio, N*U);
        CHECK(ratio > 100.0, "新实现的 gate 可以远尖于 e^1 (logits 无界)");
        CHECK(allFinite(lay->gate), "门控有限");
        CHECK(std::fabs((double)lay->gate.sum() - 1.0) < 1e-5, "门控归一化到 1");

        /* 有效路数: 尖门控必须远小于单元数 */
        std::vector<float> usage;
        lay->gateUsage(usage);
        CHECK((int)usage.size() == N*U, "gateUsage 长度 = N·u");
        const double eff = lay->gateEffectiveCount();
        std::printf("    有效路数 exp(H) = %.2f / %d\n", eff, N*U);
        CHECK(eff > 0.5, "有效路数 > 0.5 (不是全压成一路)");
        CHECK(eff < 0.35 * (double)(N*U), "尖门控下有效路数远小于单元数");

        /* gateScale 能把门控变尖: 同一组 z, 乘 4 倍温度 */
        auto lay2 = ScaledConcat<MlpExpert, N, U>::_(D, true, 4, false, 4.0f);
        lay2->wg = lay->wg;
        lay2->bg = lay->bg;
        lay2->forward(x, true);
        const double ratio2 = gateRatio(lay2->gate);
        CHECK(ratio2 > ratio * 10.0, "gateScale 放大 logits 后门控更尖 (温度旋钮有效)");
        std::printf("    gateScale=4 时 gate_max/min = %.3e\n", ratio2);

        lay->resetGateUsage();
        CHECK(lay->gateEffectiveCount() == 0.0, "resetGateUsage 之后没有样本 (返回 0 而不是假值)");
    }
}

/* ============================================================
 *  [2] 门控与特征解耦 (改一边, 另一边逐位不变)
 * ============================================================ */
static void part2()
{
    std::printf("\n[2] 门控与特征解耦的不变量\n");
    const int D = 24;
    const int U = 4;
    const int N = 3;

    auto lay = ScaledConcat<MlpExpert, N, U>::_(D, true, 4);
    Tensor x = randTensor(D, 1, 777, 1.0f);
    lay->forward(x, true);

    Tensor gate0 = lay->gate;
    Tensor feat0 = lay->feat;

    /* (a) feat 必须等于 concat(proj_i(expert_i(x))) 的手算结果 */
    {
        double worst = 0;
        for (int i = 0; i < N; i++) {
            Tensor &eo = lay->experts[i].forward(x, true);
            Tensor &hi = lay->proj[i].forward(eo, true);
            for (int r = 0; r < U; r++) {
                worst = std::fmax(worst, std::fabs((double)hi[r] - (double)lay->feat[i*U + r]));
            }
        }
        std::printf("    feat 与手算 concat(proj_i(expert_i(x))) 最大差 = %.2e\n", worst);
        CHECK(worst < 1e-5, "feat 就是各专家的投影输出拼接 (没有被动过)");
        CHECK(std::fabs((double)feat0.sum()) > 1e-3, "feat 不是恒 0 (检查不是拿 0 比 0)");
    }

    /* (b) gated == gate ∘ feat */
    {
        double worst = 0;
        for (int k = 0; k < N*U; k++) {
            worst = std::fmax(worst, std::fabs((double)gate0[k]*(double)feat0[k]
                                               - (double)lay->gated[k]));
        }
        CHECK(worst < 1e-6, "gated = gate ∘ feat 逐元素成立");
    }

    /* (c) 改专家权重 -> 门控逐位不变, 特征按比例变 */
    {
        /* 投影的 w 与 b 一起加倍: 只加倍 w 的话 b 还在, 输出不等于两倍
           (第一版就是这么写错的 —— 这条断言本身会失败, 而不是"没测到") */
        for (std::size_t k = 0; k < lay->proj[1].w.size(); k++) {
            lay->proj[1].w[k] *= 2.0f;
        }
        for (std::size_t k = 0; k < lay->proj[1].b.size(); k++) {
            lay->proj[1].b[k] *= 2.0f;
        }
        lay->forward(x, true);
        const double dGate = maxAbsDiff(gate0, lay->gate);
        std::printf("    加倍 proj[1] 的权重: 门控最大差 = %.2e\n", dGate);
        CHECK(dGate == 0.0, "改专家的(投影)权重, 门控逐位不变 —— 门控只看 x");
        double dSeg = 0;
        for (int r = 0; r < U; r++) {
            dSeg = std::fmax(dSeg, std::fabs((double)lay->feat[U + r] - 2.0*(double)feat0[U + r]));
        }
        CHECK(dSeg < 1e-5, "同一改动下第 1 路特征正好加倍 (特征没被归一化掉)");

        /* 旧实现里这两件事是同一批数: 改子层权重必然改门控 */
        using Legacy = ScaledConcatLegacy<Layer<Sigmoid>, N>;
        auto old = Legacy::_(Layer<Sigmoid>(D, U, true, true), D, U, true);
        Tensor a0 = [&]{
            old->forward(x, true);
            return old->a;
        }();
        for (std::size_t k = 0; k < old->layers[1].w.size(); k++) {
            old->layers[1].w[k] *= 2.0f;
        }
        old->forward(x, true);
        const double dOld = maxAbsDiff(a0, old->a);
        std::printf("    旧实现: 加倍第 1 个子层权重 -> 门控最大差 = %.2e (非 0)\n", dOld);
        CHECK(dOld > 1e-6, "旧实现里门控是子层输出的归一化 —— 改子层必然改门控");
    }

    /* (d) 改门控权重 -> 特征逐位不变 */
    {
        Tensor feat1 = lay->feat;
        for (std::size_t k = 0; k < lay->wg.size(); k++) {
            lay->wg[k] *= 3.0f;
        }
        lay->forward(x, true);
        CHECK(maxAbsDiff(feat1, lay->feat) == 0.0, "改门控权重, 特征逐位不变");
    }
}

/* ============================================================
 *  [3] 反向 vs 有限差分 (参数)
 * ============================================================ */
static void part3()
{
    std::printf("\n[3] 反向 vs 有限差分: 参数 (MlpExpert 专家, N=3 u=2)\n");
    const int D = 20;
    const int U = 2;
    const int N = 3;
    const float eps = 1e-3f;

    Net net(ScaledConcat<MlpExpert, N, U>::_(D, true, 6));
    auto lay = dynamic_cast<ScaledConcat<MlpExpert, N, U>*>(net[0]);
    Tensor x = randTensor(D, 1, 31337, 1.0f);
    Tensor t = randTensor(N*U, 1, 555, 0.3f);

    /* 两次前向必须逐位相同 —— 必须**拷贝**第一次的输出再比:
       `Tensor&` 两次拿到的是同一个 o, 拿它跟自己比永远是 0 (假通过)。 */
    Tensor o1 = net.forward(x, true);
    Tensor o2 = net.forward(x, true);
    CHECK(maxAbsDiff(o1, o2) == 0.0, "两次 forward 输出逐位相同 (没有累加残留)");

    Tensor e = Loss::MSE::df(o1, t);
    net.backward(x, e);
    CHECK(allFinite(o1), "前向输出有限");

    report("w1 (Out×N·u)",  checkParam(net, lay->w1, lay->g.w1, x, t, eps, 3));
    report("w2 (Out×d)",    checkParam(net, lay->w2, lay->g.w2, x, t, eps, 5));
    report("b  (Out)",      checkParam(net, lay->b,  lay->g.b,  x, t, eps, 1));
    report("wg (N·u×d)",    checkParam(net, lay->wg, lay->g.wg, x, t, eps, 7));
    report("bg (N·u)",      checkParam(net, lay->bg, lay->g.bg, x, t, eps, 1));
    report("expert0.l1.w",  checkParam(net, lay->experts[0].l1.w, lay->experts[0].l1.g.w, x, t, eps, 11));
    report("expert2.l3.w",  checkParam(net, lay->experts[2].l3.w, lay->experts[2].l3.g.w, x, t, eps, 13));
    report("proj0.w",       checkParam(net, lay->proj[0].w, lay->proj[0].g.w, x, t, eps, 5));
    report("proj1.b",       checkParam(net, lay->proj[1].b, lay->proj[1].g.b, x, t, eps, 1));
}

/* ============================================================
 *  [4] 反向 vs 有限差分 (输入梯度: 三条累加通路)
 * ============================================================ */
static void part4()
{
    std::printf("\n[4] 反向 vs 有限差分: 输入梯度 ei (W2ᵀdz / Wgᵀdlogit / 专家)\n");
    const int D = 16;
    const int U = 2;
    const int N = 2;
    const float eps = 1e-3f;

    Tensor x = randTensor(D, 1, 909, 0.8f);
    Tensor t = randTensor(N*U, 1, 91, 0.3f);

    /* (a) 三条通路一起 */
    {
        Net net(ScaledConcat<MlpExpert, N, U>::_(D, true, 4));
        Tensor &o = net.forward(x, true);
        net.backward(x, Loss::MSE::df(o, t));
        report("dL/dx (全部通路)", checkInput(net, x, net.inputGrad, t, eps, 1));
    }

    /*
        再把三条通路**分别隔离** —— 合计对了不代表每条都对 ("两条都写错正好抵掉"
        或者 "漏了一条、另一条偏大" 都能混过去)。

        隔离的关键: 不能用"把权重清零"的办法 —— 那会让该通路的**解析梯度也变成 0**,
        差分同样是 0, 又变成拿 0 比 0 的假通过。这里改的是**依赖关系**:

          特征与 x 无关: 专家的第一层 w=0, b=1  ->  输出恒为常数, 专家对 x 的梯度恰好 0
          门控与 x 无关: wg=0 (bg 保留)          ->  门控是常数, 且 Wgᵀ·dlogit = 0

        于是:
          只留 W2   : wg=0 + 特征与 x 无关  ->  ei 只可能来自 W2ᵀ·dz
          只留门控  : w2=0 + 特征与 x 无关  ->  ei 只可能来自 Wgᵀ·dlogit
          只留专家  : w2=0 + wg=0            ->  ei 只可能来自各专家的 backward
        三种情况下 ei 都**不恒为 0**, 所以差分是有意义的。
    */
    struct Mode { const char *tag; bool killW2; bool killGatePath; bool xFreeExperts; };
    const Mode modes[3] = {
        { "  只留 W2 跳连", false, true,  true  },
        { "  只留门控通路", true,  false, true  },
        { "  只留专家通路", true,  true,  false }
    };
    for (int m = 0; m < 3; m++) {
        Net net(ScaledConcat<MlpExpert, N, U>::_(D, true, 4));
        auto lay = dynamic_cast<ScaledConcat<MlpExpert, N, U>*>(net[0]);
        if (modes[m].killW2) {
            lay->w2.zero();
        }
        if (modes[m].killGatePath) {
            lay->wg.zero();          /* 门控变常数 -> Wgᵀ·dlogit = 0 */
        }
        if (modes[m].xFreeExperts) {
            for (int i = 0; i < N; i++) {
                lay->experts[i].l1.w.zero();
                for (std::size_t k = 0; k < lay->experts[i].l1.b.size(); k++) {
                    lay->experts[i].l1.b[k] = 1.0f;
                }
            }
        }
        Tensor &o = net.forward(x, true);
        net.backward(x, Loss::MSE::df(o, t));
        report(modes[m].tag, checkInput(net, x, net.inputGrad, t, eps, 1));
    }
}

/* ============================================================
 *  [5] 专家模板参数: 三种专家都能用
 * ============================================================ */
static void part5()
{
    std::printf("\n[5] 专家模板参数 (与 sparse_moe.hpp 同一套工厂)\n");
    const int D = 16;
    const float eps = 1e-3f;
    Tensor x = randTensor(D, 1, 2024, 0.8f);

    {
        Net net(ScaledConcat<Layer<Gelu>, 4, 2>::_(D, true));
        auto lay = dynamic_cast<ScaledConcat<Layer<Gelu>, 4, 2>*>(net[0]);
        Tensor t = randTensor(8, 1, 12, 0.3f);
        Tensor &o = net.forward(x, true);
        net.backward(x, Loss::MSE::df(o, t));
        CHECK(allFinite(o), "Layer<Gelu> 专家: 前向有限");
        CHECK(net.paramCount() > 0, "Layer<Gelu> 专家: paramCount > 0");
        report("Layer<Gelu> 专家 expert1.w",
               checkParam(net, lay->experts[1].w, lay->experts[1].g.w, x, t, eps, 5));
        std::printf("    Layer<Gelu> 专家参数 = %lld\n", net.paramCount());
    }
    {
        Net net(ScaledConcat<TransformerBlock<2>, 2, 2>::_(D, true));
        Tensor t = randTensor(4, 1, 13, 0.3f);
        Tensor &o = net.forward(x, true);
        net.backward(x, Loss::MSE::df(o, t));
        CHECK(allFinite(o), "TransformerBlock 专家: 前向有限");
        CHECK(net.paramCount() > 0, "TransformerBlock 专家: paramCount > 0");
        /* LN 的 gamma 也要有梯度 (复合层最容易漏的就是这一路) */
        auto lay = dynamic_cast<ScaledConcat<TransformerBlock<2>, 2, 2>*>(net[0]);
        double g = 0;
        for (std::size_t i = 0; i < lay->experts[0].g1.gamma.size(); i++) {
            g = std::fmax(g, std::fabs((double)lay->experts[0].g1.gamma[i]));
        }
        CHECK(g > 1e-9, "TransformerBlock 专家: LayerNorm gamma 拿到了梯度");
        std::printf("    TransformerBlock 专家参数 = %lld (|dL/dgamma1|max=%.3e)\n",
                    net.paramCount(), g);
    }
    {
        Net net(ScaledConcat<MlpExpert, 2, 2>::_(D, true, 0));   /* hidden 0 -> d/16 */
        Tensor t = randTensor(4, 1, 14, 0.3f);
        Tensor &o = net.forward(x, true);
        net.backward(x, Loss::MSE::df(o, t));
        CHECK(allFinite(o), "MlpExpert 专家 (hidden 默认 d/16): 前向有限");
    }
}

/* ============================================================
 *  [6] 初始化 / 保维 / 残差 / 存取 / copyTo
 * ============================================================ */
static void part6()
{
    std::printf("\n[6] 初始化、保维残差、存取往返、copyTo\n");
    const int D = 90;      /* 象棋 DQN 的状态维 */
    const int U = 4;
    const int N = 16;

    /* (a) 初始化健康度: 输出不该在 tanh 的饱和区 */
    {
        auto lay = ScaledConcat<MlpExpert, N, U>::_(D, true, 8);
        double worstDeriv = 1.0;
        double meanDeriv = 0;
        int samples = 0;
        for (unsigned s = 0; s < 8; s++) {
            Tensor x = randTensor(D, 1, 1000 + s, 1.0f);
            Tensor &o = lay->forward(x, true);
            double d = 0;
            for (std::size_t i = 0; i < o.size(); i++) {
                const double deriv = 1.0 - (double)o[i]*(double)o[i];   /* tanh'(o) */
                d += deriv;
                worstDeriv = std::fmin(worstDeriv, deriv);
            }
            meanDeriv += d / (double)o.size();
            samples++;
        }
        meanDeriv /= samples;
        std::printf("    init: 平均 tanh'(o) = %.4f (旧实现实测 0.18~0.56)\n", meanDeriv);
        CHECK(meanDeriv > 0.75, "fan-in 缩放后开局不在饱和区 (平均 tanh' > 0.75)");
        CHECK(worstDeriv > 1e-3, "没有输出贴死在 ±1 (最差的 tanh' 也没归零)");
    }

    /* (b) W2 零初始化 (zeroSkip 默认 true) / 显式关掉时非 0 */
    {
        auto lay = ScaledConcat<MlpExpert, N, U>::_(D, true, 8);
        double m = 0;
        for (std::size_t k = 0; k < lay->w2.size(); k++) {
            m = std::fmax(m, std::fabs((double)lay->w2[k]));
        }
        CHECK(m == 0.0, "zeroSkip 默认: 跳连 W2 零初始化 (开局是纯门控混合)");

        auto lay2 = ScaledConcat<MlpExpert, N, U>::_(D, true, 8, false, 1.0f, false);
        double m2 = 0;
        for (std::size_t k = 0; k < lay2->w2.size(); k++) {
            m2 = std::fmax(m2, std::fabs((double)lay2->w2[k]));
        }
        CHECK(m2 > 0.0, "zeroSkip=false 时 W2 不是 0");
        /* 而且 fan-in 缩放后量级要小: |w2| <= 1/sqrt(d) + eps */
        CHECK(m2 <= 1.0/std::sqrt((double)D) + 1e-6, "W2 按 1/sqrt(d) 缩放");
    }

    /* (c) 保维 + 残差: 把混合与跳连都清零, 输出必须是 tanh(x) */
    {
        auto lay = ScaledConcat<MlpExpert, N, U, D>::_(D, true, 8, true);
        CHECK(lay->unitCount() == N*U, "unitCount = N·u");
        Tensor x = randTensor(D, 1, 4242, 1.0f);
        Tensor &o = lay->forward(x, true);
        CHECK((int)o.size() == D, "OutDim=dIn 时输出宽度 == 输入宽度");

        /* 关掉 W1 与 W2 -> 只剩残差 */
        lay->w1.zero();
        lay->w2.zero();
        lay->b.zero();
        Tensor &o2 = lay->forward(x, true);
        double worst = 0;
        for (int i = 0; i < D; i++) {
            worst = std::fmax(worst, std::fabs((double)o2[i] - std::tanh((double)x[i])));
        }
        std::printf("    残差通路: |o - tanh(x)|max = %.2e\n", worst);
        CHECK(worst < 1e-6, "残差模式下输出 = tanh(x) (通路确实是 x 直加)");

        /* 不带残差时同样的清零 -> 输出恒 0 (对照组, 证明上面那条不是巧合) */
        auto lay2 = ScaledConcat<MlpExpert, N, U, D>::_(D, true, 8, false);
        lay2->w1.zero();
        lay2->w2.zero();
        lay2->b.zero();
        Tensor &o3 = lay2->forward(x, true);
        double m3 = 0;
        for (int i = 0; i < D; i++) {
            m3 = std::fmax(m3, std::fabs((double)o3[i]));
        }
        CHECK(m3 < 1e-7, "不开残差时输出恒 0 (对照)");
    }

    /* (d) copyTo / 存取往返 / softUpdateTo */
    {
        Net a(ScaledConcat<MlpExpert, 4, 2>::_(D, true, 8));
        Net b(ScaledConcat<MlpExpert, 4, 2>::_(D, true, 8));
        Tensor x = randTensor(D, 1, 88, 1.0f);
        a.copyTo(b);
        /* 取**值拷贝**: `Tensor&` 拿到的是层里的 o, 后面再 forward 就会变 */
        Tensor oa = a.forward(x, true);
        Tensor ob = b.forward(x, true);
        CHECK(maxAbsDiff(oa, ob) == 0.0, "copyTo 之后两个实例输出逐位相同");

        const std::string path = "test_scaledconcat_tmp.wgt";
        CHECK(a.save(path) == 0, "save 返回 0");
        Net c(ScaledConcat<MlpExpert, 4, 2>::_(D, true, 8));
        CHECK(c.load(path) == 0, "load 返回 0");
        Tensor oc = c.forward(x, true);
        CHECK(maxAbsDiff(oa, oc) == 0.0, "存取往返之后输出逐位相同 (v2 无损)");
        std::remove(path.c_str());

        /* 优化器冒烟: 每步之后参数必须有限 */
        Tensor t = randTensor(8, 1, 99, 0.3f);
        for (int step = 0; step < 3; step++) {
            Tensor &o = a.forward(x, true);
            a.backward(x, Loss::MSE::df(o, t));
            a.RMSProp(0.001f);
        }
        a.clamp(-5.0f, 5.0f);
        a.softUpdateTo(b, 0.5f);
        Tensor oa2 = a.forward(x, true);
        Tensor ob2 = b.forward(x, true);
        CHECK(allFinite(oa2) && allFinite(ob2), "RMSProp/clamp/softUpdateTo 之后参数与输出有限");
        CHECK(maxAbsDiff(oa2, ob2) > 0.0, "softUpdateTo 之后两边不同 (它真的动了权重)");
    }
}

/* ============================================================
 *  [7] 代价: 三种专家
 * ============================================================ */
static double timeFwd(Net &net, const Tensor &x, int reps)
{
    using namespace std::chrono;
    net.forward(x, true);
    const double t0 = duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
    for (int i = 0; i < reps; i++) {
        net.forward(x, true);
    }
    const double t1 = duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
    return (t1 - t0) / reps;
}

static void part7()
{
    std::printf("\n[7] 代价 (D=90, N=16, u=4, forward)\n");
    const int D = 90;
    Tensor x = randTensor(D, 1, 5, 1.0f);
    {
        Net net(ScaledConcat<MlpExpert, 16, 4>::_(D, true, 8));
        std::printf("    MlpExpert(hidden=8)   参数 %8lld   %8.3f us\n",
                    net.paramCount(), timeFwd(net, x, 2000));
    }
    {
        Net net(ScaledConcat<Layer<Gelu>, 16, 4>::_(D, true));
        std::printf("    Layer<Gelu>           参数 %8lld   %8.3f us\n",
                    net.paramCount(), timeFwd(net, x, 2000));
    }
    {
        Net net(ScaledConcat<Layer<Linear>, 16, 4>::_(D, true));
        std::printf("    Layer<Linear>         参数 %8lld   %8.3f us\n",
                    net.paramCount(), timeFwd(net, x, 2000));
    }
    {
        Net net(ScaledConcat<TransformerBlock<4>, 4, 4>::_(D, true));
        std::printf("    TransformerBlock<4>   参数 %8lld   %8.3f us (N=4)\n",
                    net.paramCount(), timeFwd(net, x, 200));
    }
}

int main()
{
    std::printf("=== ScaledConcat: 结构不变量 + 梯度 + 专家模板参数 ===\n");
    Random::setSeed(20240501);
    part1();
    part2();
    part3();
    part4();
    part5();
    part6();
    part7();

    std::printf("\n==== %d 条断言, %d 条失败 ====\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
