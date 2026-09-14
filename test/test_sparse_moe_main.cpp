/*
 * test_sparse_moe_main.cpp - 稀疏路由 MoE (rl/sparse_moe.hpp) 的正确性与代价
 * ============================================================================
 *
 * 为什么单独一个测试文件
 * ----------------------
 * `rl/moe.hpp` (上游 snakeAI 的) 是**稠密** MoE: forward 把全部 NumExperts 个专家
 * 都算一遍 (`if (gi > 1e-8f)` 只跳过累加)。在 d_model=1260 的象棋状态上, 16 个
 * TransformerBlock 专家 = 152 ms/前向, 直接不可用。`rl/sparse_moe.hpp` 是重写的
 * 稀疏版本: 只算门控选中的 top-k 个专家, 于是算力 ~ k×专家 而与 E 无关。
 *
 * 但"只算一部分"这件事会同时破坏两个原本显然的性质, 所以必须逐条验证:
 *
 *   1. **前向确实跳过了未选中的专家** —— 否则花了稀疏的钱, 得到稠密的结果。
 *   2. **反向的梯度仍然正确** —— 门控的雅可比、被选中专家的参数梯度、以及
 *      未被选中专家"梯度必须恰好为 0"。稀疏 MoE 的经典失败模式 (专家坍缩)
 *      就是反向把未选中专家一路压低造成的, 所以这里连符号都要查。
 *
 * 检查手段与 test_grad_main.cpp 一致: 解析梯度 (backward 填的 g.w) vs 中心差分
 *   (L(w+eps) - L(w-eps)) / (2*eps), 其中 L = Σ MSE 分量。
 *
 * 九个部分:
 *   [1] 门控与 top-k 选择本身 (手算对照)
 *   [2] 稀疏不变量: 未选中的专家改权重, 输出必须**逐位不变**
 *   [3] 与上游稠密 MOE 的等价性 (TopK==E 时前向/反向都要一致)
 *   [4] 反向 vs 有限差分: 门控 wg/bg + 被选中专家 + 未选中专家
 *   [5] 稠密对照 (TopK = E) 的有限差分
 *   [6] 负载均衡辅助损失的梯度 vs 有限差分
 *   [7] 辅助损失真的把坍缩的路由拉回来 (符号 + 多步)
 *   [8] 使用计数 (坍缩诊断) 与权重存取往返
 *   [9] 代价: 稀疏 vs 稠密, MLP 专家 vs TransformerBlock 专家
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
#include "rl/moe.hpp"
#include "rl/transformer.hpp"
#include "rl/sparse_moe.hpp"

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

/*
 * 中心差分核对一个参数张量。
 *
 * 两个计数分开看 (这是判断"反向漏填"还是"前向本来就没用到"的唯一办法):
 *   |num|max == 0 且 |ana|max == 0  -> 前向没用这个参数 (例如未被选中的专家), 一致
 *   |num|max != 0 而 |ana|max == 0  -> **反向没填梯度**
 * 返回 (最差相对误差, max|num|, max|ana|)。
 */
struct FdResult {
    double worst;
    double maxNum;
    double maxAna;
    std::size_t n;
};

static FdResult checkParam(Net &net, Tensor &w, Tensor &gw,
                           const Tensor &x, const Tensor &t, float eps, int stride)
{
    FdResult r;
    r.worst = 0;
    r.maxNum = 0;
    r.maxAna = 0;
    r.n = 0;
    const std::size_t n = w.size();
    for (std::size_t i = 0; i < n; i += (std::size_t)stride) {
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

/*
 * 只查一个张量的某一行 (按 (row, col) 索引扰动**真身**, 不拷贝)。
 * 必须直接改层里的那个张量: 如果传进来的是一份拷贝, 差分永远是 0, 测试会假装通过。
 */
static FdResult checkRow(Net &net, Tensor &w, Tensor &g, int row,
                         const Tensor &x, const Tensor &t, float eps, int stride)
{
    FdResult r;
    r.worst = 0;
    r.maxNum = 0;
    r.maxAna = 0;
    r.n = 0;
    const int cols = (int)w.shape[1];
    for (int j = 0; j < cols; j += stride) {
        const float save = w(row, j);
        w(row, j) = save + eps;
        const double lp = lossOf(net, x, t);
        w(row, j) = save - eps;
        const double lm = lossOf(net, x, t);
        w(row, j) = save;

        const double num = (lp - lm) / (2.0 * (double)eps);
        const double ana = (double)g(row, j);
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
        verdict = "  [前向未使用/未被选中]";
    } else if (r.maxAna == 0.0) {
        verdict = "  **[反向没填梯度]**";
    } else if (r.maxNum == 0.0) {
        verdict = "  **[数值没感觉: 梯度是假的]**";
    }
    std::printf("    %-30s n=%-5zu max_rel=%.2e  |num|max=%.3e |ana|max=%.3e%s\n",
                tag, r.n, r.worst, r.maxNum, r.maxAna, verdict);
}

/* 把一个 MlpExpert 的权重填成"条件良好"的随机值 (便于做有限差分) */
static void fillExpert(MlpExpert &ex, unsigned seed)
{
    ex.l1.w = randTensor(ex.l1.w.shape[0], ex.l1.w.shape[1], seed + 1);
    ex.l1.b = randTensor(ex.l1.b.shape[0], 1, seed + 2);
    ex.l2.w = randTensor(ex.l2.w.shape[0], ex.l2.w.shape[1], seed + 3);
    ex.l2.b = randTensor(ex.l2.b.shape[0], 1, seed + 4);
    ex.l3.w = randTensor(ex.l3.w.shape[0], ex.l3.w.shape[1], seed + 5);
    ex.l3.b = randTensor(ex.l3.b.shape[0], 1, seed + 6);
}

/* ============================================================
 *  [1] 门控与 top-k 选择
 * ============================================================ */
static void part1()
{
    std::printf("\n[1] 门控与 top-k 选择 (手算对照)\n");
    const int D = 24;
    Net net(std::make_shared<SparseMoE<MlpExpert, 4, 2> >(D, true, 6));
    SparseMoE<MlpExpert, 4, 2> *lay = dynamic_cast<SparseMoE<MlpExpert, 4, 2>*>(net[0]);
    Tensor x = randTensor(D, 1, 777, 1.0f);

    net.forward(x, true);

    /* gate = softmax(wg·x + bg), 手算一遍 */
    double z[4];
    for (int i = 0; i < 4; i++) {
        double s = (double)lay->bg[i];
        for (int j = 0; j < D; j++) {
            s += (double)lay->wg(i, j) * (double)x[j];
        }
        z[i] = s;
    }
    const double zmax = std::fmax(std::fmax(z[0], z[1]), std::fmax(z[2], z[3]));
    double zs = 0;
    for (int i = 0; i < 4; i++) {
        zs += std::exp(z[i] - zmax);
    }
    double worst = 0;
    for (int i = 0; i < 4; i++) {
        const double ref = std::exp(z[i] - zmax) / zs;
        worst = std::fmax(worst, std::fabs(ref - (double)lay->gate[i]));
    }
    std::printf("    gate = ");
    for (int i = 0; i < 4; i++) {
        std::printf("%.4f ", (double)lay->gate[i]);
    }
    std::printf(" (Σ=%.5f, 与手算最大差 %.2e)\n", (double)lay->gate.sum(), worst);
    CHECK(worst < 1e-5, "gate 与手算 softmax(wg·x+bg) 一致");
    CHECK(std::fabs((double)lay->gate.sum() - 1.0) < 1e-5, "gate 归一化到 1");

    /* top-2 必须正好是概率最大的两个 (按降序) */
    CHECK(lay->nSelected == 2, "topK=2 时选中 2 个专家");
    int rank[2];
    for (int k = 0; k < 2; k++) {
        int best = -1;
        for (int i = 0; i < 4; i++) {
            bool used = false;
            for (int t = 0; t < k; t++) {
                if (rank[t] == i) {
                    used = true;
                }
            }
            if (!used && (best < 0 || lay->gate[i] > lay->gate[best])) {
                best = i;
            }
        }
        rank[k] = best;
    }
    std::printf("    选中 = %d,%d; 概率最大两个 = %d,%d\n",
                lay->selected[0], lay->selected[1], rank[0], rank[1]);
    CHECK(lay->selected[0] == rank[0] && lay->selected[1] == rank[1],
          "选中的是被门控概率最大的两个专家");
    CHECK(lay->gate[rank[0]] >= lay->gate[rank[1]], "选中的两个专家之间也是降序");
    CHECK(lay->topK() == 2 && lay->expertCount() == 4, "topK()/expertCount() 报告正确");

    /* TopK == E 时应该退化成稠密 (全部选中) */
    Net dense(std::make_shared<SparseMoE<MlpExpert, 3, 3> >(D, true, 6));
    SparseMoE<MlpExpert, 3, 3> *dl = dynamic_cast<SparseMoE<MlpExpert, 3, 3>*>(dense[0]);
    dense.forward(x, true);
    CHECK(dl->nSelected == 3, "TopK==E 时 nSelected==E (稠密对照模式)");
    CHECK(dl->topK() == 3, "TopK==E 时 topK() 被夹到 E");
}

/* ============================================================
 *  [2] 稀疏不变量: 未被选中的专家完全不参与计算
 * ============================================================ */
static void part2()
{
    std::printf("\n[2] 稀疏不变量: 改未选中专家的权重, 输出必须逐位不变\n");
    const int D = 32;
    Net net(std::make_shared<SparseMoE<MlpExpert, 4, 1> >(D, true, 8));
    SparseMoE<MlpExpert, 4, 1> *lay = dynamic_cast<SparseMoE<MlpExpert, 4, 1>*>(net[0]);
    Tensor x = randTensor(D, 1, 4242, 1.0f);

    Tensor &o0 = net.forward(x, true);
    Tensor ref = o0;   /* 深拷贝 */

    const int sel = lay->selected[0];
    int unsel = -1;
    for (int i = 0; i < 4; i++) {
        if (i != sel) {
            unsel = i;
            break;
        }
    }
    CHECK(unsel >= 0, "top-1 时能找到未被选中的专家");

    /* 大幅改动未选中专家的权重 (0.5 的量级, 不是 1e-6 的抖动) */
    lay->experts[unsel].l1.w[0] += 0.5f;
    lay->experts[unsel].l3.b[0] += 0.5f;
    Tensor &o1 = net.forward(x, true);
    double dU = 0;
    for (int j = 0; j < D; j++) {
        dU = std::fmax(dU, std::fabs((double)o0[j] - (double)ref[j]));
    }
    std::printf("    选中专家=%d, 被改的是专家%d: 输出最大变化 = %.3e\n", sel, unsel, dU);
    CHECK(dU == 0.0, "未选中专家的权重对输出没有任何影响 (逐位相同)");

    /* 同样的改动放到被选中的专家上, 必须改变输出 */
    lay->experts[sel].l1.w[0] += 0.5f;
    Tensor &o2 = net.forward(x, true);
    double dS = 0;
    for (int j = 0; j < D; j++) {
        dS = std::fmax(dS, std::fabs((double)o2[j] - (double)ref[j]));
    }
    std::printf("    改成选中专家%d          : 输出最大变化 = %.3e\n", sel, dS);
    CHECK(dS > 1e-6, "选中专家的权重确实影响输出");
    CHECK(allFinite(o2), "输出仍然有限");

    /* 选中集合在改动前后不变 (验证上面这件事测的是稀疏性, 不是路由漂移) */
    CHECK(lay->selected[0] == sel, "改动专家权重不改变路由 (路由只看门控)");
}

/* ============================================================
 *  [3] 与上游稠密 MOE 的等价性 (TopK == E)
 * ============================================================ */
static void part3()
{
    std::printf("\n[3] 与上游稠密 MOE<E,H> 的等价性 (TopK==E 时必须一致)\n");
    const int D = 32;
    const int E = 3;
    const int H = 4;   /* d_k = 8 */

    /*
       两边都要走 Net::forward/Net::backward。**不能**直接调 layer->backward():
       Net::backward 会先把 loss 写进最后一层的 e, 直接调的话层里的 e 还是全 0,
       于是 "两边梯度都是 0, 差值也是 0", 等价性检查会假通过。
    */
    Net netRef(MOE<E, H>::_(D, true));
    Net netMine(std::make_shared<SparseMoE<TransformerBlock<H>, E, E> >(D, true, 0));
    SparseMoE<TransformerBlock<H>, E, E> *my = dynamic_cast<SparseMoE<TransformerBlock<H>, E, E>*>(netMine[0]);
    MOE<E, H> *moe = dynamic_cast<MOE<E, H>*>(netRef[0]);

    /* 把上游 MOE 的权重搬到我的层上 (copyTo 就是干这个的) */
    my->wg = moe->wg;
    my->bg = moe->b;
    for (int i = 0; i < E; i++) {
        moe->experts[i].copyTo(&my->experts[i]);
    }

    Tensor x = randTensor(D, 1, 20240501, 1.0f);
    Tensor t = randTensor(D, 1, 99, 0.3f);

    Tensor outRef = netRef.forward(x, true);
    Tensor outMine = netMine.forward(x, true);
    double dF = 0;
    for (int j = 0; j < D; j++) {
        dF = std::fmax(dF, std::fabs((double)outRef[j] - (double)outMine[j]));
    }
    double scale = 0;
    for (int j = 0; j < D; j++) {
        scale = std::fmax(scale, std::fabs((double)outRef[j]));
    }
    std::printf("    前向: |y|~%.4f, 与 MOE<%d,%d> 的最大差 = %.3e\n", scale, E, H, dF);
    CHECK(scale > 1e-3, "上游 MOE 的输出不是恒 0 (等价性检查不是拿 0 比 0)");
    CHECK(dF < 1e-5 * std::fmax(1.0, scale), "稠密模式下前向与上游 MOE 一致");

    /* 反向的门控梯度也要一致 (这是两套实现的 softmax 雅可比路径) */
    netRef.backward(x, Loss::MSE::df(outRef, t));
    netMine.backward(x, Loss::MSE::df(outMine, t));
    double dW = 0;
    double wScale = 0;
    for (std::size_t i = 0; i < moe->g.wg.size(); i++) {
        dW = std::fmax(dW, std::fabs((double)moe->g.wg[i] - (double)my->g_wg[i]));
        wScale = std::fmax(wScale, std::fabs((double)moe->g.wg[i]));
    }
    double dB = 0;
    for (std::size_t i = 0; i < moe->g.b.size(); i++) {
        dB = std::fmax(dB, std::fabs((double)moe->g.b[i] - (double)my->g_bg[i]));
    }
    std::printf("    反向: |dL/dwg|~%.3e, wg 最大差 = %.3e, bg 最大差 = %.3e\n",
                wScale, dW, dB);
    CHECK(allFinite(my->g_wg), "门控梯度有限 (不是 NaN/1e28)");
    CHECK(wScale > 1e-6, "上游 MOE 的门控梯度非零 (等价性检查不是拿 0 比 0)");
    CHECK(dW < 1e-4 * std::fmax(1e-6, wScale), "稠密模式下门控 wg 梯度与上游 MOE 一致");
    CHECK(dB < 1e-4, "稠密模式下门控 bg 梯度与上游 MOE 一致");
}

/* ============================================================
 *  [4] 反向 vs 有限差分 (稀疏, E=4 top=2)
 * ============================================================ */
static void part4()
{
    std::printf("\n[4] 反向 vs 有限差分 (稀疏 MoE + MLP 专家, E=4 top=2)\n");
    const int D = 64;
    const int E = 4;
    const int TOP = 2;
    const int HID = 16;

    Net net(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, HID));
    SparseMoE<MlpExpert, E, TOP> *lay = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net[0]);

    /* 换成条件良好的随机权重, 否则 scaleInit 之后梯度只有 1e-5 量级, 差分会被
       浮点噪声淹没 (反正初始化的缩放不是这里要验的数学)。 */
    lay->wg = randTensor(E, D, 31337);
    lay->bg = randTensor(E, 1, 555);
    for (int i = 0; i < E; i++) {
        fillExpert(lay->experts[i], (unsigned)(1000 + 97 * i));
    }

    Tensor x = randTensor(D, 1, 24680, 1.0f);
    Tensor t = randTensor(D, 1, 13579, 0.3f);

    Tensor &o = net.forward(x, false);
    net.backward(x, Loss::MSE::df(o, t));

    /* 先把路由摊开: 未被选中的专家下标, 以及"路由不会在 eps 抖动下改变"的余量 */
    int selList[TOP];
    bool isSel[E];
    for (int i = 0; i < E; i++) {
        isSel[i] = false;
    }
    for (int s = 0; s < lay->nSelected; s++) {
        selList[s] = lay->selected[s];
        isSel[selList[s]] = true;
    }
    int unsel = -1;
    for (int i = 0; i < E; i++) {
        if (!isSel[i]) {
            unsel = i;
        }
    }
    std::printf("    选中 = %d,%d (gate:", selList[0], selList[1]);
    for (int i = 0; i < E; i++) {
        std::printf(" %.4f", (double)lay->gate[i]);
    }
    std::printf(")\n");
    CHECK(unsel >= 0, "top-2 of 4 时至少有一个专家未被选中");

    /* 路由余量: 第 2 名与第 3 名之间的差要远大于 eps, 否则差分测的就不是同一个函数 */
    double g2 = -1, g3 = -1;
    {
        std::vector<double> gv;
        for (int i = 0; i < E; i++) {
            gv.push_back((double)lay->gate[i]);
        }
        std::sort(gv.begin(), gv.end());
        g3 = gv[E - 3];
        g2 = gv[E - 2];
    }
    std::printf("    路由余量: gate 第2名 %.4f - 第3名 %.4f = %.4f\n", g2, g3, g2 - g3);
    CHECK(g2 - g3 > 1e-2, "第 2/3 名门控概率差距足够大, 差分不会跨过路由切换点");

    const float eps = 1e-3f;
    double worst = 0;
    char tag[96];

    /* --- 门控: 全部 E×D 个 wg 元素都查 (包含未被选中专家的行!) ---
       未被选中专家的门控行**有**非零梯度: 改它会让 softmax 重新分配, 从而改变
       被选中专家的概率。这是最容易写错的一环。 */
    FdResult r = checkParam(net, lay->wg, lay->g_wg, x, t, eps, 1);
    report("wg (全部 4x64)", r);
    CHECK(r.maxNum > 1e-6, "wg 的数值梯度非零 (差分真的看到了 loss 的变化)");
    CHECK(r.maxAna > 1e-6, "wg 的解析梯度非零 (反向确实填了门控梯度)");
    worst = std::fmax(worst, r.worst);

    r = checkParam(net, lay->bg, lay->g_bg, x, t, eps, 1);
    report("bg (全部 4x1)", r);
    worst = std::fmax(worst, r.worst);

    /* 单独看"未被选中专家"的那一行门控权重 */
    {
        FdResult ru = checkRow(net, lay->wg, lay->g_wg, unsel, x, t, eps, 4);
        report("wg 未选中专家的行 (抽样)", ru);
        CHECK(ru.maxAna > 1e-8, "未选中专家的门控行仍有非零解析梯度 (softmax 会重新分配)");
        CHECK(ru.maxNum > 1e-8, "未选中专家的门控行确实影响 loss (差分能看到)");
        CHECK(ru.worst < 5e-3, "未选中专家的门控行解析梯度与差分一致");
        worst = std::fmax(worst, ru.worst);
    }

    /* --- 被选中的专家: 全部参数逐元素 --- */
    for (int s = 0; s < TOP; s++) {
        MlpExpert &ex = lay->experts[selList[s]];
        std::snprintf(tag, sizeof(tag), "选中专家%d l1.w", selList[s]);
        r = checkParam(net, ex.l1.w, ex.l1.g.w, x, t, eps, 1);
        report(tag, r);
        CHECK(r.maxNum > 1e-8, "选中专家 l1.w 的数值梯度非零");
        CHECK(r.maxAna > 1e-8, "选中专家 l1.w 的解析梯度非零");
        worst = std::fmax(worst, r.worst);

        std::snprintf(tag, sizeof(tag), "选中专家%d l3.w", selList[s]);
        r = checkParam(net, ex.l3.w, ex.l3.g.w, x, t, eps, 1);
        report(tag, r);
        worst = std::fmax(worst, r.worst);

        std::snprintf(tag, sizeof(tag), "选中专家%d l2.b", selList[s]);
        r = checkParam(net, ex.l2.b, ex.l2.g.b, x, t, eps, 1);
        report(tag, r);
        worst = std::fmax(worst, r.worst);
    }

    /* --- 未被选中的专家: 解析梯度必须**恰好**是 0 (否则它会自己漂走) --- */
    {
        MlpExpert &ex = lay->experts[unsel];
        std::snprintf(tag, sizeof(tag), "未选中专家%d l1.w", unsel);
        r = checkParam(net, ex.l1.w, ex.l1.g.w, x, t, eps, 7);
        report(tag, r);
        CHECK(r.maxAna == 0.0, "未选中专家的参数梯度恰好为 0");
        CHECK(r.maxNum == 0.0, "未选中专家的参数对 loss 没有影响");
    }

    std::printf("    -> 最差相对误差 = %.3e (eps=%.0e)\n", worst, (double)eps);
    CHECK(worst < 5e-3, "稀疏 MoE 全参数解析梯度与有限差分一致 (相对误差 < 5e-3)");

    /*
       eps 扫描: 中心差分的误差有两项 —— 截断误差 ~ eps²·f'''/6 (eps 越小越好) 和
       舍入噪声 ~ loss·eps_float/(2·eps) (eps 越小越糟)。float32 下前者很快见底,
       剩下的是后者, 所以这里的期望是"eps 越小时误差变大", 而不是"越小越好"。
       实际噪声底可以估出来: loss = Σ(o-t)² 量级在 1e2~1e3, float32 的分辨率约
       loss·6e-8 ~ 1e-4, 于是 eps=1e-4 时 (L(+eps)-L(-eps))/(2eps) 的噪声就有
       1e-4/2e-4 = 0.5 的绝对误差 —— 足以解释下面看到的百分之几的相对误差。
       所以断言只放在 eps >= 1e-3 上 (那里噪声 ~1e-2 而梯度是 6.3, 约 0.2%)。
    */
    {
        std::printf("    eps 扫描 (wg 抽样)      :");
        double worstBig = 0;
        for (float e = 1e-2f; e >= 1e-4f; e *= 0.1f) {
            FdResult rr = checkParam(net, lay->wg, lay->g_wg, x, t, e, 7);
            std::printf("  eps=%.0e -> %.2e", (double)e, rr.worst);
            if (e >= 1e-3f) {
                worstBig = std::fmax(worstBig, rr.worst);
            }
        }
        std::printf("\n");
        CHECK(worstBig < 5e-3, "eps=1e-2/1e-3 时门控梯度与差分一致 (eps=1e-4 只受舍入噪声影响)");
    }
}

/* ============================================================
 *  [5] 稠密对照 (TopK = E)
 * ============================================================ */
static void part5()
{
    std::printf("\n[5] 反向 vs 有限差分 (稠密对照 TopK==E, E=3)\n");
    const int D = 48;
    const int E = 3;
    const int HID = 12;

    Net net(std::make_shared<SparseMoE<MlpExpert, E, E> >(D, true, HID));
    SparseMoE<MlpExpert, E, E> *lay = dynamic_cast<SparseMoE<MlpExpert, E, E>*>(net[0]);
    lay->wg = randTensor(E, D, 8080);
    lay->bg = randTensor(E, 1, 8181);
    for (int i = 0; i < E; i++) {
        fillExpert(lay->experts[i], (unsigned)(2000 + 31 * i));
    }

    Tensor x = randTensor(D, 1, 60606, 1.0f);
    Tensor t = randTensor(D, 1, 51515, 0.3f);
    Tensor &o = net.forward(x, false);
    net.backward(x, Loss::MSE::df(o, t));

    CHECK(lay->nSelected == E, "稠密对照模式下所有专家都被选中");

    const float eps = 1e-3f;
    FdResult r = checkParam(net, lay->wg, lay->g_wg, x, t, eps, 1);
    report("wg (全部 3x48)", r);
    double worst = r.worst;
    CHECK(r.maxNum > 1e-6 && r.maxAna > 1e-6, "稠密模式下门控梯度非零");

    r = checkParam(net, lay->experts[0].l2.w, lay->experts[0].l2.g.w, x, t, eps, 1);
    report("专家0 l2.w", r);
    worst = std::fmax(worst, r.worst);

    r = checkParam(net, lay->experts[2].l3.w, lay->experts[2].l3.g.w, x, t, eps, 1);
    report("专家2 l3.w", r);
    worst = std::fmax(worst, r.worst);

    std::printf("    -> 最差相对误差 = %.3e\n", worst);
    CHECK(worst < 5e-3, "稠密对照的解析梯度与有限差分一致");
}

/* ============================================================
 *  [6] 负载均衡辅助损失的梯度 vs 有限差分
 * ============================================================ */
static void part6()
{
    std::printf("\n[6] 负载均衡辅助损失 L_aux = coef·E·Σ f_i·P_i 的梯度\n");
    const int D = 40;
    const int E = 4;
    const int TOP = 1;
    const float coef = 0.25f;

    Net net(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
    SparseMoE<MlpExpert, E, TOP> *lay = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net[0]);
    lay->wg = randTensor(E, D, 3131);
    lay->bg = randTensor(E, 1, 6262);
    Tensor x = randTensor(D, 1, 909090, 1.0f);

    /* 只做一次前向 -> f_i = 1{被选中}/topK, P_i = gate_i */
    net.forward(x, true);
    const int sel = lay->selected[0];
    const double f[4] = {
        (sel == 0) ? 1.0 / (double)TOP : 0.0,
        (sel == 1) ? 1.0 / (double)TOP : 0.0,
        (sel == 2) ? 1.0 / (double)TOP : 0.0,
        (sel == 3) ? 1.0 / (double)TOP : 0.0
    };
    std::printf("    选中专家=%d; f = {%.1f, %.1f, %.1f, %.1f}\n",
                sel, f[0], f[1], f[2], f[3]);

    /* 参考函数: 直接把 L_aux 按定义算出来 (f 固定为上面的常数) */
    auto auxRef = [&](void) -> double {
        Tensor &oo = net.forward(x, true);
        (void)oo;
        double s = 0;
        for (int i = 0; i < E; i++) {
            s += f[i] * (double)lay->gate[i];
        }
        return (double)coef * (double)E * s;
    };

    lay->addAuxGradient(coef);
    CHECK(allFinite(lay->g_wg), "辅助损失的门控梯度有限");
    CHECK(allFinite(lay->g_bg), "辅助损失的偏置梯度有限");

    /* 这次 addAuxGradient 是在**没有调用主 backward** 的情况下做的, 所以 g_wg
       里只有辅助损失那一项, 可以直接和 L_aux 的中心差分对照。 */
    const float eps = 1e-3f;
    auto fdOn = [&](Tensor &w, Tensor &g, const char *tag, int rows, int cols, int stride) {
        double worst = 0, maxNum = 0, maxAna = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j += stride) {
                const float save = w(i, j);
                w(i, j) = save + eps;
                const double lp = auxRef();
                w(i, j) = save - eps;
                const double lm = auxRef();
                w(i, j) = save;
                const double num = (lp - lm) / (2.0 * (double)eps);
                const double ana = (double)g(i, j);
                maxNum = std::fmax(maxNum, std::fabs(num));
                maxAna = std::fmax(maxAna, std::fabs(ana));
                const double denom = std::fmax(1.0, std::fmax(std::fabs(num), std::fabs(ana)));
                worst = std::fmax(worst, std::fabs(num - ana) / denom);
            }
        }
        std::printf("    %-24s max_rel=%.2e  |num|max=%.3e |ana|max=%.3e\n",
                    tag, worst, maxNum, maxAna);
        return worst;
    };

    double w1 = fdOn(lay->wg, lay->g_wg, "dL_aux/dwg (全部)", E, D, 1);
    double w2 = fdOn(lay->bg, lay->g_bg, "dL_aux/dbg (全部)", E, 1, 1);
    CHECK(w1 < 5e-3, "辅助损失对 wg 的解析梯度与有限差分一致");
    CHECK(w2 < 5e-3, "辅助损失对 bg 的解析梯度与有限差分一致");

    /* 路由在扰动下没有改变 -> 上面用的 f 常数是对的 */
    CHECK(lay->selected[0] == sel, "扰动 eps 内路由没有改变 (f 是常数这件事成立)");

    /* 辅助损失不该碰专家的参数梯度 */
    {
        Net net2(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
        SparseMoE<MlpExpert, E, TOP> *l2 = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net2[0]);
        net2.forward(x, true);
        double before = 0;
        for (int i = 0; i < E; i++) {
            for (std::size_t k = 0; k < l2->experts[i].l1.g.w.size(); k++) {
                before += std::fabs((double)l2->experts[i].l1.g.w[k]);
            }
        }
        l2->addAuxGradient(coef);
        double after = 0;
        for (int i = 0; i < E; i++) {
            for (std::size_t k = 0; k < l2->experts[i].l1.g.w.size(); k++) {
                after += std::fabs((double)l2->experts[i].l1.g.w[k]);
            }
        }
        CHECK(before == after, "辅助损失只写门控参数, 不碰专家参数梯度");
    }

    /*
       n > 1 (真正的 mini-batch): 这里**不**做有限差分核对。
       实现用的是"平均场"近似 —— 用批均值的 P̄ 和 x̄ 代替逐样本的 (P_b, x_b),
       所以它与 L = coef·E·Σ f_i·P̄_i 的精确导数只在批内门控分布接近时才相等
       (见 sparse_moe.hpp 里 addAuxGradient 的注释)。这里只查两件不会含糊的事:
       梯度有限, 而且纠偏方向仍然正确。
    */
    {
        Net net3(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
        SparseMoE<MlpExpert, E, TOP> *l3 = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net3[0]);
        for (int j = 0; j < D; j++) {
            l3->wg(0, j) = 0.0f;
            l3->wg(1, j) = 0.0f;
            l3->wg(2, j) = 0.0f;
            l3->wg(3, j) = 0.0f;
        }
        l3->bg[0] = 3.0f;
        l3->bg[1] = 0.0f;
        l3->bg[2] = 0.0f;
        l3->bg[3] = 0.0f;
        for (int b = 0; b < 5; b++) {
            Tensor xb = randTensor(D, 1, (unsigned)(500 + 13 * b), 1.0f);
            net3.forward(xb, true);
        }
        l3->addAuxGradient(coef);
        std::printf("    n=5 的批: dz = {%.4e, %.4e, %.4e, %.4e}\n",
                    (double)l3->g_bg[0], (double)l3->g_bg[1],
                    (double)l3->g_bg[2], (double)l3->g_bg[3]);
        CHECK(allFinite(l3->g_wg) && allFinite(l3->g_bg), "批模式 (n=5) 的辅助梯度有限");
        CHECK((double)l3->g_bg[0] > 0 && (double)l3->g_bg[1] < 0,
              "批模式下纠偏方向不变 (被喂爆的压下去, 饿着的抬起来)");
    }
}

/* ============================================================
 *  [7] 辅助损失真的把坍缩的路由拉回来
 * ============================================================ */
static void part7()
{
    std::printf("\n[7] 专家坍缩与辅助损失的纠偏方向\n");
    const int D = 40;
    const int E = 4;
    const int TOP = 1;
    const float coef = 0.1f;

    Net net(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
    SparseMoE<MlpExpert, E, TOP> *lay = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net[0]);

    /* 人为把路由压到专家 0: 它的 logit 给 +3, 其它给 0 */
    for (int j = 0; j < D; j++) {
        lay->wg(0, j) = 0.0f;
        lay->wg(1, j) = 0.0f;
        lay->wg(2, j) = 0.0f;
        lay->wg(3, j) = 0.0f;
    }
    lay->bg[0] = 3.0f;
    lay->bg[1] = 0.0f;
    lay->bg[2] = 0.0f;
    lay->bg[3] = 0.0f;

    Tensor x = randTensor(D, 1, 112233, 1.0f);
    net.forward(x, true);
    const double gBefore[4] = { (double)lay->gate[0], (double)lay->gate[1],
                                (double)lay->gate[2], (double)lay->gate[3] };
    std::printf("    初始 gate = {%.4f, %.4f, %.4f, %.4f}\n",
                gBefore[0], gBefore[1], gBefore[2], gBefore[3]);
    CHECK(gBefore[0] > 0.85, "构造出了一条坍缩到专家0的路由");

    /* 坍缩时: 被喂爆的专家 logit 要被压下去, 饿着的专家要被抬起来 */
    net.forward(x, true);
    lay->addAuxGradient(coef);
    std::printf("    dz = {%.4e, %.4e, %.4e, %.4e}  (被喂爆的应为正, 饿着的应为负)\n",
                (double)lay->g_bg[0], (double)lay->g_bg[1],
                (double)lay->g_bg[2], (double)lay->g_bg[3]);
    CHECK((double)lay->g_bg[0] > 0, "辅助损失把被喂爆的专家的 logit 梯度推成正 (RMSProp 下降)");
    CHECK((double)lay->g_bg[1] < 0, "辅助损失把饿着的专家的 logit 梯度推成负 (概率会被抬起来)");

    /* 多步之后路由必须比原来均匀 (这里只看辅助损失的作用, 不给专家真梯度) */
    for (int it = 0; it < 40; it++) {
        net.forward(x, true);
        lay->addAuxGradient(coef);
        lay->RMSProp(0.05f, 0.9f, 0.0f, true);
    }
    net.forward(x, true);
    const double gAfter[4] = { (double)lay->gate[0], (double)lay->gate[1],
                               (double)lay->gate[2], (double)lay->gate[3] };
    auto entropy = [](const double *g) {
        double h = 0;
        for (int i = 0; i < 4; i++) {
            if (g[i] > 1e-12) {
                h -= g[i] * std::log(g[i]);
            }
        }
        return h;
    };
    const double h0 = entropy(gBefore);
    const double h1 = entropy(gAfter);
    std::printf("    40 步之后 gate = {%.4f, %.4f, %.4f, %.4f}, 熵 %.4f -> %.4f (均匀=%.4f)\n",
                gAfter[0], gAfter[1], gAfter[2], gAfter[3], h0, h1, std::log(4.0));
    CHECK(gAfter[0] < gBefore[0], "被喂爆的专家概率下降");
    CHECK(h1 > h0, "路由的熵上升 (朝着负载均衡走)");
    CHECK(allFinite(lay->wg) && allFinite(lay->bg), "辅助损失更新后参数仍然有限 (没有数值爆炸)");
}

/* ============================================================
 *  [8] 使用计数 (坍缩诊断) 与权重存取往返
 * ============================================================ */
static void part8()
{
    std::printf("\n[8] 使用计数 (坍缩诊断) 与权重存取往返\n");
    const int D = 32;
    const int E = 4;
    const int TOP = 2;

    Net net(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
    SparseMoE<MlpExpert, E, TOP> *lay = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net[0]);
    ISparseMoE *iface = lay;

    Tensor x = randTensor(D, 1, 7, 1.0f);
    const int N = 25;
    for (int i = 0; i < N; i++) {
        net.forward(x, true);
    }
    std::vector<long long> usage;
    iface->usageSnapshot(usage);
    long long total = 0;
    std::printf("    %d 次 top-2 前向的使用计数 = {", N);
    for (int i = 0; i < E; i++) {
        total += usage[(std::size_t)i];
        std::printf("%lld%s", usage[(std::size_t)i], (i + 1 < E) ? ", " : "}\n");
    }
    CHECK(usage.size() == (std::size_t)E, "usageSnapshot 返回 E 个计数");
    CHECK(total == (long long)N * TOP, "使用计数总和 = 前向次数 × topK (= 工作量守恒)");

    iface->resetUsage();
    iface->usageSnapshot(usage);
    long long total2 = 0;
    for (int i = 0; i < E; i++) {
        total2 += usage[(std::size_t)i];
    }
    CHECK(total2 == 0, "resetUsage 把计数清零");

    /* 权重往返: 门控 + 每个专家 */
    const std::string path = "test_sparse_moe_weights.txt";
    CHECK(net.save(path) == 0, "save 成功");

    Net net2(std::make_shared<SparseMoE<MlpExpert, E, TOP> >(D, true, 8));
    Tensor before = net.forward(x, true);
    CHECK(net2.load(path) == 0, "load 成功");
    Tensor after = net2.forward(x, true);
    double d = 0;
    for (int j = 0; j < D; j++) {
        d = std::fmax(d, std::fabs((double)before[j] - (double)after[j]));
    }
    std::printf("    存取往返: 两次输出最大差 = %.3e\n", d);
    CHECK(d < 1e-6, "save/load 之后前向输出一致 (门控与所有专家都存下来了)");

    /* 载入之后路由也要能正常工作 */
    SparseMoE<MlpExpert, E, TOP> *l2 = dynamic_cast<SparseMoE<MlpExpert, E, TOP>*>(net2[0]);
    CHECK(l2->nSelected == TOP, "载入后的层仍然正常做 top-k 路由");
    CHECK(allFinite(l2->wg) && allFinite(l2->bg), "载入后的门控参数有限 (不是空张量)");
    std::remove(path.c_str());

    /*
       TransformerBlock 专家的序列化也要往返一遍。
       分层的 write/read **顺序**只要有一处不一致, 载入出来的就是另一个网络, 而且是
       静默的 —— 这类错误只能靠"存了再读、比对输出"发现。这里用小 d_model (32) 做,
       因为 1260 维上的 TB 专家有 28.7 M 参数, 而权重是十进制文本, 存一次就是几百 MB。
    */
    {
        const int D2 = 32;
        const std::string path2 = "test_sparse_moe_weights_tb.txt";
        Net tb1(std::make_shared<SparseMoE<TransformerBlock<4, 16>, 2, 1> >(D2, true, 0));
        Tensor x2 = randTensor(D2, 1, 191, 1.0f);
        Tensor before2 = tb1.forward(x2, true);
        CHECK(tb1.save(path2) == 0, "TransformerBlock 专家版本 save 成功");

        Net tb2(std::make_shared<SparseMoE<TransformerBlock<4, 16>, 2, 1> >(D2, true, 0));
        CHECK(tb2.load(path2) == 0, "TransformerBlock 专家版本 load 成功");
        Tensor after2 = tb2.forward(x2, true);
        double d2 = 0;
        for (int j = 0; j < D2; j++) {
            d2 = std::fmax(d2, std::fabs((double)before2[j] - (double)after2[j]));
        }
        SparseMoE<TransformerBlock<4, 16>, 2, 1> *lt1 =
            dynamic_cast<SparseMoE<TransformerBlock<4, 16>, 2, 1>*>(tb1[0]);
        SparseMoE<TransformerBlock<4, 16>, 2, 1> *lt2 =
            dynamic_cast<SparseMoE<TransformerBlock<4, 16>, 2, 1>*>(tb2[0]);
        double dg = 0;
        for (std::size_t i = 0; i < lt1->wg.size(); i++) {
            dg = std::fmax(dg, std::fabs((double)lt1->wg[i] - (double)lt2->wg[i]));
        }
        std::printf("    TB 专家往返: 输出最大差 = %.3e, 门控 wg 最大差 = %.3e\n", d2, dg);
        CHECK(d2 < 1e-6, "TB 专家版本存取往返后前向输出一致 (读写顺序正确)");
        CHECK(dg < 1e-6, "TB 专家版本的门控权重也往返一致");
        std::remove(path2.c_str());
    }

    /* 初始化缩放: 1260 维输入下 U(-1,1) 会把 Tanh 顶到饱和, 所以必须有这一步 */
    {
        SparseMoE<MlpExpert, 2, 1> raw(1260, true, 64);
        double m = 0;
        for (std::size_t i = 0; i < raw.experts[0].l1.w.size(); i++) {
            m = std::fmax(m, std::fabs((double)raw.experts[0].l1.w[i]));
        }
        std::printf("    scaleInit 之后 |w|max = %.4f (未缩放应是 ~1.0, 1/sqrt(1260)=%.4f)\n",
                    m, 1.0 / std::sqrt(1260.0));
        CHECK(m < 0.2, "专家权重被 1/sqrt(fan_in) 缩放了 (不会一上来就把 Tanh 顶饱和)");
        CHECK(m > 1e-3, "缩放之后权重没有被清零");
    }
}

/* ============================================================
 *  [9] 代价: 稀疏 vs 稠密, MLP 专家 vs TransformerBlock 专家
 * ============================================================ */
static double timeForward(Net &net, const Tensor &x, int n)
{
    net.forward(x, true);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        net.forward(x, true);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns =
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return ns / 1e6 / (double)n;
}

static void part9()
{
    std::printf("\n[9] 代价: 稀疏 vs 稠密, MLP 专家 vs TransformerBlock 专家\n");
    const int D = 1260;
    Tensor x = randTensor(D, 1, 31415, 0.2f);

    Net mlpSparse(std::make_shared<SparseMoE<MlpExpert, 8, 2> >(D, true, 64));
    const double msA = timeForward(mlpSparse, x, 300);

    Net tbSparse(std::make_shared<SparseMoE<TransformerBlock<15, 315>, 4, 1> >(D, true, 0));
    const double msB = timeForward(tbSparse, x, 10);

    Net tbDense(std::make_shared<SparseMoE<TransformerBlock<15, 315>, 4, 4> >(D, true, 0));
    const double msC = timeForward(tbDense, x, 10);

    std::printf("    稀疏 MoE + MLP 专家 (E=8, top-2, h=64) : %8.4f ms/前向\n", msA);
    std::printf("    稀疏 MoE + TB 专家  (E=4, top-1)       : %8.4f ms/前向\n", msB);
    std::printf("    稠密 MoE + TB 专家  (E=4, 全算)        : %8.4f ms/前向 (稀疏快 %.1fx)\n",
                msC, msC / msB);

    CHECK(msA > 0 && msB > 0 && msC > 0, "三种骨干的耗时都量到了");
    /*
       稀疏的卖点必须真的兑现: 只算 1/4 的专家要显著快于全算。
       留的余量很宽 (只要快 2 倍以上) —— 这里防的是"稀疏退化成稠密"这种回归。
    */
    CHECK(msB * 2.0 < msC, "top-1 的 TB 专家确实显著快于全算 4 个 (稀疏没退化成稠密)");

    /*
       每步走子的预算 (256 次模拟 × 3 个网络) —— 说明为什么 TB 专家只能当
       "低模拟次数"的变体, 而 MLP 专家才是能用在实际对局里的那个。
    */
    std::printf("    每步走子预算 (256 次模拟 × 3 个网络):\n");
    std::printf("      MLP 专家骨干        : %7.1f ms\n", msA * 256 * 3);
    std::printf("      TB 专家骨干 (top-1) : %7.1f ms (只能配很小的模拟次数)\n", msB * 256 * 3);
    std::printf("      TB 专家骨干 (全算)  : %7.1f ms (实际不可用)\n", msC * 256 * 3);
}

int main()
{
    /* 关掉 stdout 缓冲: 这个测试里有若干耗时较长的数值检查, 万一卡住或崩掉,
       块缓冲会让"卡在哪一步"完全看不出来 (加这一段就是因为排查时吃过这个亏)。
       其它分钟级的测试基准 (test_grad / test_ppomcts ...) 也都这么做。 */
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== 稀疏路由 MoE (rl/sparse_moe.hpp) 测试 ===\n");
    Random::setSeed(20240501);

    std::printf("[1/9] 前向/反向基础\n"); part1();
    std::printf("[2/9]\n"); part2();
    std::printf("[3/9]\n"); part3();
    std::printf("[4/9]\n"); part4();
    std::printf("[5/9]\n"); part5();
    std::printf("[6/9]\n"); part6();
    std::printf("[7/9]\n"); part7();
    std::printf("[8/9]\n"); part8();
    std::printf("[9/9]\n"); part9();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
