/*
 * bench_gate_moe_main.cpp — MoE 门控变体的受控实验 (不进 ctest)
 * ============================================================================
 *
 * 回答的问题 (按重要性排序):
 *
 *   Q1 新增的 MLP 门控 / logistic 门控 / 无辅助损失偏置均衡, **梯度写对了吗?**
 *      -> [1] 有限差分. 本工程的方法学要求: "能跑、loss 在降"完全不能证明梯度是对的。
 *
 *   Q2 MLP 门控 / logistic 门控 **比线性+softmax 更好吗?** 好在哪一类任务上?
 *      -> [2] 两个**可证伪**的合成任务, 事先写下预测:
 *         任务 L (4 个线性可分区域): 线性路由器已经够用 => 预测 MLP **无收益**。
 *         任务 X (XOR 型区域,  线性不可分): 线性路由器必须"两个专家当一个用"
 *                                        => 预测 MLP **有收益**。
 *         如果实测与预测不符, 就是这套假设错了 —— 这是本实验设计的全部意义。
 *
 *   Q3 真实 SAC+AZ-MoE-TB 那条骨干的负载不均衡, 是**门控结构的锅**还是**数据/目标的锅**?
 *      -> [3] 用真实棋局状态 (Chess + ChessState 编码, 1710 维) 做两件事:
 *         (a) 同一份门控、同一输入尺度下, 真实棋局状态 vs 各向同性高斯状态的路由偏斜对比
 *             —— 若真实状态明显更偏, 说明偏斜是"状态分布窄"贡献的, 不是门控结构。
 *         (b) **均衡可达性**: 把"均衡"当唯一目标去优化门控, 看每种结构最多能做到多均衡。
 *             若线性+softmax 也能做到接近均匀 => 结构不是瓶颈, 该换的是**目标**
 *             (也就是无辅助损失偏置均衡这条路); 若做不到 => 结构才是瓶颈。
 *
 * 口径说明 (避免误读):
 *   * 本程序**不测棋力**。它测的是路由机制与门控学习, 与"更强"没有直接关系
 *     (仓库已实测: 机制通了、能赢棋了, 但稀疏 vs 稠密等时对弈全是和棋)。
 *   * [3] 的 1710 维只用真实编码, **不跑 MCTS、不用 TB 专家** —— 因为路由只依赖 `wg`
 *     和输入分布, 与专家是什么无关。这是让实验便宜且干净的根据。
 *   * 所有对比都在**固定的前向/样本预算**下做, 不放任某一臂多算。
 *
 * 用法:
 *   bench_gate_moe [--seeds=8] [--steps=600] [--states=600] [--quick]
 *                  [--only=1|2|3]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "chessstate.h"
#include "pos.h"
#include "rl/expert.hpp"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/net.hpp"
#include "rl/sparse_moe.hpp"
#include "rl/tensor.hpp"
#include "rl/util.hpp"
#include "stone.h"

using RL::GateActivation;
using RL::GateStructure;

namespace {

int g_seeds = 8;
int g_steps = 600;
int g_states = 600;

/* ============================================================
 *  小工具
 * ============================================================ */

RL::Tensor randTensor(int r, int c, unsigned seed, float amp = 0.5f)
{
    /* 与 test_sparse_moe 同一个生成器: 均匀 [-amp, amp] (不是正态) */
    RL::Tensor t((std::size_t)r, (std::size_t)c);
    unsigned s = seed;
    for (std::size_t i = 0; i < t.size(); i++) {
        s = s * 1664525u + 1013904223u;
        t[i] = (float)((double)(s >> 8) / (double)(1u << 24) * 2.0 - 1.0) * amp;
    }
    return t;
}

/* L = 所有 MSE 分量之和; 它的梯度正好是 Loss::MSE::df */
double lossOf(RL::Net &net, const RL::Tensor &x, const RL::Tensor &t)
{
    RL::Tensor &o = net.forward(x, true);
    RL::Tensor l = RL::Loss::MSE::f(o, t);
    double s = 0.0;
    for (std::size_t i = 0; i < l.size(); i++) {
        s += (double)l[i];
    }
    return s;
}

/* 把一个 MlpExpert 填成"条件良好"的随机值 (含偏置; 与 test_sparse_moe 同口径) */
void fillExpert(RL::MlpExpert &ex, unsigned seed)
{
    ex.l1.w = randTensor((int)ex.l1.w.shape[0], (int)ex.l1.w.shape[1], seed + 1);
    ex.l1.b = randTensor((int)ex.l1.b.shape[0], 1, seed + 2);
    ex.l2.w = randTensor((int)ex.l2.w.shape[0], (int)ex.l2.w.shape[1], seed + 3);
    ex.l2.b = randTensor((int)ex.l2.b.shape[0], 1, seed + 4);
    ex.l3.w = randTensor((int)ex.l3.w.shape[0], (int)ex.l3.w.shape[1], seed + 5);
    ex.l3.b = randTensor((int)ex.l3.b.shape[0], 1, seed + 6);
}

bool allFinite(const RL::Tensor &t)
{
    for (std::size_t i = 0; i < t.size(); i++) {
        if (!std::isfinite((double)t[i])) {
            return false;
        }
    }
    return true;
}

double maxAbs(const RL::Tensor &t)
{
    double m = 0.0;
    for (std::size_t i = 0; i < t.size(); i++) {
        m = std::fmax(m, std::fabs((double)t[i]));
    }
    return m;
}

/* ---- 路由负载的读法 (统一口径, 见报告 §指标) ---- */
struct LoadStat {
    std::vector<double> share;
    double maxVio;      /* max_i share_i / (1/E) - 1 ; 与 Loss-Free 论文的 MaxVio 同口径 */
    double minShare;
    int    effective;   /* 份额 > 5% 的专家个数 */
};

LoadStat loadOf(const std::vector<long long> &usage)
{
    LoadStat s;
    const int E = (int)usage.size();
    long long total = 0;
    for (int i = 0; i < E; i++) {
        total += usage[(std::size_t)i];
    }
    s.share.assign((std::size_t)E, 0.0);
    s.maxVio = 0.0;
    s.minShare = 1.0;
    s.effective = 0;
    if (total <= 0) {
        s.maxVio = 1.0;
        s.minShare = 0.0;
        return s;
    }
    for (int i = 0; i < E; i++) {
        const double sh = (double)usage[(std::size_t)i] / (double)total;
        s.share[(std::size_t)i] = sh;
        s.maxVio = std::fmax(s.maxVio, sh * (double)E - 1.0);
        s.minShare = std::fmin(s.minShare, sh);
        if (sh > 0.05) {
            s.effective++;
        }
    }
    return s;
}

/* ============================================================
 *  [1] 有限差分校验
 * ============================================================ */

struct FdResult {
    double maxNum = 0.0;
    double maxAna = 0.0;
    double worst = 0.0;
};

/*
 *  对 `param` 的每个元素做中心差分。**与 test_sparse_moe 的 checkParam 同一口径**:
 *  loss 用 `lossOf` (即 net.forward(x, true) + Σ MSE 分量), 分母 max(1.0, ...)。
 *
 *  为什么照搬而不再写一套: top-1/top-k 路由是不可导的 argmax, eps 抖动有可能跨过
 *  路由切换点。既有那套已经被 82 项断言验过, 换一套写法引入的差异只会让"到底是我
 *  的门控梯度错了还是我的差分写错了"变得不可判。
 */
template <typename NetT>
FdResult checkParam(NetT &net, RL::Tensor &param, RL::Tensor &grad,
                    const RL::Tensor &x, const RL::Tensor &t,
                    float eps, int stride)
{
    FdResult r;
    const std::size_t n = param.size();
    for (std::size_t i = 0; i < n; i += (std::size_t)stride) {
        const float save = param[i];
        param[i] = save + eps;
        const double lp = lossOf(net, x, t);
        param[i] = save - eps;
        const double lm = lossOf(net, x, t);
        param[i] = save;

        const double num = (lp - lm) / (2.0 * (double)eps);
        const double ana = (double)grad[i];
        r.maxNum = std::fmax(r.maxNum, std::fabs(num));
        r.maxAna = std::fmax(r.maxAna, std::fabs(ana));
        const double denom = std::fmax(1.0, std::fmax(std::fabs(num), std::fabs(ana)));
        r.worst = std::fmax(r.worst, std::fabs(num - ana) / denom);
    }
    return r;
}

void reportFd(const char *tag, const FdResult &r)
{
    std::printf("    %-34s |num|max=%.3e |ana|max=%.3e  rel=%.3e\n",
                tag, r.maxNum, r.maxAna, r.worst);
}

/* 一套门控配置 */
struct GateCfg {
    const char   *name;
    GateStructure gs;
    GateActivation ga;
    int           mlpHidden;
    bool          useBias;
    float         biasRate;
    float         auxCoef;
};

const GateCfg kArms[] = {
    /* 生产现状: 线性 + softmax + Switch 辅助损失(0.1) */
    {"linear+softmax+aux0.1",  GateStructure::Linear, GateActivation::Softmax,  0, false, 0.0f,   0.1f},
    /* 只把"均衡"从损失里挪到无梯度控制回路 */
    {"linear+softmax+lossfree",GateStructure::Linear, GateActivation::Softmax,  0, true,  0.001f, 0.0f},
    /* 换 sigmoid (取消专家零和竞争) */
    {"linear+logistic+lossfree",GateStructure::Linear,GateActivation::Logistic, 0, true,  0.001f, 0.0f},
    /* 非线性路由 */
    {"mlp+softmax+lossfree",   GateStructure::Mlp,    GateActivation::Softmax,  32, true, 0.001f, 0.0f},
    {"mlp+logistic+lossfree",  GateStructure::Mlp,    GateActivation::Logistic, 32, true, 0.001f, 0.0f},
    /* 饱和陷阱的对照: sigmoid 配"靠损失均衡" */
    {"linear+logistic+aux0.1", GateStructure::Linear, GateActivation::Logistic, 0, false, 0.0f,   0.1f},
    {"mlp+logistic+aux0.1",    GateStructure::Mlp,    GateActivation::Logistic, 32, false, 0.0f,   0.1f},
};
const int kNumArms = (int)(sizeof(kArms) / sizeof(kArms[0]));

/* 按配置造一个 "MoE + 标量读出" 的网 */
struct Model {
    RL::Net net;
    RL::ISparseMoE *moe = nullptr;
    int D = 0;
};

std::shared_ptr<RL::SparseMoE<RL::MlpExpert, 4, 1> > makeMoe(int D, int hidden, const GateCfg &c)
{
    auto m = std::make_shared<RL::SparseMoE<RL::MlpExpert, 4, 1> >(D, true, hidden);
    if (c.gs == GateStructure::Mlp) {
        m->enableMlpGate(c.mlpHidden);
    }
    m->setGateActivation(c.ga);
    m->setLossFreeBias(c.useBias, c.biasRate);
    return m;
}

void buildModel(Model &mdl, int D, const GateCfg &c)
{
    auto moe = makeMoe(D, 16, c);
    RL::Net::Layers L;
    L.push_back(moe);
    L.push_back(RL::Layer<RL::Linear>::_(D, 1, true, true));
    mdl.net = RL::Net(L);
    mdl.moe = moe.get();
    mdl.D = D;
}

void part1()
{
    std::printf("\n[1] 有限差分: 四条门控路径的解析梯度\n");
    std::printf("    harness 与 test_sparse_moe 的 [4] 节逐项对齐: 单层 Net + MLP 专家, D=64,\n");
    std::printf("    top-2, 均匀权重的 wg/bg 与专家权重(含偏置), loss = Σ MSE 分量, eps=1e-3\n");

    const int D = 64;
    const int E = 4;
    const int TOP = 2;
    const int HID = 16;

    const GateCfg arms[] = {
        {"linear+softmax", GateStructure::Linear, GateActivation::Softmax,  0, false, 0.0f, 0.0f},
        {"linear+logistic",GateStructure::Linear, GateActivation::Logistic, 0, false, 0.0f, 0.0f},
        {"mlp+softmax",    GateStructure::Mlp,    GateActivation::Softmax,  8, false, 0.0f, 0.0f},
        {"mlp+logistic",   GateStructure::Mlp,    GateActivation::Logistic, 8, false, 0.0f, 0.0f},
    };

    int failures = 0;
    for (int a = 0; a < 4; a++) {
        const GateCfg &c = arms[a];
        auto moe = std::make_shared<RL::SparseMoE<RL::MlpExpert, E, TOP> >(D, true, HID);
        if (c.gs == GateStructure::Mlp) {
            moe->enableMlpGate(c.mlpHidden);
        }
        moe->setGateActivation(c.ga);
        RL::Net net(moe);
        RL::SparseMoE<RL::MlpExpert, E, TOP> *lay = moe.get();

        /* 换成条件良好的权重 —— 默认 fan-in 缩放会把梯度压到 1e-5, 被 float32 噪声淹没 */
        lay->wg = randTensor(E, D, 31337u + (unsigned)a);
        lay->bg = randTensor(E, 1, 555u + (unsigned)a);
        if (c.gs == GateStructure::Mlp) {
            lay->wg1 = randTensor(c.mlpHidden, D, 777u + (unsigned)a);
            lay->bg1 = randTensor(c.mlpHidden, 1, 888u + (unsigned)a);
            lay->wg2 = randTensor(E, c.mlpHidden, 999u + (unsigned)a);
            lay->bg2 = randTensor(E, 1, 111u + (unsigned)a);
        }
        for (int i = 0; i < E; i++) {
            fillExpert(lay->experts[i], 1000u + 97u * (unsigned)i + (unsigned)a);
        }

        RL::Tensor x = randTensor(D, 1, 24680u + (unsigned)a, 1.0f);
        RL::Tensor t = randTensor(D, 1, 13579u, 0.3f);

        RL::Tensor &o = net.forward(x, false);
        net.backward(x, RL::Loss::MSE::df(o, t));

        /* 路由余量: 第 TOP 名与第 TOP+1 名之差要远大于 eps, 否则差分跨过 argmax 切换点 */
        std::vector<double> gv;
        for (int i = 0; i < E; i++) {
            gv.push_back((double)lay->gate[(std::size_t)i]);
        }
        std::sort(gv.begin(), gv.end());
        const double margin = gv[E - TOP] - gv[E - TOP - 1];

        std::printf("\n  --- %s (选中=%d,%d 路由余量=%.4f) ---\n",
                    c.name, lay->selected[0], lay->selected[1], margin);
        if (margin < 1e-2) {
            std::printf("    [跳过] 路由余量 < 1e-2, 差分不可信\n");
            failures++;
            continue;
        }

        const float eps = 1e-3f;
        FdResult r = checkParam(net, lay->wg, lay->g_wg, x, t, eps, 1);
        reportFd("wg", r);
        FdResult rb = checkParam(net, lay->bg, lay->g_bg, x, t, eps, 1);
        reportFd("bg", rb);

        if (c.gs == GateStructure::Linear) {
            if (r.maxNum < 1e-7) { std::printf("    [失败] 数值梯度为 0: 差分没看到 loss 变化\n"); failures++; }
            if (r.maxAna < 1e-7) { std::printf("    [失败] 解析梯度为 0: 反向没填门控梯度\n"); failures++; }
            if (r.worst >= 5e-3) { std::printf("    [失败] wg 相对误差 %.3e >= 5e-3\n", r.worst); failures++; }
            if (rb.worst >= 5e-3) { std::printf("    [失败] bg 相对误差 %.3e\n", rb.worst); failures++; }
        } else {
            /*
               MLP 模式下 wg/bg **完全不被使用** —— 所以两侧都必须**恰好**为 0。
               这是"线性门控真的被旁路掉了"的正面断言, 而不是失败项
               (既有测试也把"前向没用这个参数"判为一致)。
            */
            if (r.maxNum != 0.0 || r.maxAna != 0.0) {
                std::printf("    [失败] MLP 模式下列性门控 wg 仍在起作用 (num=%.3e ana=%.3e)\n",
                            r.maxNum, r.maxAna);
                failures++;
            }
            if (rb.maxNum != 0.0 || rb.maxAna != 0.0) {
                std::printf("    [失败] MLP 模式下列性门控 bg 仍在起作用\n");
                failures++;
            }
            if (r.maxAna == 0.0 && r.maxNum == 0.0) {
                std::printf("    (wg/bg 恰好为 0 = 线性门控已被旁路, 符合预期)\n");
            }

            r = checkParam(net, lay->wg1, lay->g_wg1, x, t, eps, 7);
            reportFd("wg1 (抽样 stride=7)", r);
            if (r.maxNum < 1e-7 || r.maxAna < 1e-7) {
                std::printf("    [失败] wg1 梯度有 0 -> 差分没看到 loss 或反向漏填\n");
                failures++;
            }
            if (r.worst >= 5e-3) { std::printf("    [失败] wg1 相对误差 %.3e\n", r.worst); failures++; }

            r = checkParam(net, lay->bg1, lay->g_bg1, x, t, eps, 1);
            reportFd("bg1", r);
            if (r.maxAna < 1e-7 || r.worst >= 5e-3) { failures++; }

            r = checkParam(net, lay->wg2, lay->g_wg2, x, t, eps, 1);
            reportFd("wg2", r);
            if (r.maxAna < 1e-7 || r.worst >= 5e-3) { failures++; }

            r = checkParam(net, lay->bg2, lay->g_bg2, x, t, eps, 1);
            reportFd("bg2", r);
            if (r.maxAna < 1e-7 || r.worst >= 5e-3) { failures++; }
        }
    }
    std::printf("\n  => [1] %s (失败项 %d)\n", failures == 0 ? "全部通过" : "有失败", failures);
}

/* ============================================================
 *  [2] 两个可证伪的合成任务
 * ============================================================ */

/*
   两个任务共用同一条流水线, 只有 target 不同:
     输入 x ~ N(0,1)^D                                                    (D=16)
     区域: 由前 4 个坐标的符号决定
     目标: y = Σ_k 1[x∈R_k] · (u_k · x)      (u_k 是该区域专属的线性读出)
   MoE 若能把区域分对, 4 个专家就能各管一个区域 -> 误差低且负载均衡 (25% each)。

   任务 L: R_k = 四个象限 {x0≷0, x1≷0} —— **线性可分的 4 个区域**, 线性路由器够用。
   任务 X: R_k 由 (sign(x0*x1), sign(x2*x3)) 决定 —— 每个区域是**两个对顶象限的并**,
           单个线性函数无法把它与其余区域分开, 线性路由器只能靠"两个专家凑一个区域",
           于是 4 个专家只够覆盖 2 个区域。MLP 路由器理论上能直接表示。

   事先写下的预测:
     任务 L: MLP **没有**明显收益 (线性已经够)。
     任务 X: MLP **有**收益 (MSE 更低)。
   如果实测反了, 说明"MLP 门控能买到非线性路由"这个假设需要修正。
*/
int regionOf(const char *task, const RL::Tensor &x)
{
    const double x0 = (double)x[0], x1 = (double)x[1];
    const double x2 = (double)x[2], x3 = (double)x[3];
    if (std::strcmp(task, "L") == 0) {
        return ((x0 > 0.0) ? 1 : 0) + ((x1 > 0.0) ? 2 : 0);
    }
    const int a = ((x0 * x1) > 0.0) ? 1 : 0;
    const int b = ((x2 * x3) > 0.0) ? 1 : 0;
    return a + 2 * b;
}

void targetOf(const char *task, const RL::Tensor &x, double &y)
{
    const int region = regionOf(task, x);
    std::mt19937 rng(9000u + (unsigned)region);
    std::normal_distribution<double> nd(0.0, 1.0);
    double s = 0.0;
    for (int j = 0; j < 4; j++) {
        s += nd(rng) * (double)x[(std::size_t)j];
    }
    y = s;
}

struct ExpResult {
    double mse = 0.0;
    LoadStat load;
    double purity = 0.0;      /* 每个区域里, 被路由到"该区域多数专家"的样本占比 */
    int    regionExpertPairs = 0;  /* 负载 >10% 的 (区域,专家) 对数: 完美分离 = 4 */
};

ExpResult runTask(const char *task, const GateCfg &c, unsigned seed, int D)
{
    Model mdl;
    buildModel(mdl, D, c);

    std::mt19937 rng(seed * 7919u + 13u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int BATCH = 64;
    RL::Tensor xb((std::size_t)D, 1);
    RL::Tensor target((std::size_t)1, 1);

    const float lr = 0.02f;
    for (int step = 0; step < g_steps; step++) {
        mdl.moe->resetBatchStats();
        for (int b = 0; b < BATCH; b++) {
            for (int j = 0; j < D; j++) {
                xb[(std::size_t)j] = nd(rng);
            }
            double y = 0.0;
            targetOf(task, xb, y);
            target[0] = (float)y;

            RL::Tensor &o = mdl.net.forward(xb, false);
            RL::Tensor dLdy((std::size_t)1, (std::size_t)1);
            dLdy[0] = (float)(2.0 / (double)BATCH) * (o[0] - target[0]);
            mdl.net.backward(xb, dLdy);
        }
        /* 偏置更新必须在批统计被清零之前 */
        mdl.moe->updateLossFreeBias();
        if (c.auxCoef > 0.0f) {
            mdl.moe->addAuxGradient(c.auxCoef);
        } else {
            mdl.moe->resetBatchStats();
        }
        mdl.net.RMSProp(lr, 0.9f, 0.0f);
    }

    /* 留出集: MSE + 负载 + 路由纯度 */
    const int TEST = 2000;
    double se = 0.0;
    mdl.moe->resetUsage();
    long long cnt[4][4];
    for (int r = 0; r < 4; r++) {
        for (int e = 0; e < 4; e++) {
            cnt[r][e] = 0;
        }
    }
    for (int i = 0; i < TEST; i++) {
        for (int j = 0; j < D; j++) {
            xb[(std::size_t)j] = nd(rng);
        }
        double y = 0.0;
        targetOf(task, xb, y);
        RL::Tensor &o = mdl.net.forward(xb, true);
        const double d = (double)o[0] - y;
        se += d * d;
        /* top-1: 看这一样本被路由给谁 */
        const int e = static_cast<RL::SparseMoE<RL::MlpExpert, 4, 1> *>(mdl.moe)->selected[0];
        const int r = regionOf(task, xb);
        if (e >= 0 && e < 4) {
            cnt[r][e]++;
        }
    }
    std::vector<long long> usage;
    mdl.moe->usageSnapshot(usage);

    ExpResult r;
    r.mse = se / (double)TEST;
    r.load = loadOf(usage);

    /* 纯度 + (区域,专家) 配对数 */
    long long majorityTotal = 0;
    const long long thr = (long long)(0.10 * (double)TEST / 4.0);
    for (int rr = 0; rr < 4; rr++) {
        long long best = 0;
        for (int e = 0; e < 4; e++) {
            best = std::max(best, cnt[rr][e]);
            if (cnt[rr][e] > thr) {
                r.regionExpertPairs++;
            }
        }
        majorityTotal += best;
    }
    r.purity = (double)majorityTotal / (double)TEST;
    return r;
}

void part2()
{
    std::printf("\n[2] 合成专业化基准 (D=16, 4 专家 top-1, 每臂 %d 步 × 64 样本, %d 个种子)\n",
                g_steps, g_seeds);
    std::printf("    预测(事先写下): 任务 L 上 MLP 无收益; 任务 X 上 MLP 有收益\n\n");

    const char *tasks[2] = {"L", "X"};
    for (int ti = 0; ti < 2; ti++) {
        std::printf("  === 任务 %s (%s) ===\n", tasks[ti],
                    ti == 0 ? "4 个线性可分区域" : "XOR 型区域, 线性不可分");
        std::printf("    %-26s %10s %8s %8s %10s %8s\n",
                    "臂", "test MSE", "路由纯度", "区域x专家", "MaxVio", "有效专家");
        for (int a = 0; a < kNumArms; a++) {
            std::vector<double> mses, pur, pairs, vios, effs;
            for (int s = 0; s < g_seeds; s++) {
                ExpResult r = runTask(tasks[ti], kArms[a], (unsigned)(s + 1), 16);
                mses.push_back(r.mse);
                pur.push_back(r.purity);
                pairs.push_back((double)r.regionExpertPairs);
                vios.push_back(r.load.maxVio);
                effs.push_back((double)r.load.effective);
            }
            auto mean = [](const std::vector<double> &v) {
                double s = 0.0;
                for (double x : v) { s += x; }
                return s / (double)v.size();
            };
            std::printf("    %-26s %10.4f %8.3f %8.2f %10.3f %8.2f\n",
                        kArms[a].name, mean(mses), mean(pur), mean(pairs),
                        mean(vios), mean(effs));
        }
        std::printf("\n");
    }
}

/* ============================================================
 *  [3] 真实棋局状态
 * ============================================================ */

/* 收集真实局面 (随机开局 + 随机走子), 用工程自己的编码器 */
void collectStates(std::vector<RL::Tensor> &out, int want, unsigned seed, int D)
{
    out.clear();
    if (D != ChessState::CELLS * 19) {
        std::printf("    [错误] D=%d 与 19x90 编码不符\n", D);
        return;
    }
    std::mt19937 rng(seed);
    std::vector<float> buf((std::size_t)D, 0.0f);

    while ((int)out.size() < want) {
        Chess c;
        c.reset();
        c.sideToMove = Stone::COLOR_RED;
        const int plies = 6 + (int)(rng() % 20);
        for (int p = 0; p < plies && (int)out.size() < want; p++) {
            std::vector<Step *> legal;
            c.sample(c.sideToMove, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                break;
            }
            Step chosen(*legal[rng() % legal.size()]);
            Steps::instance().put(legal);
            double dummy = 0.0;
            c.moveForward(&chosen, dummy);

            std::fill(buf.begin(), buf.end(), 0.0f);
            ChessState::encodeWithContext(c, c.sideToMove, buf.data(), 0, ChessState::CTX_ALL);
            RL::Tensor t((std::size_t)D, 1);
            for (int j = 0; j < D; j++) {
                t[(std::size_t)j] = buf[(std::size_t)j];
            }
            out.push_back(t);
        }
    }
}

double meanNorm(const std::vector<RL::Tensor> &xs)
{
    double s = 0.0;
    for (const RL::Tensor &x : xs) {
        double n = 0.0;
        for (std::size_t j = 0; j < x.size(); j++) {
            n += (double)x[j] * (double)x[j];
        }
        s += std::sqrt(n);
    }
    return s / (double)xs.size();
}

double meanCosine(const std::vector<RL::Tensor> &xs)
{
    std::mt19937 rng(4242);
    double s = 0.0;
    int n = 0;
    for (int k = 0; k < 400; k++) {
        const RL::Tensor &a = xs[rng() % xs.size()];
        const RL::Tensor &b = xs[rng() % xs.size()];
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (std::size_t j = 0; j < a.size(); j++) {
            dot += (double)a[j] * (double)b[j];
            na += (double)a[j] * (double)a[j];
            nb += (double)b[j] * (double)b[j];
        }
        if (na > 0 && nb > 0) {
            s += dot / (std::sqrt(na) * std::sqrt(nb));
            n++;
        }
    }
    return (n > 0) ? s / (double)n : 0.0;
}

/* 在给定状态集上跑前向, 返回负载 */
LoadStat usageOn(RL::ISparseMoE *moe, RL::Net &net, const std::vector<RL::Tensor> &xs)
{
    moe->resetUsage();
    for (const RL::Tensor &x : xs) {
        net.forward(x, true);
    }
    std::vector<long long> usage;
    moe->usageSnapshot(usage);
    return loadOf(usage);
}

void printShares(const char *tag, const LoadStat &s)
{
    std::printf("    %-30s [", tag);
    for (std::size_t i = 0; i < s.share.size(); i++) {
        std::printf("%s%.3f", (i == 0) ? "" : ", ", s.share[i]);
    }
    std::printf("]  MaxVio=%.3f 最小=%.3f 有效=%d\n", s.maxVio, s.minShare, s.effective);
}

void part3()
{
    const int D = ChessState::CELLS * 19;   /* 1710, 与生产骨干同一编码 */
    std::printf("\n[3] 真实棋局状态上的路由 (D=%d = 19x90, 只测路由, 不用 TB 专家/不跑 MCTS)\n", D);

    std::vector<RL::Tensor> real;
    collectStates(real, g_states, 20240901u, D);
    const double nReal = meanNorm(real);
    const double cosReal = meanCosine(real);
    std::printf("    真实状态 %d 个; 平均 |x| = %.3f; 随机对平均余弦 = %.4f (越接近 1 越窄)\n",
                (int)real.size(), nReal, cosReal);

    /* 同尺度的高斯对照 —— 尺度不匹配的话比的是"模长"而不是"分布结构" */
    std::vector<RL::Tensor> gauss;
    {
        std::mt19937 rng(777);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const double target = nReal / std::sqrt((double)D);
        for (int i = 0; i < (int)real.size(); i++) {
            RL::Tensor t((std::size_t)D, 1);
            for (int j = 0; j < D; j++) {
                t[(std::size_t)j] = (float)((double)nd(rng) * target);
            }
            gauss.push_back(t);
        }
    }
    std::printf("    同尺度高斯对照 %d 个; 平均 |x| = %.3f; 随机对平均余弦 = %.4f\n\n",
                (int)gauss.size(), meanNorm(gauss), meanCosine(gauss));

    /* ---- (a) 同一份门控初始化下的偏斜: 真实 vs 高斯 (多种初始化) ---- */
    std::printf("  (a) 同一份门控初始化 (线性+softmax), 两种输入分布的路由\n");
    std::printf("      单次抽样会是噪声 —— 下面扫 %d 种门控初始化, 报分布\n", 15);
    {
        std::vector<double> vr, vg, er, eg;
        for (int t = 0; t < 15; t++) {
            const GateCfg c = {"linear+softmax", GateStructure::Linear, GateActivation::Softmax,
                               0, false, 0.0f, 0.0f};
            auto moe = makeMoe(D, 8, c);
            RL::Net::Layers L;
            L.push_back(moe);
            L.push_back(RL::Layer<RL::Linear>::_(D, 1, true, true));
            RL::Net net(L);
            LoadStat sr = usageOn(moe.get(), net, real);
            LoadStat sg = usageOn(moe.get(), net, gauss);
            vr.push_back(sr.maxVio);
            vg.push_back(sg.maxVio);
            er.push_back((double)sr.effective);
            eg.push_back((double)sg.effective);
        }
        auto stat = [](std::vector<double> v, double &mn, double &mx, double &av) {
            std::sort(v.begin(), v.end());
            mn = v.front();
            mx = v.back();
            av = 0.0;
            for (double x : v) { av += x; }
            av /= (double)v.size();
        };
        double a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3;
        stat(vr, a1, a2, a3);
        stat(vg, b1, b2, b3);
        stat(er, c1, c2, c3);
        stat(eg, d1, d2, d3);
        std::printf("      真实棋局状态: MaxVio 均值 %.3f [%.3f, %.3f]  有效专家均值 %.2f [%.0f, %.0f]\n",
                    a3, a1, a2, c3, c1, c2);
        std::printf("      同尺度高斯  : MaxVio 均值 %.3f [%.3f, %.3f]  有效专家均值 %.2f [%.0f, %.0f]\n",
                    b3, b1, b2, d3, d1, d2);
        std::printf("      => 同一门控、同一输入尺度, 只换分布: MaxVio %.3f vs %.3f\n\n", a3, b3);
    }

    /* ---- (b) 均衡可达性: 把均衡当唯一目标, 各结构最多能做到多均衡 ---- */
    std::printf("\n  (b) 均衡可达性: 只优化门控去最小化 Switch 均衡项 (或跑偏置控制回路)\n");
    std::printf("      口径: 冻结数据(真实状态), 300 步 × 64 样本 × 3 种门控初始化; **只更新门控**\n");
    std::printf("      lossfree 那几档没有任何梯度来源 -> 门控冻结在初始化, 纯靠偏置控制回路\n");
    std::printf("      %-30s %8s %8s %6s %10s\n",
                "配置", "MaxVio", "最小份额", "有效", "门控迁移");

    struct BalArm {
        const char *name;
        GateStructure gs;
        GateActivation ga;
        bool useBias;
        float biasRate;
        float aux;
    };
    const BalArm arms[] = {
        {"linear+softmax aux=0.1",   GateStructure::Linear, GateActivation::Softmax,  false, 0.0f,   0.1f},
        {"linear+softmax aux=1",     GateStructure::Linear, GateActivation::Softmax,  false, 0.0f,   1.0f},
        {"linear+softmax aux=10",    GateStructure::Linear, GateActivation::Softmax,  false, 0.0f,  10.0f},
        {"linear+logistic aux=10",   GateStructure::Linear, GateActivation::Logistic, false, 0.0f,  10.0f},
        {"mlp+softmax aux=10",       GateStructure::Mlp,    GateActivation::Softmax,  false, 0.0f,  10.0f},
        {"mlp+logistic aux=10",      GateStructure::Mlp,    GateActivation::Logistic, false, 0.0f,  10.0f},
        {"linear+softmax lossfree",  GateStructure::Linear, GateActivation::Softmax,  true,  0.01f,  0.0f},
        {"linear+logistic lossfree", GateStructure::Linear, GateActivation::Logistic, true,  0.01f,  0.0f},
        {"mlp+softmax lossfree",     GateStructure::Mlp,    GateActivation::Softmax,  true,  0.01f,  0.0f},
    };

    for (int a = 0; a < (int)(sizeof(arms) / sizeof(arms[0])); a++) {
        const BalArm &A = arms[a];
        const int SEEDS = 3;
        std::vector<double> vios, mins, effs, dws;
        for (int sd = 0; sd < SEEDS; sd++) {
        GateCfg c = {"", A.gs, A.ga, 32, A.useBias, A.biasRate, A.aux};
        auto moe = makeMoe(D, 8, c);
        RL::Net::Layers L;
        L.push_back(moe);
        L.push_back(RL::Layer<RL::Linear>::_(D, 1, true, true));
        RL::Net net(L);

        const int BATCH = 64;
        const int STEPS = 300;
        /* 量"门控到底动了没有": MLP 模式下要用 wg1/wg2, 量 wg 会永远读出 0 */
        std::vector<RL::Tensor> gateBefore;
        gateBefore.push_back(moe->wg);
        if (A.gs == GateStructure::Mlp) {
            gateBefore.push_back(moe->wg1);
            gateBefore.push_back(moe->wg2);
        }
        std::mt19937 rng(31337u + (unsigned)a);
        for (int step = 0; step < STEPS; step++) {
            moe->resetBatchStats();
            for (int b = 0; b < BATCH; b++) {
                net.forward(real[rng() % real.size()], true);
            }
            moe->updateLossFreeBias();
            if (A.aux > 0.0f) {
                moe->addAuxGradient(A.aux);
            } else {
                moe->resetBatchStats();
            }
            /* 只更新门控: 专家从不 backward, 梯度恒 0 -> 权重不动 */
            moe->RMSProp(0.05f, 0.9f, 0.0f, false);
        }
        auto maxDiff = [](const RL::Tensor &a, const RL::Tensor &b) {
            double m = 0.0;
            const std::size_t n = (a.size() < b.size()) ? a.size() : b.size();
            for (std::size_t k = 0; k < n; k++) {
                m = std::fmax(m, std::fabs((double)a[k] - (double)b[k]));
            }
            return m;
        };
        double dW = maxDiff(moe->wg, gateBefore[0]);
        if (A.gs == GateStructure::Mlp) {
            dW = std::fmax(dW, maxDiff(moe->wg1, gateBefore[1]));
            dW = std::fmax(dW, maxDiff(moe->wg2, gateBefore[2]));
        }
        LoadStat s = usageOn(moe.get(), net, real);
        vios.push_back(s.maxVio);
        mins.push_back(s.minShare);
        effs.push_back((double)s.effective);
        dws.push_back(dW);
        } /* seeds */

        auto mean = [](const std::vector<double> &v) {
            double s = 0.0;
            for (double x : v) { s += x; }
            return s / (double)v.size();
        };
        std::printf("      %-30s %8.3f %8.3f %6.2f %10.3e\n",
                    A.name, mean(vios), mean(mins), mean(effs), mean(dws));
    }
}

} /* namespace */

int main(int argc, char *argv[])
{
    int only = 0;
    for (int i = 1; i < argc; i++) {
        const std::string k = argv[i];
        auto val = [&](const char *p) -> std::string {
            const std::string s = k;
            const std::string pre = std::string(p) + "=";
            if (s.rfind(pre, 0) == 0) {
                return s.substr(pre.size());
            }
            return std::string();
        };
        std::string v;
        if (!(v = val("--seeds")).empty())       { g_seeds = std::atoi(v.c_str()); }
        else if (!(v = val("--steps")).empty())  { g_steps = std::atoi(v.c_str()); }
        else if (!(v = val("--states")).empty()) { g_states = std::atoi(v.c_str()); }
        else if (!(v = val("--only")).empty())   { only = std::atoi(v.c_str()); }
        else if (k == "--quick") { g_seeds = 2; g_steps = 250; g_states = 200; }
        else if (k == "--help" || k == "-h") {
            std::printf("用法: bench_gate_moe [--seeds=N] [--steps=N] [--states=N] [--only=1|2|3] [--quick]\n");
            return 0;
        }
    }

    /*
       显式播种: `Random::engine` 默认由线程身份派生 (util.hpp:26-30), 不播种的话
       两次运行的门控初始化不同 —— 实测确实如此 (同一命令两次跑出 [0,0,0,1] 与
       [0.39,0.03,0.42,0.17])。本工程要求读数逐行可复现, 所以这里钉死种子。
    */
    RL::Random::setSeed(20240901u);

    std::printf("=== MoE 门控变体受控实验 ===\n");
    std::printf("seeds=%d steps=%d states=%d  Random::setSeed(20240901) 固定\n",
                g_seeds, g_steps, g_states);

    if (only == 0 || only == 1) { part1(); }
    if (only == 0 || only == 2) { part2(); }
    if (only == 0 || only == 3) { part3(); }
    return 0;
}
