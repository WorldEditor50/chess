/*
   一次性 scratch (第二轮): 把 SAC+AZ 的 **AlphaZero 监督项**单独量出来。

   上一轮的结论 (dbg_az.cpp 的第一版):
     * 监督项**是有效的** —— α=0 的协议下 π(目标) 从 0.023 一路到 0.67 (MLP) / 0.94 (TB);
     * 但轨迹**剧烈震荡** (0.85 -> 0.68 -> 0.79 -> 0.59 -> 0.84 -> 0.00005 -> 0.92),
       因为 `RMSProp` 的 clipGrad 把整张梯度张量归一到单位长度 —— 每步位移**恒等于 lr**,
       与梯度大小无关, 于是"冲过头 -> 弹回来"是常态。
     * 所以"跑固定步数之后看 π"是一条**不可靠**的断言 (本质上是在振荡曲线的哪一点停下)。

   这一轮改成**配对 A/B**: 同权重、同池、同步数, 只改 `hasSearch` (有没有监督项),
   而且比较**轨迹的统计量** (均值/最大值) 而不是终点。控制组 (hasSearch=false) 只有
   软 Q 项 + 熵项 —— 如果它自己也会把 π(目标) 抬起来, 那"π 上升"就不是监督项的证据。

   编译: .r1build/dbg_az.bat
*/
#include <cstdio>
#include <cmath>
#include <vector>
#include "chess.h"
#include "sacazagent.h"
#include "rl/util.hpp"

struct Stats {
    double first = 0.0;   /* 更新前的 π(目标) */
    double mean = 0.0;    /* 轨迹均值 */
    double peak = 0.0;    /* 轨迹最大值 */
    double last = 0.0;    /* 轨迹终点 */
    double uniform = 0.0;
};

static void fillPool(SACAZAgent &agent,
                     const std::vector<std::uint16_t> &cells,
                     std::uint64_t lo, std::uint64_t hi,
                     int legalCount, int action, int target, bool hasSearch)
{
    agent.memories.clear();
    for (int i = 0; i < 48; i++) {
        SACAZAgent::Transition tr;
        tr.cells = cells;
        tr.nextCells = cells;
        tr.curMaskLo = lo;
        tr.curMaskHi = hi;
        tr.nextMaskLo = lo;
        tr.nextMaskHi = hi;
        tr.action = action;
        tr.legalCount = legalCount;
        tr.reward = 0.0f;
        tr.done = true;
        tr.hasSearch = hasSearch;
        if (hasSearch) { tr.pi[target] = 1.0f; }
        agent.memories.push_back(tr);
    }
}

/* 一个 agent 在给定池上跑 iters 次 learnBatch(4), 记录 π(目标) 的轨迹 */
static Stats runTraj(SACAZAgent &agent, const RL::Tensor &state, const RL::Tensor &mask,
                     int target, int iters)
{
    RL::Tensor pi(SACAZAgent::ACTION_DIM, 1);
    Stats s;
    agent.policy(state, mask, pi);
    s.first = pi[target];

    double sum = 0.0;
    s.peak = s.first;
    for (int it = 0; it < iters; it++) {
        agent.learnBatch(4);
        agent.policy(state, mask, pi);
        const double p = pi[target];
        sum += p;
        s.peak = std::fmax(s.peak, p);
        s.last = p;
    }
    s.mean = (iters > 0) ? sum / (double)iters : s.first;
    return s;
}

static void experiment(const char *tag, SACAZAgent::Backbone bb, int iters)
{
    Chess c;
    c.reset();
    SACAZAgent az(c, 64, 0.99f, 0.003f, 1.5f, bb, 64, 0.1f);
    az.batchSize = 4;

    /* 局面与掩码 */
    std::vector<Step*> legal;
    std::vector<int> idx;
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    az.getLegalActions(Stone::COLOR_RED, legal, idx, mask);
    Steps::instance().put(legal);
    std::vector<std::uint16_t> cells;
    az.encodeSparse(Stone::COLOR_RED, cells);
    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    SACAZAgent::expandSparse(cells, state);
    std::uint64_t lo = 0, hi = 0;
    SACAZAgent::maskToBits(mask, lo, hi);
    const int target = idx[idx.size() / 2];
    const int action = idx[0];

    /* 两个 agent 必须**同权重**: 五个网络全部 copyTo (Net 的拷贝是浅拷贝) */
    SACAZAgent ctrl(c, 64, 0.99f, 0.003f, 1.5f, bb, 64, 0.1f);
    ctrl.batchSize = 4;
    az.actor.copyTo(ctrl.actor);
    az.q1.copyTo(ctrl.q1);
    az.q2.copyTo(ctrl.q2);
    az.q1.copyTo(ctrl.q1Target);
    az.q2.copyTo(ctrl.q2Target);

    /* 池: A 组带监督项 (π_MCTS = one-hot(target)), B 组不带 —— 只差这一个开关 */
    fillPool(az, cells, lo, hi, (int)idx.size(), action, target, true);
    fillPool(ctrl, cells, lo, hi, (int)idx.size(), action, target, false);

    const Stats a = runTraj(az, state, mask, target, iters);
    const Stats b = runTraj(ctrl, state, mask, target, iters);

    std::printf("  %-20s 均匀=%.5f\n", tag, 1.0 / (double)idx.size());
    std::printf("    带监督项 (hasSearch=1): 起 %.5f  均值 %.5f  峰值 %.5f  终点 %.5f\n",
                a.first, a.mean, a.peak, a.last);
    std::printf("    对照组   (hasSearch=0): 起 %.5f  均值 %.5f  峰值 %.5f  终点 %.5f\n",
                b.first, b.mean, b.peak, b.last);
    std::printf("    差值: 均值 %+.5f  峰值 %+.5f  终点 %+.5f\n\n",
                a.mean - b.mean, a.peak - b.peak, a.last - b.last);
}

int main()
{
    RL::Random::setSeed(20240913);
    const int iters = 20;
    std::printf("=== SAC+AZ: 监督项的配对 A/B (同权重/同池/同步数, 只改 hasSearch) ===\n");
    std::printf("每 %d 次 learnBatch(4) 记录一次 π(目标)\n\n", iters);
    experiment("MLP", SACAZAgent::Backbone::Mlp, iters);
    experiment("稀疏MoE(TB专家)", SACAZAgent::Backbone::SparseMoeTb, iters);
    return 0;
}
