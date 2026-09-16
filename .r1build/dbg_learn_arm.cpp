/*
 * dbg_learn_arm.cpp — [5] 配对符号实验的诊断脚本 (临时, 见 .r1build 的约定)
 *
 * 疑问: 目标 −0.5 那一臂的批 loss 收敛到 0.0069 (说明训练路径上 Q ≈ −0.5),
 *       但 evaluateNode 探针测到的同一个动作的 Q 却是 +0.675。
 *       两者用的是同一套在线网络、同一个状态、同一个合法集 —— 必须查清楚是哪一环对不上。
 *
 * 这里每 5 次更新打一行: 批 loss / 探针 Q(taken) / 训练路径上的 Q(taken) / V。
 * "训练路径上的 Q" 用 accumulateGrad 同样的算法手算 (V + A_a − mean_legal(A))。
 */
#include <cmath>
#include <cstdio>

#include "chess.h"
#include "dqnabagent.h"
#include "rl/util.hpp"
#include "stone.h"

static void probe(DQNABAgent &a, Chess &c, const std::vector<int> &idx, int taken,
                  double &vOut, double &qProbe, double &qTrain)
{
    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    a.encodeStateFor(Stone::COLOR_RED, st);
    double v = 0.0;
    std::vector<double> q;
    a.evaluateNode(st, idx, v, q);
    vOut = v;
    qProbe = 0.0;
    for (std::size_t i = 0; i < idx.size(); i++) {
        if (idx[i] == taken) { qProbe = q[i]; }
    }
    /* 训练路径: 主干一次 -> V 头 + A 头的稀疏列 */
    RL::Net &trunk = a.m_trunk;
    RL::Tensor &h = trunk.forward(st, false);
    RL::Tensor &vt = a.m_vHead.forward(h, false);
    std::vector<float> aLegal;
    RL::Tensor &ht = a.m_aHead.forwardTrunk(h, false);
    if (!a.m_aHead.sparseLogits(ht, idx, aLegal)) {
        qTrain = -999.0;
        return;
    }
    double meanA = 0.0, av = 0.0;
    for (std::size_t i = 0; i < aLegal.size(); i++) { meanA += aLegal[i]; }
    meanA /= (double)aLegal.size();
    for (std::size_t i = 0; i < idx.size(); i++) {
        if (idx[i] == taken) { av = aLegal[i]; }
    }
    qTrain = (double)vt[0] + av - meanA;
}

int main()
{
    /* 扫学习率: 上一版诊断显示 lr=0.02 + "同一个样本重复 64 遍"会进入**周期 2 极限环**
       (loss 在 0.006 与 1.31 之间跳、Q 在 −1.8 与 +2.4 之间摆), 于是任何"看某个时刻的
       Q/loss"的断言都是无效的。这里量的是**轨迹统计量**: 后 50 次更新的 Q 均值/最小/最大,
       以及 loss "变大"的比例 (周期 2 的环 -> 约 50%)。*/
    const double lrs[3] = { 0.001, 0.005, 0.02 };
    for (int li = 0; li < 3; li++) {
        for (int arm = 0; arm < 2; arm++) {
            const float label = (arm == 0) ? 0.5f : -0.5f;
            Chess c;
            c.reset();
            RL::Random::setSeed(20240915u);
            DQNABAgent a(c, 32, 0.99f, (float)lrs[li], DQNABAgent::Backbone::Mlp);
            a.batchSize = 8;
            a.targetSyncEvery = 8;

            std::vector<Step*> legal;
            std::vector<int> idx;
            a.legalMoves(Stone::COLOR_RED, legal, idx);
            Steps::instance().put(legal);
            const int taken = idx[0];
            for (int i = 0; i < 64; i++) {
                DQNABAgent::Sample s;
                s.state = RL::Tensor(DQNABAgent::STATE_DIM, 1);
                a.encodeStateFor(Stone::COLOR_RED, s.state);
                s.nextState = s.state;
                s.legalIdx = idx;
                s.nextLegalIdx = idx;
                s.action = taken;
                s.reward = label;
                s.done = true;
                s.label = label;
                a.pushSample(std::move(s));
            }

            double qSum = 0.0, qMin = 1e9, qMax = -1e9, lossSum = 0.0;
            int n = 0, lossUp = 0, extraUpdates = 0;
            float prevLoss = 0.0f;
            const int total = 200;
            for (int it = 0; it < total; it++) {
                if (it % 5 == 0) {
                    double v = 0.0, qp = 0.0, qt = 0.0;
                    probe(a, c, idx, taken, v, qp, qt);
                    if (it >= total - 50) {
                        qSum += qp;
                        qMin = std::min(qMin, qp);
                        qMax = std::max(qMax, qp);
                        n++;
                    }
                    if (it >= total - 50) { extraUpdates++; }
                }
                a.learnBatch(8, 1);
                const float l = a.getLastTrainLoss();
                if (it >= total - 50) {
                    lossSum += l;
                    if (l > prevLoss) { lossUp++; }
                    prevLoss = l;
                }
            }
            double vEnd = 0.0, qEnd = 0.0, qt = 0.0;
            probe(a, c, idx, taken, vEnd, qEnd, qt);
            std::printf("lr=%.3f 目标 %+.1f: 后 50 次采样 %d 个点, Q 均值 %7.4f "
                        "(min %7.4f / max %7.4f), loss 均值 %7.4f, loss 变大 %d/%d\n",
                        lrs[li], (double)label, n, n > 0 ? qSum / n : 0.0, qMin, qMax,
                        extraUpdates > 1 ? lossSum / (extraUpdates - 1) : 0.0,
                        lossUp, extraUpdates > 1 ? extraUpdates - 1 : 1);
        }
        std::printf("\n");
    }
    return 0;
}
