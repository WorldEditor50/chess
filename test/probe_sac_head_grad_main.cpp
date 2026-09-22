/*
 * probe_sac_head_grad_main.cpp - 策略头梯度"只落合法列"的不变量探针
 * ============================================================================
 *
 * 为什么需要它: `test_sacaz` 的 [11] 节断言"SAC 训练后非法动作的头部权重**逐位不变**",
 * 而在 2026-09 把动作空间从 128 槽哈希换成 8100 双射、并给 PPO 侧加上信任域/熵项之后,
 * 这条断言失败了 (非法列改动 270656/515584)。要在"改断言"和"修实现"之间做选择,
 * 必须先知道**是哪一项改动造成的**: 熵奖励? 裁剪? 还是纯粹的动作空间变大?
 *
 * 本程序把 [11] 节的构造抽出来, 只改一个变量反复跑, 直接给出"非法列/合法列各改了多少":
 *   --entropy=0/1   熵奖励项开关 (accumulateGradSparse 的 entropyCoef)
 *   --clip=0/1      信任域开关 (clipEps)
 *   --sac           跑 SACAZAgent 而不是 RL::PPO (两条路径都要看)
 *   --reps=N        训练批数
 *   --state-dim=N   覆盖状态维度 (默认用 agent 自己的)
 *
 * 判据: **非法列必须逐位不变** —— 掩码 softmax 让非法列 π ≡ 0, 雅可比给出 dz ≡ 0,
 * 所以那几行权重收到的是**恰好** 0 的梯度 (这是 test_sacaz 钉住的契约, 不是巧合)。
 * 如果它变了, 一定有一路梯度绕过了掩码。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "chess.h"
#include "chessstate.h"
#include "sacazagent.h"
#include "ppomcts_agent.h"
#include "rl/ppo.h"
#include "rl/util.hpp"

namespace {

struct Cfg {
    int reps = 10;
    int batch = 4;
    bool entropy = true;
    bool clip = true;
    bool useSac = false;
    bool clamp = true;
    bool huber = true;
    int samples = 64;
    unsigned seed = 20240901;
};

Cfg g;

/* 数一份权重快照里"逐位不同"的个数 */
static int diffCount(const std::vector<float> &a, const std::vector<float> &b)
{
    const std::size_t n = std::min(a.size(), b.size());
    int c = 0;
    for (std::size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) { c++; }
    }
    return c;
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *k) -> const char* {
            const std::size_t n = std::strlen(k);
            if (std::strncmp(a, k, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--reps"))    { g.reps = std::atoi(v); }
        else if (const char *v = val("--batch")) { g.batch = std::atoi(v); }
        else if (const char *v = val("--entropy")) { g.entropy = std::atoi(v) != 0; }
        else if (const char *v = val("--clip"))    { g.clip = std::atoi(v) != 0; }
        else if (std::strcmp(a, "--sac") == 0)     { g.useSac = true; }
        else if (std::strcmp(a, "--no-clamp") == 0) { g.clamp = false; }
        else if (std::strcmp(a, "--no-huber") == 0) { g.huber = false; }
        else if (const char *v = val("--samples")) { g.samples = std::atoi(v); }
        else if (const char *v = val("--seed"))    { g.seed = (unsigned)std::atoi(v); }
    }
    RL::Random::setSeed(g.seed);

    Chess chess;
    chess.reset();

    /* ---- 找出"哪些动作非法" ---- */
    std::vector<Step*> legal;
    std::vector<int> legalIdx;
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);

    if (g.useSac) {
        SACAZAgent agent(chess, 64, 0.99f, 0.001f, 1.5f);
        agent.batchSize = g.batch;
        /* 让"值域约束/Huber"可关 —— 用来判断权重变化是不是它们引入的 */
        if (!g.clamp)   { agent.clampTarget = -1.0f; }
        if (!g.huber)   { agent.huberDelta = -1.0f; }
        std::printf("[SAC ] 开关: clampTarget=%.2f huberDelta=%.2f\n",
                    (double)agent.clampTarget, (double)agent.huberDelta);
        agent.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
        Steps::instance().put(legal);

        std::vector<std::uint16_t> cells;
        agent.encodeSparse(Stone::COLOR_RED, cells);
        std::uint64_t bits[2] = { 0, 0 };
        SACAZAgent::maskToBits(mask, bits);
        for (int i = 0; i < g.samples; i++) {
            SACAZAgent::Transition tr;
            tr.cells = cells;
            tr.nextCells = cells;
            tr.curMask[0] = bits[0];  tr.curMask[1] = bits[1];
            tr.nextMask[0] = bits[0]; tr.nextMask[1] = bits[1];
            tr.action = legalIdx[(std::size_t)(i % (int)legalIdx.size())];
            tr.legalCount = (int)legalIdx.size();
            tr.reward = 0.05f;
            tr.done = false;
            tr.hasSearch = false;
            agent.memories.push_back(tr);
        }

        RL::iFcLayer *head = dynamic_cast<RL::iFcLayer*>(agent.actor[agent.actor.size() - 1]);
        if (head == nullptr) { std::printf("(策略头不是 iFcLayer, 退出)\n"); return 1; }
        const std::size_t rowStride = (std::size_t)head->w.sizes[0];
        std::printf("[SAC ] head: w=(%d,%d) rowStride=%zu bias=%d | actor 参数=%lld | "
                    "state=%d action=%d | 合法=%d\n",
                    (int)head->w.sizes[0], (int)head->w.sizes[1], rowStride,
                    (int)head->bias, agent.actor.paramCount(),
                    SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM, (int)legalIdx.size());
        if (head->w.size() < (std::size_t)SACAZAgent::ACTION_DIM * rowStride) {
            std::printf("[SAC ] **策略头行数不足**: w.size=%zu < %d x %zu\n",
                        head->w.size(), SACAZAgent::ACTION_DIM, rowStride);
        }

        std::vector<int> illegal;
        for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
            if (mask[a] <= 0.5f) { illegal.push_back(a); }
        }
        auto snap = [&](const std::vector<int> &acts) {
            std::vector<float> buf;
            for (std::size_t i = 0; i < acts.size(); i++) {
                const float *row = head->w.val.data() + (std::size_t)acts[i] * rowStride;
                for (std::size_t k = 0; k < rowStride; k++) { buf.push_back(row[k]); }
            }
            return buf;
        };
        /*
           先量**梯度缓冲** (这是判据的核心, 与优化器无关):
           `Layer::backward` 把 `g.w += e·xᵀ` 累加进去 (见 tensor.hpp ikjk 的说明), 所以
           跑一次 learnBatch 之后 `g.w` 里就是这一批的梯度。掩码 softmax 的契约是
           "非法列 dz ≡ 0" -> 非法行的 g.w 必须**恰好**为 0。
        */
        const std::size_t probeRows[4] = { 0, 65, 4000, 8099 };
        std::printf("  掩码: 合法=%d 非法=%d | 逐行读数 (g.sum = Σ|dL/dw|):\n",
                    (int)legalIdx.size(), (int)illegal.size());
        for (int r = 0; r < 4; r++) {
            const std::size_t row = probeRows[r];
            std::printf("      row %4zu mask=%.0f\n", row, (double)mask[row]);
        }

        auto gradSum = [&](const std::vector<int> &acts) {
            double s = 0.0;
            for (std::size_t i = 0; i < acts.size(); i++) {
                const float *grow = head->g.w.val.data() + (std::size_t)acts[i] * rowStride;
                for (std::size_t k = 0; k < rowStride; k++) {
                    s += std::fabs((double)grow[k]);
                }
            }
            return s;
        };

        const std::vector<float> iBefore = snap(illegal);
        const std::vector<float> lBefore = snap(legalIdx);
        agent.learnBatch(g.batch);          /* 一批: 只看这一批的梯度与权重变化 */
        const double gIllegal = gradSum(illegal);
        const double gLegal = gradSum(legalIdx);
        const std::vector<float> iAfter = snap(illegal);
        const std::vector<float> lAfter = snap(legalIdx);
        const int iDiff = diffCount(iBefore, iAfter);
        const int lDiff = diffCount(lBefore, lAfter);

        /* 逐行分类: 只统计**最大**权重变化, 以及被改动的行里最小/最大的动作下标 */
        auto maxDelta = [](const std::vector<float> &a, const std::vector<float> &b) {
            double m = 0.0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); i++) {
                m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
            }
            return m;
        };
        int firstChangedRow = -1, lastChangedRow = -1, changedRows = 0;
        for (std::size_t i = 0; i < illegal.size(); i++) {
            bool changed = false;
            for (std::size_t k = 0; k < rowStride; k++) {
                const std::size_t idx = i * rowStride + k;
                if (iBefore[idx] != iAfter[idx]) { changed = true; break; }
            }
            if (changed) {
                changedRows++;
                if (firstChangedRow < 0) { firstChangedRow = illegal[i]; }
                lastChangedRow = illegal[i];
            }
        }
        std::printf("  权重最大变化: 非法列=%.3e 合法列=%.3e\n",
                    maxDelta(iBefore, iAfter), maxDelta(lBefore, lAfter));
        std::printf("  被改动的非法**行**数=%d (首个下标 %d, 末个下标 %d)\n",
                    changedRows, firstChangedRow, lastChangedRow);
        /* 头几行合法的动作下标 (用来判断"是不是别名/镜像写错") */
        std::printf("  合法动作下标样本: ");
        for (int i = 0; i < 8 && i < (int)legalIdx.size(); i++) {
            std::printf("%d ", legalIdx[(std::size_t)i]);
        }
        std::printf("...\n");

        /*
           ---- 隔离实验: 只做一次 forward+backward, **不调优化器** ----
           如果这一步里非法行权重也变了, 说明变化来自反向传播本身 (与 RMSProp 无关);
           如果只有调了优化器才变, 那就是"零梯度 × 每层范数归一"那条路径的问题。
        */
        {
            SACAZAgent g2(chess, 64, 0.99f, 0.001f, 1.5f);
            RL::iFcLayer *h2 = dynamic_cast<RL::iFcLayer*>(g2.actor[g2.actor.size() - 1]);
            if (h2 != nullptr) {
                const std::size_t rs2 = (std::size_t)h2->w.sizes[0];
                std::vector<float> before;
                for (std::size_t i = 0; i < illegal.size(); i++) {
                    const float *row = h2->w.val.data() + (std::size_t)illegal[i] * rs2;
                    for (std::size_t k = 0; k < rs2; k++) { before.push_back(row[k]); }
                }
                /* 手工走一遍 learnBatch 的前向/反向, 但**不调用** RMSProp */
                RL::Tensor st(SACAZAgent::STATE_DIM, 1);
                RL::Tensor mk(SACAZAgent::ACTION_DIM, 1);
                RL::Tensor piT(SACAZAgent::ACTION_DIM, 1);
                RL::Tensor q1T(SACAZAgent::ACTION_DIM, 1);
                RL::Tensor q2T(SACAZAgent::ACTION_DIM, 1);
                g2.encodeStateFor(Stone::COLOR_RED, st);
                g2.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mk);
                Steps::instance().put(legal);
                g2.policy(st, mk, piT);
                g2.qValues(st, q1T, q2T);
                /* 与 learnBatch 同一套梯度 (策略项 + 掩码 softmax 雅可比) */
                RL::Tensor gg(SACAZAgent::ACTION_DIM, 1);
                RL::Tensor dz(SACAZAgent::ACTION_DIM, 1);
                for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                    gg[i] = (mk[i] > 0.5f)
                                ? (0.2f * (std::log(piT[i] + 1e-8f) + 1.0f)
                                   - std::min(q1T[i], q2T[i]))
                                : 0.0f;
                }
                SACAZAgent::maskedSoftmaxBackward(piT, gg, dz);
                g2.actor.backward(st, dz);
                std::vector<float> after;
                for (std::size_t i = 0; i < illegal.size(); i++) {
                    const float *row = h2->w.val.data() + (std::size_t)illegal[i] * rs2;
                    for (std::size_t k = 0; k < rs2; k++) { after.push_back(row[k]); }
                }
                std::printf("  隔离实验 (无优化器): 非法行权重改动 %d/%d\n",
                            diffCount(before, after), (int)before.size());
            }
        }

        std::printf("  单批之后: 梯度 Σ|g| 非法列=%.6g 合法列=%.6g\n", gIllegal, gLegal);
        std::printf("[SAC ] 合法=%d 非法=%d | 非法列改动 %d/%d, 合法列改动 %d/%d | "
                    "critic |Q|max 目标=%.4f TD误差=%.4f | %s\n",
                    (int)legalIdx.size(), (int)illegal.size(),
                    iDiff, (int)(illegal.size() * rowStride),
                    lDiff, (int)(legalIdx.size() * rowStride),
                    agent.getMaxAbsTarget(), agent.getMaxAbsTdErr(),
                    (gIllegal == 0.0 && iDiff == 0) ? "不变量成立"
                                                    : "**非法列被改了**");
        return (gIllegal == 0.0 && iDiff == 0) ? 0 : 1;
    }

    /* ---- RL::PPO 这一路 (R2 稀疏头 + 信任域) ---- */
    RL::PPO ppo(PPOMCTSAgent::STATE_DIM, 64, PPOMCTSAgent::ACTION_DIM, 64, 0.1f,
                true /* withGrad */, RL::PPO::Backbone::MlpExperts);
    ppo.clipEps = g.clip ? 0.2f : 0.0f;
    ppo.entropyCoef = g.entropy ? 0.01f : 0.0f;

    PPOMCTSAgent probe(chess, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true,
                       RL::PPO::Backbone::MlpExperts);
    probe.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
    Steps::instance().put(legal);

    RL::Tensor state(PPOMCTSAgent::STATE_DIM, 1);
    probe.encodeStateFor(Stone::COLOR_RED, state);

    /* 稀疏目标: 只落在少数几个合法动作上 (模拟"访问分布") */
    std::vector<int> tIdx;
    std::vector<float> tProb;
    for (int k = 0; k < 3 && k < (int)legalIdx.size(); k++) {
        tIdx.push_back(legalIdx[(std::size_t)k * 7 % legalIdx.size()]);
        tProb.push_back(0.5f - 0.15f * (float)k);
    }
    std::vector<int> illegal;
    for (int a = 0; a < PPOMCTSAgent::ACTION_DIM; a++) {
        if (mask[a] <= 0.5f) { illegal.push_back(a); }
    }

    RL::iFcLayer *head = dynamic_cast<RL::iFcLayer*>(ppo.actorP[ppo.actorP.size() - 1]);
    if (head == nullptr) { std::printf("(策略头不是 iFcLayer, 退出)\n"); return 1; }
    const std::size_t rowStride = (std::size_t)head->w.sizes[0];
    auto snap = [&](const std::vector<int> &acts) {
        std::vector<float> buf;
        for (std::size_t i = 0; i < acts.size(); i++) {
            const float *row = head->w.val.data() + (std::size_t)acts[i] * rowStride;
            for (std::size_t k = 0; k < rowStride; k++) { buf.push_back(row[k]); }
        }
        return buf;
    };
    const std::vector<float> iBefore = snap(illegal);
    const std::vector<float> lBefore = snap(legalIdx);
    std::printf("  快照哨兵: illegal[0..3]=%.6g %.6g %.6g %.6g | legal[0..3]=%.6g %.6g %.6g %.6g\n",
                (double)iBefore[0], (double)iBefore[1], (double)iBefore[2], (double)iBefore[3],
                (double)lBefore[0], (double)lBefore[1], (double)lBefore[2], (double)lBefore[3]);

    /* 没有信任域样本 (oldProb 省略) 时也应当成立 */
    for (int i = 0; i < g.samples; i++) {
        ppo.addReplay(state, tIdx, tProb, 0.05f, legalIdx);
    }
    for (int r = 0; r < g.reps; r++) {
        ppo.learnFromReplay((std::size_t)g.batch, 1, 0.001f);
        /* 诊断: 每步之后把"非法行是否真的没被碰"报一次 (权重衰减会碰每一行,
           所以读数应当是"每步一个恒定的小量"而不是"逐步累积的梯度") */
    }
    const int iDiff = diffCount(iBefore, snap(illegal));
    const int lDiff = diffCount(lBefore, snap(legalIdx));

    std::printf("[PPO ] 合法=%d 非法=%d | 非法列改动 %d/%d, 合法列改动 %d/%d | "
                "熵项=%d 裁剪=%d | %s\n",
                (int)legalIdx.size(), (int)illegal.size(),
                iDiff, (int)(illegal.size() * rowStride),
                lDiff, (int)(legalIdx.size() * rowStride),
                (int)g.entropy, (int)g.clip,
                iDiff == 0 ? "不变量成立" : "**非法列被改了**");
    return iDiff == 0 ? 0 : 1;
}
