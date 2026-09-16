/*
 * test_sacaz_main.cpp - SAC + MCTS + AlphaZero agent 的验证
 *
 * 这个 agent 里真正容易出错的是五件事, 测试就盯这五件:
 *
 *   1. **掩码 softmax 的雅可比**。策略头输出 logits, 掩码归一化在 agent 内自己做,
 *      反向必须是 dL/dz_c = π_c·(g_c − Σ_a g_a π_a)。这是全篇最容易写错的一处,
 *      所以用**有限差分**核对 (对 L = Σ g_a π_a 求 dL/dz)。
 *   2. **搜索与走法合法性**: selectMove 必须给出合法走法; MCTS 的展开数、
 *      访问分布要能对上。
 *   3. **软价值**: α = 0 时必须退化成 Σπ·min(Q1,Q2), α > 0 时必须减去熵项。
 *   4. **critic 真的在学习**: 给一批固定经验 (reward = +0.5 且 done = true, 于是
 *      TD 目标恰好是 0.5), 跑若干次 learnBatch, 断言 Q(s,a) 向 0.5 靠近 ——
 *      这一条能把"前向对但反向错 / 优化器用错 / 符号搞反"这类问题挡住。
 *   5. **探索对棋盘零副作用** (项目里那条硬性不变量, 见 issues_review C5)。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include "chess.h"
#include "chessstate.h"
#include "sacazagent.h"
#include "rl/cpuinfo.hpp"
#include "rl/moe.hpp"
#include "rl/transformer.hpp"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

/* 棋盘摘要: 用来判断"探索前后棋盘是否完全一致" */
static std::string digest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[i];
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x) + "," + std::to_string(st->pos.y) + ";";
    }
    s += "side=" + std::to_string(c.sideToMove);
    s += "clock=" + std::to_string(c.halfMoveClock);
    s += "hist=" + std::to_string((int)c.history.size());
    s += "hash=" + std::to_string(c.computeHash());
    return s;
}

/* 造一个"合法动作掩码"位图 (用于合成经验) */
static void setMaskBits(SACAZAgent &agent, RL::Tensor &mask, const std::vector<int> &acts)
{
    (void)agent;
    mask.zero();
    for (int a : acts) {
        mask[a] = 1.0f;
    }
}

/* ============================================================
 *  1. 掩码 softmax 与它的反向 (有限差分)
 * ============================================================ */
static void testMaskedSoftmax()
{
    std::printf("\n[1] 掩码 softmax 与它的反向 (有限差分核对)\n");

    const int D = SACAZAgent::ACTION_DIM;
    RL::Tensor logits(D, 1);
    RL::Tensor mask(D, 1);
    logits.zero();
    mask.zero();
    const int legal[5] = {3, 17, 42, 99, 120};
    for (int i = 0; i < 5; i++) {
        mask[legal[i]] = 1.0f;
        logits[legal[i]] = 0.7f * (float)(i - 2) + 0.13f;
    }
    /* 非法动作的 logit 故意给极端值: 掩码必须把它彻底压掉 */
    logits[5] = 50.0f;
    logits[60] = -50.0f;

    RL::Tensor pi(D, 1);
    SACAZAgent::maskedSoftmax(logits, mask, pi);

    float sum = 0.0f;
    bool allPos = true;
    for (int i = 0; i < D; i++) {
        sum += pi[i];
    }
    for (int i = 0; i < 5; i++) {
        if (pi[legal[i]] <= 0.0f) {
            allPos = false;
        }
    }
    CHECK(std::fabs(sum - 1.0f) < 1e-5f, "策略在合法动作上归一化到 1");
    CHECK(pi[5] == 0.0f && pi[60] == 0.0f, "非法动作概率严格为 0 (logit 再极端也一样)");
    CHECK(allPos, "所有合法动作都有正概率");

    /* 反向: 随机 g, 中心差分核对 dL/dz  (L = Σ_a g_a·π_a) */
    RL::Tensor g(D, 1);
    for (int i = 0; i < D; i++) {
        g[i] = 0.3f * std::sin((float)i * 1.7f);
    }
    RL::Tensor dz(D, 1);
    SACAZAgent::maskedSoftmaxBackward(pi, g, dz);

    const float eps = 1e-3f;
    float worst = 0.0f;
    int worstIdx = -1;
    RL::Tensor lp(D, 1);
    RL::Tensor lm(D, 1);
    for (int c = 0; c < D; c++) {
        const float save = logits[c];
        logits[c] = save + eps;
        SACAZAgent::maskedSoftmax(logits, mask, lp);
        logits[c] = save - eps;
        SACAZAgent::maskedSoftmax(logits, mask, lm);
        logits[c] = save;

        double Lp = 0.0, Lm = 0.0;
        for (int i = 0; i < D; i++) {
            Lp += (double)g[i] * (double)lp[i];
            Lm += (double)g[i] * (double)lm[i];
        }
        const double num = (Lp - Lm) / (2.0 * (double)eps);
        const double ana = (double)dz[c];
        const double rel = std::fabs(num - ana)
                           / std::fmax(1e-6, std::fmax(std::fabs(num), std::fabs(ana)));
        if (rel > worst) {
            worst = (float)rel;
            worstIdx = c;
        }
    }
    std::printf("    雅可比最大相对误差 = %.3e (最差 z[%d])\n", (double)worst, worstIdx);
    CHECK(worst < 5e-3f, "掩码 softmax 的反向与数值梯度一致");
    CHECK(dz[5] == 0.0f && dz[60] == 0.0f, "非法动作的梯度严格为 0");
}

/* ============================================================
 *  2. 决策: 合法走法 / MCTS 展开 / 访问分布
 * ============================================================ */
static void testSelectMove()
{
    std::printf("\n[2] selectMove: 合法走法 / MCTS 展开 / 访问分布\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);

    std::vector<Step*> legal;
    c.sample(Stone::COLOR_RED, legal);
    const std::size_t legalCount = legal.size();
    Steps::instance().put(legal);
    std::printf("    初始局面红方合法走法 = %zu\n", legalCount);
    CHECK(legalCount > 0, "初始局面有合法走法");

    const std::string before = digest(c);
    RL::Tensor piTarget(SACAZAgent::ACTION_DIM, 1);
    const Step s = agent.selectMove(Stone::COLOR_RED, 48, 0.0f, &piTarget);

    CHECK(s.valid, "selectMove 返回的走法不为空");
    CHECK(c.isLegalMove(Stone::COLOR_RED, &s), "返回的走法是合法走法");
    CHECK(digest(c) == before, "搜索结束后棋盘逐字段复原");

    /* 访问分布: 归一化, 且只落在根的子节点上 */
    float psum = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
        psum += piTarget[i];
        if (piTarget[i] > 0.0f) {
            nonzero++;
        }
    }
    std::printf("    访问分布: sum=%.4f, 非零 %d 个 (合法 %zu)\n",
                (double)psum, nonzero, legalCount);
    CHECK(std::fabs(psum - 1.0f) < 1e-4f, "访问分布归一化到 1");
    CHECK(nonzero > 0, "访问分布非空");
    /* 48 次模拟至少能展开出一些子节点 (合法走法约 40 出头) */
    CHECK(nonzero >= 8, "MCTS 展开出了足够多的子节点");

    /* 搜索用的叶子估值计数应该 > 0 (说明真的在评估, 没有全走终局分支) */
    std::printf("    叶子估值次数 = %lld\n", agent.getLeafEvals());
    CHECK(agent.getLeafEvals() > 0, "叶子用软价值评估过");

    /* 单步耗时 (信息, 不断言): 决定它在界面上好不好用 */
    RL::Tensor pi2(SACAZAgent::ACTION_DIM, 1);
    const clock_t t0 = clock();
    const int reps = 8;
    for (int i = 0; i < reps; i++) {
        agent.selectMove(Stone::COLOR_RED, 64, 0.0f, &pi2);
    }
    const double ms = 1000.0 * (double)(clock() - t0) / (double)CLOCKS_PER_SEC / reps;
    std::printf("    64 次模拟的单步决策耗时 ≈ %.0f ms\n", ms);
}

/* ============================================================
 *  3. 软价值: α 的作用
 * ============================================================ */
static void testSoftValue()
{
    std::printf("\n[3] 软价值 V(s) = Σ π·(min Q − α·log π)\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);

    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    agent.encodeStateFor(Stone::COLOR_RED, state);

    /*
       ---- 完备 Markov 状态 (本轮从 DQNAB 推广) ----
       象棋是双人零和、完全信息、交替行动的 Markov Game, 而**裸棋盘 + 轮到谁不是
       Markov 状态**: 三次重复判和、60 回合无吃子判和都依赖历史。
       下面的断言直接量"状态里有没有规则上下文": 走 4 手可逆循环回到**同一个局面**
       (棋子平面逐位相同), 规则上下文那 3 个槽必须变。
    */
    {
        CHECK(SACAZAgent::STATE_DIM == SACAZAgent::CTX_BASE + 3,
              "STATE_DIM = 14 个平面 + 3 个规则上下文槽");
        const float ctx0 = state[(std::size_t)SACAZAgent::CTX_BASE + 1];   /* 重复次数 */
        std::vector<float> board0((std::size_t)SACAZAgent::CTX_BASE);
        for (std::size_t i = 0; i < board0.size(); i++) { board0[i] = state[i]; }

        /* 红马/黑马各出去再回来: 4 手回到起始局面 */
        const int from[4] = { 9 * 9 + 1, 0 * 9 + 1, 7 * 9 + 2, 2 * 9 + 2 };
        const int to[4]   = { 7 * 9 + 2, 2 * 9 + 2, 9 * 9 + 1, 0 * 9 + 1 };
        int played = 0;
        for (int k = 0; k < 4; k++) {
            const int turn = c.sideToMove;
            std::vector<Step*> legal;
            c.sample(turn, legal);
            Step *mv = nullptr;
            for (std::size_t i = 0; i < legal.size(); i++) {
                if (legal[i]->pos.x == from[k] / 9 && legal[i]->pos.y == from[k] % 9 &&
                    legal[i]->nextPos.x == to[k] / 9 && legal[i]->nextPos.y == to[k] % 9) {
                    mv = legal[i];
                    break;
                }
            }
            if (mv == nullptr) { Steps::instance().put(legal); break; }
            Step copy = *mv;
            Steps::instance().put(legal);
            double d = 0.0;
            c.moveForward(&copy, d);
            played++;
        }
        CHECK(played == 4, "走完 4 手可逆循环 (回到起始局面)");
        RL::Tensor after(SACAZAgent::STATE_DIM, 1);
        agent.encodeStateFor(c.sideToMove, after);
        double boardDiff = 0.0;
        for (std::size_t i = 0; i < board0.size(); i++) {
            boardDiff = std::max(boardDiff, std::fabs((double)after[i] - (double)board0[i]));
        }
        const float ctx1 = after[(std::size_t)SACAZAgent::CTX_BASE + 1];
        std::printf("  回到起始局面: 棋平面差 %.1e, 规则上下文[重复] %.3f -> %.3f\n",
                    boardDiff, (double)ctx0, (double)ctx1);
        CHECK(boardDiff == 0.0, "循环之后棋子平面逐位相同 (确实是同一个局面)");
        CHECK(ctx1 > ctx0, "**规则上下文变了** —— 状态不再是裸棋盘 (Markov 修正生效)");
    }

    /* 构造一个人工掩码 (只开放前 4 个动作), 便于手算 */
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    mask.zero();
    mask[0] = mask[1] = mask[2] = mask[3] = 1.0f;

    RL::Tensor pi(SACAZAgent::ACTION_DIM, 1);
    agent.policy(state, mask, pi);
    float psum = 0.0f;
    for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
        psum += pi[i];
    }
    CHECK(std::fabs(psum - 1.0f) < 1e-5f, "人工掩码下策略也归一化");

    RL::Tensor q1v(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q2v(SACAZAgent::ACTION_DIM, 1);
    agent.qValues(state, q1v, q2v);

    /* α = 0: 应等于 Σ π·min(Q1,Q2) */
    const float savedAlpha = agent.alpha[0];
    agent.alpha[0] = 0.0f;
    const float v0 = agent.softValueFrom(pi, mask, q1v, q2v);
    float manual = 0.0f;
    for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
        if (mask[i] > 0.5f) {
            manual += pi[i] * std::min(q1v[i], q2v[i]);
        }
    }
    CHECK(std::fabs(v0 - manual) < 1e-5f, "α = 0 时软价值退化成策略期望 Q");

    /*
       α > 0: SAC 的软价值定义是
           V = E_π[Q − α·log π] = E_π[Q] + α·H     (H = −E_π[log π] ≥ 0)
       注意熵项是**加**上去的 (因为 −log π ≥ 0), 所以 V(α=0.5) − V(α=0) 应当正好
       等于 α·H。这里核对的就是这个恒等式 —— 它是"软价值实现对不对"的直接判据。
    */
    agent.alpha[0] = 0.5f;
    const float v1 = agent.softValueFrom(pi, mask, q1v, q2v);
    float H = 0.0f;
    for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
        if (pi[i] > 0.0f) {
            H -= pi[i] * std::log(pi[i]);
        }
    }
    std::printf("    H=%.4f, V(α=0)=%.5f, V(α=0.5)=%.5f, 差=%.5f (期望 +αH=%.5f)\n",
                (double)H, (double)v0, (double)v1, (double)(v1 - v0), (double)(0.5f * H));
    CHECK(std::fabs((v1 - v0) - 0.5f * H) < 1e-4f, "熵项恰好是 +α·H (SAC 的软价值定义)");
    CHECK(v1 > v0, "α > 0 使软价值升高 (熵越大估值越高)");
    agent.alpha[0] = savedAlpha;
}

/* ============================================================
 *  4. critic 学习: Q 收敛到固定目标
 * ============================================================ */
static void testCriticLearning()
{
    std::printf("\n[4] critic 学习: 固定目标 0.5, 看 Q 是否靠近\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);

    /*
       合成一批经验: 同一个局面, 走同一个动作 10, reward = +0.5 且 done = true。
       done 时 TD 目标不含自举项, 于是 y 恰好 = 0.5。学习的唯一任务就是把
       Q(s,10) 从初始的 ~0 拉到 0.5 —— 目标明确、可断言。
    */
    const int action = 10;
    std::vector<std::uint16_t> cells;
    for (int i = 0; i < 20; i++) {
        cells.push_back((std::uint16_t)(i * 61 + 7));
    }
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    setMaskBits(agent, mask, {action, 20, 33, 70, 111});
    std::uint64_t lo = 0, hi = 0;
    SACAZAgent::maskToBits(mask, lo, hi);

    for (int i = 0; i < 48; i++) {
        SACAZAgent::Transition tr;
        tr.cells = cells;
        tr.nextCells = cells;
        tr.curMaskLo = lo;
        tr.curMaskHi = hi;
        tr.nextMaskLo = lo;
        tr.nextMaskHi = hi;
        tr.action = action;
        tr.legalCount = 5;
        tr.reward = 0.5f;
        tr.done = true;
        tr.hasSearch = false;
        agent.memories.push_back(tr);
    }

    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    SACAZAgent::expandSparse(cells, state);
    RL::Tensor q1v(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q2v(SACAZAgent::ACTION_DIM, 1);

    agent.qValues(state, q1v, q2v);
    const float err0 = std::fabs(q1v[action] - 0.5f);
    const float q0 = q1v[action];

    float lastLoss = 0.0f;
    for (int it = 0; it < 60; it++) {
        lastLoss = agent.learnBatch(8);
    }

    agent.qValues(state, q1v, q2v);
    const float err1 = std::fabs(q1v[action] - 0.5f);
    const float q1 = q1v[action];

    std::printf("    Q(s,10): %.5f -> %.5f  (目标 0.5)\n", (double)q0, (double)q1);
    std::printf("    |误差|:  %.5f -> %.5f,  最后一次 batch loss = %.6f\n",
                (double)err0, (double)err1, (double)lastLoss);
    std::printf("    alpha = %.4f, learnSteps = %d\n",
                (double)agent.alpha[0], agent.getLearnSteps());

    CHECK(q1 > q0, "Q 朝目标方向移动");
    CHECK(err1 < err0, "误差下降 (critic 确实在学)");
    CHECK(err1 < 0.25f, "误差降到 0.25 以内");
    CHECK(std::isfinite((double)lastLoss), "loss 是有限值 (没有 NaN/inf)");
    CHECK(agent.alpha[0] >= 0.02f && agent.alpha[0] <= 5.0f,
          "alpha 被夹在 [0.02, 5] 内 (自动调节没有发散)");
}

/* ============================================================
 *  5. 探索对棋盘零副作用
 * ============================================================ */
static void testExploreInvariant()
{
    std::printf("\n[5] exploreAndTrain: 棋盘必须逐字段复原\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);

    /* 先走几步, 让局面不是初始状态 */
    std::vector<Step*> legal;
    c.sample(Stone::COLOR_RED, legal);
    double dummy = 0.0;
    c.moveForward(legal[0], dummy);
    c.sideToMove = Stone::COLOR_BLACK;
    Steps::instance().put(legal);

    const std::string before = digest(c);
    const std::size_t memBefore = agent.getMemorySize();

    bool trained = false;
    for (int i = 0; i < 5; i++) {
        trained = agent.exploreAndTrain(Stone::COLOR_BLACK, 16) || trained;
        if (digest(c) != before) {
            break;
        }
    }

    std::printf("    探索信息: %s\n", agent.getExploreInfo().c_str());
    std::printf("    回放池: %zu -> %zu\n", memBefore, agent.getMemorySize());
    CHECK(digest(c) == before, "连续 5 次探索后棋盘逐字段不变 (含 sideToMove)");
    CHECK(trained, "exploreAndTrain 报告做了在线训练");
    CHECK(agent.getMemorySize() >= memBefore, "探索把经验写进了回放缓冲");

    /* 探索之后仍然要能给出合法走法 */
    const Step s = agent.getBestMove(Stone::COLOR_BLACK);
    CHECK(s.valid && c.isLegalMove(Stone::COLOR_BLACK, &s),
          "探索之后仍能给出合法走法");
    CHECK(digest(c) == before, "决策本身也不改动棋盘");
}

/* ============================================================
 *  6. 存取往返
 * ============================================================ */
static void testSaveLoad()
{
    std::printf("\n[6] saveModel / loadModel 往返\n");

    Chess c;
    c.reset();
    SACAZAgent a1(c, 32, 0.99f, 0.001f, 1.5f);

    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    a1.encodeStateFor(Stone::COLOR_RED, state);
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    std::vector<Step*> legal;
    std::vector<int> idx;
    a1.getLegalActions(Stone::COLOR_RED, legal, idx, mask);
    Steps::instance().put(legal);
    RL::Tensor pi1(SACAZAgent::ACTION_DIM, 1);
    a1.policy(state, mask, pi1);

    const std::string prefix = "test_sacaz_tmp";
    const bool saved = a1.saveModel(prefix);
    CHECK(saved, "saveModel 报告成功 (三个文件都写出来了)");

    Chess c2;
    c2.reset();
    SACAZAgent a2(c2, 32, 0.99f, 0.001f, 1.5f);
    const bool loaded = a2.loadModel(prefix);
    CHECK(loaded, "loadModel 报告成功");

    RL::Tensor pi2(SACAZAgent::ACTION_DIM, 1);
    a2.policy(state, mask, pi2);
    float diff = 0.0f;
    for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
        diff = std::fmax(diff, std::fabs(pi1[i] - pi2[i]));
    }
    std::printf("    载入前后策略最大差 = %.3e\n", (double)diff);
    CHECK(diff < 1e-6f, "载入的模型给出完全相同的策略");

    /* 清理临时文件 */
    std::remove((prefix + "_actor").c_str());
    std::remove((prefix + "_q1").c_str());
    std::remove((prefix + "_q2").c_str());

    CHECK(!a2.loadModel("no_such_prefix_xyz"), "载入不存在的权重返回 false");
}

/* ============================================================
 *  7. 自对弈训练循环 (MCTS 策略目标 -> 回放 -> learnBatch)
 *
 *  这里不看棋力 (网络是随机初始化的, 棋力要靠训练量), 只看**链路通不通**:
 *  自对弈能下完、经验进池、learnBatch 真的被调用过。
 * ============================================================ */
static void testSelfPlayLoop()
{
    std::printf("\n[7] 自对弈训练链路 (小规模, 8 次模拟 x 24 手)\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);
    /*
       learnBatch 在"回放池不足一个 batch"时是**故意**直接返回的 (不拿半个 batch
       去更新)。所以 24 手的短局要把 batch 调小, 否则一次都不会触发。
    */
    agent.batchSize = 8;

    agent.trainSelfPlay(1, 8, 24, false, 1.0f, 0.25f, 4);

    std::printf("    回放池 = %zu 条, learnSteps = %d\n",
                agent.getMemorySize(), agent.getLearnSteps());
    CHECK(agent.getMemorySize() >= 16, "自对弈把经验写进了回放池");
    CHECK(agent.getLearnSteps() > 0, "自对弈过程中调用过 learnBatch");
    CHECK(agent.getTotalEpisodes() == 1, "统计里记了 1 局");

    /* 自对弈会 reset 棋盘, 但结束后棋盘必须是**合法局面** (32 个子都在) */
    int alive = 0;
    for (int i = 0; i < 32; i++) {
        if (c.m_children[i] && c.m_children[i]->alive) {
            alive++;
        }
    }
    std::printf("    自对弈结束后棋盘上存活棋子 = %d\n", alive);
    CHECK(alive >= 2 && alive <= 32, "棋盘状态仍然合法");
}

/* ============================================================
 *  计时辅助: clock() 只有 ~1 ms 分辨率, 便宜的网络的单次前向只有 0.02 ms,
 *  40~60 次迭代会直接舍入成 0。改用 steady_clock 并按量级选迭代次数。
 * ============================================================ */
static double timeForwardMs(RL::Net &net, const RL::Tensor &x, int n)
{
    net.forward(x);   /* warm up */
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        net.forward(x);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t1 - t0).count();
    return ns / 1e6 / (double)n;
}

/* ============================================================
 *  8. 主干选型的代价对比 (信息, 不断言)
 *
 *  本 agent 用的是普通的 MLP 主干, 而不是本项目 DQN/SAC 用的
 *  `MOE<16,16>` + `TransformerBlock<16>`。这一节把两者的代价量出来, 让选择有据可依:
 *  搜索每展开一个叶子要做 3 次前向 (actor + 双 critic), 所以单次前向的耗时直接
 *  决定"MCTS 一次决策能跑多少次模拟"。
 *
 *  注意 `MOE::forward` 是**稠密**混合 —— 它循环调用全部 16 个专家的前向再做门控
 *  加权, `if (gi > 1e-8f)` 只跳过累加、不跳过计算。所以这里的代价就是 16 倍。
 * ============================================================ */
static void testBackboneCost()
{
    std::printf("\n[8] 主干代价对比 (MLP vs MOE+TransformerBlock)\n");

    const std::size_t S = SACAZAgent::STATE_DIM;   /* 1263 (平面 1260 + 3 个规则上下文槽) */
    RL::Tensor x(S, 1);
    for (std::size_t i = 0; i < S; i++) {
        x[i] = 0.1f * std::sin((float)i);
    }
    const int N = 60;
    const int NCheap = 20000;   /* 便宜的网需要更多次才能被测准 */

    /* A. 本 agent 用的 MLP: 1260 -> 64 -> 64 -> 128 */
    RL::Net mlp(RL::Layer<RL::Tanh>::_(S, 64, true, true),
                RL::Layer<RL::Tanh>::_(64, 64, true, true),
                RL::Layer<RL::Linear>::_(64, SACAZAgent::ACTION_DIM, true, true));
    const double msMlp = timeForwardMs(mlp, x, NCheap);

    /* B. 一个专家 = TransformerBlock<16> 直接在 1260 维上 (不做 16 次, 只测一个) */
    RL::Net tb(RL::TransformerBlock<16>::_(S, true));
    const double msOneExpert = timeForwardMs(tb, x, N);

    /* C. DQN 的配置: MOE<16,16> 在 **90** 维上 (可以直接建, 便宜) */
    RL::Tensor x90(90, 1);
    for (int i = 0; i < 90; i++) {
        x90[i] = 0.1f * std::sin((float)i);
    }
    RL::Net moe90(RL::MOE<16, 16>::_(90, true),
                  RL::TransformerBlock<16>::_(90, true),
                  RL::TanhNorm<RL::Sigmoid>::_(90, 64, true, true),
                  RL::Layer<RL::Sigmoid>::_(64, SACAZAgent::ACTION_DIM, true, true));
    const double msMoe90 = timeForwardMs(moe90, x90, 500);

    /* MOE<16,16> 在 1260 维上 = 16 个专家 + 门控 + 尾部层 (按上面实测外推) */
    const double msMoe1260 = 16.0 * msOneExpert;

    std::printf("    MLP  1260->64->64->128        : %8.3f ms/次\n", msMlp);
    std::printf("    TransformerBlock<16> @1260    : %8.3f ms/次  (1 个专家)\n", msOneExpert);
    std::printf("    MOE<16,16>+TB+TanhNorm @1260  : %8.3f ms/次  (16 个专家, 外推)\n",
                msMoe1260);
    std::printf("    MOE<16,16>+TB+TanhNorm @90    : %8.3f ms/次  (DQN 的配置)\n", msMoe90);
    std::printf("    -> 在 1260 维上 MOE 是 MLP 的 %.0f 倍\n", msMoe1260 / msMlp);
    std::printf("    一次决策 (256 模拟 x 3 个网络):\n");
    std::printf("       MLP 主干 ≈ %7.0f ms\n", msMlp * 3.0 * 256.0);
    std::printf("       MOE 主干 ≈ %7.0f ms\n", msMoe1260 * 3.0 * 256.0);
    std::printf("    参数量估算 (float32): 1 个专家 @1260 ≈ 19M 参数, 16 个 ≈ 305M ≈ 1.2 GB\n");
}

/* ============================================================
 *  9. 稀疏路由 / 减少专家数与 head 数, 到底能不能救 MOE?
 *
 *  三个反直觉的点, 这一节都用实测说话:
 *
 *  (a) **减少 head 数会让注意力更贵**: 本库的 MultiHeadAttention 是"从 NumHeads
 *      往下找能整除 d_model 的 head 数", d_k = d_model/head 数, 每个 head 的注意力
 *      矩阵是 d_k x d_k。总代价 ~ Σ head 数 x d_k² = d² / head 数 —— head 越少 d_k
 *      越大, 越贵。(d_model=1260 时: NumHeads=16 -> 实际 15 heads, d_k=84;
 *      NumHeads=4 -> 4 heads, d_k=315, 注意力元素数 3.75 倍)
 *
 *  (b) **稀疏路由救不了 TransformerBlock 专家**: top-k 只算 k 个专家, 但"一个专家
 *      在 1260 维上"本身就要 ~9.5 ms, top-1 依然是 ~9.5 ms/前向。
 *
 *  (c) **稀疏路由 + 廉价专家才可行**: 把专家换成普通 MLP (1260->64->64->1260),
 *      单个专家只要 ~0.02 ms, top-1/top-2 都落在几十毫秒/步的预算内。
 * ============================================================ */
static void testSparseMoECost()
{
    std::printf("\n[9] 稀疏路由 / head 数 / 专家代价 (信息, 不断言)\n");

    const std::size_t S = SACAZAgent::STATE_DIM;   /* 1260 */
    RL::Tensor x(S, 1);
    for (std::size_t i = 0; i < S; i++) {
        x[i] = 0.1f * std::sin((float)i);
    }
    const int N = 40;
    const double sims = 256.0;

    auto timeNet = [&](RL::Net &net, const RL::Tensor &in, int n) -> double {
        return timeForwardMs(net, in, n);
    };

    /* (a) head 数的影响 (d_model=1260 固定) */
    RL::Net tb1(RL::TransformerBlock<1>::_(S, true));
    RL::Net tb4(RL::TransformerBlock<4>::_(S, true));
    RL::Net tb8(RL::TransformerBlock<8>::_(S, true));
    RL::Net tb16(RL::TransformerBlock<16>::_(S, true));
    const double msH1 = timeNet(tb1, x, 20);
    const double msH4 = timeNet(tb4, x, N);
    const double msH8 = timeNet(tb8, x, N);
    const double msH16 = timeNet(tb16, x, N);
    std::printf("    TransformerBlock< 1> @1260: %8.3f ms   (1 head,  d_k=1260)\n", msH1);
    std::printf("    TransformerBlock< 4> @1260: %8.3f ms   (4 heads, d_k=315)\n", msH4);
    std::printf("    TransformerBlock< 8> @1260: %8.3f ms   (7 heads, d_k=180)\n", msH8);
    std::printf("    TransformerBlock<16> @1260: %8.3f ms   (15 heads, d_k=84)\n", msH16);
    std::printf("    -> 减少 head 数是变**贵**的: 1 head 是 16 heads 的 %.1f 倍\n",
                msH1 / msH16);

    /* (c) 稀疏路由 + 廉价专家 (MLP 专家, 1260->64->64->1260) */
    RL::Net e1(RL::Layer<RL::Tanh>::_(S, 64, true, true),
               RL::Layer<RL::Tanh>::_(64, 64, true, true),
               RL::Layer<RL::Linear>::_(64, S, true, true));
    const double msExpert = timeNet(e1, x, 20000);
    std::printf("    1 个 MLP 专家 1260->64->64->1260: %8.4f ms\n", msExpert);
    std::printf("    稀疏 MoE (E=8, 专家=MLP), 每前向:\n");
    std::printf("        top-1 : %8.4f ms   (只算 1 个专家 + 门控)\n", msExpert);
    std::printf("        top-2 : %8.4f ms\n", 2.0 * msExpert);
    std::printf("    参数量: 1 个 MLP 专家 ≈ %.2f M, E=8 ≈ %.2f M (≈ %.0f MB, float32)\n",
                2.0 * (double)S * 64.0 / 1e6,
                8.0 * 2.0 * (double)S * 64.0 / 1e6,
                8.0 * 2.0 * (double)S * 64.0 * 4.0 / 1e6);

    /* 一次走子的预算 (256 模拟 x 3 个网络) */
    std::printf("    一次决策 (256 模拟 x 3 网络) 的主干开销:\n");
    std::printf("        普通 MLP          ≈ %7.0f ms\n", 0.017 * 3.0 * sims);
    std::printf("        稀疏 MoE(MLP专家) ≈ %7.0f ms   <- 可行\n", msExpert * 3.0 * sims);
    std::printf("        稠密 MOE<TB 专家> ≈ %7.0f ms   <- 不可行\n",
                16.0 * msH16 * 3.0 * sims);
    std::printf("        top-1 稀疏(TB 专家) ≈ %7.0f ms  <- 仍然不可行\n",
                msH16 * 3.0 * sims);
}

/* ============================================================
 *  10. 骨干扫描: 四种骨干在**真实 agent 路径**上都要能跑
 *
 *  [8][9] 量的是"裸网络"的代价, 这里量的是接进 agent 之后 (actor + 双 critic +
 *  MCTS + 回放训练) 的实际表现。四种骨干:
 *    MLP          基线
 *    稀疏MoE(MLP专家) E=8 top-2  —— 容量 x8, 算力 ~2 个专家
 *    稀疏MoE(TB专家)  E=4 top-1  —— 容量最大, 但一个专家就要 3 ms 级
 *    稠密MoE(TB专家)  E=4 全算    —— 与上一个**参数量完全相同**的等参数对照
 *
 *  这里断言的是"结构自检 + 能走合法棋 + 训练链路不炸", 不是棋力 (权重是随机的)。
 * ============================================================ */
static void testBackboneSweep()
{
    std::printf("\n[10] 骨干扫描 (真实 agent 路径: 建网 / 决策 / 探索 / 学习)\n");

    struct Case {
        SACAZAgent::Backbone b;
        int sims;             /* 决策用的模拟次数 (TB 专家只能给很少) */
        int expectExperts;    /* 期望的专家数 (0 = 不是 MoE) */
        int expectTopK;
        /*
           是否在这里做"存了再读"的往返检查。TB 专家骨干参数量 28.7 M, 而权重是
           十进制文本存的 —— 存一次就是几百 MB 文本, 会让这个测试从 16 s 涨到
           180 s。序列化顺序的风险用**小 d_model** 在 test_sparse_moe 里查更划算
           (那里覆盖了 MlpExpert 与 TransformerBlock 两种专家)。
        */
        bool roundTrip;
    };
    const Case cases[4] = {
        { SACAZAgent::Backbone::Mlp,         64, 0, 0, true },
        { SACAZAgent::Backbone::SparseMoeMlp, 32, SACAZAgent::MOE_MLP_EXPERTS,
          SACAZAgent::MOE_MLP_TOPK, true },
        { SACAZAgent::Backbone::SparseMoeTb,   4, SACAZAgent::MOE_TB_EXPERTS,
          SACAZAgent::MOE_TB_TOPK, false },
        { SACAZAgent::Backbone::DenseMoeTb,    4, SACAZAgent::MOE_TB_EXPERTS,
          SACAZAgent::MOE_TB_EXPERTS, false }
    };

    for (int ci = 0; ci < 4; ci++) {
        const Case &cs = cases[ci];
        Chess c;
        c.reset();
        /*
          每轮都重新建 agent (构造里就建了 5 个网络, 这就是在测 buildNet 的各种
           分支)。注意目标网的 withGrad=false —— 载入/前向路径与在线网不同。
        */
        SACAZAgent agent(c, 64, 0.99f, 0.001f, 1.5f, cs.b, 64, 0.01f);
        agent.batchSize = 4;

        std::printf("    %-22s 专家数=%d topK=%d\n",
                    SACAZAgent::backboneName(cs.b),
                    agent.moeExpertCount(), agent.moeTopK());
        CHECK(agent.moeExpertCount() == cs.expectExperts,
              "骨干的专家数符合预期 (非 MoE 骨干为 0)");
        CHECK(agent.moeTopK() == cs.expectTopK, "骨干的 topK 符合预期");

        /* ---- 决策: 必须给出合法走法 ---- */
        const std::string before = digest(c);
        const auto t0 = std::chrono::steady_clock::now();
        const Step s = agent.selectMove(Stone::COLOR_RED, cs.sims, 0.0f);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t1 - t0).count() / 1e6;
        CHECK(s.valid, "selectMove 返回了走法");
        CHECK(c.isLegalMove(Stone::COLOR_RED, &s), "返回的走法是合法走法");
        CHECK(digest(c) == before, "搜索结束后棋盘逐字段复原");
        std::printf("        %d 次模拟的单步决策 = %8.1f ms  (%.3f ms/模拟)\n",
                    cs.sims, ms, ms / (double)cs.sims);

        /* ---- 探索 + 学习: 稀疏 MoE 的辅助损失路径也要走到 ---- */
        agent.exploreAndTrain(Stone::COLOR_RED, 6);
        CHECK(agent.getMemorySize() >= 4, "探索把经验写进了回放池");
        const float loss = agent.learnBatch(4);
        std::printf("        learnBatch(4) -> critic loss = %.4f, alpha = %.4f\n",
                    (double)loss, (double)agent.alpha[0]);
        CHECK(std::isfinite(loss), "critic loss 是有限值 (没出 NaN)");
        CHECK(agent.getLearnSteps() > 0, "learnBatch 真的更新了参数");

        /* ---- 参数必须是有限值 (稀疏 MoE 的辅助梯度写错就会在这里露出来) ---- */
        RL::Tensor state(SACAZAgent::STATE_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, state);
        RL::Tensor q1v(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor q2v(SACAZAgent::ACTION_DIM, 1);
        agent.qValues(state, q1v, q2v);
        bool finite = true;
        double qmax = 0;
        for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
            if (!std::isfinite(q1v[i]) || !std::isfinite(q2v[i])) {
                finite = false;
            }
            qmax = std::fmax(qmax, std::fabs((double)q1v[i]));
        }
        CHECK(finite, "学习之后 Q 值仍然有限");
        CHECK(qmax < 1e6, "Q 值没有爆掉 (|Q|max < 1e6)");

        /* ---- 稀疏 MoE 的使用分布 (坍缩诊断): 一共只给了几次前向, 不苛求均衡 ---- */
        std::vector<long long> usage;
        agent.moeUsage(usage);
        if (!usage.empty()) {
            long long tot = 0;
            std::printf("        专家使用计数 = {");
            for (std::size_t i = 0; i < usage.size(); i++) {
                tot += usage[i];
                std::printf("%lld%s", usage[i], (i + 1 < usage.size()) ? ", " : "}\n");
            }
            CHECK(tot > 0, "稀疏 MoE 真的路由过 (使用计数非零)");
            CHECK(usage.size() == (std::size_t)cs.expectExperts,
                  "使用计数的长度 = 专家数");
        } else {
            std::printf("        (非 MoE 骨干, 没有使用计数)\n");
            CHECK(cs.expectExperts == 0, "只有非 MoE 骨干才没有使用计数");
        }
        agent.resetMoeUsage();
        agent.moeUsage(usage);
        long long after = 0;
        for (std::size_t i = 0; i < usage.size(); i++) {
            after += usage[i];
        }
        CHECK(usage.empty() || after == 0, "resetMoeUsage 清零了计数");

        /*
           ---- 权重往返 (稀疏 MoE 的序列化顺序也要对) ----
           GUI 启动/退出走的就是这条路: saveModel(prefix) -> prefix_actor/_q1/_q2。
           分层 write/read 的**顺序**只要有一处不一致, 载入出来的就是另一个网络
           (而且是静默的), 所以这里用一个全新初始化的 agent 载入后比对策略输出。
        */
        if (cs.roundTrip) {
            const std::string prefix = std::string("test_sacaz_bb") + (char)('a' + ci);
            CHECK(agent.saveModel(prefix), "saveModel 报告成功 (三个文件)");
            SACAZAgent fresh(c, 64, 0.99f, 0.001f, 1.5f, cs.b, 64, 0.01f);
            CHECK(fresh.loadModel(prefix), "loadModel 报告成功");            RL::Tensor p1(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor p2(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor m1(SACAZAgent::ACTION_DIM, 1);
            agent.policy(state, m1, p1);
            fresh.policy(state, m1, p2);
            double dp = 0;
            for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                dp = std::fmax(dp, std::fabs((double)p1[i] - (double)p2[i]));
            }
            std::printf("        save/load 往返: 策略最大差 = %.3e\n", dp);
            CHECK(dp < 1e-6, "载入后的策略与保存前一致 (层结构/权重的读写顺序正确)");
            /* Q 网是独立的两个文件, 也顺手比一下 */
            RL::Tensor qa(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor qb(SACAZAgent::ACTION_DIM, 1);
            agent.qValues(state, qa, qb);
            RL::Tensor f1(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor f2(SACAZAgent::ACTION_DIM, 1);
            fresh.qValues(state, f1, f2);
            double dq = 0;
            for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                dq = std::fmax(dq, std::fabs((double)qa[i] - (double)f1[i]));
                dq = std::fmax(dq, std::fabs((double)qb[i] - (double)f2[i]));
            }
            std::printf("        save/load 往返: Q 值最大差 = %.3e\n", dq);
            /*
               容差为什么是 1e-4 而不是 1e-6: 权重是**十进制文本**存的
               (Tensor::toString), 有效位数有限, 所以往返一次本来就有 ~1e-6 相对
               误差 —— Q 在 0.1~1 量级上就是 1e-6~1e-5 的绝对差。策略那边因为过了
               掩码 softmax (饱和) 看起来是 0, 不代表 Q 也是 0。
               这里要挡住的是"读写顺序错位"这种**结构性**错误, 那会是 O(1) 的差。
            */
            CHECK(dq < 1e-4, "载入后的 Q 值与保存前一致 (容差按文本权重精度取 1e-4)");
            std::remove((prefix + "_actor").c_str());
            std::remove((prefix + "_q1").c_str());
            std::remove((prefix + "_q2").c_str());
        }
    }
}

/* ============================================================
 *  11. 从 RL::PPO 搬过来的两条 (P4 多 epoch / R2 只在合法列上训练)
 *
 *  这一节盯的两件事都是"机制成立但很容易写成假的"那种:
 *
 *   [R2] **非法列的头部权重必须逐位不变**。
 *        dz = π_a(g_a − Σg·π) 在 π_a = 0 的非法列上恰好为 0, 于是那一行的权重梯度
 *        也恰好为 0, RMSProp 里 w -= lr·0/(√v+ε) 就是**恒等操作**。这不是"变化很小",
 *        是逐位不变 —— 所以可以断言 ==。
 *        正对照同时要有: 合法列**必须**变, 否则上一条断言在任何"根本没学"的实现上
 *        都会通过。
 *
 *   [P4] **多 epoch = 每遍重新抽 batchSize 条**, 而优化器只调一次。
 *        用它自己的诊断数字 getLastBatchSamples() 断言 (batchSize×epochs),
 *        并用 getLearnSteps() 断言"不管几个 epoch 都只 +1"。
 * ============================================================ */
static void testPpoPort()
{
    std::printf("\n[11] PPO 移植: R2 只在合法列上训练 / P4 多 epoch\n");

    Chess chess;
    chess.reset();
    SACAZAgent agent(chess, 64, 0.99f, 0.001f, 1.5f);
    agent.batchSize = 4;

    /* ---------------- R2 ---------------- */
    std::vector<Step*> legal;
    std::vector<int> legalIdx;
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    agent.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
    Steps::instance().put(legal);
    CHECK(!legalIdx.empty(), "开局红方有合法走法");

    std::vector<std::uint16_t> cells;
    agent.encodeSparse(Stone::COLOR_RED, cells);
    std::uint64_t lo = 0, hi = 0;
    SACAZAgent::maskToBits(mask, lo, hi);

    /* 用**同一个掩码**造一批经验: 于是"某个动作在所有样本里都非法"是可判定的 */
    for (int i = 0; i < 64; i++) {
        SACAZAgent::Transition tr;
        tr.cells = cells;
        tr.nextCells = cells;
        tr.curMaskLo = lo;
        tr.curMaskHi = hi;
        tr.nextMaskLo = lo;
        tr.nextMaskHi = hi;
        tr.action = legalIdx[(std::size_t)(i % (int)legalIdx.size())];
        tr.legalCount = (int)legalIdx.size();
        tr.reward = 0.05f;
        tr.done = false;
        tr.hasSearch = false;      /* 只训练 critic 与 SAC 的软 Q 项 */
        agent.memories.push_back(tr);
    }

    RL::iFcLayer *head =
        dynamic_cast<RL::iFcLayer*>(agent.actor[agent.actor.size() - 1]);
    CHECK(head != nullptr, "actor 的最后一层是 iFcLayer (策略头)");
    if (head == nullptr) {
        return;
    }
    const std::size_t rowStride = (std::size_t)head->w.sizes[0];

    std::vector<int> illegal;
    for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
        if (mask[a] <= 0.5f) {
            illegal.push_back(a);
        }
    }
    std::printf("  合法走法 %d 个, 非法槽位 %d 个 (动作空间 %d)\n",
                (int)legalIdx.size(), (int)illegal.size(), SACAZAgent::ACTION_DIM);
    CHECK(!illegal.empty(), "确实存在非法槽位 (否则这条断言没有意义)");

    auto snapshot = [&](const std::vector<int> &acts) {
        std::vector<float> buf;
        buf.reserve(acts.size() * rowStride);
        for (std::size_t i = 0; i < acts.size(); i++) {
            const float *row = head->w.val.data() + (std::size_t)acts[i] * rowStride;
            for (std::size_t k = 0; k < rowStride; k++) {
                buf.push_back(row[k]);
            }
        }
        return buf;
    };
    const std::vector<float> guardBefore = snapshot(illegal);
    const std::vector<float> legalBefore = snapshot(legalIdx);

    for (int rep = 0; rep < 10; rep++) {
        agent.learnBatch(4);
    }

    const std::vector<float> guardAfter = snapshot(illegal);
    const std::vector<float> legalAfter = snapshot(legalIdx);
    std::size_t changedIllegal = 0;
    for (std::size_t k = 0; k < guardBefore.size(); k++) {
        if (guardBefore[k] != guardAfter[k]) { changedIllegal++; }
    }
    std::size_t changedLegal = 0;
    for (std::size_t k = 0; k < legalBefore.size(); k++) {
        if (legalBefore[k] != legalAfter[k]) { changedLegal++; }
    }
    std::printf("  10 次 learnBatch 之后: 非法列改动 %zu/%zu 个权重, 合法列改动 %zu/%zu\n",
                changedIllegal, guardBefore.size(), changedLegal, legalBefore.size());
    CHECK(changedLegal > 0, "合法列的权重确实被训练了 (正对照)");
    CHECK(changedIllegal == 0,
          "非法列的头部权重**逐位不变** (dz 在非法列上恰好为 0)");

    /* ---------------- P4 ---------------- */
    const int steps0 = agent.getLearnSteps();
    agent.replayEpochs = 1;
    CHECK(agent.learnBatch(4), "learnBatch(4) (replayEpochs=1) 成功");
    const int samples1 = agent.getLastBatchSamples();
    const int steps1 = agent.getLearnSteps();
    agent.replayEpochs = 3;
    CHECK(agent.learnBatch(4), "learnBatch(4) (replayEpochs=3) 成功");
    const int samples3 = agent.getLastBatchSamples();
    const int steps3 = agent.getLearnSteps();
    agent.replayEpochs = 1;
    CHECK(agent.learnBatch(4, 2), "learnBatch(4, 2) (显式形参覆盖成员) 成功");
    const int samplesExplicit = agent.getLastBatchSamples();
    const int stepsExplicit = agent.getLearnSteps();

    std::printf("  攒下的样本条数: epoch=1 -> %d, epoch=3 -> %d, 显式 epoch=2 -> %d\n",
                samples1, samples3, samplesExplicit);
    std::printf("  学习步数: %d -> %d -> %d -> %d (每次调用只 +1)\n",
                steps0, steps1, steps3, stepsExplicit);
    CHECK(samples1 == 4, "1 个 epoch 攒 batchSize 条");
    CHECK(samples3 == 12, "3 个 epoch 攒 batchSize×3 条 (每遍重新抽新样本)");
    CHECK(samplesExplicit == 8, "显式 epochs 形参优先于 replayEpochs 成员");
    CHECK(steps1 == steps0 + 1 && steps3 == steps1 + 1 && stepsExplicit == steps3 + 1,
          "无论几个 epoch, 每次 learnBatch 只产生 1 次优化器更新 (P3)");
}

/* ============================================================
 *  12. 带 TB 专家的 SAC agent 专项验证
 *
 *  [10] 的骨干扫描对四种骨干只回答"能建 / 能走 / 能训一次 / 能存取"。TB 专家那条
 *  (`SparseMoeTb` / `DenseMoeTb`) 还需要单独回答四个问题 —— 这个测试就是来回答的:
 *
 *    (a) **学得动吗**: 固定经验 + 已知 TD 目标 (reward=+0.5 且 done, 于是 y ≡ 0.5),
 *        Q 必须朝它走。梯度要穿过 TransformerBlock 专家 (多头注意力 + FFN + 两层
 *        LayerNorm) 与门控, 任何一处接错这一条就会露出来。
 *    (b) **策略改得动吗**: 同一批经验带 AlphaZero 监督项 (π_MCTS = one-hot),
 *        策略在目标动作上的质量必须上升 —— 这条走的是"掩码 softmax 的雅可比 + TB 骨干"。
 *    (c) **自对弈跑得动吗、路由健康吗**: 短局自对弈跑通、池子长、loss 有限、Q/π 无 NaN,
 *        以及 **4 个专家到底有没有都被用到** —— top-1 路由最容易在这里坍缩。
 *    (d) **代价**: ms/模拟 与 ms/learnBatch。TB 专家单价高, 这个数字直接决定
 *        "同一个时间预算下能跑几次模拟", 所以它和一条 MLP 骨干的对照在同一个进程里量
 *        (两块互不重叠: 一个 TB agent 的五个网络约 1.6 GB, 同时活着会顶到内存上限)。
 *
 *  不断言棋力: 权重是随机初始化的, 棋力要靠训练量 (见 bench_moe 的负面结果)。
 *  这里断言的是**机制**: 梯度通、策略动、链路通、数值稳。
 * ============================================================ */
static void testTbExpertAgent()
{
    std::printf("\n[12] 带 TB 专家的 SAC agent 专项验证\n");

    Chess c;
    c.reset();
    /*
       lr 取 0.003 (比 MLP 那条的 0.001 大): TB 骨干的 h 要穿过门控加权 (top-1 时
       输出 ≈ gate·expert, 量级被 gate ≈ 0.25 压小), 同样的步数下学得更慢。
       辅助损失系数 0.1 与 RL::PPO / RL::SAC 的默认值一致。
    */
    SACAZAgent agent(c, 64, 0.99f, 0.003f, 1.5f,
                     SACAZAgent::Backbone::SparseMoeTb, 64, 0.1f);
    agent.batchSize = 4;

    std::printf("  骨干=%s  专家=%d  topK=%d\n",
                SACAZAgent::backboneName(agent.backbone),
                agent.moeExpertCount(), agent.moeTopK());
    std::printf("  参数: actor=%lld  q1=%lld  (target 网同构但无梯度)\n",
                agent.actor.paramCount(), agent.q1.paramCount());
    CHECK(agent.moeExpertCount() == SACAZAgent::MOE_TB_EXPERTS,
          "骨干确实是 TB 专家那条 (专家数 = MOE_TB_EXPERTS)");
    CHECK(agent.moeTopK() == SACAZAgent::MOE_TB_TOPK, "topK = MOE_TB_TOPK");
    CHECK(agent.actor.paramCount() > 0, "参数量数得出来 (TransformerBlock 的 paramCount 有实现)");

    /* ---- 局面: 真实开局, 红方走 ---- */
    std::vector<Step*> legal;
    std::vector<int> legalIdx;
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    agent.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
    Steps::instance().put(legal);
    CHECK(legalIdx.size() >= 3, "开局红方有足够多的合法走法");

    std::vector<std::uint16_t> cells;
    agent.encodeSparse(Stone::COLOR_RED, cells);
    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    SACAZAgent::expandSparse(cells, state);
    std::uint64_t lo = 0, hi = 0;
    SACAZAgent::maskToBits(mask, lo, hi);

    const int action = legalIdx[0];

    /* 用**同一个局面**造一池经验; hasSearch=false -> 只训 critic 与软 Q 项 */
    auto fillPool = [&](float reward, bool done, bool hasSearch, int target) {
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
            tr.legalCount = (int)legalIdx.size();
            tr.reward = reward;
            tr.done = done;
            tr.hasSearch = hasSearch;
            if (hasSearch && target >= 0 && target < SACAZAgent::ACTION_DIM) {
                tr.pi[target] = 1.0f;
            }
            agent.memories.push_back(tr);
        }
    };

    /* ================= (a) 策略改得动吗: AZ 监督项 (配对 A/B) =================
       这条**不能**写成"跑固定步数后 π(目标) 必须比一开始高" —— 试过, 是**假失败**,
       原因有两个, 都量过 (.r1build/dbg_az.cpp):
         * `RMSProp` 的 clipGrad 把整张梯度张量归一到单位长度, 于是**每步位移恒等于 lr**、
           与梯度大小无关 —— 强监督项下"冲过头再弹回来"是常态。实测轨迹 (π(目标)):
             0.53 -> 0.61 -> 0.51 -> 0.37 -> 0.36 -> 0.39 -> 0.35 -> ... -> 0.001 -> 0.92
           终点落在振荡曲线的哪一点基本是运气。
         * 软 Q 项与熵项**自己**也会动 π, 所以"π 上升"本身不是监督项的证据。
       所以改成**配对 A/B**: 五个网络全部 copyTo 成同一份权重、同一批经验、同样步数,
       **只改 `hasSearch`** (有没有监督项), 并且比**轨迹的统计量**而不是终点。
       实测 (20 次 learnBatch(4), 均匀 = 0.0227, 全新 agent):
                          均值     峰值     终点
         带监督项 (TB)    0.402    0.917    0.455
         对照组   (TB)    0.018    0.059    0.010
         带监督项 (MLP)   0.379    0.605    0.416
         对照组   (MLP)   0.035    0.057    0.028
       两个骨干的结论一致: 监督项把 π(目标) 抬到**均匀的 10 倍以上**, 而对照组基本不动。

       **顺序上的一个硬约束**: 这条必须跑在 (b) critic 学习**之前**, 也就是用**没被
       训练过的 critic**。反过来的顺序量到的是另一件事: 先在同一个局面上用 30 批把
       critic 训到 |Q| ~ 0.4+, 同一个协议就只有 0.0139 vs 0.0100 (几乎没差别) ——
       那时监督项在 8 步内拗不过软 Q 项。那是一个**独立的、未定论的观察**
       (是 critic 的量级还是 π 的饱和在起作用, 没有隔离出来), 记在
       docs/issues_review.md 的待办里, 不要把它和"监督项有没有用"混为一谈。
    */
    std::printf("\n  (a) 策略改进: 配对 A/B (只改 hasSearch), 比轨迹统计量而不是终点\n");
    const int target = legalIdx[legalIdx.size() / 2];
    const double uniform = 1.0 / (double)legalIdx.size();
    double azMean = 0.0, azPeak = 0.0, ctrlMean = 0.0, ctrlPeak = 0.0;
    {
        /*
           对照组 agent **只在这一块里活着**: 一个 TB 专家 agent 是五个网络 ≈ 1.6 GB,
           两个同时存在已经到顶 —— 不能让它跟后面的 (c)/(d) 一起活着。
        */
        SACAZAgent ctrl(c, 64, 0.99f, 0.003f, 1.5f,
                        SACAZAgent::Backbone::SparseMoeTb, 64, 0.1f);
        ctrl.batchSize = 4;
        /* 五个网络全部 copyTo (Net 的拷贝语义是**浅拷贝**: 共享层指针, 深拷贝只能走 copyTo) */
        agent.actor.copyTo(ctrl.actor);
        agent.q1.copyTo(ctrl.q1);
        agent.q2.copyTo(ctrl.q2);
        agent.q1.copyTo(ctrl.q1Target);
        agent.q2.copyTo(ctrl.q2Target);

        fillPool(0.0f, true, true, target);       /* agent: π_MCTS = one-hot(target) */
        ctrl.memories = agent.memories;           /* 同一批经验 */
        for (std::size_t i = 0; i < ctrl.memories.size(); i++) {
            ctrl.memories[i].hasSearch = false;   /* 只差这一个开关 */
        }

        struct Traj { double mean, peak, last; };
        auto runTraj = [&](SACAZAgent &a, int iters) {
            RL::Tensor p(SACAZAgent::ACTION_DIM, 1);
            a.policy(state, mask, p);
            Traj t;
            t.peak = p[target];
            t.last = p[target];
            double sum = 0.0;
            for (int it = 0; it < iters; it++) {
                a.learnBatch(4);
                a.policy(state, mask, p);
                const double v = p[target];
                sum += v;
                t.peak = std::fmax(t.peak, v);
                t.last = v;
            }
            t.mean = sum / (double)iters;
            return t;
        };
        const int azIters = 8;
        const Traj az = runTraj(agent, azIters);
        const Traj ct = runTraj(ctrl, azIters);
        azMean = az.mean;    azPeak = az.peak;
        ctrlMean = ct.mean;  ctrlPeak = ct.peak;
        std::printf("      均匀 = %.5f; %d 次 learnBatch(4)\n", uniform, azIters);
        std::printf("      带监督项: 均值 %.5f  峰值 %.5f  终点 %.5f\n",
                    az.mean, az.peak, az.last);
        std::printf("      对照组  : 均值 %.5f  峰值 %.5f  终点 %.5f\n",
                    ct.mean, ct.peak, ct.last);
    }

    /* 掩码口径完整性: 非法列恰好为 0, 合法集合为 1 */
    RL::Tensor pi(SACAZAgent::ACTION_DIM, 1);
    agent.policy(state, mask, pi);
    double sumLegal = 0.0;
    int illegalNonZero = 0;
    for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            sumLegal += (double)pi[a];
        } else if (pi[a] != 0.0f) {
            illegalNonZero++;
        }
    }
    std::printf("      合法集 Σπ = %.9f, 非法列非零 = %d\n", sumLegal, illegalNonZero);

    CHECK(azMean > 5.0 * ctrlMean,
          "带监督项时 π(目标) 的轨迹均值远高于对照组 (AZ 项真的在起作用)");
    CHECK(azMean > 10.0 * uniform,
          "而且已经远离均匀分布 (不是被软 Q 项/熵项的噪声带着走)");
    CHECK(azPeak > 0.3, "轨迹峰值确实冲高过 (强监督项下的过冲也算数)");
    CHECK(ctrlPeak < 0.15,
          "对照组没有这个效应 (同一批经验、同一份权重, 只少了监督项)");
    CHECK(std::fabs(sumLegal - 1.0) < 1e-5, "合法集上 Σπ = 1 (掩码归一没有坏)");
    CHECK(illegalNonZero == 0, "非法列 π 恰好为 0");

    /* ================= (b) 学得动吗: Q 朝已知 TD 目标走 ================= */
    std::printf("\n  (b) critic 学习: y ≡ 0.5, 看 Q(s,a) 是否靠近\n");
    fillPool(0.5f, true, false, -1);

    RL::Tensor q1v(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q2v(SACAZAgent::ACTION_DIM, 1);
    agent.qValues(state, q1v, q2v);
    const float q0 = q1v[action];
    const float err0 = std::fabs(q0 - 0.5f);

    const auto tLearn0 = std::chrono::steady_clock::now();
    const int learnIters = 30;
    float lastLoss = 0.0f;
    for (int it = 0; it < learnIters; it++) {
        lastLoss = agent.learnBatch(4);
    }
    const auto tLearn1 = std::chrono::steady_clock::now();
    const double msPerBatch = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  tLearn1 - tLearn0).count() / 1e6 / (double)learnIters;

    agent.qValues(state, q1v, q2v);
    const float q1 = q1v[action];
    const float err1 = std::fabs(q1 - 0.5f);
    std::printf("      Q(s,a): %.5f -> %.5f (目标 0.5), |误差| %.5f -> %.5f\n",
                (double)q0, (double)q1, (double)err0, (double)err1);
    std::printf("      最后一次 loss = %.6f, alpha = %.4f\n",
                (double)lastLoss, (double)agent.getAlpha());
    CHECK(q1 > q0, "Q 朝目标方向移动 (梯度穿过了 TB 专家 + 门控)");
    CHECK(err1 < err0, "误差下降 (critic 真的在学)");
    CHECK(std::isfinite((double)lastLoss), "批平均 loss 是有限值");
    CHECK(agent.getAlpha() >= 0.02f && agent.getAlpha() <= 5.0f,
          "alpha 仍在自动调节的夹逼范围内 (没有发散)");

    /* ================= (c) 自对弈 + 路由健康 ================= */
    std::printf("\n  (c) 短局自对弈 (8 次模拟 x 16 手) + 路由健康\n");
    agent.memories.clear();
    agent.resetMoeUsage();
    const int stepsBefore = agent.getLearnSteps();
    const std::size_t 池Before = agent.getMemorySize();

    const auto tSp0 = std::chrono::steady_clock::now();
    agent.trainSelfPlay(1, 8, 16, false, 1.0f, 0.25f, 4);
    const auto tSp1 = std::chrono::steady_clock::now();
    const double msSelfPlay = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  tSp1 - tSp0).count() / 1e6;

    std::vector<long long> usage;
    agent.moeUsage(usage);
    long long usageTotal = 0;
    int unusedExperts = 0;
    std::printf("      自对弈 %d 手用时 %.1f ms; 回放池 %zu -> %zu 条; learnSteps %d -> %d\n",
                agent.getTotalEpisodes() > 0 ? 16 : 0, msSelfPlay,
                池Before, agent.getMemorySize(), stepsBefore, agent.getLearnSteps());
    std::printf("      专家使用计数 = {");
    for (std::size_t i = 0; i < usage.size(); i++) {
        usageTotal += usage[i];
        if (usage[i] == 0) { unusedExperts++; }
        std::printf("%lld%s", usage[i], (i + 1 < usage.size()) ? ", " : "}\n");
    }
    const double usageMaxMin = [&]() {
        long long lo = -1, hi = 0;
        for (long long v : usage) {
            if (lo < 0 || v < lo) { lo = v; }
            if (v > hi) { hi = v; }
        }
        return (lo > 0) ? (double)hi / (double)lo : 0.0;
    }();
    std::printf("      总前向 = %lld, 未用到的专家 = %d, max/min = %.2f\n",
                usageTotal, unusedExperts, usageMaxMin);

    CHECK(agent.getMemorySize() > 池Before, "自对弈把经验写进了回放池");
    CHECK(agent.getLearnSteps() > stepsBefore, "自对弈过程中真的调用过 learnBatch");
    CHECK(agent.getTotalEpisodes() == 1, "统计里记了 1 局");
    CHECK(usageTotal > 0, "TB 专家的稀疏 MoE 真的被前向过");
    CHECK(usage.size() == (std::size_t)agent.moeExpertCount(),
          "使用计数长度 = 专家数");

    /* 数值稳定: 训练之后 Q 与 π 都必须是有限值, 且在合法范围内 */
    agent.qValues(state, q1v, q2v);
    agent.policy(state, mask, pi);
    bool finite = true;
    double qmax = 0.0;
    for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
        if (!std::isfinite(q1v[a]) || !std::isfinite(q2v[a]) || !std::isfinite(pi[a])) {
            finite = false;
        }
        qmax = std::fmax(qmax, std::fabs((double)q1v[a]));
        qmax = std::fmax(qmax, std::fabs((double)q2v[a]));
    }
    std::printf("      |Q|max = %.4f, alpha = %.4f, 自对弈后 Q/π 全有限 = %d\n",
                qmax, (double)agent.getAlpha(), (int)finite);
    CHECK(finite, "自对弈训练之后 Q 与 π 全部是有限值 (无 NaN/inf)");
    CHECK(qmax < 1e6, "Q 没有爆掉");
    CHECK(agent.getAlpha() >= 0.02f && agent.getAlpha() <= 5.0f, "alpha 仍在夹逼范围内");

    /* ================= (d) 代价: TB 专家 vs MLP 骨干 ================= */
    std::printf("\n  (d) 代价 (每样本: actor+q1+q2 的前向+反向 + 两个目标网的前向)\n");
    std::printf("      learnBatch(4) = %.2f ms/次  -> %.2f ms/样本\n",
                msPerBatch, msPerBatch / 4.0);

    /*
       同进程对照: TB agent 占约 1.6 GB, 两个同时活着会顶到内存上限, 所以
       MLP 那个 agent 必须在**独立作用域**里建/量/销毁, 不能和上面的 agent 并存。
    */
    double mlpBatchMs = 0.0;
    {
        Chess c2;
        c2.reset();
        SACAZAgent mlp(c2, 64, 0.99f, 0.003f, 1.5f,
                       SACAZAgent::Backbone::Mlp, 64, 0.1f);
        mlp.batchSize = 4;
        std::vector<Step*> lg;
        std::vector<int> li;
        RL::Tensor mk(SACAZAgent::ACTION_DIM, 1);
        mlp.getLegalActions(Stone::COLOR_RED, lg, li, mk);
        Steps::instance().put(lg);
        std::vector<std::uint16_t> cl;
        mlp.encodeSparse(Stone::COLOR_RED, cl);
        std::uint64_t l2 = 0, h2 = 0;
        SACAZAgent::maskToBits(mk, l2, h2);
        for (int i = 0; i < 48; i++) {
            SACAZAgent::Transition tr;
            tr.cells = cl;
            tr.nextCells = cl;
            tr.curMaskLo = l2;
            tr.curMaskHi = h2;
            tr.nextMaskLo = l2;
            tr.nextMaskHi = h2;
            tr.action = li[0];
            tr.legalCount = (int)li.size();
            tr.reward = 0.5f;
            tr.done = true;
            tr.hasSearch = false;
            mlp.memories.push_back(tr);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 30; i++) { mlp.learnBatch(4); }
        const auto t1 = std::chrono::steady_clock::now();
        mlpBatchMs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         t1 - t0).count() / 1e6 / 30.0;
        std::printf("      MLP 骨干对照: learnBatch(4) = %.2f ms/次  -> %.2f ms/样本\n",
                    mlpBatchMs, mlpBatchMs / 4.0);
        std::printf("      TB / MLP 每样本比 = %.1fx\n",
                    mlpBatchMs > 0.0 ? (msPerBatch / mlpBatchMs) : 0.0);
        CHECK(mlpBatchMs > 0.0, "MLP 对照也量到了 (这个比值才有意义)");
        CHECK(msPerBatch > mlpBatchMs, "TB 专家的每批成本高于 MLP 骨干 (容量不是白拿的)");
    }

    /*
       搜索侧: 一次 selectMove 的 ms/模拟。TB 专家单价 ~8 ms/前向, 而一次模拟要跑
       actor + q1 + q2 三个网络 -> 这就是"同一个时间预算下只能跑几次模拟"的来源。
    */
    const int sims = 4;
    const int moves = 5;
    const auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < moves; i++) {
        agent.selectMove(Stone::COLOR_BLACK, sims, 0.0f);
    }
    const auto s1 = std::chrono::steady_clock::now();
    const double msPerMove = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 s1 - s0).count() / 1e6 / (double)moves;
    const double msPerSim = msPerMove / (double)sims;
    std::printf("      selectMove(%d 模拟) = %.1f ms/步 -> %.2f ms/模拟\n",
                sims, msPerMove, msPerSim);
    std::printf("      换算: GUI 里 175 ms/步的预算只能跑约 %.1f 次模拟\n",
                175.0 / msPerSim);
    CHECK(msPerSim > 0.0, "ms/模拟 量得出来");
}

int main()
{
    std::printf("=== SAC+MCTS+AlphaZero agent 测试 ===\n");
    std::printf("SIMD 内核: %s\n", RL::simdops::instructionSet());
    /* 固定随机种子, 让测试可复现 */
    RL::Random::setSeed(20240913);

    testMaskedSoftmax();
    testSelectMove();
    testSoftValue();
    testCriticLearning();
    testExploreInvariant();
    testSaveLoad();
    testSelfPlayLoop();
    testBackboneCost();
    testSparseMoECost();
    testBackboneSweep();
    testPpoPort();
    testTbExpertAgent();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
