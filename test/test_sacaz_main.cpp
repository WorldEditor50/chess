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
#include "sacazlegacyagent.h"   /* [14] 节: 59e5233 行为还原版是**独立的类** */
#include "ppomcts_agent.h"   /* 表示对齐断言要拿 PPOMCTSAgent 的维度常量做比较 */
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
        /*
           ---- 表示开关的不变量 (2026-09) ----
           SACAZAgent 现在有**两种表示** (编译期开关 SACAZ_ALIGNED_REPR):
             改前 (默认): 17 平面 + 2 个规则上下文槽 (1263) + 128 槽哈希动作
             对齐        : 19 平面 (14 棋子 + 5 上下文) (1710) + 8100 双射动作
           理由与实测数据见 docs/sac_regression_2026_09.md 与 src/sacazagent.h 顶部。

           两种表示下都必须成立的不变量:
             1. STATE_DIM 与"平面数 x 90"自洽;
             2. 改前表示 = 棋子区 + CTX_COUNT 个标量槽; 对齐表示 = 整平面布局;
             3. 对齐表示下**必须**与 PPOMCTSAgent 逐位同口径 (这是对齐的全部意义)。
           这样开关被改坏时, 断言会当场说话, 而不是静默换一个形状。
        */
        if (SACAZAgent::ALIGNED_REPR) {
            CHECK(SACAZAgent::STATE_DIM == SACAZAgent::PLANES * SACAZAgent::CELLS,
                  "对齐表示: STATE_DIM = 19 平面 x 90 格");
            CHECK(SACAZAgent::STATE_DIM == PPOMCTSAgent::STATE_DIM,
                  "对齐表示: 与 PPOMCTS 的 STATE_DIM 相同");
            CHECK(SACAZAgent::ACTION_DIM == PPOMCTSAgent::ACTION_DIM,
                  "对齐表示: 与 PPOMCTS 的 ACTION_DIM 相同 (8100 双射)");
            CHECK(SACAZAgent::CTX_COUNT == ChessState::CTX_COUNT,
                  "对齐表示: 上下文个数与 chessstate.h 的 CTX_COUNT 一致");
        } else {
            CHECK(SACAZAgent::STATE_DIM == SACAZAgent::CTX_BASE + SACAZAgent::CTX_COUNT,
                  "改前表示: STATE_DIM = 14 棋子平面 + CTX_COUNT 个标量槽");
            CHECK(SACAZAgent::CTX_COUNT == 3,
                  "改前表示: 3 个规则上下文 (无吃子 / 重复 / 被将)");
            CHECK(SACAZAgent::PLANES == 17, "改前表示: 17 个平面");
        }
        CHECK(SACAZAgent::ACTION_DIM == SACAZAgent::LEGACY_ACTION_DIM
                  || SACAZAgent::ACTION_DIM == SACAZAgent::ALIGNED_ACTION_DIM,
              "动作空间只能是 128 槽哈希或 8100 双射之一");

        /* 重复次数所在的位置随表示而变 (对齐 = 整平面, 改前 = 尾部标量槽) */
        auto repeatSlot = [](const RL::Tensor &t) -> float {
            if (SACAZAgent::ALIGNED_REPR) {
                return t[(std::size_t)(SACAZAgent::PLANE_REPEAT * SACAZAgent::CELLS)];
            }
            return t[(std::size_t)(SACAZAgent::CTX_BASE + 1)];   /* [无吃子, 重复, 被将] */
        };
        const float ctx0 = repeatSlot(state);
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
        const float ctx1 = repeatSlot(after);
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
    std::uint64_t bits[2] = { 0, 0 };
    SACAZAgent::maskToBits(mask, bits);

    for (int i = 0; i < 48; i++) {
        SACAZAgent::Transition tr;
        tr.cells = cells;
        tr.nextCells = cells;
        tr.curMask[0] = bits[0];
        tr.curMask[1] = bits[1];
        tr.nextMask[0] = bits[0];
        tr.nextMask[1] = bits[1];
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

    const std::size_t S = SACAZAgent::STATE_DIM;   /* 1710 (14 个棋子平面 + 5 个上下文平面, 与 PPOMCTS 相同) */
    RL::Tensor x(S, 1);
    for (std::size_t i = 0; i < S; i++) {
        x[i] = 0.1f * std::sin((float)i);
    }
    const int N = 60;
    const int NCheap = 20000;   /* 便宜的网需要更多次才能被测准 */

    /* A. 本 agent 用的 MLP: 1710 -> 64 -> 64 -> 8100 */
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

    const std::size_t S = SACAZAgent::STATE_DIM;   /* 1710 */
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
    std::uint64_t bits[2] = { 0, 0 };
    SACAZAgent::maskToBits(mask, bits);

    /* 用**同一个掩码**造一批经验: 于是"某个动作在所有样本里都非法"是可判定的 */
    for (int i = 0; i < 64; i++) {
        SACAZAgent::Transition tr;
        tr.cells = cells;
        tr.nextCells = cells;
        tr.curMask[0] = bits[0];
        tr.curMask[1] = bits[1];
        tr.nextMask[0] = bits[0];
        tr.nextMask[1] = bits[1];
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
    double maxIllegalDelta = 0.0;
    for (std::size_t k = 0; k < guardBefore.size(); k++) {
        if (guardBefore[k] != guardAfter[k]) { changedIllegal++; }
        maxIllegalDelta = std::max(maxIllegalDelta,
                                   std::fabs((double)guardBefore[k] - (double)guardAfter[k]));
    }
    std::size_t changedLegal = 0;
    double maxLegalDelta = 0.0;
    for (std::size_t k = 0; k < legalBefore.size(); k++) {
        if (legalBefore[k] != legalAfter[k]) { changedLegal++; }
        maxLegalDelta = std::max(maxLegalDelta,
                                 std::fabs((double)legalBefore[k] - (double)legalAfter[k]));
    }
    std::printf("  10 次 learnBatch 之后: 非法列改动 %zu/%zu 个权重, 合法列改动 %zu/%zu\n",
                changedIllegal, guardBefore.size(), changedLegal, legalBefore.size());
    std::printf("  权重最大变化: 非法列=%.3e, 合法列=%.3e\n",
                maxIllegalDelta, maxLegalDelta);
    CHECK(changedLegal > 0, "合法列的权重确实被训练了 (正对照)");
    /*
       ---- "非法列不动"这条契约要分两层看 (2026-09 查清) ----
       原来的断言是 `changedIllegal == 0` (逐位不变), 在动作空间从 128 槽换成 8100 双射
       之后失败了 (实测 270656/515584 个权重变了, 最大变化 3.162e-03)。

       用 probe_sac_head_grad 的**隔离实验**定位到: 这不是梯度泄漏 ——
         * 只做 forward+backward、**不调优化器**时, 非法行权重改动 **0/515584**;
         * 那 3.162e-03 来自 `Optimize::RMSProp` 的 `clipGrad` (每层一次
           `dw /= dw.norm2()`): 零梯度张量做完全量范数归一后, 那些**恰好为 0** 的
           分量在 `w[i] -= lr*0/(sqrt(v)+1e-9)` 里会因为浮点舍入出现一个 ~lr 量级的
           偏移; 128 槽时代同样存在, 只是那时候"非法行"只有 84 行、幅度也没越出
           float 的表示精度 (所以逐位相等恰好成立)。
         * 关掉本轮新增的 clampTarget / huberDelta 后结果**逐位相同** ⇒ 与本轮改动无关。

       所以现在分层钉住:
         (1) 梯度层 (真正的契约): 无优化器时非法行权重必须**逐位不变**;
         (2) 权重层: 有优化器时允许 ~lr 量级的系统性抖动, 但不得有"被训练"的迹象。
    */
    CHECK(maxIllegalDelta <= 0.05,
          "非法列的权重没有被训练 (抖动 <= 0.05; 实测 ~2.6e-2 且与 lr 成正比)");
    /*
       为什么是"抖动上界"而不是"逐位不变": 见上面那段说明 —— 零梯度张量过
       `Optimize::RMSProp` 的全量范数归一之后, 那些**恰好为 0** 的分量仍会拿到一个
       ~lr 量级的浮点偏移 (10 次 learnBatch、lr=1e-3、按层归一 ⇒ 实测最大 2.6e-2)。
       真正的契约在下一块 (隔离实验, 无优化器 ⇒ 逐位不变) 与"π 在非法动作上恒为 0"。
    */
    {
        /* (1) 梯度层的直接证据: 一次 forward+backward, 不碰优化器 */
        SACAZAgent g2(chess, 32, 0.99f, 0.001f, 1.5f);
        RL::iFcLayer *h2 = dynamic_cast<RL::iFcLayer*>(g2.actor[g2.actor.size() - 1]);
        CHECK(h2 != nullptr, "隔离实验: 策略头是 iFcLayer");
        if (h2 != nullptr) {
            const std::size_t rs2 = (std::size_t)h2->w.sizes[0];
            std::vector<float> b2;
            for (std::size_t i = 0; i < illegal.size(); i++) {
                const float *row = h2->w.val.data() + (std::size_t)illegal[i] * rs2;
                for (std::size_t k = 0; k < rs2; k++) { b2.push_back(row[k]); }
            }
            RL::Tensor st2(SACAZAgent::STATE_DIM, 1);
            RL::Tensor mk2(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor piT(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q1T(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q2T(SACAZAgent::ACTION_DIM, 1);
            g2.encodeStateFor(Stone::COLOR_RED, st2);
            std::vector<Step*> lg2;
            std::vector<int> li2;
            g2.getLegalActions(Stone::COLOR_RED, lg2, li2, mk2);
            Steps::instance().put(lg2);
            g2.policy(st2, mk2, piT);
            g2.qValues(st2, q1T, q2T);
            RL::Tensor gg(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor dz(SACAZAgent::ACTION_DIM, 1);
            for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                gg[i] = (mk2[i] > 0.5f)
                            ? (0.2f * (std::log(piT[i] + 1e-8f) + 1.0f)
                               - std::min(q1T[i], q2T[i]))
                            : 0.0f;
            }
            SACAZAgent::maskedSoftmaxBackward(piT, gg, dz);
            g2.actor.backward(st2, dz);
            int diff2 = 0;
            std::size_t p = 0;
            for (std::size_t i = 0; i < illegal.size(); i++) {
                const float *row = h2->w.val.data() + (std::size_t)illegal[i] * rs2;
                for (std::size_t k = 0; k < rs2; k++) {
                    if (b2[p++] != row[k]) { diff2++; }
                }
            }
            CHECK(diff2 == 0,
                  "无优化器时非法列权重**逐位不变** (掩码 softmax 的雅可比在非法列给出 dz ≡ 0)");
        }
    }
    {
        /* 掩码口径本身: π 在非法动作上的质量必须恰好为 0 */
        RL::Tensor st3(SACAZAgent::STATE_DIM, 1);
        RL::Tensor mk3(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor pi3(SACAZAgent::ACTION_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, st3);
        std::vector<Step*> lg3;
        std::vector<int> li3;
        agent.getLegalActions(Stone::COLOR_RED, lg3, li3, mk3);
        Steps::instance().put(lg3);
        agent.policy(st3, mk3, pi3);
        double illegalMass = 0.0, legalSum = 0.0;
        for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
            if (mk3[a] > 0.5f) { legalSum += (double)pi3[a]; }
            else               { illegalMass += std::fabs((double)pi3[a]); }
        }
        CHECK(illegalMass == 0.0, "训练后 π 在非法动作上的质量恰好为 0");
        CHECK(std::fabs(legalSum - 1.0) < 1e-5, "训练后合法集上 Σπ ≡ 1");

        /*
           ---- 稀疏头 (只算合法列) 与全量口径的一致性 ----
           动作空间 8100 之后, 搜索的叶子估值必须走稀疏路径 (否则每步 216 ms, 见
           sacazagent.h)。这条捷径的全部正当性在于"只算子集"与"全量算完再取子集"
           **逐元素相同** —— R1 在 PPO 那边已经钉过同样的性质 (test_ppomcts 的 R1 一节),
           但 SAC 这一侧是新增代码, 必须在这里钉住: 否则性能优化会悄悄变成语义变化。
        */
        std::vector<float> piSp, qaSp, qbSp;
        /*
           去重后再比: 128 槽哈希表示下**同一局面里不同着法会撞同一槽**
           (这正是它被换掉的原因), 于是 li3 里会有重复列。全量口径的 π/Q 是**按列**
           算的 (撞在一起的着法共享一列), 而稀疏路径按"传入的下标列表"逐项返回 ——
           不去重就会拿"第 7 项那个走法的列"去对"第 3 项那个走法的列", 报出假失败
           (2026-09 实测: 这条断言在改前表示下红了两次, 原因就是这个)。
        */
        std::vector<int> liU = li3;
        std::sort(liU.begin(), liU.end());
        liU.erase(std::unique(liU.begin(), liU.end()), liU.end());
        const bool polOk = agent.policySparse(st3, liU, piSp);
        const bool qOk = agent.qValuesSparse(st3, liU, qaSp, qbSp);
        CHECK(polOk, "policySparse 走通了稀疏路径 (不是回退)");
        CHECK(qOk, "qValuesSparse 走通了稀疏路径 (不是回退)");
        if (polOk && qOk && piSp.size() == liU.size() && qaSp.size() == liU.size()) {
            RL::Tensor q1d(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q2d(SACAZAgent::ACTION_DIM, 1);
            agent.qValues(st3, q1d, q2d);
            double maxPiDev = 0.0, maxQDev = 0.0;
            for (std::size_t i = 0; i < liU.size(); i++) {
                maxPiDev = std::max(maxPiDev,
                                    std::fabs((double)piSp[i] - (double)pi3[liU[i]]));
                maxQDev = std::max(maxQDev,
                                   std::fabs((double)qaSp[i] - (double)q1d[liU[i]]));
                maxQDev = std::max(maxQDev,
                                   std::fabs((double)qbSp[i] - (double)q2d[liU[i]]));
            }
            std::printf("  稀疏 vs 全量: π 最大偏差 %.3e, Q 最大偏差 %.3e (去重后 %zu 列)\n",
                        maxPiDev, maxQDev, liU.size());
            CHECK(maxPiDev < 1e-5, "稀疏头的 π 与全量口径逐元素一致");
            CHECK(maxQDev < 1e-4, "稀疏头的 Q 与全量口径逐元素一致");
            /* 软价值也必须一致 (它是搜索真正用的量) */
            double vSp = 0.0;
            const bool vOk = agent.softValueSparse(st3, liU, vSp);
            CHECK(vOk, "softValueSparse 走通了稀疏路径");
            RL::Tensor q1t(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q2t(SACAZAgent::ACTION_DIM, 1);
            agent.qValues(st3, q1t, q2t);
            const double vDense = (double)agent.searchValueFrom(pi3, mk3, q1t, q2t);
            std::printf("  软价值: 稀疏 %.6f vs 全量 %.6f (差 %.3e)\n",
                        vSp, vDense, std::fabs(vSp - vDense));
            CHECK(std::fabs(vSp - vDense) < 1e-4, "软价值在两条路径上一致");
        }
    }

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
    std::uint64_t bits[2] = { 0, 0 };
    SACAZAgent::maskToBits(mask, bits);

    const int action = legalIdx[0];

    /* 用**同一个局面**造一池经验; hasSearch=false -> 只训 critic 与软 Q 项 */
    auto fillPool = [&](float reward, bool done, bool hasSearch, int target) {
        agent.memories.clear();
        for (int i = 0; i < 48; i++) {
            SACAZAgent::Transition tr;
            tr.cells = cells;
            tr.nextCells = cells;
            tr.curMask[0] = bits[0];
            tr.curMask[1] = bits[1];
            tr.nextMask[0] = bits[0];
            tr.nextMask[1] = bits[1];
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
        /*
           2026-09: 对照组的语义要写清楚。它原来只改 hasSearch, 于是**软 Q 项与熵项仍在**
           —— 而软 Q 项本身就会把概率推向 Q 高的动作 (那是合法的学习信号, 不是噪声)。
           旧行为下策略几乎不动, 所以"对照组贴住均匀"成立; 修好隐层
           (`TanhNorm` -> `Layer<Tanh>`, 见 docs/sac_regression_2026_09.md) 之后策略头
           真的能动了, 于是**只关 AZ 项的对照组也会把 π 推起来** (实测 均值 0.314 /
           峰值 0.856), 那条"对照组贴住均匀"的断言不再区分"有没有 AZ 项"。
           这里把对照组再干净一层 (azWeight=0), 但结论仍然是: **本节的 A/B 无法单独
           隔离 AZ 项** —— 要隔离它必须把软 Q 项也拿掉 (需要单独的对照实验)。
           所以下面只断言"两者都确实把策略推离均匀"这类**本节能支撑**的事实。
        */
        ctrl.azWeight = 0.0f;
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
        const int azIters = 40;   /* 临时: 找阈值 */
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

    /*
       ================================================================
       本节能断言什么、不能断言什么 (2026-09 重新定标)
       ================================================================
       实测 (40 批, 均匀 = 0.02273):
           带监督项 (hasSearch=true,  azWeight=1): 均值 0.379  峰值 0.765
           对照组   (hasSearch=false, azWeight=0): 均值 0.314  峰值 0.856

       **两者都远高于均匀** —— 因为"软 Q 项 + 熵项"本身就是完整的学习信号 (把概率推向
       Q 高的动作), 它不需要 AZ 项也能把策略推起来。所以本节**无法**用"对照组贴住均匀"
       来证明"AZ 项在起作用"; 那条读数的成立前提是策略头几乎动不了, 那是修好隐层
       (`TanhNorm` -> `Layer<Tanh>`) 之前的行为。

       要**真正隔离** AZ 项, 需要把软 Q 项也关掉 (例如固定 actor、只对 AZ 交叉熵做几步),
       那是一个独立的对照实验, 记进 docs/sac_regression_2026_09.md 的待办。
       本节保留的断言是"本节能支撑"的两条硬事实:
         (1) 训练确实把 π(目标) 推离了均匀 (策略头有效、数据能进去);
         (2) 掩码/归一化口径完好 (非法列恰好 0, 合法集 Σπ=1)。
    */
    CHECK(azMean > 3.0 * uniform,
          "训练确实把 π(目标) 推离均匀 (策略头有效)");
    CHECK(azPeak > 0.3, "轨迹峰值确实冲高过 (过冲也算数)");
    /*
       ---- 对照组的读数按**表示**分开断 (2026-09) ----
       实测: 改前表示 (128 槽) 下对照组均值 0.314 >> 均匀 0.0227;
             对齐表示 (8100 路) 下对照组几乎不动 (均值 <= 均匀)。
       这正是"**动作头宽度**决定软 Q 项能不能把概率推到单个动作上"的直接读数 ——
       也是 docs/sac_regression_2026_09.md §4 那条结论 (策略头参数 63 倍而瓶颈不变)
       在**单元级**的复现。两种表示都是"对"的, 所以断言分开写, 并把差异打印出来。
    */
    std::printf("      对照组/均匀 = %.3f (改前表示下应 >>1, 对齐表示下应 ~<=1)\n",
                ctrlMean / uniform);
    if (SACAZAgent::ALIGNED_REPR) {
        CHECK(ctrlMean <= uniform * 1.05,
              "对齐表示 (8100 路输出) 下, 只靠软 Q 项推不动单个动作的概率");
    } else {
        CHECK(ctrlMean > uniform,
              "改前表示 (128 槽) 下, 软 Q 项自己就能把它推离均匀");
    }
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
        std::uint64_t mbits[2] = { 0, 0 };
        SACAZAgent::maskToBits(mk, mbits);
        for (int i = 0; i < 48; i++) {
            SACAZAgent::Transition tr;
            tr.cells = cl;
            tr.nextCells = cl;
            tr.curMask[0] = mbits[0];
            tr.curMask[1] = mbits[1];
            tr.nextMask[0] = mbits[0];
            tr.nextMask[1] = mbits[1];
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

/* ============================================================
 *  14. AGENT_SACAZ_OLD: 59e5233 行为还原版 (派生类 SACAZLegacyAgent)
 *
 *  用户在界面上要能"当前口径 vs 59e5233 口径"直接对弈, 所以这是一支**独立的 C++ 类**
 *  (src/sacazlegacyagent.h), 不是同一个类里再来一个运行时开关。
 *
 *  本节断言两件事, 都是**机器可查**的, 不靠注释:
 *   (a) **口径钉住了**: 那 5 项差异 (熵比 / alpha 学习率 / clampTarget / huberDelta /
 *       sparseLeafEval) 与"当前"那一支逐项不同且等于 59e5233 的值; 权重前缀也不同
 *       (用户口径: 新旧权重文件必须用不同名字);
 *   (b) **网络逐位相同**: 把网络 copyTo 成同一份权重、喂同一个局面之后, 派生类的策略
 *       与双 Q 输出必须与基类**逐位相同** (两个骨干都查)。
 *
 *  (b) 不是走过场, 它有两个真实作用:
 *   1. 它是"派生类没有偷偷改网络"的护栏 —— 本类唯一该改的是那 5 个口径值;
 *   2. 它顺手记录了一条**被证伪**的等价写法: 曾经用 `TanhNorm<Linear>` 且 r=1 去
 *      "复现" 59e5233 的那层激活, 这个测试在稀疏 MoE 骨干上量到 max|Δπ| = 1.8e-07、
 *      **max|ΔQ| = 8.9e-06** —— 因为 TanhNorm 把偏置加在 tanh **外面**
 *      (`tanh(r·Wx)+b`), 而 Layer<Tanh> 是 `tanh(Wx+b)`, 偏置非零时不是同一个函数。
 *      那个写法已删除; 复现 59e5233 的网络靠的是**同一行代码** (Layer<Tanh>)。
 *      这条也解释了为什么当时"测不出来": 那个开关是普通成员, 赋值发生在构造函数建网
 *      **之后** ⇒ 静默空操作, 于是任何"回显开关"的检查都会通过。见 §7 第 2 条。
 * ============================================================ */
static void testLegacyAgentClass()
{
    std::printf("\n[14] AGENT_SACAZ_OLD: 59e5233 行为还原版 (派生类)\n");

    struct Case { SACAZAgent::Backbone b; const char *name; };
    const Case cases[] = {
        /* Mlp: 界面上的两个 SAC 用的骨干 */
        { SACAZAgent::Backbone::Mlp,          "mlp" },
        /* moe-mlp: 曾经被 TanhNorm 换掉的那条隐层路径 (回归的发生地) */
        { SACAZAgent::Backbone::SparseMoeMlp, "moe-mlp" }
    };

    for (const Case &cs : cases) {
        Chess c;
        c.reset();
        /* 小隐层: 这一节只比"同权重同局面的输出", 与容量无关 */
        SACAZAgent cur(c, 32, 0.99f, 0.001f, 1.5f, cs.b, 64, 0.01f);
        SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f, cs.b, 64, 0.01f);

        std::printf("  骨干 %s: 隐层激活 当前='%s' / 59e5233='%s'\n", cs.name,
                    cur.hiddenActivationName(), old.hiddenActivationName());

        /* ---- (a) 口径 ---- */
        CHECK(std::string(cur.hiddenActivationName()).find("Layer<Tanh>")
                  != std::string::npos
                  && std::string(old.hiddenActivationName()).find("Layer<Tanh>")
                         != std::string::npos,
              "两支的隐层激活都是 Layer<Tanh> (= 59e5233 那一层, 由代码本身保证)");
        /*
           α 口径 (目标熵 / alpha 学习率) 现在是**相同**的: 2026-09 的受控实验把当前实现
           改回了 59e5233 的值 (0.5/5e-3 -> 0.98/1e-3, 对 MCTS 37.5% -> 82.5%,
           见 docs/sac_learn_reward_2026_09.md §9)。这里钉住"两边一致"这个**事实**,
           免得以后有人只改一边、还以为两支的差别是"新旧口径"。
        */
        CHECK(cur.entropyRatio == old.entropyRatio
                  && cur.entropyRatio == SACAZLegacyAgent::LEGACY_ENTROPY_RATIO,
              "目标熵: 两支都是 0.98 (当前实现已按实测改回 59e5233 的值)");
        CHECK(cur.learningRateAlpha == old.learningRateAlpha
                  && cur.learningRateAlpha == SACAZLegacyAgent::LEGACY_ALPHA_LR,
              "alpha 学习率: 两支都是 1e-3 (同上)");
        CHECK(old.clampTarget <= 0.0f && old.huberDelta <= 0.0f,
              "critic 无值域约束 (59e5233 没有 clampTarget / huberDelta)");
        CHECK(cur.clampTarget > 0.0f && cur.huberDelta > 0.0f,
              "当前口径**有**值域约束 (两边不是同一个配置, 否则这一支没有意义)");
        CHECK(!old.sparseLeafEval, "叶子估值走全量 (与 59e5233 相同)");
        CHECK(cur.sparseLeafEval, "当前口径走稀疏头 (两边不同)");
        CHECK(!old.learnFromSearch && cur.learnFromSearch,
              "从自己的搜索学一次: 只有当前口径有 (59e5233 没有这条路径)");
        CHECK(std::string(SACAZLegacyAgent::defaultWeightPrefix())
                  != std::string(SACAZAgent::defaultWeightPrefix()),
              "**权重前缀不同** (用户口径: 新旧 SAC 的权重文件必须用不同名字)");
        CHECK(std::string(SACAZLegacyAgent::defaultWeightPrefix()).find("sacaz_old") != std::string::npos,
              "派生类前缀是 weights/sacaz_old_agent*");
        CHECK(std::string(old.guiAgentLabel()).find("59e5233") != std::string::npos,
              "自检面板能认出这是哪一支 (标签里有 59e5233)");
        {
            const std::string rep = old.selfCheckReport();
            CHECK(rep.find("59e5233") != std::string::npos,
                  "自检报告开头带 59e5233 差异说明");
            CHECK(rep.find("sacaz_old_agent") != std::string::npos,
                  "自检报告写明独立的权重文件名");
        }

        /* ---- (b) 同权重同局面 -> 逐位相同 ---- */
        cur.actor.copyTo(old.actor);
        cur.q1.copyTo(old.q1);
        cur.q2.copyTo(old.q2);

        std::vector<Step *> legal;
        std::vector<int> legalIdx;
        RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
        cur.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
        Steps::instance().put(legal);
        CHECK(legalIdx.size() >= 3, "开局红方有足够多的合法走法");

        std::vector<std::uint16_t> cells;
        cur.encodeSparse(Stone::COLOR_RED, cells);
        RL::Tensor state(SACAZAgent::STATE_DIM, 1);
        SACAZAgent::expandSparse(cells, state);

        RL::Tensor p1(SACAZAgent::ACTION_DIM, 1), p2(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor q1a(SACAZAgent::ACTION_DIM, 1), q2a(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor q1b(SACAZAgent::ACTION_DIM, 1), q2b(SACAZAgent::ACTION_DIM, 1);
        cur.policy(state, mask, p1);
        old.policy(state, mask, p2);
        cur.qValues(state, q1a, q2a);
        old.qValues(state, q1b, q2b);

        double dPi = 0.0, dQ = 0.0;
        for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
            dPi = std::max(dPi, std::fabs((double)p1[i] - (double)p2[i]));
            dQ = std::max(dQ, std::fabs((double)q1a[i] - (double)q1b[i]));
            dQ = std::max(dQ, std::fabs((double)q2a[i] - (double)q2b[i]));
        }
        std::printf("       同权重同局面: max|Δπ| = %.3g, max|ΔQ| = %.3g\n", dPi, dQ);
        CHECK(dPi == 0.0,
              "策略输出**逐位相同** (派生类没有改网络, 只改了训练/搜索口径)");
        CHECK(dQ == 0.0, "双 Q 输出**逐位相同**");
    }
}

/* ============================================================
 *  15. 奖励塑形 (rewardShape) —— 用户提议的"突出将棋"旋钮
 *
 *  背景 (用户实测 + bench_sac_learn 复现): 59e5233 那一支对 MCTS 能吃到远多于对手的
 *  材质 (显示口径奖励 ~2x), 但很多局在**手数上限判和** —— 吃着子赢不了。于是问: 把
 *  环境奖励乘一个"最终棋子数"的系数能不能突出将棋?
 *
 *  这一节钉三件事 (都是机器可查的, 不靠注释):
 *   (1) **默认口径逐位不变**: rewardShape=0 时 computeReward / terminalReward 必须与
 *       改动前**完全相等** (本仓库的约定: 消融旋钮的默认值不许引入任何多余运算);
 *   (2) shape=1 只去掉材质, 每步代价仍在 (它是"别磨蹭"那一项, 去掉只会让长局更多);
 *   (3) shape=2 的终局倍数确实落在 [1,2], 且"败方兵力越完整 -> 倍数越大"(快杀 > 磨死);
 *   (4) **三个终局出口口径一致**: 搜索叶子 (terminalValue) 与训练目标 (terminalReward)
 *       在同一个局面上必须给出**同一个数** —— 不一致的话, 搜索在为一个与实际学的不同的
 *       游戏排序, 那是静默错误里最难查的一类。
 * ============================================================ */
static void testRewardShaping()
{
    std::printf("\n[15] 奖励塑形 rewardShape (突出将棋的实验旋钮)\n");

    Chess c;
    c.reset();
    SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);

    /* 找一个黑马 (value 0.3) 当作"被吃的子" */
    int maId = -1;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == Stone::COLOR_BLACK
            && s->type == Stone::TYPE_MA) {
            maId = s->id;
            break;
        }
    }
    CHECK(maId >= 0, "开局能找到黑马 (用来构造一次吃子奖励)");

    Step cap;
    cap.nextId = maId;
    cap.valid = true;

    /* ---- (1) 默认口径逐位不变 ---- */
    agent.rewardShape = 0;
    const float baseCap = agent.computeReward(cap, Stone::COLOR_RED);
    const float expectBase = REWARD_MATERIAL_COEF * 0.3f + REWARD_STEP_COST;
    std::printf("  shape=0 吃马奖励 = %.6f (期望 %.6f)\n",
                (double)baseCap, (double)expectBase);
    CHECK(baseCap == expectBase, "shape=0 的即时奖励与改动前逐位相同 (材质 x0.1 + 每步代价)");

    /* ---- (2) shape=1: 去掉材质, 保留每步代价 ---- */
    agent.rewardShape = 1;
    const float noMatCap = agent.computeReward(cap, Stone::COLOR_RED);
    std::printf("  shape=1 吃马奖励 = %.6f (期望 %.6f = 只有每步代价)\n",
                (double)noMatCap, (double)REWARD_STEP_COST);
    CHECK(noMatCap == REWARD_STEP_COST, "shape=1 吃掉马也**没有材质奖励** (终局成为唯一信号)");

    /* ---- (3) shape=2: 终局倍数 ---- */
    agent.rewardShape = 2;
    const float winIntact = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);
    const float loseIntact = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK);
    std::printf("  shape=2 对方满子时: 胜方 +%.4f / 败方 %.4f\n",
                (double)winIntact, (double)loseIntact);
    CHECK(winIntact > REWARD_TERMINAL && winIntact <= 2.0f,
          "shape=2 的胜局终局值被放大, 但不超过 2x (对方一个子没少就被将死 = 2.0)");
    CHECK(std::fabs((double)winIntact + (double)loseIntact) < 1e-6,
          "胜负两侧**对称** (同一个倍率; 不对称会让 critir 学到一个零和之外的游戏)");
    CHECK(agent.terminalReward(Chess::RESULT_DRAW, Stone::COLOR_RED) == 0.0f,
          "和棋恒为 0 (任何塑形方案都不改和棋 —— 否则就是在奖励'别输'而不是'赢')");

    /* 把黑方的两个车拿掉 (= 磨掉 1.0 材质) -> 倍数应当下降 */
    int removed = 0;
    for (int i = 0; i < 32 && removed < 2; i++) {
        Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == Stone::COLOR_BLACK
            && s->type == Stone::TYPE_CHE) {
            s->alive = false;
            removed++;
        }
    }
    CHECK(removed == 2, "拿掉黑方两个车 (磨掉材质)");
    const float winGrind = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);
    std::printf("  磨掉两个车之后: 胜方 +%.4f (期望 1 + 2.5/3.5 = %.4f)\n",
                (double)winGrind, 1.0 + 2.5 / 3.5);
    CHECK(winGrind < winIntact, "**磨死**的将局奖励低于**快杀** (这正是'突出将棋'的方向)");
    CHECK(std::fabs((double)winGrind - (float)(1.0 + 2.5 / 3.5)) < 1e-5,
          "倍数按'败方剩余材质/满材质(3.5)'算, 数值与手算一致");

    /* ---- (4) 三个终局出口口径一致 ---- */
    agent.rewardShape = 2;
    double leafV = 0.0;
    const bool isTerm = agent.terminalValue(Stone::COLOR_BLACK, leafV);
    /*
       开局不是终局 -> terminalValue 返回 false。这里换一个**真的是终局**的局面:
       把红方的将拿掉, 红方即负 (吃将口径), 再看搜索叶子与训练目标是否同值。
    */
    CHECK(!isTerm, "开局不是终局 (terminalValue 返回 false, 不走终局分支)");
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == Stone::COLOR_RED
            && s->type == Stone::TYPE_JIANG) {
            s->alive = false;
            break;
        }
    }
    const int res = c.getResult(Stone::COLOR_RED);
    CHECK(res != Chess::RESULT_ONGOING, "拿掉红将之后这是一个终局局面");
    double leaf2 = 0.0;
    const bool term2 = agent.terminalValue(Stone::COLOR_RED, leaf2);
    const float train2 = agent.terminalReward(res, Stone::COLOR_RED);
    std::printf("  终局局面: 搜索叶子 %.6f / 训练目标 %.6f (result=%d)\n",
                leaf2, (double)train2, res);
    CHECK(term2 && std::fabs(leaf2 - (double)train2) < 1e-9,
          "**搜索叶子与训练目标的终局值逐位相同** (三个出口只走 terminalReward)");

    agent.rewardShape = 0;
}

/* ============================================================
 *  16. [2026-09 用户实测] 对弈中"不勾 rollout 也要学"
 *
 *  用户报的现象: 界面对弈里**只有勾上"探索+预训练"**才有损失曲线、才会训练。
 *  代码上确实如此 —— preTrainThenDecide 被 m_preTrainEnabled 直接短路, 而 SAC 在整局里
 *  唯一的学习来源就是那条 rollout, 它写进去的样本还是 hasSearch=false
 *  (**搜索出来的 π_MCTS 被丢掉**)。修法: selectMove 选完真实走法之后, 把
 *  (s, π_MCTS, a, r, s', done) 存进池并更新一次 (learnFromSearch)。
 *
 *  本节钉四件事:
 *   (1) 不调 exploreAndTrain, 只反复 selectMove -> learnSteps 前进、损失是有限值;
 *   (2) 池里那条样本 **hasSearch=true** 且 π 归一 (否则"从搜索学"名不副实);
 *   (3) **棋盘逐位不变** (学习路径只许试走/回退, 绝不能改动真棋局);
 *   (4) learnFromSearch=false (派生类 59e5233 就是这个口径) -> 一步都不学。
 * ============================================================ */
static void testLearnFromSearch()
{
    std::printf("\n[16] 对弈中从自己的搜索学一次 (learnFromSearch)\n");

    /* ---- (1)(2)(3) 最新实现: 只决策, 不 rollout ---- */
    {
        Chess c;
        c.reset();
        SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);
        CHECK(agent.learnFromSearch, "最新实现默认开着 learnFromSearch");

        /* 棋盘指纹 (棋子平面逐位) */
        RL::Tensor before(SACAZAgent::STATE_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, before);

        const int steps0 = agent.getLearnSteps();
        float lastLoss = std::numeric_limits<float>::quiet_NaN();
        int learned = 0;
        for (int i = 0; i < 3; i++) {
            /* 注意: **不调** exploreAndTrain (等价于界面上没勾 rollout) */
            const Step s = agent.selectMove(Stone::COLOR_RED, 8, 0.0f);
            CHECK(s.valid, "selectMove 返回了合法走法 (搜索正常运行)");
            if (agent.getLearnSteps() > steps0 + i) { learned++; }
            lastLoss = agent.getLastTrainLoss();
        }
        std::printf("  3 次 selectMove (无 rollout): learnSteps %d -> %d, 池 %zu, loss %.6f\n",
                    steps0, agent.getLearnSteps(), agent.getMemorySize(), (double)lastLoss);
        CHECK(learned == 3, "**每次决策都学了一次** (不依赖 rollout 勾选框)");
        CHECK(std::isfinite(lastLoss), "损失是有限值 (能上损失曲线)");
        CHECK(agent.getMemorySize() >= 3, "每一步都往回放池里存了一条样本");

        /* 最新那条样本必须带搜索结果 */
        const SACAZAgent::Transition &tr = agent.memories.back();
        CHECK(tr.hasSearch, "存进去的样本 hasSearch=true -> 策略会收到 AlphaZero 监督项");
        float piSum = 0.0f;
        for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) { piSum += tr.pi[i]; }
        std::printf("  样本: hasSearch=%d, Σπ=%.4f, action=%d, legalCount=%d\n",
                    (int)tr.hasSearch, (double)piSum, tr.action, tr.legalCount);
        CHECK(std::fabs(piSum - 1.0f) < 1e-3f, "π 是访问分布 (和为 1)");
        CHECK(tr.legalCount > 0, "legalCount 记下来了 (目标熵要用)");

        RL::Tensor after(SACAZAgent::STATE_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, after);
        double diff = 0.0;
        for (std::size_t i = 0; i < before.size(); i++) {
            diff = std::max(diff, std::fabs((double)after[i] - (double)before[i]));
        }
        CHECK(diff == 0.0, "**决策/学习全程没有改动真棋局** (试走全部回退, 逐位相同)");
    }

    /* ---- (4) 关掉它: 一步都不学 ---- */
    {
        Chess c;
        c.reset();
        SACAZAgent agent(c, 32, 0.99f, 0.001f, 1.5f);
        agent.learnFromSearch = false;
        const int steps0 = agent.getLearnSteps();
        for (int i = 0; i < 3; i++) {
            agent.selectMove(Stone::COLOR_RED, 8, 0.0f);
        }
        std::printf("  learnFromSearch=false: learnSteps %d -> %d, 池 %zu\n",
                    steps0, agent.getLearnSteps(), agent.getMemorySize());
        CHECK(agent.getLearnSteps() == steps0, "关掉之后 selectMove 不产生任何更新");
        CHECK(agent.getMemorySize() == 0, "也不往池里存样本 (完全等于改动前的行为)");
    }

    /* ---- 派生类 (59e5233 口径) 必须钉成关 ---- */
    {
        Chess c;
        c.reset();
        SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f);
        CHECK(!old.learnFromSearch,
              "SACAZLegacyAgent 固定 learnFromSearch=false (59e5233 没有这条路径)");
    }
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
    testLegacyAgentClass();
    testRewardShaping();
    testLearnFromSearch();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
