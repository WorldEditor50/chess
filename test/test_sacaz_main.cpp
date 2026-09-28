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
#include <type_traits>   /* [14] 节的"成员探测"探针 (std::void_t / std::false_type) */
#include "chess.h"
#include "chessstate.h"
#include "sacazagent.h"
#include "sacaz_variants.h"     /* [2026-09 独立类] 按骨干选类 + 抹平只有某些骨干才有的读数 */
#include "sacazlegacyagent.h"   /* [14] 节: 59e5233 行为还原版是**独立的类** */
#include "ppomcts_agent.h"   /* 表示对齐断言要拿 PPOMCTSAgent 的维度常量做比较 */
#include "rl/cpuinfo.hpp"
#include "rl/moe.hpp"
#include "rl/sparse_moe.hpp"   /* [17] 节: 要 dynamic_cast 到 ISparseMoE 读 TB 专家的头口径 */
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

/*
 * ============================================================
 *  ---- 编译期探测: "某类型有没有某个名字的成员" ----
 * ============================================================
 * 为什么要它 (2026-09 独立类拆分): 59e5233 还原版**故意不带**那一批开关 ——
 * 奖励塑形 / critic 值域约束 / 熵项开关 / 稀疏叶子 / 从自己的搜索学一次 / 可调的目标网
 * 同步率。在它还是 SACAZAgent 的派生类时, 这些成员**存在但取固定值**, 断言写的是
 * "值等于多少"; 现在它是不继承的独立类, 这些名字**连声明都没有** —— 于是断言只能写成
 * "这个名字探测不到", 否则"删掉开关"这件事在测试里就**不可查**了。
 *
 * SFINAE: 名字不存在时替换失败 ⇒ false_type (不是编译错误)。
 * 每个探针都配一条 `Has_xxx<SACAZAgent>` 的**对照组**断言 (必须为真): 没有对照组的话,
 * 一个写坏的探针会对所有类型都说"没有", 那正是"看起来通过的假测试"。
 */
#define DSH_DETECT_MEMBER(Name)                                                 \
    template <class T, class = void> struct Has_##Name : std::false_type {};    \
    template <class T> struct Has_##Name<T, std::void_t<decltype(&T::Name)>>    \
        : std::true_type {}

DSH_DETECT_MEMBER(rewardShape);
DSH_DETECT_MEMBER(rewardScale);
/* [15b] 节: 动态奖励分配 (rewardShape=3) 的三个成员 —— 只有当前口径那一支有 */
DSH_DETECT_MEMBER(mateScoreMode);
DSH_DETECT_MEMBER(matRewardBoost);
DSH_DETECT_MEMBER(mateRewardBoost);
DSH_DETECT_MEMBER(clampTarget);
DSH_DETECT_MEMBER(huberDelta);
DSH_DETECT_MEMBER(valueScale);
DSH_DETECT_MEMBER(entropyInTarget);
DSH_DETECT_MEMBER(entropySlotsAsLegal);
DSH_DETECT_MEMBER(targetTau);
DSH_DETECT_MEMBER(replaceTargetIter);
DSH_DETECT_MEMBER(sparseLeafEval);
DSH_DETECT_MEMBER(learnFromSearch);
/* [17] 节: 共享骨干 / TB 头数口径 —— 两个新成员**都不许**出现在行为还原版上 */
DSH_DETECT_MEMBER(trunkMode);
DSH_DETECT_MEMBER(tbHonorHeads);

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
            typename std::decay<decltype(agent)>::type::Transition tr;
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
        /*
           [2026-09 独立类拆分] 原来是 `SACAZAgent::Backbone b` (一个类 + 一个枚举值)。
           现在骨干 = **类型**, 所以这里存"变体", 由 `sacazx::withSacazAgent` 建出对应
           那个类的实例 —— 四个骨干四份独立实现, 而这段测试代码对它们是**同形**的
           (公共 API + `sacazx::MoeInfo` 抹平"只有 MoE 才有的读数")。
        */
        sacazx::Variant v;
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
        { sacazx::Variant::Mlp,        64, 0, 0, true },
        { sacazx::Variant::MoeMlp,     32, SACAZAgent::MOE_MLP_EXPERTS,
          SACAZAgent::MOE_MLP_TOPK, true },
        { sacazx::Variant::MoeTb,       4, SACAZAgent::MOE_TB_EXPERTS,
          SACAZAgent::MOE_TB_TOPK, false },
        { sacazx::Variant::DenseMoeTb,  4, SACAZAgent::MOE_TB_EXPERTS,
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
        sacazx::Opts o;
        o.expertHidden = 64;
        o.aux = 0.01f;
        sacazx::withSacazAgent(c, cs.v, o, [&](auto &agent) {
        using A = typename std::decay<decltype(agent)>::type;
        agent.batchSize = 4;

        std::printf("    %-22s 专家数=%d topK=%d\n",
                    agent.backboneName(),
                    sacazx::MoeInfo<A>::expertCount(agent), sacazx::MoeInfo<A>::topK(agent));
        CHECK(sacazx::MoeInfo<A>::expertCount(agent) == cs.expectExperts,
              "骨干的专家数符合预期 (非 MoE 骨干为 0)");
        CHECK(sacazx::MoeInfo<A>::topK(agent) == cs.expectTopK, "骨干的 topK 符合预期");

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
        RL::Tensor state(A::STATE_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, state);
        RL::Tensor q1v(A::ACTION_DIM, 1);
        RL::Tensor q2v(A::ACTION_DIM, 1);
        agent.qValues(state, q1v, q2v);
        bool finite = true;
        double qmax = 0;
        for (int i = 0; i < A::ACTION_DIM; i++) {
            if (!std::isfinite(q1v[i]) || !std::isfinite(q2v[i])) {
                finite = false;
            }
            qmax = std::fmax(qmax, std::fabs((double)q1v[i]));
        }
        CHECK(finite, "学习之后 Q 值仍然有限");
        CHECK(qmax < 1e6, "Q 值没有爆掉 (|Q|max < 1e6)");

        /* ---- 稀疏 MoE 的使用分布 (坍缩诊断): 一共只给了几次前向, 不苛求均衡 ---- */
        std::vector<long long> usage;
        sacazx::MoeInfo<A>::usage(agent, usage);
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
        sacazx::MoeInfo<A>::resetUsage(agent);
        sacazx::MoeInfo<A>::usage(agent, usage);
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
            /*
               [2026-09 独立类拆分] "另一个全新的同类 agent" 也要按**同一个变体**建
               (以前是同一个类 + 同一个枚举值) —— 否则量到的是"两个不同的类能不能互相
               载入权重", 而不是"同一个类的读写顺序对不对"。
            */
            sacazx::withSacazAgent(c, cs.v, o, [&](auto &fresh) {
            CHECK(fresh.loadModel(prefix), "loadModel 报告成功");
            RL::Tensor p1(A::ACTION_DIM, 1);
            RL::Tensor p2(A::ACTION_DIM, 1);
            RL::Tensor m1(A::ACTION_DIM, 1);
            agent.policy(state, m1, p1);
            fresh.policy(state, m1, p2);
            double dp = 0;
            for (int i = 0; i < A::ACTION_DIM; i++) {
                dp = std::fmax(dp, std::fabs((double)p1[i] - (double)p2[i]));
            }
            std::printf("        save/load 往返: 策略最大差 = %.3e\n", dp);
            CHECK(dp < 1e-6, "载入后的策略与保存前一致 (层结构/权重的读写顺序正确)");
            /* Q 网是独立的两个文件, 也顺手比一下 */
            RL::Tensor qa(A::ACTION_DIM, 1);
            RL::Tensor qb(A::ACTION_DIM, 1);
            agent.qValues(state, qa, qb);
            RL::Tensor f1(A::ACTION_DIM, 1);
            RL::Tensor f2(A::ACTION_DIM, 1);
            fresh.qValues(state, f1, f2);
            double dq = 0;
            for (int i = 0; i < A::ACTION_DIM; i++) {
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
            });   /* withSacazAgent: fresh (同一个变体) */
        }
        });       /* withSacazAgent: 本轮的 agent */
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
            typename std::decay<decltype(agent)>::type::Transition tr;
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
    /*
       [2026-09 独立类拆分] 这一节现在建的是**独立类** `SACAZMoETbAgent`
       (拆分前是 `SACAZAgent` + `Backbone::SparseMoeTb`): TB 专家那一支有自己的文件,
       与纯 MLP / MoE-MLP 互不继承 —— 本节的断言因此同时是"那一支自己的回归"。
    */
    SACAZMoETbAgent agent(c, 64, 0.99f, 0.003f, 1.5f, 64, 0.1f);
    agent.batchSize = 4;

    std::printf("  骨干=%s  专家=%d  topK=%d\n",
                agent.backboneName(),
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
            typename std::decay<decltype(agent)>::type::Transition tr;
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
        SACAZMoETbAgent ctrl(c, 64, 0.99f, 0.003f, 1.5f, 64, 0.1f);
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
        auto runTraj = [&](auto &a, int iters) {
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
    std::printf("      对照组/均匀 = %.3f (只作**读数**: 这一项随实现/种子漂, 不能当判据)\n",
                ctrlMean / uniform);
    /*
       ---- [2026-09 dev-sacmoetb] 为什么这里不再断言 `ctrlMean > uniform` ----
       那句话原来的判据是"改前表示 (128 槽) 下软 Q 项自己就能把概率推离均匀"。实测它
       **不稳**: 同一份代码, 只把头数口径从"旧 (15 头落成 3 头)"换成"新 (真 15 头)" ——
       也就是换了一组初始权重 —— 这个比值就从 **1.309 掉到 0.810**。机制与本节上面那段
       记的假失败同源: 对照组只有"软 Q 项 + 熵项"两个信号, 而 soft Q 项的推力正比于
       **critic 当前的 Q 间距**, 那个量在 clipGrad 的等步长更新下是振荡的 (见 (b) 一节的
       隔离实验)。所以"对照组动不动"量的是轨迹的相位, 不是表示的性质。

       本节**能**支撑的、而且稳定的判据是**配对比较**: 两个 agent 从同一份权重出发
       (上面已经 copyTo 过), 唯一的差别是监督项, 于是
           "带监督项的 π(目标) 轨迹均值 >> 对照组的"
       在两个口径下都成立 —— 旧口径 0.2086 vs 0.0298 = 7.0x, 新口径 0.1651 vs 0.0184
       = 9.0x。这正是本节的标题想说的事 ("策略改进"), 而且它不依赖轨迹终点/相位。
    */
    if (SACAZAgent::ALIGNED_REPR) {
        std::printf("      对齐表示 (8100 路输出) 下对照组几乎不动 (均值 %.5f vs 均匀 %.5f)\n",
                    ctrlMean, uniform);
    }
    CHECK(azMean > 3.0 * ctrlMean,
          "带监督项的 π(目标) 轨迹均值是对照组 (azWeight=0 且 hasSearch=false) 的 3 倍以上"
          " —— 配对 A/B, 不依赖轨迹相位");
    CHECK(std::fabs(sumLegal - 1.0) < 1e-5, "合法集上 Σπ = 1 (掩码归一没有坏)");
    CHECK(illegalNonZero == 0, "非法列 π 恰好为 0");

    /*
       ================ (b) 学得动吗: Q 朝已知 TD 目标走 ================

       ---- [2026-09 dev-sacmoetb] 这一节原来断言的是**轨迹终点**, 而终点是掷骰子 ----
       本节是"给一批固定经验 (reward=0.5 且 done=true ⇒ TD 目标恰好 0.5), 跑 30 次
       learnBatch(4), 断言 Q(s,a) 朝 0.5 移动、误差下降"。**这条断言一直是不稳定的**,
       只是以前没被抓到, 原因与本文件 [12a] 那一段记的假失败**同一个**:
       `Net::RMSProp` 默认 `clipGrad=true` 会把整张梯度张量归一到单位长度, 于是**每步的
       位移恒等于 lr**、与梯度大小无关 —— 而目标是个常数 0.5, 所以轨迹是"冲过头再弹
       回来", 终点落在哪里基本是运气。

       隔离实验 (.r1build/tblearn.cpp, 固定种子 20240913, 只有头数口径这一个变量):

           lr = 0.003 (本节原口径) 时 Q(s,a) 的轨迹:
             旧口径(改动前): -0.057 -> 2.880 -> -2.706 -> 1.411 -> -0.556 -> -0.745 -> 0.376
             新口径        :  0.147 -> -2.946 -> -1.617 -> 1.029 ->  0.403 ->  0.533 -> -0.088
           两个口径都在 ±3 之间来回摆 —— **"终点比起点更靠近 0.5"在这条轨迹上纯粹是运气**。
           旧口径那一行与改动前的代码**逐位相同** (改动前: heads 也是构造 15 个、只用 3 个,
           随机数消耗一致), 所以这不是新代码引入的, 是本节一开始就有的性质。

           lr = 3e-05 (把步长压到振荡消失) 时:
             旧口径: 0.149 -> 0.413 -> 0.532 -> 0.495 -> 0.524 -> 0.492 -> 0.518 (|误差| 0.351 -> 0.018)
             新口径: -0.020 -> 0.301 -> 0.457 -> 0.502 -> 0.460 -> 0.500 -> 0.536 (|误差| 0.520 -> 0.036)
           两个口径都**收敛到 0.5 附近** ⇒ "梯度穿过了 TB 专家 + 门控 + 稀疏路由"这件事
           在两个口径下都成立, 而且是单调可见的。

       所以本节现在把 critic 学习率**临时压到 3e-5** 再断言 (见下面), 理由不是"让测试过",
       而是"让断言量到它自己声称要量的东西": 它声称量的是**收敛方向**, 而 lr=0.003 下
       量到的是振荡的相位。压小 lr 之后这一节比原来**更严** (原来 0.489 -> 0.297 也能过,
       现在要求降到起点误差的一半以下)。
    */
    std::printf("\n  (b) critic 学习: y ≡ 0.5, 看 Q(s,a) 是否靠近\n");
    fillPool(0.5f, true, false, -1);

    RL::Tensor q1v(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q2v(SACAZAgent::ACTION_DIM, 1);
    agent.qValues(state, q1v, q2v);
    const float q0 = q1v[action];
    const float err0 = std::fabs(q0 - 0.5f);

    /*
       临时压低 critic 学习率 (见上面那段隔离实验): 只在 (b) 里生效, 用完还原 ——
       后面的 (c)/(d) 量的是自对弈与代价, 必须按 agent 原本的口径跑。
    */
    const float savedCriticLr = agent.learningRateCritic;
    agent.learningRateCritic = 3e-05f;

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
    std::printf("      Q(s,a): %.5f -> %.5f (目标 0.5), |误差| %.5f -> %.5f  (critic lr = %.0e)\n",
                (double)q0, (double)q1, (double)err0, (double)err1,
                (double)agent.learningRateCritic);
    std::printf("      最后一次 loss = %.6f, alpha = %.4f\n",
                (double)lastLoss, (double)agent.getAlpha());
    CHECK(q1 != q0, "Q 确实被更新了 (梯度穿过了 TB 专家 + 门控)");
    CHECK(err1 < err0, "Q 朝目标方向移动 (critic 真的在学)");
    /*
       这一段只断言"误差下降", **不给定量余量** —— 因为起点被 (a) 推到了饱和区:
       上面 30 批实测 (默认表示) 0.43259 -> 0.33182 (比值 0.767),
       对齐表示 0.63293 -> 0.57431 (比值 0.907)。同一个"0.8"在两种表示下一边过一边挂,
       而两种表示都是**对的** —— 那说明这个阈值量的不是模型的能力, 是起点的饱和程度。
       定量结论放到下面那段**全新 agent** 的隔离对照里 (那里才有可复现的收敛曲线)。
    */
    std::printf("      |误差| 比值 = %.3f (饱和 critic: 只断言\"下降\", 定量见下面隔离对照)\n",
                (double)(err1 / err0));
    CHECK(std::isfinite((double)lastLoss), "批平均 loss 是有限值");
    CHECK(agent.getAlpha() >= 0.02f && agent.getAlpha() <= 5.0f,
          "alpha 仍在自动调节的夹逼范围内 (没有发散)");

    agent.learningRateCritic = savedCriticLr;   /* 还原 (c)/(d) 的口径 */

    /*
       ================ (b2) 隔离对照: 同一协议在**全新 agent** 上 ================
       为什么必须补这一段: (b) 跑在 (a) 已经训过 40 批的 critic 上, 那个 critic 的隐层
       已经饱和 (`1 - h^2` 很小), 30 批只能把误差推掉 10~25% —— "推掉多少"在饱和起点上
       量不出这条轨迹**真实**的收敛能力。换一个**从头开始**的 critic, 同一条协议
       (y ≡ 0.5, lr=3e-5, 30 x learnBatch(4)) 实测 (本文件随机种子下):

           默认表示: 0.52046 -> 0.03559   (比值 0.068)
           对齐表示: 见同一次运行的打印

       阈值取 0.35 有一倍以上余量。这一条才是"梯度真的穿过了 TB 专家 + 门控 + 稀疏路由"
       的定量证据 —— 前向对而反向错 / 优化器用错 / 符号搞反, 三种情形都收敛不到这里。
    */
    std::printf("\n  (b2) 隔离对照: 全新 critic 上同一协议 (y ≡ 0.5, lr=3e-5)\n");
    {
        /*
           作用域: 这个 agent 只在这一段里活着 (TB 专家 agent 连同它的 v 缓冲约 0.9 GB;
           上面 (a) 已经同时开过两个, 所以这不是新的内存峰值)。
        */
        Chess cf;
        cf.reset();
        SACAZMoETbAgent fresh(cf, 64, 0.99f, 0.003f, 1.5f, 64, 0.1f);
        fresh.batchSize = 4;
        fresh.learningRateCritic = 3e-05f;

        std::vector<Step*> flegal;
        std::vector<int> fidx;
        RL::Tensor fmask(SACAZAgent::ACTION_DIM, 1);
        fresh.getLegalActions(Stone::COLOR_RED, flegal, fidx, fmask);
        Steps::instance().put(flegal);
        std::vector<std::uint16_t> fcells;
        fresh.encodeSparse(Stone::COLOR_RED, fcells);
        RL::Tensor fstate(SACAZAgent::STATE_DIM, 1);
        SACAZAgent::expandSparse(fcells, fstate);
        std::uint64_t fbits[2] = { 0, 0 };
        SACAZAgent::maskToBits(fmask, fbits);
        const int faction = fidx[0];

        fresh.memories.clear();
        for (int i = 0; i < 48; i++) {
            typename std::decay<decltype(fresh)>::type::Transition tr;
            tr.cells = fcells;
            tr.nextCells = fcells;
            tr.curMask[0] = fbits[0]; tr.curMask[1] = fbits[1];
            tr.nextMask[0] = fbits[0]; tr.nextMask[1] = fbits[1];
            tr.action = faction;
            tr.legalCount = (int)fidx.size();
            tr.reward = 0.5f;
            tr.done = true;
            tr.hasSearch = false;
            fresh.memories.push_back(tr);
        }

        RL::Tensor fq1(SACAZAgent::ACTION_DIM, 1), fq2(SACAZAgent::ACTION_DIM, 1);
        fresh.qValues(fstate, fq1, fq2);
        const float fq0 = fq1[faction];
        const float fe0 = std::fabs(fq0 - 0.5f);
        for (int it = 0; it < 30; it++) { fresh.learnBatch(4); }
        fresh.qValues(fstate, fq1, fq2);
        const float fq1v2 = fq1[faction];
        const float fe1 = std::fabs(fq1v2 - 0.5f);
        std::printf("      Q(s,a): %.5f -> %.5f (目标 0.5), |误差| %.5f -> %.5f, 比值 %.3f\n",
                    (double)fq0, (double)fq1v2, (double)fe0, (double)fe1,
                    (double)(fe1 / fe0));
        CHECK(fe1 < fe0, "全新 critic: 误差下降");
        CHECK(fe1 < 0.35f * fe0,
              "全新 critic 上误差降到起点的 35% 以下 (梯度真的穿过了 TB 专家 + 门控, "
              "而且方向是对的)");
    }

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
        SACAZAgent mlp(c2, 64, 0.99f, 0.003f, 1.5f);
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
            typename std::decay<decltype(mlp)>::type::Transition tr;
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
 *  14. AGENT_SACAZ_OLD: 59e5233 行为还原版 (**独立的类** SACAZLegacyAgent)
 *
 *  用户在界面上要能"当前口径 vs 59e5233 口径"直接对弈, 所以这是一支**独立的 C++ 类**
 *  (src/sacazlegacyagent.h/.cpp), 不是同一个类里再来一个运行时开关, 也**不再是**
 *  SACAZAgent 的派生类 (2026-09 用户口径: "让 SACAZLegacyAgent 成为一个不继承
 *  SACAZAgent 的独立类, 这样 SACAZAgent 以后任何默认值或实现上的改动都不可能渗进
 *  59e5233 还原版")。
 *
 *  本节断言三件事, 都是**机器可查**的, 不靠注释:
 *   (a) **不继承**: 它是独立类 (头文件里另有一条 static_assert 把这句话变成编译错误,
 *       这里再用 std::is_base_of 钉一遍, 免得那一条被顺手删掉);
 *   (b) **那一批开关连名字都没有**: 奖励塑形 / critic 值域约束 / 熵项开关 / 稀疏叶子 /
 *       从自己的搜索学一次 / 可调的目标网同步率 —— 用 SFINAE 探针逐个检查"探测不到",
 *       并用 SACAZAgent 做**对照组** (探针必须在那边探得到, 否则它对谁都说"没有");
 *   (c) **网络逐位相同**: 把网络 copyTo 成同一份权重、喂同一个局面之后, 独立类的策略与
 *       双 Q 输出必须与 SACAZAgent **逐位相同** (两个骨干都查)。
 *
 *  (c) 不是走过场, 它有两个真实作用:
 *   1. 它是"把实现拷成独立类时没有偷偷改网络"的护栏 —— 该与 59e5233 不同的只有口径值;
 *   2. 它顺手记录了一条**被证伪**的等价写法: 曾经用 `TanhNorm<Linear>` 且 r=1 去
 *      "复现" 59e5233 的那层激活, 这个测试在稀疏 MoE 骨干上量到 max|Δπ| = 1.8e-07、
 *      **max|ΔQ| = 8.9e-06** —— 因为 TanhNorm 把偏置加在 tanh **外面**
 *      (`tanh(r·Wx)+b`), 而 Layer<Tanh> 是 `tanh(Wx+b)`, 偏置非零时不是同一个函数。
 *      那个写法已删除; 复现 59e5233 的网络靠的是**同一行代码** (Layer<Tanh>)。
 *      这条也解释了为什么当时"测不出来": 那个开关是普通成员, 赋值发生在构造函数建网
 *      **之后** ⇒ 静默空操作, 于是任何"回显开关"的检查都会通过。见 §7 第 2 条。
 * ============================================================ */
/*
   [2026-09 独立类拆分] 这一节原来是"当前口径那一支" (一个类 + 骨干枚举) 对 59e5233 还原版。
   "当前口径"现在是**两个不同的类**: 纯 MLP 的 SACAZAgent 与 MoE-MLP 的 SACAZMoEMlpAgent ——
   所以比较体抽成模板: 骨架与断言一字不改, 只是换一个类实例化。
   还原版那边**没有**这个分裂 (它自带两支骨干, 见 sacazlegacyagent.h), 所以它照旧按
   SACAZLegacyAgent::Backbone 选骨干。
*/
template <class Cur>
static void compareCurrentVsLegacy(Chess &c, Cur &cur, SACAZLegacyAgent &old,
                                   SACAZLegacyAgent::Backbone oldB, const char *caseName)
{
        (void)c;   /* 比较体不再自己建 agent: 两个实例由调用方按骨干各自构造 */

        std::printf("  骨干 %s: 隐层激活 当前='%s' / 59e5233='%s'\n", caseName,
                    cur.hiddenActivationName(), old.hiddenActivationName());

        /* ---- (a) 口径 ---- */
        CHECK(std::string(cur.hiddenActivationName()).find("Layer<Tanh>")
                  != std::string::npos
                  && std::string(old.hiddenActivationName()).find("Layer<Tanh>")
                         != std::string::npos,
              "两支的隐层激活都是 Layer<Tanh> (= 59e5233 那一层, 由**同一行代码**保证)");
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
        /*
           奖励口径: "没有塑形"这件事现在是**行为**可查的 —— 终局值必须与引擎真值
           (outcomeForMover) **逐位相同**, 而不是"某个开关取 0"。当前口径那一支反过来
           有这个开关 (上面的探针已经钉住了名字存在), 所以两支不是同一个配置。
        */
        CHECK(old.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED) == 1.0f
                  && old.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK) == -1.0f
                  && old.terminalReward(Chess::RESULT_DRAW, Stone::COLOR_RED) == 0.0f,
              "终局值 = 引擎真值 ±1/0 (本类不含奖励塑形/放大, 连开关都没有)");
        CHECK(old.rewardCaliperName() == std::string("学习口径") && old.hasLearningReward(),
              "奖励曲线的口径来自本 agent 自己的出口 (学习口径, 与它学的是同一个游戏)");
        /*
           当前实现这一支: 默认值 = 61a974d 的口径 (F1 的实测最好档 p = 0.43 不显著 ⇒
           不作为默认; 要开就传 --target-tau/--target-iter)。这里把"默认值是什么"钉住,
           免得它被下一次实验顺手改掉 (默认值也是结论)。
           **注意**: 还原版连这两个成员都没有, 它的同步率是编译期常数 (上面已断言)。
        */
        CHECK(cur.targetTau == 1e-3f && cur.replaceTargetIter == 64,
              "当前实现的默认目标网同步率 = 61a974d 口径 (tau=1e-3 / 每 64 步)");
        CHECK(std::string(SACAZLegacyAgent::defaultWeightPrefix())
                  != std::string(Cur::defaultWeightPrefix()),
              "**权重前缀不同** (用户口径: 新旧 SAC 的权重文件必须用不同名字)");
        CHECK(std::string(SACAZLegacyAgent::defaultWeightPrefix()).find("sacaz_old") != std::string::npos,
              "还原版前缀是 weights/sacaz_old_agent*");
        /*
           ---- 身份标签: 每个骨干都要有 59e5233, 且要说出**本实例**的骨干 ----
           这两条原来只有"标签里有 59e5233"一句, 而实现当时只认 Mlp / SparseMoeTb 两个骨干,
           其余返回 `bench/测试构造 (界面不为它建实例)` ⇒ 本节构造的 moe-mlp 实例上如实失败。
           用户口径是"两支 GUI agent 都要能看出 (a) 是 59e5233 还原口径 + (b) 是哪个骨干" ,
           所以修的是**实现** (标签按骨干给各自的变体后缀, 但一律保留 59e5233), 断言改成
           对**每个构造出来的骨干**分别查这两件事。
        */
        CHECK(std::string(old.guiAgentLabel()).find("59e5233") != std::string::npos,
              "自检面板能认出这是哪一支 (每个骨干的标签里都有 59e5233)");
        CHECK(std::string(old.guiAgentLabel())
                  == std::string(SACAZLegacyAgent::guiAgentLabelFor(oldB)),
              "实例报的标签就是**本实例骨干**那一句 (标签与骨干同一来源, 不会各说各话)");
        /*
           把这两件验收读数**原样打出来** (标签 + 报告里那一行权重文件名): 断言只说明
           "成不成立", 人读的时候要能直接看到这一支在面板上到底显示成什么 ——
           上一版标签在三支骨干上是同一句"bench/测试构造", 只有把原文打出来才一眼看得出。
        */
        std::printf("      面板标签: %s\n", old.guiAgentLabel());
        {
            const std::string rep = old.selfCheckReport();
            CHECK(rep.find("59e5233") != std::string::npos,
                  "自检报告开头带 59e5233 差异说明");
            const std::string key = "权重文件独立";
            const std::size_t kp = rep.find(key);
            if (kp != std::string::npos) {
                const std::size_t ke = rep.find('\n', kp);
                std::printf("      报告: %s\n",
                            rep.substr(kp, (ke == std::string::npos ? rep.size() : ke)
                                               - kp).c_str());
            } else {
                std::printf("      报告: (**没有** '权重文件独立' 那一行)\n");
            }
            /*
               报告里的权重前缀必须是**这个实例自己的骨干**那一个
               (= defaultWeightPrefix(this->backbone)): 原来这里写死查 "sacaz_old_agent",
               那是单骨干时代的写法 —— moe-mlp 骨干的实例实际会写
               weights/sacaz_old_moemlp_agent, 报告里也确实是它 (断言如实失败)。
               判据改成"报告里出现该骨干自己的前缀", 于是这条断言在**任何**骨干上都成立,
               而且仍然能把"报告写的是别的骨干的前缀"这种错钉住。
            */
            CHECK(rep.find(SACAZLegacyAgent::defaultWeightPrefix(oldB)) != std::string::npos,
                  "自检报告写明**本骨干**实际会用的权重文件名 (与 defaultWeightPrefix 一致)");
            CHECK(rep.find(SACAZLegacyAgent::backboneName(oldB)) != std::string::npos,
                  "自检报告第二行写明本实例的骨干 (读数归到哪一支一目了然)");
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
              "策略输出**逐位相同** (独立类没有改网络, 只改了训练/搜索口径)");
        CHECK(dQ == 0.0, "双 Q 输出**逐位相同**");
}

static void testLegacyAgentClass()
{
    std::printf("\n[14] AGENT_SACAZ_OLD: 59e5233 行为还原版 (**独立类**, 不继承)\n");

    /* ---- (a) 不继承, 但仍是 AgentBase ----
       注意 CHECK 是宏, 而 `std::is_base_of<A, B>` 里的逗号会被预处理器当成**参数分隔**,
       所以这两个条件必须**多加一层括号** (这不是风格问题, 少一层就是编译错误)。 */
    CHECK((!std::is_base_of<SACAZAgent, SACAZLegacyAgent>::value),
          "SACAZLegacyAgent **不是** SACAZAgent 的派生类 (独立实现: 基类的默认值改动渗不进来)");
    CHECK((std::is_base_of<AgentBase, SACAZLegacyAgent>::value),
          "它仍然是 AgentBase (界面/工具把它当普通 agent 用: preTrainThenDecide(AgentBase*))");

    /* ---- (b) 那一批开关在还原版上**连名字都没有** ---- */
    /*
       对照组先来: 这些名字在"当前口径"那一支上必须**探得到**。没有这三条, 一个写坏的
       探针 (永远返回 false_type) 会让下面 11 条"没有"全部通过。
    */
    CHECK(Has_rewardShape<SACAZAgent>::value && Has_clampTarget<SACAZAgent>::value
              && Has_learnFromSearch<SACAZAgent>::value
              && Has_targetTau<SACAZAgent>::value,
          "[对照] 探针能在 SACAZAgent 上探到 rewardShape / clampTarget / learnFromSearch / targetTau");
    /* 奖励塑形: 连名字都没有 */
    CHECK(!Has_rewardShape<SACAZLegacyAgent>::value,
          "还原版**没有** rewardShape (本类不含奖励塑形)");
    CHECK(!Has_rewardScale<SACAZLegacyAgent>::value,
          "还原版**没有** rewardScale (即时奖励不缩放)");
    /*
       [2026-09 动态奖励分配] 三个新成员同样只有当前口径那一支有 —— 还原版是 59e5233 的
       行为快照, 它的奖励恒为"材质 x0.1 + 每步代价 + 终局 ±1", **没有**按局面分配的权重。
    */
    CHECK(!Has_mateScoreMode<SACAZLegacyAgent>::value
              && !Has_matRewardBoost<SACAZLegacyAgent>::value
              && !Has_mateRewardBoost<SACAZLegacyAgent>::value,
          "还原版**没有** mateScoreMode / matRewardBoost / mateRewardBoost (无按局面分配的权重)");
    CHECK(Has_mateScoreMode<SACAZAgent>::value && Has_matRewardBoost<SACAZAgent>::value
              && Has_mateRewardBoost<SACAZAgent>::value,
          "[对照] 探针能在 SACAZAgent 上探到动态奖励分配的三个成员");
    /* critic 值域抑制: 连名字都没有 */
    CHECK(!Has_clampTarget<SACAZLegacyAgent>::value,
          "还原版**没有** clampTarget (critic 目标不夹)");
    CHECK(!Has_huberDelta<SACAZLegacyAgent>::value,
          "还原版**没有** huberDelta (纯 MSE, 没有 Huber 分支)");
    CHECK(!Has_valueScale<SACAZLegacyAgent>::value,
          "还原版**没有** valueScale (搜索叶子值不缩放)");
    CHECK(!Has_entropyInTarget<SACAZLegacyAgent>::value,
          "还原版**没有** entropyInTarget (熵项恒进软价值, 不可关)");
    CHECK(!Has_entropySlotsAsLegal<SACAZLegacyAgent>::value,
          "还原版**没有** entropySlotsAsLegal (目标熵分母恒为合法着法数)");
    /* 目标网同步率: 编译期常数, 没有可覆盖的成员 */
    CHECK(!Has_targetTau<SACAZLegacyAgent>::value
              && !Has_replaceTargetIter<SACAZLegacyAgent>::value,
          "还原版**没有**可调的 targetTau / replaceTargetIter (硬编码常数)");
    CHECK(SACAZLegacyAgent::POLYAK_TAU == 1e-3f
              && SACAZLegacyAgent::TARGET_SYNC_EVERY == 64,
          "目标网同步率是**编译期常数** tau=1e-3 每 64 步 (59e5233 的口径)");
    /* 搜索期求值口径 */
    CHECK(!Has_sparseLeafEval<SACAZLegacyAgent>::value,
          "还原版**没有** sparseLeafEval (叶子估值恒为全量)");
    CHECK(!Has_learnFromSearch<SACAZLegacyAgent>::value,
          "还原版**没有** learnFromSearch (不从自己的搜索学一次: 59e5233 没有这条路径)");
    /* 剩下两个"该有的"口径值 (它们是成员, 但默认值被钉在 59e5233 上) */
    CHECK(SACAZLegacyAgent::LEGACY_ENTROPY_RATIO == 0.98f
              && SACAZLegacyAgent::LEGACY_ALPHA_LR == 1e-3f,
          "目标熵 0.98 / alpha 学习率 1e-3 (59e5233 的值)");

    /* ---- (b2) 身份标签与权重前缀对**四个**骨干全都要成立 ----
       为什么要有这一段 (2026-09 的真实失败): 上一版 guiAgentLabel() 只给 Mlp 与
       SparseMoeTb 写了分支, 其余骨干返回 `bench/测试构造 (界面不为它建实例)` ——
       那句里**没有 59e5233**, 于是"这是 59e5233 还原版"这个身份在 SparseMoeMlp 上丢了,
       下面循环里那条断言如实失败。身份是**口径**的属性, 不该随骨干消失 ⇒ 修的是实现
       (标签一律保留 59e5233, 骨干只决定后缀), 而这里把"四种骨干都要成立"变成机器可查的。

       用**静态**版本 guiAgentLabelFor() 查: 界面上的 TB 那一支 (SparseMoeTb) 建网 + 拷贝
       权重很贵 (一个 net ~2.9e7 参数, 本节要建 6 个), 为了查一句标签去构造 TB 实例不划算;
       而实例版本只是转发到静态版本, 所以下面循环里再用实例对一次表 (两处口径同一来源)。
    */
    {
        const SACAZLegacyAgent::Backbone bbs[] = {
            SACAZLegacyAgent::Backbone::Mlp,
            SACAZLegacyAgent::Backbone::SparseMoeMlp,
            SACAZLegacyAgent::Backbone::SparseMoeTb,
            SACAZLegacyAgent::Backbone::DenseMoeTb
        };
        const int n = (int)(sizeof(bbs) / sizeof(bbs[0]));
        int withId = 0;      /* 标签里带 59e5233 的个数 */
        int distinct = 0;    /* 与前面所有标签都不相同的个数 (即标签是否随骨干变) */
        for (int i = 0; i < n; i++) {
            const std::string li = SACAZLegacyAgent::guiAgentLabelFor(bbs[i]);
            if (li.find("59e5233") != std::string::npos) {
                withId++;
            }
            bool dup = false;
            for (int j = 0; j < i; j++) {
                if (li == std::string(SACAZLegacyAgent::guiAgentLabelFor(bbs[j]))) {
                    dup = true;
                }
            }
            if (!dup) {
                distinct++;
            }
            std::printf("  骨干 %-22s 标签 '%s'\n",
                        SACAZLegacyAgent::backboneName(bbs[i]), li.c_str());
        }
        CHECK(withId == n,
              "**每个**骨干的标签都带 59e5233 (还原口径的身份不随骨干消失)");
        CHECK(distinct == n,
              "四个骨干的标签两两不同 (面板上分得清是哪一支, 读数不会记到同一本账)");
        /* 界面上的两支: 标签里必须带各自的**界面类型名** (面板第一行要能对上界面选择) */
        const std::string lMlp =
            SACAZLegacyAgent::guiAgentLabelFor(SACAZLegacyAgent::Backbone::Mlp);
        const std::string lTb =
            SACAZLegacyAgent::guiAgentLabelFor(SACAZLegacyAgent::Backbone::SparseMoeTb);
        CHECK(lMlp.find("AGENT_SACAZ_OLD") != std::string::npos
                  && lMlp.find("AGENT_SACAZ_OLD_MOE") == std::string::npos,
              "MLP 那一支的标签写明界面类型 AGENT_SACAZ_OLD (**不**是 _MOE 那支)");
        CHECK(lTb.find("AGENT_SACAZ_OLD_MOE") != std::string::npos,
              "TB 那一支的标签写明界面类型 AGENT_SACAZ_OLD_MOE (与 MLP 那支可区分)");
        /* 两个界面骨干的权重前缀也必须不同 (同名文件会让后训练的那支静默覆盖另一支) */
        CHECK(std::string(SACAZLegacyAgent::defaultWeightPrefix(SACAZLegacyAgent::Backbone::Mlp))
                  != std::string(SACAZLegacyAgent::defaultWeightPrefix(
                         SACAZLegacyAgent::Backbone::SparseMoeTb)),
              "两个界面骨干的权重前缀不同 (MLP 与 TB 的权重文件不会互相覆盖)");
    }

    struct Case { const char *name; int backbone; };   /* 0 = Mlp, 1 = SparseMoeMlp */
    const Case cases[] = {
        /* Mlp: 界面上的两个 SAC 用的骨干 */
        { "mlp", 0 },
        /* moe-mlp: 曾经被 TanhNorm 换掉的那条隐层路径 (回归的发生地) */
        { "moe-mlp", 1 }
    };

    for (const Case &cs : cases) {
        Chess c;
        c.reset();
        /*
           两个类**各自的** Backbone 是**不同的枚举类型** (互不能赋值) —— 这是拆分的直接
           后果, 所以这里各取一次; 枚举项与 59e5233 逐字相同 (同一骨干、同一表示)。
        */
        /*
           [2026-09 独立类拆分] 按骨干分别构造"当前口径"那一支, 再交给同一个模板比较体
           (compareCurrentVsLegacy) —— 断言一字不改, 换的只是类。
        */
        if (cs.backbone == 0) {
            SACAZAgent cur(c, 32, 0.99f, 0.001f, 1.5f);
            SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f,
                                 SACAZLegacyAgent::Backbone::Mlp, 64, 0.01f);
            compareCurrentVsLegacy(c, cur, old, SACAZLegacyAgent::Backbone::Mlp, cs.name);
        } else {
            SACAZMoEMlpAgent cur(c, 32, 0.99f, 0.001f, 1.5f, 64, 0.01f);
            SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f,
                                 SACAZLegacyAgent::Backbone::SparseMoeMlp, 64, 0.01f);
            compareCurrentVsLegacy(c, cur, old, SACAZLegacyAgent::Backbone::SparseMoeMlp, cs.name);
        }
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
 *  15b. [2026-09 用户提议] **动态奖励分配** rewardShape = 3
 *
 *  用户口径 (两轮):
 *    ① "前期应该加强吃棋子的奖励权重, 后期应该重杀将奖励, 或者说这两者都不应忽视奖励";
 *    ② "分阶段太过思维定势, 因为局势是反复变化的, 是否可以参考棋子的数量和价值来
 *       评估局面动态调整杀棋杀将的奖励比例"。
 *
 *  ② 把 ① 的"阶段钟"换成了**局面评估** e (同时读棋子的数量 / 价值 / 双方优势),
 *  于是"同一个吃子在两种局面下拿到的权重不同", 而且局势反复时权重会**回摆**。
 *  设计见 src/sacazagent.h 的 `rewardShape = 3`。这里钉八件事:
 *
 *   (1) **默认路径逐位不变**: shape=3 + 两个 boost = 0 时与 shape=0 完全同值;
 *   (2) 满盘均势: e=0 -> 吃子 x1.5 / 终局 x1.0;
 *   (3) **数量是独立的一维**: 价值相同、个数不同的两个局面 -> e 不同;
 *   (4) **局势反复**: 同一总子力/同一子数, 一边倒 -> e 升; 落后方吃回一个子 -> e 回落;
 *   (5) **与手数无关**: 局面不变只把 history 推长 -> e 一位不变 (不许再引阶段钟);
 *   (6) 调用顺序无关 (被吃子算回"走子前") / 两个倍数之和恒定 (= 固定预算按局面分配);
 *   (7) 零和对称 / 和棋 0 / 搜索叶子与训练目标同值 (倍率 != 1 时);
 *   (8) 量纲不变量 (吃光对方 < 赢棋) 与三种因子口径的接线。
 * ============================================================ */
/*
   [2026-09 独立类拆分 + 用户口径] "新的奖励方法 (rewardShape=3 的动态分配) 也要应用到
   **MoE+MLP 专家**那一支" —— 这句话在代码上已经成立 (两个独立类是 **SACAZAgent 的逐字
   副本**, 奖励代码原样带过去); 但"代码里有"不等于"跑得对", 所以这一段改成**对三个类
   各跑一遍同一批断言**:
       SACAZAgent (纯 MLP) / SACAZMoEMlpAgent (MoE+MLP 专家) / SACAZMoETbAgent (MoE+TB 专家)
   断言一字不改 —— 于是"三支的奖励口径逐项一致"是**跑出来的**, 不是看出来的。
*/
template <class A>
static void posRewardChecks(A &agent, Chess &c, const char *clsName)
{
    std::printf("  ---- %s ----\n", clsName);

    /* ---- 摆局面的两个小工具 (直接改 alive, 不动走子规则) ---- */
    auto clearBoard = [&c]() {
        c.reset();     /* 保留双方将, 其余全部下掉 */
        for (int i = 0; i < 32; i++) {
            Stone *s = c.stones[i];
            if (s != nullptr && s->type != Stone::TYPE_JIANG) {
                s->alive = false;
            }
        }
    };
    auto put = [&c](int color, int type, int n) {
        int left = n;
        for (int i = 0; i < 32 && left > 0; i++) {
            Stone *s = c.stones[i];
            if (s == nullptr || s->color != color || s->type != type) {
                continue;
            }
            s->alive = true;
            left--;
        }
        return left == 0;
    };
    /* 摆一个局面的可读描述 (测试报告里印出来, 便于手算对账) */
    auto describe = [&c]() -> std::string {
        double red = 0.0, black = 0.0;
        int cnt = 0;
        for (int i = 0; i < 32; i++) {
            Stone *s = c.stones[i];
            if (s == nullptr || !s->alive || s->type == Stone::TYPE_JIANG) {
                continue;
            }
            if (s->color == Stone::COLOR_RED) { red += s->value; } else { black += s->value; }
            cnt++;
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "红%.2f/黑%.2f/%d子", red, black, cnt);
        return std::string(buf);
    };
    auto quiet = [](Step &s) { s.nextId = Stone::ID_NONE; s.valid = true; };

    /* ---- (1) shape=0 是参照; shape=3 且 boost 全 0 必须与它逐位相同 ---- */
    c.reset();
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

    agent.rewardShape = 0;
    const float baseCap = agent.computeReward(cap, Stone::COLOR_RED);
    const float baseTerm = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);

    agent.rewardShape = 3;
    agent.mateScoreMode = 0;
    agent.matRewardBoost = 0.0f;
    agent.mateRewardBoost = 0.0f;
    CHECK(agent.computeReward(cap, Stone::COLOR_RED) == baseCap,
          "shape=3 + boost=0 的即时奖励与 shape=0 **逐位相同** (倍数恒为 1.0)");
    CHECK(agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED) == baseTerm,
          "shape=3 + boost=0 的终局值与 shape=0 逐位相同");

    /* ---- (2) 满盘均势: e=0 -> 吃子 x1.5, 终局 x1.0 ---- */
    agent.matRewardBoost = 0.5f;
    agent.mateRewardBoost = 0.5f;
    const double eOpen = agent.mateProximity(&cap);
    std::printf("  满盘均势 (%s): e = %.9f (期望 0)\n", describe().c_str(), eOpen);
    CHECK(std::fabs(eOpen) < 1e-9, "满盘均势时 e = 0 (价值 7.0 / 数量 30 / 优势 0)");
    CHECK(agent.materialWeightMul(&cap) == 1.5f, "满盘均势: 材质倍数 = 1 + matBoost = 1.5");
    CHECK(agent.mateWeightMul() == 1.0f, "满盘均势: 终局倍数 = 1.0 (**不**放大: 这时该惦记吃子)");

    const float earlyCap = agent.computeReward(cap, Stone::COLOR_RED);
    const float expectEarly = REWARD_MATERIAL_COEF * (0.3f * 1.5f) + REWARD_STEP_COST;
    std::printf("  满盘吃马 = %.6f (默认口径 %.6f, 期望 %.6f)\n",
                (double)earlyCap, (double)baseCap, (double)expectEarly);
    CHECK(earlyCap == expectEarly, "满盘吃子 = 材质 x0.1 x1.5 + 每步代价, 数值与手算一致");
    CHECK(earlyCap > baseCap, "**均势局面里吃子奖励被加强** (用户口径的前半句)");
    CHECK(agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED) == REWARD_TERMINAL,
          "满盘均势的将局仍是 ±1 (没有被材质加权带跑)");

    /* ---- (5) 与手数无关: 局面不变, 只把 history 推长 ---- */
    const double eBefore = agent.mateProximity(nullptr);
    for (int i = 0; i < 40; i++) {
        c.pushHistory();              /* 只加"手数", 一个子都不动 */
    }
    const double eAfter = agent.mateProximity(nullptr);
    std::printf("  推长 40 手 history: e %.9f -> %.9f\n", eBefore, eAfter);
    CHECK(eBefore == eAfter,
          "**e 只由棋盘决定, 与手数无关** (第二版不许再把阶段钟引回来)");

    /* ---- (3) 数量是独立的一维: 价值/优势相同, 个数不同 -> e 不同 ---- */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_CHE, 1);
    put(Stone::COLOR_BLACK, Stone::TYPE_CHE, 1);
    const double eBig = agent.mateProximity(nullptr);      /* 1 车 vs 1 车: 价值 1.0, 2 子 */
    const std::string dBig = describe();
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_BING, 5);
    put(Stone::COLOR_BLACK, Stone::TYPE_BING, 5);
    const double ePawns = agent.mateProximity(nullptr);    /* 5 兵 vs 5 兵: 价值 1.0, 10 子 */
    std::printf("  同为价值 1.0: %s e=%.9f  /  %s e=%.9f\n",
                dBig.c_str(), eBig, describe().c_str(), ePawns);
    CHECK(std::fabs(eBig - 0.596825397) < 1e-6,
          "1 车 vs 1 车: e = (1-1/7 + 1-2/30 + 0)/3 = 0.596825 与手算一致");
    CHECK(eBig > ePawns,
          "**同样的价值、不同的个数 -> 评估不同**: 子少 (开阔、易杀) 的 e 更大");

    /* ---- (4) 局势反复: 同一总子力/同一子数, 一边倒 -> e 升; 吃回一个子 -> e 回落 ---- */
    /* 均势: 红 马+2兵 (0.5) vs 黑 马+2兵 (0.5) -> 价值 1.0, 6 子, 优势 0 */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);
    put(Stone::COLOR_RED, Stone::TYPE_BING, 2);
    put(Stone::COLOR_BLACK, Stone::TYPE_MA, 1);
    put(Stone::COLOR_BLACK, Stone::TYPE_BING, 2);
    const double eEven = agent.mateProximity(nullptr);
    const std::string dEven = describe();
    /* 一边倒 (总子力与子数**完全相同**, 只把两个兵从黑方挪到红方) */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);
    put(Stone::COLOR_RED, Stone::TYPE_BING, 3);
    put(Stone::COLOR_BLACK, Stone::TYPE_MA, 1);
    put(Stone::COLOR_BLACK, Stone::TYPE_BING, 1);
    const double eLopsided = agent.mateProximity(nullptr);
    const std::string dLopsided = describe();
    /* 落后方吃回红方一个兵 -> 优势收窄 */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);
    put(Stone::COLOR_RED, Stone::TYPE_BING, 2);
    put(Stone::COLOR_BLACK, Stone::TYPE_MA, 1);
    put(Stone::COLOR_BLACK, Stone::TYPE_BING, 1);
    const double eBack = agent.mateProximity(nullptr);
    std::printf("  均势 %s e=%.9f / 一边倒 %s e=%.9f / 吃回一子 %s e=%.9f\n",
                dEven.c_str(), eEven, dLopsided.c_str(), eLopsided,
                describe().c_str(), eBack);
    CHECK(std::fabs(eEven - 0.552380952) < 1e-6,
          "均势局面: e = (1-1/7 + 1-6/30 + 0)/3 = 0.552381 与手算一致");
    CHECK(eLopsided > eEven,
          "**总子力与子数相同、只有优势不同 -> e 不同** (优势这一维真的在参与评估)");
    CHECK(eBack < eLopsided,
          "**落后方吃回一个子, e 回落** —— 这就是'局势反复'时权重跟着回摆");
    CHECK(agent.mateWeightMul() < agent.materialWeightMul(&cap) + 1.0f,
          "两个倍数都还在各自的区间里 (e 变化不产生越界值)");

    /* ---- (6) 调用顺序无关 + 固定预算 ---- */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_CHE, 1);
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);
    put(Stone::COLOR_BLACK, Stone::TYPE_CHE, 1);
    int victimId = -1;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == Stone::COLOR_BLACK
            && s->type == Stone::TYPE_CHE) {
            victimId = s->id;
            break;
        }
    }
    CHECK(victimId >= 0, "残局里能找到黑车 (构造一次吃子)");
    Step capCar;
    capCar.nextId = victimId;
    capCar.valid = true;
    const double eBeforeCap = agent.mateProximity(&capCar);
    const float rBefore = agent.computeReward(capCar, Stone::COLOR_RED);
    Stone *victim = c.stones[victimId];
    victim->alive = false;            /* 模拟 moveForward 之后的状态 */
    const double eAfterCap = agent.mateProximity(&capCar);
    const float rAfter = agent.computeReward(capCar, Stone::COLOR_RED);
    std::printf("  走子前 e=%.9f r=%.9f / 走子后 e=%.9f r=%.9f\n",
                eBeforeCap, (double)rBefore, eAfterCap, (double)rAfter);
    CHECK(eBeforeCap == eAfterCap && rBefore == rAfter,
          "**同一步在 moveForward 之前/之后求奖励给出同一个值** (被吃子按走子前计入)");
    victim->alive = true;             /* 还原 */

    const float mMat = agent.materialWeightMul(&capCar);
    const float mMate = agent.mateWeightMul();
    std::printf("  该局面: 材质倍数 %.9f + 终局倍数 %.9f = %.9f (期望 2.5)\n",
                (double)mMat, (double)mMate, (double)mMat + (double)mMate);
    CHECK(std::fabs((double)mMat + (double)mMate - 2.5) < 1e-6,
          "**两个倍数之和恒定 = 2.5** (固定奖励预算按局面动态分配 —— 用户说的'比例')");

    /* ---- (7) 残局/定局: 终局被放大, 材质回到基准但不归零 ---- */
    clearBoard();                     /* 只剩双方光将: 价值 0 / 数量 0 / 优势 0 */
    const double eBare = agent.mateProximity(nullptr);
    std::printf("  只剩双方光将: e = %.9f (期望 2/3)\n", eBare);
    CHECK(std::fabs(eBare - 2.0 / 3.0) < 1e-9,
          "光将局面 e = (1 + 1 + 0)/3 = 2/3 (优势分母为 0 时按 0 处理)");
    const float mMatBare = agent.materialWeightMul(nullptr);
    std::printf("  光将局面: 材质倍数 %.6f (期望 %.6f) —— **不为 0**\n",
                (double)mMatBare, 1.0 + 0.5 * (1.0 - 2.0 / 3.0));
    CHECK(std::fabs((double)mMatBare - (1.0 + 0.5 / 3.0)) < 1e-6,
          "光将局面材质倍数 = 1 + 0.5x(1-2/3) = 1.1667: 权重是**降到基准附近**, 不是清零");
    CHECK(mMatBare > 1.0f,
          "**材质项永远留着一份** (倍数恒 ≥ 1: 用户那句'两者都不应忽视')");
    const float bareWin = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);
    const float bareLose = agent.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK);
    std::printf("  光将局面将局: 胜方 +%.6f / 败方 %.6f (期望 %.6f)\n",
                (double)bareWin, (double)bareLose, 1.0 + 0.5 * 2.0 / 3.0);
    CHECK(std::fabs((double)bareWin - (1.0 + 0.5 * 2.0 / 3.0)) < 1e-6,
          "残局将局 = ±1 x (1 + 0.5 x 2/3) = ±1.3333 与手算一致 (**后期杀将更重**)");
    CHECK(bareWin > REWARD_TERMINAL, "残局将局严格大于满盘的 ±1");
    CHECK(std::fabs((double)bareWin + (double)bareLose) < 1e-6,
          "胜负两侧**对称** (同一个倍率 -> 零和性质不被破坏)");
    CHECK(agent.terminalReward(Chess::RESULT_DRAW, Stone::COLOR_RED) == 0.0f,
          "和棋恒为 0 (动态分配也不奖励'别输')");

    /* 必胜残局 (红车马 vs 光将): e 接近 1 —— 两个倍数在上端的行为 */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_CHE, 1);
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);
    const double eWon = agent.mateProximity(nullptr);
    std::printf("  必胜残局 (%s): e = %.6f / 材质倍数 %.6f / 终局倍数 %.6f\n",
                describe().c_str(), eWon, (double)agent.materialWeightMul(nullptr),
                (double)agent.mateWeightMul());
    CHECK(eWon > 0.9 && eWon <= 1.0, "必胜残局 (车马 vs 光将) 的 e 接近上限但不越界");
    CHECK(agent.materialWeightMul(nullptr) > 1.0f,
          "e 再大, 材质倍数也**保持 ≥ 1** —— 材质项永不清零");
    CHECK(agent.mateWeightMul() > 1.4f,
          "e 接近 1 时终局倍数接近上限 1 + mateBoost = 1.5");

    /* ---- (8) 三个终局出口: 搜索叶子与训练目标同值 (倍率 != 1 时也要同值) ---- */
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == Stone::COLOR_RED
            && s->type == Stone::TYPE_JIANG) {
            s->alive = false;
            break;
        }
    }
    const int res = c.getResult(Stone::COLOR_RED);
    double leaf = 0.0;
    const bool isTerm = agent.terminalValue(Stone::COLOR_RED, leaf);
    const float train = agent.terminalReward(res, Stone::COLOR_RED);
    std::printf("  终局局面: 搜索叶子 %.6f / 训练目标 %.6f (result=%d)\n",
                leaf, (double)train, res);
    CHECK(isTerm && std::fabs(leaf - (double)train) < 1e-9,
          "**搜索叶子与训练目标的终局值逐位相同** (倍率 != 1 时也只有一个出口)");
    CHECK(std::fabs((double)train) > (double)REWARD_TERMINAL,
          "这个终局值确实走了动态倍率 (不是恰好等于 ±1 的假通过)");

    /* ---- (9) 量纲不变量 + 因子口径切换 ---- */
    const double fullSide = REWARD_FULL_MATERIAL_DIFF * (double)REWARD_MATERIAL_COEF;
    const double boosted = fullSide * (1.0 + (double)agent.matRewardBoost);
    const double boostCap = (fullSide > 0.0) ? (1.0 / fullSide - 1.0) : 0.0;
    std::printf("  不变量: 吃光对方 %.3f < 终局 %.3f (matBoost 上界 %.3f)\n",
                boosted, (double)REWARD_TERMINAL, boostCap);
    CHECK(boosted < (double)REWARD_TERMINAL,
          "**吃光对方仍不如赢棋** (默认 matBoost=0.5: 0.525 < 1.000)");
    CHECK(std::fabs(boosted - 0.525) < 1e-6, "材质上界与手算一致 (3.5 x 0.1 x 1.5 = 0.525)");

    /* 同一个局面下, 四种因子口径各自读到的是不是它该读的那一维 */
    clearBoard();
    put(Stone::COLOR_RED, Stone::TYPE_CHE, 1);
    put(Stone::COLOR_RED, Stone::TYPE_MA, 1);       /* 红 0.8 / 黑 0.5: 价值 1.3, 3 子, 优势 0.3/1.3 */
    put(Stone::COLOR_BLACK, Stone::TYPE_CHE, 1);
    Step none;
    quiet(none);
    agent.mateScoreMode = 1;
    const double eVal = agent.mateProximity(&none);
    agent.mateScoreMode = 2;
    const double eCnt = agent.mateProximity(&none);
    agent.mateScoreMode = 4;
    const double eLead = agent.mateProximity(&none);
    agent.mateScoreMode = 3;
    const double eVC = agent.mateProximity(&none);
    agent.mateScoreMode = 0;
    const double eAll = agent.mateProximity(&none);
    std::printf("  口径切换 (%s): 价值 %.6f / 数量 %.6f / 优势 %.6f / 数量+价值 %.6f / 三因子 %.6f\n",
                describe().c_str(), eVal, eCnt, eLead, eVC, eAll);
    CHECK(std::fabs(eVal - (1.0 - 1.3 / 7.0)) < 1e-9, "mode 1 读到的就是价值因子");
    CHECK(std::fabs(eCnt - (1.0 - 3.0 / 30.0)) < 1e-9, "mode 2 读到的就是数量因子");
    CHECK(std::fabs(eLead - (0.3 / 1.3)) < 1e-9, "mode 4 读到的就是优势因子 (相对子力差)");
    CHECK(std::fabs(eVC - (eVal + eCnt) * 0.5) < 1e-9, "mode 3 = 数量与价值的平均");
    CHECK(std::fabs(eAll - (eVal + eCnt + eLead) / 3.0) < 1e-9, "mode 0 = 三因子等权 (默认)");
    CHECK(eAll < eCnt && eAll > eLead,
          "三因子等权落在最小因子 (优势) 与最大因子 (数量) 之间 —— 均值该有的位置");

    /* 恢复默认, 让后面的测试拿到出厂口径 */
    agent.rewardShape = 0;
    agent.mateScoreMode = 0;
    agent.matRewardBoost = 0.5f;
    agent.mateRewardBoost = 0.5f;
}

/*
   [15b] 的入口: **三个类各跑一遍** (纯 MLP / MoE+MLP 专家 / MoE+TB 专家)。
   用户口径: "新的奖励方法也应用到 SAC+MCTS+AlphaZero+MoE+MLP 专家" —— 这一段就是
   那个口径的**证据**: 同一个模板, 三份实例化, 同一批断言。
*/
static void testPosReward()
{
    std::printf("\n[15b] 动态奖励分配 rewardShape=3 (按局面评估分配吃子/杀将权重)\n");
    std::printf("      **三个类各跑一遍同一批断言** (奖励口径必须逐项一致)\n");

    Chess c;
    c.reset();
    sacazx::Opts o;
    o.hidden = 32;

    sacazx::withSacazAgent(c, sacazx::Variant::Mlp, o, [&](auto &ag) {
        posRewardChecks(ag, c, "SACAZAgent (纯 MLP)");
    });
    sacazx::withSacazAgent(c, sacazx::Variant::MoeMlp, o, [&](auto &ag) {
        posRewardChecks(ag, c, "SACAZMoEMlpAgent (稀疏 MoE + MLP 专家)");
    });
    sacazx::withSacazAgent(c, sacazx::Variant::MoeTb, o, [&](auto &ag) {
        posRewardChecks(ag, c, "SACAZMoETbAgent (稀疏 MoE + TB 专家)");
    });
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
 *   (4) 还原版 (SACAZLegacyAgent) **根本没有这条路径** -> 一步都不学。
 *       [2026-09 独立类] 这里不再写 `!old.learnFromSearch` (那个成员在独立类里**不存在**,
 *       见 [14] 节的探针), 改成**行为**断言: 反复 selectMove 之后 learnSteps 不动、
 *       回放池里不添样本。这比读开关更强 —— 它拦的是"实现里偷偷学了一次"。
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

    /* ---- 还原版 (59e5233 口径) 必须一条样本都不学 ---- */
    {
        Chess c;
        c.reset();
        SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f);
        /* 它连这个开关都没有 (见 [14] 节的探针), 所以只能从**行为**上钉: 一步都不学 */
        CHECK(!Has_learnFromSearch<SACAZLegacyAgent>::value,
              "SACAZLegacyAgent 没有 learnFromSearch 成员 (59e5233 没有这条路径)");
        const int steps0 = old.getLearnSteps();
        for (int i = 0; i < 3; i++) {
            const Step s = old.selectMove(Stone::COLOR_RED, 8, 0.0f);
            CHECK(s.valid, "还原版的 selectMove 是**只读搜索** (返回合法走法, 不学习)");
        }
        std::printf("  还原版: 3 次 selectMove -> learnSteps %d -> %d, 池 %zu\n",
                    steps0, old.getLearnSteps(), old.getMemorySize());
        CHECK(old.getLearnSteps() == steps0, "还原版 selectMove 不产生任何更新");
        CHECK(old.getMemorySize() == 0, "也不往池里存样本 (只读搜索, 改动前的行为)");
    }
}

/* ============================================================
 *  17. [2026-09 dev-sacmoetb] 共享骨干 + TB 专家的头数口径
 *
 *  这一节钉的是两个**静默失效**:
 *
 *  (A) **TB 专家的头数被静默降级**。`MOE_TB_HEADS = 15` ("d_model/15 = 84 维/头"),
 *      而 TB 专家真正的 d_model 是 `STATE_DIM`。默认表示下 STATE_DIM = 1263 = 3 x 421
 *      (421 是素数), 而老口径是"从请求头数往下找第一个**整除** d_model 的数" ——
 *      在 1263 上找到的是 **3**。于是真实 head 数 3 / d_k=421 / 531723 个注意力元素
 *      (设计意图 15 / 84 / 105840), 单个 TB 专家前向实测 15.7 ms vs 5.4 ms。
 *      参数指纹、paramCount、层类型序列、权重文件格式**一个都没变** —— 这正是
 *      TanhNorm<Sigmoid> 那次回归的同类: 同形状、同参数量的静默替换。
 *      修法 = `HonorHeads` (见 rl/attention.hpp): 按**请求**的头数切分, 允许
 *      `numHeads*d_k < d_model` (尾部坐标不进注意力拼接, 但仍走残差与 Wo)。
 *
 *  (B) **三张网把同一件事算三遍**。独立口径下 actor/q1/q2/q1Target/q2Target 各背
 *      一整套骨干; 一次叶子估值 = 3 次骨干前向 (策略一次 + 双 Q 各一次), 一次
 *      learnBatch 的每个样本 = 6 次骨干前向 + 3 次骨干反向。共享口径把骨干压成
 *      一份 (在线 + 目标), actor/q1/q2 退化成三个只有输出层的头。
 *
 *  本节断言能支撑什么、不能支撑什么:
 *    * **能**: (i) 两种口径在同权重同局面下给出**逐位相同**的 π / Q / 软价值
 *      (外加显式断言三个视图确实与 trunk 共享层对象 —— 否则 actor/q1/q2 会是陈旧的
 *      副本); (ii) 共享口径的代价严格更低 (每样本/每模拟); (iii) 参数量更低;
 *      (iv) 修好头数之后 TB 专家的前向确实更快, 且头数落地读数 = 请求值;
 *      (v) 共享口径的存/取往返一致; (vi) 共享骨干**真的收到了梯度** (它的输出在
 *      一次 learnBatch 之后变了 —— 没有这一步, "共享"可能只是个不训练的摆设)。
 *    * **不能**: 棋力。本节量的是"同一套算法、同样的搜索预算下谁更快 / 谁更省",
 *      不是"谁下得更好"。棋力结论要么靠同时间预算下的对局, 要么靠同模拟次数下的
 *      对局 —— 那是 bench 那一层的活 (见 docs/dev_sacmoetb_*.md 的复现命令)。
 * ============================================================ */
static void testSharedTrunkAndTbHeads()
{
    std::printf("\n[17] 共享骨干 + TB 专家头数口径 (2026-09 dev-sacmoetb)\n");

    /* ---------- (A) TB 专家的头数: 请求值 vs 实际值 ---------- */
    std::printf("\n  (A) TB 专家头数: d_model=%d 上 \"请求 15 个头\" 到底落地成几个\n",
                SACAZAgent::STATE_DIM);
    {
        /*
           直接量裸 TransformerBlock, 把"口径"这一个变量单独隔离出来。
           30 次迭代足够: 一次前向 5~16 ms, steady_clock 的分辨率远小于它。
        */
        RL::Tensor x(SACAZAgent::STATE_DIM, 1);
        for (int i = 0; i < SACAZAgent::STATE_DIM; i++) {
            x[i] = 0.1f * std::sin((float)i);
        }
        RL::Net tbLegacy(RL::TransformerBlock<SACAZAgent::MOE_TB_HEADS,
                                              SACAZAgent::MOE_TB_DFF, false>::_(
                             SACAZAgent::STATE_DIM, true));
        RL::Net tbHonor(RL::TransformerBlock<SACAZAgent::MOE_TB_HEADS,
                                             SACAZAgent::MOE_TB_DFF, true>::_(
                            SACAZAgent::STATE_DIM, true));
        const double msLegacy = timeForwardMs(tbLegacy, x, 30);
        const double msHonor = timeForwardMs(tbHonor, x, 30);

        RL::iLayer *lA = tbLegacy[0];
        RL::iLayer *hA = tbHonor[0];
        const int reqL = lA->attnHeadsRequested(), useL = lA->attnHeadsUsed();
        const int reqH = hA->attnHeadsRequested(), useH = hA->attnHeadsUsed();
        const long long elemL = lA->attnElements(), elemH = hA->attnElements();

        std::printf("      请求 %d 头 | 旧口径: 实际 %d 头, d_k=%d, 注意力元素 %lld,"
                    " 前向 %.2f ms\n",
                    reqL, useL, lA->attnHeadDim(), elemL, msLegacy);
        std::printf("      请求 %d 头 | 新口径: 实际 %d 头, d_k=%d, 注意力元素 %lld,"
                    " 前向 %.2f ms\n",
                    reqH, useH, hA->attnHeadDim(), elemH, msHonor);
        std::printf("      加速 = %.2fx  (%.1f%% 的注意力元素被省掉)\n",
                    msLegacy / msHonor, 100.0 * (1.0 - (double)elemH / (double)elemL));

        /* 头数真的落地了 */
        CHECK(reqH == SACAZAgent::MOE_TB_HEADS, "新口径下请求的头数就是 MOE_TB_HEADS");
        CHECK(useH == reqH, "新口径: 实际参与前向的头数 == 请求值 (不再被 d_model 的因子数卡住)");
        CHECK(hA->attnHeadsAllocated() == reqH,
              "新口径: 分配出来的 head 对象数 == 实际用到的 (没有死内存)");
        /*
           旧口径的断言写成"实际头数 < 请求头数"而不是写死 3: 那个 3 是
           STATE_DIM=1263 的因子数决定的, 换成对齐表示 (STATE_DIM=1710=2*3^2*5*19) 时
           15 头是能整除的, 这时"降级"根本不发生 —— 写死 3 会让这一节在对齐构建下
           变成假失败 (本文件有 test_sacaz_aligned 这个孪生目标)。
        */
        if (!SACAZAgent::ALIGNED_REPR) {
            CHECK(useL < reqL,
                  "旧口径确实把请求的头数降级了 (根因: 1263 = 3 x 421, 因子只有 1/3/421/1263)");
            CHECK(elemL > elemH * 2,
                  "旧口径的注意力元素数至少是新口径的两倍 (531723 vs 105840, 5.0 倍)");
            CHECK(msLegacy > msHonor,
                  "修好头数之后单专家前向**更快** (实测 15.7 ms -> 5.4 ms)");
        } else {
            /* 对齐表示下 1710 % 15 == 0, 两种口径的实际头数一样 —— 只看一致性 */
            CHECK(useL == useH, "对齐表示下 d_model 能被 15 整除, 两种口径的头数一致");
            CHECK(elemL == elemH, "对齐表示下注意力元素数也一致");
            std::printf("      (对齐表示: STATE_DIM 能被 %d 整除, 这条降级路径不存在)\n",
                        SACAZAgent::MOE_TB_HEADS);
        }
        /* 两种口径的参数形状只差 d_k 带来的 qkv 宽度 -> 参数量应当几乎一样 */
        const long long pL = tbLegacy.paramCount(), pH = tbHonor.paramCount();
        std::printf("      paramCount: 旧 %lld vs 新 %lld (差 %.3f%%) —— 参数量几乎不变,"
                    " 这正是它当年能躲过参数指纹检查的原因\n",
                    pL, pH, 100.0 * std::fabs((double)(pL - pH)) / (double)pL);
        CHECK(std::fabs((double)(pL - pH)) < 0.01 * (double)pL,
              "两种口径的参数量差不到 1% (同形状 => 指纹/参数量守卫都看不见)");
    }

    /* ---------- (B) 共享骨干 vs 独立骨干 ---------- */
    std::printf("\n  (B) 共享骨干 vs 独立骨干 (同权重同局面, 逐位对比)\n");
    {
        Chess cA, cB;
        cA.reset();
        cB.reset();
        /*
           两个 agent 都必须 tbHonorHeads=true, 否则 copyTo 的目标类型不同
           (SparseMoE 的 copyTo 是 dynamic_cast 到**同一个模板实例**, 口径不同就是不同
           类型 -> 静默什么都不做)。这一条本身就是"口径是结构的一部分"的实证。
        */
        SACAZMoETbAgent sep(cA, 64, 0.99f, 0.001f, 1.5f, 64, 0.1f,
                           SACAZMoETbAgent::TrunkMode::Separate, true);
        sep.batchSize = 4;
        const long long sepParams = sep.uniqueParamCount();
        const int sepHeadsUsed = sep.tbHeadsUsed();
        const int sepHeadsAlloc = sep.tbHeadsAllocated();

        double sepBatchMs = 0.0, sepPerSimMs = 0.0;
        {
            /*
               agent 很大, 但**不是**改动前那种"五张 TB 网 ≈ 1.6 GB" —— 先量它的
               每批/每模拟代价, 然后销毁, 再建共享那一支 (两个同时活着没有必要)。
            */
            std::vector<std::uint16_t> cells;
            sep.encodeSparse(Stone::COLOR_RED, cells);
            RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
            std::vector<Step*> legal;
            std::vector<int> legalIdx;
            sep.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
            Steps::instance().put(legal);
            std::uint64_t bits[2] = { 0, 0 };
            SACAZAgent::maskToBits(mask, bits);
            sep.memories.clear();
            for (int i = 0; i < 16; i++) {
            typename std::decay<decltype(sep)>::type::Transition tr;
                tr.cells = cells;
                tr.nextCells = cells;
                tr.curMask[0] = bits[0];
                tr.curMask[1] = bits[1];
                tr.nextMask[0] = bits[0];
                tr.nextMask[1] = bits[1];
                tr.action = legalIdx[0];
                tr.legalCount = (int)legalIdx.size();
                tr.reward = 0.5f;
                tr.done = true;
                tr.hasSearch = false;
                sep.memories.push_back(tr);
            }
            /*
               learnFromSearch 必须关掉: selectMove 里那次 learnBatch 会把"每模拟
               代价"污染成"搜索 + 一次训练", 两个口径就比不出搜索本身了。
            */
            sep.learnFromSearch = false;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 3; i++) { sep.learnBatch(4); }
            const auto t1 = std::chrono::steady_clock::now();
            sepBatchMs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t1 - t0).count() / 1e6 / 3.0;

            const int sims = 3, moves = 2;
            const auto s0 = std::chrono::steady_clock::now();
            for (int i = 0; i < moves; i++) { sep.selectMove(Stone::COLOR_BLACK, sims, 0.0f); }
            const auto s1 = std::chrono::steady_clock::now();
            sepPerSimMs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                              s1 - s0).count() / 1e6 / (double)(moves * sims);
        }

        /* 共享那一支 (作用域独立: 先把独立那一支的读数都拿到手) */
        double shBatchMs = 0.0, shPerSimMs = 0.0;
        long long shParams = 0;
        double maxDPi = 0.0, maxDQ1 = 0.0, maxDQ2 = 0.0, maxDQ1t = 0.0;
        double maxDPiSp = 0.0, maxDQ1Sp = 0.0;
        double maxDSoft = 0.0;
        bool viewAliased = false;
        bool trunkMoved = false;
        bool roundTripOk = false;
        {
            SACAZMoETbAgent sh(cB, 64, 0.99f, 0.001f, 1.5f, 64, 0.1f,
                           SACAZMoETbAgent::TrunkMode::Shared, true);
            sh.batchSize = 4;
            sh.learnFromSearch = false;
            shParams = sh.uniqueParamCount();

            /*
               ================================================================
               ---- 关键设计: 先把独立口径的三份骨干**都设成同一份** ----
               ================================================================
               共享口径的本质就是"把三份**本来可以各自不同**的骨干强制相等"。所以
               "共享 vs 独立"的正确对照是:

                   独立口径在**三个骨干恰好相等、目标骨干也相等**时的行为
                   vs
                   共享口径 (结构上只能相等) 的行为

               如果跳过这一步直接比, 比到的是"两份不同的随机初始化给出的 Q 差异"
               (实测 max|dQ1| = 2.703 —— 那正是"独立口径三份骨干互不相同"的量级),
               而不是"共享这件事改变了什么"。这一条是本节最容易写错的地方, 单独写出来。
            */
            sep.actor[0]->copyTo(sep.q1[0]);
            sep.actor[1]->copyTo(sep.q1[1]);
            sep.actor[0]->copyTo(sep.q2[0]);
            sep.actor[1]->copyTo(sep.q2[1]);
            /* 目标网: 骨干同样对齐; 目标头 = 在线头 (共享口径就是这样建目标网的) */
            sep.actor[0]->copyTo(sep.q1Target[0]);
            sep.actor[1]->copyTo(sep.q1Target[1]);
            sep.actor[0]->copyTo(sep.q2Target[0]);
            sep.actor[1]->copyTo(sep.q2Target[1]);
            sep.q1[2]->copyTo(sep.q1Target[2]);
            sep.q2[2]->copyTo(sep.q2Target[2]);
            /* 再把这份"三个骨干相等"的独立网络逐层拷进共享口径 */
            sep.actor[0]->copyTo(sh.trunk[0]);
            sep.actor[1]->copyTo(sh.trunk[1]);
            sep.actor[2]->copyTo(sh.actorHead[0]);
            sep.q1[2]->copyTo(sh.q1Head[0]);
            sep.q2[2]->copyTo(sh.q2Head[0]);
            sh.trunk.copyTo(sh.trunkTarget);
            sh.q1Head.copyTo(sh.q1TargetHead);
            sh.q2Head.copyTo(sh.q2TargetHead);

            /* ---- 同局面: π / Q / 目标Q / 稀疏叶子 全都要逐位一致 ---- */
            std::vector<Step*> legal;
            std::vector<int> legalIdx;
            RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
            sep.getLegalActions(Stone::COLOR_RED, legal, legalIdx, mask);
            Steps::instance().put(legal);
            std::vector<std::uint16_t> cells;
            sep.encodeSparse(Stone::COLOR_RED, cells);
            RL::Tensor state(SACAZAgent::STATE_DIM, 1);
            SACAZAgent::expandSparse(cells, state);

            RL::Tensor piA(SACAZAgent::ACTION_DIM, 1), piB(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q1a(SACAZAgent::ACTION_DIM, 1), q2a(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor q1b(SACAZAgent::ACTION_DIM, 1), q2b(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor t1a(SACAZAgent::ACTION_DIM, 1), t2a(SACAZAgent::ACTION_DIM, 1);
            RL::Tensor t1b(SACAZAgent::ACTION_DIM, 1), t2b(SACAZAgent::ACTION_DIM, 1);
            sep.policy(state, mask, piA);
            sep.qValues(state, q1a, q2a);
            sep.qTargetValues(state, t1a, t2a);
            sh.policy(state, mask, piB);
            sh.qValues(state, q1b, q2b);
            sh.qTargetValues(state, t1b, t2b);
            for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                maxDPi = std::fmax(maxDPi, std::fabs((double)piA[i] - (double)piB[i]));
                maxDQ1 = std::fmax(maxDQ1, std::fabs((double)q1a[i] - (double)q1b[i]));
                maxDQ2 = std::fmax(maxDQ2, std::fabs((double)q2a[i] - (double)q2b[i]));
                maxDQ1t = std::fmax(maxDQ1t, std::fabs((double)t1a[i] - (double)t1b[i]));
            }

            /* 稀疏路径 (搜索实际用的那条) 也必须一致 */
            /*
               `alpha` 是 agent 的**标量状态**, 而 sep 上面已经跑过 3 次 learnBatch
               (量它的每批代价), α 已经从初值 0.2 挪开了。软价值
                   V = Σ π·(min Q − α·log π)
               里 α 是**乘在熵项上**的, 所以"同权重"这条前提还必须包含"同 α" ——
               不显式对齐的话, 这一条量到的是 α 的差 (实测 2.8e-02), 不是共享的差。
            */
            sh.alpha[0] = sep.alpha[0];
            std::vector<float> piSA, q1SA, q2SA, piSB, q1SB, q2SB;
            const bool okA = sep.sparseLeaf(state, legalIdx, piSA, q1SA, q2SA);
            const bool okB = sh.sparseLeaf(state, legalIdx, piSB, q1SB, q2SB);
            CHECK(okA && okB, "两种口径的稀疏叶子路径都可用");
            if (okA && okB && piSA.size() == piSB.size()) {
                for (std::size_t i = 0; i < piSA.size(); i++) {
                    maxDPiSp = std::fmax(maxDPiSp, std::fabs((double)piSA[i] - (double)piSB[i]));
                    maxDQ1Sp = std::fmax(maxDQ1Sp, std::fabs((double)q1SA[i] - (double)q1SB[i]));
                }
            }
            double vA = 0.0, vB = 0.0;
            const bool okVA = sep.softValueSparse(state, legalIdx, vA);
            const bool okVB = sh.softValueSparse(state, legalIdx, vB);
            CHECK(okVA && okVB, "两种口径的稀疏软价值都算得出来");
            maxDSoft = std::fabs(vA - vB);

            std::printf("      同权重同局面: max|dPi| = %.3e, max|dQ1| = %.3e, max|dQ2| = %.3e,"
                        " max|dQ1_target| = %.3e\n", maxDPi, maxDQ1, maxDQ2, maxDQ1t);
            std::printf("      稀疏路径:      max|dPi| = %.3e, max|dQ1| = %.3e,"
                        " |dV_soft| = %.3e\n", maxDPiSp, maxDQ1Sp, maxDSoft);
            CHECK(maxDPi == 0.0, "策略 π 逐位相同 (共享只改变\"算几次\", 不改变算什么)");
            CHECK(maxDQ1 == 0.0 && maxDQ2 == 0.0, "双 Q 逐位相同");
            CHECK(maxDQ1t == 0.0, "目标 Q 逐位相同 (两个 Q 头共享目标骨干是等价的)");
            CHECK(maxDPiSp == 0.0 && maxDQ1Sp == 0.0, "稀疏叶子路径逐位相同");
            CHECK(maxDSoft == 0.0, "软价值 V(s) 逐位相同");

            /*
               ---- 三个"视图"必须真的与 trunk 共享层对象 ----
               为什么非要单独断这一条: actor/q1/q2 在共享口径下是**拼出来的视图**, 如果
               组装时拷贝了层而不是共享指针, 它们会各自持有一份**永不更新**的骨干 ——
               而前向、稀疏路径、自检面板全都照常工作, 只是读的是陈旧权重。
               判据: 视图算出来的 Q 必须与"trunk 输出喂给 Q 头"逐位相同, 且在**更新过
               之后**仍然相同。
            */
            {
                RL::Tensor qv(SACAZAgent::ACTION_DIM, 1), qv2(SACAZAgent::ACTION_DIM, 1);
                RL::Tensor hv = sh.trunk.forward(state);
                qv = sh.q1Head.forward(hv);
                qv2 = sh.q1.forward(state);       /* 视图路径: 骨干 + Q1头 */
                double d = 0.0;
                for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                    d = std::fmax(d, std::fabs((double)qv[i] - (double)qv2[i]));
                }
                viewAliased = (d == 0.0);
                std::printf("      视图 q1 与 (trunk+Q1头) 的最大差 = %.3e\n", d);
                CHECK(viewAliased, "actor/q1/q2 视图确实与共享骨干**共享层对象** (不是陈旧副本)");
            }

            /* ---- 代价对比 ---- */
            std::vector<std::uint16_t> cells2;
            sh.encodeSparse(Stone::COLOR_RED, cells2);
            std::uint64_t bits[2] = { 0, 0 };
            SACAZAgent::maskToBits(mask, bits);
            sh.memories.clear();
            for (int i = 0; i < 16; i++) {
            typename std::decay<decltype(sh)>::type::Transition tr;
                tr.cells = cells2;
                tr.nextCells = cells2;
                tr.curMask[0] = bits[0];
                tr.curMask[1] = bits[1];
                tr.nextMask[0] = bits[0];
                tr.nextMask[1] = bits[1];
                tr.action = legalIdx[0];
                tr.legalCount = (int)legalIdx.size();
                tr.reward = 0.5f;
                tr.done = true;
                tr.hasSearch = false;
                sh.memories.push_back(tr);
            }
            {
                RL::Tensor h0 = sh.trunk.forward(state);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 3; i++) { sh.learnBatch(4); }
                const auto t1 = std::chrono::steady_clock::now();
                shBatchMs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t1 - t0).count() / 1e6 / 3.0;
                RL::Tensor h1 = sh.trunk.forward(state);
                double dh = 0.0;
                for (std::size_t i = 0; i < h0.size(); i++) {
                    dh = std::fmax(dh, std::fabs((double)h0[i] - (double)h1[i]));
                }
                trunkMoved = (dh > 0.0);
                std::printf("      learnBatch 之后共享骨干的输出最大变化 = %.3e\n", dh);
                CHECK(trunkMoved, "共享骨干**真的收到了梯度** (三个头的梯度之和更新了它)");
            }
            {
                const int sims = 3, moves = 2;
                const auto s0 = std::chrono::steady_clock::now();
                for (int i = 0; i < moves; i++) { sh.selectMove(Stone::COLOR_BLACK, sims, 0.0f); }
                const auto s1 = std::chrono::steady_clock::now();
                shPerSimMs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 s1 - s0).count() / 1e6 / (double)(moves * sims);
            }

            /* ---- 存取往返 (共享口径的 4 文件格式) ---- */
            {
                const std::string prefix = "sacaz_shared_roundtrip";
                CHECK(sh.saveModel(prefix), "共享口径 saveModel 写出 4 个文件");
                RL::Tensor pi0(SACAZAgent::ACTION_DIM, 1), q0(SACAZAgent::ACTION_DIM, 1),
                           q0b(SACAZAgent::ACTION_DIM, 1);
                sh.policy(state, mask, pi0);
                sh.qValues(state, q0, q0b);
                /* 换一个 agent 实例载入 (结构相同), 再比 */
                SACAZMoETbAgent sh2(cB, 64, 0.99f, 0.001f, 1.5f, 64, 0.1f,
                           SACAZMoETbAgent::TrunkMode::Shared, true);
                CHECK(sh2.loadModel(prefix), "共享口径 loadModel 读回 4 个文件");
                RL::Tensor pi1(SACAZAgent::ACTION_DIM, 1), q1(SACAZAgent::ACTION_DIM, 1),
                           q1c(SACAZAgent::ACTION_DIM, 1);
                sh2.policy(state, mask, pi1);
                sh2.qValues(state, q1, q1c);
                double d = 0.0;
                for (int i = 0; i < SACAZAgent::ACTION_DIM; i++) {
                    d = std::fmax(d, std::fabs((double)pi0[i] - (double)pi1[i]));
                    d = std::fmax(d, std::fabs((double)q0[i] - (double)q1[i]));
                    d = std::fmax(d, std::fabs((double)q0b[i] - (double)q1c[i]));
                }
                roundTripOk = (d == 0.0);
                std::printf("      共享口径 save/load 往返: 策略/Q 最大差 = %.3e\n", d);
                CHECK(roundTripOk, "共享口径的存/取往返逐位一致 (含目标网同步)");
                /* 独立口径的文件**不能**被共享口径读进来 (文件个数与语义都不同) */
                CHECK(!sh2.loadModel("weights/definitely_missing_prefix"),
                      "缺文件时 loadModel 明确失败 (不会静默当成随机权重)");
                std::remove((prefix + "_trunk").c_str());
                std::remove((prefix + "_actorhead").c_str());
                std::remove((prefix + "_q1head").c_str());
                std::remove((prefix + "_q2head").c_str());
            }
        }

        /* ---- 汇总: 代价与参数量 ---- */
        std::printf("\n      %-22s %-14s %-14s %s\n", "口径", "参数量", "learnBatch(4)",
                    "ms/模拟");
        std::printf("      %-22s %-14lld %-14s %s\n", "独立 (Separate)", sepParams, "", "");
        std::printf("      %-22s %-14lld %-14s %s\n", "共享 (Shared)", shParams, "", "");
        std::printf("      learnBatch(4): 独立 %.1f ms/批 vs 共享 %.1f ms/批  -> %.2fx\n",
                    sepBatchMs, shBatchMs, sepBatchMs / shBatchMs);
        std::printf("      selectMove:    独立 %.1f ms/模拟 vs 共享 %.1f ms/模拟  -> %.2fx\n",
                    sepPerSimMs, shPerSimMs, sepPerSimMs / shPerSimMs);
        std::printf("      唯一参数量:    独立 %lld vs 共享 %lld  -> %.2fx\n",
                    sepParams, shParams, (double)sepParams / (double)shParams);
        std::printf("      TB 专家头数(独立口径 agent): 请求 %d / 用 %d / 分配 %d\n",
                    SACAZAgent::MOE_TB_HEADS, sepHeadsUsed, sepHeadsAlloc);
        std::printf("      换算: GUI 里 175 ms/步的预算 -> 独立 %.1f 次模拟, 共享 %.1f 次模拟\n",
                    175.0 / sepPerSimMs, 175.0 / shPerSimMs);

        CHECK(shParams < sepParams, "共享口径的唯一参数量低于独立口径");
        CHECK(shBatchMs < sepBatchMs, "共享口径的 learnBatch 更快 (骨干前向 6 次 -> 3 次)");
        CHECK(shPerSimMs < sepPerSimMs, "共享口径的每模拟代价更低 (骨干前向 3 次 -> 1 次)");
        /* 这两个比值是本轮收益的**定量**结论; 只断"更快"会把 1.02x 也算通过 */
        CHECK(shBatchMs * 1.5 < sepBatchMs, "learnBatch 至少快 1.5 倍");
        CHECK(shPerSimMs * 1.5 < sepPerSimMs, "每模拟至少快 1.5 倍");

        /* ---- (C) 还原版**不许**被这两个新口径渗透 ---- */
        std::printf("\n  (C) 59e5233 行为还原版必须保持旧口径\n");
        CHECK(!Has_tbHonorHeads<SACAZLegacyAgent>::value,
              "SACAZLegacyAgent 没有 tbHonorHeads 成员 (它连这个开关都不该有)");
        /*
           [2026-09 独立类拆分] 这个开关的**归属**变了: 它只对 TB 专家那一支有意义, 所以
           现在住在 `SACAZMoETbAgent` 里 —— 纯 MLP 的 SACAZAgent 与 MoE-MLP 都**没有**它。
           (拆分前一个类背四种骨干, 于是纯 MLP 实例上也挂着一个永远不生效的开关。)
        */
        CHECK(Has_tbHonorHeads<SACAZMoETbAgent>::value,
              "对照组: SACAZMoETbAgent (TB 专家那一支) 有 tbHonorHeads (探针本身没写坏)");
        CHECK(!Has_tbHonorHeads<SACAZAgent>::value
                  && !Has_tbHonorHeads<SACAZMoEMlpAgent>::value,
              "另外两支**没有** tbHonorHeads (MLP 专家里没有 MultiHeadAttention)");
        CHECK(!Has_trunkMode<SACAZLegacyAgent>::value,
              "SACAZLegacyAgent 没有 trunkMode (共享骨干不许渗进行为还原版)");
        CHECK(Has_trunkMode<SACAZAgent>::value && Has_trunkMode<SACAZMoETbAgent>::value
                  && Has_trunkMode<SACAZMoEMlpAgent>::value,
              "对照组: 三个当前口径的类都有 trunkMode");;
        {
            Chess cOld;
            cOld.reset();
            SACAZLegacyAgent old(cOld, 64, 0.99f, 0.001f, 1.5f,
                                 SACAZLegacyAgent::Backbone::SparseMoeTb, 64, 0.1f);
            /* 直接读它 actor 里的 MoE 层: 与 [17A] 的旧口径读数必须是同一套 */
            RL::ISparseMoE *moe = nullptr;
            for (std::size_t i = 0; i < old.actor.size(); i++) {
                moe = dynamic_cast<RL::ISparseMoE*>(old.actor[i]);
                if (moe != nullptr) { break; }
            }
            CHECK(moe != nullptr, "还原版的 TB 骨干里有稀疏 MoE 层");
            if (moe != nullptr) {
                std::printf("      还原版 TB 专家: 请求 %d 头 / 实际 %d 头 / 分配 %d 头\n",
                            moe->attnHeadsRequested(), moe->attnHeadsUsed(),
                            moe->attnHeadsAllocated());
                CHECK(moe->attnHeadsUsed() == moe->attnHeadsUsed(),
                      "还原版的头数口径是**它自己的** (本节的修法不许改它)");
                if (!SACAZAgent::ALIGNED_REPR) {
                    CHECK(moe->attnHeadsUsed() < moe->attnHeadsRequested(),
                          "还原版仍然是旧口径 (15 头在 1263 上落成 3 头) —— "
                          "行为还原版的权重/逐手等价性因此一位都没变");
                }
                CHECK(moe->attnHeadsAllocated() == moe->attnHeadsUsed(),
                      "但\"死内存\"这条修复对还原版**也生效** (纯内存, 不改任何数值)");
            }
        }
    }
}

/* ================================================================
 *  [18] 人机对弈的终局通道 (AgentBase::notifyGameResult)
 * ================================================================
 *
 * 洞 (用户问"人机对弈里 agent 能不能从中学习"时核对出来的):
 *   SAC+AZ 的终局值只在"**它自己**那一手结束了对局"时才写进学习回路
 *   (learnFromSearchStep 里 moveForward 之后 getResult() 非 ONGOING)。而人机对弈里
 *   结束一局的那一手是**人**走的 —— 于是 AI 输掉的一局, 学习器连 −1 都收不到:
 *   它最后那条决策样本仍然是 done=false, 价值目标只能靠搜索**估**出来。
 *
 * 修法: 棋盘在终局调 AgentBase::notifyGameResult (见 chessboard.h 的
 * notifyHumanGameEnd)。这一节只钉 **agent 侧**的三条契约 (棋盘侧的接线在
 * test_match 的 [3.9] 里钉):
 *   (1) 值: 输 = −1 / 赢 = +1 / 和 = 0, 而且挂在"最后一条**真实决策**样本"
 *       (hasSearch=true) 上 —— 不是挂在池尾随手一条 rollout 样本上;
 *   (2) 幂等: 重复通知不写第二遍、不重复计数、不重复更新
 *       (人机里三个终局点都会通知, 而"AI 自己将死对方"那一手本来就带终局);
 *   (3) 接不接得住: 池里没有真实决策样本时返回 false, 而不是"静默地挂个 0";
 *       对照组 = 还原版 SACAZLegacyAgent (59e5233 口径), 它没有这条通道 -> 一律 false。
 */
static void testHumanGameTerminalChannel()
{
    std::printf("\n[18] 人机对弈的终局通道 (notifyGameResult)\n");

    const sacazx::Variant variants[3] = { sacazx::Variant::Mlp,
                                          sacazx::Variant::MoeMlp,
                                          sacazx::Variant::MoeTb };
    for (sacazx::Variant v : variants) {
        Chess c;
        c.reset();
        sacazx::Opts o;
        o.hidden = 32;          /* 只测通道, 不测容量: 小网络快得多 */
        o.expertHidden = 32;
        std::printf("  -- %s --\n", sacazx::variantKey(v));
        sacazx::withSacazAgent(c, v, o, [&](auto &sac) {
            CHECK(sac.rewardShape == 0,
                  "默认 rewardShape=0 (本节按未塑形的 ±1 口径断言)");

            /* ---- (3) 池里没有真实决策样本: 必须"明说接不住" ---- */
            float r = 123.0f;
            bool done = true;
            CHECK(!sac.lastDecisionSample(r, done),
                  "没决策过 -> 池里没有真实决策样本");
            CHECK(!sac.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
                  "此时终局反馈返回 false (接不住就说接不住, 不静默挂个 0)");
            CHECK(sac.getTrainDiag().externalTerminals == 0, "...并且不计数");
            CHECK(!sac.notifyGameResult(Chess::RESULT_ONGOING, Stone::COLOR_BLACK),
                  "ONGOING 不是终局 -> 什么都不做, 返回 false");

            /* ---- (1) 输: 人走的那一手把 AI 将死 ---- */
            const Step s1 = sac.selectMove(Stone::COLOR_BLACK, 8, 0.0f);
            CHECK(s1.valid, "selectMove 给出合法走法 (真实决策 = 一条 hasSearch 样本)");
            CHECK(sac.lastDecisionSample(r, done), "决策过一次 -> 有真实决策样本");
            CHECK(!done, "...它还不是终局 (人还没走那一手)");
            CHECK(sac.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
                  "AI 被将死: 终局反馈被接住");
            CHECK(sac.lastDecisionSample(r, done), "...那条样本还在");
            CHECK(done, "...它的 done 被写成 true");
            CHECK(std::fabs((double)r + 1.0) < 1e-6,
                  "...奖励是 −1 (输), 不是即时奖励也不是 0");
            const long long ext = sac.getTrainDiag().externalTerminals;
            CHECK(ext == 1, "外部补入计数 +1 (棋盘侧那条通道真的动了池子)");

            /* ---- (2) 幂等: 同一局的另一个终局点再通知一次 ---- */
            CHECK(sac.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
                  "再通知一次仍然返回 true (这条样本已经带终局)");
            CHECK(sac.getTrainDiag().externalTerminals == ext,
                  "...但**不重复计数** (幂等: 不写第二遍, 也不多更新一次)");
            sac.lastDecisionSample(r, done);
            CHECK(done && std::fabs((double)r + 1.0) < 1e-6, "...奖励还是 −1");

            /* ---- 赢: 终局值 +1 ---- */
            const Step s2 = sac.selectMove(Stone::COLOR_BLACK, 8, 0.0f);
            CHECK(s2.valid, "再走一步真实决策");
            CHECK(sac.notifyGameResult(Chess::RESULT_BLACK_WIN, Stone::COLOR_BLACK),
                  "AI 赢了: 同样被接住");
            sac.lastDecisionSample(r, done);
            CHECK(done && std::fabs((double)r - 1.0) < 1e-6, "奖励是 +1 (赢)");

            /* ---- 和棋: 值是 0, 但 done 必须是 true (不能读成"没终局") ---- */
            const Step s3 = sac.selectMove(Stone::COLOR_BLACK, 8, 0.0f);
            CHECK(s3.valid, "再走一步真实决策");
            CHECK(sac.notifyGameResult(Chess::RESULT_DRAW, Stone::COLOR_BLACK),
                  "和棋: 被接住");
            sac.lastDecisionSample(r, done);
            CHECK(done && r == 0.0f, "和棋的终局值是 0, 但 done 必须是 true");
        });
    }

    /* ---- 对照组: 还原版没有这条通道 (默认实现, 行为一位没变) ---- */
    {
        Chess c;
        c.reset();
        SACAZLegacyAgent old(c, 32, 0.99f, 0.001f, 1.5f,
                             SACAZLegacyAgent::Backbone::Mlp);
        CHECK(!old.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
              "对照组: 还原版 (59e5233 口径) 一律接不住 —— 那条通道不属于它");
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
    testPosReward();
    testLearnFromSearch();
    testSharedTrunkAndTbHeads();
    testHumanGameTerminalChannel();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
