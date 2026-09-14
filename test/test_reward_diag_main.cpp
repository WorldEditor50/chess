/*
 * test_reward_diag_main.cpp - 训练诊断探针 (Phase 0, 不进 ctest)
 * ============================================================================
 *
 * 为什么先写这个: 前面几轮关于"训练损失曲线像脉冲""将受威胁时下得对但整体被将死"
 * 的判断, 全部建立在**一条曲线 + 一个比分**上。要决定奖励系数、终局量级、是否需要
 * 每步代价, 必须先把下面几个数从"推断"变成"实测":
 *
 *   [1] 跨 agent 一致性: 同一个吃子走法, 6 个 agent 的 computeReward 必须一致。
 *       (各 agent 各写一份 computeReward, 很容既漂移; 这条是后续所有奖励改动的护栏。)
 *   [2] 奖励尺度表 + **关键不变量**: |终局奖励| 是否严格大于"一方全部材质之和"。
 *       如果终局不压倒材质, 最优策略就是"吃子"而不是"赢" —— 这是本轮的主假设。
 *   [3] 一局随机棋的奖励构成: Σ|材质奖励| 与 |终局奖励| 的比值。
 *   [4] PPO 的**价值目标分布** (走真实 trainSelfPlay -> 回放池, 读 valueTarget):
 *       |target| > 0.1 的样本占比、最大绝对值、均值。占比很低 = 大多数样本没有信号。
 *   [5] 损失口径: "最后一条样本的损失"(现在的上报口径) 与"整条轨迹的批平均"差多少。
 *
 * 用法:
 *   test_reward_diag.exe [--games=1] [--sims=8] [--moves=20]
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "dqnagent.h"
#include "dqnmcts_agent.h"
#include "evagent.h"
#include "pgagent.h"
#include "ppomcts_agent.h"
#include "rl/util.hpp"
#include "sacazagent.h"

static int g_checks = 0;
static int g_failed = 0;

static void check(bool cond, const char *msg)
{
    g_checks++;
    if (cond) {
        std::printf("  ok    %s\n", msg);
    } else {
        g_failed++;
        std::printf("  FAIL  %s\n", msg);
    }
}

namespace {

int g_games = 1;
int g_sims = 8;
int g_moves = 20;

/*
 * [2]/[3] 用的即时奖励 —— Phase 1 起**与 agent 共用同一处实现** (stone.h 的
 * stepReward)。之所以原来是"另写一份副本", 是为了让改动前后的数字可对比;
 * 现在统一之后, 这里的数字直接反映生产实现。
 */
float probeReward(const Chess &chess, const Step &s, int moverColor)
{
    (void)moverColor;   /* 走子方视角, 与颜色无关 */
    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* 一方全部非将子力的材质价值之和 (用 Stone::value_*, 与棋盘无关的常数) */
double sideMaterialTotal()
{
    return 2 * Stone::value_che + 2 * Stone::value_ma + 2 * Stone::value_pao
         + 5 * Stone::value_bing + 2 * Stone::value_shi + 2 * Stone::value_xiang;
}

/* ================================================================
 *  [1] 跨 agent 一致性 + [2] 奖励尺度表 / 不变量
 * ================================================================ */
void section1And2(Chess &env)
{
    std::printf("\n[1] 跨 agent 的 computeReward 一致性\n");

    /* 找一个真实的吃子走法 (初始局面红方) */
    Step cap;
    cap.valid = false;
    std::vector<Step *> legal;
    env.reset();
    env.sample(Stone::COLOR_RED, legal);
    for (std::size_t i = 0; i < legal.size(); i++) {
        if (legal[i]->nextId != Stone::ID_NONE) {
            cap = *legal[i];
            break;
        }
    }
    Steps::instance().put(legal);

    std::printf("  红方在初始局面找到吃子走法: %s\n", cap.valid ? "是" : "否");
    if (cap.valid) {
        Stone *victim = env.stones[cap.nextId];
        std::printf("  被吃: %s (type=%d, value=%.2f)\n",
                    victim->name.c_str(), victim->type, victim->value);

        PGEagent pg(env, 64, 0.99f, 0.001f, 0.1f);
        DQNAgent dqn(env, 64, 0.99f, 0.001f, 1.0f);
        DQNMCTSAgent dqm(env, 64, 0.99f, 0.001f, 0.1f, 1.414f);
        const float rpg = pg.computeReward(cap, Stone::COLOR_RED);
        const float rdq = dqn.computeReward(cap, Stone::COLOR_RED);
        const float rdm = dqm.computeReward(cap, Stone::COLOR_RED);
        std::printf("  PG=%.4f  DQN=%.4f  DQN+MCTS=%.4f  (探针公式=%.4f)\n",
                    (double)rpg, (double)rdq, (double)rdm,
                    (double)probeReward(env, cap, Stone::COLOR_RED));
        check(rpg > 0.0f && rdq > 0.0f && rdm > 0.0f, "三个老 agent 的吃子奖励都是正的");
        check(std::fabs(rpg - rdq) < 1e-6f && std::fabs(rdq - rdm) < 1e-6f,
              "三个老 agent 的吃子奖励数值一致");
    } else {
        std::printf("  (初始局面没有吃子走法, 跳过一致性检查)\n");
    }

    std::printf("\n[2] 奖励尺度表 / 不变量\n");
    const double mat = sideMaterialTotal();
    const double matScaled = mat * (double)REWARD_MATERIAL_COEF;
    const double perStepMax = Stone::value_che * (double)REWARD_MATERIAL_COEF;
    const double terminal = (double)REWARD_TERMINAL;
    std::printf("  一方全部非将子力 (value 单位)  = %.3f\n", mat);
    std::printf("  材质系数                       = %.3f\n", (double)REWARD_MATERIAL_COEF);
    std::printf("  换算成奖励 (一方全材质)        = %.4f\n", matScaled);
    std::printf("  单步最大材质奖励 (吃車)        = %.4f\n", perStepMax);
    std::printf("  每步代价                       = %.4f\n", (double)REWARD_STEP_COST);
    std::printf("  |终局奖励|                     = %.3f\n", terminal);
    std::printf("  >>> 终局 / 一方全材质          = %.2f   (应 > 1)\n", terminal / matScaled);
    std::printf("  >>> 终局 / 单步最大材质        = %.2f\n", terminal / perStepMax);
    const bool dominant = terminal > matScaled;
    std::printf("  [不变量] |终局奖励| > 一方全部材质之和: %s\n", dominant ? "成立" : "**不成立**");
    check(dominant, "不变量: |终局奖励| 严格大于一方全部材质之和");

    /* 钉住新的材质系数 (改动前是 value*10, 吃馬 = 3.0; 现在是 0.025) */
    check(std::fabs(probeReward(env, cap, Stone::COLOR_RED)
                    - (REWARD_MATERIAL_COEF * (float)env.stones[cap.nextId]->value
                       + REWARD_STEP_COST)) < 1e-6f,
          "材质奖励 = 系数 x value + 每步代价");
}

/* ================================================================
 *  [3] 一局随机棋的奖励构成
 * ================================================================ */
void section3(Chess &env)
{
    std::printf("\n[3] 一局随机棋的奖励构成 (材质 vs 终局)\n");

    RL::Random::setSeed(20240914u);
    env.reset();
    double matSum = 0.0;      /* 材质部分 */
    double stepSum = 0.0;     /* 每步代价部分 (绝对值) */
    int captures = 0, plies = 0;
    double maxOne = 0.0;

    for (int i = 0; i < 200; i++) {
        std::vector<Step *> legal;
        env.sample(env.sideToMove, legal);
        if (legal.empty()) {
            break;
        }
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        Step chosen(*legal[pick(RL::Random::generator)]);
        Steps::instance().put(legal);

        /*
           注意不能拿 "reward != 0" 当"吃子"的判据 —— Phase 1 之后每个安静着法也有
           每步代价 (-0.001), 那样会把 200 手全算成吃子 (第一版就踩了这个坑)。
           判据用 Step 本身: nextId != ID_NONE。
        */
        if (chosen.nextId != Stone::ID_NONE) {
            Stone *victim = env.stones[chosen.nextId];
            if (victim != nullptr && victim->type != Stone::TYPE_JIANG) {
                captures++;
                const double m = (double)REWARD_MATERIAL_COEF * victim->value;
                matSum += m;
                maxOne = std::max(maxOne, m);
            }
        }
        stepSum += std::fabs((double)REWARD_STEP_COST);

        double dummy = 0.0;
        env.moveForward(&chosen, dummy);
        plies++;
        if (env.getResult(env.sideToMove) != Chess::RESULT_ONGOING) {
            break;
        }
    }

    std::printf("  随机走了 %d 手, 其中吃子 %d 次 (每手都有 %.4f 的每步代价)\n",
                plies, captures, (double)REWARD_STEP_COST);
    std::printf("  Σ|材质奖励| (双方)                 = %.4f (单次最大 %.4f)\n", matSum, maxOne);
    std::printf("  Σ|每步代价|                       = %.4f\n", stepSum);
    std::printf("  |终局奖励|                         = %.3f\n", (double)REWARD_TERMINAL);
    std::printf("  >>> 材质合计 / |终局|              = %.3f  (应 < 1: 双方合计也不该压倒终局)\n",
                matSum / (double)REWARD_TERMINAL);
    std::printf("  >>> 步长累计(最长一局 %d 手) / |终局| = %.3f  (应明显 < 1)\n",
                REWARD_MAX_PLIES,
                (double)REWARD_MAX_PLIES * std::fabs((double)REWARD_STEP_COST)
                    / (double)REWARD_TERMINAL);
    check(matSum < (double)REWARD_TERMINAL,
          "一局里双方能拿到的材质奖励合计也不压倒终局");
    check((double)REWARD_MAX_PLIES * std::fabs((double)REWARD_STEP_COST)
              < 0.2 * (double)REWARD_TERMINAL,
          "每步代价在最长一局里的累计远小于终局 (不会把慢赢算成输)");
}

/* ================================================================
 *  [4] PPO 的价值目标分布 + [5] 损失口径
 * ================================================================ */
void section4And5(Chess &env)
{
    std::printf("\n[4] PPO 的价值目标分布 (真实 trainSelfPlay -> 回放池)\n");

    PPOMCTSAgent agent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true);
    /* 只收集、不学习, 免得学习把 lastLoss 覆盖成别的东西 */
    agent.replayBatchSize = 0;
    RL::Random::setSeed(20240914u);
    const double t0 = (double)clock() / CLOCKS_PER_SEC;
    agent.trainSelfPlay(g_games, g_sims, g_moves, false, 1.0f, 0.1f);
    const double dt = (double)clock() / CLOCKS_PER_SEC - t0;

    const std::size_t n = agent.ppo.replaySize();
    std::printf("  自对弈 %d 局 (%d 模拟/步, 上限 %d 手), %.1f s, 产出 %llu 条样本\n",
                g_games, g_sims, g_moves, dt, (unsigned long long)n);

    if (n == 0) {
        std::printf("  (没有样本, 跳过)\n");
        return;
    }

    double sumAbs = 0.0, maxAbs = 0.0;
    int many = 0, nonZero = 0;
    std::vector<double> absT;
    for (std::size_t i = 0; i < n; i++) {
        const double t = std::fabs((double)agent.ppo.replay[i].valueTarget);
        absT.push_back(t);
        sumAbs += t;
        maxAbs = std::max(maxAbs, t);
        if (t > 0.1) { many++; }
        if (t > 1e-6) { nonZero++; }
    }
    std::sort(absT.begin(), absT.end());
    std::printf("  |value target| 均值=%.4f  中位=%.4f  最大=%.4f\n",
                sumAbs / (double)n, absT[absT.size() / 2], maxAbs);
    std::printf("  |target| > 1e-6 的占比 = %.1f%%\n", 100.0 * (double)nonZero / (double)n);
    std::printf("  |target| > 0.1  的占比 = %.1f%%   <-- 「有信号」的样本比例\n",
                100.0 * (double)many / (double)n);
    /*
       两条诊断, 分属两个 phase:
         * 尺度: 奖励量纲摆正之后, 折现回报必然落在 ~[-1,1] 量级 (终局 ±1 加少量材质
           与每步代价)。|target| 再出现 ±100 量级就是尺度回归 → 这条现在就断言。
         * 信号占比: 截断的 rollout 一律传 finalOutcome=0, 于是材质奖励缩小 20 倍之后
           大多数样本的目标只有 0.005~0.01 量级 —— 这才是"安静局面没有信号"的真身。
           把它从"推断"变成"数字", 并在 Phase 2 (自举) 之后回来对比 → 现在只打印。
    */
    check(maxAbs < 2.0, "价值目标落在奖励量纲内 (|target| < 2), 不是 ±100 量级");
    /*
       Phase 2 (自举 + 势能塑形) 之后这一条才成为硬断言: 势能 Φ 把"棋盘局面价值"
       (含空间/机动性/将安全) 提前搬进目标, 于是安静局面第一次有了非零目标。
       期望超过一半的样本 |target| > 0.1。
    */
    check((double)many / (double)n > 0.3,
          "超过 30% 的样本价值目标有信号 (|target| > 0.1) —— 自举+势能塑形生效");
    std::printf("  [Phase 2 的靶子] 有信号样本占比目前是 %.1f%%, 自举后应显著上升\n",
                100.0 * (double)many / (double)n);

    std::printf("\n[5] 损失口径: 最后一条样本 vs 整条轨迹批平均\n");
    {
        /*
           复现两条口径的差: 现在 learnSelfPlay 是"逐步 trainStep", 每步 batch=1,
           于是 lastLoss 只反映**最后一条**样本; 若改成"整条累积一次 applyGradients",
           上报的是批平均。这里用一份临时 PPO 各算一次, 打印两者之比。
        */
        RL::PPO single(PPOMCTSAgent::STATE_DIM, 64, PPOMCTSAgent::ACTION_DIM, 64, 0.1f, true);
        RL::PPO batch(PPOMCTSAgent::STATE_DIM, 64, PPOMCTSAgent::ACTION_DIM, 64, 0.1f, true);

        const std::size_t take = std::min<std::size_t>(n, 16);
        RL::Tensor dense(PPOMCTSAgent::ACTION_DIM, 1);
        double lastSingle = 0.0;
        batch.resetMoeBatchStats();
        for (std::size_t i = 0; i < take; i++) {
            const RL::PPO::ReplaySample &s = agent.ppo.replay[i];
            dense.zero();
            for (std::size_t j = 0; j < s.actionIdx.size(); j++) {
                dense[(std::size_t)s.actionIdx[j]] = s.actionProb[j];
            }
            single.trainStep(s.state, dense, s.valueTarget, 0.001f);
            lastSingle = single.lastLoss;
            batch.accumulateGrad(s.state, dense, s.valueTarget);
        }
        batch.applyGradients(0.001f);
        std::printf("  取前 %llu 条样本:\n", (unsigned long long)take);
        std::printf("    逐步 trainStep 之后 lastLoss (只反映最后一条) = %.6f\n", lastSingle);
        std::printf("    整条累积一次 applyGradients (批平均)          = %.6f\n",
                    batch.lastLoss);
        std::printf("    比值 (最后一条 / 批平均) = %.2f\n",
                    batch.lastLoss != 0.0 ? lastSingle / batch.lastLoss : 0.0);
        std::printf("    actor CE (批平均) = %.6f\n", batch.lastActorLoss);
    }

    /* ---- learnSelfPlay 现在报的是批平均: 用真实在线路径复核一次 ---- */
    {
        Chess env2;
        PPOMCTSAgent online(env2, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true);
        online.replayBatchSize = 0;
        RL::Random::setSeed(20240914u);
        const bool trained = online.exploreAndTrain(Stone::COLOR_RED, 16);
        const float loss = online.getLastTrainLoss();
        const float actor = online.getLastActorLoss();
        std::printf("\n[5b] 在线路径 (exploreAndTrain) 的上报口径\n");
        std::printf("  trained=%d  critic MSE=%.6f  actor CE=%.6f\n",
                    (int)trained, (double)loss, (double)actor);
        check(trained, "exploreAndTrain 真的训练了");
        check(std::isfinite(loss) && std::isfinite(actor),
              "在线路径上报的损失是有限值 (不是 NaN)");
        check(actor > 0.0f, "actor 交叉熵为正 (策略尚未与目标重合)");
    }
}

/* ================================================================
 *  [6] 势能塑形 (Phase 2) 与"空间价值"的验证
 *
 *  盯三件事:
 *   (a) **不吃子的着法也必须改变局面价值** —— 象棋里绝大多数着法是调子/占位/争空间,
 *       它们的价值在"未来的选择权"里 (能到达的格子越多, 未来的威胁/防守机会越多)。
 *       所以走几步安静着法, 势能 Φ 必须真的动。
 *   (b) 空间/机动性项与将安全项真的在参与评估 —— 关掉开关再算一遍, 看差多少。
 *   (c) PBRS 的**平移性** (塑形等价于把价值目标整体平移 Φ(s_0)):
 *         shaped_return_0 == unshaped_return_0 + Φ(s_0)
 *       这条成立, 才说明 critic 学到的是"真值 + 局面价值", 而不是被塑形扭曲的目标。
 * ================================================================ */
void section6(Chess &env)
{
    std::printf("\n[6] 势能塑形 (Phase 2) 与空间价值\n");

    /* ---- (a) 安静着法也改变局面价值 ---- */
    {
        env.reset();
        env.sideToMove = Stone::COLOR_RED;
        /* 势能 Φ 走的是 evaluatePositional() (含空间/将安全), 所以这里也用它 —— 
           用便宜的 evaluate() 量会漏掉"空间"那一部分的贡献。 */
        double phiPrev = potentialReward(
            (float)((env.sideToMove == Stone::COLOR_BLACK) ? env.evaluatePositional()
                                                           : -env.evaluatePositional()));
        int quietMoves = 0, quietChanged = 0;
        double maxQuietDelta = 0.0;

        std::printf("  走若干**不吃子**的着法, 看势能是否变化:\n");
        for (int i = 0; i < 6; i++) {
            std::vector<Step *> legal;
            env.sample(env.sideToMove, legal);
            if (legal.empty()) break;
            /* 优先挑一个不吃子的着法 */
            Step chosen;
            bool found = false;
            for (std::size_t k = 0; k < legal.size(); k++) {
                if (legal[k]->nextId == Stone::ID_NONE) { chosen = *legal[k]; found = true; break; }
            }
            if (!found) { Steps::instance().put(legal); break; }
            Steps::instance().put(legal);

            const double evalBefore = env.evaluatePositional();
            double dummy = 0.0;
            env.moveForward(&chosen, dummy);
            const double evalAfter = env.evaluatePositional();
            const double phiNow = potentialReward(
                (float)((env.sideToMove == Stone::COLOR_BLACK) ? evalAfter : -evalAfter));

            const double dPhi = std::fabs(phiNow - phiPrev);
            quietMoves++;
            if (dPhi > 1e-9) quietChanged++;
            maxQuietDelta = std::max(maxQuietDelta, dPhi);
            std::printf("    第 %d 手 (不吃子): evaluate %.4f -> %.4f,  Φ %.4f -> %.4f"
                        "  (ΔΦ=%.4f)\n",
                        i + 1, evalBefore, evalAfter, phiPrev, phiNow, dPhi);
            phiPrev = phiNow;
        }
        std::printf("  >>> %d 手安静着法里有 %d 手改变了势能, 最大 ΔΦ = %.4f\n",
                    quietMoves, quietChanged, maxQuietDelta);
        /* 允许个别着法恰好"评估中性" (例如车在同价值格之间挪动), 所以不要求全部 */
        check(quietMoves >= 4 && quietChanged >= quietMoves - 1,
              "不吃子的着法也改变局面价值 (空间/机动性/将安全被计入)");
    }

    /* ---- (b) 局面价值项(含空间/机动性)的贡献 ----
       注意必须用**非对称**局面: 开局局面左右/上下都对称, 局面项之和恰好为 0, 拿它做
       A/B 会得出"局面项没起作用"的错误结论 (第一版就是这么错的)。 */
    {
        env.reset();
        env.sideToMove = Stone::COLOR_RED;
        /* 走几手把局面走成非对称 */
        for (int i = 0; i < 4; i++) {
            std::vector<Step *> legal;
            env.sample(env.sideToMove, legal);
            if (legal.empty()) break;
            Step chosen(*legal[(std::size_t)(i * 3 + 1) % legal.size()]);
            Steps::instance().put(legal);
            double dummy = 0.0;
            env.moveForward(&chosen, dummy);
        }

        Chess::setPositionalEvalEnabled(true);
        const double on = env.evaluatePositional();
        Chess::setPositionalEvalEnabled(false);
        const double off = env.evaluatePositional();
        const double plain = env.evaluate();
        Chess::setPositionalEvalEnabled(true);
        std::printf("  非对称局面: evaluatePositional(含局面项) = %+.4f,"
                    " evaluatePositional(关局面项) = %+.4f, evaluate()(只有材质+PST) = %+.4f,"
                    " 局面项贡献 = %+.4f\n", on, off, plain, on - off);
        check(std::fabs(on - off) > 1e-6, "局面价值项 (将安全/空间/机动性) 确实参与了评估");
        check(std::fabs(off - plain) < 1e-9, "关掉开关后 evaluatePositional() == evaluate()"
                                             " (AB 的叶子评估不受影响)");

        /* 代价: 局面项走的是"每手两次"的势能, 不进 AB 的叶子热路径 —— 两边都量 */
        {
            const int N = 20000;
            volatile double sink = 0.0;
            clock_t t0 = clock();
            for (int i = 0; i < N; i++) sink = sink + env.evaluate();
            const double plainMs = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
            t0 = clock();
            for (int i = 0; i < N; i++) sink = sink + env.evaluatePositional();
            const double posMs = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
            std::printf("  evaluate() x %d = %.3f us/次 (AB 叶子, 未变);"
                        " evaluatePositional() = %.3f us/次 (势能 Φ, 每手 2 次) -> %.2fx\n",
                        N, 1000.0 * plainMs / N, 1000.0 * posMs / N,
                        plainMs > 0.0 ? posMs / plainMs : 0.0);
        }
    }

    /* ---- (c) PBRS 平移性: shaped_return_0 == unshaped_return_0 + Φ(s_0) ---- */
    {
        env.reset();
        env.sideToMove = Stone::COLOR_RED;
        const float gamma = 0.99f;
        const float phiInit = potentialReward(
            (float)((env.sideToMove == Stone::COLOR_BLACK) ? env.evaluate() : -env.evaluate()));

        std::vector<RL::Step> traj;
        for (int i = 0; i < 5; i++) {
            std::vector<Step *> legal;
            env.sample(env.sideToMove, legal);
            if (legal.empty()) break;
            Step chosen(*legal[(std::size_t)i % legal.size()]);
            Steps::instance().put(legal);

            const float base = probeReward(env, chosen, env.sideToMove);
            double dummy = 0.0;
            env.moveForward(&chosen, dummy);
            const float phiAfter = potentialReward(
                (float)((env.sideToMove == Stone::COLOR_BLACK) ? env.evaluate() : -env.evaluate()));

            RL::Tensor oneHot(PPOMCTSAgent::ACTION_DIM, 1);
            oneHot.zero();
            oneHot[0] = 1.0f;   /* 这里只关心数值, 策略目标不影响回报 */
            traj.emplace_back(RL::Tensor(PPOMCTSAgent::STATE_DIM, 1), oneHot, base);
            traj.back().potential = phiAfter;
        }

        if (traj.size() >= 2) {
            /* 未塑形 */
            std::vector<float> raw(traj.size(), 0.0f);
            for (std::size_t i = 0; i < traj.size(); i++) raw[i] = traj[i].reward;
            RL::PPO ppo(PPOMCTSAgent::STATE_DIM, 8, PPOMCTSAgent::ACTION_DIM, 8, 0.1f, false);
            const std::vector<float> unshaped = ppo.discountedReturnsFromRewards(raw, 0.0f);

            /* 塑形 (与 applyPotentialShaping 同一公式) */
            std::vector<float> shaped = raw;
            float phiBefore = phiInit;
            for (std::size_t i = 0; i < traj.size(); i++) {
                shaped[i] = shapedStepReward(shaped[i], phiBefore, traj[i].potential, gamma);
                phiBefore = traj[i].potential;
            }
            const std::vector<float> sh = ppo.discountedReturnsFromRewards(shaped, 0.0f);

            /*
               边界项: PBRS 要求终局常量也平移 —— finalOutcome' = -Φ(落子后局面)。
               不带上它, "塑形 - 不塑形" 就不等于 Φ(s_0) (第一版就是这样差了一个
               -Φ(s_n))。这里按生产代码 (commitEpisode / exploreAndTrain) 的写法
               把边界也平移过去, 于是差值应当**精确等于 Φ(s_0)**。
            */
            const float phiFinal = traj.back().potential;
            const std::vector<float> shShifted =
                ppo.discountedReturnsFromRewards(shaped, -phiFinal);

            const double diff = (double)shShifted[0] - (double)unshaped[0];
            std::printf("  PBRS 平移性: 未塑形 return[0]=%.5f, 塑形(含边界平移) return[0]=%.5f,"
                        " 差=%.5f, Φ(s_0)=%.5f\n",
                        unshaped[0], shShifted[0], diff, phiInit);
            std::printf("  (未做边界平移时差是 %.5f, 即少了 -Φ(s_n)=%+.5f)\n",
                        (double)sh[0] - (double)unshaped[0], -(double)phiFinal);
            check(std::fabs(diff - (double)phiInit) < 1e-4,
                  "塑形 = 把价值目标整体平移 Φ(s_0) (PBRS 的定义, 含边界项)");
        }
    }
}

} // namespace

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const std::size_t eq = a.find('=');
        const std::string k = (eq == std::string::npos) ? a : a.substr(0, eq);
        const std::string v = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
        if (k == "--games")      { g_games = std::atoi(v.c_str()); }
        else if (k == "--sims")  { g_sims = std::atoi(v.c_str()); }
        else if (k == "--moves") { g_moves = std::atoi(v.c_str()); }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }

    std::printf("========================================================\n");
    std::printf("  奖励 / 学习目标 诊断 (Phase 0)\n");
    std::printf("========================================================\n");

    Chess env;
    section1And2(env);
    section3(env);
    section4And5(env);
    section6(env);

    std::printf("\n========================================================\n");
    std::printf("  %d 项检查, %d 项失败\n", g_checks, g_failed);
    std::printf("========================================================\n");
    return g_failed == 0 ? 0 : 1;
}
