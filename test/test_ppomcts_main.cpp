#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include "rl/cpuinfo.hpp"
#include "ppomcts_agent.h"
#include "mcts.h"
#include "test_utils.h"

/* ================================================================
 *  PPO+MCTS (AlphaZero-style) Agent 测试程序
 *
 *  测试内容:
 *    1. 单步推理测试 (PPO+MCTS搜索一步)
 *    2. 自对弈训练测试 (少量迭代验证算法正确性)
 *    3. 对比随机走棋测试
 *    4. 与传统MCTS性能对比
 * ================================================================ */

static void printBoard(Chess &chess)
{
    printf("\n    0   1   2   3   4   5   6   7   8\n");
    printf("  -----------------------------\n");
    for (int i = 0; i < 10; i++) {
        printf("%d |", i);
        for (int j = 0; j < 9; j++) {
            Stone *s = chess.m_map[Pos(i, j)];
            if (s == nullptr || !s->alive) {
                printf("   ");
            } else {
                const char *names[] = {"车","马","炮","兵","帅","仕","相"};
                int type = s->type;
                printf(" %s", (type >= 0 && type < 7) ? names[type] : "?");
            }
            if (j < 8) printf("|");
        }
        printf("|\n");
        if (i < 9) {
            printf("  -----------------------------\n");
        }
    }
    printf("  -----------------------------\n\n");
}

/* ================================================================
 *  测试1: 单步推理
 * ================================================================ */
static void testInference()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS: \xe6\x8e\xa8\xe7\x90\x86\xe6\xb5\x8b\xe8\xaf\x95\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 1.414f);

    chess.reset();
    printBoard(chess);

    Timer timer;
    Step best = agent.selectMove(Stone::COLOR_BLACK, 100, 0.0f);
    long long elapsed = timer.elapsedMs();

    printf("  PPO+MCTS(100 sims) \xe6\x9c\x80\xe4\xbc\x98\xe8\xb5\xb0\xe6\xb3\x95: "
           "id=%d, (%d,%d)->(%d,%d) [%lld ms]\n",
           best.id, best.pos.x, best.pos.y,
           best.nextPos.x, best.nextPos.y, elapsed);

    /* Run a second search to compare */
    timer.reset();
    Step best2 = agent.selectMove(Stone::COLOR_BLACK, 500, 0.0f);
    elapsed = timer.elapsedMs();
    printf("  PPO+MCTS(500 sims) \xe6\x9c\x80\xe4\xbc\x98\xe8\xb5\xb0\xe6\xb3\x95: "
           "id=%d, (%d,%d)->(%d,%d) [%lld ms]\n",
           best2.id, best2.pos.x, best2.pos.y,
           best2.nextPos.x, best2.nextPos.y, elapsed);
}

/* ================================================================
 *  测试2: 自对弈训练 (少量迭代)
 * ================================================================ */
static void testSelfPlay()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS: \xe8\x87\xaa\xe5\xaf\xb9\xe5\xbc\x88\xe8\xae\xad\xe7\xbb\x83 (5\xe5\xb1\x80)\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 2.0f);

    agent.trainSelfPlay(5, 50, 100, true);

    printf("\n--- \xe7\xbb\x9f\xe8\xae\xa1 ---\n");
    printf("  \xe6\x80\xbb\xe5\xaf\xb9\xe5\xb1\x80: %d\n", agent.getTotalEpisodes());
    printf("  \xe9\xbb\x91\xe6\x96\xb9\xe8\x83\x9c\xe7\x8e\x87: %.2f%%\n",
           agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);
    printf("  \xe7\xba\xa2\xe6\x96\xb9\xe8\x83\x9c\xe7\x8e\x87: %.2f%%\n",
           agent.getWinRate(Stone::COLOR_RED) * 100.0f);
}

/* ================================================================
 *  测试3: PPO+MCTS vs 随机走棋
 * ================================================================ */
static void testVsRandom()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS \xe5\xaf\xb9\xe6\xaf\x94\xe9\x9a\x8f\xe6\x9c\xba\xe8\xb5\xb0\xe6\xa3\x8b\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 2.0f);

    int ppoWins = 0, randomWins = 0, draws = 0;
    const int games = 4;

    for (int g = 0; g < games; g++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;

        for (int moves = 0; moves < 100; moves++) {
            Step move;
            if (currentColor == Stone::COLOR_BLACK) {
                move = agent.selectMove(Stone::COLOR_BLACK, 100, 0.0f);
            } else {
                std::vector<Step*> steps;
                chess.sample(Stone::COLOR_RED, steps);
                if (steps.empty()) { ppoWins++; Steps::instance().put(steps); break; }
                int idx = std::rand() % (int)steps.size();
                move = *steps[idx];
                Steps::instance().put(steps);
            }

            if (!move.valid) {
                if (currentColor == Stone::COLOR_BLACK) randomWins++; else ppoWins++;
                break;
            }

            double dummy = 0.0;
            chess.moveForward(&move, dummy);

            int result = chess.isGameOver();
            if (result != Stone::COLOR_NONE) {
                if (result == Stone::COLOR_BLACK) ppoWins++;
                else if (result == Stone::COLOR_RED) randomWins++;
                break;
            }

            currentColor = (currentColor == Stone::COLOR_RED)
                               ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            if (moves >= 99) { draws++; break; }
        }
    }

    printf("  PPO+MCTS(\xe9\xbb\x91): %d\xe8\x83\x9c, \xe9\x9a\x8f\xe6\x9c\xba(\xe7\xba\xa2): %d\xe8\x83\x9c, \xe5\xb9\xb3\xe5\xb1\x80: %d\n",
           ppoWins, randomWins, draws);
}

/* ================================================================
 *  测试4: 与传统MCTS性能对比 (单步搜索时间)
 * ================================================================ */
static void testVsTraditionalMCTS()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS vs \xe4\xbc\xa0\xe7\xbb\x9fMCTS \xe6\x80\xa7\xe8\x83\xbd\xe5\xaf\xb9\xe6\xaf\x94\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 1.414f);
    chess.reset();

    int sims[] = {50, 100, 200};

    for (int s : sims) {
        /* PPO+MCTS */
        Timer timer;
        Step ppomove = agent.selectMove(Stone::COLOR_BLACK, s, 0.0f);
        long long ppoMs = timer.elapsedMs();

        /* Traditional MCTS (with random rollouts) */
        MCTS mcts(chess, 1.414);
        timer.reset();
        Step mctsMove = mcts.findBestMove(Stone::COLOR_BLACK, s);
        long long mctsMs = timer.elapsedMs();

        printf("  sims=%4d: PPO+MCTS=%4lld ms, MCTS(random rollout)=%4lld ms\n",
               s, ppoMs, mctsMs);
    }
}

/* ================================================================
 *  测试5: 稀疏 MoE 骨干 / 编码的诊断
 *
 *  稀疏 MoE 的经典失败模式是**路由坍缩**: softmax 的反向会持续压低"没被选中"的
 *  专家的门控概率, 于是几轮更新之内少数专家就吃掉全部流量, 其余永远不训练。
 *  RL::PPO 为此注入了负载均衡辅助损失 (见 rl/ppo.h)。这里把每个专家被选中的次数
 *  打出来 —— 只要 8 个专家都还在被使用, 就说明辅助损失压住了坍缩。
 *
 *  同时报告参数量与新编码的维度, 让"换骨干 / 换编码"这件事有可比的数字。
 * ================================================================ */
static void printExpertUsage(PPOMCTSAgent &agent, const char *label)
{
    std::vector<long long> usage;
    agent.moeUsage(usage);
    if (usage.empty()) {
        printf("%s: (no sparse MoE layer)\n", label);
        return;
    }
    long long total = 0;
    long long lo = -1;
    long long hi = 0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        total += usage[i];
        if (lo < 0 || usage[i] < lo) lo = usage[i];
        if (usage[i] > hi) hi = usage[i];
    }
    printf("%s\n    usage = [", label);
    for (std::size_t i = 0; i < usage.size(); i++) {
        printf("%s%lld", (i ? ", " : ""), usage[i]);
    }
    /* max/min 越接近 1 越均匀; moeUsage 把 actor 与 critic 的计数相加了 */
    printf("]  total=%lld  max/min=", total);
    if (lo > 0) {
        printf("%.2f\n", (double)hi / (double)lo);
    } else {
        printf("inf (some expert never selected)\n");
    }
}

static void testSparseMoeBackbone()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS: \xe7\xa8\x80\xe7\x96\x8f MoE \xe9\xaa\xa8\xe5\xb9\xb2\xe8\xaf\x8a\xe6\x96\xad\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 1.414f);
    chess.reset();

    printf("  STATE_DIM=%d  ACTION_DIM=%d\n",
           PPOMCTSAgent::STATE_DIM, PPOMCTSAgent::ACTION_DIM);
    printf("  sparse MoE layers=%d  experts=%d  top-k=%d\n",
           agent.moeLayerCount(), agent.moeExpertCount(), agent.moeTopK());
    printf("  params: actor=%lld  critic=%lld  total=%lld\n",
           agent.actorParamCount(), agent.criticParamCount(),
           agent.actorParamCount() + agent.criticParamCount());

    /* ---- 推理阶段的路由 ---- */
    agent.resetMoeUsage();
    for (int i = 0; i < 8; i++) {
        agent.selectMove(Stone::COLOR_BLACK, 40, 0.0f);
    }
    printExpertUsage(agent, "  [inference] 8 searches x 40 sims");

    /* ---- 训练阶段的路由 (走 PPO 更新, 辅助损失生效) ----
       先清零: 否则这一份统计会把上面的推理阶段也算进去, 两个快照就不独立了 */
    agent.resetMoeUsage();
    for (int i = 0; i < 8; i++) {
        agent.exploreAndTrain(Stone::COLOR_BLACK, 16);
    }
    printExpertUsage(agent, "  [after training] 8 x exploreAndTrain(16)");
    printf("  critic value MSE after training: %g\n", agent.getLastTrainLoss());
}

/* ================================================================
 *  测试6: 算力预算 —— 搜索 vs 学习
 *
 *  为什么值得单独量一次: 这两者的比例决定了"能不能靠多过几遍同一批数据来榨干它"。
 *  如果学习只占总算力的一小块, 那么"每条样本只更新一次然后丢掉"就是纯粹的浪费 ——
 *  把同一批数据再过多几轮几乎不额外花钱。
 * ================================================================ */
static void testComputeBudget()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS: compute budget (search vs learn)\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 64, 0.99f, 0.001f, 1.414f);
    chess.reset();

    const int sims = 80;

    /* ---- 纯搜索: selectMove 里除了搜索没有别的 ---- */
    const int searchMoves = 30;
    Timer timer;
    for (int i = 0; i < searchMoves; i++) {
        agent.selectMove(Stone::COLOR_BLACK, sims, 0.0f);
    }
    const double searchPerMove = (double)timer.elapsedMs() / (double)searchMoves;

    /* ---- 纯学习: 一次 trainStep = actor+critic 前向+反向 + 优化器 ---- */
    RL::Tensor state(PPOMCTSAgent::STATE_DIM, 1);
    agent.encodeState(state);
    RL::Tensor target(PPOMCTSAgent::ACTION_DIM, 1);
    target.zero();
    target[0] = 1.0f;   /* 只关心成本, 目标是哪个动作不影响计时 */

    const int trainSteps = 300;
    for (int i = 0; i < 5; i++) {          /* 预热: 别把首次分配算进平均值 */
        agent.ppo.trainStep(state, target, 0.0f, 0.001f);
    }
    timer.reset();
    for (int i = 0; i < trainSteps; i++) {
        agent.ppo.trainStep(state, target, 0.0f, 0.001f);
    }
    const double trainPerStep = (double)timer.elapsedMs() / (double)trainSteps;

    /*
       把 trainStep 拆开: 纯前向 (actor / critic) 各占多少。
       trainStep = 前向 + 反向 + 损失 + 优化器 + 每步的张量分配, 拆开之后才知道
       "梯度累积能省下的那部分"到底有多大。
    */
    const int fwdIters = 300;
    for (int i = 0; i < 5; i++) { agent.ppo.action(state); }
    timer.reset();
    for (int i = 0; i < fwdIters; i++) { agent.ppo.action(state); }
    const double actorFwd = (double)timer.elapsedMs() / (double)fwdIters;

    for (int i = 0; i < 5; i++) { agent.ppo.value(state); }
    timer.reset();
    for (int i = 0; i < fwdIters; i++) { agent.ppo.value(state); }
    const double criticFwd = (double)timer.elapsedMs() / (double)fwdIters;

    /*
       只测优化器 (RMSProp 两个网络)。梯度会被 RMSProp 清零, 但它对零梯度做的
       工作量与正常一步完全相同 (读写 w/v/g 三张全参数张量), 所以计时是有效的。
       这个数直接决定"梯度累积能省多少": 累积 B 条样本才做 1 次优化器, 那部分成本
       就被摊薄 B 倍。
    */
    timer.reset();
    for (int i = 0; i < fwdIters; i++) {
        agent.ppo.actorP.RMSProp(0.001f, 0.9f, 0.001f);
        agent.ppo.critic.RMSProp(0.001f, 0.9f, 0.001f);
    }
    const double optMs = (double)timer.elapsedMs() / (double)fwdIters;
    /*
       注意: 这里**不**单独对比 clipGrad=false 的耗时。试过 —— 那样测出来反而慢 4 倍,
       但那是测量本身的假象: 这个循环反复调用 RMSProp 而没有重新 backward, 梯度被第一次
       调用清零后, "不归一化"的那一路会让 w/v 走进退化区间 (v 衰减到浮点非规格化数)。
       真实训练每步都有新梯度, 不适用。要评 clipGrad 的真实代价, 必须跑一次真正的
       "前向+反向+优化" A/B, 而且它同时会改变学习语义 —— 这里不做。
    */

    const double perPly = searchPerMove + trainPerStep;
    const double fwdMs = actorFwd + criticFwd;
    const double restMs = trainPerStep - fwdMs - optMs;
    printf("  search : %7.2f ms / %d-sim move  (%.4f ms/sim)\n",
           searchPerMove, sims, searchPerMove / (double)sims);
    printf("  learn  : %7.2f ms / trainStep (1 sample)\n", trainPerStep);
    printf("    forward (actor %.2f + critic %.2f) = %6.2f ms  %4.0f%%\n",
           actorFwd, criticFwd, fwdMs, 100.0 * fwdMs / trainPerStep);
    printf("    optimizer (RMSProp both nets)    = %6.2f ms  %4.0f%%\n",
           optMs, 100.0 * optMs / trainPerStep);
    printf("    backward + loss + alloc          = %6.2f ms  %4.0f%%\n",
           restMs, 100.0 * restMs / trainPerStep);
    printf("  per ply: search %5.1f%%  |  learn %5.1f%%\n",
           100.0 * searchPerMove / perPly, 100.0 * trainPerStep / perPly);
    /* 梯度累积 B 条后每样本的成本 (只有优化器那一项被摊薄) */
    const int B = 64;
    const double perSampleBatched = fwdMs + restMs + optMs / (double)B;
    printf("  per-sample cost: %.2f ms now -> %.2f ms with grad-accum B=%d (%.1fx cheaper)\n",
           trainPerStep, perSampleBatched, B, trainPerStep / perSampleBatched);
    /*
       这是"数据利用效率"最关键的一行: 复用同一批数据的边际成本 = 一次前向+反向+优化器,
       而它相对搜索很便宜。K 轮复用只让每步的总成本涨 K*trainPerStep。
    */
    printf("  reusing the same data for %d extra epochs costs +%.0f%% per-move compute\n",
           10, 100.0 * 10.0 * trainPerStep / perPly);
}

/* ================================================================
 *  测试7: 折现回报的"视角逐手翻转"
 *
 *  状态编码是规范视角 (价值头输出的是**走子方**的价值), 而相邻两步的走子方互为
 *  对手, 所以正确的递推是
 *
 *      V(s_i) = reward_i - gamma * V(s_{i+1})       (注意是减号)
 *
 *  写错了 (用加号, 或者把终局值固定按黑方视角传) **不会报任何错**, 只会让一半
 *  样本的训练目标反号 —— 这种错误只能靠"手算一个 2~3 步的小例子对数值"来钉住,
 *  所以这里直接查 discountedReturns 的返回值。
 * ================================================================ */
static bool testValueTargetSign()
{
    printf("\n========================================\n");
    printf("  PPO: discounted-return sign convention\n");
    printf("========================================\n");

    /* 只用到 discountedReturns(), 不做任何训练 —— 很快 */
    RL::PPO ppo(PPOMCTSAgent::STATE_DIM, 16, PPOMCTSAgent::ACTION_DIM);
    const float g = ppo.gamma;
    printf("  gamma = %.4f\n", g);

    RL::Tensor s(PPOMCTSAgent::STATE_DIM, 1);
    RL::Tensor a(PPOMCTSAgent::ACTION_DIM, 1);
    s.zero();
    a.zero();
    a[1] = 1.0f;

    auto makeTraj = [&](const std::vector<float> &rewards) {
        std::vector<RL::Step> t;
        for (std::size_t i = 0; i < rewards.size(); i++) {
            t.emplace_back(s, a, rewards[i]);
        }
        return t;
    };
    bool ok = true;
    auto checkNear = [&ok](const char *msg, float got, float want) {
        const float tol = 1e-4f;
        const bool good = std::fabs(got - want) <= tol;
        ok = ok && good;
        printf("  %-44s got %+9.5f  want %+9.5f  %s\n",
               msg, got, want, good ? "ok" : "**FAIL**");
    };

    {   /* [1] 2 步、无即时奖励、最后一步走子方获胜 -> R[0] 必须为负 */
        std::vector<RL::Step> t = makeTraj({ 0.0f, 0.0f });
        std::vector<float> r = ppo.discountedReturns(t, +1.0f);
        checkNear("2 steps, last mover won: R[0] (must be -)", r[0], -g * g);
        checkNear("2 steps, last mover won: R[1] (must be +)", r[1], +g);
    }
    {   /* [2] 3 步 -> 符号逐手交替 */
        std::vector<RL::Step> t = makeTraj({ 0.0f, 0.0f, 0.0f });
        std::vector<float> r = ppo.discountedReturns(t, +1.0f);
        checkNear("3 steps: R[0] (+)", r[0], +(g * g * g));
        checkNear("3 steps: R[1] (-)", r[1], -(g * g));
        checkNear("3 steps: R[2] (+)", r[2], +g);
    }
    {   /* [3] 吃了一个子 (+5) 但最终输棋 -> 仍为正, 只是被终局抵消一部分 */
        std::vector<RL::Step> t = makeTraj({ 5.0f });
        std::vector<float> r = ppo.discountedReturns(t, -1.0f);
        checkNear("1 step, +5 material but lost the game", r[0], 5.0f - g);
    }
    {   /* [4] 和棋: 终局值为 0, 回报只剩即时奖励 */
        std::vector<RL::Step> t = makeTraj({ 1.0f, 0.0f });
        std::vector<float> r = ppo.discountedReturns(t, 0.0f);
        checkNear("draw: R[0]", r[0], 1.0f);
        checkNear("draw: R[1]", r[1], 0.0f);
    }

    printf("  -> %s\n", ok ? "all sign checks passed"
                           : "**SIGN CONVENTION BUG**");
    return ok;
}

/* ================================================================
 *  测试8: 策略目标 = 根的访问分布 (而不是 one-hot)
 *
 *  这是 AlphaZero 的"改进算子": 一次 80 次模拟的搜索里, 被选中的动作往往只比其它
 *  候选多访问一两次, 只保留它的 one-hot 就等于把搜索退化成一个随机采样器。
 *  这里手工搭一棵最小树, 直接核对归一化后的分布。
 * ================================================================ */
static bool testVisitDistributionTarget()
{
    printf("\n========================================\n");
    printf("  PPO+MCTS: policy target = visit distribution\n");
    printf("========================================\n");

    Chess chess;
    PPOMCTSAgent agent(chess, 16, 0.99f, 0.001f, 1.414f);
    chess.reset();

    bool ok = true;

    /* 空树: 必须返回 false (调用方据此退回 one-hot) */
    agent.nodes.clear();
    RL::Tensor pi(PPOMCTSAgent::ACTION_DIM, 1);
    if (agent.visitDistribution(0, pi)) {
        printf("  empty tree -> returned true  **FAIL**\n");
        ok = false;
    } else {
        printf("  empty tree -> false  ok\n");
    }

    /* 最小树: 根 + 3 个孩子, 访问次数 5/3/2 -> 期望 0.5 / 0.3 / 0.2 */
    const int actionIdx[3] = { 100, 200, 300 };
    const int visits[3]    = { 5, 3, 2 };
    PPOMCTSAgent::AZNode root;
    root.currentColor = Stone::COLOR_RED;
    agent.nodes.push_back(root);
    for (int k = 0; k < 3; k++) {
        PPOMCTSAgent::AZNode c(0, actionIdx[k], ::Step(), 0.0f, Stone::COLOR_BLACK);
        c.visitCount = visits[k];
        agent.nodes.push_back(c);
        agent.nodes[0].childIDs.push_back(1 + k);
    }

    const bool got = agent.visitDistribution(0, pi);
    printf("  visitDistribution -> %s\n", got ? "true" : "false");

    float sum = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < PPOMCTSAgent::ACTION_DIM; i++) {
        sum += pi[i];
        if (pi[i] > 1e-6f) { nonzero++; }
    }
    printf("  nonzero entries = %d (one-hot would be 1)   sum = %.6f\n", nonzero, sum);

    const float want[3] = { 0.5f, 0.3f, 0.2f };
    for (int k = 0; k < 3; k++) {
        const bool good = std::fabs(pi[actionIdx[k]] - want[k]) <= 1e-4f;
        printf("  pi[%3d] = %.5f  want %.5f  %s\n",
               actionIdx[k], pi[actionIdx[k]], want[k], good ? "ok" : "**FAIL**");
        ok = ok && good;
    }

    ok = ok && got;
    ok = ok && (nonzero == 3);
    ok = ok && (std::fabs(sum - 1.0f) <= 1e-4f);
    printf("  -> %s\n", ok ? "target is a proper distribution"
                           : "**TARGET BUG**");
    return ok;
}

/* ================================================================
 *  测试9: 回放池 + 梯度累积 (P3 + P4)
 *
 *  盯三件事:
 *    [a] 稀疏策略目标 -> 稠密目标 的往返是对的 (回放池存的就是稀疏形式)
 *    [b] 批量学习真的在学 (策略在目标动作上的概率明显上升, critic 靠近目标值)
 *    [c] 成本: learnFromReplay(64, 2) 应该显著便宜于"128 次 trainStep"
 *        —— 后者正是改动前的做法 (每条样本单独跑一次全参数优化器)
 * ================================================================ */
static bool testReplayPath()
{
    printf("\n========================================\n");
    printf("  PPO: replay pool + gradient accumulation\n");
    printf("========================================\n");

    Chess chess;
    /*
       固定种子。test_ppomcts 的 main 只种了 std::rand, **没有**种 RL::Random ——
       而 learnFromReplay 的批采样走的是 RL::Random::engine。不种的话每次运行的
       样本序列都不同, 这个测试就会时好时坏 (实测: 同一份代码一次过、一次不过)。
    */
    RL::Random::setSeed(20240914);

    /* lr 取小一点: 反复打同一批样本时, 大 lr 会把 softmax 推到饱和 */
    PPOMCTSAgent agent(chess, 32, 0.99f, 0.005f, 1.414f);
    chess.reset();

    RL::Tensor s(PPOMCTSAgent::STATE_DIM, 1);
    agent.encodeState(s);

    bool ok = true;
    const int target0 = 11;
    const int target1 = 222;
    const int target2 = 3333;

    /* ---- [a] 容量裁剪: 容量 10, 推 50 条 -> 只留 10 条 ---- */
    {
        const std::size_t savedCap = agent.ppo.replayCapacity;
        agent.ppo.replayCapacity = 10;
        for (int i = 0; i < 50; i++) {
            std::vector<int> idx(1, target0);
            std::vector<float> prob(1, 1.0f);
            agent.ppo.addReplay(s, idx, prob, 0.0f);
        }
        const bool capOk = (agent.ppo.replaySize() == 10);
        printf("  FIFO cap: pushed 50 into cap=10 -> size=%d  %s\n",
               (int)agent.ppo.replaySize(), capOk ? "ok" : "**FAIL**");
        ok = ok && capOk;
        agent.ppo.replayCapacity = savedCap;
        agent.ppo.clearReplay();
    }

    /* ---- [b] 学一条确定性映射: 目标动作 target0 概率 1, 价值 0.75 ---- */
    const int nSamples = 256;
    for (int i = 0; i < nSamples; i++) {
        std::vector<int> idx;
        std::vector<float> prob;
        /* 混入一点别的动作, 让目标是个真的分布而不是纯 one-hot */
        idx.push_back(target0); prob.push_back(0.8f);
        idx.push_back(target1); prob.push_back(0.15f);
        idx.push_back(target2); prob.push_back(0.05f);
        agent.ppo.addReplay(s, idx, prob, 0.75f);
    }
    printf("  replay size = %d\n", (int)agent.ppo.replaySize());

    /* 学习前先读一次 (注意 ppo.action 返回的是网络内部张量, 取到值就得立刻存下来) */
    const float pBefore[3] = {
        agent.ppo.action(s)[target0],
        agent.ppo.action(s)[target1],
        agent.ppo.action(s)[target2]
    };
    const float vBefore = agent.ppo.value(s);

    /* 学到"目标动作概率过 0.5"就停 (带轮数上限) —— 断言收敛而不是断言固定轮数后的
       数值, 这样既钉住了"确实在学", 又不会因为超调而假失败 */
    int rounds = 0;
    bool ran = false;
    while (rounds < 40 && agent.ppo.action(s)[target0] < 0.5f) {
        const float pNow = agent.ppo.action(s)[target0];
        if (!agent.ppo.learnFromReplay(64, 2, 0.005f)) {
            break;
        }
        ran = true;
        rounds++;
        if (rounds % 5 == 0 || rounds == 1) {
            printf("    round %2d: p(target0) before=%.6f  V=%+.5f  loss=%g  actorCE=%g\n",
                   rounds, pNow, agent.ppo.value(s),
                   agent.ppo.lastLoss, agent.ppo.lastActorLoss);
        }
    }
    const float pAfter[3] = {
        agent.ppo.action(s)[target0],
        agent.ppo.action(s)[target1],
        agent.ppo.action(s)[target2]
    };
    const float vAfter = agent.ppo.value(s);

    printf("  P(target0) %.6f -> %.6f   (target prob 0.80, %d rounds)\n",
           pBefore[0], pAfter[0], rounds);
    printf("  P(target1) %.6f -> %.6f   (target prob 0.15)\n", pBefore[1], pAfter[1]);
    printf("  P(target2) %.6f -> %.6f   (target prob 0.05)\n", pBefore[2], pAfter[2]);
    printf("  V(s)       %.6f -> %.6f   (value target 0.75)\n", vBefore, vAfter);
    printf("  lastLoss = %g (batch average), lastActorLoss = %g\n",
           agent.ppo.lastLoss, agent.ppo.lastActorLoss);

    /*
       只断言**方向**, 不断言"恰好落在目标上": 软目标下交叉熵的最优点确实是 p=target,
       但 RMSProp + clipGrad 每步位移固定, 冲过头是正常的, 那不代表路径错了。
    */
    const bool learned = ran && (pAfter[0] > 0.5f) && (pAfter[0] > pBefore[0] * 10.0f);
    const bool ranked  = (pAfter[0] > pAfter[1]) && (pAfter[1] > pAfter[2]);
    const bool valued  = std::fabs(vAfter - 0.75f) < 0.5f;
    const bool finite  = std::isfinite(agent.ppo.lastLoss) &&
                         std::isfinite(agent.ppo.lastActorLoss);
    printf("  learning ok=%d (P(target0)>0.5 且至少涨 10x), rank ok=%d, value ok=%d, finite ok=%d\n",
           (int)learned, (int)ranked, (int)valued, (int)finite);
    ok = ok && learned && ranked && valued && finite;

    /* ---- [c] 成本: 批量路径 vs 逐样本路径 (改动前的做法) ---- */
    {
        RL::Tensor target(PPOMCTSAgent::ACTION_DIM, 1);
        target.zero();
        target[target0] = 1.0f;

        /*
           两侧都用**生产学习率** 0.001。用 0.02 连打 128 次同一条样本会把网络推爆
           (RMSProp 的 clipGrad 把每层梯度归一成单位长度, 128 步就是 2.56 的总位移),
           那样测出来的 loss 是测量方式的产物, 不是路径的性质。
        */
        const float lr = 0.001f;
        const int steps = 128;   /* = 64 条 x 2 遍 */
        Timer t;
        for (int i = 0; i < steps; i++) {
            agent.ppo.trainStep(s, target, 0.75f, lr);
        }
        const double perSampleMs = (double)t.elapsedMs();

        t.reset();
        agent.ppo.learnFromReplay(64, 2, lr);
        const double batchedMs = (double)t.elapsedMs();

        printf("  cost for %d sample-updates: per-sample trainStep = %.1f ms  |"
               "  learnFromReplay(64,2) = %.1f ms   -> %.2fx cheaper\n",
               steps, perSampleMs, batchedMs,
               batchedMs > 0.0 ? perSampleMs / batchedMs : 0.0);
        const bool cheaper = (batchedMs > 0.0) && (perSampleMs > batchedMs);
        ok = ok && cheaper;
        if (!cheaper) { printf("  **EXPECTED the batched path to be cheaper**\n"); }
    }

    printf("  -> %s\n", ok ? "replay path verified" : "**REPLAY PATH BUG**");
    return ok;
}

/* ================================================================
 *  测试10: 左右镜像数据增广 (P6)
 *
 *  增广成立的前提是"y -> 8-y 是这个游戏的规则对称": 翻过来的局面必须还是一个真实
 *  可达的局面, 而且同一步棋翻过去还是那一步。这条性质**不能靠肉眼看代码确认** ——
 *  最容易犯的错是拿 encodeState 里那个 x -> 9-x 的规范视角镜像当增广对称 (它会把
 *  红方的子搬到黑方半场, 翻出来的局面根本不存在)。所以这里造两个真实局面:
 *  从同一个初始局面出发, 一边走原走法序列、另一边把每一步 from/to 都做 y 镜像再走,
 *  走完两者互为镜像 —— 然后逐格比平面编码。
 *
 *  另外钉住增广的接线本身: mirrorAugment=true 时一条样本进池**两条**, 第二条的
 *  状态与动作下标都必须是第一条的镜像, 价值目标必须完全一致。
 * ================================================================ */
static bool testMirrorAugmentation()
{
    bool ok = true;
    printf("\n--- 测试10: 左右镜像数据增广 (P6) ---\n");

    using A = PPOMCTSAgent;

    /* ---- [a] 代数性质: 自反 / 是双射 / 第 4 列不动 ---- */
    {
        bool involutive = true;
        for (int c = 0; c < A::CELLS; c++) {
            if (A::mirrorCell(A::mirrorCell(c)) != c) { involutive = false; break; }
        }
        /* 中间那一列 (y=4) 必须是不动点, 否则镜像会把棋盘错位 */
        bool middleFixed = true;
        for (int x = 0; x < 10; x++) {
            if (A::mirrorCell(x * 9 + 4) != x * 9 + 4) { middleFixed = false; break; }
        }
        /* mirrorActionIdx 在 8100 个动作上必须是双射 (自反 <=> 双射, 这里直接数一遍) */
        std::vector<int> hit(A::ACTION_DIM, 0);
        bool bijective = true;
        for (int a = 0; a < A::ACTION_DIM; a++) {
            const int b = A::mirrorActionIdx(a);
            if (b < 0 || b >= A::ACTION_DIM || hit[b]) { bijective = false; break; }
            hit[b] = 1;
        }
        /* 必须与规范视角的 x 镜像**交换**: 两处坐标系不一致的话先验会串帧 */
        bool commutes = true;
        for (int color = Stone::COLOR_RED; color <= Stone::COLOR_BLACK; color++) {
            for (int x = 0; x < 10 && commutes; x++) {
                for (int y = 0; y < 9; y++) {
                    const Pos p(x, y);
                    const Pos m(x, 8 - y);
                    if (A::mirrorCell(A::canonicalCell(p, color)) !=
                        A::canonicalCell(m, color)) {
                        commutes = false;
                        break;
                    }
                }
            }
        }
        printf("  [a] mirrorCell 自反=%d, 第4列不动=%d, mirrorActionIdx 双射=%d,"
               " 与 canonicalCell 交换=%d\n",
               (int)involutive, (int)middleFixed, (int)bijective, (int)commutes);
        ok = ok && involutive && middleFixed && bijective && commutes;
    }

    /* ---- [b] 初始局面本身左右对称: 平面编码必须是不动点 ---- */
    {
        Chess env;
        PPOMCTSAgent enc(env, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, /*withGrad=*/false);
        bool invariant[2] = { true, true };
        const int colors[2] = { Stone::COLOR_RED, Stone::COLOR_BLACK };
        for (int i = 0; i < 2; i++) {
            RL::Tensor s(A::STATE_DIM, 1), m(A::STATE_DIM, 1);
            s.zero();
            enc.encodeStateFor(colors[i], s);
            A::mirrorPlanes(s, m);
            for (std::size_t k = 0; k < s.size(); k++) {
                if (std::fabs(s[k] - m[k]) > 1e-6f) { invariant[i] = false; break; }
            }
        }
        printf("  [b] 初始局面镜像不变: RED=%d, BLACK=%d\n",
               (int)invariant[0], (int)invariant[1]);
        ok = ok && invariant[0] && invariant[1];
    }

    /* ---- [c] 真实局面: 镜像走法序列走出的局面, 编码必须是原编码的镜像 ----
     *
     *  构造镜像局面的**正确**做法 (第一版这里写错过): 不能把走法 Step 的坐标翻一下
     *  就丢给另一块棋盘 —— Step 带着**棋子 id**, 而 id 是按列的左右顺序分配的
     *  (RED_CHE1 在 y=0, RED_CHE2 在 y=8), 左右翻转会把车与另一侧的车对调, 于是
     *  id 指向的棋子在镜像棋盘上根本不在那个格子上, moveForward 直接失败、两块棋盘
     *  从第一步起就不同步了。正确做法是: 在镜像局面上用**它自己的走法生成**去找
     *  "动作下标恰好等于镜像下标"的那一步。
     *
     *  这顺带把增广的立论钉得更死 —— 每走一步同时验证:
     *    (1) 原局面的**全部**合法走法的下标, 镜像后正好是镜像局面的合法下标集合;
     *    (2) 镜像后的那一步确实存在 (不是"翻了个不存在的位置");
     *    (3) 走完之后两块棋盘的平面编码仍然互为镜像。
     */
    {
        Chess a, b;
        a.reset(); a.sideToMove = Stone::COLOR_BLACK;
        b.reset(); b.sideToMove = Stone::COLOR_BLACK;

        PPOMCTSAgent ea(a, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, false);
        PPOMCTSAgent eb(b, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, false);

        int played = 0;
        bool lockstep = true;
        bool foundMirror = true;
        bool moveSetsMatch = true;
        int missingAt = -1;

        for (int i = 0; i < 12; i++) {
            /*
               先把 a 的走法下标**全部拷出来**再生成 b 的走法: 走法对象来自
               thread_local 对象池, 提前归还/再取用会让上一步的指针失效。
            */
            std::vector<int> aIdx;
            Step chosenA;
            {
                std::vector<Step *> sa;
                a.sample(a.sideToMove, sa);
                if (sa.empty()) {
                    break;
                }
                for (std::size_t j = 0; j < sa.size(); j++) {
                    aIdx.push_back(ea.stepToActionIdx(*sa[j], a.sideToMove));
                }
                chosenA = *sa[(std::size_t)(i * 7 + 3) % sa.size()];
            }

            std::vector<int> bIdx;
            Step chosenB;
            bool found = false;
            {
                std::vector<Step *> sb;
                b.sample(b.sideToMove, sb);
                for (std::size_t j = 0; j < sb.size(); j++) {
                    bIdx.push_back(eb.stepToActionIdx(*sb[j], b.sideToMove));
                }
            }

            /* (1) 合法走法集合必须一一对应 */
            std::vector<int> aMirror(aIdx);
            for (std::size_t j = 0; j < aMirror.size(); j++) {
                aMirror[j] = A::mirrorActionIdx(aMirror[j]);
            }
            std::sort(aMirror.begin(), aMirror.end());
            std::vector<int> bSorted(bIdx);
            std::sort(bSorted.begin(), bSorted.end());
            if (aMirror != bSorted) {
                moveSetsMatch = false;
                break;
            }

            /* (2) 镜像后的那一步必须真的在合法走法里 */
            {
                std::vector<Step *> sb;
                b.sample(b.sideToMove, sb);
                const int want = A::mirrorActionIdx(ea.stepToActionIdx(chosenA, a.sideToMove));
                for (std::size_t j = 0; j < sb.size(); j++) {
                    if (eb.stepToActionIdx(*sb[j], b.sideToMove) == want) {
                        chosenB = *sb[j];
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                foundMirror = false;
                missingAt = i;
                break;
            }

            double d1 = 0.0, d2 = 0.0;
            a.moveForward(&chosenA, d1);
            b.moveForward(&chosenB, d2);
            played++;
            if (a.sideToMove != b.sideToMove) {
                lockstep = false;
                break;
            }
        }

        PPOMCTSAgent encA(a, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, false);
        PPOMCTSAgent encB(b, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, false);
        RL::Tensor sa(A::STATE_DIM, 1), sb(A::STATE_DIM, 1), sbM(A::STATE_DIM, 1);
        sa.zero(); sb.zero();
        encA.encodeStateFor(a.sideToMove, sa);
        encB.encodeStateFor(b.sideToMove, sb);
        A::mirrorPlanes(sb, sbM);

        int diff = 0;
        float worst = 0.0f;
        for (std::size_t k = 0; k < sa.size(); k++) {
            const float d = std::fabs(sa[k] - sbM[k]);
            if (d > 1e-6f) { diff++; }
            if (d > worst) { worst = d; }
        }
        /* 顺带确认这条路真的不是"什么都没翻" —— 走到这里的局面自己的编码不该镜像不变 */
        RL::Tensor saM(A::STATE_DIM, 1);
        A::mirrorPlanes(sa, saM);
        int asym = 0;
        for (std::size_t k = 0; k < sa.size(); k++) {
            if (std::fabs(sa[k] - saM[k]) > 1e-6f) { asym++; }
        }

        if (!foundMirror) {
            printf("  [c] **第 %d 手在原局面的合法走法里, 但镜像局面上找不到对应的那一步**\n",
                   missingAt + 1);
        }
        printf("  [c] 镜像走法序列: 走了 %d 手, 同步=%d, 走法集合对应=%d, 镜像走法存在=%d,"
               " 平面不一致格数=%d (最大差 %.2e), 原局面自身非对称格数=%d\n",
               played, (int)lockstep, (int)moveSetsMatch, (int)foundMirror,
               diff, (double)worst, asym);
        const bool mirrorOk = lockstep && moveSetsMatch && foundMirror &&
                              (played >= 6) && (diff == 0) && (asym > 0);
        ok = ok && mirrorOk;
        if (!mirrorOk) {
            printf("  **EXPECTED 镜像局面的编码逐格等于原编码的镜像**\n");
        }
    }

    /* ---- [d] 增广接线: 一条样本进池两条, 第二条是第一条的镜像 ---- */
    {
        Chess env;
        PPOMCTSAgent agent(env, 32, 0.99f, 0.001f, 1.414f, 32, 0.1f, /*withGrad=*/false);
        /* 只收集、不学习 —— 否则 batchSize 一到就会开始训练, 池子会被搅动 */
        agent.replayBatchSize = 0;
        agent.ppo.clearReplay();

        const int a0 = 10 * 90 + 30;   /* (x=1,y=1) -> (x=3,y=3) */
        const int a1 = 12 * 90 + 40;   /* (x=1,y=3) -> (x=4,y=4) */

        env.sideToMove = Stone::COLOR_RED;
        RL::Tensor state(A::STATE_DIM, 1);
        state.zero();
        agent.encodeState(state);

        RL::Tensor target(A::ACTION_DIM, 1);
        target.zero();
        target[a0] = 0.7f;
        target[a1] = 0.3f;

        std::vector<RL::Step> traj;
        traj.emplace_back(state, target, 0.0f);

        agent.mirrorAugment = true;
        agent.commitEpisode(traj, 1.0f);
        const std::size_t withMirror = agent.ppo.replaySize();

        bool pairOk = false;
        if (withMirror == 2) {
            const RL::PPO::ReplaySample &s0 = agent.ppo.replay[0];
            const RL::PPO::ReplaySample &s1 = agent.ppo.replay[1];

            RL::Tensor expectState(A::STATE_DIM, 1);
            A::mirrorPlanes(s0.state, expectState);
            bool stateOk = true;
            for (std::size_t k = 0; k < expectState.size(); k++) {
                if (std::fabs(expectState[k] - s1.state[k]) > 1e-6f) { stateOk = false; break; }
            }
            bool idxOk = (s1.actionIdx.size() == s0.actionIdx.size()) &&
                         (s1.actionProb.size() == s0.actionProb.size());
            if (idxOk) {
                for (std::size_t i = 0; i < s0.actionIdx.size(); i++) {
                    if (s1.actionIdx[i] != A::mirrorActionIdx(s0.actionIdx[i])) { idxOk = false; break; }
                    if (std::fabs(s1.actionProb[i] - s0.actionProb[i]) > 1e-6f) { idxOk = false; break; }
                }
            }
            /* 镜像后的两个动作下标必须仍然互不相同 (双射, 概率不会叠在一起) */
            bool distinct = (s1.actionIdx.size() == 2) &&
                            (s1.actionIdx[0] != s1.actionIdx[1]);
            const bool valueOk = std::fabs(s0.valueTarget - s1.valueTarget) < 1e-9f;
            pairOk = stateOk && idxOk && distinct && valueOk;

            printf("  [d] 增广: 池子 %llu 条, 状态镜像=%d, 动作下标镜像=%d,"
                   " 下标互异=%d, 价值目标相同=%d [%d -> %d]\n",
                   (unsigned long long)withMirror, (int)stateOk, (int)idxOk,
                   (int)distinct, (int)valueOk, s0.actionIdx[0], s1.actionIdx[0]);
        } else {
            printf("  [d] **EXPECTED 池子里恰好 2 条, 实际 %llu 条**\n",
                   (unsigned long long)withMirror);
        }

        /* 对照组: 关掉增广就只该进 1 条 */
        agent.mirrorAugment = false;
        agent.ppo.clearReplay();
        agent.commitEpisode(traj, 1.0f);
        const std::size_t withoutMirror = agent.ppo.replaySize();
        printf("  [d] 关闭增广: 池子 %llu 条 (应为 1)\n",
               (unsigned long long)withoutMirror);

        ok = ok && pairOk && (withoutMirror == 1);
    }

    printf("  -> %s\n", ok ? "mirror augmentation verified" : "**MIRROR BUG**");
    return ok;
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main()
{
    /* 这些程序是分钟级的训练基准: 关掉 stdout 缓冲, 这样重定向到文件或用管道
       采集时也能实时看到进度 (默认的块缓冲会在崩溃/被 kill 时把输出全部丢掉)。 */
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("  SIMD: %s\n", RL::cpuinfo::describe().c_str());
    std::srand((unsigned int)std::time(nullptr));

    printf("========================================\n");
    printf("  \xe4\xb8\xad\xe5\x9b\xbd\xe8\xb1\xa1\xe6\xa3\x8b - PPO+MCTS Agent \xe6\xb5\x8b\xe8\xaf\x95\n");
    printf("  AlphaZero\xe9\xa3\x8e\xe6\xa0\xbc: PPO\xe6\x8f\x90\xe4\xbe\x9b\xe7\xad\x96\xe7\x95\xa5\xe4\xbc\x98\xe5\x85\x88+value\n");
    printf("  MCTS\xe6\x9b\xbf\xe4\xbb\xa3\xe9\x9a\x8f\xe6\x9c\xbarollout\n");
    printf("========================================\n");

    testInference();
    testSelfPlay();
    testVsRandom();
    testVsTraditionalMCTS();
    testSparseMoeBackbone();
    testComputeBudget();
    const bool signOk = testValueTargetSign();
    const bool targetOk = testVisitDistributionTarget();
    const bool replayOk = testReplayPath();
    const bool mirrorOk = testMirrorAugmentation();

    printf("\n========================================\n");
    printf("  PPO+MCTS Agent \xe6\xb5\x8b\xe8\xaf\x95\xe5\xae\x8c\xe6\x88\x90!\n");
    printf("========================================\n");

    /* 符号约定/目标分布这类"不会自己报错"的问题要能反映到退出码上 */
    return (signOk && targetOk && replayOk && mirrorOk) ? 0 : 1;
}
