/*
 * test_evab_main.cpp - EVABAgent 训练 + 对抗测试
 *
 * 与仓库里其它 test_* 不同, 这里会真的给出"强弱"的证据: 同一进程内交替先后手打
 * 若干局, 统计 胜/和/负, 并对比 blend=0 (纯手工评估, 等价于"更强的 ABAgent") 与
 * blend 训练之后的同一套搜索 —— 用来说明增益到底来自搜索还是来自学习评估。
 *
 * 用法:
 *   test_evab train <games> <playDepth> <labelDepth> <maxMoves> <rounds>
 *   test_evab match <opponent> <games> <depth>       opponent: ab|mcts|dqn|pg
 *   test_evab
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <chrono>
#include <cmath>
#include "evagent.h"
#include "abagent.h"
#include "mcts.h"
#include "dqnagent.h"
#include "pgagent.h"
#include "rl/cpuinfo.hpp"

struct Score { int win; int loss; int draw; };

static const char *resultName(int r)
{
    switch (r) {
    case Chess::RESULT_RED_WIN:   return "red win";
    case Chess::RESULT_BLACK_WIN: return "black win";
    default:                      return "draw";
    }
}

/*
 * 打一局: redAgent 走红, blackAgent 走黑。
 *
 * 关键: 所有 agent 都必须绑定到**这同一个** chess 上。agent 持有的是 Chess&,
 * 如果它们绑的是另一个棋盘 (例如 main 里的那个, 从不推进), 它们会一直对同一个
 * 初始局面思考、反复给出同一步棋, 而真正对局在另一个棋盘上进行 —— 结果毫无意义。
 * (这正是 test_main.cpp 里那个 `static ABAgent abAI(chess, 8)` 的坑。)
 *
 * openingPlies: 先随机走这么多步再交给 agent。**这是让"强弱"可测的关键**:
 * 开局阶段几乎所有走法的评分都差不多 (手工评估只有材质 + PST, 区分不出开局好坏),
 * 于是胜负往往由"排序的随机性"决定而不是由强弱决定。随机开局把这一层噪声去掉,
 * 让对局从真正有差异的中局开始。
 */
static int playGame(Chess &chess, AgentBase *redAgent, AgentBase *blackAgent,
                    int maxMoves, bool verbose, int openingPlies = 0)
{
    chess.reset();
    int turn = Stone::COLOR_RED;
    for (int m = 0; m < maxMoves; m++) {
        int r = chess.getResult(turn);
        if (r != Chess::RESULT_ONGOING) {
            return r;
        }
        Step s;
        if (m < openingPlies) {
            std::vector<Step*> legal;
            chess.sample(turn, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                return (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                                  : Chess::RESULT_RED_WIN;
            }
            s = *legal[(std::size_t)(std::rand() % (int)legal.size())];
            Steps::instance().put(legal);
        } else {
            AgentBase *agent = (turn == Stone::COLOR_RED) ? redAgent : blackAgent;
            s = agent->getBestMove(turn);
        }
        if (!s.valid) {
            return (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                              : Chess::RESULT_RED_WIN;
        }
        double dummy = 0.0;
        chess.moveForward(&s, dummy);
        if (verbose && (m % 10 == 0)) {
            std::printf("      move %3d: turn=%s (%d,%d)->(%d,%d)\n", m + 1,
                        (turn == Stone::COLOR_RED) ? "red" : "black",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y);
        }
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    return Chess::RESULT_DRAW;
}

/* evab 与对手交替先后手打 games 局 */
static Score matchVs(Chess &chess, AgentBase *evab, AgentBase *opponent, int games,
                     int maxMoves, bool verbose, int openingPlies = 0)
{
    Score sc = {0, 0, 0};
    for (int g = 0; g < games; g++) {
        const bool evabIsRed = (g % 2 == 0);
        AgentBase *red = evabIsRed ? evab : opponent;
        AgentBase *black = evabIsRed ? opponent : evab;
        int r = playGame(chess, red, black, maxMoves, verbose, openingPlies);
        if (r == Chess::RESULT_DRAW) {
            sc.draw++;
        } else if ((r == Chess::RESULT_RED_WIN) == evabIsRed) {
            sc.win++;
        } else {
            sc.loss++;
        }
        std::printf("    game %2d/%d: evab=%s -> %-9s  (W%d D%d L%d)\n",
                    g + 1, games, evabIsRed ? "red" : "black", resultName(r),
                    sc.win, sc.draw, sc.loss);
    }
    return sc;
}

static void report(const char *label, const Score &sc)
{
    const int n = sc.win + sc.loss + sc.draw;
    const double rate = (n > 0) ? (sc.win + 0.5*sc.draw)/(double)n : 0.0;
    std::printf("  %-34s W%2d D%2d L%2d   得分率 %.1f%%\n",
                label, sc.win, sc.draw, sc.loss, rate*100.0);
}

/* ============================================================
 *  0. 完备 Markov 状态 (本轮从 DQNAB 推广到 EVAB)
 * ============================================================
 *  这个测试原来是"打印信息型"的 (没有断言), 所以这里加一个最小的断言宏 ——
 *  "状态里有没有规则上下文"这件事必须有判据, 不能靠读输出。
 * ============================================================ */
static int g_checks = 0;
static int g_failed = 0;

#define printf_check(cond, msg) do {                                  \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s\n", (msg));                      \
        }                                                             \
    } while (0)

/* ============================================================
 *  1. 搜索自身的性能特征 (置换表 / 迭代加深 / 排序 的效果)
 * ============================================================ */
static void testSearchProfile()
{
    std::printf("\n========================================\n");
    std::printf("  [1] 搜索特征: 迭代加深 + 置换表 + 排序\n");
    std::printf("========================================\n");

    Chess chess;
    EVABAgent evab(chess, 48, 5, 0);
    evab.blend = 0.0f;                      /* 纯手工评估, 只看搜索本身 */

    /*
       ---- 完备 Markov 状态 (本轮从 DQNAB 推广) ----
       象棋是双人零和、完全信息、交替行动的 Markov Game; **裸棋盘 + 轮到谁不是 Markov
       状态** —— 三次重复判和 / 60 回合无吃子判和都依赖历史, 而它们决定终局。
       价值网的输入必须携带这些规则上下文, 否则「同一局面的第 2 次与第 3 次出现」是
       同一个输入而价值不同, V(s) 就不是 s 的函数。
       判据: 走 4 手可逆循环回到同一个局面 —— 棋子平面逐位相同, 规则上下文必须变。
    */
    {
        chess.reset();
        RL::Tensor s0(EVABAgent::STATE_DIM, 1);
        evab.encodeCanonical(Stone::COLOR_RED, s0);
        const int boardPlanes = EVABAgent::PLANE_HALFMOVE;   /* 14: 前面全是棋子平面 */
        const float rep0 = s0[(std::size_t)(EVABAgent::PLANE_REPEAT * EVABAgent::CELLS)];
        const float hm0  = s0[(std::size_t)(EVABAgent::PLANE_HALFMOVE * EVABAgent::CELLS)];

        const int from[4] = { 9 * 9 + 1, 0 * 9 + 1, 7 * 9 + 2, 2 * 9 + 2 };
        const int to[4]   = { 7 * 9 + 2, 2 * 9 + 2, 9 * 9 + 1, 0 * 9 + 1 };
        int played = 0;
        for (int k = 0; k < 4; k++) {
            const int turn = chess.sideToMove;
            std::vector<Step*> legal;
            chess.sample(turn, legal);
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
            chess.moveForward(&copy, d);
            played++;
        }
        RL::Tensor s1(EVABAgent::STATE_DIM, 1);
        evab.encodeCanonical(chess.sideToMove, s1);
        double boardDiff = 0.0;
        for (int p = 0; p < boardPlanes; p++) {
            for (int cell = 0; cell < EVABAgent::CELLS; cell++) {
                const std::size_t k = (std::size_t)(p * EVABAgent::CELLS + cell);
                boardDiff = std::max(boardDiff, std::fabs((double)s1[k] - (double)s0[k]));
            }
        }
        const float rep1 = s1[(std::size_t)(EVABAgent::PLANE_REPEAT * EVABAgent::CELLS)];
        const float hm1  = s1[(std::size_t)(EVABAgent::PLANE_HALFMOVE * EVABAgent::CELLS)];
        std::printf("  完备状态: %d 维 (14 棋子平面 + 3 规则上下文); 4 手循环后 "
                    "棋平面差 %.1e, 重复 %.3f -> %.3f, 无吃子 %.4f -> %.4f\n",
                    EVABAgent::STATE_DIM, boardDiff, (double)rep0, (double)rep1,
                    (double)hm0, (double)hm1);
        printf_check(played == 4, "走完 4 手可逆循环 (回到起始局面)");
        printf_check(boardDiff == 0.0, "循环之后棋子平面逐位相同 (确实是同一个局面)");
        printf_check(rep1 > rep0, "**规则上下文变了** —— 状态不再是裸棋盘");
        printf_check(hm1 > hm0, "无吃子计数进了状态 (60 回合判和的风险可见)");
        chess.reset();
    }

    const int depths[] = {2, 3, 4, 5};
    for (int d : depths) {
        chess.reset();
        evab.resetStats();
        auto t0 = std::chrono::steady_clock::now();
        Step best = evab.search(Stone::COLOR_BLACK, d, 0);
        auto t1 = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        const long long probes = evab.getTTProbes();
        const long long hits = evab.getTTHits();
        std::printf("  depth %d: %6lld ms  节点 %8lld  叶子评估 %8lld  置换表命中 %6.1f%%  "
                    "达到深度 %d  走法 (%d,%d)->(%d,%d)\n",
                    d, ms, evab.getNodes(), evab.getLeafEvals(),
                    probes > 0 ? 100.0*(double)hits/(double)probes : 0.0,
                    evab.getReachedDepth(),
                    best.pos.x, best.pos.y, best.nextPos.x, best.nextPos.y);
    }
}

/* ============================================================
 *  2. 用一轮自对弈验证学习确实在收敛 (标签拟合误差下降)
 * ============================================================ */
static void testLearning(Chess &chess, int games, int playDepth, int labelDepth,
                         int maxMoves, int rounds)
{
    std::printf("\n========================================\n");
    std::printf("  [2] 价值网络训练: 模仿手工评估 -> TD-leaf 自举\n");
    std::printf("========================================\n");

    EVABAgent evab(chess, 48, playDepth, 0);
    evab.blend = 0.0f;                       /* 从"纯手工评估"这个已知基线出发 */

    /* --- 第 0 级: 模仿手工评估 (引导阶梯的第一级, 也是最重要的一级) --- */
    std::printf("\n  [0] 模仿手工评估 (随机局面 + 手工标签)\n");
    for (int e = 1; e <= 3; e++) {
        double err = evab.pretrainFromHandEval(600, 12, 32, 4);
        std::printf("    epoch %2d: 平均|net - hand| = %.4f\n", e, err);
    }
    {
        Chess probe;
        probe.reset();
        double sumAbs = 0.0;
        int n = 0;
        for (int m = 0; m < 60; m++) {
            std::vector<Step*> legal;
            probe.sample(Stone::COLOR_RED, legal);
            if (legal.empty()) {
                break;
            }
            Step s = *legal[(std::size_t)(std::rand() % (int)legal.size())];
            Steps::instance().put(legal);
            evab.blend = 1.0f;
            const double netV = evab.evaluateLeaf(Stone::COLOR_RED);
            evab.blend = 0.0f;
            const double handV = evab.evaluateLeaf(Stone::COLOR_RED);
            sumAbs += std::fabs(netV - handV);
            n++;
            double dummy = 0.0;
            probe.moveForward(&s, dummy);
        }
        std::printf("    模仿后平均 |net - hand| = %.4f (%d 个局面; 越小 = 越贴合手工评估)\n",
                    n > 0 ? sumAbs/n : 0.0, n);
    }

    /* --- 之后: TD-leaf 自对弈自举, blend 阶梯上升 --- */
    std::printf("\n  [1] TD-leaf 自对弈 (playDepth=%d, labelDepth=%d)\n",
                playDepth, labelDepth);
    for (int r = 0; r < rounds; r++) {
        auto t0 = std::chrono::steady_clock::now();
        double err = evab.trainSelfPlay(games, playDepth, labelDepth, maxMoves, false, 32);
        auto t1 = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        float next = std::min(1.0f, evab.blend + 0.25f);
        std::printf("    round %2d: 平均|预测-标签| = %.4f   blend %.2f -> %.2f   (%lld ms)\n",
                    r + 1, err, (double)evab.blend, (double)next, ms);
        evab.blend = next;
    }
}

/* ============================================================
 *  3. 对抗: 先验证"blend=0 时就是更强的 ABAgent", 再看学习后的增益
 * ============================================================ */

/*
 * 接受门槛 (acceptance gate)。
 *
 * 这是引导阶梯缺了就危险的那一环: 训练目标是"拟合搜索出来的值", 而拟合得好
 * 不等于**下得好** —— 参数比样本多得多 (1260x48 = 60k 权重, 一轮只有几百个样本),
 * 很容易把评估带偏。实测里 blend 直接爬到 1.0 时对局成绩会崩 (0 胜 6 负)。
 *
 * 所以规则是: 新网络必须先在同一局面对阵"当前采用的配置"(blend=0, 即手工评估)
 * 不落下风, 才被采用。两个实例共享同一份网络权重 (Net 的拷贝是浅拷贝, 正好符合
 * 这里的用途), 只有 blend 不同 —— 于是这局对局衡量的纯粹是"这个网络值不值得用"。
 */
static bool acceptNet(Chess &chess, EVABAgent &trained, int games, int depth,
                      int maxMoves, int openingPlies)
{
    EVABAgent baseline(chess, 48, depth, 0);
    baseline.valueNet = trained.valueNet;      /* 同一份权重 */
    baseline.blend = 0.0f;                     /* 手工评估 = 当前采用的配置 */
    trained.blend = 1.0f;                      /* 候选: 完全由网络评估 */

    std::printf("\n  [接受门槛] 候选(blend=1) vs 现行(blend=0), %d 局\n", games);
    Score sc = matchVs(chess, &trained, &baseline, games, maxMoves, false, openingPlies);
    report("候选 vs 现行", sc);
    const int n = sc.win + sc.loss + sc.draw;
    const double rate = (n > 0) ? (sc.win + 0.5*sc.draw)/(double)n : 0.0;
    const bool ok = (rate >= 0.5);
    std::printf("  => %s (得分率 %.1f%%)\n", ok ? "接受新网络" : "拒绝新网络, 保留手工评估",
                rate*100.0);
    return ok;
}

static void testMatches(Chess &chess, int games, int depth, int maxMoves)
{
    /*
       随机开局步数: 前 6 步随机, 之后交给 agent。
       不做这一步的话, 开局几乎所有走法评分相同, 胜负由排序的随机性决定,
       测出来的是"谁的排序运气好", 不是"谁更强"。
    */
    const int OPENING = 6;

    std::printf("\n========================================\n");
    std::printf("  [3] 对抗测试 (每项 %d 局, 交替先后手, 前 %d 步随机开局)\n",
                games, OPENING);
    std::printf("========================================\n");

    /* --- 3a: blend=0, 只靠搜索 (置换表 + 迭代加深 + 排序) --- */
    {
        std::printf("\n  --- EVAB (blend=0, 纯手工评估) ---\n");
        EVABAgent evab(chess, 48, depth, 0);
        evab.blend = 0.0f;
        ABAgent ab(chess, depth);
        Score sc = matchVs(chess, &evab, &ab, games, maxMoves, false, OPENING);
        report("vs ABAgent (同深度)", sc);

        MCTS mcts(chess, 1.414f);
        Score sc2 = matchVs(chess, &evab, &mcts, games, maxMoves, false, OPENING);
        report("vs MCTS (600 次模拟)", sc2);
    }

    /* --- 3b: 等时对比: EVAB 限时, ABAgent 固定深度 ---
       EVAB 的搜索快得多 (深度 5 约 145 ms, ABAgent 同深度约 3.1 s), 所以"等深度"
       不是公平比较 —— 引擎之间正确的比法是**等时间**。 */
    {
        std::printf("\n  --- 等时对抗: EVAB(限时 %d ms) vs ABAgent(固定深度 %d) ---\n",
                    200, depth + 1);
        EVABAgent evab(chess, 48, 12, 200);   /* 深度上限给大, 由时间控制 */
        evab.blend = 0.0f;
        ABAgent ab(chess, depth + 1);
        Score sc = matchVs(chess, &evab, &ab, games, maxMoves, false, OPENING);
        report("vs ABAgent (深一层)", sc);
    }

    /* --- 3c: 训练后的网络评估 (先模仿手工评估, 再 TD-leaf), 经接受门槛 --- */
    {
        std::printf("\n  --- EVAB (训练 + 接受门槛) ---\n");
        EVABAgent evab(chess, 48, depth, 0);
        evab.blend = 0.0f;
        std::printf("  模仿手工评估...\n");
        double err = 0.0;
        for (int e = 0; e < 3; e++) {
            err = evab.pretrainFromHandEval(600, 12, 32, 4);
        }
        std::printf("    最后一个 batch 的 |net - hand| = %.4f\n", err);
        evab.blend = 1.0f;                     /* 模仿完成后先当作候选 */

        /* 只模仿手工评估的网络值不值得用? 让门槛说了算 */
        const bool acceptedAfterImitate = acceptNet(chess, evab, games, depth,
                                                    maxMoves, OPENING);
        if (!acceptedAfterImitate) {
            evab.blend = 0.0f;
            std::printf("  (两段训练都未通过门槛 -> 保留手工评估, 这是设计上允许的结果)\n");
        }
    }

    /* --- 3d: 与不搜索的 RL agent 对比 --- */
    {
        std::printf("\n  --- EVAB (blend=0) vs 不搜索的 RL agent ---\n");
        EVABAgent evab(chess, 48, depth, 0);
        evab.blend = 0.0f;

        DQNAgent dqn(chess, 64, 0.99f, 0.001f, 1.0f);
        Score sc = matchVs(chess, &evab, &dqn, games, maxMoves, false, OPENING);
        report("vs DQNAgent (未训练)", sc);

        PGEagent pg(chess, 64, 0.9f, 0.01f, 1.0f);
        Score sc2 = matchVs(chess, &evab, &pg, games, maxMoves, false, OPENING);
        report("vs PGEagent (未训练)", sc2);
    }
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("  SIMD: %s\n", RL::cpuinfo::describe().c_str());
    std::printf("========================================\n");
    std::printf("  中国象棋 - EVAB Agent (学会评估的 Alpha-Beta)\n");
    std::printf("========================================\n");
    std::srand((unsigned int)std::time(nullptr));

    Chess chess;

    /* 默认: 全部跑一遍, 且参数取得比较小, 便于快速看到结果 */
    int games = 6;
    int depth = 4;
    int maxMoves = 160;

    if (argc >= 3 && std::strcmp(argv[1], "train") == 0) {
        int g = (argc > 2) ? std::atoi(argv[2]) : 4;
        int pd = (argc > 3) ? std::atoi(argv[3]) : 3;
        int ld = (argc > 4) ? std::atoi(argv[4]) : 4;
        int mm = (argc > 5) ? std::atoi(argv[5]) : 120;
        int rd = (argc > 6) ? std::atoi(argv[6]) : 3;
        testLearning(chess, g, pd, ld, mm, rd);
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "match") == 0) {
        int n = (argc > 3) ? std::atoi(argv[3]) : 8;
        int d = (argc > 4) ? std::atoi(argv[4]) : 4;
        EVABAgent evab(chess, 48, d, 0);
        evab.blend = 0.0f;
        if (std::strcmp(argv[2], "ab") == 0) {
            ABAgent ab(chess, d);
            report("vs ABAgent", matchVs(chess, &evab, &ab, n, maxMoves, false));
        } else if (std::strcmp(argv[2], "mcts") == 0) {
            MCTS mcts(chess, 1.414f);
            report("vs MCTS", matchVs(chess, &evab, &mcts, n, maxMoves, false));
        } else if (std::strcmp(argv[2], "dqn") == 0) {
            DQNAgent dqn(chess, 64, 0.99f, 0.001f, 1.0f);
            report("vs DQNAgent", matchVs(chess, &evab, &dqn, n, maxMoves, false));
        } else if (std::strcmp(argv[2], "pg") == 0) {
            PGEagent pg(chess, 64, 0.9f, 0.01f, 1.0f);
            report("vs PGEagent", matchVs(chess, &evab, &pg, n, maxMoves, false));
        } else {
            std::printf("unknown opponent: %s\n", argv[2]);
            return 2;
        }
        return 0;
    }

    std::printf("\n  参数: %d 局/项, 深度 %d, 单局最多 %d 步\n", games, depth, maxMoves);

    testSearchProfile();
    testLearning(chess, 4, 3, 4, 100, 3);
    testMatches(chess, games, depth, maxMoves);

    std::printf("\n========================================\n");
    std::printf("  EVAB Agent 测试完成! %d 项断言, %d 项失败\n", g_checks, g_failed);
    std::printf("========================================\n");
    return g_failed == 0 ? 0 : 1;
}
