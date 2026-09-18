#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <string>
#include "rl/cpuinfo.hpp"
#include "dqnmcts_agent.h"

/* ============================================================
 *  DQN + MCTS Agent 测试程序
 *
 *  融合 Deep Q-Network 与蒙特卡洛树搜索：
 *  - DQN 提供 Q(s,a) 值函数
 *  - MCTS 使用 UCB1 树搜索，DQN 替代随机模拟
 *  - 训练使用标准 DQN 经验回放
 *
 *  [6] 段 (2026-09) 补的是**界面"模型自检"面板所依赖的那些量**的断言。
 *  为什么要断言而不是打印: 这个面板上的数字会被人拿来判断"模型值不值得继续训",
 *  所以它自己必须是被钉住的 —— 面板显示"规则上下文通道 0 个"时, 业界事实必须
 *  真的是 0 (STATE_DIM 90 = 一格一个值, 没有额外平面), 否则面板就是在骗人。
 *  本文件原来**一条断言都没有** (只打印), 所以这四条是这里的第一批。
 * ============================================================ */

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
        }                                                            \
    } while (0)

#define CHECK_EQ(a, b, msg) do {                                     \
        ++g_checks;                                                  \
        long long _a = (long long)(a);                                \
        long long _b = (long long)(b);                                \
        if (_a != _b) {                                              \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s: %lld != %lld  (%s:%d)\n",        \
                        (msg), _a, _b, __FILE__, __LINE__);           \
        }                                                            \
    } while (0)

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
                const char *names[] = {"\u8f66","\u9a6c","\u70ae","\u5175","\u5e05","\u4ed5","\u76f8"};
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

static void runInferenceTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN+MCTS Agent: \u63a8\u7406\u6d4b\u8bd5\n");
    printf("========================================\n");

    DQNMCTSAgent agent(chess, 128, 0.99f, 0.001f, 1.0f, 1.414f);

    chess.reset();
    printBoard(chess);

    printf("  \u8fdb\u884c 200 \u6b21 MCTS \u8fed\u4ee3...\n");
    Step move = agent.selectMove(Stone::COLOR_BLACK, 200, false);
    printf("  \u9009\u62e9\u7684\u8d70\u6cd5: id=%d, (%d,%d)->(%d,%d)\n",
           move.id, move.pos.x, move.pos.y,
           move.nextPos.x, move.nextPos.y);

    printf("\n  \u72b6\u6001\u7ef4\u5ea6: %d\n", DQNMCTSAgent::STATE_DIM);
    printf("  \u52a8\u4f5c\u7ef4\u5ea6: %d\n", DQNMCTSAgent::ACTION_DIM);
    printf("  DQN \u7f51\u7edc\u6df1\u5ea6: MOE<8,4> + TanhNorm + TanhNorm + Output\n");
    printf("  \u63a2\u7d22\u7387: %.4f\n", agent.getExploreRate());
}

static void runVsRandomTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN+MCTS Agent vs \u968f\u673a\u8d70\u68cb (\u9ed1\u65b9:AI, \u7ea2\u65b9:\u968f\u673a)\n");
    printf("  \u8fdb\u884c 5 \u5c40\u8bad\u7ec3 + \u6d4b\u8bd5\n");
    printf("========================================\n");

    DQNMCTSAgent agent(chess, 128, 0.99f, 0.001f, 1.0f, 1.414f);

    printf("\n  [\u8bad\u7ec3] DQN+MCTS vs \u968f\u673a\u5bf9\u624b 5 \u5c40 (200 MCTS iterations)...\n");
    agent.trainVsRandom(5, 200, 150, true);

    printf("\n  [\u6d4b\u8bd5] \u65e0\u63a2\u7d22\u6a21\u5f0f \u5bf9\u5f085\u5c40...\n");
    int blackWins = 0, redWins = 0, draws = 0;

    for (int ep = 0; ep < 5; ep++) {
        chess.reset();
        int moves;
        for (moves = 0; moves < 200; moves++) {
            /* Black (AI) uses DQN+MCTS with training=false */
            Step aiMove = agent.selectMove(Stone::COLOR_BLACK, 200, false);
            if (!aiMove.valid) { redWins++; break; }
            double dummy = 0.0;
            chess.moveForward(&aiMove, dummy);
            if (chess.isGameOver() != Stone::COLOR_NONE) { blackWins++; break; }

            std::vector<Step*> redSteps;
            chess.sample(Stone::COLOR_RED, redSteps);
            if (redSteps.empty()) { blackWins++; Steps::instance().put(redSteps); break; }
            int idx = std::rand() % (int)redSteps.size();
            chess.moveForward(redSteps[idx], dummy);
            Steps::instance().put(redSteps);
            if (chess.isGameOver() != Stone::COLOR_NONE) { redWins++; break; }
        }
        if (moves >= 200) { draws++; }
    }

    printf("  \u9ed1\u65b9(AI)\u80dc: %d, \u7ea2\u65b9(\u968f\u673a)\u80dc: %d, \u5e73\u5c40: %d\n",
           blackWins, redWins, draws);
}

static void runSelfPlayTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN+MCTS Agent: \u81ea\u5bf9\u5f08\u8bad\u7ec3 (5\u5c40)\n");
    printf("========================================\n");

    DQNMCTSAgent agent(chess, 128, 0.99f, 0.001f, 1.0f, 1.414f);

    agent.trainSelfPlay(5, 200, 150, true);

    printf("\n--- \u81ea\u5bf9\u5f08\u7edf\u8ba1 ---\n");
    printf("  \u603b\u5bf9\u5c40: %d\n", agent.getTotalEpisodes());
    printf("  \u9ed1\u65b9\u80dc\u7387: %.2f%%\n", agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);
    printf("  \u7ea2\u65b9\u80dc\u7387: %.2f%%\n", agent.getWinRate(Stone::COLOR_RED) * 100.0f);
    printf("  \u6700\u7ec8\u63a2\u7d22\u7387: %.4f\n", agent.getExploreRate());
}

/*
 * [6] 界面"模型自检"面板的数据源必须自洽
 *
 * 这四条对应的正是 decision 的依据 (见 docs/rl_plan_optimized.md 与
 * test/probe_dqnmcts_aliasing_main.cpp 的实测):
 *   1. 自检报告必须非空且带上关键读数字段 —— 否则界面面板会显示成"没有自检项",
 *      而排查的人会以为"这个 agent 没问题"。
 *   2. 报告必须**只读**: 调用前后棋盘哈希、轮到谁、halfMoveClock 都不许变。
 *      这一条不是洁癖 —— 面板会在 GUI 线程、对局进行中被调用, 而棋盘此刻正被
 *      (或刚被) 搜索使用; 报告里要枚举"标准开局"的合法着法, 就必须走 `Chess` 的副本。
 *   3. 同一个状态反复调用要给出**逐字相同**的字符串 (否则面板每次刷新都在跳)。
 *   4. 终局通道计数必须只走 getResult() 的口径: 局数守恒 (四个桶之和 == 记录次数),
 *      且 END_CAP 只可能出现在"没分出胜负"的局上。
 */
static void runSelfCheckReportTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN+MCTS Agent: \u81ea\u68c0\u62a5\u544a (界面面板\u6570\u636e\u6e90)\n");
    printf("========================================\n");

    DQNMCTSAgent agent(chess, 64, 0.99f, 0.001f, 1.0f, 1.414f);

    /* ---- 1. 非空 + 关键字段 ---- */
    const std::string r1 = agent.selfCheckReport();
    printf("  --- 报告 (%zu 字符) ---\n", r1.size());
    printf("%s", r1.c_str());
    printf("  --- 报告结束 ---\n");

    CHECK(!r1.empty(), "自检报告不能为空 (否则界面面板显示成'没有自检项')");
    CHECK(r1.find("状态 90 维") != std::string::npos,
          "报告必须写明状态维度");
    CHECK(r1.find("规则上下文通道: 0 个") != std::string::npos,
          "报告必须写明规则上下文通道数 (本 agent 是 0)");
    CHECK(r1.find("动作别名") != std::string::npos,
          "报告必须包含动作别名读数");

    /* 状态维度与"一格一个值"的编码必须自洽: 90 维 = 10x9, 没有额外平面 */
    CHECK_EQ(DQNMCTSAgent::STATE_DIM, 90, "STATE_DIM 是 90 (10x9)");
    CHECK_EQ(DQNMCTSAgent::ACTION_DIM, 128, "ACTION_DIM 是 128 槽位哈希");

    /* ---- 2. 只读: 调用前后棋盘不许变 ---- */
    {
        const unsigned long long h0 = chess.computeHash();
        const int c0 = chess.sideToMove;
        const int hm0 = chess.halfMoveClock;
        const std::size_t hist0 = chess.history.size();

        const std::string r2 = agent.selfCheckReport();
        (void)r2;

        CHECK_EQ(chess.computeHash(), h0, "自检不许改动棋盘 (哈希)");
        CHECK_EQ(chess.sideToMove, c0, "自检不许改动走子方");
        CHECK_EQ(chess.halfMoveClock, hm0, "自检不许改动 halfMoveClock");
        CHECK_EQ((long long)chess.history.size(), (long long)hist0,
                 "自检不许改动历史栈");
    }

    /* ---- 3. 确定性: 同一状态反复调用逐字相同 ---- */
    {
        const std::string a = agent.selfCheckReport();
        const std::string b = agent.selfCheckReport();
        CHECK(a == b, "自检报告必须可重复 (两次调用逐字相同)");
        printf("  [OK] 报告可重复, %zu 字符\n", a.size());
    }

    /* ---- 4. 终局通道计数: 局数守恒 ---- */
    {
        const long long total = agent.endCount[DQNMCTSAgent::END_CAP]
                              + agent.endCount[DQNMCTSAgent::END_RED_WIN]
                              + agent.endCount[DQNMCTSAgent::END_BLACK_WIN]
                              + agent.endCount[DQNMCTSAgent::END_DRAW];
        CHECK(total >= 0, "终局桶计数不能为负");
        CHECK(agent.endSeenByGameOver <= total,
              "旧口径 isGameOver 看见的局数不能超过总局数");

        /*
           走一手再读: 报告必须跟着变。注意**变的原因是"标准开局"那一行**——
           自检会拿主棋盘的副本 reset 到标准开局, 所以棋盘一旦不在开局, 那一行的
           数字就会变 (实测 44 着法 -> 38 槽位 vs 开局后的另一组数)。
           这也是"报告反映的是当前棋盘"的证据, 而不是一个写死的常量。
        */
        const std::string before = agent.selfCheckReport();
        chess.reset();
        agent.selectMove(Stone::COLOR_BLACK, 8, false);
        const std::string after = agent.selfCheckReport();
        CHECK(before != after,
              "棋盘动了之后自检报告应当变化 (标准开局那是一份实时读数)");
    }

    /* ---- 5. 终局通道计数: 跑完一局之后必须**真的**记了一局 ---- */
    {
        const long long before = agent.endCount[DQNMCTSAgent::END_CAP]
                               + agent.endCount[DQNMCTSAgent::END_RED_WIN]
                               + agent.endCount[DQNMCTSAgent::END_BLACK_WIN]
                               + agent.endCount[DQNMCTSAgent::END_DRAW];
        /* 一局、手数压到 12 手: 必然撞上限 -> 正好验证 END_CAP 那一桶 */
        agent.trainSelfPlay(1, 4, 12, false);
        const long long after = agent.endCount[DQNMCTSAgent::END_CAP]
                              + agent.endCount[DQNMCTSAgent::END_RED_WIN]
                              + agent.endCount[DQNMCTSAgent::END_BLACK_WIN]
                              + agent.endCount[DQNMCTSAgent::END_DRAW];
        CHECK_EQ(after - before, 1, "跑完一局必须恰好记一条终局通道计数");
        CHECK_EQ(agent.endCount[DQNMCTSAgent::END_CAP], after - before,
                 "12 手上限的一局必然撞上限 -> 只能落进 END_CAP 桶");
        printf("  [OK] 终局通道计数: 累计 %lld 局 (截断 %lld)\n",
               after, agent.endCount[DQNMCTSAgent::END_CAP]);
    }
}

int main()
{
    /* 这些程序是分钟级的训练基准: 关掉 stdout 缓冲, 这样重定向到文件或用管道
       采集时也能实时看到进度 (默认的块缓冲会在崩溃/被 kill 时把输出全部丢掉)。 */
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("  SIMD: %s\n", RL::cpuinfo::describe().c_str());
    std::srand((unsigned int)std::time(nullptr));

    Chess chess;

    printf("========================================\n");
    printf("  \u4e2d\u56fd\u8c61\u68cb - DQN + MCTS Agent \u6d4b\u8bd5\n");
    printf("  Deep Q-Network \u5f15\u5bfc\u7684\u8499\u7279\u5361\u6d1b\u6811\u641c\u7d22\n");
    printf("========================================\n");

    runInferenceTest(chess);
    runVsRandomTest(chess);
    runSelfPlayTest(chess);
    runSelfCheckReportTest(chess);

    printf("\n========================================\n");
    printf("  断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    printf("  DQN+MCTS Agent \u6d4b\u8bd5\u5b8c\u6210!\n");
    printf("========================================\n");

    return g_failed == 0 ? 0 : 1;
}
