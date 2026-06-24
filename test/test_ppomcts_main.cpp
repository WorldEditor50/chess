#include <iostream>
#include <cstdlib>
#include <ctime>
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

            if (move.id == Stone::ID_NONE) {
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
 *  主函数
 * ================================================================ */
int main()
{
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

    printf("\n========================================\n");
    printf("  PPO+MCTS Agent \xe6\xb5\x8b\xe8\xaf\x95\xe5\xae\x8c\xe6\x88\x90!\n");
    printf("========================================\n");

    return 0;
}
