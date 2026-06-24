/*
 * test_mcts_main.cpp
 * MCTS算法自对弈/性能测试
 */

#include "chess.h"
#include "mcts.h"
#include "abagent.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

static SearchStats stats;

/* ============================================================
 * MCTS AI一步搜索 (返回最优走法)
 * ============================================================ */
static Step mctsSearch(Chess &chess, int color, int iterations)
{
    Timer timer;
    MCTS mcts(chess, 1.414);
    Step best = mcts.findBestMove(color, iterations);
    long long elapsedMs = timer.elapsedMs();
    stats.totalTimeMs += elapsedMs;
    stats.totalNodesExamined += iterations;
    return best;
}

/* ============================================================
 * MCTS AI对弈一场
 * ============================================================ */
static int playOneGame(int redIter, int blackIter, bool verbose, int maxMoves)
{
    Chess chess;
    chess.reset();

    int turn = Stone::COLOR_RED;
    int movesPlayed = 0;

    while (movesPlayed < maxMoves) {
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            if (verbose) {
                printf("=== 游戏结束! 胜方: %s ===\n",
                       gameResult == Stone::COLOR_RED ? "红方" : "黑方");
            }
            return gameResult;
        }

        int iterations = (turn == Stone::COLOR_RED) ? redIter : blackIter;
        Step step = mctsSearch(chess, turn, iterations);

        // 检查是否无合法走法
        if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
            if (verbose) {
                printf("%s 无合法走法, 游戏结束\n",
                       turn == Stone::COLOR_RED ? "红方" : "黑方");
            }
            return (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }

        if (verbose) {
            printf("%s 第%d步: %s\n",
                   turn == Stone::COLOR_RED ? "红方" : "黑方",
                   movesPlayed + 1,
                   stepToString(step, chess).c_str());
        }

        // 执行走法
        double totalReward = 0;
        chess.moveForward(&step, totalReward);

        // 检查是否将杀
        if (chess.isGameOver() != Stone::COLOR_NONE) {
            if (verbose) {
                printf("=== 将杀! %s胜 ===\n",
                       turn == Stone::COLOR_RED ? "红方" : "黑方");
            }
            stats.totalMoves += movesPlayed + 1;
            return turn;
        }

        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        movesPlayed++;
    }

    if (verbose) {
        printf("达到最大步数(%d), 按评估判胜\n", maxMoves);
    }
    double score = chess.evaluate();
    stats.totalMoves += maxMoves;
    return (score > 0) ? Stone::COLOR_RED : Stone::COLOR_BLACK;
}

/* ============================================================
 * 测试1: MCTS单步搜索性能
 * ============================================================ */
static void testMCTSBenchmark()
{
    printf("\n========================================\n");
    printf("  MCTS: 单步搜索性能基准测试\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();

    int iterations[] = {100, 500, 1000, 2000};
    for (int iters : iterations) {
        Timer timer;
        Step best = mctsSearch(chess, Stone::COLOR_BLACK, iters);
        long long elapsed = timer.elapsedMs();

        printf("  迭代 %d: 最优走法 = %s, 用时 = %lld ms\n",
               iters,
               stepToString(best, chess).c_str(),
               elapsed);
    }
}

/* ============================================================
 * 测试2: MCTS自对弈 (随机开局)
 * ============================================================ */
static void testMCTSSelfPlay(int iterations, int gameCount, int maxMoves)
{
    printf("\n========================================\n");
    printf("  MCTS: AI自对弈 (迭代=%d, %d局)\n", iterations, gameCount);
    printf("========================================\n");

    int localRedWins = 0, localBlackWins = 0, localDraws = 0;
    int localTotalMoves = 0;
    long long localTimeMs = 0;

    for (int g = 0; g < gameCount; g++) {
        stats.totalNodesExamined = 0;
        stats.totalTimeMs = 0;

        int winner = playOneGame(iterations, iterations, false, maxMoves);

        if (winner == Stone::COLOR_RED) {
            localRedWins++;
            printf("  局 %d/%d: 红方胜 (本局%d步)\n", g + 1, gameCount,
                   (int)(stats.totalMoves - localTotalMoves));
        } else if (winner == Stone::COLOR_BLACK) {
            localBlackWins++;
            printf("  局 %d/%d: 黑方胜 (本局%d步)\n", g + 1, gameCount,
                   (int)(stats.totalMoves - localTotalMoves));
        } else {
            localDraws++;
            printf("  局 %d/%d: 平局\n", g + 1, gameCount);
        }
        localTotalMoves = (int)stats.totalMoves;
        localTimeMs += stats.totalTimeMs;
    }

    double avgTime = gameCount > 0 ? (double)localTimeMs / gameCount : 0;
    double avgMoves = gameCount > 0 ? (double)localTotalMoves / gameCount : 0;

    printf("\n--- 对弈统计 ---\n");
    printf("  总对局: %d\n", gameCount);
    printf("  红方胜: %d (%.1f%%)\n", localRedWins, 100.0 * localRedWins / gameCount);
    printf("  黑方胜: %d (%.1f%%)\n", localBlackWins, 100.0 * localBlackWins / gameCount);
    printf("  平局:   %d (%.1f%%)\n", localDraws, 100.0 * localDraws / gameCount);
    printf("  总用时: %lld ms\n", localTimeMs);
    printf("  平均每局: %.0f ms, 平均步数: %.0f\n", avgTime, avgMoves);
}

/* ============================================================
 * 测试3: MCTS vs Alpha-Beta 对战
 * ============================================================ */
static int playABVsMCTS(Chess &chess, int turn, int abDepth, int mctsIter, int maxMoves)
{
    int movesPlayed = 0;

    while (movesPlayed < maxMoves) {
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            return gameResult;
        }

        Step step;
        if (turn == Stone::COLOR_BLACK) {
            // 黑方: MCTS
            step = mctsSearch(chess, turn, mctsIter);
        } else {
            // 红方: Alpha-Beta
            ABAgent abAI(chess);
            Timer timer;
            step = abAI.getBestMove(turn, abDepth);
            stats.totalTimeMs += timer.elapsedMs();
            stats.totalNodesExamined++;
        }

        if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
            return (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }

        double totalReward = 0;
        chess.moveForward(&step, totalReward);

        if (chess.isGameOver() != Stone::COLOR_NONE) {
            stats.totalMoves += movesPlayed + 1;
            return turn;
        }

        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        movesPlayed++;
    }

    double score = chess.evaluate();
    stats.totalMoves += maxMoves;
    return (score > 0) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}

static void testABVsMCTS()
{
    printf("\n========================================\n");
    printf("  MCTS vs Alpha-Beta 对战\n");
    printf("  MCTS(黑方, iter=1000) vs Alpha-Beta(红方, depth=3)\n");
    printf("========================================\n");

    const int gameCount = 4;
    int mctsWins = 0, abWins = 0, draws = 0;

    for (int g = 0; g < gameCount; g++) {
        Chess chess;
        chess.reset();

        stats.totalNodesExamined = 0;
        stats.totalTimeMs = 0;

        int winner = playABVsMCTS(chess, Stone::COLOR_RED, 3, 1000, 150);

        if (winner == Stone::COLOR_BLACK) {
            mctsWins++;
            printf("  局 %d/%d: MCTS(黑方)胜\n", g + 1, gameCount);
        } else if (winner == Stone::COLOR_RED) {
            abWins++;
            printf("  局 %d/%d: Alpha-Beta(红方)胜\n", g + 1, gameCount);
        } else {
            draws++;
            printf("  局 %d/%d: 平局\n", g + 1, gameCount);
        }
    }

    printf("\n--- 对战统计 ---\n");
    printf("  MCTS(黑): %d胜, Alpha-Beta(红): %d胜, 平局: %d\n",
           mctsWins, abWins, draws);
}

/* ============================================================
 * 测试4: 指定局面MCTS分析
 * ============================================================ */
static void testMCTSSpecificPositions()
{
    printf("\n========================================\n");
    printf("  MCTS: 指定局面分析\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();

    printf("\n  局面1: 初始局面 - MCTS黑方搜索\n");
    printChessBoardPlain(chess);
    printEvaluation(chess);

    Timer timer;
    MCTS mcts(chess, 1.414);
    Step best = mcts.findBestMove(Stone::COLOR_BLACK, 2000);
    long long elapsed = timer.elapsedMs();
    printf("  MCTS(iter=2000) 最优走法: %s (用时%lldms)\n",
           stepToString(best, chess).c_str(), elapsed);
}

/* ============================================================
 * 主函数
 * ============================================================ */
int main(int argc, char *argv[])
{
    printf("========================================\n");
    printf("  中国象棋 - MCTS算法沙盒测试程序\n");
    printf("========================================\n");

    if (argc > 1) {
        if (strcmp(argv[1], "benchmark") == 0) {
            testMCTSBenchmark();
            return 0;
        }
        if (strcmp(argv[1], "play") == 0) {
            int iters = (argc > 2) ? atoi(argv[2]) : 1000;
            if (iters < 100) iters = 100;
            printf("\nMCTS自对弈 (iter=%d, verbose)...\n", iters);
            int winner = playOneGame(iters, iters, true, 300);
            printf("胜方: %s\n", winner == Stone::COLOR_RED ? "红方" : "黑方");
            return 0;
        }
    }

    // 运行所有MCTS测试
    testMCTSBenchmark();
    testMCTSSpecificPositions();
    testMCTSSelfPlay(500, 2, 150);
    testABVsMCTS();

    printf("\n========================================\n");
    printf("  MCTS 测试完成!\n");
    printf("========================================\n");
    return 0;
}
