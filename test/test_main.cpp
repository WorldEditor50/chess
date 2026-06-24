/*
 * test_main.cpp
 * 无GUI沙盒测试程序 - 算法性能/正确性测试
 *
 * 功能:
 *   1. AI自对弈 (红黑AI对打)
 *   2. 搜索性能基准测试
 *   3. 指定局面测试
 *   4. 搜索深度对性能影响对比
 *   5. 交互模式: 手动下棋&AI辅助
 */

#include "chess.h"
#include "abagent.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

/* ============================================================
 * 全局统计
 * ============================================================ */
static SearchStats stats;

/* ============================================================
 * AI一步搜索 (返回最优走法)
 * ============================================================ */
static Step aiSearch(Chess &chess, int color, int depth)
{
    Timer timer;
    static ABAgent abAI(chess, 8);
    Step best = abAI.getBestMove(color, depth);
    long long elapsedUs = timer.elapsedUs();
    stats.totalTimeMs += timer.elapsedMs();
    stats.totalNodesExamined++;
    return best;
}

/* ============================================================
 * AI对弈一场
 * 返回: Stone::COLOR_RED (红胜), Stone::COLOR_BLACK (黑胜)
 * ============================================================ */
static int playOneGame(int redDepth, int blackDepth, bool verbose, int maxMoves)
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

        int depth = (turn == Stone::COLOR_RED) ? redDepth : blackDepth;
        Step step = aiSearch(chess, turn, depth);

        // 检查是否无合法走法
        if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
            if (verbose) {
                printf("%s 无合法走法, 游戏结束\n",
                       turn == Stone::COLOR_RED ? "红方" : "黑方");
            }
            // 无合法走法判负
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

    // 达到最大步数, 按评估判胜
    if (verbose) {
        printf("达到最大步数(%d), 按评估判胜\n", maxMoves);
    }
    double score = chess.evaluate();
    stats.totalMoves += maxMoves;
    return (score > 0) ? Stone::COLOR_RED : Stone::COLOR_BLACK;
}

/* ============================================================
 * 测试1: 基准性能测试 (测试不同搜索深度的单步搜索速度)
 * ============================================================ */
static void testSearchPerformance()
{
    printf("\n========================================\n");
    printf("  测试1: 搜索性能基准测试\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();

    int depths[] = {1, 2, 3, 4, 5};
    for (int d : depths) {
        Timer timer;
        long long beforeStats = stats.totalNodesExamined;

        Step best = aiSearch(chess, Stone::COLOR_RED, d);
        long long elapsed = timer.elapsedMs();
        long long nodesUsed = stats.totalNodesExamined - beforeStats;

        printf("  深度 %d: 最优走法 = %s, 用时 = %lld ms, 搜索节点数 = %lld\n",
               d,
               stepToString(best, chess).c_str(),
               elapsed,
               nodesUsed);

        // 重置状态
        chess.reset();
    }
}

/* ============================================================
 * 测试2: AI自对弈 (红黑相同深度对战)
 * ============================================================ */
static void testSelfPlay(int depth, int gameCount, int maxMoves)
{
    printf("\n========================================\n");
    printf("  测试2: AI自对弈 (深度=%d, %d局)\n", depth, gameCount);
    printf("========================================\n");

    int localRedWins = 0, localBlackWins = 0, localDraws = 0;
    int localTotalMoves = 0;
    long long localTimeMs = 0;

    for (int g = 0; g < gameCount; g++) {
        stats.totalNodesExamined = 0;
        stats.totalTimeMs = 0;

        int winner = playOneGame(depth, depth, false, maxMoves);

        if (winner == Stone::COLOR_RED) {
            localRedWins++;
            printf("  局 %d/%d: 红方胜 (本局%d步)\n", g + 1, gameCount, stats.totalMoves - localTotalMoves);
        } else if (winner == Stone::COLOR_BLACK) {
            localBlackWins++;
            printf("  局 %d/%d: 黑方胜 (本局%d步)\n", g + 1, gameCount, stats.totalMoves - localTotalMoves);
        } else {
            localDraws++;
            printf("  局 %d/%d: 平局\n", g + 1, gameCount);
        }
        localTotalMoves = stats.totalMoves;
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
 * 测试3: 深度对比测试 (红深黑浅, 测试搜索深度优势)
 * ============================================================ */
static void testDepthComparison()
{
    printf("\n========================================\n");
    printf("  测试3: 搜索深度对比测试\n");
    printf("========================================\n");

    const int matchCount = 4;
    const int maxMoves = 200;

    // 红深黑浅
    {
        int rw = 0, bw = 0, dr = 0;
        for (int g = 0; g < matchCount; g++) {
            int winner = playOneGame(3, 2, false, maxMoves);
            if (winner == Stone::COLOR_RED) rw++;
            else if (winner == Stone::COLOR_BLACK) bw++;
            else dr++;
        }
        printf("  场景A: 红方深度=3 vs 黑方深度=2\n");
        printf("  结果: 红方%d胜, 黑方%d胜, 平局%d\n", rw, bw, dr);
    }

    // 红浅黑深
    {
        int rw = 0, bw = 0, dr = 0;
        for (int g = 0; g < matchCount; g++) {
            int winner = playOneGame(1, 2, false, maxMoves);
            if (winner == Stone::COLOR_RED) rw++;
            else if (winner == Stone::COLOR_BLACK) bw++;
            else dr++;
        }
        printf("  场景B: 红方深度=1 vs 黑方深度=2\n");
        printf("  结果: 红方%d胜, 黑方%d胜, 平局%d\n", rw, bw, dr);
    }
}

/* ============================================================
 * 测试4: 指定局面测试
 * ============================================================ */
static void testSpecificPositions()
{
    printf("\n========================================\n");
    printf("  测试4: 指定局面分析\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();
    printf("\n  局面1: 初始局面\n");
    printChessBoardPlain(chess);
    printEvaluation(chess);

    printf("  红方(深度=3)搜索最优走法...\n");
    Step best = aiSearch(chess, Stone::COLOR_RED, 3);
    printf("  最优走法: %s\n", stepToString(best, chess).c_str());

    printf("\n  黑方(深度=3)搜索最优走法...\n");
    best = aiSearch(chess, Stone::COLOR_BLACK, 3);
    printf("  最优走法: %s\n", stepToString(best, chess).c_str());

    // 残局测试: 红方单车叫将
    printf("\n  局面2: 残局测试 - 红方单车vs黑方将\n");
    chess.reset();
    // 隐藏大部分棋子, 只保留关键棋子
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s) s->alive = false;
    }
    chess.redJiang.alive = true;
    chess.blackJiang.alive = true;
    chess.redChe1.alive = true;

    // 手动摆位置
    chess.redJiang.pos = Pos(9, 4);
    chess.blackJiang.pos = Pos(0, 4);
    chess.redChe1.pos = Pos(3, 4);

    // 更新map
    chess.m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s && s->alive) {
            chess.m_map[s->pos] = s;
        }
    }

    printChessBoardPlain(chess);
    printEvaluation(chess);

    printf("  红方(深度=4)搜索最优走法...\n");
    best = aiSearch(chess, Stone::COLOR_RED, 4);
    printf("  最优走法: %s\n", stepToString(best, chess).c_str());
}

/* ============================================================
 * 测试5: 走法数量统计测试
 * ============================================================ */
static void testMoveCount()
{
    printf("\n========================================\n");
    printf("  测试5: 初始局面走法数量统计\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();

    std::vector<Step*> redSteps;
    chess.sample(Stone::COLOR_RED, redSteps);
    printf("  红方初始走法数: %zu\n", redSteps.size());
    Steps::instance().put(redSteps);

    std::vector<Step*> blackSteps;
    chess.sample(Stone::COLOR_BLACK, blackSteps);
    printf("  黑方初始走法数: %zu\n", blackSteps.size());
    Steps::instance().put(blackSteps);
}

/* ============================================================
 * 交互模式: 手动输入走法
 * ============================================================ */
static void interactiveMode()
{
    printf("\n========================================\n");
    printf("  交互模式\n");
    printf("  输入: x1 y1 x2 y2 (例如: 9 4 8 4 移动帅)\n");
    printf("  输入: auto <depth> (AI自动走一步)\n");
    printf("  输入: eval (显示局面评估)\n");
    printf("  输入: reset (重置棋盘)\n");
    printf("  输入: board (显示棋盘)\n");
    printf("  输入: quit (退出)\n");
    printf("========================================\n");

    Chess chess;
    chess.reset();

    int turn = Stone::COLOR_RED;
    char cmd[256];

    while (true) {
        printChessBoardPlain(chess);
        printf("\n%s > ", turn == Stone::COLOR_RED ? "红方" : "黑方");
        if (fgets(cmd, sizeof(cmd), stdin) == nullptr) break;

        // 去除换行符
        size_t len = strlen(cmd);
        if (len > 0 && cmd[len-1] == '\n') cmd[len-1] = '\0';

        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "q") == 0) {
            break;
        }
        if (strcmp(cmd, "reset") == 0) {
            chess.reset();
            turn = Stone::COLOR_RED;
            printf("棋盘已重置\n");
            continue;
        }
        if (strcmp(cmd, "board") == 0 || strcmp(cmd, "b") == 0) {
            continue;
        }
        if (strcmp(cmd, "eval") == 0 || strcmp(cmd, "e") == 0) {
            printEvaluation(chess);
            continue;
        }
        if (strncmp(cmd, "auto", 4) == 0) {
            int depth = 3;
            if (strlen(cmd) > 5) {
                depth = atoi(cmd + 5);
                if (depth < 1) depth = 1;
                if (depth > 6) depth = 6;
            }
            printf("AI(%s) 搜索深度=%d ...\n",
                   turn == Stone::COLOR_RED ? "红方" : "黑方", depth);
            Timer timer;
            Step step = aiSearch(chess, turn, depth);

            if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
                printf("无合法走法, 游戏结束!\n");
                break;
            }

            double reward = 0;
            chess.moveForward(&step, reward);
            printf("AI走法: %s (用时%lldms)\n",
                   stepToString(step, chess).c_str(), timer.elapsedMs());
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            if (chess.isGameOver() != Stone::COLOR_NONE) {
                printChessBoardPlain(chess);
                printf("游戏结束! %s胜!\n",
                       chess.isGameOver() == Stone::COLOR_RED ? "红方" : "黑方");
                break;
            }
            continue;
        }

        // 尝试解析走法
        int x1, y1, x2, y2;
        if (sscanf(cmd, "%d %d %d %d", &x1, &y1, &x2, &y2) == 4) {
            // 在当前位置找对应棋子
            Stone *selected = nullptr;
            for (int i = 0; i < 32; i++) {
                Stone *s = chess.stones[i];
                if (s && s->alive && s->pos.x == x1 && s->pos.y == y1 && s->color == turn) {
                    selected = s;
                    break;
                }
            }
            if (selected == nullptr) {
                printf("错误: 位置(%d,%d)没有%s棋子\n", x1, y1,
                       turn == Stone::COLOR_RED ? "红方" : "黑方");
                continue;
            }

            // 生成走法并匹配
            std::vector<Step*> steps;
            selected->getPossibleSteps(steps);

            bool found = false;
            for (Step *s : steps) {
                if (s->nextPos.x == x2 && s->nextPos.y == y2) {
                    double reward = 0;
                    chess.moveForward(s, reward);
                    printf("走法: %s\n", stepToString(*s, chess).c_str());
                    found = true;

                    turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                    if (chess.isGameOver() != Stone::COLOR_NONE) {
                        printChessBoardPlain(chess);
                        printf("游戏结束! %s胜!\n",
                               chess.isGameOver() == Stone::COLOR_RED ? "红方" : "黑方");
                        Steps::instance().put(steps);
                        printf("程序退出\n");
                        return;
                    }
                    break;
                }
            }
            Steps::instance().put(steps);

            if (!found) {
                printf("错误: (%d,%d) 不能走到 (%d,%d)\n", x1, y1, x2, y2);
            }
        } else {
            printf("未知命令: %s\n", cmd);
            printf("可用命令: x1 y1 x2 y2, auto [depth], eval, reset, board, quit\n");
        }
    }
    printf("程序退出\n");
}

/* ============================================================
 * 主函数
 * ============================================================ */
int main(int argc, char *argv[])
{
    printf("========================================\n");
    printf("  中国象棋 - AI算法沙盒测试程序\n");
    printf("  Alpha-Beta Pruning with MVV-LVA + PST\n");
    printf("========================================\n");

    if (argc > 1) {
        if (strcmp(argv[1], "interactive") == 0) {
            interactiveMode();
            return 0;
        }
        if (strcmp(argv[1], "play") == 0) {
            int depth = (argc > 2) ? atoi(argv[2]) : 3;
            if (depth < 1) depth = 1;
            if (depth > 6) depth = 6;
            printf("\n自动对弈 (verbose) 深度=%d...\n", depth);
            int winner = playOneGame(depth, depth, true, 300);
            printf("胜方: %s\n", winner == Stone::COLOR_RED ? "红方" : "黑方");
            return 0;
        }
        if (strcmp(argv[1], "benchmark") == 0) {
            testMoveCount();
            testSearchPerformance();
            return 0;
        }
    }

    // 默认运行全部测试
    testMoveCount();
    testSearchPerformance();
    testSpecificPositions();
    testSelfPlay(2, 4, 150);
    testDepthComparison();

    printf("\n========================================\n");
    printf("  所有测试完成!\n");
    printf("========================================\n");
    return 0;
}
