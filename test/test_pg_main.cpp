#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "pgagent.h"

/* ============================================================
 *  策略梯度(Policy Gradient) Agent 测试程序
 *  基于 RL::DPG (REINFORCE) 算法
 * ============================================================ */

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

static void runInferenceTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  PG Agent: 策略推理测试 (argmax)\n");
    printf("========================================\n");

    PGEagent agent(chess, 64, 0.9f, 0.01f, 1.0f);

    chess.reset();
    printBoard(chess);

    /* Select a move using the current random policy */
    Step move = agent.selectMove(Stone::COLOR_BLACK, false);
    printf("  选择的走法: id=%d, (%d,%d)->(%d,%d)\n",
           move.id, move.pos.x, move.pos.y,
           move.nextPos.x, move.nextPos.y);

    printf("\n  策略网络结构: ");
    printf("  state_dim=%d, hidden_dim=%d, action_dim=%d\n",
           PGEagent::STATE_DIM, 64, PGEagent::ACTION_DIM);
}

static void runTrainingTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  PG Agent: 训练测试 (自对弈, 10局)\n");
    printf("========================================\n");

    PGEagent agent(chess, 64, 0.9f, 0.01f, 1.0f);

    agent.train(10, 100, true, true);

    printf("\n--- 训练统计 ---\n");
    printf("  总对局: %d\n", agent.getTotalEpisodes());
    printf("  黑方(AI)胜率: %.2f%%\n", agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);
    printf("  红方(AI)胜率: %.2f%%\n", agent.getWinRate(Stone::COLOR_RED) * 100.0f);
}

static void runVsRandomTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  PG Agent vs 随机走棋 (黑方:PG, 红方:随机)\n");
    printf("  进行5局测试\n");
    printf("========================================\n");

    PGEagent agent(chess, 64, 0.9f, 0.001f, 0.1f);

    /* First train a bit */
    printf("  预训练 20 局自对弈...\n");
    agent.train(20, 100, true, false);
    printf("  预训练完成, 黑方胜率=%.2f%%\n",
           agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);

    printf("\n  VS 随机对局:\n");
    int blackWins = 0;
    int redWins = 0;
    int draws = 0;
    const int numGames = 5;

    for (int ep = 0; ep < numGames; ep++) {
        chess.reset();
        int moves = 0;
        const int maxMoves = 200;

        for (moves = 0; moves < maxMoves; moves++) {
            /* Black: PG Agent */
            Step aiMove = agent.selectMove(Stone::COLOR_BLACK, false);
            if (aiMove.id == Stone::ID_NONE) {
                redWins++;  /* Black has no legal moves */
                printf("  局 %d/%d: 红方(随机)胜 (黑方无子可走)\n",
                       ep + 1, numGames);
                break;
            }
            double dummy = 0.0;
            chess.moveForward(&aiMove, dummy);

            if (chess.isGameOver() != Stone::COLOR_NONE) {
                blackWins++;
                printf("  局 %d/%d: 黑方(PG)胜, %d步\n",
                       ep + 1, numGames, moves + 1);
                break;
            }

            /* Red: Random moves */
            std::vector<Step*> redSteps;
            chess.sample(Stone::COLOR_RED, redSteps);
            if (redSteps.empty()) {
                blackWins++;  /* Red has no legal moves */
                Steps::instance().put(redSteps);
                printf("  局 %d/%d: 黑方(PG)胜 (红方无子可走)\n",
                       ep + 1, numGames);
                break;
            }

            int idx = std::rand() % (int)redSteps.size();
            chess.moveForward(redSteps[idx], dummy);
            Steps::instance().put(redSteps);

            if (chess.isGameOver() != Stone::COLOR_NONE) {
                redWins++;
                printf("  局 %d/%d: 红方(随机)胜, %d步\n",
                       ep + 1, numGames, moves + 1);
                break;
            }
        }

        if (moves >= maxMoves) {
            draws++;
            printf("  局 %d/%d: 平局\n", ep + 1, numGames);
        }
    }

    printf("\n--- VS 随机对局统计 ---\n");
    printf("  黑方(PG)胜: %d/%d (%.1f%%)\n",
           blackWins, numGames, (float)blackWins / numGames * 100.0f);
    printf("  红方(随机)胜: %d/%d (%.1f%%)\n",
           redWins, numGames, (float)redWins / numGames * 100.0f);
    printf("  平局: %d/%d (%.1f%%)\n",
           draws, numGames, (float)draws / numGames * 100.0f);
}

static void runSelfPlayTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  PG Agent: 自对弈测试 (50局, 训练)\n");
    printf("========================================\n");

    PGEagent agent(chess, 64, 0.9f, 0.01f, 1.0f);

    agent.train(50, 100, true, true);

    printf("\n--- 自对弈统计 ---\n");
    printf("  总对局: %d\n", agent.getTotalEpisodes());
    printf("  黑方胜率: %.2f%%\n", agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);
    printf("  红方胜率: %.2f%%\n", agent.getWinRate(Stone::COLOR_RED) * 100.0f);
}

int main()
{
    std::srand((unsigned int)std::time(nullptr));

    Chess chess;

    printf("========================================\n");
    printf("  中国象棋 - 策略梯度(PG) Agent 测试\n");
    printf("  基于 RL::DPG (REINFORCE) 算法\n");
    printf("========================================\n");

    runInferenceTest(chess);
    runTrainingTest(chess);
    runVsRandomTest(chess);
    runSelfPlayTest(chess);

    printf("\n========================================\n");
    printf("  PG Agent 测试完成!\n");
    printf("========================================\n");

    return 0;
}
