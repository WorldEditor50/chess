#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "dqnagent.h"

/* ============================================================
 *  DQN (Deep Q-Network) Agent 测试程序
 *  基于 RL::DQN 算法 (experience replay + target network)
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
    printf("  DQN Agent: \u63a8\u7406\u6d4b\u8bd5 (greedy argmax)\n");
    printf("========================================\n");

    DQNAgent agent(chess, 64, 0.99f, 0.001f, 0.5f);

    chess.reset();
    printBoard(chess);

    Step move = agent.selectMove(Stone::COLOR_BLACK, false);
    printf("  \u9009\u62e9\u7684\u8d70\u6cd5: id=%d, (%d,%d)->(%d,%d)\n",
           move.id, move.pos.x, move.pos.y,
           move.nextPos.x, move.nextPos.y);

    printf("\n  \u72b6\u6001\u7ef4\u5ea6: %d\n", DQNAgent::STATE_DIM);
    printf("  \u52a8\u4f5c\u7ef4\u5ea6: %d\n", DQNAgent::ACTION_DIM);
    printf("  \u63a2\u7d22\u7387: %.4f\n", agent.getExploreRate());
}

static void runVsRandomTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN Agent vs \u968f\u673a\u8d70\u68cb (\u9ed1\u65b9:DQN, \u7ea2\u65b9:\u968f\u673a)\n");
    printf("  \u8fdb\u884c10\u5c40\u8bad\u7ec3+\u6d4b\u8bd5\n");
    printf("========================================\n");

    DQNAgent agent(chess, 64, 0.99f, 0.001f, 1.0f);

    printf("\n  [\u8bad\u7ec3] DQN vs \u968f\u673a\u5bf9\u624b 10\u5c40...\n");
    agent.trainVsRandom(10, 150, true);

    printf("\n  [\u6d4b\u8bd5] \u65e0\u63a2\u7d22\u6a21\u5f0f \u5bf9\u5f085\u5c40...\n");
    int blackWins = 0, redWins = 0, draws = 0;

    for (int ep = 0; ep < 5; ep++) {
        chess.reset();
        int moves;
        for (moves = 0; moves < 200; moves++) {
            Step aiMove = agent.selectMove(Stone::COLOR_BLACK, false);
            if (aiMove.id == Stone::ID_NONE) { redWins++; break; }
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

    printf("  \u9ed1\u65b9(DQN)\u80dc: %d, \u7ea2\u65b9(\u968f\u673a)\u80dc: %d, \u5e73\u5c40: %d\n",
           blackWins, redWins, draws);
}

static void runSelfPlayTest(Chess &chess)
{
    printf("\n========================================\n");
    printf("  DQN Agent: \u81ea\u5bf9\u5f08\u8bad\u7ec3 (20\u5c40)\n");
    printf("========================================\n");

    DQNAgent agent(chess, 64, 0.99f, 0.001f, 1.0f);

    agent.trainSelfPlay(20, 150, true);

    printf("\n--- \u81ea\u5bf9\u5f08\u7edf\u8ba1 ---\n");
    printf("  \u603b\u5bf9\u5c40: %d\n", agent.getTotalEpisodes());
    printf("  \u9ed1\u65b9\u80dc\u7387: %.2f%%\n", agent.getWinRate(Stone::COLOR_BLACK) * 100.0f);
    printf("  \u7ea2\u65b9\u80dc\u7387: %.2f%%\n", agent.getWinRate(Stone::COLOR_RED) * 100.0f);
    printf("  \u6700\u7ec8\u63a2\u7d22\u7387: %.4f\n", agent.getExploreRate());
}

int main()
{
    std::srand((unsigned int)std::time(nullptr));

    Chess chess;

    printf("========================================\n");
    printf("  \u4e2d\u56fd\u8c61\u68cb - Deep Q-Network Agent \u6d4b\u8bd5\n");
    printf("  \u57fa\u4e8e RL::DQN (experience replay, target network)\n");
    printf("========================================\n");

    runInferenceTest(chess);
    runVsRandomTest(chess);
    runSelfPlayTest(chess);

    printf("\n========================================\n");
    printf("  DQN Agent \u6d4b\u8bd5\u5b8c\u6210!\n");
    printf("========================================\n");

    return 0;
}
