/*
 * test_pretrain_main.cpp - "走子前先探索环境 + 预训练" 的验证
 *
 * 这套流程仿 snakeAI 的 Agent::xxxAction(): 从**当前局面**出发滚若干步收集新鲜
 * 经验、在线训练一次, 然后再决策。它最容易出错的地方是"探索把真棋局改了" ——
 * 象棋不像贪吃蛇那样能在局部坐标上模拟, 试走只能落在真棋盘上, 所以必须原样回退。
 * 这个测试就是盯住这条性质, 外加"探索后仍能给出合法走法"。
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <chrono>
#include "chess.h"
#include "rl/cpuinfo.hpp"
#include "abagent.h"
#include "mcts.h"
#include "dqnagent.h"
#include "pgagent.h"
#include "ppomcts_agent.h"
#include "dqnmcts_agent.h"
#include "evagent.h"

static int g_checks = 0;
static int g_failed = 0;
#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

/* 棋盘摘要: 棋子存活/位置 + 映射表, 用来判断"探索前后棋盘是否完全一致" */
static std::string digest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[i];
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x);
        s += ",";
        s += std::to_string(st->pos.y);
        s += ";";
    }
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 9; y++) {
            Stone *st = c.m_map[Pos(x, y)];
            s += (st == nullptr) ? "." : std::to_string(st->id);
            s += "|";
        }
    }
    s += "side=" + std::to_string(c.sideToMove);
    s += "clock=" + std::to_string(c.halfMoveClock);
    s += "hist=" + std::to_string((int)c.history.size());
    s += "hash=" + std::to_string(c.computeHash());
    return s;
}

/* 把棋盘推到一个"真实中局", 避免只在初始局面上验证 */
static void setupMidgame(Chess &c)
{
    c.reset();
    const char *moves[] = {
        "7,1-7,4", "2,1-2,4", "7,7-7,4", "2,7-2,4",
        "9,1-7,2", "0,1-2,2", "9,7-7,6", "0,7-2,6"
    };
    for (const char *mv : moves) {
        int fx, fy, tx, ty;
        if (std::sscanf(mv, "%d,%d-%d,%d", &fx, &fy, &tx, &ty) != 4) {
            continue;
        }
        Stone *s = c.m_map[Pos(fx, fy)];
        if (s == nullptr) {
            continue;
        }
        Step st;
        st.id = s->id;
        st.pos = Pos(fx, fy);
        st.nextPos = Pos(tx, ty);
        Stone *victim = c.m_map[Pos(tx, ty)];
        st.nextId = (victim != nullptr) ? victim->id : Stone::ID_NONE;
        st.reward = 0;
        st.valid = true;
        if (!c.isLegalMove(s->color, &st)) {
            continue;
        }
        double d = 0;
        c.moveForward(&st, d);
    }
}

/*
 * 核心检查: 对一个 agent 跑 exploreAndTrain, 前后棋盘必须逐位一致;
 * 之后必须还能给出一个合法走法。
 */
static void checkAgent(const char *name, AgentBase *agent, Chess &chess, int color,
                       int steps, bool expectTraining)
{
    setupMidgame(chess);
    const std::string before = digest(chess);

    auto t0 = std::chrono::steady_clock::now();
    const bool trained = agent->exploreAndTrain(color, steps);
    auto t1 = std::chrono::steady_clock::now();
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    const std::string after = digest(chess);
    const std::string info = agent->getExploreInfo();

    std::printf("  %-24s %-6s %5lld ms  %s\n", name,
                trained ? "训练" : "跳过", ms,
                info.empty() ? "-" : info.c_str());

    CHECK(before == after, "探索后棋盘/走棋方/历史/哈希必须与探索前完全一致");
    if (before != after) {
        std::printf("       before=%s\n       after =%s\n",
                    before.substr(before.size() > 90 ? before.size() - 90 : 0).c_str(),
                    after.substr(after.size() > 90 ? after.size() - 90 : 0).c_str());
    }
    if (expectTraining) {
        CHECK(trained, "该 agent 应当确实做了在线训练");
    }

    /* 探索之后仍然要能给出合法走法 */
    Step mv = agent->getBestMove(color);
    CHECK(mv.valid, "探索后应当仍能给出合法走法");
    if (mv.valid) {
        CHECK(chess.isLegalMove(color, &mv), "给出的走法必须是合法走法");
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("  SIMD: %s\n", RL::cpuinfo::describe().c_str());
    std::printf("========================================\n");
    std::printf("  走子前: 先探索环境 + 预训练, 再决策 (验证)\n");
    std::printf("========================================\n\n");
    std::srand((unsigned int)std::time(nullptr));

    Chess chess;
    const int color = Stone::COLOR_BLACK;   /* AI 走黑 */
    const int STEPS = 32;

    std::printf("  每个 agent: 探索 %d 步, 之后检查棋盘是否被改动\n\n", STEPS);

    {
        DQNAgent a(chess, 64, 0.99f, 0.001f, 1.0f);
        checkAgent("DQNAgent", &a, chess, color, STEPS, true);
    }
    {
        PGEagent a(chess, 64, 0.9f, 0.01f, 1.0f);
        checkAgent("PGEagent", &a, chess, color, STEPS, true);
    }
    {
        PPOMCTSAgent a(chess, 64, 0.99f, 0.001f, 1.414f);
        checkAgent("PPOMCTSAgent", &a, chess, color, STEPS, true);
    }
    {
        DQNMCTSAgent a(chess, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        checkAgent("DQNMCTSAgent", &a, chess, color, STEPS, true);
    }
    {
        EVABAgent a(chess, 48, 3, 0);
        a.blend = 0.0f;
        checkAgent("EVABAgent", &a, chess, color, 8, false);
    }
    {
        /* 有监督式 agent: 没有在线可训练参数, exploreAndTrain 是空实现 */
        ABAgent a(chess, 3);
        checkAgent("ABAgent (对照)", &a, chess, color, STEPS, false);
    }

    /* 探索不改棋盘这条性质, 在多步之后仍然要成立 */
    {
        std::printf("\n  连续 5 次探索后棋盘仍然不变:\n");
        setupMidgame(chess);
        const std::string before = digest(chess);
        DQNAgent a(chess, 64, 0.99f, 0.001f, 1.0f);
        for (int i = 0; i < 5; i++) {
            a.exploreAndTrain(color, 16);
        }
        CHECK(before == digest(chess), "连续探索 5 次后棋盘仍必须一致");
        std::printf("    %s\n", (before == digest(chess)) ? "OK" : "MISMATCH");
    }

    std::printf("\n========================================\n");
    std::printf("  断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    std::printf("========================================\n");
    return (g_failed == 0) ? 0 : 1;
}
