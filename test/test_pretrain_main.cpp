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
#include "aiagent.h"
#include "agentrollout.hpp"
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

/*
 * ================================================================
 *  [P1] 对手参数: "对手那一半"改由真实对手产生 (exploreAndTrain 的对手参数)
 * ================================================================
 *
 * 为什么在这里量 (而不是只在对弈里数次数): 对弈那一层只能告诉我们"问了几次",
 * 而这一节要钉的是探索内部**三条语义**, 它们都需要看见每一步的局面:
 *   (1) budget = 0 时对手一步都不问 —— 行为与改动前**逐字相同** (默认口径不变);
 *   (2) budget > 0 时对手的着法**真的被走出来了** (轨迹与不开对手时不同),
 *       而且它**不记成学习方的样本** (样本数少掉"对手手数"那么多) —— 这条是
 *       aiagent.h 里那条"不许把对手动作冒充成自己的"纪律的可测形式;
 *   (3) 对手给不出合法着法时**完整回退** (轨迹与不开对手时逐字节相同), 并计一次
 *       unmatched —— 静默回退会让"对手参数没生效"与"没开"读数一样。
 *
 * 桩: 一个只有编码/动作/奖励三个接口的假 agent (rolloutFromCurrent 只要这些)。
 * 它的 `pick` 走**第一个**合法走法 (返回 -1 = 该约定), 而"对手"走**最后一个**合法
 * 走法 —— 两者一定不同, 于是"对手的着法到底有没有被走出来"可以直接从轨迹看出来。
 */
struct RolloutStub
{
    static constexpr int STATE_DIM = 4;
    static constexpr int ACTION_DIM = 8100;

    Chess &board;
    explicit RolloutStub(Chess &b) : board(b) {}

    /* 状态编码: 桩不需要真实编码 (断言只看"记了几条/轨迹变没变"), 填零即可 */
    void encodeState(RL::Tensor &t) const { t.zero(); }

    void getLegalActions(int turn, std::vector<Step*> &legal, std::vector<int> &idx,
                         RL::Tensor &mask)
    {
        board.sample(turn, legal);
        idx.clear();
        mask.zero();
        for (std::size_t i = 0; i < legal.size(); i++) {
            idx.push_back((int)i);      /* 桩的动作索引 = 合法列表下标 */
            mask[i] = 1.0f;
        }
    }
    float computeReward(const Step &, int) { return 0.0f; }
};

struct RolloutProbe
{
    int opponentCalls = 0;      /* 对手被问了几次 */
    int opponentUsed = 0;       /* 回填: 其中几次真的进了探索 */
    int opponentUnmatched = 0;  /* 回填: 问到但用不上 (或不合法) 的次数 */
    int recorded = 0;           /* onTrans 被调用次数 = 学习方的样本数 */
    std::vector<std::string> plyDigests;   /* 每一手开始时的局面指纹 (含对手手) */
};

/* 跑一次探索: opponentBudget = 0 表示"没有对手参数" (改动前的口径) */
static RolloutProbe runRollout(Chess &chess, int color, int steps, int opponentBudget,
                               bool opponentGivesIllegal, int *collectedOut = nullptr)
{
    RolloutProbe p;
    RolloutStub stub(chess);
    OpponentPolicy opp;
    if (opponentBudget > 0) {
        opp.budget = opponentBudget;
        opp.name = "桩";
        opp.stepFor = [&p, &chess, opponentGivesIllegal](int turn) -> Step {
            p.opponentCalls++;
            std::vector<Step*> legal;
            chess.sample(turn, legal);
            Step s;
            if (opponentGivesIllegal || legal.empty()) {
                s.valid = false;                 /* "问不到": 调用方必须回退自对弈 */
            } else {
                s = *legal.back();               /* 刻意与 pick 的"第一个"不同 */
            }
            Steps::instance().put(legal);
            return s;
        };
    }
    auto pick = [&p, &chess](const RL::Tensor &, int) -> int {
        p.plyDigests.push_back(digest(chess));
        return -1;                               /* -1 = 用第一个合法走法 */
    };
    auto onTrans = [&p](const Step &, int, const RL::Tensor &, const RL::Tensor &,
                        float, bool) { p.recorded++; };
    const int collected = rolloutFromCurrent(stub, chess, color, steps, pick, onTrans, opp);
    if (collectedOut != nullptr) {
        *collectedOut = collected;
    }
    p.opponentUsed = opp.used;
    p.opponentUnmatched = opp.unmatched;
    return p;
}

static void checkOpponentParameter()
{
    std::printf("\n  [P1] 对手参数: 探索里'对手那一半'由谁产生\n");
    const int color = Stone::COLOR_BLACK;
    const int STEPS = 8;
    const int BUDGET = 2;

    Chess a, b, c;
    setupMidgame(a);
    setupMidgame(b);
    setupMidgame(c);
    const std::string beforeA = digest(a), beforeB = digest(b), beforeC = digest(c);

    /* ---- (1) 默认: 没有对手参数 (budget = 0) ---- */
    int collected0 = 0;
    const RolloutProbe p0 = runRollout(a, color, STEPS, 0, false, &collected0);
    std::printf("    无对手参数 : 样本 %d 条, 问对手 %d 次\n", p0.recorded, p0.opponentCalls);
    CHECK(p0.opponentCalls == 0, "budget=0 时对手一次都不被问 (默认口径不变)");
    CHECK(p0.recorded == STEPS, "没有对手参数时每一手都记成样本 (改动前的行为)");

    /* ---- (2) 有对手: 它的着法要'真的被走出来', 而且不记成学习方的样本 ---- */
    int collected1 = 0;
    const RolloutProbe p1 = runRollout(b, color, STEPS, BUDGET, false, &collected1);
    std::printf("    有对手参数 : 样本 %d 条, 问对手 %d 次, 用上 %d 手, 回退 %d 次\n",
                p1.recorded, p1.opponentCalls, p1.opponentUsed, p1.opponentUnmatched);
    CHECK(p1.opponentCalls == BUDGET && p1.opponentUsed == BUDGET,
          "对手被问的次数等于预算, 且每一次都用上了 (预算就是'最多问几次')");
    CHECK(p1.opponentUnmatched == 0,
          "对手的着法**全都对得上**当前合法集 (同一套规则/编码 ⇒ 不该有回退)");
    CHECK(p1.recorded == STEPS - BUDGET && collected1 == STEPS - BUDGET,
          "对手那几手**没有**记成学习方的样本 (样本数正好少掉对手手数) —— 这是"
          " '不把对手动作冒充成自己采样' 的可测形式");
    CHECK(p1.plyDigests != p0.plyDigests,
          "轨迹变了 ⇒ 对手的着法**真的被走出来了** (不是只问不用)");

    /* ---- (3) 对手给不出合法着法: 完整回退到自对弈 ---- */
    const RolloutProbe p2 = runRollout(c, color, STEPS, BUDGET, true);
    std::printf("    对手全给无效: 样本 %d 条, 问对手 %d 次, 用上 %d 手, 回退 %d 次\n",
                p2.recorded, p2.opponentCalls, p2.opponentUsed, p2.opponentUnmatched);
    CHECK(p2.opponentUsed == 0, "无效着法一次都没被用 (不能拿无效着法去落子)");
    CHECK(p2.opponentUnmatched > 0,
          "回退**被计了数** (静默回退会让'对手参数没生效'与'没开'读数一样)");
    CHECK(p2.recorded == p0.recorded && p2.plyDigests == p0.plyDigests,
          "回退之后整条探索与'没有对手参数'**逐字节相同** (降级是完整的, 不是半截)");

    /* ---- 零副作用: 三种情况下棋盘都必须原样 ---- */
    CHECK(beforeA == digest(a), "没有对手参数时, 探索后棋盘逐字段复原");
    CHECK(beforeB == digest(b), "有对手参数时, 探索后棋盘同样逐字段复原");
    CHECK(beforeC == digest(c), "对手全给无效着法时, 探索后棋盘同样逐字段复原");

    /*
       ---- (4) 对手是**真 agent** (不是桩): 搜索必须不动探索中的那个棋盘 ----
       为什么要单独一条: 界面上的对手是**真的 agent** —— 它的决策会在这个棋盘上
       moveForward/moveBack 地搜。探索也正在**同一个棋盘**上滚。所以"问对手一手"必须
       对棋盘零副作用, 否则接下来的探索从被改坏的局面上继续, 而那种污染在读数上
       完全看不出来 (数据照样被收集, 只是局面错了)。
       这里用 ABAgent 当对手 (纯搜索、确定性、便宜), 深度 2。
    */
    Chess d;
    setupMidgame(d);
    const std::string beforeD = digest(d);
    int realCalls = 0;
    {
        RolloutProbe p;
        RolloutStub stub(d);
        ABAgent realOpponent(d, 2);
        OpponentPolicy opp;
        opp.budget = BUDGET;
        opp.name = "AB(d2)";
        opp.stepFor = [&realOpponent, &realCalls](int turn) -> Step {
            realCalls++;
            return realOpponent.getBestMove(turn);
        };
        auto pick = [&p, &d](const RL::Tensor &, int) -> int {
            p.plyDigests.push_back(digest(d));
            return -1;
        };
        auto onTrans = [&p](const Step &, int, const RL::Tensor &, const RL::Tensor &,
                            float, bool) { p.recorded++; };
        rolloutFromCurrent(stub, d, color, STEPS, pick, onTrans, opp);
        std::printf("    对手=真 AB   : 样本 %d 条, 问对手 %d 次, 用上 %d 手, 回退 %d 次\n",
                    p.recorded, realCalls, opp.used, opp.unmatched);
        CHECK(realCalls == BUDGET && opp.used == BUDGET && opp.unmatched == 0,
              "真 agent 当对手时也问得到、用得上 (它的着法来自同一套规则与编码)");
        CHECK(p.recorded == STEPS - BUDGET,
              "真对手的着法同样**不记**成学习方的样本");
    }
    CHECK(beforeD == digest(d),
          "对手是**真 agent** (它会在同一个棋盘上搜索) 时, 探索后棋盘仍逐字段复原");
}

int main()
{    setvbuf(stdout, NULL, _IONBF, 0);
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

    checkOpponentParameter();

    std::printf("\n========================================\n");
    std::printf("  断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    std::printf("========================================\n");
    return (g_failed == 0) ? 0 : 1;
}
