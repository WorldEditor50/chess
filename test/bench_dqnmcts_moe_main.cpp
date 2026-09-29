/*
 * ============================================================
 *  bench_dqnmcts_moe - DQN+MCTS (稀疏 MoE + TB 专家) 的实测工具
 * ============================================================
 *
 *  两种模式 (都不是断言, 只出读数):
 *
 *   --mode=agree  **着法一致性**: 在一组固定局面上比较两个"臂"各选哪一手。
 *                 为什么先看这个: 它**确定性**、秒级、而且直接回答"这个旋钮到底
 *                 改不改行为"。一局棋的比分在 20 局以内噪声有 ±10~25 个点
 *                 (SAC 那边实测过), 拿它去判"旋钮值不值"基本是在读噪声。
 *                 一致率 ~100% = 这个旋钮实际上什么都没改 (死旋钮), 那时再谈
 *                 "棋力差别"就是自欺。
 *
 *   --mode=arena  **对局比分**: 与 MCTS / Alpha-Beta 下 N 局 (同一套开局、轮流先手),
 *                 报比分 + Wilson 95% 区间 + 每步耗时。区间宽就是在提醒"这个数
 *                 还不能下结论" —— 本工程的口径是"棋力只有带置信区间的锚点对局能回答"。
 *
 *  用法:
 *    bench_dqnmcts_moe.exe [--mode=agree|arena] [--positions=32] [--games=12]
 *                          [--sims=40] [--plies=100] [--opening=4]
 *                          [--opp=mcts|ab1|ab4] [--opp-sims=400]
 *                          [--twin=0|1] [--sparse=0|1] [--dense=0|1]
 *                          [--clamp=2] [--huber=1] [--dbl=0|1]
 *                          [--tau=1] [--tauiter=64] [--epochs=1] [--every=8]
 *                          [--train=0|1] [--load=PREFIX] [--srand=N] [--quiet]
 *
 *  `--train=1` 会在对局里开学习 (learnFromSearch) —— 那时**两臂的权重会各自漂移**,
 *  比分不再可比 (评估对局必须冻结)。默认关 (纯评估)。
 *
 *  ⚠ 两个坑 (本工程记录过的, 这里都避开了):
 *    1. `MCTS` 的构造函数里会 `std::srand(time(nullptr))`, 所以**构造完对手之后**
 *       必须重新钉一次随机流 (RL::Random::setSeed) —— 否则两条臂的"同一套开局"
 *       其实是两副牌;
 *    2. 对手的着法要 `isLegalMove` 复核, 任何一方返回无效走法都必须记成 aborted
 *       而不是"照下"。
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "abagent.h"
#include "chess.h"
#include "dqnmctsmoetbagent.h"
#include "mcts.h"
#include "rl/util.hpp"
#include "stone.h"

namespace {

struct Cfg {
    std::string mode = "agree";
    int positions = 32;
    int games = 12;
    int sims = 40;
    int sims2 = 0;      /* 第二个臂的模拟次数 (0 = 与 --sims 相同) */
    int maxPlies = 100;
    int openingPlies = 4;
    std::string opp = "mcts";
    int oppSims = 400;
    int oppDepth = 1;
    int twin = 0;
    int sparse = 1;
    int dense = 0;
    int clampI = 2;
    double huber = 1.0;
    int dbl = 1;
    double tau = 1.0;
    int tauIter = 64;
    int epochs = 1;
    int every = 8;
    int train = 0;
    std::string loadPrefix;
    unsigned seed = 20240901u;
    bool quiet = false;
};

Cfg g;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* ---- 参数解析: 拼错的键**必须**当场退出 (静默回落到默认值会造出假结论) ---- */
static bool argEq(const char *a, const char *name)
{
    return std::strcmp(a, name) == 0;
}

static bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = nullptr;
        std::string key;
        if (a[0] == '-' && a[1] == '-') {
            const char *eq = std::strchr(a, '=');
            if (eq != nullptr) {
                key.assign(a, (std::size_t)(eq - a));
                v = eq + 1;
            } else {
                key.assign(a);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    v = argv[++i];
                }
            }
        } else {
            continue;
        }
        const std::string k = key;
        if (k == "--mode" && v) { g.mode = v; }
        else if (k == "--positions" && v) { g.positions = std::atoi(v); }
        else if (k == "--games" && v) { g.games = std::atoi(v); }
        else if (k == "--sims" && v) { g.sims = std::atoi(v); }
        else if (k == "--sims2" && v) { g.sims2 = std::atoi(v); }
        else if (k == "--plies" && v) { g.maxPlies = std::atoi(v); }
        else if (k == "--opening" && v) { g.openingPlies = std::atoi(v); }
        else if (k == "--opp" && v) { g.opp = v; }
        else if (k == "--opp-sims" && v) { g.oppSims = std::atoi(v); }
        else if (k == "--opp-depth" && v) { g.oppDepth = std::atoi(v); }
        else if (k == "--twin" && v) { g.twin = std::atoi(v); }
        else if (k == "--sparse" && v) { g.sparse = std::atoi(v); }
        else if (k == "--dense" && v) { g.dense = std::atoi(v); }
        else if (k == "--clamp" && v) { g.clampI = std::atoi(v); }
        else if (k == "--huber" && v) { g.huber = std::atof(v); }
        else if (k == "--dbl" && v) { g.dbl = std::atoi(v); }
        else if (k == "--tau" && v) { g.tau = std::atof(v); }
        else if (k == "--tauiter" && v) { g.tauIter = std::atoi(v); }
        else if (k == "--epochs" && v) { g.epochs = std::atoi(v); }
        else if (k == "--every" && v) { g.every = std::atoi(v); }
        else if (k == "--train" && v) { g.train = std::atoi(v); }
        else if (k == "--load" && v) { g.loadPrefix = v; }
        else if (k == "--srand" && v) { g.seed = (unsigned)std::strtoul(v, nullptr, 10); }
        else if (k == "--quiet") { g.quiet = true; }
        else {
            std::printf("[bench] 未知参数: %s (拼错的键必须当场失败, 不许静默回落)\n", a);
            return false;
        }
    }
    return true;
}

/* ---- 一个"臂": 一套完整的旋钮组合 ---- */
struct Arm {
    std::string name;
    bool twin = false;
    bool sparse = true;
    bool dense = false;
    int clampI = 2;
    double huber = 1.0;
    bool dbl = true;
    double tau = 1.0;
    int tauIter = 64;
    int sims = 0;      /* 0 = 用命令行的 --sims */
};

static void applyArm(DQNMCTSMOETbAgent &ag, const Arm &a)
{
    ag.twinCritic = a.twin;
    ag.sparseLeafEval = a.sparse;
    ag.clampTarget = (float)a.clampI;
    ag.huberDelta = (float)a.huber;
    ag.doubleDQN = a.dbl;
    ag.targetTau = (float)a.tau;
    ag.replaceTargetIter = a.tauIter;
}

static Arm armFromCfg(const std::string &name)
{
    Arm a;
    a.name = name;
    a.twin = (g.twin != 0);
    a.sparse = (g.sparse != 0);
    a.dense = (g.dense != 0);
    a.clampI = g.clampI;
    a.huber = g.huber;
    a.dbl = (g.dbl != 0);
    a.tau = g.tau;
    a.tauIter = g.tauIter;
    return a;
}

static std::string digest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[(std::size_t)i];
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x) + "," + std::to_string(st->pos.y) + ";";
    }
    s += "side=" + std::to_string(c.sideToMove);
    s += "hash=" + std::to_string(c.computeHash());
    return s;
}

/* 用固定种子铺一组"随机开局", 两臂与两局用的是**同一套** */
static void randomOpening(Chess &c, int &turn, int plies)
{
    for (int i = 0; i < plies; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) {
            return;
        }
        std::vector<Step*> legal;
        c.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            return;
        }
        std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
        const Step chosen = *legal[(std::size_t)pick(RL::Random::engine)];
        Steps::instance().put(legal);
        double dummy = 0.0;
        Step mv = chosen;
        c.moveForward(&mv, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
}

/* 一个局面下的着法 (只用搜索, 不学习) */
static Step moveAt(DQNMCTSMOETbAgent &ag, Chess &c, int turn, int sims,
                   double &msOut)
{
    const bool savedLfs = ag.learnFromSearch;
    ag.learnFromSearch = false;
    const double t0 = nowMs();
    const Step s = ag.selectMove(turn, sims, 0.0f);
    msOut = nowMs() - t0;
    ag.learnFromSearch = savedLfs;
    return s;
}

/* ============================================================
 *  mode = agree: 固定局面上两臂的着法一致性
 * ============================================================ */
static int runAgree()
{
    /*
       ⚠ 必须在**建 agent 之前**钉种子: 层构造时每层都从 `RL::Random` 抽初始化权重,
       而 RL::Random 的默认种子是**按线程身份派生**的 —— 不钉它, 两次运行的初始权重
       就不同, 同一个 A/B 会在两次运行里给出完全相反的结果 (实测: 同一条命令
       "--sims=120 --positions=16" 一次 0% 一致、一次 100% 一致, 差别全在初始权重上)。
       两臂之间的可比性靠"a1 存盘 / a2 载入"保证 (同一个初始权重), 跨运行的复现靠这一行。
    */
    RL::Random::setSeed(g.seed);

    Chess c;
    c.reset();
    DQNMCTSMOETbAgent a1(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, g.dense != 0);
    DQNMCTSMOETbAgent a2(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, g.dense != 0);
    /* 第二臂用同一套初始权重: 差别只来自旋钮 */
    {
        const std::string p = "weights/_bench_dqnmcts_moe_same";
        if (!a1.saveModel(p)) {
            std::printf("[bench] 无法写出临时权重 (weights/ 可写?)\n");
            return 2;
        }
        a2.loadModel(p);
        std::remove((p + "_trunk").c_str());
        std::remove((p + "_q").c_str());
        std::remove((p + "_q2").c_str());
    }

    const Arm base = armFromCfg("base");
    Arm alt = base;
    /* 交替臂: 把命令行给的旋钮**取反**, 于是同一个可执行文件就能跑 A/B */
    alt.name = "alt";
    alt.twin = !base.twin;
    alt.sparse = !base.sparse;
    alt.dbl = !base.dbl;
    alt.clampI = (base.clampI > 0) ? 0 : 2;
    alt.sims = g.sims2;
    applyArm(a1, base);
    applyArm(a2, alt);

    const int sims1 = (base.sims > 0) ? base.sims : g.sims;
    const int sims2 = (alt.sims > 0) ? alt.sims : g.sims;
    std::printf("[bench] 着法一致性: %s vs %s | 局面 %d 个 | 模拟 %d vs %d\n",
                base.name.c_str(), alt.name.c_str(), g.positions, sims1, sims2);
    std::printf("        基准旋钮: twin=%d sparse=%d doubleDQN=%d clamp=%d dense=%d\n",
                (int)base.twin, (int)base.sparse, (int)base.dbl, base.clampI,
                (int)base.dense);

    RL::Random::setSeed(g.seed);   /* 局面生成与两臂用的是同一条流 */
    int agree = 0, total = 0, invalid = 0;
    double ms1 = 0.0, ms2 = 0.0;
    Chess board;
    board.reset();
    for (int i = 0; i < g.positions; i++) {
        /* 每个局面: 从标准开局铺 openingPlies 手随机着法 (可复现) */
        board.reset();
        int turn = Stone::COLOR_RED;
        randomOpening(board, turn, g.openingPlies);
        if (board.getResult(turn) != Chess::RESULT_ONGOING) {
            continue;
        }
        const std::string before = digest(board);
        double t1 = 0.0, t2 = 0.0;
        const Step s1 = moveAt(a1, board, turn, sims1, t1);
        if (digest(board) != before) {
            std::printf("[bench] **搜索改动了棋盘** —— 这是硬缺陷, 立即停止\n");
            return 3;
        }
        const Step s2 = moveAt(a2, board, turn, sims2, t2);
        if (digest(board) != before) {
            std::printf("[bench] **搜索改动了棋盘** —— 这是硬缺陷, 立即停止\n");
            return 3;
        }
        ms1 += t1;
        ms2 += t2;
        total++;
        if (!s1.valid || !s2.valid) {
            invalid++;
            continue;
        }
        if (s1.id == s2.id && s1.nextPos.x == s2.nextPos.x
            && s1.nextPos.y == s2.nextPos.y) {
            agree++;
        }
    }
    if (total == 0) {
        std::printf("[bench] 一个有效局面都没有\n");
        return 4;
    }
    std::printf("[bench] 一致 %d/%d = %.1f%% | 无效着法 %d | 每步 %.1f ms vs %.1f ms\n",
                agree, total, 100.0 * (double)agree / (double)total, invalid,
                ms1 / (double)total, ms2 / (double)total);
    std::printf("[bench] 判读: 一致率 ~100%% ⇒ 这些旋钮在当前权重下**改不了行为**;\n");
    std::printf("        明显偏离 100%% ⇒ 它真的在改搜索/自举的口径 (但棋力仍需对局判定)。\n");
    return (invalid > 0) ? 1 : 0;
}

/* ============================================================
 *  mode = arena: 对局比分
 * ============================================================ */
struct GameStat {
    int winner = Chess::RESULT_ONGOING;
    int plies = 0;
    double ms = 0.0;
    int moves = 0;
    bool aborted = false;
    bool illegal = false;
    const char *reason = "";
};

static GameStat playGame(Chess &c, DQNMCTSMOETbAgent &ag, ABAgent *ab, MCTS *mcts,
                         bool aiIsRed)
{
    GameStat gs;
    c.reset();
    int turn = Stone::COLOR_RED;
    if (g.openingPlies > 0) {
        randomOpening(c, turn, g.openingPlies);
    }
    while (gs.plies < g.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            gs.winner = res;
            gs.reason = (res == Chess::RESULT_DRAW) ? "draw(rule)" : "mate";
            break;
        }
        const bool aiTurn = ((turn == Stone::COLOR_RED) == aiIsRed);
        const std::string before = digest(c);
        const double t0 = nowMs();
        Step s;
        if (aiTurn) {
            double ms = 0.0;
            const bool saved = ag.learnFromSearch;
            ag.learnFromSearch = (g.train != 0);
            s = ag.selectMove(turn, g.sims, 0.0f);
            ag.learnFromSearch = saved;
            ms = nowMs() - t0;
            gs.ms += ms;
            gs.moves++;
        } else if (mcts != nullptr) {
            s = mcts->getBestMove(turn);
        } else {
            s = ab->getBestMove(turn);
        }
        if (digest(c) != before) {
            gs.aborted = true;
            gs.reason = "search modified the board";
            break;
        }
        if (!s.valid) {
            gs.aborted = true;
            gs.reason = aiTurn ? "AI returned an invalid move" : "opponent invalid move";
            break;
        }
        if (!c.isLegalMove(turn, &s)) {
            gs.illegal = true;
            gs.reason = aiTurn ? "AI played an ILLEGAL move" : "opponent played ILLEGAL";
            break;
        }
        double dummy = 0.0;
        Step mv = s;
        c.moveForward(&mv, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        gs.plies++;
    }
    if (gs.winner == Chess::RESULT_ONGOING && !gs.aborted && !gs.illegal) {
        gs.winner = Chess::RESULT_DRAW;
        gs.reason = "ply limit";
    }
    return gs;
}

/* Wilson 95% 区间 (小样本下比正态近似诚实得多) */
static void wilson(int wins, int n, double &lo, double &hi)
{
    lo = hi = 0.0;
    if (n <= 0) {
        return;
    }
    const double p = (double)wins / (double)n;
    const double z = 1.959964;
    const double d = 1.0 + z * z / (double)n;
    const double c = p + z * z / (2.0 * (double)n);
    const double s = z * std::sqrt(p * (1.0 - p) / (double)n + z * z / (4.0 * (double)n * (double)n));
    lo = (c - s) / d;
    hi = (c + s) / d;
}

static int runArena()
{
    Chess c;
    c.reset();
    DQNMCTSMOETbAgent ag(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, g.dense != 0);
    applyArm(ag, armFromCfg("arm"));
    ag.learnEveryMoves = g.every;
    ag.replayEpochs = g.epochs;
    if (!g.loadPrefix.empty()) {
        if (!ag.loadModel(g.loadPrefix)) {
            std::printf("[bench] 载入权重失败: %s\n", g.loadPrefix.c_str());
            return 2;
        }
    }
    if (g.train != 0) {
        ag.learnFromSearch = true;
    }

    /* 对手: 构造完再钉随机流 (MCTS 的构造函数里会 srand(time)) */
    ABAgent *ab = nullptr;
    MCTS *mcts = nullptr;
    if (g.opp == "mcts") {
        mcts = new MCTS(c, (double)g.oppSims);
    } else if (g.opp.rfind("ab", 0) == 0) {
        ab = new ABAgent(c, g.oppDepth);
    } else {
        std::printf("[bench] 未知对手: %s\n", g.opp.c_str());
        return 2;
    }
    RL::Random::setSeed(g.seed);

    std::printf("[bench] 对局: %d 局 | 每次 %d 次模拟 | 对手 %s | 开局 %d 手 | 上限 %d 手 |"
                " twin=%d sparse=%d doubleDQN=%d clamp=%d dense=%d train=%d\n",
                g.games, g.sims, g.opp.c_str(), g.openingPlies, g.maxPlies,
                (int)ag.twinCritic, (int)ag.sparseLeafEval, (int)ag.doubleDQN,
                g.clampI, (int)g.dense, g.train);

    int wins = 0, losses = 0, draws = 0, aborted = 0, illegal = 0;
    double msSum = 0.0;
    long long moves = 0;
    for (int i = 0; i < g.games; i++) {
        const bool aiIsRed = (i % 2 == 0);
        const GameStat gs = playGame(c, ag, ab, mcts, aiIsRed);
        if (gs.aborted || gs.illegal) {
            aborted++;
            std::printf("  局 %3d: **%s** (%s)\n", i + 1,
                        gs.illegal ? "非法着法" : "中止", gs.reason);
            continue;
        }
        msSum += gs.ms;
        moves += gs.moves;
        if (gs.winner == Chess::RESULT_DRAW) {
            draws++;
        } else {
            const bool redWon = (gs.winner == Chess::RESULT_RED_WIN);
            if (redWon == aiIsRed) {
                wins++;
            } else {
                losses++;
            }
        }
        if (!g.quiet) {
            const bool redWon = (gs.winner == Chess::RESULT_RED_WIN);
            std::printf("  局 %3d: %s (AI 执%s) %3d 手 %6.1f ms/步 <%s> [池 %d 学 %d]\n",
                        i + 1,
                        (gs.winner == Chess::RESULT_DRAW) ? "和"
                                                          : (redWon == aiIsRed ? "AI 胜"
                                                                               : "AI 负"),
                        aiIsRed ? "红" : "黑", gs.plies,
                        gs.moves > 0 ? gs.ms / (double)gs.moves : 0.0, gs.reason,
                        ag.samplePoolSize(), ag.getLearnSteps());
        }
    }

    const int decided = wins + losses + draws;
    double lo = 0.0, hi = 0.0;
    wilson(wins, decided > 0 ? decided : 1, lo, hi);
    /*
       得分率: **和棋算半分** (本工程的口径, 与 SAC/PPO 那几份对局报告一致)。
       ⚠ 第一版这里直接用 `wins/n` —— 于是"8 局全和"被打成 **0.0%**, 那是个假的
       低分 (和棋不是输)。工具的输出会被人当读数用, 所以它自己必须对。
    */
    const double score = (decided > 0)
                             ? ((double)wins + 0.5 * (double)draws) / (double)decided
                             : 0.0;
    std::printf("\n[bench] 比分: %d 胜 / %d 负 / %d 和 (共 %d 局有效, %d 局中止)\n",
                wins, losses, draws, decided, aborted);
    std::printf("[bench] 得分率(和棋半分) %.1f%%  |  胜率 %.1f%%  95%% Wilson [%.1f%%, %.1f%%]"
                "  (n=%d)\n",
                100.0 * score, decided > 0 ? 100.0 * (double)wins / (double)decided : 0.0,
                100.0 * lo, 100.0 * hi, decided);
    std::printf("[bench] 每步 %.1f ms | 池 %d 条 | 更新 %d 次 | |Q|均 %.4f | 夹住 %.1f%%\n",
                moves > 0 ? msSum / (double)moves : 0.0, ag.samplePoolSize(),
                ag.getLearnSteps(),
                ag.trainDiag.n > 0 ? ag.trainDiag.qAbsMeanSum / (double)ag.trainDiag.n : 0.0,
                ag.trainDiag.n > 0
                    ? 100.0 * (double)ag.trainDiag.clamped / (double)ag.trainDiag.n
                    : 0.0);
    if (decided < 200) {
        std::printf("[bench] ⚠ n=%d 远小于 200: 上面的区间就是结论 —— 这个比分**还不足以**\n"
                    "        判定任何旋钮的棋力差别 (本工程口径: 20 局的噪声 ±10~25 点)。\n",
                    decided);
    }
    delete ab;
    delete mcts;
    return (aborted > 0 || illegal > 0) ? 1 : 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv)) {
        return 2;
    }
    const int rc = (g.mode == "arena") ? runArena()
                                       : (g.mode == "agree" ? runAgree() : -1);
    if (rc < 0) {
        std::printf("[bench] 未知模式: %s (用 agree 或 arena)\n", g.mode.c_str());
        return 2;
    }
    return rc;
}
