/*
 * bench_anchor_main.cpp - P0.3: 固定锚点评测 (不进 ctest)
 * ============================================================================
 *
 * 解决什么问题: 在它之前, "棋力有没有涨"只能靠 4~6 局随机开局的 bench —— 那个
 * 噪声带是 ±5% 得分率, 分辨不了 80 Elo (80 Elo ≈ 61.5% 得分率), 于是任何
 * "回滚权重 / 给对手降权 / 换奖励" 的决策都**不可证伪**
 * (docs/issues_review.md:1908 与 docs/training_optimization.md:916 把"快照 Elo 阶梯"
 *  列为未做, 理由正是"每次 bench_* 都是新进程、没有常驻对手池")。
 *
 * 本程序给出**可复现、可配对、带区间**的那把尺子, 三件事:
 *
 *   1. **固定开局集**: 开局由 (seed, index) 确定性地生成, 全程只用一个局部 mt19937_64,
 *      **不碰 RL::Random** —— 于是开局集与 agent 的随机流无关, 换个权重跑的还是同一批
 *      局面。开局集印出 FNV 指纹, 两次运行指纹不同 = 开局集变了 (改过走法生成?),
 *      此时两次结果**不可比** —— 宁可吵, 也不能静默地比两批不同的局面。
 *      (--dump-book 把开局导出来人工核对; 刻意**不做**从文件回读: Step 没有公开的
 *       "从坐标重建"入口, 手搓 Step 的 id/nextId 不一致是静默错误的高发区。)
 *
 *   2. **成对计分**: 每个开局下两局, 锚点 A 先执红、再执黑 —— 中国象棋先手优势大,
 *      不换先手等于在测"谁执红"。两局互为对照, 先手优势在一阶上抵消。
 *
 *   3. **区间而不是点**: 报得分率 + Elo 差 + 95% 区间 (Elo 由得分率换算:
 *      ΔElo = -400·log10(1/S - 1))。区间跨过 0 才叫"没测出差别"; 只报一个数字
 *      必然被当成结论。
 *
 * 评测路径**不开**根噪声 (selectMove 走 withRootNoise=false), 温度 0 = argmax,
 * 于是"同一权重 + 同一开局集"的对局是**逐位可复现**的 —— 这是配对比较的前提。
 *
 * 用法:
 *   cmake --build <build> --target bench_anchor
 *   # 随机初始化的 A 对 AB 深度 4 (先确认链路通)
 *   <build>/bench_anchor --openings=2 --plies=4 --sims=8 --ab-depth=2
 *   # 两个快照头对头 (换成真权重才有意义)
 *   <build>/bench_anchor --a=weights/run_10k --b=weights/run_5k --openings=20 --plies=8
 *   # 等时间对 AB (把 PPO 的模拟次数按预算标定出来)
 *   <build>/bench_anchor --a=weights/run_10k --budget=100 --ab-depth=4 --openings=20
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "abagent.h"
#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/util.hpp"

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    std::string aPrefix;              /* 锚点 A 的权重前缀 (空 = 随机初始化) */
    std::string bPrefix;              /* 对手 B 的权重前缀 (空且 abDepth>0 = 用 AB) */
    int abDepth = 4;                  /* B 是 AB 时的深度 (0 = 不用 AB) */
    int openings = 20;                /* 开局集大小 */
    int openingPlies = 8;             /* 每个开局的随机手数 */
    int maxPlies = 60;                /* 每局手数上限 (与训练一致: 60 ply) */
    int sims = 400;                   /* A 每步模拟次数 (等时间模式下由 --budget 覆写) */
    long long budgetMs = 0;           /* >0 = 等时间: 按 AB 每步实测耗时标定 PPO 的 sims */
    int minSims = 8, maxSims = 4000;
    unsigned seed = 20240901u;
    std::string csvPrefix;
    std::string dumpBook;
    bool verbose = false;
};

Cfg g_cfg;

/* ============================================================
 *  小工具
 * ============================================================ */
double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int flipColor(int c)
{
    return (c == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}

const char *resultName(int r)
{
    if (r == Chess::RESULT_RED_WIN) { return "红胜"; }
    if (r == Chess::RESULT_BLACK_WIN) { return "黑胜"; }
    if (r == Chess::RESULT_DRAW) { return "和"; }
    return "未终局";
}

/* ============================================================
 *  固定开局集
 *
 *  与各 bench 的 --opening 同一套做法 (随机走合法棋), 但**随机源是局部的** ——
 *  这样开局集只由 (seed, index) 决定, 不受 agent 内部 RNG 消耗顺序的影响。
 * ============================================================ */
typedef std::vector<Step> Opening;

unsigned long long fnv1a(unsigned long long h, unsigned long long v)
{
    h ^= v;
    return h * 1099511628211ull;
}

bool makeOpenings(Chess &scratch, int count, int plies, unsigned seed,
                  std::vector<Opening> &out, unsigned long long &fingerprint)
{
    out.clear();
    fingerprint = 1469598103934665603ull;   /* FNV-1a 偏移基 */
    for (int i = 0; i < count; i++) {
        std::mt19937_64 rng((unsigned long long)seed * 7919ull
                            + (unsigned long long)i * 104729ull + 17ull);
        scratch.reset();
        Opening op;
        for (int p = 0; p < plies; p++) {
            if (scratch.getResult(scratch.sideToMove) != Chess::RESULT_ONGOING) {
                break;
            }
            std::vector<Step *> legal;
            scratch.sample(scratch.sideToMove, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                break;
            }
            std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
            const Step s = *legal[(std::size_t)pick(rng)];
            Steps::instance().put(legal);
            op.push_back(s);
            double dummy = 0.0;
            scratch.moveForward(&s, dummy);
        }
        if (op.empty()) {
            std::printf("[错误] 第 %d 个开局造不出来 (开局的随机手数太多?)\n", i);
            return false;
        }
        for (std::size_t k = 0; k < op.size(); k++) {
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.y);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.y);
        }
        out.push_back(op);
    }
    return true;
}

/* 把开局摆上棋盘; 返回该谁走 */
int applyOpening(Chess &c, const Opening &op)
{
    c.reset();
    for (std::size_t i = 0; i < op.size(); i++) {
        double dummy = 0.0;
        c.moveForward(&op[i], dummy);
    }
    return c.sideToMove;
}

/* ============================================================
 *  一局
 * ============================================================ */
struct GameRec {
    int result = Chess::RESULT_ONGOING;   /* Chess::RESULT_* */
    int plies = 0;
    int drawReason = (int)Chess::DRAW_NONE;
    bool truncated = false;
    bool aWasRed = true;
    bool aborted = false;
    double aMs = 0.0, bMs = 0.0;
    int aMoves = 0, bMoves = 0;
};

/*
 *  A 对 B 打一局。B 可以是另一个 PPO 权重 (bPpo 非空) 或 AB 引擎 (ab 非空)。
 *  两个 agent 都绑在**同一张对局棋盘**上 (与 bench_ppo_vs_ab / bench_ppo_sims 同做法):
 *  搜索内部会 moveForward/moveBack 并复原, 所以共享对局棋盘是安全的。
 */
GameRec playGame(Chess &c, PPOMCTSAgent &a, PPOMCTSAgent *bPpo, ABAgent *ab,
                 const Opening &op, bool aIsRed, int simsForA)
{
    GameRec g;
    g.aWasRed = aIsRed;

    int turn = applyOpening(c, op);

    while (g.plies < g_cfg.maxPlies) {
        Chess::DrawReason dr = Chess::DRAW_NONE;
        const int res = c.getResult(turn, &dr);
        if (res != Chess::RESULT_ONGOING) {
            g.result = res;
            g.drawReason = (int)dr;
            break;
        }

        const bool aTurn = (turn == Stone::COLOR_RED) == aIsRed;
        const double t0 = nowMs();
        const Step s = aTurn ? a.selectMove(turn, simsForA, 0.0f)
                             : (bPpo != nullptr ? bPpo->selectMove(turn, simsForA, 0.0f)
                                                : ab->getBestMove(turn));
        const double ms = nowMs() - t0;
        if (aTurn) { g.aMs += ms; g.aMoves++; } else { g.bMs += ms; g.bMoves++; }

        if (!s.valid) {
            /* selectMove / getBestMove 都有"无走法"兜底, 走到这里说明真的没有合法走法,
               或者 agent 返回了无效走法 —— 两者都要显式报出来, 不能当和棋。 */
            g.aborted = true;
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-5s (%d,%d)->(%d,%d) %7.1f ms %s\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        aTurn ? "[A]" : "[B]");
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = flipColor(turn);
        g.plies++;
    }

    if (g.result == Chess::RESULT_ONGOING) {
        /* 走到手数上限: 这是**台架截断**, 不是规则和棋 —— 单列一栏, 别混进"和棋"的
           解释里 (见 rl/diag.h 的 GameEndKind)。 */
        g.truncated = true;
        g.result = Chess::RESULT_DRAW;
    }
    return g;
}

/* 一局里 A 的得分 (胜 1 / 和 0.5 / 负 0) */
double scoreOf(const GameRec &g)
{
    if (g.result == Chess::RESULT_DRAW) { return 0.5; }
    const bool redWon = (g.result == Chess::RESULT_RED_WIN);
    return (redWon == g.aWasRed) ? 1.0 : 0.0;
}

/* 得分率 -> Elo 差 (白方/红方视角无所谓, 差值是相对的) */
double eloFromScore(double s)
{
    const double eps = 1e-6;
    const double x = std::min(1.0 - eps, std::max(eps, s));
    return -400.0 * std::log10(1.0 / x - 1.0);
}

/* ============================================================
 *  参数
 * ============================================================ */
bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        std::string k = a, v;
        if (eq != nullptr) {
            k = a.substr(0, (std::size_t)(eq - argv[i]));
            v = a.substr((std::size_t)(eq - argv[i]) + 1);
        }
        if (k == "--a") { g_cfg.aPrefix = v; }
        else if (k == "--b") { g_cfg.bPrefix = v; }
        else if (k == "--ab-depth") { g_cfg.abDepth = std::atoi(v.c_str()); }
        else if (k == "--openings") { g_cfg.openings = std::atoi(v.c_str()); }
        else if (k == "--plies") { g_cfg.openingPlies = std::atoi(v.c_str()); }
        else if (k == "--max-plies") { g_cfg.maxPlies = std::atoi(v.c_str()); }
        else if (k == "--sims") { g_cfg.sims = std::atoi(v.c_str()); }
        else if (k == "--budget") { g_cfg.budgetMs = std::atoll(v.c_str()); }
        else if (k == "--min-sims") { g_cfg.minSims = std::atoi(v.c_str()); }
        else if (k == "--max-sims") { g_cfg.maxSims = std::atoi(v.c_str()); }
        else if (k == "--seed") { g_cfg.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--csv") { g_cfg.csvPrefix = v; }
        else if (k == "--dump-book") { g_cfg.dumpBook = v; }
        else if (k == "--verbose") { g_cfg.verbose = true; }
        else if (k == "--help" || k == "-h") {
            std::printf(
                "用法: bench_anchor [--a=PREFIX] [--b=PREFIX] [--ab-depth=N]\n"
                "                   [--openings=N] [--plies=N] [--max-plies=N]\n"
                "                   [--sims=N] [--budget=MS] [--seed=N]\n"
                "                   [--csv=PREFIX] [--dump-book=FILE] [--verbose]\n"
                "  --a        锚点 A 的权重前缀 (空 = 随机初始化, 只验链路)\n"
                "  --b        对手 B 的权重前缀 (给了就是 PPO 对 PPO)\n"
                "  --ab-depth B 用 AB 引擎时的深度 (默认 4; 给了 --b 就忽略)\n"
                "  --budget   等时间: 按 AB 每步实测耗时反推 A 的模拟次数\n");
            return false;
        } else {
            std::printf("[警告] 未知参数: %s\n", argv[i]);
        }
    }
    if (g_cfg.openings < 1) { g_cfg.openings = 1; }
    if (g_cfg.openingPlies < 0) { g_cfg.openingPlies = 0; }
    if (g_cfg.maxPlies < 1) { g_cfg.maxPlies = 1; }
    return true;
}

/* AB 每步平均耗时 (标定等时间模式用) */
double calibrateAbMsPerMove(Chess &c, ABAgent &ab)
{
    c.reset();
    for (int i = 0; i < 6; i++) {
        std::vector<Step *> legal;
        c.sample(c.sideToMove, legal);
        if (legal.empty()) { Steps::instance().put(legal); break; }
        const Step s = *legal[0];
        Steps::instance().put(legal);
        double dummy = 0.0;
        c.moveForward(&s, dummy);
    }
    const double t0 = nowMs();
    const int probes = 2;
    for (int i = 0; i < probes; i++) {
        ab.getBestMove(c.sideToMove);
    }
    return (nowMs() - t0) / (double)probes;
}

/* A 的 ms/模拟 (标定用) */
double calibrateMsPerSim(Chess &c, PPOMCTSAgent &a)
{
    const int probeSims = 40;
    c.reset();
    const double t0 = nowMs();
    a.selectMove(c.sideToMove, probeSims, 0.0f);
    const double ms = nowMs() - t0;
    return (probeSims > 0) ? ms / (double)probeSims : 0.0;
}

}  // namespace

/* ============================================================
 *  main
 * ============================================================ */
int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv)) {
        return 0;
    }
    RL::Random::setSeed(g_cfg.seed);

    const bool usePpoOpponent = !g_cfg.bPrefix.empty();
    const bool useAbOpponent = (!usePpoOpponent && g_cfg.abDepth > 0);
    if (!usePpoOpponent && !useAbOpponent) {
        std::printf("[错误] 没有对手: 给 --b=PREFIX 或 --ab-depth>0\n");
        return 2;
    }

    Chess board;   /* 对局棋盘 (两个 agent 共用, 见 playGame 的说明) */
    PPOMCTSAgent agentA(board, 64, 0.99f, 0.001f, 1.414f);
    PPOMCTSAgent *agentB = nullptr;
    ABAgent *ab = nullptr;
    if (usePpoOpponent) {
        agentB = new PPOMCTSAgent(board, 64, 0.99f, 0.001f, 1.414f);
    } else {
        ab = new ABAgent(board, g_cfg.abDepth);
    }

    std::printf("=== bench_anchor (P0.3 固定锚点评测) ===\n");
    std::printf("  锚点 A : %s\n",
                g_cfg.aPrefix.empty() ? "随机初始化 (棋力指标无意义)"
                                      : g_cfg.aPrefix.c_str());
    if (usePpoOpponent) {
        std::printf("  对手 B : PPO %s\n", g_cfg.bPrefix.c_str());
    } else {
        std::printf("  对手 B : Alpha-Beta 深度 %d\n", g_cfg.abDepth);
    }

    if (!g_cfg.aPrefix.empty() && !agentA.loadModel(g_cfg.aPrefix)) {
        std::printf("[错误] A 载入失败: %s\n", g_cfg.aPrefix.c_str());
        return 1;
    }
    if (usePpoOpponent && agentB != nullptr && !agentB->loadModel(g_cfg.bPrefix)) {
        std::printf("[错误] B 载入失败: %s\n", g_cfg.bPrefix.c_str());
        return 1;
    }
    /* 评测路径不开根噪声 (与 selectMove 的既有约定一致); 显式再关一次, 免得
       以后有人把默认值改了而这里悄悄跟着变。 */
    agentA.evalRootNoise = false;
    if (agentB != nullptr) { agentB->evalRootNoise = false; }

    /* ---- 固定开局集 ---- */
    std::vector<Opening> openings;
    unsigned long long fingerprint = 0;
    if (!makeOpenings(board, g_cfg.openings, g_cfg.openingPlies, g_cfg.seed,
                      openings, fingerprint)) {
        return 1;
    }
    std::printf("  开局集 : %d 个 x %d 手 | seed=%u | 指纹 %016llX\n",
                (int)openings.size(), g_cfg.openingPlies, g_cfg.seed, fingerprint);
    std::printf("           (指纹变了 = 开局集变了, 两次运行**不可比**)\n");

    if (!g_cfg.dumpBook.empty()) {
        RL::Diag::CsvWriter book;
        if (book.open(g_cfg.dumpBook, "opening,ply,from_x,from_y,to_x,to_y")) {
            char buf[128];
            for (std::size_t i = 0; i < openings.size(); i++) {
                for (std::size_t k = 0; k < openings[i].size(); k++) {
                    const Step &s = openings[i][k];
                    std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d,%d,%d",
                                  (int)i, (int)k + 1, s.pos.x, s.pos.y,
                                  s.nextPos.x, s.nextPos.y);
                    book.row(buf);
                }
            }
            std::printf("  开局集已导出: %s\n", g_cfg.dumpBook.c_str());
        } else {
            std::printf("  [警告] 开局集导出失败: %s\n", g_cfg.dumpBook.c_str());
        }
    }

    /* ---- 等时间标定 (只对 AB 对手有意义) ---- */
    int simsForA = g_cfg.sims;
    if (g_cfg.budgetMs > 0) {
        if (!useAbOpponent) {
            std::printf("  [警告] --budget 只在对手是 AB 时标定; PPO 对 PPO 用相等的 --sims\n");
        } else {
            const double abMs = calibrateAbMsPerMove(board, *ab);
            const double msPerSim = calibrateMsPerSim(board, agentA);
            if (msPerSim > 1e-6) {
                int s = (int)((double)g_cfg.budgetMs / msPerSim);
                s = std::max(g_cfg.minSims, std::min(g_cfg.maxSims, s));
                simsForA = s;
                std::printf("  等时间 : AB %.1f ms/步 | A %.3f ms/模拟 -> A 每步 %d 次模拟"
                            " (目标 %lld ms)\n",
                            abMs, msPerSim, simsForA, g_cfg.budgetMs);
            }
        }
    }
    std::printf("  每步   : A %d 次模拟 | 手数上限 %d ply | 每开局下 2 局 (换先手)\n",
                simsForA, g_cfg.maxPlies);

    /* ============================================================
     *  逐开局、换先手成对开打
     * ============================================================ */
    std::vector<GameRec> recs;
    RL::Diag::CsvWriter tb;
    bool csvOpen = false;
    if (!g_cfg.csvPrefix.empty()) {
        csvOpen = tb.open(g_cfg.csvPrefix + "_tb.csv", "tag,step,value");
    }

    const double t0 = nowMs();
    for (std::size_t i = 0; i < openings.size(); i++) {
        double opScore = 0.0;
        std::printf("  开局 %2d/%d (指纹已固定): ", (int)i + 1, (int)openings.size());
        for (int c = 0; c < 2; c++) {
            const bool aIsRed = (c == 0);
            GameRec g = playGame(board, agentA, agentB, ab, openings[i], aIsRed, simsForA);
            const double sc = scoreOf(g);
            opScore += sc;
            recs.push_back(g);
            /* 用 ASCII 标签对齐 (中文字符串按字节算宽度, %-6s 对不齐) */
            std::printf("%s %-4s(%.1f) ", aIsRed ? "A-red" : "A-blk",
                        resultName(g.result), sc);
        }
        std::printf("-> 该开局 A 得分 %.2f/2\n", opScore);
    }
    const double totalSec = (nowMs() - t0) / 1000.0;

    /* ============================================================
     *  汇总: 得分率 + Elo + 95% 区间 + 和棋构成
     * ============================================================ */
    const int n = (int)recs.size();
    double sum = 0.0, sumSq = 0.0;
    int wins = 0, losses = 0, draws = 0, trunc = 0, aborted = 0;
    int repDraws = 0, noCapDraws = 0;
    double aMsSum = 0.0, bMsSum = 0.0;
    int aMoveSum = 0, bMoveSum = 0;
    for (int i = 0; i < n; i++) {
        const double s = scoreOf(recs[i]);
        sum += s;
        sumSq += s * s;
        if (s == 1.0) { wins++; } else if (s == 0.0) { losses++; } else { draws++; }
        if (recs[i].truncated) { trunc++; }
        if (recs[i].aborted) { aborted++; }
        if (recs[i].drawReason == (int)Chess::DRAW_REPEAT) { repDraws++; }
        if (recs[i].drawReason == (int)Chess::DRAW_NO_CAPTURE60) { noCapDraws++; }
        aMsSum += recs[i].aMs; aMoveSum += recs[i].aMoves;
        bMsSum += recs[i].bMs; bMoveSum += recs[i].bMoves;
    }
    const double mean = (n > 0) ? sum / (double)n : 0.0;
    const double var = (n > 1) ? std::max(0.0, sumSq / (double)n - mean * mean) : 0.0;
    const double se = (n > 1) ? std::sqrt(var / (double)n) : 0.0;
    const double lo = std::max(0.0, mean - 1.96 * se);
    const double hi = std::min(1.0, mean + 1.96 * se);

    std::printf("\n=== 锚点评测结果 ===\n");
    std::printf("  对局 %d 局 (%.1f s, 平均 %.1f s/局)\n", n, totalSec,
                (n > 0) ? totalSec / (double)n : 0.0);
    std::printf("  A 得分 %.4f  (胜 %d / 和 %d / 负 %d)\n", mean, wins, draws, losses);
    std::printf("  得分率 95%% 区间 [%.4f, %.4f]  (SE %.4f%s)\n", lo, hi, se,
                (se > 0.02) ? " —— 区间偏宽: 想分辨 80 Elo 需更多开局" : "");
    std::printf("  Elo 差  %+.1f  (区间 %+.1f ~ %+.1f)  %s\n",
                eloFromScore(mean), eloFromScore(lo), eloFromScore(hi),
                (lo <= 0.5 && hi >= 0.5) ? "<- 区间跨过 50%: 没测出差别" : "");
    std::printf("  和棋构成: 三次重复 %d | 自然限着 %d | 台架截断 %d (占和棋 %.2f)\n",
                repDraws, noCapDraws, trunc,
                (draws > 0) ? (double)trunc / (double)draws : 0.0);
    if (aborted > 0) {
        std::printf("  **[警告] %d 局出现无效走法/非法终局 —— 这本身是要修的缺陷\n", aborted);
    }
    std::printf("  每步耗时: A %.1f ms (%d 手) | B %.1f ms (%d 手)\n",
                (aMoveSum > 0) ? aMsSum / (double)aMoveSum : 0.0, aMoveSum,
                (bMoveSum > 0) ? bMsSum / (double)bMoveSum : 0.0, bMoveSum);
    if (useAbOpponent && g_cfg.budgetMs <= 0) {
        std::printf("  (上面两行若量级差很多, 这次比的是谁算得多而不是谁更强;"
                    " 要等时间就加 --budget=MS)\n");
    }

    if (csvOpen) {
        RL::Diag::scalarRow(tb, "anchor_score", 1, mean);
        RL::Diag::scalarRow(tb, "anchor_score_lo", 1, lo);
        RL::Diag::scalarRow(tb, "anchor_score_hi", 1, hi);
        RL::Diag::scalarRow(tb, "anchor_elo", 1, eloFromScore(mean));
        RL::Diag::scalarRow(tb, "anchor_elo_lo", 1, eloFromScore(lo));
        RL::Diag::scalarRow(tb, "anchor_elo_hi", 1, eloFromScore(hi));
        RL::Diag::scalarRow(tb, "anchor_games", 1, (double)n);
        RL::Diag::scalarRow(tb, "anchor_draws", 1, (double)draws);
        RL::Diag::scalarRow(tb, "anchor_truncated", 1, (double)trunc);
        /*
           刻意**不**把开局集指纹写进 CSV: TB 长表只有 `tag,step,value` 三列, 64 位
           指纹塞进 double 会丢精度, 于是"CSV 里那个数"与"屏幕上印的那个指纹"对不上
           —— 一个无法与之比较的指纹比没有更糟。指纹只从 stdout 里读 (两次运行的
           指纹必须逐字符相同才能比)。
        */
        std::printf("  CSV: %s_tb.csv\n", g_cfg.csvPrefix.c_str());
    }

    delete agentB;
    delete ab;
    return (aborted > 0) ? 1 : 0;
}
