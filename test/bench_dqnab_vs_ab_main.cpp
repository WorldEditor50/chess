/*
 * bench_dqnab_vs_ab_main.cpp - DQN+AB (AB 当 DQN 的 planning head) vs ABAgent 的静默验证
 * ============================================================================
 *
 * 与 bench_sacaz_vs_ab / bench_ppo_vs_ab 同一套约定: 不创建窗口、不弹对话框、不写权重
 * (除非显式 --save=), `--quiet` 只打一行 VERDICT。**退出码只看机制**:
 *   * 每一手都合法 (逐手 isLegalMove 校验);
 *   * 任何一方都不许返回无效走法;
 *   * 每局必须在手数上限内结束;
 *   * 对局后 V/Q 必须全部有限;
 *   * 搜索不得改动棋盘 (digest 逐字段比对)。
 * 比分**不进退出码** —— 默认是随机初始化的网络, 输给固定深度的 alpha-beta 是预期内的。
 *
 * 这个 bench 想回答的是"planning head 值不值", 所以除了比分还打三件事:
 *   1. 每步实际搜到的深度与节点数 (深度是 nodeBudget 的函数, 见 DQNABAgent 类注释 §5);
 *   2. ms/步 与 ms/节点 (TB 骨干与 MLP 骨干的差别就是"能不能搜得更深");
 *   3. 与"纯 Q-argmax"打同一个对手的对照 (--pure-q): 关掉搜索, 只按 Q 最大选点 ——
 *      这是"规划头有没有用"的直接 A/B。
 *
 * 用法:
 *   bench_dqnab_vs_ab.exe [--games=6] [--plies=100] [--depth=4]
 *                            [--budget=512] [--backbone=tb|mlp]
 *                            [--warmup-games=0] [--warmup-plies=40]
 *                            [--pretrain=N] [--pretrain-epochs=2] [--pretrain-plies=12]
 *                            [--pretrain-head-only] [--gate=0|1] [--no-rule-planes]
 *                            [--load=PREFIX] [--save=PREFIX] [--seed=N]
 *                            [--pure-q] [--verbose] [--quiet]
 *
 * --pretrain=N 是"引导阶梯第一级"的开关: 先用 N 个随机局面的**手工锚**
 * (tanh(±evaluate()/3.0)) 有监督地回归 V, 再开打。这样就能量出
 * "V 准不准"对同一个 nodeBudget 下棋力的影响 (设计文档 §7.4)。
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "abagent.h"
#include "chess.h"
#include "dqnabagent.h"
#include "rl/util.hpp"
#include "stone.h"

namespace {

struct Cfg {
    int games = 6;
    int maxPlies = 100;
    int depth = 4;               /* AB 对手的搜索深度 (GUI 同值) */
    int budget = 512;            /* DQN+AB 每次决策的节点预算 */
    int openingPlies = 4;
    int warmupGames = 0;
    int warmupPlies = 40;
    std::string backbone = "tb";
    std::string loadPrefix;
    std::string savePrefix;
    unsigned seed = 20240901;
    bool pureQ = false;          /* 关掉搜索, 只按 Q 取最大 (对照) */
    bool verbose = false;
    bool quiet = false;
    /* 手工锚预训练 (引导阶梯第一级, 见 dqnabagent.h 的 "手工评估锚" 一节) */
    int pretrainPositions = 0;   /* >0 才做 */
    int pretrainEpochs = 2;
    int pretrainMaxPlies = 60;   /* 随机局的长度: 12 手时标签几乎是常数, 见 rebuildProbes 注释 */
    bool pretrainTrunk = true;   /* false = 冻结主干, 只训 V 头 (便宜) */
    bool gate = true;            /* 在线更新的手工锚门控 */
    bool rulePlanes = true;      /* false = 消融: 规则上下文平面全置零 (维度不变) */
};

Cfg g_cfg;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

static std::string digest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[i];
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x) + "," + std::to_string(st->pos.y) + ";";
    }
    s += "side=" + std::to_string(c.sideToMove);
    s += "hash=" + std::to_string(c.computeHash());
    return s;
}

struct GameStat {
    int winner = Chess::RESULT_ONGOING;
    int plies = 0;
    bool naWasRed = false;
    int naMoves = 0, abMoves = 0;
    double naMs = 0.0, abMs = 0.0;
    long long nodes = 0;
    long long depthSum = 0;
    bool illegal = false;
    bool aborted = false;
    const char *endReason = "";
};

static int naScore(const GameStat &g)
{
    if (g.winner == Chess::RESULT_DRAW) { return 0; }
    const bool redWon = (g.winner == Chess::RESULT_RED_WIN);
    return (redWon == g.naWasRed) ? 1 : -1;
}

static void randomOpening(Chess &c, int &turn, int plies)
{
    for (int i = 0; i < plies; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) { return; }
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
        c.moveForward(&chosen, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
}

/* 只按 Q 取最大 (不搜索): 用来量"规划头"本身贡献了多少 */
static Step pureQMove(DQNABAgent &na, Chess &c, int turn)
{
    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    na.encodeStateFor(turn, st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    na.legalMoves(turn, legal, idx);
    if (legal.empty()) {
        Steps::instance().put(legal);
        return Step();
    }
    double v = 0.0;
    std::vector<double> q;
    na.evaluateNode(st, idx, v, q);
    std::size_t best = 0;
    for (std::size_t i = 1; i < q.size(); i++) {
        if (q[i] > q[best]) { best = i; }
    }
    const Step out = *legal[best];
    Steps::instance().put(legal);
    return out;
}

static GameStat playGame(Chess &c, DQNABAgent &na, ABAgent &ab,
                         bool naIsRed)
{
    GameStat g;
    g.naWasRed = naIsRed;

    c.reset();
    int turn = Stone::COLOR_RED;
    randomOpening(c, turn, g_cfg.openingPlies);

    double naMs = 0.0, abMs = 0.0;
    while (g.plies < g_cfg.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            g.winner = res;
            g.endReason = (res == Chess::RESULT_DRAW) ? "repetition/60-move rule" : "mate";
            break;
        }
        const bool naTurn = (turn == Stone::COLOR_RED) == naIsRed;

        /* 搜索不得改动棋盘: 每步都比一次 digest */
        const std::string before = digest(c);
        const double t0 = nowMs();
        const Step s = naTurn ? (g_cfg.pureQ ? pureQMove(na, c, turn)
                                             : na.selectMove(turn, 0.0f))
                              : ab.getBestMove(turn);
        const double ms = nowMs() - t0;
        if (naTurn) {
            naMs += ms;
            g.naMoves++;
            g.nodes += na.m_lastNodes;
            g.depthSum += na.m_lastDepth;
        } else {
            abMs += ms;
            g.abMoves++;
        }
        if (digest(c) != before) {
            g.aborted = true;
            g.endReason = naTurn ? "DQN+AB search modified the board" : "AB modified the board";
            break;
        }

        if (!s.valid) {
            g.aborted = true;
            g.endReason = naTurn ? "DQN+AB returned an invalid move" : "AB returned an invalid move";
            break;
        }
        if (!c.isLegalMove(turn, &s)) {
            g.illegal = true;
            g.endReason = naTurn ? "DQN+AB played an ILLEGAL move" : "AB played an ILLEGAL move";
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-4s (%d,%d)->(%d,%d) %7.1f ms%s\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        naTurn ? "  [NA]" : "  [AB]");
        }

        double dummy = 0.0;
        Step mv = s;
        c.moveForward(&mv, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        g.plies++;
    }

    if (g.winner == Chess::RESULT_ONGOING && !g.aborted && !g.illegal) {
        g.winner = Chess::RESULT_DRAW;
        g.endReason = "ply limit";
    }
    g.naMs = (g.naMoves > 0) ? naMs / (double)g.naMoves : 0.0;
    g.abMs = (g.abMoves > 0) ? abMs / (double)g.abMoves : 0.0;
    return g;
}

/* 对局之后查一次数值健康 (固定局面上的 V/Q 必须有限) */
static bool numericHealth(Chess &c, DQNABAgent &na)
{
    c.reset();
    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    na.encodeStateFor(Stone::COLOR_RED, st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    na.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);
    if (idx.empty()) { return true; }
    double v = 0.0;
    std::vector<double> q;
    na.evaluateNode(st, idx, v, q);
    if (!std::isfinite(v)) { return false; }
    for (double x : q) {
        if (!std::isfinite(x)) { return false; }
    }
    return true;
}

static bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *key) -> const char* {
            const std::size_t n = std::strlen(key);
            if (std::strncmp(a, key, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--games"))        { g_cfg.games = std::atoi(v); }
        else if (const char *v = val("--plies"))   { g_cfg.maxPlies = std::atoi(v); }
        else if (const char *v = val("--depth"))   { g_cfg.depth = std::atoi(v); }
        else if (const char *v = val("--budget"))  { g_cfg.budget = std::atoi(v); }
        else if (const char *v = val("--opening")) { g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--warmup-games")) { g_cfg.warmupGames = std::atoi(v); }
        else if (const char *v = val("--warmup-plies")) { g_cfg.warmupPlies = std::atoi(v); }
        else if (const char *v = val("--backbone")){ g_cfg.backbone = v; }
        else if (const char *v = val("--load"))    { g_cfg.loadPrefix = v; }
        else if (const char *v = val("--save"))    { g_cfg.savePrefix = v; }
        else if (const char *v = val("--seed"))    { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--pretrain")){ g_cfg.pretrainPositions = std::atoi(v); }
        else if (const char *v = val("--pretrain-epochs")) { g_cfg.pretrainEpochs = std::atoi(v); }
        else if (const char *v = val("--pretrain-plies"))  { g_cfg.pretrainMaxPlies = std::atoi(v); }
        else if (const char *v = val("--gate"))    { g_cfg.gate = std::atoi(v) != 0; }
        else if (std::strcmp(a, "--no-rule-planes") == 0) { g_cfg.rulePlanes = false; }
        else if (std::strcmp(a, "--pretrain-head-only") == 0) { g_cfg.pretrainTrunk = false; }
        else if (std::strcmp(a, "--pure-q") == 0)  { g_cfg.pureQ = true; }
        else if (std::strcmp(a, "--verbose") == 0) { g_cfg.verbose = true; }
        else if (std::strcmp(a, "--quiet") == 0)   { g_cfg.quiet = true; }
        else {
            std::printf("[错误] 未知参数: %s\n", a);
            return false;
        }
    }
    if (g_cfg.games < 2) { g_cfg.games = 2; }
    if (g_cfg.games % 2 != 0) { g_cfg.games++; }
    if (g_cfg.maxPlies < 2) { g_cfg.maxPlies = 2; }
    if (g_cfg.depth < 1) { g_cfg.depth = 1; }
    if (g_cfg.budget < 8) { g_cfg.budget = 8; }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!parseArgs(argc, argv)) {
        return 1;
    }
    RL::Random::setSeed(g_cfg.seed);

    DQNABAgent::Backbone bb = DQNABAgent::Backbone::SparseMoeTb;
    if (g_cfg.backbone == "mlp") {
        bb = DQNABAgent::Backbone::Mlp;
    } else if (g_cfg.backbone != "tb") {
        std::printf("[错误] 未知骨干: %s (可选: tb / mlp)\n", g_cfg.backbone.c_str());
        return 1;
    }

    Chess c;
    c.reset();
    const double tBuild0 = nowMs();
    DQNABAgent na(c, 64, 0.99f, 0.001f, bb);
    ABAgent ab(c, g_cfg.depth);
    const double tBuild1 = nowMs();

    na.nodeBudget = g_cfg.budget;
    na.searchDepth = 6;
    /*
       消融: 关掉规则上下文平面 (状态里"没有"重复计数/无吃子/将军), 输入维度与随机初始化
       都不变 —— 这样"补全 Markov 状态"对**比分**的贡献才是干净可比的一列。
    */
    na.encodeRulePlanes = g_cfg.rulePlanes;

    if (!g_cfg.quiet) {
        std::printf("=== DQN+AB (%s) vs Alpha-Beta 静默对弈验证 ===\n",
                    DQNABAgent::backboneName(bb));
        std::printf("DQN+AB: state=%d action=%d 主干参数=%lld 头参数=%lld\n",
                    DQNABAgent::STATE_DIM, DQNABAgent::ACTION_DIM,
                    na.trunkParamCount(), na.headParamCount());
        std::printf("          规划预算 %d 节点/步%s\n", na.nodeBudget,
                    g_cfg.pureQ ? "  **--pure-q: 关掉搜索, 只按 Q 取最大 (对照)**" : "");
        std::printf("AB      : 深度 %d\n", g_cfg.depth);
        std::printf("规则    : %d 局 (交换先后手), 每局最多 %d 手, 随机开局 %d 步, seed=%u\n",
                    g_cfg.games, g_cfg.maxPlies, g_cfg.openingPlies, g_cfg.seed);
        std::printf("构建耗时: %.1f s\n", (tBuild1 - tBuild0) / 1000.0);
    }

    if (!g_cfg.loadPrefix.empty()) {
        if (!na.loadModel(g_cfg.loadPrefix)) {
            std::printf("VERDICT: FAIL (权重载入失败: %s)\n", g_cfg.loadPrefix.c_str());
            return 1;
        }
        if (!g_cfg.quiet) {
            std::printf("权重    : %s -> 载入成功\n", g_cfg.loadPrefix.c_str());
        }
    } else if (!g_cfg.quiet) {
        std::printf("权重    : 未载入 —— 网络是**随机初始化**的, 比分不是棋力\n");
    }

    if (g_cfg.warmupGames > 0) {
        const double t0 = nowMs();
        na.trainPlanDepth = 1;
        na.trainSelfPlay(g_cfg.warmupGames, g_cfg.warmupPlies, false, 1.0f, 0.25f);
        if (!g_cfg.quiet) {
            std::printf("热身    : 自对弈 %d 局 x ≤%d 手, %.1f s; 池=%zu, 学习步数=%d\n",
                        g_cfg.warmupGames, g_cfg.warmupPlies, (nowMs() - t0) / 1000.0,
                        na.replaySize(), na.learnSteps());
        }
    }

    /*
       引导阶梯第一级: 先把 V 有监督地拉向手工锚, 再让它参与搜索/自对弈。
       这一段的 before/after gap 必须打出来 —— "预训练有没有用"不能靠感觉,
       而且 gap 是同一个探针集合上的量, 前后可比。
    */
    na.valueGateEnabled = g_cfg.gate;
    if (g_cfg.pretrainPositions > 0) {
        const double gapPost = na.pretrainValueFromHand(g_cfg.pretrainPositions,
                                                        g_cfg.pretrainMaxPlies,
                                                        32, g_cfg.pretrainEpochs,
                                                        g_cfg.pretrainTrunk, true);
        if (!g_cfg.quiet) {
            std::printf("预训练  : 手工锚 %d 局面 x %d 遍 (主干%s), gap %.4f -> %.4f, "
                        "corr %.3f -> %.3f%s, %.1f s\n",
                        g_cfg.pretrainPositions, g_cfg.pretrainEpochs,
                        g_cfg.pretrainTrunk ? "参与" : "冻结",
                        na.pretrainStatsBefore().gap, gapPost,
                        na.pretrainStatsBefore().corr, na.pretrainStatsAfter().corr,
                        na.lastPretrainRolledBack() ? "  [已回滚]" : "",
                        na.lastPretrainMs() / 1000.0);
            std::printf("          (锚标准差 %.4f —— 这是这把尺子的信号量; "
                        "常数 V 的 gap 也小, 看 corr)\n",
                        na.pretrainStatsBefore().anchorStd);
        }
    }
    if (!g_cfg.quiet) {
        std::printf("门控    : 在线更新%s (阈值 %.3f x lr 缩放 = %.3f; 探针 %d 局面)\n",
                    g_cfg.gate ? "开" : "关",
                    (double)na.valueGateTolerance, na.gateToleranceNow(),
                    na.valueGateProbeCount);
        std::printf("状态    : %d 维 (14 棋子平面 + 5 规则/阶段平面)%s\n",
                    DQNABAgent::STATE_DIM,
                    g_cfg.rulePlanes ? "" : "  **消融: 规则上下文平面全置零**");
    }
    if (!g_cfg.quiet) { std::printf("\n"); }

    int naWins = 0, abWins = 0, draws = 0, broken = 0;
    int pliesTotal = 0;
    double naMsPerMove = 0.0, abMsPerMove = 0.0;
    int naMoveSamples = 0, abMoveSamples = 0;
    long long nodesTotal = 0, depthTotal = 0;
    bool allFinite = true;

    const double tStart = nowMs();
    for (int i = 0; i < g_cfg.games; i++) {
        const bool naIsRed = (i % 2 == 0);
        const GameStat g = playGame(c, na, ab, naIsRed);
        const int score = naScore(g);
        const bool bad = g.illegal || g.aborted;
        if (bad)            { broken++; }
        else if (score > 0) { naWins++; }
        else if (score < 0) { abWins++; }
        else                { draws++; }

        pliesTotal += g.plies;
        if (g.naMoves > 0) {
            naMsPerMove += g.naMs * g.naMoves;
            naMoveSamples += g.naMoves;
            nodesTotal += g.nodes;
            depthTotal += g.depthSum;
        }
        if (g.abMoves > 0) {
            abMsPerMove += g.abMs * g.abMoves;
            abMoveSamples += g.abMoves;
        }
        allFinite = allFinite && numericHealth(c, na);

        if (!g_cfg.quiet) {
            const char *verdict = bad ? "BROKEN"
                                : (score > 0) ? "NA wins"
                                : (score < 0) ? "AB wins" : "draw";
            std::printf("  game %2d/%d  NA=%-5s  %-8s %3d 手  [%s]  na %6.1f ms/步 "
                        "(%.0f 节点/步, 深度 %.1f), ab %5.1f ms/步\n",
                        i + 1, g_cfg.games, naIsRed ? "RED" : "BLACK", verdict, g.plies,
                        g.endReason, g.naMoves > 0 ? g.naMs : 0.0,
                        g.naMoves > 0 ? (double)g.nodes / g.naMoves : 0.0,
                        g.naMoves > 0 ? (double)g.depthSum / g.naMoves : 0.0, g.abMs);
        }
    }
    const double elapsed = (nowMs() - tStart) / 1000.0;

    const double naMs = naMoveSamples > 0 ? naMsPerMove / naMoveSamples : 0.0;
    const double abMs = abMoveSamples > 0 ? abMsPerMove / abMoveSamples : 0.0;
    /*
       手工锚 gap / corr 在**对局之后**量: 它们是"V 准不准"的唯一量化出口, 也是门控用的
       那把尺子。静默跑也能从 VERDICT 行看到 (否则"V 被自对弈带偏了"没有任何痕迹)。
       **corr 才是尺度无关的判据**: 常数 V 的 gap 也小 (锚的幅度只有 ±0.2), 但 corr ≡ 0。
    */
    const DQNABAgent::HandStats hs = na.netHandStats(0);
    const double gapFinal = hs.gap;

    if (!g_cfg.quiet) {
        std::printf("\n=== 结果 (DQN+AB 视角) ===\n");
        std::printf("  比分      : DQN+AB %d 胜 / AB %d 胜 / 和 %d", naWins, abWins, draws);
        if (broken > 0) { std::printf("  (+%d 局机制违规)", broken); }
        std::printf("\n");
        std::printf("  得分率    : %.1f%%  (胜=1 和=0.5; 随机权重下这个数字没有棋力含义)\n",
                    100.0 * ((double)naWins + 0.5 * (double)draws) / (double)g_cfg.games);
        std::printf("  平均手数  : %.1f\n", (double)pliesTotal / (double)g_cfg.games);
        std::printf("  思考时间  : DQN+AB %.1f ms/步, AB %.1f ms/步  ->  AB 是 NA 的 %.2fx\n",
                    naMs, abMs, naMs > 0.0 ? abMs / naMs : 0.0);
        std::printf("  规划      : %.0f 节点/步, 平均深度 %.2f\n",
                    naMoveSamples > 0 ? (double)nodesTotal / naMoveSamples : 0.0,
                    naMoveSamples > 0 ? (double)depthTotal / naMoveSamples : 0.0);
        std::printf("  数值健康  : V/Q 全有限 = %d\n", (int)allFinite);
        std::printf("  手工锚    : gap %.4f, corr %.3f (锚标准差 %.4f, V 标准差 %.4f)\n",
                    gapFinal, hs.corr, hs.anchorStd, hs.vStd);
        std::printf("  总耗时    : %.1f s\n", elapsed);
    }

    if (!g_cfg.savePrefix.empty()) {
        const bool ok = na.saveModel(g_cfg.savePrefix);
        if (!g_cfg.quiet) {
            std::printf("  权重已保存: %s_trunk / _v / _a -> %s\n",
                        g_cfg.savePrefix.c_str(), ok ? "成功" : "失败");
        }
    }

    const bool mechanismOk = (broken == 0) && allFinite;
    std::printf("VERDICT: %s | 骨干=%s 局数=%d 比分=%d/%d/%d 违规=%d 数值有限=%d "
                "平均深度=%.2f 手工锚gap=%.4f corr=%.3f\n",
                mechanismOk ? "PASS" : "FAIL",
                DQNABAgent::backboneName(bb), g_cfg.games,
                naWins, abWins, draws, broken, (int)allFinite,
                naMoveSamples > 0 ? (double)depthTotal / naMoveSamples : 0.0,
                gapFinal, hs.corr);
    return mechanismOk ? 0 : 1;
}
