/*
 * bench_ppo_sims_main.cpp - 「同一份权重、只改模拟次数」的配对头对头 (不进 ctest)
 * ============================================================================
 *
 * 为什么需要它: "把模拟次数从 80 提到 800 能不能提高胜率" 这个问题, 现有的两个尺子都
 * 回答不了:
 *   - `bench_ppo_vs_ab` 只能回答"相对 AB 深度 D 的得分率", 而 AB 是一个**固定的天花板**;
 *     分数率被压在 0~5% 的地板上时, "多搜索有没有用" 的分辨力几乎没有。
 *   - `bench_policy_agreement` 是**代理指标**(选点是否朝 AB 靠), 与胜率不是同一件事。
 *
 * 这里直接让**同一条网络的两个分身**互相对打, 唯一的差别是每步模拟次数 (--sims-a /
 * --sims-b)。于是:
 *   * 权重、编码、c_puct、温度(0, argmax)、开局、先后手规则**全部相同**;
 *   * 得分率 > 50% 就只有一个解释: 多搜索让**同一个策略**下得更好了;
 *   * 没有被模仿的老师、没有被压地板的天花板 —— 这是"搜索本身值不值钱"的干净测法。
 *
 * 三条必须说清的边界:
 *   1. **不训练** (replayBatchSize=0)。否则两边会各自学, 比的就不再是"搜索"了。
 *   2. 两边的权重必须逐位相同。--load 时靠同一个文件保证; 不给 --load 时靠**两次播种
 *      构造**保证 (RL::Random 是线程本地的, 不播种就会从同一条流里各拿一份不同权重 ——
 *      这个坑在 issues_review 零之二点十六里踩过一次)。
 *   3. 开局由**局号**决定 (不是随机流), 所以同一个 --seed 下, 不同 --sims 的两次运行
 *      打的是**同一批开局** ⇒ 跨配置也是配对的。
 *
 * 用法:
 *   bench_ppo_sims.exe [--games=20] [--plies=120] [--sims-a=80] [--sims-b=800]
 *                      [--opening=4] [--seed=20250101] [--load=PREFIX] [--verbose]
 *
 * 典型跑法 (同一权重, 三档对照):
 *   bench_ppo_sims.exe --games=20 --load=weights/sweep_prior --sims-a=80 --sims-b=80
 *   bench_ppo_sims.exe --games=20 --load=weights/sweep_prior --sims-a=80 --sims-b=800
 *   bench_ppo_sims.exe --games=20 --load=weights/sweep_prior --sims-a=400 --sims-b=800
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

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/util.hpp"

namespace {

struct Cfg {
    int games       = 20;      /* 必须是偶数: 交换先后手 */
    int maxPlies    = 120;
    int simsA       = 80;      /* A 方每步模拟次数 */
    int simsB       = 800;     /* B 方每步模拟次数 */
    int openingPlies = 4;
    unsigned seed   = 20250101u;
    std::string loadPrefix;
    bool verbose    = false;
    int probePositions = 0;    /* >0 时只做根节点访问分布诊断, 不打对局 */
    /*
       平局(手数上限/重复局面)占多数时, 单看 W/L/D 分辨不出强弱 —— 12 局全和棋等于
       什么都没量。--material=1 时, 对**平局**的棋局额外用终局局面的 evaluate()
       (材质 + PST, 黑方视角, 归一化后量程约 ±4.5) 当"谁占优"的代理读数:
         * 连续读数: A 视角的最终评估均值 (比红黑胜负灵敏得多);
         * 阈值判定: |s| > --material-thresh (默认 0.2 ≈ 一个兵) 才算一方胜。
       **决定胜负的棋局不参与材料判定** (它们已经有真实结果)。这是一个代理指标,
       不是棋力 —— 但它比"全和棋"有分辨力, 所以两条都报。
    */
    bool materialTiebreak = false;
    double materialThresh = 0.2;
};

Cfg g_cfg;

double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/*
 *  随机开局: 由**局号**播种, 与 agent 的模拟次数无关 —— 于是不同 --sims 的两次运行
 *  打的是同一批起始局面 (跨配置配对的基础)。走法用 RL::Random::engine 之外的独立
 *  引擎抽, 免得动到 agent 的随机流。
 */
void openingFor(int gameIndex, Chess &c)
{
    c.reset();
    std::mt19937 rng((unsigned)(g_cfg.seed + (unsigned)gameIndex * 7919u));
    int turn = Stone::COLOR_RED;
    for (int i = 0; i < g_cfg.openingPlies; i++) {
        std::vector<Step *> legal;
        c.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            return;
        }
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        const Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        c.moveForward(&chosen, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
}

/* ============================================================
 *  根节点诊断 (--probe-positions>0): 只回答"多搜索把访问数花到哪去了"
 *
 *  对同一批局面, 同一份权重, 分别用 A/B 两档模拟次数搜一次, 记录:
 *    * 根的孩子数 (= 该局面的分支数) —— 与"深挖余量 = sims − 分支数"这条机制直接相关
 *    * 访问最多的孩子的**访问份额** (是否因为多搜索而更集中)
 *    * 前两名的 Q 差 (多搜索有没有把好坏着法的差距拉开)
 *    * 两侧**选中的着法是否相同** (最直接的"多搜索有没有改变决策")
 * ============================================================ */
struct RootStat {
    bool valid = false;
    int  children = 0;
    int  visits = 0;
    int  topVisits = 0;
    double topQ = 0.0;
    double secondQ = 0.0;
    int  mv[4] = { 0, 0, 0, 0 };   /* 选中着法: from.x, from.y, to.x, to.y */
    double ms = 0.0;
};

RootStat probeRoot(PPOMCTSAgent &ag, Chess &c, int color, int sims)
{
    RootStat st;
    const double t0 = nowMs();
    const Step mv = ag.selectMove(color, sims, 0.0f);
    st.ms = nowMs() - t0;
    if (!mv.valid) { return st; }
    st.valid = true;
    st.mv[0] = mv.pos.x; st.mv[1] = mv.pos.y; st.mv[2] = mv.nextPos.x; st.mv[3] = mv.nextPos.y;

    const int root = ag.currentRoot();
    if (root < 0 || (std::size_t)root >= ag.nodes.size()) { return st; }

    struct Child { int visits; double q; };
    std::vector<Child> kids;
    for (int cid : ag.nodes[(std::size_t)root].childIDs) {
        if (cid < 0 || (std::size_t)cid >= ag.nodes.size()) { continue; }
        const PPOMCTSAgent::AZNode &ch = ag.nodes[(std::size_t)cid];
        Child k; k.visits = ch.visitCount; k.q = ch.getQ();
        kids.push_back(k);
    }
    std::sort(kids.begin(), kids.end(),
              [](const Child &x, const Child &y) { return x.visits > y.visits; });
    st.children = (int)kids.size();
    for (const Child &k : kids) { st.visits += k.visits; }
    if (!kids.empty()) {
        st.topVisits = kids[0].visits;
        st.topQ = kids[0].q;
        if (kids.size() > 1) { st.secondQ = kids[1].q; }
    }
    (void)c;
    return st;
}

static bool sameMove(const int x[4], const int y[4])
{
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2] && x[3] == y[3];
}

struct GameStat {

    int  winner;
    int  plies;
    bool aWasRed;
    int  aMoves, bMoves;
    double aMs, bMs;
    bool aborted;
    const char *endReason;

    GameStat()
        : winner(Chess::RESULT_ONGOING), plies(0), aWasRed(false), aMoves(0), bMoves(0),
          aMs(0.0), bMs(0.0), aborted(false), endReason("") {}
};

/* +1 A 胜, -1 B 胜, 0 和 (A 视角) */
int aScore(const GameStat &g)
{
    if (g.winner == Chess::RESULT_DRAW) return 0;
    const bool redWon = (g.winner == Chess::RESULT_RED_WIN);
    return (redWon == g.aWasRed) ? 1 : -1;
}

GameStat playGame(Chess &c, PPOMCTSAgent &a, PPOMCTSAgent &b, int gameIndex)
{
    GameStat g;
    g.aWasRed = (gameIndex % 2 == 0);

    openingFor(gameIndex, c);
    int turn = (g_cfg.openingPlies % 2 == 0) ? Stone::COLOR_RED : Stone::COLOR_BLACK;

    while (g.plies < g_cfg.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            g.winner = res;
            g.endReason = (res == Chess::RESULT_DRAW) ? "repetition/60-move rule" : "mate";
            break;
        }

        const bool aTurn = (turn == Stone::COLOR_RED) == g.aWasRed;
        const int sims = aTurn ? g_cfg.simsA : g_cfg.simsB;
        const double t0 = nowMs();
        const Step s = (aTurn ? a : b).selectMove(turn, sims, 0.0f);
        const double ms = nowMs() - t0;
        if (aTurn) { g.aMs += ms; g.aMoves++; } else { g.bMs += ms; g.bMoves++; }

        if (!s.valid) {
            g.aborted = true;
            g.endReason = aTurn ? "A returned an invalid move" : "B returned an invalid move";
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-4s (%d,%d)->(%d,%d)  %7.1f ms  [%s %d sims]\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        aTurn ? "A" : "B", sims);
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        g.plies++;
    }

    if (g.winner == Chess::RESULT_ONGOING && !g.aborted) {
        g.winner = Chess::RESULT_DRAW;   /* 到上限判和, 与 bench_ppo_vs_ab 同口径 */
        g.endReason = "ply limit";
    }
    return g;
}

void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *key) -> const char* {
            const std::size_t n = std::strlen(key);
            if (std::strncmp(a, key, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--games"))     { g_cfg.games = std::atoi(v); }
        else if (const char *v = val("--plies")){ g_cfg.maxPlies = std::atoi(v); }
        else if (const char *v = val("--sims-a")){ g_cfg.simsA = std::atoi(v); }
        else if (const char *v = val("--sims-b")){ g_cfg.simsB = std::atoi(v); }
        else if (const char *v = val("--opening")){ g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--seed"))  { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--load"))  { g_cfg.loadPrefix = v; }
        else if (const char *v = val("--probe-positions")) { g_cfg.probePositions = std::atoi(v); }
        else if (const char *v = val("--material")) { g_cfg.materialTiebreak = (std::atoi(v) != 0); }
        else if (const char *v = val("--material-thresh")) { g_cfg.materialThresh = std::atof(v); }
        else if (std::strcmp(a, "--verbose") == 0) { g_cfg.verbose = true; }
        else { std::fprintf(stderr, "未知参数: %s\n", a); std::exit(2); }
    }
    if (g_cfg.games < 2) { g_cfg.games = 2; }
    if (g_cfg.games % 2 != 0) { g_cfg.games++; }
    if (g_cfg.maxPlies < 2) { g_cfg.maxPlies = 2; }
    if (g_cfg.simsA < 1) { g_cfg.simsA = 1; }
    if (g_cfg.simsB < 1) { g_cfg.simsB = 1; }
    if (g_cfg.openingPlies < 0) { g_cfg.openingPlies = 0; }
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    parseArgs(argc, argv);

    Chess c;
    c.reset();

    /*
       两个分身必须拿到**逐位相同**的权重。--load 时由同一个文件保证; 不 load 时
       靠"两次播种构造"保证 —— RL::Random 是线程本地的, 不播种就是从同一条流里
       各拿一份不同的随机权重, 那样比的就不是搜索而是初始化了。
    */
    RL::Random::setSeed(g_cfg.seed);
    PPOMCTSAgent a(c, 64, 0.99f, 0.001f, 1.414f);
    a.replayBatchSize = 0;      /* 不训练: 只比搜索 */
    RL::Random::setSeed(g_cfg.seed);
    PPOMCTSAgent b(c, 64, 0.99f, 0.001f, 1.414f);
    b.replayBatchSize = 0;

    bool sameWeights = true;
    if (!g_cfg.loadPrefix.empty()) {
        const bool okA = a.loadModel(g_cfg.loadPrefix);
        const bool okB = b.loadModel(g_cfg.loadPrefix);
        sameWeights = (okA && okB);
        std::printf("权重     : %s -> A %s / B %s\n", g_cfg.loadPrefix.c_str(),
                    okA ? "成功" : "**失败**", okB ? "成功" : "**失败**");
    } else {
        std::printf("权重     : 未载入 —— 两边是**同一个种子**下的同一份随机初始化\n");
    }

    std::printf("=== PPO+MCTS 同权重、只改模拟次数的头对头 ===\n");
    std::printf("A 方     : 每步 %d 次模拟\n", g_cfg.simsA);
    std::printf("B 方     : 每步 %d 次模拟\n", g_cfg.simsB);
    std::printf("规则     : %d 局 (交换先后手), 每局最多 %d 手, 随机开局 %d 步"
                " (由局号决定), seed=%u\n", g_cfg.games, g_cfg.maxPlies,
                g_cfg.openingPlies, g_cfg.seed);
    if (!sameWeights) {
        std::printf("**两边的权重不一致, 这次结果不能用来归因搜索 —— 检查 --load**\n");
    }
    std::printf("\n");

    /* ============================================================
       根节点诊断模式 (--probe-positions=N): 不打对局, 只量"多搜索把访问花到哪"
       ============================================================ */
    if (g_cfg.probePositions > 0) {
        const int n = g_cfg.probePositions;
        int counted = 0, sameChosen = 0;
        double childSum = 0.0, shareA = 0.0, shareB = 0.0;
        double gapA = 0.0, gapB = 0.0, msA = 0.0, msB = 0.0;
        int visitedA = 0, visitedB = 0;   /* 有访问计数的统计 */
        int aChangedOnly = 0;

        for (int i = 0; i < n; i++) {
            openingFor(i, c);
            const int turn = (g_cfg.openingPlies % 2 == 0) ? Stone::COLOR_RED
                                                           : Stone::COLOR_BLACK;
            const RootStat sa = probeRoot(a, c, turn, g_cfg.simsA);
            const RootStat sb = probeRoot(b, c, turn, g_cfg.simsB);
            if (!sa.valid || !sb.valid) { continue; }
            counted++;
            childSum += (double)sa.children;
            msA += sa.ms; msB += sb.ms;
            if (sa.visits > 0) {
                shareA += 100.0 * (double)sa.topVisits / (double)sa.visits;
                visitedA++;
            }
            if (sb.visits > 0) {
                shareB += 100.0 * (double)sb.topVisits / (double)sb.visits;
                visitedB++;
            }
            gapA += sa.topQ - sa.secondQ;
            gapB += sb.topQ - sb.secondQ;
            const bool same = sameMove(sa.mv, sb.mv);
            if (same) { sameChosen++; } else if (sa.mv[2] >= 0) { aChangedOnly++; }

            if (g_cfg.verbose) {
                std::printf("  #%3d 根孩子 %2d  %d 模拟: 访问份额 %5.1f%% 前二 Q 差 %+.3f"
                            " (%d,%d)->(%d,%d) | %d 模拟: %5.1f%% %+.3f (%d,%d)->(%d,%d)  %s\n",
                            i, sa.children,
                            g_cfg.simsA, (sa.visits > 0) ? 100.0 * sa.topVisits / sa.visits : 0.0,
                            sa.topQ - sa.secondQ, sa.mv[0], sa.mv[1], sa.mv[2], sa.mv[3],
                            g_cfg.simsB, (sb.visits > 0) ? 100.0 * sb.topVisits / sb.visits : 0.0,
                            sb.topQ - sb.secondQ, sb.mv[0], sb.mv[1], sb.mv[2], sb.mv[3],
                            same ? "同" : "**不同**");
            }
        }
        if (counted == 0) { std::printf("**没有可比对的局面**\n"); return 1; }

        const double d = (double)counted;
        std::printf("\n=== 根节点诊断 (%d 个局面, 同一份权重) ===\n", counted);
        std::printf("  分支数(根孩子)      : 均 %.1f  ->  %d 次模拟的「深挖余量」= %.1f,"
                    " %d 次 = %.1f\n",
                    childSum / d, g_cfg.simsA, (double)g_cfg.simsA - childSum / d,
                    g_cfg.simsB, (double)g_cfg.simsB - childSum / d);
        std::printf("  访问份额(第一名孩子) : %d 模拟 %.1f%%  vs  %d 模拟 %.1f%%\n",
                    g_cfg.simsA, (visitedA > 0) ? shareA / (double)visitedA : 0.0,
                    g_cfg.simsB, (visitedB > 0) ? shareB / (double)visitedB : 0.0);
        std::printf("  前二名 Q 差          : %d 模拟 %+.4f  vs  %d 模拟 %+.4f\n",
                    g_cfg.simsA, gapA / d, g_cfg.simsB, gapB / d);
        std::printf("  选中的着法相同       : %.1f%%  (不同 %d 个)\n",
                    100.0 * (double)sameChosen / d, counted - sameChosen);
        std::printf("  每步耗时             : %d 模拟 %.1f ms  vs  %d 模拟 %.1f ms"
                    "  ->  %.2f / %.2f ms per 模拟\n",
                    g_cfg.simsA, msA / d, g_cfg.simsB, msB / d,
                    msA / d / (double)g_cfg.simsA, msB / d / (double)g_cfg.simsB);
        std::printf("\n读法: 「选中的着法相同」接近 100%% 就说明多搜索不改变决策;\n"
                    "      访问份额与 Q 差变大只说明搜索**更自信**, 不等于**更正确**。\n");
        return 0;
    }

    int aWins = 0, bWins = 0, draws = 0, aborted = 0, pliesTotal = 0;
    double aMs = 0.0, bMs = 0.0;
    int aMoves = 0, bMoves = 0;
    /* 每局的 g.aMs 是**该局每步平均**, 所以汇总必须按"局"平均 —— 按"步"平均会把
       每步耗时低估约 (总步数/局数) 倍 (第一版踩过, 单位对不上却看起来很正常)。 */
    int aGames = 0, bGames = 0;

    const double tStart = nowMs();
    int matDecidedA = 0, matDecidedB = 0, matDraws = 0;
    double matSumA = 0.0, matSumSq = 0.0;
    int matCount = 0;

    for (int i = 0; i < g_cfg.games; i++) {
        const GameStat g = playGame(c, a, b, i);
        const int sc = aScore(g);
        if (g.aborted)      { aborted++; }
        else if (sc > 0)    { aWins++; }
        else if (sc < 0)    { bWins++; }
        else                { draws++; }

        /*
           材料判定: 只对**平局**的棋局做 (决定胜负的棋局已经有真实结果)。
           evaluate() 是黑方视角的 "材质 + PST" (归一化, 量程约 ±4.5), 翻到 A 视角。
        */
        double sBlack = 0.0, sA = 0.0;
        const char *matVerdict = "";
        if (g_cfg.materialTiebreak && !g.aborted && g.winner == Chess::RESULT_DRAW) {
            sBlack = c.evaluate();
            sA = g.aWasRed ? -sBlack : sBlack;
            matSumA += sA; matSumSq += sA * sA; matCount++;
            if (sA > g_cfg.materialThresh)       { matDecidedA++; matVerdict = "材料:A 占优"; }
            else if (sA < -g_cfg.materialThresh) { matDecidedB++; matVerdict = "材料:B 占优"; }
            else                                 { matDraws++;    matVerdict = "材料:均势"; }
        }

        pliesTotal += g.plies;
        aMs += g.aMs; bMs += g.bMs;
        aMoves += g.aMoves; bMoves += g.bMoves;
        if (g.aMoves > 0) { aGames++; }
        if (g.bMoves > 0) { bGames++; }

        const char *verdict = g.aborted ? "ABORTED"
                            : (sc > 0) ? "A wins" : (sc < 0) ? "B wins" : "draw";
        std::printf("  game %2d/%d  A=%-4s  %-8s in %3d plies  [%s]"
                    "  A %6.1f ms/move, B %6.1f ms/move",
                    i + 1, g_cfg.games, g.aWasRed ? "RED" : "BLACK",
                    verdict, g.plies, g.endReason, g.aMs, g.bMs);
        if (g_cfg.materialTiebreak && matVerdict[0] != '\0') {
            std::printf("  [%s %+.2f]", matVerdict, sBlack);
        }
        std::printf("\n");
    }
    const double elapsed = (nowMs() - tStart) / 1000.0;

    const int decided = aWins + bWins + draws;
    const double scoreA = (decided > 0)
        ? 100.0 * ((double)aWins + 0.5 * (double)draws) / (double)decided : 0.0;
    /* 得分率的二项标准误 (按"局"算, 和=0.5 的方差已按 0.5 计) */
    const double se = (decided > 0)
        ? std::sqrt(scoreA * (100.0 - scoreA) / (double)decided) * 0.5 : 0.0;
    const double msPerSimA = (aGames > 0) ? (aMs / (double)aGames) / (double)g_cfg.simsA : 0.0;
    const double msPerSimB = (bGames > 0) ? (bMs / (double)bGames) / (double)g_cfg.simsB : 0.0;

    std::printf("\n=== 结果 (A = %d 模拟, B = %d 模拟) ===\n", g_cfg.simsA, g_cfg.simsB);
    std::printf("  比分      : A %d 胜 / B %d 胜 / 和 %d", aWins, bWins, draws);
    if (aborted > 0) { std::printf("  (+%d 局中止, 不计入)", aborted); }
    std::printf("\n");
    std::printf("  A 得分率  : %.1f%%  (标准误约 %.1f%%; 50%% 即「多搜索毫无区别」)\n",
                scoreA, se);
    std::printf("  平均手数  : %.1f\n", (double)pliesTotal / (double)g_cfg.games);
    std::printf("  思考时间  : A %.1f ms/步 (%.3f ms/模拟), B %.1f ms/步 (%.3f ms/模拟)"
                "  ->  B 是 A 的 %.1fx\n",
                (aGames > 0) ? aMs / (double)aGames : 0.0, msPerSimA,
                (bGames > 0) ? bMs / (double)bGames : 0.0, msPerSimB,
                (msPerSimA > 0.0) ? (msPerSimB * (double)g_cfg.simsB)
                                        / (msPerSimA * (double)g_cfg.simsA) : 0.0);
    std::printf("  对局总耗时: %.1f s\n", elapsed);
    if (g_cfg.materialTiebreak) {
        std::printf("  材料判定 (只对和棋, 阈值 ±%.2f): A 占优 %d / B 占优 %d / 均势 %d\n",
                    g_cfg.materialThresh, matDecidedA, matDecidedB, matDraws);
        if (matCount > 0) {
            const double mean = matSumA / (double)matCount;
            const double var = std::max(0.0, matSumSq / (double)matCount - mean * mean);
            const double sem = std::sqrt(var / (double)matCount);
            std::printf("  A 视角终局评估均值: %+.3f ± %.3f (n=%d; 正值 = A 占优)"
                        "  <- 比「红黑胜负」灵敏, 但仍是代理指标\n",
                        mean, sem, matCount);
        }
    }
    std::printf("\n说明: 这是**同一策略下「多搜索」的净收益**, 不含「训练」与「对手天花板」两个混淆项;\n"
                "      但它也只回答「同权重下 800 次是否强于 80 次」, 不代表整体棋力。\n");
    return 0;
}
