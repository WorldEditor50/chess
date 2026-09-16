/*
 * bench_mcts_sims_main.cpp - 纯 MCTS (随机回放) 的模拟次数标度探针 (不进 ctest)
 * ============================================================================
 *
 * 为什么需要它: 纯 MCTS (AGENT_MCTS, GUI 里 "MCTS 800 次模拟") 的每步耗时是
 * 1 次迭代 ≈ 1.3 ms、800 次 ≈ 1.0 s。它与带网络的 MCTS (PPO+MCTS / SAC+AZ) 是
 * **两种完全不同的标度问题**:
 *   * 带网络的那条, 每次模拟要读一遍网络权重 (访存受限);
 *   * 纯 MCTS 每次迭代的成本几乎全在那条**最多 200 手的随机回放**上, 而随机走子
 *     在象棋里极难走到将杀 —— 于是绝大多数回放以"手数上限按和棋收尾"结束, 返回 0.0。
 *
 * 所以本工具量两件事:
 *   [1] **回放结果分布**: 和棋(到上限)/红胜/黑胜各占多少, 平均回放多少手。
 *       如果一个孩子的均值里绝大部分质量都是 0.0, 那么"多模拟"只是在把这个均值
 *       往 0 收敛 —— 它降低方差, 但**不带来关于好坏的信息**。
 *   [2] **多模拟改变不改变决策**: 同一批局面、同一份代码, 分别用 --sims-a / --sims-b
 *       搜一次, 比选点是否相同、根上第一名的访问份额、以及每步耗时。
 *
 * 用法:
 *   bench_mcts_sims.exe [--positions=20] [--sims-a=800] [--sims-b=1600]
 *                       [--opening=6] [--seed=20240901] [--rollouts=200]
 *                       [--rollout-plies=200] [--verbose]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "chess.h"
#include "mcts.h"

namespace {

int      g_positions = 20;
int      g_simsA = 800;
int      g_simsB = 1600;
int      g_opening = 6;
unsigned g_seed = 20240901u;
int      g_rollouts = 200;       /* 回放统计的样本数 */
int      g_rolloutPlies = 200;   /* 与 MCTS::simulateRandomPlay 的上限一致 */
bool     g_verbose = false;

double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* 与探针其它部分一致: 局面由局号决定, 可复现 */
void makePosition(int index, Chess &out)
{
    out.reset();
    out.sideToMove = Stone::COLOR_RED;
    std::mt19937 rng((unsigned)(g_seed + (unsigned)index * 7919u));
    for (int i = 0; i < g_opening; i++) {
        std::vector<Step *> legal;
        out.sample(out.sideToMove, legal);
        if (legal.empty()) { Steps::instance().put(legal); break; }
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        const Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        out.moveForward(&chosen, dummy);
    }
}

bool sameStep(const Step &x, const Step &y)
{
    return x.pos.x == y.pos.x && x.pos.y == y.pos.y &&
           x.nextPos.x == y.nextPos.x && x.nextPos.y == y.nextPos.y;
}

/*
 *  复刻 MCTS::simulateRandomPlay 的口径: 双方均匀随机走子, 直到将杀/无子可走,
 *  或到达手数上限 (按和棋)。返回 +1 (color 胜) / -1 (color 负) / 0 (和棋),
 *  并把实际回放手数写进 plies。
 */
struct RolloutOutcome { int result; int plies; bool limit; };

RolloutOutcome randomRollout(Chess &c, int color, int maxPlies, std::mt19937 &rng)
{
    RolloutOutcome o; o.result = 0; o.plies = 0; o.limit = false;
    std::vector<Step> played;
    double dummy = 0.0;
    int curr = color;

    while ((int)played.size() < maxPlies) {
        const int over = c.isGameOver();
        if (over != Stone::COLOR_NONE) {
            o.result = (over == color) ? 1 : -1;
            break;
        }
        std::vector<Step *> moves;
        c.sample(curr, moves);
        if (moves.empty()) {
            Steps::instance().put(moves);
            o.result = (curr == color) ? -1 : 1;
            break;
        }
        std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
        const Step chosen(*moves[pick(rng)]);
        Steps::instance().put(moves);
        c.moveForward(&chosen, dummy);
        played.push_back(chosen);
        curr = (curr == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    o.plies = (int)played.size();
    if (o.result == 0) { o.limit = (o.plies >= maxPlies); }

    /* 回退全部走法, 把棋盘留给下一次回放 */
    for (auto it = played.rbegin(); it != played.rend(); ++it) {
        c.moveBack(&(*it), dummy);
    }
    return o;
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const std::size_t eq = a.find('=');
        const std::string k = (eq == std::string::npos) ? a : a.substr(0, eq);
        const std::string v = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
        if (k == "--positions")        { g_positions = std::atoi(v.c_str()); }
        else if (k == "--sims-a")      { g_simsA = std::atoi(v.c_str()); }
        else if (k == "--sims-b")      { g_simsB = std::atoi(v.c_str()); }
        else if (k == "--opening")     { g_opening = std::atoi(v.c_str()); }
        else if (k == "--seed")        { g_seed = (unsigned)strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--rollouts")    { g_rollouts = std::atoi(v.c_str()); }
        else if (k == "--rollout-plies"){ g_rolloutPlies = std::atoi(v.c_str()); }
        else if (k == "--verbose")     { g_verbose = true; }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }
    if (g_positions < 1) { g_positions = 1; }
    if (g_simsA < 1) { g_simsA = 1; }
    if (g_simsB < 1) { g_simsB = 1; }

    std::printf("=== 纯 MCTS (随机回放) 模拟次数标度探针 ===\n");
    std::printf("局面     : %d 个 (随机开局 %d 手, seed=%u)\n", g_positions, g_opening, g_seed);
    std::printf("模拟次数 : A=%d, B=%d\n", g_simsA, g_simsB);
    std::printf("回放统计 : 每局面 %d 次回放, 上限 %d 手 (与 MCTS::simulateRandomPlay 同口径)\n\n",
                g_rollouts, g_rolloutPlies);

    /* ---------------- [1] 随机回放的结果分布 ---------------- */
    Chess c;
    c.reset();
    std::mt19937 rng(g_seed ^ 0x9e3779b9u);

    int draws = 0, wins = 0, losses = 0;
    long long plySum = 0;
    int full = 0;                     /* 走满上限的次数 */
    const int rTotal = g_positions * g_rollouts;

    const double t0 = nowMs();
    for (int i = 0; i < g_positions; i++) {
        makePosition(i, c);
        const int color = c.sideToMove;
        for (int r = 0; r < g_rollouts; r++) {
            const RolloutOutcome o = randomRollout(c, color, g_rolloutPlies, rng);
            plySum += o.plies;
            if (o.plies >= g_rolloutPlies) { full++; }
            if (o.result > 0)      { wins++; }
            else if (o.result < 0) { losses++; }
            else                   { draws++; }
        }
    }
    const double rSec = (nowMs() - t0) / 1000.0;

    if (rTotal > 0) {
        const double pDraw = (double)draws / (double)rTotal;
        /* 单次回放的价值是一个三值随机变量: 标准差 = sqrt(E[x²]-E[x]²) */
        const double e1 = ((double)wins - (double)losses) / (double)rTotal;
        const double e2 = ((double)wins + (double)losses) / (double)rTotal;
        const double sd = std::sqrt(std::max(0.0, e2 - e1 * e1));
        const double se800 = sd / std::sqrt((double)g_simsA);
        const double se1600 = sd / std::sqrt((double)g_simsB);
        std::printf("[1] 随机回放的结果分布 (%d 次回放, %.1f s, %.2f ms/次)\n",
                    rTotal, rSec, 1000.0 * rSec / (double)rTotal);
        std::printf("    走到手数上限(按和棋) : %.1f%%  (走满 %d 次)\n", 100.0 * pDraw, full);
        std::printf("    分胜负               : %.1f%%  (我方胜 %.1f%% / 我方负 %.1f%%)\n",
                    100.0 * (1.0 - pDraw), 100.0 * (double)wins / rTotal,
                    100.0 * (double)losses / rTotal);
        std::printf("    平均回放手数         : %.1f / 上限 %d\n",
                    (double)plySum / (double)rTotal, g_rolloutPlies);
        std::printf("    单次回放标准差       : %.3f  ->  %d 次模拟的均值标准误 = %.4f,"
                    " %d 次 = %.4f\n",
                    sd, g_simsA, se800, g_simsB, se1600);
        std::printf("    (若和棋占比接近 100%%, 均值就是往 0 收敛 —— 多模拟只降方差,"
                    " 不提供关于好坏的信息)\n\n");
    }

    /* ---------------- [2] 多模拟改变不改变决策 ----------------
       纯 MCTS 是**随机**算法 (回放用 std::rand), 所以"800 vs 1600 的选点差异"里混着
       两个来源: (a) 预算, (b) 随机性。要分开它们, 必须有一个**同预算的对照**:
       同一个局面跑两次 800 次, 看它们彼此一致多少 —— 那就是噪声地板。
       (MCTS 构造函数用 std::time 播种, 同一秒内构造的两个实例会拿到同一条随机流,
        所以同一秒内的两次 800 会逐位相同; 跨秒则不同。两种读数都如实报出来。)
    ------------------------------------------------------------------ */
    int counted = 0, sameAA = 0, sameAB = 0, sameBB = 0;
    double msA = 0.0, msB = 0.0;

    for (int i = 0; i < g_positions; i++) {
        makePosition(i, c);
        const int color = c.sideToMove;

        MCTS a1(c, 1.414);
        double t = nowMs();
        const Step m1 = a1.findBestMove(color, g_simsA);
        const double t1 = nowMs() - t;

        MCTS a2(c, 1.414);           /* 同预算的第二次: 噪声地板 */
        t = nowMs();
        const Step m2 = a2.findBestMove(color, g_simsA);
        const double t2 = nowMs() - t;

        MCTS b(c, 1.414);
        t = nowMs();
        const Step m3 = b.findBestMove(color, g_simsB);
        const double t3 = nowMs() - t;

        if (!m1.valid || !m2.valid || !m3.valid) { continue; }
        counted++;
        msA += (t1 + t2) * 0.5;
        msB += t3;
        if (sameStep(m1, m2)) { sameAA++; }
        if (sameStep(m1, m3)) { sameAB++; }
        if (sameStep(m2, m3)) { sameBB++; }

        if (g_verbose) {
            std::printf("  #%2d  %d: (%d,%d)->(%d,%d) %5.0f ms | %d(重跑): (%d,%d)->(%d,%d) %5.0f ms"
                        " | %d: (%d,%d)->(%d,%d) %5.0f ms   A/A %s  A/B %s\n",
                        i, g_simsA, m1.pos.x, m1.pos.y, m1.nextPos.x, m1.nextPos.y, t1,
                        g_simsA, m2.pos.x, m2.pos.y, m2.nextPos.x, m2.nextPos.y, t2,
                        g_simsB, m3.pos.x, m3.pos.y, m3.nextPos.x, m3.nextPos.y, t3,
                        sameStep(m1, m2) ? "同" : "异", sameStep(m1, m3) ? "同" : "异");
        }
    }

    if (counted == 0) { std::printf("**没有可比对的局面**\n"); return 1; }
    const double d = (double)counted;
    auto rate = [d](int k) { return 100.0 * (double)k / d; };
    auto seOf = [d](double p) { return std::sqrt(p * (100.0 - p) / d); };

    std::printf("[2] 选点一致性 (%d 个局面, 同一份代码)\n", counted);
    std::printf("    %d 次 vs %d 次(重跑) : %.1f%% ± %.1f   <- **噪声地板** (同预算)\n",
                g_simsA, g_simsA, rate(sameAA), seOf(rate(sameAA)));
    std::printf("    %d 次 vs %d 次       : %.1f%% ± %.1f   <- 加倍的预算带来的变化\n",
                g_simsA, g_simsB, rate(sameAB), seOf(rate(sameAB)));
    std::printf("    (第二次重跑 vs %d 次  : %.1f%%)\n", g_simsB, rate(sameBB));
    std::printf("    每步耗时             : %d 次 %.0f ms (%.3f ms/次)  /  %d 次 %.0f ms"
                " (%.3f ms/次)\n",
                g_simsA, msA / d, msA / d / (double)g_simsA,
                g_simsB, msB / d, msB / d / (double)g_simsB);
    std::printf("\n读法: 若「A vs B」的一致率与「A vs A 重跑」的噪声地板差不多, 就说明\n"
                "      把模拟次数翻倍**没有改变决策**, 只是把同一个随机估计算得更久。\n");
    std::printf("说明: 纯 MCTS 每次迭代都要跑一条最多 %d 手的随机回放, 成本几乎全在那里;\n"
                "      它与「带网络」的 MCTS 不是同一个标度问题 (后者是访存受限)。\n",
                g_rolloutPlies);
    return 0;
}
