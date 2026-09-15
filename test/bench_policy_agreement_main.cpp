/*
 * bench_policy_agreement_main.cpp - 选点一致率探针 (不进 ctest)
 * ============================================================================
 *
 * 为什么需要它: 上一轮实测出**20 局等时间对局的噪声带是 ±5%**（"5% vs 0%" 就是一局
 * 之差），于是"某个改动有没有让 agent 变强"这个问题在那个协议下根本回答不了 ——
 * 300 局训练后的三组对照全部落在噪声里。
 *
 * 这个探针换一个**灵敏得多**的指标: 在**同一批固定局面**上，比较 PPO+MCTS 的选点与
 * ABAgent(深度 D) 的选点是否一致。每个局面一个二值样本 ⇒ 200 个样本的标准误约 3.5%，
 * 比 20 局胜负灵敏一个量级；而且局面集由种子固定，**两个权重文件跑的是同一批局面**,
 * 所以对比是配对的（方差进一步下降）。
 *
 * 它衡量的是"策略有没有朝更强玩家的选择靠" —— 是**代理指标**，不是棋力本身。
 * 两个必须注意的边界:
 *   1. 如果正好是"用 AB 的着法做蒸馏"，那这个指标会因构造而上升，失去独立性。
 *      这时应该把参照换成**更深的 AB**（--depth=5）, 或者直接用等时间对局（但要够多局）。
 *   2. 一致率上升不等于胜率上升（两人选同一手棋，强弱仍可能不同）。
 * 所以它只用来回答"**这一改动是否改变了策略、以及朝哪个方向改变**"，用来把噪声里的
 * 差异分出来；最终结论仍要靠足够局数的对局。
 *
 * 用法:
 *   bench_policy_agreement.exe [--positions=200] [--sims=80] [--depth=4]
 *                              [--opening=6] [--seed=20240901]
 *                              [--load=PREFIX] [--random=1] [--verbose=0]
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
#include "ppomcts_agent.h"
#include "rl/util.hpp"

namespace {

int    g_positions = 200;
int    g_sims = 80;
int    g_depth = 4;
int    g_opening = 6;
unsigned g_seed = 20240901u;
std::string g_loadPrefix;
bool   g_randomControl = true;
bool   g_verbose = false;

double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/*
 *  用"随机走 opening 手"造一个可复现的局面。
 *  刻意**不用 agent 的选点**来造局面 —— 否则两个权重文件的局面集就不同了,
 *  配对比较也就没了。
 */
void makePosition(int index, Chess &out)
{
    out.reset();
    out.sideToMove = Stone::COLOR_RED;
    std::mt19937 rng((unsigned)(g_seed + (unsigned)index * 7919u));
    for (int i = 0; i < g_opening; i++) {
        std::vector<Step *> legal;
        out.sample(out.sideToMove, legal);
        if (legal.empty()) break;
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        out.moveForward(&chosen, dummy);
    }
}

} // namespace

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const std::size_t eq = a.find('=');
        const std::string k = (eq == std::string::npos) ? a : a.substr(0, eq);
        const std::string v = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
        if (k == "--positions")   { g_positions = std::atoi(v.c_str()); }
        else if (k == "--sims")   { g_sims = std::atoi(v.c_str()); }
        else if (k == "--depth")  { g_depth = std::atoi(v.c_str()); }
        else if (k == "--opening"){ g_opening = std::atoi(v.c_str()); }
        else if (k == "--seed")   { g_seed = (unsigned)strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--load")   { g_loadPrefix = v; }
        else if (k == "--random") { g_randomControl = (std::atoi(v.c_str()) != 0); }
        else if (k == "--verbose"){ g_verbose = (std::atoi(v.c_str()) != 0); }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }

    std::printf("=== 选点一致率探针 ===\n");
    std::printf("局面     : %d 个 (随机开局 %d 手, seed=%u; 局面集与权重无关, 两次跑是同一批)\n",
                g_positions, g_opening, g_seed);
    std::printf("PPO+MCTS : %d 次模拟/步 (温度 0, 取访问最多的着法)\n", g_sims);
    std::printf("AB-Agent : 深度 %d (作为参照)\n", g_depth);

    /*
       必须**先播种再构造 agent**: 网络权重是随机初始化的, 而 RL::Random 是线程本地的,
       没显式播种时按线程身份派生 —— 于是"同一个命令跑两次"会拿到不同的初始权重,
       配对比较直接失效 (第一版实测 7.0% vs 6.0%, 就是差在这里)。
       --load 时这一步没有影响 (权重会被覆盖), 但留着它保证"未训练"那一行也可复现。
    */
    RL::Random::setSeed(g_seed);

    /* 两个 agent 各有一块独立的棋盘 —— selectMove 会在棋盘上试走再回退,
       共用一个实例会互相干扰 */
    Chess envPpo, envAb;
    envPpo.reset();
    envAb.reset();

    PPOMCTSAgent ppo(envPpo, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true);
    ppo.replayBatchSize = 0;   /* 只做决策, 不学习 */
    if (!g_loadPrefix.empty()) {
        const bool ok = ppo.loadModel(g_loadPrefix);
        std::printf("权重     : %s -> %s\n", g_loadPrefix.c_str(),
                    ok ? "载入成功" : "**载入失败**, 用随机初始权重");
    } else {
        std::printf("权重     : 未载入 (随机初始化)\n");
    }
    ABAgent ab(envAb, g_depth);

    int sameMove = 0;      /* (from, to) 完全一致 */
    int sameFrom = 0;      /* 起点相同 (同类意图) */
    int randSame = 0;      /* 随机走法 vs AB 的一致率 —— 机会水平 */
    double ppoMs = 0.0, abMs = 0.0, randMs = 0.0;
    int counted = 0;

    for (int i = 0; i < g_positions; i++) {
        Chess pos;
        makePosition(i, pos);
        const int color = pos.sideToMove;
        std::vector<Step *> legal;
        pos.sample(color, legal);
        if (legal.empty()) { continue; }
        const std::size_t nLegal = legal.size();
        Steps::instance().put(legal);

        /* 两个 agent 各在自己的棋盘上、从同一个局面出发 */
        envPpo = pos;
        envAb  = pos;

        const double t0 = nowMs();
        const Step ppoMove = ppo.selectMove(color, g_sims, 0.0f);
        ppoMs += nowMs() - t0;

        const double t1 = nowMs();
        const Step abMove = ab.getBestMove(color, g_depth);
        abMs += nowMs() - t1;

        if (!ppoMove.valid || !abMove.valid) { continue; }
        counted++;

        const bool same = (ppoMove.pos.x == abMove.pos.x && ppoMove.pos.y == abMove.pos.y
                           && ppoMove.nextPos.x == abMove.nextPos.x
                           && ppoMove.nextPos.y == abMove.nextPos.y);
        const bool fromSame = (ppoMove.pos.x == abMove.pos.x
                               && ppoMove.pos.y == abMove.pos.y);
        if (same) sameMove++;
        if (fromSame) sameFrom++;

        /* 机会水平: 随机挑一个合法着法, 看它跟 AB 一致的概率 (≈ 1/分支数) */
        if (g_randomControl) {
            std::mt19937 rng((unsigned)(g_seed * 31u + (unsigned)i));
            std::vector<Step *> lg;
            envAb.sample(color, lg);
            if (!lg.empty()) {
                std::uniform_int_distribution<std::size_t> pick(0, lg.size() - 1);
                const Step r = *lg[pick(rng)];
                if (r.pos.x == abMove.pos.x && r.pos.y == abMove.pos.y
                        && r.nextPos.x == abMove.nextPos.x
                        && r.nextPos.y == abMove.nextPos.y) {
                    randSame++;
                }
                Steps::instance().put(lg);
            }
        }

        if (g_verbose) {
            std::printf("  #%3d 合法%2llu  PPO=(%d,%d)->(%d,%d)  AB=(%d,%d)->(%d,%d)  %s\n",
                        i, (unsigned long long)nLegal,
                        ppoMove.pos.x, ppoMove.pos.y, ppoMove.nextPos.x, ppoMove.nextPos.y,
                        abMove.pos.x, abMove.pos.y, abMove.nextPos.x, abMove.nextPos.y,
                        same ? "一致" : "");
        }
    }

    if (counted == 0) {
        std::printf("\n**没有可比对的局面**\n");
        return 1;
    }

    const double agreeMove = 100.0 * (double)sameMove / (double)counted;
    const double agreeFrom = 100.0 * (double)sameFrom / (double)counted;
    const double randRate  = g_randomControl ? 100.0 * (double)randSame / (double)counted : 0.0;
    const double se = std::sqrt(agreeMove * (100.0 - agreeMove) / (double)counted);

    std::printf("\n=== 结果 (%d 个局面) ===\n", counted);
    std::printf("  完整着法一致率 : %5.1f%%  (标准误 %.1f%%)\n", agreeMove, se);
    std::printf("  起点一致率     : %5.1f%%\n", agreeFrom);
    if (g_randomControl) {
        std::printf("  随机走法的机会水平: %5.1f%%   <-- 低于它才说明策略比乱走更差\n",
                    randRate);
    }
    std::printf("  耗时           : PPO %.1f ms/步, AB %d 层 %.1f ms/步\n",
                ppoMs / (double)counted, g_depth, abMs / (double)counted);
    std::printf("\n说明: 这是**代理指标**(策略是否朝更强玩家靠), 不是棋力; 结论仍要对局来定。\n"
                "      配对比较的用法: 同一命令只换 --load, 比较两行\"完整着法一致率\"。\n");
    return 0;
}
