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
/* R1: 同权重、同局面、只改"先验口径"的配对 A/B (见 main 里的说明) */
bool   g_ab = false;
/* B-5: 同权重、同局面序列、只改"子树复用开/关"的配对 A/B (见 main 里的说明) */
bool   g_reuseAb = false;
/* R2: c_puct 重扫 (PUCT 的探索常数; 先验口径改成"合法集上归一"之后它的含义变了) */
float  g_cpuct = 1.414f;

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
        else if (k == "--ab")     { g_ab = (std::atoi(v.c_str()) != 0); }
        else if (k == "--reuse-ab") { g_reuseAb = (std::atoi(v.c_str()) != 0); }
        else if (k == "--cpuct")    { g_cpuct = (float)atof(v.c_str()); }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }

    std::printf("=== 选点一致率探针 ===\n");
    std::printf("局面     : %d 个 (随机开局 %d 手, seed=%u; 局面集与权重无关, 两次跑是同一批)\n",
                g_positions, g_opening, g_seed);
    std::printf("PPO+MCTS : %d 次模拟/步 (温度 0, 取访问最多的着法)\n", g_sims);
    std::printf("AB-Agent : 深度 %d (作为参照)\n", g_depth);
    std::printf("c_puct   : %.4f (R2 之后先验是合法集上归一后的概率, 其含义与改动前不同)\n",
                (double)g_cpuct);

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

    PPOMCTSAgent ppo(envPpo, 64, 0.99f, 0.001f, g_cpuct, 64, 0.1f, true);
    ppo.replayBatchSize = 0;   /* 只做决策, 不学习 */
    if (!g_loadPrefix.empty()) {
        const bool ok = ppo.loadModel(g_loadPrefix);
        std::printf("权重     : %s -> %s\n", g_loadPrefix.c_str(),
                    ok ? "载入成功" : "**载入失败**, 用随机初始权重");
    } else {
        std::printf("权重     : 未载入 (随机初始化)\n");
    }
    ABAgent ab(envAb, g_depth);

    /*
       R1 配对 A/B (--ab=1): 第二个 agent 用**改动前**的先验口径
       (`sparsePolicyHead=false`: 全量策略 + 先验直接用原始 p_full)。

       两个 agent 的权重、局面、模拟次数、温度**全都一样**, 唯一差别就是先验口径
       (稀疏列 + 合法集归一 vs 全量列 + 原始概率)。于是"两者选点一致率"直接量出 R1
       的语义副作用有多大 —— 而"两者 ms/步"就是在同一台机器状态下的速度比。
       只用"两次独立运行的一致率差不多"来论证是不够的: 那个指标的标准误有 3 个百分点。
    */
    Chess envDense;
    envDense.reset();
    PPOMCTSAgent ppoDense(envDense, 64, 0.99f, 0.001f, g_cpuct, 64, 0.1f, true);
    ppoDense.replayBatchSize = 0;
    ppoDense.sparsePolicyHead = false;
    if (g_ab) {
        if (!g_loadPrefix.empty()) {
            ppoDense.loadModel(g_loadPrefix);
        }
        std::printf("A/B      : 开 —— 第二个 agent 用 R1 之前的先验口径 (全量 + 原始 p_full)\n");
    }
    int modeSame = 0, modeCounted = 0;
    int denseSameAb = 0, sparseSameAb = 0;
    double denseMs = 0.0;

    /*
       ================================================================
        B-5 配对探针 (--reuse-ab=1): 子树复用到底有没有让搜索变强?
       ================================================================
       为什么不能沿用上面那套"独立局面"的探针: 那些局面互不相干, 复用在它们之间
       **必然不命中** (见 test_ppomcts 测试12 [c]), 于是探针天然对复用不敏感 ——
       拿它来量 B-5 只会得到"数字没变", 那不能说明任何问题。

       复用的收益只出现在**一盘棋之内**: 上一步落子到达的局面就在上一棵树里。
       所以这里让"复用开"的那个 agent 自己下一盘 (两边都归它走), 每 ply **同一个局面**
       再问一次"复用关"的 agent (它每 ply 从零重搜), 两边都跟 AB(深度 D) 比 ——
       同一批局面、同一份权重、同样的模拟次数, 只有"要不要复用"这一个差别。
       ================================================================
    */
    Chess envOff;
    envOff.reset();
    PPOMCTSAgent ppoOff(envOff, 64, 0.99f, 0.001f, g_cpuct, 64, 0.1f, true);
    ppoOff.replayBatchSize = 0;
    ppoOff.treeReuse = false;
    if (g_reuseAb) {
        if (!g_loadPrefix.empty()) {
            ppoOff.loadModel(g_loadPrefix);
        }
        std::printf("复用 A/B : 开 —— 同一盘棋的同一批局面上, 复用(新) vs 每 ply 从零(旧)\n");
    }

    if (g_reuseAb) {
        auto sameStep = [](const Step &x, const Step &y) {
            return x.pos.x == y.pos.x && x.pos.y == y.pos.y &&
                   x.nextPos.x == y.nextPos.x && x.nextPos.y == y.nextPos.y;
        };
        envPpo.reset();
        envPpo.sideToMove = Stone::COLOR_RED;

        const int plies = std::min(g_positions, 120);
        int reuseSameAb = 0, freshSameAb = 0, armSame = 0, n = 0;
        long long carried = 0;   /* 复用被继承的访问数总和 (= 白拿的模拟次数) */
        double reuseMs = 0.0, freshMs = 0.0;

        for (int ply = 0; ply < plies; ply++) {
            const int color = envPpo.sideToMove;

            const double t0 = nowMs();
            const Step moveReuse = ppo.selectMove(color, g_sims, 0.0f);
            reuseMs += nowMs() - t0;
            /* 根上的访问数 = 继承来的 + 本次 g_sims (每次模拟都会给根 +1) */
            if (ppo.currentRoot() >= 0 &&
                (std::size_t)ppo.currentRoot() < ppo.nodes.size()) {
                const int rootVisits = ppo.nodes[(std::size_t)ppo.currentRoot()].visitCount;
                if (rootVisits > g_sims) { carried += (long long)(rootVisits - g_sims); }
            }

            envOff = envPpo;                       /* 同一个局面给"从零重搜"那一侧 */
            const double t1 = nowMs();
            const Step moveFresh = ppoOff.selectMove(color, g_sims, 0.0f);
            freshMs += nowMs() - t1;

            envAb = envPpo;                        /* AB 必须看到同一个局面 */
            const Step moveAb = ab.getBestMove(color, g_depth);
            if (!moveReuse.valid || !moveFresh.valid || !moveAb.valid) { break; }
            n++;
            if (sameStep(moveReuse, moveAb)) { reuseSameAb++; }
            if (sameStep(moveFresh, moveAb)) { freshSameAb++; }
            if (sameStep(moveReuse, moveFresh)) { armSame++; }

            /*
               对局由 **AB 的着法推进**, 而不是由 PPO 自己推。两个理由:
                1. 让 PPO 自己下一整盘会把局面漂到"PPO 自对弈"的区域, 那里 AB 的首选与
                   PPO 的选点几乎从不重合 —— 一致率掉到 ~1.7% (地板), 什么也量不出来;
                   AB 推进则始终是正常棋局, 一致率在可分辨的区间。
                2. 复用**照样会命中**: 80 次模拟 > 开局 44 个合法着法, 所以上一棵树的
                   根**所有**孩子都被展开过, 对手(AB)走的那一步必然在里面 ⇒ 下一 ply 的
                   局面就是上一棵树深度 1 上的节点 (自对弈里是同一个道理)。
            */
            double dummy = 0.0;
            envPpo.moveForward(&moveAb, dummy);
            if (envPpo.getResult(envPpo.sideToMove) != Chess::RESULT_ONGOING) { break; }
        }

        if (n == 0) {
            std::printf("\n**没有可比对的局面**\n");
            return 1;
        }
        const double reuseRate = 100.0 * (double)reuseSameAb / (double)n;
        const double freshRate  = 100.0 * (double)freshSameAb / (double)n;
        const double armRate    = 100.0 * (double)armSame / (double)n;
        const double se = std::sqrt(reuseRate * (100.0 - reuseRate) / (double)n);
        std::printf("\n=== B-5 复用 A/B (%d 个局面 = 同一盘棋连续 %d 手) ===\n", n, n);
        std::printf("  对 AB(深度 %d) 一致率 : 复用(新) %.1f%%  vs  每 ply 从零(旧) %.1f%%"
                    "   (配对; 单侧标准误 %.1f%%)\n", g_depth, reuseRate, freshRate, se);
        std::printf("  两侧选点相同率       : %.1f%%\n", armRate);
        std::printf("  复用命中             : %lld 次 / %d ply\n", ppo.ttReuseHits(), n);
        std::printf("  继承的访问数(白拿)    : 共 %lld 次, 均 %.1f/步, 相当于每步多搜 %.0f%% 的模拟\n",
                    carried, (double)carried / (double)n,
                    100.0 * (double)carried / ((double)n * (double)g_sims));
        std::printf("  每步耗时             : 复用 %.1f ms, 从零 %.1f ms\n",
                    reuseMs / (double)n, freshMs / (double)n);
        std::printf("\n说明: 复用会**改变选点** (这是它的目的), 所以两侧相同率不会是 100%%。\n"
                    "      \"对 AB 一致率\"仍是代理指标, 单侧 SE ≈ %.1f%% —— 差异小于 2 个 SE 就别下结论。\n",
                    se);
        return 0;
    }

    int sameMove = 0;      /* (from, to) 完全一致 */
    int sameFrom = 0;      /* 起点相同 (同类意图) */
    int randSame = 0;      /* 随机走法 vs AB 的一致率 —— 机会水平 */
    double ppoMs = 0.0, abMs = 0.0, randMs = 0.0;
    int counted = 0;

    /*
       R1 诊断 (2026-09): **合法集上的策略概率质量** Z = Σ_{合法着法} p_full。

       为什么这个数决定 R1 有没有"副作用": 改动前搜索直接拿原始 p_full 当先验, 而
       改动后的稀疏路径给的是"在合法集上归一化"的概率 p_full/Z —— 两者相差一个**全局
       因子 1/Z**。因为 PUCT 的探索项是 c_puct * P * sqrt(N)/(1+n), 先验整体放大 1/Z
       就等于把 c_puct 乘了 1/Z, 所以 Z 必须被量出来:
         * Z ≈ 1 (训练过的权重应当如此: 交叉熵会把概率质量压到合法着法上) -> 这条差异
           可以忽略, 选点不变;
         * Z 明显 < 1 (随机初始化的权重, 质量摊在 8100 个槽位上) -> 先验被放大一个
           数量级, 搜索更偏向先验 (未训练时的行为会变, 但那时策略本身也是均匀的)。
       **必须在 Steps::instance().put(legal) 之前算** —— put 之后这些 Step* 会被回收复用。
    */
    double zMin = 1e9, zMax = -1e9, zSum = 0.0;
    int zCount = 0;

    for (int i = 0; i < g_positions; i++) {
        Chess pos;
        makePosition(i, pos);
        const int color = pos.sideToMove;
        std::vector<Step *> legal;
        pos.sample(color, legal);
        if (legal.empty()) { continue; }
        const std::size_t nLegal = legal.size();

        /* --- Z 诊断 (用**全量**策略, 即 R1 改动前的口径) --- */
        envPpo = pos;
        {
            RL::Tensor zState(PPOMCTSAgent::STATE_DIM, 1);
            zState.zero();
            ppo.encodeState(zState);
            RL::Tensor zPolicy = ppo.ppo.action(zState);   /* 深拷贝 */
            double z = 0.0;
            for (std::size_t k = 0; k < nLegal; k++) {
                const int a = ppo.stepToActionIdx(*legal[k], color);
                if (a >= 0 && a < PPOMCTSAgent::ACTION_DIM) {
                    z += (double)zPolicy[(std::size_t)a];
                }
            }
            zMin = std::min(zMin, z);
            zMax = std::max(zMax, z);
            zSum += z;
            zCount++;
        }

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

        /* --- R1 配对 A/B: 同一个局面, 只换先验口径再搜一次 --- */
        if (g_ab) {
            envDense = pos;
            const double t2 = nowMs();
            const Step denseMove = ppoDense.selectMove(color, g_sims, 0.0f);
            denseMs += nowMs() - t2;
            if (denseMove.valid) {
                modeCounted++;
                const bool same2 = (denseMove.pos.x == ppoMove.pos.x
                                    && denseMove.pos.y == ppoMove.pos.y
                                    && denseMove.nextPos.x == ppoMove.nextPos.x
                                    && denseMove.nextPos.y == ppoMove.nextPos.y);
                if (same2) { modeSame++; }

                /*
                   两个口径**各自**与 AB 的一致率 —— 同一进程、同一批局面、配对比较。
                   这才是"R1 有没有把代理指标带坏"的直接读数: 两次独立运行的聚合一致率
                   差 1 个百分点, 而它的标准误就有 3 个百分点, 什么都说明不了。
                */
                const bool denseVsAb = (denseMove.pos.x == abMove.pos.x
                                        && denseMove.pos.y == abMove.pos.y
                                        && denseMove.nextPos.x == abMove.nextPos.x
                                        && denseMove.nextPos.y == abMove.nextPos.y);
                if (denseVsAb) { denseSameAb++; }
                if (same) { sparseSameAb++; }
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
    if (zCount > 0) {
        std::printf("  合法集概率质量 Z: 最小 %.4f, 均 %.4f, 最大 %.4f"
                    "  (R1 后先验口径 = p_full/Z, 相对改动前放大 1/Z 倍)\n",
                    zMin, zSum / (double)zCount, zMax);
    }
    if (g_ab && modeCounted > 0) {
        const double modeAgree = 100.0 * (double)modeSame / (double)modeCounted;
        const double modeSe = std::sqrt(modeAgree * (100.0 - modeAgree) / (double)modeCounted);
        const double sparseMs = ppoMs / (double)counted;
        const double denseAvgMs = denseMs / (double)modeCounted;
        std::printf("  R1 A/B (同权重/同局面, 只换先验口径):\n");
        std::printf("    选点一致率   : %5.1f%% (标准误 %.1f%%) —— 这就是 R1 的语义副作用大小\n",
                    modeAgree, modeSe);
        std::printf("    对 AB 一致率 : 稀疏 %.1f%% vs 全量 %.1f%% (同一批局面, 配对) "
                    "—— 代理指标有没有被带坏\n",
                    100.0 * (double)sparseSameAb / (double)modeCounted,
                    100.0 * (double)denseSameAb / (double)modeCounted);
        std::printf("    每步耗时     : 稀疏(新) %.1f ms  vs  全量(旧) %.1f ms  ->  %.2fx\n",
                    sparseMs, denseAvgMs, (sparseMs > 0.0) ? denseAvgMs / sparseMs : 0.0);
    }
    std::printf("\n说明: 这是**代理指标**(策略是否朝更强玩家靠), 不是棋力; 结论仍要对局来定。\n"
                "      配对比较的用法: 同一命令只换 --load, 比较两行\"完整着法一致率\"。\n");
    return 0;
}
