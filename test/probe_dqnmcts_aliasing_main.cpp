/*
 * probe_dqnmcts_aliasing_main.cpp - DQN+MCTS 的**表示能力**探针 (不进 ctest)
 * ============================================================================
 *
 * 为什么需要这个探针
 * ------------------
 * `DQNMCTSAgent` 报告出来的训练数字 (50 胜 0 负 50 和 / DQN 损失 22) 无法回答
 * "它到底学没学会下棋"。而这个 agent 的**编码层**有几条硬约束会让它**根本学不到**,
 * 与训练多久、奖励怎么设计无关:
 *
 *   * 动作空间: `STEP_TO_ACTION_IDX` 把 (棋子 id, 目标格) 哈希进 **128** 个槽位,
 *     而真实走法空间是 16 子 x 90 格 = 1440 (静态) / 全盘 8100。多个**不同**走法
 *     落到同一个下标 ⇒ Q(s,a) 无法区分它们。
 *   * 状态空间: 90 维 = 10x9 个"该格有什么子"(归一化后的类型值), **不含轮到谁**,
 *     也不含任何规则历史 (重复次数 / 无吃子进度 / 将军)。于是三次重复、长将循环、
 *     60 回合限着这些**决定终局与回报**的规则在状态里完全不可观测。
 *
 * 这个探针把这两件事量成数字:
 *
 *   [1] 别名 (aliasing): 在若干真实局面上枚举全部**合法**走法, 统计它们落到几个
 *       动作下标上、平均每个下标要背多少个互不相同的走法、以及桶大小的分布。
 *   [2] 状态不可分性: 构造几组"规则状态不同、棋子位置相同"的局面 (缺子 / 换走子方 /
 *       被将军 / 重复进度不同), 检查 `encodeState` 的输出是否**逐字节相同**;
 *       并给出 DQNAB/PPO 用的 19 平面口径在这些组上是否可分。
 *   [3] 动作空间上界: 1440 与 128 的比值 (静态上界), 以及全盘 8100 与 128 的比值。
 *
 * 用法:
 *   cmake --build <build> --target probe_dqnmcts_aliasing
 *   <build>/probe_dqnmcts_aliasing [--positions=32] [--plies=20] [--seed=20240901]
 *
 * 判读:
 *   * [1] 的"平均每桶走法数"远大于 1 ⇒ 策略/价值函数在动作维度上是**共享**的,
 *     它不是"给每个走法一个值", 而是"给 128 个类别各一个值"。这种共享下不可能
 *     收敛到正确的棋。
 *   * [2] 里任何一组输出"完全相同" ⇒ 该规则在该编码下**不可辨识**, 与该规则有关的
 *     回报它学不到 (不是学得慢, 是学不到)。
 *   * 这两条都**不依赖 Qt / 权重 / 训练量**, 所以可以当"这个 agent 值不值得继续训"的
 *     前置闸门。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "aiagent.h"
#include "chess.h"
#include "chessstate.h"
#include "dqnmcts_agent.h"

namespace {

struct Cfg {
    int positions = 32;      /* 采多少个局面 */
    int plies = 20;          /* 每个局面由随机走子走多少手得到 (0 = 标准开局) */
    unsigned seed = 20240901u;
};

Cfg g_cfg;

/* ------------------------------------------------------------------
 *  与 DQNMCTSAgent::stepToActionIdx 逐字一致的哈希
 *
 *  刻意**重新实现一份**而不是调用成员函数: 探针要能在"那个实现改了"的时候
 *  报出不一致 (见 [4]), 而如果直接调用就永远自洽。
 * ------------------------------------------------------------------ */
int probeActionIdx(const Step &s, int actionDim)
{
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)actionDim);
}

/* 把 encodeState 的输出压成一个可比较的字节串 (逐位比较用) */
std::string fingerprint(const RL::Tensor &t)
{
    std::string out;
    out.reserve(t.size() * 4);
    for (std::size_t i = 0; i < t.size(); i++) {
        const float v = t[i];
        const unsigned char *p = reinterpret_cast<const unsigned char *>(&v);
        out.append(reinterpret_cast<const char *>(p), sizeof(float));
    }
    return out;
}

/*
 * 单个局面上的"动作别名"统计。
 *
 * 为什么要**按局面**算而不是把所有局面的走法混在一起: 混着算会把分母做大, 于是
 * 碰撞率看起来很小 —— 但 Q(s,a) 的别名是**同一个 s 内**两两动作共享输出槽位,
 * 不同 s 之间共享同一个槽位本来就是网络该做的泛化 (那不是缺陷)。判据必须是
 * "给定这个局面, 随机挑两个合法走法, 它们撞到同一个槽位的概率"。
 */
struct AliasStat {
    int legal = 0;          /* 本局面合法走法数 */
    int slotsUsed = 0;      /* 被用到的槽位数 */
    int maxPerSlot = 0;     /* 最挤的那个槽位里有几个不同走法 */
    long long sumSq = 0;    /* Σ n_i^2 (只算非空槽位), 用来算碰撞率 */
    double collide() const {
        if (legal <= 0) { return 0.0; }
        return (double)sumSq / ((double)legal * (double)legal);
    }
    int aliasedMoves() const { return legal - slotsUsed; }   /* 被挤掉的走法数 */
};

AliasStat computeAlias(const std::vector<Step *> &legal, int actionDim)
{
    std::map<int, std::set<std::string> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        const int a = probeActionIdx(s, actionDim);
        char key[64];
        std::snprintf(key, sizeof(key), "%d->%d,%d", s.id, s.nextPos.x, s.nextPos.y);
        bucket[a].insert(std::string(key));
    }
    AliasStat st;
    st.legal = (int)legal.size();
    st.slotsUsed = (int)bucket.size();
    for (std::map<int, std::set<std::string> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        const int n = (int)it->second.size();
        st.maxPerSlot = std::max(st.maxPerSlot, n);
        st.sumSq += (long long)n * (long long)n;
    }
    return st;
}

/* 只保留非零项的可读指纹: 90 维里绝大多数是 0, 直接看"哪些格有子、是什么" */
std::string sparseFingerprint(const RL::Tensor &t)
{
    char buf[64];
    std::string out;
    for (std::size_t i = 0; i < t.size(); i++) {
        if (t[i] == 0.0f) { continue; }
        std::snprintf(buf, sizeof(buf), "%d:%.3f ", (int)i, (double)t[i]);
        out += buf;
    }
    return out;
}

/*
 * 用一个**最便宜的探针 agent** 调 encodeState。
 *
 * 为什么不用 `Chess::encodeState`: 编码是 agent 的成员 (每个 agent 的平面布局不同),
 * 棋盘本身不提供它。而"那个 agent 自己的编码函数"正是本探针要量的东西 —— 用别的
 * 实现代替就等于测了另一件事。
 *
 * hidden=1 只为压内存: RL::DQN 的构造会按 hiddenDim 建网络, 探针只借它的编码方法,
 * 一次前向都不做。状态量本身与网络宽度无关 (只取决于棋盘)。
 */
DQNMCTSAgent makeProbeAgent(Chess &c)
{
    return DQNMCTSAgent(c, /*hiddenDim=*/1, 0.99f, 0.001f, 1.0f, 1.414f);
}

/*
 * 把 DQNMCTS 的编码函数作用在 `board` 上。
 *
 * **必须**把 agent 内部的棋盘引用指到 `board` —— 探针刻意构造"同一棋子分布、不同
 * 规则状态"的多个棋盘副本, 而编码函数读的是 agent 持有的那一份。
 */
void encodeProbe(DQNMCTSAgent &ag, Chess &board, RL::Tensor &out)
{
    ag.chess = board;
    ag.encodeState(out);
}

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
        if (k == "--positions") { g_cfg.positions = std::atoi(v.c_str()); }
        else if (k == "--plies") { g_cfg.plies = std::atoi(v.c_str()); }
        else if (k == "--seed") { g_cfg.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else {
            std::printf(
                "用法: probe_dqnmcts_aliasing [--positions=N] [--plies=N] [--seed=N]\n"
                "  --positions : 采多少个局面 (默认 32)\n"
                "  --plies     : 每个局面由随机合法走子走多少手 (默认 20; 0 = 标准开局)\n");
            return false;
        }
    }
    return true;
}

/* 从标准开局随机走 plies 手合法棋, 得到一个"真实"局面。撞上终局就重来。 */
bool makePosition(Chess &c, int plies, std::mt19937_64 &rng)
{
    for (int attempt = 0; attempt < 16; attempt++) {
        c.reset();
        bool ok = true;
        for (int i = 0; i < plies; i++) {
            std::vector<Step *> steps;
            c.sample(c.sideToMove, steps);
            if (steps.empty()) { Steps::instance().put(steps); ok = false; break; }
            std::uniform_int_distribution<int> pick(0, (int)steps.size() - 1);
            const Step s = *steps[(std::size_t)pick(rng)];
            Steps::instance().put(steps);
            double dummy = 0.0;
            c.moveForward(&s, dummy);
            if (c.getResult(c.sideToMove) != Chess::RESULT_ONGOING) { ok = false; break; }
        }
        if (ok) { return true; }
    }
    c.reset();
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv)) { return 0; }

    std::printf("=== probe_dqnmcts_aliasing: DQN+MCTS 表示能力探针 ===\n");
    std::printf("  STATE_DIM = %d, ACTION_DIM = %d\n",
                DQNMCTSAgent::STATE_DIM, DQNMCTSAgent::ACTION_DIM);
    std::printf("  采 %d 个局面 x 随机走 %d 手 (seed=%u)\n\n",
                g_cfg.positions, g_cfg.plies, g_cfg.seed);

    std::mt19937_64 rng(g_cfg.seed);

    /* ================================================================
     *  [1] 动作别名: 合法走法 -> 128 个下标, **同一个局面内**有多少走法被迫共槽
     * ================================================================ */
    {
        std::map<int, int> bucketMoves;      /* 下标 -> 落进它的走法数 (跨局面累计) */
        std::map<int, std::set<std::string> > bucketDistinct;  /* 下标 -> 互不相同的走法 */
        long long totalMoves = 0;
        long long positionsUsed = 0;
        int minLegal = 1 << 30, maxLegal = 0;

        /* 按局面统计 (判据见 AliasStat 的说明) */
        double collideSum = 0.0, collideMax = 0.0, collideMin = 1.0;
        int collidePerfect = 0;        /* 碰撞率 = 1.0 的局面数 (所有走法共用一个槽) */
        long long maxPerSlotPos = 0;   /* 单个局面内"最挤槽位"的最大值 */
        int everAliased = 0;           /* 至少有一对走法撞槽的局面数 */
        long long aliasedMovesSum = 0;

        Chess c;
        for (int p = 0; p < g_cfg.positions; p++) {
            makePosition(c, g_cfg.plies, rng);
            /* 双方各枚举一次: 同一局面换走子方, 合法集不同 */
            for (int color = 0; color < 2; color++) {
                std::vector<Step *> legal;
                c.sample(color, legal);
                if (legal.empty()) { continue; }
                positionsUsed++;
                const int n = (int)legal.size();
                minLegal = std::min(minLegal, n);
                maxLegal = std::max(maxLegal, n);
                for (std::size_t i = 0; i < legal.size(); i++) {
                    const Step &s = *legal[i];
                    const int a = probeActionIdx(s, DQNMCTSAgent::ACTION_DIM);
                    bucketMoves[a]++;
                    char key[64];
                    std::snprintf(key, sizeof(key), "%d->%d,%d", s.id, s.nextPos.x, s.nextPos.y);
                    bucketDistinct[a].insert(std::string(key));
                    totalMoves++;
                }

                const AliasStat st = computeAlias(legal, DQNMCTSAgent::ACTION_DIM);
                collideSum += st.collide();
                collideMax = std::max(collideMax, st.collide());
                collideMin = std::min(collideMin, st.collide());
                if (st.collide() >= 0.999) { collidePerfect++; }
                if (st.aliasedMoves() > 0) { everAliased++; }
                aliasedMovesSum += st.aliasedMoves();
                maxPerSlotPos = std::max(maxPerSlotPos, (long long)st.maxPerSlot);

                Steps::instance().put(legal);
            }
        }

        /* 动作空间上界: 静态 (16 子 x 90 格) 与全盘 (90x90) */
        const double staticUpper = 16.0 * 90.0;
        const double fullUpper = 90.0 * 90.0;

        int usedBuckets = (int)bucketMoves.size();
        int maxBucket = 0, minBucket = 1 << 30;
        for (std::map<int, int>::const_iterator it = bucketMoves.begin();
             it != bucketMoves.end(); ++it) {
            maxBucket = std::max(maxBucket, it->second);
            minBucket = std::min(minBucket, it->second);
        }

        std::printf("--- [1] 动作别名 (真实合法走法 -> %d 个动作下标) ---\n",
                    DQNMCTSAgent::ACTION_DIM);
        std::printf("  局面 x 走子方 : %lld 次枚举\n", positionsUsed);
        std::printf("  合法走法总数  : %lld, 合法走法/局面 最少 %d 最多 %d\n",
                    totalMoves, minLegal, maxLegal);
        std::printf("  被用到的下标  : %d / %d (跨局面累计; 桶大小 最小 %d 最大 %d)\n",
                    usedBuckets, DQNMCTSAgent::ACTION_DIM, minBucket, maxBucket);
        if (positionsUsed > 0) {
            /*
               这里的碰撞率是**同局面内**的: P(从本局面的合法集里随机挑两个走法,
               它们落在同一个动作下标上) = Σ n_i^2 / n^2 (n = 合法走法数)。
               不同局面之间的共享不算缺陷 (那是网络该做的泛化), 所以不混着算。
            */
            const double denom = (double)positionsUsed;
            std::printf("\n  同局面内碰撞率 (这是判据, 见 AliasStat 的说明):\n");
            std::printf("    平均 %.4f | 最小 %.4f | 最大 %.4f\n",
                        collideSum / denom, collideMin, collideMax);
            std::printf("    碰撞率 >= 0.999 的局面 : %d / %lld  (全部走法挤在同一批槽位)\n",
                        collidePerfect, positionsUsed);
            std::printf("    至少有一对走法撞槽    : %d / %lld (%.1f%%)\n",
                        everAliased, positionsUsed,
                        100.0 * (double)everAliased / denom);
            std::printf("    平均被挤掉的走法数    : %.2f (合法集里有多少走法拿不到自己的槽位)\n",
                        (double)aliasedMovesSum / denom);
            std::printf("    单局面内最挤槽位      : %lld 个不同走法共用一个下标\n",
                        maxPerSlotPos);
        }
        std::printf("\n  动作空间上界  : 静态 16x90=%.0f, 全盘 90x90=%.0f, 实际槽位 %d\n",
                    staticUpper, fullUpper, DQNMCTSAgent::ACTION_DIM);
        std::printf("  => 静态走法空间 / 槽位 = %.1f 倍, 全盘走法空间 / 槽位 = %.1f 倍\n\n",
                    staticUpper / (double)DQNMCTSAgent::ACTION_DIM,
                    fullUpper / (double)DQNMCTSAgent::ACTION_DIM);
    }

    /* ================================================================
     *  [2] 状态不可分性: 棋子分布相同、规则状态不同的局面, 编码是否相同
     *
     *  90 维编码 = "每格有什么子"(归一化后的类型值)。所以它**必然**对下列量不可分,
     *  因为它们不在"棋子位置"里。这里把它们逐条量出来 (而不是靠读代码断言)。
     * ================================================================ */
    {
        Chess base;
        base.reset();
        DQNMCTSAgent ag = makeProbeAgent(base);

        std::printf("--- [2] 状态不可分性 (encodeState 是否区分规则状态) ---\n");

        RL::Tensor a(DQNMCTSAgent::STATE_DIM, 1), b(DQNMCTSAgent::STATE_DIM, 1);

        /* (a) 走子方: 同一棋子分布, 轮到红 / 轮到黑
               —— 这一步抵消 "轮到谁" 整条信息, 是最大的一条: 中国象棋先手优势很大,
               而同一个局面在红走与黑走下的最优着法完全不同。 */
        {
            Chess red = base, black = base;
            red.sideToMove = Stone::COLOR_RED;
            black.sideToMove = Stone::COLOR_BLACK;
            encodeProbe(ag, red, a);
            encodeProbe(ag, black, b);
            const bool same = (fingerprint(a) == fingerprint(b));
            std::printf("  (a) 同一棋子分布, 轮红 vs 轮黑        : 编码 %s\n",
                        same ? "**完全相同 (不可分)**" : "不同 (可分)");
        }

        /* (b) 重复局面进度: 同一个局面, 已出现 1 次 vs 已出现 3 次 (判和)
               —— 直接构造历史栈 (与真实对局等价的判定路径: isRepetition 按
               halfMoveClock 窗口数当前 hash 的出现次数)。 */
        {
            Chess r1 = base, r3 = base;
            r1.halfMoveClock = 120;
            r3.halfMoveClock = 120;
            const unsigned long long h = r3.computeHash();
            r3.history.push_back(Chess::HistoryRecord{ h, 0 });
            r3.history.push_back(Chess::HistoryRecord{ h, 0 });
            r3.history.push_back(Chess::HistoryRecord{ h, 0 });

            const bool rep1 = r1.isRepetition();
            const bool rep3 = r3.isRepetition();
            Chess::DrawReason dr1 = Chess::DRAW_NONE, dr3 = Chess::DRAW_NONE;
            const int res1 = r1.getResult(r1.sideToMove, &dr1);
            const int res3 = r3.getResult(r3.sideToMove, &dr3);
            encodeProbe(ag, r1, a);
            encodeProbe(ag, r3, b);
            const bool same = (fingerprint(a) == fingerprint(b));
            std::printf("  (b) 同一局面, 重复 1 次 vs 3 次       : 编码 %s "
                        "(isRepetition %d -> %d; getResult %d -> %d, drawReason %d -> %d)\n",
                        same ? "**完全相同 (不可分)**" : "不同 (可分)",
                        (int)rep1, (int)rep3, res1, res3, (int)dr1, (int)dr3);
        }

        /* (c) 自然限着: halfMoveClock = 0 vs 120 (已判和) */
        {
            Chess f0 = base, f120 = base;
            f0.halfMoveClock = 0;
            f120.halfMoveClock = 120;
            Chess::DrawReason dr = Chess::DRAW_NONE;
            const int res = f120.getResult(f120.sideToMove, &dr);
            encodeProbe(ag, f0, a);
            encodeProbe(ag, f120, b);
            const bool same = (fingerprint(a) == fingerprint(b));
            std::printf("  (c) 同一局面, 无吃子 0 手 vs 120 手   : 编码 %s "
                        "(120 手 getResult=%d, drawReason=%d)\n",
                        same ? "**完全相同 (不可分)**" : "不同 (可分)",
                        res, (int)dr);
        }

        /* (d) 终局检测口径: evaluateLeaf 用的是 isGameOver(), 而它只看"将帅是否
               还在场上"。于是判和与"将杀"都**不是终局**, 叶子值继续走 max_a Q(s,a)。 */
        {
            Chess d1 = base;
            d1.halfMoveClock = 120;
            const unsigned long long h = d1.computeHash();
            d1.history.push_back(Chess::HistoryRecord{ h, 0 });
            d1.history.push_back(Chess::HistoryRecord{ h, 0 });
            d1.history.push_back(Chess::HistoryRecord{ h, 0 });
            const int viaResult = (int)d1.getResult(d1.sideToMove);
            const int viaGameOver = (int)d1.isGameOver();   /* evaluateLeaf 的口径 */
            std::printf("  (d) 已判和的局面                      : isGameOver()=%d "
                        "(COLOR_NONE=%d) vs getResult()=%d (RESULT_DRAW=%d)  %s\n",
                        viaGameOver, (int)Stone::COLOR_NONE, viaResult,
                        (int)Chess::RESULT_DRAW,
                        (viaGameOver == (int)Stone::COLOR_NONE
                         && viaResult == (int)Chess::RESULT_DRAW)
                            ? "<== 判和不会被 evaluateLeaf 当成终局" : "");
        }

        /* (e) 对照: 带规则上下文的编码口径有这 5 个通道 */
        std::printf("\n  对照: chessstate.h 的 5 个规则上下文平面 =\n");
        std::printf("    CTX_MATERIAL=%d CTX_TEMPO=%d CTX_HALFMOVE=%d CTX_REPEAT=%d "
                    "CTX_CHECK=%d\n",
                    ChessState::CTX_MATERIAL, ChessState::CTX_TEMPO,
                    ChessState::CTX_HALFMOVE, ChessState::CTX_REPEAT,
                    ChessState::CTX_CHECK);
        std::printf("    (PPO 用 19 平面 = 14 棋子 + 5 上下文; EVAB/SACAZ 用 17)\n");
        std::printf("    DQNMCTS 的 90 维里这些量的通道数是 0 —— 上面 (a)(b)(c) 的\n");
        std::printf("    不可辨识不是\"实现细节\", 而是**缺少通道**。\n\n");
    }

    /* ================================================================
     *  [3] 终局奖励通道: 自对弈里有多少局**真的**走到了终局?
     *
     *  为什么要量它: `DQNMCTSAgent::trainSelfPlay` 收尾用
     *      int gameResult = chess.isGameOver();      // 只认"将/帅还在不在场"
     *      bool done = (gameResult != COLOR_NONE);
     *  于是三次重复 / 60 回合自然限着 / 将杀 / 困毙**都不是终局** —— 那一手照常
     *  写成 (s,a,r,s',done=false), Q 目标 = r(材质) + gamma*maxQ, 一路自举下去。
     *
     *  这个循环按**同一套策略**下几局, 用 `getResult()` (可依赖的裁判) 当 oracle
     *  逐手检查, 数出每局的结束原因, 以及 isGameOver() 会不会看见它。
     * ================================================================ */
    {
        const int games = 6;
        const int maxPlies = 200;
        const int sims = 8;   /* 与 [1] 同一考虑: 量结构, 不量棋力 */

        Chess board;
        board.reset();
        DQNMCTSAgent ag = makeProbeAgent(board);

        int endedBy = 0, endedResult = 0, endedCap = 0;   /* 谁先看见终局 */
        std::map<int, int> reasonHist;                    /* 结束码 -> 局数 */

        std::printf("--- [3] 终局奖励通道 (自对弈 %d 局 x %d 手上限) ---\n", games, maxPlies);

        for (int g = 0; g < games; g++) {
            board.reset();
            int color = Stone::COLOR_RED;
            bool done = false;
            for (int ply = 0; ply < maxPlies && !done; ply++) {
                /* trainSelfPlay 的循环判据就是 isGameOver() —— 先照抄它的行为 */
                const int gameResult = board.isGameOver();
                if (gameResult != Stone::COLOR_NONE) {
                    endedBy++;
                    reasonHist[gameResult == Stone::COLOR_RED ? 1 : 2]++;
                    done = true;
                    break;
                }

                const Step mv = ag.selectMove(color, sims, false);
                if (!mv.valid) {
                    /* 没有合法走法 = 将杀/困毙: trainSelfPlay 也是这么处理的 */
                    endedBy++;
                    reasonHist[3]++;
                    done = true;
                    break;
                }
                double dummy = 0.0;
                Step copy = mv;
                board.moveForward(&copy, dummy);
                color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;

                /* 可依赖的裁判: 这一手之后真的终局了吗? */
                Chess::DrawReason dr = Chess::DRAW_NONE;
                const int res = board.getResult(color, &dr);
                if (res != Chess::RESULT_ONGOING) {
                    endedResult++;
                    reasonHist[res]++;
                    done = true;
                }
            }
            if (!done) { endedCap++; reasonHist[0]++; }   /* 撞手数上限 */
        }

        std::printf("  被 isGameOver() 看见的终局 : %d / %d 局\n", endedBy, games);
        std::printf("  被 getResult()  看见的终局 : %d / %d 局\n", endedResult, games);
        std::printf("  撞 %d 手上限而\"没终局\"   : %d / %d 局\n", maxPlies, endedCap, games);
        std::printf("  逐局结束码 (0=截断,1=红胜,2=黑胜,3=和):");
        for (std::map<int, int>::const_iterator it = reasonHist.begin();
             it != reasonHist.end(); ++it) {
            std::printf(" %d x%d", it->first, it->second);
        }
        std::printf("\n");
        if (endedResult > endedBy) {
            std::printf("  => **%d 局**的终局在训练循环里被漏掉了: 那些手的 done 恒为 false,\n",
                        endedResult - endedBy);
            std::printf("     终局 +-1/0 永远不会写进 Q 目标, 只留下 r(材质) + gamma*maxQ 的自举。\n");
        } else if (endedResult == 0 && endedCap == games) {
            std::printf("  => 这一批**全部**撞在上限上: 训练里根本没有终局信号, Q 的目标\n");
            std::printf("     恒为 r(材质) + gamma*maxQ ⇒ 学到的是\"舍不得丢子\", 不是\"怎么赢\"。\n");
        }
        std::printf("  (对照: DQNABAgent / PPOMCTSAgent 的收尾统一走 getResult(); 见\n");
        std::printf("   docs/rl_plan_optimized.md 的 Phase 6 \"终局口径统一\"。)\n\n");
    }

    /* ================================================================
     *  [4] 回放池容量: 20000 个槽位里能装多少**互不相同**的局面?
     *
     *  问题背景 (2026-09): "把 DQN 的 replay 从 4096 提到 20000 能提升收敛速度吗?"
     *
     *  容量本身有两个已知的代价面 —— 随机采样更接近 i.i.d. (容量越大越像独立),
     *  但样本更陈旧 (与当前策略/目标网的偏离更大)。而**是否值得**还要先过一个更
     *  朴素的门槛: 池子里装的是不是**新信息**。若同一批局面在池里反复出现很多份,
     *  扩大容量只是把重复存得更多份, 对"每单位算力获得的梯度信号"没有贡献。
     *
     *  这一段就在量这个: 按固定策略往下走, 记录每一步的局面指纹 (棋盘 hash +
     *  走子方 + 编码指纹), 数出 N 个样本里有多少个是互不相同的。用固定策略是
     *  **刻意的保守假设**: 不做学习 ⇒ 局面分布比真实训练更重复, 所以这里量到的
     *  "去重率"是**上界乐观**读数的反面 —— 真实训练里策略在变、局面会更散一些,
     *  但同一个编码下"标准开局 + 前几手"这一类的重复是结构性的, 不会因为策略变而消失。
     * ================================================================ */
    {
        const int games = 12;
        const int sims = 12;        /* 量分布, 不量棋力 (与 [1] 同一考虑) */
        const int maxPlies = 120;

        Chess board;
        board.reset();
        DQNMCTSAgent ag = makeProbeAgent(board);

        std::set<unsigned long long> distinct;
        long long total = 0;
        std::vector<long long> uniqueAt;   /* 每个里程碑上的去重数 */

        const long long milestones[] = { 500, 1000, 2000, 4096, 8192, 12000, 20000 };
        const int nMilestones = (int)(sizeof(milestones) / sizeof(milestones[0]));
        std::vector<long long> uniqPerMilestone(nMilestones, -1);

        std::printf("--- [4] 回放池: %d 局 x %d 手上限, 量\"新信息\"有多少 ---\n",
                    games, maxPlies);

        int next = 0;
        for (int g = 0; g < games; g++) {
            board.reset();
            int color = Stone::COLOR_RED;
            for (int ply = 0; ply < maxPlies; ply++) {
                if (board.getResult(color) != Chess::RESULT_ONGOING) { break; }
                const Step mv = ag.selectMove(color, sims, false);
                if (!mv.valid) { break; }

                /* 局面指纹: 棋盘 hash 已含走棋方 (见 Chess::computeHash 的说明) */
                const unsigned long long h = board.computeHash();
                distinct.insert(h);
                total++;

                while (next < nMilestones && total >= milestones[next]) {
                    uniqPerMilestone[next] = (long long)distinct.size();
                    next++;
                }

                double dummy = 0.0;
                Step copy = mv;
                board.moveForward(&copy, dummy);
                color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            }
        }

        std::printf("  走了 %lld 个 ply, 其中**互不相同**的局面 %lld 个 (去重率 %.1f%%)\n",
                    total, (long long)distinct.size(),
                    total > 0 ? 100.0 * (double)distinct.size() / (double)total : 0.0);
        for (int k = 0; k < nMilestones; k++) {
            if (uniqPerMilestone[k] < 0) { continue; }
            std::printf("    前 %5lld 个样本 -> %5lld 个互不相同的局面 (%.1f%%)\n",
                        milestones[k], uniqPerMilestone[k],
                        100.0 * (double)uniqPerMilestone[k] / (double)milestones[k]);
        }

        /*
           回放池的两个结构性事实 (与上面实测无关, 只看代码/参数):
             * `Transition` = state 90 + action 128 + nextState 90 = 308 float
               ≈ 1.23 KB/条 (legalMask/nextLegalMask 在本 agent 上是空张量)。
             * 淘汰策略 (rl/dqn.cpp:213) 是 `memories.size() > maxMemorySize + batchSize`
               时才丢最老的 batchSize 条 ⇒ 池子会**稳定停在 maxMemorySize 附近**,
               也就是 4096 条 ≈ 68 局 x 60 ply 才换一遍血。
        */
        std::printf("  存储: 约 1.23 KB/条 -> 4096 条约 5.0 MB, 20000 条约 24.6 MB (可忽略)\n");
        std::printf("  换血: 4096 条 x 60 ply/局 ≈ 68 局才淘汰一轮; 20000 条约 334 局\n");
        std::printf("  判读: 若上面的去重率明显低于 100%%, 扩大容量主要是**多存重复**, \n");
        std::printf("        收敛速度的瓶颈在\"梯度步数 / 每步信息量\", 不在容量。\n\n");
    }

    /* ================================================================
     *  [5] 结论行
     * ================================================================ */
    std::printf("--- [5] 判读 ---\n");
    std::printf("  * [1] 若\"平均被挤掉的走法数\"大于 0: 同一局面里若干个互不相同的走法\n");
    std::printf("    共用一个 Q 槽位 —— 它们拿不到各自的值, 只能共享一个数。\n");
    std::printf("  * [2](a)(b)(c) 若逐字节相同: 走子方 / 重复进度 / 自然限着在该编码下\n");
    std::printf("    全部不可观测。它们**决定终局与回报**, 却读不到 ⇒ 与这些规则相关\n");
    std::printf("    的回报学不到 (不是学得慢)。\n");
    std::printf("  * [2](d): evaluateLeaf 用 isGameOver() 判终局, 而它只认\"将帅还在不在\"\n");
    std::printf("    ⇒ 判和 (三次重复 / 60 回合) 与\"将杀\"都不是终局, 叶子值继续走\n");
    std::printf("    max_a Q(s,a); 而在 maxMoves 处被截断的那一手 Q = r(材质) + γmaxQ ——\n");
    std::printf("    Q 学的是**材质偏好**, 不是终局结果。\n");
    std::printf("  * [3] 上面这段循环量的是**修改前**的收尾口径 (照抄 isGameOver())。\n");
    std::printf("    2026-09 已把三处收尾统一到 getResult()+outcomeForMover():\n");
    std::printf("      trainVsRandom / trainSelfPlay / recordExperience\n");
    std::printf("    其中 trainVsRandom 原来还有第二个错: 没有\"非终局就用即时奖励\"那一支,\n");
    std::printf("    于是每一手的奖励都被写成 -1.0f (\"每走一步 = 输一盘\")。\n");
    std::printf("  * **仍未统一的一处**: `evaluateLeaf` (搜索叶子) 依旧用 isGameOver() 判终局\n");
    std::printf("    ⇒ 树内将杀/判和仍按 max_a Q(s,a) 估值, 而 Q 本身是在 90 维、无走子方\n");
    std::printf("    信息的状态上训出来的。这一处要跟 [2] 的编码问题一起解决才有意义。\n");
    std::printf("  * [1][2] 是**表示层**的限制 (无走子方 / 无规则历史 / 动作共享槽位),\n");
    std::printf("    [3] 是**代码层**的错配 (已修)。前者不解决, 修 [3] 也拿不到可用的棋力。\n");
    std::printf("  * 判据提醒: 这些数字都与\"棋力\"无关。棋力要看带置信区间的锚点评测\n");
    std::printf("    (bench_anchor)。实测 weights/ppo_bc_d4 + 24 sims 对 AB 深度 4:\n");
    std::printf("    24 局 0 胜 1 和 23 负 (Elo 差 -669) —— 与\"自对弈 50 胜\"并不矛盾:\n");
    std::printf("    自对弈里的\"胜\"只是同一个网络赢了自己。\n");
    std::printf("  * [4] 回答的是\"回放池该不该加大\": 先看容量里装的是不是新信息;\n");
    std::printf("    在 [1][2] 的表示问题解决之前, 容量是**下游**参数, 不该先动。\n");
    return 0;
}
