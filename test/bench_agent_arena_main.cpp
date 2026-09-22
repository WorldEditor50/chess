/*
 * bench_agent_arena_main.cpp - 受控对局竞技场 (SAC+AZ / PPO+MCTS / MCTS / AB)
 * ============================================================================
 *
 * 为什么要新写一个 (而不是改 bench_sacaz_vs_ab / bench_ppo_vs_ab):
 *   那两个各自只测"一个 agent 对 AB", 而"哪个算法更强"是个**三方**问题:
 *   两个学习 agent 之间、以及各自对同一个无学习基线 (MCTS) 的比分, 只有放在
 *   **同一套协议、同一个随机种子、同一份权重**下才可比。这个程序把三方装进一个
 *   二进制: `--a=` / `--b=` 各选一个 agent, 其余 (随机开局、交换先后手、手数上限、
 *   逐手合法性校验、模拟次数) 全部共用。
 *
 * 两个模式:
 *   --mode=train   自对弈训练 A 与 B (相同局数/模拟次数/温度/种子/线程数=1),
 *                  各自存盘。**每个 agent 只用自己那一个 Chess 对象**, 互不干扰。
 *   --mode=match   载入 (或不载入) 权重, 打 N 局受控对局。
 *
 * 公平性 (缺一条这个对照就没有意义):
 *   1. **交换先后手**: 局数强制偶数, 第 i 局 A 执红当且仅当 i 为偶数;
 *   2. **同一组随机开局**: 每局先用同一颗种子随机走 opening 步合法棋 (双方都不参与);
 *   3. **同一模拟预算**: `--sims` 同时给 A 和 B (MCTS/AB 各有自己的口径, 报告里打印);
 *   4. **逐手合法性校验**: 任何一方返回非法/无效走法都记成 BROKEN 并让退出码非 0;
 *   5. **非 0 退出码只看机制**, 比分不影响退出码 (随机权重输棋不是缺陷)。
 *
 * 输出:
 *   * 人类可读的逐局表 + 汇总 (比分 / 得分率 / Wilson 95% 区间 / Elo 差 / ms 每步);
 *   * --csv=<path> 写机读结果, 供后续汇总脚本用 (一行一场)。
 *
 * 统计口径 (为什么必须给区间):
 *   11 局的比分差在 ±8.8% 噪声带以内是分辨不出来的 (见 docs/mcts_sims_scaling.md §3)。
 *   所以这里一律同时报 **得分率 + Wilson 95% 置信区间**, 并把"区间跨过 50%"显式标成
 *   `inconclusive` —— 避免把一个噪声读数读成"算法更高效"。
 *
 * 用法:
 *   bench_agent_arena.exe --mode=train --a=sac --b=ppo --games=200 --sims=64
 *                         --plies=120 --opening=4 --seed=20240901 --save-prefix=weights/arena
 *   bench_agent_arena.exe --mode=match --a=sac --b=mcts --games=60 --sims=200
 *                         --load-a=<prefix> --load-b=<prefix> --csv=out.csv
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "abagent.h"
#include "mcts.h"
#include "sacazagent.h"
#include "ppomcts_agent.h"
#include "rl/ppo.h"
#include "rl/util.hpp"
#include <cstdarg>

/*
   刻意**不**写 `using namespace RL;` —— `RL::Step`(一条经验) 与 `::Step`(象棋的一步)
   同名, 把 RL 整个拉进全局作用域会让每个 `Step` 变成歧义符号 (同 bench_sacaz_vs_ab)。
*/

namespace {

/*
   ---- 临时诊断 (2026-09): 把训练进度落到文件 ----
   为什么需要它: PPO 的 trainSelfPlay 在本程序里**只要它先训练**就稳定崩在
   0xC0000005 (访问违例), 而 Release 构建没有 PDB, 拿不到符号化栈。把"崩前最后到达的
   一手/一个阶段"写进文件是零成本的定位手段 (崩溃后读最后一行即可)。
   默认**关闭**: 只有设了环境变量 ARENA_TRACE 才写。
*/
static FILE *g_traceFile = nullptr;

static void traceOpen()
{
    const char *p = std::getenv("ARENA_TRACE");
    if (p == nullptr || *p == '\0') { return; }
    g_traceFile = std::fopen(p, "w");
}

static void tracef(const char *fmt, ...)
{
    if (g_traceFile == nullptr) { return; }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_traceFile, fmt, ap);
    va_end(ap);
    std::fflush(g_traceFile);
}

/* 无条件的阶段标记 (不需要环境变量): 崩溃前最后一条就是死点 —— stderr 不缓冲, 必到 */
static void mark(const char *what)
{
    std::fprintf(stderr, "[arena] %s\n", what);
    std::fflush(stderr);
}

/* ============================================================
 *  配置
 * ============================================================ */
enum class Kind { SAC, PPO, MCTS, AB, NONE };

struct Cfg {
    std::string mode = "match";         /* train | match */
    std::string aName = "sac";          /* sac | ppo | mcts | ab */
    std::string bName = "mcts";
    int games = 20;                     /* match: 对局数; train: 每个 agent 的自对弈局数 */
    int sims = 64;                      /* A/B 每步模拟次数 (MCTS 用同一数字) */
    int maxPlies = 120;                 /* 每局手数上限 (到上限判和) */
    int openingPlies = 4;               /* 随机开局步数 */
    int abDepth = 2;                    /* --b=ab 时的搜索深度 */
    std::string backboneSac = "mlp";    /* mlp | moe-mlp | tb | dense-tb */
    int sacHidden = 64;
    float sacCpuct = 1.5f;
    float sacValueScale = 1.0f;   /* 只影响搜索期叶子估值 (默认 1 = 原行为) */
    /*
       ---- SAC 消融旋钮 (2026-09) ----
       用来回答"某一项改动到底帮忙还是帮倒忙"。每一项都对应上一轮改动里的一个决定,
       默认值 = 当前实现的取值 (所以不传参时行为与之前逐位一致)。
    */
    float sacClampTarget = 2.0f;   /* <=0: 关掉 TD 目标值域约束 */
    float sacHuber = 1.0f;         /* <=0: 关掉 Huber, 退回纯 MSE */
    float sacEntropyRatio = 0.5f;  /* 目标熵 = ratio x log(合法数); 改前是 0.98 */
    float sacAlphaLr = 5e-3f;      /* alpha 的学习率; 改前是 1e-3 */
    int sacResetInterval = 64;     /* 目标网同步; <=0: 不动 */
    float sacRewardScale = 1.0f;   /* 即时奖励整体缩放 (测"奖励塑形"的影响) */
    bool forceSparseLeaf = false;  /* 强制走稀疏头 (诊断用) */
    bool sacLegacyHash = false;    /* 表示消融: 动作退回 128 槽哈希 (状态仍 19 平面) */
    std::string loadA, loadB;           /* 权重前缀 */
    std::string savePrefix;             /* train 模式的存盘前缀 (-> <prefix>_sac_* / _ppo_*) */
    std::string csvPath;
    unsigned seed = 20240901;
    int probeSamples = 40;
    bool verbose = false;
    bool quiet = false;
};

Cfg g_cfg;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* ============================================================
 *  名字 <-> 枚举
 * ============================================================ */
static bool parseKind(const std::string &s, Kind &out)
{
    if (s == "sac")  { out = Kind::SAC;  return true; }
    if (s == "ppo")  { out = Kind::PPO;  return true; }
    if (s == "mcts") { out = Kind::MCTS; return true; }
    if (s == "ab")   { out = Kind::AB;   return true; }
    return false;
}

static const char *kindName(Kind k)
{
    switch (k) {
    case Kind::SAC:  return "SAC+AZ";
    case Kind::PPO:  return "PPO+MCTS";
    case Kind::MCTS: return "MCTS";
    case Kind::AB:   return "AB";
    default:         return "NONE";
    }
}

static bool parseSacBackbone(const std::string &s, SACAZAgent::Backbone &out)
{
    if (s == "mlp")      { out = SACAZAgent::Backbone::Mlp;          return true; }
    if (s == "moe-mlp")  { out = SACAZAgent::Backbone::SparseMoeMlp; return true; }
    if (s == "tb")       { out = SACAZAgent::Backbone::SparseMoeTb;  return true; }
    if (s == "dense-tb") { out = SACAZAgent::Backbone::DenseMoeTb;   return true; }
    return false;
}

/* ============================================================
 *  统计: Wilson 95% 区间 + Elo 差
 * ============================================================ */
struct ScoreStat {
    int wins = 0, losses = 0, draws = 0, broken = 0;
    int n() const { return wins + losses + draws; }
    double rate() const {
        const int m = n();
        return (m > 0) ? ((double)wins + 0.5 * (double)draws) / (double)m : 0.0;
    }
    /* Wilson score interval (比正态近似在小样本/极端比分下正确得多) */
    void interval(double &lo, double &hi) const
    {
        const int m = n();
        if (m == 0) { lo = 0.0; hi = 1.0; return; }
        const double z = 1.959963984540054;
        const double p = rate();
        const double denom = 1.0 + z * z / (double)m;
        const double centre = (p + z * z / (2.0 * (double)m)) / denom;
        const double half = (z / denom)
                            * std::sqrt(p * (1.0 - p) / (double)m
                                        + z * z / (4.0 * (double)m * (double)m));
        lo = centre - half;
        hi = centre + half;
        if (lo < 0.0) { lo = 0.0; }
        if (hi > 1.0) { hi = 1.0; }
    }
    /* Elo 差 = -400*log10(1/S - 1); S 必须是"胜+半和"的得分率 */
    double eloDiff() const
    {
        const double s = rate();
        if (s <= 0.0 || s >= 1.0) { return (s >= 1.0) ? 800.0 : -800.0; }
        return -400.0 * std::log10(1.0 / s - 1.0);
    }
    bool conclusive() const
    {
        double lo = 0.0, hi = 1.0;
        interval(lo, hi);
        return (lo > 0.5) || (hi < 0.5);
    }
    std::string verdict() const
    {
        if (n() == 0) { return "no games"; }
        if (broken > 0) { return "BROKEN"; }
        return conclusive() ? "decisive" : "inconclusive";
    }
};

/* ============================================================
 *  一局的结果
 * ============================================================ */
struct GameStat {
    int winner = Chess::RESULT_ONGOING;
    int plies = 0;
    bool blackToMove = false;   /* 终局时轮到谁走 (记录用) */
    int score = 0;              /* A 视角: +1 A 胜, 0 和, -1 A 负 */
    bool aIllegal = false;
    bool bIllegal = false;
    const char *endReason = "";
    long long aNodes = 0;       /* A 的搜索节点数合计 (模拟次数 * 步数) */
    long long bNodes = 0;
    int aReported = 0;          /* A 上报损失的次数 (= 优化器步数, PPO 每局 1 次) */
    int bReported = 0;
    double aMsPerMove = 0.0;
    double bMsPerMove = 0.0;
    double redMaterial = 0.0;
    double blackMaterial = 0.0;
};

/* ============================================================
 *  Agent 包装: 统一 4 个 agent 的"走一步"与"上报损失"口径
 * ============================================================ */
struct Agent {
    Kind kind = Kind::NONE;
    std::string label;
    std::string backboneName;
    std::unique_ptr<SACAZAgent> sac;
    std::unique_ptr<PPOMCTSAgent> ppo;
    std::unique_ptr<MCTS> mcts;
    std::unique_ptr<ABAgent> ab;
    float lastLoss = std::numeric_limits<float>::quiet_NaN();

    bool trainable() const { return kind == Kind::SAC || kind == Kind::PPO; }

    Step move(int color, int sims)
    {
        switch (kind) {
        case Kind::SAC:  return sac->selectMove(color, sims, 0.0f);
        case Kind::PPO:  return ppo->selectMove(color, sims, 0.0f);
        case Kind::MCTS: return mcts->findBestMove(color, sims);
        case Kind::AB:   return ab->getBestMove(color);
        default:         return Step();
        }
    }

    /*
       损失上报次数 = 优化器步数。getLastTrainLoss() 在"这一手没训练"时返回的还是
       上一次的值, 所以只有**数值变了**才计一次 —— 这正是界面损失曲线取点的口径,
       用它来量"曲线密集不密集"是直接的。
    */
    void pollLoss(int &reported, float &first, float &last)
    {
        if (!trainable()) { return; }
        const float v = (kind == Kind::SAC) ? sac->getLastTrainLoss()
                                            : ppo->getLastTrainLoss();
        if (std::isnan(v)) { return; }
        if (std::isnan(lastLoss) || v != lastLoss) {
            reported++;
            if (std::isnan(first)) { first = v; }
            last = v;
            lastLoss = v;
        }
    }

    /* 学习步数 (SAC 有独立计数; PPO 用上报次数代表) */
    int learnSteps() const
    {
        if (kind == Kind::SAC)  { return sac->getLearnSteps(); }
        if (kind == Kind::PPO)  { return ppo->getTotalEpisodes(); }
        return 0;
    }

    long long paramCount() const
    {
        if (kind == Kind::SAC) { return sac->actor.paramCount(); }
        if (kind == Kind::PPO) { return ppo->actorParamCount(); }
        return 0;
    }
};

/* ============================================================
 *  棋盘小工具
 * ============================================================ */
static void materialOf(Chess &c, double &red, double &black)
{
    red = 0.0;
    black = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s == nullptr || !s->alive) { continue; }
        if (s->color == Stone::COLOR_RED) { red += s->value; } else { black += s->value; }
    }
}

/* 随机开局: 双方都不参与, 只为把每局的起始局面岔开 */
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

/* ============================================================
 *  打一局: A 执 aColor (RED/BLACK), B 执另一色
 * ============================================================ */
static GameStat playGame(Chess &c, Agent &A, Agent &B, int aColor, int sims)
{
    GameStat g;

    c.reset();
    int turn = Stone::COLOR_RED;
    randomOpening(c, turn, g_cfg.openingPlies);

    double aMsSum = 0.0, bMsSum = 0.0;
    int aMoves = 0, bMoves = 0;
    float aFirst = std::numeric_limits<float>::quiet_NaN();
    float bFirst = std::numeric_limits<float>::quiet_NaN();
    float aLast = std::numeric_limits<float>::quiet_NaN();
    float bLast = std::numeric_limits<float>::quiet_NaN();

    while (g.plies < g_cfg.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            g.winner = res;
            g.endReason = (res == Chess::RESULT_DRAW) ? "repetition/60-move" : "mate";
            break;
        }

        const bool aTurn = (turn == aColor);
        Agent &mover = aTurn ? A : B;

        const double t0 = nowMs();
        const Step s = mover.move(turn, sims);
        const double ms = nowMs() - t0;

        if (aTurn) { aMsSum += ms; aMoves++; g.aNodes += sims; }
        else       { bMsSum += ms; bMoves++; g.bNodes += sims; }
        mover.pollLoss(aTurn ? g.aReported : g.bReported,
                       aTurn ? aFirst : bFirst,
                       aTurn ? aLast : bLast);

        /*
           **合法性校验 (逐手)**, 在 moveForward 之前 —— 一旦棋盘吃了非法走法,
           后面所有读数都失去意义。
        */
        std::vector<Step*> legal;
        c.sample(turn, legal);
        bool legalMove = false;
        if (!legal.empty() && s.valid) {
            legalMove = c.isLegalMove(turn, &s);
        }
        Steps::instance().put(legal);

        if (!s.valid || !legalMove) {
            if (aTurn) { g.aIllegal = true; } else { g.bIllegal = true; }
            g.endReason = aTurn ? "A returned an ILLEGAL/invalid move"
                                : "B returned an ILLEGAL/invalid move";
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-5s (%d,%d)->(%d,%d) %8.1f ms  [%s]\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        aTurn ? A.label.c_str() : B.label.c_str());
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        g.plies++;
    }
    g.blackToMove = (turn == Stone::COLOR_BLACK);

    if (g.winner == Chess::RESULT_ONGOING && !g.aIllegal && !g.bIllegal) {
        g.winner = Chess::RESULT_DRAW;
        g.endReason = "ply limit";
    }

    /* 棋盘视角 -> A 视角 */
    if (!g.aIllegal && !g.bIllegal) {
        if (g.winner == Chess::RESULT_DRAW) {
            g.score = 0;
        } else {
            const bool redWon = (g.winner == Chess::RESULT_RED_WIN);
            g.score = (redWon == (aColor == Stone::COLOR_RED)) ? 1 : -1;
        }
    }

    materialOf(c, g.redMaterial, g.blackMaterial);
    g.aMsPerMove = (aMoves > 0) ? aMsSum / (double)aMoves : 0.0;
    g.bMsPerMove = (bMoves > 0) ? bMsSum / (double)bMoves : 0.0;

    if (g_cfg.verbose && (!std::isnan(aFirst) || !std::isnan(bFirst))) {
        std::printf("      [损失上报] A %d 次 (%.5g -> %.5g), B %d 次 (%.5g -> %.5g)\n",
                    g.aReported, (double)aFirst, (double)aLast,
                    g.bReported, (double)bFirst, (double)bLast);
    }
    return g;
}

/* ============================================================
 *  probe 模式: 量一个 agent 的**策略塌缩**程度
 *
 *  为什么需要它: 主表里的"训练后变弱"只说明结果, 不说明机制。策略熵 / top1 份额 /
 *  Q 的尺度是**当场可量**的结构读数: 如果训练后熵掉到接近 0、top1 接近 1, 而搜索又
 *  按访问次数出招, 那么"每手都走同一个槽位 -> 三步重复判和"就有了直接证据, 与对局
 *  里出现的"11 手和棋"对得上。
 * ============================================================ */
struct ProbeStat {
    int n = 0;
    double entropy = 0.0;      /* 掩码 softmax 的香农熵 (nats) */
    double normEntropy = 0.0;  /* H / log(合法数), 1 = 均匀 */
    double top1 = 0.0;
    double qMin = 0.0, qMax = 0.0, qAbsMean = 0.0;
    double legal = 0.0;
    double alpha = 0.0;
    double loss = 0.0;
};

static void probeAgent(Agent &ag, int samples, int sims, ProbeStat &out)
{
    if (ag.kind != Kind::SAC && ag.kind != Kind::PPO) {
        std::printf("[probe] 只支持 sac / ppo (无学习参数的 agent 没有策略分布)\n");
        return;
    }
    const int actionDim = (ag.kind == Kind::SAC) ? SACAZAgent::ACTION_DIM
                                                 : PPOMCTSAgent::ACTION_DIM;
    const int stateDim = (ag.kind == Kind::SAC) ? SACAZAgent::STATE_DIM
                                                : PPOMCTSAgent::STATE_DIM;

    Chess c;
    c.reset();
    RL::Tensor state(stateDim, 1);
    RL::Tensor mask(actionDim, 1);
    RL::Tensor pi(actionDim, 1);

    const double t0 = nowMs();
    for (int i = 0; i < samples; i++) {
        /* 每 8 个样本重开一局: 局面既有开局的, 也有中局的 */
        if (i % 8 == 0) {
            c.reset();
            int turn = Stone::COLOR_RED;
            randomOpening(c, turn, 4);
        }
        const int turn = c.sideToMove;

        std::vector<Step*> legal;
        std::vector<int> idx;
        if (ag.kind == Kind::SAC) {
            ag.sac->getLegalActions(turn, legal, idx, mask);
            ag.sac->encodeStateFor(turn, state);
            ag.sac->policy(state, mask, pi);
        } else {
            /* PPO 的策略走它自己的稀疏入口 (RL::PPO::actionMasked 是公开的, 见 ppo.h);
               掩码仍由 agent 的 getLegalActions 给出, 与搜索内部同一口径。 */
            ag.ppo->getLegalActions(turn, legal, idx, mask);
            ag.ppo->encodeStateFor(turn, state);
            pi.zero();
            std::vector<float> probs;
            if (ag.ppo->ppo.actionMasked(state, idx, probs)) {
                for (std::size_t k = 0; k < idx.size() && k < probs.size(); k++) {
                    pi[idx[k]] = probs[k];
                }
            }
        }
        Steps::instance().put(legal);
        if (idx.empty()) { continue; }

        double H = 0.0, top1 = 0.0;
        for (int a = 0; a < actionDim; a++) {
            if (mask[a] <= 0.5f) { continue; }
            const double p = (double)pi[a];
            if (p > 0.0) { H -= p * std::log(p); }
            if (p > top1) { top1 = p; }
        }
        const double logLegal = std::log((double)idx.size());

        double qmin = 1e30, qmax = -1e30, qabs = 0.0;
        int qn = 0;
        if (ag.kind == Kind::SAC) {
            RL::Tensor q1(actionDim, 1), q2(actionDim, 1);
            ag.sac->qValues(state, q1, q2);
            for (int a = 0; a < actionDim; a++) {
                if (mask[a] <= 0.5f) { continue; }
                const double q = std::min((double)q1[a], (double)q2[a]);
                qmin = std::min(qmin, q);
                qmax = std::max(qmax, q);
                qabs += std::fabs(q);
                qn++;
            }
        }

        out.n++;
        out.entropy += H;
        out.normEntropy += (logLegal > 1e-9) ? H / logLegal : 0.0;
        out.top1 += top1;
        out.legal += (double)idx.size();
        if (qn > 0) {
            out.qMin += qmin;
            out.qMax += qmax;
            out.qAbsMean += qabs / (double)qn;
        }

        /* 往前走一手 (按策略采样), 让样本覆盖中局 */
        const int stepSims = std::max(1, sims / 8);
        const Step s = ag.move(turn, stepSims);
        if (!s.valid) { c.reset(); continue; }
        std::vector<Step*> legalNow;
        c.sample(turn, legalNow);
        const bool ok = c.isLegalMove(turn, &s);
        Steps::instance().put(legalNow);
        if (!ok) { c.reset(); continue; }
        double dummy = 0.0;
        c.moveForward(&s, dummy);
    }
    const double dt = (nowMs() - t0) / 1000.0;

    if (out.n > 0) {
        const double n = (double)out.n;
        out.entropy /= n;
        out.normEntropy /= n;
        out.top1 /= n;
        out.legal /= n;
        out.qMin /= n;
        out.qMax /= n;
        out.qAbsMean /= n;
    }
    out.alpha = (ag.kind == Kind::SAC) ? (double)ag.sac->getAlpha() : 0.0;
    out.loss = (double)((ag.kind == Kind::SAC) ? ag.sac->getLastTrainLoss()
                                               : ag.ppo->getLastTrainLoss());

    std::printf("  %-28s 样本=%3d  合法数=%.1f  策略熵=%.3f (归一 %.3f)  "
                "top1=%.3f  |Q|=%.3f [%.3f, %.3f]  alpha=%.3f loss=%.5g  (%.1f s)\n",
                ag.label.c_str(), out.n, out.legal, out.entropy, out.normEntropy,
                out.top1, out.qAbsMean, out.qMin, out.qMax, out.alpha, out.loss, dt);
}

/* ============================================================
 *  Agent 构造
 * ============================================================ */
static void buildAgent(Agent &ag, Kind kind, Chess &c)
{
    ag.kind = kind;
    switch (kind) {
    case Kind::SAC: {
        SACAZAgent::Backbone bb = SACAZAgent::Backbone::Mlp;
        if (!parseSacBackbone(g_cfg.backboneSac, bb)) {
            std::printf("[错误] 未知 SAC 骨干: %s\n", g_cfg.backboneSac.c_str());
            std::exit(1);
        }
        ag.sac.reset(new SACAZAgent(c, g_cfg.sacHidden, 0.99f, 0.001f,
                                    g_cfg.sacCpuct, bb, 64, 0.1f));
        ag.sac->valueScale = g_cfg.sacValueScale;
        /* 消融旋钮 (默认值 = 当前实现, 所以不传参时行为不变) */
        ag.sac->clampTarget = g_cfg.sacClampTarget;
        ag.sac->huberDelta = g_cfg.sacHuber;
        ag.sac->entropyRatio = g_cfg.sacEntropyRatio;
        ag.sac->learningRateAlpha = g_cfg.sacAlphaLr;
        ag.sac->rewardScale = g_cfg.sacRewardScale;
        ag.sac->legacyHashAction = g_cfg.sacLegacyHash;
        if (g_cfg.sacResetInterval > 0) {
            ag.sac->replaceTargetIter = g_cfg.sacResetInterval;
        }
        ag.backboneName = SACAZAgent::backboneName(bb);
        ag.label = std::string("SAC+AZ/") + ag.backboneName;
        break;
    }
    case Kind::PPO: {
        /* 与 GUI 的新 agent 同一档: MLP 专家 (E=8/top-2) + MLP 骨干 + 同 lr/c_puct */
        ag.ppo.reset(new PPOMCTSAgent(c, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true,
                                      RL::PPO::Backbone::MlpExperts));
        ag.backboneName = RL::PPO::backboneName(RL::PPO::Backbone::MlpExperts);
        ag.label = std::string("PPO+MCTS/") + ag.backboneName;
        break;
    }
    case Kind::MCTS:
        ag.mcts.reset(new MCTS(c, 1.414));
        ag.label = "MCTS";
        break;
    case Kind::AB:
        ag.ab.reset(new ABAgent(c, g_cfg.abDepth));
        ag.label = "AB";
        break;
    default:
        break;
    }
}

static bool loadWeights(Agent &ag, const std::string &prefix)
{
    if (prefix.empty()) { return true; }
    bool ok = true;
    if (ag.kind == Kind::SAC)     { ok = ag.sac->loadModel(prefix); }
    else if (ag.kind == Kind::PPO) { ok = ag.ppo->loadModel(prefix); }
    return ok;
}

static std::string saveWeights(Agent &ag, const std::string &prefix)
{
    if (prefix.empty()) { return "(未保存)"; }
    bool ok = true;
    if (ag.kind == Kind::SAC)      { ok = ag.sac->saveModel(prefix); }
    else if (ag.kind == Kind::PPO) { ok = ag.ppo->saveModel(prefix); }
    else                           { return "(不可训练: 无权重)"; }
    return ok ? "成功" : "**失败**";
}

static void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *key) -> const char* {
            const std::size_t n = std::strlen(key);
            if (std::strncmp(a, key, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--mode"))        { g_cfg.mode = v; }
        else if (const char *v = val("--a"))      { g_cfg.aName = v; }
        else if (const char *v = val("--b"))      { g_cfg.bName = v; }
        else if (const char *v = val("--games"))  { g_cfg.games = std::atoi(v); }
        else if (const char *v = val("--sims"))   { g_cfg.sims = std::atoi(v); }
        else if (const char *v = val("--plies"))  { g_cfg.maxPlies = std::atoi(v); }
        else if (const char *v = val("--opening")){ g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--depth"))  { g_cfg.abDepth = std::atoi(v); }
        else if (const char *v = val("--backbone-sac")) { g_cfg.backboneSac = v; }
        else if (const char *v = val("--sac-hidden"))   { g_cfg.sacHidden = std::atoi(v); }
        else if (const char *v = val("--sac-cpuct"))    { g_cfg.sacCpuct = (float)std::atof(v); }
        else if (const char *v = val("--sac-value-scale")) { g_cfg.sacValueScale = (float)std::atof(v); }
        else if (const char *v = val("--sac-clamp"))   { g_cfg.sacClampTarget = (float)std::atof(v); }
        else if (const char *v = val("--sac-huber"))   { g_cfg.sacHuber = (float)std::atof(v); }
        else if (const char *v = val("--sac-entropy-ratio")) { g_cfg.sacEntropyRatio = (float)std::atof(v); }
        else if (const char *v = val("--sac-alpha-lr")) { g_cfg.sacAlphaLr = (float)std::atof(v); }
        else if (const char *v = val("--sac-reset-interval")) { g_cfg.sacResetInterval = std::atoi(v); }
        else if (const char *v = val("--sac-reward-scale")) { g_cfg.sacRewardScale = (float)std::atof(v); }
        else if (std::strcmp(a, "--sac-legacy-hash") == 0) { g_cfg.sacLegacyHash = true; }
        else if (const char *v = val("--load-a")) { g_cfg.loadA = v; }
        else if (const char *v = val("--load-b")) { g_cfg.loadB = v; }
        else if (const char *v = val("--save-prefix")) { g_cfg.savePrefix = v; }
        else if (const char *v = val("--csv"))    { g_cfg.csvPath = v; }
        else if (const char *v = val("--probe-samples")) { g_cfg.probeSamples = std::atoi(v); }
        else if (const char *v = val("--seed"))   { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (std::strcmp(a, "--verbose") == 0) { g_cfg.verbose = true; }
        else if (std::strcmp(a, "--quiet") == 0)   { g_cfg.quiet = true; }
        else {
            std::printf("[警告] 未知参数被忽略: %s\n", a);
        }
    }

    if (g_cfg.maxPlies < 2)  { g_cfg.maxPlies = 2; }
    if (g_cfg.sims < 1)      { g_cfg.sims = 1; }
    if (g_cfg.abDepth < 1)   { g_cfg.abDepth = 1; }
    if (g_cfg.games < 1)     { g_cfg.games = 1; }
    if (g_cfg.probeSamples < 1) { g_cfg.probeSamples = 1; }
    if (g_cfg.mode == "match") {
        /* 交换先后手要求偶数局 */
        if (g_cfg.games % 2 != 0) { g_cfg.games++; }
    }
    /*
       mode 必须显式合法。这里踩过一次坑: 用**旧二进制**跑 `--mode=probe` 时, 旧版本
       既不认识这个模式、也不报错, 于是它静默地跑成了一场对局 —— 而报告里差点把
       那一场的"比分"当成探针读数。未知 mode 现在直接拒绝执行。
    */
    if (g_cfg.mode != "match" && g_cfg.mode != "train" && g_cfg.mode != "probe") {
        std::printf("[错误] 未知 --mode=%s (可选 match / train / probe)\n",
                    g_cfg.mode.c_str());
        std::exit(2);
    }
}

} // namespace

/* ============================================================
 *  main
 * ============================================================ */
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    parseArgs(argc, argv);
    RL::Random::setSeed(g_cfg.seed);

    Kind ka = Kind::NONE, kb = Kind::NONE;
    if (!parseKind(g_cfg.aName, ka) || !parseKind(g_cfg.bName, kb)) {
        std::printf("[错误] 未知 agent: --a=%s --b=%s (可选 sac/ppo/mcts/ab)\n",
                    g_cfg.aName.c_str(), g_cfg.bName.c_str());
        return 1;
    }

    /*
       所有 agent 必须绑定**同一个** Chess 对象 —— 也就是对局真正在走的那一个。
       这里踩过一次坑 (2026-09): 第一版让每个 agent 各持一个私有棋盘、对局再用第三个,
       结果是"第二个走子的那一方永远被判成走了非法走法" —— 因为轮到它时, 它读的是
       自己那个**还停在初始局面**的棋盘, 搜索出来的当然不是当前局面的合法走法。
       逐手串行调用时共用一个棋盘是安全的 (agent 只在调用期间持引用, 不缓存局面;
       搜索内部自己 moveForward/moveBack 成对撤销)。
    */
    Chess board;
    board.reset();

    const double tBuild0 = nowMs();
    Agent A, B;
    buildAgent(A, ka, board);
    buildAgent(B, kb, board);
    const double tBuild1 = nowMs();

    if (!g_cfg.quiet) {
        std::printf("=== bench_agent_arena: %s  vs  %s ===\n", A.label.c_str(), B.label.c_str());
        std::printf("模式    : %s\n", g_cfg.mode.c_str());
        std::printf("协议    : %s=%d 局 | 每步模拟 %d | 每局上限 %d 手 | 随机开局 %d 步 | seed=%u\n",
                    (g_cfg.mode == "match") ? "对局" : "自对弈/agent",
                    g_cfg.games, g_cfg.sims, g_cfg.maxPlies, g_cfg.openingPlies, g_cfg.seed);
        if (A.paramCount() > 0) {
            std::printf("A 结构  : %s  可训练参数=%lld\n", A.label.c_str(), A.paramCount());
        }
        if (B.paramCount() > 0) {
            std::printf("B 结构  : %s  可训练参数=%lld\n", B.label.c_str(), B.paramCount());
        }
        std::printf("构建耗时: %.2f s\n\n", (tBuild1 - tBuild0) / 1000.0);
    }

    /* ============================================================
     *  train 模式: 两个 agent 各自自对弈
     * ============================================================ */
    if (g_cfg.mode == "train") {
        if (!A.trainable() || !B.trainable()) {
            std::printf("[错误] train 模式要求 --a/--b 都是可训练 agent (sac/ppo)\n");
            return 1;
        }
        const int half = g_cfg.games;
        double aTrainSec = 0.0, bTrainSec = 0.0;

        /*
           **必须按 Kind 分派** —— 这里踩过一次坑 (2026-09): 第一版无条件写
           `A.sac->trainSelfPlay(...)`, 于是 `--a=ppo` 时 A.sac 是 nullptr (Agent 包装里
           只有一个指针非空), 直接访问违例 0xC0000005, 而且崩在**训练第一行之前**,
           看起来像"PPO 训练崩了"。教训与"逐手合法性校验"同类: 包装层必须按 kind 分派,
           不能假设某个成员一定有效。
        */
        auto trainOne = [&](Agent &ag, const char *tag, double &secs) {
            const double t0 = nowMs();
            std::fprintf(stderr, "[arena] %s (%s) trainSelfPlay 开始: games=%d sims=%d\n",
                         tag, ag.label.c_str(), half, g_cfg.sims);
            std::fflush(stderr);
            tracef("[trace] %s (%s) 开始 games=%d sims=%d plies=%d\n",
                   tag, ag.label.c_str(), half, g_cfg.sims, g_cfg.maxPlies);
            if (ag.kind == Kind::SAC) {
                ag.sac->trainSelfPlay(half, g_cfg.sims, g_cfg.maxPlies,
                                      false /* verbose */, 1.0f, 0.25f, 4);
            } else {
                ag.ppo->trainSelfPlay(half, g_cfg.sims, g_cfg.maxPlies,
                                      false /* verbose */, 1.0f, 0.25f);
            }
            secs = (nowMs() - t0) / 1000.0;
            std::fprintf(stderr, "[arena] %s 训练结束: %.0f s\n", tag, secs);
            std::fflush(stderr);
            tracef("[trace] %s 结束 %.1f s\n", tag, secs);
        };

        trainOne(A, "A", aTrainSec);
        trainOne(B, "B", bTrainSec);

        /*
           前缀必须带上**类型名**而不是位置 ( _a / _b )。踩过一次: 原来写成
           `<prefix>_<aName>` / `<prefix>_<bName>`, 而 `--a=sac --b=ppo` 训练时两个
           agent 都会把自己的文件写成 `<prefix>_sac_*` / `<prefix>_ppo_*` —— 但
           **A 的分支用的是 A.kind**, 于是 A(sac) 写 `_sac_*`、B(ppo) 也写 `_ppo_*`,
           看起来没问题; 真正出事的是 `--a=sac --b=sac`(同类型训练两份) 时两个进程写
           同一组文件名, 后一个把前一个覆盖掉。更隐蔽的一次是 A 走 PPO 分支却按
           `aName` 命名, 让 SAC 的权重文件里装的其实是 PPO 的权重 (参数量对不上,
           载入时会被守卫拒绝 —— 但"训练完发现权重是别人的"这件事本身就是缺陷)。
           现在用 `kindName()` 直接落在文件名里, 命名与内容一致。
        */
        const std::string pA = g_cfg.savePrefix.empty()
                                   ? std::string()
                                   : g_cfg.savePrefix + "_" + kindName(A.kind);
        const std::string pB = g_cfg.savePrefix.empty()
                                   ? std::string()
                                   : g_cfg.savePrefix + "_" + kindName(B.kind);
        const std::string rA = saveWeights(A, pA);
        const std::string rB = saveWeights(B, pB);

        std::printf("=== 自对弈训练完成 (每个 agent 各 %d 局, %d 模拟/步) ===\n", half, g_cfg.sims);
        std::printf("  自对弈用时: A(%.0f s)  B(%.0f s)\n", aTrainSec, bTrainSec);
        std::printf("  存盘    : A -> %s %s | B -> %s %s\n",
                    pA.empty() ? "(none)" : pA.c_str(), rA.c_str(),
                    pB.empty() ? "(none)" : pB.c_str(), rB.c_str());
        std::printf("VERDICT: PASS (train) games=%d sims=%d seed=%u\n",
                    half, g_cfg.sims, g_cfg.seed);
        return 0;
    }

    /* ============================================================
     *  match 模式: 载入权重 + 受控对局
     * ============================================================ */
    if (!loadWeights(A, g_cfg.loadA)) {
        std::printf("VERDICT: FAIL (A 权重载入失败: %s)\n", g_cfg.loadA.c_str());
        return 1;
    }
    if (!loadWeights(B, g_cfg.loadB)) {
        std::printf("VERDICT: FAIL (B 权重载入失败: %s)\n", g_cfg.loadB.c_str());
        return 1;
    }

    /* ============================================================
     *  probe 模式: 量策略塌缩 (不需要对局)
     * ============================================================ */
    if (g_cfg.mode == "probe") {
        std::printf("=== 策略探针 (样本=%d, 每步模拟=%d) ===\n", g_cfg.probeSamples, g_cfg.sims);
        ProbeStat stA, stB;
        if (A.trainable()) { probeAgent(A, g_cfg.probeSamples, g_cfg.sims, stA); }
        if (B.trainable()) { probeAgent(B, g_cfg.probeSamples, g_cfg.sims, stB); }
        std::printf("VERDICT: PASS (probe)\n");
        return 0;
    }

    if (!g_cfg.quiet) {
        if (!g_cfg.loadA.empty()) {
            std::printf("A 权重  : %s -> 载入成功\n", g_cfg.loadA.c_str());
        }
        if (!g_cfg.loadB.empty()) {
            std::printf("B 权重  : %s -> 载入成功\n", g_cfg.loadB.c_str());
        }
        if (g_cfg.loadA.empty() && A.trainable()) {
            std::printf("A 权重  : 未载入 => **随机初始化**, 下面的比分不是棋力\n");
        }
        if (g_cfg.loadB.empty() && B.trainable()) {
            std::printf("B 权重  : 未载入 => **随机初始化**, 下面的比分不是棋力\n");
        }
        std::printf("\n");
    }

    /* ---- 对局 ---- */
    ScoreStat st;
    int pliesTotal = 0;
    long long aNodesTotal = 0, bNodesTotal = 0;
    int aReportedTotal = 0, bReportedTotal = 0;
    double aMsAcc = 0.0, bMsAcc = 0.0;
    int aMoveAcc = 0, bMoveAcc = 0;
    bool anyBroken = false;

    const double tStart = nowMs();
    for (int i = 0; i < g_cfg.games; i++) {
        const bool aIsRed = (i % 2 == 0);   /* 交换先后手 */
        const int aColor = aIsRed ? Stone::COLOR_RED : Stone::COLOR_BLACK;
        const GameStat g = playGame(board, A, B, aColor, g_cfg.sims);

        const bool bad = g.aIllegal || g.bIllegal;
        if (bad)            { st.broken++; anyBroken = true; }
        else if (g.score > 0) { st.wins++; }
        else if (g.score < 0) { st.losses++; }
        else                { st.draws++; }

        pliesTotal   += g.plies;
        aNodesTotal  += g.aNodes;
        bNodesTotal  += g.bNodes;
        aReportedTotal += g.aReported;
        bReportedTotal += g.bReported;
        aMsAcc += g.aMsPerMove; bMsAcc += g.bMsPerMove;
        aMoveAcc++; bMoveAcc++;

        if (!g_cfg.quiet) {
            const char *verdict = bad ? "BROKEN"
                                : (g.score > 0) ? "A wins"
                                : (g.score < 0) ? "B wins" : "draw";
            std::printf("  game %3d/%d  A=%-5s  %-7s %3d 手  [%s]  "
                        "A %7.1f ms/步 (%d 次上报), B %7.1f ms/步 (%d 次上报)  材料 红%.1f-黑%.1f\n",
                        i + 1, g_cfg.games, aIsRed ? "RED" : "BLACK", verdict, g.plies,
                        g.endReason, g.aMsPerMove, g.aReported, g.bMsPerMove, g.bReported,
                        g.redMaterial, g.blackMaterial);
        }
    }
    const double elapsedSec = (nowMs() - tStart) / 1000.0;

    double lo = 0.0, hi = 1.0;
    st.interval(lo, hi);
    const double aMs = (aMoveAcc > 0) ? aMsAcc / (double)aMoveAcc : 0.0;
    const double bMs = (bMoveAcc > 0) ? bMsAcc / (double)bMoveAcc : 0.0;
    const double rate = st.rate();

    if (!g_cfg.quiet) {
        std::printf("\n=== 结果 (%s 视角) ===\n", A.label.c_str());
        std::printf("  比分      : A %d 胜 / B %d 胜 / 和 %d", st.wins, st.losses, st.draws);
        if (st.broken > 0) { std::printf("  (+%d 局机制违规)", st.broken); }
        std::printf("\n");
        std::printf("  得分率    : %.1f%%   95%% Wilson 区间 [%.1f%%, %.1f%%]  -> %s\n",
                    100.0 * rate, 100.0 * lo, 100.0 * hi, st.verdict());
        std::printf("  Elo 差    : %+.0f (A 相对 B)\n", st.eloDiff());
        std::printf("  平均手数  : %.1f\n", st.n() > 0 ? (double)pliesTotal / (double)st.n() : 0.0);
        std::printf("  思考时间  : A %.1f ms/步, B %.1f ms/步  -> B 是 A 的 %.2fx\n",
                    aMs, bMs, (aMs > 0.0) ? bMs / aMs : 0.0);
        std::printf("  搜索节点  : A %lld, B %lld\n", aNodesTotal, bNodesTotal);
        std::printf("  损失上报  : A %d 次, B %d 次 (每次 = 一步优化器更新)\n",
                    aReportedTotal, bReportedTotal);
        std::printf("  总耗时    : %.1f s (%.2f s/局)\n", elapsedSec,
                    st.n() > 0 ? elapsedSec / (double)st.n() : 0.0);
    }

    if (!g_cfg.csvPath.empty()) {
        FILE *f = std::fopen(g_cfg.csvPath.c_str(), "w");
        if (f != nullptr) {
            std::fprintf(f, "mode,a,b,games,sims,plies_limit,opening,seed,"
                            "a_wins,b_wins,draws,broken,score_rate,wilson_lo,wilson_hi,"
                            "elo_diff,verdict,a_ms_per_move,b_ms_per_move,"
                            "a_loss_reports,b_loss_reports,a_nodes,b_nodes,"
                            "a_params,b_params,a_backbone,b_backbone,elapsed_s\n");
            std::fprintf(f, "match,%s,%s,%d,%d,%d,%d,%u,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.2f,%s,"
                            "%.3f,%.3f,%d,%d,%lld,%lld,%lld,%lld,%s,%s,%.1f\n",
                         g_cfg.aName.c_str(), g_cfg.bName.c_str(),
                         st.n(), g_cfg.sims, g_cfg.maxPlies, g_cfg.openingPlies, g_cfg.seed,
                         st.wins, st.losses, st.draws, st.broken, rate, lo, hi,
                         st.eloDiff(), st.verdict(), aMs, bMs,
                         aReportedTotal, bReportedTotal, aNodesTotal, bNodesTotal,
                         A.paramCount(), B.paramCount(),
                         A.backboneName.empty() ? "-" : A.backboneName.c_str(),
                         B.backboneName.empty() ? "-" : B.backboneName.c_str(),
                         elapsedSec);
            std::fclose(f);
            if (!g_cfg.quiet) { std::printf("  CSV       : %s\n", g_cfg.csvPath.c_str()); }
        } else if (!g_cfg.quiet) {
            std::printf("  [警告] CSV 打不开: %s\n", g_cfg.csvPath.c_str());
        }
    }

    /* 退出码只看机制: 非法走法 => 非 0; 比分与统计不影响 */
    std::printf("VERDICT: %s | %s vs %s | 局数=%d 比分=%d/%d/%d 得分率=%.1f%% "
                "区间[%.1f,%.1f] Elo%+.0f %s\n",
                anyBroken ? "FAIL" : "PASS",
                g_cfg.aName.c_str(), g_cfg.bName.c_str(), st.n(),
                st.wins, st.losses, st.draws, 100.0 * rate,
                100.0 * lo, 100.0 * hi, st.eloDiff(), st.verdict());
    return anyBroken ? 1 : 0;
}
