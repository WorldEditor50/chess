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
    double prior = 1.0;
    int lfs = 1;              /* learnFromSearch (只在 --train=1 时有意义) */
    double cpuct = 1.5;       /* PUCT 探索常数 */
    /* 手工锚预训练 (0 = 不做): 局面数 / epoch 数 / 是否连骨干一起训 */
    int pretrain = 0;
    int pretrainEpochs = 8;
    int pretrainTrunk = 0;
    double lr = 0.001;        /* 学习率 (pretrain 与 RL 都吃它) */
    std::string loadPrefix;
    std::string loadPrefix2;  /* 臂 B 的权重 (空 = 与臂 A 同一份 ⇒ 只比旋钮) */
    std::string savePrefix;   /* train 模式把权重写到这个前缀 */
    int openings = 16;        /* 预生成的开局条数 (K): 第 i 局用第 i%K 条 ⇒ 两臂配对 */
    int adj = 0;              /* 1 = 上限到时按子力裁定胜负 (见 playGame 的说明) */
    int snapshotEvery = 0;    /* train 模式: 每 N 局存一次快照 (0 = 只存最后) */
    unsigned seed = 20240901u;
    bool quiet = false;

    /*
       ---- 臂 B 的覆盖值 (arena2 用) ----
       哨兵值表示"没给这一项 ⇒ 与臂 A 相同"。为什么不做成"自动取反": 优化实验要的是
       "只改一个变量", 而自动取反一次会翻好几个 (第一轮 agree 模式就是这样, 结果只能
       读成"这一组旋钮"的效应, 分不出是哪一个)。这里显式给谁改谁。
    */
    int bTwin = -1, bSparse = -1, bDense = -1, bClamp = -999, bDbl = -1;
    int bSims = -1, bEpochs = -1, bEvery = -1, bLfs = -1, bTauIter = -1;
    int bPretrain = -1, bPretrainEpochs = -1, bPretrainTrunk = -1;
    double bHuber = -1.0, bTau = -1.0, bPrior = -1.0, bCpuct = -1.0, bLr = -1.0;
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
        else if (k == "--prior" && v) { g.prior = std::atof(v); }
        else if (k == "--cpuct" && v) { g.cpuct = std::atof(v); }
        else if (k == "--pretrain" && v) { g.pretrain = std::atoi(v); }
        else if (k == "--pretrain-epochs" && v) { g.pretrainEpochs = std::atoi(v); }
        else if (k == "--pretrain-trunk" && v) { g.pretrainTrunk = std::atoi(v); }
        else if (k == "--lr" && v) { g.lr = std::atof(v); }
        else if (k == "--lfs" && v) { g.lfs = std::atoi(v); }
        else if (k == "--load" && v) { g.loadPrefix = v; }
        else if (k == "--load2" && v) { g.loadPrefix2 = v; }
        else if (k == "--save" && v) { g.savePrefix = v; }
        else if (k == "--openings" && v) { g.openings = std::atoi(v); }
        else if (k == "--adj" && v) { g.adj = std::atoi(v); }
        else if (k == "--snapshot" && v) { g.snapshotEvery = std::atoi(v); }
        else if (k == "--b-twin" && v) { g.bTwin = std::atoi(v); }
        else if (k == "--b-sparse" && v) { g.bSparse = std::atoi(v); }
        else if (k == "--b-dense" && v) { g.bDense = std::atoi(v); }
        else if (k == "--b-clamp" && v) { g.bClamp = std::atoi(v); }
        else if (k == "--b-huber" && v) { g.bHuber = std::atof(v); }
        else if (k == "--b-dbl" && v) { g.bDbl = std::atoi(v); }
        else if (k == "--b-tau" && v) { g.bTau = std::atof(v); }
        else if (k == "--b-tauiter" && v) { g.bTauIter = std::atoi(v); }
        else if (k == "--b-sims" && v) { g.bSims = std::atoi(v); }
        else if (k == "--b-epochs" && v) { g.bEpochs = std::atoi(v); }
        else if (k == "--b-every" && v) { g.bEvery = std::atoi(v); }
        else if (k == "--b-lfs" && v) { g.bLfs = std::atoi(v); }
        else if (k == "--b-prior" && v) { g.bPrior = std::atof(v); }
        else if (k == "--b-cpuct" && v) { g.bCpuct = std::atof(v); }
        else if (k == "--b-pretrain" && v) { g.bPretrain = std::atoi(v); }
        else if (k == "--b-pretrain-epochs" && v) { g.bPretrainEpochs = std::atoi(v); }
        else if (k == "--b-pretrain-trunk" && v) { g.bPretrainTrunk = std::atoi(v); }
        else if (k == "--b-lr" && v) { g.bLr = std::atof(v); }
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
    double prior = 1.0;
    double cpuct = 1.5;
    int epochs = 1;
    int every = 8;
    bool lfs = true;
    /* 手工锚预训练 (0 = 不做): 局面数 / epoch 数 / 是否连骨干一起训 */
    int pretrain = 0;
    int pretrainEpochs = 8;
    bool pretrainTrunk = false;
    double lr = 0.001;
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
    ag.priorTemp = (float)a.prior;
    ag.c_puct = (float)a.cpuct;
    ag.learningRate = (float)a.lr;
    ag.replayEpochs = a.epochs;
    ag.learnEveryMoves = a.every;
    ag.learnFromSearch = a.lfs;
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
    a.sims = g.sims;
    a.prior = g.prior;
    a.cpuct = g.cpuct;
    a.epochs = g.epochs;
    a.every = g.every;
    a.lfs = (g.lfs != 0);
    a.pretrain = g.pretrain;
    a.pretrainEpochs = g.pretrainEpochs;
    a.pretrainTrunk = (g.pretrainTrunk != 0);
    return a;
}

/*
 *  臂 B = 臂 A 原样, 再把命令行上显式给的那几项覆盖掉 (-b-* 那一组)。
 *  为什么要这样而不是"自动取反": 优化实验必须**一次只改一个变量**, 否则得不出
 *  "是哪一项起了作用" (第一轮 agree 模式就是一次翻四项, 只能读成"这一组的效应")。
 */
static Arm armBFromCfg(const Arm &base)
{
    Arm b = base;
    b.name = "B";
    if (g.bTwin >= 0) { b.twin = (g.bTwin != 0); }
    if (g.bSparse >= 0) { b.sparse = (g.bSparse != 0); }
    if (g.bDense >= 0) { b.dense = (g.bDense != 0); }
    if (g.bClamp != -999) { b.clampI = g.bClamp; }
    if (g.bHuber >= 0.0) { b.huber = g.bHuber; }
    if (g.bDbl >= 0) { b.dbl = (g.bDbl != 0); }
    if (g.bTau >= 0.0) { b.tau = g.bTau; }
    if (g.bTauIter >= 0) { b.tauIter = g.bTauIter; }
    if (g.bSims >= 0) { b.sims = g.bSims; }
    if (g.bPrior >= 0.0) { b.prior = g.bPrior; }
    if (g.bCpuct >= 0.0) { b.cpuct = g.bCpuct; }
    if (g.bEpochs >= 0) { b.epochs = g.bEpochs; }
    if (g.bEvery >= 0) { b.every = g.bEvery; }
    if (g.bLfs >= 0) { b.lfs = (g.bLfs != 0); }
    if (g.bPretrain >= 0) { b.pretrain = g.bPretrain; }
    if (g.bPretrainEpochs >= 0) { b.pretrainEpochs = g.bPretrainEpochs; }
    if (g.bPretrainTrunk >= 0) { b.pretrainTrunk = (g.bPretrainTrunk != 0); }
    if (g.bLr >= 0.0) { b.lr = g.bLr; }
    return b;
}

/* 手工锚预训练 (按臂的设置跑一次, 并把两个读数打出来) */
static void maybePretrain(DQNMCTSMOETbAgent &ag, const Arm &a)
{
    if (a.pretrain <= 0) {
        return;
    }
    const DQNMCTSMOETbAgent::HandPretrainStats st =
        ag.pretrainQFromHand(a.pretrain, a.pretrainEpochs, a.pretrainTrunk);
    std::printf("        [手工锚] %s: 探针 %d | 监督 %lld 列 | gap %.4f -> %.4f |"
                " argmax 一致率 %.1f%% -> %.1f%% | %.1f s (trunk=%d)\n",
                a.name.c_str(), st.probes, st.samples, st.gapBefore, st.gapAfter,
                100.0 * st.agreeBefore, 100.0 * st.agreeAfter, st.ms / 1000.0,
                (int)a.pretrainTrunk);
}

static void printArm(const char *tag, const Arm &a)
{
    std::printf("        %s: sims=%d twin=%d sparse=%d dense=%d clamp=%d huber=%.3g"
                " doubleDQN=%d tau=%.3g it=%d prior=%.3g cpuct=%.3g epochs=%d every=%d lfs=%d\n",
                tag, a.sims, (int)a.twin, (int)a.sparse, (int)a.dense, a.clampI,
                a.huber, (int)a.dbl, a.tau, a.tauIter, a.prior, a.cpuct, a.epochs,
                a.every, (int)a.lfs);
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
 *  开局集: **一次生成, 每局重用** —— 这是"配对比较"的全部机制
 * ============================================================
 *  为什么不能像第一版那样"每局现抽开局": 抽开局消耗 RL::Random, 而两臂跑的是两份
 *  独立的 agent 与独立的对局流 (MCTS 对手还有自己的随机流), 一旦某一臂多走/少走几手,
 *  后面的开局就**不是同一套**了 —— 那时"两臂的得分率之差"里混着"开局不同"的成分。
 *  先固定 K 条开局, 第 i 局用第 i%K 条, 于是"同一局号 = 同一开局 + 同一先后手"。
 */
struct Opening {
    std::vector<Step> steps;
};

static std::vector<Opening> buildOpenings(int count, int plies, unsigned seed)
{
    std::vector<Opening> out;
    if (count <= 0) {
        return out;
    }
    RL::Random::setSeed(seed);   /* 开局集本身也要可复现 */
    Chess c;
    for (int i = 0; i < count; i++) {
        c.reset();
        int turn = Stone::COLOR_RED;
        Opening op;
        for (int k = 0; k < plies; k++) {
            if (c.getResult(turn) != Chess::RESULT_ONGOING) {
                break;
            }
            std::vector<Step*> legal;
            c.sample(turn, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                break;
            }
            std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
            const Step chosen = *legal[(std::size_t)pick(RL::Random::engine)];
            Steps::instance().put(legal);
            double dummy = 0.0;
            Step mv = chosen;
            c.moveForward(&mv, dummy);
            op.steps.push_back(chosen);
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }
        out.push_back(op);
    }
    return out;
}

/* 把棋盘摆到某条开局的终点 (返回轮到谁走) */
static int applyOpening(Chess &c, const Opening &op)
{
    c.reset();
    int turn = Stone::COLOR_RED;
    for (std::size_t i = 0; i < op.steps.size(); i++) {
        double dummy = 0.0;
        Step mv = op.steps[i];
        c.moveForward(&mv, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    return turn;
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
    double matAgent = 0.0;    /* 终局时 (走棋方中"我方") 的子力 − 对手的子力 */
    bool adjudicated = false; /* 是"上限到时按子力判"的, 不是真的将死 */
};

/*
 *  子力差 (不含将): 与 `ChessState::materialPhase` 同一口径 —— 将的价值是"被吃即终局",
 *  把它算进子力差会让"被将死"变成"差了一个无穷大"。
 */
static double materialOf(Chess &c, int color)
{
    double sum = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[(std::size_t)i];
        if (s == nullptr || !s->alive || s->type == Stone::TYPE_JIANG) {
            continue;
        }
        if (s->color == color) {
            sum += s->value;
        }
    }
    return sum;
}

static GameStat playGame(Chess &c, DQNMCTSMOETbAgent &ag, ABAgent *ab, MCTS *mcts,
                         const Opening &op, bool aiIsRed, int sims, unsigned oppSeed)
{
    GameStat gs;
    /*
       ⚠ 对手的随机流必须**每局重新播种**, 而且要按 (局号, 臂) 定, 不能顺其自然:
       `MCTS` 用 `std::rand()` 做随机回放, 那是**进程级的一条流**。两臂共用一个 MCTS
       对象 ⇒ 如果只是"接着往下抽", 后下的那一臂拿到的骰子位置就取决于先下的那一臂
       走了多少手/树长什么样 —— 那是一个**与臂本身无关但与臂的顺序相关**的系统误差。
       实测过它的量级: 同一个配置 (600x20 的锚) 在两次运行里量到 **43.8%** 与 **87.5%**。
       这里改成"每局一个确定的种子" ⇒ 每局对手的骰子与其它局、其它臂都独立, 而且
       同一条命令仍然可复现。
    */
    if (mcts != nullptr) {
        std::srand(oppSeed);
    }
    int turn = applyOpening(c, op);
    (void)turn;
    while (gs.plies < g.maxPlies) {
        const int res = c.getResult(c.sideToMove);
        if (res != Chess::RESULT_ONGOING) {
            gs.winner = res;
            gs.reason = (res == Chess::RESULT_DRAW) ? "draw(rule)" : "mate";
            break;
        }
        const int side = c.sideToMove;
        const bool aiTurn = ((side == Stone::COLOR_RED) == aiIsRed);
        const std::string before = digest(c);
        const double t0 = nowMs();
        Step s;
        if (aiTurn) {
            const bool saved = ag.learnFromSearch;
            ag.learnFromSearch = (g.train != 0);
            s = ag.selectMove(side, sims, 0.0f);
            ag.learnFromSearch = saved;
            gs.ms += nowMs() - t0;
            gs.moves++;
        } else if (mcts != nullptr) {
            s = mcts->getBestMove(side);
        } else {
            s = ab->getBestMove(side);
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
        if (!c.isLegalMove(side, &s)) {
            gs.illegal = true;
            gs.reason = aiTurn ? "AI played an ILLEGAL move" : "opponent played ILLEGAL";
            break;
        }
        double dummy = 0.0;
        Step mv = s;
        c.moveForward(&mv, dummy);
        gs.plies++;
    }
    if (gs.winner == Chess::RESULT_ONGOING && !gs.aborted && !gs.illegal) {
        gs.winner = Chess::RESULT_DRAW;
        gs.reason = "ply limit";
    }
    /*
       子力差: **这是本协议下最有判别力的读数**。实测过 16 局 MCTS-800、80 手上限时
       全部是和棋 (0 胜 0 负 16 和) —— 那时 W/D/L 的分辨力是 0, 而"终局时的子力差"是
       连续的, 且正是这类 agent 唯一能学到的东西 (终局样本极稀疏)。
       `g.adj` 打开时按子力**裁定**胜负 (标准比赛做法: 超出阈值算优势方赢), 阈值 0.25
       ≈ 一个马/炮的六成 —— 小于"一个子"的差别算和。
    */
    gs.matAgent = materialOf(c, aiIsRed ? Stone::COLOR_RED : Stone::COLOR_BLACK)
                  - materialOf(c, aiIsRed ? Stone::COLOR_BLACK : Stone::COLOR_RED);
    if (g.adj != 0 && gs.winner == Chess::RESULT_DRAW && gs.reason != nullptr
        && std::strcmp(gs.reason, "ply limit") == 0 && !gs.aborted && !gs.illegal) {
        if (gs.matAgent > 0.25) {
            gs.winner = aiIsRed ? Chess::RESULT_RED_WIN : Chess::RESULT_BLACK_WIN;
            gs.adjudicated = true;
            gs.reason = "material adjudication (win)";
        } else if (gs.matAgent < -0.25) {
            gs.winner = aiIsRed ? Chess::RESULT_BLACK_WIN : Chess::RESULT_RED_WIN;
            gs.adjudicated = true;
            gs.reason = "material adjudication (loss)";
        }
    }
    return gs;
}

/* 一局的结果换算成"这一臂的得分" (胜 1 / 和 0.5 / 负 0) */
static double gameScore(const GameStat &gs, bool aiIsRed)
{
    if (gs.winner == Chess::RESULT_DRAW) {
        return 0.5;
    }
    const bool redWon = (gs.winner == Chess::RESULT_RED_WIN);
    return (redWon == aiIsRed) ? 1.0 : 0.0;
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
    const Arm arm = armFromCfg("arm");
    applyArm(ag, arm);
    if (!g.loadPrefix.empty()) {
        if (!ag.loadModel(g.loadPrefix)) {
            std::printf("[bench] 载入权重失败: %s\n", g.loadPrefix.c_str());
            return 2;
        }
    }
    maybePretrain(ag, arm);
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
    /*
       MCTS 的构造函数里会 `std::srand(time(nullptr))` (它用 std::rand 做随机回放),
       所以**构造完对手之后**必须再钉一次全局 C 随机流, 否则同一命令跑两次的对手
       是两副牌 —— 本工程记录过这个坑 ("没有这一行, 那一轮结论是对手的随机流造出来的")。
       注意 std::rand 与 RL::Random 是两条独立的流: 钉这条不影响 agent 那边。
    */
    std::srand(g.seed);

    const std::vector<Opening> openings = buildOpenings(g.openings, g.openingPlies, g.seed);
    RL::Random::setSeed(g.seed);   /* 开局集生成后重新钉一次, 让 agent 那边的流从头开始 */

    std::printf("[bench] 对局: %d 局 | 每次 %d 次模拟 | 对手 %s | 开局集 %d 条 x %d 手 |"
                " 上限 %d 手 | twin=%d sparse=%d doubleDQN=%d clamp=%d dense=%d train=%d\n",
                g.games, g.sims, g.opp.c_str(), (int)openings.size(), g.openingPlies,
                g.maxPlies, (int)ag.twinCritic, (int)ag.sparseLeafEval, (int)ag.doubleDQN,
                g.clampI, (int)g.dense, g.train);

    int wins = 0, losses = 0, draws = 0, aborted = 0, illegal = 0;
    double msSum = 0.0, matSum = 0.0;
    long long moves = 0;
    for (int i = 0; i < g.games; i++) {
        const bool aiIsRed = (i % 2 == 0);
        const Opening &op = openings[(std::size_t)(i % (int)openings.size())];
        const GameStat gs = playGame(c, ag, ab, mcts, op, aiIsRed, g.sims,
                                     g.seed ^ (0x9E3779B9u * (unsigned)(i * 2 + 1)));
        if (gs.aborted || gs.illegal) {
            aborted++;
            std::printf("  局 %3d: **%s** (%s)\n", i + 1,
                        gs.illegal ? "非法着法" : "中止", gs.reason);
            continue;
        }
        msSum += gs.ms;
        moves += gs.moves;
        matSum += gs.matAgent;
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
    std::printf("[bench] 每步 %.1f ms | 平均子力差 %+.3f | 池 %d 条 | 更新 %d 次 |"
                " |Q|均 %.4f | 夹住 %.1f%%\n",
                moves > 0 ? msSum / (double)moves : 0.0,
                decided > 0 ? matSum / (double)decided : 0.0, ag.samplePoolSize(),
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

/* ============================================================
 *  mode = arena2: **配对**比较两个臂
 * ============================================================
 *  两臂用**同一批开局、同一局号同样的先后手**, 各自与 MCTS-N 下一局, 于是每一对
 *  (i 局) 给出两个得分 (胜1/和0.5/负0) —— 配对差 d_i = score_B(i) − score_A(i)。
 *
 *  为什么必须配对: 本工程实测过, 同一个配置换一副开局 (同种子不同流) 就有 ±10~25 个点的
 *  抖动, 而未配对的 SE 约 7.8%、配对之后约 4.0%。要判一个旋钮值不值, 只能看配对差。
 *
 *  两臂**载入同一份权重** (先 A 存盘、B 载入): 否则"谁更强"里混着"两份随机初始化"的差别。
 *  这一点是本轮真踩到的 —— 第一版 agree 模式没钉种子, 同一条命令给出过 0% 与 100%
 *  两个相反结果, 差别全在初始权重上。
 */
static int runArena2()
{
    const Arm armA = armFromCfg("A");
    const Arm armB = armBFromCfg(armA);
    const int simsA = (armA.sims > 0) ? armA.sims : g.sims;
    const int simsB = (armB.sims > 0) ? armB.sims : g.sims;

    RL::Random::setSeed(g.seed);
    Chess c;
    c.reset();
    DQNMCTSMOETbAgent a1(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, armA.dense);
    applyArm(a1, armA);
    if (!g.loadPrefix.empty() && !a1.loadModel(g.loadPrefix)) {
        std::printf("[bench] 载入权重失败: %s\n", g.loadPrefix.c_str());
        return 2;
    }
    DQNMCTSMOETbAgent a2(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, armB.dense);
    if (!g.loadPrefix2.empty()) {
        /*
           臂 B 用**另一份权重** ⇒ 这时比的是"两份权重谁强" (例如 训练前 vs 训练后),
           旋钮按各自的臂设置。两臂仍然共用同一批开局与同样的先后手 ⇒ 配对成立。
        */
        if (!a2.loadModel(g.loadPrefix2)) {
            std::printf("[bench] 臂 B 载入权重失败: %s\n", g.loadPrefix2.c_str());
            return 2;
        }
    } else {
        /* 两臂同权重: 先存盘再载入, 于是差别只来自旋钮 */
        const std::string p = "weights/_bench_dqnmcts_moe_pair";
        if (!a1.saveModel(p)) {
            std::printf("[bench] 无法写出临时权重 (weights/ 可写?)\n");
            return 2;
        }
        if (!a2.loadModel(p)) {
            std::printf("[bench] 臂 B 载入同一份权重失败\n");
            return 2;
        }
        std::remove((p + "_trunk").c_str());
        std::remove((p + "_q").c_str());
        std::remove((p + "_q2").c_str());
    }
    applyArm(a2, armB);
    /*
       预训练放在**两臂的权重都就位之后**: 这样"A 锚了、B 没锚"才是一个干净的对照
       (否则 B 会通过 save/load 继承到 A 锚过的那份权重)。
    */
    maybePretrain(a1, armA);
    maybePretrain(a2, armB);

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
    std::srand(g.seed);
    const std::vector<Opening> openings = buildOpenings(g.openings, g.openingPlies, g.seed);

    std::printf("[bench] 配对对局: %d 局 (开局集 %d 条 x %d 手, 同一局号两臂同开局同先后手) |"
                " 对手 %s (%d 次模拟) | 上限 %d 手 | 子力裁定 %s\n",
                g.games, (int)openings.size(), g.openingPlies, g.opp.c_str(), g.oppSims,
                g.maxPlies, g.adj ? "开" : "关");
    printArm("A", armA);
    printArm("B", armB);

    int wA = 0, dA = 0, lA = 0, wB = 0, dB = 0, lB = 0;
    double sumA = 0.0, sumB = 0.0, sumDiff = 0.0, sumDiff2 = 0.0;
    double matA = 0.0, matB = 0.0, matDiff = 0.0, matDiff2 = 0.0;
    int pairs = 0, aborted = 0;
    for (int i = 0; i < g.games; i++) {
        const bool aiIsRed = (i % 2 == 0);
        const Opening &op = openings[(std::size_t)(i % (int)openings.size())];
        const GameStat gA = playGame(c, a1, ab, mcts, op, aiIsRed, simsA,
                                     g.seed ^ (0x9E3779B9u * (unsigned)(i * 2 + 1)));
        const GameStat gB = playGame(c, a2, ab, mcts, op, aiIsRed, simsB,
                                     g.seed ^ (0x9E3779B9u * (unsigned)(i * 2 + 2)));
        if (gA.aborted || gA.illegal || gB.aborted || gB.illegal) {
            aborted++;
            std::printf("  局 %3d: **中止** A<%s> B<%s>\n", i + 1, gA.reason, gB.reason);
            continue;
        }
        const double sA = gameScore(gA, aiIsRed);
        const double sB = gameScore(gB, aiIsRed);
        const double d = sB - sA;
        const double md = gB.matAgent - gA.matAgent;
        sumA += sA;
        sumB += sB;
        sumDiff += d;
        sumDiff2 += d * d;
        matA += gA.matAgent;
        matB += gB.matAgent;
        matDiff += md;
        matDiff2 += md * md;
        pairs++;
        if (sA > 0.75) { wA++; } else if (sA < 0.25) { lA++; } else { dA++; }
        if (sB > 0.75) { wB++; } else if (sB < 0.25) { lB++; } else { dB++; }
        if (!g.quiet) {
            std::printf("  局 %3d: AI执%s | A %s (%3d 手, 子力 %+.2f) | B %s (%3d 手,"
                        " 子力 %+.2f) | 配对差 分%+.1f 子力%+.2f\n",
                        i + 1, aiIsRed ? "红" : "黑",
                        (sA > 0.75) ? "胜" : (sA < 0.25 ? "负" : "和"), gA.plies,
                        gA.matAgent,
                        (sB > 0.75) ? "胜" : (sB < 0.25 ? "负" : "和"), gB.plies,
                        gB.matAgent, d, md);
        }
    }

    if (pairs == 0) {
        std::printf("[bench] 没有一对有效对局\n");
        delete ab;
        delete mcts;
        return 1;
    }
    const double mA = sumA / (double)pairs;
    const double mB = sumB / (double)pairs;
    const double md = sumDiff / (double)pairs;
    /* 配对差的样本标准差 -> 均值的标准误 (这就是"配对"买到的那点灵敏度) */
    double var = 0.0;
    if (pairs > 1) {
        var = (sumDiff2 - (double)pairs * md * md) / (double)(pairs - 1);
    }
    const double se = (var > 0.0) ? std::sqrt(var / (double)pairs) : 0.0;
    const double t = (se > 1e-12) ? (md / se) : 0.0;

    /* 子力差也要配对统计: W/D/L 在和棋占多数时几乎分辨不出东西, 子力差是连续的 */
    const double mmd = matDiff / (double)pairs;
    double mvar = 0.0;
    if (pairs > 1) {
        mvar = (matDiff2 - (double)pairs * mmd * mmd) / (double)(pairs - 1);
    }
    const double mse = (mvar > 0.0) ? std::sqrt(mvar / (double)pairs) : 0.0;
    const double mt = (mse > 1e-12) ? (mmd / mse) : 0.0;

    std::printf("\n[bench] A: %d胜 %d和 %d负  得分率 %.1f%% | 平均子力差 %+.3f\n",
                wA, dA, lA, 100.0 * mA, matA / (double)pairs);
    std::printf("[bench] B: %d胜 %d和 %d负  得分率 %.1f%% | 平均子力差 %+.3f\n",
                wB, dB, lB, 100.0 * mB, matB / (double)pairs);
    std::printf("[bench] 配对差 (B−A) = %+.1f 个百分点 | SE %.1f 个点 | t = %.2f | n=%d 对\n",
                100.0 * md, 100.0 * se, t, pairs);
    std::printf("[bench] 配对子力差 (B−A) = %+.3f | SE %.3f | t = %.2f  <-- 主要判据\n",
                mmd, mse, mt);
    if (std::fabs(mt) < 2.0 && std::fabs(t) < 2.0) {
        std::printf("[bench] 判读: 两个判据 |t| 都 < 2 ⇒ **不显著** —— 这条差异还不足以\n"
                    "        改默认值 (继续加局数, 或换一个更可能有效应的变量)。\n");
    } else {
        std::printf("[bench] 判读: 至少一个判据 |t| >= 2 ⇒ 在这一批开局上显著"
                    " (仍需换一批开局/换种子复核)。\n");
    }
    if (aborted > 0) {
        std::printf("[bench] ⚠ 有 %d 局中止 (不计入比分)\n", aborted);
    }
    delete ab;
    delete mcts;
    return (aborted > 0) ? 1 : 0;
}

/* ============================================================
 *  mode = train: 自对弈训练 (把权重训出来, 再拿去 arena / arena2 量)
 * ============================================================
 *  为什么训练与评估必须**分开跑**: `--train=1` 的 arena 会让两臂的权重各自漂移,
 *  比分就不再可比 (评估对局的定义就是"冻结之后量")。所以:
 *      train 模式 -> 存权重 ->  arena2 模式载入同一份权重 -> 只改旋钮
 *  训练过程中的四个读数 (learnSteps / loss / |Q|均 / 夹住比例 / done 样本) 直接打出来,
 *  它们是"到底有没有在学"的唯一现场证据 —— 只看损失曲线是分辨不出"学不动"的。
 * ============================================================ */
static int runTrain()
{
    const Arm armA = armFromCfg("train");
    RL::Random::setSeed(g.seed);
    Chess c;
    c.reset();
    DQNMCTSMOETbAgent ag(c, 64, 0.99f, 0.001f, 1.0f, 1.5f, armA.dense);
    applyArm(ag, armA);
    ag.learnFromSearch = true;   /* 训练必须真的写样本, 否则池子是空的 */
    if (!g.loadPrefix.empty() && !ag.loadModel(g.loadPrefix)) {
        std::printf("[bench] 载入权重失败: %s\n", g.loadPrefix.c_str());
        return 2;
    }
    maybePretrain(ag, armA);

    std::printf("[bench] 自对弈训练: %d 局 x %d 次模拟 | 上限 %d 手 | 每 %d 局快照\n",
                g.games, g.sims, g.maxPlies, g.snapshotEvery);
    printArm("train", armA);
    const double t0 = nowMs();
    int doneGames = 0;
    for (int i = 0; i < g.games; i++) {
        ag.trainSelfPlay(1, g.sims, g.maxPlies, false);
        doneGames++;
        const double elapsed = (nowMs() - t0) / 1000.0;
        const double n = (ag.trainDiag.n > 0) ? (double)ag.trainDiag.n : 1.0;
        std::printf("  局 %3d | learn=%d | loss=%.5f | |y|均 %.4f 夹住 %.1f%% |"
                    " |Q|均 %.4f | Qspread %.4f | done %lld 分胜负 %lld | 池 %d | %.0f s\n",
                    i + 1, ag.getLearnSteps(), ag.getLastTrainLoss(),
                    ag.trainDiag.yPreAbsSum / n,
                    100.0 * (double)ag.trainDiag.clamped / n,
                    ag.trainDiag.qAbsMeanSum / n,
                    ag.trainDiag.qSpreadSum / n,
                    ag.trainDiag.doneSamples, ag.trainDiag.decisiveSamples,
                    ag.samplePoolSize(), elapsed);
        if (g.snapshotEvery > 0 && !g.savePrefix.empty()
            && ((i + 1) % g.snapshotEvery == 0)) {
            const std::string p = g.savePrefix + "_g" + std::to_string(i + 1);
            if (ag.saveModel(p)) {
                std::printf("        快照 -> %s (3 个文件)\n", p.c_str());
            } else {
                std::printf("        **快照写盘失败**: %s\n", p.c_str());
            }
        }
    }
    if (!g.savePrefix.empty()) {
        if (!ag.saveModel(g.savePrefix)) {
            std::printf("[bench] **最终权重写盘失败**: %s\n", g.savePrefix.c_str());
            return 2;
        }
        std::printf("[bench] 权重 -> %s (3 个文件) | 共 %d 局 | %.0f s\n",
                    g.savePrefix.c_str(), doneGames, (nowMs() - t0) / 1000.0);
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv)) {
        return 2;
    }
    int rc = -1;
    if (g.mode == "arena") {
        rc = runArena();
    } else if (g.mode == "arena2") {
        rc = runArena2();
    } else if (g.mode == "agree") {
        rc = runAgree();
    } else if (g.mode == "train") {
        rc = runTrain();
    }
    if (rc < 0) {
        std::printf("[bench] 未知模式: %s (用 agree / arena / arena2 / train)\n",
                    g.mode.c_str());
        return 2;
    }
    return rc;
}
