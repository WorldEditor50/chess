/*
 * bench_ppo_mt_main.cpp - PPO+MCTS **多线程分身**自对弈训练的吞吐基准 (**不进 ctest**)
 * ============================================================================
 *
 * 它量的是"单位时间能产出并消化多少训练样本", 不是棋力。
 *
 * 三个数字, 缺一个就说不清:
 *   1. **串行基线**: 单线程 agent 跑同样的局数 (搜索与学习串行), 这是加速比的分母。
 *      基线用与 learner 完全相同的学习配置 (同一个 batch / epochs / lr), 否则
 *      "加速比"里会混进"学习配置不同"的影响。
 *   2. **worker 数扫描**: 1 / 2 / 4 / 6 / 8 条 worker, 同样总局数, 报 games/s 与
 *      samples/s, 以及相对基线的加速比和**并行效率** (加速比 / worker 数)。
 *   3. **学习是否真的发生**: 每轮前后对同一个固定局面取 V(s) 与策略 top-1,
 *      报变化量; 再报 learner 实际做的优化器更新轮数与"样本数 / batch"的理论上限
 *      之比 —— 比值明显小于 1 说明 learner 喂不饱 (它是瓶颈), 接近 1 说明 worker
 *      侧是瓶颈。
 *
 * 为什么 worker 数超过物理核还可能变快: 搜索是访存密集 + 有共享 L3, 而 learner 那条
 * 流水线大部分时间在等样本 (sleep 1ms), 超订不是纯浪费。但超过物理核之后
 * 加速比必然走平 —— 表里"并行效率"往下掉就是它的表现。
 *
 * 为什么**不进 ctest**: 它跑真实自对弈、依赖线程调度与随机开局, 耗时以分钟计,
 * 结果在几局之内抖动是常事 (同 bench_moe / bench_ppo_vs_ab)。它只负责量。
 *
 * 用法:
 *   bench_ppo_mt.exe [--games=8] [--sweep=1,2,4,6,8] [--sims=80] [--moves=120]
 *                    [--batch=64] [--epochs=2] [--lr=0.001] [--publish=1]
 *                    [--repeat=1] [--no-serial] [--no-sparse] [--seed=N]
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "ppo_selfplay_mt.hpp"
#include "rl/util.hpp"

/*
  同 bench_ppo_vs_ab: 刻意**不**写 `using namespace RL;` —— `RL::Step`(一条经验) 与
  `::Step`(象棋的一步) 同名, 拉进全局作用域会让每个 `Step` 变成歧义符号 (C2872)。
*/

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    long long games    = 8;        /* 每个配置跑多少局自对弈 (worker 分摊) */
    std::vector<int> workerList;   /* --sweep=1,2,4,6,8 */
    int sims           = 80;       /* 每次决策的模拟次数 */
    int moves          = 120;      /* 每局手数上限 */
    int batch          = 64;       /* learner 批大小 */
    int epochs         = 2;        /* 每批过几遍 */
    float lr           = 0.001f;
    int publishEvery   = 1;        /* 每几轮学习发布一次权重 */
    int repeat         = 1;        /* 每个配置重复几次 (取吞吐均值, 抗抖动) */
    bool serial        = true;     /* 是否跑串行基线 (加速比的分母) */
    bool mirror        = true;     /* 左右镜像数据增广 (P6) */
    bool shaping       = true;     /* 势能塑形 (Phase 2); --no-shaping 做消融 */
    /*
       R1 A/B: --no-sparse 让 master/worker 都退回"全量策略 + 原始 p_full 先验"的
       改动前口径。同一个进程里跑两次 (--no-sparse / 默认) 才是在同一机器状态下
       量 R1 对聚合吞吐的影响 —— 前后各跑一次会被机器负载的漂移污染。
    */
    bool sparse        = true;
    std::string savePrefix;        /* 非空: 训练结束保存权重 (供对局验证) */
    unsigned seed      = 20240914;
};

Cfg g_cfg;

/* ============================================================
 *  计时
 * ============================================================ */
double nowSec()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e9;
}

/* ============================================================
 *  固定探针局面: 只用来"看权重有没有变", 不参与训练
 *
 *  用一个一次性的 Chess + 只推理的 PPOMCTSAgent 编码 —— encodeState 是成员函数,
 *  视角取自 chess.sideToMove, 所以这里显式把 sideToMove 定成 RED 让探针固定。
 * ============================================================ */
RL::Tensor makeProbeState()
{
    Chess probeEnv;
    probeEnv.sideToMove = Stone::COLOR_RED;
    PPOMCTSAgent encoder(probeEnv, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                         /*withGrad=*/false);
    RL::Tensor s(PPOMCTSAgent::STATE_DIM, 1);
    s.zero();
    encoder.encodeState(s);
    return s;
}

/* 一个配置的吞吐结果 (串行基线与 MT 各点共用的字段) */
struct Throughput {
    int    workers = 0;
    double gamesPerSec = 0.0;
    double samplesPerSec = 0.0;
    double msPerSim = 0.0;
    double seconds = 0.0;
    long long games = 0;
    long long samples = 0;
};

/*
 *  串行基线: 单线程 agent, 搜索与学习串行 (trainSelfPlay -> commitEpisode -> 学习)。
 *  学习配置与 MT 的 learner 完全一致。
 */
Throughput runSerialBaseline()
{
    Throughput r;
    r.workers = 1;

    RL::Random::setSeed(g_cfg.seed);

    Chess env;
    PPOMCTSAgent agent(env, 64, 0.99f, g_cfg.lr, 1.414f, 64, 0.1f, /*withGrad=*/true);
    agent.replayBatchSize = g_cfg.batch;
    agent.replayEpochs    = g_cfg.epochs;
    agent.mirrorAugment   = g_cfg.mirror;
    agent.sparsePolicyHead = g_cfg.sparse;
    agent.ppo.clearReplay();

    const long long samplesBefore = 0;
    const double t0 = nowSec();
    agent.trainSelfPlay((int)g_cfg.games, g_cfg.sims, g_cfg.moves,
                        /*verbose=*/false, 1.0f, 0.1f);
    const double t1 = nowSec();

    r.seconds = t1 - t0;
    r.games   = (long long)agent.getTotalEpisodes();
    /*
      串行路径每个样本进池后不一定立刻被学 (要攒够 batch), 所以用"进池总数"作为
      产出量的口径 —— 与 MT 那边 m_samplesProduced 同口径。
      池子里剩下的算进来: 池容量足够大 (20000), 一条没丢。
    */
    r.samples = samplesBefore + (long long)agent.ppo.replaySize();
    /*
      replaySize 只是"还没被采样过的下限" —— learnFromReplay 不消费样本 (可重复过),
      所以这里报的是"至少产出了这么多"。真实产出按每局手数估更准, 见下面的估算。
    */
        r.gamesPerSec   = r.seconds > 0.0 ? (double)r.games / r.seconds : 0.0;
        r.samplesPerSec = r.seconds > 0.0 ? (double)r.samples / r.seconds : 0.0;
        /* 基线是单线程, "每线程每模拟" 就是"每模拟" */
        double plies = (double)r.samples;
        if (g_cfg.mirror) {
            plies *= 0.5;
        }
        const double totalSims = plies * (double)g_cfg.sims;
        r.msPerSim = (totalSims > 0.0 && r.seconds > 0.0)
                         ? (r.seconds * 1e3) / totalSims : 0.0;
        return r;
}

/* worker 数扫描里的一个点 (可能重复多次) */
struct MtPoint {
    int    workers = 0;
    /*
      吞吐口径 (重要): 报的是**墙钟均值反推的吞吐** = 目标局数 / 平均墙钟秒数,
      而不是"每一轮的 局数/秒 再取平均" —— 后者是均值之比的倒数次幂
      (mean(1/t) != 1/mean(t)), 在重复轮次抖动大时会把结果抬得偏高, 第一版就踩了这个坑。
      同理, 因为 worker 是在每局开始前查配额, 实际局数可能超出目标最多 workers-1 局,
      所以"实际局数"单独列出来, 免得把超跑的局数当成加速比。
    */
    double secMean = 0.0;          /* 平均墙钟秒 */
    double secMin = 0.0, secMax = 0.0;
    long long gamesTotal = 0;      /* 各轮实际完成的局数之和 */
    double gamesPerSec = 0.0;      /* (目标局数 / secMean) —— 与基线同口径 */
    double samplesPerSec = 0.0;
    double learnRounds = 0.0;
    double samples = 0.0;
    double weightSyncs = 0.0;
    double msPerSim = 0.0;         /* 每线程每模拟的毫秒数 (只在关闭增广时精确) */
    double valueBefore = 0.0, valueAfter = 0.0;
    int    topBefore = -1, topAfter = -1;
    float  policyL1 = 0.0f;
    std::vector<long long> moeUsage;
    bool   probeValid = false;
};

MtPoint runMtOnce(int workers, const RL::Tensor &probe)
{
    MtPoint p;
    p.workers = workers;

    RL::Random::setSeed(g_cfg.seed + (unsigned)workers * 7919u);

    Chess env;
    PpoSelfPlayMT::Config cfg;
    cfg.workers       = workers;
    cfg.simulations   = g_cfg.sims;
    cfg.maxMoves      = g_cfg.moves;
    cfg.learnBatch    = g_cfg.batch;
    cfg.learnEpochs   = g_cfg.epochs;
    cfg.lr            = g_cfg.lr;
    cfg.publishEveryLearnRounds = g_cfg.publishEvery;
    cfg.seed          = g_cfg.seed;
    cfg.mirrorAugment = g_cfg.mirror;
    cfg.shaping       = g_cfg.shaping;
    cfg.sparsePolicyHead = g_cfg.sparse;

    /* 构造本身也要计时之外的开销: 网络初始化 (~4 M 参数) 不便宜, 但不属于自对弈,
       所以放在计时区间之外。 */
    PpoSelfPlayMT mt(env, cfg);

    /* ---- 训练前: 固定局面的 V(s) 与策略 top-1 ---- */
    RL::Tensor piBefore = mt.master().ppo.action(probe);
    p.valueBefore = mt.master().ppo.value(probe);
    p.topBefore   = piBefore.argmax();

    mt.master().resetMoeUsage();

    const PpoSelfPlayMT::Stats s = mt.run(g_cfg.games);

    RL::Tensor piAfter = mt.master().ppo.action(probe);
    p.valueAfter = mt.master().ppo.value(probe);
    p.topAfter   = piAfter.argmax();
    p.policyL1   = RL::Norm::l1(piBefore, piAfter);
    p.probeValid = true;

    mt.master().moeUsage(p.moeUsage);

    p.secMean     = s.seconds;
    p.gamesTotal  = s.games;
    p.learnRounds = (double)s.learnRounds;
    p.samples     = (double)s.samples;
    p.weightSyncs = (double)s.weightSyncs;

    /*
      每线程每模拟成本: 总模拟次数 = 局数 x 手数 x sims。手数用样本数反推 ——
      增广把样本翻倍, 所以只有关闭增广时 samples 才等于"总手数"。
      这个数才是"多线程到底有没有真的并行起来"的直接证据: 它涨了就说明单线程变慢了。
    */
    double plies = (double)s.samples;
    if (g_cfg.mirror) {
        plies *= 0.5;   /* 每条样本存了两份 (原 + 镜像) */
    }
    const double totalSims = plies * (double)g_cfg.sims;
    p.msPerSim = (totalSims > 0.0 && s.seconds > 0.0)
                     ? (s.seconds * 1e3 * (double)workers) / totalSims
                     : 0.0;

    p.gamesPerSec   = (s.seconds > 0.0) ? (double)g_cfg.games / s.seconds : 0.0;
    p.samplesPerSec = (s.seconds > 0.0) ? (double)s.samples / s.seconds : 0.0;
    return p;
}

/* ================================================================
 *  只训练 -> 保存权重 (供棋力对局验证: bench_ppo_vs_ab --load=<prefix>)
 *
 *  与 runMtOnce 的区别只有一处: 它要**留住**训练后的 agent 好把权重写盘,
 *  而扫描里的那些对象用完即弃。
 * ================================================================ */
int trainAndSave(int workers, const RL::Tensor &probe)
{
    RL::Random::setSeed(g_cfg.seed);

    Chess env;
    PpoSelfPlayMT::Config cfg;
    cfg.workers       = workers;
    cfg.simulations   = g_cfg.sims;
    cfg.maxMoves      = g_cfg.moves;
    cfg.learnBatch    = g_cfg.batch;
    cfg.learnEpochs   = g_cfg.epochs;
    cfg.lr            = g_cfg.lr;
    cfg.publishEveryLearnRounds = g_cfg.publishEvery;
    cfg.seed          = g_cfg.seed;
    cfg.mirrorAugment = g_cfg.mirror;
    cfg.shaping       = g_cfg.shaping;
    cfg.sparsePolicyHead = g_cfg.sparse;

    PpoSelfPlayMT mt(env, cfg);

    const double v0 = mt.master().ppo.value(probe);
    const double t0 = nowSec();
    const PpoSelfPlayMT::Stats s = mt.run(g_cfg.games);
    const double dt = nowSec() - t0;
    const double v1 = mt.master().ppo.value(probe);

    std::printf("  训练完成: %lld 局 / %.1f s (%.3f 局/s), 样本 %lld, 学习轮数 %lld\n",
                s.games, dt, s.gamesPerSec(), s.samples, s.learnRounds);
    std::printf("  权重变化: V(固定局面) %.4f -> %.4f, |dpi|_1 见扫描表\n", v0, v1);

    const bool ok = mt.master().saveModel(g_cfg.savePrefix);
    std::printf("  权重已保存: %s_actor / %s_critic -> %s\n",
                g_cfg.savePrefix.c_str(), g_cfg.savePrefix.c_str(),
                ok ? "成功" : "**失败**");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char *argv[])
{
    /* ---- 解析参数 ---- */
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const std::size_t eq = a.find('=');
        const std::string k = (eq == std::string::npos) ? a : a.substr(0, eq);
        const std::string v = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
        if (k == "--games")        { g_cfg.games = atoll(v.c_str()); }
        else if (k == "--sims")    { g_cfg.sims = atoi(v.c_str()); }
        else if (k == "--moves")   { g_cfg.moves = atoi(v.c_str()); }
        else if (k == "--batch")   { g_cfg.batch = atoi(v.c_str()); }
        else if (k == "--epochs")  { g_cfg.epochs = atoi(v.c_str()); }
        else if (k == "--lr")      { g_cfg.lr = (float)atof(v.c_str()); }
        else if (k == "--publish") { g_cfg.publishEvery = atoi(v.c_str()); }
        else if (k == "--repeat")  { g_cfg.repeat = atoi(v.c_str()); }
        else if (k == "--seed")    { g_cfg.seed = (unsigned)strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--no-serial") { g_cfg.serial = false; }
        else if (k == "--no-mirror") { g_cfg.mirror = false; }
        else if (k == "--no-shaping") { g_cfg.shaping = false; }
        else if (k == "--no-sparse") { g_cfg.sparse = false; }
        else if (k == "--save") { g_cfg.savePrefix = v; }
        else if (k == "--workers") { g_cfg.workerList.push_back(atoi(v.c_str())); }
        else if (k == "--sweep") {
            /* "1,2,4,6,8" -> {1,2,4,6,8} */
            g_cfg.workerList.clear();
            std::size_t pos = 0;
            while (pos <= v.size()) {
                const std::size_t comma = v.find(',', pos);
                const std::string tok = (comma == std::string::npos)
                                            ? v.substr(pos) : v.substr(pos, comma - pos);
                if (!tok.empty()) {
                    g_cfg.workerList.push_back(atoi(tok.c_str()));
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 2;
        }
    }
    if (g_cfg.workerList.empty()) {
        g_cfg.workerList.push_back(1);
        g_cfg.workerList.push_back(2);
        g_cfg.workerList.push_back(4);
        g_cfg.workerList.push_back(6);
        g_cfg.workerList.push_back(8);
    }
    if (g_cfg.games < 1)    { g_cfg.games = 1; }
    if (g_cfg.repeat < 1)   { g_cfg.repeat = 1; }
    if (g_cfg.epochs < 1)   { g_cfg.epochs = 1; }
    if (g_cfg.batch < 1)    { g_cfg.batch = 1; }

    const unsigned hw = std::thread::hardware_concurrency();

    /* ---- 表头: 把每个影响结论的参数都写出来 ---- */
    std::printf("=== PPO+MCTS 多线程分身自对弈 吞吐基准 ===\n");
    std::printf("网络      : state=%d, action=%d, 稀疏 MoE(E=%d, top-%d), 隐层 64/64\n",
                PPOMCTSAgent::STATE_DIM, PPOMCTSAgent::ACTION_DIM,
                RL::PPO::MOE_EXPERTS, RL::PPO::MOE_TOPK);
    std::printf("自对弈    : 每配置 %lld 局, 每局最多 %d 手, %d 次模拟/步, 温度 1.0 -> 0.1\n",
                g_cfg.games, g_cfg.moves, g_cfg.sims);
    std::printf("学习      : batch=%d, epochs=%d, lr=%.4f, 每 %d 轮学习发布一次权重\n",
                g_cfg.batch, g_cfg.epochs, (double)g_cfg.lr, g_cfg.publishEvery);
    std::printf("数据增广  : 左右镜像 %s (P6)\n", g_cfg.mirror ? "开" : "关");
    std::printf("策略前向  : %s (R1: 稀疏=只算合法列, 全量=改动前口径)\n",
                g_cfg.sparse ? "稀疏" : "全量");
    std::printf("硬件      : hardware_concurrency=%u (本机 i7-12650H = 10 核 / 16 逻辑)\n", hw);
    std::printf("重复      : 每个配置 %d 次, 报吞吐均值\n\n", g_cfg.repeat);

    /* ---- 探针 (在跑之前建, 之后不再变) ---- */
    const RL::Tensor probe = makeProbeState();

    /* ---- 0. 只训练并保存 (棋力验证用): 给了 --save 就跳过扫描 ---- */
    if (!g_cfg.savePrefix.empty()) {
        const int w = g_cfg.workerList.empty() ? 4 : g_cfg.workerList.back();
        std::printf("[0] 训练并保存权重: %lld 局, workers=%d, 塑形=%s, 镜像增广=%s\n",
                    g_cfg.games, w, g_cfg.shaping ? "开" : "关",
                    g_cfg.mirror ? "开" : "关");
        std::fflush(stdout);
        const RL::Tensor probe = makeProbeState();
        const int rc = trainAndSave(w, probe);
        std::printf("\nEXIT=%d\n", rc);
        return rc;
    }

    /* ---- 1. 串行基线 ---- */
    double baseGamesPerSec = 0.0;
    double baseSamplesPerSec = 0.0;
    if (g_cfg.serial) {
        std::printf("[1] 串行基线 (单线程, 搜索与学习串行)...\n");
        std::fflush(stdout);
        const Throughput b = runSerialBaseline();
        baseGamesPerSec   = b.gamesPerSec;
        baseSamplesPerSec = b.samplesPerSec;
        std::printf("    %lld 局 / %.1f s = %.3f 局/s, %.1f 样本/s, 每模拟 %.3f ms (池中样本 %lld)\n\n",
                    b.games, b.seconds, b.gamesPerSec, b.samplesPerSec, b.msPerSim, b.samples);
    } else {
        std::printf("[1] 跳过串行基线 (--no-serial)\n\n");
    }

    /* ---- 2. worker 数扫描 ---- */
    std::printf("[2] worker 数扫描 (learner 另占一条线程)\n");
    std::printf("    %-8s %-9s %-10s %-8s %-9s %-10s %-9s %-11s %s\n",
                "workers", "秒(均值)", "局/s", "ms/模拟", "样本/s", "加速比", "并行效率",
                "学习轮数", "局数准确");
    std::printf("    ------------------------------------------------------------------------------------------------\n");
    std::printf("    注: ms/模拟 = 秒 x workers / (总手数 x sims), 即**每线程**的每模拟成本;\n");
    std::printf("        它随 worker 数上升就说明搜索本身在互相拖慢 (访存/带宽), 而不是 learner 的问题。\n");
    std::fflush(stdout);

    std::vector<MtPoint> results;
    for (std::size_t wi = 0; wi < g_cfg.workerList.size(); wi++) {
        const int w = g_cfg.workerList[wi];
        if (w < 1) {
            continue;
        }
        MtPoint acc;
        acc.workers = w;
        acc.secMin = 1e18;
        double gamesPerSecSum = 0.0, samplesPerSecSum = 0.0;
        for (int rep = 0; rep < g_cfg.repeat; rep++) {
            const MtPoint p = runMtOnce(w, probe);
            if (rep == 0) {
                acc = p;                     /* 诊断量取第一轮 (权重变化量不该被平均) */
                acc.secMin = 1e18;
                acc.secMax = 0.0;
                acc.secMean = 0.0;
                acc.gamesTotal = 0;
                acc.learnRounds = 0.0;
                acc.samples = 0.0;
                acc.weightSyncs = 0.0;
                acc.msPerSim = 0.0;
            }
            acc.secMin = std::min(acc.secMin, p.secMean);
            acc.secMax = std::max(acc.secMax, p.secMean);
            acc.secMean += p.secMean;
            acc.gamesTotal += p.gamesTotal;
            acc.learnRounds += p.learnRounds;
            acc.samples += p.samples;
            acc.weightSyncs += p.weightSyncs;
            acc.msPerSim += p.msPerSim;
            gamesPerSecSum += p.gamesPerSec;
            samplesPerSecSum += p.samplesPerSec;
        }
        const double n = (double)g_cfg.repeat;
        acc.secMean /= n;
        acc.learnRounds /= n;
        acc.samples /= n;
        acc.weightSyncs /= n;
        acc.msPerSim /= n;
        /* 主口径: 固定目标局数 / 平均墙钟秒 (与基线同口径, 不受超跑局数影响) */
        acc.gamesPerSec = (acc.secMean > 0.0) ? (double)g_cfg.games / acc.secMean : 0.0;
        acc.samplesPerSec = (acc.secMean > 0.0) ? acc.samples / acc.secMean : 0.0;
        results.push_back(acc);

        const double speedup = (baseGamesPerSec > 0.0) ? acc.gamesPerSec / baseGamesPerSec : 0.0;
        const double eff     = (baseGamesPerSec > 0.0) ? speedup / (double)w : 0.0;
        if (baseGamesPerSec > 0.0) {
            std::printf("    %-8d %-9.1f %-10.3f %-8.3f %-9.3f %-10.2f %-9.0f %-11.1f %s\n",
                        w, acc.secMean, acc.gamesPerSec, acc.msPerSim, acc.samplesPerSec,
                        speedup, eff * 100.0, acc.learnRounds,
                        (acc.gamesTotal == g_cfg.games * g_cfg.repeat) ? "是" : "**超跑**");
        } else {
            std::printf("    %-8d %-9.1f %-10.3f %-8.3f %-9.3f %-10s %-9s %-11.1f %s\n",
                        w, acc.secMean, acc.gamesPerSec, acc.msPerSim, acc.samplesPerSec,
                        "-", "-", acc.learnRounds,
                        (acc.gamesTotal == g_cfg.games * g_cfg.repeat) ? "是" : "**超跑**");
        }
        std::fflush(stdout);
    }

    /* ---- 3. 学习是否真的发生 ---- */
    std::printf("\n[3] 学习链路体检\n");
    for (std::size_t i = 0; i < results.size(); i++) {
        const MtPoint &p = results[i];
        if (!p.probeValid) {
            continue;
        }
        const double possible = (p.samples > 0.0) ? (p.samples / (double)g_cfg.batch) : 0.0;
        const double ratio = (possible > 0.0) ? (p.learnRounds / possible) : 0.0;
        std::printf("    workers=%-2d 秒 %.1f [%.1f..%.1f], 实际局数 %lld, 学习轮数 %.1f / 上限 %.1f"
                    " (样本 %.0f / batch %d) = %.2f, 每线程 %.3f ms/模拟\n",
                    p.workers, p.secMean, p.secMin, p.secMax, p.gamesTotal,
                    p.learnRounds, possible, p.samples, g_cfg.batch, ratio, p.msPerSim);
        std::printf("              策略 top-1 %d -> %d, |dpi|_1 = %.6f, V(s) %.4f -> %.4f, 权重发布 %lld 次\n",
                    p.topBefore, p.topAfter, (double)p.policyL1,
                    p.valueBefore, p.valueAfter, (long long)(p.weightSyncs / (double)g_cfg.repeat));
        if (p.learnRounds < 0.5) {
            std::printf("              ** learner 一次都没学到 —— 检查 replayBatchSize / 样本是否入池 **\n");
        } else if (ratio < 0.5) {
            std::printf("              -> learner 喂不饱 (样本不够它学): 瓶颈在 worker 侧的搜索\n");
        } else if (ratio > 0.95) {
            std::printf("              -> learner 一直在学: 瓶颈在 learner (学习比生成贵), 加 worker 收益有限\n");
        } else {
            std::printf("              -> 两条流水线大体平衡\n");
        }
        /* MoE 专家使用: 全部被用到才说明辅助损失在起作用 */
        if (!p.moeUsage.empty()) {
            long long mx = p.moeUsage[0], sum = 0;
            int unused = 0;
            for (std::size_t e = 0; e < p.moeUsage.size(); e++) {
                mx = std::max(mx, p.moeUsage[e]);
                sum += p.moeUsage[e];
                if (p.moeUsage[e] == 0) { unused++; }
            }
            /* 全 8 个专家都被用到 (unused=0) 才说明负载均衡辅助损失在起作用 ——
               初始状态下只有 4 个专家会被选中, 见 test_sparse_moe。 */
            const double mean = (double)sum / (double)p.moeUsage.size();
            std::printf("              MoE 专家使用 [");
            for (std::size_t e = 0; e < p.moeUsage.size(); e++) {
                std::printf("%s%lld", (e == 0) ? "" : ", ", p.moeUsage[e]);
            }
            std::printf("]  未使用 %d 个, max/mean=%.2f (总 %lld)\n",
                        unused, mean > 0.0 ? (double)mx / mean : 0.0, sum);
        }
    }

    /* ---- 4. 结论 ---- */
    if (baseGamesPerSec > 0.0 && !results.empty()) {        const MtPoint *best = &results[0];
        for (std::size_t i = 1; i < results.size(); i++) {
            if (results[i].gamesPerSec > best->gamesPerSec) {
                best = &results[i];
            }
        }
        std::printf("\n[4] 最优: workers=%d, %.3f 局/s = 基线的 %.2fx (并行效率 %.0f%%)\n",
                    best->workers, best->gamesPerSec, best->gamesPerSec / baseGamesPerSec,
                    100.0 * (best->gamesPerSec / baseGamesPerSec) / (double)best->workers);
        std::printf("    折算训练量: 按此吞吐, 10^6 局自对弈约需 %.1f 小时 (本机, 本配置)\n",
                    1e6 / best->gamesPerSec / 3600.0);
    }
    return 0;
}
