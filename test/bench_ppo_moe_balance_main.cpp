/*
 * bench_ppo_moe_balance_main.cpp — **PPO 这条骨干**的 MoE 负载均衡 A/B (不进 ctest)
 * ============================================================================
 *
 * 回答的问题 (只有两个, 都是可证伪的):
 *
 *   Q1 真实 PPO 训练链路上, 只靠 `moeAuxCoef` (Switch 均衡辅助损失) 能不能把专家负载拉平?
 *      预期 (由 docs/moe_gate_experiment_2026_10.md 的 [3b] 给出): **不能**。
 *      那份实验在同一批真实棋局状态上把 coef 开到 10, MaxVio 仍恒为理论最大值,
 *      有效专家 1.00 —— 根因是辅助损失的梯度**正比于饿死专家自己的门控概率 (≈0)**,
 *      放大系数乘的仍然是 0。本工具要回答的是同一个问题在**生产路径**上(真自对弈、
 *      真梯度、真优化器)是否也成立, 而不是只在合成任务上成立。
 *
 *   Q2 把均衡的作用点从"梯度"换到"top-k 的 argmax 比较项"
 *      (Loss-Free Balancing, arXiv:2408.15664) 之后, 同一条路径上能不能拉平?
 *      预期: 能 —— 而代价必须一起量出来 (路由纯度/训练损失/对局结果), 否则就是
 *      "把负载拉平了但把别的东西搞坏了" (本仓库在 SAC 支上正好踩过这个坑:
 *      `alphaCeiling=0.1` 确实把熵项压下去了, 但 Q 间距同时塌了)。
 *
 * 为什么**必须**另开一个工具, 而不是复用 bench_gate_moe:
 *   bench_gate_moe 的 [3] 刻意绕开了 MCTS 与训练, 只喂固定的 1710 维状态 —— 那让实验
 *   便宜且干净, 但它量的是"路由机制", **不是**"生产训练链路"。本工具反过来: 跑真
 *   自对弈、真 commitEpisode、真 learnFromReplay, 代价高但读数可以直接对应到
 *   agent 实际在下的那盘棋。
 *
 * 口径 (每一条都对应一次踩过的坑)
 * --------------------------------
 * 1. **按前向来源拆开** (训练侧 / 推理侧)。搜索一次决策就是几十次叶子估值, 数量上
 *    压倒训练批 —— 只看合计会把"搜索访问到的局面分布"当成"训练批的路由分布", 于是
 *    判不出均衡机制到底有没有生效 (`RL::PPO::moeUsageSplit`; 与界面 MoE 负载条同口径)。
 * 2. **用 MaxVio + 有效专家, 不用 max/min**。MaxVio = max_i share_i/(1/E) − 1
 *    (与 Loss-Free 论文同口径, 0 = 完美均衡); 有效专家 = 份额 > 5% 的个数。
 *    max/min 对"最弱专家是 1.1% 还是 2.7%"过敏, 容易把噪声当趋势。
 * 3. **四个臂除均衡开关外, 一切相同**: 同一 seed、同一自对弈数据、同样次数的
 *    `learnFromReplay` (优化器步数一致)。每换一个臂就重建 agent, 把门控拨回同一个
 *    初始化 —— 否则臂间差异里混着"初始化不同"这一项。
 * 4. **增量口径**: 每个臂在重建之前调一次 `resetMoeUsage()`, 于是读到的计数**只属于
 *    本臂** (否则后面的臂读到前面几臂的累计, 越跑越"均衡")。
 *
 * 用法:
 *   cmake --build <build> --target bench_ppo_moe_balance
 *   <build>/bench_ppo_moe_balance --quick              # 冒烟 (~1 分钟)
 *   <build>/bench_ppo_moe_balance --games=4 --moves=24 --sims=24 --batches=8
 *   <build>/bench_ppo_moe_balance --only=none,aux      # 只跑两臂
 *
 * 四个臂:
 *   none      : auxCoef=0   + 偏置关   (纯 softmax 反向, 最坏情形, 作为下界)
 *   aux       : auxCoef=0.1 + 偏置关   (现状默认)
 *   auxStrong : auxCoef=10  + 偏置关   ([3b] 里"开到 10 也拉不动"的那一档)
 *   lossfree  : auxCoef=0.1 + 偏置开   (本轮的候选修法; rate 见 --bias-rate)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/ppo.h"
#include "rl/util.hpp"

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    int games = 4;         /* 每臂自对弈局数 (只用来采数据, learnStepsPerEpisode=0) */
    int moves = 24;        /* 每局手数上限 */
    int sims = 24;         /* 每次决策的 MCTS 模拟数 (与棋力无关, 只要搜索在跑) */
    int batches = 8;       /* 每臂 learnFromReplay 的次数 (优化器步数, 四臂必须相同) */
    int batchSize = 64;    /* 每次批学习的样本数 */
    int evalGames = 2;     /* 评测局数 (只推理, 用来看推理侧路由) */
    int expertHidden = 8;  /* MLP 专家的隐层 (小: 本工具量的是路由, 不是容量) */
    unsigned seed = 20240901u;
    float lr = 0.003f;
    float biasRate = 0.01f;   /* 无辅助损失偏置的步长 (论文推荐 0.001, 见 --bias-rate) */
    /*
       三层 MLP 门控的隐层宽度 (d_model -> hidden -> E)。
       `0` = 线性门控 (现状默认)。取 32 而不是 SAC 界面口径里那个 64: 本工具的 d_model
       与 E 都小 (state=1710, E=8), 32 已经比"E 张超平面"表达力强得多, 而门控参数量
       只占骨干的千分之几 (SAC 支实测 +0.28%)。
    */
    int gateHidden = 32;
    /*
       骨干: `false` = MlpExpert (E=8 top-2, 便宜 ~25x, 本工具的默认); `true` = TB 专家
       (**界面现役的 AGENT_PPOMCTS**; E/top-k 取自 `RL::PPO` 的编译期常量 —— 2026-10 起
       是 8/2, 见 docs/moe_gate_experiment_2026_10.md §13)。
       为什么默认用 MLP 专家: 本工具量的是**路由**, 而路由只依赖门控权重与输入分布,
       与专家内部是什么无关 (同 bench_gate_moe 的根据)。运行 TB 专家只是为了让读数
       直接对得上生产配置 —— 代价是分钟级而不是秒级 (207.8 M 参数 / 20 ms 每模拟),
       所以用 `--backbone=tb` 显式打开。
    */
    bool tbExperts = false;
    /* 只跑哪几个臂 (空 = 全跑) */
    std::vector<std::string> onlyTags;
};

Cfg g_cfg;

/* ============================================================
 *  负载读数 (与 bench_gate_moe / 界面 MoE 负载条逐字同口径)
 * ============================================================ */
struct LoadStat {
    double maxVio = 0.0;      /* max_i share_i/(1/E) − 1 ; 0 = 完美均衡 */
    double minShare = 0.0;
    int    effective = 0;     /* 份额 > 5% 的专家个数 */
    long long total = 0;
    std::vector<double> share;
};

LoadStat loadOf(const std::vector<long long> &counts)
{
    LoadStat s;
    const int e = (int)counts.size();
    if (e <= 0) {
        return s;
    }
    for (std::size_t i = 0; i < counts.size(); i++) {
        s.total += counts[i];
    }
    s.share.assign((std::size_t)e, 0.0);
    if (s.total <= 0) {
        return s;
    }
    double mx = 0.0;
    s.minShare = 1.0;
    for (int i = 0; i < e; i++) {
        const double sh = (double)counts[(std::size_t)i] / (double)s.total;
        s.share[(std::size_t)i] = sh;
        if (sh > mx) { mx = sh; }
        if (sh < s.minShare) { s.minShare = sh; }
        if (sh > 0.05) { s.effective++; }
    }
    s.maxVio = mx * (double)e - 1.0;
    return s;
}

std::string shareText(const LoadStat &s)
{
    std::string out = "{";
    for (std::size_t i = 0; i < s.share.size(); i++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * s.share[i]);
        out += buf;
        if (i + 1 < s.share.size()) { out += ", "; }
    }
    out += "}";
    return out;
}

/* ============================================================
 *  一个臂
 * ============================================================ */
struct Arm {
    const char *tag;
    float aux;
    bool  lossFree;
    int   gateHidden;      /* 0 = 线性门控 (默认); >0 = 三层 MLP 门控的隐层 */
    const char *desc;
};

struct ArmResult {
    double trainMaxVio = 0.0, trainMinShare = 0.0;
    int    trainEffective = 0;
    long long trainForwards = 0;
    double batchMeanMaxVio = 0.0;   /* 逐批 MaxVio 的均值 (噪声大, 只看趋势) */
    double inferMaxVio = 0.0, inferMinShare = 0.0;
    int    inferEffective = 0;
    long long inferForwards = 0;
    double collectInferMaxVio = 0.0;   /* 采集期(训练前)的推理侧 MaxVio —— 基线 */
    double biasSpread = 0.0;
    double lossMean = 0.0, actorLossMean = 0.0, valueLossMean = 0.0;
    int    gateHiddenSeen = 0;
    std::vector<double> perBatchMaxVio;
    std::vector<long long> trainUsage;
    double seconds = 0.0;
};

/* 每臂现场打印的逐批轨迹 (用于看"是逐步收敛还是根本不动") */
void printBatchTrace(const std::vector<double> &v)
{
    /*
       两个读数一起给, 因为它们的**判据不同**:
         * 逐批 MaxVio 的**方差天生很大** —— 一批只有 `batchSize x topK` 次路由
           (64x2=128 次选择), 在 E=8 上光是均匀随机就有 ~0.4 的抖动。所以它只能看趋势,
           不能拿单批的数当结论 (这也正是"不用 max/min、要看累计份额"的理由)。
         * 累计 (训练侧合计) 才是收敛后的稳定读数。
    */
    double sum = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < v.size(); i++) {
        sum += v[i];
        if (v[i] > mx) { mx = v[i]; }
    }
    std::printf("      逐批训练侧 MaxVio:");
    for (std::size_t i = 0; i < v.size(); i++) {
        std::printf(" %.3f", v[i]);
    }
    std::printf("\n      逐批统计: 均值 %.3f, 最大 %.3f (%d 批 x %d 样本 x top-%d)\n",
                v.empty() ? 0.0 : sum / (double)v.size(), mx, (int)v.size(),
                g_cfg.batchSize,
                g_cfg.tbExperts ? RL::PPO::MOE_TOPK : RL::PPO::MOE_MLP_TOPK);
}

ArmResult runArm(const Arm &arm, int hidden)
{
    ArmResult R;
    const auto t0 = std::chrono::steady_clock::now();

    /*
       每个臂重建 agent: 门控拨回同一个初始化 (`Random::setSeed` 在同一位置调用),
       于是臂间差异只剩"均衡开关"这一项。
       ⚠ MLP 门控臂会**多消耗随机数** (enableMlpGate 里初始化 wg1/wg2), 所以它与线性臂
       的专家权重不是逐位相同 —— 这是"换门控"本身的代价, 不是本工具的缺陷。同一门控
       结构内部的对比 (如 mlp vs mlpFree) 才是逐位同初始化的。
    */
    RL::Random::setSeed(g_cfg.seed);
    Chess board;
    const RL::PPO::Backbone bb = g_cfg.tbExperts ? RL::PPO::Backbone::TbExperts
                                                 : RL::PPO::Backbone::MlpExperts;
    PPOMCTSAgent ag(board, hidden, 0.99f, g_cfg.lr, 1.414f, g_cfg.expertHidden,
                    arm.aux, true, bb);

    ag.potentialShaping = true;
    ag.materialRewardEnabled = true;
    ag.truncationBootstrap = true;
    ag.rootNoise = true;
    ag.openingPlies = 4;                 /* 打破"每局同一盘棋" */
    ag.openingSeed = g_cfg.seed;
    ag.replayBatchSize = 0;              /* 采集期**不学习** —— 学习节拍由本工具控制 */
    ag.learnStepsPerEpisode = 0;
    ag.setLossFreeBias(arm.lossFree, g_cfg.biasRate);
    /*
       门控结构必须在**任何前向之前**装好 (它会重建并重新初始化门控权重) —— 这里正是
       构造点之后、搜索/训练之前, 是唯一合法的位置。
    */
    if (arm.gateHidden > 0) {
        ag.enableMlpGate(arm.gateHidden);
    }
    R.gateHiddenSeen = ag.mlpGateHidden();
    ag.ppo.moeAuxCoef = arm.aux;
    ag.moeAuxCoef = arm.aux;
    ag.resetMoeUsage();                  /* 读数只属于本臂 */
    ag.ppo.resetCriticDiag();

    /* ---- 阶段 1: 自对弈采数据 (不学习) ---- */
    ag.trainSelfPlay(g_cfg.games, g_cfg.sims, g_cfg.moves, false, 1.0f, 0.25f);
    const std::size_t poolSize = ag.ppo.replaySize();

    /*
       采完数据、开始学习之前**划一刀批边界**。
       为什么必须划: `usageSnapshotSplit` 的口径是"训练侧 = 各批 usageBatch 之和,
       推理侧 = 合计 − 训练侧", 而 usageBatch 是"自上次 resetBatchStats 以来**所有**前向"。
       不划这一刀, 全部采集期的推理前向会被算进第一个训练批 —— 读出来的"训练侧"
       变成"推理侧", 而且数字看起来完全合理 (这是最坏的一类错读数)。
       ⚠ 这一句**不影响任何数值**: 每条学习路径本来就会在开头调一次同样的复位。
    */
    ag.resetMoeBatchStats();

    /* 阶段 1 结束时读一次: 此时训练侧应为 0 (还没学过), 全部是推理侧 */
    double collectInferMaxVio = 0.0;
    {
        std::vector<long long> t0, i0;
        ag.moeUsageSplit(t0, i0);
        const LoadStat s = loadOf(i0);
        collectInferMaxVio = s.maxVio;
        std::printf("      [采集期推理侧] 前向 %lld 次: MaxVio %.3f | 有效专家 %d/%d | %s\n",
                    s.total, s.maxVio, s.effective, (int)i0.size(), shareText(s).c_str());
    }

    /*
       ---- 阶段 2: 固定次数的批学习 ----
       逐批读数的口径是 **`moeUsage` 的增量** (不是 split 的增量): 学习循环里除了训练
       前向没有别的前向, 所以"合计的增量"就是"这一批的路由分布"。批与批之间还各自
       调一次 resetBatchStats, 所以增量干净。
       ⚠ 拿 `moeUsageSplit` 做逐批差分会把**批间**的推理前向混进来 —— 本工具第一版
       就是这么写的, 结果训练侧读数直接失真 (而且数字看起来完全合理)。
    */
    std::vector<long long> totPrev(ag.moeExpertCount(), 0), totNow;
    ag.moeUsage(totPrev);
    for (int b = 0; b < g_cfg.batches; b++) {
        const bool ok = ag.ppo.learnFromReplay((std::size_t)g_cfg.batchSize, 1, g_cfg.lr);
        if (!ok) {
            std::printf("      [警告] 批 %d 未执行 (池子 %zu < 批 %d)\n",
                        b + 1, ag.ppo.replaySize(), g_cfg.batchSize);
            break;
        }
        ag.moeUsage(totNow);
        std::vector<long long> delta(totNow.size(), 0);
        long long dTotal = 0;
        for (std::size_t e = 0; e < totNow.size(); e++) {
            delta[e] = totNow[e] - totPrev[e];
            dTotal += delta[e];
        }
        if (dTotal <= 0) {
            std::printf("      [警告] 批 %d 没有记录到任何路由 (读数口径坏了)\n", b + 1);
            break;
        }
        R.perBatchMaxVio.push_back(loadOf(delta).maxVio);
        /* 训练侧的累计份额: 各批增量逐专家相加 (这才是"均衡机制收敛到哪") */
        if (R.trainUsage.empty()) {
            R.trainUsage.assign(delta.size(), 0);
        }
        for (std::size_t e = 0; e < delta.size(); e++) {
            R.trainUsage[e] += delta[e];
        }
        totPrev = totNow;
    }

    /* ---- 阶段 3: 训练后的评测对弈 (只推理, 不上限学习) ---- */
    ag.trainSelfPlay(g_cfg.evalGames, g_cfg.sims, g_cfg.moves, false, 0.5f, 0.1f);

    /* ---- 读数 ---- */
    std::vector<long long> trUse, infUse;
    ag.moeUsageSplit(trUse, infUse);
    /* 训练侧取"各批增量之和" (见上), 推理侧取 split 报的那个 (与上面同一来源) */
    const LoadStat tr = loadOf(R.trainUsage);
    for (std::size_t i = 0; i < R.perBatchMaxVio.size(); i++) {
        R.batchMeanMaxVio += R.perBatchMaxVio[i];
    }
    if (!R.perBatchMaxVio.empty()) {
        R.batchMeanMaxVio /= (double)R.perBatchMaxVio.size();
    }
    const LoadStat inf = loadOf(infUse);
    R.trainMaxVio = tr.maxVio;
    R.trainMinShare = tr.minShare;
    R.trainEffective = tr.effective;
    R.trainForwards = tr.total;
    R.inferMaxVio = inf.maxVio;
    R.inferMinShare = inf.minShare;
    R.inferEffective = inf.effective;
    R.inferForwards = inf.total;
    R.collectInferMaxVio = collectInferMaxVio;
    (void)trUse;

    std::vector<float> bias;
    ag.moeBiasSnapshot(bias);
    if (!bias.empty()) {
        float lo = bias[0], hi = bias[0];
        for (std::size_t i = 1; i < bias.size(); i++) {
            if (bias[i] < lo) { lo = bias[i]; }
            if (bias[i] > hi) { hi = bias[i]; }
        }
        R.biasSpread = (double)(hi - lo);
    }
    R.actorLossMean = (double)ag.ppo.lastActorLoss;
    R.valueLossMean = (double)ag.ppo.lastLoss;
    R.seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();

    std::printf("    %-10s %s\n", arm.tag, arm.desc);
    std::printf("      门控 %s (隐层读数 %d) | 样本池 %zu 条 | 批学习 %d 次 x %d | 用时 %.1f s\n",
                (R.gateHiddenSeen > 0) ? "MLP: d_model->h(tanh)->E" : "线性: wg·x+bg",
                R.gateHiddenSeen, poolSize, (int)R.perBatchMaxVio.size(),
                g_cfg.batchSize, R.seconds);
    printBatchTrace(R.perBatchMaxVio);
    std::printf("      训练侧前向 %lld 次: MaxVio %.3f | 最小份额 %.1f%% | 有效专家 %d/%d | %s\n",
                tr.total, tr.maxVio, 100.0 * tr.minShare, tr.effective,
                (int)R.trainUsage.size(), shareText(tr).c_str());
    std::printf("      推理侧(采集期, 训练前) MaxVio %.3f  ->  推理侧(评测期, 训练后) "
                "MaxVio %.3f | 前向 %lld 次 | 最小份额 %.1f%% | 有效专家 %d/%d\n",
                R.collectInferMaxVio, inf.maxVio, inf.total,
                100.0 * inf.minShare, inf.effective, (int)infUse.size());
    std::printf("      偏置 b_i 极差 %.3f | 最近批 actor CE %.4f / value MSE %.6f\n",
                R.biasSpread, R.actorLossMean, R.valueLossMean);
    return R;
}

void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        std::string k = a, v;
        if (eq != nullptr) {
            k = a.substr(0, (std::size_t)(eq - argv[i]));
            v = a.substr((std::size_t)(eq - argv[i]) + 1);
        }
        if (k == "--games")        { g_cfg.games = std::atoi(v.c_str()); }
        else if (k == "--moves")   { g_cfg.moves = std::atoi(v.c_str()); }
        else if (k == "--sims")    { g_cfg.sims = std::atoi(v.c_str()); }
        else if (k == "--batches") { g_cfg.batches = std::atoi(v.c_str()); }
        else if (k == "--batch")   { g_cfg.batchSize = std::atoi(v.c_str()); }
        else if (k == "--eval-games") { g_cfg.evalGames = std::atoi(v.c_str()); }
        else if (k == "--expert")  { g_cfg.expertHidden = std::atoi(v.c_str()); }
        else if (k == "--seed")    { g_cfg.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--lr")      { g_cfg.lr = (float)std::atof(v.c_str()); }
        else if (k == "--bias-rate") { g_cfg.biasRate = (float)std::atof(v.c_str()); }
        else if (k == "--gate")    { g_cfg.gateHidden = std::atoi(v.c_str()); }
        else if (k == "--backbone") {
            const std::string b = v;
            if (b == "tb" || b == "TB" || b == "1") {
                g_cfg.tbExperts = true;
            } else if (b == "mlp" || b == "MLP" || b == "0") {
                g_cfg.tbExperts = false;
            } else {
                std::printf("[警告] --backbone=%s 不认识 (可用 mlp / tb), 按 mlp 处理\n",
                            v.c_str());
            }
        }
        else if (k == "--only") {
            /*
               按逗号**精确**切分再逐个比对。用子串匹配会静默出错: "aux" 是 "auxStrong"
               的子串 —— 那种"多跑了一臂"的错误不报任何异常, 只会让人对着汇总表算错结论。
            */
            std::string s = v;
            std::size_t pos = 0;
            while (pos <= s.size()) {
                const std::size_t comma = s.find(',', pos);
                const std::string tok = s.substr(pos, (comma == std::string::npos)
                                                          ? std::string::npos
                                                          : comma - pos);
                if (!tok.empty()) {
                    g_cfg.onlyTags.push_back(tok);
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        }
        else if (k == "--quick") {
            g_cfg.games = 2; g_cfg.moves = 16; g_cfg.sims = 16;
            g_cfg.batches = 4; g_cfg.evalGames = 1;
        }
        else if (k == "--help" || k == "-h") {
            std::printf(
                "用法: bench_ppo_moe_balance [--games=N] [--moves=N] [--sims=N]\n"
                "                           [--batches=N] [--batch=N] [--eval-games=N]\n"
                "                           [--expert=N] [--seed=N] [--lr=F]\n"
                "                           [--bias-rate=F] [--gate=N] [--backbone=mlp|tb]\n"
                "                           [--only=none,aux,auxStrong,lossfree,mlp,mlpFree,mlpNone]\n"
                "                           [--quick]\n"
                "口径: 每臂重建 agent + 同 seed + 相同的批学习次数; 读数只看本臂 (resetMoeUsage)。\n"
                "      MaxVio = max_share/(1/E) − 1 (0 = 完美均衡); 有效专家 = 份额 > 5% 的个数。\n"
                "      --gate=0 时 mlp/mlpFree/mlpNone 三档不跑 (线性门控)。\n");
            std::exit(0);
        }
        else {
            std::printf("[警告] 未知参数: %s (--help 看用法)\n", argv[i]);
        }
    }
    if (g_cfg.games < 1)   { g_cfg.games = 1; }
    if (g_cfg.moves < 2)   { g_cfg.moves = 2; }
    if (g_cfg.batches < 0) { g_cfg.batches = 0; }
}

}  /* namespace */

int main(int argc, char **argv)
{
    parseArgs(argc, argv);

    /*
       骨干: 稀疏 MoE + **MLP 专家** (E=8 top-2)。
       为什么用这一支而不是现役的 TB 专家: 本工具量的是**路由**, 而路由只依赖门控权重
       与输入分布, 与专家内部是什么无关 (同 bench_gate_moe 的根据)。TB 专家在这里
       纯粹是白烧算力 —— E=8/top-2 的 TB 支是 2.4 GB 内存 + 6 ms/模拟。
       E=8 也让"偏斜"这件事最容易看见: [3b] 实测的 MaxVio 理论最大值 = E−1 = 7。
    */
    const int hidden = 32;

    const int experts = g_cfg.tbExperts ? RL::PPO::MOE_EXPERTS : RL::PPO::MOE_MLP_EXPERTS;
    const int topK = g_cfg.tbExperts ? RL::PPO::MOE_TOPK : RL::PPO::MOE_MLP_TOPK;

    std::printf("=== PPO 骨干的 MoE 负载均衡 A/B (真实自对弈 + 真实训练) ===\n");
    std::printf("骨干: 稀疏 MoE + %s 专家, E=%d top-%d (expertHidden=%d), d_model=%d\n",
                g_cfg.tbExperts ? "TransformerBlock (界面现役)" : "MLP",
                experts, topK, g_cfg.expertHidden, PPOMCTSAgent::STATE_DIM);
    std::printf("配置: 每臂 %d 局 x %d 手 x %d 模拟 采数据; 然后 %d 次批学习 (批 %d); "
                "最后 %d 局评测\n",
                g_cfg.games, g_cfg.moves, g_cfg.sims, g_cfg.batches, g_cfg.batchSize,
                g_cfg.evalGames);
    std::printf("      seed=%u, lr=%.4g, 偏置步长=%.4g, MLP 门控隐层=%d\n", g_cfg.seed,
                (double)g_cfg.lr, (double)g_cfg.biasRate, g_cfg.gateHidden);
    std::printf("口径: 每臂重建 agent (门控回到同一初始化) + resetMoeUsage ⇒ 读数只属于本臂\n");
    std::printf("      MaxVio = max_share/(1/E) − 1 (0 = 完美均衡); 有效专家 = 份额 > 5%%\n");
    std::printf("      在 E=%d 下 MaxVio 的理论最大值 = %d − 1 = %d (一个专家吃掉全部流量)\n\n",
                experts, experts, experts - 1);

    /*
       ---- 七个臂 ----
       前四个是**线性门控**(现状默认)下的均衡对照; 后三个换门控结构。
       为什么门控结构要单独成组: "门控表达力"和"负载均衡"是**两件互相拉扯的事** ——
       受控实验里 MLP 门控买到路由纯度 +0.15~0.31, 同时把 MaxVio 从 0.30 推到 1.43~1.79。
       所以正确的问题不是"哪个门控更好", 而是"MLP 门控 + 偏置均衡"这一对能不能同时
       拿到两者。本工具把这六个格子摆在一次运行里, 就是为了不靠想象回答它。
    */
    const Arm arms[] = {
        {"none",       0.0f,  false, 0,  "对照下界 (线性门控): 纯 softmax 反向, 两条均衡路都关"},
        {"aux",        0.1f,  false, 0,  "现状默认 (线性门控): 只靠辅助损失 auxCoef=0.1"},
        {"auxStrong", 10.0f,  false, 0,  "线性门控 + 辅助损失开到 10 (受控实验里'拉不动'那档)"},
        {"lossfree",   0.1f,  true,  0,  "线性门控 + 无辅助损失偏置均衡 (候选修法)"},
        {"mlp",        0.1f,  false, g_cfg.gateHidden,
                                         "MLP 门控 (三层), 不开偏置 —— 量'换门控'自己买到什么"},
        {"mlpFree",    0.1f,  true,  g_cfg.gateHidden,
                                         "MLP 门控 + 偏置均衡 (SAC 支的推荐组合)"},
        {"mlpNone",    0.0f,  false, g_cfg.gateHidden,
                                         "MLP 门控 + 两条均衡路都关 (看纯 MLP 路由的偏斜下界)"},
    };
    const int kArms = (int)(sizeof(arms) / sizeof(arms[0]));
    std::vector<char> sel((std::size_t)kArms, 0);
    for (int i = 0; i < kArms; i++) {
        bool want = g_cfg.onlyTags.empty();
        for (std::size_t k = 0; k < g_cfg.onlyTags.size(); k++) {
            if (g_cfg.onlyTags[k] == arms[i].tag) {
                want = true;
            }
        }
        /* --gate=0 时 MLP 那三档不跑 (否则会拿 MLP 结构去跑"线性门控"的对照) */
        if (arms[i].gateHidden > 0 && g_cfg.gateHidden <= 0) {
            want = false;
        }
        sel[(std::size_t)i] = want ? 1 : 0;
    }
    /* --only 里写了不存在的臂名: 明确报出来, 不要静默少跑一臂 */
    for (std::size_t k = 0; k < g_cfg.onlyTags.size(); k++) {
        bool found = false;
        for (int i = 0; i < kArms; i++) {
            if (g_cfg.onlyTags[k] == arms[i].tag) {
                found = true;
            }
        }
        if (!found) {
            std::printf("[警告] --only 里有未知的臂名: '%s' (本臂不会跑)\n",
                        g_cfg.onlyTags[k].c_str());
        }
    }

    std::vector<ArmResult> results;
    std::vector<std::string> tags;
    for (int i = 0; i < kArms; i++) {
        if (!sel[(std::size_t)i]) {
            continue;
        }
        results.push_back(runArm(arms[i], hidden));
        tags.push_back(arms[i].tag);
        std::printf("\n");
    }

    /* ---- 汇总表 ---- */
    std::printf("=== 汇总 ===\n");
    std::printf("%-10s %-5s %13s %10s %9s %7s %11s %11s %10s %8s\n",
                "臂", "门控", "训练侧 MaxVio", "逐批均值", "最小份额", "有效",
                "推理(采集前)", "推理(评测后)", "偏置极差", "用时(s)");
    for (std::size_t i = 0; i < results.size(); i++) {
        const ArmResult &R = results[i];
        std::printf("%-10s %-5s %13.3f %10.3f %8.1f%% %7d %11.3f %11.3f %10.3f %8.1f\n",
                    tags[i].c_str(),
                    (R.gateHiddenSeen > 0) ? "MLP" : "线性",
                    R.trainMaxVio, R.batchMeanMaxVio, 100.0 * R.trainMinShare,
                    R.trainEffective, R.collectInferMaxVio, R.inferMaxVio,
                    R.biasSpread, R.seconds);
    }
    std::printf("(训练侧/推理侧的划分按**批边界**算, 不是按 forward 的 inference 参数 ——\n");
    std::printf(" 本工程的搜索路径调的是默认值 false, 靠那个参数划分会得到'推理侧恒为 0'。\n");
    std::printf(" 训练侧前向 = 各批增量之和; 推理侧 = 合计 − 训练侧。)\n");

    /*
       ---- 判据 (事先写下, 事后不改) ----
       本工具**不测棋力** (局数太少, 且开局随机) —— 它只回答"负载拉平了没有"。
       棋力的裁决必须另开会议: 见 README 的"MoE 负载均衡"一节与
       docs/moe_gate_experiment_2026_10.md §9 记录过的教训 (评测协议本身会坏)。
    */
    std::printf("\n判据 (事先写下): 训练侧 MaxVio 越小越均衡, 0 = 完美均衡。\n");
    std::printf("  1. auxStrong 与 aux 的训练侧 MaxVio 几乎一样, 且都**远高于**偏置臂\n");
    std::printf("     => 辅助损失这条路在真实训练链路上也拉不动负载 (与受控实验 [3b] 的预测一致)\n");
    std::printf("  2. lossfree 明显低于 aux\n");
    std::printf("     => '换均衡的作用点'在真实训练链路上成立\n");
    std::printf("  3. mlp 的**推理侧** MaxVio 高于 aux (受控实验的预测: MLP 门控让负载更偏),\n");
    std::printf("     而 mlpFree 相对 mlp 有改善 => 它必须与偏置均衡配对才有意义\n");
    std::printf("  读表注意: '逐批均值'那列天生噪声大 (一批只有 batchSize x topK 次路由选择,\n");
    std::printf("     在 E=8 上均匀随机本身就有 ~0.4 的抖动) —— 只有累计份额能当结论。\n");
    std::printf("  另: 推理侧与训练侧可以**方向不同** —— 偏置只在批边界更新, 它均衡的是训练分布;\n");
    std::printf("      '偏置均衡'不等于'到处都均衡' (SAC 支实测过: 训练侧 0.049->0.009 而评测侧 0.264->0.598)。\n");
    return 0;
}
