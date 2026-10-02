/*
 * bench_ppo_backbone_tb_main.cpp - PPO 的 TB 专家骨干: E/top-k 的**代价**实测 (不进 ctest)
 * ============================================================================
 *
 * 为什么需要它: `src/rl/ppo.h` 顶部的 E/top-k 是**编译期常量**, 改它们的代价只能靠
 * (a) 本文件实测, 或 (b) 把源码改回去重编。而这两种骨干在同一个二进制里就能建出来
 * (RL::PPO::Backbone 是**运行时**参数), 所以"4/1 vs 8/2 到底贵多少"可以一次跑完。
 *
 * 量四件事 (每一件都对应一个会被这次改动影响的界面/训练读数):
 *   1. **参数量** (actor / critic 分开) —— 权重文件大小与"载入要多久"的直接来源;
 *   2. **ms / 模拟** (标定后换算) —— 决定界面上那个**每步决策要等多久**, 也是
 *      "模拟次数换容量"这笔交易的价格; 本工程唯一测出过棋力的杠杆就是模拟次数
 *      (40/64/120 模拟 -> 42.2% / 50.0% / 62.5%), 所以这一列最贵;
 *   3. **峰值工作集** (PeakWorkingSet64) —— `withGrad=true` 时每张全连接有 w/g/v/m
 *      **四份**, 而 PPO **没有共享骨干** (actor 与 critic 各背一套), 所以这是启动
 *      预加载与训练 clone 的真实内存代价;
 *   4. **训练侧负载 MaxVio** (走 `moeUsageSplit` 与 `finishLearnPhase`) —— E 越大越不
 *      均衡 (SAC 支实测 E=4 是 0.049, E=8 是 0.477), 而 PPO 这条默认开着无辅助损失
 *      偏置均衡, 所以这一列是"配套机制有没有跟上"的读数。
 *
 * 用法:
 *   bench_ppo_backbone_tb [--games=2] [--plies=24] [--sims=16] [--backbone=tb|mlp|both]
 *                         [--load=PREFIX] [--quick]
 */

/*
 * 为什么**不**在这里 `#include <windows.h>` 量内存 (踩过一次, 值得记下来):
 *   `ppomcts_agent.h` 有一个成员 `PLANES` (19 个平面), 而 Windows SDK 的 `wingdi.h`
 *   有 `#define PLANES 14` —— 两个一撞, `static constexpr int PLANES = ...` 被宏展开成
 *   `static constexpr int 14 = PLANES + 5;`, 报的是 `error C2059: 语法错误:"常数"`
 *   和 `error C2238`。**报错点在 agent 头文件里, 而病灶在本文件的 include 上**,
 *   顺着报错查会查错方向。所以内存由一个**外部采样器**量
 *   (`tools/measure_peak_working_set.ps1`, 按 PID 每 0.25 s 采一次 PeakWorkingSet64),
 *   与 SAC 那一轮量 E=8 的 1,138 MB 用的是同一个办法。
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

struct Cfg {
    int games = 2;
    int plies = 24;
    int sims  = 16;
    int calibSims = 8;
    /*
       批学习步数。**必须封顶**, 而且要比"池子里的样本数"小 —— 每一次
       `learnFromReplay(64, 1)` 都要为 64 条样本各跑一遍 actor 与 critic 的**前向+反向**,
       在 E=8/top-2 的 TB 骨干上那是 2.4 GB 级的权重流量/样本 (w/g/v/m 四份)。
       实测: 不封顶地"能学多少学多少" (200+ 步) 在一台空闲的机器上跑 16 分钟还没完,
       而那对本文件要量的四件事 (参数量 / ms/模拟 / 内存 / 负载) **一点帮助都没有** ——
       负载读数只需要几十步就能看出偏斜。所以默认给 24 步, 需要更长轨迹就调大。
    */
    int learnSteps = 24;
    int batchSize = 64;
    /* 再按界面的真实预算跑一次搜索 (0 = 不跑); 默认 400 = chessboard.cpp 的 PPO_SIMS */
    int uiSims = 400;
    std::string backbone = "both";   /* tb / mlp / both */
    std::string loadPrefix;
    bool verbose = true;
};

Cfg g_cfg;

double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* 负载读数: MaxVio = max_i share_i/(1/E) − 1 (0 = 完美均衡) + 有效专家 (份额 >5%) */
struct LoadStat {
    double maxVio = 0.0;
    double minShare = 0.0;
    int effective = 0;
    long long total = 0;
};

LoadStat loadOf(const std::vector<long long> &v)
{
    LoadStat s;
    if (v.empty()) { return s; }
    for (std::size_t i = 0; i < v.size(); i++) { s.total += v[i]; }
    if (s.total <= 0) { return s; }
    double mx = 0.0;
    s.minShare = 1.0;
    for (std::size_t i = 0; i < v.size(); i++) {
        const double sh = (double)v[i] / (double)s.total;
        if (sh > mx) { mx = sh; }
        if (sh < s.minShare) { s.minShare = sh; }
        if (sh > 0.05) { s.effective++; }
    }
    s.maxVio = mx * (double)v.size() - 1.0;
    return s;
}

void runOne(const RL::PPO::Backbone bb, const char *name)
{
    const int experts = (bb == RL::PPO::Backbone::TbExperts) ? RL::PPO::MOE_EXPERTS
                                                             : RL::PPO::MOE_MLP_EXPERTS;
    const int topK = (bb == RL::PPO::Backbone::TbExperts) ? RL::PPO::MOE_TOPK
                                                          : RL::PPO::MOE_MLP_TOPK;
    RL::Random::setSeed(20240901u);
    Chess board;
    std::printf("\n=== 骨干 %s (E=%d top-%d) ===\n", name, experts, topK);
    std::printf("  (内存采样: 外部采样器从本进程启动起每 0.25 s 采一次 PeakWorkingSet64 ——"
                " 见 tools/measure_peak_working_set.ps1)\n");

    PPOMCTSAgent ag(board, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true, bb);

    std::printf("  参数量: actor %lld | critic %lld | 合计 %.1f M\n",
                ag.actorParamCount(), ag.criticParamCount(),
                (double)(ag.actorParamCount() + ag.criticParamCount()) / 1e6);
    std::printf("        (withGrad=true ⇒ 每张全连接有 w/g/v/m 四份缓冲; PPO **没有共享骨干**,"
                " actor 与 critic 各背一套完整专家)\n");

    if (!g_cfg.loadPrefix.empty()) {
        const bool ok = ag.loadModel(g_cfg.loadPrefix);
        std::printf("  权重  : %s -> %s\n", g_cfg.loadPrefix.c_str(),
                    ok ? "载入成功" : "**载入失败** (架构不匹配? 这正是 E/top-k 改动的直接后果)");
    }

    /* ---- ms/模拟: 标定 (初始局面, 与 bench_moe 同一手法) ---- */
    board.reset();
    ag.selectMove(Stone::COLOR_RED, 2, 0.0f);   /* 预热: 别把首次分配算进均值 */
    {
        const double t0 = nowMs();
        ag.selectMove(Stone::COLOR_RED, g_cfg.calibSims, 0.0f);
        const double ms = nowMs() - t0;
        const double perSim = ms / (double)g_cfg.calibSims;
        std::printf("  搜索  : %.3f ms/模拟 (小标定: %d 次模拟耗时 %.1f ms)\n",
                    perSim, g_cfg.calibSims, ms);
        /*
           再跑一次**界面预算**那一档 (400 次模拟) 实测一次决策耗时。
           为什么值得多花这几秒: 小标定 (8 次) 里固定的开销 (根建立 / 合法着法生成 /
           置换表) 摊不薄, 而用户真正感觉到的是"一步要等多久" —— 只有按真实预算跑一次,
           `PPO_SIMS` 那个常量该不该跟着这次 E/top-k 改动调整才是个实测问题。
        */
        if (g_cfg.uiSims > 0 && g_cfg.uiSims != g_cfg.calibSims) {
            board.reset();
            ag.selectMove(Stone::COLOR_RED, 2, 0.0f);
            const double t1 = nowMs();
            ag.selectMove(Stone::COLOR_RED, g_cfg.uiSims, 0.0f);
            const double msUi = nowMs() - t1;
            std::printf("         : 界面预算 %d 模拟 -> **%.2f s/步** (含根建立等固定开销)\n",
                        g_cfg.uiSims, msUi / 1000.0);
        }
    }

    /* ---- 自对弈 (不学) + 批学习一次, 量训练代价与负载 ---- */
    ag.replayBatchSize = 0;
    ag.learnStepsPerEpisode = 0;
    ag.openingPlies = 4;
    ag.openingSeed = 20240901u;
    ag.resetMoeUsage();

    const double t0 = nowMs();
    ag.trainSelfPlay(g_cfg.games, g_cfg.sims, g_cfg.plies, false, 1.0f, 0.25f);
    const double tSelf = (nowMs() - t0) / 1000.0;
    const std::size_t pool = ag.ppo.replaySize();

    const double t1 = nowMs();
    int learned = 0;
    for (int i = 0; i < g_cfg.learnSteps; i++) {
        if (!ag.ppo.learnFromReplay((std::size_t)g_cfg.batchSize, 1, 0.001f)) {
            break;
        }
        learned++;
    }
    const double tLearn = (nowMs() - t1) / 1000.0;

    std::printf("  训练  : 自对弈 %d 局 x %d 手 x %d 模拟 = %.1f s (池 %zu 条) | "
                "批学习 %d 次 (批 %d) = %.1f s (%.1f ms/次)\n",
                g_cfg.games, g_cfg.plies, g_cfg.sims, tSelf, pool, learned,
                g_cfg.batchSize, tLearn,
                (learned > 0) ? tLearn * 1000.0 / (double)learned : 0.0);

    std::vector<long long> tr, inf;
    ag.moeUsageSplit(tr, inf);
    const LoadStat lt = loadOf(tr);
    const LoadStat li = loadOf(inf);
    std::printf("  负载  : 训练侧前向 %lld MaxVio %.3f 最小份额 %.1f%% 有效 %d/%d | "
                "推理侧前向 %lld MaxVio %.3f\n",
                 lt.total, lt.maxVio, 100.0 * lt.minShare, lt.effective, experts,
                 li.total, li.maxVio);
    /* 内存峰值由外部采样器报 (见文件头: windows.h 的 PLANES 宏会撞 agent 的成员名) */
}

}  /* namespace */

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        std::string k = a, v;
        if (eq != nullptr) {
            k = a.substr(0, (std::size_t)(eq - argv[i]));
            v = a.substr((std::size_t)(eq - argv[i]) + 1);
        }
        if (k == "--games")      { g_cfg.games = std::atoi(v.c_str()); }
        else if (k == "--plies") { g_cfg.plies = std::atoi(v.c_str()); }
        else if (k == "--sims")  { g_cfg.sims = std::atoi(v.c_str()); }
        else if (k == "--calib-sims") { g_cfg.calibSims = std::atoi(v.c_str()); }
        else if (k == "--learn-steps") { g_cfg.learnSteps = std::atoi(v.c_str()); }
        else if (k == "--batch")  { g_cfg.batchSize = std::atoi(v.c_str()); }
        else if (k == "--ui-sims") { g_cfg.uiSims = std::atoi(v.c_str()); }
        else if (k == "--backbone") { g_cfg.backbone = v; }
        else if (k == "--load")  { g_cfg.loadPrefix = v; }
        else if (k == "--quick") {
            g_cfg.games = 1; g_cfg.plies = 12; g_cfg.sims = 8; g_cfg.calibSims = 4;
            g_cfg.learnSteps = 8; g_cfg.uiSims = 0;
        }
        else if (k == "--help" || k == "-h") {
            std::printf(
                "用法: bench_ppo_backbone_tb [--games=N] [--plies=N] [--sims=N]\n"
                "                           [--calib-sims=N] [--ui-sims=N] [--learn-steps=N] [--batch=N]\n"
                "                           [--backbone=tb|mlp|both] [--load=PREFIX] [--quick]\n"
                "口径: 参数量 / ms/模拟 + 界面预算下的 s/步 / 峰值工作集 (外部采样器) /\n"
                "      训练侧负载 MaxVio。\n"
                "      --learn-steps 默认 24: 每步要为 batchSize 条样本各跑一遍 actor+critic\n"
                "      的前向+反向, 在 E=8/top-2 上是 2.4 GB 级权重流量/样本, 不封顶会跑很久。\n");
            return 0;
        }
    }
    if (g_cfg.games < 1) { g_cfg.games = 1; }
    if (g_cfg.plies < 2) { g_cfg.plies = 2; }
    if (g_cfg.calibSims < 1) { g_cfg.calibSims = 1; }

    /*
       关掉 stdout 缓冲。理由与 test_ppomcts 等分钟级程序同一句话: 默认的块缓冲
       (重定向到文件/管道时) 会在进程被 kill 时把**所有**输出丢掉 —— 本文件第一次跑
       就是这样: 16 分钟的输出一个字节都没留下, 只能看到一个 pid。
    */
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("=== PPO 骨干 E/top-k 代价实测 ===\n");
    std::printf("编译期常量: TB 专家 E=%d top-%d | MLP 专家 E=%d top-%d\n",
                RL::PPO::MOE_EXPERTS, RL::PPO::MOE_TOPK,
                RL::PPO::MOE_MLP_EXPERTS, RL::PPO::MOE_MLP_TOPK);
    std::printf("配置: 自对弈 %d 局 x %d 手 x %d 模拟; 标定 %d 次模拟\n",
                g_cfg.games, g_cfg.plies, g_cfg.sims, g_cfg.calibSims);

    if (g_cfg.backbone == "tb" || g_cfg.backbone == "both") {
        runOne(RL::PPO::Backbone::TbExperts, "TB<16,360> (界面 AGENT_PPOMCTS)");
    }
    if (g_cfg.backbone == "mlp" || g_cfg.backbone == "both") {
        runOne(RL::PPO::Backbone::MlpExperts, "MlpExpert (界面 AGENT_PPOMCTS_MLP)");
    }
    return 0;
}
