/*
 * train_ppo_main.cpp - P0.1: 无界面**常驻**训练器 (不进 ctest)
 * ============================================================================
 *
 * 做什么: 一个进程里连跑 K 局 PPO+MCTS 自对弈训练, 训练完打印一份"这一轮到底在
 *         下什么样的棋"的报告 (和棋原因分桶 / 手数 / 吃子 / 开局多样性 / 损失).
 *
 * 为什么需要它 (P0.1 的全部理由): GUI 的训练线程是**每局**走一趟
 *     seed 写盘 (weights/_temp_train.dat) -> clone 读入 -> clone 写回 -> 主 agent 读入
 * 而 actor+critic 一对约 555 MB (weights/ppo_bc_d4_actor = 279 MB), 也就是**每局
 * 2.2 GB 磁盘往返**换 3.2 分钟的自对弈 (src/chessboard.cpp:1915,1991,1999,2055)。
 * 那个架构是为"界面上看得见进度"服务的, 不适合当吞吐基线。本程序直接对**一个**
 * 常驻 agent 反复调 trainSelfPlay, 每局零磁盘 I/O、零建网/拆网。
 *
 * 它同时是 P0.2/P0.4 的落点:
 *   * P0.2 —— 和棋要**分原因** (三次重复 / 60 回合自然限着 / 台架截断)。这三类性质
 *     完全不同, 混成一桶就会把"手数上限"当成"规则问题"或"棋力到顶"去治。
 *   * P0.4 —— 一次跑完就把读数落到 CSV (TB 长表口径), 不再需要人工拼几条曲线。
 *
 * 它**不做**: 根节点/逐手诊断 (那是 bench_diag 的职责 —— 它自己驱动 selectMove 采
 * RootDiag/MoveBehavior; 训练路径里没有"决策前的根"可供事后读取)。本程序量的是
 * "生态"这一层: 和棋构成、手数、吃子、开局多样性、损失、MoE 负载。
 *
 * 用法:
 *   cmake --build <build> --target train_ppo
 *   <build>/train_ppo --games=20 --sims=400 --moves=60 --opening=8 --csv=run1
 *   <build>/train_ppo --games=20 --load=weights/ppomcts_agent.dat --save=weights/run1
 *
 * P1.1 (PBRS A/B) 的开关就在这里 —— 同一个 --seed / 同一批参数下跑两次, 比报告里的
 * 吃子与和棋构成:
 *   <build>/train_ppo --games=20 --csv=shaping_on
 *   <build>/train_ppo --games=20 --no-shaping --csv=shaping_off
 *   <build>/train_ppo --games=20 --shaping-alpha=0.25 --csv=shaping_25
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
#include "rl/util.hpp"

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    int games = 4;
    int sims = 400;              /* 与 GUI 的 BG_TRAIN_SIMS 一致 (须远大于中局分支数) */
    int moves = 60;              /* 与 GUI 的 BG_TRAIN_MAX_MOVES 一致 (60 ply = 30 回合) */
    int opening = 0;             /* 每局先随机走几手 (0 = 标准开局; >0 = 打破"每局同一盘棋") */
    unsigned seed = 20240901u;
    std::string loadPrefix;
    std::string savePrefix;
    std::string csvPrefix;
    float tempRoot = 1.0f;
    float tempFinal = 0.1f;
    int replayBatch = 64;        /* 回放池触发学习的批大小 (0 = 只入池不学习) */
    /*
       网络宽度 (默认与 GUI 一致: 64/64)。留出来是为 P1.1 的消融: 一次 A/B 要跑很多局面,
       用 hidden=16/expert=16 的小骨干能把单局时间压下来 (与 bench_diag 的 --hidden/--expert
       同一约定)。**注意**: 宽度是权重文件架构的一部分, 小骨干存的权重不能被 64 的
       模型载入 (Net::load 会拒绝), 反之亦然。
    */
    int hidden = 64;
    int expert = 64;
    /* P1.1 消融开关 */
    bool potentialShaping = true;
    float shapingAlpha = 1.0f;
    bool materialReward = true;
    bool truncationBootstrap = true;
    bool rootNoise = true;
    bool verbose = true;
};

Cfg g_cfg;

/* ============================================================
 *  参数
 * ============================================================ */
bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        std::string k = a;
        std::string v;
        if (eq != nullptr) {
            k = a.substr(0, (std::size_t)(eq - argv[i]));
            v = a.substr((std::size_t)(eq - argv[i]) + 1);
        }
        if (k == "--games")  { g_cfg.games  = std::atoi(v.c_str()); }
        else if (k == "--sims")  { g_cfg.sims  = std::atoi(v.c_str()); }
        else if (k == "--moves") { g_cfg.moves = std::atoi(v.c_str()); }
        else if (k == "--opening") { g_cfg.opening = std::atoi(v.c_str()); }
        else if (k == "--seed")  { g_cfg.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--load")  { g_cfg.loadPrefix = v; }
        else if (k == "--save")  { g_cfg.savePrefix = v; }
        else if (k == "--csv")   { g_cfg.csvPrefix = v; }
        else if (k == "--replay") { g_cfg.replayBatch = std::atoi(v.c_str()); }
        else if (k == "--hidden") { g_cfg.hidden = std::atoi(v.c_str()); }
        else if (k == "--expert") { g_cfg.expert = std::atoi(v.c_str()); }
        else if (k == "--temp-root")  { g_cfg.tempRoot = (float)std::atof(v.c_str()); }
        else if (k == "--temp-final") { g_cfg.tempFinal = (float)std::atof(v.c_str()); }
        else if (k == "--shaping-alpha") { g_cfg.shapingAlpha = (float)std::atof(v.c_str()); }
        else if (k == "--no-shaping") { g_cfg.potentialShaping = false; }
        else if (k == "--no-material-reward") { g_cfg.materialReward = false; }
        else if (k == "--no-bootstrap") { g_cfg.truncationBootstrap = false; }
        else if (k == "--no-root-noise") { g_cfg.rootNoise = false; }
        else if (k == "--quiet") { g_cfg.verbose = false; }
        else if (k == "--help" || k == "-h") {
            std::printf(
                "用法: train_ppo [--games=N] [--sims=N] [--moves=N] [--opening=N]\n"
                "                 [--load=PREFIX] [--save=PREFIX] [--csv=PREFIX] [--seed=N]\n"
                "                 [--hidden=N] [--expert=N] [--replay=N]\n"
                "                 [--temp-root=F] [--temp-final=F]\n"
                "                 [--no-shaping] [--shaping-alpha=F] [--no-material-reward]\n"
                "                 [--no-bootstrap] [--no-root-noise] [--quiet]\n");
            return false;
        } else {
            std::printf("[警告] 未知参数: %s (--help 看用法)\n", argv[i]);
        }
    }
    if (g_cfg.games < 1) { g_cfg.games = 1; }
    if (g_cfg.moves < 1) { g_cfg.moves = 1; }
    return true;
}

const char *resultName(int r)
{
    if (r == Chess::RESULT_RED_WIN) { return "红胜"; }
    if (r == Chess::RESULT_BLACK_WIN) { return "黑胜"; }
    if (r == Chess::RESULT_DRAW) { return "和"; }
    return "未终局";
}

const char *endKindName(int k)
{
    switch (k) {
    case RL::Diag::END_MATE:               return "将杀";
    case RL::Diag::END_DRAW_REPEAT:        return "重复和";
    case RL::Diag::END_DRAW_NO_CAPTURE60:  return "限着和";
    case RL::Diag::END_TRUNCATED:          return "截断";
    default:                               return "未分类";
    }
}

}  // namespace

/* ============================================================
 *  main
 * ============================================================ */
int main(int argc, char **argv)
{
    if (!parseArgs(argc, argv)) {
        return 0;
    }

    RL::Random::setSeed(g_cfg.seed);

    Chess board;
    PPOMCTSAgent ag(board, g_cfg.hidden, 0.99f, 0.001f, 1.414f, g_cfg.expert);

    /* ---- 配置: 全部走成员, 不动 trainSelfPlay 的签名 ---- */
    ag.potentialShaping = g_cfg.potentialShaping;
    ag.shapingAlpha = g_cfg.shapingAlpha;
    ag.materialRewardEnabled = g_cfg.materialReward;
    ag.truncationBootstrap = g_cfg.truncationBootstrap;
    ag.rootNoise = g_cfg.rootNoise;
    ag.openingPlies = g_cfg.opening;
    ag.openingSeed = g_cfg.seed;
    ag.replayBatchSize = g_cfg.replayBatch;

    /*
       P0.1 的核心: 一个常驻 agent + 一个按局日志指针。
       gameLog 非空时 trainSelfPlay 每局结束追加一条 GameStat (含和棋原因/结束方式)。
    */
    std::vector<RL::Diag::GameStat> games;
    ag.gameLog = &games;

    std::printf("=== train_ppo (P0.1 常驻训练器) ===\n");
    std::printf("  网络   : hidden=%d expert=%d, state=%d, action=%d\n",
                g_cfg.hidden, g_cfg.expert,
                PPOMCTSAgent::STATE_DIM, PPOMCTSAgent::ACTION_DIM);
    std::printf("  自对弈 : %d 局 x %d sims x %d ply 上限 | 随机开局 %d 手\n",
                g_cfg.games, g_cfg.sims, g_cfg.moves, g_cfg.opening);
    std::printf("  温度   : %.2f -> %.2f (线性) | 根噪声 %s | 回放批 %d\n",
                (double)g_cfg.tempRoot, (double)g_cfg.tempFinal,
                g_cfg.rootNoise ? "开" : "关", g_cfg.replayBatch);
    std::printf("  塑形   : %s (alpha=%.2f) | 显式材质奖励 %s | 截断自举 %s\n",
                g_cfg.potentialShaping ? "开" : "关", (double)g_cfg.shapingAlpha,
                g_cfg.materialReward ? "开" : "关",
                g_cfg.truncationBootstrap ? "开" : "关");
    std::printf("  I/O    : 每局 0 次权重往返 (GUI 路径是 3 趟 x ~555 MB)\n");

    if (!g_cfg.loadPrefix.empty()) {
        if (ag.loadModel(g_cfg.loadPrefix)) {
            std::printf("  权重   : %s -> 载入成功\n", g_cfg.loadPrefix.c_str());
        } else {
            std::printf("  权重   : %s -> **载入失败** (架构不匹配? 路径错?)\n",
                        g_cfg.loadPrefix.c_str());
            return 1;
        }
    } else {
        std::printf("  权重   : 未指定 (随机初始化 —— 结构/生态指标有效, 棋力类指标无意义)\n");
    }

    RL::Diag::CsvWriter tb;
    if (!g_cfg.csvPrefix.empty()) {
        const std::string p = g_cfg.csvPrefix + "_tb.csv";
        if (tb.open(p, "tag,step,value")) {
            std::printf("  CSV    : %s (TB 长表口径)\n", p.c_str());
        }
    }

    /* ============================================================
     *  训练循环: 每局一次 trainSelfPlay(1, ...) —— 进程常驻, 零磁盘往返
     * ============================================================ */
    const auto t0 = std::chrono::steady_clock::now();
    double slowest = 0.0;
    for (int g = 0; g < g_cfg.games; g++) {
        const std::size_t before = games.size();
        const auto g0 = std::chrono::steady_clock::now();
        /*
           一局一调 (而不是 trainSelfPlay(games, ...) 一次跑完), 有两个理由:
             * 每局能立刻打印结果 (长跑时看得见进度);
             * 每局的计时能分开 (吞吐是 P0.1 的验收指标)。
           行为与一次跑 N 局等价: trainSelfPlay 每局都会 reset 棋盘 + 清搜索树。
        */
        ag.trainSelfPlay(1, g_cfg.sims, g_cfg.moves, false,
                         g_cfg.tempRoot, g_cfg.tempFinal);
        const double sec = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - g0).count();
        slowest = std::max(slowest, sec);

        if (games.size() > before && g_cfg.verbose) {
            const RL::Diag::GameStat &gs = games.back();
            std::printf("  局 %3d: %-4s %3d ply | %-8s | 吃子 红%d 黑%d | %.1fs\n",
                        g + 1, resultName(gs.result), gs.plies,
                        endKindName(gs.endKind), gs.capturesByRed, gs.capturesByBlack,
                        sec);
        }
        if (tb.isOpen() && games.size() > before) {
            const RL::Diag::GameStat &gs = games.back();
            const long long step = g + 1;
            RL::Diag::scalarRow(tb, "game_plies", step, (double)gs.plies);
            RL::Diag::scalarRow(tb, "game_red_win", step,
                                (gs.result == Chess::RESULT_RED_WIN) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_black_win", step,
                                (gs.result == Chess::RESULT_BLACK_WIN) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_draw", step,
                                (gs.result == Chess::RESULT_DRAW) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_end_mate", step,
                                (gs.endKind == RL::Diag::END_MATE) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_end_repeat", step,
                                (gs.endKind == RL::Diag::END_DRAW_REPEAT) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_end_no_capture60", step,
                                (gs.endKind == RL::Diag::END_DRAW_NO_CAPTURE60) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_end_truncated", step,
                                (gs.endKind == RL::Diag::END_TRUNCATED) ? 1.0 : 0.0);
            RL::Diag::scalarRow(tb, "game_captures_red", step, (double)gs.capturesByRed);
            RL::Diag::scalarRow(tb, "game_captures_black", step, (double)gs.capturesByBlack);
            RL::Diag::scalarRow(tb, "game_seconds", step, sec);
        }
    }
    const double total = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();

    /* ============================================================
     *  P0.4: 一次性报告
     * ============================================================ */
    RL::Diag::Aggregates agg;
    double capRedSum = 0.0, capBlackSum = 0.0;
    for (std::size_t i = 0; i < games.size(); i++) {
        agg.addGame(games[i]);
        capRedSum += games[i].capturesByRed;
        capBlackSum += games[i].capturesByBlack;
    }

    std::printf("\n=== 训练报告 ===\n");
    std::printf("  用时: 总 %.1f s | 平均 %.1f s/局 | 最慢 %.1f s | 吞吐 %.1f 局/小时\n",
                total, (games.empty() ? 0.0 : total / (double)games.size()), slowest,
                (total > 0.0) ? 3600.0 * (double)games.size() / total : 0.0);
    std::printf("  累计对局 (agent 自计): %d\n", ag.getTotalEpisodes());
    const double n = (games.empty() ? 1.0 : (double)games.size());
    std::printf("  吃子/局: 红 %.2f | 黑 %.2f (吃子是还敢不敢交换的代理读数)\n",
                capRedSum / n, capBlackSum / n);
    if (std::isfinite((double)ag.getLastTrainLoss())) {
        std::printf("  最近一次 value MSE: %.6f\n", (double)ag.getLastTrainLoss());
    }

    agg.print("train_ppo");

    /* MoE 路由 (只在骨干含稀疏 MoE 时非空) */
    {
        std::vector<long long> usage;
        ag.moeUsage(usage);
        if (!usage.empty()) {
            long long sum = 0, mx = 0;
            int unused = 0;
            for (std::size_t e = 0; e < usage.size(); e++) {
                sum += usage[e];
                mx = std::max(mx, usage[e]);
                if (usage[e] == 0) { unused++; }
            }
            const double mean = (double)sum / (double)usage.size();
            std::printf("  MoE 专家使用: E=%d top-%d | 最大/均值 %.2f | 未使用 %d 个\n",
                        ag.moeExpertCount(), ag.moeTopK(),
                        (mean > 0.0) ? (double)mx / mean : 0.0, unused);
        }
    }

    if (tb.isOpen()) {
        const long long step = (long long)games.size() + 1;
        RL::Diag::scalarRow(tb, "sum_games", step, (double)games.size());
        RL::Diag::scalarRow(tb, "sum_seconds_total", step, total);
        RL::Diag::scalarRow(tb, "sum_games_per_hour", step,
                            (total > 0.0) ? 3600.0 * (double)games.size() / total : 0.0);
        RL::Diag::scalarRow(tb, "sum_draw_rate", step, agg.drawRate());
        RL::Diag::scalarRow(tb, "sum_end_mate_rate", step, agg.endMateRate());
        RL::Diag::scalarRow(tb, "sum_end_repeat_rate", step, agg.drawRepeatRate());
        RL::Diag::scalarRow(tb, "sum_end_no_capture60_rate", step, agg.drawNoCapture60Rate());
        RL::Diag::scalarRow(tb, "sum_end_truncated_rate", step, agg.truncatedRate());
        RL::Diag::scalarRow(tb, "sum_truncation_share_of_draws", step,
                            agg.truncationShareOfDraws());
        RL::Diag::scalarRow(tb, "sum_captures_per_game_red", step, capRedSum / n);
        RL::Diag::scalarRow(tb, "sum_captures_per_game_black", step, capBlackSum / n);
        RL::Diag::scalarRow(tb, "sum_mean_plies", step, agg.meanPlies());
        RL::Diag::scalarRow(tb, "sum_distinct_openings", step, (double)agg.distinctOpenings());
        std::printf("  CSV 汇总行已写入 (step=%lld)\n", step);
    }

    if (!g_cfg.savePrefix.empty()) {
        if (ag.saveModel(g_cfg.savePrefix)) {
            std::printf("  权重已保存: %s_actor / %s_critic\n",
                        g_cfg.savePrefix.c_str(), g_cfg.savePrefix.c_str());
        } else {
            std::printf("  [错误] 权重保存失败: %s\n", g_cfg.savePrefix.c_str());
            return 1;
        }
    }
    return 0;
}
