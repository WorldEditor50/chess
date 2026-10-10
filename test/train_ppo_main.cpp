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
    /*
       ---- [2026-09] critic/actor 的三组新增旋钮 + 学习节拍 ----
       这三组原来**一个都没有 CLI 开关**(只在测试里直接写成员), 而按本轮对 SAC 的诊断经验,
       "口径类旋钮没有开关" 会导致结论只能靠改源码重编 (踩过"改了源码但二进制没重编")。

       `learnPerEpisode`: 一局结束后做几次批学习。**这是本节拍问题的唯一杠杆** ——
         `RL::PPO::learnFromReplay` 无论抽多少样本都只调一次优化器 (rl/ppo.cpp),
         而每步位移又被逐张量 L2 归一化定死成 ~lr (rl/optimize.h/net.hpp 的 clipGrad),
         所以"把损失/目标乘一个常数"在 PPO 这里**完全无效**, 只有步数能改。
         实测 `--games=20` 的默认值是 **20 步** RMSProp。
       `clampValue`: critic 目标的值域约束。**注释断言的前提是错的** (势能塑形让
         |r'| 上界约 2.34 > 2), 所以"夹住多少"要先量出来 —— 诊断块会打印夹住比例。
         `0` = 关掉, 负数 = 不改 (用类默认 2.0)。
       `clipEps` / `entropyCoef`: PPO 的两个信任域/熵旋钮 (默认 0.2 / 0.01),
         `<0` = 不改。
    */
    int learnPerEpisode = 1;
    float clampValue = -1.0f;
    float clipEps = -1.0f;
    float entropyCoef = -1.0f;
    /*
       ---- [2026-10] MoE 负载均衡: 无辅助损失偏置 (Loss-Free Balancing) ----
       背景 (为什么 PPO 这条骨干需要它): 本工程实测过**只靠辅助损失 `moeAuxCoef` 拉不平
       负载** —— 在同一批真实棋局状态上 `coef` 开到 10, MaxVio 仍恒为理论最大值 3.000、
       有效专家 1.00 (根因: 辅助损失的梯度正比于饿死专家自己的门控概率 ≈0, 放大系数乘的
       仍是 0)。把均衡的作用点从"梯度"换到"top-k 的 argmax 比较项"之后, 同一实验里
       MaxVio 0.29~0.32、有效专家 4.00。见 docs/moe_gate_experiment_2026_10.md 的 [3b]。

       `lossFreeBiasRate` 的刻度坑: 论文推荐 0.001, 但那是在"学习步密集"的前提下。
       本工程的节拍很稀疏 (默认一局一步 RMSProp), 所以默认给 0.01 (与上面那份实验一致)。
       `0` = 不改 (用 agent 的类默认)。
    */
    bool  lossFreeBias = false;
    float lossFreeBiasRate = -1.0f;
    bool  auxCoefGiven = false;
    float auxCoef = -1.0f;
    /*
       ---- [2026-10] 骨干与梯度裁剪口径 ----
       `backbone`: "tb" (默认, 界面现役) / "mlp" / "layer" (无 MHA 的廉价专家)。
         三支走同一个 `makeMoeLayer` 工厂, 所以除了专家类型之外逐行相同。
       `gradClip`: "legacy" (默认 = 老行为) / "global" / "none"。见 rl/optimize.h。
         为什么必须做成开关: 老口径 `clipGrad` 的真实语义是"逐张量梯度归一化到单位长度",
         它把梯度大小的信息完全抹掉 (实测: 尺度 1 与 1e6 的梯度, 一步之后 |Δw| 相同) ——
         不做成一个可切换的口径, "关掉裁剪"的对照就只能靠改源码重编。
    */
    std::string backbone = "tb";
    std::string gradClip = "legacy";
    float gradClipNorm = 1.0f;
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
        else if (k == "--learn-per-episode") { g_cfg.learnPerEpisode = std::atoi(v.c_str()); }
        else if (k == "--clamp") { g_cfg.clampValue = (float)std::atof(v.c_str()); }
        else if (k == "--clip-eps") { g_cfg.clipEps = (float)std::atof(v.c_str()); }
        else if (k == "--entropy") { g_cfg.entropyCoef = (float)std::atof(v.c_str()); }
        else if (k == "--lossfree-bias") { g_cfg.lossFreeBias = (std::atoi(v.c_str()) != 0); }
        else if (k == "--lossfree-bias-rate") {
            g_cfg.lossFreeBiasRate = (float)std::atof(v.c_str());
        }
        else if (k == "--aux-coef") { g_cfg.auxCoef = (float)std::atof(v.c_str()); g_cfg.auxCoefGiven = true; }
        /*
           [2026-10] 骨干与梯度裁剪口径 (见 rl/ppo.h 顶部 / rl/optimize.h 的 GradClipMode)。
           `--backbone=layer` 是新加的无 MHA 廉价专家那一支 —— 与 TB 那一支的**唯一**差别
           就是专家类型 (同一个工厂 makeMoeLayer), 所以直接可做受控对照。
        */
        else if (k == "--backbone") { g_cfg.backbone = v; }
        else if (k == "--grad-clip") { g_cfg.gradClip = v; }
        else if (k == "--grad-clip-norm") { g_cfg.gradClipNorm = (float)std::atof(v.c_str()); }
        else if (k == "--quiet") { g_cfg.verbose = false; }
        else if (k == "--help" || k == "-h") {
            std::printf(
                "用法: train_ppo [--games=N] [--sims=N] [--moves=N] [--opening=N]\n"
                "                 [--load=PREFIX] [--save=PREFIX] [--csv=PREFIX] [--seed=N]\n"
                "                 [--hidden=N] [--expert=N] [--replay=N]\n"
                "                 [--temp-root=F] [--temp-final=F]\n"
                "                 [--no-shaping] [--shaping-alpha=F] [--no-material-reward]\n"
                "                 [--no-bootstrap] [--no-root-noise] [--quiet]\n"
                "                 [--learn-per-episode=K] 每局几次批学习 (默认 1; 见头文件)\n"
                "                 [--clamp=F] [--clip-eps=F] [--entropy=F]\n"
                "                 [--aux-coef=F]          MoE 辅助损失系数 (默认 0.1)\n"
                "                 [--lossfree-bias=0|1]   无辅助损失偏置均衡 (默认 0 = 关)\n"
                "                 [--lossfree-bias-rate=F] 它的步长 (默认 0.01; 见 Cfg 的刻度坑)\n"
                "                 [--backbone=tb|mlp|layer|seq] 专家骨干 (默认 tb = 界面现役;\n"
                "                                          seq = 真正的序列 Transformer, 见 rl/seq_transformer.hpp)\n"
                "                 [--grad-clip=legacy|global|none] [--grad-clip-norm=F]\n");
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
    /*
       ---- [2026-10] 骨干由 --backbone 决定 (默认 tb = 界面现役) ----
       解析失败**直接退出**, 不静默退回 tb —— "以为在跑 mlp 其实跑的是 tb" 这种错
       在本工程里只会表现为"读数不对但没什么异常", 属于最贵的一类错。
    */
    RL::PPO::Backbone bb = RL::PPO::Backbone::TbExperts;
    if (!RL::PPO::parseBackbone(g_cfg.backbone.c_str(), bb)) {
        std::printf("[错误] 未知 --backbone: %s (可选 tb / mlp / layer)\n", g_cfg.backbone.c_str());
        return 2;
    }
    PPOMCTSAgent ag(board, g_cfg.hidden, 0.99f, 0.001f, 1.414f, g_cfg.expert, 0.1f, true, bb);
    {
        int mode = RL::GRAD_CLIP_PER_TENSOR_UNIT_NORM;
        if (!RL::parseGradClipMode(g_cfg.gradClip.c_str(), mode)) {
            std::printf("[错误] 未知 --grad-clip: %s (可选 legacy / global / none)\n",
                        g_cfg.gradClip.c_str());
            return 2;
        }
        ag.setGradClip(mode, g_cfg.gradClipNorm);
        if (g_cfg.verbose) {
            std::printf("[骨干] %s | [裁剪] %s%s\n", RL::PPO::backboneName(bb),
                        RL::gradClipModeName(mode),
                        mode == RL::GRAD_CLIP_GLOBAL_NORM ? " (见 --grad-clip-norm)" : "");
        }
    }

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
       [2026-09] 三个新旋钮 + 学习节拍 (见 Cfg 的说明)。
       `learnStepsPerEpisode` 是**唯一**能改变"一次会话走多远"的杠杆 (逐张量归一化让
       "把损失乘常数"完全无效), 所以 A/B 的第一件事就是把它的实际步数打出来对数。
    */
    ag.learnStepsPerEpisode = g_cfg.learnPerEpisode;
    if (g_cfg.clampValue >= 0.0f)  { ag.ppo.clampValue = g_cfg.clampValue; }
    if (g_cfg.clipEps >= 0.0f)     { ag.ppo.clipEps = g_cfg.clipEps; }
    if (g_cfg.entropyCoef >= 0.0f) { ag.ppo.entropyCoef = g_cfg.entropyCoef; }
    /*
       ---- [2026-10] MoE 负载均衡的两条路 (可以同时开, 也可以只开一条) ----
       `setLossFreeBias` 而不是直接写成员: 单一真源在 `ppo` 里, 只改 agent 上的成员
       会静默不生效 (见 ppomcts_agent.h 的 setter 说明)。
    */
    if (g_cfg.auxCoefGiven) { ag.ppo.moeAuxCoef = g_cfg.auxCoef; ag.moeAuxCoef = g_cfg.auxCoef; }
    ag.setLossFreeBias(g_cfg.lossFreeBias,
                       (g_cfg.lossFreeBiasRate >= 0.0f) ? g_cfg.lossFreeBiasRate
                                                        : ag.lossFreeBiasRate);
    ag.ppo.resetCriticDiag();
    const int stepsBefore = ag.ppo.learningSteps;
    std::printf("[配置] 每局批学习 %d 次 (批 %d x %d epoch) | clampValue=%.2f clipEps=%.2f "
                "entropyCoef=%.3f\n",
                ag.learnStepsPerEpisode, ag.replayBatchSize, ag.replayEpochs,
                (double)ag.ppo.clampValue, (double)ag.ppo.clipEps,
                (double)ag.ppo.entropyCoef);
    std::printf("       MoE 均衡: 辅助损失 auxCoef=%.3f | 无辅助损失偏置 %s (rate=%.4g)\n",
                (double)ag.ppo.moeAuxCoef,
                ag.lossFreeBias ? "开" : "关", (double)ag.lossFreeBiasRate);

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
    /*
       ---- [2026-09] critic 目标诊断 (夹前分布 + 夹住比例 + 优化器步数) ----
       为什么必须有这一段: `clampValue=2` 的注释声称价值目标不会越界, 而势能塑形让
       |r'| 的上界约 2.34 —— "夹住多少"是可测事实, 不能靠注释断言。优化器步数则回答
       "这次会话到底学了几步" (默认一局一步, 20 局 = 20 步)。
    */
    {
        const RL::PPO::CriticDiag &d = ag.ppo.criticDiag;
        const int steps = ag.ppo.learningSteps - stepsBefore;
        if (d.total > 0) {
            std::printf("  critic 目标: |target| 均值 %.4f, 最大 %.4f, 带符号均值 %+.4f | "
                        "|V| 均值 %.4f\n",
                        d.targetAbsSum / (double)d.total, d.targetAbsMax,
                        d.targetSum / (double)d.total, d.valueAbsSum / (double)d.total);
            std::printf("              被 clampValue=%.2f 夹住 %lld/%lld = **%.1f%%**  "
                        "(夹住 = 该样本的目标退化成常数)\n",
                        (double)ag.ppo.clampValue, d.clamped, d.total,
                        100.0 * (double)d.clamped / (double)d.total);
        }
        std::printf("  优化器步数: 本轮 %d 步 (每局 %d 次批学习) | "
                    "critic/actor 的位移被逐张量 L2 归一化定死成 ~lr, 所以步数是唯一杠杆\n",
                    steps, ag.learnStepsPerEpisode);
    }

    agg.print("train_ppo");

    /* MoE 路由 (只在骨干含稀疏 MoE 时非空) */
    {
        std::vector<long long> usage;
        ag.moeUsage(usage);
        if (!usage.empty()) {
            /*
               ---- 负载读数的口径 (与 bench_gate_moe / 界面 MoE 负载条同一套) ----
               MaxVio = max_i share_i/(1/E) − 1 (0 = 完美均衡; 与 Loss-Free Balancing
                        论文同口径);
               有效专家 = 份额 > 5% 的专家个数;
               **不用 max/min** —— 它对"最弱专家是 1.1% 还是 2.7%"过敏, 容易把噪声当趋势。
               必须**按前向来源拆开**: 搜索每次决策就几十~几百次前向, 数量上压倒训练批,
               只看合计会把"搜索访问到的局面分布"当成"训练批的路由分布", 于是判不出
               均衡机制有没有生效 (见 RL::PPO::moeUsageSplit)。
            */
            struct LoadStat {
                double maxVio = 0.0, minShare = 0.0;
                int effective = 0;
                long long total = 0;
            };
            auto loadOf = [](const std::vector<long long> &counts) {
                LoadStat s;
                const int e = (int)counts.size();
                for (std::size_t i = 0; i < counts.size(); i++) { s.total += counts[i]; }
                if (e <= 0 || s.total <= 0) {
                    return s;
                }
                double mx = 0.0;
                s.minShare = 1.0;
                for (int i = 0; i < e; i++) {
                    const double sh = (double)counts[(std::size_t)i] / (double)s.total;
                    if (sh > mx) { mx = sh; }
                    if (sh < s.minShare) { s.minShare = sh; }
                    if (sh > 0.05) { s.effective++; }
                }
                s.maxVio = mx * (double)e - 1.0;
                return s;
            };

            const LoadStat all = loadOf(usage);
            std::printf("  MoE 专家使用 (合计): E=%d top-%d | MaxVio %.3f | 最小份额 %.1f%% | "
                        "有效专家 %d/%d | 未使用 %d 个\n",
                        ag.moeExpertCount(), ag.moeTopK(), all.maxVio,
                        100.0 * all.minShare, all.effective, (int)usage.size(),
                        [&usage] {
                            int z = 0;
                            for (std::size_t e = 0; e < usage.size(); e++) {
                                if (usage[e] == 0) { z++; }
                            }
                            return z;
                        }());

            std::vector<long long> trainUse, inferUse;
            ag.moeUsageSplit(trainUse, inferUse);
            if (!trainUse.empty()) {
                const LoadStat tr = loadOf(trainUse);
                const LoadStat inf = loadOf(inferUse);
                std::printf("              训练侧前向 %lld 次: MaxVio %.3f | 最小份额 %.1f%% | "
                            "有效专家 %d/%d   <== 判均衡机制看这一列\n",
                            tr.total, tr.maxVio, 100.0 * tr.minShare,
                            tr.effective, (int)trainUse.size());
                std::printf("              推理侧前向 %lld 次: MaxVio %.3f | 最小份额 %.1f%% | "
                            "有效专家 %d/%d   (搜索访问分布, 不是训练批的路由分布)\n",
                            inf.total, inf.maxVio, 100.0 * inf.minShare,
                            inf.effective, (int)inferUse.size());
                if (tr.total == 0 && ag.lossFreeBias) {
                    std::printf("              [警告] 训练侧为 0 —— 偏置回路读的是批统计, 没有训练\n"
                                "                     前向它一步都不会动 (检查 replayBatch=0?)\n");
                }
            }
            std::vector<float> bias;
            ag.moeBiasSnapshot(bias);
            if (!bias.empty()) {
                std::printf("              偏置 b_i:");
                for (std::size_t i = 0; i < bias.size(); i++) {
                    std::printf(" %+.3f", (double)bias[i]);
                }
                std::printf("   (只进 top-k 比较, 不进输出加权; 极差 %.3f)\n",
                            [&bias] {
                                float lo = bias[0], hi = bias[0];
                                for (std::size_t i = 1; i < bias.size(); i++) {
                                    if (bias[i] < lo) { lo = bias[i]; }
                                    if (bias[i] > hi) { hi = bias[i]; }
                                }
                                return (double)(hi - lo);
                            }());
            }
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
