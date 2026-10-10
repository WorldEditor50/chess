/*
 * train_bc_main.cpp - 行为克隆 (BC) 训练器: 把 Alpha-Beta 的选点蒸馏进**策略头**
 * ============================================================================
 *
 * 做什么: 一个进程里跑完一整条 BC 流程 ——
 *   1. 随机开局造 N 个局面 (确定性, 只由 --seed 决定);
 *   2. 用 **ABAgent 深度 D 的选点**当监督标签 (一步 = 一个 one-hot 目标);
 *   3. 只训练**策略头** (actor): 合法列上的掩码交叉熵, 批累积 + 一次 RMSProp;
 *   4. 报告 训练/留出 两套读数 (prior top-1 / P(老师着法) / CE / 策略熵);
 *   5. 可选的 `--load` / `--save` (权重与界面的 agent 同一份格式, 结构指纹照旧)。
 *
 * 支持四种 agent (都走同一套 BC 口径, 见 src/bcagent.hpp):
 *   --agent=ppo           PPO+MCTS (TB 专家骨干, 界面现役; 建网/内存都很贵)
 *   --agent=ppo-mlp       PPO+MCTS (MLP 专家骨干; 便宜 ~25x, 冒烟与小实验用它)
 *   --agent=ppo-layer     PPO+MCTS (廉价专家骨干: Layer<Gelu>, **无 MHA**; 2026-10)
 *   --agent=ppo-seq       PPO+MCTS (**真正的**序列 Transformer 专家: 按格 token 化 +
 *                          逐行 softmax 注意力 + 位置嵌入 + Pre-LN 残差; 2026-10)
 *   --agent=sac           SAC+AZ (纯 MLP, 界面 AGENT_SACAZ)
 *   --agent=sac-moe-mlp   SAC+AZ-MoE-MLP
 *   --agent=sac-moe       SAC+AZ-MoE (TB 专家; 建网数秒)
 *
 * ⚠ [2026-10] **流程本身不在这里** —— 训练循环、留出集划分、读数口径与报告的措辞
 *   都在 `src/bcrun.hpp` 里, 与**界面上的"行为克隆预训练"按钮** (ChessBoard::
 *   startBehaviorCloning) 共用同一份。本文件只剩三件事: 解析参数、造 agent、打印。
 *   这样"界面上报 42%、命令行报 35%"这种分歧不可能发生 (它只可能来自抄错了一行,
 *   而这里没有第二行可抄)。想看这两条路怎么共用一份实现, 见 bcrun.hpp 的文件头。
 *
 * ----------------------------------------------------------------------------
 *  为什么需要它 (历史背景, 全部有实测记录)
 * ----------------------------------------------------------------------------
 *  * 本工程 2026-09 量过: "只蒸馏价值头"让留出 MSE 降 4.2 倍, 但**选点几乎没动**
 *    (docs/training_optimization.md §7.7) —— 因为先验 (actor) 还是随机的, 搜索只能
 *    靠随机先验铺开, 好价值头无从发挥。
 *  * 于是那一步改做 **actor 的行为克隆**: 1500 局面上 prior top-1 从 0% 到 4.8%
 *    (留出 7.0%), 端到端"与 AB 深度 4 的完整着法一致率" 3.0% -> 16.0% (§7.8);
 *    加到 20000 局面后 top-1 33.6%、一致率 28.0% (§7.9)。
 *  * 但那条路当时只存在于 `bench_ppo_distill --actor=1` 里, 而且是**全量 8100 维
 *    softmax** 的旧口径 —— 在线训练 (R2) 早已换成"合法列上 Z≡1"的口径, 两者
 *    归一化口径不同 (合法集上的质量 Z 实测 0.51~0.59)。本工具把它落成**生产路径**:
 *    与在线训练同一个学习问题 (见 RL::PPO::bcGradSparse), 全量口径降级为对照臂
 *    (`--masked=0`)。
 *  * SAC 那三支**在此之前没有任何 BC 路径**; 它们的策略是"掩码 softmax + logits",
 *    于是 BC 的目标就是 `dL/dz = π − t` (恰好等于经过雅可比的 −t/π 形式, 但没有
 *    那个在 π→0 时爆掉的除法)。
 *
 * ----------------------------------------------------------------------------
 *  ⚠ 三条边界 (读数必须与它们一起看)
 * ----------------------------------------------------------------------------
 *  1. **这是模仿, 上限就是老师** (AB 深度 D)。它给的是"课程起点", 不是终点。
 *  2. **prior top-1 上升不是棋力证据** —— 它正是 BC 的训练目标本身。本工程实测过
 *     "一致率涨 9 倍而胜率一动没动" (§7.10)。
 *  3. **串了视角会静默变成零信息样本**: 老师的着法必须在"与状态编码同一个 color"
 *     的动作帧里。本工具把三种失败分开数出来, 而不是让它们变成"跑得挺快但什么也没学"。
 *
 * 用法:
 *   cmake --build <build> --target train_bc
 *   <build>/train_bc --agent=ppo-mlp --positions=4000 --depth=3 --epochs=8 ^
 *                    --opening=8 --lr=0.002 --save=weights/bc_ppo
 *   <build>/train_bc --agent=sac --positions=4000 --depth=3 --epochs=8
 *   <build>/train_bc --agent=ppo --load=weights/ppomcts_agent --save=weights/ppomcts_bc
 *   <build>/train_bc --agent=ppo-mlp --masked=0 ...      :: 全量 8100 口径对照臂
 *
 *   ---- [2026-10] 骨干对照 (同一份数据/老师/seed/lr/batch, 只换专家类型) ----
 *   <build>/train_bc --agent=ppo-mlp   --positions=1000 --depth=4 --soft=1 --soft-depths=4 ^
 *                    --epochs=8 --batch=64 --lr=0.002 --seed=20240901 --holdout=5
 *   <build>/train_bc --agent=ppo       <同上>      :: 现役 TB 专家 (52.4 M 参数)
 *   <build>/train_bc --agent=ppo-layer <同上>      :: 无 MHA 廉价专家 (11.7 M 参数)
 *   实测 (2026-10, 本机): 三臂的 CE 轨迹同量级 (训练 3.65 -> 2.05, 留出 3.65 -> 2.69),
 *   而**每次更新的代价** 10.2 s / 254.1 s / ~15 s —— 见
 *   docs/tb_expert_training_2026_10.md §4。
 *
 *   ---- [2026-10] 梯度裁剪口径对照 (PPO 这条线) ----
 *   <build>/train_bc --agent=ppo --grad-clip=legacy|global|none [--grad-clip-norm=1.0]
 *   `legacy` = 老行为 (逐张量单位范数归一, 逐位不变); `global` = 全网范数裁剪。
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "abagent.h"
#include "bcagent.hpp"
#include "bcrun.hpp"
#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/util.hpp"
#include "sacazagent.h"
#include "sacazmoemlpagent.h"
#include "sacazmoetbagent.h"

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    std::string agent = "ppo-mlp";
    BCRun::Config bc;            /* 流程参数 (positions/depth/epochs/...) */
    int reportEvery = 1;
    int hidden = 64;
    int expert = 64;
    /* PPO: false = 全量 8100 维口径 (对照臂, 与 bench_ppo_distill 的旧口径同源) */
    bool masked = true;
    /* SAC: 共享骨干口径 (一个骨干 + 三个头) —— 界面的 SAC+AZ-MoE 那一支用它 */
    bool sharedTrunk = false;
    /*
       [2026-10] 梯度裁剪口径 (只对 PPO 这条线生效; 见 rl/optimize.h 的 GradClipMode)。
       默认 = 老行为 (逐张量单位范数归一), 所以不给参数时读数与改动前逐位相同。
       `ppoClip` 是命令行字符串, 建 agent 时解析 (解析失败直接报错退出, 不静默降级)。
    */
    std::string gradClip = "legacy";
    float gradClipNorm = 1.0f;
    /*
       [2026-10] 序列专家的两个旋钮 (默认 = 老行为):
         posKey   : "1d"(可学 90 个位置向量) / "2d"(可学 10 行 + 9 列) / "rope"(无参数旋转);
         dropoutP : 0 = 关闭 (不消耗随机数)。
       只对 `--agent=ppo-seq` 有效; 其它骨干会**响亮失败** (见 main 里的检查)。
    */
    std::string posKey = "1d";
    float dropoutP = 0.0f;
    std::string loadPrefix;
    std::string savePrefix;
};

Cfg g_cfg;

double nowSec()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e9;
}

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
        if (k == "--agent")        { g_cfg.agent = v; }
        else if (k == "--positions") { g_cfg.bc.positions = std::atoi(v.c_str()); }
        else if (k == "--depth")   { g_cfg.bc.teacherDepth = std::atoi(v.c_str()); }
        else if (k == "--opening") { g_cfg.bc.openingPlies = std::atoi(v.c_str()); }
        else if (k == "--epochs")  { g_cfg.bc.epochs = std::atoi(v.c_str()); }
        else if (k == "--batch")   { g_cfg.bc.batch = std::atoi(v.c_str()); }
        else if (k == "--lr")      { g_cfg.bc.lr = (float)atof(v.c_str()); }
        else if (k == "--seed")    { g_cfg.bc.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--holdout") { g_cfg.bc.holdoutEvery = std::atoi(v.c_str()); }
        else if (k == "--report-every") { g_cfg.bc.reportEvery = std::atoi(v.c_str()); }
        else if (k == "--hidden")  { g_cfg.hidden = std::atoi(v.c_str()); }
        else if (k == "--expert")  { g_cfg.expert = std::atoi(v.c_str()); }
        else if (k == "--masked")  { g_cfg.masked = (std::atoi(v.c_str()) != 0); }
        /*
           [2026-10] 软目标 (用户问: "不直接使用 onehot、通过 abagent 计算概率分布再
           进行行为克隆是否会更好?"):
             --soft=1          老师标签换成多深度一致性的分布 (见 bcagent.hpp)
             --soft-depths=N   投票深度数 (默认 = --depth)
             --soft-linear=1   深的那层更重 (w_d ∝ d), 默认每层等权
        */
        else if (k == "--soft")    { g_cfg.bc.softTargets = (std::atoi(v.c_str()) != 0); }
        else if (k == "--soft-depths") { g_cfg.bc.softDepths = std::atoi(v.c_str()); }
        else if (k == "--soft-linear") { g_cfg.bc.softLinearWeight = (std::atoi(v.c_str()) != 0); }
        else if (k == "--shared")  { g_cfg.sharedTrunk = (std::atoi(v.c_str()) != 0); }
        /*
           [2026-10] 梯度裁剪口径 (PPO 这条线; 见 rl/optimize.h 的 GradClipMode):
             --grad-clip=legacy|global|none   (默认 legacy = 老行为, 逐位不变)
             --grad-clip-norm=1.0             (只对 global 生效: 全网梯度范数上限)
           为什么要它: `clipGrad` 的真实语义是"逐张量梯度归一化到单位长度", 它抹掉梯度
           大小的信息 (实测: 尺度 1 与 1e6 的梯度, 一步之后的 |Δw| 完全相同)。不把它做成
           一个可切换的口径, 就没有办法做"关掉裁剪"的对照实验。
        */
        else if (k == "--grad-clip") { g_cfg.gradClip = v; }
        else if (k == "--grad-clip-norm") { g_cfg.gradClipNorm = (float)atof(v.c_str()); }
        /*
           [2026-10] 两个"专家级旋钮" (补的未做结构项):
             --pos=1d|2d|rope   序列专家的位置编码 (默认 1d = 老行为, 逐位不变);
                                2d = 可学 (10 行 + 9 列), rope = 无参数旋转 (相对位置)。
             --dropout=p        dropout 概率 (默认 0 = 关闭, 不消耗随机数)。
           只对 `--agent=ppo-seq` 有效: 其它骨干的转发接口返回 false, 这里**响亮失败**
           (不静默忽略 —— 本工程被"设了却没生效"咬过多次)。
        */
        else if (k == "--pos")     { g_cfg.posKey = v; }
        else if (k == "--dropout") { g_cfg.dropoutP = (float)atof(v.c_str()); }
        else if (k == "--load")    { g_cfg.loadPrefix = v; }
        else if (k == "--save")    { g_cfg.savePrefix = v; }
        else if (k == "--max-positions") { g_cfg.bc.maxPositions = std::atoi(v.c_str()); }
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

/*
 * 一次运行的输出 = 命令行打印 + 报告文本。两条使用者的差别只有"行去哪儿":
 *   * 命令行: printf;
 *   * 界面:   信号 -> 面板。
 * 所以这里把"打印"写成回调, 流程本身完全共用 (见 bcrun.hpp)。
 */
template <class AgentT>
int runBc(AgentT &agent, const char *label, const char *backboneDesc)
{
    const double tStart = nowSec();

    std::printf("%s", BCRun::formatHeader(g_cfg.bc, label, backboneDesc,
                                         g_cfg.masked
                                             ? "合法列掩码 softmax (Z≡1, 与在线训练同一个学习问题)"
                                             : "**全量 8100 维 softmax (对照臂, 与 bench_ppo_distill 旧口径同源)**")
                            .c_str());
    std::fflush(stdout);

    BCRun::Result res;
    auto onProgress = [](const std::string &line, const BCRun::Result &) -> bool {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        return true;         /* 命令行不取消 */
    };
    if (!BCRun::run(agent, g_cfg.bc, onProgress, res)) {
        std::printf("**%s**\n", res.error.c_str());
        return 1;
    }

    const std::string caliper = BCRun::caliperFor(agent, res);
    std::printf("%s", BCRun::formatReport(g_cfg.bc, res, label, backboneDesc,
                                          caliper.c_str(), nowSec() - tStart).c_str());

    if (!g_cfg.savePrefix.empty()) {
        const double ts = nowSec();
        const bool ok = agent.saveModel(g_cfg.savePrefix);
        std::printf("\n[6] 权重保存: %s (%s, %.1f s)\n", g_cfg.savePrefix.c_str(),
                    ok ? "成功" : "**失败**", nowSec() - ts);
        if (!ok) { return 1; }
    }
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!parseArgs(argc, argv)) { return 2; }

    RL::Random::setSeed(g_cfg.bc.seed);

    Chess env;
    env.reset();

    /*
       建网可能要几秒到几十秒 (TB 专家骨干实测: SAC+AZ-MoE 建网 2.9 s, PPO+MCTS 的
       TB 骨干更久), 期间屏幕上什么都不动 —— 先把这一行打出来, 免得被读成"卡死了"。
    */
    std::printf("[0] 正在建 %s ...\n", g_cfg.agent.c_str());
    std::fflush(stdout);

    /*
       ---- `--load` 失败必须**中止**, 不能继续 ----
       载入被拒 (结构指纹 / 参数量 / CRC 不匹配) 时, 网络**保持原样** —— 也就是说接着
       训练等于"从随机权重重来", 而用户以为自己是在既有权重上继续。这种"静默地从随机
       初始化开始"在本工程里被记过很多次 (见 ppomcts_agent.cpp 的 loadModel 注释),
       所以这里直接停下: 想从随机权重开始, 就别给 `--load`。
    */
    auto loadOrFail = [](auto &agent) -> bool {
        if (g_cfg.loadPrefix.empty()) { return true; }
        const bool ok = agent.loadModel(g_cfg.loadPrefix);
        std::printf("[load] %s -> %s\n", g_cfg.loadPrefix.c_str(), ok ? "成功" : "**失败**");
        if (!ok) {
            std::printf("**--load 被拒绝 (结构指纹/参数量/CRC 不匹配) 且网络保持原样 ⇒"
                        " 继续训练等于从随机权重重来。已中止**\n"
                        "    (确实是要「从随机权重开始做 BC」就请去掉 --load)\n");
        }
        return ok;
    };

    /* ---- 造 agent (与界面 / 既有 bench 同一套形状参数) ---- */
    if (g_cfg.agent == "ppo" || g_cfg.agent == "ppo-mlp" || g_cfg.agent == "ppo-layer"
        || g_cfg.agent == "ppo-seq") {
        RL::PPO::Backbone bb = RL::PPO::Backbone::MlpExperts;
        const char *label = "PPO+MCTS (MLP 专家)";
        if (g_cfg.agent == "ppo") {
            bb = RL::PPO::Backbone::TbExperts;
            label = "PPO+MCTS (TB 专家)";
        } else if (g_cfg.agent == "ppo-layer") {
            /* [2026-10] 无 MHA 的廉价专家: 与 TB 那一支的**唯一**差别是专家类型 */
            bb = RL::PPO::Backbone::LayerExperts;
            label = "PPO+MCTS (廉价专家,无MHA)";
        } else if (g_cfg.agent == "ppo-seq") {
            /* [2026-10] 真正的序列 Transformer 专家 (token 化 + 逐行 softmax + 位置嵌入) */
            bb = RL::PPO::Backbone::SeqExperts;
            label = "PPO+MCTS (序列Transformer专家)";
        }
        PPOMCTSAgent agent(env, g_cfg.hidden, 0.99f, 0.001f, 1.414f,
                           g_cfg.expert, 0.1f, true, bb);
        agent.ppo.maskedTrainHead = g_cfg.masked;
        /* 梯度裁剪口径 (默认 legacy = 老行为; 解析失败直接退出, 不静默降级) */
        {
            int mode = RL::GRAD_CLIP_PER_TENSOR_UNIT_NORM;
            if (!RL::parseGradClipMode(g_cfg.gradClip.c_str(), mode)) {
                std::fprintf(stderr, "未知 --grad-clip: %s (可选 legacy / global / none)\n",
                             g_cfg.gradClip.c_str());
                return 2;
            }
            agent.setGradClip(mode, g_cfg.gradClipNorm);
            agent.setTrackGradNorm(true);   /* 报告里要打梯度范数 (只读一遍, 只在本工具里开) */
            std::printf("[clip] %s", RL::gradClipModeName(mode));
            if (mode == RL::GRAD_CLIP_GLOBAL_NORM) {
                std::printf(" (上限 %.3g)", (double)g_cfg.gradClipNorm);
            }
            std::printf("\n");
        }
        /*
           [2026-10] 位置编码模式 / dropout: 只对 seq 骨干有效 (转发接口返回 false)。
           响亮的失败方式: 非 seq 骨干 + 显式要求了非默认值时直接退出, 而不是"设了没生效"。
        */
        {
            const bool wantPos = (g_cfg.posKey != "1d");
            const bool wantDrop = (g_cfg.dropoutP > 0.0f);
            if (wantPos || wantDrop) {
                if (wantPos) {
                    if (!agent.setExpertPosMode(g_cfg.posKey)) {
                        std::fprintf(stderr,
                                     "[--pos=%s] 骨干 %s 不支持位置编码模式 (只有 --agent=ppo-seq 支持)\n",
                                     g_cfg.posKey.c_str(), RL::PPO::backboneName(bb));
                        return 2;
                    }
                }
                if (wantDrop) {
                    if (!agent.setExpertDropout(g_cfg.dropoutP)) {
                        std::fprintf(stderr,
                                     "[--dropout=%.3f] 骨干 %s 不支持专家 dropout (只有 --agent=ppo-seq 支持)\n",
                                     (double)g_cfg.dropoutP, RL::PPO::backboneName(bb));
                        return 2;
                    }
                }
                std::printf("[seq] 位置编码 %s, dropout %.3f\n",
                            agent.posModeInUse().c_str(), (double)agent.expertDropoutInUse());
            }
        }
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, label, RL::PPO::backboneName(agent.getBackbone()));
    }
    if (g_cfg.agent == "sac") {
        SACAZAgent agent(env, g_cfg.hidden, 0.99f, 0.001f, 1.5f,
                         g_cfg.sharedTrunk ? SACAZAgent::TrunkMode::Shared
                                           : SACAZAgent::TrunkMode::Separate);
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, "SAC+AZ", agent.backboneName());
    }
    if (g_cfg.agent == "sac-moe-mlp") {
        SACAZMoEMlpAgent agent(env, g_cfg.hidden, 0.99f, 0.001f, 1.5f, g_cfg.expert, 0.1f,
                               g_cfg.sharedTrunk ? SACAZMoEMlpAgent::TrunkMode::Shared
                                                 : SACAZMoEMlpAgent::TrunkMode::Separate);
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, "SAC+AZ-MoE-MLP", agent.backboneName());
    }
    if (g_cfg.agent == "sac-moe") {
        SACAZMoETbAgent agent(env, g_cfg.hidden, 0.99f, 0.001f, 1.5f, g_cfg.expert, 0.1f,
                              g_cfg.sharedTrunk ? SACAZMoETbAgent::TrunkMode::Shared
                                                : SACAZMoETbAgent::TrunkMode::Separate);
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, "SAC+AZ-MoE (TB 专家)", agent.backboneName());
    }

    std::fprintf(stderr, "未知 agent: %s (可选 ppo / ppo-mlp / ppo-layer / sac / "
                         "sac-moe-mlp / sac-moe)\n", g_cfg.agent.c_str());
    return 2;
}
