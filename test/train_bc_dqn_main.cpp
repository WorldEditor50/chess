/*
 * train_bc_dqn_main.cpp - 两个 **Q 类 agent** 的行为克隆 (BC) 宽度扫描工具
 * ============================================================================
 *
 * 做什么 (与 train_bc 完全同一条流程, 只换被训的 agent 与"策略"的定义):
 *   1. 随机开局造 N 个局面 (确定性, 只由 --seed 决定) —— `BCRun::makePosition`;
 *   2. 用 **ABAgent 深度 D 的选点**当监督标签 (one-hot, 或 --soft=1 的多深度一致性分布);
 *   3. 训练 **Q 隐含的合法列先验**: π = softmax_合法(Q / T), T = 1.0,
 *      损失 = 合法列上的掩码交叉熵, 每批一次 RMSProp (见 src/bcdqnagent.hpp);
 *   4. 报告 训练/留出 两套读数 (prior top-1 / P(老师) / CE / 熵) + 口径自检;
 *   5. 可选的 `--load` / `--save`。
 *
 *   --agent=dqnab         DQN+AB (dueling 双头 V/A, AB 当 planning head)
 *   --agent=dqnmcts-moe   DQN+MCTS (稀疏 MoE + PUCT, 128 槽哈希动作空间)
 *
 * ----------------------------------------------------------------------------
 *  为什么需要这个工具 (它是本轮任务的**目的**, 不是附赠品)
 * ----------------------------------------------------------------------------
 *  两支 Q agent 在此之前**没有任何 BC 路径**, 于是"专家宽度 (tokDim) / 主干瓶颈宽度
 *  (hidden) 取多少"这个问题只能靠 RL 的胜率回答 —— 而胜率是对局级的噪声, 需要几百局
 *  才看得出一档差别。BC 提供了一个**廉价的代理读数**: 同一份数据、同一个老师、同一个
 *  seed 下, 留出 CE / top-1 对宽度的**单调性**是几秒~几分钟能测出来的。
 *
 *  ⚠⚠ 三条边界 (读数必须与它们一起读; 完整版在 src/bcdqnagent.hpp 的文件头)
 * ----------------------------------------------------------------------------
 *  1. 这里监督的是 **"Q 隐含的合法列先验"**, **不是** agent 的真实决策策略:
 *     DQNAB 的真实决策走根搜索值 softmax(v_a/T), DQNMCTS 走 PUCT + priors。
 *     "top-1 涨了" = "根节点 Q 的排序更像老师了", **不是**棋力证据
 *     (本工程实测过"一致率涨 9 倍而胜率一动没动", docs/training_optimization.md §7.10)。
 *  2. **CE 的绝对值与 PPO/SAC 的 BC 不可比** (那两边的 logits 是策略头, 这边是值头,
 *     量纲差几个数量级)。只有 "同 teacher / 同数据 / 同 seed 下换宽度" 的相对比较成立。
 *  3. **两支的分母不同**: DQNAB 是 8100 列**双射**; DQNMCTS 是 128 个**哈希槽**(有别名)。
 *
 * ----------------------------------------------------------------------------
 *  这个二进制里的两个编译期开关 (与本工具的语义绑死)
 * ----------------------------------------------------------------------------
 *  `-DDQNAB_MOE_EXPERT_SEQ=1 -DDQNMCTS_MOE_EXPERT_SEQ=1`
 *  (`CMakeLists.txt` 的 `train_bc_dqn` 目标) ⇒ 两支的专家**都是序列 Transformer**,
 *  于是 `--expert=` 在两边都是同一个东西: **专家的 token 宽 tokDim**
 *  (序列专家自己要求 `tokDim % MOE_SEQ_HEADS(4) == 0`, 不满足会 abort)。
 *  ⚠ 这也正是本工具必须自己编 `dqnabagent.cpp` / `dqnmctsmoetbagent.cpp` 的原因:
 *  宏决定专家类型 (模板实参) 与 `STATE_DIM`, 而 `CHESS_CORE` 是在**不带宏**的口径下
 *  编出来的 (与 `train_bc_sacaz_seq` 同一条理由)。
 *
 *  ⚠ `DQNMCTSMOETbAgent` 的 `denseMoe` / `c_puct` / `eps` 本工具固定为
 *    (false / 1.5 / 1.0) —— 与 bench/test 的默认口径一致; 它们不是本工具要扫的轴。
 *    (`--dense=1` 的等参数对照臂在 `bench_dqnmcts_moe` 里。)
 *
 * 用法:
 *   cmake --build <build> --target train_bc_dqn
 *   <build>/train_bc_dqn --agent=dqnab --positions=300 --depth=4 --epochs=8 ^
 *                        --batch=64 --lr=0.002 --holdout=5 --hidden=64 --expert=64 ^
 *                        --soft=1 --soft-depths=4 --seed=20240901
 *   <build>/train_bc_dqn --agent=dqnmcts-moe --positions=300 --depth=4 --epochs=8 ^
 *                        --batch=64 --lr=0.002 --holdout=5 --hidden=64 --expert=64
 *
 *   ---- 宽度扫描 (本工具的用途: 只换一个宽度, 其余全部固定) ----
 *   for e in 32 64 128; do <build>/train_bc_dqn --agent=dqnab --expert=$e ... ; done
 *   for h in 32 64 128; do <build>/train_bc_dqn --agent=dqnmcts-moe --hidden=$h ... ; done
 *
 * 流程本身**不在这里** (训练循环 / 留出集划分 / 报告措辞都在 src/bcrun.hpp):
 * 本文件只有 "解析参数 -> 造 agent -> 打印" 三件事 —— 这样命令行与界面报告的数是同一个数。
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
#include "dqnabagent.h"
#include "dqnmctsmoetbagent.h"
#include "rl/util.hpp"

namespace {

struct Cfg {
    std::string agent = "dqnab";
    BCRun::Config bc;             /* 流程参数 (positions/depth/epochs/...) */
    int hidden = 64;              /* 主干之后那一层瓶颈的宽度 (两支都支持) */
    int expert = 64;              /* 序列专家的 token 宽 (= --expert; 必须是 4 的倍数) */
    bool quiet = false;           /* 不打每个 epoch 的进度行 (报告照打) */
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
        /* 软目标 (与 train_bc 同一口径, 便于跨工具对照): 多深度一致性 = 分布 */
        else if (k == "--soft")        { g_cfg.bc.softTargets = (std::atoi(v.c_str()) != 0); }
        else if (k == "--soft-depths") { g_cfg.bc.softDepths = std::atoi(v.c_str()); }
        else if (k == "--soft-linear") { g_cfg.bc.softLinearWeight = (std::atoi(v.c_str()) != 0); }
        else if (k == "--quiet")   { g_cfg.quiet = (std::atoi(v.c_str()) != 0); }
        else if (k == "--load")    { g_cfg.loadPrefix = v; }
        else if (k == "--save")    { g_cfg.savePrefix = v; }
        else if (k == "--max-positions") { g_cfg.bc.maxPositions = std::atoi(v.c_str()); }
        else {
            /*
               未知参数**响亮失败**: 静默忽略一个拼错的开关 = "设了却没生效"
               (本工程记过很多次, 见 src/bcagent.hpp 顶部那段"四个副本"的由来)。
            */
            std::fprintf(stderr, "未知参数: %s\n"
                                 "(可用: --agent=dqnab|dqnmcts-moe --positions --depth --opening --epochs\n"
                                 "       --batch --lr --seed --holdout --soft --soft-depths --soft-linear\n"
                                 "       --report-every --hidden --expert --max-positions --load --save --quiet)\n",
                         a.c_str());
            return false;
        }
    }
    return true;
}

/*
 * 一次运行的输出 = 命令行打印 + 报告文本 (与 train_bc 逐字同一条用法:
 * 流程在 BCRun::run 里, 这里只决定"行去哪儿")。
 */
template <class AgentT>
int runBc(AgentT &agent, const char *label, const char *backboneDesc, const char *modeDesc,
          const char *updatesDesc)
{
    const double tStart = nowSec();

    std::printf("%s", BCRun::formatHeader(g_cfg.bc, label, backboneDesc, modeDesc).c_str());
    /*
       ⚠ formatHeader 的最后一行写死了 "(只更新策略头)" —— 那句话是给 PPO/SAC 写的,
       对 Q agent **不成立** (DQN 没有策略头)。那行字在 bcrun.hpp 里, 改它就会改掉
       `train_bc` 的既有输出 (本任务的纪律是"既有读数一位不变"), 所以这里**补一行**
       把话说明白, 而不是让报告挂着那句错的。
    */
    std::printf("补正     : 上面那行「只更新策略头」对 Q agent 不成立 —— 本路径更新的是 "
                "主干 + %s;\n"
                "           DQNAB 的 V 头、两支的目标网与回放池: 一个字节不动;\n"
                "           [4] 里的「N 次 actor 更新」= 本层的**批更新次数** "
                "(Q agent 没有 actor, 那就是 Q/优势头的梯度步)\n",
                updatesDesc);
    /*
       ---- 结构指纹: 本工具扫的是**宽度**, 所以必须有一行"网到底多大" ----
       为什么它不是装饰: `--hidden=` / `--expert=` 是**结构**参数, 而"旋钮设了却没生效"
       在本工程是记过账的事故类型 (构造之后再赋值 = 静默空操作)。参数量随宽度变化是
       "这次真的建了另一个结构"的**直接证据** (扫宽度时把这一行抄进日志,
       两个宽度参数量一模一样就说明有一个旋钮是死的)。
    */
    std::printf("结构     : 参数量 主干 %lld + 头 %lld"
                " (由 --hidden=%d / --expert(tokDim)=%d 决定; 与宽度绑定, 不是常数)\n",
                agent.trunkParamCount(), agent.headParamCount(),
                g_cfg.hidden, g_cfg.expert);
    std::fflush(stdout);

    BCRun::Result res;
    auto onProgress = [](const std::string &line, const BCRun::Result &) -> bool {
        if (g_cfg.quiet && line.compare(0, 4, "[3] ") == 0) {
            return true;            /* --quiet: 每个 epoch 那一行不打 (报告照打) */
        }
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        return true;                /* 命令行不取消 */
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

    if (g_cfg.agent != "dqnab" && g_cfg.agent != "dqnmcts-moe") {
        std::fprintf(stderr, "未知 agent: %s (可选 dqnab / dqnmcts-moe)\n", g_cfg.agent.c_str());
        return 2;
    }
    /*
       `--expert=` 是**序列专家的 token 宽** (本目标两个宏都开着 ⇒ 两支的专家都是它)。
       前置检查放在这里 (而不是 agent 里): 判据只有一处 —— 序列专家自己在构造时判
       `tokDim % MOE_SEQ_HEADS == 0` 并 abort; 这里只是把那次 abort 提前成一句人话,
       免得扫描脚本莫名其妙地死在一个 assert 上。
    */
    if (g_cfg.expert <= 0 || (g_cfg.expert % 4) != 0) {
        std::fprintf(stderr,
                     "--expert=%d 不合法: 序列专家的 token 宽必须是 MOE_SEQ_HEADS(4) 的倍数 "
                     "(否则专家构造时 abort, 见 rl/seq_transformer.hpp)。\n",
                     g_cfg.expert);
        return 2;
    }

    Chess env;
    env.reset();

    /* 建网可能要几秒 (序列 MoE 骨干), 先把这一行打出来, 免得被读成"卡死了" */
    std::printf("[0] 正在建 %s (hidden=%d, expert(tokDim)=%d) ...\n",
                g_cfg.agent.c_str(), g_cfg.hidden, g_cfg.expert);
    std::fflush(stdout);

    /*
       ---- `--load` 失败必须**中止** ----
       载入被拒 (结构指纹/参数量/CRC 不匹配) 时网络保持原样 ⇒ 接着训练等于"从随机权重
       重来", 而用户以为自己是在既有权重上继续 (与 train_bc 同一条纪律)。
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

    if (g_cfg.agent == "dqnab") {
        /*
           构造参数顺序 (见 src/dqnabagent.h): (chess, hiddenDim, gamma, lr, backbone, expertHidden)。
           `backbone` 与 `expertHidden` 都是**结构**参数: 建网在构造函数里发生,
           构造之后再赋值是静默空操作 —— 所以 `--hidden=` / `--expert=` 必须走构造参数。
        */
        DQNABAgent agent(env, g_cfg.hidden, 0.99f, 0.001f,
                         DQNABAgent::Backbone::SparseMoeTb, g_cfg.expert);
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, "DQN+AB (dueling, 稀疏MoE专家)",
                     DQNABAgent::MOE_EXPERT_SEQ ? "稀疏MoE(序列Transformer专家, E=4 top-1)"
                                                : "稀疏MoE(TB专家, E=4 top-1)",
                     "π = 合法列上 softmax(Q/T), T=1.0 (dueling: Q = V + A − mean_legal(A))",
                     "A 头 (dueling 的优势头, m_aHead)");
    }

    /*
       DQNMCTSMOETbAgent 的构造签名 (见 src/dqnmctsmoetbagent.h):
         (chess, hiddenDim, gamma, lr, eps, c_puct, denseMoe, expertHidden)
       `denseMoe=false` = 稀疏 (TopK=1) 生产口径; eps/c_puct 取 bench/test 的默认值。
    */
    {
        DQNMCTSMOETbAgent agent(env, g_cfg.hidden, 0.99f, 0.001f, 1.0f, 1.5f, false,
                                g_cfg.expert);
        if (!loadOrFail(agent)) { return 2; }
        return runBc(agent, "DQN+MCTS (稀疏MoE专家, PUCT)",
                     agent.backboneName(),
                     "π = 合法槽位上 softmax(Q/T), T=1.0 (128 个哈希槽, 有别名)",
                     "Q 头 (qHead)");
    }
}
