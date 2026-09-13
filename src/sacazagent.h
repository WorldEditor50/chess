#ifndef SACAZ_AGENT_H
#define SACAZ_AGENT_H

#include <vector>
#include <string>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <cstdint>
#include "chess.h"
#include "aiagent.h"
#include "rl/net.hpp"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"
#include "rl/parameter.hpp"

/* 稀疏 MoE 层只需要指针/引用 (定义在 rl/sparse_moe.hpp, 由 .cpp 包含) */
namespace RL {
class ISparseMoE;
}

/*
 * SACAZAgent - SAC + MCTS + AlphaZero
 * ================================================================
 *
 * 一句话: 用**最大熵 critic (SAC)** 给**AlphaZero 式 PUCT 搜索**提供叶子估值,
 * 用搜索得到的访问分布做策略改进目标, 并且**从回放缓冲里离线学习**。
 *
 * 三个成分各自负责什么
 * --------------------
 *   AlphaZero(MCTS)  : 策略改进算子。PUCT 选择 + 访问计数 -> 比当前策略更强的
 *                      目标分布 π_MCTS, 这是监督信号, 方差极低。
 *   SAC(最大熵)      : (a) **叶子估值**: 用软价值
 *                          V(s) = Σ_a π(a|s)·( min_i Q_i(s,a) − α·log π(a|s) )
 *                      而不是一个单独的价值头 —— 于是搜索是按"熵正则化的价值"
 *                      展开的, 与策略本身自洽;
 *                      (b) **双 Q + 目标网 + 回放** : 象棋奖励是稀疏终局奖励,
 *                      on-policy 方法每局只能拿到一个 ±1; 回放让一条终局经验被反复
 *                      利用, 这是 SAC 这一半真正的价值所在;
 *                      (c) **α 自动调节**: 熵低于目标就抬高 α (鼓励探索), 高于目标
 *                      就压低, 不需要手工调探索率。
 *
 * 与 `RL::SAC` 的关系
 * -------------------
 * `src/rl/sac.cpp` 已经是一个**离散动作**的 SAC (one-hot 动作 + softmax 策略 +
 * 对全部动作求期望的软备份 + 可学习 α), 本 agent 复用了它同样的数学, 但没有直接
 * 用它, 原因有三个, 都是"象棋 + 搜索"特有的:
 *   1. **合法走法掩码**: 128 维动作空间里绝大多数是非法走法, SAC 的策略必须在
 *      掩码后的分布上定义 (π(a)=0, a 非法)。`RL::SAC` 没有掩码, 也把 critic 的
 *      输入拼成 [state; prob], 与掩码策略不自洽。
 *   2. **与 MCTS 的整合**: 策略目标来自搜索访问分布, 叶子估值要换成软价值,
 *      这需要 agent 内部掌握前向/反向的顺序 (库里的 SAC 是黑盒)。
 *   3. **掩码 softmax 的梯度**: 策略头的掩码归一化必须自己实现它的雅可比
 *      (见 maskedSoftmaxBackward), 不能只对 `Layer<Softmax>` 的输出打补丁。
 * 复用的是库里的 `RL::Net` / `Layer<*> ` / `Loss::MSE` / `Optimize` / `Random`。
 *
 * 状态编码: 规范视角 14x90 = 1260 维 (与 EVABAgent 同一约定)
 *   7 种棋子 x {己方, 对方} 的 one-hot 平面, 轮到黑方时把棋盘上下镜像
 *   (x -> 9-x)。这样"轮到谁走"就被编码进视角里, 同一个网络对红黑双方都是
 *   "我方的子在自己的下方", 价值/策略不需要再从非对称编码里去猜该谁走 ——
 *   这也是 AlphaZero 必须做的规范化。注意本项目其它 RL agent (PG/DQN/PPOMCTS/
 *   DQNMCTS) 用的是 90 维非规范编码, 那种编码下价值函数的"谁走"是隐含的。
 *
 * 动作编码: 128 维 one-hot, 沿用本项目的确定性哈希 (见 stepToActionIdx)。
 *   已知局限: 该哈希会碰撞 (最多 ~50 个合法走法挤进 128 格), 碰撞会让两个不同的
 *   走法共用同一个策略/Q 槽位。选择走法是按**节点访问数**取的 (不是按动作索引),
 *   所以落子本身仍然正确; 受影响的是策略目标的精度。要彻底解决就换成无碰撞的
 *   id*90 + x*9 + y (32x90=2880) 动作空间, 见 docs/agents_design.md §11。
 */
class SACAZAgent : public AgentBase
{
public:
    /* ---- 状态编码 ---- */
    static constexpr int CELLS = 90;                  /* 10x9 */
    static constexpr int PLANES = 14;                 /* 7 类棋子 x {己方, 对方} */
    static constexpr int STATE_DIM = PLANES * CELLS;  /* 1260 */
    /* ---- 动作编码 ---- */
    static constexpr int ACTION_DIM = 128;

    /* ----------------------------------------------------------------
     *  骨干 (backbone): 决定 actor / critic 的中间层长什么样
     *
     *  为什么要有这个开关: d_model=1260 上"专家数"直接乘在算力上, 所以要把
     *  "容量"和"算力"分开量 —— 见 docs/agents_design.md §11.4。
     *  四种骨干的参数量/算力对比 (实测见 §11.4.1 与 test_sparse_moe):
     *
     *    Mlp         1260 -> h -> h -> out        基线 (算力最小, 容量也最小)
     *    SparseMoeMlp 稀疏 MoE (E=8, top-2, 专家是 MLP) + h -> out
     *                容量 8 倍于一个 MLP 专家, 算力 ~2 个专家 (与 E 无关)
     *    SparseMoeTb  稀疏 MoE (E=4, top-1, 专家是 TransformerBlock) + h -> out
     *                容量最大, 但一个 TB 专家 ~3.2 ms/前向 -> 只能配很小的模拟次数
     *    DenseMoeTb   与 SparseMoeTb **完全相同的参数**, 但 4 个专家全算
     *                这是"等参数不等算力"的对照组: 用来量稀疏路由本身值多少
     * ---------------------------------------------------------------- */
    enum class Backbone {
        Mlp = 0,
        SparseMoeMlp,
        SparseMoeTb,
        DenseMoeTb,
        Count
    };
    static const char *backboneName(Backbone b);

    /* 各骨干的固定结构 (模板参数必须编译期确定, 所以不做成运行时成员) */
    static constexpr int MOE_MLP_EXPERTS = 8;
    static constexpr int MOE_MLP_TOPK = 2;
    static constexpr int MOE_TB_EXPERTS = 4;
    static constexpr int MOE_TB_TOPK = 1;
    static constexpr int MOE_TB_HEADS = 15;   /* d_model/15 = 84 维/头; 头越多越便宜 */
    static constexpr int MOE_TB_DFF = 315;    /* = d_model/4, 压住 TB 专家的 FFN 开销 */

    /*
     * 一条 off-policy 经验。
     *
     * 局面**不存 1260 个 float** (那是 5 KB/条), 而是存"被占据的格子"的稀疏列表
     * (一局最多 32 个子, 每格一个 uint16 = plane*90+cell)。取用时再展开成 one-hot,
     * 这样 8000 条经验只有 ~6 MB, 而不是 80 MB。
     */
    struct Transition {
        std::vector<std::uint16_t> cells;      /* 当前局面: plane*CELLS + cell */
        std::vector<std::uint16_t> nextCells;  /* 下一局面 (同一编码) */
        float pi[ACTION_DIM];                  /* 策略目标 (MCTS 访问分布) */
        std::uint64_t curMaskLo = 0;           /* 当前局面合法走法掩码 (策略损失要用) */
        std::uint64_t curMaskHi = 0;
        std::uint64_t nextMaskLo = 0;          /* 下一局面合法走法掩码 (软备份要用) */
        std::uint64_t nextMaskHi = 0;
        int action = 0;                        /* 实际走的动作索引 */
        int legalCount = 1;                    /* 当前局面合法走法数 (目标熵用) */
        float reward = 0.0f;
        bool done = false;
        bool hasSearch = false;                /* pi 是否来自 MCTS (否则跳过 AZ 监督项) */
        Transition() { std::fill(pi, pi + ACTION_DIM, 0.0f); }
    };

    /* ----------------------------------------------------------------
     *  AlphaZero 式 PUCT 节点
     *  所有价值都是**当前走棋方视角** (negamax 约定)
     * ---------------------------------------------------------------- */
    struct AZNode {
        int parentID;
        int parentAction;
        Step step;

        int visitCount;
        double totalValue;      /* 累计价值 (当前走棋方视角) */
        double prior;           /* P(s,a) = 未尝试动作上的策略先验 */

        std::vector<int> childIDs;
        std::vector<int> untriedActionIndices;
        std::vector<Step> untriedSteps;
        std::vector<double> untriedPriors;   /* 与 untriedSteps 一一对应 */

        int currentColor;
        int legalCount;
        bool isTerminal;        /* 该节点已终局 (被将杀/困毙/和) */

        AZNode()
            : parentID(-1), parentAction(-1), visitCount(0), totalValue(0.0),
              prior(0.0), currentColor(Stone::COLOR_NONE), legalCount(0),
              isTerminal(false) {}
        AZNode(int pid, int pa, const Step &st, double p, int color, int legal)
            : parentID(pid), parentAction(pa), step(st), visitCount(0),
              totalValue(0.0), prior(p), currentColor(color), legalCount(legal),
              isTerminal(false) {}

        double getQ() const
        {
            return visitCount > 0 ? totalValue / (double)visitCount : 0.0;
        }
    };

    /* ----------------------------------------------------------------
     *  公开成员 (与 PPOMCTSAgent 保持一致的风格)
     * ---------------------------------------------------------------- */
    Chess &chess;

    RL::Net actor;            /* 策略: 1260 -> 骨干 -> 128 logits */
    RL::Net q1, q2;           /* 双 critic: 1260 -> 骨干 -> 128 Q 值 */
    RL::Net q1Target, q2Target;
    RL::GradValue alpha;      /* 温度 α (标量, 自动调节) */

    Backbone backbone;        /* 骨干类型 (构造时定, 之后不要改) */
    int expertHidden;         /* MLP 专家的隐层宽度 (只用 MlpExpert 骨干时有效) */
    float auxLossCoef;        /* 稀疏 MoE 负载均衡辅助损失的系数 (0 = 关掉) */    int hiddenDim;
    float gamma;
    float learningRateActor;
    float learningRateCritic;
    float learningRateAlpha;
    float c_puct;             /* PUCT 探索常数 */
    float azWeight;           /* 策略损失里 AZ 监督项(交叉熵)的权重 */
    float entropyRatio;       /* 目标熵 = entropyRatio * log(合法走法数) */
    int simulations;          /* getBestMove 默认的搜索模拟次数 */
    int batchSize;
    int replaceTargetIter;    /* 每多少次 learn 做一次 Polyak 同步 */
    std::size_t maxMemorySize;

    /* 统计 */
    int totalEpisodes;
    int totalWins[2];
    int learnSteps;
    long long m_leafEvals;
    float m_lastLoss = std::numeric_limits<float>::quiet_NaN();  /* 见 getLastTrainLoss */
    std::vector<AZNode> nodes;

    /* 回放缓冲 */
    std::deque<Transition> memories;

private:
    RL::Tensor m_stateBuf;    /* 复用, 避免每次 encode 都分配 1260 个 float */
    RL::Tensor m_logits;      /* actor 输出副本 */
    RL::Tensor m_q1, m_q2;    /* critic 输出副本 */
    /* pick 回调算出来的掩码, 供紧接其后的 onTrans 复用 (同一步之内有效) */
    int m_pendingLegalCount = 1;
    std::uint64_t m_pendingMaskLo = 0;
    std::uint64_t m_pendingMaskHi = 0;

public:
    /* ----------------------------------------------------------------
     *  编码 / 动作辅助 (rolloutFromCurrent 需要 encodeState /
     *  getLegalActions / computeReward 这三个签名)
     * ---------------------------------------------------------------- */
    /* 规范视角编码: 用 chess.sideToMove 决定视角 (MCTS 期间棋盘是同步的) */
    void encodeState(RL::Tensor &state);
    void encodeStateFor(int color, RL::Tensor &state);
    /* 稀疏编码 (存回放用) / 展开 / 稠密->稀疏 */
    void encodeSparse(int color, std::vector<std::uint16_t> &cells) const;
    static void expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state);
    static void denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells);

    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);
    int stepToActionIdx(const Step &s);
    float computeReward(const Step &s, int color);

    /* ----------------------------------------------------------------
     *  掩码 softmax 与它的反向
     * ---------------------------------------------------------------- */
    /* π(a) = m_a·exp(z_a) / Σ_b m_b·exp(z_b)  (数值稳定) */
    static void maskedSoftmax(const RL::Tensor &logits, const RL::Tensor &mask,
                              RL::Tensor &pi);
    /*
     * 掩码 softmax 的雅可比: 给定 dL/dπ = g, 返回 dL/dz。
     *   dπ_a/dz_c = π_a(δ_ac − π_c)     (c 合法)
     *   dL/dz_c   = π_c·( g_c − Σ_a g_a·π_a )
     * 形式与 `Layer<Softmax>` 的 jacobian_transpose_mul 相同, 但用的是**掩码后**
     * 的 π —— 所以不能对 Softmax 层的输出打补丁了事, 必须自己算。
     */
    static void maskedSoftmaxBackward(const RL::Tensor &pi, const RL::Tensor &g,
                                      RL::Tensor &dz);
    /* 合法掩码 <-> 位图 */
    static void maskToBits(const RL::Tensor &mask, std::uint64_t &lo, std::uint64_t &hi);
    static void bitsToMask(std::uint64_t lo, std::uint64_t hi, RL::Tensor &mask);

    /* ----------------------------------------------------------------
     *  网络前向 / 软价值
     * ---------------------------------------------------------------- */
    /* 按 backbone 造一个网络 (withGrad=false 用于目标网) */
    RL::Net buildNet(bool withGrad) const;
    /* 稀疏 MoE 的坍缩诊断: actor 的第一个稀疏 MoE 层的使用计数 */
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();
    int moeExpertCount() const;
    int moeTopK() const;
    /* 掩码策略 π(·|s) */
    void policy(const RL::Tensor &state, const RL::Tensor &mask, RL::Tensor &pi);
    /* 在线双 Q (搜索与策略损失用) */
    void qValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out);
    /* 目标双 Q (软备份用) */
    void qTargetValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out);
    /*
     * 软价值 V(s) = Σ_{a 合法} π(a)·( min_i Q_i(s,a) − α·log π(a) )
     *             = E_π[min Q] + α·H(π)          (H = −E_π[log π] ≥ 0)
     * **熵项是加上的**: 因为 −log π ≥ 0, 这正是 SAC 的软价值定义 (最大熵目标里
     * 熵是奖励的一部分)。所以 α 越大, "自己还有多种好选择"的局面估值越高。
     * α = 0 时退化成"策略期望 Q"。
     * 纯函数形式: Q 由调用方给 —— 搜索用在线网, 学习时的备份用目标网。
     */
    float softValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                        const RL::Tensor &q1In, const RL::Tensor &q2In) const;

    /* ----------------------------------------------------------------
     *  MCTS
     * ---------------------------------------------------------------- */
    double getPUCT(int childID, int parentVisits) const;
    /* 终局判定: 终局则给出 ±1/0 (color 视角) 并返回 true */
    bool terminalValue(int color, double &value) const;
    /* 访问分布: π_MCTS(a) ∝ N(a) */
    void visitDistribution(int rootID, RL::Tensor &pi);

    /* ----------------------------------------------------------------
     *  决策 / 训练
     * ---------------------------------------------------------------- */
    /* piOut != nullptr 时回填根节点的访问分布 (策略目标) */
    Step selectMove(int color, int simulations_, float temp = 0.0f,
                    RL::Tensor *piOut = nullptr);
    /* 从回放缓冲采一个 mini-batch 更新一次 (critic / actor / α), 返回 critic loss */
    float learnBatch(int batchSize_);
    /* 把 getResult 的返回值换算成 color 视角的 ±1/0; 未结束返回 false */
    static bool resultValue(int result, int color, float &out);
    void trainSelfPlay(int episodes, int simulations_, int maxMoves = 200,
                       bool verbose = true, float tempRoot = 1.0f,
                       float tempFinal = 0.25f, int learnEveryMoves = 4);
    void warmupFromCurrent(int episodes = 2, int simulations_ = 40,
                           int maxMoves = 60);

    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* ----------------------------------------------------------------
     *  AgentBase
     * ---------------------------------------------------------------- */
    SACAZAgent(Chess &chess_,
               int hiddenDim_ = 64,
               float gamma_ = 0.99f,
               float lr = 0.001f,
               float cpuct = 1.5f,
               Backbone backbone_ = Backbone::Mlp,
               int expertHidden_ = 64,
               float auxLossCoef_ = 0.1f);
    ~SACAZAgent() = default;

    Step getBestMove(int color) override;
    std::string getName() const override;
    bool exploreAndTrain(int color, int rolloutSteps) override;

    /* 统计 */
    long long getLeafEvals() const { return m_leafEvals; }
    int getLearnSteps() const { return learnSteps; }
    float getAlpha() const { return alpha[0]; }
    std::size_t getMemorySize() const { return memories.size(); }
    int getTotalEpisodes() const { return totalEpisodes; }
    /*
     * 最近一次 learnBatch 的 critic 损失 (MSE, 只对实际走过的那一步回归)。
     * 界面的"训练损失曲线"用它; 还没学过时是 NaN (曲线控件会丢弃非有限值)。
     */
    float getLastTrainLoss() const override { return m_lastLoss; }
};

#endif // SACAZ_AGENT_H
