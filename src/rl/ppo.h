#ifndef PPO_H
#define PPO_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"
#include "annealing.hpp"

namespace RL {

/*
 * Simplified PPO for AlphaZero-style Chinese Chess.
 *
 * 骨干 (2026-09 改版): 把原来的**稠密** MOE<8,4> 换成稀疏路由的
 * SparseMoE<MlpExpert, 8, 2> (见 rl/sparse_moe.hpp)。
 *
 *   actorP  : state -> SparseMoE(E=8, top-2) -> Tanh(h) -> Softmax(actionDim)
 *   critic  : state -> SparseMoE(E=8, top-2) -> Tanh(h) -> Linear(1)
 *
 * 为什么换: 稠密的 MOE 会把**全部**专家都算一遍再做门控加权和, 于是"专家数"直接
 * 乘在算力上 —— 那是稠密混合, 不是 MoE 的卖点。稀疏版只算门控选中的 top-k 个:
 *   参数量 ~ E×(每个专家)   算力 ~ k×(每个专家)      ← 算力与 E 无关
 * 实测 (docs/agents_design.md §11.4): 28.7 M 参数下 42.0 -> 10.4 ms/模拟 (4.1×)。
 *
 * 两个必须配套的东西 —— 缺任何一个, 稀疏 MoE 都学不起来:
 *
 *   1. scaleLayerInit(): 输入维数从 90 涨到上千之后, iFcLayer 默认的 U(-1,1)
 *      初始化会让 pre-activation 的标准差达到 sqrt(d/3) ≈ 22 (d=1440), Tanh 一上来
 *      就饱和、梯度接近 0, 网络基本不动。普通层按 1/sqrt(fan_in) 重缩一遍把标准差拉回
 *      ~0.6; 专家的权重由 SparseMoE 的构造函数自己用 scaleExpertInit 缩过了。
 *
 *   2. 负载均衡辅助损失 (moeAuxCoef): 没有它, softmax 的反向会把"没被选中"的专家的
 *      门控概率继续压低, 路由几轮之内就坍缩到少数专家、其余永远不训练。见
 *      sparse_moe.hpp 的 addAuxGradient() 与 test_sparse_moe 的有限差分验证。
 *
 * 全连接头 / 损失函数的学习部分与原来一致 (无 clip、无 KL 惩罚、无 actorQ):
 *   Policy loss: cross-entropy(actor(state) || MCTS 目标分布)
 *   Value loss:  MSE(critic(state), 回报)
 */
class PPO
{
public:
    /* 稀疏 MoE 骨干的结构。模板参数必须编译期确定, 所以做成常量而不是运行时成员。 */
    static constexpr int MOE_EXPERTS = 8;
    static constexpr int MOE_TOPK    = 2;

    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim,
                 int expertHidden = 64,
                 float moeAuxCoef = 0.1f,
                 /* false = 只推理 (不分配 g/v/m 梯度缓冲, 内存与构造时间约 1/4)。
                    多线程分身训练的 worker 用这个形态 —— 它们只做搜索不做反向。 */
                 bool withGrad = true);
    virtual ~PPO(){}

    /* Forward - returns policy (Softmax) probabilities */
    Tensor &action(const Tensor &state);

    /* Forward - returns scalar value V(s) */
    float value(const Tensor &state);

    /* Train on one state: cross-entropy(policy, target) + MSE(value, outcome)
     *   state       : STATE_DIM board encoding
     *   actionTarget: actionDim-dim target distribution (normalized)
     *   valueTarget : +1.0 (win for current player), -1.0 (loss), 0.0 (draw) */
    void trainStep(const Tensor &state,
                   const Tensor &actionTarget,
                   float valueTarget,
                   float lr);

    /* Self-play training: compute discounted returns and train PPO on trajectory
     *   trajectory   : list of (state, action_onehot, reward)。reward 必须是**走子方
     *                  视角**的即时奖励 (与 PPOMCTSAgent::computeReward 同一口径)
     *   finalOutcome : 终局结果, 从**最后一步走子方**的视角:
     *                  +1 胜 / -1 负 / 0 和 (或"回放被截断、没有终局"时给 0)。
     *                  函数内部会做视角逐手翻转 —— 细节见 ppo.cpp 里 learnSelfPlay
     *                  的推导注释。给错视角的话, 训练目标会在一半的样本上整体反号,
     *                  而且不会报任何错。 */
    void learnSelfPlay(std::vector<Step>& trajectory,
                       float finalOutcome,
                       float learningRate);

    /*
       按"视角逐手翻转"计算折现回报 (推导见 ppo.cpp 的 learnSelfPlay)。

       独立成 public 纯函数是为了能被单元测试**直接查数值**: 这个符号约定写错不会
       报任何错, 只会让一半样本的目标反号, 而这种错误只能靠"手算一个 2~3 步的
       小例子对一下"来发现。口径与 learnSelfPlay 完全一致 —— finalOutcome 按
       **最后一步走子方**的视角给。
    */
    std::vector<float> discountedReturns(const std::vector<Step> &trajectory,
                                         float finalOutcome) const;

    /* 与 discountedReturns 同一口径, 但直接吃"每步即时奖励"数组 —— 自对弈时轨迹里
       只留 (state, 稀疏策略目标, reward), 不必构造 RL::Step 里的 8100 维稠密目标。 */
    std::vector<float> discountedReturnsFromRewards(const std::vector<float> &rewards,
                                                    float finalOutcome) const;

    /* ================================================================
     *  梯度累积 (P3) + 回放池 (P4), 2026-09
     * ----------------------------------------------------------------
     *  原来 trainStep 是"一条样本 -> 前向+反向+优化器"。实测 (test_ppomcts 的
     *  compute budget 一节) 一步 trainStep 里 **优化器占 66%、前向只占 1%** ——
     *  每样本单独做一次全参数 RMSProp 基本是纯浪费。DQN/SAC 那边本来就是"累批再
     *  更新" (rl/dqn.cpp 的 learn(): 循环 experienceReplay 累积, 最后只调一次
     *  RMSProp), PPO 是唯一漏掉的那个。
     *
     *  但**累积本身不是净赢**: 它把优化器成本按 batchSize 摊薄, 同时把优化器**步数**
     *  也除以 batchSize —— 没有回放池时, 你只是用"更少但更便宜的更新"换了原来那批
     *  更新。必须配回放池才能真赚到: 同一批数据可以反复过很多遍, 每遍都产生新更新。
     *  实测成本基准 (同一台机器): 重新生成一条样本 ≈ 55 ms, 重放一条 ≈ 20 ms,
     *  重放 + 累积 ≈ 7 ms —— 复用比重生成便宜 3~8 倍。
     * ================================================================ */

    /*
       回放池样本。刻意省内存:
         state          : STATE_DIM 稠密 (1440 float = 5.8 KB)
         actionIdx/...  : 策略目标只存**根的访问分布里的非零项** (~40 项),
                          而不是 8100 维稠密分布 (32 KB) —— 这就是"轨迹只存 action
                          index"的落地方式, 而且不丢 P5 那套软目标。
    */
    struct ReplaySample {
        Tensor state;
        std::vector<int>   actionIdx;   /* 访问分布的非零动作下标 */
        std::vector<float> actionProb;  /* 与 actionIdx 等长, 和 ≈ 1 */
        float valueTarget;
    };

    /* 池容量 (条数)。20000 x 1440 x 4B ≈ 115 MB —— 这是稠密状态存储的代价。
       要放到 10^5 条以上就该像 SACAZAgent 那样把状态也存成稀疏 (平面编码里只有
       几十个非零格; 一局 32 个子 + 威胁平面 ≈ 200 项), 那是后续项。 */
    std::size_t replayCapacity = 20000;
    std::deque<ReplaySample> replay;

    /* 清掉 MoE 门控批统计, 让负载均衡辅助损失只反映本批的训练前向 */
    void resetMoeBatchStats();
    /* 只累积梯度 (前向 + 反向), 不碰优化器 */
    void accumulateGrad(const Tensor &state, const Tensor &actionTarget, float valueTarget);
    /* 把累积的梯度一次性应用 (注入 MoE 辅助损失 -> RMSProp), 然后清零 */
    void applyGradients(float lr);

    void addReplay(const Tensor &state,
                   const std::vector<int> &actionIdx,
                   const std::vector<float> &actionProb,
                   float valueTarget);
    /*
       从回放池随机采样 batchSize 条、过 epochs 遍, 累积梯度后做**一次**优化器更新。
       返回 false 表示池子不够大或 epochs<=0 (什么都没做)。
       两处 loss 上报取**批平均**(与 DQN 报平均 TD 误差同一个口径)。
    */
    bool learnFromReplay(std::size_t batchSize, int epochs, float lr);
    std::size_t replaySize() const { return replay.size(); }
    void clearReplay() { replay.clear(); }
    /*
       取走整个回放池并把自己清空 (deque 的 swap 是 O(1))。

       多线程分身训练用它把样本从 worker 交给共享池: worker 用自己的 agent 跑自对弈
       (replayBatchSize=0 时 commitEpisode 只入池不学习), 每局结束后把池子整体搬走。
       刻意不在 RL::PPO 里加锁 —— 让这个类保持单线程, 由调用方决定怎么并发。
    */
    std::deque<ReplaySample> takeReplay()
    {
        std::deque<ReplaySample> out;
        out.swap(replay);
        return out;
    }

    /* Save / Load weights */
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);

    /* ----------------------------------------------------------------
     *  稀疏 MoE 诊断 (只读, 不参与任何计算; 给测试与调参用)
     * ---------------------------------------------------------------- */
    /* 两个网络的稀疏 MoE 层数 (actor + critic, 应为 2) */
    int moeLayerCount() const;
    int moeExpertCount() const;
    int moeTopK() const;
    /* 参数量 (只读诊断) */
    long long actorParamCount() const { return actorP.paramCount(); }
    long long criticParamCount() const { return critic.paramCount(); }
    /* 把两个网络的专家使用次数**逐专家相加**输出 (长度 = 专家数) */
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();

public:
    int stateDim;
    int actionDim;
    int expertHidden;      /* MLP 专家的隐层宽度 */
    float gamma;
    float exploringRate;
    /* 负载均衡辅助损失系数, <=0 关闭。与 SACAZAgent 的默认值一致 (0.1) */
    float moeAuxCoef;
    int learningSteps;
    /*
     * 最近一次 trainStep 的标量损失 (界面"训练损失曲线"用, **不参与任何计算**):
     *   lastLoss      : critic 的价值 MSE (主曲线用它 —— 与 DQN 的 TD 误差同类)
     *   lastActorLoss : actor 的交叉熵 (策略离"搜索给的走法"有多远)
     */
    double lastLoss = std::numeric_limits<double>::quiet_NaN();
    double lastActorLoss = std::numeric_limits<double>::quiet_NaN();

    /* 批内损失累加 (accumulateGrad 攒, applyGradients 取平均后写进上面两个) */
    double batchLossSum = 0.0;
    double batchActorLossSum = 0.0;
    std::size_t batchLossCount = 0;

    Net actorP;      /* Policy network:  state -> actionDim Softmax */
    Net critic;      /* Value network:   state -> 1-dim scalar    */
};

} // namespace RL
#endif // PPO_H
