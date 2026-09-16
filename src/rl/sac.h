#ifndef SAC_H
#define SAC_H
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
#include "layer.h"
#include "expert.hpp"
#include "transformer.hpp"

namespace RL {

/*
 * ============================================================
 *  SAC 的稀疏 MoE 骨干配置 —— 与 RL::PPO 那一套同源 (见 rl/ppo.h)
 * ============================================================
 *
 * 2026-09 改版: 骨干从**稠密**的 `MOE<16,16>` 换成稀疏路由的
 * `SparseMoE<TransformerBlock<16>, 4, 1>`。
 *
 *   旧的 MOE<16,16> = 16 个 TransformerBlock<16> 专家, **全部算一遍**再做门控加权和
 *   —— 在 d_model=1260 上一次前向 152 ms (docs/agents_design.md §11.4), 而且稠密
 *   门控没有"路由"这回事, 于是 PPO 那套负载均衡辅助损失根本无从接线。
 *
 *   换掉之后拿到两件事:
 *     * 算力: top-1 只算 1 个专家 (与专家数 E 无关), 16 个专家 -> 1 个;
 *     * 接口: 层是 `ISparseMoE` (rl/sparse_moe.hpp), 于是"批统计复位 +
 *       按批的负载均衡辅助损失"这条 PPO 的优化方法可以直接搬过来 (见本文件的
 *       resetMoeBatchStats / applyGradients)。
 *
 * 结构参数必须编译期确定 (模板参数), 所以放在 namespace 作用域。
 */
constexpr int SAC_MOE_EXPERTS = 4;
constexpr int SAC_MOE_TOPK    = 1;
using SACExpert = TransformerBlock<16>;

/*
 * ============================================================
 *  Discrete SAC —— 把 RL::PPO 的**训练侧优化方法**搬过来 (2026-09)
 * ============================================================
 *
 * 搬过来的五项 (每一条都在下面有对应的成员/函数, 编号与 docs/issues_review.md 一致):
 *
 *  [P3] **梯度累积 + 一次优化器更新**
 *       拆成 `accumulateGrad()` (只前向+反向, 攒梯度与损失) 与 `applyGradients()`
 *       (注入 MoE 辅助损失 -> 优化器 -> 上报批平均损失 -> 清统计)。
 *       拆开的价值不只是"少调几次 RMSProp": 批统计因此有明确的**边界**
 *       (见 [MoE] 那一条)。
 *       注意累积到底改了什么: `RL::Net::RMSProp` 默认 clipGrad=true, 它对**每个
 *       张量**做 `dw /= |dw|` —— 所以"梯度整体乘一个常数"是没有任何效果的。累积
 *       真正改变的是**方向**: 一次更新走的是 batchSize 条样本的梯度和的方向, 而
 *       不是 N 个各自的单位方向串起来走 N 步。这才是 P3 的语义, 也是"优化器调用
 *       次数从 N 次降到 1 次"(实测每样本便宜 2.9×) 之外的另一半。
 *
 *  [P4] **回放池 + 多 epoch 反复复用** (`learnFromReplay(batchSize, epochs, lr)`)
 *       每个 epoch **重新从池里抽** batchSize 条 (有放回, 与 rl/dqn.cpp 同一做法),
 *       累积梯度后只做一次优化器更新。原 `learn()` 里每条样本恰好用一次 —— 而生成
 *       一条样本 (自对弈 + 搜索) 比重放一条贵一两个数量级, 所以同一批"多抽几轮"
 *       是净赚: 一次更新看到的经验从 batchSize 变成 batchSize×epochs。
 *       **不是**"把同一批复用 epochs 遍" —— 批内权重不变, 重复遍的梯度逐位相同,
 *       而 clipGrad 会把这个倍数归一掉, 于是那种写法对更新毫无影响, 只是白烧算力。
 *
 *  [MoE] **批统计复位 + 按批的负载均衡辅助损失**
 *       `resetMoeBatchStats()` 在**每个训练批的开头**调用。为什么必须:
 *       `addAuxGradient` 是按"自上次调用以来所有 forward"的均值算的, 而 forward 有
 *       两个来源 —— 训练时的, 和推理时 (策略前向) 的。不复位的话推理那部分会混进
 *       辅助损失的批统计里 (既有语义混淆, 也有 xSum 这类 float 累加器在几十万次
 *       累加后的精度损失)。RL::SAC 之前的骨干是稠密 MOE, 连这个接口都没有。
 *
 *  [Loss] **批平均损失上报** (`lastLoss` / `lastActorLoss`)
 *       与 RL::PPO / DQN 同一口径: 一个是 critic 的 MSE, 一个是策略目标
 *       `J(π) = Σ_a π_a·(α·log π_a − min_i Q_i(s,a))` 的批平均。逐样本上报会让曲线
 *       变成噪声 (而且重尾样本能差几个数量级)。
 *
 *  [R2] **训练侧只在合法列上算** (`maskedTrainHead` + `Transition::legalMask`)
 *       给了掩码时:
 *         * 前向: 骨干跑到 h 为止, 头**只算合法列** (rl/net.hpp 的 forwardTrunk +
 *           sparseLogits, 即 R1 那条捷径);
 *         * 归一化: π 在**合法集上**归一 (Z ≡ 1), 非法槽位恒为 0;
 *         * 反向: dL/dz 在非法列上**恰好为 0**, 于是非法行的权重梯度恒为 0
 *           (不是"很小", 是精确的 0 —— 因为 dz=0 时 `g.w[a] += 0·h ≡ 0`)。
 *         * 当前局面与下一局面各用**自己的**掩码 (`legalMask` / `nextLegalMask`):
 *           走一步之后合法集就变了, 混用等于把非法着法的价值算进软备份里。
 *       注意 R2 在 PPO 那边还有一条"省算力"的理由 (策略头 64x8100 = 2.07 MB,
 *       每次前向整块读一遍)。在本类里那条**不成立** —— actionDim 只有 128, 整个头
 *       只有 hidden×128×4B ≈ 8 KB。所以这里落地的是 R2 的**学习口径**
 *       (合法集 Z≡1 / 非法列梯度为 0), 不是它的吞吐收益。
 *
 * 另外顺带修的两处**静默**缺陷 (不修的话上面这些"口径"都无从谈起):
 *   1. critic 的输入被拼成 `[state; prob]` (stateDim+actionDim 维), 但 critic 的
 *      第一层是按 stateDim 建的。`MM::ikkj` 的契约是 `x1.shape[1] == x2.shape[0]`,
 *      Release 下断言被 NDEBUG 关掉, 于是它**只读了前 stateDim 个元素** ——
 *      拼进去的策略概率被静默丢掉, Debug 构建里则会直接断言失败。
 *      现在 critic/目标网的第一层按 `stateDim + actionDim` 建, 拼接是有意义的。
 *   2. `learn()` 收下 `learningRate` 却把三个优化器的学习率写死 (1e-2 / 1e-3 /
 *      1e-7), 参数完全没生效。现在参数生效 (actor 用它, critic 保持 1/10 的既有
 *      比例, α 保持 1e-7)。
 *
 * 没搬的: P6 镜像增广 / P7 多线程分身 (都是 agent 层的机制, 本类不知道棋盘)。
 */
class SAC
{
public:
    static constexpr int QNET_NUM = 4;
    /* 稀疏 MoE 骨干结构 (真正的取值在文件顶部 SAC_MOE_*, 这里只是给上层的稳定别名) */
    static constexpr int MOE_EXPERTS = SAC_MOE_EXPERTS;
    static constexpr int MOE_TOPK    = SAC_MOE_TOPK;

public:
    SAC(){}
    explicit SAC(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);

    /* ----------------------------------------------------------------
     *  回放池 (P4)。元素就是 rl_basic.h 的 Transition —— 它的 `legalMask`
     *  是 R2 的可选合法动作掩码 (空 = 旧的全量口径)。
     * ---------------------------------------------------------------- */
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    /* R2: 带合法动作掩码 (actionDim 维, 1 = 合法)。这一版把**同一套**掩码同时用于
       state 与 nextState —— 只有在"动作集与局面无关"时才正确 (象棋里走一步合法集
       就变了, 那种情形要用下面 7 参数的重载)。 */
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done,
                  const Tensor& legalMask);
    /* R2: 当前局面与下一局面**各自**的合法掩码 (可以只给其中一个, 空 = 该局面全量) */
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done,
                  const Tensor& legalMask,
                  const Tensor& nextLegalMask);

    Tensor& eGreedyAction(const Tensor& state);
    Tensor& gumbelMax(const Tensor &state);
    /* 策略 π(·|s) = 整向量 softmax (头是 Linear + 这里自己做 softmax,
       见文件顶部: 掩码归一化必须自己掌握, 不能对 Softmax 层的输出打补丁) */
    Tensor& action(const Tensor &state);

    /* 掩码策略: mask 为空 (或 maskedTrainHead=false) 时退化成整向量 softmax */
    void policy(const Tensor &state, const Tensor &legalMask, Tensor &pi);

    /* ----------------------------------------------------------------
     *  [P3] 只累积梯度 (前向 + 反向), 不碰优化器
     * ---------------------------------------------------------------- */
    void accumulateGrad(const Transition &x);
    /*
        旧名 (`experienceReplay`) 保留: 它原来就是"一条样本 -> 前向+反向"这一步,
        语义没变, 只是现在不再顺手写 loss 到别处、也不再单独调优化器。
    */
    void experienceReplay(const Transition& x) { accumulateGrad(x); }
    /* 注入 MoE 辅助损失 -> 优化器 -> 上报批平均损失 -> 清批统计 */
    void applyGradients(float learningRate);
    /* [MoE] 只清"本批"的门控统计, 保留生命周期累计 (坍缩诊断要用) */
    void resetMoeBatchStats();

    /* ----------------------------------------------------------------
     *  训练入口
     * ---------------------------------------------------------------- */
    /*
        旧入口 (保留, 语义 = learnFromReplay(batchSize, 1, learningRate))。
        另外两个形参现在写进成员 `maxMemorySize` / `replaceTargetIter`。
    */
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
    /*
        [P4] 从池里采样 batchSize 条 (有放回), **过 epochs 遍**, 累积梯度后做一次
        优化器更新。返回 false 表示池子不够大或 epochs<=0 (什么都没做)。
    */
    bool learnFromReplay(std::size_t batchSize,
                         int epochs,
                         float learningRate,
                         std::size_t maxMemorySize = 4096,
                         std::size_t replaceTargetIter = 256);

    void save();
    void load();

    /* ----------------------------------------------------------------
     *  稀疏 MoE 诊断 (只读, 不参与任何计算; 给测试与调参用)
     * ---------------------------------------------------------------- */
    int moeLayerCount() const;
    int moeExpertCount() const;
    int moeTopK() const;
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();
    long long actorParamCount() const { return actor.paramCount(); }
    long long criticParamCount() const { return critics[0].paramCount(); }

    std::size_t replaySize() const { return memories.size(); }
    void clearReplay() { memories.clear(); }
    /* 已经做过多少次优化器更新 (P3/P4 的机制断言要用它) */
    int getLearningSteps() const { return learningSteps; }
    /* 当前温度 α[0] (SAC 的自动调节量, 诊断/断言用) */
    float getAlpha() const { return alpha[0]; }
    /*
        取走整个回放池并把自己清空 (deque 的 swap 是 O(1))。与 RL::PPO 同名同义 ——
        刻意不在 SAC 里加锁: 让这个类保持单线程, 由调用方决定怎么并发。
    */
    std::deque<Transition> takeReplay()
    {
        std::deque<Transition> out;
        out.swap(memories);
        return out;
    }

    /*
        [R2] 总开关 (默认开)。关掉就退回"整向量 softmax + 全量目标"的旧口径 ——
        留给 A/B 用。
    */
    bool maskedTrainHead = true;

    /* 回放池上限 (条) */
    std::size_t maxMemorySize = 4096;
    /* 每多少次 applyGradients 做一次 Polyak 同步 */
    std::size_t replaceTargetIter = 256;

    /*
        最近一次 applyGradients 的标量损失 (界面曲线口径, 与 RL::PPO 同名同义,
        **不参与任何计算**):
          lastLoss      : critic 的 MSE (批平均)
          lastActorLoss : 策略目标 J(π) = Σ_a π_a(α·log π_a − min_i Q_i) (批平均)
                          注意它不是"概率型损失", 可以取负 —— 与 PPO 的交叉熵不同。
    */
    double lastLoss = std::numeric_limits<double>::quiet_NaN();
    double lastActorLoss = std::numeric_limits<double>::quiet_NaN();

    /*
       最近一次 applyGradients **攒了多少条样本** (P4 的诊断):
       learnFromReplay(batchSize, epochs) 应当给出 batchSize×epochs —— 这条数字把
       "多 epoch 是每遍都抽新样本"这件事变成可断言的东西 (见 test/test_sac_main.cpp)。
    */
    std::size_t lastBatchSamples = 0;

    /* 批内累加 (accumulateGrad 攒, applyGradients 取平均后写进上面两个) */
    double batchLossSum = 0.0;
    double batchActorLossSum = 0.0;
    double batchAlphaGradSum = 0.0;
    std::size_t batchLossCount = 0;

    /*
        负载均衡辅助损失系数 (<=0 关闭)。默认与 RL::PPO / SACAZAgent 一致 (0.1)。
        以前这个类用稠密 MOE, 没有路由也就没有这一项。
    */
    float moeAuxCoef = 0.1f;

    /* 三个优化器的学习率。构造时取原来的常量 (1e-2 / 1e-3 / 1e-7);
       `learningRate` 形参现在真的会写进 learningRateActor (见文件顶部第 2 条)。 */
    float learningRateActor  = 1e-2f;
    float learningRateCritic = 1e-3f;
    float learningRateAlpha  = 1e-7f;

    std::deque<Transition> memories;

    /*
       网络是 public 的 (与 RL::PPO 的 actorP/critic 一样), 这样测试与诊断可以直接
       看头部的权重/梯度 —— "非法列的梯度恰好为 0" 这条断言就是这么钉的
       (见 test/test_sac_main.cpp)。
    */
    Net actor;      /* 策略:    state -> 骨干 -> actionDim **logits** */
    Net critics[QNET_NUM];
    Net criticsTarget[QNET_NUM];

protected:
    int stateDim;
    int actionDim;
    int hiddenDim;
    float gamma;
    float H0;
    float exploringRate;
    int learningSteps;
    ExpAnnealing annealing;
    GradValue alpha;
    /* `action()` 的返回值 (π 的副本): actor 的输出缓冲是 logits, 而 π 是
       掩码/整向量 softmax 的结果, 两者不是同一个张量。 */
    Tensor m_pi;

private:
    /*
        critic 的输入 = [state; π]，所以第一层的输入维是 stateDim + actionDim。
        (原来按 stateDim 建, 拼接的 π 段被 MM 的契约静默截掉 —— 见文件顶部。)
    */
    int criticInputDim() const { return stateDim + actionDim; }
    /*
        按 "inDim -> 稀疏MoE -> Tanh(inDim -> h) -> Linear(h -> actionDim)" 造一个
        网络 (专家与门控的初始化缩放在 SparseMoE 的构造函数里做)。actor 用
        inDim=stateDim (掩码 softmax 在类内自己做, 所以头输出 **logits**)，
        critic / 目标网用 inDim=stateDim+actionDim。
    */
    Net buildNet(int inDim, bool withGrad) const;
    /* 按池上限裁掉最老的样本 */
    void trimMemory();
};

} // namespace RL
#endif // SAC_H
