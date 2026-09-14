#include "ppo.h"
#include "layer.h"
#include "loss.h"
#include "sparse_moe.hpp"

namespace {

/*
   初始化缩放 (与 sacazagent.cpp 的 scaleLayerInit 同一套依据, 数值见 ppo.h)。

   `iFcLayer` 的构造函数把权重初始化成 U(-1,1); d 维输入下 pre-activation 的标准差
   约 sqrt(d/3)。局面的平面编码里每行只有十几个 1、其余全是 0, 所以实际方差比这小,
   但 d=1440 时仍有 ~20 的量级 —— Tanh 直接饱和, 梯度接近 0, 网络看起来"能跑但不学"。
   这里按 1/sqrt(fan_in) 再缩放一遍, 把 pre-activation 的标准差拉回 ~0.6。

   只缩普通的 iFcLayer: 稀疏 MoE 层不是 iFcLayer (dynamic_cast 返回 nullptr), 它的
   专家权重已经在 SparseMoE 构造函数里用 scaleExpertInit 缩过了 —— 若在这里再缩一次,
   同一个层的初始化标准差会被压两遍。
*/
void scaleLayerInit(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer*>(net[i]);
        if (fc == nullptr) {
            continue;
        }
        const float fanIn = (float)(fc->inputDim > 1 ? fc->inputDim : 1);
        const float s = 1.0f / std::sqrt(fanIn);
        for (std::size_t k = 0; k < fc->w.size(); k++) {
            fc->w[k] *= s;
        }
        for (std::size_t k = 0; k < fc->b.size(); k++) {
            fc->b[k] *= s;
        }
    }
}

/* 网络里所有的稀疏 MoE 层 */
std::vector<RL::ISparseMoE*> sparseMoeLayers(RL::Net &net)
{
    std::vector<RL::ISparseMoE*> out;
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(net[i]);
        if (m != nullptr) {
            out.push_back(m);
        }
    }
    return out;
}

} // namespace

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_,
             int expertHidden_, float moeAuxCoef_, bool withGrad)
    :stateDim(stateDim_), actionDim(actionDim_),
     expertHidden(expertHidden_ > 0 ? expertHidden_ : 64),
     gamma(0.99f), exploringRate(1.0f),
     moeAuxCoef(moeAuxCoef_), learningSteps(0)
{
    /*
       Actor: state -> SparseMoE(E=8, top-2) -> Tanh(hidden) -> Softmax(actionDim)

       稀疏 MoE 层是"同维进出"的 (专家的输入输出必须同维才能做门控加权和), 所以后面
       必须再接一层普通层把 stateDim 压到 hiddenDim, 再进输出头。

       withGrad=false 时每个 iFcLayer **不分配** g/v/m 三份梯度缓冲 —— 参数量不变,
       但内存和构造时间都降到约 1/4。多线程分身训练里 worker 只做搜索、不做反向,
       所以它的网络应该是这个形态 (N 个 worker 省下的内存很可观)。
    */
    Net::Layers actorLayers;
    actorLayers.push_back(std::make_shared<SparseMoE<MlpExpert,
                                                     MOE_EXPERTS,
                                                     MOE_TOPK> >(
        stateDim, withGrad, expertHidden));
    actorLayers.push_back(Layer<Tanh>::_(stateDim, hiddenDim, true, withGrad));
    actorLayers.push_back(Layer<Softmax>::_(hiddenDim, actionDim, true, withGrad));
    actorP = Net(actorLayers);

    /* Critic: 同样的骨干 + Linear(1) 标量价值头 */
    Net::Layers criticLayers;
    criticLayers.push_back(std::make_shared<SparseMoE<MlpExpert,
                                                      MOE_EXPERTS,
                                                      MOE_TOPK> >(
        stateDim, withGrad, expertHidden));
    criticLayers.push_back(Layer<Tanh>::_(stateDim, hiddenDim, true, withGrad));
    criticLayers.push_back(Layer<Linear>::_(hiddenDim, 1, true, withGrad));
    critic = Net(criticLayers);

    scaleLayerInit(actorP);
    scaleLayerInit(critic);
}

RL::Tensor &RL::PPO::action(const Tensor &state)
{
    return actorP.forward(state);
}

float RL::PPO::value(const Tensor &state)
{
    RL::Tensor &v = critic.forward(state);
    return v[0];
}

void RL::PPO::resetMoeBatchStats()
{
    if (moeAuxCoef <= 0.0f) {
        return;
    }
    std::vector<ISparseMoE*> actorMoes = sparseMoeLayers(actorP);
    for (std::size_t i = 0; i < actorMoes.size(); i++) {
        actorMoes[i]->resetBatchStats();
    }
    std::vector<ISparseMoE*> criticMoes = sparseMoeLayers(critic);
    for (std::size_t i = 0; i < criticMoes.size(); i++) {
        criticMoes[i]->resetBatchStats();
    }
}

void RL::PPO::accumulateGrad(const Tensor &state,
                             const Tensor &actionTarget,
                             float valueTarget)
{
    /*
       ============================================================
        顺序很重要: 标量损失必须在 backward() **之前**从网络输出里读出来。
       ============================================================
       `Layer::backward` 结束时会把自己的 o/e 清零 (见 layer.h / MlpExpert::backward),
       而 policy / v 是**网络最后一层的输出张量引用** —— backward 之后再读只会读到 0:

         * v[0] 恒为 0        -> lastLoss 恒等于 valueTarget²
         * policy[i] 恒为 0   -> lastActorLoss 恒等于 -ln(1e-8)·Σtarget = 18.4207

       也就是说改之前"界面上那条训练损失曲线"是个与训练状态无关的常数 ——
       实测: 17 轮学习中 lastLoss 一直 0.5625 (= 0.75²)、lastActorLoss 一直 18.4207,
       而同一时刻策略已经从 0.000142 涨到 0.5139、V 从 -0.04 走到 0.90。
       这类错误不会报异常, 只会让唯一的可观测指标失真。
    */

    /* ---- Actor: cross-entropy loss ---- */
    Tensor &policy = actorP.forward(state);
    double ce = 0.0;
    /*
       注意上界是 actionTarget.size(): 动作空间 8100 维而目标通常稀疏 (访问分布只落在
       少数合法走法上), 但交叉熵仍要对整个分布求和 —— softmax 的归一化分母把概率质量
       分给了**全部** 8100 个槽位。
    */
    for (std::size_t i = 0; i < policy.size() && i < actionTarget.size(); i++) {
        if (actionTarget[i] > 0.0f) {
            ce -= (double)actionTarget[i] * std::log((double)policy[i] + 1e-8);
        }
    }
    Tensor ceLoss = Loss::CrossEntropy::df(policy, actionTarget);
    actorP.backward(state, ceLoss);

    /* ---- Critic: MSE loss ---- */
    Tensor &v = critic.forward(state);
    const double err = (double)v[0] - (double)valueTarget;   /* backward 之前读 */
    Tensor valueTargetTensor(1, 1);
    valueTargetTensor[0] = valueTarget;
    Tensor mseLoss = Loss::MSE::df(v, valueTargetTensor);
    critic.backward(state, mseLoss);

    /*
       损失攒起来由 applyGradients 取**批平均** —— 一批里逐样本上报的话, 界面的
       "训练损失曲线"会变成逐样本抖动的噪声, 而且与 DQN 报"平均平方 TD 误差"的口径
       也对不上。
    */
    batchLossSum += err * err;
    batchActorLossSum += ce;
    batchLossCount++;
}

void RL::PPO::applyGradients(float lr)
{
    /*
       ---- 稀疏 MoE 的负载均衡辅助损失 ----
       必须在优化器之前、主反向之后: 本批每个层被选中的专家次数与门控概率之和都已经
       记好了, addAuxGradient 用它们算出"哪些专家被喂爆了", 把这些专家的 logit 压下去、
       把饿着的抬起来 (Switch Transformer 的 L_aux = E·Σ f_i·P_i 对 logits 的梯度)。
       缺了它路由会在几轮更新内坍缩 —— 实测见 test_ppomcts 的 MoE 诊断节。

       批大小: 累积 N 条之后这里注入的是一份**真正的批**估计 (均值场近似, 见
       sparse_moe.hpp 的说明), 比原来"一条样本注入一次"方差小得多。
    */
    if (moeAuxCoef > 0.0f) {
        std::vector<ISparseMoE*> actorMoes = sparseMoeLayers(actorP);
        for (std::size_t i = 0; i < actorMoes.size(); i++) {
            actorMoes[i]->addAuxGradient(moeAuxCoef);
        }
        std::vector<ISparseMoE*> criticMoes = sparseMoeLayers(critic);
        for (std::size_t i = 0; i < criticMoes.size(); i++) {
            criticMoes[i]->addAuxGradient(moeAuxCoef);
        }
    }

    actorP.RMSProp(lr, 0.9f, 0.001f);
    critic.RMSProp(lr, 0.9f, 0.001f);
    learningSteps++;

    if (batchLossCount > 0) {
        lastLoss = batchLossSum / (double)batchLossCount;
        lastActorLoss = batchActorLossSum / (double)batchLossCount;
    }
    batchLossSum = 0.0;
    batchActorLossSum = 0.0;
    batchLossCount = 0;
}

void RL::PPO::trainStep(const Tensor &state,
                        const Tensor &actionTarget,
                        float valueTarget,
                        float lr)
{
    /* 单样本路径 = "清批统计 -> 累积 1 条 -> 应用"。批路径见 learnFromReplay()。 */
    resetMoeBatchStats();
    accumulateGrad(state, actionTarget, valueTarget);
    applyGradients(lr);
}

void RL::PPO::addReplay(const Tensor &state,
                        const std::vector<int> &actionIdx,
                        const std::vector<float> &actionProb,
                        float valueTarget)
{
    if (actionIdx.empty()) {
        return;
    }
    ReplaySample s;
    s.state = state;              /* Tensor 赋值 = 深拷贝 (调用方的 state 每步会被重写) */
    s.actionIdx = actionIdx;
    s.actionProb = actionProb;
    s.valueTarget = valueTarget;
    replay.push_back(std::move(s));
    /* FIFO: 满了丢最老的一条 (deque 的 pop_front 是 O(1)) */
    while (replay.size() > replayCapacity) {
        replay.pop_front();
    }
}

bool RL::PPO::learnFromReplay(std::size_t batchSize, int epochs, float lr)
{
    if (epochs <= 0 || batchSize == 0 || replay.size() < batchSize) {
        return false;
    }

    Tensor state((std::size_t)stateDim, 1);
    Tensor target((std::size_t)actionDim, 1);
    /* 有放回采样, 与 rl/dqn.cpp 的 learn() 同一做法 */
    std::uniform_int_distribution<std::size_t> pick(0, replay.size() - 1);

    resetMoeBatchStats();
    for (int e = 0; e < epochs; e++) {
        for (std::size_t b = 0; b < batchSize; b++) {
            const ReplaySample &s = replay[pick(Random::engine)];
            state = s.state;
            /* 稀疏目标 -> 稠密: 只填非零项, 其余清零 */
            target.zero();
            for (std::size_t k = 0; k < s.actionIdx.size(); k++) {
                const int a = s.actionIdx[k];
                if (a >= 0 && a < actionDim) {
                    target[(std::size_t)a] = s.actionProb[k];
                }
            }
            accumulateGrad(state, target, s.valueTarget);
        }
    }
    applyGradients(lr);
    return true;
}

void RL::PPO::learnSelfPlay(std::vector<Step>& trajectory,
                            float finalOutcome,
                            float learningRate)
{
    int end = (int)trajectory.size() - 1;
    if (end < 0) return;

    /*
       ---- 折现回报必须**逐手翻转视角** ----

       设 V(s_i) 是"第 i 步走子方"视角的价值 (状态编码是规范视角, 所以价值头输出的
       就是走子方的价值)。相邻两步的走子方互为对手, 于是:

           V(s_i) = reward_i + gamma * ( -V(s_{i+1}) )  =  reward_i - gamma * V(s_{i+1})
                     ^^^^^^^^          ^^^^^^^^^^^^^^
                     走子方视角         下一步是**对手**视角, 取负才换回自己这边

       原来的实现写的是 `r = reward + gamma * r` (加号), 等于把对手的价值当成自己的,
       相邻两手之间就错一次符号; 再叠上调用方传进来的"黑方视角终局值", 红方走的
       那些步拿到的目标整体是反的 (红方赢的棋, 对红方反而成了"在输")。这类错误不会
       报错, 只会让 critic 退化成材质计数器, 而 MCTS 的叶子估值正是靠它。

       初始值: 进来时 r 代表 V(s_{end+1}), 也就是"最后一步走完之后该走棋的那一方"
       (= m_end 的对手) 的价值; 而 finalOutcome 是从 m_end 看的, 所以取负。
    */
    const std::vector<float> returns = discountedReturns(trajectory, finalOutcome);

    /* Train each step: policy target = action taken (one-hot), value target = return */
    for (int t = 0; t <= end; t++) {
        trainStep(trajectory[t].state,
                  trajectory[t].action,
                  returns[(std::size_t)t],
                  learningRate);
    }

    exploringRate *= 0.99999f;
    if (exploringRate < 0.01f) exploringRate = 0.01f;
}

std::vector<float> RL::PPO::discountedReturns(const std::vector<Step> &trajectory,
                                              float finalOutcome) const
{
    std::vector<float> rewards(trajectory.size(), 0.0f);
    for (std::size_t i = 0; i < trajectory.size(); i++) {
        rewards[i] = trajectory[i].reward;
    }
    return discountedReturnsFromRewards(rewards, finalOutcome);
}

std::vector<float> RL::PPO::discountedReturnsFromRewards(const std::vector<float> &rewards,
                                                         float finalOutcome) const
{
    std::vector<float> returns(rewards.size(), 0.0f);
    if (rewards.empty()) {
        return returns;
    }
    /* 推导见上面 learnSelfPlay 的注释 (视角逐手翻转 + 终局取负) */
    float r = -finalOutcome;
    for (int i = (int)rewards.size() - 1; i >= 0; i--) {
        r = rewards[(std::size_t)i] - gamma * r;
        returns[(std::size_t)i] = r;
    }
    return returns;
}

void RL::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    actorP.save(actorPara);
    critic.save(criticPara);
}

void RL::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    actorP.load(actorPara);
    critic.load(criticPara);
}

/* ============================================================
 *  稀疏 MoE 诊断
 * ============================================================ */
namespace {

/*
   Net::operator[] 没有 const 重载 (SACAZAgent 里也是 const_cast 绕过去的),
   这几个诊断函数逻辑上只读, 所以统一在这里做一次转换。
*/
std::vector<RL::ISparseMoE*> moeLayersOf(const RL::Net &net)
{
    RL::Net &self = const_cast<RL::Net&>(net);
    return sparseMoeLayers(self);
}

} // namespace

int RL::PPO::moeLayerCount() const
{
    return (int)(moeLayersOf(actorP).size() + moeLayersOf(critic).size());
}

int RL::PPO::moeExpertCount() const
{
    std::vector<ISparseMoE*> layers = moeLayersOf(actorP);
    return layers.empty() ? 0 : layers[0]->expertCount();
}

int RL::PPO::moeTopK() const
{
    std::vector<ISparseMoE*> layers = moeLayersOf(actorP);
    return layers.empty() ? 0 : layers[0]->topK();
}

void RL::PPO::moeUsage(std::vector<long long> &out) const
{
    out.clear();
    const int experts = moeExpertCount();
    if (experts <= 0) {
        return;
    }
    out.assign((std::size_t)experts, 0);

    std::vector<ISparseMoE*> layers = moeLayersOf(actorP);
    std::vector<ISparseMoE*> criticLayers = moeLayersOf(critic);
    layers.insert(layers.end(), criticLayers.begin(), criticLayers.end());

    std::vector<long long> one;
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->usageSnapshot(one);
        for (std::size_t e = 0; e < one.size() && e < out.size(); e++) {
            out[e] += one[e];
        }
    }
}

void RL::PPO::resetMoeUsage()
{
    std::vector<ISparseMoE*> layers = moeLayersOf(actorP);
    std::vector<ISparseMoE*> criticLayers = moeLayersOf(critic);
    layers.insert(layers.end(), criticLayers.begin(), criticLayers.end());
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->resetUsage();
    }
}
