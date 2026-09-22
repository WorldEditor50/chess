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

/*
   按骨干造那一层稀疏 MoE (actor 与 critic 各一份)。

   为什么必须是这样一个工厂而不是"构造 PPO 时选一次类型": 专家类型与 (E, top-k) 都是
   **模板参数**, 而 PPO 的其余部分 (损失/优化器/存盘/诊断) 完全不需要知道专家是什么 ——
   它只通过 `ISparseMoE` 接口用它 (负载均衡辅助损失、专家使用直方图、读写权重都由虚函数
   分派)。所以"两种骨干"的差别**只在这一个函数里**, 复制整个 ppo.cpp 才是真正的风险。

   注意 withGrad 要透传: 多线程分身训练的 worker 用 withGrad=false (不分配梯度缓冲,
   内存与构造时间约 1/4), 而这条路径必须在两种骨干下都能用。
*/
std::shared_ptr<RL::iLayer> makeMoeLayer(int stateDim, bool withGrad, int expertHidden,
                                         RL::PPO::Backbone backbone)
{
    if (backbone == RL::PPO::Backbone::MlpExperts) {
        return std::make_shared<RL::SparseMoE<RL::MlpExpert,
                                              RL::PPO_MOE_MLP_EXPERTS,
                                              RL::PPO_MOE_MLP_TOPK> >(
            stateDim, withGrad, expertHidden);
    }
    return std::make_shared<RL::SparseMoE<RL::PPOExpert,
                                          RL::PPO_MOE_EXPERTS,
                                          RL::PPO_MOE_TOPK> >(
        stateDim, withGrad, expertHidden);
}

} // namespace

const char *RL::PPO::backboneName(Backbone b)
{
    switch (b) {
    case Backbone::TbExperts:  return "稀疏MoE(TB专家)";
    case Backbone::MlpExperts: return "稀疏MoE(MLP专家)";
    }
    return "?";
}

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_,
             int expertHidden_, float moeAuxCoef_, bool withGrad,
             Backbone backbone_)
    :stateDim(stateDim_), actionDim(actionDim_),
     expertHidden(expertHidden_ > 0 ? expertHidden_ : 64),
     backbone(backbone_),
     gamma(0.99f), exploringRate(1.0f),
     moeAuxCoef(moeAuxCoef_), learningSteps(0)
{
    /*
       Actor: state -> SparseMoE(E, top-k, 专家 = PPOExpert) -> Tanh(hidden) -> Softmax(actionDim)

       稀疏 MoE 层是"同维进出"的 (专家的输入输出必须同维才能做门控加权和), 所以后面
       必须再接一层普通层把 stateDim 压到 hiddenDim, 再进输出头。

       2026-09 第二次改版: 专家 = TransformerBlock<16,360> (原来是 MlpExpert) ——
       容量 2.15 M -> 38.0 M 参数, 代价是前向 0.139 -> 3.59 ms/次; 结构参数与
       理由 (含"为什么专家数同时从 8 降到 4") 见 ppo.h 顶部那一段。

       2026-09 第三次 (本工程扩展): 那条 MlpExpert 骨干**没有被删掉**, 而是变成
       `Backbone::MlpExperts` 这个可选项 (由界面上的 AGENT_PPOMCTS_MLP 使用) ——
       两种骨干共用这一份 PPO 实现 (搜索/训练/存盘/诊断全一样), 于是可以直接对弈
       比较。**默认值仍是 TbExperts**, 所以既有调用方的行为逐位不变。
       两种专家**结构不同 ⇒ paramCount 不同**, 而权重文件带"元素总数 == paramCount"
       的结构指纹, 所以交叉载入会当场失败, 不会静默串权重。

       expertHidden 只对 MlpExpert 有意义 (ExpertFactory 对 TransformerBlock 会忽略它),
       签名保留是为了不动一圈调用方 (PPOMCTSAgent / 测试 / bench)。

       withGrad=false 时每个 iFcLayer **不分配** g/v/m 三份梯度缓冲 —— 参数量不变,
       但内存和构造时间都降到约 1/4。多线程分身训练里 worker 只做搜索、不做反向,
       所以它的网络应该是这个形态 (N 个 worker 省下的内存很可观: TB 专家下
       worker 从 ~600 MB 降到 ~150 MB)。
    */
    Net::Layers actorLayers;
    actorLayers.push_back(makeMoeLayer(stateDim, withGrad, expertHidden, backbone));
    actorLayers.push_back(Layer<Tanh>::_(stateDim, hiddenDim, true, withGrad));
    actorLayers.push_back(Layer<Softmax>::_(hiddenDim, actionDim, true, withGrad));
    actorP = Net(actorLayers);

    /* Critic: 同样的骨干 + Linear(1) 标量价值头 */
    Net::Layers criticLayers;
    criticLayers.push_back(makeMoeLayer(stateDim, withGrad, expertHidden, backbone));
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

/* ------------------------------------------------------------------
 *  actionMasked (R1): 只算 idx 里那些动作的策略概率
 *
 *  等价性: 全量 softmax 的输出在子集上重新归一化之后 == 在子集上直接 softmax
 *  (公共因子 exp(-m) 被归一化约掉)。所以这条路径**只改速度不改语义** ——
 *  这是"R1 不需要重训"的全部依据。
 *
 *  回退: 头不支持稀疏 (非 softmax 头 / 非全连接) 或 sparseLogits 拒绝了这次的形状
 *  时, 老老实实跑一次全量前向再取子集 —— 数值一致, 只是慢。
 * ------------------------------------------------------------------ */
bool RL::PPO::actionMasked(const Tensor &state,
                           const std::vector<int> &idx,
                           std::vector<float> &probs)
{
    probs.clear();
    if (idx.empty()) {
        return false;
    }

    /* ---- 回退路径: 全量前向 + 子集归一 ---- */
    auto denseFallback = [&]() {
        Tensor &full = actorP.forward(state);
        probs.assign(idx.size(), 0.0f);
        double sum = 0.0;
        for (std::size_t i = 0; i < idx.size(); i++) {
            const int a = idx[i];
            const float p = (a >= 0 && a < actionDim) ? full[(std::size_t)a] : 0.0f;
            probs[i] = p;
            sum += (double)p;
        }
        if (sum > 1e-12) {
            const float inv = (float)(1.0 / sum);
            for (std::size_t i = 0; i < probs.size(); i++) {
                probs[i] *= inv;
            }
        } else {
            /* 全都被压到 0 的退化情形: 给合法集上的均匀分布, 而不是全 0 */
            const float u = 1.0f / (float)probs.size();
            for (std::size_t i = 0; i < probs.size(); i++) {
                probs[i] = u;
            }
        }
        return true;
    };

    if (!actorP.sparseOutputSupported()) {
        return denseFallback();
    }

    /* ---- 稀疏路径: 骨干只跑到倒数第二层, 头只算 idx 那几行 ---- */
    Tensor &h = actorP.forwardTrunk(state);
    std::vector<float> logits;
    if (!actorP.sparseLogits(h, idx, logits) || logits.size() != idx.size()) {
        return denseFallback();
    }

    /* 在 idx 上做数值稳定的 softmax (减最大值, 与 Softmax::f 同一口径) */
    float m = logits[0];
    for (std::size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > m) {
            m = logits[i];
        }
    }
    double sum = 0.0;
    probs.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); i++) {
        const float e = std::exp(logits[i] - m);
        probs[i] = e;
        sum += (double)e;
    }
    if (!(sum > 1e-12) || !std::isfinite(sum)) {
        /* 极端情形 (logits 全 NaN/inf): 退化成合法集上的均匀分布 */
        const float u = 1.0f / (float)probs.size();
        for (std::size_t i = 0; i < probs.size(); i++) {
            probs[i] = u;
        }
        return true;
    }
    const float inv = (float)(1.0 / sum);
    for (std::size_t i = 0; i < probs.size(); i++) {
        probs[i] *= inv;
    }
    return true;
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

    /* ---- Critic: MSE loss (目标按 clampValue 夹住, 见 ppo.h 的说明) ---- */
    const double vt = (clampValue > 0.0f)
                          ? (double)std::min(std::max(valueTarget, -clampValue), clampValue)
                          : (double)valueTarget;
    Tensor &v = critic.forward(state);
    const double err = (double)v[0] - vt;   /* backward 之前读 */
    Tensor valueTargetTensor(1, 1);
    valueTargetTensor[0] = (float)vt;
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
                        float valueTarget,
                        const std::vector<int> &legalIdx,
                        const std::vector<float> &oldProb)
{
    if (actionIdx.empty()) {
        return;
    }
    ReplaySample s;
    s.state = state;              /* Tensor 赋值 = 深拷贝 (调用方的 state 每步会被重写) */
    s.actionIdx = actionIdx;
    s.actionProb = actionProb;
    s.legalIdx = legalIdx;
    /*
       信任域样本: 只在"长度与 legalIdx 一致"时才收 —— 长度不符说明调用方给错了东西,
       宁可不裁剪 (退化成纯交叉熵) 也不能拿错位的分母算 ratio (那会把梯度方向弄反,
       而且不会报错)。见 ReplaySample::oldProb 的说明。
    */
    if (!oldProb.empty() && oldProb.size() == legalIdx.size()) {
        s.oldProb = oldProb;
    }
    s.valueTarget = valueTarget;
    replay.push_back(std::move(s));
    /* FIFO: 满了丢最老的一条 (deque 的 pop_front 是 O(1)) */
    while (replay.size() > replayCapacity) {
        replay.pop_front();
    }
}

/* ------------------------------------------------------------------
 *  accumulateGradSparse (R2): 训练侧只算合法列
 *
 *  与 accumulateGrad 的差别只有 Actor 那一路 (critic 一字未改):
 *    前向: forwardTrunk -> 头只算 legalIdx 那几行的 logit -> **在合法集上** softmax
 *    损失: CE = -Σ_{合法} t_a·ln p_a
 *    梯度: softmax + CE 的解析梯度就是 `dL/dlogit_a = p_a - t_a`
 *    反向: 头的权重梯度只落在合法行; 往下传的梯度也只由合法行构成 (ei = W_legalᵀ·d),
 *          之后交给 Net::backwardFrom 走骨干 (头那一层不再走通用 backward)
 *
 *  这样每条样本省掉的是: 8100 行前向 + 8100 行反向 + 一次 8100 维 softmax/Jacobian,
 *  换成 ~40 行。语义上换掉了归一化口径 (Z ≡ 1)。
 * ------------------------------------------------------------------ */
void RL::PPO::accumulateGradSparse(const Tensor &state,
                                   const std::vector<int> &legalIdx,
                                   const std::vector<int> &targetIdx,
                                   const std::vector<float> &targetProb,
                                   float valueTarget,
                                   const std::vector<float> &oldProb)
{
    const std::size_t headIndex = actorP.size() - 1;
    RL::iFcLayer *head = (actorP.size() >= 2)
                             ? dynamic_cast<RL::iFcLayer *>(actorP[headIndex])
                             : nullptr;

    /* ---- 回退: 缺任何前提就回到全量口径 (慢, 但语义是旧的, 不会算错) ---- */
    if (head == nullptr || legalIdx.empty() || !actorP.sparseOutputSupported() ||
        head->g.w.size() < head->w.size()) {
        Tensor dense((std::size_t)actionDim, 1);
        dense.zero();
        for (std::size_t k = 0; k < targetIdx.size() && k < targetProb.size(); k++) {
            const int a = targetIdx[k];
            if (a >= 0 && a < actionDim) {
                dense[(std::size_t)a] = targetProb[k];
            }
        }
        accumulateGrad(state, dense, valueTarget);
        return;
    }

    /* ---- 1) 前向: 骨干到 h, 头只算合法列 ---- */
    Tensor &h = actorP.forwardTrunk(state);
    std::vector<float> logits;
    if (!actorP.sparseLogits(h, legalIdx, logits) || logits.size() != legalIdx.size()) {
        Tensor dense((std::size_t)actionDim, 1);
        dense.zero();
        for (std::size_t k = 0; k < targetIdx.size() && k < targetProb.size(); k++) {
            const int a = targetIdx[k];
            if (a >= 0 && a < actionDim) {
                dense[(std::size_t)a] = targetProb[k];
            }
        }
        accumulateGrad(state, dense, valueTarget);
        return;
    }

    const std::size_t n = logits.size();
    /* 合法集上的数值稳定 softmax (与 R1 的推理路径同一口径) */
    float m = logits[0];
    for (std::size_t i = 1; i < n; i++) {
        if (logits[i] > m) { m = logits[i]; }
    }
    std::vector<float> probs(n, 0.0f);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const float e = std::exp(logits[i] - m);
        probs[i] = e;
        sum += (double)e;
    }
    const bool degenerate = !(sum > 1e-12) || !std::isfinite(sum);
    if (!degenerate) {
        const float inv = (float)(1.0 / sum);
        for (std::size_t i = 0; i < n; i++) { probs[i] *= inv; }
    }

    /* ---- 2) 目标对齐到合法集的顺序 (线性查找: n ~ 40, 可忽略) ---- */
    std::vector<float> tgt(n, 0.0f);
    for (std::size_t k = 0; k < targetIdx.size() && k < targetProb.size(); k++) {
        const int a = targetIdx[k];
        for (std::size_t i = 0; i < n; i++) {
            if (legalIdx[i] == a) { tgt[i] += targetProb[k]; break; }
        }
    }

    /*
       ---- 3) 策略项: 纯交叉熵 (旧口径) 或 PPO 裁剪代理目标 (给了 oldProb 时) ----

       这里**直接算 dL/dlogit**, 不走"先算 dL/dπ 再乘 softmax 雅可比"那条路 ——
       2026-09 踩过一次坑, 记下来免得再犯:
         * 交叉熵对 logit 的梯度就是 **(p − t)** (这是 z 的函数, 不是常数);
         * 而"先写 dL/dπ 再套雅可比"要求写的是 **dL/dπ = −t/p**, 不是 −t。
           我第一版把 `g_i = −t_i` 当成 dL/dπ 套进雅可比, 结果梯度被整体缩小
           `p_i` 倍 (正交性: 雅可比把常数向量投影掉了), test_grad 的 E 节用中心差分
           当场抓到 (解析/差分 = 0.129 的常数倍)。
       现在两种目标都直接给 logit 梯度, 形式统一:

         CE 目标      : L = −Σ_a t_a·log p_a          ->  dL/dz_i = p_i − t_i
         裁剪代理目标 : L = −Σ_a min(ρ_a·A_a, clip(ρ_a,1±ε)·A_a),  A_a = t_a − p_old,a
                        对**单样本**策略梯度, ∂ρ_a/∂z_i = ρ_a(δ_ai − p_i), 于是
                        dL/dz_i = −[未裁剪项]·ρ_i·A_i + p_i·Σ_a [未裁剪项]·ρ_a·A_a
                                = Σ_a Σ_a' ... 化简后 = (p_i·Σ_a k_a) − k_i,
                        其中 k_a = [未裁剪]·ρ_a·A_a。这一条由 test_grad/test_ppomcts
                        的中心差分与"收紧裁剪位移更小"两条断言共同钉住。
         熵奖励       : + entropyCoef·(−Σ_a p_a log p_a) 对 z 的梯度 =
                        entropyCoef·( −p_i·(log p_i + 1) + p_i·Σ_a p_a(log p_a + 1) )
                        = entropyCoef·p_i·(H̃ − log p_i − 1),  H̃ = Σ_a p_a(log p_a + 1)
                        (写成"p_i 乘一个中心化项"是必须的: 熵对 z 的梯度天然与 z 的
                         平移无关, 少掉那个中心化项就等价于给所有 logit 加了一个常数,
                         对 softmax 无害但会让 dlogit 不满足 Σ dlogit = 0。)
    */
    std::vector<float> dlogit(n, 0.0f);
    if (!degenerate) {
        /* k_a: 每个动作在 logit 空间里的"有效优势权重" */
        std::vector<float> k(n, 0.0f);
        const bool useRatio = (clipEps > 0.0f) && (oldProb.size() == n);
        if (useRatio) {
            for (std::size_t i = 0; i < n; i++) {
                const float po = (oldProb[i] > 1e-8f) ? oldProb[i] : 1e-8f;
                const float rho = probs[i] / po;
                const float adv = tgt[i] - oldProb[i];
                const float rhoClipped = std::min(std::max(rho, 1.0f - clipEps),
                                                  1.0f + clipEps);
                /* 取 min 的那一支: 裁剪支更小时 (且继续朝原方向不会改善) 梯度记 0 */
                const bool clipped = (rhoClipped * adv) < (rho * adv);
                k[i] = clipped ? 0.0f : (rho * adv);
            }
        } else {
            /* 纯交叉熵: dL/dz_i = p_i − t_i 正好是"k_i = t_i"这一支 */
            for (std::size_t i = 0; i < n; i++) {
                k[i] = tgt[i];
            }
        }

        double kSum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            kSum += (double)k[i];
        }
        /* 熵项的中心化常数 */
        double hTilde = 0.0;
        if (entropyCoef > 0.0f) {
            for (std::size_t i = 0; i < n; i++) {
                hTilde += (double)probs[i] * (std::log((double)probs[i] + 1e-8) + 1.0);
            }
        }
        for (std::size_t i = 0; i < n; i++) {
            double d = (double)probs[i] * kSum - (double)k[i];
            if (entropyCoef > 0.0f) {
                d += (double)entropyCoef * (double)probs[i]
                     * (hTilde - std::log((double)probs[i] + 1e-8) - 1.0);
            }
            dlogit[i] = (float)d;
        }
    }

    /* CE 数值 (上报口径, 与改动前一致) */
    double ce = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        if (tgt[i] > 0.0f) {
            ce -= (double)tgt[i] * std::log((double)probs[i] + 1e-8);
        }
    }

    /*
       诊断钩子 (2026-09, 默认关闭): `test_grad` 的 E 节要确认"解析梯度是否等于
       p − t"。把 (probs, dlogit) 暴露出去比在测试里重新推一遍可靠 ——
       避免"用同一套公式证明自己"。
    */
    if (gradProbe != nullptr) {
        gradProbe->clear();
        for (std::size_t i = 0; i < n; i++) {
            gradProbe->push_back(probs[i]);      /* [2i]   p_i */
            gradProbe->push_back(dlogit[i]);     /* [2i+1] dL/dz_i */
        }
    }

    /* ---- 4) 头的权重/偏置梯度 (只落合法行) + 往下传的梯度 ei ---- */
    const std::size_t in = head->inputDim;
    const std::size_t wRow = (std::size_t)head->w.sizes[0];     /* == in (行主序) */
    const float *hd = h.val.data();
    const float *wd = head->w.val.data();
    float *gwd = head->g.w.val.data();
    float *gbd = head->g.b.val.data();
    std::vector<float> ei(in, 0.0f);
    for (std::size_t i = 0; i < n; i++) {
        const float d = dlogit[i];
        if (d == 0.0f) {
            continue;       /* 梯度恰好为 0: 这一行不用动 (常见于 p == t 的槽位) */
        }
        const std::size_t a = (std::size_t)legalIdx[i];
        const float *wrow = wd + a * wRow;
        float *grow = gwd + a * wRow;
        for (std::size_t k = 0; k < in; k++) {
            grow[k] += d * hd[k];
            ei[k] += wrow[k] * d;
        }
        if (head->bias && a < head->g.b.size()) {
            gbd[a] += d;
        }
    }

    /* ---- 5) 骨干反向: 从头的前一层开始 (头那一层的 backward 已经手工做完) ---- */
    if (headIndex >= 1) {
        Tensor &eiTensor = actorP[headIndex - 1]->e;
        eiTensor = Tensor(in, 1);
        for (std::size_t k = 0; k < in; k++) {
            eiTensor[k] = ei[k];
        }
        actorP.backwardFrom(headIndex - 1, state);
    }

    /* ---- Critic: 与全量路径完全一致 (只多一条值域约束, 见 ppo.h 的 clampValue) ---- */
    const double vt = (clampValue > 0.0f)
                          ? (double)std::min(std::max(valueTarget, -clampValue), clampValue)
                          : (double)valueTarget;
    Tensor &v = critic.forward(state);
    const double err = (double)v[0] - vt;
    Tensor valueTargetTensor(1, 1);
    valueTargetTensor[0] = (float)vt;
    Tensor mseLoss = Loss::MSE::df(v, valueTargetTensor);
    critic.backward(state, mseLoss);

    batchLossSum += err * err;
    batchActorLossSum += ce;
    batchLossCount++;
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
    /*
       每个 epoch **重新抽** batchSize 条 (注意 `pick` 在内层循环里), 所以一次更新看到
       的是 batchSize×epochs 条经验, 而优化器只在最后调一次。
       两种写法要分清:
         * 现在这种 (每遍抽新样本) —— 等于把批放大 epochs 倍;
         * "把同一批重复过 N 遍" —— 批内权重不变, 第 N 遍的梯度与第 1 遍逐位相同,
           而 clipGrad 会把这个纯倍数归一掉, 于是对更新方向**毫无影响**, 只是白烧算力。
    */
    for (int e = 0; e < epochs; e++) {
        for (std::size_t b = 0; b < batchSize; b++) {
            const ReplaySample &s = replay[pick(Random::engine)];
            state = s.state;
            /*
               R2: 样本带完整合法集时走稀疏口径 (不建 8100 维稠密目标, 也不做全量
               softmax)。maskedTrainHead=false 就是"旧学习问题"的对照组。
            */
            if (maskedTrainHead && !s.legalIdx.empty()) {
                accumulateGradSparse(state, s.legalIdx, s.actionIdx, s.actionProb,
                                     s.valueTarget, s.oldProb);
                continue;
            }
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

    /*
       ---- 整条轨迹累积一次更新 (P3 的累积在在线路径上的落地) ----

       原来是逐步 trainStep: 每步一次完整的全参数 RMSProp, 而且 lastLoss 只留**最后
       一条样本**的值 —— 界面上那条"训练损失曲线"因此画的是单样本损失 (目标的量级
       由最后一步吃了什么决定), 实测"最后一条 / 整条批平均"能差 1.5 倍以上, 遇到
       吃車/吃將那种重尾样本差几个数量级, 曲线看起来就是低占空比的脉冲。

       改成累积整条轨迹再 applyGradients 一次, 同时拿到两件事:
         * 上报的是**批平均**损失 (与 DQN 报"平均平方 TD 误差"同一口径);
         * 优化器调用次数从 N 次降到 1 次 —— 实测每样本便宜 2.9x
           (见 test_ppomcts 的 [9] 与 docs/agents_design.md §17.3)。
    */
    resetMoeBatchStats();
    for (int t = 0; t <= end; t++) {
        accumulateGrad(trajectory[(std::size_t)t].state,
                       trajectory[(std::size_t)t].action,
                       returns[(std::size_t)t]);
    }
    applyGradients(learningRate);

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

bool RL::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    /*
       两次调用都要**无条件执行**, 所以先各自存结果再与 —— 写成
       `a.save(x) == 0 && b.save(y) == 0` 会在第一个失败时短路掉第二个,
       于是 actor 写失败时 critic 文件不会被刷新 (留下新旧不一致的一对)。
    */
    const int a = actorP.save(actorPara);
    const int c = critic.save(criticPara);
    return (a == 0) && (c == 0);
}

bool RL::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    /* 同上: 不短路, 两个都尝试 */
    const int a = actorP.load(actorPara);
    const int c = critic.load(criticPara);
    if (a != 0 || c != 0) {
        /*
           注意: 单个 Net::load 是"整文件预校验通过才改动网络"的 (见 net.hpp), 所以
           失败的那个网络本身没有被改坏; 但**两个之间**可能出现"actor 换了、critic
           没换"的混合状态。调用方必须把它当成失败处理并停止继续训练, 而不是接着用。
        */
        std::cerr << "[weights] PPO::load 失败: actor="
                  << (a == 0 ? "ok" : "拒绝") << ", critic="
                  << (c == 0 ? "ok" : "拒绝")
                  << " —— 权重可能处于 actor/critic 不一致的混合状态" << std::endl;
        return false;
    }
    return true;
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
