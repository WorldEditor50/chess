#include "sac.h"
#include "layer.h"
#include "loss.h"
#include <limits>
#include "moe.hpp"
#include "sparse_moe.hpp"

namespace {

/*
   初始化缩放 (与 rl/ppo.cpp 的 scaleLayerInit 同一套依据, 数值见 rl/ppo.h):
   `iFcLayer` 的构造函数把权重初始化成 U(-1,1), d 维输入下 pre-activation 的标准差
   约 sqrt(d/3); d=1260 时约 20 —— Tanh/Sigmoid 直接饱和、梯度接近 0。
   只缩**普通**的 iFcLayer: 稀疏 MoE 层不是 iFcLayer (dynamic_cast 返回 nullptr),
   它的专家权重已经在 SparseMoE 的构造函数里用 scaleExpertInit 缩过了。
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

/* 网络里所有的稀疏 MoE 层 (辅助损失 / 诊断都要遍历它们) */
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

/* 数值稳定的整向量 softmax: pi = exp(z - max) / Σ */
void softmaxStable(const std::vector<float> &z, RL::Tensor &pi)
{
    const std::size_t n = z.size();
    pi = RL::Tensor(n, 1);
    if (n == 0) {
        return;
    }
    float m = z[0];
    for (std::size_t i = 1; i < n; i++) {
        if (z[i] > m) { m = z[i]; }
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const float e = std::exp(z[i] - m);
        pi[i] = e;
        sum += (double)e;
    }
    if (!(sum > 1e-12) || !std::isfinite(sum)) {
        /* 退化 (logits 全 NaN/inf): 均匀分布, 而不是全 0 */
        const float u = 1.0f / (float)n;
        for (std::size_t i = 0; i < n; i++) { pi[i] = u; }
        return;
    }
    const float inv = (float)(1.0 / sum);
    for (std::size_t i = 0; i < n; i++) { pi[i] *= inv; }
}

} // namespace

/* ============================================================
 *  构造
 * ============================================================ */
RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
    :stateDim((int)stateDim_), actionDim((int)actionDim_),
     hiddenDim((int)(hiddenDim > 0 ? hiddenDim : 64)),
     gamma(0.99f), exploringRate(1.0f), learningSteps(0)
{
    annealing = ExpAnnealing(0.25, 1, 1e-7);
    alpha = GradValue((std::size_t)actionDim, 1);
    alpha.val.fill(0.65f);
    /* target entropy = -log|A| (SAC 的标准启发式); 给了掩码时按合法动作数逐样本算 */
    H0 = -std::log((float)actionDim);

    /*
        actor: stateDim -> SparseMoE(专家=TB<16>, E=4, top-1) -> Tanh(stateDim -> h)
               -> Linear(h -> actionDim)   [**logits**, 掩码 softmax 在类内自己做]

        critic 的输入是 [state; π] (见文件顶部), 所以第一层的输入维是 stateDim+actionDim。
    */
    actor = buildNet(stateDim, true);

    for (int i = 0; i < QNET_NUM; i++) {
        critics[i]       = buildNet(criticInputDim(), true);
        criticsTarget[i] = buildNet(criticInputDim(), false);
        /* 目标网必须从在线网拷一份过去, 否则自举项一开始就是纯噪声
           (Net 的拷贝是浅拷贝 = 共享层指针, 深拷贝只能走 copyTo) */
        critics[i].copyTo(criticsTarget[i]);
    }
}

RL::Net RL::SAC::buildNet(int inDim, bool withGrad) const
{
    Net::Layers layers;
    layers.push_back(std::make_shared<SparseMoE<SACExpert,
                                                 MOE_EXPERTS,
                                                 MOE_TOPK> >(
        inDim, withGrad, 0));
    layers.push_back(Layer<Tanh>::_(inDim, (std::size_t)hiddenDim, true, withGrad));
    layers.push_back(Layer<Linear>::_((std::size_t)hiddenDim, (std::size_t)actionDim,
                                      true, withGrad));
    Net net(layers);
    scaleLayerInit(net);
    return net;
}

/* ============================================================
 *  回放池
 * ============================================================ */
void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    perceive(state, action, nextState, reward, done, Tensor(), Tensor());
}

void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done,
                       const Tensor& legalMask)
{
    /* 同一套合法动作对 s 与 s' 都适用时用这个重载; s' 的合法集不同就用 7 参数版 */
    perceive(state, action, nextState, reward, done, legalMask, legalMask);
}

void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done,
                       const Tensor& legalMask,
                       const Tensor& nextLegalMask)
{
    Transition tr(state, action, nextState, reward, done);
    tr.legalMask = legalMask;          /* 空 = 旧的全量口径 */
    tr.nextLegalMask = nextLegalMask;
    memories.push_back(std::move(tr));
    trimMemory();
}

void RL::SAC::trimMemory()
{
    while (memories.size() > maxMemorySize) {
        memories.pop_front();
    }
}

/* ============================================================
 *  策略前向
 * ============================================================ */
void RL::SAC::policy(const Tensor &state, const Tensor &legalMask, Tensor &pi)
{
    const int A = actionDim;
    const bool useMask = maskedTrainHead && !legalMask.empty() &&
                         (int)legalMask.size() >= A;

    pi = Tensor((std::size_t)A, 1);
    pi.zero();
    if (!useMask) {
        /* ---- 全量口径: 整向量 softmax ---- */
        Tensor &logits = actor.forward(state);
        std::vector<float> z((std::size_t)A, 0.0f);
        for (int a = 0; a < A; a++) { z[(std::size_t)a] = logits[(std::size_t)a]; }
        softmaxStable(z, pi);
        return;
    }

    /* ---- R2: 头只算合法列, 在合法集上归一 ---- */
    std::vector<int> idx;
    idx.reserve((std::size_t)A);
    for (int a = 0; a < A; a++) {
        if (legalMask[(std::size_t)a] > 0.5f) {
            idx.push_back(a);
        }
    }
    if (idx.empty()) {
        /* 一个合法动作都没有: 退回全量口径 (而不是给一张全 0 的分布) */
        Tensor &logits = actor.forward(state);
        std::vector<float> z((std::size_t)A, 0.0f);
        for (int a = 0; a < A; a++) { z[(std::size_t)a] = logits[(std::size_t)a]; }
        softmaxStable(z, pi);
        return;
    }

    Tensor &h = actor.forwardTrunk(state);
    std::vector<float> z;
    if (!actor.sparseLogits(h, idx, z) || z.size() != idx.size()) {
        /* 头不支持稀疏路径: 全量前向再取子集 (数值等价, 只是慢) */
        Tensor &logits = actor.forward(state);
        std::vector<float> zf((std::size_t)A, 0.0f);
        for (int a = 0; a < A; a++) { zf[(std::size_t)a] = logits[(std::size_t)a]; }
        std::vector<float> zs;
        zs.reserve(idx.size());
        for (std::size_t i = 0; i < idx.size(); i++) {
            zs.push_back(zf[(std::size_t)idx[i]]);
        }
        z.swap(zs);
    }
    Tensor sub;
    softmaxStable(z, sub);
    for (std::size_t i = 0; i < idx.size(); i++) {
        pi[(std::size_t)idx[i]] = sub[i];
    }
}

RL::Tensor &RL::SAC::action(const RL::Tensor &state)
{
    policy(state, Tensor(), m_pi);
    return m_pi;
}

RL::Tensor &RL::SAC::eGreedyAction(const RL::Tensor &state)
{
    Tensor &pi = action(state);
    return eGreedy(pi, exploringRate, true);
}

RL::Tensor &RL::SAC::gumbelMax(const RL::Tensor &state)
{
    Tensor &pi = action(state);
    return RL::gumbelSoftmax(pi, 0.9);
}

/* ============================================================
 *  [MoE] 批统计复位
 * ============================================================ */
void RL::SAC::resetMoeBatchStats()
{
    if (moeAuxCoef <= 0.0f) {
        return;
    }
    Net *nets[1 + 2 * QNET_NUM];
    int n = 0;
    nets[n++] = &actor;
    for (int i = 0; i < QNET_NUM; i++) {
        nets[n++] = &critics[i];
        nets[n++] = &criticsTarget[i];
    }
    for (int i = 0; i < n; i++) {
        std::vector<ISparseMoE*> moes = sparseMoeLayers(*nets[i]);
        for (std::size_t k = 0; k < moes.size(); k++) {
            moes[k]->resetBatchStats();
        }
    }
}

/* ============================================================
 *  [P3] accumulateGrad: 一条样本的前向 + 反向 (不碰优化器)
 * ============================================================ */
void RL::SAC::accumulateGrad(const Transition &x)
{
    const int A = actionDim;
    const bool useMask = maskedTrainHead && !x.legalMask.empty() &&
                         (int)x.legalMask.size() >= A;

    /* ---- 1) 目标侧: 目标网算下一局面的软价值 ----
       指数移动的期望必须只在**该局面自己的合法动作**上算 (注意用的是 nextLegalMask,
       不是 legalMask —— 走一步之后合法集就变了, 拿 s 的掩码去算 s' 等于给非法着法
       分配价值, 而策略永远走不到那里, 那是纯噪声)。 */
    Tensor piNext;
    policy(x.nextState, x.nextLegalMask, piNext);
    Tensor nextIn = Tensor::concat(0, x.nextState, piNext);

    const Tensor *targetQ[QNET_NUM];
    for (int i = 0; i < QNET_NUM; i++) {
        targetQ[i] = &criticsTarget[i].forward(nextIn);
    }

    float nextValue = 0.0f;
    for (int a = 0; a < A; a++) {
        const float pa = piNext[(std::size_t)a];
        if (pa <= 0.0f) {
            continue;
        }
        float minQa = std::numeric_limits<float>::max();
        for (int i = 0; i < QNET_NUM; i++) {
            minQa = std::min(minQa, (*targetQ[i])[(std::size_t)a]);
        }
        nextValue += pa * (minQa - alpha[(std::size_t)a] * std::log(pa + 1e-8f));
    }

    /*
       符号: 所有价值都是"该局面走棋方视角"。s' 轮到**对手**走, 所以自举项取负
       (negamax):   y = r − γ(1−done)·V(s')
    */
    const float y = x.reward - gamma * (x.done ? 0.0f : 1.0f) * nextValue;

    /* ---- 2) 当前局面: π 与在线双 Q ---- */
    Tensor pi;
    policy(x.state, x.legalMask, pi);
    Tensor in = Tensor::concat(0, x.state, pi);

    const Tensor *onlineQ[QNET_NUM];
    for (int i = 0; i < QNET_NUM; i++) {
        onlineQ[i] = &critics[i].forward(in);
    }

    std::vector<float> minQ((std::size_t)A, 0.0f);
    for (int a = 0; a < A; a++) {
        float q = std::numeric_limits<float>::max();
        for (int i = 0; i < QNET_NUM; i++) {
            q = std::min(q, (*onlineQ[i])[(std::size_t)a]);
        }
        minQ[(std::size_t)a] = q;
    }

    /* ---- 3) critic 损失: 只对实际走过的那一步回归 ----
       (SAC 的标准做法: 其余动作的误差为 0) */
    const int k = (int)x.action.argmax();
    const float err = (*onlineQ[0])[(std::size_t)k] - y;
    batchLossSum += (double)err * (double)err;

    for (int ci = 0; ci < QNET_NUM; ci++) {
        /* 注意 target 必须是**副本**: 后面 backward 会把 onlineQ 指向的输出清零 */
        Tensor target = (ci == 0) ? *onlineQ[0] : *onlineQ[ci];
        target[(std::size_t)k] = y;
        critics[ci].backward(in,
            Loss::MSE::df(*(ci == 0 ? onlineQ[0] : onlineQ[ci]), target));
    }

    /* ---- 4) 策略梯度 ----
       J(π) = Σ_a π_a·(α_a·log π_a − min_i Q_i(s,a))
       dJ/dπ_a = α_a·(log π_a + 1) − min_i Q_i(s,a)
       (非法动作不参与: π≡0, 而且我们**绝不能**把它抬起来 —— 所以 g 在掩码外为 0)
    */
    std::vector<float> g((std::size_t)A, 0.0f);
    double actorLoss = 0.0;
    for (int a = 0; a < A; a++) {
        const float pa = pi[(std::size_t)a];
        if (useMask && x.legalMask[(std::size_t)a] <= 0.5f) {
            continue;
        }
        if (pa <= 0.0f) {
            continue;
        }
        g[(std::size_t)a] = alpha[(std::size_t)a] * (std::log(pa + 1e-8f) + 1.0f)
                            - minQ[(std::size_t)a];
        actorLoss += (double)pa * ((double)alpha[(std::size_t)a] * std::log((double)pa + 1e-8)
                                   - (double)minQ[(std::size_t)a]);
    }
    batchActorLossSum += actorLoss;

    /*
       dL/dz = Jᵀ_softmax(π)·g, 用**掩码后**的 π 算 (形式与 Softmax 层相同, 但
       归一化集合不同 —— 所以不能对 Layer<Softmax> 的输出打补丁)。掩码外 π=0,
       于是 dz 在非法列上**恰好为 0**, 头那层对应行的权重梯度也就恒为 0。
    */
    Tensor dz((std::size_t)A, 1);
    dz.zero();
    double dot = 0.0;
    for (int a = 0; a < A; a++) {
        dot += (double)g[(std::size_t)a] * (double)pi[(std::size_t)a];
    }
    for (int a = 0; a < A; a++) {
        const float pa = pi[(std::size_t)a];
        if (pa <= 0.0f) {
            continue;
        }
        dz[(std::size_t)a] = pa * (g[(std::size_t)a] - (float)dot);
    }
    /* 头的输入有两处来源 (全量 forward / forwardTrunk+sparseLogits), 但两条路径都会
       把倒数第二层的 o 写进缓存, 所以通用 backward 拿到的 x 是一致的。 */
    actor.backward(x.state, dz);

    /* ---- 5) α: 梯度整批累加, 由 applyGradients 取平均后再更新 ----
       J(α) = α·(H − H̄)  =>  dJ/dα = H − H̄
       熵低于目标 -> 该样本的梯度为负 -> α 变大 -> 更探索。 */
    {
        float H = 0.0f;
        int nLegal = 0;
        for (int a = 0; a < A; a++) {
            const float pa = pi[(std::size_t)a];
            if (pa <= 0.0f) {
                continue;
            }
            H -= pa * std::log(pa);
        }
        if (useMask) {
            for (int a = 0; a < A; a++) {
                if (x.legalMask[(std::size_t)a] > 0.5f) { nLegal++; }
            }
        }
        /*
           目标熵: 有掩码时用 **-log(合法动作数)** —— 用 -log|A| 的话目标高到根本
           达不到 (掩码把 H 的上界压到了 log(合法数)), α 会一路顶到夹逼上界。
        */
        const float Hbar = (useMask && nLegal > 1)
                               ? -std::log((float)nLegal)
                               : H0;
        batchAlphaGradSum += (double)(H - Hbar);
    }

    batchLossCount++;
}

/* ============================================================
 *  [P3]+[MoE] applyGradients: 注入辅助损失 -> 优化器 -> 上报 -> 清统计
 * ============================================================ */
void RL::SAC::applyGradients(float learningRate)
{
    if (learningRate > 0.0f) {
        learningRateActor = learningRate;         /* 参数现在真的生效 (见文件顶部) */
        learningRateCritic = learningRate * 0.1f; /* 保持原来的 1/10 比例 */
    }

    /*
       ---- 稀疏 MoE 的负载均衡辅助损失 ----
       在优化器之前、主反向之后: 本批每个层的前向次数、被选中的专家次数、门控概率
       之和都已经记好了, addAuxGradient 用它们算出"哪些专家被喂爆了", 把那些专家的
       logit 压下去、把饿着的抬起来 (Switch Transformer 的 L_aux = E·Σ f_i·P_i)。
       没有它时 softmax 的反向会持续压低没被选中的专家, 路由几轮内就坍缩。
       只对**会被优化器更新**的网络注入 (actor + 在线 critic); 目标网不训练, 给它
       注入等于往一个永远不会被应用的 g_wg 里加数。
    */
    if (moeAuxCoef > 0.0f) {
        Net *nets[1 + QNET_NUM];
        int n = 0;
        nets[n++] = &actor;
        for (int i = 0; i < QNET_NUM; i++) {
            nets[n++] = &critics[i];
        }
        for (int i = 0; i < n; i++) {
            std::vector<ISparseMoE*> moes = sparseMoeLayers(*nets[i]);
            for (std::size_t k = 0; k < moes.size(); k++) {
                moes[k]->addAuxGradient(moeAuxCoef);
            }
        }
    }

    /* ---- 一次优化器更新 (整批只调一次) ---- */
    actor.RMSProp(learningRateActor, 0.9f, 0.0f);
    for (int i = 0; i < QNET_NUM; i++) {
        critics[i].RMSProp(learningRateCritic, 0.9f, 0.0f);
    }

    if (batchLossCount > 0) {
        alpha.g[0] = (float)(batchAlphaGradSum / (double)batchLossCount);
        lastLoss = batchLossSum / (double)batchLossCount;
        lastActorLoss = batchActorLossSum / (double)batchLossCount;
        lastBatchSamples = batchLossCount;   /* [P4] 这批一共攒了多少条样本 */
    } else {
        /* 什么都没累积: 绝不能让上一批的 α 梯度被重复应用一次 */
        alpha.g.zero();
    }
    alpha.RMSProp(learningRateAlpha, 0.9f, 1e-6f);
    alpha.clamp(0.25f, 0.64f, 1.0f);
    annealing.step();

    /* ---- 目标网 Polyak 同步 ---- */
    learningSteps++;
    if (replaceTargetIter > 0 && (learningSteps % (int)replaceTargetIter) == 0) {
        /* 与原来一致: 每个 critic 的 tau 略有不同 (1e-3 + 2e-3·i) */
        float tau = 1e-3f;
        for (int i = 0; i < QNET_NUM; i++) {
            critics[i].softUpdateTo(criticsTarget[i], tau);
            tau += 2e-3f;
        }
    }

    exploringRate *= 0.99999f;
    exploringRate = exploringRate < 0.3f ? 0.3f : exploringRate;

    /* 批统计清零: 下一次 applyGradients 的损失/α 梯度只反映下一批 */
    batchLossSum = 0.0;
    batchActorLossSum = 0.0;
    batchAlphaGradSum = 0.0;
    batchLossCount = 0;
}

/* ============================================================
 *  learn / learnFromReplay
 * ============================================================ */
void RL::SAC::learn(size_t maxMemorySize_, size_t replaceTargetIter_,
                    size_t batchSize, float learningRate)
{
    learnFromReplay(batchSize, 1, learningRate, maxMemorySize_, replaceTargetIter_);
}

bool RL::SAC::learnFromReplay(std::size_t batchSize,
                              int epochs,
                              float learningRate,
                              std::size_t maxMemorySize_,
                              std::size_t replaceTargetIter_)
{
    maxMemorySize = maxMemorySize_;
    replaceTargetIter = replaceTargetIter_;

    if (epochs <= 0 || batchSize == 0 || memories.size() < batchSize) {
        return false;
    }

    std::uniform_int_distribution<std::size_t> pick(0, memories.size() - 1);
    std::size_t accumulated = 0;

    /*
       [P4] 每个 epoch **重新从池里抽** batchSize 条 (与 RL::PPO::learnFromReplay 完全
       同一做法), 累积梯度后优化器只调一次 (这才是 [P3] 的意义)。

       注意"多 epoch"到底改了什么 —— 这里有个容易记反的地方:
         * 若把**同一批**重复过 N 遍, 由于批内权重不变, 第 N 遍的梯度与第 1 遍逐位
           相同, 而在 clipGrad=true (RL::Net::RMSProp 的默认) 下 `dw /= |dw|` 会把
           这个倍数**完全归一掉** —— 于是"重复同一批"对更新方向毫无影响, 只是白烧
           算力。所以这里不是那种写法。
         * 真正的做法是每遍都抽**新的**样本: 一次更新看到 batchSize×epochs 条经验,
           等于把批放大 epochs 倍 (但仍然只调一次优化器)。回放池越大、生成一条样本
           越贵 (自对弈 + 搜索), 这笔账越划算。
    */
    resetMoeBatchStats();
    for (int e = 0; e < epochs; e++) {
        for (std::size_t b = 0; b < batchSize; b++) {
            const std::size_t idx = pick(Random::engine);
            accumulated += 1;
            accumulateGrad(memories[idx]);
        }
    }
    applyGradients(learningRate);

    trimMemory();
    return true;
}

/* ============================================================
 *  存取
 * ============================================================ */
void RL::SAC::save()
{
    actor.save("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string criticName = std::string("sac_critic_") + std::to_string(i);
        critics[i].save(criticName);
    }
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string criticName = std::string("sac_critic_") + std::to_string(i);
        critics[i].load(criticName);
        critics[i].copyTo(criticsTarget[i]);
    }
}

/* ============================================================
 *  稀疏 MoE 诊断
 * ============================================================ */
int RL::SAC::moeLayerCount() const
{
    RL::Net &a = const_cast<RL::Net&>(actor);
    RL::Net &c = const_cast<RL::Net&>(critics[0]);
    return (int)(sparseMoeLayers(a).size() + sparseMoeLayers(c).size());
}

int RL::SAC::moeExpertCount() const
{
    RL::Net &a = const_cast<RL::Net&>(actor);
    std::vector<ISparseMoE*> layers = sparseMoeLayers(a);
    return layers.empty() ? 0 : layers[0]->expertCount();
}

int RL::SAC::moeTopK() const
{
    RL::Net &a = const_cast<RL::Net&>(actor);
    std::vector<ISparseMoE*> layers = sparseMoeLayers(a);
    return layers.empty() ? 0 : layers[0]->topK();
}

void RL::SAC::moeUsage(std::vector<long long> &out) const
{
    out.clear();
    const int experts = moeExpertCount();
    if (experts <= 0) {
        return;
    }
    out.assign((std::size_t)experts, 0);

    std::vector<long long> one;
    RL::Net &a = const_cast<RL::Net&>(actor);
    std::vector<ISparseMoE*> layers = sparseMoeLayers(a);
    for (int i = 0; i < QNET_NUM; i++) {
        RL::Net &c = const_cast<RL::Net&>(critics[i]);
        std::vector<ISparseMoE*> cl = sparseMoeLayers(c);
        layers.insert(layers.end(), cl.begin(), cl.end());
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->usageSnapshot(one);
        for (std::size_t e = 0; e < one.size() && e < out.size(); e++) {
            out[e] += one[e];
        }
    }
}

void RL::SAC::resetMoeUsage()
{
    RL::Net &a = const_cast<RL::Net&>(actor);
    std::vector<ISparseMoE*> layers = sparseMoeLayers(a);
    for (int i = 0; i < QNET_NUM; i++) {
        RL::Net &c = const_cast<RL::Net&>(critics[i]);
        std::vector<ISparseMoE*> cl = sparseMoeLayers(c);
        layers.insert(layers.end(), cl.begin(), cl.end());
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->resetUsage();
    }
}
