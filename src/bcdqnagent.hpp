#ifndef BCDQNAGENT_HPP
#define BCDQNAGENT_HPP

/*
 * ============================================================================
 *  bcdqnagent.hpp - 两个 **Q 类 agent** 的 BC 胶水层 (留出 CE / top-1 评测路径)
 * ============================================================================
 *
 *  适用: `DQNABAgent` (src/dqnabagent.h, dueling + AB 搜索)
 *        `DQNMCTSMOETbAgent` (src/dqnmctsmoetbagent.h, PUCT + 稀疏 MoE 骨干)
 *
 * 为什么这里是 `evaluate` / `update` 的**非模板重载** (而不是再加一个
 * `template <class QLike>`): 这两支的"策略"**不是一个可以调用的成员函数** —— DQN 没有
 * actor, 它的策略是"Q 在合法列上隐含的先验"。PG 那两支 (PPO/SAC) 能共用一条模板是因为
 * 它们都有 `policy(state,mask,pi)` 这个出口; 这里没有, 而且两支的 Q 出口也不同
 * (DQNAB 是 dueling 双头的 `evaluateNode`, DQNMCTS 是 `qValuesFull`)。所以本层照
 * `BC::evaluate(PPOMCTSAgent&)` / `BC::update(PPOMCTSAgent&)` 的做法: **一支一个精确重载**,
 * 各自把"自己的 Q 出口"接到同一对 `rl/bc.h` 工具上。
 *
 * ---------------------------------------------------------------------------
 *  口径 (唯一一份; 报告、注释、自检三处必须逐条对得上)
 * ---------------------------------------------------------------------------
 *  * π = **合法槽位**上对 Q 做 `softmax(Q / T)`, **T 固定 1.0** (不做温度搜索);
 *  * CE = `RL::maskedCrossEntropy(pi, t)`; 对 logits 的解析梯度走
 *    `RL::maskedCeLogitGrad(pi, mask, t, dz)` (即 dz = π − t, 非法列**恒为 0**);
 *    这两个函数与 SAC 三支用的是同一对 (rl/bc.h), 不是另写一套流程;
 *  * top-1 = 合法槽位上 Q 的 argmax 命中率 (尺度无关, 比 CE 干净的正是这一列);
 *  * 反向 (与各自既有的训练路径**逐条同序**):
 *      DQNAB   : `m_aHead.backward(h, dz)` → `m_trunk.backward(state, m_aHead.inputGrad)`
 *                (同 src/dqnabagent.cpp 的 accumulateGrad: 头先读骨干缓存, 骨干只反向一次);
 *      DQNMCTS : `qHead.backward(h, dz)`   → `trunk.backward(state, qHead.inputGrad)`
 *                (同 src/dqnmctsmoetbagent.cpp 的 learnBatch)。
 *    反向必须用**最近一次前向的缓存** ⇒ 本层严格"先前向、后反向", 中间不做别的前向。
 *  * 每批末尾**一次** `RMSProp(lr, 0.9f, 0.0f)`:
 *      DQNAB   : 对 `m_trunk` 与 `m_aHead` 各一次;
 *      DQNMCTS : 对 `trunk` 与 `qHead` 各一次。
 *    ⚠ `qNet` / `q2Net` 是**视图** (与 trunk/qHead 共享同一批层对象), 对它们再调一次
 *      RMSProp = 同一批权重在同一批样本上被更新两次 = **学习率翻倍**。所以这里一次都不调
 *      (与 learnBatch 顶部那条注释同一个坑)。
 *
 * ---------------------------------------------------------------------------
 *  ⚠⚠ 这条口径的**边界** (报告里必须一起报, 否则读数会被误读 —— 三条都不许省)
 * ---------------------------------------------------------------------------
 *  1. 这里监督的是 **"Q 隐含的合法列先验"**, **不是** agent 的真实决策策略:
 *       * DQNAB 的真实决策走**根搜索值** `softmax(v_a / T)` (selectMove 的根全宽搜索),
 *         与根节点的 Q 排序只是相关、不相等;
 *       * DQNMCTS 的真实决策走 **PUCT + priors** 的搜索分布。
 *     所以 "prior top-1 涨了" 只能读成 "根节点 Q 的排序更像老师了", **不能**读成
 *     "它下得更像老师了" (后者只有对局/一致率口径能回答)。
 *  2. **CE 的绝对值与 PPO/SAC 的 BC 不可比**: 那边的 logits 是**策略头** (量纲由策略
 *     loss 定), 这边的 logits 是**值头输出** (量纲由 RL 的 TD loss 定, 幅度可能差几个
 *     数量级 —— 未训练的值头输出本来就落在 ±0.1 量级)。跨口径比 CE 是拿两把不同的尺子
 *     量同一个东西; 成立的只有 **"同 teacher / 同数据 / 同 seed 下换宽度/换骨干"** 的
 *     相对比较 (本工具存在的目的正是这个)。
 *  3. **两边的动作分母不是一回事**: DQNAB 的 8100 列是 `规范from*90 + 规范to` 的**双射**
 *     (无别名); DQNMCTS 的 128 列是**哈希槽**, 槽位有别名 (同一个槽被多个合法着法挤占,
 *     它的自检面板自己报这个数)。本口径在**槽位**上算 π/CE/top-1 (掩码 = 槽位集合),
 *     别名槽只算一次 (见 policyFromQLegal 的注释)。这是"网络真正能表达的东西", 但读
 *     top-1 时必须知道: DQNMCTS 的分母是**槽位数**, 不是着法数。
 *
 * ---------------------------------------------------------------------------
 *  DQNAB (dueling) 的数学依据 —— 这就是"能直接用 rl/bc.h 那对函数"的原因
 * ---------------------------------------------------------------------------
 *      Q_i = V + A_i − mean_legal(A)
 *  `V` 与 `mean_legal(A)` 对**这个状态**都是常数 ⇒ π = softmax_合法(Q) 逐位只由 A 决定;
 *  又因为 π 与 t 都是同一条合法集上的分布 (Σπ = 1, Σt = 1), 于是 Σ_i(π_i − t_i) = 0,
 *  dueling 的组合项给出
 *      dL/dA_j = π_j − t_j ,      dL/dV = Σ_i dL/dQ_i = Σ_i (π_i − t_i) = **0**。
 *  两条推论都进了代码:
 *    * A 头的反向直接喂 `dz = π − t` (它的输入就是 h, 与 Q 的口径一致);
 *    * **V 头一个字节都不该动** ⇒ `update()` 在每个批的**首尾各取一次 V 头权重的
 *      64 位校验和**, 不一致就响亮失败 (`std::abort`, 见 `weightHash`)。这把上面那条
 *      推导从"一段注释里的数学"变成"每次更新都真的被检查一遍的事实" —— 本工程记过
 *      太多"推导对、代码悄悄多动了一路"的事故 (SAC 共享骨干那一支就是先例)。
 *
 * ---------------------------------------------------------------------------
 *  ⚠ 三条刻意不做的事 (都是"做了会让读数不可信"的那一类)
 * ---------------------------------------------------------------------------
 *  1. **不调 MoE 负载均衡辅助损失** (`addAuxGradient`): 那会往主干里加一个与 CE 无关的
 *     方向, 于是 "dz = π − t" 就不再是"这一步实际下下去的方向", 而本工具全部读数的
 *     可信度都建立在"它就是 CE"上。代价 (已知缺口, 报告里写明): BC 期间稀疏 MoE 的
 *     路由**没有均衡回路**。
 *  2. **不碰目标网 / 回放池 / 棋盘**: 目标网 (m_trunkT/qHeadTarget 等) 一个字节不动;
 *     `sampleFrom` 会改 `ag.chess` (那是它的既定契约, bcagent.hpp 写明), 由 BCRun 还原。
 *  3. **不走 value gate** (`DQNABAgent::learnBatch` 的那套手工锚门控): 那是 RL 更新的
 *     安全机制 (防"一次毒化更新把 V 打飞"), 而这里 V 头根本不动, 没有它要防的东西。
 *
 * ⚠ 前向的副作用 (读数上要知道): 本路径复用**搜索/训练同一套**前向 (DQNAB 走
 *   `evaluateNode`, DQNMCTS 走 `qValuesFull`, 两者的 inference/withGrad 口径与各自
 *   learnBatch 一致) ⇒ 会累加 `m_leafEvals` 这个诊断计数, 并给稀疏 MoE 的 `usageBatch`
 *   记账。MoE 批统计在每个批边界被 `resetMoeBatchStats()` 清一次 (与 learnBatch 同序),
 *   所以一次 BC 跑完之后的负载读数只覆盖**最后一个批** —— 不要拿它当"整段 BC 的路由分布"。
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "bcagent.hpp"
#include "chess.h"
#include "dqnabagent.h"
#include "dqnmctsmoetbagent.h"
#include "rl/bc.h"
#include "rl/sparse_moe.hpp"   /* ISparseMoE::resetBatchStats (批边界的统计清理, 见 resetMoeBatchStatsOf) */

namespace BC {

/* ---- 口径常量: 温度固定 1.0 (写成常量而不是字面量, 免得有人"顺手"做成旋钮) ---- */
static constexpr double DQN_BC_TEMP = 1.0;

/*
 * 本口径的**运行期自检计数** (header-only ⇒ `inline` 变量, C++17)。
 * 它们由 `BCRun::caliperFor` 打进报告 —— 报告的每一个字都要有对应的读数在跑,
 * 否则那就是一句"应该没问题" (本工程把这种句子当缺陷)。
 */
namespace Dqn {

inline long long samples = 0;       /* 走过本口径 (evaluate + update) 的样本数 */
inline long long vHeadGuarded = 0;  /* DQNAB: "V 头逐位不变"被真正校验过的批数 */
inline long long aliasDropped = 0;  /* 同一槽位被多个合法着法挤占而只算一次的次数 */
inline long long piMassWarn = 0;    /* Σπ(合法槽位) 偏离 1 超过 1e-4 的样本数 (必须是 0) */

} // namespace Dqn

/* ============================================================================
 *  1) 一个小工具: 合法列上的 Q -> ACTION_DIM 维稠密策略 π
 * ============================================================================
 *  这是两支唯一的公共口径代码 (所以只有一份, 免得两边分叉)。
 *
 *  去重的理由 (DQNMCTS 那一支是真会发生的): `legalIdx` 里同一个槽位可能出现多次 ——
 *  128 个哈希槽装 ~8100 个着法, 别名是常态。同一个槽位的 Q **逐位相同** (网络只看
 *  "状态 + 槽位"), 所以把它算两次的唯一后果是"exp 进了分母两遍" ⇒ Σπ < 1 ⇒
 *  `dz = π − t` 的和不再为 0 ⇒ **dueling 那条 "dL/dV = 0" 的推导就不成立了**。
 *  这不是"丢掉一个合法着法": 槽位是网络的动作空间, 别名着法本来就不可分。
 *
 *  返回 false = 一个合法槽位都没有 (调用方按"样本不合格"计数并丢弃)。
 */
inline bool policyFromQLegal(std::size_t actionDim, const std::vector<int> &legalIdx,
                             const std::vector<double> &qLegal, RL::Tensor &pi,
                             int *aliasDroppedOut = nullptr)
{
    if (pi.size() != actionDim) {
        pi = RL::Tensor(actionDim, 1);
    }
    pi.zero();
    if (legalIdx.empty() || qLegal.size() != legalIdx.size()) {
        return false;
    }

    /* ---- 去重 (保持出现顺序) + 取 max (数值稳定的 softmax) ---- */
    std::vector<int> slot;
    std::vector<double> q;
    slot.reserve(legalIdx.size());
    q.reserve(legalIdx.size());
    double mx = 0.0;
    for (std::size_t i = 0; i < legalIdx.size(); i++) {
        const int a = legalIdx[i];
        if (a < 0 || (std::size_t)a >= actionDim) {
            continue;                       /* 越界下标按非法处理 (丢掉, 不猜) */
        }
        bool dup = false;
        for (std::size_t k = 0; k < slot.size(); k++) {
            if (slot[k] == a) { dup = true; break; }
        }
        if (dup) {
            if (aliasDroppedOut != nullptr) { (*aliasDroppedOut)++; }
            continue;
        }
        if (slot.empty() || qLegal[i] > mx) { mx = qLegal[i]; }
        slot.push_back(a);
        q.push_back(qLegal[i]);
    }
    if (slot.empty()) {
        return false;
    }

    /* ---- π = softmax(Q / T) (T = 1.0; 非法列保持 0) ---- */
    double sum = 0.0;
    for (std::size_t k = 0; k < slot.size(); k++) {
        const double e = std::exp((q[k] - mx) / DQN_BC_TEMP);
        pi[(std::size_t)slot[k]] = (float)e;
        sum += e;
    }
    if (!(sum > 1e-12) || !std::isfinite(sum)) {
        return false;                        /* 极端 logit 溢出: 这条样本不参与 */
    }
    for (std::size_t k = 0; k < slot.size(); k++) {
        pi[(std::size_t)slot[k]] = (float)((double)pi[(std::size_t)slot[k]] / sum);
    }
    return true;
}

/*
 * 一条样本的读数累计 —— 两支共用 (口径必须逐字相同, 否则"两支可比"这句话就不成立)。
 * 与 bcagent.hpp 里 SAC 那一支的读数**逐项对应** (P(老师) / CE / 熵 / top-1 / H(t)),
 * 多出来的只有 `Σπ 守卫`: 合法槽位上的概率和必须 = 1 (它正是 "Σ(π−t)=0" 的前提)。
 */
inline void accumulateMetrics(const RL::BCSample &s, int a, const RL::Tensor &pi,
                              const RL::Tensor &mask, const RL::Tensor &target,
                              std::size_t actionDim, Metrics &m)
{
    double mass = 0.0;
    for (std::size_t i = 0; i < actionDim; i++) {
        if (i < mask.size() && mask[i] > 0.5f) { mass += (double)pi[i]; }
        if (pi[i] > 0.0f) { m.entropy -= (double)pi[i] * std::log((double)pi[i]); }
    }
    if (std::fabs(mass - 1.0) > 1e-4) {
        Dqn::piMassWarn++;                   /* 去重/归一化被改坏时的唯一症状 */
    }
    m.pTeacher += (double)pi[(std::size_t)a];
    m.ce += RL::maskedCrossEntropy(pi, target);
    int best = -1;
    if (RL::argmaxOnLegal(pi, mask, best) && best == a) { m.top1Hits++; }
    m.targetEntropy += targetEntropyOf(s);
    if (s.targetIdx.size() > 1) { m.softSamples++; }
    m.n++;
}

/* 一个网络的**权重逐位校验和** (FNV-1a 64; 只覆盖 iFcLayer 的 w/b —— 本工程的头只有这种层) */
inline unsigned long long weightHash(RL::Net &net)
{
    unsigned long long h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
        if (fc == nullptr) {
            continue;                        /* 非全连接层 (MoE 块等): 不进这份校验和 */
        }
        const float *wb[2] = { fc->w.data(), fc->b.data() };
        const std::size_t wn[2] = { fc->w.size(), fc->b.size() };
        for (int t = 0; t < 2; t++) {
            for (std::size_t k = 0; k < wn[t]; k++) {
                unsigned int bits = 0;
                std::memcpy(&bits, wb[t] + k, sizeof(bits));
                h ^= (unsigned long long)bits;
                h *= 1099511628211ULL;
            }
        }
    }
    return h;
}

/* 读数的收尾 (批平均) —— 与 bcagent.hpp 的两支逐字同口径 (含 H(t) 必须除样本数那条) */
inline void finishMetrics(Metrics &m)
{
    if (m.n > 0) {
        m.top1Pct = 100.0 * (double)m.top1Hits / (double)m.n;
        m.pTeacher /= (double)m.n;
        m.ce /= (double)m.n;
        m.entropy /= (double)m.n;
        m.targetEntropy /= (double)m.n;
    }
}

/*
 * 批边界的 **MoE 批统计清理** —— 与两支 agent 自己在 learnBatch 第一句做的动作**逐字相同**
 * (它们都只是"遍历骨干里的 MoE 层, 逐个 `resetBatchStats()`")。
 *
 * 为什么本层自己写这一份 (而不是调 agent 的方法):
 *   * `DQNABAgent::resetMoeBatchStats()` 是**私有**的 (只有它的 learnBatch/applyGradients 用),
 *     而 `DQNMCTSMOETbAgent::resetMoeBatchStats()` 是公开的 —— 若"能调的调、不能调的跳过",
 *     两支就会走两条不同的路径, 而读数要可比就必须同一条;
 *   * 这一份只碰**公开**层接口 (`RL::Net` + `ISparseMoE`), 不需要给 agent 加任何入口。
 *
 * ⚠ 它的作用范围 (不许夸大): `usageBatch` 只被"负载均衡辅助损失"(本路径**不调**)与
 *   诊断面板读 —— **不进任何梯度**。所以这一句是"让诊断计数不被整段 BC 稀释", 不是
 *   正确性的一部分 (缺了它读数不会错, 只是那两行诊断会变成跨整段运行的平均)。
 */
inline void resetMoeBatchStatsOf(RL::Net &trunk)
{
    for (std::size_t i = 0; i < trunk.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE *>(trunk[i]);
        if (m != nullptr) {
            m->resetBatchStats();
        }
    }
}

/* ============================================================================
 *  2) DQN + AB (dueling 双头) —— evaluate / update 的精确重载
 * ============================================================================ */

/*
 * 留出/训练集会走的那条路径。`evaluateNode` 是**搜索用的同一个出口**
 * (主干一次 -> V + 合法列的 A -> Q = V + A − mean_legal(A)), 所以读数与 agent 自己
 * "看到"的 Q 是同一份 —— 不在这里重拼一遍 Q 的公式 (重拼就会有两份会漂移的口径)。
 */
inline void evaluate(DQNABAgent &ag, const std::vector<RL::BCSample> &set, Metrics &m)
{
    m = Metrics();
    RL::Tensor mask, target;
    RL::Tensor pi((std::size_t)DQNABAgent::ACTION_DIM, 1);
    std::vector<double> q;
    double v = 0.0;
    for (std::size_t k = 0; k < set.size(); k++) {
        const RL::BCSample &s = set[k];
        const int a = s.argmaxTarget();
        if (!s.valid() || a < 0) { m.skipped++; continue; }
        /* 形状契约在边界查一次 (见 bcagent.hpp 那两条注释: Release 下不查就是越界读) */
        if (s.state.size() != (std::size_t)DQNABAgent::STATE_DIM) { m.skipped++; continue; }
        if (!RL::denseMaskAndTarget(DQNABAgent::ACTION_DIM, s, mask, target)) {
            m.skipped++; continue;
        }
        if (a >= DQNABAgent::ACTION_DIM || mask[(std::size_t)a] <= 0.5f) { m.skipped++; continue; }

        /* 前向: 与搜索同一套 (inference=true = 不保留缓存; 恒用在线网, 见文件头) */
        ag.evaluateNode(s.state, s.legalIdx, v, q, true);
        if (q.size() != s.legalIdx.size()) { m.skipped++; continue; }
        int aliasDropped = 0;
        if (!policyFromQLegal((std::size_t)DQNABAgent::ACTION_DIM, s.legalIdx, q, pi,
                              &aliasDropped)) {
            m.skipped++; continue;
        }
        Dqn::aliasDropped += aliasDropped;   /* DQNAB 是双射 ⇒ 这一项恒为 0 (守卫) */
        Dqn::samples++;
        accumulateMetrics(s, a, pi, mask, target, (std::size_t)DQNABAgent::ACTION_DIM, m);
    }
    finishMetrics(m);
}

/*
 * 一次批更新。顺序与 accumulateGrad 逐条相同, 差别只有两处 (都是口径决定的):
 *   * `dz = π − t` 代替 DQN 的 `dQ·(1 − 1/n)` 那一组 (监督目标从 TD 换成老师分布);
 *   * **不反向 V 头** (它的梯度恒为 0, 见文件头的推导) —— 于是 V 头权重逐批不变,
 *     这件事每次都由 `weightHash` 真的验一遍。
 */
inline UpdateStat update(DQNABAgent &ag, const std::vector<RL::BCSample> &batch, float lr)
{
    UpdateStat st;
    RL::Tensor mask, target;
    RL::Tensor pi((std::size_t)DQNABAgent::ACTION_DIM, 1);
    RL::Tensor dz((std::size_t)DQNABAgent::ACTION_DIM, 1);
    std::vector<double> q;
    double v = 0.0;
    double ceSum = 0.0;
    int n = 0;

    /* 批边界: 清 MoE 批统计 (与 learnBatch 同序; 理由见文件头"前向的副作用") */
    resetMoeBatchStatsOf(ag.m_trunk);
    /* V 头守卫 (批前): 见文件头 —— 这一批跑完它必须逐位不变 */
    const unsigned long long vBefore = weightHash(ag.m_vHead);

    for (std::size_t k = 0; k < batch.size(); k++) {
        const RL::BCSample &s = batch[k];
        if (!s.valid()) { st.skipped++; continue; }
        if (s.state.size() != (std::size_t)DQNABAgent::STATE_DIM) { st.skipped++; continue; }
        const int a = s.argmaxTarget();
        if (a < 0) { st.skipped++; continue; }
        /* 目标必须落在合法集里 —— 落空时 L ≡ 0 而梯度是 π ≠ 0 (rl/bc.h 第 2 条) */
        if (!RL::denseMaskAndTarget(DQNABAgent::ACTION_DIM, s, mask, target)
            || mask[(std::size_t)a] <= 0.5f) {
            st.targetMiss++;
            continue;
        }

        /*
           前向: `inference=false` = **保留缓存**, 因为紧接着就要反向用它
           (与 accumulateGrad 的 `trunk.forward(s.state, false)` 同一个理由)。
        */
        ag.evaluateNode(s.state, s.legalIdx, v, q, false);
        if (q.size() != s.legalIdx.size()) { st.skipped++; continue; }
        int aliasDropped = 0;
        if (!policyFromQLegal((std::size_t)DQNABAgent::ACTION_DIM, s.legalIdx, q, pi,
                              &aliasDropped)) {
            st.skipped++; continue;
        }
        Dqn::aliasDropped += aliasDropped;

        /* CE 的数值必须在反向之前读 (反向会清掉层的输出缓冲) */
        ceSum += RL::maskedCrossEntropy(pi, target);
        RL::maskedCeLogitGrad(pi, mask, target, dz);

        /*
           反向: 先头后主干 (硬约束 —— 头读的是主干缓存下来的 h, 而
           `Layer<Tanh>::backward` 结束时会清掉它自己那一层的 o)。
           主干只反向**一次**: 梯度是 `m_aHead.inputGrad` (V 头不参与 ⇒ 不需要相加)。
        */
        ag.m_aHead.backward(ag.m_trunk.output(), dz);
        RL::Tensor dh = ag.m_aHead.inputGrad;
        ag.m_trunk.backward(s.state, dh);
        n++;
    }
    st.used = n;

    if (n > 0) {
        /*
           优化器: 每批只调一次, 且每个层只更新一次。
           `m_aHead` 与 `m_trunk` 是两批不同的层对象 (m_aHead 不共享主干) ⇒ 各调一次是对的;
           V 头**不调** —— 这一支的全部意义就是"BC 不动 V"。
        */
        ag.m_trunk.RMSProp(lr, 0.9f, 0.0f);
        ag.m_aHead.RMSProp(lr, 0.9f, 0.0f);
        st.ce = (float)(ceSum / (double)n);
    }

    /* V 头守卫 (批后): 不一致 = "dL/dV ≡ 0" 这条口径被破坏了, 当场停下 */
    if (weightHash(ag.m_vHead) != vBefore) {
        std::fprintf(stderr,
                     "\n[DQNAB BC] **V 头权重在本批被改动** (校验和 %llu -> %llu)。\n"
                     "  本口径的数学依据是 dueling 组合项给出的 dL/dV = Σ(π−t) = 0 ——\n"
                     "  也就是说 BC 只该动主干与 A 头。V 头被动了 ⇒ 口径已经错了,\n"
                     "  上面的 CE/top-1 读数不能再用 (去查 update() 里是不是加了 V 头的\n"
                     "  反向或 RMSProp)。\n"
                     "  见 src/bcdqnagent.hpp 文件头的推导与 weightHash 的注释。\n",
                     vBefore, weightHash(ag.m_vHead));
        std::fflush(stderr);
        std::abort();
    }
    Dqn::vHeadGuarded++;
    Dqn::samples += (long long)n;
    return st;
}

/* ============================================================================
 *  3) DQN + MCTS (稀疏 MoE + PUCT) —— evaluate / update 的精确重载
 * ============================================================================ */

inline void evaluate(DQNMCTSMOETbAgent &ag, const std::vector<RL::BCSample> &set, Metrics &m)
{
    m = Metrics();
    RL::Tensor mask, target;
    RL::Tensor pi((std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, 1);
    RL::Tensor qFull;
    std::vector<double> q;
    for (std::size_t k = 0; k < set.size(); k++) {
        const RL::BCSample &s = set[k];
        const int a = s.argmaxTarget();
        if (!s.valid() || a < 0) { m.skipped++; continue; }
        if (s.state.size() != (std::size_t)DQNMCTSMOETbAgent::STATE_DIM) { m.skipped++; continue; }
        if (!RL::denseMaskAndTarget(DQNMCTSMOETbAgent::ACTION_DIM, s, mask, target)) {
            m.skipped++; continue;
        }
        if (a >= DQNMCTSMOETbAgent::ACTION_DIM || mask[(std::size_t)a] <= 0.5f) {
            m.skipped++; continue;
        }

        /* 前向: 全量 Q (长度 = ACTION_DIM) —— 与搜索/训练同一个出口 qValuesFull */
        if (!ag.qValuesFull(s.state, qFull, nullptr)) { m.skipped++; continue; }
        if (qFull.size() != (std::size_t)DQNMCTSMOETbAgent::ACTION_DIM) { m.skipped++; continue; }
        q.assign(s.legalIdx.size(), 0.0);
        for (std::size_t i = 0; i < s.legalIdx.size(); i++) {
            const int c = s.legalIdx[i];
            if (c >= 0 && c < DQNMCTSMOETbAgent::ACTION_DIM) { q[i] = (double)qFull[(std::size_t)c]; }
        }
        int aliasDropped = 0;
        if (!policyFromQLegal((std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, s.legalIdx, q, pi,
                              &aliasDropped)) {
            m.skipped++; continue;
        }
        Dqn::aliasDropped += aliasDropped;   /* 128 槽哈希的别名: 会 > 0, 是常态 */
        Dqn::samples++;
        accumulateMetrics(s, a, pi, mask, target,
                          (std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, m);
    }
    finishMetrics(m);
}

inline UpdateStat update(DQNMCTSMOETbAgent &ag, const std::vector<RL::BCSample> &batch,
                         float lr)
{
    UpdateStat st;
    RL::Tensor mask, target;
    RL::Tensor pi((std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, 1);
    RL::Tensor dz((std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, 1);
    RL::Tensor qFull;
    std::vector<double> q;
    double ceSum = 0.0;
    int n = 0;

    resetMoeBatchStatsOf(ag.trunk);   /* 批边界清 MoE 统计 (与 learnBatch 同序) */

    for (std::size_t k = 0; k < batch.size(); k++) {
        const RL::BCSample &s = batch[k];
        if (!s.valid()) { st.skipped++; continue; }
        if (s.state.size() != (std::size_t)DQNMCTSMOETbAgent::STATE_DIM) { st.skipped++; continue; }
        const int a = s.argmaxTarget();
        if (a < 0) { st.skipped++; continue; }
        if (!RL::denseMaskAndTarget(DQNMCTSMOETbAgent::ACTION_DIM, s, mask, target)
            || mask[(std::size_t)a] <= 0.5f) {
            st.targetMiss++;
            continue;
        }

        /*
           前向: `qValuesFull` 内部是 `trunk.forward(state)` (默认 inference=false ⇒
           保留缓存) + `qHead.forward(h)` —— 与 learnBatch 那次"当前侧"前向逐字同形,
           所以紧接着的反向拿到的就是这一次的缓存。
        */
        if (!ag.qValuesFull(s.state, qFull, nullptr)) { st.skipped++; continue; }
        if (qFull.size() != (std::size_t)DQNMCTSMOETbAgent::ACTION_DIM) { st.skipped++; continue; }
        q.assign(s.legalIdx.size(), 0.0);
        for (std::size_t i = 0; i < s.legalIdx.size(); i++) {
            const int c = s.legalIdx[i];
            if (c >= 0 && c < DQNMCTSMOETbAgent::ACTION_DIM) { q[i] = (double)qFull[(std::size_t)c]; }
        }
        int aliasDropped = 0;
        if (!policyFromQLegal((std::size_t)DQNMCTSMOETbAgent::ACTION_DIM, s.legalIdx, q, pi,
                              &aliasDropped)) {
            st.skipped++; continue;
        }
        Dqn::aliasDropped += aliasDropped;

        ceSum += RL::maskedCrossEntropy(pi, target);
        RL::maskedCeLogitGrad(pi, mask, target, dz);

        /* 反向: 先头后骨干 (与 learnBatch 的 1481-1500 行同序、同一个理由) */
        ag.qHead.backward(ag.trunk.output(), dz);
        RL::Tensor gh = ag.qHead.inputGrad;
        ag.trunk.backward(s.state, gh);
        n++;
    }
    st.used = n;

    if (n > 0) {
        /* 每个层只更新一次: trunk + qHead。**不要**再调 qNet (视图 ⇒ 学习率翻倍) */
        ag.trunk.RMSProp(lr, 0.9f, 0.0f);
        ag.qHead.RMSProp(lr, 0.9f, 0.0f);
        st.ce = (float)(ceSum / (double)n);
    }
    Dqn::samples += (long long)n;
    return st;
}

} // namespace BC

#endif // BCDQNAGENT_HPP
