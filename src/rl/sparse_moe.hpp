#ifndef RL_SPARSE_MOE_HPP
#define RL_SPARSE_MOE_HPP

/*
 * sparse_moe.hpp — 稀疏路由的 Mixture-of-Experts (本工程自己实现, 不是上游 snakeAI 的)
 * ================================================================================
 *
 * 为什么要另写一个: 上游 `rl/moe.hpp` 的 `MOE::forward` 是**稠密**的 —— 它循环调用
 * 全部 NumExperts 个专家的前向再做门控加权求和 (`if (gi > 1e-8f)` 只跳过累加, 不跳过
 * 计算)。于是"专家数"直接乘在计算量上, 在 d_model=1260 这种输入上完全不可用
 * (实测: 16 个 TransformerBlock 专家 = 152 ms/前向, 见 docs/agents_design.md §11.4)。
 *
 * 稀疏 MoE 的关键只有一条: **只计算被门控选中的 top-k 个专家**。于是
 *   * 参数量  ~ E × (每个专家)
 *   * 计算量  ~ k × (每个专家)      ← 与 E 无关
 * 这就是 MoE 唯一真正的卖点(容量不按算力付费), 也是本文件存在的理由。
 *
 * 数学 (与 `moe.hpp` 保持同一套约定, 便于交叉验证):
 *   gate      = softmax(Wg·x + b)                      (E×1)
 *   S         = top-k(gate)                            (k 个下标)
 *   o         = Σ_{i∈S} gate[i] · expert_i(x)          (d_model×1)
 *
 * 反向:
 *   dL/d(expert_out_i) = gate[i]·e            (只对 i∈S)
 *   dL/d(gate_i)       = e · expert_out_i     (只对 i∈S, 其余为 0)
 *   dL/dz              = Jᵀ_softmax · dL/dgate
 *   dL/dx              = Σ_{i∈S} expert_i.backward(gate[i]·e) + Wgᵀ·dL/dz
 *
 * 注意 `dL/d(gate_i) = 0` (i∉S) 并不代表那些专家"没有梯度": 经 softmax 的雅可比之后
 * 它们会被**压低** (dL/dz_c = g_c(d_c − Σ_i d_i g_i), 而 Σ d_i g_i > 0)。这正是稀疏 MoE
 * 会**专家坍缩**的原因, 所以必须配一个负载均衡辅助损失 —— 见 `addAuxGradient()`。
 *
 * 坍缩诊断: `usageTotal()` 累计每个专家被选中的次数。如果它严重偏斜(少数专家吃掉
 * 绝大多数), 就说明路由塌了。
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "activate.h"
#include "expert.hpp"
#include "ilayer.h"
#include "layer.h"
#include "optimize.h"
#include "transformer.hpp"
#include "util.hpp"

namespace RL {

/*
 * 专家 (MlpExpert / TransformerBlock / Layer<Fn>) 与它们的工厂、初始化缩放现在
 * 统一放在 `rl/expert.hpp` —— 因为 `moe.hpp` 与 `concat.hpp` 的 ScaledConcat 都要
 * 按模板参数接受多种专家, 三份拷贝迟早漂移。这里的用法与语义一个字都没变。
 */

/* ============================================================
 *  门控变体 (2026-09, bench_gate_moe 的受控实验用)
 * ============================================================
 *
 *  背景 (实测): SAC+AZ-MoE-TB 那条骨干是 `SparseMoE<TB,4,1>`, 报告出来的专家使用
 *  直方图严重偏斜 —— 随机初始化 `[94,102,267,988]` (max/mean 2.72), 训练 89 步后
 *  `{598,353,5275,70}` (最弱专家只剩 1.1%)。文档记的两难是: coef=0 时 8 个专家饿死
 *  3 个, coef>=0.5 又开始盖过真正的梯度。本节的三个旋钮就是为"能不能治好它"做的
 *  受控对照, **默认值一律等于改动前的行为**。
 *
 *  1. 门控结构 `GateStructure`: Linear (现状, 一层仿射) / Mlp (加一个隐层)。
 *     线性门控只能按 4 个超平面切状态空间; MLP 能表达"子力 且 位置 且 被将军"这类
 *     组合判据。代价: wg 形状变化 -> 旧权重文件被 Net::load 的参数量守卫拒绝。
 *
 *  2. 门控激活 `GateActivation`: Softmax (现状) / Logistic。
 *     softmax 的概率质量守恒 -> 抬高一个专家必然压低其余 ("零和竞争"), 这正是路由
 *     坍缩的结构性来源; logistic 给每个专家独立分数, 取消该约束。
 *     ⚠ **不要用 RL::Sigmoid**: 它是 `1/(1+exp(-1.702x))` (GELU 近似用的"硬 sigmoid",
 *     斜率 1.702), 比标准 logistic 陡得多、更容易饱和, 会让本实验与文献不可比。
 *     所以这里单独实现标准 logistic (见 logisticF / dLogistic)。
 *
 *  3. 无辅助损失的偏置均衡 `lossFreeBias`: 给每个专家一个 b_i, **只进 top-k 选择、
 *     不进输出权重**; 每批之后按批次负载更新 b_i += u*sign(mean - usage_i)
 *     (Loss-Free Balancing, arXiv:2408.15664 Algorithm 1)。因为它完全走
 *     argmax 而不是损失, **梯度路径一个字节都不变** —— 这是它与"调大 auxLossCoef"
 *     的本质区别 (后者会往主目标里注入干扰梯度)。
 */
enum class GateStructure { Linear = 0, Mlp = 1 };
enum class GateActivation { Softmax = 0, Logistic = 1 };

/*
 * 标准 logistic (不是 RL::Sigmoid 的 1.702 斜率版本)。
 * 分段写法是为了避免 exp 溢出。
 */
inline float logisticF(float z)
{
    if (z >= 0.0f) {
        const float e = std::exp(-z);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(z);
    return e / (1.0f + e);
}
/* dσ/dz = σ(1-σ), 用输出值算 */
inline float dLogistic(float y) { return y * (1.0f - y); }

/* ============================================================
 *  MoERouteProbe —— "此刻哪个专家在工作"的**无锁**实时读数 (2026-10)
 * ============================================================
 *
 *  与前两个读数的分工 (三个都在, 回答三个不同的问题):
 *    * `usageSnapshot`      = 生命周期累计        —— 路由坍缩诊断
 *    * `usageSnapshotSplit` = 训练侧/推理侧分别累计 —— 均衡机制有没有生效
 *    * **本探针**           = "**现在**是谁在工作" —— 界面上的呼吸灯靠它
 *
 *  为什么必须无锁 (这不是"顺手优化", 是唯一可行的路径):
 *    AI 的**整段决策都持有 `ChessBoard::m_agentMutex`** (理由见 chessboard.cpp 的
 *    aiThinkForAgentRaw: `env` 是所有 agent 共用的试走棋盘, 决策期间不能有第二个
 *    读者)。所以任何"在锁上读"的界面路径在**思考中会一直阻塞** —— 而"思考中"恰好
 *    是唯一想看实时路由的时刻。GUI 线程读一个原子快照既不需要那把锁, 也不会拖慢搜索。
 *
 *  一致性做法 (两件事分开, 因为容错要求不同):
 *    1. **最近一次前向**: seqlock (序号奇数 = 写入中)。读侧宁可**这一帧不画**, 也不要
 *       画出半条路由 —— 选中下标与门控权重不同源的话, 会显示成"另一个专家在工作",
 *       而这类错读数正是本工程反复栽的坑。
 *    2. **热度** (呼吸灯强度): 每个专家一个 `atomic<float>`, 按**时间**指数衰减
 *       (τ = 1 s)。为什么按时间而不是按前向次数: 训练侧/搜索侧/不同骨干的前向频率
 *       差几个数量级 (TB 专家 5 ms/次 vs MLP 专家 0.139 ms/次 vs 训练批), 按次数衰减
 *       的话同一个 τ 在一支上是 0.1 s、在另一支上是 10 s。读侧再按"距上次发布过了多久"
 *       补一次衰减, 于是**停止计算之后灯会自己灭** (而不是永远停在最后一次的亮度)。
 *
 *  代价: 每次 forward 多 n+E 次 relaxed 原子写 + 一次 `steady_clock::now()` (实测
 *  ~25 ns), 相对最便宜的专家前向 (MlpExpert 0.139 ms) 是 0.02%。
 *  **不参与任何数值计算**: 探针的所有成员都不进 `paramCount`、不进权重文件、不参与
 *  梯度, 所以"默认路径逐位不变"这条约束不受影响。
 */
inline double moeProbeNowSec()
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

class MoERouteProbe
{
public:
    /* 本工程最大 E=8 / top-2; 再大也不是"不许", 只是超过的部分不进快照 (见 publish) */
    static constexpr int   kMaxExperts = 32;
    static constexpr int   kMaxPicked  = 32;
    static constexpr float kTauSec     = 1.0f;   /* 热度的指数衰减时间常数 */

    /* 最近一次前向的路由 (seqlock 保护) */
    struct Snapshot {
        bool     ok = false;          /* false = 没读到稳定快照 (或无前向) */
        unsigned serial = 0;          /* 前向序号, 每次 forward +1 —— 界面据此判断"是新的" */
        int      experts = 0;
        int      picked = 0;
        int      idx[kMaxPicked];
        float    w[kMaxPicked];       /* 对应的门控概率 (输出加权用的那个, 不是偏置后的) */
        bool     dense = false;       /* TopK >= E: 全算 (等算力对照) */
    };

    /* 最近 ~τ 秒里每个专家干活的强度 (呼吸灯用) */
    struct Heat {
        bool   ok = false;
        int    experts = 0;
        float  v[kMaxExperts];
        float  total = 0.0f;          /* Σv —— 读侧归一化用 */
        double ageSec = -1.0;         /* 距上一次前向过了多久 (<0 = 从来没过) */
    };

    MoERouteProbe()
    {
        for (int i = 0; i < kMaxExperts; i++) {
            heat_[i].store(0.0f, std::memory_order_relaxed);
        }
        for (int i = 0; i < kMaxPicked; i++) {
            idx_[i].store(-1, std::memory_order_relaxed);
            w_[i].store(0.0f, std::memory_order_relaxed);
        }
        experts_.store(0, std::memory_order_relaxed);
        n_.store(0, std::memory_order_relaxed);
        dense_.store(0, std::memory_order_relaxed);
        serial_.store(0u, std::memory_order_relaxed);
        seq_.store(0u, std::memory_order_relaxed);
        lastSec_.store(0.0, std::memory_order_relaxed);
    }

    void setExperts(int e)
    {
        if (e < 0) { e = 0; }
        if (e > kMaxExperts) { e = kMaxExperts; }
        experts_.store(e, std::memory_order_relaxed);
    }

    /* 一次前向结束后调用 (由 SparseMoE::forward 发布) */
    void publish(const int *idx, const float *w, int n, bool dense)
    {
        if (n < 0) { n = 0; }
        if (n > kMaxPicked) { n = kMaxPicked; }

        /* ---- 1. 瞬时快照 (seqlock: 写前奇数, 写后偶数) ---- */
        const unsigned s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        n_.store(n, std::memory_order_relaxed);
        dense_.store(dense ? 1 : 0, std::memory_order_relaxed);
        for (int k = 0; k < n; k++) {
            idx_[k].store(idx[k], std::memory_order_relaxed);
            w_[k].store(w[k], std::memory_order_relaxed);
        }
        serial_.store(s / 2u + 1u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        seq_.store(s + 2u, std::memory_order_relaxed);

        /* ---- 2. 热度: 先按流逝时间衰减全部, 再给本次选中的 +1 ---- */
        const double now = moeProbeNowSec();
        const double last = lastSec_.load(std::memory_order_relaxed);
        float decay = 1.0f;
        if (last > 0.0) {
            decay = (float)std::exp(-(now - last) / (double)kTauSec);
            if (!(decay >= 0.0f) || decay > 1.0f) { decay = 1.0f; }
        }
        lastSec_.store(now, std::memory_order_relaxed);
        const int e = experts_.load(std::memory_order_relaxed);
        for (int i = 0; i < e; i++) {
            heat_[i].store(heat_[i].load(std::memory_order_relaxed) * decay,
                           std::memory_order_relaxed);
        }
        for (int k = 0; k < n; k++) {
            const int e2 = idx[k];
            if (e2 >= 0 && e2 < kMaxExperts) {
                heat_[e2].store(heat_[e2].load(std::memory_order_relaxed) + 1.0f,
                                std::memory_order_relaxed);
            }
        }
    }

    /* 读"最近一次前向"。失败 = 正在写 (或从来没有前向) —— 调用方跳过这一帧即可 */
    bool read(Snapshot &out) const
    {
        for (int attempt = 0; attempt < 4; attempt++) {
            const unsigned s1 = seq_.load(std::memory_order_acquire);
            if ((s1 & 1u) != 0u) {
                continue;                       /* 正在写 */
            }
            out.experts = experts_.load(std::memory_order_relaxed);
            out.picked = n_.load(std::memory_order_relaxed);
            out.dense = dense_.load(std::memory_order_relaxed) != 0;
            out.serial = serial_.load(std::memory_order_relaxed);
            for (int k = 0; k < kMaxPicked; k++) {
                out.idx[k] = idx_[k].load(std::memory_order_relaxed);
                out.w[k] = w_[k].load(std::memory_order_relaxed);
            }
            const unsigned s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) {
                out.ok = (out.serial > 0u) && (out.experts > 0);
                return out.ok;
            }
        }
        out.ok = false;
        return false;
    }

    /* 读"最近 τ 秒谁在干活" (补一次衰减 -> 停算之后会自己灭) */
    void readHeat(Heat &out) const
    {
        out.experts = experts_.load(std::memory_order_relaxed);
        const double now = moeProbeNowSec();
        const double last = lastSec_.load(std::memory_order_relaxed);
        float decay = 1.0f;
        if (last > 0.0) {
            const double age = now - last;
            out.ageSec = age;
            decay = (float)std::exp(-age / (double)kTauSec);
            if (!(decay >= 0.0f) || decay > 1.0f) { decay = 1.0f; }
        } else {
            out.ageSec = -1.0;
        }
        out.total = 0.0f;
        for (int i = 0; i < kMaxExperts; i++) {
            float v = 0.0f;
            if (i < out.experts) {
                v = heat_[i].load(std::memory_order_relaxed) * decay;
                if (!(v > 0.0f)) { v = 0.0f; }   /* 顺手挡掉 NaN */
            }
            out.v[i] = v;
            out.total += v;
        }
        out.ok = (out.experts > 0) && (out.total > 0.0f);
    }

    /* 复位 (与 resetUsage 同一个语义: 只清读数, 不动权重) */
    void clear()
    {
        const unsigned s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        n_.store(0, std::memory_order_relaxed);
        serial_.store(0u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        seq_.store(s + 2u, std::memory_order_relaxed);
        for (int i = 0; i < kMaxExperts; i++) {
            heat_[i].store(0.0f, std::memory_order_relaxed);
        }
        lastSec_.store(0.0, std::memory_order_relaxed);
    }

private:
    std::atomic<unsigned> seq_;        /* seqlock: 偶 = 稳定, 奇 = 写入中 */
    std::atomic<int>      experts_;
    std::atomic<int>      n_;
    std::atomic<int>      dense_;
    std::atomic<int>      idx_[kMaxPicked];
    std::atomic<float>    w_[kMaxPicked];
    std::atomic<unsigned> serial_;
    std::atomic<float>    heat_[kMaxExperts];
    std::atomic<double>   lastSec_;    /* 上一次前向的时刻 (steady_clock 秒) */
};

/* ============================================================
 *  ISparseMoE — 非模板接口, 让上层可以 dynamic_cast 到"任何"稀疏 MoE 层
 *  (模板参数不同 -> 类型不同, 需要一个共同基类才能遍历网络找出来)
 * ============================================================ */
class ISparseMoE : public iLayer
{
public:
    /* 注入负载均衡辅助损失的梯度 (在 mini-batch 结束后调用一次) */
    virtual void addAuxGradient(float coef) = 0;

    /* ---- 门控变体配置 (非纯虚: 其它实现者不受影响) ---- */
    virtual void setGateStructure(GateStructure) {}
    virtual void setGateActivation(GateActivation) {}
    /*
       [2026-10] 装一个带隐层的 MLP 门控 (d_model -> hidden -> NumExperts)。
       **必须在任何 forward 之前调用** —— 它会重建并重新初始化门控权重,
       构造函数之后、训练之前调用是安全的 (那时还没有 golden 权重要保护)。
    */
    virtual void enableMlpGate(int) {}
    virtual GateActivation gateActivation() const { return GateActivation::Softmax; }
    virtual GateStructure gateStructure() const { return GateStructure::Linear; }
    virtual void setLossFreeBias(bool, float) {}
    virtual bool lossFreeBiasEnabled() const { return false; }
    /* 读一份偏置快照 (诊断) */
    virtual void lossFreeBiasSnapshot(std::vector<float> &out) const { out.clear(); }
    /*
        按"自上次复位以来的批次负载"更新偏置。**不产生任何梯度**, 必须在
        addAuxGradient 之前 (或代替它) 调用, 且要在批统计被清零之前调用。
    */
    virtual void updateLossFreeBias() {}
    /*
       只清掉"本批"的门控统计 (usageBatch / probSumBatch / xSum / batchForwardCount),
       保留 usageTotal (生命周期累计, 坍缩诊断要用)。

       为什么需要它: addAuxGradient 是按"自上次调用以来所有 forward"的均值算的。但
       forward 有两种来源 —— 训练时的 forward, 和推理时(MCTS 展开/叶子估值)的 forward。
       PPO::trainStep 是"一条样本一次更新", 两次 trainStep 之间可能已经跑了整局棋的
       MCTS 估值, 那些推理 forward 会混进辅助损失的批统计里 (既有语义上的混淆, 也有
       xSum 这种 float 累加器在几十万次累加后的精度损失)。trainStep 开头调用本函数,
       辅助损失就严格只反映本次训练前向。
    */
    virtual void resetBatchStats() = 0;
    /* 每个专家累计被选中次数 (坍缩诊断) */
    virtual void usageSnapshot(std::vector<long long> &out) const = 0;
    /*
       按**前向来源**分开的计数 (2026-10)。
       为什么需要它: `usageSnapshot` 读的是 `usageTotal`, 它把"训练批的前向"和
       "MCTS 推理/叶子估值的前向"混在一起 —— 而后者在数量上往往压倒前者
       (每步 sims 次搜索 × 每次估值的多次前向, 对比 learnSteps × batch)。于是报告
       出来的直方图**主要反映搜索访问到的局面分布**, 用它判断"辅助损失/偏置有没有把
       训练批的路由掰平"是不成立的。

       ⚠ **不能靠 forward 的 `inference` 参数来分**: 本工程几个 SAC 骨干的搜索路径
       调的是 `trunk.forward(state)` —— `inference` 走默认值 false。实测过一次:
       那种写法下"推理侧"恒为 0, 而"训练侧"里混着全部搜索前向。
       正确口径是**批边界**: agent 在 `learnBatch` 的训练循环之后、清批统计之前调
       `accumulateTrainBatch()`, 它把本批的 `usageBatch` 累加进训练侧。
       于是 推理侧 = usageTotal − 训练侧。
    */
    virtual void usageSnapshotSplit(std::vector<long long> &trainOut,
                                    std::vector<long long> &inferOut) const
    {
        trainOut.clear();
        inferOut.clear();
    }
    /* 把"本批的训练前向"计入训练侧累计 (不清批统计) */
    virtual void accumulateTrainBatch() {}

    virtual void resetUsage() = 0;
    virtual int expertCount() const = 0;
    virtual int topK() const = 0;

    /*
       [2026-10] "此刻哪个专家在工作"的**无锁**探针 (界面呼吸灯的数据源)。
       返回 nullptr = 这个实现没有探针 (默认实现, 其它 iLayer 不受影响)。

       **指针在层对象的生命周期内稳定** —— 界面正是靠这一点: 它只在持锁的路径上
       取一次指针, 之后在**不持锁**的情况下反复读 (思考中整段决策都持着
       `m_agentMutex`, 详见 MoERouteProbe 的说明)。
    */
    virtual const MoERouteProbe *routeProbe() const { return nullptr; }
};

/* 专家初始化缩放 / 工厂: 见 rl/expert.hpp (scaleExpertInit / ExpertFactory) */

/* ============================================================
 *  SparseMoE<Expert, NumExperts, TopK>
 *
 *  TopK >= NumExperts 时退化成"稠密 MoE" —— 这是给"等算力对照"用的:
 *  同样的专家、同样的门控, 只是全算一遍。
 * ============================================================ */
template<typename Expert, int NumExperts, int TopK>
class SparseMoE : public ISparseMoE
{
public:
    int d_model;
    Tensor wg;                 /* 门控权重 (NumExperts × d_model) */
    Tensor bg;                 /* 门控偏置 (NumExperts × 1) */
    Tensor gate;               /* 最近一次的门控概率 (NumExperts × 1) */

    Expert experts[NumExperts];

    Tensor expert_out[NumExperts];   /* 只对被选中的专家有效 (缓存供反向) */
    Tensor xLast;                    /* 最近一次的输入 (辅助损失梯度要用) */
    Tensor xSum;                     /* 一个 mini-batch 内输入的和 (辅助损失用批均值) */

    Tensor g_wg;
    Tensor g_bg;
    Tensor v_wg;
    Tensor v_bg;
    Tensor m_wg;
    Tensor m_bg;

    /* ============================================================
       门控变体 (2026-09 受控实验)
       ------------------------------------------------------------
       默认值 = **与改动前逐位相同**的"线性 + softmax + 无偏置"。
       MLP 门控的张量只在 `enableMlpGate()` 里构造 —— 默认模式下**不消耗任何随机数**,
       所以既有的 golden / 权重文件不受影响 (同 HonorHeads 那次的处理方式)。
       ============================================================ */
    GateStructure  gateStruct;
    GateActivation gateAct;
    bool  withGradFlag;
    int   gateHidden;

    Tensor wg1, bg1;      /* d_model     -> gateHidden   (只在 Mlp 模式分配) */
    Tensor wg2, bg2;      /* gateHidden  -> NumExperts   (只在 Mlp 模式分配) */
    Tensor hGatePre;      /* 隐层 pre-activation 缓存 (反向要 tanh') */
    Tensor hGateAct;      /* 隐层 tanh 输出缓存 */
    Tensor g_wg1, g_bg1, g_wg2, g_bg2;
    Tensor v_wg1, v_bg1, v_wg2, v_bg2;
    Tensor m_wg1, m_bg1, m_wg2, m_bg2;

    /*
       无辅助损失偏置均衡 (Loss-Free Balancing, arXiv:2408.15664)。
       **非参数缓冲**: 不进 paramCount、不进权重文件、不接收梯度。
       b_i 只参与 top-k 的选择比较, 不进输出加权 —— 这正是"没有干扰梯度"的原因。
    */
    Tensor biasGate;
    bool   biasEnabled;
    float  biasRate;

    int selected[NumExperts];        /* 本次前向选中的专家下标 */
    int nSelected;

    /*
       [2026-10] 实时路由探针 (界面"哪个专家在工作"的呼吸灯)。
       每次 forward 末尾发布一次, 全是原子写 —— 读侧可以在思考中(整段决策持锁)
       无锁读到最新路由。**不参与任何数值计算**, 见 MoERouteProbe 的说明。
       名字带 `live` 是为了与下面的访问器 `routeProbe()` 区分开 (同名成员+方法会
       直接编译不过, 这是实测撞到的第一版错误)。
    */
    MoERouteProbe liveProbe;

    /* 统计 */
    long long usageTotal[NumExperts];   /* 全生命周期: 坍缩诊断 */
    long long usageBatch[NumExperts];   /* 自上次 addAuxGradient 起 */
    /*
       [2026-10] 按前向来源拆开的计数。训练侧由 `accumulateTrainBatch()` 在批边界累计
       (= 各批 `usageBatch` 之和), 推理侧由 `usageTotal − 训练侧` 推出。
       为什么不用 forward 的 `inference` 参数: 本工程搜索路径调的是默认值 false
       (见 ISparseMoE::usageSnapshotSplit 的说明)。
    */
    long long usageTrain[NumExperts];   /* 训练批的前向 */
    double probSumBatch[NumExperts];
    long long batchForwardCount;

public:
    SparseMoE() : d_model(0), nSelected(0), batchForwardCount(0),
                  gateStruct(GateStructure::Linear),
                  gateAct(GateActivation::Softmax),
                  withGradFlag(false), gateHidden(0),
                  biasEnabled(false), biasRate(0.001f)
    {
        for (int i = 0; i < NumExperts; i++) {
            usageTotal[i] = 0;
            usageBatch[i] = 0;
            usageTrain[i] = 0;
            probSumBatch[i] = 0.0;
            selected[i] = -1;
        }
        liveProbe.setExperts(NumExperts);
    }
    SparseMoE(int d_model_, bool withGrad, int expertHidden = 0) : SparseMoE()
    {
        d_model = d_model_;
        withGradFlag = withGrad;
        type = LAYER_MOE;
        wg = Tensor((std::size_t)NumExperts, (std::size_t)d_model_);
        bg = Tensor((std::size_t)NumExperts, 1);
        Random::uniform(wg, -0.1f, 0.1f);
        Random::uniform(bg, -0.1f, 0.1f);
        gate = Tensor((std::size_t)NumExperts, 1);
        biasGate = Tensor((std::size_t)NumExperts, 1);
        biasGate.zero();

        for (int i = 0; i < NumExperts; i++) {
            experts[i] = ExpertFactory<Expert>::make(d_model_, expertHidden, withGrad);
            expert_out[i] = Tensor((std::size_t)d_model_, 1);
        }
        xLast = Tensor((std::size_t)d_model_, 1);
        xSum = Tensor((std::size_t)d_model_, 1);

        o = Tensor((std::size_t)d_model_, 1);
        e = Tensor((std::size_t)d_model_, 1);

        g_wg = Tensor((std::size_t)NumExperts, (std::size_t)d_model_);
        g_bg = Tensor((std::size_t)NumExperts, 1);
        v_wg = Tensor((std::size_t)NumExperts, (std::size_t)d_model_);
        v_bg = Tensor((std::size_t)NumExperts, 1);
        m_wg = Tensor((std::size_t)NumExperts, (std::size_t)d_model_);
        m_bg = Tensor((std::size_t)NumExperts, 1);

        for (int i = 0; i < NumExperts; i++) {
            scaleExpertInit(experts[i]);
        }
    }

    /* ============================================================
       门控变体的配置入口
       ============================================================ */

    void setGateActivation(GateActivation a) override { gateAct = a; }
    GateActivation gateActivation() const override { return gateAct; }
    GateStructure gateStructure() const override { return gateStruct; }

    /*
      开 MLP 门控: d_model -> hidden (Tanh) -> NumExperts。
      初始化的 fan-in 缩放与"线性门控那套 U(-0.1,0.1)"不同 —— 这里按
       1/sqrt(fan_in) 缩, 不然隐层 pre-activation 在 d_model=1710 上会直接饱和
       (同 scaleLayerInit 的理由)。
       **注意**: 本函数会消耗随机数, 所以只能在**非默认**模式下调用 (默认路径不碰它)。
    */
    void enableMlpGate(int hidden) override
    {
        if (hidden < 1) {
            hidden = 1;
        }
        gateStruct = GateStructure::Mlp;
        gateHidden = hidden;

        wg1 = Tensor((std::size_t)hidden, (std::size_t)d_model);
        bg1 = Tensor((std::size_t)hidden, 1);
        wg2 = Tensor((std::size_t)NumExperts, (std::size_t)hidden);
        bg2 = Tensor((std::size_t)NumExperts, 1);
        Random::uniform(wg1, -1.0f, 1.0f);
        Random::uniform(bg1, -1.0f, 1.0f);
        Random::uniform(wg2, -1.0f, 1.0f);
        Random::uniform(bg2, -1.0f, 1.0f);
        const float s1 = 1.0f / std::sqrt((float)(d_model > 0 ? d_model : 1));
        for (std::size_t k = 0; k < wg1.size(); k++) { wg1[k] *= s1; }
        const float s2 = 1.0f / std::sqrt((float)hidden);
        for (std::size_t k = 0; k < wg2.size(); k++) { wg2[k] *= s2; }
        for (std::size_t k = 0; k < bg2.size(); k++) { bg2[k] *= s2; }

        hGatePre = Tensor((std::size_t)hidden, 1);
        hGateAct = Tensor((std::size_t)hidden, 1);

        g_wg1 = Tensor((std::size_t)hidden, (std::size_t)d_model);
        g_bg1 = Tensor((std::size_t)hidden, 1);
        g_wg2 = Tensor((std::size_t)NumExperts, (std::size_t)hidden);
        g_bg2 = Tensor((std::size_t)NumExperts, 1);
        v_wg1 = Tensor((std::size_t)hidden, (std::size_t)d_model);
        v_bg1 = Tensor((std::size_t)hidden, 1);
        v_wg2 = Tensor((std::size_t)NumExperts, (std::size_t)hidden);
        v_bg2 = Tensor((std::size_t)NumExperts, 1);
        m_wg1 = Tensor((std::size_t)hidden, (std::size_t)d_model);
        m_bg1 = Tensor((std::size_t)hidden, 1);
        m_wg2 = Tensor((std::size_t)NumExperts, (std::size_t)hidden);
        m_bg2 = Tensor((std::size_t)NumExperts, 1);
    }

    void setLossFreeBias(bool on, float rate) override
    {
        biasEnabled = on;
        biasRate = (rate > 0.0f) ? rate : 0.0f;
    }
    bool lossFreeBiasEnabled() const override { return biasEnabled; }
    void lossFreeBiasSnapshot(std::vector<float> &out) const override
    {
        out.resize((std::size_t)NumExperts);
        for (int i = 0; i < NumExperts; i++) {
            out[(std::size_t)i] = biasGate[(std::size_t)i];
        }
    }

    /*
       Loss-Free Balancing 的偏置更新 (arXiv:2408.15664 Algorithm 1):
           e_i = mean_load - load_i ;  b_i += u * sign(e_i)
       负载重的专家被压低、轻的被抬高。**纯控制回路, 不产生梯度**。
       必须在批统计被清零之前调用。
    */
    void updateLossFreeBias() override
    {
        if (!biasEnabled || biasRate <= 0.0f || batchForwardCount <= 0) {
            return;
        }
        const double total = (double)(batchForwardCount * (long long)topK());
        const double meanLoad = total / (double)NumExperts;
        for (int i = 0; i < NumExperts; i++) {
            const double e = meanLoad - (double)usageBatch[i];
            const float s = (e > 0.0) ? 1.0f : ((e < 0.0) ? -1.0f : 0.0f);
            biasGate[(std::size_t)i] += biasRate * s;
        }
    }

    long long paramCount() const override
    {
        long long total = (long long)wg.size() + (long long)bg.size();
        if (gateStruct == GateStructure::Mlp) {
            total += (long long)wg1.size() + (long long)bg1.size()
                   + (long long)wg2.size() + (long long)bg2.size();
        }
        for (int i = 0; i < NumExperts; i++) {
            total += experts[i].paramCount();
        }
        return total;
    }

    /*
       iLayer 的通用自检读数: 委派给**第 0 个专家**。
       本工程的所有专家都同构 (同一个 ExpertFactory 造出来的), 所以"专家内部用了几个
       注意力头"这个问题在任何一个专家上答案都一样 —— 而它正是"MOE_TB_HEADS 被静默
       降级成 3 个头"这件事唯一能被读出来的地方。
    */
    int attnHeadsRequested() const override { return experts[0].attnHeadsRequested(); }
    int attnHeadsUsed() const override { return experts[0].attnHeadsUsed(); }
    int attnHeadDim() const override { return experts[0].attnHeadDim(); }
    int attnHeadsAllocated() const override { return experts[0].attnHeadsAllocated(); }
    long long attnElements() const override { return experts[0].attnElements(); }

    int expertCount() const override { return NumExperts; }
    int topK() const override { return (TopK < NumExperts) ? TopK : NumExperts; }
    const MoERouteProbe *routeProbe() const override { return &liveProbe; }

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        /* ---- 门控: 线性(默认) 或 MLP, 再 softmax(默认) 或标准 logistic ---- */
        gate.zero();
        if (gateStruct == GateStructure::Linear) {
            Tensor::MM::ikkj(gate, wg, x);
            gate += bg;
        } else {
            /* ⚠ MM::ikkj 是**累加**语义 (见 layer.h:94-97), 输出必须先清零 ——
               `gate` 上面已经 zero 过, 但 `hGatePre` 是另一个缓冲, 漏了它就会跨前向
               次累加、门控输出指数级发散 (实测: 有限差分当场从 6e0 变成 1e2+)。 */
            hGatePre.zero();
            Tensor::MM::ikkj(hGatePre, wg1, x);
            hGatePre += bg1;
            for (int j = 0; j < gateHidden; j++) {
                hGateAct[(std::size_t)j] = std::tanh(hGatePre[(std::size_t)j]);
            }
            Tensor::MM::ikkj(gate, wg2, hGateAct);
            gate += bg2;
        }
        if (gateAct == GateActivation::Softmax) {
            softmax(gate);
        } else {
            for (int i = 0; i < NumExperts; i++) {
                gate[(std::size_t)i] = logisticF(gate[(std::size_t)i]);
            }
        }

        /* ---- 选出 top-k (TopK >= E 时就是稠密) ----
           偏置均衡开启时, 选择比较用 gate[i] + biasGate[i], 而**输出权重仍用 gate[i]**
           (Loss-Free Balancing 的关键: 偏置只改路由, 不改加权输出)。
           关闭时走与改动前**完全相同**的那条分支, 保证默认路径逐位不变。 */
        nSelected = 0;
        if (TopK >= NumExperts) {
            for (int i = 0; i < NumExperts; i++) {
                selected[nSelected++] = i;
            }
        } else if (biasEnabled) {
            bool taken[NumExperts];
            for (int i = 0; i < NumExperts; i++) {
                taken[i] = false;
            }
            for (int k = 0; k < TopK; k++) {
                int best = -1;
                float bestV = -1e30f;
                for (int i = 0; i < NumExperts; i++) {
                    const float v = gate[(std::size_t)i] + biasGate[(std::size_t)i];
                    if (!taken[i] && v > bestV) {
                        bestV = v;
                        best = i;
                    }
                }
                if (best < 0) {
                    break;
                }
                taken[best] = true;
                selected[nSelected++] = best;
            }
        } else {
            bool taken[NumExperts];
            for (int i = 0; i < NumExperts; i++) {
                taken[i] = false;
            }
            for (int k = 0; k < TopK; k++) {
                int best = -1;
                float bestV = -1.0f;
                for (int i = 0; i < NumExperts; i++) {
                    if (!taken[i] && gate[i] > bestV) {
                        bestV = gate[i];
                        best = i;
                    }
                }
                if (best < 0) {
                    break;
                }
                taken[best] = true;
                selected[nSelected++] = best;
            }
        }

        /* ---- 只计算被选中的专家 ---- */
        o.zero();
        for (int s = 0; s < nSelected; s++) {
            const int i = selected[s];
            expert_out[i] = experts[i].forward(x, inference);
            const float gi = gate[i];
            for (int j = 0; j < d_model; j++) {
                o[j] += gi * expert_out[i][j];
            }
            usageTotal[i]++;
            usageBatch[i]++;
        }

        xLast = x;
        for (int j = 0; j < d_model; j++) {
            xSum[j] += x[j];
        }
        batchForwardCount++;
        for (int i = 0; i < NumExperts; i++) {
            probSumBatch[i] += (double)gate[i];
        }

        /*
           [2026-10] 发布实时路由 (界面呼吸灯的数据源)。
           位置刻意放在**数值全部算完之后**: 探针只写自己的原子成员, 放在这里既不影响
           任何读数, 也不给"默认路径逐位不变"留下想象空间。
           ⚠ 权重用 `gate[i]` (输出加权用的那个), **不是** `gate[i]+biasGate[i]` ——
           后者只用于 top-k 的选择比较 (Loss-Free Balancing 的口径), 拿它显示会把
           "偏置把谁抬进了 top-k"和"它对输出贡献多少"混成一件事。
        */
        {
            float wsel[MoERouteProbe::kMaxPicked];
            int np = (nSelected < MoERouteProbe::kMaxPicked) ? nSelected
                                                             : MoERouteProbe::kMaxPicked;
            for (int s = 0; s < np; s++) {
                wsel[s] = gate[(std::size_t)selected[s]];
            }
            liveProbe.publish(selected, wsel, np, TopK >= NumExperts);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* ---- 门控: dL/d(gate_i) = e·expert_out_i, 只对被选中的专家 ---- */
        Tensor d_gate((std::size_t)NumExperts, 1);
        d_gate.zero();
        for (int s = 0; s < nSelected; s++) {
            const int i = selected[s];
            float dgi = 0.0f;
            for (int j = 0; j < d_model; j++) {
                dgi += e[j] * expert_out[i][j];
            }
            d_gate[i] = dgi;
        }

        /* 经激活的雅可比: 
            softmax : dL/dz_c = g_c·(d_c − Σ_i d_i·g_i)   (耦合 —— 饿死专家仍被"抬")
            logistic: dL/dz_c = d_c·g_c·(1−g_c)          (**解耦 —— 饿死专家梯度≈0**)
           后者的饱和问题正是"换 sigmoid 就自动更均衡"不成立的原因, 见文件头注释。 */
        Tensor d_gate_logit((std::size_t)NumExperts, 1);
        if (gateAct == GateActivation::Softmax) {
            Softmax::jacobian_transpose_mul(gate, d_gate, d_gate_logit);
        } else {
            for (int i = 0; i < NumExperts; i++) {
                d_gate_logit[(std::size_t)i] =
                    d_gate[(std::size_t)i] * dLogistic(gate[(std::size_t)i]);
            }
        }

        /* ---- 门控参数梯度 (线性: 直接; MLP: 再穿一层 tanh) ---- */
        if (gateStruct == GateStructure::Linear) {
            Tensor::MM::kikj(ei, wg, d_gate_logit);
            Tensor::MM::ikjk(g_wg, d_gate_logit, x);
            g_bg += d_gate_logit;
        } else {
            /* dL/dh_act = wg2ᵀ·d_gate_logit ; dL/dh_pre = dL/dh_act·(1−tanh²) */
            Tensor d_h_act((std::size_t)gateHidden, 1);
            d_h_act.zero();
            Tensor::MM::kikj(d_h_act, wg2, d_gate_logit);
            Tensor d_h_pre((std::size_t)gateHidden, 1);
            for (int j = 0; j < gateHidden; j++) {
                const float a = hGateAct[(std::size_t)j];
                d_h_pre[(std::size_t)j] = d_h_act[(std::size_t)j] * (1.0f - a * a);
            }
            Tensor::MM::kikj(ei, wg1, d_h_pre);          /* dL/dx */
            Tensor::MM::ikjk(g_wg1, d_h_pre, x);         /* dL/dW1 */
            g_bg1 += d_h_pre;
            Tensor::MM::ikjk(g_wg2, d_gate_logit, hGateAct);   /* dL/dW2 */
            g_bg2 += d_gate_logit;
        }

        /* ---- 专家路径: 只回传被选中的 ----
           (顺序与改动前一致: 门控路径先进 ei, 专家再累加 —— 保证默认路径逐位不变) */
        for (int s = 0; s < nSelected; s++) {
            const int i = selected[s];
            const float gi = gate[i];
            for (int j = 0; j < d_model; j++) {
                experts[i].e[j] = gi * e[j];
            }
            experts[i].backward(x, ei);   /* 累加进 ei */
        }

        o.zero();
        e.zero();
        return;
    }

    /*
       负载均衡辅助损失 (Switch Transformer 形式):
           L_aux = E · Σ_i f_i · P_i
             f_i = 该专家被选中的比例 (视为常数, 不参与求导)
             P_i = 该专家的平均门控概率
       只有 P_i 可导, 所以 dL_aux/dP_i = coef·E·f_i, 再过一次 softmax 的雅可比
       变成对 logits 的梯度, 累加进 g_wg/g_bg。

       没有这一项时, softmax 的反向会把"没被选中"的专家的概率继续压低
       (dL/dz_c = g_c(d_c − Σ d_i g_i) 而 Σ d_i g_i > 0), 于是路由会迅速坍缩到
       少数专家、其余永远不训练 —— 这是稀疏 MoE 的经典失败模式。

       关于"批"的近似: P_i 与 x 都取本批的**算术平均** (P̄_i = Σ_b P_i(x_b)/n,
       x̄ = Σ_b x_b/n), 而不是只取最后一个样本。严格的批梯度是
         ∂L/∂wg = coef·E·Σ_i f_i·(1/n)Σ_b P_i(x_b)(e_i − P(x_b))·x_bᵀ
       在"批内门控分布大致相同"的平均场近似下就等于用 P̄ 和 x̄ 算出来的那一项。
       取批均值比"只看最后一个样本"方差小得多, 代价只是 forward 里一次向量加法。
       注意: 严格来说这不是精确的批梯度, 而是标准实现里常用的平均场估计。
    */
    void addAuxGradient(float coef) override
    {
        if (batchForwardCount <= 0) {
            return;
        }
        const double total = (double)(batchForwardCount * (long long)topK());
        if (total <= 0.0) {
            return;
        }
        const float inv = 1.0f / (float)batchForwardCount;

        Tensor pBar((std::size_t)NumExperts, 1);
        for (int i = 0; i < NumExperts; i++) {
            pBar[i] = (float)(probSumBatch[i] * (double)inv);
        }
        Tensor xBar((std::size_t)d_model, 1);
        for (int j = 0; j < d_model; j++) {
            xBar[j] = xSum[j] * inv;
        }

        Tensor dP((std::size_t)NumExperts, 1);
        for (int i = 0; i < NumExperts; i++) {
            const double f = (double)usageBatch[i] / total;
            dP[i] = coef * (float)NumExperts * (float)f;
        }
        Tensor dz((std::size_t)NumExperts, 1);
        /*
           激活不同 -> 雅可比不同:
             softmax : J^T·dP (耦合, 见上)
             logistic: dz_i = dP_i·P̄_i·(1−P̄_i)
           ⚠ logistic 这一支有个**结构性弱点**: 一个饿死的专家 f_i≈0 ⇒ dP_i≈0 ⇒ dz_i≈0,
           也就是"救援力恰好在最需要的时候消失"。softmax 那支因为有减项 Σd_i g_i,
           饿死专家反而会被抬起来 —— 这是本工程 coef=0.1 还能保住"0 个饿死"的原因。
           所以 sigmoid 路线必须配 lossFreeBias (无梯度控制回路), 见 updateLossFreeBias。
        */
        if (gateAct == GateActivation::Softmax) {
            Softmax::jacobian_transpose_mul(pBar, dP, dz);
        } else {
            for (int i = 0; i < NumExperts; i++) {
                dz[(std::size_t)i] = dP[(std::size_t)i] * dLogistic(pBar[(std::size_t)i]);
            }
        }
        if (gateStruct == GateStructure::Linear) {
            Tensor::MM::ikjk(g_wg, dz, xBar);
            g_bg += dz;
        } else {
            /* dL/dh_act = wg2ᵀ·dz ; dL/dh_pre = ·(1−tanh²) */
            Tensor d_h_act((std::size_t)gateHidden, 1);
            d_h_act.zero();
            Tensor::MM::kikj(d_h_act, wg2, dz);
            Tensor d_h_pre((std::size_t)gateHidden, 1);
            for (int j = 0; j < gateHidden; j++) {
                const float a = hGateAct[(std::size_t)j];
                d_h_pre[(std::size_t)j] = d_h_act[(std::size_t)j] * (1.0f - a * a);
            }
            Tensor::MM::ikjk(g_wg1, d_h_pre, xBar);
            g_bg1 += d_h_pre;
            Tensor::MM::ikjk(g_wg2, dz, hGateAct);
            g_bg2 += dz;
        }

        /* 重置 batch 统计 */
        for (int i = 0; i < NumExperts; i++) {
            usageBatch[i] = 0;
            probSumBatch[i] = 0.0;
        }
        for (int j = 0; j < d_model; j++) {
            xSum[j] = 0.0f;
        }
        batchForwardCount = 0;
    }

    void resetBatchStats() override
    {
        for (int i = 0; i < NumExperts; i++) {
            usageBatch[i] = 0;
            probSumBatch[i] = 0.0;
        }
        for (int j = 0; j < d_model; j++) {
            xSum[j] = 0.0f;
        }
        batchForwardCount = 0;
    }

    void usageSnapshot(std::vector<long long> &out) const override
    {
        out.resize((std::size_t)NumExperts);
        for (int i = 0; i < NumExperts; i++) {
            out[(std::size_t)i] = usageTotal[i];
        }
    }

    /* [2026-10] 按前向来源拆开 (训练侧 = 各批 usageBatch 之和, 推理侧 = total − 训练侧) */
    void accumulateTrainBatch() override
    {
        for (int i = 0; i < NumExperts; i++) {
            usageTrain[i] += usageBatch[i];
        }
    }

    void usageSnapshotSplit(std::vector<long long> &trainOut,
                            std::vector<long long> &inferOut) const override
    {
        trainOut.resize((std::size_t)NumExperts);
        inferOut.resize((std::size_t)NumExperts);
        for (int i = 0; i < NumExperts; i++) {
            trainOut[(std::size_t)i] = usageTrain[i];
            const long long rest = usageTotal[i] - usageTrain[i];
            inferOut[(std::size_t)i] = (rest > 0) ? rest : 0;
        }
    }

    void resetUsage() override
    {
        for (int i = 0; i < NumExperts; i++) {
            usageTotal[i] = 0;
            usageBatch[i] = 0;
            usageTrain[i] = 0;
            probSumBatch[i] = 0.0;
        }
        for (int j = 0; j < d_model; j++) {
            xSum[j] = 0.0f;
        }
        batchForwardCount = 0;
    }

    /* ---- 优化器 / 序列化 ---- */
    /* MLP 门控的张量只在 Mlp 模式下存在; 稀疏模式默认分支的调用序列**一字未改**。 */
    void SGD(float lr) override
    {
        Optimize::SGD(wg, g_wg, lr);
        Optimize::SGD(bg, g_bg, lr);
        if (gateStruct == GateStructure::Mlp) {
            Optimize::SGD(wg1, g_wg1, lr);
            Optimize::SGD(bg1, g_bg1, lr);
            Optimize::SGD(wg2, g_wg2, lr);
            Optimize::SGD(bg2, g_bg2, lr);
            g_wg1.zero(); g_bg1.zero(); g_wg2.zero(); g_bg2.zero();
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].SGD(lr);
        }
        g_wg.zero();
        g_bg.zero();
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wg, v_wg, g_wg, lr, rho, decay, clipGrad);
        Optimize::RMSProp(bg, v_bg, g_bg, lr, rho, decay, clipGrad);
        if (gateStruct == GateStructure::Mlp) {
            Optimize::RMSProp(wg1, v_wg1, g_wg1, lr, rho, decay, clipGrad);
            Optimize::RMSProp(bg1, v_bg1, g_bg1, lr, rho, decay, clipGrad);
            Optimize::RMSProp(wg2, v_wg2, g_wg2, lr, rho, decay, clipGrad);
            Optimize::RMSProp(bg2, v_bg2, g_bg2, lr, rho, decay, clipGrad);
            g_wg1.zero(); g_bg1.zero(); g_wg2.zero(); g_bg2.zero();
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g_wg.zero();
        g_bg.zero();
    }

    void Adam(float lr, float alpha, float beta, float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(wg, v_wg, m_wg, g_wg, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        Optimize::Adam(bg, v_bg, m_bg, g_bg, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        if (gateStruct == GateStructure::Mlp) {
            Optimize::Adam(wg1, v_wg1, m_wg1, g_wg1, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            Optimize::Adam(bg1, v_bg1, m_bg1, g_bg1, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            Optimize::Adam(wg2, v_wg2, m_wg2, g_wg2, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            Optimize::Adam(bg2, v_bg2, m_bg2, g_bg2, lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            g_wg1.zero(); g_bg1.zero(); g_wg2.zero(); g_bg2.zero();
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        g_wg.zero();
        g_bg.zero();
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(wg, c0, cn);
        Optimize::clamp(bg, c0, cn);
        if (gateStruct == GateStructure::Mlp) {
            Optimize::clamp(wg1, c0, cn);
            Optimize::clamp(bg1, c0, cn);
            Optimize::clamp(wg2, c0, cn);
            Optimize::clamp(bg2, c0, cn);
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].clamp(c0, cn);
        }
    }

    /*
       ⚠ 本仓库**已经因为漏掉一个 copyTo 被咬过一次** (MlpExpert: 目标网络的专家权重
       从不复制, 不报错、只让训练变差, 见 expert.hpp)。所以这里显式补上 MLP 门控那四个
       张量, 而不是指望"反正默认模式用不到"。
       `biasGate` **故意不复制**: 它是控制回路的状态而不是学出来的参数 (论文口径也是
       每批迭代更新), 且目标网络只用它做前向估值、不做路由更新。
    */
    void copyTo(iLayer* layer) override
    {
        SparseMoE *p = dynamic_cast<SparseMoE*>(layer);
        if (p == nullptr) {
            return;
        }
        p->wg = wg;
        p->bg = bg;
        if (gateStruct == GateStructure::Mlp && p->gateStruct == GateStructure::Mlp) {
            p->wg1 = wg1;
            p->bg1 = bg1;
            p->wg2 = wg2;
            p->bg2 = bg2;
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].copyTo(&p->experts[i]);
        }
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        SparseMoE *p = dynamic_cast<SparseMoE*>(layer);
        if (p == nullptr) {
            return;
        }
        Tensor tmp = wg;
        tmp *= (1.0f - alpha);
        Tensor other = p->wg;
        other *= alpha;
        tmp += other;
        p->wg = tmp;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].softUpdateTo(&p->experts[i], alpha);
        }
    }

    /*
       序列化: **默认 (线性) 模式的字节格式一字未改** —— 仍然是 wg 一行、bg 一行、
       然后专家。MLP 门控是实验模式, 它会先写一行自描述标记 `#GATE:mlp:<hidden>`,
       这样"MLP 权重喂给线性网络"(或反之) 不会静默错位 (读侧会明确拒绝)。
       `biasGate` 故意不写: 它是控制回路状态, 不是参数。
    */
    void write(std::ostream &file) override
    {
        if (gateStruct == GateStructure::Mlp) {
            file << "#GATE:mlp:" << gateHidden << std::endl;
        }
        file << wg.toString() << std::endl;
        file << bg.toString() << std::endl;
        if (gateStruct == GateStructure::Mlp) {
            file << wg1.toString() << std::endl;
            file << bg1.toString() << std::endl;
            file << wg2.toString() << std::endl;
            file << bg2.toString() << std::endl;
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].write(file);
        }
    }

    void read(std::istream &file) override
    {
        std::string s;
        std::getline(file, s);
        bool fileIsMlp = false;
        int fileHidden = 0;
        if (s.rfind("#GATE:mlp:", 0) == 0) {
            fileIsMlp = true;
            fileHidden = std::atoi(s.substr(10).c_str());
            if (gateStruct != GateStructure::Mlp) {
                std::cerr << "[SparseMoE] 拒绝: 权重文件是 MLP 门控(hidden=" << fileHidden
                          << "), 而本层是线性门控 —— 形状不同, 不静默读成别的意思\n";
                return;
            }
            std::getline(file, s);
        } else if (gateStruct == GateStructure::Mlp) {
            std::cerr << "[SparseMoE] 拒绝: 权重文件是线性门控, 而本层是 MLP 门控\n";
            return;
        }
        wg = Tensor::fromString(s);
        std::getline(file, s);
        bg = Tensor::fromString(s);
        if (fileIsMlp) {
            std::getline(file, s); wg1 = Tensor::fromString(s);
            std::getline(file, s); bg1 = Tensor::fromString(s);
            std::getline(file, s); wg2 = Tensor::fromString(s);
            std::getline(file, s); bg2 = Tensor::fromString(s);
        }
        for (int i = 0; i < NumExperts; i++) {
            experts[i].read(file);
        }
    }
};

} /* namespace RL */

#endif // RL_SPARSE_MOE_HPP
