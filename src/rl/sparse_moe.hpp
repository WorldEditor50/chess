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

#include <cmath>
#include <cstddef>
#include <fstream>
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
 *  ISparseMoE — 非模板接口, 让上层可以 dynamic_cast 到"任何"稀疏 MoE 层
 *  (模板参数不同 -> 类型不同, 需要一个共同基类才能遍历网络找出来)
 * ============================================================ */
class ISparseMoE : public iLayer
{
public:
    /* 注入负载均衡辅助损失的梯度 (在 mini-batch 结束后调用一次) */
    virtual void addAuxGradient(float coef) = 0;
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
    virtual void resetUsage() = 0;
    virtual int expertCount() const = 0;
    virtual int topK() const = 0;
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

    int selected[NumExperts];        /* 本次前向选中的专家下标 */
    int nSelected;

    /* 统计 */
    long long usageTotal[NumExperts];   /* 全生命周期: 坍缩诊断 */
    long long usageBatch[NumExperts];   /* 自上次 addAuxGradient 起 */
    double probSumBatch[NumExperts];
    long long batchForwardCount;

public:
    SparseMoE() : d_model(0), nSelected(0), batchForwardCount(0)
    {
        for (int i = 0; i < NumExperts; i++) {
            usageTotal[i] = 0;
            usageBatch[i] = 0;
            probSumBatch[i] = 0.0;
            selected[i] = -1;
        }
    }

    SparseMoE(int d_model_, bool withGrad, int expertHidden = 0) : SparseMoE()
    {
        d_model = d_model_;
        type = LAYER_MOE;
        wg = Tensor((std::size_t)NumExperts, (std::size_t)d_model_);
        bg = Tensor((std::size_t)NumExperts, 1);
        Random::uniform(wg, -0.1f, 0.1f);
        Random::uniform(bg, -0.1f, 0.1f);
        gate = Tensor((std::size_t)NumExperts, 1);

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

    long long paramCount() const override
    {
        long long total = (long long)wg.size() + (long long)bg.size();
        for (int i = 0; i < NumExperts; i++) {
            total += experts[i].paramCount();
        }
        return total;
    }

    int expertCount() const override { return NumExperts; }
    int topK() const override { return (TopK < NumExperts) ? TopK : NumExperts; }

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        /* ---- 门控 ---- */
        gate.zero();
        Tensor::MM::ikkj(gate, wg, x);
        gate += bg;
        softmax(gate);

        /* ---- 选出 top-k (TopK >= E 时就是稠密) ---- */
        nSelected = 0;
        if (TopK >= NumExperts) {
            for (int i = 0; i < NumExperts; i++) {
                selected[nSelected++] = i;
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

        /* 经 softmax 的雅可比: dL/dz_c = g_c·(d_c − Σ_i d_i·g_i) */
        Tensor d_gate_logit((std::size_t)NumExperts, 1);
        Softmax::jacobian_transpose_mul(gate, d_gate, d_gate_logit);

        /* dL/dx (门控路径) 累加进 ei */
        Tensor::MM::kikj(ei, wg, d_gate_logit);

        /* ---- 专家路径: 只回传被选中的 ---- */
        for (int s = 0; s < nSelected; s++) {
            const int i = selected[s];
            const float gi = gate[i];
            for (int j = 0; j < d_model; j++) {
                experts[i].e[j] = gi * e[j];
            }
            experts[i].backward(x, ei);   /* 累加进 ei */
        }

        /* ---- 门控参数梯度 ---- */
        Tensor::MM::ikjk(g_wg, d_gate_logit, x);
        g_bg += d_gate_logit;

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
        Softmax::jacobian_transpose_mul(pBar, dP, dz);
        Tensor::MM::ikjk(g_wg, dz, xBar);
        g_bg += dz;

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

    void resetUsage() override
    {
        for (int i = 0; i < NumExperts; i++) {
            usageTotal[i] = 0;
            usageBatch[i] = 0;
            probSumBatch[i] = 0.0;
        }
        for (int j = 0; j < d_model; j++) {
            xSum[j] = 0.0f;
        }
        batchForwardCount = 0;
    }

    /* ---- 优化器 / 序列化 ---- */
    void SGD(float lr) override
    {
        Optimize::SGD(wg, g_wg, lr);
        Optimize::SGD(bg, g_bg, lr);
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
        for (int i = 0; i < NumExperts; i++) {
            experts[i].clamp(c0, cn);
        }
    }

    void copyTo(iLayer* layer) override
    {
        SparseMoE *p = dynamic_cast<SparseMoE*>(layer);
        if (p == nullptr) {
            return;
        }
        p->wg = wg;
        p->bg = bg;
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

    void write(std::ofstream &file) override
    {
        file << wg.toString() << std::endl;
        file << bg.toString() << std::endl;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].write(file);
        }
    }

    void read(std::ifstream &file) override
    {
        std::string s;
        std::getline(file, s);
        wg = Tensor::fromString(s);
        std::getline(file, s);
        bg = Tensor::fromString(s);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].read(file);
        }
    }
};

} /* namespace RL */

#endif // RL_SPARSE_MOE_HPP
