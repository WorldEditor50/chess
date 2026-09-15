#ifndef MOE_HPP
#define MOE_HPP
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"
#include "layer.h"
#include "expert.hpp"
#include "transformer.hpp"

namespace RL {

/*
    MOE<NumExperts, NumHeads, Expert> — Mixture of Experts

    Architecture (每个专家是一个"d_model -> d_model"的 iLayer):

        Input x (d_model × 1):

        1. Gating: gate_logits = Wg · x + b        (NumExperts × 1)
                   gate       = softmax(gate_logits)

        2. For each expert i:
               expert_output_i = expert_i(x)       (d_model × 1)

        3. Weighted output: o = Σ_i gate[i] · expert_output_i   (d_model × 1)

    ---------------------------------------------------------------------------
    专家模板参数 (2026-09 扩展)
    ---------------------------------------------------------------------------
    原来 `experts[]` 写死成 `TransformerBlock<NumHeads>`: 想换一种专家就得复制一份
    这个类。现在专家类型是模板参数, 默认仍是 `TransformerBlock<NumHeads>` ——
    所以 `MOE<16,16>` / `MOE<8,4>` 这些既有写法**一个字都不用改**, 行为也逐位不变
    (等价性由 test_sparse_moe 的 [3] 钉住: 它把 MOE<3,4> 与
    SparseMoE<TransformerBlock<4>,3,3> 的权重复制对齐后比对前向与门控梯度)。

    专家的构造与初始化缩放走 `rl/expert.hpp` 里那套**与 sparse_moe.hpp 完全相同**的
    约定: `ExpertFactory<Expert>::make(d_model, expertHidden, withGrad)` +
    `scaleExpertInit(expert)`。于是可用的专家类型与 SparseMoE 完全一致:

        TransformerBlock<H,DFF>   容量最大, 参数 ~12·d²   <- 默认
        MlpExpert                 中等,   参数 ~2·d·hidden
        Layer<Fn>                 最便宜, 参数 ~d²

    隐层宽度是**构造参数** `expertHidden` (与 SparseMoE 一致), 不做成模板参数 ——
    否则同一个量有两个入口, 其中一个静默失效 (这个坑当场踩过一次)。

    例:
        MOE<4, 4>                            // 4 个 TransformerBlock<4> 专家 (老写法)
        MOE<4, 4, MlpExpert>::_(d, true, 64) // 4 个隐层 64 的 MLP 专家
        MOE<8, 4, Layer<Gelu> >::_(d, true)  // 8 个单层 FC 专家

    `scaleExperts` 默认 **false**: MOE 是 DQN / SAC 的现役主干 (`MOE<16,16>`), 给它们
    悄悄换一套初始化等于悄悄改掉已训练权重之外的一切。要 fan-in 缩放就显式传 true
    (SparseMoE 是**总是**缩放的, 因为它生来只服务新配置)。

    Type registration: LAYER_MOE
*/
template<int NumExperts, int NumHeads = 4,
         typename Expert = TransformerBlock<NumHeads> >
class MOE : public iLayer
{
public:
    struct MOEGrad {
        Tensor wg;
        Tensor b;
        MOEGrad() {}
        void zero() { wg.zero(); b.zero(); }
    };

    int d_model;             // model dimension (shared across all experts)
    int d_ff_;               // FFN hidden dimension (TransformerBlock 专家的; 其它专家忽略)

    /* Gating network */
    Tensor wg;               // gating weights (NumExperts × d_model)
    Tensor b;                // gating bias     (NumExperts × 1)
    //Tensor gate_logits;      // pre-softmax gating logits (NumExperts × 1)
    Tensor gate;             // post-softmax gating weights (NumExperts × 1)

    /* Experts — 类型由模板参数给, 由 ExpertFactory 构造 (见 rl/expert.hpp) */
    Expert experts[NumExperts];

    /* Cached intermediate values for backward/gradient */
    Tensor expert_out[NumExperts];  // each expert's output (d_model × 1)

    /* Gradients */
    MOEGrad g, v, m;

public:
    MOE() {}
    virtual ~MOE() {}

    static std::shared_ptr<MOE> _(
        int d_model_, bool withGrad,
        int expertHidden = 0, bool scaleExperts = false)
    {
        return std::make_shared<MOE>(d_model_, withGrad, expertHidden, scaleExperts);
    }

    explicit MOE(int d_model_, bool withGrad,
                 int expertHidden = 0, bool scaleExperts = false)
        : d_model(d_model_)
    {
        type = LAYER_MOE;
        d_ff_ = 4 * d_model;

        /* Gating network parameters */
        wg = Tensor(NumExperts, d_model);
        b  = Tensor(NumExperts, 1);
        Random::uniform(wg, -0.1f, 0.1f);
        Random::uniform(b,  -0.1f, 0.1f);
        gate        = Tensor(NumExperts, 1);

        /* Experts — 一律走工厂, 这样"换专家"不需要改这个类 */
        for (int i = 0; i < NumExperts; i++) {
            experts[i] = ExpertFactory<Expert>::make(d_model, expertHidden, withGrad);
            if (scaleExperts) {
                scaleExpertInit(experts[i]);
            }
        }

        /* Output and error */
        o = Tensor(d_model, 1);
        e = Tensor(d_model, 1);

        /* Cached intermediates */
        //x_orig = Tensor(d_model, 1);
        for (int i = 0; i < NumExperts; i++) {
            expert_out[i] = Tensor(d_model, 1);
        }

        /* Gradient tensors */
        if (withGrad) {
            g.wg = Tensor(NumExperts, d_model);
            g.b  = Tensor(NumExperts, 1);
            v.wg = Tensor(NumExperts, d_model);
            v.b  = Tensor(NumExperts, 1);
            m.wg = Tensor(NumExperts, d_model);
            m.b  = Tensor(NumExperts, 1);
        }
    }

    /*
        参数量 (只读诊断)。原来没实现 -> `Net::paramCount()` 对 MOE 主干只报 0
        (它只统计实现了 paramCount 的层), 于是"参数量"这件事在日志里是错的。
        这里按 门控 + 各专家 累加; 专家自己也要实现 paramCount 才有意义
        (TransformerBlock / MultiHeadAttention / MlpExpert / Layer 都已实现)。
    */
    long long paramCount() const override
    {
        long long total = (long long)wg.size() + (long long)b.size();
        for (int i = 0; i < NumExperts; i++) {
            total += experts[i].paramCount();
        }
        return total;
    }

    /* ==================== Forward ==================== */

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        /* Save original input for backward */
        //x_orig = x;

        /* Step 1: Compute gating distribution */
        /* gate_logits = Wg · x + b */
        /* MM::ikkj accumulates into gate, so it must be cleared first
           (gate is a persistent member and is never zeroed in backward). */
        gate.zero();
        Tensor::MM::ikkj(gate, wg, x);
        gate += b;
        softmax(gate);        // in-place softmax

        /* Step 2: Compute each expert's output and weighted combination */
        o.zero();
        for (int i = 0; i < NumExperts; i++) {
            /* Forward through TransformerBlock expert */
            Tensor &exp_out = experts[i].forward(x, inference);
            expert_out[i] = exp_out;  // cache

            /* o += gate[i] * expert_output_i */
            float gi = gate[i];
            if (gi > 1e-8f) {  // skip negligible contributions
                for (int j = 0; j < d_model; j++) {
                    o[j] += gi * expert_out[i][j];
                }
            }
        }

        return o;
    }

    /* ==================== Backward ==================== */
    /*
        Gradient flow:

        o = Σ_i gate[i] · expert_out[i]

        e = dL/do  (d_model × 1)

        ∂L/∂expert_out[i] = gate[i] · e          (d_model × 1)
        ∂L/∂gate[i]       = e · expert_out[i]     (scalar, dot product)

        dL/dx = Σ_i expert_i.backward(∂L/∂expert_out[i])
                + Wg^T · (J^T · ∂L/∂gate_logit)

        where J = ∂softmax/∂gate_logit (softmax Jacobian, NumExperts×NumExperts)
    */
    void backward(const Tensor& x, Tensor &ei) override
    {
        /* ---- Gating path ---- */
        /* ∂L/∂gate[i] = e · expert_out[i]  (scalar dot product) */
        Tensor d_gate(NumExperts, 1);
        for (int i = 0; i < NumExperts; i++) {
            float dgi = 0;
            for (int j = 0; j < d_model; j++) {
                dgi += e[j] * expert_out[i][j];
            }
            d_gate[i] = dgi;
        }

        /* ∂L/∂gate_logit = J^T · ∂L/∂gate  (softmax Jacobian-vector product) */
        Tensor d_gate_logit(NumExperts, 1);
        Softmax::jacobian_transpose_mul(gate, d_gate, d_gate_logit);

        /* ∂L/∂x_gate = Wg^T · ∂L/∂gate_logit (accumulate into ei) */
        Tensor::MM::kikj(ei, wg, d_gate_logit);

        /* ---- Expert path ---- */
        for (int i = 0; i < NumExperts; i++) {
            float gi = gate[i];
            if (gi < 1e-8f) continue;

            /* ∂L/∂expert_out[i] = gate[i] · e */
            for (int j = 0; j < d_model; j++) {
                experts[i].e[j] = gi * e[j];
            }

            /* Backward through expert, accumulates ∂L/∂x_expert into ei */
            experts[i].backward(x, ei);
        }

        /* ---- Gating parameter gradients ---- */
        Tensor::MM::ikjk(g.wg, d_gate_logit, x);   // g.wg += d_gate_logit · x^T
        g.b += d_gate_logit;                        // g.b  += d_gate_logit

        /* Reset output and error */
        o.zero();
        e.zero();
        return;
    }

    /* ==================== Optimizers ==================== */

    void SGD(float lr) override
    {
        Optimize::SGD(wg, g.wg, lr);
        Optimize::SGD(b,  g.b,  lr);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].SGD(lr);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wg, v.wg, g.wg, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b,  v.b,  g.b,  lr, rho, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(wg, v.wg, m.wg, g.wg,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b,  v.b,  m.b,  g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].Adam(lr, alpha, beta,
                            alpha_, beta_,
                            decay, clipGrad);
        }
        g.zero();
        return;
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(wg, c0, cn);
        Optimize::clamp(b,  c0, cn);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].clamp(c0, cn);
        }
        return;
    }

    void copyTo(iLayer* layer) override
    {
        MOE *p = static_cast<MOE*>(layer);
        p->wg = wg;
        p->b  = b;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].copyTo(&p->experts[i]);
        }
        return;
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        MOE *p = static_cast<MOE*>(layer);
        lerp(p->wg, wg, alpha);
        lerp(p->b,  b,  alpha);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].softUpdateTo(&p->experts[i], alpha);
        }
        return;
    }

    void write(std::ofstream &file) override
    {
        file << wg.toString() << std::endl;
        file << b.toString()  << std::endl;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].write(file);
        }
        return;
    }

    void read(std::ifstream &file) override
    {
        std::string s;
        std::getline(file, s); wg = Tensor::fromString(s);
        std::getline(file, s); b  = Tensor::fromString(s);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].read(file);
        }
        return;
    }
};

}
#endif // MOE_HPP
