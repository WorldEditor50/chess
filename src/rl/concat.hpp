#ifndef CONCAT_HPP
#define CONCAT_HPP
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"
#include "expert.hpp"

namespace RL {

/* ============================================================================
 *  IScaledConcat — 非模板接口 (与 ISparseMoE 同一个理由: 模板参数不同 -> 类型不同,
 *  上层要能 dynamic_cast 到"任何一种" ScaledConcat 才能做诊断)
 * ============================================================================ */
class IScaledConcat : public iLayer
{
public:
    /* 每路累计的平均门控质量 (长度 = N·u, 和为 1) —— 看得到"哪些路在闲置" */
    virtual void gateUsage(std::vector<float> &out) const = 0;
    virtual void resetGateUsage() = 0;
    /*
        有效路数 = exp(平均门控分布的熵)。N·u 表示完全均匀, 1 表示全压在一路上。
        旧实现 (softmax∘sigmoid) 的实测值是 58.6~61.3/64 —— 也就是"根本没在选"。
    */
    virtual double gateEffectiveCount() const = 0;
    virtual int unitCount() const = 0;
};

/* ============================================================================
 *  ScaledConcat<Expert, NumExperts, UnitDim, OutDim>
 * ============================================================================
 *
 * 结构 (2026-09 重写; 旧实现改名 ScaledConcatLegacy 留在本文件下半部分作对照):
 *
 *     e_i = expert_i(x)                       d -> d    ExpertFactory 构造
 *     h_i = proj_i(e_i)                       d -> u    (与 sparse_moe.hpp 同一套约定)
 *     h   = [h_1; …; h_N]                     (N·u)     **特征, 不做归一化**
 *     z   = Wg·x + bg                         (N·u)     门控 logits (实数, 无界)
 *     g   = softmax(gateScale · z)            (N·u)     逐单元门控
 *     o   = tanh( W1·(g∘h) + W2·x + b )       (Out)     (+ x, 若 residual 且 Out==d)
 *
 * 为什么这么改 (逐条都有实测, 见 docs/rl_sync.md 的 ScaledConcat 一节):
 *
 *  1. **门控 logits 必须无界。** 旧实现是 `softmax(sigmoid(W·x+b))`, 而
 *     Sigmoid::f ∈ (0,1) => logits 跨度 < 1 => gate_max/gate_min < e^1 = 2.718,
 *     与训练轮次、参数量、数据量**都无关**。实测 (D=90,u=4,N=16) 有效路数
 *     58.6~61.3/64: 16 个专家永远以几乎相同的权重被平均在一起, "门控"这个机制
 *     一个 bit 的信息都传不出去。现在 logits 就是实数。
 *
 *  2. **特征与门控解耦。** 旧实现里子层输出既当特征又当 logits, 过完 softmax 幅度
 *     信息全丢 (每路 ≈ 1/Nu), 于是 W1·g 退化成"把 W1 的列平均一下", 整层的表达力
 *     几乎全落在 W2·x 上 (@90 占 37% 参数, @1260 占 49%)。现在 h 原样保留,
 *     门控另算 —— Wg 只有 N·u × d。
 *
 *  3. **初始化按 fan-in 缩放。** 原来是 U(-1,1): 实测最终 pre-activation 落在
 *     ±3.7~11.2, |z|>3 的比例 3~45%, 平均 tanh'(o) 只有 0.18~0.56 —— 开局就在
 *     饱和区 (缩放后是 0.40~1.19 / 0% / 0.90~0.98)。专家与投影走
 *     scaleExpertInit / scaleFcInit, W1/W2 各自按 fan-in 缩小。
 *     `zeroSkip` 默认 true: W2 零初始化, 于是这层**开局就是它声称的那个门控混合**,
 *     而不是"混合 + 一条随机线性捷径"(旧实现里捷径的量级还压过混合)。
 *
 *  4. **保维可选。** OutDim 默认 N·u (特征提取器); 填 dIn 就能当保维变换块,
 *     再配合 residual=true 直接做残差块 —— 旧实现 out 恒等于 N·u, 只能当提取器,
 *     想用它就得在后面接一层投影。
 *
 * 没做的 (写在这里免得下次当成"忘了"): top-k 稀疏门控、W1 低秩分解、子层共享底座。
 * 前两者省的是 W1 的 (N·u)² 项 —— 而专家仍然全算, 省不到算力大头; 后者是参数效率
 * 问题。都值得单独一轮 + 各自的测试, 不该塞进这次改动。
 *
 * 坍缩**诊断**做了 (gateUsage / gateEffectiveCount), 但**没有**配负载均衡辅助损失:
 * 这里的门控是稠密的逐单元门控, 加均衡项等于把门控往均匀推 —— 那正是上面第 1 条
 * 刚修掉的毛病。想知道"路有没有闲着"看 gateEffectiveCount()。
 * ============================================================================ */
template<typename Expert, int NumExperts, int UnitDim, int OutDim = 0>
class ScaledConcat : public IScaledConcat
{
    static_assert(NumExperts > 0, "NumExperts 必须为正");
    static_assert(UnitDim > 0, "UnitDim 必须为正");
public:
    static constexpr int Units = NumExperts * UnitDim;
    static constexpr int Out = (OutDim > 0) ? OutDim : Units;

    struct ScaledConcatGrad
    {
        Tensor w1;
        Tensor w2;
        Tensor b;
        Tensor wg;
        Tensor bg;
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
            wg.zero();
            bg.zero();
        }
    };

public:
    int dIn;
    /* softmax 温度: 门控 logits 乘它。因为 logits 无界, Wg 自己的尺度就能学出
       "要多尖", 这个量只是给调度器/实验用的显式旋钮 (1.0 = 不动) */
    float gateScale;
    /* Out == dIn 时把 x 加回 pre-activation (残差块)。Out != dIn 时被忽略 */
    bool useResidual;

    /* 专家 (d -> d) 与每路的 d -> UnitDim 投影 */
    Expert experts[NumExperts];
    Layer<Linear> proj[NumExperts];

    /* 门控 */
    Tensor wg;      /* (Units × dIn) */
    Tensor bg;      /* (Units × 1)   */
    Tensor gate;    /* softmax(logits)        (Units × 1) */
    Tensor feat;    /* h = concat(proj_i(·))  (Units × 1) —— 不做归一化 */
    Tensor gated;   /* gate ∘ feat            (Units × 1) */

    /* 输出侧 */
    Tensor w1;      /* (Out × Units) */
    Tensor w2;      /* (Out × dIn)   */
    Tensor b;       /* (Out × 1)     */

    /* 反向 scratch (预分配: 旧实现每次 backward 都 new 3 个 Tensor) */
    Tensor dz;
    Tensor dzGated;
    Tensor dGate;
    Tensor dFeat;
    Tensor dLogit;
    Tensor dEo;

    ScaledConcatGrad g, v, m;

    /* 门控诊断 (生命周期累计) */
    std::vector<double> gateSum;
    long long gateCount;

public:
    ScaledConcat() : dIn(0), gateScale(1.0f), useResidual(false), gateCount(0) {}

    static std::shared_ptr<ScaledConcat> _(
        int dIn_, bool withGrad, int expertHidden = 0,
        bool residual = false, float gateScale_ = 1.0f, bool zeroSkip = true)
    {
        return std::make_shared<ScaledConcat>(dIn_, withGrad, expertHidden,
                                              residual, gateScale_, zeroSkip);
    }

    explicit ScaledConcat(int dIn_, bool withGrad,
                          int expertHidden = 0,
                          bool residual = false,
                          float gateScale_ = 1.0f,
                          bool zeroSkip = true)
        : dIn(dIn_), gateScale(gateScale_), useResidual(residual), gateCount(0)
    {
        type = LAYER_SCALEDCONCAT;

        /* ---- 门控: 与 MOE / SparseMoE 同尺度的小初始化 ----
           注意这里**不是**把 sigmoid 当激活用; logits 直接是实数, 训练可以把
           门控拉得多尖由数据决定 (旧实现把 logits 锁在 (0,1) 里, 永远拉不尖)。 */
        wg = Tensor(Units, dIn);
        bg = Tensor(Units, 1);
        Random::uniform(wg, -0.1f, 0.1f);
        Random::uniform(bg, -0.1f, 0.1f);
        gate  = Tensor(Units, 1);
        feat  = Tensor(Units, 1);
        gated = Tensor(Units, 1);

        /* ---- 专家 + 投影 ---- */
        for (int i = 0; i < NumExperts; i++) {
            experts[i] = ExpertFactory<Expert>::make(dIn, expertHidden, withGrad);
            scaleExpertInit(experts[i]);
            proj[i] = Layer<Linear>(dIn, UnitDim, true, withGrad);
            scaleFcInit(proj[i]);
        }

        /* ---- 输出侧 ---- */
        w1 = Tensor(Out, Units);
        w2 = Tensor(Out, dIn);
        b  = Tensor(Out, 1);
        Random::uniform(w1, -1.0f, 1.0f);
        Random::uniform(w2, -1.0f, 1.0f);
        Random::uniform(b,  -1.0f, 1.0f);
        const float s1 = 1.0f / std::sqrt((float)Units);
        const float s2 = 1.0f / std::sqrt((float)(dIn > 1 ? dIn : 1));
        for (std::size_t k = 0; k < w1.size(); k++) {
            w1[k] *= s1;
        }
        for (std::size_t k = 0; k < w2.size(); k++) {
            w2[k] *= s2;
        }
        for (std::size_t k = 0; k < b.size(); k++) {
            b[k] *= s1;
        }
        if (zeroSkip) {
            /* 跳连零初始化: 开局输出 = 纯门控混合 (梯度照样会把它推起来) */
            w2.zero();
        }

        o = Tensor(Out, 1);
        e = Tensor(Out, 1);
        dz       = Tensor(Out, 1);
        dzGated  = Tensor(Units, 1);
        dGate    = Tensor(Units, 1);
        dFeat    = Tensor(Units, 1);
        dLogit   = Tensor(Units, 1);
        dEo      = Tensor(dIn, 1);

        if (withGrad) {
            g.w1 = Tensor(Out, Units);
            v.w1 = Tensor(Out, Units);
            m.w1 = Tensor(Out, Units);
            g.w2 = Tensor(Out, dIn);
            v.w2 = Tensor(Out, dIn);
            m.w2 = Tensor(Out, dIn);
            g.b = Tensor(Out, 1);
            v.b = Tensor(Out, 1);
            m.b = Tensor(Out, 1);
            g.wg = Tensor(Units, dIn);
            v.wg = Tensor(Units, dIn);
            m.wg = Tensor(Units, dIn);
            g.bg = Tensor(Units, 1);
            v.bg = Tensor(Units, 1);
            m.bg = Tensor(Units, 1);
        }

        gateSum.assign((std::size_t)Units, 0.0);
    }

    int unitCount() const override { return Units; }

    long long paramCount() const override
    {
        long long total = (long long)wg.size() + (long long)bg.size()
                        + (long long)w1.size() + (long long)w2.size() + (long long)b.size();
        for (int i = 0; i < NumExperts; i++) {
            total += experts[i].paramCount();
            total += proj[i].paramCount();
        }
        return total;
    }

    /* ==================== 前向 ==================== */
    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        /* ---- 门控: z = gateScale · (Wg·x + bg), 然后逐单元 softmax ---- */
        gate.zero();
        Tensor::MM::ikkj(gate, wg, x);
        gate += bg;
        if (gateScale != 1.0f) {
            gate *= gateScale;
        }
        softmax(gate);

        /* ---- 特征: h_i = proj_i(expert_i(x)), 原样写进 feat 的第 i 段 ----
           直接用连续指针拷 (Tensor::embedding 是逐元素通用索引 + 每次一个
           std::vector 分配, 实测 16 次 = 1.9 us, 比这里全部的计算还贵)。 */
        for (int i = 0; i < NumExperts; i++) {
            Tensor &eo = experts[i].forward(x, inference);
            Tensor &hi = proj[i].forward(eo, inference);
            float *dst = feat.val.data() + (std::size_t)UnitDim * i;
            const float *src = hi.val.data();
            for (int r = 0; r < UnitDim; r++) {
                dst[r] = src[r];
            }
        }

        /* ---- 逐单元门控: gated = gate ∘ feat ---- */
        for (int k = 0; k < Units; k++) {
            gated[k] = gate[k] * feat[k];
        }

        /* ---- 输出 ---- */
        o.zero();
        Tensor::MM::ikkj(o, w1, gated);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        if (useResidual && Out == dIn) {
            o += x;
        }
        for (int k = 0; k < Out; k++) {
            o[k] = Tanh::f(o[k]);
        }

        /* ---- 诊断累计 ---- */
        for (int k = 0; k < Units; k++) {
            gateSum[(std::size_t)k] += (double)gate[k];
        }
        gateCount++;
        return o;
    }

    /* ==================== 反向 ==================== */
    /*
        o     = tanh(W1·gated + W2·x + b)
        gated = gate ∘ feat
        gate  = softmax(gateScale·(Wg·x + bg))
        feat  = concat(proj_i(expert_i(x)))

        dz      = tanh'(o) ⊙ e
        dW1    += dz·gatedᵀ ; dW2 += dz·xᵀ ; db += dz
        dgated  = W1ᵀ·dz
        dgate   = dgated ⊙ feat ;  dfeat = dgated ⊙ gate
        dlogit  = gateScale · Jᵀ_softmax(gate)·dgate
        dWg    += dlogit·xᵀ ; dbg += dlogit ; ei += Wgᵀ·dlogit
        ei     += W2ᵀ·dz (+ dz, 残差时)
        dfeat_i -> proj_i.backward -> expert_i.e -> expert_i.backward(x, ei)
    */
    void backward(const Tensor &x, Tensor &ei) override
    {
        for (int k = 0; k < Out; k++) {
            dz[k] = Tanh::df(o[k]) * e[k];
        }
        Tensor::MM::ikjk(g.w1, dz, gated);
        Tensor::MM::ikjk(g.w2, dz, x);
        g.b += dz;

        dzGated.zero();
        Tensor::MM::kikj(dzGated, w1, dz);
        for (int k = 0; k < Units; k++) {
            dGate[k] = dzGated[k] * feat[k];
            dFeat[k] = dzGated[k] * gate[k];
        }

        /* 门控路径 */
        Softmax::jacobian_transpose_mul(gate, dGate, dLogit);
        if (gateScale != 1.0f) {
            dLogit *= gateScale;      /* logits = gateScale·(Wg·x+bg) */
        }
        Tensor::MM::ikjk(g.wg, dLogit, x);
        g.bg += dLogit;
        Tensor::MM::kikj(ei, wg, dLogit);        /* ei += Wgᵀ·dlogit */

        /* 输入通路 */
        if (useResidual && Out == dIn) {
            for (int k = 0; k < dIn; k++) {
                ei[k] += dz[k];
            }
        }
        Tensor::MM::kikj(ei, w2, dz);            /* ei += W2ᵀ·dz */

        /* 专家路径: feat 的每一段 -> proj_i -> expert_i */
        for (int i = 0; i < NumExperts; i++) {
            float *pe = proj[i].e.val.data();
            const float *df = dFeat.val.data() + (std::size_t)UnitDim * i;
            for (int r = 0; r < UnitDim; r++) {
                pe[r] = df[r];
            }
            /* proj_i 的输入是 expert_i 的输出 —— 必须在 expert_i.backward 之前用掉
               (那个调用会把专家的 o 清零) */
            dEo.zero();
            proj[i].backward(experts[i].o, dEo);
            for (int r = 0; r < dIn; r++) {
                experts[i].e[r] = dEo[r];
            }
            experts[i].backward(x, ei);          /* 累加进 ei */
        }

        o.zero();
        e.zero();
    }

    /* ==================== 优化器 ==================== */
    void SGD(float lr) override
    {
        Optimize::SGD(w1, g.w1, lr);
        Optimize::SGD(w2, g.w2, lr);
        Optimize::SGD(b,  g.b,  lr);
        Optimize::SGD(wg, g.wg, lr);
        Optimize::SGD(bg, g.bg, lr);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].SGD(lr);
            proj[i].SGD(lr);
        }
        g.zero();
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b,  v.b,  g.b,  lr, rho, decay, clipGrad);
        Optimize::RMSProp(wg, v.wg, g.wg, lr, rho, decay, clipGrad);
        Optimize::RMSProp(bg, v.bg, g.bg, lr, rho, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].RMSProp(lr, rho, decay, clipGrad);
            proj[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w1, v.w1, m.w1, g.w1, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(b,  v.b,  m.b,  g.b,  alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(wg, v.wg, m.wg, g.wg, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(bg, v.bg, m.bg, g.bg, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            proj[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        g.zero();
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(w1, c0, cn);
        Optimize::clamp(w2, c0, cn);
        Optimize::clamp(b,  c0, cn);
        Optimize::clamp(wg, c0, cn);
        Optimize::clamp(bg, c0, cn);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].clamp(c0, cn);
            proj[i].clamp(c0, cn);
        }
    }

    void copyTo(iLayer* layer) override
    {
        ScaledConcat *p = dynamic_cast<ScaledConcat*>(layer);
        if (p == nullptr) {
            return;
        }
        p->w1 = w1;
        p->w2 = w2;
        p->b  = b;
        p->wg = wg;
        p->bg = bg;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].copyTo(&p->experts[i]);
            proj[i].copyTo(&p->proj[i]);
        }
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        ScaledConcat *p = dynamic_cast<ScaledConcat*>(layer);
        if (p == nullptr) {
            return;
        }
        lerp(p->w1, w1, alpha);
        lerp(p->w2, w2, alpha);
        lerp(p->b,  b,  alpha);
        lerp(p->wg, wg, alpha);
        lerp(p->bg, bg, alpha);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].softUpdateTo(&p->experts[i], alpha);
            proj[i].softUpdateTo(&p->proj[i], alpha);
        }
    }

    /* ==================== 门控诊断 ==================== */
    void gateUsage(std::vector<float> &out) const override
    {
        out.resize((std::size_t)Units);
        const double inv = (gateCount > 0) ? (1.0 / (double)gateCount) : 0.0;
        for (int i = 0; i < Units; i++) {
            out[(std::size_t)i] = (float)(gateSum[(std::size_t)i] * inv);
        }
    }

    double gateEffectiveCount() const override
    {
        if (gateCount <= 0) {
            return 0.0;
        }
        const double inv = 1.0 / (double)gateCount;
        double H = 0.0;
        for (int i = 0; i < Units; i++) {
            const double p = gateSum[(std::size_t)i] * inv;
            if (p > 1e-12) {
                H -= p * std::log(p);
            }
        }
        return std::exp(H);
    }

    void resetGateUsage() override
    {
        for (int i = 0; i < Units; i++) {
            gateSum[(std::size_t)i] = 0.0;
        }
        gateCount = 0;
    }

    /* ==================== 序列化 ====================
       顺序: wg, bg, w1, w2, b, 然后每个专家的权重, 再每个投影的权重。
       (与旧 ScaledConcat 的顺序不同 —— 那个实现没进过任何权重文件: 它只出现在
        dqn.cpp 一棵 #elif 0 的分支里, 见该文件的说明。) */
    void write(std::ofstream &file) override
    {
        file << wg.toString() << std::endl;
        file << bg.toString() << std::endl;
        file << w1.toString() << std::endl;
        file << w2.toString() << std::endl;
        file << b.toString()  << std::endl;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].write(file);
        }
        for (int i = 0; i < NumExperts; i++) {
            proj[i].write(file);
        }
    }

    void read(std::ifstream &file) override
    {
        std::string s;
        std::getline(file, s); wg = Tensor::fromString(s);
        std::getline(file, s); bg = Tensor::fromString(s);
        std::getline(file, s); w1 = Tensor::fromString(s);
        std::getline(file, s); w2 = Tensor::fromString(s);
        std::getline(file, s); b  = Tensor::fromString(s);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].read(file);
        }
        for (int i = 0; i < NumExperts; i++) {
            proj[i].read(file);
        }
    }
};

/*
 * ----------------------------------------------------------------------------
 *  ScaledConcatLegacy — 上游 snakeAI 的旧实现, **保留但不要用**
 *
 *  数学: softmax(sigmoid(W_i·x + b_i)) 之后再做 W1 混合 + W2 跳连。
 *  它有两个结构性缺陷 (实测见 docs/rl_sync.md):
 *    * Sigmoid 把 logits 压进 (0,1) => gate_max/gate_min < e = 2.718, 门控永远
 *      接近均匀 (有效路数 58.6~61.3/64);
 *    * 特征与门控共用同一批数 => 幅度信息被 softmax 抹掉, W1·g 退化成列平均;
 *    * U(-1,1) 初始化 => 最终 tanh 开局就饱和 (|z|>3 占 3~45%)。
 *  留在文件里的唯一理由: 与上游对照、以及当"想回滚"时的参照物。
 *  新代码请用 ScaledConcat<Expert,NumExperts,UnitDim>(见上)。
 * ----------------------------------------------------------------------------
 */

template<typename TLayer, int N>
class ScaledConcatLegacy : public iLayer
{
public:
    class ScaledConcatGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        ScaledConcatGrad(){}
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int unitDim;
    int outputDim;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    TLayer layers[N];
    ScaledConcatGrad g;
    ScaledConcatGrad v;
    ScaledConcatGrad m;
public:
    ScaledConcatLegacy(){}
    explicit ScaledConcatLegacy(const TLayer& layer, int inputDim_, int unitDim_, bool withGrad)
        :inputDim(inputDim_),unitDim(unitDim_)
    {
        type = LAYER_SCALEDCONCAT;
        outputDim = unitDim*N;
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        for (int i = 0; i < N; i++) {
            layers[i] = layer;
            layers[i].initParams();
        }
        a = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
            g.w1 = Tensor(outputDim, outputDim);
            v.w1 = Tensor(outputDim, outputDim);
            m.w1 = Tensor(outputDim, outputDim);
            g.w2 = Tensor(outputDim, inputDim);
            v.w2 = Tensor(outputDim, inputDim);
            m.w2 = Tensor(outputDim, inputDim);
            g.b = Tensor(outputDim, 1);
            v.b = Tensor(outputDim, 1);
            m.b = Tensor(outputDim, 1);
        }
    }

    static std::shared_ptr<ScaledConcatLegacy> _(const TLayer& layer, int inputDim, int unitDim, bool withGrad)
    {
        return std::make_shared<ScaledConcatLegacy>(layer, inputDim, unitDim, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        for (int i = 0; i < N; i++) {
            Tensor &out = layers[i].forward(x, inference);
            a.embedding({i*unitDim, 0}, out);
        }
        softmax(a);
        o.zero();
        Tensor::MM::ikkj(o, w1, a);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(o[i]);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* dz = dL/d(w1·a + w2·x + b) = tanh'(o) ⊙ e */
        Tensor dz(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dz[i] = Tanh::df(o[i])*e[i];
        }
        /* a-path: dL/da_softmax = w1^T·dz, then through softmax: J^T·(w1^T·dz).
           The per-sublayer error is a slice of THAT, not a slice of e. */
        Tensor dzSoft(outputDim, 1);
        Tensor::MM::kikj(dzSoft, w1, dz);
        Tensor da(outputDim, 1);
        Softmax::jacobian_transpose_mul(a, dzSoft, da);
        for (int i = 0; i < N; i++) {
            layers[i].e = da.block({unitDim*i, 0}, {unitDim, 1});
        }
        /* x-path: dL/dx += w2^T·dz */
        Tensor::MM::kikj(ei, w2, dz);

        Tensor::MM::ikjk(g.w1, dz, a);
        Tensor::MM::ikjk(g.w2, dz, x);
        g.b += dz;
        for (int i = 0; i < N; i++) {
            layers[i].backward(x, ei);
        }
        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(w1, g.w1, lr);
        Optimize::SGD(w2, g.w2, lr);
        Optimize::SGD(b, g.b, lr);
        for (int i = 0; i < N; i++) {
            layers[i].SGD(lr);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            layers[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w1, v.w1, m.w1, g.w1,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            layers[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < N; i++) {
             layers[i].clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         ScaledConcatLegacy *pLayer = static_cast<ScaledConcatLegacy*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < N; i++) {
            layers[i].copyTo(&pLayer->layers[i]);
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         ScaledConcatLegacy *pLayer = static_cast<ScaledConcatLegacy*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < N; i++) {
             layers[i].softUpdateTo(&pLayer->layers[i], alpha);
         }
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<w1.toString()<<std::endl;
         file<<w2.toString()<<std::endl;
         /* b */
         file<<b.toString()<<std::endl;
         for (int i = 0; i < N; i++) {
             layers[i].write(file);
         }
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string w1s;
         std::getline(file, w1s);
         w1 = Tensor::fromString(w1s);
         std::string w2s;
         std::getline(file, w2s);
         w2 = Tensor::fromString(w2s);
         std::string bs;
         std::getline(file, bs);
         b = Tensor::fromString(bs);
         for (int i = 0; i < N; i++) {
             layers[i].read(file);
         }
         return;
     }
};

template<typename TLayer, int N>
class ScaledConcatP : public iLayer
{
public:
    class ScaledConcatPGrad
    {
    public:
        Tensor w0;
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        ScaledConcatPGrad(){}
        void zero()
        {
            w0.zero();
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int unitDim;
    int outputDim;
    Tensor w0;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    Tensor z1;
    Tensor z2;
    TLayer layers[N];
    ScaledConcatPGrad g;
    ScaledConcatPGrad v;
    ScaledConcatPGrad m;
public:
    ScaledConcatP(){}
    explicit ScaledConcatP(const TLayer& layer, int inputDim_, int unitDim_, bool withGrad)
        :inputDim(inputDim_),unitDim(unitDim_)
    {
        type = LAYER_SCALEDCONCAT;
        outputDim = unitDim*N;
        w0 = Tensor(outputDim, outputDim);
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        for (int i = 0; i < N; i++) {
            layers[i] = layer;
            layers[i].initParams();
        }
        a = Tensor(outputDim, 1);
        z1 = Tensor(outputDim, 1);
        z2 = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
            g.w0 = Tensor(outputDim, outputDim);
            v.w0 = Tensor(outputDim, outputDim);
            m.w0 = Tensor(outputDim, outputDim);
            g.w1 = Tensor(outputDim, outputDim);
            v.w1 = Tensor(outputDim, outputDim);
            m.w1 = Tensor(outputDim, outputDim);
            g.w2 = Tensor(outputDim, inputDim);
            v.w2 = Tensor(outputDim, inputDim);
            m.w2 = Tensor(outputDim, inputDim);
            g.b = Tensor(outputDim, 1);
            v.b = Tensor(outputDim, 1);
            m.b = Tensor(outputDim, 1);
        }
    }

    static std::shared_ptr<ScaledConcatP> _(const TLayer& layer, int inputDim, int unitDim, bool withGrad)
    {
        return std::make_shared<ScaledConcatP>(layer, inputDim, unitDim, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        for (int i = 0; i < N; i++) {
            Tensor &out = layers[i].forward(x, inference);
            a.embedding({i*unitDim, 0}, out);
        }
        z1.zero();
        Tensor::MM::ikkj(z1, w0, a);
        for (std::size_t i = 0; i < z1.totalSize; i++) {
            z1[i] = Tanh::f(z1[i]);
        }
        z2 = Softmax::f(z1);
        o.zero();
        Tensor::MM::ikkj(o, w1, z2);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(o[i]);
        }
        return o;
    }

    void backward(const Tensor &x, Tensor &ei) override
    {
        /* dz = dL/d(w1·z2 + w2·x + b) = tanh'(o) ⊙ e */
        Tensor dz(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dz[i] = Tanh::df(o[i])*e[i];
        }

        Tensor::MM::ikjk(g.w1, dz, z2);
        Tensor::MM::ikjk(g.w2, dz, x);
        g.b += dz;
        /*
            z1   = tanh(w0·a)
            z2   = softmax(z1)
            o    = tanh(w1·z2 + w2·x + b)

            dz2   = w1^T · dz
            dz1   = J_softmax^T · dz2
            dpre1 = tanh'(z1) ⊙ dz1        (= dL/d(w0·a))
            g.w0 += dpre1 · a^T
            da    = w0^T · dpre1           (fed to the sublayers)
        */
        Tensor dz2(outputDim, 1);
        Tensor::MM::kikj(dz2, w1, dz);
        Tensor dpre1(outputDim, 1);
        Softmax::jacobian_transpose_mul(z2, dz2, dpre1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dpre1[i] *= Tanh::df(z1[i]);
        }
        Tensor::MM::ikjk(g.w0, dpre1, a);
        Tensor da(outputDim, 1);
        Tensor::MM::kikj(da, w0, dpre1);
        for (int i = 0; i < N; i++) {
            layers[i].e = da.block({unitDim*i, 0}, {unitDim, 1});
        }
        /* x-path: dL/dx += w2^T·dz */
        Tensor::MM::kikj(ei, w2, dz);
        for (int i = 0; i < N; i++) {
            layers[i].backward(x, ei);
        }
        z1.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(w0, g.w0, lr);
        Optimize::SGD(w1, g.w1, lr);
        Optimize::SGD(w2, g.w2, lr);
        Optimize::SGD(b, g.b, lr);
        for (int i = 0; i < N; i++) {
            layers[i].SGD(lr);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w0, v.w0, g.w0, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            layers[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w0, v.w0, m.w0, g.w0,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w1, v.w1, m.w1, g.w1,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            layers[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w0, c0, cn);
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < N; i++) {
             layers[i].clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         ScaledConcatP *pLayer = static_cast<ScaledConcatP*>(layer);
         pLayer->w0 = w0;
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < N; i++) {
            layers[i].copyTo(&pLayer->layers[i]);
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         ScaledConcatP *pLayer = static_cast<ScaledConcatP*>(layer);
         lerp(pLayer->w0, w0, alpha);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < N; i++) {
             layers[i].softUpdateTo(&pLayer->layers[i], alpha);
         }
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<w0.toString()<<std::endl;
         file<<w1.toString()<<std::endl;
         file<<w2.toString()<<std::endl;
         /* b */
         file<<b.toString()<<std::endl;
         for (int i = 0; i < N; i++) {
             layers[i].write(file);
         }
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string w0s;
         std::getline(file, w0s);
         w0 = Tensor::fromString(w0s);
         std::string w1s;
         std::getline(file, w1s);
         w1 = Tensor::fromString(w1s);
         std::string w2s;
         std::getline(file, w2s);
         w2 = Tensor::fromString(w2s);
         std::string bs;
         std::getline(file, bs);
         b = Tensor::fromString(bs);
         for (int i = 0; i < N; i++) {
             layers[i].read(file);
         }
         return;
     }
};

class Concat : public iLayer
{
public:
    class ConcatGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        ConcatGrad(){}
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int outputDim;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    std::vector<std::shared_ptr<iLayer> > layers;
    ConcatGrad g;
    ConcatGrad v;
    ConcatGrad m;
public:
    Concat(){}
    template<typename ...TLayer>
    explicit Concat(int inputDim_, bool withGrad, TLayer&&...layer)
        :inputDim(inputDim_),layers({layer...})
    {
        type = LAYER_CONCAT;
        outputDim = 0;
        for (std::size_t i = 0; i < layers.size(); i++) {
            outputDim += layers[i]->o.totalSize;
        }
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        a = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
            g.w1 = Tensor(outputDim, outputDim);
            v.w1 = Tensor(outputDim, outputDim);
            m.w1 = Tensor(outputDim, outputDim);
            g.w2 = Tensor(outputDim, inputDim);
            v.w2 = Tensor(outputDim, inputDim);
            m.w2 = Tensor(outputDim, inputDim);
            g.b = Tensor(outputDim, 1);
            v.b = Tensor(outputDim, 1);
            m.b = Tensor(outputDim, 1);
        }
    }

    static std::shared_ptr<Concat> _(int inputDim, int unitDim, bool withGrad)
    {
        return std::make_shared<Concat>(inputDim, unitDim, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        int offset = 0;
        for (int i = 0; i < layers.size(); i++) {
            Tensor &out = layers[i]->forward(x, inference);
            a.embedding({offset, 0}, out);
            offset += out.totalSize;
        }
        softmax(a);
        o.zero();
        Tensor::MM::ikkj(o, w1, a);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(o[i]);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* dz = dL/d(w1·a + w2·x + b) = tanh'(o) ⊙ e */
        Tensor dz(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dz[i] = Tanh::df(o[i])*e[i];
        }
        /* a-path: dL/da_softmax = w1^T·dz, then through softmax: J^T·(w1^T·dz) */
        Tensor dzSoft(outputDim, 1);
        Tensor::MM::kikj(dzSoft, w1, dz);
        Tensor da(outputDim, 1);
        Softmax::jacobian_transpose_mul(a, dzSoft, da);
        int offset = 0;
        for (int i = 0; i < layers.size(); i++) {
            int unitDim = layers[i]->o.totalSize;
            layers[i]->e = da.block({offset, 0}, {unitDim, 1});
            offset += unitDim;
        }

        Tensor::MM::ikjk(g.w1, dz, a);
        Tensor::MM::ikjk(g.w2, dz, x);
        g.b += dz;
        /* x-path: dL/dx += w2^T·dz */
        Tensor::MM::kikj(ei, w2, dz);
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->backward(x, ei);
        }
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate) override
    {
        Optimize::SGD(w1, g.w1, learningRate);
        Optimize::SGD(w2, g.w2, learningRate);
        Optimize::SGD(b, g.b, learningRate);
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->SGD(learningRate);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w1, v.w1, m.w1, g.w1,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         Concat *pLayer = static_cast<Concat*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < layers.size(); i++) {
            layers[i]->copyTo(pLayer->layers[i].get());
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         Concat *pLayer = static_cast<Concat*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->softUpdateTo(pLayer->layers[i].get(), alpha);
         }
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<w1.toString()<<std::endl;
         file<<w2.toString()<<std::endl;
         /* b */
         file<<b.toString()<<std::endl;
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->write(file);
         }
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string w1s;
         std::getline(file, w1s);
         w1 = Tensor::fromString(w1s);
         std::string w2s;
         std::getline(file, w2s);
         w2 = Tensor::fromString(w2s);
         std::string bs;
         std::getline(file, bs);
         b = Tensor::fromString(bs);
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->read(file);
         }
         return;
     }
};
}
#endif // CONCAT_HPP
