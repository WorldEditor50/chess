#ifndef RL_EXPERT_HPP
#define RL_EXPERT_HPP

/*
 * expert.hpp — MoE 类层共用的"专家"机制
 * ============================================================================
 *
 * 为什么单独一个文件: 原来这套东西 (MlpExpert / ExpertFactory / scaleExpertInit)
 * 全部长在 `sparse_moe.hpp` 里。现在 `moe.hpp` 与 `concat.hpp` 的 ScaledConcat 都要
 * **按模板参数接受多种专家类型**, 如果各自再抄一份工厂和初始化缩放, 三份拷贝迟早
 * 会漂移 (新增一种专家就要记得改三处, 忘记一处就编译不过或者初始化语义不一致)。
 * 所以这里抽出来共享 —— 也是"专家模板参数"这件事只有一种约定:
 *
 *   1. 专家是 `iLayer`, 输入输出**同维** (`d -> d`), 才能做门控加权和;
 *   2. 构造统一走 `ExpertFactory<Expert>::make(dIn, hidden, withGrad)`
 *      (`hidden` 只有 MlpExpert 用, TransformerBlock 的 FFN 宽度由模板参数决定);
 *   3. 建好之后调 `scaleExpertInit(expert)` 做 fan-in 缩放。
 *
 * 目前支持: TransformerBlock<H,DFF> (最贵、容量最大)、MlpExpert (便宜)、
 *           Layer<Fn> (单层 FC + 激活, 最便宜)。
 * 新增一种专家 = 加一个 ExpertFactory 特化 + 一个 scaleExpertInit 重载, 三处调用方
 * (moe.hpp / sparse_moe.hpp / concat.hpp) 一行都不用改。
 */

#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <vector>
#include "activate.h"
#include "ilayer.h"
#include "layer.h"
#include "optimize.h"
#include "transformer.hpp"
#include "util.hpp"

namespace RL {

/* ============================================================
 *  初始化缩放: 权重默认是 U(-1,1), 对高维输入会直接把激活顶到饱和
 *  (实测: d=1260 时 MLP 主干一次前向就把 Tanh 全推进 ±1, 见
 *   docs/agents_design.md §11.4; 对 ScaledConcat 更是直接毁掉门控)
 *  按 fan_in 缩放之后 pre-activation 的量级与维度无关。
 * ============================================================ */
inline void scaleFcInit(iFcLayer &fc)
{
    const float fanIn = (float)(fc.inputDim > 1 ? fc.inputDim : 1);
    const float s = 1.0f / std::sqrt(fanIn);
    for (std::size_t k = 0; k < fc.w.size(); k++) {
        fc.w[k] *= s;
    }
    for (std::size_t k = 0; k < fc.b.size(); k++) {
        fc.b[k] *= s;
    }
}

/* ============================================================
 *  MlpExpert — 把"普通 MLP"包装成一个 iLayer, 以便当专家
 *
 *  dIn -> dHidden -> dHidden -> dIn (输入输出同维, 才能做门控加权和)
 *  代价 ~ 2·dIn·dHidden, 比 TransformerBlock 专家便宜两个数量级。
 * ============================================================ */
class MlpExpert : public iLayer
{
public:
    Layer<Tanh> l1;
    Layer<Tanh> l2;
    Layer<Linear> l3;
    int dIn;
    int dHidden;

public:
    long long paramCount() const override
    {
        return l1.paramCount() + l2.paramCount() + l3.paramCount();
    }

    MlpExpert() : dIn(0), dHidden(0) {}

    MlpExpert(int dIn_, int dHidden_, bool withGrad)
        : l1(dIn_, dHidden_, true, withGrad),
          l2(dHidden_, dHidden_, true, withGrad),
          l3(dHidden_, dIn_, true, withGrad),
          dIn(dIn_), dHidden(dHidden_)
    {
        type = LAYER_FC;
        o = Tensor(dIn_, 1);
        e = Tensor(dIn_, 1);
    }

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        Tensor &h1 = l1.forward(x, inference);
        Tensor &h2 = l2.forward(h1, inference);
        o = l3.forward(h2, inference);
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /*
           链式往回走。注意 `Layer::backward` 会清掉它自己的 o/e, 所以要按
           "后层先回" 的顺序, 并且借用的中间激活 (l1.o / l2.o) 在那之前必须还活着。
        */
        Tensor &h1 = l1.o;   /* 引用: 前向缓存 */
        Tensor &h2 = l2.o;

        Tensor e2((std::size_t)dHidden, 1);
        Tensor e1((std::size_t)dHidden, 1);
        Tensor e0((std::size_t)dIn, 1);
        e2.zero();
        e1.zero();
        e0.zero();

        l3.e = e;                 /* 上一层注入的 dL/do 拷进来 */
        l3.backward(h2, e2);      /* e2 = dL/dh2 */
        l2.e = e2;
        l2.backward(h1, e1);      /* e1 = dL/dh1 */
        l1.e = e1;
        l1.backward(x, e0);       /* e0 = dL/dx */

        /* 累加进 ei (与 MOE::backward 的约定一致, 便于多个专家/门控路径叠加) */
        for (int j = 0; j < dIn; j++) {
            ei[j] += e0[j];
        }

        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        l1.SGD(lr);
        l2.SGD(lr);
        l3.SGD(lr);
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        l1.RMSProp(lr, rho, decay, clipGrad);
        l2.RMSProp(lr, rho, decay, clipGrad);
        l3.RMSProp(lr, rho, decay, clipGrad);
    }

    void Adam(float lr, float alpha, float beta, float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        l1.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        l2.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        l3.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
    }

    void clamp(float c0, float cn) override
    {
        l1.clamp(c0, cn);
        l2.clamp(c0, cn);
        l3.clamp(c0, cn);
    }

    /*
        ============================================================
         copyTo / softUpdateTo —— MlpExpert 原来**没有**这两个实现
        ============================================================
        后果非常隐蔽: `SparseMoE::copyTo` / `softUpdateTo` 都是把工作委托给专家的
        (`experts[i].copyTo(&p->experts[i])`), 而 `iLayer` 的默认实现是**空**的 ——
        于是"用 MlpExpert 当专家"时, 目标网络的专家权重根本不会被复制/软更新,
        门控与应用层却照常复制。实测 (复制前后 |Δw| 都是 1.57): 目标网络的专家
        一直停在它们自己那份随机初始化上, 且永远不随训练演进 —— 一个不报错、
        只让训练效果变差的静默 bug。`Layer<Fn>` 与 `TransformerBlock` 都有实现,
        所以只有 MlpExpert 这条路径中招 (而它正是本工程稀疏 MoE 的现役专家)。
    */
    void copyTo(iLayer* layer) override
    {
        MlpExpert *p = dynamic_cast<MlpExpert*>(layer);
        if (p == nullptr) {
            return;
        }
        l1.copyTo(&p->l1);
        l2.copyTo(&p->l2);
        l3.copyTo(&p->l3);
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        MlpExpert *p = dynamic_cast<MlpExpert*>(layer);
        if (p == nullptr) {
            return;
        }
        l1.softUpdateTo(&p->l1, alpha);
        l2.softUpdateTo(&p->l2, alpha);
        l3.softUpdateTo(&p->l3, alpha);
    }

    void write(std::ofstream &file) override
    {
        l1.write(file);
        l2.write(file);
        l3.write(file);
    }

    void read(std::ifstream &file) override
    {
        l1.read(file);
        l2.read(file);
        l3.read(file);
    }

    /* 初始化缩放: 权重默认是 U(-1,1), 对 1260 维输入会直接把 Tanh 顶到饱和 */
    void scaleInit()
    {
        iFcLayer *ls[3] = {&l1, &l2, &l3};   /* 都继承自 iFcLayer */
        for (int i = 0; i < 3; i++) {
            scaleFcInit(*ls[i]);
        }
    }
};

/*
 * 初始化缩放的三个重载 (调用方只写 `scaleExpertInit(e)` 一句, 具体怎么缩由类型决定)。
 * TransformerBlock 的注意力张量不是 iFcLayer, 得单独走 scaleTensor。
 */
inline void scaleExpertInit(MlpExpert &e)
{
    e.scaleInit();
}

template<typename Fn>
void scaleExpertInit(Layer<Fn> &e)
{
    scaleFcInit(e);
}

template<int H, int DFF>
void scaleExpertInit(TransformerBlock<H, DFF> &e)
{
    auto scaleTensor = [](Tensor &t, float s) {
        for (std::size_t k = 0; k < t.size(); k++) {
            t[k] *= s;
        }
    };

    /* FFN: fan_in 分别是 d_model 与 d_ff_ */
    scaleFcInit(e.ffn_up);
    scaleFcInit(e.ffn_down);

    /* 注意力: 每个 head 的 q/k/v 投影 fan_in 都是 d_model; 输出投影 fan_in = d_model */
    const float s = 1.0f / std::sqrt((float)(e.d_model > 1 ? e.d_model : 1));
    for (int i = 0; i < e.attn.numHeads; i++) {
        scaleTensor(e.attn.heads[i].wq, s);
        scaleTensor(e.attn.heads[i].wk, s);
        scaleTensor(e.attn.heads[i].wv, s);
    }
    scaleTensor(e.attn.wo, s);
}

/* ============================================================
 *  专家工厂: 各种专家的构造签名不同 (TransformerBlock 只要 d_model,
 *  MlpExpert 还要一个隐层宽度), 用 trait 把它统一掉。
 * ============================================================ */
template<typename E>
struct ExpertFactory;

template<int H, int DFF>
struct ExpertFactory<TransformerBlock<H, DFF> >
{
    static TransformerBlock<H, DFF> make(int d_model, int hidden, bool withGrad)
    {
        (void)hidden;   /* TransformerBlock 的 FFN 宽度由模板参数 DFF 决定 */
        return TransformerBlock<H, DFF>(d_model, withGrad);
    }
};

template<>
struct ExpertFactory<MlpExpert>
{
    static MlpExpert make(int d_model, int hidden, bool withGrad)
    {
        return MlpExpert(d_model, hidden > 0 ? hidden : (d_model / 16), withGrad);
    }
};

/*
 * 单层 FC + 激活的专家 (d -> d)。`hidden` 不适用 —— 它的"隐层"就是输出本身,
 * 所以这是一族**最便宜**的专家: 参数只有 d²+d, 而 MlpExpert 是 ~2·d·hidden。
 * 用途是"很多路很弱的专家 + 细粒度门控"这种配置 (ScaledConcat 的原始设想),
 * 现在可以在不换成另一种层的情况下直接用。
 */
template<typename Fn>
struct ExpertFactory<Layer<Fn> >
{
    static Layer<Fn> make(int d_model, int hidden, bool withGrad)
    {
        (void)hidden;
        return Layer<Fn>(d_model, d_model, true, withGrad);
    }
};

} /* namespace RL */

#endif // RL_EXPERT_HPP
