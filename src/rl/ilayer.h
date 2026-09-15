#ifndef ILAYER_H
#define ILAYER_H
#include "tensor.hpp"
#include <memory>

namespace RL {

class iLayer
{
public:
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONCAT,
        LAYER_SCALEDCONCAT,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING,
        LAYER_ATTENTION,
        LAYER_SCALEDDOTPRODUCT,
        LAYER_MHA,
        LAYER_TRANSFORMERBLOCK,
        LAYER_MOE,
        LAYER_SSM,
        LAYER_MAMBA
    };
    using sptr = std::shared_ptr<iLayer>;
public:
    int type;
    Tensor o;
    Tensor e;
public:
    iLayer(){}
    virtual ~iLayer(){}
    virtual void initParams(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        return o;
    }
    virtual void backward(const Tensor& x, Tensor &ei){}
    virtual void cacheError(const Tensor &e){}
    virtual void SGD(float lr){}
    virtual void RMSProp(float lr, float rho, float decay, bool clipGrad){}
    virtual void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad){}
    virtual void clamp(float c0, float cn){}
    virtual void copyTo(iLayer* layer){}
    virtual void softUpdateTo(iLayer* layer, float alpha){}
    /*
       参数量 (只读诊断)。
       层自己知道它有多少个可训练标量, 上层就不必对每种层做 dynamic_cast 去拼凑 ——
       骨干每换一次就得重算一遍参数量的话, 没人会去维护那个数字。
       默认 0; 容器型层 (SparseMoE / MlpExpert) 自己重载并累加子层。
    */
    virtual long long paramCount() const { return 0; }

    /*
        ============================================================
         R1 (2026-09): 稀疏输出前向 —— 只算 idx 里那些输出下标的 logits
        ============================================================
        动机 (实测见 docs/training_optimization.md §9.4): 策略头是
        `64 -> 8100` 的一层, 权重 64x8100x4B = 2.07 MB; 而一次搜索模拟里它**要整块
        读一遍**, 只为了拿那几十个合法着法的概率。而搜索是访存带宽受限的 (P7),
        于是"把 8100 列里的 8060 列白读掉"直接吃掉了模拟吞吐。

        这里只算**激活前**的 logits, 激活由调用方按整向量的口径决定 —— 对 softmax
        头来说, "全量 softmax 后取子集再归一" 与 "在子集上直接 softmax" 数学上逐元素
        相等 (`exp` 的公共因子与减掉的公共最大值都会在归一化里约掉), 所以下游语义
        不变。注意这条等价性**只对 softmax 头成立**, 所以另有 subsetSoftmax() 声明它。

        返回 false = 这一层不支持稀疏输出 (调用方必须回退到全量 forward)。
        默认 false 是刻意的: 加了新的层类型却忘了实现时, 行为是"慢"而不是"错"。
    */
    virtual bool sparseLogits(const Tensor &x, const std::vector<int> &idx,
                              std::vector<float> &out) const
    {
        (void)x; (void)idx; (void)out;
        return false;
    }

    /*
        这一层**是否**支持 sparseLogits。与上一条分开是刻意的: 调用方要能在**跑骨干
        之前**就知道该走稀疏路径还是回退全量 (否则会白跑一次前向才发现头不支持)。
    */
    virtual bool supportsSparseLogits() const { return false; }

    /*
        该层的激活是不是"整向量 softmax"型 —— 即"对子集重新归一化"与"全量 softmax
        后取同一个子集再归一化"逐元素一致。它是 sparseLogits 那条捷径成立的**前提**,
        Net::sparseOutputSupported() 用它做兜底判断。
    */
    virtual bool subsetSoftmax() const { return false; }

    virtual void write(std::ofstream &file){}
    virtual void read(std::ifstream &file){}
};

}
#endif // ILAYER_H
