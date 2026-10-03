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

    /*
        ============================================================
        注意力头口径的**只读自检读数** (2026-09 dev-sacmoetb 新增)
        ============================================================
        为什么必须有: `MOE_TB_HEADS = 15` 曾经在 `d_model = 1263 = 3 x 421` 上被
        MultiHeadAttention 的"头数必须整除 d_model"规则**静默降级成 3 个头** ——
        参数指纹、paramCount、层类型序列、权重文件格式**一个都没变**, 只有"慢了 3 倍"
        和"80% 的 head 张量是死的"这两件事变了, 而面板上原本一个字都看不到。
        这与 TanhNorm<Sigmoid> 那次回归是同一类: **同形状、同参数量的静默替换**。

        所以把"请求了几个头 / 实际用了几个 / 每个多宽 / 分配了几个 / 注意力元素数"
        做成 iLayer 的**通用读数**: 容器型层 (SparseMoE / TransformerBlock) 往下委派,
        非注意力层返回 -1。上层不必 dynamic_cast 到具体模板就能印到自检面板上。
    */
    virtual int attnHeadsRequested() const { return -1; }   /* 模板/配置里请求的头数 */
    virtual int attnHeadsUsed() const { return -1; }        /* 真正参与前向的头数 */
    virtual int attnHeadDim() const { return -1; }          /* 每个头的 d_k */
    virtual int attnHeadsAllocated() const { return -1; }   /* 分配出来的 head 对象数 */
    virtual long long attnElements() const { return -1; }   /* numHeads * d_k^2 (单价来源) */

    /*
       ---- 权重读写: 参数是 `std::ostream/std::istream` (不是 ofstream/ifstream) ----
       为什么必须是通用流: [2026-10 用户口径] "对弈期间不保存模型权重" 要求后台训练那一轮
       的权重往返**改走内存** (见 weightio.hpp), 而内存版就是往 `std::ostringstream` 写、
       从 `std::istringstream` 读。签名收在 `ofstream/ifstream` 上就写不了内存版, 只能
       把整条往返复制一份 —— 那正是"两份实现迟早漂移"的形状。
       所有调用点传的都是 ofstream/ifstream (派生类 → 基类引用), 所以这次改动对它们
       是透明的; 写出来的**字节**也完全不变 (同一份代码, 只是流的类型不同)。
    */
    virtual void write(std::ostream &file){}
    virtual void read(std::istream &file){}
};

}
#endif // ILAYER_H
