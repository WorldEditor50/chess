#ifndef ILAYER_H
#define ILAYER_H
#include "tensor.hpp"
#include <memory>
#include <cstdio>
#include <cstdlib>

namespace RL {

/*
   ============================================================
    未接线层 (死代码) 的守卫 —— 2026-10
   ============================================================
   仓库里有三个**从未被任何生产路径实例化**的层:
     * `PositionalEncoder` (attention.hpp)  —— 最早的"位置编码"尝试
     * `Attention<N>`      (attention.hpp)  —— "单向量 + 外积伪注意力", 见 tb 文档 §9
     * `Dropout<Fn>`       (layer.h)        —— 而且它的 mask 语义本身是错的, 见那里的注
   它们与上游 snakeAI 同源 (`docs/rl_sync.md`), 所以**保留不删**; 但"静默不生效的代码"
   在本工程被咬过多次 (旧 TB 的头数降级、`pe` 未分配越界写)。所以这里给一个**响亮失败**的守卫:

     * 默认**禁止**被调用 —— 一旦有人把它们接进 Net, 立刻 abort 并打印"要走 Transformer
       请用 seq_transformer.hpp";
     * 要研究这些层本身 (例如跑断言测试) 时, 显式 `setAllowUnwiredLayers(true)` (可 grep 的 opt-in)。

   判据见 `test_transformer [8]`: 默认禁止, 打开之后这些层**本身**仍然要用有限差分验证过。
*/
inline bool &unwiredLayersAllowed()
{
    static bool allowed = false;
    return allowed;
}
inline void setAllowUnwiredLayers(bool on) { unwiredLayersAllowed() = on; }

inline void requireWired(const char *who)
{
    if (unwiredLayersAllowed()) { return; }
    std::fprintf(stderr,
                 "[%s] 这个层**从未接线** (死代码), 而且历史上出过越界写/静默不生效的坑。\n"
                 "  => 不要把它接进 Net。要走 Transformer 请用 `src/rl/seq_transformer.hpp`\n"
                 "     的 `SeqTransformerExpert` (按格子 token 化 + 按 key 轴逐行 softmax)。\n"
                 "  => 确实要研究这个层本身, 先显式调用 `RL::setAllowUnwiredLayers(true)`。\n",
                 who);
    std::fflush(stderr);
    std::abort();
}

/*
   编译期探测"这个层有没有某个可选接口" (C++17 void_t)。
   用途: `SparseMoE` 要把"专家级旋钮"转发给每个专家, 但只有序列 Transformer 专家
   实现了它们; 用 if constexpr + 这两个 traits, MlpExpert / TransformerBlock /
   Layer<> 那些骨干一个字节都不动 (也不需要给它们补空实现)。
*/
template<typename T, typename = void>
struct HasPosModeByKey : std::false_type {};
template<typename T>
struct HasPosModeByKey<T, std::void_t<decltype(std::declval<T &>().setPosModeByKey(std::string()))> >
    : std::true_type {};

template<typename T, typename = void>
struct HasExpertDropout : std::false_type {};
template<typename T>
struct HasExpertDropout<T, std::void_t<decltype(std::declval<T &>().setExpertDropout(0.0f))> >
    : std::true_type {};

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
        LAYER_MAMBA,
        /*
           [2026-10] 真正的序列 Transformer 专家 (src/rl/seq_transformer.hpp)。
           与 LAYER_TRANSFORMERBLOCK 分开: 那个是"单向量 + 外积伪注意力"的旧实现,
           两者并不是同一种层 —— 混在一个 type 里会让诊断读数张冠李戴。
        */
        LAYER_SEQTRANSFORMER
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

    /*
       ============================================================
       [2026-10] 梯度范数 / 梯度缩放 (全局裁剪与"这次更新有多大信号"的读数)
       ============================================================
       为什么放在 iLayer 上: 全局范数裁剪要求**跨层**先把范数算出来、再统一等比缩小,
       而各层的梯度张量在各自的成员里 (iFcLayer 的 g.w/g.b、ScaledDotProduct 的
       g.wq/wk/wv、TransformerBlock 的 LN 与子层…)。没有这两个虚函数, 上层就只能
       dynamic_cast 到每一个具体模板 —— 那正是"新增一层就要记得改一圈"的形状。

       默认实现是**安全的"不报"**: `gradNorm2()` 返回 0, `scaleGrad()` 什么都不做。
       含义是"这一层没有梯度缓冲, 或者还没实现" —— 于是 `GRAD_CLIP_GLOBAL_NORM`
       的范数是"已实现层的范数之和", 而不是错的数。PPO 的 actor/critic 全是
       SparseMoE + iFcLayer, 覆盖是完整的 (见 ppo.h 的 gradClipMode 说明)。
    */
    virtual double gradNorm2() const { return 0.0; }
    virtual void scaleGrad(float s) { (void)s; }
    /*
       [2026-10] 专家级的两个可选旋钮 (只有序列 Transformer 专家实现)。
       为什么放在 iLayer 上而不是 dynamic_cast: 建网走的是 `makeMoeLayer` 工厂,
       返回的是 `iLayer::sptr`, 而专家藏在 `SparseMoE` 内部 —— 用虚函数转发一层
       (SparseMoE 覆写成"转发给每个专家") 就不必在每个调用点 dynamic_cast。
       默认实现是**空操作 + 返回 false** ⇒ 其它骨干 (TB/MLP/Layer) 完全不受影响。
    */
    virtual bool setPosModeByKey(const std::string &key) { (void)key; return false; }
    virtual bool setExpertDropout(float p) { (void)p; return false; }

    /*
       ============================================================
       [2026-10] 参数构成 (只读诊断): 注意力 / 非注意力 / 归一化 三块
       ============================================================
       默认实现把所有参数都算进 `ffn` (非注意力): 对 MLP / Layer<Gelu> 这些"整体就是一个
       前馈"的专家, 这就是准确的口径。`TransformerBlock` 覆盖它, 给出真正的三分法 ——
       实测 d_model=1710 时是 注意力 90.4% / FFN 9.5% / LN 0.05%, 而注意力那条支路
       在初始化时只贡献 1.45e-06 的输出方差 (见 rl/ppo.h 顶部那一节)。
       存在的理由与 gradNorm2 一样: 上层 (PPO / 自检面板) 不该 dynamic_cast 到每种专家。
    */
    virtual void paramBreakdown(long long &attn, long long &nonAttn, long long &norm) const
    {
        attn = 0;
        nonAttn = paramCount();
        norm = 0;
    }
};

/* 两个小工具: 给"实现了梯度缓冲"的层用, 免得每层各抄一遍循环 */
inline double gradNorm2Of(const Tensor &t)
{
    double s = 0.0;
    for (std::size_t i = 0; i < t.totalSize; i++) {
        const double v = t[i];
        s += v * v;
    }
    return s;
}

inline void scaleTensorGrad(Tensor &t, float s)
{
    for (std::size_t i = 0; i < t.totalSize; i++) {
        t[i] *= s;
    }
}

}
#endif // ILAYER_H
