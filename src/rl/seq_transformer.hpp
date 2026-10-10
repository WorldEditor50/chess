#ifndef RL_SEQ_TRANSFORMER_HPP
#define RL_SEQ_TRANSFORMER_HPP

/*
 * seq_transformer.hpp — **真正的** Transformer 专家 (2026-10)
 * ============================================================================
 * 背景 (为什么必须新写一个类, 而不是"修一修 TransformerBlock"):
 *
 *   `TransformerBlock` / `MultiHeadAttention` 收的是**一个 `(d_model × 1)` 向量**:
 *     * 没有 token 轴 ⇒ `z = q·kᵀ` 是 `d_k×d_k` 的**外积**, 它混合的是"特征维",
 *       不是"哪个位置该注意哪个位置";
 *     * 而 `softmax(z)` 是 `util.hpp:465` 那个**对整张张量求和**的全局 softmax
 *       ⇒ 每行只分到 `1/d_k` 的质量 ⇒ 单头输出被**恰好 d_k 倍**衰减
 *       (实测: d_k=114 时 `|attn_out|` 只有按行归一的 1/114, 这条支路只占块输出
 *        方差的 5.6e-07, 初始化时等于死的 —— 见 docs/tb_expert_training_2026_10.md §9)。
 *   这两件事都不是"调参能修"的: 前者是接口形态错了 (没有序列就没有 attention),
 *   后者是归一化轴错了。所以这个文件给出一条**新的**、可以独立验证的路:
 *
 *   ==== 本实现做的事 (标准 Pre-LN Transformer 编码器) ====
 *
 *   1. **把状态向量切成 token 序列**。本工程的状态编码是"平面 × 格"的平面优先
 *      (`state[plane * SeqLen + cell]`), 所以天然可以按**格子**切:
 *          第 t 个 token 的特征 = { state[0*SeqLen+t], state[1*SeqLen+t], ... }
 *      也就是"每个格子一个 token, 特征是 19 个平面在这一格上的取值"
 *      (featDim = dIn / SeqLen = 1710/90 = 19)。这是棋盘最自然的 token 化:
 *      90 个格子各自一个 token, 谁和谁相邻**由位置嵌入和注意力自己学**。
 *
 *   2. **位置嵌入 (可学, SeqLen × tokDim)**: 没有它, attention 对 token 置换是等变的,
 *      棋盘就被当成无序的格子袋 —— 这正是旧实现的结构性缺陷。开关 `usePos` 保留,
 *      是为了让测试能验证"关掉位置嵌入 ⇒ 对置换等变"这条不变量。
 *
 *   3. **标准多头注意力**: 每个头 `Q,K,V = X·Wᵀ`, `scores = Q Kᵀ / √d_k` 是 `(T × T)`,
 *      **按行** (对 key 轴) softmax, `O = P·V`。归一化轴是 key 轴 —— 这是"attention"
 *      这个词的全部内容: 每个 query 在 key 上分一个**概率分布**。
 *
 *   4. **Pre-LN + 残差**, 堆 `Blocks` 层: `X ← X + MHA(LN(X))`, `X ← X + FFN(LN(X))`。
 *
 *   5. **逐格输出投影**: `tokDim → featDim`, 再按平面优先的顺序散回 `(dIn × 1)`,
 *      于是它仍然满足"专家输入输出同维"的 MoE 契约 (可以和别的专家互换)。
 *
 *   ==== 与旧实现的关键差别 (都是可测的) ====
 *
 *   | | 旧 `TransformerBlock` | 本类 |
 *   |---|---|---|
 *   | 输入 | 单个向量 | **SeqLen 个 token** |
 *   | 注意力矩阵 | `q·kᵀ` 外积 (d_k×d_k) | `Q·Kᵀ` (T×T) |
 *   | 归一化 | **全局** softmax (整张 d_k²) | **按行** softmax (key 轴) |
 *   | 位置信息 | 无 (`PositionalEncoder` 从未被实例化) | 可学的位置嵌入 |
 *   | 深度 | 1 层 | `Blocks` 层 |
 *   | 输出量级 | ∝ 1/d_k (与头数耦合) | 与头数无关 |
 *
 *   ==== 参数与代价 (默认 PPO 口径: dIn=1710, SeqLen=90, tokDim=64, Heads=4, Blocks=2) ====
 *   每个专家 ≈ 92 K 参数 (旧 TB 专家是 12.9 M), 前向 ≈ 9.7 M MAC (旧的是 13.3 M) ——
 *   也就是"更小、更快、而且是对的"。具体读数由 `test_transformer` 与
 *   `docs/tb_expert_training_2026_10.md` §10 给出。
 *
 *   ==== 验证 (不是"看起来对") ====
 *     * 逐参数组的**有限差分**梯度审计 (embed / 位置嵌入 / wq,wk,wv,wo / LN γ,β /
 *       FFN / 输出投影) —— `test_transformer` [1];
 *     * 不变量: 关掉位置嵌入时对 token 置换**等变**、全 token 相同时输出逐 token 相同
 *       (后者等价于"每一行都是概率分布", 全局 softmax 会破坏它) —— `test_transformer` [2];
 *     * 权重往返 (write → read → 同输入同输出) —— `test_transformer` [3]。
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "ilayer.h"
#include "layer.h"
#include "optimize.h"
#include "util.hpp"
#include "expert.hpp"   /* ExpertFactory / scaleFcInit (本类在文件末尾注册进去) */

namespace RL {

/* 张量线性插值 (软更新用)。放在文件作用域是为了让 SeqLayerNorm 与专家共用一份 ——
   它原来是专家里的一个 private static, 抽类之后两边都要用。 */
inline void seqLerpTensor(Tensor &dst, const Tensor &src, float alpha)
{
    for (std::size_t i = 0; i < dst.totalSize && i < src.totalSize; i++) {
        dst[i] = (1.0f - alpha) * dst[i] + alpha * src[i];
    }
}

/* ============================================================
 *  SeqLayerNorm —— 序列上的 LayerNorm (逐 token 在**特征维**上归一)
 * ============================================================
 *      Y[t][j] = γ[j] · (X[t][j] − μ_t) / σ_t + β[j]
 *      μ_t, σ_t: 第 t 个 token 在 j 维上的均值 / 标准差 (σ_t = √(var_t + eps))
 *
 *  为什么把它单独一个类 (而不是像旧 `TransformerBlock` 那样把 LN 写进块里):
 *    * 一个块有 2 个 LN、专家有 `Blocks` 个块 ⇒ 现在要维护 4·Blocks 组 (γ,β,g,v,m),
 *      并且有 **10 处**"新增参数就要顺手改"的接线 (构造 / 前向 / 反向 / 三个优化器 /
 *      clamp / gradNorm2 / scaleGrad / paramCount / paramBreakdown / write / read /
 *      copyTo / softUpdateTo)。抽成类之后每一处都只剩一行循环。
 *    * 它现在可以被**单独验证** (均值/方差不变量、有限差分、累加语义、往返),
 *      而不是只能通过整个专家的有限差分间接看到 —— 见 `test_transformer` [5]。
 *    * 将来的第二个序列层可以直接复用 (真正需要第二个调用方时, 把它挪到自己的头文件
 *      也只是 5 分钟的事 —— 本工程的惯例是"第二份拷贝出现时才抽文件", 见 expert.hpp)。
 *
 *  为什么**不**直接复用 `layer.h` 的 `LayerNorm`: 那些是 `iFcLayer`, 接口是
 *  "(D×1) 向量进 / (D×1) 向量出"。要跑 (T×D) 序列就得逐 token 调它, 代价是 T 次张量
 *  拷贝 **+ 反向里为每个 token 重跑一次前向** (`Layer<Fn>::backward` 用自己缓存的 `o`,
 *  一个对象被 T 个 token 共用 —— 就是本文档 D8 那条)。FC 层那样做值得 (省掉一整份
 *  FC/激活反向), 而 LN 的反向只有 20 行、"复用"换不来什么, 拷贝/重跑却是实打实的。
 *
 *  ⚠ 两条契约 (与专家里的调用方一致, 写错不报错只出错梯度):
 *    1. `forward(X, Y)`: **Y 必须已经按 (T × dim) 分配好** —— 本层不做分配,
 *       因为它在每个 block 的每一步前向里都会被调用;
 *    2. `backward(X, dY, dX)`: `dX` 是**累加** (`+=`) 而不是赋值 —— 残差与 LN
 *       两条路都写同一个缓冲 (pre-LN 的定义就是这样)。`test_transformer [5c]` 把它钉住。
 *  反向里 μ/σ 是**重算**的 (同样的 float 运算顺序 ⇒ 逐位相同): 少一处"每 token 一个
 *  μ/σ"的状态, 而 LN 的成本在这条链上可以忽略。
 */
class SeqLayerNorm
{
public:
    int dim;             /* 特征维 (归一发生在这一维上) */
    double eps;          /* √(var+eps)。**double** 是为了与抽类前的 `var + 1e-5` 逐位相同
                            (float 的 1e-5f 转成 double 是 1.0000000116860974e-05, 会在
                            末位咬人) */
    Tensor gamma, beta;  /* 参数 */
    Tensor gG, gB;       /* 梯度 */
    Tensor vG, vB;       /* RMSProp 状态 */
    Tensor mG, mB;       /* Adam 状态 */

    SeqLayerNorm() : dim(0), eps(1e-5) {}
    SeqLayerNorm(int dim_, bool withGrad) : dim(0), eps(1e-5) { init(dim_, withGrad); }

    void init(int dim_, bool withGrad)
    {
        dim = (dim_ > 0) ? dim_ : 1;
        gamma = Tensor((std::size_t)dim, 1);
        beta  = Tensor((std::size_t)dim, 1);
        gamma.fill(1.0f);      /* 标准初值: γ=1, β=0 (不是随机值) */
        beta.fill(0.0f);
        if (withGrad) {
            gG = Tensor((std::size_t)dim, 1);
            gB = Tensor((std::size_t)dim, 1);
            vG = Tensor((std::size_t)dim, 1);
            vB = Tensor((std::size_t)dim, 1);
            mG = Tensor((std::size_t)dim, 1);
            mB = Tensor((std::size_t)dim, 1);
        }
    }

    bool hasGrad() const { return gG.totalSize == (std::size_t)dim; }
    long long paramCount() const { return 2LL * (long long)dim; }
    /* 由缓冲区长度推出 token 数 —— 这一层不需要知道 SeqLen */
    int tokensIn(const std::vector<float> &X) const
    {
        return (int)(X.size() / (std::size_t)dim);
    }

    /* 前向: Y = γ ⊙ (X−μ)/σ + β   (Y 必须已按 T×dim 分配, 见类头注释)
       ⚠ 这里的精度混合是**原样保留**抽类前 `lnFwd` 的写法: μ/var 用 double 累加,
       `(xp[j] − m) / s` 是 **double 除法** 再舍入到 float —— 而反向里的 x̂ 是
       float 重算 (`(xp[j] − mu) / r`)。两处并不逐位相同 (旧注释说的"同序"其实只对
       反向成立), 但改它就是改数值, 不是重构。要动的话必须单独一轮 A/B。 */
    void forward(const std::vector<float> &X, std::vector<float> &Y) const
    {
        const int T = tokensIn(X);
        for (int t = 0; t < T; t++) {
            const float *xp = &X[(std::size_t)(t * dim)];
            float *yp = &Y[(std::size_t)(t * dim)];
            double m = 0;
            for (int j = 0; j < dim; j++) { m += xp[j]; }
            m /= (double)dim;
            double var = 0;
            for (int j = 0; j < dim; j++) { const double d = xp[j] - m; var += d * d; }
            var /= (double)dim;
            const float s = (float)std::sqrt(var + eps);
            for (int j = 0; j < dim; j++) {
                yp[j] = gamma[(std::size_t)j] * (float)((xp[j] - m) / s) + beta[(std::size_t)j];
            }
        }
    }

    /* 反向: dY → dX (**累加**), 并累加 γ/β 的梯度 */
    void backward(const std::vector<float> &X, const std::vector<float> &dY,
                  std::vector<float> &dX)
    {
        const int T = tokensIn(X);
        const float invN = 1.0f / (float)dim;
        for (int t = 0; t < T; t++) {
            const float *xp = &X[(std::size_t)(t * dim)];
            const float *dyp = &dY[(std::size_t)(t * dim)];
            /* 重算 μ/σ (与 forward 同一串 float 运算 ⇒ 逐位相同) */
            double m = 0;
            for (int j = 0; j < dim; j++) { m += xp[j]; }
            m /= (double)dim;
            double var = 0;
            for (int j = 0; j < dim; j++) { const double d = xp[j] - m; var += d * d; }
            var /= (double)dim;
            const float r = (float)std::sqrt(var + eps);
            const float mu = (float)m;

            float sumDy = 0, sumDyXh = 0;
            for (int j = 0; j < dim; j++) {
                const float xh = (xp[j] - mu) / r;
                sumDy += dyp[j];
                sumDyXh += dyp[j] * xh;
            }
            const float meanDy = sumDy * invN, meanDyXh = sumDyXh * invN;
            for (int j = 0; j < dim; j++) {
                const float xh = (xp[j] - mu) / r;
                dX[(std::size_t)(t * dim + j)] +=
                    (gamma[(std::size_t)j] / r) * (dyp[j] - meanDy - xh * meanDyXh);
                if (hasGrad()) {
                    gG[(std::size_t)j] += dyp[j] * xh;
                    gB[(std::size_t)j] += dyp[j];
                }
            }
        }
    }

    void zeroGrad() { if (hasGrad()) { gG.zero(); gB.zero(); } }
    void SGD(float lr)
    {
        if (!hasGrad()) { return; }
        Optimize::SGD(gamma, gG, lr);
        Optimize::SGD(beta, gB, lr);
        zeroGrad();
    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad)
    {
        if (!hasGrad()) { return; }
        Optimize::RMSProp(gamma, vG, gG, lr, rho, decay, clipGrad);
        Optimize::RMSProp(beta, vB, gB, lr, rho, decay, clipGrad);
        zeroGrad();
    }
    void Adam(float lr, float alpha, float beta_, float alpha_, float beta__,
              float decay, bool clipGrad)
    {
        if (!hasGrad()) { return; }
        Optimize::Adam(gamma, vG, mG, gG, alpha_, beta__, lr, alpha, beta_, decay, clipGrad);
        Optimize::Adam(beta, vB, mB, gB, alpha_, beta__, lr, alpha, beta_, decay, clipGrad);
        zeroGrad();
    }
    void clamp(float c0, float cn)
    {
        Optimize::clamp(gamma, c0, cn);
        Optimize::clamp(beta, c0, cn);
    }
    double gradNorm2() const
    {
        return hasGrad() ? (gradNorm2Of(gG) + gradNorm2Of(gB)) : 0.0;
    }
    void scaleGrad(float s)
    {
        if (!hasGrad()) { return; }
        scaleTensorGrad(gG, s);
        scaleTensorGrad(gB, s);
    }
    void copyTo(SeqLayerNorm *p) const
    {
        if (p == nullptr) { return; }
        p->gamma = gamma;
        p->beta = beta;
    }
    void softUpdateTo(SeqLayerNorm *p, float alpha) const
    {
        if (p == nullptr) { return; }
        seqLerpTensor(p->gamma, gamma, alpha);
        seqLerpTensor(p->beta, beta, alpha);
    }
    /*
       权重往返: **顺序必须是 γ 然后 β** —— 与"抽类之前"逐字节一致。
       这一条不是洁癖: 顺序变了, 存量权重会被错位地读进来 (θ 对不上而参数量对得上,
       往返测试自己也看不出来)。`test_transformer [3]` 的 FNV 指纹是那条证据。
    */
    void write(std::ostream &f) const
    {
        f << gamma.toString() << std::endl;
        f << beta.toString() << std::endl;
    }
    void read(std::istream &f)
    {
        std::string s;
        std::getline(f, s); gamma = Tensor::fromString(s);
        std::getline(f, s); beta = Tensor::fromString(s);
    }
};


/* ============================================================
 *  SeqParam —— "自己管优化器状态"的一组参数 (val/g/v/m)
 * ============================================================
 * 原来是 `SeqTransformerExpert` 里的内嵌 `Param` (只给位置嵌入用)。位置编码抽成
 * `SeqPosEnc` 之后它必须提到文件作用域 —— 与 `SeqLayerNorm` 的 g/v/m 同一形态,
 * 但 LN 那组是 2 个定长向量、直接写了成员, 而位置嵌入的**组数随模式变** (1D 一组、
 * 2D 两组), 所以这里保留"可变数量的 Param"。
 */
struct SeqParam
{
    Tensor val, g, v, m;
    SeqParam() {}
    void alloc(int n, bool withGrad)
    {
        val = Tensor((std::size_t)n, 1);
        if (withGrad) {
            g = Tensor((std::size_t)n, 1);
            v = Tensor((std::size_t)n, 1);
            m = Tensor((std::size_t)n, 1);
        }
    }
    void zeroGrad() { if (g.totalSize > 0) { g.zero(); } }
    long long count() const { return (long long)val.totalSize; }
};

/* ============================================================
 *  SeqPosEnc —— 位置编码 (2026-10: 补"未做的结构项"里的 2D 位置嵌入 + RoPE)
 * ============================================================
 *  三种模式, **默认 LEARN_1D 与改动前逐位相同** (同样的分配、同样的 RNG 顺序、
 *  同样的加法循环):
 *
 *    LEARN_1D  可学 `(SeqLen × tokDim)`                        ← 默认
 *    LEARN_2D  可学 行 `(10 × tokDim)` + 列 `(9 × tokDim)`      ← 棋盘是 10 行 × 9 列
 *    ROPE      无参数: 对 Q/K 做旋转变换 (相对位置)
 *
 *  为什么 2D【设计理由】: 1D 的 90 个位置向量彼此**完全独立**, 相邻格之间没有任何
 *  参数共享; 2D 把参数量从 `90×64 = 5760` 降到 `19×64 = 1216`, 并把"同一行/同一列"
 *  变成共享结构 —— 棋盘的行列是先验, 让模型从数据里学出它显然更贵。
 *  ⚠ 代价 (诚实记下): 位置向量 = `row[r] + col[c]`, 两格 `(r,c)` 与 `(r′,c′)` 的向量
 *  相同当且仅当 `row[r]−row[r′] = col[c′]−col[c]` —— 这**不只**在同行同列发生,
 *  所以 2D 确实可能给不同格子相同的位置向量 (1D 不会)。这是参数共享的代价,
 *  不是信息丢失; 哪个更好只能靠读数 (§11 的 `--pos=` A/B) 回答, 不要先验下结论。
 *
 *  为什么 RoPE【设计理由】: 把位置编码成 Q/K 上的旋转 ⇒ 注意力分数只依赖**相对位置**
 *  `i−j`, 没有位置参数, 也不占 token 宽度。实测不变量见 `test_transformer [6]`
 *  ("整段平移后重叠区的注意力逐位不变")。
 *
 *  ⚠ 权重文件**自描述** (为什么必须): LEARN_1D 写出的字节保持原样 (一行张量) ——
 *  `test_transformer [3]` 的 FNV 指纹因此不变; 另外两种模式先写一行标记
 *  (`POS2D` / `POSROPE`), `read` 按"下一行首字符是不是字母"判格式并与专家当前模式
 *  **比对, 不一致就响亮失败**。"改了位置编码却不改权重格式 ⇒ 存量权重被静默错位读入"
 *  正是 tb 文档 §9.3 方案 B 那个坑, 这里用同一套办法挡住。
 */
class SeqPosEnc
{
public:
    enum Mode { LEARN_1D = 0, LEARN_2D = 1, ROPE = 2 };

    static const char *modeKey(Mode m)
    {
        return (m == LEARN_2D) ? "2d" : ((m == ROPE) ? "rope" : "1d");
    }
    static const char *modeName(Mode m)
    {
        return (m == LEARN_2D) ? "可学 2D (行+列)"
             : ((m == ROPE) ? "RoPE (无参数, 旋转 Q/K)" : "可学 1D (默认)");
    }
    static bool parseMode(const std::string &s, Mode &out)
    {
        if (s == "1d" || s == "learn" || s == "learn1d") { out = LEARN_1D; return true; }
        if (s == "2d" || s == "learn2d")                 { out = LEARN_2D; return true; }
        if (s == "rope")                                 { out = ROPE;     return true; }
        return false;
    }

    Mode mode;
    int seqLen, tokDim, rows, cols, dk;
    std::vector<SeqParam> emb;      /* LEARN_1D: 1 组 (T×D); LEARN_2D: 2 组 (行, 列) */
    std::vector<float> ropeCos, ropeSin;   /* ROPE: (T × dk/2) 的旋转表 */

    SeqPosEnc() : mode(LEARN_1D), seqLen(0), tokDim(0), rows(0), cols(0), dk(0) {}

    /*
       分配 + 初值。**RNG 顺序与抽类前一致**: LEARN_1D 只画一次
       `Random::uniform(val, -0.02, 0.02)` (5760 个数, 与原来的 `pos.val` 同一处),
       LEARN_2D 画两次 (先行后列)。ROPE 不画 —— 它没有参数。
    */
    void init(int seqLen_, int tokDim_, int rows_, int cols_, Mode m, bool withGrad, int dk_)
    {
        mode = m;
        seqLen = (seqLen_ > 0) ? seqLen_ : 1;
        tokDim = (tokDim_ > 0) ? tokDim_ : 1;
        rows = (rows_ > 0) ? rows_ : 1;
        cols = (cols_ > 0) ? cols_ : 1;
        dk = (dk_ > 0) ? dk_ : 1;
        emb.clear();
        ropeCos.clear();
        ropeSin.clear();
        if (m == LEARN_1D) {
            emb.push_back(SeqParam());
            emb[0].alloc(seqLen * tokDim, withGrad);
            Random::uniform(emb[0].val, -0.02f, 0.02f);
        } else if (m == LEARN_2D) {
            emb.push_back(SeqParam());
            emb.push_back(SeqParam());
            emb[0].alloc(rows * tokDim, withGrad);   /* 行 */
            emb[1].alloc(cols * tokDim, withGrad);   /* 列 */
            Random::uniform(emb[0].val, -0.02f, 0.02f);
            Random::uniform(emb[1].val, -0.02f, 0.02f);
        } else {
            buildRope();
        }
    }

    /* 与专家原来的 `scaleInit` 里那一步同一条: 小随机再乘 1/√tokDim */
    void scaleInit(float tokDimF)
    {
        const float s = 1.0f / std::sqrt((tokDimF > 1.0f) ? tokDimF : 1.0f);
        for (std::size_t p = 0; p < emb.size(); p++) {
            for (std::size_t i = 0; i < emb[p].val.totalSize; i++) { emb[p].val[i] *= s; }
        }
    }

    bool hasParams() const { return !emb.empty(); }
    long long paramCount() const
    {
        long long t = 0;
        for (std::size_t p = 0; p < emb.size(); p++) { t += emb[p].count(); }
        return t;
    }
    int groupCount() const { return (int)emb.size(); }
    std::string describe() const
    {
        std::string s = modeName(mode);
        if (mode == LEARN_2D) {
            s += " 行 " + std::to_string(rows) + " + 列 " + std::to_string(cols);
        } else if (mode == LEARN_1D) {
            s += " " + std::to_string(seqLen) + " 个位置";
        } else {
            s += " base=10000, dk/2=" + std::to_string(dk / 2);
        }
        if (hasParams()) { s += ", 参数 " + std::to_string(paramCount()); }
        return s;
    }

    /* ---- 可学模式的加/反传 (LEARN_*) ---- */
    void forwardAdd(std::vector<float> &X) const
    {
        if (mode == LEARN_1D) {
            const Tensor &p = emb[0].val;
            for (std::size_t i = 0; i < p.totalSize && i < X.size(); i++) { X[i] += p[i]; }
        } else if (mode == LEARN_2D) {
            const Tensor &pr = emb[0].val;   /* 行 (rows × tokDim) */
            const Tensor &pc = emb[1].val;   /* 列 (cols × tokDim) */
            for (int t = 0; t < seqLen; t++) {
                const int r = t / cols, c = t % cols;      /* cell = row*9 + col */
                for (int j = 0; j < tokDim; j++) {
                    X[(std::size_t)(t * tokDim + j)] +=
                        pr[(std::size_t)(r * tokDim + j)] + pc[(std::size_t)(c * tokDim + j)];
                }
            }
        }
        /* ROPE: 不改输入 —— 它在注意力里对 Q/K 生效 */
    }
    void backwardAdd(const std::vector<float> &dX)
    {
        if (mode == LEARN_1D) {
            for (std::size_t i = 0; i < emb[0].g.totalSize && i < dX.size(); i++) { emb[0].g[i] += dX[i]; }
        } else if (mode == LEARN_2D) {
            for (int t = 0; t < seqLen; t++) {
                const int r = t / cols, c = t % cols;
                for (int j = 0; j < tokDim; j++) {
                    const float d = dX[(std::size_t)(t * tokDim + j)];
                    emb[0].g[(std::size_t)(r * tokDim + j)] += d;
                    emb[1].g[(std::size_t)(c * tokDim + j)] += d;
                }
            }
        }
    }

    /* ---- RoPE ---- */
    void buildRope()
    {
        const int half = dk / 2;
        ropeCos.assign((std::size_t)(seqLen * half), 1.0f);
        ropeSin.assign((std::size_t)(seqLen * half), 0.0f);
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < half; i++) {
                /* θ_i = base^(-2i/dk), base = 10000 (标准 RoPE) */
                const double theta = (double)t / std::pow(10000.0, (2.0 * i) / (double)dk);
                ropeCos[(std::size_t)(t * half + i)] = (float)std::cos(theta);
                ropeSin[(std::size_t)(t * half + i)] = (float)std::sin(theta);
            }
        }
    }
    /* 对 Q/K 就地旋转: (t, head h, pair i) 上的 (2i, 2i+1) 两个分量 */
    void ropeApply(std::vector<float> &Q, std::vector<float> &K, int heads, int dk_, bool inverse) const
    {
        if (mode != ROPE || ropeCos.empty()) { return; }
        const int half = dk_ / 2;
        for (int t = 0; t < seqLen; t++) {
            for (int h = 0; h < heads; h++) {
                const std::size_t base = (std::size_t)(t * tokDim + h * dk_);
                for (int i = 0; i < half; i++) {
                    const float c = ropeCos[(std::size_t)(t * half + i)];
                    const float s = ropeSin[(std::size_t)(t * half + i)];
                    const std::size_t i0 = base + (std::size_t)(2 * i);
                    const std::size_t i1 = i0 + 1;
                    for (int pass = 0; pass < 2; pass++) {
                        std::vector<float> &X = (pass == 0) ? Q : K;
                        if (i1 >= X.size()) { continue; }
                        const float a = X[i0], b = X[i1];
                        if (!inverse) {
                            X[i0] = a * c - b * s;
                            X[i1] = a * s + b * c;
                        } else {
                            /* 旋转的转置 (= 反向传播) */
                            X[i0] = a * c + b * s;
                            X[i1] = -a * s + b * c;
                        }
                    }
                }
            }
        }
    }

    /* ---- 优化器 / 梯度 / 裁剪 / 软更新 ---- */
    void zeroGrad() { for (std::size_t p = 0; p < emb.size(); p++) { emb[p].zeroGrad(); } }
    void SGD(float lr)
    {
        for (std::size_t p = 0; p < emb.size(); p++) {
            if (emb[p].g.totalSize == 0) { continue; }
            Optimize::SGD(emb[p].val, emb[p].g, lr);
            emb[p].zeroGrad();
        }
    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad)
    {
        for (std::size_t p = 0; p < emb.size(); p++) {
            if (emb[p].g.totalSize == 0) { continue; }
            Optimize::RMSProp(emb[p].val, emb[p].v, emb[p].g, lr, rho, decay, clipGrad);
            emb[p].zeroGrad();
        }
    }
    void Adam(float lr, float alpha, float beta, float alpha_, float beta_, float decay, bool clipGrad)
    {
        for (std::size_t p = 0; p < emb.size(); p++) {
            if (emb[p].g.totalSize == 0) { continue; }
            Optimize::Adam(emb[p].val, emb[p].v, emb[p].m, emb[p].g, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
            emb[p].zeroGrad();
        }
    }
    void clamp(float c0, float cn)
    {
        for (std::size_t p = 0; p < emb.size(); p++) { Optimize::clamp(emb[p].val, c0, cn); }
    }
    double gradNorm2() const
    {
        double s = 0;
        for (std::size_t p = 0; p < emb.size(); p++) {
            if (emb[p].g.totalSize > 0) { s += gradNorm2Of(emb[p].g); }
        }
        return s;
    }
    void scaleGrad(float s)
    {
        for (std::size_t p = 0; p < emb.size(); p++) {
            if (emb[p].g.totalSize > 0) { scaleTensorGrad(emb[p].g, s); }
        }
    }
    void copyTo(SeqPosEnc *p) const
    {
        if (p == nullptr || p->emb.size() != emb.size()) { return; }
        for (std::size_t i = 0; i < emb.size(); i++) { p->emb[i].val = emb[i].val; }
    }
    void softUpdateTo(SeqPosEnc *p, float alpha) const
    {
        if (p == nullptr || p->emb.size() != emb.size()) { return; }
        for (std::size_t i = 0; i < emb.size(); i++) {
            seqLerpTensor(p->emb[i].val, emb[i].val, alpha);
        }
    }

    /*
       权重往返 + **模式自描述**。
       标记行的首字符是字母 ⇒ 与张量行 (`shape|CRC:b64`, 行首是数字) 不可能混淆。
       LEARN_1D 不写标记 —— 这样默认路径的字节与改动前完全一致 (FNV 指纹不变)。
    */
    void write(std::ostream &f) const
    {
        if (mode == LEARN_2D)        { f << "POS2D" << std::endl; }
        else if (mode == ROPE)       { f << "POSROPE" << std::endl; }
        for (std::size_t p = 0; p < emb.size(); p++) { f << emb[p].val.toString() << std::endl; }
    }
    void read(std::istream &f)
    {
        std::string s;
        if (!std::getline(f, s)) { fail("文件在位置编码这一段之前就结束了"); }
        Mode fileMode = LEARN_1D;
        std::string first = s;
        if (!first.empty() && std::isalpha((unsigned char)first[0])) {
            if (first == "POS2D")        { fileMode = LEARN_2D; }
            else if (first == "POSROPE") { fileMode = ROPE; }
            else { fail("不认识的位置编码标记: " + first); }
            if (!std::getline(f, s)) { fail("标记 " + first + " 之后没有张量行"); }
        }
        if (fileMode != mode) {
            fail(std::string("权重文件里的位置编码是 ") + modeKey(fileMode)
                 + ", 但专家配的是 " + modeKey(mode)
                 + " —— 结构不一致, 继续读下去会把后面的权重全部错位");
        }
        if (hasParams()) {
            emb[0].val = Tensor::fromString(s);
            if (emb.size() > 1) {
                if (!std::getline(f, s)) { fail("2D 位置编码缺少列嵌入那一行"); }
                emb[1].val = Tensor::fromString(s);
            }
        }
    }

private:
    /* 响亮失败: 结构不一致继续读下去 = 静默错位, 比崩掉危险得多 */
    static void fail(const std::string &why)
    {
        std::fprintf(stderr,
                     "[SeqPosEnc] 位置编码读入失败: %s\n"
                     "  ⇒ 位置编码模式必须与权重文件一致 (1d / 2d / rope)。\n"
                     "     要让权重与配置对齐, 要么用同样的 --pos 重新训练, 要么换回那个模式。\n",
                     why.c_str());
        std::fflush(stderr);
        std::abort();
    }
};


template<int SeqLen, int Heads, int Blocks>
class SeqTransformerExpert : public iLayer
{
public:
    /*
       [2026-10] 位置嵌入的"参数组"搬到了文件作用域的 `SeqParam` / `SeqPosEnc`
       (原来这里是内嵌的 `Param` + 五个自由函数, 只给位置嵌入用)。
    */

    /* 每一层的中间量缓存 (前向存、反向用) */
    struct BlockCache {
        std::vector<float> h1, q, k, v, att, m, x1, h2, fh, f;
        std::vector<float> p;          /* H 个头各自的 (T×T) 注意力概率, 行优先 */
        /* dropout 的 mask (只在 dropP > 0 时分配, 默认空 ⇒ 内存/行为与改动前一致) */
        std::vector<float> maskP;      /* (H × T × T) 注意力概率的 mask */
        std::vector<float> maskM, maskF;  /* (T × tokDim) 两条残差支路的 mask */
        float mu1, r1, mu2, r2;
        BlockCache() : mu1(0), r1(1), mu2(0), r2(1) {}
    };

public:
    int dIn, featDim, tokDim, dff, dk, heads, blocks;
    /* [测试用] 关掉它 ⇒ 对 token 置换等变 (没有位置信息) */
    bool usePos = true;

    Layer<Linear> embed;                       /* featDim -> tokDim */
    Layer<Linear> outProj;                     /* tokDim  -> featDim */
    std::vector<Layer<Linear> > wq, wk, wv, wo;/* 每层: tokDim -> tokDim (多头共享投影) */
    std::vector<Layer<Gelu> > ffnUp;           /* 每层: tokDim -> dff (激活在 Layer 内部) */
    std::vector<Layer<Linear> > ffnDown;       /* 每层: dff    -> tokDim */
    /*
       [2026-10] 两个 LayerNorm 由 `SeqLayerNorm` 承担 (它自己管 γ/β 与 g/v/m)。
       抽类之前这里是 `std::vector<Param> ln1g, ln1b, ln2g, ln2b;` 四份 —— 于是
       "加一个 LN" 要在 10 处接线里各改一遍; 现在每处只剩一行循环。
    */
    std::vector<SeqLayerNorm> ln1, ln2;
    /*
       位置编码 (2026-10 由"一个 Param + 五个自由函数"抽成 `SeqPosEnc`):
       三种模式 (1D 可学 / 2D 行列 / RoPE), 默认 1D = 与改动前逐位相同。
       见文件顶部的 `SeqPosEnc` 说明与 §9.3 的"未做结构项"。
    */
    SeqPosEnc pos;

    /* 棋盘的行列 (2D 位置编码用): 10 行 × 9 列 = 90, 与 `canonicalCell(x,y)=x*9+y` 对齐 */
    static const int POS_ROWS = 10;
    static const int POS_COLS = 9;

    /*
       dropout 概率 (默认 0 = 关闭)。见 `setDropout` 的说明: 关闭时**不消耗随机数、
       不分配缓冲**, 所以默认路径与改动前逐位相同。打开后训练用 `inference=false`,
       推理用 `inference=true` (推理不做 dropout, 不需要任何缩放 —— 倒置式)。
    */
    float dropP = 0.0f;

    std::vector<BlockCache> blk;

    /* 中间量 (float 缓冲: 这一层的所有张量都是 (T × D), 用显式下标最不容易出错) */
    std::vector<float> Xs;      /* (T × featDim) 输入序列 */
    std::vector<float> CUR;     /* (T × tokDim)  当前残差流 */
    std::vector<float> FIN;     /* (T × tokDim)  最后一层的输出 (= outProj 的输入) */
    std::vector<float> YS;      /* (T × featDim) 输出序列 */
    std::vector<float> dCUR, dYS, dTMP, dTMP2;
    std::vector<float> dQv, dKv, dVv, dXs;

    /* 逐 token 调 Layer<> 用的临时张量 (按维数取值, 见 scrOf) */
    Tensor scIn[3], scEi[3], scE[3];

public:
    long long paramCount() const override
    {
        long long t = embed.paramCount() + outProj.paramCount() + pos.paramCount();
        for (int b = 0; b < blocks; b++) {
            t += wq[(std::size_t)b].paramCount() + wk[(std::size_t)b].paramCount()
               + wv[(std::size_t)b].paramCount() + wo[(std::size_t)b].paramCount()
               + ffnUp[(std::size_t)b].paramCount() + ffnDown[(std::size_t)b].paramCount()
               + ln1[(std::size_t)b].paramCount() + ln2[(std::size_t)b].paramCount();
        }
        return t;
    }

    /* 参数构成: 本类没有"外积伪注意力", 所以注意力 = 四个投影矩阵, 其余归 FFN */
    void paramBreakdown(long long &attn, long long &ffn, long long &norm) const override
    {
        attn = 0; ffn = 0; norm = 0;
        for (int b = 0; b < blocks; b++) {
            attn += wq[(std::size_t)b].paramCount() + wk[(std::size_t)b].paramCount()
                  + wv[(std::size_t)b].paramCount() + wo[(std::size_t)b].paramCount();
            ffn  += ffnUp[(std::size_t)b].paramCount() + ffnDown[(std::size_t)b].paramCount();
            norm += ln1[(std::size_t)b].paramCount() + ln2[(std::size_t)b].paramCount();
        }
        ffn += embed.paramCount() + outProj.paramCount() + pos.paramCount();
    }

    /* 只读诊断: 注意力矩阵 (= 头数 × T²) —— 与旧实现"注意力元素 = 头数 × d_k²"对位 */
    long long attnElements() const override
    {
        return (long long)heads * (long long)SeqLen * (long long)SeqLen;
    }
    int seqLen() const { return SeqLen; }
    int tokenDim() const { return tokDim; }
    int attnHeadsUsed() const override { return heads; }
    int attnHeadsRequested() const override { return Heads; }
    int attnHeadDim() const override { return dk; }

    /* ---------------- 位置编码 (2026-10) ---------------- */
    SeqPosEnc::Mode posMode() const { return pos.mode; }
    const char *posModeKey() const { return SeqPosEnc::modeKey(pos.mode); }
    std::string posDescribe() const { return pos.describe(); }
    long long posParamCount() const { return pos.paramCount(); }
    int posGroupCount() const { return pos.groupCount(); }
    /*
       切换位置编码模式。**约定: 构造之后、开始训练之前调用一次**。
       这里自己做两件容易漏的事:
         * 重新分配该模式的参数 (LEARN_2D 两组、ROPE 零组) —— 会消耗随机数, 但
           **只在切到非默认模式时**才发生 (默认 LEARN_1D 一个数都不多画);
         * 立刻按 1/√tokDim 缩放 (与 `scaleInit` 对位置嵌入做的是同一步) —— 否则
           "先 setPosMode 再 scaleExpertInit" 会缩两次、"先 scale 再 set" 会一次不缩。
       ⚠ 权重不再兼容: 模式是**权重文件自描述**的一部分 (见 `SeqPosEnc::read`),
         换模式必须重训或换回原模式的权重, 否则载入时**响亮失败**。
    */
    void setPosMode(SeqPosEnc::Mode m)
    {
        if (m == pos.mode) { return; }
        pos.init(SeqLen, tokDim, POS_ROWS, POS_COLS, m, withGradFlag, dk);
        if (!posScaledBySetMode) { pos.scaleInit((float)tokDim); }   /* setPosMode 已经缩过一次就不再缩 */
        posScaledBySetMode = true;
    }
    bool setPosModeByKey(const std::string &key)
    {
        SeqPosEnc::Mode m;
        if (!SeqPosEnc::parseMode(key, m)) { return false; }
        setPosMode(m);
        return true;
    }

    /* ---------------- dropout (2026-10 补的"未做结构项"之二) ----------------
       标准 Transformer 的三处 dropout (倒置式: 保留概率 1−p, 保留的按 1/(1−p) 放大,
       于是**推理时不需要任何缩放**):
         * 注意力概率 P (softmax 之后、乘 V 之前)
         * 残差支路 1 的输出 M (Wo 之后)
         * 残差支路 2 的输出 F (FFN 之后)
       ⚠ 三条纪律 (都是本工程被咬过的地方):
         1. **默认 p=0**: 一条 RNG 都不画、一个字节都不多占 ⇒ 存量权重与既有读数不变
            (同 `enableMlpGate` / `HonorHeads` 的处理方式);
         2. 只在 **`inference=false`** 时生效 —— 与 `Layer<>` 的习惯不同 (那个 `Dropout`
            层用 `withGrad` 当开关, 而且 mask 保留概率与缩放不一致, 见 layer.h 里的注);
         3. mask 存在 `BlockCache` 里, 反向用**同一张** mask (否则前后向不一致,
            有限差分立刻抓到)。
       实测见 test_transformer [7]: p=0 逐位不变 / p>0 丢弃比例与 mask 一致 /
       反向只在保留位置有梯度 / 推理形态不做 dropout。
    */
    float dropoutP() const { return dropP; }
    void setDropout(float p)
    {
        if (p < 0.0f) { p = 0.0f; }
        if (p >= 1.0f) { p = 0.999f; }
        dropP = p;
    }
    /*
       只读诊断 + 测试开关 (与 `usePos` 同一类: 生产默认不使用)。
         * `lastMask(b, which)`: which = 0 注意力概率 / 1 残差 1 (M) / 2 残差 2 (F);
           只在 dropP > 0 的前向之后有内容, 供自检面板与断言读"保留比例"。
         * `setDropoutFreezeMask(true)`: 复用上一张 mask 而不是每次重抽。
           为什么需要它: 开了 dropout 之后损失函数**不是确定性的**, 中心差分没法做; 冻住
           mask 之后它又是确定函数了, `test_transformer [7]` 就能用有限差分钉住
           "反向用的是**同一张** mask" (哪怕方向写反也能抓到)。
    */
    const std::vector<float> &lastMask(int b, int which) const
    {
        static const std::vector<float> empty;
        if (b < 0 || b >= blocks) { return empty; }
        const BlockCache &c = blk[(std::size_t)b];
        return (which == 0) ? c.maskP : ((which == 1) ? c.maskM : c.maskF);
    }
    void setDropoutFreezeMask(bool on) { dropFreezeMask = on; }
    /*
       `SparseMoE`/`iLayer` 的转发接口用的名字 (见 ilayer.h 的两个 traits):
       返回 bool = "这个骨干支持该旋钮", 便于调用方判断而不是静默不生效。
    */
    bool setExpertDropout(float p) { setDropout(p); return true; }

    SeqTransformerExpert() : dIn(0), featDim(0), tokDim(0), dff(0), dk(0), heads(0), blocks(0) {}

    SeqTransformerExpert(int dIn_, int tokDim_, bool withGrad)
        : dIn(dIn_), tokDim(tokDim_)
    {
        type = LAYER_SEQTRANSFORMER;
        heads = Heads;
        blocks = Blocks;

        /*
           棋盘是 10 行 × 9 列 = 90 = SeqLen ⇒ 2D 位置编码的前提。不满足就**响亮失败**
           (这一层按格子切 token, "行列"必须能对上 `canonicalCell(x,y)=x*9+y`)。
        */
        if (POS_ROWS * POS_COLS != SeqLen) {
            std::fprintf(stderr,
                         "[SeqTransformerExpert] POS_ROWS(%d) x POS_COLS(%d) = %d 与 SeqLen=%d 不一致 —— "
                         "2D 位置编码需要它们相等。\n",
                         POS_ROWS, POS_COLS, POS_ROWS * POS_COLS, SeqLen);
            std::fflush(stderr);
            std::abort();
        }

        /*
           两个编译期常量的前置条件必须**响亮失败** —— 本工程被"静默降级"咬过不止一次
           (旧 TB 专家"请求 16 头 / 实用 15 头"就是同一个形状的坑: 不报错、只是变差)。
           dIn 必须是 SeqLen 的整数倍 (1710 = 90 × 19 ✓); tokDim 必须能被头数整除。
        */
        if (SeqLen <= 0 || dIn % SeqLen != 0) {
            std::fprintf(stderr,
                         "[SeqTransformerExpert] dIn=%d 不是 SeqLen=%d 的整数倍 —— "
                         "这个专家按格子切 token, 需要 dIn = 平面数 x %d。\n",
                         dIn, SeqLen, SeqLen);
            std::fflush(stderr);
            std::abort();
        }
        featDim = dIn / SeqLen;
        if (tokDim <= 0 || tokDim % Heads != 0) {
            std::fprintf(stderr, "[SeqTransformerExpert] tokDim=%d 不能被 Heads=%d 整除。\n",
                         tokDim, Heads);
            std::fflush(stderr);
            std::abort();
        }
        dk = tokDim / Heads;
        dff = 3 * tokDim;                 /* FFN 宽度 = 3×token 宽 (标准是 4×, 这里省一点) */

        withGradFlag = withGrad;
        embed   = Layer<Linear>(featDim, tokDim, true, withGrad);
        outProj = Layer<Linear>(tokDim, featDim, true, withGrad);
        for (int b = 0; b < blocks; b++) {
            wq.push_back(Layer<Linear>(tokDim, tokDim, true, withGrad));
            wk.push_back(Layer<Linear>(tokDim, tokDim, true, withGrad));
            wv.push_back(Layer<Linear>(tokDim, tokDim, true, withGrad));
            wo.push_back(Layer<Linear>(tokDim, tokDim, true, withGrad));
            ffnUp.push_back(Layer<Gelu>(tokDim, dff, true, withGrad));
            ffnDown.push_back(Layer<Linear>(dff, tokDim, true, withGrad));
            /* 两个 LN: γ=1, β=0 由 SeqLayerNorm::init 内部设定 (与改动前逐位相同) */
            ln1.push_back(SeqLayerNorm(tokDim, withGrad));
            ln2.push_back(SeqLayerNorm(tokDim, withGrad));
            blk.push_back(BlockCache());
        }
        /*
           位置编码: 默认 LEARN_1D —— **与改动前逐位相同** (同一处分配、同一处
           `Random::uniform(-0.02, 0.02)`、同样的加法循环)。2D / RoPE 由
           `setPosMode()` 在构造后切换 (见那里的说明: 默认模式不消耗额外随机数,
           所以存量权重与既有 golden 读数不受影响 —— 与 `enableMlpGate` 同一纪律)。
        */
        pos.init(SeqLen, tokDim, POS_ROWS, POS_COLS, SeqPosEnc::LEARN_1D, withGrad, dk);

        allocBuffers();
        o = Tensor((std::size_t)dIn, 1);
        e = Tensor((std::size_t)dIn, 1);
    }

    /* ============================================================
     *  前向
     * ============================================================ */
    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        gather(x);
        fcFwd(embed, Xs, featDim, tokDim, CUR, inference);
        if (usePos) {
            pos.forwardAdd(CUR);            /* LEARN_*: 加位置向量; ROPE: 不动输入 */
        }
        std::vector<float> cur = CUR;
        for (int b = 0; b < blocks; b++) {
            BlockCache &c = blk[(std::size_t)b];
            /* 本层的输入序列 (= LN1 的输入) —— 反向要它算 LN 的解析梯度, 而正向的
               `cur` 是局部变量, 所以这里必须存一份 (不存就得在反向重算整条链)。 */
            curInOf(b) = cur;
            ln1[(std::size_t)b].forward(cur, c.h1);
            fcFwd(wq[(std::size_t)b], c.h1, tokDim, tokDim, c.q, inference);
            fcFwd(wk[(std::size_t)b], c.h1, tokDim, tokDim, c.k, inference);
            /*
               RoPE: 位置编码作用在 Q/K 上 (相对位置)。必须在 `attnFwd` **之前**,
               而且旋转后的 Q/K 就是缓存里那一份 —— 反向的 `attnBwd` 用它们重算
               dP/dS, 所以前向/反向必须看到同一份 (见 `ropeApply` 的 inverse 用法)。
            */
            if (usePos && pos.mode == SeqPosEnc::ROPE) {
                pos.ropeApply(c.q, c.k, heads, dk, false);
            }
            fcFwd(wv[(std::size_t)b], c.h1, tokDim, tokDim, c.v, inference);
            attnFwd(c.q, c.k, c.v, c.p, c.att);
            /*
               dropout ①: 注意力概率。⚠ 不能就地改 `c.p` —— 反向的 softmax 雅可比
               必须用**未丢弃**的 P (dS = P⊙(dP−⟨P,dP⟩), 然后 dP 再乘 mask)。
               所以这里保持 c.p 原样, 把 mask 存进 c.maskP 并用 mask⊙P 重算 att。
            */
            if (dropP > 0.0f && !inference) { dropoutAttn(c); }
            fcFwd(wo[(std::size_t)b], c.att, tokDim, tokDim, c.m, inference);
            /* dropout ②: 残差支路 1 的输出 M (只作用在 M 这条路上, 残差那条不动) */
            if (dropP > 0.0f && !inference) { dropoutMaskInto(c.maskM, c.m.size()); scaleVecByMask(c.m, c.maskM); }
            for (int i = 0; i < SeqLen * tokDim; i++) { c.x1[(std::size_t)i] = cur[(std::size_t)i] + c.m[(std::size_t)i]; }
            ln2[(std::size_t)b].forward(c.x1, c.h2);
            fcFwd(ffnUp[(std::size_t)b], c.h2, tokDim, dff, c.fh, inference);
            fcFwd(ffnDown[(std::size_t)b], c.fh, dff, tokDim, c.f, inference);
            /* dropout ③: 残差支路 2 的输出 F */
            if (dropP > 0.0f && !inference) { dropoutMaskInto(c.maskF, c.f.size()); scaleVecByMask(c.f, c.maskF); }
            for (int i = 0; i < SeqLen * tokDim; i++) { cur[(std::size_t)i] = c.x1[(std::size_t)i] + c.f[(std::size_t)i]; }
        }
        FIN = cur;
        fcFwd(outProj, FIN, tokDim, featDim, YS, inference);
        scatter();
        return o;
    }

    /* ============================================================
     *  反向 (BPTT; 全部按 token 展开, 与标准 Transformer 一致)
     * ============================================================ */
    void backward(const Tensor& x, Tensor& ei) override
    {
        /* dL/d输出序列 (T × featDim) ← e (dIn × 1) 的按平面优先的逆映射 */
        dYS.assign((std::size_t)(SeqLen * featDim), 0.0f);
        for (int t = 0; t < SeqLen; t++) {
            for (int f = 0; f < featDim; f++) {
                dYS[(std::size_t)(t * featDim + f)] = e[(std::size_t)(f * SeqLen + t)];
            }
        }
        /* outProj 反向: 输出 (T × featDim) 的梯度 -> (T × tokDim) 的梯度 */
        fcBwd(outProj, FIN, tokDim, featDim, dYS, dCUR);

        for (int b = blocks - 1; b >= 0; b--) {
            BlockCache &c = blk[(std::size_t)b];
            /* ---- FFN 支路 ----
               ⚠ dropout ③ 的反向**不能**就地乘 dCUR: 那个缓冲同时承载残差(恒等)那条
               路的梯度, 乘上去会把没被丢弃的那条路也一起缩放。正确做法是把
               "dF = dCUR ⊙ maskF" 算进一个临时缓冲, 只喂给 FFN 这一支。 */
            if (dropP > 0.0f) {
                for (std::size_t i = 0; i < (std::size_t)(SeqLen * tokDim); i++) {
                    dTMP2[i] = dCUR[i] * c.maskF[i];
                }
                fcBwd(ffnDown[(std::size_t)b], c.fh, dff, tokDim, dTMP2, dTMP);  /* dFh */
            } else {
                fcBwd(ffnDown[(std::size_t)b], c.fh, dff, tokDim, dCUR, dTMP);   /* dFh */
            }
            fcBwd(ffnUp[(std::size_t)b], c.h2, tokDim, dff, dTMP, dTMP2);       /* dH2 */
            /* LN2 反向 (累加进 dCUR = dX1, 因为 x1 既走残差又走 LN2) */
            ln2[(std::size_t)b].backward(c.x1, dTMP2, dCUR);
            /* ---- 注意力支路 ----
               dropout ② 的反向同样只作用于进 Wo 的那条路 (残差那条仍然拿 dCUR) */
            if (dropP > 0.0f) {
                for (std::size_t i = 0; i < (std::size_t)(SeqLen * tokDim); i++) {
                    dTMP2[i] = dCUR[i] * c.maskM[i];
                }
                fcBwd(wo[(std::size_t)b], c.att, tokDim, tokDim, dTMP2, dTMP);   /* dAtt */
            } else {
                fcBwd(wo[(std::size_t)b], c.att, tokDim, tokDim, dCUR, dTMP);    /* dAtt */
            }
            /* dropout ① 的 mask 交给 attnBwd: dP 先乘 mask, 再走 softmax 雅可比 */
            attnBwd(c.q, c.k, c.v, c.p, dTMP, dQv, dKv, dVv,
                    (dropP > 0.0f) ? &c.maskP : nullptr);
            /*
               RoPE 的反向: `attnBwd` 给出的是对**旋转后** Q/K 的梯度, 而接下来要
               反传给 Wq/Wk 的是对**旋转前** Q/K 的梯度 ⇒ 就地做一次逆旋转
               (旋转是正交阵, 逆 = 转置)。位置在这两句之前、Wq/Wk 反向之前。
            */
            if (usePos && pos.mode == SeqPosEnc::ROPE) {
                pos.ropeApply(dQv, dKv, heads, dk, true);
            }
            fcBwd(wq[(std::size_t)b], c.h1, tokDim, tokDim, dQv, dTMP);         /* dH1 */
            fcBwd(wk[(std::size_t)b], c.h1, tokDim, tokDim, dKv, dTMP2);
            for (std::size_t i = 0; i < dTMP.size(); i++) { dTMP[i] += dTMP2[i]; }
            fcBwd(wv[(std::size_t)b], c.h1, tokDim, tokDim, dVv, dTMP2);
            for (std::size_t i = 0; i < dTMP.size(); i++) { dTMP[i] += dTMP2[i]; }
            /* LN1 反向 (累加进本层的输入梯度 = dCUR) */
            ln1[(std::size_t)b].backward(curInOf(b), dTMP, dCUR);
        }

        /*
           embed 反向: dXs = ∂L/∂(embed 输出) —— 也就是**位置嵌入之前的序列**梯度。
           位置嵌入的梯度**必须**从这里取 (它是加法路由: 两条支路同梯度), 而它的布局是
           (T × tokDim) —— 与 `pos.val` 同形。⚠ 不能拿 `dXs` 当位置嵌入的梯度: dXs 是
           (T × featDim) = (90 × 19), 而 pos 是 (90 × 64), 一个是 1710 一个是 5760 ——
           第一版就是这么写错的 (测试把 pos 的梯度当场抓了出来, 见 test_transformer [1])。
        */
        pos.backwardAdd(dCUR);          /* LEARN_*: 累加位置梯度; ROPE: 无参数 */
        fcBwd(embed, Xs, featDim, tokDim, dCUR, dXs);
        /* 输入梯度: 按平面优先散回, 并**累加** (与其它层的约定一致) */
        for (int t = 0; t < SeqLen; t++) {
            for (int f = 0; f < featDim; f++) {
                ei[(std::size_t)(f * SeqLen + t)] += dXs[(std::size_t)(t * featDim + f)];
            }
        }
    }

    /* ============================================================
     *  优化器
     * ============================================================ */
    void SGD(float lr) override
    {
        embed.SGD(lr); outProj.SGD(lr);
        for (int b = 0; b < blocks; b++) {
            wq[(std::size_t)b].SGD(lr); wk[(std::size_t)b].SGD(lr);
            wv[(std::size_t)b].SGD(lr); wo[(std::size_t)b].SGD(lr);
            ffnUp[(std::size_t)b].SGD(lr); ffnDown[(std::size_t)b].SGD(lr);
            ln1[(std::size_t)b].SGD(lr); ln2[(std::size_t)b].SGD(lr);
        }
        pos.SGD(lr);
    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        embed.RMSProp(lr, rho, decay, clipGrad);
        outProj.RMSProp(lr, rho, decay, clipGrad);
        for (int b = 0; b < blocks; b++) {
            wq[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            wk[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            wv[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            wo[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            ffnUp[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            ffnDown[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            ln1[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
            ln2[(std::size_t)b].RMSProp(lr, rho, decay, clipGrad);
        }
        pos.RMSProp(lr, rho, decay, clipGrad);
    }
    void Adam(float lr, float alpha, float beta, float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        embed.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        outProj.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        for (int b = 0; b < blocks; b++) {
            wq[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            wk[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            wv[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            wo[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            ffnUp[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            ffnDown[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            ln1[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
            ln2[(std::size_t)b].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        pos.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
    }
    void clamp(float c0, float cn) override
    {
        embed.clamp(c0, cn); outProj.clamp(c0, cn);
        for (int b = 0; b < blocks; b++) {
            wq[(std::size_t)b].clamp(c0, cn); wk[(std::size_t)b].clamp(c0, cn);
            wv[(std::size_t)b].clamp(c0, cn); wo[(std::size_t)b].clamp(c0, cn);
            ffnUp[(std::size_t)b].clamp(c0, cn); ffnDown[(std::size_t)b].clamp(c0, cn);
        }
        pos.clamp(c0, cn);
        /*
           ⚠ LN 的顺序**必须**留在 pos 之后、按块 ln1 再 ln2 —— 与抽类前逐字一致。
           `Optimize::clamp` 对越界项会**从 RNG 重采样** (util.hpp), 所以顺序变了,
           同一次训练里"哪个参数拿到哪个随机数"就会变 (值集合相同、分配不同)。
           这是抽类时唯一一处顺序真的会改数值的接线 —— 前向/反向/三个优化器的顺序
           都是逐条对齐的 (SGD/RMSProp/Adam 的 pos 也在 LN 之前)。
        */
        for (int b = 0; b < blocks; b++) {
            ln1[(std::size_t)b].clamp(c0, cn); ln2[(std::size_t)b].clamp(c0, cn);
        }
    }

    /* 梯度范数² / 梯度缩放 (全局裁剪; 见 ilayer.h 的同名虚函数) */
    double gradNorm2() const override
    {
        double s = gradNorm2Of(embed.g.w) + gradNorm2Of(embed.g.b)
                 + gradNorm2Of(outProj.g.w) + gradNorm2Of(outProj.g.b)
                 + pos.gradNorm2();
        for (int b = 0; b < blocks; b++) {
            s += gradNorm2Of(wq[(std::size_t)b].g.w) + gradNorm2Of(wk[(std::size_t)b].g.w)
               + gradNorm2Of(wv[(std::size_t)b].g.w) + gradNorm2Of(wo[(std::size_t)b].g.w)
               + gradNorm2Of(ffnUp[(std::size_t)b].g.w) + gradNorm2Of(ffnDown[(std::size_t)b].g.w)
               + ln1[(std::size_t)b].gradNorm2() + ln2[(std::size_t)b].gradNorm2();
        }
        return s;
    }
    void scaleGrad(float s) override
    {
        scaleTensorGrad(embed.g.w, s); scaleTensorGrad(embed.g.b, s);
        scaleTensorGrad(outProj.g.w, s); scaleTensorGrad(outProj.g.b, s);
        pos.scaleGrad(s);
        for (int b = 0; b < blocks; b++) {
            scaleTensorGrad(wq[(std::size_t)b].g.w, s);
            scaleTensorGrad(wk[(std::size_t)b].g.w, s);
            scaleTensorGrad(wv[(std::size_t)b].g.w, s);
            scaleTensorGrad(wo[(std::size_t)b].g.w, s);
            scaleTensorGrad(ffnUp[(std::size_t)b].g.w, s);
            scaleTensorGrad(ffnDown[(std::size_t)b].g.w, s);
            ln1[(std::size_t)b].scaleGrad(s);
            ln2[(std::size_t)b].scaleGrad(s);
        }
    }

    /* ============================================================
     *  权重往返 / 复制 (顺序必须与 write 完全一致)
     * ============================================================ */
    void write(std::ostream &file) override
    {
        embed.write(file);
        pos.write(file);   /* LEARN_1D 一行张量; 2D/RoPE 先写模式标记 */
        for (int b = 0; b < blocks; b++) {
            ln1[(std::size_t)b].write(file);    /* γ 然后 β (与抽类前逐字节一致) */
            wq[(std::size_t)b].write(file);
            wk[(std::size_t)b].write(file);
            wv[(std::size_t)b].write(file);
            wo[(std::size_t)b].write(file);
            ln2[(std::size_t)b].write(file);
            ffnUp[(std::size_t)b].write(file);
            ffnDown[(std::size_t)b].write(file);
        }
        outProj.write(file);
    }
    void read(std::istream &file) override
    {
        embed.read(file);
        pos.read(file);    /* 模式自描述: 与配置不一致就响亮失败 */
        for (int b = 0; b < blocks; b++) {
            ln1[(std::size_t)b].read(file);
            wq[(std::size_t)b].read(file);
            wk[(std::size_t)b].read(file);
            wv[(std::size_t)b].read(file);
            wo[(std::size_t)b].read(file);
            ln2[(std::size_t)b].read(file);
            ffnUp[(std::size_t)b].read(file);
            ffnDown[(std::size_t)b].read(file);
        }
        outProj.read(file);
    }
    void copyTo(iLayer* layer) override
    {
        SeqTransformerExpert *p = dynamic_cast<SeqTransformerExpert*>(layer);
        if (p == nullptr) { return; }
        embed.copyTo(&p->embed);
        pos.copyTo(&p->pos);
        for (int b = 0; b < blocks; b++) {
            ln1[(std::size_t)b].copyTo(&p->ln1[(std::size_t)b]);
            ln2[(std::size_t)b].copyTo(&p->ln2[(std::size_t)b]);
            wq[(std::size_t)b].copyTo(&p->wq[(std::size_t)b]);
            wk[(std::size_t)b].copyTo(&p->wk[(std::size_t)b]);
            wv[(std::size_t)b].copyTo(&p->wv[(std::size_t)b]);
            wo[(std::size_t)b].copyTo(&p->wo[(std::size_t)b]);
            ffnUp[(std::size_t)b].copyTo(&p->ffnUp[(std::size_t)b]);
            ffnDown[(std::size_t)b].copyTo(&p->ffnDown[(std::size_t)b]);
        }
        outProj.copyTo(&p->outProj);
    }
    void softUpdateTo(iLayer* layer, float alpha) override
    {
        SeqTransformerExpert *p = dynamic_cast<SeqTransformerExpert*>(layer);
        if (p == nullptr) { return; }
        embed.softUpdateTo(&p->embed, alpha);
        pos.softUpdateTo(&p->pos, alpha);
        for (int b = 0; b < blocks; b++) {
            ln1[(std::size_t)b].softUpdateTo(&p->ln1[(std::size_t)b], alpha);
            ln2[(std::size_t)b].softUpdateTo(&p->ln2[(std::size_t)b], alpha);
            wq[(std::size_t)b].softUpdateTo(&p->wq[(std::size_t)b], alpha);
            wk[(std::size_t)b].softUpdateTo(&p->wk[(std::size_t)b], alpha);
            wv[(std::size_t)b].softUpdateTo(&p->wv[(std::size_t)b], alpha);
            wo[(std::size_t)b].softUpdateTo(&p->wo[(std::size_t)b], alpha);
            ffnUp[(std::size_t)b].softUpdateTo(&p->ffnUp[(std::size_t)b], alpha);
            ffnDown[(std::size_t)b].softUpdateTo(&p->ffnDown[(std::size_t)b], alpha);
        }
        outProj.softUpdateTo(&p->outProj, alpha);
    }

    /* 初始化缩放: 全部 FC 按 1/√fan_in (理由见 expert.hpp 顶部) */
    void scaleInit()
    {
        scaleFcInit(embed);
        scaleFcInit(outProj);
        for (int b = 0; b < blocks; b++) {
            scaleFcInit(wq[(std::size_t)b]); scaleFcInit(wk[(std::size_t)b]);
            scaleFcInit(wv[(std::size_t)b]); scaleFcInit(wo[(std::size_t)b]);
            scaleFcInit(ffnUp[(std::size_t)b]); scaleFcInit(ffnDown[(std::size_t)b]);
        }
        /* 位置嵌入按 1/√tokDim 缩一遍, 免得它一上来就盖过 token 特征 */
        const float s = 1.0f / std::sqrt((float)(tokDim > 1 ? tokDim : 1));
        if (!posScaledBySetMode) { pos.scaleInit((float)tokDim); }   /* setPosMode 已经缩过一次就不再缩 */
    }

private:
    bool withGradFlag;
    /* setPosMode 自己缩放过位置嵌入 (见那里的说明: 免得缩两次/一次不缩) */
    bool posScaledBySetMode = false;
    /* 测试开关: 冻结 dropout 的 mask (默认 false ⇒ 生产行为不变) */
    bool dropFreezeMask = false;
    /* 每层 LN1 的输入序列 (= 进入本层时的残差流) —— 反向要它算 LN 的解析梯度 */
    std::vector<std::vector<float> > curIn;

    void allocBuffers()
    {
        const std::size_t nT = (std::size_t)(SeqLen * tokDim);
        for (int b = 0; b < blocks; b++) {
            BlockCache &c = blk[(std::size_t)b];
            c.h1.assign(nT, 0.0f); c.q.assign(nT, 0.0f); c.k.assign(nT, 0.0f);
            c.v.assign(nT, 0.0f);  c.att.assign(nT, 0.0f); c.m.assign(nT, 0.0f);
            c.x1.assign(nT, 0.0f); c.h2.assign(nT, 0.0f); c.f.assign(nT, 0.0f);
            c.fh.assign((std::size_t)(SeqLen * dff), 0.0f);
            c.p.assign((std::size_t)(heads * SeqLen * SeqLen), 0.0f);
            curIn.push_back(std::vector<float>(nT, 0.0f));
        }
        Xs.assign((std::size_t)(SeqLen * featDim), 0.0f);
        CUR.assign(nT, 0.0f);
        FIN.assign(nT, 0.0f);
        YS.assign((std::size_t)(SeqLen * featDim), 0.0f);
        dCUR.assign(nT, 0.0f);
        dYS.assign((std::size_t)(SeqLen * featDim), 0.0f);
        dTMP.assign((std::size_t)(SeqLen * (dff > tokDim ? dff : tokDim)), 0.0f);
        dTMP2.assign(dTMP.size(), 0.0f);
        dQv.assign(nT, 0.0f);
        dKv.assign(nT, 0.0f);
        dVv.assign(nT, 0.0f);
        dXs.assign((std::size_t)(SeqLen * featDim), 0.0f);
        /* 逐 token 调用 Layer<> 的临时张量: 三种维数各一套 */
        scIn[0] = Tensor((std::size_t)featDim, 1);
        scIn[1] = Tensor((std::size_t)tokDim, 1);
        scIn[2] = Tensor((std::size_t)dff, 1);
        for (int i = 0; i < 3; i++) { scEi[i] = scIn[i]; scE[i] = Tensor(scIn[i].totalSize, 1); }
    }
    Tensor& scrInFor(int d) { return (d == featDim) ? scIn[0] : ((d == tokDim) ? scIn[1] : scIn[2]); }
    Tensor& scrEiFor(int d) { return (d == featDim) ? scEi[0] : ((d == tokDim) ? scEi[1] : scEi[2]); }
    Tensor& scrEFor(int d)  { return (d == featDim) ? scE[0]  : ((d == tokDim) ? scE[1]  : scE[2]); }
    std::vector<float>& curInOf(int b) { return curIn[(std::size_t)b]; }

    /* ---- dropout 的三个帮手 (只在 dropP > 0 时被调用) ---- */
    void dropoutMaskInto(std::vector<float> &mask, std::size_t n)
    {
        const float keep = 1.0f - dropP;
        const float inv = 1.0f / keep;
        if (mask.size() != n) { mask.assign(n, inv); }
        if (dropFreezeMask) { return; }        /* 测试开关: 复用上一张 mask */
        std::bernoulli_distribution keepDraw((double)keep);
        for (std::size_t i = 0; i < n; i++) { mask[i] = keepDraw(Random::engine) ? inv : 0.0f; }
    }
    static void scaleVecByMask(std::vector<float> &v, const std::vector<float> &mask)
    {
        const std::size_t n = (v.size() < mask.size()) ? v.size() : mask.size();
        for (std::size_t i = 0; i < n; i++) { v[i] *= mask[i]; }
    }
    /* 注意力概率的 dropout: 生成 mask, 并用 (mask⊙P)V 重算 att (c.p 保持未丢弃) */
    void dropoutAttn(BlockCache &c)
    {
        dropoutMaskInto(c.maskP, c.p.size());
        for (int h = 0; h < heads; h++) {
            const float *Ph = &c.p[(std::size_t)(h * SeqLen * SeqLen)];
            const float *Mh = &c.maskP[(std::size_t)(h * SeqLen * SeqLen)];
            for (int i = 0; i < SeqLen; i++) {
                for (int cc = 0; cc < dk; cc++) {
                    float acc = 0;
                    for (int j = 0; j < SeqLen; j++) {
                        acc += Mh[(std::size_t)(i * SeqLen + j)] * Ph[(std::size_t)(i * SeqLen + j)]
                             * c.v[(std::size_t)(j * tokDim + h * dk + cc)];
                    }
                    c.att[(std::size_t)(i * tokDim + h * dk + cc)] = acc;
                }
            }
        }
    }

    /* 平面优先状态 (PLANES × SeqLen) → 序列 (SeqLen × featDim) */
    void gather(const Tensor &x)
    {
        for (int t = 0; t < SeqLen; t++) {
            for (int f = 0; f < featDim; f++) {
                Xs[(std::size_t)(t * featDim + f)] = x[(std::size_t)(f * SeqLen + t)];
            }
        }
    }
    void scatter()
    {
        for (int t = 0; t < SeqLen; t++) {
            for (int f = 0; f < featDim; f++) {
                o[(std::size_t)(f * SeqLen + t)] = YS[(std::size_t)(t * featDim + f)];
            }
        }
    }

    /* ---- 逐 token 的全连接: Y = X·Wᵀ + b (复用经过测试的 Layer<Linear>) ---- */
    void fcFwd(Layer<Linear> &l, const std::vector<float> &X, int inD, int outD,
               std::vector<float> &Y, bool inference)
    {
        Tensor &xi = scrInFor(inD);
        for (int t = 0; t < SeqLen; t++) {
            for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
            l.forward(xi, inference);
            for (int j = 0; j < outD; j++) { Y[(std::size_t)(t * outD + j)] = l.o[(std::size_t)j]; }
        }
    }
    void fcFwd(Layer<Gelu> &l, const std::vector<float> &X, int inD, int outD,
               std::vector<float> &Y, bool inference)
    {
        Tensor &xi = scrInFor(inD);
        for (int t = 0; t < SeqLen; t++) {
            for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
            l.forward(xi, inference);
            for (int j = 0; j < outD; j++) { Y[(std::size_t)(t * outD + j)] = l.o[(std::size_t)j]; }
        }
    }
    /*
       逐 token 的全连反向。**关键细节**: `Layer<Fn>::backward` 用**它自己缓存的 o** 算激活
       导数, 而一个 Layer 对象被 T 个 token 共用 —— 所以这里在每个 token 反向之前先重跑一次
       该 token 的 forward (把缓存刷成这个 token 的)。代价是 ~2× 的计算, 换来的是"不必再写
       一份 FC/激活的反向" —— 而"再写一份"正是本工程反复吃亏的地方。
    */
    void fcBwd(Layer<Linear> &l, const std::vector<float> &X, int inD, int outD,
               const std::vector<float> &dY, std::vector<float> &dX)
    {
        Tensor &xi = scrInFor(inD);
        Tensor &ei = scrEiFor(inD);
        Tensor &de = scrEFor(outD);
        dX.assign((std::size_t)(SeqLen * inD), 0.0f);
        for (int t = 0; t < SeqLen; t++) {
            for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
            l.forward(xi, true);                       /* 刷新缓存 (Linear 无害, Gelu 必需) */
            for (int j = 0; j < outD; j++) { de[(std::size_t)j] = dY[(std::size_t)(t * outD + j)]; }
            l.e = de;
            ei.zero();
            l.backward(xi, ei);
            for (int j = 0; j < inD; j++) { dX[(std::size_t)(t * inD + j)] = ei[(std::size_t)j]; }
        }
    }
    void fcBwd(Layer<Gelu> &l, const std::vector<float> &X, int inD, int outD,
               const std::vector<float> &dY, std::vector<float> &dX)
    {
        Tensor &xi = scrInFor(inD);
        Tensor &ei = scrEiFor(inD);
        Tensor &de = scrEFor(outD);
        dX.assign((std::size_t)(SeqLen * inD), 0.0f);
        for (int t = 0; t < SeqLen; t++) {
            for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
            l.forward(xi, true);
            for (int j = 0; j < outD; j++) { de[(std::size_t)j] = dY[(std::size_t)(t * outD + j)]; }
            l.e = de;
            ei.zero();
            l.backward(xi, ei);
            for (int j = 0; j < inD; j++) { dX[(std::size_t)(t * inD + j)] = ei[(std::size_t)j]; }
        }
    }

    /*
       ---- LayerNorm ----
       [2026-10] 已抽成 `SeqLayerNorm` 类 (见文件顶部): 原来这里是 `lnFwd` / `lnBwd`
       两个静态函数 + 专家成员里 4 份 `Param` (ln1g/ln1b/ln2g/ln2b), 于是"加一个 LN"
       要在 10 处接线里各改一遍。现在每个块持有 `ln1[b]` / `ln2[b]` 两个对象,
       前向/反向/优化器/范数/往返都只剩一行循环。
    */

    /* ---- 多头注意力: scores = Q Kᵀ / √d_k, **按行** softmax, O = P V ---- */
    void attnFwd(const std::vector<float> &Q, const std::vector<float> &K,
                 const std::vector<float> &V, std::vector<float> &P,
                 std::vector<float> &Out) const
    {
        const float scale = 1.0f / std::sqrt((float)dk);
        for (int h = 0; h < heads; h++) {
            float *Ph = &P[(std::size_t)(h * SeqLen * SeqLen)];
            for (int i = 0; i < SeqLen; i++) {
                /* scores 一行 */
                float mx = -1e30f;
                for (int j = 0; j < SeqLen; j++) {
                    float s = 0;
                    for (int c = 0; c < dk; c++) {
                        s += Q[(std::size_t)(i * tokDim + h * dk + c)]
                           * K[(std::size_t)(j * tokDim + h * dk + c)];
                    }
                    s *= scale;
                    Ph[(std::size_t)(i * SeqLen + j)] = s;
                    if (s > mx) { mx = s; }
                }
                float sum = 0;
                for (int j = 0; j < SeqLen; j++) {
                    const float e = std::exp(Ph[(std::size_t)(i * SeqLen + j)] - mx);
                    Ph[(std::size_t)(i * SeqLen + j)] = e;
                    sum += e;
                }
                const float inv = 1.0f / (sum + 1e-30f);
                for (int j = 0; j < SeqLen; j++) { Ph[(std::size_t)(i * SeqLen + j)] *= inv; }
            }
            for (int i = 0; i < SeqLen; i++) {
                for (int c = 0; c < dk; c++) {
                    float acc = 0;
                    for (int j = 0; j < SeqLen; j++) {
                        acc += Ph[(std::size_t)(i * SeqLen + j)] * V[(std::size_t)(j * tokDim + h * dk + c)];
                    }
                    Out[(std::size_t)(i * tokDim + h * dk + c)] = acc;
                }
            }
        }
    }
    void attnBwd(const std::vector<float> &Q, const std::vector<float> &K,
                 const std::vector<float> &V, const std::vector<float> &P,
                 const std::vector<float> &dOut, std::vector<float> &dQ,
                 std::vector<float> &dK, std::vector<float> &dV,
                 const std::vector<float> *maskP = nullptr) const
    {
        const float scale = 1.0f / std::sqrt((float)dk);
        dQ.assign((std::size_t)(SeqLen * tokDim), 0.0f);
        dK.assign(dQ.size(), 0.0f);
        dV.assign(dQ.size(), 0.0f);
        std::vector<float> dS((std::size_t)(SeqLen * SeqLen), 0.0f);
        for (int h = 0; h < heads; h++) {
            const float *Ph = &P[(std::size_t)(h * SeqLen * SeqLen)];
            for (int i = 0; i < SeqLen; i++) {
                /* dP_ij = Σ_c dOut_ic · V_jc   (再做 dropout 的 mask: P 的那一层已经乘过) */
                for (int j = 0; j < SeqLen; j++) {
                    float acc = 0;
                    for (int c = 0; c < dk; c++) {
                        acc += dOut[(std::size_t)(i * tokDim + h * dk + c)]
                             * V[(std::size_t)(j * tokDim + h * dk + c)];
                    }
                    if (maskP != nullptr) { acc *= (*maskP)[(std::size_t)(h * SeqLen * SeqLen + i * SeqLen + j)]; }
                    dS[(std::size_t)(i * SeqLen + j)] = acc;
                }
                /* 按行 softmax 的雅可比: dS_ij = P_ij (dP_ij − Σ_l P_il dP_il) */
                float dot = 0;
                for (int j = 0; j < SeqLen; j++) {
                    dot += Ph[(std::size_t)(i * SeqLen + j)] * dS[(std::size_t)(i * SeqLen + j)];
                }
                for (int j = 0; j < SeqLen; j++) {
                    dS[(std::size_t)(i * SeqLen + j)] =
                        Ph[(std::size_t)(i * SeqLen + j)]
                        * (dS[(std::size_t)(i * SeqLen + j)] - dot);
                }
            }
            /* dV_jc += Σ_i P_ij dOut_ic ; dQ_ic += Σ_j dS_ij K_jc / √d_k ; dK_jc += Σ_i dS_ij Q_ic / √d_k
               ⚠ dV 用的是**乘过 mask 的** P (前向的 PV 用的是 mask⊙P) —— 只给 dP 乘 mask
                 而给 dV 用未丢弃的 P 是错的 (dropout 的第一版就是这样, 冻住 mask 的
                 有限差分当场把它抓了出来: wq 的梯度偏大 15.6%、embed 偏大 24.6%). */
            for (int j = 0; j < SeqLen; j++) {
                for (int c = 0; c < dk; c++) {
                    float accV = 0, accK = 0;
                    for (int i = 0; i < SeqLen; i++) {
                        float pij = Ph[(std::size_t)(i * SeqLen + j)];
                        if (maskP != nullptr) {
                            pij *= (*maskP)[(std::size_t)(h * SeqLen * SeqLen + i * SeqLen + j)];
                        }
                        accV += pij * dOut[(std::size_t)(i * tokDim + h * dk + c)];
                        accK += dS[(std::size_t)(i * SeqLen + j)] * Q[(std::size_t)(i * tokDim + h * dk + c)];
                    }
                    dV[(std::size_t)(j * tokDim + h * dk + c)] += accV;
                    dK[(std::size_t)(j * tokDim + h * dk + c)] += accK * scale;
                }
            }
            for (int i = 0; i < SeqLen; i++) {
                for (int c = 0; c < dk; c++) {
                    float acc = 0;
                    for (int j = 0; j < SeqLen; j++) {
                        acc += dS[(std::size_t)(i * SeqLen + j)] * K[(std::size_t)(j * tokDim + h * dk + c)];
                    }
                    dQ[(std::size_t)(i * tokDim + h * dk + c)] += acc * scale;
                }
            }
        }
    }

    /*
       [2026-10] 位置编码的优化器/范数/缩放/软更新/往返都搬进了 `SeqPosEnc`
       (与 `SeqLayerNorm` 同一做法)。原来这里是 `sgdParam/rmsParam/adamParam/
       norm2Param/scaleParam` 五个自由函数 + 内嵌的 `Param` 结构 —— 它们**只为
       位置嵌入服务**, 而位置嵌入现在按模式持有 0/1/2 组参数, 所以那五个函数
       全部删掉 (参数组数可变 ⇒ 用不了"一个固定成员一个函数"的写法)。
    */
};

/* 工厂: `hidden` 在这里是 token 宽度 (与 MlpExpert 用 hidden 的口径一致) */
template<int SeqLen, int Heads, int Blocks>
struct ExpertFactory<SeqTransformerExpert<SeqLen, Heads, Blocks> >
{
    static SeqTransformerExpert<SeqLen, Heads, Blocks> make(int d_model, int hidden, bool withGrad)
    {
        return SeqTransformerExpert<SeqLen, Heads, Blocks>(d_model, hidden > 0 ? hidden : 64, withGrad);
    }
};

/* 初始化缩放: 全 FC 按 1/√fan_in (见 expert.hpp 顶部) */
template<int SeqLen, int Heads, int Blocks>
void scaleExpertInit(SeqTransformerExpert<SeqLen, Heads, Blocks> &e)
{
    e.scaleInit();
}

} /* namespace RL */

#endif // RL_SEQ_TRANSFORMER_HPP
