#ifndef RL_CONV_RESNET_HPP
#define RL_CONV_RESNET_HPP
/*
 * ConvResNetExpert —— 把棋盘当**图像**看的 MoE 专家 (19 通道 x 10 行 x 9 列)
 * ============================================================================
 *
 * 为什么要它 (一句话): 现有三个骨干 (TB / MLP / 序列 Transformer) 都把一个 1710 维向量
 * 当**无结构的词袋**加工 —— 而状态编码本身就是一张 `(19, 10, 9)` 的图像
 * (`cell = x*9 + y`, x 是纵向 0..9, y 是横向 0..8)。卷积把这份结构**还回去**:
 * 平移等变 + 权重共享 + 层数换感知野。完整论证见 `docs/conv_resnet_moe_analysis.md`
 * (那份文档 §6 给出的候选结构就是本文件的形状)。
 *
 * ---- 形状与口径 (用户口径: "19x9x10 特征输入") ----
 *
 *   状态 (dIn x 1) = 平面优先: `state[plane * 90 + cell]`, `cell = x*9 + y`
 *        │  零拷贝语义的重排 (数据本来就是"平面 -> 行 -> 列"的行主序)
 *        ▼
 *   图像 (19, 10, 9)   ← 19 通道 x 10 行 x 9 列。用户写的 "19x9x10" 就是这个张量
 *                        (19 通道, 盘面 9 宽 x 10 高); 本文件一律按 (通道, 行, 列) 写,
 *                        因为 `cell = x*9 + y` 的行主序**不能与列主序混用** ——
 *                        混了就是把卷积核转置, 那是静默错位。
 *        │
 *        │  + 4 个**坐标/区域通道** (构造参数 `useCoordPlanes`, 默认开): 行 ramp /
 *        │    列 ramp / 河界 / 九宫。零参数, 但补上了卷积**结构上缺**的绝对坐标:
 *        │    兵/象/士的走法在河两侧不同, 将/士永不九宫 —— 纯平移等变在这里是
 *        │    **部分错误**的假设 (doc §6.1 第 1 条: 不加这两类平面是"确定的、
 *        │    可预测的性能损失")。⚠ 它**不是零代价开关**: 通道数从 19 变 23 ⇒
 *        │    conv1 的 W 从 (32,171) 变 (32,207), 参数量差 1,152 ⇒ **两种口径的权重
 *        │    互不兼容**(必须各自训练)。要做这个 A/B 就两个都训一遍, 别指望热切换。
 *        ▼
 *   conv1 : Conv2d(23 -> C, 3x3, pad=1) + Tanh
 *   Blocks: Conv2d(C -> C, 3x3, pad=1) + Tanh, **加残差**        (C = `channels`, 默认 32)
 *   head  : Conv2d 1x1 (C -> 19), **线性** (不加激活)  → 展平回 (dIn x 1)
 *
 * ---- 四个刻意的设计决定 (每条都对应一个具体的坏结果) ----
 *
 *  1. **不做归一化** (doc §4 的结论 c)。现有 `LayerNorm` 系列对**整个张量**求一个标量
 *     均值/方差 (把 (C,10,9) 喂进去 = 把空间信息抹平; 而且那个 gamma 是 forward 里算出来
 *     的 scalar 缓存, 不是可学参数); BatchNorm 更不能用 —— 本工程"一条样本一次更新",
 *     且 MCTS 推理前向与训练前向**共用同一批层对象**, running statistics 会被推理污染
 *     (与 SparseMoE 的批统计是同一个问题)。所以用"残差支路初始化接近恒等"代替:
 *     权重按 `1/sqrt(fan_in)` 缩放 ⇒ 每个残差块初始注入 `|Tanh(conv(x))|` 是小量,
 *     恒等路占优 ⇒ 深度不是从"随机破坏前几层"开始的。
 *  2. **残差是必需的, 不是顺手加的** (doc §4 第 2 条): 本工程的 `RMSProp` 把每个张量的
 *     梯度归一到单位长度 (`clipGrad`), 于是**深浅层拿一样的步长** —— 在这种"恒步长"
 *     regime 里, 恒等路是唯一能让"至少不比浅层差"的起点。另外 top-1 MoE 的
 *     `o = Σ gate_i·expert_i(x)` (Σgate = 1) 放进残差流时, 未被选中的专家贡献**恰好为 0**,
 *     而不是一份随机噪声。
 *  3. **逐点 (1x1) 输出投影 + 展平**, 而不是"拍平后接稠密大矩阵":
 *     `Linear(32*90 -> 1710)` 是 4.9 M 参数/专家 (MoE 还要乘 E), 而 `Conv 1x1(32->19)`
 *     只要 608 个参数, 且保留"每格输出只由那一格 (及其邻域) 决定"的结构。
 *  4. **卷积走 im2col + `MM::ikkj/ikjk`** (仓库实测 0.09~0.33 ns/MAC), 而不是
 *     `conv2d.hpp` 里那个朴素三重循环 (ConvDQN 时代实测 **~10 ns/MAC** ⇒ 本专家
 *     4 M MAC/前向会变成 ~40 ms/前向, 直接不可用)。所以**不共用也不修改** `conv2d()`:
 *     ConvDQN 的数值一位都不许动。
 *
 * ---- 参数量与算力 (C=32 / Blocks=4; 实测读数见 docs) ----
 *
 *   conv1 (23->32,3x3): 6,656 | 每个块 (32->32,3x3): 9,248 | head (32->19,1x1): 627
 *   ⇒ **每专家 44,275 参数**, 前向 **3.97 M MAC**
 *   ⇒ PPO actor = 4×44,275 + 门控 6,844 + 头 636,004 = **819,948** 参数
 *     (对照: seq 骨干 1,009,772, TB 骨干 52,388,888)
 *
 * ---- 与其它骨干的边界 ----
 *   * `dIn` 必须能被 `Rows*Cols = 90` 整除 ⇒ PPO 的 1710 = 19x90 ✓;
 *     SAC/DQN 那条 1263 = 17x90+3 **不满足** (3 个规则上下文是额外追加的标量),
 *     构造时**响亮失败**, 不静默取整 (与 SeqTransformerExpert 同一条纪律)。
 *   * **写序是权重格式的一部分** (不许改): `conv1.w, conv1.b, [每块 w,b], head.w, head.b`,
 *     每个张量一行。顺序变了 ⇒ 存量权重会**错位读入而参数量对得上**,
 *     所以 `test_ppo_backbone [7]` 用 Net 级往返 (同一输入逐位相同) 钉住它。
 *   * 前向/反向**必须成对**: 反向要用最近一次前向的 `pre`/`col` 缓存 (与仓库其它层同一
 *     约定); 反向累加进 `ei` (MoE 的门控把多路专家的梯度叠在同一个缓冲上),
 *     且**只对选中它的那份门控权重负责** —— 缩放由 SparseMoE 在调用前写进 `e`。
 */
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "expert.hpp"     /* ExpertFactory 的主模板声明 (本文件在后面特化它) */
#include "ilayer.h"
#include "optimize.h"
#include "tensor.hpp"
#include "util.hpp"

namespace RL {

/* ============================================================
 *  坐标/区域通道 (4 个, 零参数)
 * ============================================================
 *  规范视角: **轮到谁走, 谁的子就在 x 大的那一侧** (见 chessstate.h 的 canonicalCell),
 *  所以 x ∈ {rows-3..rows-1} 是"我这一侧的九宫", x <= 2 是对方九宫。
 */
inline void convCoordPlanes(std::vector<float> &dst, int rows, int cols)
{
    const int N = rows * cols;
    dst.assign((std::size_t)4 * N, 0.0f);
    const float invRows = (rows > 1) ? 1.0f / (float)(rows - 1) : 0.0f;
    const float invCols = (cols > 1) ? 1.0f / (float)(cols - 1) : 0.0f;
    const int mid = (rows + 1) / 2;          /* 半盘分界 (x >= mid 是我这一侧) */
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            const int cell = x * cols + y;
            dst[(std::size_t)(0 * N + cell)] = (float)x * invRows;   /* 行 ramp */
            dst[(std::size_t)(1 * N + cell)] = (float)y * invCols;   /* 列 ramp */
            dst[(std::size_t)(2 * N + cell)] = (x >= mid) ? 1.0f : 0.0f;   /* 河界 */
            const bool midCols = (y >= 3 && y <= 5);
            const bool palace = (x >= rows - 3) || (x <= 2);
            dst[(std::size_t)(3 * N + cell)] = (midCols && palace) ? 1.0f : 0.0f;  /* 九宫 */
        }
    }
}

/* ============================================================
 *  ConvParam —— 一个卷积层的全部状态 (参数 / 梯度 / 动量 / 缓存)
 * ============================================================
 *  W 存成 `(outC, inC*k*k)` 的**行主序平铺**: 这正好是 im2col 之后 GEMM 需要的 (M,K),
 *  也是 `write/read` 写出去的那一行。b 是 `(outC,1)`。
 */
class ConvParam
{
public:
    int inC = 0, outC = 0, k = 3;
    bool withGrad = false;
    Tensor w, b, gw, gb, vw, vb, mw, mb;
    std::vector<float> pre, act;      /* 最近一次前向: (outC*CELLS) 各一份 */

    void init(int inC_, int outC_, int k_, int cells, bool withGrad_)
    {
        inC = inC_; outC = outC_; k = k_; withGrad = withGrad_;
        const std::size_t K = (std::size_t)inC * k * k;
        w = Tensor((std::size_t)outC, K);
        b = Tensor((std::size_t)outC, 1);
        Random::uniform(w, -1.0f, 1.0f);
        Random::uniform(b, -1.0f, 1.0f);
        /* fan-in 缩放: `U(-1,1)` 直接用在 207 维的 patch 上会把 Tanh 顶到饱和
           (与 expert.hpp 的 scaleFcInit 同一个理由)。注意**不额外缩小**残差支路:
           1/sqrt(fan_in) 已经把初始注入压成小量, 这正是 doc §4 要的"接近恒等的起点"。 */
        const float s = 1.0f / std::sqrt((float)(K > 1 ? K : 1));
        for (std::size_t i = 0; i < w.size(); i++) { w[i] *= s; }
        for (std::size_t i = 0; i < b.size(); i++) { b[i] *= s; }
        if (withGrad) {
            gw = Tensor((std::size_t)outC, K);
            gb = Tensor((std::size_t)outC, 1);
            vw = Tensor((std::size_t)outC, K);
            vb = Tensor((std::size_t)outC, 1);
            mw = Tensor((std::size_t)outC, K);
            mb = Tensor((std::size_t)outC, 1);
            gw.zero(); gb.zero(); vw.zero(); vb.zero(); mw.zero(); mb.zero();
        }
        pre.assign((std::size_t)outC * cells, 0.0f);
        act.assign((std::size_t)outC * cells, 0.0f);
    }
    long long paramCount() const { return (long long)w.size() + (long long)b.size(); }
    void zeroGrad() { if (withGrad) { gw.zero(); gb.zero(); } }
    void SGD(float lr)
    {
        if (!withGrad) { return; }
        Optimize::SGD(w, gw, lr);
        Optimize::SGD(b, gb, lr);
        zeroGrad();
    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad)
    {
        if (!withGrad) { return; }
        Optimize::RMSProp(w, vw, gw, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, vb, gb, lr, rho, decay, clipGrad);
        zeroGrad();
    }
    void Adam(float lr, float alpha, float beta_, float alpha_, float beta__,
              float decay, bool clipGrad)
    {
        if (!withGrad) { return; }
        Optimize::Adam(w, vw, mw, gw, alpha_, beta__, lr, alpha, beta_, decay, clipGrad);
        Optimize::Adam(b, vb, mb, gb, alpha_, beta__, lr, alpha, beta_, decay, clipGrad);
        zeroGrad();
    }
    void clamp(float c0, float cn) { Optimize::clamp(w, c0, cn); Optimize::clamp(b, c0, cn); }
    double gradNorm2() const
    {
        return withGrad ? (gradNorm2Of(gw) + gradNorm2Of(gb)) : 0.0;
    }
    void scaleGrad(float s)
    {
        if (!withGrad) { return; }
        scaleTensorGrad(gw, s);
        scaleTensorGrad(gb, s);
    }
    void writeTo(std::ostream &f) const
    {
        f << w.toString() << std::endl;
        f << b.toString() << std::endl;
    }
    void readFrom(std::istream &f)
    {
        std::string s;
        std::getline(f, s); w = Tensor::fromString(s);
        std::getline(f, s); b = Tensor::fromString(s);
    }
    void copyParamsTo(ConvParam &o) const { o.w = w; o.b = b; }
    void softUpdateParamsTo(ConvParam &o, float alpha) const
    {
        const std::size_t nw = (o.w.size() < w.size()) ? o.w.size() : w.size();
        for (std::size_t i = 0; i < nw; i++) { o.w[i] += alpha * (w[i] - o.w[i]); }
        const std::size_t nb = (o.b.size() < b.size()) ? o.b.size() : b.size();
        for (std::size_t i = 0; i < nb; i++) { o.b[i] += alpha * (b[i] - o.b[i]); }
    }
};

/* ============================================================
 *  ConvResNetExpert
 * ============================================================ */
class ConvResNetExpert : public iLayer
{
public:
    static const int ROWS = 10;      /* x: 0..9 */
    static const int COLS = 9;       /* y: 0..8 */
    static const int CELLS = ROWS * COLS;
    static const int COORD_PLANES = 4;

    int dIn = 0;
    int inC = 0;                 /* = dIn / CELLS (PPO: 19) */
    int chans = 32;
    int blocks = 4;
    bool useCoordPlanes = true;
    bool withGradFlag = false;

    ConvParam conv1;
    std::vector<ConvParam> blk;
    ConvParam head;

    /* 一次性分配的缓冲 (前向/反向都不再分配) */
    std::vector<float> img;        /* (inC+4) * CELLS */
    std::vector<float> outImg;     /* inC * CELLS */
    std::vector<float> dImg;       /* (inC+4) * CELLS -- conv1 的输入梯度 */
    std::vector<float> dCur, dNxt, dPre;   /* chans * CELLS */
    std::vector<float> dOut;               /* dIn -- 上游梯度 (来自 e) 摊成图像前的副本 */
    Tensor col1;                   /* (inCh*9, CELLS)  conv1 的 im2col */
    Tensor colB;                   /* (chans*9, CELLS) 残差块的 im2col */
    Tensor colH;                   /* (chans,   CELLS) head 的 im2col (1x1 ⇒ K = chans) */

    ConvResNetExpert() {}
    ConvResNetExpert(int dIn_, int chans_, int blocks_, bool withGrad,
                     bool useCoordPlanes_ = true)
    {
        init(dIn_, chans_, blocks_, withGrad, useCoordPlanes_);
    }

    void init(int dIn_, int chans_, int blocks_, bool withGrad, bool useCoordPlanes_ = true)
    {
        if (dIn_ <= 0 || dIn_ % CELLS != 0) {
            std::fprintf(stderr,
                         "[ConvResNetExpert] dIn=%d 不能被 %d (= %dx%d) 整除 —— 本专家要求"
                         "状态是**平面优先的棋盘图像** (cell = x*9+y)。\n"
                         "  PPO 的 1710 = 19x90 满足; SAC/DQN 的 1263 = 17x90+3 不满足"
                         " (3 个规则上下文是额外追加的标量), 那两支要先改编码口径。\n",
                         dIn_, CELLS, ROWS, COLS);
            std::fflush(stderr);
            std::abort();
        }
        dIn = dIn_;
        inC = dIn / CELLS;
        chans = (chans_ > 0) ? chans_ : 32;
        blocks = (blocks_ >= 0) ? blocks_ : 4;
        useCoordPlanes = useCoordPlanes_;
        withGradFlag = withGrad;
        type = LAYER_CONVRESNET;

        const int inCh = inChannelsWithCoord();
        conv1.init(inCh, chans, 3, CELLS, withGrad);
        blk.clear();
        blk.reserve((std::size_t)blocks);
        for (int i = 0; i < blocks; i++) {
            ConvParam l;
            l.init(chans, chans, 3, CELLS, withGrad);
            blk.push_back(l);
        }
        head.init(chans, inC, 1, CELLS, withGrad);

        img.assign((std::size_t)inCh * CELLS, 0.0f);
        outImg.assign((std::size_t)inC * CELLS, 0.0f);
        dImg.assign((std::size_t)inCh * CELLS, 0.0f);
        dCur.assign((std::size_t)chans * CELLS, 0.0f);
        dNxt.assign((std::size_t)chans * CELLS, 0.0f);
        dPre.assign((std::size_t)chans * CELLS, 0.0f);
        dOut.assign((std::size_t)dIn, 0.0f);
        /*
           三个 im2col 缓冲的**形状必须与各自那一层的 K 精确相等** —— `MM::` 内核
           带常驻形状契约检查 (Release 也生效), 给大了会在前向/反向当场 abort,
           而不是"算出一堆看起来合理的数字"。head 是 1x1 ⇒ K = chans, 所以它单独一份。
        */
        col1 = Tensor((std::size_t)inCh * 9, (std::size_t)CELLS);
        colB = Tensor((std::size_t)chans * 9, (std::size_t)CELLS);
        colH = Tensor((std::size_t)chans, (std::size_t)CELLS);

        o = Tensor((std::size_t)dIn, 1);
        e = Tensor((std::size_t)dIn, 1);
    }

    /* ---------------- 只读读数 (自检面板 / 测试) ---------------- */
    long long paramCount() const override
    {
        long long t = conv1.paramCount() + head.paramCount();
        for (std::size_t i = 0; i < blk.size(); i++) { t += blk[i].paramCount(); }
        return t;
    }
    int inputChannels() const { return inC; }
    int inChannelsWithCoord() const { return inC + (useCoordPlanes ? COORD_PLANES : 0); }
    int blockCount() const { return (int)blk.size(); }
    int channelCount() const { return chans; }
    bool coordPlanesInUse() const { return useCoordPlanes; }
    /* 一次前向的 MAC (与序列专家的 9.67 M 是同一口径的读数) */
    long long forwardMac() const
    {
        const long long inCh = inChannelsWithCoord();
        long long mac = (long long)chans * inCh * 9 * CELLS;
        mac += (long long)blocks * chans * chans * 9 * CELLS;
        mac += (long long)inC * chans * CELLS;
        return mac;
    }

    /* ============================================================
     *  im2col / col2im
     * ============================================================
     *  im2col: (Cin, ROWS, COLS) -> (Cin*k*k, CELLS) **行主序 (kernel, position)**。
     *  这个布局同时满足两个 MM 内核的契约, 所以反向**不需要**再转置一次:
     *    * 前向: z(M,N) += W(M,K) · X(K,N)         -> MM::ikkj(z, W, X)
     *    * dW  : dW(M,K) += dY(M,N) · X(K,N)ᵀ      -> MM::ikjk(dW, dY, X)
     *  (ikjk 的语义是 z(i,j) += Σ_k x1(i,k)·x2(j,k), 这里 x2 的"行"就是 kernel 下标,
     *   正好对上 X 的第一维。)
     */
    static void im2col(const float *image, int Cin, int k, float *colOut)
    {
        const int K2 = k * k;
        const int pad = k / 2;
        for (int ic = 0; ic < Cin; ic++) {
            const float *src = image + (std::size_t)ic * CELLS;
            for (int u = 0; u < k; u++) {
                for (int v = 0; v < k; v++) {
                    float *dst = colOut + (std::size_t)(ic * K2 + u * k + v) * CELLS;
                    for (int i = 0; i < ROWS; i++) {
                        const int ii = i + u - pad;
                        if (ii < 0 || ii >= ROWS) { continue; }
                        const float *srow = src + ii * COLS;
                        float *drow = dst + i * COLS;
                        for (int j = 0; j < COLS; j++) {
                            const int jj = j + v - pad;
                            if (jj < 0 || jj >= COLS) { continue; }
                            drow[j] = srow[jj];
                        }
                    }
                }
            }
        }
    }
    /* col2im: 反向散加 (**必须是 +=**: 一个输入位置被 k*k 个 patch 覆盖) */
    static void col2imAdd(const float *dcol, int Cin, int k, float *dImageAdd)
    {
        const int K2 = k * k;
        const int pad = k / 2;
        for (int ic = 0; ic < Cin; ic++) {
            float *dst = dImageAdd + (std::size_t)ic * CELLS;
            for (int u = 0; u < k; u++) {
                for (int v = 0; v < k; v++) {
                    const float *src = dcol + (std::size_t)(ic * K2 + u * k + v) * CELLS;
                    for (int i = 0; i < ROWS; i++) {
                        const int ii = i + u - pad;
                        if (ii < 0 || ii >= ROWS) { continue; }
                        const float *srow = src + i * COLS;
                        float *drow = dst + ii * COLS;
                        for (int j = 0; j < COLS; j++) {
                            const int jj = j + v - pad;
                            if (jj < 0 || jj >= COLS) { continue; }
                            drow[jj] += srow[j];
                        }
                    }
                }
            }
        }
    }

    /* ============================================================
     *  前向
     * ============================================================ */
    Tensor &forward(const Tensor &x, bool inference = false) override
    {
        (void)inference;   /* 缓存总是写: 反向要用, 而"只在推理时省缓存"会让
                              训练/推理两条路的数值分叉 (仓库其它层也是这个口径) */
        /* 1) 状态 -> 图像 */
        for (int p = 0; p < inC; p++) {
            const float *src = x.data() + (std::size_t)p * CELLS;
            float *dst = &img[(std::size_t)p * CELLS];
            for (int c = 0; c < CELLS; c++) { dst[c] = src[c]; }
        }
        if (useCoordPlanes) {
            std::vector<float> coord;
            convCoordPlanes(coord, ROWS, COLS);
            for (std::size_t c = 0; c < coord.size(); c++) {
                img[(std::size_t)inC * CELLS + c] = coord[c];
            }
        }

        /* 2) conv1 + Tanh */
        im2col(img.data(), inChannelsWithCoord(), 3, col1.data());
        convFwd(conv1, col1, true);

        /* 3) 残差块: act = a + Tanh(W * im2col(a)) */
        for (std::size_t i = 0; i < blk.size(); i++) {
            const std::vector<float> &in = (i == 0) ? conv1.act : blk[i - 1].act;
            im2col(in.data(), chans, 3, colB.data());
            convFwd(blk[i], colB, true);
            for (std::size_t c = 0; c < blk[i].act.size(); c++) { blk[i].act[c] += in[c]; }
        }

        /* 4) head (1x1, 线性) -> 输出图像 -> 状态向量 */
        im2col(lastAct().data(), chans, 1, colH.data());
        convFwd(head, colH, false);
        for (int j = 0; j < inC * CELLS; j++) { outImg[(std::size_t)j] = head.act[(std::size_t)j]; }
        for (int j = 0; j < dIn; j++) { o[(std::size_t)j] = outImg[(std::size_t)j]; }
        return o;
    }

    /* ============================================================
     *  反向 (累加进 ei)
     * ============================================================
     *  上游梯度在成员 `e` 里 (SparseMoE 已按门控权重缩放: `e = gate_i * dL/do`)。
     */
    void backward(const Tensor &x, Tensor &ei) override
    {
        (void)x;
        /* 1) head: 线性层 ⇒ dPre = dAct */
        for (int j = 0; j < dIn; j++) { dOut[(std::size_t)j] = e[(std::size_t)j]; }
        dCur.assign((std::size_t)chans * CELLS, 0.0f);
        convBackward(head, colH, dOut.data(), dCur.data());

        /* 2) 残差块 (从后往前) */
        for (int i = (int)blk.size() - 1; i >= 0; i--) {
            /* 残差路: d(输入) += dAct */
            for (std::size_t c = 0; c < dCur.size(); c++) { dNxt[c] = dCur[c]; }
            /* Tanh 路: dPre = dAct ⊙ (1 − tanh(pre)²)。
               ⚠ `act` 里存的是**加过残差**的值, 所以这里必须用 `tanh(pre)` 重算导数,
               不能拿 act 当 tanh 输出 (这是本文件最容易写错的一处, 用有限差分钉住) */
            for (std::size_t c = 0; c < dPre.size(); c++) {
                const float t = std::tanh(blk[(std::size_t)i].pre[c]);
                dPre[c] = dCur[c] * (1.0f - t * t);
            }
            const std::vector<float> &in = (i == 0) ? conv1.act : blk[(std::size_t)i - 1].act;
            im2col(in.data(), chans, 3, colB.data());
            convBackward(blk[(std::size_t)i], colB, dPre.data(), dNxt.data());
            dCur.swap(dNxt);
        }

        /* 3) conv1: Tanh */
        for (std::size_t c = 0; c < dPre.size(); c++) {
            const float t = std::tanh(conv1.pre[c]);
            dPre[c] = dCur[c] * (1.0f - t * t);
        }
        for (std::size_t c = 0; c < dImg.size(); c++) { dImg[c] = 0.0f; }
        convBackward(conv1, col1, dPre.data(), dImg.data());

        /* 4) 图像 -> 状态 (只回传前 inC 个平面; 坐标通道没有上游梯度) */
        for (int j = 0; j < dIn; j++) { ei[(std::size_t)j] += dImg[(std::size_t)j]; }

        /* 与仓库其它专家一致: 用完清掉缓存 (MoE 的 forward/backward 逐样本成对) */
        o.zero();
        e.zero();
    }

    /* ---------------- 优化器 / 范数 / 裁剪 / 复制 ---------------- */
    void SGD(float lr) override
    {
        conv1.SGD(lr);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].SGD(lr); }
        head.SGD(lr);
    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        conv1.RMSProp(lr, rho, decay, clipGrad);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].RMSProp(lr, rho, decay, clipGrad); }
        head.RMSProp(lr, rho, decay, clipGrad);
    }
    void Adam(float lr, float alpha, float beta, float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        conv1.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        for (std::size_t i = 0; i < blk.size(); i++) {
            blk[i].Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        }
        head.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
    }
    void clamp(float c0, float cn) override
    {
        conv1.clamp(c0, cn);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].clamp(c0, cn); }
        head.clamp(c0, cn);
    }
    double gradNorm2() const override
    {
        double s = conv1.gradNorm2() + head.gradNorm2();
        for (std::size_t i = 0; i < blk.size(); i++) { s += blk[i].gradNorm2(); }
        return s;
    }
    void scaleGrad(float s) override
    {
        conv1.scaleGrad(s);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].scaleGrad(s); }
        head.scaleGrad(s);
    }
    void zeroGrad()
    {
        conv1.zeroGrad();
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].zeroGrad(); }
        head.zeroGrad();
    }
    void copyTo(iLayer *layer) override
    {
        ConvResNetExpert *p = dynamic_cast<ConvResNetExpert *>(layer);
        if (p == nullptr || p->blk.size() != blk.size() || p->dIn != dIn) { return; }
        conv1.copyParamsTo(p->conv1);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].copyParamsTo(p->blk[i]); }
        head.copyParamsTo(p->head);
    }
    void softUpdateTo(iLayer *layer, float alpha) override
    {
        ConvResNetExpert *p = dynamic_cast<ConvResNetExpert *>(layer);
        if (p == nullptr || p->blk.size() != blk.size() || p->dIn != dIn) { return; }
        conv1.softUpdateParamsTo(p->conv1, alpha);
        for (std::size_t i = 0; i < blk.size(); i++) {
            blk[i].softUpdateParamsTo(p->blk[i], alpha);
        }
        head.softUpdateParamsTo(p->head, alpha);
    }

    /* 写序 (权重格式的一部分, 不许改): conv1(w,b) → 各块(w,b) → head(w,b) */
    void write(std::ostream &file) override
    {
        conv1.writeTo(file);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].writeTo(file); }
        head.writeTo(file);
    }
    void read(std::istream &file) override
    {
        conv1.readFrom(file);
        for (std::size_t i = 0; i < blk.size(); i++) { blk[i].readFrom(file); }
        head.readFrom(file);
    }

private:
    const std::vector<float> &lastAct() const
    {
        return blk.empty() ? conv1.act : blk.back().act;
    }

    /* 线性部分 + 激活, 缓存 pre/act */
    void convFwd(ConvParam &L, const Tensor &col, bool useTanh)
    {
        const int M = L.outC;
        const int N = CELLS;
        Tensor z((std::size_t)M, (std::size_t)N);
        z.zero();
        Tensor::MM::ikkj(z, L.w, col);            /* z = W(M,K) · col(K,N) */
        const float *zd = z.data();
        const float *bd = L.b.data();
        for (int o = 0; o < M; o++) {
            const float bias = bd[o];
            const float *zr = zd + (std::size_t)o * N;
            float *pr = &L.pre[(std::size_t)o * N];
            float *ar = &L.act[(std::size_t)o * N];
            for (int c = 0; c < N; c++) {
                const float v = zr[c] + bias;
                pr[c] = v;
                ar[c] = useTanh ? std::tanh(v) : v;
            }
        }
    }

    /*
       一次卷积的反向:  dY (outC,CELLS) 已给 (§ 的指针形式, 因为不同层的 outC 不同,
       而 `MM::` 的形状契约要求 Tensor 的形状**精确**匹配 ⇒ 这里按层现建 Tensor)
         gW  += dY · colᵀ         (MM::ikjk)
         gb  += Σ_n dY
         dX   = Wᵀ · dY           (MM::ikkj)  → col2im 散加进 dXOut (形状 (inC,ROWS,COLS))
    */
    void convBackward(ConvParam &L, const Tensor &col, const float *dYData, float *dXOut)
    {
        const int M = L.outC;
        const int K = L.inC * L.k * L.k;
        Tensor dY((std::size_t)M, (std::size_t)CELLS);
        for (std::size_t i = 0; i < dY.size(); i++) { dY[i] = dYData[i]; }

        if (L.withGrad) {
            Tensor::MM::ikjk(L.gw, dY, col);       /* gW += dY(M,N) · col(K,N)ᵀ */
            const float *dd = dY.data();
            for (int o = 0; o < M; o++) {
                const float *r = dd + (std::size_t)o * CELLS;
                float s = 0.0f;
                for (int c = 0; c < CELLS; c++) { s += r[c]; }
                L.gb[(std::size_t)o] += s;
            }
        }

        /* dX = Wᵀ·dY: W (M,K) → wT (K,M); K*M 次拷贝相对 K*M*90 的乘加可忽略 */
        Tensor wT((std::size_t)K, (std::size_t)M);
        for (int o = 0; o < M; o++) {
            const float *wr = L.w.data() + (std::size_t)o * K;
            for (int kk = 0; kk < K; kk++) { wT[(std::size_t)kk * M + o] = wr[kk]; }
        }
        Tensor dX((std::size_t)K, (std::size_t)CELLS);
        dX.zero();
        Tensor::MM::ikkj(dX, wT, dY);              /* (K,M)·(M,CELLS) -> (K,CELLS) */
        col2imAdd(dX.data(), L.inC, L.k, dXOut);
    }
};

/* ============================================================
 *  MoE 接入 (expert.hpp 的三条约定: 输入输出同维 / ExpertFactory / scaleExpertInit)
 * ============================================================
 *  `hidden` 在本专家里的语义是**通道数**, 在 MlpExpert 里是隐层宽度、在序列专家里是
 *  token 宽度 —— 同一个位置、不同语义, 所以自检面板必须把骨干名与它一起报出来。
 */
constexpr int CONV_RESNET_CHANNELS = 32;
constexpr int CONV_RESNET_BLOCKS = 4;

template<>
struct ExpertFactory<ConvResNetExpert>
{
    static ConvResNetExpert make(int d_model, int hidden, bool withGrad)
    {
        const int ch = (hidden > 0) ? hidden : CONV_RESNET_CHANNELS;
        return ConvResNetExpert(d_model, ch, CONV_RESNET_BLOCKS, withGrad, true);
    }
};

/* 每层已经按 fan_in 缩放过, 这里不再缩第二遍 (缩两次会把初始注入压没) */
inline void scaleExpertInit(ConvResNetExpert &e) { (void)e; }

} /* namespace RL */

#endif // RL_CONV_RESNET_HPP
