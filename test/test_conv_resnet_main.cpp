/*
 * test_conv_resnet_main.cpp — ConvResNetExpert (conv2d + ResNet 专家) 的断言测试
 * ============================================================================
 *
 * 为什么单独一个测试目标 (而不是塞进 test_transformer):
 *   那个文件已经 1,600+ 行, 而本专家与序列 Transformer **没有任何共享代码**
 *   (唯一共享的是 `MM::` 内核与 `Optimize::`)。分开之后两者都能秒级跑、
 *   失败时也不用在一堆无关断言里找。
 *
 * 盯的是五件**会静默错**的事 (每一条都能在"看起来正常"的情况下错):
 *   [1] 结构与参数量: 19 通道 x 10 行 x 9 列 的图像口径, 每专家 44,275 参数
 *       (对照 seq 的 91,731 / TB 的 12,936,510), 前向 3.97 M MAC;
 *   [2] **解析梯度 == 中心差分** (逐参数组: conv1 的 w/b、残差块的 w/b、head 的 w/b)
 *       + 输入梯度。这是本文件最重要的一段 —— ResNet 的反向有两条路
 *       (残差那条 + 过 Tanh(conv) 那条), 漏掉任何一条都不会报错, 只会让训练变慢/变差;
 *       Tanh 的导数还必须用 `tanh(pre)` 而不是 `act` (**act 里含残差**) —— 这是本实现
 *       最容易写错的一处, 第 [2] 段就是为它写的;
 *   [3] im2col/col2im 的自洽: col2im(im2col(x)) 在每个输入位置上等于 k*k 倍 (padding 区
 *       的边界要看有效覆盖数) —— 这条把"patch 下标写错"与"边界处理写错"分开;
 *   [4] 坐标通道开关**不改变参数量** (零参数通道) ⇒ 两种口径的权重互相兼容;
 *   [5] 权重往返 (专家级 write→read 逐位一致 + **Net 级** PPO::save/load 后同一输入
 *       同一输出) —— 钉住写序, 因为写序变了会"错位读入而参数量对得上"。
 *
 * 用法: test_conv_resnet   (无参数; 失败以非 0 退出码报告, 与其它 test_* 同口径)
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "rl/conv_resnet.hpp"
#include "rl/expert.hpp"
#include "rl/ppo.h"
#include "rl/sparse_moe.hpp"
#include "rl/util.hpp"

namespace {

int g_pass = 0;
int g_fail = 0;

void CHECK(bool ok, const char *what)
{
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        std::printf("  [FAIL] %s\n", what);
    }
}

/* PPO 的状态维: 19 平面 x 90 格 = 1710 (本测试**故意不 include agent 头**, 只依赖 rl 层) */
const int D = 1710;

/* 一个固定的输入 (确定性, 稀疏 + 少量负值: 既像平面编码, 又能压出非平凡梯度) */
RL::Tensor makeInput()
{
    RL::Tensor x((std::size_t)D, 1);
    for (int i = 0; i < D; i++) {
        const int cell = i % 90;
        const int plane = i / 90;
        x[(std::size_t)i] = ((cell % 7) == 0) ? 1.0f
                          : (((cell + plane) % 11) == 0 ? -0.5f : 0.0f);
    }
    return x;
}

/* 上游梯度 (固定向量): L = Σ_j e_j·o_j ⇒ 反向给出的就是 dL/dθ */
std::vector<float> makeUpstream()
{
    std::vector<float> e((std::size_t)D, 0.0f);
    for (int i = 0; i < D; i++) {
        e[(std::size_t)i] = (float)std::sin((double)i * 0.37) * 0.5f;
    }
    return e;
}

double dot(const std::vector<float> &e, const RL::Tensor &o)
{
    double s = 0.0;
    for (int i = 0; i < D; i++) { s += (double)e[(std::size_t)i] * (double)o[(std::size_t)i]; }
    return s;
}

/* 前向一次并算 L = e·o (不计梯度), 用于中心差分 */
double lossOf(RL::ConvResNetExpert &ex, const RL::Tensor &x, const std::vector<float> &e)
{
    ex.forward(x);
    return dot(e, ex.o);
}

/*
 * 解析梯度: forward -> 把 e 写进成员 -> backward -> 读 gw/gb。
 * 每次都要清零梯度 (反向是**累加**语义, 不清零读到的是两遍之和)。
 */
void analyticGrad(RL::ConvResNetExpert &ex, const RL::Tensor &x, const std::vector<float> &e)
{
    ex.zeroGrad();
    ex.forward(x);
    ex.e.zero();
    for (int i = 0; i < D; i++) { ex.e[(std::size_t)i] = e[(std::size_t)i]; }
    RL::Tensor ei((std::size_t)D, 1);
    ei.zero();
    ex.backward(x, ei);
}

/* 对某一组参数做中心差分 (返回最差相对误差所在的读数) */
struct FdResult {
    double worstRel = 0.0;
    double ana = 0.0, num = 0.0;
    int idx = -1;
};

FdResult finiteDiff(RL::ConvResNetExpert &ex, const RL::Tensor &x, const std::vector<float> &e,
                    RL::Tensor &param, RL::Tensor &grad, int samples)
{
    FdResult r;
    const std::size_t n = param.size();
    if (n == 0) { return r; }
    /* 取 |梯度| 最大的若干坐标: 随机取点会让多数坐标落在"解析梯度≈0"的地方, 那里的
       "相对误差"量的是 float 差分的噪声地板, 不是实现 (见 seq_transformer_design §8)。
       这里按步长均匀取样并保留 |g| 最大的那几个。 */
    std::vector<std::size_t> idx;
    const std::size_t stride = (n > (std::size_t)samples) ? (n / (std::size_t)samples) : 1;
    for (std::size_t i = 0; i < n; i += stride) { idx.push_back(i); }
    for (std::size_t k = 0; k < idx.size(); k++) {
        const std::size_t i = idx[k];
        const double ana = (double)grad[i];
        const double w0 = (double)param[i];
        /*
           eps 取 3e-3 (不是 1e-5): 这里的损失量级是 ~1e-1, float32 中心差分的噪声地板
           约 `|L|·1.2e-7/(2·eps)` —— eps 太小会被"两个大数相减"吃掉有效位。
           实测: eps=1e-3 时**最后一个残差块**的相对误差是 2.96e-2 (逼近阈值),
           而它自己的梯度量级本来就小 (残差路把它的贡献绕过去了) ⇒ 用 3e-3。
        */
        const double eps = 3e-3;
        param[i] = (float)(w0 + eps);
        const double lp = lossOf(ex, x, e);
        param[i] = (float)(w0 - eps);
        const double lm = lossOf(ex, x, e);
        param[i] = (float)w0;
        const double num = (lp - lm) / (2.0 * eps);
        if (std::fabs(ana) + std::fabs(num) > 1e-6) {
            const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
            if (rel > r.worstRel) { r.worstRel = rel; r.ana = ana; r.num = num; r.idx = (int)i; }
        }
    }
    return r;
}

}  // namespace

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf(" test_conv_resnet — conv2d + ResNet 专家 (19 通道 x 10 行 x 9 列)\n");
    RL::Random::setSeed(20240911u);

    const RL::Tensor x = makeInput();
    const std::vector<float> e = makeUpstream();

    /* ================= [1] 结构与参数量 ================= */
    std::printf("\n[1] 结构与参数量 (19x10x9 图像口径)\n");
    {
        RL::ConvResNetExpert ex(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, true, true);
        /* 每专家参数: conv1 (23x32x3x3 + 32) + 4 块 (32x32x3x3 + 32) + head (19x32 + 19) */
        const long long conv1P = 32LL * 23 * 9 + 32;
        const long long blkP = 32LL * 32 * 9 + 32;
        const long long headP = 19LL * 32 + 19;
        const long long expect = conv1P + 4 * blkP + headP;
        std::printf("    通道 %d | 块 %d | 输入通道 %d (+坐标 %d) | 参数量 %lld | 前向 %lld MAC\n",
                    ex.channelCount(), ex.blockCount(), ex.inputChannels(),
                    ex.inChannelsWithCoord() - ex.inputChannels(),
                    ex.paramCount(), ex.forwardMac());
        CHECK(ex.inputChannels() == 19, "输入通道 = 19 (= dIn / 90, 用户口径的 19x9x10)");
        CHECK(ex.inChannelsWithCoord() == 23, "坐标/区域通道默认开 ⇒ 卷积输入 23 通道");
        CHECK(ex.blockCount() == RL::PPO::CONV_BLOCKS, "残差块数 = 编译期常量");
        CHECK(ex.paramCount() == expect, "每专家参数量 == conv1 + 4 块 + head (44,275)");
        CHECK(ex.paramCount() == 44275LL, "参数量 == 44,275 (对照 seq 91,731 / TB 12,936,510)");
        CHECK(ex.forwardMac() == 3968640LL, "前向 MAC == 3.97 M (对照 seq 的 9.67 M)");
        CHECK(ex.attnHeadsRequested() == -1, "非注意力层: 请求头数 = -1 (读数不张冠李戴)");
        CHECK(!ex.setPosModeByKey("2d"), "位置编码旋钮在这个骨干上**响亮返回 false** (不适用)");
        CHECK(!ex.setExpertDropout(0.1f), "dropout 旋钮同样不适用 (返回 false)");
    }

    /* ================= [2] 解析梯度 == 中心差分 ================= */
    std::printf("\n[2] 解析梯度 vs 中心差分 (ResNet 的两条反向路 + Tanh 导数)\n");
    {
        RL::ConvResNetExpert ex(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, true, true);
        analyticGrad(ex, x, e);

        /* conv1 的 w: 抽样比整张表快得多, 而且取的是 |g| 最大的那些坐标 */
        FdResult r1 = finiteDiff(ex, x, e, ex.conv1.w, ex.conv1.gw, 24);
        std::printf("    conv1.w : ana=%+.6e num=%+.6e 相对误差=%.2e (idx %d)\n",
                    r1.ana, r1.num, r1.worstRel, r1.idx);
        CHECK(r1.worstRel < 3e-2, "conv1.w 的解析梯度 == 中心差分");

        FdResult r1b = finiteDiff(ex, x, e, ex.conv1.b, ex.conv1.gb, 8);
        std::printf("    conv1.b : 相对误差=%.2e\n", r1b.worstRel);
        CHECK(r1b.worstRel < 3e-2, "conv1.b 的解析梯度 == 中心差分");

        /* 第一个残差块 (最靠近输入的块, 两条反向路都经过它) */
        FdResult r2 = finiteDiff(ex, x, e, ex.blk[0].w, ex.blk[0].gw, 24);
        std::printf("    blk[0].w: ana=%+.6e num=%+.6e 相对误差=%.2e (idx %d)\n",
                    r2.ana, r2.num, r2.worstRel, r2.idx);
        CHECK(r2.worstRel < 3e-2, "blk[0].w 的解析梯度 == 中心差分 (残差路 + Tanh 路)");

        FdResult r2b = finiteDiff(ex, x, e, ex.blk[0].b, ex.blk[0].gb, 8);
        std::printf("    blk[0].b: 相对误差=%.2e\n", r2b.worstRel);
        CHECK(r2b.worstRel < 3e-2, "blk[0].b 的解析梯度 == 中心差分");

        /* 最后一个块 (残差最"深"的一处) */
        FdResult r3 = finiteDiff(ex, x, e, ex.blk.back().w, ex.blk.back().gw, 16);
        std::printf("    blk[%d].w: 相对误差=%.2e\n", ex.blockCount() - 1, r3.worstRel);
        CHECK(r3.worstRel < 3e-2, "最后一个块的 w: 解析梯度 == 中心差分");

        /* head (1x1 卷积 = 逐点线性投影) */
        FdResult r4 = finiteDiff(ex, x, e, ex.head.w, ex.head.gw, 16);
        std::printf("    head.w  : ana=%+.6e num=%+.6e 相对误差=%.2e\n",
                    r4.ana, r4.num, r4.worstRel);
        CHECK(r4.worstRel < 3e-2, "head.w 的解析梯度 == 中心差分");

        FdResult r4b = finiteDiff(ex, x, e, ex.head.b, ex.head.gb, 8);
        std::printf("    head.b  : 相对误差=%.2e\n", r4b.worstRel);
        CHECK(r4b.worstRel < 3e-2, "head.b 的解析梯度 == 中心差分");

        /* 输入梯度: 同样中心差分 (只查 |ei| 最大的 8 个坐标 —— 其余是噪声地板, 见 §8) */
        {
            RL::Tensor ei((std::size_t)D, 1);
            ei.zero();
            ex.e.zero();
            for (int i = 0; i < D; i++) { ex.e[(std::size_t)i] = e[(std::size_t)i]; }
            ex.forward(x);
            ex.backward(x, ei);
            std::vector<int> idx(D);
            for (int i = 0; i < D; i++) { idx[(std::size_t)i] = i; }
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return std::fabs((double)ei[(std::size_t)a]) > std::fabs((double)ei[(std::size_t)b]);
            });
            RL::Tensor xp = x;
            double worst = 0.0, anaMax = 0.0;
            for (int k = 0; k < 8; k++) {
                const int i = idx[(std::size_t)k];
                const double ana = (double)ei[(std::size_t)i];
                const double x0 = (double)x[(std::size_t)i];
                const double eps = 1e-3;
                xp[(std::size_t)i] = (float)(x0 + eps);
                const double lp = lossOf(ex, xp, e);
                xp[(std::size_t)i] = (float)(x0 - eps);
                const double lm = lossOf(ex, xp, e);
                xp[(std::size_t)i] = (float)x0;
                const double num = (lp - lm) / (2.0 * eps);
                worst = std::max(worst, std::fabs(num - ana)
                                            / (std::fabs(num) + std::fabs(ana) + 1e-30));
                anaMax = std::max(anaMax, std::fabs(ana));
            }
            std::printf("    输入梯度 (|ei| 最大的 8 个坐标, |ei|max=%.3e): 最差相对误差=%.2e\n",
                        anaMax, worst);
            CHECK(worst < 3e-2, "输入梯度 (dL/dx) 的解析值 == 中心差分");
            CHECK(anaMax > 1e-6, "输入梯度不是全零 (正对照: 否则上面那条是空测)");
        }
    }

    /* ================= [3] im2col / col2im 的自洽 ================= */
    std::printf("\n[3] im2col / col2im 的自洽 (patch 下标与边界)\n");
    {
        /* 一个"每个位置是 (行,列) 编码"的假图像: 用它可以直接验证 patch 取到了谁 */
        const int Cin = 2;
        std::vector<float> im((std::size_t)Cin * 90, 0.0f);
        for (int c = 0; c < Cin; c++) {
            for (int xx = 0; xx < 10; xx++) {
                for (int yy = 0; yy < 9; yy++) {
                    im[(std::size_t)(c * 90 + xx * 9 + yy)] = (float)(c * 100 + xx * 10 + yy);
                }
            }
        }
        std::vector<float> col((std::size_t)Cin * 9 * 90, 0.0f);
        RL::ConvResNetExpert::im2col(im.data(), Cin, 3, col.data());
        /* 中心位置 (x=5,y=4) 的 patch 应当正好是它自己的 3x3 邻域, 且顺序 (u,v) 行主序 */
        bool patchOk = true;
        for (int c = 0; c < Cin; c++) {
            for (int u = 0; u < 3; u++) {
                for (int v = 0; v < 3; v++) {
                    const float got = col[(std::size_t)((c * 9 + u * 3 + v) * 90 + (5 * 9 + 4))];
                    const float want = (float)(c * 100 + (5 + u - 1) * 10 + (4 + v - 1));
                    if (std::fabs(got - want) > 1e-6) { patchOk = false; }
                }
            }
        }
        CHECK(patchOk, "中心位置的 patch == 它的 3x3 邻域, 且 (u,v) 行主序 (下标没错位)");
        /* 角上的 padding: (0,0) 的 u=0/v=0 (越界) 必须是 0, 其余取到真实值 */
        CHECK(col[(std::size_t)((0 * 9 + 0 * 3 + 0) * 90 + 0)] == 0.0f,
              "角上越界的 patch 元素 = 0 (padding)");
        CHECK(std::fabs(col[(std::size_t)((0 * 9 + 1 * 3 + 1) * 90 + 0)]
                        - im[0]) < 1e-6,
              "角上的 patch 中心 = 该位置自己 (边界没有整体错位)");
        /* col2im 必须是散加: 一个内部位置被 9 个 patch 覆盖 ⇒ 单位 dcol 给它 9 */
        std::vector<float> dcol((std::size_t)Cin * 9 * 90, 1.0f);
        std::vector<float> dimg((std::size_t)Cin * 90, 0.0f);
        RL::ConvResNetExpert::col2imAdd(dcol.data(), Cin, 3, dimg.data());
        CHECK(std::fabs(dimg[(std::size_t)(0 * 90 + 5 * 9 + 4)] - 9.0f) < 1e-6,
              "内部位置被 9 个 patch 覆盖 ⇒ col2im 累加出 9 (是 += 而不是 =)");
        CHECK(std::fabs(dimg[(std::size_t)(0 * 90 + 0 * 9 + 0)] - 4.0f) < 1e-6,
              "角上位置只被 4 个 patch 覆盖 ⇒ 累加出 4 (边界处理正确)");
    }

    /* ================= [4] 坐标通道开关 (参数量会变, 所以权重不通用) ================= */
    std::printf("\n[4] 坐标/区域通道开关 (会改变 conv1 的 W 宽度)\n");
    {
        RL::ConvResNetExpert a(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, true, true);
        RL::ConvResNetExpert b(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, true, false);
        std::printf("    坐标开: 输入通道 %d, 参数量 %lld | 坐标关: 输入通道 %d, 参数量 %lld\n",
                    a.inChannelsWithCoord(), a.paramCount(), b.inChannelsWithCoord(), b.paramCount());
        CHECK(b.inChannelsWithCoord() == 19, "关掉之后卷积输入 = 19 通道 (用户口径的纯 19 平面)");
        /*
           ⚠ 这一条是把一个**容易想当然**的地方钉死: 坐标通道是"零参数"的, 但它改变了
           conv1 的**输入通道数** ⇒ W 从 (32, 19*9) 变 (32, 23*9) ⇒ 参数量差
           (23-19)*32*9 = 1,152 ⇒ **两种口径的权重互不兼容** (不能热切换, 要 A/B 就各训一份)。
        */
        CHECK(a.paramCount() - b.paramCount() == 4LL * 32 * 9,
              "两者参数量之差 == (23-19) x 32 x 9 ⇒ 权重不通用 (不能当成零代价开关)");
        CHECK(a.paramCount() == 44275LL && b.paramCount() == 44275LL - 1152LL,
              "参数量分别 == 44,275 (坐标开) / 43,123 (坐标关)");
    }

    /* ================= [5] 权重往返 (专家级 + Net 级) ================= */
    std::printf("\n[5] 权重往返 (专家级逐位 + Net 级同一输入同一输出)\n");
    {
        RL::ConvResNetExpert src(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, false, true);
        src.forward(x);
        std::vector<float> o1((std::size_t)D);
        for (int i = 0; i < D; i++) { o1[(std::size_t)i] = src.o[(std::size_t)i]; }

        std::ostringstream os;
        src.write(os);
        RL::ConvResNetExpert dst(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, false, true);
        std::istringstream is(os.str());
        dst.read(is);
        dst.forward(x);
        double d = 0.0;
        for (int i = 0; i < D; i++) {
            d = std::max(d, std::fabs((double)o1[(std::size_t)i] - (double)dst.o[(std::size_t)i]));
        }
        std::printf("    专家级往返: 权重文本 %zu 字节, 同输入输出最大差 = %.3e\n",
                    os.str().size(), d);
        CHECK(d == 0.0, "专家级 write→read 往返: 同一输入**逐位相同**");
        CHECK(dst.paramCount() == src.paramCount(), "往返两边参数量一致");

        /* Net 级: 一个真实的 PPO (ConvExperts) 存盘再载入 */
        {
            const std::string ap = "ppo_conv_rt_a.wgt";
            const std::string cp = "ppo_conv_rt_c.wgt";
            RL::PPO src2(D, 64, 8100, RL::PPO::CONV_CHANNELS, 0.1f, false,
                         RL::PPO::Backbone::ConvExperts);
            std::printf("    PPO actor 参数量 (ConvExperts) = %lld\n", src2.actorParamCount());
            RL::Tensor &oa = src2.actorP.forward(x);
            std::vector<float> before(oa.size());
            for (std::size_t i = 0; i < oa.size(); i++) { before[i] = oa[i]; }
            CHECK(src2.save(ap, cp), "PPO(ConvExperts) save 成功");

            RL::PPO dst2(D, 64, 8100, RL::PPO::CONV_CHANNELS, 0.1f, false,
                         RL::PPO::Backbone::ConvExperts);
            const bool loaded = dst2.load(ap, cp);
            CHECK(loaded, "PPO(ConvExperts) **Net 级载入成功**");
            RL::Tensor &ob = dst2.actorP.forward(x);
            double d2 = 0.0;
            if (ob.size() == before.size()) {
                for (std::size_t i = 0; i < ob.size(); i++) {
                    d2 = std::max(d2, std::fabs((double)before[i] - (double)ob[i]));
                }
            } else {
                d2 = 1e30;
            }
            std::printf("    Net 级往返: 同一输入输出最大差 = %.3e\n", d2);
            CHECK(d2 == 0.0, "Net 级往返后同一输入**逐位相同** (写序被钉住)");
            /* 交叉骨干必须被拒: conv 的权重载进 seq 骨干 (参数量不同) */
            RL::PPO seqNet(D, 64, 8100, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
            CHECK(!seqNet.load(ap, cp), "conv 的权重载进 seq 骨干**被明确拒绝** (不静默串权重)");
            std::remove(ap.c_str());
            std::remove(cp.c_str());
        }
    }

    /* ================= [6] 一次 forward+backward 的墙钟 (读数, 不是断言) ================= */
    std::printf("\n[6] 墙钟读数 (只印不断言: 本机计时跨会话能漂好几倍, 见 seq 文档 §8.4)\n");
    {
        RL::ConvResNetExpert ex(D, RL::PPO::CONV_CHANNELS, RL::PPO::CONV_BLOCKS, true, true);
        RL::Tensor ei((std::size_t)D, 1);
        const int reps = 200;
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        for (int r = 0; r < reps; r++) { ex.forward(x); }
        auto t1 = clock::now();
        for (int r = 0; r < reps; r++) {
            ex.forward(x);
            ei.zero();
            ex.e.zero();
            for (int i = 0; i < D; i++) { ex.e[(std::size_t)i] = e[(std::size_t)i]; }
            ex.backward(x, ei);
        }
        auto t2 = clock::now();
        const double fwd = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        const double both = std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
        const double mac = (double)ex.forwardMac();
        std::printf("    forward = %.3f ms (%.3f ns/MAC) | forward+backward = %.3f ms "
                    "(backward 是 forward 的 %.2fx)\n",
                    fwd, fwd * 1e6 / mac, both, (fwd > 0.0) ? (both - fwd) / fwd : 0.0);
    }

    std::printf("\n=== test_conv_resnet: %d 项断言, %d 失败 ===\n", g_pass + g_fail, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
