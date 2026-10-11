/*
 * test_ppo_backbone_main.cpp — PPO 骨干结构 + 优化器裁剪口径的断言测试 (2026-10)
 * ============================================================================
 * 为什么单独一个测试目标 (而不是塞进 test_ppomcts):
 *   它**不建 agent、不跑 MCTS、不碰棋盘**, 只建 `RL::PPO` 本身, 所以是秒级 + 只依赖
 *   rl 层 (ppo.cpp / bc.cpp / util.cpp)。而 `test_ppomcts` 在 TB 骨干下要 ~20 分钟,
 *   "改一行头数常量"这种事不该排在那个队尾 (本工程记过很多次"慢的测试等于没有测试")。
 *
 * 盯的是四件**会静默错**的事 (每一条都能在改动前通过):
 *   [1] 骨干枚举的解析/短名 (工具与文件名都靠它; 解析失败必须报错而不是静默退回默认)
 *   [2] TB 专家的**头数口径**: 请求 N 头 == 实用 N 头 (改动前是"请求 16 / 实用 15",
 *       而那一刻**任何读数都不变** —— 只有慢一点、容量少一点)
 *   [3] 参数构成 (注意力 / FFN / LN 三分法) 与参数量
 *   [4] 优化器 `clipGrad` 的**真实语义** —— 它不是裁剪, 而是"逐张量单位范数归一":
 *       ① 对正常尺度的梯度, RMSProp 的逐坐标归一已经把整张量的等比缩放抵消掉,
 *          所以"加一层再归一"在数值上几乎无差别 (这条必须钉住, 否则会有人以为
 *          "换成 global-norm 就能修训练" —— 它不是病根);
 *       ② 但它有一个**硬门限**: 梯度范数 << 1e-8 的层会被 `+1e-8` 抹成"不动",
 *          而 `GRAD_CLIP_NONE` 下同一份梯度会得到完整的一步。这两条合起来才是这个
 *          开关的真正作用域。
 *
 * 用法: test_ppo_backbone  (无参数; 失败以非 0 退出码报告, 与其它 test_* 同口径)
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "rl/ppo.h"
#include "rl/expert.hpp"
#include "rl/layer.h"
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

/* d_model = PPOMCTSAgent::STATE_DIM = 19 平面 x 90 格 = 1710。
   本测试**故意不 include agent 头**: 它只依赖 rl 层。若哪天真把 STATE_DIM 改了,
   这里的 1710 会当场失配 —— 那正是我们想要的"响亮失败"。 */
const int D = 1710;
const int H = 64;
const int A = 8100;

/* 一个干净的 actor: state -> MoE -> Tanh(H) -> Softmax(A) (与 PPO 内部同一套构造) */
double weightL2(const RL::iFcLayer &fc)
{
    double s = 0;
    for (std::size_t i = 0; i < fc.w.size(); i++) { const double v = fc.w[i]; s += v * v; }
    return std::sqrt(s);
}

}  // namespace

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf(" test_ppo_backbone — PPO 骨干结构 + 裁剪口径\n");

    /* ================= [1] 骨干枚举 ================= */
    std::printf("\n[1] 骨干枚举 (parseBackbone / backboneKey)\n");
    {
        RL::PPO::Backbone b = RL::PPO::Backbone::MlpExperts;
        CHECK(RL::PPO::parseBackbone("tb", b) && b == RL::PPO::Backbone::TbExperts, "解析 tb");
        CHECK(RL::PPO::parseBackbone("MLP", b) && b == RL::PPO::Backbone::MlpExperts, "解析 MLP (大小写不敏感)");
        CHECK(RL::PPO::parseBackbone("layer", b) && b == RL::PPO::Backbone::LayerExperts, "解析 layer");
        /* 负对照: 失败必须**不改** out, 而且返回 false (调用方据此报错退出) */
        b = RL::PPO::Backbone::TbExperts;
        CHECK(!RL::PPO::parseBackbone("nonsense", b) && b == RL::PPO::Backbone::TbExperts,
              "非法输入返回 false 且不改 out");
        CHECK(std::strcmp(RL::PPO::backboneKey(RL::PPO::Backbone::TbExperts), "tb") == 0, "backboneKey tb");
        CHECK(std::strcmp(RL::PPO::backboneKey(RL::PPO::Backbone::LayerExperts), "layer") == 0, "backboneKey layer");
    }

    /* ================= [2] TB 头数口径 ================= */
    std::printf("\n[2] TB 专家的头数口径 (d_model=%d)\n", D);
    {
        RL::TransformerBlock<RL::PPO_MOE_TB_HEADS, RL::PPO_MOE_TB_DFF, true> ex(D, false);
        const int req = ex.attnHeadsRequested();
        const int use = ex.attnHeadsUsed();
        const int dk = ex.attnHeadDim();
        const long long elems = ex.attnElements();
        std::printf("    请求 %d / 实用 %d / d_k %d / 注意力元素 %lld / 未覆盖 %d\n",
                    req, use, dk, elems, D - use * dk);
        CHECK(req == use, "请求的头数 == 实用的头数 (不再静默降级)");
        CHECK(req == RL::PPO_MOE_TB_HEADS, "请求数 = PPO_MOE_TB_HEADS");
        CHECK(D % req == 0, "d_model 被请求的头数整除 (覆盖满)");
        CHECK(use * dk == D, "numHeads * d_k == d_model (没有未覆盖坐标)");
        CHECK(elems == (long long)use * dk * dk, "attnElements = numHeads * d_k^2");
        CHECK(ex.attnHeadsAllocated() == use, "分配的 head 数 == 实用的 head 数");
    }

    /* ================= [3] 参数构成 + actor 参数量 ================= */
    std::printf("\n[3] 参数构成与参数量\n");
    {
        RL::PPO ppo(D, H, A, 64, 0.1f, false /* withGrad=false: 本节的断言不需要梯度 */,
                    RL::PPO::Backbone::TbExperts);
        long long pa = 0, pf = 0, pl = 0;
        ppo.tbExpertParamBreakdown(pa, pf, pl);
        const long long attnExpect = 4LL * D * D;
        std::printf("    一个 TB 专家: 注意力 %lld / FFN %lld / LN %lld\n", pa, pf, pl);
        CHECK(pa == attnExpect, "注意力参数 == 4*d_model^2 (3 份 qkv + 1 份 wo)");
        CHECK(pa + pf + pl > 0, "三分法不是空的");
        /* 三分法必须等于专家自己的 paramCount —— 否则以后加层就会漏报 */
        RL::TransformerBlock<RL::PPO_MOE_TB_HEADS, RL::PPO_MOE_TB_DFF, true> ex(D, false);
        CHECK(pa + pf + pl == ex.paramCount(),
              "三分法之和 == 专家 paramCount (加层不会漏报)");
        CHECK(pa * 10 > ex.paramCount() * 9, "注意力占 TB 专家参数的 90% 以上 (实测 90.4%)");
        /* actor 参数量: 这是"头数口径修正**没有**改结构"的证据 (同一份存量权重仍可载入) */
        std::printf("    actor 参数量 (TbExperts, E=4/top-1) = %lld\n", ppo.actorParamCount());
        CHECK(ppo.actorParamCount() == 52388888LL,
              "TB actor 参数量仍是 52,388,888 (改口径不改结构 ⇒ 存量权重可载)");
        CHECK(ppo.tbHeadsRequested() == RL::PPO_MOE_TB_HEADS, "PPO 转发: 请求头数");
        CHECK(ppo.tbHeadsUsed() == RL::PPO_MOE_TB_HEADS, "PPO 转发: 实用头数");
        CHECK(ppo.tbHeadDim() == D / RL::PPO_MOE_TB_HEADS, "PPO 转发: d_k");
        CHECK(ppo.tbAttnElements() == (long long)RL::PPO_MOE_TB_HEADS * (D / RL::PPO_MOE_TB_HEADS)
                                          * (D / RL::PPO_MOE_TB_HEADS),
              "PPO 转发: 注意力元素");
    }
    /* 廉价骨干 (无 MHA): 注意力读数为 -1 / 0, 三分法全落在非注意力那一列 */
    {
        RL::PPO ppo(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::LayerExperts);
        const long long expert = (long long)D * D + D;   /* Layer<Gelu>(d->d): d^2 + d */
        const long long gate = (long long)RL::PPO_MOE_LAYER_EXPERTS * D + RL::PPO_MOE_LAYER_EXPERTS;
        const long long heads = (long long)H * D + H + (long long)A * H + A;
        std::printf("    actor 参数量 (LayerExperts) = %lld (专家 %lld + 门控 %lld + 头 %lld)\n",
                    ppo.actorParamCount(), RL::PPO_MOE_LAYER_EXPERTS * expert, gate, heads);
        CHECK(ppo.actorParamCount()
                  == (long long)RL::PPO_MOE_LAYER_EXPERTS * expert + gate + heads,
              "廉价骨干参数量 == E*(d^2+d) + 门控 + 两个头");
        CHECK(ppo.tbHeadsRequested() == -1, "廉价骨干没有注意力 ⇒ 请求头数 = -1");
        CHECK(ppo.tbAttnElements() == 0, "廉价骨干 ⇒ 注意力元素 = 0");
        CHECK(ppo.actorParamCount() * 4 < 52388888LL,
              "廉价骨干至少比 TB 小 4 倍 (实测 12.35 M vs 52.39 M = 4.24x)");
    }

    /* 序列 Transformer 骨干 (无外积伪注意力): 结构读数与参数量对账 */
    {
        RL::PPO ppo(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
        /*
           actor 参数量 = E × 每专家 + 门控 + 两个头。
           每专家 91,731 的构成 (与 test_transformer [3] 的读数一致):
             embed 64×19+64 = 1,280 | 位置嵌入 90×64 = 5,760
             每层: 4×(64×64+64) = 16,640 + ffnUp 192×64+192 = 12,480
                   + ffnDown 64×192+64 = 12,352 + LN 4×64 = 256  -> 41,728
             2 层 = 83,456 | 输出投影 19×64+19 = 1,235
        */
        const long long expert = 91731;
        const long long gate = (long long)RL::PPO_MOE_SEQ_EXPERTS * D + RL::PPO_MOE_SEQ_EXPERTS;
        const long long heads2 = (long long)H * D + H + (long long)A * H + A;
        std::printf("    actor 参数量 (SeqExperts) = %lld (专家 %lld + 门控 %lld + 头 %lld)\n",
                    ppo.actorParamCount(), RL::PPO_MOE_SEQ_EXPERTS * expert, gate, heads2);
        CHECK(ppo.actorParamCount() == (long long)RL::PPO_MOE_SEQ_EXPERTS * expert + gate + heads2,
              "序列 Transformer 的 actor 参数量 == E*91,731 + 门控 + 两个头 (1,009,772)");
        CHECK(ppo.actorParamCount() * 50 < 52388888LL,
              "序列 Transformer 的 actor 至少比 TB 小 50 倍 (实测 51.9x)");
        CHECK(ppo.tbHeadsRequested() == RL::PPO::SEQ_HEADS
                  && ppo.tbHeadsUsed() == RL::PPO::SEQ_HEADS,
              "序列 Transformer 的请求头数 == 实用头数 == SEQ_HEADS");
        CHECK(ppo.tbHeadDim() == H / RL::PPO::SEQ_HEADS, "d_k == expertHidden / SEQ_HEADS");
        CHECK(ppo.tbAttnElements() == (long long)RL::PPO::SEQ_HEADS
                                          * RL::PPO::SEQ_LEN * RL::PPO::SEQ_LEN,
              "注意力矩阵元素 == Heads * T^2 (90x90 各 4 个头)");
        long long pa = 0, pf = 0, pl = 0;
        ppo.tbExpertParamBreakdown(pa, pf, pl);
        CHECK(pa == 33280 && pf == 57939 && pl == 512,
              "专家的参数构成: 注意力 33,280 / FFN+嵌入 57,939 / LN 512");
        CHECK(pa + pf + pl == expert, "参数构成之和 == 每专家参数量");
    }

    /* ================= [4] 裁剪口径的真实语义 ================= */
    std::printf("\n[4] Optimize::RMSProp 的 clipGrad 到底做了什么\n");
    {
        /* 一个"可以手算"的小网: Layer<Tanh>(8->4) + Layer<Linear>(4->1) */
        RL::Net::Layers layers;
        layers.push_back(RL::Layer<RL::Tanh>::_(8, 4, true, true));
        layers.push_back(RL::Layer<RL::Linear>::_(4, 1, true, true));
        RL::Net net(layers);

        /* 手工造一份梯度 (尺度由 scale 决定; decay 传 0, 免得权重衰减被误读成梯度位移) */
        auto fillGrad = [&](float scale) {
            for (std::size_t i = 0; i < net.size(); i++) {
                RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
                fc->g.w.zero();
                fc->g.b.zero();
                fc->g.w[0] = 1.0f * scale;
                fc->g.w[1] = -0.5f * scale;
                fc->g.w[2] = 0.25f * scale;
                fc->g.b[0] = 0.5f * scale;
            }
        };
        /* 每次都从同一份权重/同一份 v 出发, 所以 |Δw| 完全由"梯度 + 模式"决定 */
        auto stepSize = [&](int mode, float maxNorm, float scale) {
            for (std::size_t i = 0; i < net.size(); i++) {
                RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
                for (std::size_t k = 0; k < fc->w.size(); k++) { fc->w[k] = 0.024f; }
                fc->v.w.zero();
                fc->v.b.zero();
            }
            fillGrad(scale);
            net.RMSPropMode(0.002f, 0.9f, 0.0f, mode, maxNorm);
            double s = 0;
            for (std::size_t i = 0; i < net.size(); i++) {
                RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
                for (std::size_t k = 0; k < fc->w.size(); k++) {
                    const double d = (double)fc->w[k] - 0.024;
                    s += d * d;
                }
            }
            return std::sqrt(s);
        };

        const double sLegacy = stepSize(RL::GRAD_CLIP_PER_TENSOR_UNIT_NORM, 1.0f, 1.0f);
        const double sNone   = stepSize(RL::GRAD_CLIP_NONE, 1.0f, 1.0f);
        std::printf("    正常尺度梯度 (|g|~1):    legacy |Δw| = %.6g, none |Δw| = %.6g\n",
                    sLegacy, sNone);
        /* ① 两者在正常尺度上**完全相同** —— RMSProp 逐坐标归一已经抵消了整张量的
              等比缩放。所以"换个裁剪口径就能修训练"这个判断是错的 (这是本节的重点)。 */
        CHECK(std::fabs(sLegacy - sNone) <= 1e-6 * (sLegacy + 1e-30),
              "正常尺度下 legacy 与 none 的步长相同 (逐张量归一被 RMSProp 抵消)");
        const double sScale1e6 = stepSize(RL::GRAD_CLIP_NONE, 1.0f, 1e6f);
        std::printf("    同一份梯度乘 1e6 (none): |Δw| = %.6g\n", sScale1e6);
        CHECK(std::fabs(sScale1e6 - sNone) <= 1e-6 * (sNone + 1e-30),
              "梯度乘 1e6 之后步长不变 (RMSProp 的尺度不变性)");

        /* ② 唯一真正有差别的区间: |g| 在 1e-16 ~ 3e-9 之间 ——
              NONE 的 sqrt(v) 被 1e-9 的 epsilon 顶住 ⇒ 更新≈0 (更早"冻结"),
              而 legacy 先把 dw 放大 1e8 倍 ⇒ 仍在 RMSProp 的工作区间里, 照常走一步。
              注意方向: **是 legacy 让极小梯度活得久, 不是它把梯度抹掉。** */
        const double sLegacyTiny = stepSize(RL::GRAD_CLIP_PER_TENSOR_UNIT_NORM, 1.0f, 1e-14f);
        const double sNoneTiny   = stepSize(RL::GRAD_CLIP_NONE, 1.0f, 1e-14f);
        std::printf("    极小梯度 (|g|~1e-14):    legacy |Δw| = %.6g, none |Δw| = %.6g\n",
                    sLegacyTiny, sNoneTiny);
        CHECK(sLegacyTiny > 1e-3, "legacy 对 1e-14 量级的梯度仍给出完整一步");
        CHECK(sNoneTiny < 1e-3, "none 对同一份梯度几乎不动 (被 1e-9 的 epsilon 顶住)");
        CHECK(sLegacyTiny > 1000.0 * sNoneTiny, "两种模式在极小梯度上相差 3 个数量级以上");

        /* ③ 再小到 1e-30: 两者都不动 (数值上真的没了) */
        const double sLegacyZero = stepSize(RL::GRAD_CLIP_PER_TENSOR_UNIT_NORM, 1.0f, 1e-30f);
        std::printf("    更小 (|g|~1e-30):        legacy |Δw| = %.6g\n", sLegacyZero);
        CHECK(sLegacyZero < 1e-3, "1e-30 的梯度在 legacy 下也不动 (没有神奇的放大)");
    }

    /* ================= [5] 梯度范数读数 + 廉价骨干能训 ================= */
    std::printf("\n[5] 梯度范数读数 (BC 一批之后) + 廉价骨干的一步训练\n");
    {
        RL::PPO ppo(D, H, A, 64, 0.1f, true, RL::PPO::Backbone::LayerExperts);
        ppo.maskedTrainHead = true;
        RL::Tensor state((std::size_t)D, 1);
        RL::Random::setSeed(20240901u);
        for (int i = 0; i < D; i++) { state[(std::size_t)i] = ((i % 17) == 0) ? 1.0f : 0.0f; }
        std::vector<int> legal, tgt;
        std::vector<float> tp;
        for (int k = 0; k < 32; k++) { legal.push_back(k * 251 % A); }
        tgt.push_back(legal[3]);
        tgt.push_back(legal[7]);
        tp.push_back(0.7f);
        tp.push_back(0.3f);

        RL::iFcLayer *head = dynamic_cast<RL::iFcLayer *>(ppo.actorP[2]);
        const double w0 = weightL2(*head);
        ppo.resetMoeBatchStats();
        const RL::PPO::BcOutcome oc = ppo.bcGradSparse(state, legal, tgt, tp);
        CHECK(oc == RL::PPO::BcOutcome::Sparse, "BC 走合法列口径");
        const double gnorm = ppo.actorP.gradNorm();
        std::printf("    |g|(actor) = %.6g (逐层: %s)\n", gnorm,
                    ppo.actorP.gradNormReport().c_str());
        CHECK(gnorm > 0.0, "梯度范数读数 > 0 (以前没有这个数)");
        ppo.finalizeMoeBatch();          /* BC 路径新接的批边界 (见 bcagent.hpp) */
        ppo.bcApplyGradients(0.002f);
        const double w1 = weightL2(*head);
        CHECK(std::fabs(w1 - w0) > 1e-9, "一次 BC 更新真的改了权重");
        /*
           梯度范数**默认不采集** (trackGradNorm=false, 见 rl/ppo.h): 算一次要把全部参数
           的梯度读一遍, 在 TB 骨干上优化器本来就占一步的 ~91%。这里把"默认不采集"也钉住 ——
           免得以后有人把默认改回去而让训练主路径白多一遍读。
        */
        CHECK(ppo.actorGradNorm == 0.0, "默认不采集梯度范数 (trackGradNorm=false)");

        /* 打开采集之后, 读数必须真的出现 */
        ppo.trackGradNorm = true;
        ppo.resetMoeBatchStats();
        ppo.bcGradSparse(state, legal, tgt, tp);
        ppo.finalizeMoeBatch();
        ppo.bcApplyGradients(0.002f);
        CHECK(ppo.actorGradNorm > 0.0, "打开 trackGradNorm 后 PPO 记下了 actor 的梯度范数");
        std::printf("    actor 梯度范数读数 = %.6g | 头权重 |w| %.6f -> %.6f\n",
                    ppo.actorGradNorm, w0, w1);
    }

    /*
       ================= [6] 序列专家位置编码的 **Net 级**权重往返 =================
       [2026-10] 这一段是一个**实测出来的缺陷**的回归:

         现象: `train_bc --agent=ppo-seq --pos=2d --save=weights/x` 保存成功, 但
               `bench_agent_arena --backbone-ppo-a=seq --pos-ppo-a=2d --load-a=weights/x`
               载入失败 —— `weights/x_actor: 第 5 个张量校验失败 (文件被截断/损坏?)`。
         根因: `SeqPosEnc::write` 在非默认模式下先写一行**标记** (`POS2D` / `POSROPE`),
               而 `Net::load` 的预校验**要求每一行都是能解码的张量** ⇒ 标记行被当成损坏。
               `test_transformer [6]` 全绿也发现不了它: 那条路径走的是**专家级**
               `write/read`, 绕过了 `Net::load`。
         修法: `codec::isMetaMarkerLine` 白名单 (精确匹配那两个标记), 预校验跳过它们
               且**不计入元素总数**; 其它任何解码不了的行走原来的失败路径。

       所以这里钉三件事 (缺一条, 上面那个缺陷都能悄悄回来):
         (a) 2d / rope 的 PPO 权重能**存下去、载回来**, 且载回后同一输入同一输出;
         (b) 白名单**没有放宽损坏检测**: 把标记改成不在名单里的字符串 (或截断一行),
             载入必须仍然失败;
         (c) 模式不匹配 (2d 的权重载进 1d 的网) 必须**响亮失败**, 不能静默错位。
    */
    std::printf("\n[6] 序列专家位置编码的 Net 级权重往返 (2d / rope / 1d)\n");
    {
        RL::Tensor state((std::size_t)D, 1);
        for (int i = 0; i < D; i++) {
            state[(std::size_t)i] = ((i % 13) == 0) ? 0.75f : (((i % 7) == 0) ? -0.5f : 0.0f);
        }
        /* 同一输入在两个网上的输出 (逐位比较; 权重相同 ⇒ 输出必须完全相同) */
        auto actorOut = [&](RL::PPO &p) {
            RL::Tensor &o = p.actorP.forward(state);
            std::vector<float> v(o.size());
            for (std::size_t k = 0; k < o.size(); k++) { v[k] = o[k]; }
            return v;
        };
        auto maxAbsDiff = [](const std::vector<float> &a, const std::vector<float> &b) {
            if (a.size() != b.size()) { return 1e30; }
            double m = 0.0;
            for (std::size_t k = 0; k < a.size(); k++) {
                m = std::max(m, std::fabs((double)a[k] - (double)b[k]));
            }
            return m;
        };

        struct Case { const char *key; const char *tag; long long actorParams; };
        /*
           actor 参数量的三个已知值就是"模式真的生效了"的**结构指纹** (没有回读接口时
           它比"设置函数返回 true"强得多): 每专家的位置参数 1d 90×64=5,760 /
           2d (10+9)×64=1,216 / rope 0, 乘 4 个专家 ⇒ 1,009,772 / 991,596 / 986,732。
        */
        const Case cases[] = { { "1d", "1d", 1009772LL },
                               { "2d", "2d", 991596LL },
                               { "rope", "rope", 986732LL } };
        for (int c = 0; c < 3; c++) {
            const std::string base = std::string("ppo_pos_") + cases[c].tag;
            const std::string ap = base + "_a.wgt";
            const std::string cp = base + "_c.wgt";

            RL::PPO src(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
            CHECK(src.setExpertPosMode(cases[c].key), "setExpertPosMode 受理该模式");
            const long long pc = src.actorParamCount();
            CHECK(pc == cases[c].actorParams, "参数量 == 该模式的结构指纹 (模式真的生效)");
            const std::vector<float> before = actorOut(src);
            std::printf("    [%s] save -> %s\n", cases[c].tag, ap.c_str());
            std::fflush(stdout);
            CHECK(src.save(ap, cp), "save 成功");

            RL::PPO dst(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
            CHECK(dst.setExpertPosMode(cases[c].key), "目标网切到同一模式");
            std::printf("    [%s] load (往返)\n", cases[c].tag);
            std::fflush(stdout);
            const bool loaded = dst.load(ap, cp);
            CHECK(loaded, "(a) **Net 级载入成功** (标记行不再被误判成损坏)");
            CHECK(dst.actorParamCount() == pc, "载入后参数量一致");
            const std::vector<float> after = actorOut(dst);
            const double d = maxAbsDiff(before, after);
            std::printf("    %-4s 参数量 %lld | 载回后同输入输出最大差 = %.3e\n",
                        cases[c].tag, pc, d);
            CHECK(d == 0.0, "(a) 往返后同一输入**逐位相同**");

            /*
               (b) 白名单必须**窄**: 把标记换成一个不在名单里的行 (结构与它一样是纯字母),
               预校验必须仍然拒绝 —— 否则"首字符是字母就跳过"式的实现会放过损坏文件。
            */
            if (c > 0) {
                std::string blob;
                {
                    std::ifstream in(ap.c_str(), std::ios::binary);
                    blob.assign((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
                }
                const char *bad = (c == 1) ? "POSXYZ" : "POSROT";
                const std::size_t at = blob.find(cases[c].tag == std::string("2d") ? "POS2D"
                                                                                    : "POSROPE");
                if (at != std::string::npos) {
                    blob.replace(at, std::strlen(bad), bad);
                }
                const std::string bp = base + "_bad.wgt";
                {
                    std::ofstream out(bp.c_str(), std::ios::binary);
                    out.write(blob.data(), (std::streamsize)blob.size());
                }
                RL::PPO badNet(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
                badNet.setExpertPosMode(cases[c].key);
                std::printf("    [%s] load (bad marker '%s' -> 期望被拒)\n",
                            cases[c].tag, bad);
                std::fflush(stdout);
                CHECK(badNet.load(bp, cp) == false,
                      "(b) 不在白名单里的标记行仍然被拒 (损坏检测没被放宽)");
                std::remove(bp.c_str());
            }
        }

        /* (c) 模式不匹配: 2d 的权重载进 1d 的网必须失败 (不是静默错位) */
        {
            const std::string ap = "ppo_pos_2d_a.wgt";
            const std::string cp = "ppo_pos_2d_c.wgt";
            RL::PPO one(D, H, A, 64, 0.1f, false, RL::PPO::Backbone::SeqExperts);
            CHECK(one.actorParamCount() == 1009772LL, "新网的默认模式是 1d (参数量指纹)");
            std::printf("    [c] load (2d 权重 -> 1d 网, 期望被拒)\n");
            std::fflush(stdout);
            CHECK(one.load(ap, cp) == false, "(c) 2d 的权重载进 1d 的网必须响亮失败");
            std::printf("    (c) 模式不匹配: 载入被拒绝 (参数量也不同: 1d 1009772 vs 2d 991596)\n");
        }
        std::remove("ppo_pos_1d_a.wgt");
        std::remove("ppo_pos_1d_c.wgt");
        std::remove("ppo_pos_2d_a.wgt");
        std::remove("ppo_pos_2d_c.wgt");
        std::remove("ppo_pos_rope_a.wgt");
        std::remove("ppo_pos_rope_c.wgt");
    }

    std::printf("\n=== test_ppo_backbone: %d 项断言, %d 失败 ===\n", g_pass + g_fail, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
