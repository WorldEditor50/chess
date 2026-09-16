/*
 * test_sac_main.cpp - RL::SAC (库里的离散 SAC) 的验证
 *
 * 为什么单独一个测试: `RL::SAC` 在这之前**没有任何调用方**, 于是它从来没有被真正
 * 跑过 —— 这一轮把 PPO 的训练侧优化方法 (P3 梯度累积 / P4 回放池多 epoch /
 * MoE 批统计与辅助损失 / 批平均损失 / R2 只算合法列) 搬过来时, 顺手撞出两处
 * **静默**缺陷 (见 rl/sac.h 顶部):
 *
 *   1. critic 的输入被拼成 [state; π] (stateDim+actionDim 维), 但第一层是按
 *      stateDim 建的 —— `MM::ikkj` 的契约要求 x1.shape[1] == x2.shape[0],
 *      Release 下断言被 NDEBUG 关掉, 于是 π 那一段**被静默丢弃**;
 *      Debug 构建里会直接断言失败。
 *   2. `learn()` 收下 learningRate 却把三个优化器学习率写死。
 *
 * 所以这个测试盯的是三件事:
 *   A. **R2 的口径**: 合法集上 Σπ ≡ 1、非法列 π 恰好为 0、非法列的头部权重梯度
 *      **恰好为 0** (不是"很小")。这三条都是"看起来对但可能错"的地方。
 *   B. **critic 真的在学**: 一批固定经验 (reward=+0.5, done=true → TD 目标恰好 0.5),
 *      跑若干次 learnFromReplay 后 Q(s,a) 必须朝 0.5 走 —— 这一条能挡住
 *      "前向对但反向错 / 优化器用错 / 符号搞反"。
 *   C. **P3/P4 的机制**: 多 epoch 复用同一批数据时, 累积的梯度必须**成比例**变大
 *      (用同一批完全相同的样本做配对比较, 排除采样噪声), 而优化器仍然只调一次。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include "rl/sac.h"
#include "rl/sparse_moe.hpp"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

/*
   配对实验的前提: **两个实例必须真的同权重**。

   只拷 actor 是不够的 —— 策略头的梯度里有 critic 那一项 (dJ/dπ = α(logπ+1) − min_i Q_i),
   所以 critic 不同, 头部梯度就不同。这一条是踩过的坑: 第一版只 `actor.copyTo` 就在
   比较"1 遍 vs 2 遍的梯度比值", 结果量出 2.59 而不是 2.00, 看起来像梯度累加坏了,
   其实是两个实例的 critic 不一样 (方法论: "两个 agent 连着构造 = 从同一条随机流里
   取两份不同权重", 见 docs/issues_review.md 18.2 第 10 条)。
*/
static void copyAllNets(RL::SAC &src, RL::SAC &dst)
{
    src.actor.copyTo(dst.actor);
    for (int i = 0; i < RL::SAC::QNET_NUM; i++) {
        src.critics[i].copyTo(dst.critics[i]);
        src.critics[i].copyTo(dst.criticsTarget[i]);
    }
}

/* ============================================================
 *  1. 构造: 骨干、参数量、诊断
 * ============================================================ */
static void testConstruction()
{
    std::printf("\n[1] 构造 / 稀疏 MoE 骨干\n");

    RL::SAC sac(32, 16, 4);
    std::printf("  稀疏 MoE 层数 = %d, 专家数 = %d, top-k = %d\n",
                sac.moeLayerCount(), sac.moeExpertCount(), sac.moeTopK());
    std::printf("  params: actor=%lld  critic0=%lld\n",
                sac.actorParamCount(), sac.criticParamCount());

    CHECK(sac.moeLayerCount() > 0, "骨干里有稀疏 MoE 层");
    CHECK(sac.moeExpertCount() == RL::SAC::MOE_EXPERTS, "专家数等于编译期常量");
    CHECK(sac.moeTopK() == RL::SAC::MOE_TOPK, "top-k 等于编译期常量");
    CHECK(sac.actorParamCount() > 0, "参数量被真的数出来了 (不是 0)");
    CHECK(sac.moeExpertCount() == (int)RL::SAC::MOE_EXPERTS, "专家数与 SAC_MOE_EXPERTS 一致");
}

/* ============================================================
 *  2. R2: 掩码策略的口径
 * ============================================================ */
static void testMaskedPolicy()
{
    std::printf("\n[2] R2: 掩码策略 (合法集 Z≡1, 非法列恰好为 0)\n");

    const int A = 6;
    RL::SAC sac(24, 12, A);

    RL::Tensor state((std::size_t)24, 1);
    for (int i = 0; i < 24; i++) { state[(std::size_t)i] = (float)((i * 7) % 5) * 0.25f - 0.5f; }

    RL::Tensor mask((std::size_t)A, 1);
    mask.zero();
    const int legal[3] = {1, 3, 5};
    for (int i = 0; i < 3; i++) { mask[(std::size_t)legal[i]] = 1.0f; }

    /* --- 全量口径 --- */
    RL::Tensor piFull;
    sac.policy(state, RL::Tensor(), piFull);
    double sumFull = 0.0;
    for (int a = 0; a < A; a++) { sumFull += (double)piFull[(std::size_t)a]; }
    std::printf("  全量 softmax: Σπ = %.9f\n", sumFull);
    CHECK(std::fabs(sumFull - 1.0) < 1e-5, "全量口径 Σπ = 1");

    /* --- 掩码口径 --- */
    RL::Tensor piMask;
    sac.policy(state, mask, piMask);
    double sumLegal = 0.0;
    double sumAll = 0.0;
    for (int a = 0; a < A; a++) {
        sumAll += (double)piMask[(std::size_t)a];
    }
    for (int i = 0; i < 3; i++) {
        sumLegal += (double)piMask[(std::size_t)legal[i]];
    }
    std::printf("  掩码 softmax: 合法集 Σπ = %.9f, 全向量 Σπ = %.9f\n", sumLegal, sumAll);
    for (int a = 0; a < A; a++) {
        const bool ok = (mask[(std::size_t)a] > 0.5f);
        if (!ok) {
            CHECK(piMask[(std::size_t)a] == 0.0f, "非法动作的概率**恰好**为 0");
        }
    }
    CHECK(std::fabs(sumLegal - 1.0) < 1e-5, "掩码口径下合法集 Σπ = 1 (Z ≡ 1)");
    CHECK(std::fabs(sumAll - 1.0) < 1e-5, "掩码口径下全向量 Σπ 也 = 1");

    /*
        R2 的一条实质内容: 掩码改变了**学习问题**, 不是等价优化 ——
        "全量 softmax 后只在合法集上归一化"与"直接在合法集上 softmax"逐元素相等,
        (这正是 R1 那条捷径成立的依据), 拿它当参照来交叉验证实现。
    */
    if (sumFull > 1e-12) {
        /* 比的是"子集内的相对形状" (绝对概率差一个 1/Z, 那正是 R2 改掉的东西) */
        double subSum = 0.0;
        for (int i = 0; i < 3; i++) { subSum += (double)piFull[(std::size_t)legal[i]]; }
        double maxShapeErr = 0.0;
        for (int i = 0; i < 3; i++) {
            const double ref = (double)piFull[(std::size_t)legal[i]] / (subSum + 1e-30);
            const double got = (double)piMask[(std::size_t)legal[i]];
            maxShapeErr = std::max(maxShapeErr, std::fabs(got - ref));
        }
        std::printf("  与『全量后子集归一』的最大逐元素差 = %.3e\n", maxShapeErr);
        CHECK(maxShapeErr < 1e-6,
              "掩码 softmax ≡ 全量 softmax 后在合法集上归一 (R1 的等价性)");
    }
}

/* ============================================================
 *  3. R2: 非法列的头部权重梯度**恰好**为 0
 * ============================================================ */
static void testIllegalColumnGradient()
{
    std::printf("\n[3] R2: 非法列的头权重梯度恰好为 0\n");

    const int A = 5;
    const int S = 16;
    RL::SAC sac(S, 8, A);

    RL::Tensor state((std::size_t)S, 1);
    for (int i = 0; i < S; i++) { state[(std::size_t)i] = 0.1f * (float)(i % 4) - 0.15f; }
    RL::Tensor nextState = state;
    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[2] = 1.0f;
    RL::Tensor mask((std::size_t)A, 1);
    mask.zero();
    mask[0] = 1.0f;
    mask[2] = 1.0f;   /* 合法: {0, 2}; 非法: {1, 3, 4} */

    RL::Transition tr(state, action, nextState, 0.5f, true);
    tr.legalMask = mask;
    tr.nextLegalMask = mask;   /* 测试里两个局面的合法集取同一套 */

    /* 梯度缓冲此时还是全新的 (全 0) —— 这样"非法行 == 0"才是可判定的 */
    sac.resetMoeBatchStats();
    sac.accumulateGrad(tr);

    /* 策略头是 actor 的最后一层 (iFcLayer: w 是 (outputDim=actionDim, inputDim) ) */
    RL::iFcLayer *head =
        dynamic_cast<RL::iFcLayer*>(sac.actor[sac.actor.size() - 1]);
    CHECK(head != nullptr, "actor 的最后一层是 iFcLayer (策略头)");
    if (head == nullptr) {
        return;
    }

    const std::size_t rowStride = (std::size_t)head->g.w.sizes[0];
    double maxIllegal = 0.0;
    double maxLegal = 0.0;
    for (int a = 0; a < A; a++) {
        double m = 0.0;
        for (std::size_t k = 0; k < (std::size_t)head->inputDim; k++) {
            m = std::max(m, std::fabs((double)head->g.w.val[(std::size_t)a * rowStride + k]));
        }
        if (mask[(std::size_t)a] > 0.5f) {
            maxLegal = std::max(maxLegal, m);
        } else {
            maxIllegal = std::max(maxIllegal, m);
        }
    }
    std::printf("  |dL/dW| 最大: 合法行 = %.6e, 非法行 = %.6e\n", maxLegal, maxIllegal);
    CHECK(maxLegal > 0.0, "合法行确实拿到了梯度 (不是全 0, 否则这条断言没有意义)");
    CHECK(maxIllegal == 0.0, "非法行的权重梯度**恰好**为 0 (dz 在非法列上为 0)");

    /* 偏置同理 */
    double maxIllegalB = 0.0;
    for (int a = 0; a < A; a++) {
        if (mask[(std::size_t)a] <= 0.5f) {
            maxIllegalB = std::max(maxIllegalB, std::fabs((double)head->g.b.val[(std::size_t)a]));
        }
    }
    CHECK(maxIllegalB == 0.0, "非法行的偏置梯度恰好为 0");
}

/* ============================================================
 *  4. 学得动: Q(s,a) 朝 TD 目标走
 * ============================================================ */
static void testCriticLearns()
{
    std::printf("\n[4] critic 真的在学习 (TD 目标恰好 0.5)\n");

    const int A = 8;
    const int S = 24;
    RL::SAC sac(S, 16, A);
    sac.maxMemorySize = 512;

    RL::Tensor state((std::size_t)S, 1);
    state.zero();
    for (int i = 0; i < S; i += 3) { state[(std::size_t)i] = 1.0f; }
    RL::Tensor nextState = state;

    const int actionIdx = 3;
    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[(std::size_t)actionIdx] = 1.0f;

    /* 池里塞满"同一条"经验: reward = +0.5 且 done = true -> y ≡ 0.5 */
    for (int i = 0; i < 64; i++) {
        sac.perceive(state, action, nextState, 0.5f, true);
    }
    CHECK(sac.replaySize() == 64, "回放池里有 64 条经验");

    /* 起点 Q(s, a) */
    RL::Tensor pi, in;
    sac.policy(state, RL::Tensor(), pi);
    in = RL::Tensor::concat(0, state, pi);
    RL::Tensor &q0ref = sac.critics[0].forward(in);
    const double q0 = (double)q0ref[(std::size_t)actionIdx];

    float lastLoss = 0.0f;
    int updates = 0;
    for (int it = 0; it < 80; it++) {
        if (sac.learnFromReplay(8, 2, 0.05f)) {
            lastLoss = (float)sac.lastLoss;
            updates++;
        }
    }

    sac.policy(state, RL::Tensor(), pi);
    in = RL::Tensor::concat(0, state, pi);
    RL::Tensor &q1ref = sac.critics[0].forward(in);
    const double q1 = (double)q1ref[(std::size_t)actionIdx];

    std::printf("  Q(s,%d): %.5f -> %.5f (目标 0.5)\n", actionIdx, q0, q1);
    std::printf("  |误差|: %.5f -> %.5f, 最后一次 loss = %.6f, 更新 %d 次\n",
                std::fabs(q0 - 0.5), std::fabs(q1 - 0.5), (double)lastLoss, updates);
    std::printf("  alpha = %.4f, 学习步数 = %d\n",
                (double)sac.getAlpha(), sac.getLearningSteps());

    CHECK(updates > 0, "learnFromReplay 真的更新了 (返回 true)");
    CHECK(q1 > q0, "Q 朝目标方向移动");
    CHECK(std::fabs(q1 - 0.5) < std::fabs(q0 - 0.5), "误差下降");
    CHECK(std::isfinite((double)lastLoss), "批平均 loss 是有限值");
    CHECK(std::isfinite(sac.lastActorLoss), "批平均策略损失是有限值");
    CHECK(sac.getAlpha() >= 0.25f && sac.getAlpha() <= 5.0f, "alpha 被夹在 [0.25, 5] 内");
}

/* ============================================================
 *  5. P3/P4: 梯度累积与多 epoch 复用
 *
 *  这一节的三条断言各自钉住一件容易写错的事:
 *    (a) accumulateGrad **只攒不更新** —— 攒完之后权重必须逐位不变;
 *    (b) 梯度是**累加**的 —— 同一条样本多攒几个, 头部 |g| 的 L1 严格成比例;
 *    (c) learnFromReplay(batch, epochs) 的批是 **batch×epochs 条** (每遍重新抽),
 *        而优化器**只调一次**。这是 P4 的语义核心 —— 它**不是**"把同一批重复算
 *        几遍": 那种写法在 clipGrad=true 下连更新方向都改不了 (见 rl/sac.cpp)。
 * ============================================================ */
static void testGradientAccumulationAndEpochs()
{
    std::printf("\n[5] P3/P4: 梯度累积与多 epoch 复用\n");

    const int A = 4;
    const int S = 16;

    /* 同一条经验 -> 采样不可能带来差异, 于是"梯度大小"只反映累积条数 */
    RL::Tensor state((std::size_t)S, 1);
    for (int i = 0; i < S; i++) { state[(std::size_t)i] = 0.05f * (float)(i % 5) - 0.1f; }
    RL::Tensor nextState = state;
    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[1] = 1.0f;
    RL::Tensor mask((std::size_t)A, 1);
    mask.zero();
    mask[0] = 1.0f;
    mask[1] = 1.0f;
    mask[2] = 1.0f;   /* 合法: {0,1,2}, 非法: {3} */

    /*
       梯度大小探针: 头部 g.w 的 L1 范数。两个 SAC 用同一组权重与同一条经验,
       只差累积条数 —— 于是"4 条 = 2 条的 2 倍"是**配对**结论, 不受随机性影响。
    */
    auto headL1 = [](RL::SAC &s) {
        RL::iFcLayer *head =
            dynamic_cast<RL::iFcLayer*>(s.actor[s.actor.size() - 1]);
        double sum = 0.0;
        if (head == nullptr) { return sum; }
        for (std::size_t k = 0; k < head->g.w.size(); k++) {
            sum += std::fabs((double)head->g.w.val[k]);
        }
        return sum;
    };
    auto headW0 = [](RL::SAC &s) {
        RL::iFcLayer *head =
            dynamic_cast<RL::iFcLayer*>(s.actor[s.actor.size() - 1]);
        return head != nullptr ? (double)head->w.val[0] : 0.0;
    };

    RL::Transition tr(state, action, nextState, 0.3f, false);
    RL::Transition mtr(state, action, nextState, 0.3f, false);
    mtr.legalMask = mask;       /* 带掩码的那条走 R2 口径 */
    mtr.nextLegalMask = mask;

    RL::SAC one(S, 8, A);
    RL::SAC two(S, 8, A);
    /* 权重必须完全一致: 显式 copyTo (两个实例连着构造 = 两份不同的随机权重),
       而且必须**连 critic 一起拷** —— 见 copyAllNets 的说明 */
    copyAllNets(one, two);

    /* ---- (a) accumulateGrad 只攒不更新 ---- */
    const double w0 = headW0(one);
    one.resetMoeBatchStats();
    one.accumulateGrad(tr);
    one.accumulateGrad(mtr);
    const double w1 = headW0(one);
    std::printf("  两次 accumulateGrad 之后头权重的第 0 个元素: %.9f -> %.9f\n", w0, w1);
    CHECK(w0 == w1, "accumulateGrad 不碰权重 (优化器只在 applyGradients 里调)");

    /* ---- (b) 梯度累加是线性的 ---- */
    const double g1 = headL1(one);
    two.resetMoeBatchStats();
    for (int e = 0; e < 2; e++) {
        two.accumulateGrad(tr);
        two.accumulateGrad(mtr);
    }
    const double g2 = headL1(two);

    std::printf("  头部 |g| 的 L1: 2 条 = %.6e, 4 条 = %.6e, 比值 = %.4f\n",
                g1, g2, (g1 > 1e-30 ? g2 / g1 : 0.0));
    CHECK(g1 > 0.0, "累积确实产生了非零梯度");
    CHECK(std::fabs(g2 / g1 - 2.0) < 0.05, "梯度是累加的 (4 条 ≈ 2 条的 2 倍)");

    /* alpha 的梯度同样是整批累加后取平均 */
    CHECK(one.batchLossCount == 2, "批样本计数正确 (2 条)");

    /* ---- (c) learnFromReplay 的批 = batchSize × epochs, 优化器只调一次 ---- */
    RL::SAC sac(S, 8, A);
    sac.maxMemorySize = 64;
    for (int i = 0; i < 32; i++) {
        sac.perceive(state, action, nextState, 0.3f, true);
    }
    const int steps0 = sac.getLearningSteps();
    CHECK(sac.learnFromReplay(8, 1, 0.01f), "learnFromReplay(8, 1) 返回 true");
    const std::size_t samples1 = sac.lastBatchSamples;
    const int steps1 = sac.getLearningSteps();
    CHECK(sac.learnFromReplay(8, 3, 0.01f), "learnFromReplay(8, 3) 返回 true");
    const std::size_t samples3 = sac.lastBatchSamples;
    const int steps3 = sac.getLearningSteps();
    std::printf("  learnFromReplay(8,1): 攒了 %zu 条, 步数 %d -> %d\n",
                samples1, steps0, steps1);
    std::printf("  learnFromReplay(8,3): 攒了 %zu 条, 步数 %d -> %d\n",
                samples3, steps1, steps3);
    CHECK(samples1 == 8, "1 个 epoch = batchSize 条");
    CHECK(samples3 == 24, "3 个 epoch = batchSize×3 条 (每遍重新抽, 不是重复同一批)");
    CHECK(steps1 == steps0 + 1 && steps3 == steps1 + 1,
          "无论几个 epoch, 每次 learnFromReplay 只产生 1 次优化器更新");
    CHECK(!sac.learnFromReplay(8, 0, 0.01f), "epochs=0 直接返回 false (什么都没做)");
    CHECK(!sac.learnFromReplay(4096, 1, 0.01f), "池子不够大时返回 false");
}

/* ============================================================
 *  6. MoE: 辅助损失与坍缩诊断
 * ============================================================ */
static void testMoeRouting()
{
    std::printf("\n[6] MoE: 负载均衡辅助损失与使用计数\n");

    const int A = 8;
    const int S = 32;
    RL::SAC sac(S, 16, A);
    sac.maxMemorySize = 256;

    RL::Tensor state((std::size_t)S, 1);
    RL::Tensor nextState((std::size_t)S, 1);
    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[2] = 1.0f;

    for (int i = 0; i < 64; i++) {
        for (int k = 0; k < S; k++) {
            state[(std::size_t)k] = ((i * 13 + k * 7) % 11) * 0.1f - 0.5f;
            nextState[(std::size_t)k] = ((i * 5 + k * 3) % 9) * 0.1f - 0.4f;
        }
        sac.perceive(state, action, nextState, 0.2f, true);
    }

    sac.resetMoeUsage();
    for (int it = 0; it < 20; it++) {
        sac.learnFromReplay(8, 2, 0.01f);
    }

    std::vector<long long> usage;
    sac.moeUsage(usage);
    std::printf("  actor+critic 的专家累计使用 = [");
    long long total = 0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        std::printf("%s%lld", i ? ", " : "", usage[i]);
        total += usage[i];
    }
    std::printf("]  total=%lld\n", total);

    CHECK(!usage.empty(), "能取到专家使用计数");
    CHECK(total > 0, "训练时确实有前向经过稀疏 MoE");
    int unused = 0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        if (usage[i] == 0) { unused++; }
    }
    std::printf("  从未被选中的专家数 = %d / %zu\n", unused, usage.size());
    CHECK(unused < (int)usage.size(), "至少有一个专家被选中 (路由没有全死)");

    /* resetMoeUsage 清零 */
    sac.resetMoeUsage();
    std::vector<long long> after;
    sac.moeUsage(after);
    long long afterTotal = 0;
    for (std::size_t i = 0; i < after.size(); i++) { afterTotal += after[i]; }
    CHECK(afterTotal == 0, "resetMoeUsage 把累计计数清零了");
}

/* ============================================================
 *  7. 批统计的边界: 推理前向不该混进辅助损失的批统计
 *
 *  这一节必须有**正对照**: 如果不复位的版本和复位的版本给出一样的数字, 那这个测试
 *  什么也没证明。所以三个实例:
 *    A. 20 次推理前向 (输入各不相同) -> **复位** -> 1 条训练样本
 *    B. 没有推理前向            -> 复位 -> 1 条训练样本        (参照)
 *    C. 20 次推理前向 (同上)     -> **不复位** -> 1 条训练样本  (正对照)
 *  A 必须与 B 逐位相同 (复位划出了批边界), C 必须与 B **不同** (污染是真的)。
 * ============================================================ */
static void testBatchStatsBoundary()
{
    std::printf("\n[7] MoE 批统计: 推理前向不混进辅助损失\n");

    const int A = 6;
    const int S = 20;
    typedef RL::SparseMoE<RL::SACExpert, RL::SAC::MOE_EXPERTS,
                          RL::SAC::MOE_TOPK> MoeType;

    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[1] = 1.0f;

    /* 训练样本用的局面 */
    RL::Tensor trainState((std::size_t)S, 1);
    trainState.zero();
    trainState[0] = 1.0f;
    RL::Transition tr(trainState, action, trainState, 0.1f, true);

    /* 辅助损失的梯度探针: actor 第一个稀疏 MoE 层的门控权重梯度 L1 */
    auto moeGateGradL1 = [](RL::SAC &s) {
        MoeType *m = dynamic_cast<MoeType*>(s.actor[0]);
        double sum = 0.0;
        if (m == nullptr) { return sum; }
        for (std::size_t k = 0; k < m->g_wg.size(); k++) {
            sum += std::fabs((double)m->g_wg.val[k]);
        }
        return sum;
    };
    /* "跑 20 次推理前向", 输入每条都不一样 (否则 x̄ 的平均值会碰巧相同) */
    auto runInference = [&](RL::SAC &s) {
        RL::Tensor st((std::size_t)S, 1);
        RL::Tensor pi;
        for (int i = 0; i < 20; i++) {
            for (int k = 0; k < S; k++) {
                st[(std::size_t)k] = 0.05f * (float)((i * 7 + k * 3) % 13) - 0.3f;
            }
            s.policy(st, RL::Tensor(), pi);
        }
    };
    auto trainOne = [&](RL::SAC &s, bool resetFirst) {
        if (resetFirst) { s.resetMoeBatchStats(); }
        s.accumulateGrad(tr);
        MoeType *m = dynamic_cast<MoeType*>(s.actor[0]);
        if (m != nullptr) {
            m->addAuxGradient(0.1f);      /* 只注入辅助损失, 不跑优化器 */
        }
        return moeGateGradL1(s);
    };

    RL::SAC base(S, 12, A);
    RL::SAC a(S, 12, A), c(S, 12, A);
    copyAllNets(base, a);
    copyAllNets(base, c);

    const double gB = trainOne(base, true);
    runInference(a);
    const double gA = trainOne(a, true);
    runInference(c);
    const double gC = trainOne(c, false);

    std::printf("  门控 |g| (辅助损失): 复位 A = %.6e, 参照 B = %.6e, 不复位 C = %.6e\n",
                gA, gB, gC);
    CHECK(gB > 0.0, "参照组的辅助损失梯度非 0 (否则比较没有意义)");
    CHECK(gA == gB, "复位之后: 推理前向对训练批的辅助损失**零影响**");
    CHECK(std::fabs(gC - gB) > 0.0,
          "正对照: 不复位时推理前向**确实**污染了批统计 (这个测试是有分辨力的)");
}

/* ============================================================
 *  8. R2: 当前局面与下一局面**各自的**掩码
 *
 *  走一步之后合法集就变了, 所以软备份 V(s') 必须用 s' 自己的掩码。这条如果写错
 *  (两边都用 current), 不会报任何错 —— 只会把非法着法的 Q 值算进目标里。
 *  观测方式: 只改 nextLegalMask, critic 的梯度必须**变**; 而只改一个**与目标无关**的
 *  东西 (比如 nextState 之外的字段) 不该变。这里用两组互不相交的合法集做对照。
 * ============================================================ */
static void testNextMaskIsUsed()
{
    std::printf("\n[8] R2: s 与 s' 各自用自己的合法掩码\n");

    const int A = 4;
    const int S = 16;
    RL::Tensor state((std::size_t)S, 1);
    for (int i = 0; i < S; i++) { state[(std::size_t)i] = 0.07f * (float)(i % 3) - 0.1f; }
    RL::Tensor nextState = state;
    RL::Tensor action((std::size_t)A, 1);
    action.zero();
    action[0] = 1.0f;

    RL::Tensor curMask((std::size_t)A, 1);
    curMask.zero();
    curMask[0] = 1.0f;
    curMask[1] = 1.0f;
    curMask[2] = 1.0f;

    RL::Tensor nextMaskA((std::size_t)A, 1);
    nextMaskA.zero();
    nextMaskA[0] = 1.0f;
    nextMaskA[2] = 1.0f;      /* s' 的合法集 = {0,2} */

    RL::Tensor nextMaskB((std::size_t)A, 1);
    nextMaskB.zero();
    nextMaskB[1] = 1.0f;
    nextMaskB[3] = 1.0f;      /* s' 的合法集 = {1,3}, 与 A 互不相交 */

    auto criticHeadGrad = [](RL::SAC &s) {
        RL::iFcLayer *head =
            dynamic_cast<RL::iFcLayer*>(s.critics[0][s.critics[0].size() - 1]);
        double sum = 0.0;
        if (head == nullptr) { return sum; }
        for (std::size_t k = 0; k < head->g.w.size(); k++) {
            sum += std::fabs((double)head->g.w.val[k]);
        }
        return sum;
    };

    RL::SAC a(S, 8, A), b(S, 8, A);
    copyAllNets(a, b);

    RL::Transition ta(state, action, nextState, 0.1f, false);
    ta.legalMask = curMask;
    ta.nextLegalMask = nextMaskA;
    RL::Transition tb = ta;
    tb.nextLegalMask = nextMaskB;

    a.resetMoeBatchStats();
    a.accumulateGrad(ta);
    b.resetMoeBatchStats();
    b.accumulateGrad(tb);

    const double ga = criticHeadGrad(a);
    const double gb = criticHeadGrad(b);
    std::printf("  critic 头部 |g|: s'合法集={0,2} -> %.6e, ={1,3} -> %.6e\n", ga, gb);
    CHECK(ga > 0.0 && gb > 0.0, "两条都产生了非零的 critic 梯度");
    CHECK(std::fabs(ga - gb) > 1e-9,
          "改 nextLegalMask 会改变 critic 的梯度 (说明 s' 用的是它自己的掩码)");
}

int main()
{
    std::printf("========================================\n");
    std::printf("  RL::SAC 测试 (P3/P4/MoE/R2 移植)\n");
    std::printf("========================================\n");

    testConstruction();
    testMaskedPolicy();
    testIllegalColumnGradient();
    testCriticLearns();
    testGradientAccumulationAndEpochs();
    testMoeRouting();
    testBatchStatsBoundary();
    testNextMaskIsUsed();

    std::printf("\n========================================\n");
    std::printf("  %d 项检查, %d 项失败\n", g_checks, g_failed);
    std::printf("========================================\n");
    return g_failed == 0 ? 0 : 1;
}
