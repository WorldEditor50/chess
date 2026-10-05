/*
 * test_bc_main.cpp - 行为克隆 (BC) 的回归测试
 * ============================================================================
 *
 * 盯的是 **BC 的口径与承诺**, 不是棋力 (棋力只有 bench_anchor 能回答):
 *
 *   [1] 掩码交叉熵的解析梯度 == π − t —— 用**两条独立路径**对照:
 *       (a) 本工程的 `maskedCeLogitGrad` (直接给 logit 梯度);
 *       (b) SAC 那条雅可比 `maskedSoftmaxBackward(π, g)` 取 g = (π − t)/π。
 *       两者数学上恒等, 但代码路径完全不同, 所以"两边都对"比"自己证明自己"强。
 *       外加一条**中心差分**: 扰动 logits 后 CE 的变化应当 ≈ ⟨dz, δz⟩。
 *   [2] 非法列的梯度**恰好**为 0 (不是"很小" —— 掩码一旦换了位置就会暴露)。
 *   [3] 单局面过拟合: 反复更新之后 P(老师着法) -> ~1、CE -> ~0。
 *       梯度符号/方向错的话这条**只会变大** —— 它是"学没学对"的最短判据。
 *   [4] BC **不碰 critic**: PPO 的 actor 变了而 critic 逐字节不变; SAC 独立口径
 *       同理 (q1/q2 不变); SAC **共享口径**下必须"骨干变了、三个头里 Q 头不变"
 *       —— 这是两种口径语义差别最容易被读错的地方, 分开钉住。
 *   [5] 权重往返: BC 之后 save -> 新 agent load -> 读数逐位相同 (格式/结构指纹没被
 *       BC 破坏, 存下来的权重**能**被界面载入)。
 *   [6] 负对照: 老师着法不在合法集里时 `sampleFrom` 报 TeacherIllegal, 而内核
 *       `bcGradSparse` 报 TargetMissed 并**丢弃**它 (不是静默产出一条零信息样本)。
 *   [7] 口径对照: masked=1 (合法列) 与 masked=0 (全量 8100) 都能降 CE, 但两者是
 *       **不同的学习问题** —— 合法列口径下合法集的概率和恒为 1 (Z ≡ 1), 而全量口径
 *       只剩 Z < 1 (被非法槽位分走)。这条差别是"R2 之后 BC 该用哪个口径"的全部依据。
 *   [8] SAC-MoE-MLP 冒烟: 同一套胶水在稀疏 MoE 骨干上也跑得通且 CE 下降。
 *
 * 规模刻意很小: hidden=16 / expert=16 / AB 深度 1~2 / 几十个局面 —— 秒级, 可进 ctest。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "abagent.h"
#include "bcagent.hpp"
#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/bc.h"
#include "rl/util.hpp"
#include "sacazagent.h"
#include "sacazmoemlpagent.h"

static int g_checks = 0;
static int g_failed = 0;
#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

/* ============================================================
 *  工具
 * ============================================================ */

/*
 * 网络参数的**逐字节**摘要: 走内核自己的 v2 序列化 (逐比特无损 + 每张量 CRC32),
 * 再把文件字节做 FNV-1a。
 *
 * 为什么不用"把权重加起来比数值": 那种摘要对符号相反、量级相近的改动不敏感, 而这里
 * 要断言的恰恰是"一个字节都没动"。文件字节是最强的那种判据, 而且顺带证明了"这份
 * 权重真的能存下来"。
 */
static unsigned long long netDigest(RL::Net &net, const std::string &path)
{
    const int rc = net.save(path);
    if (rc != 0) {
        std::printf("  [WARN] netDigest: save 失败 (rc=%d)\n", rc);
        return 0;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) { return 0; }
    const std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::string data((std::size_t)size, '\0');
    f.read(&data[0], size);
    unsigned long long h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < data.size(); i++) {
        h ^= (unsigned char)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* 随机开局造局面 */
static void makePosition(int index, unsigned seed, Chess &out, int opening = 6)
{
    out.reset();
    out.sideToMove = Stone::COLOR_RED;
    std::mt19937 rng(seed + (unsigned)index * 7919u);
    for (int i = 0; i < opening; i++) {
        std::vector<Step *> legal;
        out.sample(out.sideToMove, legal);
        if (legal.empty()) { break; }
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        out.moveForward(&chosen, dummy);
    }
}

/* ============================================================
 *  [0] 数据: 用 AB 深度 D 打标签造一批样本
 * ============================================================ */
template <class AgentT>
static std::vector<RL::BCSample> label(AgentT &agent, int positions, int depth,
                                       int opening, unsigned seed, int &teacherIllegal)
{
    Chess envAb;
    envAb.reset();
    ABAgent ab(envAb, depth);

    std::vector<RL::BCSample> out;
    teacherIllegal = 0;
    for (int i = 0; i < positions; i++) {
        Chess pos;
        makePosition(i, seed, pos, opening);
        const int color = pos.sideToMove;
        envAb = pos;
        const Step mv = ab.getBestMove(color, depth);
        if (!ab.getScoreValid() || !mv.valid) { continue; }
        RL::BCSample s;
        BC::SampleFail why = BC::SampleFail::None;
        if (!BC::sampleFrom(agent, pos, color, mv, s, &why)) {
            if (why == BC::SampleFail::TeacherIllegal) { teacherIllegal++; }
            continue;
        }
        out.push_back(s);
    }
    return out;
}

/* ============================================================
 *  [1][2] 掩码交叉熵的解析梯度
 * ============================================================ */
static void testMaskedGradient()
{
    std::printf("\n[1] 掩码交叉熵的解析梯度 (两条独立路径 + 中心差分)\n");

    const int A = 6;                 /* 小动作空间, 便于手算 */
    RL::Tensor z(A, 1), mask(A, 1), target(A, 1), pi(A, 1), dz(A, 1), dzJac(A, 1), g(A, 1);
    z[0] = 0.4f;  z[1] = -0.7f; z[2] = 1.2f;
    z[3] = 0.1f;  z[4] = -1.5f; z[5] = 0.9f;
    /* 合法集 = {1, 2, 4}; 老师着法 = 2 */
    mask.zero();       mask[1] = 1.0f; mask[2] = 1.0f; mask[4] = 1.0f;
    target.zero();     target[2] = 1.0f;

    /* 掩码 softmax (与 agent 的 maskedSoftmax 同一公式, 这里自己写一遍以免依赖 agent) */
    const int legal[3] = { 1, 2, 4 };
    float m = z[legal[0]];
    for (int i = 1; i < 3; i++) { m = std::max(m, z[legal[i]]); }
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        pi[legal[i]] = std::exp(z[legal[i]] - m);
        sum += pi[legal[i]];
    }
    for (int i = 0; i < 3; i++) { pi[legal[i]] /= sum; }
    pi[0] = 0.0f; pi[3] = 0.0f; pi[5] = 0.0f;

    /* (a) 直接给 logit 梯度 */
    RL::maskedCeLogitGrad(pi, mask, target, dz);

    /* (b) 走雅可比: g = dL/dπ = (π − t)/π, 再 dz = Jᵀg */
    for (int a = 0; a < A; a++) {
        if (mask[a] <= 0.5f || pi[a] <= 1e-12f) { g[a] = 0.0f; continue; }
        g[a] = (pi[a] - target[a]) / pi[a];
    }
    float dot = 0.0f;
    for (int a = 0; a < A; a++) { dot += g[a] * pi[a]; }
    for (int a = 0; a < A; a++) { dzJac[a] = (pi[a] > 0.0f) ? pi[a] * (g[a] - dot) : 0.0f; }

    double maxDiff = 0.0;
    for (int a = 0; a < A; a++) { maxDiff = std::max(maxDiff, std::fabs((double)dz[a] - (double)dzJac[a])); }
    std::printf("    两条路径的最大差 = %.3e\n", maxDiff);
    CHECK(maxDiff < 1e-6, "解析梯度与雅可比路径一致 (π − t == Jᵀ((π−t)/π))");

    /* (c) 中心差分: dCE/dz_i ≈ (CE(z+εe_i) − CE(z−εe_i)) / 2ε */
    auto ceAt = [&](const std::vector<double> &zz) {
        double mm = zz[1];
        for (int i = 0; i < 3; i++) { mm = std::max(mm, zz[legal[i]]); }
        double s = 0.0;
        std::vector<double> p(A, 0.0);
        for (int i = 0; i < 3; i++) { p[legal[i]] = std::exp(zz[legal[i]] - mm); s += p[legal[i]]; }
        for (int i = 0; i < 3; i++) { p[legal[i]] /= s; }
        return -std::log(p[2] + 1e-12);      /* 老师着法 = 2 */
    };
    std::vector<double> base(A);
    for (int a = 0; a < A; a++) { base[a] = (double)z[a]; }
    const double eps = 1e-4;
    double worst = 0.0;
    for (int i = 0; i < 3; i++) {
        const int a = legal[i];
        std::vector<double> zp = base, zm = base;
        zp[a] += eps; zm[a] -= eps;
        const double fd = (ceAt(zp) - ceAt(zm)) / (2.0 * eps);
        const double an = (double)dz[a];
        worst = std::max(worst, std::fabs(fd - an));
        std::printf("      a=%d: 解析 %+.6f, 差分 %+.6f\n", a, an, fd);
    }
    CHECK(worst < 1e-4, "解析梯度与中心差分一致 (dCE/dz == p − t)");

    /* [2] 非法列恰好为 0 */
    bool allZero = true;
    for (int a = 0; a < A; a++) {
        if (mask[a] <= 0.5f && dz[a] != 0.0f) { allZero = false; }
    }
    CHECK(allZero, "非法列的梯度**恰好**为 0 (不是「很小」)");

    /* CE 数值与手算一致: −log π(2), π(2) = e^{z2}/Σ e^{z_legal} */
    const double ceHand = -std::log((double)(pi[2]));
    CHECK(std::fabs(RL::maskedCrossEntropy(pi, target) - ceHand) < 1e-6,
          "掩码 CE 的数值 == −log π(老师着法)");

    /* 合法列的 argmax */
    int best = -1;
    CHECK(RL::argmaxOnLegal(pi, mask, best) && best == 2, "合法列上的 argmax 选中老师着法");
}

/* ============================================================
 *  [3][4][5] PPO: 学得动 / 不碰 critic / 权重往返
 * ============================================================ */
static void testPPO()
{
    std::printf("\n[3] PPO+MCTS: BC 学得动 (prior top-1 上升, CE 下降)\n");
    RL::Random::setSeed(20240901u);

    Chess env;
    env.reset();
    PPOMCTSAgent agent(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                       RL::PPO::Backbone::MlpExperts);

    int illegal = 0;
    std::vector<RL::BCSample> set = label(agent, 60, 2, 6, 777u, illegal);
    CHECK(!set.empty(), "用 AB 深度 2 打出了标签");
    CHECK(illegal == 0, "老师的着法全都在合法集里 (视角一致)");
    std::printf("    样本 %llu 条 (老师着法不在合法集: %d)\n",
                (unsigned long long)set.size(), illegal);

    BC::Metrics before;
    BC::evaluate(agent, set, before);
    std::printf("    克隆前: top-1 %.2f%%, P(老师) %.5f, CE %.4f, 熵 %.4f\n",
                before.top1Pct, before.pTeacher, before.ce, before.entropy);

    /* [4] BC 之前的 critic 指纹 */
    const unsigned long long criticDigestBefore = netDigest(agent.ppo.critic, "bc_tmp_critic_a.dat");
    const double lastLossBefore = agent.ppo.lastLoss;

    /* 训练: 8 个 epoch, 每批 16 条 */
    long long used = 0, missed = 0;
    for (int ep = 0; ep < 8; ep++) {
        std::mt19937 rng(1000u + (unsigned)ep);
        std::vector<RL::BCSample> shuffled = set;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        for (std::size_t i = 0; i < shuffled.size(); i += 16) {
            const std::size_t hi = std::min(shuffled.size(), i + 16);
            std::vector<RL::BCSample> chunk(shuffled.begin() + (long)i, shuffled.begin() + (long)hi);
            const BC::UpdateStat st = BC::update(agent, chunk, 0.004f);
            used += st.used;
            missed += st.targetMiss;
        }
    }
    BC::Metrics after;
    BC::evaluate(agent, set, after);
    std::printf("    克隆后: top-1 %.2f%%, P(老师) %.5f, CE %.4f, 熵 %.4f\n",
                after.top1Pct, after.pTeacher, after.ce, after.entropy);
    std::printf("    参与更新 %lld 条, 目标落空 %lld 条\n", used, missed);

    CHECK(missed == 0, "没有样本因为目标落空被丢弃");
    CHECK(agent.ppo.bcSparseSteps == agent.ppo.bcSamples,
          "全部样本走了合法列 (R2) 口径, 没有静默退回全量");
    CHECK(after.ce < before.ce * 0.8, "BC 让 CE 明显下降");
    CHECK(after.top1Pct > before.top1Pct, "BC 让 prior top-1 上升");
    CHECK(after.entropy < before.entropy, "策略被推尖 (熵下降, 与 BC 的目标一致)");

    /* [4] critic 逐字节不变 */
    const unsigned long long criticDigestAfter = netDigest(agent.ppo.critic, "bc_tmp_critic_b.dat");
    CHECK(criticDigestBefore != 0 && criticDigestBefore == criticDigestAfter,
          "BC 之后 critic 的权重文件逐字节相同 (BC 不碰价值头)");
    /*
       lastLoss 没被 BC 写过 —— 注意**NaN 不等于 NaN**, 所以不能直接写 ==:
       一个"从没学过 critic"的 agent 上 lastLoss 就是 NaN, 而 NaN != NaN 会让这条
       断言假红 (本测试第一版正是这么红的)。
    */
    const double ll = agent.ppo.lastLoss;
    const bool lastLossUntouched =
        (std::isnan(ll) && std::isnan(lastLossBefore)) || (ll == lastLossBefore);
    CHECK(lastLossUntouched,
          "BC 没有写过 lastLoss (那是 critic 的 MSE, BC 只写 lastActorLoss)");

    /* [5] 权重往返: BC 的产物能被同一个 agent 载回来, 读数逐位相同 */
    std::printf("\n[5] 权重往返 (BC 存的权重能被载回来)\n");
    const bool saved = agent.saveModel("bc_tmp_ppo");
    CHECK(saved, "saveModel 成功");
    {
        PPOMCTSAgent reloaded(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                              RL::PPO::Backbone::MlpExperts);
        CHECK(reloaded.loadModel("bc_tmp_ppo"), "loadModel 成功 (结构指纹通过)");
        BC::Metrics m2;
        BC::evaluate(reloaded, set, m2);
        CHECK(std::fabs(m2.ce - after.ce) < 1e-9 && m2.top1Hits == after.top1Hits,
              "载入之后读数逐位相同 (权重真的存下来了)");
    }

    /* [7] 口径对照: 全量 8100 维 (maskedTrainHead=false) */
    std::printf("\n[7] 口径对照: 全量 8100 维 softmax (maskedTrainHead=false)\n");
    {
        PPOMCTSAgent denseAg(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                             RL::PPO::Backbone::MlpExperts);
        denseAg.ppo.maskedTrainHead = false;
        int il2 = 0;
        std::vector<RL::BCSample> set2 = label(denseAg, 24, 1, 4, 555u, il2);
        BC::Metrics b2;
        BC::evaluate(denseAg, set2, b2);
        long long denseUsed = 0;
        for (int ep = 0; ep < 4; ep++) {
            std::vector<RL::BCSample> chunk = set2;
            denseUsed += BC::update(denseAg, chunk, 0.004f).used;
        }
        BC::Metrics a2;
        BC::evaluate(denseAg, set2, a2);
        std::printf("    全量口径: CE %.4f -> %.4f (top-1 %.2f%% -> %.2f%%)\n",
                    b2.ce, a2.ce, b2.top1Pct, a2.top1Pct);
        CHECK(denseUsed == (long long)set2.size() * 4, "全量口径的样本都被用上");
        CHECK(a2.ce < b2.ce, "全量口径也能降 CE (对照臂本身是能跑的)");
        CHECK(denseAg.ppo.bcSparseSteps == 0, "对照臂一次都没走稀疏路径");
        CHECK(denseAg.ppo.bcSamples == (long long)set2.size() * 4, "对照臂把样本计在 Dense 口径下");

        /*
           ---- 两个口径的**归一化差别** (Z ≡ 1 vs Z < 1) ----
           这是"两个学习问题不同"的直接证据, 也是 README / docs 里那句
           "合法集上的概率质量 Z 实测 0.51~0.59" 的可复核版本:
             * 合法列口径 (actionMasked, R1 推理 = R2 训练同一个 softmax):
               概率只在合法列上归一 ⇒ Σ_{a∈合法} = **1**;
             * 全量口径 (actorP.forward): 8100 个槽位一起归一 ⇒ 合法集上只剩 Z < 1,
               剩下的质量分给了这个局面**根本走不了**的槽位。
           注意 actionMasked 是**推理**口径, 与 maskedTrainHead 无关 —— 所以这里
           用它量"合法列口径"、用全量前向量"全量口径", 两条各自都是生产代码。
        */
        {
            const RL::BCSample &s0 = set2[0];
            std::vector<float> probs;
            const bool ok0 = denseAg.ppo.actionMasked(s0.state, s0.legalIdx, probs);
            double sum0 = 0.0;
            for (std::size_t i = 0; i < probs.size(); i++) { sum0 += (double)probs[i]; }
            CHECK(ok0 && std::fabs(sum0 - 1.0) < 1e-4,
                  "合法列口径: 合法集上的概率和 == 1 (Z ≡ 1)");

            /*
               全量口径的 Z 取**整批平均**而不是单条: 单个局面上的 Z 波动很大
               (与"这个局面的合法着法先验多强"有关), 而 docs 里引用的 0.51~0.59 是
               均值口径 —— 两者的差别正是"单条读数不能当结论"的一个小例子。
            */
            double zSum = 0.0;
            int zN = 0;
            for (std::size_t k = 0; k < set2.size(); k++) {
                RL::Tensor &full = denseAg.ppo.actorP.forward(set2[k].state);
                double z = 0.0;
                for (std::size_t j = 0; j < set2[k].legalIdx.size(); j++) {
                    z += (double)full[(std::size_t)set2[k].legalIdx[j]];
                }
                zSum += z;
                zN++;
            }
            const double zMean = (zN > 0) ? zSum / (double)zN : 0.0;
            std::printf("    合法集上的质量: 合法列口径 Σ=%.6f, 全量口径 **批平均 Z=%.4f**"
                        " (Z<1 = 概率质量被非法槽位分走)\n", sum0, zMean);
            /*
               这个数**依赖网络的训练状态**, 所以断言只写"严格小于 1":
                 * 随机初始化的网络上 Z ≈ 合法着法数/8100 ≈ 0.005 (就是这次测到的量级);
                 * R2 (合法列口径) 训过的权重上, 质量几乎全在合法列上 ⇒ Z 接近 1;
                 * 历史记录里 `bench_ppo_distill` 那批旧口径权重上实测 0.51~0.59
                   (见 docs/issues_review.md 的 R1 一节)。
               三者都不是"对/错", 而是同一个问题的三个状态 —— 写在这里免得后人拿一个
               数字去断言另一个状态。
            */
            CHECK(zMean > 0.0 && zMean < 0.999,
                  "全量口径: 合法集上的质量 Z < 1 (两个学习问题确实不同)");
        }
    }
}

/* ============================================================
 *  [3][6] 单局面过拟合 + 负对照
 * ============================================================ */
static void testOverfitAndNegativeControl()
{
    std::printf("\n[3b] 单局面过拟合: P(老师) -> ~1 (梯度方向的最短判据)\n");
    RL::Random::setSeed(31415u);

    Chess env;
    env.reset();
    PPOMCTSAgent agent(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                       RL::PPO::Backbone::MlpExperts);

    int illegal = 0;
    std::vector<RL::BCSample> set = label(agent, 1, 2, 4, 42u, illegal);
    CHECK(set.size() == 1, "造出 1 条样本");

    BC::Metrics before;
    BC::evaluate(agent, set, before);
    std::vector<RL::BCSample> one(1, set[0]);
    for (int it = 0; it < 200; it++) {
        BC::update(agent, one, 0.02f);
    }
    BC::Metrics after;
    BC::evaluate(agent, set, after);
    std::printf("    P(老师) %.5f -> %.5f, CE %.4f -> %.4f\n",
                before.pTeacher, after.pTeacher, before.ce, after.ce);
    CHECK(after.pTeacher > 0.9, "过拟合到 P(老师着法) > 0.9");
    CHECK(after.ce < 0.15, "CE -> 接近 0");

    std::printf("\n[6] 负对照: 老师着法不在合法集里\n");
    /* 视角一致但着法非法: 取棋盘上第 0 个合法着法再改成"从 (0,0) 走到 (0,1)" */
    Chess pos;
    makePosition(3, 42u, pos, 4);
    const int color = pos.sideToMove;
    agent.chess = pos;
    {
        std::vector<Step *> legal;
        std::vector<int> idx;
        RL::Tensor mask((std::size_t)PPOMCTSAgent::ACTION_DIM, 1);
        agent.getLegalActions(color, legal, idx, mask);
        CHECK(!idx.empty(), "局面有合法着法");
        Steps::instance().put(legal);
    }
    Step bogus;                     /* valid=false */
    RL::BCSample s;
    BC::SampleFail why = BC::SampleFail::None;
    CHECK(!BC::sampleFrom(agent, pos, color, bogus, s, &why)
              && why == BC::SampleFail::TeacherInvalid,
          "无效的 Step -> TeacherInvalid");

    /*
       构造一个"合法集里没有它"的着法: 把某个合法着法的**终点**改到棋盘另一角。
       合法性由 sample() 判定, 所以这里只保证下标不在 legalIdx 里 (这正是要测的)。
    */
    Step fake;
    {
        std::vector<Step *> legal;
        std::vector<int> idx;
        RL::Tensor mask((std::size_t)PPOMCTSAgent::ACTION_DIM, 1);
        agent.getLegalActions(color, legal, idx, mask);
        fake = *legal[0];
        Steps::instance().put(legal);
        /* 把一个着法整体挪到"从 (9,8) 到 (0,0)": 几乎必然不在合法集里 */
        fake.pos = Pos(9, 8);
        fake.nextPos = Pos(0, 0);
        fake.valid = true;
    }
    bool notInLegal = true;
    {
        std::vector<Step *> legal;
        std::vector<int> idx;
        RL::Tensor mask((std::size_t)PPOMCTSAgent::ACTION_DIM, 1);
        agent.getLegalActions(color, legal, idx, mask);
        const int ai = agent.stepToActionIdx(fake, color);
        for (std::size_t i = 0; i < idx.size(); i++) { if (idx[i] == ai) { notInLegal = false; } }
        Steps::instance().put(legal);
    }
    CHECK(notInLegal, "构造出来的着法确实不在合法集里 (负对照成立)");
    why = BC::SampleFail::None;
    CHECK(!BC::sampleFrom(agent, pos, color, fake, s, &why)
              && why == BC::SampleFail::TeacherIllegal,
          "不在合法集里的着法 -> TeacherIllegal (而不是产出一条零信息样本)");

    /*
       内核层再查一遍: 就算调用方硬塞一条目标落空的样本, bcGradSparse 也必须
       **丢弃**它并计数 (它给的梯度是 π 而不是 0 —— 见 rl/bc.h 第 2 条)。
    */
    {
        PPOMCTSAgent g(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                       RL::PPO::Backbone::MlpExperts);
        int il = 0;
        std::vector<RL::BCSample> one2 = label(g, 1, 1, 4, 9u, il);
        CHECK(one2.size() == 1, "内核自检用的一条样本造好了");
        RL::BCSample bad = one2[0];
        bad.targetIdx.assign(1, 999999);      /* 不可能属于任何合法集 */
        bad.targetProb.assign(1, 1.0f);
        const long long missBefore = g.ppo.bcTargetMisses;
        const long long samplesBefore = g.ppo.bcSamples;
        const RL::PPO::BcOutcome oc =
            g.ppo.bcGradSparse(bad.state, bad.legalIdx, bad.targetIdx, bad.targetProb);
        CHECK(oc == RL::PPO::BcOutcome::TargetMissed, "内核把目标落空的样本判成 TargetMissed");
        CHECK(g.ppo.bcTargetMisses == missBefore + 1, "内核把它数进了 bcTargetMisses");
        CHECK(g.ppo.bcSamples == samplesBefore, "内核**没有**把它算成一次更新 (被丢弃)");
    }
}

/* ============================================================
 *  [4][5][8] SAC 三支
 * ============================================================
 * 三个 SAC 类是**互不继承的独立类** (repo 的刻意选择, 见 sacazagent.h 顶部),
 * 所以这里用模板 + 一个"再造一个同类实例"的工厂 lambda (构造签名三支不同:
 * SACAZAgent 没有 expertHidden/auxLossCoef 两个参数, 它的骨干是纯 MLP)。
 */
template <class SACLike>
static void trainSacBC(SACLike &agent, const char *tag,
                       const std::vector<RL::BCSample> &set,
                       BC::Metrics &before, BC::Metrics &after,
                       long long &used, long long &missed)
{
    BC::evaluate(agent, set, before);

    used = 0;
    missed = 0;
    for (int ep = 0; ep < 6; ep++) {
        std::mt19937 rng(2000u + (unsigned)ep);
        std::vector<RL::BCSample> shuffled = set;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        for (std::size_t i = 0; i < shuffled.size(); i += 8) {
            const std::size_t hi = std::min(shuffled.size(), i + 8);
            std::vector<RL::BCSample> chunk(shuffled.begin() + (long)i, shuffled.begin() + (long)hi);
            const BC::UpdateStat st = BC::update(agent, chunk, 0.004f);
            used += st.used;
            missed += st.targetMiss;
        }
    }
    BC::evaluate(agent, set, after);
    std::printf("    %s: top-1 %.2f%% -> %.2f%%, P(老师) %.5f -> %.5f, CE %.4f -> %.4f"
                " (用上 %lld 条, 落空 %lld)\n",
                tag, before.top1Pct, after.top1Pct, before.pTeacher, after.pTeacher,
                before.ce, after.ce, used, missed);

    CHECK(missed == 0, "SAC: 没有样本因为目标落空被丢弃");
    CHECK(after.ce < before.ce * 0.9, "SAC: BC 让 CE 明显下降");
    CHECK(after.top1Pct > before.top1Pct, "SAC: BC 让 prior top-1 上升");
}

/*
 * 独立口径 (TrunkMode::Separate): actor / q1 / q2 是三张各自背包干的网。
 * 断言: actor 变了, q1/q2 **逐字节**不变。
 */
template <class SACLike, class MakeFn>
static void runSacSeparateCase(SACLike &agent, const char *tag, MakeFn makeFresh)
{
    std::printf("\n[4] %s (独立口径): BC 学得动 / Q 网络不被碰\n", tag);
    int illegal = 0;
    std::vector<RL::BCSample> set = label(agent, 30, 1, 5, 2024u, illegal);
    CHECK(!set.empty() && illegal == 0, "SAC: 标签全部落在合法集里");

    const unsigned long long q1Before = netDigest(agent.q1, "bc_tmp_q1_a.dat");
    const unsigned long long q2Before = netDigest(agent.q2, "bc_tmp_q2_a.dat");
    const unsigned long long actorBefore = netDigest(agent.actor, "bc_tmp_actor_a.dat");

    BC::Metrics before, after;
    long long used = 0, missed = 0;
    trainSacBC(agent, tag, set, before, after, used, missed);

    const unsigned long long q1After = netDigest(agent.q1, "bc_tmp_q1_b.dat");
    const unsigned long long q2After = netDigest(agent.q2, "bc_tmp_q2_b.dat");
    const unsigned long long actorAfter = netDigest(agent.actor, "bc_tmp_actor_b.dat");
    CHECK(actorBefore != actorAfter, "SAC: actor 被更新了");
    CHECK(q1Before != 0 && q1Before == q1After && q2Before == q2After,
          "SAC: q1/q2 逐字节不变 (BC 只动策略路径)");

    /* 权重往返: 存下来的权重能被同类新实例载入, 读数逐位相同 */
    const std::string prefix = std::string("bc_tmp_") + tag;
    CHECK(agent.saveModel(prefix), "SAC: saveModel 成功");
    {
        SACLike *reloaded = makeFresh();
        CHECK(reloaded->loadModel(prefix), "SAC: loadModel 成功 (结构指纹通过)");
        BC::Metrics m2;
        BC::evaluate(*reloaded, set, m2);
        CHECK(std::fabs(m2.ce - after.ce) < 1e-9, "SAC: 载入之后 CE 逐位相同");
        delete reloaded;
    }
}

/*
 * 共享骨干口径 (TrunkMode::Shared): **一个骨干 + 三个头**。
 * 这一支的断言必须与独立口径**分开写**, 否则会把预期的语义读成缺陷:
 *   * `agent.actor` / `agent.q1` 这些"视图"都**包含骨干** (共享同一批层对象),
 *     所以它们**都会变** (骨干变了) —— 拿它们当"没被碰过"的证据是错的;
 *   * 真正的断言是: **Q 头自己的权重** (q1Head / q2Head) 逐字节不变, 而
 *     骨干 (trunk) 与策略头 (actorHead) 确实被更新了。
 */
template <class SACLike, class MakeFn>
static void runSacSharedCase(SACLike &agent, const char *tag, MakeFn makeFresh)
{
    std::printf("\n[4] %s (共享骨干): 骨干会变 / Q 头不变\n", tag);
    int illegal = 0;
    std::vector<RL::BCSample> set = label(agent, 30, 1, 5, 2024u, illegal);
    CHECK(!set.empty() && illegal == 0, "SAC(共享): 标签全部落在合法集里");

    const unsigned long long trunkBefore = netDigest(agent.trunk, "bc_tmp_trunk_a.dat");
    const unsigned long long headBefore = netDigest(agent.actorHead, "bc_tmp_ahead_a.dat");
    const unsigned long long q1hBefore = netDigest(agent.q1Head, "bc_tmp_q1h_a.dat");
    const unsigned long long q2hBefore = netDigest(agent.q2Head, "bc_tmp_q2h_a.dat");

    BC::Metrics before, after;
    long long used = 0, missed = 0;
    trainSacBC(agent, tag, set, before, after, used, missed);

    const unsigned long long trunkAfter = netDigest(agent.trunk, "bc_tmp_trunk_b.dat");
    const unsigned long long headAfter = netDigest(agent.actorHead, "bc_tmp_ahead_b.dat");
    const unsigned long long q1hAfter = netDigest(agent.q1Head, "bc_tmp_q1h_b.dat");
    const unsigned long long q2hAfter = netDigest(agent.q2Head, "bc_tmp_q2h_b.dat");
    CHECK(trunkBefore != trunkAfter, "共享口径: 骨干被策略梯度更新了 (共享表示的预期代价)");
    CHECK(headBefore != headAfter, "共享口径: 策略头被更新了");
    CHECK(q1hBefore != 0 && q1hBefore == q1hAfter && q2hBefore == q2hAfter,
          "共享口径: **Q 头自己的权重**逐字节不变");

    const std::string prefix = std::string("bc_tmp_") + tag;
    CHECK(agent.saveModel(prefix), "SAC(共享): saveModel 成功");
    {
        SACLike *reloaded = makeFresh();
        CHECK(reloaded->loadModel(prefix), "SAC(共享): loadModel 成功 (4 个文件的口径)");
        BC::Metrics m2;
        BC::evaluate(*reloaded, set, m2);
        CHECK(std::fabs(m2.ce - after.ce) < 1e-9, "SAC(共享): 载入之后 CE 逐位相同");
        delete reloaded;
    }
}

static void testSAC()
{
    Chess env;
    env.reset();

    RL::Random::setSeed(4242u);
    {
        SACAZAgent agent(env, 16, 0.99f, 0.001f, 1.5f, SACAZAgent::TrunkMode::Separate);
        runSacSeparateCase(agent, "sac_sep",
                           [&env]() { return new SACAZAgent(env, 16, 0.99f, 0.001f, 1.5f,
                                                            SACAZAgent::TrunkMode::Separate); });
    }
    {
        SACAZAgent agent(env, 16, 0.99f, 0.001f, 1.5f, SACAZAgent::TrunkMode::Shared);
        runSacSharedCase(agent, "sac_shared",
                         [&env]() { return new SACAZAgent(env, 16, 0.99f, 0.001f, 1.5f,
                                                          SACAZAgent::TrunkMode::Shared); });
    }
    /* [8] 稀疏 MoE 骨干的冒烟 (同一套胶水在另一支骨干上也要能跑) */
    {
        SACAZMoEMlpAgent agent(env, 16, 0.99f, 0.001f, 1.5f, 16, 0.1f,
                               SACAZMoEMlpAgent::TrunkMode::Separate);
        runSacSeparateCase(agent, "sac_moe_mlp",
                           [&env]() { return new SACAZMoEMlpAgent(env, 16, 0.99f, 0.001f, 1.5f, 16, 0.1f,
                                                                  SACAZMoEMlpAgent::TrunkMode::Separate); });
    }
}

/* ============================================================
 *  [9] 软目标 (多深度一致): "不直接用 onehot 会不会更好" 的落地口径
 * ============================================================
 *
 * 这一节盯的是**软目标的接口契约**, 不是"它是不是更好" —— 后者只有 A/B 实验能回答
 * (4000 局面 / 深度 3 / 8 epoch / 4 个种子: 可比口径的**留出 KL 4/4 都更低**, 而硬口径的
 * 留出 top-1 略低 3/4, `train−留出` 差 4/4 都更小; 见 docs/behavior_cloning_2026_10.md §9)。
 * 契约有四条:
 *
 *   (a) `softTargetFromAB` 的输出**真的是一个概率分布**: 每项 > 0、Σ = 1、
 *       每一项都是合法着法、同一着法被多层选中时**合并**;
 *   (b) 深度 1 (只有一层投票) 必然退化成 one-hot ⇒ H(t) = 0。这条是"软目标没有
 *       把旧口径偷偷换掉"的判据: 用户没勾那个开关时读到的数与以前逐位一致;
 *   (c) **多老师 `sampleFrom` 的合法性闸门是整条的**: 任何一个候选着法不合法 ⇒
 *       整条样本作废, 不做"丢掉不合法的那几项"。丢了之后 Σt < 1, 而 dL/dz = π − t
 *       在 Σt ≠ 1 时不再是"匹配一个分布" —— 梯度里多出一份偏置, 读数上看不出来;
 *   (d) CE 的**下界就是 H(t)**: 用一份手算的 (π, t) 对照 `maskedCrossEntropy` 与
 *       独立的 H(t) + KL(t‖π)。这是文档里那句"软目标的 CE 不能与 one-hot 的 CE
 *       直接比"的数值依据 —— 它一旦不成立, 报告里的 targetH 就是装饰。
 */
static void testSoftTargets()
{
    std::printf("\n[9] 软目标 (多深度一致): 分布契约 / one-hot 退化 / 整条闸门 / CE 下界\n");

    /* ---- (a)+(b) softTargetFromAB: 分布 + 退化 ---- */
    {
        int nonOneHot = 0, oneHotDepth1 = 0, checked = 0;
        double worstSumErr = 0.0;
        bool allLegal = true;
        for (int i = 0; i < 12; i++) {
            Chess pos;
            makePosition(i, 777u, pos, 6);
            const int color = pos.sideToMove;

            /* 这个局面的完整合法集 (判断"老师给的着法合不合法"的基准) */
            Chess env;
            env.reset();
            env = pos;
            std::vector<Step *> legal;
            env.sample(color, legal);
            std::vector<std::pair<int, int> > legalKey;
            for (std::size_t k = 0; k < legal.size(); k++) {
                legalKey.push_back(std::make_pair((int)legal[k]->id,
                                                 (int)(legal[k]->nextPos.x * 10
                                                       + legal[k]->nextPos.y)));
            }
            Steps::instance().put(legal);
            if (legalKey.empty()) { continue; }
            checked++;

            for (int depths = 1; depths <= 3; depths += 2) {
                Chess board = pos;
                std::vector<Step> mv;
                std::vector<float> pr;
                if (!BC::softTargetFromAB(board, color, depths, depths, mv, pr)) { continue; }
                CHECK(mv.size() == pr.size() && !mv.empty(),
                      "软目标: 着法与权重等长且非空");
                double sum = 0.0;
                for (std::size_t k = 0; k < pr.size(); k++) {
                    CHECK(pr[k] > 0.0f, "软目标: 每一项权重都 > 0 (0 权重的项不该出现)");
                    sum += (double)pr[k];
                    const int key = (int)mv[k].id
                                    + (int)(mv[k].nextPos.x * 10 + mv[k].nextPos.y) * 1000;
                    bool found = false;
                    for (std::size_t j = 0; j < legalKey.size(); j++) {
                        if (legalKey[j].first + legalKey[j].second * 1000 == key) { found = true; }
                    }
                    if (!found) { allLegal = false; }
                }
                worstSumErr = std::max(worstSumErr, std::fabs(sum - 1.0));
                /* 同一着法不能出现两次 (合并过的证据: 下标必须互不相同) */
                for (std::size_t x = 0; x < mv.size(); x++) {
                    for (std::size_t y = x + 1; y < mv.size(); y++) {
                        CHECK(!(mv[x].id == mv[y].id && mv[x].nextPos == mv[y].nextPos),
                              "软目标: 同一个着法被合并成一项 (不是重复投票)");
                    }
                }
                if (depths == 1) {
                    if (mv.size() == 1 && std::fabs((double)pr[0] - 1.0) < 1e-6) { oneHotDepth1++; }
                } else if (mv.size() > 1) {
                    nonOneHot++;
                }
            }
        }
        std::printf("    检查了 %d 个局面; Σp 的最大偏差 = %.3e\n", checked, worstSumErr);
        CHECK(checked >= 8, "软目标: 造出了足够多的局面");
        CHECK(worstSumErr < 1e-6, "软目标: Σp == 1 (真的是一个概率分布)");
        CHECK(allLegal, "软目标: 每一项都是该局面的**合法**着法");
        CHECK(oneHotDepth1 > 0, "软目标: 深度 1 退化成 one-hot (只有一项且 p=1 ⇒ H(t)=0)");
        CHECK(nonOneHot > 0, "软目标: 深度 3 在多数局面上真的给出多于一项 (H(t)>0)");
    }

    /* ---- 目标熵的定义 (手算一个已知分布) ---- */
    {
        RL::BCSample s;
        s.targetIdx.push_back(1);
        s.targetIdx.push_back(2);
        s.targetProb.push_back(0.5f);
        s.targetProb.push_back(0.5f);
        CHECK(std::fabs(BC::targetEntropyOf(s) - std::log(2.0)) < 1e-6,
              "H(t) 在半半分布上 == ln 2");
        s.targetProb[0] = 1.0f; s.targetProb[1] = 0.0f;
        CHECK(std::fabs(BC::targetEntropyOf(s)) < 1e-9, "H(t) 在 one-hot 上 == 0");
    }

    /* ---- (c) 多老师 sampleFrom: 归一 / 合并 / 整条闸门 ---- */
    {
        Chess env;
        env.reset();
        PPOMCTSAgent agent(env, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, true,
                           RL::PPO::Backbone::MlpExperts);
        RL::Random::setSeed(20240901u);

        Chess pos;
        makePosition(5, 42u, pos, 5);
        const int color = pos.sideToMove;

        Chess probe = pos;
        std::vector<Step *> legal;
        probe.sample(color, legal);
        CHECK(legal.size() >= 2, "多老师: 这个局面至少有两个合法着法");
        if (legal.size() >= 2) {
            Step m0 = *legal[0];
            Step m1 = *legal[1];
            Steps::instance().put(legal);

            /* 未归一权重 -> 归一 (票数写成 1/3/1, Σ=5) */
            std::vector<Step> tv;
            std::vector<float> pv;
            tv.push_back(m0); pv.push_back(1.0f);
            tv.push_back(m1); pv.push_back(3.0f);
            tv.push_back(m0); pv.push_back(1.0f);
            RL::BCSample s;
            BC::SampleFail why = BC::SampleFail::None;
            const bool ok = BC::sampleFrom(agent, pos, color, tv, pv, s, &why);
            CHECK(ok, "多老师: 合法候选造出样本");
            if (ok) {
                double sum = 0.0;
                for (std::size_t k = 0; k < s.targetProb.size(); k++) {
                    sum += (double)s.targetProb[k];
                }
                CHECK(std::fabs(sum - 1.0) < 1e-6, "多老师: 权重被强制归一 (Σt == 1)");
                CHECK(s.targetIdx.size() == 2, "多老师: 重复的着法被合并成一项");
                /* m0 两票 = 2/5 */
                for (std::size_t k = 0; k < s.targetIdx.size(); k++) {
                    if (s.targetIdx[k] == agent.stepToActionIdx(m0, color)) {
                        CHECK(std::fabs((double)s.targetProb[k] - 0.4) < 1e-5,
                              "多老师: 合并后的权重 == 票数占比 (2/5)");
                    }
                }
            }

            /* 整条闸门: 一个候选不合法 ⇒ 整条作废 (不是丢掉那一项) */
            Step fake;
            {
                std::vector<Step *> lg;
                std::vector<int> idx;
                RL::Tensor mask((std::size_t)PPOMCTSAgent::ACTION_DIM, 1);
                agent.chess = pos;
                agent.getLegalActions(color, lg, idx, mask);
                fake = *lg[0];
                Steps::instance().put(lg);
                fake.pos = Pos(9, 8);
                fake.nextPos = Pos(0, 0);
                fake.valid = true;
            }
            std::vector<Step> tv2;
            std::vector<float> pv2;
            tv2.push_back(m0); pv2.push_back(1.0f);
            tv2.push_back(fake); pv2.push_back(1.0f);
            why = BC::SampleFail::None;
            RL::BCSample s2;
            const bool ok2 = BC::sampleFrom(agent, pos, color, tv2, pv2, s2, &why);
            CHECK(!ok2 && why == BC::SampleFail::TeacherIllegal,
                  "多老师: 只要有一项不合法, **整条**样本作废 (TeacherIllegal)");

            /* 单老师重载与"只有一个候选的多老师"逐位一致 (旧路径没被换口径) */
            RL::BCSample a1, a2;
            const bool okA = BC::sampleFrom(agent, pos, color, m0, a1);
            std::vector<Step> one;
            std::vector<float> oneP;
            one.push_back(m0); oneP.push_back(1.0f);
            const bool okB = BC::sampleFrom(agent, pos, color, one, oneP, a2);
            CHECK(okA && okB, "单老师/多老师: 两条路都能造出样本");
            bool same = (a1.targetIdx == a2.targetIdx)
                        && (a1.targetProb == a2.targetProb)
                        && (a1.legalIdx == a2.legalIdx)
                        && (a1.posHash == a2.posHash);
            CHECK(same, "单老师重载 == 单候选多老师重载 (one-hot 路径逐位不变)");
            CHECK(a1.targetProb.size() == 1 && std::fabs((double)a1.targetProb[0] - 1.0) < 1e-6,
                  "one-hot 样本: 只有一项且权重为 1");
        }
    }

    /* ---- (d) CE 的下界就是 H(t): CE = H(t) + KL(t‖π) ---- */
    {
        const int A = 5;
        RL::Tensor pi(A, 1), target(A, 1);
        /* 合法集 = {0,1,2,3,4} 全用上 (掩码不影响这条等式), 目标是一个三项分布 */
        target.zero();  pi.zero();
        target[0] = 0.5f; target[1] = 0.25f; target[2] = 0.25f;
        pi[0] = 0.2f; pi[1] = 0.3f; pi[2] = 0.1f; pi[3] = 0.35f; pi[4] = 0.05f;

        double h = 0.0, kl = 0.0;
        for (int a = 0; a < A; a++) {
            const double t = (double)target[a];
            const double p = (double)pi[a];
            if (t > 0.0) { h -= t * std::log(t); kl += t * std::log(t / p); }
        }
        const double ce = RL::maskedCrossEntropy(pi, target);
        std::printf("    CE = %.6f, H(t) = %.6f, KL = %.6f (H+KL = %.6f)\n",
                    ce, h, kl, h + kl);
        CHECK(std::fabs(ce - (h + kl)) < 1e-6, "掩码 CE 与软目标一致: CE == H(t) + KL(t‖π)");
        /* π == t 时 CE 恰好落在下界上 (这条是"CE 不可能降到 0"的数值证据) */
        RL::Tensor piExact(A, 1);
        piExact.zero();
        for (int a = 0; a < A; a++) { piExact[a] = target[a]; }
        CHECK(std::fabs(RL::maskedCrossEntropy(piExact, target) - h) < 1e-6,
              "π == t 时 CE 恰好等于 H(t) (软目标的 CE 下界)");
        RL::Tensor oneHot(A, 1);
        oneHot.zero(); oneHot[0] = 1.0f;
        /*
           "只看 argmax 那一项" 的口径会算成 −log π(argmax); 软目标是**整个支撑集**求和,
           所以它严格更大 (π 在支撑集上不是一个点)。这条盯的正是"评估口径换了一半" ——
           metrics 里的 CE 一旦退回 one-hot 写法, 软目标那一臂的读数会被系统性压低。
        */
        const double ceArgmaxOnly = -std::log((double)oneHot[0] + 1e-8);
        const double ceFull = RL::maskedCrossEntropy(oneHot, target);
        std::printf("    软目标 CE = %.6f > 只看 argmax 的 %.6f\n", ceFull, ceArgmaxOnly);
        CHECK(ceFull > ceArgmaxOnly + 1e-6,
              "软目标: CE 是**整个支撑集**上的 −Σt·log π (不是只看 argmax 那一项)");
        CHECK(ceFull > h, "软目标: CE 永远不低于 H(t) (下界)");
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("=== 行为克隆 (BC) 的口径与承诺 ===\n");

    testMaskedGradient();
    testPPO();
    testOverfitAndNegativeControl();
    testSAC();
    testSoftTargets();

    std::printf("\n=== 断言 %d 项, 失败 %d 项 ===\n", g_checks, g_failed);
    if (g_failed == 0) {
        std::printf("全部通过\n");
    }
    return (g_failed == 0) ? 0 : 1;
}
