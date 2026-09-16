/* 一次性 scratch: 查 test_sac 里两处失败的原因 (梯度累加是否线性 / 批统计复位) */
#include <cstdio>
#include <cmath>
#include <vector>
#include "rl/sac.h"
#include "rl/sparse_moe.hpp"

typedef RL::SparseMoE<RL::SACExpert, RL::SAC::MOE_EXPERTS, RL::SAC::MOE_TOPK> MoeType;

static RL::iFcLayer *headOf(RL::SAC &s)
{
    return dynamic_cast<RL::iFcLayer*>(s.actor[s.actor.size() - 1]);
}

static double headL1(RL::SAC &s)
{
    RL::iFcLayer *h = headOf(s);
    double sum = 0.0;
    for (std::size_t k = 0; k < h->g.w.size(); k++) {
        sum += std::fabs((double)h->g.w.val[k]);
    }
    return sum;
}

/* 两个网络的权重是否逐位相同 (遍历所有 iFcLayer 的 w) */
static void compareWeights(RL::Net &a, RL::Net &b, const char *tag)
{
    double maxDiff = 0.0;
    for (std::size_t i = 0; i < a.size(); i++) {
        RL::iFcLayer *fa = dynamic_cast<RL::iFcLayer*>(a[i]);
        RL::iFcLayer *fb = dynamic_cast<RL::iFcLayer*>(b[i]);
        if (fa == nullptr || fb == nullptr) { continue; }
        if (fa->w.size() != fb->w.size()) {
            std::printf("  %s: 层 %zu 形状不同 (%zu vs %zu)\n", tag, i,
                        fa->w.size(), fb->w.size());
            continue;
        }
        for (std::size_t k = 0; k < fa->w.size(); k++) {
            maxDiff = std::max(maxDiff, std::fabs((double)fa->w.val[k] - (double)fb->w.val[k]));
        }
    }
    std::printf("  %s: 权重最大逐元素差 = %.3e\n", tag, maxDiff);
}

int main()
{
    RL::Random::setSeed(7u);
    const int A = 4, S = 16;

    RL::Tensor state(S, 1);
    for (int i = 0; i < S; i++) { state[i] = 0.05f * (float)(i % 5) - 0.1f; }
    RL::Tensor nextState = state;
    RL::Tensor action(A, 1);
    action.zero();
    action[1] = 1.0f;
    RL::Tensor mask(A, 1);
    mask.zero();
    mask[0] = 1.0f; mask[1] = 1.0f; mask[2] = 1.0f;

    RL::Transition tr(state, action, nextState, 0.3f, false);
    RL::Transition mtr(state, action, nextState, 0.3f, false);
    mtr.legalMask = mask;

    std::printf("\n== A. 一个网络内: 同一条样本重复累积, L1 是否线性 ==\n");
    {
        RL::SAC s(S, 8, A);
        double prev = 0.0;
        for (int n = 1; n <= 4; n++) {
            s.resetMoeBatchStats();
            s.accumulateGrad(tr);
            const double l1 = headL1(s);
            std::printf("  第 %d 次累积: L1 = %.6e   (增量 = %.6e)\n",
                        n, l1, l1 - prev);
            prev = l1;
        }
    }

    std::printf("\n== B. 掩码样本重复累积 ==\n");
    {
        RL::SAC s(S, 8, A);
        double prev = 0.0;
        for (int n = 1; n <= 3; n++) {
            s.resetMoeBatchStats();
            s.accumulateGrad(mtr);
            const double l1 = headL1(s);
            std::printf("  第 %d 次累积: L1 = %.6e   (增量 = %.6e)\n",
                        n, l1, l1 - prev);
            prev = l1;
        }
    }

    std::printf("\n== C. copyTo 之后两个网络是否同权重 ==\n");
    {
        RL::SAC one(S, 8, A), two(S, 8, A);
        one.actor.copyTo(two.actor);
        compareWeights(one.actor, two.actor, "actor");
        /* 也看 MoE 的门控 (不是 iFcLayer, 单独比) */
        MoeType *m1 = dynamic_cast<MoeType*>(one.actor[0]);
        MoeType *m2 = dynamic_cast<MoeType*>(two.actor[0]);
        double d = 0.0;
        for (std::size_t k = 0; k < m1->wg.size(); k++) {
            d = std::max(d, std::fabs((double)m1->wg.val[k] - (double)m2->wg.val[k]));
        }
        std::printf("  门控 wg 最大逐元素差 = %.3e\n", d);

        /* 同一条样本 -> 两个网络的头部梯度应该逐位相同 */
        one.resetMoeBatchStats(); one.accumulateGrad(tr);
        two.resetMoeBatchStats(); two.accumulateGrad(tr);
        RL::iFcLayer *h1 = headOf(one), *h2 = headOf(two);
        double gd = 0.0;
        for (std::size_t k = 0; k < h1->g.w.size(); k++) {
            gd = std::max(gd, std::fabs((double)h1->g.w.val[k] - (double)h2->g.w.val[k]));
        }
        std::printf("  同一样本: 头部梯度最大逐元素差 = %.3e\n", gd);
    }

    std::printf("\n== D. 两次累积 vs 一次: 逐元素比值 ==\n");
    {
        RL::SAC one(S, 8, A), two(S, 8, A);
        one.actor.copyTo(two.actor);
        one.resetMoeBatchStats();
        one.accumulateGrad(tr);
        one.accumulateGrad(mtr);
        two.resetMoeBatchStats();
        two.accumulateGrad(tr);
        two.accumulateGrad(mtr);
        two.accumulateGrad(tr);
        two.accumulateGrad(mtr);
        RL::iFcLayer *h1 = headOf(one), *h2 = headOf(two);
        double maxRel = 0.0, worst = 0.0;
        for (std::size_t k = 0; k < h1->g.w.size(); k++) {
            const double a = (double)h1->g.w.val[k];
            const double b = (double)h2->g.w.val[k];
            if (std::fabs(a) > 1e-9) {
                maxRel = std::max(maxRel, std::fabs(b / a - 2.0));
                if (std::fabs(b / a - 2.0) > maxRel - 1e-12) { worst = a; }
            }
        }
        std::printf("  |g2/g1 - 2| 的最大值 = %.6e  (g1 的某个分量 = %.6e)\n", maxRel, worst);
        std::printf("  L1: %.6e -> %.6e (比值 %.4f)\n", headL1(one), headL1(two),
                    headL1(two) / headL1(one));
    }

    std::printf("\n== E. 批统计: 复位的效果 ==\n");
    {
        RL::SAC base(S, 12, A), a(S, 12, A), c(S, 12, A);
        base.actor.copyTo(a.actor);
        base.actor.copyTo(c.actor);
        RL::Tensor st(S, 1);
        RL::Tensor pi;
        for (int i = 0; i < 20; i++) {
            for (int k = 0; k < S; k++) { st[k] = 0.05f * (float)((i * 7 + k * 3) % 13) - 0.3f; }
            a.policy(st, RL::Tensor(), pi);
            c.policy(st, RL::Tensor(), pi);
        }
        auto gate = [&tr](RL::SAC &s, bool reset) {
            if (reset) { s.resetMoeBatchStats(); }
            s.accumulateGrad(tr);
            MoeType *m = dynamic_cast<MoeType*>(s.actor[0]);
            m->addAuxGradient(0.1f);
            double sum = 0.0;
            for (std::size_t k = 0; k < m->g_wg.size(); k++) { sum += std::fabs((double)m->g_wg.val[k]); }
            return sum;
        };
        const double gB = gate(base, true);
        const double gA = gate(a, true);
        const double gC = gate(c, false);
        std::printf("  复位 A = %.8e  参照 B = %.8e  不复位 C = %.8e\n", gA, gB, gC);
        std::printf("  |A-B| = %.3e, |C-B| = %.3e\n", std::fabs(gA - gB), std::fabs(gC - gB));
    }
    return 0;
}
