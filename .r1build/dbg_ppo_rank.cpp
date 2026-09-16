/*
   一次性 scratch: test_ppomcts 的 "rank ok" 断言在两种专家下分别是什么结果。

   背景: 该测试学一条固定映射 (目标分布 0.80/0.15/0.05 落在 target0/1/2), 学到
   p(target0) > 0.5 就停, 然后断言 **p0 > p1 > p2**。换成 TB 专家之后 p1 < p2,
   于是退出码变成 1。问题是: 这是"TB 骨干下动力学变差", 还是"这条断言在早停点
   本来就不稳"?

   这个程序把那个实验**原样**复现出来 (PPO 的 actor 损失是纯交叉熵, 与 critic 无关,
   所以只需要一个 actor 网络), 两种专家 x 多个种子各跑一遍, 数一数 p1 > p2 出现几次。

   一条等价性 (让实验跑得动): 该测试每轮喂 64x2 条**完全相同**的样本, 而
   `RMSProp(clipGrad=true)` 会把 dw 按范数归一 —— 128 条相同样本的梯度和, 归一化之后
   与 1 条的梯度方向**逐位相同**。所以每轮只做一次反向就够了。
*/
#include <cstdio>
#include <cmath>
#include <memory>
#include "net.hpp"
#include "sparse_moe.hpp"
#include "expert.hpp"
#include "layer.h"
#include "loss.h"
#include "util.hpp"

using namespace RL;

/* 与 rl/ppo.cpp 的 scaleLayerInit 同一套 (普通 iFcLayer 按 1/sqrt(fan_in) 缩) */
static void scaleLayerInit(Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        iFcLayer *fc = dynamic_cast<iFcLayer*>(net[i]);
        if (fc == nullptr) { continue; }
        const float fanIn = (float)(fc->inputDim > 1 ? fc->inputDim : 1);
        const float s = 1.0f / std::sqrt(fanIn);
        for (std::size_t k = 0; k < fc->w.size(); k++) { fc->w[k] *= s; }
        for (std::size_t k = 0; k < fc->b.size(); k++) { fc->b[k] *= s; }
    }
}

struct Result {
    double p0 = 0, p1 = 0, p2 = 0;
    int rounds = 0;
    bool ok = false;
    bool ranked = false;
};

static Result runOne(bool useTb, unsigned seed, int maxRounds = 40)
{
    Random::setSeed(seed);
    const int D = 1440;
    const int A = 8100;
    const int T0 = 100, T1 = 200, T2 = 300;

    Net::Layers ls;
    if (useTb) {
        ls.push_back(std::make_shared<SparseMoE<TransformerBlock<16, 360>, 4, 1> >(D, true, 0));
    } else {
        ls.push_back(std::make_shared<SparseMoE<MlpExpert, 8, 2> >(D, true, 64));
    }
    ls.push_back(Layer<Tanh>::_(D, 64, true, true));
    ls.push_back(Layer<Softmax>::_(64, A, true, true));
    Net net(ls);
    scaleLayerInit(net);

    Tensor s(D, 1);
    s.zero();
    for (int i = 0; i < D; i += 7) { s[i] = 1.0f; }

    Tensor target(A, 1);
    target.zero();
    target[T0] = 0.8f;
    target[T1] = 0.15f;
    target[T2] = 0.05f;

    Result r;
    const double p0Before = (double)net.forward(s)[T0];
    while (r.rounds < maxRounds && (double)net.forward(s)[T0] < 0.5) {
        Tensor &policy = net.forward(s);
        Tensor loss = Loss::CrossEntropy::df(policy, target);
        net.backward(s, loss);
        net.RMSProp(0.005f, 0.9f, 0.001f);
        r.rounds++;
    }
    Tensor &out = net.forward(s);
    r.p0 = (double)out[T0];
    r.p1 = (double)out[T1];
    r.p2 = (double)out[T2];
    r.ok = (r.p0 > 0.5) && (r.p0 > p0Before * 10.0);
    r.ranked = (r.p0 > r.p1) && (r.p1 > r.p2);
    return r;
}

int main()
{
    printf("test_ppomcts 的 rank 断言: 两种专家 x 6 个种子\n");
    printf("%-28s %-8s %8s %8s %8s %6s %6s %6s\n",
           "专家", "种子", "p0", "p1", "p2", "轮数", "learn", "rank");

    int tbRanked = 0, mlpRanked = 0, tbOk = 0, mlpOk = 0;
    for (int si = 0; si < 6; si++) {
        const unsigned seed = 20240913u + (unsigned)si * 7919u;
        Result tb = runOne(true, seed);
        Result mlp = runOne(false, seed);
        tbRanked += tb.ranked ? 1 : 0;
        mlpRanked += mlp.ranked ? 1 : 0;
        tbOk += tb.ok ? 1 : 0;
        mlpOk += mlp.ok ? 1 : 0;
        printf("%-28s %-8u %8.4f %8.4f %8.4f %6d %6d %6d\n",
               "TB<16,360> E=4 top-1", seed, tb.p0, tb.p1, tb.p2, tb.rounds,
               (int)tb.ok, (int)tb.ranked);
        printf("%-28s %-8u %8.4f %8.4f %8.4f %6d %6d %6d\n",
               "MlpExpert  E=8 top-2", seed, mlp.p0, mlp.p1, mlp.p2, mlp.rounds,
               (int)mlp.ok, (int)mlp.ranked);
    }
    printf("\n汇总: TB  rank 通过 %d/6, learn 通过 %d/6\n", tbRanked, tbOk);
    printf("      MLP rank 通过 %d/6, learn 通过 %d/6\n", mlpRanked, mlpOk);
    return 0;
}
