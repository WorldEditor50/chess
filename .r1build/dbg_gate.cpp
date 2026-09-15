/* scratch: 复现 part1 的门控统计, 顺便查 copyTo 到底漏了什么 */
#include <cstdio>
#include <cmath>
#include <vector>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/concat.hpp"
#include "rl/expert.hpp"
#include "rl/net.hpp"
#include "rl/loss.h"

using namespace RL;

static Tensor randTensor(std::size_t r, std::size_t c, unsigned seed, float amp = 0.5f)
{
    Tensor t(r, c);
    unsigned s = seed;
    for (std::size_t i = 0; i < t.size(); i++) {
        s = s * 1664525u + 1013904223u;
        t[i] = (float)((double)(s >> 8) / (double)(1u << 24) * 2.0 - 1.0) * amp;
    }
    return t;
}

int main()
{
    const int D = 24, U = 4, N = 4;
    auto lay = ScaledConcat<MlpExpert, N, U>::_(D, true, 4);
    lay->wg.zero();
    lay->bg.zero();
    for (int j = 0; j < D; j++) {
        lay->wg(0, j) = 1.0f;
    }
    lay->bg[0] = -5.0f;
    Tensor x = randTensor(D, 1, 4242, 1.0f);
    lay->forward(x, true);

    double mx = -1e30, mn = 1e30, sum = 0, H = 0;
    std::printf("gate = ");
    for (int i = 0; i < N*U; i++) {
        const double g = (double)lay->gate[i];
        std::printf("%.5f ", g);
        mx = std::fmax(mx, g);
        mn = std::fmin(mn, g);
        sum += g;
    }
    std::printf("\n  sum=%.6f  max/min=%.3f\n", sum, mx/mn);
    for (int i = 0; i < N*U; i++) {
        const double g = (double)lay->gate[i];
        if (g > 1e-12) H -= g*std::log(g);
    }
    std::printf("  手算 H=%.6f  exp(H)=%.4f\n", H, std::exp(H));
    std::printf("  接口: gateEffectiveCount=%.4f  sourceUnit=%zu\n",
                lay->gateEffectiveCount(), (std::size_t)lay->gate.size());

    std::vector<float> usage;
    lay->gateUsage(usage);
    std::printf("  gateUsage = ");
    for (std::size_t i = 0; i < usage.size(); i++) {
        std::printf("%.5f ", usage[i]);
    }
    std::printf("\n");

    /* ---- copyTo 检查: MlpExpert 有没有 copyTo ---- */
    MlpExpert a(8, 4, true), b(8, 4, true);
    a.l1.w = randTensor(4, 8, 111, 1.0f);
    double d0 = 0;
    for (std::size_t i = 0; i < a.l1.w.size(); i++) {
        d0 = std::fmax(d0, std::fabs((double)a.l1.w[i] - (double)b.l1.w[i]));
    }
    a.copyTo(&b);
    double d1 = 0;
    for (std::size_t i = 0; i < a.l1.w.size(); i++) {
        d1 = std::fmax(d1, std::fabs((double)a.l1.w[i] - (double)b.l1.w[i]));
    }
    std::printf("\nMlpExpert::copyTo: 复制前差 %.4f, 复制后差 %.4f  -> %s\n",
                d0, d1, (d1 == 0.0) ? "会复制" : "**没有复制** (iLayer::copyTo 是空实现)");

    /* Layer<Fn> 作专家时呢 */
    Layer<Gelu> la(8, 8, true, true), lb(8, 8, true, true);
    la.w = randTensor(8, 8, 222, 1.0f);
    la.copyTo(&lb);
    double d2 = 0;
    for (std::size_t i = 0; i < la.w.size(); i++) {
        d2 = std::fmax(d2, std::fabs((double)la.w[i] - (double)lb.w[i]));
    }
    std::printf("Layer<Gelu>::copyTo: 复制后差 %.4f -> %s\n",
                d2, (d2 == 0.0) ? "会复制" : "**没有复制**");

    /* softUpdateTo 是否也一样 */
    MlpExpert c(8, 4, true), dd(8, 4, true);
    dd.l1.w.zero();
    c.l1.w = randTensor(4, 8, 333, 1.0f);
    c.softUpdateTo(&dd, 0.5f);
    double d3 = 0, m3 = 0;
    for (std::size_t i = 0; i < c.l1.w.size(); i++) {
        d3 = std::fmax(d3, std::fabs((double)c.l1.w[i] - (double)dd.l1.w[i]));
        m3 = std::fmax(m3, std::fabs((double)dd.l1.w[i]));
    }
    std::printf("MlpExpert::softUpdateTo: 目标 |w|max=%.4f, 与源最大差=%.4f -> %s\n",
                m3, d3, (m3 > 0.0) ? "动了" : "**目标没动** (空实现)");
    return 0;
}
