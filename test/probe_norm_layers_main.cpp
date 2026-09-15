/*
 * 独立探针 (probe), **故意不注册进 CMakeLists.txt** —— 那里的测试源列表是显式的,
 * 所以本文件不会被自动编译, 也不进 ctest。编译方法 (需要已构建的 RL_CORE.lib) 见
 * docs/tanh_norm_analysis.md §12。用法: 无参数, 一次跑完 A/B/C 三节。
 *
 * TanhNorm vs RMSNorm: 三条可证伪的断言
 *   A. 有限差分梯度检查: TanhNorm::backward 是否给出真梯度? RMSNorm::backward 呢?
 *   B. 输入整体缩放不变性: o(sx) 与 g.w(sx) 对两个层分别如何变化?
 *   C. clipGrad (= dw/‖dw‖) 下, RMSNorm 与裸 Linear 的更新方向是否逐位相同?
 */
#include <cstdio>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include "rl/tensor.hpp"
#include "rl/layer.h"

using namespace RL;

static unsigned s_ = 20260101u;
static float frand(float lo, float hi)
{
    s_ = s_*1664525u + 1013904223u;
    float u = (float)((s_ >> 8) & 0xFFFFFFu)/(float)0x1000000u;
    return lo + u*(hi - lo);
}
static void fill(Tensor &t, float lo, float hi)
{
    for (std::size_t i = 0; i < t.totalSize; i++) t[i] = frand(lo, hi);
}

/* L(w) = Σ_i c_i·o_i ,  c = dL/do  (与层的 e 契约一致) */
template <class L>
static double lossOf(L &layer, const Tensor &x, const Tensor &c)
{
    Tensor &o = layer.forward(x, true);
    double s = 0;
    for (std::size_t i = 0; i < o.totalSize; i++) s += (double)c[i]*(double)o[i];
    return s;
}

/* 中心差分 vs 解析 g.w, 返回最大相对误差 */
template <class L>
static double gradCheck(L &layer, const Tensor &x, const Tensor &c,
                        double eps, const char *tag)
{
    layer.forward(x, true);
    layer.e = c;
    Tensor ei(layer.inputDim, 1);
    layer.g.zero();
    layer.backward(x, ei);
    Tensor ga = layer.g.w;

    double worst = 0, wa = 0, wn = 0;
    std::size_t wi = 0, wj = 0;
    for (std::size_t i = 0; i < layer.outputDim; i++) {
        for (std::size_t j = 0; j < layer.inputDim; j++) {
            float w0 = layer.w(i, j);
            layer.w(i, j) = (float)(w0 + eps);
            double lp = lossOf(layer, x, c);
            layer.w(i, j) = (float)(w0 - eps);
            double lm = lossOf(layer, x, c);
            layer.w(i, j) = w0;
            double num = (lp - lm)/(2.0*eps);
            double an  = ga(i, j);
            double den = std::max(1.0, std::max(std::fabs(num), std::fabs(an)));
            double rel = std::fabs(num - an)/den;
            if (rel > worst) { worst = rel; wa = an; wn = num; wi = i; wj = j; }
        }
    }
    std::printf("  %-28s max rel err = %.3e   (worst @ w(%zu,%zu): analytic=%+.6f  numeric=%+.6f)\n",
                tag, worst, wi, wj, wa, wn);
    return worst;
}

struct FnLin { static float f(float x){return x;} static float df(float){return 1;} };
struct FnSig { static float f(float x){return 1.0f/(1.0f + std::exp(-1.702f*x));}
              static float df(float y){return 1.702f*y*(1 - y);} };

static double norm2(const Tensor &t)
{
    double s = 0;
    for (std::size_t i = 0; i < t.totalSize; i++) s += (double)t[i]*(double)t[i];
    return std::sqrt(s);
}
static double maxAbs(const Tensor &a, const Tensor &b)
{
    double m = 0;
    for (std::size_t i = 0; i < a.totalSize; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}
static void unit(Tensor &t)
{
    double n = norm2(t) + 1e-8;
    for (std::size_t i = 0; i < t.totalSize; i++) t[i] = (float)((double)t[i]/n);
}

/* 取解析 g.w: forward + backward, 返回副本 */
template <class L>
static Tensor rawGrad(L &layer, const Tensor &x, const Tensor &c)
{
    layer.forward(x, true);
    layer.e = c;
    Tensor ei(layer.inputDim, 1);
    layer.g.zero();
    layer.backward(x, ei);
    return layer.g.w;
}

int main()
{
    const std::size_t K = 5, D = 8;          /* inputDim, outputDim */
    const double eps = 1e-2;

    Tensor x(K, 1);
    fill(x, -1.0f, 1.0f);
    Tensor c(D, 1);
    fill(c, -0.5f, 0.5f);

    std::printf("=== A. 有限差分梯度检查 (中心差分, eps=%.0e, x∈[-1,1], c∈[-.5,.5]) ===\n", eps);

    auto tanhLin = TanhNorm<FnLin>::_(K, D, false, true);
    tanhLin->w = Tensor(D, K); fill(tanhLin->w, -1.0f, 1.0f);
    gradCheck(*tanhLin, x, c, eps, "TanhNorm<Linear>");

    auto tanhSig = TanhNorm<FnSig>::_(K, D, false, true);
    tanhSig->w = Tensor(D, K); fill(tanhSig->w, -1.0f, 1.0f);
    gradCheck(*tanhSig, x, c, eps, "TanhNorm<Sigmoid>");

    /* 饱和区: w 放大 6 倍, tanh 基本饱和 */
    auto tanhSat = TanhNorm<FnSig>::_(K, D, false, true);
    tanhSat->w = Tensor(D, K); fill(tanhSat->w, -6.0f, 6.0f);
    gradCheck(*tanhSat, x, c, eps, "TanhNorm<Sigmoid> (saturated)");

    auto rmsLin = RMSNorm<FnLin>::_(K, D, false, true);
    rmsLin->w = Tensor(D, K); fill(rmsLin->w, -1.0f, 1.0f);
    gradCheck(*rmsLin, x, c, eps, "RMSNorm<Linear> (as coded)");

    auto rmsSig = RMSNorm<FnSig>::_(K, D, false, true);
    rmsSig->w = Tensor(D, K); fill(rmsSig->w, -1.0f, 1.0f);
    gradCheck(*rmsSig, x, c, eps, "RMSNorm<Sigmoid> (as coded)");

    /* RMSNorm 带上缺失的径向项 (即真 RMSNorm 的雅可比) */
    {
        Tensor &o = rmsLin->forward(x, true);
        double S = 0;
        for (std::size_t i = 0; i < rmsLin->op.totalSize; i++) S += (double)rmsLin->op[i]*(double)rmsLin->op[i];
        double gamma = 1.0/std::sqrt(S/(double)rmsLin->op.totalSize + 1e-9);
        double dot = 0;
        for (std::size_t i = 0; i < D; i++) dot += (double)rmsLin->op[i]*(double)c[i];
        Tensor dOp(D, 1);
        for (std::size_t i = 0; i < D; i++)
            dOp[i] = (float)(gamma*((double)c[i] - (double)rmsLin->op[i]*dot/S));
        double worst = 0;
        for (std::size_t i = 0; i < D; i++) {
            for (std::size_t j = 0; j < K; j++) {
                float w0 = rmsLin->w(i, j);
                rmsLin->w(i, j) = (float)(w0 + eps);
                double lp = lossOf(*rmsLin, x, c);
                rmsLin->w(i, j) = (float)(w0 - eps);
                double lm = lossOf(*rmsLin, x, c);
                rmsLin->w(i, j) = w0;
                double an = (double)dOp[i]*(double)x[j];
                double num = (lp - lm)/(2.0*eps);
                double den = std::max(1.0, std::max(std::fabs(num), std::fabs(an)));
                worst = std::max(worst, std::fabs(num - an)/den);
            }
        }
        std::printf("  %-28s max rel err = %.3e   <-- 补上 (I−ôôᵀ) 投影项之后\n",
                    "RMSNorm<Linear> + radial term", worst);
    }

    std::printf("\n=== B. 输入整体缩放 x -> 4x ===\n");
    {
        Tensor x4(K, 1);
        for (std::size_t i = 0; i < K; i++) x4[i] = 4.0f*x[i];

        auto tn = TanhNorm<FnLin>::_(K, D, false, true);
        tn->w = Tensor(D, K); fill(tn->w, -1.0f, 1.0f);
        auto rn = RMSNorm<FnLin>::_(K, D, false, true);
        rn->w = tn->w;

        Tensor &ot1 = tn->forward(x, true);  Tensor o1 = ot1;
        Tensor &ot2 = tn->forward(x4, true); Tensor o2 = ot2;
        Tensor &rt1 = rn->forward(x, true);  Tensor r1 = rt1;
        Tensor &rt2 = rn->forward(x4, true); Tensor r2 = rt2;

        Tensor g1 = rawGrad(*tn, x, c);
        Tensor g2 = rawGrad(*tn, x4, c);
        Tensor h1 = rawGrad(*rn, x, c);
        Tensor h2 = rawGrad(*rn, x4, c);

        std::printf("  TanhNorm: |o(4x)-o(x)|max = %.6f   |g.w(4x)-g.w(x)|max = %.6f\n",
                    maxAbs(o1, o2), maxAbs(g1, g2));
        std::printf("  RMSNorm : |o(4x)-o(x)|max = %.3e   |g.w(4x)-g.w(x)|max = %.3e\n",
                    maxAbs(r1, r2), maxAbs(h1, h2));

        Tensor n1 = g1, n2 = g2, m1 = h1, m2 = h2;
        unit(n1); unit(n2); unit(m1); unit(m2);
        std::printf("  clipGrad 之后的方向差: TanhNorm |ĝ(4x)-ĝ(x)|max = %.6f , RMSNorm = %.3e\n",
                    maxAbs(n1, n2), maxAbs(m1, m2));
    }

    std::printf("\n=== C. clipGrad(=dw/‖dw‖) 之下 RMSNorm<Linear> vs 裸 Linear ===\n");
    {
        auto lin = std::make_shared<iFcLayer>(K, D, false, true);
        lin->w = Tensor(D, K); fill(lin->w, -1.0f, 1.0f);
        auto rn = RMSNorm<FnLin>::_(K, D, false, true);
        rn->w = lin->w;

        Tensor gl = rawGrad(*lin, x, c);
        Tensor gr = rawGrad(*rn, x, c);
        double rawT = norm2(gl), rawR = norm2(gr);
        Tensor ul = gl, ur = gr;
        unit(ul); unit(ur);
        std::printf("  ‖g.w‖: Linear = %.6f , RMSNorm = %.6f  (比值 %.6f)\n", rawT, rawR, rawR/rawT);
        std::printf("  单位化后的更新方向差 |û_lin - û_rms|max = %.3e\n", maxAbs(ul, ur));
    }
    return 0;
}
