/*
 * 独立探针 (probe), **故意不注册进 CMakeLists.txt** —— 那里的测试源列表是显式的,
 * 所以本文件不会被自动编译, 也不进 ctest。编译方法 (需要已构建的 RL_CORE.lib) 见
 * docs/tanh_norm_analysis.md §12。用法: 参数 1 / 2 / 3 / 5 / 6 / 7 / 8 分节跑,
 * 不带参数 = 全跑 (P1 的性能基准约需 1 分钟)。
 *
 * TanhNorm: what is there left to optimize?
 *
 * P1 speed  : current TanhNorm vs an otherwise-identical layer with
 *             (a) fused activation (no `o1 *= r` pass, no RL::tanh temporary)
 *             (b) preallocated dL/dy (current code heap-allocates 2 tensors per backward)
 *             (c) no redundant o1.zero() in backward
 *             Same MM:: calls, same math -> outputs verified equal.
 *
 * P2 init   : the inner scale is the ONLY knob that sets where tanh operates, and it
 *             is hard-wired to r = 1-1/D ~ 1 (i.e. no knob at all). Sweep the weight
 *             init scale instead and measure saturation / per-unit variation / dQ/dx.
 *
 * P3 Fn     : TanhNorm<Sigmoid> vs TanhNorm<Linear> - does the extra sigmoid cost
 *             usable range and slope for the downstream linear head?
 */
#include <cstdio>
#include <cmath>
#include <cstddef>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/optimize.h"

using namespace RL;

static std::mt19937 g_rng(1234567u);
static float urand(float lo, float hi)
{
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g_rng);
}
static void fillU(Tensor &t, float lo, float hi)
{
    for (std::size_t i = 0; i < t.totalSize; i++) t[i] = urand(lo, hi);
}
static double nrm2(const Tensor &t)
{
    double s = 0;
    for (std::size_t i = 0; i < t.totalSize; i++) s += (double)t[i]*(double)t[i];
    return std::sqrt(s);
}
static double maxAbsDiff(const Tensor &a, const Tensor &b)
{
    double m = 0;
    for (std::size_t i = 0; i < a.totalSize; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}
static double nowNs()
{
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

/* chess-like sparse state: 90 dims, 32 non-zero, value k/7 */
static void chessLikeState(Tensor &x)
{
    x.zero();
    for (int k = 0; k < 32; k++) {
        int idx = (int)urand(0.0f, 89.999f);
        int v = (int)urand(-7.0f, 7.99f);
        x[(std::size_t)idx] = (float)v/7.0f;
    }
}

/* ---------------- optimized-for-comparison TanhNorm ----------------
   Two levels, both keeping the same MM:: calls and the same math:
     fused=false : original loop structure, but dL/dy preallocated and the
                   redundant o1.zero() in backward removed  -> isolates the
                   two heap allocations per backward.
     fused=true  : additionally folds `o1 *= r` into the tanh argument and
                   writes straight into o2 (no RL::tanh temporary).
                   NOTE: o2 must still be stored - backward needs 1-o2^2. */
template <typename Fn>
class FastTanhNorm : public iFcLayer
{
public:
    float r;
    bool fused;
    Tensor o1, o2, dL, dy;
    FastTanhNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_, bool fused_)
        : iFcLayer(inputDim, outputDim, bias_, withGrad_), fused(fused_)
    {
        o1 = Tensor(outputDim, 1);
        o2 = Tensor(outputDim, 1);
        dL = Tensor(outputDim, 1);
        dy = Tensor(outputDim, 1);
        r = 1.0f - 1.0f/float(outputDim);
    }
    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        o1.zero();
        Tensor::MM::ikkj(o1, w, x);
        if (fused) {
            const float rr = r;
            if (bias) {
                for (std::size_t i = 0; i < o.totalSize; i++) {
                    o2[i] = std::tanh(rr*o1[i]);
                    o[i] = Fn::f(o2[i] + b[i]);
                }
            } else {
                for (std::size_t i = 0; i < o.totalSize; i++) {
                    o2[i] = std::tanh(rr*o1[i]);
                    o[i] = Fn::f(o2[i]);
                }
            }
        } else {
            o1 *= r;
            o2 = RL::tanh(o1);
            if (bias) {
                for (std::size_t i = 0; i < o.totalSize; i++) o[i] = Fn::f(o2[i] + b[i]);
            } else {
                for (std::size_t i = 0; i < o.totalSize; i++) o[i] = Fn::f(o2[i]);
            }
        }
        return o;
    }
    void backward(const Tensor &x, Tensor &ei) override
    {
        for (std::size_t i = 0; i < e.totalSize; i++) {
            float d1 = Fn::df(o[i])*e[i];
            dy[i] = d1;
            dL[i] = r*(1.0f - o2[i]*o2[i])*d1;
        }
        Tensor::MM::kikj(ei, w, dL);
        Tensor::MM::ikjk(g.w, dL, x);
        if (bias) g.b += dy;
        e.zero();
        /* o1 not zeroed here: forward() zeroes it anyway */
    }
};

/* interleaved rounds -> take the min of each, so drift/thermal noise cancels */
template <class L>
static void bench(L &layer, const Tensor &x, const Tensor &c, int iters, int rounds,
                  double &fwdNs, double &fwdBwdNs, double &checksum)
{
    Tensor ei(layer.inputDim, 1);
    for (int i = 0; i < 3000; i++) {
        Tensor &o = layer.forward(x, true);
        layer.e = c;
        layer.g.zero();
        layer.backward(x, ei);
        checksum += (double)o[0] + ei[0];
    }
    fwdNs = 1e30; fwdBwdNs = 1e30;
    for (int rr = 0; rr < rounds; rr++) {
        double t0 = nowNs();
        for (int i = 0; i < iters; i++) {
            Tensor &o = layer.forward(x, true);
            checksum += (double)o[0];
        }
        double t1 = nowNs();
        for (int i = 0; i < iters; i++) {
            Tensor &o = layer.forward(x, true);
            layer.e = c;
            layer.g.zero();
            layer.backward(x, ei);
            checksum += (double)o[0] + ei[0];
        }
        double t2 = nowNs();
        fwdNs = std::min(fwdNs, (t1 - t0)/iters);
        fwdBwdNs = std::min(fwdBwdNs, (t2 - t1)/iters);
    }
}

static void p1()
{
    std::printf("=== P1 speed: TanhNorm as-is vs preallocated dL/dy vs + fused activation ===\n");
    struct Cfg { std::size_t in, out; const char *tag; int iters, rounds; } cfgs[] = {
        {90, 64, "DQN 90->64 ", 60000, 9},
        {90, 256, "90->256    ", 20000, 5},
        {256, 256, "256->256   ", 6000, 3},
    };
    std::printf("  %-11s %9s %9s %9s | %9s %9s %9s | %s\n", "shape",
                "fwd cur", "fwd +pre", "fwd fus", "fwd+bwd cur", "+pre", "fused", "verdict");
    for (int ci = 0; ci < 3; ci++) {
        std::size_t I = cfgs[ci].in, O = cfgs[ci].out;
        const int iters = cfgs[ci].iters, rounds = cfgs[ci].rounds;
        Tensor x(I, 1), c(O, 1);
        fillU(x, -1, 1);
        fillU(c, -0.5f, 0.5f);

        Random::setSeed(99);
        auto cur  = TanhNorm<Sigmoid>::_(I, O, true, true);
        auto mid  = std::make_shared<FastTanhNorm<Sigmoid> >(I, O, true, true, false);
        auto fast = std::make_shared<FastTanhNorm<Sigmoid> >(I, O, true, true, true);
        mid->w = cur->w;  mid->b = cur->b;
        fast->w = cur->w; fast->b = cur->b;

        /* correctness: outputs AND gradients must match the original */
        Tensor &oc = cur->forward(x, true);
        Tensor ocCopy = oc;
        cur->e = c; cur->g.zero();
        Tensor eic(I, 1);
        cur->backward(x, eic);
        Tensor gwc = cur->g.w;

        Tensor &om = mid->forward(x, true);
        double dOutM = maxAbsDiff(ocCopy, om);
        mid->e = c; mid->g.zero();
        Tensor eim(I, 1);
        mid->backward(x, eim);
        double dGwM = maxAbsDiff(gwc, mid->g.w);

        Tensor &of = fast->forward(x, true);
        double dOutF = maxAbsDiff(ocCopy, of);
        fast->e = c; fast->g.zero();
        Tensor eif(I, 1);
        fast->backward(x, eif);
        double dGwF = maxAbsDiff(gwc, fast->g.w);

        double f1, fb1, f2, fb2, f3, fb3, sum = 0;
        bench(*cur,  x, c, iters, rounds, f1, fb1, sum);
        bench(*mid,  x, c, iters, rounds, f2, fb2, sum);
        bench(*fast, x, c, iters, rounds, f3, fb3, sum);
        std::printf("  %-11s %9.0f %9.0f %9.0f | %11.0f %9.0f %9.0f | fwd %+.0f%%  fwd+bwd %+.0f%%"
                    "  (dOut %.0e/%.0e dGw %.0e/%.0e)\n",
                    cfgs[ci].tag, f1, f2, f3, fb1, fb2, fb3,
                    100.0*(f3 - f1)/f1, 100.0*(fb3 - fb1)/fb1, dOutM, dOutF, dGwM, dGwF);
        if (ci == 0) std::printf("    checksum %.3f  (backward share: cur %.0f%%, fused %.0f%%)\n",
                                 sum, 100.0*(fb1 - f1)/fb1, 100.0*(fb3 - f3)/fb3);
    }
}

/* ---------------- P2 / P3 statistics ---------------- */
struct HStat {
    double sat90, sat99, perUnitMin, perUnitMed, perUnitMax;
    int dead, units;
    double hNormMean, hNormSpread;
    double dQdx, qMin, qMax;
};

template <class LayerT>
static HStat stats(std::shared_ptr<LayerT> hidden, double wScale, int nState)
{
    HStat s;
    s.units = (int)hidden->outputDim;
    std::vector<Tensor> st;
    for (int i = 0; i < nState; i++) { Tensor x(90, 1); chessLikeState(x); st.push_back(x); }

    std::vector<std::vector<double> > vals(hidden->outputDim);
    long n90 = 0, n99 = 0, tot = 0;
    double hSum = 0, hMin = 1e30, hMax = -1e30;
    for (int i = 0; i < nState; i++) {
        hidden->forward(st[i], true);
        const Tensor &o2 = hidden->o2;          /* pre-Fn tanh output (saturation) */
        for (std::size_t j = 0; j < hidden->outputDim; j++) {
            double v = std::fabs((double)o2[j]);
            if (v > 0.9) n90++;
            if (v > 0.99) n99++;
            tot++;
            vals[j].push_back((double)hidden->o[j]);   /* layer OUTPUT: what the head sees */
        }
        double hn = nrm2(hidden->o);
        hSum += hn; hMin = std::min(hMin, hn); hMax = std::max(hMax, hn);
    }
    s.sat90 = (double)n90/tot;
    s.sat99 = (double)n99/tot;
    s.hNormMean = hSum/nState;
    s.hNormSpread = hMax/(hSum/nState + 1e-12);

    std::vector<double> pu;
    s.dead = 0;
    for (std::size_t j = 0; j < hidden->outputDim; j++) {
        double m = 0;
        for (std::size_t k = 0; k < vals[j].size(); k++) m += vals[j][k];
        m /= vals[j].size();
        double v = 0;
        for (std::size_t k = 0; k < vals[j].size(); k++) v += (vals[j][k] - m)*(vals[j][k] - m);
        v = std::sqrt(v/vals[j].size());
        pu.push_back(v);
        if (v < 0.02) s.dead++;
    }
    std::sort(pu.begin(), pu.end());
    s.perUnitMin = pu.front();
    s.perUnitMed = pu[pu.size()/2];
    s.perUnitMax = pu.back();

    /* dQ/dx through hidden + linear head(64->128) */
    Random::setSeed(4321);
    auto head = Layer<Linear>::_(hidden->outputDim, 128, true, true);
    Net net(hidden, head);
    double dx = 0, qmin = 1e30, qmax = -1e30;
    for (int k = 0; k < 16; k++) {
        Tensor &q = net.forward(st[k], true);
        for (std::size_t a = 0; a < q.totalSize; a++) { qmin = std::min(qmin, (double)q[a]); qmax = std::max(qmax, (double)q[a]); }
        Tensor onehot(128, 1); onehot.zero(); onehot[(std::size_t)(k*7 % 128)] = 1.0f;
        net.backward(st[k], onehot);
        dx += nrm2(net.inputGrad);
    }
    s.dQdx = dx/16;
    s.qMin = qmin; s.qMax = qmax;
    (void)wScale;
    return s;
}

template <class LayerT>
static std::shared_ptr<LayerT> makeWithScale(std::size_t I, std::size_t O, double a, unsigned seed)
{
    Random::setSeed(seed);
    auto L = LayerT::_(I, O, true, true);
    L->w = Tensor(O, I);
    fillU(L->w, (float)-a, (float)a);
    L->b = Tensor(O, 1);
    fillU(L->b, -1.0f, 1.0f);
    return L;
}

static void p2()
{
    std::printf("\n=== P2 inner scale: the only knob on tanh's operating point is hard-wired to r=1-1/D ===\n");
    std::printf("  (hidden 90->64, TanhNorm<Linear>; sat = fraction of |tanh| > 0.9 / > 0.99)\n");
    std::printf("  %-10s %7s %7s %9s %9s %9s %6s %8s %8s %16s\n",
                "w ~ U(-a,a)", "sat .9", "sat .99", "pu_min", "pu_med", "pu_max", "dead",
                "|h|spr", "dQ/dx", "Q[min..max]");
    double as[4] = {0.25, 0.5, 1.0, 2.0};
    for (int i = 0; i < 4; i++) {
        auto L = makeWithScale<TanhNorm<Linear> >(90, 64, as[i], 777);
        HStat s = stats(L, as[i], 256);
        std::printf("  %-10.2f %7.3f %7.3f %9.4f %9.4f %9.4f %6d %8.2f %8.3f  [%6.2f,%6.2f]\n",
                    as[i], s.sat90, s.sat99, s.perUnitMin, s.perUnitMed, s.perUnitMax,
                    s.dead, s.hNormSpread, s.dQdx, s.qMin, s.qMax);
    }
    std::printf("    (repo init is a=1.0 + fan-in 90 => std(Wx)~1.9; a=0.5 lands near std(Wx)~1)\n");
}

static void p3()
{
    std::printf("\n=== P3 does the outer Sigmoid cost range? (a=1.0, repo init) ===\n");
    std::printf("  %-20s %8s %8s %8s %6s %8s %16s\n",
                "layer", "pu_min", "pu_med", "pu_max", "dead", "dQ/dx", "Q[min..max]");
    {
        auto L = makeWithScale<TanhNorm<Linear> >(90, 64, 1.0, 777);
        HStat s = stats(L, 1.0, 256);
        std::printf("  %-20s %8.4f %8.4f %8.4f %6d %8.3f  [%6.2f,%6.2f]\n",
                    "TanhNorm<Linear>", s.perUnitMin, s.perUnitMed, s.perUnitMax, s.dead, s.dQdx, s.qMin, s.qMax);
    }
    {
        auto L = makeWithScale<TanhNorm<Sigmoid> >(90, 64, 1.0, 777);
        HStat s = stats(L, 1.0, 256);
        std::printf("  %-20s %8.4f %8.4f %8.4f %6d %8.3f  [%6.2f,%6.2f]\n",
                    "TanhNorm<Sigmoid>", s.perUnitMin, s.perUnitMed, s.perUnitMax, s.dead, s.dQdx, s.qMin, s.qMax);
    }
    std::printf("    (pu_* = per-unit std of the layer OUTPUT across states: how much each unit can say)\n");
}

/* ---- P5: what does r = 1-1/D actually buy? ---- */
static void p5()
{
    std::printf("\n=== P5 effect size of r = 1-1/D (hidden 90->64, repo init a=1.0) ===\n");
    struct R { double r; const char *tag; } rs[] = {
        {1.0,             "r = 1 (no factor)"},
        {1.0 - 1.0/64.0,  "r = 1-1/64 (repo)"},
        {1.0 - 1.0/8.0,   "r = 1-1/8"},
        {0.5,             "r = 0.5 (=1/sqrt(fanin))"},
    };
    std::printf("  %-22s %8s %8s %9s %9s %11s %16s\n",
                "r", "sat .9", "sat .99", "pu_med", "dQ/dx", "vs r=1", "Q[min..max]");
    double baseDx = 0;
    for (int i = 0; i < 4; i++) {
        auto L = makeWithScale<TanhNorm<Linear> >(90, 64, 1.0, 777);
        L->r = (float)rs[i].r;
        HStat s = stats(L, 1.0, 256);
        if (i == 0) baseDx = s.dQdx;
        std::printf("  %-22s %8.3f %8.3f %9.4f %9.3f %+10.2f%%  [%6.2f,%6.2f]\n",
                    rs[i].tag, s.sat90, s.sat99, s.perUnitMed, s.dQdx,
                    100.0*(s.dQdx - baseDx)/baseDx, s.qMin, s.qMax);
    }

    /* where the real smoothing lives: distribution of the backward factor 1-o2^2 */
    {
        auto L = makeWithScale<TanhNorm<Linear> >(90, 64, 1.0, 777);
        std::vector<double> f;
        for (int i = 0; i < 256; i++) {
            Tensor x(90, 1);
            chessLikeState(x);
            L->forward(x, true);
            for (std::size_t j = 0; j < L->outputDim; j++)
                f.push_back(1.0 - (double)L->o2[j]*(double)L->o2[j]);
        }
        std::sort(f.begin(), f.end());
        std::printf("  backward factor (1-o2^2) over %zu (unit,state) pairs:\n", f.size());
        std::printf("    min %.4f  p1 %.4f  p10 %.4f  median %.4f  mean %.4f  p90 %.4f  max %.4f\n",
                    f.front(), f[f.size()/100], f[f.size()/10], f[f.size()/2],
                    [&]{ double s = 0; for (double v : f) s += v; return s/f.size(); }(),
                    f[f.size()*9/10], f.back());
        std::printf("    fraction of pairs with factor < 0.2 : %.3f\n",
                    (double)std::count_if(f.begin(), f.end(), [](double v){ return v < 0.2; })/f.size());
        std::printf("    (r itself only moves this whole distribution by %.2f%%, because it\n"
                    "     scales the tanh ARGUMENT; the spread comes from tanh, not from r.)\n",
                    100.0*(1.0/rs[1].r - 1.0));
    }
}

/* ---- P6: is "move r into the init" behaviour preserving? ---- */
static void p6()
{
    std::printf("\n=== P6 migration check: o1 *= r  vs  W <- r*W at init ===\n");
    Tensor x(90, 1), c(64, 1);
    chessLikeState(x);
    fillU(c, -0.5f, 0.5f);

    Random::setSeed(2024);
    auto oldL = TanhNorm<Sigmoid>::_(90, 64, true, true);      /* r = 1-1/64, applied in forward */
    Random::setSeed(2024);
    auto newL = TanhNorm<Sigmoid>::_(90, 64, true, true);
    float r = oldL->r;
    newL->r = 1.0f;                                            /* r folded into W instead */
    for (std::size_t i = 0; i < newL->w.size(); i++) newL->w[i] *= r;

    Tensor &o1 = oldL->forward(x, true);   Tensor oc = o1;
    oldL->e = c; oldL->g.zero(); Tensor e1(90, 1); oldL->backward(x, e1);
    Tensor gwOld = oldL->g.w;
    Tensor gbOld = oldL->g.b;

    Tensor &o2 = newL->forward(x, true);
    newL->e = c; newL->g.zero(); Tensor e2(90, 1); newL->backward(x, e2);
    Tensor gwNew = newL->g.w;
    Tensor gbNew = newL->g.b;

    /* gradient w.r.t. the ORIGINAL W is g_new * r (chain rule through W' = r*W) */
    Tensor gwScaled = gwNew;
    for (std::size_t i = 0; i < gwScaled.size(); i++) gwScaled[i] *= r;

    /* W' = r*W and dL/dW' = dL/dW / r  =>  dL/dx = W'^T dL' = W^T dL : INVARIANT in x */
    std::printf("  r = %.6f\n", (double)r);
    std::printf("  forward  max|o_new - o_old|            = %.3e   (|o| ~ %.3f)\n",
                maxAbsDiff(oc, o2), nrm2(oc)/std::sqrt((double)oc.size()));
    std::printf("  bias grad max|gb_new - gb_old|         = %.3e\n", maxAbsDiff(gbOld, gbNew));
    std::printf("  dL/dx    max|e_new - e_old|            = %.3e   (|e| ~ %.4f, ratio |e_new|/|e_old| = %.6f)\n",
                maxAbsDiff(e1, e2), nrm2(e1), nrm2(e2)/(nrm2(e1) + 1e-30));
    std::printf("  dL/dW    max|r*gw_new - gw_old|        = %.3e   (|gw| ~ %.4f)\n",
                maxAbsDiff(gwOld, gwScaled), nrm2(gwOld));
    std::printf("  => folding r into W is the same layer and the same dL/dW up to float reassociation\n");
}

/* ---- P7: tanh's gate = f(unit) x f(state). RMSProp absorbs the f(unit) part.
       Decompose so we can say WHICH part of the operation can do work. ---- */
static void p7()
{
    std::printf("\n=== P7 the backward gate 1-o2^2: which part survives a scale-free optimizer? ===\n");
    auto L = makeWithScale<TanhNorm<Linear> >(90, 64, 1.0, 777);
    const int S = 256;
    const std::size_t U = L->outputDim;
    std::vector<std::vector<double> > g(U);
    std::vector<double> all;
    for (int s = 0; s < S; s++) {
        Tensor x(90, 1);
        chessLikeState(x);
        L->forward(x, true);
        for (std::size_t j = 0; j < U; j++) {
            const double o2 = (double)L->o2[j];
            const double v = 1.0 - o2*o2;            /* == tanh'(r*op), the exact backward factor (r=1) */
            g[j].push_back(v);
            all.push_back(v);
        }
    }
    std::sort(all.begin(), all.end());
    std::printf("  gate over %zu (unit,state) pairs: p1 %.4f  p10 %.4f  median %.4f  p90 %.4f  max %.4f\n",
                all.size(), all[all.size()/100], all[all.size()/10], all[all.size()/2],
                all[all.size()*9/10], all.back());

    /* between-unit (slowly varying, gets absorbed by RMSProp's per-coordinate sqrt(v))
       vs within-unit (state to state, the part that can still change an update direction) */
    double between = 0, within = 0;
    std::vector<double> um(U), rel(U);
    for (std::size_t j = 0; j < U; j++) {
        double m = 0;
        for (int s = 0; s < S; s++) m += g[j][s]/S;
        double v = 0;
        for (int s = 0; s < S; s++) v += (g[j][s] - m)*(g[j][s] - m)/S;
        um[j] = m;
        rel[j] = std::sqrt(v)/(m + 1e-12);
        within += v/U;
    }
    double gm = 0;
    for (std::size_t j = 0; j < U; j++) gm += um[j]/U;
    for (std::size_t j = 0; j < U; j++) between += (um[j] - gm)*(um[j] - gm)/U;
    std::sort(um.begin(), um.end());
    std::sort(rel.begin(), rel.end());
    std::printf("  per-unit mean gate : min %.4f  median %.4f  max %.4f\n", um.front(), um[U/2], um.back());
    std::printf("  per-unit state fluctuation std/mean : min %.3f  median %.3f  max %.3f\n",
                rel.front(), rel[U/2], rel.back());
    std::printf("  variance decomposition: between-unit %.1f%%  |  within-unit(state) %.1f%%\n",
                100.0*between/(between + within), 100.0*within/(between + within));
    std::printf("  (the slowly-varying between-unit part is what sqrt(v) in RMSProp absorbs;\n"
                "   only the state-to-state part can still reweight one sample against another)\n");
}

/* ---- P8: does putting tanh in front of sigmoid keep sigmoid "active"?
       Measure the local gain d(o_j)/d((Wx)_j) and the sigmoid argument. ---- */
struct GainStat {
    double p1, p10, med, p90, mx;
    double fracDead, fracLow;      /* gain < 0.02  /  < 0.1 */
    double argBig2, argBig4;       /* |sigmoid argument| > 2 / > 4 */
    double deadUnits;              /* fraction of units whose output std across states < 0.02 */
};

static void collectGain(const char *tag, TanhNorm<Linear> *t,
                        const Layer<Linear> *dummy1, const Layer<Sigmoid> *dummy2,
                        GainStat &out, bool hasSigmoid);

static GainStat gainOf(int kind, int nState)
{
    /* kind: 0 = TanhNorm<Sigmoid> (repo), 1 = TanhNorm<Linear>,
             2 = Layer<Sigmoid>, 3 = Layer<Tanh>, 4 = Layer<Linear>, 5 = Layer<Relu> */
    std::shared_ptr<iFcLayer> L;
    Random::setSeed(777);
    switch (kind) {
        case 0: L = TanhNorm<Sigmoid>::_(90, 64, true, true); break;
        case 1: L = TanhNorm<Linear>::_(90, 64, true, true);  break;
        case 2: L = Layer<Sigmoid>::_(90, 64, true, true);    break;
        case 3: L = Layer<Tanh>::_(90, 64, true, true);       break;
        case 4: L = Layer<Linear>::_(90, 64, true, true);     break;
        default: L = Layer<Relu>::_(90, 64, true, true);      break;
    }
    L->w = Tensor(64, 90); fillU(L->w, -1.0f, 1.0f);
    L->b = Tensor(64, 1);  fillU(L->b, -1.0f, 1.0f);

    TanhNorm<Linear> *tn = dynamic_cast<TanhNorm<Linear>*>(L.get());
    TanhNorm<Sigmoid> *tns = dynamic_cast<TanhNorm<Sigmoid>*>(L.get());
    double r = tns ? (double)tns->r : (tn ? (double)tn->r : 1.0);

    std::vector<double> g;
    std::vector<std::vector<double> > outs(64);
    double n2 = 0, n4 = 0, tot = 0;
    for (int s = 0; s < nState; s++) {
        Tensor x(90, 1);
        chessLikeState(x);
        const Tensor &o = L->forward(x, true);
        double arg = 0;
        for (std::size_t j = 0; j < L->outputDim; j++) {
            double gv = 0;
            if (tns) {                                  /* r * tanh' * sigmoid' */
                const double o2 = (double)tns->o2[j];
                arg = o2 + (double)tns->b[j];
                gv = r*(1.0 - o2*o2)*Sigmoid::df((double)o[j]);
            } else if (tn) {                            /* r * tanh' */
                const double o2 = (double)tn->o2[j];
                gv = r*(1.0 - o2*o2);
            } else if (kind == 2) {                     /* sigmoid' */
                arg = (double)L->o[j];
                /* recover pre-activation from the output */
                arg = -std::log(1.0/o[j] - 1.0)/1.702;
                gv = Sigmoid::df((double)o[j]);
            } else if (kind == 3) {                     /* tanh' */
                arg = std::atanh(std::max(-0.999999, std::min(0.999999, (double)o[j])));
                gv = 1.0 - (double)o[j]*(double)o[j];
            } else if (kind == 4) {                     /* linear: gain 1 */
                gv = 1.0;
            } else {                                    /* relu: 0 or 1 */
                gv = (o[j] > 0) ? 1.0 : 0.0;
            }
            g.push_back(gv);
            outs[j].push_back((double)o[j]);
            if (kind == 0 || kind == 2) { if (std::fabs(arg) > 2.0) n2++; if (std::fabs(arg) > 4.0) n4++; tot++; }
        }
    }
    std::sort(g.begin(), g.end());
    GainStat st;
    st.p1 = g[g.size()/100];
    st.p10 = g[g.size()/10];
    st.med = g[g.size()/2];
    st.p90 = g[g.size()*9/10];
    st.mx = g.back();
    st.fracDead = (double)std::count_if(g.begin(), g.end(), [](double v){ return v < 0.02; })/g.size();
    st.fracLow  = (double)std::count_if(g.begin(), g.end(), [](double v){ return v < 0.1;  })/g.size();
    st.argBig2 = tot > 0 ? n2/tot : -1;
    st.argBig4 = tot > 0 ? n4/tot : -1;
    int dead = 0;
    for (std::size_t j = 0; j < 64; j++) {
        double m = 0;
        for (std::size_t k = 0; k < outs[j].size(); k++) m += outs[j][k];
        m /= outs[j].size();
        double v = 0;
        for (std::size_t k = 0; k < outs[j].size(); k++) v += (outs[j][k] - m)*(outs[j][k] - m);
        if (std::sqrt(v/outs[j].size()) < 0.02) dead++;
    }
    st.deadUnits = (double)dead/64;
    (void)collectGain;
    return st;
}

static void p8()
{
    std::printf("\n=== P8 local gain d(o_j)/d((Wx)_j): is the tanh->sigmoid sandwich 'more active'? ===\n");
    const char *names[] = {
        "TanhNorm<Sigmoid> (repo)", "TanhNorm<Linear>", "Layer<Sigmoid>", "Layer<Tanh>",
        "Layer<Linear>", "Layer<Relu>"
    };
    std::printf("  %-26s %8s %8s %8s %8s %8s %9s %9s %8s\n",
                "hidden layer", "p1", "p10", "median", "p90", "max", "<0.02", "<0.1", "deadU");
    for (int k = 0; k < 6; k++) {
        GainStat s = gainOf(k, 256);
        std::printf("  %-26s %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f %8.3f %8.3f\n",
                    names[k], s.p1, s.p10, s.med, s.p90, s.mx, s.fracDead, s.fracLow, s.deadUnits);
        if (k == 0 || k == 2)
            std::printf("      |sigmoid argument| > 2 : %5.1f%%   > 4 : %5.1f%%\n",
                        100.0*s.argBig2, 100.0*s.argBig4);
    }
    std::printf("    (<0.02 = near-dead coordinate; deadU = fraction of units whose output\n"
                "     varies < 0.02 across states, i.e. a feature that says nothing)\n");
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *only = (argc > 1) ? argv[1] : "";
    if (only[0] == 0 || std::string(only) == "1") p1();
    if (only[0] == 0 || std::string(only) == "2") p2();
    if (only[0] == 0 || std::string(only) == "3") p3();
    if (only[0] == 0 || std::string(only) == "5") p5();
    if (only[0] == 0 || std::string(only) == "6") p6();
    if (only[0] == 0 || std::string(only) == "7") p7();
    if (only[0] == 0 || std::string(only) == "8") p8();
    return 0;
}
