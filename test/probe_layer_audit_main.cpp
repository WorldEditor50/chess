/*
 * 独立探针 (probe), **故意不注册进 CMakeLists.txt** —— 那里的测试源列表是显式的,
 * 所以本文件不会被自动编译, 也不进 ctest。编译方法 (需要已构建的 RL_CORE.lib) 见
 * docs/tanh_norm_analysis.md §12。用法: 参数 0 / b / 1 / 2 / 3 / 4 分节跑,
 * 不带参数 = 跑 0,1,2,3 (b 与 4 需显式指定)。
 *
 * Why does TanhNorm help DQN converge?
 *
 * P0  layer gradient audit: for every normalization-style FC layer, check
 *     dL/dw, dL/db, dL/dx against central differences.
 *     (This is the contract test that test_grad_main.cpp does NOT cover for
 *      LayerNorm / RMSNorm - it only covers the production TanhNorm path.)
 *
 * P1  optimizer scale immunity: under the repo default RMSProp(+clipGrad),
 *     does shrinking the gradient by 1000x change the parameter trajectory?
 *
 * P2  hidden-layer geometry: same init weights, only the hidden layer type
 *     changes. Cross-state |h| spread, dQ/dx, dQ/dtheta (trunk vs head), Q scale.
 *
 * P3  mini-DQN: a synthetic but Bellman-consistent TD task with a known Q*,
 *     updated exactly like this repo's DQN (batch=32 accumulate, tau=0.01
 *     soft update every 256 steps, RMSProp+clip). RMSE to Q* vs updates,
 *     3 seeds, identical init per seed across variants.
 */
#include <cstdio>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <random>
#include <vector>
#include <memory>
#include <functional>
#include "rl/tensor.hpp"
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/loss.h"
#include "rl/optimize.h"

using namespace RL;

static std::mt19937 g_rng(987654321u);
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
static double reld(double num, double an)
{
    return std::fabs(num - an)/std::max(1.0, std::max(std::fabs(num), std::fabs(an)));
}

/* ===================== P0: layer gradient audit ===================== */
/* NOTE: K0 == D0 on purpose - LayerNorm<Pre>::backward sizes dL by outputDim
   and feeds it to kikj(), so inputDim != outputDim trips the MM shape assert
   (and is wrong anyway). Using a square layer isolates the *semantic* error. */
static const std::size_t K0 = 8, D0 = 8;

template <class L>
static double lossOf(L &layer, const Tensor &x, const Tensor &c)
{
    Tensor &o = layer.forward(x, true);
    double s = 0;
    for (std::size_t i = 0; i < o.totalSize; i++) s += (double)c[i]*(double)o[i];
    return s;
}

struct Audit { double ew, eb, ex; };

template <class L>
static Audit auditLayer(L &layer, const Tensor &x_, const Tensor &c, double eps)
{
    Tensor x = x_;
    layer.forward(x, true);
    layer.e = c;
    Tensor ei(layer.inputDim, 1);
    layer.g.zero();
    layer.backward(x, ei);
    Tensor gw = layer.g.w, gb = layer.g.b;

    Audit a; a.ew = a.eb = a.ex = 0;
    for (std::size_t i = 0; i < layer.outputDim; i++) {
        for (std::size_t j = 0; j < layer.inputDim; j++) {
            float w0 = layer.w(i, j);
            layer.w(i, j) = (float)(w0 + eps); double lp = lossOf(layer, x, c);
            layer.w(i, j) = (float)(w0 - eps); double lm = lossOf(layer, x, c);
            layer.w(i, j) = w0;
            a.ew = std::max(a.ew, reld((lp - lm)/(2*eps), gw(i, j)));
        }
    }
    for (std::size_t i = 0; i < layer.outputDim; i++) {
        float b0 = layer.b[i];
        layer.b[i] = (float)(b0 + eps); double lp = lossOf(layer, x, c);
        layer.b[i] = (float)(b0 - eps); double lm = lossOf(layer, x, c);
        layer.b[i] = b0;
        a.eb = std::max(a.eb, reld((lp - lm)/(2*eps), gb[i]));
    }
    for (std::size_t j = 0; j < layer.inputDim; j++) {
        float x0 = x[j];
        x[j] = (float)(x0 + eps); double lp = lossOf(layer, x, c);
        x[j] = (float)(x0 - eps); double lm = lossOf(layer, x, c);
        x[j] = x0;
        a.ex = std::max(a.ex, reld((lp - lm)/(2*eps), ei[j]));
    }
    return a;
}

static void p0()
{
    std::printf("=== P0 normalization-layer gradient audit (central diff, eps=1e-2) ===\n");
    std::printf("    %-26s %10s %10s %10s\n", "layer (Fn=Linear)", "err dL/dw", "err dL/db", "err dL/dx");

    Tensor x(K0, 1), c(D0, 1);
    fillU(x, -1, 1);
    fillU(c, -0.5f, 0.5f);
    const double eps = 1e-2;

#define AUDIT(NAME, ...)                                                    \
    {                                                                       \
        Random::setSeed(4242);                                              \
        auto L = __VA_ARGS__;                                               \
        L->w = Tensor(D0, K0); fillU(L->w, -1.0f, 1.0f);                    \
        L->b = Tensor(D0, 1);  fillU(L->b, -0.5f, 0.5f);                    \
        Audit a = auditLayer(*L, x, c, eps);                                \
        std::printf("    %-26s %10.2e %10.2e %10.2e\n", NAME, a.ew, a.eb, a.ex); \
    }

    AUDIT("TanhNorm<Linear>",        TanhNorm<Linear>::_(K0, D0, true, true));
    AUDIT("RMSNorm<Linear>",         RMSNorm<Linear>::_(K0, D0, true, true));
    AUDIT("LayerNorm<Def,Linear>",   LayerNorm<Linear>::_(K0, D0, true, true));
    AUDIT("LayerNorm<Pre,Linear>",   LayerNorm<Linear, LN::Pre>::_(K0, D0, true, true));
    AUDIT("LayerNorm<Post,Linear>",  LayerNorm<Linear, LN::Post>::_(K0, D0, true, true));
    AUDIT("iFcLayer (plain linear)", std::make_shared<iFcLayer>(K0, D0, true, true));
#undef AUDIT

    std::printf("    %-26s %10s %10s %10s\n", "layer (Fn=Sigmoid)", "", "", "");
#define AUDIT2(NAME, ...)                                                   \
    {                                                                       \
        Random::setSeed(4242);                                              \
        auto L = __VA_ARGS__;                                               \
        L->w = Tensor(D0, K0); fillU(L->w, -1.0f, 1.0f);                    \
        L->b = Tensor(D0, 1);  fillU(L->b, -0.5f, 0.5f);                    \
        Audit a = auditLayer(*L, x, c, eps);                                \
        std::printf("    %-26s %10.2e %10.2e %10.2e\n", NAME, a.ew, a.eb, a.ex); \
    }
    AUDIT2("TanhNorm<Sigmoid>",       TanhNorm<Sigmoid>::_(K0, D0, true, true));
    AUDIT2("RMSNorm<Sigmoid>",        RMSNorm<Sigmoid>::_(K0, D0, true, true));
    AUDIT2("LayerNorm<Def,Sigmoid>",  LayerNorm<Sigmoid>::_(K0, D0, true, true));
    AUDIT2("LayerNorm<Post,Sigmoid>", LayerNorm<Sigmoid, LN::Post>::_(K0, D0, true, true));
#undef AUDIT2
    std::printf("    (err = max relative error over all w / all b / all x entries)\n");
}

/* ---- P0b: LayerNorm<Def>::backward hands the trunk a mis-scaled dL/dx.
   forward: z = Wx ; u = mean(z) ; g = 1/std(z) ; o = Fn(g(z-u)+b)
   consistent (statistics-detached) dL/dx  = W^T [ g*(dy - mean(dy)) ] ,  dy = Fn'(o)*c
   what the code actually computes        = W^T [ dy ]        <-- no g, no centering
   Measure the ratio as the layer's own activation scale moves.  ---- */
static void p0b()
{
    std::printf("\n=== P0b LayerNorm<Def>::backward -> dL/dx handed to the trunk ===\n");
    std::printf("    %-8s %10s %10s %12s %10s\n", "w scale", "|ei_code|/|ei_ref|", "cos", "1/gamma", "FD err(code)");
    const std::size_t N = 8;
    Tensor x(N, 1), c(N, 1);
    fillU(x, -1, 1);
    fillU(c, -0.5f, 0.5f);
    float scales[3] = {0.02f, 1.0f, 5.0f};
    for (int si = 0; si < 3; si++) {
        Random::setSeed(31);
        auto L = LayerNorm<Linear>::_(N, N, true, true);
        L->w = Tensor(N, N); fillU(L->w, -scales[si], scales[si]);
        L->b = Tensor(N, 1); fillU(L->b, -0.5f, 0.5f);

        Tensor &o = L->forward(x, true);
        /* reference: statistics-detached, internally consistent */
        double sum = 0;
        for (std::size_t i = 0; i < N; i++) sum += (double)L->op[i];
        double u = sum/N;
        double var = 0;
        for (std::size_t i = 0; i < N; i++) var += ((double)L->op[i] - u)*((double)L->op[i] - u);
        var /= N;
        double g = 1.0/std::sqrt(var + 1e-9);
        Tensor dy(N, 1);
        double mdy = 0;
        for (std::size_t i = 0; i < N; i++) { dy[i] = c[i]; mdy += (double)c[i]/N; }
        Tensor dz(N, 1);
        for (std::size_t i = 0; i < N; i++) dz[i] = (float)(g*((double)dy[i] - mdy));
        Tensor eiRef(N, 1);
        Tensor::MM::kikj(eiRef, L->w, dz);

        L->forward(x, true);
        L->e = c;
        Tensor ei(N, 1);
        L->g.zero();
        L->backward(x, ei);

        double dot = 0, na = 0, nb = 0;
        for (std::size_t i = 0; i < N; i++) { dot += (double)ei[i]*(double)eiRef[i];
            na += (double)ei[i]*(double)ei[i]; nb += (double)eiRef[i]*(double)eiRef[i]; }
        double cos = dot/(std::sqrt(na)*std::sqrt(nb) + 1e-30);

        /* FD error of the code's ei */
        double fde = 0;
        for (std::size_t j = 0; j < N; j++) {
            float x0 = x[j];
            x[j] = x0 + 1e-2f; Tensor &op1 = L->forward(x, true); double lp = 0;
            for (std::size_t i = 0; i < N; i++) lp += (double)c[i]*(double)op1[i];
            x[j] = x0 - 1e-2f; Tensor &om1 = L->forward(x, true); double lm = 0;
            for (std::size_t i = 0; i < N; i++) lm += (double)c[i]*(double)om1[i];
            x[j] = x0;
            fde = std::max(fde, reld((lp - lm)/2e-2, ei[j]));
        }
        std::printf("    %-8.2f %18.3f %10.4f %12.2f %10.2e\n",
                    (double)scales[si], std::sqrt(na/(nb + 1e-30)), cos, g, fde);
    }
    std::printf("    (|ei_code|/|ei_ref| ~ 1/gamma  => the trunk's gradient is off by the\n"
                "     layer's own 1/std factor, which is state dependent.)\n");
}

/* ============================ P1 ============================ */
static void p1()
{
    const std::size_t R = 64, C = 90;
    std::printf("\n=== P1 optimizer scale immunity (RMSProp lr=0.01 rho=0.9, 200 steps) ===\n");

    std::vector<Tensor> dirs;
    for (int t = 0; t < 200; t++) { Tensor d(R, C); fillU(d, -1, 1); dirs.push_back(d); }

    for (int clip = 0; clip <= 1; clip++) {
        Tensor w0(R, C);
        Random::setSeed(7);
        Random::uniform(w0, -1, 1);
        Tensor wa = w0, wb = w0;
        Tensor va(R, C), vb(R, C);
        for (int t = 0; t < 200; t++) {
            Tensor ga = dirs[t];
            Tensor gb = dirs[t];
            for (std::size_t i = 0; i < gb.totalSize; i++) gb[i] *= 1e-3f;
            Optimize::RMSProp(wa, va, ga, 0.01f, 0.9f, 0.0f, clip == 1);
            Optimize::RMSProp(wb, vb, gb, 0.01f, 0.9f, 0.0f, clip == 1);
        }
        std::printf("  clipGrad=%-5s : grad x1 vs grad x1e-3 -> max|dw| = %.3e  (|wa|=%.4f |wb|=%.4f)\n",
                    clip ? "true" : "false", maxAbsDiff(wa, wb), nrm2(wa), nrm2(wb));
    }
    {
        Tensor w0(R, C);
        Random::setSeed(11);
        Random::uniform(w0, -1, 1);
        Tensor wa = w0, wb = w0;
        Tensor va(R, C), vb(R, C);
        for (int t = 0; t < 200; t++) {
            Tensor ga = dirs[t], gb = dirs[t];
            Optimize::RMSProp(wa, va, ga, 0.01f, 0.9f, 0.0f, true);
            Optimize::RMSProp(wb, vb, gb, 0.01f, 0.9f, 0.0f, false);
        }
        std::printf("  steady grads   : clip=true vs false  -> max|dw| = %.3e\n", maxAbsDiff(wa, wb));
    }
    for (int clip = 0; clip <= 1; clip++) {
        Tensor w0(R, C);
        Random::setSeed(13);
        Random::uniform(w0, -1, 1);
        Tensor wa = w0, va(R, C);
        double biggest = 0;
        for (int t = 0; t < 200; t++) {
            Tensor ga = dirs[t];
            if (t == 50 || t == 100 || t == 150)
                for (std::size_t i = 0; i < ga.totalSize; i++) ga[i] *= 50.0f;
            Tensor before = wa;
            Optimize::RMSProp(wa, va, ga, 0.01f, 0.9f, 0.0f, clip == 1);
            biggest = std::max(biggest, maxAbsDiff(wa, before));
        }
        std::printf("  spike x50      : clipGrad=%-5s -> max single-step move = %.5f\n",
                    clip ? "true" : "false", biggest);
    }
}

/* ===================== P2 / P3 shared ===================== */
static std::shared_ptr<iLayer> mkTanhNorm(std::size_t i, std::size_t o, bool g)
{ return TanhNorm<Sigmoid>::_(i, o, true, g); }
static std::shared_ptr<iLayer> mkSigmoid(std::size_t i, std::size_t o, bool g)
{ return Layer<Sigmoid>::_(i, o, true, g); }
static std::shared_ptr<iLayer> mkTanh(std::size_t i, std::size_t o, bool g)
{ return Layer<Tanh>::_(i, o, true, g); }
static std::shared_ptr<iLayer> mkRelu(std::size_t i, std::size_t o, bool g)
{ return Layer<Relu>::_(i, o, true, g); }
static std::shared_ptr<iLayer> mkLinear(std::size_t i, std::size_t o, bool g)
{ return Layer<Linear>::_(i, o, true, g); }
static std::shared_ptr<iLayer> mkLayerNorm(std::size_t i, std::size_t o, bool g)
{ return LayerNorm<Sigmoid>::_(i, o, true, g); }

struct Variant {
    const char *name;
    std::function<std::shared_ptr<iLayer>(std::size_t, std::size_t, bool)> make;
};

static void chessLikeState(Tensor &x)
{
    x.zero();
    for (int k = 0; k < 32; k++) {
        int idx = (int)urand(0.0f, 89.999f);
        int v = (int)urand(-7.0f, 7.99f);
        x[(std::size_t)idx] = (float)v/7.0f;
    }
}

static const std::size_t SDIM = 90, HDIM = 64, ADIM = 128;

static void p2()
{
    std::printf("\n=== P2 hidden-layer geometry (90->64->128, same init weights per seed) ===\n");
    Variant vs[] = {
        {"TanhNorm<Sigmoid>", mkTanhNorm},
        {"LayerNorm<Sigmoid>", mkLayerNorm},
        {"Layer<Sigmoid>", mkSigmoid},
        {"Layer<Tanh>", mkTanh},
        {"Layer<Relu>", mkRelu},
        {"Layer<Linear>", mkLinear},
    };
    const int S = 128;
    std::vector<Tensor> states;
    for (int i = 0; i < S; i++) {
        Tensor x(SDIM, 1);
        chessLikeState(x);
        states.push_back(x);
    }

    std::printf("  %-18s %9s %9s %16s %10s %9s %9s\n",
                "hidden layer", "|h|mean", "|h|max/me", "Q[min..max]", "|dQ/dx|", "|dQ/dth|", "head");

    for (Variant &v : vs) {
        Random::setSeed(1234);
        auto hidden = v.make(SDIM, HDIM, true);
        auto head = Layer<Linear>::_(HDIM, ADIM, true, true);
        Net net(hidden, head);

        double hSum = 0, hMax = 0;
        double qmin = 1e30, qmax = -1e30;
        for (int i = 0; i < S; i++) {
            Tensor &q = net.forward(states[i], true);
            double n = nrm2(net[0]->o);
            hSum += n;
            hMax = std::max(hMax, n);
            for (std::size_t k = 0; k < q.totalSize; k++) {
                qmin = std::min(qmin, (double)q[k]);
                qmax = std::max(qmax, (double)q[k]);
            }
        }
        double hMean = hSum/S;
        double dxSum = 0, ghSum = 0, ghdSum = 0;
        const int KC = 16;
        for (int k = 0; k < KC; k++) {
            net.forward(states[k], true);
            Tensor onehot(ADIM, 1);
            onehot.zero();
            onehot[(std::size_t)(k*7 % (int)ADIM)] = 1.0f;
            net.backward(states[k], onehot);
            dxSum += nrm2(net.inputGrad);
            iFcLayer *ph = static_cast<iFcLayer*>(net[0]);
            iFcLayer *po = static_cast<iFcLayer*>(net[1]);
            ghSum += std::sqrt(nrm2(ph->g.w)*nrm2(ph->g.w) + nrm2(ph->g.b)*nrm2(ph->g.b));
            ghdSum += std::sqrt(nrm2(po->g.w)*nrm2(po->g.w) + nrm2(po->g.b)*nrm2(po->g.b));
            ph->g.zero();
            po->g.zero();
        }
        std::printf("  %-18s %9.4f %9.2f [%6.2f,%6.2f] %10.4f %9.4f %9.4f\n",
                    v.name, hMean, hMax/(hMean + 1e-12), qmin, qmax,
                    dxSum/KC, ghSum/KC, ghdSum/KC);
    }
}

/* ============================ P3 ============================ */
static const int NMEM = 512;
static std::vector<Tensor> g_states;
static std::vector<std::vector<float> > g_theta;

static void makeTask()
{
    g_states.clear();
    for (int i = 0; i < NMEM; i++) {
        Tensor x(SDIM, 1);
        chessLikeState(x);
        g_states.push_back(x);
    }
    g_theta.assign(ADIM, std::vector<float>(SDIM, 0.0f));
    for (int a = 0; a < (int)ADIM; a++)
        for (std::size_t j = 0; j < SDIM; j++)
            if (urand(0.0f, 1.0f) < 0.3f) g_theta[a][j] = urand(-1.0f, 1.0f);
}
static double qStar(const Tensor &x, int a)
{
    double s = 0;
    for (std::size_t j = 0; j < SDIM; j++) s += (double)g_theta[a][j]*(double)x[j];
    return 3.0*std::tanh(s/3.0);
}
static double qStarMax(const Tensor &x, int &am)
{
    double best = -1e30; am = 0;
    for (int a = 0; a < (int)ADIM; a++) {
        double v = qStar(x, a);
        if (v > best) { best = v; am = a; }
    }
    return best;
}
struct Trans { int s, a, s2; float r; };

static void p3()
{
    std::printf("\n=== P3 mini-DQN (batch=32 accumulate -> RMSProp(lr=.01,.9,0,clip); tau=.01/256; gamma=.99) ===\n");
    makeTask();

    const int NT = 4096;
    std::vector<Trans> tr;
    std::mt19937 trng(555);
    std::uniform_int_distribution<int> ds(0, NMEM - 1);
    std::uniform_int_distribution<int> da(0, (int)ADIM - 1);
    for (int i = 0; i < NT; i++) {
        Trans t;
        t.s = ds(trng); t.s2 = ds(trng);
        int am;
        qStarMax(g_states[t.s], am);           /* greedy behaviour policy, eps=0.1 */
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        t.a = (u01(trng) < 0.9) ? am : da(trng);
        double qn = qStarMax(g_states[t.s2], am);
        t.r = (float)(qStar(g_states[t.s], t.a) - 0.99*qn);
        tr.push_back(t);
    }

    const int SE = 128;
    std::vector<Tensor> evalStates;
    for (int i = 0; i < SE; i++) {
        Tensor x(SDIM, 1);
        chessLikeState(x);
        evalStates.push_back(x);
    }

    Variant vs[] = {
        {"TanhNorm<Sigmoid>", mkTanhNorm},
        {"LayerNorm<Sigmoid>", mkLayerNorm},
        {"Layer<Sigmoid>", mkSigmoid},
        {"Layer<Tanh>", mkTanh},
        {"Layer<Relu>", mkRelu},
        {"Layer<Linear>", mkLinear},
    };

    const int updates[5] = {250, 500, 1000, 2000, 4000};
    const int NUPD = 4000, NSEED = 3;

    std::printf("  %-18s", "hidden layer");
    for (int i = 0; i < 5; i++) std::printf(" %7d", updates[i]);
    std::printf("   max|Q|\n");

    for (Variant &v : vs) {
        double mse[5] = {0,0,0,0,0};
        double qmaxAbs = 0;
        for (int sd = 0; sd < NSEED; sd++) {
            Random::setSeed(2026u + (unsigned)sd);
            auto hidden = v.make(SDIM, HDIM, true);
            auto head = Layer<Linear>::_(HDIM, ADIM, true, true);
            Net main(hidden, head);
            Net target(hidden, head);
            main.copyTo(target);
            std::mt19937 urng(777u + (unsigned)sd);
            std::uniform_int_distribution<int> pick(0, NT - 1);
            int ui = 0;
            for (int step = 1; step <= NUPD && ui < 5; step++) {
                if (step > 1 && step % 256 == 1) main.softUpdateTo(target, 0.01);
                for (int b = 0; b < 32; b++) {
                    const Trans &t = tr[pick(urng)];
                    Tensor out = main.forward(g_states[t.s]);
                    Tensor qT = out;
                    Tensor &vnext = target.forward(g_states[t.s2]);
                    int k = (int)vnext.argmax();
                    qT[(std::size_t)t.a] = t.r + 0.99f*vnext[(std::size_t)k];
                    main.backward(g_states[t.s], Loss::MSE::df(out, qT));
                }
                main.RMSProp(0.01f, 0.9f, 0.0f);
                if (step == updates[ui]) {
                    double sew = 0;
                    long cnt = 0;
                    for (int i = 0; i < SE; i++) {
                        Tensor &q = main.forward(evalStates[i], true);
                        for (int a = 0; a < (int)ADIM; a++) {
                            double d = (double)q[(std::size_t)a] - qStar(evalStates[i], a);
                            sew += d*d; cnt++;
                            qmaxAbs = std::max(qmaxAbs, std::fabs((double)q[(std::size_t)a]));
                        }
                    }
                    mse[ui] += std::sqrt(sew/cnt);
                    ui++;
                }
            }
        }
        std::printf("  %-18s", v.name);
        for (int i = 0; i < 5; i++) std::printf(" %7.4f", mse[i]/NSEED);
        std::printf("   %6.1f\n", qmaxAbs);
    }
    std::printf("  (numbers = RMSE to Q* on 128x128 held-out states, mean of 3 seeds)\n");
}

/* ============ P4: realizable TD task -> measures learning SPEED ============
   Ideal trunk: 10 features  phi_j(x) = tanh(theta_j . x / 3),  theta_j sparse.
   Ideal head :  Q*(x,a) = sum_j w[a][j] * phi_j(x)      (~ +-3, action dependent)
   A 64-unit hidden layer + linear head can represent this; the DQN update rule is
   exactly the same as P3. So here the RMSE to Q* should FALL, and the slope is
   the "convergence speed" the question asks about.                          */
static const int NF = 10;
static float g_thetaF[NF][90];
static std::vector<std::vector<float> > g_wa;

static void makeTask4()
{
    for (int j = 0; j < NF; j++)
        for (std::size_t k = 0; k < SDIM; k++)
            g_thetaF[j][k] = (urand(0.0f, 1.0f) < 0.3f) ? urand(-1.0f, 1.0f) : 0.0f;
    g_wa.assign(ADIM, std::vector<float>(NF, 0.0f));
    for (int a = 0; a < (int)ADIM; a++)
        for (int j = 0; j < NF; j++) g_wa[a][j] = urand(-1.0f, 1.0f)*0.95f; /* /sqrt(10) */
}
static void phiOf(const Tensor &x, double *p)
{
    for (int j = 0; j < NF; j++) {
        double s = 0;
        for (std::size_t k = 0; k < SDIM; k++) s += (double)g_thetaF[j][k]*(double)x[k];
        p[j] = std::tanh(s/3.0);
    }
}
static double qStar4(const Tensor &x, int a)
{
    double p[NF];
    phiOf(x, p);
    double s = 0;
    for (int j = 0; j < NF; j++) s += (double)g_wa[a][j]*p[j];
    return s;
}
static double qStar4Max(const Tensor &x, int &am)
{
    double p[NF];
    phiOf(x, p);
    double best = -1e30; am = 0;
    for (int a = 0; a < (int)ADIM; a++) {
        double s = 0;
        for (int j = 0; j < NF; j++) s += (double)g_wa[a][j]*p[j];
        if (s > best) { best = s; am = a; }
    }
    return best;
}

static void runDQN(Variant &v, const std::vector<Trans> &tr,
                   const std::vector<Tensor> &evalStates, bool realizable,
                   double *out, double &qmaxAbs)
{
    const int updates[5] = {250, 500, 1000, 2000, 4000};
    const int NUPD = 4000, NSEED = 3;
    double mse[5] = {0,0,0,0,0};
    for (int sd = 0; sd < NSEED; sd++) {
        Random::setSeed(2026u + (unsigned)sd);
        auto hidden = v.make(SDIM, HDIM, true);
        auto head = Layer<Linear>::_(HDIM, ADIM, true, true);
        Net main(hidden, head);
        Net target(hidden, head);
        main.copyTo(target);
        std::mt19937 urng(777u + (unsigned)sd);
        std::uniform_int_distribution<int> pick(0, (int)tr.size() - 1);
        int ui = 0;
        for (int step = 1; step <= NUPD && ui < 5; step++) {
            if (step > 1 && step % 256 == 1) main.softUpdateTo(target, 0.01);
            for (int b = 0; b < 32; b++) {
                const Trans &t = tr[pick(urng)];
                Tensor out = main.forward(g_states[t.s]);
                Tensor qT = out;
                Tensor &vnext = target.forward(g_states[t.s2]);
                int k = (int)vnext.argmax();
                qT[(std::size_t)t.a] = t.r + 0.99f*vnext[(std::size_t)k];
                main.backward(g_states[t.s], Loss::MSE::df(out, qT));
            }
            main.RMSProp(0.01f, 0.9f, 0.0f);
            if (step == updates[ui]) {
                double sew = 0; long cnt = 0;
                for (std::size_t i = 0; i < evalStates.size(); i++) {
                    Tensor &q = main.forward(evalStates[i], true);
                    for (int a = 0; a < (int)ADIM; a++) {
                        double qs = realizable ? qStar4(evalStates[i], a)
                                               : qStar(evalStates[i], a);
                        double d = (double)q[(std::size_t)a] - qs;
                        sew += d*d; cnt++;
                        qmaxAbs = std::max(qmaxAbs, std::fabs((double)q[(std::size_t)a]));
                    }
                }
                mse[ui] += std::sqrt(sew/cnt);
                ui++;
            }
        }
    }
    for (int i = 0; i < 5; i++) out[i] = mse[i]/NSEED;
}

static void p4()
{
    std::printf("\n=== P4 mini-DQN on a REALIZABLE target (10 tanh features + linear head) ===\n");
    makeTask();
    makeTask4();

    const int NT = 4096;
    std::vector<Trans> tr;
    std::mt19937 trng(555);
    std::uniform_int_distribution<int> ds(0, NMEM - 1);
    std::uniform_int_distribution<int> da(0, (int)ADIM - 1);
    for (int i = 0; i < NT; i++) {
        Trans t;
        t.s = ds(trng); t.s2 = ds(trng);
        int am;
        qStar4Max(g_states[t.s], am);          /* greedy behaviour policy, eps=0.1 */
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        t.a = (u01(trng) < 0.9) ? am : da(trng);
        double qn = qStar4Max(g_states[t.s2], am);
        t.r = (float)(qStar4(g_states[t.s], t.a) - 0.99*qn);
        tr.push_back(t);
    }
    const int SE = 128;
    std::vector<Tensor> evalStates;
    for (int i = 0; i < SE; i++) {
        Tensor x(SDIM, 1);
        chessLikeState(x);
        evalStates.push_back(x);
    }

    /* initial RMSE of an untrained net (same seed as the runs) */
    double rmse0 = 0;
    {
        Random::setSeed(2026u);
        auto hidden = mkTanhNorm(SDIM, HDIM, true);
        auto head = Layer<Linear>::_(HDIM, ADIM, true, true);
        Net net(hidden, head);
        double sew = 0; long cnt = 0;
        for (std::size_t i = 0; i < evalStates.size(); i++) {
            Tensor &q = net.forward(evalStates[i], true);
            for (int a = 0; a < (int)ADIM; a++) {
                double d = (double)q[(std::size_t)a] - qStar4(evalStates[i], a);
                sew += d*d; cnt++;
            }
        }
        rmse0 = std::sqrt(sew/cnt);
    }
    std::printf("  untrained RMSE to Q* = %.4f  (Q* range about +-3)\n", rmse0);

    Variant vs[] = {
        {"TanhNorm<Sigmoid>", mkTanhNorm},
        {"LayerNorm<Sigmoid>", mkLayerNorm},
        {"Layer<Sigmoid>", mkSigmoid},
        {"Layer<Tanh>", mkTanh},
        {"Layer<Relu>", mkRelu},
        {"Layer<Linear>", mkLinear},
    };
    const int updates[5] = {250, 500, 1000, 2000, 4000};
    std::printf("  %-18s", "hidden layer");
    for (int i = 0; i < 5; i++) std::printf(" %7d", updates[i]);
    std::printf("   max|Q|\n");
    for (Variant &v : vs) {
        double out[5]; double qmax = 0;
        runDQN(v, tr, evalStates, true, out, qmax);
        std::printf("  %-18s", v.name);
        for (int i = 0; i < 5; i++) std::printf(" %7.4f", out[i]);
        std::printf("   %6.1f\n", qmax);
    }
    std::printf("  (numbers = RMSE to Q*, mean of 3 seeds; lower+falling = converging)\n");
}


int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *only = (argc > 1) ? argv[1] : "";
    if (std::string(only) == "0" || only[0] == 0) p0();
    if (std::string(only) == "b") p0b();
    if (std::string(only) == "1" || only[0] == 0) p1();
    if (std::string(only) == "2" || only[0] == 0) p2();
    if (std::string(only) == "3" || only[0] == 0) p3();
    if (std::string(only) == "4") p4();
    return 0;
}
