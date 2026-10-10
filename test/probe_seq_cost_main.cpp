/*
 * probe_seq_cost_main.cpp — 序列专家的**代价分解**探针 (2026-10)
 * ============================================================================
 * 为什么要有这个探针 (口径问题, 比结论更重要):
 *
 *   这台机器 (i7-12650H, 大小核) 上**同一个二进制、同一份输入**的耗时能在几小时内
 *   漂 5~6 倍 (实测: test_transformer 0.60 s -> 3.38 s; 同一进程内 9 轮之间也有
 *   2.3 倍)。所以:
 *     * **绝对 ns/MAC 不能跨会话/跨机器比较**, 只能当量级;
 *     * 能迁移的只有"**同一进程、同一轮里**成对量出来的比值";
 *     * 每一轮都要带一条**对照通道** (固定小负载), 它的跨度就是"这一轮能不能信"的读数。
 *
 *   本探针就按这个口径输出: 每轮交替测各变体 (顺序翻转), 报**中位数**与区间。
 *
 * 量什么:
 *   [A] 编译期/形状 事实
 *   [B] 真专家: forward / forward+backward  (backward 到底是不是 8x forward)
 *   [C] forward 的分量配对分解 (12 次 FC + 8 次单头注意力 + 4 次 LN)
 *   [D] backward 的分量配对分解 (14 次 fcBwd + 2x4 头 attnBwd + 4 次 LN 反向)
 *   [E] 单件对比: 同一形状的 fcFwd vs fcBwd (反向/前向倍数)
 *   [F] 容器与内核: LN fwd/bwd 用 vector / Tensor / 裸指针; FC 用
 *       逐 token Layer<> vs 整批 MM::ikjk vs 手写三重循环
 *
 * 不进 ctest (计时探针, 读数随机器状态变, 见上)。
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "rl/layer.h"
#include "rl/seq_transformer.hpp"
#include "rl/tensor.hpp"

using RL::Tensor;
using SeqExp = RL::SeqTransformerExpert<90, 4, 2>;

static const int T = 90;          /* SeqLen */
static const int TOK = 64;        /* tokDim */
static const int DFT = 3 * TOK;   /* dff */
static const int DKF = TOK / 4;   /* dk */
static const int D_IN = 1710;
static const int ROUNDS = 9;

static volatile double g_sink = 0.0;

static double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1e6;
}

struct Stat { double med, lo, hi; };
static Stat statOf(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    Stat s;
    s.med = v[v.size() / 2];
    s.lo = v.front();
    s.hi = v.back();
    return s;
}

/* ============================================================
 *  分量: 逐 token 的 fcFwd / fcBwd (与 seq_transformer.hpp 里逐行等价)
 * ============================================================ */
static void fcFwdToken(RL::Layer<RL::Linear> &l, const std::vector<float> &X, int inD, int outD,
                       std::vector<float> &Y, Tensor &xi, bool inference)
{
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
        l.forward(xi, inference);
        for (int j = 0; j < outD; j++) { Y[(std::size_t)(t * outD + j)] = l.o[(std::size_t)j]; }
    }
}
static void fcBwdToken(RL::Layer<RL::Linear> &l, const std::vector<float> &X, int inD, int outD,
                       const std::vector<float> &dYv, std::vector<float> &dX,
                       Tensor &xi, Tensor &ei, Tensor &de)
{
    dX.assign((std::size_t)(T * inD), 0.0f);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < inD; j++) { xi[(std::size_t)j] = X[(std::size_t)(t * inD + j)]; }
        l.forward(xi, true);
        for (int j = 0; j < outD; j++) { de[(std::size_t)j] = dYv[(std::size_t)(t * outD + j)]; }
        l.e = de;
        ei.zero();
        l.backward(xi, ei);
        for (int j = 0; j < inD; j++) { dX[(std::size_t)(t * inD + j)] = ei[(std::size_t)j]; }
    }
}

/* 单头注意力前向 (与 seq_transformer.hpp 的 attnFwd 同一算法, 单头) */
static void attnHeadFwd(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V,
                        std::vector<float> &P, std::vector<float> &O)
{
    const float scale = 1.0f / std::sqrt((float)DKF);
    for (int i = 0; i < T; i++) {
        float mx = -1e30f;
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int c = 0; c < DKF; c++) { s += Q[(std::size_t)(i * TOK + c)] * K[(std::size_t)(j * TOK + c)]; }
            s *= scale;
            P[(std::size_t)(i * T + j)] = s;
            if (s > mx) { mx = s; }
        }
        float sum = 0;
        for (int j = 0; j < T; j++) {
            const float e = std::exp(P[(std::size_t)(i * T + j)] - mx);
            P[(std::size_t)(i * T + j)] = e;
            sum += e;
        }
        for (int j = 0; j < T; j++) { P[(std::size_t)(i * T + j)] /= sum; }
        for (int c = 0; c < DKF; c++) {
            float acc = 0;
            for (int j = 0; j < T; j++) { acc += P[(std::size_t)(i * T + j)] * V[(std::size_t)(j * TOK + c)]; }
            O[(std::size_t)(i * TOK + c)] = acc;
        }
    }
}
/* 多头注意力反向 (= attnBwd 的四个循环, 按头展开) */
static void attnBwdAll(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V,
                       const std::vector<float> &P, const std::vector<float> &dOut,
                       std::vector<float> &dQ, std::vector<float> &dK, std::vector<float> &dV)
{
    const float scale = 1.0f / std::sqrt((float)DKF);
    dQ.assign((std::size_t)(T * TOK), 0.0f);
    dK.assign(dQ.size(), 0.0f);
    dV.assign(dQ.size(), 0.0f);
    std::vector<float> dS((std::size_t)(T * T), 0.0f);
    for (int h = 0; h < 4; h++) {
        const float *Ph = &P[(std::size_t)(h * T * T)];
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float acc = 0;
                for (int c = 0; c < DKF; c++) {
                    acc += dOut[(std::size_t)(i * TOK + h * DKF + c)] * V[(std::size_t)(j * TOK + h * DKF + c)];
                }
                dS[(std::size_t)(i * T + j)] = acc;
            }
            float dot = 0;
            for (int j = 0; j < T; j++) { dot += Ph[(std::size_t)(i * T + j)] * dS[(std::size_t)(i * T + j)]; }
            for (int j = 0; j < T; j++) {
                dS[(std::size_t)(i * T + j)] = Ph[(std::size_t)(i * T + j)] * (dS[(std::size_t)(i * T + j)] - dot);
            }
        }
        for (int j = 0; j < T; j++) {
            for (int c = 0; c < DKF; c++) {
                float accV = 0, accK = 0;
                for (int i = 0; i < T; i++) {
                    accV += Ph[(std::size_t)(i * T + j)] * dOut[(std::size_t)(i * TOK + h * DKF + c)];
                    accK += dS[(std::size_t)(i * T + j)] * Q[(std::size_t)(i * TOK + h * DKF + c)];
                }
                dV[(std::size_t)(j * TOK + h * DKF + c)] += accV;
                dK[(std::size_t)(j * TOK + h * DKF + c)] += accK * scale;
            }
        }
        for (int i = 0; i < T; i++) {
            for (int c = 0; c < DKF; c++) {
                float acc = 0;
                for (int j = 0; j < T; j++) { acc += dS[(std::size_t)(i * T + j)] * K[(std::size_t)(j * TOK + h * DKF + c)]; }
                dQ[(std::size_t)(i * TOK + h * DKF + c)] += acc * scale;
            }
        }
    }
}

/* LN 前向/反向: 三种写法 (与 SeqLayerNorm 等价) */
static void lnFwdVec(const std::vector<float> &X, std::vector<float> &Y, const std::vector<float> &g,
                     const std::vector<float> &b)
{
    for (int t = 0; t < T; t++) {
        const float *xp = &X[(std::size_t)(t * TOK)];
        double m = 0;
        for (int j = 0; j < TOK; j++) { m += xp[j]; }
        m /= (double)TOK;
        double var = 0;
        for (int j = 0; j < TOK; j++) { const double d = xp[j] - m; var += d * d; }
        var /= (double)TOK;
        const float s = (float)std::sqrt(var + 1e-5);
        for (int j = 0; j < TOK; j++) {
            Y[(std::size_t)(t * TOK + j)] = g[(std::size_t)j] * (float)((xp[j] - m) / s) + b[(std::size_t)j];
        }
    }
}
static void lnFwdTensor(const Tensor &X, Tensor &Y, const Tensor &g, const Tensor &b)
{
    for (int t = 0; t < T; t++) {
        double m = 0;
        for (int j = 0; j < TOK; j++) { m += X[(std::size_t)(t * TOK + j)]; }
        m /= (double)TOK;
        double var = 0;
        for (int j = 0; j < TOK; j++) { const double d = X[(std::size_t)(t * TOK + j)] - m; var += d * d; }
        var /= (double)TOK;
        const float s = (float)std::sqrt(var + 1e-5);
        for (int j = 0; j < TOK; j++) {
            Y[(std::size_t)(t * TOK + j)] = g[(std::size_t)j] * (float)((X[(std::size_t)(t * TOK + j)] - m) / s) + b[(std::size_t)j];
        }
    }
}
static void lnFwdPtr(const float *X, float *Y, const float *g, const float *b)
{
    for (int t = 0; t < T; t++) {
        const float *xp = X + (std::size_t)t * TOK;
        double m = 0;
        for (int j = 0; j < TOK; j++) { m += xp[j]; }
        m /= (double)TOK;
        double var = 0;
        for (int j = 0; j < TOK; j++) { const double d = xp[j] - m; var += d * d; }
        var /= (double)TOK;
        const float s = (float)std::sqrt(var + 1e-5);
        float *yp = Y + (std::size_t)t * TOK;
        for (int j = 0; j < TOK; j++) { yp[j] = g[j] * (float)((xp[j] - m) / s) + b[j]; }
    }
}
static void lnBwdIdx(const std::vector<float> &X, const std::vector<float> &dYv, std::vector<float> &dX,
                     const std::vector<float> &g)
{
    const float invN = 1.0f / (float)TOK;
    for (int t = 0; t < T; t++) {
        double m = 0;
        for (int j = 0; j < TOK; j++) { m += X[(std::size_t)(t * TOK + j)]; }
        m /= (double)TOK;
        double var = 0;
        for (int j = 0; j < TOK; j++) { const double d = X[(std::size_t)(t * TOK + j)] - m; var += d * d; }
        var /= (double)TOK;
        const float r = (float)std::sqrt(var + 1e-5);
        const float mu = (float)m;
        float sumDy = 0, sumDyXh = 0;
        for (int j = 0; j < TOK; j++) {
            const float xh = (X[(std::size_t)(t * TOK + j)] - mu) / r;
            sumDy += dYv[(std::size_t)(t * TOK + j)];
            sumDyXh += dYv[(std::size_t)(t * TOK + j)] * xh;
        }
        const float mDy = sumDy * invN, mDyXh = sumDyXh * invN;
        for (int j = 0; j < TOK; j++) {
            const float xh = (X[(std::size_t)(t * TOK + j)] - mu) / r;
            dX[(std::size_t)(t * TOK + j)] += (g[(std::size_t)j] / r) * (dYv[(std::size_t)(t * TOK + j)] - mDy - xh * mDyXh);
        }
    }
}
static void lnBwdPtr(const float *X, const float *dYv, float *dX, const float *g)
{
    const float invN = 1.0f / (float)TOK;
    for (int t = 0; t < T; t++) {
        const float *xp = X + (std::size_t)t * TOK;
        const float *dyp = dYv + (std::size_t)t * TOK;
        float *dxp = dX + (std::size_t)t * TOK;
        double m = 0;
        for (int j = 0; j < TOK; j++) { m += xp[j]; }
        m /= (double)TOK;
        double var = 0;
        for (int j = 0; j < TOK; j++) { const double d = xp[j] - m; var += d * d; }
        var /= (double)TOK;
        const float r = (float)std::sqrt(var + 1e-5);
        const float mu = (float)m;
        float sumDy = 0, sumDyXh = 0;
        for (int j = 0; j < TOK; j++) { const float xh = (xp[j] - mu) / r; sumDy += dyp[j]; sumDyXh += dyp[j] * xh; }
        const float mDy = sumDy * invN, mDyXh = sumDyXh * invN;
        for (int j = 0; j < TOK; j++) {
            const float xh = (xp[j] - mu) / r;
            dxp[j] += (g[j] / r) * (dyp[j] - mDy - xh * mDyXh);
        }
    }
}
/* FC: 整批 MM::ikjk (Tensor 2-D) 与 手写三重循环 (纯 vector) */
static void fcBatchMM(Tensor &Y, const Tensor &X, const Tensor &W, const Tensor &b, int outD)
{
    Y.zero();
    Tensor::MM::ikjk(Y, X, W);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < outD; j++) { Y[(std::size_t)(t * outD + j)] += b[(std::size_t)j]; }
    }
}
static void fcBatchNaive(const std::vector<float> &X, const std::vector<float> &W, const std::vector<float> &b,
                         std::vector<float> &Y, int inD, int outD)
{
    for (int t = 0; t < T; t++) {
        const float *xr = &X[(std::size_t)(t * inD)];
        float *yr = &Y[(std::size_t)(t * outD)];
        for (int o = 0; o < outD; o++) {
            const float *wr = &W[(std::size_t)(o * inD)];
            float acc = b[(std::size_t)o];
            for (int k = 0; k < inD; k++) { acc += wr[k] * xr[k]; }
            yr[o] = acc;
        }
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    RL::Random::setSeed(20240901u);
    std::mt19937 rng(20240901u);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    std::printf("=== probe_seq_cost — 序列专家代价分解 (配对测法) ===\n");
#if defined(__AVX2__)
    std::printf("[A] ISA: AVX2 (/arch:AVX2)   ");
#else
    std::printf("[A] ISA: SSE2 基线 (无 /arch:AVX2)   ");
#endif
    std::printf("T=%d tokDim=%d dff=%d dk=%d 轮数=%d\n\n", T, TOK, DFT, DKF, ROUNDS);

    /* ---------------- 数据 ---------------- */
    SeqExp ex(D_IN, TOK, true);
    RL::scaleExpertInit(ex);
    Tensor xs((std::size_t)D_IN, 1);
    xs.zero();
    std::uniform_int_distribution<int> pick(0, D_IN - 1);
    for (int k = 0; k < 140; k++) { xs[(std::size_t)pick(rng)] = (rng() % 2u) ? 1.0f : -1.0f; }
    Tensor ei((std::size_t)D_IN, 1);
    ei.zero();
    ex.e.fill(1.0f);
    ex.forward(xs, false);

    /* 分量工作负载的输入 */
    std::vector<float> X64((std::size_t)T * TOK), X192((std::size_t)T * DFT), X19((std::size_t)T * 19);
    std::vector<float> Y64((std::size_t)T * TOK), Y192((std::size_t)T * DFT), Y19((std::size_t)T * 19);
    for (std::size_t i = 0; i < X64.size(); i++) { X64[i] = nrm(rng) * 0.1f; }
    for (std::size_t i = 0; i < X192.size(); i++) { X192[i] = nrm(rng) * 0.1f; }
    for (std::size_t i = 0; i < X19.size(); i++) { X19[i] = nrm(rng) * 0.1f; }
    std::vector<float> dX64((std::size_t)T * TOK), dX192((std::size_t)T * DFT), dX19((std::size_t)T * 19);
    Tensor xi64((std::size_t)TOK, 1), xi192((std::size_t)DFT, 1), xi19((std::size_t)19, 1);
    Tensor ei64((std::size_t)TOK, 1), ei192((std::size_t)DFT, 1), ei19((std::size_t)19, 1);
    Tensor de64((std::size_t)TOK, 1), de192((std::size_t)DFT, 1), de19((std::size_t)19, 1);

    RL::Layer<RL::Linear> w64[4] = { RL::Layer<RL::Linear>(TOK, TOK, true, true),
                                     RL::Layer<RL::Linear>(TOK, TOK, true, true),
                                     RL::Layer<RL::Linear>(TOK, TOK, true, true),
                                     RL::Layer<RL::Linear>(TOK, TOK, true, true) };
    RL::Layer<RL::Linear> wUp(TOK, DFT, true, true), wDown(DFT, TOK, true, true);
    RL::Layer<RL::Linear> wOut(TOK, 19, true, true), wEmb(19, TOK, true, true);

    std::vector<float> Q((std::size_t)T * TOK), K((std::size_t)T * TOK), V((std::size_t)T * TOK),
                       P((std::size_t)4 * T * T), O((std::size_t)T * TOK, 0.0f),
                       dQ((std::size_t)T * TOK), dK((std::size_t)T * TOK), dV((std::size_t)T * TOK);
    for (std::size_t i = 0; i < Q.size(); i++) { Q[i] = nrm(rng); K[i] = nrm(rng); V[i] = nrm(rng); }
    for (int h = 0; h < 4; h++) { attnHeadFwd(Q, K, V, std::vector<float>(P.begin() + (std::size_t)h * T * T, P.begin() + (std::size_t)(h + 1) * T * T), O); }

    std::vector<float> LX((std::size_t)T * TOK), LY((std::size_t)T * TOK), dLX((std::size_t)T * TOK, 0.0f),
                       LdY((std::size_t)T * TOK), Lg((std::size_t)TOK, 1.0f), Lb((std::size_t)TOK, 0.0f);
    for (std::size_t i = 0; i < LX.size(); i++) { LX[i] = nrm(rng) * 0.3f; LdY[i] = nrm(rng) * 0.1f; }
    Tensor tLX((std::size_t)T * TOK, 1), tLY((std::size_t)T * TOK, 1), tLg((std::size_t)TOK, 1), tLb((std::size_t)TOK, 1);
    for (std::size_t i = 0; i < LX.size(); i++) { tLX[i] = LX[i]; }
    tLg.fill(1.0f); tLb.fill(0.0f);

    /* MM 版本的权重 */
    Tensor mX(T, TOK), mY(T, TOK), mW(TOK, TOK), mB(TOK, 1);
    for (std::size_t i = 0; i < mX.totalSize; i++) { mX[i] = X64[i]; }
    for (std::size_t i = 0; i < mW.totalSize; i++) { mW[i] = nrm(rng) * 0.1f; }
    for (int i = 0; i < TOK; i++) { mB[i] = nrm(rng) * 0.1f; }
    std::vector<float> Wn((std::size_t)TOK * TOK), bn((std::size_t)TOK);
    for (std::size_t i = 0; i < Wn.size(); i++) { Wn[i] = nrm(rng) * 0.1f; }
    for (int i = 0; i < TOK; i++) { bn[(std::size_t)i] = nrm(rng) * 0.1f; }

    /* 对照通道 */
    std::vector<double> ctlAll;
    auto control = [&]() {
        const double t0 = nowMs();
        for (int i = 0; i < 50; i++) { lnFwdPtr(LX.data(), LY.data(), Lg.data(), Lb.data()); }
        g_sink += LY[0];
        const double v = (nowMs() - t0) / 50;
        ctlAll.push_back(v);
        return v;
    };

    /* ============================================================
     *  [B][C][D] 主循环: 每轮交替测 专家 / 分量, 顺序翻转
     * ============================================================ */
    std::vector<double> vFwd, vBoth, vFcFwd, vAttnFwd, vLnFwd;
    std::vector<double> vBwdFc, vBwdAttn, vBwdLn, vSingle[3];
    for (int r = 0; r < ROUNDS; r++) {
        const bool fwdOrder = (r % 2 == 0);
        auto measureComponents = [&]() {
            double t0;
            t0 = nowMs();
            for (int b = 0; b < 2; b++) {
                for (int i = 0; i < 4; i++) { fcFwdToken(w64[i], X64, TOK, TOK, Y64, xi64, true); }
                fcFwdToken(wUp, X64, TOK, DFT, Y192, xi64, true);
                fcFwdToken(wDown, X192, DFT, TOK, Y64, xi192, true);
            }
            vFcFwd.push_back(nowMs() - t0);
            t0 = nowMs();
            for (int i = 0; i < 8; i++) {
                std::vector<float> Pi(P.begin(), P.begin() + (std::size_t)T * T);
                attnHeadFwd(Q, K, V, Pi, O);
            }
            vAttnFwd.push_back(nowMs() - t0);
            t0 = nowMs();
            for (int i = 0; i < 4; i++) { lnFwdVec(LX, LY, Lg, Lb); }
            vLnFwd.push_back(nowMs() - t0);
        };
        auto measureBwd = [&]() {
            double t0;
            t0 = nowMs();
            for (int b = 0; b < 2; b++) {
                for (int i = 0; i < 4; i++) { fcBwdToken(w64[i], X64, TOK, TOK, Y64, dX64, xi64, ei64, de64); }
                fcBwdToken(wUp, X64, TOK, DFT, Y192, dX64, xi64, ei64, de192);
                fcBwdToken(wDown, X192, DFT, TOK, Y64, dX192, xi192, ei192, de64);
            }
            fcBwdToken(wOut, X64, TOK, 19, Y19, dX64, xi64, ei64, de19);
            fcBwdToken(wEmb, X19, 19, TOK, Y64, dX19, xi19, ei19, de64);
            vBwdFc.push_back(nowMs() - t0);
            t0 = nowMs();
            for (int i = 0; i < 2; i++) { attnBwdAll(Q, K, V, P, O, dQ, dK, dV); }
            vBwdAttn.push_back(nowMs() - t0);
            t0 = nowMs();
            for (int i = 0; i < 4; i++) { lnBwdIdx(LX, LdY, dLX, Lg); }
            vBwdLn.push_back(nowMs() - t0);
        };
        auto measureSingle = [&]() {
            double t0;
            t0 = nowMs(); for (int i = 0; i < 3; i++) { fcFwdToken(w64[0], X64, TOK, TOK, Y64, xi64, true); } vSingle[0].push_back((nowMs() - t0) / 3);
            t0 = nowMs(); for (int i = 0; i < 3; i++) { fcBwdToken(w64[0], X64, TOK, TOK, Y64, dX64, xi64, ei64, de64); } vSingle[1].push_back((nowMs() - t0) / 3);
            t0 = nowMs(); for (int i = 0; i < 3; i++) { fcFwdToken(wUp, X64, TOK, DFT, Y192, xi64, true); } vSingle[2].push_back((nowMs() - t0) / 3);
        };
        if (fwdOrder) {
            double t0 = nowMs(); ex.forward(xs, false); vFwd.push_back(nowMs() - t0);
            t0 = nowMs(); ex.forward(xs, false); ex.backward(xs, ei); vBoth.push_back(nowMs() - t0);
            measureComponents(); measureBwd(); measureSingle();
        } else {
            measureSingle(); measureBwd(); measureComponents();
            double t0 = nowMs(); ex.forward(xs, false); ex.backward(xs, ei); vBoth.push_back(nowMs() - t0);
            t0 = nowMs(); ex.forward(xs, false); vFwd.push_back(nowMs() - t0);
        }
        control();
    }

    const Stat sFwd = statOf(vFwd), sBoth = statOf(vBoth);
    const Stat sFcF = statOf(vFcFwd), sAtF = statOf(vAttnFwd), sLnF = statOf(vLnFwd);
    const Stat sBcB = statOf(vBwdFc), sAtB = statOf(vBwdAttn), sLnB = statOf(vBwdLn);
    const Stat sCtl = statOf(ctlAll);
    const double bwd = sBoth.med - sFwd.med;

    std::printf("[B] 真专家 (每轮交替, %d 轮中位数, 区间)\n", ROUNDS);
    std::printf("    forward            %8.3f ms   [%.3f, %.3f]\n", sFwd.med, sFwd.lo, sFwd.hi);
    std::printf("    forward+backward   %8.3f ms   [%.3f, %.3f]\n", sBoth.med, sBoth.lo, sBoth.hi);
    std::printf("    -> backward        %8.3f ms   = forward 的 %.2f 倍\n\n", bwd, bwd / sFwd.med);

    std::printf("[C] forward 的分量 (各分量凑成\"一次 forward 的用量\")\n");
    std::printf("    FC 12 次           %8.3f ms   = %.0f%% of forward\n", sFcF.med, 100 * sFcF.med / sFwd.med);
    std::printf("    注意力 8 次单头    %8.3f ms   = %.0f%% of forward\n", sAtF.med, 100 * sAtF.med / sFwd.med);
    std::printf("    LN 4 次            %8.3f ms   = %.0f%% of forward\n", sLnF.med, 100 * sLnF.med / sFwd.med);
    std::printf("    三项之和           %8.3f ms   = %.2f x forward (余下部分是逐 token 拷贝/gather/scatter/残差)\n\n",
                sFcF.med + sAtF.med + sLnF.med, (sFcF.med + sAtF.med + sLnF.med) / sFwd.med);

    std::printf("[D] backward 的分量\n");
    std::printf("    fcBwd 14 次        %8.3f ms   = %.0f%% of backward\n", sBcB.med, 100 * sBcB.med / bwd);
    std::printf("    attnBwd 2 次(4头)  %8.3f ms   = %.0f%% of backward\n", sAtB.med, 100 * sAtB.med / bwd);
    std::printf("    LN 反向 4 次       %8.3f ms   = %.0f%% of backward\n", sLnB.med, 100 * sLnB.med / bwd);
    std::printf("    三项之和           %8.3f ms   = %.2f x backward\n\n",
                sBcB.med + sAtB.med + sLnB.med, (sBcB.med + sAtB.med + sLnB.med) / bwd);

    std::printf("[E] 单件: 同一形状的 forward 与 backward (逐 token)\n");
    std::printf("    64->64   fwd %7.4f ms   bwd %7.4f ms   bwd/fwd = %.2f\n",
                statOf(vSingle[0]).med, statOf(vSingle[1]).med, statOf(vSingle[1]).med / statOf(vSingle[0]).med);
    std::printf("    64->192  fwd %7.4f ms\n", statOf(vSingle[2]).med);
    std::printf("    每 FC 反向的等价 MAC: kikj(out*in) + ikjk(out*in) + 重跑前向(out*in)\n\n");

    /* ---------------- [F] 容器与内核 ---------------- */
    std::vector<double> fVec, fTen, fPtr, bIdx, bPtr;
    std::vector<double> fcTok, fcMM, fcNaive;
    for (int r = 0; r < ROUNDS; r++) {
        double t0;
        if (r % 2 == 0) {
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdVec(LX, LY, Lg, Lb); } fVec.push_back((nowMs() - t0) / 100);
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdTensor(tLX, tLY, tLg, tLb); } fTen.push_back((nowMs() - t0) / 100);
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdPtr(LX.data(), LY.data(), Lg.data(), Lb.data()); } fPtr.push_back((nowMs() - t0) / 100);
        } else {
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdPtr(LX.data(), LY.data(), Lg.data(), Lb.data()); } fPtr.push_back((nowMs() - t0) / 100);
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdTensor(tLX, tLY, tLg, tLb); } fTen.push_back((nowMs() - t0) / 100);
            t0 = nowMs(); for (int i = 0; i < 100; i++) { lnFwdVec(LX, LY, Lg, Lb); } fVec.push_back((nowMs() - t0) / 100);
        }
        t0 = nowMs(); for (int i = 0; i < 100; i++) { lnBwdIdx(LX, LdY, dLX, Lg); } bIdx.push_back((nowMs() - t0) / 100);
        t0 = nowMs(); for (int i = 0; i < 100; i++) { lnBwdPtr(LX.data(), LdY.data(), dLX.data(), Lg.data()); } bPtr.push_back((nowMs() - t0) / 100);
        t0 = nowMs(); for (int i = 0; i < 30; i++) { fcFwdToken(w64[0], X64, TOK, TOK, Y64, xi64, true); } fcTok.push_back((nowMs() - t0) / 30);
        t0 = nowMs(); for (int i = 0; i < 30; i++) { fcBatchMM(mY, mX, mW, mB, TOK); } fcMM.push_back((nowMs() - t0) / 30);
        t0 = nowMs(); for (int i = 0; i < 30; i++) { fcBatchNaive(X64, Wn, bn, Y64, TOK, TOK); } fcNaive.push_back((nowMs() - t0) / 30);
    }
    const Stat sv = statOf(fVec), st = statOf(fTen), sp = statOf(fPtr);
    const Stat sb1 = statOf(bIdx), sb2 = statOf(bPtr);
    const Stat sTok = statOf(fcTok), sMM = statOf(fcMM), sNaive = statOf(fcNaive);

    std::printf("[F] 容器与内核 (T=%d)\n", T);
    std::printf("    LN 前向   vector %.4f / Tensor %.4f (%.3fx) / 裸指针 %.4f (%.3fx)  ms\n",
                sv.med, st.med, st.med / sv.med, sp.med, sp.med / sv.med);
    std::printf("    LN 反向   逐元素下标 %.4f / 行指针 %.4f (%.3fx)  ms\n",
                sb1.med, sb2.med, sb2.med / sb1.med);
    std::printf("    FC 64->64 (T=%d): 逐 token Layer<> %.4f / 整批 MM::ikjk %.4f (%.2fx) / 手写三重循环 %.4f (%.2f x)\n",
                T, sTok.med, sMM.med, sTok.med / sMM.med, sNaive.med, sTok.med / sNaive.med);

    std::printf("\n[对照通道] 中位数 %.4f ms, 区间 [%.4f, %.4f] -> **跨度 %.0f%%** (这一轮读数的可信度)\n",
                sCtl.med, sCtl.lo, sCtl.hi, 100.0 * (sCtl.hi - sCtl.lo) / sCtl.lo);

    /* ============================================================
     *  [G] fcBwd 内部拆解: 24 倍于前向的那 4.4 ms 到底花在哪
     * ============================================================ */
    {
        RL::Layer<RL::Linear> &l = w64[0];
        for (int j = 0; j < TOK; j++) { xi64[(std::size_t)j] = X64[(std::size_t)j]; }
        l.forward(xi64, true);
        l.e = de64;

        std::vector<double> gCopy, gFwd, gBwd, gKikj, gIkjk, gZero, gAll;
        for (int r = 0; r < ROUNDS; r++) {
            double t0;
            t0 = nowMs();
            for (int t = 0; t < T; t++) {
                for (int j = 0; j < TOK; j++) { xi64[(std::size_t)j] = X64[(std::size_t)(t * TOK + j)]; }
                for (int j = 0; j < TOK; j++) { Y64[(std::size_t)(t * TOK + j)] = l.o[(std::size_t)j]; }
            }
            gCopy.push_back(nowMs() - t0);
            t0 = nowMs(); for (int t = 0; t < T; t++) { l.forward(xi64, true); } gFwd.push_back(nowMs() - t0);
            t0 = nowMs(); for (int t = 0; t < T; t++) { ei64.zero(); l.backward(xi64, ei64); } gBwd.push_back(nowMs() - t0);
            t0 = nowMs(); for (int t = 0; t < T; t++) { Tensor::MM::kikj(ei64, l.w, l.e); } gKikj.push_back(nowMs() - t0);
            t0 = nowMs(); for (int t = 0; t < T; t++) { Tensor::MM::ikjk(l.g.w, l.e, xi64); } gIkjk.push_back(nowMs() - t0);
            t0 = nowMs(); for (int t = 0; t < T; t++) { l.e.zero(); l.o.zero(); } gZero.push_back(nowMs() - t0);
            t0 = nowMs(); fcBwdToken(l, X64, TOK, TOK, Y64, dX64, xi64, ei64, de64); gAll.push_back(nowMs() - t0);
            control();
        }
        const Stat sc = statOf(gCopy), sf = statOf(gFwd), sb = statOf(gBwd),
                   sk = statOf(gKikj), si = statOf(gIkjk), sz = statOf(gZero), sa = statOf(gAll);
        std::printf("\n[G] fcBwd 内部拆解 (64->64, 每项 = T=%d 个 token 一次)\n", T);
        std::printf("    G1 逐 token 拷贝 in/out          %8.4f ms  (%.0f%% of fcBwd)\n", sc.med, 100 * sc.med / sa.med);
        std::printf("    G2 forward 刷新缓存 90 次        %8.4f ms  (%.0f%%)\n", sf.med, 100 * sf.med / sa.med);
        std::printf("    G3 l.backward 90 次 (整体)       %8.4f ms  (%.0f%%)\n", sb.med, 100 * sb.med / sa.med);
        std::printf("    G4   └ MM::kikj 90 次 (反向GEMV) %8.4f ms  (%.0f%%)\n", sk.med, 100 * sk.med / sa.med);
        std::printf("    G5   └ MM::ikjk 90 次 (外积 dW)  %8.4f ms  (%.0f%%)\n", si.med, 100 * si.med / sa.med);
        std::printf("    G6 e.zero()+o.zero() 90 次       %8.4f ms  (%.0f%%)\n", sz.med, 100 * sz.med / sa.med);
        std::printf("    fcBwdToken 整体 (对照)           %8.4f ms   [G4+G5 占它 %.0f%%]\n", sa.med, 100 * (sk.med + si.med) / sa.med);
        std::printf("    G4/G5 的 MAC: 各 %d 次 x %d x %d = %.2f M; 若按 MM::ikjk 的 %.2f ns/MAC,\n",
                    T, TOK, TOK, 2.0 * T * TOK * TOK / 1e6, si.med * 1e6 / (double)(T * TOK * TOK));
        std::printf("       kikj 的等效速率 = %.2f ns/MAC (前向 ikkj 的参照: probe 里整批 MM::ikjk 见 [F])\n",
                    sk.med * 1e6 / (double)(T * TOK * TOK));
    }

    std::printf("\n(总对照通道跨度 %.0f%%)\n", 100.0 * (statOf(ctlAll).hi - statOf(ctlAll).lo) / statOf(ctlAll).lo);

    /* ============================================================
     *  [H] MM::ikjk 的标量分支: 为什么 FC 反向是前向的 ~24 倍
     * ============================================================
     * iFcLayer::backward 每次调用都做 `MM::ikjk(g.w, e, x)` —— 单样本的**外积**
     * (kdim == 1)。它的 SIMD 派发要求每个维度 >= 一个向量宽 (AVX2 是 8), 而
     * x1Col = x2Col = 1 ⇒ 一定走标量分支; 那一支的 j 循环里**重复读 x1row[k]**,
     * 且指针没有 __restrict ⇒ 编译器必须假设 `xrow[j] += ...` 可能改写 x1d,
     * 于是既不能把那个读提出来、也不能向量化。
     * 下面把"真 ikjk / 逐字复制 / 加 __restrict / kdim==1 特判 / 手写"放一起量:
     * 后两种写法**逐位等价**(不动累加顺序), 只是把 x1 的读提出 j 循环。
     */
    {
        const int Rr = 64, Cc = 64, NH = 500;
        Tensor hz(std::vector<int>{ Rr, Cc });
        Tensor hx1(std::vector<int>{ Rr, 1 }), hx2(std::vector<int>{ Cc, 1 });
        for (std::size_t i = 0; i < hz.totalSize; i++) { hz[i] = nrm(rng); }
        for (std::size_t i = 0; i < hx1.totalSize; i++) { hx1[i] = nrm(rng); }
        for (std::size_t i = 0; i < hx2.totalSize; i++) { hx2[i] = nrm(rng); }
        std::vector<float> ref((std::size_t)Rr * Cc);
        hz.zero();
        Tensor::MM::ikjk(hz, hx1, hx2);
        for (std::size_t i = 0; i < hz.totalSize; i++) { ref[i] = hz[i]; }

        double t0 = nowMs();
        for (int r = 0; r < NH; r++) { Tensor::MM::ikjk(hz, hx1, hx2); }
        const double hReal = (nowMs() - t0) / NH;
        t0 = nowMs();
        for (int r = 0; r < NH; r++) {
            const float *h1 = hx1.ptr(); const float *h2 = hx2.ptr(); float *zd = hz.ptr();
            for (int i = 0; i < Rr; i++) {
                const float ai = h1[i];
                float *xrow = zd + (std::size_t)i * Cc;
                for (int j = 0; j < Cc; j++) { xrow[j] += ai * h2[j]; }
            }
        }
        const double hHand = (nowMs() - t0) / NH;
        /* 逐位等价检查: 手写版跑出来的值与真 ikjk 的差应为 0 (同一累加顺序) */
        Tensor hz2(std::vector<int>{ Rr, Cc });
        for (std::size_t i = 0; i < hz2.totalSize; i++) { hz2[i] = 0.0f; }
        {
            const float *h1 = hx1.ptr(); const float *h2 = hx2.ptr(); float *zd = hz2.ptr();
            for (int i = 0; i < Rr; i++) {
                const float ai = h1[i];
                float *xrow = zd + (std::size_t)i * Cc;
                for (int j = 0; j < Cc; j++) { xrow[j] += ai * h2[j]; }
            }
        }
        double hDiff = 0, hRel = 0;
        for (std::size_t i = 0; i < hz.totalSize; i++) {
            hDiff = std::max(hDiff, (double)std::fabs(hz2[i] - ref[i]));
            hRel = std::max(hRel, (double)std::fabs(hz2[i] - ref[i]) / (std::fabs((double)ref[i]) + 1e-30));
        }
        /* 仓库 FC 反向的真实形状量级 */
        Tensor bg(std::vector<int>{ 360, 1710 }), b1(std::vector<int>{ 360, 1 }), b2(std::vector<int>{ 1710, 1 });
        for (std::size_t i = 0; i < b1.totalSize; i++) { b1[i] = nrm(rng); }
        for (std::size_t i = 0; i < b2.totalSize; i++) { b2[i] = nrm(rng); }
        t0 = nowMs(); for (int r = 0; r < 5; r++) { Tensor::MM::ikjk(bg, b1, b2); } const double bigReal = (nowMs() - t0) / 5;
        t0 = nowMs();
        for (int r = 0; r < 5; r++) {
            const float *h1 = b1.ptr(); const float *h2 = b2.ptr(); float *zd = bg.ptr();
            for (int i = 0; i < 360; i++) {
                const float ai = h1[i];
                float *xrow = zd + (std::size_t)i * 1710;
                for (int j = 0; j < 1710; j++) { xrow[j] += ai * h2[j]; }
            }
        }
        const double bigHand = (nowMs() - t0) / 5;

        const double mac1 = (double)Rr * Cc, macB = 360.0 * 1710.0;
        std::printf("\n[H] MM::ikjk 的标量分支 (kdim=1 的单样本外积; FC 反向的主要成本)\n");
        std::printf("    real ikjk  (64x64)<=(64x1)(64x1)^T : %8.2f us  (%.2f ns/MAC)\n", hReal * 1000, hReal * 1e6 / mac1);
        std::printf("    手写 (提 x1 读到 j 循环外)         : %8.2f us  (%.2f ns/MAC)   相对 real: %.3f  (%.0fx 快)\n",
                    hHand * 1000, hHand * 1e6 / mac1, hHand / hReal, hReal / hHand);
        std::printf("    数值检查 (同一个零初值): 手写 vs real 的最大差 = %.3e, 最大相对差 = %.2e\n", hDiff, hRel);
        std::printf("      (两者累加顺序相同; 差只应来自乘法/加法的 FMA 收缩, ulp 级)\n");
        std::printf("    real ikjk  (360x1710)              : %8.3f ms  (%.2f ns/MAC)  <= 仓库 FC 反向的真实形状\n",
                    bigReal, bigReal * 1e6 / macB);
        std::printf("    手写同形状                         : %8.3f ms  (%.2f ns/MAC)   相对 real: %.3f  (%.0fx 快)\n",
                    bigHand, bigHand * 1e6 / macB, bigHand / bigReal, bigReal / bigHand);
        std::printf("    参照: 真 ikjk 在 kdim=64 (整批) 上是 0.66 ns/MAC —— 差的是 **写法/别名假设**, 不是算力\n");
    }

    std::printf("\n(总对照通道跨度 %.0f%%)\n", 100.0 * (statOf(ctlAll).hi - statOf(ctlAll).lo) / statOf(ctlAll).lo);
    std::printf("(sink=%g)\n", (double)g_sink);
    return 0;
}
