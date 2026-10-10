/*
 * probe_attn_matrix_main.cpp — [2026-10] P6-前哨：注意力「矩阵化」的**只读可行性探针**
 * ============================================================================
 * 为什么要有它（方案 Part 8 第 1 条就是它）：`probe_seq_cost` 已经把 seq 专家的
 * forward 分解出来了 —— **注意力 8 次单头（手写标量四重循环）= 6.303 ms = 39%**，
 * 而 FC 12 次只有 21%。但"矩阵化之后能快多少"一直是**【推理】不是【实测】**，
 * 所以 P6 迟迟不能落地。本探针回答的就是这一个问题：
 *
 *     把 `SeqTransformerExpert::attnFwd/attnBwd` 的**现有标量实现**
 *     与"每头连续 (T×dk) + `MM::` 四个内核"的**矩阵化实现**
 *     在同一进程、同一轮里**配对**对测，给出加速比 + 数值差。
 *
 * 口径（方案 §5.1 / seq_transformer_design §8.4，一条都不能省）:
 *   * **同一进程、同一轮**交替 A/B，多轮取**中位数**的比值；绝对值不可跨机/跨次比较;
 *   * 每轮带一条**对照通道**（固定工作量的 memcpy+归约），印出它的跨度 ——
 *     跨度大就说明这一轮不可信;
 *   * 两实现都**预先分配**好缓冲: 计时区里**一次分配都没有** `MM::` 的 Tensor 也一样;
 *   * 数值差用 max|Δ| 与 max 相对差报（矩阵化会改变求和顺序 ⇒ **不要求逐位相同**，
 *     这也是方案里唯一允许不逐位等价的一项）。
 *
 * 两种矩阵化读数（因为"要不要改布局"是 P6 落地时要定的）:
 *   [M1] 含 **gather/scatter**：每头从交错布局 (T×tokDim) 收集成连续 (T×dk) —— 保守值;
 *   [M2] **不含 gather**（假设布局本来就是按头连续的）—— 上界值。
 * 真实落地时选哪种，取决于要不要顺带改 Q/K/V 的存储布局（那会牵动 seq 的其他接线）。
 *
 * 运行: cmake --build <build> --target probe_attn_matrix && <build>\probe_attn_matrix.exe
 */
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "rl/tensor.hpp"

namespace {

using RL::Tensor;
using Clock = std::chrono::steady_clock;

double nowMs()
{
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

/* ---------------- 配置（默认 = PPO 口径的一半: 见 seq_transformer.hpp 顶部） ---------------- */
struct Cfg {
    int T = 90;        /* SeqLen */
    int tokDim = 64;   /* d_model */
    int heads = 4;
    int dk = 16;       /* tokDim / heads */
};

/* 交错布局的下标（与 seq_transformer.hpp 一致: (t, head h, 分量 c) 在 t*tokDim + h*dk + c） */
inline std::size_t ix(const Cfg &c, int t, int h, int k) { return (std::size_t)(t * c.tokDim + h * c.dk + k); }

/* ============================================================
 *  [A] 现有实现 —— 逐字抄自 src/rl/seq_transformer.hpp 的 attnFwd/attnBwd
 *      （只去掉 mask 形参: 这里比的是**核心注意力数学**, mask/dropout 是另一条支路）
 * ============================================================ */
void scalarFwd(const Cfg &cf, const std::vector<float> &Q, const std::vector<float> &K,
               const std::vector<float> &V, std::vector<float> &P, std::vector<float> &Out)
{
    const int T = cf.T, tokDim = cf.tokDim, heads = cf.heads, dk = cf.dk;
    const float scale = 1.0f / std::sqrt((float)dk);
    for (int h = 0; h < heads; h++) {
        float *Ph = &P[(std::size_t)(h * T * T)];
        for (int i = 0; i < T; i++) {
            float mx = -1e30f;
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int c = 0; c < dk; c++) {
                    s += Q[ix(cf, i, h, c)] * K[ix(cf, j, h, c)];
                }
                s *= scale;
                Ph[(std::size_t)(i * T + j)] = s;
                if (s > mx) { mx = s; }
            }
            float sum = 0;
            for (int j = 0; j < T; j++) {
                const float e = std::exp(Ph[(std::size_t)(i * T + j)] - mx);
                Ph[(std::size_t)(i * T + j)] = e;
                sum += e;
            }
            const float inv = 1.0f / (sum + 1e-30f);
            for (int j = 0; j < T; j++) { Ph[(std::size_t)(i * T + j)] *= inv; }
        }
        for (int i = 0; i < T; i++) {
            for (int c = 0; c < dk; c++) {
                float acc = 0;
                for (int j = 0; j < T; j++) {
                    acc += Ph[(std::size_t)(i * T + j)] * V[ix(cf, j, h, c)];
                }
                Out[ix(cf, i, h, c)] = acc;
            }
        }
    }
}

void scalarBwd(const Cfg &cf, const std::vector<float> &Q, const std::vector<float> &K,
               const std::vector<float> &V, const std::vector<float> &P,
               const std::vector<float> &dOut, std::vector<float> &dQ,
               std::vector<float> &dK, std::vector<float> &dV)
{
    const int T = cf.T, tokDim = cf.tokDim, heads = cf.heads, dk = cf.dk;
    const float scale = 1.0f / std::sqrt((float)dk);
    std::fill(dQ.begin(), dQ.end(), 0.0f);
    std::fill(dK.begin(), dK.end(), 0.0f);
    std::fill(dV.begin(), dV.end(), 0.0f);
    std::vector<float> dS((std::size_t)(T * T), 0.0f);
    for (int h = 0; h < heads; h++) {
        const float *Ph = &P[(std::size_t)(h * T * T)];
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float acc = 0;
                for (int c = 0; c < dk; c++) {
                    acc += dOut[ix(cf, i, h, c)] * V[ix(cf, j, h, c)];
                }
                dS[(std::size_t)(i * T + j)] = acc;
            }
            float dot = 0;
            for (int j = 0; j < T; j++) {
                dot += Ph[(std::size_t)(i * T + j)] * dS[(std::size_t)(i * T + j)];
            }
            for (int j = 0; j < T; j++) {
                dS[(std::size_t)(i * T + j)] =
                    Ph[(std::size_t)(i * T + j)] * (dS[(std::size_t)(i * T + j)] - dot);
            }
        }
        for (int j = 0; j < T; j++) {
            for (int c = 0; c < dk; c++) {
                float accV = 0, accK = 0;
                for (int i = 0; i < T; i++) {
                    const float pij = Ph[(std::size_t)(i * T + j)];
                    accV += pij * dOut[ix(cf, i, h, c)];
                    accK += dS[(std::size_t)(i * T + j)] * Q[ix(cf, i, h, c)];
                }
                dV[ix(cf, j, h, c)] += accV;
                dK[ix(cf, j, h, c)] += accK * scale;
            }
        }
        for (int i = 0; i < T; i++) {
            for (int c = 0; c < dk; c++) {
                float acc = 0;
                for (int j = 0; j < T; j++) {
                    acc += dS[(std::size_t)(i * T + j)] * K[ix(cf, j, h, c)];
                }
                dQ[ix(cf, i, h, c)] += acc * scale;
            }
        }
    }
}

/* ============================================================
 *  [B] 矩阵化实现: 每头连续 (T×dk) + `MM::` 四内核
 * ============================================================ */
class MatrixAttn
{
public:
    explicit MatrixAttn(const Cfg &cf) : c(cf)
    {
        const std::size_t td = (std::size_t)(c.T * c.dk);
        const std::size_t tt = (std::size_t)(c.T * c.T);
        Qh = Tensor(c.T, c.dk);  Kh = Tensor(c.T, c.dk);  Vh = Tensor(c.T, c.dk);
        S  = Tensor(c.T, c.T);   P  = Tensor(c.T, c.T);
        Oh = Tensor(c.T, c.dk);
        dOh = Tensor(c.T, c.dk); dPh = Tensor(c.T, c.T); dSh = Tensor(c.T, c.T);
        dQh = Tensor(c.T, c.dk); dKh = Tensor(c.T, c.dk); dVh = Tensor(c.T, c.dk);
        (void)td; (void)tt;
    }

    /* gather/scatter 是"交错 → 按头连续"的搬运; M2 读数把它排除在计时之外 */
    void gather(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V, int h)
    {
        for (int t = 0; t < c.T; t++) {
            for (int k = 0; k < c.dk; k++) {
                const std::size_t dst = (std::size_t)(t * c.dk + k);
                Qh[dst] = Q[ix(c, t, h, k)];
                Kh[dst] = K[ix(c, t, h, k)];
                Vh[dst] = V[ix(c, t, h, k)];
            }
        }
    }
    void gatherDOut(const std::vector<float> &dOut, int h)
    {
        for (int t = 0; t < c.T; t++) {
            for (int k = 0; k < c.dk; k++) {
                dOh[(std::size_t)(t * c.dk + k)] = dOut[ix(c, t, h, k)];
            }
        }
    }
    void scatterGrad(std::vector<float> &dQ, std::vector<float> &dK, std::vector<float> &dV, int h)
    {
        for (int t = 0; t < c.T; t++) {
            for (int k = 0; k < c.dk; k++) {
                const std::size_t src = (std::size_t)(t * c.dk + k);
                dQ[ix(c, t, h, k)] += dQh[src];
                dK[ix(c, t, h, k)] += dKh[src];
                dV[ix(c, t, h, k)] += dVh[src];
            }
        }
    }

    /* fwd:  S = Qh·Khᵀ·scale → 逐行 softmax → Oh = P·Vh
       ⚠ `MM::*` 四个内核**一律累加**（方案 Part 0 的硬约束: `z += …`, 不是赋值）
       ⇒ 每次调用前必须把输出清零。本探针第一版就是栽在这上面: 数值对比里
       Out 的最大差到了 1.4e-1（相对 5e3）—— 逐头累积把上一次的结果也加进去了。
       计数里也包含 `zero()` 的成本（真实落地时同样要付）。 */
    void fwd(int h)
    {
        const float scale = 1.0f / std::sqrt((float)c.dk);
        S.zero();
        Tensor::MM::ikjk(S, Qh, Kh);                 /* S(i,j) += Σ_c Qh(i,c)·Kh(j,c) */
        for (int i = 0; i < c.T; i++) {
            float mx = -1e30f;
            for (int j = 0; j < c.T; j++) {
                const float s = S[(std::size_t)(i * c.T + j)] * scale;
                S[(std::size_t)(i * c.T + j)] = s;
                if (s > mx) { mx = s; }
            }
            float sum = 0;
            for (int j = 0; j < c.T; j++) {
                const float e = std::exp(S[(std::size_t)(i * c.T + j)] - mx);
                P[(std::size_t)(i * c.T + j)] = e;
                sum += e;
            }
            const float inv = 1.0f / (sum + 1e-30f);
            for (int j = 0; j < c.T; j++) { P[(std::size_t)(i * c.T + j)] *= inv; }
        }
        Oh.zero();
        Tensor::MM::ikkj(Oh, P, Vh);                 /* Oh(i,c) += Σ_j P(i,j)·Vh(j,c) */
    }

    /* bwd: dPh = dOh·Vhᵀ → softmax 雅可比 → dVh = Pᵀ·dOh, dQh = dSh·Kh, dKh = dShᵀ·Qh */
    void bwd(int h)
    {
        const float scale = 1.0f / std::sqrt((float)c.dk);
        dPh.zero();
        Tensor::MM::ikjk(dPh, dOh, Vh);              /* dPh(i,j) += Σ_c dOh(i,c)·Vh(j,c) */
        for (int i = 0; i < c.T; i++) {
            float dot = 0;
            for (int j = 0; j < c.T; j++) {
                dot += P[(std::size_t)(i * c.T + j)] * dPh[(std::size_t)(i * c.T + j)];
            }
            for (int j = 0; j < c.T; j++) {
                const std::size_t k = (std::size_t)(i * c.T + j);
                dSh[k] = P[k] * (dPh[k] - dot);
            }
        }
        dVh.zero();
        Tensor::MM::kikj(dVh, P, dOh);               /* dVh(j,c) += Σ_i P(i,j)·dOh(i,c) */
        dQh.zero();
        Tensor::MM::ikkj(dQh, dSh, Kh);              /* dQh(i,c) += Σ_j dSh(i,j)·Kh(j,c) */
        dKh.zero();
        Tensor::MM::kikj(dKh, dSh, Qh);              /* dKh(j,c) += Σ_i dSh(i,j)·Qh(i,c) */
        for (std::size_t i = 0; i < dQh.size(); i++) { dQh[i] *= scale; dKh[i] *= scale; }
    }

    const Cfg &c;
    Tensor Qh, Kh, Vh, S, P, Oh, dOh, dPh, dSh, dQh, dKh, dVh;
};

/* ---------------- 对照通道（固定工作量; 它的跨度就是这一轮的可信度） ----------------
   刻意做成与"被测量"同量级（~1.7 M 个 float 的拷贝 + 跨步归约, 和一次注意力的规模相当）:
   太小的对照通道只会读到计时器粒度（第一版 2~3 µs 时跨度 2.07×, 那是噪声不是通道）。 */
double controlChannel(const std::vector<float> &a, std::vector<float> &b)
{
    const double t0 = nowMs();
    std::copy(a.begin(), a.end(), b.begin());
    float s = 0;
    for (std::size_t i = 0; i < b.size(); i += 7) { s += b[i]; }
    volatile float keep = s;
    (void)keep;
    return nowMs() - t0;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} /* namespace */

int main()
{
    const Cfg cf;
    std::printf("=== P6-前哨: 注意力 标量 vs 矩阵化 (T=%d, tokDim=%d, heads=%d, dk=%d) ===\n",
                cf.T, cf.tokDim, cf.heads, cf.dk);
    std::printf("口径: 同进程/同轮交替 A/B、多轮取中位数比值、带对照通道 (§5.1); "
                "计时区内零分配; 数值差不要求逐位相同\n\n");

    /* 输入用**本探针自己的** mt19937（不碰 RL::Random: 探针只依赖 rl/tensor.hpp,
       而且"两次跑同一批输入"靠这个局部种子就够了） */
    std::mt19937 rng(20240901u);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    const std::size_t n = (std::size_t)(cf.T * cf.tokDim);
    std::vector<float> Q(n), K(n), V(n), dOut(n);
    for (std::size_t i = 0; i < n; i++) { Q[i] = ud(rng); K[i] = ud(rng); V[i] = ud(rng); dOut[i] = ud(rng); }

    std::vector<float> P((std::size_t)(cf.heads * cf.T * cf.T), 0.0f);
    std::vector<float> Out(n, 0.0f);
    std::vector<float> dQ(n, 0.0f), dK(n, 0.0f), dV(n, 0.0f);

    /* 矩阵化: 预分配 (计时区里不分配) */
    MatrixAttn mx(cf);
    std::vector<float> Pm((std::size_t)(cf.heads * cf.T * cf.T), 0.0f);
    std::vector<float> Outm(n, 0.0f);
    std::vector<float> dQm(n, 0.0f), dKm(n, 0.0f), dVm(n, 0.0f);

    /* ---------- 先比数值（不等价就谈不上"可行"） ---------- */
    scalarFwd(cf, Q, K, V, P, Out);
    scalarBwd(cf, Q, K, V, P, dOut, dQ, dK, dV);
    for (int h = 0; h < cf.heads; h++) {
        mx.gather(Q, K, V, h);
        mx.fwd(h);
        for (int t = 0; t < cf.T; t++) {
            for (int c = 0; c < cf.dk; c++) {
                Outm[ix(cf, t, h, c)] = mx.Oh[(std::size_t)(t * cf.dk + c)];
            }
        }
        for (int i = 0; i < cf.T; i++) {
            for (int j = 0; j < cf.T; j++) {
                Pm[(std::size_t)(h * cf.T * cf.T + i * cf.T + j)] = mx.P[(std::size_t)(i * cf.T + j)];
            }
        }
    }
    for (int h = 0; h < cf.heads; h++) {
        mx.gather(Q, K, V, h);
        mx.gatherDOut(dOut, h);
        mx.fwd(h);          /* 重新算一遍 P/Oh（fwd 会覆盖内部缓冲） */
        mx.bwd(h);
        mx.scatterGrad(dQm, dKm, dVm, h);
    }

    auto maxDiff = [](const std::vector<float> &a, const std::vector<float> &b) {
        double md = 0, mr = 0;
        for (std::size_t i = 0; i < a.size(); i++) {
            const double d = std::fabs((double)a[i] - (double)b[i]);
            if (d > md) { md = d; }
            const double den = std::fabs((double)a[i]) + 1e-12;
            if (d / den > mr) { mr = d / den; }
        }
        return std::make_pair(md, mr);
    };
    {
        const auto o = maxDiff(Out, Outm);
        printf("[数值] Out  : max|Δ| = %.3e, max 相对 = %.3e\n", o.first, o.second);
        const auto q = maxDiff(dQ, dQm);
        printf("[数值] dQ   : max|Δ| = %.3e, max 相对 = %.3e\n", q.first, q.second);
        const auto k = maxDiff(dK, dKm);
        printf("[数值] dK   : max|Δ| = %.3e, max 相对 = %.3e\n", k.first, k.second);
        const auto v = maxDiff(dV, dVm);
        printf("[数值] dV   : max|Δ| = %.3e, max 相对 = %.3e\n", v.first, v.second);
        double pd = 0;
        for (std::size_t i = 0; i < P.size(); i++) { pd = std::max(pd, std::fabs((double)P[i] - (double)Pm[i])); }
        printf("[数值] P    : max|Δ| = %.3e  (按行 softmax)\n\n", pd);
    }

    /* ---------- 计时: 同轮交替, 多轮取中位数 ---------- */
    const int rounds = 9;
    std::vector<double> sf, sb, mf1, mb1, mf2, mb2, ctl;
    std::vector<float> ctlA((std::size_t)(1710 * 1000), 1.0f), ctlB(ctlA.size(), 0.0f);
    for (int r = 0; r < rounds; r++) {
        ctl.push_back(controlChannel(ctlA, ctlB));

        /* A 臂: 标量 */
        double t0 = nowMs();
        scalarFwd(cf, Q, K, V, P, Out);
        double t1 = nowMs();
        scalarBwd(cf, Q, K, V, P, dOut, dQ, dK, dV);
        double t2 = nowMs();
        sf.push_back(t1 - t0); sb.push_back(t2 - t1);

        /* B1 臂: 矩阵化 + gather/scatter */
        t0 = nowMs();
        for (int h = 0; h < cf.heads; h++) { mx.gather(Q, K, V, h); mx.fwd(h); }
        t1 = nowMs();
        for (int h = 0; h < cf.heads; h++) {
            mx.gather(Q, K, V, h); mx.gatherDOut(dOut, h); mx.fwd(h); mx.bwd(h);
            mx.scatterGrad(dQm, dKm, dVm, h);
        }
        t2 = nowMs();
        mf1.push_back(t1 - t0); mb1.push_back(t2 - t1);

        /* B2 臂: 矩阵化, **不含** gather/scatter（假设布局本来就按头连续） */
        t0 = nowMs();
        for (int h = 0; h < cf.heads; h++) { mx.fwd(h); }
        t1 = nowMs();
        for (int h = 0; h < cf.heads; h++) { mx.fwd(h); mx.bwd(h); }
        t2 = nowMs();
        mf2.push_back(t1 - t0); mb2.push_back(t2 - t1);
    }

    const double cMin = *std::min_element(ctl.begin(), ctl.end());
    const double cMax = *std::max_element(ctl.begin(), ctl.end());
    printf("[对照通道] %.3f ~ %.3f ms (跨度 %.2fx)  ← 跨度大就说明这一轮不可信\n\n",
           cMin, cMax, cMax / std::max(1e-9, cMin));

    printf("                forward(ms)   backward(ms)\n");
    printf("  A 标量(现状)   %8.3f      %8.3f\n", median(sf), median(sb));
    printf("  B1 矩阵+gather %8.3f      %8.3f\n", median(mf1), median(mb1));
    printf("  B2 矩阵(无搬运)%8.3f      %8.3f\n", median(mf2), median(mb2));
    printf("\n  加速比 (A/B, 越大越好):\n");
    printf("    forward  : B1 %.2fx   B2 %.2fx\n", median(sf) / std::max(1e-9, median(mf1)),
                                                   median(sf) / std::max(1e-9, median(mf2)));
    printf("    backward : B1 %.2fx   B2 %.2fx\n", median(sb) / std::max(1e-9, median(mb1)),
                                                   median(sb) / std::max(1e-9, median(mb2)));
    printf("    fwd+bwd  : B1 %.2fx   B2 %.2fx\n",
           (median(sf) + median(sb)) / std::max(1e-9, median(mf1) + median(mb1)),
           (median(sf) + median(sb)) / std::max(1e-9, median(mf2) + median(mb2)));
    printf("\n读法: 方案 §1.5 实测「注意力 8 次 = 6.303 ms = forward 的 39%%」是在**旧机器**上量的;\n"
           "      本探针给的是**同机同轮的比值**, 把这个比值乘进那 39%% 就是矩阵化的上限收益。\n");
    return 0;
}
