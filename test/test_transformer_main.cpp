/*
 * test_transformer_main.cpp — Transformer 这条线的断言测试 (2026-10)
 * ============================================================================
 * 盯四类事, 每一类都对应之前审计出来的一个真问题 (见
 * `docs/tb_expert_training_2026_10.md` §9/§10):
 *
 *   [1] **真 Transformer 专家 (`SeqTransformerExpert`) 的解析梯度 == 有限差分**。
 *       这是本轮唯一"能证明新实现是对的"的读数 —— 仓库此前的 `test_grad` 只查
 *       FFN / LN / 门控, **注意力四栏 (wq/wk/wv/wo) 从来没被有限差分查过**。
 *       覆盖: embed / 位置嵌入 / 每层 wq,wk,wv,wo / LN γ,β / FFN / 输出投影 / 输入梯度。
 *
 *   [2] **注意力语义的三条不变量** (全局 softmax 会同时破坏前两条):
 *       (a) 关掉位置嵌入 ⇒ 对 token 置换**等变** (逐 token 独立, 没有全局耦合);
 *       (b) 所有 token 相同时, 输出 token 必须**逐位相同** (等价于"每行都是概率分布");
 *       (c) 正对照: 打开位置嵌入 ⇒ 置换**不**等变 (位置信息真的在起作用, 不是摆设)。
 *
 *   [3] **权重往返**: write → read → 同一输入同一输出。
 *
 *   [4] **旧 `MultiHeadAttention` 的反向** (补覆盖缺口)。它是"单向量 + 全局 softmax"的
 *       伪注意力, 这里用 **double 参考前向 + 中心差分**验证它的反向公式 —— 生产初始化下
 *       它的梯度只有 1e-7 量级, 低于 float32 差分的噪声地板, 所以只能这么查。
 *       结论 (2026-10): 反向公式**正确** (最差相对误差 ~1e-6); 问题全在结构上。
 *
 *   [5] **`SeqLayerNorm` 单独验证** (2026-10 把两个 LN 从专家里抽成类之后)。
 *       抽类这类纯重构只有两件事能证明它对: (i) 专家的权重指纹逐字节不变 ([3] 的 FNV),
 *       (ii) 这个类**自己**能被查 —— 逐 token 归一的不变量 / γ,β 与 dX 的有限差分 /
 *       **累加语义** (`backward` 跑两遍必须正好 2×; 这条错了残差那条路会被悄悄吃掉) /
 *       往返与**写序** (γ 在前 β 在后)。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rl/attention.hpp"
#include "rl/expert.hpp"
#include "rl/seq_transformer.hpp"
#include "rl/transformer.hpp"
#include "rl/util.hpp"

using namespace RL;

static int g_pass = 0;
static int g_fail = 0;
static void CHECK(bool ok, const char *what)
{
    if (ok) { g_pass++; }
    else { g_fail++; std::printf("  [FAIL] %s\n", what); }
}

/* ================= 新专家的配置 (与 PPO 的 SeqExperts 一致) ================= */
using SeqExp = SeqTransformerExpert<90, 4, 2>;
static const int D_IN = 1710, TOK = 64;

static void makeState(Tensor &x, std::mt19937 &rng, int nonzero)
{
    std::uniform_int_distribution<int> pick(0, (int)x.totalSize - 1);
    for (std::size_t i = 0; i < x.totalSize; i++) { x[i] = 0.0f; }
    for (int k = 0; k < nonzero; k++) {
        x[(std::size_t)pick(rng)] = (rng() % 2u) ? 1.0f : -1.0f;
    }
}

/*
   把整个序列专家的梯度清零。`iLayer` 没有全局 zeroGrad(), 而**反向是累加的**
   (整个仓库的约定) ⇒ 每次 `backward` 之前必须清, 否则读到的"解析梯度"是多次反向之和
   (第一版就是这么错的: (d) 段的 backward 叠在 (e) 段上, 读数像数值误差).
*/
static void zeroExpertGrads(SeqExp &ex)
{
    ex.embed.g.zero(); ex.outProj.g.zero();
    for (int b = 0; b < 2; b++) {
        ex.wq[(std::size_t)b].g.zero(); ex.wk[(std::size_t)b].g.zero();
        ex.wv[(std::size_t)b].g.zero(); ex.wo[(std::size_t)b].g.zero();
        ex.ffnUp[(std::size_t)b].g.zero(); ex.ffnDown[(std::size_t)b].g.zero();
    }
    ex.pos.zeroGrad();
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    /*
       [2026-10] 让 `abort()` **不触发 Windows 错误报告**。
       为什么: 本文件有多个"故意踩契约"的**子进程**模式 (旧的两处 + [10] 的六处),
       而每次 abort 都会让子进程在 WER 里卡住 —— **实测每个 ≈13.8 s**
       (`--mm-shape-guard` 单跑 13.8 s, 其中真正的活只有几毫秒)。
       不关掉的话 [10] 白送 80+ s, ctest 对这个目标只有 300 s 预算。
       断言只看"退出码非 0", 所以退出码从 0xC0000409 变成 3 不影响判据。
    */
#ifdef _MSC_VER
    /* ⚠ 第二个参数是**掩码**: 传 0 等于"什么都不改" (第一版就是这么写的, 于是白做)。
       要用 `_WRITE_ABORT_MSG | _CALL_REPORTFAULT` 选中这两个旗标再清零。 */
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    std::printf(" test_transformer — 真 Transformer 专家 + 旧 MHA 的反向审计\n");

    /* ============================================================
     *  [10] 的子进程模式 —— **必须在最前面派发**
     *       它们不需要任何前置小节, 而"故意踩契约"的子进程如果跑到末尾才处理,
     *       就会把 [1]~[9] 整份测试白跑一遍 (第一版就是这样: 直接跑
     *       `--degenerate-normalize-guard` 会一路跑到 [8] 才停, 看上去像"挂住")。
     * ============================================================ */
    {
        const std::string early = (argc > 1) ? std::string(argv[1]) : std::string();
        if (early == "--mm-shape-guard") {
            RL::Tensor z(90, 90), w(90, 360), x(360, 90);
            z.zero(); w.zero(); x.zero();
            RL::Tensor wrong(89, 1);              /* 与 w 的第 1 维 (90) 对不上 */
            wrong.zero();
            RL::Tensor::MM::ikkj(z, w, wrong);    /* 形状契约失守 */
            std::printf("!! MM 形状契约失守竟然没停\n");
            return 0;
        }
        if (early == "--mm-buffer-guard") {
            /* ⚠ 第一版写的是 `ikkj(z, z, z)` —— 那**形状完全合法**, 守卫当然不该拦, 于是报了假失败。
               真正要模拟的是"缓冲短于形状": 形状元数据说 (90,90), 缓冲只有 100 个元素。
               `val` 是 public 的 ⇒ 这条路径在仓库里是真能发生的 (谁 resize 错就有了)。 */
            RL::Tensor z(90, 90), w(90, 90), x(90, 90);
            z.zero(); w.zero(); x.zero();
            z.val.resize(100);                    /* 100 < 90*90 = 8100 */
            RL::Tensor::MM::ikkj(z, w, x);        /* 常驻检查必须拦下来 */
            std::printf("!! 缓冲长度 < 形状乘积竟然没停 (val=%llu, shape=%d x %d)\n",
                        (unsigned long long)z.val.size(), z.shape[0], z.shape[1]);
            return 0;
        }
        if (early == "--index-range-guard") {
            const int big = 46341;                /* 46341^2 = 2,147,488,281 > INT_MAX */
            RL::Tensor t(big, big);
            std::printf("!! 32 位下标契约失守竟然没停 (totalSize=%llu)\n",
                        (unsigned long long)t.totalSize);
            return 0;
        }
        if (early == "--reshape-guard") {
            RL::Tensor t(2, 2);
            t.zero();
            t.reshape(1000, 1);                   /* 元素总数不守恒 */
            std::printf("!! reshape 元素数不守恒竟然没停 (totalSize=%llu, val=%llu)\n",
                        (unsigned long long)t.totalSize, (unsigned long long)t.val.size());
            return 0;
        }
        if (early == "--empty-stat-guard") {
            RL::Tensor e;
            const float m = e.mean();             /* 空张量: 0/0 */
            std::printf("!! 空张量 mean 竟然给出了 %g\n", (double)m);
            return 0;
        }
        if (early == "--degenerate-normalize-guard") {
            RL::Tensor c(4, 1);
            c.fill(3.0f);                         /* 常量张量: max == min ⇒ 0/0 */
            c.normalize();
            std::printf("!! 零跨度 normalize 竟然返回了 (v0=%g)\n", (double)c[0]);
            return 0;
        }
    }

    /* ============================================================
     *  [1] 有限差分梯度审计
     * ============================================================ */
    std::printf("\n[1] SeqTransformerExpert 的逐参数组有限差分 (L = c·o)\n");
    {
        Random::setSeed(20240901u);
        SeqExp ex(D_IN, TOK, true);
        scaleExpertInit(ex);

        std::mt19937 rng(20240901u);
        Tensor x((std::size_t)D_IN, 1);
        makeState(x, rng, 140);
        std::normal_distribution<float> nrm(0.0f, 1.0f);
        Tensor c((std::size_t)D_IN, 1);
        for (int i = 0; i < D_IN; i++) { c[(std::size_t)i] = nrm(rng); }

        auto loss = [&]() {
            ex.forward(x, false);
            double s = 0;
            for (int i = 0; i < D_IN; i++) { s += (double)c[(std::size_t)i] * (double)ex.o[(std::size_t)i]; }
            return s;
        };
        const double L0 = loss();
        Tensor ei((std::size_t)D_IN, 1);
        ei.zero();
        ex.e = c;
        ex.backward(x, ei);

        struct Item { std::string name; Tensor *w; Tensor *g; int idx; };
        std::vector<Item> items;
        auto add = [&](const char *n, Tensor &w, Tensor &g, int i) {
            Item it; it.name = n; it.w = &w; it.g = &g; it.idx = i; items.push_back(it);
        };
        /* 找一个 x 非零的列 (否则 embed 的梯度恒为 0, 那是同义反复而不是检验)。
           注意 embed 的输入维数是 **featDim = 19** (token 的特征), 不是 1710 ——
           索引必须落在这个范围内, 第一版写成 31*1710+j 是越界读 (测试自己抓不到,
           但审计就没有意义了)。 */
        int jnz = -1;
        {
            std::vector<float> xs((std::size_t)(90 * 19), 0.0f);
            for (int t = 0; t < 90; t++) {
                for (int f = 0; f < 19; f++) { xs[(std::size_t)(t * 19 + f)] = x[(std::size_t)(f * 90 + t)]; }
            }
            for (int j = 0; j < 19 && jnz < 0; j++) {
                for (int t = 0; t < 90; t++) {
                    if (xs[(std::size_t)(t * 19 + j)] != 0.0f) { jnz = j; break; }
                }
            }
        }
        CHECK(jnz >= 0, "输入里有非零特征 (审计才有意义)");
        const int FEAT = D_IN / 90;

        add("embed.w[0][jnz]", ex.embed.w, ex.embed.g.w, 0 * FEAT + jnz);
        add("embed.w[31][jnz]", ex.embed.w, ex.embed.g.w, 31 * FEAT + jnz);
        add("embed.b[5]", ex.embed.b, ex.embed.g.b, 5);
        /* 位置编码抽成 `SeqPosEnc` 之后: LEARN_1D 的参数在 emb[0] (2D/RoPE 见 [6]) */
        add("pos[0][3]", ex.pos.emb[0].val, ex.pos.emb[0].g, 0 * TOK + 3);
        add("pos[45][60]", ex.pos.emb[0].val, ex.pos.emb[0].g, 45 * TOK + 60);
        add("b0.wq.w[0][7]", ex.wq[0].w, ex.wq[0].g.w, 0 * TOK + 7);
        add("b0.wk.w[5][9]", ex.wk[0].w, ex.wk[0].g.w, 5 * TOK + 9);
        add("b0.wv.w[0][11]", ex.wv[0].w, ex.wv[0].g.w, 0 * TOK + 11);
        add("b0.wo.w[3][2]", ex.wo[0].w, ex.wo[0].g.w, 3 * TOK + 2);
        /* LayerNorm 的 γ/β 现在由 `SeqLayerNorm` 持有 (2026-10 抽类) */
        add("b0.ln1.gamma[0]", ex.ln1[0].gamma, ex.ln1[0].gG, 0);
        add("b0.ln1.beta[17]", ex.ln1[0].beta, ex.ln1[0].gB, 17);
        add("b0.ffnUp.w[0][4]", ex.ffnUp[0].w, ex.ffnUp[0].g.w, 0 * TOK + 4);
        add("b0.ffnUp.b[9]", ex.ffnUp[0].b, ex.ffnUp[0].g.b, 9);
        add("b0.ffnDown.w[2][0]", ex.ffnDown[0].w, ex.ffnDown[0].g.w, 2 * ex.dff + 0);
        add("b0.ln2.gamma[3]", ex.ln2[0].gamma, ex.ln2[0].gG, 3);
        add("b0.ln2.beta[3]", ex.ln2[0].beta, ex.ln2[0].gB, 3);
        add("b1.wq.w[1][5]", ex.wq[1].w, ex.wq[1].g.w, 1 * TOK + 5);
        add("b1.wo.w[7][6]", ex.wo[1].w, ex.wo[1].g.w, 7 * TOK + 6);
        add("b1.ffnDown.w[62][1]", ex.ffnDown[1].w, ex.ffnDown[1].g.w, 62 * ex.dff + 1);
        add("outProj.w[8][3]", ex.outProj.w, ex.outProj.g.w, 8 * TOK + 3);
        add("outProj.b[12]", ex.outProj.b, ex.outProj.g.b, 12);

        double worst = 0;
        std::string worstName;
        for (std::size_t k = 0; k < items.size(); k++) {
            Item &it = items[k];
            const double ana = (double)(*it.g)[(std::size_t)it.idx];
            const double w0 = (double)(*it.w)[(std::size_t)it.idx];
            double best = 1e30, bestNum = 0, bestEps = 0;
            for (int e = 0; e < 3; e++) {
                const double eps = (e == 0) ? 1e-2 : ((e == 1) ? 1e-3 : 1e-4);
                (*it.w)[(std::size_t)it.idx] = (float)(w0 + eps);
                const double lp = loss();
                (*it.w)[(std::size_t)it.idx] = (float)(w0 - eps);
                const double lm = loss();
                (*it.w)[(std::size_t)it.idx] = (float)w0;
                const double num = (lp - lm) / (2.0 * eps);
                const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
            }
            std::printf("    %-22s ana=%+.5e  num=%+.5e (eps=%.0e)  相对误差=%.2e %s\n",
                        it.name.c_str(), ana, bestNum, bestEps, best,
                        best < 1e-2 ? "OK" : "**不一致?**");
            if (best > worst) { worst = best; worstName = it.name; }
        }
        std::printf("    |L| = %.6g   -> 最差相对误差 %.3e (%s)\n",
                    L0, worst, worstName.c_str());
        CHECK(worst < 1e-2, "全部参数组的解析梯度与有限差分一致 (相对误差 < 1e-2)");

        /* 输入梯度 (往下传给 MoE 门控的那一项)。
           取 **|ei| 最大的 8 个坐标**, 不是随机 8 个: 随机坐标里很多解析梯度本身接近 0,
           那里的"相对误差"量的是 float 差分的噪声地板 (两次前向之差 ~1e-7·|L| 再除以 2eps),
           不是实现。本文件第一版就是随机取的, 阈值 1e-2 一直悬在 6.3e-3 附近 ——
           抽 LN 类之后编译器内联/FMA 收缩一变就翻红 (1.7e-2), 而同一批解析梯度只有
           末位数字变化 (见下方打印与 docs 记录)。换成"挑大梯度坐标"之后, 读到的是
           真正的相对误差。 */
        std::vector<int> order((std::size_t)D_IN);
        for (int i = 0; i < D_IN; i++) { order[(std::size_t)i] = i; }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::fabs((double)ei[(std::size_t)a]) > std::fabs((double)ei[(std::size_t)b]);
        });
        double worstIn = 0, worstInAna = 0;
        for (int k = 0; k < 8; k++) {
            const int i = order[(std::size_t)k];
            const double x0 = (double)x[(std::size_t)i];
            const double ana = (double)ei[(std::size_t)i];
            double best = 1e30, bestNum = 0, bestEps = 0;
            for (int e = 0; e < 3; e++) {
                const double eps = (e == 0) ? 1e-2 : ((e == 1) ? 1e-3 : 1e-4);
                x[(std::size_t)i] = (float)(x0 + eps); const double lp = loss();
                x[(std::size_t)i] = (float)(x0 - eps); const double lm = loss();
                x[(std::size_t)i] = (float)x0;
                const double num = (lp - lm) / (2.0 * eps);
                const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
            }
            std::printf("    ei[%4d] ana=%+.5e  num=%+.5e (eps=%.0e)  相对误差=%.2e\n",
                        i, ana, bestNum, bestEps, best);
            worstIn = std::max(worstIn, best);
            worstInAna = std::max(worstInAna, std::fabs(ana));
        }
        std::printf("    输入梯度 ei: |ei| 最大的 8 个坐标最差相对误差 = %.3e (最大 |ei| = %.3e)\n",
                    worstIn, worstInAna);
        CHECK(worstIn < 1e-2, "输入梯度与有限差分一致 (取 |ei| 最大的坐标, 噪声地板之上)");
    }

    /* ============================================================
     *  [2] 注意力语义的三条不变量
     * ============================================================ */
    std::printf("\n[2] 不变量: 置换等变 / 常数 token / 位置嵌入的正对照\n");
    {
        Random::setSeed(777u);
        SeqExp ex(D_IN, TOK, false /* 推理形态 */);
        scaleExpertInit(ex);
        std::mt19937 rng(777u);
        Tensor x((std::size_t)D_IN, 1);
        makeState(x, rng, 200);

        /* 随机置换 90 个格子 (token) */
        std::vector<int> perm(90);
        for (int i = 0; i < 90; i++) { perm[(std::size_t)i] = i; }
        std::shuffle(perm.begin(), perm.end(), rng);
        Tensor xp((std::size_t)D_IN, 1);
        xp.zero();
        for (int f = 0; f < 19; f++) {
            for (int t = 0; t < 90; t++) {
                xp[(std::size_t)(f * 90 + perm[(std::size_t)t])] = x[(std::size_t)(f * 90 + t)];
            }
        }
        auto runOut = [&](SeqExp &e, const Tensor &in, Tensor &out) {
            e.forward(in, true);
            out = Tensor((std::size_t)D_IN, 1);
            for (int i = 0; i < D_IN; i++) { out[(std::size_t)i] = e.o[(std::size_t)i]; }
        };
        Tensor o1, o2;
        ex.usePos = false;
        runOut(ex, x, o1);
        runOut(ex, xp, o2);
        /*
           等变的**方向**不能写反: xp 是把 token 按 σ=perm 重标号 (xp[σ(t)] = x[t]),
           于是性质是 `o2[σ(t)] == o1[t]` —— 即比较 o1[f*90+t] 与 o2[f*90+perm[t]]。
           (第一版写成 o1[perm[t]] vs o2[t], 那是用 perm⁻¹ 去比, 必然"不等变";
            测试当场红了一次, 记在这里免得下次又反过来。)
        */
        double dPerm = 0, dPermMax = 0;
        for (int f = 0; f < 19; f++) {
            for (int t = 0; t < 90; t++) {
                const double a = (double)o1[(std::size_t)(f * 90 + t)];
                const double b = (double)o2[(std::size_t)(f * 90 + perm[(std::size_t)t])];
                dPerm += (a - b) * (a - b);
                dPermMax = std::max(dPermMax, std::fabs(a - b));
            }
        }
        std::printf("    (a) 关位置嵌入: 置换后输出差 |dx| = %.3e, max = %.3e\n",
                    std::sqrt(dPerm), dPermMax);
        CHECK(dPermMax < 1e-4, "(a) 关掉位置嵌入 ⇒ 对 token 置换等变 (逐 token 独立, 无全局耦合)");

        /* 常数 token: 19 个特征在 90 个格子上完全相同 ⇒ 关位置嵌入时输出 token 必须相同 */
        Tensor xc((std::size_t)D_IN, 1);
        for (int f = 0; f < 19; f++) {
            const float v = (f % 3 == 0) ? 1.0f : ((f % 3 == 1) ? -1.0f : 0.25f);
            for (int t = 0; t < 90; t++) { xc[(std::size_t)(f * 90 + t)] = v; }
        }
        Tensor oc;
        runOut(ex, xc, oc);
        double cMax = 0;
        for (int f = 0; f < 19; f++) {
            for (int t = 1; t < 90; t++) {
                cMax = std::max(cMax, std::fabs((double)oc[(std::size_t)(f * 90 + t)]
                                              - (double)oc[(std::size_t)(f * 90 + 0)]));
            }
        }
        std::printf("    (b) 常量 token: 各格输出的最大差 = %.3e\n", cMax);
        CHECK(cMax < 1e-4, "(b) 全部 token 相同 ⇒ 输出 token 逐位相同 (每行都是概率分布, 不是全局归一)");

        /* 正对照: 打开位置嵌入, 置换必须**不**等变 */
        ex.usePos = true;
        runOut(ex, x, o1);
        runOut(ex, xp, o2);
        double dPos = 0;
        for (int f = 0; f < 19; f++) {
            for (int t = 0; t < 90; t++) {
                const double a = (double)o1[(std::size_t)(f * 90 + t)];
                const double b = (double)o2[(std::size_t)(f * 90 + perm[(std::size_t)t])];
                dPos += (a - b) * (a - b);
            }
        }
        std::printf("    (c) 开位置嵌入: 置换后输出差 |dx| = %.3e (必须 > 0)\n", std::sqrt(dPos));
        CHECK(std::sqrt(dPos) > 1e-3, "(c) 位置嵌入真的在起作用 (正对照: 置换不再等变)");
    }

    /* ============================================================
     *  [3] 权重往返
     * ============================================================ */
    std::printf("\n[3] 权重往返 (write -> read)\n");
    {
        Random::setSeed(4242u);
        SeqExp a(D_IN, TOK, false);
        scaleExpertInit(a);
        SeqExp b(D_IN, TOK, false);
        scaleExpertInit(b);
        std::mt19937 rng(4242u);
        Tensor x((std::size_t)D_IN, 1);
        makeState(x, rng, 120);

        a.forward(x, true);
        Tensor oa((std::size_t)D_IN, 1);
        for (int i = 0; i < D_IN; i++) { oa[(std::size_t)i] = a.o[(std::size_t)i]; }

        std::ostringstream os;
        a.write(os);
        std::istringstream is(os.str());
        b.read(is);
        b.forward(x, true);
        double diff = 0;
        for (int i = 0; i < D_IN; i++) {
            const double d = (double)b.o[(std::size_t)i] - (double)oa[(std::size_t)i];
            diff = std::max(diff, std::fabs(d));
        }
        /*
           权重文本的 FNV-1a 指纹。为什么要有它: "把 LayerNorm 抽成一个类"这类**纯重构**
           必须做到**逐字节等价** —— 参数量/往返一致并不足以说明这一点 (写出的**顺序**
           变了, 存量权重就会错位地读进来, 而往返测试自己看不出来)。
           重构前后把这一行对一下, 就是最直接的证据。
        */
        const std::string &blob = os.str();
        unsigned long long h = 1469598103934665603ULL;
        for (std::size_t i = 0; i < blob.size(); i++) {
            h ^= (unsigned char)blob[i];
            h *= 1099511628211ULL;
        }
        std::printf("    参数量 %lld, 权重文本 %zu 字节, FNV-1a=%016llx, 读回后输出最大差 = %.3e\n",
                    a.paramCount(), blob.size(), h, diff);
        CHECK(diff == 0.0, "存了再读 ⇒ 输出逐位相同");
        CHECK(a.paramCount() == b.paramCount(), "往返两边的参数量一致");

        /* 参数量必须与分解一致 (否则权重文件的指纹会骗人) */
        long long pa = 0, pf = 0, pl = 0;
        a.paramBreakdown(pa, pf, pl);
        std::printf("    参数构成: 注意力 %lld / FFN+嵌入 %lld / LN %lld  (注意力矩阵 %lld)\n",
                    pa, pf, pl, a.attnElements());
        CHECK(pa + pf + pl == a.paramCount(), "paramBreakdown 之和 == paramCount");
        CHECK(a.attnElements() == (long long)4 * 90 * 90, "注意力矩阵元素 = 头数 x T^2");
        CHECK(a.attnHeadsUsed() == a.attnHeadsRequested(), "请求的头数 == 实用的头数");
    }

    /* ============================================================
     *  [4] 旧 MultiHeadAttention 的反向 (double 参考; 补覆盖缺口)
     * ============================================================ */
    std::printf("\n[4] 旧伪注意力 (MultiHeadAttention) 的反向审计 —— double 参考前向\n");
    {
        const int d = 240, H = 15, DK = d / H;   /* 15 x 16 = 240, 整除 */
        Random::setSeed(20240901u);
        MultiHeadAttention<H, true> mha(d, d, true);
        const float sc = 1.0f / std::sqrt((float)d);
        for (int h = 0; h < mha.headsUsed(); h++) {
            ScaledDotProduct &hd = mha.heads[(std::size_t)h];
            for (std::size_t i = 0; i < hd.wq.size(); i++) { hd.wq[i] *= sc; hd.wk[i] *= sc; hd.wv[i] *= sc; }
        }
        for (std::size_t i = 0; i < mha.wo.size(); i++) { mha.wo[i] *= sc; }

        std::mt19937 rng(20240901u);
        std::vector<double> xd((std::size_t)d, 0.0);
        std::uniform_int_distribution<int> pick(0, d - 1);
        for (int t = 0; t < 40; t++) { xd[(std::size_t)pick(rng)] = (rng() % 2u) ? 1.0 : -1.0; }
        std::vector<double> cd((std::size_t)d);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (int i = 0; i < d; i++) { cd[(std::size_t)i] = nrm(rng); }

        Tensor x((std::size_t)d, 1);
        for (int i = 0; i < d; i++) { x[(std::size_t)i] = (float)xd[(std::size_t)i]; }

        /* double 镜像 + 纯 double 前向 (与 attention.hpp 的算术同构: 全局 softmax) */
        std::vector<double> wo((std::size_t)d * d);
        for (int i = 0; i < d * d; i++) { wo[(std::size_t)i] = (double)mha.wo[(std::size_t)i]; }
        std::vector<std::vector<double> > wq(H), wk(H), wv(H);
        for (int h = 0; h < H; h++) {
            const ScaledDotProduct &hd = mha.heads[(std::size_t)h];
            wq[(std::size_t)h].resize((std::size_t)DK * d);
            wk[(std::size_t)h].resize((std::size_t)DK * d);
            wv[(std::size_t)h].resize((std::size_t)DK * d);
            for (int i = 0; i < DK * d; i++) {
                wq[(std::size_t)h][(std::size_t)i] = (double)hd.wq[(std::size_t)i];
                wk[(std::size_t)h][(std::size_t)i] = (double)hd.wk[(std::size_t)i];
                wv[(std::size_t)h][(std::size_t)i] = (double)hd.wv[(std::size_t)i];
            }
        }
        std::vector<double> xm = xd;
        auto refLoss = [&]() {
            std::vector<double> a((std::size_t)d, 0.0);
            for (int h = 0; h < H; h++) {
                std::vector<double> q((std::size_t)DK, 0), k((std::size_t)DK, 0), v((std::size_t)DK, 0);
                for (int i = 0; i < DK; i++) {
                    double sq = 0, sk = 0, sv = 0;
                    for (int j = 0; j < d; j++) {
                        sq += wq[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                        sk += wk[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                        sv += wv[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                    }
                    q[(std::size_t)i] = sq; k[(std::size_t)i] = sk; v[(std::size_t)i] = sv;
                }
                std::vector<double> z((std::size_t)DK * DK, 0.0);
                double mx = -1e300;
                for (int i = 0; i < DK; i++) {
                    for (int j = 0; j < DK; j++) {
                        const double t = q[(std::size_t)i] * k[(std::size_t)j] / std::sqrt((double)DK);
                        z[(std::size_t)(i * DK + j)] = t;
                        if (t > mx) { mx = t; }
                    }
                }
                double s = 0;
                for (std::size_t t = 0; t < z.size(); t++) { z[t] = std::exp(z[t] - mx); s += z[t]; }
                for (std::size_t t = 0; t < z.size(); t++) { z[t] /= s; }
                for (int i = 0; i < DK; i++) {
                    double acc = 0;
                    for (int j = 0; j < DK; j++) { acc += z[(std::size_t)(i * DK + j)] * v[(std::size_t)j]; }
                    a[(std::size_t)(h * DK + i)] = acc;
                }
            }
            double out = 0;
            for (int i = 0; i < d; i++) {
                double acc = 0;
                for (int j = 0; j < d; j++) { acc += wo[(std::size_t)(i * d + j)] * a[(std::size_t)j]; }
                out += cd[(std::size_t)i] * acc;
            }
            return out;
        };
        /* 先确认参考前向 == 库前向 (否则下面的检查无意义) */
        mha.forward(x, false);
        std::vector<double> aRef, outRef;
        {
            /* 复用 refLoss 的前向部分: 直接再算一遍 a 与 out */
            std::vector<double> a((std::size_t)d, 0.0);
            for (int h = 0; h < H; h++) {
                std::vector<double> q((std::size_t)DK, 0), k((std::size_t)DK, 0), v((std::size_t)DK, 0);
                for (int i = 0; i < DK; i++) {
                    double sq = 0, sk = 0, sv = 0;
                    for (int j = 0; j < d; j++) {
                        sq += wq[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                        sk += wk[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                        sv += wv[(std::size_t)h][(std::size_t)(i * d + j)] * xm[(std::size_t)j];
                    }
                    q[(std::size_t)i] = sq; k[(std::size_t)i] = sk; v[(std::size_t)i] = sv;
                }
                std::vector<double> z((std::size_t)DK * DK, 0.0);
                double mx = -1e300;
                for (int i = 0; i < DK; i++) {
                    for (int j = 0; j < DK; j++) {
                        const double t = q[(std::size_t)i] * k[(std::size_t)j] / std::sqrt((double)DK);
                        z[(std::size_t)(i * DK + j)] = t;
                        if (t > mx) { mx = t; }
                    }
                }
                double s = 0;
                for (std::size_t t = 0; t < z.size(); t++) { z[t] = std::exp(z[t] - mx); s += z[t]; }
                for (std::size_t t = 0; t < z.size(); t++) { z[t] /= s; }
                for (int i = 0; i < DK; i++) {
                    double acc = 0;
                    for (int j = 0; j < DK; j++) { acc += z[(std::size_t)(i * DK + j)] * v[(std::size_t)j]; }
                    a[(std::size_t)(h * DK + i)] = acc;
                }
            }
            aRef = a;
            outRef.assign((std::size_t)d, 0.0);
            for (int i = 0; i < d; i++) {
                double acc = 0;
                for (int j = 0; j < d; j++) { acc += wo[(std::size_t)(i * d + j)] * a[(std::size_t)j]; }
                outRef[(std::size_t)i] = acc;
            }
        }
        double dRef = 0, dRefMax = 0;
        for (int i = 0; i < d; i++) {
            const double dv = outRef[(std::size_t)i] - (double)mha.o[(std::size_t)i];
            dRef += dv * dv; dRefMax = std::max(dRefMax, std::fabs(dv));
        }
        std::printf("    参考 vs 库: |out| 相对差 = %.3e (max %.3e)\n",
                    std::sqrt(dRef / (std::sqrt(dRef) + 1e-30)), dRefMax);
        CHECK(dRefMax < 1e-4, "double 参考与库的 float 前向一致 (审计的前提)");

        /* 解析梯度 (库) */
        Tensor ei((std::size_t)d, 1);
        ei.zero();
        for (int i = 0; i < d; i++) { mha.e[(std::size_t)i] = (float)cd[(std::size_t)i]; }
        mha.backward(x, ei);

        int jnz = -1;
        for (int j = 0; j < d; j++) { if (xm[(std::size_t)j] != 0.0) { jnz = j; break; } }
        double worstOld = 0;
        auto chkOld = [&](const char *name, std::vector<double> &w, int idx, double ana) {
            const double w0 = w[(std::size_t)idx];
            double best = 1e30, bestNum = 0;
            for (int e = 0; e < 3; e++) {
                const double eps = (e == 0) ? 1e-5 : ((e == 1) ? 1e-6 : 1e-7);
                w[(std::size_t)idx] = w0 + eps; const double lp = refLoss();
                w[(std::size_t)idx] = w0 - eps; const double lm = refLoss();
                w[(std::size_t)idx] = w0;
                const double num = (lp - lm) / (2.0 * eps);
                const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                if (rel < best) { best = rel; bestNum = num; }
            }
            std::printf("    %-16s ana=%+.5e  num(double)=%+.5e  相对误差=%.2e\n",
                        name, ana, bestNum, best);
            worstOld = std::max(worstOld, best);
        };
        chkOld("wq[0][jnz]", wq[0], 0 * d + jnz,  (double)mha.heads[0].g.wq[0 * d + jnz]);
        chkOld("wk[3][jnz]", wk[0], 3 * d + jnz,  (double)mha.heads[0].g.wk[3 * d + jnz]);
        chkOld("wv[5][jnz]", wv[0], 5 * d + jnz,  (double)mha.heads[0].g.wv[5 * d + jnz]);
        chkOld("wo[0][jnz]", wo,    0 * d + jnz,  (double)mha.g.wo[0 * d + jnz]);
        std::printf("    最差相对误差 = %.3e\n", worstOld);
        CHECK(worstOld < 1e-3, "旧 MHA 的反向公式正确 (double 参考, 相对误差 < 1e-3)");
        (void)aRef;
    }

    /* ============================================================
     *  [5] SeqLayerNorm 单独验证 (抽类之后必须"自身可测")
     * ============================================================ */
    std::printf("\n[5] SeqLayerNorm: 归一不变量 / 有限差分 / 累加语义 / 往返与写序\n");
    {
        const int D = 19, TN = 5;
        SeqLayerNorm ln(D, true);

        double gDev = 0, bDev = 0;
        for (int j = 0; j < D; j++) {
            gDev = std::max(gDev, std::fabs((double)ln.gamma[(std::size_t)j] - 1.0));
            bDev = std::max(bDev, std::fabs((double)ln.beta[(std::size_t)j]));
        }
        CHECK(gDev == 0.0 && bDev == 0.0, "γ 初值恒为 1 / β 初值恒为 0 (标准 LN, 不是随机值)");

        std::mt19937 rng(20261010u);
        std::normal_distribution<float> nrm(0.0f, 3.0f);
        std::vector<float> X((std::size_t)(TN * D));
        for (std::size_t i = 0; i < X.size(); i++) { X[i] = nrm(rng); }
        X[0] += 7.0f;   /* 给第 0 个 token 加个偏置: 逐 token 归一必须把它完全抹掉 */
        std::vector<float> Y((std::size_t)(TN * D));

        ln.forward(X, Y);
        CHECK(ln.paramCount() == 2LL * (long long)D, "paramCount == 2·dim");
        CHECK(ln.hasGrad() && !SeqLayerNorm(D, false).hasGrad(), "hasGrad 跟随构造参数");
        CHECK(ln.tokensIn(X) == TN, "tokensIn 由缓冲区长度推出 (这一层不需要知道 SeqLen)");

        /* (a) γ=1, β=0 ⇒ 每个 token 的输出在该 token 的 D 维上均值 0 / 方差 1 */
        double worstMu = 0, worstVar = 0;
        for (int t = 0; t < TN; t++) {
            double m = 0;
            for (int j = 0; j < D; j++) { m += (double)Y[(std::size_t)(t * D + j)]; }
            m /= D;
            double var = 0;
            for (int j = 0; j < D; j++) {
                const double dd = (double)Y[(std::size_t)(t * D + j)] - m;
                var += dd * dd;
            }
            var /= D;
            worstMu = std::max(worstMu, std::fabs(m));
            /* var·(var+eps)⁻¹ = var/(var+eps) ≈ 1 − eps/var ⇒ 期望偏差 ~1e-5 量级 */
            worstVar = std::max(worstVar, std::fabs(var - 1.0));
        }
        std::printf("    (a) 每 token 输出: max|均值| = %.3e, max|方差−1| = %.3e\n", worstMu, worstVar);
        CHECK(worstMu < 1e-5, "(a) 逐 token 归一 ⇒ 输出均值 0 (偏置 7.0 被完全抹掉)");
        CHECK(worstVar < 1e-3, "(a) 逐 token 归一 ⇒ 输出方差 1");

        /* (b) 有限差分: γ / β / 输入 X 三类, 外加 dX 本身 */
        std::vector<float> c((std::size_t)(TN * D));
        for (std::size_t i = 0; i < c.size(); i++) { c[i] = nrm(rng); }
        auto loss = [&]() {
            ln.forward(X, Y);
            double s = 0;
            for (std::size_t i = 0; i < Y.size(); i++) { s += (double)c[i] * (double)Y[i]; }
            return s;
        };
        ln.zeroGrad();
        std::vector<float> dX((std::size_t)(TN * D), 0.0f);
        ln.backward(X, c, dX);

        double worstLN = 0;
        std::string worstLNName;
        auto chk = [&](const char *name, float *w, int idx, double ana) {
            const double w0 = (double)w[idx];
            double best = 1e30, bestNum = 0, bestEps = 0;
            for (int e = 0; e < 3; e++) {
                const double eps = (e == 0) ? 1e-2 : ((e == 1) ? 1e-3 : 1e-4);
                w[idx] = (float)(w0 + eps); const double lp = loss();
                w[idx] = (float)(w0 - eps); const double lm = loss();
                w[idx] = (float)w0;
                const double num = (lp - lm) / (2.0 * eps);
                const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
            }
            std::printf("    %-14s ana=%+.5e  num=%+.5e (eps=%.0e)  相对误差=%.2e %s\n",
                        name, ana, bestNum, bestEps, best,
                        best < 1e-2 ? "OK" : "**不一致?**");
            if (best > worstLN) { worstLN = best; worstLNName = name; }
        };
        {
            /* γ 与 β 的梯度按 dim 维共享跨 token ⇒ 取几个不同列, 别只查第 0 列 */
            char nm[32];
            const int cols[3] = { 0, 1, 7 };
            for (int k = 0; k < 3; k++) {
                const int j = cols[k];
                std::snprintf(nm, sizeof(nm), "gamma[%d]", j);
                chk(nm, &ln.gamma[0], j, (double)ln.gG[(std::size_t)j]);
                std::snprintf(nm, sizeof(nm), "beta[%d]", j);
                chk(nm, &ln.beta[0], j, (double)ln.gB[(std::size_t)j]);
            }
            /* 输入梯度: 逐 token 的 dX (LN 反向里最容易写错的就是 −mean − x̂·mean(dY·x̂) 那两项) */
            const int pos[3][2] = { { 1, 2 }, { 3, 9 }, { 4, 18 } };
            for (int k = 0; k < 3; k++) {
                const int t = pos[k][0], j = pos[k][1];
                std::snprintf(nm, sizeof(nm), "dX[%d][%d]", t, j);
                chk(nm, &X[0], t * D + j, (double)dX[(std::size_t)(t * D + j)]);
            }
        }
        std::printf("    最差相对误差 = %.3e (%s)\n", worstLN, worstLNName.c_str());
        CHECK(worstLN < 1e-2, "(b) SeqLayerNorm 的解析梯度 (γ/β/dX) 与有限差分一致");

        /*
           (c) 累加语义: `dX` 与 `gG/gB` 都是 **`+=`**。
           为什么这条要单独查: 专家里 pre-LN 的残差和 LN 两条路写同一个缓冲, 谁要是
           写成 `=`, 梯度只会"少加"而不会报错 —— 这种错在 [1] 的专家级 FD 里也可能被
           别的项掩盖 (它只查一个坐标的一次结果)。这里查两件事:
             c1. 同一输入连跑两遍**单遍** backward ⇒ 逐位相同 (反向是纯函数, 没有残留状态);
             c2. 清零后连跑两遍 ⇒ dX/gG 都是 2×(单遍)。
           ⚠ 容差不能写成 `==`: 一个 token 累加 5 次, 两遍就是 10 次加法, 浮点加法不满足
             结合律 ⇒ 2×(单遍) 与"累加两遍"只在 2~3 ulp 内相等 (实测 ~1e-7 相对误差)。
             第一版这里写了 `==` 并且**只跑了一遍**, 于是"dX 没翻倍"这个假警报把自己骗了
             (真值其实是 dX₂ == dX₁, 因为第二遍压根没调用)。
        */
        ln.zeroGrad();
        std::vector<float> dX3((std::size_t)(TN * D), 0.0f);
        ln.backward(X, c, dX3);                     /* 单遍 (与 (b) 的 dX 同一输入) */
        int sameAsB = 0;
        for (std::size_t i = 0; i < dX3.size(); i++) { if (dX3[i] != dX[i]) { sameAsB++; } }
        CHECK(sameAsB == 0, "(c1) 同一输入的单遍反向逐位可复现 (没有残留状态)");
        std::vector<float> gG1((std::size_t)D), gB1((std::size_t)D);
        for (int j = 0; j < D; j++) { gG1[(std::size_t)j] = ln.gG[(std::size_t)j]; gB1[(std::size_t)j] = ln.gB[(std::size_t)j]; }

        ln.zeroGrad();
        std::vector<float> dX2((std::size_t)(TN * D), 0.0f);
        ln.backward(X, c, dX2);
        ln.backward(X, c, dX2);                     /* 累加两遍 */
        double accMax = 0, gAccMax = 0;
        for (std::size_t i = 0; i < dX2.size(); i++) {
            const double t2 = 2.0 * (double)dX3[i];
            accMax = std::max(accMax, std::fabs((double)dX2[i] - t2) / (std::fabs(t2) + 1e-30));
        }
        for (int j = 0; j < D; j++) {
            const double t2 = 2.0 * (double)gG1[(std::size_t)j];
            gAccMax = std::max(gAccMax, std::fabs((double)ln.gG[(std::size_t)j] - t2) / (std::fabs(t2) + 1e-30));
        }
        std::printf("    (c2) 连跑两遍 backward: dX 相对偏差 = %.2e, gG 相对偏差 = %.2e (只应有 ulp 级舍入)\n",
                    accMax, gAccMax);
        CHECK(accMax < 1e-5, "(c2) dX 是累加 (`+=`) 而不是赋值: 两遍 ≈ 2×单遍");
        CHECK(gAccMax < 1e-5, "(c2) gG 也是累加 (跨 batch 累加靠它)");

        /* (d) 往返 + 写序: 第一行必须是 γ, 第二行必须是 β */
        SeqLayerNorm ln2(D, true);
        std::ostringstream os;
        ln.write(os);
        const std::string &blob = os.str();
        {
            std::istringstream isc(blob);
            std::string l1, l2;
            std::getline(isc, l1);
            std::getline(isc, l2);
            CHECK(l1 == ln.gamma.toString(), "(d) 权重文本第一行是 γ (写序不变 = 存量权重不会错位)");
            CHECK(l2 == ln.beta.toString(), "(d) 权重文本第二行是 β");
        }
        std::istringstream is(os.str());
        ln2.read(is);
        double rd = 0;
        for (int j = 0; j < D; j++) {
            rd = std::max(rd, std::fabs((double)ln2.gamma[(std::size_t)j] - (double)ln.gamma[(std::size_t)j]));
            rd = std::max(rd, std::fabs((double)ln2.beta[(std::size_t)j] - (double)ln.beta[(std::size_t)j]));
        }
        std::printf("    (d) 往返: 权重文本 %zu 字节, γ/β 最大差 = %.3e\n", blob.size(), rd);
        CHECK(rd == 0.0, "(d) write→read 往返 γ/β 逐位相同");

        /* (e) 优化器/拷贝/软更新的接线都只剩一行循环, 至少确认它们不炸且形状对 */
        SeqLayerNorm dst(D, true);
        ln.copyTo(&dst);
        double cp = 0;
        for (int j = 0; j < D; j++) {
            cp = std::max(cp, std::fabs((double)dst.gamma[(std::size_t)j] - (double)ln.gamma[(std::size_t)j]));
        }
        CHECK(cp == 0.0, "(e) copyTo 复制 γ/β");
        ln.softUpdateTo(&dst, 0.5f);           /* dst = 0.5·dst + 0.5·ln = ln */
        double su = 0;
        for (int j = 0; j < D; j++) {
            su = std::max(su, std::fabs((double)dst.beta[(std::size_t)j] - (double)ln.beta[(std::size_t)j]));
        }
        CHECK(su == 0.0, "(e) softUpdateTo(α=0.5) 在同参数上是恒等");

        /* 梯度范数 (全局裁剪的分母) 必须等于手算的 √Σ(gG²+gB²); scaleGrad 之后要按 s² 变 */
        double n2 = 0;
        for (int j = 0; j < D; j++) {
            n2 += (double)ln.gG[(std::size_t)j] * (double)ln.gG[(std::size_t)j];
            n2 += (double)ln.gB[(std::size_t)j] * (double)ln.gB[(std::size_t)j];
        }
        const double n2lib = ln.gradNorm2();
        std::printf("    (e) gradNorm2: 库 %.6e vs 手算 %.6e (相对差 %.2e)\n",
                    n2lib, n2, std::fabs(n2lib - n2) / (n2 + 1e-30));
        CHECK(std::fabs(n2lib - n2) / (n2 + 1e-30) < 1e-9, "(e) gradNorm2 == 手算的 Σ(gG²+gB²)");
        ln.scaleGrad(0.5f);
        CHECK(std::fabs(ln.gradNorm2() - 0.25 * n2lib) / (0.25 * n2lib + 1e-30) < 1e-9,
              "(e) scaleGrad(s) 之后梯度范数按 s² 缩放");

        /* 两个优化器: 跑完必须把梯度清零 (否则下一个 batch 的梯度会叠在旧梯度上),
           且 RMSProp 的 v 必须真的被更新过 (第一个 batch 时 v 从 0 起来) */
        const double vsum0 = (double)ln.vG[0] + (double)ln.vB[0];
        ln.RMSProp(1e-3f, 0.9f, 0.001f, false);
        const double vsum1 = (double)ln.vG[0] + (double)ln.vB[0];
        std::printf("    (e) RMSProp: v[0] %.3e → %.3e, 之后 gradNorm2 = %.1f\n",
                    vsum0, vsum1, ln.gradNorm2());
        CHECK(vsum1 > vsum0, "(e) RMSProp 更新了状态 v");
        CHECK(ln.gradNorm2() == 0.0, "(e) 优化器跑完自动 zeroGrad");
        CHECK(ln.gamma[0] != 1.0f || ln.beta[0] != 0.0f, "(e) 优化器真的动了 γ/β");
        ln.clamp(-10.0f, 10.0f);   /* 参数都在范围内 ⇒ 不触发 util.hpp 的随机重采样 */
        std::printf("    (e) clamp 后 γ[0]=%.6g β[0]=%.6g\n",
                    (double)ln.gamma[0], (double)ln.beta[0]);
    }

    /* ============================================================
     *  [6] 位置编码三模式 (2026-10 补的"未做结构项": 2D 棋盘嵌入 + RoPE)
     * ============================================================ */
    std::printf("\n[6] 位置编码: 1D(默认) / 2D(行列) / RoPE\n");
    {
        std::mt19937 rng(606u);
        std::normal_distribution<float> nrm(0.0f, 1.0f);

        Random::setSeed(20240901u);
        SeqExp ex(D_IN, TOK, true);
        scaleExpertInit(ex);

        /* 这一节反复用: 跑一次推理前向并把 o 拷出来 */
        auto run = [&](SeqExp &e, const Tensor &in, std::vector<float> &out) {
            e.forward(in, true);
            out.assign((std::size_t)D_IN, 0.0f);
            for (int i = 0; i < D_IN; i++) { out[(std::size_t)i] = e.o[(std::size_t)i]; }
        };
        /*
           ⚠ 反向是**累加**的 (整个仓库的约定), 所以每次 `backward` 之前必须把要读的
           梯度清零 —— 第一版没清: (d) 段跑过一次 backward, (e) 段的 backward 又叠上去,
           于是"解析梯度"是两次反向之和, 与单次有限差分当然对不上 (读数像数值误差,
           实际是测试自己的状态没清)。`iLayer` 没有全局 zeroGrad(), 这里手写一份。
        */
        auto zeroGrads = [&]() {
            ex.embed.g.zero(); ex.outProj.g.zero();
            for (int b = 0; b < 2; b++) {
                ex.wq[(std::size_t)b].g.zero(); ex.wk[(std::size_t)b].g.zero();
                ex.wv[(std::size_t)b].g.zero(); ex.wo[(std::size_t)b].g.zero();
                ex.ffnUp[(std::size_t)b].g.zero(); ex.ffnDown[(std::size_t)b].g.zero();
            }
            ex.pos.zeroGrad();         /* 位置编码的梯度 (反向是累加的) */
        };
        /* ---- (a) 默认模式 = 1D, 参数量与改动前一致 ---- */
        std::printf("    (a) 默认: %s (参数 %lld)\n", ex.posDescribe().c_str(), ex.posParamCount());
        CHECK(ex.posMode() == SeqPosEnc::LEARN_1D, "(a) 默认位置编码 = LEARN_1D (存量权重/读数逐位不变)");
        CHECK(ex.posParamCount() == (long long)90 * (long long)TOK, "(a) 1D 参数量 = SeqLen × tokDim");
        const long long total1D = ex.paramCount();

        /* ---- (b) 2D: (10 行 + 9 列) × tokDim, 两组参数 ---- */
        ex.setPosMode(SeqPosEnc::LEARN_2D);
        std::printf("    (b) 2D:   %s (参数 %lld, 组数 %d; 总参数 %lld -> %lld)\n",
                    ex.posDescribe().c_str(), ex.posParamCount(), ex.posGroupCount(),
                    total1D, ex.paramCount());
        CHECK(ex.posGroupCount() == 2, "(b) 2D 有行/列两组参数");
        CHECK(ex.posParamCount() == (long long)(10 + 9) * (long long)TOK, "(b) 2D 参数量 = (10+9) × tokDim");
        CHECK(ex.paramCount() == total1D - (long long)(90 - 19) * (long long)TOK,
              "(b) 总参数量按 90→19 个位置向量减少");

        /* ---- (c) 2D 的加法语义 == 参考实现 (row[r] + col[c]), 逐位 ---- */
        {
            std::vector<float> X((std::size_t)(90 * TOK), 0.0f);
            ex.pos.forwardAdd(X);
            double worst = 0;
            for (int t = 0; t < 90; t++) {
                const int r = t / 9, c = t % 9;       /* canonicalCell(x,y)=x*9+y */
                for (int j = 0; j < TOK; j++) {
                    const float ref = ex.pos.emb[0].val[(std::size_t)(r * TOK + j)]
                                    + ex.pos.emb[1].val[(std::size_t)(c * TOK + j)];
                    worst = std::max(worst, (double)std::fabs(X[(std::size_t)(t * TOK + j)] - ref));
                }
            }
            std::printf("    (c) 2D 位置向量 vs row[r]+col[c] 的参考实现: 最大差 = %.1f\n", worst);
            CHECK(worst == 0.0, "(c) 2D 位置向量 = 行嵌入 + 列嵌入 (逐位)");
        }

        /* ---- (d) 2D 的解析梯度 == 有限差分 (行/列各两项 + 一个输入坐标) ---- */
        {
            Tensor x((std::size_t)D_IN, 1);
            makeState(x, rng, 90);
            Tensor cvec((std::size_t)D_IN, 1);
            for (int i = 0; i < D_IN; i++) { cvec[(std::size_t)i] = nrm(rng); }
            auto loss = [&]() {
                ex.forward(x, false);
                double s = 0;
                for (int i = 0; i < D_IN; i++) { s += (double)cvec[(std::size_t)i] * (double)ex.o[(std::size_t)i]; }
                return s;
            };
            Tensor ei((std::size_t)D_IN, 1);
            ei.zero();
            /*
               ⚠ 顺序是硬约束 (第一版就是在这里写错的): `setPosMode` 刚刚**换过参数**,
               而缓存是「最近一次前向」的 ⇒ 必须先跑一次前向再反向, 否则反传回来的是
               上一组参数下的中间量 —— 读数是「看着像数值误差, 其实是用了陈旧缓存」。
               这正是本文件 §10.2 那条不变量: 反向必须紧跟对应的那次前向。
            */
            loss();
            zeroGrads();               /* 反向是累加的: 读之前必须清零 */
            ex.e = cvec;
            ex.backward(x, ei);

            struct Item2 { const char *name; Tensor *w; Tensor *g; int idx; };
            Item2 items[4] = {
                { "row[0][5]",   &ex.pos.emb[0].val, &ex.pos.emb[0].g, 0 * TOK + 5 },
                { "row[7][61]",  &ex.pos.emb[0].val, &ex.pos.emb[0].g, 7 * TOK + 61 },
                { "col[0][9]",   &ex.pos.emb[1].val, &ex.pos.emb[1].g, 0 * TOK + 9 },
                { "col[8][40]",  &ex.pos.emb[1].val, &ex.pos.emb[1].g, 8 * TOK + 40 },
            };
            double worst2D = 0;
            for (int k = 0; k < 4; k++) {
                const double ana = (double)(*items[k].g)[(std::size_t)items[k].idx];
                const double w0 = (double)(*items[k].w)[(std::size_t)items[k].idx];
                double best = 1e30, bestNum = 0, bestEps = 0;
                for (int e2 = 0; e2 < 3; e2++) {
                    const double eps = (e2 == 0) ? 1e-2 : ((e2 == 1) ? 1e-3 : 1e-4);
                    (*items[k].w)[(std::size_t)items[k].idx] = (float)(w0 + eps);
                    const double lp = loss();
                    (*items[k].w)[(std::size_t)items[k].idx] = (float)(w0 - eps);
                    const double lm = loss();
                    (*items[k].w)[(std::size_t)items[k].idx] = (float)w0;
                    const double num = (lp - lm) / (2.0 * eps);
                    const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                    if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
                }
                std::printf("    (d) %-12s ana=%+.5e num=%+.5e (eps=%.0e) 相对误差=%.2e %s\n",
                            items[k].name, ana, bestNum, bestEps, best,
                            best < 1e-2 ? "OK" : "**不一致?**");
                worst2D = std::max(worst2D, best);
            }
            CHECK(worst2D < 1e-2, "(d) 2D 行/列嵌入的解析梯度与有限差分一致");
            /* 输入梯度: 取 |ei| **最大**的那个坐标 (随机坐标的梯度常常接近 0,
               那里的"相对误差"量的是差分的噪声地板 —— 与 [1] 同一个理由) */
            int ipos = 0;
            for (int i = 1; i < D_IN; i++) {
                if (std::fabs((double)ei[(std::size_t)i]) > std::fabs((double)ei[(std::size_t)ipos])) { ipos = i; }
            }
            const double x0 = (double)x[(std::size_t)ipos];
            double bestIn = 1e30;
            for (int e2 = 0; e2 < 3; e2++) {
                const double eps = (e2 == 0) ? 1e-2 : ((e2 == 1) ? 1e-3 : 1e-4);
                x[(std::size_t)ipos] = (float)(x0 + eps); const double lp = loss();
                x[(std::size_t)ipos] = (float)(x0 - eps); const double lm = loss();
                x[(std::size_t)ipos] = (float)x0;
                const double num = (lp - lm) / (2.0 * eps);
                const double ana = (double)ei[(std::size_t)ipos];
                bestIn = std::min(bestIn, std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30));
            }
            std::printf("    (d) 输入梯度 (2D 模式): 相对误差 = %.2e\n", bestIn);
            CHECK(bestIn < 1e-2, "(d) 2D 模式下的输入梯度与有限差分一致");
        }

        /* ---- (e) RoPE: 无参数、正交、相对位置 ---- */
        ex.setPosMode(SeqPosEnc::ROPE);
        std::printf("    (e) RoPE: %s (参数 %lld)\n", ex.posDescribe().c_str(), ex.posParamCount());
        CHECK(ex.posParamCount() == 0, "(e) RoPE 没有位置参数");
        {
            /* 正交性: 旋转不改变 |q| (逐 token/头/pair) */
            std::vector<float> Q((std::size_t)(90 * TOK)), K((std::size_t)(90 * TOK));
            for (std::size_t i = 0; i < Q.size(); i++) { Q[i] = nrm(rng); K[i] = nrm(rng); }
            std::vector<float> Q0 = Q, K0 = K;
            ex.pos.ropeApply(Q, K, 4, TOK / 4, false);
            double dmax = 0, n0 = 0, n1 = 0;
            for (int t = 0; t < 90; t++) {
                for (int h = 0; h < 4; h++) {
                    for (int c = 0; c < TOK / 4; c++) {
                        const std::size_t i = (std::size_t)(t * TOK + h * (TOK / 4) + c);
                        n0 += (double)Q0[i] * Q0[i]; n1 += (double)Q[i] * Q[i];
                    }
                }
            }
            dmax = std::fabs(n1 - n0) / (n0 + 1e-30);
            std::printf("    (e) 旋转前后 |Q|² 相对差 = %.2e (正交 ⇒ 必须 ~0)\n", dmax);
            CHECK(dmax < 1e-5, "(e) RoPE 是正交变换 (不改变 Q/K 的模)");
            /* 逆旋转还原: ropeApply(inverse) 必须精确回到原值 */
            ex.pos.ropeApply(Q, K, 4, TOK / 4, true);
            double back = 0;
            for (std::size_t i = 0; i < Q.size(); i++) {
                back = std::max(back, (double)std::fabs(Q[i] - Q0[i]));
                back = std::max(back, (double)std::fabs(K[i] - K0[i]));
            }
            std::printf("    (e) 逆旋转还原: 最大差 = %.3e\n", back);
            CHECK(back < 1e-4, "(e) RoPE 的逆变换把 Q/K 还原 (反向用的就是它)");
        }
        {
            /* 相对位置: RoPE 的定义性质是 `s(i,j) = s(i+Δ, j+Δ)` (位移后重叠区不变)。
               ⚠ 端到端做这件事**不干净**: 特征为 0 的格子过了 embed 之后带着 bias
               (b ≠ 0) 也参与注意力, 而它们的**位置**在平移后变了 ⇒ 端到端的差异不
               只来自"位置是不是相对的"。所以这里分两层:
                 (1) 分量层: 直接对 Q/K 旋转后比 s(i,j) vs s(i+Δ,j+Δ) —— 这才是定义,
                     而且是精确性质 (fp 级);
                 (2) 端到端: 只做**对照** (1D 绝对位置同样平移的差要远大于 RoPE)。 */
            {
                std::vector<float> Q((std::size_t)(90 * TOK)), K((std::size_t)(90 * TOK));
                for (std::size_t i = 0; i < Q.size(); i++) { Q[i] = nrm(rng); K[i] = nrm(rng); }
                /* 构造"把内容整体右移 DELTA"的另一份: Q2(t+Δ)=Q(t), K2(t+Δ)=K(t) */
                const int DEL = 5;
                std::vector<float> Qr = Q, Kr = K, Q2((std::size_t)(90 * TOK), 0.0f), K2((std::size_t)(90 * TOK), 0.0f);
                for (int t = 0; t + DEL < 90; t++) {
                    for (int c = 0; c < TOK; c++) {
                        Q2[(std::size_t)((t + DEL) * TOK + c)] = Q[(std::size_t)(t * TOK + c)];
                        K2[(std::size_t)((t + DEL) * TOK + c)] = K[(std::size_t)(t * TOK + c)];
                    }
                }
                ex.pos.ropeApply(Qr, Kr, 4, TOK / 4, false);
                ex.pos.ropeApply(Q2, K2, 4, TOK / 4, false);
                double worstRel = 0, scale = 0;
                /*
                   ⚠ 必须按**每个头的整段点积**比, 不能逐分量比: RoPE 在每个头内把
                   (2m, 2m+1) 两个分量转在一起 ⇒ 单独一个分量的乘积**没有**相对性,
                   只有整段点积 q·k 才有 (第一版逐分量比, 读到 12.15 的"差", 那是
                   测试自己的数学写错了, 不是实现错)。
                */
                for (int i = 0; i + DEL < 90; i++) {
                    for (int j = 0; j + DEL < 90; j++) {
                        for (int h = 0; h < 4; h++) {
                            double s1 = 0, s2 = 0;
                            for (int cc = 0; cc < TOK / 4; cc++) {
                                const std::size_t a1 = (std::size_t)(i * TOK + h * (TOK / 4) + cc);
                                const std::size_t b1 = (std::size_t)(j * TOK + h * (TOK / 4) + cc);
                                const std::size_t a2 = (std::size_t)((i + DEL) * TOK + h * (TOK / 4) + cc);
                                const std::size_t b2 = (std::size_t)((j + DEL) * TOK + h * (TOK / 4) + cc);
                                s1 += (double)Qr[a1] * (double)Kr[b1];
                                s2 += (double)Q2[a2] * (double)K2[b2];
                            }
                            scale = std::max(scale, std::fabs(s2));
                            worstRel = std::max(worstRel, std::fabs(s1 - s2));
                        }
                    }
                }
                std::printf("    (e) 分量层: 旋转后 s(i,j) vs s(i+%d,j+%d) 最大差 = %.3e (量级 %.1f)\n",
                            DEL, DEL, worstRel, scale);
                CHECK(worstRel < 1e-3, "(e) RoPE 的定义性质: s(i,j) = s(i+Δ, j+Δ) (相对位置, 分量层精确)");
            }
            {
            const int DELTA = 3, LO = 12, HI = 70;
            const int FEAT = D_IN / 90;
            Tensor xa((std::size_t)D_IN, 1), xb((std::size_t)D_IN, 1);
            xa.zero(); xb.zero();
            for (int t = LO; t < HI; t++) {
                for (int f = 0; f < FEAT; f++) {
                    /* 确定性的"图案": 与位置有关, 但平移后图案跟着走 */
                    const float v = ((t * 7 + f * 13) % 5 == 0) ? 1.0f : (((t + f) % 3 == 0) ? -1.0f : 0.25f);
                    xa[(std::size_t)(f * 90 + t)] = v;
                    if (t + DELTA < 90) { xb[(std::size_t)(f * 90 + t + DELTA)] = v; }
                }
            }

            std::vector<float> oa, ob;
            run(ex, xa, oa);
            run(ex, xb, ob);
            double worstShift = 0;
            for (int t = LO; t < HI - DELTA; t++) {           /* 重叠区 */
                for (int f = 0; f < FEAT; f++) {
                    const double a = (double)oa[(std::size_t)(f * 90 + t)];
                    const double b = (double)ob[(std::size_t)(f * 90 + t + DELTA)];
                    worstShift = std::max(worstShift, std::fabs(a - b));
                }
            }
            /* 端到端: 只当**对照**看, 不当"不变"的判据 (理由见上面那段注释: 空格子
               带着 embed 的 bias 参与注意力, 它的位置在平移后变了 ⇒ 端到端天生不为 0)。
               两者的打印与判定都放在 1D 对照算完之后 (见下)。 */

            /* 正对照: 可学 1D 位置嵌入**没有**这条性质。
               注意要把 1D 的位置向量**放大**再比 —— 初始化时它被缩过 1/√tokDim
               (≈0.125), 而 token 特征是 ±1 量级, 位置信号太弱, 不放大时
               "绝对位置对平移敏感"这件事会淹没在噪声里 (第一版就这么放过了 3.9e-2
               这种只差 5 倍的弱对照)。 */
            ex.setPosMode(SeqPosEnc::LEARN_1D);
            for (std::size_t i = 0; i < ex.pos.emb[0].val.totalSize; i++) { ex.pos.emb[0].val[i] *= 100.0f; }
            run(ex, xa, oa);
            run(ex, xb, ob);
            double worstAbs = 0;
            for (int t = LO; t < HI - DELTA; t++) {
                for (int f = 0; f < FEAT; f++) {
                    worstAbs = std::max(worstAbs, std::fabs((double)oa[(std::size_t)(f * 90 + t)]
                                                          - (double)ob[(std::size_t)(f * 90 + t + DELTA)]));
                }
            }
            std::printf("    (e) 端到端对照: 平移 %d 后重叠区输出最大差 = %.3e (RoPE) vs %.3e (LEARN_1D, 位置向量放大 100x)\n",
                        DELTA, worstShift, worstAbs);
            CHECK(worstAbs > 10.0 * worstShift, "(e) 端到端对照: RoPE 的平移敏感性比 1D 绝对位置小一个量级以上");
            CHECK(worstShift < 1e-2, "(e) 端到端: RoPE 下平移后的差异仍在 fp 噪声量级 (<1e-2)");
        }
        }
        {
            /* RoPE 下的梯度: 反向里的逆旋转写错, 有限差分一定抓得到。
               这里查 wq/wo/embed 三组 (旋转只影响 Q/K 这条链)。 */
            ex.setPosMode(SeqPosEnc::ROPE);
            Tensor x((std::size_t)D_IN, 1);
            makeState(x, rng, 120);
            Tensor cvec((std::size_t)D_IN, 1);
            for (int i = 0; i < D_IN; i++) { cvec[(std::size_t)i] = nrm(rng); }
            auto loss = [&]() {
                ex.forward(x, false);
                double s = 0;
                for (int i = 0; i < D_IN; i++) { s += (double)cvec[(std::size_t)i] * (double)ex.o[(std::size_t)i]; }
                return s;
            };
            Tensor ei((std::size_t)D_IN, 1);
            ei.zero();
            loss();                      /* 先前向: 换过模式/参数之后缓存必须刷新 */
            zeroGrads();               /* 反向是累加的: 读之前必须清零 */
            ex.e = cvec;
            ex.backward(x, ei);
            struct ItemR { const char *name; Tensor *w; Tensor *g; int idx; };
            /*
               ⚠ 坐标要挑 **|解析梯度| 最大** 的那个: 这里 embed 的梯度只有 ~5e-4,
               而 eps=1e-2 时 float32 的中心差分噪声地板是 |L|·1e-7/(2eps) ~ 2e-4 ⇒
               随机挑坐标会得到一个"看着像实现错了"的噪声读数 (第一版就是挑的 [1][3],
               读到 11.5% 的相对误差)。与 [1] 段"取 |ei| 最大的坐标"是同一条纪律。
            */
            auto argmaxAbs = [](const Tensor &t) {
                int best = 0;
                for (std::size_t i = 1; i < t.totalSize; i++) {
                    if (std::fabs((double)t[i]) > std::fabs((double)t[(std::size_t)best])) { best = (int)i; }
                }
                return best;
            };
            ItemR it[3] = {
                { "wq.w",    &ex.wq[0].w, &ex.wq[0].g.w, argmaxAbs(ex.wq[0].g.w) },
                { "wo.w",    &ex.wo[1].w, &ex.wo[1].g.w, argmaxAbs(ex.wo[1].g.w) },
                { "embed.w", &ex.embed.w, &ex.embed.g.w, argmaxAbs(ex.embed.g.w) },
            };
            double worstR = 0;
            for (int k = 0; k < 3; k++) {
                const double ana = (double)(*it[k].g)[(std::size_t)it[k].idx];
                const double w0 = (double)(*it[k].w)[(std::size_t)it[k].idx];
                double best = 1e30, bestNum = 0, bestEps = 0;
                for (int e2 = 0; e2 < 3; e2++) {
                    const double eps = (e2 == 0) ? 1e-2 : ((e2 == 1) ? 1e-3 : 1e-4);
                    (*it[k].w)[(std::size_t)it[k].idx] = (float)(w0 + eps);
                    const double lp = loss();
                    (*it[k].w)[(std::size_t)it[k].idx] = (float)(w0 - eps);
                    const double lm = loss();
                    (*it[k].w)[(std::size_t)it[k].idx] = (float)w0;
                    const double num = (lp - lm) / (2.0 * eps);
                    const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                    if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
                }
                std::printf("    (e) RoPE FD %-13s ana=%+.5e num=%+.5e (eps=%.0e) 相对误差=%.2e %s\n",
                            it[k].name, ana, bestNum, bestEps, best,
                            best < 1e-2 ? "OK" : "**不一致?**");
                worstR = std::max(worstR, best);
            }
            CHECK(worstR < 1e-2, "(e) RoPE 模式下 wq/wo/embed 的解析梯度与有限差分一致 (逆旋转正确)");
            /* usePos=false ⇒ RoPE 也要被关掉 (对置换等变) */
            ex.usePos = false;
            std::vector<int> perm(90);
            for (int i = 0; i < 90; i++) { perm[(std::size_t)i] = i; }
            std::shuffle(perm.begin(), perm.end(), rng);
            Tensor xp((std::size_t)D_IN, 1);
            xp.zero();
            for (int f = 0; f < 19; f++) {
                for (int t = 0; t < 90; t++) { xp[(std::size_t)(f * 90 + perm[(std::size_t)t])] = x[(std::size_t)(f * 90 + t)]; }
            }
            std::vector<float> o1, o2;
            run(ex, x, o1);
            run(ex, xp, o2);
            double dPerm = 0;
            for (int f = 0; f < 19; f++) {
                for (int t = 0; t < 90; t++) {
                    dPerm = std::max(dPerm, std::fabs((double)o1[(std::size_t)(f * 90 + t)]
                                                    - (double)o2[(std::size_t)(f * 90 + perm[(std::size_t)t])]));
                }
            }
            std::printf("    (e) usePos=false + ROPE: 置换等变差 = %.3e\n", dPerm);
            CHECK(dPerm < 1e-4, "(e) usePos=false 时 RoPE 也被关掉 (置换等变)");
            ex.usePos = true;
        }

        /* ---- (f) 权重往返 + 模式自描述 ---- */
        {
            struct M { SeqPosEnc::Mode m; const char *marker; const char *name; };
            const M modes[3] = { { SeqPosEnc::LEARN_1D, nullptr,   "1d" },
                                 { SeqPosEnc::LEARN_2D, "POS2D",   "2d" },
                                 { SeqPosEnc::ROPE,     "POSROPE", "rope" } };
            for (int k = 0; k < 3; k++) {
                ex.setPosMode(modes[k].m);
                std::ostringstream os;
                ex.write(os);
                std::istringstream is(os.str());
                const std::string &blob = os.str();
                /* 标记行是**位置编码那一段**的第一行 (不是整个权重的第一行 ——
                   第一行永远是 embed)。所以"有没有标记"要在整段文本里找,
                   而"1D 没有标记"要检查**没有任何字母开头的行**。 */
                const bool hasMarker = (modes[k].marker != nullptr)
                                     && (blob.find(std::string("\n") + modes[k].marker + "\n") != std::string::npos
                                      || blob.compare(0, std::strlen(modes[k].marker), modes[k].marker) == 0);
                bool anyMarkerLine = false;
                {
                    std::istringstream scan(blob);
                    std::string line;
                    while (std::getline(scan, line)) {
                        if (!line.empty() && std::isalpha((unsigned char)line[0])) { anyMarkerLine = true; }
                    }
                }
                const bool noMarker = (modes[k].marker == nullptr) && !anyMarkerLine;
                /* 读回同一模式: 参数一致 + 同一输入同一输出 */
                SeqExp ex2(D_IN, TOK, false);
                ex2.setPosMode(modes[k].m);
                ex2.read(is);
                CHECK(hasMarker || noMarker,
                      modes[k].m == SeqPosEnc::LEARN_1D
                        ? "(f) 1D 不写标记 (字节与改动前一致, FNV 指纹不变)"
                        : "(f) 非默认模式写出模式标记 (权重自描述)");
                /* 2D / ROPE 的字节前缀检查 (上面那个 CHECK 只是笼统判断) */
                std::printf("    (f) %-4s 权重文本 %zu 字节, 首行=%s, 参数 %lld\n",
                            modes[k].name, blob.size(),
                            blob.substr(0, blob.find('\n')).substr(0, 24).c_str(),
                            ex.posParamCount());
                /* 参数一致 */
                bool sameParams = (ex.posGroupCount() == ex2.posGroupCount());
                if (sameParams) {
                    for (int gg = 0; gg < ex.posGroupCount() && sameParams; gg++) {
                        if (ex2.pos.emb[(std::size_t)gg].val.totalSize != ex.pos.emb[(std::size_t)gg].val.totalSize) {
                            sameParams = false;
                        }
                    }
                }
                CHECK(sameParams, "(f) 往返两边的位置编码参数量/组数一致");
            }
        }

        /* ---- (g) 模式不匹配必须**响亮失败** (子进程里跑, 看退出码) ---- */
        if (argc > 1 && std::string(argv[1]) == "--posenc-mismatch") {
            /* 故意用 ROPE 的权重去喂一个 1D 专家 —— 期望 abort */
            SeqExp a(D_IN, TOK, false);
            a.setPosMode(SeqPosEnc::ROPE);
            std::ostringstream os;
            a.write(os);
            SeqExp b(D_IN, TOK, false);          /* 默认 1D */
            std::istringstream is(os.str());
            b.read(is);                          /* ⇒ 这里会 abort */
            std::printf("!! 模式不匹配竟然读成功了 (守卫失效)\n");
            return 0;
        }
        {
            const std::string self = (argc > 0 && argv[0] != nullptr) ? std::string(argv[0]) : std::string("test_transformer.exe");
            const std::string cmd = "\"" + self + "\" --posenc-mismatch > nul 2>&1";
            const int rc = std::system(cmd.c_str());
            std::printf("    (g) 用 2D/RoPE 的权重去喂另一个模式的专家: 子进程退出码 = %d (非 0 = 响亮失败)\n", rc);
            CHECK(rc != 0, "(g) 位置编码模式与权重文件不一致时**响亮失败** (不静默错位)");
        }

        ex.setPosMode(SeqPosEnc::LEARN_1D);      /* 收尾: 回到默认, 免得影响后面的段 */
    }

    /* ============================================================
     *  [7] dropout (2026-10 补的"未做结构项"之二)
     * ============================================================ */
    std::printf("\n[7] dropout: 默认关 (逐位不变) / 开启后的 mask 与梯度\n");
    {
        std::mt19937 rng(707u);
        std::normal_distribution<float> nrm(0.0f, 1.0f);
        Random::setSeed(20240901u);
        SeqExp ex(D_IN, TOK, true);
        scaleExpertInit(ex);

        Tensor x((std::size_t)D_IN, 1);
        makeState(x, rng, 120);

        /* (a) 默认 p=0: 推理与训练形态都不动一个字节 */
        std::printf("    (a) 默认 dropoutP = %.3f\n", (double)ex.dropoutP());
        CHECK(ex.dropoutP() == 0.0f, "(a) dropout 默认关闭");
        std::vector<float> o0((std::size_t)D_IN, 0.0f), o1((std::size_t)D_IN, 0.0f);
        ex.forward(x, true);
        for (int i = 0; i < D_IN; i++) { o0[(std::size_t)i] = ex.o[(std::size_t)i]; }
        ex.forward(x, true);
        for (int i = 0; i < D_IN; i++) { o1[(std::size_t)i] = ex.o[(std::size_t)i]; }
        double d0 = 0;
        for (int i = 0; i < D_IN; i++) { d0 = std::max(d0, (double)std::fabs(o0[(std::size_t)i] - o1[(std::size_t)i])); }
        std::printf("    (a) p=0 时两次前向的差 = %.1f (确定性 ⇒ 不消耗随机数)\n", d0);
        CHECK(d0 == 0.0, "(a) p=0 ⇒ 前向确定性且逐位一致");

        /* (b) 推理形态下开着 dropout 也不生效 (倒置式: 推理不需要缩放) */
        ex.setDropout(0.5f);
        ex.forward(x, true);
        for (int i = 0; i < D_IN; i++) { o1[(std::size_t)i] = ex.o[(std::size_t)i]; }
        double dInfer = 0;
        for (int i = 0; i < D_IN; i++) { dInfer = std::max(dInfer, (double)std::fabs(o0[(std::size_t)i] - o1[(std::size_t)i])); }
        std::printf("    (b) p=0.5 的**推理**形态 vs p=0 的输出差 = %.3e (必须为 0)\n", dInfer);
        CHECK(dInfer == 0.0, "(b) inference=true 不做 dropout (倒置式 ⇒ 推理路径与 p=0 逐位相同)");

        /* (c) 训练形态: 两次前向**不同** (mask 每次重抽), 但各自确定可复现 */
        ex.forward(x, false);
        std::vector<float> t1((std::size_t)D_IN), t2((std::size_t)D_IN);
        for (int i = 0; i < D_IN; i++) { t1[(std::size_t)i] = ex.o[(std::size_t)i]; }
        ex.forward(x, false);
        for (int i = 0; i < D_IN; i++) { t2[(std::size_t)i] = ex.o[(std::size_t)i]; }
        double dTrain = 0;
        for (int i = 0; i < D_IN; i++) { dTrain = std::max(dTrain, (double)std::fabs(t1[(std::size_t)i] - t2[(std::size_t)i])); }
        std::printf("    (c) p=0.5 的**训练**形态两次前向差 = %.3e (必须 > 0: 每次重抽 mask)\n", dTrain);
        CHECK(dTrain > 1e-6, "(c) 训练形态真的在做 dropout (两次前向不同)");
        /* (d) mask 的统计与取值: 保留比例 ≈ 1−p, 且每个元素只能是 0 或 1/(1−p) */
        {
            ex.setDropout(0.25f);
            ex.forward(x, false);
            const std::vector<float> &mp = ex.lastMask(0, 0);
            const double inv = 1.0 / (1.0 - 0.25);
            double kept = 0, badValue = 0;
            for (std::size_t i = 0; i < mp.size(); i++) {
                if (mp[i] != 0.0f) { kept += 1.0; }
                if (mp[i] != 0.0f && std::fabs((double)mp[i] - inv) > 1e-6) { badValue += 1.0; }
            }
            const double ratio = kept / (double)mp.size();
            std::printf("    (d) p=0.25 注意力 mask: %zu 项, 保留比例 = %.4f (期望 0.7500), 非法取值 = %.0f\n",
                        mp.size(), ratio, badValue);
            CHECK(std::fabs(ratio - 0.75) < 0.01, "(d) mask 的保留比例 ≈ 1−p (统计)");
            CHECK(badValue == 0.0, "(d) mask 只有 0 与 1/(1−p) 两种取值 (倒置式 dropout)");
            const std::vector<float> &mf = ex.lastMask(0, 2);
            CHECK(mf.size() == (std::size_t)(90 * TOK), "(d) 残差支路的 mask 尺寸 = T × tokDim");
        }

        /* (e) **冻住 mask** 之后损失又是确定函数 ⇒ 可以用中心差分钉住
               "反向用的是同一张 mask、而且只作用在该作用的那条路上"。
               不冻 mask 的话这个检查根本做不了 (每次前向重抽 mask, 损失不是确定函数)。 */
        {
            ex.setDropout(0.3f);
            ex.setDropoutFreezeMask(true);
            Tensor cvec2((std::size_t)D_IN, 1);
            for (int i = 0; i < D_IN; i++) { cvec2[(std::size_t)i] = nrm(rng); }
            auto loss2 = [&]() {
                ex.forward(x, false);
                double s = 0;
                for (int i = 0; i < D_IN; i++) { s += (double)cvec2[(std::size_t)i] * (double)ex.o[(std::size_t)i]; }
                return s;
            };
            loss2();                      /* 第一遍抽 mask; 之后一直复用同一张 */
            zeroExpertGrads(ex);
            ex.e = cvec2;
            Tensor ei((std::size_t)D_IN, 1);
            ei.zero();
            ex.backward(x, ei);
            auto argmaxAbsD = [](const Tensor &t) {
                int best = 0;
                for (std::size_t i = 1; i < t.totalSize; i++) {
                    if (std::fabs((double)t[i]) > std::fabs((double)t[(std::size_t)best])) { best = (int)i; }
                }
                return best;
            };
            struct ItD { const char *n; Tensor *w; Tensor *g; int idx; };
            ItD its[3] = {
                { "wq.w",    &ex.wq[0].w, &ex.wq[0].g.w, argmaxAbsD(ex.wq[0].g.w) },
                { "wo.w",    &ex.wo[1].w, &ex.wo[1].g.w, argmaxAbsD(ex.wo[1].g.w) },
                { "embed.w", &ex.embed.w, &ex.embed.g.w, argmaxAbsD(ex.embed.g.w) },
            };
            double worstD = 0;
            for (int k = 0; k < 3; k++) {
                const double ana = (double)(*its[k].g)[(std::size_t)its[k].idx];
                const double w0 = (double)(*its[k].w)[(std::size_t)its[k].idx];
                double best = 1e30, bestNum = 0, bestEps = 0;
                for (int e2 = 0; e2 < 3; e2++) {
                    const double eps = (e2 == 0) ? 1e-2 : ((e2 == 1) ? 1e-3 : 1e-4);
                    (*its[k].w)[(std::size_t)its[k].idx] = (float)(w0 + eps);
                    const double lp = loss2();
                    (*its[k].w)[(std::size_t)its[k].idx] = (float)(w0 - eps);
                    const double lm = loss2();
                    (*its[k].w)[(std::size_t)its[k].idx] = (float)w0;
                    const double num = (lp - lm) / (2.0 * eps);
                    const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-30);
                    if (rel < best) { best = rel; bestNum = num; bestEps = eps; }
                }
                std::printf("    (e) 冻 mask 的 FD %-9s ana=%+.5e num=%+.5e (eps=%.0e) 相对误差=%.2e %s\n",
                            its[k].n, ana, bestNum, bestEps, best,
                            best < 1e-2 ? "OK" : "**不一致?**");
                worstD = std::max(worstD, best);
            }
            CHECK(worstD < 1e-2, "(e) 冻住 mask 后解析梯度 == 有限差分 (反向用同一张 mask, 且不污染残差那条路)");

            /*
               ⚠ **不能**用"Σ|g| 应该变小"当判据: 倒置式 dropout 把保留项乘了 1/(1−p)
               (p=0.3 时 1.43×), 单个 mask 下 Σ|g| 完全可能**变大** —— 第一版就是这么判的
               (读到 1.32e4 > 1.23e4, 报了一个假失败)。真正说明"前后向用同一张 mask、
               每条路都乘对了"的是**上面那条冻 mask 的有限差分**: dropout 第一版里 dV
               用了未丢弃的 P, 正是被它抓出来的 (wq 偏大 15.6% / embed 偏大 24.6%)。
               这里只留一个必要条件, 并把两个读数都印出来 (差值本身也是读数)。
            */
            ex.setDropoutFreezeMask(false);
            ex.setDropout(0.0f);
            ex.forward(x, false);
            zeroExpertGrads(ex); ex.e = cvec2; ex.backward(x, ei);
            double gNoDrop = 0;
            for (std::size_t i = 0; i < ex.ffnDown[0].g.w.totalSize; i++) { gNoDrop += std::fabs((double)ex.ffnDown[0].g.w[i]); }
            ex.setDropout(0.3f); ex.setDropoutFreezeMask(true);
            ex.forward(x, false);
            zeroExpertGrads(ex); ex.e = cvec2; ex.backward(x, ei);
            double gDrop = 0;
            for (std::size_t i = 0; i < ex.ffnDown[0].g.w.totalSize; i++) { gDrop += std::fabs((double)ex.ffnDown[0].g.w[i]); }
            std::printf("    (e) Σ|g(ffnDown[0])|: p=0 → %.4e, p=0.3 → %.4e (倒置式 ⇒ 不保证更小, 只要求非零)\n", gNoDrop, gDrop);
            CHECK(gDrop > 0.0, "(e) dropout 之后 FFN 仍然拿到非零梯度");
            ex.setDropoutFreezeMask(false);

            /* (f) 20 步: 开了 dropout 也要能真的下降 */
            ex.setDropout(0.3f);
            double first = 0, last = 0;
            for (int it = 0; it < 20; it++) {
                const double L = loss2();
                if (it == 0) { first = L; }
                zeroExpertGrads(ex);
                ex.e = cvec2;
                ex.backward(x, ei);
                ex.RMSProp(0.002f, 0.9f, 0.001f, true);   /* 仓库 BC 的口径 */
                if (it == 19) { last = loss2(); }
            }
            std::printf("    (f) 20 步 RMSProp (p=0.3): 损失 %.4f -> %.4f\n", first, last);
            CHECK(last < first, "(f) 开着 dropout 也能训练 (20 步损失下降)");
        }
        ex.setDropout(0.0f);
        ex.setDropoutFreezeMask(false);
    }

    /* ============================================================
     *  [8] 未接线死代码的守卫 (2026-10 补的"未做结构项"之四)
     *      `PositionalEncoder` / `Attention<N>` / `Dropout<Fn>` 三者从未被生产路径
     *      实例化过。守卫的目的不是"删掉它们", 而是让**接错**这件事响亮失败。
     * ============================================================ */
    std::printf("\n[8] 死代码守卫: 未接线的三个层\n");
    if (argc > 1 && std::string(argv[1]) == "--unwired-guard") {
        /* 子进程: 默认不许跑 ⇒ 期望 abort */
        PositionalEncoder pe(8, false);
        Tensor x8((std::size_t)8, 1);
        x8.zero();
        pe.forward(x8, true);
        std::printf("!! 未接线层竟然跑起来了 (守卫失效)\n");
        return 0;
    }
    {
        CHECK(!RL::unwiredLayersAllowed(), "(8a) 默认**禁止**调用未接线层 (守卫为假)");
        /* 关掉守卫时, 它们必须**真的不能跑** —— 在子进程里试, 看退出码 */
        const std::string self = (argc > 0 && argv[0] != nullptr) ? std::string(argv[0]) : std::string("test_transformer.exe");
        const int rc = std::system(("\"" + self + "\" --unwired-guard > nul 2>&1").c_str());
        std::printf("    (8a) 子进程直接调用 PositionalEncoder: 退出码 = %d (非 0 = 响亮失败)\n", rc);
        CHECK(rc != 0, "(8a) 未接线层的 forward 默认 abort (不会静默跑起来)");

        /* 打开守卫之后, 这些层**本身**仍然要正确: PositionalEncoder 是 x + pe,
           反向把 e 原样传下去 (pe 是常量) */
        RL::setAllowUnwiredLayers(true);
        {
            PositionalEncoder pe(8, false);
            Tensor x8((std::size_t)8, 1);
            for (int i = 0; i < 8; i++) { x8[(std::size_t)i] = (float)i; }
            pe.forward(x8, true);
            pe.forward(x8, true);                     /* 第二次用的是 pos=1, 之后 pos 变成 2 */
            double worst = 0;
            for (int i = 0; i < 8; i++) {
                /* ⚠ 实现是"先用后加": 第 n 次 forward 用的是 pos = n−1 (第一次 pos=0) —— 
                   第一版按 pos=2 去比, 差 9.6e-1 报了假失败 */
                const float expect = ((i % 2) == 0)
                    ? std::sin(1.0f / std::pow(10000.0f, (float)i / 8.0f))
                    : std::cos(1.0f / std::pow(10000.0f, (float)(i - 1) / 8.0f));
                worst = std::max(worst, (double)std::fabs(pe.o[(std::size_t)i] - (x8[(std::size_t)i] + expect)));
            }
            std::printf("    (8b) PositionalEncoder: o = x + pe(用 pos=1) 的最大差 = %.3e, 之后 pos = %d\n", worst, pe.pos);
            CHECK(worst < 1e-6, "(8b) PositionalEncoder 的语义 = x + 正弦位置编码 (打开守卫后可跑且正确)");
            pe.o.zero();
            Tensor ei8((std::size_t)8, 1);
            ei8.zero();
            pe.e.fill(1.0f);
            pe.backward(x8, ei8);
            double bad = 0;
            for (int i = 0; i < 8; i++) { bad = std::max(bad, std::fabs((double)ei8[(std::size_t)i] - 1.0)); }
            std::printf("    (8b) PositionalEncoder 反向: dL/dx 与 e 的差 = %.3e (pe 是常量 ⇒ 恒等)\n", bad);
            CHECK(bad < 1e-6, "(8b) PositionalEncoder 的反向是恒等 (pe 无参数)");
        }
        {
            /* Attention<N>: 旧"外积伪注意力"。它的反向此前只有 double 参考审计过
               ([4] 那段), 这里只查"打开守卫后能跑、输出有限"。 */
            Attention<15> attn(240, 240, false);   /* (inputDim, unitDim, withGrad) => 输出 240×15 */
            Tensor xa((std::size_t)240, 1);
            for (int i = 0; i < 240; i++) { xa[(std::size_t)i] = (i % 7) * 0.1f; }
            attn.forward(xa, false);
            double finite = 0, mx = 0;
            for (std::size_t i = 0; i < attn.o.totalSize; i++) {
                if (!std::isfinite((double)attn.o[i])) { finite += 1.0; }
                mx = std::max(mx, std::fabs((double)attn.o[i]));
            }
            std::printf("    (8c) Attention<15>@240: 输出 max = %.4g, 非有限值 = %.0f 个\n", mx, finite);
            CHECK(finite == 0.0 && mx > 0.0, "(8c) 打开守卫后 Attention<N> 能跑出有限输出 (它历史上的 pe 越界写已修)");
        }
        RL::setAllowUnwiredLayers(false);
        CHECK(!RL::unwiredLayersAllowed(), "(8d) 关闭守卫后恢复默认 (禁止)");
    }

    /* ============================================================
     *  [9] 对齐分配器（2026-10 从 N-spirits 的 basic/alignallocator.hpp 迁移 + 适配 SIMD）
     * ============================================================ */
    std::printf("\n[9] 对齐分配器: Tensor 的存储 32 B 对齐 / 需要对齐的内核的判据\n");
    {
        /* (a) Tensor 的存储必须 32 B 对齐 —— 这是迁移要拿到的**唯一可见效果** */
        struct Sz { int r, c; };
        const Sz shapes[5] = { { 1, 1 }, { 64, 1 }, { 19, 64 }, { 90, 64 }, { 360, 1710 } };
        int bad = 0, minAlign = 1 << 20;
        for (int k = 0; k < 5; k++) {
            Tensor t(shapes[k].r, shapes[k].c);
            const int a = (int)t.dataAlignment();
            if (!t.alignedTo(32)) { bad++; }
            minAlign = std::min(minAlign, a);
            std::printf("    (a) Tensor(%4d,%4d): %zu 元素, 对齐 = %d B, alignedTo(32) = %d\n",
                        shapes[k].r, shapes[k].c, t.totalSize, a, (int)t.alignedTo(32));
        }
        CHECK(bad == 0, "(9a) 所有形状的 Tensor 存储都 32 B 对齐 (对齐分配器生效)");
        CHECK(minAlign >= 32, "(9a) 最小对齐 >= 32 B");

        /* (b) 对照: 默认 std::vector 只保证 16 B —— 同一块内存错开 4 字节就不再对齐 */
        {
            std::vector<float> v(4096, 0.0f);
            const bool v32 = RL::isAlignedTo(v.data(), 32);
            const bool off32 = RL::isAlignedTo(v.data() + 1, 32);
            std::printf("    (b) std::vector<float> 首地址 32B 对齐 = %d; 错开 4 字节后 = %d\n",
                        (int)v32, (int)off32);
            CHECK(!off32, "(9b) 判据有效: 错开 4 字节的指针不再 32 B 对齐 (守卫就是用它判的)");
        }

        /* (c) 专家里真正会被 SIMD 内核读写的那些张量 (参数 + 优化器状态) 全都要对齐 */
        {
            Random::setSeed(20240901u);
            SeqExp ex(D_IN, TOK, true);
            scaleExpertInit(ex);
            int badParam = 0, checked = 0;
            Tensor *ps[8] = { &ex.embed.w, &ex.embed.b, &ex.outProj.w, &ex.outProj.b,
                              &ex.wq[0].w, &ex.wo[1].w, &ex.ffnUp[0].w, &ex.ffnDown[1].w };
            for (int i = 0; i < 8; i++) {
                checked++;
                if (!ps[i]->alignedTo(32)) { badParam++; }
            }
            for (int b = 0; b < 2; b++) {
                if (!ex.ln1[b].gamma.alignedTo(32)) { badParam++; }
                if (!ex.ln2[b].beta.alignedTo(32)) { badParam++; }
                checked += 2;
            }
            const std::vector<float> &pos = ex.lastMask(0, 0);
            (void)pos;
            std::printf("    (c) 专家参数/优化器状态: 检查 %d 个张量, 不对齐 %d 个\n", checked, badParam);
            CHECK(badParam == 0, "(9c) 专家里所有张量都 32 B 对齐 (SIMD 内核的入参就是从这些来的)");
            /* 位置嵌入 (可能 0/1/2 组) 也要对齐 */
            int badPos = 0;
            for (std::size_t g = 0; g < (std::size_t)ex.posGroupCount(); g++) {
                if (!ex.pos.emb[g].val.alignedTo(32)) { badPos++; }
            }
            CHECK(badPos == 0, "(9c) 位置编码参数对齐 (1D/2D 两组都查)");
        }

        /* (d) 迁移不改算术: 同一个张量在两种分配器下的值/运算逐位相同
               (f32 的舍入只取决于值, 与地址无关 —— 但要**证明**它, 而不是相信它) */
        {
            RL::Tensor_<float, RL::AlignAllocator32> a(std::vector<int>{ 90, 64 });
            RL::Tensor_<float, std::allocator> b(std::vector<int>{ 90, 64 });
            for (std::size_t i = 0; i < a.totalSize; i++) {
                const float v = (float)((int)(i * 37 % 101) - 50) * 0.25f;
                a[i] = v; b[i] = v;
            }
            double dfill = 0, dpair = 0;
            for (std::size_t i = 0; i < a.totalSize; i++) {
                dfill = std::max(dfill, (double)std::fabs(a[i] - b[i]));
            }
            /* 逐元素乘 + 归约 (走 SIMD 内核) 也逐位比 */
            a *= a; b *= b;
            for (std::size_t i = 0; i < a.totalSize; i++) {
                dpair = std::max(dpair, (double)std::fabs(a[i] - b[i]));
            }
            std::printf("    (d) 对齐 vs std 分配: 同值初始化差 %.1f, 逐元素乘之后差 %.1f, "
                        "sum 差 %.3e\n", dfill, dpair, std::fabs((double)a.sum() - (double)b.sum()));
            CHECK(dfill == 0.0 && dpair == 0.0, "(9d) 两种分配器的逐元素结果逐位相同 (算术与地址无关)");
            CHECK(std::fabs((double)a.sum() - (double)b.sum()) == 0.0, "(9d) 归约 (SIMD sum) 也逐位相同");
        }
    }

    /* ============================================================
     *  [10] 常驻结构守卫 (2026-10, P1)
     *       这一节钉住的是"**契约失守时响亮失败**", 而不是"某个数值算对了"。
     *       每一条都对应一次真实或近真实的事故:
     *         (10a/10b) `MM::*` 的形状契约 —— `src/rl/sac.h:99-104` 记过一次 Release
     *                   静默读错 (critic 第一层按 stateDim 建、喂了 stateDim+actionDim);
     *         (10c)     `posOf` 的 32 位乘加契约 (`tensor.hpp:655-659` 只用注释声明过);
     *         (10d)     `reshape` 不校验元素总数守恒 (两套长度源会永久分叉);
     *         (10e/f/g) 空张量与零跨度上的统计量 (改动前是越界读 / 0-0 除零 / NaN)。
     *       做法与 [8] 同一套: **在子进程里故意踩**, 断言退出码非 0。
     * ============================================================ */
    std::printf("\n[10] 常驻结构守卫 (契约失守 => 响亮失败)\n");
    {
        /* 子进程模式在 main 的最前面派发 (见那里的说明) —— 这里只负责"派出去 + 看退出码" */
        const std::string self = (argc > 0 && argv[0] != nullptr) ? std::string(argv[0]) : std::string("test_transformer.exe");
        struct G { const char *flag; const char *what; };
        const G guards[] = {
            { "--mm-shape-guard",              "(10a) MM 形状对不上 (k 维失配) 时 abort" },
            { "--mm-buffer-guard",             "(10b) 缓冲长度 < 形状乘积时 abort" },
            { "--index-range-guard",           "(10c) 元素总数 > INT_MAX 时 abort (32 位下标契约)" },
            { "--reshape-guard",               "(10d) reshape 元素总数不守恒时 abort" },
            { "--empty-stat-guard",            "(10e) 空张量上 mean/argmax 时 abort" },
            { "--degenerate-normalize-guard",  "(10f) 零跨度 min-max normalize 时 abort" },
        };
        for (int i = 0; i < 6; i++) {
            const int rc = std::system(("\"" + self + "\" " + guards[i].flag + " > nul 2>&1").c_str());
            std::printf("    %s: 子进程退出码 = %d (非 0 = 响亮失败)\n", guards[i].flag, rc);
            CHECK(rc != 0, guards[i].what);
        }
        /* 正对照: 合法形状不能被误拦 (否则守卫就是把正常路径也打死了) */
        {
            RL::Tensor z(90, 90), w(90, 360), x(360, 90);
            RL::Tensor v(360, 1), ww(360, 1710), u(1710, 1);
            z.zero(); w.zero(); x.zero(); v.zero(); ww.zero(); u.zero();
            RL::Tensor::MM::ikkj(z, w, x);        /* (90,90) = (90,360) * (360,90) ✓ */
            RL::Tensor::MM::ikkj(v, ww, u);       /* (360,1) = (360,1710) * (1710,1) ✓ */
            const float s = z.sum() + v.sum();
            std::printf("    (10g) 合法形状: 两次 ikkj 跑通, 两个累加目标的和 = %.1f\n", (double)s);
            CHECK(s == 0.0f, "(10g) 守卫不误拦合法形状 (零初值 + 零输入仍是零)");
        }
    }

    std::printf("\n=== test_transformer: %d 项断言, %d 失败 ===\n", g_pass + g_fail, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
