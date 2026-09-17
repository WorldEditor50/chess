#ifndef RL_DIAG_H
#define RL_DIAG_H

/*
 * ============================================================================
 *  diag.h - PPO+MCTS+AlphaZero 的**诊断指标** (健康度观测, 不是棋力)
 * ============================================================================
 *
 *  为什么需要这一层: 棋力(Elo/胜率)是**滞后指标** —— 它要等到几千局之后才有分辨力,
 *  而且被"老师上限 / 对手上限 / 协议噪声"三重掩盖。本工程的实测就是例子: 4 局对局的
 *  得分率三次都压在 0% 地板上, 什么也说明不了。真正能回答"哪一层设计错了"的是**中间量**:
 *
 *    * 老师(搜索)产出的 π/Q 干不干净   -> RootDiag
 *    * 学生(网络)拟合得快不快          -> TrainDiag
 *    * 行为上敢不敢做正确交换          -> MoveBehavior
 *    * 自对弈生态有没有持续给压力      -> GameStat / 战术题
 *
 *  设计约束:
 *   1. **不依赖 Qt、不依赖 Tensor** —— 只用 std 容器。这样 RL 内核、bench、单测、
 *      GUI 四条路都能用同一份口径, 不会各自漂移。
 *   2. **每个量都能单测** —— 都是可解析验证的标量公式, 单测里用手算值钉住
 *      (见 test_diag_main.cpp)。指标本身算错是最贵的一种 bug: 它会让你去修一个
 *      不存在的问题 (本工程踩过一次: "gap 降低"曾经是假进步)。
 *   3. **数值安全**: 任何分母为 0 / 全零分布 / 非有限值都必须退化成 0 或显式标记,
 *      **绝不产生 NaN** —— NaN 进了曲线就是"曲线断了", 而人不会去查。
 *
 *  口径约定 (与 ppomcts_agent 的符号约定一致):
 *    * 节点的 `totalValue/visitCount` 是**该节点走棋方**视角的价值, 所以根的孩子
 *      对根走棋方的 Q 必须取负号 (`qForParent = -child.getQ()`)。这与
 *      PPOMCTSAgent::getPUCT 的修正、以及 SACAZAgent::getPUCT 是同一件事, 这里
 *      **不重复犯错** —— 这一条有单测钉住 (test_diag 的 qForParent 一节)。
 * ============================================================================
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace RL {
namespace Diag {

/* ---------------------------------------------------------------------------
 *  小工具
 * ------------------------------------------------------------------------- */

inline bool finite(double v)
{
    return v == v
           && v != std::numeric_limits<double>::infinity()
           && v != -std::numeric_limits<double>::infinity();
}

inline double safe(double v)
{
    return finite(v) ? v : 0.0;
}

/*
   ---- 根孩子的 Q, 换算到**根走棋方**视角 ----
   子节点存的是它自己走棋方视角的价值, 而子节点的走棋方就是根的对手 ⇒ 取负号。
   (漏掉负号会让"吃子"这类着法的 Q 整体反号 —— 本工程抓到过这个 bug, 见
   PPOMCTSAgent::getPUCT 的注释。诊断层如果再犯一次, 就会指着一个不存在的病去修。)
*/
inline double qForParent(double childTotalValue, int childVisits)
{
    if (childVisits <= 0) {
        return 0.0;
    }
    return safe(-childTotalValue / (double)childVisits);
}

/* ---------------------------------------------------------------------------
 *  1. 纯数学指标 (全部可手算验证)
 * ------------------------------------------------------------------------- */

/* 归一化分布的香农熵 (nats)。全 0 分布 -> 0 (不是 NaN): "没访问" ≠ "均匀访问" */
inline double entropy(const std::vector<double> &p)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); i++) {
        if (p[i] > 0.0) {
            sum += p[i];
        }
    }
    if (sum <= 0.0) {
        return 0.0;
    }
    double h = 0.0;
    for (std::size_t i = 0; i < p.size(); i++) {
        if (p[i] > 0.0) {
            const double q = p[i] / sum;
            h -= q * std::log(q);
        }
    }
    return safe(h);
}

/* CE(t, p) = -Σ t_i·ln p_i (只累加 t_i > 0)。p_i 用 1e-12 兜底, 避免 inf 拉爆纵轴 */
inline double crossEntropy(const std::vector<double> &t, const std::vector<double> &p)
{
    const std::size_t n = (t.size() < p.size()) ? t.size() : p.size();
    const double kFloor = 1e-12;
    double ce = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        if (t[i] <= 0.0) {
            continue;
        }
        const double pi = (p[i] > kFloor) ? p[i] : kFloor;
        ce -= t[i] * std::log(pi);
    }
    return safe(ce);
}

/* KL(t || p) = Σ t_i·ln(t_i / p_i) */
inline double klDivergence(const std::vector<double> &t, const std::vector<double> &p)
{
    const std::size_t n = (t.size() < p.size()) ? t.size() : p.size();
    const double kFloor = 1e-12;
    double kl = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        if (t[i] <= 0.0) {
            continue;
        }
        const double pi = (p[i] > kFloor) ? p[i] : kFloor;
        kl += t[i] * std::log(t[i] / pi);
    }
    if (kl < 0.0) {
        kl = 0.0;   /* 浮点误差可能给 -1e-17; KL 不会为负 */
    }
    return safe(kl);
}

/* 总体方差 (除以 n) */
inline double variance(const std::vector<double> &x)
{
    const std::size_t n = x.size();
    if (n == 0) {
        return 0.0;
    }
    double m = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        m += x[i];
    }
    m /= (double)n;
    double s = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const double d = x[i] - m;
        s += d * d;
    }
    return safe(s / (double)n);
}

/*
   Explained Variance = 1 - Var(z - v) / Var(z)

   value 头最重要的单一指标: MSE 会随目标尺度变化, 而 EV 是无量纲的
   —— "比'永远预测均值'好多少"。EV < 0 表示还不如常数预测
   (典型根因是符号翻错)。Var(z) == 0 时返回 0 (无定义, 不能报成 1)。
*/
inline double explainedVariance(const std::vector<double> &z, const std::vector<double> &v)
{
    const std::size_t n = (z.size() < v.size()) ? z.size() : v.size();
    if (n == 0) {
        return 0.0;
    }
    std::vector<double> zz(z.begin(), z.begin() + (std::ptrdiff_t)n);
    std::vector<double> vv(v.begin(), v.begin() + (std::ptrdiff_t)n);
    const double vz = variance(zz);
    if (vz <= 1e-12) {
        return 0.0;
    }
    std::vector<double> r(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
        r[i] = zz[i] - vv[i];
    }
    return safe(1.0 - variance(r) / vz);
}

/*
   变异系数 CV = std / mean。用途: MoE 专家负载均衡。
   CV ≈ 0 = 专家被均匀使用; CV 很大 = 路由塌到少数专家 (本工程实测过极端: `[0,0,542,98]`,
   有专家从不被选中)。mean == 0 时返回 0。
*/
inline double coefficientOfVariation(const std::vector<double> &x)
{
    const std::size_t n = x.size();
    if (n == 0) {
        return 0.0;
    }
    double m = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        m += x[i];
    }
    m /= (double)n;
    if (std::fabs(m) <= 1e-12) {
        return 0.0;
    }
    return safe(std::sqrt(variance(x)) / std::fabs(m));
}

/*
   ---- 价值校准 ----
   把预测值 v 分桶, 看每个桶里**实际结果 z 的均值**。完美校准时两者相等。
   比 MSE 有用之处: MSE 只说"平均差多少", 校准说"**往哪个方向偏**"。
   "怕将/怕仕 → 回避交换"这种病在这里的表现就是高预测桶的实际胜率远低于预测值。
*/
struct CalibBucket {
    double vLo = 0.0;
    double vHi = 0.0;
    double predMean = 0.0;
    double actualMean = 0.0;
    int n = 0;
};

/* 按 v ∈ [-1,1] 等宽分桶; nBuckets <= 0 时用 10 */
inline std::vector<CalibBucket> calibration(const std::vector<double> &v,
                                            const std::vector<double> &z,
                                            int nBuckets = 10)
{
    if (nBuckets <= 0) {
        nBuckets = 10;
    }
    std::vector<CalibBucket> out((std::size_t)nBuckets);
    const double width = 2.0 / (double)nBuckets;
    for (int b = 0; b < nBuckets; b++) {
        out[(std::size_t)b].vLo = -1.0 + width * (double)b;
        out[(std::size_t)b].vHi = out[(std::size_t)b].vLo + width;
    }
    const std::size_t n = (v.size() < z.size()) ? v.size() : z.size();
    for (std::size_t i = 0; i < n; i++) {
        double x = v[i];
        if (!finite(x)) {
            continue;
        }
        if (x < -1.0) { x = -1.0; }
        if (x > 1.0) { x = 1.0; }
        int b = (int)((x + 1.0) / width);
        if (b < 0) { b = 0; }
        if (b >= nBuckets) { b = nBuckets - 1; }
        CalibBucket &bk = out[(std::size_t)b];
        bk.n++;
        bk.predMean += x;
        bk.actualMean += z[i];
    }
    for (int b = 0; b < nBuckets; b++) {
        CalibBucket &bk = out[(std::size_t)b];
        if (bk.n > 0) {
            bk.predMean /= (double)bk.n;
            bk.actualMean /= (double)bk.n;
        }
    }
    return out;
}

/* 校准误差 = 各桶 |预测均值 − 实际均值| 按样本数加权平均 (0 = 完美) */
inline double calibrationError(const std::vector<CalibBucket> &buckets)
{
    double num = 0.0;
    int tot = 0;
    for (std::size_t i = 0; i < buckets.size(); i++) {
        if (buckets[i].n <= 0) {
            continue;
        }
        num += std::fabs(buckets[i].predMean - buckets[i].actualMean)
               * (double)buckets[i].n;
        tot += buckets[i].n;
    }
    if (tot <= 0) {
        return 0.0;
    }
    return safe(num / (double)tot);
}

/* ---------------------------------------------------------------------------
 *  2. 逐手搜索诊断 (老师体检)
 * ------------------------------------------------------------------------- */

struct RootDiag {
    bool valid = false;
    int children = 0;               /* 已展开的根孩子数 */
    int totalVisits = 0;            /* ΣN */
    double topShare = 0.0;          /* N_max / ΣN: >0.7 说明过早锁定 */
    double visitEntropy = 0.0;      /* 访问分布熵 (nats) */
    double priorEntropy = 0.0;      /* 先验(在已展开集合上重归一)的熵 */
    double priorKl = 0.0;           /* KL(π_visit || π_prior): 搜索纠正网络的程度 */
    int legalCount = 0;             /* 根的完整合法着法数 (= 分支数) */
    double expandCoverage = 0.0;    /* children / legalCount */

    /* ---- "胆小不吃子" 的核心读数 ---- */
    int captureCount = 0;           /* 已展开的吃子着法数 */
    int captureVisits = 0;          /* 吃子着法拿到的总访问数 */
    double captureVisitShare = 0.0; /* captureVisits / ΣN */
    double qCaptureMax = 0.0;       /* 吃子着法里最大的 Q (根走棋方视角) */
    double qQuietMax = 0.0;         /* 退让着法里最大的 Q */
    int bestIsCapture = 0;          /* 访问最多的着法是不是吃子 */
};

/* ---------------------------------------------------------------------------
 *  3. 逐手行为诊断 ("敢不敢做正确交换")
 * ------------------------------------------------------------------------- */

struct MoveBehavior {
    double materialDelta = 0.0;     /* 走子方视角的材质变化 (吃子为正) */
    int isCapture = 0;
    int gaveCheck = 0;
    int wasInCheck = 0;
    double vBefore = 0.0;           /* 走之前, 走子方视角的网络估值 */
    double vAfter = 0.0;            /* 走之后, 仍是走子方视角 */
};

/* ---------------------------------------------------------------------------
 *  4. 逐次训练诊断 (学生体检)
 * ------------------------------------------------------------------------- */

struct TrainDiag {
    long long step = 0;
    double policyCe = 0.0;          /* CE(π_target, π_net) */
    double policyKl = 0.0;          /* KL(π_target || π_net) */
    double policyEntropy = 0.0;     /* 网络策略熵 —— 秒归零 = 塌了 */
    double valueMse = 0.0;
    double valueEv = 0.0;           /* explained variance */
    double moeLoadCv = 0.0;         /* MoE 专家负载 CV */
    /*
       clip fraction / ratio **对当前这条管线不适用**: actor 损失是纯交叉熵
       (AlphaZero 蒸馏口径), 没有 importance ratio。显式标成 -1 而不是填 0,
       免得看曲线的人以为"clip 永远 0 = 更新停滞"。真要用 PPO clip 得先给回放样本
       存 log π_old (见 rl/ppo.h)。
    */
    double clipFraction = -1.0;
};

/* ---------------------------------------------------------------------------
 *  5. 对局级诊断 (自对弈生态)
 * ------------------------------------------------------------------------- */

struct GameStat {
    long long gameIndex = 0;
    int result = 0;                 /* Chess::RESULT_* (1=红胜 2=黑胜 3=和) */
    int plies = 0;
    unsigned long long openingHash = 0;   /* 第 8 手局面的哈希 (开局多样性去重) */
    int capturesByRed = 0;
    int capturesByBlack = 0;
};

/* ---------------------------------------------------------------------------
 *  6. 汇总 (只累加, 不存历史序列 —— 几百万手也不涨内存)
 * ------------------------------------------------------------------------- */

class Aggregates
{
public:
    void addRoot(const RootDiag &d)
    {
        if (!d.valid) { return; }
        m_rootN++;
        m_topShare += d.topShare;
        m_visitEntropy += d.visitEntropy;
        m_priorEntropy += d.priorEntropy;
        m_priorKl += d.priorKl;
        m_children += (double)d.children;
        m_legalCount += (double)d.legalCount;
        m_coverage += d.expandCoverage;
        m_captureShare += d.captureVisitShare;
        m_qCapMax += d.qCaptureMax;
        m_qQuietMax += d.qQuietMax;
        m_qGapSum += (d.qCaptureMax - d.qQuietMax);
        if (d.captureCount > 0) { m_rootWithCapture++; }
        if (d.bestIsCapture) { m_bestIsCapture++; }
    }

    void addBehavior(const MoveBehavior &b)
    {
        m_behN++;
        m_materialDelta += b.materialDelta;
        if (b.isCapture) { m_captureChosen++; }
        if (b.gaveCheck) { m_gaveCheck++; }
        if (b.wasInCheck) { m_inCheck++; }
        m_vBefore += b.vBefore;
        m_vAfter += b.vAfter;
        /*
           额外累积两个二阶量, 用来判断"每手 V 增益"到底该不该当真:
             std(vBefore)          —— V 有没有随局面变化 (≈0 = V 是常数)
             corr(vBefore, vAfter) —— ≈ -1 且 std≈0 时, "增益"只是在读常数偏置
           这一条是实测撞出来的: 随机权重与 BC 权重下都出现 mean(vAfter) ≈ -mean(vBefore),
           于是"每手 V 增益 ≈ -2·mean(V)"看起来像"越走越差", 实际是 V 不依赖局面。
           没有这两个量, 那个数字会被当成"agent 在把自己走坏"。
        */
        m_vBefore2 += b.vBefore * b.vBefore;
        m_vAfter2 += b.vAfter * b.vAfter;
        m_vCross += b.vBefore * b.vAfter;
    }

    void addTrain(const TrainDiag &t)
    {
        m_trainN++;
        m_policyCe += t.policyCe;
        m_policyKl += t.policyKl;
        m_policyEntropy += t.policyEntropy;
        m_valueMse += t.valueMse;
        m_valueEv += t.valueEv;
        m_moeLoadCv += t.moeLoadCv;
    }

    void addGame(const GameStat &g)
    {
        m_games++;
        if (g.result == 1) { m_redWins++; }
        else if (g.result == 2) { m_blackWins++; }
        else if (g.result == 3) { m_draws++; }
        m_plies += (double)g.plies;
        m_openingHashes.push_back(g.openingHash);
    }

    long long rootSamples() const { return m_rootN; }
    long long behaviorSamples() const { return m_behN; }
    long long trainSamples() const { return m_trainN; }
    long long games() const { return m_games; }

    double meanTopShare() const { return div(m_topShare, (double)m_rootN); }
    double meanVisitEntropy() const { return div(m_visitEntropy, (double)m_rootN); }
    double meanPriorEntropy() const { return div(m_priorEntropy, (double)m_rootN); }
    double meanPriorKl() const { return div(m_priorKl, (double)m_rootN); }
    double meanChildren() const { return div(m_children, (double)m_rootN); }
    double meanLegalCount() const { return div(m_legalCount, (double)m_rootN); }
    double meanCoverage() const { return div(m_coverage, (double)m_rootN); }
    double meanCaptureVisitShare() const { return div(m_captureShare, (double)m_rootN); }
    double meanQCaptureMax() const { return div(m_qCapMax, (double)m_rootN); }
    double meanQQuietMax() const { return div(m_qQuietMax, (double)m_rootN); }
    /* **胆小的核心读数**: 长期为负 = 搜索认为吃子不如退让 */
    double meanQCaptureMinusQuiet() const { return div(m_qGapSum, (double)m_rootN); }
    double rootWithCaptureRate() const { return div((double)m_rootWithCapture, (double)m_rootN); }
    double bestIsCaptureRate() const { return div((double)m_bestIsCapture, (double)m_rootN); }
    double captureChosenRate() const { return div((double)m_captureChosen, (double)m_behN); }
    double checkRate() const { return div((double)m_gaveCheck, (double)m_behN); }
    double inCheckRate() const { return div((double)m_inCheck, (double)m_behN); }
    double meanMaterialDelta() const { return div(m_materialDelta, (double)m_behN); }
    double meanValueGain() const { return div(m_vAfter - m_vBefore, (double)m_behN); }
    double meanVBefore() const { return div(m_vBefore, (double)m_behN); }
    /* V(走子方视角) 的标准差: ≈0 说明 critic 基本是常数 */
    double stdVBefore() const
    {
        if (m_behN <= 0) { return 0.0; }
        const double n = (double)m_behN;
        const double m = m_vBefore / n;
        const double v = m_vBefore2 / n - m * m;
        return safe(v > 0.0 ? std::sqrt(v) : 0.0);
    }
    /*
       corr(V_before, V_after)。判读:
         ≈ -1 且 stdVBefore() ≈ 0  -> "每手 V 增益"没有信息 (读的是常数偏置)
         ≈ +1                      -> V 认为走子方越走越好
         ≈ -1 但 std 大            -> V 真的认为每手都在变差 (值得查 PBRS/符号)
       分母为 0 时返回 0。
    */
    double corrVBeforeAfter() const
    {
        if (m_behN <= 0) { return 0.0; }
        const double n = (double)m_behN;
        const double mb = m_vBefore / n;
        const double ma = m_vAfter / n;
        const double vb = m_vBefore2 / n - mb * mb;
        const double va = m_vAfter2 / n - ma * ma;
        if (vb <= 1e-12 || va <= 1e-12) { return 0.0; }
        const double cov = m_vCross / n - mb * ma;
        return safe(cov / std::sqrt(vb * va));
    }
    double meanPolicyCe() const { return div(m_policyCe, (double)m_trainN); }
    double meanPolicyKl() const { return div(m_policyKl, (double)m_trainN); }
    double meanPolicyEntropy() const { return div(m_policyEntropy, (double)m_trainN); }
    double meanValueMse() const { return div(m_valueMse, (double)m_trainN); }
    double meanValueEv() const { return div(m_valueEv, (double)m_trainN); }
    double meanMoeLoadCv() const { return div(m_moeLoadCv, (double)m_trainN); }
    double drawRate() const { return div((double)m_draws, (double)m_games); }
    double redWinRate() const { return div((double)m_redWins, (double)m_games); }
    double blackWinRate() const { return div((double)m_blackWins, (double)m_games); }
    double meanPlies() const { return div(m_plies, (double)m_games); }
    long long distinctOpenings() const
    {
        if (m_openingHashes.empty()) { return 0; }
        std::vector<unsigned long long> v = m_openingHashes;
        std::sort(v.begin(), v.end());
        return (long long)(std::unique(v.begin(), v.end()) - v.begin());
    }

    /*
       打印仪表盘。每个区块都给出"红灯阈值"的判读, 让读数能直接对上设计问题
       —— 光有数字没有判据, 等于没接仪表盘。
    */
    void print(const char *tag) const
    {
        std::printf("\n=== 诊断仪表盘 [%s] ===\n", (tag != nullptr) ? tag : "");
        std::printf("  样本: 根 %lld 手, 行为 %lld 手, 训练 %lld 次, 对局 %lld\n",
                    m_rootN, m_behN, m_trainN, m_games);
        if (m_rootN > 0) {
            std::printf("  -- 搜索(老师) --\n");
            std::printf("    分支数 %.1f | 展开覆盖率 %.3f | top1 访问份额 %.3f %s\n",
                        meanLegalCount(), meanCoverage(), meanTopShare(),
                        (meanTopShare() > 0.70) ? "<- >0.70: 过早锁定" : "");
            std::printf("    访问熵 %.3f nats | 先验熵 %.3f nats | KL(visit||prior) %.3f %s\n",
                        meanVisitEntropy(), meanPriorEntropy(), meanPriorKl(),
                        (meanPriorKl() < 0.05) ? "<- ~0: 搜索没纠正网络(白跑)" : "");
            std::printf("    吃子访问份额 %.3f | 吃子最优率 %.3f\n",
                        meanCaptureVisitShare(), bestIsCaptureRate());
            std::printf("    Q(吃子)max %.4f | Q(退让)max %.4f | 差 %.4f %s\n",
                        meanQCaptureMax(), meanQQuietMax(), meanQCaptureMinusQuiet(),
                        (meanQCaptureMinusQuiet() < -0.02) ? "<- 负: 搜索怕吃子!" : "");
        }
        if (m_behN > 0) {
            std::printf("  -- 行为 --\n");
            std::printf("    每手材质变化 %.4f | 选到吃子的比例 %.3f | 叫将率 %.3f | 被将率 %.3f\n",
                        meanMaterialDelta(), captureChosenRate(), checkRate(), inCheckRate());
            std::printf("    每手 V 增益 %.4f (V 均值 %.4f, 标准差 %.4f, corr %.3f)\n",
                        meanValueGain(), meanVBefore(), stdVBefore(), corrVBeforeAfter());
            /*
               这一行是"防止把常数偏置读成趋势"的护栏: V 的标准差接近 0 时,
               "每手 V 增益"没有任何信息量 (它只是 2 倍的均值), 必须先修 value。
            */
            if (m_behN > 0 && stdVBefore() < 0.02) {
                std::printf("      <- V 标准差 ≈ 0: critic 基本是常数, 上面那个'增益'"
                            "不反映局面好坏 (先修 value 再看)\n");
            }
        }
        if (m_trainN > 0) {
            std::printf("  -- 训练(学生) --\n");
            std::printf("    policy CE %.4f | KL %.4f | 策略熵 %.4f nats | MoE 负载 CV %.4f\n",
                        meanPolicyCe(), meanPolicyKl(), meanPolicyEntropy(), meanMoeLoadCv());
            std::printf("    value MSE %.6f | EV %.4f %s\n",
                        meanValueMse(), meanValueEv(),
                        (meanValueEv() < 0.0) ? "<- <0: value 不如常数, 查符号" : "");
        }
        if (m_games > 0) {
            std::printf("  -- 自对弈生态 --\n");
            std::printf("    红胜 %.3f | 黑胜 %.3f | 和 %.3f | 平均手数 %.1f\n",
                        redWinRate(), blackWinRate(), drawRate(), meanPlies());
            std::printf("    开局种类 %lld / %lld 局 %s\n",
                        distinctOpenings(), m_games,
                        (m_games >= 10 && distinctOpenings() * 4 < m_games)
                            ? "<- 开局坍塌" : "");
        }
        std::printf("\n");
    }

private:
    static double div(double a, double b) { return (b > 0.0) ? safe(a / b) : 0.0; }

    long long m_rootN = 0, m_behN = 0, m_trainN = 0, m_games = 0;
    double m_topShare = 0.0, m_visitEntropy = 0.0, m_priorEntropy = 0.0;
    double m_priorKl = 0.0, m_captureShare = 0.0, m_qCapMax = 0.0, m_qQuietMax = 0.0;
    double m_qGapSum = 0.0, m_children = 0.0, m_legalCount = 0.0, m_coverage = 0.0;
    long long m_rootWithCapture = 0, m_bestIsCapture = 0;
    double m_materialDelta = 0.0, m_vBefore = 0.0, m_vAfter = 0.0;
    double m_vBefore2 = 0.0, m_vAfter2 = 0.0, m_vCross = 0.0;
    long long m_captureChosen = 0, m_gaveCheck = 0, m_inCheck = 0;
    double m_policyCe = 0.0, m_policyKl = 0.0, m_policyEntropy = 0.0;
    double m_valueMse = 0.0, m_valueEv = 0.0, m_moeLoadCv = 0.0;
    long long m_redWins = 0, m_blackWins = 0, m_draws = 0;
    double m_plies = 0.0;
    std::vector<unsigned long long> m_openingHashes;
};

/* ---------------------------------------------------------------------------
 *  7. CSV 输出
 *
 *  两种格式:
 *    * 逐手宽表   —— 一行一手, 含 RootDiag + MoveBehavior 全部量 (pandas 直接看)
 *    * TensorBoard 长表 —— `tag,step,value`。TB 的 CSV 加载按 tag 分组, 与
 *      `add_scalar(tag, value, step)` 语义一一对应。
 *      **不写 protobuf event 文件**: 那要引入 protobuf 依赖, 而本工程刻意只依赖
 *      Widgets+Sql (见 metricsview.h 里对 Qt Charts 的同类取舍)。
 * ------------------------------------------------------------------------- */

class CsvWriter
{
public:
    CsvWriter() = default;
    CsvWriter(const CsvWriter &) = delete;
    CsvWriter &operator=(const CsvWriter &) = delete;
    ~CsvWriter() { close(); }

    bool open(const std::string &path, const std::string &header)
    {
        close();
        m_fp = std::fopen(path.c_str(), "wb");
        if (m_fp == nullptr) {
            return false;
        }
        if (!header.empty()) {
            std::fprintf(m_fp, "%s\n", header.c_str());
        }
        return true;
    }

    void row(const std::string &line)
    {
        if (m_fp != nullptr) {
            std::fprintf(m_fp, "%s\n", line.c_str());
        }
    }

    void close()
    {
        if (m_fp != nullptr) {
            std::fclose(m_fp);
            m_fp = nullptr;
        }
    }

    bool isOpen() const { return m_fp != nullptr; }

private:
    std::FILE *m_fp = nullptr;
};

/* TensorBoard 长表的一行 */
inline void scalarRow(CsvWriter &w, const char *tag, long long step, double value)
{
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%s,%lld,%.6f", tag, step, safe(value));
    w.row(buf);
}

/* 逐手宽表的表头 (列数必须与 moveRow 的格式串一致) */
inline const char *moveHeader()
{
    return "game,ply,color,children,legal,coverage,top_share,visit_entropy,"
           "prior_entropy,prior_kl,q_capture_max,q_quiet_max,capture_n,"
           "capture_visits,capture_visit_share,best_is_capture,"
           "material_delta,is_capture,gave_check,was_in_check,v_before,v_after";
}

/* 逐手宽表的一行: 列顺序与 moveHeader() 严格一致 */
inline void moveRow(CsvWriter &w, long long gameIndex, int ply, int color,
                    const RootDiag &r, const MoveBehavior &b)
{
    /*
       21 + 1 = 22 列, 与 moveHeader() 一一对应。格式串与参数的**顺序和个数**必须
       完全对齐 —— snprintf 的参数错位不会报错, 只会静默打印垃圾 (所以下面把
       每个参数单独写一行, 方便肉眼核对)。
    */
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "%lld,%d,%d,"          /* game, ply, color */
                  "%d,%d,%.6f,"          /* children, legal, coverage */
                  "%.6f,%.6f,%.6f,%.6f," /* top_share, visit_entropy, prior_entropy, prior_kl */
                  "%.6f,%.6f,"           /* q_capture_max, q_quiet_max */
                  "%d,%d,%.6f,%d,"       /* capture_n, capture_visits, capture_visit_share, best_is_capture */
                  "%.6f,%d,%d,%d,"       /* material_delta, is_capture, gave_check, was_in_check */
                  "%.6f,%.6f",           /* v_before, v_after */
                  gameIndex, ply, color,
                  r.children, r.legalCount, r.expandCoverage,
                  r.topShare, r.visitEntropy, r.priorEntropy, r.priorKl,
                  r.qCaptureMax, r.qQuietMax,
                  r.captureCount, r.captureVisits, r.captureVisitShare, r.bestIsCapture,
                  b.materialDelta, b.isCapture, b.gaveCheck, b.wasInCheck,
                  b.vBefore, b.vAfter);
    w.row(buf);
}

} // namespace Diag
} // namespace RL

#endif // RL_DIAG_H
