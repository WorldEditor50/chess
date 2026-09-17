/*
 * ============================================================================
 *  test_diag_main.cpp - 诊断指标的**解析验证** (手算值钉死)
 * ============================================================================
 *
 *  为什么指标本身要单测: 算错的指标比没有指标更贵 —— 它会让你去修一个不存在的问题。
 *  本工程已经吃过一次亏 ("gap 降低"曾经是假进步: 把 V 压成常数就能降 gap)。所以
 *  rl/diag.h 里每个公式都用手算值断言, 尤其是:
 *
 *    * entropy / crossEntropy / klDivergence 的定义与边界 (全零分布、零概率目标)
 *    * explainedVariance 的符号 (EV<0 必须真的表示"不如常数预测")
 *    * coefficientOfVariation (MoE 专家负载)
 *    * calibration 的分桶边界 (v = ±1 必须落进首尾桶)
 *    * **qForParent 的负号** —— 子节点存的是它自己走棋方的价值, 诊断层换算到父方
 *      视角必须取负; 漏掉就与 PPOMCTSAgent::getPUCT 那次 bug 同型
 *    * CSV 的**列数与表头一致** (snprintf 参数错位不会报错, 只会静默打垃圾)
 *
 *  还包含一个端到端的接线检查: 真的搜一个局面, 从 PPOMCTSAgent::rootDiag 里读
 *  RootDiag, 断言"访问数之和 == 模拟次数""覆盖率 ≤ 1""吃子份额在 [0,1]",
 *  以及 Dirichlet 噪声开关确实改变了根先验 (evalRootNoise 的 A/B 通路)。
 *
 *  只依赖 src/ 与头文件, 不依赖 Qt。
 * ============================================================================
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/diag.h"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
        }                                                            \
    } while (0)

#define CHECK_EQ(a, b, msg) do {                                     \
        ++g_checks;                                                  \
        long long _a = (long long)(a);                                \
        long long _b = (long long)(b);                                \
        if (_a != _b) {                                              \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s: %lld != %lld  (%s:%d)\n",        \
                        (msg), _a, _b, __FILE__, __LINE__);           \
        }                                                            \
    } while (0)

/* 浮点近似断言 */
static void checkNear(const char *msg, double got, double want, double tol)
{
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        ++g_failed;
        std::printf("  [FAIL] %s: got %.8f want %.8f (tol %.1e)\n",
                    msg, got, want, tol);
    }
}

static void secEntropy()
{
    std::printf("\n[1] entropy / crossEntropy / klDivergence\n");
    using namespace RL::Diag;

    /* 均匀 4 项 = ln 4 */
    std::vector<double> u4(4, 0.25);
    checkNear("H(uniform 4) = ln4", entropy(u4), std::log(4.0), 1e-12);

    /* one-hot = 0 */
    std::vector<double> oh(4, 0.0);
    oh[2] = 1.0;
    checkNear("H(one-hot) = 0", entropy(oh), 0.0, 1e-12);

    /* 全零 -> 0 (不是 NaN); "没访问"与"均匀访问"必须区分开 */
    std::vector<double> z(4, 0.0);
    checkNear("H(all zero) = 0", entropy(z), 0.0, 1e-12);
    CHECK(finite(entropy(z)), "全零分布的熵必须是有限值 (不能是 NaN)");

    /* CE(t, p) 在 t == p 时等于 H(t) */
    checkNear("CE(u,u) = H(u)", crossEntropy(u4, u4), std::log(4.0), 1e-12);

    /* CE 对"目标有质量但预测为 0"要给很大的有限值, 不能 inf */
    std::vector<double> p0(4, 0.0);
    p0[0] = 1.0;
    const double ce = crossEntropy(oh, p0);          /* 目标在 2, 预测全在 0 */
    CHECK(finite(ce), "预测为 0 的 CE 必须是有限值 (否则曲线纵轴会被拉爆)");
    CHECK(ce > 20.0, "预测为 0 的 CE 应当很大");

    /* KL(t||t) = 0; KL 对称性不成立但必须非负 */
    checkNear("KL(u,u) = 0", klDivergence(u4, u4), 0.0, 1e-12);
    std::vector<double> a(2, 0.0), b(2, 0.0);
    a[0] = 0.75; a[1] = 0.25;
    b[0] = 0.5;  b[1] = 0.5;
    /* KL = 0.75*ln(1.5) + 0.25*ln(0.5) */
    checkNear("KL(a||b)", klDivergence(a, b),
              0.75 * std::log(1.5) + 0.25 * std::log(0.5), 1e-12);
    CHECK(klDivergence(a, b) >= 0.0, "KL 必须非负");
}

static void secValue()
{
    std::printf("\n[2] explainedVariance (value 头是否比常数预测好)\n");
    using namespace RL::Diag;

    /* 完美预测: EV = 1 */
    std::vector<double> z;
    std::vector<double> v;
    for (int i = 0; i < 20; i++) {
        const double x = -1.0 + 0.1 * i;
        z.push_back(x);
        v.push_back(x);
    }
    checkNear("EV(perfect) = 1", explainedVariance(z, v), 1.0, 1e-12);

    /* 预测常数 = 均值: EV = 0 (这就是"永远预测均值"的基线) */
    double m = 0.0;
    for (std::size_t i = 0; i < z.size(); i++) { m += z[i]; }
    m /= (double)z.size();
    std::vector<double> vc(z.size(), m);
    checkNear("EV(常数=均值) = 0", explainedVariance(z, vc), 0.0, 1e-9);

    /* 符号翻错: v = -z -> EV 必须为负 (本工程抓过这类 bug) */
    std::vector<double> vneg(z.size(), 0.0);
    for (std::size_t i = 0; i < z.size(); i++) { vneg[i] = -z[i]; }
    CHECK(explainedVariance(z, vneg) < 0.0,
          "预测与目标反号时 EV 必须为负 (符号错误的判据)");

    /* 目标全相同 -> EV 无定义, 必须返回 0 而不是 1 */
    std::vector<double> zc(10, 0.5);
    std::vector<double> vv(10, 0.5);
    checkNear("Var(z)=0 -> EV=0", explainedVariance(zc, vv), 0.0, 1e-12);

    /* MSE 与 EV 是两件事: 常数预测的 MSE 可以很小而 EV 仍是 0 */
    double mse = 0.0;
    for (std::size_t i = 0; i < z.size(); i++) {
        const double d = z[i] - vc[i];
        mse += d * d;
    }
    mse /= (double)z.size();
    CHECK(mse > 0.0, "常数预测的 MSE 仍大于 0");
}

static void secMoeAndCalib()
{
    std::printf("\n[3] coefficientOfVariation / calibration\n");
    using namespace RL::Diag;

    /* 完全均匀 -> CV = 0 */
    std::vector<double> even(4, 100.0);
    checkNear("CV(uniform) = 0", coefficientOfVariation(even), 0.0, 1e-12);

    /* 一个专家吃满, 其余为 0 -> CV 很大 (本工程实测过 [0,0,542,98] 这种) */
    std::vector<double> skew(4, 0.0);
    skew[2] = 542.0;
    skew[3] = 98.0;
    CHECK(coefficientOfVariation(skew) > 0.8,
          "负载塌到少数专家时 CV 必须明显大于 0");
    /* 手动核对: mean=160, var=((160^2)+(160^2)+(382^2)+(62^2))/4 */
    {
        const double m = (0.0 + 0.0 + 542.0 + 98.0) / 4.0;
        double var = 0.0;
        const double x[4] = { 0.0, 0.0, 542.0, 98.0 };
        for (int i = 0; i < 4; i++) { var += (x[i] - m) * (x[i] - m); }
        var /= 4.0;
        checkNear("CV([0,0,542,98])", coefficientOfVariation(skew),
                  std::sqrt(var) / m, 1e-12);
    }

    /* 全 0 -> 0 (不能 NaN) */
    std::vector<double> zero4(4, 0.0);
    checkNear("CV(all zero) = 0", coefficientOfVariation(zero4), 0.0, 1e-12);

    /* ---- 校准 ---- */
    /* 完美校准的样本: 预测 == 实际 */
    std::vector<double> pv, az;
    for (int i = 0; i < 100; i++) {
        const double x = -0.9 + 0.018 * i;
        pv.push_back(x);
        az.push_back(x);
    }
    std::vector<CalibBucket> bk = calibration(pv, az, 10);
    CHECK_EQ((long long)bk.size(), 10, "10 个桶");
    checkNear("完美校准的校准误差 = 0", calibrationError(bk), 0.0, 1e-9);
    int tot = 0;
    for (std::size_t i = 0; i < bk.size(); i++) { tot += bk[i].n; }
    CHECK_EQ(tot, 100, "所有样本都要落进某个桶");

    /* 边界: v = -1 落第 0 桶, v = +1 落最后一桶 */
    {
        std::vector<double> e1, e2;
        e1.push_back(-1.0); e2.push_back(0.0);
        e1.push_back(1.0);  e2.push_back(0.0);
        std::vector<CalibBucket> b2 = calibration(e1, e2, 10);
        CHECK_EQ(b2[0].n, 1, "v = -1 落进第 0 桶");
        CHECK_EQ(b2[9].n, 1, "v = +1 落进最后一桶");
    }

    /* 系统性高估 (预测 +0.4, 实际 0): 校准误差应当约等于 0.4 */
    {
        std::vector<double> p2, a2;
        for (int i = 0; i < 50; i++) { p2.push_back(0.4); a2.push_back(0.0); }
        std::vector<CalibBucket> b3 = calibration(p2, a2, 10);
        checkNear("高估 0.4 的校准误差", calibrationError(b3), 0.4, 1e-9);
    }
}

static void secQForParent()
{
    std::printf("\n[4] qForParent 的负号 (与 PUCT 同型, 不能再错一次)\n");
    using namespace RL::Diag;

    /*
       子节点存的是**它自己走棋方**视角的累计价值。假设 4 次访问累计 -0.8,
       那么子节点视角 Q = -0.2, 而对**根的走棋方**来说 Q = +0.2 (对手越差我越好)。
    */
    checkNear("qForParent(-0.8, 4) = +0.2", qForParent(-0.8, 4), 0.2, 1e-12);
    checkNear("qForParent(+0.8, 4) = -0.2", qForParent(+0.8, 4), -0.2, 1e-12);
    checkNear("零访问 -> 0", qForParent(1.0, 0), 0.0, 1e-12);
    /* 与 PUCT 的判据一致: 父节点偏好"存储值为负"的孩子 */
    CHECK(qForParent(-0.8, 4) > qForParent(+0.8, 4),
          "父节点必须偏好存储值为负 (即对手视角差) 的孩子");
}

static void secCsvShape()
{
    std::printf("\n[5] CSV: 表头列数必须与数据行一致\n");
    using namespace RL::Diag;

    auto countCommas = [](const char *s) {
        int n = 0;
        for (const char *p = s; *p != '\0'; p++) {
            if (*p == ',') { n++; }
        }
        return n;
    };

    const char *hdr = moveHeader();
    const int hdrCols = countCommas(hdr) + 1;
    CHECK_EQ(hdrCols, 22, "moveHeader 有 22 列");

    /* 造一行数据, 数它的逗号 —— snprintf 参数错位不会报错, 只能靠这条断言 */
    RootDiag r;
    r.valid = true; r.children = 30; r.legalCount = 38; r.expandCoverage = 0.79;
    r.topShare = 0.12; r.visitEntropy = 3.1; r.priorEntropy = 3.3; r.priorKl = 0.4;
    r.qCaptureMax = 0.05; r.qQuietMax = 0.02; r.captureCount = 5; r.captureVisits = 12;
    r.captureVisitShare = 0.15; r.bestIsCapture = 1;
    MoveBehavior b;
    b.materialDelta = 0.3; b.isCapture = 1; b.gaveCheck = 0; b.wasInCheck = 0;
    b.vBefore = 0.1; b.vAfter = 0.2;

    CsvWriter w;
    const std::string path = "test_diag_tmp.csv";
    CHECK(w.open(path, moveHeader()), "CSV 能打开");
    moveRow(w, 3, 17, Stone::COLOR_BLACK, r, b);
    w.close();

    /* 读回来比对列数 */
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    CHECK(fp != nullptr, "CSV 落盘");
    if (fp != nullptr) {
        char line[2048];
        int lineNo = 0;
        int dataCols = -1;
        while (std::fgets(line, sizeof(line), fp) != nullptr) {
            lineNo++;
            if (lineNo == 2) { dataCols = countCommas(line) + 1; }
        }
        std::fclose(fp);
        CHECK_EQ(lineNo, 2, "CSV 有表头 + 1 行数据");
        CHECK_EQ(dataCols, hdrCols, "数据行列数 == 表头列数");
    }
    std::remove(path.c_str());

    /* TensorBoard 长表: tag,step,value */
    {
        CsvWriter tw;
        const std::string tp = "test_diag_tb_tmp.csv";
        CHECK(tw.open(tp, "tag,step,value"), "TB CSV 能打开");
        scalarRow(tw, "policy_ce", 7, 1.25);
        tw.close();
        std::FILE *f2 = std::fopen(tp.c_str(), "rb");
        CHECK(f2 != nullptr, "TB CSV 落盘");
        if (f2 != nullptr) {
            char line[512];
            bool sawData = false;
            while (std::fgets(line, sizeof(line), f2) != nullptr) {
                if (std::strncmp(line, "policy_ce,7,", 12) == 0) { sawData = true; }
            }
            std::fclose(f2);
            CHECK(sawData, "TB CSV 的 tag,step,value 格式正确");
        }
        std::remove(tp.c_str());
    }
}

static void secAgentWiring()
{
    std::printf("\n[6] 端到端接线: PPOMCTSAgent::rootDiag\n");
    using namespace RL::Diag;

    /*
       小网络 (hidden=16, withGrad=false): 本测试量的是**接线与不变量**, 不是棋力,
       用小网络才跑得动 (默认 64 是 5.2e7 参数)。
    */
    Chess board;
    board.reset();
    PPOMCTSAgent ag(board, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, false);

    /* 没有树的时候必须返回 false (而不是给一份垃圾诊断) */
    {
        RootDiag d;
        CHECK(!ag.rootDiag(d), "没有搜索过 -> rootDiag 返回 false");
    }

    const int sims = 60;
    board.sideToMove = Stone::COLOR_RED;
    ag.resetSearchTree();
    const Step mv = ag.selectMove(Stone::COLOR_RED, sims, 0.0f);
    CHECK(mv.valid, "初始局面能搜出一个着法");

    RootDiag d;
    const bool ok = ag.rootDiag(d);
    CHECK(ok, "rootDiag 成功");
    CHECK(d.valid, "valid 标记");
    CHECK(d.children > 0, "至少有展开的孩子");
    CHECK(d.legalCount >= d.children, "合法集 >= 已展开孩子 (覆盖率 <= 1)");
    CHECK(d.expandCoverage > 0.0 && d.expandCoverage <= 1.0, "覆盖率在 (0,1]");
    CHECK(d.topShare > 0.0 && d.topShare <= 1.0, "top-1 份额在 (0,1]");
    CHECK(d.visitEntropy >= 0.0, "访问熵非负");
    CHECK(d.captureVisitShare >= 0.0 && d.captureVisitShare <= 1.0, "吃子份额在 [0,1]");
    CHECK(d.priorKl >= 0.0, "KL 非负");
    CHECK(finite(d.qCaptureMax) && finite(d.qQuietMax), "Q 是有限值");

    /*
       关键不变量: 根孩子访问数之和 == 该根的模拟次数。
       本轮每模拟必给根 +1 (backup 沿路径走到根), 所以两者必须相等 —— 这条能同时
       抓住"backup 漏了根""诊断读错了节点"两类错误。
    */
    CHECK_EQ(d.totalVisits, sims, "根访问数之和 == 模拟次数");

    /* 手算: 遍历节点求和, 与 rootDiag 报的一致 */
    {
        long long sum = 0;
        const int root = ag.currentRoot();
        CHECK(root >= 0, "有当前根");
        if (root >= 0) {
            for (std::size_t i = 0; i < ag.nodes[(std::size_t)root].childIDs.size(); i++) {
                sum += ag.nodes[(std::size_t)ag.nodes[(std::size_t)root].childIDs[i]].visitCount;
            }
        }
        CHECK_EQ(sum, (long long)d.totalVisits, "手工求和的访问数一致");
    }

    /* 吃子分组的手工核对: 用 step.nextId 判定, 与 rootDiag 的计数一致 */
    {
        int capN = 0, capVisits = 0;
        const int root = ag.currentRoot();
        for (std::size_t i = 0; i < ag.nodes[(std::size_t)root].childIDs.size(); i++) {
            const PPOMCTSAgent::AZNode &c =
                ag.nodes[(std::size_t)ag.nodes[(std::size_t)root].childIDs[i]];
            if (c.step.nextId != Stone::ID_NONE) { capN++; capVisits += c.visitCount; }
        }
        CHECK_EQ(capN, d.captureCount, "吃子孩子数与手工统计一致");
        CHECK_EQ(capVisits, d.captureVisits, "吃子访问数与手工统计一致");
    }

    /* printSortedRoot 只是打印, 不该崩 (冒烟) */
    ag.printSortedRoot(3);
    CHECK(true, "printSortedRoot 不崩");

    /*
       Dirichlet A/B 通路: evalRootNoise 必须真的改变根访问分布, 否则诊断矩阵里
       "Dirichlet 有效性"那一项会永远报 0 差异 (而实际是开关没接上)。

       两个坑 (第一版就是这么写错的):
       1. **哈希不能用"下标"**: 模拟数 < 分支数 时, 每个被展开的孩子都恰好 1 次访问,
          于是 Σ N×(i+1) 恒等于 1+2+...+sims —— 与"展开了哪些孩子"无关, 哈希退化。
          必须把**动作下标**放进哈希里 (那才代表"哪些着法被试过")。
       2. **模拟数要大于分支数**: 初始局面 44 个合法着法, 模拟 40 次时分布被强制成
          "每个孩子 1 次"的均匀分布 (搜索信息量为 0, 这正是 BG_TRAIN_SIMS=20 那次
          结构性退化的同一件事)。要有深挖余量, 访问分布才携带噪声的影响。
    */
    {
        auto visitHash = [](PPOMCTSAgent &a) {
            unsigned long long h = 1469598103934665603ULL;   /* FNV-1a 基 */
            const int root = a.currentRoot();
            if (root < 0) { return h; }
            for (std::size_t i = 0; i < a.nodes[(std::size_t)root].childIDs.size(); i++) {
                const PPOMCTSAgent::AZNode &c =
                    a.nodes[(std::size_t)a.nodes[(std::size_t)root].childIDs[i]];
                /* 动作下标 + 访问数 一起进哈希: 既记录"试过哪些着法", 也记录"试了几次" */
                h ^= (unsigned long long)(c.parentAction + 1) * 1099511628211ULL;
                h ^= (unsigned long long)c.visitCount * 2654435761ULL;
                h *= 1099511628211ULL;
            }
            return h;
        };

        const int abSims = 200;      /* 必须 > 44 个合法着法 */

        Chess b1;
        b1.reset();
        PPOMCTSAgent a1(b1, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, false);
        b1.sideToMove = Stone::COLOR_RED;
        a1.evalRootNoise = false;
        a1.resetSearchTree();
        RL::Random::setSeed(4242u);
        a1.selectMove(Stone::COLOR_RED, abSims, 0.0f);
        const unsigned long long h1 = visitHash(a1);

        Chess b2;
        b2.reset();
        PPOMCTSAgent a2(b2, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, false);
        b2.sideToMove = Stone::COLOR_RED;
        a2.evalRootNoise = true;
        a2.rootNoise = true;
        a2.rootNoiseEps = 0.6f;          /* 加大噪声, 让差异明显 */
        a2.resetSearchTree();
        RL::Random::setSeed(4242u);      /* 同一个种子: 差异只可能来自噪声 */
        a2.selectMove(Stone::COLOR_RED, abSims, 0.0f);
        const unsigned long long h2 = visitHash(a2);

        CHECK(h1 != 0ULL && h2 != 0ULL, "两次搜索都产生了访问数");
        CHECK(h1 != h2, "开根噪声后根访问分布应当改变 (开关真的接上了)");

        /*
           反向对照: 必须用**同一个 agent** (同一份权重) 跑两次。
           踩过的坑: 构造 PPOMCTSAgent 本身会消耗全局随机流 (权重初始化), 所以
           "两个 agent + 同一个种子"并不等价 —— 它们的**权重不同**, 搜索自然不同。
           bench_ppo_sims 的头注释里记着同一个教训 ("不给 --load 时靠两次播种构造保证")。
        */
        Chess b3;
        b3.reset();
        PPOMCTSAgent a3(b3, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, false);
        b3.sideToMove = Stone::COLOR_RED;
        a3.evalRootNoise = false;
        a3.resetSearchTree();
        RL::Random::setSeed(4242u);
        a3.selectMove(Stone::COLOR_RED, abSims, 0.0f);
        const unsigned long long hA = visitHash(a3);

        a3.resetSearchTree();
        RL::Random::setSeed(4242u);
        a3.selectMove(Stone::COLOR_RED, abSims, 0.0f);
        const unsigned long long hB = visitHash(a3);
        CHECK(hA == hB, "同一份权重 + 同种子 + 关噪声 -> 访问分布逐位可复现");
    }
}

/* ---------------------------------------------------------------------------
 *  自动一步杀题的小工具 (与 bench_diag 同一套做法)
 *
 *  为什么要"自动生成 + 自校验"而不是手摆局面: 本工程没有 FEN 接口, 手摆一个将杀
 *  局面很容易摆错 (摆错了测试就变成空转, 而且不会报错)。这里随机走几步得到一个
 *  局面, 再**扫描所有合法着法**判定它是不是一步杀 —— 解的合法性由规则引擎
 *  (getResult) 自己保证。
 * ------------------------------------------------------------------------- */

static bool sameMoveXY(const Step &a, const Step &b)
{
    return a.pos.x == b.pos.x && a.pos.y == b.pos.y
           && a.nextPos.x == b.nextPos.x && a.nextPos.y == b.nextPos.y;
}

static void diagRandomOpening(Chess &c, int plies, std::mt19937 &rng)
{
    for (int i = 0; i < plies; i++) {
        std::vector<Step *> steps;
        c.sample(c.sideToMove, steps);
        if (steps.empty()) {
            Steps::instance().put(steps);
            return;
        }
        std::uniform_int_distribution<int> pick(0, (int)steps.size() - 1);
        const Step s = *steps[pick(rng)];
        Steps::instance().put(steps);
        double dummy = 0.0;
        c.moveForward(&s, dummy);
    }
}

/* 收集所有"落子后立刻判己方胜"的着法 (一步杀/一步困毙) */
static void findMateInOne(Chess &c, int mover, std::vector<Step> &out)
{
    out.clear();
    std::vector<Step *> steps;
    c.sample(mover, steps);
    const int want = (mover == Stone::COLOR_RED) ? Chess::RESULT_RED_WIN
                                                 : Chess::RESULT_BLACK_WIN;
    for (std::size_t i = 0; i < steps.size(); i++) {
        const Step s = *steps[i];
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        const bool mate = (c.getResult(c.sideToMove) == want);
        c.moveBack(&s, dummy);
        if (mate) {
            out.push_back(s);
        }
    }
    Steps::instance().put(steps);
}

/*
 *  [7] 搜索能不能"看见"一步杀 —— evaluateLeaf 终局处理的回归测试
 *
 *  这是诊断仪表盘直接抓出来的缺陷: 修之前三个搜索入口都无条件用 critic 估叶子,
 *  于是"一步杀"那步落子后的叶子被交给一个只会**评估**的网络去猜 (它甚至不知道局面
 *  已经终局), 搜索完全看不见战术终点 —— bench_diag 实测命中率 **0/20 = 0%**,
 *  而非终局的"白吃子"却有 16.7%。这个 0% vs 16.7% 的对比就是证据。
 *
 *  修好之后这条测试**用随机初始化的网络也能稳定通过**: 终局叶子的价值是确定的 ±1,
 *  与网络质量无关 —— 这正是这个修复的价值所在。
 */
static void secMateInOne()
{
    std::printf("\n[7] 搜索能不能看见一步杀 (evaluateLeaf 的终局处理)\n");
    using namespace RL::Diag;

    std::mt19937 rng(20240901u);
    Chess board;
    board.reset();
    PPOMCTSAgent ag(board, 16, 0.99f, 0.001f, 1.414f, 16, 0.1f, false);

    const int kWant = 8;
    int tried = 0, hit = 0;
    for (int attempt = 0; attempt < 12000 && tried < kWant; attempt++) {
        Chess pos;
        pos.reset();
        diagRandomOpening(pos, 4 + (int)(rng() % 9u), rng);
        const int mover = pos.sideToMove;

        std::vector<Step> sol;
        findMateInOne(pos, mover, sol);
        if (sol.empty()) {
            continue;
        }
        tried++;

        /*
           必须把题目局面拷回 agent **绑定的那个棋盘**再搜 —— PPOMCTSAgent 构造时就把
           Chess& 绑死了, 另建一个 Chess 再调 selectMove 会去搜 board 上的别的局面。
           (bench_diag 第一版在这里踩过坑, 结果是安静地报 0%。)
        */
        board = pos;
        board.sideToMove = mover;
        ag.resetSearchTree();
        const Step chosen = ag.selectMove(mover, 80, 0.0f);
        for (std::size_t i = 0; i < sol.size(); i++) {
            if (sameMoveXY(chosen, sol[i])) { hit++; break; }
        }
    }

    std::printf("    一步杀: %d/%d 被搜索选中\n", hit, tried);
    CHECK(tried >= 4, "至少生成出 4 道一步杀题 (否则这条测试是空转)");
    const double rate = (tried > 0) ? (double)hit / (double)tried : 0.0;
    CHECK(rate >= 0.5, "终局叶子给真实 ±1 之后, 大部分一步杀必须被搜索找到");
}

/*
 *  [8] 行为聚合里的两个二阶量 (std(V) / corr(V_before, V_after))
 *
 *  为什么值得单测: 仪表盘实测出现过 mean(V_after) ≈ -mean(V_before), 于是
 *  "每手 V 增益 ≈ -2·mean(V)" 看起来像"agent 每手都在把自己走坏"。真相是 **V 基本
 *  是常数** (谁走都 ≈ +0.15), 那个"增益"读的是常数偏置。没有 std/corr 这两个量,
 *  这个数字一定会被误读 —— 所以它们的语义要钉死。
 */
static void secBehaviorAgg()
{
    std::printf("\n[8] std(V) / corr(V_before, V_after)\n");
    using namespace RL::Diag;

    /* (a) V 是常数 (+0.5): std = 0, corr 无定义 -> 0, "增益" = -1 */
    {
        Aggregates a;
        for (int i = 0; i < 4; i++) {
            MoveBehavior b;
            b.vBefore = 0.5;
            b.vAfter = -0.5;
            a.addBehavior(b);
        }
        checkNear("常数 V: std = 0", a.stdVBefore(), 0.0, 1e-12);
        checkNear("常数 V: corr = 0 (无定义)", a.corrVBeforeAfter(), 0.0, 1e-12);
        checkNear("常数 V: 每手增益 = -1 (误导性, 需看 std)", a.meanValueGain(), -1.0, 1e-12);
    }

    /* (b) V 与局面同步: corr = +1, std 手算 0.1118034 */
    {
        Aggregates a;
        const double vb[4] = { 0.1, 0.2, 0.3, 0.4 };
        for (int i = 0; i < 4; i++) {
            MoveBehavior b;
            b.vBefore = vb[i];
            b.vAfter = vb[i];
            a.addBehavior(b);
        }
        checkNear("同步 V: std", a.stdVBefore(), 0.1118034, 1e-6);
        checkNear("同步 V: corr = +1", a.corrVBeforeAfter(), 1.0, 1e-9);
        checkNear("同步 V: 每手增益 = 0", a.meanValueGain(), 0.0, 1e-12);
    }

    /* (c) V 与局面反号: corr = -1 (这才是"值得查 PBRS/符号"的形状) */
    {
        Aggregates a;
        const double vb[4] = { 0.1, 0.2, 0.3, 0.4 };
        for (int i = 0; i < 4; i++) {
            MoveBehavior b;
            b.vBefore = vb[i];
            b.vAfter = -vb[i];
            a.addBehavior(b);
        }
        checkNear("反号 V: corr = -1", a.corrVBeforeAfter(), -1.0, 1e-9);
        CHECK(a.stdVBefore() > 0.1, "反号 V 时 std 不为 0 (与常数情形可区分)");
    }
}

int main()
{
    std::printf("========================================\n");
    std::printf("  诊断指标 (rl/diag.h) 解析验证\n");
    std::printf("========================================\n");

    secEntropy();
    secValue();
    secMoeAndCalib();
    secQForParent();
    secCsvShape();
    secBehaviorAgg();
    secAgentWiring();
    secMateInOne();

    std::printf("\n========================================\n");
    std::printf("  断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    std::printf("========================================\n");
    return (g_failed == 0) ? 0 : 1;
}
