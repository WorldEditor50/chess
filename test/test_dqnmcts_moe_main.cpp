/*
 * ============================================================
 *  test_dqnmcts_moe - DQN+MCTS (稀疏 MoE + TB 专家) 的断言测试
 * ============================================================
 *
 *  被测对象: src/dqnmctsmoetbagent.{h,cpp} (界面类型 AGENT_DQNMCTS_MOE)。
 *
 *  为什么这些断言值得写 (每一条都对着一个**已知会静默失效**的地方):
 *
 *   [1] 骨干指纹: TB 专家的头数口径。`MOE_TB_HEADS = 15` 在 d_model = 1263 上
 *       会被老的"头数必须整除 d_model"规则降成 **3** 个头 (d_k 421), 一个专家前向
 *       慢 3 倍, 而参数指纹 / paramCount / 权重格式**一个都不变** —— SAC 那条线上
 *       实测过 (19.06 ms vs 6.20 ms, 80% 的 head 张量是死的)。所以这里断言
 *       "请求 15 个 / 实际用 15 个 / d_k=84 / 注意力元素 105840" 四个读数。
 *   [2] 表示: 规范视角 (红黑共用一张网的前提) + 三个规则上下文通道。
 *       旧类 (DQNMCTSAgent) 的 90 维编码既没有走子方通道, 子力值还按 Stone::Type
 *       的**序数**取 (车 1/7 < 兵 4/7 —— 顺序是反的)。
 *   [3] 稀疏叶子估值 == 全量前向 (逐元素, 1e-6)。这条是"骨干只前向一次"能不能
 *       成立的全部依据; 不等价的话搜索与训练学的就是两个值。
 *   [4] PUCT 的**负号** (negamax)。漏掉它 = 搜索专挑对自己最差的着法, 而损失曲线
 *       一切正常 (旧类那一支的 exploitation 项就是正号)。
 *   [5] 学习口径: 一次 learnBatch 只前进 1 步 (P3/P4)、epochs 真的放大批、目标被
 *       clampTarget 夹住、**没被走到的动作列梯度为 0**。
 *   [6] 权重往返: 三个文件都写出来, 载回来之后前向逐元素一致。
 *   [7] 自检报告: 只读、可重复、含"不是棋力"那句 (test_match [2.11] 会检查它)。
 *   [8] 外部终局通道 (人机对弈里"人把 AI 将死"那一手): 只回填**最新那条真实决策
 *       样本**, 幂等, 而且不许挂到探索期的推演样本上。
 *   [9] learnFromSearch 关掉之后 selectMove **不产生任何更新** (对弈模式"权重也不变"
 *       那条承诺靠它成立)。
 *  [10] 搜索输出的合法性 (不能返回无效 Step 去让调用方判负)。
 *  [11] 耗时读数 (不是断言): 给界面常数与后续对照留一份"这台机器上多快"的记录。
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "rl/cpuinfo.hpp"
#include "dqnmctsmoetbagent.h"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                            \
        ++g_checks;                                                      \
        if (!(cond)) {                                                    \
            ++g_failed;                                                   \
            std::printf("  [FAIL] %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
        }                                                                \
    } while (0)

#define CHECK_EQ(a, b, msg) do {                                         \
        ++g_checks;                                                      \
        long long _a = (long long)(a);                                    \
        long long _b = (long long)(b);                                    \
        if (_a != _b) {                                                  \
            ++g_failed;                                                   \
            std::printf("  [FAIL] %s: %lld != %lld  (%s:%d)\n",            \
                        (msg), _a, _b, __FILE__, __LINE__);               \
        }                                                                \
    } while (0)

static double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}

/* 棋盘 hash + 几个"自检必须不动它"的状态量 (只读契约的判据) */
static unsigned long long boardFingerprint(Chess &c)
{
    unsigned long long h = c.computeHash();
    h = h * 1000003ULL + (unsigned long long)(c.sideToMove + 1);
    h = h * 1000003ULL + (unsigned long long)c.halfMoveClock;
    h = h * 1000003ULL + (unsigned long long)c.history.size();
    return h;
}

/*
 *  把棋盘"左右镜像 + 红黑互换" (只用于 [2] 的规范视角检验)。
 *  变换之后仍是一个合法局面, 而它正是"从对手的眼里看同一盘棋"。
 */
static void mirrorAndSwapColors(Chess &c)
{
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[(std::size_t)i];
        s->pos.x = 9 - s->pos.x;
        s->color = (s->color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    c.m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[(std::size_t)i];
        if (s != nullptr && s->alive) {
            c.m_map[s->pos] = s;
        }
    }
}

static double maxAbsDiff(const RL::Tensor &a, const RL::Tensor &b)
{
    const std::size_t n = (a.size() < b.size()) ? a.size() : b.size();
    double m = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    }
    return m;
}

int main()
{
    std::printf("============================================================\n");
    std::printf(" test_dqnmcts_moe - DQN+MCTS (稀疏 MoE + TB 专家)\n");
    std::printf("============================================================\n");

    /* 固定基种子: 本测试里"随机"的部分 (批抽样 / 温度采样) 必须可复现 */
    RL::Random::setSeed(20240901u);

    Chess chess;
    chess.reset();

    const double tBuild0 = nowMs();
    DQNMCTSMOETbAgent ag(chess, 64, 0.99f, 0.001f, 1.0f, 1.5f);
    const double tBuild = nowMs() - tBuild0;

    /* ============================================================
     *  [1] 构造 / 骨干指纹
     * ============================================================ */
    std::printf("\n[1] 骨干指纹 (稀疏 MoE + TB 专家)\n");
    CHECK_EQ(DQNMCTSMOETbAgent::CELLS, 90, "CELLS = 10x9");
    CHECK_EQ(DQNMCTSMOETbAgent::CTX_COUNT, 3, "规则上下文 3 个");
    CHECK_EQ(DQNMCTSMOETbAgent::CTX_BASE, 1260, "上下文标量的起点 = 14 平面 x 90");
    CHECK_EQ(DQNMCTSMOETbAgent::STATE_DIM, 1263, "STATE_DIM = 1260 + 3");
    CHECK_EQ(DQNMCTSMOETbAgent::ACTION_DIM, 128, "动作仍是 128 槽哈希 (SAC 默认口径)");
    CHECK_EQ(ag.moeExpertCount(), 4, "稀疏 MoE 有 4 个专家");
    CHECK_EQ(ag.moeTopK(), 1, "top-1 路由");
    CHECK(!ag.moeDenseNow(), "不是稠密对照 (TopK != E)");
    CHECK(strstr(ag.backboneName(), "\u7a00\u758f") != nullptr, "骨干名写着稀疏");
    CHECK_EQ(ag.tbHeadsRequested(), 15, "请求 15 个注意力头");
    CHECK_EQ(ag.tbHeadsUsed(), 15,
             "**实际用了 15 个** (HonorHeads 修复; 老规则会静默降成 3)");
    CHECK_EQ(ag.tbHeadDim(), 84, "d_k = 1263/15 = 84");
    CHECK_EQ(ag.tbHeadsAllocated(), 15, "分配出来的 head 对象 = 15 (没有死张量)");
    CHECK_EQ(ag.tbAttnElements(), 105840LL, "注意力元素 = 15 x 84 x 84");
    CHECK(strstr(ag.hiddenActivationName(), "tanh") != nullptr,
          "隐层激活是 Layer<Tanh> (不是 TanhNorm<Sigmoid>)");
    CHECK(ag.trunkParamCount() > 20000000LL, "骨干参数量在千万级");
    CHECK(ag.headParamCount() > 0, "输出头有参数");
    CHECK_EQ(ag.uniqueParamCount(), ag.trunkParamCount() + ag.headParamCount(),
             "唯一参数量 = 骨干 + 头 (视图不重复计)");
    std::printf("  骨干 %lld + 头 %lld = %lld 参数 | 建网 %.0f ms\n",
                ag.trunkParamCount(), ag.headParamCount(), ag.uniqueParamCount(), tBuild);

    /* ============================================================
     *  [2] 表示: 规范视角 + 规则上下文
     * ============================================================ */
    std::printf("\n[2] 表示 (规范视角 14 平面 + 3 上下文)\n");
    {
        RL::Tensor red(DQNMCTSMOETbAgent::STATE_DIM, 1);
        RL::Tensor blk(DQNMCTSMOETbAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, red);
        ag.encodeStateFor(Stone::COLOR_BLACK, blk);
        int nzRed = 0;
        for (int i = 0; i < DQNMCTSMOETbAgent::CTX_BASE; i++) {
            if (red[i] != 0.0f) {
                nzRed++;
            }
        }
        CHECK_EQ(nzRed, 32, "开局: 棋子区恰好 32 个非零格");
        CHECK(maxAbsDiff(red, blk) < 1e-12,
              "开局左右对称 ⇒ 红黑两个视角的编码逐元素相同");
        CHECK(std::fabs((double)red[DQNMCTSMOETbAgent::CTX_BASE + 0]) < 1e-6,
              "上下文[0] 无吃子进度 = 0");
        CHECK(std::fabs((double)red[DQNMCTSMOETbAgent::CTX_BASE + 1]) < 1e-6,
              "上下文[1] 重复次数 = 0");
        CHECK(std::fabs((double)red[DQNMCTSMOETbAgent::CTX_BASE + 2]) < 1e-6,
              "上下文[2] 被将 = 0");

        /* 走一手红炮: "无吃子进度"这个通道必须跟着棋盘变 */
        std::vector<Step*> mv;
        chess.sample(Stone::COLOR_RED, mv);
        const Step first = *mv[0];
        Steps::instance().put(mv);
        double dummy = 0.0;
        chess.moveForward(&first, dummy);
        ag.encodeStateFor(Stone::COLOR_BLACK, blk);
        CHECK(blk[DQNMCTSMOETbAgent::CTX_BASE + 0] > 0.0f,
              "走了一手之后无吃子进度 > 0 (这个通道真的连在棋盘上)");
        chess.moveBack(&first, dummy);

        /*
           规范视角的**关键性质**: 把棋盘镜像 + 红黑互换之后, "从黑方眼里看"的编码
           必须与原来"从红方眼里看"的编码逐元素相同。这一条等价于"红黑共用一张网",
           也是 negamax 符号自洽的前提。镜像写错是最难查的静默缺陷 (红黑变成两个
           不同的函数, 训练照样跑)。
        */
        RL::Tensor a1(DQNMCTSMOETbAgent::STATE_DIM, 1);
        RL::Tensor a2(DQNMCTSMOETbAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, a1);
        mirrorAndSwapColors(chess);
        ag.encodeStateFor(Stone::COLOR_BLACK, a2);
        mirrorAndSwapColors(chess);   /* 变回来 (镜像是对合变换) */
        CHECK(maxAbsDiff(a1, a2) < 1e-12,
              "镜像+换色之后, 黑方视角的编码 == 原局面红方视角的编码");
        std::printf("  32 个非零格 | 红黑视角编码最大差 %.3g\n", maxAbsDiff(a1, a2));
    }

    /* ============================================================
     *  [3] 稀疏叶子估值 == 全量前向
     * ============================================================ */
    std::printf("\n[3] 稀疏列叶子估值与全量前向逐元素一致\n");
    {
        RL::Tensor state(DQNMCTSMOETbAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, state);

        std::vector<Step*> legal;
        std::vector<int> idx;
        RL::Tensor mask(DQNMCTSMOETbAgent::ACTION_DIM, 1);
        ag.getLegalActions(Stone::COLOR_RED, legal, idx, mask);
        const int nLegal = (int)idx.size();
        CHECK(nLegal > 20, "开局合法着法数 > 20");
        Steps::instance().put(legal);

        RL::Tensor qFull(DQNMCTSMOETbAgent::ACTION_DIM, 1);
        CHECK(ag.qValuesFull(state, qFull, nullptr), "全量前向成功");
        std::vector<float> qSp;
        CHECK(ag.sparseQOnline(state, idx, qSp, nullptr), "稀疏前向成功");
        CHECK_EQ((long long)qSp.size(), (long long)idx.size(), "稀疏输出长度 = 合法列数");
        double maxd = 0.0;
        for (std::size_t i = 0; i < idx.size(); i++) {
            maxd = std::max(maxd, std::fabs((double)qFull[idx[i]] - (double)qSp[i]));
        }
        CHECK(maxd < 1e-6, "稀疏列 == 全量同列 (容差 1e-6)");

        std::vector<float> q2Sp;
        CHECK(ag.sparseQTarget(state, idx, q2Sp, nullptr), "目标网稀疏前向成功");
        CHECK_EQ((long long)q2Sp.size(), (long long)idx.size(), "目标网稀疏输出长度对");
        std::printf("  合法列 %d 个 | max|dQ| = %.3g\n", nLegal, maxd);
    }

    /* ============================================================
     *  [4] PUCT 的负号 (negamax)
     * ============================================================ */
    std::printf("\n[4] PUCT 符号 (父节点比较子节点时必须取负)\n");
    {
        ag.nodes.clear();
        DQNMCTSMOETbAgent::AZNode root;
        root.currentColor = Stone::COLOR_RED;
        ag.nodes.push_back(root);                       /* 0 = 父 */
        DQNMCTSMOETbAgent::AZNode good(0, 1, Step(), 0.25, Stone::COLOR_BLACK, 30);
        DQNMCTSMOETbAgent::AZNode bad(0, 2, Step(), 0.25, Stone::COLOR_BLACK, 30);
        DQNMCTSMOETbAgent::AZNode fresh(0, 3, Step(), 0.25, Stone::COLOR_BLACK, 30);
        /*
           子节点存的是**子节点走棋方 (= 对手)** 视角的价值: good 对对手有利 (+1)
           ⇒ 对父节点来说应该是 -1; bad 相反。漏掉负号的表现就是"父节点专挑对
           对手有利的着法", 而且损失曲线完全正常。
        */
        good.visitCount = 10;
        good.totalValue = 10.0;
        bad.visitCount = 10;
        bad.totalValue = -10.0;
        ag.nodes.push_back(good);                       /* 1 */
        ag.nodes.push_back(bad);                        /* 2 */
        ag.nodes.push_back(fresh);                      /* 3 (未访问) */
        const double pGood = ag.getPUCT(1, 20);
        const double pBad = ag.getPUCT(2, 20);
        const double pFresh = ag.getPUCT(3, 20);
        CHECK(pGood < pBad, "对手价值高的子节点在父节点眼里更**差** (负号生效)");
        CHECK(pFresh > pBad, "未访问的孩子优先");
        std::printf("  PUCT(对手好)=%.3f < PUCT(对手差)=%.3f < PUCT(未访问)=%.3g\n",
                    pGood, pBad, pFresh);
        ag.nodes.clear();
    }

    /* ============================================================
     *  [5] 学习口径
     * ============================================================ */
    std::printf("\n[5] 学习 (P3 梯度累积 / P4 多 epoch / 夹目标 / 合法列梯度)\n");
    ag.batchSize = 8;              /* 小批: 这个测试要快, 不是要准 */
    ag.learnEveryMoves = 100000;   /* 关掉节拍, 让下面的 learnBatch 调用可数 */
    ag.trainSelfPlay(1, 2, 10, false);
    const int poolAfterPlay = ag.samplePoolSize();
    CHECK(poolAfterPlay >= 10, "自对弈把真实决策样本记进了池子");
    {
        int decisions = 0;
        for (std::size_t i = 0; i < ag.memories.size(); i++) {
            if (ag.memories[i].decision) {
                decisions++;
            }
        }
        CHECK_EQ(decisions, poolAfterPlay, "自对弈记的每一条都是 decision 样本");
    }

    const int steps0 = ag.getLearnSteps();
    const float loss1 = ag.learnBatch(8, 1);
    CHECK(ag.getLearnSteps() == steps0 + 1, "一次 learnBatch 只让 learnSteps 前进 1");
    CHECK_EQ(ag.m_lastBatchSamples, 8, "epochs=1 ⇒ 本批样本数 = batchSize");
    CHECK(ag.m_maxAbsTarget <= (double)ag.clampTarget + 1e-6,
          "clampTarget 真的夹住了目标 (|y| <= 2)");
    ag.learnBatch(8, 3);
    CHECK_EQ(ag.m_lastBatchSamples, 24, "epochs=3 ⇒ 本批样本数 = batchSize x 3");
    CHECK(ag.getLearnSteps() == steps0 + 2, "多个 epoch 也只算一次优化器调用");
    std::printf("  池 %d 条 | loss %.6g | |y|max %.4f | 夹住 %lld/%lld\n",
                ag.samplePoolSize(), (double)loss1, ag.m_maxAbsTarget,
                ag.batchDiag.clamped, ag.batchDiag.n);

    {
        /*
           没被走到的动作列**梯度为 0** —— 这是"只回归实际那一步"的直接后果, 也是 SAC
           那边用来证明"非法列没有被抬起来"的同一条配对断言。

           ⚠ 判据必须看**输出头的权重行**, 不是 Q 值: 骨干每一批都在变, 所以 128 列 Q
           值**全部**会变 (第一版就是这么写错的 —— 它把"骨干变了"误读成"梯度泄漏了")。
           头层的权重梯度是 `e·x^T`, 而 e 只有实际走过的那一列非零 ⇒ 只有那些**行**会动,
           变化行数必须 <= 批内样本数 (样本是**有放回**抽的, 所以只会更少)。
        */
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer*>(ag.qHead[0]);
        CHECK(fc != nullptr, "Q 头是 iFcLayer (能读到权重)");
        const int outDim = fc ? fc->outputDim : 0;
        const int inDim = fc ? fc->inputDim : 0;
        CHECK_EQ(outDim, DQNMCTSMOETbAgent::ACTION_DIM, "头输出维 = 动作维");
        RL::Tensor w0;
        if (fc != nullptr) {
            w0 = fc->w;      /* 深拷贝 (Tensor 是值语义) */
        }
        const int bs = 8;
        ag.learnBatch(bs, 1);
        int changedRows = 0;
        if (fc != nullptr) {
            for (int a = 0; a < outDim; a++) {
                bool changed = false;
                for (int k = 0; k < inDim; k++) {
                    const std::size_t at = (std::size_t)a * (std::size_t)inDim
                                           + (std::size_t)k;
                    if (at < w0.size() && std::fabs((double)fc->w[at] - (double)w0[at]) > 0.0) {
                        changed = true;
                        break;
                    }
                }
                if (changed) {
                    changedRows++;
                }
            }
        }
        CHECK(changedRows >= 1, "一批之后输出头真的动了");
        CHECK(changedRows <= bs, "变化的头权重行数 <= 批内样本数 (没被走到的列梯度为 0)");
        std::printf("  一批之后变化的头权重行: %d (<= %d)\n", changedRows, bs);
    }

    /* ============================================================
     *  [6] 权重往返
     * ============================================================ */
    std::printf("\n[6] 权重往返 (三个文件 + 载回来前向一致)\n");
    {
        const std::string prefix = "weights/_test_dqnmcts_moe_tmp";
        CHECK(ag.saveModel(prefix), "saveModel 成功 (三个文件都写出来了)");
        RL::Tensor st(DQNMCTSMOETbAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, st);
        RL::Tensor before(DQNMCTSMOETbAgent::ACTION_DIM, 1);
        ag.qValuesFull(st, before, nullptr);

        ag.learnBatch(8, 2);      /* 改动权重 */

        CHECK(ag.loadModel(prefix), "loadModel 成功");
        RL::Tensor after(DQNMCTSMOETbAgent::ACTION_DIM, 1);
        ag.qValuesFull(st, after, nullptr);
        CHECK(maxAbsDiff(before, after) < 1e-9,
              "载回权重之后前向逐元素一致 (训练的改动被撤销)");

        CHECK(!ag.loadModel("weights/_no_such_prefix_dqnmcts_moe"),
              "缺文件时 loadModel 返回 false (不是静默继续)");

        std::remove((prefix + "_trunk").c_str());
        std::remove((prefix + "_q").c_str());
        std::remove((prefix + "_q2").c_str());
    }

    /* ============================================================
     *  [7] 自检报告: 只读 + 可重复
     * ============================================================ */
    std::printf("\n[7] 自检报告 (只读 / 可重复 / 含'不是棋力')\n");
    {
        const unsigned long long fp0 = boardFingerprint(chess);
        const std::string r1 = ag.selfCheckReport();
        const std::string r2 = ag.selfCheckReport();
        const unsigned long long fp1 = boardFingerprint(chess);
        CHECK(!r1.empty(), "报告非空");
        CHECK_EQ(fp0, fp1, "报告没有改动棋盘 (hash/sideToMove/halfMoveClock/history)");
        CHECK(r1 == r2, "连续两次报告逐字节相同 (没有藏在里面的计数器)");
        CHECK(r1.find("\u4e0d\u662f\u68cb\u529b") != std::string::npos,
              "报告含'不是棋力'那句 (test_match [2.11] 会检查)");
        CHECK(r1.find("\u6ce8\u610f\u529b\u5934") != std::string::npos,
              "报告里有注意力头读数");
        CHECK(r1.find("PUCT") != std::string::npos, "报告里有搜索口径");
        CHECK(r1.find("MoE") != std::string::npos, "报告里有 MoE 路由直方图");
        CHECK(r1.find("\u7ec8\u5c40\u901a\u9053") != std::string::npos,
              "报告里有终局通道读数");
    }

    /* ============================================================
     *  [8] 外部终局通道
     * ============================================================ */
    std::printf("\n[8] 外部终局通道 (人机里'人把 AI 将死'那一手)\n");
    {
        /* 先记一条真实决策样本 (selectMove 里 learnFromSearch 记的), 再推几条
           **推演**样本 (exploreAndTrain 记的) —— 回填必须命中前者。 */
        ag.learnFromSearch = true;
        const std::size_t poolBefore = ag.memories.size();
        ag.selectMove(Stone::COLOR_BLACK, 2, 0.0f);
        CHECK_EQ((long long)ag.memories.size(), (long long)poolBefore + 1,
                 "selectMove 记了一条决策样本");
        CHECK(ag.memories.back().decision, "最新一条是 decision 样本");
        /*
           ⚠ 这条断言钉的是一个**真实抓到的静默缺陷**: 第一版 `learnFromSearchStep`
           从"走子前"缓存里取局面, 而那份缓存只在 `m_trainingMode`(训练回路) 打开时
           才填 —— 界面决策路径从来不开它, 于是**界面每走一手都会拿一份空局面**
           (cells 为空 = 全零状态) 去训练, 不报任何错。
           判据: 一条真实决策样本必须带**棋子格** (开局 32 个), 不能是空的。
        */
        CHECK(ag.memories.back().cells.size() >= 20,
              "决策样本带着真实局面的棋子格 (不是空局面 —— 空局面是静默缺陷)");
        CHECK(ag.memories.back().curMask[0] != 0 || ag.memories.back().curMask[1] != 0,
              "决策样本带着当前局面的合法掩码");

        ag.exploreAndTrain(Stone::COLOR_BLACK, 3);
        CHECK(ag.memories.size() > poolBefore + 1, "探索又推进了几条推演样本");
        CHECK(!ag.memories.back().decision, "池尾现在是一条**推演**样本");

        /* 找到池里最新那条 decision 样本 (回填的目标) */
        long long backIdx = -1;
        for (long long i = (long long)ag.memories.size() - 1; i >= 0; i--) {
            if (ag.memories[(std::size_t)i].decision) {
                backIdx = i;
                break;
            }
        }
        CHECK(backIdx >= 1, "池里能找到 decision 样本 (而且它后面还有推演样本)");
        const bool doneBefore = ag.memories[(std::size_t)backIdx].done;
        const bool olderDoneBefore =
            (backIdx > 0) ? ag.memories[(std::size_t)(backIdx - 1)].done : false;

        /* 人执红把 AI (黑) 将死: 从**黑方**视角 = 输 = -1 */
        const long long ext0 = ag.externalTerminals;
        CHECK(ag.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
              "notifyGameResult 接住了 (有可挂的决策样本)");
        if (!doneBefore) {
            CHECK(ag.memories[(std::size_t)backIdx].done, "那条决策样本被写成 done");
            CHECK(std::fabs((double)ag.memories[(std::size_t)backIdx].reward - (-1.0f)) < 1e-6,
                  "终局奖励 = -1 (黑方输)");
            CHECK_EQ(ag.externalTerminals, ext0 + 1, "外部终局计数 +1");
        } else {
            /* 那一手自己就把棋下完了 (少见但合法): 终局本来就写好了 */
            CHECK(ag.memories[(std::size_t)backIdx].done, "样本本来就是 done");
            CHECK_EQ(ag.externalTerminals, ext0, "已经写过就不重复计数");
        }

        /* 幂等: 再通知一次不许把更老的样本也改掉, 也不许重复计数 */
        const long long ext1 = ag.externalTerminals;
        CHECK(ag.notifyGameResult(Chess::RESULT_RED_WIN, Stone::COLOR_BLACK),
              "重复通知返回 true (幂等)");
        CHECK_EQ(ag.externalTerminals, ext1, "重复通知不重复计数");
        CHECK_EQ(ag.memories[(std::size_t)(backIdx - 1)].done, olderDoneBefore,
                 "更老的样本没有被第二次通知改掉");
        CHECK(!ag.notifyGameResult(Chess::RESULT_ONGOING, Stone::COLOR_BLACK),
              "ONGOING 不被当成终局");
        std::printf("  挂载位置: 池尾往回第 %lld 条 (池 %d 条) | 外部终局 %lld 次\n",
                    (long long)ag.memories.size() - 1 - backIdx,
                    (int)ag.memories.size(), ag.externalTerminals);
    }

    /* ============================================================
     *  [9] learnFromSearch 关掉 ⇒ 不产生任何更新
     * ============================================================ */
    std::printf("\n[9] learnFromSearch 关掉之后 selectMove 不更新\n");
    {
        ag.learnFromSearch = false;
        const int s0 = ag.getLearnSteps();
        const std::size_t p0 = ag.memories.size();
        const Step mv = ag.selectMove(Stone::COLOR_RED, 2, 0.0f);
        CHECK(mv.valid, "仍然返回合法着法");
        CHECK_EQ(ag.getLearnSteps(), s0, "没有更新");
        CHECK_EQ((long long)ag.memories.size(), (long long)p0, "没有记样本");

        ag.learnFromSearch = true;
        ag.selectMove(Stone::COLOR_RED, 2, 0.0f);
        CHECK_EQ((long long)ag.memories.size(), (long long)p0 + 1, "打开之后记一条");
    }

    /* ============================================================
     *  [10] 搜索输出的合法性
     * ============================================================ */
    std::printf("\n[10] 搜索输出的合法性\n");
    {
        std::vector<Step*> legal;
        chess.sample(chess.sideToMove, legal);
        const int nLegal = (int)legal.size();
        Steps::instance().put(legal);
        CHECK(nLegal > 0, "局面本来就有合法着法");

        ag.learnFromSearch = false;   /* 这一节只测搜索 */
        const Step s1 = ag.selectMove(chess.sideToMove, 1, 0.0f);
        CHECK(s1.valid, "1 次模拟也返回合法着法 (不能返回空 Step 让调用方判负)");
        const Step s40 = ag.selectMove(chess.sideToMove, 40, 0.0f);
        CHECK(s40.valid, "40 次模拟返回合法着法");

        /* 树里必须真的有节点, 而且根的孩子被访问过 (搜索不是空转) */
        CHECK(ag.nodes.size() > 1, "搜索建了树");
        int visitedChildren = 0;
        for (std::size_t i = 0; i < ag.nodes[0].childIDs.size(); i++) {
            if (ag.nodes[(std::size_t)ag.nodes[0].childIDs[i]].visitCount > 0) {
                visitedChildren++;
            }
        }
        CHECK(visitedChildren > 0, "根的孩子被访问过");
        CHECK(ag.m_leafEvals > 0, "叶子估值次数 > 0");
        std::printf("  树节点 %d | 根孩子 %d (访问过 %d) | 叶估值 %lld (全量回退 %lld)\n",
                    (int)ag.nodes.size(), (int)ag.nodes[0].childIDs.size(),
                    visitedChildren, ag.m_leafEvals, ag.m_fullLeafEvals);
    }

    /* ============================================================
     *  [11] 耗时读数 (不是断言)
     * ============================================================ */
    std::printf("\n[11] 耗时读数 (本机, 仅供界面常数参考)\n");
    {
        RL::Tensor st(DQNMCTSMOETbAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, st);
        const int reps = 5;
        double t0 = nowMs();
        for (int i = 0; i < reps; i++) {
            RL::Tensor q(DQNMCTSMOETbAgent::ACTION_DIM, 1);
            ag.qValuesFull(st, q, nullptr);
        }
        const double perFull = (nowMs() - t0) / reps;

        std::vector<Step*> legal;
        std::vector<int> idx;
        RL::Tensor mask(DQNMCTSMOETbAgent::ACTION_DIM, 1);
        ag.getLegalActions(Stone::COLOR_RED, legal, idx, mask);
        Steps::instance().put(legal);
        std::vector<float> qs;
        t0 = nowMs();
        for (int i = 0; i < reps; i++) {
            ag.sparseQOnline(st, idx, qs, nullptr);
        }
        const double perSparse = (nowMs() - t0) / reps;

        ag.learnFromSearch = false;
        t0 = nowMs();
        ag.selectMove(Stone::COLOR_RED, 40, 0.0f);
        const double perStep40 = nowMs() - t0;

        /*
           批的耗时: 池子可能还不到 32 条 (训练循环按需累积), 所以按**实际条数**测,
           并给出每条的耗时 —— 界面常数要的是"一次更新多少钱", 不是"32 条多少钱"。
           ⚠ 上一版这里固定传 32, 而池里只有 14 条 ⇒ learnBatch 直接早退, 打印出
           "0.0 ms" 这个**假读数** (早退是正确行为, 但这个数什么也没测到)。
        */
        const int bsAvail = ag.samplePoolSize();
        const int bsTest = (bsAvail > 0) ? std::min(32, bsAvail) : 0;
        double perBatch = 0.0;
        if (bsTest > 0) {
            t0 = nowMs();
            ag.learnBatch(bsTest, 1);
            perBatch = nowMs() - t0;
        }
        const double perSample = (bsTest > 0) ? perBatch / bsTest : 0.0;

        std::printf("  全量前向          %.2f ms\n", perFull);
        std::printf("  稀疏叶子(合法列)   %.2f ms  (省 %.0f%%; 128 列的头太小, 省不了多少)\n",
                    perSparse, 100.0 * (1.0 - perSparse / (perFull > 0 ? perFull : 1)));
        std::printf("  一次决策(40 模拟)  %.1f ms\n", perStep40);
        std::printf("  一个批(%d 条)      %.1f ms  (每条 %.2f ms)\n",
                    bsTest, perBatch, perSample);
    }

    std::printf("\n============================================================\n");
    std::printf(" 断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    std::printf("============================================================\n");
    return g_failed;
}
