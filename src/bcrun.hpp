#ifndef BCRUN_HPP
#define BCRUN_HPP

/*
 * ============================================================================
 *  bcrun.hpp - 行为克隆 (BC) 的 **运行层**: 一次完整的"打标签 -> 训练 -> 读数"
 * ============================================================================
 *
 * 分工 (第五层, 接在 src/bcagent.hpp 之上):
 *
 *   1. `rl/bc.{h,cpp}`   —— 口径层 (BCSample / 掩码 CE 与它的解析梯度)
 *   2. `rl/ppo.{h,cpp}`  —— PPO 的稀疏内核 (actor-only 的 BC 反向)
 *   3. `bcagent.hpp`     —— agent 胶水层 (sampleFrom / update / evaluate)
 *   4. **本文件**         —— 一次完整运行: 造局面 -> AB 老师打标签 -> 划分 -> 训练
 *                           -> 训练/留出读数 -> 口径自检。**流程与措辞只有这一份**。
 *   5. 使用者             —— `test/train_bc_main.cpp` (命令行) 与
 *                           `ChessBoard::startBehaviorCloning` (界面按钮)。
 *
 * 为什么必须有这一层 (而不是界面自己再写一遍循环): 界面上那个按钮报出来的数
 * 必须与命令行工具**同源** —— 否则"界面上 top-1 到 42%"与"命令行 top-1 到 35%"
 * 会同时存在, 而两者的差别只来自"谁抄错了一行"。本工程为同一类问题付过账
 * (见 bcagent.hpp 顶部那段"四个副本"的说明)。所以:
 *   * 训练循环、留出集划分、读数口径、进度措辞**都在这里**;
 *   * 命令行只剩"解析参数 -> 建 agent -> 打印"三件事;
 *   * 界面只剩"起线程 -> 把行转发进面板"。
 *
 * ⚠ 一条已知的副作用 (调用方必须知道): `BC::sampleFrom` 会把 **agent 自己的棋盘
 *   引用** 改成它正在编码的那个局面 (agent 的编码器读的是自己那份 Chess)。本层在
 *   结束时**把它还原**成进来时的局面, 于是"跑完 BC 之后 agent 手里的棋盘还是原来
 *   那一盘" —— 界面上的决策路径虽然每次都会 `env = chess` 重新同步, 但把这条做进
 *   来之后, "BC 跑完棋盘的引用被换掉了"就不需要任何人再记得 (棋盘引用被换掉的表现
 *   会是"AI 忽然在一个不存在的局面上下棋", 而且极难查)。
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "abagent.h"
#include "bcagent.hpp"
#include "chess.h"

namespace BCRun {

/* ============================================================
 *  配置 (超参 + 数据规模)
 * ============================================================ */
struct Config {
    int positions = 1000;      /* 造多少个局面 (每个局面一条样本) */
    int teacherDepth = 3;      /* 老师 = ABAgent 的搜索深度 */
    int openingPlies = 8;      /* 造局面时先从开局随机走几手 (与 bench_ppo_distill 同一做法) */
    int epochs = 8;
    int batch = 64;
    float lr = 0.002f;
    unsigned seed = 20240901u;
    int holdoutEvery = 5;      /* 每 N 条抽 1 条进留出集 (0 = 全部进训练集) */
    int reportEvery = 1;       /* 每几个 epoch 回调一次读数 */
    int maxPositions = -1;     /* 冒烟: >0 时把训练集截断到这个条数 (留出集不动) */

    /*
       ---- [2026-10] 软目标 (用户问: "不直接使用 onehot、通过 abagent 计算概率分布
            再进行行为克隆是否会更好?") ----
       老师的标签从 one-hot 换成**多深度一致性**给出的分布 (见 bcagent.hpp 的
       softTargetFromAB): 深度 1..softDepths 各投一票, 票数占比 = 权重。
       为什么选它而不是"根分值 Boltzmann": AB 的根循环是窗口写法, 非最优孩子的返回是
       **界**而不是精确分值 ⇒ 拿它做 softmax 等于在裁剪 artifact 上克隆 (理由详述在
       softTargetFromAB 的注释里)。
       `softDepths <= 0` = 用 teacherDepth (即"最深那一层 + 它更浅的几种看法")。
       `softLinearWeight` = 让深的那层更重 (w_d ∝ d) 而不是每层等权。
    */
    bool softTargets = false;
    int softDepths = 0;
    bool softLinearWeight = false;

    /*
       ---- 取消钩子 (可省; 空 = 从不取消) ----
       为什么它必须存在 (而不是只靠 progress 回调返回 false): 取消要能发生在**任何**
       阶段。界面那条路的取消来自"关窗"(析构里置标志再 join), 而**打标签**这一段
       (8000 局面 x 深度 3 ≈ 2 分钟) 一个回调都不产生 —— 只靠 progress 的话, 用户
       按下关闭之后最多要等两分钟。

       本函数会被**频繁**调用 (打标签每 128 个局面一次、训练每个批一次), 所以它必须是
       一个廉价的读 (界面上传的是 `m_bcCancel.load()`)。
    */
    std::function<bool()> cancelRequested;
};

/* ============================================================
 *  结果 (界面与命令行都读它)
 * ============================================================ */
struct Result {
    int trainN = 0, holdoutN = 0;
    /* 三种"不合格"必须分开数 (混成一个"跳过"会把缺陷读成常态, 见 bcagent.hpp) */
    int teacherInvalid = 0;    /* 老师没给出有效着法 */
    int noLegalMoves = 0;      /* 局面本来就没有合法着法 (终局/困毙) */
    int teacherIllegal = 0;    /* **老师的着法不在合法集里** = 视角/索引算错了 */
    unsigned long long uniquePositions = 0;
    /*
       [2026-10] 软目标那一臂的两个读数:
         labelEntropy = 标签（老师分布）的平均熵 —— 它**就是 CE 的下界** (CE ≥ H(t)),
                        所以必须与 CE 一起报, 否则"CE 降不到 0"会被读成"学得差";
         softLabels   = 这一轮用的是不是软目标 (报告要能自证口径)。
    */
    double labelEntropy = 0.0;
    bool softLabels = false;
    BC::Metrics trainBefore, holdoutBefore, trainAfter, holdoutAfter;
    long long updates = 0, samplesUsed = 0, targetMissed = 0;
    double labelSec = 0.0, trainSec = 0.0;
    bool aborted = false;      /* 回调要求中止 (界面上的取消) */
    std::string error;         /* 非空 = 没能开跑 (可用样本不足等) */
};

inline double nowSec()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e9;
}

/* 从开局随机走 `opening` 手得到一个局面 (与权重无关, 只由 seed+index 决定) */
inline void makePosition(int index, const Config &cfg, Chess &out)
{
    out.reset();
    out.sideToMove = Stone::COLOR_RED;
    std::mt19937 rng((unsigned)(cfg.seed + (unsigned)index * 7919u));
    for (int i = 0; i < cfg.openingPlies; i++) {
        std::vector<Step *> legal;
        out.sample(out.sideToMove, legal);
        if (legal.empty()) break;
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        out.moveForward(&chosen, dummy);
    }
}

/*
 * 跑一次完整的 BC。
 *
 * `progress(line, result)` 每到一个里程碑被调用一次 (打完标签 / 每个 epoch / 训练结束):
 *   * line   = **已经排版好的那一行** (命令行直接 printf, 界面直接进面板 —— 两边看到的
 *              是同一句话, 这是"同源"这条纪律的落地方式);
 *   * result = 到这一步为止的读数 (界面可以边跑边显示 top-1/CE);
 *   * 返回值 = false 表示**请求中止** (界面上的取消按钮); 中止后本次不保存、
 *     `out.aborted = true`。
 *
 * 返回 false = 没跑起来 (`out.error` 说明原因: 可用样本不足)。
 */
template <class AgentT, class ProgressFn>
bool run(AgentT &agent, const Config &cfg, ProgressFn &&progress, Result &out)
{
    out = Result();

    /*
       进来时的棋盘引用: 跑完要还原 (见文件头那条副作用说明)。
       `Chess` 的拷贝是深拷贝, 这里只拷一次, 代价可以忽略。
    */
    Chess boardBefore = agent.chess;

    /* ---- 1. 造局面 + 用 AB 老师打标签 ---- */
    Chess envAb;
    envAb.reset();
    ABAgent ab(envAb, cfg.teacherDepth);

    std::vector<RL::BCSample> trainSet, holdout;
    std::set<unsigned long long> uniq;
    /* 取消钩子 (空 = 从不取消); 打标签这一段也要能被打断, 见 Config::cancelRequested */
    auto cancelled = [&cfg]() { return cfg.cancelRequested && cfg.cancelRequested(); };
    bool abortedEarly = false;
    /* 软目标那一臂: 标签熵的平均 (写进报告 —— CE 的下界就是它, 见 Metrics::targetEntropy) */
    double softLabelEntropySum = 0.0;
    long long softLabelN = 0;
    const double t0 = nowSec();
    for (int i = 0; i < cfg.positions; i++) {
        if ((i & 127) == 0 && i > 0 && cancelled()) { abortedEarly = true; break; }
        Chess pos;
        makePosition(i, cfg, pos);
        const int color = pos.sideToMove;

        /*
           老师: 一次搜索拿到根分值 + 选点。
           `getScoreValid()` 是"这次搜索有没有分数可言"的官方判据 (没有合法走法时为
           false) —— 用它当闸门, 而不是"看返回值像不像默认 Step"。
           [2026-10] 软目标那一臂: 老师的标签换成**多深度一致性**的分布 (深度
           1..softDepths 各投一票)。注意它**仍然用同一个 ABAgent 类型与同一个棋盘口径**,
           只是标签从 one-hot 变成分布 —— 这样 A/B 的差别只有"标签软硬"这一项。
        */
        envAb = pos;
        RL::BCSample s;
        BC::SampleFail why = BC::SampleFail::None;
        bool sampled = false;
        if (cfg.softTargets) {
            const int depths = (cfg.softDepths > 0) ? cfg.softDepths : cfg.teacherDepth;
            std::vector<float> weights;
            if (cfg.softLinearWeight) {
                weights.reserve((std::size_t)depths);
                for (int d = 1; d <= depths; d++) { weights.push_back((float)d); }
            }
            std::vector<Step> softMoves;
            std::vector<float> softProbs;
            if (!BC::softTargetFromAB(envAb, color, cfg.teacherDepth, depths,
                                      softMoves, softProbs, weights)) {
                out.teacherInvalid++;
                continue;
            }
            sampled = BC::sampleFrom(agent, pos, color, softMoves, softProbs, s, &why);
            if (sampled) { softLabelEntropySum += BC::targetEntropyOf(s); softLabelN++; }
        } else {
            const Step move = ab.getBestMove(color, cfg.teacherDepth);
            if (!ab.getScoreValid() || !move.valid) { out.teacherInvalid++; continue; }
            sampled = BC::sampleFrom(agent, pos, color, move, s, &why);
        }
        if (!sampled) {
            if (why == BC::SampleFail::TeacherIllegal) { out.teacherIllegal++; }
            else if (why == BC::SampleFail::NoLegalMoves) { out.noLegalMoves++; }
            else { out.teacherInvalid++; }
            continue;
        }
        uniq.insert(s.posHash);
        if (cfg.holdoutEvery > 0 && (i % cfg.holdoutEvery) == 0) {
            holdout.push_back(s);
        } else {
            trainSet.push_back(s);
        }
    }
    out.labelSec = nowSec() - t0;
    out.trainN = (int)trainSet.size();
    out.holdoutN = (int)holdout.size();
    out.uniquePositions = (unsigned long long)uniq.size();
    out.labelEntropy = (softLabelN > 0) ? (softLabelEntropySum / (double)softLabelN) : 0.0;
    out.softLabels = cfg.softTargets;

    if (trainSet.empty() || holdout.empty()) {
        if (out.holdoutN == 0 && out.trainN > 0) {
            /*
               只给了训练集 (holdoutEvery=0) 也是合法的: 那就用训练集当"留出"来报读数,
               而不是报"可用样本不足" —— 读数会因此偏乐观, 这一点在报告里写明。
            */
            holdout = trainSet;
            out.holdoutN = out.trainN;
        } else {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "可用样本不足 (训练 %d / 留出 %d) —— 检查 positions / openingPlies",
                          out.trainN, out.holdoutN);
            out.error = buf;
            agent.chess = boardBefore;
            return false;
        }
    }

    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[1] 标注完成: 训练 %d / 留出 %d 条, %.1f s (%.1f ms/局面)%s",
                      out.trainN, out.holdoutN, out.labelSec,
                      1000.0 * out.labelSec / (double)std::max(1, cfg.positions),
                      abortedEarly ? "  **打标签阶段被取消**" : "");
        if (!progress(std::string(buf), out)) { out.aborted = true; agent.chess = boardBefore; return true; }
        if (abortedEarly) {
            out.aborted = true;
            agent.chess = boardBefore;
            return true;
        }

        std::snprintf(buf, sizeof(buf),
                      "    不合格: 老师无有效着法 %d / 局面无合法着法 %d /"
                      " **老师着法不在合法集 %d** (最后一个 = 视角或索引算错)",
                      out.teacherInvalid, out.noLegalMoves, out.teacherIllegal);
        progress(std::string(buf), out);

        std::snprintf(buf, sizeof(buf),
                      "    局面去重: %llu 个互不相同的局面 / %d 个尝试过 label 的局面",
                      out.uniquePositions, cfg.positions);
        progress(std::string(buf), out);

        if (out.teacherIllegal > 0) {
            progress(std::string("    **警告**: 老师着法不在合法集里 —— 这些样本会被内核"
                                 "**丢弃** (目标落空时 L≡0 而 dL/dz = π ≠ 0: 一条没有信息"
                                 "却会改权重的样本), 读数的可信度下降"), out);
        }
    }

    if (cfg.maxPositions > 0 && (int)trainSet.size() > cfg.maxPositions) {
        trainSet.resize((std::size_t)cfg.maxPositions);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "    (冒烟: 训练集截断到 %d 条)", cfg.maxPositions);
        progress(std::string(buf), out);
    }

    /* ---- 2. 训练前的读数 ---- */
    BC::evaluate(agent, trainSet, out.trainBefore);
    BC::evaluate(agent, holdout, out.holdoutBefore);
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[2] 克隆前: 训练 top-1 %.2f%% P(老师) %.5f CE %.4f 熵 %.4f |"
                      " 留出 top-1 %.2f%% P(老师) %.5f CE %.4f 熵 %.4f",
                      out.trainBefore.top1Pct, out.trainBefore.pTeacher, out.trainBefore.ce,
                      out.trainBefore.entropy,
                      out.holdoutBefore.top1Pct, out.holdoutBefore.pTeacher, out.holdoutBefore.ce,
                      out.holdoutBefore.entropy);
        if (!progress(std::string(buf), out)) { out.aborted = true; agent.chess = boardBefore; return true; }
    }

    /* ---- 3. 训练 (只更新策略头) ---- */
    const double t1 = nowSec();
    std::mt19937 rng(cfg.seed ^ 0x5bd1e995u);
    double lastBatchCe = 0.0;
    for (int ep = 0; ep < cfg.epochs; ep++) {
        std::shuffle(trainSet.begin(), trainSet.end(), rng);
        int batches = 0;
        bool abortThisEpoch = false;
        std::vector<RL::BCSample> chunk;
        chunk.reserve((std::size_t)std::max(1, cfg.batch));
        for (std::size_t i = 0; i < trainSet.size(); i++) {
            chunk.push_back(trainSet[i]);
            if ((int)chunk.size() < cfg.batch && i + 1 < trainSet.size()) { continue; }
            /*
               每个批边界查一次取消 —— 界面上的取消来自"关窗"(析构里置标志再 join),
               而打标签那一段最长可以到分钟级, 只按 epoch 检查的话关窗要等到这一轮跑完。
            */
            if (cancelled()) { abortThisEpoch = true; break; }
            const BC::UpdateStat st = BC::update(agent, chunk, cfg.lr);
            if (std::isfinite(st.ce)) { lastBatchCe = (double)st.ce; out.updates++; }
            out.samplesUsed += st.used;
            out.targetMissed += st.targetMiss;
            chunk.clear();
            batches++;
        }
        if (abortThisEpoch) {
            out.aborted = true;
            out.trainSec = nowSec() - t1;
            agent.chess = boardBefore;
            return true;
        }
        if (cfg.reportEvery > 0
            && ((ep + 1) % cfg.reportEvery == 0 || ep + 1 == cfg.epochs)) {
            BC::evaluate(agent, trainSet, out.trainAfter);
            BC::evaluate(agent, holdout, out.holdoutAfter);
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "[3] epoch %d/%d: %d 个批 (累计 %lld 次更新 / %lld 条样本),"
                          " 最近一批 CE %.4f | top-1 训练 %5.2f%% / 留出 %5.2f%% |"
                          " CE 训练 %.4f / 留出 %.4f",
                          ep + 1, cfg.epochs, batches, out.updates, out.samplesUsed, lastBatchCe,
                          out.trainAfter.top1Pct, out.holdoutAfter.top1Pct,
                          out.trainAfter.ce, out.holdoutAfter.ce);
            if (!progress(std::string(buf), out)) {
                out.aborted = true;
                out.trainSec = nowSec() - t1;
                agent.chess = boardBefore;
                return true;
            }
        }
    }
    out.trainSec = nowSec() - t1;
    agent.chess = boardBefore;      /* 还原棋盘引用 (见文件头) */
    return true;
}

/* ============================================================
 *  报告排版 (命令行与界面共用同一套措辞)
 * ============================================================ */

/* 跑之前那一屏 (说明这一次在什么口径下跑) */
inline std::string formatHeader(const Config &cfg, const char *agentLabel,
                                const char *backboneDesc, const char *modeDesc)
{
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "=== 行为克隆 (BC): 老师 = Alpha-Beta 深度 %d ===\n"
                  "agent    : %s (%s)\n"
                  "口径     : %s\n"
                  "标签     : %s\n"
                  "样本     : %d 个局面 (随机开局 %d 手, seed=%u)\n"
                  "训练     : batch=%d, epochs=%d, lr=%.4f (只更新策略头)\n",
                  cfg.teacherDepth, agentLabel, backboneDesc, modeDesc,
                  cfg.softTargets
                      ? "**软目标 (多深度一致性)**: 深度 1..N 各投一票, 票数占比 = 分布"
                      : "one-hot (老师那一手)",
                  cfg.positions, cfg.openingPlies, cfg.seed,
                  cfg.batch, cfg.epochs, (double)cfg.lr);
    return std::string(buf);
}

/*
 * 跑完之后的报告。`totalSec` = 整段耗时 (含建网), 由调用方给。
 * 最后那三条"怎么读这份报告"是刻意写死在报告里的 —— prior top-1 是 BC 的**训练
 * 目标本身**, 不写清楚就会被当成棋力证据 (docs/training_optimization.md §7.10)。
 */
inline std::string formatReport(const Config &cfg, const Result &r,
                                const char *agentLabel, const char *backboneDesc,
                                const char *caliperLines, double totalSec)
{
    char buf[4096];
    const double trainCeDrop = (r.trainAfter.ce > 0.0) ? r.trainBefore.ce / r.trainAfter.ce : 0.0;
    const double holdCeDrop = (r.holdoutAfter.ce > 0.0) ? r.holdoutBefore.ce / r.holdoutAfter.ce : 0.0;
    std::snprintf(buf, sizeof(buf),
                  "%s\n"
                  "[4] 克隆后 (训练 %.1f s, %lld 次 actor 更新):\n"
                  "    训练   n=%-6d prior top-1 %6.2f%%   P(老师着法) %.5f   CE %.4f   策略熵 %.4f\n"
                  "    留出   n=%-6d prior top-1 %6.2f%%   P(老师着法) %.5f   CE %.4f   策略熵 %.4f\n"
                  "    CE 下降: 训练 %.4f -> %.4f (%.2fx), 留出 %.4f -> %.4f (%.2fx)\n"
                  "    标签: %s (平均 H(t): 训练 %.4f / 留出 %.4f; **CE 的下界就是它**, 软目标不可能降到 0)\n"
                  "    train/留出 CE 差 = %.3f nats (>1 说明已经在过拟合, 该加数据而不是加步数)\n"
                  "    KL 口径 (CE − H(t), 软/one-hot **唯一可比**的那一列): 训练 %.4f -> %.4f, 留出 %.4f -> %.4f"
                  "  (train/留出 KL 差 = %.3f)\n"
                  "%s"
                  "\n[7] 怎么读这份报告 (三条边界):\n"
                  "    * prior top-1 是 **BC 的训练目标本身** —— 它高只说明「克隆成功了」;\n"
                  "      这个数上升之后, 「与 AB 的一致率」就不再是独立的棋力证据 (参照要换成更深的 AB);\n"
                  "    * 上限就是老师: 模仿 AB 深度 %d, 不可能越过它;\n"
                  "    * 棋力只有锚点对局能回答 (bench_anchor / bench_policy_agreement), 本工具不测棋力。\n"
                  "总耗时 %.1f s\n",
                  (r.aborted ? "**本次被中止 (用户取消): 上面的读数是中止那一刻的**\n" : ""),
                  r.trainSec, r.updates,
                  r.trainAfter.n, r.trainAfter.top1Pct, r.trainAfter.pTeacher,
                  r.trainAfter.ce, r.trainAfter.entropy,
                  r.holdoutAfter.n, r.holdoutAfter.top1Pct, r.holdoutAfter.pTeacher,
                  r.holdoutAfter.ce, r.holdoutAfter.entropy,
                  r.trainBefore.ce, r.trainAfter.ce, trainCeDrop,
                  r.holdoutBefore.ce, r.holdoutAfter.ce, holdCeDrop,
                  r.softLabels ? "软目标 (多深度一致性)" : "one-hot",
                  r.trainAfter.targetEntropy,
                  r.holdoutAfter.targetEntropy,
                  r.holdoutAfter.ce - r.trainAfter.ce,
                  r.trainBefore.ce - r.trainBefore.targetEntropy,
                  r.trainAfter.ce - r.trainAfter.targetEntropy,
                  r.holdoutBefore.ce - r.holdoutBefore.targetEntropy,
                  r.holdoutAfter.ce - r.holdoutAfter.targetEntropy,
                  (r.holdoutAfter.ce - r.holdoutAfter.targetEntropy)
                      - (r.trainAfter.ce - r.trainAfter.targetEntropy),
                  (caliperLines != nullptr) ? caliperLines : "",
                  cfg.teacherDepth, totalSec);
    (void)agentLabel;
    (void)backboneDesc;
    return std::string(buf);
}

/* ============================================================
 *  口径自检 (PPO 有内核级计数器; SAC 三支只有"更新了哪一路")
 * ============================================================ */
inline std::string caliperFor(PPOMCTSAgent &ag, const Result &r)
{
    char buf[1200];
    std::snprintf(buf, sizeof(buf),
                  "\n[5] 口径自检 (PPO): BC 样本 %lld, 其中走合法列口径 %lld, actor 更新 %lld 次\n"
                  "    参与更新的样本 %lld 条; **目标落空 (不在合法集里) 被丢弃 %lld 条**%s\n"
                  "    critic 是否被 BC 动过: 否 (BC 路径整段不含 critic 前向/反向, 见 rl/ppo.cpp)\n"
                  "    lastLoss (critic 的 MSE) 是否被 BC 写过: 否 (BC 只写 lastActorLoss = 批平均 CE)\n"
                  /*
                     [2026-10] 两条新读数:
                       * 梯度裁剪口径 —— 老口径 `clipGrad` 的真实语义是"逐张量梯度归一化到
                         单位长度"(不是裁剪), 而 RMSProp 的逐坐标归一会把任何整张量的等比
                         缩放抵消掉 ⇒ 三种模式在实用尺度上是同一条轨迹 (见 rl/optimize.h);
                       * 最近一批的 actor 梯度范数 —— 以前没有这个数, 而"关掉裁剪之后学习率
                         要不要跟"只能靠它判断。
                  */
                  "    梯度裁剪: %s | 最近一批 actor 梯度范数: %.6g\n"
                  "    MoE 批边界: finalizeMoeBatch() 已在每个批之前调用 "
                  "(BC 期间也走无辅助损失偏置回路; 见 src/bcagent.hpp)\n",
                  ag.ppo.bcSamples, ag.ppo.bcSparseSteps, ag.ppo.bcSteps,
                  r.samplesUsed, r.targetMissed,
                  (r.targetMissed > 0) ? "  <-- 造样本那一层的视角/索引算错了" : "",
                  RL::gradClipModeName(ag.ppo.gradClipMode), ag.ppo.actorGradNorm);
    return std::string(buf);
}

template <class SACLike>
std::string caliperFor(SACLike &ag, const Result &r)
{
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "\n[5] 口径自检 (SAC): 骨干口径 = %s\n"
                  "    只更新策略路径: %s\n"
                  "    参与更新的样本 %lld 条 (共 %lld 个批); **目标落空被丢弃 %lld 条**%s\n",
                  ag.sharedTrunk() ? "共享骨干+三头 (TrunkMode::Shared)"
                                   : "独立网络 (TrunkMode::Separate)",
                  ag.sharedTrunk() ? "trunk + actorHead (Q 头权重一个字节不动, 但共享骨干的表示会变)"
                                   : "actor (q1/q2 一个字节不动)",
                  r.samplesUsed, r.updates, r.targetMissed,
                  (r.targetMissed > 0) ? "  <-- 造样本那一层的视角/索引算错了" : "");
    return std::string(buf);
}

} // namespace BCRun

#endif // BCRUN_HPP
