#ifndef BCAGENT_HPP
#define BCAGENT_HPP

/*
 * ============================================================================
 *  bcagent.hpp - 行为克隆 (BC) 的 **agent 胶水层**
 * ============================================================================
 *
 * 分工 (三层, 每一层只做一件事):
 *
 *   1. `rl/bc.{h,cpp}`      —— **口径层**: BCSample 的结构、掩码交叉熵与它对 logits
 *                              的解析梯度。纯张量数学, 不认识棋盘。
 *   2. `rl/ppo.{h,cpp}`     —— **PPO 的稀疏内核**: 合法列前向 + actor-only 反向
 *                              (bcGradSparse / bcApplyGradients)。
 *   3. **本文件**            —— **把棋盘接到上面两层**: 用 agent 自己的编码器与动作
 *                              映射造 BCSample, 用 agent 自己的策略口径算读数,
 *                              用 agent 自己的网络做一次批更新。
 *
 * 为什么本层必须是**模板 + 重载**而不是"给每个 agent 类加一个方法":
 * PPO 与 SAC 三支 (SACAZAgent / SACAZMoEMlpAgent / SACAZMoETbAgent) 的 BC 逻辑
 * 是同一件事的三个副本 —— 而"造样本"这一段在四个类上**逐字相同** (都走
 * encodeStateFor + getLegalActions + stepToActionIdx)。写四份的话, "老师着法算错
 * 视角"这类缺陷有四个地方可能出错, 而它不会报错、只会静默变成零梯度样本。
 * 本工程已经为同一类问题付过账 (chessboard.h 里的 `template <class SACLike>`
 * 是同一个手法)。
 *
 * ----------------------------------------------------------------------------
 *  鸭子类型约定 (四个 agent 类都恰好满足; 缺一个只会在实例化时报错, 不会静默)
 * ----------------------------------------------------------------------------
 *   AgentT::STATE_DIM / AgentT::ACTION_DIM   编译期常量
 *   ag.chess                                 Chess& (公开成员; 编码器读它)
 *   ag.encodeStateFor(color, state)
 *   ag.getLegalActions(color, steps, actionIndices, mask)   (steps 由调用方归还)
 *   ag.stepToActionIdx(step, color)           (必须在 color 的规范视角下算)
 *
 *   PPO 重载额外用: ag.ppo (RL::PPO) 的 bcGradSparse / bcGradDense /
 *                   bcApplyGradients / resetMoeBatchStats / actionMasked /
 *                   actorP / maskedTrainHead
 *   SAC 模板额外用: ag.policy(state, mask, pi) / ag.actor / ag.actorHead / ag.trunk /
 *                   ag.sharedTrunk()
 *
 * ----------------------------------------------------------------------------
 *  ⚠ 一条纪律: **状态与动作下标必须用同一个 color**
 * ----------------------------------------------------------------------------
 * 两个 agent 的状态编码都做规范视角镜像 (轮到黑方时 x -> 9-x), 而
 * `stepToActionIdx` 也必须用同一个 color —— 否则"状态按黑方、动作按红方"会让
 * 老师的着法落到另一个坐标帧里。那种情况下目标**大概率不在合法集里**, 于是被
 * 对齐那一步丢掉 ⇒ t ≡ 0 ⇒ 梯度恒为 0。为了让这件事不可能静默发生:
 *   * `sampleFrom` **验证**老师的动作下标确实在合法集里, 不在就返回 false;
 *   * `RL::PPO::bcTargetMisses` 在内核层再数一遍 (两道闸门)。
 */

#include <cmath>
#include <limits>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/bc.h"

namespace BC {

/* ============================================================================
 *  1) 造样本: (局面, 老师着法) -> BCSample
 * ============================================================================ */

/*
 * `sampleFrom` 失败的原因 —— **分开数出来是最重要的**: 三种情形的性质完全不同,
 * 混成一个"跳过"就会把"视角算错"读成"这些局面没合法着法, 很正常"。
 */
enum class SampleFail {
    None = 0,
    TeacherInvalid,     /* Step::valid == false (老师没给出有效着法) */
    NoLegalMoves,       /* 这个局面本来就没有合法着法 (终局/困毙) —— 正常 */
    TeacherIllegal      /* **老师的着法不在合法集里** —— 视角/索引算错了 */
};

/*
 * 用 agent **自己的**编码/动作口径把一条监督样本编出来。
 *
 * 副作用 (刻意, 与 bench_ppo_distill / bench_diag 同做法): 会把 `ag.chess` 改成
 * `pos` —— 两个 agent 的编码器读的都是"自己那份棋盘引用" (encodeStateFor 只看
 * chess.stones 与规则上下文, getLegalActions 走 chess.sample)。调用方自己负责
 * 之后把 agent 的棋盘摆回它需要的局面 (训练器/测试都是"一个局面编一条", 不需要
 * 还原)。
 *
 * 返回 false 时 `*failOut` 说明是哪一种 (默认 nullptr = 不关心), 见 SampleFail:
 *   * 老师着法无效;
 *   * 该局面没有合法着法;
 *   * **老师的动作下标不在完整合法集里** —— 视角/索引算错时的典型症状, 而它进了
 *     训练只会变成"零梯度样本" (见 rl/bc.h 第 2 条与 RL::PPO::bcTargetMisses)。
 */
template <class AgentT>
bool sampleFrom(AgentT &ag, const Chess &pos, int color, const Step &teacher,
                RL::BCSample &out, SampleFail *failOut = nullptr)
{
    /*
       单老师 (one-hot) 直接委托给下面的**多老师**重载 —— 一份实现两处用, 免得
       "单老师的合法性检查"与"多老师的"各写一遍而分叉 (本工程反复记过的坑)。
    */
    std::vector<Step> one{teacher};
    std::vector<float> oneP{1.0f};
    return sampleFrom(ag, pos, color, one, oneP, out, failOut);
}

/*
 * ---- [2026-10] 多老师 (软目标) 版本: 目标是一个**在合法集上的分布** ----
 *
 * 用户问题: "不直接使用 onehot、通过 abagent 计算概率分布再进行行为克隆是否会更好?"
 * 这条路径就是它的落地: `teachers[k]` + `probs[k]` 是老师给出的候选着法及其权重
 * (典型来源 = AB 的多深度一致性, 见 softTargetFromAB)。
 *
 * 与 one-hot 版本的差别只有两处 (其余逐行相同):
 *   1. **每一个**候选着法都必须落在完整合法集里 —— 任何一个不合法就整条样本作废
 *      (`TeacherIllegal`), **不做"丢掉不合法的那几项"**。理由: 丢掉之后 Σt < 1,
 *      而 `dL/dz = π − t` 在 Σt ≠ 1 时不再是"匹配一个分布", 梯度会多出一份往老师
 *      那一侧推的偏置 —— 那是口径错, 而且从读数上看不出来 (见 rl/bc.h 第 2 条)。
 *   2. 权重**强制归一** (Σ = 1): 调用方给的可能是票数或未归一的分值, 这里统一成分布。
 *      one-hot 时 Σ 天然为 1, 所以旧路径的数值逐位不变。
 */
template <class AgentT>
bool sampleFrom(AgentT &ag, const Chess &pos, int color,
                const std::vector<Step> &teachers, const std::vector<float> &probs,
                RL::BCSample &out, SampleFail *failOut = nullptr)
{
    if (failOut != nullptr) { *failOut = SampleFail::None; }
    out.clear();
    if (teachers.empty() || teachers.size() != probs.size()) {
        if (failOut != nullptr) { *failOut = SampleFail::TeacherInvalid; }
        return false;
    }

    /* 编码器与走法生成都读 agent 自己那份棋盘 */
    ag.chess = pos;

    std::vector<Step *> legal;
    std::vector<int> legalIdx;
    RL::Tensor mask((std::size_t)AgentT::ACTION_DIM, 1);
    ag.getLegalActions(color, legal, legalIdx, mask);

    if (legalIdx.empty()) {
        Steps::instance().put(legal);
        if (failOut != nullptr) { *failOut = SampleFail::NoLegalMoves; }
        return false;
    }

    /* 每个候选着法 -> 动作下标, 并逐个检查它在不在合法集里 */
    std::vector<int> idxOut;
    idxOut.reserve(teachers.size());
    for (std::size_t k = 0; k < teachers.size(); k++) {
        if (!teachers[k].valid) {
            Steps::instance().put(legal);
            if (failOut != nullptr) { *failOut = SampleFail::TeacherInvalid; }
            return false;
        }
        const int a = ag.stepToActionIdx(teachers[k], color);
        bool ok = false;
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            if (legalIdx[i] == a) { ok = true; break; }
        }
        if (!ok) {
            Steps::instance().put(legal);
            if (failOut != nullptr) { *failOut = SampleFail::TeacherIllegal; }
            return false;
        }
        idxOut.push_back(a);
    }
    Steps::instance().put(legal);

    /* 归一 (one-hot 时是恒等变换; 票数/未归一权重在这里变成分布) */
    double sum = 0.0;
    for (std::size_t k = 0; k < probs.size(); k++) {
        if (probs[k] > 0.0f) { sum += (double)probs[k]; }
    }
    if (sum <= 1e-12) {
        if (failOut != nullptr) { *failOut = SampleFail::TeacherInvalid; }
        return false;
    }

    out.state = RL::Tensor((std::size_t)AgentT::STATE_DIM, 1);
    /* 视角必须与 stepToActionIdx 用**同一个** color (见文件头那条纪律) */
    ag.encodeStateFor(color, out.state);
    out.legalIdx = legalIdx;
    out.targetIdx.clear();
    out.targetProb.clear();
    for (std::size_t k = 0; k < idxOut.size(); k++) {
        const float p = (float)((double)probs[k] / sum);
        if (p <= 0.0f) { continue; }          /* 0 权重的项不进目标 (它本来就等于不存在) */
        /* 同一个着法被多次投票时**合并** (多深度一致性里很常见) */
        bool merged = false;
        for (std::size_t j = 0; j < out.targetIdx.size(); j++) {
            if (out.targetIdx[j] == idxOut[k]) {
                out.targetProb[j] += p;
                merged = true;
                break;
            }
        }
        if (!merged) {
            out.targetIdx.push_back(idxOut[k]);
            out.targetProb.push_back(p);
        }
    }
    if (out.targetIdx.empty()) {
        if (failOut != nullptr) { *failOut = SampleFail::TeacherInvalid; }
        return false;
    }
    out.posHash = ag.chess.computeHash();
    return true;
}

/*
 * softTargetFromAB - **多深度一致性**的软目标 (2026-10)
 *
 * 为什么用这个而不是"根分值 Boltzmann": `ABAgent` 的根循环是"保持当前最好"的窗口写法
 * (abagent.cpp 的 getBestMove: beta = r / alpha = r), 第一个孩子之后的搜索都以当前最优
 * 为中心 ⇒ 失败低/失败高的孩子返回的是**界**, 不是精确分值。拿这些数做 softmax 等于在
 * 一个由裁剪产生的伪分布上克隆 (看起来很有道理, 实际是 artifact)。要精确必须根层全窗口
 * 搜每个孩子。
 *
 * 多深度一致性只用**现有 API** (`getBestMove(color, depth)` 的深度覆写), 语义清楚:
 * "每一层深度投一票" —— 票数/深度数 = 权重。它天然表达"这一手稳不稳" (三层都选它 vs
 * 只有最深那层选它), 而且**全部候选都是合法着法** (每次搜索返回的都是合法走法)。
 *
 * `weights`: 空 = 每层等权 (默认); 否则长度必须 = depths, 逐层权重 (例如让深的那层更重)。
 * 返回 false = 这个局面没有任何一层给出有效着法 (终局/困毙)。
 */
inline bool softTargetFromAB(Chess &board, int color, int maxDepth, int depths,
                             std::vector<Step> &movesOut, std::vector<float> &probsOut,
                             const std::vector<float> &weights = std::vector<float>())
{
    movesOut.clear();
    probsOut.clear();
    if (depths <= 0 || maxDepth <= 0) {
        return false;
    }
    ABAgent ab(board, maxDepth);
    std::vector<Step> votes;
    std::vector<float> voteW;
    for (int d = 1; d <= depths; d++) {
        const Step mv = ab.getBestMove(color, d);
        if (!mv.valid) {
            continue;                      /* 这一层没给出着法: 这一票作废 (不是整条作废) */
        }
        /*
           与已有票**合并** (同一个着法被多层选中是常态 —— 那正是"一致"的信息量)。
           Step 的相等按 (id, nextPos) 判: 同一子的同一个落点。
        */
        bool merged = false;
        for (std::size_t i = 0; i < votes.size(); i++) {
            if (votes[i].id == mv.id && votes[i].nextPos == mv.nextPos) {
                voteW[i] += weights.empty() ? 1.0f : weights[(std::size_t)(d - 1)];
                merged = true;
                break;
            }
        }
        if (!merged) {
            votes.push_back(mv);
            voteW.push_back(weights.empty() ? 1.0f : weights[(std::size_t)(d - 1)]);
        }
    }
    if (votes.empty()) {
        return false;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < voteW.size(); i++) { sum += (double)voteW[i]; }
    if (sum <= 1e-12) {
        return false;
    }
    for (std::size_t i = 0; i < votes.size(); i++) {
        movesOut.push_back(votes[i]);
        probsOut.push_back((float)((double)voteW[i] / sum));
    }
    return true;
}

/* ============================================================================
 *  2) 读数: 教师 top-1 命中率 / P(教师着法) / 交叉熵
 * ============================================================================
 * 三个量都在**同一个策略口径**下算 (PPO: 与训练口径一致的稀疏/全量; SAC: 掩码
 * softmax)。口径不一致的话, "CE 从 8.99 降到 3.49" 这种读数就没有意义。
 *
 * 语义边界 (必须与读数一起报): `top1Pct` 是**策略头单独**的命中率, 也就是 BC 的
 * **训练目标本身** —— 它上升只说明"克隆成功了", 不是棋力证据。历史上本工程量过
 * "一致率涨 9 倍而胜率一动没动" (docs/training_optimization.md §7.10)。
 */
struct Metrics {
    int    n = 0;             /* 参与统计的样本数 */
    int    top1Hits = 0;
    double top1Pct = 0.0;
    double pTeacher = 0.0;    /* 批平均 P(老师**主着法**) (软目标下 = argmax 那一项) */
    double ce = 0.0;          /* 批平均交叉熵 (软目标下是**整个分布**的 CE) */
    double entropy = 0.0;     /* 批平均策略熵 (合法集上; 看"塌没塌"用) */
    /*
       ---- [2026-10] 软目标的**下界**: 批平均 H(t) ----
       为什么必须一起报: 软目标的 CE = H(t) + KL(t‖π) ≥ H(t) ⇒ "CE 降到 0" 这个目标
       对软目标**不可能达到**, 而报告里若还按"0 是满分"去读, 软的那一臂会被当成学得更差。
       one-hot 的目标熵恒为 0 (旧读数逐位不变)。
    */
    double targetEntropy = 0.0;
    int    softSamples = 0;   /* targetProb 里不止一项 (= 真正的软目标) 的样本数 */
    int    skipped = 0;       /* 结构与口径不合格而被跳过的条数 */
};

/* 目标分布的熵 (只对该样本 targetProb 里 >0 的项算) */
inline double targetEntropyOf(const RL::BCSample &s)
{
    double h = 0.0;
    for (std::size_t k = 0; k < s.targetProb.size(); k++) {
        const double p = (double)s.targetProb[k];
        if (p > 0.0) { h -= p * std::log(p); }
    }
    return h;
}

/* PPO: 口径跟随**训练口径** (maskedTrainHead) —— 对照臂必须与训练臂同尺 */
inline void evaluate(PPOMCTSAgent &ag, const std::vector<RL::BCSample> &set,
                     Metrics &m)
{
    m = Metrics();
    std::vector<float> probs;
    for (std::size_t k = 0; k < set.size(); k++) {
        const RL::BCSample &s = set[k];
        const int a = s.argmaxTarget();
        if (!s.valid() || a < 0) { m.skipped++; continue; }
        if (s.state.size() != (std::size_t)PPOMCTSAgent::STATE_DIM) { m.skipped++; continue; }

        if (ag.ppo.maskedTrainHead) {
            if (!ag.ppo.actionMasked(s.state, s.legalIdx, probs)) { m.skipped++; continue; }
            int best = -1;
            float bestP = -1.0f;
            int hitIdx = -1;
            for (std::size_t i = 0; i < probs.size(); i++) {
                if (probs[i] > bestP) { bestP = probs[i]; best = (int)i; }
                if (s.legalIdx[i] == a) { hitIdx = (int)i; }
            }
            if (hitIdx < 0) { m.skipped++; continue; }
            m.pTeacher += (double)probs[(std::size_t)hitIdx];
            /*
               CE 用**整个目标分布**算 (软目标下这才是正口径); one-hot 时它逐位等于
               `-log π(a*)` (旧读数不变)。目标的质量只落在合法集里 —— 这是 sampleFrom
               保证的 (任何一项不合法就整条作废)。
            */
            for (std::size_t k = 0; k < s.targetIdx.size(); k++) {
                const float tp = s.targetProb[k];
                if (tp <= 0.0f) { continue; }
                for (std::size_t i = 0; i < s.legalIdx.size(); i++) {
                    if (s.legalIdx[i] == s.targetIdx[k]) {
                        m.ce -= (double)tp * std::log((double)probs[i] + 1e-8);
                        break;
                    }
                }
            }
            for (std::size_t i = 0; i < probs.size(); i++) {
                if (probs[i] > 0.0f) { m.entropy -= (double)probs[i] * std::log((double)probs[i]); }
            }
            if (best == hitIdx) { m.top1Hits++; }
            m.targetEntropy += targetEntropyOf(s);
            if (s.targetIdx.size() > 1) { m.softSamples++; }
        } else {
            /* 全量 8100 维口径 (对照臂): 概率不重归一到合法集, 与训练口径一致 */
            RL::Tensor &p = ag.ppo.actorP.forward(s.state);
            if ((std::size_t)a >= p.size()) { m.skipped++; continue; }
            m.pTeacher += (double)p[(std::size_t)a];
            m.ce -= std::log((double)p[(std::size_t)a] + 1e-8);
            double ent = 0.0;
            for (std::size_t i = 0; i < p.size(); i++) {
                if (p[i] > 0.0f) { ent -= (double)p[i] * std::log((double)p[i]); }
            }
            m.entropy += ent;
            if ((int)p.argmax() == a) { m.top1Hits++; }
        }
        m.n++;
    }
    if (m.n > 0) {
        m.top1Pct = 100.0 * (double)m.top1Hits / (double)m.n;
        m.pTeacher /= (double)m.n;
        m.ce /= (double)m.n;
        m.entropy /= (double)m.n;
        /* ⚠ 这一行必须有: 忘了除就把"总和"当成了"平均"(本轮实测: 40 条留出样本报出
           H(t)=16.14, 而真值是 0.40) —— 一个数量级级别的假读数, 而且看起来"很合理"。 */
        m.targetEntropy /= (double)m.n;
    }
}

/* SAC 三支 (SACAZAgent / SACAZMoEMlpAgent / SACAZMoETbAgent): 掩码 softmax 口径 */
template <class SACLike>
void evaluate(SACLike &ag, const std::vector<RL::BCSample> &set, Metrics &m)
{
    m = Metrics();
    RL::Tensor mask, target;
    /*
       ⚠ `pi` **必须**先按 ACTION_DIM 分配好: agent 的 maskedSoftmax 是"往调用方的
       张量里写 π", 它自己不会按 actionDim 扩容 (agent 内部的热路径都用预分配的
       成员缓冲)。给一个空张量进去 = 逐列越界写 —— 那是 0xC0000005 而不是断言。
    */
    RL::Tensor pi((std::size_t)SACLike::ACTION_DIM, 1);
    for (std::size_t k = 0; k < set.size(); k++) {
        const RL::BCSample &s = set[k];
        const int a = s.argmaxTarget();
        if (!s.valid() || a < 0) { m.skipped++; continue; }
        if (s.state.size() != (std::size_t)SACLike::STATE_DIM) { m.skipped++; continue; }
        if (!RL::denseMaskAndTarget(SACLike::ACTION_DIM, s, mask, target)) { m.skipped++; continue; }
        if (a >= SACLike::ACTION_DIM || mask[(std::size_t)a] <= 0.5f) { m.skipped++; continue; }

        ag.policy(s.state, mask, pi);
        m.pTeacher += (double)pi[(std::size_t)a];
        /* SAC 这一侧走 `maskedCrossEntropy`, 它对软目标本来就是整分布的 CE */
        m.ce += RL::maskedCrossEntropy(pi, target);
        for (int i = 0; i < SACLike::ACTION_DIM; i++) {
            if (pi[i] > 0.0f) { m.entropy -= (double)pi[i] * std::log((double)pi[i]); }
        }
        int best = -1;
        if (RL::argmaxOnLegal(pi, mask, best) && best == a) { m.top1Hits++; }
        m.targetEntropy += targetEntropyOf(s);
        if (s.targetIdx.size() > 1) { m.softSamples++; }
        m.n++;
    }
    if (m.n > 0) {
        m.top1Pct = 100.0 * (double)m.top1Hits / (double)m.n;
        m.pTeacher /= (double)m.n;
        m.ce /= (double)m.n;
        m.entropy /= (double)m.n;
        m.targetEntropy /= (double)m.n;
    }
}

/* ============================================================================
 *  3) 一次批更新
 * ============================================================================
 * 两个实现都**只动策略路径**:
 *   * PPO : RL::PPO::bcGradSparse* -> bcApplyGradients (critic 整段不存在);
 *   * SAC : 策略头 (共享口径下连骨干一起) 的 masked CE 梯度 -> 只对它调 RMSProp。
 *
 * ⚠ SAC 共享骨干口径 (TrunkMode::Shared) 的**语义边界**: 骨干只有一份, 于是这一批
 *   策略梯度**也会**改到 Q 头所依赖的表示 (而 Q 头自己的权重一个字节不动)。
 *   这是"共享表示"的固有代价, 不是 bug; 但因此 "BC 不影响 critic" 这条断言在共享
 *   口径下只对**头**成立 (test_bc 把它分开钉住: 骨干变了 / Q 头没变)。
 */
struct UpdateStat {
    float ce = std::numeric_limits<float>::infinity();  /* 批平均 CE (没有样本时 = inf) */
    int   used = 0;        /* 真正参与更新的样本数 */
    int   targetMiss = 0;  /* **目标不在合法集里**而被丢弃的条数 (见 rl/bc.h 第 2 条) */
    int   skipped = 0;     /* 结构不合格 (缺 state/legalIdx/target) 的条数 */
};

inline UpdateStat update(PPOMCTSAgent &ag, const std::vector<RL::BCSample> &batch, float lr)
{
    /* 批统计的边界: 否则上一批的推理前向会混进 MoE 辅助损失的批均值 */
    ag.ppo.resetMoeBatchStats();
    UpdateStat st;
    const long long missBefore = ag.ppo.bcTargetMisses;
    for (std::size_t i = 0; i < batch.size(); i++) {
        const RL::BCSample &s = batch[i];
        if (!s.valid()) { st.skipped++; continue; }
        /* 形状契约在边界查一次 (见 SAC 那一支的注释) */
        if (s.state.size() != (std::size_t)PPOMCTSAgent::STATE_DIM) { st.skipped++; continue; }
        if (ag.ppo.bcGradSparse(s.state, s.legalIdx, s.targetIdx, s.targetProb)
            != RL::PPO::BcOutcome::TargetMissed) {
            st.used++;
        }
    }
    st.targetMiss = (int)(ag.ppo.bcTargetMisses - missBefore);
    /*
       ============================================================
       [2026-10] 批边界必须走一次 `finalizeMoeBatch()` —— 与在线路径对齐
       ============================================================
       为什么以前没有它 (以及为什么那是个缺口):
         * 在线路径的三条学习路径 (trainStep / learnFromReplay / learnSelfPlay) 都在
           `applyGradients` 之前调 `finalizeMoeBatch()`, 它做两件事:
             ① `accumulateTrainBatch()`: 把本批的 `usageBatch` 记进"训练侧"计数,
                于是 `moeUsageSplit()` 的"训练侧 / 推理侧"归因才是对的;
             ② `applyMoeBiasUpdate()`: 无辅助损失偏置均衡的控制回路
                (`biasGate[i] += rate·sign(mean_load − load_i)`)。
         * BC 走的是另一条 (`bcApplyGradients`), 上面两步**一步都没做** ⇒
           (a) BC 期间的门控**完全没有均衡回路** —— docs/behavior_cloning_2026_10.md §7
               自己把这条列成"已知缺口: BC 期间 MoE 路由可能偏斜, 本轮没量";
           (b) BC 的 actor 前向被算在"推理侧", 所以 BC 期间读 `moeUsageSplit()` 会得到
               一个归因错位的训练侧负载。
       顺序要求 (与在线路径同一份约定): **必须在 `addAuxGradient` 之前** —— 后者读同一批
       统计、并在末尾把它们清零; 也必须在 RMSProp 之前 (`biasGate` 只影响 top-k 选择,
       所以它不改本步的梯度, 但下一步的路由要用它)。
       代价与默认行为: `moeLossFreeBias=false` (PPO 的库默认) 时这里除了
       `accumulateTrainBatch()` 的计数之外**什么都不做**; GUI 默认开着它
       (`chessboard.cpp` 的 PPO_MOE_LOSSFREE), 所以界面那条 BC 从这一版起才真正开始均衡。
    */
    ag.ppo.finalizeMoeBatch();
    if (ag.ppo.bcSampleCount > 0) {
        ag.ppo.bcApplyGradients(lr);
        st.ce = (float)ag.ppo.lastActorLoss;
    }
    return st;
}

template <class SACLike>
UpdateStat update(SACLike &ag, const std::vector<RL::BCSample> &batch, float lr)
{
    UpdateStat st;
    RL::Tensor mask, target;
    /* 与 evaluate 同一条: π/dz 必须先按 ACTION_DIM 分配 (maskedSoftmax 不回扩) */
    RL::Tensor pi((std::size_t)SACLike::ACTION_DIM, 1);
    RL::Tensor dz((std::size_t)SACLike::ACTION_DIM, 1);
    double ceSum = 0.0;
    int n = 0;
    for (std::size_t k = 0; k < batch.size(); k++) {
        const RL::BCSample &s = batch[k];
        if (!s.valid()) { st.skipped++; continue; }
        /*
           形状契约在**边界**(本层)查一次: 网络的第一层是按 STATE_DIM 建的, 喂错长度的
           状态在 Release 下不会断言, 只会读到别的内存 —— 那是 0xC0000005 或者静默学坏。
        */
        if (s.state.size() != (std::size_t)SACLike::STATE_DIM) { st.skipped++; continue; }
        /* 目标必须落在合法集里 —— 落空时损失恒 0 而梯度是 π (不是 0), 详见 rl/bc.h */
        if (!RL::denseMaskAndTarget(SACLike::ACTION_DIM, s, mask, target)) {
            st.targetMiss++;
            continue;
        }

        ag.policy(s.state, mask, pi);
        /* CE 的数值必须在反向之前读 (反向会清掉层的输出缓冲) */
        ceSum += RL::maskedCrossEntropy(pi, target);
        RL::maskedCeLogitGrad(pi, mask, target, dz);

        if (ag.sharedTrunk()) {
            /* 顺序是硬约束: 头的反向必须先做 (它读骨干的缓存输出 h), 然后骨干
               用头算出来的 inputGrad 作为自己的输出梯度 (与 learnBatch 同一写法)。 */
            ag.actorHead.backward(ag.trunk.output(), dz);
            ag.trunk.backward(s.state, ag.actorHead.inputGrad);
        } else {
            ag.actor.backward(s.state, dz);
        }
        n++;
    }
    st.used = n;
    if (n == 0) {
        return st;
    }

    /*
       优化器: 只更新策略路径。
       共享口径下**不能**对 actor 视图再调一次 RMSProp (它与 trunk/actorHead 共享同一批
       层对象 ⇒ 骨干会被更新两遍 = 学习率乘 2); 这也正是 learnBatch 里那段
       "骨干只更新一次"注释的同一个坑。
       decay=0.0 与 SAC 在线路径一致 (PPO 那条是 0.001, 两边各自保持自己的既有口径)。
    */
    if (ag.sharedTrunk()) {
        ag.trunk.RMSProp(lr, 0.9f, 0.0f);
        ag.actorHead.RMSProp(lr, 0.9f, 0.0f);
    } else {
        ag.actor.RMSProp(lr, 0.9f, 0.0f);
    }
    st.ce = (float)(ceSum / (double)n);
    return st;
}

} // namespace BC

#endif // BCAGENT_HPP
