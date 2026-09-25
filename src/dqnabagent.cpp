#include "dqnabagent.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "agentrollout.hpp"
#include "chessstate.h"     /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
#include "rl/loss.h"
#include "rl/sparse_moe.hpp"
#include "rl/util.hpp"
#include "stone.h"

/* ============================================================================
 *  小工具
 * ============================================================================ */

namespace {

/* 杀棋分值: 与 EVAB 同量级, 远离 (-1,1) 的正常评估; 减 ply 让"早杀"优先 */
constexpr double MATE_SCORE = 1000.0;

/*
   网络值的合法区间。奖励是"终局 ±1 + 每步 -0.001 + 至多吃车 0.05", 所以真值 |V| ≲ 1.02;
   而搜索返回的值可能含 ±MATE —— 直接当 TD 目标会把整批目标拉到 ±1000, MSE 一次把头打飞。
   夹到 [-1,1] 就是"必胜/必败的价值就是 ±1"。
*/
inline double clampValue(double v)
{
    if (v > 1.0) { return 1.0; }
    if (v < -1.0) { return -1.0; }
    return v;
}

/*
   初始化缩放 (与 sacazagent.cpp / rl/ppo.cpp 同一套依据):
   `iFcLayer` 默认 U(-1,1), 在 1440 维输入下 pre-activation 的标准差约 sqrt(d/3) ≈ 20,
   Tanh 一上来就饱和、梯度接近 0 (网络"能跑但不学", 见 docs/issues_review.md 的 A 段历史)。
   普通层按 1/sqrt(fan_in) 重缩; 稀疏 MoE 层不是 iFcLayer, 它的专家权重由 SparseMoE
   的构造函数用 scaleExpertInit 缩过 —— 这里不能再缩一遍。
*/
void scaleLayerInit(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer*>(net[i]);
        if (fc == nullptr) {
            continue;
        }
        const float fanIn = (float)(fc->inputDim > 1 ? fc->inputDim : 1);
        const float s = 1.0f / std::sqrt(fanIn);
        for (std::size_t k = 0; k < fc->w.size(); k++) {
            fc->w[k] *= s;
        }
        for (std::size_t k = 0; k < fc->b.size(); k++) {
            fc->b[k] *= s;
        }
    }
}

std::vector<RL::ISparseMoE*> sparseMoeLayers(RL::Net &net)
{
    std::vector<RL::ISparseMoE*> out;
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(net[i]);
        if (m != nullptr) {
            out.push_back(m);
        }
    }
    return out;
}

} // namespace

const char *DQNABAgent::backboneName(Backbone b)
{
    switch (b) {
    case Backbone::SparseMoeTb: return "稀疏MoE(TB专家)";
    case Backbone::Mlp:         return "MLP";
    default:                    return "?";
    }
}

/* ============================================================================
 *  造网 / 构造
 * ============================================================================ */

RL::Net DQNABAgent::buildTrunk(bool withGrad) const
{
    RL::Net::Layers layers;
    if (backbone == Backbone::SparseMoeTb) {
        layers.push_back(std::make_shared<RL::SparseMoE<TBExpert,
                                                         MOE_EXPERTS,
                                                         MOE_TOPK> >(
            STATE_DIM, withGrad, 0));
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, (std::size_t)hiddenDim,
                                                true, withGrad));
    } else {
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, (std::size_t)hiddenDim,
                                                true, withGrad));
        layers.push_back(RL::Layer<RL::Tanh>::_((std::size_t)hiddenDim,
                                                (std::size_t)hiddenDim, true, withGrad));
    }
    RL::Net net(layers);
    scaleLayerInit(net);
    return net;
}

RL::Net DQNABAgent::buildValueHead(bool withGrad) const
{
    RL::Net::Layers layers;
    layers.push_back(RL::Layer<RL::Tanh>::_((std::size_t)hiddenDim,
                                            (std::size_t)hiddenDim, true, withGrad));
    /*
       V 的输出头用 Tanh: V 的真值区间就是 (-1,1), 有界头顺带挡住
       "初始 Q 落在 ±8、比奖励尺度大一个数量级"那类缺陷 (未训练的槽位被 argmax 选中,
       见 docs/issues_review.md 六、缺陷 2.2)。
    */
    layers.push_back(RL::Layer<RL::Tanh>::_((std::size_t)hiddenDim, 1, true, withGrad));
    RL::Net net(layers);
    scaleLayerInit(net);
    return net;
}

RL::Net DQNABAgent::buildAdvantageHead(bool withGrad) const
{
    RL::Net::Layers layers;
    layers.push_back(RL::Layer<RL::Tanh>::_((std::size_t)hiddenDim,
                                            (std::size_t)hiddenDim, true, withGrad));
    /* 优势要能取负、也不该被夹住 -> Linear 输出。它支持 sparseLogits, 于是搜索里
       每个节点只算**合法列**那一小部分 (R1 的捷径: 8100 列里只算 ~44 列)。 */
    layers.push_back(RL::Layer<RL::Linear>::_((std::size_t)hiddenDim,
                                              (std::size_t)ACTION_DIM, true, withGrad));
    RL::Net net(layers);
    scaleLayerInit(net);
    return net;
}

DQNABAgent::DQNABAgent(Chess &chess_, int hiddenDim_, float gamma_, float lr,
                             Backbone backbone_)
    : chess(chess_)
{
    hiddenDim = hiddenDim_ > 0 ? hiddenDim_ : 64;
    gamma = gamma_;
    learningRate = lr;
    /* 骨干必须在造网之前定: 它决定主干的层结构 (TB 专家 vs 普通 MLP) */
    backbone = backbone_;

    m_trunk = buildTrunk(true);
    m_vHead = buildValueHead(true);
    m_aHead = buildAdvantageHead(true);
    m_trunkT = buildTrunk(false);
    m_vHeadT = buildValueHead(false);
    m_aHeadT = buildAdvantageHead(false);
    copyOnlineToTarget();

    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_dV = RL::Tensor(1, 1);
    m_dA = RL::Tensor(ACTION_DIM, 1);
}

void DQNABAgent::copyOnlineToTarget()
{
    /* Net 的赋值是浅拷贝 (共享层指针), 深拷贝只能走 copyTo */
    m_trunk.copyTo(m_trunkT);
    m_vHead.copyTo(m_vHeadT);
    m_aHead.copyTo(m_aHeadT);
}

/* ============================================================================
 *  编码
 * ============================================================================ */

int DQNABAgent::canonicalCell(int x, int y, int color)
{
    /* 轮到黑方时左右镜像 (x -> 9-x): "己方"永远在 x 大的那一侧。
       实现只有一份 (src/chessstate.h) —— 镜像写错是最难查的静默缺陷：
       红黑会变成两个不同的函数, 训练照样跑。 */
    return ChessState::canonicalCell(x, y, color);
}

void DQNABAgent::encodeSparse(int color, std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[i];
        if (s == nullptr || s->alive == false || s->type < 0 || s->type >= 7) {
            continue;
        }
        const int cell = canonicalCell(s->pos.x, s->pos.y, color);
        /* 平面下标与 encodePiecePlanes 同一约定: type*2 + (是己方 ? 0 : 1)
           (以前这里是 type + 7*(对方), 是**同一信息的另一种排列** —— 两套约定并存
            只会让人以为编码不同, 统一到 chessstate.h 那一份) */
        const int plane = s->type * 2 + ((s->color == color) ? 0 : 1);
        cells.push_back((std::uint16_t)(plane * CELLS + cell));
    }
}

double DQNABAgent::materialPhase() const
{
    /* 唯一实现在 src/chessstate.h (所有 agent 共用一份"完备 Markov 状态"的定义) */
    return ChessState::materialPhase(chess);
}

double DQNABAgent::tempoPhase() const
{
    return ChessState::tempoPhase(chess, REWARD_MAX_PLIES);
}

int DQNABAgent::repetitionCount()
{
    return ChessState::repetitionCount(chess);
}

double DQNABAgent::repetitionPhase()
{
    return ChessState::repetitionPhase(chess);
}

double DQNABAgent::halfmovePhase() const
{
    return ChessState::halfmovePhase(chess);
}

double DQNABAgent::checkPhase(int color) const
{
    return ChessState::checkPhase(const_cast<Chess &>(chess), color);
}

void DQNABAgent::encodeStateFor(int color, RL::Tensor &state)
{
    /*
       防御: 调用方传一个**默认构造的空张量**是很容易犯的错 (往空向量里写 [i] 就是越界,
       而且不报错 —— 实测这么做把测试进程直接打崩过一次)。这里按需重建, 代价是一次
       几乎不会发生的分配。
    */
    if (state.size() != (std::size_t)STATE_DIM) {
        state = RL::Tensor(STATE_DIM, 1);
    }
    state.zero();
    /*
       平面布局: 14 个棋子平面 (规范视角) + 5 个规则/阶段平面, 全部由 chessstate.h 里的
       公共实现写出 —— 本 agent 与其它 agent 的区别只剩"平面数量与顺序", 不再各写一份
       镜像/规则口径 (那是最难查的静默缺陷类别)。
    */
    ChessState::encodeComplete(chess, color, &state[0], 0, encodeRulePlanes,
                               REWARD_MAX_PLIES);
}

void DQNABAgent::encodeState(RL::Tensor &state)
{
    encodeStateFor(chess.sideToMove, state);
}

int DQNABAgent::actionIdxOf(const Step &s, int color)
{
    const int from = canonicalCell(s.pos.x, s.pos.y, color);
    const int to = canonicalCell(s.nextPos.x, s.nextPos.y, color);
    return from * CELLS + to;   /* 双射: 90*90 = 8100, 无碰撞 */
}

void DQNABAgent::legalMoves(int color, std::vector<Step*> &steps,
                               std::vector<int> &indices)
{
    chess.sample(color, steps);
    indices.clear();
    indices.reserve(steps.size());
    for (Step *s : steps) {
        indices.push_back(actionIdxOf(*s, color));
    }
}

void DQNABAgent::getLegalActions(int color, std::vector<Step*> &steps,
                                    std::vector<int> &actionIndices,
                                    RL::Tensor &actionMask)
{
    actionMask.zero();
    legalMoves(color, steps, actionIndices);
    for (std::size_t i = 0; i < actionIndices.size(); i++) {
        actionMask[(std::size_t)actionIndices[i]] = 1.0f;
    }
}

float DQNABAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 (与其它 agent 同一口径, 见 stone.h 的 stepReward) */
    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* ============================================================================
 *  节点前向: V + 合法列的 Q (Dueling)
 * ============================================================================ */

void DQNABAgent::evaluateNode(const RL::Tensor &state, const std::vector<int> &legalIdx,
                                double &vOut, std::vector<double> &qOut, bool inference)
{
    const std::size_t n = legalIdx.size();
    qOut.assign(n, 0.0);
    if (n == 0) {
        vOut = 0.0;
        return;
    }

    RL::Net &trunk = const_cast<RL::Net&>((m_selTrunk != nullptr) ? *m_selTrunk : m_trunk);
    RL::Net &vnet = const_cast<RL::Net&>((m_selV != nullptr) ? *m_selV : m_vHead);
    RL::Net &anet = const_cast<RL::Net&>((m_selA != nullptr) ? *m_selA : m_aHead);

    /* 主干: 每个节点**一次** (成本的全部来源) */
    RL::Tensor &h = trunk.forward(state, inference);
    m_leafEvals++;

    /* V 头: 一个标量 */
    RL::Tensor &vt = vnet.forward(h, inference);
    vOut = (double)vt[0];

    /* A 头: 只算合法列 (R1 捷径); 不支持稀疏时回退全量 */
    std::vector<float> aLegal;
    RL::Tensor &ht = anet.forwardTrunk(h, inference);
    if (!anet.sparseLogits(ht, legalIdx, aLegal) || aLegal.size() != n) {
        RL::Tensor &full = anet.forward(h, inference);
        aLegal.assign(n, 0.0f);
        for (std::size_t i = 0; i < n; i++) {
            const int a = legalIdx[i];
            aLegal[i] = (a >= 0 && a < ACTION_DIM) ? full[(std::size_t)a] : 0.0f;
        }
    }

    /*
       Dueling 合并: Q = V + A − mean_legal(A)。
       均值取**合法集**上的均值 (不是全部 8100 槽): 非法槽位的 A 没有任何意义,
       把它们算进基线只会给所有 Q 加一个与局面无关的偏移。
    */
    double meanA = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        meanA += (double)aLegal[i];
    }
    meanA /= (double)n;
    for (std::size_t i = 0; i < n; i++) {
        qOut[i] = vOut + (double)aLegal[i] - meanA;
    }
}

/* ============================================================================
 *  搜索 (Alpha-Beta, 用 Q 当排序先验 + V 当叶子)
 * ============================================================================ */

int DQNABAgent::effectiveBranch(std::size_t legalCount) const
{
    /*
       "开局宽分支浅、残局窄深" 的落地: 分支数随合法着法数自适应。
       宽局面 (开局 40+ 着) 只展开前几个, 窄局面 (残局) 相对展开更多 —— 于是同一个
       `nodeBudget` 下残局自然搜得更深。深度不是写死的常量, 而是预算的函数。
    */
    int b = (int)(legalCount / 8);
    if (b < branchMin) { b = branchMin; }
    if (b > branchMax) { b = branchMax; }
    return b;
}

double DQNABAgent::negamax(int color, int depth, double alpha, double beta, int ply)
{
    m_nodes++;
    if (m_nodes > (long long)nodeBudget) {
        return alpha;   /* 预算耗尽: 软中止 (与 EVAB 的时间中止同一手法) */
    }

    /* ---- 终局: getResult 一次覆盖 将杀 / 困毙 / 吃将 / 重复 / 60 回合判和 ---- */
    const int res = chess.getResult(color);
    if (res != Chess::RESULT_ONGOING) {
        const float oc = outcomeForMover(res, color);
        if (oc > 0.0f) { return MATE_SCORE - (double)ply; }
        if (oc < 0.0f) { return -MATE_SCORE + (double)ply; }
        return 0.0;
    }

    /* ---- 置换表 (同一棵树内复用; 换权重/换局时调用方会 clearSearchState) ---- */
    const unsigned long long key = chess.computeHash();
    if (ttCapacity > 0) {
        auto it = m_tt.find(key);
        if (it != m_tt.end() && it->second.depth >= depth) {
            m_ttHits++;
            const TTEntry &e = it->second;
            if (e.flag == 0) { return e.value; }
            if (e.flag == 1 && e.value >= beta) { return e.value; }
            if (e.flag == 2 && e.value <= alpha) { return e.value; }
        }
    }

    std::vector<Step*> legal;
    std::vector<int> idx;
    legalMoves(color, legal, idx);
    if (legal.empty()) {
        /* 无合法着法 = 被将死/困毙 (getResult 上面已覆盖, 这里只做兜底) */
        return -MATE_SCORE + (double)ply;
    }

    const std::size_t n = legal.size();
    std::vector<double> q;
    double v = 0.0;
    /* 每个节点一次前向: 既给叶子值 (depth==0), 也给排序先验 (Q) */
    encodeStateFor(color, m_stateBuf);
    evaluateNode(m_stateBuf, idx, v, q);

    if (depth <= 0) {
        Steps::instance().put(legal);
        if (leafEval == LeafEval::MaxQ) {
            double best = q[0];
            for (std::size_t i = 1; i < n; i++) {
                if (q[i] > best) { best = q[i]; }
            }
            return best;
        }
        return v;
    }

    /* 按 Q 降序: "AB 拿 Q 当排序先验" 那一条 —— 剪枝效率几乎全靠它 */
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; i++) { order[i] = (int)i; }
    std::sort(order.begin(), order.end(), [&q](int a, int b) {
        return q[(std::size_t)a] > q[(std::size_t)b];
    });

    const int branch = effectiveBranch(n);
    const int colorNext = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    const double alphaOrig = alpha;
    double best = -1e18;
    int searched = 0;

    for (std::size_t k = 0; k < order.size(); k++) {
        if (k >= (std::size_t)branch && searched > 0) {
            break;      /* 选择性展开: 按 Q 先验只展开前 branch 个 */
        }
        Step *s = legal[(std::size_t)order[k]];
        double dummy = 0.0;
        chess.moveForward(s, dummy);
        const double child = negamax(colorNext, depth - 1, -beta, -alpha, ply + 1);
        chess.moveBack(s, dummy);
        const double val = -child;
        searched++;
        if (val > best) { best = val; }
        if (best > alpha) { alpha = best; }
        if (alpha >= beta) { break; }              /* beta 截断 */
        if (m_nodes > (long long)nodeBudget) { break; }
    }

    Steps::instance().put(legal);
    if (searched == 0) {
        best = alpha;
    }

    if (ttCapacity > 0 && searched > 0) {
        TTEntry e;
        e.depth = depth;
        e.value = best;
        e.flag = (best <= alphaOrig) ? 2 : ((best >= beta) ? 1 : 0);
        if ((int)m_tt.size() >= ttCapacity) { m_tt.clear(); }
        m_tt[key] = e;
    }
    return best;
}

void DQNABAgent::clearSearchState()
{
    m_tt.clear();
    m_nodes = 0;
    m_ttHits = 0;
}

Step DQNABAgent::selectMove(int color, float temperature,
                               std::vector<double> *rootValuesOut,
                               std::vector<int> *rootIdxOut)
{
    /* 决策一律用在线网 (目标网只在算训练标签时用) */
    m_selTrunk = nullptr;
    m_selV = nullptr;
    m_selA = nullptr;
    clearSearchState();
    m_leafEvals = 0;

    std::vector<Step*> legal;
    std::vector<int> idx;
    legalMoves(color, legal, idx);
    if (legal.empty()) {
        Steps::instance().put(legal);
        return Step();
    }

    /*
       根节点**全宽 + 全窗口**: 每个合法着法都拿到一个有意义的搜索值 v_a。
         * 根上不做 alpha 截断, 是因为 v_a 还要供 `softmax(v_a/T)` 采样 (自对弈探索)
           与诊断用 —— 被截断的 fail-low 值全都退化成 ≈alpha, 那个分布就废了;
         * 深度由 `nodeBudget` 控制 (迭代加深, 搜不完的那一层不用), 而不是写死。
       内部节点才是标准 alpha-beta + 按 Q 先验的选择性展开。
    */
    const std::size_t n = legal.size();
    std::vector<double> prev(n, 0.0), cur(n, 0.0);
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; i++) { order[i] = (int)i; }

    /* 根节点前向: Q 先验 (第一次排序用) + 根局面价值 */
    encodeStateFor(color, m_stateBuf);
    double rootV = 0.0;
    std::vector<double> rootQ;
    evaluateNode(m_stateBuf, idx, rootV, rootQ);
    std::sort(order.begin(), order.end(), [&rootQ](int a, int b) {
        return rootQ[(std::size_t)a] > rootQ[(std::size_t)b];
    });
    for (std::size_t i = 0; i < n; i++) { prev[i] = rootQ[i]; }

    const int colorNext = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    int completed = 0;

    for (int d = 1; d <= searchDepth; d++) {
        for (std::size_t i = 0; i < n; i++) { cur[i] = prev[i]; }
        bool aborted = false;
        for (std::size_t k = 0; k < order.size(); k++) {
            if (m_nodes >= (long long)nodeBudget) { aborted = true; break; }
            Step *s = legal[(std::size_t)order[k]];
            double dummy = 0.0;
            chess.moveForward(s, dummy);
            const double child = negamax(colorNext, d - 1, -1e18, 1e18, 1);
            chess.moveBack(s, dummy);
            cur[(std::size_t)order[k]] = -child;
        }
        if (aborted) {
            break;      /* 这一层没搜完: 沿用上一层的结果 (上一层的值是完整的) */
        }
        prev = cur;
        completed = d;
        std::sort(order.begin(), order.end(), [&prev](int a, int b) {
            return prev[(std::size_t)a] > prev[(std::size_t)b];
        });
    }

    m_lastDepth = completed;
    m_lastNodes = m_nodes;

    /* 选点: T<=0 贪心; 否则按 softmax(v_a/T) 采样 */
    int bestLocal = 0;
    for (std::size_t i = 1; i < n; i++) {
        if (prev[i] > prev[bestLocal]) { bestLocal = (int)i; }
    }
    int chosen = bestLocal;
    if (temperature > 1e-6f && n > 1) {
        RL::Tensor dist(ACTION_DIM, 1);
        dist.zero();
        double mx = prev[0];
        for (std::size_t i = 1; i < n; i++) { mx = std::max(mx, prev[i]); }
        const double invT = 1.0 / (double)temperature;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            const double e = std::exp((prev[i] - mx) * invT);
            dist[(std::size_t)idx[i]] = (float)e;
            sum += e;
        }
        if (sum > 0.0 && std::isfinite(sum)) {
            const int a = RL::Random::categorical(dist);
            for (std::size_t i = 0; i < n; i++) {
                if (idx[i] == a) { chosen = (int)i; break; }
            }
        }
    }

    m_lastRootValue = prev[(std::size_t)chosen];
    if (rootValuesOut != nullptr) { *rootValuesOut = prev; }
    if (rootIdxOut != nullptr) { *rootIdxOut = idx; }

    const Step out = *legal[(std::size_t)chosen];
    Steps::instance().put(legal);
    return out;
}

Step DQNABAgent::getBestMove(int color)
{
    return selectMove(color, 0.0f);
}

std::string DQNABAgent::getName() const
{
    return std::string("DQN+AB (AB+DuelingDQN, ") + backboneName(backbone) + ")";
}

/* ============================================================================
 *  训练标签 / 梯度 / 优化器
 * ============================================================================ */

double DQNABAgent::plannedLabelFromCurrent(float reward, bool done)
{
    double y = (double)reward;
    if (done) {
        return clampValue(y);
    }
    /*
       棋盘此刻停在 s' (调用方刚落完子)。用**目标网**从 s' 展开 trainPlanDepth 层,
       拿到的值是"**s' 的走子方** (也就是对手) 的价值"; 我们要的是刚才那个走子方的值 -> 取负。
       这就是"AB 当 planning head"落到 TD 目标上的那一句。
    */
    int depth = trainPlanDepth;
    if (depth <= 0) {
        /* depth 0 = 单步 bootstrap: 只取 s' 的 V (一次前向), 不做展开 */
        std::vector<Step*> legal;
        std::vector<int> idx;
        legalMoves(chess.sideToMove, legal, idx);
        double v = 0.0;
        std::vector<double> q;
        if (!idx.empty()) {
            encodeStateFor(chess.sideToMove, m_stateBuf);
            m_selTrunk = &m_trunkT;
            m_selV = &m_vHeadT;
            m_selA = &m_aHeadT;
            evaluateNode(m_stateBuf, idx, v, q);
            m_selTrunk = m_selV = m_selA = nullptr;
        }
        Steps::instance().put(legal);
        return clampValue(y - (double)gamma * v);
    }

    const int savedBudget = nodeBudget;
    const int savedDepth = searchDepth;
    nodeBudget = labelPlanBudget;
    searchDepth = depth;
    m_selTrunk = &m_trunkT;
    m_selV = &m_vHeadT;
    m_selA = &m_aHeadT;
    m_nodes = 0;
    m_tt.clear();
    const double oppValue = negamax(chess.sideToMove, depth, -1e18, 1e18, 1);
    m_selTrunk = m_selV = m_selA = nullptr;
    nodeBudget = savedBudget;
    searchDepth = savedDepth;
    m_nodes = 0;
    m_tt.clear();

    return clampValue(y - (double)gamma * clampValue(oppValue));
}

double DQNABAgent::computeTarget(const Sample &s)
{
    if (targetMode == TargetMode::Planned) {
        /* 标签在采集时就算好了 (见 plannedLabelFromCurrent): 批更新里不做搜索 */
        return clampValue((double)s.label);
    }

    /* ---- OneStepDouble: a* 用在线网选 (带掩码), 值用目标网 (训练时现算) ---- */
    if (s.done || s.nextLegalIdx.empty()) {
        return clampValue((double)s.reward);
    }
    std::vector<int> nl = s.nextLegalIdx;
    /* s' 的走棋方: 与 s 相反 —— 但 target 只需要"Q_target(s', a*)",
       而 Q(s',·) 是 s' 走子方的值, 所以最后要取负换回原来那个走子方。 */
    double vOnline = 0.0, vTarget = 0.0;
    std::vector<double> qOnline, qTarget;
    /* s' 的规范视角 = 对手视角。这里用一个临时棋盘状态重编码: 样本里存的 nextState
       就是"s' 的走子方视角"的编码 (采集时 encodeState 用的就是当时的 sideToMove) ✓ */
    evaluateNode(s.nextState, nl, vOnline, qOnline, true);
    std::size_t bestI = 0;
    for (std::size_t i = 1; i < qOnline.size(); i++) {
        if (qOnline[i] > qOnline[bestI]) { bestI = i; }
    }
    const int aStar = nl[bestI];
    m_selTrunk = &m_trunkT;
    m_selV = &m_vHeadT;
    m_selA = &m_aHeadT;
    evaluateNode(s.nextState, nl, vTarget, qTarget, true);
    m_selTrunk = m_selV = m_selA = nullptr;
    double qStar = 0.0;
    for (std::size_t i = 0; i < nl.size(); i++) {
        if (nl[i] == aStar) { qStar = qTarget[i]; break; }
    }
    return clampValue((double)s.reward - (double)gamma * qStar);
}

/* ============================================================================
 *  手工评估锚: 探针 / gap / 预训练 / 门控
 * ============================================================================
 *
 *  为什么需要这一节 (设计文档 §7.3 的实测数字):
 *    纯 RL 的 TD 信号在"从零开始 + 稀疏奖励 + 3.6 ms/前向"的条件下信息量太低。
 *    bench_dqnab_vs_ab 实测 (随机初始权重, seed 20240901, 对 ABAgent depth 4):
 *        --pure-q                 0 胜 4 负 0 和   (22.0 手/局)
 *        TB  budget 256 (2.5 层)  0 胜 2 负 2 和   (25.0 手/局, 867 ms/手)
 *        MLP budget 4096 (4.4 层) 0 胜 4 负 0 和   (24.0 手/局)
 *    也就是说 **V 越不准, 搜得越深越亏** —— 深搜只是把一个不准的值放大传播。
 *    EVAB 的结论同向 (blend 爬到 1.0 之后 0 胜 6 负, 见 agent_evab.cpp:620)。
 *    所以正确的顺序是: 先把 V 拉到"至少与手工评估同向", 再让它自己往上走。
 *
 *  这一节实现的就是第一级:
 *    1) 手工锚 = evaluate() 的 tanh 压缩 (与 EVAB 的预训练标签**同一口径**);
 *    2) 探针集合 (固定局面) 上的 `netHandGap` 当"V 准不准"的尺子;
 *    3) `pretrainValueFromHand` 做有监督回归 (可选是否连主干一起训);
 *    4) 门控: 每批更新前后各测一次 gap, 明显变差就**回滚** ——
 *       数据侧给不出这种保护, 回放池里只有稀疏的 TD 信号, 而手工锚是稠密先验,
 *       它坏掉会立刻显形。回滚的代价只是白算一次更新。
 */

double DQNABAgent::handEval(int color)
{
    /*
       evaluate() 是**黑方视角** (与 evaluatePositional / positionalScore 同一口径),
       本 agent 的 V 是"走子方视角", 所以红方取负。除以 3.0 再 tanh 是 EVAB 的
       EVAL_SCALE 口径 (agent_evab.h:44), 保持两个 agent 的手工锚可以互换对照。
    */
    const double e = chess.evaluate();
    return std::tanh((color == Stone::COLOR_RED ? -e : e) / HAND_EVAL_SCALE);
}

void DQNABAgent::saveBoard(BoardSnap &s) const
{
    for (std::size_t i = 0; i < s.stones.size(); i++) {
        const Stone *st = chess.m_children[i];
        BoardSnap::S &d = s.stones[i];
        d.id = st->id;
        d.type = st->type;
        d.color = st->color;
        d.alive = st->alive;
        d.value = st->value;
        d.x = st->pos.x;
        d.y = st->pos.y;
    }
    s.sideToMove = chess.sideToMove;
    s.halfMoveClock = chess.halfMoveClock;
    s.history = chess.history;
}

void DQNABAgent::loadBoard(const BoardSnap &s)
{
    /* m_map 必须整体重建: 被吃掉的子曾经占过的格子现在要变成 nullptr */
    chess.m_map.clear();
    for (std::size_t i = 0; i < s.stones.size(); i++) {
        Stone *st = chess.m_children[i];
        const BoardSnap::S &d = s.stones[i];
        st->id = d.id;
        st->type = d.type;
        st->color = d.color;
        st->alive = d.alive;
        st->value = d.value;
        st->pos = Pos(d.x, d.y);
        if (d.alive) {
            chess.m_map[st->pos] = st;
        }
    }
    chess.sideToMove = s.sideToMove;
    chess.halfMoveClock = s.halfMoveClock;
    chess.history = s.history;
}

void DQNABAgent::buildProbeSet(std::vector<Probe> &out, int count, int maxPlies)
{
    out.clear();
    if (count <= 0) {
        return;
    }
    if (maxPlies < 2) {
        maxPlies = 2;
    }
    out.reserve((std::size_t)count);

    BoardSnap snap;
    saveBoard(snap);

    /*
       造数据方式与 EVAB::pretrainFromHandEval (agent_evab.cpp:657) 一致: 随机走子,
       每 maxPlies 步重开一局, 标签取走子方视角的手工评估。
       **唯一的区别是棋盘必须还原** —— 那个函数直接 chess.reset() 并把棋盘留在初始
       局面 (它只被独立的预训练脚本调用), 而本 agent 的训练随时可能在 GUI 里被触发,
       "训练一次顺手把调用方的棋局吃掉"是这类 agent 最典型的静默副作用。
    */
    chess.reset();
    for (int i = 0; (int)out.size() < count && i < count * 8; i++) {
        if (i % maxPlies == 0) {
            chess.reset();
        }
        const int turn = chess.sideToMove;
        if (chess.getResult(turn) != Chess::RESULT_ONGOING) {
            chess.reset();
            continue;
        }
        std::vector<Step*> legal;
        chess.sample(turn, legal);
        if (legal.empty()) {
            chess.reset();
            continue;
        }

        Probe p;
        p.state = RL::Tensor(STATE_DIM, 1);
        encodeStateFor(turn, p.state);
        p.target = (float)handEval(turn);
        out.push_back(std::move(p));

        /*
           选一步走: 以 probeCaptureBias 的概率优先走吃子。
           纯随机走子几乎永远不吃子 (实测 ≤12 手之后手工锚的标准差只有 0.0074),
           造出来的数据标签几乎是常数 —— 那种数据集上"训练"只会学到输出均值,
           看起来 gap 在降, 实际 corr 一动不动。吃子偏置直接把标签的方差做出来。
        */
        Step *mv = nullptr;
        if (probeCaptureBias > 0.0f) {
            std::vector<Step*> caps;
            for (std::size_t k = 0; k < legal.size(); k++) {
                /* 吃子 = 目标格里有人 (与 computeReward 同一判据) */
                if (chess.m_map[legal[k]->nextPos] != nullptr) { caps.push_back(legal[k]); }
            }
            const float u = std::uniform_real_distribution<float>(0.0f, 1.0f)(
                RL::Random::engine);
            if (!caps.empty() && u < probeCaptureBias) {
                mv = caps[(std::size_t)(RL::Random::engine() % caps.size())];
            }
        }
        if (mv == nullptr) {
            mv = legal[(std::size_t)(RL::Random::engine() % legal.size())];
        }
        double d = 0.0;
        chess.moveForward(mv, d);
        Steps::instance().put(legal);   /* moveForward 用完即还 (与 EVAB 同一写法) */
    }
    loadBoard(snap);
}

void DQNABAgent::rebuildProbes(int count, int maxPlies)
{
    buildProbeSet(m_probes, count, maxPlies);
}

DQNABAgent::HandStats DQNABAgent::netHandStats(int samples)
{
    HandStats out;
    if (m_probes.empty()) {
        rebuildProbes(valueGateProbeCount > 0 ? valueGateProbeCount : 12);
    }
    if (m_probes.empty()) {
        return out;
    }
    const int n = (samples > 0) ? std::min(samples, (int)m_probes.size())
                                : (int)m_probes.size();
    out.n = n;
    std::vector<double> vs((std::size_t)n, 0.0);
    double sumA = 0.0, sumV = 0.0, sumAbs = 0.0;
    for (int i = 0; i < n; i++) {
        /*
           纯前向 (withGrad=false): 探针**不许**动梯度缓冲 ——
           否则"用来判断要不要回滚"的测量本身就会改变被测量的那次更新。
       */
        RL::Tensor &h = m_trunk.forward(m_probes[(std::size_t)i].state, false);
        RL::Tensor &v = m_vHead.forward(h, false);
        const double a = (double)m_probes[(std::size_t)i].target;
        vs[(std::size_t)i] = (double)v[0];
        sumA += a;
        sumV += vs[(std::size_t)i];
        sumAbs += std::fabs(vs[(std::size_t)i] - a);
    }
    out.gap = sumAbs / (double)n;
    const double meanA = sumA / (double)n;
    const double meanV = sumV / (double)n;
    double cov = 0.0, varA = 0.0, varV = 0.0;
    for (int i = 0; i < n; i++) {
        const double da = (double)m_probes[(std::size_t)i].target - meanA;
        const double dv = vs[(std::size_t)i] - meanV;
        cov += da * dv;
        varA += da * da;
        varV += dv * dv;
    }
    out.anchorStd = std::sqrt(varA / (double)n);
    out.vStd = std::sqrt(varV / (double)n);
    /*
       相关系数而不是"归一化 MSE": 常数 V (vStd=0) 的 corr 定义为 0 ——
       这正是要的语义 ("什么都没学到"), 而 gap 在这种情况下会虚低。
    */
    out.corr = (varA > 1e-12 && varV > 1e-12)
                   ? cov / std::sqrt(varA * varV) : 0.0;
    return out;
}

double DQNABAgent::netHandGap(int samples)
{
    return netHandStats(samples).gap;
}

void DQNABAgent::snapshotWeights()
{
    if (!m_bakReady) {
        /* 备份网 withGrad=false: 它们只是权重容器, 不该再占一份梯度缓冲 (TB 骨干
           每份梯度缓冲 ≈ 150 MB, 而本 agent 已经同时持有在线网 + 目标网) */
        m_trunkBak = buildTrunk(false);
        m_vHeadBak = buildValueHead(false);
        m_aHeadBak = buildAdvantageHead(false);
        m_bakReady = true;
    }
    m_trunk.copyTo(m_trunkBak);
    m_vHead.copyTo(m_vHeadBak);
    m_aHead.copyTo(m_aHeadBak);
}

void DQNABAgent::restoreWeights()
{
    if (!m_bakReady) {
        return;
    }
    /*
       只回滚**权重**, 不回滚 RMSProp 的二阶矩 (RL::Net::copyTo 只搬 w/b)。
       后果: 回滚后紧接着的那次更新步长会偏小 (二阶矩还带着"上一次坏更新"的记忆)。
       与 EVAB 的回滚同一限制 (agent_evab.cpp:1015 一带), 实测影响是"回滚后一两轮
       内 gap 恢复得慢一点", 不是方向性错误 —— 所以先记下来, 不引入更重的机制。
    */
    m_trunkBak.copyTo(m_trunk);
    m_vHeadBak.copyTo(m_vHead);
    m_aHeadBak.copyTo(m_aHead);
}

double DQNABAgent::pretrainValueFromHand(int positions, int maxPlies, int batchSize,
                                        int epochs, bool pretrainTrunk, bool verbose)
{
    const auto t0 = std::chrono::steady_clock::now();
    m_lastPretrainRollback = false;
    m_pretrainGapBefore = 0.0;
    m_pretrainGapAfter = 0.0;
    if (positions <= 0 || batchSize <= 0 || epochs <= 0) {
        m_pretrainMs = 0.0;
        return netHandGap();
    }

    /*
       尺子先固定:**探针集合与训练集分开造**, 但探针在整个预训练期间不变。
       拿训练集当尺子会得到"自己考自己"的假进步 (那正是"训练损失下降"不能当
       评估指标的原因); 每遍重造探针则会把"换了一批更容易的局面"读成 V 变准。
       预训练用 **64 个**探针 (比门控的 24 个更稳): 它是一次性开销, 而它的结论
       ("留在锚附近"还是"退回原状")会影响后面所有的训练。
    */
    const int probeN = std::max(64, valueGateProbeCount > 0 ? valueGateProbeCount : 24);
    rebuildProbes(probeN, maxPlies);
    m_preStatsBefore = netHandStats(probeN);
    m_pretrainGapBefore = m_preStatsBefore.gap;

    std::vector<Probe> data;
    buildProbeSet(data, positions, maxPlies);
    if (data.empty()) {
        m_preStatsAfter = m_preStatsBefore;
        m_pretrainMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
        return m_pretrainGapBefore;
    }

    snapshotWeights();   /* 训不出好处就整段退回去 (见函数末尾) */

    RL::Tensor pred(1, 1), tgt(1, 1);
    const std::size_t bs = (std::size_t)batchSize;
    for (int e = 0; e < epochs; e++) {
        /* 每遍洗牌: RMSProp 的轨迹与 mini-batch 顺序有关, 固定顺序 = 有偏的下降 */
        std::shuffle(data.begin(), data.end(), RL::Random::engine);
        for (std::size_t b = 0; b < data.size(); b += bs) {
            const std::size_t n = std::min(bs, data.size() - b);
            resetMoeBatchStats();
            m_batchLossSum = 0.0;
            m_batchCount = 0;
            for (std::size_t k = b; k < b + n; k++) {
                const Probe &p = data[k];
                RL::Tensor &h = m_trunk.forward(p.state, false);
                RL::Tensor &vt = m_vHead.forward(h, false);
                const double err = (double)vt[0] - (double)p.target;
                m_batchLossSum += err * err;
                m_batchCount++;
                pred[0] = vt[0];
                tgt[0] = p.target;
                m_dV[0] = RL::Loss::MSE::df(pred, tgt)[0];
                m_vHead.backward(h, m_dV);
                if (pretrainTrunk) {
                    /*
                       主干反传是可选的: 冻结主干时 V 头只能线性读出"随机特征",
                       便宜但上限低 (TB 骨干每个样本 3.6 ms vs 32 ms);
                       连主干一起训才是真正的"把手工评估的能力装进网络"。
                    */
                    m_trunk.backward(p.state, m_vHead.inputGrad);
                }
            }
            if (pretrainTrunk) {
                if (moeAuxCoef > 0.0f) {
                    /* 稀疏 MoE 的负载均衡损失: 缺了它路由会在几轮内坍缩到单专家
                       (与 learnBatch 同一条理由, 见 design §4) */
                    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(m_trunk);
                    for (std::size_t i = 0; i < l.size(); i++) {
                        l[i]->addAuxGradient(moeAuxCoef);
                    }
                }
                m_trunk.RMSProp(learningRate, 0.9f, 0.0f);
            }
            m_vHead.RMSProp(learningRate, 0.9f, 0.0f);
            if (m_batchCount > 0) {
                m_lastLoss = (float)(m_batchLossSum / (double)m_batchCount);
            }
            m_batchLossSum = 0.0;
            m_batchCount = 0;
        }
    }

    m_preStatsAfter = netHandStats(probeN);
    m_preStatsRejected = m_preStatsAfter;   /* 先记下"试成什么样" (回滚会覆盖 after) */
    m_pretrainGapAfter = m_preStatsAfter.gap;
    /*
       回滚判据用 gap (单调、无界、便宜)。**注意它不是"V 变好"的判据**: 常数 V 的 gap
       也很小 (见头文件里那段警告) —— 所以这里同时把 corr 打出来, 由读数字的人判断
       "gap 降了"到底是"学到了"还是"V 塌回 0"。
    */
    if (m_pretrainGapAfter > m_pretrainGapBefore + (double)pretrainTolerance) {
        /*
           连**有监督的稠密标签**都把它带偏了 -> 直接退回原状。
           这不是理论洁癖: 手工锚之外的信号 (后面的 TD 更新) 只会更噪声,
           一个"连回归都退步"的更新方向必须在它污染回放池之前被拦掉。
           返回值取回滚**之后**实测的 gap, 保证"返回的就是当前状态的真实值"。
        */
        restoreWeights();
        m_lastPretrainRollback = true;
        m_preStatsAfter = netHandStats(probeN);
        m_pretrainGapAfter = m_preStatsAfter.gap;
    }
    m_pretrainMs = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    if (verbose) {
        std::printf("[dqnab] 预训练: %d 局面 x %d 遍 (batch %d, 主干%s)\n"
                    "        gap %.4f -> %.4f, corr %.3f -> %.3f "
                    "(锚标准差 %.4f), %s%.0f ms\n",
                    (int)data.size(), epochs, batchSize,
                    pretrainTrunk ? "参与" : "冻结",
                    m_preStatsBefore.gap, m_preStatsAfter.gap,
                    m_preStatsBefore.corr, m_preStatsAfter.corr,
                    m_preStatsBefore.anchorStd,
                    m_lastPretrainRollback ? "[已回滚] " : "", m_pretrainMs);
        if (m_lastPretrainRollback) {
            /* 回滚后 after 就是 before —— 把被拒掉的那一次的数值单独打出来 */
            std::printf("        被拒的那次: gap %.4f, corr %.3f, vStd %.4f "
                        "(样本量不够时会朝这个方向走, 见设计文档 §7.3 第 7 条)\n",
                        m_preStatsRejected.gap, m_preStatsRejected.corr,
                        m_preStatsRejected.vStd);
        }
        std::fflush(stdout);
    }
    return m_pretrainGapAfter;
}

void DQNABAgent::accumulateGrad(const Sample &s)
{
    const std::vector<int> &legal = s.legalIdx;
    if (legal.empty()) {
        return;
    }
    /* 被走的那一步在合法列表里的位置 (找不到就跳过 —— 样本与掩码不一致) */
    int takenI = -1;
    for (std::size_t i = 0; i < legal.size(); i++) {
        if (legal[i] == s.action) { takenI = (int)i; break; }
    }
    if (takenI < 0) {
        return;
    }

    /* ---- 前向: 主干一次 + V 头 + A 头只算合法列 (与搜索同一套代码路径) ---- */
    RL::Net &trunk = m_trunk;   /* 训练只更新在线网 */
    RL::Tensor &h = trunk.forward(s.state, false);
    RL::Tensor &vt = m_vHead.forward(h, false);
    const double v = (double)vt[0];

    std::vector<float> aLegal;
    RL::Tensor &ht = m_aHead.forwardTrunk(h, false);
    if (!m_aHead.sparseLogits(ht, legal, aLegal) || aLegal.size() != legal.size()) {
        return;     /* 头不支持稀疏: 直接跳过这条样本 (绝不静默算错) */
    }
    const std::size_t n = legal.size();
    double meanA = 0.0;
    for (std::size_t i = 0; i < n; i++) { meanA += (double)aLegal[i]; }
    meanA /= (double)n;
    const double qTaken = v + (double)aLegal[(std::size_t)takenI] - meanA;

    /* ---- 目标 (Planned: 采集时算好的标签; OneStepDouble: 现在算) ---- */
    const double y = computeTarget(s);
    const double err = qTaken - y;
    m_batchLossSum += err * err;
    m_batchCount++;

    /* ---- Dueling 双头的解析梯度 ----
       Q_a = V + A_a − (1/n)·Σ_j A_j
       dL/dQ_a = 2(Q_a − y)   (与 Loss::MSE::df 同一导数)
       dL/dV   = dQ
       dL/dA_a = dQ·(1 − 1/n)      dL/dA_j = −dQ/n  (j ≠ a, j 合法)
    */
    RL::Tensor qv(1, 1), yv(1, 1);
    qv[0] = (float)qTaken;
    yv[0] = (float)y;
    const float dQ = RL::Loss::MSE::df(qv, yv)[0];
    m_dV[0] = dQ;
    m_dA.zero();
    const float invN = 1.0f / (float)n;
    for (std::size_t i = 0; i < n; i++) {
        m_dA[(std::size_t)legal[i]] = -dQ * invN;
    }
    m_dA[(std::size_t)s.action] += dQ;

    /* ---- 两个头各自反传, 输入梯度相加后交给主干 ---- */
    m_vHead.backward(h, m_dV);
    m_aHead.backward(h, m_dA);

    RL::Tensor dh = m_vHead.inputGrad;
    for (std::size_t k = 0; k < dh.size() && k < m_aHead.inputGrad.size(); k++) {
        dh[k] += m_aHead.inputGrad[k];
    }
    m_trunk.backward(s.state, dh);
}

void DQNABAgent::resetMoeBatchStats()
{
    if (moeAuxCoef <= 0.0f) {
        return;
    }
    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(m_trunk);
    for (std::size_t i = 0; i < l.size(); i++) {
        l[i]->resetBatchStats();
    }
}

void DQNABAgent::applyGradients(float lr)
{
    /* ---- 稀疏 MoE 的负载均衡辅助损失 (缺了它路由会在几轮内坍缩) ---- */
    if (moeAuxCoef > 0.0f) {
        std::vector<RL::ISparseMoE*> l = sparseMoeLayers(m_trunk);
        for (std::size_t i = 0; i < l.size(); i++) {
            l[i]->addAuxGradient(moeAuxCoef);
        }
    }

    m_trunk.RMSProp(lr, 0.9f, 0.0f);
    m_vHead.RMSProp(lr, 0.9f, 0.0f);
    m_aHead.RMSProp(lr, 0.9f, 0.0f);

    m_learnSteps++;
    if (targetSyncEvery > 0 && (m_learnSteps % targetSyncEvery) == 0) {
        copyOnlineToTarget();   /* 硬拷贝: 不用 tau 极小值的软更新 (那等于事实上冻结) */
    }
    if (m_batchCount > 0) {
        m_lastLoss = (float)(m_batchLossSum / (double)m_batchCount);
    }
    m_batchLossSum = 0.0;
    m_batchCount = 0;
}

bool DQNABAgent::learnBatch(int batchSize_, int epochs)
{
    if (epochs <= 0 || batchSize_ <= 0 || (int)m_replay.size() < batchSize_) {
        return false;
    }
    std::uniform_int_distribution<std::size_t> pick(0, m_replay.size() - 1);

    /*
       门控 (见 "手工评估锚" 一节). 顺序很关键:
         * 更新前的 gap 必须在 resetMoeBatchStats() **之前**测 —— 探针前向同样会写
           稀疏 MoE 的批统计 (sparse_moe.hpp:225 `usageBatch[i]++` 是无条件的),
           排在后面就会把探针的局面掺进负载均衡辅助损失的统计里;
         * 更新后的 gap 在 applyGradients() 之后测 (辅助损失已经加完, 不会再被污染)。
    */
    const int probeN = (valueGateProbeCount > 0) ? valueGateProbeCount : 12;
    const bool gate = valueGateEnabled && gateToleranceNow() > 0.0;
    m_gateRollback = false;
    m_gapBefore = 0.0;
    m_gapAfter = 0.0;
    if (gate) {
        m_gapBefore = netHandGap(probeN);
        snapshotWeights();
    }

    /*
       每个 epoch **重新抽** batchSize 条 (与 RL::PPO::learnFromReplay 同一做法):
       一次更新看到 batchSize×epochs 条经验, 而优化器只在最后调一次。
       ("把同一批重复过几遍"在 clipGrad 下对更新方向毫无影响, 见 rl/sac.cpp 的说明。)
    */
    resetMoeBatchStats();
    m_batchLossSum = 0.0;
    m_batchCount = 0;
    for (int e = 0; e < epochs; e++) {
        for (int b = 0; b < batchSize_; b++) {
            accumulateGrad(m_replay[pick(RL::Random::engine)]);
        }
    }
    applyGradients(learningRate);

    if (gate) {
        m_gapAfter = netHandGap(probeN);
        if (m_gapAfter > m_gapBefore + gateToleranceNow()) {
            restoreWeights();
            /*
               回滚之后目标网可能已经被这次更新同步过 (applyGradients 里有硬拷贝),
               而"在线网退回去了、目标网留在坏值上"会让接下来的 TD 标签持续偏移 ——
               所以回滚时把目标网也拉回与在线网一致 (宁可打乱一次同步节奏)。
            */
            copyOnlineToTarget();
            m_gateRollback = true;
        }
    }
    return true;
}

void DQNABAgent::pushSample(Sample &&s)
{
    m_replay.push_back(std::move(s));
    trimReplay();
}

void DQNABAgent::trimReplay()
{
    while (m_replay.size() > replayCapacity) {
        m_replay.pop_front();
    }
}

/* ============================================================================
 *  自对弈
 * ============================================================================ */

double DQNABAgent::trainSelfPlay(int episodes, int maxMoves, bool verbose,
                                    float tempRoot, float tempFinal)
{
    double lastAvgLoss = std::numeric_limits<double>::quiet_NaN();

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int turn = Stone::COLOR_RED;
        int moves = 0;
        int learned = 0;

        while (moves < maxMoves) {
            if (chess.getResult(turn) != Chess::RESULT_ONGOING) {
                break;
            }
            const float frac = (float)moves / (float)(maxMoves > 1 ? maxMoves : 1);
            const float temp = tempRoot + (tempFinal - tempRoot) * frac;
            const float temperature = (moves < 8) ? temp : 0.0f;   /* 开局探索, 之后贪心 */

            /* ---- 根全宽搜索 + 温度采样 (自对弈用) ---- */
            std::vector<Step*> legalBefore;
            std::vector<int> idxBefore;
            legalMoves(turn, legalBefore, idxBefore);
            const std::size_t nb = legalBefore.size();
            Steps::instance().put(legalBefore);
            if (nb == 0) {
                break;
            }

            const Step chosen = selectMove(turn, temperature);
            if (!chosen.valid) {
                break;
            }

            /* ---- 采集: (s, a, r, s') + 合法集; 落子后立刻算 Planned 标签 ---- */
            Sample s;
            encodeStateFor(turn, s.state);
            s.legalIdx = idxBefore;
            s.action = actionIdxOf(chosen, turn);
            s.reward = computeReward(chosen, turn);

            Step mv = chosen;
            double dummy = 0.0;
            chess.moveForward(&mv, dummy);

            encodeStateFor(chess.sideToMove, s.nextState);
            {
                std::vector<Step*> l2;
                std::vector<int> i2;
                legalMoves(chess.sideToMove, l2, i2);
                s.nextLegalIdx = i2;
                Steps::instance().put(l2);
            }
            const int after = chess.getResult(chess.sideToMove);
            s.done = (after != Chess::RESULT_ONGOING);
            if (s.done) {
                s.reward = outcomeForMover(after, turn);
            }
            /* 标签必须在棋盘停在 s' 时算; Planned 模式下它就是 TD 目标 */
            s.label = (float)plannedLabelFromCurrent(s.reward, s.done);

            pushSample(std::move(s));
            turn = chess.sideToMove;
            moves++;

            if (learnEveryMoves > 0 && (moves % learnEveryMoves) == 0 && batchSize > 0 &&
                (int)m_replay.size() >= batchSize) {
                if (learnBatch(batchSize, replayEpochs)) {
                    learned++;
                    lastAvgLoss = (double)m_lastLoss;
                }
            }
        }

        if (verbose) {
            std::printf("  [DQN+AB] episode %d: %d 手, 池=%zu, 更新=%d, loss=%.4f\n",
                        ep + 1, moves, m_replay.size(), learned, lastAvgLoss);
        }
    }
    return lastAvgLoss;
}

bool DQNABAgent::exploreAndTrain(int color, int rolloutSteps, const OpponentPolicy &opponent)
{
    if (rolloutSteps <= 0) {
        m_exploreInfo = "DQN+AB: 探索步数为 0, 已跳过";
        return false;
    }
    const int before = (int)m_replay.size();
    /* 局部副本: rolloutFromCurrent 会把"真用了几手对手着法"回填到它里面 (P1) */
    OpponentPolicy opp = opponent;

    const int collected = rolloutFromCurrent(
        *this, chess, color, rolloutSteps,
        /* pick: 根节点的 Q 分布采样 (便宜: 一次前向, 不展开) */
        [this](const RL::Tensor &state, int turn) -> int {
            std::vector<Step*> legal;
            std::vector<int> idx;
            legalMoves(turn, legal, idx);
            if (legal.empty()) {
                Steps::instance().put(legal);
                m_pendingLegal.clear();
                return -1;
            }
            m_pendingLegal = idx;
            double v = 0.0;
            std::vector<double> q;
            evaluateNode(state, idx, v, q);
            Steps::instance().put(legal);
            /* softmax(Q/T): 线上探索只要"分布", 不需要展开 (3.6 ms/节点太贵) */
            RL::Tensor dist(ACTION_DIM, 1);
            dist.zero();
            double mx = q.empty() ? 0.0 : q[0];
            for (std::size_t i = 1; i < q.size(); i++) { mx = std::max(mx, q[i]); }
            for (std::size_t i = 0; i < q.size(); i++) {
                dist[(std::size_t)idx[i]] = (float)std::exp((q[i] - mx) / 0.7);
            }
            return RL::Random::categorical(dist);
        },
        /* onTrans: 存一条经验 (标签在这一步用目标网现算, 棋盘此刻停在 s') */
        [this](const Step &chosen, int actionIdx, const RL::Tensor &stateBefore,
               const RL::Tensor &nextState, float reward, bool done) {
            (void)chosen;
            Sample s;
            s.state = stateBefore;
            s.nextState = nextState;
            s.legalIdx = m_pendingLegal;
            s.action = actionIdx;
            s.reward = reward;
            s.done = done;
            {
                std::vector<Step*> l2;
                std::vector<int> i2;
                legalMoves(chess.sideToMove, l2, i2);
                s.nextLegalIdx = i2;
                Steps::instance().put(l2);
            }
            s.label = (float)plannedLabelFromCurrent(reward, done);
            pushSample(std::move(s));
        },
        opp);

    bool trained = false;
    if (batchSize > 0 && (int)m_replay.size() >= batchSize) {
        trained = learnBatch(batchSize, replayEpochs);
    }

    char buf[320];
    std::snprintf(buf, sizeof(buf), "DQN+AB 探索 %d 步 (池 %zu), %s%s",
                  collected, m_replay.size(),
                  trained ? "在线更新 1 次" : "池不足一个批, 未更新",
                  opponentRolloutInfo(opp).c_str());
    m_exploreInfo = buf;
    (void)before;
    return trained;
}

/* ============================================================================
 *  存取 / 诊断
 * ============================================================================ */

bool DQNABAgent::saveModel(const std::string &prefix)
{
    const int a = m_trunk.save(prefix + "_trunk");
    const int b = m_vHead.save(prefix + "_v");
    const int c = m_aHead.save(prefix + "_a");
    return a == 0 && b == 0 && c == 0;
}

bool DQNABAgent::loadModel(const std::string &prefix)
{
    if (m_trunk.load(prefix + "_trunk") != 0) { return false; }
    if (m_vHead.load(prefix + "_v") != 0) { return false; }
    if (m_aHead.load(prefix + "_a") != 0) { return false; }
    copyOnlineToTarget();
    /* 换了权重就把搜索缓存丢掉 (树/置换表里的值都是旧网络算的) */
    clearSearchState();
    return true;
}

int DQNABAgent::moeExpertCount() const
{
    RL::Net &self = const_cast<RL::Net&>(m_trunk);
    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(self);
    return l.empty() ? 0 : l[0]->expertCount();
}

int DQNABAgent::moeTopK() const
{
    RL::Net &self = const_cast<RL::Net&>(m_trunk);
    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(self);
    return l.empty() ? 0 : l[0]->topK();
}

void DQNABAgent::moeUsage(std::vector<long long> &out) const
{
    out.clear();
    const int e = moeExpertCount();
    if (e <= 0) {
        return;
    }
    out.assign((std::size_t)e, 0);
    std::vector<long long> one;
    RL::Net &self = const_cast<RL::Net&>(m_trunk);
    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(self);
    for (std::size_t i = 0; i < l.size(); i++) {
        l[i]->usageSnapshot(one);
        for (std::size_t k = 0; k < one.size() && k < out.size(); k++) {
            out[k] += one[k];
        }
    }
}

void DQNABAgent::resetMoeUsage()
{
    RL::Net &self = const_cast<RL::Net&>(m_trunk);
    std::vector<RL::ISparseMoE*> l = sparseMoeLayers(self);
    for (std::size_t i = 0; i < l.size(); i++) {
        l[i]->resetUsage();
    }
}

/* ============================================================================
 *  自检报告 (界面"模型自检"面板的数据源)
 * ============================================================================ */

/* ------------------------------------------------------------------
 *  selfCheckReport
 *
 *  为什么要有这些读数: 面板要回答的是"这个模型**值不值得继续训**", 而损失曲线与
 *  自对弈胜率都回答不了 —— 损失只说明网络与自己的目标一致 (实测 DQN+MCTS 报 22、
 *  PPO 报 0.003, 两个数量纲不同、都不可比), 自对弈的赢家和输家是**同一份权重**。
 *  本 agent 的前置判据落在前两处 (表示层与值门控), 第三处是规格/成本读数。
 *
 *    (1) **表示层**: 状态 1710 维 = 19 平面 x 90 格, 5 个规则/阶段平面 (剩余子力 /
 *        总手数 / 无吃子进度 / 重复次数 / 被将) 直接进状态, 于是 V(s) 真的是 s 的
 *        函数、Bellman 备份成立。老编码 (DQN/PG/DQN+MCTS/SACAZ 的 90 维哈希) 里
 *        走子方 / 重复进度 / 无吃子 / 被将**逐字节不可分** (probe_dqnmcts_aliasing [2]),
 *        而"三次重复判和"这类规则决定的正是终局与回报。动作侧同理: `规范from*90 +
 *        规范to` 是 8100 上的双射, 老编码的 128 槽哈希平均把 22 个着法挤进一槽 ——
 *        互不相同的着法拿不到各自的 Q 列。这两条是"模型值不值得继续训"的**前置**
 *        判据 (表示错了, 再训也不会好), 所以它们在报告的最前面。
 *
 *    (2) **值门控**: 每次批更新前后各测一次 `netHandGap` (固定探针上的 |V − 手工锚|),
 *        单次更新把它顶高超过容忍度就回滚权重 (见头文件那一节)。这里报上一条读数:
 *        gap 前 -> 后、有没有回滚, 以及探针上的完整 HandStats。**只看 gap 会骗人**:
 *        tanh(evaluate()/3) 的典型幅度只有 ±0.2, 于是"V 恒等于 0"的未训练网络也能拿到
 *        好看的数字 (TB 骨干随机初始化实测 0.0410) —— 而 corr 对常数 V 恒为 0,
 *        是尺度无关的判据; 锚自身的标准差则告诉你**这把尺子还剩多少信号**。
 *
 *    (3) **搜索/训练规格**: 节点预算是真正的成本控制量, 单位是**网络前向次数**
 *        (不是 MCTS 的模拟次数) —— GUI 每步传 `DQNAB_NODES=256`
 *        (src/chessboard.cpp:141, TB 骨干约 0.8 s/步), 深度只是这个预算的函数。
 *        连同回放池/超参/参数个数一起报, 是为了让"曲线好看"与"规格"能被分开读。
 *
 *  只读契约 (aiagent.h 的口径):
 *    * 全程**不碰 `this->chess`** —— 面板会在对局中途被 GUI 线程调用, 而搜索线程
 *      正在用它 moveForward/moveBack。动作双射那一段用**新构造的**标准开局棋盘,
 *      于是那份读数与本实例的棋局无关, 每次打开面板都是同一个确定性结果 (可以当
 *      回归指示器用)。
 *    * **不做任何前向**: TB 骨干一次前向 3.6 ms (docs/agents_design.md §18.1),
 *      而本函数每手"探索+预训练"之后都会被调一次; 所以 HandStats 一律读**已有缓存**
 *      (netHandStats 不是 const, 也不该在这里跑)。
 *    * 不改任何成员 (函数是 const), 借用 `Steps` 池的 Step* 全部还回去。
 *
 *  刻意**不**在这里报棋力: 棋力只有 bench_anchor 那种带 95% 区间的锚点对局能回答。
 * ------------------------------------------------------------------ */
std::string DQNABAgent::selfCheckReport() const
{
    char buf[640];
    std::string out;

    /* ---- 1. 表示: 平面布局 + 规则上下文是否真的可观测 ---- */
    std::snprintf(buf, sizeof(buf),
                  "状态 %d 维 = %d 平面 x %d 格: 14 棋子平面(规范视角) + [%d]剩余子力 "
                  "+ [%d]总手数 + [%d]无吃子 + [%d]重复 + [%d]被将\n",
                  STATE_DIM, PLANES, CELLS, PLANE_MATERIAL, PLANE_TEMPO,
                  PLANE_HALFMOVE, PLANE_REPEAT, PLANE_CHECK);
    out += buf;

    /* 这一行是本 agent 与老 agent 在**表示层**上最重要的区别, 值得单独一行 */
    out += "规则上下文: 可观测 (上一行那 5 个通道都在状态里) —— 本 agent 用的是"
           "**完备 Markov 状态**那一份编码; DQN/PG/DQN+MCTS/SACAZ 的 90 维哈希编码里"
           "一个都没有 (逐字节不可分)\n";

    std::snprintf(buf, sizeof(buf),
                  "规则平面开关: %s (encodeRulePlanes=%s) —— 关掉只是把 5 个平面清零: "
                  "输入维度与随机初始化都不变, 所以是干净的消融\n",
                  encodeRulePlanes ? "开" : "关",
                  encodeRulePlanes ? "true" : "false");
    out += buf;

    /*
       动作双射的确定性验证。用**新构造**的标准开局棋盘, 完全不碰 this->chess。
       合法着法数在运行期数出来 (写死一个数字迟早与走法生成脱节)。
    */
    int openLegal = 0, openIdx = 0;
    {
        Chess probe;
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        std::vector<int> idx;
        idx.reserve(legal.size());
        for (std::size_t i = 0; i < legal.size(); i++) {
            idx.push_back(actionIdxOf(*legal[i], probe.sideToMove));
        }
        openLegal = (int)legal.size();
        Steps::instance().put(legal);   /* Step* 借自线程本地池, 必须还回去 */
        std::sort(idx.begin(), idx.end());
        openIdx = (int)(std::unique(idx.begin(), idx.end()) - idx.begin());
    }
    if (openLegal == openIdx) {
        std::snprintf(buf, sizeof(buf),
                      "动作编码: 双射 (开局 %d 个合法着法 -> %d 个动作下标, 无别名)\n",
                      openLegal, openIdx);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "动作编码: **非双射** (开局 %d 个合法着法只映射到 %d 个动作下标, "
                      "撞掉 %d 个 —— 这些着法拿不到各自的 Q 列)\n",
                      openLegal, openIdx, openLegal - openIdx);
    }
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "  下标 = 规范 from*%d + 规范 to, 全空间 ACTION_DIM = %d x %d = %d; "
                  "镜像 x->9-x 是单射, 所以红黑两边都成立 (老编码的 128 槽哈希: 开局 44 个"
                  "合法着法就只落在 38 个槽位上, 挤掉 6 个 —— 见 DQN+MCTS 的自检)\n",
                  CELLS, CELLS, CELLS, ACTION_DIM);
    out += buf;

    /* ---- 2. 搜索 / 规划规格 (真正的成本控制量是节点预算, 不是深度) ---- */
    std::snprintf(buf, sizeof(buf),
                  "搜索/规划: 深度上限 %d | 节点预算 %d 节点 (= 网络前向次数, 不是模拟次数; "
                  "GUI 每步传 DQNAB_NODES=256, src/chessboard.cpp:141) | 选择性分支 %d..%d "
                  "(按合法着法数/8 自适应)\n",
                  searchDepth, nodeBudget, branchMin, branchMax);
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  标签/表格: trainPlanDepth %d 层 | labelPlanBudget %d 节点 | "
                  "TT 容量 %d | 叶子口径=%s | 目标口径=%s\n",
                  trainPlanDepth, labelPlanBudget, ttCapacity,
                  (leafEval == LeafEval::MaxQ) ? "maxQ" : "V",
                  (targetMode == TargetMode::OneStepDouble) ? "OneStepDouble" : "Planned");
    out += buf;

    if (m_lastNodes > 0) {
        std::snprintf(buf, sizeof(buf),
                      "  上次搜索: 深度 %d | 节点 %lld | TT 命中 %lld | 叶子前向 %lld "
                      "(自上次 selectMove 起, 含标签展开) | 根值 %.4f\n",
                      m_lastDepth, m_lastNodes, m_ttHits, m_leafEvals, lastRootValue());
        out += buf;
    } else {
        out += "  上次搜索: 没有记录 (这个实例还没跑过 selectMove)\n";
    }

    /* ---- 3. 值门控 (本 agent 的 headline 安全机制) ---- */
    char tolNote[64];
    tolNote[0] = '\0';
    if (std::fabs(gateToleranceNow() - (double)valueGateTolerance) > 1e-9) {
        /* 阈值按 lr 放大过 (见头文件): 不报出来, "容差 0.25"就不是实际判据 */
        std::snprintf(tolNote, sizeof(tolNote), ", lr 放大后 %.3f", gateToleranceNow());
    }
    if (!valueGateEnabled) {
        std::snprintf(buf, sizeof(buf),
                      "值门控: **关** (容差 %.2f%s) —— 批更新不测 gap、不回滚: "
                      "毒化更新一次就能把 gap 从 0.03 顶到 0.63 (test_dqnab [9] 的对照段)\n",
                      (double)valueGateTolerance, tolNote);
    } else if (learnSteps() <= 0) {
        std::snprintf(buf, sizeof(buf),
                      "值门控: 开 (容差 %.2f%s) | 还没有批更新, gap 前/后未测\n",
                      (double)valueGateTolerance, tolNote);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "值门控: 开 (容差 %.2f%s) | 上次更新 gap %.4f -> %.4f, 回滚=%s%s\n",
                      (double)valueGateTolerance, tolNote,
                      lastGapBefore(), lastGapAfter(),
                      lastUpdateRolledBack() ? "是" : "否",
                      lastUpdateRolledBack() ? " (这次更新被判为毒化, 权重已回滚)" : "");
    }
    out += buf;

    out += "  语义: \"**单次**更新不许把 V 打飞\", 不是\"V 不许离开手工锚\" (否则 V 永远"
           "不可能变得比手工锚更好); 阈值按 lr 缩放, 因为 RMSProp 把每个张量的位移"
           "整成 ≈ lr —— 固定 0.25 在 lr=0.02 时曾把 60 次正常更新回滚掉 59 次\n";

    if (probeCount() <= 0) {
        out += "手工锚: 还没有探针 (rebuildProbes 未跑)\n";
    } else {
        const HandStats &ha = pretrainStatsAfter();
        if (ha.n > 0) {
            std::snprintf(buf, sizeof(buf),
                          "手工锚 (上次预训练测得: %d 个探针, 门控每次用 %d): |V-锚| %.4f | "
                          "corr %.3f | 锚标准差 %.4f | V 标准差 %.4f\n",
                          ha.n, valueGateProbeCount, ha.gap, ha.corr, ha.anchorStd, ha.vStd);
            out += buf;
            out += "  corr 才是尺度无关判据 (常数 V 恒为 0); gap 会因 V 恰好接近 0 而虚低 "
                   "—— 随机初始化 TB 实测 0.0410, 其实什么都没学到\n";
            std::snprintf(buf, sizeof(buf),
                          "  锚标准差 %.4f = 这把尺子带的信号量: 它接近 0 时\"gap 降了\"是"
                          "在常数上判的 (纯随机、不吃子的探针实测只有 0.0074; 现在 "
                          "probeCaptureBias %.2f)\n",
                          ha.anchorStd, (double)probeCaptureBias);
            out += buf;
        } else if (learnSteps() > 0) {
            std::snprintf(buf, sizeof(buf),
                          "手工锚 (缓存 %d 个探针, 门控每次用 %d): 只有门控测过 gap "
                          "(上次 %.4f -> %.4f); corr/锚标准差要跑过一次 "
                          "pretrainValueFromHand 才有\n",
                          probeCount(), valueGateProbeCount,
                          lastGapBefore(), lastGapAfter());
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "手工锚 (缓存 %d 个探针): 还没测过 (既没有批更新也没有预训练)\n",
                          probeCount());
            out += buf;
        }
    }

    if (pretrainStatsBefore().n > 0) {
        std::snprintf(buf, sizeof(buf),
                      "预训练 (手工锚当稠密标签): gap %.4f -> %.4f | 回滚=%s | 容忍度 %.3f "
                      "| 耗时 %.0f ms\n",
                      lastPretrainGapBefore(), pretrainStatsAfter().gap,
                      lastPretrainRolledBack() ? "是" : "否",
                      (double)pretrainTolerance, lastPretrainMs());
        out += buf;
        if (lastPretrainRolledBack() && pretrainStatsRejected().n > 0) {
            /* after 已被回滚复原成 before, "它到底试成什么样"只剩这个数能回答 */
            const HandStats &hr = pretrainStatsRejected();
            std::snprintf(buf, sizeof(buf),
                          "  被回滚前那一次的尝试: |V-锚| %.4f | corr %.3f | V 标准差 "
                          "%.4f —— 这往往是\"样本量不够\", 不是\"网络不行\"\n",
                          hr.gap, hr.corr, hr.vStd);
            out += buf;
        }
    } else {
        out += "预训练: 还没有跑过 (手工锚 gap 的 before/after 只在跑过之后才有)\n";
    }

    /* ---- 4. 训练 / 回放规格 ---- */
    std::snprintf(buf, sizeof(buf),
                  "训练/回放: 回放池 %zu/%zu 条 | 已批更新 %d 次 | batch %d x %d 遍 | "
                  "每 %d 手学一次 | 目标网每 %d 次硬拷贝\n",
                  replaySize(), replayCapacity, learnSteps(), batchSize, replayEpochs,
                  learnEveryMoves, targetSyncEvery);
    out += buf;

    std::snprintf(buf, sizeof(buf),
                  "  超参: gamma %.3f | lr %.5f | MoE 辅助系数 %.2f\n",
                  (double)gamma, (double)learningRate, (double)moeAuxCoef);
    out += buf;

    /*
       这两个成员在本工程里**只声明、没有任何读取点** (线上探索 exploreAndTrain
       写死 /0.7, trainSelfPlay 用调用方传的 tempRoot/tempFinal) —— 面板把"配置里
       有这个旋钮"和"它真的在起作用"分开写, 免得读成"温度退火正在被使用"。
    */
    std::snprintf(buf, sizeof(buf),
                  "  探索温度成员: 根 %.2f -> 终 %.2f —— 注意这两个成员在本工程里"
                  "**没有读取点** (exploreAndTrain 写死 0.7, trainSelfPlay 用传参)\n",
                  (double)exploringTempRoot, (double)exploringTempFinal);
    out += buf;

    const int experts = moeExpertCount();
    if (experts > 0) {
        std::vector<long long> usage;
        moeUsage(usage);
        long long mx = 0, sum = 0;
        for (std::size_t i = 0; i < usage.size(); i++) {
            mx = std::max(mx, usage[i]);
            sum += usage[i];
        }
        if (sum > 0) {
            const double mean = (double)sum / (double)usage.size();
            const double ratio = (mean > 0.0) ? (double)mx / mean : 0.0;
            /* 路由偏斜是**规格**读数: 坍缩到少数专家时"专家数"等于白给 (靠辅助损失拉) */
            std::snprintf(buf, sizeof(buf),
                          "  MoE: %d 专家 topK=%d | 路由累计 最大 %lld / 均值 %.1f = %.2fx%s\n",
                          experts, moeTopK(), mx, mean, ratio,
                          (ratio > 2.0) ? " (偏斜: 靠 moeAuxCoef 的均衡损失往回拉)" : "");
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "  MoE: %d 专家 topK=%d | 还没有前向统计 (usage 全 0)\n",
                          experts, moeTopK());
            out += buf;
        }
    } else {
        out += "  MoE: 无 (MLP 骨干: 没有路由, moeAuxCoef 不起作用)\n";
    }

    std::snprintf(buf, sizeof(buf),
                  "  容量: 主干 %s / hidden %d = %lld 参数 + 双头 %lld 参数 "
                  "(参数多只说明容量, 与棋力无关)\n",
                  backboneName(backbone), hiddenDim,
                  trunkParamCount(), headParamCount());
    out += buf;

    const float loss = getLastTrainLoss();
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "  最近训练损失: %.5f (TD 的 MSE: 只说明网络与自己的目标一致; "
                      "与别的 agent 量纲不同、不可比, 更不是棋力)\n",
                      (double)loss);
    } else {
        out += "  最近训练损失: 未上报 (NaN: 还没做过一次批更新)\n";
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局 "
           "(Elo 差带 95% 区间)\n";
    return out;
}
