#include "sacazagent.h"

#include "chessstate.h"   /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
#include "agentrollout.hpp"
#include "rl/sparse_moe.hpp"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>

/* ============================================================
 *  构造: 三个网络
 *
 *  策略网: 1260 -> h -> h -> 128 **logits** (输出层不带激活)
 *    掩码 softmax 在 agent 内自己做 (见 maskedSoftmax), 所以这里输出 logits 而不是
 *    概率 —— 否则"掩码后归一化"的雅可比与 `Layer<Softmax>` 内部缓存的 o 对不上。
 *
 *  双 critic: 1260 -> h -> h -> 128 个 Q 值, 输出层同样是 `Layer<Linear>`。
 *    **刻意不用 Sigmoid 头**: 象棋奖励含负值 (输棋 −1, 丢子为负), 而
 *    `Layer<Sigmoid>` 值域是 (0,1), 结构上无法表示负 Q —— 这正是
 *    docs/issues_review.md 里 B18 记的问题 (`dqn.cpp` 至今仍用 Sigmoid 头)。
 *
 *  目标网: 只前向, withGrad=false。
 * ============================================================ */
namespace {

/*
   初始化缩放。`iFcLayer` 的构造函数把权重初始化成 U(-1,1); 对 1260 维输入来说
   pre-activation 的标准差约 sqrt(1260/3) ≈ 20 —— Tanh 一上来就饱和, 梯度接近 0,
   网络基本学不动。这里按 1/sqrt(fan_in) 再缩放一遍, 把 pre-activation 的标准差
   拉回 ~0.6。隐藏层是标准做法; 输出层顺带让初始 Q ≈ 0、初始策略接近均匀, 对探索有利。
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

/* 找出网络里第一个稀疏 MoE 层 (没有就返回 nullptr) */
RL::ISparseMoE *findSparseMoe(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(net[i]);
        if (m != nullptr) {
            return m;
        }
    }
    return nullptr;
}

} // namespace

const char *SACAZAgent::backboneName(Backbone b)
{
    switch (b) {
    case Backbone::Mlp:          return "MLP";
    case Backbone::SparseMoeMlp: return "稀疏MoE(MLP专家)";
    case Backbone::SparseMoeTb:  return "稀疏MoE(TB专家)";
    case Backbone::DenseMoeTb:   return "稠密MoE(TB专家,对照)";
    default:                     return "?";
    }
}

/*
   按 backbone 造网络。
   输出层始终是 `Layer<Linear>` (不是 Sigmoid): 象棋奖励含负值, Q 必须能取负
   (docs/issues_review.md B18)。掩码 softmax 在 agent 里自己做, 所以策略头输出
   **logits**, 不是概率。
   两个结构上的注意点:
     * 稀疏 MoE 层是"同维进出"的 (专家的输入输出必须同维才能做门控加权和),
       所以后面必须再接一层 Tanh(d_model -> h) 把 1260 维压到 h 维, 再进 Linear 头。
     * 专家权重由 SparseMoE 的构造函数调用 scaleExpertInit 缩放; 其余普通层由
       scaleLayerInit 缩放 (两者的依据都是 1/sqrt(fan_in))。
*/
RL::Net SACAZAgent::buildNet(bool withGrad) const
{
    const std::size_t h = (std::size_t)(hiddenDim > 0 ? hiddenDim : 64);
    RL::Net::Layers layers;

    switch (backbone) {
    case Backbone::Mlp:
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        layers.push_back(RL::Layer<RL::Tanh>::_(h, h, true, withGrad));
        break;
    case Backbone::SparseMoeMlp:
        layers.push_back(std::make_shared<RL::SparseMoE<RL::MlpExpert,
                                                        MOE_MLP_EXPERTS,
                                                        MOE_MLP_TOPK> >(
            STATE_DIM, withGrad, expertHidden > 0 ? expertHidden : 64));
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    case Backbone::SparseMoeTb:
        layers.push_back(std::make_shared<RL::SparseMoE<RL::TransformerBlock<MOE_TB_HEADS, MOE_TB_DFF>,
                                                        MOE_TB_EXPERTS,
                                                        MOE_TB_TOPK> >(STATE_DIM, withGrad, 0));
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    case Backbone::DenseMoeTb:
        /* TopK == NumExperts => 门控照旧, 但四个专家全算 (等参数不等算力的对照) */
        layers.push_back(std::make_shared<RL::SparseMoE<RL::TransformerBlock<MOE_TB_HEADS, MOE_TB_DFF>,
                                                        MOE_TB_EXPERTS,
                                                        MOE_TB_EXPERTS> >(STATE_DIM, withGrad, 0));
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    default:
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        layers.push_back(RL::Layer<RL::Tanh>::_(h, h, true, withGrad));
        break;
    }

    layers.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, withGrad));
    RL::Net net(layers);
    /*
       只缩放普通的 iFcLayer。稀疏 MoE 层本身不是 iFcLayer (dynamic_cast 会返回
       nullptr), 它的专家权重已经在 SparseMoE 的构造函数里用 scaleExpertInit 缩过了。
       (这里若重复缩放同一个层会把初始化标准差再压一次, 所以每层只走一次。)
    */
    scaleLayerInit(net);
    return net;
}

int SACAZAgent::moeExpertCount() const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->expertCount() : 0;
}

int SACAZAgent::moeTopK() const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->topK() : 0;
}

void SACAZAgent::moeUsage(std::vector<long long> &out) const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    if (m == nullptr) {
        out.clear();
        return;
    }
    m->usageSnapshot(out);
}

void SACAZAgent::resetMoeUsage()
{
    for (std::size_t i = 0; i < actor.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(actor[i]);
        if (m != nullptr) {
            m->resetUsage();
        }
    }
}

SACAZAgent::SACAZAgent(Chess &chess_,
                       int hiddenDim_,
                       float gamma_,
                       float lr,
                       float cpuct,
                       Backbone backbone_,
                       int expertHidden_,
                       float auxLossCoef_)
    : chess(chess_),
      backbone(backbone_),
      expertHidden(expertHidden_ > 0 ? expertHidden_ : 64),
      auxLossCoef(auxLossCoef_),
      hiddenDim(hiddenDim_ > 0 ? hiddenDim_ : 64),
      gamma(gamma_),
      learningRateActor(lr),
      learningRateCritic(lr),
      learningRateAlpha(1e-3f),
      c_puct(cpuct),
      azWeight(1.0f),
      entropyRatio(0.98f),
      simulations(64),
      batchSize(32),
      replaceTargetIter(64),
      maxMemorySize(4096),
      totalEpisodes(0),
      learnSteps(0),
      m_leafEvals(0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;

    actor = buildNet(true);
    q1 = buildNet(true);
    q2 = buildNet(true);
    q1Target = buildNet(false);
    q2Target = buildNet(false);

    /*
       目标网构建时 withGrad=false, 但参数仍然是随机初始化的 —— 必须从在线网拷一份
       过去, 否则目标网从"另一个随机点"开始, 自举项一开始就是纯噪声。
       (Net 的拷贝是**浅拷贝**: 共享层指针; 深拷贝必须走 copyTo。)
    */
    q1.copyTo(q1Target);
    q2.copyTo(q2Target);

    /* 温度 α: 标量, 自动调节 (SAC-Discrete 的做法) */
    alpha = RL::GradValue(1, 1);
    alpha[0] = 0.2f;

    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_logits = RL::Tensor(ACTION_DIM, 1);
    m_q1 = RL::Tensor(ACTION_DIM, 1);
    m_q2 = RL::Tensor(ACTION_DIM, 1);
}

std::string SACAZAgent::getName() const
{
    return "SAC+MCTS+AlphaZero (最大熵搜索)";
}

/* ============================================================
 *  状态编码: 规范视角 14x90 + 3 个规则上下文槽 (与 EVABAgent 同一约定)
 * ============================================================ */
void SACAZAgent::encodeSparse(int color, std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    const bool redToMove = (color == Stone::COLOR_RED);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[i];
        if (s == nullptr || s->alive == false || s->type < 0 || s->type >= 7) {
            continue;
        }
        /* 轮到黑方时左右镜像 (x -> 9-x): "己方"永远在 x 大的那一侧。
           镜像只有一份实现 (src/chessstate.h), 这里调它。 */
        const int cell = ChessState::canonicalCell(s->pos.x, s->pos.y, color);
        /* 平面下标 = type*2 + (是己方?0:1), 与棋盘状态公共约定一致 */
        const int plane = s->type * 2 + ((s->color == color) ? 0 : 1);
        cells.push_back((std::uint16_t)(plane * CELLS + cell));
    }
    (void)redToMove;
}

void SACAZAgent::contextOf(Chess &c, int color, float out[3])
{
    /* 规则上下文: 无吃子进度 / 重复次数 / 将军 —— 数值口径取 chessstate.h 那一份 */
    out[0] = (float)ChessState::halfmovePhase(c);
    out[1] = (float)ChessState::repetitionPhase(c);
    out[2] = (float)ChessState::checkPhase(c, color);
}

void SACAZAgent::writeContext(RL::Tensor &state, const float ctx[3])
{
    if (state.size() < (std::size_t)STATE_DIM) {
        return;
    }
    for (int i = 0; i < 3; i++) {
        state[(std::size_t)(CTX_BASE + i)] = ctx[i];
    }
}

void SACAZAgent::readContext(const RL::Tensor &state, float out[3])
{
    for (int i = 0; i < 3; i++) {
        out[i] = 0.0f;
        if (state.size() >= (std::size_t)STATE_DIM) {
            out[i] = state[(std::size_t)(CTX_BASE + i)];
        }
    }
}

void SACAZAgent::expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state)
{
    state.zero();
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

void SACAZAgent::denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells)
{
    cells.clear();
    cells.reserve(32);
    /* 只取**棋子平面**里非零的格: 规则上下文那 3 个槽不是"格子", 由 ctx[] 单独带 */
    const std::size_t boardLimit = (state.size() < (std::size_t)CTX_BASE)
                                       ? state.size() : (std::size_t)CTX_BASE;
    for (std::size_t i = 0; i < boardLimit; i++) {
        if (state[i] != 0.0f) {
            cells.push_back((std::uint16_t)i);
        }
    }
}

void SACAZAgent::encodeStateFor(int color, RL::Tensor &state)
{
    if (state.size() != (std::size_t)STATE_DIM) {
        state = RL::Tensor(STATE_DIM, 1);
    }
    std::vector<std::uint16_t> cells;
    encodeSparse(color, cells);
    expandSparse(cells, state);
    float ctx[3];
    contextOf(chess, color, ctx);
    writeContext(state, ctx);
}

void SACAZAgent::encodeState(RL::Tensor &state)
{
    /*
       视角由棋盘当前的 sideToMove 决定 —— MCTS 里走法是真正落在棋盘上的
       (`moveForward` 会翻转 sideToMove), 所以这里读到的就是该节点的走棋方。
       `rolloutFromCurrent` 进来时也会把 sideToMove 设成 color, 探索阶段同样正确。
    */
    encodeStateFor(chess.sideToMove, state);
}

/* ============================================================
 *  动作
 * ============================================================ */
int SACAZAgent::stepToActionIdx(const Step &s)
{
    /* 与 PG/DQN/PPOMCTS 相同的确定性哈希 (碰撞的局限见头文件) */
    const unsigned long long h = (unsigned long long)s.id * 37ULL
                               + (unsigned long long)s.nextPos.x * 13ULL
                               + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)ACTION_DIM);
}

void SACAZAgent::getLegalActions(int color,
                                 std::vector<Step*> &steps,
                                 std::vector<int> &actionIndices,
                                 RL::Tensor &actionMask)
{
    actionMask.zero();
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());
    for (Step *s : steps) {
        const int aidx = stepToActionIdx(*s);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

float SACAZAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    /*
       奖励尺度 (Phase 1 起与其他 agent 统一):
       本 agent 原来就用 `Stone::value` 原值 (车 0.5 / 马炮 0.3 / 兵 0.1), 方向是对的
       ——"单个吃子 < 终局 ±1"。但**一局累积**下来一方全材质 = 3.5 > 终局 1.0, 也就
       是说"吃光对方"在数值上仍然是"赢棋"的 3.5 倍 (诊断 [2] 的不变量当场抓住这一条,
       它对 SACAZ 同样成立)。现在统一到 stone.h 的 REWARD_MATERIAL_COEF = 0.1:
       一方全材质 0.35 < 终局 1.0, 终局真正主导。

       吃將仍然不给即时奖励 (这一点 SACAZ 原来就是对的, 现在推广到全部 agent):
       它必然是终局, 终局奖励会覆盖; 若按 value_jiang = 1000 给, Q/价值的尺度会被
       彻底压倒, 软备份与 MSE 都没法看。

       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来的黑方视角写法会让红方白吃一个黑子
       拿到负奖励, 与终局 (resultValue 给的走子方视角 ±1) 相反。
       (Chess::moveForward 的 totalReward 仍是黑方视角, 那是它的记账约定; 界面上的
       奖励曲线在 ChessBoard::playMatchGame 里显式换算成走子方视角。)
       回归钉在 test_match 的 [2.6] 节。
    */
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* ============================================================
 *  掩码 softmax 及其反向
 * ============================================================ */
void SACAZAgent::maskedSoftmax(const RL::Tensor &logits, const RL::Tensor &mask,
                               RL::Tensor &pi)
{
    pi.zero();
    float maxLogit = -std::numeric_limits<float>::max();
    bool any = false;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            any = true;
            if (logits[a] > maxLogit) {
                maxLogit = logits[a];
            }
        }
    }
    if (!any) {
        return;   /* 无合法动作: 保持全 0, 调用方必须先处理这种情况 */
    }
    float sum = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            const float e = std::exp(logits[a] - maxLogit);
            pi[a] = e;
            sum += e;
        }
    }
    if (sum <= 0.0f) {
        pi.zero();
        return;
    }
    const float inv = 1.0f / sum;
    for (int a = 0; a < ACTION_DIM; a++) {
        pi[a] *= inv;
    }
}

void SACAZAgent::maskedSoftmaxBackward(const RL::Tensor &pi, const RL::Tensor &g,
                                       RL::Tensor &dz)
{
    /*
       dL/dz_c = π_c·( g_c − Σ_a g_a·π_a )
       非法动作 π_c = 0 -> 梯度为 0, 与掩码语义一致 (永远不会把非法动作抬起来)。
    */
    float dot = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        dot += g[a] * pi[a];
    }
    for (int a = 0; a < ACTION_DIM; a++) {
        dz[a] = (pi[a] > 0.0f) ? (pi[a] * (g[a] - dot)) : 0.0f;
    }
}

void SACAZAgent::maskToBits(const RL::Tensor &mask, std::uint64_t &lo, std::uint64_t &hi)
{
    lo = 0;
    hi = 0;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] > 0.5f) {
            if (i < 64) {
                lo |= (std::uint64_t(1) << i);
            } else {
                hi |= (std::uint64_t(1) << (i - 64));
            }
        }
    }
}

void SACAZAgent::bitsToMask(std::uint64_t lo, std::uint64_t hi, RL::Tensor &mask)
{
    mask.zero();
    for (int i = 0; i < ACTION_DIM; i++) {
        const std::uint64_t bit = (i < 64) ? (lo >> i) : (hi >> (i - 64));
        mask[i] = (bit & 1ULL) ? 1.0f : 0.0f;
    }
}

/* ============================================================
 *  前向 / 软价值
 * ============================================================ */
void SACAZAgent::policy(const RL::Tensor &state, const RL::Tensor &mask,
                        RL::Tensor &pi)
{
    /* actor 的输出缓冲会被下一次 forward 覆盖, 先拷出来再做掩码归一化 */
    m_logits = actor.forward(state);
    maskedSoftmax(m_logits, mask, pi);
}

void SACAZAgent::qValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out)
{
    q1Out = q1.forward(state);
    q2Out = q2.forward(state);
}

void SACAZAgent::qTargetValues(const RL::Tensor &state, RL::Tensor &q1Out,
                               RL::Tensor &q2Out)
{
    q1Out = q1Target.forward(state);
    q2Out = q2Target.forward(state);
}

float SACAZAgent::softValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                                const RL::Tensor &q1In, const RL::Tensor &q2In) const
{
    const float a = alpha[0];
    float v = 0.0f;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] <= 0.5f || pi[i] <= 0.0f) {
            continue;
        }
        const float qmin = std::min(q1In[i], q2In[i]);
        v += pi[i] * (qmin - a * std::log(pi[i]));
    }
    return v;
}

/* ============================================================
 *  MCTS
 * ============================================================ */
double SACAZAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[childID];
    if (child.visitCount == 0) {
        /* 未访问过的子节点优先 (AlphaZero 的 PUCT 里 U 项在 N=0 时最大) */
        return std::numeric_limits<double>::max();
    }
    const double q = child.getQ();
    const double u = c_puct * child.prior
                     * std::sqrt((double)parentVisits)
                     / (1.0 + (double)child.visitCount);
    return q + u;
}

bool SACAZAgent::terminalValue(int color, double &value) const
{
    const int res = chess.getResult(color);
    if (res == Chess::RESULT_ONGOING) {
        return false;
    }
    if (res == Chess::RESULT_DRAW) {
        value = 0.0;
        return true;
    }
    const bool redWon = (res == Chess::RESULT_RED_WIN);
    value = (redWon == (color == Stone::COLOR_RED)) ? 1.0 : -1.0;
    return true;
}

bool SACAZAgent::resultValue(int result, int color, float &out)
{
    if (result == Chess::RESULT_ONGOING) {
        return false;
    }
    if (result == Chess::RESULT_DRAW) {
        out = 0.0f;
        return true;
    }
    const bool redWon = (result == Chess::RESULT_RED_WIN);
    out = (redWon == (color == Stone::COLOR_RED)) ? 1.0f : -1.0f;
    return true;
}

void SACAZAgent::visitDistribution(int rootID, RL::Tensor &pi)
{
    pi.zero();
    int total = 0;
    for (int childID : nodes[rootID].childIDs) {
        total += nodes[childID].visitCount;
    }
    if (total <= 0) {
        return;
    }
    const float inv = 1.0f / (float)total;
    for (int childID : nodes[rootID].childIDs) {
        const AZNode &c = nodes[childID];
        pi[c.parentAction] += (float)c.visitCount * inv;
    }
}

/* ============================================================
 *  selectMove: AlphaZero 式 PUCT + SAC 软价值叶子
 *
 *  一次模拟 = 选择(按 PUCT 下潜) -> 展开(先验最高的未尝试动作) ->
 *             估值(终局用真实胜负, 否则用**在线**双 Q 的软价值) ->
 *             回传(negamax 翻转) -> 撤销试走
 *
 *  与 PPOMCTSAgent 的三点差别:
 *    1. 叶子估值是软价值 min_i Q_i − α·log π (最大熵), 而不是一个价值头;
 *    2. 终局节点用真实胜负, 不再自举 (PPOMCTS 在终局也用网络值);
 *    3. 展开时按**先验**挑动作, 而不是随机挑 (同样模拟次数下更有效)。
 * ============================================================ */
Step SACAZAgent::selectMove(int color, int simulations_, float temp, RL::Tensor *piOut)
{
    nodes.clear();
    if (simulations_ < 1) {
        simulations_ = 1;
    }
    nodes.reserve((std::size_t)simulations_ + 64);

    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor qa(ACTION_DIM, 1);
    RL::Tensor qb(ACTION_DIM, 1);

    /* ---- 根节点 ---- */
    std::vector<Step*> rootSteps;
    std::vector<int> rootIdx;
    getLegalActions(color, rootSteps, rootIdx, mask);

    AZNode root;
    root.currentColor = color;
    root.legalCount = (int)rootIdx.size();
    root.parentID = -1;
    root.parentAction = -1;

    if (root.legalCount > 0) {
        encodeStateFor(color, m_stateBuf);
        policy(m_stateBuf, mask, pi);
    }
    for (std::size_t i = 0; i < rootIdx.size(); i++) {
        root.untriedActionIndices.push_back(rootIdx[i]);
        root.untriedSteps.push_back(*rootSteps[i]);
        root.untriedPriors.push_back((double)pi[rootIdx[i]]);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(root);
    const int rootID = 0;

    if (root.legalCount == 0) {
        /* 无合法走法: 返回无效 Step, 由调用方按"真无棋可走"处理 (见 issues_review C2) */
        if (piOut != nullptr) {
            piOut->zero();
        }
        return Step();
    }

    /* ---- 主循环 ---- */
    for (int sim = 0; sim < simulations_; sim++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* --- 1. SELECT --- */
        std::vector<Step> toApply;
        while (nodes[nodeID].untriedActionIndices.empty()
               && !nodes[nodeID].childIDs.empty()) {
            int bestChild = -1;
            double bestPuct = -std::numeric_limits<double>::max();
            for (int childID : nodes[nodeID].childIDs) {
                const double p = getPUCT(childID, nodes[nodeID].visitCount);
                if (p > bestPuct) {
                    bestPuct = p;
                    bestChild = childID;
                }
            }
            if (bestChild < 0) {
                break;
            }
            toApply.push_back(nodes[bestChild].step);
            nodeID = bestChild;
            path.push_back(nodeID);
        }
        double dummy = 0.0;
        for (const Step &s : toApply) {
            chess.moveForward(&s, dummy);
        }

        double leafValue = 0.0;
        bool haveLeafValue = false;

        /* 到达的节点本身可能已经终局 (走到这里时棋盘就是该节点的局面) */
        if (terminalValue(nodes[nodeID].currentColor, leafValue)) {
            haveLeafValue = true;
        }

        /* --- 2. EXPANSION --- */
        if (!haveLeafValue && !nodes[nodeID].untriedActionIndices.empty()) {
            std::size_t pickIdx = 0;
            for (std::size_t i = 1; i < nodes[nodeID].untriedPriors.size(); i++) {
                if (nodes[nodeID].untriedPriors[i] > nodes[nodeID].untriedPriors[pickIdx]) {
                    pickIdx = i;
                }
            }
            const int chosenAction = nodes[nodeID].untriedActionIndices[pickIdx];
            const Step chosenStep = nodes[nodeID].untriedSteps[pickIdx];
            const double chosenPrior = nodes[nodeID].untriedPriors[pickIdx];
            nodes[nodeID].untriedActionIndices.erase(
                nodes[nodeID].untriedActionIndices.begin() + (long)pickIdx);
            nodes[nodeID].untriedSteps.erase(
                nodes[nodeID].untriedSteps.begin() + (long)pickIdx);
            nodes[nodeID].untriedPriors.erase(
                nodes[nodeID].untriedPriors.begin() + (long)pickIdx);

            chess.moveForward(&chosenStep, dummy);

            const int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                      ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            std::vector<Step*> childSteps;
            std::vector<int> childIdx;
            getLegalActions(nextColor, childSteps, childIdx, mask);

            AZNode child(nodeID, chosenAction, chosenStep, chosenPrior, nextColor,
                         (int)childIdx.size());

            double v = 0.0;
            if (terminalValue(nextColor, v)) {
                /* 被将杀 / 困毙 / 和: 终局节点, 价值取自真实胜负 */
                child.isTerminal = true;
                leafValue = v;
            } else {
                encodeStateFor(nextColor, m_stateBuf);
                policy(m_stateBuf, mask, pi);
                for (std::size_t i = 0; i < childIdx.size(); i++) {
                    child.untriedActionIndices.push_back(childIdx[i]);
                    child.untriedSteps.push_back(*childSteps[i]);
                    child.untriedPriors.push_back((double)pi[childIdx[i]]);
                }
                /* 叶子估值: 用**在线**双 Q 的软价值 (目标网只用于学习时的备份) */
                qValues(m_stateBuf, qa, qb);
                leafValue = (double)softValueFrom(pi, mask, qa, qb);
                m_leafEvals++;
            }
            Steps::instance().put(childSteps);

            nodes.push_back(child);
            const int newID = (int)nodes.size() - 1;
            nodes[nodeID].childIDs.push_back(newID);
            nodeID = newID;
            path.push_back(newID);
            haveLeafValue = true;
        }

        /* --- 3. 已全展开的节点: 就地用软价值 (终局已在上面处理) --- */
        if (!haveLeafValue) {
            std::vector<Step*> ls;
            std::vector<int> li;
            getLegalActions(nodes[nodeID].currentColor, ls, li, mask);
            Steps::instance().put(ls);
            encodeStateFor(nodes[nodeID].currentColor, m_stateBuf);
            policy(m_stateBuf, mask, pi);
            qValues(m_stateBuf, qa, qb);
            leafValue = (double)softValueFrom(pi, mask, qa, qb);
            m_leafEvals++;
        }

        /* --- 4. BACKUP (negamax) --- */
        double v = leafValue;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[path[i]].visitCount++;
            nodes[path[i]].totalValue += v;
            v = -v;
        }

        /* --- 5. 撤销本轮试走 (path[0] 是根, 没有对应的 step) --- */
        for (std::size_t i = path.size(); i > 1; i--) {
            const Step &s = nodes[path[i - 1]].step;
            chess.moveBack(&s, dummy);
        }
    }

    if (piOut != nullptr) {
        visitDistribution(rootID, *piOut);
    }

    /* ---- 选择走法 ---- */
    int bestChildID = -1;
    if (temp <= 1e-6f) {
        int maxVisits = -1;
        for (int childID : nodes[rootID].childIDs) {
            if (nodes[childID].visitCount > maxVisits) {
                maxVisits = nodes[childID].visitCount;
                bestChildID = childID;
            }
        }
    } else {
        /* 按 N^(1/T) 采样 (自对弈用, 保证开局多样性) */
        RL::Tensor dist(ACTION_DIM, 1);
        dist.zero();
        double sum = 0.0;
        const double invT = 1.0 / (double)temp;
        for (int childID : nodes[rootID].childIDs) {
            const double w = std::pow((double)nodes[childID].visitCount, invT);
            dist[nodes[childID].parentAction] += (float)w;
            sum += w;
        }
        if (sum > 0.0) {
            const int a = RL::Random::categorical(dist);
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].parentAction == a) {
                    bestChildID = childID;
                    break;
                }
            }
        }
        if (bestChildID < 0) {
            int maxVisits = -1;
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].visitCount > maxVisits) {
                    maxVisits = nodes[childID].visitCount;
                    bestChildID = childID;
                }
            }
        }
    }

    if (bestChildID >= 0) {
        return nodes[bestChildID].step;
    }
    return Step();
}

void SACAZAgent::resetMoeBatchStats()
{
    if (auxLossCoef <= 0.0f) {
        return;
    }
    /*
       目标网 q1Target/q2Target 也要复位: 它们只前向、不训练, 门控统计永远用不到
       (辅助损失只注入在线网), 但 xSum/probSumBatch 是 float 累加器 —— 不复位的话
       会随一局的模拟次数一路涨上去, 精度慢慢烂掉。
    */
    RL::Net *nets[5] = {&actor, &q1, &q2, &q1Target, &q2Target};
    for (int ni = 0; ni < 5; ni++) {
        for (std::size_t li = 0; li < nets[ni]->size(); li++) {
            RL::ISparseMoE *moe = dynamic_cast<RL::ISparseMoE*>((*nets[ni])[li]);
            if (moe != nullptr) {
                moe->resetBatchStats();
            }
        }
    }
}

/* ============================================================
 *  learnBatch: 一次 mini-batch 的 SAC 更新 (critic / actor / α)
 * ============================================================ */
float SACAZAgent::learnBatch(int batchSize_, int epochs)
{
    if (batchSize_ < 1 || (int)memories.size() < batchSize_) {
        return 0.0f;
    }
    if (epochs <= 0) {
        epochs = replayEpochs > 0 ? replayEpochs : 1;
    }
    if (epochs < 1) {
        epochs = 1;
    }

    RL::Tensor state(STATE_DIM, 1);
    RL::Tensor nextState(STATE_DIM, 1);
    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor nextMask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor piNext(ACTION_DIM, 1);
    RL::Tensor q1n(ACTION_DIM, 1);
    RL::Tensor q2n(ACTION_DIM, 1);
    RL::Tensor q1o(ACTION_DIM, 1);
    RL::Tensor q2o(ACTION_DIM, 1);
    RL::Tensor g(ACTION_DIM, 1);
    RL::Tensor dz(ACTION_DIM, 1);

    /*
       [P4] **每个 epoch 重新从池里抽** batchSize 条 (与 RL::PPO::learnFromReplay 同一
       做法), 累积梯度后优化器只在最后调一次 (P3)。

       为什么不是"把同一批复用 epochs 遍": 批内权重不变, 所以第 N 遍的梯度与第 1 遍
       **逐位相同**; 而 `RL::Net::RMSProp` 默认 clipGrad=true (`dw /= |dw|`) 会把这个
       纯倍数完全归一掉 —— 重复同一批对更新方向**毫无影响**, 只是白烧算力。
       每遍抽新样本才是真东西: 一次更新看到 batchSize×epochs 条经验 (等于把批放大
       epochs 倍, 但仍然只调一次优化器)。生成一条样本要走整棵 MCTS, 比抽一条贵得多,
       所以这个放大基本是白拿的。
    */
    std::uniform_int_distribution<int> pick(0, (int)memories.size() - 1);

    /*
       [MoE] 批统计的**边界**: 只反映本批的训练前向。
       搜索期间每次模拟都会跑一次策略/价值前向, 那些是"推理前向", 不该混进负载均衡
       辅助损失的批均值里 (见 resetMoeBatchStats 的说明)。
    */
    resetMoeBatchStats();

    float lossSum = 0.0f;
    int n = 0;
    float alphaGrad = 0.0f;
    const float a = alpha[0];

    for (int ep = 0; ep < epochs; ep++) {
    for (int it = 0; it < batchSize_; it++) {
        const Transition &tr = memories[(std::size_t)pick(RL::Random::engine)];
        expandSparse(tr.cells, state);
        writeContext(state, tr.ctx);
        expandSparse(tr.nextCells, nextState);
        writeContext(nextState, tr.nextCtx);
        bitsToMask(tr.curMaskLo, tr.curMaskHi, mask);
        bitsToMask(tr.nextMaskLo, tr.nextMaskHi, nextMask);

        /* ---- 目标侧: 用目标网算下一局面的软价值 ---- */
        policy(nextState, nextMask, piNext);
        qTargetValues(nextState, q1n, q2n);
        const float vNext = softValueFrom(piNext, nextMask, q1n, q2n);

        /*
           符号: 所有价值都是"该局面走棋方视角"。s' 轮到**对手**走, 所以自举项要取负
           (negamax):   y = r − γ(1−done)·V(s')
        */
        const float y = tr.reward - gamma * (tr.done ? 0.0f : 1.0f) * vNext;

        /* ---- 当前局面 ---- */
        policy(state, mask, pi);
        qValues(state, q1o, q2o);

        /* critic 损失: 只对实际走的那一步回归 (SAC 的标准做法, 其余动作误差为 0) */
        const float err = q1o[tr.action] - y;
        lossSum += err * err;

        for (int ci = 0; ci < 2; ci++) {
            RL::Net &qnet = (ci == 0) ? q1 : q2;
            RL::Tensor target = (ci == 0) ? q1o : q2o;
            target[tr.action] = y;
            qnet.backward(state, RL::Loss::MSE::df((ci == 0) ? q1o : q2o, target));
        }

        /*
           策略损失 —— 两项都是"对 softmax 输出 π 的梯度", 相加后交给掩码 softmax
           的雅可比:
             (a) SAC 软 Q 项   dJ/dπ_a = α·(log π_a + 1) − min_i Q_i(s,a)
             (b) AlphaZero 监督项  CE(π_MCTS, π) 的梯度正好是 (π_a − π_MCTS,a);
                 用这个形式而不是 −π_MCTS/π, 因为后者在 π_a -> 0 时会爆掉。
        */
        for (int i = 0; i < ACTION_DIM; i++) {
            if (mask[i] <= 0.5f) {
                g[i] = 0.0f;   /* 非法动作不参与: π_i = 0, 绝不能把它抬起来 */
                continue;
            }
            float gi = a * (std::log(pi[i] + 1e-8f) + 1.0f)
                       - std::min(q1o[i], q2o[i]);
            if (tr.hasSearch) {
                gi += azWeight * (pi[i] - tr.pi[i]);
            }
            g[i] = gi;
        }
        maskedSoftmaxBackward(pi, g, dz);
        actor.backward(state, dz);

        /*
           α 自动调节:
             J(α) = α·(H − H̄)  =>  dJ/dα = H − H̄   (H = 策略熵, H̄ = 目标熵)
           Optimize::RMSProp 是梯度下降 (w -= lr·g/√v), 所以
             熵低于目标 -> g < 0 -> α 变大 -> 更探索;
             熵高于目标 -> g > 0 -> α 变小 -> 更利用。
        */
        float H = 0.0f;
        for (int i = 0; i < ACTION_DIM; i++) {
            if (pi[i] > 0.0f) {
                H -= pi[i] * std::log(pi[i]);
            }
        }
        const int lc = (tr.legalCount > 1) ? tr.legalCount : 2;
        const float Hbar = entropyRatio * std::log((float)lc);
        alphaGrad += (H - Hbar);
        n++;
    }
    }   /* ---- epochs 循环结束 (P4) ---- */

    if (n == 0) {
        return 0.0f;
    }

    /*
       本批的平均 critic 损失 (界面曲线用, 见 getLastTrainLoss)。
       取**批平均**而不是最后一条: 逐样本上报会让曲线变成低占空比的脉冲
       (重尾样本能差几个数量级), 而且与 DQN 报"平均平方 TD 误差"的口径对不上。
       P4 之后分母是 batchSize × epochs (每个 epoch 各抽 batchSize 条)。
    */
    m_lastLoss = lossSum / (float)n;
    m_lastBatchSamples = n;

    /*
       ---- 稀疏 MoE 的负载均衡辅助损失 ----
       放在优化器之前、主反向之后: 这个 mini-batch 里每个层的前向次数、被选中的
       专家次数、门控概率之和都已经累计好了, addAuxGradient 用这些统计算出
       "哪些专家被喂爆了", 把它们的 logit 压下去、把饿着的抬起来 (Switch
       Transformer 的 L_aux = E·Σ f_i·P_i 对 logits 的梯度)。
       没有这一项时, softmax 的反向会把没被选中的专家的概率继续压低, 路由会迅速
       坍缩到少数专家、其余永远不训练。细节与有限差分验证见 rl/sparse_moe.hpp
       和 test/test_sparse_moe_main.cpp [6][7]。
    */
    if (auxLossCoef > 0.0f) {
        RL::Net *nets[3] = {&actor, &q1, &q2};
        for (int ni = 0; ni < 3; ni++) {
            for (std::size_t li = 0; li < nets[ni]->size(); li++) {
                RL::ISparseMoE *moe = dynamic_cast<RL::ISparseMoE*>((*nets[ni])[li]);
                if (moe != nullptr) {
                    moe->addAuxGradient(auxLossCoef);
                }
            }
        }
    }

    /* ---- 应用梯度 ---- */
    actor.RMSProp(learningRateActor, 0.9f, 0.0f);
    q1.RMSProp(learningRateCritic, 0.9f, 0.0f);
    q2.RMSProp(learningRateCritic, 0.9f, 0.0f);

    /* α 的梯度是整批累加的, 取平均后再更新 */
    alpha.g[0] = alphaGrad / (float)n;
    alpha.RMSProp(learningRateAlpha, 0.9f, 0.0f);
    alpha.clamp(0.02f, 0.02f, 5.0f);

    /* ---- 目标网 Polyak 同步 ---- */
    learnSteps++;
    if (learnSteps % (replaceTargetIter > 0 ? replaceTargetIter : 1) == 0) {
        const float tau = 1e-3f;
        q1.softUpdateTo(q1Target, tau);
        q2.softUpdateTo(q2Target, tau);
    }

    /* ---- 回放缓冲上限 ---- */
    while (memories.size() > maxMemorySize) {
        memories.pop_front();
    }

    return lossSum / (float)n;
}

/* ============================================================
 *  trainSelfPlay: 自对弈 (MCTS 访问分布做策略目标) + 回放训练
 * ============================================================ */
void SACAZAgent::trainSelfPlay(int episodes, int simulations_, int maxMoves,
                               bool verbose, float tempRoot, float tempFinal,
                               int learnEveryMoves)
{
    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int turn = Stone::COLOR_RED;
        int moves = 0;

        while (moves < maxMoves) {
            const int res = chess.getResult(turn);
            if (res != Chess::RESULT_ONGOING) {
                break;
            }

            /* 温度退火: 开局高 (多样性), 中后盘低 (质量) */
            const float frac = (float)moves / (float)(maxMoves > 1 ? maxMoves : 1);
            const float temp = tempRoot + (tempFinal - tempRoot) * frac;

            RL::Tensor piTarget(ACTION_DIM, 1);
            const Step s = selectMove(turn, simulations_, temp, &piTarget);
            if (!s.valid) {
                break;
            }

            Transition tr;
            encodeSparse(turn, tr.cells);
            contextOf(chess, turn, tr.ctx);
            for (int i = 0; i < ACTION_DIM; i++) {
                tr.pi[i] = piTarget[i];
            }
            tr.action = stepToActionIdx(s);
            tr.hasSearch = true;
            tr.reward = computeReward(s, turn);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(turn, legal, idx, mask);
                tr.legalCount = (int)idx.size();
                maskToBits(mask, tr.curMaskLo, tr.curMaskHi);
                Steps::instance().put(legal);
            }

            double dummy = 0.0;
            chess.moveForward(&s, dummy);

            const int nextTurn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                            : Stone::COLOR_RED;
            encodeSparse(nextTurn, tr.nextCells);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(nextTurn, legal, idx, mask);
                maskToBits(mask, tr.nextMaskLo, tr.nextMaskHi);
                Steps::instance().put(legal);
            }
            const int after = chess.getResult(nextTurn);
            tr.done = (after != Chess::RESULT_ONGOING);
            if (tr.done) {
                float rv = 0.0f;
                if (resultValue(after, turn, rv)) {
                    tr.reward = rv;   /* 终局奖励 ±1/0 覆盖即时奖励 */
                }
            }

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }

            turn = nextTurn;
            moves++;

            if (learnEveryMoves > 0 && (moves % learnEveryMoves) == 0) {
                learnBatch(batchSize);
            }
        }

        totalEpisodes++;
        const int res = chess.getResult(turn);
        if (res == Chess::RESULT_RED_WIN) {
            totalWins[0]++;
        } else if (res == Chess::RESULT_BLACK_WIN) {
            totalWins[1]++;
        }

        if (verbose) {
            std::printf("  [SAC+AZ] episode %d: %d 手, result=%d, 池=%zu, alpha=%.3f\n",
                        ep + 1, moves, res, memories.size(), (double)alpha[0]);
        }
    }
}

void SACAZAgent::warmupFromCurrent(int episodes, int simulations_, int maxMoves)
{
    /* 从当前局面继续 (不 reset) —— 调用方负责棋盘状态 */
    const int savedTurn = chess.sideToMove;
    trainSelfPlay(episodes, simulations_, maxMoves, false, 1.0f, 0.25f, 4);
    chess.sideToMove = savedTurn;
}

/* ============================================================
 *  exploreAndTrain: 走子前的"探索环境 + 在线训练一次"
 *
 *  与自对弈的区别: 这里**不做搜索**, 直接用掩码策略 π(·|s) 采样滚若干步 (最大熵
 *  策略本身就是探索策略), 把经验塞进回放缓冲后做一次 mini-batch 更新。
 *  这些样本的策略目标不是搜索出来的, 所以 hasSearch=false —— 只训练 critic 与
 *  SAC 的软 Q 项, 不用 AlphaZero 的监督项 (否则就是把策略往它自己身上拉)。
 * ============================================================ */
bool SACAZAgent::exploreAndTrain(int color, int rolloutSteps)
{
    if (rolloutSteps <= 0) {
        m_exploreInfo = "SAC+AZ: 探索步数为 0, 已跳过";
        return false;
    }

    const int collected = rolloutFromCurrent(
        *this, chess, color, rolloutSteps,
        /* pick: 用掩码策略采样 */
        [this](const RL::Tensor &state, int turn) -> int {
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(turn, legal, idx, mask);
            if (legal.empty()) {
                Steps::instance().put(legal);
                return -1;
            }
            RL::Tensor pi(ACTION_DIM, 1);
            policy(state, mask, pi);
            /* 把这一步的合法数与掩码存下来, 给紧接着的 onTrans 复用 */
            m_pendingLegalCount = (int)idx.size();
            maskToBits(mask, m_pendingMaskLo, m_pendingMaskHi);
            Steps::instance().put(legal);
            return RL::Random::categorical(pi);
        },
        /* onTrans: 存一条 off-policy 经验 */
        [this](const Step &chosen, int actionIdx, const RL::Tensor &stateBefore,
               const RL::Tensor &nextState, float reward, bool done) {
            (void)chosen;
            Transition tr;
            denseToSparse(stateBefore, tr.cells);
            readContext(stateBefore, tr.ctx);
            denseToSparse(nextState, tr.nextCells);
            readContext(nextState, tr.nextCtx);
            tr.action = actionIdx;
            tr.legalCount = m_pendingLegalCount;
            tr.curMaskLo = m_pendingMaskLo;
            tr.curMaskHi = m_pendingMaskHi;
            tr.reward = reward;
            tr.done = done;
            tr.hasSearch = false;   /* 策略目标不来自搜索 */

            /* 下一局面的合法掩码: 此刻棋盘正好停在 nextState */
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(chess.sideToMove, legal, idx, mask);
            maskToBits(mask, tr.nextMaskLo, tr.nextMaskHi);
            Steps::instance().put(legal);

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }
        });

    /*
       在线训练一次。**批大小要按池里的实际条数夹一下**: learnBatch 在
       "池 < batchSize(默认 32)" 时是**故意**直接返回 0 的 (不拿半个 batch 去更新),
       于是刚开局时(池里只有探索刚收集的十几条)这一轮就白跑了 —— 界面上表现为
       "损失曲线一直是空的", 而且在线学习在开局阶段完全没发生。
    */
    const int onlineBatch = std::min(batchSize, (int)memories.size());
    const float loss = learnBatch(onlineBatch);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "SAC+AZ 探索 %d 步, 训练 1 次 (池 %zu, critic loss %.4f, alpha %.3f)",
                  collected, memories.size(), (double)loss, (double)alpha[0]);
    m_exploreInfo = buf;
    return collected > 0;
}

/* ============================================================
 *  存取: 一个路径前缀 -> 三个文件
 * ============================================================ */
bool SACAZAgent::saveModel(const std::string &filepath)
{
    actor.save(filepath + "_actor");
    q1.save(filepath + "_q1");
    q2.save(filepath + "_q2");
    return weightFileWritten(filepath + "_actor")
           && weightFileWritten(filepath + "_q1")
           && weightFileWritten(filepath + "_q2");
}

bool SACAZAgent::loadModel(const std::string &filepath)
{
    if (!weightFileReadable(filepath + "_actor")
        || !weightFileReadable(filepath + "_q1")
        || !weightFileReadable(filepath + "_q2")) {
        return false;
    }
    actor.load(filepath + "_actor");
    q1.load(filepath + "_q1");
    q2.load(filepath + "_q2");
    q1.copyTo(q1Target);
    q2.copyTo(q2Target);
    return true;
}

/* ============================================================
 *  AgentBase
 * ============================================================ */
Step SACAZAgent::getBestMove(int color)
{
    return selectMove(color, simulations, 0.0f);
}
