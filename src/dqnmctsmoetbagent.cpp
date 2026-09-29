#include "dqnmctsmoetbagent.h"

#include "chessstate.h"       /* 规范格 / 规则上下文的公共实现 (只有一份) */
#include "agentrollout.hpp"   /* rolloutFromCurrent: 探索期收样本 */
#include "rl/sparse_moe.hpp"  /* SparseMoE / ISparseMoE (头文件只前置声明) */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>    /* selfCheckReport 的动作别名统计: 槽位 -> 互不相同的走法 */
#include <set>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

/* ================================================================
 *  DQNMCTSMOETbAgent — 实现
 *
 *  三条"读代码之前先知道"的约定 (每一条都是本工程踩过坑才写下来的):
 *
 *  1. **价值都是"当前走棋方视角" (negamax)**。状态是规范视角 (己方永远在 x 大的
 *     那一侧), 所以 Q(s,a) 表示"轮到我走时, 我走 a 之后的价值"。于是:
 *         叶子价值        = max_a Q(s,a)
 *         自举目标        = r − γ·(1−done)·Q_target(s', a*)      <- **减号**
 *         PUCT 比较子节点 = −child.getQ()                        <- **负号**
 *     这两处符号是同一个约定的两个后果: s' 轮到**对手**走, 所以它的价值是"对手的
 *     价值", 要取负才是我的。旧类 (`DQNMCTSAgent`) 用的是"黑为正"的非规范编码 +
 *     正号自举 + 正号 UCB1, 它自成一个体系 (即时奖励要换算到黑方框架); 本类换成
 *     SAC/PPOMCTS 的规范口径, 因为那条线上这套符号已经被实测核对过。
 *
 *  2. **搜索侧与学习侧必须同口径**。终局值只有一个出口 (`terminalReward()`),
 *     搜索结果与训练目标都走它; 叶子估值的口径与自举用的口径必须是同一个数,
 *     否则"搜索估的"与"训练学的"是两个不同的游戏。
 *
 *  3. **热路径不许重复跑骨干**。`trunk` / `qHead` / `qNet` 是共享同一批层的三个视图:
 *     搜索与训练都只调 `trunk.forward()` 一次, 头只算需要的列。写 `qNet.forward()`
 *     在热路径上是**静默**的 3 倍开销 (结果还是对的, 所以不会有任何报错)。
 * ================================================================ */

namespace {

/* 上下文个数必须与 chessstate.h 的那三项一一对应 (顺序见 contextOf) */
static_assert(DQNMCTSMOETbAgent::CTX_COUNT == 3,
              "CTX_COUNT 变了就必须同时改 contextOf 的顺序与注释");
static_assert(DQNMCTSMOETbAgent::STATE_DIM
                  == DQNMCTSMOETbAgent::PIECE_PLANES * DQNMCTSMOETbAgent::CELLS
                         + DQNMCTSMOETbAgent::CTX_COUNT,
              "STATE_DIM 必须是 14 个棋子平面 + CTX_COUNT 个上下文标量");

/*
 *  造稀疏 MoE 层。TopK 是**模板参数**, 所以"稀疏"与"等参数稠密对照"是两个不同的
 *  模板实例, 共同的基类是 ISparseMoE (不是 shared_ptr 能协变的类型) —— 这里
 *  static_pointer_cast 一次, 调用点就不必写两份几乎相同的 make_shared。
 */
template<typename Expert, int E, int K>
RL::iLayer::sptr makeMoeLayer(int d, bool withGrad)
{
    return std::static_pointer_cast<RL::iLayer>(
        std::make_shared<RL::SparseMoE<Expert, E, K> >(d, withGrad, 0));
}

/*
 *  TB 专家那一层。`HonorHeads=true` 是 SAC 那条线上实测的**修复**:
 *  老的"头数必须整除 d_model"规则在 d_model = 1263 = 3 x 421 上会把请求的 15 个头
 *  静默降成 3 个 (d_k 从 84 涨到 421), 一个专家前向 6.20 -> 19.06 ms (3.07x),
 *  而且 15 个 head 对象全都分配了、只有 3 个参与前向。本类的 STATE_DIM 与
 *  MOE_TB_HEADS 就是那一对数字, 所以这里必须用 HonorHeads=true。
 *  `dense` 是等参数对照组 (TopK == E)。
 */
RL::iLayer::sptr makeTbExpertMoe(bool withGrad, bool dense)
{
    typedef RL::TransformerBlock<DQNMCTSMOETbAgent::MOE_TB_HEADS,
                                 DQNMCTSMOETbAgent::MOE_TB_DFF, true> TbExpert;
    if (dense) {
        return makeMoeLayer<TbExpert, DQNMCTSMOETbAgent::MOE_TB_EXPERTS,
                            DQNMCTSMOETbAgent::MOE_TB_EXPERTS>(DQNMCTSMOETbAgent::STATE_DIM,
                                                               withGrad);
    }
    return makeMoeLayer<TbExpert, DQNMCTSMOETbAgent::MOE_TB_EXPERTS,
                        DQNMCTSMOETbAgent::MOE_TB_TOPK>(DQNMCTSMOETbAgent::STATE_DIM,
                                                        withGrad);
}

/*
 *  初始化缩放: `iFcLayer` 的构造把权重抽成 U(-1,1), 对 1263 维输入来说
 *  pre-activation 的标准差约 sqrt(1263/3) ≈ 20 —— Tanh 一上来就饱和, 梯度接近 0。
 *  按 1/sqrt(fan_in) 再缩一次, 把标准差拉回 ~0.6。
 *  **只缩 iFcLayer**: 稀疏 MoE 层不是 iFcLayer (dynamic_cast 返回 nullptr), 它的
 *  专家与门控权重已经在 SparseMoE 的构造函数里用 scaleExpertInit 缩过了 ——
 *  重复缩同一个层会把初始化标准差再压一次 (SAC 那条线上有这个注释)。
 */
void scaleFcLayers(RL::Net &net)
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

/* 找出网络里第一个稀疏 MoE 层 (本类只有一个) */
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

/* Chess::Result -> 自检面板的结束方式编码 (一局恰好记一条, 见 noteEnd) */
int endCodeOfResult(int chessResult)
{
    switch (chessResult) {
    case Chess::RESULT_RED_WIN:   return DQNMCTSMOETbAgent::END_RED_WIN;
    case Chess::RESULT_BLACK_WIN: return DQNMCTSMOETbAgent::END_BLACK_WIN;
    case Chess::RESULT_DRAW:      return DQNMCTSMOETbAgent::END_DRAW;
    default:                      return DQNMCTSMOETbAgent::END_CAP;   /* ONGOING */
    }
}

/*
 *  一个着法落到哪个 Q 槽位 —— **与 stepToActionIdx 必须逐字相同**。
 *  为什么抄一份而不是调成员函数: 下面的别名统计拿 `chess` 的**副本**反复试走回退,
 *  而成员函数没有任何状态, 逐字抄一份即可; 更要紧的是"探针/面板统计的槽位"与
 *  "训练真正用的槽位"必须是同一个数, 所以两份公式由 test 的 [2] 逐项对齐。
 */
int aliasActionIdxOf(const Step &s)
{
    const unsigned long long h = (unsigned long long)s.id * 37ULL
                               + (unsigned long long)s.nextPos.x * 13ULL
                               + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)DQNMCTSMOETbAgent::ACTION_DIM);
}

/* 某个局面上"合法着法 -> Q 槽位"的别名情况 (只读: 棋盘引用不被改动) */
void aliasOfPosition(const std::vector<Step*> &legal,
                     int &legalCount, int &slotCount, int &worstSlot)
{
    std::map<int, std::set<std::string> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        char key[64];
        std::snprintf(key, sizeof(key), "%d>%d,%d", s.id, s.nextPos.x, s.nextPos.y);
        bucket[aliasActionIdxOf(s)].insert(std::string(key));
    }
    legalCount = (int)legal.size();
    slotCount = (int)bucket.size();
    worstSlot = 0;
    for (std::map<int, std::set<std::string> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        worstSlot = std::max(worstSlot, (int)it->second.size());
    }
}

}  // namespace

/* ================================================================
 *  位图助手 (合法槽位)
 * ================================================================ */
bool DQNMCTSMOETbAgent::bitSet(const std::uint64_t bits[2], int idx)
{
    if (idx < 0 || idx >= ACTION_DIM) {
        return false;
    }
    return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
}

void DQNMCTSMOETbAgent::bitSetTo(std::uint64_t bits[2], int idx)
{
    if (idx < 0 || idx >= ACTION_DIM) {
        return;
    }
    bits[idx >> 6] |= (1ULL << (idx & 63));
}

bool DQNMCTSMOETbAgent::bitsEmpty(const std::uint64_t bits[2])
{
    return (bits[0] | bits[1]) == 0ULL;
}

/* ================================================================
 *  构造 / 建网
 * ================================================================ */
DQNMCTSMOETbAgent::DQNMCTSMOETbAgent(Chess &chess_,
                                     int hiddenDim_,
                                     float gamma_,
                                     float lr,
                                     float eps,
                                     float c_puct_,
                                     bool denseMoe_)
    : AgentBase(),
      chess(chess_),
      hiddenDim(hiddenDim_ > 0 ? hiddenDim_ : 64),
      denseMoe(denseMoe_),
      gamma(gamma_),
      learningRate(lr),
      batchSize(32),
      maxMemorySize(4096),
      replayEpochs(1),
      targetTau(1.0f),
      replaceTargetIter(64),
      clampTarget(2.0f),
      huberDelta(1.0f),
      doubleDQN(true),
      twinCritic(false),
      learnEveryMoves(8),
      learnFromSearch(true),
      auxLossCoef(0.1f),
      exploringRate(eps),
      /*
         模拟次数默认 120 是**实测定的**: 40 次时开局 ~40 个合法着法 ⇒ 每个孩子恰好
         1 次访问、严格打平, 根选择退化成"按先验顺序取第一个孩子" = 1 层 Q 贪心
         (证据: --sims=40 --sims2=120 的着法一致率 0/16; 而 40 次 = 123~135 ms/步,
         120 次 = 364~371 ms/步)。这是"边界值必须让搜索真的能回访孩子"那条纪律。
      */
      simulations(120),
      c_puct(c_puct_),
      priorTemp(1.0f),
      sparseLeafEval(true),
      trainTempRoot(1.0f),
      trainTempFinal(0.25f),
      trainTempMoves(8),
      m_trainingMode(false),
      m_onlineStepCount(0),
      totalEpisodes(0),
      m_learnSteps(0),
      m_learnCounter(0),
      m_leafEvals(0),
      m_fullLeafEvals(0),
      m_lastLoss(std::numeric_limits<double>::quiet_NaN()),
      m_lastBatchSamples(0),
      m_maxAbsTarget(0.0),
      m_maxAbsTdErr(0.0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;

    /*
        ⚠ 这里**没有** `std::srand(time(nullptr))` —— 旧类那一支有, 而它与本工程的
       `RL::Random` 是两条互相独立的流: 只要有一个 agent 在构造时按 time() 播种,
       同一命令行跑两次的着法序列就会不同 (SAC 那边记录过这个坑: 一个"看着单调"的
       模拟次数扫描其实是对手随机流造出来的)。本类的所有随机性 (ε-greedy / N^(1/T)
       采样 / 批抽样) 一律走 `RL::Random`, 于是"同基种子 ⇒ 同结果"是可复现的。
    */
    buildNets();
}

/*
 *  建网。**层构造顺序是契约的一部分**: 每层构造时都从 `RL::Random` 抽初始化权重,
 *  所以顺序一变, 后面所有层的初始值 (以及所有金标读数) 都会变。这里的顺序是
 *  "在线: MoE -> Tanh -> Q 头 -> Q2 头; 目标: 同样的顺序", 目标网随后由在线网
 *  copyTo 覆盖, 所以它自己的初始值不参与任何计算 (但它照样消耗随机数 —— 固定顺序
 *  才能保证"同一个基种子 ⇒ 同一份权重")。
 */
void DQNMCTSMOETbAgent::buildNets()
{
    const std::size_t h = (std::size_t)hiddenDim;

    RL::Net::Layers trunkL;      /* 在线骨干 */
    RL::Net::Layers headL;       /* 在线 Q1 头 */
    RL::Net::Layers head2L;      /* 在线 Q2 头 (twinCritic 用; 见头文件说明) */
    RL::Net::Layers trunkT, headT, head2T;

    trunkL.push_back(makeTbExpertMoe(true, denseMoe));
    trunkL.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, true));
    /* Q 头必须是 Linear (不是 Sigmoid): 象棋奖励含负值, Q 必须能取负 */
    headL.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, true));
    head2L.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, true));

    trunkT.push_back(makeTbExpertMoe(false, denseMoe));
    trunkT.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, false));
    headT.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, false));
    head2T.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, false));

    /*
       视图组装: 同一批 shared_ptr 同时进 `trunk`/`qHead`/`qNet` —— 这就是
       "骨干只前向一次"的**全部机制**。三个 Net 之间没有任何权重拷贝:
       改 `trunk` 的层对象, `qNet` 看到的就是同一份权重。
    */
    trunk = RL::Net(trunkL);
    qHead = RL::Net(headL);
    q2Head = RL::Net(head2L);
    {
        RL::Net::Layers all = trunkL;
        all.insert(all.end(), headL.begin(), headL.end());
        qNet = RL::Net(all);
        RL::Net::Layers all2 = trunkL;
        all2.insert(all2.end(), head2L.begin(), head2L.end());
        q2Net = RL::Net(all2);
    }

    trunkTarget = RL::Net(trunkT);
    qHeadTarget = RL::Net(headT);
    q2HeadTarget = RL::Net(head2T);
    {
        RL::Net::Layers all = trunkT;
        all.insert(all.end(), headT.begin(), headT.end());
        qTargetNet = RL::Net(all);
        RL::Net::Layers all2 = trunkT;
        all2.insert(all2.end(), head2T.begin(), head2T.end());
        q2TargetNet = RL::Net(all2);
    }

    /* 初始化缩放 (只缩 iFcLayer; 稀疏 MoE 内部已经缩过) */
    scaleFcLayers(trunk);
    scaleFcLayers(qHead);
    scaleFcLayers(q2Head);

    /*
       目标网 = 在线网的一份**深拷贝**。`RL::Net` 的赋值运算符是浅拷贝 (共享
       layer 指针), 所以必须走 copyTo —— 直接用 `=` 会让目标网就是在线网本身,
       自举目标跟着在线网每步变化 (那等于没有目标网, 而训练照样跑)。
    */
    trunk.copyTo(trunkTarget);
    qHead.copyTo(qHeadTarget);
    q2Head.copyTo(q2HeadTarget);

    /* 搜索/训练复用的缓冲区 */
    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_nextBuf = RL::Tensor(STATE_DIM, 1);
    m_maskBuf = RL::Tensor(ACTION_DIM, 1);
    m_qBuf = RL::Tensor(ACTION_DIM, 1);
    m_q2Buf = RL::Tensor(ACTION_DIM, 1);
    m_qLegal.reserve(64);
    m_q2Legal.reserve(64);
    m_priors.reserve(64);
    memories.clear();
    m_lastDecisionFromBack = -1;
}

const char *DQNMCTSMOETbAgent::backboneName() const
{
    return denseMoe ? "稠密MoE(TB专家) x4 全算 (等参数对照)"
                    : "稀疏MoE(TB专家, E=4 top-1)";
}

/*
 *  隐层激活的**真实类型** —— 读的是 `qNet` 第 2 层的类型, 不是任何开关的回显。
 *  为什么必须有这个读数: 这一层曾经被换成 `TanhNorm<Sigmoid>` 而让随机权重下的棋力
 *  掉 26 个点, 而当时参数指纹 / paramCount / 层类型序列**一个都没变**, 面板上一个字
 *  都看不出来。以后谁再动这一层, 自检面板会直接显示出来。
 */
const char *DQNMCTSMOETbAgent::hiddenActivationName() const
{
    RL::Net &self = const_cast<RL::Net &>(qNet);   /* Net::operator[] 没有 const 重载 */
    if (self.size() < 2) {
        return "? (网络层数不足, buildNets 被改坏了?)";
    }
    if (dynamic_cast<RL::Layer<RL::Tanh> *>(self[1]) != nullptr) {
        return "tanh (Layer<Tanh>)";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Sigmoid> *>(self[1]) != nullptr) {
        return "**TanhNorm<Sigmoid> —— 就是那个掉 26 个点的回归!**";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Linear> *>(self[1]) != nullptr) {
        return "TanhNorm<Linear> —— 偏置加在 tanh **外面**, 与 Layer<Tanh> 不等价";
    }
    return "其它 (既不是 Layer<Tanh> 也不是已知的 TanhNorm —— 检查 buildNets 第 2 层)";
}

/* ================================================================
 *  表示: 规范视角 14 平面 + 3 个规则上下文标量 (STATE_DIM = 1263)
 * ================================================================ */

void DQNMCTSMOETbAgent::encodeSparse(int color, std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[(std::size_t)i];
        if (s == nullptr || s->alive == false || s->type < 0 || s->type >= 7) {
            continue;
        }
        /* 轮到黑方时左右镜像 (x -> 9-x): "己方"永远在 x 大的那一侧。
           镜像只有一份实现 (src/chessstate.h), 这里调它。 */
        const int cell = ChessState::canonicalCell(s->pos.x, s->pos.y, color);
        const int plane = s->type * 2 + ((s->color == color) ? 0 : 1);
        cells.push_back((std::uint16_t)(plane * CELLS + cell));
    }
}

void DQNMCTSMOETbAgent::expandSparse(const std::vector<std::uint16_t> &cells,
                                     RL::Tensor &state)
{
    state.zero();
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

void DQNMCTSMOETbAgent::denseToSparse(const RL::Tensor &state,
                                      std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    /* 只取**棋子平面**里非零的格: 上下文那 3 个槽不是"格子", 由 ctx[] 单独带。
       上界必须是 CTX_BASE 而不是 state.size() —— 否则会把上下文标量当成格子。 */
    const std::size_t limit = (state.size() < (std::size_t)CTX_BASE)
                                  ? state.size() : (std::size_t)CTX_BASE;
    for (std::size_t i = 0; i < limit; i++) {
        if (state[i] != 0.0f) {
            cells.push_back((std::uint16_t)i);
        }
    }
}

/*
 *  3 个规则上下文 —— 顺序与数值口径就是 `chessstate.h` 的 CTX_HALFMOVE /
 *  CTX_REPEAT / CTX_CHECK (与 SAC 的"改前表示"、以及 EVABAgent 的 plane 14..16
 *  逐个相同)。这一致性是"两支 agent 的状态同构"的全部内容: 只要有一边换了公式,
 *  对照就不再受控。
 */
void DQNMCTSMOETbAgent::contextOf(Chess &c, int color, float out[CTX_COUNT]) const
{
    out[0] = (float)ChessState::halfmovePhase(c);
    out[1] = (float)ChessState::repetitionPhase(c);
    out[2] = (float)ChessState::checkPhase(c, color);
}

void DQNMCTSMOETbAgent::writeContext(RL::Tensor &state, const float ctx[CTX_COUNT]) const
{
    if (state.size() < (std::size_t)STATE_DIM) {
        return;
    }
    for (int i = 0; i < CTX_COUNT; i++) {
        state[(std::size_t)(CTX_BASE + i)] = ctx[i];
    }
}

void DQNMCTSMOETbAgent::readContext(const RL::Tensor &state, float out[CTX_COUNT]) const
{
    for (int i = 0; i < CTX_COUNT; i++) {
        out[i] = 0.0f;
        if (state.size() >= (std::size_t)STATE_DIM) {
            out[i] = state[(std::size_t)(CTX_BASE + i)];
        }
    }
}

void DQNMCTSMOETbAgent::encodeStateFor(int color, RL::Tensor &state)
{
    if (state.size() != (std::size_t)STATE_DIM) {
        state = RL::Tensor(STATE_DIM, 1);
    }
    state.zero();
    std::vector<std::uint16_t> cells;
    encodeSparse(color, cells);
    expandSparse(cells, state);
    float ctx[CTX_COUNT];
    contextOf(chess, color, ctx);
    writeContext(state, ctx);
}

void DQNMCTSMOETbAgent::encodeState(RL::Tensor &state)
{
    /*
       视角由棋盘当前的 sideToMove 决定 —— 搜索里走法真的落在棋盘上
       (`moveForward` 会翻转 sideToMove), 所以这里读到的就是该节点的走棋方;
       `rolloutFromCurrent` 进来时也会把 sideToMove 设成 color, 探索阶段同样正确。
    */
    encodeStateFor(chess.sideToMove, state);
}

/* ================================================================
 *  动作: 128 槽哈希 (与 SAC 默认口径、以及旧类**逐字相同**的公式)
 * ================================================================
 *  为什么在 SAC 那条线上"要把 8100 双射换回来": 同一数据量下双射动作净亏约 200 Elo
 *  (200 局 71.8% vs 54.8%, 区间不重叠), 机制是 Q 头从 128 列涨到 8100 列 (参数量
 *  63 倍) 而瓶颈宽度不变 —— 参数/数据比失衡。所以这里沿用哈希, 别名问题交给自检
 *  面板如实报告 (`aliasMoves` / `aliasWorstSlot`), 而不是假装它不存在。
 * ================================================================ */
int DQNMCTSMOETbAgent::stepToActionIdx(const Step &s, int color) const
{
    (void)color;   /* 哈希按"棋子 id + 目标格", 与颜色无关 (照抄原口径) */
    return aliasActionIdxOf(s);
}

void DQNMCTSMOETbAgent::getLegalActions(int color,
                                        std::vector<Step*> &steps,
                                        std::vector<int> &actionIndices,
                                        RL::Tensor &actionMask)
{
    actionMask.zero();
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());
    for (std::size_t i = 0; i < steps.size(); i++) {
        const int aidx = stepToActionIdx(*steps[i], color);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

float DQNMCTSMOETbAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    /*
       不检查 victim->alive: 调用方常在 moveForward() **之后**求奖励, 那时被吃子已经
       alive=false, 加判断会让吃子奖励恒为 0 (pgagent.cpp 里有同一处的说明)。
    */
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

float DQNMCTSMOETbAgent::terminalReward(int chessResult, int mover) const
{
    if (chessResult == Chess::RESULT_ONGOING) {
        return 0.0f;
    }
    return outcomeForMover(chessResult, mover);
}

bool DQNMCTSMOETbAgent::terminalValueOf(int color, double &value) const
{
    const int res = const_cast<Chess &>(chess).getResult(color);
    if (res == Chess::RESULT_ONGOING) {
        return false;
    }
    value = (double)terminalReward(res, color);
    return true;
}

/* ================================================================
 *  前向: 稀疏叶子路径 / 全量回退
 * ================================================================ */

bool DQNMCTSMOETbAgent::sparseQGeneric(const RL::Tensor &state,
                                       const std::vector<int> &legalIdx,
                                       std::vector<float> &qOut,
                                       std::vector<float> *q2Out,
                                       RL::Net &trunkRef,
                                       RL::Net &headRef,
                                       RL::Net *head2Ref)
{
    if (legalIdx.empty()) {
        qOut.clear();
        if (q2Out != nullptr) {
            q2Out->clear();
        }
        return true;
    }
    /* 骨干**只前向一次**, 三个头 (或两个) 共用这一个 h */
    RL::Tensor &h = trunkRef.forward(state);
    if (!headRef.sparseLogits(h, legalIdx, qOut)) {
        return false;
    }
    if (q2Out != nullptr) {
        if (head2Ref == nullptr || !head2Ref->sparseLogits(h, legalIdx, *q2Out)) {
            return false;
        }
    }
    return true;
}

bool DQNMCTSMOETbAgent::sparseQOnline(const RL::Tensor &state,
                                      const std::vector<int> &legalIdx,
                                      std::vector<float> &qOut,
                                      std::vector<float> *q2Out)
{
    return sparseQGeneric(state, legalIdx, qOut, q2Out, trunk, qHead,
                          (q2Out != nullptr) ? &q2Head : nullptr);
}

bool DQNMCTSMOETbAgent::sparseQTarget(const RL::Tensor &state,
                                      const std::vector<int> &legalIdx,
                                      std::vector<float> &qOut,
                                      std::vector<float> *q2Out)
{
    return sparseQGeneric(state, legalIdx, qOut, q2Out, trunkTarget, qHeadTarget,
                          (q2Out != nullptr) ? &q2HeadTarget : nullptr);
}

bool DQNMCTSMOETbAgent::qValuesFull(const RL::Tensor &state, RL::Tensor &qOut,
                                    RL::Tensor *q2Out)
{
    RL::Tensor &h = trunk.forward(state);
    qOut = qHead.forward(h);          /* 深拷贝: 头返回的是它自己的 o 缓冲 */
    if (q2Out != nullptr) {
        *q2Out = q2Head.forward(h);
    }
    return true;
}

void DQNMCTSMOETbAgent::legalPriors(const std::vector<float> &qLegal,
                                    std::vector<double> &priorsOut) const
{
    priorsOut.assign(qLegal.size(), 0.0);
    if (qLegal.empty()) {
        return;
    }
    const double t = (priorTemp > 1e-6f) ? (double)priorTemp : 1.0;
    double mx = -1e30;
    for (std::size_t i = 0; i < qLegal.size(); i++) {
        mx = std::max(mx, (double)qLegal[i]);
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < qLegal.size(); i++) {
        const double e = std::exp(((double)qLegal[i] - mx) / t);
        priorsOut[i] = e;
        sum += e;
    }
    if (sum > 0.0) {
        for (std::size_t i = 0; i < priorsOut.size(); i++) {
            priorsOut[i] /= sum;
        }
    } else {
        const double u = 1.0 / (double)qLegal.size();
        for (std::size_t i = 0; i < priorsOut.size(); i++) {
            priorsOut[i] = u;
        }
    }
}

double DQNMCTSMOETbAgent::leafValueFrom(const std::vector<float> &q,
                                        const std::vector<float> *q2) const
{
    double best = -1e30;
    for (std::size_t i = 0; i < q.size(); i++) {
        double v = (double)q[i];
        if (q2 != nullptr && i < q2->size()) {
            v = std::min(v, (double)(*q2)[i]);
        }
        if (v > best) {
            best = v;
        }
    }
    return (best <= -1e29) ? 0.0 : best;
}

/* ================================================================
 *  搜索: PUCT (带 negamax 负号) + 稀疏叶子估值
 * ================================================================ */

double DQNMCTSMOETbAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[(std::size_t)childID];
    if (child.visitCount == 0) {
        /* 未访问过的孩子优先 (AlphaZero 的 PUCT 里 U 项在 N=0 时最大) */
        return std::numeric_limits<double>::max();
    }
    /*
       符号: `totalValue` 按**当前走棋方视角**累计 (见 AZNode 的字段注释与 backup 的
       逐层翻号), 而子节点的走棋方就是父节点的对手 —— 所以父节点比较时必须取负号。
       漏掉它等于最大化对手的价值: 搜索专挑对自己最差的着法, 而且评估越准越糟
       (症状是"损失降、棋力不涨", 以及不敢吃子 —— 吃子后对手少大子, 子节点的 Q 对
       对手为负, 被算成亏着)。与 `SACAZMoETbAgent::getPUCT` / `PPOMCTSAgent::getPUCT` /
       `MCTS::getUCB1` 四处同一口径 —— 这几个 agent 的符号约定必须一致。
       ⚠ 旧类 `DQNMCTSAgent::getUCB1` 用的是 `+child.totalReward/visitCount`, 而它的
       backup 同样逐层翻号 —— 那一支的 exploitation 项是反的 (见 docs 的说明)。
    */
    const double q = -child.getQ();
    const double u = c_puct * child.prior
                     * std::sqrt((double)parentVisits)
                     / (1.0 + (double)child.visitCount);
    return q + u;
}

void DQNMCTSMOETbAgent::visitDistribution(int rootID, RL::Tensor &pi) const
{
    pi.zero();
    int total = 0;
    for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
        total += nodes[(std::size_t)nodes[(std::size_t)rootID].childIDs[i]].visitCount;
    }
    if (total <= 0) {
        return;
    }
    const float inv = 1.0f / (float)total;
    for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
        const AZNode &c = nodes[(std::size_t)nodes[(std::size_t)rootID].childIDs[i]];
        pi[c.parentAction] += (float)c.visitCount * inv;
    }
}

Step DQNMCTSMOETbAgent::selectMove(int color, int simulations_, float temp)
{
    nodes.clear();
    int sims = (simulations_ > 0) ? simulations_ : simulations;
    if (sims < 1) {
        sims = 1;
    }
    nodes.reserve((std::size_t)sims + 64);

    RL::Tensor &mask = m_maskBuf;
    std::vector<Step*> rootSteps;
    std::vector<int> rootIdx;
    getLegalActions(color, rootSteps, rootIdx, mask);

    /*
       自检计数: 记下**这个局面**的合法集与它用到几个 Q 槽位。必须在 `Steps::put()`
       之前算 —— put() 会把 Step 还回对象池, 之后读到的是已被复用的内容。
       每手只算一次 (不是每个模拟一次): 迭代里做这件事要给每个孩子分配 map/set/字符串,
       在几十次模拟下已经是可观测的常数开销。
    */
    {
        int legalN = 0, slotN = 0, worst = 0;
        aliasOfPosition(rootSteps, legalN, slotN, worst);
        aliasMoves += legalN;
        aliasIndexed += slotN;
        aliasClearedMoves += (legalN - slotN);
        if (worst > aliasWorstSlot) {
            aliasWorstSlot = worst;
        }
    }

    AZNode root;
    root.currentColor = color;
    root.legalCount = (int)rootIdx.size();
    root.parentID = -1;
    root.parentAction = -1;

    /* 根局面的先验: 合法槽位上 Q 的 softmax (见头文件: DQN 没有策略头) */
    if (root.legalCount > 0) {
        encodeStateFor(color, m_stateBuf);
        std::vector<float> &qL = m_qLegal;
        std::vector<float> &q2L = m_q2Legal;
        std::vector<float> *q2Ptr = twinCritic ? &q2L : nullptr;
        bool sparseOk = sparseLeafEval && sparseQOnline(m_stateBuf, rootIdx, qL, q2Ptr);
        if (!sparseOk) {
            /* 根的先验也要算 (这一步不算"叶子估值", 所以不记进 m_leafEvals) */
            std::vector<float> q1v, q2v;
            if (qValuesFull(m_stateBuf, m_qBuf, twinCritic ? &m_q2Buf : nullptr)) {
                qL.clear();
                qL.reserve(rootIdx.size());
                if (q2Ptr != nullptr) {
                    q2v.clear();
                    q2v.reserve(rootIdx.size());
                }
                for (std::size_t i = 0; i < rootIdx.size(); i++) {
                    qL.push_back(m_qBuf[rootIdx[i]]);
                    if (q2Ptr != nullptr) {
                        q2v.push_back(m_q2Buf[rootIdx[i]]);
                    }
                }
                if (q2Ptr != nullptr) {
                    q2L = q2v;
                }
            } else {
                qL.assign(rootIdx.size(), 0.0f);
                if (q2Ptr != nullptr) {
                    q2L.assign(rootIdx.size(), 0.0f);
                }
            }
        }
        if (q2Ptr != nullptr) {
            /* 双 critic: 先验与叶子都用 min(Q1,Q2) */
            for (std::size_t i = 0; i < qL.size() && i < q2L.size(); i++) {
                qL[i] = std::min(qL[i], q2L[i]);
            }
        }
        legalPriors(qL, m_priors);
    }

    for (std::size_t i = 0; i < rootIdx.size(); i++) {
        root.untriedActionIndices.push_back(rootIdx[i]);
        root.untriedSteps.push_back(*rootSteps[i]);
        root.untriedPriors.push_back((i < m_priors.size()) ? m_priors[i] : 0.0);
    }

    /*
       在线训练口径: 缓存"走子前"的局面 (recordExperience 要用)。
       缓存的是**稀疏格 + 上下文 + 合法位图**, 不是 1263 个 float。
       ⚠ 只有训练回路 (m_trainingMode) 走这条路; 界面决策路径的样本由
       `learnFromSearchStep` **自己现编码** —— 它读不到这份缓存, 也不该读 (见那里的说明)。
    */
    if (m_trainingMode) {
        encodeSparse(color, m_cachedCells);
        contextOf(chess, color, m_cachedCtx);
        m_cachedLegal = (int)rootIdx.size();
        m_cachedMask[0] = 0;
        m_cachedMask[1] = 0;
        for (std::size_t i = 0; i < rootIdx.size(); i++) {
            bitSetTo(m_cachedMask, rootIdx[i]);
        }
        m_haveCached = true;
    }

    Steps::instance().put(rootSteps);

    if (root.legalCount == 0) {
        /* 无合法走法: 返回无效 Step, 由调用方按"真无棋可走"处理 */
        return Step();
    }

    nodes.push_back(root);
    const int rootID = 0;

    /* ---- 主循环 ---- */
    for (int sim = 0; sim < sims; sim++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* --- 1. SELECT (PUCT 下行) --- */
        std::vector<Step> toApply;
        while (nodes[(std::size_t)nodeID].untriedActionIndices.empty()
               && !nodes[(std::size_t)nodeID].childIDs.empty()) {
            int bestChild = -1;
            double bestPuct = -std::numeric_limits<double>::max();
            for (std::size_t ci = 0;
                 ci < nodes[(std::size_t)nodeID].childIDs.size(); ci++) {
                const int childID = nodes[(std::size_t)nodeID].childIDs[ci];
                const double p = getPUCT(childID, nodes[(std::size_t)nodeID].visitCount);
                if (p > bestPuct) {
                    bestPuct = p;
                    bestChild = childID;
                }
            }
            if (bestChild < 0) {
                break;
            }
            toApply.push_back(nodes[(std::size_t)bestChild].step);
            nodeID = bestChild;
            path.push_back(nodeID);
        }
        double dummy = 0.0;
        for (std::size_t i = 0; i < toApply.size(); i++) {
            chess.moveForward(&toApply[i], dummy);
        }

        double leafValue = 0.0;
        bool haveLeafValue = false;

        /* 到达的节点本身可能已经终局 (走到这里时棋盘就是该节点的局面) */
        if (terminalValueOf(nodes[(std::size_t)nodeID].currentColor, leafValue)) {
            haveLeafValue = true;
        }

        /* --- 2. EXPANSION (按先验挑一个未试过的着法) --- */
        if (!haveLeafValue && !nodes[(std::size_t)nodeID].untriedActionIndices.empty()) {
            std::size_t pickIdx = 0;
            for (std::size_t i = 1;
                 i < nodes[(std::size_t)nodeID].untriedPriors.size(); i++) {
                if (nodes[(std::size_t)nodeID].untriedPriors[i]
                    > nodes[(std::size_t)nodeID].untriedPriors[pickIdx]) {
                    pickIdx = i;
                }
            }
            const int chosenAction =
                nodes[(std::size_t)nodeID].untriedActionIndices[pickIdx];
            const Step chosenStep = nodes[(std::size_t)nodeID].untriedSteps[pickIdx];
            const double chosenPrior =
                nodes[(std::size_t)nodeID].untriedPriors[pickIdx];
            nodes[(std::size_t)nodeID].untriedActionIndices.erase(
                nodes[(std::size_t)nodeID].untriedActionIndices.begin() + (long)pickIdx);
            nodes[(std::size_t)nodeID].untriedSteps.erase(
                nodes[(std::size_t)nodeID].untriedSteps.begin() + (long)pickIdx);
            nodes[(std::size_t)nodeID].untriedPriors.erase(
                nodes[(std::size_t)nodeID].untriedPriors.begin() + (long)pickIdx);

            chess.moveForward(&chosenStep, dummy);

            const int nextColor = (nodes[(std::size_t)nodeID].currentColor == Stone::COLOR_RED)
                                      ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            std::vector<Step*> childSteps;
            std::vector<int> childIdx;
            getLegalActions(nextColor, childSteps, childIdx, mask);

            AZNode child(nodeID, chosenAction, chosenStep, chosenPrior, nextColor,
                         (int)childIdx.size());

            double v = 0.0;
            if (terminalValueOf(nextColor, v)) {
                /* 被将杀 / 困毙 / 判和: 终局节点, 价值取自真实胜负 */
                child.isTerminal = true;
                leafValue = v;
            } else {
                encodeStateFor(nextColor, m_stateBuf);
                std::vector<float> &qL = m_qLegal;
                std::vector<float> &q2L = m_q2Legal;
                std::vector<float> *q2Ptr = twinCritic ? &q2L : nullptr;
                bool sparseOk = sparseLeafEval
                                && sparseQOnline(m_stateBuf, childIdx, qL, q2Ptr);
                if (!sparseOk) {
                    /*
                       回退全量口径 (语义与稀疏逐元素相同, 只是慢)。**必须回退**:
                       稀疏头支持的判据不满足时(下标越界/头不支持), 静默算错比慢得多。
                       与 RL::PPO 的 R1 同一条纪律: "加了新层类型却忘了实现时, 行为是
                       慢而不是错" —— 所以内核默认返回 false, 由调用方兜底。
                    */
                    m_fullLeafEvals++;
                    qValuesFull(m_stateBuf, m_qBuf, twinCritic ? &m_q2Buf : nullptr);
                    std::vector<float> q1v;
                    q1v.reserve(childIdx.size());
                    for (std::size_t i = 0; i < childIdx.size(); i++) {
                        q1v.push_back(m_qBuf[childIdx[i]]);
                    }
                    qL = q1v;
                    if (q2Ptr != nullptr) {
                        std::vector<float> q2v;
                        q2v.reserve(childIdx.size());
                        for (std::size_t i = 0; i < childIdx.size(); i++) {
                            q2v.push_back(m_q2Buf[childIdx[i]]);
                        }
                        q2L = q2v;
                    }
                }
                if (q2Ptr != nullptr) {
                    for (std::size_t i = 0; i < qL.size() && i < q2L.size(); i++) {
                        qL[i] = std::min(qL[i], q2L[i]);
                    }
                }
                legalPriors(qL, m_priors);
                for (std::size_t i = 0; i < childIdx.size(); i++) {
                    child.untriedActionIndices.push_back(childIdx[i]);
                    child.untriedSteps.push_back(*childSteps[i]);
                    child.untriedPriors.push_back((i < m_priors.size()) ? m_priors[i]
                                                                        : 0.0);
                }
                leafValue = leafValueFrom(qL, nullptr);
            }
            m_leafEvals++;
            Steps::instance().put(childSteps);

            nodes.push_back(child);
            const int newID = (int)nodes.size() - 1;
            nodes[(std::size_t)nodeID].childIDs.push_back(newID);
            nodeID = newID;
            path.push_back(newID);
            haveLeafValue = true;
        }

        /* --- 3. 已全展开的节点: 就地用叶子价值 (终局已在上面处理) --- */
        if (!haveLeafValue) {
            std::vector<Step*> ls;
            std::vector<int> li;
            getLegalActions(nodes[(std::size_t)nodeID].currentColor, ls, li, mask);
            Steps::instance().put(ls);
            encodeStateFor(nodes[(std::size_t)nodeID].currentColor, m_stateBuf);
            std::vector<float> &qL = m_qLegal;
            std::vector<float> &q2L = m_q2Legal;
            std::vector<float> *q2Ptr = twinCritic ? &q2L : nullptr;
            bool sparseOk = sparseLeafEval && sparseQOnline(m_stateBuf, li, qL, q2Ptr);
            if (!sparseOk) {
                m_fullLeafEvals++;
                qValuesFull(m_stateBuf, m_qBuf, twinCritic ? &m_q2Buf : nullptr);
                std::vector<float> q1v;
                q1v.reserve(li.size());
                for (std::size_t i = 0; i < li.size(); i++) {
                    q1v.push_back(m_qBuf[li[i]]);
                }
                qL = q1v;
                if (q2Ptr != nullptr) {
                    std::vector<float> q2v;
                    q2v.reserve(li.size());
                    for (std::size_t i = 0; i < li.size(); i++) {
                        q2v.push_back(m_q2Buf[li[i]]);
                    }
                    q2L = q2v;
                }
            }
            if (q2Ptr != nullptr) {
                for (std::size_t i = 0; i < qL.size() && i < q2L.size(); i++) {
                    qL[i] = std::min(qL[i], q2L[i]);
                }
            }
            leafValue = leafValueFrom(qL, nullptr);
            m_leafEvals++;
        }

        /* --- 4. BACKUP (negamax: 逐层翻号) --- */
        double v = leafValue;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[(std::size_t)path[(std::size_t)i]].visitCount++;
            nodes[(std::size_t)path[(std::size_t)i]].totalValue += v;
            v = -v;
        }

        /* --- 5. 撤销本轮试走 (path[0] 是根, 没有对应的 step) --- */
        for (std::size_t i = path.size(); i > 1; i--) {
            const Step &s = nodes[(std::size_t)path[i - 1]].step;
            chess.moveBack(&s, dummy);
        }
    }

    /* ---- 选择走法 ---- */
    int bestChildID = -1;
    if (temp <= 1e-6f) {
        int maxVisits = -1;
        for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
            const int childID = nodes[(std::size_t)rootID].childIDs[i];
            if (nodes[(std::size_t)childID].visitCount > maxVisits) {
                maxVisits = nodes[(std::size_t)childID].visitCount;
                bestChildID = childID;
            }
        }
    } else {
        /*
           按 N^(1/T) 采样访问分布 (自对弈用, 保证开局多样性 —— AlphaZero 的根温度)。
           为什么不是"以 ε 概率随机挑一个孩子": 那个孩子可能**一次都没被访问过**,
           它的价值与先验都没有被搜索检验过, 等于在那一步把搜索丢掉。旧类那一支
           就是这么做的。
        */
        RL::Tensor dist(ACTION_DIM, 1);
        dist.zero();
        double sum = 0.0;
        const double invT = 1.0 / (double)temp;
        for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
            const int childID = nodes[(std::size_t)rootID].childIDs[i];
            const double w = std::pow((double)nodes[(std::size_t)childID].visitCount, invT);
            dist[nodes[(std::size_t)childID].parentAction] += (float)w;
            sum += w;
        }
        if (sum > 0.0) {
            const int a = RL::Random::categorical(dist);
            for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
                const int childID = nodes[(std::size_t)rootID].childIDs[i];
                if (nodes[(std::size_t)childID].parentAction == a) {
                    bestChildID = childID;
                    break;
                }
            }
        }
        if (bestChildID < 0) {
            int maxVisits = -1;
            for (std::size_t i = 0; i < nodes[(std::size_t)rootID].childIDs.size(); i++) {
                const int childID = nodes[(std::size_t)rootID].childIDs[i];
                if (nodes[(std::size_t)childID].visitCount > maxVisits) {
                    maxVisits = nodes[(std::size_t)childID].visitCount;
                    bestChildID = childID;
                }
            }
        }
    }

    if (bestChildID >= 0) {
        const Step chosen = nodes[(std::size_t)bestChildID].step;
        const int chosenAction = nodes[(std::size_t)bestChildID].parentAction;
        /*
           ---- 从自己的搜索学一次 (SAC 的 learnFromSearch) ----
           位置刻意放在这里: 棋盘已经恢复成根局面 (上面第 5 步把试走的都回退了),
           所以这个函数可以"试走一手再退回"拿 r 与 s'。训练回路
           (trainSelfPlay / trainVsRandom / warmup) 自己记样本, 由 m_inTrainLoop 排除。
           `rootIdx` 是根本局的合法动作下标 —— 样本的"当前局面"由它在根局面上现编码,
           不读任何缓存 (见头文件那条说明)。
        */
        if (learnFromSearch && !m_inTrainLoop) {
            learnFromSearchStep(color, chosen, chosenAction, rootIdx);
        }
        return chosen;
    }

    /*
       根有合法走法却没选出一个孩子 (例如一次都没展开): 兜底取根节点未展开的第一手,
       **不要**返回 Step() —— 那会让调用方把"还有棋可下"读成"无棋可走"并判负
       (与 ABAgent 的"全负不返回走法"同类)。
    */
    if (!nodes[(std::size_t)rootID].untriedSteps.empty()) {
        return nodes[(std::size_t)rootID].untriedSteps.front();
    }
    return Step();
}

Step DQNMCTSMOETbAgent::getBestMove(int color)
{
    return selectMove(color, 0, 0.0f);
}

std::string DQNMCTSMOETbAgent::getName() const
{
    return std::string("DQN+MCTS (") + backboneName() + ", c_puct="
           + std::to_string(c_puct) + ")";
}

/*
 *  界面"训练损失曲线"取这一份 (见 aiagent.h 的 getLastTrainLoss)。
 *  **口径提醒** (与 SAC 那份文档同一条): 这条曲线**不能**读成"学得好不好" ——
 *  实测过的两个陷阱是 "损失在下但 critic 没学到排序" 与 "损失很小其实是 critic
 *  根本没动 (|Q| 只有随机初始化尺度)"。真正的判据在自检面板的 |y| / 夹住比例 /
 *  Q spread / |Q_target| 那几个读数上。NaN = 还没学过 (曲线对非有限值是直接丢点的)。
 */
float DQNMCTSMOETbAgent::getLastTrainLoss() const
{
    if (!(m_lastLoss == m_lastLoss)) {   /* NaN 检查 (不用 std::isnan, 免得拉进 <cmath> 的分支) */
        return std::numeric_limits<float>::quiet_NaN();
    }
    return (float)m_lastLoss;
}

/* ================================================================
 *  回放池 / 终局通道
 * ================================================================ */

void DQNMCTSMOETbAgent::pushSample(const Sample &s)
{
    memories.push_back(s);
    if (s.decision) {
        m_lastDecisionFromBack = 0;
    } else if (m_lastDecisionFromBack >= 0) {
        m_lastDecisionFromBack++;
    }
    const int bs = (batchSize > 0) ? batchSize : 1;
    if (memories.size() > maxMemorySize + (std::size_t)bs) {
        const std::size_t k = std::min((std::size_t)bs, memories.size() - maxMemorySize);
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
            /* 淘汰只从队首弹出, 所以"从队尾数的偏移"不受影响; 只有当它已经被弹掉
               (整个池子比那个偏移还短) 时才失效。 */
            if (m_lastDecisionFromBack >= (long long)memories.size()) {
                m_lastDecisionFromBack = -1;
            }
        }
    }
}

int DQNMCTSMOETbAgent::attachTerminalToLastDecision(float terminalValue)
{
    if (m_lastDecisionFromBack < 0
        || (std::size_t)m_lastDecisionFromBack >= memories.size()) {
        return -1;      /* 池里没有真实决策样本 (这一局它一步没走过) */
    }
    Sample &s = memories[memories.size() - 1 - (std::size_t)m_lastDecisionFromBack];
    if (s.done) {
        return 0;       /* 幂等: 这一局的终局已经写过了, 不重复写 */
    }
    s.done = true;
    s.reward = terminalValue;
    /*
       终局是**这一手之后**发生的, 所以 s' 的合法集在这条样本里已经不再重要
       (done=true 时自举项被 γ(1−done) 归零)。这里刻意不动 nextMask —— 改它反而
       会让"这条样本的 s' 是什么"变得不可解释。
    */
    return 1;
}

bool DQNMCTSMOETbAgent::notifyGameResult(int chessResult, int perspective)
{
    if (chessResult == Chess::RESULT_ONGOING) {
        return false;
    }
    /*
       走**同一个**终局出口 (terminalReward): 界面给的是 Chess::Result + 视角方,
       与 agent 自己算终局时用的是同一个函数 —— 差一个符号就会让"人把 AI 将死"
       变成给学习器发 +1。
    */
    const float v = terminalReward(chessResult, perspective);
    const int r = attachTerminalToLastDecision(v);
    if (r < 0) {
        return false;   /* 没有可挂的真实决策样本: 如实返回 false, 调用方会打日志 */
    }
    if (r == 1) {
        externalTerminals++;   /* 只统计**真的写进去**的次数 (重复通知不重复计) */
        /*
           终局样本留在池里, **不额外强制一次 learnBatch**: notifyHumanGameEnd 是在 GUI
           线程上持锁调用的, 而 TB 骨干一个批就是两百多毫秒。按 learnEveryMoves 的节拍
           走, 这条样本不会被丢 (池子按 FIFO 淘汰, 4096 条以内一定还在)。
        */
        maybeLearn();
    }
    return true;
}

void DQNMCTSMOETbAgent::noteEnd(int result, bool seenByGameOver)
{
    const int code = endCodeOfResult(result);
    if (code >= 0 && code < 4) {
        endCount[code]++;
    }
    if (seenByGameOver) {
        endSeenByGameOver++;
    }
}

/* ================================================================
 *  学习: 一次 mini-batch
 * ================================================================
 *  与 SAC 的 learnBatch 同一套骨架 (P3 梯度累积 + P4 多 epoch):
 *    每抽一条样本 → 目标侧前向 → 当前侧前向一次 → 头反向 → **骨干反向一次**
 *    (梯度在骨干处累积) → 批末 MoE 辅助损失 → 优化器只调一次 → 目标网同步。
 * ================================================================ */
float DQNMCTSMOETbAgent::learnBatch(int batchSize_, int epochs)
{
    if (batchSize_ < 1 || (int)memories.size() < batchSize_) {
        return 0.0f;
    }
    if (epochs <= 0) {
        epochs = (replayEpochs > 0) ? replayEpochs : 1;
    }
    if (epochs < 1) {
        epochs = 1;
    }

    const bool twin = twinCritic;
    RL::Tensor state(STATE_DIM, 1);
    RL::Tensor nextState(STATE_DIM, 1);
    RL::Tensor dq(ACTION_DIM, 1);
    RL::Tensor target(ACTION_DIM, 1);
    std::vector<int> nextIdx;
    nextIdx.reserve(64);
    std::vector<float> qn, q2n, qc, q2c, qt, q2t;

    /*
       [MoE] 批统计的**边界**: 只反映本批的训练前向。搜索期间每次模拟都跑一次骨干
       前向 (那是推理前向), 不划边界的话辅助损失会被整局棋的推理前向稀释
       (还有 xSum 这种 float 累加器在几十万次累加后的精度损失)。
    */
    resetMoeBatchStats();

    batchDiag = BatchDiag();
    m_maxAbsTarget = 0.0;
    m_maxAbsTdErr = 0.0;
    float lossSum = 0.0f;
    int n = 0;

    /* [P4] 每个 epoch **重新抽** batchSize 条 (不是把同一批复用几遍, 理由见头文件) */
    std::uniform_int_distribution<int> pick(0, (int)memories.size() - 1);

    for (int ep = 0; ep < epochs; ep++) {
        for (int it = 0; it < batchSize_; it++) {
            const Sample &tr = memories[(std::size_t)pick(RL::Random::engine)];

            expandSparse(tr.cells, state);
            writeContext(state, tr.ctx);
            expandSparse(tr.nextCells, nextState);
            writeContext(nextState, tr.nextCtx);

            /* s' 的合法槽位 (位图为空 = 没有掩码: 终局样本或采样失败) */
            nextIdx.clear();
            const bool hasNextMask = !bitsEmpty(tr.nextMask);
            if (hasNextMask) {
                for (int a = 0; a < ACTION_DIM; a++) {
                    if (bitSet(tr.nextMask, a)) {
                        nextIdx.push_back(a);
                    }
                }
            }

            /* ---- 目标侧 (done 时不需要任何前向) ---- */
            double vNext = 0.0;
            double qTargetAbs = 0.0;
            bool usedDouble = false;
            float y = tr.reward;
            if (!tr.done) {
                /*
                   Double DQN: 用**在线**网在 s' 的合法集上选动作 a*, 用**目标**网估它的值。
                   旧类那一支是 vanilla DQN (argmax 与取值都来自目标网), 而"选动作"与
                   "估价值"用同一张网正是过估计的来源 —— SAC 的 min(Q1,Q2) 要解决的
                   是同一个问题, 这里用不增加任何参数的那一半。
                */
                bool okOnline = false;
                bool okTarget = false;
                double aStarValForSelect = 0.0;
                int aStar = -1;
                if (hasNextMask) {
                    std::vector<float> *q2Ptr = twin ? &q2n : nullptr;
                    okOnline = sparseLeafEval
                               && sparseQOnline(nextState, nextIdx, qn, q2Ptr);
                    if (!okOnline) {
                        qValuesFull(nextState, m_qBuf, twin ? &m_q2Buf : nullptr);
                        qn.clear();
                        qn.reserve(nextIdx.size());
                        if (twin) {
                            q2n.clear();
                            q2n.reserve(nextIdx.size());
                        }
                        for (std::size_t i = 0; i < nextIdx.size(); i++) {
                            qn.push_back(m_qBuf[nextIdx[i]]);
                            if (twin) {
                                q2n.push_back(m_q2Buf[nextIdx[i]]);
                            }
                        }
                    }
                    for (std::size_t i = 0; i < qn.size(); i++) {
                        double v = (double)qn[i];
                        if (twin && i < q2n.size()) {
                            v = std::min(v, (double)q2n[i]);
                        }
                        if (aStar < 0 || v > aStarValForSelect) {
                            aStarValForSelect = v;
                            aStar = nextIdx[i];
                        }
                    }
                } else {
                    /* 没有掩码: 退回全列 argmax (只在采样失败时发生) */
                    qValuesFull(nextState, m_qBuf, twin ? &m_q2Buf : nullptr);
                    aStar = 0;
                    aStarValForSelect = (double)m_qBuf[0];
                    for (int a = 1; a < ACTION_DIM; a++) {
                        double v = (double)m_qBuf[a];
                        if (twin) {
                            v = std::min(v, (double)m_q2Buf[a]);
                        }
                        if (v > aStarValForSelect) {
                            aStarValForSelect = v;
                            aStar = a;
                        }
                    }
                }
                if (aStar >= 0) {
                    std::vector<int> oneIdx(1, aStar);
                    std::vector<float> *q2Ptr = twin ? &q2t : nullptr;
                    okTarget = sparseLeafEval
                               && sparseQTarget(nextState, oneIdx, qt, q2Ptr);
                    if (!okTarget) {
                        /* 回退: 目标网全量前向 (目标头是独立的一张网, 不会碰在线网的缓存) */
                        RL::Tensor &hT = trunkTarget.forward(nextState);
                        RL::Tensor qT = qHeadTarget.forward(hT);
                        qt.assign(1, qT[aStar]);
                        if (twin) {
                            RL::Tensor q2T = q2HeadTarget.forward(hT);
                            q2t.assign(1, q2T[aStar]);
                        }
                    }
                    double vT = (double)qt[0];
                    if (twin && !q2t.empty()) {
                        vT = std::min(vT, (double)q2t[0]);
                    }
                    qTargetAbs = std::fabs(vT);
                    vNext = vT;   /* 诊断: 自举项的取值 (V(s') = 目标网上的 Q) */
                    /*
                       符号: 所有价值都是"该局面走棋方视角"。s' 轮到**对手**走, 所以
                       自举项取负 (negamax):  y = r − γ(1−done)·Q_target(s',a*)
                    */
                    y = tr.reward - gamma * (float)vT;
                    usedDouble = doubleDQN;
                }
            }

            /*
               ---- 值域约束 (从 SAC 搬过来的两条之一) ----
               真实 Q 必然落在 [-1.5, 1.5] 量级内 (即时奖励上界 0.35 = 材质系数 x 一方
               满子, 终局 ±1), 但自举把 Q 反复回代, 而优化器没有任何把 Q 拉回该区间的
               机制 ⇒ SAC 那边实测发散 |Q| 0.063 -> 4.15 (40 局) -> 13.4 (150 局),
               并把搜索的 PUCT 打坏 (对 MCTS 得分率 73.3% -> 32.5%)。
               这里夹的是**目标**不是奖励 —— 不改变各着法的排序, 只挡住发散。
            */
            const float yClamped = (clampTarget > 0.0f)
                                       ? std::min(std::max(y, -clampTarget), clampTarget)
                                       : y;
            m_maxAbsTarget = std::max(m_maxAbsTarget, std::fabs((double)yClamped));
            batchDiag.yPreAbsSum += std::fabs((double)y);
            batchDiag.ySum += (double)y;
            if (std::fabs((double)y) > batchDiag.yPreAbsMax) {
                batchDiag.yPreAbsMax = std::fabs((double)y);
            }
            if (clampTarget > 0.0f && std::fabs((double)y) > (double)clampTarget) {
                batchDiag.clamped++;
            }
            batchDiag.vNextSum += vNext;

            /* ---- 当前侧: 骨干前向**一次** (这一次就是后面反向要用的那次) ---- */
            RL::Tensor &h = trunk.forward(state);
            RL::Tensor q1o = qHead.forward(h);
            RL::Tensor q2o;
            if (twin) {
                q2o = q2Head.forward(h);
            }

            const int aidx = (tr.action >= 0 && tr.action < ACTION_DIM) ? tr.action : 0;
            const float qTaken = twin ? std::min(q1o[aidx], q2o[aidx]) : q1o[aidx];
            const float err = qTaken - yClamped;
            const double ae = std::fabs((double)err);
            m_maxAbsTdErr = std::max(m_maxAbsTdErr, ae);
            /*
               损失数值走 Huber (超出 delta 转线性: 单个离群样本不会再把整批的平均方向
               带跑), 梯度仍走 `RL::Loss::MSE::df` —— 它在 |err| <= delta 时与 Huber
               逐位相同。**不要**"顺手统一"成纯 MSE: SAC 那边专门留了这条注释。
            */
            lossSum += (float)((huberDelta > 0.0f && ae > (double)huberDelta)
                                   ? (double)huberDelta * (ae - 0.5 * (double)huberDelta)
                                   : 0.5 * ae * ae);

            /*
               ---- 反向: 先头后骨干 (硬约束) ----
               1. 头的 `backward` 读的是骨干缓存下来的 h, 而 `Layer<Tanh>::backward`
                  结束时会把 o 清掉 ⇒ 所有头必须在骨干反向**之前**跑;
               2. 骨干的梯度 = 各头 `inputGrad` 之**和**, 不是"反向三次" ——
                  骨干只有一份, 分三次反向 = 三次更新同一份权重 = 学习率 x3。
            */
            target = q1o;
            target[aidx] = yClamped;
            dq = RL::Loss::MSE::df(q1o, target);
            qHead.backward(h, dq);
            RL::Tensor gh = qHead.inputGrad;
            if (twin) {
                target = q2o;
                target[aidx] = yClamped;
                dq = RL::Loss::MSE::df(q2o, target);
                q2Head.backward(h, dq);
                gh += q2Head.inputGrad;
            }
            trunk.backward(state, gh);

            /* ---- 诊断 (只读, 不进任何梯度/更新路径) ---- */
            {
                /*
                   Q spread: **只在当前局面的合法槽位上**算 (与 SAC 的读数同口径)。
                   为什么要"只统计合法列": 128 个槽位里绝大多数在任何局面下都不可达,
                   把它们算进标准差会把读数稀释 (假性偏小), 而这个数的用途是判断
                   "critic 有没有给出排序信号" —— PUCT 的探索项量级是 0.3~1.0, 只有
                   Q spread 到了同一量级, Q 才影响得了选择。
                   `curMask` 为空 (老样本/采样失败) 时退回"只统计实际走的那一列"。
                */
                const bool hasCurMask = !bitsEmpty(tr.curMask);
                double slots = 0.0, qm = 0.0, qs2 = 0.0;
                for (int a = 0; a < ACTION_DIM; a++) {
                    if (hasCurMask) {
                        if (!bitSet(tr.curMask, a)) {
                            continue;
                        }
                    } else if (a != aidx) {
                        continue;
                    }
                    const double qq = (double)q1o[a];
                    slots += 1.0;
                    qm += qq;
                    qs2 += qq * qq;
                }
                double spread = 0.0;
                if (slots > 0.0) {
                    qm /= slots;
                    const double var = qs2 / slots - qm * qm;
                    spread = (var > 0.0) ? std::sqrt(var) : 0.0;
                }
                batchDiag.qSpreadSum += spread;
                batchDiag.qAbsMeanSum += std::fabs(qm);
                batchDiag.qTargetAbsSum += qTargetAbs;
                batchDiag.slotSum += slots;
                batchDiag.legalSum += (double)((tr.legalCount > 1) ? tr.legalCount : 2);
                if (tr.done) {
                    batchDiag.doneSamples++;
                    if (tr.reward != 0.0f) {
                        batchDiag.decisiveSamples++;
                    }
                }
                if (usedDouble) {
                    batchDiag.doubleDQNSamples++;
                }
                batchDiag.n++;
            }
            n++;
        }
    }

    if (n == 0) {
        return 0.0f;
    }

    m_lastLoss = (double)lossSum / (double)n;
    m_lastBatchSamples = n;

    /*
       ---- 稀疏 MoE 的负载均衡辅助损失 ----
       放在优化器之前、主反向之后, 而且**每层只调一次**: 共享骨干口径下
       trunk/qHead/qNet 指向同一个 MoE 层, 遍历三个视图会让同一个层被调三次
       (SAC 那边的注释: "显式只调一次, 不靠运气")。
    */
    if (auxLossCoef > 0.0f) {
        RL::ISparseMoE *moe = findSparseMoe(trunk);
        if (moe != nullptr) {
            moe->addAuxGradient(auxLossCoef);
        }
    }

    /*
       ---- 优化器: 每个层**只更新一次** ----
       `trunk.RMSProp` 更新骨干的两层, `qHead.RMSProp` 更新 Q 头。**不要**再调
       `qNet.RMSProp` —— 那会让同一批权重在同一批样本上被更新两次 (等于学习率翻倍)。
    */
    trunk.RMSProp(learningRate, 0.9f, 0.0f);
    qHead.RMSProp(learningRate, 0.9f, 0.0f);
    if (twin) {
        q2Head.RMSProp(learningRate, 0.9f, 0.0f);
    }

    /*
       ---- 目标网同步 ([F1]) ----
       `targetTau >= 1` = 硬拷贝。旧类那条路径是 `tau=0.01` + 每 256 次 learn
       (≈ 每 1024 手) —— 20 局之后目标网只挪动百分之几, 自举项等于 0 (SAC 的 F1
       诊断就是同一类缺陷)。本类默认取修好的那一档, 并把两个量留成成员供 A/B。
    */
    const int kSync = (replaceTargetIter > 0) ? replaceTargetIter : 1;
    if (m_learnSteps % kSync == 0) {
        const float tau = (targetTau > 1.0f) ? 1.0f : targetTau;
        trunk.softUpdateTo(trunkTarget, tau);
        qHead.softUpdateTo(qHeadTarget, tau);
        if (twin) {
            q2Head.softUpdateTo(q2HeadTarget, tau);
        }
    }

    /* 累计诊断 (自检面板读这一份) */
    trainDiag.n += batchDiag.n;
    trainDiag.clamped += batchDiag.clamped;
    trainDiag.yPreAbsSum += batchDiag.yPreAbsSum;
    trainDiag.yPreAbsMax = std::max(trainDiag.yPreAbsMax, batchDiag.yPreAbsMax);
    trainDiag.ySum += batchDiag.ySum;
    trainDiag.vNextSum += batchDiag.vNextSum;
    trainDiag.qSpreadSum += batchDiag.qSpreadSum;
    trainDiag.qAbsMeanSum += batchDiag.qAbsMeanSum;
    trainDiag.qTargetAbsSum += batchDiag.qTargetAbsSum;
    trainDiag.slotSum += batchDiag.slotSum;
    trainDiag.legalSum += batchDiag.legalSum;
    trainDiag.doneSamples += batchDiag.doneSamples;
    trainDiag.decisiveSamples += batchDiag.decisiveSamples;
    trainDiag.doubleDQNSamples += batchDiag.doubleDQNSamples;

    /* ε 衰减 (与旧类/RL::DQN 同一口径: 0.9999^n, 下界 0.1) */
    exploringRate *= 0.9999f;
    if (exploringRate < 0.1f) {
        exploringRate = 0.1f;
    }

    m_learnSteps++;
    return (float)m_lastLoss;
}

void DQNMCTSMOETbAgent::resetMoeBatchStats()
{
    for (std::size_t i = 0; i < trunk.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(trunk[i]);
        if (m != nullptr) {
            m->resetBatchStats();
        }
    }
}

/* ================================================================
 *  MoE / 骨干读数 (全部只读)
 * ================================================================ */
int DQNMCTSMOETbAgent::moeExpertCount() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->expertCount() : 0;
}

int DQNMCTSMOETbAgent::moeTopK() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->topK() : 0;
}

bool DQNMCTSMOETbAgent::moeDenseNow() const
{
    const int e = moeExpertCount();
    return (e > 0) && (moeTopK() == e);
}

void DQNMCTSMOETbAgent::moeUsage(std::vector<long long> &out) const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    if (m == nullptr) {
        out.clear();
        return;
    }
    m->usageSnapshot(out);
}

void DQNMCTSMOETbAgent::resetMoeUsage()
{
    RL::ISparseMoE *m = findSparseMoe(trunk);
    if (m != nullptr) {
        m->resetUsage();
    }
}

/* TB 专家的实际注意力头口径: 委派给 MoE 层的第 0 个专家 (见 ilayer.h 的说明) */
int DQNMCTSMOETbAgent::tbHeadsRequested() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->attnHeadsRequested() : -1;
}
int DQNMCTSMOETbAgent::tbHeadsUsed() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->attnHeadsUsed() : -1;
}
int DQNMCTSMOETbAgent::tbHeadDim() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->attnHeadDim() : -1;
}
int DQNMCTSMOETbAgent::tbHeadsAllocated() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->attnHeadsAllocated() : -1;
}
long long DQNMCTSMOETbAgent::tbAttnElements() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->attnElements() : -1;
}

long long DQNMCTSMOETbAgent::trunkParamCount() const
{
    RL::Net &self = const_cast<RL::Net&>(trunk);
    return self.paramCount();
}

long long DQNMCTSMOETbAgent::headParamCount() const
{
    RL::Net &h1 = const_cast<RL::Net&>(qHead);
    long long total = h1.paramCount();
    if (twinCritic) {
        RL::Net &h2 = const_cast<RL::Net&>(q2Head);
        total += h2.paramCount();
    }
    return total;
}

long long DQNMCTSMOETbAgent::uniqueParamCount() const
{
    /* 视图 (qNet/q2Net) 与 trunk/qHead 共享同一批层对象, 所以**不能**把它们也加起来 */
    return trunkParamCount() + headParamCount();
}

/* ================================================================
 *  在线训练: 记样本 / 节拍 / 终局
 * ================================================================ */

void DQNMCTSMOETbAgent::nextMaskBits(int color, std::uint64_t bits[2], int &legalCount)
{
    bits[0] = 0;
    bits[1] = 0;
    std::vector<Step*> steps;
    std::vector<int> idx;
    RL::Tensor mask(ACTION_DIM, 1);
    getLegalActions(color, steps, idx, mask);
    for (std::size_t i = 0; i < idx.size(); i++) {
        bitSetTo(bits, idx[i]);
    }
    legalCount = (int)idx.size();
    Steps::instance().put(steps);
}

void DQNMCTSMOETbAgent::maybeLearn()
{
    m_learnCounter++;
    const int every = (learnEveryMoves > 0) ? learnEveryMoves : 1;
    if (m_learnCounter % every == 0) {
        learnBatch(batchSize, 0);
    }
}

int DQNMCTSMOETbAgent::pickActionEpsilon(const RL::Tensor &state,
                                         const std::vector<int> &legalIdx)
{
    if (legalIdx.empty()) {
        return -1;
    }
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(RL::Random::engine) < exploringRate) {
        const int k = (int)(std::min<std::size_t>(
            legalIdx.size() - 1,
            (std::size_t)(u01(RL::Random::engine) * (float)legalIdx.size())));
        return legalIdx[(std::size_t)std::max(0, k)];
    }
    std::vector<float> qL;
    std::vector<float> q2L;
    std::vector<float> *q2Ptr = twinCritic ? &q2L : nullptr;
    if (!sparseLeafEval || !sparseQOnline(state, legalIdx, qL, q2Ptr)) {
        qValuesFull(state, m_qBuf, twinCritic ? &m_q2Buf : nullptr);
        qL.clear();
        qL.reserve(legalIdx.size());
        if (q2Ptr != nullptr) {
            q2L.clear();
            q2L.reserve(legalIdx.size());
        }
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            qL.push_back(m_qBuf[legalIdx[i]]);
            if (q2Ptr != nullptr) {
                q2L.push_back(m_q2Buf[legalIdx[i]]);
            }
        }
    }
    int best = legalIdx[0];
    double bestV = -1e30;
    for (std::size_t i = 0; i < legalIdx.size(); i++) {
        double v = (double)qL[i];
        if (q2Ptr != nullptr && i < q2L.size()) {
            v = std::min(v, (double)q2L[i]);
        }
        if (v > bestV) {
            bestV = v;
            best = legalIdx[i];
        }
    }
    return best;
}

void DQNMCTSMOETbAgent::recordExperience(const Step &chosenStep, int color)
{
    if (!chosenStep.valid) {
        return;
    }
    /*
       契约: 调用方必须**已经**走过这一手 (棋盘在 s'), 并且之前调过
       `selectMove(..., training=true)` 来缓存"走子前"的局面。理由见头文件。
       ⚠ 没有缓存时必须**拒绝**而不是"用空局面凑一条": 空局面 (cells 为空) 是一条
       完全错误的训练数据, 而且它不会报任何错。这里把拒绝次数记下来给自检面板 ——
       "静默什么都不做"与"做了但做错了"必须分得开。
    */
    if (!m_haveCached) {
        m_recordSkipped++;
        return;
    }
    m_haveCached = false;   /* 一份缓存只用一次: 重复调用不会记出两条一样的样本 */

    Sample tr;
    tr.cells = m_cachedCells;
    for (int i = 0; i < CTX_COUNT; i++) {
        tr.ctx[i] = m_cachedCtx[i];
    }
    tr.curMask[0] = m_cachedMask[0];
    tr.curMask[1] = m_cachedMask[1];
    tr.legalCount = (m_cachedLegal > 0) ? m_cachedLegal : 1;
    tr.action = stepToActionIdx(chosenStep, color);
    tr.mover = color;
    tr.decision = true;

    encodeSparse(chess.sideToMove, tr.nextCells);
    contextOf(chess, chess.sideToMove, tr.nextCtx);
    nextMaskBits(chess.sideToMove, tr.nextMask, tr.nextLegalCount);

    /*
       即时奖励 (走子方视角)。⚠ 这一步在 `moveForward` **之后**: 被吃子已经
       alive=false, 但 `computeReward` 只读它的 `value` (不检查 alive), 所以口径与
       "落子前算"逐字相同 —— 这条在 DQNMCTSAgent 那边有同样的说明。
    */
    float r = computeReward(chosenStep, color);
    bool done = false;
    {
        const int res = chess.getResult(chess.sideToMove);
        if (res != Chess::RESULT_ONGOING) {
            done = true;
            r = terminalReward(res, color);   /* 终局值 = 走子方视角 */
        }
    }
    tr.reward = r;
    tr.done = done;
    pushSample(tr);

    m_onlineStepCount++;
    maybeLearn();
}

void DQNMCTSMOETbAgent::endOnlineEpisode(int gameResult)
{
    totalEpisodes++;
    if (gameResult == Stone::COLOR_BLACK) {
        totalWins[1]++;
    }
    if (gameResult == Stone::COLOR_RED) {
        totalWins[0]++;
    }
    /*
       这一局是"对手没有合法走法"结束的 (调用方在 selectMove 返回无效着法时用它收尾):
       最后一条决策样本自己不知道终局, 现在补上。和棋 (COLOR_NONE) 补 0。
    */
    if (m_lastDecisionFromBack >= 0
        && (std::size_t)m_lastDecisionFromBack < memories.size()) {
        Sample &s = memories[memories.size() - 1 - (std::size_t)m_lastDecisionFromBack];
        if (!s.done) {
            const float v = (gameResult == Stone::COLOR_NONE)
                                ? 0.0f
                                : ((gameResult == s.mover) ? 1.0f : -1.0f);
            attachTerminalToLastDecision(v);
        }
    }
    /* 一局结束再学一次 (与旧类 endOnlineEpisode 同一节拍; 具体更不更新由 maybeLearn 决定) */
    maybeLearn();
    m_onlineStepCount = 0;
}

/*
 *  ---- 从自己的搜索学一次 (SAC 的 learnFromSearch) ----
 *
 *  棋盘此刻还在**根局面** (selectMove 把试走的都回退了)。这个函数只多做一件事:
 *  试走这一手拿 r 与 s', 编码 / 采掩码之后**原样退回**。硬契约: 返回时棋盘逐字节复原
 *  (含 sideToMove) —— 调用方紧接着就要按真实对局落子。
 */
void DQNMCTSMOETbAgent::learnFromSearchStep(int color, const Step &chosenStep,
                                            int actionIdx,
                                            const std::vector<int> &rootIdx)
{
    if (!chosenStep.valid) {
        return;
    }
    const int savedSide = chess.sideToMove;

    /* 即时奖励必须在落子之前算 (吃子奖励读的是落子前的棋盘) */
    const float r = computeReward(chosenStep, color);

    Sample tr;
    /*
       "当前局面" = **根局面**, 在这里现编码 (棋盘此刻正是根局面)。
       不要读 m_cached*: 那份缓存只在 m_trainingMode 打开时填, 而界面决策路径不开它 ——
       第一版读了缓存, 于是每一条样本的 cells 都是空的 (= 空局面进训练集)。
    */
    encodeSparse(color, tr.cells);
    contextOf(chess, color, tr.ctx);
    tr.curMask[0] = 0;
    tr.curMask[1] = 0;
    for (std::size_t i = 0; i < rootIdx.size(); i++) {
        bitSetTo(tr.curMask, rootIdx[i]);
    }
    tr.legalCount = (int)rootIdx.size();
    tr.action = (actionIdx >= 0) ? actionIdx : stepToActionIdx(chosenStep, color);
    tr.mover = color;
    tr.decision = true;

    double dummy = 0.0;
    Step stepCopy = chosenStep;
    chess.moveForward(&stepCopy, dummy);

    encodeSparse(chess.sideToMove, tr.nextCells);
    contextOf(chess, chess.sideToMove, tr.nextCtx);
    nextMaskBits(chess.sideToMove, tr.nextMask, tr.nextLegalCount);

    float reward = r;
    bool done = false;
    const int res = chess.getResult(chess.sideToMove);
    if (res != Chess::RESULT_ONGOING) {
        done = true;
        reward = terminalReward(res, color);
    }
    tr.reward = reward;
    tr.done = done;

    chess.moveBack(&stepCopy, dummy);
    chess.sideToMove = savedSide;   /* 逐字节复原 */

    pushSample(tr);
    maybeLearn();
}

/* ================================================================
 *  探索 + 预训练 (界面"① 探索环境 + 预训练"那一跳)
 * ================================================================ */
bool DQNMCTSMOETbAgent::exploreAndTrain(int color, int rolloutSteps,
                                        const OpponentPolicy &opponent)
{
    if (rolloutSteps <= 0 || batchSize <= 0) {
        return false;
    }

    /*
       pick: ε-greedy (从**合法槽位**上取 Q 的最大值, 以小概率随机取一个合法着法)。
       为什么要在这里重新采样一次合法集: `rolloutFromCurrent` 只把 (state, turn) 交给
       pick, 不传合法集。这样写与 SAC 那一支的 pick 同一手法 (它在 pick 里取掩码策略),
       而 Steps 空闲链是 thread_local 的"取出即独占"结构, 所以"外面持有一份合法集、
       里面再取一份"不会互相踩 (取出的对象不在空闲链里)。顺带把这一手的合法集与掩码
       存给紧接着的 onTrans 复用 (它只需要当前局面的那一份)。
    */
    auto pick = [this](const RL::Tensor &state, int turn) -> int {
        std::vector<Step*> legal;
        std::vector<int> idx;
        RL::Tensor mask(ACTION_DIM, 1);
        getLegalActions(turn, legal, idx, mask);
        Steps::instance().put(legal);
        if (idx.empty()) {
            m_pendingLegalCount = 0;
            m_pendingMask[0] = 0;
            m_pendingMask[1] = 0;
            return -1;
        }
        m_pendingLegalCount = (int)idx.size();
        m_pendingMask[0] = 0;
        m_pendingMask[1] = 0;
        for (std::size_t i = 0; i < idx.size(); i++) {
            bitSetTo(m_pendingMask, idx[i]);
        }
        return pickActionEpsilon(state, idx);
    };
    auto onTrans = [this](const Step &chosen, int actionIdx,
                          const RL::Tensor &s, const RL::Tensor &ns,
                          float r, bool done) {
        (void)chosen;
        Sample tr;
        denseToSparse(s, tr.cells);
        readContext(s, tr.ctx);
        denseToSparse(ns, tr.nextCells);
        readContext(ns, tr.nextCtx);
        tr.action = actionIdx;
        tr.mover = chess.sideToMove;
        tr.curMask[0] = m_pendingMask[0];
        tr.curMask[1] = m_pendingMask[1];
        tr.legalCount = (m_pendingLegalCount > 0) ? m_pendingLegalCount : 1;
        tr.reward = r;        /* rolloutFromCurrent 给的已经是走子方视角 */
        tr.done = done;
        tr.decision = false;  /* 这是**推演**样本, 不是真实决策 */
        /*
           下一局面的合法掩码: onTrans 被调用时棋盘正停在 s' (rollout 刚走完那一手),
           但 `chess.sideToMove` 已经是**下一个走子方** —— 所以直接用 sideToMove 采它。
           done=true 的样本不需要掩码 (自举项被 γ(1−done) 归零)。
        */
        if (!done) {
            nextMaskBits(chess.sideToMove, tr.nextMask, tr.nextLegalCount);
        }
        pushSample(tr);
    };

    /* 局部副本: rolloutFromCurrent 会把"真用了几手对手着法"回填到它里面 (P1) */
    OpponentPolicy opp = opponent;
    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick,
                                             onTrans, opp);

    bool trained = false;
    if (collected > 0) {
        /*
           "这一次有没有真的更新"要看 learnSteps 有没有前进, 不能看 loss 或池子大小:
           池里不足一个批时 learnBatch 直接早退 (正确行为), 那时 lastLoss 是上一批的
           残留值、池子大小也非零 —— 都会把"没训练"读成"训练了" (界面上的损失曲线就
           会多出一个假点)。
        */
        const int stepsBefore = m_learnSteps;
        learnBatch(batchSize, 0);
        trained = (m_learnSteps > stepsBefore);
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, 训练 "
                    + std::to_string(trained ? 1 : 0) + " 次(池 "
                    + std::to_string((int)memories.size()) + ")"
                    + opponentRolloutInfo(opp);
    return trained;
}

/* ================================================================
 *  三条训练回路 (自对弈 / 对随机 / 从当前局面热启动)
 * ================================================================ */

void DQNMCTSMOETbAgent::trainSelfPlay(int episodes, int simulations_,
                                      int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);
    const int sims = (simulations_ > 0) ? simulations_ : simulations;

    m_inTrainLoop = true;
    m_trainingMode = true;

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        m_onlineStepCount = 0;
        int currentColor = Stone::COLOR_BLACK;
        int movesPlayed = 0;
        int winner = Stone::COLOR_NONE;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /*
               根温度: 前 trainTempMoves 手按 N^(1/T) 采样 (开局多样性), 之后 argmax。
               这是 AlphaZero 的标准做法, 也是本类相对旧类的另一处改动 (旧类是"以 ε
               概率随机挑一个孩子")。
            */
            float temp = 0.0f;
            if (trainTempMoves > 0 && moveNum < trainTempMoves) {
                const float frac = (float)moveNum / (float)trainTempMoves;
                temp = trainTempRoot + (trainTempFinal - trainTempRoot) * frac;
            }
            Step step = selectMove(currentColor, sims, temp);
            if (!step.valid) {
                winner = (currentColor == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                            : Stone::COLOR_RED;
                break;
            }
            double dummy = 0.0;
            Step stepCopy = step;
            chess.moveForward(&stepCopy, dummy);
            recordExperience(stepCopy, currentColor);
            movesPlayed = moveNum + 1;
            if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                break;
            }
            currentColor = (currentColor == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                              : Stone::COLOR_RED;
        }

        const int finalResult = chess.getResult(chess.sideToMove);
        const bool decided = (finalResult != Chess::RESULT_ONGOING);
        totalEpisodes++;
        if (winner == Stone::COLOR_RED) {
            totalWins[0]++;
        } else if (winner == Stone::COLOR_BLACK) {
            totalWins[1]++;
        } else if (decided) {
            const int w = winnerOfResult(finalResult);
            if (w == Stone::COLOR_RED) {
                totalWins[0]++;
            } else if (w == Stone::COLOR_BLACK) {
                totalWins[1]++;
            }
        }
        /* 一局恰好一条 (界面自检面板的终局通道读数, 见 test 的 [8]) */
        noteEnd(decided ? finalResult : Chess::RESULT_ONGOING,
                chess.isGameOver() != Stone::COLOR_NONE);
        if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
            std::printf("  Episode %4d/%d: %s, %d moves, win_rate=%.2f, eps=%.4f, "
                        "pool=%d, learn=%d\n",
                        ep + 1, episodes,
                        decided ? "decided" : "draw/cap", movesPlayed,
                        getWinRate(), exploringRate, (int)memories.size(),
                        m_learnSteps);
        }
    }

    m_trainingMode = false;
    m_inTrainLoop = false;
}

void DQNMCTSMOETbAgent::trainVsRandom(int episodes, int simulations_,
                                       int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);
    const int sims = (simulations_ > 0) ? simulations_ : simulations;

    m_inTrainLoop = true;
    m_trainingMode = true;

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        m_onlineStepCount = 0;
        int currentColor = Stone::COLOR_BLACK;
        int movesPlayed = 0;
        int winner = Stone::COLOR_NONE;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            if (currentColor == Stone::COLOR_BLACK) {
                Step step = selectMove(Stone::COLOR_BLACK, sims, 0.0f);
                if (!step.valid) {
                    winner = Stone::COLOR_RED;   /* 学习方无棋可走 = 输 */
                    break;
                }
                double dummy = 0.0;
                Step stepCopy = step;
                chess.moveForward(&stepCopy, dummy);
                recordExperience(stepCopy, Stone::COLOR_BLACK);
                movesPlayed = moveNum + 1;
                if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                    break;
                }
                currentColor = Stone::COLOR_RED;
            } else {
                std::vector<Step*> redSteps;
                chess.sample(Stone::COLOR_RED, redSteps);
                if (redSteps.empty()) {
                    winner = Stone::COLOR_BLACK;
                    break;
                }
                std::uniform_int_distribution<int> u(0, (int)redSteps.size() - 1);
                const int k = u(RL::Random::engine);
                double dummy = 0.0;
                chess.moveForward(redSteps[(std::size_t)k], dummy);
                Steps::instance().put(redSteps);
                if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                    break;
                }
                currentColor = Stone::COLOR_BLACK;
            }
        }

        const int finalResult = chess.getResult(chess.sideToMove);
        const bool decided = (finalResult != Chess::RESULT_ONGOING);
        totalEpisodes++;
        if (winner == Stone::COLOR_RED) {
            totalWins[0]++;
        } else if (winner == Stone::COLOR_BLACK) {
            totalWins[1]++;
        } else if (decided) {
            const int w = winnerOfResult(finalResult);
            if (w == Stone::COLOR_RED) {
                totalWins[0]++;
            } else if (w == Stone::COLOR_BLACK) {
                totalWins[1]++;
            }
        }
        noteEnd(decided ? finalResult : Chess::RESULT_ONGOING,
                chess.isGameOver() != Stone::COLOR_NONE);
        if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
            std::printf("  Episode %4d/%d: %s, %d moves, win_rate=%.2f, eps=%.4f, "
                        "pool=%d, learn=%d\n",
                        ep + 1, episodes,
                        decided ? "decided" : "draw/cap", movesPlayed,
                        getWinRate(), exploringRate, (int)memories.size(),
                        m_learnSteps);
        }
    }

    m_trainingMode = false;
    m_inTrainLoop = false;
}

void DQNMCTSMOETbAgent::warmupFromCurrent(int episodes, int simulations_,
                                          int maxMoves)
{
    if (episodes <= 0) {
        return;
    }

    /* ---- 存下当前棋盘 (棋子位置 + 存活 + sideToMove + 历史 + 无吃子计数) ---- */
    struct StoneSave { int x, y, alive; };
    std::vector<StoneSave> saved(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[(std::size_t)i];
        StoneSave sv;
        sv.x = s->pos.x;
        sv.y = s->pos.y;
        sv.alive = s->alive ? 1 : 0;
        saved[(std::size_t)i] = sv;
    }
    const int savedSide = chess.sideToMove;
    const int savedHalf = chess.halfMoveClock;
    /*
       历史也要存下来 (旧类那一支只 clear 不还原): `repetitionCount` 读的正是它,
       而"复原"是这个函数的硬契约 —— 留下一条被清空的历史会让调用方那一局的
       重复判和口径悄悄变掉。
    */
    const std::vector<Chess::HistoryRecord> savedHistory = chess.history;

    m_inTrainLoop = true;
    m_trainingMode = true;

    for (int ep = 0; ep < episodes; ep++) {
        /* 回到起点 */
        for (int i = 0; i < 32; i++) {
            Stone *s = chess.stones[(std::size_t)i];
            s->alive = (saved[(std::size_t)i].alive != 0);
            s->pos.x = saved[(std::size_t)i].x;
            s->pos.y = saved[(std::size_t)i].y;
        }
        chess.m_map.clear();
        for (int i = 0; i < 32; i++) {
            Stone *s = chess.stones[(std::size_t)i];
            if (s != nullptr && s->alive) {
                chess.m_map[s->pos] = s;
            }
        }
        chess.history.clear();
        chess.sideToMove = savedSide;
        chess.halfMoveClock = savedHalf;

        m_onlineStepCount = 0;
        int currentColor = savedSide;
        int winner = Stone::COLOR_NONE;
        const int sims = (simulations_ > 0) ? simulations_ : simulations;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            Step step = selectMove(currentColor, sims, 0.0f);
            if (!step.valid) {
                winner = (currentColor == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                            : Stone::COLOR_RED;
                break;
            }
            double dummy = 0.0;
            Step stepCopy = step;
            chess.moveForward(&stepCopy, dummy);
            recordExperience(stepCopy, currentColor);
            if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                break;
            }
            currentColor = (currentColor == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                              : Stone::COLOR_RED;
        }
        endOnlineEpisode(winner);
    }

    m_trainingMode = false;
    m_inTrainLoop = false;

    /* ---- 原样复原 (含 sideToMove / halfMoveClock / history) ---- */
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[(std::size_t)i];
        s->alive = (saved[(std::size_t)i].alive != 0);
        s->pos.x = saved[(std::size_t)i].x;
        s->pos.y = saved[(std::size_t)i].y;
    }
    chess.m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[(std::size_t)i];
        if (s != nullptr && s->alive) {
            chess.m_map[s->pos] = s;
        }
    }
    chess.history.clear();
    chess.sideToMove = savedSide;
    chess.halfMoveClock = savedHalf;
    chess.history = savedHistory;
    m_onlineStepCount = 0;
}

/* ================================================================
 *  权重
 * ================================================================ */
const char *DQNMCTSMOETbAgent::defaultWeightPrefix()
{
    return "weights/dqnmcts_moe_agent";
}

bool DQNMCTSMOETbAgent::saveModel(const std::string &filepath)
{
    /*
       只写**唯一的那些层**: 骨干一份 + Q 头一份 (+ 双 critic 的第二个头)。
       `qNet` / `q2Net` 是视图, 写它们等于把同一批权重写第二遍 (而且载入时会被
       当成"另一些层")。
       文件个数与名字是**固定的三个**, 不依赖任何运行时开关 —— 否则"要不要有 _q2"
       会随 twinCritic 变, 载入方就得猜。
    */
    trunk.save(filepath + "_trunk");
    qHead.save(filepath + "_q");
    q2Head.save(filepath + "_q2");
    return weightFileWritten(filepath + "_trunk")
           && weightFileWritten(filepath + "_q")
           && weightFileWritten(filepath + "_q2");
}

bool DQNMCTSMOETbAgent::loadModel(const std::string &filepath)
{
    if (!weightFileReadable(filepath + "_trunk")
        || !weightFileReadable(filepath + "_q")
        || !weightFileReadable(filepath + "_q2")) {
        return false;
    }
    /* 三个都要成功: 半载入 (骨干换了、头没换) 是最坏的状态 —— 而且它不报错 */
    if (trunk.load(filepath + "_trunk") != 0) {
        return false;
    }
    if (qHead.load(filepath + "_q") != 0) {
        return false;
    }
    if (q2Head.load(filepath + "_q2") != 0) {
        return false;
    }
    /* 目标网由在线网派生 (与构造时同一条路: 深拷贝, 不是 Net 的浅拷贝赋值) */
    trunk.copyTo(trunkTarget);
    qHead.copyTo(qHeadTarget);
    q2Head.copyTo(q2HeadTarget);
    return true;
}

/* ================================================================
 *  自检报告 (界面右侧"模型自检"面板)
 *
 *  只报告**结构 / 口径**类事实, 不报告棋力。契约 (aiagent.h):
 *    * 只读、可重复调用、**绝不动棋盘** —— 涉及棋盘的部分一律用副本;
 *    * 空字符串没有意义 (本 agent 永远有内容);
 *    * 报告的是"这个模型值不值得继续训", 不是"它有多强"。
 *  这一节是 SAC 那条线上唯一"每次都真的抓到东西"的东西: 本工程每一次静默失效
 *  (头数被降级 / 目标网是死的 / 终局样本 0 条 / 开关建网后才赋值) 都是被一个读数
 *  抓到的, 不是被测试抓到的。
 * ================================================================ */
std::string DQNMCTSMOETbAgent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    std::snprintf(buf, sizeof(buf), "== DQN+MCTS (%s) ==\n", backboneName());
    out += buf;

    /* ---- 1. 骨干指纹 ---- */
    std::snprintf(buf, sizeof(buf),
                  "骨干: 专家 %d 个, topK=%d (真实读数: %s) | 注意力头 请求 %d / 实用 %d"
                  " | d_k=%d | 分配 %d | 注意力元素 %lld\n",
                  moeExpertCount(), moeTopK(),
                  moeDenseNow() ? "**全算 (稠密对照)**" : "稀疏路由",
                  tbHeadsRequested(), tbHeadsUsed(), tbHeadDim(), tbHeadsAllocated(),
                  tbAttnElements());
    out += buf;
    if (tbHeadsRequested() > 0 && tbHeadsUsed() > 0
        && tbHeadsRequested() != tbHeadsUsed()) {
        out += "  ⚠ 请求的头数与实际用的不一致 —— 这就是 SAC 那边实测过的静默降级"
               " (慢 3 倍, 参数指纹/paramCount/权重格式都不变)。\n";
    }
    std::snprintf(buf, sizeof(buf),
                  "参数量: 骨干 %lld + 头 %lld = %lld (视图 qNet/q2Net 与它们共享层对象,"
                  " 不重复计)\n",
                  trunkParamCount(), headParamCount(), uniqueParamCount());
    out += buf;
    std::snprintf(buf, sizeof(buf), "隐层激活(实测) %s | hidden=%d\n",
                  hiddenActivationName(), hiddenDim);
    out += buf;

    /* ---- 2. 表示 ---- */
    std::snprintf(buf, sizeof(buf),
                  "状态 %d 维 = %d 棋子平面 + %d 规则上下文 (无吃子/重复/被将) |"
                  " 动作 %d 槽位 (哈希)\n",
                  STATE_DIM, PIECE_PLANES, CTX_COUNT, ACTION_DIM);
    out += buf;
    out += "规则上下文通道: 3 个 (规范视角, 轮到谁走谁的子在 x 大的一侧)\n";

    /* ---- 3. 动作别名 (只读副本, 绝不碰 this->chess) ---- */
    {
        int legalN = 0, slotN = 0, worst = 0;
        Chess probe(chess);
        probe.reset();
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(legal, legalN, slotN, worst);
        Steps::instance().put(legal);
        std::snprintf(buf, sizeof(buf),
                      "动作别名(标准开局): %d 个合法着法 -> %d 个 Q 槽位, 挤掉 %d 个"
                      " (最挤槽位 %d 个着法)\n",
                      legalN, slotN, legalN - slotN, worst);
        out += buf;
    }
    if (aliasMoves > 0) {
        std::snprintf(buf, sizeof(buf),
                      "动作别名(对局累计): %lld 个着法, 平均每次挤掉 %.2f 个"
                      " (最挤槽位 %lld 个着法)\n",
                      aliasMoves, (double)aliasClearedMoves / (double)aliasMoves,
                      aliasWorstSlot);
        out += buf;
    } else {
        out += "动作别名(对局累计): 还没有对局数据\n";
    }

    /* ---- 4. 搜索口径 ---- */
    std::snprintf(buf, sizeof(buf),
                  "搜索: 模拟 %d 次 | c_puct=%.3f | 先验温度 %.3f (合法槽位上 Q 的"
                  " softmax) | 叶子 %s | 叶估值 %lld 次 (其中全量回退 %lld)\n",
                  simulations, (double)c_puct, (double)priorTemp,
                  sparseLeafEval ? "稀疏列" : "**全量**", m_leafEvals, m_fullLeafEvals);
    out += buf;
    out += "PUCT 符号: -child.getQ() + c_puct*P*sqrt(N)/(1+n) (negamax; 与 PPOMCTS/"
           "SACAZ/MCTS 四处同口径)\n";
    out += "叶子终局: 走 terminalReward (与训练目标同一个出口; 不是 isGameOver)\n";

    /* ---- 5. 学习口径 ---- */
    std::snprintf(buf, sizeof(buf),
                  "学习: batch=%d | 池 %d/%d | epochs=%d | 每 %d 手一次 | 步数 %d |"
                  " 目标网 tau=%.4g 每 %d 步 | 夹目标 %s | Huber %s | DoubleDQN=%d |"
                  " twinCritic=%d | MoE辅助 %.3g\n",
                  batchSize, (int)memories.size(), (int)maxMemorySize, replayEpochs,
                  learnEveryMoves, m_learnSteps, (double)targetTau, replaceTargetIter,
                  clampTarget > 0.0f ? std::to_string(clampTarget).c_str() : "关",
                  huberDelta > 0.0f ? std::to_string(huberDelta).c_str() : "关(纯MSE)",
                  (int)doubleDQN, (int)twinCritic, (double)auxLossCoef);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "从自己的搜索学一次: %s | ε=%.4f\n",
                  learnFromSearch ? "开" : "关", (double)exploringRate);
    out += buf;
    if (m_recordSkipped > 0) {
        std::snprintf(buf, sizeof(buf),
                      "  ⚠ recordExperience 被拒绝 %lld 次 (没有先 selectMove 缓存走子前局面)"
                      " —— 那几手**没有**进训练集, 而这是有意的: 拿空局面凑一条样本比\n"
                      "    什么都不记更坏。训练回路的正常路径不该出现这个数。\n",
                      m_recordSkipped);
        out += buf;
    }

    /* ---- 6. 最近一批的 critic 读数 ---- */
    if (batchDiag.n > 0) {
        const double n = (double)batchDiag.n;
        std::snprintf(buf, sizeof(buf),
                      "最近一批(%lld 条): |y|均 %.4f 最大 %.4f | 被夹 %.1f%% |"
                      " |Q|均 %.4f | Q spread %.4f | |Q_target|均 %.4f |"
                      " done %lld (其中分胜负 %lld)\n",
                      batchDiag.n, batchDiag.yPreAbsSum / n, batchDiag.yPreAbsMax,
                      100.0 * (double)batchDiag.clamped / n,
                      batchDiag.qAbsMeanSum / n, batchDiag.qSpreadSum / n,
                      batchDiag.qTargetAbsSum / n, batchDiag.doneSamples,
                      batchDiag.decisiveSamples);
        out += buf;
        if (batchDiag.clamped * 2 > batchDiag.n) {
            out += "  ⚠ 过半样本的目标被 clampTarget 夹住 —— 那些样本对 critic 的"
                   "**排序**学习没有贡献 (而搜索要的正是排序)。\n";
        }
        if (batchDiag.qTargetAbsSum / n < 0.05) {
            out += "  ⚠ |Q_target| 只有随机初始化尺度 —— 自举项等于 0, TD 目标退化成"
                   " r 本身 (SAC 的 F1 诊断就是这条)。\n";
        }
    } else {
        out += "最近一批: 还没有训练数据\n";
    }
    if (trainDiag.n > 0) {
        const double n = (double)trainDiag.n;
        std::snprintf(buf, sizeof(buf),
                      "累计(%lld 条): |y|均 %.4f | 被夹 %.1f%% | Q spread 均 %.4f |"
                      " done %lld / 分胜负 %lld\n",
                      trainDiag.n, trainDiag.yPreAbsSum / n,
                      100.0 * (double)trainDiag.clamped / n,
                      trainDiag.qSpreadSum / n, trainDiag.doneSamples,
                      trainDiag.decisiveSamples);
        out += buf;
        if (trainDiag.doneSamples == 0) {
            out += "  ⚠ 一条终局样本都没有 ⇒ 终局 ±1 从没进过目标 (截断的比例太高,"
                   " 或者对手终局那一手没被通知到)。\n";
        }
    }

    /* ---- 7. MoE 路由直方图 ---- */
    {
        std::vector<long long> usage;
        moeUsage(usage);
        if (!usage.empty()) {
            long long total = 0;
            long long mx = 0;
            for (std::size_t i = 0; i < usage.size(); i++) {
                total += usage[i];
                mx = std::max(mx, usage[i]);
            }
            std::snprintf(buf, sizeof(buf), "MoE 路由(累计 %lld 次前向): ", total);
            out += buf;
            for (std::size_t i = 0; i < usage.size(); i++) {
                std::snprintf(buf, sizeof(buf), "E%u=%lld(%.1f%%) ", (unsigned)i,
                              usage[i],
                              total > 0 ? 100.0 * (double)usage[i] / (double)total
                                        : 0.0);
                out += buf;
            }
            out += "\n";
            const double mean = (double)total / (double)usage.size();
            if (mean > 0.0 && (double)mx / mean > 3.0) {
                out += "  ⚠ 路由偏斜 (最大/均值 > 3): 有专家几乎不被选中 ——"
                       " 辅助损失系数 %.3g 可能太小。\n";
            }
        }
    }

    /* ---- 8. 终局通道 ---- */
    {
        const long long total = endCount[END_CAP] + endCount[END_RED_WIN]
                              + endCount[END_BLACK_WIN] + endCount[END_DRAW];
        if (total > 0) {
            const long long decided = total - endCount[END_CAP];
            const long long missed = decided - endSeenByGameOver;
            std::snprintf(buf, sizeof(buf),
                          "终局通道: %lld 局 | 截断 %lld | 红胜 %lld | 黑胜 %lld |"
                          " 和 %lld\n",
                          total, endCount[END_CAP], endCount[END_RED_WIN],
                          endCount[END_BLACK_WIN], endCount[END_DRAW]);
            out += buf;
            std::snprintf(buf, sizeof(buf),
                          "  已分胜负/和棋的 %lld 局里, 旧口径 isGameOver 只看见 %lld 局"
                          " (漏 %lld) | 外部终局通道接住 %lld 次\n",
                          decided, endSeenByGameOver, missed > 0 ? missed : 0,
                          externalTerminals);
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "终局通道: 还没有对局数据 | 外部终局通道接住 %lld 次\n",
                          externalTerminals);
            out += buf;
        }
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力只有带置信区间的锚点对局能回答\n";
    return out;
}
