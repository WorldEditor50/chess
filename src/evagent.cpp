#include "evagent.h"

#include "chessstate.h"   /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
#include <algorithm>
#include <cmath>
#include <cstdio>

/*
 * EVABAgent - 实现说明见 evagent.h 与 docs/agent_evab_design.md。
 */

namespace {

/*
   初始化缩放 (1/sqrt(fan_in))。
   `Layer` 的构造函数把权重初始化成 U(-1,1); 而 EVAB 的输入是 1260 维 one-hot
   (一局最多 32 个子 -> 约 32 个 1), 于是第一层 pre-activation 的标准差约
   sqrt(32/3) ≈ 3.3 —— tanh **一开始就饱和**, 网络输出恒为 ±1:
   实测探针 (build/evab_probe2.cpp) 里任意局面的输出都是 0.998, 与手工评估的差距
   |net-hand| ≈ 0.95。这样的网络既学不动 (梯度 ~0) 也帮不上忙 (输出是常数),
   "学习到的评估"从来没有真正生效过。
   按 1/sqrt(fan_in) 缩放一遍把 pre-activation 拉回 ~0.5, 网络才有表达力。
   (同一个修法在 sacazagent.cpp / dqn 系里已经用了。
*/
void scaleLayerInit(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
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

} // namespace

/* ============================================================
 *  构造 / 基本信息
 * ============================================================ */
EVABAgent::EVABAgent(Chess &chess_, int hiddenDim, int depth, long long budgetMs)
    : AgentBase(),
      chess(chess_),
      valueNet(RL::Layer<RL::Tanh>::_(STATE_DIM, hiddenDim, true, true),
               RL::Layer<RL::Tanh>::_(hiddenDim, 1, true, true)),
      blend(0.0f),
      blendMax(0.3f),
      blendStep(0.05f),
      maxDepth(depth),
      timeBudgetMs(budgetMs),
      exploreBudgetMs(600),
      outcomeWeight(0.5f),
      exploreEps(0.15f),
      learningRate(0.002f),
      m_aborted(false),
      m_horizon(0),
      m_nodes(0),
      m_ttProbes(0),
      m_ttHits(0),
      m_ttStores(0),
      m_leafEvals(0),
      m_reachedDepth(0),
      m_lastRootScore(0.0),
      m_hiddenDim(hiddenDim)
{
    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_killers.resize(2 * 64);
    m_tt.reserve(1 << 16);

    /*
       初始化缩放: 见文件开头 scaleLayerInit 的长注释 —— 不缩放的话网络一上来就
       tanh 饱和, 输出恒为 ±1, "学习到的评估"这件事从来没有真正生效过。
    */
    scaleLayerInit(valueNet);
}

std::string EVABAgent::getName() const
{
    char buf[128];
    /*
       名字里**不放 blend**。它是损失曲线的分线键 (MainWindow::lossSeriesFor 用 agent
       名字建曲线), 而 blend 在训练过程中是会变的 —— 一开始写成 "..., blend=%.2f" 的
       结果是 EVAB 每变一次 blend 就多出一条曲线 (实测一局下来 6 条同名曲线)。
       当前 blend 值照样看得到: 它在 getExploreInfo() 里 ("... blend 0.00->0.05")。
    */
    std::snprintf(buf, sizeof(buf), "EVAB (learned eval, depth=%d)", maxDepth);
    return buf;
}

void EVABAgent::resetStats()
{
    m_nodes = m_ttProbes = m_ttHits = m_ttStores = m_leafEvals = 0;
    m_reachedDepth = 0;
}

/* ============================================================
 *  状态编码: 规范视角 (轮到谁走, 谁就在"下方")
 * ============================================================ */
void EVABAgent::encodeCanonical(int color, RL::Tensor &state) const
{
    state.zero();
    /*
       14 个棋子平面 (规范视角) + **3 个**规则上下文平面 (无吃子/重复/将军), 由公共实现
       写出 (src/chessstate.h) —— 镜像与规则口径都只有一份, 不在这里重写。
       ⚠ 掩码必须显式给 CTX_RULES: 本 agent 的布局是 17 个平面, 而"5 个全写"的那条
       便捷入口会写满 19 个平面 —— 第一版就是那么调的, 结果越界写 180 个 float,
       实测直接堆损坏崩溃 (0xC0000374)。
    */
    ChessState::encodeWithContext(const_cast<Chess &>(chess), color, &state[0], 0,
                                  ChessState::CTX_RULES, true, REWARD_MAX_PLIES);
}

/* ============================================================
 *  叶子评估
 * ============================================================ */

double EVABAgent::evaluateLeaf(int color)
{
    const int over = chess.isGameOver();
    if (over == Stone::COLOR_BLACK) {
        return (color == Stone::COLOR_BLACK) ? MATE : -MATE;
    }
    if (over == Stone::COLOR_RED) {
        return (color == Stone::COLOR_RED) ? MATE : -MATE;
    }

    /* 手工评估: evaluate() 是黑方视角, 换算到当前走棋方视角, 再压到 (-1,1) */
    double hand = chess.evaluate();
    if (color == Stone::COLOR_RED) {
        hand = -hand;
    }
    const double handN = std::tanh(hand / (double)EVAL_SCALE);
    if (blend <= 0.0f) {
        return handN;
    }

    m_leafEvals++;
    encodeCanonical(color, m_stateBuf);
    RL::Tensor &out = valueNet.forward(m_stateBuf, true);
    const double v = (double)out[0];
    if (blend >= 1.0f) {
        return v;
    }
    return (double)blend * v + (1.0 - (double)blend) * handN;
}

/* ============================================================
 *  走法排序: 置换表走法 -> 吃子(MVV-LVA) -> 杀手 -> 历史启发
 * ============================================================ */
namespace {
int victimValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 10000;
    case Stone::TYPE_CHE:   return 500;
    case Stone::TYPE_MA:    return 300;
    case Stone::TYPE_PAO:   return 300;
    case Stone::TYPE_SHI:   return 200;
    case Stone::TYPE_XIANG: return 200;
    case Stone::TYPE_BING:  return 100;
    default:                return 0;
    }
}
int attackerValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 0;
    case Stone::TYPE_BING:  return 1;
    case Stone::TYPE_SHI:   return 2;
    case Stone::TYPE_XIANG: return 3;
    case Stone::TYPE_MA:    return 4;
    case Stone::TYPE_PAO:   return 4;
    case Stone::TYPE_CHE:   return 5;
    default:                return 10;
    }
}
} /* namespace */

void EVABAgent::recordKiller(int ply, const Step &s)
{
    if (ply < 0 || (std::size_t)(2*ply + 1) >= m_killers.size()) {
        return;
    }
    Step &k0 = m_killers[2*ply + 0];
    Step &k1 = m_killers[2*ply + 1];
    const bool same0 = (k0.id == s.id && k0.nextPos == s.nextPos && k0.pos == s.pos);
    if (same0) {
        return;
    }
    k1 = k0;
    k0 = s;
}

void EVABAgent::scoreMoves(std::vector<Step*> &steps, int ply, const Step *ttMove)
{
    const std::size_t n = steps.size();
    std::vector<int> score(n, 0);
    for (std::size_t i = 0; i < n; i++) {
        Step *s = steps[i];
        int sc = 0;
        if (s->nextId != Stone::ID_NONE) {
            Stone *victim = chess.m_children[s->nextId];
            Stone *attacker = chess.m_children[s->id];
            if (victim != nullptr && attacker != nullptr) {
                sc = 100000 + victimValue(victim->type)*10 - attackerValue(attacker->type);
            }
        }
        if (ttMove != nullptr && ttMove->valid &&
            s->id == ttMove->id && s->nextPos == ttMove->nextPos) {
            sc += 1000000;
        } else if (ply >= 0 && (std::size_t)(2*ply + 1) < m_killers.size()) {
            const Step &k0 = m_killers[2*ply + 0];
            const Step &k1 = m_killers[2*ply + 1];
            if (k0.id == s->id && k0.nextPos == s->nextPos && k0.pos == s->pos) {
                sc += 50000;
            } else if (k1.id == s->id && k1.nextPos == s->nextPos && k1.pos == s->pos) {
                sc += 40000;
            }
        }
        /* 历史启发: 安静走法之间也要有顺序 */
        std::unordered_map<int, int>::const_iterator it = m_history.find(historyIndex(*s));
        if (it != m_history.end()) {
            sc += it->second;
        }
        score[i] = sc;
    }
    /* 选择排序: n 只有几十, 且要求稳定地只看前几个, 用简单选择排序即可 */
    for (std::size_t i = 0; i + 1 < n; i++) {
        std::size_t best = i;
        for (std::size_t j = i + 1; j < n; j++) {
            if (score[j] > score[best]) {
                best = j;
            }
        }
        if (best != i) {
            std::swap(steps[i], steps[best]);
            std::swap(score[i], score[best]);
        }
    }
}

bool EVABAgent::timeUp()
{
    if (timeBudgetMs <= 0) {
        return false;
    }
    return std::chrono::steady_clock::now() >= m_deadline;
}

void EVABAgent::clearSearchState()
{
    m_tt.clear();
    m_history.clear();
    std::fill(m_killers.begin(), m_killers.end(), Step());
}

/* ============================================================
 *  静态搜索 (negamax, 只搜吃子)
 * ============================================================ */
double EVABAgent::quiescence(int color, double alpha, double beta, int ply)
{
    if ((++m_nodes & 1023) == 0 && timeUp()) {
        m_aborted = true;
        return alpha;
    }
    const int over = chess.isGameOver();
    if (over != Stone::COLOR_NONE) {
        return (over == color) ? (MATE - ply) : -(MATE - ply);
    }
    if (chess.isRepetition()) {
        return 0.0;
    }

    double standPat = evaluateLeaf(color);
    if (standPat >= beta) {
        return standPat;
    }
    if (standPat > alpha) {
        alpha = standPat;
    }
    if (ply >= m_horizon + 4) {
        /* 吃子延伸的层数上限, 防止极端局面下的爆炸 */
        return alpha;
    }

    std::vector<Step*> all;
    chess.samplePseudo(color, all);
    std::vector<Step*> captures;
    std::vector<Step*> quiet;
    quiet.reserve(all.size());
    for (Step *s : all) {
        if (s->nextId != Stone::ID_NONE) {
            captures.push_back(s);
        } else {
            quiet.push_back(s);
        }
    }
    Steps::instance().put(quiet);

    /* 只对吃子做合法性校验 (逐个), 不合法的还回对象池 */
    {
        const bool inCheck = chess.isInCheck(color);
        std::vector<Step*> legal;
        std::vector<Step*> rejected;
        legal.reserve(captures.size());
        for (Step *s : captures) {
            if (chess.isLegalMoveInternal(color, s, inCheck)) {
                legal.push_back(s);
            } else {
                rejected.push_back(s);
            }
        }
        if (!rejected.empty()) {
            Steps::instance().put(rejected);
        }
        captures.swap(legal);
    }

    scoreMoves(captures, ply, nullptr);

    double best = alpha;
    for (std::size_t i = 0; i < captures.size(); i++) {
        Step *s = captures[i];
        double r = 0;
        chess.moveForward(s, r);
        const double score = -quiescence((color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                                     : Stone::COLOR_RED,
                                         -beta, -best, ply + 1);
        chess.moveBack(s, r);
        if (m_aborted) {
            Steps::instance().put(captures);
            return best;
        }
        if (score > best) {
            best = score;
        }
        if (best >= beta) {
            break;
        }
    }
    Steps::instance().put(captures);
    return best;
}

/* ============================================================
 *  Alpha-Beta 主搜索 (negamax)
 * ============================================================ */
double EVABAgent::negamax(int color, int depth, double alpha, double beta, int ply)
{
    if ((++m_nodes & 1023) == 0 && timeUp()) {
        m_aborted = true;
        return alpha;
    }

    const int over = chess.isGameOver();
    if (over != Stone::COLOR_NONE) {
        return (over == color) ? (MATE - ply) : -(MATE - ply);
    }
    if (chess.isRepetition()) {
        return 0.0;
    }
    if (depth <= 0) {
        return quiescence(color, alpha, beta, ply);
    }

    /* ---- 置换表探测 ---- */
    const unsigned long long key = chess.computeHash();
    const Step *ttMove = nullptr;
    Step ttStep;
    std::unordered_map<unsigned long long, TTEntry>::const_iterator it = m_tt.find(key);
    m_ttProbes++;
    if (it != m_tt.end()) {
        const TTEntry &e = it->second;
        if (e.hasMove) {
            ttStep.id = e.id;
            ttStep.nextId = e.nextId;
            ttStep.pos = e.from;
            ttStep.nextPos = e.to;
            ttStep.reward = 0;
            ttStep.valid = true;
            ttMove = &ttStep;
        }
        if (e.depth >= depth) {
            m_ttHits++;
            /* 杀棋分数要按 ply 还原 (存储时存的是"离根的距离"版本) */
            double sc = e.score;
            if (sc >= MATE - 100) {
                sc -= ply;
            } else if (sc <= -(MATE - 100)) {
                sc += ply;
            }
            if (e.flag == 0) {
                return sc;
            }
            if (e.flag == 1 && sc > alpha) {
                alpha = sc;
            } else if (e.flag == 2 && sc < beta) {
                beta = sc;
            }
            if (alpha >= beta) {
                return sc;
            }
        }
    }

    /* ---- 生成合法走法 ---- */
    std::vector<Step*> steps;
    chess.sample(color, steps);
    if (steps.empty()) {
        /* 将杀 / 困毙: 中国象棋里困毙同样判负 */
        return -(MATE - ply);
    }
    scoreMoves(steps, ply, ttMove);

    if (ply + 1 > m_reachedDepth) {
        m_reachedDepth = ply + 1;
    }

    const double alphaOrig = alpha;
    double best = -INF;
    int bestIdx = -1;
    for (std::size_t i = 0; i < steps.size(); i++) {
        Step *s = steps[i];
        double r = 0;
        chess.moveForward(s, r);
        const double score = -negamax((color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                                  : Stone::COLOR_RED,
                                      depth - 1, -beta, -alpha, ply + 1);
        chess.moveBack(s, r);
        if (m_aborted) {
            Steps::instance().put(steps);
            return best > -INF ? best : alpha;
        }
        if (score > best) {
            best = score;
            bestIdx = (int)i;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            /* 剪枝: 记录杀手 + 历史启发 */
            if (s->nextId == Stone::ID_NONE) {
                recordKiller(ply, *s);
                m_history[historyIndex(*s)] += depth*depth;
                if (m_history[historyIndex(*s)] > 100000) {
                    for (std::unordered_map<int, int>::iterator h = m_history.begin();
                         h != m_history.end(); ++h) {
                        h->second /= 2;
                    }
                }
            }
            break;
        }
    }
    if (bestIdx < 0) {
        bestIdx = 0;
    }

    /* ---- 置换表存储 ----
       TT 里存的是**具体走法**, 使用时还要在合法走法列表里匹配上才采用,
       所以即使 computeHash() 发生碰撞, 也只会损失一点排序质量, 不会走出非法棋。 */
    {
        TTEntry e;
        e.depth = depth;
        e.score = best;
        if (best <= alphaOrig) {
            e.flag = 2;                       /* UPPER */
        } else if (best >= beta) {
            e.flag = 1;                       /* LOWER */
        } else {
            e.flag = 0;                       /* EXACT */
        }
        /* 存"离根的距离"版本, 便于按 ply 还原 */
        if (e.score >= MATE - 100) {
            e.score += ply;
        } else if (e.score <= -(MATE - 100)) {
            e.score -= ply;
        }
        Step *bm = steps[bestIdx];
        e.hasMove = true;
        e.id = bm->id;
        e.nextId = bm->nextId;
        e.from = bm->pos;
        e.to = bm->nextPos;
        if (m_tt.size() > (std::size_t)400000) {
            m_tt.clear();
        }
        m_tt[key] = e;
        m_ttStores++;
    }

    Steps::instance().put(steps);
    return best;
}

/* ============================================================
 *  搜索入口: 迭代加深
 * ============================================================ */
Step EVABAgent::search(int color, int depth, long long budgetMs)
{
    if (budgetMs > 0) {
        m_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs);
    } else if (timeBudgetMs > 0) {
        m_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeBudgetMs);
    }
    m_aborted = false;

    std::vector<Step*> rootSteps;
    chess.sample(color, rootSteps);
    if (rootSteps.empty()) {
        Steps::instance().put(rootSteps);
        return Step();                       /* valid == false */
    }

    /* 根节点的合法走法先存下来 (池里的对象在递归中不会被回收, 但保守起见还是拷一份) */
    std::vector<Step> roots;
    roots.reserve(rootSteps.size());
    for (Step *s : rootSteps) {
        roots.push_back(*s);
    }
    Steps::instance().put(rootSteps);

    Step best = roots[0];
    double bestScore = -INF;

    for (int d = 1; d <= depth; d++) {
        m_horizon = d;
        /* 上一层的根走法先试: 迭代加深最直接的收益 */
        std::stable_partition(roots.begin(), roots.end(), [&best](const Step &s) {
            return s.id == best.id && s.nextPos == best.nextPos;
        });

        double alpha = -INF;
        double localBest = -INF;
        Step localBestStep = best;
        bool completed = true;
        for (std::size_t i = 0; i < roots.size(); i++) {
            Step s = roots[i];
            double r = 0;
            chess.moveForward(&s, r);
            const double score = -negamax((color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                                      : Stone::COLOR_RED,
                                          d - 1, -INF, -alpha, 1);
            chess.moveBack(&s, r);
            if (m_aborted) {
                completed = false;
                break;
            }
            if (score > localBest) {
                localBest = score;
                localBestStep = s;
            }
            if (score > alpha) {
                alpha = score;
            }
        }
        if (!completed) {
            break;                            /* 用上一层已完成的结果 */
        }
        best = localBestStep;
        bestScore = localBest;
        /* 已经算出必胜/必败, 再深搜没有意义 */
        if (bestScore >= MATE - 100 || bestScore <= -(MATE - 100)) {
            break;
        }
        if (timeUp()) {
            break;
        }
    }
    m_lastRootScore = bestScore;
    return best;
}

Step EVABAgent::getBestMove(int color)
{
    clearSearchState();
    resetStats();
    return search(color, maxDepth, timeBudgetMs);
}

/*
 * scoreMove: 走 s 之后的局面评分 (当前走棋方视角, 完整窗口)。
 * 用于对称性测试: 一个颜色无关的搜索, 对镜像等价的走法必须给出相同的分数。
 */
double EVABAgent::scoreMove(int color, const Step &s, int depth)
{
    if (depth <= 0) {
        return evaluateLeaf(color);
    }
    double r = 0;
    chess.moveForward(const_cast<Step *>(&s), r);
    const double v = -negamax((color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                          : Stone::COLOR_RED,
                              depth - 1, -INF, INF, 1);
    chess.moveBack(const_cast<Step *>(&s), r);
    return v;
}

/* ============================================================
 *  自对弈训练: TD-leaf + 引导阶梯
 * ============================================================ */
/*
 * trainBatch: 把样本按 batchSize 分组, 每组累积梯度后做一次 RMSProp。
 *
 * 这一点是训练能不能work的关键: 早先的实现把整轮 (几百个样本) 的梯度累积起来
 * 只调用一次 RMSProp —— 于是"一轮 = 一步梯度", 三轮训练总共走了 3 步, 学不到
 * 任何东西 (实测 blend 爬到 1.0 之后对局 0 胜 6 负)。按小批量更新后, 一轮就有
 * 几十步。
 * 返回本轮最后一个 batch 的平均绝对误差。
 */
static double trainBatch(RL::Net &net, std::vector<EVABAgent::Sample> &samples,
                         int batchSize, float lr)
{
    if (samples.empty()) {
        return 0.0;
    }
    if (batchSize < 1) {
        batchSize = 1;
    }
    RL::Tensor target(1, 1);
    double lastErr = 0.0;
    int inBatch = 0;
    double batchErr = 0.0;
    for (std::size_t i = 0; i < samples.size(); i++) {
        RL::Tensor &out = net.forward(samples[i].state);
        batchErr += std::fabs((double)out[0] - (double)samples[i].target);
        target[0] = samples[i].target;
        net.backward(samples[i].state, RL::Loss::MSE::df(out, target));
        inBatch++;
        if (inBatch >= batchSize) {
            net.RMSProp(lr, 0.9f, 0.0f);
            lastErr = batchErr / (double)inBatch;
            batchErr = 0.0;
            inBatch = 0;
        }
    }
    if (inBatch > 0) {
        net.RMSProp(lr, 0.9f, 0.0f);
        lastErr = batchErr / (double)inBatch;
    }
    return lastErr;
}

double EVABAgent::pretrainFromHandEval(int positions, int maxPlies,
                                       int batchSize, int epochs)
{
    std::vector<Sample> samples;
    samples.reserve((std::size_t)positions);

    chess.reset();
    for (int i = 0; i < positions; i++) {
        if (i % maxPlies == 0) {
            chess.reset();
        }
        int turn = chess.sideToMove;
        std::vector<Step*> legal;
        chess.sample(turn, legal);
        if (legal.empty()) {
            chess.reset();
            Steps::instance().put(legal);
            continue;
        }
        /* 标签 = 手工评估 (规范视角下与 evaluateLeaf(blend=0) 完全一致) */
        Sample s;
        s.state = RL::Tensor(STATE_DIM, 1);
        encodeCanonical(turn, s.state);
        const double hand = std::tanh((turn == Stone::COLOR_RED ? -chess.evaluate()
                                                                : chess.evaluate())
                                      / (double)EVAL_SCALE);
        s.target = (float)hand;
        samples.push_back(s);

        Step *mv = legal[(std::size_t)(std::rand() % (int)legal.size())];
        double d = 0;
        chess.moveForward(mv, d);
        Steps::instance().put(legal);
    }
    chess.reset();

    double err = 0.0;
    for (int e = 0; e < epochs; e++) {
        err = trainBatch(valueNet, samples, batchSize, learningRate);
    }
    m_lastLoss = (float)err;   /* 界面"训练损失曲线"用, 见 getLastTrainLoss */
    return err;
}

double EVABAgent::trainSelfPlay(int games, int playDepth, int labelDepth,
                                int maxMoves, bool verbose, int batchSize)
{
    if (labelDepth < playDepth) {
        labelDepth = playDepth;
    }
    std::vector<Sample> samples;
    samples.reserve((std::size_t)games * maxMoves);

    for (int g = 0; g < games; g++) {
        chess.reset();
        int turn = Stone::COLOR_RED;
        int moveCount = 0;
        int result = Chess::RESULT_DRAW;

        /* 本局按访问顺序记录的 (状态, 搜索根评分, 行棋方) */
        struct Rec { RL::Tensor state; float score; int color; };
        std::vector<Rec> recs;

        for (moveCount = 0; moveCount < maxMoves; moveCount++) {
            int r = chess.getResult(turn);
            if (r != Chess::RESULT_ONGOING) {
                result = r;
                break;
            }

            std::vector<Step*> legal;
            chess.sample(turn, legal);
            if (legal.empty()) {
                result = (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                                    : Chess::RESULT_RED_WIN;
                Steps::instance().put(legal);
                break;
            }

            /* 探索: 以 eps 概率随机走, 保证自对弈数据的多样性 */
            const bool randomMove = ((float)std::rand() / (float)RAND_MAX) < exploreEps;
            Step chosen;

            /* 先记录"走之前"的状态快照 */
            Rec rec;
            rec.state = RL::Tensor(STATE_DIM, 1);
            encodeCanonical(turn, rec.state);
            rec.color = turn;
            rec.score = 0.0f;

            if (randomMove) {
                chosen = *legal[(std::size_t)(std::rand() % (int)legal.size())];
                /* 随机走法没有搜索评分: 标签只由对局结果提供 */
            } else {
                /*
                   走法用 playDepth 选 (便宜), 标签用 labelDepth 的根评分 (更准)。
                   两者相同就只搜一次。这正是 TD-leaf 的做法: V 的目标值是
                   "以 V 为叶子再搜 labelDepth 层"的结果, 比 V 本身更接近真值,
                   反复迭代就把更深的信息蒸馏进网络。
                */
                clearSearchState();
                chosen = search(turn, playDepth, 0);
                if (!chosen.valid) {
                    chosen = *legal[0];
                }
                rec.score = (float)m_lastRootScore;
                if (labelDepth > playDepth) {
                    clearSearchState();
                    search(turn, labelDepth, 0);
                    rec.score = (float)m_lastRootScore;
                }
            }
            recs.push_back(rec);

            /* 落子 */
            double dummy2 = 0.0;
            chess.moveForward(&chosen, dummy2);
            Steps::instance().put(legal);
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            int after = chess.getResult(turn);
            if (after != Chess::RESULT_ONGOING) {
                result = after;
                moveCount++;
                break;
            }
        }

        /* ---- 生成标签 ---- */
        for (std::size_t i = 0; i < recs.size(); i++) {
            float searchLabel = recs[i].score;
            if (searchLabel > 1.0f) {
                searchLabel = 1.0f;
            } else if (searchLabel < -1.0f) {
                searchLabel = -1.0f;
            }
            float outcome = 0.0f;
            if (result == Chess::RESULT_RED_WIN) {
                outcome = (recs[i].color == Stone::COLOR_RED) ? 1.0f : -1.0f;
            } else if (result == Chess::RESULT_BLACK_WIN) {
                outcome = (recs[i].color == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
            }
            Sample s;
            s.state = recs[i].state;
            s.target = outcomeWeight*outcome + (1.0f - outcomeWeight)*searchLabel;
            samples.push_back(s);
        }

        if (verbose) {
            std::printf("    game %3d/%d: %s, %d moves, samples=%zu\n",
                        g + 1, games,
                        (result == Chess::RESULT_RED_WIN) ? "red win"
                        : (result == Chess::RESULT_BLACK_WIN) ? "black win" : "draw",
                        moveCount, samples.size());
        }
    }

    if (samples.empty()) {
        return 0.0;
    }

    const double err = trainBatch(valueNet, samples, batchSize, learningRate);
    m_lastLoss = (float)err;   /* 界面"训练损失曲线"用, 见 getLastTrainLoss */
    return err;
}

bool EVABAgent::saveModel(const std::string &path)
{
    if (valueNet.save(path) != 0) {
        return false;
    }
    return weightFileWritten(path);
}

bool EVABAgent::loadModel(const std::string &path)
{
    if (!weightFileReadable(path)) {
        return false;
    }
    return valueNet.load(path) == 0;
}

/* ============================================================
 *  走子前的"探索环境 + 预训练": 把这次探索蒸馏回价值网络
 * ============================================================ */
double EVABAgent::netHandGap(int samples)
{
    /* 沿随机走法采样若干局面, 比较网络输出与手工评估 */
    const double keepBlend = blend;
    std::vector<Step> path;
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < samples; i++) {
        const int turn = chess.sideToMove;
        std::vector<Step*> legal;
        chess.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            break;
        }
        blend = 1.0f;
        const double netV = evaluateLeaf(turn);
        blend = 0.0f;
        const double handV = evaluateLeaf(turn);
        sum += std::fabs(netV - handV);
        n++;
        Step s = *legal[(std::size_t)(std::rand() % (int)legal.size())];
        double d = 0;
        chess.moveForward(&s, d);
        path.push_back(s);
        Steps::instance().put(legal);
    }
    for (std::size_t i = path.size(); i > 0; i--) {
        double d = 0;
        chess.moveBack(&path[i - 1], d);
    }
    blend = keepBlend;
    return (n > 0) ? sum/(double)n : 0.0;
}

bool EVABAgent::exploreAndTrain(int color, int rolloutSteps)
{
    if (rolloutSteps <= 0) {
        return false;
    }

    /*
       1) 备份当前权重以便回滚。
       注意 Net 的拷贝构造是**浅拷贝** (只复制 shared_ptr<layer>), 所以
       `RL::Net backup = valueNet;` 与本体共用同一组层, 回滚会变成空操作。
       必须按同样的结构新建一份, 再用 copyTo() 把权重复制过去。
    */
    RL::Net backup(RL::Layer<RL::Tanh>::_(STATE_DIM, m_hiddenDim, true, false),
                   RL::Layer<RL::Tanh>::_(m_hiddenDim, 1, true, false));
    valueNet.copyTo(backup);
    const double gapBefore = netHandGap(12);

    /*
       2) 从当前局面滚若干步 ("探索环境"), 每步用**搜索的根评分**做 TD-leaf 标签。
       探索走的是随机合法走法 (不是搜索最优, 否则样本没有多样性), 全程在真棋盘上
       用 moveForward 试走, 结束时按相反顺序回退。
    */
    std::vector<Sample> samples;
    std::vector<Step> path;
    int turn = color;
    /*
       探索必须对棋盘零副作用, 包括 sideToMove: moveForward/moveBack 会让它来回
       翻转, 回退完最后一手后它停在 color, 如果调用方传的 color 与棋盘当前的
       sideToMove 不一致就会改掉棋盘状态。所以先存原值, 结束时无条件恢复。
    */
    const int savedSideToMove = chess.sideToMove;
    chess.sideToMove = color;
    /*
       标签深度 = 决策深度 - 1, 但**上限 4**: 探索每一步都要跑一次这个深度的
       negamax, 而界面上"预训步数"默认 64 —— 深度 5 的标签就是 64 × ~90 ms ≈ 6 s/步,
       深度 6 是 ~12 s/步, 界面直接没法用。标签只是训练目标, 不需要和决策一样深。
    */
    int labelDepth = (maxDepth > 2) ? (maxDepth - 1) : 2;
    if (labelDepth > 4) {
        labelDepth = 4;
    }
    /* 整轮探索的时间上限 (见 exploreBudgetMs 的注释); 0 = 不限 */
    const auto exploreStart = std::chrono::steady_clock::now();
    int rolled = 0;

    for (int i = 0; i < rolloutSteps; i++) {
        if (exploreBudgetMs > 0) {
            const long long used =
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - exploreStart).count();
            if (used >= exploreBudgetMs) {
                break;
            }
        }
        std::vector<Step*> legal;
        chess.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            break;
        }
        Step chosen = *legal[(std::size_t)(std::rand() % (int)legal.size())];
        Steps::instance().put(legal);
        rolled++;

        const int next = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double d = 0;
        chess.moveForward(&chosen, d);

        /* 标签 = 0.5 * 手工评估 + 0.5 * 搜索评分, 都归一到 (-1,1), 刻意的保守 */
        /*
           视角修正 (2026-09): 这里必须用 **turn**, 不能用 next。
           moveForward 之后轮到 next 走, 所以 Chess::evaluate() (黑为正的绝对口径)
           翻到"走子方视角"得到的是 **next** 的视角; 而这一条样本的状态是按 turn 编码的
           (下面 encodeCanonical(turn, ...)), searchV = -negamax(next, ...) 也是 turn
           的视角。原来 hand 用 next、另两项用 turn, 于是手工那一半的符号是反的 ——
           两项按 0.5/0.5 相加等于互相抵消掉一半信号, 而且不报任何错。
           见 docs/agents_design.md §17。
        */
        const double hand = std::tanh((turn == Stone::COLOR_RED ? -chess.evaluate()
                                                                : chess.evaluate())
                                      / (double)EVAL_SCALE);
        const double searchV = -negamax(next, labelDepth - 1, -INF, INF, 1);
        const double clamped = std::max(-1.0, std::min(1.0, searchV));

        Sample s;
        s.state = RL::Tensor(STATE_DIM, 1);
        encodeCanonical(turn, s.state);
        s.target = (float)(0.5*hand + 0.5*clamped);
        samples.push_back(s);

        path.push_back(chosen);
        turn = next;
        if (chess.isGameOver() != Stone::COLOR_NONE) {
            break;
        }
    }

    for (std::size_t i = path.size(); i > 0; i--) {
        double d = 0;
        chess.moveBack(&path[i - 1], d);
    }
    chess.sideToMove = savedSideToMove;
    m_history.clear();

    if (samples.empty()) {
        m_exploreInfo = "探索 0 步 (没有可用局面)";
        return false;
    }

    /* 3) 小批量更新: 学习率刻意压到离线训练的 1/10 */
    RL::Tensor target(1, 1);
    const int batch = 16;
    int inBatch = 0;
    double errSum = 0.0;   /* 本轮的训练损失 (界面"训练损失曲线"用) */
    for (std::size_t i = 0; i < samples.size(); i++) {
        RL::Tensor &out = valueNet.forward(samples[i].state);
        target[0] = samples[i].target;
        errSum += std::fabs((double)out[0] - (double)samples[i].target);
        valueNet.backward(samples[i].state, RL::Loss::MSE::df(out, target));
        if (++inBatch >= batch) {
            valueNet.RMSProp(learningRate*0.1f, 0.9f, 0.0f);
            inBatch = 0;
        }
    }
    if (inBatch > 0) {
        valueNet.RMSProp(learningRate*0.1f, 0.9f, 0.0f);
    }
    /*
       上报这一轮的**训练损失** (平均 |预测 - 标签|)。以前只有离线训练
       (trainBatch) 会写 m_lastLoss, 而界面走的是这里的在线训练 —— 于是界面上
       EVAB 的损失曲线永远是空的 ("其他 agent 在对弈时无法显示训练损失")。
       放在回滚判断**之前**报告: 即使这一轮被回滚, "这轮训练到了什么程度"也是真实的
       数据, 而且回滚本身也要能在曲线上看出来 (下一轮损失会跳回去)。
    */
    m_lastLoss = (float)(errSum / (double)samples.size());

    /* 4) 变差就回滚: 与手工评估的差距明显变大 = 这一轮把网络带偏了 */
    const double gapAfter = netHandGap(12);
    if (gapAfter > gapBefore + 0.05) {
        backup.copyTo(valueNet);
        /*
           回滚时把 blend 也退回去: 网络刚被证明"这一轮把它带偏了", 那就更不该让
           它参与决策。退两步、进一步, 所以坏网络会自然被压回 0。
        */
        blend = std::max(0.0f, blend - 2.0f * blendStep);
        m_exploreInfo = "探索 " + std::to_string((int)samples.size())
                        + (samples.size() < (std::size_t)rolloutSteps ? " 步(时间上限)"
                                                                     : " 步")
                        + ", 更新后偏差 " + std::to_string(gapAfter)
                        + " > " + std::to_string(gapBefore) + " -> 已回滚 (blend "
                        + std::to_string(blend) + ")";
        return false;
    }

    /*
       5) 让学习到的评估真的参与决策 (blend 阶梯上升)。
       为什么需要这一步: `blend` 的初值是 0, 而在这次修改之前**没有任何地方改过它**
       —— 于是 EVAB 在界面上永远只用手工评估, 那个"learned eval"一次都没生效,
       与 ABAgent 在同一深度下几乎只是"更快的 AB"(实测两者走法 22/30 相同),
       自然很难赢。
       阶梯是保守的: 只在**这一轮没有把网络带偏**(上面那道门) 时才爬一小步, 上限
       blendMax (默认 0.5 —— 手工评估仍然是主力, 网络提供修正)。
       (test_evab_main.cpp 里量过"blend 直接跳到 1.0 会 0 胜 6 负", 所以既不能不爬,
       也不能一次爬到位。)
    */
    const float blendBefore = blend;
    blend = std::min(blendMax, blend + blendStep);
    if (blend != blendBefore) {
        m_leafEvals = 0;   /* 统计口径变了: 从这里开始叶子会真的走网络 */
    }
    m_exploreInfo = "探索 " + std::to_string((int)samples.size())
                    + " 步" + (samples.size() < (std::size_t)rolloutSteps ? "(时间上限)" : "")
                    + ", 价值网络更新 1 次 (|net-hand| "
                    + std::to_string(gapAfter) + ", blend "
                    + std::to_string(blendBefore) + "->" + std::to_string(blend) + ")";
    return true;
}

/* ============================================================
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  这里只报告**结构 / 口径**类事实, 判读写在各行末尾。EVAB 与别的 agent 不同的
 *  两点, 也正是这个面板最该说清楚的两点:
 *
 *   (1) **动作不是网络输出的**。它没有 128 槽的哈希动作头 (DQN/PG/SAC+AZ 那套
 *       编码在开局就会把几十个合法着法挤进更少的槽位), 走法由 alpha-beta 搜索
 *       在合法着法集上直接选出 —— 动作层不存在别名, 这一项天生干净。
 *   (2) **叶子评估是混合的**: blend=0 时叶子完全等于手工评估 (Chess::evaluate),
 *       此时它的行为就是 "ABAgent + 置换表 + 迭代加深 + 更好的走法排序"; blend>0
 *       时网络才参与。所以"网络到底有没有在起作用"要读 blend 与叶子评估次数,
 *       而不是读损失 —— 这曾经是个真 bug: blend 初值 0 且**没有任何地方改它**,
 *       界面上那个 "learned eval" 一次都没生效过 (见 evagent.h 里 blend 的注释)。
 *
 *  刻意**不**在这里跑搜索 (那是几百毫秒), 也**不**调 netHandGap() —— 后者会在
 *  this->chess 上随机试走, 而本函数被约定为只读 (GUI 线程可能在搜索线程工作时
 *  调用它, 见 aiagent.h 的契约)。
 *
 *  也不报"棋力": 损失只说明网络与自己的目标一致, 自对弈胜负里两边是同一份权重。
 *  棋力只有带置信区间的锚点对局 (bench_anchor) 能回答。
 * ============================================================ */
std::string EVABAgent::selfCheckReport() const
{
    char buf[320];
    std::string out;

    /* ---- 1. 表示层: 状态编码 ---- */
    std::snprintf(buf, sizeof(buf),
                  "状态 %d 维 = %d 平面 x %d 格 (14 棋子平面 + 3 规则上下文)\n",
                  STATE_DIM, PLANES, CELLS);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "规则上下文通道: 3 个 (无吃子进度 / 重复次数 / 被将) -> 三次重复、"
                  "自然限着、将军都可观测\n");
    out += buf;

    /* ---- 2. 动作层: 由搜索给出, 没有动作头 ---- */
    std::snprintf(buf, sizeof(buf),
                  "动作层: alpha-beta 在**合法着法集**上直接选 (无 %d 槽哈希头), "
                  "不存在动作别名\n", 128);
    out += buf;

    /* ---- 3. 叶子评估: 手工 / 网络的混合比例 ---- */
    std::snprintf(buf, sizeof(buf),
                  "叶子评估: blend=%.2f (0=纯手工评估, 1=网络接管), 上限 %.2f, "
                  "每步 +%.2f, 手工尺度 %.1f\n",
                  (double)blend, (double)blendMax, (double)blendStep, (double)EVAL_SCALE);
    out += buf;
    if (getLeafEvals() == 0) {
        std::snprintf(buf, sizeof(buf),
                      "  blend=0 时网络前向一次都没发生 -> 这一支现在等价于"
                      "\"AB + 置换表 + 迭代加深\"\n");
    } else {
        std::snprintf(buf, sizeof(buf),
                      "  本轮已发生网络叶子评估 %lld 次 (blend>0 的代价: 每次前向 "
                      "1260->%d->1, 实测让 depth5 从 60 ms/步涨到 ~800 ms/步)\n",
                      getLeafEvals(), m_hiddenDim);
    }
    out += buf;

    /* ---- 4. 搜索预算与上一次搜索的规模 ---- */
    const long long probes = getTTProbes();
    const double hitRate = (probes > 0) ? (double)getTTHits() / (double)probes : 0.0;
    std::snprintf(buf, sizeof(buf),
                  "搜索: 深度上限 %d, 时间上限 %lld ms | 上次到达深度 %d, 节点 %lld\n",
                  maxDepth, timeBudgetMs, getReachedDepth(), getNodes());
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "置换表: %zu 项 | 探测 %lld 次, 命中 %lld (%.1f%%), 写入 %lld\n",
                  m_tt.size(), probes, getTTHits(), 100.0 * hitRate, getTTStores());
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "最近一次根评分 (走棋方视角, ±%.0f 为将杀): %.3f\n",
                  MATE, m_lastRootScore);
    out += buf;

    /* ---- 5. 在线训练口径 ---- */
    const float loss = getLastTrainLoss();
    std::snprintf(buf, sizeof(buf),
                  "在线训练: 学习率 %.4f, 结果标签权重 %.2f, 随机走子概率 %.2f, "
                  "探索时间上限 %lld ms\n",
                  (double)learningRate, (double)outcomeWeight, (double)exploreEps,
                  exploreBudgetMs);
    out += buf;
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "最近一次在线损失 (平均 |预测-标签|): %.4f | 参数量 %lld\n",
                      (double)loss, valueNet.paramCount());
    } else {
        std::snprintf(buf, sizeof(buf),
                      "最近一次在线损失: 还没训练过 | 参数量 %lld\n",
                      valueNet.paramCount());
    }
    out += buf;

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局\n";
    return out;
}