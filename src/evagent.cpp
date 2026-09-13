#include "evagent.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

/*
 * EVABAgent - 实现说明见 evagent.h 与 docs/agent_evab_design.md。
 */

/* ============================================================
 *  构造 / 基本信息
 * ============================================================ */
EVABAgent::EVABAgent(Chess &chess_, int hiddenDim, int depth, long long budgetMs)
    : AgentBase(),
      chess(chess_),
      valueNet(RL::Layer<RL::Tanh>::_(STATE_DIM, hiddenDim, true, true),
               RL::Layer<RL::Tanh>::_(hiddenDim, 1, true, true)),
      blend(0.0f),
      maxDepth(depth),
      timeBudgetMs(budgetMs),
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
}

std::string EVABAgent::getName() const
{
    char buf[128];
    std::snprintf(buf, sizeof(buf), "EVAB (learned eval, depth=%d, blend=%.2f)",
                  maxDepth, (double)blend);
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
    const bool redToMove = (color == Stone::COLOR_RED);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        if (s->type < 0 || s->type >= 7) {
            continue;
        }
        /* 镜像: 轮到黑方时把棋盘上下翻转, 于是"己方"永远在 x 大的那一侧 */
        const int x = redToMove ? s->pos.x : (9 - s->pos.x);
        const int cell = x * 9 + s->pos.y;
        const bool own = (s->color == color);
        const int plane = s->type + (own ? 0 : 7);
        state[plane * CELLS + cell] = 1.0f;
    }
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

    return trainBatch(valueNet, samples, batchSize, learningRate);
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
    const int labelDepth = (maxDepth > 2) ? (maxDepth - 1) : 2;

    for (int i = 0; i < rolloutSteps; i++) {
        std::vector<Step*> legal;
        chess.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            break;
        }
        Step chosen = *legal[(std::size_t)(std::rand() % (int)legal.size())];
        Steps::instance().put(legal);

        const int next = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double d = 0;
        chess.moveForward(&chosen, d);

        /* 标签 = 0.5 * 手工评估 + 0.5 * 搜索评分, 都归一到 (-1,1), 刻意的保守 */
        const double hand = std::tanh((next == Stone::COLOR_RED ? -chess.evaluate()
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
    for (std::size_t i = 0; i < samples.size(); i++) {
        RL::Tensor &out = valueNet.forward(samples[i].state);
        target[0] = samples[i].target;
        valueNet.backward(samples[i].state, RL::Loss::MSE::df(out, target));
        if (++inBatch >= batch) {
            valueNet.RMSProp(learningRate*0.1f, 0.9f, 0.0f);
            inBatch = 0;
        }
    }
    if (inBatch > 0) {
        valueNet.RMSProp(learningRate*0.1f, 0.9f, 0.0f);
    }

    /* 4) 变差就回滚: 与手工评估的差距明显变大 = 这一轮把网络带偏了 */
    const double gapAfter = netHandGap(12);
    if (gapAfter > gapBefore + 0.05) {
        backup.copyTo(valueNet);
        m_exploreInfo = "探索 " + std::to_string((int)samples.size())
                        + " 步, 更新后偏差 " + std::to_string(gapAfter)
                        + " > " + std::to_string(gapBefore) + " -> 已回滚";
        return false;
    }
    m_exploreInfo = "探索 " + std::to_string((int)samples.size())
                    + " 步, 价值网络更新 1 次 (|net-hand| "
                    + std::to_string(gapAfter) + ")";
    return true;
}
