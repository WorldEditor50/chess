#include "abagent.h"
#include <algorithm>
#include <cstdlib>

/*
 * ABAgent - Alpha-Beta Pruning Chess Agent
 *
 * Full alpha-beta pruning search implementation, migrated from Chess.
 * Uses chess.evaluate() for leaf evaluation, chess.sample() for move
 * generation, and chess.moveForward()/moveBack() for board state.
 */

/* ============================================================
 *  辅助函数 (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
 * ============================================================ */

/* 棋子价值排序 (值越大越有价值被吃) */
static int victimValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 1000;
    case Stone::TYPE_CHE:   return 500;
    case Stone::TYPE_MA:    return 300;
    case Stone::TYPE_PAO:   return 300;
    case Stone::TYPE_SHI:   return 200;
    case Stone::TYPE_XIANG: return 200;
    case Stone::TYPE_BING:  return 100;
    default:                return 0;
    }
}

/* 攻击方价值排序 (LV: 用小棋子吃大棋子应优先搜索) */
static int attackerValue(int type)
{
    switch (type) {
    case Stone::TYPE_JIANG: return 0;    /* 将/帅吃子 */
    case Stone::TYPE_BING:  return 1;    /* 兵吃子 */
    case Stone::TYPE_SHI:   return 2;
    case Stone::TYPE_XIANG: return 3;
    case Stone::TYPE_MA:    return 4;
    case Stone::TYPE_PAO:   return 4;
    case Stone::TYPE_CHE:   return 5;    /* 车是最"贵"的攻击方，应后搜 */
    default:                return 10;
    }
}

/* ============================================================
 *  Constructor
 * ============================================================ */
ABAgent::ABAgent(Chess &chess_, int depth)
    : AgentBase(), chess(chess_), maxDepth(depth)
{
}

/* ============================================================
 *  AgentBase interface
 * ============================================================ */
Step ABAgent::getBestMove(int color)
{
    return findBestMove(color);
}

Step ABAgent::getBestMove(int color, int depth)
{
    int savedDepth = maxDepth;
    maxDepth = depth;
    Step result = findBestMove(color);
    maxDepth = savedDepth;
    return result;
}

std::string ABAgent::getName() const
{
    return "Alpha-Beta Pruning (depth=" + std::to_string(maxDepth) + ")";
}

Step ABAgent::findBestMove(int color)
{
    std::vector<Step*> steps;
    chess.sample(color, steps);

    /* 优化: 走法排序 */
    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double totalReward = 0;
    Step* best = nullptr;

    /*
     * 根据走棋方正确选择根节点类型:
     *   - 黑方 (AI): 根为 MAX, 最大化 black-perspective 得分
     *   - 红方:     根为 MIN, 最小化 black-perspective 得分
     */
    if (color == Stone::COLOR_BLACK) {
        /* 黑方 = MAX 节点 */
        double beta = -Stone::value_infi;
        for (Step *s : steps) {
            chess.moveForward(s, totalReward);
            double r = minimizeAlpha(Stone::COLOR_RED, maxDepth - 1, beta, totalReward);
            chess.moveBack(s, totalReward);
            if (r > beta) {
                best = s;
                beta = r;
            }
        }
    } else {
        /* 红方 = MIN 节点 */
        double alpha = Stone::value_infi;
        for (Step *s : steps) {
            chess.moveForward(s, totalReward);
            double r = maximizeBeta(Stone::COLOR_BLACK, maxDepth - 1, alpha, totalReward);
            chess.moveBack(s, totalReward);
            if (r < alpha) {
                best = s;
                alpha = r;
            }
        }
    }

    Step step;
    if (best != nullptr) {
        step = *best;
    }
    Steps::instance().put(steps);
    return step;
}

/* ============================================================
 *  orderMoves - 走法排序 (MVV-LVA)
 * ============================================================ */
void ABAgent::orderMoves(std::vector<Step*> &steps)
{
    std::vector<int> scores(steps.size(), 0);
    for (std::size_t i = 0; i < steps.size(); i++) {
        Step *s = steps[i];
        if (s->nextId != Stone::ID_NONE) {
            Stone *victim = chess.stones[s->nextId];
            Stone *attacker = chess.stones[s->id];
            if (victim != nullptr && attacker != nullptr) {
                scores[i] = victimValue(victim->type) * 100 - attackerValue(attacker->type);
            }
        }
    }
    for (std::size_t i = 0; i + 1 < steps.size(); i++) {
        for (std::size_t j = i + 1; j < steps.size(); j++) {
            if (scores[j] > scores[i]) {
                std::swap(steps[i], steps[j]);
                std::swap(scores[i], scores[j]);
            }
        }
    }
}

/* ============================================================
 *  quiescenceSearch - 静态搜索 (缓解水平线效应)
 *
 *  只搜索吃子走法, 稳定局面评估。
 *  注意: chess.evaluate() 始终返回黑方(AI)视角的评估值。
 *
 *  外层搜索保证:
 *    - MAXIMIZE 节点 (黑方走) 调用时 color=BLACK, 查找让黑方得分更高的吃子
 *    - MINIMIZE 节点 (红方走) 调用时 color=RED,   查找让黑方得分更低的吃子
 *
 *  返回值: 始终是黑方(AI)视角的评估值 (正值=黑方有利)
 * ============================================================ */
double ABAgent::quiescenceSearch(int color, double alpha, double beta, int depth)
{
    /* 静态评估: chess.evaluate() 返回黑方视角, 需要转换到当前走棋方视角 */
    double standPat = chess.evaluate();
    if (color == Stone::COLOR_RED) {
        standPat = -standPat;  /* RED 要走, 评估从 RED 视角看是 neg(黑方视角) */
    }

    if (depth <= 0) {
        return standPat;
    }

    /* Negamax 标准剪枝 */
    if (standPat >= beta) {
        return beta;
    }
    if (standPat > alpha) {
        alpha = standPat;
    }

    /* 生成当前方的所有走法, 只保留吃子走法 */
    std::vector<Step*> allSteps;
    chess.sample(color, allSteps);

    std::vector<Step*> captures;
    for (Step *s : allSteps) {
        if (s->nextId != Stone::ID_NONE) {
            captures.push_back(s);
        } else {
            Steps::instance().put(std::vector<Step*>(1, s));
        }
    }

    /* MVV-LVA 排序 */
    if (captures.size() > 1) {
        std::vector<int> scores(captures.size(), 0);
        for (std::size_t i = 0; i < captures.size(); i++) {
            Stone *victim = chess.stones[captures[i]->nextId];
            Stone *attacker = chess.stones[captures[i]->id];
            if (victim != nullptr && attacker != nullptr) {
                scores[i] = victimValue(victim->type) * 100 - attackerValue(attacker->type);
            }
        }
        for (std::size_t i = 0; i + 1 < captures.size(); i++) {
            for (std::size_t j = i + 1; j < captures.size(); j++) {
                if (scores[j] > scores[i]) {
                    std::swap(captures[i], captures[j]);
                    std::swap(scores[i], scores[j]);
                }
            }
        }
    }

    double totalReward = 0;
    int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    for (Step *s : captures) {
        chess.moveForward(s, totalReward);

        Stone *victim = (s->nextId != Stone::ID_NONE) ? chess.stones[s->nextId] : nullptr;
        if (victim != nullptr && victim->type == Stone::TYPE_JIANG) {
            chess.moveBack(s, totalReward);
            Steps::instance().put(captures);
            /* 吃将: 当前走棋方视角下的最大值 */
            return Stone::value_infi;
        }

        /* Negamax: 递归到对方节点, 取负交换窗口 */
        double r = -quiescenceSearch(color_, -beta, -alpha, depth - 1);
        chess.moveBack(s, totalReward);

        if (r >= beta) {
            Steps::instance().put(captures);
            return beta;
        }
        if (r > alpha) {
            alpha = r;
        }
    }

    Steps::instance().put(captures);
    return alpha;
}

/* ============================================================
 *  minimizeAlpha - MIN 节点
 *
 *  当前走棋方试图最小化评估值.
 *  返回值: 从黑方 (AI) 视角的评估值 (正值=对黑方有利)
 * ============================================================ */
double ABAgent::minimizeAlpha(int color, int depth, double beta, double &totalReward)
{
    int gameResult = chess.isGameOver();
    if (gameResult == Stone::COLOR_BLACK) {
        return Stone::value_infi;   /* 黑方赢 -> AI(黑方)有利 */
    }
    if (gameResult == Stone::COLOR_RED) {
        return -Stone::value_infi;  /* 红方赢 -> AI(黑方)不利 */
    }

    if (depth == 0) {
        return chess.evaluate();
    }

    std::vector<Step*> steps;
    chess.sample(color, steps);

    if (steps.empty()) {
        /* 当前方无合法走法 → 对方获胜 */
        return (color == Stone::COLOR_RED) ? Stone::value_infi : -Stone::value_infi;
    }

    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double alpha = Stone::value_infi;
    for (Step *s : steps) {
        chess.moveForward(s, totalReward);
        int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double r = maximizeBeta(color_, depth - 1, alpha, totalReward);
        chess.moveBack(s, totalReward);

        if (r < alpha) {
            alpha = r;
        }
        if (alpha <= beta) {
            break;
        }
    }
    Steps::instance().put(steps);
    return alpha;
}

/* ============================================================
 *  maximizeBeta - MAX 节点
 *
 *  当前走棋方试图最大化评估值.
 * ============================================================ */
double ABAgent::maximizeBeta(int color, int depth, double alpha, double &totalReward)
{
    int gameResult = chess.isGameOver();
    if (gameResult == Stone::COLOR_BLACK) {
        return Stone::value_infi;   /* 黑方赢 -> AI(黑方)有利 */
    }
    if (gameResult == Stone::COLOR_RED) {
        return -Stone::value_infi;  /* 红方赢 -> AI(黑方)不利 */
    }

    /* 进入静态搜索时使用完全开放的窗口 (-infi, +infi)
     * 避免父节点 (MIN 节点) 传入的 alpha = value_infi 导致零宽窗口误剪枝 */
    if (depth == 0) {
        return quiescenceSearch(color, -Stone::value_infi, Stone::value_infi, 3);
    }

    std::vector<Step*> steps;
    chess.sample(color, steps);

    if (steps.empty()) {
        /* 当前方无合法走法 → 对方获胜 */
        return (color == Stone::COLOR_RED) ? Stone::value_infi : -Stone::value_infi;
    }

    if (steps.size() > 1) {
        orderMoves(steps);
    }

    double beta = -Stone::value_infi;
    for (Step *s : steps) {
        chess.moveForward(s, totalReward);
        int color_ = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        double r = minimizeAlpha(color_, depth - 1, beta, totalReward);
        chess.moveBack(s, totalReward);

        if (r > beta) {
            beta = r;
        }
        if (beta >= alpha) {
            break;
        }
    }
    Steps::instance().put(steps);
    return beta;
}
