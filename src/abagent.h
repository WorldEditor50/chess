#ifndef ABAGENT_H
#define ABAGENT_H

#include <string>
#include "chess.h"
#include "aiagent.h"

/*
 * ABAgent - Alpha-Beta Pruning Chess Agent
 *
 * Implements full alpha-beta pruning search with:
 *   - MVV-LVA move ordering
 *   - Quiescence search (horizon effect mitigation)
 *   - Piece-Square Tables (via chess.evaluate())
 *
 * The algorithm is self-contained in this class; no search logic
 * remains in Chess.
 */
class ABAgent : public AgentBase
{
private:
    Chess &chess;
    int maxDepth;

    /* Alpha-beta search internals */
    double minimizeAlpha(int color, int depth, double beta, double &totalReward);
    double maximizeBeta(int color, int depth, double alpha, double &totalReward);
    double quiescenceSearch(int color, double alpha, double beta, int depth);
    /*
     * 静态搜索入口, 统一转换到"黑方视角"(与 minimizeAlpha/maximizeBeta 的
     * 返回值约定一致)。quiescenceSearch 内部是 negamax, 返回的是**当前走棋方**
     * 视角的分值, 所以轮到红方走时必须取负。
     *   lo / hi 是黑方视角的窗口上下界。
     */
    double quiescenceBlackView(int color, double lo, double hi);
    void orderMoves(std::vector<Step*> &steps);

public:
    ABAgent(Chess &chess_, int depth = 4);

    /* AgentBase interface */
    Step getBestMove(int color) override;
    Step getBestMove(int color, int depth);  /* with temporary depth override */
    std::string getName() const override;

    /* Legacy alias (delegates to getBestMove) */
    Step findBestMove(int color);

    /* Get/set search depth */
    void setDepth(int depth) { maxDepth = depth; }
    int getDepth() const { return maxDepth; }
};

#endif // ABAGENT_H
