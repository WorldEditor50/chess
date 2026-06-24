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
