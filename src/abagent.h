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
    /* 最近一次搜索的根分值 (黑方视角) 与"这次搜索有没有分数" */
    double m_lastScore = 0.0;
    bool   m_scoreValid = false;

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
    /*
     * 最近一次 findBestMove() 的**根节点搜索分**（黑方视角: 正 = 黑优）。
     * 口径来自 findBestMove 的根节点类型选择: 黑方是 MAX 节点、红方是 MIN 节点,
     * 所以两个分支给出的 beta/alpha 都是"black-perspective 得分"。
     *
     * 用途: 价值头蒸馏 (Step 2) —— 把"AB 搜了 3~4 层的结论"当作 critic 的监督目标,
     * 它比手工局面评估多了一层**搜索**的信息 (手工评估是深度 0)。
     * getScoreValid() 为 false 表示那次搜索没有合法走法 (没有分数可言)。
     */
    double getLastScore() const { return m_lastScore; }
    bool getScoreValid() const { return m_scoreValid; }
};

#endif // ABAGENT_H
