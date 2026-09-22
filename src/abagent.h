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

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  先说结论性的一句: ABAgent 是**纯搜索 agent, 没有任何可学习权重** ——
     *  叶子价值全部来自 Chess::evaluate() (材质 + 子力位置表, 常量写死在 chess.cpp)。
     *  所以 getLastTrainLoss() 恒为 NaN、训练损失曲线对它**永远**是空的; 面板上必须
     *  写出这一句, 否则"没有曲线"会被读成"训练没跑起来/权重没加载"。
     *
     *  报告四类事实:
     *   1. **搜索配置**: 深度 (界面用 AB_DEPTH = 4, 见 chessboard.cpp 的常量说明) 与
     *      评估来源 (手工 evaluate(), 不是网络)。
     *   2. **决策合法性自检** (回归指示器): 在标准开局的**棋盘副本**上跑一次深度 1
     *      搜索, 检查返回的走法是否在合法走法集里。它钉的是 2026-09 用户报的那个
     *      bug —— findBestMove 在"每一步都必输"时返回默认构造的 Step (valid=false),
     *      调用方读成"无棋可走" -> 判负, 而棋盘上明明还有棋。详见 abagent.cpp 里
     *      findBestMove 根节点分支的长注释 (那里已修) 与本函数实现处的注释。
     *   3. 标准开局的**合法走法数** + evaluate() 的黑方视角读数: 与训练进度无关的
     *      确定性读数, 当"尺子"用 —— 谁改了走法生成或评估口径, 这两个数会动。
     *   4. 最近一次搜索的根分值 (黑方视角) 和它有没有分数 (getScoreValid())。
     *
     *  **只读**: 只用 chess 的**副本**加一个**临时局部 agent**, 不改任何成员、不动
     *  棋盘 —— 它会在对局中途被 GUI 线程调用, 而 this->chess 可能正被搜索线程使用。
     *  临时 agent 是刻意为之, 不是绕路: findBestMove() 不是 const (要写 m_lastScore/
     *  m_scoreValid), 与其 const_cast 掉 this 去跑一次真搜索 (还会**覆盖**本 agent
     *  供价值头蒸馏用的 m_lastScore), 不如在副本上另起 ABAgent(probe, 1) —— 它的
     *  搜索结果只落在临时对象里, 本 agent 的分值一个字节都不动。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;
};

#endif // ABAGENT_H
