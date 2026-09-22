#ifndef MCTS_H
#define MCTS_H

#include <vector>
#include <limits>
#include "chess.h"
#include "aiagent.h"

class MCTSNode
{
public:
    int visitCount;                 // N(s) - how many times this node was visited
    double totalReward;             // W(s) - total reward accumulated
    int parentID;                   // index of parent node (-1 for root)
    Step step;                      // the move that led to this node from its parent
    std::vector<int> childIDs;      // indices of child nodes
    std::vector<Step> untriedMoves; // moves from this node's state not yet expanded
    int currentColor;              // color to move at this node's position

    MCTSNode();
    MCTSNode(int parentID_, const Step &step_, int color_);

    double getUCB1(double totalParentVisits, double C) const;
    bool isFullyExpanded() const { return untriedMoves.empty(); }
};

class MCTS : public AgentBase
{
private:
    Chess &chess;
    std::vector<MCTSNode> nodes;
    double C; // UCB1 exploration constant

    /*
     * Select the most promising leaf node from the given root using UCB1.
     *
     * Traverses from rootID downwards, always picking the child with the
     * highest UCB1 score, until reaching a node that has untried moves or
     * has no children. Each selected move is executed on the board so the
     * board state reflects the traversal path.
     *
     * @param rootID  Index of the node to start selection from
     * @param path    Output: filled with node IDs along the traversed path
     * @return        The nodeID of the selected leaf node
     */
    int selectPromisingNode(int rootID, std::vector<int> &path);

    /*
     * Expand a node by picking one untried move uniformly at random.
     *
     * Executes the chosen move, creates a new child node, generates all
     * legal replies for the opponent (as the child's untriedMoves), and
     * registers the child in the tree. The board state is advanced by
     * the chosen move.
     *
     * @param nodeID  Index of the node to expand
     * @param path    Updated to include the new child node ID
     * @return        The ID of the newly created child node
     */
    int expandNode(int nodeID, std::vector<int> &path);

    /*
     * Simulate (rollout) from the current board position by playing random
     * legal moves for both sides until the game ends. All simulation moves
     * are undone before returning.
     *
     * @param color  The side to move at the current leaf position
     * @return       +1.0 if color wins, -1.0 if color loses, 0.0 for draw
     */
    double simulateRandomPlay(int color);

    /*
     * Backpropagate the simulation result up the search path.
     *
     * Increments visitCount and accumulates totalReward for each node
     * in the path (from leaf to root). Reward is negated at each level
     * to alternate between the two players' perspectives.
     *
     * @param path    The node IDs from root to the leaf (inclusive)
     * @param reward  The result from the leaf's perspective (+1/-1/0)
     */
    void backpropagate(const std::vector<int> &path, double reward);

public:
    MCTS(Chess &chess_, double explorationConstant = 1.414);

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /* Primary search entry point */
    Step findBestMove(int color, int iterations);
    void clearTree();

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  MCTS 同样是**纯搜索 agent, 没有任何可学习权重**: 叶子信号来自
     *  simulateRandomPlay() 的随机走子终局结果 (±1/0), 树本身每次 findBestMove
     *  都会 clear() 重建。所以 getLastTrainLoss() 恒为 NaN、损失曲线对它永远是空的。
     *
     *  报告四类事实:
     *   1. **搜索配置**: UCB1 探索常数 C 与当前树规模 (nodes.size(), 只读快照 ——
     *      它是"量级"不是精确值: 搜索线程若正在同一对象上跑, 这个数正在变; 契约
     *      禁止碰的是 this->chess, 不是这个计数) 以及叶子信号是**随机对局结果**、
     *      天生高方差这件事 —— 单次模拟的回报与棋力无关, 只有多次平均才有意义。
     *   2. 界面给它的模拟预算: MCTS_SIMS = 800 (chessboard.cpp 的常量; 这里写
     *      字面量, 不去 include GUI 头)。
     *   3. **决策合法性自检** (回归指示器): 在标准开局的副本上跑 1 次模拟, 检查返回
     *      的走法在合法集里。它钉的是 findBestMove 在"根有合法走法、但一次扩展都没
     *      发生"时返回默认 Step (valid=false) 的问题 —— 调用方把这个读成"没棋可走"
     *      而判负。详见 mcts.cpp 里 findBestMove 的兜底注释。
     *   4. 标准开局的合法走法数 —— 也就是那 800 次模拟实际摊到的分支因子。
     *
     *  **只读**: 只用 chess 的**副本**加一个**临时局部 agent** (findBestMove 与
     *  getBestMove 都不是 const, 而本函数是 const), 不改任何成员、不动棋盘。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;
};

#endif // MCTS_H
