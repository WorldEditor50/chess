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

};

#endif // MCTS_H
