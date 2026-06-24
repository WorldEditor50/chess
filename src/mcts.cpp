#include "mcts.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>

/* ============================================================
 * MCTSNode implementation
 * ============================================================ */

MCTSNode::MCTSNode()
    : visitCount(0), totalReward(0.0), parentID(-1), currentColor(Stone::COLOR_NONE)
{
}

MCTSNode::MCTSNode(int parentID_, const Step &step_, int color_)
    : visitCount(0), totalReward(0.0), parentID(parentID_), step(step_), currentColor(color_)
{
}

double MCTSNode::getUCB1(double totalParentVisits, double C) const
{
    /* UCB1 = W/N + C * sqrt(ln(N_parent) / N)
     *
     * If visitCount is 0, return infinity to encourage exploring
     * unvisited nodes first.
     */
    if (visitCount == 0) {
        return std::numeric_limits<double>::max();
    }
    double exploitation = totalReward / (double)visitCount;
    double exploration = C * std::sqrt(std::log(totalParentVisits) / (double)visitCount);
    return exploitation + exploration;
}

/* ============================================================
 * MCTS implementation
 * ============================================================ */

MCTS::MCTS(Chess &chess_, double explorationConstant)
    : AgentBase(), chess(chess_), C(explorationConstant)
{
    /* Seed the RNG for random move selection during rollout */
    std::srand((unsigned int)std::time(nullptr));
}

/* AgentBase interface */
Step MCTS::getBestMove(int color)
{
    return findBestMove(color, 800);
}

std::string MCTS::getName() const
{
    return "MCTS (C=" + std::to_string(C) + ")";
}

void MCTS::clearTree()
{
    nodes.clear();
}

/* ------------------------------------------------------------
 * selectPromisingNode
 *
 * Traverses the tree from rootID downwards using the UCB1 formula
 * to select the most promising child at each level. Continues as
 * long as the current node is fully expanded (no untriedMoves) and
 * has at least one child.
 *
 * Each chosen move is executed on the board, keeping the board
 * state synchronized with the current node in the tree.
 *
 * Returns the nodeID of the leaf where selection stopped.
 * Fills `path` with all node IDs from rootID to the leaf (inclusive).
 * ---------------------------------------------------------- */
int MCTS::selectPromisingNode(int rootID, std::vector<int> &path)
{
    int nodeID = rootID;

    while (nodes[nodeID].isFullyExpanded()
           && !nodes[nodeID].childIDs.empty()) {

        double parentVisits = (double)nodes[nodeID].visitCount;
        int bestChild = -1;
        double bestUCB = -std::numeric_limits<double>::max();

        for (int childID : nodes[nodeID].childIDs) {
            double ucb = nodes[childID].getUCB1(parentVisits, C);
            if (ucb > bestUCB) {
                bestUCB = ucb;
                bestChild = childID;
            }
        }

        if (bestChild < 0) {
            break;  /* no reachable child (should not happen) */
        }

        /* Execute the move leading to this child */
        double dummyReward = 0.0;
        const Step &move = nodes[bestChild].step;
        chess.moveForward(&move, dummyReward);

        nodeID = bestChild;
        path.push_back(nodeID);
    }

    return nodeID;
}

/* ------------------------------------------------------------
 * expandNode
 *
 * If the given node has untried moves, pick one uniformly at
 * random, execute it on the board, create a new child node,
 * pre-generate all legal moves for the opponent (stored as
 * untriedMoves in the child), and register the child in the tree.
 *
 * Returns the ID of the newly expanded child node.
 * ---------------------------------------------------------- */
int MCTS::expandNode(int nodeID, std::vector<int> &path)
{
    double dummyReward = 0.0;

    /* Pick a random untried move */
    int moveIdx = std::rand() % (int)nodes[nodeID].untriedMoves.size();
    Step chosenMove = nodes[nodeID].untriedMoves[moveIdx];

    /* Remove it from the untried list */
    nodes[nodeID].untriedMoves.erase(
        nodes[nodeID].untriedMoves.begin() + moveIdx);

    /* Execute the move */
    chess.moveForward(&chosenMove, dummyReward);

    /* Determine the color to move at the new position */
    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                        ? Stone::COLOR_BLACK
                        : Stone::COLOR_RED;

    /* Create the new child node */
    MCTSNode newNode(nodeID, chosenMove, nextColor);

    /* Pre-compute all legal moves for the child so they
     * can be expanded later */
    std::vector<Step*> newMoves;
    chess.sample(nextColor, newMoves);
    for (Step *s : newMoves) {
        newNode.untriedMoves.push_back(*s);
    }
    Steps::instance().put(newMoves);

    /* Register the child in the tree */
    nodes.push_back(newNode);
    int newNodeID = (int)nodes.size() - 1;
    nodes[nodeID].childIDs.push_back(newNodeID);

    /* Update path to include the new node */
    nodeID = newNodeID;
    path.push_back(nodeID);

    return nodeID;
}

/* ------------------------------------------------------------
 * backpropagate
 *
 * Walk back up the path, updating visitCount and totalReward
 * for every node. The reward sign is flipped at each level
 * because the perspective alternates between the two players.
 * ---------------------------------------------------------- */
void MCTS::backpropagate(const std::vector<int> &path, double reward)
{
    for (int i = (int)path.size() - 1; i >= 0; i--) {
        nodes[path[i]].visitCount++;
        nodes[path[i]].totalReward += reward;
        reward = -reward;   /* flip for the opponent */
    }
}

/* ------------------------------------------------------------
 * simulateRandomPlay
 *
 * Play random legal moves for both sides starting from the
 * current board state until the game ends (checkmate or
 * stalemate). Returns +1 if `color` wins, -1 if `color`
 * loses, and 0 if it's a draw.
 *
 * All moves played during the simulation are undone before
 * returning, leaving the board in its original state.
 * ---------------------------------------------------------- */
double MCTS::simulateRandomPlay(int color)
{
    std::vector<Step> simSteps;
    double dummy = 0.0;
    int currColor = color;

    while (true) {
        /* Check for terminal state (checkmate) */
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            /* gameResult is the winning color */
            for (auto it = simSteps.rbegin();
                 it != simSteps.rend(); ++it) {
                chess.moveBack(&(*it), dummy);
            }
            return (gameResult == color) ? 1.0 : -1.0;
        }

        /* Generate all legal moves for the current player */
        std::vector<Step*> moves;
        chess.sample(currColor, moves);

        if (moves.empty()) {
            /* Current player has no legal moves (stalemate / get
             * mated) -> they lose */
            Steps::instance().put(moves);
            for (auto it = simSteps.rbegin();
                 it != simSteps.rend(); ++it) {
                chess.moveBack(&(*it), dummy);
            }
            return (currColor == color) ? -1.0 : 1.0;
        }

        /* Pick a move uniformly at random */
        int idx = std::rand() % (int)moves.size();
        Step *chosen = moves[idx];
        chess.moveForward(chosen, dummy);
        simSteps.push_back(*chosen);
        Steps::instance().put(moves);

        /* Switch to the other side */
        currColor = (currColor == Stone::COLOR_RED)
                        ? Stone::COLOR_BLACK
                        : Stone::COLOR_RED;
    }
}

/* ------------------------------------------------------------
 * findBestMove
 *
 * Main MCTS search entry point. Builds the root node from the
 * current board state, then performs `iterations` MCTS iterations.
 *
 * Each iteration:
 *   1. SELECTION   - traverse tree using UCB1 (selectPromisingNode)
 *   2. EXPANSION   - add a new child node if untried moves exist
 *                    (expandNode)
 *   3. SIMULATION  - random rollout from leaf until game ends
 *                    (simulateRandomPlay)
 *   4. BACKPROPAGATION - propagate result up the path
 *                        (backpropagate)
 *
 * After all iterations, returns the child of the root with the
 * highest visit count.
 * ---------------------------------------------------------- */
Step MCTS::findBestMove(int color, int iterations)
{
    nodes.clear();

    /* --------------------------------------------------------
     * Build the root node: generate all legal moves for the
     * current color at the current board state.
     * -------------------------------------------------------- */
    MCTSNode root;
    root.currentColor = color;
    root.parentID = -1;

    std::vector<Step*> rootMoves;
    chess.sample(color, rootMoves);
    for (Step *s : rootMoves) {
        root.untriedMoves.push_back(*s);
    }
    Steps::instance().put(rootMoves);

    nodes.push_back(root);
    int rootID = 0;

    /* --------------------------------------------------------
     * Main MCTS loop
     * -------------------------------------------------------- */
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<int> path;
        path.push_back(rootID);

        /* Phase 1 + 2: SELECTION & EXPANSION
         *
         * First, traverse the tree using UCB1 to find a leaf.
         * The leaf will be either a node with untried moves or
         * a terminal node. As we descend, moves are executed on
         * the board.
         */
        int nodeID = selectPromisingNode(rootID, path);

        /* If the selected node has untried moves, expand it */
        if (!nodes[nodeID].untriedMoves.empty()) {
            nodeID = expandNode(nodeID, path);
        }

        /* Phase 3: SIMULATION (ROLLOUT) */
        double reward = simulateRandomPlay(
                              nodes[nodeID].currentColor);

        /* Phase 4: BACKPROPAGATION */
        backpropagate(path, reward);

        /* ==============================================
         * Undo all moves that were played during this
         * iteration (selection + expansion), restoring
         * the original board state.
         * ============================================== */
        for (int i = (int)path.size() - 1; i > 0; i--) {
            double dummyReward = 0.0;
            const Step &s = nodes[path[i]].step;
            chess.moveBack(&s, dummyReward);
        }
    }

    /* --------------------------------------------------------
     * Choose the best move: the child of the root with the
     * highest visit count.
     * -------------------------------------------------------- */
    int bestChildID = -1;
    int maxVisits = -1;
    for (int childID : nodes[rootID].childIDs) {
        if (nodes[childID].visitCount > maxVisits) {
            maxVisits = nodes[childID].visitCount;
            bestChildID = childID;
        }
    }

    if (bestChildID >= 0) {
        return nodes[bestChildID].step;
    }

    /* No legal moves - return empty Step */
    return Step();
}
