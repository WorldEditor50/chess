#ifndef AIAGENT_H
#define AIAGENT_H

#include <string>
#include "chess.h"

/*
 * AgentBase - Abstract base class for all chess AI agents
 *
 * Provides a unified interface for move selection across all agent types:
 *   - ABAgent        : Alpha-Beta Pruning
 *   - MCTS           : Monte Carlo Tree Search (random rollouts)
 *   - PGEagent       : Policy Gradient (REINFORCE)
 *   - DQNAgent       : Deep Q-Network
 *   - PPOMCTSAgent   : PPO + MCTS (AlphaZero-style)
 */
class AgentBase
{
public:
    virtual ~AgentBase() = default;

    /* --- Core interface --- */

    /* Select the best move for the given color */
    virtual Step getBestMove(int color) = 0;

    /* Human-readable agent name */
    virtual std::string getName() const = 0;

    /* Optional: reset internal state (tree, history, etc.) */
    virtual void resetState() {}
};

#endif // AIAGENT_H
