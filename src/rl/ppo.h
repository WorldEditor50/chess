#ifndef PPO_H
#define PPO_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"
#include "annealing.hpp"

namespace RL {

/*
 * Simplified PPO for AlphaZero-style Chinese Chess.
 *
 * Architecture:
 *   actorP  : state (90) -> Tanh(64) -> Softmax(128)     policy prior
 *   critic  : state (90) -> Tanh(64) -> Linear(1)        value V(s)
 *
 * Learning (no clipped obj, no KL-penalty, no actorQ):
 *   Policy loss: cross-entropy(actor(state) || MCTS_visit_target)
 *   Value loss:  MSE(critic(state), game_outcome * gamma^steps)
 *
 * Removed: eGreedyAction, noiseAction, gumbelMax, actorQ, alpha,
 *          beta, delta, epsilon, entropy0, annealing, learnWithKLpenalty,
 *          learnWithClipObjective
 */
class PPO
{
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim);
    virtual ~PPO(){}

    /* Forward - returns policy (Softmax) probabilities */
    Tensor &action(const Tensor &state);

    /* Forward - returns scalar value V(s) */
    float value(const Tensor &state);

    /* Train on one state: cross-entropy(policy, mctsTarget) + MSE(value, outcome)
     *   state       : 90-dim board encoding
     *   actionTarget: 128-dim MCTS visit distribution (normalized)
     *   valueTarget : +1.0 (win for current player), -1.0 (loss), 0.0 (draw) */
    void trainStep(const Tensor &state,
                   const Tensor &actionTarget,
                   float valueTarget,
                   float lr);

    /* Self-play training: compute discounted returns and train PPO on trajectory
     *   trajectory : list of (state, action_onehot, reward)
     *   finalOutcome : +1.0 (current player wins), -1.0 (loses), 0.0 (draw) */
    void learnSelfPlay(std::vector<Step>& trajectory,
                       float finalOutcome,
                       float learningRate);

    /* Save / Load weights */
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);

public:
    int stateDim;
    int actionDim;
    float gamma;
    float exploringRate;
    int learningSteps;

    Net actorP;      /* Policy network:  state -> 128-dim Softmax */
    Net critic;      /* Value network:   state -> 1-dim scalar    */
};

} // namespace RL
#endif // PPO_H
