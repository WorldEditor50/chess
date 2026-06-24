#ifndef DQNAGENT_H
#define DQNAGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "chess.h"
#include "aiagent.h"
#include "rl/dqn.h"

/*
 * DQNAgent - Deep Q-Network Chess Agent
 *
 * Based on RL::DQN from src/rl/dqn.cpp
 *
 * State Encoding:   90-dim board (10×9)
 *                   Position i*9+j encoded as:
 *                   0 = empty
 *                   +1..+7 = Black piece by type
 *                   -1..-7 = Red piece by type
 *
 * Action Encoding:  128-dim one-hot (covers all legal moves in Xingqi)
 *                   Hashing: actionIdx = (fromID * prime1 + toPos.x * prime2
 *                                          + toPos.y * prime3) % ACTION_DIM
 *
 * Training:         DQN with experience replay, target network,
 *                   ε-greedy exploration
 */
class DQNAgent : public AgentBase
{
public:
    /* Board state dimension (10 × 9) */
    static constexpr int STATE_DIM = 90;

    /* Maximum action dimension */
    static constexpr int ACTION_DIM = 128;

public:
    Chess &chess;
    RL::DQN dqn;
    float gamma;
    float initialExploringRate;
    float learningRate;

    /* Experience replay parameters */
    int maxMemorySize;
    int batchSize;
    int replaceTargetInterval;
    int learnCounter;

    /* Training statistics */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red wins, [1]=black wins */

    /* Encode current board state into state tensor (90-dim) */
    void encodeState(RL::Tensor &state);

    /* Gather legal moves, return action indices + mask */
    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);

    /* Step → action index (deterministic hash) */
    int stepToActionIdx(const Step &s);

    /* Step → action one-hot tensor */
    void stepToOneHot(const Step &s, RL::Tensor &onehot);

    /* Material reward from a move (from color's perspective) */
    float computeReward(const Step &s, int color);

    /* Online training: call after each AI move */
    void trainAfterMove(const RL::Tensor& stateBefore,
                        const Step& chosenStep,
                        int color,
                        const RL::Tensor& nextState,
                        bool done);

    DQNAgent(Chess &chess_,

             int hiddenDim = 64,
             float gamma_ = 0.99f,
             float lr = 0.001f,
             float eps = 1.0f);

    ~DQNAgent() = default;

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /* Select a move:
     *   training=true  : ε-greedy (sample random with prob ε)
     *   training=false : argmax over Q-values */
    Step selectMove(int color, bool training = true);

    /* Train via self-play against a random opponent */
    void trainVsRandom(int episodes, int maxMoves = 200,
                       bool verbose = true);

    /* Train via self-play (both sides use the same DQN) */
    void trainSelfPlay(int episodes, int maxMoves = 200,
                       bool verbose = true);

    /* Warmup: self-play several episodes from current board state (in place) */
    void warmupFromCurrent(int episodes = 5, int maxMoves = 200);

    /* Save / Load weights */
    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* Statistics */
    int getTotalEpisodes() const { return totalEpisodes; }
    float getWinRate(int color = Stone::COLOR_BLACK) const {
        int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / totalEpisodes : 0.0f;
    }
    float getExploreRate() const { return dqn.exploringRate; }
};

#endif // DQNAGENT_H
