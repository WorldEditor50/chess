#ifndef DQNMCTS_AGENT_H
#define DQNMCTS_AGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include "chess.h"
#include "aiagent.h"
#include "rl/dqn.h"

/*
 * DQNMCTSAgent - DQN + MCTS Chess Agent
 *
 * Combines Deep Q-Network (DQN) with Monte Carlo Tree Search (MCTS).
 * Unlike PPOMCTSAgent (which uses PPO's policy + value heads), this agent
 * uses only DQN's Q-value estimates to guide MCTS:
 *
 *   - DQN provides Q(s,a) for all actions in a state
 *   - MCTS uses UCB1 for selection (no prior needed — unlike PUCT)
 *   - Leaf evaluation uses max_a Q(s,a) from DQN instead of random rollouts
 *   - Training uses standard DQN experience replay
 *
 * State Encoding:  90-dim board (10x9) — same as DQNAgent
 * Action Encoding: 128-dim one-hot — same hash as other agents
 */

class DQNMCTSAgent : public AgentBase
{
public:
    static constexpr int STATE_DIM = 90;
    static constexpr int ACTION_DIM = 128;
    static constexpr float PIECE_VALUES[7] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
    };

public:
    Chess &chess;
    RL::DQN dqn;                    /* DQN: Q(s,a) value network */
    float gamma;
    float learningRate;
    float C;                        /* UCB1 exploration constant */

    /* Experience replay parameters */
    int maxMemorySize;
    int batchSize;
    int replaceTargetInterval;
    int learnCounter;

    /* Training statistics */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red, [1]=black */

    /* Online training mode (used by UI self-play) */
    bool m_trainingMode;            /* set true to enable experience collection during selectMove */
    RL::Tensor m_cachedState;       /* state captured before calling selectMove */
    int m_onlineStepCount;          /* steps taken in current online episode */

    /* ----------------------------------------------------------------
     *  MCTS Node
     * ---------------------------------------------------------------- */
    struct DQNMCTSNode {
        int parentID;                /* index of parent (-1 for root) */
        Step step;                   /* the move that led to this node */

        /* Tree statistics (same as pure MCTS) */
        int visitCount;
        double totalReward;          /* sum of leaf evaluations */

        /* Children */
        std::vector<int> childIDs;
        std::vector<Step> untriedSteps; /* moves not yet expanded */

        /* Current color at this node */
        int currentColor;

        DQNMCTSNode()
            : parentID(-1),
              visitCount(0), totalReward(0.0),
              currentColor(Stone::COLOR_NONE) {}
        DQNMCTSNode(int pid, const Step &st, int color)
            : parentID(pid), step(st),
              visitCount(0), totalReward(0.0),
              currentColor(color) {}
    };

    std::vector<DQNMCTSNode> nodes;

    /* ----------------------------------------------------------------
     *  Encoding / Action Helpers
     * ---------------------------------------------------------------- */
    void encodeState(RL::Tensor &state);
    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);
    int stepToActionIdx(const Step &s);
    float computeReward(const Step &s, int color);

    /* ----------------------------------------------------------------
     *  DQN + MCTS core
     * ---------------------------------------------------------------- */
    /* Evaluate a state using DQN: returns max_a Q(s,a) (value) and all Q-values */
    double evaluateLeaf(int color, RL::Tensor &qValues);

    /* UCB1 score for a child node */
    double getUCB1(int childID, int parentVisits) const;

public:
    DQNMCTSAgent(Chess &chess_,
                 int hiddenDim = 64,
                 float gamma_ = 0.99f,
                 float lr = 0.001f,
                 float eps = 1.0f,
                 float uc = 1.414f);

    ~DQNMCTSAgent() = default;

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /* ----------------------------------------------------------------
     *  Public API
     * ---------------------------------------------------------------- */

    /* Select a move using DQN+MCTS search
     *  color        : side to move
     *  iterations   : number of MCTS iterations
     *  training     : if true, use ε-greedy exploration on top of MCTS;
     *                 also caches the pre-move state for online training */
    Step selectMove(int color, int iterations = 400,
                    bool training = false);

    /* Online training helpers — used when m_trainingMode is true.
     * The caller should:
     *   1. call selectMove(color, iter, true) — this caches the state
     *   2. play the returned move on the board (chess.moveForward)
     *   3. call recordExperience(chosenStep, color) — records transition
     *   4. if game over, call endOnlineEpisode(gameResult) */
    void recordExperience(const Step &chosenStep, int color);
    void endOnlineEpisode(int gameResult);

    /* Warmup: self-play several episodes from current board state (in place) */
    void warmupFromCurrent(int episodes = 5, int iterations = 200,
                           int maxMoves = 200);

    /* Train vs random opponent (DQN+MCTS black vs random red) */
    void trainVsRandom(int episodes, int iterations = 200,
                       int maxMoves = 200, bool verbose = true);

    /* Train via self-play (both sides use the same DQN+MCTS) */
    void trainSelfPlay(int episodes, int iterations = 200,
                       int maxMoves = 200, bool verbose = true);

    /* Save / Load DQN weights */
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

#endif // DQNMCTS_AGENT_H
