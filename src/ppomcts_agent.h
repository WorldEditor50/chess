#ifndef PPOMCTS_AGENT_H
#define PPOMCTS_AGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include "chess.h"
#include "aiagent.h"
#include "rl/ppo.h"

/*
 * PPOMCTSAgent - AlphaZero-style PPO+MCTS Chess Agent
 *
 * Combines Proximal Policy Optimization (PPO) neural network with
 * Monte Carlo Tree Search (MCTS), following the AlphaZero paradigm:
 *
 *   - PPO's actor network provides policy priors P(s,a) for MCTS
 *   - PPO's critic network provides state value estimates V(s)
 *   - MCTS uses PUCT formula for selection:
 *       U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *   - Instead of random rollouts, evaluation uses the PPO value head
 *   - After MCTS, the improved policy target is proportional to visit counts
 *
 * State Encoding:  90-dim board (10x9) -- same as PGEagent/DQNAgent
 * Action Encoding: 128-dim one-hot -- same hash as PGEagent
 */

class PPOMCTSAgent : public AgentBase
{
public:
    static constexpr int STATE_DIM = 90;
    static constexpr int ACTION_DIM = 128;
    static constexpr float PIECE_VALUES[7] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
    };

public:
    Chess &chess;
    RL::PPO ppo;                    /* PPO: actorP (policy), critic (value) */
    float gamma;
    float learningRate;
    float c_puct;                   /* PUCT exploration constant */

    /* Training statistics */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red, [1]=black */

    /* Online training trajectory */
    std::vector<RL::Step> m_onlineTrajectory;

    /* ----------------------------------------------------------------
     *  AlphaZero-style MCTS Node
     * ---------------------------------------------------------------- */
    struct AZNode {
        int parentID;                /* index of parent (-1 for root) */
        int parentAction;            /* action index that led to this node */
        Step step;                   /* the move that led to this node */

        /* Tree statistics */
        int visitCount;
        double totalValue;           /* sum of value estimates */
        double prior;                /* PPO policy prior P(s,a) */

        /* Children */
        std::vector<int> childIDs;
        std::vector<int> untriedActionIndices; /* actions not yet expanded */
        std::vector<Step> untriedSteps;

        /* Current color at this node */
        int currentColor;

        AZNode()
            : parentID(-1), parentAction(-1),
              visitCount(0), totalValue(0.0), prior(0.0),
              currentColor(Stone::COLOR_NONE) {}
        AZNode(int pid, int pa, const Step &st, float p, int color)
            : parentID(pid), parentAction(pa), step(st),
              visitCount(0), totalValue(0.0), prior(p),
              currentColor(color) {}

        double getQ() const {
            return visitCount > 0 ? totalValue / (double)visitCount : 0.0;
        }
    };

    std::vector<AZNode> nodes;

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
     *  PPO + MCTS core
     * ---------------------------------------------------------------- */
    /* Evaluate a state: returns (policy_prior, value_estimate) */
    void evaluateState(const RL::Tensor &state,
                       RL::Tensor &policyOut, float &valueOut);

    /* PUCT score for a child node */
    double getPUCT(int childID, int parentVisits) const;

    /* Online training (human-vs-AI) */
    void beginOnline();
    void recordOnline(const Step& s, int color, const RL::Tensor& stateBefore);
    void endOnline(int winner, int myColor);

public:
    PPOMCTSAgent(Chess &chess_,
                 int hiddenDim = 64,
                 float gamma_ = 0.99f,
                 float lr = 0.001f,
                 float cpuct = 1.414f);

    ~PPOMCTSAgent() = default;

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /* ----------------------------------------------------------------
     *  Public API
     * ---------------------------------------------------------------- */

    /* Select a move using PPO+MCTS search (no exploration during search)
     *  color        : side to move
     *  simulations  : number of MCTS iterations
     *  temp         : temperature for final move selection (0 = argmax) */
    Step selectMove(int color, int simulations,
                    float temp = 0.0f);

    /* Train via self-play using PPO+MCTS:
     *   - At each position, run MCTS guided by PPO
     *   - Sample the move from the MCTS visit distribution
     *   - Store trajectories and train PPO at the end */
    void trainSelfPlay(int episodes, int simulations,
                       int maxMoves = 200, bool verbose = true,
                       float tempRoot = 1.0f, float tempFinal = 0.1f);

    /* Warmup: self-play several episodes from current board state (in place) */
    void warmupFromCurrent(int episodes = 5, int simulations = 200,
                           int maxMoves = 200);

    /* Save / Load PPO weights */
    bool saveModel(const std::string &actorPath,
                   const std::string &criticPath);
    bool loadModel(const std::string &actorPath,
                   const std::string &criticPath);

    /* Convenience: save both actor & critic from a single file prefix.
       (e.g., saveModel("model") saves to "model_actor" and "model_critic") */
    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* Statistics */
    int getTotalEpisodes() const { return totalEpisodes; }
    float getWinRate(int color = Stone::COLOR_BLACK) const {
        int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / totalEpisodes : 0.0f;
    }
};

#endif // PPOMCTS_AGENT_H
