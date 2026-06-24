#ifndef PGAGENT_H
#define PGAGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "chess.h"
#include "aiagent.h"
#include "rl/dpg.h"

/*
 * PGEagent - Policy Gradient Chess Agent (REINFORCE with baseline)
 *
 * Based on RL::DPG from src/rl/dpg.cpp
 *
 * State Encoding:   90-dim board (10×9)
 *                   Position i*9+j encoded as:
 *                   0 = empty
 *                   +1..+7 = Black piece by type
 *                   -1..-7 = Red piece by type
 *
 * Action Encoding:  128-dim one-hot (covers all legal moves in Xingqi)
 *                   Mapping: actionIdx = (fromID * 90 + toIdx) % ACTION_DIM
 *                   Masked for illegal moves during selection
 *
 * Training:         REINFORCE with discount factor γ and baseline
 *                   Uses RL::DPG::reinforce1 (standard REINFORCE)
 */
class PGEagent : public AgentBase
{
public:
    /* Board state dimension (10 × 9) */
    static constexpr int STATE_DIM = 90;

    /* Maximum action dimension */
    static constexpr int ACTION_DIM = 128;

    /* Piece type → encoding value */
    static constexpr float PIECE_VALUES[7] = {
        1.0f,   /* TYPE_CHE   */
        2.0f,   /* TYPE_MA    */
        3.0f,   /* TYPE_PAO   */
        4.0f,   /* TYPE_BING  */
        5.0f,   /* TYPE_JIANG */
        6.0f,   /* TYPE_SHI   */
        7.0f    /* TYPE_XIANG */
    };

public:
    Chess &chess;
    RL::DPG dpg;                    /* Policy network */
    float gamma;                    /* Discount factor */
    float initialExploringRate;
    float learningRate;

    /* Training statistics */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red wins, [1]=black wins */

    /* Online training trajectory */
    std::vector<RL::Step> m_onlineTrajectory;

    /* Encode current board state into state tensor (90-dim) */
    void encodeState(RL::Tensor &state);

    /* Build list of legal moves, return their action indices + mask */
    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);

    /* Step → action index (deterministic hash) */
    int stepToActionIdx(const Step &s);

    /* Step → one-hot reward (for RL::Step in reinforce) */
    static inline float pieceTypeValue(int type) {
        if (type >= 0 && type < 7) return PIECE_VALUES[type];
        return 0.0f;
    }

    /* Compute immediate reward for a move */
    float computeReward(const Step &s, int color);

    /* Online training (human-vs-AI) */
    void beginOnline();
    void recordOnline(const Step& s, int color, const RL::Tensor& stateBefore);
    void endOnline(int winner, int myColor);

    PGEagent(Chess &chess_,
             int hiddenDim = 64,
             float gamma_ = 0.9f,
             float lr = 0.01f,
             float eps = 1.0f);

    ~PGEagent() = default;

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /* Select a move:
     *   training=true  : sample from policy softmax distribution
     *   training=false : argmax */
    Step selectMove(int color, bool training = true);

    /* Full training loop via self-play or vs random opponent */
    void train(int episodes, int maxMoves = 200,
               bool selfPlay = true, bool verbose = true);

    /* Warmup: self-play several episodes from current board state (in place) */
    void warmupFromCurrent(int episodes = 5, int maxMoves = 200);

    /* Save / Load weights */
    bool savePolicy(const std::string &filepath);
    bool loadPolicy(const std::string &filepath);

    /* Statistics */
    int getTotalEpisodes() const { return totalEpisodes; }
    float getWinRate(int color = Stone::COLOR_BLACK) const {
        int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / totalEpisodes : 0.0f;
    }
};

#endif // PGAGENT_H
