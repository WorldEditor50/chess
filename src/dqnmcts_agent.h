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

    /* [④] 学习口径的奖励: 界面奖励曲线取这一份 (见 AgentBase 的说明) */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }

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
    /*
     * 走子前先"探索环境 + 在线训练一次"再决策 (仿 snakeAI 的决策流程, 见 aiagent.h):
     * 从当前局面用本 agent 的探索策略滚若干步收集经验, 在线训练一次, 然后才走子。
     * 探索期间用 moveForward/moveBack 试走, 结束时会原样回退, 不影响真棋局。
     */
    bool exploreAndTrain(int color, int rolloutSteps) override;
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
    /* 最近一次 learn 的平均平方 TD 误差 (界面"训练损失曲线"用, 见 rl/dqn.h) */
    float getLastTrainLoss() const override { return (float)dqn.lastLoss; }

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  报告两类事实:
     *   1. **表示健康度**: 状态里有哪些规则上下文通道 (本 agent 是 0 个),
     *      以及**动作别名** —— 同一个局面里有多少互不相同的合法着法被迫共用
     *      一个 Q 槽位 (`stepToActionIdx` 是 128 槽位的哈希, 真实走法空间 8100)。
     *      前一项用"标准开局"这一份确定性样本算一次 (与棋盘状态无关); 后一项用
     *      搜索过程累计的增量统计 (所以对局中会变)。
     *   2. **终局通道计数**: 每局结束时记下 `getResult()` 判出的结束方式,
     *      以及 `isGameOver()` 能不能看见它。两者的差就是"终局信号漏掉了多少局"
     *      —— 漏掉的那些局的 Q 目标只有 r(材质) + gamma*maxQ, 没有胜负。
     *
     *  **只读**: 全部走 `chess` 的副本与局部解码, 不改任何成员、不动棋盘。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;

    /*
     * 一局的结束方式 (训练循环收尾时累加)。口径与 chess.h 的 Chess::Result 一致,
     * 另加 END_CAP = 撞手数上限、根本没走到终局 (台架截断)。
     */
    enum EndCode {
        END_CAP = 0,        /* 跑到 maxMoves 上限, 未终局 */
        END_RED_WIN = 1,    /* Chess::RESULT_RED_WIN */
        END_BLACK_WIN = 2,  /* Chess::RESULT_BLACK_WIN */
        END_DRAW = 3        /* Chess::RESULT_DRAW (三次重复 / 60 回合自然限着) */
    };
    long long endCount[4] = {0, 0, 0, 0};
    /* 其中被 `isGameOver()` 看见的次数 (它只认将/帅是否还在场) */
    long long endSeenByGameOver = 0;

    /*
     * 动作别名的增量统计 (每次自对弈取着法时累加)。
     * 为什么要增量而不是只看标准开局: 开局 44 个合法着法只落进 39 个槽位, 而中局
     * 的合法集会更大、碰撞也不同 —— 只有把对局里真实遇到的那些局面统计进来,
     * 面板上的数才代表"训练时实际发生了什么"。
     */
    long long aliasMoves = 0;         /* 累计取过的合法着法数 */
    long long aliasIndexed = 0;       /* 其中拿到独立 Q 槽位的着法数 (碰撞后剩余) */
    long long aliasWorstSlot = 0;     /* 单局面内最挤槽位背了几个不同走法 */
    long long aliasClearedMoves = 0;  /* 累计被挤掉的走法数 (合法数 - 用到的槽位数) */

private:
    /*
     * 记一局的结束方式 (三个训练收尾点共用)。
     *   result         : `chess.getResult(chess.sideToMove)` 的返回值;
     *                    RESULT_ONGOING 表示"撞手数上限", 归到 END_CAP。
     *   seenByGameOver : 旧口径 `isGameOver()` 是否看见了这一局的终局。
     */
    void noteEnd(int result, bool seenByGameOver);
};

#endif // DQNMCTS_AGENT_H
