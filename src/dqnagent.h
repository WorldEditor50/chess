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

    /*
     * 最近一次 learn 的**平均平方 TD 误差** (界面"训练损失曲线"用)。
     * 直接转发 RL::DQN::lastLoss —— 一个是窗口内平均, 一个是最近一批, 这里取后者。
     */
    float getLastTrainLoss() const override { return (float)dqn.lastLoss; }

    /* Encode current board state into state tensor (90-dim) */
    void encodeState(RL::Tensor &state);

    /* Gather legal moves, return action indices + mask */
    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);

    /*
     * Step → action index (deterministic hash)
     *
     * 加 const 是为了让 selfCheckReport() (它是 const 的) 能用**同一份**公式算动作
     * 别名 —— 否则报告里只能抄一份局部哈希, 于是"面板读数"与"训练时真正用的下标"
     * 就有了两个会各自漂移的来源。这个函数**不改任何成员**, 只是一次整数哈希,
     * const 化不改变任何行为; 调用方全部是非 const 对象, 所以逐字兼容。
     */
    int stepToActionIdx(const Step &s) const;

    /* Step → action one-hot tensor */
    void stepToOneHot(const Step &s, RL::Tensor &onehot);

    /* Material reward from a move (from color's perspective) */
    float computeReward(const Step &s, int color);

    /* [④] 学习口径的奖励: 界面奖励曲线取这一份 (见 AgentBase 的说明) */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }

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

    /*
     * 走子前先"探索环境 + 在线训练一次"再决策 (仿 snakeAI 的 dqnAction)。
     * 用 dqn.noiseAction() 从当前局面滚 rolloutSteps 步, 每条转移 perceive 进
     * 回放池, 然后 learn() 一次。
     */
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;

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

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  报告的是**结构 / 口径**类事实, **不是棋力**:
     *   1. 表示健康度 (本 agent 的表示就是本工程的已知弱点 "表示闸门"):
     *      * 规则上下文通道 **0 个** —— 90 维状态只有"每格是什么子", 没有走子方、
     *        没有重复进度、没有无吃子进度、没有被将标记。于是"三次重复判和"
     *        "60 回合无吃子判和""被将/将杀"这些**决定终局与回报**的规则, 网络读不到。
     *      * 动作别名 —— `stepToActionIdx` 是 128 槽位的哈希, 而真实走法空间是
     *        8100 = 90×90。同一个局面里互不相同的合法着法会挤进同一个 Q 槽位,
     *        Q 头因此**表达不出**它们的区别 (两个着法的梯度被平均), 这是**表示层的
     *        天花板**, 再训多久也不会消失。
     *      这一份读数用"标准开局"这一份**确定性**样本算 —— 与当前棋盘、训练进度
     *      都无关, 所以打开界面就能看到。
     *   2. 训练侧口径读数: 超参 / 回放与学习节奏 / 总对局数 / 最近一次 learn 的平均
     *      平方 TD 误差 (量纲是原始 Q 尺度, 跨 agent 不可比, 也不说明棋力)。
     *
     *  **只读**: 全部走**局部**棋盘与局部解码 (连 this->chess 都不读), 不改任何
     *  成员、不动棋盘 —— 它会在 GUI 线程上被调, 而搜索线程可能正在使用同一个
     *  `chess`。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;
};

#endif // DQNAGENT_H
