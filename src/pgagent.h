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

    /*
     * Step → action index (deterministic hash)
     *
     * 加 const 是为了让 selfCheckReport() (它是 const 的) 能用**同一份**公式算动作
     * 别名 —— 否则报告里只能抄一份局部哈希, 那样"面板的读数"与"训练时真正用的下标"
     * 就有了两个会各自漂移的来源 (dqnmcts_agent.cpp 的 aliasActionIdxOf 就是被逼抄的
     * 那一份)。这个函数**不改任何成员**, 只是一次整数哈希, const 化不改变任何行为;
     * 调用方全部是非 const 对象, 所以这一改动逐字兼容。
     */
    int stepToActionIdx(const Step &s) const;

    /* Step → one-hot reward (for RL::Step in reinforce) */
    static inline float pieceTypeValue(int type) {
        if (type >= 0 && type < 7) return PIECE_VALUES[type];
        return 0.0f;
    }

    /* Compute immediate reward for a move */
    float computeReward(const Step &s, int color);

    /* [④] 学习口径的奖励: 界面奖励曲线取这一份 (见 AgentBase 的说明) */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }

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
    /*
     * 走子前先"探索环境 + 在线训练一次"再决策 (仿 snakeAI 的决策流程, 见 aiagent.h):
     * 从当前局面用本 agent 的探索策略滚若干步收集经验, 在线训练一次, 然后才走子。
     * 探索期间用 moveForward/moveBack 试走, 结束时会原样回退, 不影响真棋局。
     */
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;
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
    /*
     * 最近一次 reinforce 的**策略梯度损失** (界面"训练损失曲线"用)。
     * REINFORCE 没有"误差"量, 这里报 surrogate = -Σ A·log π(a|s) 的批均值:
     * 策略越把概率压到正优势的动作上它越小。advantage 已被标准化, 所以量级只反映
     * 信噪比, 不要跨 agent 比较 (曲线本来就是按 agent 分线的)。
     */
    float getLastTrainLoss() const override { return (float)dpg.lastLoss; }
    float getWinRate(int color = Stone::COLOR_BLACK) const {
        int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / totalEpisodes : 0.0f;
    }

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  报告的是**结构 / 口径**类事实, **不是棋力**:
     *   1. 表示健康度 (本 agent 的表示就是本工程的已知弱点 "表示闸门"):
     *      * 规则上下文通道 **0 个** —— 90 维状态只有"每格是什么子", 没有走子方、
     *        没有重复进度、没有无吃子进度、没有被将标记。于是"三次重复判和"
     *        "60 回合无吃子判和""被将/将杀"这些**决定终局与回报**的规则, 网络读不到。
     *      * 动作别名 —— `stepToActionIdx` 是 128 槽位的哈希, 而真实走法空间是
     *        8100 = 90×90。同一个局面里互不相同的合法着法会挤进同一个策略槽位,
     *        策略头因此**表达不出**它们的区别 (两个着法的梯度被平均), 这是**表示层
     *        的天花板**, 再训多久也不会消失。
     *      这一份读数用"标准开局"这一份**确定性**样本算 —— 与当前棋盘、训练进度
     *      都无关, 所以打开界面就能看到。
     *   2. 训练侧口径读数: 超参 / 网络规模 / 总对局数 / 自对弈胜率 / 最近一次
     *      reinforce 的 surrogate 损失。后两个都**不是棋力**: 自对弈里赢家与输家是
     *      同一份权重; surrogate 的量级只反映信噪比 (advantage 已被标准化), 跨 agent
     *      不可比较。
     *
     *  **只读**: 全部走**局部**棋盘与局部解码 (连 this->chess 都不读), 不改任何
     *  成员、不动棋盘 —— 它会在 GUI 线程上被调, 而搜索线程可能正在使用同一个
     *  `chess`。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;
};

#endif // PGAGENT_H
