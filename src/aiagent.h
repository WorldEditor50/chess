#ifndef AIAGENT_H
#define AIAGENT_H

#include <string>
#include <fstream>
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

    /* --- 决策流程: 先探索环境 + 预训练, 再决策 (仿 snakeAI) --- */

    /*
     * snakeAI 的 Agent::xxxAction() 每次走子前都是同一套三步:
     *
     *   1. observe(state)              —— 把**当前局面**编码下来留一份 state0
     *   2. if (trainFlag) {            —— 从当前局面出发, 用**自己的探索策略**滚若干步
     *          for (i < N) {             (noiseAction / gumbelMax / 采样 ...)
     *              选动作 -> 模拟走子 -> 算即时奖励 -> perceive / 收集 Step
     *              撞墙或到达目标就 break
     *          }
     *          learn() / reinforce()  —— 用这批**新鲜经验**在线训练一次
     *      }
     *   3. action(state0).argmax()      —— 再基于**当前局面**做决策
     *
     * 也就是"先探索环境、预先训练, 再进行决策"。把它抽成 AgentBase 的接口, 各自
     * 实现的差别只在"用什么策略滚、滚完怎么训练、训练用哪个模型"。
     *
     * 有监督式的 agent (Alpha-Beta / MCTS) 没有需要在线更新的参数,
     * 默认实现什么都不做并返回 false —— 它们的"探索"就是搜索本身。
     *
     * 已实现该接口的: PGEagent / DQNAgent / PPOMCTSAgent / DQNMCTSAgent / EVABAgent。
     *
     * 实现的**硬性契约**: 返回时棋盘必须逐字节复原 (含 sideToMove), 因为调用方
     * (ChessBoard::preTrainThenDecide) 紧接着就要基于**当前**局面做决策。
     * 共用实现见 src/agentrollout.hpp 的 rolloutFromCurrent()。
     *
     *   color        : 轮到谁走
     *   rolloutSteps : 最多滚多少步
     * 返回 true 表示这次确实做了在线训练。
     */
    virtual bool exploreAndTrain(int color, int rolloutSteps)
    {
        (void)color;
        (void)rolloutSteps;
        return false;
    }

    /* 最近一次 exploreAndTrain 的说明, 供界面显示 */
    virtual std::string getExploreInfo() const { return m_exploreInfo; }

protected:
    std::string m_exploreInfo;
};

/*
 * 权重文件读写状态的校验。
 *
 * RL 层的 save()/load() 不返回状态 (Net::save() 返回的 -1 被丢掉了), 于是各 agent
 * 的 saveModel()/loadModel() 一律 `return true` —— 写盘失败时 GUI 照样弹"保存成功"。
 * 这里用"文件存在且非空"给出真实结果, 不改变 RL 层的接口。
 */
inline bool weightFileWritten(const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return false;
    }
    return f.tellg() > 0;
}

inline bool weightFileReadable(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

#endif // AIAGENT_H
