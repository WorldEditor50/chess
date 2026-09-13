#ifndef AGENTROLLOUT_HPP
#define AGENTROLLOUT_HPP

#include <vector>
#include <functional>
#include "chess.h"
#include "rl/tensor.hpp"

/*
 * rolloutFromCurrent - "先探索环境"这一步的公共实现 (仿 snakeAI 的决策流程)。
 *
 * snakeAI 里每个 Agent::xxxAction() 开头都有一段结构相同的代码: 从**当前局面**
 * 出发, 用该 agent 自己的探索策略滚若干步, 边滚边收集经验, 撞到终止状态就停,
 * 然后用这批新鲜经验在线训练一次, 最后才基于当前局面做决策。
 * 各 agent 的差别只有两点: "怎么选探索动作" 和 "收集到一条转移之后怎么处理",
 * 所以把那两点做成回调, 其余骨架放在这里 —— 四个 agent 就不必各抄一遍。
 *
 * 象棋里没法像贪吃蛇那样在局部坐标上模拟 (走法合法性依赖整个棋盘), 所以探索是
 * 在**真棋盘**上用 moveForward 试走、结束时按相反顺序 moveBack 原样回退。
 *
 *   agent    : 提供 encodeState / getLegalActions / computeReward 的 agent
 *   color    : 轮到谁走
 *   steps    : 最多滚多少步
 *   pick     : (state, turn) -> 选中的动作索引; 返回 -1 表示"用第一个合法走法"
 *   onTrans  : (chosenStep, actionIdx, stateBefore, nextState, reward, done)
 * 返回实际收集到的转移条数。
 */
template<typename Agent, typename PickAction, typename OnTransition>
int rolloutFromCurrent(Agent &agent, Chess &chess, int color, int steps,
                       PickAction pick, OnTransition onTrans)
{
    if (steps <= 0) {
        return 0;
    }
    const int STATE_DIM = Agent::STATE_DIM;
    const int ACTION_DIM = Agent::ACTION_DIM;

    /*
       探索必须对棋盘**零副作用**, 包括 sideToMove。
       moveForward/moveBack 会让 sideToMove 来回翻转, 回退完最后一手之后它停在
       "第一个走子方的颜色", 也就是 color —— 如果调用方传进来的 color 与棋盘当前
       的 sideToMove 不一致 (测试里就是这么压的), 那就会改掉棋盘状态。
       所以进来先存原值, 结束时无条件恢复。
    */
    const int savedSideToMove = chess.sideToMove;
    chess.sideToMove = color;
    RL::Tensor state(STATE_DIM, 1);
    agent.encodeState(state);

    int turn = color;
    int collected = 0;
    std::vector<Step> path;
    path.reserve((std::size_t)steps);

    for (int i = 0; i < steps; i++) {
        std::vector<Step*> legal;
        std::vector<int> actionIndices;
        RL::Tensor mask(ACTION_DIM, 1);
        mask.zero();
        agent.getLegalActions(turn, legal, actionIndices, mask);
        if (legal.empty()) {
            Steps::instance().put(legal);
            break;
        }

        int selectedAction = pick(state, turn);
        Step *chosen = nullptr;
        for (std::size_t j = 0; j < actionIndices.size(); j++) {
            if (actionIndices[j] == selectedAction) {
                chosen = legal[j];
                break;
            }
        }
        if (chosen == nullptr) {
            chosen = legal[0];
            selectedAction = actionIndices[0];
        }

        /* 即时奖励必须在落子之前算: 落子会把被吃子置为 alive=false */
        const float reward = agent.computeReward(*chosen, turn);
        Step recorded = *chosen;

        double dummy = 0.0;
        chess.moveForward(chosen, dummy);
        path.push_back(recorded);
        Steps::instance().put(legal);

        RL::Tensor nextState(STATE_DIM, 1);
        agent.encodeState(nextState);

        const int gameResult = chess.isGameOver();
        const bool done = (gameResult != Stone::COLOR_NONE);
        float r = reward;
        if (done) {
            r = (gameResult == turn) ? 1.0f : -1.0f;
        }
        onTrans(recorded, selectedAction, state, nextState, r, done);
        collected++;

        if (done) {
            break;
        }
        state = nextState;
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }

    /* 原样回退: 探索绝不能改动真实对局 */
    for (std::size_t i = path.size(); i > 0; i--) {
        double dummy = 0.0;
        chess.moveBack(&path[i - 1], dummy);
    }
    chess.sideToMove = savedSideToMove;
    return collected;
}

#endif // AGENTROLLOUT_HPP
