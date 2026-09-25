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
 *   opponent : "对手在这个局面上会怎么走" (P1)。见 aiagent.h 的 OpponentPolicy:
 *              budget > 0 时, 探索里**对手那一半**最前面若干手由真实对手给出。
 *              **非 const 引用**: 返回时 used/unmatched 两个计数被回填, 调用方据此
 *              在探索说明里写清"真的用了几手对手着法"(那是"参数传进来了"与"真的生效"
 *              的区别 —— 本工程最怕的就是这种静默失效)。
 * 返回实际收集到的转移条数。
 *
 * ---- P1: 对手的棋怎么进训练数据 (为什么这么写) ----
 *
 * 自对弈里每个 ply 都由 `pick` (学习方自己的策略) 选动作, 于是"对手那一半"是学习方
 * 自己的猜测。有对手参数时, `turn != color` 的那些 ply 改问真实对手。
 *
 * ⚠ 问到的着法**照旧走 onTrans**?**不**。这里刻意**不记**:
 *   自对弈时"对手那一半"是学习方自己采样出来的, 记成学习方的动作转移是 on-policy 的;
 *   换成真对手之后同一份记录会变成"off-policy 却打着 on-policy 标签" —— 对 REINFORCE/
 *   PPO 这类要行为策略的方法是实打实的偏差 (冒充进来的动作从来没以那个概率被采样过)。
 *   所以对手着法只**推进局面**: 学习方自己的转移因此落在"真对手应手之后的局面"上,
 *   回报也带上对手的影响 —— 这就是"从对手身上学"的内容, 而且是干净的 on-policy 样本。
 *
 * 回退规则 (两条都要有, 否则一个"问不到"就变成整轮探索失真):
 *   * `stepFor` 返回无效着法 -> 用 `pick` (与改动前一致);
 *   * 对手给的着法**不在当前合法集里** (编码/规则不一致) -> 同样回退, 并计一次
 *     unmatched (诊断: 计数一直涨说明对手的着法表示与棋盘对不上, 那不是"对手弱")。
 */
template<typename Agent, typename PickAction, typename OnTransition>
int rolloutFromCurrent(Agent &agent, Chess &chess, int color, int steps,
                       PickAction pick, OnTransition onTrans,
                       OpponentPolicy &opponent)
{
    if (steps <= 0) {
        return 0;
    }
    const int STATE_DIM = Agent::STATE_DIM;
    const int ACTION_DIM = Agent::ACTION_DIM;
    int opponentBudget = opponent.valid() ? opponent.budget : 0;

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
        /*
           ---- P1: 对手那一半改由真实对手给着法 ----
           放在 pick **之后**: pick 是学习方的探索策略, 只在"轮到学习方"时才有意义;
           轮到对手时我们不用它 (但保留调用是刻意的 —— 有些 agent 的 pick 有副作用,
           而"轮到对手就不调 pick"会让行为随对手参数开关而变, 那不是我们要的变量)。
           命中条件: 轮到对手 + 还有预算 + 对手真的给了合法着法。
        */
        if (turn != color && opponentBudget > 0 && (bool)opponent.stepFor) {
            const Step ostep = opponent.stepFor(turn);
            if (ostep.valid) {
                Step *match = nullptr;
                int matchIdx = selectedAction;
                for (std::size_t j = 0; j < legal.size(); j++) {
                    const Step &cand = *legal[j];
                    if (cand.id == ostep.id && cand.nextPos.x == ostep.nextPos.x
                        && cand.nextPos.y == ostep.nextPos.y && cand.nextId == ostep.nextId) {
                        match = legal[j];
                        matchIdx = actionIndices[j];
                        break;
                    }
                }
                if (match != nullptr) {
                    chosen = match;
                    selectedAction = matchIdx;
                    opponentBudget--;
                    opponent.used++;
                    /*
                       对手这一手**不记**成学习方的转移 (理由见函数头注释):
                       直接落子、推进局面, 然后继续下一手。跳过的是 onTrans 与
                       collected —— 学习方的样本数因此不受对手手数影响。
                    */
                    double dummyOpp = 0.0;
                    chess.moveForward(chosen, dummyOpp);
                    path.push_back(*chosen);
                    Steps::instance().put(legal);
                    turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                      : Stone::COLOR_RED;
                    /*
                       对手这一手也可能**直接结束**这一局 (吃将/将死/困毙/判和)。
                       与上面那处一样统一走 Chess::getResult (它一次覆盖四种终局),
                       结束就停 —— 否则探索会从"已经结束的局面"继续往下走,
                       而后面那些样本的目标是假的 (局面本身不合法)。
                       注意: 这里**没有** onTrans 可标 done, 所以学习方最后一条样本
                       的 done 仍是 false —— 那是"截断"而不是"终局", 与"撞到步数上限"
                       同一种处理 (函数末尾本来就什么都不补)。
                    */
                    if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                        break;
                    }
                    state = RL::Tensor(STATE_DIM, 1);
                    agent.encodeState(state);
                    continue;
                }
                opponent.unmatched++;      /* 对不上: 回退到 pick (见函数头) */
            } else {
                opponent.unmatched++;
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

        /*
           终局判定统一走 Chess::getResult() (Phase 6): 它一次覆盖 将杀 / 困毙 /
           吃将 / 三次重复 / 60 回合无吃子判和。原来这里用的是 isGameOver() ——
           而它**只认"将不在了"**, 于是：
             * rollout 里"被将死"不终止, 会继续往下走 (走到无合法走法时才由调用方
               的另一条分支兜底);
             * 三次重复 / 判和 完全不是终局;
           两条合起来造成"截断处一律给 0" ⇒ 价值目标没有信号 (诊断实测 |target|>0.1
           的样本占比一度是 0.0%)。
           调用点在 moveForward 之后, 所以 getResult 的参数是 chess.sideToMove
           (刚被翻转成对手, 也就是"可能已被将死"的那一方)。
        */
        const int gameResult = chess.getResult(chess.sideToMove);
        const bool done = (gameResult != Chess::RESULT_ONGOING);
        float r = reward;
        if (done) {
            /*
               走子方视角的终局值 (胜 + / 负 - / 和 0): 优先用 agent 自己的口径
               (`terminalOutcomeOf`, 见文件末尾的说明), 没有这个成员的走共享的
               outcomeForMover —— 与 stone.h 的 REWARD_TERMINAL 同源。
            */
            r = terminalOutcomeOf(agent, gameResult, turn, 0);
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

/*
 * 终局值的**可选 agent 钩子** (2026-09)。
 *
 * 为什么要有它: 终局奖励本来统一走共享的 `outcomeForMover()` (胜 +1 / 负 -1 / 和 0),
 * 但"依赖局面"的终局塑形 (例如 SACAZAgent 的 rewardShape=2: 败方兵力越完整地被将死
 * 越值钱) 必须能读到棋盘。改 `outcomeForMover` 的签名会牵动全部 agent; 让每个 agent
 * 都加一份成员又是 5 份可能漂移的实现。所以: **有 `terminalReward(result, mover)` 这个
 * 成员的 agent 就用它, 没有的走共享口径** —— 用返回值类型重载分派 (int/long 惯用法),
 * SFINAE 失败就落到第二支, 于是 PG / DQN / PPO / DQNAB 一行都不用改。
 */
template<typename Agent>
inline auto terminalOutcomeOf(const Agent &a, int result, int mover, int)
    -> decltype(a.terminalReward(result, mover))
{
    return a.terminalReward(result, mover);
}

template<typename Agent>
inline float terminalOutcomeOf(const Agent &, int result, int mover, long)
{
    return outcomeForMover(result, mover);
}

#endif // AGENTROLLOUT_HPP
