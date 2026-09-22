#include "pgagent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "agentrollout.hpp"
#include "rl/util.hpp"

#include <cstdio>   /* selfCheckReport 的 snprintf */
#include <map>      /* 动作别名的槽位分组 */
#include <set>      /* 动作别名的"互不相同的走法"去重 */

/* ------------------------------------------------------------------ */
/*  Constructor                                                        */
/* ------------------------------------------------------------------ */
PGEagent::PGEagent(Chess &chess_,
                   int hiddenDim,
                   float gamma_,
                   float lr,
                   float eps)
    : AgentBase(),
      chess(chess_),
      dpg(STATE_DIM, hiddenDim, ACTION_DIM),
      gamma(gamma_),
      initialExploringRate(eps),
      learningRate(lr),
      totalEpisodes(0)
{
    totalWins[0] = 0; /* red */
    totalWins[1] = 0; /* black */
    dpg.exploringRate = eps;
    dpg.learningRate = lr;
}

/* AgentBase interface */
Step PGEagent::getBestMove(int color)
{
    return selectMove(color, false);
}

std::string PGEagent::getName() const
{
    return "Policy Gradient (REINFORCE)";
}

/* ------------------------------------------------------------------ */
/*  encodeState:  10×9 board → 90-dim tensor                           */
/* ------------------------------------------------------------------ */
void PGEagent::encodeState(RL::Tensor &state)
{
    state.zero();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || s->alive == false) continue;

        int idx = s->pos.x * 9 + s->pos.y;  /* row-major 0..89 */
        float val = pieceTypeValue(s->type);
        if (s->color == Stone::COLOR_RED) {
            val = -val;     /* Red = negative */
        }
        state[idx] = val / 7.0f;  /* normalize to [-1, +1] */
    }
}

/* ------------------------------------------------------------------ */
/*  getLegalActions:  gather legal moves, map to action indices, mask  */
/* ------------------------------------------------------------------ */
void PGEagent::getLegalActions(int color,
                               std::vector<Step*> &steps,
                               std::vector<int> &actionIndices,
                               RL::Tensor &actionMask)
{
    actionMask.zero();          /* all illegal initially */
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());

    for (Step *s : steps) {
        int aidx = stepToActionIdx(*s);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;   /* mark legal */
    }
}

/* ------------------------------------------------------------------ */
/*  stepToActionIdx:  deterministic hash from Step → [0, ACTION_DIM)   */
/*  Uses prime coefficients for good distribution.                     */
/*  const: 无状态, 只为让 selfCheckReport() const 复用同一份公式       */
/*  (见 pgagent.h 的说明)。                                            */
/* ------------------------------------------------------------------ */
int PGEagent::stepToActionIdx(const Step &s) const
{
    /* from_id * prime1 + toPos.x * prime2 + toPos.y * prime3 */
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)ACTION_DIM);
}

/* ------------------------------------------------------------------ */
/*  computeReward:  一步的即时奖励 (走子方视角, 含每步代价)             */
/*                                                                     */
/*  Phase 1 起统一走 stone.h 的 stepReward(): 材质系数 0.1、吃將不给材质 */
/*  奖励 (终局常量负责)、每步代价 -0.005。实测原来"一方全材质奖励 =    */
/*  35.0 vs 终局 1.0", 最优策略是吃子而不是赢棋。见 stone.h REWARD_*。  */
/* ------------------------------------------------------------------ */
float PGEagent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) return stepReward(false, false, 0.0);

    Stone *victim = chess.stones[s.nextId];
    /*
       这里**不**检查 victim->alive。
       调用方普遍在 chess.moveForward() 之后才求即时奖励 (pgagent.cpp 的
       trainSelfPlay/warmupFromCurrent、ppomcts_agent、dqnmcts_agent 都是如此),
       而 moveTo() 会把被吃子置 alive=false —— 加上 alive 判断会让吃子奖励恒为
       0.0f, 于是只能靠终局 ±1 学习。奖励只应由"这一步吃了谁"决定, 与调用时机无关。
    */
    if (victim == nullptr) return stepReward(false, false, 0.0);

    /*
       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来写的是
       `(color == COLOR_BLACK) ? reward : -reward` (黑方视角), 于是红方白吃一个
       黑车会拿到负奖励, 与同一批经验里的终局奖励 (走子方视角的 ±1) 正好相反。
       回归钉在 test_match 的 [2.6] 节。
    */
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* ------------------------------------------------------------------ */
/*  selectMove:  pick a move using the policy network                  */
/*    training=true  → softmax sampling with masking                   */
/*    training=false → argmax with masking                             */
/* ------------------------------------------------------------------ */
Step PGEagent::selectMove(int color, bool training)
{
    /* 1. Encode state */
    RL::Tensor state(STATE_DIM, 1);
    encodeState(state);

    /* 2. Get legal moves */
    std::vector<Step*> steps;
    std::vector<int> actionIndices;
    RL::Tensor actionMask(ACTION_DIM, 1);
    actionMask.zero();
    getLegalActions(color, steps, actionIndices, actionMask);

    if (steps.empty()) {
        return Step();   /* no legal moves */
    }

    /* 3. Forward pass through policy network to get raw action logits */
    RL::Tensor &policyOut = dpg.action(state);

    /* 4. Apply action mask: set illegal actions to 0, renormalize */
    for (int i = 0; i < ACTION_DIM; i++) {
        if (actionMask[i] < 0.5f) {
            policyOut[i] = 0.0f;
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < ACTION_DIM; i++) {
        sum += policyOut[i];
    }
    if (sum > 1e-9f) {
        for (int i = 0; i < ACTION_DIM; i++) {
            policyOut[i] /= sum;
        }
    } else {
        /* Fallback: uniform over legal actions */
        for (int i = 0; i < ACTION_DIM; i++) {
            policyOut[i] = actionMask[i] / (float)steps.size();
        }
    }

    /* 5. Select action */
    int selectedAction;
    if (training) {
        /* ε-greedy */
        float r = (float)std::rand() / (float)RAND_MAX;
        if (r < dpg.exploringRate) {
            /* Random legal move */
            int idx = std::rand() % (int)steps.size();
            selectedAction = actionIndices[idx];
        } else {
            /* Sample from masked distribution */
            selectedAction = RL::Random::categorical(policyOut);
        }
    } else {
        /* Argmax over legal actions */
        selectedAction = -1;
        float maxVal = -1e9f;
        for (int i = 0; i < ACTION_DIM; i++) {
            if (actionMask[i] > 0.5f && policyOut[i] > maxVal) {
                maxVal = policyOut[i];
                selectedAction = i;
            }
        }
    }

    /* 6. Map selected action back to a Step */
    for (std::size_t i = 0; i < actionIndices.size(); i++) {
        if (actionIndices[i] == selectedAction) {
            Step result = *steps[i];
            Steps::instance().put(steps);
            return result;
        }
    }

    /* Fallback: first legal move */
    Step result = *steps[0];
    Steps::instance().put(steps);
    return result;
}

/* ------------------------------------------------------------------ */
/*  train:  self-play or vs-random training loop                       */
/*  Uses RL::DPG::reinforce1 which implements REINFORCE with baseline  */
/* ------------------------------------------------------------------ */
void PGEagent::train(int episodes, int maxMoves,
                     bool selfPlay, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        std::vector<RL::Step> trajectory;
        int currentColor = Stone::COLOR_BLACK;  /* Black moves first (AI side) */

        /* ---- Play one episode ---- */
        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* 1. Encode current state */
            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* 2. Get legal moves */
            std::vector<Step*> steps;
            std::vector<int> actionIndices;
            RL::Tensor actionMask(ACTION_DIM, 1);
            actionMask.zero();
            getLegalActions(currentColor, steps, actionIndices, actionMask);

            if (steps.empty()) {
                /* Current player loses (no legal moves = checkmate/stalemate) */
                int loser = currentColor;
                int winner = (loser == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                         : Stone::COLOR_RED;

                /* Back-propagate terminal reward through the trajectory */
                float terminalReward = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                for (auto &step : trajectory) {
                    step.reward = terminalReward;
                }

                /* Call REINFORCE on the trajectory */
                if (!trajectory.empty()) {
                    dpg.reinforce1(trajectory, learningRate);
                }

                /* Update stats */
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;

                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black(AI)" : "Red",
                           moveNum, getWinRate(Stone::COLOR_BLACK));
                }
                break;
            }

            /* 3. Select move using current policy */
            RL::Tensor &policyOut = dpg.action(state);

            /* Mask illegal actions */
            for (int i = 0; i < ACTION_DIM; i++) {
                if (actionMask[i] < 0.5f) policyOut[i] = 0.0f;
            }
            float sum = 0.0f;
            for (int i = 0; i < ACTION_DIM; i++) sum += policyOut[i];
            if (sum > 1e-9f) {
                for (int i = 0; i < ACTION_DIM; i++) policyOut[i] /= sum;
            } else {
                for (int i = 0; i < ACTION_DIM; i++)
                    policyOut[i] = actionMask[i] / (float)steps.size();
            }

            int selectedAction;
            float r = (float)std::rand() / (float)RAND_MAX;
            if (r < dpg.exploringRate) {
                /* Random exploration */
                int idx = std::rand() % (int)steps.size();
                selectedAction = actionIndices[idx];
            } else {
                selectedAction = RL::Random::categorical(policyOut);
            }

            /* Map to Step */
            Step *chosenStep = nullptr;
            for (std::size_t i = 0; i < actionIndices.size(); i++) {
                if (actionIndices[i] == selectedAction) {
                    chosenStep = steps[i];
                    break;
                }
            }
            if (chosenStep == nullptr) {
                chosenStep = steps[0];
                selectedAction = actionIndices[0];
            }

            /* 4. Create one-hot action tensor */
            RL::Tensor oneHotAction(ACTION_DIM, 1);
            oneHotAction.zero();
            oneHotAction[selectedAction] = 1.0f;

            /* 5. Execute the move */
            double dummy = 0.0;
            chess.moveForward(chosenStep, dummy);

            /* 6. Compute immediate reward */
            /*
               注意: 这一条即时奖励**走不到学习者那里** —— 下面三个出口 (无合法走法 /
               吃将 / 步数用尽) 都会用终局常量把整条轨迹的 reward 覆盖掉。所以本函数的
               训练目标是"纯终局 ±1 的黑方视角", 是自洽的 (编码也是黑为正), 不存在符号
               错 —— 但**即时奖励被整条丢掉了**, 物质收益完全没有进入学习信号。要接上它
               需要先把量纲调平 (computeReward 给的是 value*10, 与 ±1 差一个量级),
               属于独立一项。符号口径的审查见 docs/agents_design.md §17。
            */
            float reward = computeReward(*chosenStep, currentColor);

            /* 7. Record transition */
            trajectory.emplace_back(state, oneHotAction, reward);

            Steps::instance().put(steps);

            /* 8. Check game over after move */
            int gameResult = chess.isGameOver();
            if (gameResult != Stone::COLOR_NONE) {
                float terminalReward = (gameResult == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                for (auto &step : trajectory) {
                    step.reward = terminalReward;
                }
                if (!trajectory.empty()) {
                    dpg.reinforce1(trajectory, learningRate);
                }
                totalEpisodes++;
                if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f\n",
                           ep + 1, episodes,
                           (gameResult == Stone::COLOR_BLACK) ? "Black(AI)" : "Red",
                           moveNum + 1, getWinRate(Stone::COLOR_BLACK));
                }
                break;
            }

            /* 9. Switch side */
            if (selfPlay) {
                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK
                                   : Stone::COLOR_RED;
            } else {
                /* AI always plays Black, opponent is random */
                break;  /* one move per episode for AI training */
            }
        }

        /* If we hit maxMoves without terminal, treat as draw */
        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经 reinforce + 计数过一次
       之后, 这里会再 reinforce 一次、再计一次局数。改用 getResult() 判断。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            /* Draw: reward = 0 for all steps */
            for (auto &step : trajectory) {
                step.reward = 0.0f;
            }
            if (!trajectory.empty()) {
                dpg.reinforce1(trajectory, learningRate);
            }
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw (%d moves)\n",
                       ep + 1, episodes, maxMoves);
            }
        }

        /* Decay exploration rate */
        dpg.exploringRate *= 0.9999f;
        if (dpg.exploringRate < 0.05f) dpg.exploringRate = 0.05f;
    }
}

/* ------------------------------------------------------------------ */
/*  warmupFromCurrent                                                   */
/* ------------------------------------------------------------------ */
void PGEagent::warmupFromCurrent(int episodes, int maxMoves)
{
    if (episodes <= 0) return;

    /* ---- Save current board state ---- */
    struct StoneSave { int x, y, alive; };
    std::vector<StoneSave> saved(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        saved[i] = {s->pos.x, s->pos.y, s->alive};
    }

    /* ---- Self-play episodes ---- */
    for (int ep = 0; ep < episodes; ep++) {
        /* Restore to saved state */
        for (int i = 0; i < 32; i++) {
            Stone *s = chess.stones[i];
            s->alive = saved[i].alive;
            s->pos.x = saved[i].x;
            s->pos.y = saved[i].y;
        }
        /* Rebuild map */
        chess.m_map.clear();
        for (int i = 0; i < 32; i++) {
            Stone *s = chess.stones[i];
            if (s && s->alive) chess.m_map[s->pos] = s;
        }
        /* Clear history (repetition detection not needed) */
        chess.history.clear();

        /* Online training for this episode */
        beginOnline();
        int currentColor = Stone::COLOR_BLACK;
        int moveCount = 0;

        for (moveCount = 0; moveCount < maxMoves; moveCount++) {
            RL::Tensor stateBefore(STATE_DIM, 1);
            encodeState(stateBefore);

            Step step = selectMove(currentColor, true);  /* training=true (ε-greedy) */
            if (!step.valid) {
                /* No legal moves → current player loses */
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                endOnline(winner, Stone::COLOR_BLACK);
                break;
            }

            double dummy = 0.0;
            chess.moveForward(&step, dummy);

            recordOnline(step, currentColor, stateBefore);

            int gameResult = chess.isGameOver();
            if (gameResult != Stone::COLOR_NONE) {
                endOnline(gameResult, Stone::COLOR_BLACK);
                break;
            }

            currentColor = (currentColor == Stone::COLOR_RED)
                               ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }

        if (moveCount >= maxMoves && chess.isGameOver() == Stone::COLOR_NONE) {
            endOnline(Stone::COLOR_NONE, Stone::COLOR_BLACK);
        }
    }

    /* ---- Restore original board state ---- */
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        s->alive = saved[i].alive;
        s->pos.x = saved[i].x;
        s->pos.y = saved[i].y;
    }
    chess.m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s && s->alive) chess.m_map[s->pos] = s;
    }
    chess.history.clear();
}

/* ------------------------------------------------------------------ */
/*  savePolicy / loadPolicy                                            */
/* ------------------------------------------------------------------ */
bool PGEagent::savePolicy(const std::string &filepath)
{
    if (dpg.policyNet.save(filepath) == 0) {
        return true;
    }
    return false;
}

bool PGEagent::loadPolicy(const std::string &filepath)
{
    if (dpg.policyNet.load(filepath) == 0) {
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/*  Online training (human-vs-AI)                                      */
/* ------------------------------------------------------------------ */
void PGEagent::beginOnline()
{
    m_onlineTrajectory.clear();
}

void PGEagent::recordOnline(const Step& s, int color, const RL::Tensor& stateBefore)
{
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    int aidx = stepToActionIdx(s);
    oneHotAction[aidx] = 1.0f;
    float reward = computeReward(s, color);
    /* 同 trainSelfPlay: 这里存的即时奖励会被 endOnline 的终局常量整条覆盖
       (524-533), 所以线上路径也是"纯终局 ±1 的黑方视角", 自洽但丢了物质收益。 */
    m_onlineTrajectory.emplace_back(stateBefore, oneHotAction, reward);
}

void PGEagent::endOnline(int winner, int myColor)
{
    float terminalReward = 0.0f;
    if (winner == myColor) terminalReward = 1.0f;
    else if (winner != Stone::COLOR_NONE) terminalReward = -1.0f;

    /* Assign terminal reward to all steps */
    for (auto &step : m_onlineTrajectory) {
        step.reward = terminalReward;
    }

    if (!m_onlineTrajectory.empty()) {
        dpg.reinforce1(m_onlineTrajectory, learningRate);
    }

    totalEpisodes++;
    if (winner == Stone::COLOR_BLACK) totalWins[1]++;
    if (winner == Stone::COLOR_RED) totalWins[0]++;

    m_onlineTrajectory.clear();

    /* Decay exploration rate */
    dpg.exploringRate *= 0.9999f;
    if (dpg.exploringRate < 0.05f) dpg.exploringRate = 0.05f;
}



/* ------------------------------------------------------------------ */
/*  exploreAndTrain: 走子前"先探索环境 + 在线训练一次" (仿 snakeAI)      */
/* ------------------------------------------------------------------ */
bool PGEagent::exploreAndTrain(int color, int rolloutSteps)
{
    if (rolloutSteps <= 0) {
        return false;
    }

    /* 探索策略: snakeAI 的 dpgAction 用 gumbelMax() 采样 */
    auto pick = [this](const RL::Tensor &state, int /*turn*/) -> int {
        RL::Tensor &a = dpg.gumbelMax(state);
        return a.argmax();
    };

    std::vector<RL::Step> traj;
    traj.reserve((std::size_t)rolloutSteps);
    /* 视角换算: 本 agent 的编码是"黑为正"的绝对坐标, 而 rolloutFromCurrent 给的
       奖励是**走子方视角**的 (吃子者为正) —— 红方走子时符号必须翻过来, 否则红方那
       一半样本的目标整体是反的。判定与全部同类位置见 docs/agents_design.md §17。 */
    auto onTrans = [&traj](const Step &chosen, int actionIdx,
                           const RL::Tensor &s, const RL::Tensor &/*ns*/,
                           float r, bool /*done*/) {
        RL::Tensor oneHot(ACTION_DIM, 1);
        oneHot.zero();
        oneHot[actionIdx] = 1.0f;
        traj.emplace_back(s, oneHot, moverRewardToBlackFrame(r, chosen));
    };

    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick, onTrans);

    bool trained = false;
    if (!traj.empty()) {
        /* snakeAI 用的是 reinforce(); 这里保持同样的选择 (它会给"当时犹豫"的步子降权) */
        dpg.reinforce(traj, learningRate);
        trained = true;
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, reinforce 1 次";
    return trained;
}

/* ==================================================================
 *  自检报告 (界面"模型自检"面板) —— 契约见 aiagent.h, 说明见 pgagent.h
 * ================================================================== */

namespace {

/*
 * dpg 的隐层宽度 (只读诊断)。
 *
 * PGEagent 自己**不存**这个数 (构造函数直接把它转给了 RL::DPG), 所以只能从网里读
 * 回来: policyNet 的输出头是 `Layer<Softmax>::_(hiddenDim, ACTION_DIM)`, 它的
 * inputDim 就是构造时传进来的 hiddenDim。iFcLayer 的两个维度是 public 的。
 *
 * 为什么这里出现 const_cast: selfCheckReport() 是 const 的, 而 `Net::operator[]`
 * 没有 const 重载。这里**只读**一个公开的维度字段 —— 不调 forward/backward, 不写
 * 任何成员, 也不长期持有这个引用; 而对象本身在 GUI 里本来就不是 const 的
 * (m_sfPG 是 `PGEagent*`), 所以不违反"只读"契约。读不到就返回 0, 面板里不印这一项
 * (层结构以后换成不含 iFcLayer 的骨干时, 这里会安静地降级, 而不是崩)。
 */
int dpgHiddenWidthOf(const RL::DPG &dpg)
{
    RL::Net &net = const_cast<RL::Net &>(dpg.policyNet);
    if (net.size() == 0) {
        return 0;
    }
    RL::iFcLayer *head = dynamic_cast<RL::iFcLayer *>(net[net.size() - 1]);
    return (head != nullptr) ? (int)head->inputDim : 0;
}

/*
 * 一个局面上的动作别名 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了几个
 * 互不相同的着法)。
 *
 * 与 dqnmcts_agent.cpp 的 aliasOfPosition() 量的是同一件事, 但这里**不抄哈希**:
 * 唯一要用到的就是 `stepToActionIdx` 那一份公式, 所以签名只要 `const PGEagent &` 就够
 * (那个成员已经是 const, 见头文件)。于是面板数的一定是训练时用的同一个公式 ——
 * DQNMCTS 那边因为成员函数不是 const, 只能把哈希再抄一份并靠注释提醒"改一处必须改
 * 两处"; 这里换成编译期耦合: 谁把 `stepToActionIdx` 的 const 去掉, 这里当场编不过。
 *
 * 去重: 每个槽位里存 (棋子 id, 目标格) 的**位压缩键** —— id ≤ 31、x ≤ 9、y ≤ 8, 所以
 * `(id << 16) | (x << 8) | y` 无碰撞, 且不分配字符串。统计的是**互不相同的走法**数,
 * 而不是合法着法列表的条数 (列表里若有重复项, 也不该被算成"两个走法挤在一起")。
 *
 * 只读: 只看传进来的走法列表 (调用方负责 `Steps::instance().put()` 归还)。
 */
void aliasOfPosition(const PGEagent &ag, const std::vector<Step *> &legal,
                     int &legalCount, int &slotCount, int &worstSlot)
{
    std::map<int, std::set<long long> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        const long long key = ((long long)s.id << 16)
                            | ((long long)s.nextPos.x << 8)
                            | (long long)s.nextPos.y;
        bucket[ag.stepToActionIdx(s)].insert(key);
    }
    legalCount = (int)legal.size();
    slotCount = (int)bucket.size();
    worstSlot = 0;
    for (std::map<int, std::set<long long> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        if ((int)it->second.size() > worstSlot) {
            worstSlot = (int)it->second.size();
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  只报告**结构 / 口径**类事实, 不报棋力。三类读数各自对应一个"曲线好看但棋力
 *  没动"的已知原因:
 *
 *   (1) 表示层 —— 规则上下文通道 0 个。
 *       90 维状态是"10×9 每格一个子力值" (本文件 encodeState), 里面**没有**走子方、
 *       没有重复进度、没有无吃子进度、没有被将标记。实测 (probe_dqnmcts_aliasing [2],
 *       见 docs/issues_review.md 零之二点二十二): 同一个局面 vs 重复 1 次/3 次、
 *       无吃子 0 手 vs 120 手、轮到红 vs 轮到黑, encodeState 的输出**逐字节相同**。
 *       那次实测用的编码与本文件是**同一份** (90 维、±子力/7.0、同一个哈希), 所以
 *       结论逐字适用。
 *       代价: "三次重复判和"与"60 回合无吃子判和"是**规则结果**, 会直接决定终局与
 *       回报 (和棋 0 / 分出胜负 ±1), 而被将/将杀更是决定胜负; 这些量在网络的输入里
 *       根本不存在, 于是同一份输入对应着多个不同的终局期望 —— 网络只能学到它们的
 *       **平均**, 学不到区分。(顺带: 60 回合自然限着要 120 半回合, 而 GUI 训练一局
 *       上限只有 60 ply, 所以它在训练路径上甚至不可达, 见 chess.h DrawReason 的说明。)
 *
 *   (2) 动作层 —— 动作别名。
 *       `stepToActionIdx` 把 (棋子 id, 目标格) 哈希进 128 个槽位, 而真实走法空间是
 *       8100 = 90×90。实测 (同一份探针 [1]): 标准开局 44 个合法着法只落在 38 个槽位
 *       上 (挤掉 6 个, 最挤槽位背 3 个着法); 中局 96 个局面平均挤掉 5.16 个; 而
 *       **跨局面**的累计碰撞率只有 0.03 —— 也就是说"128 个槽位不够用"不是问题,
 *       问题是**同一个局面内**的挤压。
 *       代价是表示层的地板: 两个不同的走法共用同一个 logit 槽位时, 策略头**无法
 *       表达**"走 A 不走 B"这件事, 反传时两个着法的梯度被平均到同一组参数上, 于是
 *       这个槽位的概率永远同时代表它们。这跟训练预算无关, 再训多久都不会消失。
 *       本 agent 还有一处具体后果 (别名在这里比在 DQN 里更硬): selectMove()/train()
 *       把采样到的槽位映射回着法时用的是"取第一个下标相同的着法", 于是一个被别名
 *       占用的槽位**永远只会走出那批着法里的第一个**, 同槽的其余着法一步都走不到,
 *       连探索都覆盖不到它们。
 *
 *   (3) 回报口径 —— per-agent 契约, 所以单独印一行。
 *       computeReward 是**走子方视角**的 (吃子者恒为正, 见那里的注释), 终局常量 ±1
 *       也是走子方, 和棋 0。这一条 2026-09 修过一次 (原先是黑方视角, 红方白吃一个车
 *       拿负奖励), 回归钉在 test_match 的 [2.6] 节 —— 印在面板上是防止它再被改回去。
 *
 *  刻意**不**在这里报棋力: 面板能回答的是"这个模型值不值得继续训", 而不是"它有多强"。
 *  自对弈胜率与 surrogate 损失都在下面印出来了, 但两者都不是棋力 (赢家与输家是同一
 *  份权重; 损失只说明网络与自己的目标一致)。棋力只有带置信区间的锚点对局能回答。
 *
 *  **只读**: 全程只用局部构造的棋盘与局部解码, 不碰 this->chess、不改任何成员 ——
 *  它会在对局中途被 GUI 线程调用, 而那正是搜索线程在用同一个棋盘的时候。
 * ------------------------------------------------------------------ */
std::string PGEagent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /* ---- 1. 规模 ---- */
    std::snprintf(buf, sizeof(buf),
                  "状态 %d 维 (10x9 每格一个子力值) | 动作 %d 槽位\n",
                  STATE_DIM, ACTION_DIM);
    out += buf;

    /* ---- 2. 规则上下文通道: 这个编码一个都没有 (理由见函数头) ---- */
    std::snprintf(buf, sizeof(buf),
                  "规则上下文通道: 0 个 (走子方/重复/无吃子/被将 全不可观测)\n");
    out += buf;

    /* ---- 3. 动作别名 ----
       标准开局那一份是**确定性**的 (与当前棋盘、训练进度都无关), 所以打开面板就有读数。

       局面用**默认构造**的棋盘: `Chess::Chess()` 末尾就是 reset() (见 chess.cpp), 拿到
       的正是初始局面。这里刻意**不**写 `Chess probe(chess)` (dqnmcts/sacaz 那两处是那样
       写的): 那会去**读 this->chess**, 而本函数是 GUI 线程调的, 那一刻搜索线程可能正在
       同一个棋盘上 moveForward/moveBack —— 与写线程并发的读; 而这份副本随后就被丢掉,
       读它本来就是白读。整段只碰局部对象, 这才是"绝不动棋盘"最省事的守法。 */
    int legalN = 0, slotN = 0, worstN = 0;
    {
        Chess probe;
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(*this, legal, legalN, slotN, worstN);
        Steps::instance().put(legal);   /* 借出的 Step* 必须归还: 面板每刷新一次都会走到这里 */
    }
    std::snprintf(buf, sizeof(buf),
                  "动作别名(标准开局): %d 个合法着法 -> %d 个策略槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法)\n",
                  legalN, slotN, legalN - slotN, worstN);
    out += buf;

    /* ---- 4. 即时回报口径 (per-agent 契约) ---- */
    std::snprintf(buf, sizeof(buf),
                  "即时奖励口径: 走子方视角 (吃子恒为正; 终局 ±1 给走子方, 和棋 0)\n");
    out += buf;

    /* ---- 5. 超参 / 网络规模 ----
       eps 印成 "初始 -> 当前": 衰减有两处 (本文件的 train/endOnline 用 0.9999/下限 0.05,
       RL::DPG::reinforce1 用 0.99999/下限 0.1), 后者的下限更高, 所以实际稳定在 0.1。
       这里只印两个端点, 不写"下限"字样 —— 免得与那两处不一致的常数对口。 */
    std::snprintf(buf, sizeof(buf),
                  "超参: gamma=%g | lr=%g | eps %g -> %g (初始 -> 当前)\n",
                  (double)gamma, (double)learningRate,
                  (double)initialExploringRate, (double)dpg.exploringRate);
    out += buf;

    const int hidden = dpgHiddenWidthOf(dpg);
    if (hidden > 0) {
        std::snprintf(buf, sizeof(buf), "网络: 隐层 %d 维 | 骨干 %d 层 | 参数 %lld\n",
                      hidden, (int)dpg.policyNet.size(),
                      dpg.policyNet.paramCount());
    } else {
        std::snprintf(buf, sizeof(buf), "网络: 隐层宽度读不到 (骨干 %d 层) | 参数 %lld\n",
                      (int)dpg.policyNet.size(), dpg.policyNet.paramCount());
    }
    out += buf;

    /* ---- 6. 对局与"自对弈"胜率 ---- */
    std::snprintf(buf, sizeof(buf),
                  "对局: %d 局 | 自对弈胜率(黑) %.2f (赢家与输家是同一份权重 -> 不是棋力)\n",
                  getTotalEpisodes(), (double)getWinRate(Stone::COLOR_BLACK));
    out += buf;

    /* ---- 7. 最近一次训练损失 ----
       NaN = 还没 reinforce 过 (见 aiagent.h: 非有限值不上报, 曲线控件会丢弃它们, 所以
       不画假的水平线 —— 只是没有点)。 */
    const float loss = getLastTrainLoss();
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "最近损失: %.6g (REINFORCE surrogate -ΣA·logπ 的批均值; advantage 已"
                      "标准化 -> 量级只反映信噪比, 不可跨 agent 比较, 也不是棋力)\n",
                      (double)loss);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "最近损失: 未上报 (NaN, 还没 reinforce 过; 曲线控件会丢弃非有限值)\n");
    }
    out += buf;

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局"
           " (带 95% 置信区间的 Elo 差) 回答\n";
    return out;
}
