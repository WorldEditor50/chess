#include "pgagent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "agentrollout.hpp"
#include "rl/util.hpp"

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
/* ------------------------------------------------------------------ */
int PGEagent::stepToActionIdx(const Step &s)
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
