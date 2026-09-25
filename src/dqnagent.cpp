#include "dqnagent.h"
#include "agentrollout.hpp"
#include "rl/util.hpp"

#include <cstdio>   /* selfCheckReport 的 snprintf */
#include <map>      /* 动作别名的槽位分组 */
#include <set>      /* 动作别名的"互不相同的走法"去重 */

/* ------------------------------------------------------------------ */
/*  Piece type encoding values                                         */
/* ------------------------------------------------------------------ */
static constexpr float PIECE_VALUES[7] = {
    1.0f,   /* TYPE_CHE   */
    2.0f,   /* TYPE_MA    */
    3.0f,   /* TYPE_PAO   */
    4.0f,   /* TYPE_BING  */
    5.0f,   /* TYPE_JIANG */
    6.0f,   /* TYPE_SHI   */
    7.0f    /* TYPE_XIANG */
};

/* ------------------------------------------------------------------ */
/*  Constructor                                                        */
/* ------------------------------------------------------------------ */
DQNAgent::DQNAgent(Chess &chess_,
                   int hiddenDim,
                   float gamma_,
                   float lr,
                   float eps)
    : AgentBase(),
      chess(chess_),
      dqn(STATE_DIM, hiddenDim, ACTION_DIM),
      gamma(gamma_),
      initialExploringRate(eps),
      learningRate(lr),
      maxMemorySize(4096),
      batchSize(32),
      replaceTargetInterval(256),
      learnCounter(0),
      totalEpisodes(0)
{
    totalWins[0] = 0; /* red */
    totalWins[1] = 0; /* black */
    dqn.gamma = gamma_;
    dqn.exploringRate = eps;
}

/* AgentBase interface */
Step DQNAgent::getBestMove(int color)
{
    return selectMove(color, false);
}

std::string DQNAgent::getName() const
{
    return "Deep Q-Network (DQN)";
}

/* ------------------------------------------------------------------ */
/*  encodeState:  10x9 board -> 90-dim tensor                          */
/* ------------------------------------------------------------------ */
void DQNAgent::encodeState(RL::Tensor &state)
{
    state.zero();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || !s->alive) continue;

        int idx = s->pos.x * 9 + s->pos.y;
        float val = (s->type >= 0 && s->type < 7)
                        ? PIECE_VALUES[s->type]
                        : 0.0f;
        if (s->color == Stone::COLOR_RED) {
            val = -val;
        }
        state[idx] = val / 7.0f;  /* normalize to [-1, +1] */
    }
}

/* ------------------------------------------------------------------ */
/*  getLegalActions                                                     */
/* ------------------------------------------------------------------ */
void DQNAgent::getLegalActions(int color,
                               std::vector<Step*> &steps,
                               std::vector<int> &actionIndices,
                               RL::Tensor &actionMask)
{
    actionMask.zero();
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());

    for (Step *s : steps) {
        int aidx = stepToActionIdx(*s);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

/* ------------------------------------------------------------------ */
/*  stepToActionIdx                                                     */
/*  const: 无状态, 只为让 selfCheckReport() const 复用同一份公式       */
/*  (见 dqnagent.h 的说明)。                                           */
/* ------------------------------------------------------------------ */
int DQNAgent::stepToActionIdx(const Step &s) const
{
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)ACTION_DIM);
}

/* ------------------------------------------------------------------ */
/*  stepToOneHot                                                        */
/* ------------------------------------------------------------------ */
void DQNAgent::stepToOneHot(const Step &s, RL::Tensor &onehot)
{
    onehot.zero();
    int idx = stepToActionIdx(s);
    onehot[idx] = 1.0f;
}

/* ------------------------------------------------------------------ */
/*  computeReward:  一步的即时奖励 (走子方视角, 含每步代价)             */
/*  Phase 1 起统一走 stone.h 的 stepReward(): 材质系数 0.1、吃將不给材质 */
/*  奖励 (终局常量负责)、每步代价 -0.005。见 stone.h REWARD_* 的实测。  */
/* ------------------------------------------------------------------ */
float DQNAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) return stepReward(false, false, 0.0);

    Stone *victim = chess.stones[s.nextId];
    /*
       不检查 victim->alive —— 见 pgagent.cpp 里同一处的说明: DQNAgent::trainAfterMove
       是在 moveForward() 之后被调用的, 那时被吃子已 alive=false, 加判断会让吃子
       奖励恒为 0。
    */
    if (victim == nullptr) return stepReward(false, false, 0.0);

    /*
       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来的黑方视角写法会让红方白吃一个
       黑车拿到负奖励, 与终局 (走子方视角 ±1) 相反。
       回归钉在 test_match 的 [2.6] 节。
    */
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* ------------------------------------------------------------------ */
/*  selectMove                                                          */
/* ------------------------------------------------------------------ */
Step DQNAgent::selectMove(int color, bool training)
{
    RL::Tensor state(STATE_DIM, 1);
    encodeState(state);

    std::vector<Step*> steps;
    std::vector<int> actionIndices;
    RL::Tensor actionMask(ACTION_DIM, 1);
    actionMask.zero();
    getLegalActions(color, steps, actionIndices, actionMask);

    if (steps.empty()) {
        return Step();
    }

    RL::Tensor &qValues = dqn.action(state);

    for (int i = 0; i < ACTION_DIM; i++) {
        if (actionMask[i] < 0.5f) {
            qValues[i] = -1e9f;
        }
    }

    int selectedAction;
    if (training) {
        float r = (float)std::rand() / (float)RAND_MAX;
        if (r < dqn.exploringRate) {
            int idx = std::rand() % (int)steps.size();
            selectedAction = actionIndices[idx];
        } else {
            selectedAction = qValues.argmax();
        }
    } else {
        selectedAction = qValues.argmax();
    }

    for (std::size_t i = 0; i < actionIndices.size(); i++) {
        if (actionIndices[i] == selectedAction) {
            Step result = *steps[i];
            Steps::instance().put(steps);
            return result;
        }
    }

    Step result = *steps[0];
    Steps::instance().put(steps);
    return result;
}

/* ------------------------------------------------------------------ */
/*  trainVsRandom                                                       */
/* ------------------------------------------------------------------ */
void DQNAgent::trainVsRandom(int episodes, int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;
        RL::Tensor state(STATE_DIM, 1);
        encodeState(state);

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* Black (AI) */
            if (currentColor == Stone::COLOR_BLACK) {
                std::vector<Step*> steps;
                std::vector<int> actionIndices;
                RL::Tensor actionMask(ACTION_DIM, 1);
                actionMask.zero();
                getLegalActions(currentColor, steps, actionIndices, actionMask);

                if (steps.empty()) {
                    RL::Tensor dummyAction(ACTION_DIM, 1);
                    dummyAction.zero();
                    RL::Tensor zeroState(STATE_DIM, 1);
                    zeroState.zero();
                    dqn.perceive(state, dummyAction, zeroState, -1.0f, true);
                    Steps::instance().put(steps);
                    totalEpisodes++;
                    totalWins[0]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: Red(random) wins (AI no moves), %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes, moveNum, getWinRate(), dqn.exploringRate);
                    }
                    break;
                }

                RL::Tensor &qValues = dqn.action(state);
                for (int i = 0; i < ACTION_DIM; i++) {
                    if (actionMask[i] < 0.5f) qValues[i] = -1e9f;
                }

                int selectedAction;
                float r = (float)std::rand() / (float)RAND_MAX;
                if (r < dqn.exploringRate) {
                    int idx = std::rand() % (int)steps.size();
                    selectedAction = actionIndices[idx];
                } else {
                    selectedAction = qValues.argmax();
                }

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

                RL::Tensor oneHotAction(ACTION_DIM, 1);
                oneHotAction.zero();
                oneHotAction[selectedAction] = 1.0f;
                float reward = computeReward(*chosenStep, currentColor);

                double dummy = 0.0;
                chess.moveForward(chosenStep, dummy);
                Steps::instance().put(steps);

                RL::Tensor nextState(STATE_DIM, 1);
                encodeState(nextState);

                int gameResult = chess.isGameOver();
                bool done = (gameResult != Stone::COLOR_NONE);
                float terminalReward = (gameResult == Stone::COLOR_BLACK) ? 1.0f : -1.0f;

                dqn.perceive(state, oneHotAction, nextState,
                             done ? terminalReward : reward, done);

                if (done) {
                    totalEpisodes++;
                    if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                    if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes,
                               (gameResult == Stone::COLOR_BLACK) ? "Black(AI)" : "Red(random)",
                               moveNum + 1, getWinRate(), dqn.exploringRate);
                    }
                    break;
                }

                state = nextState;
                currentColor = Stone::COLOR_RED;

                learnCounter++;
                if (learnCounter % 4 == 0) {
                    dqn.learn(maxMemorySize, replaceTargetInterval,
                              batchSize, learningRate);
                }
            }

            /* Red (random) */
            if (currentColor == Stone::COLOR_RED) {
                std::vector<Step*> redSteps;
                chess.sample(Stone::COLOR_RED, redSteps);

                if (redSteps.empty()) {
                    RL::Tensor dummyAction(ACTION_DIM, 1);
                    dummyAction.zero();
                    RL::Tensor zeroState(STATE_DIM, 1);
                    zeroState.zero();
                    dqn.perceive(state, dummyAction, zeroState, 1.0f, true);
                    Steps::instance().put(redSteps);
                    totalEpisodes++;
                    totalWins[1]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: Black(AI) wins (Red no moves), %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes, moveNum + 1, getWinRate(), dqn.exploringRate);
                    }
                    break;
                }

                int idx = std::rand() % (int)redSteps.size();
                double dummy = 0.0;
                chess.moveForward(redSteps[idx], dummy);
                Steps::instance().put(redSteps);

                int gameResult = chess.isGameOver();
                if (gameResult != Stone::COLOR_NONE) {
                    totalEpisodes++;
                    if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                    if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes,
                               (gameResult == Stone::COLOR_BLACK) ? "Black(AI)" : "Red(random)",
                               moveNum + 1, getWinRate(), dqn.exploringRate);
                    }
                    break;
                }

                state = RL::Tensor(STATE_DIM, 1);
                encodeState(state);
                currentColor = Stone::COLOR_BLACK;
            }
        }

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw, win_rate=%.2f, eps=%.4f\n",
                       ep + 1, episodes, getWinRate(), dqn.exploringRate);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  trainSelfPlay                                                       */
/* ------------------------------------------------------------------ */
void DQNAgent::trainSelfPlay(int episodes, int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;
        RL::Tensor state(STATE_DIM, 1);
        encodeState(state);

        int totalMoves = 0;
        for (totalMoves = 0; totalMoves < maxMoves; totalMoves++) {
            std::vector<Step*> steps;
            std::vector<int> actionIndices;
            RL::Tensor actionMask(ACTION_DIM, 1);
            actionMask.zero();
            getLegalActions(currentColor, steps, actionIndices, actionMask);

            if (steps.empty()) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK
                                 : Stone::COLOR_RED;
                RL::Tensor dummyAction(ACTION_DIM, 1);
                dummyAction.zero();
                RL::Tensor zeroState(STATE_DIM, 1);
                zeroState.zero();
                float terminalReward = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                dqn.perceive(state, dummyAction, zeroState, terminalReward, true);
                Steps::instance().put(steps);
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black" : "Red",
                           totalMoves, getWinRate(), dqn.exploringRate);
                }
                break;
            }

            RL::Tensor &qValues = dqn.action(state);
            for (int i = 0; i < ACTION_DIM; i++) {
                if (actionMask[i] < 0.5f) qValues[i] = -1e9f;
            }

            int selectedAction;
            float r = (float)std::rand() / (float)RAND_MAX;
            if (r < dqn.exploringRate) {
                int idx = std::rand() % (int)steps.size();
                selectedAction = actionIndices[idx];
            } else {
                selectedAction = qValues.argmax();
            }

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

            RL::Tensor oneHotAction(ACTION_DIM, 1);
            oneHotAction.zero();
            oneHotAction[selectedAction] = 1.0f;
            float reward = computeReward(*chosenStep, currentColor);
            /*
               视角换算 (2026-09): 本 agent 的编码是"黑为正"的绝对坐标, 网络表达的
               是**对黑方的价值**; 而 computeReward 现在给的是**走子方视角** (吃子者为
               正)。原来这里直接存走子方视角的值, 于是红方吃子是 + 而终局红方胜是 -,
               同一条轨迹里两种口径混用。终局常量本来就是黑方视角, 所以这里补上红方
               的符号翻转。判定与全部同类位置见 docs/agents_design.md §17。
            */
            reward = moverRewardToBlackFrame(reward, currentColor);

            double dummy = 0.0;
            chess.moveForward(chosenStep, dummy);
            Steps::instance().put(steps);

            RL::Tensor nextState(STATE_DIM, 1);
            encodeState(nextState);

            int gameResult = chess.isGameOver();
            bool done = (gameResult != Stone::COLOR_NONE);
            float terminalReward;
            if (done) {
                terminalReward = (gameResult == Stone::COLOR_BLACK) ? 1.0f
                               : (gameResult == Stone::COLOR_RED) ? -1.0f : 0.0f;
            } else {
                terminalReward = reward;
            }

            dqn.perceive(state, oneHotAction, nextState,
                         done ? terminalReward : reward, done);

            if (done) {
                totalEpisodes++;
                if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                           ep + 1, episodes,
                           (gameResult == Stone::COLOR_BLACK) ? "Black" : "Red",
                           totalMoves + 1, getWinRate(), dqn.exploringRate);
                }
                break;
            }

            state = nextState;
            currentColor = (currentColor == Stone::COLOR_RED)
                               ? Stone::COLOR_BLACK
                               : Stone::COLOR_RED;

            learnCounter++;
            if (learnCounter % 4 == 0) {
                dqn.learn(maxMemorySize, replaceTargetInterval,
                          batchSize, learningRate);
            }
        }

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw, win_rate=%.2f, eps=%.4f\n",
                       ep + 1, episodes, getWinRate(), dqn.exploringRate);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  warmupFromCurrent                                                   */
/* ------------------------------------------------------------------ */
void DQNAgent::warmupFromCurrent(int episodes, int maxMoves)
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
        chess.m_map.clear();
        for (int i = 0; i < 32; i++) {
            Stone *s = chess.stones[i];
            if (s && s->alive) chess.m_map[s->pos] = s;
        }
        chess.history.clear();

        /* Self-play episode using DQN ε-greedy + experience replay */
        int currentColor = Stone::COLOR_BLACK;
        RL::Tensor state(STATE_DIM, 1);
        encodeState(state);

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            std::vector<Step*> steps;
            std::vector<int> actionIndices;
            RL::Tensor actionMask(ACTION_DIM, 1);
            actionMask.zero();
            getLegalActions(currentColor, steps, actionIndices, actionMask);

            if (steps.empty()) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                RL::Tensor dummyAction(ACTION_DIM, 1);
                dummyAction.zero();
                RL::Tensor zeroState(STATE_DIM, 1);
                zeroState.zero();
                float terminalReward = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                dqn.perceive(state, dummyAction, zeroState, terminalReward, true);
                Steps::instance().put(steps);
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                break;
            }

            RL::Tensor &qValues = dqn.action(state);
            for (int i = 0; i < ACTION_DIM; i++) {
                if (actionMask[i] < 0.5f) qValues[i] = -1e9f;
            }

            int selectedAction;
            float r = (float)std::rand() / (float)RAND_MAX;
            if (r < dqn.exploringRate) {
                int idx = std::rand() % (int)steps.size();
                selectedAction = actionIndices[idx];
            } else {
                selectedAction = qValues.argmax();
            }

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

            RL::Tensor oneHotAction(ACTION_DIM, 1);
            oneHotAction.zero();
            oneHotAction[selectedAction] = 1.0f;
            float reward = computeReward(*chosenStep, currentColor);
            /* 视角换算 (2026-09, 同 trainSelfPlay): 走子方视角 -> 黑方视角,
               否则红方那一步的即时奖励符号是反的。见 docs/agents_design.md §17。 */
            reward = moverRewardToBlackFrame(reward, currentColor);

            double dummy = 0.0;
            chess.moveForward(chosenStep, dummy);
            Steps::instance().put(steps);

            RL::Tensor nextState(STATE_DIM, 1);
            encodeState(nextState);

            int gameResult = chess.isGameOver();
            bool done = (gameResult != Stone::COLOR_NONE);
            if (done) {
                if (gameResult == Stone::COLOR_BLACK) reward = 1.0f;
                else if (gameResult == Stone::COLOR_RED) reward = -1.0f;
                else reward = 0.0f;
            }

            dqn.perceive(state, oneHotAction, nextState, reward, done);

            if (done) {
                totalEpisodes++;
                if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                break;
            }

            state = nextState;
            currentColor = (currentColor == Stone::COLOR_RED)
                               ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            learnCounter++;
            if (learnCounter % 4 == 0) {
                dqn.learn(maxMemorySize, replaceTargetInterval,
                          batchSize, learningRate);
            }
        }

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            totalEpisodes++;
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
/*  saveModel / loadModel                                              */
/* ------------------------------------------------------------------ */
bool DQNAgent::saveModel(const std::string &filepath)
{
    dqn.save(filepath);
    /* 不无条件返回 true: 写盘失败时 UI 会弹假的"保存成功" */
    return weightFileWritten(filepath);
}

bool DQNAgent::loadModel(const std::string &filepath)
{
    if (!weightFileReadable(filepath)) {
        return false;
    }
    /*
       传播内核的真实结果 (见 ppomcts_agent.cpp 里同一处修正的说明): 载入被拒时
       必须让调用方知道, 而不是按"文件可读"报成功。
    */
    return dqn.load(filepath);
}

/* ------------------------------------------------------------------ */
/*  exploreAndTrain: 走子前"先探索环境 + 在线训练一次" (仿 snakeAI)      */
/* ------------------------------------------------------------------ */
bool DQNAgent::exploreAndTrain(int color, int rolloutSteps, const OpponentPolicy &opponent)
{
    if (rolloutSteps <= 0 || batchSize <= 0) {
        return false;
    }

    /* 探索策略: snakeAI 的 dqnAction 用的是 noiseAction() */
    auto pick = [this](const RL::Tensor &state, int /*turn*/) -> int {
        RL::Tensor &q = dqn.noiseAction(state);
        return q.argmax();
    };
    /* 每收集一条转移, 就把它放进回放池 (perceive) */
    /* 视角换算 (2026-09): rolloutFromCurrent 给的 r 是**走子方视角**的, 而本 agent
       的编码是"黑为正" —— 用 chosen.id 认出走子方再翻符号。见 docs/agents_design.md §17。 */
    auto onTrans = [this](const Step &chosen, int actionIdx,
                          const RL::Tensor &s, const RL::Tensor &ns,
                          float r, bool done) {
        RL::Tensor oneHot(ACTION_DIM, 1);
        oneHot.zero();
        oneHot[actionIdx] = 1.0f;
        dqn.perceive(s, oneHot, ns, moverRewardToBlackFrame(r, chosen), done);
    };

    /* 局部副本: rolloutFromCurrent 会把"真用了几手对手着法"回填到它里面 (P1) */
    OpponentPolicy opp = opponent;
    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick, onTrans, opp);

    bool trained = false;
    if (collected > 0) {
        /* 用这批新鲜经验在线训练一次 (回放池不足 batchSize 时 learn() 自己会跳过) */
        dqn.learn(maxMemorySize, replaceTargetInterval, batchSize, learningRate);
        learnCounter++;
        trained = true;
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, 训练 1 次(池 "
                    + std::to_string((int)dqn.memories.size()) + ")"
                    + opponentRolloutInfo(opp);
    return trained;
}

void DQNAgent::trainAfterMove(const RL::Tensor& stateBefore,
                               const Step& chosenStep,
                               int color,
                               const RL::Tensor& nextState,
                               bool done)
{
    /* Build one-hot action */
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    int aidx = stepToActionIdx(chosenStep);
    oneHotAction[aidx] = 1.0f;

    /* Compute reward */
    /* 视角换算 (2026-09, 同 trainSelfPlay): 走子方视角 -> 黑方视角。
       trainAfterMove 目前没有调用方, 一并修掉只是不让它留在那里当反例。 */
    float reward = moverRewardToBlackFrame(computeReward(chosenStep, color), color);

    if (done) {
        /* Determine terminal reward from game result */
        int gameResult = chess.isGameOver();
        if (gameResult == Stone::COLOR_BLACK) reward = 1.0f;
        else if (gameResult == Stone::COLOR_RED) reward = -1.0f;
        else reward = 0.0f;
    }

    /* Store experience */
    dqn.perceive(stateBefore, oneHotAction, nextState, reward, done);

    /* Increment learning counter and learn periodically */
    learnCounter++;
    if (learnCounter % 4 == 0) {
        dqn.learn(maxMemorySize, replaceTargetInterval,
                  batchSize, learningRate);
    }
}

/* ==================================================================
 *  自检报告 (界面"模型自检"面板) —— 契约见 aiagent.h, 说明见 dqnagent.h
 * ================================================================== */

namespace {

/*
 * 一个局面上的动作别名 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了几个
 * 互不相同的着法)。
 *
 * 与 dqnmcts_agent.cpp 的 aliasOfPosition() 量的是同一件事, 但这里**不抄哈希**:
 * 唯一要用到的就是 `stepToActionIdx` 那一份公式, 所以签名只要 `const DQNAgent &` 就够
 * (那个成员已经是 const, 见头文件)。于是面板数的一定是训练时用的同一个公式 ——
 * DQNMCTS 那边因为成员函数不是 const, 只能把哈希再抄一份并靠注释提醒"改一处必须改
 * 两处"; 这里换成编译期耦合: 谁把 `stepToActionIdx` 的 const 去掉, 这里当场编不过。
 *
 * 去重: 每个槽位里存 (棋子 id, 目标格) 的**位压缩键** —— id ≤ 31、x ≤ 9、y ≤ 8, 所以
 * `(id << 16) | (x << 8) | y` 无碰撞, 且不分配字符串。统计的是**互不相同的走法**数,
 * 而不是合法着法列表的条数 (列表里若有重复项, 那也不该被算成"两个走法挤在一起")。
 *
 * 只读: 只看传进来的走法列表 (调用方负责 `Steps::instance().put()` 归还)。
 */
void aliasOfPosition(const DQNAgent &ag, const std::vector<Step *> &legal,
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
 *       回报 (和棋 0 / 分出胜负 ±1), 被将/将杀更是决定胜负; 这些量在网络的输入里
 *       根本不存在, 于是同一份输入对应着多个不同的终局期望 —— Q 只能学到它们的
 *       **平均**, 学不到区分。(顺带: 60 回合自然限着要 120 半回合, 而 GUI 训练一局
 *       上限只有 60 ply, 所以它在训练路径上甚至不可达, 见 chess.h DrawReason 的说明。)
 *
 *   (2) 动作层 —— 动作别名。
 *       `stepToActionIdx` 把 (棋子 id, 目标格) 哈希进 128 个 Q 槽位, 而真实走法空间
 *       是 8100 = 90×90。实测 (同一份探针 [1]): 标准开局 44 个合法着法只落在 38 个
 *       槽位上 (挤掉 6 个, 最挤槽位背 3 个着法); 中局 96 个局面平均挤掉 5.16 个;
 *       而**跨局面**的累计碰撞率只有 0.03 —— 也就是说"128 个槽位不够用"不是问题,
 *       问题是**同一个局面内**的挤压。
 *       代价是表示层的地板: 两个不同的走法共用同一个 Q 槽位时, Q 头**无法表达**
 *       "走 A 不走 B"这件事, 反传时两个着法的梯度被平均到同一组参数上, 于是这个
 *       槽位的值永远同时代表它们 (DQN 里还多一层: 掩码把同槽的合法着法一起点亮,
 *       argmax 选中它之后, selectMove() 按"第一个下标相同的着法"落子 —— 落的是哪
 *       一个不由网络决定)。这跟训练预算无关, 再训多久都不会消失。
 *
 *   (3) 回报口径 —— per-agent 契约, 所以单独印一行。
 *       computeReward 是**走子方视角**的 (吃子者恒为正, 见那里的注释), 终局常量 ±1
 *       也是走子方, 和棋 0。这一条 2026-09 修过一次 (原先是黑方视角, 红方白吃一个车
 *       拿负奖励), 回归钉在 test_match 的 [2.6] 节 —— 印在面板上是防止它再被改回去。
 *
 *  刻意**不**在这里报棋力: 面板能回答的是"这个模型值不值得继续训", 而不是"它有多强"。
 *  平均平方 TD 误差在下面印出来了, 但它只说明网络与自己的目标一致, 而且量纲是原始 Q
 *  尺度 —— 与 PPO 那种 [-1,1] 值域的 MSE 不可比 (实测 DQN+MCTS 是 22、PPO 是 0.003)。
 *
 *  **只读**: 全程只用局部构造的棋盘与局部解码, 不碰 this->chess、不改任何成员 ——
 *  它会在对局中途被 GUI 线程调用, 而那正是搜索/训练线程在用同一个棋盘的时候。
 * ------------------------------------------------------------------ */
std::string DQNAgent::selfCheckReport() const
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
                  "动作别名(标准开局): %d 个合法着法 -> %d 个 Q 槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法)\n",
                  legalN, slotN, legalN - slotN, worstN);
    out += buf;

    /* ---- 4. 即时回报口径 (per-agent 契约) ---- */
    std::snprintf(buf, sizeof(buf),
                  "即时奖励口径: 走子方视角 (吃子恒为正; 终局 ±1 给走子方, 和棋 0)\n");
    out += buf;

    /* ---- 5. 超参 ----
       eps 印成 "初始 -> 当前": 衰减在 RL::DQN::learn() 里 (×0.9999, 下限 0.1),
       而训练循环每 4 手才调一次 learn, 所以它有对局时会明显高于 0.1。 */
    std::snprintf(buf, sizeof(buf),
                  "超参: gamma=%g | lr=%g | eps %g -> %g (初始 -> 当前)\n",
                  (double)gamma, (double)learningRate,
                  (double)initialExploringRate, (double)getExploreRate());
    out += buf;

    /* ---- 6. 回放与学习节奏 ----
       前面三个数是**每次调 learn() 时传进去的实参**, 不是 learn() 内部的状态;
       learnCounter 是"每 4 手触发一次 learn"的那个计数器 (exploreAndTrain 里还会额外
       调一次), 所以 learn() 的真实调用次数比 learnCounter/4 略多。 */
    std::snprintf(buf, sizeof(buf),
                  "回放: 池 %d (现状 %d 条) | batch %d | 每 %d 次 learn 软同步目标网"
                  " (Polyak 0.01) | learnCounter %d (每 4 次触发一次 learn)\n",
                  maxMemorySize, (int)dqn.memories.size(), batchSize,
                  replaceTargetInterval, learnCounter);
    out += buf;

    std::snprintf(buf, sizeof(buf), "对局: %d 局\n", getTotalEpisodes());
    out += buf;

    /* ---- 7. 最近一次训练损失 ----
       NaN = 还没 learn 过 (见 aiagent.h: 非有限值不上报, 曲线控件会丢弃它们, 所以不画
       假的水平线 —— 只是没有点)。 */
    const float loss = getLastTrainLoss();
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "最近损失: %.6g (最近一批的平均平方 TD 误差; 它只说明网络与自己的"
                      "目标一致, 量纲是原始 Q 尺度, 不可跨 agent 比较, 也不是棋力)\n",
                      (double)loss);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "最近损失: 未上报 (NaN, 还没 learn 过或池 < batchSize; 曲线控件会丢弃"
                      "非有限值)\n");
    }
    out += buf;

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局"
           " (带 95% 置信区间的 Elo 差) 回答\n";
    return out;
}



