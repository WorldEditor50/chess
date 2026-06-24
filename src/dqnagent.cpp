#include "dqnagent.h"
#include "rl/util.hpp"

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
/* ------------------------------------------------------------------ */
int DQNAgent::stepToActionIdx(const Step &s)
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
/*  computeReward                                                       */
/* ------------------------------------------------------------------ */
float DQNAgent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;

    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr || !victim->alive) return 0.0f;

    if (victim->type == Stone::TYPE_JIANG) {
        return 100.0f;
    }

    float reward = victim->value * 10.0f;
    return (color == Stone::COLOR_BLACK) ? reward : -reward;
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

        if (chess.isGameOver() == Stone::COLOR_NONE) {
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

        if (chess.isGameOver() == Stone::COLOR_NONE) {
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

        if (chess.isGameOver() == Stone::COLOR_NONE) {
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
    return true;
}

bool DQNAgent::loadModel(const std::string &filepath)
{
    dqn.load(filepath);
    return true;
}

/* ------------------------------------------------------------------ */
/*  Online training: observe and learn after each AI move              */
/*  Called by ChessBoard after executing the AI move.                  */
/*  DQN trains incrementally via perceive()+learn().                   */
/* ------------------------------------------------------------------ */
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
    float reward = computeReward(chosenStep, color);

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



