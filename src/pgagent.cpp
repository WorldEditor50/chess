#include "pgagent.h"
#include "rl/layer.h"
#include "rl/loss.h"
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
/*  computeReward:  immediate material reward from a move              */
/*  Returns value from `color`'s perspective:  positive = good         */
/* ------------------------------------------------------------------ */
float PGEagent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;

    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr || victim->alive == false) return 0.0f;

    /* Capturing the Jiang/Shuai is an instant win */
    if (victim->type == Stone::TYPE_JIANG) {
        return 100.0f;
    }

    float reward = victim->value * 10.0f;   /* scale up material gain */
    return (color == Stone::COLOR_BLACK) ? reward : -reward;
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
        if (chess.isGameOver() == Stone::COLOR_NONE) {
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
            if (step.id == Stone::ID_NONE) {
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


