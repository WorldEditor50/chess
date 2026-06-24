#include "dqnmcts_agent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"

/* ================================================================
 *  DQNMCTSAgent - DQN + MCTS implementation
 *
 *  Combines Deep Q-Network (DQN) with Monte Carlo Tree Search.
 *  DQN provides Q(s,a) values that replace random rollouts.
 *  MCTS uses UCB1 (no prior — unlike PUCT in PPOMCTSAgent).
 *  Training uses standard DQN experience replay.
 * ================================================================ */

/* ------------------------------------------------------------------
 *  Constructor
 * ------------------------------------------------------------------ */
DQNMCTSAgent::DQNMCTSAgent(Chess &chess_,
                           int hiddenDim,
                           float gamma_,
                           float lr,
                           float eps,
                           float uc)
    : AgentBase(),
      chess(chess_),
      dqn(STATE_DIM, hiddenDim, ACTION_DIM),
      gamma(gamma_),
      learningRate(lr),
      C(uc),
      maxMemorySize(4096),
      batchSize(32),
      replaceTargetInterval(256),
      learnCounter(0),
      totalEpisodes(0),
      m_trainingMode(false),
      m_onlineStepCount(0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;
    dqn.gamma = gamma_;
    dqn.exploringRate = eps;
    std::srand((unsigned int)std::time(nullptr));
}

/* AgentBase interface */
Step DQNMCTSAgent::getBestMove(int color)
{
    return selectMove(color, 400, false);
}

std::string DQNMCTSAgent::getName() const
{
    return "DQN+MCTS (C=" + std::to_string(C) + ")";
}

/* ------------------------------------------------------------------
 *  encodeState: 10x9 board -> 90-dim tensor
 *    Same encoding as DQNAgent/PPOMCTSAgent:
 *      0 = empty
 *      +val = Black piece
 *      -val = Red piece
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::encodeState(RL::Tensor &state)
{
    state.zero();
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || s->alive == false) continue;

        int idx = s->pos.x * 9 + s->pos.y;
        float val = (s->type >= 0 && s->type < 7) ? PIECE_VALUES[s->type] : 0.0f;
        if (s->color == Stone::COLOR_RED) {
            val = -val;
        }
        state[idx] = val / 7.0f;  /* normalize to [-1, +1] */
    }
}

/* ------------------------------------------------------------------
 *  getLegalActions
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::getLegalActions(int color,
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

/* ------------------------------------------------------------------
 *  stepToActionIdx:  deterministic hash
 * ------------------------------------------------------------------ */
int DQNMCTSAgent::stepToActionIdx(const Step &s)
{
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)ACTION_DIM);
}

/* ------------------------------------------------------------------
 *  computeReward:  immediate material reward from a move
 * ------------------------------------------------------------------ */
float DQNMCTSAgent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;

    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr || victim->alive == false) return 0.0f;

    if (victim->type == Stone::TYPE_JIANG) {
        return 100.0f;
    }

    float reward = victim->value * 10.0f;
    return (color == Stone::COLOR_BLACK) ? reward : -reward;
}

/* ------------------------------------------------------------------
 *  evaluateLeaf
 *
 *  Evaluate the current board state using DQN:
 *   - Encode the state
 *   - Get Q-values from DQN
 *   - Mask illegal actions
 *   - Return max_a Q(s,a) as the leaf value estimate
 *
 *  The Q-values are clamped to [-1, 1] for consistency with the
 *  MCTS +1/-1 reward range.
 * ------------------------------------------------------------------ */
double DQNMCTSAgent::evaluateLeaf(int color, RL::Tensor &qValues)
{
    /* Check terminal state first */
    int gameResult = chess.isGameOver();
    if (gameResult != Stone::COLOR_NONE) {
        /* Return ±1 from the current player's perspective */
        return (gameResult == color) ? 1.0 : -1.0;
    }

    /* Encode state */
    RL::Tensor state(STATE_DIM, 1);
    encodeState(state);

    /* Get Q-values from DQN */
    qValues = dqn.action(state);  /* deep copy */

    /* Get legal action mask */
    std::vector<Step*> steps;
    std::vector<int> actionIndices;
    RL::Tensor actionMask(ACTION_DIM, 1);
    getLegalActions(color, steps, actionIndices, actionMask);
    Steps::instance().put(steps);

    /* Check if there are any legal moves */
    bool hasLegal = false;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (actionMask[i] > 0.5f) {
            hasLegal = true;
            break;
        }
    }

    if (!hasLegal) {
        /* No legal moves -> current player loses */
        return -1.0;
    }

    /* Mask illegal actions and find max Q among legal ones */
    double maxQ = -1e9;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (actionMask[i] > 0.5f) {
            double q = qValues[i];
            if (q > maxQ) maxQ = q;
        }
    }

    /* Clamp to [-1, 1] for consistency */
    if (maxQ > 1.0) maxQ = 1.0;
    if (maxQ < -1.0) maxQ = -1.0;

    return maxQ;
}

/* ------------------------------------------------------------------
 *  getUCB1:  UCB1 score for a child node
 *
 *  Formula:
 *    UCB1 = W/N + C * sqrt(ln(N_parent) / N)
 *
 *  Unvisited children return a very large score to ensure they
 *  are explored first.
 * ------------------------------------------------------------------ */
double DQNMCTSAgent::getUCB1(int childID, int parentVisits) const
{
    const DQNMCTSNode &child = nodes[childID];

    if (child.visitCount == 0) {
        return std::numeric_limits<double>::max();
    }

    double exploitation = child.totalReward / (double)child.visitCount;
    double exploration = C * std::sqrt(std::log((double)parentVisits) / (double)child.visitCount);
    return exploitation + exploration;
}

/* ------------------------------------------------------------------
 *  selectMove:  DQN+MCTS for a single move decision
 *
 *  1. Create root node with all legal moves
 *  2. Run MCTS iterations:
 *     - SELECTION: UCB1 traversal
 *     - EXPANSION: add a new node (pick random untried move)
 *     - EVALUATION: DQN max Q(s,a) instead of random rollout
 *     - BACKPROP: propagate value through path
 *  3. Return the move at the root with highest visit count
 * ------------------------------------------------------------------ */
Step DQNMCTSAgent::selectMove(int color, int iterations, bool training)
{
    /* Cache state for online training */
    if (m_trainingMode) {
        m_cachedState = RL::Tensor(STATE_DIM, 1);
        encodeState(m_cachedState);
    }

    nodes.clear();

    /* ---- Create root node ---- */
    std::vector<Step*> rootSteps;
    std::vector<int> rootActionIndices;
    RL::Tensor rootActionMask(ACTION_DIM, 1);
    rootActionMask.zero();
    getLegalActions(color, rootSteps, rootActionIndices, rootActionMask);

    if (rootSteps.empty()) {
        return Step();
    }

    DQNMCTSNode rootNode;
    rootNode.currentColor = color;
    rootNode.parentID = -1;
    for (Step *s : rootSteps) {
        rootNode.untriedSteps.push_back(*s);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(rootNode);
    int rootID = 0;

    /* ---- Main MCTS loop ---- */
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* ====== Phase 1: SELECTION ======
         *
         * Traverse the tree using UCB1 until we reach a node
         * that still has untried moves or is a terminal node.
         */
        while (nodes[nodeID].untriedSteps.empty()
               && !nodes[nodeID].childIDs.empty()) {

            int parentVisits = nodes[nodeID].visitCount;
            int bestChild = -1;
            double bestUCB = -std::numeric_limits<double>::max();

            for (int childID : nodes[nodeID].childIDs) {
                double ucb = getUCB1(childID, parentVisits);
                if (ucb > bestUCB) {
                    bestUCB = ucb;
                    bestChild = childID;
                }
            }

            if (bestChild < 0) break;

            /* Execute the move on the board */
            double dummyReward = 0.0;
            chess.moveForward(&nodes[bestChild].step, dummyReward);

            nodeID = bestChild;
            path.push_back(nodeID);
        }

        /* ====== Phase 2: EXPANSION ====== */
        if (!nodes[nodeID].untriedSteps.empty()) {
            int moveIdx = std::rand() % (int)nodes[nodeID].untriedSteps.size();
            Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];

            nodes[nodeID].untriedSteps.erase(
                nodes[nodeID].untriedSteps.begin() + moveIdx);

            /* Execute the move */
            double dummyReward = 0.0;
            chess.moveForward(&chosenStep, dummyReward);

            /* Create child node */
            int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                ? Stone::COLOR_BLACK
                                : Stone::COLOR_RED;

            DQNMCTSNode newNode(nodeID, chosenStep, nextColor);

            /* Pre-compute legal moves for the child */
            std::vector<Step*> childSteps;
            std::vector<int> childActionIndices;
            RL::Tensor childActionMask(ACTION_DIM, 1);
            childActionMask.zero();
            getLegalActions(nextColor, childSteps,
                            childActionIndices, childActionMask);

            for (Step *s : childSteps) {
                newNode.untriedSteps.push_back(*s);
            }
            Steps::instance().put(childSteps);

            /* Register child in tree */
            nodes.push_back(newNode);
            int newNodeID = (int)nodes.size() - 1;
            nodes[nodeID].childIDs.push_back(newNodeID);

            nodeID = newNodeID;
            path.push_back(nodeID);
        }

        /* ====== Phase 3: EVALUATION ======
         *
         * Use DQN max Q(s,a) as the leaf evaluation instead of
         * random rollouts. This is the key difference from pure MCTS.
         */
        RL::Tensor qValues(ACTION_DIM, 1);
        double reward = evaluateLeaf(nodes[nodeID].currentColor, qValues);

        /* ====== Phase 4: BACKPROPAGATION ====== */
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[path[i]].visitCount++;
            nodes[path[i]].totalReward += reward;
            reward = -reward;       /* flip perspective */
        }

        /* Undo all moves played during this iteration */
        for (int i = (int)path.size() - 1; i > 0; i--) {
            double dummyReward = 0.0;
            const Step &s = nodes[path[i]].step;
            chess.moveBack(&s, dummyReward);
        }
    }

    /* ---- Select the best move ---- */
    int bestChildID = -1;
    int maxVisits = -1;

    /* If training, use ε-greedy on top of MCTS visit distribution */
    if (training) {
        float r = (float)std::rand() / (float)RAND_MAX;
        if (r < dqn.exploringRate && !nodes[rootID].childIDs.empty()) {
            /* Pick a random child */
            int idx = std::rand() % (int)nodes[rootID].childIDs.size();
            bestChildID = nodes[rootID].childIDs[idx];
            return nodes[bestChildID].step;
        }
    }

    /* Default: pick the child with most visits */
    for (int childID : nodes[rootID].childIDs) {
        if (nodes[childID].visitCount > maxVisits) {
            maxVisits = nodes[childID].visitCount;
            bestChildID = childID;
        }
    }

    if (bestChildID >= 0) {
        return nodes[bestChildID].step;
    }

    return Step();
}

/* ------------------------------------------------------------------
 *  trainVsRandom:  DQN+MCTS vs random opponent
 *
 *  Black (DQN+MCTS) vs Red (random moves).
 *  DQN learns via experience replay from the MCTS-guided moves.
 *
 *  Key difference from DQNAgent::trainVsRandom:
 *  - Each AI move is decided by MCTS search (guided by DQN)
 *  - DQN trains on the actual MCTS-selected moves
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::trainVsRandom(int episodes, int iterations,
                                  int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;
        RL::Tensor state(STATE_DIM, 1);
        encodeState(state);

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* Black (AI: DQN+MCTS) */
            if (currentColor == Stone::COLOR_BLACK) {
                /* Run MCTS to select a move (with exploration during training) */
                Step chosenStep = selectMove(Stone::COLOR_BLACK, iterations, true);
                if (chosenStep.id == Stone::ID_NONE) {
                    /* No moves - black loses */
                    RL::Tensor dummyAction(ACTION_DIM, 1);
                    dummyAction.zero();
                    RL::Tensor zeroState(STATE_DIM, 1);
                    zeroState.zero();
                    dqn.perceive(state, dummyAction, zeroState, -1.0f, true);
                    totalEpisodes++;
                    totalWins[0]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: Red(random) wins (AI no moves), %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes, moveNum, getWinRate(), dqn.exploringRate);
                    }
                    break;
                }

                /* Record the experience */
                int selectedAction = stepToActionIdx(chosenStep);
                RL::Tensor oneHotAction(ACTION_DIM, 1);
                oneHotAction.zero();
                oneHotAction[selectedAction] = 1.0f;
                float reward = computeReward(chosenStep, Stone::COLOR_BLACK);

                double dummy = 0.0;
                chess.moveForward(&chosenStep, dummy);

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

/* ------------------------------------------------------------------
 *  trainSelfPlay:  DQN+MCTS self-play training
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::trainSelfPlay(int episodes, int iterations,
                                  int maxMoves, bool verbose)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;
        RL::Tensor state(STATE_DIM, 1);
        encodeState(state);

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* Run MCTS to select a move */
            Step chosenStep = selectMove(currentColor, iterations, true);
            if (chosenStep.id == Stone::ID_NONE) {
                /* Current player has no moves -> loses */
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK
                                 : Stone::COLOR_RED;
                RL::Tensor dummyAction(ACTION_DIM, 1);
                dummyAction.zero();
                RL::Tensor zeroState(STATE_DIM, 1);
                zeroState.zero();
                float terminalReward = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                dqn.perceive(state, dummyAction, zeroState, terminalReward, true);
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black" : "Red",
                           moveNum, getWinRate(), dqn.exploringRate);
                }
                break;
            }

            /* Record the experience */
            int selectedAction = stepToActionIdx(chosenStep);
            RL::Tensor oneHotAction(ACTION_DIM, 1);
            oneHotAction.zero();
            oneHotAction[selectedAction] = 1.0f;
            float reward = computeReward(chosenStep, currentColor);

            double dummy = 0.0;
            chess.moveForward(&chosenStep, dummy);

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
                           moveNum + 1, getWinRate(), dqn.exploringRate);
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

/* ------------------------------------------------------------------
 *  warmupFromCurrent
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::warmupFromCurrent(int episodes, int iterations,
                                     int maxMoves)
{
    if (episodes <= 0) return;

    /* ---- Save current board state ---- */
    struct StoneSave { int x, y, alive; };
    std::vector<StoneSave> saved(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        saved[i] = {s->pos.x, s->pos.y, s->alive};
    }

    /* ---- Self-play episodes using DQN+MCTS ---- */
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

        /* ---- Online training episode via selectMove/recordExperience ---- */
        m_trainingMode = true;
        int currentColor = Stone::COLOR_BLACK;
        m_onlineStepCount = 0;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* selectMove(..., training=true) caches state into m_cachedState */
            Step step = selectMove(currentColor, iterations, true);
            if (step.id == Stone::ID_NONE) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                endOnlineEpisode(winner);
                break;
            }

            double dummy = 0.0;
            chess.moveForward(&step, dummy);

            recordExperience(step, currentColor);

            int gameResult = chess.isGameOver();
            if (gameResult != Stone::COLOR_NONE) {
                endOnlineEpisode(gameResult);
                break;
            }

            currentColor = (currentColor == Stone::COLOR_RED)
                               ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }

        if (chess.isGameOver() == Stone::COLOR_NONE) {
            endOnlineEpisode(Stone::COLOR_NONE);
        }
        m_trainingMode = false;
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

/* ------------------------------------------------------------------
 *  recordExperience
 *
 *  Records a transition (s, a, r, s') into DQN's replay buffer.
 *  Must be called AFTER the caller has executed chess.moveForward(&chosenStep, ...)
 *  i.e. the board is already in the next state s'.
 *
 *  preconditions:
 *    - selectMove(..., true) was called earlier (caching m_cachedState)
 *    - chess.moveForward has been called (board shows s')
 *    - chosenStep is the move that was played
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::recordExperience(const Step &chosenStep, int color)
{
    if (chosenStep.id == Stone::ID_NONE) return;

    int selectedAction = stepToActionIdx(chosenStep);
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    oneHotAction[selectedAction] = 1.0f;
    float reward = computeReward(chosenStep, color);

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

    dqn.perceive(m_cachedState, oneHotAction, nextState,
                 done ? terminalReward : reward, done);

    /* Update the cached state for the next step */
    m_cachedState = nextState;
    m_onlineStepCount++;
}

/* ------------------------------------------------------------------
 *  endOnlineEpisode
 *
 *  Called when the game ends (during online training mode).
 *  Updates statistics and triggers a learning step.
 *
 *  gameResult: Stone::COLOR_BLACK, Stone::COLOR_RED, or Stone::COLOR_NONE (draw)
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::endOnlineEpisode(int gameResult)
{
    totalEpisodes++;
    if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
    if (gameResult == Stone::COLOR_RED)   totalWins[0]++;

    /* Perform a learning step after the episode ends */
    learnCounter++;
    if (learnCounter % 4 == 0) {
        dqn.learn(maxMemorySize, replaceTargetInterval,
                  batchSize, learningRate);
    }

    m_onlineStepCount = 0;
}

/* ------------------------------------------------------------------
 *  saveModel / loadModel
 * ------------------------------------------------------------------ */
bool DQNMCTSAgent::saveModel(const std::string &filepath)
{
    dqn.save(filepath);
    return true;
}

bool DQNMCTSAgent::loadModel(const std::string &filepath)
{
    dqn.load(filepath);
    return true;
}
