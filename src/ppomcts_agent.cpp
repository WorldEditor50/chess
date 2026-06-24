#include "ppomcts_agent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"

/* ================================================================
 *  PPOMCTSAgent - PPO + MCTS (AlphaZero-style) implementation
 *
 *  Combines PPO neural network (policy + value) with MCTS tree
 *  search. The PPO actor provides prior probabilities P(s,a) and
 *  the PPO critic provides state values V(s) that replace random
 *  rollouts.
 * ================================================================ */

/* ------------------------------------------------------------------
 *  Constructor
 * ------------------------------------------------------------------ */
PPOMCTSAgent::PPOMCTSAgent(Chess &chess_,
                           int hiddenDim,
                           float gamma_,
                           float lr,
                           float cpuct)
    : AgentBase(),
      chess(chess_),
      ppo(STATE_DIM, hiddenDim, ACTION_DIM),
      gamma(gamma_),
      learningRate(lr),
      c_puct(cpuct),
      totalEpisodes(0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;
    ppo.exploringRate = 1.0f;
    std::srand((unsigned int)std::time(nullptr));
}

/* AgentBase interface */
Step PPOMCTSAgent::getBestMove(int color)
{
    return selectMove(color, 400, 0.0f);
}

std::string PPOMCTSAgent::getName() const
{
    return "PPO+MCTS (AlphaZero)";
}

/* ------------------------------------------------------------------
 *  encodeState: 10x9 board -> 90-dim tensor
 *    Same encoding as PGEagent/DQNAgent:
 *      0 = empty
 *      +val = Black piece
 *      -val = Red piece
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::encodeState(RL::Tensor &state)
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
void PPOMCTSAgent::getLegalActions(int color,
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
int PPOMCTSAgent::stepToActionIdx(const Step &s)
{
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)ACTION_DIM);
}

/* ------------------------------------------------------------------
 *  computeReward:  immediate material reward from a move
 * ------------------------------------------------------------------ */
float PPOMCTSAgent::computeReward(const Step &s, int color)
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
 *  evaluateState
 *
 *  Runs the PPO network's actor and critic heads on `state`.
 *  Returns:
 *    policyOut : softmax policy probabilities P(s,a) from the actor
 *    valueOut  : scalar value estimate V(s) from the critic head
 *
 *  The PPO API (rl/ppo.h):
 *    Tensor &action(const Tensor &state);   // returns policy probabilities
 *    float   value(const Tensor &state);   // returns scalar value V(s)
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::evaluateState(const RL::Tensor &state,
                                 RL::Tensor &policyOut,
                                 float &valueOut)
{
    /* PPO actor head: returns policy (Softmax) probabilities */
    policyOut = ppo.action(state);

    /* PPO critic head: returns scalar value V(s) */
    valueOut = ppo.value(state);
}

/* ------------------------------------------------------------------
 *  getPUCT:  PUCT score for a child node
 *
 *  Formula:
 *    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *
 *  Unvisited children return a very large score to ensure they
 *  are explored first.
 * ------------------------------------------------------------------ */
double PPOMCTSAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[childID];

    if (child.visitCount == 0) {
        /* Always explore unvisited nodes first */
        return std::numeric_limits<double>::max();
    }

    double q = child.getQ();                /* Q(s,a) = W/N */
    double puct = c_puct * child.prior
                  * std::sqrt((double)parentVisits)
                  / (1.0 + (double)child.visitCount);

    return q + puct;
}

/* ------------------------------------------------------------------
 *  selectMove:  PPO+MCTS for a single move decision
 *
 *  1. Encode the current board state
 *  2. Run MCTS simulations:
 *     - Selection: traverse tree using PUCT
 *     - Expansion: add a new node
 *     - Evaluation: use PPO value head (no random rollouts)
 *     - Backpropagation: propagate value through path
 *  3. Return the move at the root with highest visit count
 *     (or sample from visit distribution if temp > 0)
 * ------------------------------------------------------------------ */
Step PPOMCTSAgent::selectMove(int color, int simulations, float temp)
{
    nodes.clear();

    /* ---- Create root node ---- */
    RL::Tensor rootState(STATE_DIM, 1);
    encodeState(rootState);

    /* Evaluate root state with PPO to get policy prior */
    RL::Tensor rootPolicy(ACTION_DIM, 1);
    float rootValue = 0.0f;
    evaluateState(rootState, rootPolicy, rootValue);

    /* Gather legal moves for root */
    std::vector<Step*> rootSteps;
    std::vector<int> rootActionIndices;
    RL::Tensor rootActionMask(ACTION_DIM, 1);
    rootActionMask.zero();
    getLegalActions(color, rootSteps, rootActionIndices, rootActionMask);

    /* Create root AZNode */
    AZNode rootNode;
    rootNode.currentColor = color;
    rootNode.parentID = -1;
    rootNode.parentAction = -1;

    /* Store untried actions (filtered by legal mask) */
    for (std::size_t i = 0; i < rootActionIndices.size(); i++) {
        int aidx = rootActionIndices[i];
        rootNode.untriedActionIndices.push_back(aidx);
        rootNode.untriedSteps.push_back(*rootSteps[i]);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(rootNode);
    int rootID = 0;

    /* ---- Main MCTS loop ---- */
    for (int sim = 0; sim < simulations; sim++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* ====== Phase 1: SELECTION ======
         *
         * Traverse the tree using PUCT until we reach a node
         * that still has untried actions or is a leaf.
         *
         * We need to execute moves on the actual board to
         * keep it in sync with tree traversal.
         */

        /* We need to replay moves from root to current node */
        /* Build a sequence of steps from root to current node */
        std::vector<Step> stepsToExecute;

        while (nodes[nodeID].untriedActionIndices.empty()
               && !nodes[nodeID].childIDs.empty()) {

            int parentVisits = nodes[nodeID].visitCount;
            int bestChild = -1;
            double bestPUCT = -std::numeric_limits<double>::max();

            for (int childID : nodes[nodeID].childIDs) {
                double puct = getPUCT(childID, parentVisits);
                if (puct > bestPUCT) {
                    bestPUCT = puct;
                    bestChild = childID;
                }
            }

            if (bestChild < 0) break;

            /* Record the step to execute later */
            stepsToExecute.push_back(nodes[bestChild].step);

            nodeID = bestChild;
            path.push_back(nodeID);
        }

        /* Execute all steps along the path */
        double dummyReward = 0.0;
        for (const Step &s : stepsToExecute) {
            chess.moveForward(&s, dummyReward);
        }

        /* ====== Phase 2: EXPANSION ====== */
        if (!nodes[nodeID].untriedActionIndices.empty()) {
            /* Pick a random untried action */
            int moveIdx = std::rand()
                          % (int)nodes[nodeID].untriedActionIndices.size();
            int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
            Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];

            /* Remove from untried list */
            nodes[nodeID].untriedActionIndices.erase(
                nodes[nodeID].untriedActionIndices.begin() + moveIdx);
            nodes[nodeID].untriedSteps.erase(
                nodes[nodeID].untriedSteps.begin() + moveIdx);

            /* Execute the move */
            chess.moveForward(&chosenStep, dummyReward);

            /* Create child node */
            int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                ? Stone::COLOR_BLACK
                                : Stone::COLOR_RED;

            /* Evaluate the new state to get the prior for this action */
            RL::Tensor childState(STATE_DIM, 1);
            encodeState(childState);
            RL::Tensor childPolicy(ACTION_DIM, 1);
            float childValue = 0.0f;
            evaluateState(childState, childPolicy, childValue);

            /* The prior for this specific action is P(s, a) from PPO policy */
            float prior = childPolicy[chosenAction];
            if (prior < 1e-9f) prior = 1e-9f; /* avoid zero prior */

            AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

            /* Pre-compute legal moves for the child */
            std::vector<Step*> childSteps;
            std::vector<int> childActionIndices;
            RL::Tensor childActionMask(ACTION_DIM, 1);
            childActionMask.zero();
            getLegalActions(nextColor, childSteps,
                            childActionIndices, childActionMask);

            for (std::size_t i = 0; i < childActionIndices.size(); i++) {
                newNode.untriedActionIndices.push_back(childActionIndices[i]);
                newNode.untriedSteps.push_back(*childSteps[i]);
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
         * Instead of random rollouts, use the PPO critic value
         * estimate V(s) as the leaf evaluation.
         */
        RL::Tensor leafState(STATE_DIM, 1);
        encodeState(leafState);
        RL::Tensor leafPolicy(ACTION_DIM, 1);
        float leafValue = 0.0f;
        evaluateState(leafState, leafPolicy, leafValue);

        double reward = (double)leafValue;

        /* ====== Phase 4: BACKPROPAGATION ====== */
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[path[i]].visitCount++;
            nodes[path[i]].totalValue += reward;
            reward = -reward;       /* flip perspective */
        }

        /* Undo all moves played during this iteration */
        /* We need to undo in reverse order */
        for (int i = (int)path.size() - 1; i > 0; i--) {
            const Step &s = nodes[path[i]].step;
            chess.moveBack(&s, dummyReward);
        }
    }

    /* ---- Select the best move ---- */
    int bestChildID = -1;
    int maxVisits = -1;

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
 *  trainSelfPlay:  PPO+MCTS self-play training loop
 *
 *  For each episode:
 *    1. Run MCTS from the current position
 *    2. Sample a move from the MCTS visit distribution (with temp)
 *    3. Store (state, MCTS policy target, final game result)
 *    4. At the end of the game, use the stored trajectories to
 *       train the PPO network (both actor and critic)
 *
 *  NOTE: This is a simplified version. A full AlphaZero
 *  implementation would also store MCTS visit distributions as
 *  policy targets and do multiple epochs of training.
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::trainSelfPlay(int episodes, int simulations,
                                 int maxMoves, bool verbose,
                                 float tempRoot, float tempFinal)
{
    const int printInterval = std::max(1, episodes / 10);

    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int currentColor = Stone::COLOR_BLACK;

        /* Store trajectories for training */
        std::vector<RL::Step> trajectory;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* Compute temperature: linearly annealed */
            float temp = tempRoot
                       + (tempFinal - tempRoot)
                       * (float)moveNum / (float)maxMoves;

            /* Encode current state */
            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* Run MCTS to get improved policy */
            nodes.clear();

            /* Create root for MCTS */
            RL::Tensor rootPolicy(ACTION_DIM, 1);
            float rootValue = 0.0f;
            evaluateState(state, rootPolicy, rootValue);

            std::vector<Step*> rootSteps;
            std::vector<int> rootActionIndices;
            RL::Tensor rootActionMask(ACTION_DIM, 1);
            rootActionMask.zero();
            getLegalActions(currentColor, rootSteps,
                            rootActionIndices, rootActionMask);

            if (rootSteps.empty()) {
                /* Current player loses */
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK
                                 : Stone::COLOR_RED;
                float finalOutcome = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;

                /* Train PPO on trajectory with terminal outcome */
                if (!trajectory.empty()) {
                    ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
                }

                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;

                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins (no moves), %d moves, win_rate=%.2f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black(AI)" : "Red",
                           moveNum, getWinRate(Stone::COLOR_BLACK));
                }
                Steps::instance().put(rootSteps);
                break;
            }

            /* Build root AZNode */
            AZNode rootNode;
            rootNode.currentColor = currentColor;
            rootNode.parentID = -1;
            rootNode.parentAction = -1;
            for (std::size_t i = 0; i < rootActionIndices.size(); i++) {
                rootNode.untriedActionIndices.push_back(rootActionIndices[i]);
                rootNode.untriedSteps.push_back(*rootSteps[i]);
            }
            nodes.push_back(rootNode);
            int rootID = 0;

            /* MCTS simulations */
            for (int sim = 0; sim < simulations; sim++) {
                /* --- Selection + Expansion --- */
                std::vector<int> path;
                path.push_back(rootID);
                int nodeID = rootID;

                std::vector<Step> stepsToExecute;

                while (nodes[nodeID].untriedActionIndices.empty()
                       && !nodes[nodeID].childIDs.empty()) {

                    int parentVisits = nodes[nodeID].visitCount;
                    int bestChild = -1;
                    double bestPUCT = -std::numeric_limits<double>::max();

                    for (int childID : nodes[nodeID].childIDs) {
                        double puct = getPUCT(childID, parentVisits);
                        if (puct > bestPUCT) {
                            bestPUCT = puct;
                            bestChild = childID;
                        }
                    }
                    if (bestChild < 0) break;

                    stepsToExecute.push_back(nodes[bestChild].step);
                    nodeID = bestChild;
                    path.push_back(nodeID);
                }

                double dummyReward = 0.0;
                for (const Step &s : stepsToExecute) {
                    chess.moveForward(&s, dummyReward);
                }

                if (!nodes[nodeID].untriedActionIndices.empty()) {
                    int moveIdx = std::rand()
                        % (int)nodes[nodeID].untriedActionIndices.size();
                    int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
                    Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];

                    nodes[nodeID].untriedActionIndices.erase(
                        nodes[nodeID].untriedActionIndices.begin() + moveIdx);
                    nodes[nodeID].untriedSteps.erase(
                        nodes[nodeID].untriedSteps.begin() + moveIdx);

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK
                                        : Stone::COLOR_RED;

                    RL::Tensor childState(STATE_DIM, 1);
                    encodeState(childState);
                    RL::Tensor childPolicy(ACTION_DIM, 1);
                    float childValue = 0.0f;
                    evaluateState(childState, childPolicy, childValue);

                    float prior = childPolicy[chosenAction];
                    if (prior < 1e-9f) prior = 1e-9f;

                    AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

                    std::vector<Step*> childSteps;
                    std::vector<int> childActionIndices;
                    RL::Tensor childActionMask(ACTION_DIM, 1);
                    childActionMask.zero();
                    getLegalActions(nextColor, childSteps,
                                    childActionIndices, childActionMask);
                    for (std::size_t i = 0; i < childActionIndices.size(); i++) {
                        newNode.untriedActionIndices.push_back(childActionIndices[i]);
                        newNode.untriedSteps.push_back(*childSteps[i]);
                    }
                    Steps::instance().put(childSteps);

                    nodes.push_back(newNode);
                    int newNodeID = (int)nodes.size() - 1;
                    nodes[nodeID].childIDs.push_back(newNodeID);
                    nodeID = newNodeID;
                    path.push_back(nodeID);
                }

                /* Evaluation with PPO value head */
                RL::Tensor leafState(STATE_DIM, 1);
                encodeState(leafState);
                RL::Tensor leafPolicy(ACTION_DIM, 1);
                float leafValue = 0.0f;
                evaluateState(leafState, leafPolicy, leafValue);

                double reward = (double)leafValue;

                /* Backpropagation */
                for (int i = (int)path.size() - 1; i >= 0; i--) {
                    nodes[path[i]].visitCount++;
                    nodes[path[i]].totalValue += reward;
                    reward = -reward;
                }

                /* Undo moves */
                for (int i = (int)path.size() - 1; i > 0; i--) {
                    const Step &s = nodes[path[i]].step;
                    chess.moveBack(&s, dummyReward);
                }
            }

            /* --- Select move from MCTS visit distribution --- */
            int bestChildID = -1;
            int maxVisits = -1;
            int totalVisits = 0;

            for (int childID : nodes[rootID].childIDs) {
                totalVisits += nodes[childID].visitCount;
                if (nodes[childID].visitCount > maxVisits) {
                    maxVisits = nodes[childID].visitCount;
                    bestChildID = childID;
                }
            }

            Step chosenStep;
            int chosenAction = -1;

            if (bestChildID >= 0 && totalVisits > 0) {
                if (temp > 0.1f) {
                    /* Sample from visit distribution with temperature */
                    RL::Tensor visitProbs(ACTION_DIM, 1);
                    visitProbs.zero();
                    for (int childID : nodes[rootID].childIDs) {
                        const AZNode &child = nodes[childID];
                        double prob = std::pow((double)child.visitCount,
                                               1.0 / (double)temp);
                        visitProbs[child.parentAction] = (float)prob;
                    }
                    /* Normalize */
                    float sum = 0.0f;
                    for (int i = 0; i < ACTION_DIM; i++) sum += visitProbs[i];
                    if (sum > 1e-9f) {
                        for (int i = 0; i < ACTION_DIM; i++) visitProbs[i] /= sum;
                    }
                    chosenAction = RL::Random::categorical(visitProbs);

                    /* Find the Step for this action */
                    for (int childID : nodes[rootID].childIDs) {
                        if (nodes[childID].parentAction == chosenAction) {
                            chosenStep = nodes[childID].step;
                            break;
                        }
                    }
                } else {
                    /* Argmax: pick the child with most visits */
                    chosenStep = nodes[bestChildID].step;
                    chosenAction = nodes[bestChildID].parentAction;
                }

                /* Execute move on the board */
                double dummyReward = 0.0;
                chess.moveForward(&chosenStep, dummyReward);

                /* Check game over */
                int gameResult = chess.isGameOver();
                if (gameResult != Stone::COLOR_NONE) {
                    float finalOutcome = (gameResult == Stone::COLOR_BLACK) ? 1.0f
                                        : (gameResult == Stone::COLOR_RED) ? -1.0f : 0.0f;

                    /* Store final transition */
                    RL::Tensor oneHotAction(ACTION_DIM, 1);
                    oneHotAction.zero();
                    if (chosenAction >= 0) oneHotAction[chosenAction] = 1.0f;
                    trajectory.emplace_back(state, oneHotAction, finalOutcome);

                    /* Train PPO on complete trajectory */
                    if (!trajectory.empty()) {
                        ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
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
                    Steps::instance().put(rootSteps);
                    break;
                }

                /* Store transition */
                RL::Tensor oneHotAction(ACTION_DIM, 1);
                oneHotAction.zero();
                if (chosenAction >= 0) oneHotAction[chosenAction] = 1.0f;
                float reward = computeReward(chosenStep, currentColor);
                trajectory.emplace_back(state, oneHotAction, reward);

                /* Switch side */
                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK
                                   : Stone::COLOR_RED;
            } else {
                /* No legal moves - game over */
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK
                                 : Stone::COLOR_RED;
                float finalOutcome = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;

                if (!trajectory.empty()) {
                    ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black(AI)" : "Red",
                           moveNum, getWinRate(Stone::COLOR_BLACK));
                }
                Steps::instance().put(rootSteps);
                break;
            }

            Steps::instance().put(rootSteps);
        }

        /* Draw if maxMoves reached */
        if (chess.isGameOver() == Stone::COLOR_NONE) {
            if (!trajectory.empty()) {
                ppo.learnSelfPlay(trajectory, 0.0f, learningRate);
            }
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw (%d moves)\n",
                       ep + 1, episodes, maxMoves);
            }
        }
    }
}

/* ------------------------------------------------------------------
 *  warmupFromCurrent
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::warmupFromCurrent(int episodes, int simulations,
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

    /* ---- Self-play episodes using MCTS + PPO ---- */
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

        /* ---- Play one episode with MCTS-guided PPO ---- */
        int currentColor = Stone::COLOR_BLACK;
        std::vector<RL::Step> trajectory;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* ---- MCTS: Selection + Expansion + Backprop ---- */
            nodes.clear();

            RL::Tensor rootPolicy(ACTION_DIM, 1);
            float rootValue = 0.0f;
            evaluateState(state, rootPolicy, rootValue);

            std::vector<Step*> rootSteps;
            std::vector<int> rootActionIndices;
            RL::Tensor rootActionMask(ACTION_DIM, 1);
            rootActionMask.zero();
            getLegalActions(currentColor, rootSteps,
                            rootActionIndices, rootActionMask);

            if (rootSteps.empty()) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                float finalOutcome = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                if (!trajectory.empty()) {
                    ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                Steps::instance().put(rootSteps);
                break;
            }

            AZNode rootNode;
            rootNode.currentColor = currentColor;
            rootNode.parentID = -1;
            rootNode.parentAction = -1;
            for (std::size_t i = 0; i < rootActionIndices.size(); i++) {
                rootNode.untriedActionIndices.push_back(rootActionIndices[i]);
                rootNode.untriedSteps.push_back(*rootSteps[i]);
            }
            nodes.push_back(rootNode);
            int rootID = 0;

            /* MCTS simulations */
            for (int sim = 0; sim < simulations; sim++) {
                std::vector<int> path;
                path.push_back(rootID);
                int nodeID = rootID;
                std::vector<Step> stepsToExecute;

                while (nodes[nodeID].untriedActionIndices.empty()
                       && !nodes[nodeID].childIDs.empty()) {
                    int parentVisits = nodes[nodeID].visitCount;
                    int bestChild = -1;
                    double bestPUCT = -std::numeric_limits<double>::max();
                    for (int childID : nodes[nodeID].childIDs) {
                        double puct = getPUCT(childID, parentVisits);
                        if (puct > bestPUCT) {
                            bestPUCT = puct;
                            bestChild = childID;
                        }
                    }
                    if (bestChild < 0) break;
                    stepsToExecute.push_back(nodes[bestChild].step);
                    nodeID = bestChild;
                    path.push_back(nodeID);
                }

                double dummyReward = 0.0;
                for (const Step &s : stepsToExecute) {
                    chess.moveForward(&s, dummyReward);
                }

                if (!nodes[nodeID].untriedActionIndices.empty()) {
                    int moveIdx = std::rand()
                        % (int)nodes[nodeID].untriedActionIndices.size();
                    int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
                    Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];

                    nodes[nodeID].untriedActionIndices.erase(
                        nodes[nodeID].untriedActionIndices.begin() + moveIdx);
                    nodes[nodeID].untriedSteps.erase(
                        nodes[nodeID].untriedSteps.begin() + moveIdx);

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK : Stone::COLOR_RED;

                    RL::Tensor childState(STATE_DIM, 1);
                    encodeState(childState);
                    RL::Tensor childPolicy(ACTION_DIM, 1);
                    float childValue = 0.0f;
                    evaluateState(childState, childPolicy, childValue);

                    float prior = childPolicy[chosenAction];
                    if (prior < 1e-9f) prior = 1e-9f;

                    AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

                    std::vector<Step*> childSteps;
                    std::vector<int> childActionIndices;
                    RL::Tensor childActionMask(ACTION_DIM, 1);
                    childActionMask.zero();
                    getLegalActions(nextColor, childSteps,
                                    childActionIndices, childActionMask);
                    for (std::size_t i = 0; i < childActionIndices.size(); i++) {
                        newNode.untriedActionIndices.push_back(childActionIndices[i]);
                        newNode.untriedSteps.push_back(*childSteps[i]);
                    }
                    Steps::instance().put(childSteps);

                    nodes.push_back(newNode);
                    int newNodeID = (int)nodes.size() - 1;
                    nodes[nodeID].childIDs.push_back(newNodeID);
                    nodeID = newNodeID;
                    path.push_back(nodeID);
                }

                RL::Tensor leafState(STATE_DIM, 1);
                encodeState(leafState);
                RL::Tensor leafPolicy(ACTION_DIM, 1);
                float leafValue = 0.0f;
                evaluateState(leafState, leafPolicy, leafValue);
                double reward = (double)leafValue;

                for (int i = (int)path.size() - 1; i >= 0; i--) {
                    nodes[path[i]].visitCount++;
                    nodes[path[i]].totalValue += reward;
                    reward = -reward;
                }

                for (int i = (int)path.size() - 1; i > 0; i--) {
                    const Step &s = nodes[path[i]].step;
                    chess.moveBack(&s, dummyReward);
                }
            }

            /* Select move from MCTS visit distribution */
            int bestChildID = -1;
            int maxVisits = -1;
            int totalVisits = 0;
            for (int childID : nodes[rootID].childIDs) {
                totalVisits += nodes[childID].visitCount;
                if (nodes[childID].visitCount > maxVisits) {
                    maxVisits = nodes[childID].visitCount;
                    bestChildID = childID;
                }
            }

            Step chosenStep;
            int chosenAction = -1;

            if (bestChildID >= 0 && totalVisits > 0) {
                /* Argmax (temperature=0) for warmup — deterministic */
                chosenStep = nodes[bestChildID].step;
                chosenAction = nodes[bestChildID].parentAction;

                double dummyReward = 0.0;
                chess.moveForward(&chosenStep, dummyReward);

                int gameResult = chess.isGameOver();
                if (gameResult != Stone::COLOR_NONE) {
                    float finalOutcome = (gameResult == Stone::COLOR_BLACK) ? 1.0f
                                        : (gameResult == Stone::COLOR_RED) ? -1.0f : 0.0f;
                    RL::Tensor oneHotAction(ACTION_DIM, 1);
                    oneHotAction.zero();
                    if (chosenAction >= 0) oneHotAction[chosenAction] = 1.0f;
                    trajectory.emplace_back(state, oneHotAction, finalOutcome);
                    if (!trajectory.empty()) {
                        ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
                    }
                    totalEpisodes++;
                    if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                    if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                    Steps::instance().put(rootSteps);
                    break;
                }

                RL::Tensor oneHotAction(ACTION_DIM, 1);
                oneHotAction.zero();
                if (chosenAction >= 0) oneHotAction[chosenAction] = 1.0f;
                float reward = computeReward(chosenStep, currentColor);
                trajectory.emplace_back(state, oneHotAction, reward);

                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            } else {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                float finalOutcome = (winner == Stone::COLOR_BLACK) ? 1.0f : -1.0f;
                if (!trajectory.empty()) {
                    ppo.learnSelfPlay(trajectory, finalOutcome, learningRate);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                Steps::instance().put(rootSteps);
                break;
            }

            Steps::instance().put(rootSteps);
        }

        if (chess.isGameOver() == Stone::COLOR_NONE) {
            if (!trajectory.empty()) {
                ppo.learnSelfPlay(trajectory, 0.0f, learningRate);
            }
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

/* ------------------------------------------------------------------
 *  saveModel / loadModel
 * ------------------------------------------------------------------ */
bool PPOMCTSAgent::saveModel(const std::string &actorPath,
                             const std::string &criticPath)
{
    ppo.save(actorPath, criticPath);
    return true;
}

bool PPOMCTSAgent::loadModel(const std::string &actorPath,
                             const std::string &criticPath)
{
    ppo.load(actorPath, criticPath);
    return true;
}

bool PPOMCTSAgent::saveModel(const std::string &filepath)
{
    return saveModel(filepath + "_actor", filepath + "_critic");
}

bool PPOMCTSAgent::loadModel(const std::string &filepath)
{
    return loadModel(filepath + "_actor", filepath + "_critic");
}

/* ------------------------------------------------------------------ */
/*  Online training (human-vs-AI)                                      */
/* ------------------------------------------------------------------ */
void PPOMCTSAgent::beginOnline()
{
    m_onlineTrajectory.clear();
}

void PPOMCTSAgent::recordOnline(const Step& s, int color, const RL::Tensor& stateBefore)
{
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    int aidx = stepToActionIdx(s);
    oneHotAction[aidx] = 1.0f;
    float reward = computeReward(s, color);
    m_onlineTrajectory.emplace_back(stateBefore, oneHotAction, reward);
}

void PPOMCTSAgent::endOnline(int winner, int myColor)
{
    float finalOutcome = 0.0f;
    if (winner == myColor) finalOutcome = 1.0f;
    else if (winner != Stone::COLOR_NONE) finalOutcome = -1.0f;

    if (!m_onlineTrajectory.empty()) {
        ppo.learnSelfPlay(m_onlineTrajectory, finalOutcome, learningRate);
    }

    totalEpisodes++;
    if (winner == Stone::COLOR_BLACK) totalWins[1]++;
    if (winner == Stone::COLOR_RED) totalWins[0]++;

    m_onlineTrajectory.clear();
}


