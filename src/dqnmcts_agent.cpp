#include "dqnmcts_agent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "agentrollout.hpp"
#include "rl/util.hpp"

#include <cstdio>   /* selfCheckReport 的 snprintf */
#include <map>      /* aliasOfPosition 的槽位分组 */
#include <set>      /* aliasOfPosition 的"互不相同的走法"去重 */

/* ================================================================
 *  DQNMCTSAgent - DQN + MCTS implementation
 *
 *  Combines Deep Q-Network (DQN) with Monte Carlo Tree Search.
 *  DQN provides Q(s,a) values that replace random rollouts.
 *  MCTS uses UCB1 (no prior — unlike PUCT in PPOMCTSAgent).
 *  Training uses standard DQN experience replay.
 * ================================================================ */

namespace {

/*
 * 一手的终局判定, 统一走 `getResult()` (Phase 6 的"终局口径统一")。
 *
 * 为什么必须有这个函数: 本文件原先在三处收尾都写
 *     int gameResult = chess.isGameOver();
 *     bool done = (gameResult != Stone::COLOR_NONE);
 * 而 `Chess::isGameOver()` 只认"将/帅还在不在场上" —— 它**不认**将杀、困毙、
 * 三次重复、60 回合自然限着。于是:
 *
 *   * 正常对局几乎永远不 done: 双方走到手数上限被截断, 那一手照常写成
 *     (s,a,r,s',done=false), Q 目标 = r(材质) + gamma*maxQ 一路自举。
 *     实测 (probe_dqnmcts_aliasing [3]): 6 局全部撞在 200 手上限, 终局信号 0 条。
 *     也就是说 Q 学到的是"舍不得丢子", 不是"怎么赢"—— 终局 ±1 从来没进过目标。
 *   * `trainVsRandom` 更糟: 它没有"非终局就用即时奖励"那一支, 于是**每一手**
 *     都按 done 写入、奖励取 `(gameResult==BLACK)?1:-1` = **-1**。等于告诉网络
 *     "你走的每一步都是输棋"。这解释了那个"最低 22"的损失量级。
 *
 * `outcomeForMover()` (chess.h) 把 Chess::Result 换算成**走子方视角**的 ±1/0,
 * 三处即时奖励 (`computeReward`) 也都是走子方视角 —— 口径一致后再按各调用点的
 * 需要换算到黑方/白方视角。
 *
 *   result : 必须是 chess.getResult(chess.sideToMove) 的结果 (落子之后)
 * 返回 true 表示**这一手之后已终局** (outcome 已写好)。
 */
bool terminalOutcomeOf(const Chess &c, float &outcome)
{
    const int res = const_cast<Chess &>(c).getResult(c.sideToMove);
    if (res == Chess::RESULT_ONGOING) {
        return false;
    }
    /* mover = 刚走完的那一方 = 走子方的对手 (getResult 的参数是"轮到谁走") */
    const int moverColor = (c.sideToMove == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                              : Stone::COLOR_RED;
    outcome = outcomeForMover(res, moverColor);   /* chess.h 的自由函数 */
    return true;
}

/*
 * 把 Chess::Result / "撞上限" 归到 DQNMCTSAgent::EndCode。
 * 单独一个函数是为了让**三个**收尾点写同一条映射, 不再各写各的 if。
 */
int endCodeOfResult(int chessResult)
{
    switch (chessResult) {
    case Chess::RESULT_RED_WIN:   return DQNMCTSAgent::END_RED_WIN;
    case Chess::RESULT_BLACK_WIN: return DQNMCTSAgent::END_BLACK_WIN;
    case Chess::RESULT_DRAW:      return DQNMCTSAgent::END_DRAW;
    default:                      return DQNMCTSAgent::END_CAP;   /* ONGOING */
    }
}

/*
 * 一个局面上的动作别名 (走法数 -> 用到的 Q 槽位数)。
 *
 * 必须**局部解码**而不是调 `ag.stepToActionIdx`: 这个函数会拿 `chess` 的副本反复
 * 走子回退地枚举, 而成员函数没有任何状态, 逐字抄一份哈希即可; 更要紧的是
 * `selectMove` 用的是同一个下标口径, 两者必须一致 —— 不一致时探针/面板会与实际
 * 训练脱节, 所以这里的哈希与 `stepToActionIdx` 是同一份公式 (改一处必须改两处,
 * probe_dqnmcts_aliasing 会把两份都对一遍)。
 */
int aliasActionIdxOf(const Step &s)
{
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % (unsigned long long)DQNMCTSAgent::ACTION_DIM);
}

/*
 * 统计某个局面上"合法着法 -> Q 槽位"的别名情况。**只读**: 棋盘引用不再改动。
 * legalCount/slotCount 写回给调用方累加。
 */
void aliasOfPosition(const std::vector<Step *> &legal,
                     int &legalCount, int &slotCount, int &worstSlot)
{
    std::map<int, std::set<std::string> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        char key[64];
        std::snprintf(key, sizeof(key), "%d>%d,%d", s.id, s.nextPos.x, s.nextPos.y);
        bucket[aliasActionIdxOf(s)].insert(std::string(key));
    }
    legalCount = (int)legal.size();
    slotCount = (int)bucket.size();
    worstSlot = 0;
    for (std::map<int, std::set<std::string> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        worstSlot = std::max(worstSlot, (int)it->second.size());
    }
}

}  // namespace

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
 *  computeReward:  一步的即时奖励 (走子方视角, 含每步代价)
 *
 *  Phase 1 起统一走 stone.h 的 stepReward(): 材质系数 0.1、吃將不给材质奖励
 *  (终局常量负责)、每步代价 -0.005。见 stone.h REWARD_* 的说明与实测数字。
 * ------------------------------------------------------------------ */
float DQNMCTSAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) return stepReward(false, false, 0.0);

    Stone *victim = chess.stones[s.nextId];
    /*
       不检查 victim->alive —— 见 pgagent.cpp 里同一处的说明: 调用方常在
       moveForward() 之后求奖励, 那时被吃子已 alive=false, 加判断会让吃子奖励恒为 0。
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

    /*
       自检计数: 记下**这一个局面**的合法集与它用到几个 Q 槽位。
         * 先在 `getLegalActions` 之后、`put(rootSteps)` **之前**算 ——
           `put()` 会把 Step 还回对象池, 之后读到的是已被复用的内容。
         * 每手只算一次 (不是每个 MCTS 迭代一次): 迭代里那次 (childSteps) 每次
           都要分配 map/set/字符串, 在 400 次迭代下会变成可观测的常数开销;
           而根节点的数已经代表"真实对局里遇到的局面"。
    */
    {
        int legalN = 0, slotN = 0, worst = 0;
        aliasOfPosition(rootSteps, legalN, slotN, worst);
        aliasMoves += legalN;
        aliasIndexed += slotN;
        aliasClearedMoves += (legalN - slotN);
        if (worst > aliasWorstSlot) { aliasWorstSlot = worst; }
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
                if (!chosenStep.valid) {
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

                /*
                   终局判定: 原来这里是
                       int gameResult = chess.isGameOver();
                       bool done = (gameResult != Stone::COLOR_NONE);
                       float terminalReward = (gameResult == COLOR_BLACK) ? 1.0f : -1.0f;
                   两个错都在这一处:
                     (1) isGameOver() 不认将杀/判和 ⇒ 正常对局永远不 done;
                     (2) **没有"非终局就用即时奖励"那一支** ⇒ 每一手的奖励都被写成
                         terminalReward; 而 gameResult 非黑胜时它恒为 -1.0f, 于是
                         "每走一步 = 输一盘"进了回放池。这是损失下不去的直接原因。
                   现在统一走 getResult() + outcomeForMover() (Phase 6 "终局口径统一"),
                   再换算到黑方视角 —— 本函数其余部分的即时奖励也是黑方视角。
                */
                float outcomeMover = 0.0f;
                const bool done = terminalOutcomeOf(chess, outcomeMover);
                const float terminalReward =
                    moverRewardToBlackFrame(outcomeMover, Stone::COLOR_BLACK);

                dqn.perceive(state, oneHotAction, nextState,
                             done ? terminalReward : reward, done);

                if (done) {
                    totalEpisodes++;
                    /* 统计一律走 winnerOfResult(): Chess::Result 与 Stone::Color 数值
                       错位, 直接比较会把红胜记成黑胜 (见 chess.h 的说明)。 */
                    const int winner = winnerOfResult(chess.getResult(chess.sideToMove));
                    if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                    if (winner == Stone::COLOR_RED) totalWins[0]++;
                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                               ep + 1, episodes,
                               (winner == Stone::COLOR_BLACK) ? "Black(AI)" : "Red(random)",
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
        const bool decidedInLoop = (finalResult != Chess::RESULT_ONGOING);
        if (!decidedInLoop) {
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw, win_rate=%.2f, eps=%.4f\n",
                       ep + 1, episodes, getWinRate(), dqn.exploringRate);
            }
        }
        /*
           自检计数 (界面"模型自检"的终局通道读数) —— **一局恰好一条**, 所以放在
           循环外、并且两个出口都归到这里。
           这个位置是被断言逼出来的: 第一版写在"每走一手"之后, 于是 12 手的一局被
           记成 12 局 (test_dqnmcts 第 [6](5) 条断言 "跑完一局必须恰好记一条" 当场
           抓到 12 != 1)。自检面板上的数字是要给人当决策依据的, 所以它自己必须被钉住。
           `chess.isGameOver()` 另记一次, 用来量"旧口径会不会看见这一局" —— 两个数
           的差就是 2026-09 那次"终局口径统一"修掉了多少被漏掉的终局。
        */
        noteEnd(decidedInLoop ? finalResult : Chess::RESULT_ONGOING,
                chess.isGameOver() != Stone::COLOR_NONE);
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
            if (!chosenStep.valid) {
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
            /* 视角换算 (2026-09): 本 agent 的编码是"黑为正", 而 computeReward 给的
               是走子方视角 —— 红方走子时符号要翻。终局常量 (gameResult == BLACK ?
               1 : -1) 本来就是黑方视角, 所以只有即时奖励那一支需要换算。
               见 docs/agents_design.md §17。 */
            reward = moverRewardToBlackFrame(reward, currentColor);

            double dummy = 0.0;
            chess.moveForward(&chosenStep, dummy);

            RL::Tensor nextState(STATE_DIM, 1);
            encodeState(nextState);

            /*
               终局判定 (Phase 6 口径统一): 原来用 isGameOver(), 它只认"将/帅还在
               不在场" —— 将杀/困毙/三次重复/60 回合自然限着都不算终局, 于是自对弈
               几乎永远走不到 done, Q 目标一路用 r(材质) + gamma*maxQ 自举。
               实测 (probe_dqnmcts_aliasing [3], 6 局 x 200 手): **终局信号 0 条**。
            */
            float outcomeMover = 0.0f;
            const bool done = terminalOutcomeOf(chess, outcomeMover);
            const int winner = done ? winnerOfResult(chess.getResult(chess.sideToMove))
                                    : Stone::COLOR_NONE;

            /* 终局常量按黑方视角 (本函数编码与即时奖励都是黑为正) */
            const float terminalReward =
                moverRewardToBlackFrame(outcomeMover, Stone::COLOR_BLACK);

            dqn.perceive(state, oneHotAction, nextState,
                         done ? terminalReward : reward, done);

            if (done) {
                /* 注意: 自检计数**不在这里** —— 一局只记一条, 所以放在循环外
                   (见下面 noteEnd 处的说明与 test_dqnmcts 第 [6](5) 条的断言)。 */
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                    printf("  Episode %4d/%d: %s wins, %d moves, win_rate=%.2f, eps=%.4f\n",
                           ep + 1, episodes,
                           (winner == Stone::COLOR_BLACK) ? "Black" : "Red",
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

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        const bool decidedInLoop = (finalResult != Chess::RESULT_ONGOING);
        if (!decidedInLoop) {
            totalEpisodes++;
            if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                printf("  Episode %4d/%d: Draw, win_rate=%.2f, eps=%.4f\n",
                       ep + 1, episodes, getWinRate(), dqn.exploringRate);
            }
        }
        /*
           自检计数 (界面"模型自检"的终局通道读数) —— **一局恰好一条**, 所以放在
           循环外、并且两个出口都归到这里。
           这个位置是被断言逼出来的: 第一版写在"每走一手"之后, 于是 12 手的一局被
           记成 12 局 (test_dqnmcts 第 [6](5) 条断言 "跑完一局必须恰好记一条" 当场
           抓到 12 != 1)。自检面板上的数字是要给人当决策依据的, 所以它自己必须被钉住。
           `chess.isGameOver()` 另记一次, 用来量"旧口径会不会看见这一局" —— 两个数
           的差就是 2026-09 那次"终局口径统一"修掉了多少被漏掉的终局。
        */
        noteEnd(decidedInLoop ? finalResult : Chess::RESULT_ONGOING,
                chess.isGameOver() != Stone::COLOR_NONE);
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
            if (!step.valid) {
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

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
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
    if (!chosenStep.valid) return;

    int selectedAction = stepToActionIdx(chosenStep);
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    oneHotAction[selectedAction] = 1.0f;
    float reward = computeReward(chosenStep, color);
    /* 视角换算 (2026-09): 走子方视角 -> 黑方视角 (见 docs/agents_design.md §17) */
    reward = moverRewardToBlackFrame(reward, color);

    RL::Tensor nextState(STATE_DIM, 1);
    encodeState(nextState);

    /* 终局判定与上面两处同一口径 (getResult + outcomeForMover), 理由见那里的注释 */
    float outcomeMover = 0.0f;
    const bool done = terminalOutcomeOf(chess, outcomeMover);
    const float terminalReward =
        moverRewardToBlackFrame(outcomeMover, Stone::COLOR_BLACK);

    dqn.perceive(m_cachedState, oneHotAction, nextState,
                 done ? terminalReward : reward, done);

    /* Update the cached state for the next step */
    m_cachedState = nextState;
    m_onlineStepCount++;
}

/* ------------------------------------------------------------------
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  这里只报告**结构 / 口径**类事实。判读写在每一行末尾, 因为面板的读者是看训练
 *  曲线的人, 而这两类数恰恰是"曲线好看但棋力没动"的两个已知原因:
 *
 *    (1) 表示层: 状态 90 维只有"每格有什么子" —— 没有走子方、没有重复进度、
 *        没有无吃子进度、没有被将标记。实测这些量在该编码下**逐字节不可分**
 *        (probe_dqnmcts_aliasing [2])。于是"三次重复/自然限着判和"这类**决定
 *        终局与回报**的规则, 网络读不到。
 *    (2) 动作层: `stepToActionIdx` 把 (棋子 id, 目标格) 哈希进 128 个槽位,
 *        而真实走法空间是 8100。同一个局面里若干个互不相同的走法会共用同一个
 *        Q 槽位 —— 它们拿不到各自的值。
 *
 *  刻意**不**在这里报"棋力": 面板能回答的是"这个模型值不值得继续训", 而不是
 *  "它有多强"。后者只有 bench_anchor 那种带置信区间的锚点对局能回答。
 * ------------------------------------------------------------------ */
std::string DQNMCTSAgent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /* ---- 1. 规模 ---- */
    std::snprintf(buf, sizeof(buf), "状态 %d 维 (10x9 每格一个子力值) | 动作 %d 槽位\n",
                  STATE_DIM, ACTION_DIM);
    out += buf;

    /* ---- 2. 规则上下文通道: 这个编码一个都没有 ---- */
    std::snprintf(buf, sizeof(buf),
                  "规则上下文通道: 0 个 (走子方/重复/无吃子/被将 全不可观测)\n");
    out += buf;

    /* ---- 3. 动作别名 ----
       标准开局那一份是**确定性**的 (与当前棋盘无关), 增量统计则来自真实对局。
       两者都给: 前者说明"这个编码在最常见的局面下就撞", 后者说明训练里撞多少。 */
    int initLegal = 0, initSlots = 0, initWorst = 0;
    int finalLegal = 0, finalSlots = 0, finalWorst = 0;
    {
        /* 只读副本: 绝不能在 GUI 线程碰 this->chess (搜索可能正在用它) */
        Chess probe(chess);
        probe.reset();
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(legal, initLegal, initSlots, initWorst);
        Steps::instance().put(legal);

        Chess probe2(chess);   /* 保持 reset 后的状态; 这里只为不改动 probe */
        std::vector<Step *> l2;
        probe2.sample(probe2.sideToMove, l2);
        aliasOfPosition(l2, finalLegal, finalSlots, finalWorst);
        Steps::instance().put(l2);
    }
    std::snprintf(buf, sizeof(buf),
                  "动作别名(标准开局): %d 个合法着法 -> %d 个 Q 槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法)\n",
                  initLegal, initSlots, initLegal - initSlots, initWorst);
    out += buf;

    if (aliasMoves > 0) {
        const double cleared = (double)aliasClearedMoves / (double)aliasMoves;
        const double perGame = 0.0;   /* 留白: 每局的量由局数换算, 不在这里算 */
        (void)perGame;
        std::snprintf(buf, sizeof(buf),
                      "动作别名(对局累计): %lld 个着法, 平均每次挤掉 %.2f 个"
                      " (最挤槽位 %lld 个着法)\n",
                      aliasMoves, cleared, aliasWorstSlot);
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf), "动作别名(对局累计): 还没有对局数据\n");
        out += buf;
    }

    /* ---- 4. 终局通道: getResult 判出的终局里, isGameOver 漏了多少 ---- */
    const long long total = endCount[END_CAP] + endCount[END_RED_WIN]
                          + endCount[END_BLACK_WIN] + endCount[END_DRAW];
    if (total > 0) {
        const long long decided = total - endCount[END_CAP];
        const long long missed = decided - endSeenByGameOver;
        std::snprintf(buf, sizeof(buf),
                      "终局通道: %lld 局 | 截断 %lld | 红胜 %lld | 黑胜 %lld | 和 %lld\n",
                      total, endCount[END_CAP], endCount[END_RED_WIN],
                      endCount[END_BLACK_WIN], endCount[END_DRAW]);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "  已分出胜负/和棋的 %lld 局里, 旧口径 isGameOver 只看见 %lld 局"
                      " (漏 %lld)\n",
                      decided, endSeenByGameOver, missed > 0 ? missed : 0);
        out += buf;
        if (endCount[END_CAP] * 2 > total) {
            std::snprintf(buf, sizeof(buf),
                          "  截断占比 %.0f%% -> 这一批多数对局没有终局信号, "
                          "Q 目标只有 材质+gamma*maxQ\n",
                          100.0 * (double)endCount[END_CAP] / (double)total);
            out += buf;
        }
    } else {
        std::snprintf(buf, sizeof(buf), "终局通道: 还没有对局数据\n");
        out += buf;
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局\n";
    return out;
}

/* ------------------------------------------------------------------
 *  noteEnd —— 记一局的结束方式 (界面"模型自检"面板的终局通道读数)
 *
 *  为什么要记: 训练损失与自对弈胜率都无法回答"终局信号到底进没进过目标"。
 *  把 `getResult()` 的判定与旧口径 `isGameOver()` 的判定**一起**记下来, 两者的差
 *  就是"有多少局的终局被漏掉了" —— 漏掉的那些局 Q 目标只有 材质 + gamma*maxQ。
 *  （2026-09 已把三处收尾统一到 getResult(); 这个计数是那笔改动的**验收读数**,
 *   同时也是回归指示器: 若 endSeenByGameOver 又追上 decided, 说明有人把口径改回去了。）
 * ------------------------------------------------------------------ */
void DQNMCTSAgent::noteEnd(int result, bool seenByGameOver)
{
    const int code = endCodeOfResult(result);
    if (code >= 0 && code < 4) {
        endCount[code]++;
    }
    if (seenByGameOver) {
        endSeenByGameOver++;
    }
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
    /* 不无条件返回 true: 写盘失败时 UI 会弹假的"保存成功" */
    return weightFileWritten(filepath);
}

bool DQNMCTSAgent::loadModel(const std::string &filepath)
{
    if (!weightFileReadable(filepath)) {
        return false;
    }
    /* 传播内核的真实结果, 理由同 ppomcts_agent.cpp / dqnagent.cpp 的同一处修正 */
    return dqn.load(filepath);
}

/* ------------------------------------------------------------------ */
/*  exploreAndTrain: 走子前"先探索环境 + 在线训练一次" (仿 snakeAI)      */
/* ------------------------------------------------------------------ */
bool DQNMCTSAgent::exploreAndTrain(int color, int rolloutSteps)
{
    if (rolloutSteps <= 0 || batchSize <= 0) {
        return false;
    }

    /* 探索策略与 DQN 一致 (noiseAction), 收集的转移同样进 DQN 回放池 */
    auto pick = [this](const RL::Tensor &state, int /*turn*/) -> int {
        RL::Tensor &q = dqn.noiseAction(state);
        return q.argmax();
    };
    /* 视角换算 (2026-09, 同 trainSelfPlay): rolloutFromCurrent 给的 r 是走子方视角的,
       本 agent 的编码是"黑为正", 用 chosen.id 认出走子方再翻符号。
       见 docs/agents_design.md §17。 */
    auto onTrans = [this](const Step &chosen, int actionIdx,
                          const RL::Tensor &s, const RL::Tensor &ns,
                          float r, bool done) {
        RL::Tensor oneHot(ACTION_DIM, 1);
        oneHot.zero();
        oneHot[actionIdx] = 1.0f;
        dqn.perceive(s, oneHot, ns, moverRewardToBlackFrame(r, chosen), done);
    };

    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick, onTrans);

    bool trained = false;
    if (collected > 0) {
        dqn.learn(maxMemorySize, replaceTargetInterval, batchSize, learningRate);
        trained = true;
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, 训练 1 次(池 "
                    + std::to_string((int)dqn.memories.size()) + ")";
    return trained;
}
