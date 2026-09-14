#include "ppomcts_agent.h"
#include "rl/layer.h"
#include "rl/loss.h"
#include "agentrollout.hpp"
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
                           float cpuct,
                           int expertHidden_,
                           float moeAuxCoef_,
                           bool withGrad_)
    : AgentBase(),
      chess(chess_),
      ppo(STATE_DIM, hiddenDim, ACTION_DIM,
          expertHidden_ > 0 ? expertHidden_ : 64,
          moeAuxCoef_,
          withGrad_),
      gamma(gamma_),
      learningRate(lr),
      c_puct(cpuct),
      expertHidden(expertHidden_ > 0 ? expertHidden_ : 64),
      moeAuxCoef(moeAuxCoef_),
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
 *  encodeStateFor: 局面 -> PLANES x CELLS 的平面编码 (规范视角)
 *
 *    plane 0..13 : 7 类棋子 x {己方, 对方}, 下标 = type*2 + (是己方 ? 0 : 1)
 *    plane 14    : 该格被对方攻击
 *    plane 15    : 该格有己方棋子且被对方攻击
 *
 *  "己方"= color, 并且轮到黑方时把 x 反射成 9-x, 于是同一个网络看到的永远是
 *  "我方的子在自己这边" (见头文件的说明)。平面里的值是 0/1 而不是 ±类型值 ——
 *  平面编码的整个意义就是让"某格上是哪一方的什么子"各占一个独立通道。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::encodeStateFor(int color, RL::Tensor &state)
{
    state.zero();

    const int me   = (color == Stone::COLOR_BLACK) ? Stone::COLOR_BLACK
                                                   : Stone::COLOR_RED;
    const int them = (me == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                              : Stone::COLOR_RED;

    /* ---- 棋子平面 ---- */
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || s->alive == false) continue;
        if (s->type < 0 || s->type >= PIECE_TYPES) continue;

        const int isMine = (s->color == me) ? 0 : 1;
        const int plane  = s->type * 2 + isMine;
        const int cell   = canonicalCell(s->pos, me);
        state[plane * CELLS + cell] = 1.0f;
    }

    /* ---- 威胁平面 ----
       逐格问 Chess::isAttacked(): 它是按棋子类型做几何判定 (而不是为每个敌方棋子
       重新生成一遍走法形状), 本来就是为搜索热路径写的。90 次调用/次编码, 代价实测
       见 docs/agents_design.md 的性能一节。
       先在**绝对坐标**下算一遍, 再统一映射到规范视角 —— 两个坐标系混用是这类
       编码最容易出的错。
    */
    bool attackedAbs[CELLS];
    for (int c = 0; c < CELLS; c++) {
        const Pos p(c / 9, c % 9);
        attackedAbs[c] = chess.isAttacked(p, them);
        if (attackedAbs[c]) {
            state[PLANE_ATTACKED * CELLS + canonicalCell(p, me)] = 1.0f;
        }
    }

    /* ---- 己方子受威胁: "该格上有己方子" 与 "该格被攻击" 的交集 ---- */
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || s->alive == false) continue;
        if (s->color != me) continue;

        if (attackedAbs[cellOf(s->pos)]) {
            state[PLANE_UNDER_ATTACK * CELLS + canonicalCell(s->pos, me)] = 1.0f;
        }
    }
}

void PPOMCTSAgent::encodeState(RL::Tensor &state)
{
    /*
       视角取自棋盘当前的 sideToMove。搜索里走法是真正落在棋盘上的 (moveForward
       会翻转它), 所以每个节点读到的就是该节点的走棋方; rolloutFromCurrent 进来时
       也会先把 sideToMove 设成 color, 探索阶段同样正确。
    */
    encodeStateFor(chess.sideToMove, state);
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
        int aidx = stepToActionIdx(*s, color);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

/* ------------------------------------------------------------------
 *  stepToActionIdx:  无碰撞的 (from, to) 双射
 *
 *      actionIdx = fromCell * CELLS + toCell        (cell = x*9 + y)
 *
 *  8100 = 90x90 覆盖中国象棋所有可能的 (起点, 终点) 组合, 不同走法永不共用槽位。
 *
 *  两个刻意的选择:
 *    * 与棋子 id 无关。旧哈希用的是 (棋子id, 终点), 但 id 会随吃子被回收复用,
 *      于是"同一格走到同一格"在不同局面下可能落到不同槽位 —— 策略学不到稳定的东西。
 *      用格子坐标就没有这个问题。
 *    * 与状态一样按 color 做规范镜像, 所以红黑双方的"同一步棋"共享同一个动作槽位,
 *      与规范视角的棋盘编码保持一致。
 * ------------------------------------------------------------------ */
int PPOMCTSAgent::stepToActionIdx(const Step &s, int color)
{
    const int from = canonicalCell(s.pos, color);
    const int to   = canonicalCell(s.nextPos, color);
    return from * CELLS + to;
}

/* ------------------------------------------------------------------
 *  mirrorPlanes: 逐平面做 y -> 8-y 的格子镜像 (P6, 见头文件)
 *
 *  先把形状与内容整体拷过来 (src/dst 允许是同一块内存也无所谓, 因为所有读都发生在
 *  写之前的那份 src 上), 然后逐平面覆盖。第 4 列 (y=4) 自映射, 其余两两互换,
 *  所以每次写都能从 src 读到正确的源值, 不存在"读到自己刚写的"的顺序依赖。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::mirrorPlanes(const RL::Tensor &src, RL::Tensor &dst)
{
    dst = src;
    const std::size_t n = src.size();
    const std::size_t planes = n / (std::size_t)CELLS;
    for (std::size_t p = 0; p < planes; p++) {
        const std::size_t base = p * (std::size_t)CELLS;
        for (int c = 0; c < CELLS; c++) {
            const int mc = mirrorCell(c);
            if (mc == c) {
                continue;   /* y=4 那一列不动 */
            }
            dst[base + (std::size_t)mc] = src[base + (std::size_t)c];
        }
    }
}

/* ------------------------------------------------------------------
 *  computeReward:  immediate material reward from a move
 * ------------------------------------------------------------------ */
float PPOMCTSAgent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;

    Stone *victim = chess.stones[s.nextId];
    /*
       不检查 victim->alive —— 见 pgagent.cpp 里同一处的说明: 调用方常在
       moveForward() 之后求奖励, 那时被吃子已 alive=false, 加判断会让吃子奖励
       恒为 0 (吃将的 +100 也不可达)。
    */
    if (victim == nullptr) return 0.0f;

    if (victim->type == Stone::TYPE_JIANG) {
        return 100.0f;
    }

    float reward = victim->value * 10.0f;
    /*
       符号约定 (2026-09 修正): 即时奖励是**走子方视角**的 —— 吃掉对方一个子永远是
       收益, 所以这里直接返回 +reward。

       原来写的是 `(color == COLOR_BLACK) ? reward : -reward`, 那是"黑方视角"
       (黑方吃子为正), 于是红方白吃一个黑车会拿到 **-0.5** —— 与同一批经验里的终局
       奖励 (走子方视角的 ±1) 正好相反, 对红方等于在教它"吃子是坏事"。
       Chess::moveForward 的 totalReward 也是黑方视角 (吃红子 +), 两者同源;
       实测探针 build/reward_probe.cpp: 红炮吃黑马 totalReward = -0.30。
       回归钉在 test_match 的 [2.6] 节。
    */
    return reward;
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
 *  visitDistributionSparse / visitDistribution: 根节点访问计数 -> 策略目标 (π ∝ N)
 *
 *  约定与 SACAZAgent::visitDistribution 完全一致 (那边是 AlphaZero 路线的
 *  参考实现), 这样两个 MCTS agent 的策略目标是同一口径, 便于交叉比较。
 *
 *  实现在稀疏版里, 稠密版只是它的展开 —— 免得两处各写一遍归一化再慢慢漂移。
 * ------------------------------------------------------------------ */
bool PPOMCTSAgent::visitDistributionSparse(int rootID,
                                           std::vector<int> &actionIdx,
                                           std::vector<float> &actionProb) const
{
    actionIdx.clear();
    actionProb.clear();
    if (rootID < 0 || (std::size_t)rootID >= nodes.size()) {
        return false;
    }
    const AZNode &root = nodes[(std::size_t)rootID];

    int total = 0;
    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const AZNode &c = nodes[(std::size_t)root.childIDs[k]];
        if (c.parentAction >= 0 && c.parentAction < ACTION_DIM) {
            total += c.visitCount;
        }
    }
    if (total <= 0) {
        return false;
    }

    const float inv = 1.0f / (float)total;
    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const AZNode &c = nodes[(std::size_t)root.childIDs[k]];
        if (c.parentAction >= 0 && c.parentAction < ACTION_DIM && c.visitCount > 0) {
            actionIdx.push_back(c.parentAction);
            actionProb.push_back((float)c.visitCount * inv);
        }
    }
    return !actionIdx.empty();
}

bool PPOMCTSAgent::visitDistribution(int rootID, RL::Tensor &pi) const
{
    pi.zero();
    std::vector<int> idx;
    std::vector<float> prob;
    if (!visitDistributionSparse(rootID, idx, prob)) {
        return false;
    }
    for (std::size_t k = 0; k < idx.size(); k++) {
        pi[(std::size_t)idx[k]] = prob[k];
    }
    return true;
}

/* ------------------------------------------------------------------
 *  commitEpisode: 一局 -> 回放池 -> 批量学习 (P3 + P4, 见 ppomcts_agent.h)
 *
 *  为什么要"进池时转稀疏": 回放池是长期占内存的那个 (20000 条 x 1440 状态 ≈ 115 MB),
 *  策略目标如果按 8100 维稠密存就是每步 32 KB —— 20000 条直接 640 MB。存非零项
 *  (~40 项) 只有 320 B, 差 100 倍, 而且不丢 P5 那套软目标的信息。
 *
 *  mirrorAugment (P6): 每条样本再存一份左右镜像 —— 价值目标不变 (局面对称, 胜负
 *  关系当然不变), 动作下标整体镜像 (双射, 所以同一份目标分布里的不同动作镜像后
 *  仍然互不相同, 不会把概率叠到同一个槽位上)。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::commitEpisode(const std::vector<RL::Step> &trajectory,
                                 float finalOutcome)
{
    if (trajectory.empty()) {
        return;
    }
    const std::vector<float> returns = ppo.discountedReturns(trajectory, finalOutcome);

    std::vector<int> idx;
    std::vector<float> prob;
    std::vector<int> mirrorIdx;
    RL::Tensor mirrorState;
    for (std::size_t t = 0; t < trajectory.size(); t++) {
        const RL::Tensor &a = trajectory[t].action;
        idx.clear();
        prob.clear();
        const std::size_t n = (a.size() < (std::size_t)ACTION_DIM) ? a.size()
                                                                   : (std::size_t)ACTION_DIM;
        for (std::size_t i = 0; i < n; i++) {
            if (a[i] > 0.0f) {
                idx.push_back((int)i);
                prob.push_back(a[i]);
            }
        }
        if (idx.empty()) {
            continue;   /* 没有目标的样本不推进去 (addReplay 也会挡掉) */
        }
        ppo.addReplay(trajectory[t].state, idx, prob, returns[t]);

        if (mirrorAugment) {
            mirrorIdx.resize(idx.size());
            for (std::size_t i = 0; i < idx.size(); i++) {
                mirrorIdx[i] = mirrorActionIdx(idx[i]);
            }
            mirrorPlanes(trajectory[t].state, mirrorState);
            ppo.addReplay(mirrorState, mirrorIdx, prob, returns[t]);
        }
    }

    /* 池子够大就开始批量学习: 每次采样 batchSize 条、过 epochs 遍, 一次优化器更新 */
    if (replayBatchSize > 0 && ppo.replaySize() >= (std::size_t)replayBatchSize) {
        ppo.learnFromReplay((std::size_t)replayBatchSize, replayEpochs, learningRate);
    }
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

    /* ---- Create root node ----
       根节点的策略不再在这里预先求: 它的子节点先验会在第一次展开根节点时算出来
       (那时根节点就是"父节点"), 而原来那次 rootValue / rootPolicy 从头到尾没被用过 ——
       纯浪费一次 actor + 一次 critic 前向。 */
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

    /*
       搜索期间 encodeState 要靠 chess.sideToMove 决定规范视角, 而 moveForward /
       moveBack 会翻转它。调用方传进来的 color 不一定等于棋盘当前的 sideToMove
       (测试里就是 chess.reset() 之后直接 selectMove(BLACK)), 所以先对齐, 结束时
       原样恢复 —— 与 agentrollout.hpp 里同样的理由, 对调用方的棋盘零副作用。
    */
    const int savedSideToMove = chess.sideToMove;
    chess.sideToMove = color;

    /* 每模拟复用同一对缓存, 避免在循环里反复分配 (STATE_DIM 已经上千维) */
    RL::Tensor parentState(STATE_DIM, 1);
    RL::Tensor leafState(STATE_DIM, 1);

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

            /*
               先把**父节点**的策略求出来 —— 边 (parent -> child) 的先验是
                   P(s_parent, a) = pi_theta(s_parent)[a]
               必须用父节点的网络输出, 而且必须与 chosenAction 处在同一个规范视角。
               原来的代码拿的是子节点的策略 (`childPolicy[chosenAction]`): 那既用错了
               网络 (子节点的策略描述的是下一手走棋方的选择), 也用错了视角。在旧的
               "绝对坐标 + 哈希"编码下两者恰好吃同一个坐标帧, 错误只表现为先验质量差;
               换成规范视角后父/子两个帧会直接错开 (父按红方、子按黑方), 先验会变成
               完全无关的数。此刻棋子还没落, 棋盘正是父节点局面。
            */
            encodeState(parentState);
            RL::Tensor &parentPolicy = ppo.action(parentState);
            float prior = parentPolicy[chosenAction];
            if (prior < 1e-9f) prior = 1e-9f; /* avoid zero prior */

            /* Execute the move */
            chess.moveForward(&chosenStep, dummyReward);

            /* Create child node */
            int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                ? Stone::COLOR_BLACK
                                : Stone::COLOR_RED;

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
         *
         * 这里只要**价值**: 叶子自己的策略先验会在它被展开的那次模拟里用到 (那时它
         * 扮演的是"父节点"), 所以不必现在再算一遍。原来的代码每个模拟把同一个局面
         * 算了两遍 (子节点策略 + 叶子估值), 其中策略那一次还是被用错帧的。
         */
        encodeState(leafState);
        double reward = (double)ppo.value(leafState);

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

    /* 恢复调用方的棋盘视角 (见本函数开头保存 savedSideToMove 处的说明) */
    chess.sideToMove = savedSideToMove;

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

            /*
               规范视角靠 chess.sideToMove 决定, 而它不一定等于 currentColor:
               Chess::reset() 把 sideToMove 置成 RED, 本函数却每局都从 BLACK 开始
               (自对弈的既有约定)。不对齐的话每局第一步都会用**对手**的视角编码。
               之后 moveForward 会按落子方的颜色重设 sideToMove, 两者自然一致。
            */
            chess.sideToMove = currentColor;

            /* Encode current state */
            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* Run MCTS to get improved policy */
            nodes.clear();

            /* 根节点的策略不在这里预先求 —— 它的子节点先验会在第一次展开根节点时
               算出来。原来这里的 rootPolicy / rootValue 从头到尾没被使用过。 */
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
                /*
                   没有合法走法的是 currentColor, 赢家是对手 —— 而"最后一手"正是对手
                   走的, 所以最后一步走子方就是赢家, 终局值 +1。
                   (原来这里也是 (winner == BLACK) ? 1 : -1 的黑方视角, 与下面 else
                    分支是同一个错, 一并修掉。)
                */
                const float outcomeForLastMover = 1.0f;

                /* Train PPO on trajectory with terminal outcome */
                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover);
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

            /* 每个模拟复用同一对缓存, 避免在循环里反复分配 (STATE_DIM 已经上千维) */
            RL::Tensor parentState(STATE_DIM, 1);
            RL::Tensor leafState(STATE_DIM, 1);

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

                    /* 边的先验来自**父节点**的策略, 且必须与 chosenAction 处在同一个
                       规范视角 —— 详细理由见 selectMove 里同一处的长注释。此刻棋子
                       还没落, 棋盘正是父节点局面。 */
                    encodeState(parentState);
                    RL::Tensor &parentPolicy = ppo.action(parentState);
                    float prior = parentPolicy[chosenAction];
                    if (prior < 1e-9f) prior = 1e-9f;

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK
                                        : Stone::COLOR_RED;

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
                /* 这里只要**价值**: 叶子自己的策略先验会在它被展开的那次模拟里用到
                   (那时它扮演"父节点"), 所以不必现在再算一遍 —— 原来的代码把同一个
                   局面算了两遍, 其中策略那一次还是被用错帧的。 */
                encodeState(leafState);
                double reward = (double)ppo.value(leafState);

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
                    /*
                       终局值必须按**最后一步走子方**的视角给 —— 此刻走子方就是
                       currentColor (还没翻转)。

                       原来传的是 (gameResult == BLACK) ? 1 : -1, 也就是**黑方视角**;
                       而每一步的即时奖励 computeReward() 是走子方视角, 状态编码又是
                       规范视角 (价值头输出的就是走子方的价值)。三者不一致的后果:
                       红方走的每一步拿到的终局分量符号都是反的 —— 红方赢的棋, 对红方
                       反而成了"在输"。另外原来那条 emplace_back 把"奖励"直接写成了
                       黑方视角的终局值, 等于把终局信号记了两遍 (一遍当奖励、一遍当
                       finalOutcome)。
                    */
                    const float outcomeForLastMover =
                        (gameResult == currentColor) ? 1.0f : -1.0f;

                    /* Store final transition:
                       即时奖励与其它步同一口径, 终局由 finalOutcome 统一加。
                       策略目标用根节点的访问分布 (见 visitDistribution 的说明) */
                    RL::Tensor policyTarget(ACTION_DIM, 1);
                    if (!visitDistribution(rootID, policyTarget)) {
                        policyTarget.zero();
                        if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                    }
                    trajectory.emplace_back(state, policyTarget,
                                            computeReward(chosenStep, currentColor));

                    /* Train PPO on complete trajectory */
                    if (!trajectory.empty()) {
                        commitEpisode(trajectory, outcomeForLastMover);
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

                /* Store transition: 策略目标 = 根节点的访问分布, 不再是"被采样那一步"的 one-hot */
                RL::Tensor policyTarget(ACTION_DIM, 1);
                if (!visitDistribution(rootID, policyTarget)) {
                    policyTarget.zero();
                    if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                }
                float reward = computeReward(chosenStep, currentColor);
                trajectory.emplace_back(state, policyTarget, reward);

                /* Switch side */
                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK
                                   : Stone::COLOR_RED;
            } else {
                /* No legal moves - game over */
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK
                                 : Stone::COLOR_RED;
                /*
                   没有合法走法的是 currentColor (被将死/困毙), 赢家是它的**对手** ——
                   而"最后一手"正是对手走的, 所以最后一步走子方就是赢家, 终局值 +1。
                   原来传的是黑方视角的 (winner == BLACK) ? 1 : -1, 同样是错帧。
                */
                const float outcomeForLastMover = 1.0f;

                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover);
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
        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            if (!trajectory.empty()) {
                commitEpisode(trajectory, 0.0f);
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
            /* 规范视角靠 chess.sideToMove, 而 Chess::reset() 把它置成 RED、本函数却
               从 BLACK 开始走 —— 不对齐的话第一步会用**对手**的视角编码 (同 trainSelfPlay) */
            chess.sideToMove = currentColor;

            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* ---- MCTS: Selection + Expansion + Backprop ---- */
            nodes.clear();

            /* 根节点的策略在展开根节点时才算 (原来这里的 rootPolicy/rootValue 未被使用) */
            std::vector<Step*> rootSteps;
            std::vector<int> rootActionIndices;
            RL::Tensor rootActionMask(ACTION_DIM, 1);
            rootActionMask.zero();
            getLegalActions(currentColor, rootSteps,
                            rootActionIndices, rootActionMask);

            if (rootSteps.empty()) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                /* 最后一手是赢家走的 -> 最后一步走子方 = 赢家 -> 终局值 +1
                   (同 trainSelfPlay 那两处修正, 原来这里是黑方视角) */
                const float outcomeForLastMover = 1.0f;
                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover);
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

            /* 每个模拟复用同一对缓存, 避免在循环里反复分配 (STATE_DIM 已经上千维) */
            RL::Tensor parentState(STATE_DIM, 1);
            RL::Tensor leafState(STATE_DIM, 1);

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

                    /* 边的先验来自**父节点**的策略, 同一规范视角 (理由见 selectMove) */
                    encodeState(parentState);
                    RL::Tensor &parentPolicy = ppo.action(parentState);
                    float prior = parentPolicy[chosenAction];
                    if (prior < 1e-9f) prior = 1e-9f;

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK : Stone::COLOR_RED;

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

                /* 只要价值: 叶子的策略先验在它被展开那次模拟里才用到 (同 trainSelfPlay) */
                encodeState(leafState);
                double reward = (double)ppo.value(leafState);

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
                    /* 终局值按最后一步走子方 (= currentColor, 还没翻转) 的视角 ——
                       与 trainSelfPlay 同一处修正, 理由见那里的长注释 */
                    const float outcomeForLastMover =
                        (gameResult == currentColor) ? 1.0f : -1.0f;
                    /* 策略目标 = 根节点的访问分布 (同 trainSelfPlay) */
                    RL::Tensor policyTarget(ACTION_DIM, 1);
                    if (!visitDistribution(rootID, policyTarget)) {
                        policyTarget.zero();
                        if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                    }
                    trajectory.emplace_back(state, policyTarget,
                                            computeReward(chosenStep, currentColor));
                    if (!trajectory.empty()) {
                        commitEpisode(trajectory, outcomeForLastMover);
                    }
                    totalEpisodes++;
                    if (gameResult == Stone::COLOR_BLACK) totalWins[1]++;
                    if (gameResult == Stone::COLOR_RED) totalWins[0]++;
                    Steps::instance().put(rootSteps);
                    break;
                }

                RL::Tensor policyTarget(ACTION_DIM, 1);
                if (!visitDistribution(rootID, policyTarget)) {
                    policyTarget.zero();
                    if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                }
                float reward = computeReward(chosenStep, currentColor);
                trajectory.emplace_back(state, policyTarget, reward);

                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            } else {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                /* 最后一手是赢家走的 -> 最后一步走子方 = 赢家 -> 终局值 +1
                   (同 trainSelfPlay 那处修正) */
                const float outcomeForLastMover = 1.0f;
                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                Steps::instance().put(rootSteps);
                break;
            }

            Steps::instance().put(rootSteps);
        }

        /*
       只有在"循环跑到步数上限、并且没有分出胜负"时才计为和棋。
       这里原来只判断 isGameOver() == COLOR_NONE, 而"轮到走的一方没有合法走法"
       (将杀/困毙) 并不会让将帅消失 —— 于是那条分支已经计数过一次之后, 这里会再
       计一次局数。改用 getResult() 判断是否已分胜负。
    */
        const int finalResult = chess.getResult(chess.sideToMove);
        if (finalResult == Chess::RESULT_ONGOING || finalResult == Chess::RESULT_DRAW) {
            if (!trajectory.empty()) {
                commitEpisode(trajectory, 0.0f);
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
    /* 不无条件返回 true: 两个文件都写成功才算成功 */
    return weightFileWritten(actorPath) && weightFileWritten(criticPath);
}

bool PPOMCTSAgent::loadModel(const std::string &actorPath,
                             const std::string &criticPath)
{
    if (!weightFileReadable(actorPath) || !weightFileReadable(criticPath)) {
        return false;
    }
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
    int aidx = stepToActionIdx(s, color);
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



/* ------------------------------------------------------------------ */
/*  exploreAndTrain: 走子前"先探索环境 + 在线训练一次" (仿 snakeAI)      */
/* ------------------------------------------------------------------ */
bool PPOMCTSAgent::exploreAndTrain(int color, int rolloutSteps)
{
    if (rolloutSteps <= 0) {
        return false;
    }

    /*
       探索策略: 从 PPO actor 的策略分布里采样 (对应 snakeAI 的 gumbelMax/采样)。

       **必须按合法走法掩码**: 策略头覆盖全部 8100 个 (起点, 终点) 组合, 而任一局面
       只有几十个合法走法。rolloutFromCurrent 拿到索引后是在"合法走法索引表"里查的,
       查不到就退化成"走第一个合法走法"。不掩码地采样, 命中合法走法的概率只有百分之
       几, 探索实际会变成恒定走 legal[0] —— 策略永远得不到按自己意愿走子的机会, 也就
       学不到东西。(旧的 128 槽哈希编码下, 碰撞让这个漏洞大部分时候看不出来。)
    */
    auto pick = [this](const RL::Tensor &state, int turn) -> int {
        /* 拷贝一份再改: ppo.action() 返回的是网络内部的输出张量, 不能就地清零 */
        RL::Tensor policy = ppo.action(state);

        RL::Tensor mask(ACTION_DIM, 1);
        mask.zero();
        std::vector<Step*> legal;
        std::vector<int> actionIndices;
        getLegalActions(turn, legal, actionIndices, mask);
        Steps::instance().put(legal);

        float sum = 0.0f;
        for (std::size_t i = 0; i < policy.size(); i++) {
            if (mask[i] > 0.0f) {
                sum += policy[i];
            } else {
                policy[i] = 0.0f;
            }
        }
        if (sum <= 1e-12f) {
            /* 策略把全部合法走法都压到了 0 (未归一化的 logits 极端情形):
               退化成"合法走法上的均匀分布", 而不是把全 0 分布交给采样器 */
            for (std::size_t i = 0; i < policy.size(); i++) {
                if (mask[i] > 0.0f) {
                    policy[i] = 1.0f;
                }
            }
        } else {
            for (std::size_t i = 0; i < policy.size(); i++) {
                policy[i] /= sum;
            }
        }

        return RL::Random::categorical(policy);
    };

    std::vector<RL::Step> traj;
    traj.reserve((std::size_t)rolloutSteps);
    /*
       只记"轨迹是否真的走到了终局"以及那一手的奖励。终局值必须来自终局本身,
       不能拿任意一手(可能是中局)的即时奖励冒充 —— 见下面 learnSelfPlay 调用处的说明。
    */
    bool rolloutEnded = false;
    float terminalReward = 0.0f;
    auto onTrans = [&traj, &rolloutEnded, &terminalReward](
                       const Step &/*chosen*/, int actionIdx,
                       const RL::Tensor &s, const RL::Tensor &/*ns*/,
                       float r, bool done) {
        RL::Tensor oneHot(ACTION_DIM, 1);
        oneHot.zero();
        oneHot[actionIdx] = 1.0f;
        traj.emplace_back(s, oneHot, r);
        if (done) {
            /* agentrollout.hpp 在 done 时把 r 换成了 (gameResult == turn) ? 1 : -1,
               也就是**走子方视角**的终局结果 —— 正是 learnSelfPlay 需要的口径 */
            rolloutEnded = true;
            terminalReward = r;
        }
    };

    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick, onTrans);

    bool trained = false;
    if (!traj.empty()) {
        /*
           finalOutcome 的口径是"最后一步走子方视角" (见 rl/ppo.h)。

           原来这里无条件传 lastReward —— 那是**最后一手的即时奖励**(吃子分/0),
           不是终局结果。探索大多在终局前就截断了, 于是价值目标变成"这盘棋的最终
           回报等于最后一步吃了个马", 基本是噪声。

           走到终局时: 用 agentrollout 给的那个走子方视角的 ±1 ✓
           没走到终局时: 传 0。这是一个**截断**更新 (等于假设"此后双方均势"),
           有偏但方向中性; 真正正确的做法是自举 (用 V(最终状态) 当终局值),
           learnSelfPlay 目前不支持, 列为后续项。
        */
        const float finalOutcome = rolloutEnded ? terminalReward : 0.0f;
        ppo.learnSelfPlay(traj, finalOutcome, learningRate);
        trained = true;
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, PPO 更新 1 次";
    return trained;
}
