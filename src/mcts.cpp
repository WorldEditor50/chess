#include "mcts.h"
#include <cmath>
#include <cstdio>   /* selfCheckReport 的 snprintf */
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include <string>

/* ============================================================
 * MCTSNode implementation
 * ============================================================ */

MCTSNode::MCTSNode()
    : visitCount(0), totalReward(0.0), parentID(-1), currentColor(Stone::COLOR_NONE)
{
}

MCTSNode::MCTSNode(int parentID_, const Step &step_, int color_)
    : visitCount(0), totalReward(0.0), parentID(parentID_), step(step_), currentColor(color_)
{
}

double MCTSNode::getUCB1(double totalParentVisits, double C) const
{
    /* UCB1 = -W/N + C * sqrt(ln(N_parent) / N)
     *
     * If visitCount is 0, return infinity to encourage exploring
     * unvisited nodes first.
     *
     * 符号 (2026-09 修正): `totalReward` 由 backpropagate 沿父链**逐层翻号**写入,
     * 所以它记的是**本节点走棋方**视角的回报 (simulateRandomPlay 也是按
     * nodes[nodeID].currentColor 返回的)。子节点的走棋方是父节点的对手, 因此父节点
     * 比较时必须取**负号**。漏掉它, UCB1 会在最大化对手的收益 —— 表现是搜索专挑
     * 对自己最差的着法, 而 rollout/估值越准错得越狠。
     * 与 PPOMCTSAgent::getPUCT / SACAZAgent::getPUCT 是同一处口径。
     */
    if (visitCount == 0) {
        return std::numeric_limits<double>::max();
    }
    double exploitation = -totalReward / (double)visitCount;
    double exploration = C * std::sqrt(std::log(totalParentVisits) / (double)visitCount);
    return exploitation + exploration;
}

/* ============================================================
 * MCTS implementation
 * ============================================================ */

MCTS::MCTS(Chess &chess_, double explorationConstant)
    : AgentBase(), chess(chess_), C(explorationConstant)
{
    /*
       Seed the RNG for random move selection during rollout.

       ---- 只播一次 (2026-09) ----
       原来是**每次构造**都 `std::srand(time(nullptr))`。MCTS 对象在 GUI 里是"每一步
       现场构造一个" (见 ChessBoard::aiThinkRaw), 于是整局棋的随机流每一步都被重新
       播种到同一秒的种子上 —— 回放/复现没有意义; 更糟的是自检 (getAgentSelfCheck
       会造一个临时 MCTS 来验证"返回值合法") 会在 GUI 线程重新播种, 而搜索线程可能
       正在另一个线程里用 `std::rand()` —— `std::srand` 不是线程安全的。
       现在用一次性标志: 进程内第一次构造时播种 (行为与以前的第一步相同), 之后的
       构造不再动全局随机流。
    */
    static bool s_seeded = false;
    if (!s_seeded) {
        std::srand((unsigned int)std::time(nullptr));
        s_seeded = true;
    }
}

/* AgentBase interface */
Step MCTS::getBestMove(int color)
{
    return findBestMove(color, 800);
}

std::string MCTS::getName() const
{
    return "MCTS (C=" + std::to_string(C) + ")";
}

void MCTS::clearTree()
{
    nodes.clear();
}

/* ------------------------------------------------------------
 * selectPromisingNode
 *
 * Traverses the tree from rootID downwards using the UCB1 formula
 * to select the most promising child at each level. Continues as
 * long as the current node is fully expanded (no untriedMoves) and
 * has at least one child.
 *
 * Each chosen move is executed on the board, keeping the board
 * state synchronized with the current node in the tree.
 *
 * Returns the nodeID of the leaf where selection stopped.
 * Fills `path` with all node IDs from rootID to the leaf (inclusive).
 * ---------------------------------------------------------- */
int MCTS::selectPromisingNode(int rootID, std::vector<int> &path)
{
    int nodeID = rootID;

    while (nodes[nodeID].isFullyExpanded()
           && !nodes[nodeID].childIDs.empty()) {

        double parentVisits = (double)nodes[nodeID].visitCount;
        int bestChild = -1;
        double bestUCB = -std::numeric_limits<double>::max();

        for (int childID : nodes[nodeID].childIDs) {
            double ucb = nodes[childID].getUCB1(parentVisits, C);
            if (ucb > bestUCB) {
                bestUCB = ucb;
                bestChild = childID;
            }
        }

        if (bestChild < 0) {
            break;  /* no reachable child (should not happen) */
        }

        /* Execute the move leading to this child */
        double dummyReward = 0.0;
        const Step &move = nodes[bestChild].step;
        chess.moveForward(&move, dummyReward);

        nodeID = bestChild;
        path.push_back(nodeID);
    }

    return nodeID;
}

/* ------------------------------------------------------------
 * expandNode
 *
 * If the given node has untried moves, pick one uniformly at
 * random, execute it on the board, create a new child node,
 * pre-generate all legal moves for the opponent (stored as
 * untriedMoves in the child), and register the child in the tree.
 *
 * Returns the ID of the newly expanded child node.
 * ---------------------------------------------------------- */
int MCTS::expandNode(int nodeID, std::vector<int> &path)
{
    double dummyReward = 0.0;

    /* Pick a random untried move */
    int moveIdx = std::rand() % (int)nodes[nodeID].untriedMoves.size();
    Step chosenMove = nodes[nodeID].untriedMoves[moveIdx];

    /* Remove it from the untried list */
    nodes[nodeID].untriedMoves.erase(
        nodes[nodeID].untriedMoves.begin() + moveIdx);

    /* Execute the move */
    chess.moveForward(&chosenMove, dummyReward);

    /* Determine the color to move at the new position */
    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                        ? Stone::COLOR_BLACK
                        : Stone::COLOR_RED;

    /* Create the new child node */
    MCTSNode newNode(nodeID, chosenMove, nextColor);

    /* Pre-compute all legal moves for the child so they
     * can be expanded later */
    std::vector<Step*> newMoves;
    chess.sample(nextColor, newMoves);
    for (Step *s : newMoves) {
        newNode.untriedMoves.push_back(*s);
    }
    Steps::instance().put(newMoves);

    /* Register the child in the tree */
    nodes.push_back(newNode);
    int newNodeID = (int)nodes.size() - 1;
    nodes[nodeID].childIDs.push_back(newNodeID);

    /* Update path to include the new node */
    nodeID = newNodeID;
    path.push_back(nodeID);

    return nodeID;
}

/* ------------------------------------------------------------
 * backpropagate
 *
 * Walk back up the path, updating visitCount and totalReward
 * for every node. The reward sign is flipped at each level
 * because the perspective alternates between the two players.
 * ---------------------------------------------------------- */
void MCTS::backpropagate(const std::vector<int> &path, double reward)
{
    for (int i = (int)path.size() - 1; i >= 0; i--) {
        nodes[path[i]].visitCount++;
        nodes[path[i]].totalReward += reward;
        reward = -reward;   /* flip for the opponent */
    }
}

/* ------------------------------------------------------------
 * simulateRandomPlay
 *
 * Play random legal moves for both sides starting from the
 * current board state until the game ends (checkmate or
 * stalemate). Returns +1 if `color` wins, -1 if `color`
 * loses, and 0 if it's a draw.
 *
 * All moves played during the simulation are undone before
 * returning, leaving the board in its original state.
 * ---------------------------------------------------------- */
double MCTS::simulateRandomPlay(int color)
{
    std::vector<Step> simSteps;
    double dummy = 0.0;
    int currColor = color;

    /*
       回放步数上限: 原来是 while(true), 只要随机走子一直没吃到将/帅就会无限
       循环下去 (没有重复局面判和、也没有自然限着)。到上限按和棋处理。
    */
    const int maxPlies = 200;

    while ((int)simSteps.size() < maxPlies) {
        /* Check for terminal state (checkmate) */
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            /* gameResult is the winning color */
            for (auto it = simSteps.rbegin();
                 it != simSteps.rend(); ++it) {
                chess.moveBack(&(*it), dummy);
            }
            return (gameResult == color) ? 1.0 : -1.0;
        }

        /* Generate all legal moves for the current player */
        std::vector<Step*> moves;
        chess.sample(currColor, moves);

        if (moves.empty()) {
            /* Current player has no legal moves (stalemate / get
             * mated) -> they lose */
            Steps::instance().put(moves);
            for (auto it = simSteps.rbegin();
                 it != simSteps.rend(); ++it) {
                chess.moveBack(&(*it), dummy);
            }
            return (currColor == color) ? -1.0 : 1.0;
        }

        /* Pick a move uniformly at random */
        int idx = std::rand() % (int)moves.size();
        Step *chosen = moves[idx];
        chess.moveForward(chosen, dummy);
        simSteps.push_back(*chosen);
        Steps::instance().put(moves);

        /* Switch to the other side */
        currColor = (currColor == Stone::COLOR_RED)
                        ? Stone::COLOR_BLACK
                        : Stone::COLOR_RED;
    }

    /* 到达步数上限仍未终局 -> 按和棋收尾, 并回退全部走法 */
    for (auto it = simSteps.rbegin(); it != simSteps.rend(); ++it) {
        chess.moveBack(&(*it), dummy);
    }
    return 0.0;
}

/* ------------------------------------------------------------
 * findBestMove
 *
 * Main MCTS search entry point. Builds the root node from the
 * current board state, then performs `iterations` MCTS iterations.
 *
 * Each iteration:
 *   1. SELECTION   - traverse tree using UCB1 (selectPromisingNode)
 *   2. EXPANSION   - add a new child node if untried moves exist
 *                    (expandNode)
 *   3. SIMULATION  - random rollout from leaf until game ends
 *                    (simulateRandomPlay)
 *   4. BACKPROPAGATION - propagate result up the path
 *                        (backpropagate)
 *
 * After all iterations, returns the child of the root with the
 * highest visit count.
 * ---------------------------------------------------------- */
Step MCTS::findBestMove(int color, int iterations)
{
    nodes.clear();

    /* --------------------------------------------------------
     * Build the root node: generate all legal moves for the
     * current color at the current board state.
     * -------------------------------------------------------- */
    MCTSNode root;
    root.currentColor = color;
    root.parentID = -1;

    std::vector<Step*> rootMoves;
    chess.sample(color, rootMoves);
    for (Step *s : rootMoves) {
        root.untriedMoves.push_back(*s);
    }
    Steps::instance().put(rootMoves);

    nodes.push_back(root);
    int rootID = 0;

    /* --------------------------------------------------------
     * Main MCTS loop
     * -------------------------------------------------------- */
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<int> path;
        path.push_back(rootID);

        /* Phase 1 + 2: SELECTION & EXPANSION
         *
         * First, traverse the tree using UCB1 to find a leaf.
         * The leaf will be either a node with untried moves or
         * a terminal node. As we descend, moves are executed on
         * the board.
         */
        int nodeID = selectPromisingNode(rootID, path);

        /* If the selected node has untried moves, expand it */
        if (!nodes[nodeID].untriedMoves.empty()) {
            nodeID = expandNode(nodeID, path);
        }

        /* Phase 3: SIMULATION (ROLLOUT) */
        double reward = simulateRandomPlay(
                              nodes[nodeID].currentColor);

        /* Phase 4: BACKPROPAGATION */
        backpropagate(path, reward);

        /* ==============================================
         * Undo all moves that were played during this
         * iteration (selection + expansion), restoring
         * the original board state.
         * ============================================== */
        for (int i = (int)path.size() - 1; i > 0; i--) {
            double dummyReward = 0.0;
            const Step &s = nodes[path[i]].step;
            chess.moveBack(&s, dummyReward);
        }
    }

    /* --------------------------------------------------------
     * Choose the best move: the child of the root with the
     * highest visit count.
     * -------------------------------------------------------- */
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

    /*
       根有合法走法、但一次扩展都没发生 (iterations <= 0, 或循环里没能走到 EXPANSION):
       以前这里直接返回 Step() (valid=false), 调用方会把它读成"真无棋可走" -> 判负,
       而棋盘上明明还有棋可下 (与 ABAgent 那个"全负时不返回走法"是同一类错误)。
       兜底取根节点未展开列表的第一手 —— 列表就是 rootMoves 的副本, 一定在合法集里。
    */
    if (!nodes[rootID].untriedMoves.empty()) {
        return nodes[rootID].untriedMoves.front();
    }

    /* 确实没有合法走法 (将杀 / 困毙) - return empty Step */
    return Step();
}

/* ============================================================
 *  自检报告
 * ============================================================ */

/*
 * stepInMoveList: s 是否就是 legal 里的某一步 (决策合法性自检用)。
 *
 * 比对 (起点棋子 id, 起点格, 终点格) 三项, 不比对整个对象: 走法生成器给同一个走法
 * 填的 reward 与回填/复制来的不一定同源, 拿整对象比会出现"明明是这一步却说不相等"。
 * `!s.valid` 直接算不在集合里 —— 树里存的 Step 是从生成器拷来的 (valid=true), 而
 * 兜底失败时返回的是默认构造的占位对象 (valid=false), 只有 valid 位能区分这两者
 * (stone.h 里 valid 字段的存在理由就是这个)。
 */
static bool stepInMoveList(const Step &s, const std::vector<Step *> &legal)
{
    if (!s.valid) {
        return false;
    }
    for (const Step *l : legal) {
        if (l == nullptr || !l->valid) {
            continue;
        }
        if (l->id == s.id && l->nextId == s.nextId
            && l->pos.x == s.pos.x && l->pos.y == s.pos.y
            && l->nextPos.x == s.nextPos.x && l->nextPos.y == s.nextPos.y) {
            return true;
        }
    }
    return false;
}

/* ------------------------------------------------------------------
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  报的全是**结构 / 口径**事实, 一条棋力都没有。MCTS 这一路的读数和 AB 不同,
 *  因为它的"评估"不是函数而是**随机对局的经验平均**, 面板上要能看出这件事:
 *
 *    (1) 它没有可训练参数 (树每次 findBestMove 都 clear() 重建, 权重为 0 个),
 *        所以损失曲线空着是**正确**状态。
 *    (2) 它的叶子信号是 simulateRandomPlay() 的终局结果 (±1/0) —— 方差极高:
 *        同一个局面跑两次随机对局很可能得到相反的符号 (见 simulateRandomPlay 里
 *        200 手上限那条注释: 大部分随机对局其实是"走到上限按和棋收尾")。于是
 *        "800 次模拟"分摊到 40 多个根走法上, 每个走法只有十几次采样 —— 面板给出
 *        这个摊薄比, 免得把 ±1 的平均值当成可靠的局面评估。
 *    (3) **决策合法性自检** (回归指示器): findBestMove 在"根节点有合法走法, 但
 *        一次扩展都没发生"时 (iterations <= 0, 或者循环里没能走到 EXPANSION) 以前
 *        直接返回 Step() (valid=false), 调用方把它读成"真无棋可走" -> 判负, 与
 *        ABAgent 那个"全负时不返回走法"是同一类错误。mcts.cpp 的 findBestMove 现在
 *        兜底取根节点未展开列表的第一手 (那列表就是 rootMoves 的副本, 必在合法集
 *        里); 这一行就是那笔兜底的验收读数。
 *    (4) 标准开局的合法走法数 = 那 800 次模拟实际摊到的分支因子 (与训练进度无关,
 *        一打开面板就能看; 改走法生成会让它动, 当尺子用)。
 *
 *  **只读**: 全程用 chess 的**副本** + 一个临时局部 agent, 不碰 this->chess (GUI
 *  线程调用它时搜索线程可能正在用同一个棋盘), 不改任何成员 (本函数是 const)。
 *  临时 agent 是刻意为之, 不是绕路: findBestMove()/getBestMove() 都不是 const
 *  (要写 nodes), 与其 const_cast 掉 this 去跑一次真搜索 (那会**清掉**主 agent 正在
 *  累计的树), 不如在副本上另起 MCTS(probe, 1.414)。构造它**不会**动全局随机流 ——
 *  构造函数里的 std::srand 是一次性的 (见那里的注释: 自检会在 GUI 线程反复构造它,
 *  而 std::srand 不是线程安全的, 每次构造都播种会踩到搜索线程的 rand())。
 *  成本上限: 1 次模拟 —— 一个孩子 + 一局最多 200 手的随机回放 (上限由
 *  simulateRandomPlay 的 maxPlies 钉住, 所以这一步是**有界**的, 只是界比 AB 那一路
 *  大; 面板是**每一手棋**刷新一次 (见 mainwindow 的 updateSelfCheckPanel 调用点),
 *  不在搜索热路径上调它, 所以这点开销可以接受)。
 * ------------------------------------------------------------------ */
std::string MCTS::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /* ---- 1. 搜索配置 (C / 树规模) 与"无学习参数" ---- */
    std::snprintf(buf, sizeof(buf),
                  "搜索配置: UCB1 探索常数 C = %.3f | 当前树节点 %d 个\n",
                  C, (int)nodes.size());
    out += buf;
    /*
       树规模是**只读快照**: 主 agent 的树在每次 findBestMove 里 clear() 重建, 搜索
       线程若正在跑, 这个数此刻正在变。面板只把它当量级读 (0 = 还没搜过 / 已清空)。
    */
    std::snprintf(buf, sizeof(buf),
                  "可学习权重: 0 个 (纯搜索 agent) -> getLastTrainLoss() 恒为 NaN, "
                  "训练损失曲线对它永远是空的\n");
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "叶子信号: 随机走子到终局的胜负 (±1/0), 天生高方差 —— "
                  "单次模拟的回报不是局面评估\n");
    out += buf;

    /* ---- 2+3+4. 标准开局副本上的两次有界自检 ---- */
    {
        Chess probe(chess);
        probe.reset();                  /* 标准开局, sideToMove = 红 */

        /*
           第二个副本取合法走法集: sample() 现生成, 与 s 的来源无关, 两份副本互不影响
        */
        Chess probe2(chess);
        probe2.reset();
        std::vector<Step *> legal;
        probe2.sample(Stone::COLOR_RED, legal);
        const int legalCount = (int)legal.size();

        /*
           ---- 为什么要跑**两次**、而且只报一次走法坐标 (2026-09) ----
           自检的契约里有"**可重复调用**、同一局面下两次结果必须一致" (它会在 GUI 线程被
           反复调用, test_match [2.11] 把这条钉成了断言)。而 MCTS 的每一次模拟都要
           `std::rand()`: EXPANSION 是"从未展开列表里**随机**挑一个", 模拟是随机走子。
           所以带坐标地报"这一次选中了哪一步"必然逐次不同 —— 第一版就是这么写的, 当场
           被断言抓住 (可重复=0)。
           于是拆成两条:
             (a) 0 次模拟: 走的是"根有合法走法但一个孩子都没展开"那条**兜底**路径,
                 返回根未展开列表的第一手 —— 完全确定, 可以连坐标一起报;
             (b) 1 次模拟: 走正常路径 (展开一个孩子 + 随机回放), 只报"合不合法",
                 不报坐标 —— 坐标每次都在变, 但"它一定是合法走法"这件事是稳定的。
           C 直接用构造默认值 1.414 而不是本对象的 C: C 只决定探索/利用的平衡, 与
           "返回值合不合法"无关 (本对象的 C 在行首那行配置读数里已经报了)。
        */
        MCTS tmpFallback(probe, 1.414);
        const Step sFallback = tmpFallback.findBestMove(Stone::COLOR_RED, 0);
        MCTS tmpNormal(probe, 1.414);
        const Step sNormal = tmpNormal.findBestMove(Stone::COLOR_RED, 1);

        const bool legalFallback = stepInMoveList(sFallback, legal);
        const bool legalNormal = stepInMoveList(sNormal, legal);

        if (legalFallback) {
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检 (0 次模拟, 兜底路径): 合法 "
                          "(走法 (%d,%d)->(%d,%d))\n",
                          sFallback.pos.x, sFallback.pos.y,
                          sFallback.nextPos.x, sFallback.nextPos.y);
            out += buf;
        } else {
            /*
               走到这里说明兜底失效了 (或走法生成/合法集口径被改坏)。把 valid 位也
               打出来: valid=false 是"根节点一次都没扩展"那条路径, valid=true 却不在
               合法集里则是走法生成本身的问题, 两者要能分开看。
            */
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检 (0 次模拟, 兜底路径): 非法 (返回了 valid=%d 的 "
                          "Step: (%d,%d)->(%d,%d), 开局仍有 %d 个合法走法) "
                          "<-- 回归! 见 findBestMove 的根节点兜底\n",
                          sFallback.valid ? 1 : 0, sFallback.pos.x, sFallback.pos.y,
                          sFallback.nextPos.x, sFallback.nextPos.y, legalCount);
            out += buf;
        }
        /* 第 (b) 条: 不报坐标 (随机会变), 只报"合不合法" */
        if (legalNormal) {
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检 (1 次模拟, 正常路径): 合法 "
                          "(走法每次不同 —— 展开是随机挑的, 所以这里不报坐标)\n");
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "决策合法性自检 (1 次模拟, 正常路径): 非法 (valid=%d) "
                          "<-- 回归! 见 findBestMove 的根节点兜底\n",
                          sNormal.valid ? 1 : 0);
            out += buf;
        }

        /* ---- 4. 分支因子 + 界面预算摊薄 (800 是 chessboard.cpp 的 MCTS_SIMS 字面量) ---- */
        std::snprintf(buf, sizeof(buf),
                      "标准开局(红先): 合法走法 %d 个 (分支因子)\n",
                      legalCount);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "界面模拟预算: MCTS_SIMS = 800 次模拟 (chessboard.cpp) -> "
                      "每个根走法平均 %.1f 次\n",
                      legalCount > 0 ? 800.0 / (double)legalCount : 0.0);
        out += buf;

        /* 用完必须还回对象池, 否则池会碎片化 (sample 每次都从这里取) */
        Steps::instance().put(legal);
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局 "
           "(给出 Elo 差与 95% 置信区间)\n";
    return out;
}
