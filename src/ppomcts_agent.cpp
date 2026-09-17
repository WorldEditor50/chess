#include "ppomcts_agent.h"

#include "chessstate.h"   /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
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

    /*
       ---- 规则上下文平面 (本轮从 DQNAB 推广过来, 见 src/chessstate.h) ----
       三个全局标量铺满 90 格: 无吃子进度 / 重复次数 / 是否被将军。
       **必须进状态**: 同一局面的第 1 次与第 2 次出现棋盘逐位相同, 但第 3 次直接判和 ——
       不编码它, V(s) 就不是 s 的函数, Bellman 备份的前提 (P(s'|s,a) 只依赖 s) 就破了。
       数值口径一律取 ChessState 里那一份 (与引擎 isRepetition 同窗口同判据), 不另写。
    */
    ChessState::fillPlane(&state[0], PLANE_HALFMOVE, (float)ChessState::halfmovePhase(chess));
    ChessState::fillPlane(&state[0], PLANE_REPEAT, (float)ChessState::repetitionPhase(chess));
    ChessState::fillPlane(&state[0], PLANE_CHECK, (float)ChessState::checkPhase(chess, me));
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
 *  computeReward:  一步的即时奖励 (走子方视角, 含每步代价)
 *
 *  Phase 1 起统一走 stone.h 的 stepReward(): 材质系数 0.1、吃將按终局量级、
 *  每步代价 -0.005。原因与实测数字见 stone.h 里 REWARD_* 的说明。
 * ------------------------------------------------------------------ */
float PPOMCTSAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (!materialRewardEnabled) {
        /*
           材质完全交给势能 Φ (A/B 对照, 见头文件 materialRewardEnabled 的说明):
           显式奖励只留每步代价。两处都给会让"吃子"变成不受 PBRS 不变性保护的额外
           稠密奖励, 把策略推向吃子优先。
        */
        return REWARD_STEP_COST;
    }

    if (s.nextId == Stone::ID_NONE) return stepReward(false, false, 0.0);

    Stone *victim = chess.stones[s.nextId];
    /*
       不检查 victim->alive —— 见 pgagent.cpp 里同一处的说明: 调用方常在
       moveForward() 之后求奖励, 那时被吃子已 alive=false, 加判断会让吃子奖励
       恒为 0 (吃将那条分支也不可达)。
    */
    if (victim == nullptr) return stepReward(false, false, 0.0);

    /*
       吃將不再单设 +100: 吃將本来就是**终局**, 单设 100 会让同一个胜负事件同时有
       100 (即时) 与 1 (终局常量) 两种量级, 而 100 还会经折现递推放大成 ±100 量级的
       价值目标。现在按终局量级给。

       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来写的是
       `(color == COLOR_BLACK) ? reward : -reward` (黑方视角), 于是红方白吃一个
       黑车会拿到负奖励, 与同一批经验里的终局奖励 (走子方视角的 ±1) 正好相反。
       Chess::moveForward 的 totalReward 是另一套 (黑方视角) 记账, 两者同源;
       回归钉在 test_match 的 [2.6] 节。
    */
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/* ------------------------------------------------------------------
 *  potentialOf: 当前棋盘的势能 Φ, 走子方视角, 归一化到 (-1,1)
 *
 *  Chess::evaluate() 是**黑方视角** (正 = 黑好), Φ 要的是**走子方视角**, 所以
 *  colorToMove 是红方时取负 —— 与 encodeState 的规范视角是同一件事的两面。
 * ------------------------------------------------------------------ */
float PPOMCTSAgent::potentialOf(int colorToMove)
{
    /*
       用 evaluatePositional() 而不是 evaluate(): 前者额外含"将安全 / 空间 / 机动性 /
       士象完整度"这些**局面价值** —— 势能 Φ 正是把这份"藏在空间位置里的未来价值"
       提前搬进当前学习目标的那条通道 (推导见 stone.h 的 PBRS 一节)。
       它只在这里被调用 (每手两次), 所以可以算得细; ABAgent 的叶子评估仍用便宜的
       evaluate(), 两边的分工见 chess.h 的说明。
    */
    const double e = chess.evaluatePositional();   /* 黑方视角 */
    const double moverFrame = (colorToMove == Stone::COLOR_BLACK) ? e : -e;
    return potentialReward((float)moverFrame);
}

/* ------------------------------------------------------------------
 *  applyPotentialShaping: 把势能塑形原地写进轨迹 (Phase 2, 见头文件)
 *
 *  相邻两步共享同一个局面, 所以 Φ_before(i) = Φ_after(i-1) = trajectory[i-1].potential,
 *  只有第 0 步需要额外的 m_phiInit (调用方在一局开始前设好)。
 * ------------------------------------------------------------------ */
/* 边界项用的"最后一步落子后"的势能 (已乘 α; 塑形关闭或 α=0 时为 0) */
float PPOMCTSAgent::finalPhiScaled(const std::vector<RL::Step> &traj) const
{
    if (!potentialShaping || shapingAlpha == 0.0f || traj.empty()) {
        return 0.0f;
    }
    return traj.back().potential * shapingAlpha;
}

void PPOMCTSAgent::applyPotentialShaping(std::vector<RL::Step> &trajectory) const
{
    if (!potentialShaping || shapingAlpha == 0.0f || trajectory.empty()) {
        return;   /* A/B 消融 (α=0) 或关掉塑形: Φ 恒为 0, 奖励保持原样 */
    }
    /*
       α 是势能强度 (可退火): α·Φ 仍然是状态函数, 所以**任何 α 都保持策略不变性**
       (推导里把 Φ 换成 αΦ 即可), 于是"由弱到强"连续过渡不会让价值函数失效。
    */
    float phiBefore = m_phiInit * shapingAlpha;
    for (std::size_t t = 0; t < trajectory.size(); t++) {
        const float phiAfter = trajectory[t].potential * shapingAlpha;
        trajectory[t].reward = shapedStepReward(trajectory[t].reward,
                                               phiBefore,
                                               phiAfter,
                                               gamma);
        phiBefore = phiAfter;
    }
}

/* ------------------------------------------------------------------
 *  pickUntriedByPrior: 按先验挑未展开着法 (Phase 4.1 + R1)
 *
 *  先验现在是**稀疏策略前向**的结果 (ppo.actionMasked): 只算这个节点完整合法着法
 *  集合那几十列, 而不是把策略头 (64x8100 = 2.07 MB) 整块读一遍。R1 的收益全部
 *  来自这里 —— 而且语义等价 (子集重新归一 == 全量 softmax 后取子集), 合法动作之间
 *  的相对大小完全不变, 所以**展开顺序与全量写法逐次相同**。
 *
 *  合法集为什么是"untried + 已展开孩子的 parentAction":
 *    节点的合法着法集在它被创建时就是完整的 (untried = 全量合法集), 之后每展开一个
 *    就把它从 untried 移走、变成一个孩子。所以两者的并集恒等于该节点的完整合法集。
 *    **必须用完整合法集, 不能用"当前还剩的 untried"**: 稀疏概率是在这个集合上归一化
 *    的, 如果每次只用剩下的未展开项, 那么最后一个孩子的先验会变成 1.0 (集合只剩它
 *    一个), 兄弟之间的先验尺度会随展开顺序漂移 —— 那是 PUCT 的分数被污染。
 *
 *  线性扫一遍未展开列表 (中局约 40 项, 代价可忽略), 取概率最大的那个。
 * ------------------------------------------------------------------ */
int PPOMCTSAgent::pickUntriedByPrior(int nodeID, const RL::Tensor &parentState,
                                     std::vector<int> &legalIdx,
                                     std::vector<float> &probs,
                                     float &priorOut)
{
    const AZNode &node = nodes[nodeID];

    legalIdx.clear();
    legalIdx.reserve(node.untriedActionIndices.size() + node.childIDs.size());
    for (std::size_t i = 0; i < node.untriedActionIndices.size(); i++) {
        legalIdx.push_back(node.untriedActionIndices[i]);
    }
    for (std::size_t i = 0; i < node.childIDs.size(); i++) {
        legalIdx.push_back(nodes[(std::size_t)node.childIDs[i]].parentAction);
    }

    if (sparsePolicyHead) {
        ppo.actionMasked(parentState, legalIdx, probs);
    } else {
        /*
           R1 之前的对照口径 (A/B 用, 见头文件 sparsePolicyHead): 全量策略前向, 先验
           直接取 p_full(a) —— 合法集上的质量 Z 没有被除掉。这是**逐字复现**改动前的
           搜索行为, 所以 "sparsePolicyHead=true/false" 的配对差就是 R1 的全部影响。
        */
        RL::Tensor &full = ppo.action(parentState);
        probs.resize(legalIdx.size());
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            const int a = legalIdx[i];
            probs[i] = (a >= 0 && a < ppo.actionDim) ? full[(std::size_t)a] : 0.0f;
        }
    }

    /*
       ---- 根节点的 Dirichlet 噪声 (2026-09) ----
       根上开着噪声时, 改用 applyRootNoise 预先算好的"加噪先验" (按动作下标索引)。
       覆盖发生在**算完网络先验之后**, 因为噪声的定义就是"在网络先验上做凸组合";
       这个覆盖同时作用于下面两个出口 —— "挑哪个未展开着法"和返回给调用方的 prior
       (它会存进孩子、进 PUCT 的 U 项)。只改前者等于噪声没接上探索强度。
    */
    if (m_rootNoiseActive && nodeID == m_rootID && !m_rootPriorNoised.empty()) {
        probs.resize(legalIdx.size());
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            const int a = legalIdx[i];
            probs[i] = (a >= 0 && a < ACTION_DIM)
                           ? m_rootPriorNoised[(std::size_t)a]
                           : 0.0f;
        }
    }

    /*
        probs 与 legalIdx 一一对应, 而 legalIdx 的**前 untried.size() 项**就是
        untriedActionIndices 本身 (上面按这个顺序填的) —— 于是"挑最大"和"查选中动作
        的概率"都是直接下标访问, 不需要再查表。
    */
    int best = 0;
    float bestP = -1.0f;
    for (std::size_t i = 0; i < node.untriedActionIndices.size(); i++) {
        const float p = (i < probs.size()) ? probs[i] : 0.0f;
        if (p > bestP) {
            bestP = p;
            best = (int)i;
        }
    }
    priorOut = (bestP > 0.0f) ? bestP : 0.0f;
    return best;
}

/* ------------------------------------------------------------------
 *  getPUCT:  PUCT score for a child node
 *
 *  Formula:
 *    U(s,a) = -Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *
 *  Unvisited children return a very large score to ensure they
 *  are explored first.
 *
 *  ---- 符号约定 (2026-09 修正, 这是一个"不报错但致命"的 bug) ----
 *
 *  子节点里存的 Q 是**子节点走棋方**的价值: backup 沿父链逐层翻号 (见 selectMove
 *  的 Phase 4 与 trainSelfPlay 里同一段), 而价值头的训练口径也正是"走子方视角"
 *  (RL::PPO::discountedReturns 的 r = reward - gamma*r; test_ppomcts 的
 *  "discounted-return sign convention" 一节把这条钉成断言)。
 *
 *  子节点的走棋方**恰好就是父节点走棋方的对手** (父走一手, 轮到对手), 所以父节点
 *  要挑"对自己好"的着法, 必须取**负号**再比大小。漏掉它不会报任何错, 只会让搜索
 *  去最大化**对手**的价值 —— 即专挑对自己最差的着法, 而且**评估越准错得越狠**。
 *
 *  症状 (与本次排查的两条报告完全吻合):
 *   * "loss 一路降、棋力不涨甚至越训越臭" —— 策略被蒸馏向搜索选出的坏着法;
 *   * "胆小不吃子" —— 吃子后轮到对手、对手少一个大子, 该子节点 Q 对对手为负,
 *     于是吃子在根上被算成亏着, 而"平推不交换"的子节点 Q ≈ 0 反而更大。
 *
 *  最小验算 (可直接手算): 根为红方, A 走后红将死黑 ⇒ A 的 Q = -1;
 *  B 走后黑将死红 ⇒ B 的 Q = +1。红方该选 A, 而漏负号会选 B。
 *  单测: test_ppomcts 的 "PUCT sign" 一节用一棵手搭的两孩子树把这条钉住。
 * ------------------------------------------------------------------ */
double PPOMCTSAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[childID];

    if (child.visitCount == 0) {
        /* Always explore unvisited nodes first */
        return std::numeric_limits<double>::max();
    }

    /* 取负号的理由见上面的符号约定: 子节点的 Q 是对手视角的 */
    const double q = -child.getQ();         /* -Q(s,a) = -W/N */
    const double puct = c_puct * child.prior
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
 *  pickRootChildByVisits: 从根的访问计数里出招 (唯一口径, 2026-09)
 *
 *  temp > 0.1  : 按 N^(1/temp) 采样 —— 探索就靠它 (训练/自对弈取数据)
 *  temp <= 0.1 : 取访问最多的孩子 (argmax) —— 评测/对局要确定性
 *
 *  这段逻辑原来内联在 trainSelfPlay 里, 而 `selectMove(color, sims, temp)` 的 temp
 *  **从头到尾没被读过**: 它永远走 argmax。也就是说公开 API 承诺的"temp > 0 时按访问
 * 分布采样"根本没实现 —— 任何传 temp>0 的调用方 (例如想给自对弈或基准加探索的)
 *  都会静默地拿到确定性结果。现在两处共用这一个实现, 口径不可能再漂移。
 *
 *  返回 -1 表示"根的访问计数总和为 0"。这与改动前 trainSelfPlay 的分支条件一致
 *  (`bestChildID >= 0 && totalVisits > 0`), 所以对既有行为是保守的。
 * ------------------------------------------------------------------ */
int PPOMCTSAgent::pickRootChildByVisits(int rootID, float temp)
{
    if (rootID < 0 || (std::size_t)rootID >= nodes.size()) {
        return -1;
    }
    const AZNode &root = nodes[(std::size_t)rootID];

    int bestChildID = -1;
    int maxVisits = -1;
    int totalVisits = 0;
    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const int childID = root.childIDs[k];
        const int v = nodes[(std::size_t)childID].visitCount;
        totalVisits += v;
        if (v > maxVisits) {
            maxVisits = v;
            bestChildID = childID;
        }
    }
    if (bestChildID < 0 || totalVisits <= 0) {
        return -1;
    }

    if (temp <= 0.1f) {
        return bestChildID;                 /* 低温柔性: 确定性 argmax */
    }

    /*
       温度采样。注意**保持与原实现逐位相同**的写法 (同一个稠密 ACTION_DIM 张量、
       同样的填充顺序、同一次 categorical 调用): RL::Random::categorical 内部是
       std::discrete_distribution, 输入相同才给出同一条随机序列, 否则自对弈的可复现性
       (RL::Random::setSeed) 会变。
    */
    RL::Tensor visitProbs(ACTION_DIM, 1);
    visitProbs.zero();
    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const AZNode &child = nodes[(std::size_t)root.childIDs[k]];
        if (child.parentAction < 0 || child.parentAction >= ACTION_DIM) {
            continue;
        }
        const double prob = std::pow((double)child.visitCount,
                                     1.0 / (double)temp);
        visitProbs[(std::size_t)child.parentAction] = (float)prob;
    }
    float sum = 0.0f;
    for (int i = 0; i < ACTION_DIM; i++) {
        sum += visitProbs[(std::size_t)i];
    }
    if (sum > 1e-9f) {
        for (int i = 0; i < ACTION_DIM; i++) {
            visitProbs[(std::size_t)i] /= sum;
        }
    }
    const int chosenAction = RL::Random::categorical(visitProbs);

    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const int childID = root.childIDs[k];
        if (nodes[(std::size_t)childID].parentAction == chosenAction) {
            return childID;
        }
    }
    /* 采样到了一个没有对应孩子的下标 (理论上不会发生): 退回 argmax 而不是给无效着法 */
    return bestChildID;
}

/* ------------------------------------------------------------------
 *  evaluateLeaf: 叶子估值 (三个搜索入口的唯一口径, 2026-09)
 *
 *  这是诊断仪表盘直接抓出来的一个缺陷: 原来三处都是无条件
 *      encodeState(leafState); reward = ppo.value(leafState);
 *  于是"一步杀"那步落子后, 叶子被交给一个只会**评估**的 critic 去猜 (它甚至不知道
 *  这个局面已经终局), 搜索因此看不见任何战术终点。bench_diag 实测: 一步杀命中率
 *  **0/20**, 而白吃子(非终局, 靠 V 比较)有 16.7% —— 这个 0% 与 16.7% 的对比就是
 *  "终局没做特殊处理"的直接证据。
 *
 *  成本: 每个模拟多一次 getResult (以将/困毙/判和/60 回合的判定), 相对一次网络前向
 *  (当前骨干约 8 ms/模拟) 可以忽略; 小网络下它是可观的一小块, 但正确性优先 ——
 *  "搜索看不见将杀"是不可接受的。
 * ------------------------------------------------------------------ */
double PPOMCTSAgent::evaluateLeaf(RL::Tensor &leafStateScratch)
{
    const int leafResult = chess.getResult(chess.sideToMove);
    if (leafResult != Chess::RESULT_ONGOING) {
        const int winner = winnerOfResult(leafResult);
        if (winner == Stone::COLOR_NONE) {
            return 0.0;                                   /* 判和 */
        }
        /* 必须按**叶子走棋方**视角给: 此刻 chess.sideToMove 就是叶子的走棋方 */
        return (winner == chess.sideToMove) ? 1.0 : -1.0;
    }
    encodeState(leafStateScratch);
    return (double)ppo.value(leafStateScratch);
}

/* ------------------------------------------------------------------
 *  rootDiag: 把根的访问/Q/先验整理成诊断量 (只读, 2026-09)
 *
 *  为什么这些量值得单独抽出来: 棋力是滞后指标, 而"搜索这一层设计对不对"当场就能看。
 *  尤其是**吃子 vs 退让的 Q 分组** —— 报告里那句"胆小不吃子"在代码层只有这一个
 *  地方能被直接量化。判读口径写进 rl/diag.h 的 Aggregates::print。
 *
 *  三个必须注意的口径:
 *   1. Q 取**负号**换算到根走棋方视角 (子节点存的是它自己走棋方的价值)。
 *   2. 先验在**已展开集合**上重归一后再算熵/KL —— 否则未展开着的概率质量会把
 *      KL 压低, 看起来像"搜索没纠正网络", 而实际原因是"那些着法还没被展开"。
 *   3. `legalCount` 用 legalIndicesOf (未展开 ∪ 已展开), 它等于该节点的**完整**合法集;
 *      用它做分母才是真正的"展开覆盖率"。
 * ------------------------------------------------------------------ */
bool PPOMCTSAgent::rootDiag(RL::Diag::RootDiag &out) const
{
    out = RL::Diag::RootDiag();
    if (m_rootID < 0 || (std::size_t)m_rootID >= nodes.size()) {
        return false;
    }
    const AZNode &root = nodes[(std::size_t)m_rootID];

    /* 完整合法集 (未展开 ∪ 已展开) —— 展开覆盖率的分母 */
    std::vector<int> legal;
    legalIndicesOf(m_rootID, legal);
    out.legalCount = (int)legal.size();

    const std::size_t nc = root.childIDs.size();
    if (nc == 0) {
        return false;                      /* 一次模拟都没跑过: 没有可诊断的东西 */
    }

    std::vector<double> visits(nc, 0.0);
    std::vector<double> priors(nc, 0.0);
    double sumPrior = 0.0;
    for (std::size_t k = 0; k < nc; k++) {
        const int cid = root.childIDs[k];
        if (cid < 0 || (std::size_t)cid >= nodes.size()) {
            continue;
        }
        const AZNode &c = nodes[(std::size_t)cid];
        visits[k] = (double)c.visitCount;
        priors[k] = (double)c.prior;
        sumPrior += (double)c.prior;
        out.totalVisits += c.visitCount;
    }
    if (out.totalVisits <= 0) {
        return false;                      /* 零访问: 分布无意义 */
    }

    std::vector<double> pv(nc, 0.0);
    for (std::size_t k = 0; k < nc; k++) {
        pv[k] = visits[k] / (double)out.totalVisits;
    }
    std::vector<double> pp(nc, 0.0);
    if (sumPrior > 1e-12) {
        for (std::size_t k = 0; k < nc; k++) {
            pp[k] = priors[k] / sumPrior;
        }
    }

    out.children = (int)nc;
    out.expandCoverage = (out.legalCount > 0)
                             ? (double)nc / (double)out.legalCount : 0.0;
    out.topShare = 0.0;
    for (std::size_t k = 0; k < nc; k++) {
        if (pv[k] > out.topShare) { out.topShare = pv[k]; }
    }
    out.visitEntropy = RL::Diag::entropy(pv);
    out.priorEntropy = RL::Diag::entropy(pp);
    out.priorKl = RL::Diag::klDivergence(pv, pp);

    /* ---- 吃子 / 退让 分组 (核心: "搜索是不是怕吃子") ---- */
    const double kNegInf = -std::numeric_limits<double>::infinity();
    double qCapMax = kNegInf;
    double qQuietMax = kNegInf;
    int bestVisits = -1;
    for (std::size_t k = 0; k < nc; k++) {
        const int cid = root.childIDs[k];
        if (cid < 0 || (std::size_t)cid >= nodes.size()) {
            continue;
        }
        const AZNode &c = nodes[(std::size_t)cid];
        /* Q 换算到**根走棋方**视角 (子节点是对手视角 ⇒ 取负) */
        const double q = RL::Diag::qForParent(c.totalValue, c.visitCount);
        /* nextId != ID_NONE 就是吃子 (见 stone.h 的 Step 说明) */
        const bool isCap = (c.step.nextId != Stone::ID_NONE);
        if (isCap) {
            out.captureCount++;
            out.captureVisits += c.visitCount;
            if (q > qCapMax) { qCapMax = q; }
        } else if (q > qQuietMax) {
            qQuietMax = q;
        }
        /* 访问最多的着法是否为吃子: 用严格大于, 与 pickRootChildByVisits 的 argmax 同规则 */
        if (c.visitCount > bestVisits) {
            bestVisits = c.visitCount;
            out.bestIsCapture = isCap ? 1 : 0;
        }
    }
    out.captureVisitShare = (double)out.captureVisits / (double)out.totalVisits;
    out.qCaptureMax = (out.captureCount > 0) ? RL::Diag::safe(qCapMax) : 0.0;
    out.qQuietMax = (qQuietMax != kNegInf) ? RL::Diag::safe(qQuietMax) : 0.0;
    out.valid = true;
    return true;
}

/* ------------------------------------------------------------------
 *  printSortedRoot: 把根的已展开着法按访问数排序打印 (P/Q/N + 是否吃子)
 *
 *  这是最省事的单局面定位手段: 挑一个"明显该吃"的局面跑一次搜索, 看吃子着排第几、
 *  它的 Q 是正还是负 —— 一眼就能分开"网络不给吃子先验"与"搜索把吃子算成亏着"。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::printSortedRoot(int limit) const
{
    if (m_rootID < 0 || (std::size_t)m_rootID >= nodes.size()) {
        std::printf("[root] 没有树 (先跑一次 selectMove)\n");
        return;
    }
    const AZNode &root = nodes[(std::size_t)m_rootID];

    struct Row {
        int visits = 0;
        double q = 0.0;
        double prior = 0.0;
        int isCap = 0;
        int fx = 0, fy = 0, tx = 0, ty = 0;
    };
    std::vector<Row> rows;
    rows.reserve(root.childIDs.size());
    for (std::size_t k = 0; k < root.childIDs.size(); k++) {
        const int cid = root.childIDs[k];
        if (cid < 0 || (std::size_t)cid >= nodes.size()) {
            continue;
        }
        const AZNode &c = nodes[(std::size_t)cid];
        Row r;
        r.visits = c.visitCount;
        r.q = RL::Diag::qForParent(c.totalValue, c.visitCount);
        r.prior = c.prior;
        r.isCap = (c.step.nextId != Stone::ID_NONE) ? 1 : 0;
        r.fx = c.step.pos.x; r.fy = c.step.pos.y;
        r.tx = c.step.nextPos.x; r.ty = c.step.nextPos.y;
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row &a, const Row &b) { return a.visits > b.visits; });

    std::printf("[root] 走棋方=%s, 已展开 %d 个孩子 (按访问数排序)\n",
                (root.currentColor == Stone::COLOR_RED) ? "RED" : "BLACK",
                (int)rows.size());
    std::printf("   %-16s %6s %8s %9s %s\n", "move", "N", "Q(parent)", "prior", "吃子");
    const int n = (limit > 0 && (int)rows.size() > limit) ? limit : (int)rows.size();
    for (int i = 0; i < n; i++) {
        char mv[32];
        std::snprintf(mv, sizeof(mv), "(%d,%d)->(%d,%d)",
                      rows[(std::size_t)i].fx, rows[(std::size_t)i].fy,
                      rows[(std::size_t)i].tx, rows[(std::size_t)i].ty);
        std::printf("   %-16s %6d %8.4f %9.4f %s\n",
                    mv, rows[(std::size_t)i].visits, rows[(std::size_t)i].q,
                    rows[(std::size_t)i].prior,
                    rows[(std::size_t)i].isCap ? "吃" : "");
    }
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
 *
 *  势能塑形 (Phase 2): 进池前先把"棋盘局面价值评估"作为势能加进每步奖励 ——
 *  见下面 shaped[] 那一段与 stone.h 的推导。没有它, 截断 rollout 的价值目标全是
 *  0.005~0.01 量级 (诊断 [4] 实测 |target|>0.1 的样本占 0%), critic 只能学成常数。
 * ------------------------------------------------------------------ */
/* ------------------------------------------------------------------
 *  legalIndicesOf (R2): 节点的**完整合法着法集**
 *
 *  与 pickUntriedByPrior 里那套构造同源: 节点创建时 untried = 全量合法集, 之后每展开
 *  一个就从 untried 移走、变成一个孩子 —— 所以"未展开 ∪ 已展开孩子的 parentAction"
 *  恒等于该节点的完整合法集。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::legalIndicesOf(int nodeID, std::vector<int> &out) const
{
    out.clear();
    if (nodeID < 0 || (std::size_t)nodeID >= nodes.size()) {
        return;
    }
    const AZNode &node = nodes[(std::size_t)nodeID];
    out.reserve(node.untriedActionIndices.size() + node.childIDs.size());
    for (std::size_t i = 0; i < node.untriedActionIndices.size(); i++) {
        out.push_back(node.untriedActionIndices[i]);
    }
    for (std::size_t i = 0; i < node.childIDs.size(); i++) {
        const int c = node.childIDs[i];
        if (c >= 0 && (std::size_t)c < nodes.size()) {
            out.push_back(nodes[(std::size_t)c].parentAction);
        }
    }
}

/* ------------------------------------------------------------------
 *  commitEpisode (P3 + P4 + R2)
 * ------------------------------------------------------------------ */
/* ------------------------------------------------------------------
 *  bootstrapOutcome: 截断局的"终局值"用 critic 自举 (2026-09)
 *
 *  调用时机: 自对弈跑到手数上限仍未分胜负时, 棋盘停在**最后一步落子之后**的局面,
 *  此时 chess.sideToMove 是**对手** (moveForward 会翻转走棋方)。
 *
 *  口径: `encodeState` 按 chess.sideToMove 编码 ⇒ ppo.value(...) 是**对手视角**的
 *  价值; 取负号才是"最后一步走子方"视角 —— 这正是 commitEpisode 的 finalOutcome
 *  所需的口径 (终局值按最后一步走子方给)。
 *
 *  为什么是"取代 0"而不是"加到 0 上": finalOutcome 是回报递推的起点
 *  (returns[i] = reward_i - gamma*returns[i+1], 起点是 -finalOutcome)。传 0 等于断言
 *  "这局是均势和棋", 传 ±V 才是"按当前估值继续下下去会怎样"。
 *
 *  量纲: 与在线路径 endOnline 一致 (同样 clamp 到 ±1, 与 REWARD_TERMINAL 同量级)。
 *  势能塑形的边界项由 commitEpisode 自己扣 (它减掉 Φ(s_end+1)), 这里不必管 ——
 *  这正是"塑形不改最优策略"那条不变性所要求的。
 * ------------------------------------------------------------------ */
float PPOMCTSAgent::bootstrapOutcome()
{
    RL::Tensor lastState(STATE_DIM, 1);
    encodeState(lastState);                 /* 按 chess.sideToMove = 对手 编码 */
    float v = -(float)ppo.value(lastState);
    if (!(v > -1.0f)) {
        v = -1.0f;                          /* 同时挡住 NaN */
    }
    if (v > 1.0f) {
        v = 1.0f;
    }
    return v;
}

void PPOMCTSAgent::commitEpisode(std::vector<RL::Step> &trajectory,
                                 float finalOutcome,
                                 const std::vector<std::vector<int>> *legalPerStep)
{
    if (trajectory.empty()) {
        return;
    }

    /*
       ---- 势能塑形: r'_i = r_i + Φ(s_i) + γ·Φ(s_{i+1}) ----
       直接原地写进轨迹, 之后照常算折现回报。Φ 的作用是给"安静局面"一个非零目标
       (棋盘局面价值评估), 而且**不改变最优策略**: 同一局面下各着法的 Q 只被平移
       同一个 Φ(s), 排序不变 (推导见 stone.h)。
       (参数因此不再是 const —— 轨迹是调用方的局部变量, 用完即弃。)
    */
    applyPotentialShaping(trajectory);
    /*
       终局常量也必须跟着势能一起平移 —— 这是 PBRS 的**边界项**, 漏了它就不是严格的
       势能塑形 (数值上表现为"塑形与不塑形之差 ≠ Φ(s_0)")。

       推导: V'(x) = V(x) + Φ(x), 而 finalOutcome 的口径是"最后一步走子方"对
       **落子后局面**的价值 (= -V(s_end+1), 因为那个局面轮到对手走)。平移后
       -V'(s_end+1) = -V(s_end+1) - Φ(s_end+1) = finalOutcome - Φ(s_end+1),
       而 Φ(s_end+1) 正是最后一步记下的 trajectory.back().potential。
    */
    const float shiftedOutcome = finalOutcome - finalPhiScaled(trajectory);
    const std::vector<float> returns = ppo.discountedReturns(trajectory, shiftedOutcome);

    std::vector<int> idx;
    std::vector<float> prob;
    std::vector<int> legal;
    std::vector<int> mirrorIdx;
    std::vector<int> mirrorLegal;
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
        /*
           R2: 这一手局面的完整合法集 (调用方按 ply 收集)。没有就给空 -> 旧口径。
        */
        if (legalPerStep != nullptr && t < legalPerStep->size() &&
            !(*legalPerStep)[t].empty()) {
            legal = (*legalPerStep)[t];
        } else {
            legal.clear();
        }
        ppo.addReplay(trajectory[t].state, idx, prob, returns[t], legal);

        if (mirrorAugment) {
            mirrorIdx.resize(idx.size());
            for (std::size_t i = 0; i < idx.size(); i++) {
                mirrorIdx[i] = mirrorActionIdx(idx[i]);
            }
            mirrorLegal.resize(legal.size());
            for (std::size_t i = 0; i < legal.size(); i++) {
                mirrorLegal[i] = mirrorActionIdx(legal[i]);
            }
            mirrorPlanes(trajectory[t].state, mirrorState);
            ppo.addReplay(mirrorState, mirrorIdx, prob, returns[t], mirrorLegal);
        }
    }

    /* 池子够大就开始批量学习: 每次采样 batchSize 条、过 epochs 遍, 一次优化器更新 */
    if (replayBatchSize > 0 && ppo.replaySize() >= (std::size_t)replayBatchSize) {
        ppo.learnFromReplay((std::size_t)replayBatchSize, replayEpochs, learningRate);
    }
}

/* ------------------------------------------------------------------
 *  B-5: 置换表 + 子树复用 (见头文件里那一大段说明)
 *
 *  acquireRoot 是三个搜索入口唯一的"取根"入口。它做三件事:
 *    1. 算当前局面的 Zobrist 键, 查置换表;
 *    2. 命中且**从上一棵树的根 ≤ 2 步可达** -> 复用那个节点 (连带子树与统计);
 *       否则新建一个根 (并把键登记进表);
 *    3. 节点数超过 treeNodeCap 时先把整棵树丢掉 (内存兜底)。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::resetSearchTree()
{
    nodes.clear();
    m_tt.clear();
    m_rootID = -1;
    m_reuseHits = 0;
    m_nodesCreated = 0;
    /* 新一局: 根噪声的手数计数与"加噪后先验"一并失效 (见 rootNoise 的说明) */
    m_plyInGame = 0;
    m_rootNoiseActive = false;
    m_rootPriorNoised.clear();
}

bool PPOMCTSAgent::reachableWithin(int fromID, int targetID, int maxDepth) const
{
    if (fromID < 0 || targetID < 0) {
        return false;
    }
    if (fromID == targetID) {
        return true;
    }
    if (maxDepth <= 0 || (std::size_t)fromID >= nodes.size()) {
        return false;
    }
    /* 逐层向下 BFS (只看 childIDs; 两层在 80 次模拟的树上最多几千个节点, 可忽略) */
    std::vector<int> frontier;
    frontier.push_back(fromID);
    for (int d = 0; d < maxDepth; d++) {
        std::vector<int> next;
        for (std::size_t k = 0; k < frontier.size(); k++) {
            const int id = frontier[k];
            if ((std::size_t)id >= nodes.size()) {
                continue;
            }
            const std::vector<int>& kids = nodes[(std::size_t)id].childIDs;
            for (std::size_t c = 0; c < kids.size(); c++) {
                if (kids[c] == targetID) {
                    return true;
                }
                next.push_back(kids[c]);
            }
        }
        if (next.empty()) {
            return false;
        }
        frontier.swap(next);
    }
    return false;
}

int PPOMCTSAgent::acquireRoot(int color, bool withRootNoise)
{
    /*
       根噪声的手数: `ply` 是**本手**的编号 (0 起), 只在自对弈取数据时递增 ——
       评测/对局路径不计数, 于是噪声退火不会被"下过多少盘"污染。
       每次取根都要先把上一手的 m_rootNoiseActive 清掉, 否则关掉噪声后根上还会
       残留上一手的加噪先验。
    */
    const int ply = m_plyInGame;
    if (withRootNoise) {
        m_plyInGame++;
    }
    m_rootNoiseActive = false;

    /*
       treeReuse = false 时**逐字复现改动前**的行为: 每 ply 一棵新树 (原来是 nodes.clear())。
       注意只清树、不清计数器 —— m_nodesCreated / m_reuseHits 是整局的累计量, 两条路都要能比。
    */
    if (!treeReuse) {
        nodes.clear();
        m_rootID = -1;
    }

    /*
       键必须按"轮到 color 走"算: Chess::computeHash() 把 sideToMove 也算进去, 而调用方
       棋盘上的 sideToMove 不一定等于 color (测试里就是 reset() 之后直接走 BLACK)。
       这里临时对齐再还原, 保证同一个局面在任何调用路径下算出的键都一样。
    */
    const int savedSide = chess.sideToMove;
    chess.sideToMove = color;
    const unsigned long long key = chess.computeHash();
    chess.sideToMove = savedSide;

    int reused = -1;
    if (treeReuse) {
        std::unordered_map<unsigned long long, int>::const_iterator it = m_tt.find(key);
        if (it != m_tt.end()) {
            const int idx = it->second;
            if (idx >= 0 && (std::size_t)idx < nodes.size() &&
                nodes[(std::size_t)idx].hash == key) {
                /*
                   只在"从上一棵树的根往下 ≤ 2 步可达"时接受 —— 上一局的节点 (以及每个
                   新对局都会遇到的初始局面) 必然不可达, 于是自动退化成新开一棵树,
                   不会把上一局的访问计数/陈旧先验带进新一局。理由见头文件。
                */
                if (m_rootID < 0 || reachableWithin(m_rootID, idx, 2)) {
                    reused = idx;
                }
            }
        }
    }

    if (reused < 0 && nodes.size() >= treeNodeCap) {
        /* 内存兜底: 整棵重来 (置换表与计数器一并清掉) */
        resetSearchTree();
    }

    if (reused >= 0) {
        m_reuseHits++;
        m_rootID = reused;
        nodes[(std::size_t)reused].parentID = -1;
        nodes[(std::size_t)reused].parentAction = -1;
        if (withRootNoise) {
            applyRootNoise(reused, color, ply);
        }
        return reused;
    }

    /* ---- 新建一个根 ---- */
    std::vector<Step*> rootSteps;
    std::vector<int> rootActionIndices;
    RL::Tensor rootActionMask(ACTION_DIM, 1);
    rootActionMask.zero();
    getLegalActions(color, rootSteps, rootActionIndices, rootActionMask);

    if (rootActionIndices.empty()) {
        /*
            这个局面没有合法走法 (被将死/困毙/无棋可走) —— 不建节点, 返回 -1 让调用方
            走"当前走子方输了"的分支 (三个入口原来各自用 rootSteps.empty() 判这一条,
            B-5 把它收进这里, 移动生成只做一次)。
        */
        Steps::instance().put(rootSteps);
        m_rootID = -1;
        return -1;
    }

    AZNode rootNode(-1, -1, ::Step(), 0.0f, color, key, 0);
    for (std::size_t i = 0; i < rootActionIndices.size(); i++) {
        rootNode.untriedActionIndices.push_back(rootActionIndices[i]);
        rootNode.untriedSteps.push_back(*rootSteps[i]);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(rootNode);
    const int id = (int)nodes.size() - 1;
    m_nodesCreated++;
    if (treeReuse && ttMaxDepth >= 0) {
        m_tt[key] = id;
    }
    m_rootID = id;
    if (withRootNoise) {
        applyRootNoise(id, color, ply);
    }
    return id;
}

/* ------------------------------------------------------------------
 *  applyRootNoise: 把根节点的先验换成 (1-eps)P + eps·Dir(alpha)
 *
 *  为什么在"取根"这里做, 而不是等第一个孩子被展开时: 根的先验要同时影响两件事 ——
 *  (a) 未展开着法的**展开顺序** (pickUntriedByPrior 按先验挑最大的),
 *  (b) 已展开孩子存下来的 `prior`, 也就是 PUCT 的 U 项。
 *  两者都必须用加噪后的先验, 否则噪声只改了顺序、没改探索强度, 等于白加。
 *
 *  口径: 稀疏前向 `ppo.actionMasked`, 与 pickUntriedByPrior 完全一致 (合法集上归一),
 *  所以"加噪前的 P"和搜索实际用的 P 是同一个东西。
 *
 *  eps 按手数线性退火: 第 0 手 eps = rootNoiseEps, 到 rootNoiseMoves 手降到 0
 *  (之后整个函数直接返回, 不再有任何开销)。
 * ------------------------------------------------------------------ */
void PPOMCTSAgent::applyRootNoise(int rootID, int color, int ply)
{
    if (!rootNoise || rootNoiseEps <= 0.0f) {
        return;
    }
    if (ply >= rootNoiseMoves) {
        return;                     /* 开局阶段已过: 残局要收敛, 不加噪声 */
    }
    if (rootID < 0 || (std::size_t)rootID >= nodes.size()) {
        return;
    }

    /* 根的**完整合法着法集** (未展开 ∪ 已展开孩子的 parentAction) —— 与先验同一口径 */
    std::vector<int> legalIdx;
    legalIndicesOf(rootID, legalIdx);
    if (legalIdx.empty()) {
        return;
    }

    /* 网络先验: 显式按 color 编码 (此刻 chess.sideToMove 未必等于 color) */
    const int savedSide = chess.sideToMove;
    chess.sideToMove = color;
    RL::Tensor rootState(STATE_DIM, 1);
    encodeState(rootState);
    chess.sideToMove = savedSide;

    std::vector<float> probs;
    ppo.actionMasked(rootState, legalIdx, probs);

    /* Dirichlet(alpha) 噪声与线性退火 */
    std::vector<float> noise(legalIdx.size());
    RL::Random::dirichlet(noise, rootNoiseAlpha);

    const float t = (rootNoiseMoves > 1)
                        ? (float)ply / (float)(rootNoiseMoves - 1)
                        : 0.0f;
    float eps = rootNoiseEps * (1.0f - (t > 1.0f ? 1.0f : t));
    if (eps <= 0.0f) {
        return;
    }

    m_rootPriorNoised.assign((std::size_t)ACTION_DIM, 0.0f);
    for (std::size_t i = 0; i < legalIdx.size(); i++) {
        const int a = legalIdx[i];
        if (a < 0 || a >= ACTION_DIM) {
            continue;
        }
        const float p = (i < probs.size()) ? probs[i] : 0.0f;
        m_rootPriorNoised[(std::size_t)a] = (1.0f - eps) * p + eps * noise[i];
    }
    m_rootNoiseActive = true;
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
    /*
       B-5: 这里原来是无条件 `nodes.clear()` —— 每一步都从零重搜。现在改成向置换表
       要根 (见 acquireRoot): 同一局里"对手刚走的那个局面"如果在我们上一棵树的
       两步以内, 那棵子树连同统计一起接着用。

       注意: 局面来自**另一局**时 (bench/测试里每次都是独立局面) 命中不可达 ⇒ 新开
       一棵树, 行为与改动前逐位相同 —— 这也是"改动不影响独立局面选点"那条断言的依据。
    */
    /*
       评测路径默认**不加**根噪声 (evalRootNoise=false) —— 量棋力时不能掺探索噪声。
       打开它只为诊断 A/B ("Dirichlet 有效性": 同一局面开/关噪声, 看吃子着拿到多少访问)。
    */
    const int rootID = acquireRoot(color, evalRootNoise);

    /* 这个局面没有合法走法 (被将死/困毙): 返回"无效走法"让调用方自己判负。
       acquireRoot 已经把棋盘恢复原样 (它只临时对齐 sideToMove 算哈希)。 */
    if (rootID < 0) {
        return Step();
    }

    /* 这一步的真棋盘视角对齐在下面 (搜索期间 encodeState 依赖 chess.sideToMove) */

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
    /* R1: 稀疏先验的三个复用缓冲 (合法动作集 / 概率 / 选中动作的先验) */
    std::vector<int> legalIdxScratch;
    std::vector<float> probsScratch;
    float priorScratch = 0.0f;

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
            /*
               Phase 4.1 + R1: 先求**父节点**的策略 (稀疏: 只算合法列), 再按先验挑
               未展开着法。

               边 (parent -> child) 的先验是 P(s_parent, a) = pi_theta(s_parent)[a],
               必须用父节点的网络输出, 而且必须与动作下标处在同一个规范视角。原来的
               代码拿的是子节点的策略 (`childPolicy[chosenAction]`): 那既用错了网络
               (子节点的策略描述的是下一手走棋方的选择), 也用错了视角 —— 换成规范视角
               后两个帧会直接错开, 先验会变成完全无关的数。此刻棋子还没落, 棋盘正是
               父节点局面。

               挑法也从 `std::rand() % size` (随机) 改成**按先验最大**: 随机挑等于让
               先验完全不参与"展开哪个孩子", 而中局约 40 个合法着法、一次决策只有 80
               次模拟, 于是前 ~40 次模拟全花在随机铺开 40 个孩子上。

               R1 (2026-09): `ppo.action(parentState)` (全量 8100 列, 2.07 MB 权重流量)
               换成 `ppo.actionMasked` (只算合法列, ~10 KB) —— 两者在合法集上的相对
               大小完全一致, 所以展开顺序不变 (见 pickUntriedByPrior 的说明)。
            */
            encodeState(parentState);
            const int moveIdx = pickUntriedByPrior(nodeID, parentState,
                                                   legalIdxScratch, probsScratch,
                                                   priorScratch);
            int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
            Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];
            float prior = priorScratch;
            if (prior < 1e-9f) prior = 1e-9f; /* avoid zero prior */

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

            AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

            /*
               B-5: 记下这个子局面 (棋子位置 + 走棋方) 的置换表键并登记进去 ——
               下一步真的落子走到这里时, 整棵子树连同访问计数/Q 会被直接复用
               (自对弈里必然命中: 那边两边都是同一个 agent)。键必须现在算: 棋盘刚走完
               chosenStep, 正停在这个子局面上。
            */
            newNode.hash = chess.computeHash();
            newNode.depth = nodes[nodeID].depth + 1;
            if (treeReuse && newNode.depth <= ttMaxDepth) {
                m_tt[newNode.hash] = (int)nodes.size();   /* push_back 之后就是它的下标 */
            }
            m_nodesCreated++;

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
        /*
           叶子估值统一走 evaluateLeaf(): **终局叶子给真实 ±1** (而不是交给 critic 猜),
           非终局才用 ppo.value。这一步是诊断仪表盘抓出来的 —— 不做它, 搜索看不见
           一步杀 (bench_diag 实测命中率 0/20)。详见 evaluateLeaf 的注释。
        */
        double reward = evaluateLeaf(leafState);

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
    /*
       出招现在走唯一的那个函数 (pickRootChildByVisits):
         temp <= 0.1 -> argmax over visits (评测/对局, 确定性)
         temp >  0.1 -> 按 N^(1/temp) 采样 (探索)
       在本次改动之前, 这里的 `temp` 参数**从未被读取**, 所以无论调用方传什么都是
       argmax —— 与函数注释承诺的行为不符。修复后调用方传 0.0f 的结果与改动前**逐位
       相同** (走同一个 argmax 分支), 所以评测基线与既有基准都不受影响。
    */
    chess.sideToMove = savedSideToMove;   /* 恢复调用方的棋盘视角 (见本函数开头的说明) */

    const int chosenID = pickRootChildByVisits(rootID, temp);
    if (chosenID >= 0) {
        return nodes[chosenID].step;
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
        /*
           B-5: **每局**清一次树 (而不是每 ply 清一次)。一局之内子树跨 ply 复用,
           跨局必须失效 —— 否则每个新对局都会在初始局面上命中上一局的根。
           (acquireRoot 的"≤2 步可达"判据也会挡住跨局命中, 这里是显式的双保险。)
        */
        resetSearchTree();

        /* Store trajectories for training */
        std::vector<RL::Step> trajectory;
        /* R2: 每一手局面的完整合法着法集 (与 trajectory 同步, 供训练侧稀疏口径用) */
        std::vector<std::vector<int>> legalPerStep;
        std::vector<int> legalScratch;

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
            /* 势能塑形 (Phase 2): 首手之前那个局面的势能, 是第 0 步的 Φ_before */
            m_phiInit = potentialOf(currentColor);

            /* Encode current state */
            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /*
               Run MCTS to get improved policy.
               B-5: 不再 `nodes.clear()` —— 向置换表要根: 上一步我们落子到达的那个局面
               就在上一棵树里, 它的子树/先验/访问计数直接接着用 (见 acquireRoot)。

               ---- 自对弈取数据: 这里开根节点 Dirichlet 噪声 (2026-09) ----
               这是**唯一**开噪声的入口 (理由见头文件 rootNoise 一节): 这套搜索的展开是
               确定性的 (按先验挑最大), 若再没有根噪声, 低先验着法 (典型是吃子) 可能
               整局都不会被模拟到, 搜索拿不到它的 Q, 网络就永远学不到它。
               selectMove (评测/对局) 与 warmupFromCurrent 都保持不开 —— 量棋力时不能
               掺探索噪声, 而且必须与改动前可比。
            */
            const int rootID = acquireRoot(currentColor, /*withRootNoise=*/true);

            if (rootID < 0) {
                /* 没有合法走法: 当前走子方输 (acquireRoot 返回 -1, 见那里的说明) */
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
                    commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
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
                break;
            }

            /*
               R2: 记下这一手局面的**完整合法集**, 与这一手进轨迹的样本一一对应
               (展开只是把着法从 untried 挪进 childIDs, 并集不变, 所以什么时候取都一样)。
            */
            legalIndicesOf(rootID, legalScratch);
            legalPerStep.push_back(legalScratch);

            /* 每个模拟复用同一对缓存, 避免在循环里反复分配 (STATE_DIM 已经上千维) */
            RL::Tensor parentState(STATE_DIM, 1);
            RL::Tensor leafState(STATE_DIM, 1);
            /* R1: 稀疏先验的三个复用缓冲 (合法动作集 / 概率 / 选中动作的先验) */
            std::vector<int> legalIdxScratch;
            std::vector<float> probsScratch;
            float priorScratch = 0.0f;

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
                    /* Phase 4.1 + R1: 先求父节点策略 (稀疏: 只算合法列), 再**按先验**
                       挑未展开着法 (原来这里是 std::rand() 随机挑, 先验没参与展开)。 */
                    encodeState(parentState);
                    const int moveIdx = pickUntriedByPrior(nodeID, parentState,
                                                           legalIdxScratch, probsScratch,
                                                           priorScratch);
                    int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
                    Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];
                    float prior = priorScratch;
                    if (prior < 1e-9f) prior = 1e-9f;

                    nodes[nodeID].untriedActionIndices.erase(
                        nodes[nodeID].untriedActionIndices.begin() + moveIdx);
                    nodes[nodeID].untriedSteps.erase(
                        nodes[nodeID].untriedSteps.begin() + moveIdx);

                    /* 边的先验来自**父节点**的策略, 且必须与动作下标处在同一个
                       规范视角 —— 详细理由见 selectMove 里同一处的长注释。 */

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK
                                        : Stone::COLOR_RED;

                    AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

            /*
               B-5: 记下这个子局面 (棋子位置 + 走棋方) 的置换表键并登记进去 ——
               下一步真的落子走到这里时, 整棵子树连同访问计数/Q 会被直接复用
               (自对弈里必然命中: 那边两边都是同一个 agent)。键必须现在算: 棋盘刚走完
               chosenStep, 正停在这个子局面上。
            */
            newNode.hash = chess.computeHash();
            newNode.depth = nodes[nodeID].depth + 1;
            if (treeReuse && newNode.depth <= ttMaxDepth) {
                m_tt[newNode.hash] = (int)nodes.size();   /* push_back 之后就是它的下标 */
            }
            m_nodesCreated++;

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
                /* 叶子估值统一走 evaluateLeaf (终局给真实 ±1, 见那个函数的注释) */
                double reward = evaluateLeaf(leafState);

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
            /*
               出招统一走 pickRootChildByVisits (2026-09)。原来这里内联了一整份
               "temp>0.1 就按 N^(1/temp) 采样、否则 argmax" 的实现 —— 而
               selectMove() 的 temp 参数从未被读取, 于是同一个语义在两条路径上行为不同。
               抽出去之后两边不可能再漂移; 采样的数学与随机数调用顺序**逐位保持不变**
               (同一个稠密张量、同一次 categorical), 所以自对弈的可复现性不受影响。
            */
            Step chosenStep;
            int chosenAction = -1;

            const int chosenID = pickRootChildByVisits(rootID, temp);
            if (chosenID >= 0) {
                chosenStep = nodes[(std::size_t)chosenID].step;
                chosenAction = nodes[(std::size_t)chosenID].parentAction;

                /* Execute move on the board */
                double dummyReward = 0.0;
                chess.moveForward(&chosenStep, dummyReward);

                /* Check game over —— Phase 6: 统一走 getResult()
                   (一次覆盖 将杀/困毙/吃将/三次重复/60 回合判和; 原来 isGameOver()
                   只认"将不在了", 于是将杀与判和都不会让这一局结束)。 */
                int gameResult = chess.getResult(chess.sideToMove);
                if (gameResult != Chess::RESULT_ONGOING) {
                    /*
                       终局值必须按**最后一步走子方**的视角给 —— 此刻走子方就是
                       currentColor (还没翻转)。原先各处按黑方视角算, 与走子方视角的
                       即时奖励、规范视角的状态编码三者不一致 ⇒ 红方赢的棋对红方
                       反而成了"在输" (不报错, 只是学不动)。
                    */
                    const float outcomeForLastMover =
                        outcomeForMover(gameResult, currentColor);

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
                    /* 势能塑形 (Phase 2): 记下落子**之后**局面的势能 Φ(s_{i+1}) */
                    trajectory.back().potential = potentialOf(chess.sideToMove);

                    /* Train PPO on complete trajectory */
                    if (!trajectory.empty()) {
                        commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
                    }

                    totalEpisodes++;
                    /*
                       统计必须走 winnerOfResult(): `gameResult` 是 Chess::Result, 而
                       RESULT_RED_WIN(1) 与 COLOR_BLACK(1) 数值撞号 —— 直接比较会把红胜
                       记成黑胜, 黑胜则谁都匹配不上 (胜率面板一直是错的, 且不报错)。
                    */
                    const int winnerColor = winnerOfResult(gameResult);
                    if (winnerColor == Stone::COLOR_BLACK) totalWins[1]++;
                    if (winnerColor == Stone::COLOR_RED) totalWins[0]++;

                    if (verbose && (ep % printInterval == 0 || ep == episodes - 1)) {
                        printf("  Episode %4d/%d: %s, %d moves, win_rate=%.2f\n",
                               ep + 1, episodes,
                               (winnerColor == Stone::COLOR_BLACK) ? "Black(AI) wins"
                             : (winnerColor == Stone::COLOR_RED)   ? "Red wins"
                                                                   : "Draw",
                               moveNum + 1, getWinRate(Stone::COLOR_BLACK));
                    }
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
                /* 势能塑形 (Phase 2): 落子后局面的势能 Φ(s_{i+1}) */
                trajectory.back().potential = potentialOf(chess.sideToMove);

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
                    commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
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
                break;
            }
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
                /*
                   ---- 截断局的自举 (2026-09) ----
                   原来一律传 0.0f, 等于告诉 critic "这些局面的回报就是 0"。而象棋和棋
                   极多、手数上限又低 (GUI 训练只给 60 手), 于是大部分轨迹的价值目标都是
                   0, critic 只学得到"和棋", 搜索的叶子估值也就没有区分度。
                   现在用 critic 对"最后一步之后局面"的估值自举 —— 口径与在线路径
                   endOnline 完全一致 (那边一直这么做), 两条路的学习问题不再不同。
                   truncationBootstrap=false 可切回旧口径做 A/B。
                */
                const float bootOutcome = truncationBootstrap ? bootstrapOutcome() : 0.0f;
                commitEpisode(trajectory, bootOutcome, &legalPerStep);
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
        /* B-5: 每局清一次树 (一局之内 subtree 跨 ply 复用), 同 trainSelfPlay */
        resetSearchTree();

        /* ---- Play one episode with MCTS-guided PPO ---- */
        int currentColor = Stone::COLOR_BLACK;
        std::vector<RL::Step> trajectory;
        /* R2: 每一手局面的完整合法着法集 (与 trajectory 同步) */
        std::vector<std::vector<int>> legalPerStep;
        std::vector<int> legalScratch;

        for (int moveNum = 0; moveNum < maxMoves; moveNum++) {
            /* 规范视角靠 chess.sideToMove, 而 Chess::reset() 把它置成 RED、本函数却
               从 BLACK 开始走 —— 不对齐的话第一步会用**对手**的视角编码 (同 trainSelfPlay) */
            chess.sideToMove = currentColor;
            /* 势能塑形 (Phase 2): 首手之前那个局面的势能, 是第 0 步的 Φ_before */
            m_phiInit = potentialOf(currentColor);

            RL::Tensor state(STATE_DIM, 1);
            encodeState(state);

            /* ---- MCTS: Selection + Expansion + Backprop ----
               B-5: 不再每 ply `nodes.clear()`, 改成向置换表要根 (见 acquireRoot);
               本局的树在每局开头由 resetSearchTree() 清过一次。 */
            const int rootID = acquireRoot(currentColor);

            if (rootID < 0) {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                /* 最后一手是赢家走的 -> 最后一步走子方 = 赢家 -> 终局值 +1
                   (同 trainSelfPlay 那两处修正, 原来这里是黑方视角) */
                const float outcomeForLastMover = 1.0f;
                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                break;
            }

            /* R2: 这一手局面的完整合法集 (与 trajectory 一一对应) */
            legalIndicesOf(rootID, legalScratch);
            legalPerStep.push_back(legalScratch);

            /* 每个模拟复用同一对缓存, 避免在循环里反复分配 (STATE_DIM 已经上千维) */
            RL::Tensor parentState(STATE_DIM, 1);
            RL::Tensor leafState(STATE_DIM, 1);
            /* R1: 稀疏先验的三个复用缓冲 (合法动作集 / 概率 / 选中动作的先验) */
            std::vector<int> legalIdxScratch;
            std::vector<float> probsScratch;
            float priorScratch = 0.0f;

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
                    /* Phase 4.1 + R1: 先求父节点策略 (稀疏: 只算合法列), 再**按先验**
                       挑未展开着法 (原来这里是 std::rand() 随机挑, 先验没参与展开)。 */
                    encodeState(parentState);
                    const int moveIdx = pickUntriedByPrior(nodeID, parentState,
                                                           legalIdxScratch, probsScratch,
                                                           priorScratch);
                    int chosenAction = nodes[nodeID].untriedActionIndices[moveIdx];
                    Step chosenStep = nodes[nodeID].untriedSteps[moveIdx];
                    float prior = priorScratch;
                    if (prior < 1e-9f) prior = 1e-9f;

                    nodes[nodeID].untriedActionIndices.erase(
                        nodes[nodeID].untriedActionIndices.begin() + moveIdx);
                    nodes[nodeID].untriedSteps.erase(
                        nodes[nodeID].untriedSteps.begin() + moveIdx);

                    /* 边的先验来自**父节点**的策略, 同一规范视角 (理由见 selectMove) */

                    chess.moveForward(&chosenStep, dummyReward);

                    int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                        ? Stone::COLOR_BLACK : Stone::COLOR_RED;

                    AZNode newNode(nodeID, chosenAction, chosenStep, prior, nextColor);

            /*
               B-5: 记下这个子局面 (棋子位置 + 走棋方) 的置换表键并登记进去 ——
               下一步真的落子走到这里时, 整棵子树连同访问计数/Q 会被直接复用
               (自对弈里必然命中: 那边两边都是同一个 agent)。键必须现在算: 棋盘刚走完
               chosenStep, 正停在这个子局面上。
            */
            newNode.hash = chess.computeHash();
            newNode.depth = nodes[nodeID].depth + 1;
            if (treeReuse && newNode.depth <= ttMaxDepth) {
                m_tt[newNode.hash] = (int)nodes.size();   /* push_back 之后就是它的下标 */
            }
            m_nodesCreated++;

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
                /* 叶子估值统一走 evaluateLeaf (终局给真实 ±1, 见那个函数的注释) */
                double reward = evaluateLeaf(leafState);

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

                int gameResult = chess.getResult(chess.sideToMove);
                if (gameResult != Chess::RESULT_ONGOING) {
                    /* 终局值按最后一步走子方 (= currentColor, 还没翻转) 的视角 ——
                       与 trainSelfPlay 同一处修正, Phase 6 起统一走 getResult() */
                    const float outcomeForLastMover =
                        outcomeForMover(gameResult, currentColor);
                    /* 策略目标 = 根节点的访问分布 (同 trainSelfPlay) */
                    RL::Tensor policyTarget(ACTION_DIM, 1);
                    if (!visitDistribution(rootID, policyTarget)) {
                        policyTarget.zero();
                        if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                    }
                    trajectory.emplace_back(state, policyTarget,
                                            computeReward(chosenStep, currentColor));
                    /* 势能塑形 (Phase 2): 记下落子**之后**局面的势能 Φ(s_{i+1}) */
                    trajectory.back().potential = potentialOf(chess.sideToMove);
                    if (!trajectory.empty()) {
                        commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
                    }
                    totalEpisodes++;
                    /* 同 trainSelfPlay: 必须用 winnerOfResult 换算 (Chess::Result 与
                       Stone::Color 数值错位, 直接比较会把红胜记成黑胜) */
                    const int winnerColor = winnerOfResult(gameResult);
                    if (winnerColor == Stone::COLOR_BLACK) totalWins[1]++;
                    if (winnerColor == Stone::COLOR_RED) totalWins[0]++;
                    break;
                }

                RL::Tensor policyTarget(ACTION_DIM, 1);
                if (!visitDistribution(rootID, policyTarget)) {
                    policyTarget.zero();
                    if (chosenAction >= 0) policyTarget[chosenAction] = 1.0f;
                }
                float reward = computeReward(chosenStep, currentColor);
                trajectory.emplace_back(state, policyTarget, reward);
                /* 势能塑形 (Phase 2): 落子后局面的势能 Φ(s_{i+1}) */
                trajectory.back().potential = potentialOf(chess.sideToMove);

                currentColor = (currentColor == Stone::COLOR_RED)
                                   ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            } else {
                int winner = (currentColor == Stone::COLOR_RED)
                                 ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                /* 最后一手是赢家走的 -> 最后一步走子方 = 赢家 -> 终局值 +1
                   (同 trainSelfPlay 那处修正) */
                const float outcomeForLastMover = 1.0f;
                if (!trajectory.empty()) {
                    commitEpisode(trajectory, outcomeForLastMover, &legalPerStep);
                }
                totalEpisodes++;
                if (winner == Stone::COLOR_BLACK) totalWins[1]++;
                if (winner == Stone::COLOR_RED) totalWins[0]++;
                break;
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
            if (!trajectory.empty()) {
                /*
                   ---- 截断局的自举 (2026-09) ----
                   原来一律传 0.0f, 等于告诉 critic "这些局面的回报就是 0"。而象棋和棋
                   极多、手数上限又低 (GUI 训练只给 60 手), 于是大部分轨迹的价值目标都是
                   0, critic 只学得到"和棋", 搜索的叶子估值也就没有区分度。
                   现在用 critic 对"最后一步之后局面"的估值自举 —— 口径与在线路径
                   endOnline 完全一致 (那边一直这么做), 两条路的学习问题不再不同。
                   truncationBootstrap=false 可切回旧口径做 A/B。
                */
                const float bootOutcome = truncationBootstrap ? bootstrapOutcome() : 0.0f;
                commitEpisode(trajectory, bootOutcome, &legalPerStep);
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
    /*
       内核的 save 自己做了原子写 + 落盘检查, 现在也会返回真实结果 (见 rl/ppo.cpp)。
       两个判据都用上: 内核结果为主, "文件是否真的在且非空"作为兜底。
    */
    const bool ok = ppo.save(actorPath, criticPath);
    return ok && weightFileWritten(actorPath) && weightFileWritten(criticPath);
}

bool PPOMCTSAgent::loadModel(const std::string &actorPath,
                             const std::string &criticPath)
{
    if (!weightFileReadable(actorPath) || !weightFileReadable(criticPath)) {
        return false;
    }
    /*
       ---- 必须传播内核的真实结果 (2026-09 修) ----
       内核的 load 在"层数 / 结构指纹 / CRC 不匹配"时会**拒绝载入并保持网络不变**, 但
       原来这里丢弃了它的返回值, 只按"文件可读"就返回 true —— 于是**过期或损坏的检查点
       被静默忽略, 同时报告成功**。对"每轮保存 -> 载入 -> 继续训"的循环这是最坏的失败
       形态: 静默退化成每次从随机权重重来, 表现就是"跑了很多轮完全没有效果", 而且
       一句报错都没有 (实测: bench_ppo_sims 打印 "A 成功 / B 成功", 而 stderr 同时
       在喊"参数量不匹配 ... 拒绝载入")。
    */
    if (!ppo.load(actorPath, criticPath)) {
        /* 载入失败 ⇒ 网络没变 ⇒ 树仍然有效, 不 resetSearchTree() */
        return false;
    }
    /*
       B-5: 换了权重就把树丢掉 —— 树里的先验与 Q 全是旧网络算出来的, 复用它们等于
       拿旧策略继续搜。这里与"新一局"是同一类失效条件。
    */
    resetSearchTree();
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
    /* B-5: 新的一局 (人机) —— 上一局的搜索树作废, 否则同一个初始局面会命中旧树 */
    resetSearchTree();
    /* 势能塑形 (Phase 2): 线上路径同样需要首手之前的势能 */
    m_phiInit = potentialOf(chess.sideToMove);
}

void PPOMCTSAgent::recordOnline(const Step& s, int color, const RL::Tensor& stateBefore)
{
    RL::Tensor oneHotAction(ACTION_DIM, 1);
    oneHotAction.zero();
    int aidx = stepToActionIdx(s, color);
    oneHotAction[aidx] = 1.0f;
    float reward = computeReward(s, color);
    m_onlineTrajectory.emplace_back(stateBefore, oneHotAction, reward);
    /* 势能塑形 (Phase 2): 线上路径的 Φ(s_{i+1})(调用方在落子之后才记录) */
    m_onlineTrajectory.back().potential = potentialOf(chess.sideToMove);
}

void PPOMCTSAgent::endOnline(int winner, int myColor)
{
    float finalOutcome = 0.0f;
    if (winner == myColor) finalOutcome = 1.0f;
    else if (winner != Stone::COLOR_NONE) finalOutcome = -1.0f;

    if (!m_onlineTrajectory.empty()) {
        /* 势能塑形 (Phase 2): 与自对弈路径同一套 */
        applyPotentialShaping(m_onlineTrajectory);
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

       R1 (2026-09): 这里本来就是"全量策略 -> 只在合法集上归一 -> 采样", 与
       `ppo.actionMasked` 是**同一件事** (它内部就是子集 softmax), 所以直接换成稀疏
       路径: 每步省掉策略头 2.07 MB 的权重读取 (在线路径一次 rollout 有几十步)。
       采样只在合法集上做 (紧凑向量), 数学上与原写法一致 (原写法在 8100 维上采样,
       非法槽位已被置 0)。
    */
    auto pick = [this](const RL::Tensor &state, int turn) -> int {
        std::vector<Step*> legal;
        std::vector<int> actionIndices;
        RL::Tensor mask(ACTION_DIM, 1);
        getLegalActions(turn, legal, actionIndices, mask);
        Steps::instance().put(legal);
        if (actionIndices.empty()) {
            return -1;   /* 没有合法走法: rolloutFromCurrent 会用第一个合法走法兜底 */
        }

        std::vector<float> probs;
        if (sparsePolicyHead) {
            if (!ppo.actionMasked(state, actionIndices, probs) || probs.empty()) {
                return -1;
            }
        } else {
            /* R1 之前的对照口径 (A/B 用): 全量策略 -> 只在合法集上归一 */
            RL::Tensor policy = ppo.action(state);
            probs.resize(actionIndices.size());
            double sum = 0.0;
            for (std::size_t k = 0; k < actionIndices.size(); k++) {
                const float p = policy[(std::size_t)actionIndices[k]];
                probs[k] = p;
                sum += (double)p;
            }
            if (sum > 1e-12) {
                for (std::size_t k = 0; k < probs.size(); k++) {
                    probs[k] = (float)((double)probs[k] / sum);
                }
            }
        }
        if (probs.empty()) {
            return -1;
        }
        RL::Tensor compact(probs.size(), 1);
        for (std::size_t k = 0; k < probs.size(); k++) {
            compact[k] = probs[k];
        }
        const int pos = RL::Random::categorical(compact);
        if (pos < 0 || (std::size_t)pos >= actionIndices.size()) {
            return actionIndices[0];
        }
        return actionIndices[(std::size_t)pos];
    };

    std::vector<RL::Step> traj;
    traj.reserve((std::size_t)rolloutSteps);
    /*
       只记"轨迹是否真的走到了终局"以及那一手的奖励。终局值必须来自终局本身,
       不能拿任意一手(可能是中局)的即时奖励冒充。
    */
    bool rolloutEnded = false;
    float terminalReward = 0.0f;
    /*
       自举 (Phase 2) 用: 留下最后一手的 nextState —— 截断时用它的价值当头。
       Tensor 赋值是深拷贝, 所以这里存下来不会被后续 rollout 覆盖。
    */
    RL::Tensor lastNextState;
    bool haveNextState = false;
    auto onTrans = [this, &traj, &rolloutEnded, &terminalReward,
                    &lastNextState, &haveNextState](
                       const Step &/*chosen*/, int actionIdx,
                       const RL::Tensor &s, const RL::Tensor &ns,
                       float r, bool done) {
        RL::Tensor oneHot(ACTION_DIM, 1);
        oneHot.zero();
        oneHot[actionIdx] = 1.0f;
        traj.emplace_back(s, oneHot, r);
        /*
           势能塑形 (Phase 2): agentrollout 是**先落子再回调**, 所以此刻棋盘的
           sideToMove 已经是走子方的对手 —— potentialOf(chess.sideToMove) 正是
           Φ(s_{i+1}) (落子后局面的势能), 与 Step::potential 的定义一致。
        */
        traj.back().potential = potentialOf(chess.sideToMove);
        lastNextState = ns;
        haveNextState = true;
        if (done) {
            /* agentrollout.hpp 在 done 时把 r 换成了 (gameResult == turn) ? 1 : -1,
               也就是**走子方视角**的终局结果 —— 正是 learnSelfPlay 需要的口径 */
            rolloutEnded = true;
            terminalReward = r;
        }
    };

    /* 首手之前的势能 (第 0 步的 Φ_before); 必须在下棋之前取 */
    m_phiInit = potentialOf(chess.sideToMove);
    const int collected = rolloutFromCurrent(*this, chess, color, rolloutSteps, pick, onTrans);

    bool trained = false;
    if (!traj.empty()) {
        /*
           finalOutcome 的口径是"最后一步走子方视角" (见 rl/ppo.h)。

           走到终局时: 用 agentrollout 给的那个走子方视角的 ±1 ✓
           **没走到终局时: 用自举** —— V(s_end+1) 是"轮到走棋的一方(即最后一步
           走子方的对手)"的价值, 取负就换到最后一步走子方的视角。

           原来是传 0 (截断)。那等于假设"此后双方均势", 而材质奖励缩小 20 倍之后
           "0" 就等于"没有信号": 诊断实测改动前 |value target|>0.1 的样本占比 0%,
           critic 只能学成一个常数 —— 这是"安静局面没有位置感"的直接原因。
        */
        float finalOutcome = 0.0f;
        if (rolloutEnded) {
            /*
               终局常量要跟着势能一起平移 (PBRS 的边界项): 见 commitEpisode 里同一处
               的推导 —— finalOutcome' = finalOutcome - Φ(落子后局面)。
            */
            finalOutcome = terminalReward - finalPhiScaled(traj);
        } else if (haveNextState) {
            /*
               自举: V(s_end+1) 是"轮到走棋的一方(即最后一步走子方的对手)"的价值,
               取负就换到最后一步走子方的视角。
               **注意这里不需要再做势能平移** —— 价值头学到的目标已经是 V + Φ
               (塑形后的口径), 所以 -V'(s_end+1) 本身就是平移过的终局值。
            */
            finalOutcome = -(float)ppo.value(lastNextState);
            /* 价值头理论上在 (-1,1); 夹一下防止未收敛时给出离谱的自举值 */
            finalOutcome = std::max(-1.0f, std::min(1.0f, finalOutcome));
        }
        /* 势能塑形 (Phase 2): 原地写进 reward, 再交给 RL 内核算折现回报 */
        applyPotentialShaping(traj);
        ppo.learnSelfPlay(traj, finalOutcome, learningRate);
        trained = true;
    }
    m_exploreInfo = "rollout " + std::to_string(collected) + " 步, PPO 更新 1 次"
                    + (rolloutEnded ? " (到终局)" : " (截断+自举)");
    return trained;
}
