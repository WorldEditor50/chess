#ifndef PPOMCTS_AGENT_H
#define PPOMCTS_AGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include "chess.h"
#include "aiagent.h"
#include "rl/ppo.h"
#include "rl/diag.h"

/*
 * PPOMCTSAgent - AlphaZero-style PPO+MCTS Chess Agent
 *
 * Combines Proximal Policy Optimization (PPO) neural network with
 * Monte Carlo Tree Search (MCTS), following the AlphaZero paradigm:
 *
 *   - PPO's actor network provides policy priors P(s,a) for MCTS
 *   - PPO's critic network provides state value estimates V(s)
 *   - MCTS uses PUCT formula for selection:
 *       U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *   - Instead of random rollouts, evaluation uses the PPO value head
 *   - After MCTS, the improved policy target is proportional to visit counts
 *
 * ==========================================================================
 *  2026-09 改版 (三处)
 * ==========================================================================
 *
 * 1) 状态编码: 90 维标量棋盘 -> **16 个平面 x 90 格 = 1440 维**
 *
 *    原来是"每格一个数"(`0=空 / ±棋子类型值`)。那等于把 14 种棋子塞进 1 个通道,
 *    既表达不了"这个格子上同时是哪一方的什么子"之外的任何结构, 也表达不了任何
 *    派生信息。现在改成平面 (plane) 编码:
 *
 *      plane 0..13 : 7 类棋子 x {己方, 对方}, 即 `type*2 + (是己方 ? 0 : 1)`
 *      plane 14    : 该格**被对方攻击** (威胁平面)
 *      plane 15    : 该格有**己方棋子且被对方攻击** (己方子受威胁)
 *
 *    并且整盘按**规范视角**镜像: 轮到黑方时把 x 反射成 9-x。于是同一个网络对红黑
 *    双方看到的都是"我方的子在自己的下方、对方在上面", 不需要再从非对称编码里
 *    反推"该谁走" —— 这也是 AlphaZero 必须做的规范化。本工程 EVABAgent 与
 *    SACAZAgent 的 1260 维编码用的是同一约定 (那两个只有 14 个平面, 没有威胁平面)。
 *
 * 2) 动作编码: 128 维哈希 -> **8100 维无碰撞的 (from, to)**
 *
 *    原来 `stepToActionIdx` 是 `(id*37 + x*13 + y*7) % 128`: 2880 种可能落进
 *    128 个槽位, 平均 22 个不同的走法共用一个槽位。后果是策略目标被摊平到错误
 *    的槽位上 (落子本身仍正确 —— 选择走法是按节点访问数取的, 不是按动作索引)。
 *    现在换成双射:
 *
 *      actionIdx = fromCell * 90 + toCell      (cell = x*9 + y, 规范视角)
 *
 *    8100 = 90x90, 覆盖中国象棋所有可能的 (起点, 终点), 且**与棋子 id 无关**
 *    (棋子的 id 会随吃子变化, 用它做索引会让"同一个走法"在不同局面下不稳定)。
 *    与状态一样按规范视角镜像, 所以红黑双方的"同一步棋"共享同一个动作槽位。
 *
 * 3) 网络骨干: 稠密 MOE -> 稀疏 MoE (见 rl/ppo.h)
 *
 * ==========================================================================
 *  注意: 改动 2 牵出了搜索里的一个既有缺陷, 一并修了
 * ==========================================================================
 *
 * 边 (parent -> child) 的先验必须来自**父节点**的策略:
 *     P(parent, a) = pi_theta(s_parent)[a]
 * 原来的代码是拿**子节点**的策略来索引父节点的动作
 * (`childPolicy[chosenAction]`) —— 那既用错了网络 (子节点的策略描述的是下一手
 * 走棋方的选择), 也用错了视角。在旧的"绝对坐标 + 哈希"编码下两者恰好共用同一个
 * 坐标帧, 所以这个错误只表现为先验质量差; 换成规范视角之后两个帧会直接错开
 * (父节点按红方视角、子节点按黑方视角), 先验会变成完全无关的数。现在两处搜索
 * (selectMove 与 trainSelfPlay) 都改成在**展开前**求父节点策略。
 *
 * 顺带去掉了一次纯浪费的前向: 原来每个模拟要算 2 次 actor + 2 次 critic
 * (子节点策略 + 叶子估值, 而"子节点"和"叶子"就是同一个局面 —— 算了两次),
 * 现在每个模拟是 1 次 actor (父节点策略) + 1 次 critic (叶子价值)。
 */
class PPOMCTSAgent : public AgentBase
{
public:
    /* ----------------------------------------------------------------
     *  状态编码: 19 个平面 x 90 格  (14 棋子 + 2 威胁 + 3 规则上下文)
     *
     *  规则上下文那 3 个平面是本轮从 DQNAB 推广过来的（见 src/chessstate.h）：
     *  象棋是双人零和、完全信息、交替行动的 Markov Game，而**裸棋盘 + 轮到谁不是
     *  Markov 状态** —— 三次重复判和、60 回合无吃子判和、长将/循环都依赖历史，
     *  而它们决定终局。只喂棋子平面时，「同一局面的第 2 次出现」与「第 3 次出现
     *  （立刻判和）」会编码成**同一个向量**，那 V(s) 就不是 s 的函数。
     * ---------------------------------------------------------------- */
    static constexpr int CELLS = 90;                 /* 10 行 x 9 列 */
    static constexpr int PIECE_TYPES = 7;
    static constexpr int PIECE_PLANES = PIECE_TYPES * 2;   /* 7 类 x {己方, 对方} */
    static constexpr int PLANE_ATTACKED = PIECE_PLANES;          /* 14: 被对方攻击 */
    static constexpr int PLANE_UNDER_ATTACK = PIECE_PLANES + 1;  /* 15: 己方子被攻击 */
    /* 规则上下文: 无吃子进度 / 重复次数 / 将军 (顺序与 ChessState::CTX_* 一致) */
    static constexpr int PLANE_HALFMOVE = PIECE_PLANES + 2;      /* 16 */
    static constexpr int PLANE_REPEAT   = PIECE_PLANES + 3;      /* 17 */
    static constexpr int PLANE_CHECK    = PIECE_PLANES + 4;      /* 18 */
    static constexpr int PLANES = PIECE_PLANES + 5;              /* 19 */
    static constexpr int STATE_DIM = PLANES * CELLS;             /* 1710 */

    /* ----------------------------------------------------------------
     *  动作编码: from*90 + to (无碰撞, 规范视角)
     * ---------------------------------------------------------------- */
    static constexpr int ACTION_DIM = CELLS * CELLS;             /* 8100 */

public:
    Chess &chess;
    RL::PPO ppo;                    /* PPO: actorP (policy), critic (value) */
    float gamma;
    float learningRate;
    float c_puct;                   /* PUCT exploration constant */
    int expertHidden;               /* 稀疏 MoE 的 MLP 专家隐层宽度 (只在把
                                       PPOExpert 换回 MlpExpert 时才有效;
                                       TransformerBlock 专家的 FFN 宽度由
                                       PPO_MOE_TB_DFF 决定, 见 rl/ppo.h) */
    float moeAuxCoef;               /* 稀疏 MoE 负载均衡辅助损失系数 */

    /* Training statistics */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red, [1]=black */

    /* Online training trajectory */
    std::vector<RL::Step> m_onlineTrajectory;

    /* ----------------------------------------------------------------
     *  AlphaZero-style MCTS Node
     * ---------------------------------------------------------------- */
    struct AZNode {
        int parentID;                /* index of parent (-1 for root) */
        int parentAction;            /* action index that led to this node */
        Step step;                   /* the move that led to this node */

        /* ----------------------------------------------------------------
         *  B-5: 置换表的键与深度
         * ----------------------------------------------------------------
         *  hash  : 这个节点对应局面的 Zobrist 键 (`Chess::computeHash()`, 已含走棋方)。
         *          创建子节点时算一次 (32 次 XOR, 便宜), 用它把节点登记进置换表,
         *          下一次搜索同一个局面时就能**直接复用这棵子树**。
         *  depth : 相对**当前根**的深度 (根 = 0)。只用来决定"哪些节点进置换表"
         *          (浅层节点的统计更可信、复用价值也更高)。
         * ---------------------------------------------------------------- */
        unsigned long long hash;
        int depth;

        /* Tree statistics */
        int visitCount;
        double totalValue;           /* sum of value estimates */
        double prior;                /* PPO policy prior P(s,a) */

        /* Children */
        std::vector<int> childIDs;
        std::vector<int> untriedActionIndices; /* actions not yet expanded */
        std::vector<Step> untriedSteps;

        /* Current color at this node */
        int currentColor;

        AZNode()
            : parentID(-1), parentAction(-1), hash(0), depth(0),
              visitCount(0), totalValue(0.0), prior(0.0),
              currentColor(Stone::COLOR_NONE) {}
        AZNode(int pid, int pa, const Step &st, float p, int color,
               unsigned long long h = 0, int d = 0)
            : parentID(pid), parentAction(pa), step(st), hash(h), depth(d),
              visitCount(0), totalValue(0.0), prior(p),
              currentColor(color) {}

        double getQ() const {
            return visitCount > 0 ? totalValue / (double)visitCount : 0.0;
        }
    };

    std::vector<AZNode> nodes;

    /* ================================================================
     *  B-5: 置换表 + 子树复用 (2026-09)
     * ================================================================
     *  在这之前每 ply 都是 `nodes.clear()` —— 每一步都从零重搜。而搜索的绝大部分
     *  结果本来就是**可以留给下一步**的: 走子之后, 我们落子到达的那个局面正是上一棵
     *  树里的一个子节点 (自对弈里 100% 命中), 它的子节点、先验、访问计数、Q 估计
     *  全都还在。把它们丢掉等于每一步都白扔一次搜索。
     *
     *  这里做两件事:
     *   1. **置换表**: `hash -> 节点下标` 的映射 (键是 `Chess::computeHash()`, 已含
     *      走棋方)。子节点创建时登记, 下次搜索同一局面直接命中。
     *   2. **子树复用**: 命中之后把那个节点**当作新的根**, 它的子树与统计全部继续用
     *      (PUCT 从继承来的访问计数/Q 上继续, 而不是从 0 开始)。
     *
     *  为什么安全 (以及哪些地方是刻意的取舍):
     *   * **同一局内**复用: 网络权重在一局里不变, 先验仍然有效。
     *   * **跨局必须失效**: 命中只有在"从上一棵树的根往下 ≤ 2 步可达"时才接受
     *     (自对弈里"我们的着法子节点"= 1 步、"对手的应着"= 2 步)。于是上一局的节点
     *     ——包括每个新对局都会遇到的初始局面——**必然不可达 ⇒ 新开一棵树**,
     *     不会把上一局的访问计数带进新一局。另外 `trainSelfPlay` / `warmupFromCurrent`
     *     在每局开头显式 `resetSearchTree()`, `loadModel()` 也会重置 (换了权重就作废)。
     *   * **重复局面/60 回合**: Zobrist 键只看棋子位置与走棋方, 不含 halfMoveClock 与
     *     历史。树内本来就不做终局判定 (叶子用 critic 估值), 所以这与既有口径一致;
     *     真正判和仍由 `Chess::getResult()` 在 rollout/自对弈那一层负责。
     *   * **策略目标会变** (这是一处**行为变化**, 不是等价优化): 自对弈的策略目标
     *     π ∝ 根节点访问计数, 而复用之后这些计数**跨 ply 累积** (AlphaZero 的标准做法,
     *     如 ELF/Leela)。所以等模拟数下的目标分布更尖。要关掉整个机制就用
     *     `treeReuse = false` 做 A/B。
     *   * 内存: 复用让同一局里的树**越滚越大** (这正是收益来源), 所以有 `treeNodeCap`
     *     兜底 —— 超了就把整棵树和置换表一起丢掉重来。
     * ================================================================ */
    bool treeReuse = true;
    std::size_t treeNodeCap = 50000;   /* 节点数上限 (超过就整棵重来) */
    int ttMaxDepth = 12;               /* 只把深度 ≤ 它的节点登记进置换表 */

    /* ================================================================
     *  根节点 Dirichlet 噪声 (AlphaZero 的探索机制, 2026-09 补)
     * ================================================================
     *  为什么必须有它: 这套搜索的"扩展开关"是**确定性**的 (按先验挑最大的未展开着法),
     *  而象棋终局判罚又让和棋极多。没有根噪声时, 自对弈的多样性只剩"按访问分布采样"
     *  这一条 —— 而访问分布本身还是被同一份先验决定的。于是低先验的着法 (典型是吃子)
     *  可能**整局都不会被模拟一次**, 搜索永远拿不到它的 Q, 网络也就永远学不到它
     *  (这正是"越训越窄 / 开局塌成一条线"的直接机制之一)。
     *
     *  做法 (标准 AlphaZero): 只在**自对弈取数据**时把根先验改成
     *      P' = (1-eps)·P + eps·Dir(alpha)
     *  并按手数把 eps 线性退火到 0 (开局探索、残局收敛)。
     *  **评测 / 对局一律不开** (selectMove 走 withRootNoise=false) —— 否则量出来的棋力
     *  会被探索噪声污染, 而且改动前后不可比。
     *
     *  取值: alpha 与合法着法数有关, 象棋中局约 40 个候选, 取 0.4 (既不至于退化成
     *  均匀分布、也不会全压在一两个着法上); eps=0.25 是 AlphaZero 的常用值;
     *  只在前 rootNoiseMoves 手加 (之后 eps 退到 0 自动关闭)。
     * ================================================================ */
    bool rootNoise = true;
    float rootNoiseAlpha = 0.4f;
    float rootNoiseEps = 0.25f;
    int rootNoiseMoves = 30;           /* 只在前 N 手加噪声 (之后自动关闭) */
    /*
       评测路径 (selectMove) 是否也加根噪声。**默认 false**, 打开只为做诊断 A/B:
       诊断矩阵里的"Dirichlet 有效性"就是"关掉 η 重跑同一个根, 比较吃子着的访问数变化"
       —— 加了噪声吃子 N 从 0 涨起来 = 噪声在救命; 加了也没用 = leaf Q 把吃子压得太死,
       回去修 value。生产路径 (对局/评测) 必须保持 false, 否则量出来的棋力掺了探索噪声。
    */
    bool evalRootNoise = false;
    /* 根先验的"加噪后"版本 (按动作下标索引); 只在 m_rootNoiseActive 时被采用 */
    std::vector<float> m_rootPriorNoised;
    bool m_rootNoiseActive = false;
    int m_plyInGame = 0;               /* 本局已搜索过的手数 (噪声退火用; resetSearchTree 归零) */

    /* 新一局 / 换权重: 清空树与置换表 (计数器一并归零, 便于诊断) */
    void resetSearchTree();
    /* 当前根在 `nodes` 里的下标 (-1 = 还没有树) */
    int currentRoot() const { return m_rootID; }
    /* 只读诊断: 置换表命中次数 / 共创建过多少节点 / 置换表条目数 */
    long long ttReuseHits() const { return m_reuseHits; }
    long long ttNodesCreated() const { return m_nodesCreated; }
    std::size_t ttSize() const { return m_tt.size(); }

    /*
     *  取"当前局面的根": 置换表命中且**从上一棵树的根 ≤ maxDepth 步可达**时复用那个
     *  节点 (连同它的子树与统计), 否则新建一个根。三个搜索入口统一走它。
     *  `color` 必须等于棋盘当前的走棋方 (规范视角与哈希都依赖它)。
     *
     *  withRootNoise (2026-09): 是否给本局的根先验加 Dirichlet 噪声。**只有自对弈取
     *  数据那条路传 true**; 评测与对局保持默认 false (见上面 rootNoise 的说明)。
     *  为 true 时每次取根都会把 m_plyInGame +1, 作为噪声退火的手数。
     */
    int acquireRoot(int color, bool withRootNoise = false);

    /*
     *  给根节点的先验加 Dirichlet 噪声 (只在 withRootNoise=true 的取根路径里调用)。
     *  把"加噪后的完整合法集先验"写进 m_rootPriorNoised 并置 m_rootNoiseActive,
     *  pickUntriedByPrior 在根节点上改用它 (于是 U 项与存进孩子的 prior 都是加噪的)。
     */
    void applyRootNoise(int rootID, int color, int ply);

    /* targetID 是否在 fromID 的子树里、且在 maxDepth 层以内 (只向下走 childIDs) */
    bool reachableWithin(int fromID, int targetID, int maxDepth) const;

    /* 置换表本体与计数器 (全部是搜索状态, 不参与任何对外语义) */
    std::unordered_map<unsigned long long, int> m_tt;   /* 局面键 -> 节点下标 */
    int m_rootID = -1;                                  /* 当前根 (-1 = 无树) */
    long long m_reuseHits = 0;                          /* 复用命中次数 */
    long long m_nodesCreated = 0;                       /* 本轮共创建多少节点 */

    /* ----------------------------------------------------------------
     *  Encoding / Action Helpers
     * ---------------------------------------------------------------- */

    /* 棋盘绝对格子索引 (10x9) */
    static int cellOf(const Pos &p) { return p.x * 9 + p.y; }

    /*
       规范视角下的格子索引: 轮到黑方时上下镜像 (x -> 9-x)。
       状态平面与动作索引都用它, 两者必须一致 —— 否则先验/策略会串帧。
    */
    static int canonicalCell(const Pos &p, int color)
    {
        const int x = (color == Stone::COLOR_BLACK) ? (9 - p.x) : p.x;
        return x * 9 + p.y;
    }

    /* 用棋盘当前的 sideToMove 作为视角 (rolloutFromCurrent 会先把它设成 color) */
    void encodeState(RL::Tensor &state);
    /* 显式指定视角 */
    void encodeStateFor(int color, RL::Tensor &state);

    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);
    /* Step -> 动作索引 (双射, 无碰撞; 必须在 color 的规范视角下算) */
    int stepToActionIdx(const Step &s, int color);
    float computeReward(const Step &s, int color);

    /* ----------------------------------------------------------------
     *  势能塑形 (PBRS, Phase 2): 把"棋盘局面价值评估"接进训练信号
     * ----------------------------------------------------------------
     *  Φ(局面) = tanh(走子方视角的 Chess::evaluate() / SCALE) ∈ (-1,1),
     *  每步奖励加 Φ(s_i) + γ·Φ(s_{i+1})。推导与"不改变最优策略"的验证写在
     *  stone.h 的 potentialReward()/shapedStepReward() 上面。
     *
     *  evaluate() 是**黑方视角** (正 = 黑好), 而 Φ 要的是**走子方视角**,
     *  所以轮到红方时取负。这个换算与 encodeState 的规范视角是同一件事的两面。
     * ---------------------------------------------------------------- */
    float potentialOf(int colorToMove);

    /*
     *  把势能塑形**原地**写进轨迹每一手的 reward:
     *      reward_i <- reward_i + Φ(s_i) + γ·Φ(s_{i+1})
     *  (推导见 stone.h 的 shapedStepReward())。之后 learnSelfPlay / commitEpisode
     *  正常算折现回报即可, RL 内核不需要知道"势能"这件事 —— 依赖方向保持
     *  "agent 依赖 RL 内核", 不是反过来。
     *
     *  **只能对同一条轨迹调用一次**: 它原地累加, 重复调用会把塑形项叠加两次。
     */
    void applyPotentialShaping(std::vector<RL::Step> &trajectory) const;

    /* 边界项用的"最后一步落子后"的势能 (已乘 α; 塑形关闭或 α=0 时为 0) */
    float finalPhiScaled(const std::vector<RL::Step> &trajectory) const;

    /*
     *  势能塑形总开关 (默认开)。留它出来**只为做 A/B 消融**: 关掉之后 Φ 恒为 0,
     *  价值目标就退回到"自举 + 材质"这条基线, 于是"塑形到底有没有用"能被量出来
     *  (见 docs/training_optimization.md 的验证一节)。
     *  做成**成员**而不是全局静态: 多线程分身训练的每个 worker 都要能各带一份设置,
     *  用全局变量会变成数据竞争。
     */
    bool potentialShaping = true;

    /*
     *  势能强度 α: 实际加进奖励的是 `α · (Φ(s_i) + γ·Φ(s_{i+1}))`。
     *
     *  留它出来是为了**退火**与 A/B: 因为 α·Φ 仍然是**状态函数**, 所以**任何 α 都
     *  保持策略不变性** (见 stone.h 的推导), 于是"先弱塑形后强塑形"可以连续过渡 ——
     *  比"先只给材质、再换 PST、最后切纯终端"那种三段硬切稳得多 (换奖励函数会让
     *  已经拟合好的价值函数瞬间失效)。
     *  α = 0 就是"无塑形"的对照。
     */
    float shapingAlpha = 1.0f;

    /*
     *  材质奖励是否由**显式奖励**承担 (默认 true = 当前口径)。
     *
     *  PBRS 的不变性只覆盖**势能那一半**; 显式材质项是不受保护的额外稠密奖励, 它
     *  会实打实地把策略推向"吃子优先"。而 Φ = tanh(evaluatePositional/2) 里**也含
     *  材质** —— 两处都给就是重复计账: 一次吃子既拿显式奖励、又拿势能差。
     *  标准做法是二选一 (Ng et al. 1999: 中间奖励只留势能差)。
     *  置 false 即为那个对照: 显式奖励只留每步代价, 材质完全交给 Φ。
     */
    bool materialRewardEnabled = true;

    /*
     * ====================================================================
     *  P0.1 (2026-09): 无界面常驻训练器的两个钩子
     * ====================================================================
     *
     * ---- (1) 按局统计的日志指针 ----
     * 非空时, trainSelfPlay **每局结束追加一条** RL::Diag::GameStat (结果/手数/
     * 和棋原因/结束方式/开局哈希/吃子数)。
     *
     * 为什么做成"写进调用方的 vector"而不是回调: 训练器要的是**全部分桶**而不是
     * 逐局事件, 一个 vector 指针就够了, 不需要 <functional> 与动态分配; 而且
     * nullptr (默认) 时行为与改动前逐位相同 —— 不做统计的调用方 (GUI / 单测 /
     * 其它 bench) 零影响。
     *
     * 与 bench_diag 的分工: 那边是**自己驱动** selectMove 收集逐手诊断 (不训练);
     * 这里是**训练路径** (trainSelfPlay) 的按局统计 —— 前者量"搜索干不干净",
     * 后者量"这一轮自对弈到底在下什么样的棋"。
     */
    std::vector<RL::Diag::GameStat> *gameLog = nullptr;

    /*
     * ---- (2) 随机开局手数 ----
     * trainSelfPlay 每局是 `chess.reset()` 起手, 于是**同一个权重下每一局都是同一盘棋**
     * (温度退火到 0.1 之后更是如此) —— 开局多样性会退化成 1 种, 而且"和棋分桶"会
     * 反复量到同一盘棋。>0 时每局先随机走 N 手合法棋再开始自对弈。
     *
     * 与各 bench 的 `--opening=N` 是同一套做法 (bench_diag 的 randomOpening), 只是
     * 这里做进了训练路径本身。默认 0 = 行为不变。
     * 只影响**对局起点**, 不影响 π 目标 / 价值目标 / PBRS 的任何口径。
     */
    int openingPlies = 0;
    /* 随机开局的种子 (固定 = 可复现; 只被 openingPlies > 0 时使用) */
    unsigned long long openingSeed = 20240901ull;

    /*
     *  一局开始前必须先设好: 首手**之前**那个局面的势能 (它是第 0 步的 Φ_before)。
     *  之所以不在 commitEpisode 里算, 是因为那时棋盘已经停在终局局面了。
     */
    float m_phiInit = 0.0f;

    /* ----------------------------------------------------------------
     *  左右镜像 (P6, 2026-09)
     * ----------------------------------------------------------------
     *  中国象棋在 **y -> 8-y** (把棋盘左右翻一下) 下是一个**规则对称**:
     *  九宫在 y=3..5 正好居中、河界横跨全部 9 列、車/馬/炮/兵/帥/仕/相 的走法形状
     *  在 y 方向上都对称。所以一个合法局面左右翻过来还是合法局面, "同一步棋"翻过来
     *  还是那一步。注意这跟 encodeState 里那个 x -> 9-x 的**规范视角镜像**是两件事:
     *  那个是为了"红黑共用一套权重"而做的坐标变换 (必须做, 且它同时换了视角),
     *  这个纯粹是数据增广的对称性。x -> 9-x **不是**增广可用的对称 —— 它会把红方的子
     *  搬到黑方半场去, 翻出来的不是一个真实局面。
     *
     *  于是 **(state, 策略目标, 价值目标) 三元组可以整体翻一倍使用**: 数据量白拿一倍,
     *  而且两个样本是同一个权重的同一侧, 不存在"红黑权重不共享"的老问题。
     *  验证见 test_ppomcts 的 [测试10] (镜像走法序列 + 增广接线),
     *  多线程下的数据效率见 docs/agents_design.md §17.7。
     * ---------------------------------------------------------------- */
    /* 规范视角下格子 (x, y) -> (x, 8-y) 的像; 自反 (mirrorCell(mirrorCell(c)) == c) */
    static int mirrorCell(int cell) { return (cell / 9) * 9 + (8 - cell % 9); }
    /* 动作索引的像: (from, to) 两个格子各自镜像 (CELLS=90 进制) */
    static int mirrorActionIdx(int actionIdx)
    {
        return mirrorCell(actionIdx / CELLS) * CELLS + mirrorCell(actionIdx % CELLS);
    }
    /* 逐平面把格子镜像: dst[p*CELLS + mirrorCell(c)] = src[p*CELLS + c] */
    static void mirrorPlanes(const RL::Tensor &src, RL::Tensor &dst);

    /* ----------------------------------------------------------------
     *  PPO + MCTS core
     * ---------------------------------------------------------------- */
    /* PUCT score for a child node */
    double getPUCT(int childID, int parentVisits) const;

    /*
     *  Phase 4.1 + R1: 用**稀疏策略前向**求父节点先验, 并按先验挑一个未展开着法。
     *  三个搜索入口 (selectMove / trainSelfPlay / warmupFromCurrent) 共用它。
     *
     *  为什么必须改 (Phase 4.1): 原来三处都是 `std::rand() % size` **随机**挑 ——
     *  先验完全没参与"展开哪个孩子"。中局约 40 个合法着法, 而一次决策只有 80 次模拟,
     *  于是前 ~40 次模拟全花在随机铺开 40 个孩子上 (每次还都要跑一遍 actor + critic
     *  前向), PUCT 根本没机会起作用。按先验挑之后, 最初几次模拟就集中在最有希望的
     *  候选上 —— 这正是"把有效搜索空间压小"的机制本身 (见 docs/agents_design.md 的
     *  搜索一节)。
     *
     *  顺带: 搜索里不再用 std::rand(), 于是 RL::Random::setSeed() 能真正控制整条搜索的
     *  可复现性 (多线程分身训练也受影响)。
     *
     *  R1 之后, 先验不再来自"全量 8100 维策略", 而是来自 ppo.actionMasked ——
     *  **只算这个节点完整合法着法集合那几列** (策略头 2.07 MB -> ~10 KB 权重流量,
     *  见 docs/training_optimization.md §9.4)。数值上等价于"全量 softmax 后取合法集
     *  再归一化", 所以**合法动作之间的相对大小完全不变 ⇒ 展开顺序不变**。
     *
     *  返回 untriedActionIndices 里的下标 (调用方还要从 untriedSteps 里取同步的那一项);
     *  priorOut = 选中动作的先验 P(s_parent, a)。
     *  legalIdx / probs 是复用的出参 (节点的完整合法动作集与其稀疏概率, 一一对应) ——
     *  调用方在模拟循环外各留一份, 免得每次都重新分配。
     *
     *  **不是 const**: 稀疏路径要在网络上跑一次部分前向。
     */
    int pickUntriedByPrior(int nodeID, const RL::Tensor &parentState,
                           std::vector<int> &legalIdx,
                           std::vector<float> &probs,
                           float &priorOut);

    /*
       把根节点的**访问计数**归一化成策略目标分布 (π ∝ N, τ=1), 与
       SACAZAgent::visitDistribution 同一约定。

       为什么必须换成它: 搜索的全部价值就在"根节点各候选的相对访问次数"上。一次
       80 次模拟的搜索里, 被选中的那个动作往往只比其它候选多访问一两次 —— 只保留
       它的 one-hot, 等于把搜索退化成一个"随机挑一步"的采样器, actor 学的只是行为
       克隆, AlphaZero 赖以工作的那个**改进算子**被丢掉了。

       返回 false 表示没有任何访问计数 (全部为 0), 调用方应退回 one-hot。
    */
    bool visitDistribution(int rootID, RL::Tensor &pi) const;

    /*
       稀疏版: 只输出非零项 (下标 + 概率)。回放池里存的就是它 —— 存 8100 维稠密分布
       是每步 32 KB, 存非零项只有 ~40 项 (~320 B)。稠密版 visitDistribution 就是
       它加一层展开, 两者共用同一份实现。
    */
    bool visitDistributionSparse(int rootID,
                                 std::vector<int> &actionIdx,
                                 std::vector<float> &actionProb) const;

    /*
     *  ---- 从根的访问计数里"出招" (2026-09: 唯一口径) ----
     *
     *  temp > 0.1 : 按 N^(1/temp) 归一化后**采样** (AlphaZero 的出招口径, 探索靠它);
     *  temp <= 0.1: 取访问数最多的孩子 (argmax, 确定性)。
     *
     *  为什么要抽成一个函数: 原来这段逻辑在 trainSelfPlay 里内联了一份, 而
     *  `selectMove(color, simulations, temp)` 的 temp 参数**根本没被读取** —— 它
     *  永远走 argmax, 与文档注释 (\"or sample from visit distribution if temp > 0\")
     *  不符, 任何传 temp>0 想拿到探索采样的调用方都静默地拿到了确定性结果。
     *  现在两处共用同一实现, 口径不可能再漂移。
     *
     *  返回 -1 表示"没有任何访问计数"(sum(visits)==0), 调用方按无效着法处理 ——
     *  这与改动前 trainSelfPlay 的分支条件完全一致。
     */
    int pickRootChildByVisits(int rootID, float temp);

    /* ----------------------------------------------------------------
     *  ---- 叶子估值 (三个搜索入口的唯一口径) ----
     *
     *  **终局叶子必须给真实胜负, 不能交给 critic。** 走到评估段时棋盘正停在叶子局面
     *  (moveForward 已把 sideToMove 翻成叶子的走棋方)。若这个局面已经终局
     *  (将杀/困毙/吃将/三次重复/60 回合), 它的价值是**确定的**, 让一个只学过"评估"
     *  的网络去猜, 搜索就永远看不见"一步杀" —— 诊断仪表盘实测过: 一步杀命中率
     *  **0/20 = 0%**, 白吃子 16.7% (见 bench_diag)。SACAZAgent 一直有 terminalValue(),
     *  这条路径一直没有, 两个 agent 的搜索口径本来就不一致。
     *
     *  返回**叶子走棋方**视角的价值 (与 backup 的视角约定一致):
     *  判和 -> 0, 叶子走棋方被将死/困毙 -> -1, 对方被将死 -> +1。
     *  非终局才走 encodeState + ppo.value。
     * ---------------------------------------------------------------- */
    double evaluateLeaf(RL::Tensor &leafStateScratch);

    /* ----------------------------------------------------------------
     *  ---- 搜索诊断 (健康度观测, 2026-09) ----
     *
     *  棋力是滞后指标; 能回答"搜索这一层设计对不对"的是根上的这几个量。本函数把它们
     *  整理成 RL::Diag::RootDiag (纯只读, 不改任何搜索状态, 不影响可复现性)。
     *
     *  **必须在一次搜索刚结束时调用** (selectMove / trainSelfPlay 的一手之后):
     *  那时棋盘已回退到根局面, 节点统计也还是那一手的。
     *
     *  它做三件别处做不到的事:
     *   1. **吃子 vs 退让 的 Q 分组** —— "搜索是不是怕吃子"只有这里能回答。
     *      注意 Q 要取负号换算到根走棋方视角 (见 rl/diag.h 的 qForParent)。
     *   2. **KL(访问分布 || 先验)** —— 搜索到底有没有给出网络先验之外的信息。
     *      ≈0 就说明搜索白跑 (目标只是在复现先验), 这正是本工程量过的"地板"问题。
     *   3. **展开覆盖率 children/legal** —— 20 次模拟对 38.7 个分支时它是 0.5,
     *      意味着**一次深挖都没有** (见 chessboard.cpp 里 BG_TRAIN_SIMS 的长注释)。
     *
     *  返回 false 表示"没有可诊断的根" (无树 / 零访问)。
     * ---------------------------------------------------------------- */
    bool rootDiag(RL::Diag::RootDiag &out) const;

    /*
     *  调试钩子: 把根的已展开着法按访问数排序打印 (P/Q/N 三列 + 是否吃子)。
     *  用来人工核对"吃子着排第几" —— 这是最省事的单局面定位手段。
     */
    void printSortedRoot(int limit = 10) const;

    /* ----------------------------------------------------------------
     *  自对弈 -> 回放池 -> 多 epoch 批量学习 (P3 + P4, 2026-09)
     *
     *  旧路径是"一整条轨迹做完折现回报, 然后逐步 trainStep" —— 每条样本恰好被用一次,
     *  而且每次都单独跑一遍全参数优化器 (实测优化器占一步的 66%)。
     *  新路径把样本推进回放池, 每次从池里采样 batchSize 条、过 epochs 遍, 梯度累积后
     *  只做一次优化器更新。实测: 一局 100 步, 2.3 倍便宜的同时更新次数还从 100 涨到
     *  128 (rl/ppo.h 里算了这笔账)。
     * ---------------------------------------------------------------- */
    /* 每次学习的批大小 / 过几遍 */
    int replayBatchSize = 64;
    int replayEpochs = 2;

    /*
       左右镜像数据增广 (P6, 见上面 mirrorCell 的说明): 每一条样本进池时, 顺手把
       (state, 动作下标, 价值目标) 整体翻一下再存一份。

       代价是回放池的消耗速度翻倍 (learner 每轮抽 64 条, 其中一半是镜像样本),
       换来的是同一局自对弈产生的**有效数据量翻倍** —— 而生成一局样本实测约 2.4 s,
       重放一条约 20 ms, 镜像一条只是几次内存拷贝, 这笔账非常划算。
       关掉它 (`= false`) 可以得到严格未增广的对照组。
    */
    bool mirrorAugment = true;

    /*
        R1 A/B 开关 (2026-09, 默认 true = 稀疏路径: 策略头只算合法列)。

        false 复现 **R1 之前**的口径: 全量策略前向, 而且先验**直接用原始概率**
        (`p_full(a)`, 合法集上的质量 Z 没有被除掉)。

        为什么要留这个开关: R1 的两条路径在"合法动作之间的相对大小"上完全一致
        (所以展开顺序不变), 但**先验的绝对尺度**不同 —— 稀疏路径给的是"在合法集上
        归一化"的概率, 比原始 p_full 大了 1/Z 倍, 而 PUCT 的探索项 `c_puct·P·√N/(1+n)`
        对 P 是线性的, 于是等于把 c_puct 乘了 1/Z。同权重、同局面、只改这一处的配对
        A/B (bench_policy_agreement --ab=1) 才能把"选点变了多少"量出来, 而不是靠
        "两次运行的聚合一致率差不多"这种弱证据。
    */
    bool sparsePolicyHead = true;

    /*
       一局结束: 按折现回报把这条轨迹推进回放池, 池子够大时触发一次批量学习。
       轨迹里的策略目标是稠密的 (RL::Step 只放得下一个 Tensor), 但**进池时立刻转成
       稀疏** —— 长期占内存的是回放池, 不是这条临时轨迹。

       参数是**非 const** 引用: 函数会先把势能塑形原地写进每一手的 reward
       (见 applyPotentialShaping)。轨迹是调用方的局部变量, 用完即弃。

       `legalPerStep` (R2, 可选): 与 trajectory **同步**的"每一手局面的完整合法着法下标"。
       训练侧的稀疏口径需要它才能划出 softmax 的分母 —— 轨迹只带策略目标 (访问分布),
       而访问分布只是合法集的一个**子集**, 拿它当分母等于把没被搜索访问到的合法着法
       从策略里抹掉。不给 = 旧的全量 8100 维口径。
    */
    void commitEpisode(std::vector<RL::Step> &trajectory, float finalOutcome,
                       const std::vector<std::vector<int>> *legalPerStep = nullptr);

    /*
       ---- 截断局的自举 (2026-09) ----
       自对弈走到手数上限仍未分胜负时, 原来一律 `commitEpisode(traj, 0.0f)` —— 也就是
       告诉 critic "这些局面的回报是 0"。而象棋和棋极多、手数上限又低 (GUI 训练只给
       60 手), 于是**大部分轨迹的价值目标都是 0**, critic 只学得到"和棋"。
       正确的做法是用 bootstrapping: 把最后一步之后的局面交给 critic 估值, 取负号
       换算到最后一步走子方的视角 (此刻轮到对手走, 所以 V 是对手视角的)。
       口径与在线路径 endOnline 完全一致 (那边一直是这么做的), 所以两条路的学习问题
       不再不同。用 truncationBootstrap=false 可回到旧口径做 A/B。
    */
    bool truncationBootstrap = true;
    /* 估值最后一步之后的局面, 返回"最后一步走子方视角"的自举终局值 (clamp 到 ±1) */
    float bootstrapOutcome();

    /*
       R2: 取某个节点的**完整合法着法集** (未展开项 ∪ 已展开孩子的 parentAction)。
       与 pickUntriedByPrior 里那套构造同源, 理由见那里的长注释。
    */
    void legalIndicesOf(int nodeID, std::vector<int> &out) const;

    /* ----------------------------------------------------------------
     *  稀疏 MoE 诊断 (只读, 不参与决策; 给测试与调参用)
     * ---------------------------------------------------------------- */
    int moeExpertCount() const { return ppo.moeExpertCount(); }
    int moeTopK() const { return ppo.moeTopK(); }
    int moeLayerCount() const { return ppo.moeLayerCount(); }
    long long actorParamCount() const { return ppo.actorParamCount(); }
    long long criticParamCount() const { return ppo.criticParamCount(); }
    void moeUsage(std::vector<long long> &out) const { ppo.moeUsage(out); }
    void resetMoeUsage() { ppo.resetMoeUsage(); }

    /* Online training (human-vs-AI) */
    void beginOnline();
    void recordOnline(const Step& s, int color, const RL::Tensor& stateBefore);
    void endOnline(int winner, int myColor);

public:
    PPOMCTSAgent(Chess &chess_,
                 int hiddenDim = 64,
                 float gamma_ = 0.99f,
                 float lr = 0.001f,
                 float cpuct = 1.414f,
                 int expertHidden_ = 64,
                 float moeAuxCoef_ = 0.1f,
                 /* false = 只做搜索/推理的网络 (供多线程分身训练的 worker 用,
                    见 rl/ppo.h 的同名参数)。worker 必须同时把 replayBatchSize 设成 0,
                    否则它会在没有梯度的网络上尝试学习。 */
                 bool withGrad_ = true);

    ~PPOMCTSAgent() = default;

    /* AgentBase interface */
    Step getBestMove(int color) override;
    std::string getName() const override;
    /*
     * 走子前先"探索环境 + 在线训练一次"再决策 (仿 snakeAI 的决策流程, 见 aiagent.h):
     * 从当前局面用本 agent 的探索策略滚若干步收集经验, 在线训练一次, 然后才走子。
     * 探索期间用 moveForward/moveBack 试走, 结束时会原样回退, 不影响真棋局。
     */
    bool exploreAndTrain(int color, int rolloutSteps) override;
    /* ----------------------------------------------------------------
     *  Public API
     * ---------------------------------------------------------------- */

    /* Select a move using PPO+MCTS search (no exploration during search)
     *  color        : side to move
     *  simulations  : number of MCTS iterations
     *  temp         : temperature for final move selection (0 = argmax) */
    Step selectMove(int color, int simulations,
                    float temp = 0.0f);

    /* Train via self-play using PPO+MCTS:
     *   - At each position, run MCTS guided by PPO
     *   - Sample the move from the MCTS visit distribution
     *   - Store trajectories and train PPO at the end */
    void trainSelfPlay(int episodes, int simulations,
                       int maxMoves = 200, bool verbose = true,
                       float tempRoot = 1.0f, float tempFinal = 0.1f);

    /* Warmup: self-play several episodes from current board state (in place) */
    void warmupFromCurrent(int episodes = 5, int simulations = 200,
                           int maxMoves = 200);

    /* Save / Load PPO weights */
    bool saveModel(const std::string &actorPath,
                   const std::string &criticPath);
    bool loadModel(const std::string &actorPath,
                   const std::string &criticPath);

    /* Convenience: save both actor & critic from a single file prefix.
       (e.g., saveModel("model") saves to "model_actor" and "model_critic") */
    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* Statistics */
    int getTotalEpisodes() const { return totalEpisodes; }
    /*
     * 最近一次 trainStep 的 critic 价值 MSE (界面"训练损失曲线"用):
     *   loss = (V(s) - 目标回报)²  —— 与 DQN 报的"平均平方 TD 误差"同类可比。
     * (actor 的交叉熵在 RL::PPO::lastActorLoss 里, 目前不上图。)
     */
    float getLastTrainLoss() const override { return (float)ppo.lastLoss; }
    /*
       actor 的交叉熵 (同一批的批平均)。界面目前只画了 critic 的 MSE —— 只看一半训练:
       策略离"搜索给出的走法"有多远, 是这个量才反映得出来的。留一个只读访问器,
       供诊断程序与后续的曲线接线使用。
    */
    float getLastActorLoss() const { return (float)ppo.lastActorLoss; }
    float getWinRate(int color = Stone::COLOR_BLACK) const {
        int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / totalEpisodes : 0.0f;
    }
};

#endif // PPOMCTS_AGENT_H
