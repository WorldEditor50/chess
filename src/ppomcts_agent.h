#ifndef PPOMCTS_AGENT_H
#define PPOMCTS_AGENT_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include "chess.h"
#include "aiagent.h"
#include "rl/ppo.h"

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
     *  状态编码: 16 个平面 x 90 格
     * ---------------------------------------------------------------- */
    static constexpr int CELLS = 90;                 /* 10 行 x 9 列 */
    static constexpr int PIECE_TYPES = 7;
    static constexpr int PIECE_PLANES = PIECE_TYPES * 2;   /* 7 类 x {己方, 对方} */
    static constexpr int PLANE_ATTACKED = PIECE_PLANES;          /* 14: 被对方攻击 */
    static constexpr int PLANE_UNDER_ATTACK = PIECE_PLANES + 1;  /* 15: 己方子被攻击 */
    static constexpr int PLANES = PIECE_PLANES + 2;              /* 16 */
    static constexpr int STATE_DIM = PLANES * CELLS;             /* 1440 */

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
    int expertHidden;               /* 稀疏 MoE 的 MLP 专家隐层宽度 */
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
            : parentID(-1), parentAction(-1),
              visitCount(0), totalValue(0.0), prior(0.0),
              currentColor(Stone::COLOR_NONE) {}
        AZNode(int pid, int pa, const Step &st, float p, int color)
            : parentID(pid), parentAction(pa), step(st),
              visitCount(0), totalValue(0.0), prior(p),
              currentColor(color) {}

        double getQ() const {
            return visitCount > 0 ? totalValue / (double)visitCount : 0.0;
        }
    };

    std::vector<AZNode> nodes;

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

    /*
     *  势能塑形总开关 (默认开)。留它出来**只为做 A/B 消融**: 关掉之后 Φ 恒为 0,
     *  价值目标就退回到"自举 + 材质"这条基线, 于是"塑形到底有没有用"能被量出来
     *  (见 docs/training_optimization.md 的验证一节)。
     *  做成**成员**而不是全局静态: 多线程分身训练的每个 worker 都要能各带一份设置,
     *  用全局变量会变成数据竞争。
     */
    bool potentialShaping = true;

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
     *  Phase 4.1: 按**先验**挑一个未展开着法 (AlphaZero 的做法), 返回它在
     *  untriedActionIndices 里的下标。三个搜索入口 (selectMove / trainSelfPlay /
     *  warmupFromCurrent) 共用它。
     *
     *  为什么必须改: 原来三处都是 `std::rand() % size` **随机**挑 —— 先验完全没参与
     *  "展开哪个孩子"。中局约 40 个合法着法, 而一次决策只有 80 次模拟, 于是前 ~40 次
     *  模拟全花在随机铺开 40 个孩子上 (每次还都要跑一遍 actor + critic 前向), PUCT 根本
     *  没机会起作用。按先验挑之后, 最初几次模拟就集中在最有希望的候选上 —— 这正是
     *  "把有效搜索空间压小"的机制本身 (见 docs/agents_design.md 的搜索一节)。
     *
     *  顺带: 搜索里不再用 std::rand(), 于是 RL::Random::setSeed() 能真正控制整条搜索的
     *  可复现性 (多线程分身训练也受影响)。
     *
     *  parentPolicy 必须是**父节点**的策略输出 pi(s_parent)[a], 且与 untriedActionIndices
     *  处在同一个规范视角 (理由见 selectMove 里那段长注释)。
     */
    int pickUntriedByPrior(int nodeID, const RL::Tensor &parentPolicy) const;

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
       一局结束: 按折现回报把这条轨迹推进回放池, 池子够大时触发一次批量学习。
       轨迹里的策略目标是稠密的 (RL::Step 只放得下一个 Tensor), 但**进池时立刻转成
       稀疏** —— 长期占内存的是回放池, 不是这条临时轨迹。

       参数是**非 const** 引用: 函数会先把势能塑形原地写进每一手的 reward
       (见 applyPotentialShaping)。轨迹是调用方的局部变量, 用完即弃。
    */
    void commitEpisode(std::vector<RL::Step> &trajectory, float finalOutcome);

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
