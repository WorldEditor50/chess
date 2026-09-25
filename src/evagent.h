#ifndef EVAGENT_H
#define EVAGENT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "chess.h"
#include "aiagent.h"
#include "rl/net.hpp"
#include "rl/layer.h"
#include "rl/loss.h"

/*
 * EVABAgent - 学会评估的 Alpha-Beta (Evaluation-network + Alpha-Beta)
 *
 * 设计见 docs/agent_evab_design.md。一句话: 用**搜索**保证棋理正确 (合法性/将杀/
 * 困毙/和棋/时间控制), 用**学习到的评估网络**提供"看得准"的叶子值, 两者用 blend
 * 混合, 于是 blend=0 时它就是"ABAgent + 置换表 + 迭代加深 + 更好的走法排序",
 * blend 随训练爬到 1 时网络接管评估。
 *
 * 用到的现有组件:
 *   RL::Net / RL::Layer<Tanh>      : 价值网络 (1260 -> 48 -> 1)
 *   RL::Loss::MSE::df              : 值回归损失
 *   Net::RMSProp                   : 优化器 (mini-batch: 逐样本累积梯度后更新一次)
 *   RL::Tensor + rl/simd_ops.hpp   : 1260 维 GEMV 走 SIMD 快路径 (~6 us/次评估)
 *   Chess::sample                  : 只产生合法走法的生成器
 *   Chess::evaluate()              : 手工评估, 作为引导基线
 *   Chess::computeHash / isRepetition / getResult : 置换表键 与 和棋判定
 */
class EVABAgent : public AgentBase
{
public:
    /* ---- 状态编码: 7 种棋子 x 2 方 x 90 格 ---- */
    static constexpr int PLANES = 17;
    static constexpr int CELLS = 90;
    /* 14 个棋子平面 + 3 个规则上下文 (无吃子进度/重复次数/将军), 见 src/chessstate.h */
    static constexpr int PLANE_HALFMOVE = 14;
    static constexpr int PLANE_REPEAT   = 15;
    static constexpr int PLANE_CHECK    = 16;
    static constexpr int STATE_DIM = PLANES * CELLS;   /* 1530 */

    /* 终局分值: 与 (-1,1) 的正常评估拉开距离, 保证"将杀优先于一切" */
    static constexpr double MATE = 1000.0;
    /* 无胜负时的搜索无穷大 */
    static constexpr double INF = 1e9;
    /* 手工评估 -> (-1,1) 的压缩尺度 (见设计文档第四节) */
    static constexpr float EVAL_SCALE = 3.0f;

    /* 置换表条目 */
    struct TTEntry {
        int depth;
        double score;
        int flag;          /* 0 = EXACT, 1 = LOWER, 2 = UPPER */
        bool hasMove;
        int id;
        int nextId;
        Pos from;
        Pos to;
    };

    /* 自对弈样本 */
    struct Sample {
        RL::Tensor state;
        float target;
    };

public:
    Chess &chess;
    RL::Net valueNet;

    /*
     * 学习评估与手工评估的混合比例。
     *   0.0 -> 叶子 = tanh(手工评估/S), 行为等价于 ABAgent (但搜索更强)
     *   1.0 -> 叶子 = 价值网络输出
     *
     * **它不再是"永远 0"**: 每次成功的在线更新之后 blend 会向 blendMax 爬一步
     * (见 exploreAndTrain 里的阶梯), 回滚时退回去。以前这里只有初值 0、没有任何
     * 地方改它, 于是界面上那个"learned eval"其实一次都没参与过决策 —— 实测
     * (build/evab_probe2.cpp) 20 次在线更新里网络输出只动了 1.4e-4, 而 blend 恒为 0。
     */
    float blend;
    /*
     * blend 的爬升上限与每步增量 (见 exploreAndTrain)。
     * 上限取 0.3 而不是 1.0 有两个实测理由:
     *   1. 网络是**照着"0.5 手工评估 + 0.5 搜索评分"训练的**, 训好之后它本来就
     *      接近手工评估 (实测 |net-hand| ≈ 0.02), 再往上加权重买不到多少差别;
     *   2. 只要 blend > 0, 叶子评估就要跑一次网络前向 (1260->48->1 ≈ 60k MAC,
     *      而手工评估只有几十次运算) —— 实测 depth=5 时 blend=0 是 60 ms/步,
     *      blend=0.5 是 724~858 ms/步 (12~14 倍)。**评估变贵 = 搜索变浅**,
     *      所以让网络"接管评估"在当前实现下是亏的。
     */
    float blendMax;
    float blendStep;
    /* 搜索预算: 深度上限 + 时间上限 (毫秒, 0 = 不限时) */
    int maxDepth;
    long long timeBudgetMs;
    /*
     * "探索 + 在线训练"这一轮的**时间上限** (毫秒, 0 = 不限时)。
     * 为什么需要它: 探索的每一步都要跑一次 labelDepth 层的 negamax 当标签, 而界面上
     * "预训步数"默认 64 —— 深度 4 时就是 64 × ~25 ms ≈ 1.6 s, 深度 5 时 ~12 s。
     * 有了时间上限, 步数就变成"上限"而不是"承诺", 单步思考时间可控;
     * 实际滚了多少步会写进 getExploreInfo()。
     */
    long long exploreBudgetMs;
    /* 训练时对局结果标签的权重 (其余权重给 TD-leaf 的搜索根评分) */
    float outcomeWeight;
    /* 训练时以 eps 概率随机走子, 保证自对弈数据的多样性 */
    float exploreEps;
    float learningRate;

public:
    explicit EVABAgent(Chess &chess_, int hiddenDim = 48, int depth = 4,
                       long long budgetMs = 0);

    /* AgentBase */
    Step getBestMove(int color) override;
    std::string getName() const override;

    /*
     * 自检报告 (界面"模型自检"面板的数据源)。
     * 契约见 aiagent.h: **只读、可重复调用、不动棋盘** —— 所以这里只报静态结构
     * 与搜索/训练计数, 不跑搜索、不调 netHandGap() (那个函数会在 this->chess 上
     * 随机试走, 违反只读契约)。
     */
    std::string selfCheckReport() const override;

    /*
     * EVAB 版的"先探索环境 + 预训练再决策"。
     *
     * 它的"探索"本来就是搜索 (迭代加深会把当前局面看到第 maxDepth 层), 所以这里
     * 额外做的是**把这次探索的结果蒸馏回价值网络**: 从当前局面滚若干步, 用搜索的
     * 根评分当 TD-leaf 标签更新一次网络。
     *
     * 保守是刻意的: 实测(见 docs/agent_evab_design.md 8.3)不受约束的在线更新会把
     * 评估带偏、棋力反而下降。因此这里 (a) 目标值向手工评估锚定一半, (b) 更新后
     * 检查网络与手工评估的差距, 超过阈值就**整段回滚**。
     */
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;

    /* 搜索入口: 迭代加深 + alpha-beta; 返回的 Step.valid == false 表示无合法走法 */
    Step search(int color, int depth, long long budgetMs = 0);

    /*
     * 指定走法的搜索评分 (完整窗口), depth 为总层数。
     * 主要用于测试: 用来验证"镜像等价的走法必须得到相同的分数"这条对称性 ——
     * 搜索里任何颜色相关的偏差都会在这里暴露, 而在对局统计里会被淹掉。
     */
    double scoreMove(int color, const Step &s, int depth);

    /* 叶子评估 (当前走棋方视角, (-1,1); 终局为 +/-MATE) */
    double evaluateLeaf(int color);

    /*
     * 用"随机走子 + 手工评估"生成样本, 先把价值网络拉到"不比手工评估差"的起点。
     *
     * 这一步是引导阶梯的第一级, 也是最关键的一级: 直接让一个随机初始化的网络接管
     * 评估 (blend=1) 会把引擎瞬间拉垮 —— 网络的输出和手工评估差着十万八千里,
     * 搜索拿到的叶子值就是噪声。先模仿, 再自举。
     *   positions : 生成多少样本
     *   maxPlies  : 每个局面的最大随机步数
     *   batchSize : 每次参数更新的样本数
     *   epochs    : 遍历样本的轮数
     * 返回最后一个 epoch 的平均绝对误差。
     */
    double pretrainFromHandEval(int positions, int maxPlies,
                                int batchSize = 32, int epochs = 8);

    /*
     * 自对弈训练 (见设计文档第六节)。
     *   games        : 本轮的局数
     *   playDepth    : 对局时搜索深度 (要快)
     *   labelDepth   : 生成 TD-leaf 标签时的搜索深度 (>= playDepth 才会"蒸馏"更深的信息)
     *   maxMoves     : 单局步数上限
     *   batchSize    : 每次参数更新的样本数 (这一项很关键: 把整轮样本攒成一次更新
     *                  等于整轮只走一步梯度, 那是什么都学不到的)
     * 返回本轮的平均 |预测 - 标签|。
     */
    double trainSelfPlay(int games, int playDepth, int labelDepth,
                         int maxMoves, bool verbose, int batchSize = 32);

    bool saveModel(const std::string &path);
    bool loadModel(const std::string &path);

    /*
     * 在线更新前后, 网络与手工评估在若干局面上的平均差距。
     * 用来做"变差就回滚"的判据: 偏离手工评估太远说明网络已经跑飞了。
     */
    double netHandGap(int samples = 24);

    /* 统计 (供测试/UI 报告) */
    long long getNodes() const { return m_nodes; }
    long long getTTProbes() const { return m_ttProbes; }
    long long getTTHits() const { return m_ttHits; }
    long long getTTStores() const { return m_ttStores; }
    int getReachedDepth() const { return m_reachedDepth; }
    long long getLeafEvals() const { return m_leafEvals; }
    /* 最近一次 search() 的根评分 (当前走棋方视角): 训练时用作 TD-leaf 标签 */
    double getLastRootScore() const { return m_lastRootScore; }
    /*
     * 最近一次在线训练 (trainBatch) 的平均绝对误差, 界面的"训练损失曲线"用。
     * 这里用 MAE 而不是 MSE: EVAB 的价值网目标就是"网络分 vs 手写评估分"的差,
     * 报差值的绝对值最直观。
     */
    float getLastTrainLoss() const override { return m_lastLoss; }
    void resetStats();

private:
    /* 搜索状态 */
    std::unordered_map<unsigned long long, TTEntry> m_tt;
    std::vector<Step> m_killers;          /* ply*2 + 0/1 */
    std::unordered_map<int, int> m_history; /* id*90+cell -> 分值 */
    std::vector<Step> m_path;             /* 当前搜索路径, 用于重复局面判断 */
    std::chrono::steady_clock::time_point m_deadline;
    bool m_aborted;
    int m_horizon;                        /* 当前迭代加深的层数上限 */
    /* 统计 */
    long long m_nodes;
    long long m_ttProbes;
    long long m_ttHits;
    long long m_ttStores;
    long long m_leafEvals;
    int m_reachedDepth;
    double m_lastRootScore;
    float m_lastLoss = std::numeric_limits<float>::quiet_NaN();  /* 见 getLastTrainLoss */
    /* 网络隐藏层宽度 (在线更新时要按同样的结构造一份备份用于回滚) */
    int m_hiddenDim;
    /* 复用缓冲, 避免每次评估都分配 STATE_DIM 维张量 */
    RL::Tensor m_stateBuf;

public:
    /*
       公开是为了让测试能断言"状态里确实带规则上下文" (test_evab [1]): 走 4 手可逆循环
       回到同一个局面时, 棋子平面必须逐位相同而规则上下文必须变。
       与 PPOMCTSAgent / DQNABAgent 的 encodeStateFor 一样是公开的 —— 编码是 agent 与
       网络之间的**契约**, 检查它不该需要 friend。
    */
    void encodeCanonical(int color, RL::Tensor &state) const;

private:
    double negamax(int color, int depth, double alpha, double beta, int ply);
    double quiescence(int color, double alpha, double beta, int ply);
    void scoreMoves(std::vector<Step*> &steps, int ply, const Step *ttMove);
    void recordKiller(int ply, const Step &s);
    int historyIndex(const Step &s) const { return s.id * CELLS + (s.nextPos.x * 9 + s.nextPos.y); }
    bool timeUp();
    void clearSearchState();
};

#endif // EVAGENT_H
