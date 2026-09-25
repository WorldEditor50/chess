#ifndef AIAGENT_H
#define AIAGENT_H

#include <string>
#include <fstream>
#include <limits>
#include <functional>   /* OpponentPolicy 的"对手在这一手会怎么走" (P1) */
#include "chess.h"

/*
 * ================================================================
 *  OpponentPolicy - "对手在这个局面上会怎么走" (P1, 2026-09)
 * ================================================================
 *
 * 为什么需要它 (用户口径): "对手的棋仍然不进训练数据 (exploreAndTrain 没有对手参数)
 * —— 模式解决的是'能不能归因', 不是'能不能从对手身上学'"。
 *
 * 现状: 探索 (rolloutFromCurrent) 是**纯自对弈** —— 从当前局面出发滚 N 手, 每一步
 * (包括"对手那一半")都由**学习方自己的策略**产生。于是学习方学到的永远是"怎么跟
 * 自己下", 而真实对手的着法从未参与数据生成。选了 Alpha-Beta 当陪练也一样:
 * 学的是"自己滚出来的经验" (见 chessboard.h 里 AGENT_ALPHABETA 的注释)。
 *
 * 这个结构把"对手的着法从哪来"传进探索: `stepFor(turn)` 由**调用方**实现, 它在
 * **当前探索局面**上给出真实对手会走的那一手 (界面里就是对手 agent 的决策协议)。
 * 于是探索里"对手那一半"不再是学习方自己的猜测。
 *
 * ⚠ 两条**刻意的**设计选择 (都有代价, 写在这里免得以后被当成遗漏):
 *
 *  1. **对手着法不记成学习方的样本**。自对弈里"对手那一半"是学习方自己采样出来的,
 *     所以把它记成学习方的动作转移是**on-policy** 的; 换成真对手之后同一份记录会变成
 *     **off-policy 却打着 on-policy 的标签** —— 对 REINFORCE/PPO 这类要行为策略的东西
 *     是实打实的偏差 (冒充进来的动作从来没有以那个概率被采样过)。
 *     所以这里只用它**推进局面**: 学习方自己那些转移落在"被真对手应手之后的局面"上,
 *     回报也因此带着对手的影响 —— 这正是"从对手身上学"的内容。
 *     (把对手着法当**样本**收进来是另一件事: 那要求"给网络对手特征"或按 off-policy
 *     方法处理, 否则价值目标不可辨识; 见 docs 里的待办。)
 *
 *  2. **有预算上限** (`budget`): 一次决策里的探索可能滚几十手, 而"问一次对手"= 一次
 *     完整决策 (SAC+AZ 实测 2.3 s/手)。所以对手只在**最前面 budget 个对手手**上被问到,
 *     之后回退到原来的自对弈。默认 0 = 完全不问 (行为与改动前逐字相同)。
 */
struct OpponentPolicy
{
    /* 在当前舞台 (棋盘) 上给出对手的着法; 返回无效 Step 表示"问不到, 回退自对弈" */
    std::function<Step(int turn)> stepFor;
    /* 名字 (日志/探索说明用), 例如 "PPO+MCTS" */
    std::string name;
    /* 本次探索里最多问几次 (0 = 不问) */
    int budget = 0;
    /* 问到的着法**真的**被用在探索里的次数 (由 rolloutFromCurrent 回填) */
    int used = 0;
    /* 问到了但不在合法集里 (编码/规则不一致) 而回退的次数 —— 诊断用 */
    int unmatched = 0;

    bool valid() const { return budget > 0 && (bool)stepFor; }
};

/*
 * AgentBase - Abstract base class for all chess AI agents
 *
 * Provides a unified interface for move selection across all agent types:
 *   - ABAgent        : Alpha-Beta Pruning
 *   - MCTS           : Monte Carlo Tree Search (random rollouts)
 *   - PGEagent       : Policy Gradient (REINFORCE)
 *   - DQNAgent       : Deep Q-Network
 *   - PPOMCTSAgent   : PPO + MCTS (AlphaZero-style)
 */
class AgentBase
{
public:
    virtual ~AgentBase() = default;

    /* --- Core interface --- */

    /* Select the best move for the given color */
    virtual Step getBestMove(int color) = 0;

    /* Human-readable agent name */
    virtual std::string getName() const = 0;

    /* Optional: reset internal state (tree, history, etc.) */
    virtual void resetState() {}

    /*
     * ================================================================
     *  ---- ④ 学习口径的奖励 (界面奖励曲线用, 2026-09) ----
     * ================================================================
     * 背景 (用户在界面导出 CSV 上实测到的, 见 docs/sac_learn_reward_2026_09.md §1.1):
     * 界面那条奖励曲线用的是 `Chess::moveForward` 的 totalReward (**材质按原值 ×1**,
     * 而且是黑方视角记账), 而 agent 在线学习用的是**自己的** `computeReward()`
     * (**材质 ×REWARD_MATERIAL_COEF = 0.1** + 每步代价 −0.001) 加终局 ±1。
     * 两个口径差 **10 倍** —— 于是"曲线上的比例"永远解释不了"学习信号的比例":
     * 用户从 CSV 里奖励最大值 4.5 读出"材质比赢棋重要 3.5 倍", 而 agent 学的是 0.35 : 1。
     *
     * 这三个虚函数把"这个 agent 的学习口径"暴露给界面 (由 ChessBoard 在每手/局末取数):
     *   hasLearningReward()      : 有没有学习口径。纯搜索 agent (Alpha-Beta / MCTS / EVAB)
     *                              没有 —— 它们的"环境奖励"就是引擎那本账, 界面照旧显示
     *                              引擎口径并**标注**出来。
     *   learningStepReward()     : 即时奖励 (走子方视角)。**必须在 moveForward 之前调**:
     *                              实现要按 `s.nextId` 去读被吃子的 value, 而落子后它已经
     *                              alive=false (见 stone.h 的 stepReward 注释)。
     *   learningTerminalReward() : 终局值。默认与引擎一致 (±1, 和棋 0); SAC 开着塑形时
     *                              它是 ±(1+败方剩余材质/3.5) —— 必须走 agent 自己的
     *                              `terminalReward()`, 否则界面上显示的游戏与它学的是两个。
     *
     * 契约: 只读棋盘, 不改任何状态, 不训练 (对局线程持有 mutex 时调用)。
     */
    virtual bool hasLearningReward() const { return false; }
    virtual float learningStepReward(const Step &s, int color)
    {
        (void)s;
        (void)color;
        return 0.0f;
    }
    virtual float learningTerminalReward(int chessResult, int perspective) const
    {
        return outcomeForMover(chessResult, perspective);
    }
    /* 界面标签用: 这个 agent 的奖励曲线是哪个口径 */
    virtual std::string rewardCaliperName() const
    {
        return hasLearningReward() ? std::string("学习口径")
                                   : std::string("引擎口径(材质x1)");
    }

    /* --- 决策流程: 先探索环境 + 预训练, 再决策 (仿 snakeAI) --- */

    /*
     * snakeAI 的 Agent::xxxAction() 每次走子前都是同一套三步:
     *
     *   1. observe(state)              —— 把**当前局面**编码下来留一份 state0
     *   2. if (trainFlag) {            —— 从当前局面出发, 用**自己的探索策略**滚若干步
     *          for (i < N) {             (noiseAction / gumbelMax / 采样 ...)
     *              选动作 -> 模拟走子 -> 算即时奖励 -> perceive / 收集 Step
     *              撞墙或到达目标就 break
     *          }
     *          learn() / reinforce()  —— 用这批**新鲜经验**在线训练一次
     *      }
     *   3. action(state0).argmax()      —— 再基于**当前局面**做决策
     *
     * 也就是"先探索环境、预先训练, 再进行决策"。把它抽成 AgentBase 的接口, 各自
     * 实现的差别只在"用什么策略滚、滚完怎么训练、训练用哪个模型"。
     *
     * 有监督式的 agent (Alpha-Beta / MCTS) 没有需要在线更新的参数,
     * 默认实现什么都不做并返回 false —— 它们的"探索"就是搜索本身。
     *
     * 已实现该接口的: PGEagent / DQNAgent / PPOMCTSAgent / DQNMCTSAgent / EVABAgent。
     *
     * 实现的**硬性契约**: 返回时棋盘必须逐字节复原 (含 sideToMove), 因为调用方
     * (ChessBoard::preTrainThenDecide) 紧接着就要基于**当前**局面做决策。
     * 共用实现见 src/agentrollout.hpp 的 rolloutFromCurrent()。
     *
     *   color        : 轮到谁走
     *   rolloutSteps : 最多滚多少步
     *   opponent     : (P1, 可选) "对手在这个局面上会怎么走"。见上面的 OpponentPolicy:
     *                  budget = 0 (默认) 时行为与改动前**逐字相同** (纯自对弈);
     *                  budget > 0 时, 探索里"对手那一半"最前面若干手改由**真实对手**
     *                  产生。它只推进局面, **不**记成学习方的样本 (理由见那里的注释)。
     * 返回 true 表示这次确实做了在线训练。
     */
    virtual bool exploreAndTrain(int color, int rolloutSteps,
                                 const OpponentPolicy &opponent = OpponentPolicy())
    {
        (void)color;
        (void)rolloutSteps;
        (void)opponent;
        return false;
    }

    /* 最近一次 exploreAndTrain 的说明, 供界面显示 */
    virtual std::string getExploreInfo() const { return m_exploreInfo; }

    /*
     * 最近一次在线训练的损失, 供界面画"训练损失曲线"(见 mainwindow / metricsview)。
     *
     * 返回 NaN 表示这个 agent **不上报**损失 (纯搜索 agent 没有可训练参数, 或者该
     * agent 的训练循环里没有 scalar loss 可报)。曲线控件对非有限值是**直接丢弃**的,
     * 所以不上报不会画出一条假的水平线, 只会没有点。
     *
     * 已上报的: SACAZAgent (critic 的 MSE)、DQNAgent (平均平方 TD 误差)、
     * EVABAgent (价值网蒸馏的 MSE)。
     */
    virtual float getLastTrainLoss() const
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    /*
     * ---- agent 自检报告 (界面右侧"模型自检"面板) ----
     *
     * 为什么要有它: 训练损失与自对弈胜率都**不能**判断"这个模型值不值得继续训" ——
     * 损失只说明网络与自己的目标一致, 自对弈胜率里赢家和输家是同一份权重。
     * 真正的前置判据落在**表示层与终局口径**上 (动作编码有没有别名、状态能不能
     * 观测到规则历史、终局信号有没有真的进过目标), 这些量原来在界面里一个都看不到,
     * 只能靠命令行探针。见 test/probe_dqnmcts_aliasing_main.cpp 的实测。
     *
     * 契约:
     *   * **只读、可重复调用、不动棋盘** —— 它会在对局中途被 GUI 线程调用, 而棋盘
     *     正被搜索使用。要算"动作别名"这类统计就必须用 `Chess` 的**副本 + 局部解码**,
     *     绝不能碰 `this->chess` (那会与搜索线程抢同一个棋盘)。
     *   * 返回多行纯文本 (`\n` 分隔), 供 QPlainTextEdit 直接显示。
     *   * 空字符串 = 这个 agent 没有可报告的自检项 (默认实现)。
     *   * 报告的是**结构/口径**类事实, **不是棋力**。棋力只有带置信区间的锚点对局
     *     (bench_anchor) 能回答 —— 报告里要写明, 免得又被读成"模型变强了"。
     *
     * 已实现: DQNMCTSAgent (表示健康度 + 终局通道计数)。
     */
    virtual std::string selfCheckReport() const { return std::string(); }

protected:
    std::string m_exploreInfo;
};

/*
 * 权重文件读写状态的校验。
 *
 * RL 层的 save()/load() 不返回状态 (Net::save() 返回的 -1 被丢掉了), 于是各 agent
 * 的 saveModel()/loadModel() 一律 `return true` —— 写盘失败时 GUI 照样弹"保存成功"。
 * 这里用"文件存在且非空"给出真实结果, 不改变 RL 层的接口。
 */
inline bool weightFileWritten(const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return false;
    }
    return f.tellg() > 0;
}

inline bool weightFileReadable(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

/*
 * 终局值的**可选 agent 钩子** 之类的小工具放在各自文件里; 这一个专门给"探索说明"
 * 用: 把对手参数**真的生效了多少**写成半句话 (P1)。
 *
 * 为什么必须写出来而不是只报"开了对手参数": "参数传进来了"与"对手着法真的被用上了"
 * 是两件事 —— 对手返回无效着法、或者它的着法不在当前合法集里 (编码/规则不一致) 时
 * 都会回退到自对弈, 而那种回退在读数上与"没开这个功能"完全一样。unmatched 就是
 * 这条回退的计数: 它一直涨说明对手的着法表示与棋盘对不上 (那不是"对手弱")。
 */
inline std::string opponentRolloutInfo(const OpponentPolicy &o)
{
    if (!o.valid()) {
        return std::string();      /* 没开: 说明里不出现这一项 (默认口径逐字不变) */
    }
    std::string s = ", 对手(" + (o.name.empty() ? std::string("未知") : o.name) + ")应手 "
                    + std::to_string(o.used) + "/" + std::to_string(o.budget) + " 手";
    if (o.unmatched > 0) {
        s += ", 回退 " + std::to_string(o.unmatched) + " 次 (对手着法不在合法集里)";
    }
    return s;
}

#endif // AIAGENT_H
