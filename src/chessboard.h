#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QElapsedTimer>
#include <QMutex>
#include <QWaitCondition>
#include <QVector>
#include <QString>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>   /* finishDecision 的"决策回调"参数 (P0-a 对弈模式) */
#include <vector>
#include <map>
#include <fstream>
#include "chess.h"
#include "gamedb.h"
#include "abagent.h"
#include "mcts.h"
#include "pgagent.h"
#include "dqnagent.h"
#include "ppomcts_agent.h"
#include "dqnmcts_agent.h"
#include "evagent.h"
#include "sacazagent.h"
#include "dqnabagent.h"

/*
   AGENT_SACAZ_OLD 的实例类型。这里只**前置声明**就够了 (成员是指针, 上报损失是个模板),
   真正的定义在 src/sacazlegacyagent.cpp 里 —— 头文件不必把那份实现拖进来。
   [2026-09] 它不再是 SACAZAgent 的派生类, 所以不能拿 SACAZAgent* 存它 (那是编译错误,
   刻意如此: 两个类现在是**独立实现**)。见 sacazlegacyagent.h。
*/
class SACAZLegacyAgent;

class ChessBoard : public QWidget
{
    Q_OBJECT
public:
    enum State {
        STATE_IDEL = 0,      /* 等待玩家走棋 */
        STATE_THINKING,      /* AI正在后台思考 */
        STATE_TERMINATE      /* 游戏结束 */
    };

    /* AI Agent type identifiers */
    enum AgentType {
        AGENT_ALPHABETA = 0,    /* Alpha-Beta Pruning (ABAgent) */
        AGENT_MCTS,              /* Monte Carlo Tree Search */
        AGENT_PG,                /* Policy Gradient (PGEagent) */
        AGENT_DQN,               /* Deep Q-Network (DQNAgent) */
        AGENT_PPOMCTS,           /* PPO + MCTS AlphaZero-style (PPOMCTSAgent) */
        AGENT_DQNMCTS,           /* DQN + MCTS (DQNMCTSAgent) */
        AGENT_EVAB,              /* EVAB: 学会评估的 Alpha-Beta (EVABAgent) */
        AGENT_SACAZ,             /* SAC + MCTS + AlphaZero (SACAZAgent) */
        /*
           SAC + MCTS + AlphaZero, 但骨干换成**稀疏路由 MoE + TransformerBlock
           专家** (E=4, top-1): 参数量 ~4 个 TB 专家, 算力只算 1 个。
           代价实测 ~10.9 ms/模拟 (MLP 骨干 0.07 ms/模拟), 所以模拟次数要小得多
           (见 SACAZ_MOE_SIMS), 一次走子约 175 ms。
        */
        AGENT_SACAZ_MOE,
        /*
           DQN+AB: **把 Alpha-Beta 当成 DQN 的 planning head**。
           网络 (稀疏 MoE + TB 专家骨干 + Dueling 双头: V + A) 学 Q(s,a); AB 用它排序、
           用它当叶子; TD 目标来自"从 s' 展开若干层后的值"。搜索与训练细节见
           src/dqnabagent.h 顶部的长注释。
        */
        AGENT_DQNAB,
        /*
           PPO+MCTS+AlphaZero 的**另一个骨干**: 稀疏 MoE + **MLP 专家** (E=8 top-2),
           也就是 2026-09 那次改版之前的配置 (当时它是唯一骨干, 见 rl/ppo.h 顶部)。
           与 AGENT_PPOMCTS (TransformerBlock<16,360> 专家, E=4 top-1) 的关系:
             * **算法、搜索、训练、自检是同一份代码** (同一个 PPOMCTSAgent 类, 只是
               构造时传的 Backbone 不同) —— 与 AGENT_SACAZ / AGENT_SACAZ_MOE 同一种
               做法, 所以两者能在界面上直接对弈比较, 而不是两份会漂移的实现;
             * 差别只在骨干: MlpExpert 便宜 ~25x (前向 0.139 ms vs 3.59 ms)、容量小
               ~18x (2.15 M vs 38.0 M 参数), 所以同一时间预算下它能跑更多模拟
               (界面预算见 chessboard.cpp 的 PPO_MLP_SIMS);
             * **权重文件不同** (weights/ppomcts_mlp_agent.dat): 两种骨干参数量不同,
               权重格式的结构指纹也保证交叉载入当场失败, 不会静默串权重。
           **必须追加在枚举末尾**: 这些值会经 GUI 下拉框的 userData 传出去
           (见 mainwindow.cpp 的 kAgents), 插在中间会静默改变既有 agent 的编号。
        */
        AGENT_PPOMCTS_MLP,
        /*
           SAC+MCTS+AlphaZero 的**行为还原版**: 复现提交 59e5233 的那一支。
           它是一个**独立的 C++ 类** SACAZLegacyAgent (src/sacazlegacyagent.h/.cpp),
           不是同一个类里的运行时开关 —— 用户口径 (2026-09): "用不同的 C++ 类把新旧
           SAC agent 区分开"。
           [2026-09 后续] 它**不再继承** SACAZAgent (那是它 2026-09 之前的样子):
           用户口径是"以后任何对 SACAZAgent 的默认值或实现改动都不可能渗进还原版",
           所以这一支自带一份 SAC 实现 —— 见 sacazlegacyagent.h 的头注释 (那里有一张
           口径表 + 一条"不许再继承回去"的 static_assert), 代价是共享算法上的修复要
           **刻意**决定要不要同步过来。与 AGENT_SACAZ 的关系:
             * **两份独立实现** (代码是刻意的 1:1 拷贝 + 59e5233 的口径): 那边改默认值
               或改行为, 这边**不会跟着变** (这正是要的效果);
             * 口径按 `git show 59e5233:src/sacazagent.cpp` 逐项核对过: 目标熵 0.98 /
               alpha 学习率 1e-3 / critic 目标**不钳位且纯 MSE** / 叶子估值走**全量** /
               没有"从自己的搜索学一次" / 目标网 tau=1e-3 每 64 步。完整表见
               sacazlegacyagent.h 的头注释;
             * **不含奖励塑形、不含任何 critic 值域约束** (用户 2026-09 追加口径):
               连成员都没有, 任何 flag 都打不开;
             * **权重文件独立** (`weights/sacaz_old_agent_*`): 两者参数结构完全相同
               (都是 iFcLayer 的 w/b), 结构指纹挡不住串权重; 而训练口径不同 ⇒ 共用
               前缀会让两边**静默**互相覆盖 (用户口径: 新旧权重必须用不同名字)。
           **必须追加在枚举末尾**: 这些值会经 GUI 下拉框的 userData 传出去
           (见 mainwindow.cpp 的 kAgents), 插在中间会静默改变既有 agent 的编号。
        */
        AGENT_SACAZ_OLD,
        /*
           ---- 59e5233 行为还原版的**另一个骨干**: 稀疏 MoE + TransformerBlock 专家 ----
           与 AGENT_SACAZ_OLD 的关系 = AGENT_SACAZ_MOE 与 AGENT_SACAZ 的关系:
           **同一个类 (SACAZLegacyAgent)、同一套 59e5233 口径**, 只有构造时传的
           Backbone 不同 (Mlp -> SparseMoeTb)。用途是在**同一个还原口径**下量"骨干换
           TB 专家值多少", 而不是让骨干与口径两个变量混在一次对比里。
             * 口径仍然全部硬编码在那个类里 (目标熵 0.98 / alpha 学习率 1e-3 /
               critic 目标不夹 + 纯 MSE / 叶子估值全量 / 目标网 tau=1e-3 每 64 步);
               本类型**不引入任何新开关** —— 它只是同一个类的另一个骨干;
             * 算力贵得多 (一个 TB 专家前向实测 ~3.2 ms, MLP 骨干 0.07 ms/模拟),
               所以模拟次数按 AGENT_SACAZ_MOE 那一档给 (SACAZ_MOE_SIMS = 16,
               后台训练 BG_TRAIN_SACAZ_MOE_SIMS = 64);
             * **权重文件独立** (`weights/sacaz_old_moe_agent_*`): 前缀与
               AGENT_SACAZ_OLD 的不同 —— 两者的参数量差不多 (都是 iFcLayer 的 w/b),
               命名上分开才不用靠猜"这是哪一支的权重"。
           **必须追加在枚举末尾**: 这些值会经 GUI 下拉框的 userData 传出去
           (见 mainwindow.cpp 的 kAgents), 插在中间会静默改变既有 agent 的编号。
        */
        AGENT_SACAZ_OLD_MOE,
        /*
           ================================================================
           ---- Alpha-Beta 的三档弱等级 (2026-09, 用户口径: "1 到 3 level 加进下拉框") ----
           ================================================================
           AGENT_ALPHABETA 是同一个搜索、深度 AB_DEPTH(=4)。这三档**只是深度不同**:
               AGENT_AB_L1 -> 深度 1
               AGENT_AB_L2 -> 深度 2
               AGENT_AB_L3 -> 深度 3
           【实测 2026-09 · `test_ab benchmark` · 本机 · 初始局面】每步 0 / 3 / 34 ms
           (对照: 深度 4 = 90 ms, 深度 5 = 1920 ms)。

           用途: **当陪练与标尺**。它们是"棋力可调、且完全不随训练漂移"的对手 ——
           既要跟它下 (对弈), 也要拿它当"模型有没有变强"的参照。

           ⚠ **它们没有任何可训练的权重**: 叶子价值全部来自 Chess::evaluate()
           (材质 + 子力位置表, 常量写死在 chess.cpp), 所以:
             * exploreAndTrain() 走 AgentBase 的默认空实现 (aiagent.h:108),
               即"轮到它走"时**不会**训练, 也不可能被训练;
             * getLastTrainLoss() 恒为 NaN -> 损失曲线上没有它的点 (这是正确行为,
               不是"训练没跑起来"); selfCheckReport 里写明了这一句;
             * 与它下棋时, 学习型 agent 学的是**它自己滚出来的经验** —— 对手是谁
               不进入它的训练数据。要"照着 AB 的棋学"必须另接一条对手条件化的
               采样路径 (本轮未做, 见 docs 里的待办)。
           这一条不是可以靠"多跑几轮"绕过的: 本工程对"纯搜索 agent"的口径一直是
           "有棋力、没有学习" (见 aiagent.h 的 hasLearningReward / AB 的自检报告)。

           **必须追加在枚举末尾**: 同上面几条 —— 值会经 GUI 下拉框的 userData 传出去。
           **不要**把它们插到 AGENT_ALPHABETA 旁边去"排得好看"。
        */
        AGENT_AB_L1,
        AGENT_AB_L2,
        AGENT_AB_L3
    };

public:
    explicit ChessBoard(QWidget *parent = nullptr);
    ~ChessBoard();

    /* 回放功能 */
    bool isReplayMode() const { return m_replayGameId >= 0; }
    void loadReplayGame(int gameId, const QVector<DBStep> &steps);
    bool replayPrev();
    bool replayNext();
    int replayIndex() const { return m_replayIndex; }
    int replayTotal() const { return m_replaySteps.size(); }
    int replayGameId() const { return m_replayGameId; }

    /* Agent 选择 */
    void setAgentType(AgentType type);
    AgentType getAgentType() const { return m_agentType; }

    /*
     * ================================================================
     *  ---- 对弈模式 (P0-a, 2026-09, dev-selfplay) ----
     * ================================================================
     *
     * 为什么需要它 (用户提问: "对弈的 A/B 双方都是自己实现 (自对弈), 是否合理?"):
     * 在它之前, 界面上**没有** "这是训练还是评估" 这个概念 —— 每一手都会走
     * aiThinkForAgent -> preTrainThenDecide -> exploreAndTrain, 于是对弈总是"双方各自
     * 在学", 而"谁能赢"这件事因此**不可归因**: 得分变化既可能来自我方变强, 也可能来自
     * 对手(也在学)变弱。更糟的是"学"不止一条路径 (见下面 MatchMode 的说明) ——
     * 实测 (test_match [2.7c]) SAC+AZ 在关掉"探索+预训练"之后**仍然每手更新**。
     *
     * 三条语义 (由 perMoveLearningEnabled() / searchLearningEnabled() 统一给出):
     *
     *   MATCH_TRAIN   双方各自学习 —— **双方各自自对弈**, 不是互相学(对手的棋不进训练
     *                 数据, 见 aiagent.h 里 exploreAndTrain 没有对手参数这件事)。
     *                 这是"训练对局"。
     *   MATCH_EVAL    **冻结对手, 只让一方学**: 谁被冻结是调用方的事 —— 对弈里
     *                 "A 方是待评估者、B 方是参照物", 所以只让 A 那一侧学。
     *                 这是"评估对局": 比分变化因此可以归因到 A 自己的变化上。
     *   MATCH_NO_LEARN 双方都不学, **两条学习路径都关掉** ⇒ "只对弈不学习"的纯对照。
     *                 这是唯一能让"这一场比分"直接可比的口径 (docs/agents_design.md
     *                 §10.3 第 7 条: 棋力结论只能来自"预训练 = 0"的对照)。
     *
     * ⚠ MATCH_EVAL / MATCH_NO_LEARN 都必须**同时**关掉两条学习路径, 否则"不学"是假的:
     *     ① preTrainThenDecide 的探索+预训练 (受界面"探索步数"控制);
     *     ② SAC+AZ 的 learnFromSearch (在 selectMove 里, **不受那个勾选框控制**,
     *        原来界面上根本关不掉它)。
     *   两条都由本模式统一管辖 —— 这正是 [2.7c] 实测出来的那个洞。
     *
     * ⚠ 本模式**只作用于对弈** (matchAgents)。人机对战那条路 (aiThink) 不受影响。
     * ⚠ 对局的**双方**用同一个模式 (不是每方一个开关): "评估对局"的定义是"A 学、B 冻结",
     *   把它做成两个独立开关会允许"A 学、B 也学"(= MATCH_TRAIN) 与"A 不学、B 也不学"
     *   (= MATCH_NO_LEARN) 这两种等价写法, 于是"模式"这个词就失去意义了。
     */
    enum MatchMode {
        MATCH_TRAIN = 0,      /* 双方各自学习 (默认; 与改动前的行为一致) */
        MATCH_EVAL,           /* 冻结对手, 只让 A 方学 */
        MATCH_NO_LEARN        /* 双方都不学 (只对弈) */
    };
    void setMatchMode(MatchMode m) { m_matchMode = m; }
    MatchMode getMatchMode() const { return m_matchMode; }
    /* 界面/报告用的名字 (同时是"这一场到底学不学"的唯一表述) */
    static QString matchModeName(MatchMode m);
    /* 这一场会不会发生在线学习 (用于对局报告与界面提示) */
    bool matchLearnsSomething() const;

    /* 启动加载: 异步加载数据库和AI模型权重 */
    void startupLoad();
    bool isStartupComplete() const { return m_startupComplete.load(); }

    /*
     * ---- Agent 对 Agent 对弈 ----
     *
     * A/B 是"两个参赛者", 不是红黑。中国象棋先手优势很大, 固定谁执红的话
     * 结果只是在测"谁执红", 所以 matchAgents 每局交换先后手, 胜负按参赛者统计。
     */
    struct MatchStats {
        QString agentA;
        QString agentB;
        int games = 0;              /* 实际打完的局数 (可能被中止) */
        int winA = 0;
        int winB = 0;
        int draws = 0;
        int plies = 0;              /* 总手数 */
        int agentErrors = 0;        /* agent 返回无效走法的次数 (用合法走法兜底) */
        long long totalThinkMs = 0; /* 累计思考时间 */
        long long maxThinkMs = 0;   /* 单步最长思考时间 */
        QString log;                /* 每局一行 */
        bool aborted = false;
        /*
           本场的对弈模式 (P0-a)。**必须有**: 比分本身不带前提, 而"这一场学不学"
           决定了它能不能被读成棋力结论。由 matchAgents 在开场时写进报告。
             modeName        : 模式名 (界面/报告显示)
             learnsSomething : 本场会不会发生在线学习 (用于那句"不能当棋力结论"的提示)
        */
        QString modeName;
        bool learnsSomething = true;
        /*
           ================================================================
           [O1, 2026-09] 吃子行为 —— "该吃的时候吃了吗"（整场累计）
           ================================================================
           用户报的现象是"杀将棋有一定的效果, 但吃其他棋子又变得无动于衷"。这句话在界面上
           原来**没有任何读数**能回答: 奖励曲线不是合适的仪器 —— 实测 (见
           docs/capture_readings_2026_09.md) 单次吃一个車, 学习口径下只占纵轴 1.3 px
           (118 px 高的图), 换成引擎口径也只是 3.9 px, **而终局会从 26 px 缩到 8 px**:
           换口径是拆东墙补西墙, 不是修好。所以这件事只能靠**数字**回答, 就放在这里。

           口径 (与 rl/diag.h 的 isCaptureStep、bench_sac_learn 的"吃子行为"一节一致):
             * capAvail  : 轮到自己**且有**吃子着法可选的手数 (条件比例的分母);
             * capChosen : 真走了吃子着法的手数;
             * nextId != ID_NONE 就是吃子 (见 stone.h 的 Step 说明);
             * matGained : 吃到的材质**原值**, **不含将** (吃将必然是终局,
               value_jiang=1000 会把这一个数顶爆)。
           分母为 0 时 captureLine() 会说"无机会"而不是印 0.0% ——
           "一次机会都没遇到"与"有机会一次都没吃"是两件完全不同的事。
        */
        int capAvailA = 0, capChosenA = 0;
        int capAvailB = 0, capChosenB = 0;
        double matGainedA = 0.0, matGainedB = 0.0;
        QString summary() const;    /* 一行比分 */
        QString detail() const;     /* 比分 + 每局明细 + 耗时 */
        /* 一行"该吃的时候吃了吗" (A/B 各一个条件比例); 三处显示共用这一份格式化 */
        QString captureLine() const;
    };

    /* 让两个 agent 互相对弈 games 局 (每局交换先后手), 阻塞直到结束或被中止 */
    MatchStats matchAgents(AgentType typeA, AgentType typeB, int games);
    /*
     * 每局的手数上限 (默认 300)。达到上限即判和棋。
     * 主要给自动化测试用: 把上限调小就能在几秒内跑完一整场对弈, 从而验证
     * "交换先后手 / 比分归属 / 中止"这些逻辑, 而不必真下几百手。
     */
    static constexpr int DEFAULT_MAX_PLIES = 300;
    void setMaxPliesPerGame(int n) { m_maxPliesPerGame = n > 0 ? n : 1; }
    int getMaxPliesPerGame() const { return m_maxPliesPerGame; }
    /* 请求中止对弈: 在每一手之间检查, 最迟一手之内生效 */
    void abortMatch() { m_matchAbort = true; }
    bool isMatchRunning() const { return m_matchRunning.load(); }

    /* 保存当前agent的权重文件 */
    bool saveCurrentAgentModel(AgentType agentType, const std::string &filepath);
    /*
     * 把权重文件**装进该类型的常驻实例** (saveCurrentAgentModel 的反向操作)。
     *
     * 为什么需要它 (dev-selfplay, 2026-09): "对弈评估"要能回答"只换一个因素, 结果变不变",
     * 而 matchAgents 会让常驻 agent **就地更新** —— 两次实验之间必须能把权重还原到同一个
     * 出发点, 否则比出来的是训练历史, 不是那个因素 (test_match [2.7c] 第一版就是这么
     * 假失败的)。界面的"评估模式"也要用它来装冻结对手的权重。
     *
     * 语义:
     *   * 实例还没建 -> 先按**决策路径同一套构造参数**建出来 (与 aiThinkRaw 的分支逐字
     *     一致, 否则结构指纹对不上、load 会直接失败);
     *   * 纯搜索 agent (AB/MCTS 各档) 没有权重 -> 返回 false;
     *   * 载入失败 (文件不在 / 结构与指纹不匹配 / 参数量守卫没过) 返回 false, 且
     *     **不改动**已有网络 (各 agent 的 loadModel 保证失败时不半写)。
     */
    bool loadAgentModel(AgentType agentType, const std::string &filepath);
    /*
     * 某个 agent 的"标准权重路径" (weights/ 下的正式文件名)。
     * shutdownSave() 与"对弈结束后静默保存"都用它 —— 以前这两处各写一份, 很容易
     * 出现"存到 A 处、启动时读 B 处"这种静默失效。
     * SAC+AZ 系的一个模型是三个文件, 所以它返回的是**前缀**。
     */
    static std::string defaultWeightPath(AgentType agentType);
    /* 这个 agent 是否已经实例化过 (没实例化就没有权重可存, 静默保存要跳过) */
    bool hasAgentInstance(AgentType agentType) const;

    /*
     * 是否在每次走子前先"探索环境 + 预训练一次" (仿 snakeAI 的决策流程, 见 aiagent.h)。
     * 默认打开; 关掉就是"直接决策"(原来的行为)。
     */
    void setPreTrainEnabled(bool on) { m_preTrainEnabled = on; }
    bool isPreTrainEnabled() const { return m_preTrainEnabled; }
    /* 每次探索的步数上限; 设为 0 等价于不做探索 */
    void setPreTrainSteps(int steps) { m_preTrainSteps = steps; }
    int getPreTrainSteps() const { return m_preTrainSteps.load(); }
    /* 每次"探索环境 + 预训练"的说明 (线程安全) */
    std::string getLastExploreInfo() const;

    /*
     * 当前 agent 的**自检报告** (界面"模型自检"面板的数据源)。
     *
     * 转发给 `AgentBase::selfCheckReport()` (见 aiagent.h 的口径说明)。
     * 为什么要有这一层转发而不是让 GUI 直接拿 agent:
     *   * GUI 线程拿到的 agent 指针是 `m_agents` 里的那个, 而**自对弈训练跑在
     *     后台线程的 clone 上** —— 主 agent 的计数器在训练期间不会动。这个转发
     *     就是明确标注这件事的地方 (面板上会打印"主 agent 计数, 后台训练在 clone 上"),
     *     免得面板显示 0 时被读成"没训练过"。
     *   * 没有选中 agent / 该 agent 不支持自检时返回空串, 由 GUI 决定怎么显示。
     */
    std::string getAgentSelfCheck() const;

    /*
     * 指定 agent 类型的自检报告 (同上, 但可以查"不是当前选中"的那一个)。
     * 面板上的"全部模型自检"与启动日志都用它 —— 自检的价值在于**横向对比**:
     * 哪几个 agent 的动作编码有别名、哪几个能看见规则上下文, 一眼就能排出来。
     */
    std::string getAgentSelfCheck(AgentType type) const;

    /*
     * 这个 agent 的权重文件在磁盘上长什么样 (存不存在、多大)。
     *
     * 为什么要放进面板: "权重到底载进来了没有"是**静默失效**的高发区 ——
     * PPO+MCTS 就曾经因为"扫描的名字与实际写出的名字不一致"而从来没被载入过,
     * 界面上却看不出任何异常 (见 weightFilesOf 的注释)。把这几个文件的存在性
     * 与大小直接打在自检面板上, 这类问题就不再需要靠猜。
     */
    std::string getAgentWeightStatus(AgentType type) const;

    /* 程序退出时保存所有已初始化的agent权重 */
    void shutdownSave();

    /* 后台持续训练: 克隆agent在后台自我对弈, 每4轮同步权重回主agent */
    void startBackgroundTraining();
    void stopBackgroundTraining();

    /*
     * 后台训练**一轮**的规模 (默认 1 局 x 60 手, 见 chessboard.cpp 的 BG_TRAIN_*)。
     *
     * 为什么要有这个开关: 一轮的时长直接决定"关窗要等多久" (训练线程只在每轮开头看
     * 停止标志), 而不同 agent 的一轮成本差两三个数量级 —— PPO+MCTS / DQN+MCTS 是
     * 分钟级, SAC+AZ-MoE 几十秒, PG/DQN 秒级。把它做成可调的, 测试就能把一轮缩到
     * 几手, 在秒级验证"某个 agent 的训练往返到底接没接上" (EVAB 曾经因为支路根本
     * 不存在而空转了很多轮, 正是缺这样一条断言)。
     * 参数会被夹到 >= 1; 下一轮开始时生效 (不会打断正在跑的那一轮)。
     *
     * ⚠️ maxMoves 低于 **32** 时, SAC+AZ / DQN+AB 这一轮会一次梯度更新都不做
     * (它们的 learnBatch 在"回放池 < batchSize(32)"时直接返回), 损失曲线也不会上报
     * —— 表现是"训练在跑但什么都没发生"。缩短轮次只适合验证**接线**是否通。
     */
    void setBackgroundTrainRound(int episodes, int maxMoves);
    int getBackgroundTrainEpisodes() const { return m_bgTrainEpisodes.load(); }
    int getBackgroundTrainMaxMoves() const { return m_bgTrainMaxMoves.load(); }

signals:
    /*
     * 终局结果, 取值是 Chess::Result (RESULT_RED_WIN / RESULT_BLACK_WIN /
     * RESULT_DRAW)。以前这里传的是 Stone::COLOR_* 并且用 COLOR_NONE 表示平局,
     * 结果"无法判断平局"这件事在整条链路上都表达不出来。
     */
    void sendResult(int result);
    /* AI思考完成, 耗时(毫秒) */
    void aiThinkFinished(long long elapsedMs);

    /*
     * ---- 思考过程可视化 (在后台线程 emit, 队列投递到 GUI 线程) ----
     *
     * 思考期间 mousePressEvent 会拒绝落子 (这是对的: 轮到 AI 走), 但界面上原本
     * 没有任何反馈 —— 玩家看到的是"棋子没动, 我也点不动", 分不清"还在算"和"卡死"。
     * 加入"走子前先探索+预训练"之后单步思考涨到秒级, 这三个信号就是为此加的:
     *   aiThinkingStarted : 开始思考 (谁在想 + 本次探索步数上限)
     *   aiThinkingStage   : 阶段推进 ("① 探索环境 + 预训练" / "② 搜索 / 决策" / "③ 落子")
     *   aiThinkingStopped : 思考真正结束 (含"思考途中按了开局"这种被打断的情况)
     */
    void aiThinkingStarted(const QString &agentName, int exploreSteps);
    void aiThinkingStage(const QString &stage);
    void aiThinkingStopped();
    /* 本次"探索环境 + 预训练"的结果说明, 供界面显示 */
    void aiExploreInfo(const QString &info);

    /* ---- Agent 对弈进度 ---- */
    void matchStarted(const QString &agentA, const QString &agentB, int games);
    void matchGameFinished(int gameNo, int games, const QString &line);
    void matchFinished(const QString &summary, const QString &detail);
    /*
     * ---- 对弈实时比分 ----
     * 每打完一局 emit 一次。界面用它把"当前几比几"实时显示出来 (以前只有在整场
     * 结束时才弹一个结果框, 中途只能看到"第 n/N 局完成")。
     * scoreLine 形如 "A 2 : 1 B   和 1", 已经算好可以直接显示。
     */
    void matchScoreChanged(const QString &scoreLine);
    /*
     * ---- 指标曲线用的采样 ----
     * gameRewardSample   : 每局结束后两位参赛者各自拿到的**本局环境奖励累计**
     *                      (走子方视角; 即时奖励与终局值都在里面)
     * matchRewardProgress: 一局**进行中**每手一次的"本局累计"环境奖励 (同样走子方
     *                      视角, 同样含即时奖励, 但**不含**局末的终局值 —— 那个由
     *                      gameRewardSample 补上最后一点)
     * trainLossSample    : 每完成一次在线训练上报一次损失 (哪个 agent / 第几次)
     * 三者都在后台线程 emit, 队列投递到 GUI 线程后进曲线。
     *
     * **[2026-09 ④] 奖励的"口径"**: 奖励曲线现在取 agent **自己的学习口径**
     *   (材质 x REWARD_MATERIAL_COEF(0.1) + 每步代价 + 终局; SAC 开着塑形时终局是
     *    ±(1+败方材质/3.5))。没有学习口径的纯搜索 agent (Alpha-Beta / MCTS / EVAB)
     *   仍旧是引擎口径 (材质 x1 + 终局 ±1), 曲线名与逐局明细里都会标出来。
     *   **两个口径差 10 倍** —— 同一张图上混着两种口径的线时不可直接比大小
     *   (见 RewardAccounting 与 docs/sac_learn_reward_2026_09.md §1.1)。
     *
     * 为什么要有 matchRewardProgress: 一局可能有几百手、跑十几分钟 (实测 276 手
     * 621 秒), 而 gameRewardSample 一局只发一次 —— 用户看到的是"对弈时奖励曲线
     * 一直不动", 会以为曲线坏了。现在每手一个点, 曲线在对局过程中就在走。
     */
    void gameRewardSample(int gameNo, const QString &agentA, const QString &agentB,
                          double rewardA, double rewardB);
    void matchRewardProgress(int gameNo, int ply, double rewardA, double rewardB);
    void trainLossSample(double loss, const QString &agent, int step);

    /*
     * ---- "请稍候"(载入/保存模型权重) ----
     * 权重读写可能很慢 (老格式的十进制文本权重一个文件 16 MB; 稀疏 MoE 骨干
     * 28.7 M 参数), 期间界面必须给出"在干活"的反馈, 否则看起来就是卡死。
     * 这里只负责**报告状态**, 弹窗长什么样由界面决定 (见 src/busydialog.h)。
     *   busyStarted  : 开始读写 (标题 + 第一行说明)
     *   busyMessage  : 进度说明变了 (例如"正在载入 EVAB 权重…")
     *   busyFinished : 结束 (无论成功失败都要发, 否则弹窗会一直挂着)
     */
    void busyStarted(const QString &title, const QString &message);
    void busyMessage(const QString &message);
    void busyFinished();
    /* 回放状态变更信号 */
    void replayIndexChanged(int index, int total);
    void replayModeExited();
    /* 启动加载完成 */
    void startupComplete();
public slots:
    void checkGameOver(int result);
    void reset();
private:
    Pos getStonePos(const QPoint &pos);
    QPoint getStoneCenter(int x, int y);
    QRect getRect(QPoint &center);
    void drawStone(QPainter &p, const Stone *stone);
    Stone *selectStone(const QPoint &point);
    bool moveStone(const QPoint &point);
    void process();
    /* 回放内部: 将棋盘重置到指定步数 */
    void applyReplayStep(int targetIndex);
    /* 回放内部: 按数据库记录 (起点/终点坐标) 构造一步并落子 */
    bool applyDbStep(const DBStep &dbStep);
    /* AI决策 - 根据当前选中的agent类型选择走法 */
    Step aiThink(int color);

    /* Self Play 内部: 使用指定agent决策 */
    Step aiThinkForAgent(int color, AgentType agentType);

    /*
     * 上面两个函数的"纯决策"部分 (不带合法性闸门)。
     * 拆出来的理由: 决策结果必须先过一遍 legalStepOrFallback() —— 直接在两个大
     * switch 的每个 return 上套一层的话, 以后新增 agent 分支时必然漏掉几个。
     */
    Step aiThinkRaw(int color);
    Step aiThinkForAgentRaw(int color, AgentType agentType);
    /*
     * 决策输出的合法性闸门 (见 chessboard.cpp 的实现注释):
     * agent 返回无效走法、而棋盘上还有合法走法时, 用第一个合法走法兜底并报警。
     * 真的无棋可走 (将杀/困毙/和棋前的终局) 时把无效 Step 原样返回。
     */
    Step legalStepOrFallback(int color, const Step &step, const QString &who);
protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
private:
    constexpr static int offsetX = 50;
    constexpr static int offsetY = 50;
    constexpr static int gridSize = 60;
    constexpr static int stoneRadius = 24;
    Chess chess;
    Chess env;
    int selectID;
    int color;
    std::atomic<State> state;
    /*
     * mutex 声明成 mutable: 自检 (const 方法) 要对真棋盘取一份**副本**再交给 agent,
     * 所以它必须能在 const 里上锁 (见 ChessBoard::getAgentSelfCheck 的说明)。
     * 保护的仍然是同一批数据 (chess / env / state), 语义没变。
     */
    mutable QMutex mutex;
    QWaitCondition condit;
    std::thread processThread;
    /* Self-Play 期间为 true: 只用于屏蔽玩家点击 (AI 工作线程不看它) */
    std::atomic<bool> m_selfPlaying{false};
    /* 数据库记录 */
    int m_currentGameId;
    int m_moveCount;
    bool m_dbEnabled;
    /* 回放状态 */
    int m_replayGameId;
    int m_replayIndex;
    QVector<DBStep> m_replaySteps;
    /* AI Agent */
    AgentType m_agentType;

    /* Self Play agent 实例 (静态以保证跨函数调用存活) */
    static PGEagent *m_sfPG;
    static DQNAgent *m_sfDQN;
    static PPOMCTSAgent *m_sfPPOMCTS;
    /* PPO+MCTS 的 MLP 专家骨干那一个变体 (AGENT_PPOMCTS_MLP): 同一个类, 不同 Backbone */
    static PPOMCTSAgent *m_sfPPOMCTSMLP;
    static DQNMCTSAgent *m_sfDQNMCTS;
    static EVABAgent *m_sfEVAB;
    static SACAZAgent *m_sfSACAZ;
    static SACAZAgent *m_sfSACAZMoe;   /* 稀疏 MoE + TB 专家骨干的那个变体 */
    static SACAZLegacyAgent *m_sfSACAZOld;      /* 行为还原版: 独立类 (59e5233), MLP 骨干 */
    static SACAZLegacyAgent *m_sfSACAZOldMoe;   /* 同上, 但骨干换成稀疏 MoE + TB 专家 */
    static DQNABAgent *m_sfDQNAB; /* AB 当 DQN 的 planning head (见 dqnabagent.h) */

    /* "走子前先探索环境 + 预训练"开关 (仿 snakeAI) */
    std::atomic<bool> m_preTrainEnabled{true};    std::atomic<int> m_preTrainSteps{64};
    /* 训练损失样本的序号 (背景线程与 AI 线程都会 +1, 故用 atomic) */
    std::atomic<int> m_trainSampleNo{0};
    std::string m_lastExploreInfo;
    mutable QMutex m_infoMutex;            /* 保护 m_lastExploreInfo (跨线程读写) */

    /*
     * 对当前 agent 执行一次"探索环境 + 预训练", 然后是决策。
     * 返回探索出来的说明文字 (给界面用)。
     */
    std::string preTrainThenDecide(AgentBase *agent, int color);

    /*
     * [2026-09 新] 决策里"从自己的搜索学了一次"之后, 把损失送上损失曲线。
     *
     * 为什么需要它: 界面的 per-move 学习原来**只**由 preTrainThenDecide 驱动 ——
     * 关掉"探索+预训练"勾选框就完全不训练, 损失曲线也永远是空的 (用户实测报的现象)。
     * SAC 现在在 selectMove 里会把自己的搜索样本学一次 (SACAZAgent::learnFromSearch),
     * 但那一次更新发生在 preTrainThenDecide **之外**, 没人上报 —— 于是"都在学, 曲线
     * 却不动"就又出现了。这个方法把那条路径补上。
     *
     * 判据是 **learnSteps 是否前进** (不是"每手无条件上报"): 一次决策最多可能有两次
     * 更新 (rollout 一次 + 搜索样本一次), 无条件上报会把"每手一个点"的口径弄乱。
     *
     * [2026-09 独立类拆分] 这里是个**模板**, 参数从 `SACAZAgent *` 放宽成"任何有
     * getLearnSteps() / getLastTrainLoss() / getName() 的 agent": 59e5233 还原版
     * (SACAZLegacyAgent) 现在**不是** SACAZAgent 的派生类, 一个非模板签名接不住它,
     * 而两个类之间**不许**做类型转换 (见 sacazlegacyagent.h 的头注释)。实际口径写在
     * reportLearnedLossOf() 里 —— 只有一份, 不会两边漂移。
     */
    template <class AgentT>
    void reportLearnedLoss(AgentT *agent, int teachStepsBefore)
    {
        /*
           getLearnSteps() 只在"会学习的 agent"上有 (AgentBase 没有这个接口), 所以读取
           必须留在模板里: 模板参数负责**取数**, 非模板的 reportLearnedLossOf 负责**口径**。
        */
        reportLearnedLossOf(agent, agent != nullptr ? agent->getLearnSteps() : 0,
                            teachStepsBefore);
    }

    /* reportLearnedLoss 模板的公共实现 (定义在 .cpp: 信号与曲线口径不进头文件) */
    void reportLearnedLossOf(AgentBase *agent, int learnSteps, int teachStepsBefore);

    /* ----------------------------------------------------------------
     *  ---- 对弈模式 (P0-a) 的判据与决策收尾钩子 ----
     *  只有一处实现, 决策路径与"装/拆掩码"都从这里取, 免得两处漂移。
     * ---------------------------------------------------------------- */

    /*
     * 这一手**要不要做"探索环境 + 预训练"**。
     *  = 不在对弈中 (人机对战照旧) 或 当前对局处于 MATCH_TRAIN;
     *    再叠加界面上的"探索+预训练"开关与步数 (>0)。
     * 注意顺序: "不在对弈中"必须放前面 —— aiThink (人机) 与 aiThinkForAgent (对弈)
     * 共用这一条判据, 而人机对战不该被对弈模式影响。
     */
    bool perMoveLearningEnabled() const;
    /*
     * 这一手**要不要更新** (泛指: 回放池里攒样本 / learnBatch / learnFromSearchStep)。
     * MATCH_NO_LEARN 一律 false; MATCH_EVAL 时只有 isA 那一侧为 true。
     * 不在对弈中 -> true (人机对战保持原有行为)。
     */
    bool updateEnabledForSide(bool isA) const;
    /*
     * 决策的两个收尾钩子 (把"决策 + 损失上报 + 模式掩码"收在一处)。
     *
     * 为什么要有它们: 原来每个 RL 分支各自写 `const int stepsBefore = ...; selectMove(...);
     * reportLearnedLoss(...)`, 而 SAC 系还要额外处理 learnFromSearch —— 五处重复。
     * 模式一加进来, 那种写法必然漏掉某一支, 于是"评估模式"在某些 agent 上静默失效。
     *
     * 两个都是**模板** (与 reportLearnedLoss 同一手法): getLearnSteps() / learnFromSearch
     * 只在具体 agent 上有, AgentBase 上没有这些接口 —— 一个非模板签名要么接不住它们,
     * 要么就得在基类上开洞。
     *
     *   finishDecision       : 普通 RL 分支 (PG / DQN / PPO+MCTS / DQN+MCTS / EVAB / DQNAB)
     *   finishDecisionSearch : 带 learnFromSearch 的分支 (SAC+AZ 的两支)
     */
    template <class AgentT>
    Step finishDecision(AgentT *agent, const std::function<Step()> &decide)
    {
        const int stepsBefore = agent->getLearnSteps();
        Step s = decide();
        reportLearnedLoss(agent, stepsBefore);
        return s;
    }
    template <class SACLike>
    Step finishDecisionSearch(SACLike *agent, const std::function<Step()> &decide)
    {
        const int stepsBefore = agent->getLearnSteps();
        /*
           掩码: 本模式若禁止"从自己的搜索学一次", 就在这一手内把它关掉, 决策完再还原。
           用 before 暂存原值 —— 还原回去的是**agent 自己的配置**, 不是我们改成的值,
           所以不会把用户/测试设过的 learnFromSearch=false 悄悄打开。
        */
        const bool before = agent->learnFromSearch;
        agent->learnFromSearch = before && searchLearningEnabledForSide(m_sideBeingDecided);
        Step s = decide();
        agent->learnFromSearch = before;
        reportLearnedLoss(agent, stepsBefore);
        return s;
    }
    /* "从自己的搜索学一次"(SAC 的 learnFromSearch) 这一手允不允许 */
    bool searchLearningEnabledForSide(bool isA) const;
    /*
     * 诊断计数: 本次对局里"因为模式而被禁止更新"的次数 (每手最多一次)。
     * 为什么需要它: 模式生效与否**在读数上极难分辨** —— "B 侧更新 20 次而不是 40 次"
     * 既可能是"掩码关掉了一条路径", 也可能是"另一条路径本来就没触发"。只有把闸门
     * 被踩下的次数直接数出来, 才能区分这两种情况 (第一版 [2.7d] 就卡在这里)。
     */
    std::atomic<int> m_matchLearningBlocked{0};
public:
    int matchLearningBlockedCount() const { return m_matchLearningBlocked.load(); }
private:
    /*
     * 当前这一手是在替哪一方决策 (对弈里 "红方是不是 A 方")。
     * 人机路径 (aiThink) 保持 true —— 它不在对弈里, 两个判据都只看"是否在对弈中"。
     * 由 playMatchGame 在每一手之前写 (同一线程, 与决策串行)。
     */
    bool m_sideBeingDecided = true;

    /* ---- 思考过程可视化 (全部只在 GUI 线程读写, 除了 m_thinkGeneration) ---- */
    void initThinkVisuals();
    /* 棋盘顶部那条状态提示的矩形 (只看这块重绘, 不重画整个棋盘) */
    QRect thinkingOverlayRect() const;
    void drawThinkingOverlay(QPainter &painter);
    /* 发阶段信号时带上"对弈 2/4 局 · 第 17 手"之类的前缀 */
    void emitStage(const QString &stage);
    QString stagePrefix() const;
    /* "开局"会 +1, 用来丢弃"思考途中被重开"的那一步棋 */
    std::atomic<unsigned> m_thinkGeneration{0};
    std::atomic<int> m_selfPlayMoveNo{0};
    QTimer *m_animTimer = nullptr;         /* 状态条动画, ~25 fps */
    int m_animPhase = 0;
    QElapsedTimer m_thinkClock;            /* 本轮思考的实时计时 */
    QString m_thinkStage;                  /* 当前阶段文字 */
    QTimer *m_busyClickTimer = nullptr;    /* "思考中点击已被忽略"提示的消退计时 */
    bool m_busyClickSeen = false;

    /* ---- Agent 对弈状态 (对弈线程写, 状态条/前缀读, 故用 atomic) ---- */
    std::atomic<bool> m_matchRunning{false};
    std::atomic<bool> m_matchAbort{false};
    std::atomic<int> m_matchGameNo{0};
    std::atomic<int> m_matchGames{0};
    /*
       对弈模式 (P0-a): 写在"开始对弈"之前 (GUI/测试线程), 由对弈线程每手读。
       atomic 是因为它可能在另一线程被改 (例如勾选框变化), 而决策路径每手都要查它。
    */
    std::atomic<MatchMode> m_matchMode{MATCH_TRAIN};
    /* 对弈中"A 方是不是执红" (每局开始写; MATCH_EVAL 靠它判断这一手替谁决策) */
    bool m_matchAIsRed = true;
    int m_maxPliesPerGame = DEFAULT_MAX_PLIES;   /* 单局手数上限, 到顶判和 */
    /*
     * 打一局: 红方用 redType, 黑方用 blackType; 返回 Chess::RESULT_*。
     * aIsRed 说明"这一局 A 方是不是执红", 只用来把红/黑两本账换算成 A/B 两本账
     * (换算出来的值同时用于 rewardA/rewardB 出参和每手的 matchRewardProgress)。
     * rewardRed / rewardBlack / rewardA / rewardB 都是**出参**: 本局各方累计到的
     * **奖励曲线上那条口径**的累计值 (见下面的 RewardAccounting) + 终局值。
     * A/B 的换算是**唯一**在这里做的, 调用方不要再自己算一遍。
     * 被中止时返回 ONGOING, 四个出参停在中止那一刻的值。
     */
    int playMatchGame(AgentType redType, AgentType blackType, bool aIsRed, MatchStats &st,
                      double &rewardRed, double &rewardBlack, double &rewardA,
                      double &rewardB);

public:
    /*
     * ================================================================
     *  ---- ④ 一局的奖励账 (两种口径都记, 2026-09) ----
     * ================================================================
     * 为什么两种都记 (而不是只留"曲线用的那一种"):
     *   1. 端到端断言需要一个**可比对**的参照。学习口径与引擎口径之间有一条精确关系:
     *        学习即时 = 0.1 x 引擎即时 − 0.001 x 该方手数      (与棋局无关, 逐手可验)
     *      只留一种口径的话, 这条换算规则就只能"看着代码相信", 而本工程的教训是
     *      "看着等价"不算数 (见 docs/session_2026_09_sac.md §4)。
     *   2. 换口径会**破坏界面历史读数的可比性** (旧截图/旧 CSV 是引擎口径)。把两本账
     *      都留在结构里, 事后至少要能解释差多少。
     *
     * 字段口径:
     *   engineImmediate* : 引擎口径的即时奖励累计 (材质 x1, 已换算成**走子方视角**)
     *   engineTerminal*  : 引擎口径的终局值 (±1 / 和棋: 0)
     *   learnImmediate*  : 学习口径的即时奖励累计 (材质 x0.1 + 每步代价)
     *   learnTerminal*   : 学习口径的终局值 (SAC 塑形开着时是 ±(1+败方材质/3.5))
     *   moves*           : 该方在本局走子的手数 (每步代价那一项的乘数)
     *   learnCaliper*    : true = 这个 agent **有**学习口径 (曲线就用它); false = 纯搜索
     *                      agent, 曲线用引擎口径 (界面上以标签标注)
     *   curve*()         : 曲线上实际取的那个值 (先取即时累计再取终局, 与信号的分工一致)
     */
    struct RewardAccounting {
        double engineImmediateA = 0.0, engineImmediateB = 0.0;
        double engineTerminalA = 0.0, engineTerminalB = 0.0;
        double learnImmediateA = 0.0, learnImmediateB = 0.0;
        double learnTerminalA = 0.0, learnTerminalB = 0.0;
        int movesA = 0, movesB = 0;
        bool learnCaliperA = false, learnCaliperB = false;
        /* 曲线上的"本局累计"(不含终局) —— 每手 emit 的那个值 */
        double curveImmediateA() const
        {
            return learnCaliperA ? learnImmediateA : engineImmediateA;
        }
        double curveImmediateB() const
        {
            return learnCaliperB ? learnImmediateB : engineImmediateB;
        }
        /* 曲线上的"本局最终值" —— 局末 emit 的那个值 */
        double curveFinalA() const
        {
            return learnCaliperA ? (learnImmediateA + learnTerminalA)
                                 : (engineImmediateA + engineTerminalA);
        }
        double curveFinalB() const
        {
            return learnCaliperB ? (learnImmediateB + learnTerminalB)
                                 : (engineImmediateB + engineTerminalB);
        }
        void clear() { *this = RewardAccounting(); }
    };
    /*
     * 最近**打完**的那一局的奖励账。
     * 线程契约: 由对弈线程在 playMatchGame 里写; 读它要等这一局结束 (测试是同步调
     * matchAgents 之后读)。界面不用它 (界面走信号)。
     */
    const RewardAccounting &lastRewardAccounting() const { return m_lastReward; }

    /* 该 agent 类型有没有"学习口径"的奖励 (纯搜索 agent: 没有) */
    static bool agentHasLearningReward(AgentType type);

    /*
     * ---- Alpha-Beta 的等级口径 (单一来源) ----
     *
     * 返回这个 agent 类型的 AB 搜索深度; 返回 0 = **这个类型不是 Alpha-Beta**。
     *
     * 为什么要有它: 深度原来写死在各决策分支里 (HEAD 上是 5 处
     * `ABAgent abAI(env, AB_DEPTH)` + 5 处 `emitStage(... AB_DEPTH)`), 一加等级就
     * 必然出现"某一支还印着深度 4、实际按别的深度在下"这种假状态文字 (本工程已经栽过
     * 好几次同类问题: 注释写深度 8 / 标签写深度 4 / 实际 5)。现在"谁是多少级"只有
     * 这一个地方知道。
     *
     * ⚠ 两个 `default:` 兜底分支 (`aiThinkRaw` / `aiThinkForAgentRaw`) **仍然**用
     *   AB_DEPTH —— 那是"这个类型没有自己的决策 case"的兜底, 不是任何一档等级的深度,
     *   所以它**故意**不查这张表 (并且会打一条 qWarning, 免得降级是静默的)。
     *   也就是说"仍然写死 AB_DEPTH"的地方只剩这两处兜底, 它们是刻意的。
     *
     * `AGENT_ALPHABETA` 保持**原有深度 AB_DEPTH(=4) 不变** —— 它是对照组的既有口径,
     * 老读数/老脚本可比性不能被这次新增改动破坏。
     */
    static int abDepthOf(AgentType type);
    /*
     * 奖励曲线的口径标签 (界面用, 挂在曲线名后面):
     *   有学习口径 -> "学习口径(材质x0.1+每步代价+终局±1)"
     *   没有       -> "引擎口径(材质x1+终局±1)"
     * 为什么要标出来: 两个口径差 10 倍, 同一张图上 A 用学习口径、B 用引擎口径时
     * **两条线不可直接比大小** —— 标签是这件事唯一的提示 (见 §1.1 的坑)。
     */
    static QString agentRewardCaliperLabel(AgentType type);

private:
    /* 取该类型的 agent 实例 (没建过 = nullptr; 纯搜索 agent 永远 nullptr) */
    AgentBase *agentInstance(AgentType type) const;
    /* 学习口径的即时奖励; 没有口径 (或实例还没建) 时返回 NaN, 调用方回退引擎口径 */
    double learningStepRewardOrNaN(AgentType type, const Step &step, int mover);
    /* 学习口径的终局值; 同上 */
    double learningTerminalRewardOrNaN(AgentType type, int result, int mover);
    RewardAccounting m_lastReward;
    /* 是否轮到这个 agent 走 (对弈中 arena 用) */
    AgentType typeForTurn(int turn, AgentType redType, AgentType blackType) const;

    /* 启动加载状态 */
    std::atomic<bool> m_startupComplete{false};
    static std::map<AgentType, std::string> s_weightPaths;  /* 已发现的权重文件路径 */

    /* 后台训练 */
    std::thread m_bgTrainThread;
    std::atomic<bool> m_bgTraining{false};
    /*
     * 一轮的规模 (见 setBackgroundTrainRound)。用 atomic: GUI/测试线程写、训练线程
     * 每轮开头读, 两边不需要更强的同步 (读到的是"上一轮或这一轮"的规模, 都合法)。
     * 初值 = chessboard.cpp 里的 BG_TRAIN_EPISODES / BG_TRAIN_MAX_MOVES。
     */
    std::atomic<int> m_bgTrainEpisodes{1};
    std::atomic<int> m_bgTrainMaxMoves{60};
    /*
     * mutable: 自检 (const 方法) 要读常驻 agent 的内部状态, 而那与"决策中的搜索"和
     * "后台训练的载入/保存"是同一批数据 —— 必须进同一把锁
     * (见 ChessBoard::getAgentSelfCheck 的说明)。
     */
    mutable std::mutex m_agentMutex;       /* 保护主agent权重读写 */
    void backgroundTrainLoop();            /* 训练线程主循环 */
};

#endif // CHESSBOARD_H
