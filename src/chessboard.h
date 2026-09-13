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
        AGENT_EVAB               /* EVAB: 学会评估的 Alpha-Beta (EVABAgent) */
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
        QString summary() const;    /* 一行比分 */
        QString detail() const;     /* 比分 + 每局明细 + 耗时 */
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

    /* 程序退出时保存所有已初始化的agent权重 */
    void shutdownSave();

    /* 后台持续训练: 克隆agent在后台自我对弈, 每4轮同步权重回主agent */
    void startBackgroundTraining();
    void stopBackgroundTraining();

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
    QMutex mutex;
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
    static DQNMCTSAgent *m_sfDQNMCTS;
    static EVABAgent *m_sfEVAB;

    /* "走子前先探索环境 + 预训练"开关 (仿 snakeAI) */
    std::atomic<bool> m_preTrainEnabled{true};
    std::atomic<int> m_preTrainSteps{64};
    std::string m_lastExploreInfo;
    mutable QMutex m_infoMutex;            /* 保护 m_lastExploreInfo (跨线程读写) */

    /*
     * 对当前 agent 执行一次"探索环境 + 预训练", 然后是决策。
     * 返回探索出来的说明文字 (给界面用)。
     */
    std::string preTrainThenDecide(AgentBase *agent, int color);

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
    int m_maxPliesPerGame = DEFAULT_MAX_PLIES;   /* 单局手数上限, 到顶判和 */
    /* 打一局: 红方用 redType, 黑方用 blackType; 返回 Chess::RESULT_* */
    int playMatchGame(AgentType redType, AgentType blackType, MatchStats &st);
    /* 是否轮到这个 agent 走 (对弈中 arena 用) */
    AgentType typeForTurn(int turn, AgentType redType, AgentType blackType) const;

    /* 启动加载状态 */
    std::atomic<bool> m_startupComplete{false};
    static std::map<AgentType, std::string> s_weightPaths;  /* 已发现的权重文件路径 */

    /* 后台训练 */
    std::thread m_bgTrainThread;
    std::atomic<bool> m_bgTraining{false};
    std::mutex m_agentMutex;               /* 保护主agent权重读写 */
    void backgroundTrainLoop();            /* 训练线程主循环 */
};

#endif // CHESSBOARD_H
