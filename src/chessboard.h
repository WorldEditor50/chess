#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QMutex>
#include <QWaitCondition>
#include <QVector>
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
        AGENT_ALPHABETA = 0,    /* Alpha-Beta Pruning (built-in Chess::alphaBetaPruning) */
        AGENT_MCTS,              /* Monte Carlo Tree Search */
        AGENT_PG,                /* Policy Gradient (PGEagent) */
        AGENT_DQN,               /* Deep Q-Network (DQNAgent) */
        AGENT_PPOMCTS,           /* PPO + MCTS AlphaZero-style (PPOMCTSAgent) */
        AGENT_DQNMCTS            /* DQN + MCTS (DQNMCTSAgent) */
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

    /* AI Self Play: 同一agent同时控制红黑双方直到游戏结束 */
    int selfPlay(AgentType agentType);

    /* 保存当前agent的权重文件 */
    bool saveCurrentAgentModel(AgentType agentType, const std::string &filepath);

    /* 程序退出时保存所有已初始化的agent权重 */
    void shutdownSave();

    /* 后台持续训练: 克隆agent在后台自我对弈, 每4轮同步权重回主agent */
    void startBackgroundTraining();
    void stopBackgroundTraining();

signals:
    void sendResult(int color);
    /* AI思考完成, 耗时(毫秒) */
    void aiThinkFinished(long long elapsedMs);
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
