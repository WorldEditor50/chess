#include "chessboard.h"
#include <QDebug>
#include <QDir>
#include <chrono>

/*
 * ChessBoard - 核心游戏面板
 * 负责: 棋盘绘制、走法执行、AI调度、数据库记录、回放
 */

/* Static member initialization */
PGEagent *ChessBoard::m_sfPG = nullptr;
DQNAgent *ChessBoard::m_sfDQN = nullptr;
PPOMCTSAgent *ChessBoard::m_sfPPOMCTS = nullptr;
DQNMCTSAgent *ChessBoard::m_sfDQNMCTS = nullptr;
std::map<ChessBoard::AgentType, std::string> ChessBoard::s_weightPaths;

/*
 * startupLoad - 异步加载数据库和AI模型权重
 *
 * 在后台线程中运行, 完成以下工作:
 *   1. 打开 SQLite 棋局数据库
 *   2. 扫描 weights/ 目录下的预训练权重文件
 *   3. 预创建 agent 实例并加载权重
 *
 * 完成后通过 QMetaObject::invokeMethod 在主线程发送
 * startupComplete 信号, 通知 MainWindow 启用界面.
 */
void ChessBoard::startupLoad()
{
    /* ---- 1. 打开数据库 ---- */
    GameDatabase::instance().open("chess_games.db");

    /* ---- 2. 扫描权重文件 ---- */
    struct WeightEntry {
        AgentType type;
        std::string path;
    };

    std::vector<WeightEntry> weightFiles = {
        {AGENT_PG,        "weights/pg_agent.dat"},
        {AGENT_DQN,       "weights/dqn_agent.dat"},
        {AGENT_PPOMCTS,   "weights/ppomcts_agent.dat"},
        {AGENT_DQNMCTS,   "weights/dqnmcts_agent.dat"}
    };

    for (const auto &we : weightFiles) {
        std::ifstream f(we.path);
        if (f.good()) {
            f.close();
            s_weightPaths[we.type] = we.path;
        }
    }

    /* ---- 3. 预创建 self-play agent 并加载权重 ---- */
    if (s_weightPaths.count(AGENT_PG)) {
        if (m_sfPG == nullptr)
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
        m_sfPG->loadPolicy(s_weightPaths[AGENT_PG]);
    }
    if (s_weightPaths.count(AGENT_DQN)) {
        if (m_sfDQN == nullptr)
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
        m_sfDQN->loadModel(s_weightPaths[AGENT_DQN]);
    }
    if (s_weightPaths.count(AGENT_PPOMCTS)) {
        if (m_sfPPOMCTS == nullptr)
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
        m_sfPPOMCTS->loadModel(s_weightPaths[AGENT_PPOMCTS]);
    }
    if (s_weightPaths.count(AGENT_DQNMCTS)) {
        if (m_sfDQNMCTS == nullptr)
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        m_sfDQNMCTS->loadModel(s_weightPaths[AGENT_DQNMCTS]);
    }

    /* ---- 4. 启动后台训练 & 通知主线程加载完成 ---- */
    m_startupComplete = true;

    /* 启动后台训练线程 (神经网络agent持续自我对弈提升棋力) */
    startBackgroundTraining();

    QMetaObject::invokeMethod(this, [this]() {
        emit startupComplete();
    }, Qt::QueuedConnection);
}

ChessBoard::ChessBoard(QWidget *parent) :
    QWidget(parent),
    selectID(-1),
    color(Stone::COLOR_RED),
    state(STATE_IDEL),
    m_currentGameId(-1),
    m_moveCount(0),
    m_dbEnabled(false),
    m_replayGameId(-1),
    m_replayIndex(0),
    m_agentType(AGENT_ALPHABETA)
{
    connect(this, &ChessBoard::sendResult,
            this, &ChessBoard::checkGameOver, Qt::QueuedConnection);
    processThread = std::thread(&ChessBoard::process, this);
}

ChessBoard::~ChessBoard()
{
    /* 先停止后台训练 */
    stopBackgroundTraining();

    {
        QMutexLocker locker(&mutex);
        state = STATE_TERMINATE;
        condit.wakeAll();
    }
    if (processThread.joinable()) {
        processThread.join();
    }
}

/* 获取石头中心的点坐标 */
QPoint ChessBoard::getStoneCenter(int x, int y)
{
    return QPoint(offsetX + y * gridSize, offsetY + x * gridSize);
}

/* 获取石头位置 (棋盘坐标) */
Pos ChessBoard::getStonePos(const QPoint &point)
{
    int x = (point.y() - offsetY + gridSize / 2) / gridSize;
    int y = (point.x() - offsetX + gridSize / 2) / gridSize;
    if (x < 0 || x > 9 || y < 0 || y > 8) {
        return Pos(-1, -1);
    }
    return Pos(x, y);
}

QRect ChessBoard::getRect(QPoint &center)
{
    QPoint topLeft(center.x() - stoneRadius, center.y() - stoneRadius);
    QSize size(stoneRadius * 2, stoneRadius * 2);
    QRect rect(topLeft, size);
    return rect;
}

Stone *ChessBoard::selectStone(const QPoint &point)
{
    Pos pos = getStonePos(point);
    if (pos.x < 0 || pos.y < 0) {
        return nullptr;
    }
    return chess.m_map[pos];
}

bool ChessBoard::moveStone(const QPoint &point)
{
    bool ret = false;
    Pos pos = getStonePos(point);
    if (pos.x < 0 || pos.y < 0) {
        return false;
    }
    Stone *stone = chess.m_map[pos];
    if (stone != nullptr) {
        if (stone->color != color) {
            /* 点击对方棋子: 如果当前有选中, 尝试吃子 */
            if (selectID != -1) {
                Stone *selected = chess.m_map.get(selectID);
                if (selected != nullptr) {
                    if (selected->tryMoveTo(pos)) {
                        /* 执行吃子走法 */
                        Step *step = Steps::instance().get();
                        step->id = selectID;
                        step->pos = selected->pos;
                        step->nextId = stone->id;
                        step->nextPos = pos;
                        step->reward = 0;
                        double totalReward = 0;
                        chess.moveForward(step, totalReward);
                        Steps::instance().put(std::vector<Step*>(1, step));
                        selectID = -1;
                        color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                        ret = true;
                    }
                }
            }
        } else {
            /* 点击己方棋子: 选中它 */
            selectID = stone->id;
            ret = false;
        }
    } else {
        /* 点击空位: 如果当前有选中, 尝试移动 */
        if (selectID != -1) {
            Stone *selected = chess.m_map.get(selectID);
            if (selected != nullptr) {
                if (selected->tryMoveTo(pos)) {
                    Step *step = Steps::instance().get();
                    step->id = selectID;
                    step->pos = selected->pos;
                    step->nextId = Stone::ID_NONE;
                    step->nextPos = pos;
                    step->reward = 0;
                    double totalReward = 0;
                    chess.moveForward(step, totalReward);
                    Steps::instance().put(std::vector<Step*>(1, step));
                    selectID = -1;
                    color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
                    ret = true;
                }
            }
        }
    }
    return ret;
}

void ChessBoard::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    /* 绘制棋盘背景 */
    painter.setBrush(QColor(222, 184, 135));
    painter.drawRect(rect());

    /* 绘制网格线 */
    QPen pen(Qt::black, 2);
    painter.setPen(pen);
    for (int i = 0; i < 10; i++) {
        painter.drawLine(offsetX, offsetY + i * gridSize, offsetX + 8 * gridSize, offsetY + i * gridSize);
    }
    for (int i = 0; i < 9; i++) {
        painter.drawLine(offsetX + i * gridSize, offsetY, offsetX + i * gridSize, offsetY + 4 * gridSize);
        painter.drawLine(offsetX + i * gridSize, offsetY + 5 * gridSize, offsetX + i * gridSize, offsetY + 9 * gridSize);
    }

    /* 绘制棋子 */
    for (int i = 0; i < 32; i++) {
        Stone *stone = chess.m_children[i];
        if (stone == nullptr || stone->alive == false) {
            continue;
        }
        drawStone(painter, stone);
    }

    /* 绘制选中高亮 (灰色边框) */
    if (selectID != -1) {
        Stone *selected = chess.m_map.get(selectID);
        if (selected != nullptr) {
            QPoint center = getStoneCenter(selected->pos.x, selected->pos.y);
            QRect highlightRect = getRect(center).adjusted(-3, -3, 3, 3);

            /* 1. 外发光光晕 (半透明灰色圆环) */
            QRadialGradient glow(center, stoneRadius + 8);
            glow.setColorAt(0.0, QColor(160, 160, 160, 120));
            glow.setColorAt(0.7, QColor(160, 160, 160, 60));
            glow.setColorAt(1.0, QColor(160, 160, 160, 0));
            painter.setBrush(glow);
            painter.setPen(Qt::NoPen);
            painter.drawEllipse(center, stoneRadius + 8, stoneRadius + 8);

            /* 2. 高亮边框 (灰色粗线) */
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(QColor(128, 128, 128), 4));
            painter.drawEllipse(highlightRect);

            /* 3. 内发光 (半透明灰色填充) */
            painter.setBrush(QColor(160, 160, 160, 30));
            painter.setPen(QPen(QColor(128, 128, 128), 2));
            painter.drawEllipse(getRect(center));
        }
    }
}

void ChessBoard::drawStone(QPainter &painter, const Stone *stone)
{
    QPoint center = getStoneCenter(stone->pos.x, stone->pos.y);
    QRect rect = getRect(center);

    /* 绘制棋子背景 */
    painter.setBrush(QColor(255, 206, 158));
    painter.setPen(QPen(Qt::black, 2));
    painter.drawEllipse(rect);

    /* 绘制棋子文字 */
    painter.setFont(QFont("SimSun", 18, QFont::Bold));
    if (stone->color == Stone::COLOR_RED) {
        painter.setPen(Qt::red);
    } else {
        painter.setPen(Qt::black);
    }
    painter.drawText(rect, Qt::AlignCenter, QString::fromStdString(stone->name));
}
void ChessBoard::mousePressEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton) return;
    if (state != STATE_IDEL) return;
    if (isReplayMode()) return;

    if (selectID == Stone::ID_NONE) {
        /* 选择己方棋子 */
        Stone *stone = selectStone(event->pos());
        if (stone == nullptr) return;
        if (stone->color != color) return; /* 不能选对方棋子 */
        selectID = stone->id;
        update();
        return;
    }

    /* 已有选中棋子 -> 尝试移动 */
    bool moved = moveStone(event->pos());
    if (!moved) {
        /* 移动失败: 可能是点到了己方另一棋子,重新选 */
        Stone *stone = selectStone(event->pos());
        if (stone != nullptr && stone->color == color) {
            selectID = stone->id;
        } else {
            selectID = Stone::ID_NONE;
        }
        update();
        return;
    }

    /* 玩家走棋成功 → 检查是否将杀 */
    if (chess.isGameOver() != Stone::COLOR_NONE) {
        state = STATE_TERMINATE;
        update();
        emit sendResult(chess.isGameOver());
        return;
    }

    /* 切换为AI思考 */
    selectID = Stone::ID_NONE;
    state = STATE_THINKING;
    update();
    condit.wakeAll();
}

void ChessBoard::process()
{
    while (state != STATE_TERMINATE) {
        {
            QMutexLocker locker(&mutex);
            if (state != STATE_THINKING) {
                condit.wait(&mutex);
            }
            if (state == STATE_TERMINATE) {
                break;
            }
        }

        /* 计时开始 */
        auto t0 = std::chrono::steady_clock::now();

        /* AI(黑方) 决策 */
        Step step = aiThink(Stone::COLOR_BLACK);

        /* 计时结束 */
        auto t1 = std::chrono::steady_clock::now();
        long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        /* 检查是否有合法走法 */
        if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
            /* AI无合法走法 */
            emit sendResult(Stone::COLOR_RED); /* 红方胜 */
            state = STATE_TERMINATE;
            update();
            continue;
        }

        /* 通过引擎执行走法 */
        double totalReward = 0;
        chess.moveForward(&step, totalReward);

        /* 检查将杀 */
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            emit sendResult(gameResult);
            state = STATE_TERMINATE;
            update();
            continue;
        }

        /* 通知UI更新 (在主线程更新QLabel等) */
        emit aiThinkFinished(elapsedMs);

        /* 回到等待玩家状态 */
        {
            QMutexLocker locker(&mutex);
            color = Stone::COLOR_RED;
            state = STATE_IDEL;
            selectID = Stone::ID_NONE;
            condit.wakeAll();
        }
        QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
    }
}

void ChessBoard::reset()
{
    chess.reset();
    selectID = -1;
    color = Stone::COLOR_RED;
    state = STATE_IDEL;
    update();
}

void ChessBoard::checkGameOver(int result)
{
    QString msg;
    if (result == Stone::COLOR_RED) {
        msg = "红方胜!";
    } else if (result == Stone::COLOR_BLACK) {
        msg = "黑方胜!";
    } else {
        msg = "平局!";
    }
    QMessageBox::information(this, "游戏结束", msg);
}

void ChessBoard::setAgentType(AgentType type)
{
    m_agentType = type;
}

/* ================================================================
 *  aiThink - 根据当前选中的agent类型选择走法
 *
 *  AI Agent 类型:
 *    AGENT_ALPHABETA : 内置Alpha-Beta剪枝 (深度=8)
 *    AGENT_MCTS      : 蒙特卡洛树搜索 (800次模拟)
 *    AGENT_PG        : Policy Gradient (PGEagent)
 *    AGENT_DQN       : Deep Q-Network (DQNAgent)
 *    AGENT_PPOMCTS   : PPO+MCTS AlphaZero风格 (800次模拟)
 *    AGENT_DQNMCTS   : DQN+MCTS
 * ================================================================ */
Step ChessBoard::aiThink(int color)
{
    /* Copy current game state to env so agents can mutate env freely
     * during search (moveForward/moveBack) without touching the main board. */
    env = chess;

    switch (m_agentType) {
    case AGENT_ALPHABETA: {
        /* Alpha-Beta Pruning: 深度=8 */
        static ABAgent abAI(env, 5);
        return abAI.getBestMove(color);
    }
    case AGENT_MCTS: {
        /* 蒙特卡洛树搜索: 800次模拟 */
        static MCTS mctsAI(env, 1.414);
        return mctsAI.findBestMove(color, 800);
    }
    case AGENT_PG: {
        /* Policy Gradient: 按概率分布采样 */
        if (m_sfPG == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
            auto it = s_weightPaths.find(AGENT_PG);
            if (it != s_weightPaths.end())
                m_sfPG->loadPolicy(it->second);
        }
        return m_sfPG->selectMove(color, false); /* false = greedy/deterministic */
    }
    case AGENT_DQN: {
        /* Deep Q-Network: argmax Q-value */
        if (m_sfDQN == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
            auto it = s_weightPaths.find(AGENT_DQN);
            if (it != s_weightPaths.end())
                m_sfDQN->loadModel(it->second);
        }
        return m_sfDQN->selectMove(color, false); /* false = no exploration */
    }
    case AGENT_PPOMCTS: {
        /* PPO + MCTS (AlphaZero风格): 800次模拟, argmax */
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            auto it = s_weightPaths.find(AGENT_PPOMCTS);
            if (it != s_weightPaths.end())
                m_sfPPOMCTS->loadModel(it->second);
        }
        return m_sfPPOMCTS->selectMove(color, 80, 0.0f);
    }
    case AGENT_DQNMCTS: {
        /* DQN + MCTS: 400次迭代, 无探索 */
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
            auto it = s_weightPaths.find(AGENT_DQNMCTS);
            if (it != s_weightPaths.end())
                m_sfDQNMCTS->loadModel(it->second);
        }
        return m_sfDQNMCTS->selectMove(color, 6, true);
    }
    default:
        static ABAgent abAIDefault(env, 5);
        return abAIDefault.getBestMove(color);
    }
}

/* ================================================================
 *  aiThinkForAgent - 使用指定agent类型在env副本上决策
 *
 *  用于 self play. 先将 chess 复制到 env, 然后在 env 上搜索.
 *  所有 agent 实例引用 env 而非 chess, 确保搜索过程中的
 *  moveForward/moveBack 不影响主棋盘状态.
 *
 *  注意: 在调用之前必须确保 chess 处于正确的当前局面
 * ================================================================ */
Step ChessBoard::aiThinkForAgent(int color, AgentType agentType)
{
    /* Copy current game state to env so agents can mutate env freely
     * during search without touching the main board. */
    env = chess;

    switch (agentType) {
    case AGENT_ALPHABETA: {
        static ABAgent abAIForAgent(env, 5);
        return abAIForAgent.getBestMove(color);
    }
    case AGENT_MCTS: {
        static MCTS mctsAI(env, 1.414);
        return mctsAI.findBestMove(color, 800);
    }
    case AGENT_PG: {
        if (m_sfPG == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
        }
        return m_sfPG->selectMove(color, false);
    }
    case AGENT_DQN: {
        if (m_sfDQN == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
        }
        return m_sfDQN->selectMove(color, false);
    }
    case AGENT_PPOMCTS: {
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
        }
        return m_sfPPOMCTS->selectMove(color, 8, 0.0f);
    }
    case AGENT_DQNMCTS: {
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        }
        return m_sfDQNMCTS->selectMove(color, 4, false);
    }
    default:
        static ABAgent abAIForAgentDef(env, 5);
        return abAIForAgentDef.getBestMove(color);
    }
}

/* ================================================================
 *  selfPlay - AI Self Play（带强化学习训练）
 *
 *  同一 AI agent 同时控制红黑双方自对弈, 直到游戏结束.
 *
 *  对于支持在线训练的 agent (目前 DQNMCTS), 每步走完后自动将经验
 *  存入 replay buffer, 并在对局结束时触发一次 learn 步骤.
 *
 *  参数:
 *    agentType - 使用的 AI agent 类型
 *
 *  返回:
 *    胜方: Stone::COLOR_RED 或 Stone::COLOR_BLACK
 *    平局: Stone::COLOR_NONE
 * ================================================================ */
int ChessBoard::selfPlay(AgentType agentType)
{
    chess.reset();
    int turn = Stone::COLOR_RED;
    int maxMoves = 300;
    int moves = 0;

    while (moves < maxMoves) {
        int gameResult = chess.isGameOver();
        if (gameResult != Stone::COLOR_NONE) {
            return gameResult;
        }

        Step step = aiThinkForAgent(turn, agentType);

        /* 无合法走法 */
        if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
            return (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        }

        double totalReward = 0;
        chess.moveForward(&step, totalReward);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        moves++;
    }

    /* 达到最大步数, 按评估判胜 */
    double score = chess.evaluate();
    return (score > 0) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}

/* ================================================================
 *  保存AI模型权重
 * ================================================================ */
bool ChessBoard::saveCurrentAgentModel(AgentType agentType, const std::string &filepath)
{
    switch (agentType) {
    case AGENT_PG: {
        if (m_sfPG == nullptr) return false;
        return m_sfPG->savePolicy(filepath);
    }
    case AGENT_DQN: {
        if (m_sfDQN == nullptr) return false;
        return m_sfDQN->saveModel(filepath);
    }
    case AGENT_PPOMCTS: {
        if (m_sfPPOMCTS == nullptr) return false;
        return m_sfPPOMCTS->saveModel(filepath);
    }
    case AGENT_DQNMCTS: {
        if (m_sfDQNMCTS == nullptr) return false;
        return m_sfDQNMCTS->saveModel(filepath);
    }
    default:
        return false;
    }
}

/* ================================================================
 *  后台训练线程
 *
 *  克隆当前agent, 在独立棋盘上持续自我对弈训练.
 *  每4轮训练后将权重同步回主agent.
 *  每次训练前从主棋盘复制当前棋局状态.
 * ================================================================ */

void ChessBoard::startBackgroundTraining()
{
    if (m_bgTraining) return;
    m_bgTraining = true;
    m_bgTrainThread = std::thread(&ChessBoard::backgroundTrainLoop, this);
}

void ChessBoard::stopBackgroundTraining()
{
    m_bgTraining = false;
    if (m_bgTrainThread.joinable()) {
        m_bgTrainThread.join();
    }
}

void ChessBoard::backgroundTrainLoop()
{
    static const char *TMP_WEIGHTS = "weights/_temp_train.dat";
    QDir().mkpath("weights");

    while (m_bgTraining) {
        AgentType type = m_agentType;

        /* Alpha-Beta 和 MCTS 没有可训练的权重, 休眠后重试 */
        if (type == AGENT_ALPHABETA || type == AGENT_MCTS) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        /* 确保主agent已创建 (可能还未首次使用) */
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            switch (type) {
            case AGENT_PG:
                if (m_sfPG == nullptr)
                    m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
                break;
            case AGENT_DQN:
                if (m_sfDQN == nullptr)
                    m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS == nullptr)
                    m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS == nullptr)
                    m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
                break;
            default: break;
            }
        }

        /* ---- 克隆主agent权重到临时文件 ---- */
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            switch (type) {
            case AGENT_PG:
                if (m_sfPG) m_sfPG->savePolicy(TMP_WEIGHTS);
                break;
            case AGENT_DQN:
                if (m_sfDQN) m_sfDQN->saveModel(TMP_WEIGHTS);
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS) m_sfPPOMCTS->saveModel(TMP_WEIGHTS);
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS) m_sfDQNMCTS->saveModel(TMP_WEIGHTS);
                break;
            default: break;
            }
        }

        /* ---- 在独立棋盘上创建克隆agent并训练4轮 ---- */
        {
            Chess trainChess;  /* 独立训练棋盘: 从初始局面开始 */

            switch (type) {
            case AGENT_PG: {
                PGEagent clone(trainChess, 64, 0.9f, 0.01f, 1.0f);
                clone.loadPolicy(TMP_WEIGHTS);
                clone.train(4, 200, true, false); /* 4 episodes, self-play, quiet */
                clone.savePolicy(TMP_WEIGHTS);
                break;
            }
            case AGENT_DQN: {
                DQNAgent clone(trainChess, 64, 0.99f, 0.001f, 1.0f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(4, 200, false); /* 4 episodes, quiet */
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            case AGENT_PPOMCTS: {
                PPOMCTSAgent clone(trainChess, 64, 0.99f, 0.001f, 1.414f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(4, 50, 200, false); /* 4 episodes, 50 sims, quiet */
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            case AGENT_DQNMCTS: {
                DQNMCTSAgent clone(trainChess, 128, 0.99f, 0.001f, 1.0f, 1.414f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(4, 50, 200, false); /* 4 episodes, 50 iters, quiet */
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            default: break;
            }
        }

        /* ---- 4轮完成后, 将训练好的权重同步回主agent ---- */
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            switch (type) {
            case AGENT_PG:
                if (m_sfPG) m_sfPG->loadPolicy(TMP_WEIGHTS);
                break;
            case AGENT_DQN:
                if (m_sfDQN) m_sfDQN->loadModel(TMP_WEIGHTS);
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS) m_sfPPOMCTS->loadModel(TMP_WEIGHTS);
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS) m_sfDQNMCTS->loadModel(TMP_WEIGHTS);
                break;
            default: break;
            }
        }
    }
}

/* ================================================================
 *  shutdownSave - 程序退出时保存所有已初始化agent的权重
 *
 *  将权重保存到 weights/ 目录下的标准路径.
 *  只保存已创建(非nullptr)的agent.
 * ================================================================ */
void ChessBoard::shutdownSave()
{
    /* 确保 weights 目录存在 */
    QDir().mkpath("weights");

    /* 保存 aiThink / aiThinkForAgent 共享的 neural agent 权重 */
    if (m_sfPG != nullptr)
        m_sfPG->savePolicy("weights/pg_agent.dat");
    if (m_sfDQN != nullptr)
        m_sfDQN->saveModel("weights/dqn_agent.dat");
    if (m_sfPPOMCTS != nullptr)
        m_sfPPOMCTS->saveModel("weights/ppomcts_agent.dat");
    if (m_sfDQNMCTS != nullptr)
        m_sfDQNMCTS->saveModel("weights/dqnmcts_agent.dat");
}

/* ================================================================
 *  回放功能
 * ================================================================ */
void ChessBoard::loadReplayGame(int gameId, const QVector<DBStep> &steps)
{
    m_replayGameId = gameId;
    m_replaySteps = steps;
    m_replayIndex = 0;
    chess.reset();
    /* 应用到所有步 */
    for (int i = 0; i < steps.size(); i++) {
        const DBStep &dbStep = steps[i];
        /* 通过坐标查找棋子 */
        Stone *stone = chess.m_map[Pos(dbStep.fromX, dbStep.fromY)];
        if (stone == nullptr) {
            continue;
        }
        /* 构造Step */
        Step *step = Steps::instance().get();
        step->id = stone->id;
        step->pos = stone->pos;
        step->nextId = dbStep.toX;
        step->nextPos = Pos(dbStep.toX, dbStep.toY);
        step->reward = 0;
        double totalReward = 0;
        chess.moveForward(step, totalReward);
        Steps::instance().put(std::vector<Step*>(1, step));
    }
    m_replayIndex = steps.size();
    update();
    emit replayIndexChanged(m_replayIndex, m_replaySteps.size());
}

bool ChessBoard::replayPrev()
{
    if (m_replayGameId < 0) return false;
    if (m_replayIndex <= 0) return false;
    m_replayIndex--;
    applyReplayStep(m_replayIndex);
    update();
    emit replayIndexChanged(m_replayIndex, m_replaySteps.size());
    return true;
}

bool ChessBoard::replayNext()
{
    if (m_replayGameId < 0) return false;
    if (m_replayIndex >= m_replaySteps.size()) return false;
    const DBStep &dbStep = m_replaySteps[m_replayIndex];
    Stone *stone = chess.m_map[Pos(dbStep.fromX, dbStep.fromY)];
    if (stone == nullptr) {
        m_replayIndex++;
        return false;
    }
    Step *step = Steps::instance().get();
    step->id = stone->id;
    step->pos = stone->pos;
    step->nextId = dbStep.toX;
    step->nextPos = Pos(dbStep.toX, dbStep.toY);
    step->reward = 0;
    double totalReward = 0;
    chess.moveForward(step, totalReward);
    Steps::instance().put(std::vector<Step*>(1, step));
    m_replayIndex++;
    update();
    emit replayIndexChanged(m_replayIndex, m_replaySteps.size());
    return true;
}

void ChessBoard::applyReplayStep(int targetIndex)
{
    chess.reset();
    for (int i = 0; i < targetIndex; i++) {
        const DBStep &dbStep = m_replaySteps[i];
        Stone *stone = chess.m_map[Pos(dbStep.fromX, dbStep.fromY)];
        if (stone == nullptr) continue;
        Step *step = Steps::instance().get();
        step->id = stone->id;
        step->pos = stone->pos;
        step->nextId = dbStep.toX;
        step->nextPos = Pos(dbStep.toX, dbStep.toY);
        step->reward = 0;
        double totalReward = 0;
        chess.moveForward(step, totalReward);
        Steps::instance().put(std::vector<Step*>(1, step));
    }
}
