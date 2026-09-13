#include "chessboard.h"
#include "rl/cpuinfo.hpp"
#include <QDebug>
#include <QDir>
#include <QFontMetrics>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {

constexpr double kPi = 3.14159265358979323846;

/* agent 的短名字, 显示在"AI 正在思考"提示里 */
QString agentDisplayName(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_ALPHABETA: return QStringLiteral("Alpha-Beta");
    case ChessBoard::AGENT_MCTS:      return QStringLiteral("MCTS");
    case ChessBoard::AGENT_PG:        return QStringLiteral("Policy Gradient");
    case ChessBoard::AGENT_DQN:       return QStringLiteral("DQN");
    case ChessBoard::AGENT_PPOMCTS:   return QStringLiteral("PPO+MCTS");
    case ChessBoard::AGENT_DQNMCTS:   return QStringLiteral("DQN+MCTS");
    case ChessBoard::AGENT_EVAB:      return QStringLiteral("EVAB");
    case ChessBoard::AGENT_SACAZ:     return QStringLiteral("SAC+AZ");
    case ChessBoard::AGENT_SACAZ_MOE: return QStringLiteral("SAC+AZ-MoE");
    }
    return QStringLiteral("agent");
}

/* 哪些 agent 真的实现了 exploreAndTrain (决定状态条显示"① 探索"还是"① 搜索") */
bool agentCanExplore(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PG:
    case ChessBoard::AGENT_DQN:
    case ChessBoard::AGENT_PPOMCTS:
    case ChessBoard::AGENT_DQNMCTS:
    case ChessBoard::AGENT_EVAB:
    case ChessBoard::AGENT_SACAZ:
    case ChessBoard::AGENT_SACAZ_MOE:
        return true;
    default:
        return false;
    }
}

/* 状态条上的短耗时: "3.24s" / "1:05" */
QString shortElapsed(long long ms)
{
    if (ms < 0) {
        ms = 0;
    }
    const long long totalSec = ms / 1000;
    const int centi = static_cast<int>((ms % 1000) / 10);
    if (totalSec >= 60) {
        return QStringLiteral("%1:%2").arg(totalSec / 60).arg(totalSec % 60, 2, 10, QChar('0'));
    }
    return QStringLiteral("%1.%2s").arg(totalSec).arg(centi, 2, 10, QChar('0'));
}

} // namespace

/*
 * ChessBoard - 核心游戏面板
 * 负责: 棋盘绘制、走法执行、AI调度、数据库记录、回放
 */

/*
 * AI 搜索预算。这几个数字以前散落在注释里, 而且和真实参数对不上
 * (注释写"深度=8"、UI 标签写"深度=4"、实际是 5; 注释写 800 次模拟、实际 80;
 *  注释写 400 次迭代、实际 6)。集中在这里, 只留一个来源。
 *
 * AB_DEPTH 取 4 的依据 (本机实测, test_ab 的搜索基准):
 *   深度 3 = 63 ms, 深度 4 = 169 ms, 深度 5 = 3605 ms
 * 走法合法性过滤 (sample() 会剔除自杀/不应将/照面) 与"两个节点类型都做静态
 * 搜索"都会增加节点成本, 深度 5 在 GUI 里已经是 3.6 秒一步。需要更强棋力时把
 * 这里调大即可。
 */
static constexpr int AB_DEPTH = 4;            /* Alpha-Beta 搜索深度 */
static constexpr int MCTS_SIMS = 800;         /* MCTS 模拟次数 */
static constexpr int PPO_SIMS = 80;           /* PPO+MCTS 每次决策的模拟次数 */
static constexpr int DQNMCTS_ITERATIONS = 200;/* DQN+MCTS 每次决策的迭代次数 */
/*
 * SAC+AZ 每次决策的 MCTS 模拟次数。实测 (test_sacaz, 本机 AVX2): 64 次模拟约 3 ms,
 * 也就是 ~0.05 ms/模拟 —— 它的网络是 1260->64->64->128 的小 MLP, 比 EVAB 的
 * alpha-beta 搜索便宜得多, 所以这里给到 256 次 (约 12 ms), 换更准的访问分布。
 * 需要更强就继续加大: 成本是线性的。
 */
/*
 * EVAB 的搜索预算。**不能用 AB_DEPTH**: EVAB 有置换表 + 迭代加深 + 走法排序 +
 * 静态搜索, 同一深度比 ABAgent 快一个数量级 (本机实测初始局面:
 *   EVAB depth 4 = 90 ms,  depth 5 = 188 ms,  depth 6 = 890 ms
 *   AB   depth 4 = 89 ms,  depth 5 = 1873 ms
 * 也就是 EVAB 用 AB 深度 4 的时间可以搜到深度 5。原来这里写的是 AB_DEPTH(4),
 * 等于把"学到的评估 + 更强的搜索"这两个卖点都关掉了 —— 实测那样下的 EVAB 与
 * ABAgent 走法 22/30 相同, 只会和棋。
 *
 * 深度给到 6 而把**时间上限**当真正的约束 (800 ms): 叶子评估的代价取决于 blend (见 evagent.h 里 blendMax
 * 的注释: blend>0 时每步要跑网络前向, 会贵 10 倍以上), 用迭代加深 + 时间上限就
 * 自动"贵了就搜浅一点", 不需要为两个模式各调一个深度。
 */
static constexpr int EVAB_DEPTH = 6;
static constexpr long long EVAB_BUDGET_MS = 800;
static constexpr int SACAZ_SIMS = 256;
static constexpr int SACAZ_HIDDEN = 64;       /* SAC+AZ 的网络隐层宽度 */
/*
 * 稀疏 MoE (TB 专家) 骨干的那个变体: 一次模拟 ~10.9 ms (实测, test_sacaz [10]),
 * 所以模拟次数只能给到 16 (约 175 ms/步)。这是"容量换算力"的直接后果 ——
 * 同样的参数量下, 稀疏路由让它比全算 4 个专家快 3.8 倍 (42.0 -> 10.9 ms/模拟),
 * 但要和 0.07 ms/模拟的 MLP 骨干比算力, 无论如何都差两个数量级。
 */
static constexpr int SACAZ_MOE_SIMS = 16;
/*
 * 负载均衡辅助损失的系数。0.1 是实测选出来的 (bench_moe --cases=B --pretrain=3):
 *   aux=0     -> 8 个专家里有 3 个一次都没被选中 (路由塌了)
 *   aux=0.01  -> 都被用到, 但最大/均值仍是 4.00
 *   aux=0.1   -> 都被用到, 最大/均值 2.91   <- 选这个
 *   aux=0.5/2 -> 更均匀 (2.63/2.32), 但辅助项开始盖过真正的策略梯度
 * 这里的"批"只有 32 个样本, 而主干给门控的梯度比辅助项大两个数量级, 所以系数比
 * Switch Transformer 论文里的 0.01 要大得多才起作用。
 */
static constexpr float SACAZ_MOE_AUX = 0.1f;
/* 单局手数上限已集中到 ChessBoard::DEFAULT_MAX_PLIES (界面上可用 setMaxPliesPerGame 调) */

/*
 * 后台训练的单轮规模。训练线程只在每轮开始时检查停止标志, 所以这几个数字直接
 * 决定"关窗要等多久"。原来是 4 局 x 200 步 x 50 次 MCTS 模拟 —— 每步都要跑网络
 * 前向, 一轮可能几分钟, 关窗时 GUI 线程会冻在 join() 上。
 */
static constexpr int BG_TRAIN_EPISODES = 1;   /* 每轮训练局数 */
static constexpr int BG_TRAIN_MAX_MOVES = 60; /* 每局步数上限 */
static constexpr int BG_TRAIN_SIMS = 20;      /* MCTS 类 agent 每步的模拟次数 */

/* Static member initialization */
PGEagent *ChessBoard::m_sfPG = nullptr;
DQNAgent *ChessBoard::m_sfDQN = nullptr;
PPOMCTSAgent *ChessBoard::m_sfPPOMCTS = nullptr;
DQNMCTSAgent *ChessBoard::m_sfDQNMCTS = nullptr;
EVABAgent *ChessBoard::m_sfEVAB = nullptr;
SACAZAgent *ChessBoard::m_sfSACAZ = nullptr;
SACAZAgent *ChessBoard::m_sfSACAZMoe = nullptr;
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
    /*
       报告本构建实际启用的 SIMD 指令集。SIMD 内核是编译期选的 (见 rl/cpuinfo.hpp),
       所以"这个二进制到底会不会用 AVX2"没法靠猜 —— 启动时打一行日志, 排查性能
       问题时就不用先怀疑环境。
    */
    qInfo().noquote() << "[SIMD]" << QString::fromStdString(RL::cpuinfo::describe());

    /*
       启动阶段的"请稍候": 读 7 组权重可能要好几秒 (老格式是十进制文本, DQN+MCTS
       一个文件就 16 MB), 界面上必须有个沙漏在转。这些信号是队列投递到 GUI 线程的,
       所以在这里(工作线程)emit 是安全的。
    */
    emit busyStarted(QStringLiteral("正在载入"),
                     QStringLiteral("读取棋局数据库与模型权重…"));

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
        {AGENT_DQNMCTS,   "weights/dqnmcts_agent.dat"},
        {AGENT_EVAB,      "weights/evab_agent.dat"},
        /*
           SAC+AZ 的一个模型是三个文件 (actor / q1 / q2), 所以这里的路径是**前缀**:
           weights/sacaz_agent -> sacaz_agent_actor / _q1 / _q2。
           判定"有没有已训练的权重"用 actor 那一个即可。
        */
        {AGENT_SACAZ,     "weights/sacaz_agent_actor"},
        /* 稀疏 MoE 骨干的变体: 权重不能共用 —— 层结构完全不同 */
        {AGENT_SACAZ_MOE, "weights/sacaz_moe_agent_actor"}
    };

    for (const auto &we : weightFiles) {
        std::ifstream f(we.path);
        if (f.good()) {
            f.close();
            s_weightPaths[we.type] = we.path;
        }
    }

    /* ---- 3. 预创建 self-play agent 并加载权重 ----
       每一组都记一条耗时日志: 权重可能有几百 MB, 启动慢的时候要能一眼看出是哪一组
       (实测: 稀疏 MoE 那 3 个 146 MB 文件占了大头, 其余六组加起来不到 1 秒)。 */
    auto loadClock = std::chrono::steady_clock::now();
    auto logLoad = [&loadClock](const char *what) {
        const auto now = std::chrono::steady_clock::now();
        const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - loadClock).count();
        qInfo().noquote() << QStringLiteral("[weights] %1: %2 ms")
                                 .arg(QString::fromUtf8(what)).arg(ms);
        loadClock = now;
    };
    if (s_weightPaths.count(AGENT_PG)) {
        emit busyMessage(QStringLiteral("正在载入 Policy Gradient 权重…"));
        if (m_sfPG == nullptr)
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
        m_sfPG->loadPolicy(s_weightPaths[AGENT_PG]);
        logLoad("PG");
    }
    if (s_weightPaths.count(AGENT_DQN)) {
        emit busyMessage(QStringLiteral("正在载入 DQN 权重…"));
        if (m_sfDQN == nullptr)
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
        m_sfDQN->loadModel(s_weightPaths[AGENT_DQN]);
        logLoad("DQN");
    }
    if (s_weightPaths.count(AGENT_PPOMCTS)) {
        emit busyMessage(QStringLiteral("正在载入 PPO+MCTS 权重…"));
        if (m_sfPPOMCTS == nullptr)
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
        m_sfPPOMCTS->loadModel(s_weightPaths[AGENT_PPOMCTS]);
        logLoad("PPO+MCTS");
    }
    if (s_weightPaths.count(AGENT_EVAB)) {
        emit busyMessage(QStringLiteral("正在载入 EVAB 权重…"));
        if (m_sfEVAB == nullptr)
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
        m_sfEVAB->loadModel(s_weightPaths[AGENT_EVAB]);
        logLoad("EVAB");
    }
    if (s_weightPaths.count(AGENT_DQNMCTS)) {
        emit busyMessage(QStringLiteral("正在载入 DQN+MCTS 权重… (文件较大，可能要几秒)"));
        if (m_sfDQNMCTS == nullptr)
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        m_sfDQNMCTS->loadModel(s_weightPaths[AGENT_DQNMCTS]);
        logLoad("DQN+MCTS");
    }
    if (s_weightPaths.count(AGENT_SACAZ)) {
        emit busyMessage(QStringLiteral("正在载入 SAC+AZ 权重…"));
        if (m_sfSACAZ == nullptr)
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
        /* 路径是前缀: 去掉结尾的 "_actor" */
        std::string prefix = s_weightPaths[AGENT_SACAZ];
        const std::string suffix = "_actor";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfSACAZ->loadModel(prefix);
        logLoad("SAC+AZ");
    }
    if (s_weightPaths.count(AGENT_SACAZ_MOE)) {
        /*
           用户要求"程序启动时加载所有模型": 这个变体也在这里预加载 (以前为了启动速度
           改成了懒加载 —— 它一个模型是三个文件、每个 146 MB)。
           代价是启动变慢, 所以: (a) 它在七组权重里放**最后**, 前面几组几秒就完事、
           用户能在沙漏窗里看到进度一条条过; (b) 期间由"请稍候"沙漏窗给反馈;
           (c) 下面每载一组都记一条耗时日志, 慢了能一眼看出是哪一组。
        */
        emit busyMessage(QStringLiteral("正在载入 SAC+AZ (稀疏MoE) 权重… (3 个 146 MB 文件)"));
        if (m_sfSACAZMoe == nullptr)
            m_sfSACAZMoe = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                          SACAZAgent::Backbone::SparseMoeTb,
                                          64, SACAZ_MOE_AUX);
        /* 拆开量: 5 个网络 x 28.7 M 参数的"分配 + 随机初始化"本身就不是小数目 */
        logLoad("SAC+AZ-MoE 建网(5 x 28.7M 参数)");
        std::string prefix = s_weightPaths[AGENT_SACAZ_MOE];
        const std::string suffix = "_actor";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfSACAZMoe->loadModel(prefix);
        logLoad("SAC+AZ-MoE 读权重(3 x 146 MB)");
    }

    /* ---- 4. 启动后台训练 & 通知主线程加载完成 ---- */
    emit busyMessage(QStringLiteral("初始化完成"));
    emit busyFinished();
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
    initThinkVisuals();
    processThread = std::thread(&ChessBoard::process, this);
}

/*
 * initThinkVisuals - 建立"思考中"状态条的动画与信号连线
 *
 * 所有跨线程信号都用 AutoConnection: emit 发生在工作线程/AI 线程, 而本对象
 * (以及主窗口) 住在 GUI 线程, Qt 在 emit 时判定接收者线程不同 -> 自动排队,
 * 于是下面这些槽都在 GUI 线程执行, 可以安全地读写控件状态。
 */
void ChessBoard::initThinkVisuals()
{
    m_animTimer = new QTimer(this);
    m_animTimer->setInterval(40);            /* 25 fps, 只重绘顶部一小条 */
    connect(m_animTimer, &QTimer::timeout, this, [this]() {
        ++m_animPhase;
        if (m_animPhase > 100000) {
            m_animPhase = 0;
        }
        update(thinkingOverlayRect());
    });

    m_busyClickTimer = new QTimer(this);
    m_busyClickTimer->setSingleShot(true);
    m_busyClickTimer->setInterval(2500);
    connect(m_busyClickTimer, &QTimer::timeout, this, [this]() {
        if (!m_busyClickSeen) {
            return;
        }
        m_busyClickSeen = false;
        update(thinkingOverlayRect());
    });

    connect(this, &ChessBoard::aiThinkingStarted, this,
            [this](const QString &, int) {
                m_thinkClock.start();
                m_thinkStage.clear();
                m_busyClickSeen = false;
                if (m_animTimer != nullptr && !m_animTimer->isActive()) {
                    m_animPhase = 0;
                    m_animTimer->start();
                }
                update(thinkingOverlayRect());
            });
    connect(this, &ChessBoard::aiThinkingStage, this, [this](const QString &stage) {
        m_thinkStage = stage;
        update(thinkingOverlayRect());
    });
    connect(this, &ChessBoard::aiThinkingStopped, this, [this]() {
        if (m_animTimer != nullptr) {
            m_animTimer->stop();
        }
        m_busyClickTimer->stop();
        m_busyClickSeen = false;
        m_thinkStage.clear();
        update(thinkingOverlayRect());
    });
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
    /*
       用 floor 语义做舍入。原来写的是 (p - offset + grid/2) / grid, C++ 的整数
       除法向零截断, 于是 point.y() ∈ [0,19] 时 (py-20)/60 得到 0 而不是 -1 ——
       棋盘上/左各 20px 的空白被误判成第 0 行/第 0 列; 相与地, x<0 / y<0 这两个
       守卫永远不会成立。
    */
    const int x = static_cast<int>(std::floor((point.y() - offsetY + gridSize / 2.0) / gridSize));
    const int y = static_cast<int>(std::floor((point.x() - offsetX + gridSize / 2.0) / gridSize));
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
    Pos pos = getStonePos(point);
    if (pos.x < 0 || pos.y < 0) {
        return false;
    }
    Stone *stone = chess.m_map[pos];

    /* 点击己方棋子: 选中它 (不落子) */
    if (stone != nullptr && stone->color == color) {
        selectID = stone->id;
        return false;
    }
    if (selectID == -1) {
        return false;
    }

    Stone *selected = chess.m_map.get(selectID);
    if (selected == nullptr || selected->alive == false) {
        selectID = -1;
        return false;
    }

    /*
       构造候选走法, 由 Chess::isLegalMove 统一校验: 走法形状 + 走后自己是否被将
       (含"不应将"与两将照面)。原来这里只看 selected->tryMoveTo(pos), 于是玩家
       可以自杀、可以在被将时走别的子、也可以主动走出照面。
    */
    Step step;
    step.id = selectID;
    step.pos = selected->pos;
    step.nextId = (stone != nullptr) ? stone->id : Stone::ID_NONE;
    step.nextPos = pos;
    step.reward = 0;
    step.valid = true;

    if (chess.isLegalMove(color, &step) == false) {
        return false;
    }

    double totalReward = 0;
    chess.moveForward(&step, totalReward);
    selectID = -1;
    color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    chess.sideToMove = color;
    return true;
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

    /*
       棋子与选中高亮都在锁内画: AI 工作线程 (process) 和 Agent 对弈线程都在改
       chess, 而绘制发生在 GUI 线程。锁的临界区很短 (只读一遍棋子), 不与搜索重叠
       —— 两个写者都只在"落子"那一瞬间持锁, 思考期间不持锁。
    */
    {
        QMutexLocker locker(&mutex);

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

    /* "AI 正在思考"的状态条画在最上层 */
    drawThinkingOverlay(painter);
}

/*
 * thinkingOverlayRect - 状态条的位置
 *
 * 画在棋盘**上方的空白带**里: 最上面一排棋子的圆心在 offsetY=50, 半径 24, 所以
 * y < 26 这条横带是完全空着的。放在这里既不遮挡任何棋子, 又正好在玩家盯着的
 * 棋盘内部, 不需要去右边的控制栏找。
 */
QRect ChessBoard::thinkingOverlayRect() const
{
    constexpr int kBoxH = 22;
    int boxW = width() - 16;
    if (boxW > 470) {
        boxW = 470;
    }
    if (boxW < 160) {
        boxW = qMax(100, width() - 4);
    }
    return QRect((width() - boxW) / 2, 2, boxW, kBoxH);
}

void ChessBoard::drawThinkingOverlay(QPainter &painter)
{
    const bool selfPlay = m_selfPlaying.load();
    if (state != STATE_THINKING && !selfPlay) {
        return;
    }

    const QRect box = thinkingOverlayRect();
    /* 40ms 一帧 * 40 帧 = 1.6 秒转一圈 */
    const qreal phase = (m_animPhase % 40) / 40.0;
    const qreal wave = 0.5 + 0.5 * std::sin(phase * 2.0 * kPi);

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);

    /* 深色药丸底 (棋盘是米黄, 深底浅字最醒目) */
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(28, 38, 54, 230));
    painter.drawRoundedRect(box, box.height() / 2.0, box.height() / 2.0);

    /* 左端: 一圈旋转的粒子 (呼吸 + 公转 + 彗尾), 与 ThinkingIndicator 同款视觉 */
    const QPointF c(box.left() + 13.0, box.center().y() + 0.5);
    const qreal ringR = 6.4 + 0.9 * wave;
    constexpr int kDots = 8;
    for (int i = 0; i < kDots; ++i) {
        const qreal k = 1.0 - i / static_cast<qreal>(kDots);   /* 彗头 = 1 */
        const qreal ang = phase * 2.0 * kPi - i * (2.0 * kPi / kDots);
        QColor dot(96, 180, 255);
        dot.setAlphaF(0.18 + 0.82 * k * k);
        painter.setBrush(dot);
        const qreal r = 1.0 + 1.5 * k;
        painter.drawEllipse(QPointF(c.x() + ringR * std::cos(ang),
                                    c.y() + ringR * std::sin(ang)), r, r);
    }

    /* 文字: [agent] 正在思考 3.24s · ① 探索环境 + 预训练 (≤64 步) */
    const long long ms = m_thinkClock.isValid() ? m_thinkClock.elapsed() : 0;
    QString text = selfPlay ? QStringLiteral("AI 对弈中") : QStringLiteral("AI 正在思考");
    text += QStringLiteral("  ") + shortElapsed(ms);
    if (!m_thinkStage.isEmpty()) {
        text += QStringLiteral("   ·   ") + m_thinkStage;
    }

    QFont f = painter.font();
    f.setPointSize(8);
    f.setBold(true);
    painter.setFont(f);

    const int textLeft = box.left() + 25;
    const int textRight = box.right() - 8;
    const QFontMetrics fm(f);
    if (m_busyClickSeen) {
        const QString hint = QStringLiteral("请稍候, 现在还不能走子");
        const int hintW = fm.horizontalAdvance(hint) + 10;
        painter.setPen(QColor(255, 196, 92));
        painter.drawText(QRect(textRight - hintW, box.top(), hintW, box.height()),
                         Qt::AlignVCenter | Qt::AlignRight, hint);
        painter.setPen(QColor(236, 242, 250));
        painter.drawText(QRect(textLeft, box.top(), textRight - textLeft - hintW, box.height()),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         fm.elidedText(text, Qt::ElideRight, textRight - textLeft - hintW));
    } else {
        painter.setPen(QColor(236, 242, 250));
        painter.drawText(QRect(textLeft, box.top(), textRight - textLeft, box.height()),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         fm.elidedText(text, Qt::ElideRight, textRight - textLeft));
    }

    painter.restore();
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

    /*
       AI 思考期间点棋盘: 拒绝落子是对的 (轮到 AI 走), 但必须给出反馈。
       原来的实现只是静默 return, 玩家看到的是"棋子没动, 我也点不动", 分不清
       是"还在算"还是"已经卡死" —— 加入"走子前先探索+预训练"之后单步思考涨到
       秒级, 这个问题就很突出了。这里在状态条上闪一句"请稍候"。
    */
    if (state == STATE_THINKING || m_selfPlaying.load()) {
        if (state == STATE_THINKING && !m_busyClickSeen) {
            m_busyClickSeen = true;
            m_busyClickTimer->start();
            update(thinkingOverlayRect());
        }
        return;
    }
    if (state != STATE_IDEL) return;
    if (isReplayMode()) return;

    if (selectID == -1) {
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
        /* 移动失败 (含非法走法): 可能是点到了己方另一棋子, 重新选 */
        Stone *stone = selectStone(event->pos());
        if (stone != nullptr && stone->color == color) {
            selectID = stone->id;
        } else {
            selectID = -1;
        }
        update();
        return;
    }

    /* 玩家走棋成功 → 判定是否将杀 / 困毙 / 和棋 */
    int result = chess.getResult(chess.sideToMove);
    if (result != Chess::RESULT_ONGOING) {
        state = STATE_TERMINATE;
        update();
        emit sendResult(result);
        return;
    }

    /* 切换为AI思考 */
    selectID = -1;
    state = STATE_THINKING;
    update();
    /*
       wakeAll() 必须在持有同一个 mutex 时调用。原来这里是无锁调用, 存在经典的
       丢唤醒窗口: worker 已判定 state != STATE_THINKING (此时还是 IDLE) 但还没
       进入 wait(), GUI 此刻把 state 改成 THINKING 并 wakeAll() —— 由于 GUI 不
       持锁, 这一步可以被插进 worker 的判断与等待之间, worker 随后长睡, 棋盘
       再也不会有 AI 落子。
    */
    {
        QMutexLocker locker(&mutex);
        condit.wakeAll();
    }
}

void ChessBoard::process()
{
    /*
       回到"等待玩家走棋"并宣告思考结束。
       顺序很重要: 必须**先把 state 落成 IDLE, 再**发 aiThinkingStopped ——
       状态条的可见性看的就是 state, 反过来的话 GUI 处理队列消息时可能又多画一帧。
    */
    auto backToIdle = [this]() {
        {
            QMutexLocker locker(&mutex);
            color = Stone::COLOR_RED;
            state = STATE_IDEL;
            selectID = -1;
            condit.wakeAll();
        }
        emit aiThinkingStopped();
        QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
    };

    while (state != STATE_TERMINATE) {
        {
            QMutexLocker locker(&mutex);
            /*
               这里必须是 while 而不是 if, 而且醒来后**必须重新检查 state**。

               原来写的是 `if (state != STATE_THINKING) condit.wait(&mutex);`, 醒来就
               直接往下走去搜索。条件变量本来就会虚假唤醒, 而 reset() 和 matchAgents()
               都会在 state 仍是 IDLE 的时候 wakeAll() (它们只是想叫醒"正在思考"的
               worker, 让它作废在飞的那一步) —— 于是空闲的 worker 被叫醒后照跑
               aiThink(), 一边搜索一边往 env 的 history 里 push_back。
               如果这时另一个线程 (Agent 对弈线程) 也在用同一个 env, 两边同时改
               env.history 就是 double-free / 堆损坏 (ASan 实测抓到的就是这个)。
            */
            while (state != STATE_THINKING && state != STATE_TERMINATE) {
                condit.wait(&mutex);
            }
            if (state == STATE_TERMINATE) {
                break;
            }
        }

        /* 计时开始 */
        auto t0 = std::chrono::steady_clock::now();
        /* 记下代数, 用来识别"思考途中被按了开局" */
        const unsigned gen = m_thinkGeneration.load();

        emit aiThinkingStarted(agentDisplayName(m_agentType), m_preTrainSteps.load());

        /* AI(黑方) 决策 */
        Step step = aiThink(Stone::COLOR_BLACK);

        /* 计时结束 */
        auto t1 = std::chrono::steady_clock::now();
        long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        /* 思考期间玩家按了"开局" -> 这一步连同它的计时一起作废 */
        if (gen != m_thinkGeneration.load()) {
            backToIdle();
            continue;
        }

        /* 检查AI是否有合法走法 (将杀 / 困毙) */
        if (!step.valid) {
            /*
               先分清"真的没棋可走"和"agent 返回了一个无效走法"。后者不是将杀:
               以前直接把 !step.valid 当成"AI 被将死", 于是一个抽风的 agent 会让对局
               在**一步都没落子**的情况下判红方胜 —— 界面上就是"棋子没动, 我却赢了",
               而且从此不能走棋。这里用第一个合法走法兜底。
            */
            std::vector<Step *> legal;
            {
                QMutexLocker locker(&mutex);
                chess.sample(Stone::COLOR_BLACK, legal);
                if (!legal.empty()) {
                    std::fprintf(stderr,
                                 "[aiThink] agent 返回无效走法 (有 %zu 个合法走法), "
                                 "已用第一个合法走法兜底\n", legal.size());
                    step = *legal[0];
                }
            }
            Steps::instance().put(legal);
        }

        if (!step.valid) {
            /* 确实没有合法走法 -> 红方胜 */
            state = STATE_TERMINATE;
            emit aiThinkingStopped();
            emit sendResult(Chess::RESULT_RED_WIN);
            QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
            continue;
        }

        emitStage(QStringLiteral("③ 落子"));

        /* 通过引擎执行走法 (与 GUI 线程的绘制互斥) */
        int result = Chess::RESULT_ONGOING;
        bool discarded = false;
        {
            QMutexLocker locker(&mutex);
            /* 在锁内再确认一次代数: reset() 也是在这个锁内 +1 的, 两者不会交错 */
            if (gen != m_thinkGeneration.load()) {
                discarded = true;
            } else {
                double totalReward = 0;
                chess.moveForward(&step, totalReward);
                chess.sideToMove = Stone::COLOR_RED;
                result = chess.getResult(chess.sideToMove);
            }
        }
        if (discarded) {
            backToIdle();
            continue;
        }

        if (result != Chess::RESULT_ONGOING) {
            emit aiThinkFinished(elapsedMs);
            state = STATE_TERMINATE;
            emit aiThinkingStopped();
            emit sendResult(result);
            QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
            continue;
        }

        /* 通知UI更新 (在主线程更新QLabel等) */
        emit aiThinkFinished(elapsedMs);

        /* 回到等待玩家状态 */
        backToIdle();
    }
}

void ChessBoard::reset()
{
    /*
       reset() 必须同时退出回放模式。原来的实现只重置棋盘, 不清 m_replayGameId,
       而 mousePressEvent 开头是 `if (isReplayMode()) return;` —— 只要在"历史对局"
       下拉框里选过一次, 之后按"开局"也只重置棋盘、点击继续被吞掉, 只能重启程序
       (replayModeExited 信号声明了、也连接了, 但从来没有被 emit 过)。
    */
    bool wasReplay = isReplayMode();
    m_replayGameId = -1;
    m_replayIndex = 0;
    m_replaySteps.clear();

    {
        QMutexLocker locker(&mutex);
        /*
            +1 之后, 正在思考的那一步会在 process() 里发现"代数变了"而主动作废。
           没有这一条时, reset() 只把 state 改回 IDEL 就返回, 而工作线程还在算,
           算完照样落子 —— 落到**重置后的新棋盘**上, 表现为"我刚开了新局, 对方
           却已经走了一步"。
        */
        ++m_thinkGeneration;
        chess.reset();
        selectID = -1;
        color = Stone::COLOR_RED;
        state = STATE_IDEL;
        condit.wakeAll();
    }
    if (m_animTimer != nullptr) {
        m_animTimer->stop();
    }
    if (m_busyClickTimer != nullptr) {
        m_busyClickTimer->stop();
    }
    m_busyClickSeen = false;
    m_thinkStage.clear();
    if (wasReplay) {
        emit replayModeExited();
    }
    update();
}

void ChessBoard::checkGameOver(int result)
{
    QString msg;
    if (result == Chess::RESULT_RED_WIN) {
        msg = "红方胜!";
    } else if (result == Chess::RESULT_BLACK_WIN) {
        msg = "黑方胜!";
    } else {
        /* RESULT_DRAW: 三次重复局面 或 60 回合自然限着 */
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
 *    AGENT_ALPHABETA : Alpha-Beta 剪枝 (默认深度 5, 见 AB_DEPTH)
 *    AGENT_MCTS      : 蒙特卡洛树搜索 (800次模拟)
 *    AGENT_PG        : Policy Gradient (PGEagent)
 *    AGENT_DQN       : Deep Q-Network (DQNAgent)
 *    AGENT_PPOMCTS   : PPO+MCTS AlphaZero风格 (默认 80 次模拟, 见 PPO_SIMS)
 *    AGENT_DQNMCTS   : DQN+MCTS (默认 200 次迭代, 见 DQNMCTS_ITERATIONS)
 *    AGENT_EVAB      : EVAB - 学会评估的 Alpha-Beta (见 docs/agent_evab_design.md)
 * ================================================================ */
/* ================================================================
 *  preTrainThenDecide - 走子前的"先探索环境 + 预训练" (仿 snakeAI)
 *
 *  snakeAI 的每个 Agent::xxxAction() 都是三步: 先把当前状态记下来, 再从当前状态
 *  出发用探索策略滚若干步、用这批新鲜经验在线训练一次, 最后才基于当前状态做决策。
 *  这里把同一套流程套在所有 agent 上 —— 有监督式的 agent (Alpha-Beta / MCTS)
 *  的 exploreAndTrain() 是空实现, EVAB 则把探索结果蒸馏回评估网络, 所以调用它对
 *  它们无害; 而且探索全程用 moveForward/moveBack 试走并原样回退, 不会改动真棋局。
 * ================================================================ */
std::string ChessBoard::preTrainThenDecide(AgentBase *agent, int color)
{    if (agent == nullptr) {
        emitStage(QStringLiteral("① 搜索 / 决策"));
        return std::string();
    }
    if (!m_preTrainEnabled.load()) {
        emitStage(QStringLiteral("① 搜索 / 决策"));
        return std::string("探索+预训练: 已关闭");
    }
    const int steps = m_preTrainSteps.load();
    if (steps <= 0) {
        /* 步数设成 0 等价于关掉探索 (界面上允许这么设, 用来做对照) */
        emitStage(QStringLiteral("① 搜索 / 决策 (探索步数=0)"));
        return std::string("探索+预训练: 已关闭 (步数=0)");
    }
    emitStage(QStringLiteral("① 探索环境 + 预训练 (≤%1 步)").arg(steps));
    const bool trained = agent->exploreAndTrain(color, steps);
    std::string info = agent->getExploreInfo();
    if (info.empty()) {
        info = trained ? "已预训练" : "无需预训练 (该 agent 没有在线可训练参数)";
    }
    /*
       上报这次在线训练的损失 (界面的"训练损失曲线")。不上报损失的 agent 返回 NaN
       (见 AgentBase::getLastTrainLoss), 曲线控件会直接丢弃这个点 —— 所以这里不需要
       区分"是 0"和"没上报", 用 isfinite 判断即可。
    */
    {
        const float loss = agent->getLastTrainLoss();
        if (std::isfinite(loss)) {
            emit trainLossSample((double)loss, QString::fromStdString(agent->getName()),
                                 m_trainSampleNo.fetch_add(1) + 1);
        }
    }
    {
        QMutexLocker locker(&m_infoMutex);
        m_lastExploreInfo = info;
    }
    /*
       用信号把说明送到界面, 而不是让 GUI 线程去读 m_lastExploreInfo ——
       这里在 AI 工作线程, 直接读同一个 std::string 是数据竞争 (A7 那一类)。
    */
    emit aiExploreInfo(QString::fromStdString(info));
    emitStage(QStringLiteral("② 搜索 / 决策"));
    return info;
}

std::string ChessBoard::getLastExploreInfo() const
{
    QMutexLocker locker(&m_infoMutex);
    return m_lastExploreInfo;
}

QString ChessBoard::stagePrefix() const
{
    if (m_matchRunning.load()) {
        const int g = m_matchGameNo.load();
        const int tot = m_matchGames.load();
        const int ply = m_selfPlayMoveNo.load();
        if (g > 0) {
            return QStringLiteral("对弈 %1/%2 局 · 第 %3 手 · ").arg(g).arg(tot).arg(ply);
        }
        return QStringLiteral("对弈 · ");
    }
    if (m_selfPlaying.load()) {
        const int no = m_selfPlayMoveNo.load();
        if (no > 0) {
            return QStringLiteral("自对弈 第 %1 步 · ").arg(no);
        }
        return QStringLiteral("自对弈 · ");
    }
    return QString();
}

void ChessBoard::emitStage(const QString &stage)
{
    emit aiThinkingStage(stagePrefix() + stage);
}

Step ChessBoard::aiThink(int color)
{
    /* Copy current game state to env so agents can mutate env freely
     * during search (moveForward/moveBack) without touching the main board. */
    env = chess;

    switch (m_agentType) {
    case AGENT_ALPHABETA: {
        /*
           Agent 以前声明成函数内的 static, 于是它只在第一次调用时构造, 永远绑定
           在"当时那个 env" 上 —— 一旦出现第二个 ChessBoard (或 env 先被销毁),
           就是悬垂引用。ABAgent 本身只是一个引用 + 一个深度整数, 每步新建的代价
           可以忽略。
        */
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAI(env, AB_DEPTH);
        return abAI.getBestMove(color);
    }
    case AGENT_MCTS: {
        /* 蒙特卡洛树搜索 */
        emitStage(QStringLiteral("① 搜索 / 决策 (MCTS %1 次模拟)").arg(MCTS_SIMS));
        MCTS mctsAI(env, 1.414f);
        return mctsAI.findBestMove(color, MCTS_SIMS);
    }
    case AGENT_PG: {
        /* Policy Gradient: 按概率分布采样 */
        /* 与后台训练线程互斥: 训练线程会在同一把锁内 loadPolicy() 改写网络权重 */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPG == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
            auto it = s_weightPaths.find(AGENT_PG);
            if (it != s_weightPaths.end())
                m_sfPG->loadPolicy(it->second);
        }
        preTrainThenDecide(m_sfPG, color);   /* 先探索环境+预训练, 再决策 */
        return m_sfPG->selectMove(color, false); /* false = greedy/deterministic */
    }
    case AGENT_DQN: {
        /* Deep Q-Network: argmax Q-value */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQN == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
            auto it = s_weightPaths.find(AGENT_DQN);
            if (it != s_weightPaths.end())
                m_sfDQN->loadModel(it->second);
        }
        preTrainThenDecide(m_sfDQN, color);
        return m_sfDQN->selectMove(color, false); /* false = no exploration */
    }
    case AGENT_PPOMCTS: {
        /* PPO + MCTS (AlphaZero风格): argmax */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            auto it = s_weightPaths.find(AGENT_PPOMCTS);
            if (it != s_weightPaths.end())
                m_sfPPOMCTS->loadModel(it->second);
        }
        preTrainThenDecide(m_sfPPOMCTS, color);
        return m_sfPPOMCTS->selectMove(color, PPO_SIMS, 0.0f);
    }
    case AGENT_DQNMCTS: {
        /*
           training 必须是 false。以前传 true, 而 DQNMCTS 的 exploringRate 初值
           是 1.0 且只在 endOnlineEpisode()/learn() 里衰减 (GUI 从不调用),
           于是 selectMove 几乎必然在根节点挑一个**随机**子节点 —— 界面上的
           "DQN+MCTS" 实际上一直在随机走子。迭代数也从 6 提到 200。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
            auto it = s_weightPaths.find(AGENT_DQNMCTS);
            if (it != s_weightPaths.end())
                m_sfDQNMCTS->loadModel(it->second);
        }
        preTrainThenDecide(m_sfDQNMCTS, color);
        return m_sfDQNMCTS->selectMove(color, DQNMCTS_ITERATIONS, false);
    }
    case AGENT_EVAB: {
        /* EVAB: 学会评估的 Alpha-Beta (见 docs/agent_evab_design.md) */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfEVAB == nullptr) {
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
            auto it = s_weightPaths.find(AGENT_EVAB);
            if (it != s_weightPaths.end())
                m_sfEVAB->loadModel(it->second);
        }
        /* EVAB 的"探索"就是它的搜索; 这里额外把探索结果蒸馏回评估网络 */
        preTrainThenDecide(m_sfEVAB, color);
        return m_sfEVAB->getBestMove(color);
    }
    case AGENT_SACAZ: {
        /* SAC + MCTS + AlphaZero: 最大熵 critic 给 PUCT 搜索提供叶子估值 */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZ == nullptr) {
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
            auto it = s_weightPaths.find(AGENT_SACAZ);
            if (it != s_weightPaths.end()) {
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZ->loadModel(prefix);
            }
        }
        preTrainThenDecide(m_sfSACAZ, color);
        /* temp = 0: 取访问数最多的走法 (确定性) */
        return m_sfSACAZ->selectMove(color, SACAZ_SIMS, 0.0f);
    }
    case AGENT_SACAZ_MOE: {
        /*
           SAC + MCTS + AlphaZero, 骨干 = 稀疏路由 MoE + TransformerBlock 专家
           (E=4, top-1)。刻意用很少的模拟次数: 一次模拟 ~10.9 ms, 16 次约 175 ms。
           注意这 175 ms **只是搜索**: 每步还要跑一次在线 learnBatch, 界面上实测
           约 2.3 s/手 (见 docs/agents_design.md 13.6)。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZMoe == nullptr) {
            /*
               正常路径下这里**不会**是 nullptr —— startupLoad() 已经预加载过它了
               (用户要求"程序启动时加载所有模型")。留着这一支是兜底: 权重文件缺失、
               或者以后有人又把预加载改掉时, 至少能构造出一个可用 agent, 而不是
               空指针崩掉。
            */
            m_sfSACAZMoe = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                          SACAZAgent::Backbone::SparseMoeTb,
                                          64, SACAZ_MOE_AUX);
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE);
            if (it != s_weightPaths.end()) {
                /*
                   兜底路径同样要弹"请稍候": 这个变体的权重是 3 x 146 MB, 读一次
                   实测 6 秒。这些信号是队列投递到 GUI 线程的, 在工作线程 emit 是安全的。
                */
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE): 读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZMoe->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(m_sfSACAZMoe, color);
        return m_sfSACAZMoe->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
    }
    default: {
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAIDefault(env, AB_DEPTH);
        return abAIDefault.getBestMove(color);
    }
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
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAIForAgent(env, AB_DEPTH);
        return abAIForAgent.getBestMove(color);
    }
    case AGENT_MCTS: {
        emitStage(QStringLiteral("① 搜索 / 决策 (MCTS %1 次模拟)").arg(MCTS_SIMS));
        MCTS mctsAI(env, 1.414f);
        return mctsAI.findBestMove(color, MCTS_SIMS);
    }
    case AGENT_PG: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPG == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
        }
        preTrainThenDecide(m_sfPG, color);
        return m_sfPG->selectMove(color, false);
    }
    case AGENT_DQN: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQN == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
        }
        preTrainThenDecide(m_sfDQN, color);
        return m_sfDQN->selectMove(color, false);
    }
    case AGENT_PPOMCTS: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
        }
        preTrainThenDecide(m_sfPPOMCTS, color);
        return m_sfPPOMCTS->selectMove(color, PPO_SIMS, 0.0f);
    }
    case AGENT_DQNMCTS: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        }
        preTrainThenDecide(m_sfDQNMCTS, color);
        /* self-play 走贪心 (training=false), 否则恒为 1.0 的探索率会让它随机走子 */
        return m_sfDQNMCTS->selectMove(color, DQNMCTS_ITERATIONS, false);
    }
    case AGENT_EVAB: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfEVAB == nullptr) {
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
        }
        preTrainThenDecide(m_sfEVAB, color);
        return m_sfEVAB->getBestMove(color);
    }
    case AGENT_SACAZ: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZ == nullptr) {
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
        }
        preTrainThenDecide(m_sfSACAZ, color);
        return m_sfSACAZ->selectMove(color, SACAZ_SIMS, 0.0f);
    }
    case AGENT_SACAZ_MOE: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZMoe == nullptr) {
            m_sfSACAZMoe = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                          SACAZAgent::Backbone::SparseMoeTb,
                                          64, SACAZ_MOE_AUX);
            /*
               以前这里**只建对象、不载权重** —— 于是对弈里用到这个 agent 时跑的是
               随机初始化的网络 (界面上"选了它却像没训练过"), 而它自己在 aiThink
               路径里又会载权重, 两条路径行为不一致。
               现在正常路径都由 startupLoad() 预加载, 这一支只是兜底, 但仍然保持
               "建了对象就把权重载上"的行为, 免得两条兜底路径再分叉。
            */
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE): 读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZMoe->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(m_sfSACAZMoe, color);
        return m_sfSACAZMoe->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
    }
    default:
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAIForAgentDef(env, AB_DEPTH);
        return abAIForAgentDef.getBestMove(color);
    }
}

/* ================================================================
 *  Agent 对 Agent 对弈 (arena)
 *
 *  和原来的 selfPlay 的区别: selfPlay 是"同一个 agent 自己跟自己下", 只能看
 *  它能不能收敛; 这里让**两个不同的 agent** 打若干局, 用来比较强弱。
 *
 *  两条方法学上的要求 (少任何一条, 结果都没有解释力):
 *
 *   1. 每局交换先后手。中国象棋先手(红)优势很大, 固定谁执红的话最后只是在测
 *      "谁执红", 而不是"谁更强"。所以胜负按参赛者 A/B 记, 不按红黑记。
 *   2. 局数要够。单局的偶然性足以翻转结论, 所以界面上局数做成可配置, 并且
 *      逐局列出明细 (谁执红、谁胜、多少手)。
 * ================================================================ */

QString ChessBoard::MatchStats::summary() const
{
    QString s = QStringLiteral("%1 %2 : %3 %4")
                    .arg(agentA).arg(winA).arg(winB).arg(agentB);
    if (draws > 0) {
        s += QStringLiteral(" (和 %1)").arg(draws);
    }
    s += QStringLiteral("  共 %1 局 / %2 手").arg(games).arg(plies);
    if (agentErrors > 0) {
        s += QStringLiteral("  [%1 次无效走法已兜底]").arg(agentErrors);
    }
    if (aborted) {
        s += QStringLiteral("  [已中止]");
    }
    return s;
}

QString ChessBoard::MatchStats::detail() const
{
    QString d = summary();
    d += QStringLiteral("\n\n参赛方:\n  A = %1\n  B = %2").arg(agentA, agentB);
    d += QStringLiteral("\n\n每局明细 (每局交换先后手):\n");
    d += log;
    if (plies > 0) {
        const double avg = double(totalThinkMs) / double(plies);
        d += QStringLiteral("\n思考耗时: 累计 %1 s, 平均 %2 ms/手, 单步最长 %3 ms")
                 .arg(totalThinkMs / 1000.0, 0, 'f', 1)
                 .arg(avg, 0, 'f', 0)
                 .arg(maxThinkMs);
    }
    return d;
}

ChessBoard::AgentType ChessBoard::typeForTurn(int turn, AgentType redType,
                                              AgentType blackType) const
{
    return (turn == Stone::COLOR_RED) ? redType : blackType;
}

/* 打一局: 红方 redType, 黑方 blackType。返回 Chess::RESULT_*, 被中止则返回 ONGOING */
int ChessBoard::playMatchGame(AgentType redType, AgentType blackType, MatchStats &st,
                              double &rewardRed, double &rewardBlack, double &rewardA,
                              double &rewardB)
{
    rewardRed = 0.0;
    rewardBlack = 0.0;
    rewardA = 0.0;
    rewardB = 0.0;
    {
        QMutexLocker locker(&mutex);
        chess.reset();
    }
    int turn = Stone::COLOR_RED;
    int moves = 0;
    int ret = Chess::RESULT_DRAW;

    while (moves < m_maxPliesPerGame) {
        if (m_matchAbort.load()) {
            return Chess::RESULT_ONGOING;
        }
        m_selfPlayMoveNo = moves + 1;

        {
            QMutexLocker locker(&mutex);
            const int r = chess.getResult(turn);
            if (r != Chess::RESULT_ONGOING) {
                ret = r;
                break;
            }
        }

        const AgentType who = typeForTurn(turn, redType, blackType);
        auto t0 = std::chrono::steady_clock::now();
        Step step = aiThinkForAgent(turn, who);
        auto t1 = std::chrono::steady_clock::now();
        const long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        st.totalThinkMs += ms;
        if (ms > st.maxThinkMs) {
            st.maxThinkMs = ms;
        }
        st.plies++;
        /* 让界面的"AI思考时间"标签逐手跳动 */
        emit aiThinkFinished(ms);

        /* 无合法走法 (将杀 / 困毙) */
        if (!step.valid) {
            /*
               先分清"真的无棋可走"和"agent 返回了一个无效走法"。
               后者不是将杀 —— 以前直接把 !step.valid 当成"走棋方被将死", 于是一个
               抽风的 agent 会让对局在没走过任何一步的情况下被判负 (界面上就是
               "棋子没动, 我却输了")。这里用第一个合法走法兜底, 并把次数记下来。
            */
            std::vector<Step *> legal;
            {
                QMutexLocker locker(&mutex);
                chess.sample(turn, legal);
                if (!legal.empty()) {
                    /*
                       这个分支正常永远不该进。真进来了说明某个 agent 的搜索返回了
                       无效走法 —— 那是个 bug, 但**不能让它决定胜负**: 用第一个合法
                       走法兜底, 并把次数记进 MatchStats (显示在比分行里)。
                    */
                    std::fprintf(stderr,
                                 "[arena] agent(%d) 返回无效走法 (第 %d 局第 %d 手): "
                                 "valid=%d id=%d pos=(%d,%d)->(%d,%d), "
                                 "仍有 %zu 个合法走法, 已兜底\n",
                                 (int)who, st.games + 1, moves + 1,
                                 (int)step.valid, step.id,
                                 step.pos.x, step.pos.y, step.nextPos.x, step.nextPos.y,
                                 legal.size());
                    step = *legal[0];
                    st.agentErrors++;
                }
            }
            Steps::instance().put(legal);
        }

        if (!step.valid) {
            /* 确实没有合法走法 -> 走棋方被将杀 / 困毙 */
            ret = (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                             : Chess::RESULT_RED_WIN;
            break;
        }

        {
            QMutexLocker locker(&mutex);
            double totalReward = 0;
            /* 走这一步的是 turn 方, moveForward 之前先记下来 */
            const int mover = turn;
            chess.moveForward(&step, totalReward);
            /*
               环境奖励记账。`Chess::moveForward` 的 totalReward 是**黑方视角**的记账
               (吃红子 +value, 吃黑子 −value), 而 RL agent 学的是"走子方视角"的奖励
               (与 SACAZAgent::computeReward、终局 ±1 同一套约定), 所以这里换算过去。
               换算公式: 黑方视角 == −(红方视角), 于是
                 红方走: 走子方收益 = −totalReward
                 黑方走: 走子方收益 = +totalReward
               (实测探针: 红炮吃黑马 totalReward = −0.30, 即红方收益 +0.30。)
            */
            const double moverReward =
                (mover == Stone::COLOR_RED) ? -totalReward : totalReward;
            if (mover == Stone::COLOR_RED) {
                rewardRed += moverReward;
            } else {
                rewardBlack += moverReward;
            }
            /* A/B 的归属由 matchAgents 按"这一局谁执红"换算, 这里只管红黑 */
            /* sideToMove 必须跟着 turn 走, 否则下一手 getResult 会看错方 */
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            chess.sideToMove = turn;
        }
        moves++;
    }

    /*
       终局奖励 ±1 也算进"本局环境奖励"里 —— 否则这条曲线只反映吃子, 看不出输赢。
       和棋不加不减 (0)。
    */
    if (ret == Chess::RESULT_RED_WIN) {
        rewardRed += 1.0;
        rewardBlack -= 1.0;
    } else if (ret == Chess::RESULT_BLACK_WIN) {
        rewardBlack += 1.0;
        rewardRed -= 1.0;
    }

    /*
       达到步数上限: 原来 selfPlay 按 evaluate() 判胜 (score == 0 也返回红胜),
       于是"平局"分支永远不可达。现在直接判和棋 —— 中国象棋的自然限着本来就是和棋。
    */
    return ret;
}

ChessBoard::MatchStats ChessBoard::matchAgents(AgentType typeA, AgentType typeB, int games)
{
    MatchStats st;
    st.agentA = agentDisplayName(typeA);
    st.agentB = agentDisplayName(typeB);
    if (games < 1) {
        games = 1;
    }

    /*
       对弈期间屏蔽玩家点击。用独立标志而不是改 state —— state 是给 AI 工作线程
       看的, 改它会把那个线程也唤醒起来思考, 于是两个线程同时写同一张棋盘。
    */
    m_selfPlaying = true;
    m_selfPlayMoveNo = 0;
    m_matchAbort = false;
    m_matchRunning = true;
    m_matchGames = games;
    m_matchGameNo = 0;

    /*
       如果玩家是在"AI 正在思考"的时候按下的开始对弈, 棋盘上还有一个在飞的一手。
       把代数 +1 让它作废、把 state 收回 IDLE, 否则工作线程会和这里的对弈线程
       同时写 chess。
    */
    {
        QMutexLocker locker(&mutex);
        ++m_thinkGeneration;
        state = STATE_IDEL;
        selectID = -1;
        condit.wakeAll();
    }

    emit aiThinkingStarted(QStringLiteral("%1 vs %2").arg(st.agentA, st.agentB),
                           m_preTrainEnabled.load() ? m_preTrainSteps.load() : 0);
    emit matchStarted(st.agentA, st.agentB, games);

    for (int g = 0; g < games; ++g) {
        if (m_matchAbort.load()) {
            st.aborted = true;
            break;
        }
        m_matchGameNo = g + 1;

        /* 每局交换先后手: 偶数局 A 执红, 奇数局 B 执红 */
        const bool aIsRed = (g % 2 == 0);
        const AgentType redType = aIsRed ? typeA : typeB;
        const AgentType blackType = aIsRed ? typeB : typeA;

        double rewardRed = 0.0;
        double rewardBlack = 0.0;
        double rewardA = 0.0;
        double rewardB = 0.0;
        const int res = playMatchGame(redType, blackType, st, rewardRed, rewardBlack,
                                      rewardA, rewardB);
        if (res == Chess::RESULT_ONGOING) {
            st.aborted = true;      /* playMatchGame 用 ONGOING 表示"被中止" */
            break;
        }

        /* A/B 视角的本局环境奖励 (谁执红就把红方那份记给谁) */
        const double rA = aIsRed ? rewardRed : rewardBlack;
        const double rB = aIsRed ? rewardBlack : rewardRed;

        QString line = QStringLiteral("  第 %1 局: 红=%2 黑=%3 -> ")
                           .arg(g + 1)
                           .arg(aIsRed ? st.agentA : st.agentB,
                                aIsRed ? st.agentB : st.agentA);
        if (res == Chess::RESULT_DRAW) {
            line += QStringLiteral("和棋");
            st.draws++;
        } else {
            const bool redWon = (res == Chess::RESULT_RED_WIN);
            const bool aWon = (redWon == aIsRed);
            line += aWon ? QStringLiteral("%1 胜").arg(st.agentA)
                         : QStringLiteral("%1 胜").arg(st.agentB);
            if (aWon) {
                st.winA++;
            } else {
                st.winB++;
            }
        }
        st.games++;
        /*
           每局明细里带上"本局环境奖励": 这条曲线的意义是"模型下完一局拿到了多少
           奖励", 所以它同时也是逐局明细的一部分 (界面上的列表直接用它)。
        */
        const QString rewardText =
            QStringLiteral("  奖励 A=%1 B=%2").arg(rA, 0, 'f', 2).arg(rB, 0, 'f', 2);
        st.log += line + QStringLiteral("  (%1 手)").arg(m_selfPlayMoveNo.load())
                  + rewardText + QStringLiteral("\n");

        emit matchGameFinished(st.games, games, line + rewardText);
        /* ---- 实时比分 (界面用, 见 matchScoreChanged 的注释) ---- */
        {
            QString score = QStringLiteral("%1 %2 : %3 %4")
                                .arg(st.agentA).arg(st.winA).arg(st.winB).arg(st.agentB);
            score += QStringLiteral("   和 %1").arg(st.draws);
            score += QStringLiteral("   (%1/%2 局)").arg(st.games).arg(games);
            emit matchScoreChanged(score);
        }
        emit gameRewardSample(st.games, st.agentA, st.agentB, rA, rB);
    }

    m_matchRunning = false;
    m_selfPlaying = false;
    m_selfPlayMoveNo = 0;
    m_matchGameNo = 0;
    m_matchGames = 0;
    emit aiThinkingStopped();
    emit matchFinished(st.summary(), st.detail());
    return st;
}

/* ================================================================
 *  保存AI模型权重
 * ================================================================ */

/*
 * defaultWeightPath - 每个 agent 的正式权重路径 (单一来源)
 *
 * 这些字符串以前散在 shutdownSave() 里, 而"对弈结束后保存"在 GUI 里另有一套
 * (走文件对话框, 默认文件名还不一样) —— 两套名字不一致就会静默失效: 用户以为
 * 存上了, 下次启动读的却是另一个文件。现在统一到这里。
 */
std::string ChessBoard::defaultWeightPath(AgentType agentType)
{
    switch (agentType) {
    case AGENT_PG:        return "weights/pg_agent.dat";
    case AGENT_DQN:       return "weights/dqn_agent.dat";
    case AGENT_PPOMCTS:   return "weights/ppomcts_agent.dat";
    case AGENT_DQNMCTS:   return "weights/dqnmcts_agent.dat";
    case AGENT_EVAB:      return "weights/evab_agent.dat";
    /* SAC+AZ 系: 前缀 -> <prefix>_actor / _q1 / _q2 */
    case AGENT_SACAZ:     return "weights/sacaz_agent";
    case AGENT_SACAZ_MOE: return "weights/sacaz_moe_agent";
    default:              return std::string();
    }
}

bool ChessBoard::hasAgentInstance(AgentType agentType) const
{
    switch (agentType) {
    case AGENT_PG:        return m_sfPG != nullptr;
    case AGENT_DQN:       return m_sfDQN != nullptr;
    case AGENT_PPOMCTS:   return m_sfPPOMCTS != nullptr;
    case AGENT_DQNMCTS:   return m_sfDQNMCTS != nullptr;
    case AGENT_EVAB:      return m_sfEVAB != nullptr;
    case AGENT_SACAZ:     return m_sfSACAZ != nullptr;
    case AGENT_SACAZ_MOE: return m_sfSACAZMoe != nullptr;
    default:              return false;
    }
}

bool ChessBoard::saveCurrentAgentModel(AgentType agentType, const std::string &filepath)
{
    /*
       保存也可能很慢 (稀疏 MoE 骨干 28.7 M 参数, 权重按文本编码时是几百 MB), 所以
       同样报一次"请稍候"。用 RAII 保证**任何返回路径**都会发 busyFinished ——
       以前手工在每个 return 前收尾的话, 漏一个分支就是"弹窗永远挂着"。
    */
    struct BusyGuard {
        ChessBoard *self;
        explicit BusyGuard(ChessBoard *s, const QString &what) : self(s)
        {
            emit self->busyStarted(QStringLiteral("正在保存"), what);
        }
        ~BusyGuard() { emit self->busyFinished(); }
    } guard(this, QStringLiteral("写出 %1 的权重…")
                       .arg(agentDisplayName(agentType)));

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
    case AGENT_EVAB: {
        if (m_sfEVAB == nullptr) return false;
        return m_sfEVAB->saveModel(filepath);
    }
    case AGENT_SACAZ: {
        if (m_sfSACAZ == nullptr) return false;
        /* 一个模型三个文件: filepath 是前缀 -> filepath_actor / _q1 / _q2 */
        return m_sfSACAZ->saveModel(filepath);
    }
    case AGENT_SACAZ_MOE: {
        if (m_sfSACAZMoe == nullptr) return false;
        return m_sfSACAZMoe->saveModel(filepath);
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
        /*
           注意: 训练线程只在**每轮**开始时检查 m_bgTraining, 而一轮是"克隆权重 ->
           训练 BG_TRAIN_EPISODES 局", 所以关窗的等待时间由单轮时长决定。
           轮次规模就是按这个约束选的 (见 BG_TRAIN_* 常量), 不要随手调大。
        */
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
            /*
               这一轮训练结束后的损失 (界面的"训练损失曲线")。不上报损失的 agent
               (PG/PPO/DQN+MCTS 的训练循环里没有 scalar loss) 保持 NaN -> 不上图。
            */
            float roundLoss = std::numeric_limits<float>::quiet_NaN();

            switch (type) {
            case AGENT_PG: {
                PGEagent clone(trainChess, 64, 0.9f, 0.01f, 1.0f);
                clone.loadPolicy(TMP_WEIGHTS);
                clone.train(BG_TRAIN_EPISODES, BG_TRAIN_MAX_MOVES, true, false);
                roundLoss = clone.getLastTrainLoss();
                clone.savePolicy(TMP_WEIGHTS);
                break;
            }
            case AGENT_DQN: {
                DQNAgent clone(trainChess, 64, 0.99f, 0.001f, 1.0f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(BG_TRAIN_EPISODES, BG_TRAIN_MAX_MOVES, false);
                roundLoss = clone.getLastTrainLoss();
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            case AGENT_PPOMCTS: {
                PPOMCTSAgent clone(trainChess, 64, 0.99f, 0.001f, 1.414f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(BG_TRAIN_EPISODES, BG_TRAIN_SIMS,
                                    BG_TRAIN_MAX_MOVES, false);
                roundLoss = clone.getLastTrainLoss();
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            case AGENT_DQNMCTS: {
                DQNMCTSAgent clone(trainChess, 128, 0.99f, 0.001f, 1.0f, 1.414f);
                clone.loadModel(TMP_WEIGHTS);
                clone.trainSelfPlay(BG_TRAIN_EPISODES, BG_TRAIN_SIMS,
                                    BG_TRAIN_MAX_MOVES, false);
                roundLoss = clone.getLastTrainLoss();
                clone.saveModel(TMP_WEIGHTS);
                break;
            }
            default: break;
            }

            if (std::isfinite(roundLoss)) {
                emit trainLossSample((double)roundLoss, agentDisplayName(type),
                                     m_trainSampleNo.fetch_add(1) + 1);
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

    /*
       保存 aiThink / aiThinkForAgent 共享的 neural agent 权重。
       路径全部走 defaultWeightPath() —— 与"对弈结束后静默保存"用的是同一份,
       不会再出现"存这边、读那边"的静默失效。
       (EVAB 曾经漏在这里: 它是界面上可选、能被在线训练的 agent, 但退出时从来不落盘,
        于是每次启动都从随机价值网络重新开始, 上一局学到的东西全丢。)
    */
    const AgentType all[] = { AGENT_PG, AGENT_DQN, AGENT_PPOMCTS, AGENT_DQNMCTS,
                              AGENT_EVAB, AGENT_SACAZ, AGENT_SACAZ_MOE };
    for (AgentType t : all) {
        if (hasAgentInstance(t)) {
            saveCurrentAgentModel(t, defaultWeightPath(t));
        }
    }
}

/* ================================================================
 *  回放功能
 * ================================================================ */

/*
 * applyDbStep: 把一条数据库走法记录落到棋盘上。
 *
 * 三处回放代码原来都写成 `step->nextId = dbStep.toX;` —— 把终点**列坐标**
 * (0..8) 当成了棋子 id。moveForward/moveBack 会用 nextId 去索引 m_children
 * 并结算收益, 语义完全是错的; 而且 step->valid 一直是 false。
 */
bool ChessBoard::applyDbStep(const DBStep &dbStep)
{
    const Pos from(dbStep.fromX, dbStep.fromY);
    const Pos to(dbStep.toX, dbStep.toY);
    if (chess.m_map.isInner(from) == false || chess.m_map.isInner(to) == false) {
        return false;
    }
    Stone *stone = chess.m_map[from];
    if (stone == nullptr || stone->alive == false) {
        return false;
    }
    Stone *victim = chess.m_map[to];
    if (victim != nullptr && victim->color == stone->color) {
        return false;   /* 记录与当前局面不一致 */
    }

    Step step;
    step.id = stone->id;
    step.pos = from;
    step.nextId = (victim != nullptr) ? victim->id : Stone::ID_NONE;
    step.nextPos = to;
    step.reward = 0;
    step.valid = true;

    double totalReward = 0;
    chess.moveForward(&step, totalReward);
    return true;
}

void ChessBoard::loadReplayGame(int gameId, const QVector<DBStep> &steps)
{
    m_replayGameId = gameId;
    m_replaySteps = steps;
    chess.reset();
    /* 应用到所有步 */
    for (int i = 0; i < steps.size(); i++) {
        applyDbStep(steps[i]);
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
    if (applyDbStep(m_replaySteps[m_replayIndex]) == false) {
        m_replayIndex++;
        return false;
    }
    m_replayIndex++;
    update();
    emit replayIndexChanged(m_replayIndex, m_replaySteps.size());
    return true;
}

void ChessBoard::applyReplayStep(int targetIndex)
{
    chess.reset();
    for (int i = 0; i < targetIndex; i++) {
        applyDbStep(m_replaySteps[i]);
    }
}
