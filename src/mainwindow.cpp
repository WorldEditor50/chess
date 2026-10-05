#include "mainwindow.h"
#include <chrono>
#include <cstdlib>             /* std::getenv —— MoE 负载读数的按需诊断 (见 CHESS_MOE_TRACE) */
#include "ui_mainwindow.h"
#include "chessboard.h"
#include "thinkingindicator.h"
#include "metricsview.h"
#include "busydialog.h"
#include "qssloader.hpp"
#include "rl/cpuinfo.hpp"
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>
#include <QCheckBox>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTextCursor>      /* BC 的进度行要追加到自检视图末尾 (2026-10) */
#include <QTabWidget>       /* "训练损失 / 行为克隆"两个 tab (2026-10 用户口径) */
#include <QLabel>           /* BC 曲线下面的读数行 (代码里建) */
#include <QVBoxLayout>      /* tab 页的布局 */
#include <QFileDialog>
#include <QMessageBox>
#include <QListWidgetItem>
#include <QFile>
#include <QTextStream>
#include <QDateTime>

namespace {

/* 三个下拉框共用的 agent 清单 (名称 + 枚举), 只留一个来源 */
struct AgentChoice {
    const char *name;
    ChessBoard::AgentType type;
};

const AgentChoice kAgents[] = {
    { "Alpha-Beta Pruning (深度=4)", ChessBoard::AGENT_ALPHABETA },
    /*
       ---- Alpha-Beta 的三档弱等级 (2026-09, 用户口径: "1 到 3 level 加入下拉框") ----
       同一个搜索 (ABAgent), **只有深度不同**: L1=1 / L2=2 / L3=3。
       【实测 2026-09 · test_ab benchmark · 本机 · 初始局面】每步 0 / 3 / 34 ms
       (上面那一档深度 4 = 90 ms, 深度 5 = 1920 ms; 常量见 chessboard.cpp 的 AB_*_DEPTH)。
       放在最前面, 是因为它们是这套列表里**最弱、最快**的对手 —— 当"陪练/标尺"用,
       在对弈里跟学习型 agent 交手 (对弈双方共用一个下拉框清单, 见 fillAgentCombo)。
       ⚠ 它们是**纯搜索、没有任何可训练权重**: 叶子价值来自 Chess::evaluate()。
       "轮到它走"时不会训练, 也不上报训练损失 (损失曲线上没有它的点, 这是正确行为);
       在自检面板里会写明这一句, 免得"没有曲线"被读成"训练没跑起来"。
    */
    { "Alpha-Beta L1 (深度=1, 最弱)", ChessBoard::AGENT_AB_L1 },
    { "Alpha-Beta L2 (深度=2)",       ChessBoard::AGENT_AB_L2 },
    { "Alpha-Beta L3 (深度=3)",       ChessBoard::AGENT_AB_L3 },
    { "MCTS (800次模拟)",            ChessBoard::AGENT_MCTS },
    /*
       ================================================================
       [2026-10 用户口径] 下拉框移除 PG / DQN / SAC 行为还原版(两支)
       ================================================================
       用户要求: "界面下拉框移除 PG, DQN, SAC legacy agent, 减少加载时间"。

       **这四个 agent 的类、权重文件、bench/测试全都留着**, 只是不再出现在界面上、
       也不再在启动时预加载。省下的时间是**实测**的 (本机, 每次都是完整冷启动):

           [weights] PG:                        146 ms
           [weights] DQN:                        171 ms
           [weights] SAC+AZ-59e5233:              43 ms     (MLP 还原版, 权重只 0.5 MB x3)
           [weights] SAC+AZ-59e5233-MoE 建网:   4499 ms     (5 个 TB 专家网络)
           [weights] SAC+AZ-59e5233-MoE 读权重: 6300 ms     (3 x 146 MB)
                                         合计 ≈ 11.2 s / 40.8 s 启动

       也就是说: 真正贵的是**还原版的 MoE+TB 那一支**(建网 + 读 440 MB), 而 PG/DQN
       各只有 ~0.15 s。要省时间就必须把这四个一起摘掉。

       ⚠ 被摘掉的**能力** (诚实记一笔, 因为这些正是它们存在的理由):
         1. 界面上再也**选不到** "当前口径 vs 59e5233 口径" 的直接对弈 —— 那条对比
            现在只能走 bench (test_sacaz 的 [14] 节与 bench_sacaz_vs_ab --legacy 仍在);
         2. PG / DQN 是这套列表里最"教科书"的两个基线, 界面上没有了 (代码与
            test_grad / test_dqnab 不受影响);
         3. "全部模型自检"那份横向对照里也不再列它们 (见 chessboard.cpp 的 kAll 说明)。
       想恢复某一条: 把对应项加回这张表 **并且** 加回 chessboard.cpp 的 kWeightAgents
       (两处都要 —— 只加前者会"能选中但没预热", 只加后者会"预热了但选不到")。
    */
    /*
       ---- "同一算法, 两种骨干" 的两组, 刻意**成对排列** ----
       界面上把它们挨着放, 就是为了能直接选中互相对弈比较:
         PPO+MCTS    (稀疏MoE + TB 专家, E=4 top-1, 现役)
         PPO+MCTS    (稀疏MoE + MLP 专家, E=8 top-2, 便宜 ~25x / 容量小 ~18x)
         SAC+MCTS+AZ (MLP 骨干, 最大熵搜索)
         SAC+MCTS+AZ (稀疏MoE + TB 专家)
       两组各自是"同一份实现 + 不同 Backbone" (见 chessboard.h 的枚举注释与 rl/ppo.h)。
    */
    { "PPO+MCTS (AlphaZero)",        ChessBoard::AGENT_PPOMCTS },
    /*
       PPO+MCTS 的另一个骨干: 稀疏 MoE + **MLP 专家** (E=8 top-2) —— 也就是 2026-09
       那次改版之前 PPO 用的配置 (见 rl/ppo.h 顶部那张实测表)。
       与上面那一项**算法/搜索/训练完全是同一份代码**, 差别只有骨干:
         MlpExpert 便宜 ~25x (前向 0.139 ms vs TB 3.59 ms)、容量小 ~18x
         (2.15 M vs 38.0 M 参数), 所以同一时间预算下它能跑更多模拟 (PPO_MLP_SIMS=1600)。
    */
    { "PPO+MCTS (AlphaZero, 稀疏MoE+MLP专家)", ChessBoard::AGENT_PPOMCTS_MLP },
    { "DQN+MCTS (AlphaZero)",        ChessBoard::AGENT_DQNMCTS },
    /*
       [2026-09 dev-dqnmcts-moetb] **同一个算法, 另一个骨干** —— 与上面那一项成对排列
       (房子里的惯例: "同算法不同骨干"挨着放, 才能直接选中互相对弈比较)。
       它**不是**上面那一支的运行时开关, 而是一个**独立的 C++ 类**
       (DQNMCTSMOETbAgent, src/dqnmctsmoetbagent.h): 骨干换成稀疏 MoE (E=4 top-1) +
       TransformerBlock 专家, 表示换成规范视角的 1263 维, 搜索用 PUCT + negamax 符号,
       学习用 Double DQN + clampTarget/Huber。旧的那一支一位没动, 于是两条可以直接对弈。
       代价: 一个 TB 专家前向 ~5-6 ms ⇒ 每次走子只给 DQNMCTS_MOE_SIMS = 40 次模拟
       (约 175 ms/步, 与 SACAZ_MOE_SIMS 同一预算档; 旧类的 200 是 d_model=90 的预算)。
       **权重文件独立** (weights/dqnmcts_moe_agent_trunk / _q, 旧类是单文件
       weights/dqnmcts_agent.dat) —— 两者结构不同, 共用一个前缀只会静默互相覆盖。
    */
    { "DQN+MCTS (稀疏MoE+TB专家)",   ChessBoard::AGENT_DQNMCTS_MOE },
    { "EVAB (学会评估的 Alpha-Beta)", ChessBoard::AGENT_EVAB },
    { "SAC+MCTS+AlphaZero (最大熵搜索)", ChessBoard::AGENT_SACAZ },
    /*
       同一个算法, 骨干换成"稀疏路由 MoE + TransformerBlock 专家" (E=4, top-1)。
       与上一项相比: 参数量大 ~4 倍 (4 个 TB 专家), 算力只算 1 个专家 —— 实测
       10.9 ms/模拟 (MLP 骨干 0.07), 所以每次走子只给 16 次模拟 (约 175 ms)。
    */
    { "SAC+MCTS+AlphaZero (稀疏MoE+TB专家)", ChessBoard::AGENT_SACAZ_MOE },
    /*
       [2026-09 独立类] 同算法、骨干换成**稀疏 MoE + MLP 专家** (E=8, top-2) 的那一支,
       也就是独立类 `SACAZMoEMlpAgent`。它以前**只能在 bench 里跑到**; 现在接进下拉框。
       实测比上面那一支便宜 (2.90 M 参数 / ~2.9 ms/模拟 vs 57.6 M / ~5.4 ms),
       所以模拟次数给到 SACAZ_MOE_MLP_SIMS = 40。
    */
    { "SAC+MCTS+AlphaZero (稀疏MoE+MLP专家)", ChessBoard::AGENT_SACAZ_MOE_MLP },
    /*
       (原来这里还有两支 SAC 的 59e5233 **行为还原版**: MLP 骨干与 MoE+TB 骨干。
        2026-10 按用户口径从下拉框移除 —— 理由与实测省下的时间见本表顶部那段注释。
        两个类 (SACAZLegacyAgent 的两个 Backbone) 与它们的权重前缀都还在,
        `test_sacaz` [14] 节与 `bench_sacaz_vs_ab --legacy` 仍在用它们。)
    */
    /*
       DQN+AB: **把 Alpha-Beta 当成 DQN 的 planning head**。
       网络 (稀疏 MoE + TB 专家 + Dueling 双头) 给 AB 排序与叶子值, AB 的展开结果
       反过来当 TD 目标。与上面几个 agent 的关键差别: 它的"搜索"是**对抗展开**,
       杀棋/战术看得见 (实测一步杀局面 8/8 命中, 而纯 Q-argmax 是 0/8)。
       每步给 256 个搜索节点 (TB 骨干约 0.8 s/步、2~3 层), 见 DQNAB_NODES。
    */
    { "DQN+AB (AB+DuelingDQN, 稀疏MoE+TB专家)", ChessBoard::AGENT_DQNAB },
};

/*
   ---- 下拉框: 整份列表都要**看得见** (2026-09) ----
   Qt 的 QComboBox 默认 maxVisibleItems = **10**, 而列表比这多 (加
   "PPO+MCTS (...MLP专家)" 那次是 11 项, 加 Alpha-Beta 三档弱等级之后一度到 18 项,
   2026-10 移除 PG/DQN/SAC 还原版两支之后是 **14 项**) —— 第 11 项及以后会被折叠在
   滚动区里, 打开下拉框只看到 10 行。
   表现就是"明明加进列表了, 界面上却找不到" (UIA 实测: 展开后只有 10 行可见).
   所以这里按条数放宽: 全部条目一次性可见, 不需要滚动。
   (这一行**不要**写死数字: 它读的是表的真实条数, 以后再加 agent 也不会忘 ——
    上面那个"14 项"只是当时的读数, 会过期, 而这一行代码不会。)
*/
void fillAgentCombo(QComboBox *combo, ChessBoard::AgentType defaultType)
{
    combo->clear();
    for (const AgentChoice &c : kAgents) {
        combo->addItem(QString::fromUtf8(c.name), static_cast<int>(c.type));
    }
    combo->setMaxVisibleItems((int)(sizeof(kAgents) / sizeof(kAgents[0])) + 2);
    /*
       默认项按**类型**选, 而不是按下标: 这个列表是会被插入/重排的
       (上面就把新 agent 插进了 PPO+MCTS 后面), 而写死的下标会**静默**指到别的 agent
       —— 原来 matchB 用的是 `fillAgentCombo(..., 6)`, 插入一项之后就会变成 DQN+MCTS。
    */
    const int idx = combo->findData(static_cast<int>(defaultType));
    combo->setCurrentIndex(idx >= 0 ? idx : 0);
}

/* 只有带参数的 agent 才有权重可存 (Alpha-Beta / MCTS 是纯搜索) */
bool agentIsTrainable(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PG:
    case ChessBoard::AGENT_DQN:
    case ChessBoard::AGENT_PPOMCTS:
    case ChessBoard::AGENT_DQNMCTS:
    case ChessBoard::AGENT_EVAB:
    case ChessBoard::AGENT_SACAZ:
    case ChessBoard::AGENT_SACAZ_MOE:
    case ChessBoard::AGENT_SACAZ_MOE_MLP:
    case ChessBoard::AGENT_SACAZ_OLD:
    case ChessBoard::AGENT_SACAZ_OLD_MOE:
    case ChessBoard::AGENT_DQNAB:
    /* [2026-09 dev-dqnmcts-moetb] DQN+MCTS (稀疏MoE+TB专家): 有参数可存 (主干 + Q 头) */
    case ChessBoard::AGENT_DQNMCTS_MOE:
    case ChessBoard::AGENT_PPOMCTS_MLP:
        return true;
    default:
        return false;
    }
}

QString agentLongName(ChessBoard::AgentType type)
{
    for (const AgentChoice &c : kAgents) {
        if (c.type == type) {
            return QString::fromUtf8(c.name);
        }
    }
    return QStringLiteral("agent");
}

/*
    这里原来还有一个 agentWeightFilename(): 它是"另存为"文件对话框的默认文件名,
    与 ChessBoard::defaultWeightPath() 是**两套**名字 —— 正是那种"存到 A、读 B"的
    隐患。改成静默保存(存到标准路径)之后它没有用处了, 删掉, 只留一个来源。
*/

/*
 * 曲线配色: 按"第几条曲线"取, 与 agent 名无关 (同一对 agent 每次对弈颜色一致,
 * 便于跨场比较)。取值都在米色底 (#fef9e3) 上够清楚。
 */
const QColor kSeriesColors[] = {
    QColor(47, 127, 214),    /* 蓝 */
    QColor(198, 90, 40),     /* 砖红 */
    QColor(46, 125, 50),     /* 绿 */
    QColor(140, 82, 172),    /* 紫 */
    QColor(186, 140, 30),    /* 金 */
    QColor(0, 131, 143)      /* 青 */
};
const int kSeriesColorCount = 6;

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(1360, 760);

    /*
      把本构建实际启用的 SIMD 指令集写进窗口标题。
      它不是"这台机器支持什么", 而是"这个二进制带了什么内核" —— SIMD 是编译期
      选定的 (见 rl/cpuinfo.hpp), 带 AVX2 的二进制要求 CPU 支持 AVX2。写在标题上
      是为了回答一个很常见的问题: "现在到底有没有在跑 AVX2?"
    */
    setWindowTitle(QString("中国象棋 - Qt/AI  [%1]")
                   .arg(QString::fromStdString(RL::cpuinfo::describe())));
    qInfo().noquote() << "[SIMD]" << QString::fromStdString(RL::cpuinfo::describe());

    /* 填充AI Agent下拉框 */
    populateAgentComboBox();

    /*
       ---- [2026-10] "对局中行为克隆"那个勾选框的接线 ----
       勾选框本身在 populateAgentComboBox 里建 (与 matchModeRow 后面那四行同一处);
       这里接三件事:
         1. 板子上的 bcProgress 一行 -> 追加进自检面板 (对局期间的实时读数);
         2. A/B 两个下拉框一变就重算"能不能勾" (老师/学生换了);
         3. 启动完成后再刷一次 (启动前 agent 实例还没建出来, 判据要用到类型而不是实例,
            所以严格说这里只是为了让 tooltip 的文字在启动后立刻是正确的)。
    */
    QObject::connect(ui->gameWidget, &ChessBoard::bcProgress, this,
                     [this](const QString &line) {
                         if (ui->selfCheckView == nullptr) {
                             return;
                         }
                         ui->selfCheckView->moveCursor(QTextCursor::End);
                         ui->selfCheckView->insertPlainText(line + QStringLiteral("\n"));
                         ui->selfCheckView->moveCursor(QTextCursor::End);
                     });
    QObject::connect(ui->matchAComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                     this, [this](int) { updateBcMatchControlsEnabled(); });
    QObject::connect(ui->matchBComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                     this, [this](int) { updateBcMatchControlsEnabled(); });
    /*
       "对战AI" 那一支也要接: 人机对战时**学生就是它** (老师来自 BC 下拉框, 与"对面"
       是人是 AI 无关 —— 2026-10 口径)。不接的话, 在人机模式下把 AI 换成 PPO/SAC 之后
       BC 下拉框还是灰的, 用户看到的是"功能没生效"。
    */
    if (ui->agentComboBox != nullptr) {
        QObject::connect(ui->agentComboBox,
                         QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, [this](int) { updateBcMatchControlsEnabled(); });
    }
    updateBcMatchControlsEnabled();

    /*
       ================================================================
       [2026-10 门控实验] 稀疏 MoE 专家负载的可视化小控件
       ================================================================
       为什么必须有它: 稀疏 MoE 的失败模式**全是静默的** —— 路由坍缩之后前向照跑、
       loss 照降、权重照存, 界面上一点异常都没有。本工程实测过两个极端:
         * 只靠辅助损失: 训练侧 MaxVio 0.318~0.963 (最弱专家只有 1.1% 流量);
         * 开了无辅助损失偏置均衡: MaxVio 0.006~0.009 (几乎完美)。
       这两者在**其它任何读数上都分不出来**, 只有把每个专家的流量画出来才看得见。

       [2026-10 用户口径] **放在最右列 (metricsPanel)**, 紧跟"当前比分"那一行之下。
       它原来跟着 selfCheckView 走 (插在自检文本框前面) —— 但自检面板已经被用户口径
       搬到**中间列**了, 而 MoE 负载要留在右侧, 所以这里**解耦**: 不再按 selfCheckView
       定位, 而是直接插进 metricsPanel 的竖向布局。
       仍然用代码插入而不是写进 mainwindow.ui: .ui 是 Qt Designer 的 XML, 为一个只读
       小控件去动它有回归风险, 而 `insertWidget` 的落点是稳定的。
    */
    m_moeLoadView = new MoeLoadView(this);
    {
        QBoxLayout *lay = (ui->metricsPanel != nullptr)
                              ? qobject_cast<QBoxLayout *>(ui->metricsPanel->layout())
                              : nullptr;
        if (lay != nullptr) {
            /* 插在"当前比分"之后 (它占着 0 号位): 比分 -> MoE 负载 -> 曲线 */
            const int at = (lay->count() > 0) ? 1 : 0;
            lay->insertWidget(at, m_moeLoadView);
        } else {
            /* 布局结构变了也不静默丢掉这个读数: 退化成独立窗口 (仍然可用) */
            m_moeLoadView->setWindowTitle(QStringLiteral("稀疏 MoE 专家负载"));
            m_moeLoadView->resize(340, 108);
            m_moeLoadView->show();
        }
    }
    /*
       双击 -> 弹一个放大的独立窗口 (与 CurveChart 的双击放大同一约定)。
       窗口是**非模态 + WA_DeleteOnClose**: 可以一边跑对弈一边看; 已经开着就抬到前面,
       不重复开。数据不在这里取 —— 它由下面那个每手刷新的自检 worker 一起喂
       (见 applySelfCheckPanel), 因为读 agent 计数器要在锁上等。
    */
    connect(m_moeLoadView, &MoeLoadView::doubleClicked, this, [this]() {
        if (m_moeLoadDialog == nullptr) {
            m_moeLoadDialog = new MoeLoadDialog(this);
        }
        m_moeLoadDialog->setSnapshot(m_moeLoadSnapshot);
        /*
           新窗口的开关默认是关的, 所以这里必须把**当前**状态推过去 —— 否则"主面板
           开着呼吸高亮、双击放大之后新的那一份不呼吸", 一处开一处关 (而放大窗口本来
           就是用来"看得更清楚"的)。
        */
        m_moeLoadDialog->setHighlightEnabled(m_moeHighlightCheck != nullptr
                                             && m_moeHighlightCheck->isChecked());
        m_moeLoadDialog->setLiveRoute(m_moeLiveRoute);
        m_moeLoadDialog->show();
        m_moeLoadDialog->raise();
        m_moeLoadDialog->activateWindow();
    });

    /*
       ================================================================
       [2026-10] 呼吸高亮的取数定时器 —— **默认不启动** (用户口径: 开关默认关)
       ================================================================
       为什么必须单独一条不取锁的路径: 上面那份快照走 worker + `m_agentMutex`, 而
       **AI 的整段决策都持着那把锁** (chessboard.cpp 的 aiThinkForAgentRaw) —— 所以
       "在锁上读"的路径在思考中会一直阻塞, 而思考中恰好是唯一想看实时路由的时刻。
       `liveMoeRoute()` 读的是 MoE 层里的**无锁**探针 (RL::MoERouteProbe), 所以这里可以
       按自己的节奏调, 既不阻塞界面也不影响搜索。

       节奏取 15 Hz (66 ms): 人眼对"呼吸"这种慢变化够用, 而每次只是几个原子读;
       控件自己还有 25 fps 的呼吸动画 (只在可见且有实时数据时跑, 见 MoeLoadView)。

       ⚠ **开关关着的时候这个定时器是停的** (见 applyMoeHighlight): 关掉一个功能就该
       连它的取数一起停, 而不是"取了不画" —— 后者会让"关了"这件事在 CPU 上仍然是真的,
       而且一旦哪天有人把画的那一段改回去, 它就**静默地又开始动了**。
    */
    m_moeLiveTimer = new QTimer(this);
    m_moeLiveTimer->setInterval(66);
    connect(m_moeLiveTimer, &QTimer::timeout, this, [this]() {
        ChessBoard::MoeLiveRoute live;
        ui->gameWidget->liveMoeRoute(live);
        m_moeLiveRoute = live;
        if (m_moeLoadView != nullptr) {
            m_moeLoadView->setLiveRoute(live);
        }
        if (m_moeLoadDialog != nullptr) {
            m_moeLoadDialog->setLiveRoute(live);
        }
    });
    /* 不在这里 start(): 默认关, 勾选框那边统一管 (唯一一个启停点) */

    /*
       开局就先取一份读数。否则这块控件要等到"第一手棋下完"或"切一次 agent"才有内容
       (自检的刷新点只有那几个), 而它显示的正是"这个 agent 的路由健康度" ——
       开局就空着会被读成"没有 MoE"(其实只是还没请求)。
       用 singleShot(0) 而不是直接调: 刷新走后台 worker, 让它晚于构造函数、
       在事件循环起来之后再启动, 免得在构造期就碰正在初始化的成员。
    */
    QTimer::singleShot(0, this, [this]() { requestSelfCheckPanelUpdate(false); });

    /*
       对弈参数: 局数与每手探索(预训练)步数。把这两个数字放到界面上是为了让
       "对弈结果"可解释 —— 局数太少结论会被单局偶然性翻转, 探索步数直接决定
       每一步的思考成本。

       上限 10000: 原来卡在 100。而 80 Elo ≈ 61.5% 得分率, 用 4~100 局去分辨它
       是在噪声里读结论 (与 bench_anchor 同一件事: 100 局时得分率的 95% 区间仍有
       ±10 个百分点)。需要"这个改动到底有没有变强"这种可证伪的结论时, 就把局数
       拉到几百以上; 对局过程中按钮会变成"停止对弈", 随时可以中止, 不会锁死界面。
       逐局明细会一局一行写进右侧列表 —— 上千局时那个列表会很长 (数据本身没问题,
       MatchStats 只累加计数 + 一行文本), 只是别指望一眼扫完。
    */
    ui->gamesSpin->setRange(1, 10000);
    ui->gamesSpin->setValue(4);
    ui->gamesSpin->setSuffix(QStringLiteral(" 局"));
    /* 上千局时按 1 递增太慢: 步进给 10 (仍然可以敲键盘直接输入精确值) */
    ui->gamesSpin->setSingleStep(10);
    ui->preTrainStepsSpin->setRange(0, 2000);
    ui->preTrainStepsSpin->setSingleStep(16);
    ui->preTrainStepsSpin->setValue(ui->gameWidget->getPreTrainSteps());
    ui->preTrainStepsSpin->setSuffix(QStringLiteral(" 步"));

    /* ---- 指标曲线与逐局明细 (右侧面板, 见 metricsview.h) ---- */
    setupMetricsPanel();

    /*
       ---- [2026-10 用户口径] "训练损失 / 行为克隆" 两个 tab ----
       必须**在** setupMetricsPanel 之后: 它把 .ui 里的 lossChart(+读数行) 搬进第一个
       tab 页, 而 setupMetricsPanel 刚把这两张图的标题/窗口/接线配好 (搬动的是配好的控件)。
    */
    setupChartTabs();
    /* BC 保真度采样 -> 那条曲线 (对局线程 emit, auto 连接排队投递到 GUI 线程) */
    QObject::connect(ui->gameWidget, &ChessBoard::bcFidelitySample, this,
                     &MainWindow::onBcFidelitySample);

    /* Agent选择 (与你对战的AI; 你执红, 它执黑) */
    connect(ui->agentComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onAgentSelected);

    /*
       ---- 沙漏上那一行"当前对战 agent" (2026-09 用户报障) ----
       报障: "黑方胜利后, 沙漏显示 agent 未加载"。原因是空闲时那一行显示的是指示器
       内部的 m_agent, 而按"开局"时 resetToIdle() 把它清空了 —— 于是明明选着 agent,
       沙漏上却写着 "(未选择 agent)"。现在把当前选中项喂给指示器 (选中即更新),
       空闲时它就显示真正要跟你下棋的那个 agent。
    */
    ui->thinkIndicator->setConfiguredAgent(ui->agentComboBox->currentText());

    /* 开局按钮 */
    connect(ui->resetBtn, &QPushButton::clicked,
            ui->gameWidget, &ChessBoard::reset);
    /* 开局也把"思考中"指示器清回初始外观 */
    connect(ui->resetBtn, &QPushButton::clicked,
            ui->thinkIndicator, &ThinkingIndicator::resetToIdle);

    /*
       "走子前先探索+预训练"开关 (仿 snakeAI 的决策流程: 先探索环境、用新鲜经验
       在线训练一次, 再做决策)。默认打开; 关掉就是原来的"直接决策"。
    */
    connect(ui->preTrainCheck, &QCheckBox::toggled,
            ui->gameWidget, &ChessBoard::setPreTrainEnabled);
    ui->gameWidget->setPreTrainEnabled(ui->preTrainCheck->isChecked());

    /* 探索步数实时同步给棋盘 (0 = 不探索) */
    connect(ui->preTrainStepsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            ui->gameWidget, &ChessBoard::setPreTrainSteps);

    /*
       ---- "自由走子"调试开关 (2026-09 用户要求) ----
       用户要"能故意输给 AI", 而规则过滤 (Chess::sample / ChessBoard::moveStone 里的
       isLegalMove) 恰好剔掉了"走后自家将被攻击"的着法 ⇒ 规则之内无法快速送死。
       打开这个勾选框后, 玩家那一侧的着法只校验**形状**, 不再要求"走后不被将"。
       默认关闭, 界面上带 "(调试)" 标注。
    */
    connect(ui->freeMoveCheck, &QCheckBox::toggled,
            ui->gameWidget, &ChessBoard::setFreeMoveEnabled);
    ui->gameWidget->setFreeMoveEnabled(ui->freeMoveCheck->isChecked());

    /* AI Self Play / Agent 对弈 按钮 (对弈进行中兼作"停止") */
    connect(ui->selfPlayBtn, &QPushButton::clicked,
            this, &MainWindow::onStartMatch);

    /* [2026-10 移除] 棋谱回放的三个连接 (recordcomboBox / prevBtn / nextBtn) 已删,
       理由见 mainwindow.h 里 private slots 顶部那段说明 */

    /* AI思考时间信号 */
    connect(ui->gameWidget, &ChessBoard::aiThinkFinished, this,
        [this](long long elapsedMs) {
            if (elapsedMs < 1000) {
                ui->timeLabel->setText(QString("AI思考时间: %1ms").arg(elapsedMs));
            } else {
                double sec = elapsedMs / 1000.0;
                ui->timeLabel->setText(QString("AI思考时间: %1s").arg(sec, 0, 'f', 3));
            }
        });

    /*
       ---- 思考过程可视化 ----
       AI 在后台线程思考, 思考期间棋盘拒绝落子 (这是对的), 但界面上原来没有任何
       反馈: 玩家看到的就是"棋子没动, 我也点不动", 分不清是"还在算"还是"卡死"。
       加入"走子前先探索+预训练"之后单步思考从毫秒级涨到秒级, 这个问题就很突出。

       这几条连线把 ChessBoard 从工作线程发来的 (队列投递, 所以槽都在 GUI 线程跑)
       阶段信号接到:
         thinkIndicator —— 沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时
         timeLabel      —— 思考过程中就跟着跳, 而不是等结束才出数字
    */
    connect(ui->gameWidget, &ChessBoard::aiThinkingStarted, this,
        [this](const QString &agentName, int exploreSteps) {
            QString stage = exploreSteps > 0
                ? QString("① 探索环境 + 预训练 (≤%1 步)").arg(exploreSteps)
                : QString("① 搜索 / 决策");
            ui->thinkIndicator->start(agentName, stage);
            ui->timeLabel->setText("AI思考时间: 0.00s (思考中)");
        });
    connect(ui->gameWidget, &ChessBoard::aiThinkingStage, this,
        [this](const QString &stage) {
            ui->thinkIndicator->setStage(stage);
        });
    /* 指示器每帧把实时耗时播出来, 让"AI思考时间"标签也一起跳 */
    connect(ui->thinkIndicator, &ThinkingIndicator::elapsedChanged, this,
        [this](long long ms) {
            if (!ui->thinkIndicator->isRunning()) {
                return;
            }
            ui->timeLabel->setText(
                QString("AI思考时间: %1s (思考中)").arg(ms / 1000.0, 0, 'f', 2));
        });
    connect(ui->gameWidget, &ChessBoard::aiThinkingStopped, this,
        [this]() {
            ui->thinkIndicator->stop();
            /*
               对弈时不要用指示器的总耗时覆盖"AI思考时间": 对弈是整场算一次
               "思考", 那个数字是整场时长; 逐手耗时由 aiThinkFinished 给。
            */
            if (m_matchRunning) {
                return;
            }
            /*
               停表后再把最终耗时写一遍。aiThinkFinished 与 aiThinkingStopped 是两条
               独立排队的事件, 中间那一小段时间里指示器的定时器还可能再跳一帧, 把标签
               覆写成 "…(思考中)"; 这里以指示器自己的最终值为准收尾。
            */
            const long long ms = ui->thinkIndicator->elapsedMs();
            if (ms > 0) {
                ui->timeLabel->setText(
                    QString("AI思考时间: %1s").arg(ms / 1000.0, 0, 'f', 3));
            }
        });

    QObject::connect(ui->gameWidget, &ChessBoard::matchStarted, this,
        [this](const QString &a, const QString &b, int games) {
            ui->matchResultLabel->setText(
                QString("对弈 %1 vs %2 · 共 %3 局 · 进行中").arg(a, b).arg(games));
            /*
               新的一场: 重建奖励曲线 (两条, 分别是 A 与 B), 并在明细列表里插一个
               分组标题。**损失曲线不清空** —— 训练是跨场持续的, 清掉就看不出趋势了。
            */
            resetMetricsForMatch(a, b);
            ui->gameListWidget->addItem(
                QString("—— 第 %1 场: %2 vs %3, %4 局 ——")
                    .arg(++m_metricsMatchNo).arg(a, b).arg(games));
            ui->gameListWidget->scrollToBottom();
        });
    connect(ui->gameWidget, &ChessBoard::matchGameFinished, this,
        [this](int no, int games, const QString &line) {
            m_matchLog += line + "\n";
            ui->matchResultLabel->setText(QString("对弈 %1/%2 局完成").arg(no).arg(games));
            ui->matchResultLabel->setToolTip(m_matchLog);
            /* 逐局明细: 一局一行, 结束时列表里就是完整战绩 (不再只放在 tooltip 里) */
            ui->gameListWidget->addItem(line.trimmed());
            ui->gameListWidget->scrollToBottom();
        });
    /* 实时比分: 每局结束后刷新 (以前只有整场结束才弹一个结果框) */
    connect(ui->gameWidget, &ChessBoard::matchScoreChanged, this,
        [this](const QString &scoreLine) {
            ui->scoreLabel->setText(QString("当前比分: %1").arg(scoreLine));
        });
    /*
       ---- [2026-10] 人机对弈: 每局开始时给"行为克隆"那条曲线建线 ----
       人机那条路没有 `matchStarted`(它只是"人对 AI", 不是"一场几局的比赛"), 只有
       `humanGameStarted`(玩家落下本局第一子)。缺了这一步的后果是**静默**的:
       `CurveChart::addPoint` 在序列不存在时直接 return ⇒ 人机里 BC 明明在训练
       (面板有 samples=/updates=), 曲线却一直空着。用户 2026-10 报的就是这个。
       建线判据与 Agent 对局那条路**共用** armBcChartSeries(见它的注释)。
    */
    connect(ui->gameWidget, &ChessBoard::humanGameStarted, this, [this]() {
        armBcChartSeries(ChessBoard::bcSupported(humanAiSide()),
                         ui->gameWidget->bcTeacherDepth(), /*humanMode=*/true);
    });
    /*
       ---- [2026-10] 价值评估曲线的一个点 (每局结束时一次) ----
       与 BC 那条一样是**跨场连续**的 (窗口本身跨局滚动), 所以不在 resetMetricsForMatch
       里清线 —— 它量的是"当前这个 critic 准不准", 清掉就只剩最后一局的点数了。
    */
    connect(ui->gameWidget, &ChessBoard::valueDiagSample, this,
            &MainWindow::onValueDiagSample);
    /* ---- 指标曲线采样 ---- */
    connect(ui->gameWidget, &ChessBoard::trainLossSample, this,
        [this](double loss, const QString &agent, int step) {
            (void)step;
            ui->lossChart->addPoint(lossSeriesFor(agent), loss);
            updateMetricsLabels();
        });
    connect(ui->gameWidget, &ChessBoard::gameRewardSample, this,
        [this](int gameNo, const QString &agentA, const QString &agentB,
               double rewardA, double rewardB) {
            (void)gameNo;
            (void)agentA;
            (void)agentB;
            if (m_rewardSeriesA >= 0) {
                ui->rewardChart->addPoint(m_rewardSeriesA, rewardA);
            }
            if (m_rewardSeriesB >= 0) {
                ui->rewardChart->addPoint(m_rewardSeriesB, rewardB);
            }
            updateMetricsLabels();
        });
    /*
        ---- 一局进行中的奖励进度 (每手一个点) ----
       只有上面那个"每局一个点"的话, 一局几百手、十几分钟里奖励曲线一动不动
       (用户反馈: "对弈时奖励曲线没有更新")。这里每手补一个点, 值是本局**到目前为止**
       累计到的环境奖励, 于是曲线随着对局往前走; 局末的 ±1 由上面那个信号补上最后一点
       (所以每局最后一个点会比倒数第二个"多出一个胜负")。
    */
    connect(ui->gameWidget, &ChessBoard::matchRewardProgress, this,
        [this](int gameNo, int ply, double rewardA, double rewardB) {
            (void)gameNo;
            (void)ply;
            if (m_rewardSeriesA >= 0) {
                ui->rewardChart->addPoint(m_rewardSeriesA, rewardA);
            }
            if (m_rewardSeriesB >= 0) {
                ui->rewardChart->addPoint(m_rewardSeriesB, rewardB);
            }
            updateMetricsLabels();
        });
    /* 本次"探索环境 + 预训练"到底做了什么 */
    connect(ui->gameWidget, &ChessBoard::aiExploreInfo, this,
        [this](const QString &info) {
            ui->exploreLabel->setText(QString("探索+预训练: %1").arg(info));
        });
    /*
       ---- 模型自检面板 ----
       时机刻意选在这里: 预训练做完 = 棋盘上又走了一步、训练又更新过一轮权重,
       于是"对局累计"那一类读数会跟着走。报告本身是只读的, 随时可以再刷 (见
       requestSelfCheckPanelUpdate 的说明)。
    */
    connect(ui->gameWidget, &ChessBoard::aiExploreInfo, this,
        [this](const QString &) { requestSelfCheckPanelUpdate(false); });

    /* [2026-10 移除] replayIndexChanged / replayModeExited 两个连接已删
       (ChessBoard 侧的同名信号一并移除), 理由见 mainwindow.h 的 private slots 说明 */

    /*
       ---- 人机对战终局**不再保存权重** (2026-09 用户口径) ----
       用户要求: "只有程序退出时再保存模型"。
       这里原来连了 sendResult -> onHumanGameFinished 做终局保存, 而那正是"黑方赢了
       之后卡住"最可疑的一环: 保存要拿 AI 决策用的同一把锁 (m_agentMutex), 而它的
       写盘量是 530 MB 级。既然口径改成"只在退出时保存", 这条连接与那个槽一并删掉 ——
       终局路径上不再有任何写盘。
       (唯一落盘点是 ChessBoard::shutdownSave, 见 MainWindow 析构。)
    */

    /* 启动加载完成信号: 启用界面并刷新对局列表 */
    connect(ui->gameWidget, &ChessBoard::startupComplete, this,
        [this]() {
            ui->resetBtn->setEnabled(true);
            ui->selfPlayBtn->setEnabled(true);
            ui->agentComboBox->setEnabled(true);
            ui->matchAComboBox->setEnabled(true);
            ui->matchBComboBox->setEnabled(true);
            ui->gamesSpin->setEnabled(true);
            ui->preTrainStepsSpin->setEnabled(true);
            ui->timeLabel->setText("AI思考时间: -");
            ui->exploreLabel->setText("探索+预训练: -");
            ui->matchResultLabel->setText("对弈结果: -");
            ui->thinkIndicator->resetToIdle();
            ui->gameWidget->setEnabled(true);
            /* 启动加载完成: 现在才有 agent 可以自检 (之前都是 nullptr) */
            requestSelfCheckPanelUpdate(false);
            /*
               BC 控件同理 (2026-10): 启动加载之前 `getAgentType()` 读得到类型, 但
               **实例还没建**(预加载要几十秒) —— 那段时间把按钮点亮会让用户按下去
               得到一个"要现建网"的 BC (几十秒的额外等待, 而且与他以为的状态不符)。
            */
            updateBcMatchControlsEnabled();
            /*
               ---- 把"有没有载入模型"顶到对局列表最上面 (2026-09, 用户报障) ----
               报障是"点击开局模型未载入": 查下来代码没错 (weights/ 是 gitignore 的运行期
               产物, 没存过就是空的), 但这句话原来只在日志与自检面板里 —— 用户按"开局"
               看到的是"AI 下得像随机", 没有任何人能看见的提示。
               放在**列表最上面**: 对局列表是开局后必然被看的那一块。
               细节放 tooltip (单行条目里塞换行会被压平成一团)。
            */
            {
                auto *item = new QListWidgetItem(
                    QStringLiteral("—— %1 ——")
                        .arg(QString::fromStdString(ui->gameWidget->weightLoadSummary())));
                item->setToolTip(QString::fromStdString(ui->gameWidget->weightLoadHint()));
                ui->gameListWidget->addItem(item);
            }
        });

    /* ---- 启动加载阶段: 禁用所有交互控件 ---- */
    ui->resetBtn->setEnabled(false);    ui->selfPlayBtn->setEnabled(false);
    ui->agentComboBox->setEnabled(false);
    ui->matchAComboBox->setEnabled(false);
    ui->matchBComboBox->setEnabled(false);
    ui->gamesSpin->setEnabled(false);
    ui->preTrainStepsSpin->setEnabled(false);
    updateBcMatchControlsEnabled();     /* BC 也一起置灰 (启动加载期间没有实例可用) */
    ui->timeLabel->setText("正在加载...");
    ui->exploreLabel->setText("探索+预训练: -");
    ui->matchResultLabel->setText("对弈结果: -");

    /* ---- 载入/保存模型权重时的"请稍候"弹窗 (里面是那个沙漏控件) ---- */
    /*
       ChessBoard 只负责报告状态 (busyStarted/busyMessage/busyFinished), 弹窗长什么样
       由界面决定 —— 见 src/busydialog.h。启动加载会读 7 组权重 (老格式是十进制文本,
       DQN+MCTS 一个文件 16 MB), 以前这段时间界面只有一行"正在加载...", 看起来像卡死。
    */
    m_busy = new BusyDialog(this);
    /*
       延迟显示 (300 ms): 小权重的读写只要几十毫秒, 每次都弹一下窗反而是干扰
       (用户要求"对弈结束后静默保存权重")。所以只有**确实慢**的操作才把沙漏亮出来:
         busyStarted -> 起 300 ms 单发定时器 -> 到点还在忙才 show
         busyFinished -> 停定时器 + 收起 (没亮过就什么都不做)
       启动加载是个例外: 构造函数里已经直接把它亮起来了(它是秒级以上的等待, 用户需要
       立刻看到"在启动"), 这里的定时器只是让它别被重复 show 打断。
    */
    m_busyDelay = new QTimer(this);
    m_busyDelay->setSingleShot(true);
    m_busyDelay->setInterval(300);
    connect(m_busyDelay, &QTimer::timeout, this, [this]() {
        if (m_busyPending) {
            m_busy->startBusy(m_busyTitle, m_busyMessage);
        }
    });
    connect(ui->gameWidget, &ChessBoard::busyStarted, this,
            [this](const QString &title, const QString &message) {
                m_busyPending = true;
                m_busyTitle = title;
                m_busyMessage = message;
                if (m_busy->isBusy()) {
                    /* 已经在等 (例如启动加载): 只更新文字, 不重新弹 */
                    m_busy->setMessage(message);
                } else {
                    m_busyDelay->start();
                }
            });
    connect(ui->gameWidget, &ChessBoard::busyMessage, this,
            [this](const QString &message) {
                m_busyMessage = message;
                m_busy->setMessage(message);
            });
    connect(ui->gameWidget, &ChessBoard::busyFinished, this, [this]() {
        m_busyPending = false;
        m_busyDelay->stop();
        m_busy->stopBusy();
    });

    /*
       ---- "呼吸高亮"的最后对齐: **唯一**一次把勾选框的状态应用到定时器与控件上 ----
       必须是构造函数末尾: 勾选框在 populateAgentComboBox() 里建 (第 227 行, 很早),
       而取数定时器要到上面 MoE 面板那一段才存在。中间任何一次调用都会被那个空指针
       守卫吃掉 —— 于是"接线看起来好了、其实没接", 正是本工程反复栽的那种静默失效。
       默认**关**: 定时器不 start, 控件侧的开关也是默认 false, 这一句只是把三处
       口径钉成同一个 (勾选框 / 控件 / 定时器)。
    */
    if (m_moeHighlightCheck != nullptr) {
        applyMoeHighlight(m_moeHighlightCheck->isChecked());
    }

    /* 在后台线程启动异步加载 (数据库 + AI模型权重) */
    ui->gameWidget->setEnabled(false);
    m_busy->startBusy(QStringLiteral("正在载入"),
                      QStringLiteral("读取棋局数据库与模型权重…"));
    m_loadThread = std::thread([this]() {
        ui->gameWidget->startupLoad();
    });
}

MainWindow::~MainWindow()
{
    /*
       先等后台线程结束, 再销毁 ui。线程函数体里访问了 ui->gameWidget (ChessBoard),
       而 delete ui 会连同它一起销毁 —— 顺序反了就是 use-after-free。
       不 detach 是这里能 join 的前提。
    */
    if (m_loadThread.joinable()) {
        m_loadThread.join();
    }
    if (m_selfPlayThread.joinable()) {
        /* 对弈可能还在跑: 先请求中止, 否则关窗会一直等到整场对弈打完 */
        ui->gameWidget->abortMatch();
        m_selfPlayThread.join();
    }
    /*
       存权重线程 (常驻) 也访问 ui->gameWidget, 所以必须先停掉再 delete ui。
    */
    {
        std::lock_guard<std::mutex> lk(m_saveMutex);
        m_saveStop = true;
    }
    m_saveCv.notify_all();
    if (m_saveThread.joinable()) {
        m_saveThread.join();
    }
    /*
       自检面板的 worker 同理: 它会调 getAgentSelfCheck() (要在 agent 锁上等),
       必须先停掉再 delete ui —— 否则 worker 醒来时 ui 已经没了。
    */
    {
        std::lock_guard<std::mutex> lk(m_selfCheckMutex);
        m_selfCheckStop = true;
    }
    m_selfCheckCv.notify_all();
    if (m_selfCheckThread.joinable()) {
        m_selfCheckThread.join();
    }
    /*
       ---- 退出保存前先停后台训练 (2026-09) ----
       原来这里的顺序是"先 shutdownSave(), 再 delete ui (那时才停训练线程)" ——
       于是退出保存与后台训练的一轮**同时**在动同一张网 (正是用户报的崩溃那类竞争),
       而且保存还要排队等训练写那 558 MB。停掉训练再存, 既没有竞争, 存下去的也正好是
       "跑完最后一轮"的权重。stopBackgroundTraining() 会等当前这一轮结束 (关窗的等待
       时间由它决定, 与 ~ChessBoard 里那次是同一个代价; 重复调用安全)。
    */
    ui->gameWidget->stopBackgroundTraining();
    /*
       ---- 唯一的落盘点: 退出时保存 (2026-09 用户口径) ----
       用户要求"只有程序退出时再保存模型"。顺序很重要: 先停后台训练 (上一行),
       再保存 —— 否则退出保存会与后台训练那一轮的收尾同步抢同一张网 (2026-09 用户报的
       那个 0xC0000374 堆损坏就是这么来的), 而且保存还要排队等它写 558 MB。
       这里用**同步**保存 (不是后台线程): 退出路径上本来就要等, 而且必须保证
       "写完再 delete ui"。代价是关窗会等几秒到十几秒 (PPO+MCTS 两份 266/264 MB,
       稀疏 MoE 三支各 3 x 146 MB), 这是用户明确要的行为。
       (原来这里还接着调了一次 shutdownSave() —— 它内部就是同一个动作, 会**存两遍**;
        已经合并成这一处。shutdownSave 现在只是它的一个转发, 保留给外部调用点。)
    */
    const int exitSaved = ui->gameWidget->saveAllInstantiatedAgentsOnExit();
    qInfo().noquote() << QStringLiteral("[weights] 退出保存完成: %1 个 agent").arg(exitSaved);
    delete ui;
}

/*
   ================================================================
   [2026-10 移除] 棋谱回放的 UI 与槽函数
   ================================================================
   这里原有 refreshGameList / onGameSelected / onReplayPrev / onReplayNext /
   onReplayIndexChanged / onReplayModeExited 六个函数, 连同 .ui 里的
   recordcomboBox（选择历史对局）、stepLabel、prevBtn/nextBtn（Previous/Next Step）。

   为什么删: 这条链路**结构性不可达**, 不是"没用"。
     * 写入端从未接线 —— `GameDatabase::startGame/recordMove/endGame` **全仓零调用点**
       (本工程自己的文档就记着: docs/analysis.md 第 357 行、docs/issues_review.md
        第 1000 行与第 2739 行"每步落库 -> 可回放 实际未接线, 需要设计决定");
     * 于是 `chess_games.db` 只有表结构、没有新增行; 列表里永远只有占位项;
       两个步进键开局 `setEnabled(false)`、只有选中一局之后才会启用 => **永远点不动**。
     * 全仓没有测试或验证脚本覆盖它们 (只出现在 mainwindow.{h,cpp,ui} 里)。

   **保留**: `GameDatabase`、`chess_games.db`、以及 ChessBoard 里那句
   `GameDatabase::instance().open(...)` —— 那一层是独立的东西, 将来真要接线
   "每步落库 -> 可回放"时不用重写。接线前需要先解决两个已记录的缺陷:
   跨线程用 SQLite（连接在加载线程建、查询在 GUI 线程用）、以及 `recordMove`
   每步一次自动提交（一局约 200 次 fsync, 无事务）。
*/

void MainWindow::populateAgentComboBox()
{
    /* 与你对战的AI: 默认 Alpha-Beta */
    fillAgentCombo(ui->agentComboBox, ChessBoard::AGENT_ALPHABETA);
    /*
       Agent 对弈的双方。默认 A=Alpha-Beta, B=EVAB: 两个都快 (每手 ~150 ms),
       而且正好是"纯搜索"对"学会评估的搜索", 是这套 agent 里最有意义的一组对照。
       注意 A/B 不是红黑 —— 每局交换先后手, 见 ChessBoard::matchAgents。
       (默认项按**类型**给; 以前写的是下标 6, 而列表一旦插入新 agent 就会静默指错。)
    */
    fillAgentCombo(ui->matchAComboBox, ChessBoard::AGENT_ALPHABETA);
    fillAgentCombo(ui->matchBComboBox, ChessBoard::AGENT_EVAB);

    /*
       ---- 对弈模式 (P0-a) ----
       三项与 ChessBoard::MatchMode 一一对应, 用 userData 存枚举值 (与 agent 下拉框
       同一手法: **不按下标**取, 以后插入新项也不会静默指错)。
       为什么必须让用户能选: 在它之前, "这一场学不学"没有任何界面表达 —— 而实测
       (test_match [2.7c]) 表明"预训步数 = 0"只关得住 PPO 那类 agent 的第一条学习
       路径, SAC+AZ 在 selectMove 里还有第二条 (learnFromSearch), 那个勾选框压根管不到。
       默认 = 训练对局, 与改动前的行为一致 (不改变老用户的既有读数口径)。
    */
    {
        QComboBox *c = ui->matchModeCombo;
        c->clear();
        c->addItem(ChessBoard::matchModeName(ChessBoard::MATCH_TRAIN),
                   static_cast<int>(ChessBoard::MATCH_TRAIN));
        c->addItem(ChessBoard::matchModeName(ChessBoard::MATCH_EVAL),
                   static_cast<int>(ChessBoard::MATCH_EVAL));
        c->addItem(ChessBoard::matchModeName(ChessBoard::MATCH_NO_LEARN),
                   static_cast<int>(ChessBoard::MATCH_NO_LEARN));
        c->setMaxVisibleItems(3 + 2);
        const int idx = c->findData(static_cast<int>(ChessBoard::MATCH_TRAIN));
        c->setCurrentIndex(idx >= 0 ? idx : 0);
        QObject::connect(c, QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, &MainWindow::onMatchModeSelected);
        /* 把当前模式同步给棋盘 (构造函数里 combo 已经填好了) */
        onMatchModeSelected(c->currentIndex());
    }

    /*
       ================================================================
       ---- P1: "对手的棋进训练数据"的开关 (2026-09) ----
       ================================================================
       用户口径: "对手的棋仍然不进训练数据 (exploreAndTrain 没有对手参数) —— 模式解决的
       是'能不能归因', 不是'能不能从对手身上学'"。

       为什么这两个控件**在代码里建**而不是写进 mainwindow.ui: .ui 是中英混排的大文件,
       而这两个控件的语义 (下面这一整段) 必须贴着控件走; 工具提示里要写清"默认关 +
       每次探索最多问几手 + 代价", 那些字写在 .ui 的 XML 里反而更难读。
       默认**关着**: 打开它会让价值/策略目标条件化于**当前这个对手** (同一份权重里混进
       对不同对手的数据时 V(s) 学的是平均值), 所以它适合"专门练一个对手", 不适合常开。
    */
    /*
       ================================================================
       [2026-10 布局修] 把追加到 matchModeRow 的那几组控件分到独立行
       ================================================================
       原来下面三段一共对 `ui->matchModeRow->addWidget(...)` 调了 6 次, 于是那一行里
       挤了**七个控件**: 模式标签 + 模式下拉框 + 对手入训 + 它的步数框 + 动态奖励 +
       每轮训练局数标签 + 它的框。

       实测代价 (UIA 读 BoundingRectangle): 模式下拉框被压到 **54 px** 宽, 而同一列的
       另外三个下拉框是 340 / 356 / 356 —— 中文模式名 ("只对弈不学习" 之类) 根本显示
       不全。用户报的"模式那一行非常拥挤、看不清文字"就是这一条。

       现在每一组各自成行, 插在 matchModeRow **之后**, 顺序与原来完全一致。
       仍然留在代码里建 (而不是搬回 .ui) 的理由见下面每一段自己的注释: 那几个控件的
       语义必须贴着控件走, 工具提示又长 —— 写进 .ui 的 XML 里反而更难读。
    */
    QHBoxLayout *rowOpp = nullptr;      /* 对手入训 + 它的步数 */
    QHBoxLayout *rowReward = nullptr;   /* 动态奖励 */
    QHBoxLayout *rowBg = nullptr;       /* 每轮训练局数 + 它的框 */
    QHBoxLayout *rowMoe = nullptr;      /* 呼吸高亮 (MoE 负载面板的实时显示开关) */
    {
        QWidget *host = ui->matchModeRow->parentWidget();
        QBoxLayout *vb = (host != nullptr) ? qobject_cast<QBoxLayout *>(host->layout())
                                           : nullptr;
        int at = (vb != nullptr) ? vb->indexOf(ui->matchModeRow) : -1;
        auto newRow = [&]() -> QHBoxLayout * {
            if (vb == nullptr || at < 0) {
                return nullptr;   /* 布局结构变了: 调用方退回 matchModeRow, 不静默丢控件 */
            }
            QHBoxLayout *r = new QHBoxLayout();
            vb->insertLayout(++at, r);
            return r;
        };
        rowOpp = newRow();
        rowReward = newRow();
        rowBg = newRow();
        rowMoe = newRow();
    }
    /* 控件落位: 有独立行就进独立行, 否则退回原来那一行 (不会丢) */
    auto place = [this](QHBoxLayout *row, QWidget *w) {
        if (row != nullptr) {
            row->addWidget(w);
        } else {
            ui->matchModeRow->addWidget(w);
        }
    };

    {
        QCheckBox *cb = new QCheckBox(QStringLiteral("对手入训"), this);
        cb->setObjectName(QStringLiteral("opponentRolloutCheck"));
        cb->setChecked(ui->gameWidget->isOpponentInRolloutEnabled());
        cb->setToolTip(QStringLiteral(
            "P1: 让**对手的棋**进入训练数据 (默认关闭)\n\n"
            "打开后: 学习方每手决策前的探索里, \"对手那一半\"不再由学习方自己的策略猜测,"
            " 而是去问**真实对手**在这个局面上会怎么走 —— 走的是它对局时同一条决策代码。\n"
            "于是学习方的样本落在\"被真对手应手之后的局面\"上 (这才是\"从对手身上学\")。\n\n"
            "三条边界 (都由实现强制):\n"
            "  1. 对手的着法**不记成学习方的样本** —— 自对弈时那一半是学习方自己采样的,"
            " 换成真对手就会变成\"off-policy 却打着 on-policy 标签\", 对 REINFORCE/PPO"
            " 是实打实的偏差; 所以它只推进局面。\n"
            "  2. 对手**被问到时不许学习** (SAC+AZ 的 learnFromSearch 那条路径会被关掉)。\n"
            "  3. 只在**对弈**里有对手 (人机对战那条路保持自对弈 —— 人没有策略可问)。\n\n"
            "⚠ 问一次对手 = 一次完整决策 (SAC+AZ 实测 2.3 s/手), 所以要给手数上限。\n"
            "⚠ 口径: 打开后价值/策略目标**条件化于这个对手**; 混着不同对手训练时 V(s)"
            " 学的是平均值 (对手特征没有进状态)。"));
        place(rowOpp, cb);

        QSpinBox *sp = new QSpinBox(this);
        sp->setObjectName(QStringLiteral("opponentRolloutSpin"));
        sp->setRange(0, 64);
        sp->setValue(ui->gameWidget->getOpponentRolloutPlies());
        sp->setToolTip(QStringLiteral(
            "P1: 每次探索最多问对手几手 (0 = 不问)\n\n"
            "默认 1 = 只问\"对手对学习方第一步的应手\" (能负担又有意义的那个点)。\n"
            "调大 = 探索的更多手由真对手产生, 代价是每多一手就多一次完整决策。"));
        place(rowOpp, sp);
        if (rowOpp != nullptr) {
            rowOpp->addStretch(1);   /* 左对齐 (与 .ui 里 freeMoveRow 同一个做法) */
        }

        QObject::connect(cb, &QCheckBox::toggled, this,
                         &MainWindow::onOpponentRolloutToggled);
        QObject::connect(sp, QOverload<int>::of(&QSpinBox::valueChanged), this,
                         &MainWindow::onOpponentRolloutPliesChanged);
    }

    /*
       ---- [2026-09 奖励方法开关] "动态奖励" (默认不勾 = 保留旧奖励) ----
       用户口径: "保留旧的奖励方法" —— 所以默认**不勾**, 勾上才切到 `rewardShape = 3`
       (按局面评估在"吃子 / 杀将"之间分配一个固定预算)。
       为什么也放进 matchModeRow 程序化建: 与上面两个控件同一套理由 (语义要贴着控件走),
       而且三条边界必须写进工具提示 —— 否则它会被读成"变强的旋钮"。
    */
    {
        QCheckBox *rcb = new QCheckBox(QStringLiteral("动态奖励"), this);
        rcb->setObjectName(QStringLiteral("dynamicRewardCheck"));
        rcb->setChecked(ui->gameWidget->isDynamicRewardEnabled());
        rcb->setToolTip(QStringLiteral(
            "奖励方法开关 (默认**不勾** = 保留旧奖励)\n\n"
            "不勾 = 旧奖励 (rewardShape=0):\n"
            "  即时 = 材质 x0.1 + 每步代价(-0.001); 终局 = 引擎真值 ±1。\n"
            "  这一支的代码一个字没动, 历史读数与它逐位对得上。\n\n"
            "勾上 = 动态奖励 (rewardShape=3):\n"
            "  按**局面评估** e (剩余价值 / 剩余个数 / 相对子力差, 三因子等权) 在\n"
            "  \"吃子\"与\"杀将\"之间分配一个**固定预算**: 材质倍数 1+0.5(1-e),\n"
            "  终局倍数 1+0.5e, 两者之和恒为 2.5; 两条都**永不归零**。\n\n"
            "⚠ 三条边界 (实测, 不是推测):\n"
            "  1. **它不是棋力旋钮**: 四条训练臂 (各 60 局自对弈 + 16 局/锚点) 的得分率\n"
            "     全部落在噪声里 (docs/sacmoetb_pos_reward_2026_09.md §9)。\n"
            "  2. **\"杀将\"那一半的作用面很窄**: 终局倍率只作用在 terminalReward() 上, 而\n"
            "     自对弈里一局大多走到手数上限被截断 (截断那一步 done 仍是 false) ——\n"
            "     实测 60 局自对弈里\"分胜负的终局样本\"只占 ~0.14% (53/37504)。\n"
            "     [2026-09 更正] 但它**不是死的**: 把 mateBoost 从 0 开到 10, 两次存下来的\n"
            "     权重不同 (221FB956… vs D419EF76…); 而人机对弈那条终局通道补上之后\n"
            "     (人把 AI 将死也会交给学习器), 这个旋钮在人机对弈里也会被碰到。\n"
            "  3. **它改变训练出来的权重**: 两套奖励训出的不是同一个东西, 别共用权重文件。\n\n"
            "要做对照实验请用 bench 工具 (固定开局集 + 配对 + 区间):\n"
            "  bench_sacaz_vs_ab --backbone=moe-mlp --reward-shape=3 --warmup-games=60 ...\n"
            "  bench_sacmoetb_train --reward-shape=3 ..."));
        place(rowReward, rcb);
        if (rowReward != nullptr) {
            rowReward->addStretch(1);
        }
        QObject::connect(rcb, &QCheckBox::toggled, this,
                         &MainWindow::onDynamicRewardToggled);
    }

    /*
       ---- [2026-09] "每轮训练局数" (先训练 N 局再对弈) ----
       后台训练本来就在跑 (startBackgroundTraining), 但一轮的规模写死在
       BG_TRAIN_EPISODES x BG_TRAIN_MAX_MOVES。这个框让"先练多少再打"变成可调的。
       用户口径: **先用 MCTS 当对手练, 才谈得上对 AB 有胜率** —— 实测支持它:
       对 AB level=1 时 SAC 四臂一局都没赢 (0/23/57 等), 得分率只能在"输多输少"之间动;
       换 MCTS 锚点才有胜/负 (8 胜 / 2 负 / 30 和)。
       见 docs/sacmoetb_independent_classes_2026_09.md §9。
    */
    {
        QLabel *lb = new QLabel(QStringLiteral("每轮训练局数"), this);
        place(rowBg, lb);
        QSpinBox *tb = new QSpinBox(this);
        tb->setObjectName(QStringLiteral("bgTrainEpisodesSpin"));
        tb->setRange(1, 200);
        tb->setValue(ui->gameWidget->getBackgroundTrainEpisodes());
        tb->setToolTip(QStringLiteral(
            "后台自对弈训练**每轮**跑几局 (默认 1)。\n\n"
            "一轮 = 克隆权重 -> 独立棋盘上自对弈 N 局 -> 训好写回 -> 同步回主 agent。\n"
            "调大 = 每轮练得更多; 但\"同步回主 agent\"那一步的等待也更长\n"
            "(关窗时会等这一轮跑完, 见 stopBackgroundTraining 的说明)。\n\n"
            "为什么要有它 (用户口径: \"先使用 MCTS agent 对弈, 才有可能在与 AB agent 对弈有胜率\"):\n"
            "  对 AB level=1 的实测里, SAC 四臂**一局都没赢** (0/23/57 等), 得分率只能在\n"
            "  \"输多输少\"之间动 —— 那个锚点量不出胜率; 换成 MCTS 锚点才有胜/负\n"
            "  (8 胜 / 2 负 / 30 和)。所以**练棋与量棋都优先拿 MCTS 当对手**。\n"
            "  做法: 这一栏选 MCTS 当对手 + 下拉框选\"训练对局\"模式 (+ 需要时勾\"对手入训\")。"));
        place(rowBg, tb);
        if (rowBg != nullptr) {
            rowBg->addStretch(1);
        }
        QObject::connect(tb, QOverload<int>::of(&QSpinBox::valueChanged), this,
                         &MainWindow::onBgTrainEpisodesChanged);
    }

    /*
       ================================================================
       [2026-10 用户口径] "呼吸高亮"开关 —— **默认关**
       ================================================================
       控制的是一整块**实时**显示 (MoE 负载面板上"此刻哪个专家在工作"):
       打开后, 正在干活的那几根柱子会被暖色呼吸高亮, 标题多一个前向序号 `#N`,
       末行多一句"此刻 3:34%,7:20%"。
       关着 = **与加这个功能之前完全一样**, 而且连取数都停 (定时器不跑, 不读探针)。

       为什么默认关 (三条, 都是"这一版的选择"而不是定论):
         1. 这是一个 25 fps 的动画面板。旁边三块静态读数 (loss / reward / 比分) 里
            突然多一块一直在呼吸的东西, 默认打开会让"看盘"变成"看动画";
         2. 它有真实的代价: 每次 forward 多约 25 ns 的原子发布 + 每 66 ms 一次无锁读。
            前者可以忽略, 但"默认路径逐位不变 / 默认不额外做功"是本工程的硬约束;
         3. 关着的时候它是**可以断言**的 (`hl=off`), 于是"默认关"这条约定本身也进了
            回归 (见 tools/verify_moe_load_view.ps1 -Live 的第一段断言)。

       放在**中间这一列**(而不是贴着 MoE 面板): 右边那一列 (metricsPanel) 已经被
       "比分 / 负载面板 / loss / reward / 按钮行 / 逐局明细"填满到只剩 ~9 px (UIA 实测),
       再塞一行会把曲线或对局列表压矮 —— 中间这一列有 verticalSpacer, 加一行不挤任何东西。
    */
    {
        QCheckBox *hcb = new QCheckBox(QStringLiteral("呼吸高亮"), this);
        hcb->setObjectName(QStringLiteral("moeHighlightCheck"));
        hcb->setChecked(false);          /* 默认关 */
        hcb->setToolTip(QStringLiteral(
            "MoE 负载面板上的**实时**显示：此刻哪个专家在工作（默认**不勾**）\n\n"
            "勾上之后，面板上正在干活的那几根柱子会被暖色**呼吸高亮**（强度 = 最近约 1 秒的\n"
            "前向活跃度 × 呼吸相位，越忙越亮），柱顶的亮块 = 最近**一次**前向选中的那 top-k 个，\n"
            "标题里多一个前向序号 #N（一直在跳就说明读数在走）。\n\n"
            "柱子的**高度与底色仍然只表示累计份额**（蓝=正常、橙红=超额），高亮是叠在上面的一层，\n"
            "两个口径不混。\n\n"
            "为什么它是**无锁**读数：AI 的整段决策都持着 agent 锁，走锁的读数在思考中会一直阻塞，\n"
            "而思考中恰好是唯一想看它的时刻 —— 所以它读的是 MoE 层里的原子快照。\n"
            "同理，累计份额只在选中 agent / 每手棋的探索+预训练之后刷新一次：没走预训练时柱子会停在\n"
            "旧读数（此时面板画的是整列光柱，并写明\"累计计数还没刷新\"）。\n\n"
            "⚠ 关着 = 连取数都停（不读探针、不跑动画）；开/关都只影响**显示**，\n"
            "不参与任何训练或决策。"));
        place(rowMoe, hcb);
        if (rowMoe != nullptr) {
            rowMoe->addStretch(1);
        }
        QObject::connect(hcb, &QCheckBox::toggled, this, [this](bool on) {
            applyMoeHighlight(on);
            qInfo().noquote() << QStringLiteral("[MoE] 呼吸高亮 = %1")
                                     .arg(on ? QStringLiteral("开") : QStringLiteral("关"));
        });
        m_moeHighlightCheck = hcb;
        /*
           这里**不调** applyMoeHighlight: 本函数可能跑在 `m_moeLiveTimer` 创建之前
           (构造顺序), 那样调用会被那个空指针守卫吃掉, 看起来"接线好了"其实没接。
           对齐放在 MainWindow 构造函数**末尾**那一次统一调用 (见那里的说明)。
        */
    }

    /*
       ================================================================
       ---- [2026-10 用户口径] 对局中行为克隆: 一个**下拉框**选老师 ----
       ================================================================
       用户口径 (原话): **"行为克隆勾选框改成下拉框选择要克隆的 abagent，与将要对弈的
       对方 agent 或者人类棋手无关，训练的时候参考下拉框选择的 abagent 的决策进行行为
       克隆训练"**。于是这里是一个下拉框 (第一项 = 关), 而不是勾选框:

         * **老师由它选**, 与这一场的对手是谁完全无关 —— 对手是 AB 某一档、是别的
           agent、还是人机里的**人**, 都照常克隆;
         * 每一项对应一档 Alpha-Beta 的深度 (L1/L2/L3/(深度=4)), 深度是**唯一**的老师
           参数, 所以 itemData 直接存深度 (0 = 关);
         * 文本里刻意带 ASCII 记号 (`off` / `depth=1`…): verify_bc_ui.ps1 要用 UIA
           按名字选中某一项, 而那个脚本必须 ASCII-only (无 BOM 的 .ps1 会被按 ANSI
           解码, 中文会把脚本解析坏)。与自检面板里的 "MaxVio / Loss-Free" 同一个做法。

       为什么放在中间这一列、且用代码插入: 与上面四行 (对手入训 / 动态奖励 / 每轮训练
       局数 / 呼吸高亮) 同一套理由; 而"能不能选"这件事**跟着 A/B 两个下拉框走**, 放在
       它们附近才看得懂 (见 updateBcMatchControlsEnabled 的说明)。
    */
    QHBoxLayout *rowBc = nullptr;
    {
        QWidget *host = ui->matchModeRow->parentWidget();
        QBoxLayout *vb = (host != nullptr) ? qobject_cast<QBoxLayout *>(host->layout())
                                           : nullptr;
        int at = (vb != nullptr) ? vb->indexOf(ui->matchModeRow) : -1;
        if (vb != nullptr && at >= 0) {
            /* 插在 matchModeRow 之后、上面那四行**之前**: 它是"这一场怎么学"的开关,
               与"模式 / 对手入训 / 动态奖励"是一组; 呼吸高亮是显示开关, 排在后面。 */
            rowBc = new QHBoxLayout();
            vb->insertLayout(at + 1, rowBc);
        }
    }
    {
        QLabel *lab = new QLabel(QStringLiteral("行为克隆训练:"), this);
        lab->setObjectName(QStringLiteral("bcTeacherLabel"));
        QComboBox *cb = new QComboBox(this);
        cb->setObjectName(QStringLiteral("bcTeacherCombo"));
        /*
           文本 = "关" + 四档 Alpha-Beta。名字用 ChessBoard::bcTeacherName (唯一来源),
           所以这里选的档位与 A/B 下拉框里的叫法逐字一致 (不会出现"下拉框写 L2、
           报告里印深3"那种看起来像功能坏了的错位)。
           深度与档位一一对应, itemData 存的就是深度。
        */
        cb->addItem(QStringLiteral("关 (off)"), 0);
        for (int d = 1; d <= 4; d++) {
            cb->addItem(QStringLiteral("克隆 %1 (depth=%2)")
                            .arg(ChessBoard::bcTeacherName(d)).arg(d), d);
        }
        cb->setCurrentIndex(0);          /* 默认关 (与"动态奖励 / 对手入训"同一个口径) */
        cb->setToolTip(QStringLiteral(
            "对局中行为克隆 (BC): 选一个 **Alpha-Beta 老师**, 学生的每个局面都由它当标签\n\n"
            "选好之后 (默认**关**):\n"
            "  * **与对手是谁无关** —— 对手是 AB 的某一档、是别的 agent、还是人机里的\n"
            "    人类棋手, 都照常克隆;\n"
            "  * 只在这一局/这一场、且只在\"训练对局\"模式下生效 (评估/只对弈里权重必须不变);\n"
            "  * **学生每走一手** => 那个局面 + \"下拉框这一档 AB 在该局面上搜出来的一手\"\n"
            "    = 一条监督样本 => 立刻更新一次**策略头**;\n"
            "  * 损失 = 该局面**完整合法集**上的掩码交叉熵 (dL/dz = π − t, 非法列恒为 0),\n"
            "    与在线训练同一个归一化口径, 也与命令行 train_bc 同一套口径;\n"
            "  * 它是**额外**的: 学生自己的在线训练照常进行。\n\n"
            "三条边界 (读数必须与它们一起看):\n"
            "  1. 这是**模仿**: 上限就是所选那一档 AB (L1/L2/L3 一档比一档弱);\n"
            "  2. 它**不碰 critic** (PPO 的 BC 路径整段不含 critic 前向/反向),\n"
            "     SAC 独立口径下 q1/q2 逐字节不变;\n"
            "  3. 跑完**只改内存里的权重** —— 唯一的落盘点是退出程序时。\n\n"
            "读数: 对局期间每若干次更新会在下面的\"模型自检\"面板里打一行\n"
            "      (`samples= / updates= / CE=`), 整场结束后面板里有一份完整摘要;\n"
            "      曲线在\"行为克隆 (BC)\"那个 tab 里, 双击可放大。\n\n"
            "⚠ 只有场上**至少一侧能做 BC 学生** (PPO 两支 / SAC 三支) 时才选得动;\n"
            "  两边都是纯搜索 (AB 各档 / MCTS) 时这个下拉框会置灰并把原因写在这里。"));
        if (rowBc != nullptr) {
            rowBc->addWidget(lab);
            rowBc->addWidget(cb);
            /*
               ---- [2026-10] 软目标复选框 (用户问题: "不直接使用 onehot 通过 abagent 计算
                    概率分布再进行行为克隆是否会更好?") ----
               放在**老师下拉框旁边**: 它改的是"老师给什么" (一只手 vs 一个分布),
               不是另开一个功能。默认**不勾** —— 实测两个指标方向相反 (可比口径的 KL 更好,
               而硬口径的 top-1 略低), 只压小 train−留出差 (见文档 §9), 所以是可选档。
            */
            QCheckBox *soft = new QCheckBox(QStringLiteral("软目标 (多深度一致)"), this);
            soft->setObjectName(QStringLiteral("bcSoftCheck"));
            soft->setChecked(false);
            soft->setToolTip(QStringLiteral(
                "行为克隆的**目标**形态 (改的是老师给什么, 不是另开一个功能)\n\n"
                "不勾 (默认, one-hot): 老师只给最深那一层搜出来的**一手**,\n"
                "  目标熵 H(t)=0, 损失 = 掩码 NLL (CE 就是 NLL)。\n"
                "勾上 (软目标): 深度 1..D **各投一票**, 票数/D 当概率 ⇒ 一个分布,\n"
                "  目标熵 H(t)>0, 损失仍是合法集上的交叉熵 —— 它的**下界变成 H(t)**\n"
                "  (CE = H(t) + KL(t‖π)), 所以看 CE 时必须与报告里那行 targetH 一起看。\n\n"
                "为什么是\"多深度一致\"而不是\"根分值 softmax\": AB 的根循环是窗口写法,\n"
                "非最优孩子返回的是**界**而不是精确分值 ⇒ 拿根分值做 softmax 等于在搜索\n"
                "裁剪的产物上克隆。多深度投票只用\"各深度的最优手\"这一个稳定的量。\n\n"
                "实测 (4000 局面 / 深度 3 / 8 epoch / **4 个种子**成对比较, 见文档 §9):\n"
                "两个指标**方向相反** —— 可比口径 **留出 KL (CE − H(t)) 4/4 都更低**\n"
                "(2.379→1.866), 而硬口径 **留出 top-1 略低** (38.59%→36.78%, 3/4 个种子,\n"
                "符号不一致 ⇒ 只能算「没变好」); train−留出 差 4/4 都更小 (0.427→0.280),\n"
                "代价约 +12~14% 打标签时间 —— 所以它是**可选档**, 默认关。\n\n"
                "⚠ 老师选\"关\"时它没有任何作用 (没有采样就没有目标)。"));
            QObject::connect(soft, &QCheckBox::toggled, this, &MainWindow::onBcSoftToggled);
            rowBc->addWidget(soft);
            rowBc->addStretch(1);        /* 左对齐 (与其它几行同一做法) */
            m_bcSoftCheck = soft;
        } else {
            ui->matchModeRow->addWidget(lab);
            ui->matchModeRow->addWidget(cb);
        }
        QObject::connect(cb, QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, &MainWindow::onBcTeacherChanged);
        m_bcTeacherCombo = cb;
    }
}

/*
 * humanAiSide - "人机那条路的学生"是哪一支
 *
 * 就是"对战AI"下拉框 (index 0) 选中的那一支 —— 人机对弈里 AI 执黑、玩家执红, 所以
 * **AI 那一支才是学生** (玩家没有策略头可克隆)。两处需要它: 曲线建线判据、下拉框启用判据;
 * 各写一遍的话会出现"下拉框亮着但曲线不建线"这种自相矛盾的状态。
 */
ChessBoard::AgentType MainWindow::humanAiSide() const
{
    if (ui->agentComboBox != nullptr && ui->agentComboBox->currentIndex() >= 0) {
        return static_cast<ChessBoard::AgentType>(
            ui->agentComboBox->itemData(ui->agentComboBox->currentIndex()).toInt());
    }
    return ChessBoard::AGENT_ALPHABETA;      /* 拿不到就当作"没有学生" */
}

/*
 * onBcTeacherChanged - "行为克隆训练"下拉框 (0 = 关, 1..4 = AB 深度)
 *
 * 只做两件事: 把选择交给棋盘, 然后在面板上留一行"这一次选的是谁"。
 * **不在这里判断能不能用**: 下拉框在不能用的时候是**置灰**的 (见
 * updateBcMatchControlsEnabled), 所以"选得动"就等于"可以用"; 而棋盘那一侧还会再判一次
 * (模式 / 场上有没有学生), 判不通过时它会在面板上说明原因 (见 matchAgents 里那段)。
 *
 * [2026-10] 另外: 选完之后**立刻按新选择重新布置曲线**(建线或清线) —— 曲线是"这一场/这一局
 * 会不会有读数"的预告, 让用户不必等到开局才发现"原来没建线"。
 */
void MainWindow::onBcTeacherChanged(int index)
{
    if (ui->gameWidget == nullptr || m_bcTeacherCombo == nullptr) {
        return;
    }
    const int depth = (index >= 0) ? m_bcTeacherCombo->itemData(index).toInt() : 0;
    ui->gameWidget->setBcTeacherDepth(depth);
    /*
       重新布置 BC 曲线 (选回"关"就清掉那两条线): 不清的话图上还挂着上一次的线, 而这一场
       根本没在克隆 —— 那正是"两条空平线被读成保真度 0"的同一类误读 (见 armBcChartSeries)。
       走的是与开局同一份实现, 所以"选了老师但场上没有学生"这句话在两种情形下一致。
    */
    if (!m_matchRunning && !ui->gameWidget->isMatchRunning()) {
        armBcChartSeries(ChessBoard::bcHasStudent(m_matchTypeA, m_matchTypeB)
                             || ChessBoard::bcSupported(humanAiSide()),
                         depth, /*humanMode=*/!ChessBoard::bcHasStudent(m_matchTypeA, m_matchTypeB));
    }
    qInfo().noquote() << QStringLiteral("[BC] 对局中行为克隆 = %1")
                             .arg(depth > 0 ? ChessBoard::bcTeacherName(depth)
                                            : QStringLiteral("关"));
    if (ui->selfCheckView != nullptr) {
        ui->selfCheckView->moveCursor(QTextCursor::End);
        ui->selfCheckView->insertPlainText(
            QStringLiteral("[BC] 对局中行为克隆 = %1 (老师 = 下拉框选的 AB, 与对手无关)\n")
                .arg(depth > 0 ? ChessBoard::bcTeacherName(depth)
                               : QStringLiteral("关")));
        ui->selfCheckView->moveCursor(QTextCursor::End);
    }
}

/*
 * onBcSoftToggled - "软目标 (多深度一致)" 复选框 (2026-10)
 *
 * 分工与 onBcTeacherChanged 一样: 只把勾选交给棋盘 + 在面板留一行, **不在这里判断
 * 能不能用** (老师下拉框停在"关"时它就是没作用, 报告里也没有读数 —— 判据在棋盘那侧)。
 * 面板上的那一行是必要的: 这个勾选不改变任何控件的可见状态, 不留字的话用户没法确认
 * "到底生效了没有"; 而它与老师档位一起决定**报告里 CE 的可比性** (软目标下 CE 的下界
 * 是 H(t)), 所以必须写清"生效中的目标形态"。
 */
void MainWindow::onBcSoftToggled(bool on)
{
    if (ui->gameWidget != nullptr) {
        ui->gameWidget->setBcSoftTarget(on);
    }
    const QString what = on ? QStringLiteral("软目标 (深度 1..D 各投一票, 分布; H(t)>0)")
                            : QStringLiteral("one-hot (只取最深一层的一手; H(t)=0)");
    qInfo().noquote() << QStringLiteral("[BC] 目标形态 = %1").arg(what);
    if (ui->selfCheckView != nullptr) {
        ui->selfCheckView->moveCursor(QTextCursor::End);
        ui->selfCheckView->insertPlainText(
            QStringLiteral("[BC] 目标形态 = %1%2\n")
                .arg(what,
                     on ? QStringLiteral(" —— CE 的下界变成 H(t), 与报告里的 targetH 一起看")
                        : QString()));
        ui->selfCheckView->moveCursor(QTextCursor::End);
    }
}

/*
 * updateBcMatchControlsEnabled - "那个下拉框此刻该不该亮"的**唯一**实现
 *
 * 判据有两层, 缺一层就会出现一种"看起来能用其实没用"的状态:
 *   1. **场上得有学生**: 至少一侧能做 BC (PPO 两支 / SAC 三支) —— 老师由下拉框给,
 *      所以"对手是不是 AB"**不参与**判定 (2026-10 口径)。判据用库侧的
 *      `ChessBoard::bcHasStudent` (界面不重写一遍)。人机那条路的学生是"对战AI"那一支,
 *      所以它也参与判定 —— 否则"人机想克隆"会被 A/B 的选择连坐置灰。
 *   2. **时机**: 对局进行中不让改 (改了之后这一场算不算? 没有定义) ——
 *      与 selfPlayBtn / preTrainCheck 在开赛后置灰同一个口径。
 *
 * 置灰时**必须**把原因写进 tooltip: "下拉框是灰的"既可能是"场上没有学生",
 * 也可能是"正在对弈" —— 两者的下一步动作完全不同。
 */
void MainWindow::updateBcMatchControlsEnabled()
{
    if (m_bcTeacherCombo == nullptr || ui->matchAComboBox == nullptr
        || ui->matchBComboBox == nullptr) {
        return;
    }
    const int ia = ui->matchAComboBox->currentIndex();
    const int ib = ui->matchBComboBox->currentIndex();
    if (ia < 0 || ib < 0) {
        m_bcTeacherCombo->setEnabled(false);
        return;
    }
    const ChessBoard::AgentType a = static_cast<ChessBoard::AgentType>(
        ui->matchAComboBox->itemData(ia).toInt());
    const ChessBoard::AgentType b = static_cast<ChessBoard::AgentType>(
        ui->matchBComboBox->itemData(ib).toInt());
    /* 人机那条路的学生 = "对战AI" 那一支 */
    ChessBoard::AgentType aiSide = ChessBoard::AGENT_ALPHABETA;
    if (ui->agentComboBox != nullptr && ui->agentComboBox->currentIndex() >= 0) {
        aiSide = static_cast<ChessBoard::AgentType>(
            ui->agentComboBox->itemData(ui->agentComboBox->currentIndex()).toInt());
    }

    const bool matchStudent = ChessBoard::bcHasStudent(a, b);
    const bool humanStudent = ChessBoard::bcSupported(aiSide);
    const bool anyStudent = matchStudent || humanStudent;
    const bool running = m_matchRunning || ui->gameWidget->isMatchRunning();
    m_bcTeacherCombo->setEnabled(anyStudent && !running);
    /*
       软目标复选框与下拉框**同进同出** (2026-10): 它是"老师给什么"的一种, 场上一旦
       没有学生、或者对弈已经开始 (这一场定下来了), 改它同样没有定义。
       ⚠ 只在**启用**时才跟着走: 对弈结束后 `anyStudent` 仍然为真, 于是用户的勾选
       会被原样保留 —— 用 setChecked 同步"当前值"会把勾选**清掉**, 那是另一类 bug。
    */
    if (m_bcSoftCheck != nullptr) {
        m_bcSoftCheck->setEnabled(anyStudent && !running);
    }

    if (!anyStudent) {
        m_bcTeacherCombo->setToolTip(QStringLiteral(
            "这一场没有能做行为克隆的**学生**。\n"
            "老师由本下拉框选 (与对手无关), 但**场上至少要有一支**有策略头的 agent:\n"
            "PPO+MCTS / PPO+MCTS(MLP专家) / SAC+AZ / SAC+AZ-MoE / SAC+AZ-MoE-MLP 这五支。\n"
            "现在 A/B 与\"对战AI\"选的都是纯搜索 agent (Alpha-Beta 各档 / MCTS), 它们没有\n"
            "可克隆的策略头 —— 把其中一处换成那五支里的一支即可。"));
    } else if (running) {
        m_bcTeacherCombo->setToolTip(QStringLiteral(
            "对弈进行中: 这个选择**这一场已经定下来了** (中途改它没有定义),\n"
            "等这一场结束再切。"));
    } else {
        m_bcTeacherCombo->setToolTip(QStringLiteral(
            "对局中行为克隆 (BC): 选一个 **Alpha-Beta 老师**, 学生的每个局面都由它当标签\n\n"
            "选好之后 (默认**关**):\n"
            "  * **与对手是谁无关** —— 对手是 AB 某一档、是别的 agent、还是人机里的\n"
            "    人类棋手, 都照常克隆;\n"
            "  * 只在这一场、且只在\"训练对局\"模式下生效;\n"
            "  * **学生每走一手** => 那个局面 + 老师在该局面搜出来的一手 = 一条样本\n"
            "     => 立刻更新一次**策略头**;\n"
            "  * 损失 = 完整合法集上的掩码交叉熵, 与在线训练 / 命令行 train_bc 同一套口径;\n"
            "  * 它是**额外**的: 学生自己的在线训练照常进行。\n\n"
            "读数: 对局期间在下面的\"模型自检\"面板里逐次打一行 (samples= / updates= / CE=),\n"
            "整场结束后面板里有一份完整摘要 (含\"只改了内存里的权重\"那句提醒);\n"
            "曲线在\"行为克隆 (BC)\"那个 tab 里, 双击可放大。"));
    }
}

void MainWindow::applyMoeHighlight(bool on)
{
    if (m_moeLoadView != nullptr) {
        m_moeLoadView->setHighlightEnabled(on);
    }
    if (m_moeLoadDialog != nullptr) {
        m_moeLoadDialog->setHighlightEnabled(on);
    }
    if (m_moeLiveTimer == nullptr) {
        return;              /* 构造期早于定时器创建: 由后面那次调用收尾 */
    }
    if (on) {
        if (!m_moeLiveTimer->isActive()) {
            m_moeLiveTimer->start();
        }
    } else {
        m_moeLiveTimer->stop();
        m_moeLiveRoute = ChessBoard::MoeLiveRoute();   /* 关掉就别留旧读数 */
    }
}

/*
 * onOpponentRolloutToggled / onOpponentRolloutPliesChanged - P1 的两个旋钮
 *
 * 只做一件事: 把值写进 ChessBoard。**不**在这里判"现在对弈跑没跑" —— 那两个 setter
 * 只在下一手决策时被读, 中途改也只是让"接下来的探索"换口径 (与对弈模式的快照口径
 * 不同: 模式在开场就锁进报告, 而这一项是每手现读的, 所以报告不能声称全场一致 ——
 * 报告里只写"本场有没有开启过", 见 MatchStats 的说明)。
 */
void MainWindow::onOpponentRolloutToggled(bool on)
{
    ui->gameWidget->setOpponentInRolloutEnabled(on);
    qInfo().noquote() << QStringLiteral("[P1] 对手入训 = %1 (探索里%2问真实对手)")
                             .arg(on ? QStringLiteral("开") : QStringLiteral("关"))
                             .arg(on ? QString() : QStringLiteral("不"));
}

void MainWindow::onOpponentRolloutPliesChanged(int n)
{
    ui->gameWidget->setOpponentRolloutPlies(n);
    qInfo().noquote() << QStringLiteral("[P1] 每次探索最多问对手 %1 手").arg(n);
}

/*
 * [2026-09 奖励方法开关] "动态奖励" 勾选框回调。
 * 口径与三条边界见 chessboard.h 的 setDynamicRewardEnabled 与那个复选框的工具提示。
 * 注意它**不会**重建任何 agent —— 四个旋钮是普通成员, setter 直接写到现有实例上
 * (对比: moeDense 那种建网期参数必须走构造参数)。
 */
void MainWindow::onBgTrainEpisodesChanged(int n)
{
    ui->gameWidget->setBackgroundTrainRound(n, ui->gameWidget->getBackgroundTrainMaxMoves());
    qInfo().noquote() << QStringLiteral("[train] 后台训练每轮 %1 局").arg(n);
}

void MainWindow::onDynamicRewardToggled(bool on)
{
    ui->gameWidget->setDynamicRewardEnabled(on);
    qInfo().noquote() << QStringLiteral("[reward] 动态奖励 = %1 (%2)")
                             .arg(on ? QStringLiteral("开") : QStringLiteral("关"))
                             .arg(on ? QStringLiteral("rewardShape=3: 按局面在吃子/杀将间分配固定预算; "
                                                      "杀将那一半只在\"分胜负的终局样本\"上生效 "
                                                      "(自对弈里约 0.14%, 人机对弈里由终局通道补上)")
                                     : QStringLiteral("rewardShape=0: 旧奖励, 与历史读数一致"));
}

/*
 * onMatchModeSelected - 对弈模式下拉框回调 (P0-a)
 *
 * 只做一件事: 把模式写进 ChessBoard。**模式是"开场时快照"的** (见
 * ChessBoard::matchAgents 里的 modeName), 所以对弈进行中改这一项不会改变
 * 正在跑的那一场 —— 报告里的模式永远是这一场实际用的那个。
 */
void MainWindow::onMatchModeSelected(int index)
{
    if (index < 0 || ui->matchModeCombo == nullptr) {
        return;
    }
    const int raw = ui->matchModeCombo->itemData(index).toInt();
    ChessBoard::MatchMode m = ChessBoard::MATCH_TRAIN;
    switch (raw) {
    case ChessBoard::MATCH_EVAL:     m = ChessBoard::MATCH_EVAL; break;
    case ChessBoard::MATCH_NO_LEARN: m = ChessBoard::MATCH_NO_LEARN; break;
    default:                         m = ChessBoard::MATCH_TRAIN; break;
    }
    ui->gameWidget->setMatchMode(m);
    /*
       [P0-b 收尾] 把"这个模式还会顺手关掉什么"写进日志: 非训练模式下**后台训练整体
       停摆** (它每轮会把新权重同步回主 agent, 而主 agent 正是对局/人机在用的那份),
       而且评估模式下 A/B 同类型时 B 会走**开场权重快照**。以前这两件事都是静默的:
       用户看到的是"我明明关了学习, 权重却还在变"(见 test_match [2.7j]/[2.7k])。
    */
    QString extra;
    if (m == ChessBoard::MATCH_TRAIN) {
        extra = QStringLiteral(" (后台训练照常; 权重本来就会变)");
    } else {
        extra = QStringLiteral(" (后台训练本模式下停摆: 不再把权重同步回主 agent)");
        if (m == ChessBoard::MATCH_EVAL) {
            extra += QStringLiteral("; A/B 同类型时 B 方走开场权重快照");
        }
    }
    qInfo().noquote() << QStringLiteral("[match] 对弈模式 = %1%2")
                             .arg(ChessBoard::matchModeName(m), extra);
}

void MainWindow::onAgentSelected(int index)
{
    if (index < 0) return;
    ChessBoard::AgentType type =
        static_cast<ChessBoard::AgentType>(
            ui->agentComboBox->itemData(index).toInt());
    ui->gameWidget->setAgentType(type);

    QString name = ui->agentComboBox->currentText();
    qDebug("AI Agent switched to: %s", qPrintable(name));

    /*
       换 agent 立刻反映到沙漏上 (用户报障里的"选择黑方对弈 agent 失效": 那时应手线程
       已经没了, 换谁都不会有反应; 现在线程还在, 而且这一行当场就能看到选中项生效)。
    */
    ui->thinkIndicator->setConfiguredAgent(name);

    /* 换了 agent 就换一份自检报告 (不支持的 agent 显示"没有自检项") */
    requestSelfCheckPanelUpdate(false);

    /*
       BC 的按钮跟着"当前 agent 能不能做"走 (2026-10): 换成纯搜索 agent 时它是灰的,
       并把**原因**写在提示里 (否则"按钮是灰的"读不出是"这一支不支持"还是"正在忙")。
    */
    updateBcMatchControlsEnabled();
}

/* ================================================================
 *  onStartMatch - Agent 对 Agent 对弈按钮回调
 *
 *  同一个按钮兼作"停止": 对弈进行中再点一次就请求中止 (在每一手之间检查,
 *  最迟一手之内生效)。对弈在后台线程里跑, 所以界面不会被卡住 —— 每一步的
 *  进度通过 ChessBoard 的 matchStarted / matchGameFinished 信号回到 GUI 线程。
 *
 *  方法学: 每局交换先后手, 胜负按 A/B 记。中国象棋先手优势很大, 固定谁执红
 *  的话最后只是在测"谁执红"。见 ChessBoard::matchAgents。
 * ================================================================ */
void MainWindow::onStartMatch()
{
    ChessBoard *board = ui->gameWidget;

    /* 正在对弈 -> 这次点击是"停止" */
    if (board->isMatchRunning()) {
        board->abortMatch();
        ui->selfPlayBtn->setEnabled(false);
        ui->selfPlayBtn->setText("正在停止...");
        return;
    }

    const int idxA = ui->matchAComboBox->currentIndex();
    const int idxB = ui->matchBComboBox->currentIndex();
    if (idxA < 0 || idxB < 0) {
        return;
    }
    const ChessBoard::AgentType typeA = static_cast<ChessBoard::AgentType>(
        ui->matchAComboBox->itemData(idxA).toInt());
    const ChessBoard::AgentType typeB = static_cast<ChessBoard::AgentType>(
        ui->matchBComboBox->itemData(idxB).toInt());
    const int games = ui->gamesSpin->value();

    /* [④] 记下类型: 奖励曲线的口径标签要靠它 (见 mainwindow.h 的 m_matchTypeA/B) */
    m_matchTypeA = typeA;
    m_matchTypeB = typeB;

    /* 界面上的设置同步给棋盘 (对弈线程会读这两个开关) */
    board->setPreTrainSteps(ui->preTrainStepsSpin->value());
    board->setPreTrainEnabled(ui->preTrainCheck->isChecked());

    m_matchRunning = true;
    m_matchLog.clear();
    ui->selfPlayBtn->setText("停止对弈");
    /*
       对弈期间禁用 BC (2026-10): BC 要独占主 agent (整段持有 agent 锁), 而对弈的决策
       路径正在用它 —— 两者不能同时在跑 (不然 BC 会让对弈停在一半等锁, 看起来像卡死)。
    */
    updateBcMatchControlsEnabled();
    ui->matchResultLabel->setToolTip(QString());
    ui->matchResultLabel->setText(
        QString("对弈 %1 vs %2 · 共 %3 局 · 准备中")
            .arg(agentLongName(typeA), agentLongName(typeB)).arg(games));

    /* 在后台线程里跑整场对弈, 避免阻塞 UI */
    if (m_selfPlayThread.joinable()) {
        m_selfPlayThread.join();      /* 上一次若已结束, 回收它 */
    }
    m_selfPlayThread = std::thread([this, board, typeA, typeB, games]() {
        ChessBoard::MatchStats st = board->matchAgents(typeA, typeB, games);
        const QString summary = st.summary();
        const QString detail = st.detail();

        /* 回到主线程处理结果 */
        QMetaObject::invokeMethod(this, [this, summary, detail, typeA, typeB]() {
            m_matchRunning = false;
            ui->selfPlayBtn->setEnabled(true);
            ui->selfPlayBtn->setText("开始对弈");
            updateBcMatchControlsEnabled();   /* 对弈结束: 勾选框重新可用 */
            /*
               对局结束后刷一次自检面板: 这一场的"对局中行为克隆"摘要就存在 ChessBoard 里
               (bcReportText), 而面板的刷新时机原来是"选 agent / 每手预训练 / 启动完成"——
               对局结束时正好是"这一场克隆了多少"最该被看到的一刻, 少了这一次刷新,
               用户要等下一手棋才看得到 (而这一场可能已经结束了)。
            */
            requestSelfCheckPanelUpdate(false);
            ui->matchResultLabel->setText(summary);
            ui->matchResultLabel->setToolTip(detail);
            ui->scoreLabel->setText(QStringLiteral("最终比分: %1").arg(summary));

            /*
               逐局明细已经在 matchGameFinished 里一局一行地写进列表了, 这里只补
               两条收尾信息。原来是一个**模态**结果框 (要手动关掉, 而且关掉之后
               明细就只剩 tooltip 里那一大段文字) —— 现在明细常驻在右侧列表里,
               可以逐行对照, 也可以事后回看。
            */
            ui->gameListWidget->addItem(QStringLiteral("—— 本场结束: %1 ——").arg(summary));
            const QStringList detailLines =
                detail.split(QChar('\n'), Qt::SkipEmptyParts);
            for (int i = detailLines.size() - 1; i >= 0; --i) {
                if (detailLines[i].startsWith(QStringLiteral("思考耗时"))) {
                    ui->gameListWidget->addItem(detailLines[i].trimmed());
                    break;
                }
            }
            ui->gameListWidget->scrollToBottom();

            /*
               ---- 对局结束**不再保存权重** (2026-09 用户口径) ----
               用户要求: "只有程序退出时再保存模型"。
               原来这里会调 saveWeightsAfterMatch(...) —— 而保存要么把 530 MB 的权重
               (PPO+MCTS: 266 MB actor + 264 MB critic) 在 `m_agentMutex` 锁内序列化写盘,
               要么排队等这把锁; 而 AI 决策/自检也拿同一把锁。实测: 保存 2.4 s 期间一次
               决策从 6140 ms 被挡到 8411 ms; 长成一局的保存会直接把"黑方该走棋"卡住。
               现在唯一落盘点是**程序退出**(ChessBoard::shutdownSave), 于是对局路径上
               再也不会有写盘 —— 界面流畅优先, 代价是"未正常退出则这一场的训练成果丢失"。
            */
            ui->gameListWidget->addItem(
                QStringLiteral("—— 本场结束 (权重将在**退出程序时**统一保存) ——"));

        }, Qt::QueuedConnection);
    });
}

/* ================================================================
 *  setupMetricsPanel - 右侧"曲线 + 逐局明细"面板的初始化
 *
 *  三条曲线:
 *    lossChart   : 训练损失。**每个 agent 一条**(名字进图例) —— 不同 agent 的
 *                  损失尺度完全不同 (SAC+AZ 是 critic 的 MSE, EVAB 是价值网蒸馏
 *                  的 MAE, DQN 是平方 TD 误差), 混在一条线上没有可比性。
 *    rewardChart : 环境奖励。每场对弈两条 (A / B), **每手一个点** —— 值是"本局
 *                  到目前为止"的累计, 局末再补一个含终局 ±1 的点 (每局最后一个点
 *                  因此会比它前面那个多出胜负那一份)。一局可能几百手、跑十几分钟,
 *                  只在局末给一个点的话整局都看不到变化。
 *                  纵轴含 0 线, 所以"正贡献/负贡献"一眼能看出来。
 *
 *  曲线只保留最近 2000 个点 (setWindow): 后台训练会一直往里塞样本, 不设上限就是
 *  内存泄漏。横轴是"第几个样本", 不是时间 —— 训练是事件驱动的, 时间轴没有意义。
 * ================================================================ */
void MainWindow::setupMetricsPanel()
{
    ui->lossChart->setTitle(QStringLiteral("训练损失 (每完成一次在线训练一个点)"));
    ui->lossChart->setValueSuffix(QString());
    ui->lossChart->setWindow(2000);

    /*
       [④] 奖励曲线的**口径** (2026-09): 现在取 agent **自己学的那一份** —— 材质 x0.1
       + 每步代价 + 终局 (SAC 塑形开着时终局是 ±(1+败方材质/3.5))。纯搜索 agent
       (Alpha-Beta / MCTS / EVAB) 没有学习口径, 仍旧画引擎口径 (材质 x1 + 终局 ±1)。
       两个口径差 10 倍 (docs/sac_learn_reward_2026_09.md §1.1), 所以
         * 曲线名上标出各自的口径 (见 resetMetricsForMatch);
         * 标题里写清规则 —— 曲线上的比例现在**就是**学习信号的比例。
    */
    ui->rewardChart->setTitle(
        QStringLiteral("环境奖励 (每手累计, 走子方视角) · 学习口径: 材质x0.1+每步代价+终局; "
                       "标[引擎口径]的是纯搜索 agent (材质x1)"));
    ui->rewardChart->setValueSuffix(QString());
    ui->rewardChart->setWindow(2000);

    connect(ui->clearMetricsBtn, &QPushButton::clicked, this, [this]() {
        /*
           "清空曲线"要连**曲线本身**一起清 (否则 m_lossSeries 清空了、图上那几条线
           还在, 下次同一个 agent 上报损失时会按名字新建一条 —— 于是同名线并排出现,
           和 resetMetricsForMatch 那个坑是同一类)。
        */
        ui->lossChart->removeAllSeries();
        ui->rewardChart->removeAllSeries();
        if (m_bcChart != nullptr) {
            m_bcChart->removeAllSeries();       /* "清空曲线"也要清掉 BC 那一张 */
            m_bcLastFidTop1 = -1.0;
            m_bcLastFidPTeacher = -1.0;
            m_bcLastFidCe = -1.0;
            m_bcLastFidWindow = 0;
        }
        /*
           [2026-10 用户报] 价值评估那一张也必须被"清空曲线"清掉 —— 第一版漏了它, 于是
           "点了清空、别的图空了、它还留着", 用户看到的是"这个按钮对那条曲线没用"。
           同时把读数行的状态一起复位: 曲线清了而读数行还在报"第 5 局, pairs 138"的话,
           那个读数就成了**假的**(它描述的是已经被清掉的那批点)。
           ⚠ 探针本身**不清** (m_valueWinV/Z 那些在 ChessBoard 里, 而且它是"最近 2000 手"
             的滚动窗口): 这里清的是**画出来的曲线**, 下一个局末的点会接着画 —— 语义与
             "清空损失曲线"一致 (清的是显示, 不是把已经发生的训练忘掉)。
        */
        if (m_valueChart != nullptr) {
            m_valueChart->removeAllSeries();
            m_valueChart->addSeries(QStringLiteral("EV (z = 真实胜负 ±1/0; 0 = 常数预测水平)"),
                                    kSeriesColors[3 % kSeriesColorCount]);
            m_valueChart->addSeries(QStringLiteral("ρ (z = 引擎口径折扣回报; 0 = 无相关)"),
                                    kSeriesColors[4 % kSeriesColorCount]);
        }
        m_valueCalibErr = 0.0;
        m_valueSampleCount = 0;
        m_valueZVar = 0.0;
        m_valueZVarEng = 0.0;
        m_valueEvEng = 0.0;
        m_valueGameNo = 0;
        m_lossSeries.clear();
        m_rewardSeriesA = -1;
        m_rewardSeriesB = -1;
        ui->gameListWidget->clear();
        ui->scoreLabel->setText(QStringLiteral("当前比分: -"));
        updateMetricsLabels();
    });
    connect(ui->exportMetricsBtn, &QPushButton::clicked,
            this, &MainWindow::exportMetricsCsv);
    /* 单独的"导出损失曲线"按钮 (2026-09 用户要求): 只写损失那一段, 见 exportLossCsv */
    connect(ui->exportLossBtn, &QPushButton::clicked,
            this, &MainWindow::exportLossCsv);
    /* "全部模型自检": 一次性快照, 下一手棋的自检刷新会切回当前 agent (见它的注释) */
    connect(ui->selfCheckAllBtn, &QPushButton::clicked,
            this, &MainWindow::showAllAgentsSelfCheck);

    ui->scoreLabel->setText(QStringLiteral("当前比分: -"));
    ui->gameListWidget->addItem(QStringLiteral("(还没有对局)"));
    updateMetricsLabels();

    /* ---- 双击曲线 -> 放大到独立窗口 (用户要求的手动放大) ---- */
    connect(ui->lossChart, &CurveChart::doubleClicked, this, [this]() {
        openLargeChart(ui->lossChart, QStringLiteral("训练损失 (放大)"));
    });
    connect(ui->rewardChart, &CurveChart::doubleClicked, this, [this]() {
        openLargeChart(ui->rewardChart,
                       QStringLiteral("环境奖励 (每手累计, 局末含 ±1) — 放大"));
    });
    /*
       ---- BC 那条曲线的双击放大**不在这里接** ----
       这一节 (setupMetricsPanel) 跑在 setupChartTabs() **之前**, 此刻 m_bcChart 还是
       nullptr, 在这里 connect 等于什么都没接 —— 而且它是**静默**的: 没有编译错、
       没有运行期警告, 现象只是"双击行为曲线没反应"。2026-10 用户报的正是这个。
       接线放在创建它的地方: setupChartTabs() 末尾。
    */
}

/* ============================================================================
 *  ---- [2026-10] 把"训练损失"与"行为克隆"放进两个 tab (用户口径) ----
 * ============================================================================
 * 用户口径: "在 loss 曲线窗口增加一个 tab 显示"。
 *
 * 三件事:
 *   1. 把 .ui 里的 lossChart + lossValueLabel **搬**进第一个 tab 页 (从原布局里
 *      removeWidget, 再 addTab —— Qt 的标准做法, 控件本身还是原来那两个, 于是
 *      导出/放大/画点这些接线一行都不用改);
 *   2. 新建第二个 tab 页: BC 保真度曲线 + 它自己的读数行;
 *   3. 把 tab 控件插回**原布局里 lossChart 原来的位置** —— 于是右侧那一列的排布
 *      (比分 / MoE / 曲线 / 奖励曲线 / 按钮 / 对局列表) 与改动前完全一致, 只是原来
 *      "训练损失"占的那一格现在是两个 tab。
 *
 * 为什么 BC 曲线是**两条同量纲的线** (一致率% 与 P(老师)%): CurveChart 只有一根纵轴,
 * 混量纲的线画在一起就是"两个口径混着比大小"(本工程反复记过的坑)。CE 因此不进曲线,
 * 它进图下的读数行 (与"损失/奖励"那两行读数同一个做法: 曲线给趋势, 读数给确切的数)。
 */
void MainWindow::setupChartTabs()
{
    if (ui->lossChart == nullptr || ui->lossValueLabel == nullptr || ui->metricsPanel == nullptr) {
        return;
    }
    QBoxLayout *lay = qobject_cast<QBoxLayout *>(ui->metricsPanel->layout());
    if (lay == nullptr) {
        return;      /* 布局结构变了: 不搬了 (宁可没有 tab, 也不要把图搬丢) */
    }
    const int at = lay->indexOf(ui->lossChart);
    if (at < 0) {
        return;
    }

    QTabWidget *tabs = new QTabWidget(ui->metricsPanel);
    tabs->setObjectName(QStringLiteral("chartTabs"));

    /* ---- tab 1: 训练损失 (原来那两个控件, 只是换了个父亲) ---- */
    QWidget *pageLoss = new QWidget(tabs);
    pageLoss->setObjectName(QStringLiteral("chartTabLoss"));
    auto *lossLay = new QVBoxLayout(pageLoss);
    lossLay->setContentsMargins(0, 0, 0, 0);
    lossLay->setSpacing(2);
    lay->removeWidget(ui->lossChart);
    lay->removeWidget(ui->lossValueLabel);
    lossLay->addWidget(ui->lossChart, 1);
    lossLay->addWidget(ui->lossValueLabel);
    /*
       tab 文字里刻意带一个 ASCII 记号 ((loss) / (BC)): verify_bc_ui.ps1 要在两个 tab
       之间切换并断言控件可见性, 而那个脚本必须 ASCII-only (无 BOM 的 .ps1 会被按 ANSI
       解码)。与自检面板里的 "MaxVio / Loss-Free" 同一个做法。
    */
    tabs->addTab(pageLoss, QStringLiteral("训练损失 (loss)"));

    /* ---- tab 2: 行为克隆 (新建的曲线 + 读数行) ---- */
    QWidget *pageBc = new QWidget(tabs);
    pageBc->setObjectName(QStringLiteral("chartTabBc"));
    auto *bcLay = new QVBoxLayout(pageBc);
    bcLay->setContentsMargins(0, 0, 0, 0);
    bcLay->setSpacing(2);
    m_bcChart = new CurveChart(pageBc);
    m_bcChart->setObjectName(QStringLiteral("bcChart"));
    /*
       标题把"怎么读这张图"写进去: 这是一条**代理指标**曲线 —— 它上升说明"更像老师",
       不是"更强" (docs/training_optimization.md §7.10 的负面结果就写在标题里, 免得
       曲线被单独引用)。
    */
    m_bcChart->setTitle(QStringLiteral(
        "行为克隆保真度 (每 4 次 actor 更新一个点, 最近 64 条样本窗口): "
        "一致率 = 策略头选中的着法 == AB 老师那一手 · **代理指标, 不是棋力**"));
    m_bcChart->setValueSuffix(QStringLiteral(" %"));
    m_bcChart->setWindow(2000);
    m_bcValueLabel = new QLabel(pageBc);
    m_bcValueLabel->setObjectName(QStringLiteral("bcValueLabel"));
    m_bcValueLabel->setWordWrap(true);
    m_bcValueLabel->setText(QStringLiteral("行为克隆: 未启用 (在 A/B 面板勾选\"行为克隆训练\")"));
    bcLay->addWidget(m_bcChart, 1);
    bcLay->addWidget(m_bcValueLabel);
    tabs->addTab(pageBc, QStringLiteral("行为克隆 (BC)"));

    /* ---- tab 3 [2026-10]: 价值评估 (critic 的 EV / 校准) ----
       用户口径: "在奖励窗口增加一个 tab 显示价值评估曲线"。
       为什么它必须与损失曲线分开一张图 (而不是画在同一条损失曲线上): 量纲与含义都不同 ——
       损失是 critic 的 MSE (而且目标是**自举**的), 这里的 EV 是"比'永远预测均值'好多少",
       无量纲、可以为负 (<0 = 还不如常数预测, 本工程实测过 −0.0293)。混在一根纵轴上
       就是本工程反复记过的"两个口径混着比大小"。
       ⚠ EV 可以**为负**: CurveChart 的纵轴会自动把 0 线包进来 (recomputeRange), 于是
       "0 = 不如常数预测"这条参考线天然在图里 —— 这正是读这条曲线唯一需要的参照。 */
    QWidget *pageValue = new QWidget(tabs);
    pageValue->setObjectName(QStringLiteral("chartTabValue"));
    auto *valueLay = new QVBoxLayout(pageValue);
    valueLay->setContentsMargins(0, 0, 0, 0);
    valueLay->setSpacing(2);
    m_valueChart = new CurveChart(pageValue);
    m_valueChart->setObjectName(QStringLiteral("valueChart"));
    /*
       标题写清"怎么读": 0 = 与常数预测同水平, 负 = 还不如常数 (符号/尺度错了),
       而且它是**跨对局滚动窗口**上的量 (量的是"现在准不准", 不是全程平均)。
    */
    m_valueChart->setTitle(QStringLiteral(
        "价值评估: critic 的 V(s) vs 两种 z (每局结束时算一次, 最近 2000 手窗口) · "
        "线1 = EV(z=真实胜负); 线2 = **ρ**(z=引擎口径折扣回报, 尺度无关) · "
        "**0 = 没有信息, 1 = 完美, < 0 = 反着**"));
    m_valueChart->setValueSuffix(QString());
    m_valueChart->setWindow(2000);
    /*
       两条线 (口径写在线名里, 免得事后分不清哪条是什么):
         0: **EV**(z = 真实胜负 ±1/0) —— "V 能不能预测胜负"。EV 是尺度敏感的, 所以它只在
            "同一个尺度的 z" 下才有意义 (z=±1 与 V 同尺度是这个口径的**前提**, 不是巧合)。
         1: **ρ**(z = 引擎口径折扣回报, γ=0.99) —— "V 与'子力+胜负'的走向是否同向"。
            ρ 是**尺度无关**的, 所以 V 学的是学习口径 (材质×0.1) 而 z 是引擎口径 (×1) 也
            能读 —— 这正是用户报"与 MCTS 对弈数值 < 0"暴露出来的那个问题的解:
            (a) 折扣回报必须用**引擎口径**逐手累加 (与对手是谁无关, 不混口径);
            (b) EV 在跨口径下会被尺度差吃满 (实测 −24.3) ⇒ 曲线上画 ρ, EV_eng 放读数行。
       两条线都在"0 = 没有信息"这个意义上可读 (EV: 等价常数预测; ρ: 无相关)。
    */
    m_valueChart->addSeries(QStringLiteral("EV (z = 真实胜负 ±1/0; 0 = 常数预测水平)"),
                            kSeriesColors[3 % kSeriesColorCount]);
    m_valueChart->addSeries(QStringLiteral("ρ (z = 引擎口径折扣回报; 0 = 无相关)"),
                            kSeriesColors[4 % kSeriesColorCount]);
    m_valueValueLabel = new QLabel(pageValue);
    m_valueValueLabel->setObjectName(QStringLiteral("valueValueLabel"));
    m_valueValueLabel->setWordWrap(true);
    m_valueValueLabel->setText(QStringLiteral(
        "价值评估: 还没有读数 (需要走完一局: 每手的 V(s) 要与该局真实胜负对照)"));
    valueLay->addWidget(m_valueChart, 1);
    valueLay->addWidget(m_valueValueLabel);
    tabs->addTab(pageValue, QStringLiteral("价值评估 (value)"));

    /* ---- 插回原位置 ---- */
    lay->insertWidget(at, tabs);

    /*
       ---- [2026-10] BC 曲线的"双击放大"接线**必须在这里** ----
       它原来写在 setupMetricsPanel() 里 (那一节先跑, 那会儿 m_bcChart 还是 nullptr),
       于是 connect 被 `if (m_bcChart != nullptr)` 静默跳过: 编译过、跑起来没警告,
       现象只是"双击行为曲线没反应"(用户 2026-10 报的)。凡是在别处创建的控件,
       接线就放在**创建它的同一个函数**里 —— 这条比"记住调用顺序"可靠。
       放大窗口与另外两张同一个入口 (openLargeChart 内部按 source 缓存窗口,
       所以三张图各有一个放大窗, 互不干扰)。
       extraReadout: 把 CE 一起带过去 —— 它不在曲线上 (百分比 vs 交叉熵, 不同量纲),
       只在读数行里, 而放大窗口的读数行是**从曲线数据重新格式化**出来的, 不问一句就
       会少掉 CE。问的是一个回调, 所以窗口开着时 CE 会跟着对局更新。
    */
    connect(m_bcChart, &CurveChart::doubleClicked, this, [this]() {
        openLargeChart(m_bcChart, QStringLiteral("行为克隆保真度 (放大)"),
                       [this]() { return bcFidelityCeTail(); });
    });
    /*
       [2026-10] 价值评估那条曲线同样能双击放大 (与另外三张同一个入口): 它的读数行里
       有校准误差/样本数/方差, 那些**不在曲线上**, 所以同样用 extraReadout 回调带过去 ——
       否则放大窗口一开就少三个数 (这个坑在 BC 那张图上已经踩过一次)。
    */
    connect(m_valueChart, &CurveChart::doubleClicked, this, [this]() {
        openLargeChart(m_valueChart, QStringLiteral("价值评估 (放大)"),
                       [this]() { return valueDiagTail(); });
    });
}

/*
   "不在曲线上的那一段"读数 (BC 的 CE 与窗口大小) —— 单独抽出来是为了让**源控件那行
   标签**与**放大窗口里的读数行**用同一份格式化: 两处各写一遍迟早会分叉, 而"放大窗口
   少一个数"这种分叉在界面上几乎看不出来。没采过样时返回空串 (那时不该出现 CE)。
*/
QString MainWindow::bcFidelityCeTail() const
{
    if (m_bcLastFidTop1 < 0.0) {
        return QString();
    }
    return QStringLiteral(" | 最近 CE %1 (窗口 %2 条)")
        .arg(m_bcLastFidCe, 0, 'f', 4)
        .arg(m_bcLastFidWindow);
}

/*
   "不在价值曲线上"的那一段读数 (校准误差 / 样本对数 / Var(z)) —— 与 BC 的 CE 同一个
   理由: 它们的量纲与 EV 不同 (校准误差 ∈ [0,2], 样本数是计数), 混在同一根纵轴上就是
   "两个口径混着比大小"。源标签与放大窗口共用这一份, 免得两处各写一遍而分叉。
   还没出过点时返回空串 (那时它不该出现)。
*/
QString MainWindow::valueDiagTail() const
{
    /*
       "不在曲线上的数" —— 而且**必须包含"为什么没有点"**: 读数行是常驻控件, 而面板里
       那行说明会被对局结束时的整段刷新冲掉 (用户实测"几轮都没有曲线"时, 界面上什么都
       看不到, 原因就在这)。三种原因写清楚, 下一步动作完全不同:
         * pairs < 32        -> 再走几局 (只采到有 V 头那一方的着手: PPO 两支)
         * Var(z_胜负) = 0   -> 被采样的手结果符号全一样 (全和棋 / 一路输) ⇒ EV 无定义
         * Var(z_引擎回报)=0 -> 连引擎口径折扣回报都一样 (几乎不会发生)
    */
    const bool needMore = (m_valueSampleCount < 32);
    QString t = QStringLiteral(" | EV_eng %1 · 校准误差 %2 · Var(z): 胜负 %3 / 引擎回报 %4"
                               " · 第 %5 局 · pairs %6%7")
                    .arg(std::isnan(m_valueEvEng) ? QStringLiteral("n/a")
                                                  : QString::number(m_valueEvEng, 'f', 3))
                    .arg(m_valueCalibErr, 0, 'f', 3)
                    .arg(m_valueZVar, 0, 'f', 3)
                    .arg(m_valueZVarEng, 0, 'f', 3)
                    .arg(m_valueGameNo)
                    .arg(m_valueSampleCount)
                    .arg(needMore ? QStringLiteral(" (需 >= 32)") : QString());
    if (m_valueZVar <= 1e-9) {
        t += QStringLiteral(" · 胜负口径无方差: 被采样的着手结果符号全一样 (全和棋/一路输)");
    }
    return t;
}

/*
   双击曲线 -> 弹一个 900x560 的独立窗口 (可缩放、可拖到别的屏幕), 内容与源控件
   实时同步 (CurveChartDialog::follow 接的是源控件的 dataChanged 信号)。
   同一个源只留一个窗口: 已经开着就抬到前面 (再双击不会开出一堆重复窗口)。
   extraReadout: 见 CurveChartDialog::follow 的说明 (BC 那张图用它把 CE 带过去)。
*/
void MainWindow::openLargeChart(CurveChart *source, const QString &title,
                                const std::function<QString()> &extraReadout)
{
    const auto it = m_largeCharts.find(source);
    if (it != m_largeCharts.end() && it.value() != nullptr) {
        it.value()->show();
        it.value()->raise();
        it.value()->activateWindow();
        return;
    }
    auto *dlg = new CurveChartDialog(title, this);
    /*
       纵轴单位在 CurveChartDialog::syncFromSource() 里从源控件抄 (follow 会给它),
       这里不再自己写一份 —— 这里原来那行是 `source == ui->lossChart ? QString() :
       QString()`, 两个分支一模一样, 等于没设, 于是"保真度"放大之后数字没有 "%"。
    */
    dlg->follow(source, extraReadout);
    /* 关掉时把表里的指针清掉 (窗口是 WA_DeleteOnClose, 会自己析构) */
    connect(dlg, &QObject::destroyed, this, [this, source]() {
        m_largeCharts.remove(source);
    });
    m_largeCharts.insert(source, dlg);
    dlg->show();
    dlg->raise();
    dlg->activateWindow();
}

/*
   把两条曲线的"最新值 / 均值 / 样本数"写进图下面的标签。
   曲线本身是画出来的 (UIA 读不到数字), 而图下的数字读数既有用 (一眼看到当前值),
   又让自动化脚本能读到确凿信息 (tools/verify_match_ui.ps1 就靠它判断"有没有数据")。
*/
void MainWindow::updateMetricsLabels()
{
    /* 读数格式化统一在 CurveChart::readoutText 里, 免得三处各写一份 (以前就不一致) */
    ui->lossValueLabel->setText(
        ui->lossChart->readoutText(QStringLiteral("损失")));
    /*
       奖励那条的前缀写明"局内累计": 它的点不是"每局一个终值", 而是**一局之内逐步
       累加**的曲线 (每手一个点), 所以"均值/最小/最大"描述的是**累计值**的分布 ——
       不写清楚的话, 均值很容易被误读成"平均每局奖励"。
    */
    ui->rewardValueLabel->setText(
        ui->rewardChart->readoutText(QStringLiteral("奖励(局内累计)")));
    /*
       ---- [2026-10] BC 保真度那一条 ----
       图表只画"一致率 / P(老师)"两条**同量纲**的线, 而 CE 与它们不同量纲 ⇒ 不画, 放在
       这一行里。三件事一起给 (与"损失/奖励"两行同一个做法): 曲线给趋势, 读数给确切的数。
       `readoutText` 里已经带了每条线的最新值/均值/样本数, 这里再补 CE 与窗口大小。
    */
    if (m_bcChart != nullptr && m_bcValueLabel != nullptr) {
        /* 曲线上的数走 readoutText, "不在曲线上的数"(CE) 由 bcFidelityCeTail 给 ——
           放大窗口用的是同一对 (见 setupChartTabs 末尾的接线), 两处不会分叉。 */
        m_bcValueLabel->setText(
            m_bcChart->readoutText(QStringLiteral("克隆保真度")) + bcFidelityCeTail());
    }
    /*
       ---- [2026-10] 价值评估那一行 ----
       与 BC 那一行同一个做法: 曲线给趋势 (EV), 读数行给"不在曲线上的数"(校准误差 /
       样本对数 / Var(z))。没出过点时给一句解释, 而不是留空 —— 空行会被读成"没这个功能"。
    */
    if (m_valueChart != nullptr && m_valueValueLabel != nullptr) {
        /*
           读数行**永远**给状态 (曲线可能是空的, 但原因必须看得见):
             * 已经有局结束过 (m_valueGameNo > 0) -> 曲线读数 + "不在曲线上的数" + 为什么没点;
             * 一局都还没结束 -> 说明它在等什么。
        */
        QString t = m_valueChart->readoutText(QStringLiteral("价值评估 EV"));
        if (m_valueGameNo > 0) {
            t += valueDiagTail();
        } else {
            t = QStringLiteral("价值评估: 还没有读数 —— 每个点是**一局结束时**算的 "
                               "(需要: 走完一局 + 该方有 V 头(PPO 两支) + 折扣回报有方差); "
                               "只采有 V 头那一方的着手, 所以 pairs 约等于它走过的步数");
        }
        m_valueValueLabel->setText(t);
    }
}

/*
 * requestSelfCheckPanelUpdate - 请求把当前 agent 的自检报告写进右侧面板
 *
 * 为什么要有这个面板 (这一节的全部理由):
 *   面板上原来只有两条曲线 + 一个"逐局明细"列表。而**这两样都不能判断"这个模型
 *   值不值得继续训"**:
 *     * 损失只说明网络与自己的目标一致 —— 一个把 Q 学成常数、或者用自举把材质
 *       当终局的网络, 损失一样可以很低 (实测 DQN+MCTS 是 22, PPO 是 0.003,
 *       两个数**量纲不同、都不可比**, 也都与棋力无关);
 *     * 自对弈的"胜负"里赢家和输家是同一份权重 —— "50 胜 50 和"里没有任何一条
 *       信息是关于棋力的 (实测同一份 PPO 权重对 AB 深度 4 是 0 胜 1 和 23 负)。
 *   真正的前置判据是**结构/口径**类事实: 动作编码有没有别名、状态能不能观测到
 *   规则历史、终局信号有没有真的进过目标。这些原来只能靠命令行探针
 *   (probe_dqnmcts_aliasing) 看, 现在接到界面上。
 *
 * 线程与时机:
 *   * 数据源是 `ChessBoard::getAgentSelfCheck()`, 它转发到 agent 的
 *     `selfCheckReport()`。那个函数**只读**, 但它读的是常驻 agent 的内部状态, 而
 *     ChessBoard 那一层要等 `m_agentMutex` (后台训练正在 loadModel 时能等上几秒;
 *     2026-09 用户报的"自动保存权重时崩溃"就是这条路径与保存抢同一个网络, 见
 *     chessboard.cpp 的 saveCurrentAgentModel)。
 *   * **所以刷新走后台 worker**: `requestSelfCheckPanelUpdate()` 只置一个标志,
 *     worker 去算 (它可以安心地等锁), 算完用队列信号回调 `applySelfCheckPanel()`
 *     上屏。同步做的话, 面板每一手刷新一次就会把 GUI 线程按在锁上几秒
 *     ("界面卡死"), 那是另一个已经被记过的老问题。
 *   * 调用时机: 选中 agent 时、每一手"探索+预训练"之后、以及启动加载完成时。
 *     这些都是"棋盘状态刚变过"的点, 于是对局累计读数会跟着走。
 *   * 报告里那些**对局累计**的计数来自主 agent, 而后台训练跑在 clone 上 ——
 *     面板里写明了这一点, 否则显示 0 会被读成"没训练过"。
 */
void MainWindow::requestSelfCheckPanelUpdate(bool allAgents)
{
    {
        std::lock_guard<std::mutex> lk(m_selfCheckMutex);
        /*
           请求合并: 面板可能在一手棋里被请求多次 (选 agent + 探索完成), 而 worker
           算一份要等锁 —— 只保留"最新一次的意图", 中间那些没有意义的中间态。
        */
        m_selfCheckPending = true;
        m_selfCheckAll = allAgents;
        m_selfCheckType = ui->gameWidget->getAgentType();
    }
    if (!m_selfCheckThread.joinable()) {
        m_selfCheckThread = std::thread(&MainWindow::selfCheckWorkerLoop, this);
    }
    m_selfCheckCv.notify_one();
}

void MainWindow::selfCheckWorkerLoop()
{
    for (;;) {
        bool all = false;
        ChessBoard::AgentType type = ChessBoard::AGENT_ALPHABETA;
        {
            std::unique_lock<std::mutex> lk(m_selfCheckMutex);
            m_selfCheckCv.wait(lk, [this] { return m_selfCheckStop || m_selfCheckPending; });
            if (m_selfCheckStop) {
                return;
            }
            m_selfCheckPending = false;
            all = m_selfCheckAll;
            type = m_selfCheckType;
        }

        /* ---- 这里可能等 m_agentMutex 几秒: 这是 worker 线程, 界面不受影响 ---- */
        QString text;
        /*
           "全部模型自检"的列举名单。
           [2026-10 用户口径] 这里原来还列着 PG / DQN / SAC 59e5233 还原版两支 ——
           那四支已经从下拉框与**启动预加载**里移除 (见 kAgents 顶部那段注释: 省下
           ~11.2 s 启动时间), 于是它们在这一份快照里只会印出
           "(没有实例/没有自检项: 该 agent 尚未被创建)" —— 四行噪声, 读起来像"坏了"。
           名单跟着"这个程序实际提供什么"走, 所以一并删掉。
           (四个类本身还在, 它们的自检文本仍被 test_sacaz 断言 —— 只是不再从这里列举。)
        */
        static const ChessBoard::AgentType kAll[] = {
            ChessBoard::AGENT_ALPHABETA,
            /* Alpha-Beta 三档弱等级: 与上面那一档并列列出, 自检报告里会各自印出
               **实际搜索深度**, 一眼能核"L1/L2/L3 到底是不是 1/2/3 层" */
            ChessBoard::AGENT_AB_L1, ChessBoard::AGENT_AB_L2, ChessBoard::AGENT_AB_L3,
            ChessBoard::AGENT_MCTS,
            /* [2026-10] PG / DQN 已从这里删除: 它们不再被启动预加载 (见上面那段),
               没有实例 ⇒ 只会印 "(没有实例/没有自检项: 该 agent 尚未被创建)"。 */
            ChessBoard::AGENT_PPOMCTS,   ChessBoard::AGENT_DQNMCTS,
            ChessBoard::AGENT_EVAB,      ChessBoard::AGENT_SACAZ,
            ChessBoard::AGENT_SACAZ_MOE, ChessBoard::AGENT_DQNAB,
            ChessBoard::AGENT_PPOMCTS_MLP,
            /* [2026-10] SAC 59e5233 还原版两支同样删除 (同上: 不再预加载, 列出来只是噪声)。
               两个类与它们的自检文本都还在, `test_sacaz` [14] 节仍在断言。 */
            /* [2026-09 dev-dqnmcts-moetb] DQN+MCTS (稀疏MoE+TB专家): 独立类, 自检面板
               要能回答"骨干里真的有几个专家/几个注意力头在用"这类只能靠读数发现的事 */
            ChessBoard::AGENT_DQNMCTS_MOE
        };
        if (all) {
            text = QStringLiteral(
                "=== 全部模型自检 (快照; 下一手棋会自动刷回当前 agent) ===\n"
                "读法: 先看\"动作别名\"与\"规则上下文通道\"两行 —— 它们决定这个模型"
                "**能不能**学到某些东西; 再看权重文件那两行 —— 它决定这份权重"
                "**有没有**被载进来。\n");
            for (ChessBoard::AgentType t : kAll) {
                text += QStringLiteral("\n");
                const std::string w = ui->gameWidget->getAgentWeightStatus(t);
                if (!w.empty()) {
                    text += QString::fromStdString(w);
                }
                const std::string r = ui->gameWidget->getAgentSelfCheck(t);
                if (r.empty()) {
                    text += QStringLiteral("(没有实例/没有自检项: 该 agent 尚未被创建)\n");
                } else {
                    text += QString::fromStdString(r);
                    if (text.right(1) != QLatin1String("\n")) {
                        text += QStringLiteral("\n");
                    }
                }
                text += QStringLiteral("--------------------------------------------------\n");
            }
        } else {
            /*
               面板顶端几行是**权重文件状态** (启动时有没有扫到、文件在不在、多大),
               下面才是 agent 自己的结构/口径读数。
               为什么把前者也放进来: "权重没载进来"的默认表现是**静默**地从随机初始化
               开始跑 —— PPO+MCTS 就曾经因为"启动扫描的名字与实际写出的名字不一致"而
               从来没被载入过 (那 279 MB x 2 的文件一直躺在盘上), 界面上一点异常都没有。
            */
            const std::string weightStatus = ui->gameWidget->getAgentWeightStatus(type);
            const std::string report = ui->gameWidget->getAgentSelfCheck(type);
            if (!weightStatus.empty()) {
                text += QString::fromStdString(weightStatus);
                text += QStringLiteral("\n");
            }
            if (report.empty()) {
                text += QStringLiteral(
                    "当前 agent 没有可报告的自检项。\n"
                    "(所有已实现的 agent 类型都有自检; 空串一般表示这个 agent 还没有实例 ——\n"
                    " 常见原因是它的权重文件没被扫到, 见上面那几行)\n"
                    "注意: 自检报告的是**结构与口径**, 不是棋力。\n"
                    "要判断棋力用 bench_anchor 的锚点对局 (带 95% 区间的 Elo 差)。");
            } else {
                text += QString::fromStdString(report);
                text += QStringLiteral("\n(点\"全部模型自检\"可以把所有 agent 排在一起对比)");
            }
            /*
               ---- [2026-10] 最近一次行为克隆 (BC) 的报告 ----
               为什么贴在**这里** (而不是新建一个文本视图): 见 mainwindow.h 里
               buildBcControls 上面那三条理由 —— 核心是"这一列没有滚动区, 少一个视图就少
               一类'看不见'的坑", 而且报告存在 ChessBoard 里, 每次刷新都拼回来, 于是
               下一手棋的自检刷新不会把 BC 的结果冲掉。
               报告自带抬头 (=== 行为克隆 (BC) ... / agent : <名字>), 所以即使它属于
               **另一个** agent, 也不会被误读成"当前 agent 的数据"。
            */
            const QString bc = ui->gameWidget->bcReportText();
            if (!bc.isEmpty()) {
                text += QStringLiteral("\n\n---------- 最近一次行为克隆 (BC) ----------\n");
                text += bc;
                if (text.right(1) != QLatin1String("\n")) {
                    text += QStringLiteral("\n");
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk(m_selfCheckMutex);
            m_selfCheckText = text;
            m_selfCheckReady = true;
        }
        /*
           [2026-10 门控实验] 顺手取一份稀疏 MoE 的负载快照 —— **在同一个 worker 里**,
           因为它要在 agent 锁上读 (与自检同一条约束), 而这条 worker 线程就是为那件事
           存在的。只对"当前 agent"取 (all=true 是横向快照, 那一眼看的是口径不是负载)。
        */
        if (!all) {
            ChessBoard::MoeLoadSnapshot snap;
            const bool okLoad = ui->gameWidget->getMoeLoad(type, snap);
            /*
               [2026-10 诊断] 这一行是"控件为什么停在 state=na"唯一能定位的地方:
               它把三件事分开报 —— 类型对不对 (type)、实例建出来没有 (experts)、
               以及 getMoeLoad 自己的返回值 (ok)。没有它, "na" 既可能是"选错 agent"、
               也可能是"实例还没建 (懒建, 还没走过一步)"、也可能是"这一支没接线",
               三种原因的修法完全不同 —— 而界面上它们显示成同一个 "不适用"。
               与 `ppoTrainTrace` 同一做法: 默认关, 用环境变量打开
               (`set CHESS_MOE_TRACE=1`), 免得每手一条日志把启动日志淹掉。
            */
            static const bool kMoeTrace = (std::getenv("CHESS_MOE_TRACE") != nullptr);
            if (kMoeTrace) {
                qInfo("[moe-load] type=%d experts=%d topK=%d applicable=%d split=%d ok=%d",
                      (int)type, snap.experts, snap.topK,
                      (int)snap.applicable, (int)snap.splitReady, (int)okLoad);
            }
            std::lock_guard<std::mutex> lk(m_selfCheckMutex);
            m_moeLoadSnapshot = snap;
            m_moeLoadReady = true;
        }
        /* 上屏必须在 GUI 线程: 队列投递 (worker 不碰控件) */
        QMetaObject::invokeMethod(this, [this]() { applySelfCheckPanel(); },
                                  Qt::QueuedConnection);
    }
}

void MainWindow::applySelfCheckPanel()
{
    QString text;
    bool loadReady = false;
    ChessBoard::MoeLoadSnapshot snap;
    {
        std::lock_guard<std::mutex> lk(m_selfCheckMutex);
        if (!m_selfCheckReady) {
            return;
        }
        m_selfCheckReady = false;
        text = m_selfCheckText;
        if (m_moeLoadReady) {
            m_moeLoadReady = false;
            snap = m_moeLoadSnapshot;
            loadReady = true;
        }
    }
    ui->selfCheckView->setPlainText(text);
    if (loadReady && m_moeLoadView != nullptr) {
        m_moeLoadView->setSnapshot(snap);
        /* 放大窗口开着的话一起更新 (两边永远显示同一份快照, 不会一个新一个旧) */
        if (m_moeLoadDialog != nullptr) {
            m_moeLoadDialog->setSnapshot(snap);
        }
    }
}

/*
 * showAllAgentsSelfCheck - 所有 agent 的自检报告排在一起 (一次性快照)
 *
 * 为什么要横向对比: 单个 agent 的报告只能说明"我这个模型有没有表示/口径问题",
 * 而设计上的差别 (谁的动作编码是 128 槽哈希、谁能看见规则上下文、谁的权重文件
 * 根本没被扫到) 只有**排在一起**才看得出来。这一份就是那张对照表。
 *
 * 刻意不做成"常驻视图": 自检会在每手棋之后自动刷新 (见 requestSelfCheckPanelUpdate
 * 的调用点), 那才是面板的默认语义。所以按钮的提示里写明了"之后会刷回当前 agent"。
 */
void MainWindow::showAllAgentsSelfCheck()
{
    requestSelfCheckPanelUpdate(true);
}

/*
   每场对弈开始: 重建奖励曲线的两条序列 (名字换成这一场的两位参赛者)。

   注意这里必须用 `removeAllSeries()` 而不是 `clearData()`: 后者只清点、不清线,
   于是每跑一场图上就**多挂两条**空线 —— 第三场时读数标签会变成
   "SAC+AZ-MoE: 暂无 | Alpha-Beta: 暂无 | SAC+AZ-MoE: 最新 0 ... | Alpha-Beta: ..."
   (用户在 100 局对弈里的实录, 见 docs/agents_design.md 13.8), 导出的 CSV 也会多出
   几列同名空数据。
*/
void MainWindow::resetMetricsForMatch(const QString &agentA, const QString &agentB)
{
    ui->rewardChart->removeAllSeries();
    /*
       [④] 曲线名带上**口径标签**。奖励曲线取的是 agent 的学习口径 (见 setupMetricsPanel
       与 aiagent.h 的说明), 而纯搜索 agent 没有学习口径、画的是引擎口径 —— 两个口径差
       10 倍, 同一张图上混着两种口径的线时**不能直接比大小**。标签是这件事唯一的提示
       (用户以前就是拿引擎口径 CSV 里的 4.5 推出"材质比赢棋重要 3.5 倍", 而 agent 学的
       是 0.35 : 1, 见 docs/sac_learn_reward_2026_09.md §1.1)。
    */
    const QString suffixA = ChessBoard::agentHasLearningReward(m_matchTypeA)
                                ? QStringLiteral(" [学习口径]")
                                : QStringLiteral(" [引擎口径]");
    const QString suffixB = ChessBoard::agentHasLearningReward(m_matchTypeB)
                                ? QStringLiteral(" [学习口径]")
                                : QStringLiteral(" [引擎口径]");
    m_rewardSeriesA = ui->rewardChart->addSeries(agentA + suffixA, kSeriesColors[0]);
    m_rewardSeriesB = ui->rewardChart->addSeries(agentB + suffixB, kSeriesColors[1]);

    /*
       ---- [2026-10] "行为克隆"那张曲线: 每场换一次线 ----
       建线判据与"清线"都在 armBcChartSeries 里 (人机那条路共用同一份实现 ——
       否则两条路各写一遍,"人机里曲线一直是空的"这类缺口会再长出来一次)。
    */
    armBcChartSeries(ChessBoard::bcHasStudent(m_matchTypeA, m_matchTypeB),
                     ui->gameWidget->bcTeacherDepth(), false);
}

/*
 * ============================================================================
 *  armBcChartSeries - "行为克隆"那张曲线上该不该有线"的唯一一处实现 (2026-10)
 * ============================================================================
 *
 * 两种模式共用它:
 *   * **Agent 对 Agent**: 每场开始 (`resetMetricsForMatch` ← `matchStarted`);
 *   * **人机对弈**: 每局开始 (`humanGameStarted` ← 玩家落下本局第一子)。
 *
 * 为什么必须共用: 两条路的触发时机不同, 但"什么时候该有线"是同一条判据 ——
 *   `老师下拉框 ≠ 关` **且** `场上/对战AI 那一侧能做 BC 学生`。
 * 两条路各写一遍的代价, 本工程刚刚付过一次: 人机那条路没建线, 而
 * `CurveChart::addPoint` 在序列不存在时是**静默 return** ⇒ "BC 明明在训练(面板有
 * samples/updates), 曲线却一直空着"。所以这里不只是"抽个函数", 是把**静默**那一半堵掉。
 *
 * `studentAvailable`: 这一场/这一局里有没有学生 (对局: `bcHasStudent(A,B)`;
 *                     人机: `bcSupported(对战AI)`)。判据在调用方算, 因为它只有调用方知道。
 * `humanMode`: 只影响"没开"时那句读数的措辞 (两种模式的原因不同, 下一步动作也不同)。
 *
 * 清线 (而不是清点) 是刻意的: 横轴是"**这一场/这一局**的第几次 actor 更新", 而人机那条路
 * 的计数器每局归零 (`bcResetMatchStats`), 所以点数必须跟着从零开始 —— 否则同一张图上会
 * 出现"上一局的 4,8,12…"与"这一局的 4,8,12…"接在一起, 横轴就不是一个时间轴了。
 */
bool MainWindow::armBcChartSeries(bool studentAvailable, int teacherDepth, bool humanMode)
{
    if (m_bcChart == nullptr) {
        return false;
    }
    m_bcChart->removeAllSeries();
    m_bcLastFidTop1 = -1.0;
    m_bcLastFidPTeacher = -1.0;
    m_bcLastFidCe = -1.0;
    m_bcLastFidWindow = 0;
    /*
       [2026-10 用户口径] 建线的条件: 老师来自**下拉框** (与对手无关), 所以只要
       "下拉框不是关" **且** "有一侧能做学生"就建线。
       旧条件是"一边 AB + 一边可训练 agent" —— 那会让"PPO vs MCTS 并且选了老师"这一场
       明明在克隆却不画线 (读数与曲线对不上)。
    */
    const bool ok = (teacherDepth > 0) && studentAvailable;
    if (ok) {
        m_bcChart->addSeries(QStringLiteral("一致率 (策略头 top-1 == 老师那一手)"),
                             kSeriesColors[0]);
        m_bcChart->addSeries(QStringLiteral("P(老师着法)"),
                             kSeriesColors[2 % kSeriesColorCount]);
    }
    updateMetricsLabels();
    if (!ok && m_bcValueLabel != nullptr) {
        /*
           没开的原因要**分开写**: "老师是关的"与"对手/对战AI 那一支没有策略头"是两件事,
           用户的下一步动作完全不同 (拧下拉框 vs 换 agent)。与人机那条路的说明一一对应。
        */
        m_bcValueLabel->setText(
            humanMode
                ? QStringLiteral("行为克隆: 没开 (人机对局里 学生 = \"对战AI\" 那一支, 老师 = "
                                 "上面那个下拉框; 现在 %1)")
                      .arg(teacherDepth <= 0
                               ? QStringLiteral("老师还是\"关\"")
                               : QStringLiteral("\"对战AI\" 那一支不是 PPO/SAC, 没有策略头可克隆"))
                : QStringLiteral("行为克隆: 没开 (在 A/B 面板把\"行为克隆训练\"选成某一档 "
                                 "Alpha-Beta; 场上还需要有一支 PPO/SAC —— 老师与对手无关)"));
    }
    /*
       ---- [2026-10] 面板里留一行机器可读的"曲线准备好了没有" ----
       为什么必须有它: "曲线是空白的"有**两种完全不同的原因** ——
         (a) 还没建线 (人机对弈那条路在 2026-10 之前一直如此: 没有 matchStarted, 而
             `addPoint` 在序列不存在时静默 return ⇒ 读数在动、曲线永远空);
         (b) 建了线但样本还不够 (保真度要 ≥8 条样本、每 4 次 actor 更新才出第一个点)。
       两种在图上看起来**一模一样**, 而下一步动作完全不同 (前者是缺陷, 后者是等一会儿)。
       所以把状态写成一行带 ASCII 记号的读数: verify_bc_ui.ps1 据此断言"人机那条路真的
       建了线", 用户也能一眼看出卡在哪一步。与自检面板里的 MaxVio / Loss-Free 同一做法。
    */
    if (ui->selfCheckView != nullptr) {
        ui->selfCheckView->moveCursor(QTextCursor::End);
        ui->selfCheckView->insertPlainText(
            QStringLiteral("[BC] 曲线%1 (chart-%2): 老师=%3 [teacher-depth=%4] [mode=%5]\n")
                .arg(ok ? QStringLiteral("已建线") : QStringLiteral("未建线"),
                     ok ? QStringLiteral("armed") : QStringLiteral("not-armed"),
                     (teacherDepth > 0) ? ChessBoard::bcTeacherName(teacherDepth)
                                        : QStringLiteral("关"))
                .arg(teacherDepth)
                .arg(humanMode ? QStringLiteral("human") : QStringLiteral("match")));
        ui->selfCheckView->moveCursor(QTextCursor::End);
    }
    return ok;
}

/*
 * onBcFidelitySample - "行为克隆"那条曲线的一个点 (2026-10)
 *
 * 数据源: `ChessBoard::bcFidelitySample`, 在对弈线程里每 4 次 actor 更新发一次
 * (口径与命令行 `train_bc` 的 `BC::evaluate` 完全同一份实现)。
 *
 * 两条线都是**百分比**, 所以画在同一根纵轴上不会出现"两个口径比大小"的问题:
 *   * 一致率 top-1 (%) : 策略头单独选中的着法 == AB 老师那一手 的比例
 *   * P(老师着法) (%)  : 策略头给老师那一手的平均概率
 * CE 不进曲线 (不同量纲) —— 它进图下的读数行, 与上面两个数一起显示。
 */
void MainWindow::onBcFidelitySample(int updateNo, double top1Pct, double pTeacher,
                                    double ce, int windowN)
{
    Q_UNUSED(updateNo);
    m_bcLastFidTop1 = top1Pct;
    m_bcLastFidPTeacher = pTeacher;
    m_bcLastFidCe = ce;
    m_bcLastFidWindow = windowN;
    if (m_bcChart != nullptr) {
        m_bcChart->addPoint(0, top1Pct);
        m_bcChart->addPoint(1, pTeacher * 100.0);
    }
    updateMetricsLabels();
}

/*
 * onValueDiagSample - "价值评估"那条曲线的一个点 (2026-10)
 *
 * 数据源: `ChessBoard::valueDiagSample`, 在**每局结束时**发一次: 把本局每手记下的
 * V(s) 与该局的真实结果 (走子方视角的 +1/0/−1) 对照, 在"最近 2000 手"的滚动窗口上算
 * **解释方差 EV** (以及校准误差)。口径与 `bench_diag` 的 [1b] 完全同一份实现
 * (`RL::Diag::explainedVariance` / `calibration`), 所以命令行与界面上的两个数可比。
 *
 * 为什么曲线上只有一个 EV 而校准误差只在读数行: 量纲不同 (EV ∈ (−∞,1] 无量纲,
 * 校准误差 ∈ [0,2] 是"概率差"), 画在一根轴上就是两个口径混着比大小。
 *
 * ⚠ 这个槽**只在真正有读数时被调用** —— "全是和棋 (Var(z)=0) 导致 EV 无定义"那种情况
 *   ChessBoard 侧就不发信号了 (它会往自检面板写一行说明), 所以曲线里不会出现
 *   "假 0"(看起来像"V 和常数预测一样烂")。
 */
void MainWindow::onValueDiagSample(int gameNo, double evWin, double rhoEng, double evEng,
                                   double calibErr, int pairs, double zVarWin, double zVarEng)
{
    m_valueCalibErr = std::isnan(calibErr) ? 0.0 : calibErr;
    m_valueSampleCount = pairs;
    m_valueZVar = zVarWin;
    m_valueZVarEng = zVarEng;
    m_valueEvEng = evEng;
    m_valueGameNo = gameNo;      /* > 0 = 已经有过至少一局结束 (读数行据此决定说什么) */
    /*
       两条线各画各的点。**NaN 不上图** —— CurveChart::addPoint 对非有限值直接丢弃,
       所以"某一口径无方差"时那条线就是不增长, 而不是掉到 0 (看起来像"critic 很烂")。
    */
    if (m_valueChart != nullptr) {
        /* NaN 不上图 (CurveChart::addPoint 对非有限值直接丢弃) ⇒ 算不出来的那条线不增长,
           而不是掉到 0 (0 会被读成"没有信息"/"反着")。 */
        m_valueChart->addPoint(0, evWin);
        m_valueChart->addPoint(1, rhoEng);
    }
    updateMetricsLabels();
}

/* agent 名 -> 损失曲线下标; 第一次见到这个 agent 时新建一条 */
int MainWindow::lossSeriesFor(const QString &agentName)
{
    const auto it = m_lossSeries.constFind(agentName);
    if (it != m_lossSeries.constEnd()) {
        return it.value();
    }
    /*
       分线键是 **agent 名字**, 所以这个名字必须**稳定**: 以前 EVAB 的名字里带着
       当前的 blend ("..., blend=0.30"), 于是它每变一次 blend 就多出一条曲线 ——
       一局下来同一张图上出现 6 条 "EVAB ..." (实测)。现在 getName() 只留结构信息。
       这里再加一道防线: 条数超过 6 条就警告一次 —— 那说明又有 agent 的名字不稳定。
    */
    if (m_lossSeries.size() >= 6) {
        qWarning() << "[metrics] 损失曲线已经有" << m_lossSeries.size()
                   << "条, 又出现新名字:" << agentName
                   << "(agent 的 getName() 是不是把会变的状态写进名字了?)";
    }
    const int idx = ui->lossChart->addSeries(
        agentName, kSeriesColors[m_lossSeries.size() % kSeriesColorCount]);
    m_lossSeries.insert(agentName, idx);
    return idx;
}

/*
 * writeChartCsv - 把**一张**曲线的数据写成 CSV (2026-09)。
 *
 * 为什么要抽出来: 原来只有"导出 CSV"一个按钮, 它把损失与奖励**两段**拼在同一个文件里。
 * 两张曲线的行数口径不同 (损失 = 每完成一次在线训练一个点; 奖励 = 每手一个点), 于是
 * 想单独看损失就得先手工切段, 而"从哪一行开始是奖励"只靠一行注释区分 —— 很容易切错。
 * 现在损失曲线有自己的按钮 (`exportLossBtn`), 走的就是这个函数。
 *
 * 文件格式 (与合并导出里的同一段逐字相同):
 *   # <注释行>
 *   sample,<曲线名 1>,<曲线名 2>,...
 *   1,<v>,<v>,...
 *   ...
 * 某条曲线比别的短时后面的列留空 (不是补 0 —— 补 0 会被读成"当时损失是 0")。
 */
bool MainWindow::writeChartCsv(CurveChart *chart, const QString &what,
                               const QString &sectionComment, const QString &defaultName)
{
    if (chart == nullptr || chart->seriesCount() <= 0) {
        QMessageBox::information(this, QStringLiteral("没有数据可导出"),
                                 QStringLiteral("%1还没有任何数据点 (先跑一局或等后台训练上报)。")
                                     .arg(what));
        return false;
    }
    const QString path = QFileDialog::getSaveFileName(
        this, QStringLiteral("导出%1").arg(what),
        QStringLiteral("%1_%2.csv").arg(defaultName,
                                       QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")),
        QStringLiteral("CSV (*.csv);;所有文件 (*.*)"));
    if (path.isEmpty()) {
        return false;      /* 用户取消 */
    }
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导出失败"),
                             QStringLiteral("打不开文件: %1").arg(f.errorString()));
        return false;
    }
    QTextStream out(&f);
    /* 文本由 CurveChart::toCsv() 生成 —— 与"导出 CSV"里的那一段逐字相同 (格式单一来源) */
    out << chart->toCsv(sectionComment);
    f.close();
    QMessageBox::information(this, QStringLiteral("导出完成"),
                             QStringLiteral("已写出: %1\n(%2: %3 个采样点, %4 条曲线)")
                                 .arg(path).arg(what)
                                 .arg(chart->sampleCount()).arg(chart->seriesCount()));
    return true;
}

/*
 * exportLossCsv - "导出损失曲线"按钮 (2026-09 用户要求增加的控件)。
 *
 * 只写损失那一张图的数据。分析训练时最常要的就是这一段: 每个 agent 一条线, 每条线的
 * 点数 = 它完成在线训练的次数。**注意它不是"每手一个点"** —— 池子没攒够 batchSize 时
 * learnBatch 故意不更新 (见 SACAZAgent::learnBatch 的说明), 所以曲线的密度本身就是
 * "有没有真的在学"的读数 (用户就是这么发现两个 SAC 的损失曲线疏密不同的)。
 */
void MainWindow::exportLossCsv()
{
    writeChartCsv(ui->lossChart, QStringLiteral("训练损失曲线"),
                  QStringLiteral("训练损失 (每完成一次在线训练一个点; 每个 agent 一列)"),
                  QStringLiteral("loss"));
}

/* 把当前曲线导出成 CSV (两列不同长度, 所以分两段写, 带表头) */
void MainWindow::exportMetricsCsv()
{
    const QString path = QFileDialog::getSaveFileName(
        this, QStringLiteral("导出指标曲线"),
        QStringLiteral("metrics_%1.csv")
            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")),
        QStringLiteral("CSV (*.csv);;所有文件 (*.*)"));
    if (path.isEmpty()) {
        return;
    }
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("导出失败"),
                             QStringLiteral("打不开文件: %1").arg(f.errorString()));
        return;
    }
    QTextStream out(&f);
    /*
       两段的文本由 CurveChart::toCsv() 生成 —— 与"导出损失曲线"按钮用的是**同一份**
       格式化 (格式放在控件里, 写文件/选路径留在这里; 于是格式本身能被 test_match
       的 [2.9] 节断言, 而不用去驱动文件对话框)。
    */
    out << ui->lossChart->toCsv(
        QStringLiteral("训练损失 (每完成一次在线训练一个点; 每个 agent 一列)"));
    /*
       奖励这一段的行号是**采样序号**(每手一个点), 不是局数 —— 表头写清楚,
       否则导出的 CSV 很容易被当成"每行一局"来解读 (那是修复前的口径)。
    */
    out << "\n";
    out << ui->rewardChart->toCsv(
        QStringLiteral("环境奖励 (每手一个点; 值是本局累计, 局末那点含终局 +-1)"));
    /*
       ---- [2026-10] 第三段: 行为克隆保真度 ----
       与上面两段同一个理由 (行号口径不同, 所以分段 + 各自的注释行): 这一段的行号是
       **actor 更新次数**, 而不是"第几手"或"第几局"。两个量纲 (一致率 % / P(老师) %) 相同,
       所以能并排; CE 不在这一张图里 (见 setupChartTabs 的说明)。
    */
    if (m_bcChart != nullptr && m_bcChart->seriesCount() > 0) {
        out << "\n";
        out << m_bcChart->toCsv(QStringLiteral(
            "行为克隆保真度 (每 4 次 actor 更新一个点, 最近 64 条样本窗口; "
            "一致率 = 策略头选中的着法 == AB 老师那一手; 代理指标, 不是棋力)"));
    }
    /*
       ---- [2026-10] 第四段: 价值评估 (critic 的 EV) ----
       行号口径 = **局号** (每个点一局), 与上面三段都不同, 所以同样要单独分段 + 写注释行。
       为什么值得导出: EV 是需要"跨配置比较"的量 (改了 critic/骨干/塑形之后, 它是唯一能
       回答"价值有没有变准"的数), 而界面上的曲线只留最近 2000 个点。
    */
    if (m_valueChart != nullptr && m_valueChart->seriesCount() > 0) {
        out << "\n";
        out << m_valueChart->toCsv(QStringLiteral(
            "价值评估 (每局一个点, 最近 2000 手滚动窗口; EV = 1 − Var(z−V)/Var(z), "
            "z = 该局真实结果(走子方视角); 0 = 不如常数预测, 1 = 完美)"));
    }
    f.close();
    QMessageBox::information(this, QStringLiteral("导出完成"),
                             QStringLiteral("已写出: %1").arg(path));
}

/*
 *  saveWeightsAfterMatch - 对弈结束后**静默**把权重存到标准路径
 *
 *  用户要求: 不再弹窗 (既不要"存到哪里"的文件对话框, 也不要"保存成功"的消息框)。
 *  所以这里:
 *    * 目标路径 = ChessBoard::defaultWeightPath() —— 与启动加载/退出保存同一条路径,
 *      下次启动自然读到这次训练的结果;
 *    * 只在**确实实例化过**的 agent 上存 (没跑过的 agent 没有权重可存);
 *    * 放到**常驻后台线程** (m_saveThread) 做: GUI 线程同步写盘会把事件循环堵住,
 *      而权重可能有几百 MB; 读写期间 ChessBoard 会发 busyStarted/busyFinished,
 *      "请稍候"沙漏弹窗由那两个信号驱动, 这里不用管;
 *    * 结果用**界面上的文字**汇报 (逐局明细列表里加一行 + 结果标签的 tooltip),
 *      不用模态框打断用户 —— 失败也看得到, 但不会挡住操作。
 *
 *  ---- 为什么是"常驻线程 + 请求" 而不是"每次起一个线程再 join" (2026-09) ----
 *  原来是每场对弈结束就 `m_saveThread = std::thread(...)`, 而下一场结束时先
 *  `join()` 上一个再起新的。那个 `join()` 在 **GUI 线程**上 —— 于是"上一场正在写
 *  558 MB (约 9 s)"就会把界面冻住 9 秒 (用户点不动、曲线也不动, 正是这个仓库反复
 *  记过的"界面像死了")。短对局连续跑时这个问题每一场都发生。
 *  现在保存线程常驻, 请求 (待保存的 agent 列表) 通过标志交给它, GUI 线程只置标志,
 *  一秒都不等; 多个请求会合并成一次 (去重)。
 * ================================================================ */
void MainWindow::saveWeightsAfterMatch(const QVector<ChessBoard::AgentType> &types)
{
    QVector<ChessBoard::AgentType> todo;
    for (ChessBoard::AgentType t : types) {
        if (!agentIsTrainable(t)) {
            continue;               /* Alpha-Beta / MCTS 是纯搜索, 没有参数 */
        }
        if (!ui->gameWidget->hasAgentInstance(t)) {
            continue;               /* 这次没用过它 -> 没有权重 */
        }
        if (!todo.contains(t)) {
            todo.append(t);
        }
    }
    if (todo.isEmpty()) {
        return;
    }

    if (!todo.isEmpty()) {
        std::lock_guard<std::mutex> lk(m_saveMutex);
        for (ChessBoard::AgentType t : todo) {
            if (!m_saveQueue.contains(t)) {
                m_saveQueue.append(t);
            }
        }
        m_savePending = true;
    }
    if (!m_saveThread.joinable()) {
        m_saveThread = std::thread(&MainWindow::saveWorkerLoop, this);
    }
    m_saveCv.notify_one();
}

/*
 * ---- onHumanGameFinished 已删除 (2026-09 用户口径: 只在退出时保存) ----
 *
 * 它原来做的是"人机对局终局时把当前 agent 的权重落盘"。删掉的理由有两条:
 *   1. **口径变了**: 用户要求"只有程序退出时再保存模型" —— 唯一落盘点改为
 *      ChessBoard::saveAllInstantiatedAgentsOnExit() (由本窗口析构调用)。
 *   2. **它是卡顿的嫌疑人**: 保存要在 `m_agentMutex` 锁内序列化 530 MB 级权重
 *      (PPO+MCTS 266 + 264 MB), 而 AI 决策也拿这把锁 —— 实测 2.4 s 的保存让一次决策
 *      从 6140 ms 涨到 8411 ms。终局恰好是"玩家马上要再走一步"的时刻, 所以它最容易
 *      表现为"黑方无限等待"。
 *
 * 顺带删掉的还有 sendResult -> onHumanGameFinished 那条连接 (见构造函数里的说明)。
 */

void MainWindow::saveWorkerLoop()
{
    for (;;) {
        QVector<ChessBoard::AgentType> todo;
        {
            std::unique_lock<std::mutex> lk(m_saveMutex);
            m_saveCv.wait(lk, [this] { return m_saveStop || m_savePending; });
            if (m_saveStop) {
                return;
            }
            m_savePending = false;
            todo = m_saveQueue;
            m_saveQueue.clear();
        }

        QStringList lines;
        bool allOk = true;
        for (ChessBoard::AgentType t : todo) {
            const std::string path = ChessBoard::defaultWeightPath(t);
            const auto t0 = std::chrono::steady_clock::now();
            /*
               注意 saveCurrentAgentModel 内部会取 agent 锁 (见 chessboard.cpp):
               保存期间后台训练/决策会排队等它, 但那都在**别的线程**上, 界面不受影响。
            */
            const bool ok = ui->gameWidget->saveCurrentAgentModel(t, path);
            const long long ms =
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            /*
               记一条耗时: 稀疏 MoE 变体的权重是 3 x 146 MB, 存一次要十几秒 (后台,
               界面不卡)。慢了/失败了要能一眼看出来, 而不是只看到列表里一行字。
            */
            qInfo().noquote() << QStringLiteral("[weights] 保存 %1: %2 ms -> %3")
                                     .arg(agentLongName(t)).arg(ms)
                                     .arg(QString::fromStdString(path));
            allOk = allOk && ok;
            lines << QStringLiteral("%1 -> %2%3")
                         .arg(agentLongName(t), QString::fromStdString(path),
                              ok ? QString() : QStringLiteral("  [失败]"));
        }
        /* 回到 GUI 线程写界面 (在工作线程里碰控件是错的) */
        QMetaObject::invokeMethod(this, [this, lines, allOk]() {
            for (const QString &l : lines) {
                ui->gameListWidget->addItem(
                    QStringLiteral("—— 已静默保存权重: %1 ——").arg(l));
            }
            if (!allOk) {
                ui->gameListWidget->addItem(
                    QStringLiteral("—— 有权重保存失败, 详见上面带 [失败] 的行 ——"));
            }
            ui->gameListWidget->scrollToBottom();
        }, Qt::QueuedConnection);
    }
}
