#include "mainwindow.h"
#include <chrono>
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
    { "MCTS (800次模拟)",            ChessBoard::AGENT_MCTS },
    { "Policy Gradient (PGEagent)",  ChessBoard::AGENT_PG },
    { "Deep Q-Network (DQN)",        ChessBoard::AGENT_DQN },
    { "PPO+MCTS (AlphaZero)",        ChessBoard::AGENT_PPOMCTS },
    { "DQN+MCTS (AlphaZero)",        ChessBoard::AGENT_DQNMCTS },
    { "EVAB (学会评估的 Alpha-Beta)", ChessBoard::AGENT_EVAB },
    { "SAC+MCTS+AlphaZero (最大熵搜索)", ChessBoard::AGENT_SACAZ },
    /*
       同一个算法, 骨干换成"稀疏路由 MoE + TransformerBlock 专家" (E=4, top-1)。
       与上一项相比: 参数量大 ~4 倍 (4 个 TB 专家), 算力只算 1 个专家 —— 实测
       10.9 ms/模拟 (MLP 骨干 0.07), 所以每次走子只给 16 次模拟 (约 175 ms)。
       界面上把它单独列出来, 就是为了能直接和 MLP 骨干的版本对弈比较。
    */
    { "SAC+MCTS+AlphaZero (稀疏MoE+TB专家)", ChessBoard::AGENT_SACAZ_MOE },
};

void fillAgentCombo(QComboBox *combo, int defaultIndex)
{
    combo->clear();
    for (const AgentChoice &c : kAgents) {
        combo->addItem(QString::fromUtf8(c.name), static_cast<int>(c.type));
    }
    if (defaultIndex >= 0 && defaultIndex < combo->count()) {
        combo->setCurrentIndex(defaultIndex);
    }
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
       对弈参数: 局数与每手探索(预训练)步数。把这两个数字放到界面上是为了让
       "对弈结果"可解释 —— 局数太少结论会被单局偶然性翻转, 探索步数直接决定
       每一步的思考成本。
    */
    ui->gamesSpin->setRange(1, 100);
    ui->gamesSpin->setValue(4);
    ui->gamesSpin->setSuffix(QStringLiteral(" 局"));
    ui->preTrainStepsSpin->setRange(0, 2000);
    ui->preTrainStepsSpin->setSingleStep(16);
    ui->preTrainStepsSpin->setValue(ui->gameWidget->getPreTrainSteps());
    ui->preTrainStepsSpin->setSuffix(QStringLiteral(" 步"));

    /* ---- 指标曲线与逐局明细 (右侧面板, 见 metricsview.h) ---- */
    setupMetricsPanel();

    /* Agent选择 (与你对战的AI; 你执红, 它执黑) */
    connect(ui->agentComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onAgentSelected);

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

    /* AI Self Play / Agent 对弈 按钮 (对弈进行中兼作"停止") */
    connect(ui->selfPlayBtn, &QPushButton::clicked,
            this, &MainWindow::onStartMatch);

    /* 回放控制 */
    connect(ui->recordcomboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onGameSelected);
    connect(ui->prevBtn, &QPushButton::clicked,
            this, &MainWindow::onReplayPrev);
    connect(ui->nextBtn, &QPushButton::clicked,
            this, &MainWindow::onReplayNext);

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

    /* ---- Agent 对弈进度 ---- */
    connect(ui->gameWidget, &ChessBoard::matchStarted, this,
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

    /* 棋盘回放状态信号 */
    connect(ui->gameWidget, &ChessBoard::replayIndexChanged,
            this, &MainWindow::onReplayIndexChanged);
    connect(ui->gameWidget, &ChessBoard::replayModeExited,
            this, &MainWindow::onReplayModeExited);

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
            ui->recordcomboBox->setEnabled(true);
            ui->timeLabel->setText("AI思考时间: -");
            ui->exploreLabel->setText("探索+预训练: -");
            ui->matchResultLabel->setText("对弈结果: -");
            ui->thinkIndicator->resetToIdle();
            ui->gameWidget->setEnabled(true);
            refreshGameList();
        });

    /* ---- 启动加载阶段: 禁用所有交互控件 ---- */
    ui->resetBtn->setEnabled(false);    ui->selfPlayBtn->setEnabled(false);
    ui->agentComboBox->setEnabled(false);
    ui->matchAComboBox->setEnabled(false);
    ui->matchBComboBox->setEnabled(false);
    ui->gamesSpin->setEnabled(false);
    ui->preTrainStepsSpin->setEnabled(false);
    ui->recordcomboBox->setEnabled(false);
    ui->prevBtn->setEnabled(false);
    ui->nextBtn->setEnabled(false);
    ui->stepLabel->setText("步数: -/-");
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
       存权重也可能在后台跑 (见 offerSaveWeights: 保存放到线程里, 好让沙漏能转)。
       它同样访问 ui->gameWidget, 所以必须先 join 再 delete ui。
    */
    if (m_saveThread.joinable()) {
        m_saveThread.join();
    }
    /* 程序退出前保存所有已训练的agent权重 */
    ui->gameWidget->shutdownSave();
    delete ui;
}

void MainWindow::refreshGameList()
{
    ui->recordcomboBox->blockSignals(true);
    ui->recordcomboBox->clear();

    /* 添加提示项 */
    ui->recordcomboBox->addItem("--- 选择历史对局 ---", -1);

    GameDatabase &db = GameDatabase::instance();
    if (db.isOpen()) {
        QStringList games = db.getRecentGames(50);
        for (const QString &line : games) {
            /* 从行头解析对局ID */
            int id = -1;
            if (line.startsWith("对局#")) {
                int endPos = line.indexOf(" |");
                if (endPos > 3) {
                    id = line.mid(3, endPos - 3).toInt();
                }
            }
            ui->recordcomboBox->addItem(line, id);
        }
    }

    ui->recordcomboBox->blockSignals(false);
}

void MainWindow::onGameSelected(int index)
{
    if (index < 0) return;

    int gameId = ui->recordcomboBox->itemData(index).toInt();
    if (gameId < 0) return;

    GameDatabase &db = GameDatabase::instance();
    if (!db.isOpen()) return;

    /* 加载该对局的走法记录 */
    m_currentReplaySteps = db.getGameMoves(gameId);

    if (m_currentReplaySteps.isEmpty()) {
        qDebug() << "No moves for game" << gameId;
        return;
    }

    /* 传入棋盘开始回放 */
    ChessBoard *board = ui->gameWidget;
    board->loadReplayGame(gameId, m_currentReplaySteps);

    /* 启用回放按钮 */
    ui->prevBtn->setEnabled(true);
    ui->nextBtn->setEnabled(true);
}

void MainWindow::onReplayPrev()
{
    ChessBoard *board = ui->gameWidget;
    board->replayPrev();
}

void MainWindow::onReplayNext()
{
    ChessBoard *board = ui->gameWidget;
    board->replayNext();
}

void MainWindow::onReplayIndexChanged(int index, int total)
{
    if (total <= 0) {
        ui->stepLabel->setText("步数: -/-");
        return;
    }
    ui->stepLabel->setText(QString("步数: %1/%2").arg(index + 1).arg(total));

    /* 边界按钮禁用 */
    ui->prevBtn->setEnabled(index >= 0);
    ui->nextBtn->setEnabled(index < total - 1);
}

void MainWindow::onReplayModeExited()
{
    /* 退出回放模式 (例如点了"开局") */
    ui->stepLabel->setText("步数: -/-");
    ui->prevBtn->setEnabled(false);
    ui->nextBtn->setEnabled(false);
}

void MainWindow::populateAgentComboBox()
{
    /* 与你对战的AI: 默认 Alpha-Beta */
    fillAgentCombo(ui->agentComboBox, 0);
    /*
       Agent 对弈的双方。默认 A=Alpha-Beta, B=EVAB: 两个都快 (每手 ~150 ms),
       而且正好是"纯搜索"对"学会评估的搜索", 是这套 agent 里最有意义的一组对照。
       注意 A/B 不是红黑 —— 每局交换先后手, 见 ChessBoard::matchAgents。
    */
    fillAgentCombo(ui->matchAComboBox, 0);
    fillAgentCombo(ui->matchBComboBox, 6);
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

    /* 界面上的设置同步给棋盘 (对弈线程会读这两个开关) */
    board->setPreTrainSteps(ui->preTrainStepsSpin->value());
    board->setPreTrainEnabled(ui->preTrainCheck->isChecked());

    m_matchRunning = true;
    m_matchLog.clear();
    ui->selfPlayBtn->setText("停止对弈");
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
               打过的可训练 agent **静默存盘** (用户要求: 不再弹任何窗口)。
               存到标准路径, 下次启动自然加载; 期间由"请稍候"沙漏提示, 结果写进
               右侧逐局明细列表。见 saveWeightsAfterMatch() 的注释。
            */
            saveWeightsAfterMatch(QVector<ChessBoard::AgentType>{typeA, typeB});

            refreshGameList();
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

    ui->rewardChart->setTitle(
        QStringLiteral("环境奖励 (每手累计, 局末含终局 ±1; 走子方视角)"));
    ui->rewardChart->setValueSuffix(QString());
    ui->rewardChart->setWindow(2000);

    connect(ui->clearMetricsBtn, &QPushButton::clicked, this, [this]() {
        ui->lossChart->clearData();
        ui->rewardChart->clearData();
        m_lossSeries.clear();
        m_rewardSeriesA = -1;
        m_rewardSeriesB = -1;
        ui->gameListWidget->clear();
        ui->scoreLabel->setText(QStringLiteral("当前比分: -"));
        updateMetricsLabels();
    });
    connect(ui->exportMetricsBtn, &QPushButton::clicked,
            this, &MainWindow::exportMetricsCsv);

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
}

/*
   双击曲线 -> 弹一个 900x560 的独立窗口 (可缩放、可拖到别的屏幕), 内容与源控件
   实时同步 (CurveChartDialog::follow 接的是源控件的 dataChanged 信号)。
   同一个源只留一个窗口: 已经开着就抬到前面 (再双击不会开出一堆重复窗口)。
*/
void MainWindow::openLargeChart(CurveChart *source, const QString &title)
{
    const auto it = m_largeCharts.find(source);
    if (it != m_largeCharts.end() && it.value() != nullptr) {
        it.value()->show();
        it.value()->raise();
        it.value()->activateWindow();
        return;
    }
    auto *dlg = new CurveChartDialog(title, this);
    dlg->chart()->setValueSuffix(source == ui->lossChart ? QString() : QString());
    dlg->follow(source);
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
}

/* 每场对弈开始: 重建奖励曲线的两条序列 (名字换成这一场的两位参赛者) */
void MainWindow::resetMetricsForMatch(const QString &agentA, const QString &agentB)
{
    ui->rewardChart->clearData();
    m_rewardSeriesA = ui->rewardChart->addSeries(agentA, kSeriesColors[0]);
    m_rewardSeriesB = ui->rewardChart->addSeries(agentB, kSeriesColors[1]);
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
    out << "# 训练损失\n";
    out << "sample";
    for (int s = 0; s < ui->lossChart->seriesCount(); ++s) {
        out << "," << ui->lossChart->series(s).name;
    }
    out << "\n";
    int maxN = ui->lossChart->sampleCount();
    for (int i = 0; i < maxN; ++i) {
        out << (i + 1);
        for (int s = 0; s < ui->lossChart->seriesCount(); ++s) {
            const CurveChart::Series &sr = ui->lossChart->series(s);
            out << ",";
            if (i < sr.pts.size()) {
                out << QString::number(sr.pts[i], 'g', 8);
            }
        }
        out << "\n";
    }
    /*
       奖励这一段的行号是**采样序号**(每手一个点), 不是局数 —— 表头写清楚,
       否则导出的 CSV 很容易被当成"每行一局"来解读 (那是修复前的口径)。
    */
    out << "\n# 环境奖励 (每手一个点; 值是本局累计, 局末那点含终局 +-1)\n";
    out << "sample";
    for (int s = 0; s < ui->rewardChart->seriesCount(); ++s) {
        out << "," << ui->rewardChart->series(s).name;
    }
    out << "\n";
    maxN = ui->rewardChart->sampleCount();
    for (int i = 0; i < maxN; ++i) {
        out << (i + 1);
        for (int s = 0; s < ui->rewardChart->seriesCount(); ++s) {
            const CurveChart::Series &sr = ui->rewardChart->series(s);
            out << ",";
            if (i < sr.pts.size()) {
                out << QString::number(sr.pts[i], 'g', 8);
            }
        }
        out << "\n";
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
 *    * 放到后台线程 (m_saveThread) 做: GUI 线程同步写盘会把事件循环堵住, 而权重
 *      可能有几百 MB; 读写期间 ChessBoard 会发 busyStarted/busyFinished, "请稍候"
 *      沙漏弹窗由那两个信号驱动, 这里不用管;
 *    * 结果用**界面上的文字**汇报 (逐局明细列表里加一行 + 结果标签的 tooltip),
 *      不用模态框打断用户 —— 失败也看得到, 但不会挡住操作。
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

    /* 上一次保存若还没结束, 先收回来 (同一时刻只允许一个保存任务) */
    if (m_saveThread.joinable()) {
        m_saveThread.join();
    }

    m_saveThread = std::thread([this, todo]() {
        QStringList lines;
        bool allOk = true;
        for (ChessBoard::AgentType t : todo) {
            const std::string path = ChessBoard::defaultWeightPath(t);
            const auto t0 = std::chrono::steady_clock::now();
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
    });
}
