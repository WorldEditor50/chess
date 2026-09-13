#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "chessboard.h"
#include "thinkingindicator.h"
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

QString agentWeightFilename(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PG:      return QStringLiteral("pg_agent_weights.dat");
    case ChessBoard::AGENT_DQN:     return QStringLiteral("dqn_agent_weights.dat");
    case ChessBoard::AGENT_PPOMCTS: return QStringLiteral("ppomcts_actor_weights.dat");
    case ChessBoard::AGENT_DQNMCTS: return QStringLiteral("dqnmcts_agent_weights.dat");
    case ChessBoard::AGENT_EVAB:    return QStringLiteral("evab_agent_weights.dat");
    default:                        return QStringLiteral("agent_weights.dat");
    }
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(900, 650);

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
        });
    connect(ui->gameWidget, &ChessBoard::matchGameFinished, this,
        [this](int no, int games, const QString &line) {
            m_matchLog += line + "\n";
            ui->matchResultLabel->setText(QString("对弈 %1/%2 局完成").arg(no).arg(games));
            ui->matchResultLabel->setToolTip(m_matchLog);
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
    ui->resetBtn->setEnabled(false);
    ui->selfPlayBtn->setEnabled(false);
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

    /* 在后台线程启动异步加载 (数据库 + AI模型权重) */
    ui->gameWidget->setEnabled(false);
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

            /* 结果 + 可展开的逐局明细 */
            QMessageBox box(QMessageBox::Information, "Agent 对弈结果",
                            summary, QMessageBox::Ok, this);
            box.setDetailedText(detail.isEmpty() ? m_matchLog : detail);
            box.exec();

            /* 打过的 RL agent 可以存盘 (Alpha-Beta / MCTS 没有权重) */
            offerSaveWeights(typeA);
            if (typeB != typeA) {
                offerSaveWeights(typeB);
            }

            refreshGameList();
        }, Qt::QueuedConnection);
    });
}

/* ================================================================
 *  offerSaveWeights - 对弈结束后询问是否把某个 agent 的权重存盘
 * ================================================================ */
void MainWindow::offerSaveWeights(ChessBoard::AgentType type)
{
    if (!agentIsTrainable(type)) {
        return;      /* Alpha-Beta / MCTS 是纯搜索, 没有需要保存的参数 */
    }

    const QString filePath = QFileDialog::getSaveFileName(
        this,
        QString("保存 %1 权重文件").arg(agentLongName(type)),
        agentWeightFilename(type),
        "权重文件 (*.dat);;所有文件 (*.*)");
    if (filePath.isEmpty()) {
        return;
    }

    if (ui->gameWidget->saveCurrentAgentModel(type, filePath.toStdString())) {
        QMessageBox::information(this, "保存成功",
            QString("%1 的权重已保存到:\n%2").arg(agentLongName(type), filePath));
    } else {
        QMessageBox::warning(this, "保存失败",
            QString("%1 保存权重文件时发生错误!").arg(agentLongName(type)));
    }
}
