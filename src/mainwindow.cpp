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
    { "EVAB (学会评估的 Alpha-Beta)", ChessBoard::AGENT_EVAB },
    { "SAC+MCTS+AlphaZero (最大熵搜索)", ChessBoard::AGENT_SACAZ },
    /*
       同一个算法, 骨干换成"稀疏路由 MoE + TransformerBlock 专家" (E=4, top-1)。
       与上一项相比: 参数量大 ~4 倍 (4 个 TB 专家), 算力只算 1 个专家 —— 实测
       10.9 ms/模拟 (MLP 骨干 0.07), 所以每次走子只给 16 次模拟 (约 175 ms)。
    */
    { "SAC+MCTS+AlphaZero (稀疏MoE+TB专家)", ChessBoard::AGENT_SACAZ_MOE },
    /*
       SAC+MCTS+AlphaZero 的**行为还原版** (提交 59e5233)。它是一个**独立的 C++ 类**
       (SACAZLegacyAgent, src/sacazlegacyagent.h), 不是同一个类里的运行时开关 ——
       与上面两项放在一起是为了能直接对弈比较"当前口径 vs 59e5233 口径"。
       差别只有四项 (目标熵 0.98 / alpha lr 1e-3 / critic 不钳位+纯 MSE / 叶子全量估值)
       加一处等价的激活写法, 完整表见那个头文件; **权重文件独立**
       (weights/sacaz_old_agent_*), 与上面的 weights/sacaz_agent 不共用 ——
       两者参数结构相同, 结构指纹挡不住串权重, 而训练口径不同会让共用变成静默覆盖。
    */
    { "SAC+MCTS+AlphaZero (59e5233 行为还原版)", ChessBoard::AGENT_SACAZ_OLD },
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
   Qt 的 QComboBox 默认 maxVisibleItems = **10**, 而列表已经有 11 项 —— 第 11 项
   (当时正是新加的 "PPO+MCTS (...MLP专家)") 会被折叠在滚动区里, 打开下拉框只看到 10 行。
   表现就是"明明加进列表了, 界面上却找不到" (UIA 实测: 展开后只有 10 行可见).
   所以这里按条数放宽: 全部条目一次性可见, 不需要滚动。
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
    case ChessBoard::AGENT_SACAZ_OLD:
    case ChessBoard::AGENT_DQNAB:
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
    /*
       ---- 模型自检面板 ----
       时机刻意选在这里: 预训练做完 = 棋盘上又走了一步、训练又更新过一轮权重,
       于是"对局累计"那一类读数会跟着走。报告本身是只读的, 随时可以再刷 (见
       requestSelfCheckPanelUpdate 的说明)。
    */
    connect(ui->gameWidget, &ChessBoard::aiExploreInfo, this,
        [this](const QString &) { requestSelfCheckPanelUpdate(false); });

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
            /* 启动加载完成: 现在才有 agent 可以自检 (之前都是 nullptr) */
            requestSelfCheckPanelUpdate(false);
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
    fillAgentCombo(ui->agentComboBox, ChessBoard::AGENT_ALPHABETA);
    /*
       Agent 对弈的双方。默认 A=Alpha-Beta, B=EVAB: 两个都快 (每手 ~150 ms),
       而且正好是"纯搜索"对"学会评估的搜索", 是这套 agent 里最有意义的一组对照。
       注意 A/B 不是红黑 —— 每局交换先后手, 见 ChessBoard::matchAgents。
       (默认项按**类型**给; 以前写的是下标 6, 而列表一旦插入新 agent 就会静默指错。)
    */
    fillAgentCombo(ui->matchAComboBox, ChessBoard::AGENT_ALPHABETA);
    fillAgentCombo(ui->matchBComboBox, ChessBoard::AGENT_EVAB);
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

    /* 换了 agent 就换一份自检报告 (不支持的 agent 显示"没有自检项") */
    requestSelfCheckPanelUpdate(false);
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
        static const ChessBoard::AgentType kAll[] = {
            ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_MCTS,
            ChessBoard::AGENT_PG,        ChessBoard::AGENT_DQN,
            ChessBoard::AGENT_PPOMCTS,   ChessBoard::AGENT_DQNMCTS,
            ChessBoard::AGENT_EVAB,      ChessBoard::AGENT_SACAZ,
            ChessBoard::AGENT_SACAZ_MOE, ChessBoard::AGENT_DQNAB,
            ChessBoard::AGENT_PPOMCTS_MLP, ChessBoard::AGENT_SACAZ_OLD
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
                    "(十个 agent 都已实现自检; 空串一般表示这个 agent 还没有实例 ——\n"
                    " 常见原因是它的权重文件没被扫到, 见上面那几行)\n"
                    "注意: 自检报告的是**结构与口径**, 不是棋力。\n"
                    "要判断棋力用 bench_anchor 的锚点对局 (带 95% 区间的 Elo 差)。");
            } else {
                text += QString::fromStdString(report);
                text += QStringLiteral("\n(点\"全部模型自检\"可以把十个 agent 排在一起对比)");
            }
        }

        {
            std::lock_guard<std::mutex> lk(m_selfCheckMutex);
            m_selfCheckText = text;
            m_selfCheckReady = true;
        }
        /* 上屏必须在 GUI 线程: 队列投递 (worker 不碰控件) */
        QMetaObject::invokeMethod(this, [this]() { applySelfCheckPanel(); },
                                  Qt::QueuedConnection);
    }
}

void MainWindow::applySelfCheckPanel()
{
    QString text;
    {
        std::lock_guard<std::mutex> lk(m_selfCheckMutex);
        if (!m_selfCheckReady) {
            return;
        }
        m_selfCheckReady = false;
        text = m_selfCheckText;
    }
    ui->selfCheckView->setPlainText(text);
}

/*
 * showAllAgentsSelfCheck - 十个 agent 的自检报告排在一起 (一次性快照)
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

    {
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
