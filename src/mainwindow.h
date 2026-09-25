#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QHash>
#include <QTimer>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "gamedb.h"
#include "chessboard.h"
#include "metricsview.h"

#include "busydialog.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onAgentSelected(int index);
    void onGameSelected(int index);
    void onReplayPrev();
    void onReplayNext();
    void onReplayIndexChanged(int index, int total);
    void onReplayModeExited();
    /* 开始 / 停止 Agent 对 Agent 对弈 (按钮兼作"停止") */
    void onStartMatch();
    /* 对弈模式下拉框 (P0-a): 把"训练 / 评估 / 只对弈不学习"写进 ChessBoard */
    void onMatchModeSelected(int index);

private:
    void refreshGameList();
    void populateAgentComboBox();
    /*
     * 权重**静默**存到标准路径 (不弹任何窗口): 放在**常驻后台线程**里做。
     * 两个调用方: Agent 对弈结束 (onStartMatch 那条路) 与人机对局终局
     * (onHumanGameFinished) —— 两者共用同一套命名 (defaultWeightPath) 与同一个队列。
     */
    void saveWeightsAfterMatch(const QVector<ChessBoard::AgentType> &types);

    /*
     * ---- 指标面板 (右侧: 曲线 + 逐局明细) ----
     * 曲线的数据源是 ChessBoard 的 trainLossSample / gameRewardSample 信号;
     * 这里只负责把它们塞进 CurveChart, 并维护"agent 名 -> 曲线下标"的映射。
     */
    void setupMetricsPanel();
    void resetMetricsForMatch(const QString &agentA, const QString &agentB);
    int  lossSeriesFor(const QString &agentName);
    void exportMetricsCsv();
    /*
     * 只导出**训练损失曲线** (2026-09 用户要求"增加 loss 曲线导出控件")。
     * 为什么单列一个而不是让用户自己去切那个合并 CSV: 两张曲线的采样序号口径不同
     * (损失 = 每完成一次在线训练一个点; 奖励 = 每手一个点, 长度不同), 合并导出后
     * 想单独分析损失就得先手工切段; 而"哪一段是损失"靠的是注释行, 很容易切错。
     */
    void exportLossCsv();
    /* 导出实现 (两个按钮共用): 一条曲线一段, 带表头与注释行 */
    bool writeChartCsv(class CurveChart *chart, const QString &what,
                       const QString &sectionComment, const QString &defaultName);
    /* 把曲线的"最新值/均值/样本数"写进图下面的标签 (见 .cpp 的注释) */
    void updateMetricsLabels();
    /*
     * ---- 模型自检面板 (异步) ----
     * 数据源 ChessBoard::getAgentSelfCheck() -> AgentBase::selfCheckReport()。
     * 为什么需要它 (损失与自对弈胜率都答不了"值不值得继续训") 见 .cpp 里的长注释。
     *
     * **为什么必须异步**: 报告读的是常驻 agent 的内部状态, 而 ChessBoard 里那条
     * 路径会等 `m_agentMutex` —— 一次 PPO 决策约 3.2 s、一份 558 MB 权重的读写
     * 约 9 s。同步调用等于把这些秒数直接搬到 GUI 线程上 (面板每一手都要刷新一次),
     * 表现就是"界面卡死"。所以请求交给 m_selfCheckThread, 算完再用队列信号贴回面板。
     */
    void requestSelfCheckPanelUpdate(bool allAgents);
    /* worker 算完后的"上屏"步骤 (只在 GUI 线程执行) */
    void applySelfCheckPanel();
    /*
     * "全部模型自检": 把所有 agent 的自检报告**一次性**拼到一起写进同一个面板
     * (走的也是上面那个 worker, 因为每份报告要依次过 agent 锁)。
     * 为什么需要它: 单个 agent 的报告只能回答"我这一个模型有没有表示/口径问题",
     * 而真正好用的是**横向对比** —— 谁的动作编码有别名、谁能看见规则上下文、
     * 谁的权重文件没扫到, 排在一起一眼就分得出来。
     * 这是一次性快照: 下一手棋的自检刷新会切回"当前 agent"的视图。
     */
    void showAllAgentsSelfCheck();
    /*
     * 双击曲线 -> 弹一个放大的独立窗口 (见 metricsview.h 的 CurveChartDialog)。
     * 同一个源控件只保留一个窗口: 已经开着就抬到前面, 不再新开一个。
     */
    void openLargeChart(CurveChart *source, const QString &title);

    Ui::MainWindow *ui;
    /* 缓存当前加载的走法列表 */
    QVector<DBStep> m_currentReplaySteps;

    /* agent 名 -> 损失曲线下标 (同名复用, 见 lossSeriesFor) */
    QHash<QString, int> m_lossSeries;
    /* 本场对弈的 A/B 两条奖励曲线 (每次开赛时重建) */
    int m_rewardSeriesA = -1;
    int m_rewardSeriesB = -1;
    int m_metricsMatchNo = 0;   /* 用于给明细列表分组 (第几场) */
    /* 已打开的放大窗口 (源控件 -> 窗口); 关掉时自动从表里移除 */
    QHash<CurveChart *, CurveChartDialog *> m_largeCharts;

    /* 载入/保存模型权重时的"请稍候"弹窗 (见 busydialog.h) */
    BusyDialog *m_busy = nullptr;
    /*
     * "忙"的延迟显示: 读写很快时不弹窗(静默), 超过 300 ms 才把沙漏亮出来。
     * m_busyDelay 是那个单发定时器, m_busyPending 表示"后台确实还有活在跑"。
     */
    QTimer *m_busyDelay = nullptr;
    bool m_busyPending = false;
    QString m_busyTitle;
    QString m_busyMessage;

    /*
     * 后台线程用成员持有, 由析构函数 join。
     * 原来是 `std::thread(...).detach()`, 线程捕获 this 并访问 ui/ChessBoard ——
     * 用户关窗时线程可能还在跑, 于是访问已析构对象 (use-after-free)。
     */
    std::thread m_loadThread;
    std::thread m_selfPlayThread;
    /*
     * 保存权重 (常驻后台线程, 见 saveWeightsAfterMatch 的说明: 原来每场起一个线程再
     * 在 GUI 线程 join, 上一场还在写 558 MB 时界面会冻住).
     */
    std::thread m_saveThread;
    std::mutex m_saveMutex;
    std::condition_variable m_saveCv;
    bool m_saveStop = false;
    bool m_savePending = false;
    QVector<ChessBoard::AgentType> m_saveQueue;
    void saveWorkerLoop();

    /*
     * 自检面板的 worker (见 requestSelfCheckPanelUpdate 的说明)。
     * 它只有一个任务队列深度: "有没有新请求" + "最新一份算好的文本"。
     * 析构时必须 stop + join, 再 delete ui (worker 会访问 ui->gameWidget)。
     */
    std::thread m_selfCheckThread;
    std::mutex m_selfCheckMutex;
    std::condition_variable m_selfCheckCv;
    bool m_selfCheckStop = false;
    bool m_selfCheckPending = false;      /* 有新请求待处理 */
    bool m_selfCheckAll = false;          /* 这一份请求是"全部模型"还是"当前 agent" */
    ChessBoard::AgentType m_selfCheckType = ChessBoard::AGENT_ALPHABETA;
    /* worker 的产出 (上屏前暂存) */
    QString m_selfCheckText;
    bool m_selfCheckReady = false;
    void selfCheckWorkerLoop();

    /* Agent 对弈状态 (只在 GUI 线程读写) */
    bool m_matchRunning = false;
    QString m_matchLog;
    /*
       [④] 本场对弈双方的 agent 类型 —— **唯一**用途是给奖励曲线的名字加上口径标签
       (学习口径 / 引擎口径, 见 ChessBoard::agentRewardCaliperLabel)。
       为什么在本窗口存一份而不从 matchStarted 信号里取: 那个信号只带两个**名字**, 而
       口径是**类型的属性**; 主窗口本来就在 onStartMatch 里拿着两个类型。
    */
    ChessBoard::AgentType m_matchTypeA = ChessBoard::AGENT_ALPHABETA;
    ChessBoard::AgentType m_matchTypeB = ChessBoard::AGENT_ALPHABETA;
};

#endif // MAINWINDOW_H
