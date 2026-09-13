#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QHash>
#include <QTimer>
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

private:
    void refreshGameList();
    void populateAgentComboBox();
    /*
     * 对弈结束后把可训练 agent 的权重**静默**存到标准路径 (不弹任何窗口)。
     * 保存跑在后台线程, 结果写进逐局明细列表; 期间由"请稍候"沙漏提示。
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
    /* 把曲线的"最新值/均值/样本数"写进图下面的标签 (见 .cpp 的注释) */
    void updateMetricsLabels();
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
    /* 保存权重 (放在后台线程做, 见 saveWeightsAfterMatch) */
    std::thread m_saveThread;

    /* Agent 对弈状态 (只在 GUI 线程读写) */
    bool m_matchRunning = false;
    QString m_matchLog;
};

#endif // MAINWINDOW_H
