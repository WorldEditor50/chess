#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <thread>
#include "gamedb.h"
#include "chessboard.h"
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
    /* 对弈结束后, 询问是否把某个可训练 agent 的权重存盘 */
    void offerSaveWeights(ChessBoard::AgentType type);

    Ui::MainWindow *ui;
    /* 缓存当前加载的走法列表 */
    QVector<DBStep> m_currentReplaySteps;

    /*
     * 后台线程用成员持有, 由析构函数 join。
     * 原来是 `std::thread(...).detach()`, 线程捕获 this 并访问 ui/ChessBoard ——
     * 用户关窗时线程可能还在跑, 于是访问已析构对象 (use-after-free)。
     */
    std::thread m_loadThread;
    std::thread m_selfPlayThread;

    /* Agent 对弈状态 (只在 GUI 线程读写) */
    bool m_matchRunning = false;
    QString m_matchLog;
};

#endif // MAINWINDOW_H
