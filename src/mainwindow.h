#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include "gamedb.h"
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
    void onSelfPlay();

private:
    void refreshGameList();
    void populateAgentComboBox();

    Ui::MainWindow *ui;
    /* 缓存当前加载的走法列表 */
    QVector<DBStep> m_currentReplaySteps;
};

#endif // MAINWINDOW_H
