#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "chessboard.h"
#include "qssloader.hpp"
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>
#include <QComboBox>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(900, 650);

    /* 填充AI Agent下拉框 */
    populateAgentComboBox();

    /* Agent选择 */
    connect(ui->agentComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onAgentSelected);

    /* 开局按钮 */
    connect(ui->resetBtn, &QPushButton::clicked,
            ui->gameWidget, &ChessBoard::reset);

    /* AI Self Play 按钮 */
    connect(ui->selfPlayBtn, &QPushButton::clicked,
            this, &MainWindow::onSelfPlay);

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
            ui->recordcomboBox->setEnabled(true);
            ui->timeLabel->setText("AI思考时间: -");
            ui->gameWidget->setEnabled(true);
            refreshGameList();
        });

    /* ---- 启动加载阶段: 禁用所有交互控件 ---- */
    ui->resetBtn->setEnabled(false);
    ui->selfPlayBtn->setEnabled(false);
    ui->agentComboBox->setEnabled(false);
    ui->recordcomboBox->setEnabled(false);
    ui->prevBtn->setEnabled(false);
    ui->nextBtn->setEnabled(false);
    ui->stepLabel->setText("步数: -/-");
    ui->timeLabel->setText("正在加载...");

    /* 在后台线程启动异步加载 (数据库 + AI模型权重) */
    ui->gameWidget->setEnabled(false);
    std::thread loadThread([this]() {
        ui->gameWidget->startupLoad();
    });
    loadThread.detach();
}

MainWindow::~MainWindow()
{
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
    ui->agentComboBox->clear();
    ui->agentComboBox->addItem("Alpha-Beta Pruning (深度=4)",
                                ChessBoard::AGENT_ALPHABETA);
    ui->agentComboBox->addItem("MCTS (800次模拟)",
                                ChessBoard::AGENT_MCTS);
    ui->agentComboBox->addItem("Policy Gradient (PGEagent)",
                                ChessBoard::AGENT_PG);
    ui->agentComboBox->addItem("Deep Q-Network (DQN)",
                                ChessBoard::AGENT_DQN);
    ui->agentComboBox->addItem("PPO+MCTS (AlphaZero)",
                                ChessBoard::AGENT_PPOMCTS);
    ui->agentComboBox->addItem("DQN+MCTS (AlphaZero)",
                                ChessBoard::AGENT_DQNMCTS);
    /* 默认选中 Alpha-Beta */
    ui->agentComboBox->setCurrentIndex(0);
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
 *  onSelfPlay - AI Self Play 按钮回调
 *
 *  触发 AI vs AI 对弈, 对局结束后:
 *    1. 显示结果对话框
 *    2. 保存当前 agent 的权重文件
 * ================================================================ */
void MainWindow::onSelfPlay()
{
    ChessBoard *board = ui->gameWidget;
    ChessBoard::AgentType agentType = board->getAgentType();

    /* 禁用按钮, 防止重复点击 */
    ui->selfPlayBtn->setEnabled(false);
    ui->selfPlayBtn->setText("Self Play 进行中...");

    /* 在后台线程中执行 self play, 避免阻塞 UI */
    std::thread selfPlayThread([this, board, agentType]() {
        /* 执行 self play */
        int result = board->selfPlay(agentType);

        /* 回到主线程处理结果 */
        QMetaObject::invokeMethod(this, [this, result, agentType, board]() {
            /* 更新按钮状态 */
            ui->selfPlayBtn->setEnabled(true);
            ui->selfPlayBtn->setText("AI Self Play");

            /* ---- 1. 显示结果 ---- */
            QString resultStr;
            switch (result) {
            case Stone::COLOR_RED:
                resultStr = "红方胜 (AI vs AI)";
                break;
            case Stone::COLOR_BLACK:
                resultStr = "黑方胜 (AI vs AI)";
                break;
            default:
                resultStr = "平局 (AI vs AI)";
                break;
            }
            QMessageBox::information(this, "Self Play 结束", resultStr);

            /* ---- 2. 保存 agent 权重文件 ---- */
            QString agentName;
            QString defaultFilename;
            switch (agentType) {
            case ChessBoard::AGENT_PG:
                agentName = "Policy Gradient";
                defaultFilename = "pg_agent_weights.dat";
                break;
            case ChessBoard::AGENT_DQN:
                agentName = "DQN";
                defaultFilename = "dqn_agent_weights.dat";
                break;
            case ChessBoard::AGENT_PPOMCTS:
                agentName = "PPO+MCTS";
                defaultFilename = "ppomcts_actor_weights.dat";
                break;
            case ChessBoard::AGENT_DQNMCTS:
                agentName = "DQN+MCTS";
                defaultFilename = "dqnmcts_agent_weights.dat";
                break;
            default:
                /* Alpha-beta 和 MCTS 没有权重文件 */
                QMessageBox::information(this, "提示",
                    QString("Self Play 结束: %1\n(%2 无需保存权重)")
                        .arg(resultStr).arg(ui->agentComboBox->currentText()));
                return;
            }

            /* 弹出保存对话框 */
            QString filePath = QFileDialog::getSaveFileName(
                this,
                QString("保存 %1 权重文件").arg(agentName),
                defaultFilename,
                "权重文件 (*.dat);;所有文件 (*.*)");

            if (!filePath.isEmpty()) {
                bool ok = board->saveCurrentAgentModel(agentType, filePath.toStdString());
                if (ok) {
                    QMessageBox::information(this, "保存成功",
                        QString("权重已保存到:\n%1").arg(filePath));
                } else {
                    QMessageBox::warning(this, "保存失败",
                        "保存权重文件时发生错误!");
                }
            }

            /* 刷新对局列表 */
            refreshGameList();
        }, Qt::QueuedConnection);
    });
    selfPlayThread.detach();
}
