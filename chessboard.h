#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QMutex>
#include <QWaitCondition>
#include <QtConcurrent>
#include "chess.h"

class ChessBoard : public QWidget
{
    Q_OBJECT
public:
    enum State {
        STATE_IDEL = 0,
        STATE_READY,
        STATE_TERMINATE
    };
public:
    explicit ChessBoard(QWidget *parent = nullptr);
    ~ChessBoard();

signals:
    void sendResult(int color);
public slots:
    void checkGameOver(int result);
    void reset();
private:
    Pos getStonePos(const QPoint &pos);
    QPoint getStoneCenter(int x, int y);
    QRect getRect(QPoint &center);
    void drawStone(QPainter &p, const Stone *stone);
    Stone *selectStone(const QPoint &point);
    bool moveStone(const QPoint &point);
    void process();
protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
private:
    constexpr static int offsetX = 50;
    constexpr static int offsetY = 50;
    constexpr static int gridSize = 60;
    constexpr static int stoneRadius = 24;
    Chess chess;
    int selectID;
    int color;
    State state;
    QMutex mutex;
    QWaitCondition condit;
    QFuture<void> processFuture;
};

#endif // CHESSBOARD_H
