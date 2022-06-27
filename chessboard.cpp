#include "chessboard.h"

ChessBoard::ChessBoard(QWidget *parent) :
    QWidget(parent),
    selectID(Stone::ID_NONE),
    state(STATE_IDEL)
{
    setMinimumSize(offsetX*2 + Stone::map.row*gridSize,
                   offsetY*2 + Stone::map.col*gridSize);

    connect(this, &ChessBoard::sendResult,
            this, &ChessBoard::checkGameOver, Qt::QueuedConnection);
    processFuture = QtConcurrent::run(this, &ChessBoard::process);
    //chess.reset();
#if 0
    std::cout<<"size:"<<chess.stones.size()<<std::endl;
    for (Stone *stone : chess.stones) {
        if (stone != nullptr) {
            stone->show();
        }
    }
    Stone::map.show();
#endif
}

ChessBoard::~ChessBoard()
{
    {
        QMutexLocker locker(&mutex);
        state = STATE_TERMINATE;
        condit.wakeAll();
    }
    processFuture.waitForFinished();
}

Pos ChessBoard::getStonePos(const QPoint &pos)
{
    Pos gridPos(-1, -1);
    for (size_t i= 0; i <= Stone::map.col; i++) {
        for (size_t j = 0; j <= Stone::map.row; j++) {
            QPoint center = getStoneCenter(i, j);
            int dx = center.x() - pos.x();
            int dy = center.y() - pos.y();
            int l2 = dx*dx + dy*dy;
            if (l2 < stoneRadius*stoneRadius) {
                gridPos = Pos(i, j);
                break;
            }
        }
    }
    return gridPos;
}

QPoint ChessBoard::getStoneCenter(int x, int y)
{
    return QPoint(offsetX + y*gridSize, offsetY + x*gridSize);
}

QRect ChessBoard::getRect(QPoint &center)
{
    QPoint offset(gridSize, gridSize);
    return QRect(center - offset, center + offset);
}

void ChessBoard::drawStone(QPainter &painter, const Stone *stone)
{
    if (stone->alive == false) {
        return;
    }
    //获取圆心
    painter.setBrush(Qt::yellow);
    QPoint center = getStoneCenter(stone->pos.x, stone->pos.y);
    if (stone->id == selectID && stone->color == Stone::COLOR_RED) {
        painter.setBrush(Qt::gray);
    }
    painter.setPen(Qt::black);
    //画圆
    painter.drawEllipse(center, stoneRadius, stoneRadius);
    //画字
    QRect rect = getRect(center);
    if (stone->color == Stone::COLOR_BLACK) {
        painter.setPen(Qt::black);
    } else {
        painter.setPen(Qt::red);
    }
    painter.setFont(QFont("宋体", stoneRadius/2));
    painter.drawText(rect, stone->name.c_str(), QTextOption(Qt::AlignCenter));
    return;
}

Stone* ChessBoard::selectStone(const QPoint &point)
{
    Pos pos = getStonePos(point);
    if (Stone::map.isInner(pos) == false) {
        return nullptr;
    }
    return Stone::map[pos];
}

bool ChessBoard::moveStone(const QPoint &point)
{
    Pos pos = getStonePos(point);
    if (Stone::map.isInner(pos) == false) {
        return false;
    }
    Stone *stone = chess.get(selectID);
    if (stone == nullptr) {
        return false;
    }
    return stone->moveTo(pos);
}

void ChessBoard::checkGameOver(int resault)
{
    if (resault == Stone::COLOR_RED) {
        QMessageBox::information(this, "Game Over", "红方胜");
    } else if(resault == Stone::COLOR_BLACK) {
        QMessageBox::information(this, "Game Over", "黑方胜");
    }
    return;
}

void ChessBoard::reset()
{
    /* reset */
    chess.reset();
    update();
    return;
}

void ChessBoard::process()
{
    while (1) {
        {
            QMutexLocker locker(&mutex);
            condit.wait(&mutex);
            if (state == STATE_TERMINATE) {
                break;
            }
        }
        /* 黑方ai */
        Step step = chess.alphaBetaPruning(Stone::COLOR_BLACK, 5);
        Stone *stone = Stone::children[step.id];
        stone->moveTo(step.nextPos);
        /* 游戏是否结束 */
        emit sendResult(chess.isGameOver());
        {
            QMutexLocker locker(&mutex);
            state = STATE_IDEL;
            selectID = Stone::ID_NONE;
            update();
        }

    }
    return;
}

void ChessBoard::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    //画棋盘
    /* 横线 */
    for (size_t i = 0; i < Stone::map.row; i++) {
        painter.drawLine(QPoint(offsetX,i*gridSize+offsetY),QPoint(8*gridSize+offsetX,i*gridSize+offsetY));
    }
    /* 竖线 */
    for (size_t i = 0; i < Stone::map.col; i++) {
        if (i == 0 || i == 8) {
            painter.drawLine(QPoint(offsetX + i*gridSize, offsetY), QPoint(i*gridSize + offsetX, 9*gridSize + offsetY));
        } else {
            painter.drawLine(QPoint(offsetX + i*gridSize, offsetY), QPoint(i*gridSize + offsetX, 4*gridSize + offsetY));
            painter.drawLine(QPoint(offsetX + i*gridSize, 5*gridSize+offsetY), QPoint(i*gridSize + offsetX, 9*gridSize + offsetY));
        }
    }
    /* 九宫线 */
    painter.drawLine(QPoint(offsetX + 3*gridSize, offsetY), QPoint(5*gridSize + offsetX, 2*gridSize + offsetY));
    painter.drawLine(QPoint(offsetX + 3*gridSize, 2*gridSize + offsetY), QPoint(5*gridSize + offsetX, offsetY));
    painter.drawLine(QPoint(offsetX + 3*gridSize, 7*gridSize + offsetY), QPoint(5*gridSize + offsetX, 9*gridSize + offsetY));
    painter.drawLine(QPoint(offsetX + 3*gridSize, 9*gridSize + offsetY), QPoint(5*gridSize + offsetX, 7*gridSize + offsetY));
    /* 棋子 */
    for (Stone *stone : chess.stones) {
        if (stone != nullptr) {
            drawStone(painter, stone);
        }
    }
    return;
}

void ChessBoard::mousePressEvent(QMouseEvent *event)
{
    {
        QMutexLocker locker(&mutex);
        if (state == STATE_READY || state == STATE_TERMINATE) {
            return;
        }
    }
    if (selectID == Stone::ID_NONE) {
        /* 选择棋子 */
        Stone *stone = selectStone(event->pos());
        if (stone == nullptr) {
            return;
        }
        selectID = stone->id;
    } else {
        /* 移动棋子 */
        if (moveStone(event->pos()) == false) {
            selectID = Stone::ID_NONE;
            return;
        }
        /* AI */
        QTimer::singleShot(500, Qt::PreciseTimer, this, [=](){
            QMutexLocker locker(&mutex);
            state = STATE_READY;
            condit.wakeAll();
        });
    }
    update();
    return;
}
