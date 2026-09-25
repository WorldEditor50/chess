/*
 * probe_hvai_flow_main.cpp - 人机对弈整条 UI 状态机的回归探针 (2026-09 用户报障)
 * ============================================================================
 *
 * 用户报障 (原文):
 *   "人机对弈时黑方胜利后, 沙漏显示 agent 未加载, 选择黑方对弈 agent 失效,
 *    红方下第一个棋后黑方无限等待"
 *
 * 这条报障的全部现象都属于**同一块状态机**: `ChessBoard::process()` 是人机对弈里
 * 唯一的"黑方应手"线程, 而它的主循环写的是
 *     while (state != STATE_TERMINATE) { ... }
 * 于是**只要有一局是在 process() 内部结束的** (AI 自己走出将杀/困毙/判和),
 * 那个 continue 之后循环条件立刻为假 ⇒ 线程函数 return ⇒ **应手线程永久消失**。
 * 之后:
 *   * 按"开局" -> reset() 只把 state 改回 IDEL, 没有任何人重新起线程;
 *   * 红方落子 -> state=THINKING + wakeAll, 但**没有等待者**;
 *   * 于是"黑方无限等待"; 换 agent 下拉框也不会有人用新 agent 去应手
 *     (用户说的"选择黑方对弈 agent 失效");
 *   * 沙漏 (ThinkingIndicator) 从此再也收不到 aiThinkingStarted, 而按"开局"时
 *     resetToIdle() 会把 agent 名清空 ⇒ 那一行显示 "(未选择 agent)"
 *     (用户说的"沙漏显示 agent 未加载")。
 *
 * 这个探针把这些**用真实点击路径**钉住 (它不进 ctest 默认集, 见 CMakeLists):
 *
 *   [A] 被将军时的走法过滤到底允许什么 (走 mousePressEvent -> moveStone 真实链路)。
 *       用户口径: 这个问题**不改** —— 这一节的作用是把"规则本来允许挡将/吃将军的子/
 *       走帅"变成事实, 免得下次又把规则问题当 bug 查。
 *
 *   [B] 一局在 process() 里结束之后, 下一局黑方还会不会应手 (用户报障的核心)。
 *         * 摆一个"红方只剩帅、黑方车马"的局面, 让 **AI 自己**把这一局结束掉
 *           (红方由探针代打: 每次走引擎给的第一手合法着法, 直到终局);
 *         * 按"开局" (board.reset());
 *         * 红方再走第一步 -> 黑方必须应手 (aiThinkFinished 必须到);
 *         * 再换一次"对战 AI"下拉框 (AGENT_AB_L2), 重复一遍 —— 用户报的
 *           "选择黑方对弈 agent 失效"也在这里。
 *       修复前: 第一局结束后应手线程已经消失 => 后面两节全部超时 (FAIL)。
 *
 * 用法: <build>/probe_hvai_flow
 */
#include <QApplication>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QMouseEvent>
#include <QObject>
#include <QTimer>

#include <chrono>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

#include "chessboard.h"
#include "thinkingindicator.h"

namespace {

int gFailed = 0;

void check(bool ok, const char *what)
{
    std::printf("  [%s] %s\n", ok ? " OK " : " !! ", what);
    if (!ok) {
        ++gFailed;
    }
}

/*
   棋盘坐标 -> 控件坐标。
   ChessBoard 的几何是**私有常量**, 但它们是写死的 (offsetX/offsetY = 50,
   gridSize = 60; 见 chessboard.h), 而这里要走的正是"用户点在棋盘上"那条路,
   所以只能照抄这两个数 (改了几何这里会立刻失败 —— 那是好事)。
*/
QPointF widgetPoint(int x, int y)
{
    return QPointF(50.0 + y * 60.0, 50.0 + x * 60.0);
}

/* 一次左键点击 (真正投递给 ChessBoard::mousePressEvent) */
void clickAt(ChessBoard &board, int x, int y)
{
    const QPointF p = widgetPoint(x, y);
    QMouseEvent ev(QEvent::MouseButtonPress, p, p,
                   Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&board, &ev);
}

/* 走一步棋 (先点起点选中, 再点终点) —— 与玩家操作逐字相同 */
void playMove(ChessBoard &board, int fx, int fy, int tx, int ty)
{
    clickAt(board, fx, fy);
    clickAt(board, tx, ty);
}

Stone *at(ChessBoard &board, int x, int y)
{
    return board.boardForTest().m_map[Pos(x, y)];
}

bool alive(ChessBoard &board, int id)
{
    Stone *s = board.boardForTest().m_children[id];
    return s != nullptr && s->alive != 0;
}

/*
   摆局面: 清空棋盘, 只放下给出的那些子。
   这样做 (而不是造一个合法开局再走到目标局面) 是因为被测的是**状态机**, 不是棋谱;
   局面只需要在规则上自洽: 双方将帅都在, 轮到红方, 半回合计数指定。
*/
void rigBoard(ChessBoard &board, const std::vector<std::pair<int, Pos>> &pieces,
              int halfMoveClock)
{
    Chess &c = board.boardForTest();
    for (std::size_t i = 0; i < c.m_children.size(); ++i) {
        if (c.m_children[i] != nullptr) {
            c.m_children[i]->alive = 0;
        }
    }
    c.m_map.clear();
    c.history.clear();
    c.halfMoveClock = halfMoveClock;
    c.sideToMove = Stone::COLOR_RED;
    for (const std::pair<int, Pos> &p : pieces) {
        Stone *s = c.m_children[p.first];
        s->alive = 1;
        s->pos = p.second;
        c.m_map[p.second] = s;
    }
}

/*
   ---- 黑方应手的判据: **棋盘上黑方的子动了** ----
   为什么不用 aiThinkFinished 信号: 它当然是对的判据, 但本探针是**控制台程序**,
   没有界面那个常驻事件循环; 实测在"等的那一刻才建立的连接"上收不到那一次 emit
   (同一时刻常驻连接能收到, 见本文件 commit 记录里的实测), 而那属于探针的观察手段
   问题 —— 界面用的就是常驻连接, 不受影响。
   而"黑方到底动没动"正是用户能看见的那件事 (报障原文: "黑方无限等待"), 所以这里
   直接看盘面: 与等待开始时的快照比, 只要有一个黑子换了位置/死了/多出来了, 就算应手。
   读盘面没有拿 ChessBoard 的 mutex (它是 private), 对"有没有动"这种粗判据是够的;
   这不是产品代码, 只是探针的观察手段。
*/
struct BlackSnapshot {
    int n = 0;
    int id[32] = {0};
    Pos pos[32];
    bool operator==(const BlackSnapshot &o) const
    {
        if (n != o.n) {
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (id[i] != o.id[i] || !(pos[i] == o.pos[i])) {
                return false;
            }
        }
        return true;
    }
};

BlackSnapshot snapshotBlack(ChessBoard &board)
{
    BlackSnapshot s;
    const Chess &c = board.boardForTest();
    for (int i = Stone::ID_BLACK; i < Stone::ID_BLACK_END && s.n < 32; ++i) {
        const Stone *st = c.m_children[i];
        if (st == nullptr || st->alive == 0) {
            continue;
        }
        s.id[s.n] = st->id;
        s.pos[s.n] = st->pos;
        s.n++;
    }
    return s;
}

/*
   等黑方应手 (轮询盘面)。返回 true = 动了; false = 超时
   —— 超时就是用户报障里的"黑方无限等待"。

   ⚠ 快照必须在**红方落子之前**取 (见 playRedThenWaitAi 的说明)。
*/
bool waitForBlackChange(ChessBoard &board, const BlackSnapshot &before, int timeoutMs,
                        long long *elapsedOut)
{
    QElapsedTimer clock;
    clock.start();
    while (clock.elapsed() < timeoutMs) {
        QApplication::processEvents(QEventLoop::AllEvents, 10);
        if (!(snapshotBlack(board) == before)) {
            if (elapsedOut != nullptr) {
                *elapsedOut = clock.elapsed();
            }
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

/*
   让上一手棋带起的 AI 思考彻底跑完 (否则它落地时会与下一次摆局面打架)。

   ⚠ processEvents **一定要给 maxtime**: 不带 maxtime 的那个重载会"一直处理到没有
     待处理事件为止", 而棋盘在思考期间有 40 ms 的动画定时器在不停地投递重绘事件
     ⇒ 它会一直转下去 (实测: 每手从 16 ms 变成 ~5 s, 一局 80 手要几分钟才跑完,
     看起来就像探针卡死)。
*/
void quiesce(ChessBoard &board, int ms)
{
    (void)board;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    QApplication::processEvents(QEventLoop::AllEvents, 10);
}

/*
   红方(玩家)代打一手 + 等黑方应手。返回 false = 红方那一步没走成 / 黑方没应手。

   三个细节都是踩过的坑:
   1. **快照在点击之前取**: 这个残局里 AB 一步只要 ~4 ms, 而探针的"玩家"是瞬时落子
      —— 先点击再取快照的话, AI 那一步可能已经落地了, 于是"等变化"永远等不到
      (第一版就是这么假超时的: 起点快照里黑车已经在 (8,0) 了)。
   2. 红方那一步走没走成要**当场确认** (棋盘真的动了), 否则后面整局都歪。
   3. 走不成时**重试几次**: `mousePressEvent` 一开始的 `state` 检查是无锁读的, 玩家
      恰好点在"AI 刚落子、工作线程还没回到 IDLE"那一瞬, 点击会被**静默丢弃**
      (这是产品里早就存在的窄窗口, 不是本探针引入的)。重试 3 次、每次隔 30 ms:
      真·卡死 (state 永远不回到 IDLE) 时三次都失败, 判据依然成立。
*/
bool playRedThenWaitAi(ChessBoard &board, int timeoutMs, long long *elapsedOut)
{
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::vector<Step *> legal;
        board.boardForTest().sample(Stone::COLOR_RED, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            return false;                    /* 红方真的没棋可走 */
        }
        const Step s = *legal[0];
        Steps::instance().put(legal);

        Stone *mover = board.boardForTest().m_children[s.id];
        const BlackSnapshot before = snapshotBlack(board);   /* 点击之前 */
        playMove(board, s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y);
        if (mover != nullptr && mover->alive != 0 && mover->pos == s.nextPos) {
            return waitForBlackChange(board, before, timeoutMs, elapsedOut);
        }
        quiesce(board, 30);                  /* 让工作线程先回到 IDLE, 再重试 */
    }
    return false;                            /* 红方那一步三次都没走成 */
}

/* 盘面快照 (断言挂了的时候靠它看"到底哪一步没发生") */
void dumpBoard(ChessBoard &board, const char *title)
{
    std::printf("    %s\n", title);
    const Chess &c = board.boardForTest();
    for (std::size_t i = 0; i < c.m_children.size(); ++i) {
        const Stone *s = c.m_children[i];
        if (s == nullptr || s->alive == 0) {
            continue;
        }
        std::printf("      %s %-6s id=%2d (%d,%d)\n",
                    s->color == Stone::COLOR_RED ? "红" : "黑",
                    s->name.c_str(), s->id, s->pos.x, s->pos.y);
    }
}

}  // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    ChessBoard board;
    /*
       终局弹窗 (checkGameOver -> QMessageBox) 是 **exec() 起的模态嵌套事件循环**:
       探针没法点它, 一旦弹出来整个探针就挂在那里 (实测: 第一局结束的那一刻)。
       所以必须挂一个"关掉模态窗"的看门狗定时器 (定时器在嵌套循环里照样会触发)。
       ⚠ 这里**不能**只靠 `QObject::disconnect(&board, SIGNAL(sendResult(int)), &board,
         SLOT(checkGameOver(int)))`: ctor 里那条连接是用**新语法 + QueuedConnection**
         建的, 实测这种混用断开返回 false (连接根本不断), 弹窗照旧 —— 探针第一版就是
         这么被挂住的 (见日志里的 "[watchdog] 关掉模态窗口: 游戏结束")。
    */
    auto *modalCloser = new QTimer(&board);
    QObject::connect(modalCloser, &QTimer::timeout, &board, []() {
        QWidget *w = QApplication::activeModalWidget();
        if (w != nullptr) {
            std::printf("      [watchdog] 关掉模态窗口: %s\n",
                        w->windowTitle().toUtf8().constData());
            w->close();
        }
    });
    modalCloser->start(50);

    board.setAgentType(ChessBoard::AGENT_ALPHABETA);   /* 快 (~90 ms/手), 且不依赖权重 */
    board.setPreTrainEnabled(false);

    std::printf("=== 人机对弈状态机探针 (用户报障: 黑方胜利后黑方无限等待) ===\n");
    std::printf("QT_QPA_PLATFORM=%s\n", qgetenv("QT_QPA_PLATFORM").constData());

    /* ================================================================
     *  [A] 被将军时的走法过滤 (用户口径: 这个问题不改, 只把事实钉住)
     * ================================================================ */
    std::printf("\n[A] 红方被将军时, 真实点击路径允许/拒绝哪些着法\n");
    {
        /*
           局面: 红帅 (9,4); 黑车 (7,4) 同列 => 红方被将军。
                 红车1 (7,0) 可以**吃掉**将军的车;
                 红车2 (8,0) 可以**垫**到 (8,4);
                 红兵  (6,4) 与这次将军无关 (挪它不解决将军)。
                 黑将 (0,4) 放上是为了局面自洽 (两将不能照面: 中间有黑车)。
        */
        const std::vector<std::pair<int, Pos>> pos = {
            { Stone::ID_RED_JIANG,   Pos(9, 4) },
            { Stone::ID_RED_CHE1,    Pos(7, 0) },
            { Stone::ID_RED_CHE2,    Pos(8, 0) },
            { Stone::ID_RED_BING3,   Pos(6, 4) },
            { Stone::ID_BLACK_JIANG, Pos(0, 4) },
            { Stone::ID_BLACK_CHE1,  Pos(7, 4) },
        };
        board.reset();
        quiesce(board, 400);

        rigBoard(board, pos, 0);
        check(board.boardForTest().isInCheck(Stone::COLOR_RED),
              "摆好后红方**确实**被将军 (isInCheck(红) == true)");

        {   /* 引擎面: 合法着法里应当**同时**有解将着法, 且不含挪兵 */
            std::vector<Step *> legal;
            board.boardForTest().sample(Stone::COLOR_RED, legal);
            bool blockOk = false;
            bool captureOk = false;
            bool unrelatedOk = false;
            for (Step *s : legal) {
                if (s == nullptr) {
                    continue;
                }
                const Pos n = s->nextPos;
                if (n == Pos(8, 4)) { blockOk = true; }
                if (n == Pos(7, 4)) { captureOk = true; }
                if (s->id == Stone::ID_RED_BING3) { unrelatedOk = true; }
            }
            std::printf("    红方合法着法数 = %d\n", (int)legal.size());
            Steps::instance().put(legal);
            check(blockOk,     "垫将 (8,4) 在合法着法里");
            check(captureOk,   "吃将军的车 (7,4) 在合法着法里");
            check(!unrelatedOk, "挪无关的兵**不**在合法着法里 (这是象棋规则)");
        }

        /* 真实点击: 挪无关的兵 -> 必须走不动 */
        playMove(board, 6, 4, 5, 4);
        check(at(board, 6, 4) != nullptr && at(board, 6, 4)->id == Stone::ID_RED_BING3,
              "[点击] 被将军时挪无关的兵被拒 (兵还在 (6,4))");
        check(at(board, 5, 4) == nullptr, "[点击] (5,4) 仍然是空的");

        /* 真实点击: 车垫到将前 -> 必须走成 */
        playMove(board, 8, 0, 8, 4);
        check(at(board, 8, 4) != nullptr && at(board, 8, 4)->id == Stone::ID_RED_CHE2,
              "[点击] 车垫将 (8,0)->(8,4) 被接受 —— 解将着法能走");
        quiesce(board, 600);

        /* 真实点击: 吃掉将军的车 -> 必须走成 */
        board.reset();
        quiesce(board, 400);
        rigBoard(board, pos, 0);
        playMove(board, 7, 0, 7, 4);
        check(at(board, 7, 4) != nullptr && at(board, 7, 4)->id == Stone::ID_RED_CHE1,
              "[点击] 车吃将军的车 (7,0)->(7,4) 被接受");
        check(!alive(board, Stone::ID_BLACK_CHE1), "[点击] 黑车确实被吃掉了 (alive=0)");
        quiesce(board, 600);

        /* 真实点击: 帅走出被攻击的列 -> 必须走成 */
        board.reset();
        quiesce(board, 400);
        rigBoard(board, pos, 0);
        playMove(board, 9, 4, 9, 3);
        check(at(board, 9, 3) != nullptr && at(board, 9, 3)->id == Stone::ID_RED_JIANG,
              "[点击] 帅走出被攻击的列 (9,4)->(9,3) 被接受");
        quiesce(board, 600);
    }

    /* ================================================================
     *  [B] 一局由 AI 的落子结束之后, 按"开局"再走第一步 (报障核心)
     * ================================================================ */
    std::printf("\n[B] 一局由 AI 的落子结束之后, 按\"开局\"再走第一步 (报障核心)\n");
    {
        /*
           局面 (用户报障的现场: "黑方胜利"):
             红: 只有帅 (9,4)
             黑: 将 (1,3)、车 (0,0)、马 (6,2)、马 (6,6)
           红方只有帅、黑方有车马 => AI 会把红方将死 (实测黑车一步 (0,0)->(0,4) 就是杀;
           而 AB 对**任何**将杀都给 value_infi, 所以它可能选一步更长的杀 —— 因此
           下面不假设"AI 一步杀", 而是**把整局打完**: 红方由探针代打, 直到 sendResult
           到达)。关键点: 这一局是在 `ChessBoard::process()` **内部**结束的 ——
           也就是用户报障里那条路径 (终局分支里的 continue)。
        */
        const std::vector<std::pair<int, Pos>> pos = {
            { Stone::ID_RED_JIANG,   Pos(9, 4) },
            { Stone::ID_BLACK_JIANG, Pos(1, 3) },
            { Stone::ID_BLACK_CHE1,  Pos(0, 0) },
            { Stone::ID_BLACK_MA1,   Pos(6, 2) },
            { Stone::ID_BLACK_MA2,   Pos(6, 6) },
        };

        board.reset();
        quiesce(board, 400);

        /* ---- [摆局面自检] 先确认这个摆法在规则上自洽 ---- */
        {
            Chess &c = board.boardForTest();
            rigBoard(board, pos, 0);
            check(!c.isInCheck(Stone::COLOR_RED), "[摆局面自检] 红方没有被将军");
            std::vector<Step *> legal;
            c.sample(Stone::COLOR_RED, legal);
            const bool redCanMove = !legal.empty();
            Steps::instance().put(legal);
            check(redCanMove, "[摆局面自检] 红方有合法着法 (探针代打的每一步都从这里取)");
        }

        int firstResult = Chess::RESULT_ONGOING;
        bool sawResult = false;
        QMetaObject::Connection resConn =
            QObject::connect(&board, &ChessBoard::sendResult, &board,
                             [&](int r) { firstResult = r; sawResult = true; });

        board.reset();
        quiesce(board, 400);
        rigBoard(board, pos, 0);

        /* ---- 把这一局打完: 红方代打 -> 黑方应手 -> ... 直到 sendResult 到达 ---- */
        int plies = 0;
        bool aiAlwaysReplied = true;
        QElapsedTimer loopClock;
        loopClock.start();
        for (plies = 0; plies < 80 && !sawResult; ++plies) {
            long long aiMs = -1;
            const qint64 t0 = loopClock.elapsed();
            if (!playRedThenWaitAi(board, 20000, &aiMs)) {
                aiAlwaysReplied = false;
                break;
            }
            std::printf("    [ply %d] 黑方应手耗时 %.0f ms (本手共 %.0f ms, 累计 %.0f ms)\n",
                        plies, (double)aiMs, (double)(loopClock.elapsed() - t0),
                        (double)loopClock.elapsed());
            quiesce(board, 30);
        }
        check(aiAlwaysReplied, "第一局: 整局里 AI 每次都应手了");
        check(sawResult, "第一局: 收到 sendResult —— 这一局是**AI 的落子**结束的");
        check(firstResult != Chess::RESULT_ONGOING, "第一局: 结果不是 ONGOING (确实终局了)");
        std::printf("    第一局: 共走 %d 手结束, 结果 = %d (1=红胜 2=黑胜 3=和)\n",
                    plies, firstResult);
        dumpBoard(board, "第一局结束时的盘面:");

        /* ---- 用户操作: 按"开局", 再走红方第一步 (本轮的第 2/3 次) ---- */
        for (int round = 0; round < 2; ++round) {
            board.reset();
            quiesce(board, 400);
            if (round == 1) {
                /* 用户报障里的另一半: "选择黑方对弈 agent 失效" */
                board.setAgentType(ChessBoard::AGENT_AB_L2);
            }
            rigBoard(board, pos, 0);
            check(at(board, 9, 4) != nullptr && at(board, 9, 4)->id == Stone::ID_RED_JIANG,
                  "开局后棋盘已重置 (帅回到 (9,4))");

            long long ms = -1;
            const bool replied = playRedThenWaitAi(board, 15000, &ms);
            check(replied, round == 0
                  ? "第二局: 红方走完第一步后黑方应手了 —— 用户报障的\"黑方无限等待\"就是这一条"
                  : "换对弈 agent 之后: 黑方仍然应手 —— 用户报障的\"选择 agent 失效\"就是这一条");
            if (replied) {
                std::printf("    这一手 AI 应手耗时 %.0f ms\n", (double)ms);
            } else {
                dumpBoard(board, "黑方没有应手时的盘面:");
            }
        }
        QObject::disconnect(resConn);
    }

    /* ================================================================
     *  [C] 沙漏上那一行 agent 名 (用户报障: "沙漏显示 agent 未加载")
     * ================================================================
     *
     * 报障的现象是那一行写着 "(未选择 agent)"。根因: `resetToIdle()` (按"开局"时调用)
     * 会把 agent 名清空, 而界面上其实选着一个 agent —— 两处说法不一致。
     * 这一节钉住指示器自己的语义 (界面把选中项喂进来的那 3 行在 mainwindow.cpp):
     *   * 喂进来之后, 空闲/开局/思考结束都显示它;
     *   * 思考期间显示的是"正在想的那个" (start() 的名字), 不会被配置项盖掉。
     */
    std::printf("\n[C] 沙漏上那一行 agent 名\n");
    {
        ThinkingIndicator ind;
        check(ind.displayedAgent().isEmpty(),
              "从没设过时才是空的 (也就是画成\"(未选择 agent)\")");
        ind.setConfiguredAgent(QStringLiteral("Alpha-Beta Pruning (深度=4)"));
        check(ind.displayedAgent() == QStringLiteral("Alpha-Beta Pruning (深度=4)"),
              "setConfiguredAgent 之后显示当前选中的对战 agent");
        ind.resetToIdle();      /* 按"开局" */
        check(ind.displayedAgent() == QStringLiteral("Alpha-Beta Pruning (深度=4)"),
              "按\"开局\"之后仍然显示它 —— 报障的\"agent 未加载\"就是这条被清空过");
        ind.start(QStringLiteral("PPO+MCTS"), QStringLiteral("① 搜索 / 决策"));
        check(ind.displayedAgent() == QStringLiteral("PPO+MCTS"),
              "思考期间显示的是**正在想的那个** agent");
        ind.stop();
        check(ind.displayedAgent() == QStringLiteral("PPO+MCTS"),
              "思考结束后保留那一次的名字");
        ind.resetToIdle();
        check(ind.displayedAgent() == QStringLiteral("Alpha-Beta Pruning (深度=4)"),
              "再按一次\"开局\"又回到当前选中的 agent");
    }

    std::printf("\n==== 失败项: %d ====\n", gFailed);
    return gFailed == 0 ? 0 : 1;
}
