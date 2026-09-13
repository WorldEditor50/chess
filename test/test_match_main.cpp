/*
 * test_match_main.cpp - Agent 对 Agent 对弈 (arena) 的验证
 *
 * 这个功能是"让两个 agent 打若干局, 用来比较强弱", 所以真正需要盯住的不是
 * "能不能下完", 而是几件统计上的事:
 *
 *   1. 每局**交换先后手**。中国象棋先手(红)优势很大, 固定谁执红的话, 结果
 *      只是在测"谁执红"而不是"谁更强"。日志里必须两局的执红方是相反的。
 *   2. 胜负按**参赛者 A/B** 归属, 不是按红黑。归错一边的话整个比分就没有意义。
 *   3. 达到手数上限判**和棋** (老代码在这里按静态评估判"红胜", 于是平局分支
 *      永远不可达)。
 *   4. 中止要在若干手之内生效, 且不能把半局算成完整一局。
 *
 * 做法上不去跑几百手的真对局: 把 setMaxPliesPerGame 调小 (这里是 4), 一局
 * 4 手必然到上限判和 —— 于是每局结果可预测, 断言就能做得很硬。
 */
#include <QApplication>
#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#include "chessboard.h"

static int g_checks = 0;
static int g_failed = 0;
#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

static bool contains(const QString &hay, const QString &needle)
{
    return hay.contains(needle);
}

int main(int argc, char *argv[])
{
    /* 无缓冲: 崩溃时也能看到走到了哪一步 (printf 默认是行/块缓冲, 段错误会把它丢掉) */
    setvbuf(stdout, nullptr, _IONBF, 0);

    /* 无显示器也能跑: 强制用 offscreen 平台插件 */
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    std::printf("=== Agent 对 Agent 对弈 (arena) ===\n");
    std::printf("[build] %s %s\n", __DATE__, __TIME__);

    ChessBoard board;
    std::printf("[ok] ChessBoard 构造完成\n");
    /* 不探索: 这个测试关心的是对弈统计, 不是探索 (那是 test_pretrain 的事) */
    board.setPreTrainEnabled(false);
    board.setPreTrainSteps(0);

    /* ---------------------------------------------------------------- 1. 交换先后手 + 比分归属 */
    std::printf("\n[1] 2 局, 每局限 4 手 (必然判和), 检查先后手交换与统计\n");
    board.setMaxPliesPerGame(4);

    std::printf("[..] 调用 matchAgents(AlphaBeta, EVAB, 2)\n");
    ChessBoard::MatchStats st =
        board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_EVAB, 2);
    std::printf("[ok] matchAgents 返回\n");

    std::printf("    %s\n", st.summary().toUtf8().constData());
    std::printf("%s", st.detail().toUtf8().constData());
    std::printf("\n");

    CHECK(st.games == 2, "打完 2 局");
    CHECK(st.plies == 2 * 4, "总手数 = 局数 x 每局上限");
    CHECK(st.winA == 0 && st.winB == 0, "4 手之内不可能将杀, 不应有胜局");
    CHECK(st.draws == 2, "到上限判和 (不是判红胜)");
    CHECK(st.winA + st.winB + st.draws == st.games, "比分之和等于局数");
    CHECK(!st.aborted, "正常跑完不算被中止");
    CHECK(!board.isMatchRunning(), "跑完之后不再处于对弈中");

    /* 第 1 局 A 执红, 第 2 局 B 执红 —— 这一条是方法学的核心 */
    CHECK(contains(st.log, QStringLiteral("第 1 局: 红=Alpha-Beta 黑=EVAB")),
          "第 1 局 A(Alpha-Beta) 执红");
    CHECK(contains(st.log, QStringLiteral("第 2 局: 红=EVAB 黑=Alpha-Beta")),
          "第 2 局 B(EVAB) 执红 (先后手已交换)");

    /* ---------------------------------------------------------------- 2. 同一 agent 打两边 */
    std::printf("\n[2] 同一个 agent 打两边 (= 原来的 self-play)\n");
    ChessBoard::MatchStats self =
        board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_ALPHABETA, 1);
    std::printf("    %s\n", self.summary().toUtf8().constData());
    CHECK(self.games == 1, "自对弈也能作为对弈的一个特例跑完");
    CHECK(self.agentA == self.agentB, "两边是同一个 agent");

    /* ---------------------------------------------------------------- 3. 中止 */
    std::printf("\n[3] 中止: 请求 50 局, 跑一会儿后叫停\n");
    board.setMaxPliesPerGame(300);
    ChessBoard::MatchStats aborted;
    std::thread worker([&board, &aborted]() {
        aborted = board.matchAgents(ChessBoard::AGENT_ALPHABETA,
                                    ChessBoard::AGENT_EVAB, 50);
    });
    /* 等它真的开始动手, 再叫停 */
    for (int i = 0; i < 200 && !board.isMatchRunning(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(board.isMatchRunning(), "对弈已经在跑");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    board.abortMatch();
    worker.join();

    std::printf("    中止后: %s\n", aborted.summary().toUtf8().constData());
    CHECK(aborted.aborted, "标记为已中止");
    CHECK(aborted.games < 50, "没有打满请求的局数");
    CHECK(!board.isMatchRunning(), "中止后不再处于对弈中");
    /* 中止时那一局没打完, 不应被算进比分 */
    CHECK(aborted.winA + aborted.winB + aborted.draws == aborted.games,
          "半局没有被算成完整一局");

    /* ---------------------------------------------------------------- 4. 手数上限恢复 */
    std::printf("\n[4] 手数上限可恢复 (界面上默认 300)\n");
    board.setMaxPliesPerGame(300);
    CHECK(board.getMaxPliesPerGame() == 300, "上限设回 300");
    board.setMaxPliesPerGame(0);
    CHECK(board.getMaxPliesPerGame() == 1, "非法值 (0) 被夹到 1, 不会死循环");
    board.setMaxPliesPerGame(300);

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
