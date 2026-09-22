/*
 * repro_concurrency_main.cpp - "对弈 x 后台训练" 并发崩溃的最小复现 (调试用)
 *
 * 2026-09 用户报障: "对弈时自动保存权重的时候导致程序崩溃了"。
 * Windows 事件日志里的异常码是 **0xC0000374 (STATUS_HEAP_CORRUPTION, ntdll)** ——
 * 堆损坏, 也就是"有人写越界 / 用了已释放的内存", 而不是某个函数返回了错误。
 *
 * 缩小范围的过程 (见 docs/issues_review.md 零之二点二十五):
 *   1. 真界面里逐项排除: "对战AI(后台训练的目标) == 对弈的某一方" **必须成立**才会崩;
 *      训练目标与参赛者不同、或根本不训练时都不崩。**用 EVAB 也一样崩** ——
 *      所以这跟新加的 PPO+MCTS-MLP 无关, 是一个更早就存在的问题。
 *   2. 把同样的形状搬到无界面的测试里 (本文件): 对弈跑在**另一个线程**,
 *      主线程同时按"每手一次"的频率调 getAgentSelfCheck() (界面那个自检 worker 干的事),
 *      后台训练开着 —— 于是不需要界面就能稳定复现 (test_match 里那些"对弈在主线
 *      程上跑"的用例复现不出来, 这一点本身就是线索)。
 *
 * 这个文件**故意不是一个 ctest**: 它复现的是一类时序相关的堆损坏, 跑得快不快、
 * 崩不崩取决于调度; 它的用途是"改完并发相关的代码之后, 在这里压一下"。
 *
 * 用法:
 *   repro_concurrency.exe [轮数] [每局手数] [agent 序号]
 *   :: ASan 版本能得到准确的越界/释放栈 (见 docs/issues_review.md 的复现命令)
 *
 * agent 序号 = ChessBoard::AgentType 的值 (0 Alpha-Beta, 1 MCTS, 2 PG, 3 DQN,
 * 4 PPO+MCTS, 5 PPO+MCTS-MLP, 6 DQN+MCTS, 7 EVAB, 8 SAC+AZ, 9 SAC+AZ-MoE, 10 DQN+AB)。
 */
#include <QApplication>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include "chessboard.h"

int main(int argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    const int rounds = (argc > 1) ? std::atoi(argv[1]) : 3;
    const int plies = (argc > 2) ? std::atoi(argv[2]) : 12;
    const int agentIdx = (argc > 3) ? std::atoi(argv[3]) : (int)ChessBoard::AGENT_PPOMCTS_MLP;
    const ChessBoard::AgentType type = (ChessBoard::AgentType)agentIdx;

    std::printf("=== 对弈 x 后台训练 并发复现 ===\n");
    std::printf("[build] %s %s | agent=%d 轮数=%d 每局手数=%d\n",
                __DATE__, __TIME__, agentIdx, rounds, plies);

    ChessBoard board;
    /* 界面里的设置: 走子前探索 + 预训练 (默认 64 步) 也照旧打开 */
    std::printf("[ok] ChessBoard 构造完成\n");

    /* ---- 关键: 后台训练的目标 == 对弈的 A 方 (同一个实例) ---- */
    board.setAgentType(type);
    board.setBackgroundTrainRound(1, 40);
    board.startBackgroundTraining();
    board.setMaxPliesPerGame(plies);
    std::printf("[ok] 后台训练已启动 (目标 = agent %d), 对弈每局 %d 手\n", agentIdx, plies);

    for (int g = 0; g < rounds; g++) {
        /* ---- 对弈跑在独立线程 (MainWindow 用的是 m_selfPlayThread) ---- */
        std::atomic<bool> done{false};
        std::thread matchThread([&]() {
            board.matchAgents(type, ChessBoard::AGENT_ALPHABETA, 1);
            done = true;
        });

        /* ---- 主线程按"每手一次"的频率刷自检 (GUI 的自检 worker) ---- */
        long long checks = 0;
        while (!done.load()) {
            board.getAgentSelfCheck(type);
            checks++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        matchThread.join();

        /* ---- 对弈结束后立刻保存权重 (GUI 的"静默保存") ---- */
        const bool ok = board.saveCurrentAgentModel(type,
                                                    ChessBoard::defaultWeightPath(type));
        std::printf("  第 %d 轮: 完成 (并发自检 %lld 次, 保存=%d)\n", g + 1, checks, (int)ok);
    }

    board.stopBackgroundTraining();
    std::printf("=== 全部跑完, 没有崩溃 ===\n");
    return 0;
}
