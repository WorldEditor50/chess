/*
 * repro_save_blocks_ai_main.cpp - 复现"静默保存权重"把 AI 决策/界面卡住
 * ============================================================================
 *
 * 用户报障 (2026-09): "人机对弈, 黑方赢了之后, 再次开局沙漏显示模型未加载,
 *                      红方下棋子后黑方进入无限等待"
 *
 * 待验证的机制: 权重保存 (saveCurrentAgentModel) 在 `m_agentMutex` **锁内**做整份
 * 序列化 + 写盘 (PPO+MCTS 是 266 MB actor + 264 MB critic)。而 AI 决策
 * (aiThinkRaw 的 RL 分支) 与界面上的若干只读操作都要拿同一把锁 ⇒
 * 写盘期间决策线程排队等锁, 表现就是"无限等待"。
 *
 * 这个程序把时间量出来:
 *   1. 主线程调 humanTurnAiMoveForTest(黑) —— 一次 AI 决策要多久 (基线);
 *   2. 起一个线程调 saveCurrentAgentModel —— 保存要多久;
 *   3. 保存进行中再调一次 humanTurnAiMoveForTest —— 它被挡了多久。
 *
 * 判据: 若第 3 步的等待时间 ≈ 第 2 步的保存时间, 则"保存持锁"就是卡顿的来源。
 *
 * 用法: repro_save_blocks_ai [--sims=N]
 * 不进 ctest (它是定位工具)。
 */
#include <QApplication>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "chessboard.h"

namespace {
double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}
}  // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    int sims = 16;
    for (int i = 1; i < argc; i++) {
        const char *eq = std::strchr(argv[i], '=');
        if (eq == nullptr) { continue; }
        const std::string k = std::string(argv[i]).substr(0, (std::size_t)(eq - argv[i]));
        if (k == "--sims") { sims = std::atoi(eq + 1); }
    }

    ChessBoard board;
    board.setAgentType(ChessBoard::AGENT_PPOMCTS);
    board.setPreTrainEnabled(true);
    board.setPreTrainSteps(sims);

    std::printf("=== 复现: 保存权重 vs AI 决策 (同一把 m_agentMutex) ===\n");

    /* ---- 1. 先建出常驻 agent 并量一次决策基线 ---- */
    board.reset();
    const double t0 = nowMs();
    const Step s1 = board.humanTurnAiMoveForTest(Stone::COLOR_BLACK);
    const double msFirst = nowMs() - t0;
    std::printf("  1) 首次 AI 决策 (含建网+载权重): %.0f ms, 走法合法=%d\n",
                msFirst, (int)s1.valid);

    board.reset();
    const double t1 = nowMs();
    const Step s2 = board.humanTurnAiMoveForTest(Stone::COLOR_BLACK);
    const double msWarm = nowMs() - t1;
    std::printf("  2) 第二次 AI 决策 (热): %.0f ms, 走法合法=%d\n",
                msWarm, (int)s2.valid);

    /* ---- 2. 后台线程保存权重, 量它多久 ---- */
    const std::string prefix = "weights/_repro_save_blocks";
    std::atomic<double> saveMs{-1.0};
    std::atomic<bool> saveDone{false};
    std::thread saver([&] {
        const double ts = nowMs();
        board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS, prefix);
        saveMs = nowMs() - ts;
        saveDone = true;
    });

    /* 等保存真的开始 (给它 300 ms 进入临界区) */
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    /* ---- 3. 保存进行中再决策一次, 量被挡了多久 ---- */
    board.reset();
    const double t2 = nowMs();
    const Step s3 = board.humanTurnAiMoveForTest(Stone::COLOR_BLACK);
    const double msBlocked = nowMs() - t2;
    std::printf("  3) 保存进行中的一次 AI 决策: %.0f ms (走法合法=%d)\n",
                msBlocked, (int)s3.valid);

    saver.join();
    std::printf("  4) 保存本身耗时: %.0f ms\n", saveMs.load());
    std::printf("\n判读: 若第 3 步 ≈ 第 4 步, 则'保存持锁'就是用户看到的"
                "'黑方无限等待' —— 因为 AI 决策要等整份权重写完。\n");
    std::printf("      基线(第 2 步) %.0f ms vs 被挡(第 3 步) %.0f ms, 倍数 %.1fx\n",
                msWarm, msBlocked, (msWarm > 1.0) ? msBlocked / msWarm : 0.0);

    /* 清理临时权重 */
    std::remove((prefix + "_actor").c_str());
    std::remove((prefix + "_critic").c_str());
    return 0;
}
