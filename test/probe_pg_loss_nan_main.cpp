/*
 * probe_pg_loss_nan_main.cpp - 定位"PG 的 lastLoss 变成非有限值且不复位"
 * ============================================================================
 *
 * 现象 (test_match [2.7b] 时真时假的原因):
 *   同一份代码、同样的 32 手对局, PG 有时上报 16~20 次损失, 有时**一次都不上报**。
 *   曲线的数据源是 preTrainThenDecide -> getLastTrainLoss() (PG 转发 dpg.lastLoss),
 *   而曲线控件对非有限值是直接丢弃的 ⇒ "0 次"意味着 dpg.lastLoss 已经成了 NaN/Inf。
 *
 * 这个探针把"哪一步开始坏"直接打出来: 每手决策前调一次 exploreAndTrain(rollout),
 * 然后读 dpg.lastLoss 并检查有限性; 一旦变坏就打印上下文 (手数 / 是否终局 / 轨迹长度)。
 *
 * 用法: probe_pg_loss_nan [--games=N] [--plies=N] [--steps=N]
 * 不进 ctest (它是定位工具)。
 */
#include <QApplication>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "chess.h"
#include "pgagent.h"

namespace {
bool finite(double v) { return std::isfinite(v); }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    int games = 40;
    int plies = 32;
    int steps = 32;
    for (int i = 1; i < argc; i++) {
        const char *eq = std::strchr(argv[i], '=');
        if (eq == nullptr) { continue; }
        const std::string k = std::string(argv[i]).substr(0, (std::size_t)(eq - argv[i]));
        const int v = std::atoi(eq + 1);
        if (k == "--games") { games = v; }
        else if (k == "--plies") { plies = v; }
        else if (k == "--steps") { steps = v; }
    }

    Chess chess;
    chess.reset();
    PGEagent pg(chess, 64, 0.9f, 0.01f, 1.0f);

    std::printf("=== PG lastLoss 有限性探针 ===\n");
    std::printf("  %d 局 x %d 手 x 每手 rollout %d 步\n\n", games, plies, steps);

    int bad = 0;
    int total = 0;
    for (int g = 0; g < games; g++) {
        chess.reset();
        int firstBadPly = -1;
        double lastGood = 0.0, firstBadVal = 0.0;
        for (int ply = 0; ply < plies; ply++) {
            if (chess.getResult(chess.sideToMove) != Chess::RESULT_ONGOING) {
                break;
            }
            const bool trained = pg.exploreAndTrain(chess.sideToMove, steps);
            const double loss = (double)pg.getLastTrainLoss();
            total++;
            if (trained && !finite(loss)) {
                if (firstBadPly < 0) {
                    firstBadPly = ply;
                    firstBadVal = loss;
                    std::printf("\n    (首次坏值: 局 %d 第 %d 手, loss=%.6g)\n",
                                g, ply, firstBadVal);
                }
            } else if (trained) {
                lastGood = loss;
            }
            /* 走一手, 让这一局继续 (用 agent 自己的决策, 与对弈路径一致) */
            const Step s = pg.selectMove(chess.sideToMove, false);
            if (!s.valid) {
                break;
            }
            double dummy = 0.0;
            chess.moveForward(&s, dummy);
        }
        const bool thisBad = (firstBadPly >= 0);
        if (thisBad) { bad++; }
        std::printf("  局 %2d: %s", g, thisBad ? "**非有限**" : "正常");
        if (thisBad) {
            std::printf("  首个坏值出现在第 %d 手 (值 %.6g); 此前最后一个好值 %.9g",
                        firstBadPly, firstBadVal, lastGood);
        } else {
            std::printf("  最后 loss = %.9g", lastGood);
        }
        std::printf("\n");
    }
    std::printf("\n汇总: %d/%d 手产生了可读损失; **%d/%d 局出现过非有限 lastLoss**\n",
                total - bad, total, bad, games);
    std::printf("判读: 只要 bad > 0, 就是'lastLoss 变 NaN 且不复位'复现了 ——\n");
    std::printf("      此时损失曲线会静默停止上报 (曲线控件丢弃非有限值)。\n");
    return 0;
}
