/*
 * bench_match_modes_main.cpp - 对弈模式 (P0-a) 的成本与语义读数
 * ============================================================================
 *
 * 为什么单独一个探针: "评估模式到底快多少"决定了它能不能当日常工具用 ——
 * 而模式同时关掉两条学习路径 (rollout + SAC 的 learnFromSearch), 省下的时间不是
 * 一个常数比例, 而是与 agent 组合有关 (SAC+AZ 每手要跑一次 learnBatch 约 2.3 s)。
 * 这里把三种模式在**同一对 agent、同一局数**下的墙钟时间与更新次数并排打出来。
 *
 * 用法:
 *   bench_match_modes [--games=N] [--plies=N] [--sims=N] [--a=ppo|sac|pg] [--b=...]
 * 不进 ctest (它是读数工具, 不是回归断言 —— 回归在 test_match [2.7d])。
 */
#include <QApplication>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <QMetaObject>

#include "chessboard.h"

namespace {

double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}

ChessBoard::AgentType parseAgent(const std::string &s)
{
    if (s == "ppo")    { return ChessBoard::AGENT_PPOMCTS; }
    if (s == "ppo-mlp"){ return ChessBoard::AGENT_PPOMCTS_MLP; }
    if (s == "sac")    { return ChessBoard::AGENT_SACAZ; }
    if (s == "pg")     { return ChessBoard::AGENT_PG; }
    if (s == "dqn")    { return ChessBoard::AGENT_DQN; }
    if (s == "ab")     { return ChessBoard::AGENT_ALPHABETA; }
    if (s == "ab2")    { return ChessBoard::AGENT_AB_L2; }
    if (s == "d qnab" || s == "dqnab") { return ChessBoard::AGENT_DQNAB; }
    return ChessBoard::AGENT_PPOMCTS;
}

const char *modeTag(ChessBoard::MatchMode m)
{
    switch (m) {
    case ChessBoard::MATCH_TRAIN:    return "TRAIN   ";
    case ChessBoard::MATCH_EVAL:     return "EVAL    ";
    case ChessBoard::MATCH_NO_LEARN: return "NO_LEARN";
    }
    return "?";
}

}  // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    int games = 1;
    int plies = 40;
    int sims = 32;
    std::string aName = "ppo";
    std::string bName = "sac";
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        if (eq == nullptr) { continue; }
        const std::string k = arg.substr(0, (std::size_t)(eq - argv[i]));
        const std::string v = arg.substr((std::size_t)(eq - argv[i]) + 1);
        if (k == "--games") { games = std::atoi(v.c_str()); }
        else if (k == "--plies") { plies = std::atoi(v.c_str()); }
        else if (k == "--sims") { sims = std::atoi(v.c_str()); }
        else if (k == "--a") { aName = v; }
        else if (k == "--b") { bName = v; }
    }

    ChessBoard board;
    board.setMaxPliesPerGame(plies);
    board.setPreTrainEnabled(true);
    board.setPreTrainSteps(sims);
    board.setAgentType(ChessBoard::AGENT_PPOMCTS);   /* 后台训练的目标 (本轮不启动它) */

    const ChessBoard::AgentType A = parseAgent(aName);
    const ChessBoard::AgentType B = parseAgent(bName);

    std::printf("=== 对弈模式成本/语义读数 (P0-a) ===\n");
    std::printf("  A=%s  B=%s | %d 局 x %d 手上限 | 探索步数 %d\n\n",
                aName.c_str(), bName.c_str(), games, plies, sims);
    std::printf("  %-9s %10s %8s %8s %10s\n",
                "模式", "墙钟(s)", "A更新", "B更新", "闸门次数");

    const ChessBoard::MatchMode modes[] = {
        ChessBoard::MATCH_TRAIN, ChessBoard::MATCH_EVAL, ChessBoard::MATCH_NO_LEARN
    };
    for (ChessBoard::MatchMode m : modes) {
        int cntA = 0, cntB = 0;
        QMetaObject::Connection conn = QObject::connect(
            &board, &ChessBoard::trainLossSample,
            [&cntA, &cntB](double loss, const QString &agent, int step) {
                (void)loss; (void)step;
                if (agent.contains(QStringLiteral("PPO"))) { cntA++; } else { cntB++; }
            });
        const int blockedBefore = board.matchLearningBlockedCount();
        board.setMatchMode(m);
        const double t0 = nowMs();
        const ChessBoard::MatchStats st = board.matchAgents(A, B, games);
        const double dt = (nowMs() - t0) / 1000.0;
        QObject::disconnect(conn);
        std::printf("  %-9s %10.1f %8d %8d %10d   (%d 局 / %d 手)\n",
                    modeTag(m), dt, cntA, cntB,
                    board.matchLearningBlockedCount() - blockedBefore,
                    st.games, st.plies);
    }
    std::printf("\n读法: NO_LEARN 的墙钟 = 纯搜索成本; TRAIN 与它的差 = 每手在线学习的成本。\n");
    std::printf("      A/B 更新次数为 0 表示该侧确实没学 (不是'损失恰好没上报')。\n");
    return 0;
}
