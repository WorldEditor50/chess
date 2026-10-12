/*
 * test_mixed_opponents_main.cpp — 混合对手训练池的断言测试 (2026-10 第四轮)
 * ============================================================================
 *
 * 用户口径: "实现 mcts, abagent-1,2,3,4 混合对局, mcts 对局数量占 50%,
 *           其余 abagent 平分, 按照均匀分布采样选择 agent 对弈训练"。
 *
 * 为什么单独一个 ctest 目标: 这段逻辑住在 `PPOMCTSAgent::trainSelfPlay` 里, 而它是
 * **训练主路径**的一部分 —— 采样比例错一位 (比如 AB 只抽到 3 档、或 MCTS 抽成 100%)
 * 都不会报错, 只会让训练分布与用户口径不一致, 而损失曲线**完全看不出来**。
 * `test_ppomcts` 虽然覆盖这个类, 但它不在 ctest 里 (~20 分钟, TB 骨干) ——
 * 这条口径必须秒级可查, 所以单开一个用**廉价骨干**跑的套件。
 *
 * 六段 (每段都对应一个"不报错但结果不对"的可能性):
 *   [1] 池子的构成与权重: MCTS 0.5 / AB-L1..L4 各 0.125 (和为 1);
 *   [2] **采样直方图** (真正跑 N 局, 只给小模拟数): 每个对手的局数落在
 *       二项分布的合理区间内 —— 这条是"均匀分布采样"的直接证据;
 *   [3] **每局换先手**: 奇数局对手执红 (否则红黑样本量不对称);
 *   [4] **对手的手不进学习方的样本**: 用 `learnerPlyCount()` 与总手数对账;
 *   [5] **默认关**: 不打开池子时行为与纯自对弈一致 (对手局数 == 0, 且不建任何实例);
 *   [6] 参数越界时的**纠正读数** (mctsShare=2 或 abMaxDepth=0 都不能静默变成别的分布)。
 *
 * 用法: test_mixed_opponents   (无参数; 失败以非 0 退出码报告, 与其它 test_* 同口径)
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/util.hpp"

namespace {

int g_pass = 0;
int g_fail = 0;

void CHECK(bool ok, const char *what)
{
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        std::printf("  [FAIL] %s\n", what);
    }
}

/*
   廉价的训练网: **MLP 专家骨干** (E=8/top-2, 每专家 2·d·h 参数)。
   为什么不用 TB: 本测试要跑几百局 (每局只给 1~2 次模拟), 建网/前向必须是毫秒级 ——
   这条测试的口径是**采样**, 不是棋力, 所以骨干与保真度无关。
*/
const RL::PPO::Backbone kBackbone = RL::PPO::Backbone::MlpExperts;
const int kExpertHidden = 16;



/*
   跑 `games` 局, 每局只给 1 手 (`maxMoves=1` ⇒ 一局最多落一子, 采样与"对手走一手"
   两条路径都被走到, 而代价最小)。返回池子的直方图。
   `verbose=false` ⇒ 不打印逐局行。
*/
std::vector<PPOMCTSAgent::OpponentUse> runPool(PPOMCTSAgent &ag, int games, bool mixed,
                                               float mctsShare = 0.5f, int abMaxDepth = 4,
                                               int mctsSims = 8)
{
    if (mixed) {
        ag.enableMixedOpponentTraining(mctsShare, mctsSims, abMaxDepth);
    } else {
        ag.disableMixedOpponentTraining();
    }
    ag.trainSelfPlay(games, /*simulations=*/1, /*maxMoves=*/1, /*verbose=*/false,
                     /*tempRoot=*/1.0f, /*tempFinal=*/1.0f);
    return ag.trainOpponentUsage();
}

}  // namespace

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf(" test_mixed_opponents — 混合对手池 (MCTS 50%% + AB-L1..L4 各 12.5%%)\n");
    RL::Random::setSeed(20240901u);

    /* ================= [1] 池子的构成与权重 ================= */
    std::printf("\n[1] 池子构成 (权重和为 1, 名字与深度对得上)\n");
    {
        Chess env;
        env.reset();
        PPOMCTSAgent ag(env, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, false,
                        kBackbone);
        CHECK(!ag.mixedOpponentTrainingEnabled(), "默认**关** (纯自对弈 = 改动前的行为)");
        CHECK(ag.trainOpponentUsage().empty(), "默认没有对手实例");

        ag.enableMixedOpponentTraining(0.5f, 8, 4);
        const std::vector<PPOMCTSAgent::OpponentUse> pool = ag.trainOpponentUsage();
        CHECK(ag.mixedOpponentTrainingEnabled(), "打开之后池子非空");
        CHECK(pool.size() == 5, "池子 = MCTS + AB-L1..L4 (5 个对手)");
        if (pool.size() == 5) {
            std::printf("    ");
            for (std::size_t i = 0; i < pool.size(); i++) {
                std::printf("%s=%.3f  ", pool[i].name.c_str(), (double)pool[i].weight);
            }
            std::printf("\n");
            CHECK(pool[0].name.rfind("MCTS", 0) == 0, "第 0 个是 MCTS (名字带预算, 如 MCTS(800))");
            CHECK(std::fabs((double)pool[0].weight - 0.5) < 1e-6, "MCTS 权重 = 0.50 (用户口径)");
            bool abOk = true;
            for (std::size_t i = 1; i < pool.size(); i++) {
                const std::string want = std::string("AB-L") + std::to_string((int)i);
                if (pool[i].name != want) { abOk = false; }
                if (std::fabs((double)pool[i].weight - 0.125) > 1e-6) { abOk = false; }
            }
            CHECK(abOk, "AB-L1..L4 名字正确且权重各 = 0.125 (剩下的一半**平分**)");
            double sum = 0.0;
            for (std::size_t i = 0; i < pool.size(); i++) { sum += (double)pool[i].weight; }
            CHECK(std::fabs(sum - 1.0) < 1e-6, "权重之和 == 1 (采样按累积权重取, 不和 1 会偏)");
            CHECK(env.stones[0] != nullptr, "棋盘仍然可用 (对手绑定在同一个棋盘上)");
        }
    }

    /* ================= [2] 采样直方图 (均匀分布的直接证据) ================= */
    std::printf("\n[2] 采样直方图 (200 局)\n");
    {
        Chess env;
        env.reset();
        PPOMCTSAgent ag(env, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, true, kBackbone);
        const int games = 200;
        const std::vector<PPOMCTSAgent::OpponentUse> pool =
            runPool(ag, games, /*mixed=*/true);
        CHECK(ag.mixedOpponentGames() == games,
              "每一局都抽到了对手 (mixedOpponentGames == 局数)");
        int total = 0;
        for (std::size_t i = 0; i < pool.size(); i++) { total += pool[i].games; }
        CHECK(total == games, "逐对手局数之和 == 总对局数 (没有漏抽/多抽)");
        /*
           区间怎么给 (不是拍脑袋): 每个对手的局数是**二项分布** B(n, p)。
           n=200 时: MCTS 的 std = sqrt(200*0.5*0.5) ≈ 7.07, AB 各 std ≈ 4.68。
           给 ±5 sigma 的区间 —— 采样正确时几乎不可能越界, 而"抽成 100% MCTS"或
           "只抽 3 档 AB"这类错误必然越界。
        */
        bool inRange = true;
        for (std::size_t i = 0; i < pool.size(); i++) {
            const double p = (double)pool[i].weight;
            const double n = (double)games;
            const double sigma = std::sqrt(n * p * (1.0 - p));
            const double lo = n * p - 5.0 * sigma;
            const double hi = n * p + 5.0 * sigma;
            const int got = pool[i].games;
            std::printf("    %-6s 期望 %6.1f 实测 %3d  (5sigma 区间 [%.1f, %.1f]) 占比 %.3f\n",
                        pool[i].name.c_str(), n * p, got, lo, hi,
                        (double)got / n);
            if ((double)got < lo || (double)got > hi) { inRange = false; }
        }
        CHECK(inRange, "每个对手的局数都落在 5sigma 区间内 (均匀分布采样生效)");
        /* 对手真的走了棋 (不是"抽了但从没轮到它") */
        int oppPlies = 0;
        for (std::size_t i = 0; i < pool.size(); i++) { oppPlies += pool[i].plies; }
        std::printf("    对手总手数 %d, 学习方进轨迹的手数 %d\n", oppPlies,
                    ag.learnerPlyCount());
        CHECK(oppPlies > 0, "对手真的走了棋 (plies > 0)");
        CHECK(ag.learnerPlyCount() > 0, "学习方也真的走了棋并进了轨迹");
    }

    /* ================= [3] 每局换先手 ================= */
    std::printf("\n[3] 每局换先手 (奇数局对手执红)\n");
    {
        Chess env;
        env.reset();
        PPOMCTSAgent ag(env, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, true, kBackbone);
        CHECK(ag.mixedOpponentColorForEpisode(0) == Stone::COLOR_BLACK,
              "第 0 局: 对手执黑 (trainSelfPlay 的起手方是黑)");
        CHECK(ag.mixedOpponentColorForEpisode(1) == Stone::COLOR_RED, "第 1 局: 对手执红");
        CHECK(ag.mixedOpponentColorForEpisode(2) == Stone::COLOR_BLACK, "第 2 局: 又回到黑");
        CHECK(ag.mixedOpponentColorForEpisode(3) != ag.mixedOpponentColorForEpisode(2),
              "相邻两局对手的颜色必须不同 (否则红黑样本量不对称)");
    }

    /* ================= [4] 对手的手不进学习方的样本 ================= */
    std::printf("\n[4] 对手的手不进学习方的轨迹 (只推进局面)\n");
    {
        /*
           构造: 每局只给 1 手 (`maxMoves=1`), 对手执黑 (第 0 局) ⇒ 这一局的第一手就是
           对手的, 学习方**一手都没走**。于是:
             纯自对弈: learnerPly == 局数 (每局一手)
             混合对局: learnerPly == 0 (那一手是对手的, 不进轨迹)
           这条把"对手的着法被当成了学习方的动作"这个错误直接抓出来 ——
           那种错会让 PPO 学到 off-policy 的转移, 而不报任何错。
        */
        Chess envA;
        envA.reset();
        PPOMCTSAgent solo(envA, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, true, kBackbone);
        runPool(solo, /*games=*/20, /*mixed=*/false);
        const int soloPlies = solo.learnerPlyCount();
        std::printf("    纯自对弈 20 局 (每局 1 手): 学习方进轨迹 %d 手\n", soloPlies);
        CHECK(soloPlies == 20, "纯自对弈: 每局一手都进学习方的轨迹 (对照臂成立)");

        Chess envB;
        envB.reset();
        PPOMCTSAgent mixed(envB, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, true, kBackbone);
        std::vector<PPOMCTSAgent::OpponentUse> pool = runPool(mixed, /*games=*/20, true);
        int oppPlies = 0;
        for (std::size_t i = 0; i < pool.size(); i++) { oppPlies += pool[i].plies; }
        std::printf("    混合对局 20 局 (每局 1 手, 第 0 局对手执黑): 对手 %d 手 / 学习方 %d 手\n",
                    oppPlies, mixed.learnerPlyCount());
        CHECK(oppPlies == 10, "对手走的手数 == 偶数局数 (10 局里它是起手方)");
        CHECK(mixed.learnerPlyCount() == 10,
              "学习方进轨迹的手数 == 奇数局数 (对手那一半**没有**进样本)");
        CHECK(mixed.learnerPlyCount() + oppPlies == 20,
              "两者之和 == 总手数 (每一手都有人走, 且只归一方)");
    }

    /* ================= [5] 默认关 = 纯自对弈 ================= */
    std::printf("\n[5] 不打开池子时与改动前一致\n");
    {
        Chess env;
        env.reset();
        PPOMCTSAgent ag(env, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, true, kBackbone);
        runPool(ag, /*games=*/20, /*mixed=*/false);
        CHECK(!ag.mixedOpponentTrainingEnabled(), "池子仍然是关的");
        CHECK(ag.mixedOpponentGames() == 0, "没有一局被记为'对手局'");
        CHECK(ag.trainOpponentUsage().empty(), "没有对手实例 (也不会有对手的读数)");
        CHECK(ag.learnerPlyCount() == 20, "20 局 × 每局 1 手, 全部是学习方自己的");
    }

    /* ================= [6] 参数越界时的纠正读数 ================= */
    std::printf("\n[6] 参数越界不静默变成另一种分布\n");
    {
        Chess env;
        env.reset();
        PPOMCTSAgent ag(env, 64, 0.99f, 0.001f, 1.414f, kExpertHidden, 0.1f, false,
                        kBackbone);
        /* mctsShare 给 2.0 (越界): 纠正到 1.0 ⇒ 只有 MCTS, AB 一个都没有 */
        ag.enableMixedOpponentTraining(2.0f, 8, 4);
        CHECK(ag.mixedOpponentMctsShare() == 1.0f, "mctsShare 被纠正到 1.0");
        CHECK(ag.trainOpponentUsage().size() == 1, "⇒ 池子里只剩 MCTS (没有负权重的 AB)");
        /* mctsShare 给负数: 纠正到 0 ⇒ 只有 AB, 四档各 0.25 */
        ag.enableMixedOpponentTraining(-1.0f, 8, 4);
        CHECK(ag.mixedOpponentMctsShare() == 0.0f, "mctsShare 被纠正到 0.0");
        const std::vector<PPOMCTSAgent::OpponentUse> onlyAb = ag.trainOpponentUsage();
        CHECK(onlyAb.size() == 4, "⇒ 池子 = AB-L1..L4 (没有 MCTS)");
        if (onlyAb.size() == 4) {
            CHECK(std::fabs((double)onlyAb[0].weight - 0.25) < 1e-6,
                  "MCTS 占 0 时, 剩下的一半之外**全部**平分给 4 档 AB (各 0.25)");
        }
        /* abMaxDepth = 0: 只剩 MCTS */
        ag.enableMixedOpponentTraining(0.5f, 8, 0);
        CHECK(ag.trainOpponentUsage().size() == 1, "abMaxDepth=0 ⇒ 只有 MCTS (不建 AB)");
        /* 关闭 */
        ag.disableMixedOpponentTraining();
        CHECK(!ag.mixedOpponentTrainingEnabled() && ag.trainOpponentUsage().empty(),
              "disable 之后池子清空");
    }

    std::printf("\n=== test_mixed_opponents: %d 项断言, %d 失败 ===\n", g_pass + g_fail, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
