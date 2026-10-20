/*
 * test/test_bc_auto_main.cpp - 连败自动克隆的**纯判据** (2026-10-16 用户口径)
 * ============================================================================
 *
 * 用户口径: "在对弈训练中如果连续负5局就启用行为克隆模仿 ab agent-4 直到胜一局"。
 *
 * 为什么这个测试存在 (它盯的是一次**真实的功能短缺**):
 *   判据原本写在 `ChessBoard::autoBcOnGameEnd` 里 —— 要起一场对局才跑得到, 于是
 *   "阈值前不触发 / 和棋怎么算 / 胜一局怎么还原 / 没有学生怎么办 / 能不能再次触发"
 *   这些边界在测试里**一条都不存在**。而本工程反复记过的教训是: 判据不可注入 ⇒
 *   只能靠改源码做实验。所以判据被抽成 `src/rl/bc_auto.hpp` 的纯函数, 这里逐条钉住它。
 *
 * 口径 (与 bc_auto.hpp 的注释同一份, 不许在这里另写一套):
 *   * 负 +1、**胜清零**、**和棋也清零**(和棋不是负, 连败打断);
 *   * 到阈值且**没开着**且**有学生** -> Engage;
 *   * 到阈值但**没学生** -> CountLoss (调用方在"刚好到阈值"那局报一次原因);
 *   * 胜一局且正开着 -> Disengage (调用方据此还原老师深度);
 *   * 和棋**不关**已经开着的开关 (用户口径是"直到胜一局");
 *   * 阈值 <= 0 夹成 1 (免得"阈值 0 = 每负一局就触发"这种不可读状态);
 *   * 关掉之后还能**再次**触发 (同一场里连败两次是可能的)。
 *
 * 它**不测棋力、不测 BC 本身** —— 只测"什么时候该开、什么时候该关"。
 */
#include <cstdio>

#include "rl/bc_auto.hpp"

namespace {

int g_total = 0;
int g_fail = 0;

void check(bool ok, const char *what)
{
    g_total++;
    if (ok) {
        std::printf("[ok]   %s\n", what);
    } else {
        g_fail++;
        std::printf("[FAIL] %s\n", what);
    }
}

using RL::BcAuto::Decision;
using RL::BcAuto::State;
using RL::BcAuto::Step;

Decision step(const State &in, int threshold, bool won, bool draw, bool student)
{
    return RL::BcAuto::advance(in, threshold, won, draw, student);
}

/* 一局"负" */
Decision lose(const State &in, int threshold = 5, bool student = true)
{
    return step(in, threshold, /*won=*/false, /*draw=*/false, student);
}

} // namespace

int main()
{
    std::printf("=== test_bc_auto: 连败自动克隆的判据 (阈值 / 和棋 / 学生 / 还原) ===\n\n");

    /* ========================================================================
     *  [1] 阈值前: 一局都不该触发 (用户口径是"连续负 5 局")
     * ======================================================================== */
    std::printf("--- [1] 阈值之前 (1..4 局) 一律不触发 ---\n");
    {
        State s;
        bool engagedTooEarly = false;
        for (int i = 1; i <= 4; i++) {
            const Decision d = lose(s);
            if (d.step == Step::Engage || d.active) { engagedTooEarly = true; }
            if (d.streak != i) { engagedTooEarly = true; }
            if (d.step != Step::CountLoss) { engagedTooEarly = true; }
            s.streak = d.streak;
            s.active = d.active;
        }
        check(!engagedTooEarly, "连负 1..4 局: step 恒为 CountLoss, 从不 Engage, streak 逐局递增");
        check(s.streak == 4 && !s.active, "连负 4 局后 streak == 4 且开关仍关着");
    }

    /* ========================================================================
     *  [2] 第 5 局: 正好触发 (边界 = 阈值, 不是阈值+1)
     * ======================================================================== */
    std::printf("\n--- [2] 第 5 局负 -> Engage ---\n");
    {
        State s;
        s.streak = 4;
        const Decision d = lose(s);
        check(d.step == Step::Engage, "streak 4 -> 再负一局 (第 5 局) 就 Engage");
        check(d.streak == 5 && d.active,
              "Engage 时 streak == 5 且 active == true (调用方据此把老师换成 AB-4)");
    }
    {
        /* 阈值可注入: 3 -> 第 3 局触发 (证明判据没有被写死成 5) */
        State s;
        s.streak = 2;
        const Decision d = lose(s, /*threshold=*/3);
        check(d.step == Step::Engage, "阈值 = 3 时第 3 局负就触发 (阈值是参数, 不是写死的 5)");
    }
    {
        /* 阈值 <= 0 夹成 1: 第一局负就触发 */
        State s;
        const Decision d = lose(s, /*threshold=*/0);
        check(d.step == Step::Engage && d.streak == 1, "阈值 <= 0 夹成 1 (第一局负就触发)");
    }

    /* ========================================================================
     *  [3] 和棋: 打断连败, 但**不**关掉已经开着的开关
     * ======================================================================== */
    std::printf("\n--- [3] 和棋: 清零连败, 不动开关 ---\n");
    {
        State s;
        s.streak = 4;
        const Decision d = step(s, 5, false, /*draw=*/true, true);
        check(d.streak == 0 && !d.active && d.step == Step::None,
              "和棋把 streak 清零, 不 Engage (和棋不是负)");
    }
    {
        State s;
        s.streak = 7;
        s.active = true;      /* 自动克隆开着, 来了一局和棋 */
        const Decision d = step(s, 5, false, /*draw=*/true, true);
        check(d.streak == 0 && d.active && d.step == Step::None,
              "和棋**不关**已经开着的自动克隆 (用户口径: 直到胜一局)");
    }

    /* ========================================================================
     *  [4] 胜一局: 关掉并清零; 没开着时只是清零
     * ======================================================================== */
    std::printf("\n--- [4] 胜一局 -> Disengage / 或只是清零 ---\n");
    {
        State s;
        s.streak = 9;
        s.active = true;
        const Decision d = step(s, 5, /*won=*/true, false, true);
        check(d.step == Step::Disengage && !d.active && d.streak == 0,
              "开着时胜一局 -> Disengage (调用方还原用户选的老师深度)");
    }
    {
        State s;
        s.streak = 4;
        const Decision d = step(s, 5, /*won=*/true, false, true);
        check(d.step == Step::None && !d.active && d.streak == 0,
              "没开着时胜一局 -> 只是清零 (不该发'结束'通知)");
    }

    /* ========================================================================
     *  [5] 没有学生: 到阈值也不触发 (但计数继续)
     * ======================================================================== */
    std::printf("\n--- [5] 没有 BC 路径的学生: 到阈值也不 Engage ---\n");
    {
        State s;
        s.streak = 4;
        const Decision d = lose(s, 5, /*student=*/false);
        check(d.step == Step::CountLoss && !d.active && d.streak == 5,
              "到阈值但没有学生 -> CountLoss (不 Engage), 计数照样到 5");
        const Decision d2 = lose({d.streak, d.active}, 5, /*student=*/false);
        check(d2.step == Step::CountLoss && d2.streak == 6,
              "继续负: 计数继续涨, 仍不 Engage (不会'假装'触发)");
    }

    /* ========================================================================
     *  [6] 已经开着时继续负: 不重复 Engage (报告只该有一次"触发")
     * ======================================================================== */
    std::printf("\n--- [6] 已经开着: 继续负不重复触发 ---\n");
    {
        State s;
        s.streak = 9;
        s.active = true;
        const Decision d = lose(s);
        check(d.step == Step::CountLoss && d.active && d.streak == 10,
              "开着时再负: CountLoss (不重复 Engage), 开关保持");
    }

    /* ========================================================================
     *  [7] 整条轨迹: 连负 5 -> 触发; 和 -> 不关; 负 -> 仍开着; 胜 -> 关; 再连负 5 -> 再触发
     * ======================================================================== */
    std::printf("\n--- [7] 一整场的轨迹 (可重复触发) ---\n");
    {
        State s;
        int engages = 0;
        int disengages = 0;
        auto run = [&](bool won, bool draw) {
            const Decision d = step(s, 5, won, draw, true);
            if (d.step == Step::Engage) { engages++; }
            if (d.step == Step::Disengage) { disengages++; }
            s.streak = d.streak;
            s.active = d.active;
            return d.step;
        };
        for (int i = 0; i < 5; i++) { run(false, false); }          /* 连负 5 -> 触发 */
        const bool engagedAt5 = (engages == 1 && s.active && s.streak == 5);
        run(false, true);                                           /* 和: 不关 */
        const bool stillOnAfterDraw = s.active && s.streak == 0;
        run(false, false);                                          /* 负: 仍开着 */
        const bool stillOnAfterLoss = s.active;
        run(true, false);                                           /* 胜: 关 */
        const bool offAfterWin = !s.active && disengages == 1;
        for (int i = 0; i < 4; i++) { run(false, false); }          /* 再连负 4: 还不该开 */
        const bool notYetSecond = !s.active;
        run(false, false);                                          /* 第 5 局: 再触发 */
        const bool engagedAgain = s.active && engages == 2;

        check(engagedAt5, "连负 5 局 -> 第 5 局触发 (engages == 1)");
        check(stillOnAfterDraw, "和棋不关 (开关仍开, streak 清零)");
        check(stillOnAfterLoss, "和棋之后又负一局: 开关仍开 (不重复触发)");
        check(offAfterWin, "胜一局 -> 关闭 (disengages == 1)");
        check(notYetSecond, "关闭后再连负 4 局: 还不该开");
        check(engagedAgain, "再连负到第 5 局 -> **再次**触发 (engages == 2)");
    }

    /* ========================================================================
     *  [8] 纯函数: 同输入同输出, 且不修改输入
     * ======================================================================== */
    std::printf("\n--- [8] 纯函数契约 ---\n");
    {
        State s;
        s.streak = 3;
        s.active = false;
        const State before = s;
        const Decision a = step(s, 5, false, false, true);
        const Decision b = step(s, 5, false, false, true);
        check(a.streak == b.streak && a.active == b.active && a.step == b.step,
              "同一输入两次调用 -> 逐位相同 (无时钟/随机数依赖)");
        check(s.streak == before.streak && s.active == before.active,
              "不修改入参 (State 是 const 引用)");
    }

    std::printf("\n=== 断言 %d 条, 失败 %d 条 ===\n", g_total, g_fail);
    if (g_fail != 0) {
        std::printf("**判据错了**: 检查 src/rl/bc_auto.hpp 的 advance() ——\n"
                    "界面那一支的口径是\"连续负 5 局触发, 胜一局结束\" (docs/behavior_cloning_2026_10.md §11)。\n");
    }
    return g_fail == 0 ? 0 : 1;
}
