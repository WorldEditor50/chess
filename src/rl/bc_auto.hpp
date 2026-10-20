#ifndef RL_BC_AUTO_HPP
#define RL_BC_AUTO_HPP

/*
 * ============================================================================
 *  rl/bc_auto.hpp - 连败自动行为克隆的**纯判据** (2026-10-16 用户口径)
 * ============================================================================
 *
 * 用户口径: "在对弈训练中如果连续负5局就启用行为克隆模仿 ab agent-4 直到胜一局"。
 *
 * 为什么把它抽成一个**纯函数** (而不是留在 ChessBoard::autoBcOnGameEnd 里):
 *   那一层要 QWidget / 棋盘 / 对局循环才能跑, 于是判据本身**测不到** —— 本工程反复
 *   记过的教训是"判据不可注入 ⇒ 只能靠改源码做实验"。这里只有 int/bool 进、int/bool 出,
 *   `test_bc_auto` 就能把每一条边界 (胜/负/和 × 阈值前后 × 有无学生) 逐条钉住,
 *   而 `ChessBoard` 只负责"把结果落到成员 + 发信号"。
 *
 * 判据 (每一条都是可以被读成另一件事的地方, 所以写死在这里):
 *
 *   1. **只在训练对局计数**: 评估/只对弈的契约是"权重不变"(BC 是学习), 这一条由调用方
 *      在进入本函数**之前**判 (`MATCH_TRAIN`); 本函数不重复判, 因为它不认识"模式"。
 *   2. **胜**: 连败清零; 如果自动克隆正开着 -> `Disengage` (调用方据此还原老师深度)。
 *   3. **和棋**: 连败清零, **不改**开关。和棋不是负 ⇒ "连续负"被打断; 但已经开着的
 *      自动克隆不因为一局和棋就关掉 (那是"连续胜"才该有的语义, 用户没要求)。
 *   4. **负**: 计数 +1; 当 `计数 >= 阈值` 且**当前没开**且**有学生** -> `Engage`;
 *      没有学生时返回 `CountLoss`, 由调用方在"刚好到阈值那一局"报一次
 *      "连败 N 局但没有 BC 路径 -> 未启用" (静默的判据就是下一次的误读)。
 *   5. **阈值** 由调用方给 (`setBcAutoLossThreshold`, 默认 5); `<= 0` 在这里夹成 1,
 *      免得"阈值 0 = 每负一局就触发"这种不可读的状态被写进配置。
 *   6. **关掉只在胜局发生**: 用户口径是"直到胜一局", 所以和棋/中止都不关。
 *
 * 本函数**无副作用、无全局状态、不读时钟/随机数** ⇒ 同样输入永远同样输出。
 * ============================================================================ */

namespace RL {
namespace BcAuto {

/* 一次局面结果该做什么 */
enum class Step {
    None = 0,     /* 什么都没变 (胜但没开着 / 和棋 / 负但没到阈值) */
    CountLoss,    /* 记了一次负 (可能刚好到阈值, 但没有学生/已经开着) */
    Engage,       /* **现在**该把老师换成 AB-4 并开始克隆 */
    Disengage     /* 胜一局 -> 该还原用户选的老师深度 */
};

/* 进入本函数前的状态 */
struct State {
    int streak = 0;    /* 已经连续负了几局 (A 方 = 学习者视角) */
    bool active = false;   /* 自动克隆是否正开着 */
};

/* 出来后的状态 + 该做什么 */
struct Decision {
    Step step = Step::None;
    int streak = 0;
    bool active = false;
};

/*
 * `aWon` / `aDraw` 是**A 方 (学习者)** 的本局结果, 两者互斥 (调用方保证);
 * 两个都为 false 表示"本局是负"。
 * `hasStudent` = A 方那一支有没有 BC 路径 (`ChessBoard::bcSupported(typeA)`)。
 */
inline Decision advance(const State &in, int threshold, bool aWon, bool aDraw, bool hasStudent)
{
    Decision d;
    d.streak = in.streak;
    d.active = in.active;
    if (threshold < 1) { threshold = 1; }

    if (aWon) {
        d.streak = 0;
        if (in.active) {
            d.active = false;
            d.step = Step::Disengage;
        }
        return d;
    }

    if (aDraw) {
        /* 和棋打断连败, 但不动已开着的开关 */
        d.streak = 0;
        return d;
    }

    /* ---- 负一局 ---- */
    d.streak = in.streak + 1;
    if (!in.active && d.streak >= threshold) {
        if (hasStudent) {
            d.active = true;
            d.step = Step::Engage;
        } else {
            /* 到阈值但没学生: 调用方在 streak == threshold 时报一次原因 */
            d.step = Step::CountLoss;
        }
        return d;
    }
    d.step = Step::CountLoss;
    return d;
}

} // namespace BcAuto
} // namespace RL

#endif // RL_BC_AUTO_HPP
