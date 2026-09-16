#ifndef CHESSSTATE_H
#define CHESSSTATE_H

/*
 * ============================================================================
 *  ChessState — "完备 Markov 状态"的公共实现（所有 RL agent 共用）
 * ============================================================================
 *
 * 为什么要有这个文件（本工程三次踩坑的总结，见 docs/agent_dqnab_design.md §3 与 §11）：
 *
 *  1. **裸棋盘 + 轮到谁不是 Markov 状态**。三次重复判和、60 回合无吃子判和、长将/循环
 *     全都依赖历史，而它们决定终局与回报。只喂棋子平面时，「同一局面的第 2 次出现」
 *     与「第 3 次出现（立刻判和）」会编码成**同一个向量** —— 那 V(s) 就不是 s 的函数，
 *     Bellman 备份 `V(s) = max_a[r − γV(s')]` 的前提（P(s'|s,a) 只依赖 s）直接失效。
 *     补法：把规则上下文做成常数平面（等价于 concat 标量，只是形状与棋子平面对齐）。
 *
 *  2. **规范视角（"己方永远在 x 大的那一侧"）是"一个网络服务红黑双方"的唯一依据**，
 *     也是 negamax 的 `val = −child` 自洽的前提。散落在各 agent 里的
 *     `(color == RED) ? x*9+y : (9-x)*9+y` 写法必须**只有一份** —— 镜像写错是最难查的
 *     静默缺陷（红黑变成两个不同的函数，训练照样跑）。
 *
 *  3. **动作索引的双射**：`fromCell*90 + toCell`（8100）。相比 `(id*37+x*13+y*7)%128`
 *     那种哈希（平均 22 个走法挤一槽），它没有碰撞，于是"两个不同走法共用一个 Q 列"
 *     这类静默缺陷不存在。
 *
 * 各 agent 的用法：先选自己的平面布局（`PLANES` / `STATE_DIM`），编码时调用
 * `encodePiecePlanes()` 写 14 个棋子平面，再按需要调用 `fillPlane()` 写规则/阶段平面；
 * 规则上下文的数值一律来自本文件，**不要各写一份**。
 *
 * ⚠ 状态维度变化会让旧权重文件失效。权重文件格式的第二版已经在 Net::load 里加了
 * **参数量守卫**（层类型相同但维度不同 -> 明确拒绝，而不是静默换成文件里的形状），
 * 所以这一步是安全的、可见的，不会悄悄把网络读成别的形状。
 */

#include <cmath>
#include <cstddef>

#include "chess.h"
#include "stone.h"

namespace ChessState {

/* ---- 平面布局 ---- */
constexpr int CELLS        = 90;    /* 10 行 x 9 列 */
constexpr int PIECE_PLANES = 14;    /* 7 类棋子 x {己方, 对方}, 下标 = type*2 + (是己方?0:1) */

/*
   规则上下文 / 局面阶段的**平面序号**（相对某个基址）。它们的值都是**当前状态**的函数 ——
   不是"环境悄悄记着"的东西，这一点是它们能修复 Markov 性的全部原因。
*/
constexpr int CTX_MATERIAL = 0;     /* 双方非将子力 / 满子: 局面阶段（动态子力价值的条件化） */
constexpr int CTX_TEMPO    = 1;     /* 总手数 / REWARD_MAX_PLIES: 局面阶段之二 */
constexpr int CTX_HALFMOVE = 2;     /* halfMoveClock / 120: 60 回合判和的风险 */
constexpr int CTX_REPEAT   = 3;     /* min(重复次数,3)/3: >=3 次判和 */
constexpr int CTX_CHECK    = 4;     /* 走子方是否被将军: 将军链 / 长将上下文的入口 */
constexpr int CTX_COUNT    = 5;

/* ---- 格子与动作索引 ---- */

/* 左右镜像（黑方取景用）: x -> 9-x。写在这一个地方，别在各 agent 里重写。 */
inline int mirrorCell(int x, int y) { return (9 - x) * 9 + y; }

/*
   规范格: 轮到谁走, 谁的子就在 x 大的那一侧。
   这只是"镜像"一件事 —— 但它决定了 V 是"走子方视角", 从而红黑共用一个网络。
*/
inline int canonicalCell(int x, int y, int color)
{
    return (color == Stone::COLOR_RED) ? (x * 9 + y) : ((9 - x) * 9 + y);
}

inline int canonicalCellOf(const Pos &p, int color) { return canonicalCell(p.x, p.y, color); }

/* 动作索引: 双射, 无碰撞 (8100) */
inline int actionIndexOf(int fromCell, int toCell) { return fromCell * CELLS + toCell; }

inline int actionIdxOf(const Step &s, int color)
{
    return actionIndexOf(canonicalCell(s.pos.x, s.pos.y, color),
                         canonicalCell(s.nextPos.x, s.nextPos.y, color));
}

/* ---- 规则上下文的标量（全部是当前局面的函数）---- */

/*
   剩余子力比例 (1 = 满子, 0 = 只剩两个将)。
   这是"动态棋子价值"的条件化输入: 网络自己学"残局兵升值", 没有任何 if-else。
*/
inline double materialPhase(const Chess &c)
{
    double sum = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = const_cast<Chess &>(c).m_children[(std::size_t)i];
        if (s == nullptr || s->alive == false || s->type == Stone::TYPE_JIANG) {
            continue;
        }
        sum += (double)s->value;
    }
    const double full = 7.0;   /* 一方满子 ≈ 3.5, 双方 ≈ 7 */
    const double p = sum / full;
    return (p < 0.0) ? 0.0 : ((p > 1.0) ? 1.0 : p);
}

/* 总手数 / 对局截断长度（reward 设计里的 REWARD_MAX_PLIES = 120） */
inline double tempoPhase(const Chess &c, int maxPlies = 120)
{
    const double p = (double)c.history.size() / (double)((maxPlies > 0) ? maxPlies : 120);
    return (p < 0.0) ? 0.0 : ((p > 1.0) ? 1.0 : p);
}

/* 无吃子进度: 60 回合 (120 半回合) 判和 */
inline double halfmovePhase(const Chess &c)
{
    const double p = (double)c.halfMoveClock / 120.0;
    return (p < 0.0) ? 0.0 : ((p > 1.0) ? 1.0 : p);
}

/*
   当前局面在 history 里被记录到的次数 (0/1/2/3...) —— 与 Chess::isRepetition()
   **完全同一窗口、同一判据** (chess.cpp: 窗口 = 最近 halfMoveClock 步, 判和阈值 = 3),
   于是 `repetitionCount(c) >= 3` ⇔ `c.isRepetition()`。
   特征的判据必须与引擎的判和判据逐字一致, 否则网络学的是另一个游戏。
   (history 存的是"每一手落子之后"的局面, 所以初始局面不在其中 —— 引擎的口径也是这样。)
*/
inline int repetitionCount(Chess &c)
{
    const int n = (int)c.history.size();
    if (n <= 0) {
        return 0;
    }
    const unsigned long long h = c.computeHash();
    int window = c.halfMoveClock;
    if (window > n) {
        window = n;
    }
    const int begin = n - window;
    int count = 0;
    for (int i = n - 1; i >= begin; i--) {
        if (c.history[(std::size_t)i].hash == h) { count++; }
    }
    return count;
}

/* 0 / 1/3 / 2/3 (1.0 = 判和, 那是终局, 不会作为叶子喂给 V) */
inline double repetitionPhase(Chess &c)
{
    const double p = (double)repetitionCount(c) / 3.0;
    return (p > 1.0) ? 1.0 : p;
}

/* 取景方是否被将军 (镜像不改变将军关系, 所以它不破坏规范视角的对称性) */
inline double checkPhase(Chess &c, int color)
{
    return c.isInCheck(color) ? 1.0 : 0.0;
}

/* 一次取齐 5 个上下文标量（顺序 = CTX_* 常量） */
inline void contextValues(Chess &c, int color, double out[CTX_COUNT], int maxPlies = 120)
{
    out[CTX_MATERIAL] = materialPhase(c);
    out[CTX_TEMPO]    = tempoPhase(c, maxPlies);
    out[CTX_HALFMOVE] = halfmovePhase(c);
    out[CTX_REPEAT]   = repetitionPhase(c);
    out[CTX_CHECK]    = checkPhase(c, color);
}

/* ---- 写平面 ---- */

inline void fillPlane(float *dst, int plane, float value)
{
    float *p = dst + (std::size_t)plane * CELLS;
    for (int c = 0; c < CELLS; c++) {
        p[c] = value;
    }
}

/*
   14 个棋子平面 (规范视角): 下标 = type*2 + (是己方 ? 0 : 1)。
   `dst` 的前 PIECE_PLANES*CELLS 个 float 会被**覆盖写** (调用方负责先清零其他平面)。
*/
inline void encodePiecePlanes(Chess &c, int color, float *dst)
{
    for (int p = 0; p < PIECE_PLANES; p++) {
        fillPlane(dst, p, 0.0f);
    }
    for (int i = 0; i < 32; i++) {
        Stone *s = c.m_children[(std::size_t)i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        if (s->type < 0 || s->type >= 7) {
            continue;
        }
        const int isMine = (s->color == color) ? 0 : 1;
        const int plane  = s->type * 2 + isMine;
        const int cell   = canonicalCell(s->pos.x, s->pos.y, color);
        dst[(std::size_t)plane * CELLS + cell] = 1.0f;
    }
}

/*
   一站式: 14 个棋子平面 + 选中的规则/阶段平面, **按 CTX_* 升序紧凑写出**。

   `ctxMask` 是 CTX_* 的位掩码 —— **必须显式给出**, 因为各 agent 的平面布局不同:
     * DQNAB  : 5 个全要 (子力/总手数/无吃子/重复/将军) -> 14..18 (共 19 平面)
     * EVAB   : 只要 3 个规则上下文 (无吃子/重复/将军) -> 14..16 (共 17 平面)
   (`encodeAllContext` 是第一版的写法: 无条件写 5 个。EVAB 的 STATE_DIM 只留了 3 个
    平面, 于是它把张量后面 180 个 float 写越界了 —— 实测直接堆损坏崩溃
    0xC0000374。这正是"公共组件必须让调用方把布局说清楚"的例子。)
*/
constexpr unsigned CTX_ALL   = 0x1Fu;                       /* 5 个全要 */
constexpr unsigned CTX_RULES = (1u << CTX_HALFMOVE) | (1u << CTX_REPEAT) | (1u << CTX_CHECK);

inline void encodeWithContext(Chess &c, int color, float *dst, int basePlane,
                              unsigned ctxMask, bool rulePlanes = true, int maxPlies = 120)
{
    float *base = dst + (std::size_t)basePlane * CELLS;
    encodePiecePlanes(c, color, base);
    double ctx[CTX_COUNT];
    if (rulePlanes) {
        contextValues(c, color, ctx, maxPlies);
    } else {
        /* 消融开关: 值置零而不是省掉平面 —— 输入维度必须保持不变 */
        for (int i = 0; i < CTX_COUNT; i++) { ctx[i] = 0.0; }
    }
    int out = 0;
    for (int i = 0; i < CTX_COUNT; i++) {
        if ((ctxMask & (1u << i)) == 0) {
            continue;
        }
        fillPlane(base, PIECE_PLANES + out, (float)ctx[i]);
        out++;
    }
}

/* 5 个全要的便捷入口 (DQNAB 的布局) */
inline void encodeComplete(Chess &c, int color, float *dst, int basePlane = 0,
                           bool rulePlanes = true, int maxPlies = 120)
{
    encodeWithContext(c, color, dst, basePlane, CTX_ALL, rulePlanes, maxPlies);
}

/* ---- 奖励 / 终局口径 ---- */

/*
   终局值换到"某一步走子方"视角。**红胜不是永远 +1** —— 全局写死"红胜"会让根节点的
   价值符号反掉 (零和博弈里最常见的符号错误)。
   实现**只有一份**, 就在 chess.h:177 —— 这里只是把它引进来, 免得各 agent 各抄一遍符号换算。
*/
using ::outcomeForMover;

} // namespace ChessState

#endif // CHESSSTATE_H
