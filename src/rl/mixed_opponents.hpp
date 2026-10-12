#ifndef RL_MIXED_OPPONENTS_HPP
#define RL_MIXED_OPPONENTS_HPP
/*
 * mixed_opponents.hpp — **混合对手池**的公共件（权重 / 采样 / 统计）
 * ============================================================================
 *
 * 用户口径（原文）："实现 mcts, abagent-1,2,3,4 混合对局, mcts 对局数量占 50%,
 * 其余 abagent 平分, 按照均匀分布采样选择 agent 对弈训练"，后续追加：
 * "增加 B 方混合 agent"。
 *
 * 同一个口径有两处使用者，**采样与权重必须完全一致**（否则"训练时的对手分布"与
 * "对局时的对手分布"会漂开，而两者的读数会被放在一起比较）：
 *   1. **训练**：`PPOMCTSAgent::trainSelfPlay` 的陪练（见 `ppo_train_opponents.cpp`
 *      与 `ppomcts_agent.h` 的"训练对手池"一节）—— 学习方 vs 抽中的陪练；
 *   2. **对局**：界面 A/B 下拉框里的 **`AGENT_MIXED_POOL`**（见 chessboard.cpp）——
 *      被选中的那一方**每局**从池子里抽一个真 agent 来下这一局。
 * 所以抽出来一个头文件：第二份拷贝出现时才抽文件，这是本工程的惯例。
 *
 * 三条由本文件**唯一定义**的东西：
 *   * **权重**：MCTS 拿 `mctsShare`（默认 0.5），剩下的**平分**给 AB 深度 1..abMaxDepth
 *     （默认 4 档 ⇒ 每档 0.125）；和为 1（归一化在 `defaultPool` 里做）。
 *   * **采样**：一次 `U(0,1)` 落在权重累积区间的那一段（`sampleIndex` 是**纯函数**，
 *     便于用直方图断言钉住"均匀分布"）。
 *   * **名字**：`"MCTS(800)"` / `"AB-L1"` … —— 报告与日志里必须能一眼看出抽到了谁。
 *
 * ⚠ 使用者的责任（本文件不碰随机数引擎）：**在同一颗 `RL::Random` 引擎上取 U(0,1)**，
 *   这样同一 seed 下对手序列可复现（本工程所有随机性的统一口径）。
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace RL {
namespace MixedPool {

/* 池子的一项：名字 / 权重 / "怎么造出这一支"的标识 */
struct Entry
{
    std::string name;      /* "MCTS(800)" / "AB-L1" … */
    float weight = 0.0f;   /* 采样概率（`defaultPool` 归一化后和为 1） */
    /*
       `abDepth == 0` ⇒ 这一项是 MCTS（预算在 name 里写着，使用者自己知道怎么建）；
       `abDepth > 0` ⇒ ABAgent 的搜索深度。
       （用 int 而不是枚举：两处使用者的 agent 类不同 —— 训练侧建 `BudgetMcts`/`ABAgent`，
       对局侧建界面的 `AGENT_MCTS`/`AGENT_AB_L*`，所以这里只描述"是什么"。）
    */
    int abDepth = 0;
    int mctsSims = 0;      /* 仅 MCTS 项用（0 = 使用者的默认预算） */
};

/*
   默认池：MCTS 占 `mctsShare`，其余平分给深度 1..`abMaxDepth`。
   ⚠ **权重为 0 的档位不生成**（mctsShare=1 时没有 AB、mctsShare=0 时没有 MCTS）：
   留在池子里会让读数变成"5 个对手（其中 4 个权重 0）"，而读数必须与"实际会被抽到的
   对手"一一对应（与训练侧那条口径相同）。
*/
inline std::vector<Entry> defaultPool(float mctsShare = 0.5f, int abMaxDepth = 4,
                                      int mctsSims = 800)
{
    if (mctsShare < 0.0f) { mctsShare = 0.0f; }
    if (mctsShare > 1.0f) { mctsShare = 1.0f; }
    if (abMaxDepth < 0) { abMaxDepth = 0; }

    std::vector<Entry> pool;
    if (mctsShare > 0.0f) {
        Entry e;
        e.name = std::string("MCTS") + (mctsSims > 0 ? ("(" + std::to_string(mctsSims) + ")")
                                                     : std::string());
        e.weight = mctsShare;
        e.abDepth = 0;
        e.mctsSims = mctsSims;
        pool.push_back(e);
    }
    const float abW = (abMaxDepth > 0) ? (1.0f - mctsShare) / (float)abMaxDepth : 0.0f;
    for (int d = 1; d <= abMaxDepth && abW > 0.0f; d++) {
        Entry e;
        e.name = std::string("AB-L") + std::to_string(d);
        e.weight = abW;
        e.abDepth = d;
        pool.push_back(e);
    }
    /* 归一化：采样按累积权重取，和不等于 1 会让"最后一个对手"多吃一点 */
    float sum = 0.0f;
    for (std::size_t i = 0; i < pool.size(); i++) { sum += pool[i].weight; }
    if (sum > 0.0f) {
        for (std::size_t i = 0; i < pool.size(); i++) { pool[i].weight /= sum; }
    }
    return pool;
}

/*
   采样：`u ∈ [0,1)` 落在哪一段累积权重上。返回 -1 = 空池（使用者应回退到"不抽"）。
   刻意做成**纯函数**（不碰随机数引擎）：分布性质因此可以用"喂一批 u"直接断言，
   而两处使用者共用它 ⇒ 训练与对局的对手分布**同源**。
*/
inline int sampleIndex(const std::vector<Entry> &pool, float u)
{
    if (pool.empty()) { return -1; }
    float acc = 0.0f;
    for (std::size_t i = 0; i < pool.size(); i++) {
        acc += pool[i].weight;
        if (u <= acc || i + 1 == pool.size()) { return (int)i; }
    }
    return (int)pool.size() - 1;
}

/* "MCTS 0.500 + AB-L1 0.125 + …" —— 报告/日志里的一行说明 */
inline std::string describe(const std::vector<Entry> &pool)
{
    std::string s;
    for (std::size_t i = 0; i < pool.size(); i++) {
        if (i > 0) { s += " + "; }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s %.3f", pool[i].name.c_str(),
                      (double)pool[i].weight);
        s += buf;
    }
    return s;
}

/* 逐对手的计数（两处使用者各自的统计表都用它） */
struct Tally
{
    std::string name;
    float weight = 0.0f;
    int games = 0;
    int plies = 0;
};

/*
   `abDepth` -> **界面上的 AB 档位序号**（对局侧用; 训练侧直接用深度建 ABAgent, 不经此处）。

   ⚠ 这个函数存在的唯一原因是一次真实事故口径: 界面枚举里 AB 只有 **L1/L2/L3** 三档
   （深度 1/2/3）, **深度 4 就是下拉框里那一档"Alpha-Beta"**（AB_DEPTH=4）, 它**不是**
   `AGENT_AB_L3 + 1`。曾经对局侧写的是 `AGENT_AB_L1 + (abDepth - 1)`, 于是"AB-L4"落到
   枚举值 16 = **AGENT_SACAZ_MOE_MLP** —— 一个完全不同的 agent（SAC+AZ 稀疏 MoE）被当成
   "深度 4 的 AB"上场, 而报告里那一行仍写着 AB-L4, 甚至因为对方有"学习口径"奖励而
   打印出 SAC 的奖励数字: 看着正常、归因全错。所以映射必须**显式**列出来:
       返回 1..3 ⇒ 第 k 档 (AGENT_AB_L1 + k - 1)
       返回 0    ⇒ "主 Alpha-Beta 档" (深度 4)
       返回 -1   ⇒ 界面上**没有**这一档 (调用方必须报警, 不许静默取近似)
*/
inline int matchAbLevelOfDepth(int abDepth)
{
    switch (abDepth) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    case 4: return 0;      /* 深度 4 == 主 Alpha-Beta 档 */
    default: return -1;    /* 5 及以上: 界面上没有对应档位 */
    }
}

} /* namespace MixedPool */
} /* namespace RL */

#endif // RL_MIXED_OPPONENTS_HPP
