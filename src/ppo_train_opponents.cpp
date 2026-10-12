/*
 * ppo_train_opponents.cpp — 混合对手训练池的**构造**（2026-10 第四轮）
 * ============================================================================
 *
 * 为什么这一段单独一个 TU（而不是写在 ppomcts_agent.cpp 里）：
 *   构造对手池需要 `MCTS` 与 `ABAgent` 两个类（它们的构造函数与 `findBestMove`
 *   定义在 mcts.cpp / abagent.cpp 里）。把它们拖进 ppomcts_agent.cpp，会让
 *   **15 个**已经链接了该文件的目标（train_ppo / bench_ppo_* / test_diag /
 *   test_sacaz_aligned …）突然都需要再加两个 .cpp —— 而其中大多数根本不用这个功能。
 *   放到独立 TU 之后：**只有真正打开混合对手的目标**才需要加这三个文件
 *   （CHESS_CORE 与 train_ppo / test_mixed_opponents），其余一行都不用改。
 *
 * 用户口径（原文）："实现 mcts, abagent-1,2,3,4 混合对局, mcts 对局数量占 50%,
 * 其余 abagent 平分, 按照均匀分布采样选择 agent 对弈训练"。
 * 语义、两条边界（对手的手不进样本 / 每局换先手）与默认行为见
 * `src/ppomcts_agent.h` 的"训练对手池"一节；这里是它的落地。
 *
 * ⚠ 一个实现上的细节：`trainSelfPlay` 里**只通过虚函数** `AgentBase::getBestMove`
 *   驱动对手（不 static_cast 到 MCTS）—— 否则 ppomcts_agent.cpp 又会重新引用
 *   `MCTS::findBestMove` 这个符号，把 15 个目标的链接问题带回来。
 *   于是"对手的模拟预算"由这里的 `BudgetMcts` 子类实现：它是 MCTS 的一个薄壳，
 *   把 `getBestMove` 覆写成 `findBestMove(color, budget)`。
 */
#include "ppomcts_agent.h"

#include "abagent.h"
#include "mcts.h"

namespace {

/*
   带预算的 MCTS 对手。
   为什么需要它: `MCTS::getBestMove(color)` 写死了 800 次迭代（= 界面 MCTS agent 的
   预算），而 `findBestMove(color, iterations)` 是**非虚**的成员 —— 在 ppomcts_agent.cpp
   里直接调它就会产生对 mcts.cpp 的符号依赖。这个薄壳把预算做成虚函数路径上的东西，
   于是"混合对手"这条链路只需要**一个** TU 知道 MCTS 的存在。
*/
class BudgetMcts : public MCTS
{
public:
    BudgetMcts(Chess &env, int iterations)
        : MCTS(env, 1.414), iters(iterations > 0 ? iterations : 800) {}
    Step getBestMove(int color) override { return findBestMove(color, iters); }
    std::string getName() const override
    {
        return std::string("MCTS(") + std::to_string(iters) + ")";
    }
private:
    int iters;
};

}  // namespace

void PPOMCTSAgent::enableMixedOpponentTraining(float mctsShare, int mctsIterations,
                                               int abMaxDepth)
{
    /*
       与用户口径逐条对齐（见头文件那一节）：
         * MCTS 拿到 `mctsShare`（默认 0.5）；
         * 剩下的 `1 - mctsShare` **平分**给深度 1..abMaxDepth 的 AB（默认 4 档 ⇒ 各 0.125）；
         * 采样用 `RL::Random`（在 trainSelfPlay 里），同一 seed 下对手序列可复现。
       参数越界时**纠正到合法区间并写进读数**，而不是静默换一种分布训练 ——
       "以为在用 50/12.5x4，其实全抽到 MCTS" 这种偏差在损失曲线上完全看不出来。
    */
    if (mctsShare < 0.0f) { mctsShare = 0.0f; }
    if (mctsShare > 1.0f) { mctsShare = 1.0f; }
    if (abMaxDepth < 0) { abMaxDepth = 0; }
    m_mctsShare = mctsShare;
    m_mctsIterations = (mctsIterations > 0) ? mctsIterations : 800;

    m_opponents.clear();
    m_oppGames = 0;
    if (mctsShare > 0.0f) {
        TrainOpponent o;
        o.name = std::string("MCTS(") + std::to_string(m_mctsIterations) + ")";
        o.weight = mctsShare;
        o.abDepth = 0;
        o.agent = std::make_shared<BudgetMcts>(chess, m_mctsIterations);
        m_opponents.push_back(o);
    }
    const int nAb = abMaxDepth;
    const float abW = (nAb > 0) ? (1.0f - mctsShare) / (float)nAb : 0.0f;
    for (int d = 1; d <= nAb; d++) {
        /*
           **权重为 0 的档位不建** (mctsShare=1 时 AB 的权重全是 0): 采样上它们永远不会
           被抽到, 但留在池子里会让读数变成"5 个对手 (其中 4 个权重 0)" —— 而读数是给人
           看的, 它必须与"实际会被抽到的对手"一一对应。也省下 4 个 ABAgent 实例。
        */
        if (abW <= 0.0f) { break; }
        TrainOpponent o;
        o.name = std::string("AB-L") + std::to_string(d);
        o.weight = abW;
        o.abDepth = d;
        o.agent = std::make_shared<ABAgent>(chess, d);
        m_opponents.push_back(o);
    }
    /*
       权重归一（浮点上界）—— 采样是按累积权重取的，和不等于 1 会让"最后一个对手"
       多吃一点（差多少吃多少）。这里显式归一，让"权重"与"概率"永远是一回事。
    */
    float sum = 0.0f;
    for (std::size_t i = 0; i < m_opponents.size(); i++) { sum += m_opponents[i].weight; }
    if (sum > 0.0f) {
        for (std::size_t i = 0; i < m_opponents.size(); i++) { m_opponents[i].weight /= sum; }
    }
    m_mixedOpponents = !m_opponents.empty();
}

void PPOMCTSAgent::disableMixedOpponentTraining()
{
    m_opponents.clear();
    m_mixedOpponents = false;
    m_oppGames = 0;
}

std::vector<PPOMCTSAgent::OpponentUse> PPOMCTSAgent::trainOpponentUsage() const
{
    std::vector<OpponentUse> out;
    out.reserve(m_opponents.size());
    for (std::size_t i = 0; i < m_opponents.size(); i++) {
        OpponentUse u;
        u.name = m_opponents[i].name;
        u.weight = m_opponents[i].weight;
        u.games = m_opponents[i].games;
        u.plies = m_opponents[i].plies;
        u.unmatched = m_opponents[i].unmatched;
        out.push_back(u);
    }
    return out;
}
