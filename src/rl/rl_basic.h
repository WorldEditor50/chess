#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>
#include "tensor.hpp"

namespace RL {

class Transition
{
public:
    Tensor state;
    Tensor action;
    Tensor nextState;
    float reward;
    bool done;
    /*
       R2 (2026-09): **合法动作掩码** (actionDim 维, 1 = 合法), 可以为空。

       `legalMask`      : **当前局面** state 的合法动作
       `nextLegalMask`  : **下一局面** nextState 的合法动作

       两个局面必须分开存: 象棋里 s 与 s' 的合法集本来就不同 (走一步之后合法着法集合
       就变了), 拿当前局面的掩码去算 s' 的期望会把非法着法的价值也算进去 ——
       而那正是 R2 要干掉的东西。空 = 该局面走"全量口径" (旧行为)。

       只有 `RL::SAC` 用它 —— 那边把 PPO 的 R2(训练侧只算合法列) 搬了过来: 给了掩码
       就在合法集上归一 (Z ≡ 1)、而且非法列的梯度恰好为 0。其它 agent (DQN/PG/...)
       不给这两个字段, 于是行为与以前逐位相同 (空 = 全量口径)。
    */
    Tensor legalMask;
    Tensor nextLegalMask;
public:
    Transition(){}
    explicit Transition(const Tensor& s,
                        const Tensor& a,
                        const Tensor& s_,
                        float r,
                        bool d)
        :state(s), action(a), nextState(s_), reward(r), done(d){}
};

class Step
{
public:
    Tensor state;
    Tensor action;
    float reward;
    /*
       势能塑形 (Phase 2) 用: 该步**落子之后**局面的势能 Φ(s_{i+1}), 走子方视角
       (正 = 轮到走棋的一方占优), 取值 (-1,1)。

       为什么必须由记录方**在棋盘上**算好存进来: 轨迹里只留了编码后的张量, 事后无法
       从张量反推局面评估 —— 而 Φ 需要真实的棋盘 (材质 + 位置 + 将安全)。
       相邻两步之间 Φ_before(i+1) = Φ_after(i), 所以一条轨迹只需要这个字段, 配合
       首手之前的 Φ 初值, 就能在 commitEpisode 里一次性完成塑形。
       详见 stone.h 的 potentialReward() 与 docs/agents_design.md 的 Phase 2 一节。
    */
    float potential;
public:
    Step() : reward(0.0f), potential(0.0f) {}
    Step(const Tensor& s, const Tensor& a, float r)
        :state(s), action(a), reward(r), potential(0.0f) {}
};

}
#endif // RL_DEF_H
