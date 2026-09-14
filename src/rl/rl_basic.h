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
