#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"

namespace RL {

class DPG
{
public:
    DPG(){}
    explicit DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    Tensor &eGreedyAction(const Tensor &state);
    Tensor &noiseAction(const Tensor &state);
    Tensor &gumbelMax(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    void reinforce(std::vector<Step>& x, float learningRate);
    void reinforce1(std::vector<Step>& x, float learningRate);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
    /*
       chess-side divergence from snakeAI/rl/dpg.h:
       upstream made these members `protected`, but src/pgagent.cpp reads and
       writes `exploringRate` / `learningRate` and calls `policyNet.save/load`
       directly. They are kept public here so the chess agent code keeps
       compiling; tighten them to `protected` only together with adding
       accessors to PGEagent.
    */
public:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    float H0;
    GradValue alpha;
    Net policyNet;
    /*
     * 最近一次 reinforce1 的**策略梯度损失** (界面"训练损失曲线"用, 不参与计算):
     *    surrogate = -Σ_t A_t · log π(a_t|s_t)   (取平均)
     * REINFORCE 没有"误差"这种量, 这个代替目标就是它的标准损失 —— 策略越把概率放到
     * 正优势的动作上, 它越小。advantage 已经在 reinforce1 里被标准化过, 所以它的
     * 绝对量级只表示"信噪比", 跨 agent 之间不可比 (曲线是按 agent 分线的)。
     */
    double lastLoss = std::numeric_limits<double>::quiet_NaN();
};
}
#endif // POLICY_GRADIENT_H
