#ifndef DQNN_H
#define DQNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include "net.hpp"
#include "rl_basic.h"

namespace RL {

class DQN
{
public:
    DQN(){}
    explicit DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    Tensor& eGreedyAction(const Tensor& state);
    Tensor& noiseAction(const Tensor& state);
    Tensor& action(const Tensor &state);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
    /*
       chess-side divergence from snakeAI/rl/dqn.h:
       upstream made these members `protected`, but the chess agents
       (src/dqnagent.cpp, src/dqnmcts_agent.cpp, src/dqnagent.h:getExploreRate,
       src/dqnmcts_agent.h:getExploreRate) read/write `gamma` and `exploringRate`
       directly. They are kept public here so the chess agent code keeps compiling;
       tighten them to `protected` only together with adding accessors.
    */
public:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    int learningSteps = 0;
    /*
     * 最近一次 learn() 的平均平方 TD 误差 ("训练损失", 只给界面画曲线用)。
     * NaN = 还没学过。**不参与任何训练计算** —— 加它是因为象棋 RL 的训练在 GUI 里
     * 是后台不可见的, 没有一条损失曲线就完全看不出"到底有没有在学"。
     */
    double lastLoss = std::numeric_limits<double>::quiet_NaN();
    /* learn() 内部用: 本次批量的 TD 误差累加 */
    double lossSum = 0.0;
    int lossCount = 0;
    Net QMainNet;
    Net QTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DQNN_H
