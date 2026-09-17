#include "dqn.h"
#include "layer.h"
#include "loss.h"
#include "concat.hpp"
#include "attention.hpp"
#include "transformer.hpp"
#include "moe.hpp"

RL::DQN::DQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
    /*
       Q 头必须是**线性**(无界)的, 不能是 Sigmoid  (2026-09 补修, 与 convdqn.cpp 对齐)。

       原来的 QMainNet/QTargetNet 最后一层是 `Layer<Sigmoid>`, 值域被压到 (0,1)。而象棋
       这边的奖励里有**负值**: 终局输棋是 -1、被吃红子按 `R_STEP_*` 一类的负项进账,
       TD 目标因此经常为负 —— 网络在结构上根本表示不出来。convdqn.cpp 早就因为同一个
       理由 (snake 的奖励以负为主) 改成了 Linear 并在那边留了实测: 旧的 Sigmoid 配置下
       Q 全部卡在 0.000、四选一的贪心正确率等于瞎猜, 因为负目标把 sigmoid 推进饱和区、
       导数 ≈ 0, 梯度直接死掉。dqn.cpp 当时漏改了, 这里补齐 (四个候选骨干分支一起改,
       避免换分支时又把 bug 换回来)。
    */
#if 1
    /*
       chess-side divergence from snakeAI/rl/dqn.cpp.
       The chess DQN used an MOE<16,16> + TransformerBlock<16> backbone (the
       comment that used to sit here still described the 3-layer MOE<8,4>
       network, which is why it looked like a mismatch). That architecture is
       kept ACTIVE so no chess-side work is lost; the three snakeAI variants
       follow below as #if 0 branches. They are mutually exclusive — move the
       `1` to the branch you want.

       NOTE: TransformerBlock<16> / MOE<16,16> over the 90-wide chess state only
       works with the CURRENT rl/attention.hpp, which clamps the head count to
       the largest divisor of d_model (15 heads, d_k = 6) instead of rounding
       d_model UP to d_k*NumHeads. With the previous attention.hpp, d_model was
       silently bumped from 90 to 96 while TransformerBlock kept 90-wide cached
       buffers, so this branch wrote and read out of bounds.
    */
    QMainNet = Net(MOE<16, 16>::_(stateDim, true),
                   TransformerBlock<16>::_(stateDim, true),
                   TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true),
                   Layer<Linear>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(MOE<16, 16>::_(stateDim, false),
                     TransformerBlock<16>::_(stateDim, false),
                     TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, false),
                     Layer<Linear>::_(hiddenDim, actionDim, true, false));
#elif 0
    /* snakeAI variant A: 4 trainable layers, no MOE */
    QMainNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                   Layer<Linear>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                     Layer<Linear>::_(hiddenDim, actionDim, true, false));
#elif 0
    /* snakeAI variant B: 16-way ScaledConcat feature extractor

       2026-09: `ScaledConcat` 重写过 (门控与特征解耦 / 门控 logits 无界 / 初始化按
       fan-in 缩放 / 专家类型是模板参数), 所以这一分支跟着换成新 API:
       `ScaledConcat<专家类型, 专家数, 每路单元数>`, 输出宽度仍是 16*4 = 64, 与下面
       那层 TanhNorm(16*4, hiddenDim) 对得上。
       专家取 MlpExpert (隐层 16): 一个才 ~1.6K 参数, 而 d->d 的 Layer<Gelu> 专家是
       90x90 = 8.2K/个。 */
    QMainNet = Net(ScaledConcat<MlpExpert, 16, 4>::_(stateDim, true, 16),
                   TanhNorm<Sigmoid>::_(16*4, hiddenDim, true, true),
                   Layer<Linear>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(ScaledConcat<MlpExpert, 16, 4>::_(stateDim, false, 16),
                     TanhNorm<Sigmoid>::_(16*4, hiddenDim, true, false),
                     Layer<Linear>::_(hiddenDim, actionDim, true, false));
#else
    /* snakeAI audited default: MOE<8,4> + TanhNorm */
    QMainNet = Net(MOE<8, 4>::_(stateDim, true),
                   TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true),
                   Layer<Linear>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(MOE<8, 4>::_(stateDim, false),
                     TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, false),
                     Layer<Linear>::_(hiddenDim, actionDim, true, false));
#endif
    QMainNet.copyTo(QTargetNet);
}

void RL::DQN::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor& RL::DQN::eGreedyAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor& RL::DQN::noiseAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return noise(out, exploringRate);
}

RL::Tensor &RL::DQN::action(const Tensor &state)
{
    return QMainNet.forward(state);
}

void RL::DQN::experienceReplay(const Transition& x)
{
    /*
     * IMPORTANT: Compute Q-target BEFORE the QMainNet.backward() call.
     *
     * CRITICAL BUG FIX: We must save QMainNet.forward(x.state) results
     * BEFORE calling QMainNet.forward(x.nextState), because forward()
     * overwrites the network's internal cached o-values. If we call
     * forward(x.nextState) first, then backward(x.state, ...) uses the
     * wrong cached intermediate values, corrupting all gradients.
     *
     * The safe ordering is:
     *   1. Forward x.state → out (deep copy)
     *   2. Compute qTarget from out (already have action index i)
     *   3. Call forward(x.nextState) only on QTargetNet (separate network)
     *   4. Backward on QMainNet with x.state → uses correctly cached state
     */
    int i = x.action.argmax();

    if (x.done) {
        /* Terminal state: Q-target = reward directly */
        Tensor out = QMainNet.forward(x.state);
        Tensor qTarget = out;
        qTarget[i] = x.reward;
        /*
           顺便把这次 TD 误差记下来 (只给界面画"训练损失曲线"用, 不参与任何计算)。
           批量里逐样本累加, learn() 结束时除以 batchSize 得到平均 TD 误差。
        */
        const double td = (double)out[i] - (double)qTarget[i];
        lossSum += td * td;
        lossCount++;
        QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    } else {
        /*
         * Non-terminal: need max_a' Q(s',a')
         * Use QTargetNet to evaluate, QMainNet to select action.
         * QMainNet.forward(x.state) saves the cached forward for backward.
         * CRITICAL: Don't call QMainNet.forward(x.nextState) BEFORE backward!
         */
        /* Forward on x.state first (caches values for backward) */
        Tensor out = QMainNet.forward(x.state);
        Tensor qTarget = out;

        /* Select best next action using QMainNet on SEPARATE network copy
           to avoid overwriting the cached x.state forward in QMainNet */
        int k = QTargetNet.forward(x.nextState).argmax();

        /* Evaluate next-state value using QTargetNet */
        Tensor &v = QTargetNet.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];

        const double td = (double)out[i] - (double)qTarget[i];
        lossSum += td * td;
        lossCount++;

        /* Backward on QMainNet with x.state (cached values intact) */
        QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    }
    return;
}

void RL::DQN::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    lossSum = 0.0;
    lossCount = 0;

    /* update target network periodically (Polyak soft update) */
    if (learningSteps % replaceTargetIter == 0) {
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        learningSteps = 0;
    }

    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* apply optimizer */
    QMainNet.RMSProp(learningRate, 0.9, 0);

    /* 本批的平均 TD 误差 (界面曲线用, 见 dqn.h 的 lastLoss) */
    if (lossCount > 0) {
        lastLoss = lossSum / (double)lossCount;
    }

    /* manage replay buffer: drop oldest entries when full */
    if (memories.size() > maxMemorySize + batchSize) {
        std::size_t k = std::min(batchSize, memories.size() - maxMemorySize);
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }

    /* decay exploring rate: 0.9999^n: 1→0.5 at ~6931 steps, 1→0.1 at ~23026 steps */
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::DQN::save(const std::string &fileName)
{
    QMainNet.save(fileName);
    return;
}

bool RL::DQN::load(const std::string &fileName)
{
    const int r = QMainNet.load(fileName);
    if (r != 0) {
        /*
           载入被拒 (结构指纹 / CRC 不匹配等) ⇒ 网络**没有被改动**。此时**不能**把它
           拷到目标网: 那会把"上次成功载入的权重"覆盖成"当前主网", 而两者此刻未必
           一致。直接返回失败, 让调用方(各 agent 的 loadModel)把失败传到上层。
        */
        std::cerr << "[weights] DQN::load 失败, 网络保持不变 (文件: "
                  << fileName << ")" << std::endl;
        return false;
    }
    QMainNet.copyTo(QTargetNet);
    return true;
}
