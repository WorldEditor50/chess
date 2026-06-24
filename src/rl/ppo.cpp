#include "ppo.h"
#include "layer.h"
#include "loss.h"
#include "moe.hpp"

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99f),
     exploringRate(1.0f), learningSteps(0)
{
    /* Actor: state -> Tanh(hidden) -> Softmax(actionDim) */
    actorP = Net(MOE<8, 4>::_(stateDim, true),
                 LayerNorm<Sigmoid, LN::Pre>::_(stateDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    /* Critic: state -> Tanh(hidden) -> Linear(1) — scalar value V(s) */
    critic = Net(MOE<8, 4>::_(stateDim, true),
                 TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true),
                 Layer<Linear>::_(hiddenDim, 1, true, true));
}

RL::Tensor &RL::PPO::action(const Tensor &state)
{
    return actorP.forward(state);
}

float RL::PPO::value(const Tensor &state)
{
    RL::Tensor &v = critic.forward(state);
    return v[0];
}

void RL::PPO::trainStep(const Tensor &state,
                        const Tensor &actionTarget,
                        float valueTarget,
                        float lr)
{
    /* ---- Actor: cross-entropy loss ---- */
    Tensor &policy = actorP.forward(state);
    Tensor ceLoss = Loss::CrossEntropy::df(policy, actionTarget);
    actorP.backward(state, ceLoss);

    /* ---- Critic: MSE loss ---- */
    Tensor &v = critic.forward(state);
    Tensor valueTargetTensor(1, 1);
    valueTargetTensor[0] = valueTarget;
    Tensor mseLoss = Loss::MSE::df(v, valueTargetTensor);
    critic.backward(state, mseLoss);

    /* ---- Update both networks ---- */
    actorP.RMSProp(lr, 0.9f, 0.001f);
    critic.RMSProp(lr, 0.9f, 0.001f);
    learningSteps++;
}

void RL::PPO::learnSelfPlay(std::vector<Step>& trajectory,
                            float finalOutcome,
                            float learningRate)
{
    int end = (int)trajectory.size() - 1;
    if (end < 0) return;

    /* Compute discounted returns from the end */
    std::vector<float> returns(trajectory.size(), 0.0f);
    float r = finalOutcome;
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        returns[i] = r;
    }

    /* Train each step: policy target = action taken (one-hot), value target = return */
    for (int t = 0; t <= end; t++) {
        trainStep(trajectory[t].state,
                  trajectory[t].action,
                  returns[t],
                  learningRate);
    }

    exploringRate *= 0.99999f;
    if (exploringRate < 0.01f) exploringRate = 0.01f;
}

void RL::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    actorP.save(actorPara);
    critic.save(criticPara);
}

void RL::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    actorP.load(actorPara);
    critic.load(criticPara);
}
