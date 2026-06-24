#include "sac.h"
#include "layer.h"
#include "loss.h"
#include <limits>

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
    annealing = ExpAnnealing(0.01, 0.12, 1e-4);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    /* target entropy = -actionDim (standard SAC heuristic) */
    entropy0 = -actionDim;
    actor = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    for (int i = 0; i < QNET_NUM; i++) {
        critics[i] = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                         TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                         Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));
        criticsTarget[i] = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                               TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                               Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
        /* Add noise to break symmetry between critics */
        if (i > 0) {
            critics[i].softUpdateTo(criticsTarget[i], 0.1);
        }
        critics[i].copyTo(criticsTarget[i]);
    }
}

void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor &RL::SAC::eGreedyAction(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return eGreedy(out, exploringRate, true);
}

RL::Tensor &RL::SAC::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return RL::gumbelSoftmax(out, alpha.val);
}

RL::Tensor& RL::SAC::action(const RL::Tensor &state)
{
    return actor.forward(state);
}

void RL::SAC::experienceReplay(const RL::Transition &x)
{
    /* ================================================
     * 1. Compute Q-target using Clipped Double Q-learning
     *    with expectation over actions (not argmax)
     * ================================================ */
    {
        /* Get policy probabilities for next state */
        const Tensor& nextProb = actor.forward(x.nextState);

        /* Compute min Q over all target critics for each action dimension */
        Tensor minQ(actionDim, 1);
        for (int j = 0; j < actionDim; j++) {
            float q_min = std::numeric_limits<float>::max();
            for (int i = 0; i < QNET_NUM; i++) {
                const Tensor& qi = criticsTarget[i].forward(x.nextState);
                q_min = std::min(q_min, qi[j]);
            }
            minQ[j] = q_min;
        }

        /* V(s') = Σ π(a'|s') * (minQ(s',a') - α*log(π(a'|s'))) */
        float nextValue = 0;
        for (int j = 0; j < actionDim; j++) {
            float logp = std::log(nextProb[j] + 1e-8);
            nextValue += nextProb[j] * (minQ[j] - alpha.val[j] * logp);
        }

        /* Compute Q-target for the taken action */
        std::size_t k = x.action.argmax();
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma * nextValue;
        }

        /* Train each critic with MSE loss on the taken action only */
        for (int i = 0; i < QNET_NUM; i++) {
            const Tensor &out = critics[i].forward(x.state);
            Tensor p = out;
            p[k] = qTarget;
            critics[i].backward(x.state, Loss::MSE::df(out, p));
        }
    }

    /* ================================================
     * 2. Train Policy Net
     *    J(π) = Σ π(a|s) * (α*log(π(a|s)) - minQ(s,a))
     *    gradient w.r.t π output: α*log(π) + α - Q
     * ================================================ */
    {
        /* Get current policy and min Q for current state */
        const Tensor& p = actor.forward(x.state);

        /* Compute min Q over all critics (not target) for current state */
        Tensor minQ(actionDim, 1);
        for (int j = 0; j < actionDim; j++) {
            float q_min = std::numeric_limits<float>::max();
            for (int i = 0; i < QNET_NUM; i++) {
                const Tensor& qi = critics[i].forward(x.state);
                q_min = std::min(q_min, qi[j]);
            }
            minQ[j] = q_min;
        }

        /* SAC policy gradient error on softmax output:
         * dJ/dπ(a|s) = α*log(π(a|s)) + α - Q(s,a)
         * The softmax layer's gradient function handles the
         * backpropagation through the softmax nonlinearity. */
        Tensor err(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            float logp = std::log(p[i] + 1e-8);
            err[i] = alpha.val[i] * logp + alpha.val[i] - minQ[i];
        }
        actor.backward(x.state, err);
    }

    /* ================================================
     * 3. Update Temperature (alpha)
     *    J(α) = -α * (H - H₀) where H = Σ π*log(π)
     *    ∇α = -(H + actionDim) = logπ_avg + actionDim
     *        = -(Σ π*log(π) - entropy0)
     * ================================================ */
    {
        const Tensor& p = actor.forward(x.state);
        /* Compute current entropy H = -Σ π(a|s)*log(π(a|s)) */
        float H = 0;
        for (int i = 0; i < actionDim; i++) {
            H -= p[i] * std::log(p[i] + 1e-8);
        }
        /* ∇α = -(H - H₀) = H₀ - H */
        float alphaGrad = entropy0 - H;
        for (int i = 0; i < actionDim; i++) {
            alpha.g[i] += alphaGrad;
        }
    }
    return;
}

void RL::SAC::learn(size_t maxMemorySize, size_t replaceTargetIter, size_t batchSize, float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        /* Polyak averaging with consistent tau for all critics */
        float tau = 5e-3;
        for (int i = 0; i < QNET_NUM; i++) {
            critics[i].softUpdateTo(criticsTarget[i], tau);
        }
        learningSteps = 0;
    }

    /* Experience replay with random mini-batch */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* Apply gradient updates */
    actor.RMSProp(1e-2, 0.9, 0);

#if 1
    std::cout<<"annealing:"<<annealing.val<<",alpha:";
    alpha.val.printValue();
#endif

    alpha.RMSProp(1e-5, 0.9, 0);
    /* Keep alpha in reasonable range */
    alpha.clamp(0.01, 20.0);

    for (int i = 0; i < QNET_NUM; i++) {
        critics[i].RMSProp(1e-3, 0.9, 0);
    }

    /* manage replay buffer: drop oldest entries when full */
    if (memories.size() > maxMemorySize + batchSize) {
        std::size_t k = std::min(batchSize, memories.size() - maxMemorySize);
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.3 ? 0.3 : exploringRate;
    learningSteps++;
    return;
}

void RL::SAC::save()
{
    actor.save("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].save(critiscName);
    }
    return;
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].load(critiscName);
        critics[i].copyTo(criticsTarget[i]);
    }
    return;
}
