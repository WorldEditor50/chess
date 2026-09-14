#include "dpg.h"
#include "layer.h"
#include "loss.h"
#include "attention.hpp"
#include "transformer.hpp"
#include "moe.hpp"

RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.9), exploringRate(1)
{
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    H0 = RL::entropy(0.25);
#if 0
    policyNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
#else
    /*
       MOE<NumExperts, NumHeads> with d_model == stateDim. The head count must
       divide d_model: this used to be MOE<4, 8> with stateDim = 2, i.e. 8 heads
       for a 2-wide model. MultiHeadAttention now clamps itself to the largest
       divisor of d_model (2 heads here) instead of silently resizing d_model,
       but the template argument is written to match d_model so the intent is
       explicit.

       第二层由 `Layer<Tanh>` **换成 `TransformerBlock<16>`** (2026-09): 与
       rl/sac.cpp 里 actor/critics 的搭法一致 (MOE + TransformerBlock + 一层线性),
       同一套骨干超参在两处可比。TB 不改变宽度, 所以进 LN::Pre 的激活仍是 stateDim
       宽; "撑开/投影到 hiddenDim"这一步现在由下面那层
       `LayerNorm<Sigmoid, LN::Pre>::_(stateDim, hiddenDim, ...)` 负责。

       这段历史必须留着 (它解释了为什么中间**必须**有一层模块): `LN::Pre` 标准化的是
       它的**输入**, 曾经直接坐在 2 宽的 state 上 (inputDim = stateDim = 2), 于是
       (x - mean)/std = [d/2, -d/2] (d = x0 - x1) —— 2 维状态被压到一个方向上,
       后面的投影退化成秩 1, 策略会周期性收敛成与状态无关的分布 ("永远走 action 0")。
    */

    policyNet = Net(MOE<16, 16>::_(stateDim, true),
                    TransformerBlock<16>::_(stateDim, true),
                    LayerNorm<Sigmoid, LN::Pre>::_(stateDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
#endif
}

RL::Tensor &RL::DPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::DPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return noise(out);
}

RL::Tensor &RL::DPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::DPG::action(const Tensor &state)
{
    return policyNet.forward(state);
}

void RL::DPG::reinforce(std::vector<Step>& x, float learningRate)
{
    /*
       This is the estimator the shipped game actually trains with — it was
       measured to play better than reinforce1() below. The two differ by more
       than side effects: with CrossEntropy::df(out, t)[i] = -t[i]/out[i] and the
       in-place edit x[t].action[k] = p_k * A_t (p_k = the probability the
       sampled Gumbel-Softmax gave to the action actually taken), the logit
       update here is

           dz = eta * p_k * A_t * (e_k - pi)

       whereas reinforce1() produces

           dz = eta *        A_t * (e_k - pi)

       So this version weights each step by how confident the policy was about
       the action it took. Both are kept; the test suite pins down reinforce1().
    */
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    double lossSum = 0.0;   /* 界面曲线的策略梯度损失 (见 dpg.h 的 lastLoss) */
    for (std::size_t t = 0; t < x.size(); t++) {
        const Tensor &prob = x[t].action;
        int k = x[t].action.argmax();
        float H = RL::entropy(prob[k]);
        alpha.g[k] += H0 - H;
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state);
        Tensor dLoss = Loss::CrossEntropy::df(out, x[t].action);
        /*
           记录策略梯度损失 (只给界面画曲线用): 这里用的是与 dLoss 同一个
           加权交叉熵, 所以它和实际优化的是同一个量。
           注: 界面上的 PG agent 走的是这个 reinforce() (agentrollout 那条路),
           不是 reinforce1() —— 两处都要填 lastLoss, 否则损失曲线还是空的。
        */
        const double w = (double)x[t].action[k];
        lossSum += -w * std::log((double)out[k] + 1e-8);
        policyNet.backward(x[t].state, dLoss);
    }
    lastLoss = lossSum / (double)x.size();
    alpha.RMSProp(1e-7, 0.9, 0);
    alpha.clamp(0.2, 0.2, 1);
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::reinforce1(std::vector<Step>& x, float learningRate)
{
    /*
        Standard REINFORCE with baseline — clean, no side effects version.

        ∇J = E[∇log π(a|s) · (G_t - b)]

        For softmax policy π(a|s) = exp(z_a)/Σexp(z_i):
          ∂log π(a|s)/∂z_i = δ_{ik} - π_i = (e_k - π)_i
          ∇_z J = A · (e_k - π)

        We set dLoss such that the softmax Jacobian-vector product gives -∇J:
          J · dLoss = -A · (e_k - π)

        Setting dLoss[k] = -A/π_k, dLoss[i≠k] = 0:
          (J·dLoss)_i = y_i · (dLoss_i - Σ y_j·dLoss_j)
                      = y_i · (0 if i≠k else -A/π_k  -  y_k·(-A/π_k))
                      = y_i · (-δ_{ik}·A/π_k + A)
                      = -A · (δ_{ik} - y_i) = -A · (e_k - y)_i = -∇_i J ✓

        Then RMSProp does θ -= η·g.w = θ -= η·(-A·(e_k-π)·x^T)
        = θ += η·A·(e_k-π)·x^T = gradient ascent. ✓

        Note: dLoss[k] uses NEGATIVE advantage because RMSProp optimizer
        subtracts gradients from parameters. Without negation, we'd get
        gradient descent on the policy gradient objective.
    */
    const std::size_t n = x.size();
    if (n == 0) {
        return;
    }

    /* Discounted returns G_t */
    Tensor discountedReward(n, 1);
    float r = 0;
    for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }

    /* Mean baseline, plus a rescale by the return's standard deviation.
       NOTE on why this rescale is a no-op as the code currently stands:
       reinforce1() ends with Net::RMSProp(), which calls Optimize::RMSProp with
       clipGrad = true, and that divides the WHOLE accumulated gradient by its L2
       norm (dw /= dw.norm2()). A single global factor applied to every advantage
       is therefore divided out again and the parameter update is bit-for-bit
       unchanged — so `scale` only ever matters if this method is later used with
       an optimizer invoked as clipGrad = false. What it must NOT be is a
       per-step rescale: the relative weights of the t-th step's advantage are
       what survives the normalization, and those are untouched here. The divisor
       is floored at 1.0 so a nearly-flat return sequence (advantage already ~0)
       cannot have its numerical noise amplified. */
    Tensor advantage(n, 1);
    float u = discountedReward.mean();
    float sd = std::sqrt(discountedReward.variance(u));
    float scale = (sd > 1.0f) ? sd : 1.0f;
    for (std::size_t t = 0; t < n; t++) {
        advantage[t] = (discountedReward[t] - u)/scale;
    }

    double lossSum = 0.0;   /* 界面曲线的策略梯度损失累加 (见 dpg.h 的 lastLoss) */

    for (std::size_t t = 0; t < n; t++) {
        /* argmax of the stored action selects the action that was taken. The
           action tensor is only READ here: the original reinforce() above
           rewrites it in place, which is why that version is only exact when
           the stored action is a one-hot vector (it produces dLoss[k] scaled by
           the stored probability instead of 1). */
        int k = x[t].action.argmax();

        /* --- Forward pass to get current policy --- */
        Tensor &out = policyNet.forward(x[t].state);

        /*
           记录策略梯度损失 (只给界面画曲线用, 见 dpg.h):
           surrogate = -A_t·log π(a_t|s_t), 最后取平均。
           log 用 out[k] (已经过 softmax), 与下面 dLoss 用的是同一个 probK。
        */
        lossSum += -(double)advantage[t] * std::log((double)out[k] + 1e-8);

        /* --- alpha (temperature) gradient ---
           Entropy of the FULL policy distribution, H = -Σ π_i·log(π_i).
           The previous version used entropy(out[k]) — a single action's
           contribution — which is not the policy entropy at all.

           Sign: the SAC dual objective is J(α) = α·(H(π) − H_target), so
           dJ/dα = H(π) − H_target; RMSProp then applies α −= η·g, which LOWERS
           α when the policy is too random. This method previously used
           H_target − H, i.e. the opposite sign (the older reinforce() above
           still does, and is left as it was). */
        float H = 0;
        for (std::size_t i = 0; i < actionDim; i++) {
            H += RL::entropy(out[i]);
        }
        alpha.g[k] += H - H0;

        /* --- REINFORCE policy gradient (ascent via negated dLoss) --- */
        float probK = out[k] < 1e-6f ? 1e-6f : out[k];  /* guard 1/π blow-up */
        Tensor dLoss(actionDim, 1);                     /* zero-filled */
        dLoss[k] = -advantage[t]/probK;  /* negative = gradient ascent */

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
    /* Keep the temperature inside the same range reinforce() uses. */
    alpha.clamp(0.2f, 0.2f, 1.0f);
    policyNet.RMSProp(learningRate, 0.9, 0);
    /* 本轮的策略梯度损失 (界面曲线用, 见 dpg.h) */
    lastLoss = lossSum / (double)n;
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::save(const std::string &fileName)
{
    policyNet.save(fileName);
    return;
}

void RL::DPG::load(const std::string &fileName)
{
    //policyNet.load(fileName);
    return;
}
