/*
 * bc.cpp - 行为克隆的口径层 (见 rl/bc.h)
 *
 * 这一份**只有张量数学**, 不认识棋盘也不认识任何 agent:
 *   * 稠密掩码/目标的摊开 (SAC 那条小的动作空间用)
 *   * 掩码交叉熵对 logits 的解析梯度 (dz = π − t)
 *   * 掩码交叉熵的数值
 *   * 合法列上的 argmax
 *
 * 为什么这些函数不放进 bc.h 做成 inline: 它们被多个翻译单元包含
 * (rl/ppo.cpp 与 test_bc / train_bc), 放头文件里既要 inline 又会让
 * "改一处重编一片"; 而本工程的 RL 层本来就是 "头文件声明 + .cpp 定义" 的写法
 * (见 rl/ppo.cpp / rl/util.cpp), 保持一致。
 */
#include "bc.h"

#include <cmath>

bool RL::denseMaskAndTarget(int actionDim, const RL::BCSample &s,
                            Tensor &mask, Tensor &target)
{
    mask = Tensor((std::size_t)actionDim, 1);
    target = Tensor((std::size_t)actionDim, 1);
    mask.zero();
    target.zero();

    /* 掩码 = 完整合法集 (softmax 的分母) */
    for (std::size_t k = 0; k < s.legalIdx.size(); k++) {
        const int a = s.legalIdx[k];
        if (a >= 0 && a < actionDim) {
            mask[(std::size_t)a] = 1.0f;
        }
    }

    /*
       目标只收"既是目标、又在掩码里"的下标。
       收进来之后还要看有没有任何一项落进来: 一个都没落进来时这条样本不能用
       (t ≡ 0 的梯度是 π 而不是 0, 见 bc.h 第 2 条) —— 返回值就是这件事。
    */
    bool covered = false;
    for (std::size_t k = 0; k < s.targetIdx.size() && k < s.targetProb.size(); k++) {
        const int a = s.targetIdx[k];
        if (a < 0 || a >= actionDim) {
            continue;
        }
        if (mask[(std::size_t)a] <= 0.0f) {
            continue;
        }
        target[(std::size_t)a] += s.targetProb[k];
        if (s.targetProb[k] > 0.0f) {
            covered = true;
        }
    }
    return covered;
}

bool RL::targetCoveredByLegal(const std::vector<int> &legalIdx,
                              const std::vector<int> &targetIdx,
                              const std::vector<float> &targetProb)
{
    for (std::size_t k = 0; k < targetIdx.size() && k < targetProb.size(); k++) {
        if (!(targetProb[k] > 0.0f)) {
            continue;
        }
        const int a = targetIdx[k];
        for (std::size_t i = 0; i < legalIdx.size(); i++) {
            if (legalIdx[i] == a) {
                return true;
            }
        }
    }
    return false;
}

void RL::maskedCeLogitGrad(const Tensor &pi, const Tensor &mask, const Tensor &target,
                           Tensor &dz)
{
    const std::size_t n = pi.size();
    if (dz.size() != n || dz.shape.size() != 2) {
        dz = Tensor(n, 1);
    }
    for (std::size_t a = 0; a < n; a++) {
        /*
           掩码之外恒为 0。**精确的 0**(而不是"很小"): 策略在非法列上的概率恒为 0,
           任何非零梯度都只会在若干次更新之后把非法着法的 logit 抬起来 ——
           而掩码一挡, 抬起来也看不见, 直到某次掩码换了位置才暴露 (那是本工程
           反复记过的一类静默错)。
           掩码**比 π 短**时, 越界的列按非法处理 (见 bc.h 的契约): 宁可少更新几列,
           也不要把"没覆盖到"当成"合法"喂进策略。
        */
        if (a >= mask.size() || mask[a] <= 0.5f) {
            dz[a] = 0.0f;
            continue;
        }
        const float t = (a < target.size()) ? target[a] : 0.0f;
        dz[a] = pi[a] - t;
    }
}

double RL::maskedCrossEntropy(const Tensor &pi, const Tensor &target)
{
    double ce = 0.0;
    const std::size_t n = pi.size();
    for (std::size_t a = 0; a < n && a < target.size(); a++) {
        const float t = target[a];
        if (t > 0.0f) {
            ce -= (double)t * std::log((double)pi[a] + 1e-8);
        }
    }
    return ce;
}

bool RL::argmaxOnLegal(const Tensor &pi, const Tensor &mask, int &best)
{
    best = -1;
    float bestP = -1.0f;
    for (std::size_t a = 0; a < pi.size(); a++) {
        if (a < mask.size() && mask[a] <= 0.5f) {
            continue;
        }
        if (pi[a] > bestP) {
            bestP = pi[a];
            best = (int)a;
        }
    }
    return best >= 0;
}
