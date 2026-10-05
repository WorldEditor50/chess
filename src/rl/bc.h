#ifndef RL_BC_H
#define RL_BC_H

#include <vector>
#include "tensor.hpp"

namespace RL {

/*
 * ============================================================================
 *  行为克隆 (Behavior Cloning) 的**口径层** —— 纯 C++, 不认识棋盘
 * ============================================================================
 *
 * 为什么单独一层: PPO 与 SAC 两条线的网络结构、动作空间 (8100 双射 vs 128 槽),
 * 归一化口径 (稀疏合法列 vs 掩码 softmax) 都不同, 但"监督学习的**目标**"应该是
 * 同一件事 —— 一条样本 = (状态, 该局面的**完整**合法集, 老师的动作分布)。
 * 把这个结构与环境无关地定义一次, 两个 agent 各自只负责"怎么把自己的编码与
 * 动作下标填进来"(见 src/bcagent.hpp), 于是"老师的着法算错帧"这类错误只有一处
 * 可能出错, 而不是两处。
 *
 * ----------------------------------------------------------------------------
 *  ⚠ 三条必须记住的口径 (写错了不会报错, 只会静默学坏)
 * ----------------------------------------------------------------------------
 *  1. **legalIdx 必须是该局面的完整合法集**, 不是"搜索访问到的那几个"。
 *     它是 softmax 的分母: 拿子集当分母等于把没写进去的合法着法从策略里抹掉
 *     (那不是"不该走", 而是"不存在") —— 与 rl/ppo.h 的 ReplaySample::legalIdx
 *     是同一条纪律。
 *  2. **targetIdx/prob 是合法集的子集** (老师的着法本来就是合法着法)。
 *     实测过的静默失败: 目标里出现了不在 legalIdx 里的下标时, 对齐那一步会把它
 *     丢掉, 于是 t ≡ 0 —— 而**"目标落空"比"白跑"更坏**:
 *         L = −Σ t_a·log π_a ≡ **0**            (损失读数看起来完美)
 *         dL/dz_a = π_a − t_a = **π_a ≠ 0**     (不是零梯度!)
 *     也就是说它给出一个"把这个局面所有合法 logit 一起抬高"的方向: 对这一条样本
 *     本身无效 (softmax 平移不变), 但对**别的**局面实实在在地改权重。
 *     所以造样本的那一层必须**验证**老师着法在合法集里, 内核也要再查一次并**丢弃**
 *     这条样本 (bcagent.hpp 的 sampleFrom 返回 SampleFail::TeacherIllegal,
 *     RL::PPO::bcTargetMisses / BC::UpdateStat::targetMiss 负责计数)。
 *  3. **老师分布的和应当是 1**。本轮实现里老师是 Alpha-Beta 的**单步选点**
 *     (one-hot), 但结构上留了软目标的余地 (P(老师着法) 之外的合法着法目标为 0
 *     = "不该走", 这正是 BC 想要的信号)。
 *
 * 本轮**不含** value 目标: 只克隆策略头 (actor)。理由与实测背景见
 * docs/behavior_cloning_2026_10.md —— 价值头蒸馏在历史上被量过, "成功了但选点
 * 几乎没动", 而先验才是瓶颈。
 */
struct BCSample {
    /* 该 agent 自己的状态编码 (规范视角; 由 agent 的 encodeStateFor 填) */
    Tensor state;
    /* 该局面的**完整**合法着法动作下标 (softmax 的分母) */
    std::vector<int> legalIdx;
    /* 老师目标的支撑集 (通常只有一项 = AB 选的那一步) */
    std::vector<int> targetIdx;
    /* 与 targetIdx 等长, 和 ≈ 1 */
    std::vector<float> targetProb;
    /*
       可选: 局面键 (Zobrist)。**不参与任何计算** —— 只用给"这批样本里有多少个
       互不相同的局面"这类读数用 (回放池去重率那种诊断)。
    */
    unsigned long long posHash = 0;

    void clear()
    {
        state = Tensor();
        legalIdx.clear();
        targetIdx.clear();
        targetProb.clear();
        posHash = 0;
    }

    /* 结构自检: 缺任何一部分这条样本都用不了 (调用方应当当场丢弃并计数) */
    bool valid() const
    {
        return !state.empty() && !legalIdx.empty() && !targetIdx.empty()
               && targetIdx.size() == targetProb.size();
    }

    /* 目标里第一个非零项的下标 (老师是 one-hot 时就是那一个着法); -1 = 空 */
    int argmaxTarget() const
    {
        int best = -1;
        float bestP = 0.0f;
        for (std::size_t k = 0; k < targetIdx.size() && k < targetProb.size(); k++) {
            if (targetProb[k] > bestP) {
                bestP = targetProb[k];
                best = targetIdx[k];
            }
        }
        return best;
    }
};

/*
 * 把 (legalIdx, targetIdx/targetProb) 摊成 actionDim 维的**稠密掩码**与**稠密目标**。
 *
 * 为什么不做成稀疏: SAC 系的动作空间是 128 (改前表示) 或 8100 (对齐表示), 稠密
 * 掩码最坏 32 KB —— 一个批内复用同一对缓冲, 代价可以忽略; 而 PPO 那条 8100 维的
 * 路**不走这里**, 它走 forwardTrunk + sparseLogits 的稀疏路径 (见 RL::PPO::
 * bcGradSparse), 只算合法列。
 *
 * 目标里的下标若不在 legalIdx 里会被**丢掉** (掩码之外的概率恒为 0, 留着它就是
 * 在给一个恒 0 的量算交叉熵)。返回值 = "至少有一项目标落进了掩码" —— 为 false 时
 * 这条样本**不能**拿去做更新 (见上面第 2 条: t ≡ 0 的梯度是 π 而不是 0)。
 */
bool denseMaskAndTarget(int actionDim, const BCSample &s, Tensor &mask, Tensor &target);

/*
 * 目标是否**至少有一项**落在 legalIdx 里 (只数权重大于 0 的那些项)。
 *
 * 这是 BC 的"目标落空"自检, 两个学习口径 (稀疏合法列 / 全量 8100) 共用:
 * 落空的目标等于"给了一个这个局面根本走不了的着法", 而且它不是零梯度 (见文件头
 * 第 2 条) —— 所以内核必须**丢弃**这条样本, 而不是让它静默地推权重。
 */
bool targetCoveredByLegal(const std::vector<int> &legalIdx,
                          const std::vector<int> &targetIdx,
                          const std::vector<float> &targetProb);

/*
 * 掩码交叉熵对 **logits** 的解析梯度:
 *
 *      L = −Σ_a t_a·log π_a ,   π = mask ⊙ softmax(z)
 *      dL/dz_a = π_a − t_a   (合法列)       0 (非法列, 精确的 0)
 *
 * **不**走"先写 dL/dπ 再乘 softmax 雅可比"那条路: 后者要写 −t/π, π→0 时会爆
 * (rl/ppo.cpp 里记过同一个坑, 结论是直接给 logit 梯度)。SAC 系那边等价的形式是
 * g = (π − t)/π 再经 maskedSoftmaxBackward —— 两者数学上恒等, 但这个形式既不用
 * 那个除法, 也不依赖"Σπ = Σt = 1"这条前提 (软目标没归一化时仍然是对的)。
 *
 * 非法列恰好为 0 (而不是"很小"): 掩码语义要求非法动作永远不被抬起来。
 *
 * **契约**: `pi` / `mask` / `target` / `dz` 必须是同一个 actionDim 的稠密张量
 * (dz 长度不符时按 pi 的长度重建)。掩码比 π 短时, 越界的那些列**按非法处理** ——
 * 宁可让它们不参与更新, 也不要拿"掩码没覆盖到"当"合法"用 (那是把非法着法喂进
 * 策略的方式; 而本工程的调用方都走 `denseMaskAndTarget` 按 actionDim 建掩码)。
 */
void maskedCeLogitGrad(const Tensor &pi, const Tensor &mask, const Tensor &target,
                       Tensor &dz);

/* 掩码交叉熵的**数值** (批平均的分子; 只对 t_a > 0 的项求 −t·log π) */
double maskedCrossEntropy(const Tensor &pi, const Tensor &target);

/*
 * 合法列上的 argmax (评估"策略头单独能不能选中老师的着法"用)。
 * 返回 false = 掩码全 0 (没有合法动作, 无 argmax 可言)。
 */
bool argmaxOnLegal(const Tensor &pi, const Tensor &mask, int &best);

} // namespace RL

#endif // RL_BC_H
