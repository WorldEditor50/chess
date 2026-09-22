#ifndef PPO_H
#define PPO_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"
#include "annealing.hpp"
#include "expert.hpp"          /* 专家类型 / ExpertFactory / scaleExpertInit */
#include "transformer.hpp"

namespace RL {

/*
 * ============================================================
 *  PPO 的稀疏 MoE 骨干配置 —— 换专家 / 换专家数 / 换 top-k 只改这一段
 * ============================================================
 *
 * 结构参数必须编译期确定 (模板参数), 所以放在 namespace 作用域而不是类成员。
 *
 *  TB 专家的结构:
 *    PPO_MOE_TB_HEADS = 16 -> 1440/16 = 90 维/头 (与 SACAZAgent 的
 *                              MOE_TB_HEADS=15 @1260 同一个"每头 ~90 维"的口径)
 *    PPO_MOE_TB_DFF   = 360 -> = d_model/4 (压住 TB 专家里 FFN 的开销;
 *                              注意力那 4·d_model² 才是大头, FFN 只占 1/8)
 *
 *  换专家只改 `PPOExpert` 这一行 (两种专家都在 ExpertFactory 里注册过):
 *    using PPOExpert = TransformerBlock<PPO_MOE_TB_HEADS, PPO_MOE_TB_DFF>;  // 现役
 *    using PPOExpert = MlpExpert;                                          // 旧配置 (专家数要配 8/2)
 */
constexpr int PPO_MOE_TB_HEADS = 16;
constexpr int PPO_MOE_TB_DFF   = 360;
constexpr int PPO_MOE_EXPERTS  = 4;
constexpr int PPO_MOE_TOPK     = 1;

using PPOExpert = TransformerBlock<PPO_MOE_TB_HEADS, PPO_MOE_TB_DFF>;

/*
 * ================================================================
 *  MLP 专家配置 (2026-09: 从"唯一骨干"变成"两个骨干之一")
 * ================================================================
 * 上面那套 TB 专家是现役骨干, 而 E=8 / top-2 的 **MlpExpert** 配置并没有被删掉 ——
 * 它现在由**另一个 agent** 使用 (界面上的 "PPO+MCTS (AlphaZero, MLP专家)",
 * ChessBoard::AGENT_PPOMCTS_MLP), 于是可以在同一个算法下直接对弈比较两种骨干:
 *
 *   配置                        参数量     前向       前向+反向
 *   MlpExpert        E=8 top-2   2.15 M   0.139 ms    1.94 ms
 *   TB<16,360>       E=4 top-1  38.0  M   3.59  ms   32.1  ms
 *
 * (数字来自 ppo.h 上面那张实测表; MLP 专家便宜 ~25×、容量小 ~18×。)
 * 选择方式见 `PPO::Backbone` —— 运行时参数, 默认仍是 TB 专家, 所以现役 agent、
 * 测试与 bench 的行为**逐位不变**。
 */
constexpr int PPO_MOE_MLP_EXPERTS = 8;
constexpr int PPO_MOE_MLP_TOPK    = 2;

/*
 * Simplified PPO for AlphaZero-style Chinese Chess.
 *
 * 骨干 (2026-09 第二次改版): **专家从 MlpExpert 换成 TransformerBlock**。
 *   旧: SparseMoE<MlpExpert, 8, 2>                  (便宜的 MLP 专家, 容量小)
 *   新: SparseMoE<TransformerBlock<16,360>, 4, 1>   (与 SAC+AZ 那条骨干同一族)
 * 第一次改版是"稠密 MOE<8,4> -> 稀疏路由" (下面那段注释), 那件事没有回退。
 *
 *   actorP  : state -> SparseMoE(E=4, top-1, 专家 = TB<16,360>) -> Tanh(h) -> Softmax(actionDim)
 *   critic  : state -> SparseMoE(E=4, top-1, 专家 = TB<16,360>) -> Tanh(h) -> Linear(1)
 *
 * 为什么换: MlpExpert 的容量被它的隐层宽度锁死 (2·d·h ≈ 0.18 M MAC/专家), 而
 * TransformerBlock 专家带完整的注意力 + FFN (4·d² + 2·d·d_ff ≈ 9.3 M MAC/专家) ——
 * 参数量 2.15 M -> 38.0 M (**17.7×**), 这是"专家"这个词在本工程里第一次真的代表容量。
 *
 * 代价 (实测: 单网络, d=1440/h=64/头=8100, MSVC Release + AVX2;
 *       复现脚本 .r1build/bench_ppo_expert.cpp):
 *
 *   配置                        参数量     前向       前向+反向    每层四份缓冲
 *   MlpExpert        E=8 top-2   2.15 M   0.139 ms    1.94 ms      34 MB
 *   TB<16,360>       E=4 top-1  38.0  M   3.59  ms   32.1  ms     608 MB
 *   TB<16,360>       E=8 top-2  75.3  M   6.24  ms   41.0  ms    1205 MB
 *
 * **专家数与 top-k 一起从 8/2 降到 4/1**, 理由有两条, 都是实测而不是偏好:
 *   1. 内存: withGrad=true 时每个全连接张量有 w/g/v/m **四份** (见 rl/layer.h 的
 *      iFcLayer 构造函数), 于是 E=8/top-2 的 actor+critic ≈ 2.4 GB —— 训练侧直接
 *      不可用; E=4/top-1 是 1.22 GB, 与改版前同一量级。
 *   2. 算力: top-k 直接乘在算力上, 而一个 TB 专家比一个 MlpExpert 贵 ~50×,
 *      所以"容量不按 k 付费"这条稀疏 MoE 的性质在这里比 MLP 专家重要得多。
 *   这正是 SACAZAgent 那条 TB 骨干选 E=4/top-1 (MOE_TB_EXPERTS/MOE_TB_TOPK)
 *   的同一套理由。要回到"容量优先": 改上面 PPO_MOE_EXPERTS/PPO_MOE_TOPK 两个常量。
 *
 * 第一次改版 (稠密 MOE -> 稀疏路由, 未回退): 稠密的 MOE 会把**全部**专家都算一遍
 * 再做门控加权和, 于是"专家数"直接乘在算力上 —— 那是稠密混合, 不是 MoE 的卖点。
 * 稀疏版只算门控选中的 top-k 个:
 *   参数量 ~ E×(每个专家)   算力 ~ k×(每个专家)      ← 算力与 E 无关
 * 实测 (docs/agents_design.md §11.4): 28.7 M 参数下 42.0 -> 10.4 ms/模拟 (4.1×)。
 *
 * 两个必须配套的东西 —— 缺任何一个, 稀疏 MoE 都学不起来:
 *
 *   1. scaleLayerInit(): 输入维数从 90 涨到上千之后, iFcLayer 默认的 U(-1,1)
 *      初始化会让 pre-activation 的标准差达到 sqrt(d/3) ≈ 22 (d=1440), Tanh 一上来
 *      就饱和、梯度接近 0, 网络基本不动。普通层按 1/sqrt(fan_in) 重缩一遍把标准差拉回
 *      ~0.6; 专家的权重由 SparseMoE 的构造函数自己用 scaleExpertInit 缩过了。
 *
 *   2. 负载均衡辅助损失 (moeAuxCoef): 没有它, softmax 的反向会把"没被选中"的专家的
 *      门控概率继续压低, 路由几轮之内就坍缩到少数专家、其余永远不训练。见
 *      sparse_moe.hpp 的 addAuxGradient() 与 test_sparse_moe 的有限差分验证。
 *
 * 全连接头 / 损失函数的学习部分与原来一致 (无 clip、无 KL 惩罚、无 actorQ):
 *   Policy loss: cross-entropy(actor(state) || MCTS 目标分布)
 *   Value loss:  MSE(critic(state), 回报)
 */
class PPO
{
public:
    /* 稀疏 MoE 骨干的结构。模板参数必须编译期确定, 所以做成常量而不是运行时成员
       (真正的取值在文件顶部: PPO_MOE_* —— 这里只是给上层留的稳定别名)。 */
    static constexpr int MOE_EXPERTS  = PPO_MOE_EXPERTS;
    static constexpr int MOE_TOPK     = PPO_MOE_TOPK;
    static constexpr int MOE_TB_HEADS = PPO_MOE_TB_HEADS;
    static constexpr int MOE_TB_DFF   = PPO_MOE_TB_DFF;
    /* MLP 专家那一套的别名 (E=8 / top-2) */
    static constexpr int MOE_MLP_EXPERTS = PPO_MOE_MLP_EXPERTS;
    static constexpr int MOE_MLP_TOPK    = PPO_MOE_MLP_TOPK;

    /*
     * ---- 骨干选择 (本工程的扩展, 2026-09) ----
     * 上游 snakeAI 只有一种编译期骨干 (PPOExpert), 而"MLP 专家"这条配置在本工程里
     * 已经被 SACAZAgent 用作对照骨干 (见它的 Backbone::SparseMoeMlp)。把它做成
     * **构造参数**之后, 同一个 PPO 实现 (搜索、训练、存盘、诊断全部共用) 就能带两种
     * 专家, 而不必复制一份 rl/ppo.cpp —— 复制才是真正的风险 (三份拷贝迟早漂移,
     * 见 expert.hpp 顶部的教训)。
     *
     * 两种骨干的**结构差异只有 MoE 层里的专家类型与 (E, top-k)**; 其余层
     * (Tanh(h) / Softmax / Linear 头)、损失、优化器、权重格式全部相同。
     * 权重文件带结构指纹 (张量元素总数 == paramCount()), 所以两种骨干的检查点
     * **互相拒绝载入**而不是静默串权重。
     */
    enum class Backbone {
        TbExperts = 0,   /* TransformerBlock<16,360> 专家, E=4 top-1 (现役) */
        MlpExperts       /* MlpExpert 专家, E=8 top-2 (便宜 ~25x, 容量小 ~18x) */
    };
    static const char *backboneName(Backbone b);

    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim,
                 /* 只有 MlpExpert 专家用它; TransformerBlock 专家的 FFN 宽度由
                    PPO_MOE_TB_DFF 决定, 这个参数被忽略 (签名保持不变, 免得改一圈调用方) */
                 int expertHidden = 64,
                 float moeAuxCoef = 0.1f,
                 /* false = 只推理 (不分配 g/v/m 梯度缓冲, 内存与构造时间约 1/4)。
                    多线程分身训练的 worker 用这个形态 —— 它们只做搜索不做反向。 */
                 bool withGrad = true,
                 /* 骨干 (见上面的 Backbone)。**默认值 = 现役的 TB 专家**, 所以所有
                    既有调用方 (agent / 测试 / bench / train_ppo) 行为逐位不变。 */
                 Backbone backbone = Backbone::TbExperts);
    virtual ~PPO(){}

    /* Forward - returns policy (Softmax) probabilities */
    Tensor &action(const Tensor &state);

    /*
        ================================================================
         R1 (2026-09): 只算合法列的策略前向
        ================================================================
        `action(state)` 每次都要把策略头 (64x8100) 整块 2.07 MB 读一遍, 而搜索里
        每个模拟只需要**那几十个合法着法**的概率 —— 而搜索是访存带宽受限的 (P7:
        聚合吞吐在 ~15 GB/s 到顶)。这里改成:
            forwardTrunk (跑到 Tanh(h) 为止) + 只算 idx 那几行的 logits + 在 idx 上 softmax

        语义:**与"action(state) 之后只在 idx 上取子集再归一化"逐元素相等**
        (推导见 rl/layer.h 的 Layer<Softmax>)。所以不需要重训、不动权重格式,
        下游的选点口径也不变。逐元素断言 (容差 1e-6) 在 test_ppomcts 的 R1 一节。

        probs 与 idx 一一对应 (probs[i] = P(idx[i])), 和 ≈ 1。
        返回 false 只表示"idx 为空" (没有任何可算的动作); 头不支持稀疏输出时
        内部自动回退全量前向 (只影响速度, 不影响数值)。
    */
    bool actionMasked(const Tensor &state,
                      const std::vector<int> &idx,
                      std::vector<float> &probs);

    /* Forward - returns scalar value V(s) */
    float value(const Tensor &state);

    /* Train on one state: cross-entropy(policy, target) + MSE(value, outcome)
     *   state       : STATE_DIM board encoding
     *   actionTarget: actionDim-dim target distribution (normalized)
     *   valueTarget : +1.0 (win for current player), -1.0 (loss), 0.0 (draw) */
    void trainStep(const Tensor &state,
                   const Tensor &actionTarget,
                   float valueTarget,
                   float lr);

    /* Self-play training: compute discounted returns and train PPO on trajectory
     *   trajectory   : list of (state, action_onehot, reward)。reward 必须是**走子方
     *                  视角**的即时奖励 (与 PPOMCTSAgent::computeReward 同一口径)
     *   finalOutcome : 终局结果, 从**最后一步走子方**的视角:
     *                  +1 胜 / -1 负 / 0 和 (或"回放被截断、没有终局"时给 0)。
     *                  函数内部会做视角逐手翻转 —— 细节见 ppo.cpp 里 learnSelfPlay
     *                  的推导注释。给错视角的话, 训练目标会在一半的样本上整体反号,
     *                  而且不会报任何错。 */
    void learnSelfPlay(std::vector<Step>& trajectory,
                       float finalOutcome,
                       float learningRate);

    /*
       按"视角逐手翻转"计算折现回报 (推导见 ppo.cpp 的 learnSelfPlay)。

       独立成 public 纯函数是为了能被单元测试**直接查数值**: 这个符号约定写错不会
       报任何错, 只会让一半样本的目标反号, 而这种错误只能靠"手算一个 2~3 步的
       小例子对一下"来发现。口径与 learnSelfPlay 完全一致 —— finalOutcome 按
       **最后一步走子方**的视角给。
    */
    std::vector<float> discountedReturns(const std::vector<Step> &trajectory,
                                         float finalOutcome) const;

    /* 与 discountedReturns 同一口径, 但直接吃"每步即时奖励"数组 —— 自对弈时轨迹里
       只留 (state, 稀疏策略目标, reward), 不必构造 RL::Step 里的 8100 维稠密目标。 */
    std::vector<float> discountedReturnsFromRewards(const std::vector<float> &rewards,
                                                    float finalOutcome) const;

    /* ================================================================
     *  梯度累积 (P3) + 回放池 (P4), 2026-09
     * ----------------------------------------------------------------
     *  原来 trainStep 是"一条样本 -> 前向+反向+优化器"。实测 (test_ppomcts 的
     *  compute budget 一节) 一步 trainStep 里 **优化器占 66%、前向只占 1%** ——
     *  每样本单独做一次全参数 RMSProp 基本是纯浪费。DQN/SAC 那边本来就是"累批再
     *  更新" (rl/dqn.cpp 的 learn(): 循环 experienceReplay 累积, 最后只调一次
     *  RMSProp), PPO 是唯一漏掉的那个。
     *
     *  (2026-09 附注: 那两个百分比是在 MLP 专家骨干 (3.8 M 参数) 上量的。换成
     *  TransformerBlock 专家 (75 M 参数) 之后**这个占比更高**: 空载实测
     *  trainStep = 305 ms、单独计时的优化器 ≈ 277 ms ⇒ **优化器占约 91%** ——
     *  优化器成本按参数量线性涨, 而前向+反向只是其中一小部分。
     *  同一段计时还有个坑: 它是"反复调 RMSProp 而不重新 backward"的纯访存微基准,
     *  机器一忙就被挤 (同一次改动里量到过 439 ms > 整步 335 ms, 每样本成本成负数),
     *  而且梯度被清零后 v = rho·v 会衰减到浮点非规格化数 (R1.5 记过这个效应)。
     *  test_ppomcts 现在遇到这种失真会打印"不可用"而不是负数; 要拿到可信占比,
     *  得让每次调用前梯度都非零。)
     *
     *  但**累积本身不是净赢**: 它把优化器成本按 batchSize 摊薄, 同时把优化器**步数**
     *  也除以 batchSize —— 没有回放池时, 你只是用"更少但更便宜的更新"换了原来那批
     *  更新。必须配回放池才能真赚到: 回放池让"一次更新能看到 batchSize×epochs 条经验"
     *  而样本生成成本不涨。
     *
     *  (2026-09 更正: 这一条原来写的是"同一批数据可以反复过很多遍, **每遍都产生新更新**"
     *  —— 不准确。`learnFromReplay` 的优化器调用在**所有 epoch 之后只有一次**, 所以
     *  只产生一次更新; 而且多 epoch 是**每遍重新抽样本**, 不是把同一批重复算几遍
     *  (后者在 clipGrad 下连方向都改不了, 见 rl/sac.cpp 里那段说明)。
     *  真正的收益是"每单位样本生成成本拿到更多梯度信号"。实测成本基准 (同一台机器):
     *  重新生成一条样本 ≈ 55 ms, 重放一条 ≈ 20 ms, 重放 + 累积 ≈ 7 ms ——
     *  复用比重生成便宜 3~8 倍。)
     * ================================================================ */

    /*
       回放池样本。刻意省内存:
         state          : STATE_DIM 稠密 (1440 float = 5.8 KB)
         actionIdx/...  : 策略目标只存**根的访问分布里的非零项** (~40 项),
                          而不是 8100 维稠密分布 (32 KB) —— 这就是"轨迹只存 action
                          index"的落地方式, 而且不丢 P5 那套软目标。
    */
    struct ReplaySample {
        Tensor state;
        std::vector<int>   actionIdx;   /* 访问分布的非零动作下标 */
        std::vector<float> actionProb;  /* 与 actionIdx 等长, 和 ≈ 1 */
        /*
           R2: 该局面的**完整合法着法**下标。

           注意它与 actionIdx 是两件事: actionIdx 只是"访问分布落在哪几个动作上"
           (搜索展开过的那些), 而训练侧的 softmax 分母要覆盖的是**全部合法着法** ——
           用 actionIdx 当分母等于把没被搜索访问到的合法着法从策略里抹掉, 那是另一种
           (更糟的) 学习问题。空 = 不知道 -> 退回全量 8100 维口径 (旧行为)。
        */
        std::vector<int>   legalIdx;
        /*
          信任域 (2026-09): **采集这条样本时**的策略在该样本 legalIdx 上的概率
          (`oldProb[k]` 对应 `legalIdx[k]`, 与 targetProb 同一子集空间)。
          为什么要在**入库时**存: PPO 的 ratio 只能对"旧策略"取 —— 训练时 actor
          已经被更新过若干次, 现场重算出来的 p 是**新策略**, 用它当分母等于恒等
          比值 1, 裁剪项直接失效 (那是本仓库改版前的状态: 只有交叉熵, 完全没有
          信任域, 见 rl/ppo.h 顶部"无 clip、无 KL 惩罚"那段自述)。
          为空 = 旧口径 (没有信任域, 行为与改版前逐位相同)。
        */
        std::vector<float> oldProb;
        float valueTarget;
    };

    /* 池容量 (条数)。20000 x 1440 x 4B ≈ 115 MB —— 这是稠密状态存储的代价。
       要放到 10^5 条以上就该像 SACAZAgent 那样把状态也存成稀疏 (平面编码里只有
       几十个非零格; 一局 32 个子 + 威胁平面 ≈ 200 项), 那是后续项。 */
    std::size_t replayCapacity = 20000;
    std::deque<ReplaySample> replay;

    /* 清掉 MoE 门控批统计, 让负载均衡辅助损失只反映本批的训练前向 */
    void resetMoeBatchStats();
    /* 只累积梯度 (前向 + 反向), 不碰优化器 */
    void accumulateGrad(const Tensor &state, const Tensor &actionTarget, float valueTarget);

    /*
        ================================================================
         R2 (2026-09): 训练侧也只算合法列
        ================================================================
        `accumulateGrad` 那条路每条样本都要: 把 8100 维策略头整个算一遍 (前向 + 反向),
        再对**全部** 8100 个槽位做 softmax, 交叉熵的分母因此把概率质量也分给了非法槽位
        (于是合法集上的质量 Z < 1, R1 测到的 Z 均值 0.51–0.59 就是这件事).

        这里换成: 骨干跑到 h -> 头只算 legalIdx 那几行 -> **只在合法集上 softmax** ->
        CE 对合法 logits 的解析梯度正好是 `p - t` -> 只更新合法行的权重、只从合法行
        反传梯度 (再往下走 `Net::backwardFrom`)。

        这是**换学习问题**, 不是等价优化 (见 docs/issues_review.md 的 R2 一节):
        合法集上的概率和为 1 (Z ≡ 1), 非法槽位不再被训练也不再参与归一化。

        legalIdx 必须与该局面的全部合法着法一致; target (targetIdx/targetProb) 是它
        的子集 (访问分布)。目标没覆盖到的合法着法目标值为 0 = "不该走", 这正是我们
        想要的信号。legalIdx 为空、头不支持稀疏、或没有梯度缓冲时**自动回退**到
        `accumulateGrad` 的全量口径。
    */
    void accumulateGradSparse(const Tensor &state,
                              const std::vector<int> &legalIdx,
                              const std::vector<int> &targetIdx,
                              const std::vector<float> &targetProb,
                              float valueTarget,
                              /*
                                信任域 (可省): 与 legalIdx 等长的**旧策略**概率。
                                给出时策略项从"纯交叉熵"换成 PPO 的裁剪代理目标
                                  L = -min(rho*A, clip(rho,1±eps)*A),  rho = p_new/p_old
                                再加上熵奖励项 (entropyCoef)。
                                省略时行为与改版前逐位相同 (纯交叉熵)。
                              */
                              const std::vector<float> &oldProb = std::vector<float>());

    /*
       R2 总开关 (默认开)。关掉就退回"全量 8100 维 softmax + 全量目标"的旧学习问题 ——
        留给 A/B 用, 也是"两种口径下同一批数据怎么学"的对照。
    */
    bool maskedTrainHead = true;
    /* 把累积的梯度一次性应用 (注入 MoE 辅助损失 -> RMSProp), 然后清零 */
    void applyGradients(float lr);

    void addReplay(const Tensor &state,
                   const std::vector<int> &actionIdx,
                   const std::vector<float> &actionProb,
                   float valueTarget,
                   /* R2: 该局面的完整合法着法 (可省; 省了就是旧的全量口径) */
                   const std::vector<int> &legalIdx = std::vector<int>(),
                   /*
                     信任域: 采集时策略在 legalIdx 上的概率 (与 legalIdx 等长, 可省)。
                     省略 = 这一条不参与 ratio 裁剪 (退化成纯交叉熵), 调用方应尽量给。
                   */
                   const std::vector<float> &oldProb = std::vector<float>());
    /*
       从回放池随机采样 batchSize 条、过 epochs 遍, 累积梯度后做**一次**优化器更新。
       返回 false 表示池子不够大或 epochs<=0 (什么都没做)。
       两处 loss 上报取**批平均**(与 DQN 报平均 TD 误差同一个口径)。
    */
    bool learnFromReplay(std::size_t batchSize, int epochs, float lr);
    std::size_t replaySize() const { return replay.size(); }
    void clearReplay() { replay.clear(); }
    /*
       取走整个回放池并把自己清空 (deque 的 swap 是 O(1))。

       多线程分身训练用它把样本从 worker 交给共享池: worker 用自己的 agent 跑自对弈
       (replayBatchSize=0 时 commitEpisode 只入池不学习), 每局结束后把池子整体搬走。
       刻意不在 RL::PPO 里加锁 —— 让这个类保持单线程, 由调用方决定怎么并发。
    */
    std::deque<ReplaySample> takeReplay()
    {
        std::deque<ReplaySample> out;
        out.swap(replay);
        return out;
    }

    /*
       Save / Load weights

       返回**真实结果** (两个网络都成功才 true)。原来是 void, 把内核 Net::save/load
       的 int 结果吞掉了 —— 于是"文件在但结构/CRC 不匹配而被拒载"这件事到不了调用方,
       各 agent 的 loadModel() 只好按"文件可读"返回 true, 表现为**静默从随机权重重来**
       却报告成功。签名从 void 改成 bool 对所有既有调用方源码兼容 (可以忽略返回值)。
    */
    bool save(const std::string &actorPara, const std::string &criticPara);
    bool load(const std::string &actorPara, const std::string &criticPara);

    /* ----------------------------------------------------------------
     *  稀疏 MoE 诊断 (只读, 不参与任何计算; 给测试与调参用)
     * ---------------------------------------------------------------- */
    /* 两个网络的稀疏 MoE 层数 (actor + critic, 应为 2) */
    int moeLayerCount() const;
    int moeExpertCount() const;
    int moeTopK() const;
    /* 参数量 (只读诊断) */
    long long actorParamCount() const { return actorP.paramCount(); }
    long long criticParamCount() const { return critic.paramCount(); }
    /* 把两个网络的专家使用次数**逐专家相加**输出 (长度 = 专家数) */
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();

public:
    int stateDim;
    int actionDim;
    int expertHidden;      /* MLP 专家的隐层宽度 (只有 MlpExperts 骨干用它) */
    /* 本实例的骨干 (构造时定, 之后不要改: 网络结构已经按它建好了) */
    Backbone backbone = Backbone::TbExperts;
    float gamma;
    float exploringRate;
    /* 负载均衡辅助损失系数, <=0 关闭。与 SACAZAgent 的默认值一致 (0.1) */
    float moeAuxCoef;

    /*
       ================================================================
       信任域 + 值域约束 (2026-09 新增, 修的是"名字叫 PPO 但没有 PPO 的机制")
       ================================================================
       本文件顶部原来自己写着"**无 clip、无 KL 惩罚、无 actorQ**" —— 也就是说策略项
       就是 `cross-entropy(actor, MCTS 访问分布)`, 那是**行为克隆**, 不是 PPO。
       实测 (docs/arena_sac_vs_ppo_report.md §5.4): 这个 agent 训练 150 局 / 400 局后
       对同一个 MCTS 基线的得分率是 48.3% / 56.7% / 50.0%, **区间全部跨 50%**
       (inconclusive) —— 策略确实在变尖 (熵 0.999 -> 0.906 -> 0.828), 但棋力没有
       任何可测的变化, 和棋率极高 (PPO400 对 MCTS 是 2 胜 2 负 36 和)。

       三个旋钮, 各自对应一条已知缺陷:
         clipEps    : ratio 裁剪。ρ = p_new/p_old, 目标
                      `L = -min(ρ·A, clip(ρ,1±ε)·A)`, A = t_a - p_old(a)。
                      旧策略概率在**入库时**随样本存下 (见 ReplaySample::oldProb) ——
                      训练时重算出来的只能是新策略, 拿它当分母等于没有信任域。
                      `clipEps <= 0` 关掉 (退回纯交叉熵, 与改版前逐位一致)。
         entropyCoef: 熵奖励。改版前策略**没有任何熵项**, 只能被搜索目标推着走,
                      于是很容易收敛到"求稳/求循环"(和棋率 87% 就是这么来的)。
         clampValue : critic 目标夹到 [-clampValue, +clampValue]。价值目标本身是
                      折扣回报 (|r|<=1 + 势能塑形), 越界只可能来自自举发散;
                      与 SACAZAgent 的 clampTarget 同一思路 (那边实测 |Q| 从 0.063
                      漂到 13.4, 直接把搜索的 PUCT 打坏)。
                       `clampValue <= 0` 关掉。
    */
    float clipEps = 0.2f;
    float entropyCoef = 0.01f;
    float clampValue = 2.0f;

    /*
       诊断钩子 (默认 nullptr = 零开销): 非空时 `accumulateGradSparse` 会把
       (probs, g, dlogit) 三组各 n 个数按序 push 进去 —— 供 `test_grad` 直接核对
       "解析梯度是否等于 p − t", 而不是在测试里用同一套公式重推一遍
       (那样是"自己证明自己")。只在测试里挂, 生产路径不碰。
    */
    std::vector<float> *gradProbe = nullptr;

    int learningSteps;
    /*
     * 最近一次 trainStep 的标量损失 (界面"训练损失曲线"用, **不参与任何计算**):
     *   lastLoss      : critic 的价值 MSE (主曲线用它 —— 与 DQN 的 TD 误差同类)
     *   lastActorLoss : actor 的交叉熵 (策略离"搜索给的走法"有多远)
     */
    double lastLoss = std::numeric_limits<double>::quiet_NaN();
    double lastActorLoss = std::numeric_limits<double>::quiet_NaN();

    /* 批内损失累加 (accumulateGrad 攒, applyGradients 取平均后写进上面两个) */
    double batchLossSum = 0.0;
    double batchActorLossSum = 0.0;
    std::size_t batchLossCount = 0;

    Net actorP;      /* Policy network:  state -> actionDim Softmax */
    Net critic;      /* Value network:   state -> 1-dim scalar    */
};

} // namespace RL
#endif // PPO_H
