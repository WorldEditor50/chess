#include "sacazagent.h"

#include "chessstate.h"   /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
#include "agentrollout.hpp"
#include "rl/sparse_moe.hpp"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>      /* selfCheckReport 的别名统计: 槽位 -> 该槽位上的互不相同走法 */
#include <set>

/* ============================================================
 *  构造: 三个网络
 *
 *  策略网: 1260 -> h -> h -> 128 **logits** (输出层不带激活)
 *    掩码 softmax 在 agent 内自己做 (见 maskedSoftmax), 所以这里输出 logits 而不是
 *    概率 —— 否则"掩码后归一化"的雅可比与 `Layer<Softmax>` 内部缓存的 o 对不上。
 *
 *  双 critic: 1260 -> h -> h -> 128 个 Q 值, 输出层同样是 `Layer<Linear>`。
 *    **刻意不用 Sigmoid 头**: 象棋奖励含负值 (输棋 −1, 丢子为负), 而
 *    `Layer<Sigmoid>` 值域是 (0,1), 结构上无法表示负 Q —— 这正是
 *    docs/issues_review.md 里 B18 记的问题 (`dqn.cpp` 至今仍用 Sigmoid 头)。
 *
 *  目标网: 只前向, withGrad=false。
 * ============================================================ */
namespace {

/*
   初始化缩放。`iFcLayer` 的构造函数把权重初始化成 U(-1,1); 对 1260 维输入来说
   pre-activation 的标准差约 sqrt(1260/3) ≈ 20 —— Tanh 一上来就饱和, 梯度接近 0,
   网络基本学不动。这里按 1/sqrt(fan_in) 再缩放一遍, 把 pre-activation 的标准差
   拉回 ~0.6。隐藏层是标准做法; 输出层顺带让初始 Q ≈ 0、初始策略接近均匀, 对探索有利。
*/
void scaleLayerInit(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer*>(net[i]);
        if (fc == nullptr) {
            continue;
        }
        const float fanIn = (float)(fc->inputDim > 1 ? fc->inputDim : 1);
        const float s = 1.0f / std::sqrt(fanIn);
        for (std::size_t k = 0; k < fc->w.size(); k++) {
            fc->w[k] *= s;
        }
        for (std::size_t k = 0; k < fc->b.size(); k++) {
            fc->b[k] *= s;
        }
    }
}

/* 找出网络里第一个稀疏 MoE 层 (没有就返回 nullptr) */
RL::ISparseMoE *findSparseMoe(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(net[i]);
        if (m != nullptr) {
            return m;
        }
    }
    return nullptr;
}

} // namespace

/* 本类骨干的名字: 纯 MLP —— 只有这一种 (见头文件"骨干"那一节) */
const char *SACAZAgent::backboneName() const
{
    return "MLP";
}

const char *SACAZAgent::trunkModeName(TrunkMode m)
{
    switch (m) {
    case TrunkMode::Separate: return "独立骨干x5 (Separate)";
    case TrunkMode::Shared:   return "共享骨干+三头 (Shared)";
    default:                  return "?";
    }
}

/* ----------------------------------------------------------------
 *  共享口径的权重前缀 —— 见头文件 sharedWeightPrefix 的说明。
 *  共享口径写 **4 个文件** (trunk + 三个头), 没有 _actor/_q1/_q2;
 *  独立口径写 3 个 (actor/q1/q2), 目标网在载入时由在线网 copyTo 派生。
 *  两套文件的个数、名字、语义都不同 ⇒ 绝不能共用一个前缀。
 * ---------------------------------------------------------------- */
const char *SACAZAgent::sharedWeightPrefix()
{
    /*
       [2026-09 独立类拆分] 这个前缀原来是 **TB 那一支在用** (两个骨干挤在一个类里,
       所以只能有一个值)。拆分之后**每个类必须有自己的**: 三个类的骨干不同 ⇒ 权重
       结构不同, 共用一个前缀 = 两个骨干互相覆盖 (结构指纹挡得住"明显不同",
       挡不住"同名同构、训练口径不同"的那种错)。
       本类 (纯 MLP) 用这一个; TB 那一支保留历史值 (界面 `AGENT_SACAZ_MOE` 的存量模型
       就是那套文件), MoE-MLP 那一支见 `SACAZMoEMlpAgent::sharedWeightPrefix()`。
    */
    return "weights/sacaz_mlp_shared_agent";
}

/*
 * 一个 SparseMoE 层 -> iLayer::sptr 的小工厂。
 *
 * 为什么要它: 稀疏/稠密两种 TopK 是两个不同的模板实例 (TopK 是模板参数), 而它们的
 * 共同基类 ISparseMoE 不是 shared_ptr 能直接 covariant 转换的类型 ——
 * 显式 static_pointer_cast 一次, 调用点就不必写 4 份几乎相同的 make_shared。
 */
namespace {

template<typename Expert, int E, int K>
RL::iLayer::sptr makeMoeLayer(int d, bool withGrad, int hidden)
{
    return std::static_pointer_cast<RL::iLayer>(
        std::make_shared<RL::SparseMoE<Expert, E, K> >(d, withGrad, hidden));
}

} // namespace

/*
 * ================================================================
 *  建网: 拆成"骨干层"与"输出头层"两半 (2026-09 dev-sacmoetb)
 * ================================================================
 *  为什么拆: 独立口径要的是"骨干 + 头"拼成一张完整的网 (与改动前逐字相同的构造
 *  顺序与随机数消耗); 共享口径要的是"骨干造**一次**、三个头各造一次、然后按
 *  shared_ptr 组装成多张视图"。两种需求共用同一段层构造代码 —— 否则"两种模式除了
 *  共享关系之外逐位相同"这条前提会在下一次改结构时静默失效。
 *
 *  **构造顺序是契约的一部分**: 每一层在构造时会从 RL::Random 抽初始化权重, 谁先谁后
 *  决定了后面所有层的初始值。makeTrunkLayers 必须在 makeHeadLayer 之前调用, 且
 *  SparseMoE 内部专家的构造顺序不变 —— 独立口径因此与改动前**逐位相同**。
 * ================================================================
 */
void SACAZAgent::makeTrunkLayers(RL::Net::Layers &out, bool withGrad) const
{
    const std::size_t h = (std::size_t)(hiddenDim > 0 ? hiddenDim : 64);
    out.clear();

    /*
       隐层激活 = `Layer<RL::Tanh>` (与 59e5233 **逐字相同**的代码)。
       **不要**改用 `TanhNorm<Linear>` 且 r=1 去"复现"它: 那个层的偏置是加在 tanh
       外面的 (`tanh(r·Wx)+b`), 与这里的 `tanh(Wx+b)` 不是同一个函数 —— 实测同权重同
       局面下 max|ΔQ| = 8.9e-06 (test_sacaz [14])。完整来龙去脉见 sacazagent.h 里
       `hiddenActivationName()` 上面那段注释。
    */
    /*
       [2026-09 独立类] 这里原来是按 `Backbone` 枚举分四支的 switch; 现在本类只有
       一支骨干 (纯 MLP), 所以直接往下写。**层构造顺序是契约的一部分** (每层构造时从
       RL::Random 抽初始化权重): 与拆分前 `case Backbone::Mlp` 那一支逐字相同, 所以
       拆分前后同种子下的权重**逐位相同** (回归证据: 四个骨干的指纹与拆分前逐行相同)。
    */
    out.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
    out.push_back(RL::Layer<RL::Tanh>::_(h, h, true, withGrad));
}

void SACAZAgent::makeHeadLayer(RL::Net::Layers &out, bool withGrad) const
{
    const std::size_t h = (std::size_t)(hiddenDim > 0 ? hiddenDim : 64);
    out.clear();
    /*
       输出层始终是 `Layer<Linear>` (不是 Sigmoid): 象棋奖励含负值, Q 必须能取负
       (docs/issues_review.md B18)。掩码 softmax 在 agent 里自己做, 所以策略头输出
       **logits**, 不是概率。三个头 (策略 / Q1 / Q2) 结构完全相同 —— 差别只在各自的
       权重与各自收到的梯度。
    */
    out.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, withGrad));
}

RL::Net SACAZAgent::buildNet(bool withGrad) const
{
    RL::Net::Layers all;
    makeTrunkLayers(all, withGrad);
    RL::Net::Layers head;
    makeHeadLayer(head, withGrad);
    all.insert(all.end(), head.begin(), head.end());

    RL::Net net(all);
    /*
       只缩放普通的 iFcLayer。稀疏 MoE 层本身不是 iFcLayer (dynamic_cast 会返回
       nullptr), 它的专家权重已经在 SparseMoE 的构造函数里用 scaleExpertInit 缩过了。
       (这里若重复缩放同一个层会把初始化标准差再压一次, 所以每层只走一次。)
       这一段与 59e5233 逐字相同 —— 谁能被缩、缩多少, 也是"逐位复现"的一部分
       (当年用 TanhNorm 去顶替那一层时, 它同样是 iFcLayer 的子类, 所以连这个缩放
       都看不出差别; 见上面 hiddenActivationName() 那段)。
    */
    scaleLayerInit(net);
    return net;
}

/*
 * ================================================================
 *  共享骨干口径的建网 (TrunkMode::Shared)
 * ================================================================
 *  层对象只造 **一次**, 然后用 shared_ptr 组装成多张"视图"网:
 *
 *      trunk        = [骨干层]                       (在线, 有梯度)
 *      trunkTarget  = [骨干层]                       (目标, 无梯度)
 *      actorHead / q1Head / q2Head                    (在线三个头)
 *      q1TargetHead / q2TargetHead                    (目标两个 Q 头)
 *
 *      actor   = [骨干层 + 策略头]    <- 与独立口径同名同义的**完整网**, 只是与
 *      q1      = [骨干层 + Q1头]         q2/trunk 共享同一批层对象
 *      q2      = [骨干层 + Q2头]
 *      q1Target= [目标骨干 + Q1目标头]
 *      q2Target= [目标骨干 + Q2目标头]
 *
 *  三个"为什么":
 *   1. **actor/q1/q2 仍必须是完整的 Net**: test / GUI / selfCheckReport 里到处是
 *      `agent.actor.paramCount()` / `agent.actor.copyTo(...)` / `forwardTrunk()`。
 *      共享的是**层对象**, 不是接口 —— 所以那些用法一行都不用改, 而且语义仍然正确
 *      (actor.forward(state) 就是"骨干+策略头", 与独立口径数值一致)。
 *   2. **目标骨干只有一份** (独立口径是 q1Target/q2Target 各一份)。softValueFrom 用的是
 *      min(Q1,Q2), 两个 Q 头共享同一份目标骨干完全等价, 而目标网本来只前向不训练,
 *      于是训练侧每样本的骨干前向从 6 次降到 3 次。
 *   3. **热路径不再走 actor/q1/q2 这三个视图**: learnBatch / selectMove 显式地
 *      "trunk.forward 一次 + 三个头各 forward 一次"。走视图也能得到**正确**结果,
 *      只是会把骨干跑三遍 (那就是独立口径的开销)。
 * ================================================================
 */
void SACAZAgent::buildSharedNets()
{
    const std::size_t h = (std::size_t)(hiddenDim > 0 ? hiddenDim : 64);
    (void)h;

    RL::Net::Layers trunkL, trunkTL;
    makeTrunkLayers(trunkL, true);      /* 顺序契约: 骨干先造 */
    makeTrunkLayers(trunkTL, false);

    RL::Net::Layers hA, hQ1, hQ2, hQ1T, hQ2T;
    makeHeadLayer(hA, true);
    makeHeadLayer(hQ1, true);
    makeHeadLayer(hQ2, true);
    makeHeadLayer(hQ1T, false);
    makeHeadLayer(hQ2T, false);

    /*
       初始化缩放: 与 buildNet 同一套 (scaleLayerInit 只做缩放、**不抽随机数**,
       所以"分几次调"不改变任何权重, 也不改变随机数流)。
    */
    {
        RL::Net t(trunkL);   scaleLayerInit(t);
        RL::Net t2(trunkTL); scaleLayerInit(t2);
        RL::Net a(hA);       scaleLayerInit(a);
        RL::Net b(hQ1);      scaleLayerInit(b);
        RL::Net c(hQ2);      scaleLayerInit(c);
        RL::Net d(hQ1T);     scaleLayerInit(d);
        RL::Net e(hQ2T);     scaleLayerInit(e);
    }

    trunk       = RL::Net(trunkL);
    trunkTarget = RL::Net(trunkTL);
    actorHead   = RL::Net(hA);
    q1Head      = RL::Net(hQ1);
    q2Head      = RL::Net(hQ2);
    q1TargetHead = RL::Net(hQ1T);
    q2TargetHead = RL::Net(hQ2T);

    /* 组装完整视图 (共享同一批层对象: Net 的 Layers 就是 shared_ptr 的 vector) */
    RL::Net::Layers all;
    all = trunkL;  all.insert(all.end(), hA.begin(),  hA.end());  actor = RL::Net(all);
    all = trunkL;  all.insert(all.end(), hQ1.begin(), hQ1.end()); q1    = RL::Net(all);
    all = trunkL;  all.insert(all.end(), hQ2.begin(), hQ2.end()); q2    = RL::Net(all);
    all = trunkTL; all.insert(all.end(), hQ1T.begin(), hQ1T.end()); q1Target = RL::Net(all);
    all = trunkTL; all.insert(all.end(), hQ2T.begin(), hQ2T.end()); q2Target = RL::Net(all);

    /* 目标网必须从在线网拷一份 (与独立口径同一条理由: 否则目标网从另一个随机点开始) */
    trunk.copyTo(trunkTarget);
    q1Head.copyTo(q1TargetHead);
    q2Head.copyTo(q2TargetHead);
}


/*
 * 实际生效的隐层激活 —— **读建好的网络**, 不是回显某个开关。
 *
 * 为什么要有它: 上一轮的回归就是这一层被从 `Layer<Tanh>` 换成了 `TanhNorm<Sigmoid>`
 * (随机权重下棋力 -26 个点), 而**面板上一个字都看不出来** —— 参数量、权重指纹全都
 * 一样 (同形状层)。这里把 actor 第 2 层的**真实类型**报出来, 于是"激活被换了"这件事
 * 在自检面板上直接可见。它同时是一条回归断言: 正常情况下必须报 `tanh (Layer<Tanh>)`,
 * 正常的两个骨干 (Mlp / 稀疏 MoE 的专家骨干) 都是这一层。
 */
const char *SACAZAgent::hiddenActivationName() const
{
    RL::Net &self = const_cast<RL::Net &>(actor);   /* Net::operator[] 没有 const 重载 */
    if (self.size() < 2) {
        return "? (网络层数不足, buildNet 改坏了?)";
    }
    if (dynamic_cast<RL::Layer<RL::Tanh> *>(self[1]) != nullptr) {
        return "tanh (Layer<Tanh>) [59e5233 同款]";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Sigmoid> *>(self[1]) != nullptr) {
        return "**TanhNorm<Sigmoid> —— 这就是那个掉 26 个点的回归!**";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Linear> *>(self[1]) != nullptr) {
        return "TanhNorm<Linear> —— 与 Layer<Tanh> **不等价** (偏置加在 tanh 外,"
               " 实测 max|dQ|=8.9e-06), 见 sacazagent.h";
    }
    return "其它 (既不是 Layer<Tanh> 也不是已知的 TanhNorm —— 检查 buildNet 第 2 层)";
}

/*
 * 界面上的哪一支。**[2026-09 独立类] 本类就是界面上的 `AGENT_SACAZ` (纯 MLP 骨干)**;
 * 拆分前它和 AGENT_SACAZ_MOE (TB 专家) 挤在同一个类里, 所以这一行必须按骨干分叉。
 * 59e5233 还原版与两个 MoE 支都是**独立的类** (互不继承), 各有自己的同名实现。
 */
const char *SACAZAgent::guiAgentLabel() const
{
    return "SAC+AZ (AGENT_SACAZ, 当前口径)";
}

/* 当前 SAC 的权重前缀。其余三支各有独立前缀 (见两个 MoE 独立类与自己那一支的说明):
 *   纯 MLP        : "weights/sacaz_agent"    (本类)
 *   稀疏MoE(TB)   : "weights/sacaz_moe_tb"
 *   稀疏MoE(MLP)  : "weights/sacaz_moe_mlp"
 *   59e5233 还原版: 在 SACAZLegacyAgent 里
 * 为什么不能共用: 四个类可训练的口径不同, 共用一个前缀 = 后训练的那一支静默覆盖另一支。
 */
const char *SACAZAgent::defaultWeightPrefix()
{
    return "weights/sacaz_agent";
}

/*
 * [2026-09 独立类] 从这一行往下, 原来还有四个 MoE 路由读数 (moeExpertCount / moeTopK /
 * moeUsage / resetMoeUsage) 与五个 TB 头数读数 (tbHeadsRequested/Used/Dim/Allocated /
 * tbAttentionElements) —— 它们**整段搬到了两个 MoE 独立类**里。本类是纯 MLP, 没有 MoE
 * 层也没有 TransformerBlock 专家, 这些读数在本类里恒为 0/-1。
 */

/*
 * 唯一参数量: 共享口径下骨干只算一次。
 * 为什么不能直接用 actor.paramCount(): 共享时 actor/q1/q2 是**同一批层对象的三张
 * 视图**, 各自都会把整份骨干数进去 —— 三个数相加等于把骨干算了三遍, 那正是这次要
 * 消掉的东西。独立口径下三张网本来就不共享, actor+q1 (同构) 就是全部。
 */
long long SACAZAgent::uniqueParamCount() const
{
    if (trunkMode != TrunkMode::Shared) {
        /* 独立口径: actor/q1/q2 同构, 目标网无梯度但同样占内存 -> 5 份 */
        return 5LL * actor.paramCount();
    }
    return trunk.paramCount() + trunkTarget.paramCount()
           + actorHead.paramCount() + q1Head.paramCount() + q2Head.paramCount()
           + q1TargetHead.paramCount() + q2TargetHead.paramCount();
}

SACAZAgent::SACAZAgent(Chess &chess_,
                       int hiddenDim_,
                       float gamma_,
                       float lr,
                       float cpuct,
                       TrunkMode trunkMode_)
    : chess(chess_),
      trunkMode(trunkMode_),
      hiddenDim(hiddenDim_ > 0 ? hiddenDim_ : 64),
      gamma(gamma_),
      learningRateActor(lr),
      learningRateCritic(lr),
      learningRateTrunk(lr),
      learningRateAlpha(1e-3f),
      c_puct(cpuct),
      azWeight(1.0f),
      /*
         ---- α 自动调节的口径: 目标熵 0.98 / 学习率 1e-3 (2026-09 实测改回) ----
         上一轮把目标熵降到 0.5、把 alpha 学习率提到 5e-3, 理由是"实测 α 恒为初值 0.200
         ⇒ 自动调节名存实亡"(见 docs/arena_sac_vs_ppo_report.md §8.1)。**那个理由本身
         没错, 但结论是错的**: 在"边下边学 + 对 MCTS"的受控协议下 (bench_sac_learn, 20 局
         一档, MCTS 200 次模拟, 固定 mcts-srand, 同一 seed):

             目标熵 0.5  + αlr 5e-3 (改后)  -> 2 胜 7 负 11 和   37.5%
             目标熵 0.98 + αlr 1e-3 (改回)  -> 13 胜 0 负 7 和   **82.5%**
             只改目标熵 0.98 (αlr 仍 5e-3)  -> 10 胜 2 负 8 和   70.0%
             只改 αlr 1e-3 (目标熵仍 0.5)   -> 8 胜 11 负 1 和   42.5%
             四个 α 口径改回去的配置合池 80 局 -> 38 胜 6 负 36 和 = 71.9%
         也就是说:**改后那一支是对弈里"训练越久越弱"的直接原因** —— 把 α 口径改回去之后
         训练才开始有用 (未训练的对照只有 52.5% / 70.0%, 而改后 82.5% 已经超过它)。

         机制上也能对上 (同一工具训练后的读数): 改后那一支的 critic **尺度死掉了**
         (|Q| 均值 **0.035**, 与随机初始化 0.06 同量级), 而 αlr 大 5 倍 + 目标熵低一半
         会让 α 迅速缩小、策略被"没有信息的 Q"推着走; 改回 0.98/1e-3 之后 |Q| 落在
         **2.04** (钳位边界附近), 搜索的 PUCT 重新拿到可用的排序信号。

         **不是**"改回 59e5233 的一切": clampTarget=2 + huberDelta=1 保留着, 而且实测比
         改回 0/0 更好 (67.5% vs 82.5%) —— 这两条约束压发散, α 口径决定 critic 有没有
         信号, 是两件事。完整数据与复现命令见 docs/sac_learn_reward_2026_09.md §9。 */
      entropyRatio(0.98f),
      simulations(64),
      batchSize(32),
      /*
         [F1 2026-09] 目标网同步: **默认保持 61a974d 的口径** (tau=1e-3 / 每 64 步, 逐位相同)。
         200 局 x 4 种子的配对实测确实证明"这个节拍太小 ⇒ 自举项里没有游戏信息"
         (|Q_target| 只有 0.113 = 随机尺度; 调快之后能到 1.5~2.0, 见
         docs/sac_critic_diagnosis_2026_09.md §13), 但**棋力差别不显著** (最好的一档
         65.0% -> 67.3%, p = 0.43)。"默认值也是结论" ⇒ 不把未达显著的改动设成默认;
         要开就显式传 `--target-tau=1 --target-iter=256`。
         注意: `SACAZLegacyAgent` 把这个对**显式钉住**了 (基类默认不许渗进行为还原版)。
      */
      replaceTargetIter(64),
      maxMemorySize(4096),
      totalEpisodes(0),
      learnSteps(0),
      m_leafEvals(0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;

    if (trunkMode == TrunkMode::Shared) {
        /*
           共享骨干: 层对象只造一次, actor/q1/q2 退化成"骨干 + 各自一个头"的视图。
           目标网在 buildSharedNets 里已经 copyTo 过 (见那边的注释)。
        */
        buildSharedNets();
    } else {
        actor = buildNet(true);
        q1 = buildNet(true);
        q2 = buildNet(true);
        q1Target = buildNet(false);
        q2Target = buildNet(false);

        /*
           目标网构建时 withGrad=false, 但参数仍然是随机初始化的 —— 必须从在线网拷一份
           过去, 否则目标网从"另一个随机点"开始, 自举项一开始就是纯噪声。
           (Net 的拷贝是**浅拷贝**: 共享层指针; 深拷贝必须走 copyTo。)
        */
        q1.copyTo(q1Target);
        q2.copyTo(q2Target);
    }

    /* 温度 α: 标量, 自动调节 (SAC-Discrete 的做法) */
    alpha = RL::GradValue(1, 1);
    /*
       [2026-09 实验轮] `alphaCeiling` (默认 5.0 = 改动前逐位相同)。
       为什么需要在**构造时**也夹一次: 初值 0.2 可能已经高于调用方给的上界
       (例如测试想固定 α ≤ 0.1) —— 只在 learnBatch 末尾夹的话, 第一个 batch
       用的还是 0.2, 那会让"这个上界到底生效没有"变得不可复现。
    */
    alpha[0] = std::min(0.2f, alphaCeiling > 0.0f ? alphaCeiling : 0.2f);

    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_logits = RL::Tensor(ACTION_DIM, 1);
    m_q1 = RL::Tensor(ACTION_DIM, 1);
    m_q2 = RL::Tensor(ACTION_DIM, 1);
}

std::string SACAZAgent::getName() const
{
    return "SAC+MCTS+AlphaZero (最大熵搜索)";
}

/* ============================================================
 *  状态编码: 规范视角 19 平面 (14 棋子 + 5 规则/阶段上下文)
 *  **与 PPOMCTSAgent 逐位同口径** —— 同一份 chessstate.h 的 CTX_* 顺序、
 *  同一个 canonicalCell 镜像、同一个 STATE_DIM=1710。下面有一条
 *  static_assert 把"两边维度一致"钉成编译期事实。
 * ============================================================ */
void SACAZAgent::encodeSparse(int color, std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[i];
        if (s == nullptr || s->alive == false || s->type < 0 || s->type >= 7) {
            continue;
        }
        /* 轮到黑方时左右镜像 (x -> 9-x): "己方"永远在 x 大的那一侧。
           镜像只有一份实现 (src/chessstate.h), 这里调它。 */
        const int cell = ChessState::canonicalCell(s->pos.x, s->pos.y, color);
        /* 平面下标 = type*2 + (是己方?0:1), 与棋盘状态公共约定一致 */
        const int plane = s->type * 2 + ((s->color == color) ? 0 : 1);
        cells.push_back((std::uint16_t)(plane * CELLS + cell));
    }
}

/*
   5 个规则/阶段上下文 —— **顺序与数值口径就是 chessstate.h 的 CTX_* 那一份**,
   与 PPOMCTSAgent 写进 plane 14..18 的值逐个相同 (那边调 contextValues 再铺平面,
   这边取同样的 5 个数进 Transition::ctx)。
   这一致性是"两边状态同构"的全部内容: 只要有一边换了公式, 对照就不再受控。
*/
void SACAZAgent::contextOf(Chess &c, int color, float out[CTX_COUNT])
{
    /*
       两种表示下的**语义不同**, 这点很关键:
         * 对齐表示 (5 个): 与 PPOMCTS 的 plane 14..18 逐位相同, 顺序 = chessstate.h 的
           CTX_* (子力阶段 / 总手数 / 无吃子 / 重复 / 被将);
         * 改前表示 (3 个): **只有 3 个规则上下文**, 顺序 = [无吃子, 重复, 被将]
           —— 与 59e5233 的 ctx[3] 逐位一致 (那时没有"子力阶段/总手数"两项)。
       混用这两种顺序会静默错位 (值还是有限数, 只是喂错了通道), 所以写成显式分支。
    */
    /*
       注意: 这里**不能**用 static_assert(CTX_COUNT == ChessState::CTX_COUNT) 做门槛 ——
       CTX_COUNT 是依赖表示开关的常量, 断言在编译期无条件求值, 于是在**改前表示**
       (CTX_COUNT=3) 下必然失败, 哪怕那一支根本不会执行 (if constexpr 也一样, 因为它
       的丢弃规则不覆盖"模板外但依赖常量"的 static_assert)。两种口径的一致性改由
       test_sacaz 的运行期断言钉住 (它会在对齐表示下比较 CTX_COUNT 与 chessstate.h)。
    */
    if (ALIGNED_REPR) {
        double ctx[ChessState::CTX_COUNT];
        ChessState::contextValues(c, color, ctx);
        for (int i = 0; i < CTX_COUNT; i++) {
            out[i] = (float)ctx[i];
        }
    } else {
        out[0] = (float)ChessState::halfmovePhase(c);
        out[1] = (float)ChessState::repetitionPhase(c);
        out[2] = (float)ChessState::checkPhase(c, color);
    }
}

void SACAZAgent::writeContext(RL::Tensor &state, const float ctx[CTX_COUNT])
{
    if (state.size() < (std::size_t)STATE_DIM) {
        return;
    }
    if (ALIGNED_REPR) {
        /* 对齐: 上下文是**整平面** (每平面 90 个相同值), 与 PPOMCTS 的布局逐位相同 */
        for (int p = 0; p < CTX_COUNT; p++) {
            const float v = ctx[p];
            float *dst = &state[(std::size_t)(PIECE_PLANES + p) * CELLS];
            for (int i = 0; i < CELLS; i++) {
                dst[i] = v;
            }
        }
    } else {
        /* 改前: 上下文是棋子区之后的 CTX_COUNT 个标量槽 (稀疏回放友好) */
        for (int i = 0; i < CTX_COUNT; i++) {
            state[(std::size_t)(CTX_BASE + i)] = ctx[i];
        }
    }
}

void SACAZAgent::readContext(const RL::Tensor &state, float out[CTX_COUNT])
{
    for (int i = 0; i < CTX_COUNT; i++) {
        out[i] = 0.0f;
        if (state.size() < (std::size_t)STATE_DIM) {
            continue;
        }
        if (ALIGNED_REPR) {
            out[i] = state[(std::size_t)((PIECE_PLANES + i) * CELLS)];
        } else {
            out[i] = state[(std::size_t)(CTX_BASE + i)];
        }
    }
}

void SACAZAgent::expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state)
{
    state.zero();
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

void SACAZAgent::denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells)
{
    cells.clear();
    cells.reserve(32);
    /* 只取**棋子平面**里非零的格: 上下文那 5 个槽不是"格子", 由 ctx[] 单独带。
       注意上界必须是 CTX_BASE 而不是 state.size() —— 否则 flat 布局下会把
       plane 14..18 的常数整平面 (90 个/平面) 全当成"非零格"塞进稀疏列表。 */
    const std::size_t boardLimit = (state.size() < (std::size_t)CTX_BASE)
                                       ? state.size() : (std::size_t)CTX_BASE;
    for (std::size_t i = 0; i < boardLimit; i++) {
        if (state[i] != 0.0f) {
            cells.push_back((std::uint16_t)i);
        }
    }
}

/* 稀疏格列表 -> 14 个棋子平面的 one-hot (plane*CELLS+cell 的下标直接就是扁平下标,
   所以与 flat 布局的棋子区完全对应)。 */
static void expandSparseGrids(const std::vector<std::uint16_t> &cells, RL::Tensor &state)
{
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

/* 5 个上下文标量铺成整平面 (每条 90 个相同值) —— 只在对齐表示下用。
   为什么值走标量而不是"再塞 450 个非零格": 稀疏回放只存棋子格; 稠密展开时按平面铺开
   是**确定性的函数**, 不增加任何回放内存。 */
static void fillContextPlanes(RL::Tensor &state, const float ctx[SACAZAgent::CTX_COUNT])
{
    for (int p = 0; p < SACAZAgent::CTX_COUNT; p++) {
        const float v = ctx[p];
        if (v == 0.0f) {
            continue;   /* 0 平面本来就是零, 跳过 (省 90 次写) */
        }
        float *dst = &state[(std::size_t)(SACAZAgent::PIECE_PLANES + p) * SACAZAgent::CELLS];
        for (int c = 0; c < SACAZAgent::CELLS; c++) {
            dst[c] = v;
        }
    }
}

void SACAZAgent::encodeStateFor(int color, RL::Tensor &state)
{
    if (state.size() != (std::size_t)STATE_DIM) {
        state = RL::Tensor(STATE_DIM, 1);
    }
    state.zero();
    std::vector<std::uint16_t> cells;
    encodeSparse(color, cells);
    expandSparseGrids(cells, state);
    float ctx[CTX_COUNT];
    contextOf(chess, color, ctx);
    /* 两种布局共用同一份上下文数值, 只是"铺平面"与"塞尾部槽"的区别 */
    writeContext(state, ctx);
}

void SACAZAgent::encodeState(RL::Tensor &state)
{
    /*
       视角由棋盘当前的 sideToMove 决定 —— MCTS 里走法是真正落在棋盘上的
       (`moveForward` 会翻转 sideToMove), 所以这里读到的就是该节点的走棋方。
       `rolloutFromCurrent` 进来时也会把 sideToMove 设成 color, 探索阶段同样正确。
    */
    encodeStateFor(chess.sideToMove, state);
}

/* ============================================================
 *  动作: **与 PPOMCTSAgent 同一个双射**
 *
 *      actionIdx = canonicalCell(from) * 90 + canonicalCell(to)      (8100)
 *
 *  两条要点 (与 PPOMCTS 的注释同源, 因为这就是同一个口径):
 *    * 与棋子 id 无关 —— id 会随吃子回收复用, 所以"同一格走到同一格"在不同局面下
 *      会落到不同槽位, 策略学不到稳定东西。用格子坐标没有这个问题。
 *    * 按 color 做规范镜像, 于是红黑双方的"同一步棋"共享同一个动作槽位,
 *      与状态编码的规范视角一致。
 *
 *  为什么这次必须换掉 128 槽哈希: 同局面平均挤掉 5.16 个着法 ⇒ 两个不同着法的
 *  策略目标与 Q 被强制共用一列, 那是**结构性上限** (训练多少次都消不掉), 而且
 *  它让 SAC 与 PPO 的动作空间不同构, "算法对照"就没法解释。见
 *  docs/agents_design.md §11 / §20.4 与 docs/arena_sac_vs_ppo_report.md §8.4。
 * ============================================================ */
int SACAZAgent::stepToActionIdx(const Step &s, int color) const
{
    /*
       两种表示的索引公式 (与头文件的开关一一对应):
         * 改前 (默认): `(id*37 + nextPos.x*13 + nextPos.y*7) % 128` —— 与 59e5233 逐字相同;
         * 对齐      : `canonicalCell(from)*90 + canonicalCell(to)` —— 与 PPOMCTS 同一公式。
       注意改前那一条**不用规范镜像**: 它是按"棋子 id + 目标格"哈希的, 而 id 与颜色相关
       (红黑棋子 id 不同段)。这是原实现的口径, 照抄才能复现它的棋力。
       `legacyHashAction` 只在对齐表示下生效 (用来单独隔离动作空间这个变量)。
    */
    if (!ALIGNED_REPR || legacyHashAction) {
        const unsigned long long h = (unsigned long long)s.id * 37ULL
                                   + (unsigned long long)s.nextPos.x * 13ULL
                                   + (unsigned long long)s.nextPos.y * 7ULL;
        return (int)(h % (unsigned long long)LEGACY_ACTION_DIM);
    }
    const int from = ChessState::canonicalCell(s.pos.x, s.pos.y, color);
    const int to   = ChessState::canonicalCell(s.nextPos.x, s.nextPos.y, color);
    return ChessState::actionIndexOf(from, to);
}

void SACAZAgent::getLegalActions(int color,
                                 std::vector<Step*> &steps,
                                 std::vector<int> &actionIndices,
                                 RL::Tensor &actionMask)
{
    actionMask.zero();
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());
    for (Step *s : steps) {
        const int aidx = stepToActionIdx(*s, color);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

/* ============================================================
 *  动态奖励分配 (rewardShape = 3): 局面评估 e 与两个权重倍数
 *
 *  设计与全部理由见 sacazagent.h 的 `rewardShape = 3` 一节。这里只写实现约定:
 *
 *   1. **e 是局面的函数, 不是手数的函数**: 第一版按"剩余子力"做了一个单向的阶段钟,
 *      用户指出"局势是反复变化的" —— 单调坐标表示不了反复。这一版同时读
 *      **棋子数量 / 棋子价值 / 双方子力差**: 子被吃光 e 升, 一方吃掉对方的子
 *      (差变大) e 升, 落后一方吃回来 (差变小) e **降** —— 权重跟着局势回摆。
 *   2. **按"走子前"的局面算**: 本步正要吃的那个子算回"还在", 于是 moveForward 之前
 *      与之后调 computeReward 得到同一个 e (两个调用点在工程里都真实存在)。
 *   3. **两个倍数只放大、永不归零**: m_mat ∈ [1, 1+matBoost], m_mate ∈ [1, 1+mateBoost];
 *      默认 matBoost = mateBoost = 0.5 时两者之和恒为 2.5 = "固定预算按局面动态分配"。
 *   4. **rewardShape != 3 时全部退化成 1.0**: 默认路径一位不变 (工程硬约定)。
 * ============================================================ */

/* 局面评估的三个因子 (都归一到 [0,1], 都随"越接近残局 / 越定局"增大) */
namespace {
struct PosFactors {
    double value = 0.0;   /* 1 − 双方剩余非将子力价值 / 7.0      (稀疏度: 价值) */
    double count = 0.0;   /* 1 − 双方剩余非将棋子个数  / 30       (稀疏度: 数量) */
    double lead  = 0.0;   /* |红方 − 黑方| / (红方 + 黑方)        (定局度: 相对子力差) */
};
}   /* namespace */

double SACAZAgent::mateProximity(const Step *pendingCapture) const
{
    const int pendingVictim = (pendingCapture != nullptr) ? pendingCapture->nextId
                                                          : Stone::ID_NONE;
    /* 两个"满盘"刻度来自 stone.h (单一来源, 测试与文档都引它) */
    constexpr double FULL_VALUE = REWARD_FULL_MATERIAL;         /* 7.0 = 3.5 x 2 */
    constexpr double FULL_COUNT = REWARD_FULL_PIECE_COUNT;      /* 30 = 15 x 2 */

    double red = 0.0, black = 0.0;
    int count = 0;
    for (int i = 0; i < 32; i++) {
        const Stone *st = chess.stones[i];
        if (st == nullptr || st->type == Stone::TYPE_JIANG) {
            continue;   /* 将不算材质: 它的价值就是"被吃"= 终局本身 */
        }
        /*
           已经被吃掉的子不算; 但**本步正要吃的那个算"还在"** —— 见上面约定 2。
           (调用方可能在 moveForward 之前或之后求奖励, 两个时机必须同值。)
        */
        if (!st->alive && st->id != pendingVictim) {
            continue;
        }
        if (st->color == Stone::COLOR_RED) {
            red += st->value;
        } else {
            black += st->value;
        }
        count++;
    }

    PosFactors f;
    f.value = 1.0 - (red + black) / FULL_VALUE;
    f.count = 1.0 - (double)count / FULL_COUNT;
    /*
       "定局度"用**相对**子力差 (|红−黑| / (红+黑)), 不用绝对差:
         * 绝对差在残局里会被"总量本来就小"放大 —— 只剩两个兵 vs 一个兵 也会得到
           一个"看起来很大"的差;
         * 相对差是尺度无关的: 0 = 完全均势, 1 = 一边被吃光, 与还剩多少子无关。
       分母为 0 (双方都只剩光将) 时定义成 0 (没有优势可言)。
    */
    const double total = red + black;
    f.lead = (total > 0.0)
                 ? ((red > black ? (red - black) : (black - red)) / total)
                 : 0.0;

    double e = 0.0;
    switch (mateScoreMode) {
    case 1:  e = f.value; break;                             /* 只用价值 (≈第一版的阶段钟) */
    case 2:  e = f.count; break;                             /* 只用数量 */
    case 3:  e = (f.value + f.count) * 0.5; break;            /* 数量 + 价值, 不看优势 */
    case 4:  e = f.lead; break;                              /* 只看优势 (定局度) */
    default: e = (f.value + f.count + f.lead) / 3.0; break;   /* 三因子等权 (默认) */
    }
    if (e < 0.0) { e = 0.0; }
    if (e > 1.0) { e = 1.0; }
    return e;
}

float SACAZAgent::materialWeightMul(const Step *pendingCapture) const
{
    if (rewardShape != 3 || matRewardBoost <= 0.0f) {
        return 1.0f;
    }
    const double e = mateProximity(pendingCapture);
    return (float)(1.0 + (double)matRewardBoost * (1.0 - e));
}

float SACAZAgent::mateWeightMul() const
{
    if (rewardShape != 3 || mateRewardBoost <= 0.0f) {
        return 1.0f;
    }
    /* 终局倍率取**当前**局面的 e —— 三个调用点都保证棋盘就在终局那个局面上 */
    const double e = mateProximity(nullptr);
    return (float)(1.0 + (double)mateRewardBoost * e);
}

float SACAZAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    /*
       奖励尺度 (Phase 1 起与其他 agent 统一):
       本 agent 原来就用 `Stone::value` 原值 (车 0.5 / 马炮 0.3 / 兵 0.1), 方向是对的
       ——"单个吃子 < 终局 ±1"。但**一局累积**下来一方全材质 = 3.5 > 终局 1.0, 也就
       是说"吃光对方"在数值上仍然是"赢棋"的 3.5 倍 (诊断 [2] 的不变量当场抓住这一条,
       它对 SACAZ 同样成立)。现在统一到 stone.h 的 REWARD_MATERIAL_COEF = 0.1:
       一方全材质 0.35 < 终局 1.0, 终局真正主导。

       吃將仍然不给即时奖励 (这一点 SACAZ 原来就是对的, 现在推广到全部 agent):
       它必然是终局, 终局奖励会覆盖; 若按 value_jiang = 1000 给, Q/价值的尺度会被
       彻底压倒, 软备份与 MSE 都没法看。

       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来的黑方视角写法会让红方白吃一个黑子
       拿到负奖励, 与终局 (resultValue 给的走子方视角 ±1) 相反。
       (Chess::moveForward 的 totalReward 仍是黑方视角, 那是它的记账约定; 界面上的
       奖励曲线在 ChessBoard::playMatchGame 里显式换算成走子方视角。)
       回归钉在 test_match 的 [2.6] 节。

       rewardScale 是**消融旋钮** (默认 1.0 = 逐位不变): 只缩放即时奖励, 不动终局 ±1。
       终局是环境的真值, 缩放它等于换一个游戏; 即时项是"塑形", 缩它才是在问
       "这个塑形值多少"。见 sacazagent.h 的说明。

       **默认路径必须逐位等于改动前的代码** (2026-09 排查教训): `1.0f * x` 数学上是恒等,
       但在"与参考实现逐位对比"的场合不该留任何多余运算 —— 一旦出现偏差, 排查者会先
       怀疑这一行。所以写成显式分支: 1.0 时**原式返回**, 不进乘法。
    */
    const bool capturedJiang = (victim->type == Stone::TYPE_JIANG);

    /*
       rewardShape = 1: **去掉材质塑形** (见 sacazagent.h 的说明)。
       每步代价**保留**: 它是"别磨蹭"的那一项, 去掉只会让长局更多 —— 而这次实验要问的
       恰恰是不是长局(和棋)太多。
    */
    if (rewardShape == 1) {
        return stepReward(false, false, 0.0);
    }
    /*
       ---- rewardShape = 3: **按局面动态加权的吃子奖励** (杀将那一半在 terminalReward) ----
       材质系数乘一个随局面评估 e 递减的倍数 ∈ [1, 1+matRewardBoost]:
         均势满盘 (e=0): 0.1 x (1+boost);  残局/大势已定 (e=1): 0.1 (基准, **不归零**)。
       e 同时读棋子**数量** / **价值** / 双方**子力差**, 而且是局面的函数 ——
       局势反复时 (落后方吃回来) e 会**退回**, 权重跟着回摆 (见 mateProximity)。
       吃将仍然不给即时奖励: 它必然是终局, 由 terminalReward 给 (与 shape=0 同口径)。
       每步代价**不参与**放大 —— 它是"别磨蹭"的计时项, 与局面权重无关。
    */
    if (rewardShape == 3 && !capturedJiang) {
        const float mul = materialWeightMul(&s);
        const float mat = REWARD_MATERIAL_COEF * ((float)victim->value * mul)
                          + REWARD_STEP_COST;
        const float s3 = (rewardScale == 1.0f) ? mat : (rewardScale * mat);
        if (rewardTanhGain > 0.0f) {
            return std::tanh(rewardTanhGain * s3);
        }
        return s3;
    }
    const float base = stepReward(true, capturedJiang, victim->value);
    const float scaled = (rewardScale == 1.0f) ? base : (rewardScale * base);
    /*
       [2026-09 实验轮] 提议②: `rewardTanhGain > 0` 时把**即时**奖励换成
       `tanh(gain · r)`。gain<=0 (默认) 直接返回上面的原式 —— 默认路径一位不变。

       两条先算好的预期 (见 sacazagent.h 的说明):
         * gain = 1 是**恒等**: 即时奖励只有 0.029, tanh(0.029)=0.02899;
         * 要起作用必须先放大, 而那等于 `rewardScale`, 已经实测过 5x/20x **无效果**
           (TD 目标的主项是 −γ·V(s') 而不是 r)。
       终局**不参与** tanh (terminalReward 是另一个出口), 因为压小 ±1 只会更坏。
    */
    if (rewardTanhGain > 0.0f) {
        return std::tanh(rewardTanhGain * scaled);
    }
    return scaled;
}

/*
 * 终局值 (走子方视角) —— 塑形方案的**唯一出口**。三个产生点 (搜索叶子 terminalValue /
 * 自对弈 resultValue / rollout 的 outcomeForMover) 全部走这里, 理由是"搜索估的"与
 * "训练学的"必须是同一个游戏 (见 sacazagent.h 的 rewardShape 说明)。
 *
 * 读的是 this->chess 的**当前**局面 —— 三个调用点都保证棋盘就在终局那个局面上
 * (搜索是沿着路径 moveForward 走过来的; rollout 与自对弈都刚 moveForward 完)。
 */
float SACAZAgent::terminalReward(int chessResult, int perspective) const
{
    const float base = outcomeForMover(chessResult, perspective);
    if (base == 0.0f) {
        return base;   /* 和棋/未终局恒为 0: 任何塑形都不奖励"别输" */
    }
    /*
       ---- rewardShape = 3: **按局面动态加权的杀将奖励** ----
       终局值乘一个随局面评估 e 递增的倍数 ∈ [1, 1+mateRewardBoost]:
         均势满盘将局 (e=0): ±1;  残局/大势已定将局 (e=1): ±(1+boost)。
       方向与 shape=2 **相反** (2 奖励"对方满子就被将死"= 快杀), 两者不叠加。
       倍率对**双方同一个数** ⇒ 零和对称保持 (+m / −m); 只放大不改符号 ⇒
       "赢 > 和 > 输"的排序一位不变。**注意**: 这个倍数与 γ 的折现方向相反 ——
       "磨到残局再杀"在 `Δ < 100·ln(1+boost)` 手之内折现价值更高, 这是本旋钮
       已知的代价 (见头文件), 默认 boost=0.5 (窗口 40 手) 就是为了压住它。
    */
    if (rewardShape == 3) {
        return base * mateWeightMul();
    }
    if (rewardShape != 2) {
        return base;
    }
    /*
       2 = "败方兵力越完整地被将死, 越值钱" (快杀 > 磨死)。
       满材质 = 一方全部**非将**子力 = 3.5 (车 2x0.5 + 马 2x0.3 + 炮 2x0.3 +
       相 2x0.2 + 仕 2x0.2 + 兵 5x0.1); 将不参与材质 (它的价值就是"被吃"= 终局本身)。
       倍数落在 [1, 2]: 对方一个子没少就被将死 -> 2.0; 磨到只剩光将 -> 1.0。
    */
    const int loser = (base > 0.0f)
                          ? ((perspective == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                               : Stone::COLOR_RED)
                          : perspective;
    double rem = 0.0;
    for (int i = 0; i < 32; i++) {
        const Stone *s = chess.stones[i];
        if (s == nullptr || !s->alive || s->color != loser
            || s->type == Stone::TYPE_JIANG) {
            continue;
        }
        rem += s->value;
    }
    const double full = 3.5;      /* 一方全部非将子力 */
    const double k = 1.0 + (rem / full);
    return (float)((double)base * k);
}

/* ============================================================
 *  掩码 softmax 及其反向
 * ============================================================ */
void SACAZAgent::maskedSoftmax(const RL::Tensor &logits, const RL::Tensor &mask,
                               RL::Tensor &pi)
{
    pi.zero();
    float maxLogit = -std::numeric_limits<float>::max();
    bool any = false;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            any = true;
            if (logits[a] > maxLogit) {
                maxLogit = logits[a];
            }
        }
    }
    if (!any) {
        return;   /* 无合法动作: 保持全 0, 调用方必须先处理这种情况 */
    }
    float sum = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            const float e = std::exp(logits[a] - maxLogit);
            pi[a] = e;
            sum += e;
        }
    }
    if (sum <= 0.0f) {
        pi.zero();
        return;
    }
    const float inv = 1.0f / sum;
    for (int a = 0; a < ACTION_DIM; a++) {
        pi[a] *= inv;
    }
}

void SACAZAgent::maskedSoftmaxBackward(const RL::Tensor &pi, const RL::Tensor &g,
                                       RL::Tensor &dz)
{
    /*
       dL/dz_c = π_c·( g_c − Σ_a g_a·π_a )
       非法动作 π_c = 0 -> 梯度为 0, 与掩码语义一致 (永远不会把非法动作抬起来)。
    */
    float dot = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        dot += g[a] * pi[a];
    }
    for (int a = 0; a < ACTION_DIM; a++) {
        dz[a] = (pi[a] > 0.0f) ? (pi[a] * (g[a] - dot)) : 0.0f;
    }
}

void SACAZAgent::maskToBits(const RL::Tensor &mask, std::uint64_t bits[2])
{
    bits[0] = 0;
    bits[1] = 0;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] > 0.5f) {
            bits[(i < 64) ? 0 : 1] |= (std::uint64_t(1) << (i & 63));
        }
    }
}

void SACAZAgent::bitsToMask(const std::uint64_t bits[2], RL::Tensor &mask)
{
    mask.zero();
    for (int i = 0; i < ACTION_DIM; i++) {
        const std::uint64_t bit = (bits[(i < 64) ? 0 : 1] >> (i & 63)) & 1ULL;
        mask[i] = bit ? 1.0f : 0.0f;
    }
}

/* ============================================================
 *  前向 / 软价值
 *
 *  [2026-09 dev-sacmoetb] 这三个函数都有一句 `if (trunkMode == Shared)` 的分支。
 *  共享口径下 actor / q1 / q2 这三张"视图"仍然可用且语义正确 (它们与 trunk/头共享
 *  同一批层对象), 但**走视图 = 骨干被跑三遍** —— 那正是独立口径的开销。
 *  所以热路径显式地"trunk.forward 一次 + 头各 forward 一次"。
 * ============================================================ */
void SACAZAgent::policy(const RL::Tensor &state, const RL::Tensor &mask,
                        RL::Tensor &pi)
{
    /* actor 的输出缓冲会被下一次 forward 覆盖, 先拷出来再做掩码归一化 */
    if (trunkMode == TrunkMode::Shared) {
        m_logits = actorHead.forward(trunk.forward(state));
    } else {
        m_logits = actor.forward(state);
    }
    maskedSoftmax(m_logits, mask, pi);
}

void SACAZAgent::qValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out)
{
    if (trunkMode == TrunkMode::Shared) {
        RL::Tensor &h = trunk.forward(state);
        q1Out = q1Head.forward(h);
        q2Out = q2Head.forward(h);
        squashQInPlace(q1Out);
        squashQInPlace(q2Out);
        return;
    }
    q1Out = q1.forward(state);
    q2Out = q2.forward(state);
    squashQInPlace(q1Out);
    squashQInPlace(q2Out);
}

/* ------------------------------------------------------------------
 *  稀疏头推理 (只算合法列) —— 动作空间 8100 之后这是**必需**的性能路径
 *  语义等价的依据见头文件与 ilayer.h 的 sparseLogits 说明。
 * ------------------------------------------------------------------ */
namespace {

/*
   把**已经算好的骨干输出 h** 在 legalIdx 上的列抽出来。
   affine: 故意不传激活, 因为三个头都是无激活的 Linear —— 与 iFcLayer::sparseLogits
   的注释一致: 它返回的是**激活前**的值 (策略的归一化由调用方在合法集上做)。
*/
bool sparseHeadCols(const RL::iLayer *head, const RL::Tensor &h,
                    const std::vector<int> &idx, std::vector<float> &out)
{
    if (head == nullptr || !head->supportsSparseLogits()) { return false; }
    return head->sparseLogits(h, idx, out);
}

/* 独立口径: 网络自己跑一次骨干, 再取头的合法列 */
bool sparseCols(RL::Net &net, const RL::Tensor &state, const std::vector<int> &idx,
                std::vector<float> &out)
{
    if (idx.empty() || net.size() < 2) { return false; }
    RL::iLayer *head = net[net.size() - 1];
    if (head == nullptr || !head->supportsSparseLogits()) { return false; }
    RL::Tensor &h = net.forwardTrunk(state);
    return head->sparseLogits(h, idx, out);
}

/* 合法集上的数值稳定 softmax (与 maskedSoftmax 的 Z≡1 口径等价) */
bool softmaxOnSubset(const std::vector<float> &logits, std::vector<float> &piOut)
{
    if (logits.empty()) { return false; }
    const std::size_t n = logits.size();
    float m = logits[0];
    for (std::size_t i = 1; i < n; i++) {
        if (logits[i] > m) { m = logits[i]; }
    }
    piOut.assign(n, 0.0f);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const float e = std::exp(logits[i] - m);
        piOut[i] = e;
        sum += (double)e;
    }
    if (!(sum > 1e-12) || !std::isfinite(sum)) {
        return false;
    }
    const float inv = (float)(1.0 / sum);
    for (std::size_t i = 0; i < n; i++) { piOut[i] *= inv; }
    return true;
}

} // namespace

/*
 * ================================================================
 *  sparseLeaf: 一次叶子求值的**完整**稀疏路径 (搜索热路径)
 * ================================================================
 *  共享口径下骨干**只前向一次** —— 这是 Shared 模式在搜索侧的全部收益来源:
 *    * 独立口径: policySparse (actor 骨干) + qValuesSparse (q1 骨干 + q2 骨干)
 *                = 三次全网前向;
 *    * 共享口径: trunk.forward 一次 + 三个头各算一遍合法列 (头是 64->128 的一层,
 *                代价可以忽略)。
 *  返回 false = 走不了稀疏路径, 调用方**必须**回退全量口径 (与改动前同一条契约)。
 * ================================================================
 */
bool SACAZAgent::sparseLeaf(const RL::Tensor &state, const std::vector<int> &legalIdx,
                            std::vector<float> &pi, std::vector<float> &q1Out,
                            std::vector<float> &q2Out)
{
    if (legalIdx.empty()) { return false; }
    std::vector<float> logits, l1, l2;

    if (trunkMode == TrunkMode::Shared) {
        if (actorHead.size() < 1 || q1Head.size() < 1 || q2Head.size() < 1) {
            return false;
        }
        RL::Tensor &h = trunk.forward(state);   /* 唯一的一次骨干前向 */
        if (!sparseHeadCols(actorHead[actorHead.size() - 1], h, legalIdx, logits)) { return false; }
        if (!sparseHeadCols(q1Head[q1Head.size() - 1], h, legalIdx, l1)) { return false; }
        if (!sparseHeadCols(q2Head[q2Head.size() - 1], h, legalIdx, l2)) { return false; }
    } else {
        if (!sparseCols(actor, state, legalIdx, logits)) { return false; }
        if (!sparseCols(q1, state, legalIdx, l1)) { return false; }
        if (!sparseCols(q2, state, legalIdx, l2)) { return false; }
    }
    squashQInPlace(l1);
    squashQInPlace(l2);

    if (logits.size() != legalIdx.size() || l1.size() != logits.size()
        || l2.size() != logits.size()) {
        return false;
    }
    if (!softmaxOnSubset(logits, pi)) { return false; }
    q1Out.swap(l1);
    q2Out.swap(l2);
    squashQInPlace(q1Out);
    squashQInPlace(q2Out);
    return true;
}

bool SACAZAgent::qValuesSparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                               std::vector<float> &q1Out, std::vector<float> &q2Out)
{
    if (trunkMode == TrunkMode::Shared) {
        if (q1Head.size() < 1 || q2Head.size() < 1) { return false; }
        RL::Tensor &h = trunk.forward(state);
        const bool ok = sparseHeadCols(q1Head[q1Head.size() - 1], h, legalIdx, q1Out)
                        && sparseHeadCols(q2Head[q2Head.size() - 1], h, legalIdx, q2Out);
        if (ok) { squashQInPlace(q1Out); squashQInPlace(q2Out); }
        return ok;
    }
    const bool ok = sparseCols(q1, state, legalIdx, q1Out)
                    && sparseCols(q2, state, legalIdx, q2Out);
    if (ok) { squashQInPlace(q1Out); squashQInPlace(q2Out); }
    return ok;
}

bool SACAZAgent::policySparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                              std::vector<float> &piOut)
{
    std::vector<float> logits;
    if (trunkMode == TrunkMode::Shared) {
        if (actorHead.size() < 1) { return false; }
        RL::Tensor &h = trunk.forward(state);
        if (!sparseHeadCols(actorHead[actorHead.size() - 1], h, legalIdx, logits)) { return false; }
    } else {
        if (!sparseCols(actor, state, legalIdx, logits)) { return false; }
    }
    if (logits.size() != legalIdx.size()) { return false; }
    return softmaxOnSubset(logits, piOut);
}

bool SACAZAgent::softValueSparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                                 double &valueOut)
{
    std::vector<float> pi, qa, qb;
    /*
       走 sparseLeaf 而不是 policySparse + qValuesSparse: 后两者各跑一遍骨干, 在共享
       口径下等于把唯一的那次前向白做一遍 (而且两边看到的 h 是同一个, 结果一样)。
    */
    if (!sparseLeaf(state, legalIdx, pi, qa, qb)) { return false; }
    if (qa.size() != pi.size() || qb.size() != pi.size()) { return false; }

    const float a = effectiveAlpha();
    /*
       熵项居中时与 softValueFrom 同口径: 在 H 上减去 log(合法着法数)。
       **默认路径走下面那个分支的原式** —— 不写成 `+ 0.0`, 免得在"逐位复现"的场合
       多一次浮点运算 (本工程对默认路径的纪律)。
    */
    double v = 0.0;
    if (entropyCenter) {
        const double logSlots = std::log((double)((pi.size() > 1) ? pi.size() : 2));
        for (std::size_t i = 0; i < pi.size(); i++) {
            if (!(pi[i] > 0.0f)) { continue; }
            const double qmin = (double)std::min(qa[i], qb[i]);
            v += (double)pi[i] * (qmin - (double)a * (std::log((double)pi[i]) + logSlots));
        }
    } else {
        for (std::size_t i = 0; i < pi.size(); i++) {
            if (!(pi[i] > 0.0f)) { continue; }
            const double qmin = (double)std::min(qa[i], qb[i]);
            v += (double)pi[i] * (qmin - (double)a * std::log((double)pi[i]));
        }
    }
    valueOut = (double)valueScale * v;
    return true;
}

void SACAZAgent::qTargetValues(const RL::Tensor &state, RL::Tensor &q1Out,
                               RL::Tensor &q2Out)
{
    if (trunkMode == TrunkMode::Shared) {
        /*
           目标侧两个 Q 头共享**同一份**目标骨干 —— 独立口径下它们是两份各自的目标网。
           数学上完全等价: softValueFrom 用的就是 min(Q1,Q2), 而两个头吃的是同一个 h;
           代价上少一次骨干前向 (learnBatch 每样本从 6 次骨干前向降到 3 次)。
        */
        RL::Tensor &h = trunkTarget.forward(state);
        q1Out = q1TargetHead.forward(h);
        q2Out = q2TargetHead.forward(h);
        squashQInPlace(q1Out);
        squashQInPlace(q2Out);
        return;
    }
    q1Out = q1Target.forward(state);
    q2Out = q2Target.forward(state);
    squashQInPlace(q1Out);
    squashQInPlace(q2Out);
}

float SACAZAgent::softValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                                const RL::Tensor &q1In, const RL::Tensor &q2In) const
{
    const float a = effectiveAlpha();
    /*
       [2026-09 实验轮] 熵项的**居中**: `entropyCenter` 时用 α·(H − log(合法槽位数))
       代替 α·H —— 也就是"超出这个局面编码能容纳的最大熵的那一部分"。
       下面先数一次合法槽位 (只在开关打开时算, 默认路径一次都不多走)。
    */
    float logSlots = 0.0f;
    if (entropyCenter) {
        int slots = 0;
        for (int i = 0; i < ACTION_DIM; i++) {
            if (mask[i] > 0.5f) { slots++; }
        }
        logSlots = std::log((float)((slots > 1) ? slots : 2));
    }
    float v = 0.0f;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] <= 0.5f || pi[i] <= 0.0f) {
            continue;
        }
        const float qmin = std::min(q1In[i], q2In[i]);
        /*
           `entropyInTarget == 1.0f` 且未居中时走**原式** (不改动前逐位一致; 这里刻意
           不写成 `* entropyInTarget`, 免得后人以为默认路径有额外运算) —— 见头文件对该
           开关的说明: 熵项一旦进 V(s'), 它就以 −γ·α·H(s') 的形式进了 critic 的回归目标,
           而那是一个与棋局无关的常数偏置。
        */
        float ent;
        if (entropyCenter) {
            /*
               居中: V = E[min Q] + α·(H − log n)。
               写成 `α·(log π_i + log n)` 是**对的**: 循环累加的是 Σ_i π_i·(...)，
               而 Σ_i π_i = 1 ⇒ 减去 α·log n 恰好只减一次，与"在 H 上减"等价。
            */
            ent = a * (std::log(pi[i]) + logSlots);
        } else if (entropyInTarget == 1.0f) {
            ent = a * std::log(pi[i]);
        } else {
            ent = entropyInTarget * a * std::log(pi[i]);
        }
        v += pi[i] * (qmin - ent);
    }
    return v;
}

/* ============================================================
 *  MCTS
 * ============================================================ */
double SACAZAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[childID];
    if (child.visitCount == 0) {
        /* 未访问过的子节点优先 (AlphaZero 的 PUCT 里 U 项在 N=0 时最大) */
        return std::numeric_limits<double>::max();
    }
    /*
       符号 (2026-09 修正): `totalValue` 按**当前走棋方视角**累计 (见 sacazagent.h 的
       字段注释与 backup 的逐层翻号), 而子节点的走棋方就是父节点的对手 ——
       所以父节点比较时必须取负号。漏掉它等于最大化对手的价值:
       搜索专挑对自己最差的着法, 且评估越准越糟 (症状是"loss 降、棋力不涨",
       以及不敢吃子 —— 吃子后对手少大子, 子节点 Q 对对手为负, 被算成亏着)。
       与 PPOMCTSAgent::getPUCT 的修正同一处口径, 两个 agent 必须一致。
    */
    const double q = -child.getQ();
    const double u = c_puct * child.prior
                     * std::sqrt((double)parentVisits)
                     / (1.0 + (double)child.visitCount);
    return q + u;
}

bool SACAZAgent::terminalValue(int color, double &value) const
{
    const int res = chess.getResult(color);
    if (res == Chess::RESULT_ONGOING) {
        return false;
    }
    /*
       走**同一家**的终局口径 (terminalReward): 塑形开着时, 搜索叶子估的值必须与训练
       目标同一个数, 否则 PUCT 是在为一个与实际学的不同的游戏排序 (见 rewardShape 说明)。
    */
    value = (double)terminalReward(res, color);
    return true;
}

bool SACAZAgent::resultValue(int result, int color, float &out)
{
    if (result == Chess::RESULT_ONGOING) {
        return false;
    }
    /* 同样走 terminalReward (这条老路径保留公开签名, 供测试/诊断单独调用) */
    out = terminalReward(result, color);
    return true;
}

void SACAZAgent::visitDistribution(int rootID, RL::Tensor &pi)
{
    pi.zero();
    int total = 0;
    for (int childID : nodes[rootID].childIDs) {
        total += nodes[childID].visitCount;
    }
    if (total <= 0) {
        return;
    }
    const float inv = 1.0f / (float)total;
    for (int childID : nodes[rootID].childIDs) {
        const AZNode &c = nodes[childID];
        pi[c.parentAction] += (float)c.visitCount * inv;
    }
}

/* ============================================================
 *  selectMove: AlphaZero 式 PUCT + SAC 软价值叶子
 *
 *  一次模拟 = 选择(按 PUCT 下潜) -> 展开(先验最高的未尝试动作) ->
 *             估值(终局用真实胜负, 否则用**在线**双 Q 的软价值) ->
 *             回传(negamax 翻转) -> 撤销试走
 *
 *  与 PPOMCTSAgent 的三点差别:
 *    1. 叶子估值是软价值 min_i Q_i − α·log π (最大熵), 而不是一个价值头;
 *    2. 终局节点用真实胜负, 不再自举 (PPOMCTS 在终局也用网络值);
 *    3. 展开时按**先验**挑动作, 而不是随机挑 (同样模拟次数下更有效)。
 * ============================================================ */
bool SACAZAgent::learnFromSearchStep(int color, int actionIdx, const Step &step,
                                     const RL::Tensor &piVisit)
{
    /*
       ============================================================
        "这一步真实决策" -> 一条 AlphaZero 样本 (hasSearch=true) [+ 一次更新]
       ============================================================
       为什么需要它 (用户实测 + 代码核对, 2026-09):
         界面里 SAC 的 per-move 学习**只**由 preTrainThenDecide 的 rollout 驱动, 而那批
         样本是 hasSearch=false —— 策略损失里**只**有 SAC 的软 Q 项, 搜索出来的 π_MCTS
         在对弈中被完全丢掉 (只有后台训练 trainSelfPlay 用它)。关掉 rollout 勾选框更是
         一步都不学。这个方法把"已经花掉 256 次模拟算出来的那棵搜索树"接进学习回路。

       三条必须有保证的细节:
        1. **不改动真棋局**: 即时奖励必须在落子**之前**算 (落子会把被吃子置 alive=false),
           然后 moveForward 取 s'/done, 再 moveBack 原样退回 —— 与
           rolloutFromCurrent 同一条纪律 (那边也是这么做的)。sideToMove 必须显式存取,
           因为 moveForward/moveBack 会来回翻转它。
        2. **掩码与动作下标来自同一套编码**: `actionIdx` 用树里的 parentAction,
           `getLegalActions` 给的掩码就是那套编码的掩码; 两者不一致会让更新学错列。
        3. **终局值走 terminalReward()** (与搜索叶子、自对弈同一个出口), 否则搜索估的
           与训练学的会是两个游戏。
    */
    const int savedSideToMove = chess.sideToMove;

    Transition tr;
    /* ---- s ---- */
    encodeSparse(color, tr.cells);
    contextOf(chess, color, tr.ctx);
    for (int i = 0; i < ACTION_DIM; i++) {
        tr.pi[i] = piVisit[i];
    }
    tr.action = actionIdx;
    tr.hasSearch = true;                 /* 关键: 允许 AlphaZero 监督项 */
    {
        std::vector<Step*> legal;
        std::vector<int> idx;
        RL::Tensor mask(ACTION_DIM, 1);
        getLegalActions(color, legal, idx, mask);
        tr.legalCount = (int)idx.size();
        maskToBits(mask, tr.curMask);
        Steps::instance().put(legal);
    }
    /* ---- r (落子之前算: 被吃子此刻还活着) ---- */
    tr.reward = computeReward(step, color);

    /* ---- s' / done: 试走一手再原样退回 (与搜索的试走同一套 moveForward/moveBack) ---- */
    Step applied = step;
    double dummy = 0.0;
    chess.moveForward(&applied, dummy);
    const int nextColor = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                      : Stone::COLOR_RED;
    encodeSparse(nextColor, tr.nextCells);
    contextOf(chess, nextColor, tr.nextCtx);
    {
        std::vector<Step*> legal;
        std::vector<int> idx;
        RL::Tensor mask(ACTION_DIM, 1);
        getLegalActions(nextColor, legal, idx, mask);
        maskToBits(mask, tr.nextMask);
        Steps::instance().put(legal);
    }
    const int res = chess.getResult(nextColor);
    tr.done = (res != Chess::RESULT_ONGOING);
    if (tr.done) {
        tr.reward = terminalReward(res, color);   /* 终局: 真实胜负覆盖即时奖励 */
    }
    chess.moveBack(&applied, dummy);
    chess.sideToMove = savedSideToMove;           /* 试走会翻转它, 必须还原 */

    /* ---- 入池 ---- */
    memories.push_back(tr);
    while (memories.size() > maxMemorySize) {
        memories.pop_front();
    }

    /*
       更新一次。批大小按池内实际条数夹一下 (与 exploreAndTrain 同一条理由:
       learnBatch 在"池 < batchSize"时**故意**直接返回、不拿半个批去更新, 于是开局
       前两手会白跑)。
    */
    const int onlineBatch = std::min(batchSize, (int)memories.size());
    if (onlineBatch < 1) {
        return false;
    }
    learnBatch(onlineBatch);
    return true;
}

/* ============================================================
 *  notifyGameResult / lastDecisionSample —— 人机对弈的终局通道
 * ============================================================
 *
 * 洞与修法见 aiagent.h 的 `AgentBase::notifyGameResult` 长注释。这里只说明本实现的两处
 * 细节:
 *
 *  1. **幂等**: 先找出"最后一条真实决策样本"; 它已经是 done=true 就直接返回 true 而
 *     什么都不做 —— "AI 自己把对方将死"那一手在 learnFromSearchStep 里已经写过终局,
 *     而棋盘的终局通知在三个点上都会发 (人走的一手 / AI 没有合法走法 / AI 落子之后),
 *     同一条样本被通知两次是常态, 不是异常。
 *  2. **奖励走 terminalReward()**: 与搜索叶子、rollout、自对弈同一个出口, 于是塑形
 *     (`rewardShape=2/3`) 对这条通道同样生效。它读的是**当前**棋盘 (= 终局局面), 与
 *     learnFromSearchStep 内部那一支的口径一致。
 *       ⚠ 前提: 调用时棋盘停在终局那一手**之后**。ChessBoard::notifyHumanGameEnd 的
 *       三个调用点全部满足 (都在落子之后、reset 之前)。
 */
bool SACAZAgent::notifyGameResult(int chessResult, int perspective)
{
    if (chessResult == Chess::RESULT_ONGOING) {
        return false;      /* 还有棋可走: 不是终局, 什么都不该发生 */
    }
    for (auto it = memories.rbegin(); it != memories.rend(); ++it) {
        if (!it->hasSearch) {
            continue;      /* rollout / 自对弈样本: 不是"这一局的真实决策", 跳过 */
        }
        if (it->done) {
            return true;   /* 已经带终局 (自己将死对方那一手): 幂等, 不重复写 */
        }
        it->done = true;
        it->reward = terminalReward(chessResult, perspective);
        trainDiag.externalTerminals++;
        /*
           立刻学一次 (与 learnFromSearchStep 同一个做法)。池里条数不足一个批时
           learnBatch 故意直接返回 —— 那条样本先躺在池里等后面的批次抽到它, 这是既有
           行为, 不在这里另造一条旁路。
        */
        const int onlineBatch = std::min(batchSize, (int)memories.size());
        if (onlineBatch >= 1) {
            learnBatch(onlineBatch);
        }
        return true;
    }
    return false;          /* 池里没有真实决策样本 (learnFromSearch 关着 / 一步没走) */
}

bool SACAZAgent::lastDecisionSample(float &reward, bool &done) const
{
    for (auto it = memories.rbegin(); it != memories.rend(); ++it) {
        if (it->hasSearch) {
            reward = it->reward;
            done = it->done;
            return true;
        }
    }
    return false;
}

Step SACAZAgent::selectMove(int color, int simulations_, float temp, RL::Tensor *piOut)
{
    nodes.clear();
    if (simulations_ < 1) {
        simulations_ = 1;
    }
    nodes.reserve((std::size_t)simulations_ + 64);

    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor qa(ACTION_DIM, 1);
    RL::Tensor qb(ACTION_DIM, 1);

    /* ---- 根节点 ---- */
    std::vector<Step*> rootSteps;
    std::vector<int> rootIdx;
    getLegalActions(color, rootSteps, rootIdx, mask);

    AZNode root;
    root.currentColor = color;
    root.legalCount = (int)rootIdx.size();
    root.parentID = -1;
    root.parentAction = -1;

    if (root.legalCount > 0) {
        encodeStateFor(color, m_stateBuf);
        policy(m_stateBuf, mask, pi);
    }
    for (std::size_t i = 0; i < rootIdx.size(); i++) {
        root.untriedActionIndices.push_back(rootIdx[i]);
        root.untriedSteps.push_back(*rootSteps[i]);
        root.untriedPriors.push_back((double)pi[rootIdx[i]]);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(root);
    const int rootID = 0;

    if (root.legalCount == 0) {
        /* 无合法走法: 返回无效 Step, 由调用方按"真无棋可走"处理 (见 issues_review C2) */
        if (piOut != nullptr) {
            piOut->zero();
        }
        return Step();
    }

    /* ---- 主循环 ---- */
    for (int sim = 0; sim < simulations_; sim++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* --- 1. SELECT --- */
        std::vector<Step> toApply;
        while (nodes[nodeID].untriedActionIndices.empty()
               && !nodes[nodeID].childIDs.empty()) {
            int bestChild = -1;
            double bestPuct = -std::numeric_limits<double>::max();
            for (int childID : nodes[nodeID].childIDs) {
                const double p = getPUCT(childID, nodes[nodeID].visitCount);
                if (p > bestPuct) {
                    bestPuct = p;
                    bestChild = childID;
                }
            }
            if (bestChild < 0) {
                break;
            }
            toApply.push_back(nodes[bestChild].step);
            nodeID = bestChild;
            path.push_back(nodeID);
        }
        double dummy = 0.0;
        for (const Step &s : toApply) {
            chess.moveForward(&s, dummy);
        }

        double leafValue = 0.0;
        bool haveLeafValue = false;

        /* 到达的节点本身可能已经终局 (走到这里时棋盘就是该节点的局面) */
        if (terminalValue(nodes[nodeID].currentColor, leafValue)) {
            haveLeafValue = true;
        }

        /* --- 2. EXPANSION --- */
        if (!haveLeafValue && !nodes[nodeID].untriedActionIndices.empty()) {
            std::size_t pickIdx = 0;
            for (std::size_t i = 1; i < nodes[nodeID].untriedPriors.size(); i++) {
                if (nodes[nodeID].untriedPriors[i] > nodes[nodeID].untriedPriors[pickIdx]) {
                    pickIdx = i;
                }
            }
            const int chosenAction = nodes[nodeID].untriedActionIndices[pickIdx];
            const Step chosenStep = nodes[nodeID].untriedSteps[pickIdx];
            const double chosenPrior = nodes[nodeID].untriedPriors[pickIdx];
            nodes[nodeID].untriedActionIndices.erase(
                nodes[nodeID].untriedActionIndices.begin() + (long)pickIdx);
            nodes[nodeID].untriedSteps.erase(
                nodes[nodeID].untriedSteps.begin() + (long)pickIdx);
            nodes[nodeID].untriedPriors.erase(
                nodes[nodeID].untriedPriors.begin() + (long)pickIdx);

            chess.moveForward(&chosenStep, dummy);

            const int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                      ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            std::vector<Step*> childSteps;
            std::vector<int> childIdx;
            getLegalActions(nextColor, childSteps, childIdx, mask);

            AZNode child(nodeID, chosenAction, chosenStep, chosenPrior, nextColor,
                         (int)childIdx.size());

            double v = 0.0;
            if (terminalValue(nextColor, v)) {
                /* 被将杀 / 困毙 / 和: 终局节点, 价值取自真实胜负 */
                child.isTerminal = true;
                leafValue = v;
            } else {
                encodeStateFor(nextColor, m_stateBuf);
                /*
                   稀疏头路径 (只算合法列): 动作空间 8100 之后这是必须的 —— 否则每次叶子
                   估值要算 8100 个 Q 值, 而这一步只有 ~44 个合法着法 (实测 216 ms/步 ->
                   见 sacazagent.h 的说明)。取不到 (头不支持/下标越界) 就回退全量口径。

                   [2026-09 dev-sacmoetb] 用 sparseLeaf 而不是 policySparse +
                   qValuesSparse: 后者在**共享骨干**口径下会把唯一的那次骨干前向白做
                   两遍 (三次前向里两次的结果完全一样), 那正好把 Shared 的 3 倍收益
                   全吃掉。语义与"两次分开调"逐元素相同 (同一个 h, 同一套合法集归一)。
                */
                std::vector<float> piSp, qaSp, qbSp;
                bool sparseOk = sparseLeafEval
                                && sparseLeaf(m_stateBuf, childIdx, piSp, qaSp, qbSp);
                if (sparseOk) {
                    for (std::size_t i = 0; i < childIdx.size(); i++) {
                        child.untriedActionIndices.push_back(childIdx[i]);
                        child.untriedSteps.push_back(*childSteps[i]);
                        child.untriedPriors.push_back((double)piSp[i]);
                    }
                    const float a = alpha[0];
                    double v = 0.0;
                    for (std::size_t i = 0; i < piSp.size(); i++) {
                        if (!(piSp[i] > 0.0f)) { continue; }
                        const double qmin = (double)std::min(qaSp[i], qbSp[i]);
                        v += (double)piSp[i] * (qmin - (double)a * std::log((double)piSp[i]));
                    }
                    leafValue = (double)valueScale * v;
                } else {
                    /* 回退: 全量口径 (语义与上面逐元素相同, 只是慢) */
                    pi.zero();
                    policy(m_stateBuf, mask, pi);
                    qValues(m_stateBuf, qa, qb);
                    for (std::size_t i = 0; i < childIdx.size(); i++) {
                        child.untriedActionIndices.push_back(childIdx[i]);
                        child.untriedSteps.push_back(*childSteps[i]);
                        child.untriedPriors.push_back((double)pi[childIdx[i]]);
                    }
                    leafValue = (double)searchValueFrom(pi, mask, qa, qb);
                }
                m_leafEvals++;
            }
            Steps::instance().put(childSteps);

            nodes.push_back(child);
            const int newID = (int)nodes.size() - 1;
            nodes[nodeID].childIDs.push_back(newID);
            nodeID = newID;
            path.push_back(newID);
            haveLeafValue = true;
        }

        /* --- 3. 已全展开的节点: 就地用软价值 (终局已在上面处理) --- */
        if (!haveLeafValue) {
            std::vector<Step*> ls;
            std::vector<int> li;
            getLegalActions(nodes[nodeID].currentColor, ls, li, mask);
            Steps::instance().put(ls);
            encodeStateFor(nodes[nodeID].currentColor, m_stateBuf);
            /* 同样先走稀疏头 (只算合法列), 失败再回退全量 */
            if (!sparseLeafEval || !softValueSparse(m_stateBuf, li, leafValue)) {
                pi.zero();
                policy(m_stateBuf, mask, pi);
                qValues(m_stateBuf, qa, qb);
                leafValue = (double)searchValueFrom(pi, mask, qa, qb);
            }
            m_leafEvals++;
        }

        /* --- 4. BACKUP (negamax) --- */
        double v = leafValue;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[path[i]].visitCount++;
            nodes[path[i]].totalValue += v;
            v = -v;
        }

        /* --- 5. 撤销本轮试走 (path[0] 是根, 没有对应的 step) --- */
        for (std::size_t i = path.size(); i > 1; i--) {
            const Step &s = nodes[path[i - 1]].step;
            chess.moveBack(&s, dummy);
        }
    }

    if (piOut != nullptr) {
        visitDistribution(rootID, *piOut);
    }

    /* ---- 选择走法 ---- */
    int bestChildID = -1;
    if (temp <= 1e-6f) {
        int maxVisits = -1;
        for (int childID : nodes[rootID].childIDs) {
            if (nodes[childID].visitCount > maxVisits) {
                maxVisits = nodes[childID].visitCount;
                bestChildID = childID;
            }
        }
    } else {
        /* 按 N^(1/T) 采样 (自对弈用, 保证开局多样性) */
        RL::Tensor dist(ACTION_DIM, 1);
        dist.zero();
        double sum = 0.0;
        const double invT = 1.0 / (double)temp;
        for (int childID : nodes[rootID].childIDs) {
            const double w = std::pow((double)nodes[childID].visitCount, invT);
            dist[nodes[childID].parentAction] += (float)w;
            sum += w;
        }
        if (sum > 0.0) {
            const int a = RL::Random::categorical(dist);
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].parentAction == a) {
                    bestChildID = childID;
                    break;
                }
            }
        }
        if (bestChildID < 0) {
            int maxVisits = -1;
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].visitCount > maxVisits) {
                    maxVisits = nodes[childID].visitCount;
                    bestChildID = childID;
                }
            }
        }
    }

    if (bestChildID >= 0) {
        /*
           ---- [2026-09 新] 从**自己的搜索**学一次 (见 sacazagent.h 的 learnFromSearch) ----
           位置刻意放在这里: 棋盘已经恢复成根局面 (上面第 5 步把试走的都回退了),
           π 也已经在 nodes 里算好 —— 于是这一步不需要重新搜索, 代价只有"试走一手再退回"。
           `piOut` 为空时也要自己算一份访问分布 (调用方不一定要 π)。
        */
        if (learnFromSearch) {
            RL::Tensor piVisit(ACTION_DIM, 1);
            visitDistribution(rootID, piVisit);
            learnFromSearchStep(color, nodes[bestChildID].parentAction,
                                nodes[bestChildID].step, piVisit);
        }
        return nodes[bestChildID].step;
    }
    /*
       根有合法走法却没选出一个孩子 (例如一次都没展开): 兜底取根节点未展开的第一手,
       不要返回 Step() —— 那会让调用方把"还有棋可下"读成"无棋可走"并判负
       (同 ABAgent 的"全负不返回走法", 见 abagent.cpp 的长注释)。
    */
    if (!nodes[rootID].untriedSteps.empty()) {
        if (piOut != nullptr) {
            /* π 目标交给**这一步**: 单点分布, 至少与真正走出的那一手一致 */
            piOut->zero();
            (*piOut)[nodes[rootID].untriedActionIndices.front()] = 1.0f;
        }
        return nodes[rootID].untriedSteps.front();
    }
    if (piOut != nullptr) {
        piOut->zero();
    }
    return Step();
}

/*
 * [2026-09 独立类] 这里原来是 `resetMoeBatchStats()` —— 纯 MLP 骨干没有 MoE 层,
 * 没有"门控批统计"这回事, 所以整段删掉 (它只服务于 MoE 的负载均衡辅助损失,
 * 而那套东西现在与 `auxLossCoef` 一起留在两个 MoE 独立类里)。
 */

/* ============================================================
 *  learnBatch: 一次 mini-batch 的 SAC 更新 (critic / actor / α)
 * ============================================================ */
float SACAZAgent::learnBatch(int batchSize_, int epochs)
{
    if (batchSize_ < 1 || (int)memories.size() < batchSize_) {
        return 0.0f;
    }
    if (epochs <= 0) {
        epochs = replayEpochs > 0 ? replayEpochs : 1;
    }
    if (epochs < 1) {
        epochs = 1;
    }

    RL::Tensor state(STATE_DIM, 1);
    RL::Tensor nextState(STATE_DIM, 1);
    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor nextMask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor piNext(ACTION_DIM, 1);
    RL::Tensor q1n(ACTION_DIM, 1);
    RL::Tensor q2n(ACTION_DIM, 1);
    RL::Tensor q1o(ACTION_DIM, 1);
    RL::Tensor q2o(ACTION_DIM, 1);
    RL::Tensor g(ACTION_DIM, 1);
    RL::Tensor dz(ACTION_DIM, 1);

    /*
       [P4] **每个 epoch 重新从池里抽** batchSize 条 (与 RL::PPO::learnFromReplay 同一
       做法), 累积梯度后优化器只在最后调一次 (P3)。

       为什么不是"把同一批复用 epochs 遍": 批内权重不变, 所以第 N 遍的梯度与第 1 遍
       **逐位相同**; 而 `RL::Net::RMSProp` 默认 clipGrad=true (`dw /= |dw|`) 会把这个
       纯倍数完全归一掉 —— 重复同一批对更新方向**毫无影响**, 只是白烧算力。
       每遍抽新样本才是真东西: 一次更新看到 batchSize×epochs 条经验 (等于把批放大
       epochs 倍, 但仍然只调一次优化器)。生成一条样本要走整棵 MCTS, 比抽一条贵得多,
       所以这个放大基本是白拿的。
    */
    std::uniform_int_distribution<int> pick(0, (int)memories.size() - 1);

    /*
       [2026-09 独立类] 这里原来还有一次 `resetMoeBatchStats()` (MoE 门控批统计的边界)。
       纯 MLP 骨干没有 MoE 层, 因此没有"批统计"要复位。
    */

    /* 诊断读数按**本批**重置 (它们是"最近一次 learnBatch 的极值", 见头文件说明) */
    m_maxAbsTarget = 0.0;
    m_maxAbsTdErr = 0.0;

    float lossSum = 0.0f;
    int n = 0;
    float alphaGrad = 0.0f;

    for (int ep = 0; ep < epochs; ep++) {
    for (int it = 0; it < batchSize_; it++) {
        const Transition &tr = memories[(std::size_t)pick(RL::Random::engine)];

        /*
           ---- [2026-09 实验轮] 提议③: α 的 Gumbel 扰动 ----
           `alphaGumbelSigma > 0` 时, **每个样本**把 α 换成一个随机量
               α_eff = α · exp( σ·(G − γ) ),  G = −log(−log U),  γ = 0.5772
           并让**目标侧与策略梯度侧用同一个 α_eff** —— 两边分开抽等于
           "搜索估的"与"训练学的"不是一个数 (与本文件反复强调的同一条纪律)。
           注意它**不进** α 的学习: `alphaGrad` 仍然按真实 α 的梯度累加,
           否则我们会把一个被噪声扰动的量当成策略熵的函数去求导。
           默认 σ=0 ⇒ 这一整段被跳过, 与改动前逐位相同。
        */
        alphaSample = -1.0f;
        if (alphaGumbelSigma > 0.0f) {
            const double u = std::max(1e-12, (double)std::uniform_real_distribution<float>(0.0f, 1.0f)(RL::Random::engine));
            const double g = -std::log(-std::log(u));
            alphaSample = (float)((double)alpha[0] * std::exp((double)alphaGumbelSigma * (g - 0.5772156649)));
        }
        const float a = effectiveAlpha();

        expandSparse(tr.cells, state);
        writeContext(state, tr.ctx);
        expandSparse(tr.nextCells, nextState);
        writeContext(nextState, tr.nextCtx);
        bitsToMask(tr.curMask, mask);
        bitsToMask(tr.nextMask, nextMask);

        /* ---- 目标侧: 用目标网算下一局面的软价值 ----
           [2026-09 dev-sacmoetb] 共享口径下这两件事各跑**一次**骨干 (在线一次、目标
           一次), 而独立口径是三次 (actor / q1Target / q2Target 各一次)。
           π(s') 仍然取**在线**骨干+策略头, V(s') 仍然取**目标**骨干+两个 Q 头 ——
           口径一个字没改, 只是"同一个 h 不再算两遍"。 */
        if (trunkMode == TrunkMode::Shared) {
            RL::Tensor &hNext = trunk.forward(nextState);
            m_logits = actorHead.forward(hNext);
            maskedSoftmax(m_logits, nextMask, piNext);
            RL::Tensor &hNextT = trunkTarget.forward(nextState);
            q1n = q1TargetHead.forward(hNextT);
            q2n = q2TargetHead.forward(hNextT);
            squashQInPlace(q1n);
            squashQInPlace(q2n);
        } else {
            policy(nextState, nextMask, piNext);
            qTargetValues(nextState, q1n, q2n);
        }
        const float vNext = softValueFrom(piNext, nextMask, q1n, q2n);

        /*
           符号: 所有价值都是"该局面走棋方视角"。s' 轮到**对手**走, 所以自举项要取负
           (negamax):   y = r − γ(1−done)·V(s')
        */
        const float y = tr.reward - gamma * (tr.done ? 0.0f : 1.0f) * vNext;
        /*
           ---- 值域约束 (2026-09) ----
           真实 Q 必然落在 [-1.5, 1.5] 量级内: 即时奖励上界 0.35 (=REWARD_MATERIAL_COEF
           x 一方满子), 终局 ±1。但软备份把 V 反复回代, 而优化器没有任何把 V 拉回该区间
           的机制 => 实测发散 |Q| 0.063 -> 4.15 (40 局) -> 13.4 (150 局), 并把搜索的
           PUCT 打坏 (对 MCTS 得分率 73.3% -> 32.5%, 见
           docs/arena_sac_vs_ppo_report.md §5.2)。这里把目标夹住 = 把"这个游戏的 Q 值域"
           写进学习目标; 只夹目标不夹奖励, 所以不改变各着法的排序, 只挡住发散。
        */
        const float yClamped = (clampTarget > 0.0f)
                                   ? std::min(std::max(y, -clampTarget), clampTarget)
                                   : y;
        m_maxAbsTarget = std::max(m_maxAbsTarget, std::fabs((double)yClamped));

        /* ---- 当前局面 ----
           [2026-09 dev-sacmoetb] 共享口径: **骨干只前向一次**, 三个头各算一遍。
           这一次前向同时是后面 trunk.backward(state, ...) 要用的那次 —— 顺序上它必须是
           反向之前**最后一次**在线骨干前向 (独立口径下 actor/q1/q2 各跑一次, 但三次的
           输入都是同一个 state, 所以哪一次的缓存都等价)。 */
        if (trunkMode == TrunkMode::Shared) {
            RL::Tensor &hCur = trunk.forward(state);
            m_logits = actorHead.forward(hCur);
            maskedSoftmax(m_logits, mask, pi);
            q1o = q1Head.forward(hCur);
            q2o = q2Head.forward(hCur);
            squashQInPlace(q1o);
            squashQInPlace(q2o);
        } else {
            policy(state, mask, pi);
            qValues(state, q1o, q2o);
        }

        /* critic 损失: 只对实际走的那一步回归 (SAC 的标准做法, 其余动作误差为 0)。
           Huber: |err| <= delta 时与 MSE 完全一致, 超出后转线性 —— 单个离群样本不会
           再把 32 条样本的批平均方向带跑。
           为什么**不**用 RL::Loss::MSE::df: 那个 helper 只支持平方损失, 而这里要报的
           是 Huber 的数值+MSE 的梯度 (RL::Loss::MSE::df 给的就是 MSE 梯度, 与 Huber
           在 |err|<=delta 时逐位相同)。所以梯度仍走它, 数值改由上面算 —— 这条注释就是
           为了防止后人"顺手统一口径"时把 Huber 改回纯 MSE。 */
        const float err = q1o[tr.action] - yClamped;
        const double ae = std::fabs((double)err);
        m_maxAbsTdErr = std::max(m_maxAbsTdErr, ae);
        lossSum += (float)((huberDelta > 0.0f && ae > (double)huberDelta)
                               ? (double)huberDelta * (ae - 0.5 * (double)huberDelta)
                               : 0.5 * ae * ae);

        for (int ci = 0; ci < 2; ci++) {
            /*
               共享口径: 两个 Q 头各自的梯度, 反向时把梯度累加到**同一个骨干**上
               (见下面 gh 那一段)。独立口径: 两张完整的网各自反向。

               [2026-09 实验轮] 提议① 的反向: `criticTanh` 打开时 Q = tanh(z),
               所以 `dL/dz = dL/dQ · (1 − Q²)` —— 在 backward **之前**乘上去
               (`squashQBackward` 在开关关着时是空操作, 默认路径逐位相同)。
            */
            if (trunkMode == TrunkMode::Shared) {
                RL::Tensor target = (ci == 0) ? q1o : q2o;
                target[tr.action] = yClamped;
                RL::Net &head = (ci == 0) ? q1Head : q2Head;
                RL::Tensor dq = RL::Loss::MSE::df((ci == 0) ? q1o : q2o, target);
                squashQBackward(dq, (ci == 0) ? q1o : q2o);
                head.backward(trunk.output(), dq);
            } else {
                RL::Net &qnet = (ci == 0) ? q1 : q2;
                RL::Tensor target = (ci == 0) ? q1o : q2o;
                target[tr.action] = yClamped;
                RL::Tensor dq = RL::Loss::MSE::df((ci == 0) ? q1o : q2o, target);
                squashQBackward(dq, (ci == 0) ? q1o : q2o);
                qnet.backward(state, dq);
            }
        }

        /*
           策略损失 —— 两项都是"对 softmax 输出 π 的梯度", 相加后交给掩码 softmax
           的雅可比:
             (a) SAC 软 Q 项   dJ/dπ_a = α·(log π_a + 1) − min_i Q_i(s,a)
             (b) AlphaZero 监督项  CE(π_MCTS, π) 的梯度正好是 (π_a − π_MCTS,a);
                 用这个形式而不是 −π_MCTS/π, 因为后者在 π_a -> 0 时会爆掉。
        */
        for (int i = 0; i < ACTION_DIM; i++) {
            if (mask[i] <= 0.5f) {
                g[i] = 0.0f;   /* 非法动作不参与: π_i = 0, 绝不能把它抬起来 */
                continue;
            }
            float gi = a * (std::log(pi[i] + 1e-8f) + 1.0f)
                       - std::min(q1o[i], q2o[i]);
            if (tr.hasSearch) {
                gi += azWeight * (pi[i] - tr.pi[i]);
            }
            g[i] = gi;
        }
        maskedSoftmaxBackward(pi, g, dz);
        if (trunkMode == TrunkMode::Shared) {
            /*
               ---- 共享骨干: 三份梯度在骨干处**相加**, 然后只反向一次 ----
               顺序是硬约束:
                 1. 三个头的 backward 都要在 trunk.backward 之前 (它们读的是骨干的
                    缓存输出 h, 而 Layer<Tanh>::backward 结束时会把 o 清掉);
                 2. 每个头 backward 之后, 它对输入的梯度就在 `Net::inputGrad` 里
                    (Net::backwardFrom 把 layers[0]->backward 的结果存在那里) ——
                    三个 inputGrad 相加就是 dL/dh。
               为什么是"相加"而不是"三次反向": 骨干只有一份, 它的梯度本来就应该等于
               三条损失路径各自对它的梯度之和; 分三次反向 = 三次更新同一份权重, 那等于
               把学习率乘 3。
            */
            actorHead.backward(trunk.output(), dz);
            RL::Tensor gh = actorHead.inputGrad;
            gh += q1Head.inputGrad;
            gh += q2Head.inputGrad;
            trunk.backward(state, gh);
        } else {
            actor.backward(state, dz);
        }

        /*
           α 自动调节:
             J(α) = α·(H − H̄)  =>  dJ/dα = H − H̄   (H = 策略熵, H̄ = 目标熵)
           Optimize::RMSProp 是梯度下降 (w -= lr·g/√v), 所以
             熵低于目标 -> g < 0 -> α 变大 -> 更探索;
             熵高于目标 -> g > 0 -> α 变小 -> 更利用。
        */
        float H = 0.0f;
        for (int i = 0; i < ACTION_DIM; i++) {
            if (pi[i] > 0.0f) {
                H -= pi[i] * std::log(pi[i]);
            }
        }
        const int lc0 = (tr.legalCount > 1) ? tr.legalCount : 2;
        int lc = lc0;
        if (entropySlotsAsLegal) {
            /*
               H̄ 的分母换成**合法槽位数** (见头文件): π 只分布在槽位上, 而 128 槽哈希有
               碰撞 ⇒ H ≤ log(槽位数) < log(着法数); H̄ 按着法数算时可能永远达不到, 而
               α 的梯度恰好是 (H − H̄) ⇒ α 被单向推走。
            */
            int sc = 0;
            for (int i = 0; i < ACTION_DIM; i++) {
                if (mask[i] > 0.5f) { sc++; }
            }
            lc = (sc > 1) ? sc : 2;
        }
        const float Hbar = entropyRatio * std::log((float)lc);
        alphaGrad += (H - Hbar);
        n++;

        /*
           ---- [2026-09 ①] 训练中的 critic/α 诊断 (只累加, 不进任何梯度/更新路径) ----
           三个问题见 sacazagent.h 的 TrainDiag 说明。这里全部是**读**已经算好的量,
           不改变任何前向/反向/随机流 ⇒ 默认口径下的数值与改动前逐位一致。
        */
        {
            TrainDiag &D = trainDiag;
            D.n++;
            /*
               [2026-09 动态奖励分配] 终局通道的样本量: "后期重杀将"这类旋钮只作用在
               done=true 且**分胜负**的样本上 (和棋的终局值恒 0)。见 TrainDiag 的说明。
            */
            if (tr.done) {
                D.doneSamples++;
                if (tr.reward != 0.0f) { D.decisiveSamples++; }
            }
            const double yAbs = std::fabs((double)y);
            D.yPreAbsSum += yAbs;
            D.yPreSum += (double)y;
            if (yAbs > D.yPreAbsMax) { D.yPreAbsMax = yAbs; }
            if (clampTarget > 0.0f && yAbs > (double)clampTarget) { D.clamped++; }

            /* V(s') 的两项分解: E_π[min Q] 与熵项 α·H (y = r − γV) */
            double vQ = 0.0, vEnt = 0.0;
            for (int i = 0; i < ACTION_DIM; i++) {
                if (nextMask[i] <= 0.5f || piNext[i] <= 0.0f) { continue; }
                const double p = (double)piNext[i];
                vQ += p * (double)std::min(q1n[i], q2n[i]);
                vEnt += p * (-(double)a * std::log(p));
            }
            D.vNextSum += (double)vNext;
            D.vQSum += vQ;
            D.vEntSum += vEnt;

            /* 当前局面的合法槽位数 / 着法数, 以及 H 与两种 H̄ */
            double slotsCur = 0.0;
            double qm = 0.0, qs2 = 0.0;
            for (int i = 0; i < ACTION_DIM; i++) {
                if (mask[i] <= 0.5f) { continue; }
                slotsCur += 1.0;
                const double qq = (double)std::min(q1o[i], q2o[i]);
                qm += qq;
                qs2 += qq * qq;
            }
            double qSpread = 0.0;
            if (slotsCur > 0.0) {
                qm /= slotsCur;
                const double var = qs2 / slotsCur - qm * qm;
                qSpread = (var > 0.0) ? std::sqrt(var) : 0.0;
            }
            D.qSpreadSum += qSpread;
            D.qAbsMeanSum += std::fabs(qm);
            D.slotsSum += slotsCur;
            D.legalSum += (double)lc0;
            D.hSum += (double)H;
            D.hBarSum += (double)Hbar;
            const double hBarSlots = (double)entropyRatio
                                     * std::log((double)((slotsCur > 1.0) ? slotsCur : 2.0));
            D.hBarSlotsSum += hBarSlots;
            if ((double)H < (double)Hbar) { D.hBelowHbar++; }
            if ((double)H < hBarSlots) { D.hBelowHbarSlots++; }
            if (D.alphaFirst < 0.0) { D.alphaFirst = (double)a; }
            D.alphaLast = (double)a;
        }
    }
    }   /* ---- epochs 循环结束 (P4) ---- */

    if (n == 0) {
        return 0.0f;
    }

    /*
       本批的平均 critic 损失 (界面曲线用, 见 getLastTrainLoss)。
       取**批平均**而不是最后一条: 逐样本上报会让曲线变成低占空比的脉冲
       (重尾样本能差几个数量级), 而且与 DQN 报"平均平方 TD 误差"的口径对不上。
       P4 之后分母是 batchSize × epochs (每个 epoch 各抽 batchSize 条)。
    */
    m_lastLoss = lossSum / (float)n;
    m_lastBatchSamples = n;

    /*
       ---- [2026-09 独立类] 这里原来是"稀疏 MoE 的负载均衡辅助损失" ----
       纯 MLP 骨干没有 MoE 层, 没有路由会坍缩, 因此没有这一项。
       整套 L_aux (addAuxGradient + 批统计 + `auxLossCoef`) 连同 `resetMoeBatchStats()`
       一起**在两个 MoE 独立类里** (`SACAZMoEMlpAgent` / `SACAZMoETbAgent`) ——
       细节与有限差分验证见 rl/sparse_moe.hpp 与 test/test_sparse_moe_main.cpp [6][7]。
    */

    /* ---- 应用梯度 ----
       共享口径: 骨干只更新**一次** (用 learningRateTrunk)。若照旧对 actor/q1/q2 各调
       一次 RMSProp, 同一份骨干权重会被同一个累积梯度更新三次 = 学习率乘 3 ——
       一个完全静默、只让训练变坏的写法。 */
    if (trunkMode == TrunkMode::Shared) {
        trunk.RMSProp(learningRateTrunk, 0.9f, 0.0f);
        actorHead.RMSProp(learningRateActor, 0.9f, 0.0f);
        q1Head.RMSProp(learningRateCritic, 0.9f, 0.0f);
        q2Head.RMSProp(learningRateCritic, 0.9f, 0.0f);
    } else {
        actor.RMSProp(learningRateActor, 0.9f, 0.0f);
        q1.RMSProp(learningRateCritic, 0.9f, 0.0f);
        q2.RMSProp(learningRateCritic, 0.9f, 0.0f);
    }

    /* α 的梯度是整批累加的, 取平均后再更新 */
    alpha.g[0] = alphaGrad / (float)n;
    alpha.RMSProp(learningRateAlpha, 0.9f, 0.0f);
    /*
       [2026-09 实验轮] `alphaCeiling` 默认 5.0 = 改动前逐位相同。
       实测 α 会被推到上界, 而 α·H 是 TD 目标里的主项之一 ⇒ 上界决定"目标被顶多远"。
    */
    alpha.clamp(0.02f, 0.02f, (alphaCeiling > 0.0f) ? alphaCeiling : 5.0f);
    alphaSample = -1.0f;   /* 批结束后清掉采样值 (批外一律用学到的 alpha[0]) */

    /*
       ---- 目标网 Polyak 同步 ----
       步长是 `targetTau` (见头文件的 [F1] 说明), **不再是硬编码 1e-3**: 那个值配上
       "每 64 步一次"在"一次会话几千步"的尺度上等于不更新 (实测 20 局只移动 2~4%,
       目标网一直停在随机初始化尺度 0.07~0.10), 于是自举项里没有任何游戏信息。
       `targetTau >= 1` 时 softUpdateTo 等价于硬拷贝 (对照臂)。
       共享口径: 骨干只同步**一次**到唯一的目标骨干 (照旧写两遍会是同一个源的两次
       Polyak —— 第二次把刚写进去的值又往同一处推一次, 相当于步长变了)。
    */
    learnSteps++;
    if (learnSteps % (replaceTargetIter > 0 ? replaceTargetIter : 1) == 0) {
        if (trunkMode == TrunkMode::Shared) {
            trunk.softUpdateTo(trunkTarget, targetTau);
            q1Head.softUpdateTo(q1TargetHead, targetTau);
            q2Head.softUpdateTo(q2TargetHead, targetTau);
        } else {
            q1.softUpdateTo(q1Target, targetTau);
            q2.softUpdateTo(q2Target, targetTau);
        }
    }

    /* ---- 回放缓冲上限 ---- */
    while (memories.size() > maxMemorySize) {
        memories.pop_front();
    }

    return lossSum / (float)n;
}

/* ============================================================
 *  trainSelfPlay: 自对弈 (MCTS 访问分布做策略目标) + 回放训练
 * ============================================================ */
void SACAZAgent::trainSelfPlay(int episodes, int simulations_, int maxMoves,
                               bool verbose, float tempRoot, float tempFinal,
                               int learnEveryMoves)
{
    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int turn = Stone::COLOR_RED;
        int moves = 0;

        while (moves < maxMoves) {
            const int res = chess.getResult(turn);
            if (res != Chess::RESULT_ONGOING) {
                break;
            }

            /* 温度退火: 开局高 (多样性), 中后盘低 (质量) */
            const float frac = (float)moves / (float)(maxMoves > 1 ? maxMoves : 1);
            const float temp = tempRoot + (tempFinal - tempRoot) * frac;

            RL::Tensor piTarget(ACTION_DIM, 1);
            const Step s = selectMove(turn, simulations_, temp, &piTarget);
            if (!s.valid) {
                break;
            }

            Transition tr;
            encodeSparse(turn, tr.cells);
            contextOf(chess, turn, tr.ctx);
            for (int i = 0; i < ACTION_DIM; i++) {
                tr.pi[i] = piTarget[i];
            }
            tr.action = stepToActionIdx(s, turn);
            tr.hasSearch = true;
            tr.reward = computeReward(s, turn);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(turn, legal, idx, mask);
                tr.legalCount = (int)idx.size();
                maskToBits(mask, tr.curMask);
                Steps::instance().put(legal);
            }

            double dummy = 0.0;
            chess.moveForward(&s, dummy);

            const int nextTurn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                            : Stone::COLOR_RED;
            encodeSparse(nextTurn, tr.nextCells);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(nextTurn, legal, idx, mask);
                maskToBits(mask, tr.nextMask);
                Steps::instance().put(legal);
            }
            const int after = chess.getResult(nextTurn);
            tr.done = (after != Chess::RESULT_ONGOING);
            if (tr.done) {
                float rv = 0.0f;
                if (resultValue(after, turn, rv)) {
                    tr.reward = rv;   /* 终局奖励 ±1/0 覆盖即时奖励 */
                }
            }

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }

            turn = nextTurn;
            moves++;

            if (learnEveryMoves > 0 && (moves % learnEveryMoves) == 0) {
                learnBatch(batchSize);
            }
        }

        totalEpisodes++;
        const int res = chess.getResult(turn);
        if (res == Chess::RESULT_RED_WIN) {
            totalWins[0]++;
        } else if (res == Chess::RESULT_BLACK_WIN) {
            totalWins[1]++;
        }

        if (verbose) {
            std::printf("  [SAC+AZ] episode %d: %d 手, result=%d, 池=%zu, alpha=%.3f\n",
                        ep + 1, moves, res, memories.size(), (double)alpha[0]);
        }
    }
}

void SACAZAgent::warmupFromCurrent(int episodes, int simulations_, int maxMoves)
{
    /* 从当前局面继续 (不 reset) —— 调用方负责棋盘状态 */
    const int savedTurn = chess.sideToMove;
    trainSelfPlay(episodes, simulations_, maxMoves, false, 1.0f, 0.25f, 4);
    chess.sideToMove = savedTurn;
}

/* ============================================================
 *  exploreAndTrain: 走子前的"探索环境 + 在线训练一次"
 *
 *  与自对弈的区别: 这里**不做搜索**, 直接用掩码策略 π(·|s) 采样滚若干步 (最大熵
 *  策略本身就是探索策略), 把经验塞进回放缓冲后做一次 mini-batch 更新。
 *  这些样本的策略目标不是搜索出来的, 所以 hasSearch=false —— 只训练 critic 与
 *  SAC 的软 Q 项, 不用 AlphaZero 的监督项 (否则就是把策略往它自己身上拉)。
 * ============================================================ */
bool SACAZAgent::exploreAndTrain(int color, int rolloutSteps, const OpponentPolicy &opponent)
{
    if (rolloutSteps <= 0) {
        m_exploreInfo = "SAC+AZ: 探索步数为 0, 已跳过";
        return false;
    }
    /* 局部副本: rolloutFromCurrent 会把"真用了几手对手着法"回填到它里面 (P1) */
    OpponentPolicy opp = opponent;

    const int collected = rolloutFromCurrent(
        *this, chess, color, rolloutSteps,
        /* pick: 用掩码策略采样 */
        [this](const RL::Tensor &state, int turn) -> int {
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(turn, legal, idx, mask);
            if (legal.empty()) {
                Steps::instance().put(legal);
                return -1;
            }
            RL::Tensor pi(ACTION_DIM, 1);
            policy(state, mask, pi);
            /* 把这一步的合法数与掩码存下来, 给紧接着的 onTrans 复用 */
            m_pendingLegalCount = (int)idx.size();
            maskToBits(mask, m_pendingMask);
            Steps::instance().put(legal);
            return RL::Random::categorical(pi);
        },
        /* onTrans: 存一条 off-policy 经验 */
        [this](const Step &chosen, int actionIdx, const RL::Tensor &stateBefore,
               const RL::Tensor &nextState, float reward, bool done) {
            (void)chosen;
            Transition tr;
            denseToSparse(stateBefore, tr.cells);
            readContext(stateBefore, tr.ctx);
            denseToSparse(nextState, tr.nextCells);
            readContext(nextState, tr.nextCtx);
            tr.action = actionIdx;
            tr.legalCount = m_pendingLegalCount;
            tr.curMask[0] = m_pendingMask[0];
            tr.curMask[1] = m_pendingMask[1];
            tr.reward = reward;
            tr.done = done;
            tr.hasSearch = false;   /* 策略目标不来自搜索 */

            /* 下一局面的合法掩码: 此刻棋盘正好停在 nextState */
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(chess.sideToMove, legal, idx, mask);
            maskToBits(mask, tr.nextMask);
            Steps::instance().put(legal);

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }
        },
        opp);

    /*
       在线训练一次。**批大小要按池里的实际条数夹一下**: learnBatch 在
       "池 < batchSize(默认 32)" 时是**故意**直接返回 0 的 (不拿半个 batch 去更新),
       于是刚开局时(池里只有探索刚收集的十几条)这一轮就白跑了 —— 界面上表现为
       "损失曲线一直是空的", 而且在线学习在开局阶段完全没发生。
    */
    const int onlineBatch = std::min(batchSize, (int)memories.size());
    const float loss = learnBatch(onlineBatch);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "SAC+AZ 探索 %d 步, 训练 1 次 (池 %zu, critic loss %.4f, alpha %.3f)%s",
                  collected, memories.size(), (double)loss, (double)alpha[0],
                  opponentRolloutInfo(opp).c_str());
    m_exploreInfo = buf;
    return collected > 0;
}

/* ============================================================
 *  存取
 *  ------------------------------------------------------------
 *  两个口径的**文件个数与语义都不同**, 所以各写一份, 且靠前缀区分 (见头文件
 *  sharedWeightPrefix 的说明):
 *    Separate: 3 个文件 —— actor / q1 / q2, 目标网在载入时由在线网 copyTo 派生;
 *    Shared  : 4 个文件 —— trunk / actorhead / q1head / q2head。
 *
 *  为什么共享口径**不**写成"_trunk + _q1 + _q2"这种"看起来兼容"的形式: 共享口径没有
 *  独立的 actor 网, 而独立口径的 `_q1` 里含着一整份骨干。两种文件的**第 2 个文件**
 *  语义完全不同 (一个是 Q1 头, 一个是"骨干+Q1"), 共用一个前缀就等于给"载错文件"
 *  留门 —— 而载错的后果不是报错, 是把一份 28.8 M 的骨干当成 64x128 的头去用。
 * ============================================================ */
bool SACAZAgent::saveModel(const std::string &filepath)
{
    if (trunkMode == TrunkMode::Shared) {
        trunk.save(filepath + "_trunk");
        actorHead.save(filepath + "_actorhead");
        q1Head.save(filepath + "_q1head");
        q2Head.save(filepath + "_q2head");
        return weightFileWritten(filepath + "_trunk")
               && weightFileWritten(filepath + "_actorhead")
               && weightFileWritten(filepath + "_q1head")
               && weightFileWritten(filepath + "_q2head");
    }
    actor.save(filepath + "_actor");
    q1.save(filepath + "_q1");
    q2.save(filepath + "_q2");
    return weightFileWritten(filepath + "_actor")
           && weightFileWritten(filepath + "_q1")
           && weightFileWritten(filepath + "_q2");
}

bool SACAZAgent::loadModel(const std::string &filepath)
{
    if (trunkMode == TrunkMode::Shared) {
        if (!weightFileReadable(filepath + "_trunk")
            || !weightFileReadable(filepath + "_actorhead")
            || !weightFileReadable(filepath + "_q1head")
            || !weightFileReadable(filepath + "_q2head")) {
            return false;
        }
        const int rt = trunk.load(filepath + "_trunk");
        const int ra = actorHead.load(filepath + "_actorhead");
        const int r1 = q1Head.load(filepath + "_q1head");
        const int r2 = q2Head.load(filepath + "_q2head");
        if (rt != 0 || ra != 0 || r1 != 0 || r2 != 0) {
            std::cerr << "[weights] SACAZAgent(Shared)::loadModel 失败 (trunk=" << rt
                      << ", actorHead=" << ra << ", q1Head=" << r1 << ", q2Head=" << r2
                      << "), 未同步目标网" << std::endl;
            return false;
        }
        /* 与独立口径同一条纪律: 载入成功后**才**把目标网同步成在线网的副本 */
        trunk.copyTo(trunkTarget);
        q1Head.copyTo(q1TargetHead);
        q2Head.copyTo(q2TargetHead);
        return true;
    }

    if (!weightFileReadable(filepath + "_actor")
        || !weightFileReadable(filepath + "_q1")
        || !weightFileReadable(filepath + "_q2")) {
        return false;
    }
    /*
       三个网络都**必须**检查载入结果 (见 ppomcts_agent.cpp 同一处修正的说明):
       原来只按"文件可读"就返回 true, 于是结构/CRC 不匹配的检查点会被静默忽略,
       训练循环表现成"每轮从随机权重重来却报告成功"。
       任何一个失败都直接返回 false —— 此时不碰 q1Target/q2Target, 避免把目标网
       拷成"载入失败后的混合状态"。
    */
    const int ra = actor.load(filepath + "_actor");
    const int r1 = q1.load(filepath + "_q1");
    const int r2 = q2.load(filepath + "_q2");
    if (ra != 0 || r1 != 0 || r2 != 0) {
        std::cerr << "[weights] SACAZAgent::loadModel 失败 (actor=" << ra
                  << ", q1=" << r1 << ", q2=" << r2 << "), 未同步目标网" << std::endl;
        return false;
    }
    q1.copyTo(q1Target);
    q2.copyTo(q2Target);
    return true;
}

/* ============================================================
 *  AgentBase
 * ============================================================ */
Step SACAZAgent::getBestMove(int color)
{
    return selectMove(color, simulations, 0.0f);
}

/* ============================================================
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  为什么这个函数必须存在, 而且必须长这样: 训练损失与自对弈胜率都**不能**回答
 *  "这个模型值不值得继续训" —— 后者里赢家和输家是同一份权重, 前者只说明网络与
 *  自己的目标一致 (critic 把自己估平了, MSE 也在降)。真正的前置判据落在表示层与
 *  终局口径上, 而这两类事实原来在界面里一个都看不到。本报告逐条给出它们, 判读写
 *  在同一行末尾 (面板的读者是看训练曲线的人, 不是读代码的人)。
 *
 *  具体到这个 agent, 有四条读数最值得看:
 *
 *   (1) **本实例是哪个骨干**。这一个类同时被两个界面 agent 类型构造 —
 *       AGENT_SACAZ (Mlp) 与 AGENT_SACAZ_MOE (SparseMoeTb) —— 两块面板的数字会差
 *       很多 (参数量、有没有路由、每次前向 3 ms 级的 TB 专家开销)。所以第一行先报
 *       backboneName(backbone) 与它对应的界面类型, 否则"同名的两个 agent 读数不同"
 *       会被记到错的账上。
 *
 *   (2) **规则上下文可观测 / 动作层有别名** —— 一好一坏, 都写在同一节里。
 *       好的一面: 14 个棋子平面之后跟着 3 个标量槽 (无吃子进度 / 重复
 *       次数 / 将军), 三次重复与自然限着因此进了 V(s); DQNMCTS 那种 90 维
 *       "每格一个子力值" 编码下这三个量逐字节不可观测 (probe_dqnmcts_aliasing [2]),
 *       于是同一局面的第 2、第 3 次出现会编码成同一个向量。坏的一面:
 *       `stepToActionIdx` 把 (棋子 id, 目标格) 哈希进 128 个槽位, 而真实走法空间是
 *       8100 个 (from,to) 对 —— 同一个局面里若干个互不相同的着法被迫共用同一个
 *       策略头/Q 槽位。
 *       代价是**梯度被平均**: 策略头在那一维上收到的是两个不同着法梯度的平均,
 *       而"着法 A 与着法 B 不同"这件事在 128 维里根本没有坐标 —— 所以这是**结构性
 *       上限, 训练多少次都消不掉**, 不是收敛慢。落子本身仍然正确 (走法按**节点
 *       访问数**取, 不按动作索引, 见 selectMove 的收尾), 受损的是策略目标的精度。
 *       要根治只能换无碰撞动作空间 (id*90 + x*9 + y = 2880), 见
 *       docs/agents_design.md §11。
 *       标准开局那一份是**确定性**的参照点 (与棋盘现状无关, 只要走法生成器不变就
 *       一直一样); 当前局面那一份是实时读数, 所以面板每手刷新时它会变。
 *
 *   (3) **终局用真实胜负, 不自举** —— 搜索走到终局节点时 `terminalValue` 直接给出
 *       ±1/0, 训练目标的自举项也被 (1-done) 截断。这是本 agent 与 PPOMCTS 的差别
 *       (那边终局也用网络值), 也是"一局结束时的价值目标值得信"的原因: 最后那几步
 *       回归的是真结果, 不是网络自己的估计。
 *
 *   (4) **MoE 使用直方图** —— 路由坍缩 (少数专家吃掉全部样本, 其余永远拿不到梯度)
 *       在损失曲线上完全看不出来, 只能读这个直方图: 只要有一个专家的计数是 0 而
 *       别的不是, 那一支就已经死了。
 *
 *  **只读 / 可重复 / 不动棋盘** (aiagent.h 的契约): 需要在棋盘上数别名的两处都用
 *  `Chess probe(chess)` 的**副本**并局部解码 —— 本函数会在对局中途由 GUI 线程调用,
 *  而那一刻搜索线程可能正拿着 `this->chess` 走子/回退, 碰它就是与搜索抢棋盘。
 *  不改任何成员 (函数是 const), 不跑搜索/前向/权重 IO, 也不复位 MoE 计数
 *  (那是写操作; 直方图按**全生命周期**累计正好是坍缩诊断要的口径)。
 *
 *  刻意**不**报棋力: 棋力只有带置信区间的锚点对局 (bench_anchor, Elo 差 + 95% 区间)
 *  能回答 —— 见最后一行。
 * ============================================================ */
namespace {

/*
 *  一个局面上的动作别名 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了几个
 *  互不相同的着法)。
 *
 *  为什么是**自由函数**而不是成员: 它只做统计, 不碰棋盘、不碰网络, 唯一要用的是
/*
 *  一个局面上的动作**别名**读数 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了
 *  几个互不相同的着法)。
 *
 *  2026-09 换成 8100 双射之后这个函数的期望值是**恒等式**:
 *      slotCount == legalCount  且  worstSlot == 1
 *  它不再是"缺陷读数"而是一个**回归指示器**: 只要 slotCount < legalCount, 就说明
 *  动作索引又变成了会碰撞的写法 (或者规范镜像写错导致两个着法映到同一槽)。
 *  这正是本仓库的惯例 —— 把"本该成立的不变量"变成可见的断言, 而不是删掉读数。
 *
 *  为什么是**自由函数**: 它只做统计, 不碰棋盘、不碰网络, 唯一要用的是
 *  `stepToActionIdx` 那一份公式 —— 签名只要 `const SACAZAgent &` + color 就够。
 *  这样一来**面板数的一定是训练用的同一个公式** (DQNMCTS 那边只能把哈希再抄一份
 *  并靠注释提醒"改一处必须改两处"; 这里换成编译期耦合: 谁把那个 const 去掉, 这里
 *  当场编不过)。
 *
 *  去重: 每个槽位里存 (棋子 id, 目标格) 的集合, 统计的是**互不相同的走法**数,
 *  而不是合法着法列表的条数 —— 列表里若有重复项, 那也不该被算成"两个走法挤在一起"。
 *
 *  只读: 只看传进来的走法列表 (调用方负责 `Steps::instance().put()` 归还)。
 */
void aliasOfPosition(const SACAZAgent &ag, const std::vector<Step *> &legal, int color,
                     int &legalCount, int &slotCount, int &worstSlot)
{
    std::map<int, std::set<long long> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        const long long key = ((long long)s.id << 16)
                            | ((long long)s.nextPos.x << 8)
                            | (long long)s.nextPos.y;
        bucket[ag.stepToActionIdx(s, color)].insert(key);
    }
    legalCount = (int)legal.size();
    slotCount = (int)bucket.size();
    worstSlot = 0;
    for (std::map<int, std::set<long long> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        if ((int)it->second.size() > worstSlot) {
            worstSlot = (int)it->second.size();
        }
    }
}

} // namespace

std::string SACAZAgent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /* ---- 0. 本实例是**哪一支** ----
       [2026-09 独立类] 这一段原来要解释"一个类背着两个界面 agent 类型"的坑 —— 现在
       坑被结构性地填掉了: 本类**只有一种骨干** (纯 MLP), 界面类型也只有一个
       (AGENT_SACAZ)。这一行从"分辨身份"退化成"自报家门", 但仍然要在: 面板上还有
       MoE-MLP / TB 专家 / 59e5233 还原版, 读数不能被记到错的账上。 */
    std::snprintf(buf, sizeof(buf),
                  "界面 agent 类型 %s | 骨干 %s\n", guiAgentLabel(), backboneName());
    out += buf;
    /*
       隐层激活报**实测值** (读 actor 第 2 层的类型), 不是回显开关 —— 理由见
       hiddenActivationName() 的注释 (开关曾因"建网之后才赋值"而静默失效)。
       权重文件名不在这里印: 面板顶端那两行由 weightFilesOf() 给出**真实文件名**,
       比在这里复述一遍前缀更可靠 (前缀与实际写出的文件名曾经漂移过一次)。
    */
    std::snprintf(buf, sizeof(buf), "隐层激活(实测) %s\n", hiddenActivationName());
    out += buf;
    /*
       ---- 0b. 骨干共享口径 (2026-09 dev-sacmoetb) ----
       这一行是本次改动唯一能在面板上看见的东西, 它对应一个"原本完全看不见"的失效:
       Shared 与 Separate 的权重文件**互不通用**, 不写出来就没法解释"为什么这个 agent
       的权重载入失败了"。
       **[2026-09 独立类]** 原来这里还有一节"TB 专家的实际头数" —— 那一节**整段搬到了
       `SACAZMoETbAgent`**: 纯 MLP 骨干里没有 TransformerBlock, 那一节在本类里恒为
       "没有 TB 专家", 留着只是噪声。
    */
    std::snprintf(buf, sizeof(buf),
                  "骨干口径 %s | 唯一参数量 %lld (actor.paramCount=%lld 在共享口径下会把"
                  "骨干重复计入三张视图)\n",
                  trunkModeName(trunkMode), uniqueParamCount(), actor.paramCount());
    out += buf;

    /* ---- 1. 表示层: 状态编码 ----
       这一节的两行**必须随 SACAZ_ALIGNED_REPR 分支**: 默认构建是 1263 维 / 128 槽,
       而面板原来写死了"与 PPOMCTS 同口径 / 8100 双射" —— 那是一句在默认构建下
       不成立的话, 面板说谎比不说更坏 (用户口径: SAC 回退 1263/128, PPO 才用 8100)。 */
    if (ALIGNED_REPR) {
        std::snprintf(buf, sizeof(buf),
                      "状态 %d 维 = %d 平面 x %d 格 = %d 棋子平面 (CTX_BASE=%d)"
                      " + 5 上下文平面 (%d/%d/%d/%d/%d)\n",
                      STATE_DIM, PLANES, CELLS, PIECE_PLANES, CTX_BASE,
                      PLANE_MATERIAL, PLANE_TEMPO, PLANE_HALFMOVE, PLANE_REPEAT,
                      PLANE_CHECK);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "**与 PPOMCTSAgent 逐位同口径** (2026-09 对齐): 同一个 STATE_DIM=%d、"
                      "同一份 chessstate.h 的 CTX_* 顺序、同一个规范镜像、同一个 8100 双射"
                      "动作空间 -> 两个 agent 的差别只剩算法本身\n",
                      STATE_DIM);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "规则/阶段上下文: 5 个 (子力阶段 / 总手数 / 无吃子进度 / 重复次数 / 被将)"
                      " -> 三次重复 / 自然限着 / 将军都是 V(s) 的输入 (可观测)\n");
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf),
                      "状态 %d 维 = %d 棋子平面 x %d 格 (%d) + **3 个规则上下文标量**"
                      " (无吃子进度 / 重复次数 / 被将)\n",
                      STATE_DIM, PIECE_PLANES, CELLS, CTX_BASE);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "**改前表示 (SACAZ_ALIGNED_REPR 未定义, 默认)**: 与 PPO+MCTS 的 "
                      "1710 维/8100 双射**不是同一口径** —— 用户口径 (2026-09) 是"
                      " SAC 回退 1263/128、PPO 保持 8100。跨 agent 状态对齐按口径撤销;"
                      " 对齐表示仍可编译进来 (-DSACAZ_ALIGNED_REPR=1)\n");
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "规则/阶段上下文: 3 个标量 (子力阶段与总手数在改前表示里**没有**"
                      "通道, 那是后来对齐时才加的)\n");
        out += buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "  对照: PGE / DQN / DQNMCTS 用的是 90 维每格一个子力值, 规则上下文"
                  "通道 0 个 (同一局面的第 2/第 3 次出现在那里逐字节不可分)\n");
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "视角: 规范视角 (轮到黑方时 x->9-x, 己方永远在 x 大的一侧) -> 红黑共用"
                  "一套权重, 价值函数不必从编码里反推该谁走\n");
    out += buf;

    /* ---- 2. 动作层: 双射 / 哈希 + **不变量**读数 (标准开局 / 当前局面) ----
       对齐表示下"别名"应当恒为 0 (回归指示器); 改前表示下它是**结构性上限**
       (两个不同着法共用一槽 -> 策略头收到两者梯度的平均), 见 docs 的 13.83 vs 0.43。 */
    if (ALIGNED_REPR) {
        std::snprintf(buf, sizeof(buf),
                      "动作 %d = 双射 canonicalCell(from)*%d + canonicalCell(to)"
                      " (与 PPOMCTS 同一公式, 无碰撞; 曾是 128 槽哈希)\n",
                      ACTION_DIM, CELLS);
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf),
                      "动作 %d = **%d 槽哈希别名空间** (改动前表示; 真实着法空间是 %d 个"
                      " (from,to) 对 -> 别名是策略精度的结构性上限, 不是训练量的问题)\n",
                      ACTION_DIM, ACTION_DIM, ALIGNED_ACTION_DIM);
        out += buf;
    }

    int initLegal = 0, initSlots = 0, initWorst = 0;
    int curLegal = 0, curSlots = 0, curWorst = 0;
    {
        /* 只看副本: 绝不能在 GUI 线程碰 this->chess (搜索可能正在用它) */
        Chess probe(chess);
        probe.reset();
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(*this, legal, probe.sideToMove, initLegal, initSlots, initWorst);
        Steps::instance().put(legal);
    }
    {
        /* 当前局面的同一份读数 (副本保留现状, 不做 reset) */
        Chess probe(chess);
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(*this, legal, probe.sideToMove, curLegal, curSlots, curWorst);
        Steps::instance().put(legal);
    }
    /*
       判据字符串随表示变: 对齐表示下 slotCount==legalCount 是**恒等式**, 不成立就是
       索引公式/规范镜像写错 (回归); 改前表示下挤掉几个是**正常现象**, 那里只看数值。
    */
    const char *badInit = ALIGNED_REPR
                              ? ((initLegal == initSlots && initWorst <= 1)
                                     ? "[不变量成立]" : "**回归!**")
                              : "[结构性上限读数, 不是回归]";
    const char *badCur = ALIGNED_REPR
                             ? ((curLegal == curSlots && curWorst <= 1)
                                    ? "[不变量成立]" : "**回归!**")
                             : "[结构性上限读数, 不是回归]";
    std::snprintf(buf, sizeof(buf),
                  "动作别名(标准开局): %d 个合法着法 -> %d 个槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法) %s\n",
                  initLegal, initSlots, initLegal - initSlots, initWorst, badInit);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "动作别名(当前局面): %d 个合法着法 -> %d 个槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法) %s\n",
                  curLegal, curSlots, curLegal - curSlots, curWorst, badCur);
    out += buf;
    if (ALIGNED_REPR) {
        out += "  为什么这项现在是回归指示器: 128 槽哈希时代它量的是**结构性上限**"
               "(两个不同着法共用一列 -> 策略头收到两者梯度的平均); 换成双射后"
               " slotCount==legalCount 是恒等式, 一旦不成立就是索引公式或规范镜像写错了\n";
    } else {
        out += "  改前表示下这一项**不是缺陷**: 槽位比合法着法多时可能无碰撞, 着法变多时"
               "必然开始挤 —— 它是策略精度的上限读数 (上面那两行就是当前局面的实际值)\n";
    }

    /* ---- 3. 算法 / 口径 ---- */
    /*
       [F1] 这里**必须打印实际生效的 tau**, 而不是写死 "tau=1e-3":
       本轮的教训是"回显开关的检查永远通过" —— 自检面板写死一个常数时, 就算代码里的
       默认值已经改掉, 面板也照样显示旧值 (而它会被人当成"实际口径")。
       后面那行把"一次会话 (~2600 次 learn) 能移动多少"直接算出来: 低于 50% 就等于
       自举项里没有游戏信息 (实测老口径只有 2~4%)。
    */
    const double kSessionLearnSteps = 2600.0;   /* 20 局 x ~130 步的典型 learn 次数 */
    const double kIter = (double)(replaceTargetIter > 0 ? replaceTargetIter : 1);
    const double perIter = 1.0 - std::pow(1.0 - (double)targetTau, 1.0 / kIter);
    const double moved = 1.0 - std::pow(1.0 - perIter, kSessionLearnSteps);
    std::snprintf(buf, sizeof(buf),
                  "双 critic q1/q2 + 目标网 q1Target/q2Target | 每 %d 次 learn 做一次 Polyak"
                  " 同步 (tau=%.4f) | 叶子价值 = min_i Q_i - alpha*log pi (最大熵软价值)\n",
                  replaceTargetIter, (double)targetTau);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "目标网移动率: 2600 次 learn (~20 局) 后相对随机初始化 %.1f%% %s\n",
                  moved * 100.0,
                  moved < 0.5 ? "**< 50%: 自举项 V(s') 几乎还是随机网, TD 目标里没有游戏信息 ([F1])**"
                              : "(跟得上在线网)");
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "alpha=%.3f (自动调节, 界 [0.02, 5]) | 目标熵 %.2f x log(合法着法数) |"
                  " azWeight=%.2f | c_puct=%.2f | 模拟次数=%d | gamma=%.2f\n",
                  (double)getAlpha(), (double)entropyRatio, (double)azWeight,
                  (double)c_puct, simulations, (double)gamma);
    out += buf;
    /*
       **口径行的价值**: AGENT_SACAZ 与 AGENT_SACAZ_OLD 用的是同一份算法, 差别只剩
       下面这一行的几个数 (再加大括号里的激活)。不印出来, "两个 SAC 谁强"就没法归因。
       `clampTarget/huberDelta <= 0` = 不夹目标 / 纯 MSE (59e5233 没有这两条约束)。
    */
    std::snprintf(buf, sizeof(buf),
                  "口径: clampTarget=%s | huberDelta=%s | 叶子估值=%s | rewardScale=%.2f |"
                  " valueScale=%.2f | 动作空间=%s\n",
                  clampTarget > 0.0f ? std::to_string(clampTarget).c_str() : "关(不夹)",
                  huberDelta > 0.0f ? std::to_string(huberDelta).c_str() : "关(纯 MSE)",
                  sparseLeafEval ? "稀疏头(只算合法列)" : "全量(与 59e5233 相同)",
                  (double)rewardScale, (double)valueScale,
                  legacyHashAction ? "对齐双射 8100" : "128 槽哈希");
    out += buf;
    out += "  这两行就是 AGENT_SACAZ 与 AGENT_SACAZ_OLD 的全部差异 (再加隐层激活);"
           " 两边都保留是为了能在界面上直接对弈比较, 而不是靠两份会漂移的实现\n";
    std::snprintf(buf, sizeof(buf),
                  " (-log pi >= 0), 所以 alpha 越大, 选择多的局面估值越高\n",
                  (double)learningRateActor, (double)learningRateCritic,
                  (double)learningRateAlpha);
    out += buf;
    out += "终局: 搜索到终局节点取**真实胜负** (terminalValue 给 ±1/0), 不自举; 训练目标"
           "的自举项也被 (1-done) 截断 -> 一局最后几步回归的是真结果而不是网络自己的"
           "估计, 这是它局末价值目标可信的原因 (PPOMCTS 那边终局仍用网络值)\n";

    /* ---- 4. 骨干 ----
       [2026-09 独立类] 本类只有一种骨干 (纯 MLP): 两行就报完。
       原来这里还有"专家隐层 / auxLossCoef"与一整节 MoE 路由读数 (专家数 / topK /
       使用计数直方图 / 坍缩判读) —— 那些**整段搬到了两个 MoE 独立类**的自检报告里,
       在本类里它们恒为"无稀疏层"。
    */
    std::snprintf(buf, sizeof(buf),
                  "骨干结构: 纯 MLP %d -> %d -> %d -> %d (两个 Tanh 隐层, 无 MoE 层)\n",
                  STATE_DIM, hiddenDim, hiddenDim, ACTION_DIM);
    out += buf;
    out += "MoE: 无稀疏层 (本骨干是纯 MLP) -> 没有路由, 不存在专家坍缩\n";

    /* 参数量: Net::paramCount() 各层求和 (MoE/TransformerBlock 也实现了它, 所以这里
       报的是真实总数, 不是"只算 Linear" 的旧口径)。目标网只前向, 单独列出来。 */
    std::snprintf(buf, sizeof(buf),
                  "参数量: actor %lld | q1 %lld | q2 %lld | 目标网合计 %lld (只前向) |"
                  " alpha 1\n",
                  actor.paramCount(), q1.paramCount(), q2.paramCount(),
                  q1Target.paramCount() + q2Target.paramCount());
    out += buf;

    /* ---- 5. 训练进度 ---- */
    const int epochs = (replayEpochs > 0) ? replayEpochs : 1;
    std::snprintf(buf, sizeof(buf),
                  "训练进度: learnSteps=%d | 自对弈局数=%d | 回放池 %zu/%zu | 每次更新"
                  " %d 条 x %d epochs (上次实际攒 %d 条) | 叶子评估 %lld 次\n",
                  learnSteps, totalEpisodes, memories.size(), maxMemorySize,
                  batchSize, epochs, getLastBatchSamples(), getLeafEvals());
    out += buf;
    out += "  叶子评估 = 软价值前向次数 (每次模拟一次, 是**算力**读数, 不是搜索质量)"
           " —— 它随模拟次数线性长, 不代表搜得更准\n";
    if (getLastBatchSamples() > 0 && getLastBatchSamples() != batchSize * epochs) {
        std::snprintf(buf, sizeof(buf),
                      "  本批样本数 != batchSize x epochs -> 多 epoch 复用可能没生效"
                      " (每遍应当**重新抽** batchSize 条, 见 replayEpochs 的说明)\n");
        out += buf;
    }

    const float loss = getLastTrainLoss();
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "最近一次 learnBatch 的 critic MSE: %.6g (批平均) —— 它衡量 critic"
                      " 与自己的目标差多少, **不是棋力**\n", (double)loss);
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf),
                      "最近一次 learnBatch 的 critic MSE: 未上报 (NaN, 还没学过或池 <"
                      " batchSize; 曲线控件会丢弃非有限值)\n");
        out += buf;
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局"
           " (带 95% 置信区间的 Elo 差) 回答\n";
    return out;
}
