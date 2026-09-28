#ifndef SACAZ_MOEMLP_AGENT_H
#define SACAZ_MOEMLP_AGENT_H

#include <vector>
#include <string>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <cstdint>
#include "chess.h"
#include "aiagent.h"
#include "rl/net.hpp"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"
#include "rl/parameter.hpp"

/* 稀疏 MoE 层只需要指针/引用 (定义在 rl/sparse_moe.hpp, 由 .cpp 包含) */
namespace RL {
class ISparseMoE;
}

/*
 * SACAZMoEMlpAgent - SAC + MCTS + AlphaZero
 * ================================================================
 *
 * 一句话: 用**最大熵 critic (SAC)** 给**AlphaZero 式 PUCT 搜索**提供叶子估值,
 * 用搜索得到的访问分布做策略改进目标, 并且**从回放缓冲里离线学习**。
 *
 * 三个成分各自负责什么
 * --------------------
 *   AlphaZero(MCTS)  : 策略改进算子。PUCT 选择 + 访问计数 -> 比当前策略更强的
 *                      目标分布 π_MCTS, 这是监督信号, 方差极低。
 *   SAC(最大熵)      : (a) **叶子估值**: 用软价值
 *                          V(s) = Σ_a π(a|s)·( min_i Q_i(s,a) − α·log π(a|s) )
 *                      而不是一个单独的价值头 —— 于是搜索是按"熵正则化的价值"
 *                      展开的, 与策略本身自洽;
 *                      (b) **双 Q + 目标网 + 回放** : 象棋奖励是稀疏终局奖励,
 *                      on-policy 方法每局只能拿到一个 ±1; 回放让一条终局经验被反复
 *                      利用, 这是 SAC 这一半真正的价值所在;
 *                      (c) **α 自动调节**: 熵低于目标就抬高 α (鼓励探索), 高于目标
 *                      就压低, 不需要手工调探索率。
 *
 * 与 `RL::SAC` 的关系
 * -------------------
 * `src/rl/sac.cpp` 已经是一个**离散动作**的 SAC (one-hot 动作 + softmax 策略 +
 * 对全部动作求期望的软备份 + 可学习 α), 本 agent 复用了它同样的数学, 但没有直接
 * 用它, 原因有三个, 都是"象棋 + 搜索"特有的:
 *   1. **合法走法掩码**: 128 维动作空间里绝大多数是非法走法, SAC 的策略必须在
 *      掩码后的分布上定义 (π(a)=0, a 非法)。`RL::SAC` 没有掩码, 也把 critic 的
 *      输入拼成 [state; prob], 与掩码策略不自洽。
 *   2. **与 MCTS 的整合**: 策略目标来自搜索访问分布, 叶子估值要换成软价值,
 *      这需要 agent 内部掌握前向/反向的顺序 (库里的 SAC 是黑盒)。
 *   3. **掩码 softmax 的梯度**: 策略头的掩码归一化必须自己实现它的雅可比
 *      (见 maskedSoftmaxBackward), 不能只对 `Layer<Softmax>` 的输出打补丁。
 * 复用的是库里的 `RL::Net` / `Layer<*> ` / `Loss::MSE` / `Optimize` / `Random`。
 *
 * 状态编码: 规范视角 14x90 = 1260 维 (与 EVABAgent 同一约定)
 *   7 种棋子 x {己方, 对方} 的 one-hot 平面, 轮到黑方时把棋盘上下镜像
 *   (x -> 9-x)。这样"轮到谁走"就被编码进视角里, 同一个网络对红黑双方都是
 *   "我方的子在自己的下方", 价值/策略不需要再从非对称编码里去猜该谁走 ——
 *   这也是 AlphaZero 必须做的规范化。注意本项目其它 RL agent (PG/DQN/PPOMCTS/
 *   DQNMCTS) 用的是 90 维非规范编码, 那种编码下价值函数的"谁走"是隐含的。
 *
 * 动作编码: 128 维 one-hot, 沿用本项目的确定性哈希 (见 stepToActionIdx)。
 *   已知局限: 该哈希会碰撞 (最多 ~50 个合法走法挤进 128 格), 碰撞会让两个不同的
 *   走法共用同一个策略/Q 槽位。选择走法是按**节点访问数**取的 (不是按动作索引),
 *   所以落子本身仍然正确; 受影响的是策略目标的精度。要彻底解决就换成无碰撞的
 *   id*90 + x*9 + y (32x90=2880) 动作空间, 见 docs/agents_design.md §11。
 */
class SACAZMoEMlpAgent : public AgentBase
{
public:
    /* ================================================================
     *  状态/动作表示: **编译期二选一** (2026-09 回归排查的结论)
     * ================================================================
     *
     *  `SACAZ_ALIGNED_REPR` 未定义 (默认) = **改前表示**: 17 平面 + 128 槽哈希动作。
     *  定义它 (= 1) = **与 PPOMCTSAgent 对齐的表示**: 19 平面 + 8100 双射动作。
     *
     *  为什么默认是"退回去": 两套表示在**同一套工具、同一协议、同一随机种子、
     *  同一个 moe-mlp 骨干**下量出来的差距是决定性的 (随机权重, 对 MCTS 80 局):
     *
     *      改前 1263 / 128 槽 :  72.5%  [61.9, 81.1]   (另一颗种子复核 68.8%)
     *      对齐 1710 / 8100   :  45.0%  [34.6, 55.9]
     *
     *  两个区间不重叠 ⇒ 对齐在**当前数据量**下净亏约 200 Elo。逐项消融排除了
     *  用户最初怀疑的三项 (奖励塑形 / 温度 α / 抑制 critic), 也排除了"隐藏层不足"
     *  (§4: 加宽隐藏层在两种表示下都更差)。真正的机制是**动作头宽度**:
     *  策略头参数从 8,192 涨到 518,400 (63 倍) 而瓶颈 h=64 不动 ⇒ 参数/数据比失衡,
     *  106 局自对弈的数据撑不起 8100 路输出的策略头。
     *
     *  所以:
     *    * **默认**用改前表示 (实测更强, 也是界面一直的默认);
     *    * 对齐表示保留为**可编译进来的选项**, 供"数据量足够时重新评估"用;
     *    * 两边都保留 critic 值域约束与 Huber —— 它们在**两种**表示下都不损失棋力
     *      (改前表示 + 钳位 = 73.1%, 与不加钳位的 72.5% 同水平)。
     *
     *  完整数据与复现命令见 docs/sac_regression_2026_09.md。
     * ================================================================ */
    static constexpr int CELLS = 90;                  /* 10x9 */
    static constexpr int PIECE_PLANES = 14;           /* 7 类棋子 x {己方, 对方} */

#if defined(SACAZ_ALIGNED_REPR) && SACAZ_ALIGNED_REPR
    static constexpr bool ALIGNED_REPR = true;
#else
    static constexpr bool ALIGNED_REPR = false;
#endif

    /* ---- 上下文个数: 对齐 = 5 (与 PPOMCTS 同), 改前 = 3 (无吃子/重复/将军) ---- */
    static constexpr int CTX_COUNT = ALIGNED_REPR ? 5 : 3;
    /* ---- 平面数 / 状态维 ---- */
    static constexpr int PLANES = PIECE_PLANES + CTX_COUNT;   /* 19 或 17 */
    static constexpr int CTX_BASE = PIECE_PLANES * CELLS;     /* 1260: 棋子区之后是上下文 */
    /* 对齐: 上下文铺成整平面 (与 PPOMCTS 逐位相同) -> 14..18 平面
       改前: 上下文只占尾部 3 个标量槽 (稀疏回放友好)      -> 1260..1262 */
    static constexpr int STATE_DIM = ALIGNED_REPR ? (PLANES * CELLS) : (CTX_BASE + CTX_COUNT);
    /* 上下文平面序号 (只在 ALIGNED_REPR 下有意义) */
    static constexpr int PLANE_MATERIAL = 14;         /* CTX_MATERIAL: 子力阶段 */
    static constexpr int PLANE_TEMPO    = 15;         /* CTX_TEMPO:    总手数阶段 */
    static constexpr int PLANE_HALFMOVE = ALIGNED_REPR ? 16 : 14;   /* 无吃子进度 */
    static constexpr int PLANE_REPEAT   = ALIGNED_REPR ? 17 : 15;   /* 重复次数 */
    static constexpr int PLANE_CHECK    = ALIGNED_REPR ? 18 : 16;   /* 是否被将 */

    /* ---- 动作编码 ---- */
    static constexpr int LEGACY_ACTION_DIM = 128;
    static constexpr int ALIGNED_ACTION_DIM = CELLS * CELLS;        /* 8100 */
    static constexpr int ACTION_DIM = ALIGNED_REPR ? ALIGNED_ACTION_DIM : LEGACY_ACTION_DIM;

    /*
       运行时消融开关 (只在**对齐**表示下有意义): 把动作退回 128 槽哈希, 用来单独
       隔离"动作空间"这个变量。ALIGNED_REPR=1 编译进来时可用, 否则恒为 false。
    */
    bool legacyHashAction = false;

    /* ----------------------------------------------------------------
     *  骨干: **本类只有一种** —— 稀疏 MoE (E=8, top-2) + MLP 专家
     *
     *  [2026-09 独立类] 这一支原来和另外三种骨干挤在 `SACAZAgent` 一个类里, 靠
     *  `Backbone` 枚举 + 运行时 switch 选层。现在它是**自己的类**
     *  (`SACAZMoEMlpAgent`), 骨干在 `makeTrunkLayers` 里硬编码:
     *
     *      稀疏 MoE (E=8, top-2, 专家 = MLP) + h -> out
     *
     *  为什么拆成独立类 (用户口径, 与 59e5233 还原版 `SACAZLegacyAgent` 同一套理由):
     *    * **隔离是结构性的, 不是约定**: "改 TB 专家那一支不许影响 MLP 专家那一支"
     *      原来只能靠 git diff 自证 (而 diff 只证明"这一次没改"); 现在两支在**不同的
     *      类、不同的文件**里, 改一个文件碰不到另一个。
     *    * **一个类 = 一个骨干**: 不再有"同一个类背着两个界面 agent 类型"的坑。
     *    * 全部实验旋钮**照旧保留** (trunkMode / rewardShape=3 的动态奖励分配 /
     *      目标网 / 熵项 / α …), 所以 A/B 能力一位没丢。
     *
     *  本类**没有**的东西 (它们只属于 TB 那一支, 见 sacazmoetbagent.h):
     *    `tbHonorHeads` / TB 头数读数 / `moeDense` (EqualParameter 对照组是 TB 才有的)。
     * ---------------------------------------------------------------- */
    /* 本类骨干的**名字** (自检面板/工具报告第一行要能回答"我是哪一支").
       非静态: 与 TB 那一支保持同一套调用写法 (`sac.backboneName()`) —— 工具的报告
       代码对四个骨干是同形的, 少一个特例。 */
    const char *backboneName() const;

    /*
       ================================================================
        [2026-09 dev-sacmoetb] **共享骨干** (trunkMode): 一个骨干 + 三个头
       ================================================================
       现状 (TrunkMode::Separate, 默认): actor / q1 / q2 / q1Target / q2Target
       **五张各自独立的网**, 每张都背着一整套骨干 (TB 专家时 28.8 M 参数/张)。
       代价有三个, 全是实测出来的:

         * **搜索**: 一次叶子估值 = policySparse(actor 前向) + qValuesSparse(q1 前向 +
           q2 前向) —— 同一个局面, 骨干被算了**三遍**, 而这三遍的差别只在最后那层
           64->128 的线性头。实测 1 个 TB 专家前向 5.4 ms (修完头数口径之后), 也就是说
           一次模拟里 16 ms 花在"把同一件事算三遍"上。
         * **训练**: learnBatch 每个样本要跑 6 次全网前向 (actor(s'), 两个目标网(s'),
           actor(s), q1(s), q2(s)) + 3 次全网反向。共享之后只需要 **3 次骨干前向**
           (在线骨干 s' / 目标骨干 s' / 在线骨干 s) + 3 次头反向 + 1 次骨干反向。
         * **参数/数据比**: 5 张网 = 5 x 28.8 M = 144 M 参数, 而自对弈数据以"千条"计。
           本文件顶部 §表示那条结论(策略头 63 倍参数量就吃掉 200 Elo)在这里同样成立 ——
           只是被"五张网"放大了 5 倍。

       TrunkMode::Shared: **骨干只留一份** (在线 + 目标各一份, 目标网照旧只前向),
       actor/q1/q2 退化成三个只有输出层的头。这是 AlphaZero 的标准结构 (共享 body +
       policy/value 两个头), 三点好处:
         1. 搜索每次叶子估值 3 次骨干前向 -> **1 次** (精确 3 倍);
         2. 训练每样本 6 次骨干前向 -> 3 次, 3 次骨干反向 -> 1 次;
         3. 骨干同时收到策略与双 Q 的梯度 (3 份信号), 而不是三张网各自从零学表示。

       **actor / q1 / q2 这三个成员在两种模式下都仍然是"端到端可用"的完整网络**
       (共享模式下它们与 trunk / 各 head **共享同一批层对象**, 见 .cpp 的 buildNet),
       所以 test / GUI / selfCheckReport 里所有 `agent.actor.xxx` 的用法一行都不用改;
       区别只在 agent 内部的热路径显式地"骨干跑一次、三个头各跑一次"。
       `Shared` 只保证**语义**不变 (同样的 π、同样的 Q), 不保证逐位 —— MoE 的门控与
       专家的中间量被复用, 浮点加法结合顺序会有极小差别 (test_sacaz [17] 钉住量级)。

       **默认值是 Separate** (与改动前逐位一致, 所有历史读数/权重不受影响);
       GUI 的 AGENT_SACAZ_MOE 那一支显式改用 Shared (权重前缀也换成独立的一支,
       见 defaultWeightPrefix / sharedWeightPrefix —— 两个口径的权重**绝不能互相载入**)。
    */
    enum class TrunkMode {
        Separate = 0,
        Shared,
        Count
    };
    static const char *trunkModeName(TrunkMode m);
    bool sharedTrunk() const { return trunkMode == TrunkMode::Shared; }
    /* 构造时定, 之后不要改 (改了 Net 的组装关系就与它不一致了) */
    TrunkMode trunkMode;

    /*
       ================================================================
        [2026-09 独立类] 本类**没有** TB 专家的头数口径开关 (`tbHonorHeads`)
       ================================================================
       MLP 专家里没有 `MultiHeadAttention`, 所以那条"请求 15 个头却只跑 3 个"的静默
       降级与本类无关 —— 那个开关、那四个头数读数 (`tbHeadsRequested/Used/Dim/
       Allocated`) 以及 `tbAttentionElements()` 都**只留在 `SACAZMoETbAgent`** 里。

       这条正是"独立类"想买到的东西: 拆分前它们在一个类里, MLP 专家那一支也要背着
       一个恒为 -1 的读数与一个恒不生效的开关 (读面板的人还得先判断骨干才知道该不该
       看那几行)。现在**结构上就没有**: 本类的自检报告里不会出现"TB 专家头数"这一节。
       来龙去脉 (降级机制 / 为什么参数指纹没报警 / 实测 15.71 ms vs 5.37 ms) 见
       `src/sacazmoetbagent.h` 的同名小节与 `docs/dev_sacazmoetb_2026_09.md`。
       ================================================================
     */

    /* ----------------------------------------------------------------
     *  共享骨干 (只在 trunkMode == Shared 时有内容)
     *  非共享模式下这几个 Net 是空的 (size() == 0), 热路径不会碰它们。
     * ---------------------------------------------------------------- */
    RL::Net trunk;          /* 共享骨干, 有梯度 (在线) */
    RL::Net trunkTarget;    /* 共享骨干的目标副本, 无梯度 */
    RL::Net actorHead;      /* 三个头: 输入 = 骨干输出 h (hiddenDim 维) */
    RL::Net q1Head;
    RL::Net q2Head;
    RL::Net q1TargetHead;
    RL::Net q2TargetHead;

    /*
       ================================================================
       隐层激活: 59e5233 用的是 `Layer<Tanh>`, 而**不能**用 TanhNorm 去"复现"它
       ================================================================
       背景: 这一层激活在**未提交的工作区改动**里被从 `RL::Layer<RL::Tanh>` 换成了
       `RL::TanhNorm<RL::Sigmoid>`, 后者让随机权重下的棋力掉 26 个点 (对 MCTS 70%
       -> 44%)。改回 `Layer<Tanh>` 之后当前 agent 与 59e5233 逐手等价
       (tools/verify_sac_equiv_59e5233.ps1)。**现在 buildNet 与 59e5233 是逐字相同的
       代码**, 所以"复现那一层"靠的是"同一行代码", 不需要任何开关。

       ---- 一条被实测否掉的"等价写法" (留在这里, 省得下次再想一遍) ----
       曾经试过用 `RL::TanhNorm<RL::Linear>` 且 `r = 1` 来"复现"那一层, 理由是
       "tanh 套在外面、Fn = 恒等、r = 1 ⇒ 就是 tanh(Wx+b)"。**这个推理是错的**:
       `TanhNorm::forward` 把偏置加在 tanh **外面**
           o1 = W·x;  o1 *= r;  o2 = tanh(o1);  o = Fn::f(o2 + b)
       即 `tanh(r·Wx) + b`; 而 `Layer<Tanh>::forward` 是 `tanh(W·x + b)`。
       两者只在 b ≡ 0 时相同, 偏置非零时**不是同一个函数**。
       实测 (test_sacaz 的 [14] 节, 同权重同局面, 稀疏 MLP 专家骨干):
       max|Δπ| = 1.8e-07, **max|ΔQ| = 8.9e-06** —— 既不逐位, 也不是"浮点噪声级"。
       而且这个错误当时**测不出来**: 那个开关写成了普通成员, 赋值发生在构造函数建网
       **之后** ⇒ 静默空操作, 于是"两个变体是否等价"的检查永远通过。
       教训与 docs/sac_regression_2026_09.md §7 第 2 条同源: **"数学上看着等价"不算数,
       要拿同一输入比输出**。结论: 需要 59e5233 的那一层, 就用 `Layer<RL::Tanh>` 这一行
       本身。

       `hiddenActivationName()` 保留下来当**回归指示器**: 它读的是 actor 第 2 层的
       真实类型, 以后谁再动这一层 (换成 Sigmoid / TanhNorm / 加一层), 自检面板上会直接
       显示出来 —— 上一轮那个 26 个点的回归, 面板上原本一个字都看不出来。
    */
    const char *hiddenActivationName() const;

    /*
       ---- "我是界面上的哪一支" ----
       自检面板第一行必须能回答这个问题。这里它是**虚函数**而不是写死的字符串,
       因为同一个类被两个界面类型复用 (AGENT_SACAZ / AGENT_SACAZ_MOE 按骨干分),
       若第一行还印"AGENT_SACAZ", 看面板的人会把两套读数混成一个 agent (本文件顶部
       那条"一个类背着两个界面类型"的教训就是这个坑的第一次)。
       [2026-09] 59e5233 还原版**不是**本类的派生类了 (它是独立类 SACAZLegacyAgent,
       见 src/sacazlegacyagent.h), 所以它不覆盖这个虚函数 —— 它有自己的同名实现。
    */
    virtual const char *guiAgentLabel() const;

    /*
       ---- 权重文件前缀 (把"是哪一个类"与"写到哪个文件"绑在一起) ----
       用户口径 (2026-09): **新旧 SAC 的权重文件必须用不同名字**。
       为什么不能共用一个前缀:
         * 两者可训练的口径不同 (见 SACAZLegacyAgent 的头注释: 熵比 / alpha 学习率 /
           critic 值域约束 / 叶子估值口径), 同一局面对弈会被训成两组不同的权重;
           共用一个前缀 = 后训练的那一支**静默覆盖**另一支, 而界面上一切正常;
         * 载入也一样: 载进来的权重看起来"能用" (结构指纹相同), 于是错的那一份会被
           当成对的那一份用。
       所以前缀跟着**类**走 (当前口径一个、还原版一个), 界面按 agent 类型取默认值,
       不靠各处手抄字符串。还原版的返回值在**另一个类**里 (SACAZLegacyAgent, 不是本类的
       派生类: 两个类没有继承关系, 只是刻意共用同一套命名约定)。
    */
    static const char *defaultWeightPrefix();

    /*
       共享骨干口径的权重前缀 (2026-09 dev-sacmoetb)。
       为什么**必须**与 defaultWeightPrefix() 不同 (用户口径: 新旧 SAC 的权重文件
       必须用不同名字, 见上面那一段): 共享口径把"5 张各自独立的网"换成"1 个骨干 +
       3 个头", 两者的**权重文件个数与语义都不同** (共享口径写 4 个文件, 没有 _actor,
       骨干只存一份)。如果共用一个前缀, `loadModel` 会找不到文件而返回 false ——
       看起来是"载入失败", 但如果有人为了"让老文件也能读"而补一条回退, 载进来的
       `_actor/_q1/_q2` 会被当成"骨干 + 头"使用: 那是把三份**互不相同**的骨干随机
       初始化里的第一份当共享骨干用, 另外两份直接丢掉 (静默, 不报错)。
       所以两个口径各有各的前缀, 界面按 agent 类型 + 模式取默认值。
    */
    static const char *sharedWeightPrefix();

    /* 各骨干的固定结构 (模板参数必须编译期确定, 所以不做成运行时成员) */
    static constexpr int MOE_MLP_EXPERTS = 8;
    static constexpr int MOE_MLP_TOPK = 2;
    static constexpr int MOE_TB_EXPERTS = 4;
    static constexpr int MOE_TB_TOPK = 1;
    static constexpr int MOE_TB_HEADS = 15;   /* d_model/15 = 84 维/头; 头越多越便宜 */
    static constexpr int MOE_TB_DFF = 315;    /* = d_model/4, 压住 TB 专家的 FFN 开销 */

    /*
     * 一条 off-policy 经验。
     *
     * 局面**不存 1260 个 float** (那是 5 KB/条), 而是存"被占据的格子"的稀疏列表
     * (一局最多 32 个子, 每格一个 uint16 = plane*90+cell)。取用时再展开成 one-hot,
     * 这样 8000 条经验只有 ~6 MB, 而不是 80 MB。
     */
    struct Transition {
        std::vector<std::uint16_t> cells;      /* 当前局面: plane*CELLS + cell */
        std::vector<std::uint16_t> nextCells;  /* 下一局面 (同一编码) */
        /*
           规则/阶段上下文 (5 个, 与 PPOMCTS 的 plane 14..18 同值) —— 它们**不是**
           "某个格子上的 1", 所以进不了上面的稀疏格列表, 必须单独随身携带:
           否则训练时 expandSparse() 重建出来的状态会丢掉上下文, 编码就退回成
           "裸棋盘" (静默失效)。
        */
        /*
           上下文用**值初始化** `{}` 而不是写死 5 个 0 —— CTX_COUNT 会随表示开关变
           (对齐 5 / 改前 3), 写死个数在另一个表示下就是"初始化项太多"的编译错误。
        */
        float ctx[CTX_COUNT] = {};
        float nextCtx[CTX_COUNT] = {};
        float pi[ACTION_DIM];                  /* 策略目标 (MCTS 访问分布) */
        /*
           合法走法掩码: 8100 位 -> 两个 uint64 (`mask[0]` = 动作 0..63)。
           为什么不是 128 个 bool: 回放池里条目数上万, 128B/条只是为掩码就多占几 MB,
           而位图只要 16B (与改动前 128 槽时用 16B 位图的成本完全一样)。
        */
        std::uint64_t curMask[2] = { 0, 0 };   /* 当前局面合法走法掩码 (策略损失要用) */
        std::uint64_t nextMask[2] = { 0, 0 };  /* 下一局面合法走法掩码 (软备份要用) */
        int action = 0;                        /* 实际走的动作索引 */
        int legalCount = 1;                    /* 当前局面合法走法数 (目标熵用) */
        float reward = 0.0f;
        bool done = false;
        bool hasSearch = false;                /* pi 是否来自 MCTS (否则跳过 AZ 监督项) */
        Transition() { std::fill(pi, pi + ACTION_DIM, 0.0f); }
    };

    /* ----------------------------------------------------------------
     *  AlphaZero 式 PUCT 节点
     *  所有价值都是**当前走棋方视角** (negamax 约定)
     * ---------------------------------------------------------------- */
    struct AZNode {
        int parentID;
        int parentAction;
        Step step;

        int visitCount;
        double totalValue;      /* 累计价值 (当前走棋方视角) */
        double prior;           /* P(s,a) = 未尝试动作上的策略先验 */

        std::vector<int> childIDs;
        std::vector<int> untriedActionIndices;
        std::vector<Step> untriedSteps;
        std::vector<double> untriedPriors;   /* 与 untriedSteps 一一对应 */

        int currentColor;
        int legalCount;
        bool isTerminal;        /* 该节点已终局 (被将杀/困毙/和) */

        AZNode()
            : parentID(-1), parentAction(-1), visitCount(0), totalValue(0.0),
              prior(0.0), currentColor(Stone::COLOR_NONE), legalCount(0),
              isTerminal(false) {}
        AZNode(int pid, int pa, const Step &st, double p, int color, int legal)
            : parentID(pid), parentAction(pa), step(st), visitCount(0),
              totalValue(0.0), prior(p), currentColor(color), legalCount(legal),
              isTerminal(false) {}

        double getQ() const
        {
            return visitCount > 0 ? totalValue / (double)visitCount : 0.0;
        }
    };

    /* ----------------------------------------------------------------
     *  公开成员 (与 PPOMCTSAgent 保持一致的风格)
     * ---------------------------------------------------------------- */
    Chess &chess;

    RL::Net actor;            /* 策略: 1260 -> 骨干 -> 128 logits */
    RL::Net q1, q2;           /* 双 critic: 1260 -> 骨干 -> 128 Q 值 */
    RL::Net q1Target, q2Target;
    RL::GradValue alpha;      /* 温度 α (标量, 自动调节) */

    /*
       [2026-09 独立类] 这里原来还有一个 `Backbone backbone;` 成员 —— 现在没有了:
       本类的骨干在 `makeTrunkLayers()` 里硬编码 (稀疏 MoE(MLP 专家))。
    */
    int expertHidden;         /* MLP 专家的隐层宽度 (本类**用得上**: 专家就是 MlpExpert) */
    float auxLossCoef;        /* 稀疏 MoE 负载均衡辅助损失的系数 (0 = 关掉) */    int hiddenDim;
    float gamma;
    float learningRateActor;
    float learningRateCritic;
    /*
       [2026-09 dev-sacmoetb] 共享骨干的学习率 (只在 trunkMode == Shared 时有意义)。
       默认取构造参数里的 lr —— 与 learningRateActor / learningRateCritic 同一个值,
       也就是"共享这件事本身不改变任何一层的有效步长"。单独留一个成员是因为共享之后
       骨干同时吃策略梯度与双 Q 梯度, 想单独调它必须有一处**显式**的地方可调, 而不是
       靠"顺手改 actor 的学习率"。
    */
    float learningRateTrunk;
    float learningRateAlpha;
    float c_puct;             /* PUCT 探索常数 */
    float azWeight;           /* 策略损失里 AZ 监督项(交叉熵)的权重 */
    float entropyRatio;       /* 目标熵 = entropyRatio * log(合法走法数) */
    int simulations;          /* getBestMove 默认的搜索模拟次数 */
    int batchSize;
    int replaceTargetIter;    /* 每多少次 learn 做一次 Polyak 同步 */
    /*
       ================================================================
       [F1 2026-09] 目标网的 Polyak 步长 —— 本轮实测出来的**主缺陷**
       ================================================================
       症状 (docs/sac_critic_diagnosis_2026_09.md): 一次 20 局的对弈 (~1300~2900 次
       learn) 之后,**目标网的 |Q| 仍然是 0.072~0.097**, 也就是"随机初始化尺度"
       (0.06~0.074) —— 而软备份用的就是这张网 (`qTargetValues`) ⇒ 自举项
       `V(s') = E[min Q_target] + α·H` 里的游戏信息恒为 0, TD 目标退化成
       `y ≈ r − γαH`, critic 学到的只是一个与棋局无关的常数。

       为什么会这样: 原来 tau 是**硬编码 1e-3**、而且每 64 步才同步一次
       ⇒ 20 局只把目标网挪动 `1-(1-1e-3)^(步数/64)` = **2~4%**。
       那个默认值是从"长训练"(DQN 动辄 10^5~10^6 步) 抄来的, 与本工程
       "边下边学、一次会话几千步" 的尺度差两个数量级。

       修法: 把 tau 变成可调成员 (`targetTau`) + 节拍可调 (`replaceTargetIter`),
       **但默认值保持 61a974d 的口径不变** (`tau=1e-3` / 每 64 步):
         * 为什么没有把实测最好的那一档设成默认 (逐臂 200 局 x 4 种子的配对实测见
           docs/sac_critic_diagnosis_2026_09.md §13): 三种"把目标网动起来"的变体都让
           目标网与在线网**对齐** (|Q_target| 0.113 → 1.27~2.01, E[minQ] −0.068 → −1.5~−1.7),
           但**棋力差别不显著** —— 最好的那一档 (硬拷贝每 256 步) 65.0% → 67.3%,
           Fisher **p = 0.43**; 另两档是 65.3% (p = 0.88) 与 62.0% (p = 0.55)。
           **默认值也是结论**: 一个"机制上确实修好了、但棋力未达显著"的改动, 不该
           悄悄改掉所有历史读数所依赖的默认口径。要用就显式开:
             `bench_sac_learn --target-tau=1     --target-iter=256`  (实测最好)
             `bench_sac_learn --target-tau=0.01  --target-iter=8`    (Polyak 变体)
         * 老口径 (`targetTau=1e-3` + `replaceTargetIter=64`) 与 61a974d 逐位相同 ——
           重构只把硬编码的常量变成成员, 已用参考二进制对照验证过。
       **注意**: `SACAZLegacyAgent`(59e5233 行为还原版) 把这个对也**显式钉住**了 ——
       基类默认值的任何改动都不许渗进"行为还原版"(2026-09 就渗进过一次: 见
       docs/ppo_optimization_2026_09.md 之后的那次修正)。
       ================================================================
    */
    float targetTau = 1e-3f;
    std::size_t maxMemorySize;

    /*
        ================================================================
         [P4] 回放池的**多 epoch 复用** (2026-09, 从 RL::PPO 搬过来)
        ================================================================
        `learnBatch()` 原来把 batchSize 条经验**每条只过一遍**。现在每个 epoch 都
        **重新从池里抽** batchSize 条 (与 `RL::PPO::learnFromReplay` 同一做法),
        累积梯度之后仍然只做**一次**优化器更新 (P3 的那条: 优化器占一步的 66%,
        必须摊薄)。于是"刷一批"看到的经验从 batchSize 变成 batchSize×epochs,
        而优化器调用次数不变。

        **不是**"把同一批重复过几遍": 批内权重不变, 所以重复遍的梯度与第一遍逐位
        相同, 而 `RL::Net::RMSProp` 默认 clipGrad=true 会把这个倍数归一掉 ——
        那种写法对更新方向毫无影响, 只是白烧算力。

        与 `PPOMCTSAgent::replayEpochs` 同名同义 (那边默认 2)。本 agent 默认取 1
        (与改版前的更新量逐位一致): 它的学习率不是配着"批放大 2 倍"调的, 而在线
        路径 (exploreAndTrain) 里用户在等这一步, 把它拉到 2 就是把这个等待翻倍。
        自对弈训练路径上想开就设 2。
    */
    int replayEpochs = 1;

    /*
       [诊断/对照] 搜索期软价值的缩放因子 (默认 1.0 = 与改动前逐位一致)。
       只作用于**搜索**的叶子估值 (`searchValueFrom`), 不动学习侧的目标
       (`softValueFrom` 仍然照原样给 critic 的软备份用)。

       为什么需要它: 训练后的 critic 尺度会漂到 |Q|≈13 且几乎无区分度, 而搜索用的是
       PUCT `Q + c_puct·P·√N/(1+n)` —— Q 的量级一旦远大于探索项, 搜索就退化成
       "按 Q 排序", 丢掉先验与访问计数的信息。把 Q 缩回初始量级可以**单独**检验
       "是 critic 的尺度把搜索弄坏了" 这个假设 (见 docs/arena_sac_vs_ppo_report.md §7)。
    */
    float valueScale = 1.0f;

    /* ================================================================
     *  [2026-09 实验轮] 三组"用户提议"的旋钮 + 两个"修好版"
     * ================================================================
     * **全部默认关闭 ⇒ 与改动前逐位相同**；只作用于建这个 agent 的进程/实例。
     * [2026-09 独立类] 这些旋钮现在写在**本类自己的文件**里 —— 所以"只作用于本实例"
     * 这件事从"约定"变成了**结构事实**: TB 专家那一支在 `SACAZMoETbAgent` 里,
     * 两边连文件都不同, 改这里改不到那里。
     *
     * 背景是实测出来的那一条耦合 (见 docs/dev_sacmoetb_strength_2026_09.md §3.1/§6.1):
     *     V(s') = E_π[min Q] + α·H,   y = r − γ·V(s')
     *   实测 α·H ≈ 1.9, 而游戏真实值域只有 ±1（终局）±0.35（满材质）——
     *   熵偏置是值域的**两倍** ⇒ 44% 的 TD 目标被 clampTarget 夹住、
     *   0.029 量级的子力信号被淹没; 而把 α·H 拿掉, critic 又会塌成 0
     *   （没有别的自举信号）。三组提议各自会撞上这条耦合的哪一边，见各成员的注释。
     * ================================================================ */

    /*
       ---- 提议① critic 输出用 tanh: Q = tanh(z) (z 是 Linear 头的原始输出) ----
       实现方式刻意**不改网络结构**（不把输出层换成 `Layer<Tanh>`）：只在产出 Q 的地方
       套 tanh，并在反向时按链式法则乘 `1 − Q²`。这样权重文件格式、结构指纹、
       参数计数全部不变 —— 一个实验旋钮不该顺带改掉可载入性。

       **先算一遍**: TD 目标 `y = r − γ·V(s')`，而 `V(s') = E[minQ] + α·H`。
       在 α·H ≈ 1.9 的现状下 |y| 常在 1~2（44% 超过 clampTarget=2）。
       而 tanh 把 Q 限在 (−1, 1) ⇒ **critic 结构上就拟合不上 y**，
       而且残差最大的那批样本恰好落在 tanh 的饱和区（`1 − Q² → 0`），
       梯度也一起消失。所以**单开这一条应当更差** —— 这正是要实测的预测。
       它与 `clampTarget` 是同一类东西（都是"把 Q 夹进值域"），
       但 `clampTarget` 夹的是**目标**（不改变模型的表达力），tanh 夹的是**输出**（会）。
       **修好版**：tanh 输出 + 把熵项移出目标（`entropyInTarget = 0`）⇒
       `V(s') = E[minQ] ∈ (−1,1)`、`y ∈ (−1.35, 1.35)`，这时 tanh 与值域天然匹配。
    */
    bool criticTanh = false;

    /*
       ---- 提议② 奖励用 tanh 塑形: 即时奖励 r → tanh(gain · r) ----
       gain <= 0 = 关（默认）。

       **先算一遍**: 现在的即时奖励里，吃一个马/炮是 **0.029**
       （REWARD_MATERIAL_COEF=0.1 × 子力），每步代价 −0.001。
       `tanh(0.029) = 0.02899` —— **就是恒等**，gain=1 等于什么都没做。
       要让 tanh 起作用必须先放大（gain 十几以上），而那正好**已经测过**：
       `rewardScale` 5x / 20x 对 critic 读数与得分率都**没有效果**
       （docs/dev_sacmoetb_strength_2026_09.md §6），因为 TD 目标的主项是
       `−γ·V(s')` 而不是 `r`。
       所以这一条要么是恒等、要么是已测过的负结果；仍然实测，因为"算出来的"不算数。
       **注意终局不参与**：终局 ±1 是环境的真值，`tanh(±1) = ±0.7616` 会把它压小，
       只会让棋局信号相对熵偏置更小（更坏）。
    */
    float rewardTanhGain = 0.0f;

    /*
       ---- 提议③ α 用 Gumbel 方法调整 ----
       `sigma > 0` 时，每个训练样本把 α 换成一个被 Gumbel 噪声扰动的随机量：
           α_eff = α · exp( sigma · (G − γ) ),   G = −log(−log U),  γ = 0.5772
       （指数形式保证 α_eff > 0；减掉 γ 让它零均值。）

       **说清楚它不是什么**: `rl/util.hpp` 的 `gumbelSoftmax(x, tau)` 是
       **可微采样器**（给 logits 加 Gumbel 噪声再 softmax，用来从 categorical 里
       可微地采样），它不是一个"控制器"。α 也不是一个 categorical 分布 ——
       它是标量温度。所以"用 gumbelSoftMax 调 α"没有现成的对应物，
       这里实现的是**最接近的读法**：把 α 当成随机量、用 Gumbel 噪声去做探索。

       **先算一遍**: α·H 是**线性**进 TD 目标的，所以给 α 加噪 = **给每个 TD 目标加噪**。
       而当前的问题恰恰是目标被一个常数偏置顶住、排序信息被淹没 ——
       加噪只会让回归更难。所以**单开这一条应当不改善甚至更差**。

       **但它背后的直觉是对的**，而且已经实测：α 的自动调节**没有在工作** ——
       实测 `H均 = 3.39 < H̄ = 0.98·log(38.7) = 3.59` ⇒ α 的梯度 `(H − H̄)` 恒为负
       ⇒ α 被单向推到上界（5.0），`α·H` 因此可以涨到 17。
       **修好版**用不着 Gumbel，而是直接对付那件事：`alphaCeiling`（把上界压下来）
       与 `entropyCenter`（让熵项不再是一个大常数）—— 两个都实现了，见下。
    */
    float alphaGumbelSigma = 0.0f;

    /* 当前样本实际用的 α（`alphaSample >= 0` 时覆盖 `alpha[0]`；Gumbel 噪声走这里） */
    float alphaSample = -1.0f;

    /*
       ---- α 的**上界**（默认 5.0 = 改动前逐位相同）----
       实测 α 会被推到上界，而 `α·H` 是 TD 目标里的主项之一 ⇒ 上界直接决定
       "目标被顶多远"。把它压到 0.1 会让 `α·H ≈ 0.34`（而不是 1.9），
       于是目标不再被夹、但熵项还留着一个**小的、随局面变化**的分量
       —— 这是"1.9 淹没一切"与"0 让 critic 塌掉"之间的中间点，本轮之前没测过。
    */
    float alphaCeiling = 5.0f;

    /*
       ---- 熵项**居中**（默认 false = 改动前逐位相同）----
       `true` 时软价值里的熵项从 `α·H` 改成 `α·(H − log(合法槽位数))`，
       也就是"超出该局面编码能容纳的最大熵的那一部分"。

       为什么这可能同时避开两个极端：
         * `α·H`（现状）：几乎是个常数（H 随局面的变化很小），
           `V(s')` 被顶到 1.07 而 `E[minQ]` 只有 −0.85 ⇒ 目标里 97.7% 与棋局无关；
         * `α·H → 0`（第二轮测过）：自举项只剩 `E[minQ]`，60 局来不及把终局值传回来
           ⇒ critic 塌成 0（Q 间距 0.083）;
         * **居中**：把那个大常数减掉（实测 `H − log n ≈ −0.27` ⇒ 项变成 −0.15 量级），
           但保留 H **随局面变化**的那一部分 ⇒ 目标既不被顶飞、又不恒等于 0。
       搜索侧 (`sparseLeaf`) 与学习侧 (`softValueFrom`) **必须同口径** —— 两边都改了，
       否则"搜索估的"与"训练学的"会是两个游戏（本文件反复强调的那条纪律）。
    */
    bool entropyCenter = false;

    /* 搜索用: valueScale · softValueFrom(...) */
    float searchValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                          const RL::Tensor &q1In, const RL::Tensor &q2In) const
    {
        return valueScale * softValueFrom(pi, mask, q1In, q2In);
    }

    /* ----------------------------------------------------------------
     *  [2026-09 实验轮] 上面那批旋钮的内部支撑
     * ---------------------------------------------------------------- */
    /* 当前实际生效的 α: `alphaSample >= 0` 时用采样值 (Gumbel), 否则用学到的 alpha[0] */
    float effectiveAlpha() const
    {
        return (alphaSample >= 0.0f) ? alphaSample : alpha[0];
    }
    /* 提议① 的单点形式: Q = tanh(z) 或不改 */
    float squashQ(float z) const { return criticTanh ? std::tanh(z) : z; }
    /* 把一整块 Q 就地套上同一个 squash (只作用于模型**输出**, 不改网络结构) */
    void squashQInPlace(RL::Tensor &q) const
    {
        if (!criticTanh) { return; }
        for (std::size_t i = 0; i < q.size(); i++) { q[i] = std::tanh(q[i]); }
    }
    void squashQInPlace(std::vector<float> &q) const
    {
        if (!criticTanh) { return; }
        for (std::size_t i = 0; i < q.size(); i++) { q[i] = std::tanh(q[i]); }
    }
    /*
       提议① 的反向: 给定 dL/dQ, 乘上链式因子 dQ/dz = 1 − Q²。
       **必须在 backward 之前做**: 对 `Layer<Tanh>` 那种"把激活写进层里"的写法,
       squeeze 是层自己算的; 这里刻意不改结构, 所以因子由 agent 自己乘。
    */
    void squashQBackward(RL::Tensor &dq, const RL::Tensor &q) const
    {
        if (!criticTanh) { return; }
        for (std::size_t i = 0; i < dq.size() && i < q.size(); i++) {
            dq[i] *= (1.0f - q[i] * q[i]);
        }
    }

    /*
       ================================================================
       稀疏头推理 (R1 同款, 2026-09 对齐 8100 动作时必须补上)
       ================================================================
       为什么必须补: 动作空间从 128 槽换成 8100 双射之后, 每次叶子估值都要算 8100 个
       Q 值, 而一个局面只有 ~44 个合法着法 —— 实测每步从 55 ms 涨到 **216 ms**
       (同一个 MCTS 预算下)。这不是算法变贵, 是**算了一大堆永远不会被走到的列**。

       `qValuesSparse` / `policySparse` 只算 legalIdx 那几列:
         * 骨干 (forwardTrunk) 照常跑一次;
         * 头用 `iFcLayer::sparseLogits` 只算这几行 (PPO 的 R1 已经验证过它与全量前向
           逐元素一致, test_ppomcts 的 R1 一节);
         * 策略在**合法集上**归一化 (与 maskedSoftmax 的 Z≡1 口径等价, 因为非法列
           本来就恒为 0)。

       返回 false = 走不了稀疏路径 (头不支持 / 下标越界), 调用方**必须**回退全量口径 ——
       与 `RL::PPO` 的做法完全一致 (那条判据在 ilayer.h 里有详细说明, 包括"加了新层类型
       却忘了实现时行为是慢而不是错")。
    */
    bool policySparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                      std::vector<float> &piOut);
    bool qValuesSparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                       std::vector<float> &q1Out, std::vector<float> &q2Out);
    /* legalIdx 上的软价值: E_π[min Q − α log π] (π 是合法集上的策略, 由本函数内部算) */
    bool softValueSparse(const RL::Tensor &state, const std::vector<int> &legalIdx,
                         double &valueOut);
    /*
       ================================================================
        [2026-09 dev-sacmoetb] 一次叶子求值的**完整稀疏路径** (搜索热路径专用)
       ================================================================
       上面的 policySparse / qValuesSparse / softValueSparse 是**公开 API**(测试与诊断
       单独调它们), 而搜索里每次展开都要"π + 双 Q"三样一起要 —— 分开调就会把骨干
       跑两遍 (policySparse 一遍 + qValuesSparse 一遍)。

       本函数把三样一次算完:
          * 非共享口径: 语义与"policySparse + qValuesSparse"逐元素相同, 但**只跑一遍
            公共部分**没有可省的 (三张网), 所以这里只是把三件事写在一处;
          * **共享口径: 骨干只前向一次** —— 这正是 Shared 模式在搜索侧的全部收益来源。
        `pi` / `q1Out` / `q2Out` 的长度都等于 legalIdx.size()。
       ================================================================
    */
    bool sparseLeaf(const RL::Tensor &state, const std::vector<int> &legalIdx,
                    std::vector<float> &pi, std::vector<float> &q1Out,
                    std::vector<float> &q2Out);

    /*
       [消融] 搜索的叶子估值是否走稀疏头 (默认 true)。
       为什么留这个开关 (2026-09 回归排查): 动作空间从 128 变成 8100 时, 全量叶子估值
       要算 8100 列 Q (实测 216 ms/步), 所以加了稀疏路径。但**在改前表示 (128 槽) 下**
       全量只需要算 128 列, 稀疏路径省不了什么, 反而多走了 `forwardTrunk` + 稀疏头这条
       与训练路径不同的代码 —— 于是两个口径之间任何一点差异都会被放大成棋力差。
       关掉它就退回"与改动前逐位相同"的求值方式。
    */
    bool sparseLeafEval = true;

    /*
       ================================================================
       critic 值域约束 (2026-09 新增, 修的是实测到的**发散**)
       ================================================================
       实测 (docs/arena_sac_vs_ppo_report.md §5.2, 工具 --mode=probe):

           权重         |Q| 均值      Q 区间
           随机初始化     0.063      [-0.14, 0.10]
           自对弈 40 局   4.146      [-4.41, -3.63]
           自对弈 150 局 13.381      [-14.06, -11.94]

       单调发散, 而且**所有动作一起变负** ⇒ 不是在学更好的排序, 而是在整体漂移。
       后果不是"数值不好看", 而是**搜索坏掉**: 叶子估值就是 softValueFrom(...),
       而 PUCT 是 `Q + c_puct·P·√N/(1+n)` —— Q 的量级一旦压过探索项, 先验与访问
       计数提供的信息被淹没, 于是对 MCTS 的得分率从 73.3% 掉到 32.5%。

       为什么会发散: 即时奖励上界只有 0.35 (REWARD_MATERIAL_COEF × 一方满子),
       终局 ±1, 所以**真实 Q 必然落在 [-1.5, 1.5] 量级内**; 但软备份
       `y = r − γ(1−done)·V(s')` 把 V 反复回代, 而优化器 (RMSProp + clipGrad,
       按向量归一) 没有任何把 V 拉回该区间的机制 ⇒ 正反馈发散。

       修法 (两条, 都要, 因为它们的**性质不同**):
         1. `clampTarget` —— **结构性**: 目标 y 直接夹到 [−2, 2]。这不是启发式,
            而是把"这个游戏的 Q 值域"写进目标 (与 DQN 的 reward clipping 同一思路,
            但更弱: 只夹目标、不夹奖励, 所以不改变 MDP 的排序, 只挡住发散);
         2. `huberDelta` —— **鲁棒性**: |err| > δ 时损失从平方变成线性 (Huber),
            单个离群样本不会再把 32 条样本的批平均方向带跑。
        `clampTarget <= 0` 可以把第 1 条关掉 (A/B 对照用, 用来证明"发散就是主因")。
    */
    float clampTarget = 2.0f;    /* <=0 表示不夹 (对照用) */
    float huberDelta = 1.0f;     /* <=0 表示用纯 MSE (对照用) */

    /*
       ================================================================
       [2026-09 ①] 熵项的**去处**与目标熵的**分母** (两个实验开关, 默认 = 现状)
       ================================================================
       背景 (本轮要验证的机制假设, 见 docs/session_2026_09_sac.md §3-①):
       软价值是 `softValueFrom = E_π[min Q] + α·H(π)` (H = 策略熵), 而 TD 目标是
           y = r − γ(1−done)·V(s')
       于是 **熵项以 −γ·α·H(s') 的形式直接进入 critic 的回归目标** —— 它与棋局无关,
       是一个常数偏置。若 Q 在各动作上近似相等 (= q), 不动点解就是
           q = (r − γ·α·H) / (1 + γ)
       即"α·H 有多大, critic 就被推到多远": αH ≳ 4 时 q 已经越过 clampTarget=2 的边界,
       整个 critic 退化成"所有动作都等于钳位边界上的同一个常数"(实测 |Q| 均值 2.008 /
       最大 2.162, 而钳位是 2.0 —— 贴着边界正是这个形状); αH → 0 时 q → 0
       (实测 0.5/5e-3 档 |Q| = 0.035)。**两种都是没有排序信息**, 只是被推到的常数不同。

       开关 (都是"关掉才变", 默认口径与改动前逐位一致):
         entropyInTarget = 1.0 : 现状 —— 熵项进 V(s'), 因此既进搜索叶子也进 critic 目标;
                          = 0.0 : 熵项**不进**软价值 (只留在策略损失里) —— 用来单独检验
                                  "是熵项把 critic 顶走的" 这条因果, 而不是只看相关性。
         entropySlotsAsLegal = false : 现状 —— 目标熵 H̄ = 熵比·log(合法**着法**数);
                              = true  : H̄ = 熵比·log(合法**槽位**数)。
                                 为什么这是个真差别: 128 槽哈希有碰撞 (实测同局面平均挤掉
                                 5.16 个着法), 而 π 只分布在**槽位**上, 所以
                                 H ≤ log(槽位数) < log(着法数) —— H̄ 按着法数算时**可能永远
                                 达不到**, 而 α 的梯度恰好是 (H − H̄): 达不到 ⇒ 单向推走
                                 (推到 5.0 上界或 0.02 下界, 两种都坏)。
       */
    float entropyInTarget = 1.0f;
    bool entropySlotsAsLegal = false;

    /*
       ================================================================
       [2026-09 ①] 训练中的 critic/α 诊断累计量 (只读; 不参与任何计算)
       ================================================================
       为什么需要它: 原有读数 (m_maxAbsTarget / m_maxAbsTdErr / 训练后的 |Q| 快照) 只能看到
       "最后一次"和"最后的状态", 而本轮要回答的是**过程**里的三个问题:
         1. 目标 y 有多少比例被 clampTarget 夹住? 夹住 = 那部分样本的目标是个常数,
            对 critic 的**排序**学习毫无贡献 (而搜索要的正是排序);
         2. V(s') 的量级是不是被 α·H 撑起来的? (vQSum 与 vEntSum 的均值一比就见分晓);
         3. α 的轨迹长什么样? (alphaFirst/alphaLast, 工具再按局打印差值)
       另外两条:
         hSum / hBarSum / hBarSlotsSum —— α 的梯度是 (H − H̄), 所以"目标熵可不可达"直接
           决定 α 往哪边跑 (配上 hBelowHbar / hBelowHbarSlots 两个计数就是直接证据);
         qSpreadSum —— min(Q1,Q2) 在合法槽位上的**标准差**: 这是 critic 真正能给搜索的
           排序信号尺度。Q 被推成常数时它会塌到 0, 于是 PUCT 里 Q 项恒等, 选择完全由
           先验与访问计数决定 (搜索与 critic 脱钩)。
       全部是累加量 (自 agent 构造起), 调用方读两次取差值即可得到"这一段的分布"。
       */
    struct TrainDiag {
        long long n = 0;               /* 样本数 */
        long long clamped = 0;         /* 夹前 |y| > clampTarget 的条数 */
        long long hBelowHbar = 0;      /* H < H̄ (把 α 推大) 的条数 */
        long long hBelowHbarSlots = 0; /* H < 熵比·log(槽位数) 的条数 (反事实) */
        double yPreAbsSum = 0.0, yPreSum = 0.0, yPreAbsMax = 0.0;
        double vNextSum = 0.0;         /* V(s') = E_π[min Q] + α·H */
        double vQSum = 0.0;            /* E_π[min Q(s')] */
        double vEntSum = 0.0;          /* α·H(s') —— 假设里那个偏置项 */
        double hSum = 0.0, hBarSum = 0.0, hBarSlotsSum = 0.0;
        double slotsSum = 0.0, legalSum = 0.0;
        double qSpreadSum = 0.0, qAbsMeanSum = 0.0;
        double alphaFirst = -1.0, alphaLast = -1.0;
        /*
           [2026-09 动态奖励分配] **终局通道到底有没有样本** —— 这两个计数器是
           "后期重杀将加权"这一类旋钮的生死判据:

             doneSamples     : 样本里 done=true 的条数 (一局的最后一步);
             decisiveSamples : 其中**分胜负**的条数 (|reward| > 0; 和棋/截断记 0)。

           为什么必须单独数: 终局倍率 (`rewardShape=3` 的 `mateWeightMul`) 只作用在
           `terminalReward()` 上, 而它对和棋恒返回 0 ⇒ **只有 decisiveSamples 那一类
           样本才携带"杀将奖励"**。实测 (`--train-games=2 --train-plies=120`): 把
           `mateBoost` 从 0 开到 10, 存下来的权重**逐字节相同**; 而同一口径下把
           `matBoost` 从 0 开到 0.5 权重立刻不同 —— 也就是说自对弈里
           decisiveSamples ≈ 0 (一局大多走到手数上限或重复判和), **这个旋钮在训练里
           是"死的"而不是"没用的"**。于是"把杀将信号变重"的下一步只剩两条: 让自对弈
           真的打到终局 (改课程/改截断口径), 或给价值函数任务外的直接监督 (蒸馏)。
           见 docs/sacmoetb_pos_reward_2026_09.md 的 §10。
        */
        long long doneSamples = 0;
        long long decisiveSamples = 0;
    };
    TrainDiag trainDiag;
    const TrainDiag &getTrainDiag() const { return trainDiag; }

    /*
       [诊断/消融] 即时奖励的整体缩放 (默认 1.0 = 与改动前逐位一致)。
       为什么需要它: 奖励 = 材质(0.1 x 子力) + 每步代价, 而终局是 ±1 —— 也就是说
       **即时奖励只占终局奖励的百分之几**。把它整体放大/缩小可以单独检验
       "奖励塑形的尺度是不是在帮忙"。只缩放即时项, **不动终局 ±1**
       (终局是环境的真值, 缩放它等于换一个游戏)。
    */
    float rewardScale = 1.0f;

    /*
       ================================================================
       终局/材质塑形方案 (2026-09, 用户提议的实验旋钮)
       ================================================================
       背景 (用户实测): "59e5233 还原版" 对 MCTS 的**奖励累计**能到对手的 2 倍, 但很多局
       在 300 手判和 —— 吃着子却赢不了, 怀疑是奖励结构把"吃子"抬得太高、把"将死"压得
       太低。这个旋钮把它变成可测量的问题:

         rewardShape = 0 (默认) : 现状。即时 = 材质 x REWARD_MATERIAL_COEF + 每步代价;
                                  终局 = ±1 (REWARD_TERMINAL)
         rewardShape = 1        : **去掉材质**。即时只留每步代价 (-0.001) -> 终局 ±1 成为
                                  唯一的学习信号 (纯胜负/和棋)
         rewardShape = 2        : **终局放大**: |终局| x (1 + 败方剩余材质/满材质),
                                  取值 [1, 2) —— "对方兵力越完整就被将死 = 越值钱"
                                  (即快杀 > 磨死)

       用户原话是"环境奖励乘以 (1 + 最终棋子数/总棋子数)"。两点必须说清楚 (见 docs):
         * "**最终**棋子数"在**即时奖励**上不可实现 —— 在线 TD 里落子时并不知道结局;
         * 所以这个量只能落在**终局**上, 那是唯一"结局已确定"的时刻。本旋钮就是这么做
           的 (方案 2), 而且把它同时用于**搜索叶子**与**训练目标**, 两者不许分叉。
         * "整个 episode 的奖励乘一个系数"这种写法在本算法里**做不到**: SAC 是
           off-policy 回放, 一条经验可能在局终之前就被学过好几次, 事后改它的 reward
           等于改历史 (对已更新过的权重无效, 对没学到的又是另一套口径)。要做这种缩放
           只能换成蒙特卡洛回报 —— 那是换个算法, 不是调奖励。

       **只有一个出口**: 终局值有**三**个产生点 (搜索叶子 `terminalValue` / 自对弈
       `resultValue` / rollout 的 `outcomeForMover`), 三者口径不一致的话, 搜索估的与
       训练学的就是两个游戏 —— 所以三个点全部走 `terminalReward()`, 由它一家说了算。
       (`agentrollout.hpp` 用 SFINAE 检测这个可选成员: 有就用, 没有就退回共享的
       `outcomeForMover`, 于是其它 agent 一行都不用改。)
    */
    int rewardShape = 0;

    /*
       ================================================================
       rewardShape = 3: **按局面动态分配"吃子 / 杀将"两个权重** (2026-09 用户提议)
       ================================================================
       用户原话 (两轮):

         ① "前期应该加强吃棋子的奖励权重, 后期应该重杀将奖励, 或者说这两者都不应忽视奖励";
         ② "分阶段太过思维定势, 因为局势是反复变化的, 是否可以参考棋子的数量和价值来
            评估局面动态调整杀棋杀将的奖励比例"。

       ② 是对 ① 的**修正**, 而且修得对 —— 真实的分歧点在这里:

         * 第一版把"前期/后期"做成了**一个单向的阶段钟** (剩余子力越多 = 越前期),
           它是**单调**的: 一局棋从满盘走到光将, 阶段只往一个方向走。而"局势"不是
           单调的: 落后一方吃掉一个车, 局势当场翻面; 双方互兑之后子力骤减, 局面立刻
           进入残局。**一个单调坐标表示不了会反复的局势。**
         * 所以这一版把 ① 的那个标量换成**局面评估** e (见 `mateProximity()`):
           它同时读**棋子的数量**、**棋子的价值**、以及**双方的子力差**, 于是
              - 子力被吃光 -> e 上升 (进入残局, 杀棋临近);
              - 一方吃掉对方的子 -> |子力差| 变大 -> e 上升;
              - 落后一方吃回来 -> |子力差| 变小 -> e **下降** (局势反复, 权重跟着回摆)。
           它是**局面的函数**, 不是"第几手"的函数 —— 同一手在两种局面下拿到的权重不同,
           这才是"动态"。

       ---- 两个权重与"比例" ----
         m_mat(e) = 1 + matRewardBoost  x (1 − e)     材质系数倍数 ∈ [1, 1+matBoost]
         m_mate(e)= 1 + mateRewardBoost x e           终局值倍数   ∈ [1, 1+mateBoost]

       默认 matBoost = mateBoost = 0.5 时两条**恒有 m_mat + m_mate = 2.5** —— 这才是
       用户说的"比例": 一个**固定的奖励预算**, 按局面在"吃子"与"杀将"之间动态分配。
       (e=0 开局均势: 吃子 1.5x / 杀将 1.0x; e=1 大势已定: 吃子 1.0x / 杀将 1.5x。)
       两次都**永不归零**: 材质倍数最小是 1 (基准 0.1), 终局倍数最小是 1 (±1 真值),
       这就是"两者都不应忽视"的代码含义。

       ---- 局面评估 e 的三个因子 (见 `mateScoreMode`) ----
         f_值 = 1 − 双方剩余非将子力 / 7.0            (稀疏度: 价值口径, 3.5/方)
         f_数 = 1 − 双方剩余非将棋子**个数** / 30      (稀疏度: 数量口径, 15 个/方)
         f_势 = |红方剩余子力 − 黑方剩余子力| / (红方 + 黑方)   (定局度: **相对**子力差)

         mode 0 (默认) e = clip((f_值 + f_数 + f_势)/3, 0, 1)  <- "数量和價值"都用
         mode 1         e = f_值                               <- 只用价值 (≈第一版的阶段钟)
         mode 2         e = f_数                               <- 只用数量
         mode 3         e = clip((f_值 + f_数)/2, 0, 1)        <- 数量 + 价值, 不看优势
         mode 4         e = f_势                               <- 只看优势 (定局度)

       为什么"个数"是独立的一维 (而不是价值的附庸): 同样是 1.0 的价值,
       "两个车"与"十个兵"对**杀棋**的难度天差地别 —— 子少则棋盘开阔、将好露头, 杀棋近;
       子多则互相牵扯, 杀棋远。只看价值会把这两者算成同一个局面。

       为什么"优势"用**相对差**而不是绝对差: 绝对差在残局里会被"总量本来就小"放大
       (只剩两个兵 vs 一个兵也会得出一个"看起来很大"的差); 相对差是尺度无关的 ——
       0 = 完全均势, 1 = 一边被吃光。**局势反复就体现在这一维上**:
       领先方被吃回一个车, f_势 立刻掉下来, e 跟着回摆。

       ---- 三条必须钉住的约束 ----
       1. **量纲不变量仍在**: "吃光对方全部子力"必须仍然**小于**赢一盘棋:
          0.35 x (1 + matRewardBoost) < 1.0  ⇒  matRewardBoost < 1.857。
          默认 0.5 -> 0.525 < 1.0。超过这条线是**换一个游戏**, 允许 CLI 传进去做消融。
       2. **只放大, 不改符号与排序**: 两个倍数恒 ≥ 1, 终局倍率对**双方同一个数**
          (零和对称: +m / −m), 和棋恒 0 ⇒ "赢 > 和 > 输"一位不变。
       3. **一个 e 贯穿即时与终局**: 即时奖励 (computeReward) 与终局 (`terminalReward()`
          这个唯一出口 —— 搜索叶子 / 自对弈 / rollout 全走它) 用的是**同一个
          `mateProximity()`**, 否则"搜索估的"与"训练学的"就是两个游戏。

       ---- 已知的代价 (写在明处) ----
       * **终局加权与 γ=0.99 的折现方向相反**。把 mateRewardBoost 开到 b 之后,
         "晚 Δ 手再杀"的折现价值 `0.99^Δ x (1+b)` 会在 `Δ < 100·ln(1+b)` 手之内
         **超过立刻杀** —— b=0.5 是 40 手, b=1.0 是 69 手。也就是说这个旋钮在
         **奖励"磨到残局再动手"**, 而 state 评估里"越磨 e 越大"会把它喂大。
         默认取 0.5 (窗口 40 手) 而不是更大值, 就是为了不让这条通道变成主策略;
         实测里 b=0.5 / 1.0 两档都测 (见 docs)。
       * 奖励结构随局面变化 = 价值函数要拟合的靶子本身在动。这是 §6 那条实测结论
         ("奖励尺度不是 critic 的约束, `α·H` 才是") 的加强版, 不是新问题。

       一句话: 关心的不是"第几手", 而是"这盘棋还剩多少子、差多少子 —— 因此现在
       该惦记吃子还是该惦记杀将"。

       实测口径与结果见 `docs/sacmoetb_pos_reward_2026_09.md`
       (工具: `bench_sacmoetb_train --reward-shape=3 --pos-mat-boost=.. --pos-mate-boost=..`)。
    */
    int mateScoreMode = 0;          /* 局面评估的因子组合 (0 数量+价值+优势 / 1 价值 / 2 数量 / 3 数量+价值 / 4 只看优势) */
    float matRewardBoost = 0.5f;    /* 吃子权重上限: e=0 (均势满盘) 时材质系数 x (1+boost) */
    float mateRewardBoost = 0.5f;   /* 杀将权重上限: e=1 (残局/大势已定) 时终局值 x (1+boost) */

    /*
       局面评估 e ∈ [0,1]: 0 = 满盘均势 (该惦记吃子), 1 = 残局/大势已定 (该惦记杀将)。
       `pendingCapture` = 本步正要吃的那一步 (可为 nullptr): 它只用来把**被吃子**算回
       "走子前"的局面 —— 于是"走前求奖励"与"走后求奖励"给出**同一个 e**
       (工程里两个调用点都存在, 这个差异不该变成两个不同的奖励)。见 test_sacaz [15b]。
    */
    double mateProximity(const Step *pendingCapture) const;
    /* 材质系数的动态倍数 m_mat(e) (rewardShape != 3 或 boost<=0 时恒为 1.0) */
    float materialWeightMul(const Step *pendingCapture) const;
    /* 终局值的动态倍数 m_mate(e) (取**当前**局面 = 终局局面的 e) */
    float mateWeightMul() const;

    /*
       终局值 —— 塑形方案的**唯一**出口。
       `perspective` = 视角方 (语义与 chess.h 的 `outcomeForMover` 一致: 返回 +1 表示
       这个颜色赢了); 三个调用点传的都是"该视角的颜色": rollout 里是**刚走子**的一方
       (他可能就是赢家), 搜索里是**轮到走**的一方 (终局节点上他是被将死的那一方)。
    */
    float terminalReward(int chessResult, int perspective) const;

    /*
       ================================================================
        [2026-09 新] 每次真实决策都从**自己的搜索**学一次 (AlphaZero 的核心信号)
       ================================================================
       用户实测报的现象: "界面对弈里只有勾上 rollout(探索+预训练) 才有损失曲线、
       对弈过程才会训练"。代码上确实如此 ——

         * 界面的 per-move 学习**完全**由 ChessBoard::preTrainThenDecide 驱动, 而那个
           函数被 `preTrainCheck` 直接短路 (`m_preTrainEnabled` 关掉就直接返回);
         * 而 SAC 在整局里唯一的学习来源就是 exploreAndTrain 的 rollout, 那条路径写进
           去的样本是 `hasSearch=false` —— 也就是**搜索算出来的 π_MCTS 被丢掉了**。
           AlphaZero 的监督项在对弈中一次都没用上 (只有后台训练 trainSelfPlay 用它)。

       结果: 关掉 rollout 就完全不学; 开着也只是"从自己策略滚 64 步 + 更新一次",
       而那个 256 次模拟的搜索成果直接扔掉。

       本开关把这一步补上: **选完真实走法之后**, 把 (s, π_MCTS, a, r, s', done) 存进
       回放池 (hasSearch=true), 并做一次 learnBatch。于是
         * 对弈中**总会**训练 (不再依赖 rollout 勾选框);
         * 策略终于收到自己的搜索结果当监督 —— 那正是它比自身策略更准的地方;
         * 代价只有一次棋盘试走/回退 + 一次已经存在的 learnBatch (搜索本身早已付过)。
       `learnFromSearch = false` 恢复改动前的行为 (只在 rollout / 自对弈里学)。
       **59e5233 还原版没有这个成员**: 那一支是独立类 SACAZLegacyAgent, 它的
       selectMove 是**只读搜索** (59e5233 没有这条路径) —— 那边是"连名字都没有",
       不是"默认关着" (见 src/sacazlegacyagent.h)。
    */
    bool learnFromSearch = true;

    /*
       把这一步的真实决策存成一条 AlphaZero 样本 (hasSearch=true) 并按需更新一次。
       返回 true = 做过一次 learnBatch (界面据此上报损失曲线的点)。
       `piVisit` = 根节点的访问分布 (visitDistribution), `actionIdx` = 树里那一步的下标
       (父节点的 parentAction, 与 π 的下标同一套编码)。
    */
    bool learnFromSearchStep(int color, int actionIdx, const Step &step,
                             const RL::Tensor &piVisit);

    /*
       诊断 (只读上界面/自检, 不参与任何计算): 最近一次 learnBatch 里
         m_maxAbsTarget : 夹过之后 |TD 目标| 的最大值 —— 它应当稳定在 ~1 附近
                          (终局 ±1 + 即时奖励 0.35 的量级)。持续上涨 = 又在发散。
         m_maxAbsTdErr  : |Q(s,a) − y| 的最大值 —— 发散时这个数会先于 loss 爆掉。
       这两个读数是为了让"critic 是否发散"**当场可看**, 而不是只能靠离线探针
       (--mode=probe) 事后发现。见 docs/arena_sac_vs_ppo_report.md §5.2。
    */
    double m_maxAbsTarget = 0.0;
    double m_maxAbsTdErr = 0.0;
    double getMaxAbsTarget() const { return m_maxAbsTarget; }
    double getMaxAbsTdErr() const { return m_maxAbsTdErr; }

    /*
       [R2] **本 agent 的 R2 口径是结构性的, 不是一个开关**: 掩码 softmax
       (maskedSoftmax) 决定了非法动作 π ≡ 0, 而策略梯度的 g 在非法列上恒为 0 + 掩码
       雅可比也给出 dz ≡ 0, 于是非法列的头部权重梯度**恰好为 0** —— 与 RL::PPO 的
       R2 (训练侧只算合法列) 同一口径。证据钉在 test_sacaz 的 R2 一节里。
       这里不设 `maskedTrainHead` 开关: 对局面的合法性做"可关掉的掩码"等于允许
       非法着法, 那不是 A/B, 那是另一个 (错的) 算法。
    */

    /* 统计 */
    int totalEpisodes;
    int totalWins[2];
    int learnSteps;
    long long m_leafEvals;
    float m_lastLoss = std::numeric_limits<float>::quiet_NaN();  /* 见 getLastTrainLoss */
    /* 最近一次 learnBatch 一共攒了多少条样本 (P4 诊断: 应为 batchSize×epochs) */
    int m_lastBatchSamples = 0;
    std::vector<AZNode> nodes;

    /* 回放缓冲 */
    std::deque<Transition> memories;

private:
    RL::Tensor m_stateBuf;    /* 复用, 避免每次 encode 都分配 1260 个 float */
    RL::Tensor m_logits;      /* actor 输出副本 */
    RL::Tensor m_q1, m_q2;    /* critic 输出副本 */
    /* pick 回调算出来的掩码, 供紧接其后的 onTrans 复用 (同一步之内有效) */
    int m_pendingLegalCount = 1;
    std::uint64_t m_pendingMask[2] = { 0, 0 };

public:
    /* ----------------------------------------------------------------
     *  编码 / 动作辅助 (rolloutFromCurrent 需要 encodeState /
     *  getLegalActions / computeReward 这三个签名)
     * ---------------------------------------------------------------- */
    /* 规范视角编码: 用 chess.sideToMove 决定视角 (MCTS 期间棋盘是同步的) */
    void encodeState(RL::Tensor &state);
    void encodeStateFor(int color, RL::Tensor &state);
    /* 稀疏编码 (存回放用) / 展开 / 稠密->稀疏 */
    void encodeSparse(int color, std::vector<std::uint16_t> &cells) const;
    static void expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state);
    static void denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells);
    /* 规则/阶段上下文的读写 (放在 plane 14..18; 与 PPOMCTS 同值同序) */
    static void contextOf(Chess &c, int color, float out[CTX_COUNT]);
    static void writeContext(RL::Tensor &state, const float ctx[CTX_COUNT]);
    static void readContext(const RL::Tensor &state, float out[CTX_COUNT]);

    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);
    /*
       动作索引是 `Step` + 走棋方的**纯函数**, 所以是 const。
       为什么要带 color: 与状态一样要按**规范视角**镜像 (黑方 x->9-x), 红黑双方的
       "同一步棋"才会共享同一个动作槽位 —— 这是"一套权重服务双方"的另一半
       (状态镜像在本文件, 动作镜像必须同一口径; PPOMCTS 的 stepToActionIdx 同签名)。
    */
    int stepToActionIdx(const Step &s, int color) const;
    float computeReward(const Step &s, int color);

    /*
       [④] 学习口径的奖励 —— 界面奖励曲线现在取的是**这一份** (见 AgentBase 的说明)。
       终局那一条必须覆写成 terminalReward(): 塑形开着 (rewardShape=2) 时它是
       ±(1+败方剩余材质/3.5), 而引擎口径永远是 ±1 —— 覆写它, 界面上显示的游戏才与它
       真正学的是同一个。
    */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    float learningTerminalReward(int chessResult, int perspective) const override
    {
        return terminalReward(chessResult, perspective);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }

    /* ----------------------------------------------------------------
     *  掩码 softmax 与它的反向
     * ---------------------------------------------------------------- */
    /* π(a) = m_a·exp(z_a) / Σ_b m_b·exp(z_b)  (数值稳定) */
    static void maskedSoftmax(const RL::Tensor &logits, const RL::Tensor &mask,
                              RL::Tensor &pi);
    /*
     * 掩码 softmax 的雅可比: 给定 dL/dπ = g, 返回 dL/dz。
     *   dπ_a/dz_c = π_a(δ_ac − π_c)     (c 合法)
     *   dL/dz_c   = π_c·( g_c − Σ_a g_a·π_a )
     * 形式与 `Layer<Softmax>` 的 jacobian_transpose_mul 相同, 但用的是**掩码后**
     * 的 π —— 所以不能对 Softmax 层的输出打补丁了事, 必须自己算。
     */
    static void maskedSoftmaxBackward(const RL::Tensor &pi, const RL::Tensor &g,
                                      RL::Tensor &dz);
    /* 合法掩码 <-> 位图 (8100 位 = 2 x uint64; `bits[0]` 是动作 0..63) */
    static void maskToBits(const RL::Tensor &mask, std::uint64_t bits[2]);
    static void bitsToMask(const std::uint64_t bits[2], RL::Tensor &mask);

    /* ----------------------------------------------------------------
     *  网络前向 / 软价值
     * ---------------------------------------------------------------- */
    /* 造一个完整网络 (withGrad=false 用于目标网)。骨干已硬编码在本类里, 无参数可选。 */
    RL::Net buildNet(bool withGrad) const;
    /*
       共享骨干的建网 (2026-09 dev-sacmoetb): 造出**一层不重复**的骨干层与三个头层,
       再把它们**按共享指针**组装成 actor / q1 / q2 / trunk / 目标网。
       `buildNet` 与 `buildShare` 必须走**同一套**层构造代码 (makeTrunkLayers /
       makeHeadLayer), 否则"两种模式除了共享关系之外逐位相同"这条前提就不成立了。
    */
    void buildSharedNets();
    void makeTrunkLayers(RL::Net::Layers &out, bool withGrad) const;
    void makeHeadLayer(RL::Net::Layers &out, bool withGrad) const;
    /*
       [2026-09 独立类] 这里原来还有 `makeTbExpertMoe()` 与四个 TB 头数读数
       (`tbHeadsRequested/Used/Dim/Allocated`) —— 它们**只属于 TB 那一支**, 现在
       在 `SACAZMoETbAgent` 里。本类没有 TransformerBlock 专家, 所以结构上不存在
       "头数被降级"这件事, 自检报告也不会再出现那一节。
    */
    /* 共享口径下的"唯一参数量"(骨干只算一次); 非共享口径 == actor+q1 (同构) */
    long long uniqueParamCount() const;
    /* 稀疏 MoE 的坍缩诊断: actor 的第一个稀疏 MoE 层的使用计数 */
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();
    int moeExpertCount() const;
    int moeTopK() const;
    /* 掩码策略 π(·|s) */
    void policy(const RL::Tensor &state, const RL::Tensor &mask, RL::Tensor &pi);
    /* 在线双 Q (搜索与策略损失用) */
    void qValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out);
    /* 目标双 Q (软备份用) */
    void qTargetValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out);
    /*
     * 软价值 V(s) = Σ_{a 合法} π(a)·( min_i Q_i(s,a) − α·log π(a) )
     *             = E_π[min Q] + α·H(π)          (H = −E_π[log π] ≥ 0)
     * **熵项是加上的**: 因为 −log π ≥ 0, 这正是 SAC 的软价值定义 (最大熵目标里
     * 熵是奖励的一部分)。所以 α 越大, "自己还有多种好选择"的局面估值越高。
     * α = 0 时退化成"策略期望 Q"。
     * 纯函数形式: Q 由调用方给 —— 搜索用在线网, 学习时的备份用目标网。
     */
    float softValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                        const RL::Tensor &q1In, const RL::Tensor &q2In) const;

    /* ----------------------------------------------------------------
     *  MCTS
     * ---------------------------------------------------------------- */
    double getPUCT(int childID, int parentVisits) const;
    /* 终局判定: 终局则给出 ±1/0 (color 视角) 并返回 true */
    bool terminalValue(int color, double &value) const;
    /* 访问分布: π_MCTS(a) ∝ N(a) */
    void visitDistribution(int rootID, RL::Tensor &pi);

    /* ----------------------------------------------------------------
     *  决策 / 训练
     * ---------------------------------------------------------------- */
    /* piOut != nullptr 时回填根节点的访问分布 (策略目标) */
    Step selectMove(int color, int simulations_, float temp = 0.0f,
                    RL::Tensor *piOut = nullptr);
    /* 从回放缓冲采一个 mini-batch 更新一次 (critic / actor / α), 返回 critic loss */
    /*
       epochs <= 0 时用成员 `replayEpochs` (P4: 同一批过几遍)。
       批大小**必须**按池里的实际条数夹一下: 池 < batchSize 时本函数故意直接返回 0。
    */
    float learnBatch(int batchSize_, int epochs = 0);
    /*
       [MoE] 只清"本批"的门控统计 (usageBatch / probSumBatch / xSum / batchForwardCount),
       保留生命周期累计 (坍缩诊断用的 usageTotal)。

       为什么必须: 辅助损失的批均值是按"自上次 addAuxGradient 以来所有 forward"算的,
       而 forward 有两个来源 —— learnBatch 里的训练前向, 和**搜索 (MCTS 叶子估值) 的
       推理前向**。不复位的话, 一次 learnBatch 的辅助损失会被它之前那一整局的模拟
       (几十到几百次叶子估值) 稀释/污染。PPO 那边 (RL::PPO::resetMoeBatchStats)
       是同一条机制。
    */
    void resetMoeBatchStats();
    /*
       把 getResult 的返回值换算成 color 视角的终局值; 未结束返回 false。
       **不再是 static**: 它要经过 `terminalReward()` (塑形方案的唯一出口), 而那个
       需要读棋盘 (rewardShape=2 要看败方剩余兵力)。见 rewardShape 的说明。
    */
    bool resultValue(int result, int color, float &out);
    void trainSelfPlay(int episodes, int simulations_, int maxMoves = 200,
                       bool verbose = true, float tempRoot = 1.0f,
                       float tempFinal = 0.25f, int learnEveryMoves = 4);
    void warmupFromCurrent(int episodes = 2, int simulations_ = 40,
                           int maxMoves = 60);

    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* ----------------------------------------------------------------
     *  AgentBase
     * ---------------------------------------------------------------- */
    SACAZMoEMlpAgent(Chess &chess_,
               int hiddenDim_ = 64,
               float gamma_ = 0.99f,
               float lr = 0.001f,
               float cpuct = 1.5f,
               int expertHidden_ = 64,
               float auxLossCoef_ = 0.1f,
               /*
                  [2026-09 独立类] 构造签名里**去掉了 `Backbone` 与 `tbHonorHeads_`**:
                  本类只有一种骨干 (稀疏 MoE(MLP 专家)), 而 MLP 专家里没有
                  MultiHeadAttention, 所以"头数口径"这个开关在本类里**不存在**
                  (它在 `SACAZMoETbAgent` 里)。剩下的默认值 = **改动前的口径**:
                    * trunkMode_ 默认 Separate —— 既有调用点构造出来的东西与改动前**逐位相同**。
               */
               TrunkMode trunkMode_ = TrunkMode::Separate);
    ~SACAZMoEMlpAgent() = default;

    Step getBestMove(int color) override;
    std::string getName() const override;
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 契约与口径见 aiagent.h 的 selfCheckReport
     *
     *  为什么这个类特别需要它: **[2026-09 独立类] 本类现在只对应一个界面 agent 类型**
     *  (AGENT_SACAZ_MOE_MLP, GUI 标签 SAC+AZ-MoE-MLP, 骨干 = 稀疏 MoE(MLP 专家)) ——
     *  原来这一支和 AGENT_SACAZ (Mlp) / AGENT_SACAZ_MOE (TB 专家) 挤在同一个类里,
     *  几支的参数量、每次前向的耗时、有没有路由、有没有注意力头都不同, 所以报告
     *  **第一行**必须自报家门 (`backboneName()` + `guiAgentLabel()`): 没有这一行,
     *  看面板的人会把骨干之间的差别记到错的账上 (例如把 TB 专家贵 3 ms/前向 读成
     *  "这个 agent 就是慢")。拆分之后这条纪律仍然保留 —— 面板上还有别的 agent。
     *
     *  报告三类事实 (全是**结构 / 口径**, 一条棋力结论都没有):
     *    1. **表示健康度**: 状态 = 14 棋子平面 x 90 格 + 3 个规则上下文标量
     *       (规则历史在本编码里**可观测**, 与 PGE/DQN 的 90 维相反); 动作 = 128
     *       槽位哈希, 而真实走法空间是 8100 个 (from,to) 对 —— 所以在**标准开局**
     *       (确定性参照点) 与**当前局面**上各报一份别名读数。别名是策略精度的
     *       结构性上限, 不是训练量的问题。
     *    2. **算法 / 口径**: 双 critic + 目标网 + Polyak 同步周期、最大熵叶子价值
     *       min_i Q_i − α·log π、α / 熵比 / azWeight / c_puct / 模拟数 / 学习率,
     *       以及**终局取真实胜负而不是自举** (这是局末价值目标可信的原因)。
     *    3. **骨干与训练进度**: 参数量、专家数 / topK、MoE 使用直方图与均衡判读
     *       (某专家计数为 0 = 路由坍缩)、learnSteps / 批大小 / epochs / 池 /
     *       叶子评估次数 / 最近一次 critic MSE (NaN = 还没上报)。
     *
     *  **只读、可重复、不动棋盘**: 全部走 `chess` 的副本 (`Chess probe(chess)`),
     *  绝不碰 `this->chess` (面板会在对局中途被 GUI 线程调用, 而那一刻搜索线程可能
     *  正拿着同一个棋盘)。不跑搜索、不跑前向、不读写权重 —— 只读现成的计数器,
     *  预算几毫秒。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;

    /* 统计 */
    long long getLeafEvals() const { return m_leafEvals; }
    int getLearnSteps() const { return learnSteps; }
    float getAlpha() const { return alpha[0]; }
    std::size_t getMemorySize() const { return memories.size(); }
    int getTotalEpisodes() const { return totalEpisodes; }
    /*
     * 最近一次 learnBatch 的 critic 损失 (MSE, 只对实际走过的那一步回归)。
     * 界面的"训练损失曲线"用它; 还没学过时是 NaN (曲线控件会丢弃非有限值)。
     */
    float getLastTrainLoss() const override { return m_lastLoss; }
    /*
       [P4] 最近一次 learnBatch 攒下的样本条数 (batchSize × epochs)。把它做成公开的
       只读数字, 是为了让"多 epoch 真的是每遍都抽新样本"这件事可以被断言
       (test_sacaz 的 P4 一节)。
    */
    int getLastBatchSamples() const { return m_lastBatchSamples; }
};

#endif // SACAZ_MOEMLP_AGENT_H
