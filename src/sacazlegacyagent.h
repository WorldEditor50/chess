#ifndef SACAZ_LEGACY_AGENT_H
#define SACAZ_LEGACY_AGENT_H

#include <vector>
#include <string>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <type_traits>      /* 只为文件末尾那条"不许再继承回去"的 static_assert */
#include "chess.h"
#include "aiagent.h"
#include "rl/net.hpp"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"
#include "rl/parameter.hpp"
/*
   下面这个 include **只为编译期护栏** (文件末尾那条 static_assert), 不是复用:
   SACAZLegacyAgent 与 SACAZAgent 之间**没有任何继承/复用关系** (见头注释)。
   引它进来是为了让"以后有人把它改回派生类"这件事**编不过**, 而不是为了少写几行。
*/
#include "sacazagent.h"

/* 稀疏 MoE 层只需要指针/引用 (定义在 rl/sparse_moe.hpp, 由 .cpp 包含) */
namespace RL {
class ISparseMoE;
}

/*
 * ============================================================================
 *  SACAZLegacyAgent —— SAC+MCTS+AlphaZero 的**行为还原版** (提交 59e5233)
 * ============================================================================
 *
 *  用户口径 (2026-09): "另外实现 agent 还原回 59e5233 的 SAC agent", 并且要求**新旧
 *  SAC 用不同的 C++ 类区分开** —— 于是这一支是界面上的 AGENT_SACAZ_OLD。
 *
 * ---------------------------------------------------------------------------
 *  2026-09 后续: 本类改为**独立类**, 不再继承 SACAZAgent
 * ---------------------------------------------------------------------------
 *  它原来是 `class SACAZLegacyAgent : public SACAZAgent` (一个薄派生类: 只在自己的
 *  构造函数里钉住 5 项口径 + 覆盖两段文案)。那个做法有一个**已经发生过**的失效模式:
 *
 *    * "还原"是靠**枚举差异**实现的 —— 凡是没有被显式钉住的量, 都会随基类默认值一起
 *      漂走。2026-09 就漂过一次: F1 那一轮把基类的目标网同步率默认值改成"硬拷贝 / 每
 *      256 步", 而派生类没钉这一对 ⇒ "59e5233 行为还原版"不再是 59e5233 的行为
 *      (docs/sac_critic_diagnosis_2026_09.md §13.5, 用户发现的就是这一条);
 *    * 修法是"再钉一项"——但那是**打地鼠**: 基类以后每加一个开关/改一个默认值, 都有
 *      一次"忘了同步钉住"的机会, 而错误是**静默**的 (两者参数结构完全相同, load 不会
 *      失败、参数量不会变、界面上看不出任何异常)。
 *
 *  所以用户改了口径: **本类与 SACAZAgent 没有任何继承关系** —— 它自带一份 SAC 实现
 *  (网络 actor/q1/q2/q1Target/q2Target + α、MCTS 搜索树、learnBatch、selectMove、
 *  exploreAndTrain、computeReward、编码/动作/存取权重、自检), 也就是 sacazagent.cpp
 *  里那份代码的 1:1 拷贝 + 59e5233 的口径。文件末尾有一条 static_assert 把这条约束
 *  **变成编译错误**: 谁再写 `: public SACAZAgent`, 编译当场失败。
 *
 *  ---- 代价 (必须说清楚, 这是刻意的取舍) ----
 *  1. **代码重复**: 这一支与 SACAZAgent 是两份实现, 共享算法上的修复(本仓库刚修过好几个
 *     bug)**不会自动流过来**。以后每次在 sacazagent.cpp 上修 bug 或做改进, 都要**刻意
 *     决定**要不要同步到这一份 —— 而"刻意决定"意味着**不决定就是不修**。
 *     这是用户明确接受的代价: 这一支的价值是"行为可复现", 而可复现的前提是**不被别人
 *     改动牵连** (旧头注释主张的"一份实现不会漂移"在"还原版"这个场景下不成立:
 *     这里要的恰恰是**冻住**, 而不是跟着共享实现一起演进)。
 *  2. 编译/维护成本翻倍 (两个 .cpp 都要进 CMakeLists 的每个相关目标)。
 *  3. 想要"两个 SAC 谁强"的对局时, 读的人必须知道: 它们现在是**两份独立实现**,
 *     任何一处的算法修复都会让两边重新分叉。
 *
 * ---------------------------------------------------------------------------
 *  本类**刻意不含**的东西 (59e5233 就是这样; 加了就说明被污染了)
 * ---------------------------------------------------------------------------
 *  SACAZAgent 在 2026-09 之后长出了一批实验开关与"抑制 critic"的机制。它们**全部不在
 *  本类里** —— 不是"默认关掉", 而是成员与代码路径都不存在, 所以没有任何 flag 能打开,
 *  也没有"复制粘贴顺手带进来"的入口:
 *
 *    * **奖励塑形**: 即时奖励就是 `材质 x REWARD_MATERIAL_COEF + 每步代价`, 终局就是
 *      引擎真值 ±1/0 (唯一出口 `terminalReward()` 只有一行 `outcomeForMover(...)`)。
 *    * **critic 值域约束**: 目标**不夹**, 损失是**纯 MSE**。没有"目标钳位"、没有
 *      "Huber 分段"、没有"搜索叶子整体缩放"、也没有"熵项去处 / 目标熵分母"这类开关。
 *    * **稀疏头叶子估值**: 每片叶子算**全量** Q (那时动作空间 128 槽, 全量只要 128 列;
 *      稀疏头是为 8100 列才加的)。连"稀疏策略 / 稀疏 Q / 稀疏软价值"那三个只算合法列的
 *      函数一起删掉了。
 *    * **"从自己的搜索学一次"**: 59e5233 只在 rollout 里学 (那批样本 hasSearch=false),
 *      所以 `selectMove` 在本类里是**只读搜索**。
 *    * **可调的目标网同步率**: Polyak 步长与同步节拍是**编译期常数**
 *      (POLYAK_TAU / TARGET_SYNC_EVERY), 外部无法覆盖。
 *
 *  需要以上任何一项时, 请用 **SACAZAgent 那一支** (AGENT_SACAZ / AGENT_SACAZ_MOE) 做
 *  实验, 不要在"行为还原版"上做 —— 在那上面做消融等于把它变成另一支算法。
 *
 * ---------------------------------------------------------------------------
 *  59e5233 口径表 (本类的**全部**行为参数)
 * ---------------------------------------------------------------------------
 *  项                      | 值 (= 59e5233)          | 在本类里的形态
 *  ------------------------|-------------------------|--------------------------------
 *  目标熵 entropyRatio     | 0.98                    | 成员默认值 (构造函数初始化表)
 *  alpha 学习率            | 1e-3                    | 成员默认值
 *  critic 目标钳位         | 无 (纯 MSE)             | **没有这个成员 / 这条代码路径**
 *  Huber δ                 | 无 (纯 MSE)             | 同上
 *  搜索叶子估值            | 全量 Q                  | 硬编码 (没有开关)
 *  "从搜索学一次"          | 无                      | 硬编码 (没有成员, 也没有那个函数)
 *  目标网同步率            | tau=1e-3 / 每 64 次 learn | 编译期常数 POLYAK_TAU / TARGET_SYNC_EVERY
 *  奖励塑形                | 无                      | 硬编码 (没有成员)
 *  网络结构 / 隐层激活     | 与 59e5233 逐字相同     | buildNet 同一份代码 (Layer<Tanh>)
 *  动作/状态表示           | 128 槽 / 1263 维        | 默认构建 (SACAZ_ALIGNED_REPR 未定义)
 *  ------------------------|-------------------------|--------------------------------
 *
 *  α 的口径 (0.98 / 1e-3) 与当前实现**恰好相同**, 但原因不同: 2026-09 的受控实验证明
 *  "目标熵 0.5 + αlr 5e-3"那一套在对弈里**有害** (训练后 37.5% vs 未训练 52.5%), 于是
 *  当前实现改回了 59e5233 的值 (见 docs/sac_learn_reward_2026_09.md §9)。本类的这两个
 *  数来自 59e5233 本身, 不是为了跟随当前实现。
 *
 * ---------------------------------------------------------------------------
 *  权重文件独立 (用户口径: "新旧 sac agent 的权重文件用不同名字区分开")
 * ---------------------------------------------------------------------------
 *   AGENT_SACAZ      -> weights/sacaz_agent      (_actor / _q1 / _q2)
 *   AGENT_SACAZ_OLD  -> weights/sacaz_old_agent  (_actor / _q1 / _q2)   <- 本类
 *   AGENT_SACAZ_MOE  -> weights/sacaz_moe_agent  (_actor / _q1 / _q2)
 *
 *  为什么不能共用一个前缀 (两个后果都是**静默**的):
 *    * 两者的训练口径不同 (上表), 同一个局面会被训成两组不同的权重 —— 共用前缀 =
 *      后训练的那一支直接覆盖另一支, 界面上一切正常, 只有"棋力对不上训练量"这种
 *      无法归因的现象;
 *    * 载入也一样: 两者**参数结构完全相同** (都是 iFcLayer 的 w/b), 结构指纹挡不住,
 *      于是错的那一份会被当成对的那一份用 (本仓库 PPO+MCTS 就栽在"名字漂移 ⇒ 权重
 *      从来没被载入过"上, 见 chessboard.cpp 的 weightFilesOf 注释)。
 *  前缀由本类的 defaultWeightPrefix() 给出; 后台训练的临时前缀也独立
 *  (weights/_temp_train_sacaz_old*)。
 *
 * ---------------------------------------------------------------------------
 *  本文件 / 实现的位置
 * ---------------------------------------------------------------------------
 *  声明在本头文件, 实现在 src/sacazlegacyagent.cpp (与 sacazagent.cpp 同一套写法:
 *  一份实现一个 .cpp, CMakeLists 里每个用到它的目标自己列源文件 —— 见 CMakeLists
 *  顶部"不用 GLOB"的说明)。它不再是"头文件里的薄类", 因为它现在带整套算法。
 */
class SACAZLegacyAgent : public AgentBase
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
     *  骨干 (backbone): 决定 actor / critic 的中间层长什么样
     *
     *  为什么要有这个开关: d_model=1260 上"专家数"直接乘在算力上, 所以要把
     *  "容量"和"算力"分开量 —— 见 docs/agents_design.md §11.4。
     *  四种骨干的参数量/算力对比 (实测见 §11.4.1 与 test_sparse_moe):
     *
     *    Mlp         1260 -> h -> h -> out        基线 (算力最小, 容量也最小)
     *    SparseMoeMlp 稀疏 MoE (E=8, top-2, 专家是 MLP) + h -> out
     *                容量 8 倍于一个 MLP 专家, 算力 ~2 个专家 (与 E 无关)
     *    SparseMoeTb  稀疏 MoE (E=4, top-1, 专家是 TransformerBlock) + h -> out
     *                容量最大, 但一个 TB 专家 ~3.2 ms/前向 -> 只能配很小的模拟次数
     *    DenseMoeTb   与 SparseMoeTb **完全相同的参数**, 但 4 个专家全算
     *                这是"等参数不等算力"的对照组: 用来量稀疏路由本身值多少
     * ---------------------------------------------------------------- */
    enum class Backbone {
        Mlp = 0,
        SparseMoeMlp,
        SparseMoeTb,
        DenseMoeTb,
        Count
    };
    static const char *backboneName(Backbone b);

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
       自检面板第一行必须能回答这个问题。本类**不是**虚函数覆写: 它只服务一个界面类型
       (AGENT_SACAZ_OLD), 没有第二个身份来源 (同族的 AGENT_SACAZ / AGENT_SACAZ_MOE 由
       SACAZAgent 自己报)。写死在这里是因为"哪一支"完全由**类**决定。
    */
    const char *guiAgentLabel() const;

    /*
       ---- 权重文件前缀 (把"是哪一支"与"写到哪个文件"绑在一起) ----
       用户口径 (2026-09): **新旧 SAC 的权重文件必须用不同名字** —— 本类返回
       `weights/sacaz_old_agent` (SACAZAgent 那一支是 `weights/sacaz_agent`)。
       为什么不能共用一个前缀:
         * 两者的训练口径不同 (见头注释里的口径表), 同一局面对弈会被训成两组不同的
           权重; 共用一个前缀 = 后训练的那一支**静默覆盖**另一支, 而界面上一切正常;
         * 载入也一样: 两者**参数结构完全相同** (都是 iFcLayer 的 w/b), 结构指纹挡不住,
           于是错的那一份会被当成对的那一份用。
       所以前缀跟着**类**走, 界面按 agent 类型取默认值, 不靠各处手抄字符串。
    */
    static const char *defaultWeightPrefix();

    /*
       ---- 59e5233 的口径常数 (编译期, 不可从外部覆盖) ----
       这些是原来那个派生类里用来"显式钉住"的常量。现在本类是独立实现, 它们不只是给
       测试对表用的**证据**, 而是真正被算法使用的常数:
         * LEGACY_ENTROPY_RATIO / LEGACY_ALPHA_LR : 目标熵 / α 学习率 (构造函数初始化表);
         * POLYAK_TAU / TARGET_SYNC_EVERY          : 目标网 Polyak 步长与同步节拍
           (learnBatch 里直接用, **没有对应的成员**, 外部改不了)。
       保留成 public 是为了让"这一支的口径到底是什么"可以被 test_sacaz [14] 逐项断言,
       而不是只能读注释。
    */
    static constexpr float LEGACY_ENTROPY_RATIO = 0.98f;   /* 59e5233: 目标熵系数 */
    static constexpr float LEGACY_ALPHA_LR = 1e-3f;         /* 59e5233: α 的学习率 */
    static constexpr float POLYAK_TAU = 1e-3f;              /* 59e5233: Polyak tau */
    static constexpr int TARGET_SYNC_EVERY = 64;            /* 59e5233: 每 64 次 learn 同步 */

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

    Backbone backbone;        /* 骨干类型 (构造时定, 之后不要改) */
    int expertHidden;         /* MLP 专家的隐层宽度 (只用 MlpExpert 骨干时有效) */
    float auxLossCoef;        /* 稀疏 MoE 负载均衡辅助损失的系数 (0 = 关掉) */    int hiddenDim;
    float gamma;
    float learningRateActor;
    float learningRateCritic;
    float learningRateAlpha;
    float c_puct;             /* PUCT 探索常数 */
    float azWeight;           /* 策略损失里 AZ 监督项(交叉熵)的权重 */
    float entropyRatio;       /* 目标熵 = entropyRatio * log(合法走法数) */
    int simulations;          /* getBestMove 默认的搜索模拟次数 */
    int batchSize;
    /*
       ================================================================
        [F1 2026-09] 目标网的 Polyak 同步 —— **本类是硬编码常数, 没有成员**
       ================================================================
       为什么这里刻意**不留**可调成员 (这是 2026-09 事故的直接教训):
       派生类时代, 本类只显式钉住了"5 项口径", 而目标网同步率 (tau + 节拍) 是**基类默认
       值**。F1 那一轮把基类默认改成了"硬拷贝 / 每 256 步", 于是"59e5233 行为还原版"
       **跟着一起变了** —— 而它的全部意义就是行为还原 (用户发现的就是这一条, 见
       docs/sac_critic_diagnosis_2026_09.md §13.5)。
       现在的做法是结构性的: 步长与节拍是 `POLYAK_TAU` / `TARGET_SYNC_EVERY` 两个
       **编译期常数**, 直接用在使用点 (learnBatch), 没有可以覆盖的成员 ⇒ "面板显示的
       值"与"实际生效的值"不可能不一致, 也不可能被命令行/别的类顺手改掉。

       它的量级意味着什么 (读数而非缺陷): 一次 20 局 (~1300~2900 次 learn) 只把目标网
       从随机初始化挪动 2~4%, 所以 `|Q_target|` 一直停在 0.07~0.10 (随机尺度), 自举项
       `V(s') = E[min Q_target] + alpha*H` 里的游戏信息几乎为 0 —— 这是**还原对象的一部分**,
       不是待修的 bug。要修它请到 SACAZAgent 那一支去 (那边有可调的目标网同步旋钮,
       实测"硬拷贝每 256 步" 65.0% -> 67.3%, p = 0.43)。
       ================================================================
    */
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
       ================================================================
        ---- [2026-09] 本类**没有**"搜索叶子缩放 / 稀疏头推理"这两套东西 ----
       ================================================================
       SACAZAgent 那一支有两组与**搜索期求值**有关的东西, 本类一个都没有:
         * "搜索叶子值的整体缩放因子" + 它的包装函数 (搜索用缩放 * 软价值): 那是用来
           单独检验"是 critic 的尺度把搜索弄坏了"的对照旋钮 (见
           docs/arena_sac_vs_ppo_report.md §7)。本类里搜索叶子值就是 `softValueFrom`
           本身, 没有任何乘子 —— 59e5233 就是这样。
         * 稀疏头推理 (三个只算合法列的函数: 稀疏策略 / 稀疏 Q / 稀疏软价值): 它们是
           动作空间扩到 8100 双射之后才加的**性能**路径 (全量叶子估值实测 216 ms/步)。
           本类的叶子估值一律走**全量** (128 槽下一次只要算 128 列), 所以这三个函数
           连同它们的静态助手一起删掉了 —— 少一条与训练路径不同的代码, 也少一个
           "两个口径之间差在哪"的变量。
       需要这两样请用 SACAZAgent 那一支 (对齐表示 + 稀疏头是那边的性能选项)。
    */

    /*
       ================================================================
       ---- [2026-09] 本类**不含任何 critic 值域约束** (目标不夹 + 纯 MSE) ----
       ================================================================
       用户口径 (逐字要求): "本类刻意不含奖励塑形、不含任何 critic 值域约束 (目标钳位 /
       Huber 分段 / 搜索叶子缩放 / 熵项开关都没有) —— 59e5233 就是这样; 加了就说明被
       污染了。"

       SACAZAgent 那一支有两条约束, 它们是 2026-09 之后**新增**的、59e5233 里没有:
         1. "目标钳位" —— 结构性: 把 TD 目标 y 夹到 [-2, 2];
         2. "Huber 分段" —— 鲁棒性: |err| > δ 时损失从平方变成线性。
       本类两条都**不存在** (成员与代码路径都没有): `learnBatch` 里 y 原样进目标,
       损失是 `0.5 * err^2` 的纯 MSE, 梯度走 `RL::Loss::MSE::df` (两者逐位一致)。

       ---- 为什么它们当时被加进去 (背景, 也是本类**刻意**不修的那个缺陷) ----
       实测 (docs/arena_sac_vs_ppo_report.md §5.2, 工具 --mode=probe):

           权重         |Q| 均值      Q 区间
           随机初始化     0.063      [-0.14, 0.10]
           自对弈 40 局   4.146      [-4.41, -3.63]
           自对弈 150 局 13.381      [-14.06, -11.94]

       单调发散, 而且**所有动作一起变负** ⇒ 不是在学更好的排序, 而是在整体漂移。
       机制: 即时奖励上界 0.35 (= REWARD_MATERIAL_COEF x 一方满子), 终局 ±1, 所以真实 Q
       必然落在 [-1.5, 1.5] 量级内; 但软备份 `y = r − γ(1−done)·V(s')` 把 V 反复回代,
       而优化器 (RMSProp + clipGrad, 按向量归一) 没有任何把 V 拉回该区间的机制。
       后果不是"数值不好看": 叶子估值就是软价值, 而 PUCT 是 `Q + c_puct·P·√N/(1+n)` ——
       Q 一旦压过探索项, 先验与访问计数的信息被淹没 (对 MCTS 得分率 73.3% -> 32.5%)。

       **本类保留这个缺陷**: 训练 20 局后 |Q| 均值 ~5.6 (远超"真实 Q 应该在 2 以内"),
       这正是"没有任何东西在抑制 critic"的**可观测证据** (bench_sac_learn --legacy 的
       自检读数)。要压发散就到 SACAZAgent 那一支去开那两条约束 —— 在这里加任何一条,
       这一支就不再是 59e5233 的行为。
       ================================================================
    */

    /*
       ================================================================
       ---- [2026-09] 本类没有"熵项去处 / 目标熵分母"这两个实验开关 ----
       ================================================================
       背景 (值得留着, 因为它解释了本类**为什么**保持原式):
       软价值是 `softValueFrom = E_π[min Q] + α·H(π)` (H = 策略熵), 而 TD 目标是
           y = r − γ(1−done)·V(s')
       于是 **熵项以 −γ·α·H(s') 的形式直接进入 critic 的回归目标** —— 它与棋局无关,
       是一个常数偏置。若 Q 在各动作上近似相等 (= q), 不动点解就是
           q = (r − γ·α·H) / (1 + γ)
       即"α·H 有多大, critic 就被推到多远" (实测: αH ≳ 4 的那一档 q 贴着 ±2 的钳位边界,
       |Q| 均值 2.008; αH → 0 的那一档 |Q| = 0.035)。**两种都是没有排序信息**, 只是被
       推到的常数不同。

       SACAZAgent 那一支为此加了两个开关: ① 把熵项**从软价值里摘掉** (只留在策略损失
       里), 用来单独检验"是熵项把 critic 顶走的"; ② 把目标熵的分母从"合法着法数"换成
       "合法槽位数" (128 槽哈希有碰撞, 而 π 只分布在槽位上, 所以按着法数算的目标熵
       **可能永远达不到**, 而 α 的梯度恰好是 (H − H̄) ⇒ α 被单向推走)。

       **本类两个都不含**: 熵项就是加上的 (59e5233 的原式, 逐位一致), 目标熵分母恒为
       "合法着法数"。要做这两个消融, 请到 SACAZAgent 那一支。
    */

    /*
       ================================================================
       [2026-09 ①] 训练中的 critic/α 诊断累计量 (只读; 不参与任何计算)
       ================================================================
       为什么需要它: 原有读数 (m_maxAbsTarget / m_maxAbsTdErr / 训练后的 |Q| 快照) 只能看到
       "最后一次"和"最后的状态", 而本轮要回答的是**过程**里的三个问题:
         1. 目标 y 的分布是什么样 (被值域约束夹住的比例) —— 夹住 = 那部分样本的目标是个
            常数, 对 critic 的**排序**学习毫无贡献。**本类不夹目标, 所以那个计数恒为 0**
            (它是给"有约束"的变体留的读数, 保留字段是为了两边日志口径能直接对照);
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
        long long clamped = 0;         /* 目标被值域约束夹住的条数 (**本类恒为 0**) */
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
    };
    TrainDiag trainDiag;
    const TrainDiag &getTrainDiag() const { return trainDiag; }

    /*
       ================================================================
       ---- [2026-09] 本类**不含奖励塑形** (既没有即时项缩放, 也没有终局放大) ----
       ================================================================
       用户口径 (2026-09): "本类刻意不含奖励塑形 ... 59e5233 就是这样; 加了就说明被
       污染了。" 具体是两样东西, 本类**连成员都不留**:
         * 即时奖励的整体缩放 (原方案: 只缩放材质项, 不动终局 ±1, 用来问"这个塑形值
           多少");
         * 终局放大 (原方案 `1 + 败方剩余材质/满材质`, 取值 [1,2) ——
           "对方兵力越完整就被将死 = 越值钱", 即快杀 > 磨死)。
       历史背景 (为什么当初想做, 以及它为什么只能落在终局上):
         "59e5233 还原版"对 MCTS 的**奖励累计**能到对手的 2 倍, 但很多局在 300 手判和
         —— 吃着子却赢不了, 于是怀疑奖励结构把"吃子"抬得太高、把"将死"压得太低。
         用户原话是"环境奖励乘以 (1 + 最终棋子数/总棋子数)", 但:
           * "**最终**棋子数"在**即时奖励**上不可实现 (在线 TD 里落子时并不知道结局);
           * 所以那个量只能落在**终局**, 那是唯一"结局已确定"的时刻;
           * "整个 episode 的奖励统一乘一个系数"在本算法里也做不到: SAC 是 off-policy
             回放, 一条经验可能在局终之前就被学过好几次, 事后改它的 reward 等于改历史。
       本类的口径: 即时 = `材质 x REWARD_MATERIAL_COEF + 每步代价`, 终局 = 引擎真值
       ±1/0 (唯一出口 `terminalReward()`, 只有一行 `outcomeForMover(...)`)。

       **只有一个出口**这条纪律仍然成立: 终局值有三个产生点 (搜索叶子 `terminalValue` /
       自对弈 `resultValue` / rollout 的 `outcomeForMover`), 三者口径不一致的话, 搜索估
       的与训练学的就是两个游戏 —— 所以三个点全部走 `terminalReward()`。
       (`agentrollout.hpp` 用 SFINAE 检测这个可选成员: 有就用, 没有就退回共享的
       `outcomeForMover`; 本类的返回值与回退路径**逐位相同**, 所以这条探测现在是等价的。)
    */

    /*
       终局值 —— 塑形方案的**唯一**出口。
       `perspective` = 视角方 (语义与 chess.h 的 `outcomeForMover` 一致: 返回 +1 表示
       这个颜色赢了); 三个调用点传的都是"该视角的颜色": rollout 里是**刚走子**的一方
       (他可能就是赢家), 搜索里是**轮到走**的一方 (终局节点上他是被将死的那一方)。
    */
    float terminalReward(int chessResult, int perspective) const;

    /*
       ================================================================
       ---- [2026-09] 本类**没有**"从自己的搜索学一次"这条路径 ----
       ================================================================
       背景 (SACAZAgent 那一支 2026-09 新加的机制, 值得留着当对照):
       用户实测报过"界面对弈里只有勾上 rollout(探索+预训练) 才有损失曲线"。原因: 界面
       的 per-move 学习完全由 ChessBoard::preTrainThenDecide 驱动 (关掉就短路返回), 而
       SAC 在整局里唯一的学习来源就是 `exploreAndTrain` 的 rollout —— 那条路径写进去的
       样本是 `hasSearch=false`, 于是**搜索算出来的 π_MCTS 在对弈中被丢掉了**, AlphaZero
       的监督项只有后台训练 `trainSelfPlay` 用得上。
       那边的修法是: 选完真实走法之后, 把 (s, π_MCTS, a, r, s', done) 存进回放池并做
       一次 `learnBatch`, 顺带把损失送上曲线。

       **59e5233 没有这条路径** (它对弈时只在 rollout 里学), 所以本类:
         * 没有那个成员开关;
         * 没有"把这一步存成 AlphaZero 样本"的函数 (整个函数一起删掉了);
         * `selectMove(color, sims, temp, piOut)` 因此是**只读搜索** —— 不改权重、
           不往池里写样本 (界面/工具上表现为: 本类每手的"损失曲线"只由 rollout 与
           自对弈产生)。
       想要在线搜索监督信号, 用 SACAZAgent 那一支 (工具侧 `bench_sac_learn` 不带
       `--no-search-learn` 就是开着的)。
    */

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
       终局那一条覆写成 terminalReward(): 走 agent 自己的出口才能保证"界面上显示的
       游戏"与"它真正学的游戏"是同一个 (本类没有塑形, 所以它与引擎口径都是 ±1/0 ——
       覆写是**纪律**, 不是为了改数值)。
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
    /* 按 backbone 造一个网络 (withGrad=false 用于目标网) */
    RL::Net buildNet(bool withGrad) const;
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
       **不再是 static**: 它要经过 `terminalReward()` —— 终局口径的**唯一出口**。
       (SACAZAgent 那一支的出口还需要读棋盘, 因为它的终局放大要看败方剩余兵力;
       本类没有塑形, 所以这里只是"所有终局值都必须走同一个出口"这条纪律的载体。)
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
    SACAZLegacyAgent(Chess &chess_,
               int hiddenDim_ = 64,
               float gamma_ = 0.99f,
               float lr = 0.001f,
               float cpuct = 1.5f,
               Backbone backbone_ = Backbone::Mlp,
               int expertHidden_ = 64,
               float auxLossCoef_ = 0.1f);
    ~SACAZLegacyAgent() = default;

    Step getBestMove(int color) override;
    std::string getName() const override;
    bool exploreAndTrain(int color, int rolloutSteps) override;

    /* ----------------------------------------------------------------
     *  自检 (界面"模型自检"面板) —— 契约与口径见 aiagent.h 的 selfCheckReport
     *
     *  为什么这一支特别需要它: 它的全部价值就是"行为还原", 而**还原不还原**只能看
     *  口径读数 (结构 / 参数 / 开关的实际取值), 不能看棋力 —— 报告**第一行**就报
     *  界面 agent 类型与骨干, 然后是一条"本类的口径全是硬编码"的说明行:
     *  59e5233 的关键读数 (目标网同步率只有 2~4%、|Q| 会发散到 5.6) 在面板上必须
     *  **看得见**, 否则会被当成"模型坏了"而不是"这就是还原对象的行为"。
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

/*
 * ============================================================================
 *  编译期护栏: 本类**不许**再变回 SACAZAgent 的派生类
 * ============================================================================
 *  用户口径 (2026-09): "让 SACAZLegacyAgent 成为一个**不继承** SACAZAgent 的独立类,
 *  这样 SACAZAgent 以后任何默认值或实现上的改动都不可能渗进 59e5233 还原版。"
 *
 *  这条 static_assert 把那句要求变成**编译错误**, 而不是一条注释:
 *  谁把它改回 `class SACAZLegacyAgent : public SACAZAgent` (那是它 2026-09 之前的样子,
 *  当时"还原"靠"枚举差异 + 逐项钉住", 已经因为基类默认值被改而漂过一次, 见
 *  docs/sac_critic_diagnosis_2026_09.md §13.5), 编译当场失败, 而不是等到某次对局
 *  读数对不上才发现。
 *  要"让两支共享实现"就得先删掉这一条并说清楚为什么 —— 那正是这条断言存在的意义。
 */
static_assert(!std::is_base_of<SACAZAgent, SACAZLegacyAgent>::value,
              "SACAZLegacyAgent 必须是独立类, 不许继承 SACAZAgent —— 否则基类的默认值/"
              "行为改动会再次渗进 59e5233 行为还原版 (见本文件头注释与 "
              "docs/sac_critic_diagnosis_2026_09.md §13.5)");

#endif // SACAZ_LEGACY_AGENT_H
