#ifndef DQNAB_AGENT_H
#define DQNAB_AGENT_H

/*
   ============================================================================
   [2026-10 第五轮] 编译期守卫 + 说明: 序列 Transformer 专家 (DQNAB_MOE_EXPERT_SEQ=1)
   ============================================================================
   用户口径: "把各 agent 的 TB 专家换成序列 transformer"。

   本类 (DQN+AB) 的做法与 `SACAZMoETbAgent` 那一支**完全同一套**: 用编译期开关把
   骨干里的专家从 `RL::TransformerBlock<16,360>` 换成
   `RL::SeqTransformerExpert<90,4,2>` —— 与界面 PPO 支是**同一份实现**
   (`src/rl/seq_transformer.hpp`), 输入就是本类现有的 19x90 = 1710 维平面优先编码
   (由 `ChessState::encodeComplete` 写出, 无需改编码)。

   开关方式 (默认**不开**, 与 PPO 支 / SACAZ 支同一手法):
       cmake -S . -B <build> -DCMAKE_CXX_FLAGS="-DDQNAB_MOE_EXPERT_SEQ=1"
   未定义 (= 0) 的构建: 专家仍是 `TransformerBlock<16,360>`, 网络结构/参数量/权重前缀
   与**全部读数**与改动前逐位相同 ("默认路径一位不动")。

   为什么这一段守卫写在文件最前面 (而不是等运行时 abort): 序列专家按**格子**切 token,
   它是靠 `dIn / SeqLen` 反推平面数的, 所以要求 `dIn % SeqLen == 0` 且
   `SeqLen == POS_ROWS*POS_COLS == 90`。本类的
   `STATE_DIM = PLANES(19) * CELLS(90) = 1710` 天然满足, 于是这个条件是一件
   **编译期事实**, 就地用 `static_assert` 钉住 (见类里那一组断言) —— 写成运行时检查
   只会在"哪天换了表示"时才炸, 而且是在用户的棋局跑到一半时炸。本工程的纪律:
   编译期常量的错必须响亮失败 (参见 seq_transformer.hpp 构造函数里那两处 abort)。
   ============================================================================
*/

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "aiagent.h"
#include "chess.h"
#include "rl/expert.hpp"        /* TransformerBlock / ExpertFactory / scaleExpertInit */
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/parameter.hpp"
#include "rl/transformer.hpp"
/*
   序列 Transformer 专家 (真正的 Transformer: 按格子切 token / 逐行 softmax / 位置嵌入)。
   **只在开关打开时 include** —— 默认构建里这一支一个符号都不参与编译, 于是"关掉开关 =
   与改动前同一份代码"这件事不靠人守纪律, 而是编译期事实。
*/
#if defined(DQNAB_MOE_EXPERT_SEQ) && DQNAB_MOE_EXPERT_SEQ
#include "rl/seq_transformer.hpp"
#endif

namespace RL {
class ISparseMoE;
}

/*
 * ============================================================================
 *  DQNABAgent — "Neural Alpha-Beta + Dueling DQN"
 * ============================================================================
 *
 * 一句话: **把 Alpha-Beta 当成 DQN 的 planning head**。
 *   * 网络学 Q(s,a), 用 Dueling 分解成 `Q = V + A − mean_legal(A)`;
 *   * AB 用它做三件事:  (a) **走法排序先验** (按 Q 降序), (b) **叶子评估** (V(s)),
 *     (c) **产 TD 目标** (从 s' 展开若干层后的值, 而不是单步 bootstrap);
 *   * AB **不是**探索噪声源, 也**不拿手工 PST** 当评估: 根节点用全宽 + 温度采样做探索,
 *     内部节点按 Q 先验选择性展开, 评估完全来自网络。
 *
 * ---------------------------------------------------------------------------
 * 1. 组件分工 (与用户草案一致, 也是这个类唯一的存在理由)
 * ---------------------------------------------------------------------------
 *   DQN 头      : 学 Q/V; 给 AB 提供叶子值 + 排序先验; 自己不直接决定落子
 *   Alpha-Beta  : 推理/采集时的对抗展开器; 根全宽 (带温度采样), 内部选择性展开
 *   动态棋子价值: 由**网络条件化输出** —— 状态里带两个阶段平面 (见 §2), 没有任何
 *                 if-else 改子力权重
 *
 * ---------------------------------------------------------------------------
 * 2. 编码: 规范视角 + 两个阶段平面 (动态价值的条件化输入)
 * ---------------------------------------------------------------------------
 *   plane 0..13 : 7 类棋子 x {己方, 对方} one-hot, **规范视角** (轮到谁走, 谁的子就在
 *                 x 大的那一侧)。所以 V(s) 永远是"走子方视角"的值, 一个网络服务红黑双方。
 *   plane 14    : 剩余子力比例 (双方非将子力 / 满子), 全局标量铺满 90 格
 *   plane 15    : 手数比例 (history.size() / REWARD_MAX_PLIES), 全局标量铺满 90 格
 *   STATE_DIM   = 16 x 90 = 1440
 *
 *   两个阶段平面就是"中局马炮值钱、残局兵升值"的条件化信号: 网络自己在不同阶段学到
 *   不同的子力/位置权重。镜像不改变常数平面, 所以规范视角不受影响。
 *
 *   动作: `fromCell*90 + toCell` (规范格), **双射无碰撞** (与 PPOMCTS 同一套) ——
 *   DQN/PG/DQN+MCTS/SACAZ 用的 128 维哈希 (平均 22 个走法挤一槽) 在这里不存在。
 *
 * ---------------------------------------------------------------------------
 * 3. 零和对称是**编码保证**的, 不是约定
 * ---------------------------------------------------------------------------
 *   因为状态是规范视角, negamax 递推 `V(s) = max_a [r_a − γ·V(s'_a)]` 里的 V 对
 *   红黑是同一个函数, 于是 `Q_red(s,a) + Q_black(flip(s), a) = 0` 自动成立。
 *   不需要两套头, 也不需要手工 flip 符号 (草案里那个 `-ab_negamax(...)` 的符号
 *   在规范视角下就是"对手视角取负"这一件事)。
 *
 * ---------------------------------------------------------------------------
 * 4. 两个口径开关 (都有明确理由)
 * ---------------------------------------------------------------------------
 *   leafEval:
 *     Value (默认) —— 叶子返回 V(s)。"局面价值", 与 negamax 备份同源, 也是用户指定的形式。
 *     MaxQ         —— 叶子返回 max_a Q(s,a), TD 意义上的值; 与 V 差一个 `max_a A`。
 *   targetMode:
 *     Planned (默认) —— 目标 = `r − γ(1−done)·planValue(s')`, 其中 planValue 是
 *                      用**目标网**从 s' 展开 `trainPlanDepth` 层的结果。
 *                      这是在**采集时刻**算好的标签 (label 进回放池) ——
 *                      见 §5 的成本说明, 这样训练步不需要再做搜索。
 *     OneStepDouble  —— 教科书 Double DQN, 在**训练时刻**用目标网算:
 *                      `a* = argmax_a Q_online(s',a)` (带掩码),
 *                      `y = r − γ(1−done)·Q_target(s', a*)`。
 *
 * ---------------------------------------------------------------------------
 * 5. 成本: 深度是**预算**决定的, 不是写死的 (与草案最大的偏差)
 * ---------------------------------------------------------------------------
 *   草案假设"小网络做叶子微秒级 → 可以搜 6~12 层"。本工程实测的**稀疏 MoE + TB 专家**
 *   在 d=1440 上一次前向 **3.6 ms** (docs/agents_design.md §18.1), 比 MLP 骨干贵两个
 *   数量级; 而 AB 每个节点都要一次前向:
 *       节点数 ≈ 1 + k + k² + … + k^d          (k = 按合法着法数自适应的分支数)
 *       TB  骨干 384 节点 ≈ 1.4 s/步, 能搜到 2~3 层
 *       MLP 骨干 384 节点 ≈ 5 ms/步,   能搜到 4~6 层 (同一套算法)
 *   所以真正的控制量是 `nodeBudget` (迭代加深 + 预算截断): 宽局面自动浅、窄局面自动深
 *   —— 这就是"开局浅、残局深"的落地方式, 而且随骨干自动伸缩。bench 里两种骨干都量。
 *
 *   训练侧: `Planned` 的标签在采集时算 (每条样本一次 `1+k` 次前向), 于是**批更新里
 *   没有搜索**, 只有 forward+backward —— 这是让 3.6 ms/前向的骨干也能训得动的关键取舍。
 *
 *   ⚠ 上面这些毫秒数都是 **TB 专家**的实测值。开关 `DQNAB_MOE_EXPERT_SEQ=1` 的构建里
 *   专家是序列 Transformer (每专家 91,731 参数 / 前向 ≈ 9.7 M MAC, 见
 *   src/rl/seq_transformer.hpp §参数与代价), 它比 TB 专家**便宜**, 所以那些数字在这一支
 *   下**不适用** —— 要报这一支的 ms/步, 只能现量 (本工程不拿另一支的读数顶替)。
 *
 * ---------------------------------------------------------------------------
 * 6. 为什么**不做**势能塑形 (PBRS)
 * ---------------------------------------------------------------------------
 *   用户要求"叶子是纯终局预期, 不要把 Φ 混进去"; 而 PBRS 的作用正是把 Φ 平移进价值目标
 *   (V+Φ)。两者不能同时成立: 叶子 V 学成什么样就评估什么。所以本 agent 的奖励就是
 *   `stone.h` 的 `stepReward` + 终局 ±1 (与其它 agent 同量纲), 不做塑形。
 *   (PPO 用 Φ 是因为它没有 AB 这个规划器; 这里规划器提供了中期的位置信息。)
 *
 * ---------------------------------------------------------------------------
 * 7. 训练循环
 * ---------------------------------------------------------------------------
 *   自对弈: 根全宽评估每个合法着法 → 按 `softmax(v_a/T)` 采样落子 (T 退火); 落子后立刻
 *   用目标网从 s' 做一次小搜索算出这条样本的 TD 标签; 样本进回放池。
 *   每 `learnEveryMoves` 手做一次批更新 (梯度累积 → 一次 RMSProp → 一次 MoE 辅助损失),
 *   目标网每 `targetSyncEvery` 次**硬拷贝** (不用 tau=0.01/256 那种事实上冻结的软更新)。
 */
class DQNABAgent : public AgentBase
{
public:
    /* ---- 编码 (见 §2 / §3) ---- */
    /*
       **状态必须是严格的 Markov 状态**: 裸棋盘 + 轮到谁 *不* 够 —— 三次重复判和、
       60 回合无吃子判和、长将/循环都与"历史"有关, 而它们决定终局与回报。
       只喂 14 个棋子平面的话, "同一局面第二次出现"和"第三次出现(立刻判和)"
       会编码成**同一个向量** —— 那 V(s) 就不是 s 的函数, Bellman 备份的前提直接破掉。
       所以这里把规则上下文铺成 5 个常数平面 (等价于 concat 标量, 只是形状与棋子平面
       对齐, 卷积/线性都吃得到):
         PLANE_MATERIAL : 剩余子力比例        (局面阶段 —— 动态子力价值的条件化输入)
         PLANE_TEMPO    : 总手数 / 120        (局面阶段之二)
         PLANE_HALFMOVE : halfMoveClock / 120 (60 回合判和的风险: 越接近 1 越危险)
         PLANE_REPEAT   : 本局面已出现次数 / 2 (≥3 次就是判和; 1.0 = 再重复一次就和)
         PLANE_CHECK    : 走子方是否被将军     (将军链 / 长将上下文的入口)
    */
    static constexpr int CELLS            = 90;
    static constexpr int PIECE_PLANES     = 14;
    static constexpr int PLANE_MATERIAL   = 14;
    static constexpr int PLANE_TEMPO      = 15;
    static constexpr int PLANE_HALFMOVE   = 16;
    static constexpr int PLANE_REPEAT     = 17;
    static constexpr int PLANE_CHECK      = 18;
    static constexpr int PLANES           = 19;
    static constexpr int STATE_DIM        = PLANES * CELLS;      /* 1710 */

    /* ---- 动作: 双射, 无碰撞 ---- */
    static constexpr int ACTION_DIM     = CELLS * CELLS;       /* 8100 */

    /* ---- 骨干: 稀疏 MoE + 专家 (默认 TB; 开关打开时这一支的专家换成序列 Transformer) ---- */
    static constexpr int MOE_TB_HEADS   = 16;
    static constexpr int MOE_TB_DFF     = 360;
    static constexpr int MOE_EXPERTS    = 4;
    static constexpr int MOE_TOPK       = 1;
    using TBExpert = RL::TransformerBlock<MOE_TB_HEADS, MOE_TB_DFF>;

    /*
       ================================================================
       [2026-10 第五轮] 编译期开关: 专家 = 序列 Transformer (DQNAB_MOE_EXPERT_SEQ)
       ================================================================
       开 (`-DDQNAB_MOE_EXPERT_SEQ=1`): `Backbone::SparseMoeTb` 这一支的专家**被换掉**,
       换成 `SeqTransformerExpert<CELLS(90), 4, 2>`。为什么这不是"顺手换个写法":
         * 旧 TB 专家收的是 `(d_model x 1)` **单个向量** ⇒ `q·kᵀ` 是 `d_k×d_k` 的
           **外积**, 混合的是特征维而不是"哪个格子该注意哪个格子"; 而它的 softmax 是
           `util.hpp` 那个对**整张张量**求和的全局 softmax ⇒ 每行只分到 1/d_k 的质量
           (实测初始化时这条支路只占块输出方差的 5.6e-07, 等于死的 —— 见
           docs/tb_expert_training_2026_10.md §9);
         * 序列专家按**格子**切 90 个 token (每个 token = 19 个平面在这一格上的取值)、
           **逐行** softmax、带可学位置嵌入, 参数从每专家 12.9 M 降到 **91,731**。
       关 (未定义, 默认): 专家仍是 `TransformerBlock<16,360>`, 结构与读数逐位同改动前。

       ⚠ 三个后果 (逐条记账, 与 SACAZ 支切 seq 骨干时同一口径):
         1. **存量权重不兼容** ⇒ 前缀换成 `weights/dqnab_seq_agent`
            (见 `defaultWeightPrefix()`): 一个骨干一个前缀, 免得"新的空网把旧的盖掉"
            (载入那一侧有 `Net::load` 的参数结构守卫, 但写出那一侧没有守卫, 所以
            `saveModel/loadModel` 在开关打开时**直接拒绝** TB 那一份前缀, 见
            `canUseWeightPrefix`)。
         2. `MOE_TB_HEADS / MOE_TB_DFF` 在这一支下**不生效** —— 那是 TB 专家的参数;
            宽度改由构造参数 `expertHidden` (默认 `MOE_SEQ_TOKDIM = 64`) 决定。
         3. 位置编码走序列专家自己的旋钮 (`moeSeqPosMode()`, 与界面 PPO 支同选 `2d`);
            **设不上就 abort** —— 本工程记过很多次"旋钮设了却没生效"的事故, 而
            "专家类型不是序列专家"恰好是那种不报错、只是变差的失效。

       ⚠ 前置条件 (`STATE_DIM % CELLS == 0` 且 `CELLS == 90`) 由下面那组 static_assert
         在**编译期**钉住: 本类 1710 = 19 x 90 满足, 换表示时若不满足会当场编译失败。
    */
#if defined(DQNAB_MOE_EXPERT_SEQ) && DQNAB_MOE_EXPERT_SEQ
    static constexpr bool MOE_EXPERT_SEQ = true;
#else
    static constexpr bool MOE_EXPERT_SEQ = false;
#endif
    static constexpr int MOE_SEQ_HEADS  = 4;    /* 与 PPO 支同口径 (dk = tokDim/4) */
    static constexpr int MOE_SEQ_BLOCKS = 2;    /* Pre-LN 块数 (与 PPO 支相同) */
    static constexpr int MOE_SEQ_TOKDIM = 64;   /* token 宽 = 序列专家的 tokDim (与 PPO 支同值) */
    /* 位置编码模式: 与界面 PPO 支一致 —— 可学 2D (10 行 + 9 列) */
    static const char *moeSeqPosMode() { return "2d"; }
    /* 编译期守卫: 序列专家构造时会 abort 的那两条前置条件, 在这里先变成编译错误 */
    static_assert(STATE_DIM % CELLS == 0,
                  "DQNAB_MOE_EXPERT_SEQ=1: 序列专家按格子切 token, 要求 STATE_DIM 能被 CELLS(90) 整除 "
                  "(本类 STATE_DIM = PLANES*CELLS 恒成立; 换个表示就要在这里重新对一遍)");
    static_assert(CELLS == 90,
                  "DQNAB_MOE_EXPERT_SEQ=1: 序列专家的 SeqLen 必须是 10 行 x 9 列 = 90 "
                  "(POS_ROWS*POS_COLS), 与 canonicalCell(x,y)=x*9+y 对齐");
    static_assert(MOE_SEQ_TOKDIM % MOE_SEQ_HEADS == 0,
                  "DQNAB_MOE_EXPERT_SEQ=1: tokDim 必须能被 MOE_SEQ_HEADS 整除");
#if defined(DQNAB_MOE_EXPERT_SEQ) && DQNAB_MOE_EXPERT_SEQ
    /*
       序列专家类型。**只在开关打开时存在**: 它依赖 rl/seq_transformer.hpp, 而那个头文件
       也只在这一支里被 include (见文件顶部)。
    */
    using SeqExpert = RL::SeqTransformerExpert<CELLS, MOE_SEQ_HEADS, MOE_SEQ_BLOCKS>;
#endif

    /*
       骨干枚举。
       **取值与含义一位不动** (`SparseMoeTb = 0` / `Mlp` / `Count`): 骨干是构造参数, 取值一旦
       漂移, 任何按整数传过骨干的地方都会**静默换支** —— 本工程对"静默换支"的纪律是绝不接受
       (同"一个骨干一个权重前缀")。
       ⚠ [2026-10 第五轮] 刻意**不新增**枚举值 (例如 `SparseMoeSeq`) —— 与 `SACAZMoETbAgent`
         的 `makeTbExpertMoe()` 取同一形态: **同一个骨干身份, 专家类型由编译宏决定**。
         理由是可数的: `src/chessboard.cpp` 里有 **7 处构造点**显式传
         `DQNABAgent::Backbone::SparseMoeTb` —— 6 处 `new DQNABAgent(...)` (L1476 预加载 /
         L2890 兜底重建 / L4451 / L4804 / L6231 / L7281) + 1 处后台训练 clone 的栈上构造
         (L7689)。若改成"新增一个枚举值 + 各处按开关挑值", 就得在那 7 处各加一次 `#if`
         —— 漏掉任何一处就是"界面上一半 agent 用序列专家、一半还在用 TB"的**静默不一致**
         (而它在权重往返上看不出来: 两种结构各自自洽)。按现在的做法, 调用点**一行都不用改**,
         界面那一支只要构建时带上 `-DDQNAB_MOE_EXPERT_SEQ=1` 就整支换过来。
       ⇒ 所以 `SparseMoeTb` 这个名字在开关打开的构建里指的是"同一支骨干 (MoE + 专家),
         只是专家换成了序列 Transformer"; 名字由 `backboneName()` 按开关给出, 读数不撒谎。
    */
    enum class Backbone { SparseMoeTb = 0, Mlp, Count };
    static const char *backboneName(Backbone b);
    /*
       本结构应当使用的权重前缀 (**单一来源**, 与 `SACAZLegacyAgent::defaultWeightPrefix` /
       `DQNMCTSMOETbAgent::defaultWeightPrefix` 同一手法)。
       为什么必须按骨干分前缀: TB 专家与序列专家是**两种结构** (每专家 12.9 M vs 91,731
       参数), 共用一个前缀的后果是"新的空网把盘上那份旧权重盖掉" —— 载入那一侧有
       `Net::load` 的参数结构守卫会拒绝, 但写出那一侧没有守卫, 所以只能靠前缀分开
       (同 SACAZ 支换 seq 骨干时换 `weights/sacaz_moe_seq_shared_agent` 的理由)。
       ⚠ `saveModel / loadModel` 收的是**调用方给的前缀** (界面走
       `ChessBoard::defaultWeightPath(AGENT_DQNAB)`, src/chessboard.cpp:5979), 所以这一支
       真正用上序列前缀需要把那一行改成 `DQNABAgent::defaultWeightPrefix()` (一行;
       `weightFilesOf()` 的扫描表与退出保存都是从它推出来的, 不用改第二处) ——
       本轮只改这两个文件, **没碰它**。在它落地之前, 开关打开而调用方仍传 TB 那一份前缀时
       `saveModel / loadModel` 会**当场拒绝并按失败返回** (见 `canUseWeightPrefix`):
       宁可看得见的失败, 也不要"把盘上那份 TB 权重静默盖掉"。
    */
    static const char *defaultWeightPrefix();

    enum class LeafEval   { Value = 0, MaxQ };
    enum class TargetMode { Planned = 0, OneStepDouble };

    /*
     * 一条回放样本。
     * 局面**存稠密编码** (1440 float = 5.8 KB/条): PPO 那边为了省内存把状态存稀疏
     * (它是每步 32 KB 的 8100 维目标), 而这里动作目标只是一个下标, 稠密状态换来的是
     * "训练时不需要从稀疏列表重建棋盘/阶段平面" —— 少一整类静默错误。
     * 容量默认 4096 条 ≈ 47 MB, 相对本 agent 的网络 (TB 骨干约 600 MB) 可以忽略。
     */
    struct Sample {
        RL::Tensor state;          /* STATE_DIM 稠密 (含两个阶段平面) */
        RL::Tensor nextState;      /* 同上, s' */
        std::vector<int> legalIdx; /* s 的全部合法着法 (规范动作索引) */
        std::vector<int> nextLegalIdx; /* s' 的全部合法着法 (OneStepDouble 要用) */
        int   action = 0;
        float reward = 0.0f;
        bool  done = false;
        /*
           Planned 模式下在**采集时刻**算好的 TD 标签 (含终局/塑形口径)。
           oneStep 模式不用它, 训练时现算。
        */
        float label = 0.0f;
    };

public:
    /* ---- 配置 (构造后可改) ---- */
    Chess &chess;                  /* 与其它 agent 一样: 直接引用棋盘, 搜索用 moveForward/moveBack 试走 */
    /*
       骨干**在构造时定** (它决定网络结构, 见构造函数最后那两个参数);
       构造之后再改这个字段不会重建网络, 只会让诊断字符串与实际结构不符。
       ⚠ 最后两个参数 (`backbone_` / `expertHidden_`) 都是**结构**参数, 必须走构造函数。
    */
    Backbone backbone = Backbone::SparseMoeTb;
    /*
       序列专家的 token 宽 (= `SeqTransformerExpert` 的 tokDim)。**只对序列专家生效**:
       TB 专家的宽度是另一组编译期常量 (`MOE_TB_HEADS` / `MOE_TB_DFF`), MLP 骨干与它无关。
       它与 `backbone` 一样**在构造时定** (构造参数 `expertHidden_`): 构造之后再改这个
       字段不会重建网络, 只会让自检读数与实际结构不符 (同上面 backbone 那条说明)。
       默认 64 = 界面 PPO 支的口径; 必须能被 `MOE_SEQ_HEADS`(4) 整除 —— 不满足时序列
       专家自己会 abort 并把话说清楚, 这里不重复判一次 (判两处就会有两份会漂移的口径)。
    */
    int expertHidden = MOE_SEQ_TOKDIM;
    LeafEval leafEval = LeafEval::Value;
    TargetMode targetMode = TargetMode::Planned;

    int hiddenDim = 64;
    int searchDepth = 6;        /* 迭代加深深度上限 */
    /*
       **真正的成本控制量**: 每次决策允许做多少次网络前向 (= 搜索节点数)。
       默认 512: MLP 骨干约 7 ms/步 (能搜 4~6 层), TB 骨干约 1.5 s/步 (能搜 2~3 层)。
       深度是它的函数, 不是常量 —— 见类注释 §5。
    */
    int nodeBudget = 512;
    int branchMin = 2;
    int branchMax = 5;
    int trainPlanDepth = 1;     /* Planned 标签的展开层数 (0 会退化成单步 bootstrap) */
    int labelPlanBudget = 64;   /* 采集标签时用的小预算 */
    int ttCapacity = 1 << 16;

    float gamma = 0.99f;
    float learningRate = 0.001f;
    int batchSize = 32;
    int replayEpochs = 1;
    std::size_t replayCapacity = 4096;
    int learnEveryMoves = 4;
    int targetSyncEvery = 64;
    float exploringTempRoot = 1.0f;
    float exploringTempFinal = 0.25f;
    float moeAuxCoef = 0.1f;

    /* ---- 诊断 (只读) ---- */
    long long m_leafEvals = 0;
    int m_lastDepth = 0;
    long long m_lastNodes = 0;
    long long m_ttHits = 0;

    /*
      **消融开关** (只给实验用, 不是生产旋钮): 关掉之后 3 个"规则上下文"平面照旧存在、
      但内容全是 0 —— 网络输入维度不变、随机初始化**逐位相同**, 于是"状态里有没有
      规则上下文"成为唯一变量 (对比 1440 与 1710 两种维度是**不干净**的对照:
      那同时换了输入维与随机初始化)。
      用途: 量"补全 Markov 状态"这件事对**对局比分**有没有影响, 而不是对建模有没有影响。
    */
    bool encodeRulePlanes = true;

public:
    /*
       最后两个参数是**结构**参数 (与 hiddenDim/gamma/lr 那种超参不同, 它们决定网络形状):
         backbone_      : 骨干 (见枚举)。开关 DQNAB_MOE_EXPERT_SEQ=1 的构建里,
                          `SparseMoeTb` 的专家已经是序列 Transformer (见文件顶部);
         expertHidden_  : **只对序列专家生效**的 token 宽 (<=0 时取 MOE_SEQ_TOKDIM)。
                          加在最后且带默认值 ⇒ 现有调用点 (界面/测试/bench 都是 5 个实参)
                          一字不改, 默认构建的结构与今天逐位相同。
    */
    explicit DQNABAgent(Chess &chess_,
                           int hiddenDim_ = 64,
                           float gamma_ = 0.99f,
                           float lr = 0.001f,
                           Backbone backbone_ = Backbone::SparseMoeTb,
                           int expertHidden_ = MOE_SEQ_TOKDIM);

    /* ---------------- AgentBase ---------------- */
    Step getBestMove(int color) override;
    std::string getName() const override;
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;
    float getLastTrainLoss() const override { return m_lastLoss; }

    /* ----------------------------------------------------------------
     *  自检 (界面右侧"模型自检"面板) —— 口径说明见 aiagent.h 的 selfCheckReport
     *
     *  报的是**结构 / 规格**类事实, 不是棋力。三类读数各有各的存在理由:
     *
     *   1. **表示**: 1710 维 = 19 平面 x 90 格 (14 棋子平面 + 剩余子力 / 总手数 /
     *      无吃子 / 重复 / 被将), 也就是"**完备 Markov 状态**"那一份编码 ——
     *      规则上下文在这里**可观测**, 与 DQN/PG/DQN+MCTS/SACAZ 的 90 维哈希编码
     *      正好相反 (那边走子方 / 重复进度 / 无吃子 / 被将**逐字节不可分**,
     *      见 probe_dqnmcts_aliasing [2]); 那个缺口决定的是"三次重复判和"这类
     *      **终局与回报**能不能被学出来。`encodeRulePlanes` 消融开关的当前状态
     *      也在这一行 (关掉只是把 5 个平面清零, 维度与随机初始化不变)。
     *      动作侧: 在**标准开局**这份确定性样本上验证 `规范from*90 + 规范to` 是
     *      8100 上的双射 (合法着法数 == 不同的动作下标数), 老编码那种 128 槽哈希
     *      (平均 22 个着法挤一槽) 的别名在这里不存在。
     *
     *   2. **值门控** —— 本 agent 的安全机制, 也是面板里最值钱的一行: 上次批更新
     *      前后各测一次 `netHandGap` (固定探针上的 |V − 手工锚|), 单次更新把它顶高
     *      超过容忍度就回滚权重。这里报 gap 前 -> 后、有没有回滚, 以及探针上的
     *      完整 `HandStats`。**只看 gap 会骗人**: "V 恒等于 0"的未训练网络 (TB 骨干
     *      随机初始化实测 gap 0.0410) 比"学到一点但有偏移"的网络数字更好看, 而
     *      `corr` 对常数 V 恒为 0, 是尺度无关的判据; `anchorStd` 则说明这把尺子
     *      还剩多少信号 (探针少 + 纯随机不吃子时实测只有 0.0074, 那种读数上的
     *      "变好/变坏"是在常数上判的)。
     *
     *   3. **搜索 / 训练规格**: 深度上限、节点预算 (真正的成本控制量, 单位是
     *      **网络前向次数**)、选择性分支、标签规划预算、TT 容量、最近一次搜索的
     *      读数, 以及回放池 / 超参 / MoE 路由负载 / 参数量 / 最近损失。
     *      这些是"规格", 所以参数多、损失小都**不能**读成棋力。
     *
     *  **只读、可重复、不动棋盘** (契约见 aiagent.h: 面板会被 GUI 线程在对局中途
     *  调用, 而搜索线程正在用 `this->chess`): 全程不碰 `this->chess` —— 动作双射
     *  那一段用**新构造的**标准开局棋盘; 网络统计只读已有缓存, 本函数**不做任何
     *  前向** (TB 骨干一次前向 3.6 ms, 而面板每手"探索+预训练"之后都会被调一次)。
     * ---------------------------------------------------------------- */
    std::string selfCheckReport() const override;

    /* ---------------- 编码 / 动作 / 奖励 (rolloutFromCurrent 需要) ---------------- */
    void encodeSparse(int color, std::vector<std::uint16_t> &cells) const;
    void encodeStateFor(int color, RL::Tensor &state);
    void encodeState(RL::Tensor &state);
    static int canonicalCell(int x, int y, int color);
    static int actionIdxOf(const Step &s, int color);
    /*
       [2026-10] 与 `DQNMCTSMOETbAgent` / PPO / SAC 四支**同名同义**的那个出口
       (行为克隆的胶水层 src/bcagent.hpp 与 src/bcdqnagent.hpp 的鸭子类型契约要求它):
       "这个着法在 `color` 的规范视角下是哪个动作下标"。
       纯转发给 `actionIdxOf` (静态、无副作用、公式只有一份) ⇒ **不改变任何既有行为**;
       加它是因为 BC 的两层 (造样本 / 算读数) 都按名字调这个成员函数, 而两支 agent
       各写一份公式正是本工程反复记过的"两份会漂移的口径"。
    */
    int stepToActionIdx(const Step &s, int color) const { return actionIdxOf(s, color); }
    /* 只生成合法着法 (搜索热路径用, 不分配 8100 维掩码) */
    void legalMoves(int color, std::vector<Step*> &steps, std::vector<int> &indices);
    void getLegalActions(int color, std::vector<Step*> &steps,
                         std::vector<int> &actionIndices, RL::Tensor &actionMask);
    float computeReward(const Step &s, int color);

    /* [④] 学习口径的奖励: 界面奖励曲线取这一份 (见 AgentBase 的说明) */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }
    double materialPhase() const;
    double tempoPhase() const;
    /*
      **规则上下文** (Markov 性必需, 见 §3):
        repetitionCount() —— 当前局面在"最近 halfMoveClock 步"窗口里出现过几次
                            (与 Chess::isRepetition 同一窗口与同一判据; ≥3 就是判和)。
        repetitionPhase() —— 上面那个数 / 2 (0 / 0.5 / 1.0)。
        halfmovePhase()   —— halfMoveClock / 120 (60 回合判和的进度)。
        checkPhase(color) —— color 是否被将军 (1/0)。
      它们都是**当前状态**的函数, 不是"环境悄悄记着"的东西 —— 这是把象棋从
      "带历史的非马尔可夫过程"变成"扩状态上的严格 Markov 过程"的那一步。
    */
    int repetitionCount();
    double repetitionPhase();
    double halfmovePhase() const;
    double checkPhase(int color) const;

    /* ---------------- 评估 / 搜索 ---------------- */
    /*
     * 一次节点前向: 主干一次 -> V (标量) + 合法列的 A (稀疏头, R1 捷径) -> Q。
     * `inference=false` 时保留缓存以便反传 (训练路径用)。
     */
    void evaluateNode(const RL::Tensor &state, const std::vector<int> &legalIdx,
                      double &vOut, std::vector<double> &qOut, bool inference = true);

    /* 根节点决策: temperature<=0 贪心, >0 按 softmax(v_a/T) 采样 (自对弈探索) */
    Step selectMove(int color, float temperature = 0.0f,
                    std::vector<double> *rootValuesOut = nullptr,
                    std::vector<int> *rootIdxOut = nullptr);

    /* ---------------- 训练 ---------------- */
    /*
     * 只累积一条样本的梯度 (前向 + 反向), 不碰优化器 —— P3 的那一半。
     * 公开出来是为了能被**有限差分**验证 (test_dqnab 拿它核对 Dueling 双头的
     * 解析梯度: Q = V + A − mean_legal(A) 对 V 与 A 的偏导必须与数值梯度一致)。
     */
    void accumulateGrad(const Sample &s);
    bool learnBatch(int batchSize_, int epochs);
    double trainSelfPlay(int episodes, int maxMoves, bool verbose,
                         float tempRoot = 1.0f, float tempFinal = 0.25f);
    std::size_t replaySize() const { return m_replay.size(); }
    void clearReplay() { m_replay.clear(); }
    double lastRootValue() const { return m_lastRootValue; }
    int learnSteps() const { return m_learnSteps; }

    /* 采集一条样本的 TD 标签 (Planned 模式在落子后用目标网算; 棋盘必须停在 s') */
    double plannedLabelFromCurrent(float reward, bool done);

    /* ---------------- 手工评估锚: 引导阶梯第一级 + 门控 (设计文档 §7.3/§7.4) ---------------- */
    /*
      **手工评估与网络值的口径**: evaluate() 是**黑方视角**的, 而本 agent 的 V 是
      "走子方视角", 所以红方要取负, 再 tanh(e/3.0) 压到 (-1,1) —— 与 EVAB 的
      `pretrainFromHandEval` 标签**完全同一口径**, 这样两个 agent 的手工锚可以直接比。
      (刻意用 evaluate() 而不是 evaluatePositional(): 后者只给 PPO 的势能 Φ 用。)
    */
    static constexpr double HAND_EVAL_SCALE = 3.0;
    double handEval(int color);

    /*
      V 与手工锚在**探针局面**上的平均绝对差 (越小 = V 越准)。
      探针懒构造: 从当前局面随机走子取 count 个局面, 走完逐手回退,
      对棋盘零副作用 (本工程 agent 的硬约束; 见 §8)。
      samples<=0 表示"用当前已有的探针"。

      **警告: 这个数单独看会骗人**。手工锚 tanh(evaluate()/3) 的典型幅度只有 ±0.2
      (一个车 = tanh(0.5/3) = 0.165), 所以"V 恒等于 0"这种什么都没学到的网络
      也能拿到 gap ≈ 0.04 的好看数字 —— 实测 TB 骨干随机初始化就是 0.0410。
      要判断 V 有没有真的学到东西, 必须同时看 `netHandStats` 里的 corr (相关系数,
      对常数 V 恒为 0, 与尺度无关)。
    */
    double netHandGap(int samples = 0);

    /*
      同一批探针上的完整统计量。`corr` 是 Pearson 相关系数 —— **这才是"V 学没学到"
      的尺度无关判据**: gap 会因为 V 恰好接近 0 而虚低, corr 不会。
    */
    struct HandStats {
        int n = 0;
        double gap = 0.0;        /* mean |V − 锚| */
        double corr = 0.0;       /* Pearson(V, 锚); 常数 V ⇒ 0 */
        double anchorStd = 0.0;  /* 锚自身的标准差 = 这把尺子有多少信号 */
        double vStd = 0.0;
    };
    HandStats netHandStats(int samples = 0);
    int probeCount() const { return (int)m_probes.size(); }
    void rebuildProbes(int count, int maxPlies = 60);
    /*
      **吃子偏置**: 造探针 / 训练数据时, 如果当前有吃子着法, 就以这个概率选一个吃子
      (否则在全部合法着法里纯随机)。
      为什么必须有它 —— 这是实测逼出来的: 纯随机从初始局面走 ≤12 手, 双方几乎不可能
      吃到子, 于是所有局面的 `evaluate()` 几乎一样, **手工锚的标准差只有 0.0074**
      (≈ 常数)。在这种标签上训练网络只能学到"输出均值": 实测 corr 从 −0.50 到 −0.55
      (**什么都没学到**), 而 gap 却从 0.1698 掉到 0.0262, 看起来像进步。
      也就是说 —— 问题不在网络, 在**标签根本没有信号**。
    */
    float probeCaptureBias = 0.5f;

    /*
      导阶梯第一级 (照 EVAB 的配方): 用**手工锚**当标签有监督地回归 V。
        标签 = tanh(±evaluate()/3.0)  (走子方视角)
      随机走子造 positions 个局面, 过 epochs 遍, 每 batchSize 条做一次 RMSProp。
        pretrainTrunk=true  : 主干 + V 头一起训 (TB 骨干约 32 ms/样本, 贵)
        pretrainTrunk=false : 只训 V 头 (主干冻结; 便宜, 但只能线性读出随机特征)
      返回**调用后**状态下的 netHandGap (拿它跟 lastPretrainGapBefore() 比)。
      若训练后比训练前更差 (超过 pretrainTolerance), 整段回滚 (主干 + V 头) 并置
      lastPretrainRolledBack() = true, 返回值是回滚**之后**实测的 gap。
      棋盘: 内部快照 / 逐字段还原, 返回时与调用前一致 (含 history / halfMoveClock /
      sideToMove)。这一条是硬约束 —— 训练入口也不许动调用方的棋局。
    */
    double pretrainValueFromHand(int positions, int maxPlies = 60,
                                 int batchSize = 32, int epochs = 2,
                                 bool pretrainTrunk = true, bool verbose = false);
    bool lastPretrainRolledBack() const { return m_lastPretrainRollback; }
    /* 预训练前后在同一批探针上的完整统计 (gap + corr) —— 只看 gap 会骗人, 见上 */
    const HandStats &pretrainStatsBefore() const { return m_preStatsBefore; }
    const HandStats &pretrainStatsAfter() const { return m_preStatsAfter; }
    /*
      被回滚**之前**那一次尝试的统计。回滚会把 `pretrainStatsAfter` 复原成 before 的值
      (因为权重确实退回去了), 于是"它到底试成什么样"就看不见了 —— 而这个数恰恰是诊断
      "样本量够不够"的关键 (实测: MLP 512/2048 局面时它在变差, 8192 局面时 corr 0.433→0.869)。
    */
    const HandStats &pretrainStatsRejected() const { return m_preStatsRejected; }
    double lastPretrainGapBefore() const { return m_pretrainGapBefore; }
    /* 最后一次预训练的耗时 (ms) —— 报成本用, 免得"训了个更准的 V"变成不可复现的传说 */
    double lastPretrainMs() const { return m_pretrainMs; }

    /*
      门控: 每次批更新前测一次 netHandGap, 更新后**再测一次**, 单次更新把 gap 顶高
      超过 `valueGateTolerance` 就回滚权重 (EVAB 的 agent_evab.cpp:1014 那套机制)。
      数据侧给不出这种保护 —— 回放池里只有稀疏的 TD 信号, 而手工锚是**稠密的**先验,
      它坏掉会立刻显形。代价: 每次批更新多 2 次探针前向 (TB 骨干 12 局 ≈ 72 ms)
      + 一次权重快照 (≈20 ms)。

      **0.25 这个阈值是量出来的, 不是拍的**: 一开始照抄 EVAB 的 0.05, 结果
      test_dqnab [5] 的固定目标学习被整体冻结 (60 次更新后 Q 一动不动,
      误差 0.909 -> 0.909) —— 0.05 在"每次更新本来就会正常偏离手工锚"的学习面前太紧。
      门控的正确语义是"**单次**更新不许把 V 打飞", 不是"V 不许离开手工锚"
      (否则 V 永远不可能变得比手工锚更好, 而那正是 RL 存在的意义)。
      实测的空隙: clipGrad 下默认 lr=0.001 的单次更新只让 gap 动 1e-3~1e-1 量级;
      而毒化更新一次能把 gap 从 0.03 顶到 0.63 (test_dqnab [9] 的对照段)。
    */
    bool valueGateEnabled = true;
    float valueGateTolerance = 0.25f;
    /*
      探针个数。**12 个太少** —— 实测"12 个随机局面"上的锚标准差在不同 seed 下从
      0.0365 到 0.2228 (差 6 倍), 于是"这次更新让 gap 变差了吗"这个问题是在噪声上判的
      (TB 骨干的一次预训练就因此被误判成"更差"而回滚)。24 个是"每次批更新多 2 次前向"
      与"尺子够稳"之间的折中 (TB 骨干 24 局 ≈ 160 ms/次检查)。0 = 用探针缓存里的全部。
    */
    int valueGateProbeCount = 24;
    /*
      阈值**按步长缩放**。理由在 clipGrad 里: `Optimize::RMSProp` 把每个张量的梯度
      整成单位 L2 范数 (optimize.h:52 `dw /= dw.norm2()+1e-8`), 于是每次更新每个张量的
      位移 ≈ lr —— 也就是说"一次更新合法能顶多高的 gap"与 lr 成正比, 固定阈值必然在
      大 lr 下失效。**这是实测出来的**: 阈值固定 0.25 时, test_dqnab [5] (lr=0.02)
      的 60 次正常更新里 59 次被回滚 (Q 只动了一次), 而把阈值按 lr 放大后同一测试
      一次都不回滚。lr <= 0.001 (默认) 时不做放大, 保持绝对阈值的语义。
    */
    double gateToleranceNow() const
    {
        const double scale = (learningRate > 0.001f)
                                 ? (double)learningRate / 0.001 : 1.0;
        return (double)valueGateTolerance * scale;
    }
    /*
      预训练的容忍度另算, 而且**更紧**: 那里比的是"同一批稠密标签的拟合误差",
      变差没有任何正当理由 —— 与在线更新那种"正常的偏离"不是一回事。
      0.01 的来源: `bench_dqnab_vs_ab --pretrain=512 --pretrain-epochs=2` 在 **TB 骨干**
      上实测 gap **0.0411 -> 0.0814** (37.45 M 参数的主干只喂 1024 个有监督样本, 学不动
      反而被扰动) —— 0.05 的阈值放过了这 0.040 的退步, 0.01 会把它整段拦下来。
      MLP 骨干上同一段预训练是 0.1698 -> 0.0262 (大幅改善), 不受这个阈值影响。
    */
    float pretrainTolerance = 0.01f;
    bool lastUpdateRolledBack() const { return m_gateRollback; }
    double lastGapBefore() const { return m_gapBefore; }
    double lastGapAfter() const { return m_gapAfter; }

    /*
       把一条样本推进回放池 (满了按 FIFO 丢最老的)。
       公开是为了让测试能**直接构造一批固定经验** (例如"已知 TD 目标"的收敛检查),
       而不必为了造数据去跑一整局自对弈。
    */
    void pushSample(Sample &&s);

    /* ---------------- 存取 (prefix_trunk / _v / _a) ---------------- */
    bool saveModel(const std::string &prefix);
    bool loadModel(const std::string &prefix);

    /* ---------------- 稀疏 MoE 诊断 ---------------- */
    int moeExpertCount() const;
    int moeTopK() const;
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();
    long long trunkParamCount() const { return m_trunk.paramCount(); }
    long long headParamCount() const { return m_vHead.paramCount() + m_aHead.paramCount(); }

    /*
       三个**在线**网络是公开的 (与 RL::PPO 的 actorP/critic 同一做法):
       "Dueling 双头的解析梯度 vs 有限差分"这条断言必须能直接看头的权重与梯度缓冲,
       否则就只能靠"训练看起来在动"下结论。
       目标网 (m_trunkT/m_vHeadT/m_aHeadT) 保持私有: 它们只由 syncTarget 内部管理。
    */
    RL::Net m_trunk, m_vHead, m_aHead;

    /*
      公开出来是为了让测试能**对着符号下断言** (test_dqnab [10]):
      把 A 头清零 + V 头钉成 +1 之后, `computeTarget` 必须返回 `r − γ` 而不是 `r + γ`
      —— "双人零和要用 negamax, 不能照抄单 agent DQN 的 max_a Q" 这条在代码里
      只体现在一个负号上, 而负号写错照样训练、照样不报错。
    */
    double computeTarget(const Sample &s);

private:
    /* ---- 目标网: 与在线网同构, withGrad=false ---- */
    RL::Net m_trunkT, m_vHeadT, m_aHeadT;
    RL::Net buildTrunk(bool withGrad) const;
    /*
       [2026-10] 造"MoE + 序列 Transformer 专家"那一层 —— 开关打开时 `Backbone::SparseMoeTb`
       这一支的专家就是它 (与 SAC 支的 makeTbExpertMoe 同一形态: 骨干身份不变, 专家类型由
       宏决定)。返回 `iLayer::sptr` 而不是具体类型, 是因为具体类型要 rl/sparse_moe.hpp
       —— 那只需要在 .cpp 里, 头文件不必多暴露一层。
    */
#if defined(DQNAB_MOE_EXPERT_SEQ) && DQNAB_MOE_EXPERT_SEQ
    RL::iLayer::sptr makeSeqExpertMoe(bool withGrad) const;
#endif
    RL::Net buildValueHead(bool withGrad) const;
    RL::Net buildAdvantageHead(bool withGrad) const;
    void copyOnlineToTarget();

    /* 搜索用哪一套网络 (nullptr = 在线)。用 const 指针 + const_cast 调用 forward。 */
    const RL::Net *m_selTrunk = nullptr, *m_selV = nullptr, *m_selA = nullptr;

    /* ---- 搜索状态 ---- */
    struct TTEntry { int depth; double value; int flag; };
    std::unordered_map<unsigned long long, TTEntry> m_tt;
    RL::Tensor m_stateBuf;
    long long m_nodes = 0;
    double m_lastRootValue = 0.0;

    double negamax(int color, int depth, double alpha, double beta, int ply);
    int effectiveBranch(std::size_t legalCount) const;
    void clearSearchState();

    /* ---- 回放 / 训练 ---- */
    std::deque<Sample> m_replay;
    RL::Tensor m_dV;
    RL::Tensor m_dA;
    float m_lastLoss = std::numeric_limits<float>::quiet_NaN();
    double m_batchLossSum = 0.0;
    std::size_t m_batchCount = 0;
    int m_learnSteps = 0;
    /* pick 回调算出的合法集, 供紧接着的 onTrans 复用 (与 SACAZ 的 m_pendingMask 同一手法) */
    std::vector<int> m_pendingLegal;

    void trimReplay();

    void applyGradients(float lr);
    void resetMoeBatchStats();

#if defined(DQNAB_MOE_EXPERT_SEQ) && DQNAB_MOE_EXPERT_SEQ
    /*
       权重前缀的**准入检查** (saveModel / loadModel 进门第一件事, 见实现里的说明)。
       两件事一起做:
         1. 记下这次**实际用的**前缀 —— 自检面板报的是它, 不是"应该用的"那一个 (读数不许撒谎);
         2. 开关打开时**不允许碰 TB 那一份前缀** (`weights/dqnab_agent`, 与
            src/chessboard.cpp:5979 的 `defaultWeightPath(AGENT_DQNAB)` 逐字相同):
            那是另一个结构 (TB 专家) 的文件。载入它会被 `Net::load` 的参数结构守卫拒绝
            (还算看得见), 但**写出**它会把盘上那份 TB 权重直接盖掉 —— 写出侧没有守卫,
            所以只能在这里当场拦住 (返回 false = 调用方看得见的"失败", 退出保存会把它
            打进日志; 而不是"报一句就继续写")。
       返回 false = 拒绝这次存取 (一个字节都不动)。自定义前缀 (测试的 `test_dqnab_tmp`、
       后台训练的 `weights/_temp_train_dqnab`) **不受影响** —— 那些是同一进程内自己写自己读
       的临时文件, 没有"两个结构共用一份文件"的风险。
       只在开关打开时存在 ⇒ 默认构建的成员布局与所有行为一位不动。
    */
    bool canUseWeightPrefix(const std::string &prefix);
    std::string m_lastWeightPrefix;
#endif

    /* ---- 手工锚 / 门控 (实现见 dqnabagent.cpp 的 "手工评估锚" 一节) ---- */
    struct Probe { RL::Tensor state; float target = 0.0f; };
    std::vector<Probe> m_probes;                    /* 固定探针集合 = 训练前后同一把尺子 */
    void buildProbeSet(std::vector<Probe> &out, int count, int maxPlies);

    /*
       棋盘快照。造数据要 reset() 走随机局, 而 reset() 会清掉 history / sideToMove /
       halfMoveClock 与所有棋子位置 —— 不还原就等于"训练一次顺手把调用方的棋局吃了"。
       (棋子对象本身不重建, 所以 m_children / m_map 里的指针仍然指向它们。)
    */
    struct BoardSnap {
        struct S { int id, type, color, alive; double value; int x, y; };
        std::array<S, 32> stones;
        int sideToMove = 0;
        int halfMoveClock = 0;
        std::vector<Chess::HistoryRecord> history;
    };
    void saveBoard(BoardSnap &s) const;
    void loadBoard(const BoardSnap &s);

    /*
       权重快照。**必须**另建同构网络再 copyTo: RL::Net 的赋值是浅拷贝 (共享层指针),
       `RL::Net bak = m_trunk;` 只是给同一个主干起了个别名 —— 回滚时"备份"跟着一起变了,
       而且这种失效是静默的 (回滚语句照样执行, 只是回到被污染后的值)。
    */
    RL::Net m_trunkBak, m_vHeadBak, m_aHeadBak;
    bool m_bakReady = false;
    void snapshotWeights();
    void restoreWeights();

    /* 门控 / 预训练的状态 (只读出口在 public 段) */
    bool m_gateRollback = false;
    double m_gapBefore = 0.0, m_gapAfter = 0.0;
    bool m_lastPretrainRollback = false;
    double m_pretrainGapBefore = 0.0, m_pretrainGapAfter = 0.0;
    HandStats m_preStatsBefore, m_preStatsAfter;
    HandStats m_preStatsRejected;    /* 回滚前那一次尝试的统计 (见访问器注释) */
    double m_pretrainMs = 0.0;
};

#endif // DQNAB_AGENT_H
