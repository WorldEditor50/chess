#ifndef DQNMCTS_MOETB_AGENT_H
#define DQNMCTS_MOETB_AGENT_H

#include <vector>
#include <string>
#include <deque>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>

#include "chess.h"
#include "aiagent.h"
#include "rl/net.hpp"
#include "rl/layer.h"
#include "rl/loss.h"
#include "rl/util.hpp"

/* 稀疏 MoE 只需要指针/引用 (定义在 rl/sparse_moe.hpp, 由 .cpp 包含) */
namespace RL {
class ISparseMoE;
}

/*
 * ================================================================
 *  DQNMCTSMOETbAgent — DQN + MCTS, 骨干 = 稀疏 MoE(TransformerBlock 专家)
 * ================================================================
 *
 * **[2026-09 新分支 dev-dqnmcts-moetb] 本类是 `DQNMCTSAgent`(AGENT_DQNMCTS) 的
 *  "把 SAC 那条线上已经被实测过的优化搬过来" 的那一支**, 它是一个**独立类**
 * (`src/dqnmctsmoetbagent.{h,cpp}`, 界面类型 `AGENT_DQNMCTS_MOE`), 与
 * `DQNMCTSAgent` **互不继承、互不包含** —— 旧的那一支(以及它的权重文件
 * `weights/dqnmcts_agent.dat`)一位不动, 两者可以在界面上直接对弈比较。
 *
 * 为什么是独立类而不是就地改 (用户口径, 与 `SACAZMoETbAgent` / `SACAZLegacyAgent`
 * 那两次拆分同一套理由):
 *   * **默认值是结论**: 旧类的每一次读数 (损失曲线 / 自检面板 / 权重文件) 都建立在
 *     它的口径上。把骨干、表示、搜索选择公式一次性换掉会**静默**改掉所有历史读数,
 *     而且换完之后再也没有"改前"可以对照。
 *   * **隔离是结构性的**: "改这一支不许碰那一支" 从 git diff 的自证变成事实 ——
 *     两个类在两个文件里。
 *
 * ---- 它到底把 SAC 的哪些优化搬过来了 (逐条, 附实测出处) ----
 *
 *  1. **稀疏路由 MoE + TB 专家** (`SparseMoE<TransformerBlock<...>,4,1>`) 取代旧类的
 *     **稠密** `MOE<16,16>`(16 个专家全算). 实测 (`bench_moe`): 同参数下稀疏路由
 *     **3.8x** 快 (9.918 vs 37.675 ms/sim). 旧类那一支的稠密 MoE 从来没有"按需路由"
 *     的机会, 它每次叶子估值都把 16 个专家 + 1 个独立 TransformerBlock 全算一遍。
 *  2. **TB 专家的头数口径** (`HonorHeads=true`, 见 rl/attention.hpp 的第三模板参数)。
 *     SAC 那边实测的静默降级: `d_model = 1263 = 3 x 421`(421 是素数) 而老的
 *     "头数必须整除 d_model" 规则把请求的 15 个头降成 **3** 个、d_k 从 84 涨到 421,
 *     一个专家前向 6.20 -> 19.06 ms (**3.07x**), 而且 15 个 head 对象全都分配了、
 *     只有 3 个参与前向 (80% 张量是死的)。本类的 `MOE_TB_HEADS = 15` 与
 *     `STATE_DIM = 1263` 是**同一对**数字, 所以这条修复是必需的, 不是可选项。
 *     `selfCheckReport` 把"请求几个头/实际用几个/每个多宽/注意力元素数"印出来 ——
 *     这类静默替换**只能靠读数发现** (参数指纹/paramCount/权重格式一个都不变)。
 *  3. **共享骨干 (一个骨干 + 头)**: `trunk` (稀疏 MoE + Tanh) 与 `qHead` 是分开的
 *     两个 `RL::Net`, 它们与完整视图 `qNet` **共享同一批层对象**。于是:
 *       * 搜索叶子 = **骨干前向一次** + 只在合法列上算输出头 (`sparseLogits`);
 *       * 训练每个样本 = 3 次骨干前向 (s 在线 / s' 在线 / s' 目标) + 3 次很小的头;
 *     旧类那一支的 `RL::DQN` 把骨干与头缝在一张网里, 于是每次叶子估值都要把
 *     128 列 Q 全算一遍, 而一个局面只有 ~40 个合法槽位。
 *     **语义不变、逐位不保证** (同一个 h 被复用, MoE 门控与专家中间量不再重算)。
 *  4. **稀疏叶子估值** (`Net::forwardTrunk` + `iLayer::sparseLogits`, 即 PPO 的 R1):
 *     只算合法槽位那几列。`sparseLeafEval=false` 时退回全量口径 (A/B 用)。两条路径
 *     的等价性由 test 的 [3] 逐元素钉住 (容差 1e-6)。
 *  5. **PUCT + 先验, 且符号按 negamax**. 旧类的 `getUCB1` 直接用
 *     `+child.totalReward/visitCount` —— 而 `totalReward` 是**逐层翻号**写进去的
 *     (backprop 里 `reward = -reward`), 所以子节点的价值是**对手视角**的, 父节点
 *     比较时必须取负。漏掉负号的后果 SAC 那边有完整记录 (搜索专挑对自己最差的着法,
 *     "评估越准错得越狠", 症状是不敢吃子), 见 `docs/session_2026_09_sac.md`。
 *     本类: `double q = -child.getQ(); return q + c_puct*P*sqrt(N)/(1+n);`
 *     (与 `SACAZMoETbAgent::getPUCT`、`PPOMCTSAgent::getPUCT`、`MCTS::getUCB1`
 *     四处同一口径 —— 工程纪律: 这几个 agent 的符号约定必须一致。)
 *     先验 P(s,a) 在**合法槽位**上用 Q 的 softmax 给出 (温度 `priorTemp`): DQN 没有
 *     策略头, 所以"先验"只能来自它自己的价值排序 —— 与 AlphaZero 的 P 角色相同,
 *     只是来源不同 (这条是本类与 SAC 的**唯一**算法差别, 见下面的"没有搬什么")。
 *  6. **叶子终局只走一个出口**: `terminalReward(result, mover)`, 与训练目标用同一个
 *     函数 (工程纪律 "搜索侧与学习侧同口径")。旧类的 `evaluateLeaf` 用的是
 *     `chess.isGameOver()` —— 它**只认"将/帅还在不在场上"**, 将杀/困毙/三次重复/
 *     60 回合自然限着一概不算终局, 而那是 2026-09 "Phase 6 终局口径统一" 唯一漏掉的
 *     一处 (探针 `probe_dqnmcts_aliasing` §[5] 专门记录了它)。
 *  7. **Double DQN** (选动作与估价值分开): `a* = argmax_{合法} Q_online(s',a)` 但用
 *     `Q_target(s',a*)` 估值。旧类是 vanilla DQN (argmax 与取值都来自目标网), 而
 *     SAC 的 `min(Q1,Q2)` 要解决的是**同一个** max 偏置问题 —— 这里用不增加任何参数的
 *     那一半。`doubleDQN=false` 可退回 vanilla (A/B 用)。
 *  8. **目标钳位 + Huber** (`clampTarget=2.0` / `huberDelta=1.0`, 与 SAC 同为默认):
 *     只夹**目标**不夹奖励 (不改变各着法的排序, 只挡住发散), 损失数值走 Huber、
 *     梯度仍走 MSE (`RL::Loss::MSE::df`) —— 这一对在 SAC 那边是有意这么写的,
 *     注释里专门防"顺手统一成纯 MSE"。旧类那一支的 Q 头是无界 `Linear`, 而
 *     `RL::DQN` 既不夹目标也不用 Huber, 正是 SAC 实测发散 (|Q| 0.063 -> 4.15 -> 13.4)
 *     的同一套结构。
 *  9. **梯度累积 + 每批只调一次优化器** (PPO 的 P3 / SAC 的 `learnBatch`):
 *     批内每个样本反向累积, `RMSProp` 在批末调**一次**。注意语义: `clipGrad=true`
 *     会把每个张量归一成单位长度 (`dw /= |dw|`), 所以"梯度乘常数"是空操作,
 *     累积改变的是**方向** (沿 batchSize 条样本梯度之和走一步, 而不是走 N 步)。
 * 10. **多 epoch 回放** (`replayEpochs`, P4): 每个 epoch **重新抽** batchSize 条,
 *     梯度累积、优化器仍只调一次 ⇒ 一次更新看到 batchSize×epochs 条经验。
 *     **不是** "把同一批复用几遍": 批内权重不变, 重复遍的梯度逐位相同, 而 clipGrad
 *     会把它归一掉 —— 那只是白烧算力。默认 1 (与旧类每批的更新量一致), 自对弈路径
 *     想开就设 2。
 * 11. **目标网同步率是成员, 且默认取"机制上确定修好的"那一档**
 *     (`targetTau=1.0` 硬拷贝 + `replaceTargetIter=64`)。SAC 的 F1 诊断: 老的
 *     `tau=1e-3` + 每 64 步 ⇒ 20 局后目标网只挪动 2~4%, `|Q_target|` 仍停在随机
 *     初始化尺度 (0.072~0.097), 自举项等于 0, TD 目标退化成常数。旧的 `RL::DQN` 是
 *     `tau=0.01` + 每 256 次 learn (≈ 每 1024 手) —— **同一类缺陷**。
 *     ⚠ 但要说清楚: SAC 那边"把目标网动起来"的**棋力提升不显著**
 *     (67.3% vs 65.0%, Fisher p=0.43), 所以那一边**刻意没改默认值**。
 *     本类是**新类**, 没有需要保护的历史读数, 于是默认取修好的那一档,
 *     并把 `targetTau/targetReplaceIter` 留成成员供 A/B。
 * 12. **MoE 负载均衡辅助损失的批边界**: `resetMoeBatchStats()` 在每批**开头**调,
 *     `addAuxGradient(auxLossCoef)` 在主反向之后、优化器之前调**一次**。
 *     为什么必须有这个边界: `addAuxGradient` 是按"自上次调用以来所有 forward"的
 *     均值算的, 而搜索期间每次模拟都跑一次骨干前向 —— 不划边界的话, 一次更新的
 *     辅助损失会被整局棋的推理前向稀释 (还有 `xSum` 这种 float 累加器的精度损失)。
 *     `auxLossCoef=0.1` 是 SAC 那边实测选出来的 (0 会让专家坍缩: 8 个专家里 3 个
 *     从不被选中; 0.5 以上开始盖过真正的策略梯度)。
 * 13. **稀疏回放表示**: 一条经验只存"被占据的格子" (`plane*90+cell`, uint16) 与
 *     3 个上下文标量 + 下一局面的合法槽位位图 (2×uint64), 而不是 1263×2 个 float。
 *     1263 维稠密存法 ≈ 10 KB/条, 稀疏 ≈ 100 B/条 (旧类存的是 90 维稠密, 更小,
 *     但它同时丢掉了走子方与全部规则上下文)。
 *     上下文**必须随身携带**: 不存它, 展开时重建出来的状态就是"裸棋盘", 编码静默退回。
 * 14. **外部终局通道** (`AgentBase::notifyGameResult`): 人机对弈里"结束这一局的
 *     那一手"是**人**走的, 学习器看不到终局 —— AI 输掉的那一局拿不到 −1, 最后一条
 *     决策样本的 done 仍是 false。本类保留真实决策样本 (`Sample::decision`), 于是
 *     终局时能**回填** done + 终局值 (幂等)。见 `docs/` 里 66ab33c 那次修复的说明。
 * 15. **只读诊断 (自检面板)**: 这是 SAC 那条线上唯一"每次都真的抓到东西"的东西 ——
 *     本工程每一次静默失效都是被一个读数抓到的, 不是被测试抓到的。本类报告:
 *     骨干指纹 (专家数/topK/参数/注意力头请求数 vs 实用数)、表示 (状态构成/上下文
 *     通道/动作别名)、搜索 (sims/c_puct/叶子路径/PUCT 符号口径/叶估值次数)、
 *     学习 (batch/池/epochs/目标钳位/夹住比例/|y|/|Q|/Q spread/done 样本/MoE 路由
 *     直方图)、终局通道 (一局恰好一条 + 旧口径 `isGameOver` 漏了多少)。
 *
 * ---- 刻意**没有**搬过来的 (连同理由, 免得下次再想一遍) ----
 *
 *  * **8100 双射动作空间** (PPOMCTS/DQNAB/对齐版 SAC 用的那个): SAC 在**同一数据量**
 *    下实测净亏约 200 Elo (200 局: 71.8% vs 54.8%, 区间不重叠), 机制是 Q 头从
 *    128 列涨到 8100 列 (参数量 63 倍) 而瓶颈宽度不变 —— 参数/数据比失衡。
 *    所以本类沿用 128 槽哈希, 别名问题交给自检面板如实报告 (`aliasMoves` 等)。
 *  * **最大熵 / 温度 α / 熵项进目标**: 那是 SAC 的 actor 那一半, DQN 没有策略头,
 *    没有"目标熵"可谈。本类的探索是 ε (rollout) + 访问分布的 N^(1/T) 采样 (搜索)。
 *  * **soft value / 熵中心化 / α 上界**: 同上, 无 α。
 *  * **tanh 输出 (criticTanh) / tanh 奖励 / α Gumbel**: SAC 那边实测三条都更差
 *    (Q spread 0.1945 -> 0.0629 / 恒等 / 夹住比例 6.4% -> 23.7%), 所以不搬。
 *  * **奖励塑形 (rewardShape 2/3)**: SAC 实测在新实现上不显著 (p=0.45~0.56),
 *    而且它的 mate 那一半在自对弈里是死旋钮 (decisive done 样本 0~0.14%)。
 *    本类只保留"走子方视角 + 每步代价 + 终局 ±1"这一套 (与其它 agent 同源)。
 *  * **双 critic 的 min(Q1,Q2)**: 实现了 (`twinCritic`) 但**默认关闭**。理由:
 *    它在 SAC 那边是与最大熵策略耦合着测的, 本工程从未单独量过它; 而它要解决的
 *    max 偏置已经由 Double DQN 直接处理 (不增加参数)。打开它 = 一条 A/B 臂,
 *    不是默认口径 —— 见 `test/bench_dqnmcts_moe_main.cpp --twin=0/1`。
 *
 * ---- 权重文件 ----
 *   前缀 `weights/dqnmcts_moe_agent`, 两个文件: `_trunk` 与 `_q`.
 *   为什么必须与旧类的 `weights/dqnmcts_agent.dat` 分开: 两者**参数量与结构都不同**
 *   (本类的结构指纹会让交叉载入当场失败), 但更重要的是命名上必须能回答
 *   "这是哪一支的权重" —— 共用前缀 = 后训练的那一支静默覆盖另一支。
 *   ⚠ 状态维度变化 (90 -> 1263) 会让**旧类的权重文件**在本类上载入失败, 这是
 *   期望行为 (Net::load 有参数量守卫), 不是 bug。
 *
 * ---- 已知代价 (实测, 不藏着) ----
 *   一个 TB 专家前向在 d_model=1263 上约 5~6 ms (SAC 那边实测 6.20 ms/专家前向,
 *   共享骨干下 4.02 ms/sim 含 π+双 Q), 而旧类在 d_model=90 上的稠密 MoE 前向是
 *   亚毫秒级。所以**同一个 MCTS 预算下本类每步都更慢** —— 它的意义是"同样的时间
 *   预算里做更强的估值", 而不是"同样的模拟次数更快"。界面上的模拟次数因此按
 *   `DQNMCTS_MOE_SIMS` 那一档给 (40), 不是旧类的 200。
 */
class DQNMCTSMOETbAgent : public AgentBase
{
public:
    /* ================================================================
     *  表示: **SAC 的默认口径** (14 子力平面 + 3 规则上下文标量)
     * ================================================================
     *  棋子平面 = 14 (7 类棋子 x {己方, 对方}), 轮到谁走谁的子就在 x 大的那一侧
     *  (规范视角, `ChessState::canonicalCell` 的镜像只有一份实现)。
     *  上下文 3 个 = [无吃子进度, 重复次数, 是否被将] —— 顺序与数值口径就是
     *  `src/chessstate.h` 的 CTX_HALFMOVE / CTX_REPEAT / CTX_CHECK。
     *
     *  为什么这三项是**必需**而不是锦上添花: 三次重复判和 / 60 回合自然限着 /
     *  长将循环都依赖历史, 而它们**决定终局与回报**。少了它们, "同一局面的第 2 次
     *  出现"与"第 3 次出现 (立刻判和)"编码成同一个向量, 于是 V(s) 不是 s 的函数,
     *  Bellman 备份的前提 (P(s'|s,a) 只依赖 s) 直接失效。
     *
     *  与旧类的 90 维编码的两处**具体**差别 (不是"更好看", 是两件错事):
     *   1. 旧类没有走子方通道 —— 同一个棋盘红先与黑先逐字节不可分
     *      (`probe_dqnmcts_aliasing` §[2] 实测);
     *   2. 旧类的 `PIECE_VALUES[7] = {1..7}` 是按 `Stone::Type` 的**序数**取的:
     *      车(CHE=0)= 1/7, 兵(BING=3)= 4/7, 帅(JIANG=4)= 5/7 —— 而真实价值是
     *      车 0.5 / 兵 0.1 / 帅 "被吃即终局"。也就是说**子力顺序是反的**
     *      (兵比车"值钱"5.7 倍)。规范平面的 one-hot 编码没有这个问题。
     */
    static constexpr int CELLS = 90;                        /* 10 行 x 9 列 */
    static constexpr int PIECE_PLANES = 14;                 /* 7 类棋子 x {己方, 对方} */
    static constexpr int CTX_COUNT = 3;                     /* 无吃子 / 重复 / 被将 */
    static constexpr int CTX_BASE = PIECE_PLANES * CELLS;   /* 1260: 上下文标量的起点 */
    static constexpr int STATE_DIM = CTX_BASE + CTX_COUNT;  /* 1263 */
    /*
     *  动作: 与 SAC 默认口径**逐字同一个** 128 槽哈希 (见 .cpp 的 stepToActionIdx)。
     *  已知局限: 同局面的不同着法会撞进同一个槽位 (实测标准开局 44 -> 39 槽),
     *  所以它是一条**结构性上限**, 训练多少次都消不掉 —— 自检面板如实报告撞掉多少。
     */
    static constexpr int ACTION_DIM = 128;

    /* ================================================================
     *  骨干几何 (与 SACAZMoETbAgent 的 TB 那一支**同一组数字**)
     * ================================================================
     *  稀疏 MoE: E = 4 专家, top-1 (算力只算 1 个, 参数背 4 个)。
     *  专家 = TransformerBlock<15, 315, HonorHeads=true>:
     *     15 个头, d_k = 1263/15 = 84 (截断, 余下的坐标显式置零);
     *     FFN 宽度 315 = d_model/4 (压住 TB 专家的 FFN 开销)。
     *  `HonorHeads=true` 是 SAC 那条线上实测的**修复**, 不是可选项 ——
     *  老的整除规则会把 15 个请求头静默降成 3 个 (d_k 421) 并慢 3.07 倍。
     *
     *  `denseMoe` (构造参数) 是**同一个骨干的等参数对照组**: 4 个专家全算
     *  (TopK == E) —— "等参数不等算力", 用来量稀疏路由本身值多少。
     *  ⚠ 它**必须是构造参数**: 建网发生在构造函数里, 构造之后再赋值是**静默空操作**
     *  (SAC 那条线的同类坑: `moeDense` 写成普通成员, 报告印"稠密"而实际跑的是稀疏)。
     */
    static constexpr int MOE_TB_EXPERTS = 4;
    static constexpr int MOE_TB_TOPK = 1;
    static constexpr int MOE_TB_HEADS = 15;
    static constexpr int MOE_TB_DFF = 315;

    /* 本类骨干的**名字** (自检面板第一行要能回答"我是哪一支") */
    const char *backboneName() const;
    /*
     *  隐层激活的**真实类型名** (读的是 qNet 第 2 层的 layer 类型, 不是开关回显)。
     *  为什么要有这个读数: 同一层激活曾经被换成 `TanhNorm<Sigmoid>` 而让随机权重下的
     *  棋力掉 26 个点, 而当时**面板上一个字都看不出来** (同形状、同参数量、不在 diff 里)。
     *  以后谁再动这一层, 这里会直接显示出来。
     */
    const char *hiddenActivationName() const;

    /* ----------------------------------------------------------------
     *  一条 off-policy 经验
     * ----------------------------------------------------------------
     *  局面**不存 1263 个 float** (那是 5 KB/条), 而是存"被占据的格子"的稀疏列表
     *  (一局最多 32 个子, 每格一个 uint16 = plane*CELLS+cell)。取用时再展开成
     *  one-hot。上下文 3 个标量**必须随身带** —— 它们不是"某个格子上的 1", 进不了
     *  稀疏格列表; 不存它们, 展开时重建出来的就是"裸棋盘"(静默失效)。
     *
     *  合法掩码存**两份** (各 2×uint64 = 16 B):
     *    * `nextMask`: Double DQN 要在 s' 的合法集上取 argmax, 没有它就会把非法着法的
     *      Q 也算进 max (那正是本工程反复说的"把非法列抬起来");
     *    * `curMask`: 只给诊断用 (自检面板的 **Q spread 只统计合法槽位** —— 全 128 列
     *      的标准差会被那些从不被走到的列稀释, 读数会假性偏大)。
     *  约定: `{0,0}` 表示"没有掩码" (终局样本 / 采样失败), 此时退回全列口径。
     */
    struct Sample {
        std::vector<std::uint16_t> cells;      /* 当前局面: plane*CELLS + cell */
        std::vector<std::uint16_t> nextCells;  /* 下一局面 (同一编码) */
        float ctx[CTX_COUNT] = {};             /* 当前局面的 3 个上下文 */
        float nextCtx[CTX_COUNT] = {};         /* 下一局面的 3 个上下文 */
        std::uint64_t curMask[2] = { 0, 0 };   /* 当前局面合法槽位位图 (诊断用) */
        std::uint64_t nextMask[2] = { 0, 0 };  /* 下一局面合法槽位位图 */
        int action = 0;                        /* 实际走的动作索引 */
        int mover = Stone::COLOR_NONE;         /* 走这一手的是哪一方 (回填终局值时要用) */
        int legalCount = 1;                    /* 当前局面合法着法数 */
        int nextLegalCount = 1;                /* 下一局面合法着法数 */
        float reward = 0.0f;                   /* 走子方视角的即时奖励 */
        bool done = false;
        /*
         *  这一条是不是"真实对局的决策样本" (相对于 rollout 里的推演样本)。
         *  外部终局通道 (notifyGameResult) 只回填带这个标记的样本: 回放池里混着
         *  探索期的推演样本, 把 done=true 挂到一个**无关局面**上比不挂更坏。
         */
        bool decision = false;
    };

    /* ================================================================
     *  AlphaZero 式 PUCT 节点 (所有价值都是**当前走棋方视角** = negamax 约定)
     * ================================================================ */
    struct AZNode {
        int parentID;
        int parentAction;
        Step step;

        int visitCount;
        double totalValue;      /* 累计价值 (当前走棋方视角) */
        double prior;           /* P(s,a) = 合法槽位上 Q 的 softmax */

        std::vector<int> childIDs;
        std::vector<int> untriedActionIndices;
        std::vector<Step> untriedSteps;
        std::vector<double> untriedPriors;   /* 与 untriedSteps 一一对应 */

        int currentColor;
        int legalCount;
        bool isTerminal;

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
     *  一次 mini-batch 的只读诊断累计量 (**不参与任何计算**)
     * ----------------------------------------------------------------
     *  为什么需要它: 训练损失与自对弈胜率都回答不了"critic 到底有没有在学排序"。
     *  这里四个读数各自对应一个已知的静默失效:
     *    clamped       : 目标被 `clampTarget` 夹住的比例 —— 夹住的样本其目标是个常数,
     *                    对 critic 的**排序**学习毫无贡献 (而搜索要的正是排序);
     *    yPreAbsMax/Sum: 夹之前的 |y| —— "目标被顶到多远"的直接读数;
     *    qSpreadSum    : 合法槽位上 Q 的**标准差**。这是 critic 真正能给搜索的东西:
     *                    PUCT 的探索项量级是 c_puct·P·√N/(1+n) ≈ 0.3~1.0, 只有
     *                    Q spread 到了同一量级, Q 才影响得了选择;
     *    qAbsMeanSum / qTargetAbsSum: 在线网与**目标网**分开报 —— 软备份用的是目标网,
     *                    只报在线网会看不出"目标网是死的"(SAC 的 F1 就是这样查出来的);
     *    doneSamples / decisiveSamples: 终局样本有多少。实测这类样本可以**恰好是 0**
     *                    (SAC 那边 12 局 11296 条里 0 条), 那时"给杀将加权"这类旋钮
     *                    是死代码, 而曲线上一片正常。
     */
    struct BatchDiag {
        long long n = 0;
        long long clamped = 0;
        double yPreAbsSum = 0.0, yPreAbsMax = 0.0, ySum = 0.0;
        double vNextSum = 0.0;
        double qSpreadSum = 0.0, qAbsMeanSum = 0.0, qTargetAbsSum = 0.0;
        double slotSum = 0.0, legalSum = 0.0;
        long long doneSamples = 0, decisiveSamples = 0;
        long long doubleDQNSamples = 0;
    };

    /* ----------------------------------------------------------------
     *  公开成员 (与 PPOMCTSAgent / SACAZMoETbAgent 保持一致的风格)
     * ---------------------------------------------------------------- */
    Chess &chess;

    /*
     *  三个**视图**共享同一批层对象 (见 .cpp 的 buildNets):
     *    trunk = [稀疏 MoE, Tanh]      ← 骨干 (搜索/训练都只前向它一次)
     *    qHead = [Linear h->ACTION]    ← 输出头 (只有它需要"只算合法列")
     *    qNet  = [稀疏 MoE, Tanh, Linear] ← 完整视图, 只为 `backward` 存在
     *  目标网是同构的三张 (withGrad=false)。
     *  ⚠ 纪律: **热路径不要用 qNet.forward()** —— 它会把骨干重算一遍, 静默吃掉
     *    "骨干只前向一次"的全部收益 (结果还是对的, 所以不会有任何报错)。
     */
    RL::Net trunk, qHead, qNet;
    RL::Net trunkTarget, qHeadTarget, qTargetNet;
    /*
     *  双 critic 的第二套头 (`twinCritic` 打开时才有梯度)。
     *  为什么要**无条件**造出来: 建网在构造函数里, 所以"构造之后再打开 twinCritic"
     *  如果依赖一张还不存在的网, 那就又是一个"静默空操作"的开关 —— 本工程栽过两次
     *  (`legacyNet` / `moeDense`)。多造这两个头的代价是 2 x (64x128+128) ≈ 1.6 万个
     *  参数 (骨干是 2870 万), 换来的是"这个开关在既有实例上真的生效"。
     */
    RL::Net q2Head, q2Net;
    RL::Net q2HeadTarget, q2TargetNet;

    /* ---- 结构回显 (只读; 建网参数的真实取值) ---- */
    int hiddenDim;
    bool denseMoe;

    /* ---- 学习口径 ---- */
    float gamma;
    float learningRate;
    int batchSize;
    std::size_t maxMemorySize;
    /*
     *  [P4] 每个 epoch **重新抽** batchSize 条 (梯度累积, 优化器仍只调一次)。
     *  默认 1 = 与旧类每批的更新量一致; 自对弈路径想开就设 2。
     */
    int replayEpochs;
    /*
     *  [F1] 目标网步长与节拍: `targetTau >= 1` 是硬拷贝 (默认), 每 `replaceTargetIter`
     *  次 learnBatch 同步一次。**默认取修好的那一档**, 理由见类头注释第 11 条。
     *  `replaceTargetIter <= 0` 当作 1。
     */
    float targetTau;
    int replaceTargetIter;
    /* 值域约束: `clampTarget <= 0` = 不夹 (对照); `huberDelta <= 0` = 纯 MSE (对照) */
    float clampTarget;
    float huberDelta;
    /* Double DQN: 用在线网选 s' 的动作、用目标网估它的值 (默认开) */
    bool doubleDQN;
    /*
     *  双 critic 的 `min(Q1,Q2)` (SAC 的 clipped double Q)。
     *  ⚠ **默认关闭**, 见类头注释 "刻意没有搬过来的" 最后一条: 它是 A/B 臂。
     *  打开时多一个 q2 头 (h->ACTION 的一层, 很小) 与其目标副本, 骨干不变。
     */
    bool twinCritic;
    /*
     *  每走多少手做一次 learnBatch (1 = 每一手都学)。
     *  **默认 8 是实测定的, 不是抄来的**:
     *    * 本机一个训练样本 = 3 次骨干前向 + 1 次骨干反向 = **43.4 ms**
     *      (test_dqnmcts_moe 的 [11] 节, d_model=1263 / E=4 top-1 / 28.77 M 参数);
     *    * 一次决策 (40 次模拟) = **134.8 ms**;
     *    * 于是 batchSize=32 的一个批 = **1.39 s**。界面每手的时间预算就是"一次搜索"
     *      那个量级, 所以把它摊到 8 手上 = 每手均摊 174 ms ≈ 一次搜索。
     *  旧类 (DQNMCTSAgent) 用的是 4 —— 那个数是在 d_model=90 的稠密 MoE 上定的,
     *  照搬过来会让每手多背 350 ms。要更快就把它调大, 要更多更新就调小 (代价线性)。
     */
    int learnEveryMoves;
    /*
     *  **从自己的搜索学一次** (SAC 的 `learnFromSearch`, 默认开):
     *  每次真实决策之后, 把 (s, a, r, s', done) 存进回放池 —— 也就是让**真的走过的那一手**
     *  进训练回路, 而不是只有探索期的推演样本。
     *
     *  三件事同时靠它:
     *   1. 界面上的决策路径 (`preTrainThenDecide` 的探索) 可以整个关掉, 而"每一手仍然在
     *      学"这件事仍然成立 —— 旧类那一支关掉探索就**完全不学**了;
     *   2. 外部终局通道 (notifyGameResult) 需要一个"真实决策样本"来挂终局值, 而这条样本
     *      就是它;
     *   3. 它自己那一手将死对方时, 终局 ±1 当场就写进了目标 (不用等下一次通知)。
     *
     *  工程纪律: GUI 的对弈模式 (评估 / 只对弈) 要能把它关掉 —— 界面用
     *  `finishDecisionSearch` 在一手之内关掉再还原 (与 SAC 两支同一处闸门)。
     *  关掉之后 `selectMove` **不产生任何更新** (test 的 [9] 钉住这条)。
     */
    bool learnFromSearch;
    /* 稀疏 MoE 的负载均衡辅助损失系数 (0 = 关掉; 见类头注释第 12 条) */
    float auxLossCoef;
    /* ε 探索率 (只作用于 rollout 的取动作; 在 learnBatch 里按 RL::DQN 的口径衰减) */
    float exploringRate;

    /* ---- 搜索口径 ---- */
    int simulations;          /* getBestMove 默认的模拟次数 */
    float c_puct;             /* PUCT 探索常数 (SAC/PPOMCTS 都是 1.5) */
    float priorTemp;          /* 先验温度: P = softmax_{合法}(Q / priorTemp) */
    bool sparseLeafEval;      /* 叶子估值走稀疏列 (默认 true; false = 全量对照) */
    /*
     *  自对弈时的根温度 (AlphaZero 口径): 按 N^(1/T) 采样访问分布, T 从
     *  `trainTempRoot` 线性退火到 `trainTempFinal` (前 `trainTempMoves` 手), 之后 argmax。
     *  旧类那一支的做法是"以 ε 概率随机挑一个根孩子" —— 那个孩子可能**一次都没被访问过**,
     *  它的价值与先验都没有被搜索检验过, 等于在那一步把搜索丢掉。这里换掉。
     */
    float trainTempRoot;
    float trainTempFinal;
    int trainTempMoves;

    /* ---- 在线训练 (界面/测试用) ---- */
    bool m_trainingMode;
    int m_onlineStepCount;

    /* ---- 回放池 ---- */
    std::deque<Sample> memories;

    /* ---- 搜索树 (selectMove 内清空重建; 公开是为了让测试能直接查节点) ---- */
    std::vector<AZNode> nodes;

    /* ---- 统计 ---- */
    int totalEpisodes;
    int totalWins[2];               /* [0]=red, [1]=black */
    int m_learnSteps;               /* learnBatch 被调用的次数 */
    int m_learnCounter;             /* 走子计数 (决定何时 learnBatch) */
    long long m_leafEvals;
    long long m_fullLeafEvals;      /* 其中走全量口径的次数 (稀疏路径失败/被关掉) */
    double m_lastLoss;              /* 最近一批的平均损失 (界面曲线用) */
    int m_lastBatchSamples;
    double m_maxAbsTarget;          /* 最近一批 clamp 之后的 |y| 最大值 */
    double m_maxAbsTdErr;
    BatchDiag batchDiag;            /* 最近一批 */
    BatchDiag trainDiag;            /* 自进程启动以来的累计 */

    /* 终局通道计数 (与 DQNMCTSAgent 同一套口径, 便于两支直接对照) */
    enum EndCode {
        END_CAP = 0,
        END_RED_WIN = 1,
        END_BLACK_WIN = 2,
        END_DRAW = 3
    };
    long long endCount[4] = { 0, 0, 0, 0 };
    long long endSeenByGameOver = 0;
    long long externalTerminals = 0;   /* notifyGameResult 真的接住并改了样本的次数 */

    /* 动作别名的增量统计 (每次自对弈取着法时累加; 自检面板读它) */
    long long aliasMoves = 0;
    long long aliasIndexed = 0;
    long long aliasWorstSlot = 0;
    long long aliasClearedMoves = 0;

    /*
     *  ---- 外部终局通道要挂的那条样本在哪儿 ----
     *  从**池尾**数的偏移 (0 = 最新一条)。为什么不是"局号"或 `memories.back()`:
     *    * `memories.back()` 可能是一条探索期的**推演**样本 (与真实棋局无关),
     *      把 done=true 挂上去比不挂更坏;
     *    * 用"局号"需要 agent 知道"新的一局什么时候开始", 而人机对弈里它不知道
     *      (界面只告诉它终局, 不告诉它开局)。
     *  用"最新那条 decision 样本"就同时解决两件事: 它一定是**真实走过**的那一手,
     *  而且每推一条新样本就把这个偏移更新一次 (淘汰只从队首弹出, 不影响尾部偏移)。
     *  -1 = 池里还没有 decision 样本 (这一局它一步没走过) —— 那时 `notifyGameResult`
     *  如实返回 false (调用方会打日志), 而不是随便找一条挂上去。
     */
    long long m_lastDecisionFromBack = -1;

public:
    DQNMCTSMOETbAgent(Chess &chess_,
                      int hiddenDim = 64,
                      float gamma_ = 0.99f,
                      float lr = 0.001f,
                      float eps = 1.0f,
                      float c_puct_ = 1.5f,
                      bool denseMoe = false);

    ~DQNMCTSMOETbAgent() = default;

    /* ---- AgentBase 接口 ---- */
    Step getBestMove(int color) override;
    std::string getName() const override;
    bool exploreAndTrain(int color, int rolloutSteps,
                         const OpponentPolicy &opponent = OpponentPolicy()) override;
    bool notifyGameResult(int chessResult, int perspective) override;
    float getLastTrainLoss() const override;
    std::string selfCheckReport() const override;

    /* 奖励口径: 界面奖励曲线取这一份 (与其它 RL agent 同一条约定) */
    bool hasLearningReward() const override { return true; }
    float learningStepReward(const Step &s, int color) override
    {
        return computeReward(s, color);
    }
    std::string rewardCaliperName() const override { return std::string("学习口径"); }

    /* ---- 表示 / 动作 ---- */
    void encodeSparse(int color, std::vector<std::uint16_t> &cells) const;
    void expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state);
    void denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells) const;
    /*
     *  3 个上下文 = [无吃子进度, 重复次数, 是否被将], 顺序与数值口径就是
     *  `chessstate.h` 的 CTX_HALFMOVE / CTX_REPEAT / CTX_CHECK。
     *  `Chess &` 而不是 `const Chess &`: 重复次数要读 `history` 与 `computeHash()`
     *  (与引擎判和的窗口逐字一致), 它们的签名不接受 const。
     */
    void contextOf(Chess &c, int color, float out[CTX_COUNT]) const;
    void writeContext(RL::Tensor &state, const float ctx[CTX_COUNT]) const;
    void readContext(const RL::Tensor &state, float out[CTX_COUNT]) const;
    void encodeStateFor(int color, RL::Tensor &state);
    void encodeState(RL::Tensor &state);          /* 视角 = chess.sideToMove */
    int stepToActionIdx(const Step &s, int color) const;
    void getLegalActions(int color,
                         std::vector<Step*> &steps,
                         std::vector<int> &actionIndices,
                         RL::Tensor &actionMask);
    /*
     *  即时奖励 (走子方视角, 与 PG/DQN/PPOMCTS 同源: `stone.h::stepReward`)。
     *  ⚠ 必须在 `moveForward` **之前**调 (吃子奖励要读被吃子的 value)。
     *  走子方视角 = 规范视角下网络的价值视角, 所以**不需要**任何"换到黑方框架"的换算
     *  (旧类那一支需要 `moverRewardToBlackFrame`, 因为它的编码是"黑为正")。
     */
    float computeReward(const Step &s, int color);
    /* 终局值 (走子方视角): 搜索叶子与训练目标**共用这一个出口** */
    float terminalReward(int chessResult, int mover) const;
    /* 该局面是否已终局 (走 `Chess::getResult`, 一次覆盖 将杀/困毙/吃将/重复/限着) */
    bool terminalValueOf(int color, double &value) const;

    /* ---- 网络前向 ---- */
    /*
     *  稀疏叶子前向: 骨干**前向一次**, 再把输出头只在 legalIdx 那几列上算
     *  (`Net::sparseLogits`)。返回 legalIdx 上的 Q (长度 = legalIdx.size())。
     *  走不了稀疏路径 (头不支持/下标越界) 时返回 false, **调用方必须回退全量** ——
     *  与 RL::PPO 的 R1 同一条判据 ("加了新层类型却忘了实现时, 行为是慢而不是错")。
     *  线性头的稀疏列与全量前向在那些列上**逐元素相等** (没有激活需要整向量归一),
     *  这条等价性由 test 的 [3] 钉住 (容差 1e-6)。
     */
    bool sparseQOnline(const RL::Tensor &state, const std::vector<int> &legalIdx,
                       std::vector<float> &qOut, std::vector<float> *q2Out);
    bool sparseQTarget(const RL::Tensor &state, const std::vector<int> &legalIdx,
                       std::vector<float> &qOut, std::vector<float> *q2Out);
    /* 全量口径的 Q (长度 = ACTION_DIM), 失败时 qOut 清零并返回 false */
    bool qValuesFull(const RL::Tensor &state, RL::Tensor &qOut, RL::Tensor *q2Out);
    /*
     *  先验: 合法槽位上 Q 的 softmax (温度 priorTemp)。DQN 没有策略头, 所以 P 只能
     *  来自它自己的价值排序 —— 这条是 PUCT 的第一个来源, 也是全部来源。
     */
    void legalPriors(const std::vector<float> &qLegal, std::vector<double> &priorsOut) const;

    /* ---- 搜索 ---- */
    /* PUCT 分数 (含 negamax 负号; 见类头注释第 5 条) */
    double getPUCT(int childID, int parentVisits) const;
    void visitDistribution(int rootID, RL::Tensor &pi) const;
    Step selectMove(int color, int simulations_ = 0, float temp = 0.0f);

    /* ---- 学习 ---- */
    /* 一次 mini-batch 更新; 返回平均损失 (池里不足 batchSize 条时返回 0 且不更新) */
    float learnBatch(int batchSize_, int epochs = 0);
    void resetMoeBatchStats();
    int samplePoolSize() const { return (int)memories.size(); }
    /*
     *  GUI 的损失上报与"这一手有没有学"的判据都读这个名字 (`getLearnSteps`) ——
     *  与 SAC / PPO / DQNAB 同一约定 (见 chessboard.h 的 reportLearnedLoss 模板)。
     */
    int getLearnSteps() const { return m_learnSteps; }

    /* ---- 在线训练助手 (界面/测试) ---- */
    void recordExperience(const Step &chosenStep, int color);
    void endOnlineEpisode(int gameResult);
    /*
     *  "从自己的搜索学一次" 的实际动作 (在 selectMove 选定走法之后调, 棋盘此时还是根局面):
     *  试走一手拿 r 与 s', 编码后**原样退回**, 存一条 decision 样本, 并按 `learnEveryMoves`
     *  的节拍做一次 learnBatch。棋盘逐字节复原是这个函数的硬契约。
     *
     *  ⚠ `rootIdx` 必须是**根局面的合法动作下标** (selectMove 刚算出来的那一份)。
     *  为什么不让它去读"走子前"的缓存: 那份缓存只在 `m_trainingMode` 打开时才填
     *  (训练回路用), 而**界面上的决策路径从来不开那个标志** —— 第一版就是这么写的,
     *  于是界面每走一手都会拿一份**空局面**(cells 为空 = 全零状态)去训练。
     *  这种错不会报任何错, 只会让训练数据变成噪声 (SAC 那条线上同类: "开关/缓存
     *  写错位置 ⇒ 静默空操作")。所以这里改成**自己在根局面上现编码**。
     */
    void learnFromSearchStep(int color, const Step &chosenStep, int actionIdx,
                             const std::vector<int> &rootIdx);
    void warmupFromCurrent(int episodes = 5, int simulations_ = 40, int maxMoves = 200);
    void trainVsRandom(int episodes, int simulations_ = 40,
                       int maxMoves = 200, bool verbose = true);
    void trainSelfPlay(int episodes, int simulations_ = 40,
                       int maxMoves = 200, bool verbose = true);

    /* ---- MoE / 骨干读数 (自检面板与工具用; 全部只读) ---- */
    int moeExpertCount() const;
    int moeTopK() const;
    bool moeDenseNow() const;      /* 骨干内部真实的 TopK == E (不是开关回显) */
    void moeUsage(std::vector<long long> &out) const;
    void resetMoeUsage();
    int tbHeadsRequested() const;
    int tbHeadsUsed() const;
    int tbHeadDim() const;
    int tbHeadsAllocated() const;
    long long tbAttnElements() const;
    long long trunkParamCount() const;
    long long headParamCount() const;
    long long uniqueParamCount() const;

    /* ---- 权重 ---- */
    static const char *defaultWeightPrefix();
    bool saveModel(const std::string &filepath);
    bool loadModel(const std::string &filepath);

    /* ---- 统计 ---- */
    int getTotalEpisodes() const { return totalEpisodes; }
    float getExploreRate() const { return exploringRate; }
    float getWinRate(int color = Stone::COLOR_BLACK) const
    {
        const int idx = (color == Stone::COLOR_BLACK) ? 1 : 0;
        return totalEpisodes > 0 ? (float)totalWins[idx] / (float)totalEpisodes : 0.0f;
    }

private:
    /* 建网 (构造函数里调一次; 之后任何"改骨干"的赋值都是静默空操作) */
    void buildNets();
    /* 稀疏/全量两条叶子前向的公共实现 (骨架是"骨干一次 + 头只算合法列") */
    bool sparseQGeneric(const RL::Tensor &state, const std::vector<int> &legalIdx,
                        std::vector<float> &qOut, std::vector<float> *q2Out,
                        RL::Net &trunkRef, RL::Net &headRef, RL::Net *head2Ref);
    /*
     *  把一条新样本推进回放池并做淘汰。
     *  `decision = true` 表示这是真实对局的决策样本 (外部终局通道要回填它)。
     */
    void pushSample(const Sample &s);
    /* 记一局的结束方式 (三个训练收尾点共用; 一局恰好一条) */
    void noteEnd(int result, bool seenByGameOver);
    /* 从当前棋盘 (已经落在 s') 采一份合法槽位位图 */
    void nextMaskBits(int color, std::uint64_t bits[2], int &legalCount);
    /* ε-greedy 从 Q 里取一个合法动作 (rollout 的探索策略) */
    int pickActionEpsilon(const RL::Tensor &state, const std::vector<int> &legalIdx);
    /* 叶子价值: 合法槽位上 max Q (twinCritic 时取 min(Q1,Q2) 之后的最大) */
    double leafValueFrom(const std::vector<float> &q, const std::vector<float> *q2) const;
    /*
     *  把终局值回填到**最新那条还没写过的 decision 样本**上。
     *  返回值: **1** = 这次真的写进去了, **0** = 已经写过 (幂等, 什么都不做),
     *          **-1** = 池里没有可挂的决策样本 (这一局它一步没走过)。
     *  回填对象用"从池尾往回找第一条 decision 样本"锁定, 不许用 `memories.back()` ——
     *  池里混着探索期的推演样本, 把 done 挂到无关局面上比不挂更坏 (SAC 那边有同样的
     *  注释)。三个返回值分开是为了让"接住了"与"本来就写过"在**读数**上分得开:
     *  外部终局计数只统计真的写进去的次数。
     */
    int attachTerminalToLastDecision(float terminalValue);
    /* 走子计数 + 节拍学习 (真实决策与训练回路共用) */
    void maybeLearn();
    /* 位图助手 */
    static bool bitSet(const std::uint64_t bits[2], int idx);
    static void bitSetTo(std::uint64_t bits[2], int idx);
    static bool bitsEmpty(const std::uint64_t bits[2]);

    /* 搜索/训练复用的缓冲区 (避免每次模拟都分配 1263 维张量) */
    RL::Tensor m_stateBuf;
    RL::Tensor m_nextBuf;
    RL::Tensor m_maskBuf;
    RL::Tensor m_qBuf;
    RL::Tensor m_q2Buf;
    std::vector<float> m_qLegal;
    std::vector<float> m_q2Legal;
    std::vector<double> m_priors;

    /* 在线训练时缓存的"走子前"局面 (recordExperience 要用) */
    std::vector<std::uint16_t> m_cachedCells;
    float m_cachedCtx[CTX_COUNT] = {};
    int m_cachedLegal = 1;
    std::uint64_t m_cachedMask[2] = { 0, 0 };
    /*
     *  探索期 (rollout) 的"这一步的合法集"快照: pick 刚算出来的那一份, 给紧接着的
     *  onTrans 复用 (那时棋盘已经走到 s', 但**这一条样本**的 s 是 pick 时的局面)。
     *  与 SAC 那一支的 m_pendingLegalCount/m_pendingMask 同一手法。
     */
    int m_pendingLegalCount = 1;
    std::uint64_t m_pendingMask[2] = { 0, 0 };
    /*
     *  "走子前"缓存是否有效 (recordExperience 的前置条件)。一份缓存**只用一次** ——
     *  于是"没先 selectMove 就记样本"与"同一手记两条"都会落进 m_recordSkipped,
     *  在自检面板上看得见, 而不是静默写进一条错的样本。
     */
    bool m_haveCached = false;
    long long m_recordSkipped = 0;
    /*
     *  训练回路 (trainSelfPlay / trainVsRandom / warmupFromCurrent) 自己负责记样本
     *  (它们要控制局计数、终局与节拍), 所以在这三条路径上调用 selectMove 时要把
     *  "从自己的搜索学一次"关掉 —— 否则同一手会记两条样本、还会多跑一次更新。
     *  这个标志是**显式**的, 不靠"池里有没有重复"去猜。
     */
    bool m_inTrainLoop = false;
};

#endif // DQNMCTS_MOETB_AGENT_H
