#include "chessboard.h"
#include "bcagent.hpp"      /* 行为克隆的 agent 胶水层 (sampleFrom / update; 2026-10) */
#include "rl/cpuinfo.hpp"
#include "rl/diag.h"        /* 价值评估曲线的口径: zFromGameResult / valueDiagOf (2026-10) */
/*
   [2026-10 用户口径 "对弈期间不更新保存模型权重"] 临时权重路径改走内存的通道。
   放在 chessboard.cpp 顶层: 本文件里"武装/解除 + 三条边界"是它唯一的用法。
*/
#include "rl/weightio.hpp"
/*
   [2026-10] 实时路由探针的**完整定义** (界面呼吸灯)。
   chessboard.h 只有它的前向声明 (那个头要保持轻), 而 `liveMoeRoute` 要读它的原子
   快照, 所以完整定义在这里要进来 —— 与下面几个 agent 的 .cpp 同一个做法。
*/
#include "rl/sparse_moe.hpp"
/* AGENT_SACAZ_OLD 用的是这个**独立类** (不是 SACAZAgent + 开关, 也不是它的派生类), 见头注释 */
#include "sacazlegacyagent.h"
#include "sacazmoetbagent.h"   /* AGENT_SACAZ_MOE = 稀疏 MoE + TB 专家 (独立类) */
#include "sacazmoemlpagent.h"  /* AGENT_SACAZ_MOE_MLP = 稀疏 MoE + MLP 专家 (独立类) */
#include <QDebug>
#include <QDir>
#include <QFile>        /* 冻结快照的清理 (releaseFrozenOpponent) 与审计比对 */
#include <QFontMetrics>
#include <QStringList>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>      /* 自检面板要报"权重文件在不在、多大" */
#include <iterator>     /* 审计的逐字节比对用 istreambuf_iterator */
#include <limits>
#include <set>          /* backgroundTrainLoop: "没接后台训练"每种 agent 只报一次 */
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

/* agent 的短名字, 显示在"AI 正在思考"提示里 */
QString agentDisplayName(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_ALPHABETA: return QStringLiteral("Alpha-Beta");
    /*
       Alpha-Beta 的三档弱等级: 名字里**必须**带上深度 —— 对弈日志/奖励曲线/比分表上
       只用 "Alpha-Beta" 的话, "L1 对 L3" 这种对局两个参赛方同名, 读数就没法读了。
       深度值来自 abDepthOf() (单一来源), 不在这里手抄数字。
    */
    case ChessBoard::AGENT_AB_L1:
        return QStringLiteral("Alpha-Beta L1(深%1)").arg(ChessBoard::abDepthOf(type));
    case ChessBoard::AGENT_AB_L2:
        return QStringLiteral("Alpha-Beta L2(深%1)").arg(ChessBoard::abDepthOf(type));
    case ChessBoard::AGENT_AB_L3:
        return QStringLiteral("Alpha-Beta L3(深%1)").arg(ChessBoard::abDepthOf(type));
    case ChessBoard::AGENT_MCTS:      return QStringLiteral("MCTS");
    case ChessBoard::AGENT_PG:        return QStringLiteral("Policy Gradient");
    case ChessBoard::AGENT_DQN:       return QStringLiteral("DQN");
    case ChessBoard::AGENT_PPOMCTS:   return QStringLiteral("PPO+MCTS");
    case ChessBoard::AGENT_DQNMCTS:   return QStringLiteral("DQN+MCTS");
    case ChessBoard::AGENT_EVAB:      return QStringLiteral("EVAB");
    case ChessBoard::AGENT_SACAZ:     return QStringLiteral("SAC+AZ");
    case ChessBoard::AGENT_SACAZ_MOE: return QStringLiteral("SAC+AZ-MoE");
    /* [2026-09 独立类] 稀疏 MoE + **MLP 专家**那一支 (SACAZMoEMlpAgent) */
    case ChessBoard::AGENT_SACAZ_MOE_MLP: return QStringLiteral("SAC+AZ-MoE-MLP");
    /* 59e5233 行为还原版 (独立类 SACAZLegacyAgent): MLP 骨干 / 稀疏 MoE+TB 专家骨干 */
    case ChessBoard::AGENT_SACAZ_OLD: return QStringLiteral("SAC+AZ-59e5233");
    case ChessBoard::AGENT_SACAZ_OLD_MOE: return QStringLiteral("SAC+AZ-59e5233-MoE");
    case ChessBoard::AGENT_DQNAB:  return QStringLiteral("DQN+AB");
    /*
       [2026-09 dev-dqnmcts-moetb] DQN+MCTS 的**MoE+TB 专家骨干**那一支 (独立类
       DQNMCTSMOETbAgent)。名字里必须带上骨干口径 —— 与 AGENT_DQNMCTS 在同一个
       下拉框里, 两者在日志/奖励曲线/比分表上不能同名 (同名 = 读数没法归因)。
    */
    case ChessBoard::AGENT_DQNMCTS_MOE: return QStringLiteral("DQN+MCTS (稀疏MoE+TB专家)");
    case ChessBoard::AGENT_PPOMCTS_MLP: return QStringLiteral("PPO+MCTS-MLP");
    }
    return QStringLiteral("agent");
}

/* 哪些 agent 真的实现了 exploreAndTrain (决定状态条显示"① 探索"还是"① 搜索") */
bool agentCanExplore(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PG:
    case ChessBoard::AGENT_DQN:
    case ChessBoard::AGENT_PPOMCTS:
    case ChessBoard::AGENT_DQNMCTS:
    case ChessBoard::AGENT_EVAB:
    case ChessBoard::AGENT_SACAZ:
    case ChessBoard::AGENT_SACAZ_MOE:
    case ChessBoard::AGENT_SACAZ_MOE_MLP:
    case ChessBoard::AGENT_SACAZ_OLD:
    case ChessBoard::AGENT_SACAZ_OLD_MOE:
    case ChessBoard::AGENT_DQNAB:
    /* 本类实现了 exploreAndTrain (AgentBase 口径), 所以状态条上是"① 探索" */
    case ChessBoard::AGENT_DQNMCTS_MOE:
    case ChessBoard::AGENT_PPOMCTS_MLP:
        return true;
    default:
        return false;
    }
}

/*
 * [④] 哪些 agent 有"学习口径"的奖励 (即时奖励有它自己的量纲: 材质 x0.1 + 每步代价)。
 *
 * 为什么这里要有一张**静态表** (而不是只问对象): 界面在**开局前**就要把口径标签写在
 * 曲线名上, 那时对象可能还没建 (它们是每手决策时按需 new 出来的)。运行期取数时仍然
 * 以对象自己的 `hasLearningReward()` 为准 (见 learningStepRewardOrNaN) —— 两者必须
 * 一致, test_match [2.19] 拿真实对象对过表。
 *
 * EVAB 故意不在表里: 它的"探索"产物是蒸馏给评估网络的 (没有 computeReward),
 * 环境奖励对它就是引擎那本账。
 */
bool agentHasLearningRewardTable(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PG:
    case ChessBoard::AGENT_DQN:
    case ChessBoard::AGENT_PPOMCTS:
    case ChessBoard::AGENT_PPOMCTS_MLP:
    case ChessBoard::AGENT_DQNMCTS:
    case ChessBoard::AGENT_SACAZ:
    case ChessBoard::AGENT_SACAZ_MOE:
    case ChessBoard::AGENT_SACAZ_MOE_MLP:
    case ChessBoard::AGENT_SACAZ_OLD:
    case ChessBoard::AGENT_SACAZ_OLD_MOE:
    case ChessBoard::AGENT_DQNAB:
    /* DQN+MCTS (MoE+TB 专家) 有 computeReward 口径 (走子方视角的材质 + 每步代价) */
    case ChessBoard::AGENT_DQNMCTS_MOE:
        return true;
    default:   /* AGENT_ALPHABETA / AGENT_MCTS / AGENT_EVAB */
        return false;
    }
}

/* ================================================================
 *  [诊断] 卡顿定位日志 (2026-09, 用户报障 "人机对弈黑方赢了之后卡住")
 * ================================================================
 *
 * 为什么要有它: "黑方赢了之后再次开局, 黑方进入无限等待" 这类现象在**代码上看不出来**
 * —— 它有可能是 (a) 等某把锁, (b) 状态机停在某个 case, (c) 搜索真的在跑但极慢。
 * 三者在界面上完全一样 (都是"没反应")。所以这里在关键路径上打**带线程号与耗时**的日志,
 * 一眼就能区分:
 *     [dbg] ... (tid=12345)  -> 谁在执行
 *     [dbg] ... waited 8200 ms -> 它等了多久
 *
 * 判读方法 (用户报障的现场):
 *   * 如果最后一行是 "aiThink: 等 m_agentMutex ..." 且 waited 很大
 *       -> 有人长时间持有该锁 (最可能是**保存权重**, 见 saveCurrentAgentModel)。
 *   * 如果最后一行是 "process: state=..." 而之后再没有 "aiThink" 行
 *       -> 状态机没把 AI 唤醒 (重置/终局路径漏了唤醒)。
 *   * 如果 "aiThink: 开始" 与 "aiThink: 结束" 之间只差正常耗时
 *       -> AI 其实在跑, 是**搜索太慢** (例如随机初始化权重 + 大模拟次数)。
 *
 * 用环境变量 CHESS_DEBUG_LOCKS=1 打开。**默认关闭**: 这些行在落子/决策路径上, 常态
 * 跑不该为它们付出格式化字符串 + 写 stderr 的代价 (排查卡死时再打开, 见上表的判读方法)。
 * (原来的注释写的是"默认打开", 与 `dbgEnabled()` 的实际取值不一致 —— 以代码为准。)
 */
namespace {
bool dbgEnabled()
{
    static const bool on = (qEnvironmentVariableIntValue("CHESS_DEBUG_LOCKS") != 0);
    return on;
}
long long dbgTid()
{
    return (long long)std::hash<std::thread::id>()(std::this_thread::get_id());
}
double dbgNowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}
/*
   直接 printf 到 **stderr** (用户口径: "直接printf")。

   为什么是 stderr 而不是 stdout, 为什么不用 qInfo:
     * 本程序是 **Console 子系统** (PE Subsystem=3, 实测), 所以两个流都有去处;
       但 **stdout 是带缓冲的** —— 重定向到文件时整块才 flush, 排查卡死时
       "日志里什么都没有"就是这么来的 (本仓库已经吃过一次这个亏)。stderr 不带缓冲,
       卡死/崩溃时最后几行一定已经在终端上。
     * `qInfo()` 在 MSVC 构建下走 Qt 的消息处理器, 默认送给 `OutputDebugString`,
       只有挂调试器才看得到 —— 用户在 cmd 里跑却"一行日志都没有", 就是因为它。
   主程序另外装了一个 Qt 消息处理器把这些转发到文件 (见 main.cpp), 但诊断路径
   这里是**直连 printf**, 不再经过 Qt。
*/
void dbgPrint(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}
void dbgLog(const QString &what)
{
    if (!dbgEnabled()) { return; }
    dbgPrint("[dbg] %s (tid=%lld)", what.toUtf8().constData(), dbgTid());
}
void dbgWait(const QString &what, double ms)
{
    if (!dbgEnabled()) { return; }
    dbgPrint("[dbg] %s -- waited %lld ms (tid=%lld)",
             what.toUtf8().constData(), (long long)ms, dbgTid());
}
}  // namespace

/* 状态条上的短耗时: "3.24s" / "1:05" */
QString shortElapsed(long long ms){
    if (ms < 0) {
        ms = 0;
    }
    const long long totalSec = ms / 1000;
    const int centi = static_cast<int>((ms % 1000) / 10);
    if (totalSec >= 60) {
        return QStringLiteral("%1:%2").arg(totalSec / 60).arg(totalSec % 60, 2, 10, QChar('0'));
    }
    return QStringLiteral("%1.%2s").arg(totalSec).arg(centi, 2, 10, QChar('0'));
}

} // namespace

/*
 * ChessBoard - 核心游戏面板
 * 负责: 棋盘绘制、走法执行、AI调度、数据库记录、回放
 */

/*
 * AI 搜索预算。这几个数字以前散落在注释里, 而且和真实参数对不上
 * (注释写"深度=8"、UI 标签写"深度=4"、实际是 5; 注释写 800 次模拟、实际 80;
 *  注释写 400 次迭代、实际 6)。集中在这里, 只留一个来源。
 *
 * AB_DEPTH 取 4 的依据 (本机实测, test_ab 的搜索基准):
 *   深度 3 = 63 ms, 深度 4 = 169 ms, 深度 5 = 3605 ms
 * 走法合法性过滤 (sample() 会剔除自杀/不应将/照面) 与"两个节点类型都做静态
 * 搜索"都会增加节点成本, 深度 5 在 GUI 里已经是 3.6 秒一步。需要更强棋力时把
 * 这里调大即可。
 */
/*
 * AGENT_ALPHABETA 的搜索深度 (= 界面上那一档"Alpha-Beta Pruning (深度=4)")。
 * ⚠ 这里是**既有对照组的基准深度, 不要改**: 三档弱等级 (AGENT_AB_L1/L2/L3) 是
 *   另外三个类型、另外三个数字 (见下面 AB_L2_DEPTH / AB_L3_DEPTH), 它们**不**动这一个。
 */
static constexpr int AB_DEPTH = 4;            /* Alpha-Beta 搜索深度 */
/*
 * ---- Alpha-Beta 三档弱等级的深度 (2026-09) ----
 * 深度 1 / 2 / 3。**只改这三个常量就能调等级**: 决策路径、状态条、自检都从
 * ChessBoard::abDepthOf() 取数, 不会再出现"某一支还印着别的深度"。
 *
 * 【实测 2026-09 · `test_ab benchmark` · 本机 Release/AVX2 · 初始局面】
 *     深度 1 = 0 ms | 深度 2 = 3 ms | 深度 3 = 34 ms | 深度 4 = 90 ms | 深度 5 = 1920 ms
 *   (同一份基线的历史读数在 docs/issues_review.md:179-180 记成 3≈57/4≈148/5≈3.1 s,
 *    与本次实测有出入 —— 那份是旧机器/旧构建, 两者都不是"错", 但**不要**把两份数字
 *    混着引用。要用就把命令与日期一起抄下来, 与 docs/mcts_sims_scaling.md 的惯例一致。)
 * 结论: L1~L3 每步都是 0~34 ms 量级 —— 便宜到可以拿来当"每局都要打的陪练",
 * 而它们与 AGENT_ALPHABETA (深度 4, ~90 ms) 一起构成一条**不随训练漂移**的棋力阶梯。
 */
static constexpr int AB_L1_DEPTH = 1;
static constexpr int AB_L2_DEPTH = 2;
static constexpr int AB_L3_DEPTH = 3;
static constexpr int MCTS_SIMS = 800;         /* MCTS 模拟次数 */
/*
 * PPO+MCTS 每次决策的模拟次数。**必须远大于中局分支数 (~38.7)**, 理由见下面
 * BG_TRAIN_SIMS 的长注释: 少于分支数时搜索一次深挖都没有, 出招与策略目标都退化成
 * "先验前 N 名的均匀分布"。80 次时实测仍有约 48% 的访问落在"每个孩子一次"的
 * 地板里 (bench_ppo_sims 的根节点诊断), 400 次降到约 10%。
 *
 * ---- [2026-10] E/top-k 已经来回切过一次, 这个预算的**代价**跟着变 (实测) ----
 * 同一台机器 (bench_ppo_backbone_tb + tools/measure_peak_working_set.ps1):
 *   E=4/top-1 (当前): 参数量 actor+critic **104.3 M**, **11.02 ms/模拟 ->
 *                     400 模拟 = 4.31 s/步**, 峰值工作集 **1,608 MB**
 *                     (2026-10 切回 4/1 当天实测; 早期注释里那个 "≈3.2 s" 是另一个
 *                      量法/另一个版本的旧读数, 以本节这组同工具同日读数为准)
 *   E=8/top-2 (已回退): 20.1 ms/模拟 -> 400 模拟 = 8.0 s/步; 参数量 207.8 M;
 *                     峰值工作集 3,301 MB (actor+critic 各背一套完整 TB 专家, 四份缓冲)
 * **2026-10 第二次用户口径已切回 4/1** (`rl/ppo.h` 的 PPO_MOE_EXPERTS/TOPK), 于是
 * 那个"每步等到 8 秒"的时代结束了 —— 而 400 这个数**刻意不动**:
 *   * 用户口径原文是"保持 400, 恢复到历史配对": 4/1 配 400 模拟本来就是 2026-10 之前
 *     的那一档, 切回去等于**免费把每步从 8.0 s 拿回 4.3 s**, 搜索深度一点没减;
 *   * "模拟次数 ≫ 分支数 (~38.7)"是 π 目标有没有信息量的分水岭 (见 BG_TRAIN_SIMS 的
 *     长注释), 而"模拟次数"又是本工程**唯一测出过棋力**的杠杆 (40/64/120 ->
 *     42.2%/50.0%/62.5%) —— 所以在"省下来的算力怎么花"这件事上, 唯一有证据的答案是
 *     **别花掉**: 留着它换墙钟。
 *   * 哪天真想把这省下来的时间再换成搜索, 就把这个数抬到 ~1000 (回到 ~10 s/步) —— 那是
 *     一个需要用户开口的取舍, 不该由代码偷偷改掉。
 *   * 想"又快又深"还有第三个出口: 换 `PPO+MCTS (MLP专家)` 那一支, 同一份搜索代码、
 *     0.22 ms/模拟, 1600 模拟 ≈ 0.35 s/步 —— 它的骨干便宜 ~90 倍 (见 rl/ppo.h 顶部那张表)。
 * 一句话: **top-k 那一半本来就是拿"模拟次数"换"容量"的**, 切回 4/1 就是把这个交换退掉。
 */
static constexpr int PPO_SIMS = 400;          /* PPO+MCTS 每次决策的模拟次数 */
/*
 * PPO+MCTS (MLP 专家骨干) 每次决策的模拟次数。
 *
 * 为什么比上面那个大 4 倍: 骨干便宜 ~25x (单网络前向实测 MlpExpert E=8 top-2
 * 0.139 ms vs TB<16,360> E=4 top-1 3.59 ms, 见 rl/ppo.h 顶部那张表), 而"模拟次数 ≫
 * 分支数(~39)"是 π 目标有没有信息量的分水岭 (见下面 BG_TRAIN_SIMS 的长注释)。
 * 给 4 倍既是"用便宜的算力换更准的访问分布", 也仍然比 TB 那一支**快**得多:
 * 实测本机 0.23 ~ 0.39 s/手 (test_match [2.14] 的两次 6 手小局; TB 那一支在 400 次
 * 模拟下约 3.2 s/手, 见 test_match [2.7] 的思考耗时读数)。
 * 想再换"更快"或"更准", 只需改这一个数 (成本线性)。
 */
static constexpr int PPO_MLP_SIMS = 1600;     /* PPO+MCTS (MLP 专家) 每次决策的模拟次数 */
static constexpr int DQNMCTS_ITERATIONS = 200;/* DQN+MCTS 每次决策的迭代次数 */
/*
 * SAC+AZ 每次决策的 MCTS 模拟次数。实测 (test_sacaz, 本机 AVX2): 64 次模拟约 3 ms,
 * 也就是 ~0.05 ms/模拟 —— 它的网络是 1260->64->64->128 的小 MLP, 比 EVAB 的
 * alpha-beta 搜索便宜得多, 所以这里给到 256 次 (约 12 ms), 换更准的访问分布。
 * 需要更强就继续加大: 成本是线性的。
 */
/*
 * EVAB 的搜索预算。**不能用 AB_DEPTH**: EVAB 有置换表 + 迭代加深 + 走法排序 +
 * 静态搜索, 同一深度比 ABAgent 快一个数量级 (本机实测初始局面:
 *   EVAB depth 4 = 90 ms,  depth 5 = 188 ms,  depth 6 = 890 ms
 *   AB   depth 4 = 89 ms,  depth 5 = 1873 ms
 * 也就是 EVAB 用 AB 深度 4 的时间可以搜到深度 5。原来这里写的是 AB_DEPTH(4),
 * 等于把"学到的评估 + 更强的搜索"这两个卖点都关掉了 —— 实测那样下的 EVAB 与
 * ABAgent 走法 22/30 相同, 只会和棋。
 *
 * 深度给到 6 而把**时间上限**当真正的约束 (800 ms): 叶子评估的代价取决于 blend (见 evagent.h 里 blendMax
 * 的注释: blend>0 时每步要跑网络前向, 会贵 10 倍以上), 用迭代加深 + 时间上限就
 * 自动"贵了就搜浅一点", 不需要为两个模式各调一个深度。
 */
static constexpr int EVAB_DEPTH = 6;
static constexpr long long EVAB_BUDGET_MS = 800;
static constexpr int SACAZ_SIMS = 256;
static constexpr int SACAZ_HIDDEN = 64;       /* SAC+AZ 的网络隐层宽度 */
/*
   稀疏 MoE (TB 专家) 骨干的那个变体。
   ---- [2026-09 dev-sacmoetb] 这一档从 16 提到 40, 依据是同机实测 ----
   提出 16 时的一次模拟是 ~10.9 ms (老口径: 请求 15 个头却只跑 3 个, 而且一次叶子估值要
   把骨干前向三遍)。本轮两处改动把这两个数都压下来了 (test_sacaz [17A]/[17B]):
     * TB 专家头数口径修复: 单专家前向 15.7 ms -> 5.4 ms (2.9x);
     * 共享骨干: 一次叶子估值的骨干前向 3 次 -> 1 次 (selectMove 实测 9.3 -> 4.1 ms/模拟);
   于是同样 175 ms/步的预算能跑的模拟数从 ~19 变成 ~42。取 **40** (约 165 ms/步),
   留一点余量给"棋盘恢复 / 合法着法生成"这些不随模拟数伸缩的开销。
   **这条数字必须能复跑**: `bench_sacaz_vs_ab --budget=175 --trunk=shared` 会现场标定
   ms/模拟 并回算模拟次数; 换机器/换编译器之后以那个读数为准。
   为什么值得提: 16 < 分支数 (~39) 意味着**一次深挖都没有** —— π 目标退化成"把自己
   先验前 16 名抹平" (issues_review C16)。40 刚过分支数地板, 才开始有真正的深挖。
*/
static constexpr int SACAZ_MOE_SIMS = 40;
/*
   [2026-09 独立类 + 实测] MoE+MLP 专家那一支的界面模拟次数。
   原来照抄 TB 那一档 (40), 但**实测它根本不吃模拟预算** —— 本机 Release/AVX2,
   `bench_sacaz_vs_ab --backbone=moe-mlp --trunk=shared --games=8 --plies=80`:

     模拟/步   每步墙钟 (界面口径, 每手还跑一次 learnBatch)   每步墙钟 (只搜索)
        40              50.1 ms                                 3.3 ms
        64              52.8 ms                                 5.2 ms
        96              55.5 ms                                 8.0 ms
       128              58.0 ms                                10.0 ms
       256              67.8 ms                                19.3 ms

   两条读法:
     * **一次模拟只要 ~0.08 ms** (共享骨干 + 稀疏路由, 比 TB 的 ~5.4 ms 便宜 65 倍);
     * **每步的大头是"从自己的搜索学一次"那次 learnBatch (~47 ms), 不是搜索**。
   所以把这个常数抬到 **256 (= `SACAZ_SIMS`, 与 AGENT_SACAZ 同档)** 只多花 ~18 ms/步,
   总墙钟 67.8 ms/步 —— 仍在界面那条 175 ms 的预算以内。
   为什么值得抬: 用户口径是"这两支在界面上应当能公平比较", 而原来 256 vs 40 的差
   会被读成"MLP 骨干比 MoE-MLP 强" (实测:A 臂 mlp@256 = 57.5% vs B 臂 moe-mlp@40 = 51.2%,
   而把 MoE 的模拟数也抬到 256 它回到 56.2% —— 差距跟着预算走, 不跟着骨干走,
   见 docs/sacmoetb_independent_classes_2026_09.md §8.4)。
*/
static constexpr int SACAZ_MOE_MLP_SIMS = 256;
/*
 * 负载均衡辅助损失的系数。0.1 是实测选出来的 (bench_moe --cases=B --pretrain=3):
 *   aux=0     -> 8 个专家里有 3 个一次都没被选中 (路由塌了)
 *   aux=0.01  -> 都被用到, 但最大/均值仍是 4.00
 *   aux=0.1   -> 都被用到, 最大/均值 2.91   <- 选这个
 *   aux=0.5/2 -> 更均匀 (2.63/2.32), 但辅助项开始盖过真正的策略梯度
 * 这里的"批"只有 32 个样本, 而主干给门控的梯度比辅助项大两个数量级, 所以系数比
 * Switch Transformer 论文里的 0.01 要大得多才起作用。
 */
static constexpr float SACAZ_MOE_AUX = 0.1f;
/*
 * ================================================================
 *  [2026-10 门控实验] 两个 MoE 骨干的**无辅助损失偏置均衡** (界面口径)
 * ================================================================
 *  依据: docs/moe_gate_experiment_2026_10.md 的"建议 1"。
 *  机制 (Loss-Free Balancing, arXiv:2408.15664 Algorithm 1): 给每个专家一个
 *  **只影响 top-k 选择、不进输出加权** 的偏置 b_i, 每批 b_i += rate*sign(mean-load_i)。
 *  因为它走 argmax 而不是损失, **不产生任何干扰梯度** —— 与"把 SACAZ_MOE_AUX 调大"
 *  有本质区别 (后者的可用窗口只有 [0.1, 0.5), 再大就盖过主目标)。
 *
 *  实测 (端到端, 训练侧专家负载的 MaxVio; 0 = 完美均衡):
 *    TB 支   : 基线 0.963  ->  开偏置 0.009   (最小份额 0.129 -> 0.247)
 *    MLP 支  : 基线 0.318  ->  开偏置 0.006   (最小份额 0.102 -> 0.124; 8 个专家)
 *  为什么这条路必须走: [3b] 实测在真实棋局状态分布上, softmax + 辅助损失
 *  **即使 coef=10 也拉不动负载** (MaxVio 恒为理论最大 3.000) —— 它的梯度正比于
 *  饿死专家自己的门控概率(≈0), 放大系数乘的仍是 0。
 *
 *  rate 取 0.01: 这是本工程实测用的值 (20 局热身 ≈2494 个学习步内到位)。
 *  论文推荐 0.001, 但在界面这种"学习步稀疏"的节奏下太慢, 未在本工程扫过。
 *  关掉 = 回到改动前的行为 (setLossFreeBias(false, ...) 等价于这一支不存在)。
 */
static constexpr bool  SACAZ_MOE_LOSSFREE = true;
static constexpr float SACAZ_MOE_LOSSFREE_RATE = 0.01f;

/*
 * ================================================================
 *  [2026-10] PPO 两支 (AGENT_PPOMCTS / AGENT_PPOMCTS_MLP) 的 MoE 均衡口径
 * ================================================================
 *  为什么 PPO 这条也要有: 它同样是**稀疏 MoE 骨干** (TB 专家 E=4 top-1 与 MLP 专家
 *  E=8 top-2), 而此前只有辅助损失 `moeAuxCoef` —— 受控实验 ([3b]) 与端到端实测都表明
 *  **只靠辅助损失拉不平负载** (coef 开到 10, 训练侧 MaxVio 仍贴在上界; 根因是它的梯度
 *  正比于饿死专家自己的门控概率 ≈0)。SAC 那两支已经拿到这条修法 (训练侧 MaxVio
 *  0.963 -> 0.009 / 0.318 -> 0.006), PPO 这里此前是**空缺**, 这次补齐。
 *
 *  ---- 偏置均衡 (非参数缓冲, 默认开是安全的) ----
 *  它不进权重文件、不进参数量、不产生梯度 ⇒ 打开它**不会**让存量权重作废。
 *  刻度: rate=0.01 (本工程实测值)。论文推荐 0.001, 但那是在"学习步密集"的设定下,
 *  而本工程一局才一步 RMSProp —— 0.001 在会话尺度上几乎不动。
 *
 *  ---- MLP 门控 (默认关, 打开会让存量权重作废) ----
 *  三层 `d_model -> hidden (tanh) -> E`, 与 SAC 那两支同一个接口 (`enableMlpGate`)。
 *  为什么默认仍然是关 (三条, 都是实测):
 *    1. 形状变了 ⇒ `Net::load` 的参数量守卫会**拒绝**存量权重 (只能重训);
 *    2. 棋力没有结论 (SAC 支: 20 局 + 8 局对照, 区间跨 50%);
 *    3. 它**不是**均衡手段, 方向甚至相反: SAC 支实测训练侧更均衡 (0.049->0.009)
 *       而评测侧更偏 (0.264->0.598)。正确用法是**与偏置均衡配对**。
 *  要与不要, 都靠这个常量一处决定 (它是 >0 才装, 且必须在任何前向之前装)。
 * ================================================================
 */
static constexpr bool  PPO_MOE_LOSSFREE = true;
static constexpr float PPO_MOE_LOSSFREE_RATE = 0.01f;
static constexpr int   PPO_MOE_MLP_GATE_HIDDEN = 0;
/*
 * DQN+AB (AB 当 DQN 的 planning head) 的一次决策预算。
 * 单位是**搜索节点数** (= 网络前向次数), 不是模拟次数: TB 骨干实测 ~3 ms/节点,
 * 所以 256 节点 ≈ 0.8 s/步、能搜到 2~3 层 (bench_dqnab_vs_ab 实测深度 2.5)。
 * 要更深的规划就把它调大 (线性变贵), 或者把骨干换成 MLP (同一套算法便宜两个数量级,
 * 同样预算能搜 4~6 层)。
 */
static constexpr int DQNAB_NODES = 256;
static constexpr int DQNAB_HIDDEN = 64;    /* DQN+AB 的头隐层宽度 */
/*
   DQN+MCTS (稀疏MoE+TB专家) 每次决策的模拟次数。

   实测 (本机, Release): 一次叶子估值 (骨干 1 次前向) = 2.9~3.0 ms ⇒ 40 次模拟 = 123~135
   ms/步、120 次 = 364~371 ms/步。

   为什么取 120 而**不是** 40 (我最初填的就是 40): 40 次时搜索**退化了** —— 开局有 ~40 个
   合法着法, 每个孩子恰好被访问 1 次、严格打平, 于是根选择退化成"按先验顺序取第一个孩子"
   (先验 = 合法槽位上 Q 的 softmax) = 一层 Q 贪心, MCTS 那部分等于没干活。
   证据 (`bench_dqnmcts_moe --mode=agree --positions=16 --sims=40 --sims2=120`): 两个预算
   选出的着法一致率 **0/16 = 0%** —— 预算确实是这条路径上真的会改行为的旋钮。
   同一份工具里也能看到: 稀疏路由 vs 等参数稠密 = 3.99x, 所以"省算力"这件事已经由骨干
   拿到了, 这里把省下来的时间还给搜索。

   旧类 DQNMCTS_ITERATIONS=200 不能照搬: 那是 d_model=90 的稠密 MoE 预算 (一次模拟便宜
   两个数量级), 搬过来就是 1.3 s/步。
*/
static constexpr int DQNMCTS_MOE_SIMS = 120;
static constexpr int DQNMCTS_MOE_HIDDEN = 64;
/* 单局手数上限已集中到 ChessBoard::DEFAULT_MAX_PLIES (界面上可用 setMaxPliesPerGame 调) */

/*
 * 后台训练的单轮规模。训练线程只在每轮开始时检查停止标志, 所以这几个数字直接
 * 决定"关窗要等多久"。原来是 4 局 x 200 步 x 50 次 MCTS 模拟 —— 每步都要跑网络
 * 前向, 一轮可能几分钟, 关窗时 GUI 线程会冻在 join() 上。
 *
 * ---- BG_TRAIN_SIMS 为什么从 20 提到 400 (2026-09) ----
 * 20 次模拟在象棋中局是**结构性退化**: 实测根节点平均有 38.7 个合法着法
 * (bench_ppo_sims --probe-positions), 而选择阶段只在"未展开列表为空"时才向下深挖
 * (见 ppomcts_agent.cpp 的 SELECTION)。于是
 *     前 #legal 次模拟 = 把先验最高的那些孩子各展开一次 (每人恰好 1 次访问),
 *     max(0, sims − #legal) 次才是真正的深挖。
 * 20 < 38.7 ⇒ **一次深挖都没有**, 搜索只铺开了先验最高的 20 个孩子。于是
 *     π_target = "我自己前 20 名上各 1/20 的均匀分布",
 *     出招      = 从这个均匀分布里采样 (温度 1.0)。
 * 也就是说搜索提供的信息量是**零**: 目标只教网络"对你自己的先验前 20 名保持均匀",
 * 是一个把先验抹平的算子 —— 这直接解释了"跑很多轮棋力不动"。
 * 提到 400 后深挖余量约 361 次 (90%), 目标才真正携带搜索结果。
 * 代价: 每步约 8 ms/模拟 (当前 5.2e7 参数的骨干) ⇒ 单步从 ~0.16 s 涨到 ~3.2 s;
 * 一轮 (1 局 x 60 手上限) 从秒级变成分钟级, 所以 BG_TRAIN_EPISODES 保持 1,
 * 关窗等待时间仍由单轮决定。
 */
static constexpr int BG_TRAIN_EPISODES = 1;   /* 每轮训练局数 */
/*
 * 每局步数上限。
 * ⚠️ **不要调到 32 以下**: SAC+AZ / DQN+AB 的 learnBatch 有"回放池 < batchSize(=32)
 * 就直接返回、连 m_lastLoss 都不写"的门控 (sacazagent.cpp:857 / dqnabagent.cpp:1152),
 * 而 clone 每轮都是从空池开始 —— 少于 32 手的一轮会**一次梯度更新都不做**, 却照样
 * 把"没变过的权重"写回主 agent (损失曲线也不会上报, 因为 m_lastLoss 还是 NaN)。
 * test_match [2.13] 的第一版给了 6 手, 正是这么失败的 (三个 agent 全部"上报 0 次")。
 */
static constexpr int BG_TRAIN_MAX_MOVES = 60; /* 每局步数上限 (必须 > 32, 见上) */
static constexpr int BG_TRAIN_SIMS = 400;     /* MCTS 类 agent 每步的模拟次数 (须远大于分支数) */
/*
 * EVAB 后台训练的搜索深度。**不能照抄界面的 EVAB_DEPTH(6)**: 一轮是
 * BG_TRAIN_EPISODES 局 x BG_TRAIN_MAX_MOVES 手, 而 EVAB 的每一手要搜两次
 * (playDepth 选步 + labelDepth 生成 TD-leaf 标签, 见 evagent.h 的 trainSelfPlay)。
 * 本机实测 (初始局面): depth 4 = 90 ms, depth 5 = 188 ms, depth 6 = 890 ms,
 * 于是一轮 60 手在 3+4 下约 10 s、在 4+5 下约 17 s、照抄 6 就是分钟级 ——
 * 而关窗要等整整一轮 (训练线程只在每轮开头看停止标志, 见 stopBackgroundTraining)。
 * 取 3/4 是"一轮留在 10 s 量级"与"标签必须比选步更深"这两条的交点。
 */
static constexpr int BG_TRAIN_EVAB_PLAY_DEPTH = 3;
static constexpr int BG_TRAIN_EVAB_LABEL_DEPTH = 4;
/*
 * SAC+AZ 后台训练每步的模拟次数。MLP 骨干实测 ~0.05 ms/模拟, 所以这里直接用
 * BG_TRAIN_SIMS(400): 一轮 60 手约 1.2 s, 而且 400 远大于分支数 (~39), π 目标是
 * 真搜索出来的 (与 C16 那条"20 次模拟 = 零深挖"同一个道理)。
 */
static constexpr int BG_TRAIN_SACAZ_SIMS = BG_TRAIN_SIMS;
/*
 * 稀疏 MoE 骨干那一个变体: 实测 ~10.9 ms/模拟, 所以**不能用 400**
 * (400 x 60 手 ≈ 4.4 分钟/步…… 实际是 4.4 分钟一轮都不止)。
 * 界面上的决策预算是 16 次 (175 ms/步, 见 SACAZ_MOE_SIMS), 但训练要的是 π 目标质量:
 * 16 < 分支数 38.7 ⇒ **一次深挖都没有**, 目标退化成"把自己先验前 16 名抹平"(C16)。
 * 给 64: 刚过分支数地板, 深挖余量约 25 次 (39%), 一轮 60 手约 42 s —— 与
 * PPO+MCTS 那一档 (分钟级) 同一量级, 关窗等待可以接受。
 */
static constexpr int BG_TRAIN_SACAZ_MOE_SIMS = 64;

/*
 * 后台训练用的**临时权重路径** (前缀; 单文件 agent 就是文件本身)。
 *
 * 为什么每个多文件家族要给**不同的**前缀: 它们写出的文件名是
 *   PPO+MCTS      -> <prefix>_actor / <prefix>_critic
 *   PPO+MCTS-MLP  -> <prefix>_actor / <prefix>_critic   (与上一行**同名**, 所以前缀必须不同)
 *   SAC+AZ        -> <prefix>_actor / _q1 / _q2
 *   SAC+AZ-MoE    -> <prefix>_actor / _q1 / _q2
 *   DQN+AB        -> <prefix>_trunk / _v / _a
 * 而 `TMP_WEIGHTS` ("weights/_temp_train.dat") 是共用前缀 —— 于是 PPO 的 `_actor`
 * 与 SAC+AZ 的 `_actor` **同名**。虽然结构指纹会让误读当场失败 (不会静默串权重),
 * 但那是"靠断言兜住的设计", 不值得留着; 同一时刻只有一支在跑, 用不同前缀最省事,
 * 也免得把上一支的残留文件读进来。
 */
static const char *TMP_WEIGHTS       = "weights/_temp_train.dat";        /* 单文件 + PPO(TB) */
static const char *TMP_WEIGHTS_PPOMCTS_MLP = "weights/_temp_train_ppomcts_mlp";
static const char *TMP_WEIGHTS_SACAZ = "weights/_temp_train_sacaz";
/*
   [2026-09 dev-sacmoetb] AGENT_SACAZ_MOE 改成共享骨干口径 ⇒ 临时文件前缀也跟着换。
   不换的后果与正式前缀那条一样 (共享口径写 _trunk/_actorhead/..., 独立口径写
   _actor/_q1/_q2): 上一次会话留下的 _actor/_q1/_q2 会被新一轮的 loadModel 找不到
   _trunk 而失败 —— 或者在有人补了回退之后被**静默当成共享骨干**读进来。
*/
static const char *TMP_WEIGHTS_SACAZ_MOE = "weights/_temp_train_sacaz_moe_shared";
/* [2026-09 独立类] MoE+MLP 那一支的临时前缀: **必须独立** (与另外几支共用会让后台训练互相覆盖) */
static const char *TMP_WEIGHTS_SACAZ_MOE_MLP = "weights/_temp_train_sacaz_moe_mlp_shared";
/*
   行为还原版 SAC (AGENT_SACAZ_OLD = 独立类 SACAZLegacyAgent) 的临时前缀。
   **必须有独立前缀**: 与 AGENT_SACAZ 共用会让两个 agent 的后台训练互相覆盖权重
   (两者的训练口径不同, 覆盖之后是"棋力对不上训练量"这种没法归因的现象)。
*/
static const char *TMP_WEIGHTS_SACAZ_OLD = "weights/_temp_train_sacaz_old";
/*
   还原版的**TB 专家骨干**那一支 (AGENT_SACAZ_OLD_MOE) 的临时前缀。
   同样必须独立: 它与 AGENT_SACAZ_OLD 是"同一个类、不同骨干", 参数量差不多但结构不同 ——
   共用前缀至少会让"这个临时文件是哪一支的"变成只能靠猜。
*/
static const char *TMP_WEIGHTS_SACAZ_OLD_MOE = "weights/_temp_train_sacaz_old_moe";
static const char *TMP_WEIGHTS_DQNAB = "weights/_temp_train_dqnab";
/*
   [2026-09 dev-dqnmcts-moetb] DQN+MCTS (MoE+TB 专家) 那一支的临时前缀。
   **必须独立**: 它写出的文件名是 `<prefix>_trunk` / `<prefix>_q`, 而 DQN+AB 写的是
   `<prefix>_trunk` / `_v` / `_a` —— `_trunk` **同名**。共用前缀就是"两支持续互相
   覆盖主干" (同一时刻只有一支在跑, 但上一轮的残留会让本轮 loadModel 读到别的结构),
   与上面 PPO/SAC 那几段是同一条理由。
*/
static const char *TMP_WEIGHTS_DQNMCTS_MOE = "weights/_temp_train_dqnmcts_moe";

/* 这个 agent 的后台训练临时前缀 (见上面那段注释) */
static const char *tmpWeightsOf(ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_PPOMCTS_MLP: return TMP_WEIGHTS_PPOMCTS_MLP;
    case ChessBoard::AGENT_SACAZ:     return TMP_WEIGHTS_SACAZ;
    case ChessBoard::AGENT_SACAZ_MOE: return TMP_WEIGHTS_SACAZ_MOE;
    /* [2026-09 独立类] MoE+MLP 专家那一支的临时前缀: 与另外三支**必须不同** (同一条纪律) */
    case ChessBoard::AGENT_SACAZ_MOE_MLP: return TMP_WEIGHTS_SACAZ_MOE_MLP;
    case ChessBoard::AGENT_SACAZ_OLD: return TMP_WEIGHTS_SACAZ_OLD;
    case ChessBoard::AGENT_SACAZ_OLD_MOE: return TMP_WEIGHTS_SACAZ_OLD_MOE;
    case ChessBoard::AGENT_DQNAB:     return TMP_WEIGHTS_DQNAB;
    /* 独立前缀 (见上面那条注释: 它与 DQN+AB 的 _trunk 同名) */
    case ChessBoard::AGENT_DQNMCTS_MOE: return TMP_WEIGHTS_DQNMCTS_MOE;
    default:                          return TMP_WEIGHTS;
    }
}

/*
 * ================================================================
 *  applyRewardMethod —— "用哪一套奖励"的唯一一处写法
 * ================================================================
 *  用户口径: **保留旧的奖励方法**, 新的那一套只在显式打开时生效。
 *
 *  这个模板对**任何**带那四个旋钮的 SAC 类都成立（三个当前口径的类同名同义）:
 *    * `dynamic = false` -> `rewardShape = 0` = 旧奖励（材质 ×0.1 + 每步代价 + 终局 ±1）。
 *      这一支的代码一个字没动, 历史读数与它逐位对得上。
 *    * `dynamic = true`  -> `rewardShape = 3` + 出厂参数（三因子等权 / 0.5 / 0.5）。
 *
 *  三个按钮/克隆点都调它: 两个构造工厂、后台训练的 clone。
 *  写在同一处是刻意的 —— 三份实现各自手抄一遍, 迟早有一支漏掉,
 *  而"漏掉"的表现只是"这一支还在用旧奖励", 面板上完全看不出来。
 * ================================================================
 */
template <class A>
static void applyRewardMethod(A &a, bool dynamic)
{
    a.rewardShape = dynamic ? 3 : 0;
    a.mateScoreMode = 0;                 /* 0 = 数量+价值+优势 三因子等权 (出厂口径) */
    a.matRewardBoost = 0.5f;             /* 均势满盘时材质 ×1.5, 永不归零 */
    a.mateRewardBoost = 0.5f;            /* 残局/大势已定时终局 ×1.5, 永不归零 */
}

/*
 * ================================================================
 *  applyLossFreeBias —— "两个 MoE 支要不要开偏置均衡"的唯一一处写法
 * ================================================================
 *  与 applyRewardMethod 同一个理由: **四个点**(两个构造工厂 + 两个后台训练的 clone)
 *  各手抄一遍的话, 迟早有一支漏掉, 而"漏掉"的表现只是"这一支还在用旧路由",
 *  面板上完全看不出来 —— 更糟的是 clone 漏掉会让**训练用的那一支**与决策用的实例
 *  不是同一个东西 (权重往返看不出来, 因为偏置不进权重文件、不进参数量)。
 *
 *  只对两个 MoE 独立类成立 (它们有同名同义的这两个成员); 纯 MLP / legacy 支没有
 *  稀疏 MoE 层 —— 所以**不要**把这个模板用到它们上面 (调用点只有那四个)。
 * ================================================================
 */
template <class A>
static void applyLossFreeBias(A &a)
{
    /*
       走 setter 而不是直接写两个成员 (2026-10): 单一真源在 `RL::PPO` / agent 内部的
       MoE 层上, "只改了 agent 的成员"是**静默不生效**的 —— 那正是本工程反复栽的
       "旋钮没接线"那一类。`SACAZMoETbAgent` / `SACAZMoEMlpAgent` / `PPOMCTSAgent`
       三者都有这个同名同签名的 setter, 所以这一个模板对三支都成立。
    */
    a.setLossFreeBias(SACAZ_MOE_LOSSFREE, SACAZ_MOE_LOSSFREE_RATE);
}

/*
 * ================================================================
 *  [2026-10] PPO 两支的同一套"均衡 + 门控"接线 (唯一一处写法)
 * ================================================================
 *  为什么必须写成函数而不是每个构造点手抄: PPO 两支在 chessboard.cpp 里有**七个**
 *  构造点 (预加载 ×2、三个按需决策实例、后台兜底建网、后台训练 clone ×2)。手抄七遍
 *  迟早漏一处, 而"漏掉"的表现只是"这一支还在用旧路由" —— 面板上完全看不出来;
 *  clone 漏掉更糟: 训练用的那一支与决策用的实例就不是同一个东西 (权重往返看不出来,
 *  因为 `biasGate` 是控制回路状态, 不进权重文件、不进参数量)。
 *  ⚠ `enableMlpGate` 必须在**任何前向之前**调用 (它会重建并重新初始化门控权重),
 *  所以只能在构造点之后立刻调 —— 这里就是。
 * ================================================================
 */
template <class A>
static void applyPpoBalance(A &a)
{
    a.setLossFreeBias(PPO_MOE_LOSSFREE, PPO_MOE_LOSSFREE_RATE);
    if (PPO_MOE_MLP_GATE_HIDDEN > 0) {
        a.enableMlpGate(PPO_MOE_MLP_GATE_HIDDEN);
    }
}

/*
 * ================================================================
 *  [2026-10 门控实验] MLP 门控 (三层: d_model -> hidden -> E)
 * ================================================================
 *  把骨干的**线性门控**（`wg·x + bg`，即 4 个超平面切状态空间）换成
 *  `d_model -> hidden (tanh) -> E -> softmax` 的 MLP 门控。
 *
 *  ---- 为什么默认是 0（关）而不是像偏置均衡那样默认开 ----
 *  三条，都是实测而不是偏好:
 *   1. **它会让现有权重文件作废。** 线性门控的 `wg` 是 (E, d_model)，MLP 门控是
 *      `wg1 (hidden,d_model) + wg2 (E,hidden)` —— 形状不同，`Net::load` 的参数量守卫
 *      会**明确拒绝**。也就是说：打开它的那一刻，界面上这两支 agent 的存量模型
 *      （weights/sacaz_moe_tb* 与 sacaz_moe_mlp*）全部载不进来，只能从随机初始化重练。
 *      偏置均衡没有这个问题（它是非参数缓冲），所以它可以默认开。
 *   2. **棋力没有结论。** 20 局自对弈 + 8 局对 MCTS-800 的对照里，得分率
 *      50.0% -> 43.8%，95% 区间 [31.5, 56.0] **跨过 50%**；而且同一次运行里
 *      "材料差"的方向与得分率**相反**（−0.24 -> +0.40）。n=8 不足以谈差别。
 *   3. **推理侧的负载均衡反而更差**（这是反直觉但可解释的一条）：
 *      训练侧 MaxVio 0.049 -> **0.009**（更均衡），而评测（MCTS-800 对局）侧的
 *      MaxVio 0.264 -> **0.598**（更不均衡），最弱专家份额 19.65% -> 12.35%。
 *      原因：偏置只在 `learnBatch` 的批边界更新，它均衡的是**训练分布**；MLP 门控
 *      表达力更强 ⇒ 能把训练分布拟合得更平，但换到对局分布上更偏。
 *      **"偏置均衡"不等于"到处都均衡"。**
 *
 *  ---- 要启用就把它改成 >0（例如 64）----
 *  `static constexpr int SACAZ_MOE_MLP_GATE_HIDDEN = 64;`
 *  然后**必须重训**这两支（旧权重会被拒），并用锚点 Elo 判定
 *  （`bench_anchor`，~80 Elo 分辨力；8 局分辨不出来）。
 *
 *  成本（实测，20 局自对弈）: 参数量 +162,312（+0.28%），训练耗时 +0.8%
 *  —— 门控本身几乎免费，代价全在"旧权重作废"与"训练难度"上。
 * ================================================================
 */
static constexpr int SACAZ_MOE_MLP_GATE_HIDDEN = 0;

/*
 *  与 applyLossFreeBias 同一个理由写成一处: **四个点**(两个构造工厂 + 两个后台训练
 *  clone) 各手抄一遍的话迟早漏一支, 而"漏掉"的表现只是"这一支还在用线性门控",
 *  面板上完全看不出来 —— 更糟的是 clone 漏掉会让训练用的那一支与决策实例不是同一个东西。
 *  ⚠ `enableMlpGate` 必须在**任何前向之前**调用, 所以只能在构造点后立刻调 (这里就是)。
 */
template <class A>
static void applyMlpGate(A &a)
{
    if (SACAZ_MOE_MLP_GATE_HIDDEN > 0) {
        a.enableMlpGate(SACAZ_MOE_MLP_GATE_HIDDEN);
    }
}

/*
 * ================================================================
 *  createSACAZAgent / createSACAZLegacyAgent —— SAC 三支的**唯一**构造点
 * ================================================================
 * 为什么必须只有一处: 三支的**参数结构完全相同** (都是 iFcLayer 的 w/b),
 * 所以"构造错了哪一支 / 少传了哪个参数"不会让 save/load 失败 —— 它会**静默地**按另一套
 * 口径跑 (AGENT_SACAZ_MOE 少传 backbone 就是 TB->MLP 的静默换骨干;
 * AGENT_SACAZ_OLD 建成当前口径那一支, 两者连参数量都一样)。
 * 本文件已经因为"三处各写一遍构造参数"栽过一次 (见 weightFilesOf 的注释), 所以:
 *   * 决策路径 (aiThinkRaw / aiThinkForAgentRaw)、启动预加载、后台训练的兜底建网、
 *     以及每轮的训练 clone, 全部走这两个函数;
 *   * AGENT_SACAZ_OLD 构造的是**独立类** SACAZLegacyAgent (口径硬编码在那边的构造函数里),
 *     界面这一层不手抄它的口径值。
 *
 * ---- [2026-09 拆分] 为什么是**两个**函数而不是一个 ----
 * 还原版与当前口径现在**没有任何继承关系** (用户口径: "用不同的 C++ 类把新旧 SAC agent
 * 区分开", 见 sacazlegacyagent.h 的头注释), 于是两者的指针类型互不兼容 ——
 * 一个返回 `SACAZAgent *` 的函数**装不下** SACAZLegacyAgent。这正是要的效果:
 * 谁想让两支共用一条构造路径, 编译器当场拦下来, 不用靠注释提醒。
 * 反过来, 也**不许**用 reinterpret_cast / static_cast 或"改回继承"来消除这个错误 ——
 * 那会静默地把还原版变回当前口径 (两者的权重结构相同, save/load 不会报错)。
 *
 * 返回 nullptr = 这个 agent 类型不是**本函数负责的那一支** (调用方不该走到这里)。
 */
static SACAZAgent *createSACAZAgent(Chess &board, ChessBoard::AgentType type,
                                   bool dynamicReward = false)
{
    switch (type) {
    case ChessBoard::AGENT_SACAZ: {
        SACAZAgent *a = new SACAZAgent(board, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
        /* [2026-09 奖励方法开关] 新建的实例按当前开关写一遍 (默认 = 旧奖励) */
        applyRewardMethod(*a, dynamicReward);
        return a;
    }
    default:
        return nullptr;
    }
}

/*
 * createSACAZMoETbAgent —— 稀疏 MoE + **TB 专家**那一支的**唯一**构造点
 * (界面类型 AGENT_SACAZ_MOE)。
 *
 * **[2026-09 独立类拆分]** 这一支原来是 `SACAZAgent` + `Backbone::SparseMoeTb`,
 * 现在它是**自己的类** `SACAZMoETbAgent` (src/sacazmoetbagent.h/.cpp):
 * 骨干在那边硬编码, 与纯 MLP / MoE-MLP / 59e5233 还原版**互不继承** —— 改一个文件
 * 碰不到另一个。构造参数与拆分前**逐字对应** (只是少了一个 Backbone 实参)。
 *
 * 这里只给**形状参数**: hidden 64 / gamma 0.99 / lr 0.001 / cpuct 1.5, 与 AGENT_SACAZ
 * 那一支逐字相同, 于是"口径 vs 骨干"这两个变量不会混在一起。
 *
 * `shared` = 是否用**共享骨干 + 三头**口径 (TrunkMode::Shared):
 *   * AGENT_SACAZ_MOE (界面那一支) 用 Shared;
 *   * 后台自对弈快照 / 兜底构造用 Separate (与拆分前 m_sfSACAZMoe 那条路径一致)。
 */
static SACAZMoETbAgent *createSACAZMoETbAgent(Chess &board, bool shared,
                                            bool dynamicReward = false)
{
    /*
       [2026-09 dev-sacmoetb] 这一支用**共享骨干 + 三头**口径 (TrunkMode::Shared)。
       两个改动一起进, 它们的账要分开记:

       (1) **TB 专家的头数口径** (tbHonorHeads: 默认就是修好的口径, 这里不传)。
           `MOE_TB_HEADS = 15` 原来在 `d_model = STATE_DIM = 1263 = 3 x 421` 上被
           "头数必须整除 d_model" 的规则**静默降成 3 个头** (实测单专家前向
           15.7 ms vs 5.4 ms, 3.0 倍; 而且 15 个 head 对象全都分配、只用 3 个)。
           这是修 bug, 所以默认就是修好的; 想复现旧读数要显式传 false。

       (2) **共享骨干**: 原来五张网各自背一整套骨干 (TB 专家下 28.8 M 参数/张,
           合计 144 M)。现在骨干只留在线/目标各一份, actor/q1/q2 退化成三个只有
           输出层的头。实测 (test_sacaz [17B], 同权重同局面, **当时是 E=4/top-1**):
               参数量      143,903,940 -> 57,586,536   (2.50x 少)
               learnBatch  649.9 ms/批 -> 225.6 ms/批  (2.88x 快)
               selectMove  9.3 ms/模拟 -> 4.1 ms/模拟   (2.24x 快)
           而且 π / Q / V **逐位相同** (把独立口径的三个骨干也设成同一份之后)。
           于是"同一套算法、同样的预算"下能跑的模拟次数从 ~19 变成 ~42。

       ⚠ **[2026-10 E/top-k 又切回 4/1] 上面那三行数字现在是"当前档"的读数, 因为
            `MOE_TB_EXPERTS/MOE_TB_TOPK` 已经按用户口径降回 4/1。** 中间那一版
           (E=8/top-2) 的读数保留在这里做对照 —— 它正是"要不要再提上去"的依据:
               唯一参数量 57,586,536 (E=4/1) -> **114,969,680** (E=8/2, 2.00x, 实测);
               峰值工作集 577 MB -> **1,138 MB** (1.97x, 实测);
               learnBatch 225.6 ms/批 -> 约 **450 ms/批** (32 条 ≈ 3.6 s/手, top-2 翻倍)。
           工作集的量法 (E=8 那个数; 换成 E=4 只需同命令重跑):
               bench_sacmoetb_train --train-games=2 --train-sims=8 --train-plies=24
                 --learn-every=4 --openings=1 --opening=4 --eval-plies=20 --eval-sims=8
                 --opponent=ab --aux=0.1 --lossfree-bias=1 --lossfree-bias-rate=0.01
           跑的时候每 0.4 s 采一次 WorkingSet64/PeakWorkingSet64 取最大
           (E=8 那次 = **1,138 MB**, 日志 build/gate_moe_out/E8k2_workingset.txt)。
           改这两个常量时**这里也要改**, 见 `sacazmoetbagent.h` 的 MOE_TB_EXPERTS。

       **权重文件必须换前缀** (weights/sacaz_moe_shared_agent): 共享口径写 4 个文件
       (trunk + 三个头), 独立口径写 3 个 (_actor/_q1/_q2), 两者的 `_q1` 语义完全
       不同 —— 共用一个前缀等于给"载错文件"留门。见 sacazagent.h 的
       sharedWeightPrefix 说明。**旧权重载不进这一支**, 它要从头训练。
    */
    SACAZMoETbAgent *a = new SACAZMoETbAgent(
        board, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f, 64, SACAZ_MOE_AUX,
        shared ? SACAZMoETbAgent::TrunkMode::Shared
               : SACAZMoETbAgent::TrunkMode::Separate);
    /*
       ---- 这个变体**不**开"从搜索学一次" (2026-09 用户口径) ----
       一次 learnBatch(32) 在稀疏 MoE+TB 骨干上实测 **228 ms/样本**(见 test_sacaz [12d]),
       32 条就是 **7.3 s/手** —— 加上界面本来就有的那一轮, 一步要十几秒。MLP 骨干那一支
       是 0.95 ms/样本(32 条约 30 ms), 所以那条路径的开销可以忽略, 这一条不行。
       等它有了便宜的头/批(见 docs 里的待办)再一起打开。

       ---- [2026-09 dev-sacmoetb] 那个"待办"已经做了一半, 但这一行仍然保留 ----
       共享骨干 + 头数口径之后, 同一台机器上实测 learnBatch 从 649.9 降到 225.6 ms/批
       (2.88x, test_sacaz [17B])。但 32 条仍然要 ~1.8 s/手 —— 对 175 ms/步的界面预算
       还是太贵, 所以这一行**不动**。要开就删掉它, 并用
       `bench_sacaz_vs_ab --trunk=shared --budget=175` 复核一步的真实耗时。
    */
    a->learnFromSearch = false;
    /*
       [2026-10] 但**终局通道必须留着**: learnFromSearch 关掉之后, 池里就再也没有
       hasSearch=true 的样本, 而 `notifyGameResult`(人机对弈终局) 正是要把结果挂到
       那样一条样本上 —— 关着 learnFromSearch 且不记录的话, 人把 AI 将死的那一局
       输赢被直接丢掉 (用户实测日志: "没有可挂的真实决策样本 ... 这一局的输赢没有进
       学习回路")。这正是文档里"解决终局样本稀疏"的四个出口之一 (§11 人机对弈通道)。
       记录本身几乎免费 (一次稀疏编码 + π 拷贝); 那批样本会在下一次 learnBatch
       (界面上是"走子前先探索+预训练"那条路径) 被一起训练。
    */
    a->recordDecisionSamples = true;
    /* [2026-09 奖励方法开关] 同上 (默认 = 旧奖励) */
    applyRewardMethod(*a, dynamicReward);
    /* [2026-10 门控实验] 无辅助损失偏置均衡 (见 SACAZ_MOE_LOSSFREE 的说明) */
    applyLossFreeBias(*a);
    /* [2026-10 门控实验] MLP 门控 (默认关; 见 SACAZ_MOE_MLP_GATE_HIDDEN 的说明)。
       必须在任何前向之前装上, 所以就在构造点后立刻调。 */
    applyMlpGate(*a);
    return a;
}

/*
 * createSACAZMoEMlpAgent —— 稀疏 MoE + **MLP 专家**那一支的**唯一**构造点
 * (界面类型 AGENT_SACAZ_MOE_MLP)。
 *
 * **[2026-09 独立类拆分]** 它是独立类 `SACAZMoEMlpAgent`。这一支以前**只能在 bench 里
 * 跑到** (拆分前它是 `Backbone::SparseMoeMlp` 的取值), 现在接进界面下拉框 ——
 * "两个独立类都在产品路径上"这句话才算成立。
 *
 * 口径与另外两支**逐字对齐**: hidden 64 / gamma 0.99 / lr 0.001 / cpuct 1.5、
 * expertHidden 64 + aux = SACAZ_MOE_AUX; `shared` = 共享骨干 + 三头口径。
 * 实测这一支比 TB 便宜得多 (2.90 M 参数 / ~2.9 ms/模拟 vs 57.6 M / ~5.4 ms),
 * 所以界面模拟次数按 SACAZ_MOE_MLP_SIMS 给。
 */
static SACAZMoEMlpAgent *createSACAZMoEMlpAgent(Chess &board, bool shared,
                                              bool dynamicReward = false)
{
    SACAZMoEMlpAgent *a = new SACAZMoEMlpAgent(
        board, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f, 64, SACAZ_MOE_AUX,
        shared ? SACAZMoEMlpAgent::TrunkMode::Shared
               : SACAZMoEMlpAgent::TrunkMode::Separate);
    /*
       "从自己的搜索学一次"这一支**开着**: MoE-MLP 的 learnBatch 便宜得多
       (TB 那一支实测 225.6 ms/批 ⇒ 32 条约 1.8 s/手, 所以那边必须关掉)。
       开着的收益是 AlphaZero 的核心信号 (π_MCTS 进策略目标), 见 sacazagent.h 那段说明。
    */
    a->learnFromSearch = true;
    /* [2026-09 奖励方法开关] 同上 (默认 = 旧奖励) */
    applyRewardMethod(*a, dynamicReward);
    /* [2026-10 门控实验] 无辅助损失偏置均衡 (与 TB 支同一口径) */
    applyLossFreeBias(*a);
    /* [2026-10 门控实验] MLP 门控 (与 TB 支同一口径; 默认关) */
    applyMlpGate(*a);
    return a;
}

/*
 * createSACAZLegacyAgent —— 59e5233 行为还原版 (AGENT_SACAZ_OLD / AGENT_SACAZ_OLD_MOE)
 * 的**唯一**构造点。
 *
 * 为什么单独一个函数: 见上面那段 (两个类互不兼容, 不能合用一个返回类型)。
 * 这里只给**形状参数**: hidden 64 / gamma 0.99 / lr 0.001 / cpuct 1.5, 与 AGENT_SACAZ
 * 那一支逐字相同, 于是"口径 vs 骨干"这两个变量不会混在一起。骨干由界面类型决定:
 *   * AGENT_SACAZ_OLD     -> Mlp (与 AGENT_SACAZ 同骨干、同表示, 差别只剩口径);
 *   * AGENT_SACAZ_OLD_MOE -> SparseMoeTb + expertHidden 64 + aux SACAZ_MOE_AUX
 *                            (与 AGENT_SACAZ_MOE 逐字相同, 差别同样只剩口径)。
 * 口径 (目标熵 0.98 / alpha 学习率 1e-3 / critic 目标不夹 + 纯 MSE / 叶子全量估值 /
 * 目标网 tau=1e-3 每 64 步 / 没有"从自己的搜索学一次") 全部硬编码在
 * SACAZLegacyAgent 自己的构造函数与实现里, 界面这一层一个字都不手抄。
 * 本类**没有**任何奖励塑形开关, 所以这里也没有可传的口径参数。
 *
 * 返回 nullptr = 这个 agent 类型不是**这两支中的任何一个** (调用方不该走到这里)。
 */
static SACAZLegacyAgent *createSACAZLegacyAgent(Chess &board, ChessBoard::AgentType type)
{
    switch (type) {
    case ChessBoard::AGENT_SACAZ_OLD:
        return new SACAZLegacyAgent(board, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
    case ChessBoard::AGENT_SACAZ_OLD_MOE:
        /*
           TB 专家骨干: 参数与 AGENT_SACAZ_MOE **逐字相同** (只有类不同)。
           注意本类**没有** learnFromSearch 那条路径, 所以 SACAZAgent 那一支在这里的
           "关掉从搜索学一次" 那行代码在本函数里没有对应物 —— 还原版本来就是只读搜索。
        */
        return new SACAZLegacyAgent(board, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                    SACAZLegacyAgent::Backbone::SparseMoeTb,
                                    64, SACAZ_MOE_AUX);
    default:
        return nullptr;      /* 不是这两支 (调用方不该走到这里) */
    }
}

/* 这个 agent 类型的后台训练每步给多少模拟次数 (见 BG_TRAIN_SACAZ* 的注释) */
static int sacazTrainSims(ChessBoard::AgentType type)
{
    /*
       TB 专家骨干的两支 (AGENT_SACAZ_MOE / AGENT_SACAZ_OLD_MOE) 走 64 次那一档:
       一次模拟 ~10.9 ms, 界面预算 (16 次) 对**训练**不够 —— 训练要的是 π 目标质量。
    */
    return (type == ChessBoard::AGENT_SACAZ_MOE || type == ChessBoard::AGENT_SACAZ_OLD_MOE)
               ? BG_TRAIN_SACAZ_MOE_SIMS
               : BG_TRAIN_SACAZ_SIMS;
}

/*
 * trainSACRound —— SAC 三支共享的"一轮训练"往返
 *   clone 载入种子权重 -> 自对弈 -> 写回临时文件
 *
 * 三支的这份往返**逐字相同** (只有模拟次数不同), 所以写成一处; 返回 false = 本轮不能
 * 同步回主 agent (载入或写回失败)。诊断里带上 agent 名字: 三支共用前缀会让日志里的
 * "SAC 载入失败"分不清是哪一支。
 *
 * [2026-09] 这里是个**模板**, 不是 `SACAZAgent &`: AGENT_SACAZ / AGENT_SACAZ_MOE 的
 * clone 是 SACAZAgent, 而 AGENT_SACAZ_OLD 的是 SACAZLegacyAgent —— 两个**没有继承
 * 关系**的类 (见 createSACAZAgent 的注释)。参数只要满足"有 loadModel / trainSelfPlay /
 * getLastTrainLoss / saveModel"就能用, 这比给两个类各写一遍往返、或硬塞一个公共基类都小,
 * 也不会让两支的口径有机会混起来。
 */
template <class AgentT>
static bool trainSACRound(AgentT &clone,
                          const char *tmpWeights,
                          const QString &label,
                          int episodes,
                          int sims,
                          int maxMoves,
                          float &lossOut)
{
    if (!clone.loadModel(tmpWeights)) {
        qWarning() << "[train]" << label << "clone 载入种子权重失败, 本轮丢弃"
                   << "(临时文件与当前网络口径不匹配? 路径" << tmpWeights << ")";
        return false;
    }
    clone.trainSelfPlay(episodes, sims, maxMoves, false);
    lossOut = clone.getLastTrainLoss();
    if (!clone.saveModel(tmpWeights)) {
        qWarning() << "[train]" << label << "训练权重写回失败, 本轮丢弃";
        return false;
    }
    return true;
}

/*
 * ---- 有权重文件的 agent 名单 (单一来源) ----
 * 顺序 = "启动加载顺序" = "自检面板的列举顺序": 大的/慢的放后面, 这样启动日志里
 * 一眼能看出是哪一组在耗时 (稀疏 MoE 那一组是 3 x 146 MB)。
 *
 * [2026-10 用户口径] **PG / DQN / SAC 行为还原版(两支) 移出这张表** ——
 * 用户要求"界面下拉框移除 PG, DQN, SAC legacy agent, 减少加载时间"。界面上不再可选,
 * 启动时也就不该再为它们读盘 + 建网 (实测这两件事在它们身上要 ~11.2 s)。
 * 类、权重前缀、bench/测试全都留着; 自检面板的横向列举也随之不再列它们
 * (见 mainwindow.cpp 的 kAll 与 kAgents 两处注释里的同一段说明):
 *   * PG                 —— 实测 146 ms (权重只 8.6 MB)
 *   * DQN                —— 实测 171 ms (权重只 8.6 MB)
 *   * AGENT_SACAZ_OLD    —— 实测 43 ms (MLP 还原版, 权重 0.5 MB x3)
 *   * AGENT_SACAZ_OLD_MOE —— **实测 建网 4499 ms + 读权重 6300 ms** (5 个 TB 专家网络
 *                             + 3 x 146 MB), 这一支就是省下来的大头
 * 要恢复某一条: 这里加回枚举 + 下面那段加载代码加回对应分支 (两处都要)。
 */
static const ChessBoard::AgentType kWeightAgents[] = {
    ChessBoard::AGENT_EVAB,
    ChessBoard::AGENT_PPOMCTS_MLP,
    ChessBoard::AGENT_DQNMCTS,
    ChessBoard::AGENT_PPOMCTS,
    ChessBoard::AGENT_SACAZ,
    ChessBoard::AGENT_DQNAB,
    ChessBoard::AGENT_SACAZ_MOE,
  /*
     [2026-09 独立类] MoE + MLP 专家那一支: 排在 TB 那一组之后。
     它比 TB 小得多 (2.9 M 参数), 但也属于"稀疏 MoE 那一组", 所以紧跟其后读起来最清楚。
  */
  ChessBoard::AGENT_SACAZ_MOE_MLP,
    /*
       [2026-09 dev-dqnmcts-moetb] DQN+MCTS (MoE+TB 专家) 排在最后: 它也是
       "3 x 146 MB 量级"的那一组 (主干 + 头), 放在大的那一组之后读启动日志最清楚。
    */
    ChessBoard::AGENT_DQNMCTS_MOE
};

/*
 * weightFilesOf - 这个 agent 的 saveModel(prefix) 到底会写出哪些文件。
 *
 * 为什么要有它: "前缀"和"真实文件名"的关系原来散在三处
 *   (a) 启动扫描权重的那张表, (b) defaultWeightPath(), (c) 各 agent 的 saveModel)。
 * 三份知识一旦漂移就是**静默失效**: PPO+MCTS 的扫描表探测的是裸的
 * `weights/ppomcts_agent.dat`, 而它写出的是 `..._actor` / `..._critic` ——
 * 于是那 279 MB x 2 的权重从来没被载入过, 界面上一句报错都没有。
 * 现在只有这一个函数知道命名规则, 启动扫描与自检面板都用它。
 */
static std::vector<std::string> weightFilesOf(ChessBoard::AgentType type)
{
    const std::string p = ChessBoard::defaultWeightPath(type);
    if (p.empty()) {
        return std::vector<std::string>();      /* 纯搜索 agent: 没有权重可存 */
    }
    switch (type) {
    /* 一个模型两个文件: actor + critic (PPO 系的两个骨干都是这样) */
    case ChessBoard::AGENT_PPOMCTS:
    case ChessBoard::AGENT_PPOMCTS_MLP:
        return { p + "_actor", p + "_critic" };
    /* 一个模型三个文件: actor + 双 critic */
    case ChessBoard::AGENT_SACAZ:
    case ChessBoard::AGENT_SACAZ_MOE:
    case ChessBoard::AGENT_SACAZ_MOE_MLP:
    case ChessBoard::AGENT_SACAZ_OLD:
    case ChessBoard::AGENT_SACAZ_OLD_MOE:
        return { p + "_actor", p + "_q1", p + "_q2" };
    /* 一个模型三个文件: 主干 + V 头 + A 头 */
    case ChessBoard::AGENT_DQNAB:
        return { p + "_trunk", p + "_v", p + "_a" };
    /*
       一个模型三个文件: 主干 + Q 头 + 第二个 Q 头 (twinCritic 用)。
       ⚠ 第二个头**无条件写**: 文件个数与名字不许依赖运行时开关 —— 否则载入方
       就得猜"这一次有没有 _q2", 而猜错的表现是"载入失败"或者更坏: 半载入。
       (类里 `twinCritic` 默认关, 但头照样存在, 打开时立刻生效。)
       ⚠ 两个家族的 `_trunk` 同名 —— 所以前缀必须不同 (见 TMP_WEIGHTS_DQNMCTS_MOE
       与 defaultWeightPath() 的说明)。
    */
    case ChessBoard::AGENT_DQNMCTS_MOE:
        return { p + "_trunk", p + "_q", p + "_q2" };
    /* 单文件 */
    default:
        return { p };
    }
}

/* Static member initialization */
PGEagent *ChessBoard::m_sfPG = nullptr;
DQNAgent *ChessBoard::m_sfDQN = nullptr;
PPOMCTSAgent *ChessBoard::m_sfPPOMCTS = nullptr;
PPOMCTSAgent *ChessBoard::m_sfPPOMCTSMLP = nullptr;
DQNMCTSAgent *ChessBoard::m_sfDQNMCTS = nullptr;
EVABAgent *ChessBoard::m_sfEVAB = nullptr;
SACAZAgent *ChessBoard::m_sfSACAZ = nullptr;
SACAZMoETbAgent *ChessBoard::m_sfSACAZMoe = nullptr;   /* 独立类: 稀疏 MoE + TB 专家 */
SACAZMoEMlpAgent *ChessBoard::m_sfSACAZMoeMlp = nullptr; /* 独立类: 稀疏 MoE + MLP 专家 */
SACAZLegacyAgent *ChessBoard::m_sfSACAZOld = nullptr;      /* 独立类, 不是 SACAZAgent 的派生类 */
SACAZLegacyAgent *ChessBoard::m_sfSACAZOldMoe = nullptr;   /* 同一口径 + TB 专家骨干 */
DQNABAgent *ChessBoard::m_sfDQNAB = nullptr;
/* DQN+MCTS (MoE+TB 专家) 那一支: 独立类 (src/dqnmctsmoetbagent.h/.cpp) */
DQNMCTSMOETbAgent *ChessBoard::m_sfDQNMCTSMOE = nullptr;
std::map<ChessBoard::AgentType, std::string> ChessBoard::s_weightPaths;

/*
 * startupLoad - 异步加载数据库和AI模型权重
 *
 * 在后台线程中运行, 完成以下工作:
 *   1. 打开 SQLite 棋局数据库
 *   2. 扫描 weights/ 目录下的预训练权重文件
 *   3. 预创建 agent 实例并加载权重
 *
 * 完成后通过 QMetaObject::invokeMethod 在主线程发送
 * startupComplete 信号, 通知 MainWindow 启用界面.
 */
void ChessBoard::startupLoad()
{
    /*
       报告本构建实际启用的 SIMD 指令集。SIMD 内核是编译期选的 (见 rl/cpuinfo.hpp),
       所以"这个二进制到底会不会用 AVX2"没法靠猜 —— 启动时打一行日志, 排查性能
       问题时就不用先怀疑环境。
    */
    qInfo().noquote() << "[SIMD]" << QString::fromStdString(RL::cpuinfo::describe());

    /*
       启动阶段的"请稍候": 读 7 组权重可能要好几秒 (老格式是十进制文本, DQN+MCTS
       一个文件就 16 MB), 界面上必须有个沙漏在转。这些信号是队列投递到 GUI 线程的,
       所以在这里(工作线程)emit 是安全的。
    */
    emit busyStarted(QStringLiteral("正在载入"),
                     QStringLiteral("读取棋局数据库与模型权重…"));

    /* ---- 1. 打开数据库 ---- */
    GameDatabase::instance().open("chess_games.db");

    /* ---- 2. 扫描权重文件 ---- */
    /*
       ---- 这一段的教训 (2026-09): "写出来的文件名"必须只有**一个来源** ----
       启动扫描的表、defaultWeightPath()、各 agent 的 saveModel(prefix) 原来是三份
       各自维护的名字。PPO+MCTS 就栽在这里: saveModel(prefix) 写的是
       `weights/ppomcts_agent.dat_actor` / `_critic`, 而扫描表里探测的是裸的
       `weights/ppomcts_agent.dat` —— 于是它的权重**从来没在启动时被载入过**
       (磁盘上确实有那 279 MB 的两个文件), 界面上却看不出任何异常。
       现在扫描的名字由 weightFilesOf() 统一给出, 它就是"saveModel(prefix) 会写出
       哪些文件"的同一份表述; 自检面板显示的也是这一份。
    */
    for (AgentType t : kWeightAgents) {
        for (const std::string &f : weightFilesOf(t)) {
            std::ifstream probe(f, std::ios::binary);
            if (probe.good()) {
                /* s_weightPaths 存的是**前缀** (单文件 agent 就是文件本身):
                   下面 load*() 都按"前缀"用 (PPO 系再各自拼 _actor/_critic)。 */
                s_weightPaths[t] = defaultWeightPath(t);
                break;
            }
        }
    }

    /* ---- 3. 预创建 self-play agent 并加载权重 ----
       每一组都记一条耗时日志: 权重可能有几百 MB, 启动慢的时候要能一眼看出是哪一组
       (实测: 稀疏 MoE 那 3 个 146 MB 文件占了大头, 其余六组加起来不到 1 秒)。 */
    auto loadClock = std::chrono::steady_clock::now();
    auto logLoad = [&loadClock](const char *what) {
        const auto now = std::chrono::steady_clock::now();
        const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - loadClock).count();
        qInfo().noquote() << QStringLiteral("[weights] %1: %2 ms")
                                 .arg(QString::fromUtf8(what)).arg(ms);
        loadClock = now;
    };
    /*
       [2026-10 用户口径] 这里原来还有 PG / DQN 两段预加载 (各 ~0.15 s), 与后面
       SAC 59e5233 还原版那两段。用户要求"下拉框移除 PG, DQN, SAC legacy agent,
       减少加载时间" ⇒ 这四支既不预热也不在界面上出现 (名单见 kWeightAgents 的注释)。
       它们的类与权重文件都还在: 真要用 (`decideOnEnvRawLocked` 的对应 case 仍在,
       会按需构造), 或者靠 bench/测试自己 new —— 只是不再为它们付启动时间。
    */
    if (s_weightPaths.count(AGENT_PPOMCTS)) {
        emit busyMessage(QStringLiteral("正在载入 PPO+MCTS 权重…"));
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            /* MoE 均衡 + 门控接线 (必须在任何前向之前; 见 applyPpoBalance) */
            applyPpoBalance(*m_sfPPOMCTS);
        }
        m_sfPPOMCTS->loadModel(s_weightPaths[AGENT_PPOMCTS]);
        logLoad("PPO+MCTS");
    }
    if (s_weightPaths.count(AGENT_EVAB)) {
        emit busyMessage(QStringLiteral("正在载入 EVAB 权重…"));
        if (m_sfEVAB == nullptr)
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
        m_sfEVAB->loadModel(s_weightPaths[AGENT_EVAB]);
        logLoad("EVAB");
    }
    if (s_weightPaths.count(AGENT_DQNMCTS)) {
        emit busyMessage(QStringLiteral("正在载入 DQN+MCTS 权重… (文件较大，可能要几秒)"));
        if (m_sfDQNMCTS == nullptr)
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        m_sfDQNMCTS->loadModel(s_weightPaths[AGENT_DQNMCTS]);
        logLoad("DQN+MCTS");
    }
    if (s_weightPaths.count(AGENT_SACAZ)) {
        emit busyMessage(QStringLiteral("正在载入 SAC+AZ 权重…"));
        if (m_sfSACAZ == nullptr)
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
        /* 路径是前缀: 去掉结尾的 "_actor" */
        std::string prefix = s_weightPaths[AGENT_SACAZ];
        const std::string suffix = "_actor";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfSACAZ->loadModel(prefix);
        logLoad("SAC+AZ");
    }
    if (s_weightPaths.count(AGENT_SACAZ_MOE)) {
        /*
           用户要求"程序启动时加载所有模型": 这个变体也在这里预加载 (以前为了启动速度
           改成了懒加载 —— 它一个模型是三个文件、每个 146 MB)。
           代价是启动变慢, 所以: (a) 它在七组权重里放**最后**, 前面几组几秒就完事、
           用户能在沙漏窗里看到进度一条条过; (b) 期间由"请稍候"沙漏窗给反馈;
           (c) 下面每载一组都记一条耗时日志, 慢了能一眼看出是哪一组。
        */
        emit busyMessage(QStringLiteral("正在载入 SAC+AZ (稀疏MoE) 权重… (3 个 146 MB 文件)"));
        if (m_sfSACAZMoe == nullptr)
            m_sfSACAZMoe = createSACAZMoETbAgent(env, false, m_dynamicReward);
        /* 拆开量: 5 个网络 x 28.7 M 参数的"分配 + 随机初始化"本身就不是小数目 */
        logLoad("SAC+AZ-MoE 建网(5 x 28.7M 参数)");
        std::string prefix = s_weightPaths[AGENT_SACAZ_MOE];
        const std::string suffix = "_actor";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfSACAZMoe->loadModel(prefix);
        logLoad("SAC+AZ-MoE 读权重(3 x 146 MB)");
    }
    if (s_weightPaths.count(AGENT_SACAZ_MOE_MLP)) {
        /*
           [2026-09 独立类] MoE+MLP 那一支的预加载: 与 TB 那一块对称, 只换类/前缀/文件数。
           共享口径写 4 个文件 (trunk + 三个头), 所以拨掉的是 "_actorhead" 后缀
           (不是独立口径的 "_actor")。
        */
        emit busyMessage(QStringLiteral("正在载入 SAC+AZ (稀疏MoE-MLP) 权重…"));
        if (m_sfSACAZMoeMlp == nullptr)
            m_sfSACAZMoeMlp = createSACAZMoEMlpAgent(env, false, m_dynamicReward);
        logLoad("SAC+AZ-MoE-MLP 建网(稀疏 MoE + MLP 专家)");
        std::string prefix = s_weightPaths[AGENT_SACAZ_MOE_MLP];
        const std::string suffix = "_actorhead";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfSACAZMoeMlp->loadModel(prefix);
        logLoad("SAC+AZ-MoE-MLP 读权重(4 个文件)");
    }
    /*
       [2026-10 用户口径] 这里原来是 SAC 的 59e5233 **行为还原版**两支的预加载
       (MLP 骨干 43 ms + MoE/TB 骨干 "建网 4499 ms + 读权重 6300 ms")。
       与 PG/DQN 一起从启动路径上摘掉了 —— 省下来的 ~11.2 s 里, **这一支占 10.8 s**。
       还原版的能力没有被删: `SACAZLegacyAgent` 两个 Backbone 都在,
       `test_sacaz` [14] 节与 `bench_sacaz_vs_ab --legacy` 仍在跑它们; 界面上只是选不到。
    */
    if (s_weightPaths.count(AGENT_PPOMCTS_MLP)) {
        /*
           PPO+MCTS 的 MLP 专家变体 (2026-09)。它排在 TB 那一支**前面**: MLP 专家
           的权重小得多 (约 3.6 M 参数, 几 MB), 读起来是毫秒级, 放在大的那组之前
           能让启动日志的顺序更好读。
        */
        emit busyMessage(QStringLiteral("正在载入 PPO+MCTS (MLP专家) 权重…"));
        if (m_sfPPOMCTSMLP == nullptr) {
            /*
               前 5 个参数与 TB 那一支的构造**逐字一致** (hidden 64 / gamma 0.99 /
               lr 0.001 / c_puct 1.414), 后三个是 PPOMCTSAgent 的默认值
               (expertHidden 64 / moeAuxCoef 0.1 / withGrad true) —— 它们必须写出来,
               因为骨干是第 9 个参数。参数含 MoE 结构, 写错会让 save/load 的结构指纹
               对不上 (每一轮都载入失败)。
            */
            m_sfPPOMCTSMLP = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                              true, RL::PPO::Backbone::MlpExperts);
            applyPpoBalance(*m_sfPPOMCTSMLP);
        }
        m_sfPPOMCTSMLP->loadModel(s_weightPaths[AGENT_PPOMCTS_MLP]);
        logLoad("PPO+MCTS (MLP专家)");
    }
    if (s_weightPaths.count(AGENT_DQNAB)) {
        emit busyMessage(QStringLiteral("正在载入 DQN+AB 权重… (主干 146 MB + 两个头)"));
        if (m_sfDQNAB == nullptr) {
            m_sfDQNAB = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                             DQNABAgent::Backbone::SparseMoeTb);
        }
        logLoad("DQN+AB 建网(主干 37.5 M 参数)");
        /* 探测串是 `<prefix>_trunk`, 去掉后缀得到 loadModel 要的前缀 */
        std::string prefix = s_weightPaths[AGENT_DQNAB];
        const std::string suffix = "_trunk";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfDQNAB->loadModel(prefix);
        logLoad("DQN+AB 读权重(主干 + V 头 + A 头)");
    }
    if (s_weightPaths.count(AGENT_DQNMCTS_MOE)) {
        /*
           [2026-09 dev-dqnmcts-moetb] DQN+MCTS (MoE+TB 专家) 的预加载: 与 DQN+AB 那块对称。
           ⚠ 这里"建网"本身就不便宜 (稀疏 MoE + 4 个 TB 专家 = ~29 M 参数的
           "分配 + 随机初始化"), 与 TB 那一组 SAC 一样: 单独记一条耗时日志,
           启动慢的话能一眼看出是这一步, 而不是"读文件慢"。DQN+AB 也是同样地
           在启动时**急切**建网 (用户口径: 启动时加载所有模型), 所以这里照做。
           `denseMoe` 走默认值 false (就是稀疏路由那一支; 等参数对照组只在 bench 里开)。
        */
        emit busyMessage(QStringLiteral("正在载入 DQN+MCTS (稀疏MoE+TB专家) 权重…"));
        if (m_sfDQNMCTSMOE == nullptr) {
            m_sfDQNMCTSMOE = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                                  1.0f, 1.5f);
        }
        logLoad("DQN+MCTS-MoE 建网(稀疏 MoE + 4 个 TB 专家)");
        /* 探测串是 `<prefix>_trunk`, 去掉后缀得到 loadModel 要的前缀 */
        std::string prefix = s_weightPaths[AGENT_DQNMCTS_MOE];
        const std::string suffix = "_trunk";
        if (prefix.size() > suffix.size()
            && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
            prefix.erase(prefix.size() - suffix.size());
        }
        m_sfDQNMCTSMOE->loadModel(prefix);
        logLoad("DQN+MCTS-MoE 读权重(主干 + Q 头)");
    }

    /* ---- 4. 启动时给每个**已加载的模型**跑一遍自检 (2026-09) ---- */
    /*
       用户的要求是"所有模型都配上自检方法", 而自检的价值不只是"界面上有个面板":
       它的契约是**只读、可重复调用、不动棋盘** (见 aiagent.h), 所以启动时把每个已
       加载的模型都跑一遍非常便宜 (每个几十微秒, 大头是 alpha-beta 那一次 1 层搜索),
       而且能立刻暴露"权重根本没载进来 / 结构与文件对不上"这类**静默失效**。
       这里每个模型只打一行 (表示层摘要), 完整报告在界面右侧的"模型自检"面板
       (以及那个"全部模型自检"按钮给的横向对照)。
    */
    {
        QStringList checked;
        for (AgentType t : kWeightAgents) {
            if (!hasAgentInstance(t)) {
                continue;
            }
            const QString firstLine =
                QString::fromStdString(getAgentSelfCheck(t)).section(QLatin1Char('\n'), 0, 0);
            qInfo().noquote() << QStringLiteral("[selfcheck] %1: %2")
                                     .arg(agentDisplayName(t), firstLine);
            checked << agentDisplayName(t);
        }
        if (checked.isEmpty()) {
            qInfo().noquote()
                << QStringLiteral("[selfcheck] 没有已加载的模型 (weights/ 下没有权重文件; "
                                  "纯搜索 agent 的自检在界面上看)");
        }
    }

    /* ---- 5. 启动后台训练 & 通知主线程加载完成 ---- */
    emit busyMessage(QStringLiteral("初始化完成"));
    emit busyFinished();
    m_startupComplete = true;

    /* 启动后台训练线程 (神经网络agent持续自我对弈提升棋力) */
    startBackgroundTraining();

    QMetaObject::invokeMethod(this, [this]() {
        emit startupComplete();
    }, Qt::QueuedConnection);
}

/*
 * weightLoadSummary - 启动时"有没有模型"的一句话 (界面用; 见头文件里的说明)
 *
 * 为什么要有它: 用户报障"点击开局模型未载入", 查下来是"`weights/` 是 gitignore 的运行期
 * 产物, 没存过权重时它就是空的" —— 代码没错, 但这件事**只在日志/自检面板里**说, 而用户
 * 按"开局"时看不到那两处。于是"没有模型"表现为"AI 下得像随机", 没有任何人能看见的提示。
 * 这一行就挂在**对局列表**(开局后必看的那个面板) 的最上面。
 *
 * 判据用的是 `s_weightPaths` (启动扫描的命中结果), 所以它只有在扫描之后才有意义 ——
 * 调用点是 startupComplete 之后 (见 MainWindow 的连接)。
 */
std::string ChessBoard::weightLoadSummary() const
{
    QStringList have;
    QStringList missing;
    for (AgentType t : kWeightAgents) {
        if (s_weightPaths.count(t) != 0) {
            have << agentDisplayName(t);
        } else {
            missing << agentDisplayName(t);
        }
    }

    /*
       ⚠ 这一行会被放进 QListWidget 的**单个条目**里 —— 所以不许有换行
         (换行会被压平成一团, 比不写还难读)。细节在 weightLoadHint() 里 (tooltip)。
    */
    if (have.isEmpty()) {
        return QStringLiteral("⚠ 未载入任何模型权重 (%1 组全缺) —— 权重是运行期产物, "
                              "跑一场对弈或正常关窗后才会写到 weights/ (悬停看说明)")
                   .arg(missing.size()).toStdString();
    }
    if (!missing.isEmpty()) {
        return QStringLiteral("启动时载入 %1/%2 组权重; 缺失: %3 (这些 agent 从随机初始化开始)")
                   .arg(have.size()).arg(have.size() + missing.size())
                   .arg(missing.join(QStringLiteral(", "))).toStdString();
    }
    return QStringLiteral("启动时载入 %1/%2 组权重 (全部命中)")
               .arg(have.size()).arg(have.size()).toStdString();
}

std::string ChessBoard::weightLoadHint() const
{
    QStringList have;
    QStringList missing;
    for (AgentType t : kWeightAgents) {
        if (s_weightPaths.count(t) != 0) {
            have << agentDisplayName(t);
        } else {
            missing << agentDisplayName(t);
        }
    }

    QString s;
    if (have.isEmpty()) {
        s += QStringLiteral("⚠ 没有载入任何模型权重\n");
        s += QStringLiteral("weights/ 下没有权重文件 (%1 组全部缺失): %2\n")
                 .arg(missing.size()).arg(missing.join(QStringLiteral(", ")));
        s += QStringLiteral("此时每个 agent 都从**随机初始化**开始 —— 棋力接近随机。\n"
                            "这不是模型坏了, 而是**还没有权重**。\n");
        s += QStringLiteral("\n权重是**运行期产物** (weights/ 被 .gitignore 忽略, "
                            "所以新克隆的仓库里本来就没有):\n"
                            "  · 跑一场【对弈】结束后会自动按标准名字写出;\n"
                            "  · 或者正常关窗 (shutdownSave) 时写出。\n");
        s += QStringLiteral("\n每个文件在不在, 看右侧\"模型自检\"面板第一段 "
                            "(它印的是**实际文件名**, 不是前缀)。");
    } else {
        s += QStringLiteral("启动时载入 %1/%2 组权重\n")
                 .arg(have.size()).arg(have.size() + missing.size());
        if (!missing.isEmpty()) {
            s += QStringLiteral("缺失: %1\n(这些 agent 从随机初始化开始)")
                     .arg(missing.join(QStringLiteral(", ")));
        }
    }
    return s.toStdString();
}

ChessBoard::ChessBoard(QWidget *parent) :
    QWidget(parent),
    selectID(-1),
    color(Stone::COLOR_RED),
    state(STATE_IDEL),
    m_currentGameId(-1),
    m_moveCount(0),
    m_dbEnabled(false),
    m_agentType(AGENT_ALPHABETA)
{
    connect(this, &ChessBoard::sendResult,
            this, &ChessBoard::checkGameOver, Qt::QueuedConnection);
    initThinkVisuals();
    processThread = std::thread(&ChessBoard::process, this);
}

/*
 * initThinkVisuals - 建立"思考中"状态条的动画与信号连线
 *
 * 所有跨线程信号都用 AutoConnection: emit 发生在工作线程/AI 线程, 而本对象
 * (以及主窗口) 住在 GUI 线程, Qt 在 emit 时判定接收者线程不同 -> 自动排队,
 * 于是下面这些槽都在 GUI 线程执行, 可以安全地读写控件状态。
 */
void ChessBoard::initThinkVisuals()
{
    m_animTimer = new QTimer(this);
    m_animTimer->setInterval(40);            /* 25 fps, 只重绘顶部一小条 */
    connect(m_animTimer, &QTimer::timeout, this, [this]() {
        ++m_animPhase;
        if (m_animPhase > 100000) {
            m_animPhase = 0;
        }
        update(thinkingOverlayRect());
    });

    m_busyClickTimer = new QTimer(this);
    m_busyClickTimer->setSingleShot(true);
    m_busyClickTimer->setInterval(2500);
    connect(m_busyClickTimer, &QTimer::timeout, this, [this]() {
        if (!m_busyClickSeen) {
            return;
        }
        m_busyClickSeen = false;
        update(thinkingOverlayRect());
    });

    connect(this, &ChessBoard::aiThinkingStarted, this,
            [this](const QString &, int) {
                m_thinkClock.start();
                m_thinkStage.clear();
                m_busyClickSeen = false;
                if (m_animTimer != nullptr && !m_animTimer->isActive()) {
                    m_animPhase = 0;
                    m_animTimer->start();
                }
                update(thinkingOverlayRect());
            });
    connect(this, &ChessBoard::aiThinkingStage, this, [this](const QString &stage) {
        m_thinkStage = stage;
        update(thinkingOverlayRect());
    });
    connect(this, &ChessBoard::aiThinkingStopped, this, [this]() {
        if (m_animTimer != nullptr) {
            m_animTimer->stop();
        }
        m_busyClickTimer->stop();
        m_busyClickSeen = false;
        m_thinkStage.clear();
        update(thinkingOverlayRect());
    });
}

ChessBoard::~ChessBoard()
{
    /* 先停止后台训练 */
    stopBackgroundTraining();

    {
        QMutexLocker locker(&mutex);
        /*
           m_processStop 是 process() **唯一**的退出条件 (见 chessboard.h 的成员注释):
           state = TERMINATE 只表示"这一局完了", 线程会留在等待里继续服务下一局 ——
           所以关窗时必须显式置位, 否则 join() 会一直等下去。
        */
        m_processStop = true;
        state = STATE_TERMINATE;
        condit.wakeAll();
    }
    if (processThread.joinable()) {
        processThread.join();
    }
}

/* 获取石头中心的点坐标 */
QPoint ChessBoard::getStoneCenter(int x, int y)
{
    return QPoint(offsetX + y * gridSize, offsetY + x * gridSize);
}

/* 获取石头位置 (棋盘坐标) */
Pos ChessBoard::getStonePos(const QPoint &point)
{
    /*
       用 floor 语义做舍入。原来写的是 (p - offset + grid/2) / grid, C++ 的整数
       除法向零截断, 于是 point.y() ∈ [0,19] 时 (py-20)/60 得到 0 而不是 -1 ——
       棋盘上/左各 20px 的空白被误判成第 0 行/第 0 列; 相与地, x<0 / y<0 这两个
       守卫永远不会成立。
    */
    const int x = static_cast<int>(std::floor((point.y() - offsetY + gridSize / 2.0) / gridSize));
    const int y = static_cast<int>(std::floor((point.x() - offsetX + gridSize / 2.0) / gridSize));
    if (x < 0 || x > 9 || y < 0 || y > 8) {
        return Pos(-1, -1);
    }
    return Pos(x, y);
}

QRect ChessBoard::getRect(QPoint &center)
{
    QPoint topLeft(center.x() - stoneRadius, center.y() - stoneRadius);
    QSize size(stoneRadius * 2, stoneRadius * 2);
    QRect rect(topLeft, size);
    return rect;
}

Stone *ChessBoard::selectStone(const QPoint &point)
{
    Pos pos = getStonePos(point);
    if (pos.x < 0 || pos.y < 0) {
        return nullptr;
    }
    return chess.m_map[pos];
}

bool ChessBoard::moveStone(const QPoint &point)
{
    Pos pos = getStonePos(point);
    if (pos.x < 0 || pos.y < 0) {
        return false;
    }
    Stone *stone = chess.m_map[pos];

    /* 点击己方棋子: 选中它 (不落子) */
    if (stone != nullptr && stone->color == color) {
        selectID = stone->id;
        return false;
    }
    if (selectID == -1) {
        return false;
    }

    Stone *selected = chess.m_map.get(selectID);
    if (selected == nullptr || selected->alive == false) {
        selectID = -1;
        return false;
    }

    /*
       构造候选走法, 由 Chess::isLegalMove 统一校验: 走法形状 + 走后自己是否被将
       (含"不应将"与两将照面)。原来这里只看 selected->tryMoveTo(pos), 于是玩家
       可以自杀、可以在被将时走别的子、也可以主动走出照面。

       ---- 自由走子 (调试开关, 2026-09 用户要求) ----
       用户要"能故意输给 AI"(送将/不应将), 而上面那条校验恰好挡住了它 ——
       打开 m_freeMove 时**只跳过"走后是否被将"这一半**, 走法形状仍然要合法
       (马走日、象走田…), 于是可以自杀/不应将。默认关闭。
       (为什么不整个跳过: 那样连形状都不判, 会走出"炮直飞"这种局面,
        棋盘状态可能自相矛盾 —— 调试开关不该制造比问题更难查的状态。)
    */
    Step step;
    step.id = selectID;
    step.pos = selected->pos;
    step.nextId = (stone != nullptr) ? stone->id : Stone::ID_NONE;
    step.nextPos = pos;
    step.reward = 0;
    step.valid = true;

    /*
       形状校验: 直接用棋子自己的走法规则 (tryMoveTo 是"形状 + 目标占用"那一层,
       与 isLegalMoveInternal 里的形状判定同源)。自由走子时不再要求"走后不被将"。
    */
    /*
       形状/规则校验。
       ---- 自由走子 (调试开关, 2026-09 用户口径) ----
       用户要求"被将军时我希望能移动所有棋子"。在它之前, 这个开关只跳过了
       "走后是否被将"这一半, 形状仍按棋子走 (马走日、兵只能前进或过河横走) ——
       于是在被将军时挪别的子仍然被拒, 用户看到的仍然是"走不动"。
       现在口径是: 打开后**任意棋子可以走到任意一格**, 所以这里整个跳过校验。
       落地由 `Chess::freeMove` 传给 Stone::moveTo (跳过 tryMoveTo), 保证吃子结算、
       m_map、alive 仍走同一条路 (见 stone.h 的 moveTo 说明)。
    */
    if (m_freeMove) {
        chess.freeMove = true;
        dbgLog(QStringLiteral("自由走子: 跳过形状校验, 从 (%1,%2) 到 (%3,%4), 类型=%5")
                   .arg(step.pos.x).arg(step.pos.y)
                   .arg(step.nextPos.x).arg(step.nextPos.y)
                   .arg((int)selected->type));
    } else {
        chess.freeMove = false;
        const bool shapeOk = chess.isLegalMove(color, &step);
        dbgLog(QStringLiteral("规则校验: isLegalMove %1 (从 (%2,%3) 到 (%4,%5), 类型=%6)")
                   .arg(shapeOk ? QStringLiteral("通过") : QStringLiteral("不通过"))
                   .arg(step.pos.x).arg(step.pos.y)
                   .arg(step.nextPos.x).arg(step.nextPos.y)
                   .arg((int)selected->type));
        if (shapeOk == false) {
            /*
               [诊断] 被拒时把"这个子**实际**能走到哪"列出来 —— 用户报"红兵过河后不能
               左右走"的时候, 这一行直接给出答案: 横走的目标在不在列表里。
               (列表用与规则同源的 getPossibleSteps + tryMoveTo, 不是另写一套判定。)
            */
            std::vector<Step *> cand;
            selected->getPossibleSteps(cand);
            QStringList targets;
            for (Step *c : cand) {
                if (c != nullptr && selected->tryMoveTo(c->nextPos)) {
                    targets << QStringLiteral("(%1,%2)").arg(c->nextPos.x).arg(c->nextPos.y);
                }
            }
            Steps::instance().put(cand);
            dbgLog(QStringLiteral("  该子的形状可达目标: %1")
                       .arg(targets.isEmpty() ? QStringLiteral("(空)") : targets.join(' ')));
            return false;
        }
    }

    double totalReward = 0;
    chess.moveForward(&step, totalReward);
    selectID = -1;
    color = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    chess.sideToMove = color;
    return true;
}

void ChessBoard::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    /* 绘制棋盘背景 */
    painter.setBrush(QColor(222, 184, 135));
    painter.drawRect(rect());

    /* 绘制网格线 */
    QPen pen(Qt::black, 2);
    painter.setPen(pen);
    for (int i = 0; i < 10; i++) {
        painter.drawLine(offsetX, offsetY + i * gridSize, offsetX + 8 * gridSize, offsetY + i * gridSize);
    }
    for (int i = 0; i < 9; i++) {
        painter.drawLine(offsetX + i * gridSize, offsetY, offsetX + i * gridSize, offsetY + 4 * gridSize);
        painter.drawLine(offsetX + i * gridSize, offsetY + 5 * gridSize, offsetX + i * gridSize, offsetY + 9 * gridSize);
    }

    /*
       棋子与选中高亮都在锁内画: AI 工作线程 (process) 和 Agent 对弈线程都在改
       chess, 而绘制发生在 GUI 线程。锁的临界区很短 (只读一遍棋子), 不与搜索重叠
       —— 两个写者都只在"落子"那一瞬间持锁, 思考期间不持锁。
    */
    {
        QMutexLocker locker(&mutex);

        /* 绘制棋子 */
        for (int i = 0; i < 32; i++) {
            Stone *stone = chess.m_children[i];
            if (stone == nullptr || stone->alive == false) {
                continue;
            }
            drawStone(painter, stone);
        }

        /* 绘制选中高亮 (灰色边框) */
        if (selectID != -1) {
            Stone *selected = chess.m_map.get(selectID);
            if (selected != nullptr) {
                QPoint center = getStoneCenter(selected->pos.x, selected->pos.y);
                QRect highlightRect = getRect(center).adjusted(-3, -3, 3, 3);

                /* 1. 外发光光晕 (半透明灰色圆环) */
                QRadialGradient glow(center, stoneRadius + 8);
                glow.setColorAt(0.0, QColor(160, 160, 160, 120));
                glow.setColorAt(0.7, QColor(160, 160, 160, 60));
                glow.setColorAt(1.0, QColor(160, 160, 160, 0));
                painter.setBrush(glow);
                painter.setPen(Qt::NoPen);
                painter.drawEllipse(center, stoneRadius + 8, stoneRadius + 8);

                /* 2. 高亮边框 (灰色粗线) */
                painter.setBrush(Qt::NoBrush);
                painter.setPen(QPen(QColor(128, 128, 128), 4));
                painter.drawEllipse(highlightRect);

                /* 3. 内发光 (半透明灰色填充) */
                painter.setBrush(QColor(160, 160, 160, 30));
                painter.setPen(QPen(QColor(128, 128, 128), 2));
                painter.drawEllipse(getRect(center));
            }
        }
    }

    /* "AI 正在思考"的状态条画在最上层 */
    drawThinkingOverlay(painter);
}

/*
 * thinkingOverlayRect - 状态条的位置
 *
 * 画在棋盘**上方的空白带**里: 最上面一排棋子的圆心在 offsetY=50, 半径 24, 所以
 * y < 26 这条横带是完全空着的。放在这里既不遮挡任何棋子, 又正好在玩家盯着的
 * 棋盘内部, 不需要去右边的控制栏找。
 */
QRect ChessBoard::thinkingOverlayRect() const
{
    constexpr int kBoxH = 22;
    int boxW = width() - 16;
    if (boxW > 470) {
        boxW = 470;
    }
    if (boxW < 160) {
        boxW = qMax(100, width() - 4);
    }
    return QRect((width() - boxW) / 2, 2, boxW, kBoxH);
}

void ChessBoard::drawThinkingOverlay(QPainter &painter)
{
    const bool selfPlay = m_selfPlaying.load();
    if (state != STATE_THINKING && !selfPlay) {
        return;
    }

    const QRect box = thinkingOverlayRect();
    /* 40ms 一帧 * 40 帧 = 1.6 秒转一圈 */
    const qreal phase = (m_animPhase % 40) / 40.0;
    const qreal wave = 0.5 + 0.5 * std::sin(phase * 2.0 * kPi);

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);

    /* 深色药丸底 (棋盘是米黄, 深底浅字最醒目) */
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(28, 38, 54, 230));
    painter.drawRoundedRect(box, box.height() / 2.0, box.height() / 2.0);

    /* 左端: 一圈旋转的粒子 (呼吸 + 公转 + 彗尾), 与 ThinkingIndicator 同款视觉 */
    const QPointF c(box.left() + 13.0, box.center().y() + 0.5);
    const qreal ringR = 6.4 + 0.9 * wave;
    constexpr int kDots = 8;
    for (int i = 0; i < kDots; ++i) {
        const qreal k = 1.0 - i / static_cast<qreal>(kDots);   /* 彗头 = 1 */
        const qreal ang = phase * 2.0 * kPi - i * (2.0 * kPi / kDots);
        QColor dot(96, 180, 255);
        dot.setAlphaF(0.18 + 0.82 * k * k);
        painter.setBrush(dot);
        const qreal r = 1.0 + 1.5 * k;
        painter.drawEllipse(QPointF(c.x() + ringR * std::cos(ang),
                                    c.y() + ringR * std::sin(ang)), r, r);
    }

    /* 文字: [agent] 正在思考 3.24s · ① 探索环境 + 预训练 (≤64 步) */
    const long long ms = m_thinkClock.isValid() ? m_thinkClock.elapsed() : 0;
    QString text = selfPlay ? QStringLiteral("AI 对弈中") : QStringLiteral("AI 正在思考");
    text += QStringLiteral("  ") + shortElapsed(ms);
    if (!m_thinkStage.isEmpty()) {
        text += QStringLiteral("   ·   ") + m_thinkStage;
    }

    QFont f = painter.font();
    f.setPointSize(8);
    f.setBold(true);
    painter.setFont(f);

    const int textLeft = box.left() + 25;
    const int textRight = box.right() - 8;
    const QFontMetrics fm(f);
    if (m_busyClickSeen) {
        const QString hint = QStringLiteral("请稍候, 现在还不能走子");
        const int hintW = fm.horizontalAdvance(hint) + 10;
        painter.setPen(QColor(255, 196, 92));
        painter.drawText(QRect(textRight - hintW, box.top(), hintW, box.height()),
                         Qt::AlignVCenter | Qt::AlignRight, hint);
        painter.setPen(QColor(236, 242, 250));
        painter.drawText(QRect(textLeft, box.top(), textRight - textLeft - hintW, box.height()),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         fm.elidedText(text, Qt::ElideRight, textRight - textLeft - hintW));
    } else {
        painter.setPen(QColor(236, 242, 250));
        painter.drawText(QRect(textLeft, box.top(), textRight - textLeft, box.height()),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         fm.elidedText(text, Qt::ElideRight, textRight - textLeft));
    }

    painter.restore();
}

void ChessBoard::drawStone(QPainter &painter, const Stone *stone)
{
    QPoint center = getStoneCenter(stone->pos.x, stone->pos.y);
    QRect rect = getRect(center);

    /* 绘制棋子背景 */
    painter.setBrush(QColor(255, 206, 158));
    painter.setPen(QPen(Qt::black, 2));
    painter.drawEllipse(rect);

    /* 绘制棋子文字 */
    painter.setFont(QFont("SimSun", 18, QFont::Bold));
    if (stone->color == Stone::COLOR_RED) {
        painter.setPen(Qt::red);
    } else {
        painter.setPen(Qt::black);
    }
    painter.drawText(rect, Qt::AlignCenter, QString::fromStdString(stone->name));
}
void ChessBoard::mousePressEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton) return;

    /*
       AI 思考期间点棋盘: 拒绝落子是对的 (轮到 AI 走), 但必须给出反馈。
       原来的实现只是静默 return, 玩家看到的是"棋子没动, 我也点不动", 分不清
       是"还在算"还是"已经卡死" —— 加入"走子前先探索+预训练"之后单步思考涨到
       秒级, 这个问题就很突出了。这里在状态条上闪一句"请稍候"。
    */
    if (state == STATE_THINKING || m_selfPlaying.load()) {
        /*
           [诊断 2026-09] 用户报"红兵过河后不能左右走": 规则本身允许 (见 Bing::tryMoveTo),
           所以要先排除"点击根本没被处理"这一类 —— 状态不对时点击是被**静默丢弃**的,
           玩家看到的"走不动"和"规则不许走"完全一样。这里把真相打出来。
        */
        dbgLog(QStringLiteral("点击被丢弃: state=%1 (0=IDLE/1=THINKING/2=TERMINATE), "
                              "m_selfPlaying=%2, 红方=(%3,%4)")
                   .arg((int)state.load())
                   .arg((int)m_selfPlaying.load())
                   .arg(getStonePos(event->pos()).x)
                   .arg(getStonePos(event->pos()).y));
        if (state == STATE_THINKING && !m_busyClickSeen) {
            m_busyClickSeen = true;
            m_busyClickTimer->start();
            update(thinkingOverlayRect());
        }
        return;
    }
    if (state != STATE_IDEL) {
        dbgLog(QStringLiteral("点击被丢弃: state=%1 (既不是 IDLE 也不是 THINKING)")
                   .arg((int)state.load()));
        return;
    }
    /* [2026-10 移除] 这里原有 `if (isReplayMode()) { 点击被丢弃: 回放模式 }` ——
       回放功能整体删除后这个守卫不可能成立 (见 mainwindow.h 的说明) */

    if (selectID == -1) {
        /* 选择己方棋子 */
        Stone *stone = selectStone(event->pos());
        if (stone == nullptr) return;
        if (stone->color != color) return; /* 不能选对方棋子 */
        selectID = stone->id;
        dbgLog(QStringLiteral("选中棋子 id=%1 类型=%2 位置=(%3,%4)")
                   .arg(selectID).arg((int)stone->type)
                   .arg(stone->pos.x).arg(stone->pos.y));
        update();
        return;
    }

    /* 已有选中棋子 -> 尝试移动 */
    bool moved = moveStone(event->pos());
    if (!moved) {
        /*
           [诊断] 走不动时把"为什么不许"打出来: 形状/被将/目标占用/自由走子开关。
           这条日志就是为了回答"到底是规则不许、还是开关没生效、还是没选中"。
        */
        const Pos from = getStonePos(event->pos());
        dbgLog(QStringLiteral("走子被拒: selectID=%1 目标=(%2,%3) 自由走子=%4")
                   .arg(selectID).arg(from.x).arg(from.y)
                   .arg((int)m_freeMove));
        /* 移动失败 (含非法走法): 可能是点到了己方另一棋子, 重新选 */
        Stone *stone = selectStone(event->pos());
        if (stone != nullptr && stone->color == color) {
            selectID = stone->id;
        } else {
            selectID = -1;
        }
        update();
        return;
    }

    /* 玩家走棋成功 → 判定是否将杀 / 困毙 / 和棋 */
    int result = chess.getResult(chess.sideToMove);
    /*
       ---- 对弈期间: 权重不落盘 (2026-10 用户口径) ----
       "人机这一局"从**玩家落第一子**起算 (不按"按了开局"算: 启动后直接走子也是
       一局棋, 而那时没人按过开局 —— 按"按过开局"算会漏掉这一整类用法)。
       解除点是终局 (notifyHumanGameEnd, 三条终局路径都经过它) 或重开一局 (reset)。
       幂等: 每走一手调一次也没关系 (状态没变时 updatePlayWeightDivert 直接返回)。
    */
    setHumanGameInProgress(true);
    if (result != Chess::RESULT_ONGOING) {
        state = STATE_TERMINATE;
        /*
           [P1 人机终局通道] 先把结果交给学习器 (AI 执黑, 这一手是**人**把它将死的 ——
           而它自己的 learnFromSearchStep 只在"它自己那一手结束对局"时才写终局, 于是
           这一局的 −1 原来谁都收不到)。必须在 state/update 之前调用: agent 要按**当前**
           棋盘 (终局局面) 算终局奖励, 而 reset() 会把它推回开局。
        */
        notifyHumanGameEnd(result);
        update();
        emit sendResult(result);
        return;
    }

    /* 切换为AI思考 */
    selectID = -1;
    state = STATE_THINKING;
    update();
    /*
       wakeAll() 必须在持有同一个 mutex 时调用。原来这里是无锁调用, 存在经典的
       丢唤醒窗口: worker 已判定 state != STATE_THINKING (此时还是 IDLE) 但还没
       进入 wait(), GUI 此刻把 state 改成 THINKING 并 wakeAll() —— 由于 GUI 不
       持锁, 这一步可以被插进 worker 的判断与等待之间, worker 随后长睡, 棋盘
       再也不会有 AI 落子。
    */
    {
        QMutexLocker locker(&mutex);
        condit.wakeAll();
    }
}

void ChessBoard::process()
{
    /*
       回到"等待玩家走棋"并宣告思考结束。
       顺序很重要: 必须**先把 state 落成 IDLE, 再**发 aiThinkingStopped ——
       状态条的可见性看的就是 state, 反过来的话 GUI 处理队列消息时可能又多画一帧。
    */
    auto backToIdle = [this]() {
        {
            QMutexLocker locker(&mutex);
            color = Stone::COLOR_RED;
            state = STATE_IDEL;
            selectID = -1;
            condit.wakeAll();
        }
        emit aiThinkingStopped();
        QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
    };

    /*
       ================================================================
        ---- 主循环为什么是 for (;;) 而不是 while (state != STATE_TERMINATE) ----
        ================================================================
        用户报障 (2026-09): "人机对弈时黑方胜利后, 沙漏显示 agent 未加载, 选择黑方
        对弈 agent 失效, 红方下第一个棋后黑方无限等待"。

        原来的条件 `while (state != STATE_TERMINATE)` 与**终局**用的是同一个 state:
        终局分支里 `state = STATE_TERMINATE; ... continue;` 之后循环条件立刻为假 ⇒
        线程函数 return ⇒ **AI 自己结束一局之后, 这个应手线程就永久消失了**。
        于是:
          * 按"开局" (reset()) 只把 state 改回 IDEL —— 没有任何人重新起线程;
          * 红方再落子 -> state=THINKING + wakeAll, 但**没有等待者** ⇒ "黑方无限等待";
          * 换对战 agent 也没有人去用它应手 ⇒ "选择黑方对弈 agent 失效";
          * 沙漏从此收不到 aiThinkingStarted, 而 resetToIdle() 又把 agent 名清空了
            ⇒ 显示 "(未选择 agent)" ⇒ 用户读成"agent 未加载"。

        为什么不是"在 reset() 里重新起线程": reset() 与工作线程之间**无法**安全地
        重新 join/新建 —— reset() 可能在"工作线程刚判定完终局、还没跑出循环"的那一瞬
        发生, 那样 join() 会一直等一个永远不会退出的循环 (它看到 state 又是 IDLE 就继续
        转)。把"线程该不该活着"与"这一局结没结束"解耦, 才是这件事的正解。

        所以: 终局只把 state 停在 STATE_TERMINATE (棋盘继续拒收点击, 这仍然是对的),
        **线程留在等待里**; 下一次 reset() 把它叫醒, 它接着服务新的一局。
        唯一退出点是 m_processStop (析构里置位 + wakeAll, 见成员注释)。
    */
    for (;;) {
        {
            QMutexLocker locker(&mutex);
            /*
               这里必须是 while 而不是 if, 而且醒来后**必须重新检查 state**。

               原来写的是 `if (state != STATE_THINKING) condit.wait(&mutex);`, 醒来就
               直接往下走去搜索。条件变量本来就会虚假唤醒, 而 reset() 和 matchAgents()
               都会在 state 仍是 IDLE 的时候 wakeAll() (它们只是想叫醒"正在思考"的
               worker, 让它作废在飞的那一步) —— 于是空闲的 worker 被叫醒后照跑
               aiThink(), 一边搜索一边往 env 的 history 里 push_back。
               如果这时另一个线程 (Agent 对弈线程) 也在用同一个 env, 两边同时改
               env.history 就是 double-free / 堆损坏 (ASan 实测抓到的就是这个)。

               现在多一个退出条件: m_processStop (析构里置位)。state == TERMINATE
               不再退出循环 —— 那只是"这一局完了, 先等着"。
            */
            while (state != STATE_THINKING && !m_processStop.load()) {
                condit.wait(&mutex);
            }
            if (m_processStop.load()) {
                break;
            }
        }

        /* 计时开始 */
        auto t0 = std::chrono::steady_clock::now();
        /* 记下代数, 用来识别"思考途中被按了开局" */
        const unsigned gen = m_thinkGeneration.load();

        emit aiThinkingStarted(agentDisplayName(m_agentType), m_preTrainSteps.load());

        /* AI(黑方) 决策 */
        dbgLog(QStringLiteral("process: 准备调用 aiThink(黑方)"));
        const double tThinkDbg = dbgNowMs();
        Step step = aiThink(Stone::COLOR_BLACK);
        dbgWait(QStringLiteral("process: aiThink 返回 (valid=%1)")
                    .arg((int)step.valid), dbgNowMs() - tThinkDbg);

        /* 计时结束 */
        auto t1 = std::chrono::steady_clock::now();
        long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        /* 思考期间玩家按了"开局" -> 这一步连同它的计时一起作废 */
        if (gen != m_thinkGeneration.load()) {
            backToIdle();
            continue;
        }

        /* 检查AI是否有合法走法 (将杀 / 困毙) */
        if (!step.valid) {
            /*
               先分清"真的没棋可走"和"agent 返回了一个无效走法"。后者不是将杀:
               以前直接把 !step.valid 当成"AI 被将死", 于是一个抽风的 agent 会让对局
               在**一步都没落子**的情况下判红方胜 —— 界面上就是"棋子没动, 我却赢了",
               而且从此不能走棋。这里用第一个合法走法兜底。
            */
            std::vector<Step *> legal;
            {
                QMutexLocker locker(&mutex);
                chess.sample(Stone::COLOR_BLACK, legal);
                if (!legal.empty()) {
                    std::fprintf(stderr,
                                 "[aiThink] agent 返回无效走法 (有 %zu 个合法走法), "
                                 "已用第一个合法走法兜底\n", legal.size());
                    step = *legal[0];
                }
            }
            Steps::instance().put(legal);
        }

        if (!step.valid) {
            /* 确实没有合法走法 -> 红方胜 */
            state = STATE_TERMINATE;
            emit aiThinkingStopped();
            /* [P1 人机终局通道] AI 被将死/困毙在这一手之前就发生了 (它没棋可走) ——
               终局反馈要挂在它**上一步**的决策样本上, 见 notifyHumanGameEnd。 */
            notifyHumanGameEnd(Chess::RESULT_RED_WIN);
            emit sendResult(Chess::RESULT_RED_WIN);
            QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
            continue;
        }

        emitStage(QStringLiteral("③ 落子"));

        /* 通过引擎执行走法 (与 GUI 线程的绘制互斥) */
        int result = Chess::RESULT_ONGOING;
        bool discarded = false;
        {
            QMutexLocker locker(&mutex);
            /* 在锁内再确认一次代数: reset() 也是在这个锁内 +1 的, 两者不会交错 */
            if (gen != m_thinkGeneration.load()) {
                discarded = true;
            } else {
                double totalReward = 0;
                chess.moveForward(&step, totalReward);
                chess.sideToMove = Stone::COLOR_RED;
                result = chess.getResult(chess.sideToMove);
            }
        }
        if (discarded) {
            backToIdle();
            continue;
        }

        if (result != Chess::RESULT_ONGOING) {
            emit aiThinkFinished(elapsedMs);
            state = STATE_TERMINATE;
            emit aiThinkingStopped();
            /* [P1 人机终局通道] 这一手是 AI 自己走完的 —— 它的 learnFromSearchStep 已经
               写过终局 (幂等, 重复通知不会写第二遍/不会多更新一次)。仍然统一在这里调:
               "凡终局必通知"只有一条规则, 才不会将来漏掉某一支 (例如和棋那一路)。 */
            notifyHumanGameEnd(result);
            emit sendResult(result);
            QMetaObject::invokeMethod(this, [this](){ update(); }, Qt::QueuedConnection);
            continue;
        }

        /* 通知UI更新 (在主线程更新QLabel等) */
        emit aiThinkFinished(elapsedMs);

        /* 回到等待玩家状态 */
        backToIdle();
    }
}

void ChessBoard::reset()
{
    /*
       [2026-10 移除] 这里原来是"reset() 必须同时退出回放模式"的修复 —— 原实现只重置
       棋盘、不清 m_replayGameId, 而 mousePressEvent 开头有 `if (isReplayMode()) return;`,
       于是选过一次历史对局之后按"开局"也点不动 (replayModeExited 声明了、连接了、
       却从来没被 emit 过)。回放功能整体删除后这个失效模式连同修复一起消失:
       现在没有回放模式, 也就不存在"退出"这件事。
    */
    {
        QMutexLocker locker(&mutex);
        /*
            +1 之后, 正在思考的那一步会在 process() 里发现"代数变了"而主动作废。
           没有这一条时, reset() 只把 state 改回 IDEL 就返回, 而工作线程还在算,
           算完照样落子 —— 落到**重置后的新棋盘**上, 表现为"我刚开了新局, 对方
           却已经走了一步"。
        */
        ++m_thinkGeneration;
        chess.reset();
        selectID = -1;
        color = Stone::COLOR_RED;
        state = STATE_IDEL;
        condit.wakeAll();
    }
    /*
       ---- "开局"= 上一局到此为止 (2026-10 对弈期间权重不落盘) ----
       解除而不是武装, 理由有两条:
         * **"人机这一局"的起点是"玩家落第一子"** (见 mousePressEvent): 按下开局之后
           可能很久都不落子, 那段时间算不算"对弈期间"没有客观答案, 而按"第一子"算
           既明确又已经是"棋在下"的状态;
         * **reset() 在界面之外被大量当成"摆回开局"的管道** (test/bench/probe 十几个
           main 都调它)。在那里武装改道 = 那些工具**后续保存的权重全部改道到内存**,
           盘上什么都不留 —— 不报错、只是文件不见了, 正是本工程最怕的那种静默失效。
        (press 开局 时若上一局还没走完, 它也一样到此为止 —— 与棋盘被重摆这件事一致。)
    */
    setHumanGameInProgress(false);
    if (m_animTimer != nullptr) {
        m_animTimer->stop();
    }
    if (m_busyClickTimer != nullptr) {
        m_busyClickTimer->stop();
    }
    m_busyClickSeen = false;
    m_thinkStage.clear();
    update();
}

void ChessBoard::checkGameOver(int result)
{
    QString msg;
    if (result == Chess::RESULT_RED_WIN) {
        msg = "红方胜!";
    } else if (result == Chess::RESULT_BLACK_WIN) {
        msg = "黑方胜!";
    } else {
        /* RESULT_DRAW: 三次重复局面 或 60 回合自然限着 */
        msg = "平局!";
    }
    QMessageBox::information(this, "游戏结束", msg);
}

void ChessBoard::setAgentType(AgentType type)
{
    m_agentType = type;
}

/*
 * setDynamicRewardEnabled —— 界面上的"动态奖励"开关（用户口径: 保留旧的奖励方法）
 *
 * 说明、边界与三条警告都写在 chessboard.h 的那一段里。这里只说实现上的两点:
 *
 *  1. **四个旋钮是普通成员**（不是建网期参数）⇒ 对**已经建好的实例**当场生效。
 *     所以这个 setter 不需要重建任何 agent（重建会让"这一局下到一半"的实例被换掉,
 *     而 `moeDense` 那类建网期参数才必须走构造参数 —— 两者的区别见
 *     sacazmoetbagent.h 里那条警告）。
 *  2. **要遍历三支**：三个类各有自己的实例（`m_sfSACAZ` / `m_sfSACAZMoe` /
 *     `m_sfSACAZMoeMlp`），漏一支的表现是"那一支还在用旧奖励"而在面板上完全看不出来。
 *     59e5233 还原版**不在**这里 —— 它连这几个成员都没有（刻意的, 见 sacazlegacyagent.h）。
 */
void ChessBoard::setDynamicRewardEnabled(bool on)
{
    m_dynamicReward = on;
    if (m_sfSACAZ != nullptr) {
        applyRewardMethod(*m_sfSACAZ, on);
    }
    if (m_sfSACAZMoe != nullptr) {
        applyRewardMethod(*m_sfSACAZMoe, on);
    }
    if (m_sfSACAZMoeMlp != nullptr) {
        applyRewardMethod(*m_sfSACAZMoeMlp, on);
    }
    qInfo().noquote() << QStringLiteral("[reward] 奖励方法 = %1%2")
                             .arg(on ? QStringLiteral("动态奖励 (rewardShape=3, 材质 ×1.5 / 终局 ×1.5)")
                                     : QStringLiteral("旧奖励 (rewardShape=0, 材质 ×0.1 + 终局 ±1)"))
                             .arg(on ? QStringLiteral("  ⚠ 自对弈训练里 done 样本=0 ⇒ 杀将那一半目前不生效, "
                                                      "实测只改到吃子那一半")
                                     : QString());
}

/* ================================================================
 *  aiThink - 根据当前选中的agent类型选择走法
 *
 *  AI Agent 类型:
 *    AGENT_ALPHABETA : Alpha-Beta 剪枝 (深度 AB_DEPTH=4 —— 这一行原文写"深度 5",
 *                      与实际常量不符, 2026-09 一并改正)
 *    AGENT_AB_L1/L2/L3 : 同一搜索的三档弱等级 (深度 1/2/3)。**纯搜索、无权重**:
 *                      深度由 ChessBoard::abDepthOf() 单一来源给出。
 *    AGENT_MCTS      : 蒙特卡洛树搜索 (800次模拟)
 *    AGENT_PG        : Policy Gradient (PGEagent)
 *    AGENT_DQN       : Deep Q-Network (DQNAgent)
 *    AGENT_PPOMCTS   : PPO+MCTS AlphaZero风格 (默认 400 次模拟, 见 PPO_SIMS ——
 *                      这一行原文写"80", 与实际常量不符, 一并改正)
 *    AGENT_DQNMCTS   : DQN+MCTS (默认 200 次迭代, 见 DQNMCTS_ITERATIONS)
 *    AGENT_EVAB      : EVAB - 学会评估的 Alpha-Beta (见 docs/agent_evab_design.md)
 * ================================================================ */
/*
 * opponentStepForRollout - "问对手一手" (P1)
 *
 * 时机与契约:
 *   * 由 preTrainThenDecide 装进 OpponentPolicy.stepFor, 在学习方的**探索**里被调用;
 *   * 调用发生时, 学习方的决策正持有 m_agentMutex (整段决策都在锁内, 见
 *     aiThinkForAgentRaw), 所以这里可以直接调 decideOnEnvRawLocked —— 它的契约正是
 *     "调用方已持有 m_agentMutex"。**绝不能**改成调用 aiThinkForAgentRaw: 那会再拿一次
 *     同一把非递归锁 ⇒ 自锁死锁 (这条注释就是为了拦住那次"看起来更省事"的改动)。
 *
 * 三件必须做的事 (每件都对应一个具体的坏结果):
 *   1. **角色换成对手那一方**: 评估模式下 A 是学习者、B 是冻结方; 学习方的探索里问的是
 *      B, 所以这一手要按 B 的角色判 (实例选择/搜索学习掩码都读它)。忘了换的后果是
 *      评估模式下 B 被问到时**用了常驻实例或允许学习** —— 冻结失效且没有读数。
 *   2. **m_opponentQueryDepth +1**: 让 preTrainThenDecide 与 searchLearningEnabled 在
 *      查询期间一律拒绝学习 (见这两处的注释)。
 *   3. **失败计数**: 问不到 (无效 Step) 时计一次, 由探索回退到自对弈 —— 静默回退会让
 *      "对手参数没生效"看起来与"对手参数没开"一模一样。
 */
Step ChessBoard::opponentStepForRollout(AgentType type, int turn)
{
    m_opponentQueryCount.fetch_add(1);
    const SideRole saved = m_sideRole;
    setSideRole(saved == SIDE_LEARNER ? SIDE_FROZEN : SIDE_LEARNER);
    m_opponentQueryDepth++;
    const Step s = decideOnEnvRawLocked(turn, type);
    m_opponentQueryDepth--;
    setSideRole(saved);
    if (!s.valid) {
        m_opponentQueryUnmatched.fetch_add(1);
    }
    return s;
}

/* ================================================================
 *  preTrainThenDecide - 走子前的"先探索环境 + 预训练" (仿 snakeAI)
 *
 *  snakeAI 的每个 Agent::xxxAction() 都是三步: 先把当前状态记下来, 再从当前状态
 *  出发用探索策略滚若干步、用这批新鲜经验在线训练一次, 最后才基于当前状态做决策。
 *  这里把同一套流程套在所有 agent 上 —— 有监督式的 agent (Alpha-Beta / MCTS)
 *  的 exploreAndTrain() 是空实现, EVAB 则把探索结果蒸馏回评估网络, 所以调用它对
 *  它们无害; 而且探索全程用 moveForward/moveBack 试走并原样回退, 不会改动真棋局。
 * ================================================================ */
std::string ChessBoard::preTrainThenDecide(AgentBase *agent, int color)
{    if (agent == nullptr) {
        emitStage(QStringLiteral("① 搜索 / 决策"));
        return std::string();
    }
    /*
       ---- P1: "正在问对手一手"期间不许探索/训练 ----
       问对手 = 走一次它的决策协议 (decideOnEnvRawLocked), 而那条路也会走到本函数。
       不拦住的话, "问对手一手"会让对手在**学习方的探索过程中**自己再探索 + 训练一次:
       对手的权重被悄悄改了 (评估模式当场失效), 而且会无限递归 (对手的探索又要问它的
       对手)。所以查询期间直接返回 —— 问对手**只要它的着法**, 不要它的学习。
    */
    if (m_opponentQueryDepth > 0) {
        m_opponentQueryLearnBlocked.fetch_add(1);
        return std::string("对手查询: 只取着法, 不探索不训练");
    }
    if (!m_preTrainEnabled.load()) {
        emitStage(QStringLiteral("① 搜索 / 决策"));
        return std::string("探索+预训练: 已关闭");
    }
    /*
       ---- 对弈模式的闸门 (P0-a) ----
       评估模式 (只让 A 方学) 与"只对弈不学习"模式下, 这一手**不做**在线更新。
       闸门放在这里而不是放在界面那个勾选框上: 后者只是"用户想不想训", 而这里是
       "本场对局的语义允不允许训" —— 两者都在, 取与。理由见 chessboard.h 的 MatchMode。
    */
    if (!perMoveLearningEnabled()) {
        m_matchLearningBlocked.fetch_add(1);
        emitStage(QStringLiteral("① 搜索 / 决策 (本场对局禁用在线学习: %1)")
                      .arg(matchModeName(m_matchMode.load())));
        return std::string("探索+预训练: 本场对局模式已禁用 (评估/只对弈模式)");
    }
    const int steps = m_preTrainSteps.load();
    if (steps <= 0) {
        /* 步数设成 0 等价于关掉探索 (界面上允许这么设, 用来做对照) */
        emitStage(QStringLiteral("① 搜索 / 决策 (探索步数=0)"));
        return std::string("探索+预训练: 已关闭 (步数=0)");
    }
    emitStage(QStringLiteral("① 探索环境 + 预训练 (≤%1 步)").arg(steps));
    /*
       ---- P1: 把"对手在这个局面上会怎么走"交给探索 ----
       默认 budget = 0 ⇒ 探索里对手那一半仍然由学习方自己的策略产生, 与改动前**逐字相同**。
       打开的条件有四条, 每条都对应一个"不打开"的正当理由:
         * 开关与预算 (用户没开 / 预算是 0);
         * **正在对弈** —— 只有对局循环知道对手是谁 (m_rolloutOpponentType);
         * 那一手是不是在问对手 (m_opponentQueryDepth, 否则递归);
         * 对手类型是不是一个**有决策分支**的 agent —— 哨兵 AGENT_MCTS 表示"没有对手"
           (人机对战), 那时保持自对弈。MCTS 自己作为对手类型是合法的, 所以哨兵用它
           与"真选了 MCTS 当对手"共用同一个值, 判据落在 m_matchRunning 上 (见上面第二条)。
    */
    OpponentPolicy opp;
    if (m_opponentInRollout && m_opponentRolloutPlies > 0
        && m_matchRunning.load() && m_opponentQueryDepth == 0) {
        const AgentType oppType = m_rolloutOpponentType;
        opp.budget = m_opponentRolloutPlies;
        opp.name = agentDisplayName(oppType).toStdString();
        opp.stepFor = [this, oppType](int turn) -> Step {
            return opponentStepForRollout(oppType, turn);
        };
    }
    const bool trained = agent->exploreAndTrain(color, steps, opp);
    std::string info = agent->getExploreInfo();
    if (info.empty()) {
        info = trained ? "已预训练" : "无需预训练 (该 agent 没有在线可训练参数)";
    }
    /*
       上报这次在线训练的损失 (界面的"训练损失曲线")。不上报损失的 agent 返回 NaN
       (见 AgentBase::getLastTrainLoss), 曲线控件会直接丢弃这个点 —— 所以这里不需要
       区分"是 0"和"没上报", 用 isfinite 判断即可。
    */
    {
        const float loss = agent->getLastTrainLoss();
        if (std::isfinite(loss)) {
            emit trainLossSample((double)loss, QString::fromStdString(agent->getName()),
                                 m_trainSampleNo.fetch_add(1) + 1);
        }
    }
    {
        QMutexLocker locker(&m_infoMutex);
        m_lastExploreInfo = info;
    }
    /*
       用信号把说明送到界面, 而不是让 GUI 线程去读 m_lastExploreInfo ——
       这里在 AI 工作线程, 直接读同一个 std::string 是数据竞争 (A7 那一类)。
    */
    emit aiExploreInfo(QString::fromStdString(info));
    emitStage(QStringLiteral("② 搜索 / 决策"));
    return info;
}

/* ================================================================
 *  ---- 对弈模式 (P0-a, 2026-09, dev-selfplay) ----
 * ================================================================
 *
 * 动机与三条语义见 chessboard.h 的 MatchMode 注释。这里只说明**为什么判据写在这里**:
 *
 *   * 在它之前, "对弈时学不学"这件事**没有任何单一判据** —— 每一手都无条件走
 *     preTrainThenDecide, 而 SAC 还有第二条路径 (selectMove 里的 learnFromSearch)。
 *     实测 (test_match [2.7c]): 关掉界面的"探索+预训练"之后, PPO 0 次更新、
 *     而 SAC+AZ **仍然每手 20 次** —— 于是"关掉探索 = 只看不下"只对一部分 agent 成立。
 *   * 所以本模式必须**同时**管住两条路径, 而且判据只能有一处, 否则一定会漏:
 *       ① 每手一次更新 (preTrainThenDecide)          -> updateEnabledForSide
 *       ② 从自己的搜索学一次 (SAC learnFromSearch)   -> searchLearningEnabled
 *     两者在 MATCH_NO_LEARN 下都返回 false; MATCH_EVAL 下只放行 A 那一侧。
 *
 * ⚠ 用 `m_matchRunning` 区分"这是对弈还是人机": 人机对战 (aiThink) 沿用原有行为,
 *   不受对弈模式影响 (用户没有要求改人机那条路)。
 */

QString ChessBoard::matchModeName(MatchMode m)
{
    switch (m) {
    case MATCH_TRAIN:    return QStringLiteral("训练对局 (双方各自学习)");
    case MATCH_EVAL:     return QStringLiteral("评估对局 (冻结 B 方, 只让 A 方学)");
    case MATCH_NO_LEARN: return QStringLiteral("只对弈不学习 (纯对照, 两条学习路径都关)");
    }
    return QStringLiteral("未知模式");
}

bool ChessBoard::matchLearnsSomething() const
{
    return m_matchMode.load() != MATCH_NO_LEARN;
}

/*
 * ---- 学习闸门 (唯一判据) ----
 *
 * 语义表 (role x mode):
 *
 *      role             TRAIN   EVAL    NO_LEARN
 *      SIDE_LEARNER      学      学       不学
 *      SIDE_FROZEN       学     不学      不学
 *      SIDE_PURE_SEARCH  不学    不学      不学      (没有可训练参数, 与模式无关)
 *
 * 对弈: 每一手由 playMatchGame 按"红方是否 A 方"填 role (A=LEARNER, B=FROZEN)。
 * 人机: AI 固定执黑 ⇒ role 固定 = SIDE_FROZEN (见 process())。所以"评估对局/只对弈"
 *       对人机**同样生效**, 一个控件一种语义 (这是 P0-b 补掉的那个洞)。
 */
bool ChessBoard::updateEnabledForSide(SideRole role) const
{
    if (role == SIDE_PURE_SEARCH) {
        return false;      /* 纯搜索 agent 本来就不学 —— 与模式无关 */
    }
    switch (m_matchMode.load()) {
    case MATCH_TRAIN:    return true;                                   /* 双方各自学 */
    case MATCH_EVAL:     return role == SIDE_LEARNER;                   /* 只让学习者学 */
    case MATCH_NO_LEARN: return false;                                  /* 都不学 */
    }
    return true;
}

bool ChessBoard::searchLearningEnabled() const
{
    /*
       两条学习路径必须同进同退: 口径就是 updateEnabledForSide 那一个函数。
       [P1] 另加一条:**正在问对手一手**时一律不许学 —— SAC+AZ 的 selectMove 里带着
       learnFromSearch 那条更新路径, 而"问对手"就是走一次 selectMove。不拦住的话,
       学习方的探索会让**对手**每被问到一次就更新一次 (评估模式下 B 方当场就不是冻结的了,
       而且这条更新发生在谁都看不见的地方)。
    */
    if (m_opponentQueryDepth > 0) {
        m_opponentQueryLearnBlocked.fetch_add(1);
        return false;
    }
    return updateEnabledForSide(m_sideRole);
}

bool ChessBoard::perMoveLearningEnabled() const
{
    if (m_sideRole == SIDE_PURE_SEARCH) {
        return false;
    }
    return updateEnabledForSide(m_sideRole);
}

/*
 * 决策收尾的两个模板都定义在头文件里 (reportLearnedLoss 也在那里) —— 它们必须看到
 * agent 的具体类型才能取 getLearnSteps()/learnFromSearch, 而 AgentBase 上没有这些接口。
 * 这里只保留**非模板**的判据实现 (matchModeName / updateEnabledForSide / ...),
 * 它们是模式语义的唯一来源。
 */

/*
 * reportLearnedLossOf - "决策里学了一次"的**口径唯一实现** (头文件里那个模板转到这里)。
 *
 * 只有 learnSteps **前进**了才上报: 一次决策可能有两次更新 (rollout + 搜索样本),
 * 无条件上报会让损失曲线一步出现两个点, 把"每手一个点"的口径弄乱。
 *
 * [2026-09] 参数是 `AgentBase *` + 两个进度数 (不是 `SACAZAgent *`): 还原版
 * SACAZLegacyAgent 与 SACAZAgent 是**两个没有继承关系的类**, 只有 AgentBase 这一层
 * 是公共的 —— 而"是否上报"的口径本来就只依赖 learnSteps 有没有前进, 与是哪一支无关
 * (还原版没有"从搜索学一次"这条路径, 所以它永远走到不前进那一支, 这里不需要特判)。
 */
void ChessBoard::reportLearnedLossOf(AgentBase *agent, int learnSteps, int learnStepsBefore)
{
    if (agent == nullptr || learnSteps <= learnStepsBefore) {
        return;      /* 这一步没发生更新 (例如 learnFromSearch 关着, 或池子还不够一个批) */
    }
    const float loss = agent->getLastTrainLoss();
    if (std::isfinite(loss)) {
        emit trainLossSample((double)loss, QString::fromStdString(agent->getName()),
                             m_trainSampleNo.fetch_add(1) + 1);
    }
}

/* ================================================================
 *  ---- ④ 奖励曲线的"学习口径" (2026-09, 见 aiagent.h 的说明) ----
 * ================================================================ */

bool ChessBoard::agentHasLearningReward(AgentType type)
{
    return agentHasLearningRewardTable(type);
}

QString ChessBoard::agentRewardCaliperLabel(AgentType type)
{
    if (!agentHasLearningRewardTable(type)) {
        return QStringLiteral("引擎口径(材质x1+终局±1)");
    }
    return QStringLiteral("学习口径(材质x0.1+每步代价+终局±1)");
}

/*
 * abDepthOf - "这个 agent 类型用多深的 Alpha-Beta" (单一来源)
 *
 * 返回 0 = **不是 Alpha-Beta**, 调用方不该拿它去构造 ABAgent。判 0 而不是判
 * ">= 1", 是为了让"忘了给新类型登记深度"变成**当场可见**的行为 (走到 default 兜底,
 * 而不是静默按某个深度下棋)。AGENT_ALPHABETA 保持在 AB_DEPTH(=4), 与新增的
 * 三档无关 —— 见 chessboard.h 里那个函数的注释。
 */
int ChessBoard::abDepthOf(AgentType type)
{
    switch (type) {
    case AGENT_ALPHABETA: return AB_DEPTH;
    case AGENT_AB_L1:     return AB_L1_DEPTH;
    case AGENT_AB_L2:     return AB_L2_DEPTH;
    case AGENT_AB_L3:     return AB_L3_DEPTH;
    default:              return 0;      /* 不是 Alpha-Beta */
    }
}

AgentBase *ChessBoard::agentInstance(AgentType type) const
{
    /* 静态成员: 与 aiThinkForAgentRaw 里按需 new 出来的是**同一批对象** (单一来源)。 */
    switch (type) {
    case AGENT_PG:          return m_sfPG;
    case AGENT_DQN:         return m_sfDQN;
    case AGENT_PPOMCTS:     return m_sfPPOMCTS;
    case AGENT_PPOMCTS_MLP: return m_sfPPOMCTSMLP;
    case AGENT_DQNMCTS:     return m_sfDQNMCTS;
    case AGENT_EVAB:        return m_sfEVAB;
    case AGENT_SACAZ:       return m_sfSACAZ;
    case AGENT_SACAZ_MOE:   return m_sfSACAZMoe;
    case AGENT_SACAZ_MOE_MLP: return m_sfSACAZMoeMlp;
    case AGENT_SACAZ_OLD:   return m_sfSACAZOld;
    case AGENT_SACAZ_OLD_MOE: return m_sfSACAZOldMoe;
    case AGENT_DQNAB:       return m_sfDQNAB;
    /* DQN+MCTS (MoE+TB 专家): 独立类 (src/dqnmctsmoetbagent.h) */
    case AGENT_DQNMCTS_MOE: return m_sfDQNMCTSMOE;
    default:                return nullptr;   /* Alpha-Beta / MCTS: 没有常驻对象 */
    }
}

/* ================================================================
 *  ---- 冻结对手 (B 方) 的权重快照 (P0-b 收尾) ----
 *
 *  语义与理由见 chessboard.h 的那一段。这里只写**三个容易写错的地方**:
 *
 *  1. 快照必须在**对局循环之前**取。放到"B 第一手决策时再取"看起来更省 (那时常驻
 *     实例一定已经建好了), 但**A 已经学过一手了** —— 而 A 与 B 共用同一个常驻实例,
 *     于是那份"开场快照"里已经混进了 A 的第一次更新。表现是"评估仍然不完全干净",
 *     且完全看不出来 (差的那一点梯度没有任何读数)。
 *  2. 常驻实例不存在时, 先按**决策路径同一套参数**把它建出来 + 载入扫描到的权重,
 *     再取快照。判据用 s_weightPaths (与 startupLoad 同一份), 不能自己拼文件名 ——
 *     PPO+MCTS 曾经因为"扫描的名字与实际写出的名字不一致"而从来没被载入过。
 *     ⚠ 这里**只在该类型没有实例时**才调 loadAgentModel: 实例已存在时再 load 会把
 *     内存里已经学到的东西覆盖掉 (评估模式里 A 的在线学习成果)。
 *  3. 冻结实例的类型分派与决策路径共用 makeAgentInstance: 构造参数写错的后果是
 *     save/load 结构指纹不匹配 (当场失败) 或**静默换骨干** (SAC 三支), 两种都不该
 *     靠"记得同步改两处"来避免。
 * ================================================================ */

/*
 * frozenOpponentOverrideFor - 这一手要不要改用冻结实例 (唯一判据)
 *
 * 三个条件必须同时成立:
 *   * 有冻结实例 (评估模式开场建好了);
 *   * 类型对得上 (每个类型的 case 只会问到自己的类型);
 *   * **这一手是冻结方** (SIDE_FROZEN) 且**正在对弈**。
 * 最后一条是"人机那条路不许被卷进来"的保险: 冻结实例只在 matchAgents 生命周期内存在,
 * 而人机决策 (aiThinkRaw) 从不在对局期间发生 —— 就算以后有人让它发生, 这里也会拦住
 * (人机里 AI 也是 SIDE_FROZEN, 但它用的必须是它自己那份常驻权重)。
 */
AgentBase *ChessBoard::frozenOpponentOverrideFor(AgentType type) const
{
    if (m_frozenOpponent == nullptr || type != m_frozenOpponentType) {
        return nullptr;
    }
    if (!m_matchRunning.load() || m_sideRole != SIDE_FROZEN) {
        return nullptr;
    }
    /*
       诊断计数: "B 方实际用快照走了多少手"。
       为什么需要它: "建了冻结实例"与"决策真的走了它"是两件事 —— 只断言前者的话,
       decisionInstance 那一行被谁改回 m_sfXXX 都测不出来 (而那就是本工程最怕的
       "改了但静默失效")。计数为 0 = 快照建了却没被用上。
    */
    m_frozenDecisions.fetch_add(1);
    return m_frozenOpponent;
}

/*
 * makeAgentInstance - 按类型新建一个实例 (不登记到常驻指针上)
 *
 * 构造参数必须与 aiThinkForAgentRaw / loadAgentModel / 后台训练那几处**逐字一致**;
 * 与 SAC 三支共用 createSACAZAgent / createSACAZLegacyAgent 那一条"唯一构造点"
 * 约定 (见那两个函数的注释)。
 */
AgentBase *ChessBoard::makeAgentInstance(AgentType type)
{
    switch (type) {
    case AGENT_PG:        return new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
    case AGENT_DQN:       return new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
    case AGENT_PPOMCTS: {
        /*
           `applyPpoBalance` 必须在**任何前向之前**调用 —— 它给 MoE 门控接线
           (偏置均衡默认开; MLP 门控按常量决定)。见 applyPpoBalance 的说明。
           `enableMlpGate` 是幂等的, 所以"工厂建网"与"决策路径兜底建网"两处都接
           也不会互相破坏。
        */
        PPOMCTSAgent *ag = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
        applyPpoBalance(*ag);
        return ag;
    }
    case AGENT_PPOMCTS_MLP: {
        /* 骨干参数 (64, 0.1f, true, MlpExperts) 必须与决策路径一致, 否则结构指纹对不上 */
        PPOMCTSAgent *ag = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                            true, RL::PPO::Backbone::MlpExperts);
        applyPpoBalance(*ag);
        return ag;
    }
    case AGENT_DQNMCTS:   return new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
    case AGENT_EVAB:      return new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
    case AGENT_SACAZ:     return createSACAZAgent(env, AGENT_SACAZ, m_dynamicReward);
    /*
       [2026-09 独立类] AGENT_SACAZ_MOE 走它自己的构造点 (`SACAZMoETbAgent`) ——
       与 AGENT_SACAZ (纯 MLP 的 SACAZAgent) 已经不是同一个类了, 手写
       `new SACAZAgent(...)` 也建不出这支骨干 (它会静默变成纯 MLP)。
    */
    case AGENT_SACAZ_MOE: return createSACAZMoETbAgent(env, true, m_dynamicReward);
    /* [2026-09 独立类] MoE+MLP 那一支: 同样走它自己的构造点 (手写 new SACAZAgent 建不出这个骨干) */
    case AGENT_SACAZ_MOE_MLP: return createSACAZMoEMlpAgent(env, true, m_dynamicReward);
    /* 还原版是**独立类**: 必须走它自己那个构造点, 手写 new SACAZAgent 会静默换口径 */
    case AGENT_SACAZ_OLD:     return createSACAZLegacyAgent(env, AGENT_SACAZ_OLD);
    case AGENT_SACAZ_OLD_MOE: return createSACAZLegacyAgent(env, AGENT_SACAZ_OLD_MOE);
    case AGENT_DQNAB: {
        DQNABAgent *a = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                       DQNABAgent::Backbone::SparseMoeTb);
        a->nodeBudget = DQNAB_NODES;      /* 与 aiThinkRaw 同一预算, 否则搜索行为不一致 */
        return a;
    }
    case AGENT_DQNMCTS_MOE: {
        /*
           [2026-09 dev-dqnmcts-moetb] 独立类 DQNMCTSMOETbAgent。构造参数必须与
           aiThinkRaw / loadAgentModel / 后台训练那几处**逐字一致** (见本函数头注释) ——
           它不含骨干开关 (denseMoe 走默认 false), 所以两处 ctor 的差别只可能在
           隐层宽度与那几个超参上。
        */
        DQNMCTSMOETbAgent *a = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                                     1.0f, 1.5f);
        a->simulations = DQNMCTS_MOE_SIMS;   /* 与 aiThinkRaw 同一预算 */
        return a;
    }
    default:
        return nullptr;   /* 纯搜索 agent (AB 各档 / MCTS): 没有实例可建 */
    }
}

/*
 * saveWeightsOf / loadWeightsInto - "指定实例"的权重读写 (类型分派只有这一份)
 *
 * saveCurrentAgentModel / loadAgentModel 与冻结快照的取数、审计都走这里:
 * 三处各写一遍 switch 的话, "新加一个 agent 类型"时漏掉一处就是静默失效
 * (而那正是本工程反复栽的那个跟头)。
 */
bool ChessBoard::saveWeightsOf(AgentBase *inst, AgentType type, const std::string &filepath)
{
    if (inst == nullptr) {
        return false;
    }
    switch (type) {
    case AGENT_PG:          return static_cast<PGEagent *>(inst)->savePolicy(filepath);
    case AGENT_DQN:         return static_cast<DQNAgent *>(inst)->saveModel(filepath);
    case AGENT_PPOMCTS:     return static_cast<PPOMCTSAgent *>(inst)->saveModel(filepath);
    case AGENT_PPOMCTS_MLP: return static_cast<PPOMCTSAgent *>(inst)->saveModel(filepath);
    case AGENT_DQNMCTS:     return static_cast<DQNMCTSAgent *>(inst)->saveModel(filepath);
    case AGENT_EVAB:        return static_cast<EVABAgent *>(inst)->saveModel(filepath);
    case AGENT_SACAZ:       return static_cast<SACAZAgent *>(inst)->saveModel(filepath);
    case AGENT_SACAZ_MOE:   return static_cast<SACAZMoETbAgent *>(inst)->saveModel(filepath);
    case AGENT_SACAZ_MOE_MLP: return static_cast<SACAZMoEMlpAgent *>(inst)->saveModel(filepath);
    case AGENT_SACAZ_OLD:
        return static_cast<SACAZLegacyAgent *>(inst)->saveModel(filepath);
    case AGENT_SACAZ_OLD_MOE:
        return static_cast<SACAZLegacyAgent *>(inst)->saveModel(filepath);
    case AGENT_DQNAB:       return static_cast<DQNABAgent *>(inst)->saveModel(filepath);
    /* DQN+MCTS (MoE+TB 专家): 独立类, 写 <prefix>_trunk + <prefix>_q (见 weightFilesOf) */
    case AGENT_DQNMCTS_MOE:
        return static_cast<DQNMCTSMOETbAgent *>(inst)->saveModel(filepath);
    default:                return false;      /* 纯搜索 agent: 没有权重 */
    }
}

bool ChessBoard::loadWeightsInto(AgentBase *inst, AgentType type, const std::string &filepath)
{
    if (inst == nullptr) {
        return false;
    }
    switch (type) {
    case AGENT_PG:          return static_cast<PGEagent *>(inst)->loadPolicy(filepath);
    case AGENT_DQN:         return static_cast<DQNAgent *>(inst)->loadModel(filepath);
    case AGENT_PPOMCTS:     return static_cast<PPOMCTSAgent *>(inst)->loadModel(filepath);
    case AGENT_PPOMCTS_MLP: return static_cast<PPOMCTSAgent *>(inst)->loadModel(filepath);
    case AGENT_DQNMCTS:     return static_cast<DQNMCTSAgent *>(inst)->loadModel(filepath);
    case AGENT_EVAB:        return static_cast<EVABAgent *>(inst)->loadModel(filepath);
    case AGENT_SACAZ:       return static_cast<SACAZAgent *>(inst)->loadModel(filepath);
    case AGENT_SACAZ_MOE:   return static_cast<SACAZMoETbAgent *>(inst)->loadModel(filepath);
    case AGENT_SACAZ_MOE_MLP: return static_cast<SACAZMoEMlpAgent *>(inst)->loadModel(filepath);
    case AGENT_SACAZ_OLD:
        return static_cast<SACAZLegacyAgent *>(inst)->loadModel(filepath);
    case AGENT_SACAZ_OLD_MOE:
        return static_cast<SACAZLegacyAgent *>(inst)->loadModel(filepath);
    case AGENT_DQNAB:       return static_cast<DQNABAgent *>(inst)->loadModel(filepath);
    /* DQN+MCTS (MoE+TB 专家): 独立类, 读同样的 <prefix>_trunk + <prefix>_q */
    case AGENT_DQNMCTS_MOE:
        return static_cast<DQNMCTSMOETbAgent *>(inst)->loadModel(filepath);
    default:                return false;      /* 纯搜索 agent: 没有权重 */
    }
}

/*
 * 冻结快照的文件前缀 (单一来源): 每种类型一个, 免得两个类型互相覆盖
 * (与 tmpWeightsOf 同一条理由 —— 那种覆盖是静默的, 表现只是"快照不是我取的那份")。
 */
static std::string frozenSnapshotPathOf(ChessBoard::AgentType type)
{
    return std::string("weights/_temp_match_frozen_") + std::to_string((int)type);
}
static std::string frozenAuditPathOf(ChessBoard::AgentType type)
{
    return std::string("weights/_temp_match_frozen_audit_") + std::to_string((int)type);
}

/* ================================================================
 *  ---- 对弈期间的"权重不落盘" (2026-10 用户口径) ----
 * ================================================================
 *
 * 用户口径与三条边界写在 chessboard.h 的 public 段 (那里是给人看的), 机制写在
 * rl/weightio.hpp (那里是实现)。这里只做三件事: 自检前缀判据、把两个作用域合到
 * 那一个开关上、以及退出前的强制解除。
 */

/*
 * 运行期自检: "所有临时权重路径都以 WeightIO 的前缀开头"。
 * 改道的判据就是这个前缀, 而它一旦漂移 (有人新加一条临时路径用了别的名字/别的位置),
 * "对弈期间不落盘"就会**静默失效**: 那一条临时文件照旧写盘, 而读数上与生效完全一样。
 * 所以在第一次武装改道时核对一遍 —— 本工程的惯例是"约定必须有一条能响的检查"。
 */
static void verifyTransientWeightPathsOnce()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, []() {
        const std::string paths[] = {
            TMP_WEIGHTS,
            TMP_WEIGHTS_PPOMCTS_MLP,
            TMP_WEIGHTS_SACAZ,
            TMP_WEIGHTS_SACAZ_MOE,
            TMP_WEIGHTS_SACAZ_MOE_MLP,
            TMP_WEIGHTS_SACAZ_OLD,
            TMP_WEIGHTS_SACAZ_OLD_MOE,
            TMP_WEIGHTS_DQNAB,
            TMP_WEIGHTS_DQNMCTS_MOE,
            /* 快照/审计那两条是拼出来的 (每种类型一个), 至少抽查两种类型 */
            frozenSnapshotPathOf(ChessBoard::AGENT_SACAZ),
            frozenSnapshotPathOf(ChessBoard::AGENT_DQNAB),
            frozenAuditPathOf(ChessBoard::AGENT_PPOMCTS),
        };
        QStringList bad;
        for (const std::string &p : paths) {
            if (!RL::WeightIO::isTransientPath(p)) {
                bad << QString::fromStdString(p);
            }
        }
        if (!bad.isEmpty()) {
            qWarning().noquote()
                << QStringLiteral("[weights] 这些临时权重路径没命中 '%1' 前缀, "
                                  "对弈期间它们会**照旧落盘** (改道对它们无效): %2")
                       .arg(QString::fromLatin1(RL::WeightIO::kTransientPrefix),
                            bad.join(QStringLiteral(", ")));
        }
    });
}

void ChessBoard::updatePlayWeightDivert()
{
    verifyTransientWeightPathsOnce();
    const bool want = m_playDivertMatch.load() || m_playDivertHuman.load();
    if (want == RL::WeightIO::divertEnabled()) {
        return;                     /* 状态没变: 不重复打日志 (每手都会调到这里) */
    }
    RL::WeightIO::setDivertEnabled(want);
    if (want) {
        /*
           计数从零开始: "本场改道了几次"才是报告要的数, 跨场累加会让第二场之后
           的读数说不清是哪一场的 (与 m_frozenDecisions 每次归零同一条理由)。
        */
        RL::WeightIO::resetCounters();
        qInfo().noquote()
            << QStringLiteral("[weights] 对弈期间权重不落盘: 已开启 (临时权重路径 %1* "
                              "改走内存 —— 后台训练照常, 退出时统一保存)")
                   .arg(QString::fromLatin1(RL::WeightIO::kTransientPrefix));
    } else {
        qInfo().noquote()
            << QStringLiteral("[weights] 对弈期间权重不落盘: 已关闭 "
                              "(本场改道 %1 次 / %2 字节; 之后临时权重照旧写盘)")
                   .arg((qlonglong)RL::WeightIO::divertedWrites())
                   .arg((qlonglong)RL::WeightIO::divertedBytes());
    }
}

bool ChessBoard::isPlayWeightDivertOn() const
{
    return RL::WeightIO::divertEnabled();
}

int ChessBoard::divertedWeightWrites() const
{
    return (int)RL::WeightIO::divertedWrites();
}

long long ChessBoard::divertedWeightBytes() const
{
    return RL::WeightIO::divertedBytes();
}

void ChessBoard::setHumanGameInProgress(bool on)
{
    /*
       [2026-10] "这一局人机开始了" 的**唯一**时刻: 玩家落下这一局的**第一子**时
       (`m_playDivertHuman` false -> true)。为什么在这里发信号、而不是新加一个 reset 钩子:
       * 本函数在**每次**玩家落子时都会被调 (幂等), 所以只有这个跳变能代表"新的一局";
       * 人机那条路**没有** `matchStarted` 那种"场开始"信号, 而界面需要它来给
         "行为克隆" 那条曲线建线 —— 不建线的话 `CurveChart::addPoint` 是**静默 return**,
         表现就是"人机里 BC 明明在训练, 曲线却一直空着"(2026-10 用户报的缺口)。
       为什么按"玩家第一子"算而不是按"按了开局": 启动后直接走子也是一局棋, 而那时没人
       按过开局 —— 与上面那条"对弈期间权重不落盘"的起算口径逐字一致。
    */
    const bool was = m_playDivertHuman.exchange(on);
    if (on && !was) {
        emit humanGameStarted();
    }
    /*
       [2026-10] 人机那一局的 BC 读数在**这一局结束时**(或重开一局时) 收尾:
       清掉"本局已经归零过"的标志 ⇒ 下一局的第一次采样会重新归零 (见 bcSampleForHumanTurn)。
       人机没有"场"的概念, 所以"一局"就是读数的单位 —— 与对局那条路每场归零同一个口径。
    */
    if (!on) {
        m_bcHumanSession.store(false);
    }
    updatePlayWeightDivert();
}

/*
 * freezeOpponentToSnapshot - 取开场快照 + 建冻结实例
 *
 * 返回 true = 冻结实例建好了 (B 那一手从此走它)。任何一步失败都返回 false, 并把
 * **原因**写进 outNote (由调用方写进对局报告) —— 静默失败在这里最危险: 报告照样说
 * "评估对局", 而 B 其实还是那张会被 A 改的网。
 */
bool ChessBoard::freezeOpponentToSnapshot(AgentType type, QString &outNote)
{
    outNote.clear();
    if (abDepthOf(type) > 0 || type == AGENT_MCTS) {
        outNote = QStringLiteral("%1 没有可训练权重 (纯搜索) ⇒ '冻结'平凡成立")
                      .arg(agentDisplayName(type));
        return false;
    }
    /*
       常驻实例必须先在 (快照要从它身上取)。**只在它不存在时**才走 loadAgentModel:
       已存在时再 load 会把 A 已经学到的权重覆盖掉。
    */
    if (agentInstance(type) == nullptr) {
        std::string path = defaultWeightPath(type);
        auto it = s_weightPaths.find(type);
        if (it != s_weightPaths.end()) {
            path = it->second;
        }
        /* 返回值故意忽略: 载入失败时实例仍在 (随机初始化), 与决策路径的兜底同一行为;
           这里要的只是"取快照的对象存在", 而"有没有载入"由下面那句 note 说明。 */
        const bool loaded = loadAgentModel(type, path);
        if (!loaded) {
            outNote += QStringLiteral("(提示: %1 的权重没能从 %2 载入, 快照取的是当前内存里的权重) ")
                           .arg(agentDisplayName(type))
                           .arg(QString::fromStdString(path));
        }
    }

    m_frozenSnapshotPath = frozenSnapshotPathOf(type);
    {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (!saveWeightsOf(agentInstance(type), type, m_frozenSnapshotPath)) {
            outNote += QStringLiteral("!! B 方权重快照**写盘失败** ⇒ 本场没有冻结到快照 "
                                      "(回退为'只冻结学习', 见报告末行的说明)");
            m_frozenSnapshotPath.clear();
            return false;
        }
    }

    m_frozenOpponent = makeAgentInstance(type);
    if (m_frozenOpponent == nullptr) {
        outNote += QStringLiteral("!! 建冻结实例失败 (%1) ⇒ 本场没有冻结到快照")
                       .arg(agentDisplayName(type));
        m_frozenSnapshotPath.clear();
        return false;
    }
    {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (!loadWeightsInto(m_frozenOpponent, type, m_frozenSnapshotPath)) {
            outNote += QStringLiteral("!! 快照载入冻结实例失败 ⇒ 本场没有冻结到快照"
                                      " (绝不能让它拿随机权重去下棋)");
            delete m_frozenOpponent;
            m_frozenOpponent = nullptr;
            QFile::remove(QString::fromStdString(m_frozenSnapshotPath));
            m_frozenSnapshotPath.clear();
            return false;
        }
    }
    m_frozenOpponentType = type;
    m_frozenDecisions.store(0);      /* 本场重新计数 (见 frozenOpponentOverrideFor) */
    /*
       ---- 快照已经把冻结实例喂饱了 ⇒ 没开审计就可以放掉内存里那一份 (2026-10) ----
       对弈期间 `m_frozenSnapshotPath` 是**写进内存**的 (它命中 weights/_temp 前缀 ⇒
       走 rl/weightio.hpp 的改道), 而它在正常口径下只被"建冻结实例"读一次。
       不开审计时那份字节再没人要 —— 留着就是"评估一场多占一份 B 方权重" (稀疏 MoE
       是 3 x 146 MB)。开了审计则**必须留着**: 审计是拿"冻结实例现在的权重"与它逐字节
       比, 而快照从未落盘, 放掉之后就没有比对方了 (见 auditFrozenOpponentWeights)。
    */
    if (!m_frozenWeightAudit) {
        RL::WeightIO::drop(m_frozenSnapshotPath);
    }
    outNote += QStringLiteral("B 方 (%1) 已冻结为**开场权重快照**: 独立实例, A 的学习"
                              "写不到它身上, 后台训练也不会碰它")
                   .arg(agentDisplayName(type));
    qInfo().noquote() << QStringLiteral("[match] 冻结对手快照: %1 -> %2")
                             .arg(agentDisplayName(type))
                             .arg(QString::fromStdString(m_frozenSnapshotPath));
    return true;
}

/*
 * releaseFrozenOpponent - 拆掉冻结实例 (并删掉两份临时文件)
 *
 * 一定由 matchAgents 的 RAII 守卫调用 (正常结束 / 被中止 / 提前 return 三条路都要走)。
 * 不释放的后果不只是内存: 下一次对弈的"类型对得上 + 角色是冻结方"会让它**继续被用**,
 * 而它的权重是上一场的快照 —— 那种错没有任何读数能反映出来。
 */
void ChessBoard::releaseFrozenOpponent()
{
    if (m_frozenOpponent != nullptr) {
        delete m_frozenOpponent;
        m_frozenOpponent = nullptr;
    }
    /*
       [2026-10 对弈期间不落盘] 这两份文件在对弈期间是**内存里**的 (改道), 所以下面那些
       QFile::remove 会是 no-op —— 真正要放掉的是内存副本, 由这两句做。放在最前面:
       路径稍后就被 clear(), 那时再也找不到它们了 (那就是"几百 MB 再也放不掉"的形状)。
    */
    if (!m_frozenSnapshotPath.empty()) {
        RL::WeightIO::drop(m_frozenSnapshotPath);
    }
    if (!m_frozenAuditPath.empty()) {
        RL::WeightIO::drop(m_frozenAuditPath);
    }
    if (!m_frozenSnapshotPath.empty()) {
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_actor"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_critic"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_q1"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_q2"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_trunk"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_v"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath + "_a"));
        QFile::remove(QString::fromStdString(m_frozenSnapshotPath));
        m_frozenSnapshotPath.clear();
    }
    if (!m_frozenAuditPath.empty()) {
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_actor"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_critic"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_q1"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_q2"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_trunk"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_v"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath + "_a"));
        QFile::remove(QString::fromStdString(m_frozenAuditPath));
        m_frozenAuditPath.clear();
    }
}

/*
 * frozenAuditOk - 逐字节审计: 冻结实例此刻的权重 vs 开场快照
 *
 * 为什么值得写: 逻辑上冻结实例没有任何写入路径, 但"逻辑上不会"在本工程不算证据 ——
 * 用户要的是"B 的权重确实没变"这个可量的结论。做法是把冻结实例的权重再存一份, 与
 * 开场快照逐字节比对 (权重文件是确定性序列化, 同一个网络存两次字节相同)。
 * 多文件家族 (PPO 两文件 / SAC 三文件 / DQNAB 三文件) 逐个后缀比。
 *
 * 只在 m_frozenWeightAudit 打开时调用 (它要额外写一次权重文件)。
 */
/*
 * [2026-10 对弈期间不落盘] 逐字节比对必须**走同一套"内存优先、磁盘回落"的读法**
 * (见 rl/weightio.hpp): 对弈期间这两份权重都在内存里, 用 ifstream 去读会两个都打不开
 * ⇒ 审计恒假 ⇒ 报告说"B 的权重变了", 而真相是"它们根本没在磁盘上"。
 * 这是"机制生效了, 读数说它没生效"的典型形状, 也正是 weightFileWritten/Readable
 * 那两处要一起改的原因。
 */
static bool weightBytesEqual(const std::string &a, const std::string &b)
{
    std::string sa, sb;
    if (!RL::WeightIO::readBytes(a, sa) || !RL::WeightIO::readBytes(b, sb)) {
        return false;
    }
    return !sa.empty() && sa == sb;
}

/* 这一份权重此刻在不在 (内存里有, 或者磁盘上有) */
static bool weightBlobExists(const std::string &path)
{
    return RL::WeightIO::exists(path);
}

bool ChessBoard::auditFrozenOpponentWeights()
{
    if (m_frozenOpponent == nullptr || m_frozenSnapshotPath.empty()) {
        return false;
    }
    const AgentType type = m_frozenOpponentType;
    const std::string audit = frozenAuditPathOf(type);
    m_frozenAuditPath = audit;
    {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (!saveWeightsOf(m_frozenOpponent, type, audit)) {
            return false;
        }
    }
    /* 后缀表与 weightFilesOf 一致: 单文件 agent 就是前缀本身 */
    static const char *kSuffixes[] = { "", "_actor", "_critic", "_q1", "_q2",
                                       "_trunk", "_v", "_a" };
    bool anyCompared = false;
    for (const char *sfx : kSuffixes) {
        const std::string a = m_frozenSnapshotPath + sfx;
        if (!weightBlobExists(a)) {
            continue;      /* 这个后缀不存在 (该 agent 不是这种文件布局) */
        }
        if (!weightBytesEqual(a, audit + sfx)) {
            return false;
        }
        anyCompared = true;
    }
    return anyCompared;
}

double ChessBoard::learningStepRewardOrNaN(AgentType type, const Step &step, int mover)
{
    if (!agentHasLearningRewardTable(type)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    AgentBase *agent = agentInstance(type);
    /*
       对象还没建 = 该方在本局还没走过一手。返回 NaN 表示"这一手拿不到学习口径值",
       调用方按"还没拿到"处理 (learnOk* 只在真拿到值时才置真), 于是曲线在拿到之前
       显示引擎口径 —— 这种局面只会出现在"某一方一手都没走"的极端短局里, 正常对局
       第一手决策就会把对象建出来 (见 aiThinkForAgentRaw 的 lazy new)。
    */
    if (agent == nullptr || !agent->hasLearningReward()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (double)agent->learningStepReward(step, mover);
}

double ChessBoard::learningTerminalRewardOrNaN(AgentType type, int result, int mover)
{
    if (!agentHasLearningRewardTable(type)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    AgentBase *agent = agentInstance(type);
    if (agent == nullptr || !agent->hasLearningReward()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (double)agent->learningTerminalReward(result, mover);
}

std::string ChessBoard::getLastExploreInfo() const
{
    QMutexLocker locker(&m_infoMutex);
    return m_lastExploreInfo;
}

/*
 * 指定 agent 的自检报告 (界面"模型自检"面板 / 启动日志)。
 *
 * 按**指定的 agent 类型**取对应的那一份实例 (s_ 系列的静态指针, 与
 * preTrainThenDecide 用的是同一批对象) —— 不能"遍历所有 agent 返回第一个非空的",
 * 那会在切到别的 agent 之后继续显示上一个的自检结果。
 *
 * 两条路径的差别值得写下来:
 *   * PG / DQN / PPO+MCTS / DQN+MCTS / EVAB / SAC+AZ x2 / DQN+AB 都有**常驻实例**
 *     (m_sfXXX), 直接转发它们自己的 selfCheckReport();
 *   * Alpha-Beta / MCTS **没有常驻实例** —— 它们每一步都是现场构造一个 (见
 *     aiThinkRaw), 所以这里也现场造一个, 棋盘用**副本** (只读契约不受影响)。
 * 指针可能为 nullptr (该 agent 的权重文件没找到 ⇒ 没建过网), 此时返回空串,
 * 由 GUI 显示"没有自检项"。
 *
 * 报告函数被约定为**只读、可重复调用、不动棋盘** (见 aiagent.h 的契约), 所以可以在
 * GUI 线程直接调用。
 *
 * ---- 但"只读"不等于"可以并发" (2026-09, 用户报的崩溃) ----
 * 报告读的是**常驻 agent 的内部状态** (搜索树、网络计数器、MoE 直方图), 而同一时刻
 * 后台训练线程可能正在 `loadModel()` 把整份权重写进那个网络, AI/对弈线程也可能正在
 * 搜索 (它会往 `nodes` 里 push_back —— 迭代器失效, 读到已释放的内存)。
 * 所以只要是常驻实例, 一律取 `m_agentMutex` 再读: 这条锁与
 * `aiThink*` 的决策、`backgroundTrainLoop` 的载入/保存是同一把。
 * 代价是调用方可能等上几秒 (一次 PPO 决策 ~3.2 s, 一份 558 MB 权重的读写 ~9 s),
 * 所以 **GUI 线程不能直接调它** —— 面板刷新走 MainWindow 的后台 worker
 * (见 mainwindow.cpp 的 requestSelfCheckPanelUpdate)。
 *
 * 注意 (界面上也写明了, 否则 0 会被误读): 那些**对局累计**的计数来自主 agent,
 * 而 GUI 的后台训练跑在 clone 上、每轮才把权重同步回来 —— 训练期间这里的计数
 * 不会增长。所以各 agent 的报告里都同时给出"标准开局"那一份**确定性**读数
 * (与训练进度无关, 一打开就能看)。
 */
std::string ChessBoard::getAgentSelfCheck() const
{
    return getAgentSelfCheck(m_agentType);
}

std::string ChessBoard::getAgentSelfCheck(AgentType type) const
{
    /*
       Alpha-Beta (含三档弱等级) / MCTS 没有常驻实例 (每一步现场构造一个), 它们只碰
       **棋盘副本**, 与 agent 锁无关, 所以先处理掉 —— 这样下面那段"常驻实例"的临界区里
       不会同时持有 `mutex`(棋盘) 与 `m_agentMutex`, 两把锁的获取顺序只有一种。
    */
    if (abDepthOf(type) > 0 || type == AGENT_MCTS) {
        Chess probe;
        {
            /*
               只在这一个地方对真棋盘取副本, 所以要上锁: 调用方 (GUI worker) 读自检时,
               AI 工作线程可能正在走子 (其它 agent 的报告只碰自己的棋盘/网络)。
            */
            QMutexLocker locker(&mutex);
            probe = chess;
        }
        if (type != AGENT_MCTS) {
            /* 深度按**类型**取 (三档弱等级各自一份报告), 不写死 AB_DEPTH */
            ABAgent ab(probe, abDepthOf(type));
            return ab.selfCheckReport();
        }
        MCTS mcts(probe, 1.414f);
        return mcts.selfCheckReport();
    }

    /* 常驻实例: 与决策/训练串行 (理由见函数头那段"只读不等于可以并发") */
    std::lock_guard<std::mutex> agentLock(m_agentMutex);
    switch (type) {
    case AGENT_DQNMCTS:
        if (m_sfDQNMCTS != nullptr) {
            return m_sfDQNMCTS->selfCheckReport();
        }
        break;
    case AGENT_PG:
        if (m_sfPG != nullptr) {
            return m_sfPG->selfCheckReport();
        }
        break;
    case AGENT_DQN:
        if (m_sfDQN != nullptr) {
            return m_sfDQN->selfCheckReport();
        }
        break;
    case AGENT_PPOMCTS:
        if (m_sfPPOMCTS != nullptr) {
            return m_sfPPOMCTS->selfCheckReport();
        }
        break;
    case AGENT_EVAB:
        if (m_sfEVAB != nullptr) {
            return m_sfEVAB->selfCheckReport();
        }
        break;
    case AGENT_SACAZ:
        if (m_sfSACAZ != nullptr) {
            return m_sfSACAZ->selfCheckReport();
        }
        break;
    case AGENT_SACAZ_MOE:
        if (m_sfSACAZMoe != nullptr) {
            return m_sfSACAZMoe->selfCheckReport();
        }
        break;
    /* [2026-09 独立类] MoE+MLP 那一支有自己的实例与自检报告 (不能落到上面那一支去) */
    case AGENT_SACAZ_MOE_MLP:
        if (m_sfSACAZMoeMlp != nullptr) {
            return m_sfSACAZMoeMlp->selfCheckReport();
        }
        break;
    case AGENT_SACAZ_OLD:
        if (m_sfSACAZOld != nullptr) {
            return m_sfSACAZOld->selfCheckReport();
        }
        break;
    case AGENT_SACAZ_OLD_MOE:
        if (m_sfSACAZOldMoe != nullptr) {
            return m_sfSACAZOldMoe->selfCheckReport();
        }
        break;
    case AGENT_DQNAB:
        if (m_sfDQNAB != nullptr) {
            return m_sfDQNAB->selfCheckReport();
        }
        break;
    /*
       [2026-09 dev-dqnmcts-moetb] DQN+MCTS (MoE+TB 专家) 的只读自检: 它报的是骨干指纹
       (专家数/topK/注意力头请求数 vs 实用数)、表示口径、搜索的 PUCT 符号、学习读数
       与终局通道 —— 全是**只能靠读数发现**的静默失效, 所以面板上必须有它一格。
       没有实例时返回空串 (与上面各支一致: "没有实例"与"报告为空"在界面上由
       getAgentWeightStatus 那一行区分)。
    */
    case AGENT_DQNMCTS_MOE:
        if (m_sfDQNMCTSMOE != nullptr) {
            return m_sfDQNMCTSMOE->selfCheckReport();
        }
        break;
    case AGENT_PPOMCTS_MLP:
        if (m_sfPPOMCTSMLP != nullptr) {
            return m_sfPPOMCTSMLP->selfCheckReport();
        }
        break;
    /* AGENT_ALPHABETA / AGENT_MCTS 在函数开头就返回了 (它们不碰常驻 agent) */
    default:
        break;
    }
    return std::string();
}

/*
 * ================================================================
 *  getMoeLoad —— 稀疏 MoE 的专家负载快照 (界面 MoeLoadView 的数据源)
 * ================================================================
 *  与 `getAgentSelfCheck` **同一把锁、同一条约束**: 常驻实例的内部计数器要在
 *  `m_agentMutex` 上读, 所以 GUI 线程不能直接调它 —— 界面侧复用自检那个 worker。
 *
 *  哪些 agent 有稀疏 MoE 层 (本工程目前三个):
 *    * AGENT_SACAZ_MOE      -> SACAZMoETbAgent    (TB 专家,  E=4 top-1)
 *    * AGENT_SACAZ_MOE_MLP  -> SACAZMoEMlpAgent   (MLP 专家, E=8 top-2)
 *    * AGENT_DQNMCTS_MOE    -> DQNMCTSMOETbAgent  (TB 专家,  E=4 top-1)
 *  前两个的**训练侧/推理侧拆分**已经接线 (`moeUsageSplit`: 训练侧由 learnBatch 在批
 *  边界累计); 第三个目前只有生命周期合计 (`moeUsage`), 所以 `splitReady=false` ——
 *  界面会明说这一点, 而不是拿合计冒充训练侧。
 *
 *  注意 `AGENT_SACAZ_OLD_MOE` (59e5233 还原版) **故意不报**: 那一支的存在意义就是
 *  复现旧行为, 它的路由读数属于"历史口径", 混进这张均衡视图只会误导。
 */
bool ChessBoard::getMoeLoad(MoeLoadSnapshot &out) const
{
    return getMoeLoad(m_agentType, out);
}

bool ChessBoard::getMoeLoad(AgentType type, MoeLoadSnapshot &out) const
{
    out = MoeLoadSnapshot();
    /* 显示名一律先填上 (见 MoeLoadSnapshot::agentName 的说明): 控件的"不适用"文案
       要靠它说清"你选的是哪一个 agent", 否则用户分不清"选错了"和"功能坏了" */
    out.agentName = agentDisplayName(type).toStdString();

    /*
       [2026-10] 实时探针的登记也在这条路径上 (见 m_liveMoeProbe 的说明)。
       **先无条件清空**: 下面每个 case 只在真拿到探针时才登记 —— 于是"换成没有 MoE 的
       agent 之后呼吸灯还留着上一个 agent 的读数"这件事不可能发生 (那种残留正是本工程
       反复栽的"静默错读数")。
    */
    m_liveMoeProbe.store(nullptr, std::memory_order_release);

    /* 与 getAgentSelfCheck 一样: 先处理掉不碰常驻 agent 的类型, 让临界区里只有一把锁 */
    if (abDepthOf(type) > 0 || type == AGENT_MCTS) {
        return false;
    }

    std::lock_guard<std::mutex> agentLock(m_agentMutex);
    switch (type) {
    case AGENT_SACAZ_MOE:
        /* 先记下"这个类型有已接线的读数" —— 实例可能是懒建的 (还没走过一步), 那时
           applicable 仍为 false, 但 capable 为 true, 界面据此报"实例还没建出来" */
        out.capable = true;
        if (m_sfSACAZMoe != nullptr) {
            out.experts = m_sfSACAZMoe->moeExpertCount();
            out.topK = m_sfSACAZMoe->moeTopK();
            m_sfSACAZMoe->moeUsageSplit(out.train, out.infer);
            m_sfSACAZMoe->moeUsage(out.total);
            out.splitReady = true;
            out.applicable = (out.experts > 0);
            m_liveMoeProbe.store(m_sfSACAZMoe->moeRouteProbe(), std::memory_order_release);
        }
        break;
    case AGENT_SACAZ_MOE_MLP:
        out.capable = true;
        if (m_sfSACAZMoeMlp != nullptr) {
            out.experts = m_sfSACAZMoeMlp->moeExpertCount();
            out.topK = m_sfSACAZMoeMlp->moeTopK();
            m_sfSACAZMoeMlp->moeUsageSplit(out.train, out.infer);
            m_sfSACAZMoeMlp->moeUsage(out.total);
            out.splitReady = true;
            out.applicable = (out.experts > 0);
            m_liveMoeProbe.store(m_sfSACAZMoeMlp->moeRouteProbe(), std::memory_order_release);
        }
        break;
    case AGENT_DQNMCTS_MOE:
        out.capable = true;
        if (m_sfDQNMCTSMOE != nullptr) {
            out.experts = m_sfDQNMCTSMOE->moeExpertCount();
            out.topK = m_sfDQNMCTSMOE->moeTopK();
            m_sfDQNMCTSMOE->moeUsage(out.total);
            /* 这一支还没接训练/推理拆分 -> 明说, 不拿合计冒充训练侧 */
            out.splitReady = false;
            out.applicable = (out.experts > 0);
            m_liveMoeProbe.store(m_sfDQNMCTSMOE->moeRouteProbe(), std::memory_order_release);
        }
        break;
    case AGENT_PPOMCTS:
        /*
           [2026-10] PPO+MCTS 两类骨干也**有**稀疏 MoE, 而且 `PPOMCTSAgent::moeUsage`
           一直是接好的 (它把 actor 与 critic 两个网络的逐专家计数相加)。第一版只覆盖
           了三个 agent, 结果是"选到 PPO 那一支就什么都不显示" —— 用户当场报了
           "负载均衡状态控件并没有显示"。这里补上。
           **[2026-10 第二轮]** 训练侧/推理侧的拆分也补齐了: `RL::PPO::finalizeMoeBatch()`
           在批边界累计训练侧 (`moeUsageSplit`), 于是界面上这一支与 SAC 那两支**同口径**
           —— 而这一列正是判断"均衡机制有没有生效"的唯一依据 (合计里混着数量上压倒性的
           搜索前向, 见 rl/ppo.h 的说明)。
        */
        out.capable = true;
        if (m_sfPPOMCTS != nullptr) {
            out.experts = m_sfPPOMCTS->moeExpertCount();
            out.topK = m_sfPPOMCTS->moeTopK();
            m_sfPPOMCTS->moeUsage(out.total);
            m_sfPPOMCTS->moeUsageSplit(out.train, out.infer);
            out.splitReady = !out.train.empty();
            out.applicable = (out.experts > 0);
            m_liveMoeProbe.store(m_sfPPOMCTS->moeRouteProbe(), std::memory_order_release);
        }
        break;
    case AGENT_PPOMCTS_MLP:
        out.capable = true;
        if (m_sfPPOMCTSMLP != nullptr) {
            out.experts = m_sfPPOMCTSMLP->moeExpertCount();
            out.topK = m_sfPPOMCTSMLP->moeTopK();
            m_sfPPOMCTSMLP->moeUsage(out.total);
            m_sfPPOMCTSMLP->moeUsageSplit(out.train, out.infer);
            out.splitReady = !out.train.empty();
            out.applicable = (out.experts > 0);
            m_liveMoeProbe.store(m_sfPPOMCTSMLP->moeRouteProbe(), std::memory_order_release);
        }
        break;
    default:
        break;
    }
    return out.applicable;
}

/*
 * ================================================================
 *  liveMoeRoute —— "此刻哪个专家在工作" (**不加锁**)
 * ================================================================
 *  读的是 MoE 层里的无锁探针, 不是 agent 的内部状态 —— 所以本函数可以在 AI 思考中
 *  每几十毫秒调一次 (那时 `m_agentMutex` 被整段决策占着, 加锁的路径会一直阻塞)。
 *  详细理由见 chessboard.h 的 MoeLiveRoute 与 rl/sparse_moe.hpp 的 MoERouteProbe。
 *
 *  返回 false 的三种情形 (界面据此不画呼吸灯, 而不是画一排 0):
 *    * 当前 agent 不是带稀疏 MoE 的那几个 (探针没登记);
 *    * 登记过但这一帧读到"正在写"(seqlock 奇数) —— 宁可少画一帧;
 *    * 从来没有前向过 (还没走过子 / 还没训练过)。
 */
bool ChessBoard::liveMoeRoute(MoeLiveRoute &out) const
{
    /*
       两个上限必须与探针的一致 (理由见 chessboard.h 的 MoeLiveRoute): 头文件不包含
       sparse_moe.hpp, 所以只能各写一份 —— 这一行就是"两份不许漂移"的钉子。
       漂移的后果是**静默截断** (呼吸灯认不出某个专家), 所以宁可编译不过。
    */
    static_assert(MoeLiveRoute::kMaxExperts == RL::MoERouteProbe::kMaxExperts,
                  "MoeLiveRoute::kMaxExperts 与 RL::MoERouteProbe::kMaxExperts 不一致");
    static_assert(MoeLiveRoute::kMaxPicked == RL::MoERouteProbe::kMaxPicked,
                  "MoeLiveRoute::kMaxPicked 与 RL::MoERouteProbe::kMaxPicked 不一致");

    out = MoeLiveRoute();
    const RL::MoERouteProbe *probe = m_liveMoeProbe.load(std::memory_order_acquire);
    if (probe == nullptr) {
        return false;
    }

    RL::MoERouteProbe::Snapshot snap;
    if (!probe->read(snap)) {
        return false;
    }
    out.serial = snap.serial;
    out.experts = snap.experts;
    out.pickedCount = snap.picked;
    out.dense = snap.dense;
    for (int k = 0; k < MoeLiveRoute::kMaxPicked; k++) {
        out.picked[k] = snap.idx[k];
        out.pickedW[k] = snap.w[k];
    }

    RL::MoERouteProbe::Heat heat;
    probe->readHeat(heat);
    out.heatTotal = heat.total;
    out.ageSec = heat.ageSec;
    for (int i = 0; i < MoeLiveRoute::kMaxExperts; i++) {
        out.heat[i] = heat.v[i];
    }
    /* active 的判据用**热度的总量**而不是"读到了快照": 快照只是"有过前向", 而灯要表达的是
       "**现在**在干活" —— 停止计算 ~3τ 之后热度衰减到 5% 以下, 灯自己灭。 */
    out.active = (heat.total > 0.0f);
    out.available = true;
    return true;
}

/*
 * 权重文件在磁盘上的状态 (自检面板顶端那两行)。
 *
 * 判据只有一个: weightFilesOf() 给出的那些文件**在不在**、多大。文件名单一来源的
 * 理由见 weightFilesOf 的注释 (PPO+MCTS 曾经因为三份名字漂移而从来没被载入过)。
 * 这里刻意连"没找到"也报出来 —— "权重没载进来"的默认表现就是**静默地**从随机
 * 初始化开始跑, 面板上必须能看见。
 */
std::string ChessBoard::getAgentWeightStatus(AgentType type) const
{
    const std::vector<std::string> files = weightFilesOf(type);
    std::string out;
    if (files.empty()) {
        return out;      /* 纯搜索 agent: 没有权重, 不占版面 */
    }
    char buf[512];
    /*
       "启动加载有没有真的载上它"与"文件在不在"是两件事:
       文件在、但 s_weightPaths 里没有这一项, 说明**扫描的名字与实际写出的名字
       不一致** —— 那正是 PPO+MCTS 那个静默失效的形状, 所以两件事都报。
    */
    const bool loaded = (s_weightPaths.find(type) != s_weightPaths.end());
    std::snprintf(buf, sizeof(buf), "权重文件: %s\n",
                  loaded ? "启动时已载入 (扫描命中)" : "启动时**未**载入 (扫描没命中)");
    out += buf;
    for (const std::string &f : files) {
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        if (!in.good()) {
            std::snprintf(buf, sizeof(buf), "  %s : 不存在\n", f.c_str());
        } else {
            const long long bytes = (long long)in.tellg();
            std::snprintf(buf, sizeof(buf), "  %s : %lld KB\n", f.c_str(), bytes / 1024);
        }
        out += buf;
    }
    if (!loaded) {
        out += "  -> 扫描没命中时, 这个 agent 跑的是随机初始化的网络"
               " (名字对不上就是静默失效, 见 weightFilesOf 的注释)\n";
    }
    return out;
}

QString ChessBoard::stagePrefix() const
{
    if (m_matchRunning.load()) {
        const int g = m_matchGameNo.load();
        const int tot = m_matchGames.load();
        const int ply = m_selfPlayMoveNo.load();
        if (g > 0) {
            return QStringLiteral("对弈 %1/%2 局 · 第 %3 手 · ").arg(g).arg(tot).arg(ply);
        }
        return QStringLiteral("对弈 · ");
    }
    if (m_selfPlaying.load()) {
        const int no = m_selfPlayMoveNo.load();
        if (no > 0) {
            return QStringLiteral("自对弈 第 %1 步 · ").arg(no);
        }
        return QStringLiteral("自对弈 · ");
    }
    return QString();
}

void ChessBoard::emitStage(const QString &stage)
{
    /*
       [P1] "问对手一手"时, 状态条上必须能看出这行字是**对手**的阶段:
       那一手走的是对手的决策代码 (它自己会 emit 自己的阶段文字, 例如"Alpha-Beta 深度 1"),
       而此刻轮到的是**学习方** —— 不加这个前缀, 用户看到的就是"轮到我走, 屏幕上却写着
       Alpha-Beta 深度 1"这种解释不通的读数 (一次决策最多闪一下, 但那一下会让人以为选错了 agent)。
    */
    const QString who = (m_opponentQueryDepth > 0) ? QStringLiteral("对手查询 · ") : QString();
    emit aiThinkingStage(stagePrefix() + who + stage);
}

/* ================================================================
 *  aiThink - 使用当前选中的 agent 类型决策
 *
 *  这里是"决策 -> 合法性闸门"两步里的第一步: 那个大 switch 现在在 aiThinkRaw 里,
 *  结果统一过一遍 legalStepOrFallback (见它的注释)。
 * ================================================================ */
Step ChessBoard::aiThink(int color)
{
    /*
       ---- 人机对战: AI 固定是"冻结的对手" (P0-b) ----
       为什么必须在这里填角色: 在它之前, 人机这条路**完全绕过**了对弈模式 ——
       用户把模式设成"评估对局 / 只对弈不学习", 然后跟 AI 下棋, AI 照常每手训练,
       界面上一个字都没说 (一个控件两种语义)。
       语义定成"人 = 学习者(A), AI = 冻结的对手(B)": 因为人机对战里**人的棋力不会被
       这个程序改变**, 所以"待评估的那一方"只能是人。于是:
         * 训练模式   : 双方都能学 ⇒ AI 照旧 (与改动前一致);
         * 评估模式   : 只让人学 —— 人没有可训练参数 ⇒ AI 不学 (等价于"只对弈");
         * 只对弈模式 : AI 不学。
       AI 执黑是 process() 的既有约定 (见那里的 aiThink(Stone::COLOR_BLACK))。
       ⚠ 这里**委托**给 humanTurnAiMoveForTest, 一份实现两处用: 否则测试钩子与生产
         路径会各写一遍, 而"两处各写一遍"正是本工程反复栽跟头的地方 (SAC 掩码漏改那次)。
    */
    return humanTurnAiMoveForTest(color);
}

/*
 * humanTurnAiMoveForTest - 测试钩子: 走一次人机对战的 AI 决策 (P0-b)
 *
 * 与 process() 里那一步逐行相同 (先填角色、再 aiThinkRaw), 见 chessboard.h 的说明。
 * 存在的唯一理由是"人机路径受不受对弈模式约束"必须能被测试钉住 —— 而 aiThink 是私有。
 */
Step ChessBoard::humanTurnAiMoveForTest(int color)
{
    const double t0 = dbgNowMs();
    dbgLog(QStringLiteral("aiThink 开始 (color=%1, agent=%2)")
               .arg(color).arg(agentDisplayName(m_agentType)));
    setSideRole(sideRoleForHumanGame());
    const Step step = aiThinkRaw(color);
    /*
       ---- [2026-10 用户口径] 人机对战里也要能"顺便克隆" ----
       用户口径要的就是"与人类棋手对弈时照常做行为克隆" (老师来自下拉框, 与人无关)。
       时机: **决策之后**、落子之前 —— 此刻 `chess` 还是"AI 要走的那一步"的原局面,
       正是那条样本的定义 (学生=AI 的局面 + 下拉框那一档 AB 在该局面的着法)。
       为什么在 humanTurnAiMoveForTest 而不是 aiThinkRaw 里: 这个函数是**两份实现共用**
       的那一份 (aiThink 与测试钩子都委托给它), 放这里人机路径不会漏、测试也覆盖得到。
       三个闸门 (老师>0 / 训练模式 / 这一支能做学生) 都在 bcSampleForHumanTurn 里。
    */
    bcSampleForHumanTurn(m_agentType, color);
    const double waited = dbgNowMs() - t0;
    dbgWait(QStringLiteral("aiThink 结束 (valid=%1)").arg((int)step.valid), waited);
    return legalStepOrFallback(color, step, agentDisplayName(m_agentType));
}

/* ================================================================
 *  notifyHumanGameEnd - 把"人机这一局的结果"交给当前 agent
 * ================================================================
 *
 * 洞的形态、为什么必须有它、以及闸门口径见 chessboard.h 的同名声明 (那里是唯一的
 * 说明处)。这里是三件事的实现要点:
 *
 *  1. **闸门只有一处**: `updateEnabledForSide(sideRoleForHumanGame())` —— 与两条在线
 *     学习路径同一个判据。人机里 AI 固定是 SIDE_FROZEN, 所以"评估对局 / 只对弈不学习"
 *     下这条反馈不写 (那两个模式承诺"权重也不变", 而挂终局会立刻触发一次 learnBatch)。
 *  2. **锁**: 要改 agent 的回放池 (还可能做一次 learnBatch), 与"决策 / 权重保存 /
 *     后台训练同步回主 agent"共用 `m_agentMutex`。调用方不持有棋盘 mutex, 锁序与别处
 *     一致。对局线程 / GUI 线程都会调它 (三个终局点分别在 process() 与人落子的那条路)。
 *  3. **agent 决定接不接**: 三支 SAC+AZ 与 DQN+MCTS (MoE+TB 专家) 会接 (把终局挂到最后
 *     一条真实决策样本上, 幂等); 纯搜索与 rollout 系 agent 默认 no-op 返回 false。
 *     **返回 false 必须打日志** ——
 *     否则这条通道会静默地什么都不做, 而"没生效"与"没输过"在读数上长得一样。
 */
/*
 * ================================================================
 *  humanEndChannel - 人机终局反馈: 这个 agent 归哪一类 (2026-10, 见头文件)
 * ================================================================
 * 写成**显式名单**而不是"默认归某一类": 新增一个 agent 类型时, 默认归类会让它静默地
 * 落到"无需反馈"那一档 —— 而"这一支到底学不学、有没有可挂的样本"是必须逐支想清楚的事
 * (这正是 2026-10 那次"PPO 被说成纯搜索"的根因: 它落在了一个**凑合**的默认分支上)。
 * 名单里每一支的机制依据:
 *   PureSearch        : AgentBase 没有可训练参数 (AB 各档按深度, MCTS 无权重);
 *   RealSample        : 实现了 notifyGameResult 且保留 decision 样本 (SAC 三支 + DQN+MCTS MoE);
 *   Rollout           : 在学, 但样本是"从某局面滚出去的一串" —— PPO 两支 / DQN / DQN+AB /
 *                       EVAB / SAC 还原版两支 (后两支同样没有 decision 样本这条通道)。
 */
ChessBoard::HumanEndChannel ChessBoard::humanEndChannel(AgentType type)
{
    switch (type) {
    case AGENT_ALPHABETA:
    case AGENT_AB_L1:
    case AGENT_AB_L2:
    case AGENT_AB_L3:
    case AGENT_MCTS:
        return HUMAN_END_PURE_SEARCH;
    case AGENT_SACAZ:
    case AGENT_SACAZ_MOE:
    case AGENT_SACAZ_MOE_MLP:
    case AGENT_DQNMCTS_MOE:
        return HUMAN_END_REAL_SAMPLE;
    /* 显式列出而不是 default: 见上面的理由 (新增一支时编译器会提醒这里要想一次) */
    case AGENT_PG:
    case AGENT_DQN:
    case AGENT_PPOMCTS:
    case AGENT_PPOMCTS_MLP:
    case AGENT_DQNMCTS:
    case AGENT_EVAB:
    case AGENT_DQNAB:
    case AGENT_SACAZ_OLD:
    case AGENT_SACAZ_OLD_MOE:
        return HUMAN_END_ROLLOUT;
    }
    return HUMAN_END_ROLLOUT;
}

bool ChessBoard::notifyHumanGameEnd(int result)
{
    if (result == Chess::RESULT_ONGOING) {
        return false;      /* 还有棋可走: 三个调用点都不会传它, 这是兜底 */
    }
    /*
       ---- 人机这一局结束 ⇒ 解除"对弈期间权重不落盘" (2026-10) ----
       放在**所有提前 return 之前**: 下面有好几条"本模式下不写学习回路"的提前返回,
       把解除写在后面就会漏掉它们 —— 而漏掉的表现是"这一局之后权重再也不落盘",
       直到退出为止 (不崩、不报错, 只是盘上的临时文件从此不再更新)。
       放在这里也保证"凡终局都解除": 人走的那一手将死 / AI 被将死 / 和棋三条路都经过
       本函数 (见 mousePressEvent 与 process 的调用点)。
    */
    setHumanGameInProgress(false);
    const char *resText = (result == Chess::RESULT_RED_WIN)   ? "红胜 (人赢了)"
                        : (result == Chess::RESULT_BLACK_WIN) ? "黑胜 (AI 赢了)"
                        : (result == Chess::RESULT_DRAW)      ? "和棋"
                                                              : "未知结果";
    if (!updateEnabledForSide(sideRoleForHumanGame())) {
        m_humanEndMissed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: 本模式 (%2) 下不写入学习回路 —— "
                              "评估/只对弈要的是权重也不变")
                   .arg(QString::fromUtf8(resText), matchModeName(m_matchMode.load()));
        return false;
    }

    /* AI 执黑是 process() 的既有约定; 终局反馈挂在**它**那一方最后那次决策上 */
    const int aiColor = Stone::COLOR_BLACK;
    /*
       ---- [2026-10] "没有这条通道" 必须**分成三类**说清楚, 不能一句话盖过去 ----
       原来的代码把纯搜索 (AB / MCTS) 与 rollout 系 (PPO / DQN / EVAB) 合成一类, 日志
       于是打出 "PPO+MCTS 没有终局通道 (纯搜索 / 终局值随 rollout 进池的那几支), 无需反馈"。
       用户当场指出来: **PPO+MCTS 不是纯搜索, 它在学**。那句话有三个问题:

         1. **事实错**: PPO 有可训练参数, 而且每手都在学 (走子前 `exploreAndTrain` ->
            `learnSelfPlay`; 自对弈那条路还会 `commitEpisode` 进池)。把它与 AB/MCTS
            并列会让人以为人机对弈这一局白下了;
         2. **机制说反了一半**: rollout 的终局值确实进得去 —— 但只在**它自己的 rollout
            走到终局**时 (agentrollout 在 `done` 时把 r 换成走子方视角的 ±1,
            `ppomcts_agent.cpp` 的 `terminalReward`)。而**结束这一局的那一手是"人"走的**,
            AI 最后那次决策样本的终局值只是 `-V(s')` 自举 (同文件
            `exploreAndTrain` 里"自举: V(s_end+1) …"那一段), 真实胜负**不会**回填到它身上;
         3. 于是"无需反馈"是**结论性措辞**, 而正确的说法是"这条通道对它们**不适用**,
            因为池里没有可挂的"真实决策样本" —— 它们的学习发生在自对弈/rollout 里。

       所以判据换成"**这个 agent 到底学不学**" —— 现在由 `humanEndChannel(AgentType)`
       (纯函数, 见本文件上面的实现与 chessboard.h 的声明) 一次给出三档之一:
         * 不学 (AB 各档 / MCTS) -> 保持原话;
         * 在学但样本来自 rollout -> 单独一条日志, 并如实说明"这一局的胜负只经由
           `-V(s')` 间接进去".
       ⚠ 与"DQMCTS_MOE 也有这条通道"同一类教训: 把不同机制压成一句话, 只会让排查走错方向。
    */
    const bool channelClass = (humanEndChannel(m_agentType) == HUMAN_END_REAL_SAMPLE);
    /* 纯搜索那一档 (没有可训练参数) 由**类型**决定 —— 与实例建没建出来无关 */
    const bool pureSearchClass = (humanEndChannel(m_agentType) == HUMAN_END_PURE_SEARCH);
    bool instanceExists = false;

    bool taken = false;
    {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        AgentBase *agent = nullptr;
        switch (m_agentType) {
        /*
           只有"有真实决策样本"这一类的四支会接 (见 humanEndChannel): 它们实现了
           `notifyGameResult`, 池里也真的存了那种样本。其余类型的默认实现返回 false
           已经表达了"接不住" —— 那一句日志由 humanEndChannel 分类决定怎么写。
        */
        case AGENT_SACAZ:         agent = m_sfSACAZ;         break;
        case AGENT_SACAZ_MOE:     agent = m_sfSACAZMoe;      break;
        case AGENT_SACAZ_MOE_MLP: agent = m_sfSACAZMoeMlp;   break;
        /* DQN+MCTS (MoE+TB 专家): 独立类, 池里的 decision 样本由终局回填 (幂等) */
        case AGENT_DQNMCTS_MOE:   agent = m_sfDQNMCTSMOE;    break;
        default:                                             break;
        }
        instanceExists = (agent != nullptr);
        if (agent != nullptr) {
            taken = agent->notifyGameResult(result, aiColor);
        }
    }

    if (taken) {
        m_humanEndFed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: 已交给 %2 —— 它最后那一步的决策样本"
                              "按此改写 (done + 终局奖励), 并已更新一次")
                   .arg(QString::fromUtf8(resText), agentDisplayName(m_agentType));
    } else if (pureSearchClass) {
        m_humanEndMissed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: %2 是纯搜索 agent (没有可训练参数), "
                              "这一局的胜负对它没有意义 —— 无需反馈")
                   .arg(QString::fromUtf8(resText), agentDisplayName(m_agentType));
    } else if (!channelClass) {
        /*
           ---- [2026-10] rollout 系 (PPO / DQN / DQN+AB / EVAB) 单独一条 ----
           它们**在学**, 只是"学习"发生在自对弈/rollout 里: 池里的样本是它自己从某个
           局面滚出去的那一串, 而**结束这一局的那一手是"人"走的** ⇒ 它最后那次真实决策
           的终局值只能由 `-V(s')` 自举 (见 ppomcts_agent.cpp 的自举那一段), 真实胜负
           不会回填。
           这与 SAC 那三支的差别是**有没有存"真实决策样本"可挂**, 不是"学不学"。
        */
        m_humanEndMissed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: %2 **在学** (每一手都在学), 但它池里没有"
                              "\"真实决策样本\"可挂 —— 这一局的胜负不会回填到它最后那一步, "
                              "只经自举 -V(s') 与后续 rollout/自对弈间接进入学习回路 "
                              "(SAC 那几支存了决策样本, 所以它们能回填; 这是两支口径的差别)")
                   .arg(QString::fromUtf8(resText), agentDisplayName(m_agentType));
    } else if (!instanceExists) {
        m_humanEndMissed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: %2 还没有实例 (这一局它一步没走过), 无需反馈")
                   .arg(QString::fromUtf8(resText), agentDisplayName(m_agentType));
    } else {
        m_humanEndMissed.fetch_add(1);
        qInfo().noquote()
            << QStringLiteral("[human] 终局 %1: %2 没有可挂的真实决策样本 "
                              "(learnFromSearch 关着 / 这局还没走过一手) —— "
                              "这一局的输赢没有进学习回路")
                   .arg(QString::fromUtf8(resText), agentDisplayName(m_agentType));
    }
    return taken;
}

/* ================================================================
 *  legalStepOrFallback - 决策输出的合法性闸门
 *
 *  为什么要有它: agent 的搜索在若干局面下会返回**默认构造**的 Step
 *  (valid=false, id=0, pos=(0,0)), 而调用方一律把 valid=false 读成"这一步真无棋
 *  可走" -> 直接判走棋方负。用户 2026-09 的报障就是这么来的:
 *      [arena] agent(0) 返回无效走法 (第 6 局第 55 手): valid=0 id=0 pos=(0,0)->(0,0),
 *              仍有 1 个合法走法, 已兜底
 *  根因已修在源头 (ABAgent 在"每一步都输"时不再丢掉候选; MCTS 系在"根有合法走法
 *  但一个孩子都没展开"时不再返回空 Step), 但这类错误的**形状**会在每个 agent、
 *  每个新分支里重复出现, 而后果 (无中生有地判负) 比"走一步不理想的棋"严重得多。
 *  所以在这里再加一道统一闸门:
 *    * 无效走法 + 棋盘上还有合法走法 -> 取第一个合法走法兜底, 并打一行日志,
 *      **带上是谁返回的** (原来那条日志只报 agent 编号, 排查时要回去数枚举);
 *    * 真的无棋可走 -> 原样返回无效 Step, 由调用方判负 (这是唯一正确的语义)。
 *
 *  时机刻意选在"决策之后、落子之前", 所以它不能改棋盘: 只做一次 sample() (纯读),
 *  用完立刻把 Step 还回对象池。
 * ================================================================ */
Step ChessBoard::legalStepOrFallback(int color, const Step &step, const QString &who)
{
    if (step.valid) {
        return step;
    }
    dbgLog(QStringLiteral("legalStepOrFallback: 收到无效走法, 正在等棋盘锁 (who=%1)").arg(who));
    const double tw = dbgNowMs();
    std::vector<Step *> legal;
    {
        QMutexLocker locker(&mutex);
        dbgWait(QStringLiteral("legalStepOrFallback: 拿到棋盘锁"), dbgNowMs() - tw);
        chess.sample(color, legal);
    }
    Step fallback;
    if (!legal.empty()) {
        qWarning().noquote()
            << QStringLiteral("[gate] %1 返回无效走法 (id=%2 pos=(%3,%4)->(%5,%6)), "
                              "棋盘上仍有 %7 个合法走法, 已用第一手兜底")
                   .arg(who)
                   .arg(step.id)
                   .arg(step.pos.x).arg(step.pos.y)
                   .arg(step.nextPos.x).arg(step.nextPos.y)
                   .arg((int)legal.size());
        fallback = *legal[0];
    }
    Steps::instance().put(legal);
    return fallback;
}

Step ChessBoard::aiThinkRaw(int color)
{
    /*
       Copy current game state to env so agents can mutate env freely
       during search (moveForward/moveBack) without touching the main board.

       ================================================================
        ---- env 归 `m_agentMutex` 管, 不是 `mutex` (2026-09, 崩溃修复) ----
       ================================================================
       `env` 是**所有 agent 共用的那张试走棋盘** (每个 agent 构造时都拿了它的引用),
       而这一行是**写**它: `history` 会被整份替换掉。原来它写在锁外, 于是
       "对弈线程在搜 (env.history 一路 push_back)" 与 "另一条线程在读 env" 可以同时发生
       —— 后者包括: 界面自检 worker 的 `Chess probe(agent->chess)` (走 getAgentSelfCheck,
       是**加锁**的)、AI 工作线程的决策、以及另一场对弈。
       加锁的读者挡不住不加锁的写者: ASan 实测抓到的是
         `Chess::Chess(const Chess&)` (chess.cpp:526 的 `history(other.history)`)
         **堆越界写** —— 拷贝构造按撕裂的 size/capacity 分配, 再按另一个数字搬元素,
         与 Windows 事件日志里的 0xC0000374 (堆损坏) 完全对应。
       所以规则统一成一句: **凡是碰 `env` 的地方都拿 `m_agentMutex`**
       (它同时也是"主 agent 的网络"那把锁, 见 chessboard.h 的成员注释)。

       锁的粒度: 这里只包住"复制"这一步; 紧接着的决策在各自的 case 里再拿同一把锁
       (RL 分支本来就拿了, AB/MCTS 这两支以前**没拿** —— 它们也在 env 上搜索, 同样补上)。
       中途不会有人插进来改 env: 所有 env 的写者都在这把锁上排队。
    */
    {
        /* [诊断] 这一行下面就是"等 m_agentMutex" —— 卡住时这里会报出等了多久 */
        const double tw = dbgNowMs();
        std::lock_guard<std::mutex> envLock(m_agentMutex);
        dbgWait(QStringLiteral("aiThinkRaw: 拿到 env 锁 (复制棋盘)"), dbgNowMs() - tw);
        env = chess;
    }

    switch (m_agentType) {
    case AGENT_ALPHABETA:
    case AGENT_AB_L1:
    case AGENT_AB_L2:
    case AGENT_AB_L3: {
        /*
           Agent 以前声明成函数内的 static, 于是它只在第一次调用时构造, 永远绑定
           在"当时那个 env" 上 —— 一旦出现第二个 ChessBoard (或 env 先被销毁),
           就是悬垂引用。ABAgent 本身只是一个引用 + 一个深度整数, 每步新建的代价
           可以忽略。

           锁: Alpha-Beta 与 MCTS 没有"网络"可保护, 但它们**在这张共用的 env 上
           搜索** (moveForward/moveBack 会改 env.history), 所以同样要持锁 ——
           否则与上面那段注释里说的读者/写者撞在一起。

           深度按**类型**取 (三档弱等级 = 深度 1/2/3, AGENT_ALPHABETA = AB_DEPTH):
           状态条上印的必须是**实际用于搜索的那个深度** —— 以前三处都写死 AB_DEPTH,
           于是选了别的等级时状态条会报一个假数字。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        const int depth = abDepthOf(m_agentType);
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(depth));
        ABAgent abAI(env, depth);
        return abAI.getBestMove(color);
    }
    case AGENT_MCTS: {
        /* 蒙特卡洛树搜索 (同样在共用的 env 上搜索, 见 AB 分支的说明) */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        emitStage(QStringLiteral("① 搜索 / 决策 (MCTS %1 次模拟)").arg(MCTS_SIMS));
        MCTS mctsAI(env, 1.414f);
        return mctsAI.findBestMove(color, MCTS_SIMS);
    }
    case AGENT_PG: {
        /* Policy Gradient: 按概率分布采样 */
        /* 与后台训练线程互斥: 训练线程会在同一把锁内 loadPolicy() 改写网络权重 */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPG == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
            auto it = s_weightPaths.find(AGENT_PG);
            if (it != s_weightPaths.end())
                m_sfPG->loadPolicy(it->second);
        }
        preTrainThenDecide(m_sfPG, color);   /* 先探索环境+预训练, 再决策 */
        return m_sfPG->selectMove(color, false); /* false = greedy/deterministic */
    }
    case AGENT_DQN: {
        /* Deep Q-Network: argmax Q-value */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQN == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
            auto it = s_weightPaths.find(AGENT_DQN);
            if (it != s_weightPaths.end())
                m_sfDQN->loadModel(it->second);
        }
        preTrainThenDecide(m_sfDQN, color);
        return m_sfDQN->selectMove(color, false); /* false = no exploration */
    }
    case AGENT_PPOMCTS: {
        /* PPO + MCTS (AlphaZero风格): argmax */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            applyPpoBalance(*m_sfPPOMCTS);
            auto it = s_weightPaths.find(AGENT_PPOMCTS);
            if (it != s_weightPaths.end())
                m_sfPPOMCTS->loadModel(it->second);
        }
        preTrainThenDecide(m_sfPPOMCTS, color);
        return m_sfPPOMCTS->selectMove(color, PPO_SIMS, 0.0f);
    }
    case AGENT_PPOMCTS_MLP: {
        /* 同一个算法, 骨干 = 稀疏 MoE + MLP 专家 (E=8 top-2); 见 aiThinkRaw 的同一支 */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfPPOMCTSMLP == nullptr) {
            m_sfPPOMCTSMLP = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                              true, RL::PPO::Backbone::MlpExperts);
            applyPpoBalance(*m_sfPPOMCTSMLP);
            auto it = s_weightPaths.find(AGENT_PPOMCTS_MLP);
            if (it != s_weightPaths.end())
                m_sfPPOMCTSMLP->loadModel(it->second);
        }
        preTrainThenDecide(m_sfPPOMCTSMLP, color);
        return m_sfPPOMCTSMLP->selectMove(color, PPO_MLP_SIMS, 0.0f);
    }
    case AGENT_DQNMCTS: {
        /*
           training 必须是 false。以前传 true, 而 DQNMCTS 的 exploringRate 初值
           是 1.0 且只在 endOnlineEpisode()/learn() 里衰减 (GUI 从不调用),
           于是 selectMove 几乎必然在根节点挑一个**随机**子节点 —— 界面上的
           "DQN+MCTS" 实际上一直在随机走子。迭代数也从 6 提到 200。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
            auto it = s_weightPaths.find(AGENT_DQNMCTS);
            if (it != s_weightPaths.end())
                m_sfDQNMCTS->loadModel(it->second);
        }
        preTrainThenDecide(m_sfDQNMCTS, color);
        return m_sfDQNMCTS->selectMove(color, DQNMCTS_ITERATIONS, false);
    }
    case AGENT_EVAB: {
        /* EVAB: 学会评估的 Alpha-Beta (见 docs/agent_evab_design.md) */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfEVAB == nullptr) {
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
            auto it = s_weightPaths.find(AGENT_EVAB);
            if (it != s_weightPaths.end())
                m_sfEVAB->loadModel(it->second);
        }
        /* EVAB 的"探索"就是它的搜索; 这里额外把探索结果蒸馏回评估网络 */
        preTrainThenDecide(m_sfEVAB, color);
        return m_sfEVAB->getBestMove(color);
    }
    case AGENT_SACAZ: {
        /* SAC + MCTS + AlphaZero: 最大熵 critic 给 PUCT 搜索提供叶子估值 */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZ == nullptr) {
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
            auto it = s_weightPaths.find(AGENT_SACAZ);
            if (it != s_weightPaths.end()) {
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZ->loadModel(prefix);
            }
        }
        preTrainThenDecide(m_sfSACAZ, color);
        /* temp = 0: 取访问数最多的走法 (确定性)。
           走 finishDecisionSearch 而不是裸调 selectMove: 它同时负责
             (a) selectMove 里那次 learnFromSearch 的损失上报, 以及
             (b) **对弈模式的掩码** —— 评估/只对弈模式下把 learnFromSearch 关掉。
           实测 (test_match [2.7c]): 这条路径不受界面"探索+预训练"勾选框控制, 所以
           模式掩码只能在这里装。 */
        return finishDecisionSearch(m_sfSACAZ, [&] {
            return m_sfSACAZ->selectMove(color, SACAZ_SIMS, 0.0f);
        });
    }
    case AGENT_SACAZ_MOE: {
        /*
           SAC + MCTS + AlphaZero, 骨干 = 稀疏路由 MoE + TransformerBlock 专家
           (E=4, top-1)。刻意用很少的模拟次数: 一次模拟 ~10.9 ms, 16 次约 175 ms。
           注意这 175 ms **只是搜索**: 每步还要跑一次在线 learnBatch, 界面上实测
           约 2.3 s/手 (见 docs/agents_design.md 13.6)。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZMoe == nullptr) {
            /*
               正常路径下这里**不会**是 nullptr —— startupLoad() 已经预加载过它了
               (用户要求"程序启动时加载所有模型")。留着这一支是兜底: 权重文件缺失、
               或者以后有人又把预加载改掉时, 至少能构造出一个可用 agent, 而不是
               空指针崩掉。
            */
            m_sfSACAZMoe = createSACAZMoETbAgent(env, false, m_dynamicReward);
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE);
            if (it != s_weightPaths.end()) {
                /*
                   兜底路径同样要弹"请稍候": 这个变体的权重是 3 x 146 MB, 读一次
                   实测 6 秒。这些信号是队列投递到 GUI 线程的, 在工作线程 emit 是安全的。
                */
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE): 读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZMoe->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(m_sfSACAZMoe, color);
{
            /* 走 finishDecisionSearch: 上报那次 learnFromSearch 的损失 + 装对弈模式掩码 */
            return finishDecisionSearch(m_sfSACAZMoe, [&] {
                return m_sfSACAZMoe->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
            });
        }
    }
    /*
       [2026-09 独立类] **MoE + MLP 专家**那一支 (AGENT_SACAZ_MOE_MLP)。与上面那一支
       完全对称 —— 同一个算法、同一套口径, 只有骨干与类不同。它自己那些参数:
         * 模拟次数 SACAZ_MOE_MLP_SIMS (比 TB 大一档: 一次模拟 ~2.9 ms vs ~5.4 ms);
         * 权重是**共享口径的 4 个文件** (trunk + 三个头), 前缀由它自己的类给出
           (weights/sacaz_moe_mlp_shared_agent) —— 与 TB 那一支**不同前缀**。
    */
    case AGENT_SACAZ_MOE_MLP: {
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZMoeMlp == nullptr) {
            /* 兜底构造 (正常路径由 startupLoad() 预加载, 同上面那一支) */
            m_sfSACAZMoeMlp = createSACAZMoEMlpAgent(env, false, m_dynamicReward);
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE_MLP);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE-MLP): 读取权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actorhead";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZMoeMlp->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(m_sfSACAZMoeMlp, color);
        return finishDecisionSearch(m_sfSACAZMoeMlp, [&] {
            return m_sfSACAZMoeMlp->selectMove(color, SACAZ_MOE_MLP_SIMS, 0.0f);
        });
    }
    case AGENT_SACAZ_OLD: {
        /*
           SAC + MCTS + AlphaZero 的 59e5233 行为还原版: 同一份搜索/训练代码, 另一套口径
           (熵比 0.98 / alpha lr 1e-3 / critic 不钳位+纯 MSE / 叶子全量估值), 见
           src/sacazlegacyagent.h。模拟次数与 AGENT_SACAZ 相同 (256): 两者是同一骨干、
           同一表示, 给的预算不同就没法把差别归给口径。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZOld == nullptr) {
            m_sfSACAZOld = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD);
            auto it = s_weightPaths.find(AGENT_SACAZ_OLD);
            if (it != s_weightPaths.end()) {
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZOld->loadModel(prefix);
            }
        }
        preTrainThenDecide(m_sfSACAZOld, color);
        /* temp = 0: 取访问数最多的走法 (确定性), 与 AGENT_SACAZ 同一口径 */
{
            /*
               59e5233 还原版**没有** learnFromSearch 那条路径 (见
               createSACAZLegacyAgent 的注释), 所以这里用普通收尾钩子: 它只负责
               "决策 + 那次更新的损失上报", 不需要装掩码。
            */
            return finishDecision(m_sfSACAZOld, [&] {
                return m_sfSACAZOld->selectMove(color, SACAZ_SIMS, 0.0f);
            });
        }
    }
    case AGENT_SACAZ_OLD_MOE: {
        /*
           ---- 59e5233 行为还原版 + TB 专家骨干 ----
           与上面 AGENT_SACAZ_MOE 完全对称: 同一个类、同一套还原口径, 只换骨干。
           模拟次数也照那一档给 (SACAZ_MOE_SIMS = 16, 一次模拟 ~10.9 ms ≈ 175 ms/手):
           给 256 次的话一步就是 2.8 s, 而且与 AGENT_SACAZ_MOE 的对照就不再是
           "同预算、不同口径"了。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfSACAZOldMoe == nullptr) {
            /* 兜底: 正常路径下 startupLoad() 已经预加载过它 */
            m_sfSACAZOldMoe = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD_MOE);
            auto it = s_weightPaths.find(AGENT_SACAZ_OLD_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ-59e5233 (稀疏MoE): "
                                                "读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfSACAZOldMoe->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(m_sfSACAZOldMoe, color);
        /* temp = 0: 取访问数最多的走法 (确定性), 与两支 SACAZ 同一口径 */
        {
            /* 同上面两支: 还原版没有 learnFromSearch, 用普通收尾钩子即可 */
            return finishDecision(m_sfSACAZOldMoe, [&] {
                return m_sfSACAZOldMoe->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
            });
        }
    }
    case AGENT_DQNAB: {
        /*
           DQN+AB: AB 当 DQN 的 planning head。
           每步的搜索预算是 `DQNAB_NODES` (节点数, 不是模拟次数): TB 骨干实测
           ~3 ms/节点, 512 节点 ≈ 1.5 s/步, 能搜到 2~3 层; 同样预算换成 MLP 骨干
           只要几毫秒、能搜到 4~6 层 (同一套算法, 见 dqnabagent.h §5)。
           这里给 256 (约 0.8 s/步) —— 与 SACAZ_MOE 的 175 ms 同一量级, 但那 175 ms
           是"16 次 MCTS 模拟", 这里是"256 个节点的 alpha-beta", 信息量不同。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQNAB == nullptr) {
            /* 正常路径下 startupLoad() 已经预加载过它; 这里只是兜底 */
            m_sfDQNAB = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                             DQNABAgent::Backbone::SparseMoeTb);
            auto it = s_weightPaths.find(AGENT_DQNAB);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 DQN+AB: 读取主干 + 两个头的权重…"));
                std::string prefix = it->second;
                const std::string suffix = "_trunk";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfDQNAB->loadModel(prefix);
                emit busyFinished();
            }
        }
        m_sfDQNAB->nodeBudget = DQNAB_NODES;
        preTrainThenDecide(m_sfDQNAB, color);
        /* temp = 0: 取搜索值最大的那一手 (确定性) */
        return m_sfDQNAB->selectMove(color, 0.0f);
    }
    case AGENT_DQNMCTS_MOE: {
        /*
           ---- DQN+MCTS (稀疏MoE+TB专家), 独立类 DQNMCTSMOETbAgent ----
           与上面 AGENT_DQNAB 那一支同形 (懒构造 + 载权重 + 定预算 + 决策), 差别只在
           "预算"这个词的口径: DQNAB 是**节点数**, 本类是 MCTS 的**模拟次数**
           (DQNMCTS_MOE_SIMS = 120, 见那个常数的注释: 一次叶子估值 ~2.9 ms, 120 次约
           364 ms/步; 40 次时搜索会退化成"每个孩子各访问一次 + 按先验取第一手",
           实测两个预算的着法一致率 0%)。
           ⚠ `selectMove` 的第二个参数传 0 = "用成员 simulations", 所以上面那行赋值是
           **必需**的, 不是冗余: 不设它就跑构造函数的默认值 (与界面常数脱钩)。
           ⚠ 构造参数必须与 makeAgentInstance / loadAgentModel / 后台训练那几处**逐字一致**
           (隐层 64 / gamma 0.99 / lr 0.001 / eps 1.0 / c_puct 1.5, denseMoe 走默认 false),
           否则 save/load 的结构指纹对不上 (每一轮都载入失败)。
        */
        std::lock_guard<std::mutex> agentLock(m_agentMutex);
        if (m_sfDQNMCTSMOE == nullptr) {
            /* 正常路径下 startupLoad() 已经预加载过它; 这里只是兜底 */
            m_sfDQNMCTSMOE = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                                   1.0f, 1.5f);
            auto it = s_weightPaths.find(AGENT_DQNMCTS_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 DQN+MCTS (稀疏MoE+TB专家): "
                                                "读取主干 + Q 头的权重…"));
                std::string prefix = it->second;
                const std::string suffix = "_trunk";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                m_sfDQNMCTSMOE->loadModel(prefix);
                emit busyFinished();
            }
        }
        m_sfDQNMCTSMOE->simulations = DQNMCTS_MOE_SIMS;
        preTrainThenDecide(m_sfDQNMCTSMOE, color);
        /*
           走 finishDecisionSearch (与 SAC 两支同一个收尾钩子), 而不是直接 return:
             1. **对弈模式掩码**: 评估 / 只对弈模式下必须把"从自己的搜索学一次"
                (learnFromSearch) 在一手之内关掉 —— 那两条模式承诺"权重也不变",
                而每次真实决策都会存一条 decision 样本并按节拍更新一次;
             2. **损失上报**: 这一手真的学过就要上曲线 (按 getLearnSteps 有没有前进判断)。
           temp = 0: 取访问数最多的走法 (确定性), 与旧类 AGENT_DQNMCTS 同一收尾口径。
        */
        return finishDecisionSearch(m_sfDQNMCTSMOE, [&] {
            return m_sfDQNMCTSMOE->selectMove(color, 0, 0.0f);
        });
    }
    default: {
        /*
           兜底: 走到这里说明**这个 agent 类型没有自己的决策 case** (新增类型忘了接线),
           于是退回 Alpha-Beta 默认深度。以前这里连一句日志都没有 —— 表现是"选中的
           agent 像 Alpha-Beta"而界面上看不出原因。加一行 warning 让它当场可见
           (与 backgroundTrainLoop 里那条"该 agent 的后台训练尚未接入"同一个思路)。
        */
        qWarning() << "[aiThink] 该 agent 类型没有决策分支, 回退 Alpha-Beta:"
                   << agentDisplayName(m_agentType);
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAIDefault(env, AB_DEPTH);
        return abAIDefault.getBestMove(color);
    }
    }
}

/* ================================================================
 *  aiThinkForAgent - 使用指定agent类型在env副本上决策
 *
 *  用于 self play. 先将 chess 复制到 env, 然后在 env 上搜索.
 *  所有 agent 实例引用 env 而非 chess, 确保搜索过程中的
 *  moveForward/moveBack 不影响主棋盘状态.
 *
 *  注意: 在调用之前必须确保 chess 处于正确的当前局面
 * ================================================================ */
Step ChessBoard::aiThinkForAgent(int color, AgentType agentType)
{
    const Step step = aiThinkForAgentRaw(color, agentType);
    return legalStepOrFallback(color, step, agentDisplayName(agentType));
}

Step ChessBoard::aiThinkForAgentRaw(int color, AgentType agentType)
{
    /*
       Copy current game state to env so agents can mutate env freely
       during search without touching the main board.
       **这一步与下面的 AB/MCTS 分支都要拿 `m_agentMutex`** —— 理由见
       aiThinkRaw 顶部那段长注释 (env 是所有 agent 共用的试走棋盘, ASan 实测:
       不加锁的 `env = chess` 与加锁的读者相撞, 在 Chess 的拷贝构造里堆越界写)。

       ---- [P0-b/P1] 锁从"每个 case 各拿一次"提到**整段决策** ----
       原来这里是"锁内复制 env" + "每个 case 各自再拿一次同一把锁"。两处新增的调用者
       需要一段**不加锁、也不锁第二次**的决策体:
         * P1: 学习方的探索滚到"对手那一手"时要问真实对手 —— 问的时机在**学习方决策的
           锁内**, 而那时再进一次同一个 case 就是 std::mutex 的**自锁死锁** (非递归锁)。
       所以改成: 本函数拿一次锁, 整段决策都在锁内; 真正的 switch 在
       decideOnEnvRawLocked() 里, 它的契约就是"调用方已持有 m_agentMutex"。
       锁的粒度只是**变大** (原来是 env 复制之后有一个小窗口是不锁的), 没有任何调用者
       依赖那个窗口, 而"决策全程串行"本来就是这套代码的口径 (见 saveCurrentAgentModel)。
    */
    std::lock_guard<std::mutex> agentLock(m_agentMutex);
    const double tw = dbgNowMs();
    dbgWait(QStringLiteral("aiThinkForAgentRaw: 拿到 env 锁 (复制棋盘, agent=%1)")
                .arg(agentDisplayName(agentType)), dbgNowMs() - tw);
    env = chess;
    return decideOnEnvRawLocked(color, agentType);
}

/*
 * decideOnEnvRawLocked - "在 env 上按类型决策" (契约: **调用方必须已持有 m_agentMutex**)
 *
 * 为什么单独存在 (而不是并进 aiThinkForAgentRaw): 见上面那段 —— P1 的"问对手一手"
 * 发生在学习方决策的锁内, 那里**不能**再进一次这个 switch 的加锁版本。
 * 它**不动 env 的内容**: 调用方负责把局面摆好 (aiThinkForAgentRaw 是 `env = chess`;
 * 探索路径是"探索本来就在 env 上试走到了那一手")。
 */
Step ChessBoard::decideOnEnvRawLocked(int color, AgentType agentType)
{
    switch (agentType) {
    case AGENT_ALPHABETA:
    case AGENT_AB_L1:
    case AGENT_AB_L2:
    case AGENT_AB_L3: {
        const int depth = abDepthOf(agentType);     /* 同上: 深度按类型取, 不写死 */
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(depth));
        ABAgent abAIForAgent(env, depth);
        return abAIForAgent.getBestMove(color);
    }
    case AGENT_MCTS: {
        emitStage(QStringLiteral("① 搜索 / 决策 (MCTS %1 次模拟)").arg(MCTS_SIMS));
        MCTS mctsAI(env, 1.414f);
        return mctsAI.findBestMove(color, MCTS_SIMS);
    }
    case AGENT_PG: {
        /*
           ---- 这一手用哪个实例 (P0-b 收尾: 冻结快照) ----
           decisionInstance 只在"这一手是冻结的 B 方 + 本类型有开场快照"时换成快照实例,
           其余情况原样返回常驻实例 —— 判断只有一处 (frozenOpponentOverrideFor),
           11 个 case 都不自己写 if (那种写法必然漏掉某一支, 见 aiagent 的 SAC 掩码事故)。
        */
        PGEagent *ag = decisionInstance(m_sfPG, AGENT_PG);
        if (ag == nullptr) {
            m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
            ag = m_sfPG;
        }
        preTrainThenDecide(ag, color);
        return ag->selectMove(color, false);
    }
    case AGENT_DQN: {
        DQNAgent *ag = decisionInstance(m_sfDQN, AGENT_DQN);
        if (ag == nullptr) {
            m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
            ag = m_sfDQN;
        }
        preTrainThenDecide(ag, color);
        return ag->selectMove(color, false);
    }
    case AGENT_PPOMCTS: {
        PPOMCTSAgent *ag = decisionInstance(m_sfPPOMCTS, AGENT_PPOMCTS);
        if (ag == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            applyPpoBalance(*m_sfPPOMCTS);
            ag = m_sfPPOMCTS;
        }
        preTrainThenDecide(ag, color);
        return ag->selectMove(color, PPO_SIMS, 0.0f);
    }
    case AGENT_DQNMCTS: {
        DQNMCTSAgent *ag = decisionInstance(m_sfDQNMCTS, AGENT_DQNMCTS);
        if (ag == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
            ag = m_sfDQNMCTS;
        }
        preTrainThenDecide(ag, color);
        /* self-play 走贪心 (training=false), 否则恒为 1.0 的探索率会让它随机走子 */
        return ag->selectMove(color, DQNMCTS_ITERATIONS, false);
    }
    case AGENT_EVAB: {
        EVABAgent *ag = decisionInstance(m_sfEVAB, AGENT_EVAB);
        if (ag == nullptr) {
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
            ag = m_sfEVAB;
        }
        preTrainThenDecide(ag, color);
        return ag->getBestMove(color);
    }
    case AGENT_SACAZ: {
        SACAZAgent *ag = decisionInstance(m_sfSACAZ, AGENT_SACAZ);
        if (ag == nullptr) {
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
            ag = m_sfSACAZ;
        }
        preTrainThenDecide(ag, color);
        /* 走 finishDecisionSearch: 上报那次 learnFromSearch 的损失 + 装对弈模式掩码。
           ⚠ 这一支曾经漏改过: aiThinkRaw 与 aiThinkForAgentRaw 各有一份 SAC 分支,
             而**对弈走的是后者** —— 只改了前者的话, 评估模式下 SAC 照样每手更新,
             在读数上表现为"B 侧 20 次而不是 0 次", 极难归因 (见 [2.7d] 的注释)。 */
        return finishDecisionSearch(ag, [&] {
            return ag->selectMove(color, SACAZ_SIMS, 0.0f);
        });
    }
    case AGENT_SACAZ_MOE: {
        /* [2026-09 独立类] 决策实例的类型 = SACAZMoETbAgent (不再是 SACAZAgent) */
        SACAZMoETbAgent *ag = decisionInstance(m_sfSACAZMoe, AGENT_SACAZ_MOE);
        if (ag == nullptr) {
            m_sfSACAZMoe = createSACAZMoETbAgent(env, false, m_dynamicReward);
            ag = m_sfSACAZMoe;
            /*
               以前这里**只建对象、不载权重** —— 于是对弈里用到这个 agent 时跑的是
               随机初始化的网络 (界面上"选了它却像没训练过"), 而它自己在 aiThink
               路径里又会载权重, 两条路径行为不一致。
               现在正常路径都由 startupLoad() 预加载, 这一支只是兜底, 但仍然保持
               "建了对象就把权重载上"的行为, 免得两条兜底路径再分叉。
            */
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE): 读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(ag, color);
{
            /* 走 finishDecisionSearch: 上报那次 learnFromSearch 的损失 + 装对弈模式掩码 */
            return finishDecisionSearch(ag, [&] {
                return ag->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
            });
        }
    }
    /*
       [2026-09 独立类] **MoE + MLP 专家**那一支 (与上面那一支对称: 同一条兜底约定
       "建了对象就把权重载上", 只有骨干/类/前缀/模拟次数不同)。
    */
    case AGENT_SACAZ_MOE_MLP: {
        SACAZMoEMlpAgent *ag = decisionInstance(m_sfSACAZMoeMlp, AGENT_SACAZ_MOE_MLP);
        if (ag == nullptr) {
            m_sfSACAZMoeMlp = createSACAZMoEMlpAgent(env, false, m_dynamicReward);
            ag = m_sfSACAZMoeMlp;
            auto it = s_weightPaths.find(AGENT_SACAZ_MOE_MLP);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ (稀疏MoE-MLP): 读取权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actorhead";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(ag, color);
        return finishDecisionSearch(ag, [&] {
            return ag->selectMove(color, SACAZ_MOE_MLP_SIMS, 0.0f);
        });
    }
    case AGENT_SACAZ_OLD: {
        /*
           59e5233 行为还原版: 与上面 AGENT_SACAZ 那一支同一条兜底约定
           (建了对象就把权重载上), 但构造的是**独立类** SACAZLegacyAgent ——
           口径不同, 所以权重前缀也独立 (weights/sacaz_old_agent*)。
        */
        SACAZLegacyAgent *ag = decisionInstance(m_sfSACAZOld, AGENT_SACAZ_OLD);
        if (ag == nullptr) {
            m_sfSACAZOld = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD);
            ag = m_sfSACAZOld;
            auto it = s_weightPaths.find(AGENT_SACAZ_OLD);
            if (it != s_weightPaths.end()) {
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
            }
        }
        preTrainThenDecide(ag, color);
{
            /* 决策里可能用自己的搜索样本学了一次 (learnFromSearch), 那次损失也要上曲线 */
            /*
               59e5233 还原版**没有** learnFromSearch 那条路径 (见
               createSACAZLegacyAgent 的注释), 所以这里用普通收尾钩子: 它只负责
               "决策 + 那次更新的损失上报", 不需要装掩码。
            */
            return finishDecision(ag, [&] {
                return ag->selectMove(color, SACAZ_SIMS, 0.0f);
            });
        }
    }
    case AGENT_SACAZ_OLD_MOE: {
        /*
           59e5233 行为还原版 + TB 专家骨干 (与上面 AGENT_SACAZ_MOE 对称):
           同一条"建了对象就把权重载上"的兜底约定, 前缀独立
           (weights/sacaz_old_moe_agent*)。
        */
        SACAZLegacyAgent *ag = decisionInstance(m_sfSACAZOldMoe, AGENT_SACAZ_OLD_MOE);
        if (ag == nullptr) {
            m_sfSACAZOldMoe = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD_MOE);
            ag = m_sfSACAZOldMoe;
            auto it = s_weightPaths.find(AGENT_SACAZ_OLD_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 SAC+AZ-59e5233 (稀疏MoE): "
                                                "读取 3 个 146 MB 权重文件…"));
                std::string prefix = it->second;
                const std::string suffix = "_actor";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
                emit busyFinished();
            }
        }
        preTrainThenDecide(ag, color);
        {
            /* 同上面两支: 还原版没有 learnFromSearch, 用普通收尾钩子即可 */
            return finishDecision(ag, [&] {
                return ag->selectMove(color, SACAZ_MOE_SIMS, 0.0f);
            });
        }
    }
    case AGENT_DQNAB: {
        DQNABAgent *ag = decisionInstance(m_sfDQNAB, AGENT_DQNAB);
        if (ag == nullptr) {
            m_sfDQNAB = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                             DQNABAgent::Backbone::SparseMoeTb);
            ag = m_sfDQNAB;
            /* 与上面 SACAZ_MOE 同一条约定: 建了对象就把权重载上, 两条兜底路径不许分叉 */
            auto it = s_weightPaths.find(AGENT_DQNAB);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 DQN+AB: 读取主干 + 两个头的权重…"));
                std::string prefix = it->second;
                const std::string suffix = "_trunk";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
                emit busyFinished();
            }
        }
        ag->nodeBudget = DQNAB_NODES;
        preTrainThenDecide(ag, color);
        return ag->selectMove(color, 0.0f);
    }
    case AGENT_DQNMCTS_MOE: {
        /*
           ---- DQN+MCTS (稀疏MoE+TB专家), 独立类 DQNMCTSMOETbAgent ----
           与上面 AGENT_DQNAB 那一支同形。**这一支必须与 aiThinkRaw 里那一支同时存在**:
           两条决策路径各有一份 switch, 人机走 aiThinkRaw, 对弈 (含问对手一手) 走这里 ——
           以前 SAC 就漏改过其中一处 (见上面 AGENT_SACAZ 分支的注释: 表现是"评估模式下
           B 方仍然每手更新"), 那种漏洞在读数上极难归因。
           `simulations` 用 DQNMCTS_MOE_SIMS (120): 一次叶子估值 ~2.9 ms ⇒ 约 364 ms/步,
           而 40 次会让搜索退化成"每个孩子各访问一次 + 按先验取第一手"
           (见那个常数的注释与 bench 的一致率证据); `selectMove` 的第二参传 0 = 用该成员值。
        */
        DQNMCTSMOETbAgent *ag = decisionInstance(m_sfDQNMCTSMOE, AGENT_DQNMCTS_MOE);
        if (ag == nullptr) {
            m_sfDQNMCTSMOE = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                                   1.0f, 1.5f);
            ag = m_sfDQNMCTSMOE;
            /* 与上面 SACAZ_MOE / DQNAB 同一条约定: 建了对象就把权重载上, 两条兜底路径不许分叉 */
            auto it = s_weightPaths.find(AGENT_DQNMCTS_MOE);
            if (it != s_weightPaths.end()) {
                emit busyStarted(QStringLiteral("正在载入"),
                                 QStringLiteral("首次使用 DQN+MCTS (稀疏MoE+TB专家): "
                                                "读取主干 + Q 头的权重…"));
                std::string prefix = it->second;
                const std::string suffix = "_trunk";
                if (prefix.size() > suffix.size()
                    && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    prefix.erase(prefix.size() - suffix.size());
                }
                ag->loadModel(prefix);
                emit busyFinished();
            }
        }
        ag->simulations = DQNMCTS_MOE_SIMS;   /* 与 aiThinkRaw 同一预算 */
        preTrainThenDecide(ag, color);
        /* 走 finishDecisionSearch (对弈模式掩码 + 损失上报), 与 aiThinkRaw 那一支一致 */
        return finishDecisionSearch(ag, [&] {
            return ag->selectMove(color, 0, 0.0f);
        });
    }
    case AGENT_PPOMCTS_MLP: {
        /*
           PPO+MCTS+AlphaZero, 骨干 = 稀疏 MoE + **MLP 专家** (E=8 top-2): 与
           AGENT_PPOMCTS 共享同一份实现, 只有骨干这一项不同 (见 chessboard.h 的枚举注释)。
           模拟次数给 PPO_MLP_SIMS —— 远比 TB 那一支多, 因为一次模拟便宜 ~25x。
        */
        PPOMCTSAgent *ag = decisionInstance(m_sfPPOMCTSMLP, AGENT_PPOMCTS_MLP);
        if (ag == nullptr) {
            m_sfPPOMCTSMLP = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                              true, RL::PPO::Backbone::MlpExperts);
            applyPpoBalance(*m_sfPPOMCTSMLP);
            ag = m_sfPPOMCTSMLP;
            /* 与上面 DQNAB / SACAZ_MOE 同一条约定: 建了对象就把权重载上 */
            auto it = s_weightPaths.find(AGENT_PPOMCTS_MLP);
            if (it != s_weightPaths.end()) {
                ag->loadModel(it->second);
            }
        }
        preTrainThenDecide(ag, color);
        return ag->selectMove(color, PPO_MLP_SIMS, 0.0f);
    }
    default:
        /* 兜底: 同 aiThinkRaw 的 default (加一行 warning, 让它不是静默的降级) */
        qWarning() << "[aiThinkForAgent] 该 agent 类型没有决策分支, 回退 Alpha-Beta:"
                   << agentDisplayName(agentType);
        emitStage(QStringLiteral("① 搜索 / 决策 (Alpha-Beta 深度 %1)").arg(AB_DEPTH));
        ABAgent abAIForAgentDef(env, AB_DEPTH);
        return abAIForAgentDef.getBestMove(color);
    }
}

/* ================================================================
 *  Agent 对 Agent 对弈 (arena)
 *
 *  和原来的 selfPlay 的区别: selfPlay 是"同一个 agent 自己跟自己下", 只能看
 *  它能不能收敛; 这里让**两个不同的 agent** 打若干局, 用来比较强弱。
 *
 *  两条方法学上的要求 (少任何一条, 结果都没有解释力):
 *
 *   1. 每局交换先后手。中国象棋先手(红)优势很大, 固定谁执红的话最后只是在测
 *      "谁执红", 而不是"谁更强"。所以胜负按参赛者 A/B 记, 不按红黑记。
 *   2. 局数要够。单局的偶然性足以翻转结论, 所以界面上局数做成可配置, 并且
 *      逐局列出明细 (谁执红、谁胜、多少手)。
 * ================================================================ */

/*
 * captureLine - "该吃的时候吃了吗" 的一行读数 (2026-09, O1)
 *
 * 为什么单独一个函数: 它要出现在**三处** (实时比分行 / 每局明细 / 整场汇总), 三处各写
 * 一份格式化迟早会漂 —— 本工程已经有过"同一个量在屏幕与导出文件里不一致"的事故。
 *
 * 两个刻意的选择:
 *   * 分母为 0 时说"无机会", **不印 0.0%**: "一次吃子机会都没遇到"与"有机会一次都没吃"
 *     是两件完全不同的事, 混成一个 0% 会把后者的严重性稀释掉 (前者只是这局没碰面);
 *   * 同时印 **分子/分母** 与百分比: 只印百分比时, 5% 背后的 "2/40" 与 "20/400"
 *     在读数上是两回事, 而后者才说明"样本够多, 结论可信"。
 */
QString ChessBoard::MatchStats::captureLine() const
{
    auto one = [](const QString &who, int chosen, int avail) -> QString {
        if (avail <= 0) {
            return QStringLiteral("%1 无机会").arg(who);
        }
        return QStringLiteral("%1 %2/%3=%4%")
            .arg(who)
            .arg(chosen)
            .arg(avail)
            .arg(100.0 * (double)chosen / (double)avail, 0, 'f', 1);
    };
    return QStringLiteral("该吃时吃到: ")
           + one(QStringLiteral("A"), capChosenA, capAvailA) + QStringLiteral("  ")
           + one(QStringLiteral("B"), capChosenB, capAvailB);
}

QString ChessBoard::MatchStats::summary() const
{
    QString s = QStringLiteral("%1 %2 : %3 %4")
                    .arg(agentA).arg(winA).arg(winB).arg(agentB);
    if (draws > 0) {
        s += QStringLiteral(" (和 %1)").arg(draws);
    }
    s += QStringLiteral("  共 %1 局 / %2 手").arg(games).arg(plies);
    if (agentErrors > 0) {
        s += QStringLiteral("  [%1 次无效走法已兜底]").arg(agentErrors);
    }
    if (aborted) {
        s += QStringLiteral("  [已中止]");
    }
    /*
       [O1] 把"该吃的时候吃了吗"放进**比分那一行**: 这一行是用户在对弈过程中与结束时
       都会看到的位置, 而吃子与否正是他报告的那个现象。A/B 的指代由 detail() 的
       "参赛方"一段给出 (与比分行 "%1 : %3" 同一套 A/B 约定)。
    */
    if (capAvailA > 0 || capAvailB > 0) {
        s += QStringLiteral("  |  ") + captureLine();
    }
    return s;
}

QString ChessBoard::MatchStats::detail() const
{
    QString d = summary();
    d += QStringLiteral("\n\n参赛方:\n  A = %1\n  B = %2").arg(agentA, agentB);
    /*
       ---- 这一场到底学不学 (P0-a 对弈模式) ----
       必须写在报告里: 在它之前, "比分"与"训练"是混在一起的 —— 双方都在学, 于是
       "A 赢了 B" 既可能因为 A 变强、也可能因为 B 变弱。用户读到的却只是一行比分。
       这里把模式**直接印出来**, 于是读数自带前提, 不再需要读者自己去悟。
    */
    d += QStringLiteral("\n对弈模式: %1").arg(modeName);
    /*
       ---- B 方到底冻住了什么 (P0-b 收尾) ----
       "这一场学不学"与"这一场的权重会不会变"是**两件事**, 而比分只在前者被写清时
       才有解释力。这一行把后者也印出来: 报告里说"冻结"时, 读者必须能知道冻的是
       "B 那一手不学习"还是"B 的权重逐字节不变" —— 两者对结论的支持强度完全不同。
    */
    if (!frozenNote.isEmpty()) {
        d += QStringLiteral("\nB 方冻结口径: %1").arg(frozenNote);
    }
    if (frozenToSnapshot) {
        d += QStringLiteral("\n  B 方用开场快照走了 %1 手").arg(frozenDecisions);
    }
    if (frozenAuditDone) {
        d += QStringLiteral("\n  逐字节审计: %1")
                 .arg(frozenWeightsUnchanged
                          ? QStringLiteral("B 方的权重与开场快照**逐字节相同**")
                          : QStringLiteral("!! B 方的权重与开场快照**不同** (冻结是假的)"));
    }
    if (bgRoundsSkippedByMode > 0) {
        d += QStringLiteral("\n  后台训练: 本模式下跳过了 %1 轮 (不写主 agent 权重)")
                 .arg(bgRoundsSkippedByMode);
    }
    if (!learnsSomething) {
        d += QStringLiteral("\n  注: 本场**不能**当作棋力结论 —— 只有固定参照物 +"
                            " 对照口径下的比分才可归因 (见 docs/agents_design.md §10.3"
                            " 与 bench_anchor 的锚点对局)");
    }
    d += QStringLiteral("\n\n每局明细 (每局交换先后手):\n");
    d += log;
    if (plies > 0) {
        const double avg = double(totalThinkMs) / double(plies);
        d += QStringLiteral("\n思考耗时: 累计 %1 s, 平均 %2 ms/手, 单步最长 %3 ms")
                 .arg(totalThinkMs / 1000.0, 0, 'f', 1)
                 .arg(avg, 0, 'f', 0)
                 .arg(maxThinkMs);
    }
    return d;
}

ChessBoard::AgentType ChessBoard::typeForTurn(int turn, AgentType redType,
                                              AgentType blackType) const
{
    return (turn == Stone::COLOR_RED) ? redType : blackType;
}

/* 打一局: 红方 redType, 黑方 blackType。返回 Chess::RESULT_*, 被中止则返回 ONGOING */
int ChessBoard::playMatchGame(AgentType redType, AgentType blackType, bool aIsRed,
                              MatchStats &st, double &rewardRed, double &rewardBlack,
                              double &rewardA, double &rewardB)
{
    rewardRed = 0.0;
    rewardBlack = 0.0;
    rewardA = 0.0;
    rewardB = 0.0;
    /*
       [2026-10] 价值评估: 本局的 (V, mover) 采样从**空**开始。
       放在函数头而不是"第一次采样时清": 上一局如果是**被中止**的 (matchAgents 里那条
       break), 它的残留会被这一局接着用 —— 而那些样本对应的 z 谁也说不清。
    */
    m_valueGameV.clear();
    m_valueGameSampleIdx.clear();
    m_valueGameStep.clear();
    m_valueGameStepMover.clear();
    /*
      红/黑 -> A/B 的换算只写一次 (以前 matchAgents 里又算了一遍, 两处规则必须永远
      一致, 否则"逐局明细里的奖励"和"曲线上的点"会对不上)。被中止时也要同步一次,
      所以包成 lambda, 每个出口都调。
    */
    auto syncAB = [&]() {
        rewardA = aIsRed ? rewardRed : rewardBlack;
        rewardB = aIsRed ? rewardBlack : rewardRed;
    };
    /*
       本局"A 方是不是执红" —— 每局都要重写 (matchAgents 每局交换先后手),
       决策路径靠它把"这一手替谁下"换算成 SideRole (见下面 setSideRole 那一处)。
    */
    m_matchAIsRed = aIsRed;
    /*
       ---- ④ 两本账 (本次改动新增) ----
       `rewardRed/rewardBlack` 从此是"**曲线口径**的即时累计": 哪一方有学习口径就用
       学习口径, 没有就用引擎口径 (见下面每手的取值)。另外两对本地的账只用于
         * 本局的 RewardAccounting (断言/自检要能同时看到两个口径);
         * 曲线口径的**选择** (learnOk* 一旦为真就一直用学习口径)。
       `movesRed/movesBlack` 是"每步代价"那一项的乘数, 断言里的换算关系要用到它。
    */
    double engineRed = 0.0, engineBlack = 0.0;
    double learnRed = 0.0, learnBlack = 0.0;
    int movesRed = 0, movesBlack = 0;
    bool learnOkRed = false, learnOkBlack = false;
    {
        QMutexLocker locker(&mutex);
        chess.reset();
    }
    int turn = Stone::COLOR_RED;
    int moves = 0;
    int ret = Chess::RESULT_DRAW;

    while (moves < m_maxPliesPerGame) {
        if (m_matchAbort.load()) {
            syncAB();
            return Chess::RESULT_ONGOING;
        }
        m_selfPlayMoveNo = moves + 1;

        {
            QMutexLocker locker(&mutex);
            const int r = chess.getResult(turn);
            if (r != Chess::RESULT_ONGOING) {
                ret = r;
                break;
            }
        }

        const AgentType who = typeForTurn(turn, redType, blackType);
        /*
           ---- 告诉决策路径"这一手在替谁下" (P0-a/P0-b 对弈模式) ----
           A 方 = 学习者 (待评估/待训练那一方), B 方 = 冻结的对手。
           决策路径只看到一个 AgentType, 分不出它是 A 还是 B (同一个类型可能两边都在用),
           所以这个角色必须由对局循环来填。
        */
        setSideRole(((turn == Stone::COLOR_RED) == aIsRed) ? SIDE_LEARNER : SIDE_FROZEN);
        /*
           ---- [P1] 顺手把"这一手的对手是哪个类型"也填上 ----
           探索的对手参数要用它 (preTrainThenDecide 只知道自己替谁决策, 而对手是**另一边**)。
           这里是唯一知道"A/B 与红黑怎么对应"的地方 —— 在别处按 m_agentType 猜会猜错:
           对弈里 m_agentType 是界面上选中的那个, 与场上这手是谁毫无关系。
        */
        m_rolloutOpponentType = (who == redType) ? blackType : redType;
        auto t0 = std::chrono::steady_clock::now();
        Step step = aiThinkForAgent(turn, who);
        auto t1 = std::chrono::steady_clock::now();
        const long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        st.totalThinkMs += ms;
        if (ms > st.maxThinkMs) {
            st.maxThinkMs = ms;
        }
        st.plies++;
        /* 让界面的"AI思考时间"标签逐手跳动 */
        emit aiThinkFinished(ms);

        /* 无合法走法 (将杀 / 困毙) */
        if (!step.valid) {
            /*
               先分清"真的无棋可走"和"agent 返回了一个无效走法"。
               后者不是将杀 —— 以前直接把 !step.valid 当成"走棋方被将死", 于是一个
               抽风的 agent 会让对局在没走过任何一步的情况下被判负 (界面上就是
               "棋子没动, 我却输了")。这里用第一个合法走法兜底, 并把次数记下来。
            */
            std::vector<Step *> legal;
            {
                QMutexLocker locker(&mutex);
                chess.sample(turn, legal);
                if (!legal.empty()) {
                    /*
                       这个分支正常永远不该进。真进来了说明某个 agent 的搜索返回了
                       无效走法 —— 那是个 bug, 但**不能让它决定胜负**: 用第一个合法
                       走法兜底, 并把次数记进 MatchStats (显示在比分行里)。
                    */
                    std::fprintf(stderr,
                                 "[arena] agent(%d) 返回无效走法 (第 %d 局第 %d 手): "
                                 "valid=%d id=%d pos=(%d,%d)->(%d,%d), "
                                 "仍有 %zu 个合法走法, 已兜底\n",
                                 (int)who, st.games + 1, moves + 1,
                                 (int)step.valid, step.id,
                                 step.pos.x, step.pos.y, step.nextPos.x, step.nextPos.y,
                                 legal.size());
                    step = *legal[0];
                    st.agentErrors++;
                }
            }
            Steps::instance().put(legal);
        }

        if (!step.valid) {
            /* 确实没有合法走法 -> 走棋方被将杀 / 困毙 */
            ret = (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                             : Chess::RESULT_RED_WIN;
            break;
        }

        /*
           ---- [2026-10 用户口径] 对局中行为克隆: **学生走一手** -> 一条样本 ----
           判据与上一版不同 (这是本轮的口径改动): 老师不再由对手充当, 而是**下拉框**选的
           那一档 AB, 所以这里判的是"**走这一手的这一方是不是能做 BC 的学生**" ——
           对手是谁 (AB / 别的 agent / 人) 完全不参与判定。
           三个条件缺一不可 (与 bcSampleForHumanTurn 那三个逐条对齐):
             * 下拉框不是"关" (老师深度 > 0);
             * 模式是**训练对局** —— 评估/只对弈声称"权重不变", 而 BC 是学习;
             * 走这一手的 agent 有 BC 路径 (PPO 两支 / SAC 三支)。
           判定放在棋盘锁**外面** (只读几个 atomic/常量), 位置样本在锁内拷, 干活在锁外 ——
           锁的顺序因此只有一种 (棋盘 -> agent), 与 getAgentSelfCheck / 下面那段
           终局值同步 (envLock) 完全一致。
        */
        const bool bcOn = (m_bcTeacherDepth.load() > 0) && (m_matchMode.load() == MATCH_TRAIN)
                          && bcSupported(who);
        const AgentType bcStudent = who;
        Chess bcPos;
        /* 走这一手的是哪一方 (BC 样本的 color 必须是它, 而不能读下面已经被翻过的 turn) */
        int bcMover = Stone::COLOR_NONE;
        /*
           ---- [2026-10] 价值评估: 这一手的 V(s) 探针 ----
           同一个时序要求 (必须在 moveForward 之前取局面), 但它服务的是另一件事:
           BC 要的是"学生这一手"当监督样本, 而价值评估要的是"agent 在**这个局面**上
           把价值估成了多少", 局末再与真实胜负对照 (EV/校准)。
           只在**走这一手的 agent 有标量 V 头**时采 (PPO 两支); 代价是一次 critic 前向。
        */
        const bool valueProbeOn = valueProbeOfSupported(who);
        Chess vPos;
        int vMover = Stone::COLOR_NONE;
        /*
           本手之前的**引擎口径**累计奖励 —— 用来求"这一手的引擎口径即时奖励"。
           ⚠ 必须用引擎口径 (engineRed/engineBlack), **不是** rewardRed/rewardBlack:
           后者是"有学习口径就用学习口径、否则回退引擎口径"的两本账, 与 AB/MCTS 对弈时
           一局之内会混两种口径 (材质 ×0.1 与 ×1), 于是折扣回报被对手那本账主导, V 与它
           反着走 ⇒ 曲线上读到负值 (用户报的"与 MCTS 对弈数值小于 0")。
           引擎口径对每一手都有定义、且与 agent 类型无关 ⇒ 不再混口径。
        */
        const double rewBeforeEngRed = engineRed;
        const double rewBeforeEngBlack = engineBlack;
        /*
           这一手的走子方: **必须在棋盘锁之前取** —— 锁内 `turn` 会被翻到下一手
           (下面那段"sideToMove 必须跟着 turn 走"), 锁后再读就是下一手的走子方了。
        */
        const int valueMoverNow = turn;

        {
            QMutexLocker locker(&mutex);
            double totalReward = 0;
            /* 走这一步的是 turn 方, moveForward 之前先记下来 */
            const int mover = turn;
            /*
               ---- 对局中行为克隆: 把"学生这一手"的原局面拷一份 ----
               必须在 moveForward **之前**: 此刻棋盘还是这一步的原局面, 而"轮到走棋的
               就是学生" —— 这正是那条样本的定义 (学生的局面 + 老师在该局面上的着法)。
               与下面的吃子读数、学习口径即时奖励是同一个时序要求, 三处必须在同一刻取。
            */
            if (bcOn) {
                bcPos = chess;
                bcMover = turn;
            }
            /* 价值探针的局面副本 (同一个"落子之前"的时刻) */
            if (valueProbeOn) {
                vPos = chess;
                vMover = turn;
            }
            /*
               ================================================================
               [O1, 2026-09] 吃子行为: "该吃的时候吃了吗"
               ================================================================
               位置与下面那条学习口径即时奖励**同一个时序要求**: 必须在 moveForward
               之前 —— 此刻棋盘还没动, 于是 `chess.sample()` 给出的就是这一手的**完整**
               合法集, 而 `step` 就是真正要走出的那一手 (无效走法已在上面兜底替换过)。
               两件事因此严格对应同一个局面, 这就是这个读数全部准确性的来源。

               代价: 每手多一次走法生成 (~40 手)。相对一次 agent 决策 (ms 到 s 级)
               可以忽略, 而它换来的是界面上**唯一**能回答"吃子无动于衷"的数
               (奖励曲线不是合适的仪器 —— 见 MatchStats 里那段与 docs 的说明)。

               累计**直接写进 st**、不存本局局部量: 本函数有一条"被中止就 early return"
               的路径, 用局部量 + 函数末尾累加的话, 被中止的那一局会被漏掉。
               归属用 `aIsRed` 换算成 A/B, 与奖励记账、胜负记账同一处规则。
            */
            {
                std::vector<Step *> legalCap;
                chess.sample(turn, legalCap);
                bool capAvail = false;
                for (std::size_t li = 0; li < legalCap.size(); li++) {
                    if (legalCap[li]->nextId != Stone::ID_NONE) {
                        capAvail = true;
                        break;
                    }
                }
                Steps::instance().put(legalCap);
                const bool capChosen = (step.nextId != Stone::ID_NONE);
                const bool moverIsA = ((turn == Stone::COLOR_RED) == aIsRed);
                if (capAvail) {
                    (moverIsA ? st.capAvailA : st.capAvailB)++;
                }
                if (capChosen) {
                    (moverIsA ? st.capChosenA : st.capChosenB)++;
                    /* 材质**原值**, 不含将 (吃将必然是终局, value_jiang=1000 会顶爆这个数) */
                    if (step.nextId >= 0 && step.nextId < 32
                        && chess.stones[step.nextId] != nullptr
                        && chess.stones[step.nextId]->type != Stone::TYPE_JIANG) {
                        (moverIsA ? st.matGainedA : st.matGainedB)
                            += chess.stones[step.nextId]->value;
                    }
                }
            }
            /*
               ---- ④ 学习口径的即时奖励: **必须在 moveForward 之前算** ----
               `computeReward` 要按 `s.nextId` 去读被吃子的 value, 而 moveForward 会把
               它置成 alive=false (见 stone.h 的 stepReward 注释)。这是本改动唯一一处
               顺序敏感的地方, 放错位置的表现是"吃子奖励恒为 0" —— 曲线看起来正常,
               只是永远只有每步代价。
            */
            const double learnNow = learningStepRewardOrNaN(who, step, mover);
            chess.moveForward(&step, totalReward);
            /*
               环境奖励记账。`Chess::moveForward` 的 totalReward 是**黑方视角**的记账
               (吃红子 +value, 吃黑子 −value), 而 RL agent 学的是"走子方视角"的奖励
               (与 SACAZAgent::computeReward、终局 ±1 同一套约定), 所以这里换算过去。
               换算公式: 黑方视角 == −(红方视角), 于是
                 红方走: 走子方收益 = −totalReward
                 黑方走: 走子方收益 = +totalReward
               (实测探针: 红炮吃黑马 totalReward = −0.30, 即红方收益 +0.30。)
            */
            const double moverReward =
                (mover == Stone::COLOR_RED) ? -totalReward : totalReward;
            /* 两本账都记 (见 RewardAccounting 的说明): 引擎口径恒记; 学习口径在 agent
               给了数的时候记 —— 纯搜索 agent (Alpha-Beta / MCTS) 没有学习口径, 它的曲线
               就继续用引擎口径, 只是界面上会**标注**出来。 */
            if (mover == Stone::COLOR_RED) {
                engineRed += moverReward;
                movesRed++;
                if (std::isfinite(learnNow)) {
                    learnRed += learnNow;
                    learnOkRed = true;
                }
            } else {
                engineBlack += moverReward;
                movesBlack++;
                if (std::isfinite(learnNow)) {
                    learnBlack += learnNow;
                    learnOkBlack = true;
                }
            }
            /*
               曲线取哪一本账: 有学习口径就用学习口径 (本次改动的**目的**), 否则回退引擎
               口径。学习口径一旦拿到过值 (learnOk*), 本局就一直用它 —— 不能逐手在两个
               口径之间跳 (那会把曲线变成一条量纲来回变的折线)。
            */
            rewardRed = learnOkRed ? learnRed : engineRed;
            rewardBlack = learnOkBlack ? learnBlack : engineBlack;
            /* A/B 的归属由 aIsRed 换算, 这里只管红黑 */
            /* sideToMove 必须跟着 turn 走, 否则下一手 getResult 会看错方 */
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            chess.sideToMove = turn;
        }
        /*
           ---- [2026-10 用户口径] 对局中行为克隆: 学生这一手 -> 一条样本 -> 一次 actor 更新 ----
           放在棋盘锁**外面**、下一手决策**之前**: 那一刻没有别的 agent 在工作, 所以
           只短暂持 agent 锁; 一次更新在 MLP 骨干上是毫秒级, 相对一步决策 (0.3~8 s)
           可以忽略。老师那一手由 bcOnStudentMove 在**局面副本**上现场搜 (深度取自下拉框)。
        */
        if (bcOn) {
            bcOnStudentMove(bcStudent, bcPos, bcMover);
        }
        /*
           ---- [2026-10] 价值评估: 采这一手的 V(s) ----
           与 BC 一样在棋盘锁**外面**做 (只短暂持 agent 锁): 探针会临时换 agent 的棋盘
           副本再还原, 所以必须在锁内 (m_agentMutex 保护常驻实例的全部读写)。
           采不到就跳过 —— 不支持的 agent (SAC 是 Q 不是 V) / 实例还没建 / NaN 都不算错。

           ⚠ 即时奖励要**每一手都记** (不只采到 V 的那些): z 是"该手之后的折扣回报",
             求和要从这一手走到局末, 中间对手那些手一样要计入 (漏掉它们, z 就不是回报)。
        */
        {
            const double stepEngRed = engineRed - rewBeforeEngRed;
            const double stepEngBlack = engineBlack - rewBeforeEngBlack;
            m_valueGameStep.push_back((valueMoverNow == Stone::COLOR_RED) ? stepEngRed
                                                                         : stepEngBlack);
            m_valueGameStepMover.push_back(valueMoverNow);
        }
        if (valueProbeOn) {
            double v = 0.0;
            bool got = false;
            {
                std::lock_guard<std::mutex> agentLock(m_agentMutex);
                got = valueProbeOf(who, vPos, vMover, v);
            }
            if (got) {
                m_valueGameV.push_back(v);
                m_valueGameSampleIdx.push_back((int)m_valueGameStep.size() - 1);
            }
        }
        /*
            每手报一次"本局累计"奖励进度 (锁外 emit: 信号是队列投递到 GUI 线程的,
            持锁期间碰 Qt 的元对象系统没必要)。
            一局几百手、十几分钟, 只在局末报一次的话, 整局过程中奖励曲线一动不动 ——
            用户反馈"对弈时奖励曲线没有更新"就是这个原因。
        */
        syncAB();
        emit matchRewardProgress(m_matchGameNo.load(), moves + 1, rewardA, rewardB);
        moves++;
    }

    /*
       ---- 终局值 (④ 之后按**口径**取, 不再一律 ±1) ----
       学习口径的 agent 走它自己的 `terminalReward()`: SAC 塑形开着 (rewardShape=2) 时
       那是 ±(1 + 败方剩余材质/3.5)。**必须**问 agent 要, 不能在界面这侧照抄一份公式 ——
       抄一份就等于"界面上显示的游戏"与"它真正学的游戏"是两个 (见 sacazagent.h 的
       rewardShape 说明: 三个终局出口只允许有一个来源)。
       引擎口径 (Alpha-Beta / MCTS / EVAB) 仍然是 ±1, 和棋 0。
    */
    if (learnOkRed || learnOkBlack) {
        /*
           同步 env 到**终局局面**再取终局值: SAC 的 terminalReward 在塑形开着时要数
           "败方还剩多少材质", 而 env 此刻停在最后一手**之前** (它只在每手决策前 sync
           一次)。不同步的话它数的是上一手, 曲线上的终局值会比它真正学到的少一个子。
        */
        std::lock_guard<std::mutex> envLock(m_agentMutex);
        env = chess;
    }
    double termRed = 0.0, termBlack = 0.0;
    if (learnOkRed) {
        const double t = learningTerminalRewardOrNaN(redType, ret, Stone::COLOR_RED);
        termRed = std::isfinite(t) ? t : outcomeForMover(ret, Stone::COLOR_RED);
    } else {
        termRed = outcomeForMover(ret, Stone::COLOR_RED);
    }
    if (learnOkBlack) {
        const double t = learningTerminalRewardOrNaN(blackType, ret, Stone::COLOR_BLACK);
        termBlack = std::isfinite(t) ? t : outcomeForMover(ret, Stone::COLOR_BLACK);
    } else {
        termBlack = outcomeForMover(ret, Stone::COLOR_BLACK);
    }
    rewardRed += termRed;
    rewardBlack += termBlack;
    /* 局末这一点也换算进去, 于是出参就是"本局最终值" */
    syncAB();

    /* ---- ④ 记下本局的两种口径账 (断言/自检用, 见 RewardAccounting) ---- */
    m_lastReward.clear();
    m_lastReward.engineImmediateA = aIsRed ? engineRed : engineBlack;
    m_lastReward.engineImmediateB = aIsRed ? engineBlack : engineRed;
    m_lastReward.learnImmediateA = aIsRed ? learnRed : learnBlack;
    m_lastReward.learnImmediateB = aIsRed ? learnBlack : learnRed;
    m_lastReward.learnTerminalA = aIsRed ? termRed : termBlack;
    m_lastReward.learnTerminalB = aIsRed ? termBlack : termRed;
    m_lastReward.engineTerminalA = aIsRed ? outcomeForMover(ret, Stone::COLOR_RED)
                                          : outcomeForMover(ret, Stone::COLOR_BLACK);
    m_lastReward.engineTerminalB = aIsRed ? outcomeForMover(ret, Stone::COLOR_BLACK)
                                          : outcomeForMover(ret, Stone::COLOR_RED);
    m_lastReward.movesA = aIsRed ? movesRed : movesBlack;
    m_lastReward.movesB = aIsRed ? movesBlack : movesRed;
    m_lastReward.learnCaliperA = aIsRed ? learnOkRed : learnOkBlack;
    m_lastReward.learnCaliperB = aIsRed ? learnOkBlack : learnOkRed;

    /*
       达到步数上限: 原来 selfPlay 按 evaluate() 判胜 (score == 0 也返回红胜),
       于是"平局"分支永远不可达。现在直接判和棋 —— 中国象棋的自然限着本来就是和棋。
    */
    return ret;
}

ChessBoard::MatchStats ChessBoard::matchAgents(AgentType typeA, AgentType typeB, int games)
{
    MatchStats st;
    st.agentA = agentDisplayName(typeA);
    st.agentB = agentDisplayName(typeB);
    /*
       ---- 把本场的对弈模式写进报告 (P0-a) ----
       在这里**快照**模式而不是在 detail() 里现读: 一场对弈可能跑很久, 期间用户可能
       又改了模式 —— 报告必须描述**这一场实际用的**那个模式, 否则"报告说评估、其实是
       在训练"就是一个假的读数 (本工程最怕的那类静默失效)。
    */
    st.modeName = matchModeName(m_matchMode.load());
    st.learnsSomething = matchLearnsSomething() && m_preTrainEnabled.load();
    if (games < 1) {
        games = 1;
    }

    /*
       ================================================================
        ---- B 方冻住的是什么 (P0-b 收尾): 学习, 还是**权重**? ----
       ================================================================
       用户口径: "现在冻结的只是'学习', 不是'权重'". 前者拦得住 B 自己更新, 拦不住
       **A 的更新经由共用实例流到 B 身上** —— 而"每个 agent 类型只有一个常驻实例"
       (m_sfXXX) 意味着 A == B 类型时两者就是同一张网: A 学一次, B 的棋力当场变一次。
       所以这里在开场做一次判定并把结论写进报告, 三种强度分得清:
         · 评估模式 + A/B 同类型  -> **取快照 + 建冻结实例** (最强口径);
         · 评估模式 + A/B 不同类型-> 两个实例, A 的学习写不到 B (后台训练已停摆);
         · 只对弈不学习           -> 双方都不学, 且后台训练本模式停摆。
       ⚠ 快照必须在**对局循环之前**取: A 的第一手决策里就有一次在线更新, 而它与 B
         共用实例 —— 放到后面取, 快照里就混进了 A 的更新 (见 freezeOpponentToSnapshot)。
       RAII: 无论正常结束、被中止还是提前 return, 冻结实例与临时文件都必须拆掉:
         留着的后果是**下一场**会继续用上一场的快照下棋, 而那没有任何读数能反映。
    */
    struct FrozenGuard {
        ChessBoard *self;
        ~FrozenGuard() { self->releaseFrozenOpponent(); }
    } frozenGuard{this};

    /*
       ---- 对弈期间: 权重不落盘 (2026-10 用户口径) ----
       必须**早于**下面那句 freezeOpponentToSnapshot 武装: 评估模式的开场快照本身也是一份
       权重文件, 它就在这场对弈的准备阶段写出来 —— 晚一步武装就漏掉它 (而那一份是
       3 x 146 MB 量级, 正好是最该被拦下的那种)。

       RAII: 正常结束 / 被中止 / 提前 return 三条路都要解除 (与 FrozenGuard 同一手法)。
       两个作用域取或 (见头文件): 这里是"整场", 人机那一局是另一个开关。

       ⚠ 训练对弈模式下后台训练**照常跑**, 它那一轮的权重往返因此走内存 —— 这就是本
         开关存在的理由; 评估/只对弈模式下后台训练本来就被 isBackgroundTrainingPaused()
         整体停摆, 那时改道只是"顺手也成立"。
    */
    m_playDivertMatch.store(true);
    updatePlayWeightDivert();
    struct PlayDivertGuard {
        ChessBoard *self;
        ~PlayDivertGuard()
        {
            self->m_playDivertMatch.store(false);
            self->updatePlayWeightDivert();
        }
    } playDivertGuard{this};
    /*
       计数归零: 报告里的"B 方用快照走了几手"必须是**本场**的数 ——
       不归零的话, 一场没有快照的对局会带上上一场留下的数字 (那是假的读数)。
    */
    m_frozenDecisions.store(0);
    const int bgSkippedBefore = m_bgRoundsSkippedByMode.load();

    /*
       ---- [2026-10 用户口径] 对局中行为克隆: 每场归零 + 记下这一场的"学生/老师" ----
       归零的理由与上面那条计数归零相同: 报告里的 samples/updates/CE 必须是**本场**的数,
       不归零的话"这一场一项都没克隆"会带上上一场的读数 (那是假的读数)。
       学生/老师只在这里认得出来 (对局循环里只知道"这一手是谁在走"), 所以在这里算好。
       老师来自**下拉框** (与对手无关), 学生在 A/B 里挑 —— 两侧都能做 BC 时**两边都在
       克隆**, 报告里如实写出两个人的名字。
       `bcHasStudent` 与界面上的下拉框启用条件**同一份实现** —— 两边各写一遍的话,
       "下拉框亮着但场上什么也没发生"迟早会发生。
    */
    {
        const int teacherDepth = m_bcTeacherDepth.load();
        const bool anyStudent = bcHasStudent(typeA, typeB);
        const bool bcWillRun = (teacherDepth > 0) && (m_matchMode.load() == MATCH_TRAIN)
                               && anyStudent;
        QString studentLabel;
        if (bcWillRun) {
            const bool sa = bcSupported(typeA);
            const bool sb = bcSupported(typeB);
            if (sa && sb) {
                studentLabel = QStringLiteral("%1 + %2 (两侧都在克隆)")
                                   .arg(agentDisplayName(typeA), agentDisplayName(typeB));
            } else {
                studentLabel = agentDisplayName(sa ? typeA : typeB);
            }
        }
        bcResetMatchStats(studentLabel, bcWillRun ? bcTeacherName(teacherDepth) : QString());
        if (teacherDepth > 0 && !bcWillRun) {
            /*
               选了老师但没生效, 必须**说清原因** (否则用户只会看到"选了没用"):
               两种原因是完全不同的两件事 —— 模式不对, 或者这一对选手没有学生。
            */
            emit bcProgress(QStringLiteral(
                "[BC] 本场没有做行为克隆: %1")
                .arg(m_matchMode.load() == MATCH_TRAIN
                         ? QStringLiteral("A/B 两边都不是能做 BC 学生的 agent "
                                          "(只有 PPO+MCTS / PPO+MCTS(MLP专家) / SAC+AZ / "
                                          "SAC+AZ-MoE / SAC+AZ-MoE-MLP 这五支能做学生)")
                         : QStringLiteral("对弈模式不是\"训练对局\" (评估/只对弈模式下"
                                          " 权重必须不变, 而 BC 是学习)")));
        }
    }
    {
        const MatchMode m = m_matchMode.load();
        if (m == MATCH_NO_LEARN) {
            st.frozenNote = QStringLiteral("双方都不学, 且后台训练在本模式下停摆"
                                           " ⇒ 本场没有任何写权重的路径");
        } else if (m == MATCH_TRAIN) {
            st.frozenNote = QStringLiteral("训练模式: 不冻结任何一方 (双方都在学,"
                                           " 权重本来就会变)");
        } else if (typeA != typeB) {
            st.frozenNote = QStringLiteral("A/B 不是同一个 agent 类型 ⇒ 各自一个常驻实例,"
                                           " A 的学习写不到 B 身上; 后台训练本模式停摆"
                                           " ⇒ B 的权重无写入路径 (不需要快照)");
        } else if (!m_opponentSnapshotEnabled) {
            /* 对照口径: 显式关掉快照, 让"共用实例导致 B 跟着变"这件事能被测出来 */
            st.frozenNote = QStringLiteral("!! 权重快照被显式关闭 (对照口径): A/B 共用同一个"
                                           " 实例 ⇒ A 每学一次 B 就变一次, 本场比分**不可"
                                           " 归因** (这正是 P0-b 要修掉的那个洞)");
        } else {
            QString note;
            st.frozenToSnapshot = freezeOpponentToSnapshot(typeB, note);
            st.frozenNote = note;
            if (!st.frozenToSnapshot && st.frozenNote.isEmpty()) {
                st.frozenNote = QStringLiteral("B 方的权重快照没能建起来");
            }
        }
    }

    /*
       ---- 暂停后台训练 (P0-b): "冻结"必须包括权重不变 ----
       后台训练每轮会把权重同步回主 agent, 而对弈用的就是这个主 agent。若不暂停,
       "评估对局 / 只对弈不学习"这两个模式声称的冻结只是"这一手不学习", 权重仍会在
       局与局之间被换掉 —— 报告与读数都会骗人。
       放在这里 (而不是每手) 的理由: 一次调用就够, 且它内部会等"正在飞的那一轮"收尾,
       所以对局期间不会有任何后台写权重。
       ⚠ 训练模式下**不暂停**: 那种模式下双方本来就在学, 停不停后台都改变不了
         "权重会变"这件事, 而停掉会让"一边对弈一边后台训练"这个既有用法失效。
       [P0-b 收尾] 这一层之外还有第二层: 训练线程自己也会在**每一轮开头**与**写权重
       之前**判一次对弈模式 (见 isBackgroundTrainingPaused), 于是人机对战那条路
       (不在本函数里) 与"局与局之间"同样不会被后台权重改动污染。保留本层的理由:
       它是个**屏障** —— 它返回时保证"在飞的那一轮"已经收尾或被丢弃, 而且即使用户
       在对局中途把模式切回"训练对局", 这一场也仍然全程无后台写入。
       RAII: 无论正常结束、被中止还是提前 return, 都要恢复。
    */
    const bool pauseBg = (m_matchMode.load() != MATCH_TRAIN) && m_bgTraining.load();
    if (pauseBg) {
        pauseBackgroundTraining();
    }
    struct BgResumeGuard {
        ChessBoard *self;
        bool active;
        ~BgResumeGuard() { if (active) { self->resumeBackgroundTraining(); } }
    } bgGuard{this, pauseBg};

    /*
       对弈期间屏蔽玩家点击。用独立标志而不是改 state —— state 是给 AI 工作线程
       看的, 改它会把那个线程也唤醒起来思考, 于是两个线程同时写同一张棋盘。
    */
    m_selfPlaying = true;
    m_selfPlayMoveNo = 0;
    m_matchAbort = false;
    m_matchRunning = true;
    m_matchGames = games;
    m_matchGameNo = 0;

    /*
       如果玩家是在"AI 正在思考"的时候按下的开始对弈, 棋盘上还有一个在飞的一手。
       把代数 +1 让它作废、把 state 收回 IDLE, 否则工作线程会和这里的对弈线程
       同时写 chess。
    */
    {
        QMutexLocker locker(&mutex);
        ++m_thinkGeneration;
        state = STATE_IDEL;
        selectID = -1;
        condit.wakeAll();
    }

    emit aiThinkingStarted(QStringLiteral("%1 vs %2").arg(st.agentA, st.agentB),
                           m_preTrainEnabled.load() ? m_preTrainSteps.load() : 0);
    emit matchStarted(st.agentA, st.agentB, games);

    for (int g = 0; g < games; ++g) {
        if (m_matchAbort.load()) {
            st.aborted = true;
            break;
        }
        m_matchGameNo = g + 1;

        /* 每局交换先后手: 偶数局 A 执红, 奇数局 B 执红 */
        const bool aIsRed = (g % 2 == 0);
        const AgentType redType = aIsRed ? typeA : typeB;
        const AgentType blackType = aIsRed ? typeB : typeA;

        double rewardRed = 0.0;
        double rewardBlack = 0.0;
        double rewardA = 0.0;
        double rewardB = 0.0;
        /*
           [O1] 本局的吃子行为 = 打完这一局之后 st 里那两个累计量的**差**。
           为什么要取差而不是让 playMatchGame 多返回几个出参: 出参已经有 4 个了, 再加
           4 个会让签名难以阅读; 而"累计量的差"是现成的、且**天然覆盖被中止的局**
           (playMatchGame 里有一条 early return, 用出参就得在两处都写一遍)。
        */
        const int capAvailA0 = st.capAvailA, capChosenA0 = st.capChosenA;
        const int capAvailB0 = st.capAvailB, capChosenB0 = st.capChosenB;
        const int res = playMatchGame(redType, blackType, aIsRed, st, rewardRed,
                                      rewardBlack, rewardA, rewardB);
        const int gCapAvailA = st.capAvailA - capAvailA0;
        const int gCapChosenA = st.capChosenA - capChosenA0;
        const int gCapAvailB = st.capAvailB - capAvailB0;
        const int gCapChosenB = st.capChosenB - capChosenB0;
        if (res == Chess::RESULT_ONGOING) {
            st.aborted = true;      /* playMatchGame 用 ONGOING 表示"被中止" */
            /*
               [2026-10] 被中止的这一局**不折算 z** (没有真实结果), 但本局已经采到的
               (V, mover) 必须丢掉 —— 留着的话它们会在下一局结束时被当成"有胜负的样本"
               折进去, 而 z 是拿下一局的结果算的 (那是**跨局错配**, 会让 EV 完全失真)。
            */
            m_valueGameV.clear();
            m_valueGameSampleIdx.clear();
            m_valueGameStep.clear();
            m_valueGameStepMover.clear();
            break;
        }
        /*
           ---- [2026-10] 价值评估: 本局结束 -> 折算 z -> 更新滚动窗口 -> 可能出点 ----
           放在这里 (而不是 playMatchGame 里) 的原因: **只有这里知道这一局算不算数**
           (上面那条 break 就是"不算数"), 而且 st.games 的语义 = 已完成的局数。
           两种 z 口径都在 valueDiagOnGameEnd 里算: 真实胜负 (±1/0) 与**该手之后的折扣
           回报** (与 critic 的 γ 同口径, 是"曲线一直空着"那个问题的解 —— 胜负口径在
           "被采样的手结果符号全一样"时 zVar=0, EV 数学上无定义)。
        */
        valueDiagOnGameEnd(res, st.games + 1);

        /*
            A/B 视角的本局环境奖励: playMatchGame 已经按 aIsRed 换算好了 (出参),
            这里直接用 —— 不要再自己算一遍 (两份规则迟早会分叉)。
        */
        const double rA = rewardA;
        const double rB = rewardB;

        QString line = QStringLiteral("  第 %1 局: 红=%2 黑=%3 -> ")
                           .arg(g + 1)
                           .arg(aIsRed ? st.agentA : st.agentB,
                                aIsRed ? st.agentB : st.agentA);
        if (res == Chess::RESULT_DRAW) {
            line += QStringLiteral("和棋");
            st.draws++;
        } else {
            const bool redWon = (res == Chess::RESULT_RED_WIN);
            const bool aWon = (redWon == aIsRed);
            line += aWon ? QStringLiteral("%1 胜").arg(st.agentA)
                         : QStringLiteral("%1 胜").arg(st.agentB);
            if (aWon) {
                st.winA++;
            } else {
                st.winB++;
            }
        }
        st.games++;
        /*
           每局明细里带上"本局环境奖励": 这条曲线的意义是"模型下完一局拿到了多少
           奖励", 所以它同时也是逐局明细的一部分 (界面上的列表直接用它)。
           **口径必须一起写出来** (2026-09): 学习口径 (材质 x0.1) 与引擎口径 (材质 x1)
           差 10 倍, 不标口径的数字在事后回看时无法解释 (用户就是拿引擎口径的 4.5 推出
           "材质比赢棋重要 3.5 倍" 的)。哪一方用哪个口径由 agent 自己决定 —— 纯搜索
           agent (Alpha-Beta / MCTS) 没有学习口径, 仍旧是引擎口径。
        */
        const QString rewardText =
            QStringLiteral("  奖励 A=%1[%2] B=%3[%4]")
                .arg(rA, 0, 'f', 2)
                .arg(m_lastReward.learnCaliperA ? QStringLiteral("学习口径")
                                                : QStringLiteral("引擎口径"))
                .arg(rB, 0, 'f', 2)
                .arg(m_lastReward.learnCaliperB ? QStringLiteral("学习口径")
                                                : QStringLiteral("引擎口径"));
        /*
           [O1] 本局的"该吃的时候吃了吗"。放在**每局那一行**里 (而不只放整场汇总):
           逐局明细是用户事后回看的地方, 而"吃子无动于衷"往往是**某一类局面**才发生
           (例如对方送子时不吃) —— 只有逐局能看到它是不是均匀分布的。
           口径与整场那一行 (captureLine) 完全一致, 只是这里只印本局。
        */
        const QString capText =
            QStringLiteral("  该吃时吃到 A=%1/%2 B=%3/%4")
                .arg(gCapChosenA).arg(gCapAvailA)
                .arg(gCapChosenB).arg(gCapAvailB);
        st.log += line + QStringLiteral("  (%1 手)").arg(m_selfPlayMoveNo.load())
                  + rewardText + capText + QStringLiteral("\n");

        emit matchGameFinished(st.games, games, line + rewardText + capText);
        /* ---- 实时比分 (界面用, 见 matchScoreChanged 的注释) ---- */
        {
            QString score = QStringLiteral("%1 %2 : %3 %4")
                                .arg(st.agentA).arg(st.winA).arg(st.winB).arg(st.agentB);
            score += QStringLiteral("   和 %1").arg(st.draws);
            score += QStringLiteral("   (%1/%2 局)").arg(st.games).arg(games);
            /*
               [O1] 实时比分那一行也带上"该吃的时候吃了吗"的**整场累计**:
               这一行是用户在对弈过程中一直盯着的位置, 而吃子与否正是他报告的现象。
               整场累计 + 每局明细里各有一份, 于是"一直是这个水平"与"只是这一局异常"
               能当场分开 (见 captureLine 的说明)。
            */
            if (st.capAvailA > 0 || st.capAvailB > 0) {
                score += QStringLiteral("   该吃时吃到 A=%1/%2 B=%3/%4")
                             .arg(st.capChosenA).arg(st.capAvailA)
                             .arg(st.capChosenB).arg(st.capAvailB);
            }
            emit matchScoreChanged(score);
        }
        emit gameRewardSample(st.games, st.agentA, st.agentB, rA, rB);
    }

    /*
       ---- 冻结的**证据** (P0-b 收尾) ----
       审计在拆冻结实例之前做 (它比的就是那个实例此刻的权重): 把 B 方全程用过的实例
       的权重再存一份, 与开场快照逐字节比对。默认关闭 —— 它要额外写一次权重文件;
       打开它的场合是"要证据"(测试 / 需要写进报告的评估)。
    */
    if (m_frozenWeightAudit && m_frozenOpponent != nullptr) {
        st.frozenAuditDone = true;
        st.frozenWeightsUnchanged = auditFrozenOpponentWeights();
    }
    st.bgRoundsSkippedByMode = m_bgRoundsSkippedByMode.load() - bgSkippedBefore;
    st.frozenDecisions = m_frozenDecisions.load();

    /*
       ---- [2026-10] 对局中行为克隆的收尾报告 ----
       放在这里 (而不是每局) 的理由: 与"这条曲线是一整场的"同一口径 —— 用户勾的是一个
       **开关**, 他要知道的是"这一场克隆了多少、学没学动", 而不是每一局的碎数。
       它同时写 m_bcReport (自检面板) 与对局明细 (逐局列表里那一行)。
    */
    bcFinishMatchReport(st);

    m_matchRunning = false;
    m_selfPlaying = false;
    m_selfPlayMoveNo = 0;
    m_matchGameNo = 0;
    m_matchGames = 0;
    emit aiThinkingStopped();
    emit matchFinished(st.summary(), st.detail());
    return st;
}

/* ================================================================
 *  保存AI模型权重
 * ================================================================ */

/*
 * defaultWeightPath - 每个 agent 的正式权重路径 (单一来源)
 *
 * 这些字符串以前散在 shutdownSave() 里, 而"对弈结束后保存"在 GUI 里另有一套
 * (走文件对话框, 默认文件名还不一样) —— 两套名字不一致就会静默失效: 用户以为
 * 存上了, 下次启动读的却是另一个文件。现在统一到这里。
 */
std::string ChessBoard::defaultWeightPath(AgentType agentType)
{
    switch (agentType) {
    case AGENT_PG:        return "weights/pg_agent.dat";
    case AGENT_DQN:       return "weights/dqn_agent.dat";
    /* PPO+MCTS 系: 前缀 -> <prefix>_actor / _critic (两种骨干各一个文件前缀) */
    case AGENT_PPOMCTS:   return "weights/ppomcts_agent.dat";
    case AGENT_PPOMCTS_MLP: return "weights/ppomcts_mlp_agent.dat";
    case AGENT_DQNMCTS:   return "weights/dqnmcts_agent.dat";
    case AGENT_EVAB:      return "weights/evab_agent.dat";
    /* SAC+AZ 系: 前缀 -> <prefix>_actor / _q1 / _q2 (共享口径那一支是 _trunk + 三个头) */
    case AGENT_SACAZ:     return SACAZAgent::defaultWeightPrefix();
    /*
       [2026-09 dev-sacmoetb] SAC+AZ-MoE 改用**共享骨干**口径 ⇒ 必须换前缀。
       为什么不能继续用 "weights/sacaz_moe_agent": 两个口径写出的文件**个数与语义都
       不同** (独立: _actor/_q1/_q2 三份完整网络; 共享: _trunk/_actorhead/_q1head/_q2head),
       共用一个前缀时 `loadModel` 会去找 _trunk 而找不到 -> 返回 false (还算看得见),
       但更坏的是"有人为了兼容而补一条回退": 那会把独立口径的 `_actor/_q1/_q2` 里的
       **第一份骨干**当成共享骨干、另外两份直接丢掉 —— 静默、不报错、训练还在继续。
    */
    /*
       [2026-09 独立类] TB 那一支的前缀现在由**它自己的类**给出 (拆分前借的是
       SACAZAgent 的静态函数, 因为两个骨干挤在一个类里)。值不变 = 界面存量模型照旧能载入。
    */
    case AGENT_SACAZ_MOE: return SACAZMoETbAgent::sharedWeightPrefix();
    /*
       MoE+MLP 那一支: 它自己的前缀 (weights/sacaz_moe_mlp_shared_agent)。
       **绝不能用 TB 那个值**: 两者骨干不同 ⇒ 权重结构不同, 共用前缀会让一个骨干
       静默覆盖另一个 (见 sacazmoemlpagent.cpp 里 sharedWeightPrefix 的说明)。
    */
    case AGENT_SACAZ_MOE_MLP: return SACAZMoEMlpAgent::sharedWeightPrefix();
    /*
       59e5233 行为还原版: **独立前缀**, 由那个类自己给出 (weights/sacaz_old_agent)。
       绝不能用 SACAZAgent::defaultWeightPrefix() —— 那是"当前口径"那一支的文件,
       两者参数结构相同, 结构指纹挡不住, 于是错误只会在训练很多轮之后以
       "棋力对不上训练量"的形式出现。见 sacazlegacyagent.h 的头注释。
       它的 TB 专家骨干那一支再分一个前缀 (同一个类的另一个 Backbone ⇒ 另一个
       参数量), 由类的重载 defaultWeightPrefix(Backbone) 给出**单一来源** ——
       这里不再手抄字符串, 免得两条前缀漂移。
    */
    case AGENT_SACAZ_OLD: return SACAZLegacyAgent::defaultWeightPrefix();
    case AGENT_SACAZ_OLD_MOE:
        return SACAZLegacyAgent::defaultWeightPrefix(SACAZLegacyAgent::Backbone::SparseMoeTb);
    /* DQN+AB: 前缀 -> <prefix>_trunk / _v / _a */
    case AGENT_DQNAB:  return "weights/dqnab_agent";
    /*
       [2026-09 dev-dqnmcts-moetb] DQN+MCTS (稀疏MoE+TB专家): **前缀** ->
       <prefix>_trunk / <prefix>_q (不是文件名)。
       与旧类 AGENT_DQNMCTS 的 `weights/dqnmcts_agent.dat` 刻意分开: 两者参数量与结构
       都不同, 共用前缀 = 后训练的那一支静默覆盖另一支 (理由见类的头注释"权重文件"段)。
       字符串取自类自己的 `defaultWeightPrefix()` (与三支 SAC 同一手法): 前缀一旦手抄成
       两份, weightFilesOf() 探测的名字就会与 saveModel 写出的名字漂移 —— 那正是
       PPO+MCTS 当年 279 MB 权重从来没被载入过的形状 (见上面那段说明)。
    */
    case AGENT_DQNMCTS_MOE: return DQNMCTSMOETbAgent::defaultWeightPrefix();
    default:              return std::string();
    }
}

bool ChessBoard::hasAgentInstance(AgentType agentType) const
{
    switch (agentType) {
    case AGENT_PG:        return m_sfPG != nullptr;
    case AGENT_DQN:       return m_sfDQN != nullptr;
    case AGENT_PPOMCTS:   return m_sfPPOMCTS != nullptr;
    case AGENT_DQNMCTS:   return m_sfDQNMCTS != nullptr;
    case AGENT_EVAB:      return m_sfEVAB != nullptr;
    case AGENT_SACAZ:     return m_sfSACAZ != nullptr;
    case AGENT_SACAZ_MOE: return m_sfSACAZMoe != nullptr;
    case AGENT_SACAZ_MOE_MLP: return m_sfSACAZMoeMlp != nullptr;
    case AGENT_SACAZ_OLD: return m_sfSACAZOld != nullptr;
    case AGENT_SACAZ_OLD_MOE: return m_sfSACAZOldMoe != nullptr;
    case AGENT_DQNAB:  return m_sfDQNAB != nullptr;
    /* DQN+MCTS (MoE+TB 专家): 独立类, 有自己的常驻实例 */
    case AGENT_DQNMCTS_MOE: return m_sfDQNMCTSMOE != nullptr;
    case AGENT_PPOMCTS_MLP: return m_sfPPOMCTSMLP != nullptr;
    default:              return false;
    }
}

bool ChessBoard::saveCurrentAgentModel(AgentType agentType, const std::string &filepath)
{
    /*
       保存也可能很慢 (稀疏 MoE 骨干 28.7 M 参数, 权重按文本编码时是几百 MB), 所以
       同样报一次"请稍候"。用 RAII 保证**任何返回路径**都会发 busyFinished ——
       以前手工在每个 return 前收尾的话, 漏一个分支就是"弹窗永远挂着"。
    */
    struct BusyGuard {
        ChessBoard *self;
        explicit BusyGuard(ChessBoard *s, const QString &what) : self(s)
        {
            emit self->busyStarted(QStringLiteral("正在保存"), what);
        }
        ~BusyGuard() { emit self->busyFinished(); }
    } guard(this, QStringLiteral("写出 %1 的权重…")
                       .arg(agentDisplayName(agentType)));

    /*
       ================================================================
        ---- 必须与"决策"和"后台训练"串行 (2026-09, 用户报的崩溃) ----
       ================================================================
       这条路径原来是**裸的**: 它直接从调用方的线程去序列化主 agent 的网络。
       而同时对同一个网络动手的还有两处, 都规规矩矩拿着 `m_agentMutex`:
         * AI / 对弈线程: aiThink* 的 RL 分支 (整段决策都在锁内);
         * 后台训练线程: 每轮开头的 `saveModel(_temp_train*)` 与结尾的
           `loadModel(_temp_train*)` 同步回主 agent。
       于是"对弈结束 → 静默保存权重"正好撞上"后台训练正在把新权重写进同一个网络":
       一个是 `Net::save` 在遍历层、逐个张量编码, 另一个是 `Net::load` 在往那些张量里
       写 —— 数据竞争, 表现就是**保存到一半崩掉** (用户报的就是这个)。
       现在这条路径也进同一把锁: 保存期间决策与训练会等它 (一次 558 MB 权重的写入
       约 9 s, 这在"对局刚结束"的时刻不挡任何人的操作), 但谁都别想同时改那张网。
       调用方 (MainWindow 的保存线程 / shutdownSave) 都不持有这把锁, 所以不会自锁。
    */
    std::lock_guard<std::mutex> agentLock(m_agentMutex);
    /*
       [诊断] 拿到锁 = 从这一刻起到写盘结束, **任何** AI 决策/自检/训练都要排队。
       用户报障"黑方无限等待"如果指向这里, 日志会显示:
           saveCurrentAgentModel: 已拿到 agent 锁, 开始写盘 ...
           saveCurrentAgentModel: 写盘结束 —— waited N ms
       而 AI 那边会显示 "aiThinkRaw: 拿到 env 锁 —— waited M ms", M 就是被挡的时间。
    */
    const double tSave = dbgNowMs();
    dbgLog(QStringLiteral("saveCurrentAgentModel: 已拿到 agent 锁, 开始写盘 (agent=%1)")
               .arg(agentDisplayName(agentType)));
    /* RAII: 无论从哪个分支 return, 都会记下这次"持锁多久" */
    struct SaveTimer {
        double t0;
        const char *who;
        ~SaveTimer()
        {
            dbgWait(QStringLiteral("saveCurrentAgentModel: 写盘结束 (agent=%1)")
                        .arg(QString::fromUtf8(who)), dbgNowMs() - t0);
        }
    } saveTimer{tSave, agentDisplayName(agentType).toUtf8().constData()};

    /*
       ---- 类型分派走 saveWeightsOf (P0-b 收尾: 与"冻结快照的取数/审计"共用一份) ----
       原来这里是一个 11 分支的 switch, 而"把指定实例的权重写出去"这件事现在有**三个**
       调用方 (本函数 / 冻结快照 / 逐字节审计) —— 三份 switch 意味着"以后加一个 agent
       类型"时漏掉一处就是静默失效 (保存说成功、文件却没写)。所以口径统一到
       saveWeightsOf: 它只认"实例 + 类型 + 路径", 不关心实例是不是常驻的那一个。
       注意 filepath 的语义**仍按类型** (单文件 / 前缀), 这一点由各 agent 的 save* 决定,
       与 defaultWeightPath 的约定一致。
    */
    return saveWeightsOf(agentInstance(agentType), agentType, filepath);
}

/*
 * loadAgentModel - 把权重装进常驻实例 (saveCurrentAgentModel 的反向操作)
 *
 * 用途见 chessboard.h 的说明: "评估对局"要求两次实验之间能还原到同一个出发点,
 * 因为 matchAgents 会让常驻 agent 就地更新。
 *
 * 三条约定, 每条都是为了不留下静默失效:
 *   1. **实例不存在就先建**: 用与 aiThinkRaw / aiThinkForAgentRaw 的分支**逐字相同**的
 *      构造参数。写错一个参数 (骨干/宽度) 不会报错, 而是让 load 因结构指纹失败 ——
 *      那时本函数返回 false, 调用方必须当失败处理 (绝不"当作成功继续跑")。
 *   2. **纯搜索 agent 返回 false**: AB 各档 / MCTS 没有权重可装。用 abDepthOf() 判,
 *      这样以后再加 AB 档位也不会漏 (与 backgroundTrainLoop 同一判据)。
 *   3. 与"决策/训练/保存"共用 `m_agentMutex`: loadModel 会往网络里写整份权重,
 *      与搜索/保存抢同一批张量就会崩 (2026-09 用户报的那个崩溃, 见 saveCurrentAgentModel)。
 */
bool ChessBoard::loadAgentModel(AgentType agentType, const std::string &filepath)
{
    if (abDepthOf(agentType) > 0 || agentType == AGENT_MCTS) {
        return false;      /* 纯搜索 agent: 没有参数可载 */
    }
    std::lock_guard<std::mutex> agentLock(m_agentMutex);
    switch (agentType) {
    case AGENT_PG: {
        if (m_sfPG == nullptr) { m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f); }
        return m_sfPG->loadPolicy(filepath);
    }
    case AGENT_DQN: {
        if (m_sfDQN == nullptr) { m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f); }
        return m_sfDQN->loadModel(filepath);
    }
    case AGENT_PPOMCTS: {
        if (m_sfPPOMCTS == nullptr) {
            m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
            applyPpoBalance(*m_sfPPOMCTS);
        }
        return m_sfPPOMCTS->loadModel(filepath);
    }
    case AGENT_PPOMCTS_MLP: {
        /* 构造参数必须与 aiThinkRaw 那一支逐字一致 (含骨干), 否则结构指纹对不上 */
        if (m_sfPPOMCTSMLP == nullptr) {
            m_sfPPOMCTSMLP = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                              true, RL::PPO::Backbone::MlpExperts);
            applyPpoBalance(*m_sfPPOMCTSMLP);
        }
        return m_sfPPOMCTSMLP->loadModel(filepath);
    }
    case AGENT_DQNMCTS: {
        if (m_sfDQNMCTS == nullptr) {
            m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
        }
        return m_sfDQNMCTS->loadModel(filepath);
    }
    case AGENT_EVAB: {
        if (m_sfEVAB == nullptr) {
            m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
        }
        return m_sfEVAB->loadModel(filepath);
    }
    case AGENT_SACAZ: {
        if (m_sfSACAZ == nullptr) {
            m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
        }
        /* filepath 是前缀 -> _actor / _q1 / _q2 */
        return m_sfSACAZ->loadModel(filepath);
    }
    case AGENT_SACAZ_MOE: {
        if (m_sfSACAZMoe == nullptr) {
            m_sfSACAZMoe = createSACAZMoETbAgent(env, false, m_dynamicReward);
        }
        return m_sfSACAZMoe->loadModel(filepath);
    }
    /* [2026-09 独立类] MoE+MLP 那一支: 自己的实例 + 自己的 loadModel (shared 口径 4 个文件) */
    case AGENT_SACAZ_MOE_MLP: {
        if (m_sfSACAZMoeMlp == nullptr) {
            m_sfSACAZMoeMlp = createSACAZMoEMlpAgent(env, false, m_dynamicReward);
        }
        return m_sfSACAZMoeMlp->loadModel(filepath);
    }
    case AGENT_SACAZ_OLD: {
        /* 必须走 createSACAZLegacyAgent: 手写 new SACAZAgent 会静默退回当前口径 */
        if (m_sfSACAZOld == nullptr) {
            m_sfSACAZOld = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD);
        }
        return m_sfSACAZOld->loadModel(filepath);
    }
    case AGENT_SACAZ_OLD_MOE: {
        if (m_sfSACAZOldMoe == nullptr) {
            m_sfSACAZOldMoe = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD_MOE);
        }
        return m_sfSACAZOldMoe->loadModel(filepath);
    }
    case AGENT_DQNAB: {
        if (m_sfDQNAB == nullptr) {
            m_sfDQNAB = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                       DQNABAgent::Backbone::SparseMoeTb);
            m_sfDQNAB->nodeBudget = DQNAB_NODES;
        }
        return m_sfDQNAB->loadModel(filepath);
    }
    case AGENT_DQNMCTS_MOE: {
        /*
           [2026-09 dev-dqnmcts-moetb] 独立类 DQNMCTSMOETbAgent。
           构造参数与 aiThinkRaw / decideOnEnvRawLocked / makeAgentInstance / 后台训练
           那几处**逐字一致** (见本函数头注释), 预算也一起定下来 (selectMove 传 0 时用成员值)。
        */
        if (m_sfDQNMCTSMOE == nullptr) {
            m_sfDQNMCTSMOE = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                                   1.0f, 1.5f);
            m_sfDQNMCTSMOE->simulations = DQNMCTS_MOE_SIMS;
        }
        /* filepath 是前缀 -> _trunk / _q */
        return m_sfDQNMCTSMOE->loadModel(filepath);
    }
    default:
        return false;
    }
}

/* ================================================================
 *  后台训练线程
 *
 *  克隆当前agent, 在独立棋盘上持续自我对弈训练.
 *  每4轮训练后将权重同步回主agent.
 *  每次训练前从主棋盘复制当前棋局状态.
 * ================================================================ */

void ChessBoard::startBackgroundTraining()
{
    if (m_bgTraining) return;
    m_bgTraining = true;
    m_bgTrainThread = std::thread(&ChessBoard::backgroundTrainLoop, this);
}

void ChessBoard::stopBackgroundTraining()
{
    m_bgTraining = false;
    if (m_bgTrainThread.joinable()) {
        /*
           注意: 训练线程只在**每轮**开始时检查 m_bgTraining, 而一轮是"克隆权重 ->
           训练 BG_TRAIN_EPISODES 局", 所以关窗的等待时间由单轮时长决定。
           轮次规模就是按这个约束选的 (见 BG_TRAIN_* 常量), 不要随手调大。
           想亲自量这个等待时间, 用 setBackgroundTrainRound() 把一轮缩小即可。
        */
        m_bgTrainThread.join();
    }
}

void ChessBoard::setBackgroundTrainRound(int episodes, int maxMoves)
{
    m_bgTrainEpisodes = (episodes > 0) ? episodes : 1;
    m_bgTrainMaxMoves = (maxMoves > 0) ? maxMoves : 1;
}

/*
 * pauseBackgroundTraining - 让后台训练**停止改动主 agent 的权重** (P0-b)
 *
 * 为什么是"停止改动权重"而不是"停掉线程": 后台训练的一轮是
 *     克隆权重 -> 独立棋盘上自对弈 -> 训好写回临时文件 -> **同步回主 agent**
 * 而"同步回主 agent"那一步才是会污染评估的那件事。线程本身停不停无所谓。
 *
 * 做法: 置 m_bgPaused, 然后**等训练线程确认它不再持有主 agent 的锁**。
 * 等锁这一下是必要的: 否则"对局开始"与"某一轮的收尾同步"可以交错 —— 那一轮在
 * 暂停之前就已经在飞, 它会在对局中途把权重换掉 (这正是要拦的东西)。
 *
 * 死锁分析 (为什么从对弈线程调它是安全的):
 *   * 对弈线程不持有 m_agentMutex 时调它 (调用点在 matchAgents 开场, 见那里的注释);
 *   * 训练线程要么还没拿到锁 (立刻放行), 要么正持有锁做一次 loadModel —— 它会很快放锁,
 *     然后看到 m_bgPaused 并去 wait (不再开始新一轮)。
 *   * 训练线程**不会**在持有 m_agentMutex 的时候等条件变量 (见 backgroundTrainLoop
 *     里 pause 检查的位置), 所以不存在"我等你放锁、你等我 resume"的环。
 */
void ChessBoard::pauseBackgroundTraining()
{
    m_bgPaused = true;
    {
        /* 拿一次锁就够了: 能拿到 = 当前没有人在写主 agent 的权重 */
        std::unique_lock<std::mutex> lock(m_agentMutex);
    }
    /* 通知训练线程醒来去 wait (它可能正等在 wait_for 的超时上) */
    m_bgPauseCv.notify_all();
}

void ChessBoard::resumeBackgroundTraining()
{
    {
        std::lock_guard<std::mutex> lock(m_bgPauseMutex);
        m_bgPaused = false;
    }
    m_bgPauseCv.notify_all();
}

/* ============================================================================
 *  ---- 行为克隆 (BC) 的界面入口 (2026-10) ----
 * ============================================================================
 * 五件事 (顺序即实现顺序):
 *   1. bcSupported  : 哪些 agent 有 BC 路径 (PPO 两支 + SAC 三支);
 *   2. runBcOn      : 一次运行 (四种 agent 类共用; 流程在 src/bcrun.hpp);
 *   3. behaviorCloningWorker: 后台线程主体 (独占 agent + 暂停后台训练 + 发信号);
 *   4. startBehaviorCloning / bcReportText: 界面调的那两个入口。
 *
 * 为什么这些放在 chessboard.cpp 而不是 mainwindow.cpp: 独占 agent 的两个机制
 * (`m_agentMutex` 与 `pauseBackgroundTraining`) 都在这一侧。理由与三条行为约定
 * 见 chessboard.h 里 startBehaviorCloning 上面那一整段。
 */

bool ChessBoard::bcSupported(AgentType type)
{
    switch (type) {
    case AGENT_PPOMCTS:
    case AGENT_PPOMCTS_MLP:
    case AGENT_SACAZ:
    case AGENT_SACAZ_MOE:
    case AGENT_SACAZ_MOE_MLP:
        return true;
    default:
        /*
           剩下的类型各有**明确**的理由, 而不是"还没做"含糊过去:
             * Alpha-Beta 四档 / MCTS : 没有可训练参数 (没有策略头可克隆);
             * PG / DQN / DQN+MCTS / DQN+AB / EVAB: 有策略头, 但本轮只把 BC 接到了
               PPO 与 SAC 两条线 (PG/DQN 已从界面下拉框移除, EVAB/DQN+AB 的 BC 口径
               需要各自单独定义 —— 见 docs/behavior_cloning_2026_10.md 的"边界");
             * SAC+AZ-59e5233 两支: 行为还原版, 按本工程惯例不许新机制渗进去。
        */
        return false;
    }
}

QString ChessBoard::bcReportText() const
{
    std::lock_guard<std::mutex> lock(m_bcMutex);
    return m_bcReport;
}

/* ============================================================================
 *  ---- 行为克隆 (BC): 与 AB 对弈时"顺便"克隆 (2026-10) ----
 * ============================================================================
 * 用户口径: **"在与 abagent 对弈时可另外勾选行为克隆训练, 不另外单独做行为克隆训练"**。
 * 接口与三条行为约定的说明在 chessboard.h; 这里只记实现上的四个决定:
 *
 *   1. **样本在 moveForward 之前取**: 那一刻棋盘还是"对手将要走的那一步"的**原局面**,
 *      而 `step` 就是它真正要走出的那一手 —— 两者严格对应同一个局面, 这正是这条样本
 *      全部准确性的来源 (与 [O1] 的吃子读数、学习口径的即时奖励是同一个时序要求,
 *      三个都必须在同一处取, 否则它们描述的不是同一手棋)。
 *      做法: 在棋盘锁里面**拷一份** Chess (几 KB), 出了锁再去做 BC ——
 *      锁的顺序因此只有一种 (棋盘锁 -> agent 锁), 与 getAgentSelfCheck 一致。
 *   2. **每一条 AB 着法立刻更新一次**, 而不是攒够 64 条再更新: 一局只有几十手 AB 着法,
 *      攒批等于这一局白克隆。批 = 缓冲区里最近 `kBcBatch` 条 (不足就用全部)。
 *   3. **只在"训练对局"模式下生效**: 评估/只对弈声称"权重不变", 而 BC 是学习 ——
 *      让它在那两种模式下偷偷改权重, 正是 P0-b 修掉的那类静默失效。
 *   4. **不额外占时间**: 更新发生在"对手走完、下一手决策之前"的空档里, 只短暂持有
 *      agent 锁; 一次更新在 MLP 骨干上是毫秒级 (TB 骨干几十毫秒), 相对一步决策
 *      (0.3~8 s) 可以忽略。
 */

/* 这几个刻度放在文件里 (与其它 BG_* 同一手法), 改它们等于改学习问题 */
namespace {
constexpr int    kBcBatch     = 32;      /* 一次更新的批 (缓冲区不足时用全部) */
constexpr std::size_t kBcBufCap = 512;   /* 样本缓冲上限 (FIFO; 512 x 1710 x 4B ≈ 3.5 MB) */
constexpr float  kBcLr        = 0.002f;  /* 与命令行工具 train_bc 的默认一致 */
constexpr int    kBcReportEveryUpdates = 8;   /* 面板上每几次更新打一行 (第一次必打) */
/*
 * ---- 保真度采样 (那条"行为克隆"曲线) ----
 * `kBcFidEvery`: 每几次 actor 更新算一次一致率/CE。
 * `kBcFidWindow`: 在"最近多少条样本"上算 —— 曲线量的是**当前窗口**的保真度, 不是
 *   全程平均 (全程平均会把"早期很烂"永久混在里面, 趋势就看不出来了)。
 * `kBcFidMin`: 窗口至少几条才开始算: 1 条样本上的一致率只有 0% 或 100%, 那是噪声,
 *   画上去会被读成"一开始就很差/很好"。
 * 代价: 一次 evaluate = 窗口条数的前向 (≤64 次, MLP 骨干约 10~30 ms), 每 4 次更新一次
 *   ⇒ 相对一步决策 (0.3~8 s) 可以忽略。
 */
constexpr int    kBcFidEvery  = 4;
constexpr std::size_t kBcFidWindow = 64;
constexpr std::size_t kBcFidMin    = 8;
} // namespace

bool ChessBoard::bcHasStudent(AgentType a, AgentType b)
{
    /*
       只看"有没有学生"。老师不再由对手充当 (2026-10 用户口径: 下拉框选 AB, 与对手
       无关), 所以"对手是不是 AB"这件事**完全不参与**判定 —— 两边都是 AB 时
       bcSupported 全为假, 照样返回 false (没有学生就没有可克隆的策略头)。
    */
    return bcSupported(a) || bcSupported(b);
}

/*
 * bcTeacherName - 深度 -> 显示名 (唯一来源)
 *
 * 界面下拉框、对局报告、模型自检面板全都用这一份: 三处各写一遍的话, 迟早出现
 * "下拉框写着 L2 而报告里印着深3"这种**看起来像功能坏了**的不一致。
 * 名字本身取自 agentDisplayName (与 A/B 下拉框同一份), 所以选老师的下拉框与
 * 选对手的下拉框用的是同一套叫法。
 */
QString ChessBoard::bcTeacherName(int depth)
{
    switch (depth) {
    case AB_L1_DEPTH: return agentDisplayName(AGENT_AB_L1);
    case AB_L2_DEPTH: return agentDisplayName(AGENT_AB_L2);
    case AB_L3_DEPTH: return agentDisplayName(AGENT_AB_L3);
    case AB_DEPTH:    return agentDisplayName(AGENT_ALPHABETA);
    default:
        /* 0 = 关; 其它值不该出现 (下拉框只给这四个) —— 但不要静默: 报告里要看得出来 */
        return QStringLiteral("(未选老师)");
    }
}

void ChessBoard::bcResetMatchStats(const QString &agentLabel, const QString &teacherLabel)
{
    /*
       ⚠ 这里**不清 `m_bcHumanSession`**: 那个人机侧的"一局一次"标志的清除点是
       setHumanGameInProgress(false) (终局 / 重开), 而不是每次归零。混在一起的话,
       人机那条路会在"第一次采样归零"之后又被谁重置一次标志 ⇒ 每手都归零一次 ⇒
       读数永远只有 1 条样本 (看起来像"克隆没生效")。
    */
    m_bcBuf.clear();
    m_bcSamples = 0;
    m_bcUpdates = 0;
    m_bcMissed = 0;
    m_bcLastCe = -1.0;
    m_bcFirstCe = -1.0;
    m_bcLastTop1 = -1.0;
    m_bcLastPTeacher = -1.0;
    m_bcLastFidCe = -1.0;
    m_bcLastWindow = 0;
    /* 软目标那条路的累计量也要清 (否则报告里的 targetH 会跨场平均) */
    m_bcTargetEntropySum = 0.0;
    m_bcSoftSamples = 0;
    m_bcMatchAgent = agentLabel;
    m_bcMatchTeacher = teacherLabel;
    /* 在线训练那条路的基线 (见 m_bcOnlineBase 的说明): onlineSteps = 当前 - 基线 */
    m_bcOnlineBase = m_trainSampleNo.load();
    std::lock_guard<std::mutex> lock(m_bcMutex);
    m_bcReport.clear();
}

/*
 * 一条样本 -> 一次 actor 更新 (学生自己的局面 + 下拉框那一档 AB 的着法当标签)。
 *
 * **模板**: 四种 agent 类 (PPO 两支共用 PPOMCTSAgent; SAC 三支各一个类) 的
 * sampleFrom/update 是不同的重载, 所以由调用方按具体类型分发 (见 bcOnStudentMove)。
 */
template <class AgentT>
void ChessBoard::bcCollectFromStudent(AgentT *ag, const Chess &pos, int color,
                                      const Step &teacher,
                                      const std::vector<Step> &softMoves,
                                      const std::vector<float> &softProbs)
{
    /*
       ⚠ `sampleFrom` 会把 agent 自己的棋盘引用改成它正在编码的局面 (agent 的编码器
       读的是自己那份 Chess) —— 而那份就是本棋盘的 `env`。跑完要还原: 不还原的话,
       下一个决策虽然会 `env = chess` 覆盖掉它, 但"BC 顺手换掉了 agent 手里的棋盘"
       这件事一旦在别的路径上暴露, 表现是"AI 在一个不存在的局面上下棋", 极难查。
    */
    Chess boardBefore = ag->chess;

    RL::BCSample s;
    /* 软目标时用多老师重载 (分布); 否则 one-hot。两条路的合法性闸门是同一份实现。 */
    const bool ok = softMoves.empty()
                        ? BC::sampleFrom(*ag, pos, color, teacher, s)
                        : BC::sampleFrom(*ag, pos, color, softMoves, softProbs, s);
    ag->chess = boardBefore;
    if (!ok) {
        /*
           没做成样本的两种情形都**不是**崩溃, 但都不能静默: 要么这个局面没有合法
           着法 (终局), 要么老师的着法不在合法集里 (视角/索引不一致 —— 那是缺陷)。
           记进 m_bcMissed, 报告里会印出来。
        */
        m_bcMissed++;
        return;
    }

    if (m_bcBuf.size() >= kBcBufCap) {
        m_bcBuf.erase(m_bcBuf.begin());
    }
    m_bcBuf.push_back(s);
    m_bcSamples++;
    /*
       [2026-10] 软目标那条路上顺手累加**老师侧**的目标熵 H(t) (报告里要用):
       软目标的 CE 下界是 H(t) (CE = H(t) + KL(t‖π)), 不印 H(t) 的话"CE 变大了"这句话
       就没法解读 —— 是学生学差了, 还是老师本身就模糊? 这里记的是**样本上**的 H(t)
       (与 `train_bc` 报告里那个"标签"行同一口径: 同一条样本的同一个量)。
    */
    if (!softMoves.empty()) {
        m_bcTargetEntropySum += BC::targetEntropyOf(s);
        m_bcSoftSamples++;
    }

    const std::size_t take = std::min<std::size_t>((std::size_t)kBcBatch, m_bcBuf.size());
    std::vector<RL::BCSample> batch(m_bcBuf.end() - (long)take, m_bcBuf.end());
    const BC::UpdateStat u = BC::update(*ag, batch, kBcLr);
    if (!std::isfinite(u.ce)) {
        /* 这一批一条都没用上 (目标全落空 / 结构不合格): 只记不计 */
        return;
    }
    if (m_bcFirstCe < 0.0) {
        m_bcFirstCe = (double)u.ce;
    }
    m_bcLastCe = (double)u.ce;
    m_bcUpdates++;

    /*
       面板上的实时一行 (每 kBcReportEveryUpdates 次一次, 第一次必打)。
       刻意留 ASCII 记号 (samples= / updates= / CE= / teacher-depth=): verify_bc_ui.ps1
       要用 UIA 断言它 —— 脚本必须是 ASCII-only (无 BOM 的 .ps1 会被按 ANSI 解码)。
       `teacher-depth=` 这一项 (2026-10) 是给"报告里的老师是不是下拉框选的那一档"用的:
       面板在**对局进行中**只会追加这一行 (整段刷新要等自检刷新时机), 所以老师档位必须
       出现在这一行里 —— 否则脚本只能等到对局结束才断言得到 (第一版就是这么假失败的)。
    */
    if (m_bcUpdates == 1 || (m_bcUpdates % kBcReportEveryUpdates) == 0) {
        emit bcProgress(QStringLiteral("[BC] 克隆 AB 的着法: samples=%1 updates=%2"
                                       " 批=%3 CE=%4 (目标落空 %5) [teacher-depth=%6 soft=%7]")
                            .arg(m_bcSamples).arg(m_bcUpdates).arg((qulonglong)take)
                            .arg(m_bcLastCe, 0, 'f', 4).arg(m_bcMissed)
                            .arg(m_bcTeacherDepth.load())
                            .arg(m_bcSoftTarget.load() ? 1 : 0));
    }
    /*
       ---- 报告正文也刷新一次 (2026-10) ----
       人机那条路没有"场结束"这个时机, 而面板在别的时机 (选中 agent / 每一手探索后 /
       每场结束) 会**整段刷新**(setPlainText) —— 只在场末写报告的话, 人机里勾了克隆
       也读不到任何读数 (用户看到的就是"勾了没用")。正文只有几行字符串拼接, 相对一次
       actor 更新 (毫秒级) 可以忽略。
    */
    {
        std::lock_guard<std::mutex> lock(m_bcMutex);
        m_bcReport = bcComposeReport();
    }

    /*
       ---- 保真度采样: "克隆得像不像老师" (曲线与图下读数用) ----
       与上面那行**文本进度**是两件事: 进度行说的是"练到第几次", 这里说的是"练得怎么样"。
       四个量的口径与命令行 `train_bc` 完全一致 (同一个 `BC::evaluate`):
         * 一致率 = 策略头单独选中的着法 == 老师那一手 的比例 —— 这是 BC 的**训练目标本身**,
           也是唯一能回答"克隆到底成不成功"的数 (它上升**不是**棋力证据, 报告里写明了);
         * P(老师着法) = 策略头给老师那一手的平均概率 (0~1, 画图时 ×100);
         * CE = 同一个合法集掩码口径下的批平均交叉熵;
         * 窗口 = 最近 `kBcFidWindow` 条样本 (量的是**当前**水平, 不是全程平均)。
    */
    if (m_bcUpdates > 0 && (m_bcUpdates % kBcFidEvery) == 0
        && m_bcBuf.size() >= kBcFidMin) {
        const std::size_t w = std::min<std::size_t>(kBcFidWindow, m_bcBuf.size());
        std::vector<RL::BCSample> window(m_bcBuf.end() - (long)w, m_bcBuf.end());
        BC::Metrics fm;
        BC::evaluate(*ag, window, fm);
        if (fm.n > 0) {
            m_bcLastTop1 = fm.top1Pct;
            m_bcLastPTeacher = fm.pTeacher;
            m_bcLastFidCe = fm.ce;        /* 注意: **不是** m_bcLastCe (那是训练批的口径) */
            m_bcLastWindow = fm.n;
            emit bcFidelitySample((int)m_bcUpdates, fm.top1Pct, fm.pTeacher, fm.ce, fm.n);
        }
    }
}

void ChessBoard::bcOnStudentMove(AgentType studentType, const Chess &pos, int color)
{
    const int depth = m_bcTeacherDepth.load();
    if (depth <= 0) {
        return;
    }
    /*
       ---- 老师那一手: 在**局面副本**上现场搜一次 ----
       为什么用副本: ABAgent 的搜索会真的 moveForward/moveBack (它会改 `history`,
       见 aiThinkRaw 里那段 AB 分支的说明) —— 直接拿 `pos` 去搜, 就会把搜索过程写进
       这条样本的**状态编码**里 (编码要看规则历史), 样本当场被污染, 而且看不出来。
       所以搜索用另一个副本, `pos` 保持"学生要落子前那一刻"的原样。
       为什么**不加锁**: 搜索只用这个局部副本、不碰 agent、也不碰共享的 `env`,
       所以它不属于"棋盘锁 / agent 锁"那一对临界区; 深度 1~4 实测 0~90 ms 一次。
    */
    Chess teacherBoard = pos;
    Step teacher;
    /*
       ---- [2026-10] 软目标 (用户问题: "不直接使用 onehot、通过 abagent 计算概率分布
            再进行行为克隆是否会更好?") ----
       打开时老师给出的是**多深度一致性的分布** (深度 1..D 各投一票), 而不是一只手。
       为什么用"多深度一致性"而不是"根分值 Boltzmann": AB 的根循环是窗口写法, 非最优
       孩子的返回是**界**而不是精确分值 ⇒ 拿它做 softmax 等于在裁剪 artifact 上克隆
       (理由与实测都在 bcagent.hpp 的 softTargetFromAB 与文档 §9)。
       实测结论 (4000 局面 / 深度 3 / 8 epoch / **4 个种子**成对比较, 见文档 §9): 两个指标
       **方向相反** —— 可比口径 **留出 KL (CE − H(t)) 4/4 都更低** (2.379→1.866), 硬口径
       **留出 top-1 略低** (38.59%→36.78%, 3/4 个种子且符号不一致 ⇒ 只能算"没变好");
       `train−留出` 差 4/4 都更小 (0.427→0.280) 且策略熵不塌; 代价约 +12~14% 打标签时间。
       所以它是**可选开关**, 默认关 (与"动态奖励/对手入训"同一口径)。
    */
    std::vector<Step> softMoves;
    std::vector<float> softProbs;
    if (m_bcSoftTarget.load()) {
        if (!BC::softTargetFromAB(teacherBoard, color, depth, depth, softMoves, softProbs)) {
            m_bcMissed++;
            return;
        }
        teacher = softMoves.front();        /* 报告/落空计数仍用最深那一层的主着法 */
    } else {
        ABAgent ab(teacherBoard, depth);
        teacher = ab.getBestMove(color);
    }
    if (!teacher.valid) {
        /* 这个局面没有合法着法 (终局) —— 与"目标落空"合并计数, 报告里会印出来 */
        m_bcMissed++;
        return;
    }

    AgentBase *base = agentInstance(studentType);
    if (base == nullptr) {
        /* 学生还没有实例 (懒建): 这一手当作没选老师 —— 建实例是决策路径的职责 */
        m_bcMissed++;
        return;
    }
    /*
       只短暂持有 agent 锁: 对弈线程在这一刻没有别的 agent 工作, 而自检 worker
       可能正好在读同一个 agent 的权重 (它是另一条线程) —— 那一层必须挡住。
    */
    std::lock_guard<std::mutex> lock(m_agentMutex);
    switch (studentType) {
    case AGENT_PPOMCTS:
        bcCollectFromStudent(static_cast<PPOMCTSAgent *>(base), pos, color, teacher, softMoves, softProbs);
        break;
    case AGENT_PPOMCTS_MLP:
        bcCollectFromStudent(static_cast<PPOMCTSAgent *>(base), pos, color, teacher, softMoves, softProbs);
        break;
    case AGENT_SACAZ:
        bcCollectFromStudent(static_cast<SACAZAgent *>(base), pos, color, teacher, softMoves, softProbs);
        break;
    case AGENT_SACAZ_MOE:
        bcCollectFromStudent(static_cast<SACAZMoETbAgent *>(base), pos, color, teacher, softMoves, softProbs);
        break;
    case AGENT_SACAZ_MOE_MLP:
        bcCollectFromStudent(static_cast<SACAZMoEMlpAgent *>(base), pos, color, teacher, softMoves, softProbs);
        break;
    default:
        break;      /* bcHasStudent 已经拦过了; 这里只是不让它静默走错路 */
    }
}

/*
 * bcSampleForHumanTurn - 人机那条路上的"学生走一手 -> 一条样本" (2026-10)
 *
 * 用户口径要的就是这个: **与人类棋手对弈时照常克隆** (老师来自下拉框, 与人无关)。
 * 对局循环 (playMatchGame) 里已经有一个"落子前拷好的局面", 人机这条路没有 ——
 * 所以在这里自己拷一份, 之后与对局那条路**共用同一段** bcOnStudentMove。
 *
 * 锁序: 先拿棋盘锁拷完就放, 再进 bcOnStudentMove 里的 agent 锁 —— 与对局循环、
 * getAgentSelfCheck 一样只有"棋盘 -> agent"这一种顺序。
 *
 * 三个闸门 (与对局那条路逐条对齐, 否则"人机能偷偷改权重"就会重演 P0-b 那个洞):
 *   * 老师深度 > 0 (下拉框不是"关");
 *   * 模式 = 训练对局;
 *   * 这一手的 AI 那一支**能做 BC 学生** (纯搜索的 AB/MCTS 没有策略头可克隆)。
 */
void ChessBoard::bcSampleForHumanTurn(AgentType studentType, int color)
{
    if (m_bcTeacherDepth.load() <= 0) {
        return;
    }
    if (m_matchMode.load() != MATCH_TRAIN) {
        return;
    }
    if (!bcSupported(studentType)) {
        return;
    }
    /*
       人机没有"场"的概念, 所以读数按**一局**归零: 第一次采样时归零一次
       (exchange 保证只有第一次为真), 终局/重开时由 setHumanGameInProgress(false) 清标志。
       不归零的话"这一局一条都没克隆"会带上上一局的读数 (那是假的读数);
       每手都归零的话读数永远只有 1 条样本 (看起来像"克隆没生效")。
    */
    if (!m_bcHumanSession.exchange(true)) {
        bcResetMatchStats(agentDisplayName(studentType),
                          bcTeacherName(m_bcTeacherDepth.load()));
    }
    Chess pos;
    {
        QMutexLocker locker(&mutex);
        pos = chess;
    }
    bcOnStudentMove(studentType, pos, color);
}

/*
 * 报告正文 (每做一次 actor 更新就刷新一次 -> 面板随时读得到当前读数)。
 *
 * 为什么抽成一个函数 (2026-10): 报告有两个消费时机 —— "对局结束" (对局那条路) 与
 * "面板每次刷新" (人机那条路**没有场结束**, 只在场末写的话面板一刷新就把读数冲掉,
 * 用户看到的就是"勾了没用")。正文只写一份, 两个时机都调它, 读数不会分叉。
 *
 * 三条读数必须一起给, 缺一个就会被误读:
 *   * samples/updates: "克隆到底有没有在发生" (全是 0 就说明下拉框还停在"关"上);
 *   * CE 的变化 (第一条 -> 最后一条): "有没有在学" —— 只看最后一个 CE 没法判断方向;
 *   * targetMissed: **目标落空**的条数 (视角/索引不一致的信号), 非零必须显眼。
 */
QString ChessBoard::bcComposeReport() const
{
    if (m_bcMatchAgent.isEmpty() && m_bcSamples == 0 && m_bcMissed == 0) {
        return QString();       /* 本场没开这个功能: 不写报告 (面板上保留上一次的读数) */
    }
    QString text;
    /*
       ⚠ 行尾那个 `[teacher-depth=N]` 是**机器记号** (ASCII, 0 = 关): verify_bc_ui.ps1
       要断言"报告里印的老师就是下拉框选的那一档" —— 而档位的显示名是中文, 那个脚本
       必须 ASCII-only (无 BOM 的 .ps1 会被 Windows PowerShell 按 ANSI 解码, 代码里的
       中文会把脚本解析坏)。与自检面板里的 "MaxVio / Loss-Free" 同一个做法。
    */
    text += QStringLiteral("=== 对局中行为克隆 (BC): 学生=%1, 老师=%2 (下拉框选)"
                           " [teacher-depth=%3] ===\n")
                .arg(m_bcMatchAgent.isEmpty() ? QStringLiteral("(未记录)") : m_bcMatchAgent,
                     m_bcMatchTeacher.isEmpty() ? QStringLiteral("(未记录)") : m_bcMatchTeacher,
                     QString::number(m_bcTeacherDepth.load()));
    /*
       ⚠ 这一行里的 ASCII 记号 (match-summary / samples= / updates= / targetMissed= /
       onlineSteps=) 是**刻意**的: tools/verify_bc_ui.ps1 用 UIA 读面板文本断言它,
       而那个脚本必须 ASCII-only (无 BOM 的 .ps1 会被 Windows PowerShell 按 ANSI 解码,
       代码里的中文会把脚本解析坏)。与自检面板里的 "MaxVio / Loss-Free" 同一个做法:
       机器要读的记号用 ASCII, 其余照旧用中文。
       [2026-10] `onlineSteps=` = 这一场里**在线训练(探索+预训练)**上报过多少次损失。
       BC 与它是两个独立开关 (都能开), 所以"都开着"与"只有一个在跑"必须分得开 ——
       否则用户看到的 CE 趋势到底是谁造成的就没法判断。
       [2026-10 修] 这条报告原来把上面那句写在 emit 出去的一行里 —— 而面板在**对局结束时
       会整段刷新**(setPlainText), 那一行当场被冲掉 ⇒ 脚本读不到 "match-summary"。
       现在它放在**存进 bcReportText 的报告正文**里, 于是每次刷新都跟着回来。
    */
    text += QStringLiteral("本场汇总 (match-summary): samples=%1 updates=%2 targetMissed=%3"
                           " onlineSteps=%4 soft=%5\n")
                .arg(m_bcSamples).arg(m_bcUpdates).arg(m_bcMissed)
                .arg(std::max<long long>(0, m_trainSampleNo.load() - m_bcOnlineBase))
                .arg(m_bcSoftTarget.load() ? 1 : 0);
    /*
       ---- [2026-10] 软目标那一行: 老师给的分布有多"软" ----
       只在真的攒到软样本时印。`targetH` 是**老师侧**的量 (平均目标熵 H(t)):
         * 软目标下 CE 的下界 = H(t) (CE = H(t) + KL(t‖π)) —— 于是
           "CE 2.41 (软) vs 2.58 (one-hot)" 这类对比只在同 H(t) 下可比, 必须一起看;
         * H(t) = 0 只可能来自 one-hot (深度只有一层, 票全压在一手上);
         * H(t) 越大 = 各深度越不一致 = 这个局面上"老师自己都不确定",
           此时 CE 高**不代表**学生差 (KL 才是"像不像老师"的那部分)。
    */
    if (m_bcSoftSamples > 0) {
        const double h = m_bcTargetEntropySum / (double)m_bcSoftSamples;
        text += QStringLiteral("软目标 (soft targets, 多深度一致): 软样本=%1/%2 targetH=%3\n")
                    .arg(m_bcSoftSamples).arg(m_bcSamples).arg(h, 0, 'f', 4);
        /*
           ⚠ **深度 1 时软目标退化成 one-hot** (`softTargetFromAB` 只有一层可投票 ⇒
           唯一候选 p=1 ⇒ H(t)=0)。这不是缺陷 (它正是"深度 1 与旧口径逐位一致"那条保证),
           但如果报告里只印一个 `targetH=0`, 用户看到的就是"勾了软目标、读数却是 0" ——
           分不清"没生效"和"退化了"。所以这里明说原因与下一步动作。
        */
        if (h <= 1e-9 && m_bcTeacherDepth.load() <= 1) {
            text += QStringLiteral("  ⚠ 深度=1 时**只有一票** ⇒ 目标退化成 one-hot (H(t)=0), "
                                   "软目标此时与默认口径**完全一致**; 想真的用软目标请把老师"
                                   "选成 L2 或更深\n");
        }
    }
    /*
       ---- [2026-10] 保真度一行 (与"行为克隆"那条曲线同一口径) ----
       口径说明写在 `bcCollectFromStudent` 的采样处; 这里只把最后那个窗口的读数写进报告。
       `fidelity` 这个词刻意保留英文 (机器要读): verify_bc_ui.ps1 断言这一行。
       ⚠ top-1 是 BC 的**训练目标本身**, 报告末尾那三条边界里写着"它不是棋力证据" ——
         这一行也不例外 (docs/training_optimization.md §7.10: 一致率涨 9 倍而胜率没动)。
    */
    if (m_bcLastTop1 >= 0.0) {
        text += QStringLiteral("保真度 (fidelity, 最近 %1 条样本): top-1=%2% P(teacher)=%3% CE=%4\n")
                    .arg(m_bcLastWindow)
                    .arg(m_bcLastTop1, 0, 'f', 2)
                    .arg(m_bcLastPTeacher * 100.0, 0, 'f', 2)
                    .arg(m_bcLastFidCe, 0, 'f', 4);
    }
    if (m_bcUpdates > 0) {
        const double drop = (m_bcFirstCe > 0.0 && m_bcLastCe > 0.0)
                                ? (m_bcFirstCe / m_bcLastCe) : 0.0;
        /*
           ⚠ 这里必须用 Qt 的 %1/%2,**不是** printf 的 %.4f —— 写错的表现是运行期刷
           "QString::arg: Argument missing" 并且文本里留下一串 "%.4f" (本轮踩过)。
           命令行那一侧之所以是 %.4f: 那边走 std::snprintf。
        */
        text += QStringLiteral("批平均 CE: 第一条 %1 -> 最后一条 %2")
                    .arg(m_bcFirstCe, 0, 'f', 4).arg(m_bcLastCe, 0, 'f', 4);
        /*
           ⚠ 措辞必须**跟着方向走**: 第一条样本时的批只有 1 条 (批 = 缓冲区里最多 32 条),
           它的 CE 可能恰好很低 —— 那么"最后一条更小"这件事就不成立, 而报告里却写着
           "(降 0.32x)" 的话, 读起来像"学坏了"。所以 <1 时明说方向未定。
        */
        if (drop >= 1.0) {
            text += QStringLiteral(" (降 %1x)").arg(drop, 0, 'f', 2);
        } else {
            text += QStringLiteral(" (最后一条更大: 前几条是 1 条样本的批, 噪声大 —— "
                                   "一局的样本太少, 方向看多场累计)");
        }
        text += QStringLiteral("\n");
    } else {
        text += QStringLiteral("**一次 actor 更新都没做** —— 检查: 老师下拉框是不是还停在\"关\" / "
                               "模式是不是被切成\"评估 / 只对弈\"了 (那两种模式下本功能不生效)\n");
    }
    if (m_bcSoftTarget.load()) {
        text += QStringLiteral("口径: 只更新**策略头** (PPO 的 BC 路径整段不含 critic; "
                               "SAC 独立口径下 q1/q2 逐字节不变), 损失 = 合法集上的掩码交叉熵; "
                               "目标是**软分布** (深度 1..D 各投一票) ⇒ 其下界是 H(t), "
                               "看 CE 时必须与上面那行的 targetH 一起看\n");
    } else {
        text += QStringLiteral("口径: 只更新**策略头** (PPO 的 BC 路径整段不含 critic; "
                               "SAC 独立口径下 q1/q2 逐字节不变), 损失 = 合法集上的掩码交叉熵 "
                               "(目标是 one-hot ⇒ H(t)=0, CE 就是 NLL)\n");
    }
    text += QStringLiteral("[注] 只改了**内存里的权重**: 与既有约定一致, 唯一的落盘点是"
                           "退出程序时 (或你自己按保存)。\n");
    return text;
}

/*
 * 一场结束时: 把报告正文落进 m_bcReport, 并在对局明细里留一行。
 *
 * 另外写明"只改了内存里的权重" —— 否则用户会去 weights/ 找一个不存在的文件。
 * (正文本身在 bcComposeReport 里; 这里只管"落盘时机"与那一行明细。)
 */
void ChessBoard::bcFinishMatchReport(MatchStats &st)
{
    const QString text = bcComposeReport();
    if (text.isEmpty()) {
        return;         /* 本场没开这个功能: 不动报告 (面板上保留上一次的读数) */
    }
    {
        std::lock_guard<std::mutex> lock(m_bcMutex);
        m_bcReport = text;
    }
    /* 对局明细里也留一行 (用户回看逐局列表时看得到), 同时让面板实时更新一次。
       刻意带 ASCII 记号 (match-summary / samples= / updates= / targetMissed=):
       verify_bc_ui.ps1 要用 UIA 断言这一行, 而那个脚本必须是 ASCII-only
       (无 BOM 的 .ps1 会被 Windows PowerShell 按 ANSI 解码) —— 与自检面板里的
       "MaxVio / Loss-Free" 是同一个做法: 机器要读的记号用 ASCII。 */
    const QString oneLine = QStringLiteral("[BC] 本场汇总 (match-summary): samples=%1"
                                           " updates=%2 targetMissed=%3")
                                .arg(m_bcSamples).arg(m_bcUpdates).arg(m_bcMissed);
    st.log += QStringLiteral("  ") + oneLine + QStringLiteral("\n");
    emit bcProgress(oneLine);
}

/* ============================================================================
 *  ---- [2026-10] 价值评估 (EV/校准): 从每手的 V(s) 到一条曲线 ----
 * ============================================================================
 *
 * 用户口径: "在奖励窗口增加一个 tab 显示价值评估曲线"。
 *
 * 那条曲线画的是 **解释方差 EV**, 不是损失 —— 因为损失曲线 (critic MSE) 是最弱的一层
 * 证据: 它的目标是**自举**的 (TD/GAE 用 V(s')), 于是"损失在降"可以与"价值是错的"并存
 * (一个把 Q 学成常数、或者靠自举把材质当终局的网络, MSE 一样可以很低)。能与真实结果
 * 对照的两个量在 `rl/diag.h` 里早就有了 (`explainedVariance` / `calibration` /
 * `calibrationError`) —— `bench_diag` 用它们定位过一次真正的缺陷 (实测 **EV = −0.0293**
 * = "还不如常数预测", 283 个样本里 281 个落在同一个桶)。缺的只是**界面上的曲线**。
 *
 * 三个刻意的决定 (每条都对应一类"曲线会给出假读数"的风险):
 *
 *  1. **z 用真实胜负, 不用自举目标**: 每手记 (V, 走子方), 局末按胜负折算 (+1/0/−1,
 *     走子方视角)。用自举目标算 EV 是循环论证 —— "V 能预测 V 的目标"不含任何信息。
 *     折算口径抽成纯函数 (`RL::Diag::zFromGameResult`), 由 test_diag 钉住。
 *  2. **zVar ≈ 0 时不出点**: 全是和棋 (或台架截断判和) 时 EV 按定义无解, 而曲线上的 0
 *     看起来就是"V 和常数预测一样烂" —— 本工程第一次使用 EV 时踩的就是这个假 0。
 *  3. **滚动窗口而不是全程平均**: 量的是"现在准不准"。窗口**跨局保留** (与损失曲线一样
 *     是跨场连续的), 但每手都进窗口, 所以它天然是"最近这些手"。
 *
 * 代价: 每手一次 critic 前向 (MLP 骨干 ~0.14 ms, TB 骨干 ~3.6 ms), 相对一步决策
 * (0.3~8 s) 可以忽略, 而且它是**纯只读探针** (不写任何训练状态)。
 */

namespace {
/* 滚动窗口上限 (对数): 一局 ~60 手, 2000 对 ≈ 最近 30 局。 */
constexpr std::size_t kValueWindowCap = 2000;
/*
 * 出一个点至少要多少对样本。**32** (原为 64): 用户实测的典型配置是"PPO 和 AB 打 1~3 局",
 * 而**只采有 V 头那一方**的着手 ⇒ 一局 ~30 手对手 + ~30 手自己 ≈ 30 对, 两局才刚好 60 对
 * —— 于是"打开克隆训练后没有价值曲线"(用户报的)里有一半原因就是这个门槛。
 * 代价是噪声: n=32 时 ρ 的标准误差 ≈ 1/√32 ≈ 0.18, n=64 时 ≈ 0.125 ⇒ 读数行必须**把
 * pairs 一起写出来**(它已经不画在曲线上), 让人知道这个点有多可信。
 */
constexpr int kValueMinPairs = 32;
/* 折扣因子: 必须与 critic 的价值目标同一个 γ (PPOMCTSAgent 构造里的 0.99)。 */
constexpr double kValueGamma = 0.99;

/* Chess::RESULT_* -> z 的胜者口径 (0 = 红, 1 = 黑, −1 = 和/未定) */
int zWinnerOf(int result)
{
    if (result == Chess::RESULT_RED_WIN) {
        return Stone::COLOR_RED;
    }
    if (result == Chess::RESULT_BLACK_WIN) {
        return Stone::COLOR_BLACK;
    }
    return -1;
}
} // namespace

bool ChessBoard::valueProbeOf(AgentType type, Chess &pos, int color, double &out)
{
    AgentBase *base = agentInstance(type);
    if (base == nullptr) {
        return false;               /* 实例还没建 (懒建): 这一手不采 */
    }
    /*
       只有 PPO 两支有**标量 V 头** (critic)。SAC 三支的批评家是 Q(s,a) 而不是 V(s),
       EV/校准在那边要换一套口径 (目标不同), 所以这里明确返回 false —— 让界面保持
       "没有这条读数的 agent 就说明原因"的语义, 而不是塞一个含义不同的数进去。
    */
    switch (type) {
    case AGENT_PPOMCTS:
    case AGENT_PPOMCTS_MLP:
        return static_cast<PPOMCTSAgent *>(base)->valueOfPosition(pos, color, out);
    default:
        return false;
    }
}

void ChessBoard::valueDiagOnGameEnd(int result, int gameNo)
{
    /*
       两种 z 口径各进一份窗口 (2026-10, 用户实测"完成 4 轮对弈都没有曲线"之后补的):
         z1 = 真实胜负 (±1/0, 走子方视角) —— "V 能不能预测胜负";
         z2 = 本局最终学习口径回报 (连续量) —— "V 与 critic 真正在学的那个目标是否一致"。
       为什么两个都给: z1 只有三个取值, 被采样的那些手**结果符号全一样**时 Var(z1) = 0
       ⇒ EV 无定义 ⇒ 曲线刻意不出点 (那是对的, 但实测里很常见: 只采到一边的手 (对手不是
       PPO) 而它一路输 ⇒ 全是 −1 ⇒ 永远空)。z2 是连续量, 几乎总有方差, 于是曲线能用。
    */
    std::vector<double> zWin, zEng;
    /* z 口径 1 的 mover 序列: 从"整局数组"里按采样下标取出 (两个数组一一对应) */
    std::vector<int> sampledMovers;
    sampledMovers.reserve(m_valueGameSampleIdx.size());
    for (std::size_t s = 0; s < m_valueGameSampleIdx.size(); s++) {
        const int i = m_valueGameSampleIdx[s];
        if (i >= 0 && (std::size_t)i < m_valueGameStepMover.size()) {
            sampledMovers.push_back(m_valueGameStepMover[(std::size_t)i]);
        }
    }
    RL::Diag::zFromGameResult(m_valueGameV, sampledMovers, zWinnerOf(result), zWin);
    /*
       z 口径 2 = 该手之后的**折扣回报** (引擎口径, 每一手都有定义 ⇒ 与对手是谁无关)。
       γ 取 0.99 —— 与 `PPOMCTSAgent(env, 64, 0.99f, …)` 里那个 gamma 同一个数。
       ⚠ 它**不**是 V 的训练口径 (V 学的是学习口径、材质×0.1), 所以两者尺度不同 ⇒
         对这条 z 用的是**尺度无关**的 ρ (见 rl/diag.h 的 pearson): EV 在跨口径下会被
         尺度差吃满 (实测读到过 −24.3), 而 ρ 只问"同向吗"。
    */
    RL::Diag::zFromDiscountedReturnToGo(m_valueGameStep, m_valueGameStepMover,
                                        m_valueGameSampleIdx, kValueGamma, zEng);
    if (!zWin.empty()) {
        RL::Diag::appendWindow(m_valueWinV, m_valueWinZ, m_valueGameV, zWin,
                               kValueWindowCap);
        RL::Diag::appendWindow(m_valueWinV, m_valueWinZEng, m_valueGameV, zEng,
                               kValueWindowCap);
        m_valuePairs += (long long)zWin.size();
        m_valueGames++;
    }
    m_valueGameV.clear();
    m_valueGameSampleIdx.clear();
    m_valueGameStep.clear();
    m_valueGameStepMover.clear();

    const RL::Diag::ValueDiag dw = RL::Diag::valueDiagOf(m_valueWinV, m_valueWinZ,
                                                         kValueMinPairs, 10);
    /*
       第二条线用 **ρ(V, 引擎口径折扣回报)** 而不是 EV: 尺度无关, 所以"V 学的是学习口径、
       z 是引擎口径"这件事不再把读数毁掉 (那是用户实测"与 MCTS 对弈数值 < 0"的根因之一)。
       同时把 EV_eng 也算出来放进读数行 —— 谁想看尺度敏感的那个版本都有, 但曲线上画 ρ。
    */
    const RL::Diag::ValueDiag de = RL::Diag::valueDiagOf(m_valueWinV, m_valueWinZEng,
                                                         kValueMinPairs, 10);
    const bool rhoOk = de.enough && de.zHasVariance;
    const double rhoEng = rhoOk ? RL::Diag::pearson(m_valueWinV, m_valueWinZEng) : 0.0;
    /*
       ⚠ **每一局结束都发一次信号**, 哪怕算不出来 (值为 NaN, CurveChart 会丢掉不画点)。
       为什么: 用户实测"几轮对弈都没有曲线"时, 界面上**必须能看出为什么** —— 而"没采到
       样本(pairs 不足)"与"采到了但没有方差"是两件事, 下一步动作完全不同。
       原来把这种情况写成面板里的一行说明, 但**面板在对局结束时会整段刷新**(setPlainText),
       那一行当场被冲掉 —— 于是用户看到的就是"什么都没有"(实测踩到)。
       现在状态由**读数行**承载 (常驻控件, 不会被冲掉)。
    */
    const double nan = std::numeric_limits<double>::quiet_NaN();
    emit valueDiagSample(gameNo, dw.ok ? dw.ev : nan, rhoOk ? rhoEng : nan,
                         de.zHasVariance ? de.ev : nan,
                         dw.zHasVariance ? dw.calibErr : nan,
                         dw.n, dw.zVar, de.zVar);
    const QString nf = QStringLiteral("(n/a)");
    /*
       ⚠ 面板行**只在真的采到样本时才写** (`dw.n > 0`): 否则一场 AB-vs-EVAB 的百局对局会
       每局往面板扔一行 "pairs=0", 把面板刷满 —— 那和"每 2 秒刷一条警告"是同一类噪声,
       刷屏之后就不再有人看它了。曲线/读数行的状态仍然每局更新 (信号照发)。
    */
    if (dw.n > 0) {
        emit bcProgress(QStringLiteral(
            "[value] 价值评估: EV_win=%1 rho_eng=%2 EV_eng=%3 pairs=%4"
            " zVar_win=%5 zVar_eng=%6")
            .arg(dw.ok ? QString::number(dw.ev, 'f', 4) : nf)
            .arg(rhoOk ? QString::number(rhoEng, 'f', 4) : nf)
            .arg(de.zHasVariance ? QString::number(de.ev, 'f', 4) : nf)
            .arg(dw.n)
            .arg(dw.zVar, 0, 'f', 6).arg(de.zVar, 0, 'f', 6));
    }
}

/*
 * isBackgroundTrainingPaused - "后台训练此刻不许改主 agent 的权重吗" (P0-b 收尾)
 *
 * 两个来源, 取或:
 *   * m_bgPaused     : 显式 pause (matchAgents 用 RAII 包住整场, 见那里的注释);
 *   * 对弈模式 ≠ 训练: 评估/只对弈模式声称"冻结", 而冻结必须包含**权重不变**。
 *
 * 为什么模式这一条要做成**整体停摆**而不是"对局期间暂停一下": 见头文件里那段说明
 * (人机那条路不在 matchAgents 里; 局与局之间也在同步权重)。
 */
bool ChessBoard::isBackgroundTrainingPaused() const
{
    return m_bgPaused.load() || m_matchMode.load() != MATCH_TRAIN;
}

void ChessBoard::backgroundTrainLoop()
{
    QDir().mkpath("weights");
    /*
       "这一支没接后台训练"的消息**每种 agent 只报一次**:
       这个循环是每 2 秒重试一次的, 每轮都报的话就会看到一行警告每 2 秒刷一次 ——
       警告刷屏之后就不再是警告了 (EVAB 没接的那段时间正是这样: 那条"种子权重写入
       失败"每 2 秒刷一次, 反倒把真实原因淹掉了)。切到别的 agent 再切回来也不重报。
    */
    std::set<AgentType> notWiredWarned;
    /*
       "因为模式停摆"的消息同样**只报一次**: 这个循环每 200 ms 转一圈, 每圈都报的话
       日志会被同一句话刷满 (与上面那条"尚未接入"同一条理由)。
    */
    bool modeGateReported = false;

    while (m_bgTraining) {
        /*
           ---- 暂停闸门 (P0-b) ----
           等在这里而不是 sleep 轮询: 对局结束时会 notify (见 resumeBackgroundTraining)。
           ⚠ 条件变量的 wait **不能**在持有 m_agentMutex 时做 (那会与
             pauseBackgroundTraining 的"拿一次锁"互等, 见那里的死锁分析)。

           [P0-b 收尾] 闸门条件从"m_bgPaused"扩成 isBackgroundTrainingPaused():
           后者还把**对弈模式**算进来 —— 评估/只对弈模式下这条线程整体停摆, 于是在
           人机对战与局间这两个 matchAgents 管不到的时刻, 它也不会去改主 agent 的权重。
        */
        {
            std::unique_lock<std::mutex> pauseLock(m_bgPauseMutex);
            m_bgPauseCv.wait_for(pauseLock, std::chrono::milliseconds(200),
                                 [this] { return !isBackgroundTrainingPaused()
                                                 || !m_bgTraining; });
        }
        if (!m_bgTraining) {
            break;
        }
        if (isBackgroundTrainingPaused()) {
            /*
               区分"模式拦下的"与"显式暂停的": 只有前者要计数与报一次 —— 显式暂停是
               对弈期间每场都会发生的事, 报出来没有信息量。
            */
            if (m_matchMode.load() != MATCH_TRAIN) {
                m_bgRoundsSkippedByMode.fetch_add(1);
                if (!modeGateReported) {
                    modeGateReported = true;
                    qInfo().noquote()
                        << QStringLiteral("[train] 后台训练已按对弈模式停摆 (模式 = %1): "
                                          "本模式下不再把训练权重同步回主 agent —— "
                                          "评估/只对弈要的是**权重也不变**, 不只是"
                                          "'这一手不学'。切回'训练对局'即自动恢复。")
                               .arg(matchModeName(m_matchMode.load()));
                }
            }
            continue;      /* 不开始新一轮 (也不碰主 agent) */
        }
        modeGateReported = false;   /* 恢复后, 下一次停摆再报一次 */

        AgentType type = m_agentType;
        /* 本轮的规模 (默认 = BG_TRAIN_*, 测试可以调小, 见 setBackgroundTrainRound) */
        const int roundEpisodes = m_bgTrainEpisodes.load();
        const int roundMaxMoves = m_bgTrainMaxMoves.load();

        /*
           ---- 这一步的后台训练接没接? (2026-09) ----
           原来这里只写了一句"Alpha-Beta / MCTS 没有可训练权重就跳过", 而真正的分支
           判断散在下面三个 switch 里。于是 EVAB (界面上可选、也确实有在线训练) 在
           **三个 switch 里都没有 case**: seeded 永远是 false, 日志报的却是
           "[train] 种子权重写入失败 ... 路径 weights/_temp_train.dat" ——
           一条把"这一支根本没接"说成"文件写不出去"的误导性诊断 (用户报障)。
           现在把"有没有接入"和"写盘成不成功"分成两件事报:
             * 没接入 (还没为它写训练分支的 agent) -> 明说"尚未接入";
             * 接入了但写盘失败 -> 保留原来那条"种子权重写入失败"。
        */
        bool trainable = false;
        switch (type) {
        case AGENT_PG:
        case AGENT_DQN:
        case AGENT_PPOMCTS:
        case AGENT_DQNMCTS:
        case AGENT_EVAB:
        /* 2026-09: 补齐剩下三个有可训练权重的 agent; 后来又加了 59e5233 还原版的
           TB 专家骨干那一支 —— 至此**界面上的十一个 agent 里所有"有权重可训"的都
           接上了** (AB / MCTS 没有权重, 不在此列)。 */
        case AGENT_SACAZ:
        case AGENT_SACAZ_MOE:
        case AGENT_SACAZ_MOE_MLP:
        case AGENT_SACAZ_OLD:
        case AGENT_SACAZ_OLD_MOE:
        case AGENT_DQNAB:
        /* [2026-09 dev-dqnmcts-moetb] DQN+MCTS (稀疏MoE+TB专家): 有可训练权重, 已接入
           (它下面四个 switch 里都有 case —— 漏一个的表现是"该 agent 的后台训练尚未
           接入"或"种子权重写入失败", 见 4981 行那段说明) */
        case AGENT_DQNMCTS_MOE:
        case AGENT_PPOMCTS_MLP:
            trainable = true;
            break;
        default:
            break;
        }
        if (!trainable) {
            /*
               纯搜索 agent (Alpha-Beta 各等级 / MCTS) 是**预期**没有后台训练的:
               它们没有可训练权重, 落到这里不报警。
               [2026-09] 这里的判据从"枚举点名"改成 abDepthOf(type) > 0 —— 加了三档
               AB 等级之后, 再按枚举逐个点名就会漏掉新类型, 于是每次轮转都打一条
               "该 agent 的后台训练尚未接入"的假警告, 把真正的"忘了接线"淹掉。
            */
            if (abDepthOf(type) == 0 && type != AGENT_MCTS
                && notWiredWarned.insert(type).second) {
                /* 现在十一个 agent 里"有权重可训"的**全部**接上了, 所以这一支只有在
                   以后新增 agent 类型而忘了接线时才会响 —— 留着它就是为了那一天:
                   这条消息以前被"种子权重写入失败"顶替, 结果 EVAB 空转了很多轮而
                   没人发现。 */
                qWarning() << "[train] 该 agent 的后台训练尚未接入, 跳过本轮:"
                           << agentDisplayName(type);
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        /* 确保主agent已创建 (可能还未首次使用) */
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            switch (type) {
            case AGENT_PG:
                if (m_sfPG == nullptr)
                    m_sfPG = new PGEagent(env, 64, 0.9f, 0.01f, 1.0f);
                break;
            case AGENT_DQN:
                if (m_sfDQN == nullptr)
                    m_sfDQN = new DQNAgent(env, 64, 0.99f, 0.001f, 1.0f);
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS == nullptr) {
                    m_sfPPOMCTS = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f);
                    applyPpoBalance(*m_sfPPOMCTS);
                }
                break;
            case AGENT_PPOMCTS_MLP:
                /* 构造参数必须与 aiThinkRaw / aiThinkForAgentRaw 那一支逐字一致
                   (含骨干), 否则训练 clone 与主 agent 的结构指纹对不上, 每轮都载入失败 */
                if (m_sfPPOMCTSMLP == nullptr) {
                    m_sfPPOMCTSMLP = new PPOMCTSAgent(env, 64, 0.99f, 0.001f, 1.414f, 64,
                                                      0.1f, true,
                                                      RL::PPO::Backbone::MlpExperts);
                    applyPpoBalance(*m_sfPPOMCTSMLP);
                }
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS == nullptr)
                    m_sfDQNMCTS = new DQNMCTSAgent(env, 128, 0.99f, 0.001f, 1.0f, 1.414f);
                break;
            case AGENT_EVAB:
                /* 与 aiThink/aiThinkForAgent 用同一组构造参数 (宽度/深度/时间预算),
                   否则"主 agent 的权重"和"训练 clone 的网络"结构会对不上, load 直接失败 */
                if (m_sfEVAB == nullptr)
                    m_sfEVAB = new EVABAgent(env, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
                break;
            /*
               ---- 这三个的构造参数同样必须与 aiThinkRaw 那一支逐字一致 ----
               (宽度 / 骨干 / c_puct / 专家宽度 / 辅助系数)。任何一个不同, 主 agent 与
               训练 clone 的网络结构就对不上, 而 save/load 的结构指纹会让整轮 load 失败
               —— 表现是"每轮都在报载入失败", 不是静默错误, 但也白跑。
               权重已经在 startupLoad() 里载好了 (正常路径下这里不会是 nullptr); 这一支
               只是兜底: 没扫到权重文件时也建一个能训的实例。
            */
            case AGENT_SACAZ:
                if (m_sfSACAZ == nullptr)
                    m_sfSACAZ = new SACAZAgent(env, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
                break;
            case AGENT_SACAZ_MOE:
                if (m_sfSACAZMoe == nullptr)
                    m_sfSACAZMoe = createSACAZMoETbAgent(env, false, m_dynamicReward);
                break;
            /* [2026-09 独立类] MoE+MLP 那一支: 自己的实例 + 自己的构造点
               (落到上面那一支去就会**用 TB 的骨干**建出来, 而且一声不响) */
            case AGENT_SACAZ_MOE_MLP:
                if (m_sfSACAZMoeMlp == nullptr)
                    m_sfSACAZMoeMlp = createSACAZMoEMlpAgent(env, false, m_dynamicReward);
                break;
            /*
               ---- 59e5233 行为还原版 ----
               构造必须走 createSACAZLegacyAgent(): 这一支用的是**独立类**
               SACAZLegacyAgent, 在这里手写 `new SACAZAgent(...)` 会静默退回当前口径
               (两者参数结构完全相同, 连 save/load 都不会报错)。
            */
            case AGENT_SACAZ_OLD:
                if (m_sfSACAZOld == nullptr)
                    m_sfSACAZOld = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD);
                break;
            /* 同上, 但骨干是稀疏 MoE + TB 专家 (同一个类的另一个 Backbone) */
            case AGENT_SACAZ_OLD_MOE:
                if (m_sfSACAZOldMoe == nullptr)
                    m_sfSACAZOldMoe = createSACAZLegacyAgent(env, AGENT_SACAZ_OLD_MOE);
                break;
            case AGENT_DQNAB:
                if (m_sfDQNAB == nullptr) {
                    m_sfDQNAB = new DQNABAgent(env, DQNAB_HIDDEN, 0.99f, 0.001f,
                                               DQNABAgent::Backbone::SparseMoeTb);
                    m_sfDQNAB->nodeBudget = DQNAB_NODES;   /* 与 aiThinkRaw 同一预算 */
                }
                break;
            /*
               [2026-09 dev-dqnmcts-moetb] DQN+MCTS (稀疏MoE+TB专家): 构造参数与
               aiThinkRaw / decideOnEnvRawLocked / makeAgentInstance / loadAgentModel
               **逐字一致** (含 denseMoe 走默认 false), 预算同样在这里定下来 ——
               主 agent 与下面那个训练 clone 的网络结构必须一模一样, 否则结构指纹
               让整轮 load 失败 (表现是"每轮都在报载入失败": 不静默, 但白跑)。
            */
            case AGENT_DQNMCTS_MOE:
                if (m_sfDQNMCTSMOE == nullptr) {
                    m_sfDQNMCTSMOE = new DQNMCTSMOETbAgent(env, DQNMCTS_MOE_HIDDEN, 0.99f,
                                                           0.001f, 1.0f, 1.5f);
                    m_sfDQNMCTSMOE->simulations = DQNMCTS_MOE_SIMS;  /* 与 aiThinkRaw 同一预算 */
                }
                break;
            default: break;
            }
        }

        /* ---- 克隆主agent权重到临时文件 ---- */
        const char *tmpWeights = tmpWeightsOf(type);
        /*
           ---- 先放掉上一轮留在内存里的那一份 (2026-10 对弈期间不落盘) ----
           一轮的往返是"写种子 -> clone 读 -> clone 写回 -> 主 agent 读", 对弈期间这四步
           都在内存里 (见 rl/weightio.hpp), 而这份字节的寿命就到"主 agent 读完"为止
           (上一轮结尾已经 drop 过一次)。这里再 drop 一次是给"上一轮被中途丢弃"那种
           情况兜底: 模式切换/暂停会让一轮在同步之前就 continue 掉, 那份内存副本就会
           一直留到下一次写同一个路径 —— 单看内存不是灾难, 但"看不见的几百 MB" 不值得留。
           在这一行之后才写种子, 所以这次 drop 不可能命中本轮要用的数据。
        */
        RL::WeightIO::drop(tmpWeights);
        bool seeded = false;
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            switch (type) {
            case AGENT_PG:
                if (m_sfPG) seeded = m_sfPG->savePolicy(tmpWeights);
                break;
            case AGENT_DQN:
                if (m_sfDQN) seeded = m_sfDQN->saveModel(tmpWeights);
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS) seeded = m_sfPPOMCTS->saveModel(tmpWeights);
                break;
            case AGENT_PPOMCTS_MLP:
                if (m_sfPPOMCTSMLP) seeded = m_sfPPOMCTSMLP->saveModel(tmpWeights);
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS) seeded = m_sfDQNMCTS->saveModel(tmpWeights);
                break;
            case AGENT_EVAB:
                if (m_sfEVAB) seeded = m_sfEVAB->saveModel(tmpWeights);
                break;
            /* 多文件家族: 路径是**前缀** -> <prefix>_actor/_q1/_q2 或 _trunk/_v/_a */
            case AGENT_SACAZ:
                if (m_sfSACAZ) seeded = m_sfSACAZ->saveModel(tmpWeights);
                break;
            case AGENT_SACAZ_MOE:
                if (m_sfSACAZMoe) seeded = m_sfSACAZMoe->saveModel(tmpWeights);
                break;
            /* [2026-09 独立类] MoE+MLP: 自己的实例 + **自己的临时前缀** (TMP_WEIGHTS_*) */
            case AGENT_SACAZ_MOE_MLP:
                if (m_sfSACAZMoeMlp) seeded = m_sfSACAZMoeMlp->saveModel(tmpWeights);
                break;
            case AGENT_SACAZ_OLD:
                if (m_sfSACAZOld) seeded = m_sfSACAZOld->saveModel(tmpWeights);
                break;
            case AGENT_SACAZ_OLD_MOE:
                if (m_sfSACAZOldMoe) seeded = m_sfSACAZOldMoe->saveModel(tmpWeights);
                break;
            case AGENT_DQNAB:
                if (m_sfDQNAB) seeded = m_sfDQNAB->saveModel(tmpWeights);
                break;
            /* [2026-09 dev-dqnmcts-moetb] 同上: 前缀 -> <prefix>_trunk / _q */
            case AGENT_DQNMCTS_MOE:
                if (m_sfDQNMCTSMOE) seeded = m_sfDQNMCTSMOE->saveModel(tmpWeights);
                break;
            default: break;
            }
        }

        /*
           ---- 种子权重写不出去就**不要训这一轮** (2026-09) ----
           一轮训练是个往返: 主agent --写--> 临时文件 --读--> clone(训练) --写-->
           临时文件 --读--> 主agent。这些 save/load 原来**全部忽略返回值**, 于是第一步
           写失败时: clone 拿不到种子 ⇒ 从**它自己的随机初始化**开始训 ⇒ 把随机权重写回
           临时文件 ⇒ 再同步回主 agent。结果是"每轮把模型重置一次, 却一句报错都没有",
           表现就是"跑了很多轮完全没有效果"。
           所以这里把写失败当成硬失败: 报出来、睡 2 秒重试 (与上面"没有可训练权重"那条
           同样的节奏, 避免忙等), 绝不带着"随机权重"继续。

           注意这条消息**只**表示"写盘失败"这一个意思 —— "这一支还没接入训练" 由上面
           的 trainable 分支单独报 (以前两者共用这条消息, 于是 EVAB 那种"没写"被读成
           "写不出去", 排查方向完全错了)。
        */
        if (!seeded) {
            qWarning() << "[train] 种子权重写入失败, 跳过本轮训练: agent"
                       << agentDisplayName(type) << "路径" << tmpWeights;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        /* ---- 在独立棋盘上创建克隆agent并训练4轮 ---- */
        bool roundApplied = false;   /* 本轮成果是否真的写回并同步 (见下面各 case) */
        {
            Chess trainChess;  /* 独立训练棋盘: 从初始局面开始 */
            /*
               这一轮训练结束后的损失 (界面的"训练损失曲线")。不上报损失的 agent
               (PG/PPO/DQN+MCTS 的训练循环里没有 scalar loss) 保持 NaN -> 不上图。
            */
            float roundLoss = std::numeric_limits<float>::quiet_NaN();

            /*
               ---- 每个 case 都必须检查载入与写回的返回值 (2026-09) ----
               种子文件刚由上面写好, 所以 clone 的 loadModel **应该**成功; 若失败,
               说明架构/版本对不上 (例如改了状态编码而临时文件是旧的), 那么这一轮
               训练的就是一个"随机初始化的 clone", 它的成果绝不能写回主 agent。
               同理, 写回失败表示本轮成果没落盘, 也不该拿去同步主 agent。
            */
            switch (type) {
            case AGENT_PG: {
                PGEagent clone(trainChess, 64, 0.9f, 0.01f, 1.0f);
                if (!clone.loadPolicy(tmpWeights)) {
                    qWarning() << "[train] PG clone 载入种子权重失败, 本轮丢弃";
                    break;
                }
                clone.train(roundEpisodes, roundMaxMoves, true, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.savePolicy(tmpWeights)) {
                    qWarning() << "[train] PG 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            case AGENT_DQN: {
                DQNAgent clone(trainChess, 64, 0.99f, 0.001f, 1.0f);
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN clone 载入种子权重失败, 本轮丢弃";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] DQN 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            case AGENT_PPOMCTS: {
                PPOMCTSAgent clone(trainChess, 64, 0.99f, 0.001f, 1.414f);
                /*
                   训练 clone 必须与主 agent 用**同一套** MoE 均衡/门控接线 —— 否则
                   "训练用的那一支"与"决策用的那一支"不是同一个东西, 而且这件事情
                   在权重往返上**看不出来** (偏置是非参数缓冲, 不进权重文件)。
                */
                applyPpoBalance(clone);
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS clone 载入种子权重失败, 本轮丢弃"
                               << "(临时文件与当前网络架构不匹配? 路径" << tmpWeights << ")";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, BG_TRAIN_SIMS,
                                    roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            case AGENT_PPOMCTS_MLP: {
                /*
                   MLP 专家骨干的同一套往返。模拟次数用 PPO_MLP_SIMS (与界面同一预算):
                   一轮 60 手在 MLP 专家上约 12 s (一次模拟便宜 ~25x), 而 TB 那一支
                   同规模要分钟级 —— 这正是这条支路值得接的原因。
                   骨干必须显式传: 默认是 TB, 传漏了 clone 的结构就与主 agent 不一致,
                   loadModel 会因结构指纹不匹配而每一轮都失败。
                */
                PPOMCTSAgent clone(trainChess, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f,
                                   true, RL::PPO::Backbone::MlpExperts);
                /* 与 AGENT_PPOMCTS 那一支同一条纪律: 均衡/门控接线必须与主 agent 一致
                   (`enableMlpGate` 还必须在 loadModel **之前** —— 门控形状不同的话
                   权重文件读侧会明确拒绝, 而不是静默错位)。 */
                applyPpoBalance(clone);
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS-MLP clone 载入种子权重失败, 本轮丢弃"
                               << "(临时文件与当前网络架构不匹配? 路径" << tmpWeights << ")";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, PPO_MLP_SIMS,
                                    roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS-MLP 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            case AGENT_DQNMCTS: {
                DQNMCTSAgent clone(trainChess, 128, 0.99f, 0.001f, 1.0f, 1.414f);
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS clone 载入种子权重失败, 本轮丢弃";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, BG_TRAIN_SIMS,
                                    roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            case AGENT_EVAB: {
                /*
                   EVAB 的"训练"是 TD-leaf 自对弈蒸馏 (见 evagent.h):
                     trainSelfPlay(games, playDepth, labelDepth, maxMoves, verbose, batch)
                   选步用 playDepth (便宜), 标签用 labelDepth 的根评分 (更准) ——
                   参数取值见 BG_TRAIN_EVAB_* 的注释。
                */
                EVABAgent clone(trainChess, 48, EVAB_DEPTH, EVAB_BUDGET_MS);
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] EVAB clone 载入种子权重失败, 本轮丢弃"
                               << "(临时文件与当前网络结构不匹配? 路径" << tmpWeights << ")";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, BG_TRAIN_EVAB_PLAY_DEPTH,
                                    BG_TRAIN_EVAB_LABEL_DEPTH, roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] EVAB 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            /*
               ---- SAC+AZ / SAC+AZ-MoE / SAC+AZ-59e5233 (2026-09 补齐) ----
               与 SAC+AZ 的其它入口同一条往返: clone 用**独立棋盘** trainChess 自对弈
               (trainSelfPlay 内部自己 reset 棋盘), 模拟次数按骨干分开给 (见
               BG_TRAIN_SACAZ_SIMS / BG_TRAIN_SACAZ_MOE_SIMS 的注释)。
               损失是 critic 的 MSE (getLastTrainLoss), 上界面那条"训练损失曲线"。

               **clone 必须与主 agent 同一支**: 三支的参数结构完全相同 (都是 iFcLayer
               的 w/b), 所以建成另一支不会让 loadModel 失败 —— 它只会静默地按另一套口径
               训练 (目标熵/alpha 学习率/critic 目标/叶子估值口径不同)。AGENT_SACAZ_OLD
               因此在这里构造**独立类** SACAZLegacyAgent; 共享的往返写成 trainSACRound()
               (模板: 两个类没有继承关系, 参数化比共用基类小, 也不会让口径混起来)。
            */
            case AGENT_SACAZ: {
                SACAZAgent clone(trainChess, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
                /* [2026-09 奖励方法开关] 克隆体与主 agent 必须同一套奖励, 否则训出来的
                   权重与"面板上那一支"不是同一个东西 */
                applyRewardMethod(clone, m_dynamicReward);
                roundApplied = trainSACRound(clone, tmpWeights, agentDisplayName(type),
                                             roundEpisodes, sacazTrainSims(type),
                                             roundMaxMoves, roundLoss);
                break;
            }
            case AGENT_SACAZ_MOE: {
                /*
                   [2026-09 独立类] 克隆体也是**独立类** `SACAZMoETbAgent` ——
                   与 AGENT_SACAZ_MOE 的构造点同一套形状参数 (Separate 口径),
                   否则"训练用的克隆"与"决策用的实例"就是两支不同的网。
                */
                SACAZMoETbAgent clone(trainChess, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                      64, SACAZ_MOE_AUX);
                /* [2026-10 门控实验] clone 必须与主 agent 同一套路由口径,
                   否则"训练用的克隆"与"决策用的实例"不是同一个东西 */
                applyLossFreeBias(clone);
                applyMlpGate(clone);
                roundApplied = trainSACRound(clone, tmpWeights, agentDisplayName(type),
                                             roundEpisodes, sacazTrainSims(type),
                                             roundMaxMoves, roundLoss);
                break;
            }
            /*
               [2026-09 独立类] MoE+MLP 那一支的克隆体: 同样是**独立类**、
               同一套形状参数 (Separate 口径 + expertHidden 64 + aux SACAZ_MOE_AUX)。
               形状参数必须与 createSACAZMoEMlpAgent(env, false, m_dynamicReward) 逐字一致, 否则
               "训练用的克隆"与"回主 agent 的权重"两边结构对不上 (loadModel 当场失败 = 每轮白跑)。
            */
            case AGENT_SACAZ_MOE_MLP: {
                SACAZMoEMlpAgent clone(trainChess, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                       64, SACAZ_MOE_AUX,
                                       SACAZMoEMlpAgent::TrunkMode::Separate);
                /* [2026-10 门控实验] clone 必须与主 agent 同一套路由口径 */
                applyLossFreeBias(clone);
                applyMlpGate(clone);
                roundApplied = trainSACRound(clone, tmpWeights, agentDisplayName(type),
                                             roundEpisodes, sacazTrainSims(type),
                                             roundMaxMoves, roundLoss);
                break;
            }
            case AGENT_SACAZ_OLD: {
                SACAZLegacyAgent clone(trainChess, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f);
                roundApplied = trainSACRound(clone, tmpWeights, agentDisplayName(type),
                                             roundEpisodes, sacazTrainSims(type),
                                             roundMaxMoves, roundLoss);
                break;
            }
            /*
               还原版 + TB 专家骨干: clone 的骨干必须与主 agent **同一支**
               (SparseMoeTb + 同样的专家宽度/辅助系数), 否则结构指纹会让 loadModel
               当场失败 —— 那种失败是"每轮都丢弃", 不是静默错误, 但白跑。
            */
            case AGENT_SACAZ_OLD_MOE: {
                SACAZLegacyAgent clone(trainChess, SACAZ_HIDDEN, 0.99f, 0.001f, 1.5f,
                                       SACAZLegacyAgent::Backbone::SparseMoeTb,
                                       64, SACAZ_MOE_AUX);
                roundApplied = trainSACRound(clone, tmpWeights, agentDisplayName(type),
                                             roundEpisodes, sacazTrainSims(type),
                                             roundMaxMoves, roundLoss);
                break;
            }
            /*
               ---- DQN+AB (2026-09 补齐) ----
               trainSelfPlay(episodes, maxMoves, verbose, tempRoot, tempFinal) 内部按
               nodeBudget 搜 (与界面同一预算 DQNAB_NODES), 标签来自 Planned 目标。
               注意它的主干是 37.5 M 参数的稀疏 MoE: 一次临时文件的往返是 ~276 MB 写 +
               两次读 (实测启动读一次 3.8 s, 见 [weights] DQN+AB 那两行), 所以这一支的
               一轮明显比 PG/DQN 贵 —— 仍然远小于它自己那 60 手的搜索开销。
            */
            case AGENT_DQNAB: {
                DQNABAgent clone(trainChess, DQNAB_HIDDEN, 0.99f, 0.001f,
                                 DQNABAgent::Backbone::SparseMoeTb);
                clone.nodeBudget = DQNAB_NODES;      /* 与主 agent / 界面同一预算 */
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+AB clone 载入种子权重失败, 本轮丢弃"
                               << "(临时文件与当前网络结构不匹配? 路径" << tmpWeights << ")";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] DQN+AB 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            /*
               ---- DQN+MCTS (稀疏MoE+TB专家), 2026-09 dev-dqnmcts-moetb ----
               与上面 DQN+AB 那一支同一条往返 (clone -> 独立棋盘 -> 载种子 -> 自对弈 ->
               写回), 差别在训练接口与预算口径:
                 * `trainSelfPlay(episodes, simulations, maxMoves, verbose)` 的第三个参数是
                   **模拟次数** (不是节点数), 给 DQNMCTS_MOE_SIMS;
                 * 训练用多少次模拟与界面决策**同一个常数** —— 它是"这次训练花多少时间"
                   的唯一旋钮 (一个 TB 专家前向 ~5-6 ms, 40 次 ≈ 175 ms/步)。
               ⚠ clone 的构造参数必须与主 agent **逐字一致** (denseMoe 走默认 false),
                 否则结构指纹让 loadModel 每轮失败 (不静默, 但白跑)。
            */
            case AGENT_DQNMCTS_MOE: {
                DQNMCTSMOETbAgent clone(trainChess, DQNMCTS_MOE_HIDDEN, 0.99f, 0.001f,
                                        1.0f, 1.5f);
                clone.simulations = DQNMCTS_MOE_SIMS;   /* 与主 agent / 界面同一预算 */
                if (!clone.loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS (稀疏MoE+TB专家) clone 载入种子权重失败, "
                                  "本轮丢弃 (临时文件与当前网络结构不匹配? 路径"
                               << tmpWeights << ")";
                    break;
                }
                clone.trainSelfPlay(roundEpisodes, DQNMCTS_MOE_SIMS, roundMaxMoves, false);
                roundLoss = clone.getLastTrainLoss();
                if (!clone.saveModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS (稀疏MoE+TB专家) 训练权重写回失败, 本轮丢弃";
                    break;
                }
                roundApplied = true;
                break;
            }
            default: break;
            }

            if (std::isfinite(roundLoss)) {
                emit trainLossSample((double)roundLoss, agentDisplayName(type),
                                     m_trainSampleNo.fetch_add(1) + 1);
            }
        }

        /*
           ---- 训练好的权重同步回主agent ----
           只有 roundApplied (载入成功 && 写回成功) 才同步。否则主 agent 会去读一个
           没被更新过、甚至可能不存在的临时文件 —— 而它的 loadModel 一旦静默失败,
           主 agent 就还是旧权重, 界面上却看不出任何异常。
        */
        if (!roundApplied) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        /*
           ---- 第二轮暂停检查 (P0-b): **在锁内**判, 就在写主 agent 之前 ----
           这一轮可能是在"暂停请求到达之前"就已经开跑的在飞轮次, 所以必须再确认一次。
           顺序很关键: **先拿 m_agentMutex, 再读 m_bgPaused**。反过来 (先读再抢锁) 会留下
           一个窗口 —— pauseBackgroundTraining 在那个窗口里拿到锁并返回 (它以为已经没人在
           写了), 而我们随后拿到锁把权重写了进去, 于是"冻结"是假的。
           拿到锁之后读到的 paused 一定是最新的: pause 置位在前、抢锁在后。
           丢弃这一轮是"浪费一点算力", 写进去则是"评估结论被悄悄污染"。
        */
        {
            std::lock_guard<std::mutex> lock(m_agentMutex);
            /*
               [P0-b 收尾] 判据与循环开头是**同一个** isBackgroundTrainingPaused():
               它把对弈模式也算进来 ⇒ 用户在某一轮训练**跑到一半时**把模式切成
               评估/只对弈, 那一轮的成果同样会被丢弃, 而不是"赶在冻结生效之前先写进去"。
            */
            if (isBackgroundTrainingPaused()) {
                if (m_matchMode.load() != MATCH_TRAIN) {
                    m_bgRoundsSkippedByMode.fetch_add(1);
                }
                continue;      /* 安全: lock_guard 会析构 (见暂停闸门的死锁分析) */
            }
            switch (type) {
            case AGENT_PG:
                if (m_sfPG && !m_sfPG->loadPolicy(tmpWeights)) {
                    qWarning() << "[train] PG 权重同步回主 agent 失败";
                }
                break;
            case AGENT_DQN:
                if (m_sfDQN && !m_sfDQN->loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN 权重同步回主 agent 失败";
                }
                break;
            case AGENT_PPOMCTS:
                if (m_sfPPOMCTS && !m_sfPPOMCTS->loadModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS 权重同步回主 agent 失败";
                }
                break;
            case AGENT_PPOMCTS_MLP:
                if (m_sfPPOMCTSMLP && !m_sfPPOMCTSMLP->loadModel(tmpWeights)) {
                    qWarning() << "[train] PPO+MCTS-MLP 权重同步回主 agent 失败";
                }
                break;
            case AGENT_DQNMCTS:
                if (m_sfDQNMCTS && !m_sfDQNMCTS->loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS 权重同步回主 agent 失败";
                }
                break;
            case AGENT_EVAB:
                if (m_sfEVAB && !m_sfEVAB->loadModel(tmpWeights)) {
                    qWarning() << "[train] EVAB 权重同步回主 agent 失败";
                }
                break;
            /* 这三个是 2026-09 补齐的 (多文件模型: 路径是前缀, 与写种子用的是同一个) */
            case AGENT_SACAZ:
                if (m_sfSACAZ && !m_sfSACAZ->loadModel(tmpWeights)) {
                    qWarning() << "[train] SAC+AZ 权重同步回主 agent 失败";
                }
                break;
            case AGENT_SACAZ_MOE:
                if (m_sfSACAZMoe && !m_sfSACAZMoe->loadModel(tmpWeights)) {
                    qWarning() << "[train] SAC+AZ-MoE 权重同步回主 agent 失败";
                }
                break;
            /* [2026-09 独立类] MoE+MLP: 从**自己的**临时文件同步回**自己的**实例 */
            case AGENT_SACAZ_MOE_MLP:
                if (m_sfSACAZMoeMlp && !m_sfSACAZMoeMlp->loadModel(tmpWeights)) {
                    qWarning() << "[train] SAC+AZ-MoE-MLP 权重同步回主 agent 失败";
                }
                break;
            case AGENT_SACAZ_OLD:
                /*
                   注意 loadModel 只看**参数结构**, 两支完全相同 ⇒ 它不会替我们发现
                   "同步错了哪一支"。错误的防线在构造处 (createSACAZAgent) 与
                   tmpWeightsOf (独立前缀), 不在这里。
                */
                if (m_sfSACAZOld && !m_sfSACAZOld->loadModel(tmpWeights)) {
                    qWarning() << "[train] SAC+AZ-59e5233 权重同步回主 agent 失败";
                }
                break;
            case AGENT_SACAZ_OLD_MOE:
                /* 同上: 防线在构造处 (createSACAZLegacyAgent 的骨干分支) 与
                   tmpWeightsOf (独立前缀), 不在这里 */
                if (m_sfSACAZOldMoe && !m_sfSACAZOldMoe->loadModel(tmpWeights)) {
                    qWarning() << "[train] SAC+AZ-59e5233-MoE 权重同步回主 agent 失败";
                }
                break;
            case AGENT_DQNAB:
                if (m_sfDQNAB && !m_sfDQNAB->loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+AB 权重同步回主 agent 失败";
                }
                break;
            /* [2026-09 dev-dqnmcts-moetb] 从**自己的**临时前缀同步回**自己的**实例 */
            case AGENT_DQNMCTS_MOE:
                if (m_sfDQNMCTSMOE && !m_sfDQNMCTSMOE->loadModel(tmpWeights)) {
                    qWarning() << "[train] DQN+MCTS (稀疏MoE+TB专家) 权重同步回主 agent 失败";
                }
                break;
            default: break;
            }
        }

        /*
           ---- 一轮结束 ⇒ 放掉内存里的那一份临时权重 (2026-10 对弈期间不落盘) ----
           对弈期间这条往返全在内存里 (见 rl/weightio.hpp), 而它的字节可能与整个网络
           同量级 (稀疏 MoE 那几支是 3 x 146 MB)。**在这里 drop** 而不是等改道关闭:
           改道可能整场对弈都开着, 放着不管就是"每一支都留一份在内存里"。
           位置必须在**同步回主 agent 之后**: 上面那个 switch 用的还是这份字节。
           另外每轮开头也会 drop 一次 (见那段注释): 万一某一轮在中途被丢弃 (模式切换 /
           暂停), 那份内存副本就不会留到下一轮才放掉。
        */
        RL::WeightIO::drop(tmpWeights);
    }
}

/* ================================================================
 *  shutdownSave - 程序退出时保存所有已初始化agent的权重
 *
 *  将权重保存到 weights/ 目录下的标准路径.
 *  只保存已创建(非nullptr)的agent.
 * ================================================================ */
void ChessBoard::shutdownSave()
{
    /* 确保 weights 目录存在 */
    QDir().mkpath("weights");

    /*
       保存 aiThink / aiThinkForAgent 共享的 neural agent 权重。
       路径全部走 defaultWeightPath() —— 与"退出时统一保存"用的是同一份,
       不会再出现"存这边、读那边"的静默失效。
       (EVAB 曾经漏在这里: 它是界面上可选、能被在线训练的 agent, 但退出时从来不落盘,
        于是每次启动都从随机价值网络重新开始, 上一局学到的东西全丢。)
    */
    const int saved = saveAllInstantiatedAgentsOnExit();
    qInfo().noquote() << QStringLiteral("[weights] 退出保存: %1 个 agent 已写盘").arg(saved);
}

/*
 * saveAllInstantiatedAgentsOnExit - 退出时把**所有已实例化**的可训练 agent 存到标准路径
 *
 * 用户口径 (2026-09): "只有程序退出时再保存模型"。
 *
 * 这是**唯一**会把权重写到正式路径的地方 (另一处写盘是后台训练的临时文件
 * weights/_temp_train*)。之所以要把它单独成函数并公开: 退出路径在 MainWindow 的析构里,
 * 那里要能明确表达"这一步就是在保存", 而不是让"哪里写盘"散在多个事件回调里 ——
 * 后者正是这次卡顿的来源: 终局回调里也存一次, 而保存持 m_agentMutex 写 530 MB,
 * 把同一把锁上的 AI 决策挡住 (实测 2.4 s 的保存让一次决策从 6.1 s 涨到 8.4 s)。
 *
 * 只存 hasAgentInstance 为真的 agent: 纯搜索的 AB 各档 / MCTS 从来没有实例,
 * 也就没有权重可存 —— 与 backgroundTrainLoop 同一判据。
 */
int ChessBoard::saveAllInstantiatedAgentsOnExit()
{
    QDir().mkpath("weights");

    /*
       ---- 退出保存之前: **强行解除**对弈期间的权重改道 (2026-10) ----
       这是"不落盘"那条口径的最后一条边界: 用户完全可以在**一局棋下到一半**时关窗,
       那时 m_playDivertHuman 还开着。标准权重路径 (下面 defaultWeightPath) 本来就不
       受改道影响, 所以这条防线**不是**为了这一次保存 —— 它是为了"将来有人在退出路径上
       写临时文件"的那一天: 改道还开着的话, 那些字节会写进内存, 然后随进程一起消失,
       而日志上一切正常 (静默丢数据, 本工程最怕的形状)。
       把判据写死在这里 (而不是靠"调用方记得先解除"), 是因为退出路径只有这一个入口。
    */
    if (RL::WeightIO::divertEnabled()) {
        qInfo().noquote()
            << QStringLiteral("[weights] 退出保存: 先解除对弈期间的权重改道 "
                              "(本场改道 %1 次 / %2 字节) —— 标准权重路径本来就写磁盘")
                   .arg((qlonglong)RL::WeightIO::divertedWrites())
                   .arg((qlonglong)RL::WeightIO::divertedBytes());
        m_playDivertMatch.store(false);
        m_playDivertHuman.store(false);
        RL::WeightIO::setDivertEnabled(false);
    }

    const AgentType all[] = { AGENT_PG, AGENT_DQN, AGENT_PPOMCTS, AGENT_DQNMCTS,
                              AGENT_EVAB, AGENT_SACAZ, AGENT_SACAZ_MOE, AGENT_DQNAB,
                              AGENT_PPOMCTS_MLP, AGENT_SACAZ_OLD, AGENT_SACAZ_OLD_MOE,
                              /* [2026-09 dev-dqnmcts-moetb] 独立类, 正式前缀
                                 weights/dqnmcts_moe_agent_* (与旧类的单文件不同) */
                              AGENT_DQNMCTS_MOE };
    /*
       ⚠ [2026-10] 表里这四个 (PG / DQN / AGENT_SACAZ_OLD / AGENT_SACAZ_OLD_MOE)
       现在是**不可达**的: 它们已从下拉框与启动预加载里移除 (见 kWeightAgents 注释),
       所以 `hasAgentInstance()` 恒为假 ⇒ 下面直接 continue。留着它们是为了"哪天真要
       恢复"时只需要改两处 (kAgents + kWeightAgents), 而不用再回来补这里 ——
       但**列表本身仍然必须与"能实例化的 agent"一致**: 这里漏一个, 表现是
       "界面上能选、能训练, 退出后权重从来没写过" (EVAB 当年就是这么漏的, 见上面那段)。
    */
    int saved = 0;
    for (AgentType t : all) {
        if (!hasAgentInstance(t)) {
            continue;
        }
        const bool ok = saveCurrentAgentModel(t, defaultWeightPath(t));
        qInfo().noquote() << QStringLiteral("[weights] 退出保存 %1: %2")
                                 .arg(agentDisplayName(t))
                                 .arg(ok ? QStringLiteral("成功") : QStringLiteral("失败"));
        if (ok) {
            saved++;
        }
    }
    return saved;
}

/* ================================================================
 *  [2026-10 移除] 回放功能 (applyDbStep / loadReplayGame / replayPrev /
 *  replayNext / applyReplayStep)
 * ================================================================
 *  这五个函数连同 MainWindow 侧的回放 UI 一起删除。删的理由是这条链路
 *  **结构性不可达**, 不是"没用": 写入端从未接线 (`GameDatabase::startGame /
 *  recordMove / endGame` 全仓零调用, 见 docs/analysis.md 第 357 行与
 *  docs/issues_review.md 第 1000 / 2739 行), 所以"选历史对局 -> 逐步回放"
 *  永远走不到。`GameDatabase` 与 chess_games.db 保留 (构造函数里那句 open 不动)。
 *
 *  顺带记一条被删掉的修正, 免得将来重写时再踩: 原来的三处回放代码都写成
 *  `step->nextId = dbStep.toX;` —— 把终点**列坐标**(0..8) 当成了棋子 id;
 *  moveForward/moveBack 会用 nextId 去索引 m_children 并结算收益, 语义完全错,
 *  而且 step->valid 一直是 false。
 */
