/*
 * bench_sac_learn_main.cpp - "SAC 边下边学" 的受控复现 + 奖励塑形 A/B (2026-09)
 * ============================================================================
 *
 * 为什么要单独写这个工具 (而不是用 bench_sac_mcts_min / bench_agent_arena):
 *
 *   * `bench_sac_mcts_min` 测的是**随机权重**的 SAC —— 它一次都不训练, 回答的是
 *     "网络结构与搜索口径对不对"。用户报的现象 (奖励累计、损失曲线、300 手和棋)
 *     全部来自**边下边学**的过程, 那个工具量不到。
 *   * `bench_agent_arena --mode=match` 调的是 `agent.selectMove()`, **不调用
 *     exploreAndTrain()** —— 也就是没有"走子前先探索 64 步 + 在线更新一次"这一步。
 *     界面上 (ChessBoard::aiThinkForAgent) 每手都要做这一步, 它正是损失曲线与学习
 *     信号的来源。少了它, 对局就只是"用随机权重下棋"。
 *
 * 所以本工具的协议**逐条对齐界面** (ChessBoard::playMatchGame + preTrainThenDecide):
 *   每局 reset 到标准开局; 红先; 每手
 *     SAC 方   : exploreAndTrain(color, preTrainSteps)  -> selectMove(color, sims, 0)
 *     对手     : MCTS::findBestMove(color, mctsSims)
 *   合法性校验 + 无效走法兜底 (与界面同一套), 走满 --plies 判和。
 *
 * 量出来的东西 (都是用户在界面上看到的那几条曲线的分子/分母):
 *   1. **胜负和与和棋的成因**: 将死/吃将 vs 60 回合规则判和 vs 手数上限判和;
 *   2. **奖励累计**: 两种口径都给 ——
 *        "学习口径" = agent 自己的 computeReward (材质 x0.1 + 每步代价) + 终局 ±1;
 *        "显示口径" = 界面奖励曲线用的 Chess::moveForward totalReward (材质原值 x1)
 *                     + 终局 ±1 —— 用户 CSV 里的 3.1 / 4.5 是这个口径;
 *      **两个口径差 10 倍**, 所以"奖励是 MCTS 的两倍"这句话必须先说清是哪一个;
 *   3. **损失曲线**: 点数/均值/最大值, 以及"每局几个点";
 *   4. **critic 尺度**: 训练结束时的 |Q| 均值/区间 —— 用来分开"学得好"与
 *      "critic 发散"这两种都会让 loss 变大的原因;
 *   5. 奖励塑形 (--reward-shape) 与手数上限 (--plies) 的 A/B。
 *
 * 用法:
 *   bench_sac_learn.exe [--games=20] [--plies=300] [--sims=256] [--mcts-sims=800]
 *                       [--pre-train=64] [--seed=20240901] [--mcts-srand=12345]
 *                       [--legacy] [--reward-shape=0|1|2] [--gamma=0.99]
 *                       [--no-train] [--csv=path] [--quiet] [--label=名字]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "mcts.h"
#include "sacazagent.h"
#include "sacazlegacyagent.h"
#include "rl/util.hpp"

namespace {

/*
   界面默认值 (ChessBoard::DEFAULT_MAX_PLIES / SACAZ_SIMS 等), 这里**手写常量而不是
   包含 chessboard.h**: 那是 Qt Widgets 的头 (QWidget/QFontMetrics), 一个无界面基准
   工具没必要把它拖进来。数值与 chessboard.cpp 的那几行保持一致, 不一致会在输出里
   直接被看见 (每行都印了实际用的值)。
*/
constexpr int kGuiMaxPlies = 300;    /* ChessBoard::DEFAULT_MAX_PLIES */
constexpr int kGuiSacSims = 256;     /* SACAZ_SIMS */
constexpr int kGuiMctsSims = 800;    /* MCTS (纯搜索) 的模拟次数 */
constexpr int kGuiPreTrain = 64;     /* ChessBoard::m_preTrainSteps 默认值 */

struct Cfg {
    int games = 20;
    int maxPlies = kGuiMaxPlies;   /* 界面默认 300 */
    int sims = kGuiSacSims;        /* 界面 SACAZ_SIMS */
    int mctsSims = kGuiMctsSims;   /* 界面 MCTS 的模拟次数 */
    int preTrain = kGuiPreTrain;   /* 界面 preTrainSteps 默认值 */
    unsigned seed = 20240901;
    unsigned mctsSrand = 12345;
    bool legacy = false;       /* true = 用 SACAZLegacyAgent (59e5233 口径) */
    int rewardShape = 0;
    float gamma = 0.99f;
    bool train = true;         /* false = 只决策不学习 (对照组) */
    /*
       --no-search-learn: 关掉"每次真实决策从自己的搜索学一次"
       (SACAZAgent::learnFromSearch, 2026-09 新加; 类里默认**开**)。
       开着的效果是: 即使 preTrain=0 (等价于界面不勾 rollout), 对弈过程仍然在训练,
       而且策略会收到自己的 π_MCTS 当监督 —— 见 docs/sac_learn_reward_2026_09.md §9。
    */
    bool searchLearn = true;
    /*
       ---- 学习口径的消融旋钮 (2026-09) ----
       用途: 实测"最新实现为什么训练之后反而更弱"。59e5233 (还原版) 与当前实现在**学习
       口径**上的差别只有这四个数 (再加隐层激活, 而两者代码相同):
         还原版: clampTarget=0 (不夹目标) / huberDelta=0 (纯 MSE) / 熵比 0.98 / alpha lr 1e-3
         当前:   clampTarget=2 / huberDelta=1 / 熵比 0.5 / alpha lr 5e-3
       把当前实现逐个换成还原版的值, 就能看出是哪一个把棋力弄坏的 (而不是靠猜)。
       0 = 用类里的默认值 (不覆盖)。
    */
    float clampTarget = -1.0f, huberDelta = -1.0f, entropyRatio = -1.0f, alphaLr = -1.0f;
    /*
       --no-sparse-leaf: 搜索叶子估值走**全量**(128 槽下一次只要算 128 列, 稀疏头省不了
       什么, 却多走一条与训练不同的代码路径)。类里默认开着; 59e5233 那一支是关的。
    */
    bool sparseLeaf = true;
    /*
       ---- [2026-09 ①] 熵项与目标熵的两个实验开关 (默认 = 现状) ----
       `--entropy-in-target=0` : 熵项不进软价值 ⇒ 不进 critic 的自举目标 (只留在策略损失)。
                                假设: y 里的 −γ·α·H(s') 是个与棋局无关的常数偏置, 它才是
                                critic 被顶到钳位边界 (或压到 0) 的原因。
       `--entropy-slots`       : 目标熵 H̄ 的分母从"合法着法数"换成"合法槽位数"。
                                假设: 128 槽哈希有碰撞 ⇒ H ≤ log(槽位) < log(着法) ⇒ 按着法
                                算的 H̄ 永远达不到 ⇒ α 被单向推到边界。
       负数 = 不覆盖, 用类里的默认值 (与改动前逐位一致)。
    */
    float entropyInTarget = -1.0f;
    bool entropySlots = false;
    /*
       ---- [①] 搜索叶子值的**整体缩放** (SACAZAgent::valueScale 的透传) ----
       用它做一个只动"搜索侧"的对照: `--no-train --value-scale=0/1/4` 时策略与 critic
       完全不动 (随机权重、一次都不学), 只有叶子值的**常数**在变。若得分率跟着这个常数
       走, 就说明本轮看到的那些得分率差异是**搜索侧的常数偏置**造成的, 不是"学到了更多"。
       (`valueScale` 只乘搜索用的软价值, 不动学习侧的目标 —— 见 sacazagent.h。)
       负数 = 不覆盖 (默认 1.0 = 逐位不变)。
    */
    float valueScale = -1.0f;
    /*
       ---- [F1] 目标网同步率 (本轮定位到的主缺陷) ----
       默认 (tau=1e-3 / 每 64 步) 在一次会话里只把目标网移动 2~4% ⇒ 自举项里没有游戏信息。
       `--target-tau` / `--target-iter` 用来扫"多快才算跟得上"; 负数 = 不覆盖。
       `--target-tau=1` 等价于硬拷贝 (softUpdateTo 的 tau>=1)。
    */
    float targetTau = -1.0f;
    int targetIter = 0;
    std::string csv;
    std::string label = "sac";
    bool quiet = false;
};

Cfg g;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* 统计量 (与其它 bench 工具同一套口径: 得分率 = 胜 1 / 和 0.5) */
struct Score {
    int wins = 0, losses = 0, draws = 0, broken = 0;
    int drawByCap = 0, drawByRule = 0;    /* 和棋的两种成因 */
    /*
       终局类型 (分胜负的那些局): 吃将 / 将死 / 困毙。
       **"会不会将棋"的判据是"将死"(或吃将)**, 不是"赢了" —— 困毙同样判负, 但它不是
       将杀能力 (中国象棋规则里困毙也算赢, 所以"赢"里混着两种完全不同的能力)。
       这一条是本轮踩过的坑: 第一版只判"将还在不在", 于是"将死"被算成了"没将死"。
    */
    int jiangCaptured = 0, checkmate = 0, stalemate = 0;
    int noCapture60 = 0, repetition = 0, drawOther = 0, drawCap = 0;
    int plies = 0;
    /* 走满手数上限那些局, 终局时的"无吃子半回合数"(见 GameLog::endHalfMoveClock) */
    long long capClockSum = 0;
    int capGames = 0, clockMax = 0;
    int n() const { return wins + losses + draws; }
    double rate() const {
        const int m = n();
        return (m > 0) ? ((double)wins + 0.5 * (double)draws) / (double)m : 0.0;
    }
    void interval(double &lo, double &hi) const {
        const int m = n();
        if (m <= 0) { lo = 0.0; hi = 1.0; return; }
        const double p = rate();
        const double z = 1.96;
        const double d = 1.0 + z * z / (double)m;
        const double c = p + z * z / (2.0 * (double)m);
        const double s = z * std::sqrt(p * (1.0 - p) / (double)m + z * z / (4.0 * (double)m * (double)m));
        lo = (c - s) / d;
        hi = (c + s) / d;
        if (lo < 0.0) { lo = 0.0; }
        if (hi > 1.0) { hi = 1.0; }
    }
};

/* 一局的详细账 */
struct GameLog {
    int plies = 0;
    int result = Chess::RESULT_DRAW;     /* Chess::Result */
    bool hitCap = false;                 /* 是"走满手数上限"判和 */
    bool brokenFlag = false;             /* agent 返回过无效走法 (已兜底) */
    /*
       终局分类 (2026-09, 修过一次口径):
         winJiangCaptured : 赢在**吃掉对方的将** (isGameOver 那条路径);
         winCheckmate     : 赢在对方**被将死** (无合法走法 **且** 正被将军);
         winStalemate     : 赢在对方**困毙** (无合法走法但没被将军; 中国象棋同样判负);
         drawNoCapture60  : 60 回合无吃子判和 (REWARD_MAX_PLIES 那个规则);
         drawRepetition   : 三次重复 / 长将等着法 (Chess::isDraw 的 reason);
         drawByCap        : 走满手数上限 (本工具/界面自己的上限, 不是棋规)。
       为什么必须分开: "还没学会将棋"这句话的判据是**能不能将死**, 而"赢"里有一部分是
       吃掉将、还有一部分是困毙 —— 把它们混成一个"胜"会把结论说错 (第一版就把将死与
       困毙混成了"将还在 = 没将死", 结论会反过来)。
    */
    int winJiangCaptured = 0, winCheckmate = 0, winStalemate = 0;
    int drawNoCapture60 = 0, drawRepetition = 0, drawOther = 0;
    bool jiangAliveLoser = true;
    /*
       终局时的"无吃子半回合数"(Chess::halfMoveClock)。
       这是判断"**吃子之后在磨蹭**还是**一直在吃**"的直接读数: 走满手数上限判和时,
          * 若它很小 -> 终盘还在吃子 (材质还没吃完 / 一直在换子);
          * 若它接近 120 -> 终盘已经静下来在空走 (那才是典型的"磨蹭"和棋)。
       60 回合无吃子判和本身就是"连续 120 半回合没吃子", 所以这两个读数是互补的。
    */
    int endHalfMoveClock = 0;
    double learnRewardSAC = 0.0;         /* 学习口径: 材质 x0.1 + 每步代价 (SAC 自己吃的) */
    double learnRewardOpp = 0.0;
    double displayRewardSAC = 0.0;       /* 显示口径: 材质原值 + 终局 (界面曲线) */
    double displayRewardOpp = 0.0;
    double sacMaterial = 0.0;            /* 终局时 SAC 一侧剩余材质 */
    double oppMaterial = 0.0;
    int lossPoints = 0;                  /* 这一局产生的损失曲线点数 */
};

/*
 * [2026-09 ①] 训练中诊断的**一段**(两个累计量之差) —— 用来把"整轮"切成"每局"看。
 * 为什么必须能切片: 本轮要回答的是 α 的**轨迹** (它是不是被单向推到边界) 与 clamp 比例,
 * 只看训练结束后的快照会把"一路推到 5.0"和"一直在 0.2 附近"看成同一个结果。
 */
struct DiagDelta {
    long long n = 0, clamped = 0, hBelow = 0, hBelowSlots = 0;
    double yPreAbsMean = 0.0, yPreMean = 0.0, yPreAbsMax = 0.0;
    double vMean = 0.0, vQMean = 0.0, vEntMean = 0.0;
    double hMean = 0.0, hBarMean = 0.0, hBarSlotsMean = 0.0;
    double slotsMean = 0.0, legalMean = 0.0;
    double qSpreadMean = 0.0, qAbsMean = 0.0;
    double alphaFirst = 0.0, alphaLast = 0.0;
    DiagDelta() = default;
    DiagDelta(const SACAZAgent::TrainDiag &a, const SACAZAgent::TrainDiag &b)
    {
        n = b.n - a.n;
        if (n <= 0) { return; }
        const double d = (double)n;
        clamped = b.clamped - a.clamped;
        hBelow = b.hBelowHbar - a.hBelowHbar;
        hBelowSlots = b.hBelowHbarSlots - a.hBelowHbarSlots;
        yPreAbsMean = (b.yPreAbsSum - a.yPreAbsSum) / d;
        yPreMean = (b.yPreSum - a.yPreSum) / d;
        yPreAbsMax = b.yPreAbsMax;
        vMean = (b.vNextSum - a.vNextSum) / d;
        vQMean = (b.vQSum - a.vQSum) / d;
        vEntMean = (b.vEntSum - a.vEntSum) / d;
        hMean = (b.hSum - a.hSum) / d;
        hBarMean = (b.hBarSum - a.hBarSum) / d;
        hBarSlotsMean = (b.hBarSlotsSum - a.hBarSlotsSum) / d;
        slotsMean = (b.slotsSum - a.slotsSum) / d;
        legalMean = (b.legalSum - a.legalSum) / d;
        qSpreadMean = (b.qSpreadSum - a.qSpreadSum) / d;
        qAbsMean = (b.qAbsMeanSum - a.qAbsMeanSum) / d;
        alphaFirst = (a.alphaFirst >= 0.0) ? a.alphaFirst : b.alphaFirst;
        alphaLast = b.alphaLast;
    }
    double clampFrac() const { return n > 0 ? (double)clamped / (double)n : 0.0; }
    double hBelowFrac() const { return n > 0 ? (double)hBelow / (double)n : 0.0; }
    double hBelowSlotsFrac() const {
        return n > 0 ? (double)hBelowSlots / (double)n : 0.0;
    }
};

static int otherColor(int c) {
    return (c == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}
static void materialOf(Chess &c, int color, double &nonJiang)
{
    nonJiang = 0.0;
    for (int i = 0; i < 32; i++) {
        const Stone *s = c.stones[i];
        if (s == nullptr || !s->alive || s->color != color) { continue; }
        if (s->type == Stone::TYPE_JIANG) { continue; }
        nonJiang += s->value;
    }
}

static bool jiangAlive(Chess &c, int color)
{
    for (int i = 0; i < 32; i++) {
        const Stone *s = c.stones[i];
        if (s != nullptr && s->alive && s->color == color && s->type == Stone::TYPE_JIANG) {
            return true;
        }
    }
    return false;
}

/*
 * 一局。sac 走 sacColor, 对手固定为 MCTS。
 * 协议逐条对齐 ChessBoard::playMatchGame / preTrainThenDecide (见文件头)。
 */
static GameLog playGame(Chess &board, SACAZAgent &sac, MCTS &mcts, int sacColor,
                        float &lastLoss, int &lossTotal, double &lossSum, double &lossMax)
{
    GameLog g1;
    board.reset();
    int turn = Stone::COLOR_RED;
    int moves = 0;

    while (moves < g.maxPlies) {
        const int r = board.getResult(turn);
        if (r != Chess::RESULT_ONGOING) { break; }

        const bool sacTurn = (turn == sacColor);
        Step step;
        double displayReward = 0.0;   /* 界面口径的即时奖励 (moveForward 的 totalReward) */

        if (sacTurn) {
            /*
               界面的 preTrainThenDecide: 先"探索环境 + 在线训练一次", 再基于当前局面
               决策。--no-train 时跳过第一步 (对照组: 只看随机权重 + 搜索能走成什么样)。
               注意 preTrain=0 时 **selectMove 里仍可能学一次** (learnFromSearch), 所以
               损失的记账放在决策**之后**, 判据是 learnSteps 是否前进 (与界面
               ChessBoard::reportLearnedLoss 同一条判据) —— 否则 preTrain=0 的那些配置
               会把"学了"记成"没学"。
            */
            const int stepsBefore = sac.getLearnSteps();
            if (g.train && g.preTrain > 0) {
                sac.exploreAndTrain(turn, g.preTrain);
            }
            step = sac.selectMove(turn, g.sims, 0.0f);
            if (sac.getLearnSteps() > stepsBefore) {
                const float lv = sac.getLastTrainLoss();
                if (std::isfinite(lv)) {
                    lastLoss = lv;
                    lossTotal++;
                    lossSum += (double)lv;
                    lossMax = std::max(lossMax, (double)lv);
                    g1.lossPoints++;
                } else {
                    /* 学了但没上报损失 (NaN): 也算一个"更新点", 但不进损失的均值 */
                    g1.lossPoints++;
                }
            }
        } else {
            step = mcts.findBestMove(turn, g.mctsSims);
        }

        /* ---- 合法性校验 + 兜底 (与界面同一条规则, 不能让无效走法决定胜负) ---- */
        std::vector<Step *> legal;
        board.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            g1.plies = moves;
            g1.result = (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                                   : Chess::RESULT_RED_WIN;
            return g1;      /* 无棋可走 = 将杀/困毙 */
        }
        const bool ok = board.isLegalMove(turn, &step);
        if (!ok) {
            step = *legal[0];
            g1.brokenFlag = true;
        }

        /* ---- 即时奖励 (必须在落子之前算: 落子会把被吃子置 alive=false) ---- */
        const float learnNow = sac.computeReward(step, turn);
        if (sacTurn) { g1.learnRewardSAC += (double)learnNow; }
        else         { g1.learnRewardOpp += (double)learnNow; }

        Steps::instance().put(legal);
        board.moveForward(&step, displayReward);
        /*
           totalReward 是**黑方视角**的记账 (吃红子 +, 吃黑子 -), 而奖励曲线按走子方视角
           画 —— 与 ChessBoard::playMatchGame 的换算逐字相同。
        */
        const double moverDisplay = (turn == Stone::COLOR_RED) ? -displayReward : displayReward;
        if (sacTurn) { g1.displayRewardSAC += moverDisplay; }
        else         { g1.displayRewardOpp += moverDisplay; }

        turn = otherColor(turn);
        board.sideToMove = turn;
        moves++;
    }

    g1.plies = moves;
    Chess::DrawReason drawReason = Chess::DRAW_NONE;
    g1.result = board.getResult(board.sideToMove, &drawReason);
    if (g1.result == Chess::RESULT_ONGOING) {
        g1.result = Chess::RESULT_DRAW;   /* 走满上限 = 和 */
        g1.hitCap = true;
    }
    /* ---- 终局分类 (见 GameLog 的说明) ---- */
    const int loser = (g1.result == Chess::RESULT_RED_WIN) ? Stone::COLOR_BLACK
                     : (g1.result == Chess::RESULT_BLACK_WIN) ? Stone::COLOR_RED
                                                              : Stone::COLOR_NONE;
    if (loser != Stone::COLOR_NONE) {
        if (!jiangAlive(board, loser)) {
            g1.winJiangCaptured++;
        } else if (board.isInCheck(loser)) {
            g1.winCheckmate++;
        } else {
            g1.winStalemate++;
        }
    } else if (g1.result == Chess::RESULT_DRAW) {
        if (g1.hitCap) {
            /* 手数上限: 与棋规无关, 是本工具的截断 */
        } else if (drawReason == Chess::DRAW_NO_CAPTURE60) {
            g1.drawNoCapture60++;
        } else if (drawReason != Chess::DRAW_NONE) {
            g1.drawRepetition++;
        } else {
            g1.drawOther++;
        }
    }

    /* 终局那一点: 两种口径都要加 (学习口径走 agent 的 terminalReward, 含塑形) */
    const float termSAC = sac.terminalReward(g1.result, sacColor);
    const float termOpp = sac.terminalReward(g1.result, otherColor(sacColor));
    if (g1.result != Chess::RESULT_DRAW) {
        g1.learnRewardSAC += (double)termSAC;
        g1.learnRewardOpp += (double)termOpp;
    }
    /* 界面口径的终局常量 (REWARD_TERMINAL = ±1, 和棋 0) */
    const double dTermSAC = outcomeForMover(g1.result, sacColor);
    const double dTermOpp = outcomeForMover(g1.result, otherColor(sacColor));
    g1.displayRewardSAC += dTermSAC;
    g1.displayRewardOpp += dTermOpp;

    materialOf(board, sacColor, g1.sacMaterial);
    materialOf(board, otherColor(sacColor), g1.oppMaterial);
    g1.endHalfMoveClock = board.halfMoveClock;
    g1.jiangAliveLoser = jiangAlive(board, (sacColor == Stone::COLOR_RED)
                                               ? Stone::COLOR_BLACK : Stone::COLOR_RED);
    return g1;
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *k) -> const char * {
            const std::size_t n = std::strlen(k);
            if (std::strncmp(a, k, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--games"))      { g.games = std::atoi(v); }
        else if (const char *v = val("--plies"))  { g.maxPlies = std::atoi(v); }
        else if (const char *v = val("--sims"))   { g.sims = std::atoi(v); }
        else if (const char *v = val("--mcts-sims")) { g.mctsSims = std::atoi(v); }
        else if (const char *v = val("--pre-train")) { g.preTrain = std::atoi(v); }
        else if (const char *v = val("--seed"))   { g.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--mcts-srand")) { g.mctsSrand = (unsigned)std::atoi(v); }
        else if (const char *v = val("--reward-shape")) { g.rewardShape = std::atoi(v); }
        else if (const char *v = val("--gamma"))  { g.gamma = (float)std::atof(v); }
        else if (const char *v = val("--csv"))    { g.csv = v; }
        else if (const char *v = val("--label"))  { g.label = v; }
        else if (std::strcmp(a, "--legacy") == 0) { g.legacy = true; }
        else if (std::strcmp(a, "--no-train") == 0) { g.train = false; }
        else if (std::strcmp(a, "--no-search-learn") == 0) { g.searchLearn = false; }
        else if (const char *v = val("--clamp"))   { g.clampTarget = (float)std::atof(v); }
        else if (const char *v = val("--huber"))   { g.huberDelta = (float)std::atof(v); }
        else if (const char *v = val("--entropy-ratio")) { g.entropyRatio = (float)std::atof(v); }
        else if (const char *v = val("--alpha-lr")) { g.alphaLr = (float)std::atof(v); }
        else if (const char *v = val("--entropy-in-target")) { g.entropyInTarget = (float)std::atof(v); }
        else if (std::strcmp(a, "--entropy-slots") == 0) { g.entropySlots = true; }
        else if (const char *v = val("--value-scale")) { g.valueScale = (float)std::atof(v); }
        else if (const char *v = val("--target-tau")) { g.targetTau = (float)std::atof(v); }
        else if (const char *v = val("--target-iter")) { g.targetIter = std::atoi(v); }
        else if (std::strcmp(a, "--no-sparse-leaf") == 0) { g.sparseLeaf = false; }
        else if (std::strcmp(a, "--quiet") == 0)  { g.quiet = true; }
        else { std::fprintf(stderr, "[warn] 未知参数: %s\n", a); }
    }
    if (g.games % 2 != 0) { g.games++; }
    RL::Random::setSeed(g.seed);

    Chess board;
    board.reset();
    /*
       与界面同一个构造: SACAZ_HIDDEN=64 / gamma / lr / cpuct=1.5。
       --legacy 用 59e5233 行为还原版 (独立派生类), 它的 4 项口径在类里固定, 这里不覆盖。
    */
    SACAZAgent *sacPtr = nullptr;
    if (g.legacy) {
        sacPtr = new SACAZLegacyAgent(board, 64, g.gamma, 0.001f, 1.5f);
    } else {
        sacPtr = new SACAZAgent(board, 64, g.gamma, 0.001f, 1.5f);
    }
    SACAZAgent &sac = *sacPtr;
    sac.rewardShape = g.rewardShape;     /* 唯一需要外部设的旋钮 (界面没有它) */
    /*
       只在**显式关闭**时覆盖: 派生类 (59e5233 还原版) 在自己的构造函数里把它钉成 false,
       那是它的口径 —— 无条件赋 true 会把这个钉住的口径抹掉 (第一版工具就这么错过一次,
       于是"--legacy --pre-train=0" 那个"完全不训练"的对照组其实训练了 1300 次)。
    */
    if (!g.searchLearn) {
        sac.learnFromSearch = false;
    }
    /* 学习口径的消融 (>=0 才覆盖; 派生类的口径在它自己的构造函数里, 这里会**改掉它**,
       所以 --legacy 时不建议再传这四个 —— 打印出来的实际值可以核对) */
    if (g.clampTarget >= 0.0f)  { sac.clampTarget = g.clampTarget; }
    if (g.huberDelta >= 0.0f)   { sac.huberDelta = g.huberDelta; }
    if (g.entropyRatio >= 0.0f) { sac.entropyRatio = g.entropyRatio; }
    if (g.alphaLr >= 0.0f)      { sac.learningRateAlpha = g.alphaLr; }
    if (g.entropyInTarget >= 0.0f) { sac.entropyInTarget = g.entropyInTarget; }
    if (g.entropySlots)         { sac.entropySlotsAsLegal = true; }
    if (g.valueScale >= 0.0f)   { sac.valueScale = g.valueScale; }
    if (g.targetTau >= 0.0f)    { sac.targetTau = g.targetTau; }
    if (g.targetIter > 0)       { sac.replaceTargetIter = g.targetIter; }
    if (!g.sparseLeaf)          { sac.sparseLeafEval = false; }
    MCTS mcts(board, 1.414);

    /*
       对手的随机流固定住 (构造之后重播): MCTS 用 std::rand(), 而 srand 播的是 time()
       —— 不固定的话两版跑的就不是同一副牌 (见 docs/sac_regression_2026_09.md §7 第 0 条)。
    */
    if (g.mctsSrand != 0u) { std::srand(g.mctsSrand); }

    const char *shapeName[] = { "base(材质x0.1+终局±1)", "no-material(只留每步代价+终局±1)",
                                "mate-bonus(终局 x(1+败方材质/3.5))" };
    std::printf("=== bench_sac_learn: %s ===\n", g.label.c_str());
    std::printf("agent      : %s (rewardShape=%d %s, gamma=%.4f)\n",
                g.legacy ? "SACAZLegacyAgent (59e5233 口径)" : "SACAZAgent (当前口径)",
                g.rewardShape, shapeName[(g.rewardShape >= 0 && g.rewardShape <= 2)
                                             ? g.rewardShape : 0],
                (double)g.gamma);
    std::printf("协议       : 对 MCTS(%d 次模拟) %d 局, SAC %d 次模拟, 每手预训练 %d 步, "
                "手数上限 %d, 交换先后手, 标准开局\n",
                g.mctsSims, g.games, g.sims, g.preTrain, g.maxPlies);
    std::printf("可复现性   : seed=%u mcts-srand=%u  学习=%s\n",
                g.seed, g.mctsSrand, g.train ? "开 (每手 exploreAndTrain)" : "**关** (只看随机权重)");
    std::printf("搜索学习   : learnFromSearch=%d (每次真实决策是否用**自己的搜索**样本学一次;"
                " 关了就等于改动前的行为)\n", (int)sac.learnFromSearch);
    std::printf("学习口径   : clampTarget=%.2f huberDelta=%.2f 熵比=%.3f alphaLr=%.4f "
                "| 叶子估值=%s\n"
                "             (实际生效值; 59e5233 是 0 / 0 / 0.98 / 1e-3 / 全量)\n",
                (double)sac.clampTarget, (double)sac.huberDelta,
                (double)sac.entropyRatio, (double)sac.learningRateAlpha,
                sac.sparseLeafEval ? "稀疏头" : "全量");
    std::printf("熵项去处   : entropyInTarget=%.2f (0 = 熵项不进软价值/critic 目标) | "
                "目标熵分母=%s | 搜索叶子缩放 valueScale=%.2f\n",
                (double)sac.entropyInTarget,
                sac.entropySlotsAsLegal ? "**合法槽位数** (反事实)" : "合法着法数 (现状)",
                (double)sac.valueScale);
    {
        const double it = (double)(sac.replaceTargetIter > 0 ? sac.replaceTargetIter : 1);
        const double perIter = 1.0 - std::pow(1.0 - (double)sac.targetTau, 1.0 / it);
        const double moved = 1.0 - std::pow(1.0 - perIter, 2600.0);
        std::printf("目标网同步 : tau=%.4f 每 %d 次 learn (2600 步 ~20 局的移动率 %.1f%%) | "
                    "老口径 tau=0.001/64 步 = 2~4%%\n",
                    (double)sac.targetTau, sac.replaceTargetIter, 100.0 * moved);
    }

    Score st;
    int lossTotal = 0;
    double lossSum = 0.0, lossMax = 0.0;
    double learnSAC = 0.0, learnOpp = 0.0, dispSAC = 0.0, dispOpp = 0.0;
    double sacMatSum = 0.0, oppMatSum = 0.0;
    std::vector<double> pliesList;
    /* [①] 训练中诊断: 累计量起点 (每局取差值得"这一局"的分布) */
    const SACAZAgent::TrainDiag diagStart = sac.getTrainDiag();
    SACAZAgent::TrainDiag diagPrev = diagStart;
    const double t0 = nowMs();

    for (int i = 0; i < g.games; i++) {
        const bool sacIsRed = (i % 2 == 0);
        const int sacColor = sacIsRed ? Stone::COLOR_RED : Stone::COLOR_BLACK;
        float lastLoss = std::numeric_limits<float>::quiet_NaN();
        const GameLog gl = playGame(board, sac, mcts, sacColor, lastLoss,
                                    lossTotal, lossSum, lossMax);
        if (gl.brokenFlag) { st.broken++; }
        const bool sacWon = (gl.result == Chess::RESULT_RED_WIN && sacIsRed)
                            || (gl.result == Chess::RESULT_BLACK_WIN && !sacIsRed);
        const bool sacLost = (gl.result == Chess::RESULT_RED_WIN && !sacIsRed)
                             || (gl.result == Chess::RESULT_BLACK_WIN && sacIsRed);
        if (sacWon)       { st.wins++; }
        else if (sacLost) { st.losses++; }
        else {
            st.draws++;
            if (gl.hitCap) { st.drawByCap++; } else { st.drawByRule++; }
        }
        /* 终局类型 (SAC 视角: 赢在哪 / 输在哪) */
        if (sacWon || sacLost) {
            st.jiangCaptured += gl.winJiangCaptured;
            st.checkmate += gl.winCheckmate;
            st.stalemate += gl.winStalemate;
        } else {
            st.noCapture60 += gl.drawNoCapture60;
            st.repetition += gl.drawRepetition;
            st.drawOther += gl.drawOther;
        }
        st.drawCap += gl.hitCap ? 1 : 0;
        if (gl.hitCap) { st.capClockSum += gl.endHalfMoveClock; st.capGames++; }
        if (gl.endHalfMoveClock > st.clockMax) { st.clockMax = gl.endHalfMoveClock; }
        st.plies += gl.plies;
        pliesList.push_back((double)gl.plies);
        learnSAC += gl.learnRewardSAC;  learnOpp += gl.learnRewardOpp;
        dispSAC += gl.displayRewardSAC; dispOpp += gl.displayRewardOpp;
        sacMatSum += gl.sacMaterial;    oppMatSum += gl.oppMaterial;

        if (!g.quiet) {
            const char *verdict = gl.brokenFlag ? "BROKEN"
                                  : sacWon ? "SAC 胜" : sacLost ? "SAC 负" : "和";
            std::printf("  局 %3d/%d  SAC=%s  %-7s %3d 手  学习奖励 %+6.2f:%+6.2f  "
                        "显示奖励 %+6.2f:%+6.2f  材质 %.1f:%.1f  损失点 %d\n",
                        i + 1, g.games, sacIsRed ? "RED " : "BLACK", verdict, gl.plies,
                        gl.learnRewardSAC, gl.learnRewardOpp,
                        gl.displayRewardSAC, gl.displayRewardOpp,
                        gl.sacMaterial, gl.oppMaterial, gl.lossPoints);
        }
        /*
           [①] 这一局的训练分布 (α / 目标 y 被夹的比例 / V 的两项分解 / H 与 H̄)。
           每局一行: 判断"α 是不是被单向推到边界""critic 的目标是不是常被夹成常数"
           靠的就是这个**轨迹**, 而不是训练后的快照。
        */
        if (!g.quiet) {
            const SACAZAgent::TrainDiag now = sac.getTrainDiag();
            const DiagDelta dx(diagPrev, now);
            diagPrev = now;
            if (dx.n > 0) {
                std::printf("              [①] α %.4f→%.4f | y夹前均值|y| %.2f (符号 %+.2f, "
                            "最大 %.2f) 被夹 %.0f%% | V %+.2f = E[minQ] %+.2f + αH %+.2f | "
                            "H %.2f vs H̄ %.2f (H<H̄ %.0f%%) | Qspread %.3f | 槽位/着法 %.1f/%.1f\n",
                            dx.alphaFirst, dx.alphaLast, dx.yPreAbsMean, dx.yPreMean,
                            dx.yPreAbsMax, 100.0 * dx.clampFrac(),
                            dx.vMean, dx.vQMean, dx.vEntMean,
                            dx.hMean, dx.hBarMean, 100.0 * dx.hBelowFrac(),
                            dx.qSpreadMean, dx.slotsMean, dx.legalMean);
            }
        }
    }
    const double sec = (nowMs() - t0) / 1000.0;

    double lo = 0.0, hi = 1.0;
    st.interval(lo, hi);
    const int n = (st.n() > 0) ? st.n() : 1;
    std::printf("\n--- 结果 (SAC 视角) ---\n");
    std::printf("  比分        : SAC %d 胜 / MCTS %d 胜 / 和 %d%s\n",
                st.wins, st.losses, st.draws,
                st.broken > 0 ? "  (+机制违规)" : "");
    std::printf("  得分率      : %.1f%%  95%% Wilson [%.1f%%, %.1f%%]\n",
                100.0 * st.rate(), 100.0 * lo, 100.0 * hi);
    std::printf("  和棋成因    : 手数上限 %d 局 / 无吃子60回合 %d 局 / 重复等着法 %d 局 / 其它 %d 局\n",
                st.drawCap, st.noCapture60, st.repetition, st.drawOther);
    std::printf("  分胜负的终局: 吃将 %d / **将死 %d** / 困毙 %d  (将死 + 吃将 = 真的会将棋)\n",
                st.jiangCaptured, st.checkmate, st.stalemate);
    /*
       "终盘还在吃子吗" —— 判和局的 endHalfMoveClock 均值。
       这套读数的用处: 判和局有两种完全不同的形状, 而它们的对策相反:
         * 均值小 (还在吃子) -> 材质目标还没吃完, 属于"贪吃/换子";
         * 均值接近 120 (早已没吃子) -> 典型的"磨蹭空走", 该罚的是时间而不是材质。
    */
    std::printf("  判和局的终盘无吃子半回合数: 均值 %.1f (最大 %d, 120 = 触发自然限着)  "
                "%s\n",
                st.capGames > 0 ? (double)st.capClockSum / (double)st.capGames : 0.0,
                st.clockMax,
                (st.capGames > 0 && (double)st.capClockSum / (double)st.capGames > 80.0)
                    ? "-> 终盘基本静下来了 (磨蹭型)"
                    : "-> 终盘还在吃子 (贪吃型)");
    std::printf("  平均手数    : %.1f (上限 %d)\n", (double)st.plies / (double)n, g.maxPlies);
    std::printf("  奖励/局     : 学习口径 SAC %+.3f vs 对手 %+.3f (差 %+.3f)\n",
                learnSAC / n, learnOpp / n, (learnSAC - learnOpp) / n);
    std::printf("                显示口径 SAC %+.3f vs 对手 %+.3f (差 %+.3f)  "
                "<- 界面上那条曲线/用户 CSV 用的口径\n",
                dispSAC / n, dispOpp / n, (dispSAC - dispOpp) / n);
    std::printf("  剩余材质/局 : SAC %.2f / 对手 %.2f (满 3.5)\n", sacMatSum / n, oppMatSum / n);
    std::printf("  损失曲线    : %d 点 (%.1f 点/局), 均值 %.5f, 最大 %.5f\n",
                lossTotal, (double)lossTotal / (double)n,
                lossTotal > 0 ? lossSum / (double)lossTotal : 0.0, lossMax);

    /*
       ---- critic 尺度: 分开"学得好"与"发散了" ----
       两种都会让 loss 变大。文档 (docs/arena_sac_vs_ppo_report.md §5.2) 记过实测:
       无约束的 critic 从 |Q| 0.063 一路漂到 13.38, 那时搜索的 PUCT 被 Q 压坏。
       这里在**训练后的权重**上量同一组读数 (随机开局 12 个局面)。
    */
    /* [①] 探针读数 (块外 CSV 也要用) */
    double probeQAbsMean = 0.0, probeQAbsMax = 0.0, probeNormEntropy = 0.0;
    double probeQTargetAbsMean = 0.0, probeQTargetSpread = 0.0;
    double probeHMean = 0.0, probeSlotMean = 0.0, probeMoveMean = 0.0;
    {
        double qAbs = 0.0, qAbsMax = 0.0;
        int qn = 0;
        /*
           策略熵 (归一化: H / log(合法数)) —— 与 critic 尺度一起看, 才能分清两种"学坏了":
             * |Q| 很小 且 熵接近 1 -> critic 没提供信息, 策略基本还是均匀分布;
             * |Q| 很小 但 熵很低   -> 策略被一个**没有信息的** critic 推尖了 (最坏的一种:
               相当于在噪声上做决断, 搜索的先验也跟着坏)。
           这也是本轮"α 目标熵"那个改动的机制读数 (见 docs §9)。
        */
        double normEntropy = 0.0;
        double hProbeSum = 0.0, slotProbeSum = 0.0, moveProbeSum = 0.0;
        /*
           [①] **目标网**的尺度 —— 这一条是"critic 为什么没有游戏信息"的直接证据:
           软备份 V(s') 用的是 q1Target/q2Target, 而它们每 64 步才做一次 tau=1e-3 的
           Polyak 同步 ⇒ 一次 20 局的对弈 (~2600 步) 只把目标网从随机初始化挪动了
           1-(1-1e-3)^(步数/64) ≈ **4%**。也就是说训练目标里的 E[min Q(s')] 几乎一直是
           "随机网络的输出", 与棋局无关; 真正决定目标量级的是 α·H 那一项 (见 [①] 诊断)。
        */
        double qTAbs = 0.0, qTAbsMax = 0.0;
        int qTn = 0;
        /* [F1] 目标网的**排序**信号: 它才是自举项 V(s') 里能用的那部分 (见下) */
        double qTSpreadSum = 0.0;
        int qTSpreadN = 0;
        int en = 0;
        for (int k = 0; k < 12; k++) {
            board.reset();
            for (int p = 0; p < 8; p++) {
                std::vector<Step *> legal;
                board.sample(board.sideToMove, legal);
                if (legal.empty()) { Steps::instance().put(legal); break; }
                std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
                const Step s = *legal[(std::size_t)pick(RL::Random::engine)];
                Steps::instance().put(legal);
                double d = 0.0;
                board.moveForward(&s, d);
                board.sideToMove = otherColor(board.sideToMove);
            }
            const int color = board.sideToMove;
            std::vector<Step *> legal;
            std::vector<int> idx;
            RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
            sac.getLegalActions(color, legal, idx, mask);
            Steps::instance().put(legal);
            if (idx.empty()) { continue; }
            RL::Tensor stx(SACAZAgent::STATE_DIM, 1);
            sac.encodeStateFor(color, stx);
            RL::Tensor q1(SACAZAgent::ACTION_DIM, 1), q2(SACAZAgent::ACTION_DIM, 1);
            sac.qValues(stx, q1, q2);
            for (int a : idx) {
                const double q = std::min((double)q1[a], (double)q2[a]);
                qAbs += std::fabs(q);
                qAbsMax = std::max(qAbsMax, std::fabs(q));
                qn++;
            }
            /* [①] 目标网的同一组读数 (V(s') 就是拿它算的) */
            {
                RL::Tensor qt1(SACAZAgent::ACTION_DIM, 1), qt2(SACAZAgent::ACTION_DIM, 1);
                sac.qTargetValues(stx, qt1, qt2);
                double qm = 0.0, qs = 0.0;
                int cnt = 0;
                for (int a : idx) {
                    const double q = std::min((double)qt1[a], (double)qt2[a]);
                    qTAbs += std::fabs(q);
                    qTAbsMax = std::max(qTAbsMax, std::fabs(q));
                    qTn++;
                    qm += q; qs += q * q; cnt++;
                }
                if (cnt > 0) {
                    qm /= (double)cnt;
                    const double var = qs / (double)cnt - qm * qm;
                    qTSpreadSum += (var > 0.0) ? std::sqrt(var) : 0.0;
                    qTSpreadN++;
                }
            }
            /* 策略熵 (只算合法动作上) */
            RL::Tensor piA(SACAZAgent::ACTION_DIM, 1);
            sac.policy(stx, mask, piA);
            /*
               熵必须按**掩码**累加 (槽位去重)。旧版按 idx 累加: 128 槽哈希有碰撞, 同一个
               槽位可能在 idx 里出现两次, 那份质量被重复计进熵 ⇒ 会出现"归一化熵 1.098 > 1"
               这种数学上不可能的数 (_H 不可能超过 log(支撑大小)_), 而它正是本轮 ① 要找的
               那条线索: **槽位数 < 着法数**, 而 α 的目标熵 H̄ = 熵比·log(着法数)。
            */
            double H = 0.0;
            int slotCount = 0;
            for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
                if (mask[a] <= 0.5f) { continue; }
                slotCount++;
                const double p = (double)piA[a];
                if (p > 0.0) { H -= p * std::log(p); }
            }
            if (slotCount > 1) {
                normEntropy += H / std::log((double)slotCount);
                hProbeSum += H;
                slotProbeSum += (double)slotCount;
                moveProbeSum += (double)idx.size();
                en++;
            }
        }
        std::printf("  策略熵      : H 均值 %.3f, 归一化 H/log(槽位) 均值 %.3f (%d 个局面; "
                    "1.0 = 完全均匀)\n",
                    en > 0 ? hProbeSum / (double)en : 0.0,
                    en > 0 ? normEntropy / (double)en : 0.0, en);
        /*
           "分母"这一行是 ① 的关键读数: π 只分布在**槽位**上, 而目标熵 H̄ 按**着法数**算,
           两者的 log 之差就是 α 的梯度里那个永远补不平的缺口。
        */
        std::printf("  熵的分母    : 平均 %.1f 槽位 vs %.1f 着法 = 平均挤掉 %.1f 个着法 ⇒ "
                    "log 差 %.3f (H̄ 按着法算时 H 达不到)\n",
                    en > 0 ? slotProbeSum / (double)en : 0.0,
                    en > 0 ? moveProbeSum / (double)en : 0.0,
                    en > 0 ? (moveProbeSum - slotProbeSum) / (double)en : 0.0,
                    en > 0 ? (std::log(moveProbeSum / (double)en)
                             - std::log(slotProbeSum / (double)en)) : 0.0);
        std::printf("  critic 尺度 : |Q| 均值 %.3f, |Q| 最大 %.3f (%d 个合法动作)  "
                    "%s\n", qn > 0 ? qAbs / (double)qn : 0.0, qAbsMax, qn,
                    (qn > 0 && qAbs / (double)qn > 3.0)
                        ? "**疑似发散: 搜索的 PUCT 会被 Q 压坏 (见 docs §5.2)**"
                        : "(量级正常)");
        /*
           [①] 目标网 = 软备份 V(s') 的来源。它几乎停在随机初始化上 (Polyak 太慢),
           所以"训练目标里那一部分游戏信息"其实是 0 —— 详见下面 [①] 诊断的 V(s') 分解。
        */
        {
            const double steps = (double)sac.getLearnSteps();
            const double iters = steps / (double)(sac.replaceTargetIter > 0
                                                      ? sac.replaceTargetIter : 1);
            const double moved = 1.0 - std::pow(1.0 - 1e-3, iters);
            std::printf("  目标网尺度  : |Q_target| 均值 %.3f, 最大 %.3f (%d 个合法动作), "
                        "排序信号 Qspread=%.4f ⇒ 相对随机初始化只移动了 %.2f%% "
                        "(%d 步 / 每 %d 步 tau=%.4f)\n",
                        qTn > 0 ? qTAbs / (double)qTn : 0.0, qTAbsMax, qTn,
                        qTSpreadN > 0 ? qTSpreadSum / (double)qTSpreadN : 0.0,
                        100.0 * moved, sac.getLearnSteps(), sac.replaceTargetIter,
                        (double)sac.targetTau);
            std::printf("                (V(s') 就是拿这张网算的: |Q_target| 停在随机尺度 "
                        "(≈0.07) 就说明**自举项里的游戏信息 ≈ 0**, 目标量级由 α·H 决定; "
                        "目标网的 Qspread 才是自举项能提供的排序信号)\n");
        }
        probeQAbsMean = (qn > 0) ? qAbs / (double)qn : 0.0;
        probeQAbsMax = qAbsMax;
        probeQTargetAbsMean = (qTn > 0) ? qTAbs / (double)qTn : 0.0;
        probeQTargetSpread = (qTSpreadN > 0) ? qTSpreadSum / (double)qTSpreadN : 0.0;
        probeNormEntropy = (en > 0) ? normEntropy / (double)en : 0.0;
        probeHMean = (en > 0) ? hProbeSum / (double)en : 0.0;
        probeSlotMean = (en > 0) ? slotProbeSum / (double)en : 0.0;
        probeMoveMean = (en > 0) ? moveProbeSum / (double)en : 0.0;
    }

    /*
       ================================================================
       [2026-09 ①] 训练**过程中**的 critic/α 诊断 (整轮累计)
       ================================================================
       三条要看的事:
         1. `y` 被 clampTarget 夹住的比例 —— 夹住的部分目标是个常数, 对"排序"零贡献;
         2. V(s') 的两项 (E[min Q] 与 α·H) 谁大谁小 —— 假设是"αH 项把 critic 顶走的";
         3. α 从哪走到哪, 以及 H 与两种 H̄ 的对比 (α 的梯度是 H − H̄)。
       `m_maxAbsTarget` 是**夹后**的极值 (所以它永远 ≤ clampTarget, 天生看不见"夹了多少"),
       这里补的正是它看不见的那一面。
       ================================================================
    */
    {
        const DiagDelta dx(diagStart, sac.getTrainDiag());
        std::printf("\n--- [①] 训练中的 critic/α 诊断 (整轮累计, %lld 个样本) ---\n", dx.n);
        if (dx.n > 0) {
            std::printf("  α 轨迹      : %.4f → %.4f  (上界 5.0 / 下界 0.02; 落到边界 = "
                        "被单向推走)\n",
                        dx.alphaFirst, dx.alphaLast);
            std::printf("  目标 y      : 夹**前** 均值 |y| %.2f (符号 %+.2f, 最大 %.2f); "
                        "被夹 %.1f%% (%lld/%lld)\n",
                        dx.yPreAbsMean, dx.yPreMean, dx.yPreAbsMax,
                        100.0 * dx.clampFrac(), dx.clamped, dx.n);
            std::printf("                (对照: 夹后的极值 m_maxAbsTarget=%.3f, "
                        "|td err| 极值 %.3f)\n",
                        sac.getMaxAbsTarget(), sac.getMaxAbsTdErr());
            std::printf("  V(s') 分解  : 总 %+.3f = E_π[min Q] %+.3f + α·H %+.3f  "
                        "⇒ 熵项占 %.0f%%\n",
                        dx.vMean, dx.vQMean, dx.vEntMean,
                        (std::fabs(dx.vMean) > 1e-9)
                            ? 100.0 * std::fabs(dx.vEntMean) / std::fabs(dx.vMean) : 0.0);
            std::printf("  熵与目标熵  : H %.3f | H̄(按着法) %.3f | H̄(按槽位) %.3f  ⇒  "
                        "H<H̄ %.1f%% / H<H̄(槽位) %.1f%%\n",
                        dx.hMean, dx.hBarMean, dx.hBarSlotsMean,
                        100.0 * dx.hBelowFrac(), 100.0 * dx.hBelowSlotsFrac());
            std::printf("                (%s)\n",
                        (dx.hBelowFrac() > 0.9 && dx.hBelowSlotsFrac() < 0.1)
                            ? "**H̄ 按着法算时基本达不到 ⇒ α 被单向推大** (换分母即翻号)"
                            : (dx.hBelowFrac() < 0.1)
                                  ? "H̄ 基本能达到 ⇒ α 被推小 (趋向下界)"
                                  : "两种方向都出现 (α 在中间找平衡)");
            std::printf("  critic 排序 : 合法槽位上 min(Q1,Q2) 的标准差均值 %.4f, "
                        "|均值| %.3f  (塌到 0 = Q 对搜索不再提供排序; PUCT 的 U 项量级 "
                        "≈ c_puct·P·√N)\n", dx.qSpreadMean, dx.qAbsMean);
            std::printf("  槽位/着法   : 平均 %.1f / %.1f (哈希碰撞平均挤掉 %.1f 个着法)\n",
                        dx.slotsMean, dx.legalMean, dx.legalMean - dx.slotsMean);
        }
    }
    std::printf("  学习步数    : %d, 回放池 %zu, 用时 %.1f s\n",
                sac.getLearnSteps(), sac.getMemorySize(), sec);

    if (!g.csv.empty()) {
        FILE *fp = std::fopen(g.csv.c_str(), "w");
        if (fp != nullptr) {
            /* [①] 诊断列: |Q| 与 α/y/clamp/熵 的口径都在这里, 目的是让"|Q| 区间 → 得分率"
               这条曲线能从 CSV 直接画出来 (而不是靠人工抄 log)。 */
            const DiagDelta dx(diagStart, sac.getTrainDiag());
            std::fprintf(fp, "label,legacy,rewardShape,gamma,games,pliesCap,mctsSims,sims,"
                             "seed,mctsSrand,train,"
                             "wins,losses,draws,drawCap,drawNoCapture60,drawRepetition,"
                             "drawOther,scoreRate,ciLo,ciHi,"
                             "meanPlies,jiangCaptured,checkmate,stalemate,"
                             "learnRewardSAC,learnRewardOpp,displayRewardSAC,displayRewardOpp,"
                             "sacMaterial,oppMaterial,lossPoints,lossMean,lossMax,"
                             "qAbsMean,qAbsMax,alphaFirst,alphaLast,clampFrac,yPreAbsMean,"
                             "vMean,vQMean,vEntMean,hMean,hBarMean,hBarSlotsMean,"
                             "hBelowHbarFrac,hBelowHbarSlotsFrac,qSpreadMean,"
                             "slotsMean,legalMean,"
                             "entropyInTarget,entropySlotsAsLegal,sparseLeaf,"
                             "qTargetAbsMean,normEntropyProbe,qTargetSpread,"
                             "targetTau,targetIter\n");
            std::fprintf(fp, "%s,%d,%d,%.4f,%d,%d,%d,%d,%u,%u,%d,"
                             "%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%d,%d,%d,"
                             "%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%d,%.6f,%.6f,"
                             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                             "%.4f,%.4f,%.4f,%.2f,%.2f,"
                             "%.2f,%d,%d,%.4f,%.4f,%.4f,%d\n",
                         g.label.c_str(), g.legacy ? 1 : 0, g.rewardShape, (double)g.gamma,
                         g.games, g.maxPlies, g.mctsSims, g.sims, g.seed, g.mctsSrand,
                         g.train ? 1 : 0,
                         st.wins, st.losses, st.draws, st.drawCap, st.noCapture60,
                         st.repetition, st.drawOther,
                         st.rate(), lo, hi, (double)st.plies / (double)n,
                         st.jiangCaptured, st.checkmate, st.stalemate,
                         learnSAC / n, learnOpp / n, dispSAC / n, dispOpp / n,
                         sacMatSum / n, oppMatSum / n,
                         lossTotal, lossTotal > 0 ? lossSum / (double)lossTotal : 0.0, lossMax,
                         probeQAbsMean, probeQAbsMax, dx.alphaFirst, dx.alphaLast,
                         dx.clampFrac(),
                         dx.yPreAbsMean, dx.vMean, dx.vQMean, dx.vEntMean,
                         dx.hMean, dx.hBarMean, dx.hBarSlotsMean,
                         dx.hBelowFrac(), dx.hBelowSlotsFrac(), dx.qSpreadMean,
                         dx.slotsMean, dx.legalMean,
                         (double)sac.entropyInTarget, sac.entropySlotsAsLegal ? 1 : 0,
                         sac.sparseLeafEval ? 1 : 0,
                         probeQTargetAbsMean, probeNormEntropy, probeQTargetSpread,
                         (double)sac.targetTau, sac.replaceTargetIter);
            std::fclose(fp);
            std::printf("  CSV         : %s\n", g.csv.c_str());
        } else {
            std::printf("  **CSV 写不出去**: %s\n", g.csv.c_str());
        }
    }
    delete sacPtr;
    return 0;
}
