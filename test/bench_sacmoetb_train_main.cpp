/*
 * bench_sacmoetb_train_main.cpp — TB 专家 SAC+AZ 的"自对弈训练 -> 对锚点评测"闭环
 * ============================================================================
 *
 * 解决什么问题
 * ------------
 * `dev-sacmoetb` 那一轮把 TB 专家的每样本训练代价降到 1/3.59、每模拟搜索代价降到
 * 1/5.0，但**一次对局 A/B 都没跑**（`docs/dev_sacmoetb_2026_09.md` §6.1 明确列成未测）。
 * "更快"不等于"更强"：本轮要回答的是**棋力**，而棋力只能由对局给出。
 *
 * 所以这个工具只做一件事，但把它的每一个前提都写死：
 *
 *   1. **锚点是 MCTS 与 ABAgent level=1**（`--opponent=ab|mcts|both`）。两者都是
 *      **纯搜索、无权重、不学习**，所以它们是一把不随训练漂移的尺子 ——
 *      这与 `docs/dev_sacmoetb_2026_09.md` 里"陪练与标尺"的口径一致。
 *      AB 深度 1 = 界面下拉框里的 `AGENT_AB_L1`（`chessboard.cpp` 的
 *      `AB_L1_DEPTH = 1`；这里刻意**不**手抄那个常量，而是在启动时打印它并断言
 *      与本工具的 `--ab-depth` 默认值一致 —— 单一来源在 chessboard.cpp，本工具只是引用）。
 *
 *   2. **固定开局集 + 成对换先**（做法抄自 `bench_anchor`）：开局由 (seed, index)
 *      确定性地生成，随机源是**局部** mt19937_64，**不碰 `RL::Random`** ——
 *      于是开局集与 agent 内部随机流无关，换个权重跑的还是同一批局面。
 *      每个开局下打两局（SAC 先执红、再执黑），先手优势在一阶上抵消。
 *      开局集打 FNV 指纹：两次运行指纹不同 = 开局集变了，此时结果**不可比**。
 *
 *   3. **区间而不是点**：报 W/D/L + 得分率 + 95% 区间 + Elo 差。区间跨过 50% 才叫
 *      "没测出差别"。只报一个数字必然被当成结论（`bench_anchor` 的同一套理由）。
 *
 *   4. **先训练再评测，且两头都可复现**：`--train-games=N` 先自对弈 N 局
 *      （`SACAZMoETbAgent::trainSelfPlay`），再评测。训练侧与评测侧的每一个旋钮都是
 *      命令行参数，默认值 = `SACAZMoETbAgent` 的当前默认 —— 于是"默认口径"是一个可复现的
 *      基线，而不是一个藏在源代码里的魔数。
 *
 * 与 MLP 专家的隔离（用户口径：**不许污染 MLP 专家相关的代码逻辑**）
 * ---------------------------------------------------------------
 * 本工具建的 agent 永远是 `Backbone::SparseMoeTb`（TB 专家），所有旋钮都写在
 * **这个进程里这个实例上**，一行都不改 `SACAZMoETbAgent` 的默认值、也不碰
 * `MlpExpert` / `SparseMoE<MlpExpert,...>` / `MOE_MLP_*` 的任何代码路径。
 * 回归证据：`mlp` 与 `moe-mlp` 两个骨干的走法序列与改动前逐行相同（见
 * `docs/dev_sacmoetb_2026_09.md` §6.0.1 的复现命令）。
 *
 * 用法
 * ----
 *   cmake --build <build> --target bench_sacmoetb_train
 *
 *   :: ① 基线：随机权重，不训练，看链路与"未训练到底多弱"
 *   bench_sacmoetb_train --train-games=0 --openings=8 --eval-plies=100 --eval-sims=40
 *
 *   :: ② 训练一轮再看（这是本工具的主用法）
 *   bench_sacmoetb_train --train-games=60 --train-sims=64 --openings=16 \
 *                        --eval-sims=40 --opponent=both --save=weights/tb_run1
 *
 *   :: ③ 等时间：按实测 ms/模拟 反推评测模拟数（"谁更快"只有在同一时间预算下才公平）
 *   bench_sacmoetb_train --load=weights/tb_run1 --sims-time=175 --openings=16
 *
 *   :: ④ 消融一个训练旋钮（例如 [F1] 目标网同步率，见 docs 的 F1）
 *   bench_sacmoetb_train --train-games=60 --target-tau=1 --target-iter=256 \
 *                        --openings=16 --csv=out/ab_tau256.csv
 *
 * 退出码
 *   0 = 机制全部成立（没有非法走法 / 无效走法 / NaN）；1 = 有机制性缺陷。
 *   **得分率不影响退出码** —— 棋力弱不是缺陷，是读数。
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "abagent.h"
#include "chess.h"
#include "mcts.h"
#include "sacazmoetbagent.h"   /* [2026-09 独立类] 本工具固定跑 TB 专家那一支 */
#include "rl/util.hpp"

/*
   刻意**不**写 `using namespace RL;` —— `RL::Step`(一条经验) 与 `::Step`(象棋的一步)
   同名，把 RL 整个拉进全局作用域会让每个 `Step` 变成歧义符号。
*/

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    /* ---- 对局锚点 ---- */
    std::string opponent = "both";   /* ab | mcts | both */
    int abDepth = 1;                 /* ABAgent 深度: 1 = 界面的 AGENT_AB_L1 */
    int mctsIters = 400;             /* MCTS 每步迭代数 (与界面 MCTS 那一档同量级) */
    unsigned mctsSrand = 12345u;     /* MCTS / 随机走子用的 std::rand 种子 (必须能固定) */

    /* ---- 训练 (自对弈) ---- */
    int trainGames = 0;              /* 0 = 不训练 (纯评测: 随机权重或 --load 的权重) */
    int trainSims = 64;              /* 自对弈每步模拟数 */
    int trainPlies = 120;            /* 自对弈每局手数上限 */
    int learnEvery = 4;              /* 自对弈里每几手更新一次 */
    float tempRoot = 1.0f;           /* 自对弈根温度 (开局多样性) */
    float tempFinal = 0.25f;         /* 自对弈末段温度 */
    /*
       `learnFromSearch` 默认 **关**。为什么 (一个必须写清楚的交互):
       `SACAZMoETbAgent::trainSelfPlay` 自己就会 `selectMove(..., &piTarget)` 并把
       (s, π_MCTS, a, r, s', done) 存进回放池、每 `learnEveryMoves` 手更新一次。
       而 `learnFromSearch=true` 会让 `selectMove` **再**存一条同样的经验并**再**跑一次
       `learnBatch` —— 于是每个手都更新一次、池里每条经验都存两份, 训练节拍变成
       "`--learn-every` 基本失效"。那不是错, 但是另一套协议; 要它就是显式开
       `--learn-from-search=1`, 并且知道池与更新次数都会翻倍。
       评测阶段同样默认关: 否则评测中的对局也在改权重, "这一版权重有多强"就答不了。
    */
    bool learnFromSearch = false;

    /* ---- 评测 ---- */
    int openings = 8;                /* 开局集大小 (每个开局两局 -> 总对局数 = 2*openings) */
    int openingPlies = 4;            /* 每个开局的随机手数 */
    int evalPlies = 100;             /* 评测每局手数上限 */
    int evalSims = 40;               /* 评测每步模拟数 (GUI 的 SACAZ_MOE_SIMS = 40) */
    double simsTimeMs = 0.0;         /* >0 = 等时间: 标定 ms/模拟 后反推 evalSims */

    /* ---- 训练侧旋钮 (默认全部 = SACAZMoETbAgent 的当前默认) ---- */
    float lr = 0.001f;
    int batch = 32;
    int epochs = 1;                  /* replayEpochs */
    float targetTau = 1e-3f;
    int targetIter = 64;
    float entropyRatio = 0.98f;
    float alphaLr = 1e-3f;
    float azWeight = 1.0f;
    float cpuct = 1.5f;
    float clampTarget = 2.0f;
    float huberDelta = 1.0f;
    float aux = 0.1f;
    float gamma = 0.99f;
    int memory = 4096;
    int rewardShape = 0;
    /*
       ---- [2026-09 动态奖励分配] rewardShape=3 的三个旋钮 ----
       用户口径 (两轮): ① "前期加强吃子奖励 / 后期重杀将奖励, 两者都不应忽视";
       ② "分阶段太过思维定势, 因为局势是反复变化的, 是否可以参考棋子的数量和价值来
       评估局面动态调整杀棋杀将的奖励比例"。
       所以这一版**没有阶段钟**: 权重由一个局面评估 e ∈ [0,1] 决定 (同时读棋子的
       **数量** / **价值** / 双方**子力差**), 局势反复时 e 会退回, 权重跟着回摆。
       设计、量纲不变量与已知代价见 src/sacazagent.h 的 `rewardShape = 3` 一节,
       实测见 docs/sacmoetb_pos_reward_2026_09.md。
         mateScoreMode   0 = 数量+价值+子力差 (默认) / 1 = 只用价值 / 2 = 只用数量 / 3 = 数量+价值
         matRewardBoost  均势满盘 (e=0) 时材质系数 x (1+boost)
         mateRewardBoost 残局/大势已定 (e=1) 时终局值 x (1+boost)
       默认 0.5 / 0.5 -> 两个倍数之和恒为 2.5 = "固定预算按局面动态分配"。
       只有 `--reward-shape=3` 才读这三个数; 其它 shape 下一位都不变。
    */
    int mateScoreMode = 0;
    float matRewardBoost = 0.5f;
    float mateRewardBoost = 0.5f;
    float valueScale = 1.0f;
    /*
       ------------------------------------------------------------------
       即时奖励的整体缩放 (`rewardScale`, 默认 1.0 = 与改动前逐位一致)
       ------------------------------------------------------------------
       为什么这是本轮最该试的一个训练旋钮: 材料诊断 (`--eval` 汇总里那一行
       "材料差(SAC视角)") 实测**未训练**的 TB agent 对 AB L1 **每一局都落后**
       (32 局: 占优 0 / 落后 32, 平均 −2.48), 也就是它在**送子**。
       而它为什么看不见子力, 一算就清楚:

           吃一个马/炮的即时奖励 = 0.029   (test_sacaz [15] 的读数)
           终局                      = ±1
           熵项 α·H                  ≈ 1.9   (见文档 §3.1)

       也就是说**子力信号比熵偏置小 65 倍、比终局小 34 倍**, 在 TD 目标里基本被淹没。
       `rewardScale` 只缩放**即时项**、不动终局 ±1 (终局是环境的真值, 缩放它等于换一个
       游戏), 所以它是"把子力抬到与终局同一量级"的唯一旋钮。
       **它会改变 MDP 的最优策略** (更贪子), 所以这是一个需要实测的取舍, 不是白拿。
    */
    float rewardScale = 1.0f;
    /*
       ---- [2026-09 实验轮] 用户提议的三个旋钮 + 两个"修好版" (全部默认关) ----
       详见 src/sacazagent.h 里各成员的注释与 docs/dev_sacmoetb_strength_2026_09.md。
    */
    bool criticTanh = false;         /* 提议①: Q = tanh(z) */
    float rewardTanhGain = 0.0f;     /* 提议②: 即时奖励 -> tanh(gain * r) */
    float alphaGumbelSigma = 0.0f;   /* 提议③: α 每样本乘 exp(sigma*(G-gamma)) */
    float alphaCeiling = 5.0f;       /* α 上界 (默认 5.0 = 改动前) */
    bool entropyCenter = false;      /* V 里熵项改成 alpha*(H - log n) */
    bool sparseLeaf = true;
    /*
       [2026-09 ①] 熵项的**去处**。这是本工具最想 A/B 的一个旋钮, 因为本机的训练诊断
       (见 printDiag 打的 "V(s')均 / 其中熵项均") 直接把那条因果摊开了:
       软价值 V(s) = E_π[min Q] + α·H(π), 而 TD 目标是 y = r − γ·V(s'),
       于是**熵项以 −γ·α·H(s') 的形式进了 critic 的回归目标**, 而它与棋局无关。
       实测 (随机权重, 300 个样本): V(s')均 = 0.7096, 其中熵项均 = 0.7578 ——
       也就是说自举项里 E[min Q] 只有 −0.048, **几乎全是 α·H 这个常数偏置**。
       `--entropy-in-target=0` 把熵项**移出** V(s') (只留在策略损失里),
       用来单独检验"是熵项把 critic 顶走的"这条因果, 而不是只看相关性。
    */
    float entropyInTarget = 1.0f;
    bool entropySlots = false;

    /* ---- 结构口径 ---- */
    std::string trunk = "shared";    /* shared | separate */
    std::string tbHeads = "honor";   /* honor | legacy */

    /* ---- 其它 ---- */
    unsigned seed = 20240901u;
    std::string loadPrefix;
    std::string savePrefix;
    std::string csvPath;
    bool quiet = false;
    bool verbose = false;
};

Cfg g_cfg;

/* ============================================================
 *  小工具
 * ============================================================ */
double nowMs()
{
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int flipColor(int c)
{
    return (c == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}

const char *resultName(int r)
{
    if (r == Chess::RESULT_RED_WIN) { return "红胜"; }
    if (r == Chess::RESULT_BLACK_WIN) { return "黑胜"; }
    if (r == Chess::RESULT_DRAW) { return "和"; }
    return "未终局";
}

/*
 *  和棋原因: 这个引擎只有两类规则和棋 (见 chess.h 的 DrawReason)。
 *  **困毙不在这里** —— 象棋里困毙是**负**, 由 getResult 报成对方的 RESULT_*_WIN。
 */
const char *drawReasonName(int dr)
{
    switch ((Chess::DrawReason)dr) {
    case Chess::DRAW_REPEAT:       return "重复";
    case Chess::DRAW_NO_CAPTURE60: return "60回合";
    default:                       return "台架截断";
    }
}

/* 得分率 -> Elo 差 (红黑视角无所谓, 差值是相对的) */
double eloFromScore(double s)
{
    if (s <= 0.0) { return -800.0; }
    if (s >= 1.0) { return 800.0; }
    return -400.0 * std::log10(1.0 / s - 1.0);
}

/* ============================================================
 *  固定开局集 (与 bench_anchor 同一套做法: 局部 RNG, 不碰 RL::Random)
 * ============================================================ */
typedef std::vector<Step> Opening;

unsigned long long fnv1a(unsigned long long h, unsigned long long v)
{
    h ^= v;
    return h * 1099511628211ull;
}

bool makeOpenings(Chess &scratch, int count, int plies, unsigned seed,
                  std::vector<Opening> &out, unsigned long long &fingerprint)
{
    out.clear();
    fingerprint = 1469598103934665603ull;
    for (int i = 0; i < count; i++) {
        std::mt19937_64 rng((unsigned long long)seed * 7919ull
                            + (unsigned long long)i * 104729ull + 17ull);
        scratch.reset();
        Opening op;
        for (int p = 0; p < plies; p++) {
            if (scratch.getResult(scratch.sideToMove) != Chess::RESULT_ONGOING) { break; }
            std::vector<Step *> legal;
            scratch.sample(scratch.sideToMove, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                break;
            }
            std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
            const Step s = *legal[(std::size_t)pick(rng)];
            Steps::instance().put(legal);
            op.push_back(s);
            double dummy = 0.0;
            scratch.moveForward(&s, dummy);
        }
        if (op.empty()) {
            std::printf("[错误] 第 %d 个开局造不出来 (--opening 的随机手数太多?)\n", i);
            return false;
        }
        for (std::size_t k = 0; k < op.size(); k++) {
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.y);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.y);
        }
        out.push_back(op);
    }
    return true;
}

int applyOpening(Chess &c, const Opening &op)
{
    c.reset();
    for (std::size_t i = 0; i < op.size(); i++) {
        double dummy = 0.0;
        c.moveForward(&op[i], dummy);
    }
    return c.sideToMove;
}

/* ============================================================
 *  对手: 一个极小的多态壳, 让 AB 与 MCTS 走同一条对局循环
 *
 *  两个对手都是**纯搜索、无权重、不学习** —— 这正是它们能当尺子的原因
 *  (一个会随训练漂移的对手答不了"这一版比上一版强吗")。
 * ============================================================ */
struct Opponent {
    virtual ~Opponent() {}
    virtual Step move(int color) = 0;
    virtual const char *name() const = 0;
    virtual const char *detail() const = 0;

    struct Stats {
        long long moves = 0;
        double ms = 0.0;
    } st;
};

struct AbOpponent : Opponent {
    ABAgent ab;
    explicit AbOpponent(Chess &c, int depth) : ab(c, depth) {}
    Step move(int color) override { return ab.getBestMove(color); }
    const char *name() const override { return "AB"; }
    const char *detail() const override { return ""; }
};

struct MctsOpponent : Opponent {
    MCTS m;
    int iters;
    MctsOpponent(Chess &c, int it) : m(c, 1.414), iters(it) {}
    Step move(int color) override { return m.findBestMove(color, iters); }
    const char *name() const override { return "MCTS"; }
    const char *detail() const override { return ""; }
};

/* ============================================================
 *  一局: SAC+AZ(TB) 对锚点
 *  双方共用**同一张对局棋盘** (与 bench_sacaz_vs_ab 同做法: 搜索内部会
 *  moveForward/moveBack 并复原, 所以共享是安全的)。
 * ============================================================ */
struct GameRec {
    int result = Chess::RESULT_ONGOING;
    int drawReason = (int)Chess::DRAW_NONE;
    int plies = 0;
    bool truncated = false;
    bool sacWasRed = true;
    bool broken = false;            /* 非法/无效走法 —— 这是机制性缺陷, 不是棋力 */
    const char *brokenWhy = "";
    double sacMs = 0.0;
    int sacMoves = 0;
    double oppMs = 0.0;
    int oppMoves = 0;
    double redMaterial = 0.0, blackMaterial = 0.0;
};

/* 得分 (SAC 视角): 胜 1 / 和 0.5 / 负 0 */
double scoreOf(const GameRec &g)
{
    if (g.result == Chess::RESULT_DRAW) { return 0.5; }
    const bool redWon = (g.result == Chess::RESULT_RED_WIN);
    return (redWon == g.sacWasRed) ? 1.0 : 0.0;
}

void materialOf(Chess &c, double &red, double &black)
{
    red = 0.0;
    black = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.m_children[i];
        if (s == nullptr || !s->alive) { continue; }
        if (s->color == Stone::COLOR_RED) { red += s->value; } else { black += s->value; }
    }
}

GameRec playGame(Chess &c, SACAZMoETbAgent &sac, Opponent &opp,
                 const Opening &op, bool sacIsRed, int simsForSac)
{
    GameRec g;
    g.sacWasRed = sacIsRed;

    int turn = applyOpening(c, op);

    while (g.plies < g_cfg.evalPlies) {
        Chess::DrawReason dr = Chess::DRAW_NONE;
        const int res = c.getResult(turn, &dr);
        if (res != Chess::RESULT_ONGOING) {
            g.result = res;
            g.drawReason = (int)dr;
            break;
        }

        const bool sacTurn = (turn == Stone::COLOR_RED) == sacIsRed;
        const double t0 = nowMs();
        const Step s = sacTurn ? sac.selectMove(turn, simsForSac, 0.0f)
                               : opp.move(turn);
        const double ms = nowMs() - t0;
        if (sacTurn) { g.sacMs += ms; g.sacMoves++; }
        else         { g.oppMs += ms; g.oppMoves++; }

        /*
           **逐手合法性校验**, 并且在 moveForward **之前**做: 棋盘一旦吃了非法走法,
           后面所有读数都失去意义。分开记"有合法走法却返回无效走法"与"返回了非法走法"
           —— 前者是 agent 的兜底逻辑坏了, 后者是它走了一步不存在的棋。
        */
        std::vector<Step*> legal;
        c.sample(turn, legal);
        const bool anyLegal = !legal.empty();
        bool legalMove = false;
        if (anyLegal && s.valid) { legalMove = c.isLegalMove(turn, &s); }
        Steps::instance().put(legal);

        if (!s.valid || !legalMove) {
            g.broken = true;
            g.brokenWhy = sacTurn ? (s.valid ? "SAC+AZ 走了非法着法" : "SAC+AZ 返回无效着法")
                                  : (s.valid ? "锚点走了非法着法" : "锚点返回无效着法");
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-4s (%d,%d)->(%d,%d) %7.1f ms%s\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        sacTurn ? "  [SAC+AZ]" : "  [锚点]");
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = flipColor(turn);
        g.plies++;
    }

    if (g.result == Chess::RESULT_ONGOING && !g.broken) {
        /* 走到手数上限: 这是**台架截断**, 不是规则和棋 —— 单列一栏 (与 bench_anchor
           同一条纪律: 混进"和棋"会让和棋率无法解释)。 */
        g.truncated = true;
        g.result = Chess::RESULT_DRAW;
        g.drawReason = (int)Chess::DRAW_NONE;
    }

    materialOf(c, g.redMaterial, g.blackMaterial);
    return g;
}

/* ============================================================
 *  一个锚点的评测结果
 * ============================================================ */
struct EvalResult {
    std::string oppName;
    int games = 0, wins = 0, draws = 0, losses = 0, broken = 0, truncated = 0;
    double score = 0.0, scoreLo = 0.0, scoreHi = 0.0;
    double scoreSd = 0.0;
    double sacMsPerMove = 0.0, oppMsPerMove = 0.0;
    double avgPlies = 0.0;
    /*
       ---- 材料差 (SAC 视角, 终局时的红-黑子力再按 SAC 的颜色取号) ----
       为什么必须报它: "输给一个**深度 1** 的对手"这句话本身没有诊断价值 ——
       深度 1 是"能吃就吃"的贪心, 所以输给它几乎只有一条路: **送子**。
       材料差把这个假设变成读数: 如果 SAC 的平均材料差明显为负, 那它的问题不是
       "搜索不够深", 而是"叶子估值看不见子力"; 如果材料差≈0 而仍然输, 那问题在别处
       (例如被将死而没丢子 —— 那是纯粹的价值函数/搜索问题)。
    */
    double avgMaterialDiff = 0.0;      /* 正值 = SAC 一方子力占优 */
    int sacMaterialWorse = 0, sacMaterialBetter = 0;
    int drawRepetition = 0, drawHalfmove = 0, drawStalemate = 0, drawByTruncation = 0;
    /* 分胜负的终局里 SAC 的胜方是不是"将死"这类 —— 只统计, 不下结论 */
    int sacMates = 0, oppMates = 0;
};

EvalResult evaluate(Chess &c, SACAZMoETbAgent &sac, Opponent &opp,
                    const std::vector<Opening> &book, int simsForSac)
{
    EvalResult r;
    r.oppName = opp.name();

    std::vector<double> scores;
    scores.reserve(book.size() * 2);
    double sacMs = 0.0, oppMs = 0.0;
    long long sacMoves = 0, oppMoves = 0;
    long long pliesSum = 0;
    double matSum = 0.0;

    for (std::size_t i = 0; i < book.size(); i++) {
        for (int side = 0; side < 2; side++) {
            const bool sacIsRed = (side == 0);
            const GameRec g = playGame(c, sac, opp, book[i], sacIsRed, simsForSac);
            r.games++;
            const double sc = scoreOf(g);
            scores.push_back(sc);
            if (g.broken) { r.broken++; }
            else if (sc >= 0.999) { r.wins++; }
            else if (sc <= 0.001) { r.losses++; }
            else { r.draws++; }

            if (g.truncated) { r.truncated++; r.drawByTruncation++; }
            else if (g.result == Chess::RESULT_DRAW) {
                switch ((Chess::DrawReason)g.drawReason) {
                case Chess::DRAW_REPEAT:       r.drawRepetition++; break;
                case Chess::DRAW_NO_CAPTURE60: r.drawHalfmove++;   break;
                default: break;
                }
            }
            if (!g.broken && g.result != Chess::RESULT_DRAW) {
                const bool redWon = (g.result == Chess::RESULT_RED_WIN);
                if (redWon == g.sacWasRed) { r.sacMates++; } else { r.oppMates++; }
            }

            sacMs += g.sacMs;  sacMoves += g.sacMoves;
            oppMs += g.oppMs;  oppMoves += g.oppMoves;
            pliesSum += g.plies;

            /* 材料差按 SAC 的颜色取号 (它执红时红多 = 它占优) */
            {
                const double mine = g.sacWasRed ? g.redMaterial : g.blackMaterial;
                const double theirs = g.sacWasRed ? g.blackMaterial : g.redMaterial;
                const double d = mine - theirs;
                matSum += d;
                if (d < -1e-9) { r.sacMaterialWorse++; }
                else if (d > 1e-9) { r.sacMaterialBetter++; }
            }

            if (!g_cfg.quiet) {
                std::printf("      g%-3zu %-4s SAC=%-4s  %-3d 手  %-10s %s%s\n",
                            i * 2 + (std::size_t)side + 1,
                            sacIsRed ? "RED" : "BLK",
                            resultName(g.result), g.plies,
                            g.truncated ? "台架截断" : (g.result == Chess::RESULT_DRAW
                                                            ? drawReasonName(g.drawReason)
                                                            : "分胜负"),
                            g.broken ? "**" : "",
                            g.broken ? g.brokenWhy : "");
            }
        }
    }

    const int n = (int)scores.size();
    double sum = 0.0;
    for (int i = 0; i < n; i++) { sum += scores[(std::size_t)i]; }
    r.score = (n > 0) ? sum / (double)n : 0.0;
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = scores[(std::size_t)i] - r.score;
        var += d * d;
    }
    r.scoreSd = (n > 1) ? std::sqrt(var / (double)(n - 1)) : 0.0;
    const double se = (n > 1) ? r.scoreSd / std::sqrt((double)n) : 0.0;
    r.scoreLo = std::max(0.0, r.score - 1.96 * se);
    r.scoreHi = std::min(1.0, r.score + 1.96 * se);

    r.sacMsPerMove = (sacMoves > 0) ? sacMs / (double)sacMoves : 0.0;
    r.oppMsPerMove = (oppMoves > 0) ? oppMs / (double)oppMoves : 0.0;
    r.avgPlies = (n > 0) ? (double)pliesSum / (double)n : 0.0;
    r.avgMaterialDiff = (n > 0) ? matSum / (double)n : 0.0;
    return r;
}

/* ============================================================
 *  参数解析
 * ============================================================ */
bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *k) -> const char* {
            const std::size_t n = std::strlen(k);
            if (std::strncmp(a, k, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--opponent"))       { g_cfg.opponent = v; }
        else if (const char *v = val("--ab-depth"))  { g_cfg.abDepth = std::atoi(v); }
        else if (const char *v = val("--mcts-iters")){ g_cfg.mctsIters = std::atoi(v); }
        else if (const char *v = val("--mcts-srand")){ g_cfg.mctsSrand = (unsigned)std::atoi(v); }
        else if (const char *v = val("--train-games")){ g_cfg.trainGames = std::atoi(v); }
        else if (const char *v = val("--train-sims")){ g_cfg.trainSims = std::atoi(v); }
        else if (const char *v = val("--train-plies")){ g_cfg.trainPlies = std::atoi(v); }
        else if (const char *v = val("--learn-every")){ g_cfg.learnEvery = std::atoi(v); }
        else if (const char *v = val("--temp-root")) { g_cfg.tempRoot = (float)std::atof(v); }
        else if (const char *v = val("--temp-final")){ g_cfg.tempFinal = (float)std::atof(v); }
        else if (const char *v = val("--learn-from-search")){ g_cfg.learnFromSearch = (std::atoi(v) != 0); }
        else if (const char *v = val("--openings"))  { g_cfg.openings = std::atoi(v); }
        else if (const char *v = val("--opening"))   { g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--eval-plies")){ g_cfg.evalPlies = std::atoi(v); }
        else if (const char *v = val("--eval-sims")) { g_cfg.evalSims = std::atoi(v); }
        else if (const char *v = val("--sims-time")) { g_cfg.simsTimeMs = std::atof(v); }
        else if (const char *v = val("--lr"))        { g_cfg.lr = (float)std::atof(v); }
        else if (const char *v = val("--batch"))     { g_cfg.batch = std::atoi(v); }
        else if (const char *v = val("--epochs"))    { g_cfg.epochs = std::atoi(v); }
        else if (const char *v = val("--target-tau")){ g_cfg.targetTau = (float)std::atof(v); }
        else if (const char *v = val("--target-iter")){ g_cfg.targetIter = std::atoi(v); }
        else if (const char *v = val("--entropy-ratio")){ g_cfg.entropyRatio = (float)std::atof(v); }
        else if (const char *v = val("--alpha-lr"))  { g_cfg.alphaLr = (float)std::atof(v); }
        else if (const char *v = val("--az-weight")) { g_cfg.azWeight = (float)std::atof(v); }
        else if (const char *v = val("--cpuct"))     { g_cfg.cpuct = (float)std::atof(v); }
        else if (const char *v = val("--clamp"))     { g_cfg.clampTarget = (float)std::atof(v); }
        else if (const char *v = val("--huber"))     { g_cfg.huberDelta = (float)std::atof(v); }
        else if (const char *v = val("--aux"))       { g_cfg.aux = (float)std::atof(v); }
        else if (const char *v = val("--gamma"))     { g_cfg.gamma = (float)std::atof(v); }
        else if (const char *v = val("--memory"))    { g_cfg.memory = std::atoi(v); }
        else if (const char *v = val("--reward-shape")){ g_cfg.rewardShape = std::atoi(v); }
        else if (const char *v = val("--pos-score-mode")){ g_cfg.mateScoreMode = std::atoi(v); }
        else if (const char *v = val("--pos-mat-boost")){ g_cfg.matRewardBoost = (float)std::atof(v); }
        else if (const char *v = val("--pos-mate-boost")){ g_cfg.mateRewardBoost = (float)std::atof(v); }
        else if (const char *v = val("--value-scale")){ g_cfg.valueScale = (float)std::atof(v); }
        else if (const char *v = val("--reward-scale")){ g_cfg.rewardScale = (float)std::atof(v); }
        else if (const char *v = val("--critic-tanh")){ g_cfg.criticTanh = (std::atoi(v) != 0); }
        else if (const char *v = val("--reward-tanh")){ g_cfg.rewardTanhGain = (float)std::atof(v); }
        else if (const char *v = val("--alpha-gumbel")){ g_cfg.alphaGumbelSigma = (float)std::atof(v); }
        else if (const char *v = val("--alpha-ceiling")){ g_cfg.alphaCeiling = (float)std::atof(v); }
        else if (const char *v = val("--entropy-center")){ g_cfg.entropyCenter = (std::atoi(v) != 0); }
        else if (const char *v = val("--sparse-leaf")){ g_cfg.sparseLeaf = (std::atoi(v) != 0); }
        else if (const char *v = val("--entropy-in-target")){ g_cfg.entropyInTarget = (float)std::atof(v); }
        else if (const char *v = val("--entropy-slots")){ g_cfg.entropySlots = (std::atoi(v) != 0); }
        else if (const char *v = val("--trunk"))     { g_cfg.trunk = v; }
        else if (const char *v = val("--tb-heads"))  { g_cfg.tbHeads = v; }
        else if (const char *v = val("--seed"))      { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--load"))      { g_cfg.loadPrefix = v; }
        else if (const char *v = val("--save"))      { g_cfg.savePrefix = v; }
        else if (const char *v = val("--csv"))       { g_cfg.csvPath = v; }
        else if (std::strcmp(a, "--quiet") == 0)     { g_cfg.quiet = true; }
        else if (std::strcmp(a, "--verbose") == 0)   { g_cfg.verbose = true; }
        else {
            std::printf("[错误] 未知参数: %s\n", a);
            return false;
        }
    }

    /* 取值校验: 打错一个字母就必须当场失败 —— 静默退回默认 = 量的是另一个口径,
       而报告里只差一个词 (与 bench_sacaz_vs_ab 的同一套纪律)。 */
    if (g_cfg.opponent != "ab" && g_cfg.opponent != "mcts" && g_cfg.opponent != "both") {
        std::printf("[错误] --opponent 只接受 ab / mcts / both, 收到 '%s'\n",
                    g_cfg.opponent.c_str());
        return false;
    }
    if (g_cfg.trunk != "shared" && g_cfg.trunk != "separate") {
        std::printf("[错误] --trunk 只接受 shared / separate, 收到 '%s'\n",
                    g_cfg.trunk.c_str());
        return false;
    }
    if (g_cfg.tbHeads != "honor" && g_cfg.tbHeads != "legacy") {
        std::printf("[错误] --tb-heads 只接受 honor / legacy, 收到 '%s'\n",
                    g_cfg.tbHeads.c_str());
        return false;
    }
    if (g_cfg.openings < 1) { g_cfg.openings = 1; }
    if (g_cfg.evalPlies < 2) { g_cfg.evalPlies = 2; }
    if (g_cfg.evalSims < 1) { g_cfg.evalSims = 1; }
    if (g_cfg.trainSims < 1) { g_cfg.trainSims = 1; }
    if (g_cfg.batch < 1) { g_cfg.batch = 1; }
    if (g_cfg.learnEvery < 1) { g_cfg.learnEvery = 1; }
    return true;
}

/* ============================================================
 *  一份配置快照 (打印用: "这一跑到底用的是哪一套口径")
 * ============================================================ */
void printConfig(const SACAZMoETbAgent &sac)
{
    std::printf("口径      : %s | trunk=%s tb-heads=%s | 头 请求%d/实际%d/分配%d d_k=%d 元素%lld\n",
                sac.backboneName(),
                SACAZMoETbAgent::trunkModeName(sac.trunkMode), g_cfg.tbHeads.c_str(),
                sac.tbHeadsRequested(), sac.tbHeadsUsed(), sac.tbHeadsAllocated(),
                sac.tbHeadDim(), sac.tbAttentionElements());
    std::printf("训练口径  : lr=%.4g batch=%d epochs=%d memory=%d gamma=%.3f | "
                "target tau=%.4g / 每%d步 | entropy=%.3f alphaLr=%.4g | azW=%.2f cpuct=%.2f\n",
                (double)g_cfg.lr, g_cfg.batch, g_cfg.epochs, g_cfg.memory,
                (double)g_cfg.gamma, (double)g_cfg.targetTau, g_cfg.targetIter,
                (double)g_cfg.entropyRatio, (double)g_cfg.alphaLr,
                (double)g_cfg.azWeight, (double)g_cfg.cpuct);
    std::printf("            : clamp=%.3g huber=%.3g aux=%.3g rewardShape=%d valueScale=%.3g rewardScale=%.3g "
                "sparseLeaf=%d learnFromSearch=%d entropyInTarget=%.3g entropySlots=%d\n"
                "            : [动态奖励] mateScoreMode=%d matBoost=%.3g mateBoost=%.3g (只有 rewardShape=3 读)\n"
                "            : [实验轮] criticTanh=%d rewardTanhGain=%.3g alphaGumbel=%.3g alphaCeiling=%.3g entropyCenter=%d\n",
                (double)g_cfg.clampTarget, (double)g_cfg.huberDelta, (double)g_cfg.aux,
                g_cfg.rewardShape, (double)g_cfg.valueScale, (double)g_cfg.rewardScale,
                (int)g_cfg.sparseLeaf, (int)g_cfg.learnFromSearch,
                (double)g_cfg.entropyInTarget, (int)g_cfg.entropySlots,
                g_cfg.mateScoreMode, (double)g_cfg.matRewardBoost,
                (double)g_cfg.mateRewardBoost,
                (int)g_cfg.criticTanh, (double)g_cfg.rewardTanhGain,
                (double)g_cfg.alphaGumbelSigma, (double)g_cfg.alphaCeiling,
                (int)g_cfg.entropyCenter);
    std::printf("参数量    : 唯一 %lld (actor.paramCount()=%lld, 共享口径下它会把骨干重复计入三张视图)\n",
                sac.uniqueParamCount(), sac.actor.paramCount());
}

/* 训练完/评测完的 agent 诊断 (只读) */
void printDiag(const SACAZMoETbAgent &sac, const char *tag)
{
    std::vector<long long> usage;
    sac.moeUsage(usage);
    long long lo = -1, hi = 0, tot = 0;
    int unused = 0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        tot += usage[i];
        if (usage[i] == 0) { unused++; }
        if (lo < 0 || usage[i] < lo) { lo = usage[i]; }
        if (usage[i] > hi) { hi = usage[i]; }
    }
    std::printf("  %s诊断: learnSteps=%d 池=%zu alpha=%.4f 叶子评估=%lld 专家使用{",
                tag, sac.getLearnSteps(), sac.getMemorySize(), (double)sac.getAlpha(),
                sac.getLeafEvals());
    for (std::size_t i = 0; i < usage.size(); i++) {
        std::printf("%lld%s", usage[i], (i + 1 < usage.size()) ? "," : "}");
    }
    std::printf(" 没用到的专家=%d max/min=%.2f\n", unused,
                (lo > 0) ? (double)hi / (double)lo : 0.0);
    const SACAZMoETbAgent::TrainDiag &D = sac.getTrainDiag();
    if (D.n > 0) {
        std::printf("  训练诊断: 样本=%lld 被夹=%lld(%.1f%%) |y|均=%.4f |Q|均=%.4f "
                    "Q间距均=%.4f H均=%.4f H目标均=%.4f V(s')均=%.4f (其中熵项均=%.4f)\n",
                    D.n, D.clamped, 100.0 * (double)D.clamped / (double)D.n,
                    D.yPreAbsSum / (double)D.n, D.qAbsMeanSum / (double)D.n,
                    D.qSpreadSum / (double)D.n, D.hSum / (double)D.n,
                    D.hBarSum / (double)D.n, D.vNextSum / (double)D.n,
                    D.vEntSum / (double)D.n);
        /*
           [2026-09 动态奖励分配] **终局通道的样本量** —— 这一行是判"后期重杀将加权"
           这类旋钮活/死的唯一读数: 只有 done 且**分胜负**的样本携带杀将奖励
           (和棋的终局值恒 0)。实测: 60 局自对弈里 decisive 只有几十条 (占比 <0.1%),
           把 mateBoost 从 0 开到 10 权重逐字节相同 —— 旋钮在训练里是死的。
        */
        std::printf("  终局通道: done 样本=%lld (%.4f%%) 其中分胜负=%lld (%.4f%%) 外部补入=%lld\n",
                    D.doneSamples, 100.0 * (double)D.doneSamples / (double)D.n,
                    D.decisiveSamples, 100.0 * (double)D.decisiveSamples / (double)D.n,
                    D.externalTerminals);
    }
}

} // namespace

/* ============================================================
 *  main
 * ============================================================ */
int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (!parseArgs(argc, argv)) { return 1; }

    std::printf("=== bench_sacmoetb_train: TB 专家 SAC+AZ 的训练 + 对锚点评测 ===\n");
    std::printf("SIMD=%s  STATE_DIM=%d ACTION_DIM=%d seed=%u\n",
                RL::simdops::instructionSet(), SACAZMoETbAgent::STATE_DIM,
                SACAZMoETbAgent::ACTION_DIM, g_cfg.seed);

    /*
       MCTS 的对局随机性走 `std::rand()`, 而 `std::srand` 默认播的是 time(nullptr)
       —— 不固定的话每次运行的对局随机性都不同, 跨版本比较就变成"每跑一次换一副牌"。
       (同 bench_sac_mcts_min 的 --mcts-srand。)
    */
    std::srand(g_cfg.mctsSrand);
    RL::Random::setSeed(g_cfg.seed);

    /*
       ---- 开局集先造, 且**在 agent 构造与任何 RL::Random 消耗之前** ----
       为什么顺序重要: 开局用局部 mt19937_64, 所以它本来就与 agent 随机流无关; 但
       "先造好" 让这件事在代码上也成立 —— 后人在中间插一句抽随机数的代码也不会
       悄悄换掉开局集 (换了的话指纹会变, 而指纹印在报告第一行)。
    */
    Chess scratch;
    scratch.reset();
    std::vector<Opening> book;
    unsigned long long bookFp = 0;
    if (!makeOpenings(scratch, g_cfg.openings, g_cfg.openingPlies, g_cfg.seed,
                      book, bookFp)) {
        return 1;
    }
    std::printf("开局集    : %d 个开局 x %d 随机手, FNV 指纹 %016llx (%d 局对局)\n",
                g_cfg.openings, g_cfg.openingPlies, bookFp, (int)book.size() * 2);

    /* ---- 对局棋盘 (与两个 agent 共用) ---- */
    Chess c;
    c.reset();

    const SACAZMoETbAgent::TrunkMode trunkMode =
        (g_cfg.trunk == "shared") ? SACAZMoETbAgent::TrunkMode::Shared
                                  : SACAZMoETbAgent::TrunkMode::Separate;
    const bool honorHeads = (g_cfg.tbHeads == "honor");

    const double tBuild0 = nowMs();
    /* [2026-09 独立类] 骨干硬编码在类里 —— 构造签名少了一个 Backbone 实参 */
    SACAZMoETbAgent sac(c, 64, g_cfg.gamma, g_cfg.lr, g_cfg.cpuct,
                        64, g_cfg.aux, trunkMode, honorHeads);
    /* 训练/评测侧旋钮: 只写在**这个实例**上, 不改任何默认值 */
    sac.batchSize = g_cfg.batch;
    sac.replayEpochs = g_cfg.epochs;
    sac.targetTau = g_cfg.targetTau;
    sac.replaceTargetIter = g_cfg.targetIter;
    sac.entropyRatio = g_cfg.entropyRatio;
    sac.learningRateAlpha = g_cfg.alphaLr;
    sac.azWeight = g_cfg.azWeight;
    sac.clampTarget = g_cfg.clampTarget;
    sac.huberDelta = g_cfg.huberDelta;
    sac.maxMemorySize = (std::size_t)g_cfg.memory;
    sac.rewardShape = g_cfg.rewardShape;
    sac.mateScoreMode = g_cfg.mateScoreMode;
    sac.matRewardBoost = g_cfg.matRewardBoost;
    sac.mateRewardBoost = g_cfg.mateRewardBoost;
    sac.valueScale = g_cfg.valueScale;
    sac.rewardScale = g_cfg.rewardScale;
    sac.criticTanh = g_cfg.criticTanh;
    sac.rewardTanhGain = g_cfg.rewardTanhGain;
    sac.alphaGumbelSigma = g_cfg.alphaGumbelSigma;
    sac.alphaCeiling = g_cfg.alphaCeiling;
    sac.entropyCenter = g_cfg.entropyCenter;
    /*
       `alphaCeiling` 必须在**构造时**就生效 (见 agent 构造函数的注释), 所以这里
       再夹一次: harness 是先构造、后设旋钮的, 构造那一刻用的还是默认上界 5.0。
       只在"上界确实比初值 0.2 小"时才改变——否则这一行是空操作。
    */
    if (g_cfg.alphaCeiling > 0.0f && sac.alpha[0] > g_cfg.alphaCeiling) {
        sac.alpha[0] = g_cfg.alphaCeiling;
    }
    sac.sparseLeafEval = g_cfg.sparseLeaf;
    sac.learnFromSearch = g_cfg.learnFromSearch;
    sac.entropyInTarget = g_cfg.entropyInTarget;
    sac.entropySlotsAsLegal = g_cfg.entropySlots;
    const Opponent *probeOpp = nullptr;
    (void)probeOpp;
    const double tBuild1 = nowMs();

    std::printf("构建      : %.1f s\n", (tBuild1 - tBuild0) / 1000.0);
    printConfig(sac);

    /* ---- 载入权重 (可选) ---- */
    if (!g_cfg.loadPrefix.empty()) {
        const bool ok = sac.loadModel(g_cfg.loadPrefix);
        std::printf("载入      : %s -> %s\n", g_cfg.loadPrefix.c_str(),
                    ok ? "成功" : "**失败**");
        if (!ok) {
            /* 显式要评测一个已训练模型却载不进来 —— 这是失败, 不能装作没看见 */
            std::printf("VERDICT: FAIL (权重载入失败: %s)\n", g_cfg.loadPrefix.c_str());
            return 1;
        }
    } else if (g_cfg.trainGames == 0) {
        std::printf("权重      : **随机初始化** —— 下面的得分率不是棋力结论, 只是链路与下界\n");
    }

    /* ============================================================
     *  阶段 1: 自对弈训练
     * ============================================================ */
    if (g_cfg.trainGames > 0) {
        std::printf("\n---- 训练: 自对弈 %d 局 (%d 模拟/步, 每局最多 %d 手, 每 %d 手更新一次)\n",
                    g_cfg.trainGames, g_cfg.trainSims, g_cfg.trainPlies, g_cfg.learnEvery);
        const double t0 = nowMs();
        sac.trainSelfPlay(g_cfg.trainGames, g_cfg.trainSims, g_cfg.trainPlies,
                          /*verbose=*/false, g_cfg.tempRoot, g_cfg.tempFinal,
                          g_cfg.learnEvery);
        const double t1 = nowMs();
        std::printf("      训练耗时 %.1f s  (%.2f s/局, %.1f ms/手)\n",
                    (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / (double)g_cfg.trainGames,
                    (t1 - t0) / (double)std::max(1, g_cfg.trainGames * g_cfg.trainPlies));
        printDiag(sac, "训练后");
        if (!g_cfg.savePrefix.empty()) {
            const bool ok = sac.saveModel(g_cfg.savePrefix);
            std::printf("      存盘 %s -> %s\n", g_cfg.savePrefix.c_str(),
                        ok ? "成功" : "**失败**");
        }
    }

    /* ============================================================
     *  阶段 2: 等时间标定 (可选)
     * ============================================================ */
    int simsForEval = g_cfg.evalSims;
    if (g_cfg.simsTimeMs > 0.0) {
        /*
           标定: 用**当前权重**在同一批开局上跑几步, 量 ms/模拟。
           注意这里刻意**先 warm up 再计时** —— 冷启动那一步要付第一遍触碰
           ~0.5 GB 权重页的代价, 把它算进去会把 ms/模拟 抬高一个数量级, 于是
           "等时间"模式会给出一个荒唐的小模拟数 (实测过: 48 ms/模拟 vs 稳态 4 ms)。
        */
        Chess wc;
        wc.reset();
        sac.selectMove(Stone::COLOR_RED, 2, 0.0f);   /* warm up */
        const int probeMoves = 4;
        const int probeSims = 8;
        double total = 0.0;
        int samples = 0;
        for (int i = 0; i < probeMoves; i++) {
            const double t0 = nowMs();
            const Step s = sac.selectMove(Stone::COLOR_RED, probeSims, 0.0f);
            total += nowMs() - t0;
            samples++;
            if (!s.valid) { break; }
            double dummy = 0.0;
            wc.moveForward(&s, dummy);
        }
        const double msPerSim = (samples > 0) ? total / (double)samples / (double)probeSims : 0.0;
        if (msPerSim > 0.0) {
            simsForEval = (int)(g_cfg.simsTimeMs / msPerSim);
            simsForEval = std::max(1, std::min(simsForEval, 4096));
            std::printf("\n等时间    : 标定 %.3f ms/模拟 (稳态, 已 warm up) -> 每步 %.0f ms = %d 次模拟\n",
                        msPerSim, g_cfg.simsTimeMs, simsForEval);
        } else {
            std::printf("\n等时间    : 标定失败, 退回固定 %d 次模拟\n", simsForEval);
        }
    }

    /* ============================================================
     *  阶段 3: 对锚点评测
     * ============================================================ */
    std::printf("\n---- 评测: 每个开局两局 (交换先手), 每步 %d 次模拟, 每局最多 %d 手\n",
                simsForEval, g_cfg.evalPlies);

    /*
       ---- 评测前先 warm up 一次 ----
       为什么必须: 这台机器上第一次 `selectMove` 要付"第一次触碰 ~0.5 GB 权重页"的代价,
       实测能让 ms/模拟 从稳态 4 ms 变成 48 ms (差一个数量级)。不 warm up 的话报告里的
       "SAC ms/步"量到的是冷启动, 而它会被当成稳态代价去和别的东西比。
       (等时间模式那边也单独 warm up 过一次 —— 这里补的是"不做等时间标定"的路径。)
    */
    {
        Chess wc;
        wc.reset();
        sac.selectMove(Stone::COLOR_RED, 2, 0.0f);
    }

    std::vector<EvalResult> results;
    int abDepthSeen = 0;
    (void)abDepthSeen;
    if (g_cfg.opponent == "ab" || g_cfg.opponent == "both") {
        std::printf("\n  [锚点 AB 深度 %d  (= 界面的 AGENT_AB_L1, 见 chessboard.cpp 的 AB_L1_DEPTH)]\n",
                    g_cfg.abDepth);
        AbOpponent ab(c, g_cfg.abDepth);
        results.push_back(evaluate(c, sac, ab, book, simsForEval));
    }
    if (g_cfg.opponent == "mcts" || g_cfg.opponent == "both") {
        std::printf("\n  [锚点 MCTS %d 次迭代/步, srand=%u]\n", g_cfg.mctsIters, g_cfg.mctsSrand);
        MctsOpponent mo(c, g_cfg.mctsIters);
        /*
           ================================================================
           **必须在构造之后重新播种 `std::rand`** —— 这是一个实测出来的坑
           ================================================================
           `MCTS` 的构造函数里有 `std::srand((unsigned int)std::time(nullptr))`
           (src/mcts.cpp:67, 那一条是为了"GUI 里反复构造 MCTS 时不要每次都播种",
           注释写在那个构造函数上方)。于是 main 开头那次 `std::srand(--mcts-srand)`
           **会被它整个覆盖掉**, MCTS 锚点就变成"每跑一次换一副牌"。

           实测 (修之前): 同一条命令行跑两次, `--opponent=ab` 的 12 局**逐手相同**
           (只差 ms 计时行), 而 `--opponent=mcts` 的 12 局有 3 局不同、总分从 50.0%
           变成别的值。也就是说**AB 锚点的比较是配对的, MCTS 锚点的不是** ——
           而"配对"正是这个工具存在的理由 (同一批开局、同一个对手随机流)。

           修法就是下面这一行: 构造完 MCTS 之后把随机流钉回 `--mcts-srand`。
           它与 `bench_sac_mcts_min` 的 `--mcts-srand` 是同一个口径。
        */
        std::srand(g_cfg.mctsSrand);
        results.push_back(evaluate(c, sac, mo, book, simsForEval));
    }

    /* ============================================================
     *  汇总
     * ============================================================ */
    int broken = 0;
    std::printf("\n=== 汇总 (SAC+AZ(TB) 视角; 胜=1 和=0.5 负=0) ===\n");
    std::printf("  %-6s %-6s %5s %4s %4s %4s %8s  %-18s %9s %9s\n",
                "锚点", "局数", "胜", "和", "负", "缺陷", "得分率", "95% 区间",
                "Elo差", "SAC ms/步");
    for (std::size_t i = 0; i < results.size(); i++) {
        const EvalResult &r = results[i];
        broken += r.broken;
        const char *verdict = (r.scoreLo > 0.5) ? "**强于锚点**"
                              : (r.scoreHi < 0.5) ? "弱于锚点"
                                                  : "区间跨 50%: 没测出差别";
        std::printf("  %-6s %-6d %5d %4d %4d %4d %7.1f%%  [%5.1f%%, %5.1f%%]  %+8.1f %9.1f  %s\n",
                    r.oppName.c_str(), r.games, r.wins, r.draws, r.losses, r.broken,
                    r.score * 100.0, r.scoreLo * 100.0, r.scoreHi * 100.0,
                    eloFromScore(r.score), r.sacMsPerMove, verdict);
        std::printf("         平均 %5.1f 手 | 材料差(SAC视角) 均 %+.2f (占优 %d 局 / 落后 %d 局)"
                    " | 和棋: 重复 %d, 60回合 %d, 台架截断 %d | 分胜负: SAC 赢 %d / 锚点赢 %d\n",
                    r.avgPlies, r.avgMaterialDiff, r.sacMaterialBetter, r.sacMaterialWorse,
                    r.drawRepetition, r.drawHalfmove,
                    r.drawByTruncation, r.sacMates, r.oppMates);
    }
    printDiag(sac, "评测后");

    if (!g_cfg.csvPath.empty()) {
        FILE *fp = std::fopen(g_cfg.csvPath.c_str(), "w");
        if (fp != nullptr) {
            std::fprintf(fp, "opponent,games,wins,draws,losses,broken,score,score_lo,score_hi,"
                             "elo,sac_ms_per_move,opp_ms_per_move,avg_plies,avg_material_diff,"
                             "mat_better,mat_worse,"
                             "train_games,train_sims,eval_sims,seed,book_fp,"
                             "target_tau,target_iter,entropy_ratio,alpha_lr,lr,batch,epochs,"
                             "az_weight,cpuct,clamp,huber,aux,memory,reward_shape,"
                             "value_scale,reward_scale,sparse_leaf,learn_from_search,trunk,tb_heads,"
                             "entropy_in_target,entropy_slots,"
                             "mate_score_mode,mat_boost,mate_boost,"
                             "critic_tanh,reward_tanh_gain,alpha_gumbel,alpha_ceiling,entropy_center\n");
            for (std::size_t i = 0; i < results.size(); i++) {
                const EvalResult &r = results[i];
                std::fprintf(fp,
                    "%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.2f,%.3f,%d,%d,"
                    "%d,%d,%d,%u,%016llx,"
                    "%.6g,%d,%.6g,%.6g,%.6g,%d,%d,"
                    "%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,"
                    /*
                       ---- [2026-09 修复] 这一段的**格式串原来比表头少三列**
                       (`reward_scale` / `entropy_in_target` / `entropy_slots` 只有名字没有
                       转换符), 后面五个 [实验轮] 旋钮则完全没有列 —— 于是 CSV 的**表头与
                       数据行、数据行与数据行之间全部错位**, 而且不报任何错。现在把
                       表头列名、转换符、实参**一一对齐** (顺序 = 表头顺序)。
                    */
                    "%.6g,%.6g,%d,%d,%s,%s,%.6g,%d,%d,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%d\n",
                    r.oppName.c_str(), r.games, r.wins, r.draws, r.losses, r.broken,
                    r.score, r.scoreLo, r.scoreHi, eloFromScore(r.score),
                    r.sacMsPerMove, r.oppMsPerMove, r.avgPlies, r.avgMaterialDiff,
                    r.sacMaterialBetter, r.sacMaterialWorse,
                    g_cfg.trainGames, g_cfg.trainSims, simsForEval, g_cfg.seed, bookFp,
                    (double)g_cfg.targetTau, g_cfg.targetIter, (double)g_cfg.entropyRatio,
                    (double)g_cfg.alphaLr, (double)g_cfg.lr, g_cfg.batch, g_cfg.epochs,
                    (double)g_cfg.azWeight, (double)g_cfg.cpuct, (double)g_cfg.clampTarget,
                    (double)g_cfg.huberDelta, (double)g_cfg.aux, g_cfg.memory,
                    g_cfg.rewardShape, (double)g_cfg.valueScale,
                    (double)g_cfg.rewardScale, (int)g_cfg.sparseLeaf, (int)g_cfg.learnFromSearch,
                    g_cfg.trunk.c_str(), g_cfg.tbHeads.c_str(),
                    (double)g_cfg.entropyInTarget, (int)g_cfg.entropySlots,
                    g_cfg.mateScoreMode, (double)g_cfg.matRewardBoost,
                    (double)g_cfg.mateRewardBoost,
                    (int)g_cfg.criticTanh, (double)g_cfg.rewardTanhGain,
                    (double)g_cfg.alphaGumbelSigma, (double)g_cfg.alphaCeiling,
                    (int)g_cfg.entropyCenter);
            }
            std::fclose(fp);
            std::printf("\nCSV       : %s\n", g_cfg.csvPath.c_str());
        } else {
            std::printf("\n[警告] CSV 写不进去: %s\n", g_cfg.csvPath.c_str());
        }
    }

    if (broken > 0) {
        std::printf("\nVERDICT: FAIL | 有 %d 局出现非法/无效走法 —— 这是机制性缺陷, 不是棋力\n",
                    broken);
        return 1;
    }
    std::printf("\nVERDICT: PASS | 机制成立 (0 非法走法); 得分率的解释见上面的 95%% 区间\n");
    return 0;
}
