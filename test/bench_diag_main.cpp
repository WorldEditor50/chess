/*
 * ============================================================================
 *  bench_diag_main.cpp - PPO+MCTS+AlphaZero 的**诊断仪表盘** (不进 ctest)
 * ============================================================================
 *
 *  为什么要有这个程序: 棋力(Elo/胜率)是**滞后指标**。本工程的实测就是例子 ——
 *  4 局对 AB 的得分率三次都压在 0% 地板上 (0/4, 0/4, 0/4), 棋力协议什么也答不了。
 *  能当场回答"哪一层设计错了"的是中间量, 也就是这个程序打印的东西:
 *
 *    搜索(老师)  根访问熵 / top-1 份额 / 展开覆盖率 / KL(访问||先验)
 *                **吃子 vs 退让 的 Q 分组**  <- "胆小不吃子"只有这里能量化
 *    行为        每手材质变化 / 选到吃子的比例 / 叫将率 / 每手 V 增益
 *    战术        自动生成的"一步杀 / 白吃子"题库命中率 (有标准答案, 不是代理指标)
 *    生态        W/D/L 与先后手对称 / 和棋率 / 开局种类数
 *
 *  自动题库为什么可行 (不用手工摆局面): 随机走若干步得到一个局面后, **扫描所有合法
 *  着法**就能判定这个局面是不是"一步杀"题 (落子后 getResult 立刻是己方胜) 或
 *  "白吃子"题 (吃子后该子**不被对方攻击**, 即没有反吃)。标准答案是搜索自己找出来的,
 *  不依赖任何外部引擎 —— 本工程没有 FEN 接口, 手工摆局面不现实 (见 issues_review)。
 *
 *  用法:
 *    bench_diag.exe [--games=4] [--sims=80] [--plies=80] [--opening=4]
 *                   [--hidden=64] [--expert=64] [--grad=0]
 *                   [--load=PREFIX] [--seed=N]
 *                   [--tactics=40] [--tactic-sims=80]
 *                   [--dirichlet-ab=20] [--print-root=1] [--csv=PREFIX]
 *
 *  典型用法 (快速冒烟: 小网络, 几十秒):
 *    bench_diag.exe --games=2 --plies=40 --sims=40 --hidden=16 --expert=16 \
 *                   --tactics=20 --tactic-sims=40
 *
 *  典型用法 (带真实权重):
 *    bench_diag.exe --games=4 --sims=400 --tactics=60 --tactic-sims=400 \
 *                   --load=D:\home\lab\chess\weights\ppo_bc_d4 --csv=diag
 * ============================================================================
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

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/diag.h"
#include "rl/util.hpp"

namespace {

struct Cfg {
    int games = 4;
    int sims = 80;
    int maxPlies = 80;
    int openingPlies = 4;
    int hidden = 64;
    int expert = 64;
    bool withGrad = false;
    std::string loadPrefix;
    unsigned seed = 20240901u;
    int tactics = 0;
    int tacticSims = 80;
    int dirichletAb = 0;
    int printRoot = 0;
    std::string csvPrefix;
    bool verbose = false;
};

Cfg g_cfg;

int flipColor(int c)
{
    return (c == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
}

/* 一方在场上的材质总和 (单位: 子力值, 帅=1000 不计 —— 它不计入材质比较) */
double materialOf(Chess &c, int color)
{
    double m = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s == nullptr || !s->alive) { continue; }
        if (s->color != color) { continue; }
        if (s->type == Stone::TYPE_JIANG) { continue; }   /* 帅/将不参与材质 */
        m += s->value;
    }
    return m;
}

bool isCaptureStep(const Step &s)
{
    return s.nextId != Stone::ID_NONE;
}

bool sameMove(const Step &a, const Step &b)
{
    return a.pos.x == b.pos.x && a.pos.y == b.pos.y
           && a.nextPos.x == b.nextPos.x && a.nextPos.y == b.nextPos.y;
}

/* 把一个局面里"一方是否被将"读出来 (对手视角的 isInCheck) */
bool opponentInCheck(Chess &c, int mover)
{
    return c.isInCheck(flipColor(mover));
}

/* ---------------------------------------------------------------------------
 *  逐手行为诊断: 材质变化 / 是否吃子 / 是否叫将 / V 前后
 *
 *  **会落子** (调用方负责 moveBack), 因为"材质变化"和"叫将"必须在落子后才知道。
 *  这是刻意为之: 把"读 V"与"落子"放在同一个函数里, 避免调用方忘了对齐视角。
 * ------------------------------------------------------------------------- */
void measureMove(PPOMCTSAgent &ag, Chess &board, int mover, const Step &chosen,
                 RL::Diag::MoveBehavior &b)
{
    b = RL::Diag::MoveBehavior();
    const int opp = flipColor(mover);

    const double matBefore = materialOf(board, mover) - materialOf(board, opp);
    b.wasInCheck = board.isInCheck(mover) ? 1 : 0;
    b.isCapture = isCaptureStep(chosen) ? 1 : 0;

    /* V(走之前): 必须按走子方视角编码 (encodeState 用 board.sideToMove) */
    RL::Tensor st(PPOMCTSAgent::STATE_DIM, 1);
    const int savedSide = board.sideToMove;
    board.sideToMove = mover;
    ag.encodeState(st);
    b.vBefore = (double)ag.ppo.value(st);

    /* 落子 (moveForward 会把 sideToMove 翻成对手) */
    double dummy = 0.0;
    board.moveForward(&chosen, dummy);

    const double matAfter = materialOf(board, mover) - materialOf(board, opp);
    b.materialDelta = matAfter - matBefore;
    b.gaveCheck = opponentInCheck(board, mover) ? 1 : 0;

    /*
       V(走之后): 现在轮到**对手**, 所以 ppo.value 给的是对手视角的价值,
       取负号才是走子方视角 —— 与 commitEpisode 的 finalOutcome 口径一致。
       漏掉这个负号,"每手 V 增益"就会整体反号 (那正是本工程抓到过的那类 bug)。
    */
    ag.encodeState(st);
    b.vAfter = -(double)ag.ppo.value(st);

    (void)savedSide;
    /* board.sideToMove 现在是对手, 调用方接着 moveBack 就会恢复成 mover */
}

/* ---------------------------------------------------------------------------
 *  自动战术题库
 *
 *  两类题, 都**不需要手工摆局面**:
 *    mate-in-1 : 存在一步着法, 落子后 getResult 判己方胜 (将杀/困毙)
 *    free-capture: 存在一个吃子着法, 落子后该子**不被对方攻击** (没有反吃)
 *
 *  "白吃子"用的是几何近似 (isAttacked), 不看反吃着法是否合法 —— 因此它对"有根子"
 *  这类复杂情形会漏判, 但作为"敢不敢吃"的行为指标已经足够, 且完全可复现。
 * ------------------------------------------------------------------------- */
struct Puzzle {
    Chess board;
    int mover = Stone::COLOR_RED;
    std::vector<Step> solutions;
    int kind = 0;                 /* 0 = 一步杀, 1 = 白吃子 */
};

void findSolutions(Chess &c, int mover, int kind, std::vector<Step> &out)
{
    out.clear();
    std::vector<Step *> steps;
    c.sample(mover, steps);
    const int opp = flipColor(mover);
    for (std::size_t i = 0; i < steps.size(); i++) {
        const Step s = *steps[i];
        if (kind == 1 && !isCaptureStep(s)) {
            continue;                                   /* 白吃子题只看吃子着法 */
        }
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        bool good = false;
        if (kind == 0) {
            const int res = c.getResult(c.sideToMove);
            const int want = (mover == Stone::COLOR_RED) ? Chess::RESULT_RED_WIN
                                                        : Chess::RESULT_BLACK_WIN;
            good = (res == want);
        } else {
            good = !c.isAttacked(s.nextPos, opp);        /* 没有被反吃 */
        }
        c.moveBack(&s, dummy);
        if (good) {
            out.push_back(s);
        }
    }
    Steps::instance().put(steps);
}

/* 随机走几步造一个局面 (与 bench_policy_agreement 的 makePosition 同一思路) */
void randomOpening(Chess &c, int plies, std::mt19937 &rng)
{
    for (int i = 0; i < plies; i++) {
        std::vector<Step *> steps;
        c.sample(c.sideToMove, steps);
        if (steps.empty()) {
            Steps::instance().put(steps);
            return;
        }
        std::uniform_int_distribution<int> pick(0, (int)steps.size() - 1);
        const Step s = *steps[pick(rng)];
        Steps::instance().put(steps);
        double dummy = 0.0;
        c.moveForward(&s, dummy);
    }
}

/* ---------------------------------------------------------------------------
 *  参数
 * ------------------------------------------------------------------------- */
bool parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const char *eq = std::strchr(argv[i], '=');
        std::string k = a;
        std::string v;
        if (eq != nullptr) {
            k = a.substr(0, (std::size_t)(eq - argv[i]));
            v = std::string(eq + 1);
        }
        if (k == "--games")        { g_cfg.games = std::atoi(v.c_str()); }
        else if (k == "--sims")    { g_cfg.sims = std::atoi(v.c_str()); }
        else if (k == "--plies")   { g_cfg.maxPlies = std::atoi(v.c_str()); }
        else if (k == "--opening") { g_cfg.openingPlies = std::atoi(v.c_str()); }
        else if (k == "--hidden")  { g_cfg.hidden = std::atoi(v.c_str()); }
        else if (k == "--expert")  { g_cfg.expert = std::atoi(v.c_str()); }
        else if (k == "--grad")    { g_cfg.withGrad = (std::atoi(v.c_str()) != 0); }
        else if (k == "--load")    { g_cfg.loadPrefix = v; }
        else if (k == "--seed")    { g_cfg.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--tactics") { g_cfg.tactics = std::atoi(v.c_str()); }
        else if (k == "--tactic-sims") { g_cfg.tacticSims = std::atoi(v.c_str()); }
        else if (k == "--dirichlet-ab") { g_cfg.dirichletAb = std::atoi(v.c_str()); }
        else if (k == "--print-root")   { g_cfg.printRoot = std::atoi(v.c_str()); }
        else if (k == "--csv")     { g_cfg.csvPrefix = v; }
        else if (k == "--verbose") { g_cfg.verbose = true; }
        else {
            std::printf("未知参数: %s\n", argv[i]);
            return false;
        }
    }
    if (g_cfg.games < 1) { g_cfg.games = 1; }
    if (g_cfg.sims < 1) { g_cfg.sims = 1; }
    if (g_cfg.maxPlies < 4) { g_cfg.maxPlies = 4; }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!parseArgs(argc, argv)) {
        return 2;
    }
    RL::Random::setSeed(g_cfg.seed);

    std::printf("========================================\n");
    std::printf("  PPO+MCTS+AZ 诊断仪表盘\n");
    std::printf("========================================\n");
    std::printf("  网络   : hidden=%d expert=%d grad=%d\n",
                g_cfg.hidden, g_cfg.expert, g_cfg.withGrad ? 1 : 0);
    std::printf("  搜索   : %d 次模拟/步\n", g_cfg.sims);
    std::printf("  自对弈 : %d 局, 每局最多 %d 手, 随机开局 %d 步\n",
                g_cfg.games, g_cfg.maxPlies, g_cfg.openingPlies);
    std::printf("  战术题 : %d 道 (%d 次模拟), Dirichlet A/B: %d 个局面\n",
                g_cfg.tactics, g_cfg.tacticSims, g_cfg.dirichletAb);
    std::printf("  seed   : %u\n", g_cfg.seed);

    Chess board;
    PPOMCTSAgent ag(board, g_cfg.hidden, 0.99f, 0.001f, 1.414f,
                    g_cfg.expert, 0.1f, g_cfg.withGrad);

    if (!g_cfg.loadPrefix.empty()) {
        /*
           注意: loadModel 现在返回**真实结果** (见 ppomcts_agent.cpp 的修正)。
           载入失败必须当成失败处理 —— 否则整份诊断都是在随机权重上做的, 数字全是假的。
        */
        if (!ag.loadModel(g_cfg.loadPrefix)) {
            std::printf("\n  **权重载入失败**: %s\n", g_cfg.loadPrefix.c_str());
            std::printf("  (架构不匹配或文件损坏) —— 拒绝在随机权重上产出诊断数字。\n");
            return 3;
        }
        std::printf("  权重   : %s -> 载入成功\n", g_cfg.loadPrefix.c_str());
    } else {
        std::printf("  权重   : 未指定 (随机初始化 —— 结构性指标有效, 棋力类指标无意义)\n");
    }

    RL::Diag::Aggregates agg;
    RL::Diag::CsvWriter moveCsv;
    RL::Diag::CsvWriter tbCsv;
    if (!g_cfg.csvPrefix.empty()) {
        const std::string mp = g_cfg.csvPrefix + "_moves.csv";
        const std::string tp = g_cfg.csvPrefix + "_tb.csv";
        if (moveCsv.open(mp, RL::Diag::moveHeader())) {
            std::printf("  逐手 CSV: %s\n", mp.c_str());
        }
        if (tbCsv.open(tp, "tag,step,value")) {
            std::printf("  TB   CSV: %s\n", tp.c_str());
        }
    }

    /* =====================================================================
     *  [1] 自对弈 + 逐手诊断
     * ===================================================================== */
    long long tbStep = 0;
    /*
       ---- V 到底能不能预测胜负 (value 校准) ----
       自对弈里每一手记下 (走子方视角的 V, 走子方), 局末再按最终结果给这个走子方
       打标签 z (+1 胜 / 0 和 / -1 负)。于是可以算:
         EV = 1 - Var(z - V)/Var(z)   —— <0 表示 V 还不如"永远预测均值"
         校准误差                       —— 预测 +0.4 的局面实际胜率是不是也 ~40%
       这是"该不该回去修 value"的最终判据: 仪表盘里 Q(吃子) < Q(退让) 时, 如果同时
       EV < 0, 那问题就在叶子估值本身, 而不是在搜索/噪声/温度。
    */
    std::vector<double> vAll;
    std::vector<double> zAll;
    {
        std::printf("\n---- [1] 自对弈 %d 局 (逐手采集) ----\n", g_cfg.games);
        /*
           每局先随机走 openingPlies 手 —— 否则 temp=0 (确定性 argmax) 会让每一局
           走成**完全相同**的一盘棋, "开局多样性"那一栏就永远报 1 种 (第一版就是
           这么写的, 结果量出的是自己的缺陷而不是模型的问题)。这与各 bench 的
           randomOpening 是同一套做法。
        */
        std::mt19937 gRng(g_cfg.seed ^ 0x27d4eb2fu);
        for (int g = 0; g < g_cfg.games; g++) {
            board.reset();
            ag.resetSearchTree();              /* 每局清树 + 根噪声手数归零 */
            randomOpening(board, g_cfg.openingPlies, gRng);
            int turn = board.sideToMove;
            int gameResult = Chess::RESULT_ONGOING;
            int plies = 0;
            unsigned long long openingHash = 0;
            int capRed = 0, capBlack = 0;
            /* 本局的 (V, 走子方) 序列, 局末按结果打标签 */
            std::vector<double> gameV;
            std::vector<int> gameMover;

            for (int ply = 0; ply < g_cfg.maxPlies; ply++) {
                board.sideToMove = turn;       /* 与 selectMove 的约定对齐 */
                const Step mv = ag.selectMove(turn, g_cfg.sims, 0.0f);
                if (!mv.valid) {
                    /* 无合法走法: 当前走子方输 */
                    gameResult = (turn == Stone::COLOR_RED) ? Chess::RESULT_BLACK_WIN
                                                            : Chess::RESULT_RED_WIN;
                    plies = ply;
                    break;
                }

                /* (a) 搜索诊断: 树的根还停在**这一手之前**的局面 */
                RL::Diag::RootDiag rd;
                const bool haveRoot = ag.rootDiag(rd);
                agg.addRoot(rd);

                /* (b) 行为诊断 (内部会落子) */
                RL::Diag::MoveBehavior b;
                measureMove(ag, board, turn, mv, b);
                agg.addBehavior(b);
                gameV.push_back(b.vBefore);
                gameMover.push_back(turn);
                if (isCaptureStep(mv)) {
                    if (turn == Stone::COLOR_RED) { capRed++; } else { capBlack++; }
                }
                if (ply + 1 == 8) {
                    openingHash = board.computeHash();   /* 开局多样性: 第 8 手局面 */
                }

                const int res = board.getResult(board.sideToMove);
                plies = ply + 1;
                if (res != Chess::RESULT_ONGOING) {
                    gameResult = res;
                    if (g_cfg.verbose) {
                        std::printf("    局 %d: %s, %d 手\n", g + 1,
                                    (res == Chess::RESULT_RED_WIN) ? "红胜"
                                  : (res == Chess::RESULT_BLACK_WIN) ? "黑胜" : "和",
                                    plies);
                    }
                    break;
                }

                if (!g_cfg.csvPrefix.empty()) {
                    RL::Diag::moveRow(moveCsv, g + 1, ply + 1, turn, rd, b);
                }
                if (tbCsv.isOpen()) {
                    RL::Diag::scalarRow(tbCsv, "root_top_share", tbStep, rd.topShare);
                    RL::Diag::scalarRow(tbCsv, "root_visit_entropy", tbStep, rd.visitEntropy);
                    RL::Diag::scalarRow(tbCsv, "root_prior_kl", tbStep, rd.priorKl);
                    RL::Diag::scalarRow(tbCsv, "root_q_capture_minus_quiet", tbStep,
                                        rd.qCaptureMax - rd.qQuietMax);
                    RL::Diag::scalarRow(tbCsv, "move_material_delta", tbStep, b.materialDelta);
                    RL::Diag::scalarRow(tbCsv, "move_value_gain", tbStep, b.vAfter - b.vBefore);
                    RL::Diag::scalarRow(tbCsv, "is_capture", tbStep, (double)b.isCapture);
                    tbStep++;
                }
                (void)haveRoot;
                turn = flipColor(turn);
            }

            if (gameResult == Chess::RESULT_ONGOING) {
                gameResult = Chess::RESULT_DRAW;   /* 走到手数上限 */
            }
            RL::Diag::GameStat gs;
            gs.gameIndex = g + 1;
            gs.result = gameResult;
            gs.plies = plies;
            gs.openingHash = openingHash;
            gs.capturesByRed = capRed;
            gs.capturesByBlack = capBlack;
            agg.addGame(gs);

            /* ---- 局末打标签: 每个样本的 z 按**它自己那个走子方**的胜负给 ---- */
            for (std::size_t i = 0; i < gameV.size(); i++) {
                double z = 0.0;
                if (gameResult == Chess::RESULT_DRAW) {
                    z = 0.0;
                } else if (gameResult == Chess::RESULT_RED_WIN) {
                    z = (gameMover[i] == Stone::COLOR_RED) ? 1.0 : -1.0;
                } else if (gameResult == Chess::RESULT_BLACK_WIN) {
                    z = (gameMover[i] == Stone::COLOR_BLACK) ? 1.0 : -1.0;
                }
                vAll.push_back(gameV[i]);
                zAll.push_back(z);
            }
        }
    }

    /* =====================================================================
     *  [1b] V 能不能预测胜负 (value 校准) —— "该不该回去修 value"的最终判据
     * ===================================================================== */
    if (!vAll.empty()) {
        const double ev = RL::Diag::explainedVariance(zAll, vAll);
        const double zVar = RL::Diag::variance(zAll);
        const std::vector<RL::Diag::CalibBucket> bk =
            RL::Diag::calibration(vAll, zAll, 10);
        const double ce = RL::Diag::calibrationError(bk);
        std::printf("\n---- [1b] value 校准 (%d 个样本) ----\n", (int)vAll.size());
        /*
           **必须先查 z 有没有方差**。全和棋 (z ≡ 0) 时 Var(z) = 0, EV 按定义无解,
           本函数会返回 0 —— 那个 0 看起来像"V 和常数预测一样烂", 实际是"这一轮
           没有任何胜负可供校准"。不加这条判据, 读数一定会被误读。
           (自对弈弱策略 + 手数上限的组合经常就是全和棋, 本工程实测过 100% 和棋。)
        */
        if (zVar <= 1e-9) {
            std::printf("  ** z 无方差 (全部和棋/同一结果) -> EV 与校准均无意义 **\n");
            std::printf("  这不是 value 的结论, 而是自对弈生态的结论: 和棋率 100%%\n");
            std::printf("  (见下面'自对弈生态'一节; 弱策略 + 手数上限很容易走到这一步)\n");
        } else {
            std::printf("  EV = %.4f  %s\n", ev,
                        (ev < 0.0) ? "<- <0: V 还不如'永远预测均值', 先修 value" : "");
            std::printf("  校准误差 = %.4f (0 = 完美)\n", ce);
            std::printf("  分桶 (预测区间 -> 实际胜率, 样本数):\n");
            for (std::size_t i = 0; i < bk.size(); i++) {
                if (bk[i].n <= 0) { continue; }
                std::printf("    [%+.2f,%+.2f) 预测 %+.3f -> 实际 %+.3f  (n=%d)\n",
                            bk[i].vLo, bk[i].vHi, bk[i].predMean, bk[i].actualMean, bk[i].n);
            }
            std::printf("  判读: 预测与实际的列应当**同向**。若高预测桶实际胜率接近 0,\n");
            std::printf("        说明 V 把'看着好'当成了'会赢' —— 怕将/怕破仕那类偏差。\n");
        }
        if (tbCsv.isOpen()) {
            RL::Diag::scalarRow(tbCsv, "value_explained_variance", tbStep, ev);
            RL::Diag::scalarRow(tbCsv, "value_calibration_error", tbStep, ce);
            RL::Diag::scalarRow(tbCsv, "value_z_variance", tbStep, zVar);
        }
    } else {
        std::printf("\n---- [1b] value 校准: 无样本 (自对弈一局都没走完) ----\n");
    }

    /* =====================================================================
     *  [2] 自动战术题库 (有标准答案的硬指标)
     * ===================================================================== */
    if (g_cfg.tactics > 0) {
        std::printf("\n---- [2] 自动战术题库 (%d 道, %d 次模拟) ----\n",
                    g_cfg.tactics, g_cfg.tacticSims);
        std::mt19937 rng(g_cfg.seed ^ 0x5bd1e995u);
        int made = 0, mateMade = 0, capMade = 0;
        int mateHit = 0, mateTot = 0, capHit = 0, capTot = 0;
        int guard = 0;
        /*
           两类题**各自独立配额**: 第一版是"优先一步杀, 找不到就退而求其次出白吃子题",
           结果是随机开局里一步杀极罕见, 20 道题全变成了白吃子, 一步杀那行永远是 0/0。
           改成对每个候选局面**两类都试**, 各自计题 —— 这样两类指标都能有样本。
       */
        while ((mateMade < g_cfg.tactics || capMade < g_cfg.tactics)
               && guard < g_cfg.tactics * 400) {
            guard++;
            Chess pos;
            pos.reset();
            const int open = g_cfg.openingPlies + (int)(rng() % 9u);   /* 4..12 手随机开局 */
            randomOpening(pos, open, rng);
            const int mover = pos.sideToMove;

            for (int kind = 0; kind < 2; kind++) {
                if (kind == 0 && mateMade >= g_cfg.tactics) { continue; }
                if (kind == 1 && capMade  >= g_cfg.tactics) { continue; }

                std::vector<Step> sol;
                findSolutions(pos, mover, kind, sol);
                if (sol.empty()) {
                    continue;
                }
                if (kind == 0) { mateMade++; } else { capMade++; }
                made++;

                /*
                   **必须把题目局面拷回 agent 绑定的那个棋盘**再搜索。
                   踩过的坑: `PPOMCTSAgent ag(board, ...)` 在构造时就把 Chess& **绑死**,
                   另建一个 `Chess probe = pos` 再调 ag.selectMove 只会搜索 **board**
                   上当时的局面 (自对弈结束时的残局), 于是"战术准确率"量的是完全无关的
                   东西 —— 而且它会安静地给出 0%, 看起来像"模型很差"。
                */
                board = pos;
                board.sideToMove = mover;
                ag.resetSearchTree();
                const Step chosen = ag.selectMove(mover, g_cfg.tacticSims, 0.0f);
                bool hit = false;
                for (std::size_t i = 0; i < sol.size(); i++) {
                    if (sameMove(chosen, sol[i])) { hit = true; break; }
                }
                if (kind == 0) { mateTot++; if (hit) { mateHit++; } }
                else           { capTot++;  if (hit) { capHit++; } }

                if (g_cfg.verbose) {
                    std::printf("    题 %d (%s): %d 个解, 选中 %s\n",
                                made, (kind == 0) ? "一步杀" : "白吃子", (int)sol.size(),
                                hit ? "正确" : "**错**");
                }
            }
        }
        std::printf("  生成: 一步杀 %d 道, 白吃子 %d 道 (共 %d)\n", mateMade, capMade, made);
        const double mateAcc = (mateTot > 0) ? 100.0 * mateHit / mateTot : 0.0;
        const double capAcc = (capTot > 0) ? 100.0 * capHit / capTot : 0.0;
        const int tot = mateTot + capTot;
        const double acc = (tot > 0) ? 100.0 * (mateHit + capHit) / tot : 0.0;
        std::printf("  一步杀命中: %d/%d = %.1f%%\n", mateHit, mateTot, mateAcc);
        std::printf("  白吃子命中: %d/%d = %.1f%%  %s\n", capHit, capTot, capAcc,
                    (capTot > 0 && capAcc < 60.0) ? "<- <60%: 不敢吃子/看不出一手得子" : "");
        std::printf("  战术总准确率: %.1f%%\n", acc);
        if (tbCsv.isOpen()) {
            RL::Diag::scalarRow(tbCsv, "tactic_accuracy", tbStep, acc);
            RL::Diag::scalarRow(tbCsv, "mate_in_1_accuracy", tbStep, mateAcc);
            RL::Diag::scalarRow(tbCsv, "free_capture_accuracy", tbStep, capAcc);
        }
    }

    /* =====================================================================
     *  [3] Dirichlet 有效性 A/B: 同一局面开/关根噪声, 比较吃子着的访问
     * ===================================================================== */
    if (g_cfg.dirichletAb > 0) {
        std::printf("\n---- [3] Dirichlet 有效性 (%d 个局面) ----\n", g_cfg.dirichletAb);
        std::mt19937 rng(g_cfg.seed ^ 0x9e3779b9u);
        double shareOff = 0.0, shareOn = 0.0;
        int nOff = 0, nOn = 0;
        for (int i = 0; i < g_cfg.dirichletAb; i++) {
            Chess pos;
            pos.reset();
            randomOpening(pos, g_cfg.openingPlies + (int)(rng() % 5u), rng);
            const int mover = pos.sideToMove;

            /* 关噪声 */
            board = pos;
            board.sideToMove = mover;
            ag.evalRootNoise = false;
            ag.resetSearchTree();
            ag.selectMove(mover, g_cfg.sims, 0.0f);
            RL::Diag::RootDiag d1;
            if (ag.rootDiag(d1)) { shareOff += d1.captureVisitShare; nOff++; }

            /* 开噪声 (重新播种, 保证两次搜索的随机数起点相同, 差异只来自噪声) */
            board = pos;
            board.sideToMove = mover;
            ag.evalRootNoise = true;
            ag.resetSearchTree();
            RL::Random::setSeed(g_cfg.seed + 1000u + (unsigned)i);
            ag.selectMove(mover, g_cfg.sims, 0.0f);
            RL::Diag::RootDiag d2;
            if (ag.rootDiag(d2)) { shareOn += d2.captureVisitShare; nOn++; }
            ag.evalRootNoise = false;
        }
        const double mOff = (nOff > 0) ? shareOff / nOff : 0.0;
        const double mOn = (nOn > 0) ? shareOn / nOn : 0.0;
        std::printf("  吃子着访问份额: 关噪声 %.4f  vs  开噪声 %.4f  (差 %+.4f)\n",
                    mOff, mOn, mOn - mOff);
        std::printf("  判读: 开了噪声吃子访问份额明显上升 = 噪声在把低先验着法拉进搜索;\n");
        std::printf("        几乎不变 = 叶子 Q 已把吃子压死, 该回去修 value 而不是调噪声。\n");
        if (tbCsv.isOpen()) {
            RL::Diag::scalarRow(tbCsv, "dirichlet_capture_share_off", tbStep, mOff);
            RL::Diag::scalarRow(tbCsv, "dirichlet_capture_share_on", tbStep, mOn);
        }
    }

    /* =====================================================================
     *  [4] 调试钩子: 打印排序后的根 (P/Q/N)
     * ===================================================================== */
    if (g_cfg.printRoot > 0) {
        std::printf("\n---- [4] 根节点排序打印 (%d) ----\n", g_cfg.printRoot);
        std::mt19937 rng(g_cfg.seed ^ 0x85ebca6bu);
        for (int i = 0; i < g_cfg.printRoot; i++) {
            Chess pos;
            pos.reset();
            randomOpening(pos, g_cfg.openingPlies + (int)(rng() % 5u), rng);
            const int mover = pos.sideToMove;
            board = pos;                      /* agent 绑定的棋盘 (见 [2] 里的说明) */
            board.sideToMove = mover;
            ag.resetSearchTree();
            ag.selectMove(mover, g_cfg.sims, 0.0f);
            std::printf("\n  局面 %d (走棋方 %s):\n", i + 1,
                        (mover == Stone::COLOR_RED) ? "RED" : "BLACK");
            ag.printSortedRoot(10);
        }
    }

    /* =====================================================================
     *  汇总仪表盘 + CSV
     * ===================================================================== */
    agg.print("全部");

    if (tbCsv.isOpen()) {
        RL::Diag::scalarRow(tbCsv, "avg_root_top_share", tbStep, agg.meanTopShare());
        RL::Diag::scalarRow(tbCsv, "avg_root_visit_entropy", tbStep, agg.meanVisitEntropy());
        RL::Diag::scalarRow(tbCsv, "avg_root_prior_kl", tbStep, agg.meanPriorKl());
        RL::Diag::scalarRow(tbCsv, "avg_q_capture_minus_quiet", tbStep,
                            agg.meanQCaptureMinusQuiet());
        RL::Diag::scalarRow(tbCsv, "capture_accept_rate", tbStep, agg.captureChosenRate());
        RL::Diag::scalarRow(tbCsv, "mean_material_delta", tbStep, agg.meanMaterialDelta());
        RL::Diag::scalarRow(tbCsv, "draw_rate", tbStep, agg.drawRate());
        RL::Diag::scalarRow(tbCsv, "distinct_openings", tbStep, (double)agg.distinctOpenings());
        std::printf("  TB CSV 已写入汇总标量 (共 %lld 行)\n", tbStep);
    }

    std::printf("完成。\n");
    return 0;
}
