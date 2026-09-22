/*
 * bench_sac_mcts_min_main.cpp - 最小版"SAC vs MCTS"受控对局 (跨版本可比)
 * ============================================================================
 *
 * 为什么单独写一个而不用 bench_agent_arena: 那个工具引用了本轮新增的接口
 * (`policySparse` / `entropyRatio` 等), 编译不到**改前**那棵树上去。要回答
 * "现在的 SAC 是不是比改前弱", 必须在两棵树上跑**同一份源码、同一套协议**。
 *
 * 所以这里只用两个版本都有的最小接口:
 *   SACAZAgent::selectMove(color, sims, temp) / getLegalActions / encodeState 无涉
 *   MCTS::findBestMove(color, sims)
 *   Chess::reset / sample / isLegalMove / moveForward / getResult / computeHash
 * 协议与 bench_agent_arena 一致: 交换先后手、同一随机开局、同一模拟预算、
 * 逐手合法性校验、得分率 + Wilson 95% 区间。
 *
 * 用法:
 *   bench_sac_mcts_min.exe [--games=30] [--sims=160] [--plies=120] [--opening=4]
 *                          [--seed=20240901] [--backbone=moe-mlp] [--load=prefix]
 *                          [--csv=path] [--verbose]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "mcts.h"
#include "sacazagent.h"
#include "rl/util.hpp"

namespace {

struct Cfg {
    int games = 30;
    int sims = 160;
    int maxPlies = 120;
    int openingPlies = 4;
    unsigned seed = 20240901;
    std::string backbone = "moe-mlp";
    std::string load;
    std::string csv;
    bool verbose = false;
    /* 消融旋钮 (与 bench_agent_arena 同名同义; 默认值 = 保守/与改动前一致) */
    float clamp = 0.0f;        /* <=0: 关掉 critic 值域约束 */
    float huber = 0.0f;        /* <=0: 用纯 MSE */
    bool sparseLeaf = false;   /* 搜索叶子是否走稀疏头 (默认关, 见 main 里的说明) */
    bool dumpRoot = false;     /* 只打印第一个局面的根分布然后退出 (定位用) */
    bool dumpForward = false;  /* 只打印第一个局面的先验/Q 然后退出 (定位用) */
    /*
       MCTS 对手的随机种子。为什么必须能固定 (2026-09 排查教训):
       MCTS 的展开与随机走子用的是 `std::rand()`, 而 `std::srand` 播的是
       `std::time(nullptr)` —— 于是**每次运行的对局随机性都不同**。比较两个 SAC 版本时,
       这就等于"每跑一次就换一副牌": 同一个版本的得分率在 40~75% 之间摆, 任何跨版本的
       "谁更高"都可能只是对手换了随机流。
       这里在两版**都**允许显式重播同一个种子, 于是对手行为可复现、跨版本可比。
       (`std::srand` 是全局的: 基线的 MCTS 在构造时播一次, 所以必须在**构造之后**重播。)
    */
    unsigned mctsSrand = 0;    /* 0 = 不干预 (保持原行为) */
    /*
       --dump-moves=<path>: 把**每一局每一手的走法**写成一行文本。
       为什么需要它 (2026-09): "当前代码是否与 59e5233 逐手等价"只能靠**重放同一局**
       来判定 —— 比分只能给统计等价, 给不了"同一局面下走同一步"。两个二进制各导一份,
       逐字节 diff 即可。
    */
    std::string dumpMoves;
};

Cfg g;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

static bool parseBackbone(const std::string &s, SACAZAgent::Backbone &out)
{
    if (s == "mlp")      { out = SACAZAgent::Backbone::Mlp;          return true; }
    if (s == "moe-mlp")  { out = SACAZAgent::Backbone::SparseMoeMlp; return true; }
    if (s == "tb")       { out = SACAZAgent::Backbone::SparseMoeTb;  return true; }
    if (s == "dense-tb") { out = SACAZAgent::Backbone::DenseMoeTb;   return true; }
    return false;
}

static void randomOpening(Chess &c, int &turn, int plies)
{
    for (int i = 0; i < plies; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) { return; }
        std::vector<Step*> legal;
        c.sample(turn, legal);
        if (legal.empty()) { Steps::instance().put(legal); return; }
        std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
        const Step chosen = *legal[(std::size_t)pick(RL::Random::engine)];
        Steps::instance().put(legal);
        double dummy = 0.0;
        c.moveForward(&chosen, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
}

struct Score {
    int wins = 0, losses = 0, draws = 0, broken = 0;
    int n() const { return wins + losses + draws; }
    double rate() const {
        const int m = n();
        return (m > 0) ? ((double)wins + 0.5 * (double)draws) / (double)m : 0.0;
    }
    void interval(double &lo, double &hi) const {
        const int m = n();
        if (m == 0) { lo = 0.0; hi = 1.0; return; }
        const double z = 1.959963984540054, p = rate();
        const double denom = 1.0 + z * z / (double)m;
        const double centre = (p + z * z / (2.0 * (double)m)) / denom;
        const double half = (z / denom) * std::sqrt(p * (1.0 - p) / (double)m
                                                    + z * z / (4.0 * (double)m * (double)m));
        lo = std::max(0.0, centre - half);
        hi = std::min(1.0, centre + half);
    }
    double eloDiff() const {
        const double s = rate();
        if (s <= 0.0) { return -800.0; }
        if (s >= 1.0) { return 800.0; }
        return -400.0 * std::log10(1.0 / s - 1.0);
    }
    bool conclusive() const {
        double lo = 0.0, hi = 1.0;
        interval(lo, hi);
        return (lo > 0.5) || (hi < 0.5);
    }
};

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *k) -> const char* {
            const std::size_t n = std::strlen(k);
            if (std::strncmp(a, k, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--games"))   { g.games = std::atoi(v); }
        else if (const char *v = val("--sims"))    { g.sims = std::atoi(v); }
        else if (const char *v = val("--plies"))   { g.maxPlies = std::atoi(v); }
        else if (const char *v = val("--opening")) { g.openingPlies = std::atoi(v); }
        else if (const char *v = val("--seed"))    { g.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--backbone")){ g.backbone = v; }
        else if (const char *v = val("--load"))    { g.load = v; }
        else if (const char *v = val("--csv"))     { g.csv = v; }
        else if (std::strcmp(a, "--verbose") == 0) { g.verbose = true; }
        else if (std::strcmp(a, "--sparse-leaf") == 0) { g.sparseLeaf = true; }
        else if (std::strcmp(a, "--dump-root") == 0) { g.dumpRoot = true; }
        else if (std::strcmp(a, "--dump-forward") == 0) { g.dumpForward = true; }
        else if (const char *v = val("--clamp")) { g.clamp = (float)std::atof(v); }
        else if (const char *v = val("--huber")) { g.huber = (float)std::atof(v); }
        else if (const char *v = val("--mcts-srand")) { g.mctsSrand = (unsigned)std::atoi(v); }
        else if (const char *v = val("--dump-moves")) { g.dumpMoves = v; }
        /*
           --legacy-net 已删除 (2026-09): 那个开关想用 `TanhNorm<Linear>` 且 r=1 复现
           59e5233 的隐层激活, 实测**不等价** (TanhNorm 的偏置加在 tanh 外面;
           max|ΔQ| = 8.9e-06)。59e5233 的那一层就是 `Layer<Tanh>`, 而当前 buildNet 用的
           也是它 —— 复现靠"同一行代码", 没有开关可给。见 src/sacazagent.h。
        */
        else {
            std::fprintf(stderr, "[warn] 未知参数: %s\n", a);
        }
    }
    if (g.games % 2 != 0) { g.games++; }
    RL::Random::setSeed(g.seed);

    SACAZAgent::Backbone bb = SACAZAgent::Backbone::SparseMoeMlp;
    if (!parseBackbone(g.backbone, bb)) {
        std::printf("[错误] 未知骨干 %s\n", g.backbone.c_str());
        return 1;
    }

    Chess board;
    board.reset();
    SACAZAgent sac(board, 64, 0.99f, 0.001f, 1.5f, bb, 64, 0.1f);
    MCTS mcts(board, 1.414);

    /*
       搜索叶子估值默认走**全量口径** (与改动前逐位相同)。
       稀疏头是"8100 动作下每次叶子要算 8100 列 Q"的优化 (216ms -> 75ms); 在 128 槽
       表示下它省不了多少, 却多引入一条与训练路径不同的代码 —— 实测
       (docs/sac_regression_2026_09.md) 它伴随约 20 个百分点的棋力差, 而这个差目前
       无法用"等价性"解释 (单局面上两种口径逐元素一致到 1e-8)。保守起见默认关。
    */
    sac.sparseLeafEval = g.sparseLeaf;
    sac.clampTarget = g.clamp;
    sac.huberDelta = g.huber;
    /*
       对手的随机流固定住 (构造之后重播, 因为 MCTS 构造时自己播了一次 time())。
       这样两版面对**完全相同的对手行为**, 得分率之差才只反映 SAC 的差别。
    */
    if (g.mctsSrand != 0u) {
        std::srand(g.mctsSrand);
    }

    /*
       权重指纹 (2026-09 排查用): 两个二进制若真的跑同一套代码, 同一颗种子下这些数
       必须逐位相同 —— 它把"代码差异"与"随机流差异"分开。
       **注意**: 只用"加权求和"当指纹是不够的 (浮点求和会饱和/抵消, 实测同一个和
       对应了两套不同的权重)。所以这里同时打印**前 8 个权重原值**。
    */
    {
        auto dumpLayer = [](const char *tag, RL::Net &net) {
            for (std::size_t i = 0; i < net.size(); i++) {
                RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer *>(net[i]);
                if (fc != nullptr) {
                    std::printf("[WF] %s layer%zu w[0..3]= %.9g %.9g %.9g %.9g  (size=%zu)\n",
                                tag, i,
                                (double)fc->w.val.data()[0], (double)fc->w.val.data()[1],
                                (double)fc->w.val.data()[2], (double)fc->w.val.data()[3],
                                fc->w.size());
                    return;
                }
            }
            std::printf("[WF] %s: 没有 iFcLayer\n", tag);
        };
        dumpLayer("actor", sac.actor);
        dumpLayer("q1", sac.q1);
        dumpLayer("q2", sac.q2);
    }
    if (!g.load.empty() && !sac.loadModel(g.load)) {
        std::printf("VERDICT: FAIL (权重载入失败: %s)\n", g.load.c_str());
        return 1;
    }

    std::printf("=== SAC(min) vs MCTS ===\n");
    std::printf("SAC     : state=%d action=%d 骨干=%s 参数(actor)=%lld%s\n",
                SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM,
                SACAZAgent::backboneName(bb), sac.actor.paramCount(),
                g.load.empty() ? " [随机权重]" : " [已载入权重]");
    std::printf("协议    : %d 局(交换先后手) 每步 %d 模拟 上限 %d 手 随机开局 %d 步 seed=%u\n\n",
                g.games, g.sims, g.maxPlies, g.openingPlies, g.seed);

    /*
       --dump-forward: 不跑搜索, 只把第一个局面的**策略先验与 Q** 打出来。
       2026-09 定位用: 两个二进制的根分布除 top1 外完全一致 (同样 40 个槽位、同样的
       数值), 但 top1 的槽位不同 —— 说明差异在"每个槽位拿到的先验/Q", 不在搜索结构。
       这一项把"前向"与"搜索"一刀切开。
    */
    if (g.dumpForward) {
        board.reset();
        int turn0 = Stone::COLOR_RED;
        randomOpening(board, turn0, g.openingPlies);
        const int turn = board.sideToMove;
        RL::Tensor st(SACAZAgent::STATE_DIM, 1);
        RL::Tensor mk(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor pv(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor q1(SACAZAgent::ACTION_DIM, 1);
        RL::Tensor q2(SACAZAgent::ACTION_DIM, 1);
        std::vector<Step*> lg;
        std::vector<int> li;
        sac.getLegalActions(turn, lg, li, mk);
        Steps::instance().put(lg);
        sac.encodeStateFor(turn, st);
        sac.policy(st, mk, pv);
        sac.qValues(st, q1, q2);
        std::printf("[dump-forward] 状态维=%d 动作维=%d 合法=%zu\n",
                    SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM, li.size());
        double ssum = 0.0, psum = 0.0, qsum = 0.0;
        for (std::size_t i = 0; i < st.size(); i++) {
            ssum += std::fabs((double)st[i]) * (double)((i % 89) + 1);
        }
        for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
            psum += (double)pv[a] * (double)(a + 1);
            qsum += (double)q1[a] * (double)(a + 1);
        }
        std::printf("[dump-forward] 状态指纹=%.6f  先验加权和=%.6f  Q1加权和=%.6f\n",
                    ssum, psum, qsum);
        for (std::size_t i = 0; i < li.size() && i < 12; i++) {
            const int a = li[i];
            std::printf("    leg#%zu slot=%d prior=%.8f q1=%.8f q2=%.8f\n",
                        i, a, (double)pv[a], (double)q1[a], (double)q2[a]);
        }
        return 0;
    }

    /*
       --dump-root: 第一手之前把**根节点的访问分布**打出来 (top-8)。
       为什么需要它 (2026-09 定位用): 两个二进制在同一初始权重下第 1 手就不同, 而
       `selectMove` / `getPUCT` / `visitDistribution` / 选点收尾都已逐行核对过、逐字相同
       —— 那么差异要么在"根分布"本身 (搜索内部), 要么在"从分布选点"那一步的并列处理。
       打印分布就能把这两者一刀切开。
    */
    if (g.dumpRoot) {
        board.reset();
        int turn0 = Stone::COLOR_RED;
        randomOpening(board, turn0, g.openingPlies);
        std::printf("[dump-root] 每步 %d 模拟, 走子方=%d, 历史手数=%zu\n",
                    g.sims, board.sideToMove, board.history.size());
        std::vector<Step*> before;
        board.sample(board.sideToMove, before);
        std::printf("[dump-root] 合法走法 %zu 个\n", before.size());
        for (std::size_t i = 0; i < before.size() && i < 8; i++) {
            const Step &s = *before[i];
            std::printf("    #%zu (%d,%d)->(%d,%d)\n", i, s.pos.x, s.pos.y,
                        s.nextPos.x, s.nextPos.y);
        }
        Steps::instance().put(before);

        RL::Tensor piDump(SACAZAgent::ACTION_DIM, 1);
        const Step chosen = sac.selectMove(board.sideToMove, g.sims, 0.0f, &piDump);
        std::printf("[dump-root] 选点 (%d,%d)->(%d,%d)\n",
                    chosen.pos.x, chosen.pos.y, chosen.nextPos.x, chosen.nextPos.y);
        double sum = 0.0;
        for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) { sum += (double)piDump[a]; }
        std::printf("[dump-root] 根分布: Σ=%f, 非零槽位=%d\n", sum, [&] {
            int c = 0;
            for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) { if (piDump[a] > 0.0f) { c++; } }
            return c;
        }());
        for (int rank = 0; rank < 8; rank++) {
            int best = -1;
            double bv = -1.0;
            for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
                if ((double)piDump[a] > bv) { bv = (double)piDump[a]; best = a; }
            }
            if (best < 0 || bv <= 0.0) { break; }
            std::printf("    top%d: action=%d  pi=%.6f\n", rank + 1, best, bv);
            piDump[best] = 0.0f;
        }
        return 0;
    }

    Score st;
    int pliesTotal = 0;
    double sacMs = 0.0, mctsMs = 0.0;
    int sacMoves = 0, mctsMoves = 0;
    const double t0 = nowMs();

    /* 走法序列导出 (--dump-moves): 两个二进制各导一份, 逐字节 diff 判"逐手等价" */
    FILE *mvDump = nullptr;
    if (!g.dumpMoves.empty()) {
        mvDump = std::fopen(g.dumpMoves.c_str(), "w");
        if (mvDump != nullptr) {
            std::fprintf(mvDump, "# state=%d action=%d sims=%d plies=%d opening=%d "
                                 "seed=%u mctsSrand=%u\n",
                         SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM, g.sims,
                         g.maxPlies, g.openingPlies, g.seed, g.mctsSrand);
        }
    }

    for (int i = 0; i < g.games; i++) {
        const bool sacIsRed = (i % 2 == 0);
        const int sacColor = sacIsRed ? Stone::COLOR_RED : Stone::COLOR_BLACK;
        board.reset();
        int turn = Stone::COLOR_RED;
        randomOpening(board, turn, g.openingPlies);

        int plies = 0;
        const char *reason = "";
        bool broken = false;
        int winner = Chess::RESULT_ONGOING;

        while (plies < g.maxPlies) {
            const int res = board.getResult(turn);
            if (res != Chess::RESULT_ONGOING) {
                winner = res;
                reason = (res == Chess::RESULT_DRAW) ? "repetition/60-move" : "mate";
                break;
            }
            const bool sacTurn = (turn == sacColor);
            const double tt = nowMs();
            const Step s = sacTurn ? sac.selectMove(turn, g.sims, 0.0f)
                                   : mcts.findBestMove(turn, g.sims);
            const double ms = nowMs() - tt;
            if (sacTurn) { sacMs += ms; sacMoves++; } else { mctsMs += ms; mctsMoves++; }

            std::vector<Step*> legal;
            board.sample(turn, legal);
            const bool anyLegal = !legal.empty();
            bool ok = false;
            if (anyLegal && s.valid) { ok = board.isLegalMove(turn, &s); }
            Steps::instance().put(legal);
            if (!s.valid || !ok) {
                broken = true;
                reason = sacTurn ? "SAC 走了非法/无效走法" : "MCTS 走了非法/无效走法";
                break;
            }
            if (g.verbose) {
                std::printf("      ply %3d %-5s (%d,%d)->(%d,%d) %7.1f ms [%s]\n",
                            plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                            s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                            sacTurn ? "SAC" : "MCTS");
            }
            /* 走法序列落盘 (逐手等价判定用): 一行 = 局号/手数/走子方/起止/谁走的 */
            if (mvDump != nullptr) {
                std::fprintf(mvDump, "g%d p%03d %s (%d,%d)->(%d,%d) %s\n",
                             i, plies + 1,
                             (turn == Stone::COLOR_RED) ? "RED  " : "BLACK",
                             s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y,
                             sacTurn ? "SAC" : "MCTS");
            }
            double dummy = 0.0;
            board.moveForward(&s, dummy);
            turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            plies++;
        }
        if (winner == Chess::RESULT_ONGOING && !broken) {
            winner = Chess::RESULT_DRAW;
            reason = "ply limit";
        }
        pliesTotal += plies;

        int score = 0;
        if (!broken) {
            if (winner == Chess::RESULT_DRAW) { score = 0; }
            else {
                const bool redWon = (winner == Chess::RESULT_RED_WIN);
                score = (redWon == sacIsRed) ? 1 : -1;
            }
        }
        if (broken)          { st.broken++; }
        else if (score > 0)  { st.wins++; }
        else if (score < 0)  { st.losses++; }
        else                 { st.draws++; }

        std::printf("  game %3d/%d  SAC=%-5s  %-8s %3d 手  [%s]  sac %6.1f ms/步, mcts %6.1f ms/步\n",
                    i + 1, g.games, sacIsRed ? "RED" : "BLACK",
                    broken ? "BROKEN" : (score > 0 ? "SAC wins" : (score < 0 ? "MCTS wins" : "draw")),
                    plies, reason,
                    sacMoves > 0 ? sacMs / sacMoves : 0.0,
                    mctsMoves > 0 ? mctsMs / mctsMoves : 0.0);
    }

    const double sec = (nowMs() - t0) / 1000.0;
    if (mvDump != nullptr) { std::fclose(mvDump); mvDump = nullptr; }
    double lo = 0.0, hi = 1.0;
    st.interval(lo, hi);
    std::printf("\n=== 结果 (SAC 视角) ===\n");
    std::printf("  比分      : SAC %d 胜 / MCTS %d 胜 / 和 %d%s\n",
                st.wins, st.losses, st.draws,
                st.broken > 0 ? " (+机制违规)" : "");
    std::printf("  得分率    : %.1f%%  95%% Wilson [%.1f%%, %.1f%%]  -> %s\n",
                100.0 * st.rate(), 100.0 * lo, 100.0 * hi,
                st.conclusive() ? "decisive" : "inconclusive");
    std::printf("  Elo 差    : %+.0f | 平均手数 %.1f | 总耗时 %.1f s\n",
                st.eloDiff(), st.n() > 0 ? (double)pliesTotal / st.n() : 0.0, sec);
    std::printf("  思考时间  : SAC %.1f ms/步, MCTS %.1f ms/步\n",
                sacMoves > 0 ? sacMs / sacMoves : 0.0,
                mctsMoves > 0 ? mctsMs / mctsMoves : 0.0);

    if (!g.csv.empty()) {
        FILE *f = std::fopen(g.csv.c_str(), "w");
        if (f != nullptr) {
            std::fprintf(f, "tool,games,sims,plies,opening,seed,backbone,state_dim,action_dim,"
                            "wins,losses,draws,broken,score_rate,lo,hi,elo\n");
            std::fprintf(f, "min,%d,%d,%d,%d,%u,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.2f\n",
                         st.n(), g.sims, g.maxPlies, g.openingPlies, g.seed,
                         g.backbone.c_str(), SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM,
                         st.wins, st.losses, st.draws, st.broken,
                         st.rate(), lo, hi, st.eloDiff());
            std::fclose(f);
        }
    }

    std::printf("VERDICT: %s | 比分=%d/%d/%d 得分率=%.1f%% [%.1f,%.1f] Elo%+.0f %s\n",
                st.broken > 0 ? "FAIL" : "PASS",
                st.wins, st.losses, st.draws, 100.0 * st.rate(),
                100.0 * lo, 100.0 * hi, st.eloDiff(),
                st.conclusive() ? "decisive" : "inconclusive");
    return st.broken > 0 ? 1 : 0;
}
