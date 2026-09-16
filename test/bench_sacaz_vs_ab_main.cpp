/*
 * bench_sacaz_vs_ab_main.cpp - 带 **TB 专家** 的 SAC agent 对 ABAgent 的静默对弈验证
 * ============================================================================
 *
 * 这是 GUI 里 "SAC+AZ-MoE" 那个 agent (`ChessBoard::AGENT_SACAZ_MOE`, 骨干 =
 * `SACAZAgent::Backbone::SparseMoeTb`, 16 次模拟/步, AB 深度 4) 的无界面对应物:
 * 不创建窗口、不弹对话框、不写权重文件 (除非显式 --save=)。
 *
 * 它回答两个不同的问题, 别混在一起:
 *
 *   1. **机制对不对 (这是它可以断言的部分)**
 *      * 每一手都必须**合法** (`Chess::isLegalMove` 逐手校验, 不是"看起来能跑");
 *      * 任何一方都不许返回无效走法 (返回了就是缺陷, 显式记成 ABORTED 并让退出码
 *        非 0 —— 不许悄悄当和棋);
 *      * 每局必须在手数上限内结束 (将被杀/困毙/重复/限着), 不许卡住;
 *      * 对局之后策略与 Q 必须全部是有限值 (TB 专家的 LN/注意力一旦写错, 这里会出
 *        NaN, 而且往往先表现为"走法变得毫无道理");
 *      * 稀疏 MoE 的门控真的被前向过, 并报告**专家使用分布** (top-1 路由最容易坍缩)。
 *
 *   2. **棋力 (它**不能**断言, 只能量)**
 *      默认不载入任何权重, 所以 SAC+AZ 是**随机初始化**的网络。这时比分既不代表
 *      这个骨干的上限也不代表下限, 只代表"从随机权重出发、和固定深度的 alpha-beta
 *      下几盘会怎样"。要谈棋力必须先有训练量, 而那件事 (自对弈几千局量级 +
 *      预训=0 的对照) 到现在还没做 —— 见 docs/issues_review.md 五、P2 第 9 条。
 *      想验"训练之后的它": --load=<prefix> 或 --warmup-games=K。
 *
 * 公平性 (缺一条比较就没有意义, 与 bench_ppo_vs_ab 同一套):
 *   1. **交换先后手**: 局数强制偶数 (中国象棋先手优势大, 固定谁执红等于在测"谁执红");
 *   2. **随机开局**: 每局先随机走 openingPlies 步合法棋, 否则同权重 + temp=0 会把每局
 *      下成同一盘棋, 比分只有 0/1 两个取值;
 *   3. **等时间 (可选 --budget=MS)**: 先标定 SAC+AZ 的 ms/模拟, 再按预算反推每步模拟
 *      次数。TB 专家的单价是 MLP 骨干的两个数量级, "固定 16 次模拟 vs AB 深度 4"
 *      只在后者恰好是同一时间预算时才公平 —— 报告里会把双方实测 ms/步打出来。
 *
 * 退出码 (静默验证靠它):
 *   0 = 机制全部成立;  1 = 有非法走法 / 无效走法 / 未结束的对局 / NaN。
 *   **比分不影响退出码** —— 随机权重输棋是预期内的, 那不是缺陷。
 *
 * 用法:
 *   bench_sacaz_vs_ab.exe [--games=6] [--plies=100] [--sims=16] [--depth=4]
 *                         [--opening=4] [--budget=0] [--min-sims=2] [--max-sims=64]
 *                         [--backbone=tb|dense-tb|moe-mlp|mlp]
 *                         [--warmup-games=0] [--warmup-sims=16]
 *                         [--load=PREFIX] [--save=PREFIX] [--seed=N]
 *                         [--verbose] [--quiet]
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "abagent.h"
#include "sacazagent.h"
#include "rl/util.hpp"

/*
   刻意**不**写 `using namespace RL;` —— `RL::Step`(一条经验) 与 `::Step`(象棋的一步)
   同名, 把 RL 整个拉进全局作用域会让每个 `Step` 变成歧义符号。
*/

namespace {

/* ============================================================
 *  配置 (默认值 = GUI 里 SAC+AZ-MoE 那一档)
 * ============================================================ */
struct Cfg {
    int games = 6;              /* 对局数 (强制偶数, 便于交换先后手) */
    int maxPlies = 100;         /* 每局手数上限 (到上限判和) */
    int sims = 16;              /* SAC+AZ 每步模拟次数 (GUI 的 SACAZ_MOE_SIMS) */
    int depth = 4;              /* ABAgent 搜索深度 (GUI 的 AB_DEPTH) */
    int openingPlies = 4;       /* 随机开局步数 */
    double msBudget = 0.0;      /* >0: 按"等时间"反推 SAC+AZ 的模拟次数 */
    int minSims = 2;
    int maxSims = 64;
    int warmupGames = 0;        /* 赛前自对弈训练局数 (0 = 不热身) */
    int warmupSims = 16;
    std::string backbone = "tb";
    std::string loadPrefix;
    std::string savePrefix;
    unsigned seed = 20240901;
    bool verbose = false;       /* 打印每一手 */
    bool quiet = false;         /* 只打最终一行结论 (静默验证) */
};

Cfg g_cfg;

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* SACAZAgent::backboneName() 的反查 (命令行 -> 枚举) */
static bool parseBackbone(const std::string &s, SACAZAgent::Backbone &out)
{
    if (s == "mlp")          { out = SACAZAgent::Backbone::Mlp;          return true; }
    if (s == "moe-mlp")      { out = SACAZAgent::Backbone::SparseMoeMlp; return true; }
    if (s == "tb")           { out = SACAZAgent::Backbone::SparseMoeTb;  return true; }
    if (s == "dense-tb")     { out = SACAZAgent::Backbone::DenseMoeTb;   return true; }
    return false;
}

/* ============================================================
 *  一局的统计
 * ============================================================ */
struct GameStat {
    int winner;                /* Chess::RESULT_* (棋盘视角: 红胜 / 黑胜 / 和) */
    int plies;
    bool sacWasRed;
    int simsUsed;
    int sacMoves, abMoves;
    double sacMs, abMs;        /* 每步平均耗时 */
    bool sacIllegal;           /* SAC 走了非法走法 (致命, 要退出码非 0) */
    bool abInvalid;            /* AB 返回了无效走法 */
    bool sacNoMove;            /* SAC 在有合法走法时返回了无效走法 */
    const char *endReason;
    double redMaterial, blackMaterial;

    GameStat()
        : winner(Chess::RESULT_ONGOING), plies(0), sacWasRed(false), simsUsed(0),
          sacMoves(0), abMoves(0), sacMs(0.0), abMs(0.0), sacIllegal(false),
          abInvalid(false), sacNoMove(false), endReason(""),
          redMaterial(0.0), blackMaterial(0.0) {}
};

/* 棋盘视角 -> SAC 视角: +1 SAC 胜, -1 SAC 负, 0 和 */
static int sacScore(const GameStat &g)
{
    if (g.winner == Chess::RESULT_DRAW) return 0;
    const bool redWon = (g.winner == Chess::RESULT_RED_WIN);
    return (redWon == g.sacWasRed) ? 1 : -1;
}

static void materialOf(Chess &c, double &red, double &black)
{
    red = 0.0;
    black = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = c.stones[i];
        if (s == nullptr || !s->alive) { continue; }
        if (s->color == Stone::COLOR_RED) { red += s->value; } else { black += s->value; }
    }
}

/* 随机开局: 双方都不参与, 只为把每局的起始局面岔开 */
static void randomOpening(Chess &c, int &turn, int plies)
{
    for (int i = 0; i < plies; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) { return; }
        std::vector<Step*> legal;
        c.sample(turn, legal);
        if (legal.empty()) {
            Steps::instance().put(legal);
            return;
        }
        std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
        const Step chosen = *legal[(std::size_t)pick(RL::Random::engine)];
        Steps::instance().put(legal);

        double dummy = 0.0;
        c.moveForward(&chosen, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
}

/* ============================================================
 *  打一局 (逐手校验合法性 —— 这是这个基准唯一的"断言")
 * ============================================================ */
static GameStat playGame(Chess &c, SACAZAgent &sac, ABAgent &ab,
                         bool sacIsRed, int simsForSac)
{
    GameStat g;
    g.sacWasRed = sacIsRed;
    g.simsUsed = simsForSac;

    c.reset();
    int turn = Stone::COLOR_RED;
    randomOpening(c, turn, g_cfg.openingPlies);

    double sacMs = 0.0, abMs = 0.0;

    while (g.plies < g_cfg.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            g.winner = res;
            g.endReason = (res == Chess::RESULT_DRAW) ? "repetition/60-move rule" : "mate";
            break;
        }

        const bool sacTurn = (turn == Stone::COLOR_RED) == sacIsRed;
        const double t0 = nowMs();
        const Step s = sacTurn ? sac.selectMove(turn, simsForSac, 0.0f)
                               : ab.getBestMove(turn);
        const double ms = nowMs() - t0;
        if (sacTurn) { sacMs += ms; g.sacMoves++; } else { abMs += ms; g.abMoves++; }

        /*
           **合法性校验 (逐手)**。放在 moveForward **之前**: 一旦 Board 吃了非法走法,
           后面所有断言都失去意义 (棋盘已经被污染)。
        */
        std::vector<Step*> legal;
        c.sample(turn, legal);
        const bool anyLegal = !legal.empty();
        bool legalMove = false;
        if (anyLegal && s.valid) {
            legalMove = c.isLegalMove(turn, &s);
        }
        Steps::instance().put(legal);

        if (!s.valid || !legalMove) {
            /*
               两种情况要分开记:
                 * 有合法走法却返回无效/非法走法 -> 这是**缺陷** (sacNoMove / sacIllegal);
                 * 对方也一样 (abInvalid)。
               注意: "没有合法走法"时 getResult 已经在上面把对局判结束了, 走不到这里。
            */
            if (sacTurn) {
                if (!s.valid) { g.sacNoMove = true; } else { g.sacIllegal = true; }
            } else {
                g.abInvalid = true;
            }
            g.endReason = sacTurn
                              ? (s.valid ? "SAC+AZ played an ILLEGAL move"
                                         : "SAC+AZ returned an invalid move")
                              : "AB returned an invalid move";
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-4s (%d,%d)->(%d,%d)  %7.1f ms%s\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        sacTurn ? "  [SAC+AZ]" : "  [AB]");
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        g.plies++;
    }

    if (g.winner == Chess::RESULT_ONGOING && !g.sacNoMove &&
        !g.sacIllegal && !g.abInvalid) {
        g.winner = Chess::RESULT_DRAW;
        g.endReason = "ply limit";
    }

    materialOf(c, g.redMaterial, g.blackMaterial);
    g.sacMs = (g.sacMoves > 0) ? sacMs / (double)g.sacMoves : 0.0;
    g.abMs  = (g.abMoves  > 0) ? abMs  / (double)g.abMoves  : 0.0;
    return g;
}

/* SAC+AZ 的 ms/模拟 (等时间模式的标定) */
static double calibrateMsPerSim(Chess &c, SACAZAgent &sac)
{
    const int probeSims = 4;
    double total = 0.0;
    int samples = 0;

    c.reset();
    for (int i = 0; i < 3; i++) {
        const double t0 = nowMs();
        const Step s = sac.selectMove(Stone::COLOR_RED, probeSims, 0.0f);
        const double ms = nowMs() - t0;
        if (!s.valid) { break; }
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        total += ms;
        samples++;
    }
    c.reset();

    if (samples == 0) { return 0.0; }
    return total / (double)samples / (double)probeSims;
}

/*
 * 数值健康: 在**固定局面 (初始局面)** 上跑一次策略与 Q, 全部必须是有限值,
 * 而且掩码口径必须成立 (合法集上 Σπ ≡ 1)。
 *
 * 两处刻意的讲究:
 *   * **先 c.reset()**: 这个函数原本直接拿"上一局结束时的棋盘"来查 —— 而那是终局
 *     (被将杀/困毙), `getLegalActions` 给出的掩码里一个合法动作都没有, 于是
 *     `maskedSoftmax` 走的是它那条"无合法动作 -> 保持全 0"的分支, Σπ = 0,
 *     报出来就是"Σπ 偏差 = 1.00" —— 一个看着像缺陷、其实是量错位置的数字
 *     (实测踩到过)。初始局面永远有 44 个合法走法, 是稳定的参照点。
 *   * 棋盘被 reset 不影响 agent (它们只在调用期间持有 Chess&, 不缓存局面)。
 */
static bool numericHealth(Chess &c, SACAZAgent &sac, double &qmax, double &piDev)
{
    c.reset();
    RL::Tensor state(SACAZAgent::STATE_DIM, 1);
    std::vector<Step*> legal;
    std::vector<int> idx;
    RL::Tensor mask(SACAZAgent::ACTION_DIM, 1);
    sac.getLegalActions(Stone::COLOR_RED, legal, idx, mask);
    Steps::instance().put(legal);
    sac.encodeStateFor(Stone::COLOR_RED, state);

    if (idx.empty()) {
        /* 防御: 真出现这种局面就把这次检查标成"跳过" (偏差记 0), 而不是报假缺陷 */
        qmax = 0.0;
        piDev = 0.0;
        return true;
    }

    RL::Tensor pi(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q1(SACAZAgent::ACTION_DIM, 1);
    RL::Tensor q2(SACAZAgent::ACTION_DIM, 1);
    sac.policy(state, mask, pi);
    sac.qValues(state, q1, q2);

    bool finite = true;
    qmax = 0.0;
    double piSum = 0.0;
    for (int a = 0; a < SACAZAgent::ACTION_DIM; a++) {
        if (!std::isfinite(pi[a]) || !std::isfinite(q1[a]) || !std::isfinite(q2[a])) {
            finite = false;
        }
        qmax = std::fmax(qmax, std::fmax(std::fabs((double)q1[a]),
                                         std::fabs((double)q2[a])));
        if (mask[a] > 0.5f) { piSum += (double)pi[a]; }
    }
    piDev = std::fabs(piSum - 1.0);
    return finite;
}

static void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *key) -> const char* {
            const std::size_t n = std::strlen(key);
            if (std::strncmp(a, key, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--games"))        { g_cfg.games = std::atoi(v); }
        else if (const char *v = val("--plies"))   { g_cfg.maxPlies = std::atoi(v); }
        else if (const char *v = val("--sims"))    { g_cfg.sims = std::atoi(v); }
        else if (const char *v = val("--depth"))   { g_cfg.depth = std::atoi(v); }
        else if (const char *v = val("--opening")) { g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--budget"))  { g_cfg.msBudget = std::atof(v); }
        else if (const char *v = val("--min-sims")){ g_cfg.minSims = std::atoi(v); }
        else if (const char *v = val("--max-sims")){ g_cfg.maxSims = std::atoi(v); }
        else if (const char *v = val("--warmup-games")) { g_cfg.warmupGames = std::atoi(v); }
        else if (const char *v = val("--warmup-sims"))  { g_cfg.warmupSims = std::atoi(v); }
        else if (const char *v = val("--backbone")){ g_cfg.backbone = v; }
        else if (const char *v = val("--load"))    { g_cfg.loadPrefix = v; }
        else if (const char *v = val("--save"))    { g_cfg.savePrefix = v; }
        else if (const char *v = val("--seed"))    { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (std::strcmp(a, "--verbose") == 0) { g_cfg.verbose = true; }
        else if (std::strcmp(a, "--quiet") == 0)   { g_cfg.quiet = true; }
    }

    if (g_cfg.games < 2)    { g_cfg.games = 2; }
    if (g_cfg.games % 2 != 0) { g_cfg.games++; }
    if (g_cfg.maxPlies < 2) { g_cfg.maxPlies = 2; }
    if (g_cfg.sims < 1)     { g_cfg.sims = 1; }
    if (g_cfg.depth < 1)    { g_cfg.depth = 1; }
    if (g_cfg.minSims < 1)  { g_cfg.minSims = 1; }
    if (g_cfg.maxSims < g_cfg.minSims) { g_cfg.maxSims = g_cfg.minSims; }
}

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    parseArgs(argc, argv);
    RL::Random::setSeed(g_cfg.seed);

    SACAZAgent::Backbone backbone = SACAZAgent::Backbone::SparseMoeTb;
    if (!parseBackbone(g_cfg.backbone, backbone)) {
        std::printf("[错误] 未知骨干: %s (可选: mlp / moe-mlp / tb / dense-tb)\n",
                    g_cfg.backbone.c_str());
        return 1;
    }

    /* 所有 agent 共用同一个 Chess 对象 (与 GUI 一致); 逐手串行, 没有并发 */
    Chess c;
    c.reset();

    const double tBuild0 = nowMs();
    /*
       参数与 GUI 的 SAC+AZ-MoE 完全一致 (SACAZ_HIDDEN/SACAZ_MOE_AUX + lr 0.001,
       c_puct 1.5, 模拟次数见 --sims 的默认值 16)。
       注意这个 agent 有**五个**网络 (actor + q1/q2 + 两个目标网), TB 专家下约 1.6 GB。
    */
    SACAZAgent sac(c, 64, 0.99f, 0.001f, 1.5f, backbone, 64, 0.1f);
    ABAgent ab(c, g_cfg.depth);
    const double tBuild1 = nowMs();

    if (!g_cfg.quiet) {
        std::printf("=== SAC+AZ(骨干=%s) vs Alpha-Beta 静默对弈验证 ===\n",
                    SACAZAgent::backboneName(backbone));
        std::printf("SAC+AZ  : state=%d action=%d  专家=%d topK=%d  actor 参数=%lld\n",
                    SACAZAgent::STATE_DIM, SACAZAgent::ACTION_DIM,
                    sac.moeExpertCount(), sac.moeTopK(), sac.actor.paramCount());
        std::printf("AB      : 深度 %d\n", g_cfg.depth);
        std::printf("规则    : %d 局 (交换先后手), 每局最多 %d 手, 随机开局 %d 步, seed=%u\n",
                    g_cfg.games, g_cfg.maxPlies, g_cfg.openingPlies, g_cfg.seed);
        std::printf("构建耗时: %.1f s (五个 TB 专家网络)\n", (tBuild1 - tBuild0) / 1000.0);
    }

    bool loaded = true;
    if (!g_cfg.loadPrefix.empty()) {
        loaded = sac.loadModel(g_cfg.loadPrefix);
        if (!g_cfg.quiet) {
            std::printf("权重    : %s -> %s\n", g_cfg.loadPrefix.c_str(),
                        loaded ? "载入成功" : "**载入失败**");
        }
        if (!loaded) {
            /* 显式要验一个已训练模型, 却载不进来 —— 这是失败, 不能装作没看见 */
            std::printf("VERDICT: FAIL (权重载入失败: %s)\n", g_cfg.loadPrefix.c_str());
            return 1;
        }
    } else if (!g_cfg.quiet) {
        std::printf("权重    : 未载入 —— SAC+AZ 是**随机初始化**的网络, "
                    "下面的比分不是棋力\n");
    }

    if (g_cfg.warmupGames > 0) {
        const double t0 = nowMs();
        sac.trainSelfPlay(g_cfg.warmupGames, g_cfg.warmupSims, g_cfg.maxPlies,
                          false /* verbose */, 1.0f, 0.25f);
        if (!g_cfg.quiet) {
            std::printf("热身    : 自对弈 %d 局 (%d 模拟/步), %.1f s; learnSteps=%d\n",
                        g_cfg.warmupGames, g_cfg.warmupSims, (nowMs() - t0) / 1000.0,
                        sac.getLearnSteps());
        }
    }

    int simsForSac = g_cfg.sims;
    if (g_cfg.msBudget > 0.0) {
        const double msPerSim = calibrateMsPerSim(c, sac);
        if (msPerSim > 0.0) {
            simsForSac = (int)(g_cfg.msBudget / msPerSim);
            simsForSac = std::max(g_cfg.minSims, std::min(g_cfg.maxSims, simsForSac));
            if (!g_cfg.quiet) {
                std::printf("等时间  : 标定 %.3f ms/模拟 -> 每步预算 %.0f ms = %d 次模拟\n",
                            msPerSim, g_cfg.msBudget, simsForSac);
            }
        } else if (!g_cfg.quiet) {
            std::printf("等时间  : 标定失败, 退回固定 %d 次模拟\n", simsForSac);
        }
    }
    if (!g_cfg.quiet) { std::printf("\n"); }

    /* ---- 对局 + 逐手校验 ---- */
    int sacWins = 0, abWins = 0, draws = 0, broken = 0;
    int pliesTotal = 0;
    double sacMsPerMove = 0.0, abMsPerMove = 0.0;
    int sacMoveSamples = 0, abMoveSamples = 0;
    bool allFinite = true;
    double worstQmax = 0.0;
    double worstPiDev = 0.0;

    const double tStart = nowMs();
    for (int i = 0; i < g_cfg.games; i++) {
        const bool sacIsRed = (i % 2 == 0);
        const GameStat g = playGame(c, sac, ab, sacIsRed, simsForSac);

        const int score = sacScore(g);
        const bool bad = g.sacIllegal || g.sacNoMove || g.abInvalid;
        if (bad)            { broken++; }
        else if (score > 0) { sacWins++; }
        else if (score < 0) { abWins++; }
        else                { draws++; }

        pliesTotal += g.plies;
        if (g.sacMoves > 0) { sacMsPerMove += g.sacMs * g.sacMoves; sacMoveSamples += g.sacMoves; }
        if (g.abMoves > 0)  { abMsPerMove  += g.abMs  * g.abMoves;  abMoveSamples  += g.abMoves; }

        /* 每局结束后查一次数值健康 + 掩码口径 (在初始局面上, 见 numericHealth) */
        double qmax = 0.0, piDev = 0.0;
        const bool finite = numericHealth(c, sac, qmax, piDev);
        allFinite = allFinite && finite;
        worstQmax = std::fmax(worstQmax, qmax);
        worstPiDev = std::fmax(worstPiDev, piDev);

        if (!g_cfg.quiet) {
            const char *verdict = bad ? "BROKEN"
                                : (score > 0) ? "SAC wins"
                                : (score < 0) ? "AB wins" : "draw";
            std::printf("  game %2d/%d  SAC=%-5s  %-9s %3d 手  [%s]"
                        "  sac %6.1f ms/步, ab %5.1f ms/步  材料 红%.1f-黑%.1f\n",
                        i + 1, g_cfg.games, sacIsRed ? "RED" : "BLACK", verdict, g.plies,
                        g.endReason, g.sacMs, g.abMs, g.redMaterial, g.blackMaterial);
        }
    }
    const double elapsed = (nowMs() - tStart) / 1000.0;

    /* ---- MoE 路由诊断 (top-1 最容易坍缩) ---- */
    std::vector<long long> usage;
    sac.moeUsage(usage);
    long long usageTotal = 0;
    int unused = 0;
    long long usageMax = 0;
    double usageMean = 0.0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        usageTotal += usage[i];
        if (usage[i] == 0) { unused++; }
        usageMax = std::max(usageMax, usage[i]);
    }
    if (!usage.empty()) { usageMean = (double)usageTotal / (double)usage.size(); }

    if (!g_cfg.quiet) {
        std::printf("\n=== 结果 (SAC+AZ 视角) ===\n");
        std::printf("  比分      : SAC+AZ %d 胜 / AB %d 胜 / 和 %d", sacWins, abWins, draws);
        if (broken > 0) { std::printf("  (+%d 局机制违规)", broken); }
        std::printf("\n");
        std::printf("  得分率    : %.1f%%  (胜=1 和=0.5; 随机权重下这个数字没有棋力含义)\n",
                    100.0 * ((double)sacWins + 0.5 * (double)draws) / (double)g_cfg.games);
        std::printf("  平均手数  : %.1f\n", (double)pliesTotal / (double)g_cfg.games);
        if (sacMoveSamples > 0 && abMoveSamples > 0) {
            const double sMs = sacMsPerMove / sacMoveSamples;
            const double aMs = abMsPerMove / abMoveSamples;
            std::printf("  思考时间  : SAC+AZ %.1f ms/步, AB %.1f ms/步  ->  AB 是 SAC 的 %.2fx\n",
                        sMs, aMs, sMs > 0.0 ? aMs / sMs : 0.0);
        }
        std::printf("  数值健康  : 全部有限=%d, |Q|max=%.3f, 合法集 Σπ 最大偏差=%.2e\n",
                    (int)allFinite, worstQmax, worstPiDev);        std::printf("  专家路由  : 总前向=%lld, 使用=[", usageTotal);
        for (std::size_t i = 0; i < usage.size(); i++) {
            std::printf("%lld%s", usage[i], (i + 1 < usage.size()) ? ", " : "]");
        }
        std::printf("  max/均值=%.2f, 未用到=%d\n",
                    usageMean > 0.0 ? (double)usageMax / usageMean : 0.0, unused);
        std::printf("  总耗时    : %.1f s\n", elapsed);
    }

    if (!g_cfg.savePrefix.empty()) {
        const bool ok = sac.saveModel(g_cfg.savePrefix);
        if (!g_cfg.quiet) {
            std::printf("  权重已保存: %s_actor / _q1 / _q2 -> %s\n",
                        g_cfg.savePrefix.c_str(), ok ? "成功" : "失败");
        }
    }

    /*
       退出码只看**机制**: 非法走法 / 无效走法 / 未结束的对局 / NaN / 掩码归一坏掉
       都是缺陷; 比分与路由均匀度只是诊断 (随机权重下输棋与路由偏斜都是预期内的)。
    */
    const bool piOk = (worstPiDev < 1e-4);
    const bool mechanismOk = (broken == 0) && allFinite && piOk;

    std::printf("VERDICT: %s | 骨干=%s 局数=%d 比分=%d/%d/%d 违规=%d 数值有限=%d "
                "Σπ偏差=%.1e 专家使用=%lld(未用到 %d)\n",
                mechanismOk ? "PASS" : "FAIL",
                SACAZAgent::backboneName(backbone), g_cfg.games,
                sacWins, abWins, draws, broken, (int)allFinite, worstPiDev,
                usageTotal, unused);

    return mechanismOk ? 0 : 1;
}
