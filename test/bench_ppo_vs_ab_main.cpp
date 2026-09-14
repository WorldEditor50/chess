/*
 * bench_ppo_vs_ab_main.cpp - PPO+MCTS 对 ABAgent 的**静默**对弈基准 (**不进 ctest**)
 * ============================================================================
 *
 * 这是 GUI 里"Agent 对 Agent 对弈"的无界面对应物: 不创建窗口、不弹对话框、
 * 不写权重文件 (除非显式 --save=), 只在控制台输出每局结果与最终统计。
 *
 * 为什么不放进 ctest: 它跑真实对局, 结果依赖随机开局、并且默认双方都是**随机
 * 初始权重** —— 几局之内翻转结论是常事, 放进 ctest 只会得到偶发失败
 * (与 bench_moe 同样的理由)。它只负责**量**。
 *
 * 公平性怎么保证 (缺一条比较就没有意义):
 *   1. **交换先后手**: 局数强制为偶数, 一半局 PPO 执红、一半执黑。中国象棋先手
 *      优势很大, 固定谁执红等于在测"谁执红"而不是"谁强"。
 *   2. **随机开局**: 每局先随机走 openingPlies 步合法棋。否则 temp=0 加上相同的
 *      初始权重会让每局下成同一盘棋, 胜率只有 0/1 两个取值。
 *   3. **等时间 (可选 --budget=MS)**: 固定"PPO 80 次模拟 vs AB 深度 4"其实是在比
 *      谁算得多。给了 --budget 就先标定 PPO 的 ms/模拟, 再按预算反推每步模拟次数,
 *      让双方每步的思考时间大体相同。不给则用固定 --sims, 报告里会把双方实测的
 *      ms/步 一并打出来 —— 自己看一眼就知道这两组数字是不是等时间的。
 *
 * 关于"棋力"结论的重要前提:
 *   默认**不载入任何权重**, 所以 PPO+MCTS 是随机初始化的网络。这时测出来的不是
 *   棋力, 而是"这个搜索 + 随机策略网络"的下限。要测训练之后的样子:
 *     --warmup-games=K     先自对弈 K 局在线训练 (杯水车薪, 只确认链路不炸)
 *     --load=<prefix>      载入 weightFile 保存的 <prefix>_actor / <prefix>_critic
 *   (GUI 存的是 weights/ppomcts_agent.dat, 对应 weights/ppomcts_agent.dat_actor
 *    与 ..._critic —— 所以 --load=weights/ppomcts_agent.dat 就是载入 GUI 的权重。)
 *
 * 用法:
 *   bench_ppo_vs_ab.exe [--games=6] [--plies=120] [--sims=80] [--depth=4]
 *                       [--opening=4] [--budget=0] [--min-sims=8] [--max-sims=2000]
 *                       [--warmup-games=0] [--warmup-sims=50]
 *                       [--load=PREFIX] [--save=PREFIX] [--seed=N] [--verbose]
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
#include "ppomcts_agent.h"
#include "rl/util.hpp"

/*
  刻意**不**写 `using namespace RL;` —— `RL::Step`(一条经验) 与 `::Step`(象棋的一步)
  同名, 一旦把 RL 整个拉进全局作用域, 文件里每个 `Step` 都会变成歧义符号
  (实测: error C2872)。这里只把用到的 RL::Random 写全。
*/

namespace {

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    int games = 6;              /* 对局数 (强制偶数, 便于交换先后手) */
    int maxPlies = 120;         /* 每局手数上限 (到上限判和) */
    int sims = 80;              /* PPO+MCTS 每步模拟次数 (默认与 GUI 的 PPO_SIMS 一致) */
    int depth = 4;              /* ABAgent 搜索深度 (默认与 GUI 一致) */
    int openingPlies = 4;       /* 随机开局步数 */
    double msBudget = 0.0;      /* >0: 按"等时间"反推 PPO 的模拟次数 */
    int minSims = 8;
    int maxSims = 2000;
    int warmupGames = 0;        /* 赛前 PPO 自对弈训练局数 (0 = 不热身) */
    int warmupSims = 50;
    std::string loadPrefix;     /* 非空: 赛前载入权重 */
    std::string savePrefix;     /* 非空: 赛后保存权重 */
    unsigned seed = 20240901;
    bool verbose = false;       /* 打印每一手 */
};

Cfg g_cfg;

/* ============================================================
 *  计时
 * ============================================================ */
static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* ============================================================
 *  一局的统计
 * ============================================================ */
struct GameStat {
    int winner;            /* Chess::RESULT_* (棋盘视角: 红胜 / 黑胜 / 和) */
    int plies;
    bool ppoWasRed;
    int simsUsed;          /* 本局 PPO 实际用的模拟次数 (等时间模式下会变) */
    int ppoMoves, abMoves;
    double ppoMs, abMs;    /* 每步平均耗时 */
    bool aborted;          /* 有 agent 返回了空走法 (不该发生, 要显式报出来) */
    const char *endReason; /* 终局原因 */

    GameStat()
        : winner(Chess::RESULT_ONGOING), plies(0), ppoWasRed(false), simsUsed(0),
          ppoMoves(0), abMoves(0), ppoMs(0.0), abMs(0.0), aborted(false),
          endReason("") {}
};

/* 把棋盘视角的结果翻成"PPO 视角": +1 PPO 胜, -1 PPO 负, 0 和 */
static int ppoScore(const GameStat &g)
{
    if (g.winner == Chess::RESULT_DRAW) return 0;
    const bool redWon = (g.winner == Chess::RESULT_RED_WIN);
    return (redWon == g.ppoWasRed) ? 1 : -1;
}

/*
 *  随机开局: 双方都不参与, 只为把每局的起始局面岔开。
 *  注意先把选中的走法**拷贝**出来再归还对象池 —— 反过来 (先 put 再解引用) 虽然在
 *  当前实现下也能跑 (池只存指针、不析构), 但那是依赖"中间没有别的 get()"这个
 *  隐含前提的写法, 换个调用顺序就会踩到。
 */
static void randomOpening(Chess &c, int &turn, int plies)
{
    for (int i = 0; i < plies; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) {
            return;
        }
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
 *  打一局
 * ============================================================ */
static GameStat playGame(Chess &c, PPOMCTSAgent &ppo, ABAgent &ab,
                         bool ppoIsRed, int simsForPpo)
{
    GameStat g;
    g.ppoWasRed = ppoIsRed;
    g.simsUsed = simsForPpo;

    c.reset();
    int turn = Stone::COLOR_RED;
    randomOpening(c, turn, g_cfg.openingPlies);

    double ppoMs = 0.0, abMs = 0.0;

    while (g.plies < g_cfg.maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            g.winner = res;
            g.endReason = (res == Chess::RESULT_DRAW) ? "repetition/60-move rule" : "mate";
            break;
        }

        const bool ppoTurn = (turn == Stone::COLOR_RED) == ppoIsRed;
        const double t0 = nowMs();
        const Step s = ppoTurn ? ppo.selectMove(turn, simsForPpo, 0.0f)
                               : ab.getBestMove(turn);
        const double ms = nowMs() - t0;
        if (ppoTurn) { ppoMs += ms; g.ppoMoves++; } else { abMs += ms; g.abMoves++; }

        if (!s.valid) {
            /* selectMove / getBestMove 都有"无走法"兜底, 正常不该走到这里。
               显式记下来而不是悄悄当和棋 —— 无效走法本身就是个要修的缺陷。 */
            g.aborted = true;
            g.endReason = ppoTurn ? "PPO returned an invalid move"
                                  : "AB returned an invalid move";
            break;
        }

        if (g_cfg.verbose) {
            std::printf("      ply %3d %-4s (%d,%d)->(%d,%d)  %7.1f ms%s\n",
                        g.plies + 1, (turn == Stone::COLOR_RED) ? "RED" : "BLACK",
                        s.pos.x, s.pos.y, s.nextPos.x, s.nextPos.y, ms,
                        ppoTurn ? "  [PPO]" : "  [AB]");
        }

        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        g.plies++;
    }

    if (g.winner == Chess::RESULT_ONGOING && !g.aborted) {
        /* 到上限: 判和 (与 ChessBoard::matchAgents / bench_moe 的约定一致) */
        g.winner = Chess::RESULT_DRAW;
        g.endReason = "ply limit";
    }

    g.ppoMs = (g.ppoMoves > 0) ? ppoMs / (double)g.ppoMoves : 0.0;
    g.abMs  = (g.abMoves  > 0) ? abMs  / (double)g.abMoves  : 0.0;
    return g;
}

/* ============================================================
 *  标定: 测出 PPO 的 ms/模拟, 好把 --budget 换算成模拟次数
 * ============================================================ */
static double calibrateMsPerSim(Chess &c, PPOMCTSAgent &ppo)
{
    const int probeSims = 40;
    double total = 0.0;
    int samples = 0;

    c.reset();
    for (int i = 0; i < 3; i++) {
        const double t0 = nowMs();
        const Step s = ppo.selectMove(Stone::COLOR_RED, probeSims, 0.0f);
        const double ms = nowMs() - t0;
        if (!s.valid) {
            break;
        }
        /* 走一步再测, 避免三次都在同一个局面上重复同样的树 */
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        total += ms;
        samples++;
    }
    c.reset();

    if (samples == 0) {
        return 0.0;
    }
    return total / (double)samples / (double)probeSims;
}

/* ============================================================
 *  参数
 * ============================================================ */
static void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *key) -> const char* {
            const std::size_t n = std::strlen(key);
            if (std::strncmp(a, key, n) == 0 && a[n] == '=') {
                return a + n + 1;
            }
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
        else if (const char *v = val("--load"))    { g_cfg.loadPrefix = v; }
        else if (const char *v = val("--save"))    { g_cfg.savePrefix = v; }
        else if (const char *v = val("--seed"))    { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (std::strcmp(a, "--verbose") == 0) { g_cfg.verbose = true; }
    }

    if (g_cfg.games < 2)    { g_cfg.games = 2; }
    if (g_cfg.games % 2 != 0) { g_cfg.games++; }   /* 保证先后手各占一半 */
    if (g_cfg.maxPlies < 2) { g_cfg.maxPlies = 2; }
    if (g_cfg.sims < 1)     { g_cfg.sims = 1; }
    if (g_cfg.depth < 1)    { g_cfg.depth = 1; }
    if (g_cfg.minSims < 1)  { g_cfg.minSims = 1; }
    if (g_cfg.maxSims < g_cfg.minSims) { g_cfg.maxSims = g_cfg.minSims; }
}

} // namespace

/* ============================================================
 *  main
 * ============================================================ */
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* 长跑时也能实时看到进度 */
    parseArgs(argc, argv);
    RL::Random::setSeed(g_cfg.seed);

    /* 所有 agent 共用同一个 Chess 对象 —— 与 GUI 里所有 agent 共用 env 一致。
       agent 只持有 Chess&, 搜索用完会把棋盘复原; 逐手串行调用, 没有并发。 */
    Chess c;
    c.reset();

    PPOMCTSAgent ppo(c, 64, 0.99f, 0.001f, 1.414f);
    ABAgent ab(c, g_cfg.depth);

    std::printf("=== PPO+MCTS vs Alpha-Beta 静默对弈基准 ===\n");
    std::printf("PPO+MCTS : state=%d action=%d 稀疏MoE(E=%d,top-%d) 每步 %d 次模拟\n",
                PPOMCTSAgent::STATE_DIM, PPOMCTSAgent::ACTION_DIM,
                ppo.moeExpertCount(), ppo.moeTopK(), g_cfg.sims);
    std::printf("AB-Agent : 深度 %d\n", g_cfg.depth);
    std::printf("规则     : %d 局 (交换先后手), 每局最多 %d 手, 随机开局 %d 步, seed=%u\n",
                g_cfg.games, g_cfg.maxPlies, g_cfg.openingPlies, g_cfg.seed);

    /* ---- 权重来源: 明确写出来, 免得拿"未训练"的结果当棋力 ---- */
    bool loaded = false;
    if (!g_cfg.loadPrefix.empty()) {
        loaded = ppo.loadModel(g_cfg.loadPrefix);
        std::printf("权重     : %s -> %s\n", g_cfg.loadPrefix.c_str(),
                    loaded ? "载入成功" : "载入失败(文件缺失或结构不匹配), 继续用随机初始权重");
    } else {
        std::printf("权重     : 未载入 —— PPO 是**随机初始化**的网络, 下面的数字不是棋力\n");
    }
    if (g_cfg.warmupGames > 0) {
        std::printf("热身     : PPO 自对弈 %d 局 (%d 次模拟/步), 之后才对局\n",
                    g_cfg.warmupGames, g_cfg.warmupSims);
    }

    /* ---- 热身 (静默: 不让 trainSelfPlay 自己打进度) ---- */
    if (g_cfg.warmupGames > 0) {
        const double t0 = nowMs();
        ppo.trainSelfPlay(g_cfg.warmupGames, g_cfg.warmupSims, g_cfg.maxPlies,
                          false /* verbose */, 1.0f, 0.1f);
        std::printf("           热身完成, %.1f s\n", (nowMs() - t0) / 1000.0);
    }

    /* ---- 等时间模式: 标定后把预算换算成模拟次数 ---- */
    int simsForPpo = g_cfg.sims;
    if (g_cfg.msBudget > 0.0) {
        const double msPerSim = calibrateMsPerSim(c, ppo);
        if (msPerSim > 0.0) {
            simsForPpo = (int)(g_cfg.msBudget / msPerSim);
            simsForPpo = std::max(g_cfg.minSims, std::min(g_cfg.maxSims, simsForPpo));
            std::printf("等时间   : 标定 %.3f ms/模拟 -> 每步预算 %.0f ms = %d 次模拟\n",
                        msPerSim, g_cfg.msBudget, simsForPpo);
        } else {
            std::printf("等时间   : 标定失败, 退回固定 %d 次模拟\n", simsForPpo);
        }
    }
    std::printf("\n");

    /* ---- 对局 ---- */
    int ppoWins = 0, abWins = 0, draws = 0, aborted = 0;
    int pliesTotal = 0;
    double ppoMsPerMove = 0.0, abMsPerMove = 0.0;
    int ppoMoveSamples = 0, abMoveSamples = 0;

    const double tStart = nowMs();
    for (int i = 0; i < g_cfg.games; i++) {
        const bool ppoIsRed = (i % 2 == 0);   /* 一半执红、一半执黑 */
        const GameStat g = playGame(c, ppo, ab, ppoIsRed, simsForPpo);

        const int score = ppoScore(g);
        if (g.aborted)      { aborted++; }
        else if (score > 0) { ppoWins++; }
        else if (score < 0) { abWins++; }
        else                { draws++; }

        pliesTotal += g.plies;
        if (g.ppoMoves > 0) { ppoMsPerMove += g.ppoMs * g.ppoMoves; ppoMoveSamples += g.ppoMoves; }
        if (g.abMoves > 0)  { abMsPerMove  += g.abMs  * g.abMoves;  abMoveSamples  += g.abMoves; }

        const char *verdict = g.aborted ? "ABORTED"
                            : (score > 0) ? "PPO wins"
                            : (score < 0) ? "AB wins" : "draw";
        std::printf("  game %2d/%d  PPO=%-4s  %-9s in %3d plies  [%s]"
                    "  ppo %5.1f ms/move, ab %5.1f ms/move\n",
                    i + 1, g_cfg.games, ppoIsRed ? "RED" : "BLACK",
                    verdict, g.plies, g.endReason, g.ppoMs, g.abMs);
    }
    const double elapsed = (nowMs() - tStart) / 1000.0;

    /* ---- 汇总 ---- */
    std::printf("\n=== 结果 (PPO+MCTS 视角) ===\n");
    std::printf("  比分      : PPO %d 胜 / AB %d 胜 / 和 %d", ppoWins, abWins, draws);
    if (aborted > 0) {
        std::printf("  (+%d 局因无效走法中止, 不计入比分)", aborted);
    }
    std::printf("\n");
    std::printf("  得分率    : %.1f%%  (胜=1 和=0.5)\n",
                100.0 * ((double)ppoWins + 0.5 * (double)draws) / (double)g_cfg.games);
    std::printf("  平均手数  : %.1f\n", (double)pliesTotal / (double)g_cfg.games);
    if (ppoMoveSamples > 0 && abMoveSamples > 0) {
        std::printf("  思考时间  : PPO %.1f ms/步, AB %.1f ms/步  ->  AB 是 PPO 的 %.2fx\n",
                    ppoMsPerMove / ppoMoveSamples, abMsPerMove / abMoveSamples,
                    (ppoMsPerMove / ppoMoveSamples) > 0.0
                        ? (abMsPerMove / abMoveSamples) / (ppoMsPerMove / ppoMoveSamples)
                        : 0.0);
    }
    std::printf("  对局总耗时: %.1f s\n", elapsed);

    if (!g_cfg.savePrefix.empty()) {
        const bool ok = ppo.saveModel(g_cfg.savePrefix);
        std::printf("  权重已保存: %s_actor / %s_critic  -> %s\n",
                    g_cfg.savePrefix.c_str(), g_cfg.savePrefix.c_str(),
                    ok ? "成功" : "失败");
    }

    return 0;
}
