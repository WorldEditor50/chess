/*
 * bench_moe_main.cpp - 骨干 A/B/C/D 的等时对弈基准 (手动运行, **不进 ctest**)
 * ============================================================================
 *
 * 为什么单独一个可执行文件而不是测试: 它跑的是"对局", 结果依赖随机开局, 而且
 * TB 专家骨干一步要几十毫秒 —— 放进 ctest 会既慢又不稳定。这里只负责**量**,
 * 结论写进 docs/agents_design.md §11.4.2。
 *
 * 四个参赛者 (同一个 SAC+MCTS+AlphaZero 算法, 只换骨干):
 *   A  MLP              1260 -> 64 -> 64 -> 128            参数量最小, 算力最小
 *   B  稀疏MoE(MLP专家)  E=8 top-2, 专家 1260->64->64->1260  容量 x8, 算力 ~2 个专家
 *   C  稀疏MoE(TB专家)   E=4 top-1, 专家 = TransformerBlock  容量最大
 *   D  稠密MoE(TB专家)   E=4 全算  —— 与 C **参数量完全相同**, 只是全部计算
 *
 * 公平性是怎么保证的 (这三点缺一个, 比较就没有意义):
 *   1. **等时间**: 每个参赛者的模拟次数不是固定值, 而是先用一次标定测出它自己的
 *      ms/模拟, 再取 "预算 / ms每模拟"。于是四个参赛者每一步的思考时间大体相同。
 *      固定模拟次数去比"谁强"是没意义的 —— 那只是在比谁算得多。
 *   2. **交换先后手**: 同一对选手打偶数局, 一半局 A 执红、一半执黑, 避免先手优势
 *      被算到某一个 agent 头上。
 *   3. **随机开局**: 每局先随机走 openingPlies 步合法棋。否则 temp=0 + 相同的
 *      随机初始权重会让每局都下成同一盘棋, 胜率只有 0/1 两个取值。
 *   4. **预训练步数可设 (默认 0)**: 默认两边都是随机初始权重 —— 这时比的是
 *      "结构与容量的归纳偏置", **不是棋力**。任何棋力结论都需要真正的训练预算,
 *      这一点在报告里会写明 (--pretrain=K 可以给两方同样的 K 局自对弈热身,
 *      但那依然是杯水车薪, 只用来确认训练链路在真实对局里也不会炸)。
 *
 * 用法:
 *   bench_moe.exe [--games=4] [--plies=40] [--budget=60] [--opening=6] [--pretrain=0]
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
#include "sacazagent.h"
#include "rl/sparse_moe.hpp"

using namespace RL;

/* ============================================================
 *  配置
 * ============================================================ */
struct Cfg {
    int games = 4;          /* 每一对选手的对局数 (会强制为偶数, 便于交换先后手) */
    int maxPlies = 40;      /* 每局手数上限 (到上限判和) */
    double msBudget = 60.0; /* 每步思考的时间预算 (毫秒) */
    int maxSims = 64;       /* 模拟次数上限 (便宜的骨干不要无限加) */
    int openingPlies = 6;   /* 随机开局步数 */
    int pretrain = 0;       /* 每方在赛前自对弈的局数 (0 = 不热身) */
    unsigned seed = 20240501;
    float aux = 0.1f;       /* 稀疏 MoE 的负载均衡辅助损失系数 */
    /*
       参与标定/预训练/对弈的参赛者 (字符串里出现的字母)。默认四个全上。
       只跑其中一个主要用于"诊断模式" —— 例如只跑 B 来看它的路由分布随 aux 的变化。
       注意: 不参赛的选手不会被预训练, 所以这种跑法**不能**用来比棋力。
    */
    bool sel[4] = { true, true, true, true };
};

static Cfg g_cfg;

/* 参赛者: 一个骨干 + 它的权重前缀 (只用于报告) */
struct Case {
    SACAZAgent::Backbone b;
    const char *tag;
};

static const Case kCases[4] = {
    { SACAZAgent::Backbone::Mlp,         "A  MLP            " },
    { SACAZAgent::Backbone::SparseMoeMlp, "B  稀疏MoE(MLP专家)" },
    { SACAZAgent::Backbone::SparseMoeTb,  "C  稀疏MoE(TB专家) " },
    { SACAZAgent::Backbone::DenseMoeTb,   "D  稠密MoE(TB专家) " }
};

/* 分析式参数量 (只用来在报告里说明"等参数"这件事) */
static double paramCount(SACAZAgent::Backbone b, int hidden)
{
    const double D = (double)SACAZAgent::STATE_DIM;
    const double A = (double)SACAZAgent::ACTION_DIM;
    const double h = (double)hidden;
    const double mlp = D * h + h + h * h + h + h * A + A;   /* 1260->h->h->A */
    switch (b) {
    case SACAZAgent::Backbone::Mlp:
        return mlp;
    case SACAZAgent::Backbone::SparseMoeMlp: {
        const double eh = 64.0;
        /* 每个专家: D->eh->eh->D */
        const double expert = D * eh + eh + eh * eh + eh + eh * D + D;
        const double gate = (double)SACAZAgent::MOE_MLP_EXPERTS * D
                            + (double)SACAZAgent::MOE_MLP_EXPERTS;
        /* MoE 层 + Tanh(D->h) + Linear(h->A) */
        return (double)SACAZAgent::MOE_MLP_EXPERTS * expert + gate + D * h + h + h * A + A;
    }
    case SACAZAgent::Backbone::SparseMoeTb:
    case SACAZAgent::Backbone::DenseMoeTb: {
        const double dff = (double)SACAZAgent::MOE_TB_DFF;
        /* 一个 TB 专家: 4 个 d x d 的投影 + LN(2*2*d) + FFN(d->dff->d) */
        const double expert = 4.0 * D * D + 4.0 * D + (D * dff + dff) + (dff * D + D);
        const double gate = (double)SACAZAgent::MOE_TB_EXPERTS * D
                            + (double)SACAZAgent::MOE_TB_EXPERTS;
        return (double)SACAZAgent::MOE_TB_EXPERTS * expert + gate + D * h + h + h * A + A;
    }
    default:
        return mlp;
    }
}

/* ============================================================
 *  一局对局
 *
 *  两边共用一个 Chess (与 GUI 里所有 agent 共用 env 是同一个模式), 逐手串行,
 *  所以不存在并发问题。记录每一边的实际思考耗时, 用来验证"等时间"这件事真的成立。
 * ============================================================ */
struct GameResult {
    int winner;          /* Chess::RESULT_* */
    int plies;
    double msCase;       /* case 一方每步的平均耗时 */
    double msRef;        /* 参照方每步的平均耗时 */
    bool aborted;
};

static GameResult playGame(Chess &c, SACAZAgent &caseAgent, SACAZAgent &refAgent,
                           int simsCase, int simsRef, bool caseIsRed, int maxPlies,
                           int openingPlies)
{
    GameResult r;
    r.winner = Chess::RESULT_ONGOING;
    r.plies = 0;
    r.msCase = 0.0;
    r.msRef = 0.0;
    r.aborted = false;

    c.reset();
    int turn = Stone::COLOR_RED;
    int caseMoves = 0, refMoves = 0;
    double caseMs = 0.0, refMs = 0.0;

    /* ---- 随机开局: 双方都不参与, 只为了制造不同的起始局面 ---- */
    for (int i = 0; i < openingPlies && turn >= 0; i++) {
        if (c.getResult(turn) != Chess::RESULT_ONGOING) {
            break;
        }
        std::vector<Step*> legal;
        c.sample(turn, legal);
        Steps::instance().put(legal);
        if (legal.empty()) {
            break;
        }
        std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
        const Step s = *legal[(std::size_t)pick(Random::engine)];
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }

    while (r.plies < maxPlies) {
        const int res = c.getResult(turn);
        if (res != Chess::RESULT_ONGOING) {
            r.winner = res;
            break;
        }
        const bool isCaseTurn = (turn == Stone::COLOR_RED) == caseIsRed;
        SACAZAgent &me = isCaseTurn ? caseAgent : refAgent;
        const int sims = isCaseTurn ? simsCase : simsRef;

        const auto t0 = std::chrono::steady_clock::now();
        const Step s = me.selectMove(turn, sims, 0.0f);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        if (isCaseTurn) {
            caseMs += ms;
            caseMoves++;
        } else {
            refMs += ms;
            refMoves++;
        }

        if (!s.valid) {
            /* 不该发生: selectMove 有终局/无走法兜底。记下来, 别静默当成和棋。 */
            r.aborted = true;
            std::printf("      [警告] selectMove 返回了空走法 (第 %d 手)\n", r.plies);
            break;
        }
        double dummy = 0.0;
        c.moveForward(&s, dummy);
        turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        r.plies++;
    }

    if (r.winner == Chess::RESULT_ONGOING) {
        /* 到上限: 判和 (与 ChessBoard::matchAgents 的约定一致) */
        r.winner = Chess::RESULT_DRAW;
    }
    r.msCase = caseMoves > 0 ? caseMs / (double)caseMoves : 0.0;
    r.msRef = refMoves > 0 ? refMs / (double)refMoves : 0.0;
    return r;
}

/* ============================================================
 *  主流程
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
        if (const char *v = val("--games"))    { g_cfg.games = std::atoi(v); }
        else if (const char *v = val("--plies"))   { g_cfg.maxPlies = std::atoi(v); }
        else if (const char *v = val("--budget"))  { g_cfg.msBudget = std::atof(v); }
        else if (const char *v = val("--maxsims")) { g_cfg.maxSims = std::atoi(v); }
        else if (const char *v = val("--opening")) { g_cfg.openingPlies = std::atoi(v); }
        else if (const char *v = val("--pretrain")){ g_cfg.pretrain = std::atoi(v); }
        else if (const char *v = val("--seed"))    { g_cfg.seed = (unsigned)std::atoi(v); }
        else if (const char *v = val("--aux"))     { g_cfg.aux = (float)std::atof(v); }
        else if (const char *v = val("--cases")) {
            for (int k = 0; k < 4; k++) {
                g_cfg.sel[k] = false;
            }
            for (const char *p = v; *p != '\0'; p++) {
                const int k = (*p >= 'a') ? (*p - 'a') : (*p - 'A');
                if (k >= 0 && k < 4) {
                    g_cfg.sel[k] = true;
                }
            }
        }
    }
    if (g_cfg.games < 2) {
        g_cfg.games = 2;
    }
    if (g_cfg.games % 2 != 0) {
        g_cfg.games++;   /* 保证先后手各占一半 */
    }
}

int main(int argc, char **argv)
{
    parseArgs(argc, argv);
    Random::engine.seed(g_cfg.seed);

    std::printf("=== 骨干 A/B/C/D 等时对弈基准 ===\n");
    std::printf("配置: 每对 %d 局, 每局最多 %d 手, 每步预算 %.0f ms, 随机开局 %d 步, "
                "预训练 %d 局\n\n",
                g_cfg.games, g_cfg.maxPlies, g_cfg.msBudget, g_cfg.openingPlies,
                g_cfg.pretrain);

    /*
       所有 agent 共用同一个 Chess 对象 —— 与 GUI 里所有 agent 共用 env 完全一致。
       (agent 内部只持有 Chess&, 搜索时用完会把棋盘复原; 逐手串行调用, 没有并发。)
    */
    Chess c;
    c.reset();

    const int hidden = 64;
    SACAZAgent::Backbone backs[4] = {
        kCases[0].b, kCases[1].b, kCases[2].b, kCases[3].b
    };
    SACAZAgent *agents[4] = { nullptr, nullptr, nullptr, nullptr };
    for (int i = 0; i < 4; i++) {
        agents[i] = new SACAZAgent(c, hidden, 0.99f, 0.001f, 1.5f, backs[i], 64, g_cfg.aux);
        agents[i]->batchSize = 8;
    }
    if (!g_cfg.sel[0]) {
        std::printf("注意: A (MLP 参照) 不在 --cases 里, 它不会被预训练 —— 这种跑法\n"
                    "      只能用于看路由分布/代价这类诊断, **不能**用来比棋力。\n\n");
    }

    /* ---- 1. 标定: 每个骨干自己的 ms/模拟 ---- */
    std::printf("[1] 标定 (初始局面, 16 次模拟) -> 等时间预算下的模拟次数\n");
    double msPerSim[4] = { 0, 0, 0, 0 };
    int simsFor[4] = { 0, 0, 0, 0 };
    const int calibSims = 16;
    for (int i = 0; i < 4; i++) {
        c.reset();
        agents[i]->selectMove(Stone::COLOR_RED, 2, 0.0f);   /* warm up */
        const auto t0 = std::chrono::steady_clock::now();
        const Step s = agents[i]->selectMove(Stone::COLOR_RED, calibSims, 0.0f);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        msPerSim[i] = ms / (double)calibSims;
        int s2 = (int)std::lround(g_cfg.msBudget / (msPerSim[i] > 1e-6 ? msPerSim[i] : 1e-6));
        s2 = std::max(1, std::min(g_cfg.maxSims, s2));
        simsFor[i] = s2;
        std::printf("    %s  %7.3f ms/模拟   参数量 %6.2f M   等时模拟次数 = %d"
                    "  (约 %.0f ms/步)%s\n",
                    kCases[i].tag, msPerSim[i], paramCount(backs[i], hidden) / 1e6, s2,
                    msPerSim[i] * (double)s2, s.valid ? "" : "  [标定时走法无效?]");
    }

    /*
       预训练 (默认关闭)。给两方**相同**的预算, 否则比的是训练量而不是骨干。
       这里的预算小得只够验证"训练链路在真实对局里不炸", 不足以谈棋力。
    */
    if (g_cfg.pretrain > 0) {
        std::printf("\n[2] 预训练 %d 局自对弈 (两方同预算, 只用于验证训练链路)\n",
                    g_cfg.pretrain);
        for (int i = 0; i < 4; i++) {
            if (!g_cfg.sel[i]) {
                continue;
            }
            const int sims = std::max(4, std::min(16, simsFor[i]));
            const auto t0 = std::chrono::steady_clock::now();
            agents[i]->trainSelfPlay(g_cfg.pretrain, sims, 30, false, 1.0f, 0.25f, 4);
            const auto t1 = std::chrono::steady_clock::now();
            std::printf("    %s  %5.1f s, learnSteps=%d, alpha=%.3f, 池=%zu\n",
                        kCases[i].tag,
                        (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                            t1 - t0).count() / 1000.0,
                        agents[i]->getLearnSteps(), (double)agents[i]->getAlpha(),
                        agents[i]->getMemorySize());
            /*
               训练期间的专家使用分布 —— 这才是"辅助损失有没有把路由养活"的证据:
               如果某个专家在训练中一次都没被选中, 说明不加辅助损失时路由塌了
               (对比: auxLossCoef=0 时会迅速只剩一个专家)。
            */
            std::vector<long long> usage;
            agents[i]->moeUsage(usage);
            if (!usage.empty()) {
                long long tot = 0, mx = 0, zero = 0;
                for (std::size_t k = 0; k < usage.size(); k++) {
                    tot += usage[k];
                    mx = std::max(mx, usage[k]);
                    if (usage[k] == 0) {
                        zero++;
                    }
                }
                const double mean = (double)tot / (double)usage.size();
                std::printf("        训练期专家使用 = {");
                for (std::size_t k = 0; k < usage.size(); k++) {
                    std::printf("%lld%s", usage[k], (k + 1 < usage.size()) ? ", " : "}");
                }
                std::printf("  最大/均值 = %.2f, 一个都没用到的专家 = %lld 个\n",
                            mean > 0 ? (double)mx / mean : 0.0, zero);
            }
            agents[i]->resetMoeUsage();
        }
    }

    /* ---- 3. 对弈: A(MLP) 作为参照, B/C/D 各自与它打 ---- */
    std::printf("\n[3] 对弈 (A = MLP 骨干作为参照; 每对交换先后手, 随机开局 %d 步)\n",
                g_cfg.openingPlies);

    struct Tally {
        int win = 0, draw = 0, loss = 0;
        double msCase = 0.0, msRef = 0.0;
        int games = 0, aborted = 0;
        int plies = 0;
    };
    Tally tally[4];

    for (int i = 0; i < 4; i++) {
        if (!g_cfg.sel[i]) {
            continue;
        }
        /* i == 0 是 A 对 A 的对照: 两边完全一样, 胜率应当接近 50% —— 这就是噪声底 */
        SACAZAgent &caseAgent = *agents[i];
        SACAZAgent &refAgent = *agents[0];
        const int simsCase = simsFor[i];
        const int simsRef = simsFor[0];
        tally[i].games = g_cfg.games;

        std::printf("    %s vs A  (模拟 %d vs %d)%s\n", kCases[i].tag, simsCase, simsRef,
                    (i == 0) ? "   <- 自我对照 (噪声底)" : "");

        for (int gme = 0; gme < g_cfg.games; gme++) {
            const bool caseIsRed = (gme % 2 == 0);
            const GameResult gr = playGame(c, caseAgent, refAgent, simsCase, simsRef,
                                           caseIsRed, g_cfg.maxPlies, g_cfg.openingPlies);
            /* 从 case 一方的视角记胜负 */
            const int caseColor = caseIsRed ? Stone::COLOR_RED : Stone::COLOR_BLACK;
            bool win = false, loss = false;
            if (gr.winner == Chess::RESULT_DRAW) {
                tally[i].draw++;
            } else if ((gr.winner == Chess::RESULT_RED_WIN && caseColor == Stone::COLOR_RED)
                       || (gr.winner == Chess::RESULT_BLACK_WIN
                           && caseColor == Stone::COLOR_BLACK)) {
                tally[i].win++;
                win = true;
            } else {
                tally[i].loss++;
                loss = true;
            }
            tally[i].msCase += gr.msCase;
            tally[i].msRef += gr.msRef;
            tally[i].plies += gr.plies;
            if (gr.aborted) {
                tally[i].aborted++;
            }
            std::printf("      第 %d 局: %s 执%s, %3d 手, %s  (%.0f vs %.0f ms/步)\n",
                        gme + 1, "case", caseIsRed ? "红" : "黑", gr.plies,
                        win ? "胜" : (loss ? "负" : "和"), gr.msCase, gr.msRef);
        }
        c.reset();
    }

    /* ---- 4. 汇总 ---- */
    std::printf("\n[4] 汇总 (等时间预算 %.0f ms/步)\n", g_cfg.msBudget);
    std::printf("    %-20s %6s %6s %6s | %10s %10s | %8s\n",
                "参赛者", "胜", "和", "负", "case ms/步", "A ms/步", "平均手数");
    for (int i = 0; i < 4; i++) {
        if (!g_cfg.sel[i]) {
            continue;
        }
        const double mc = tally[i].games > 0 ? tally[i].msCase / (double)tally[i].games : 0.0;
        const double mr = tally[i].games > 0 ? tally[i].msRef / (double)tally[i].games : 0.0;
        const double pl = tally[i].games > 0 ? (double)tally[i].plies / (double)tally[i].games : 0.0;
        std::printf("    %-20s %6d %6d %6d | %10.1f %10.1f | %8.1f%s\n",
                    kCases[i].tag, tally[i].win, tally[i].draw, tally[i].loss,
                    mc, mr, pl, tally[i].aborted > 0 ? "  <有异常>" : "");
    }
    std::printf("    注: 第一行 (A vs A) 是自我对照 —— 两边同结构同权重初始化, 只有随机\n"
                "        开局和先后手不同, 所以它的偏离就是本样本量下的噪声底 (每对只有\n"
                "        %d 局)。B/C/D 相对 A 的胜负要明显超出这个噪声才有意义。\n",
                g_cfg.games);

    /* ---- 5. 稀疏 MoE 的坍缩诊断 ---- */
    std::printf("\n[5] 稀疏 MoE 路由分布 (对局期间 actor 的专家使用计数)\n");
    for (int i = 1; i < 4; i++) {
        if (!g_cfg.sel[i]) {
            continue;
        }
        std::vector<long long> usage;
        agents[i]->moeUsage(usage);
        if (usage.empty()) {
            continue;
        }
        long long tot = 0, mx = 0;
        double mean = 0.0;
        for (std::size_t k = 0; k < usage.size(); k++) {
            tot += usage[k];
            mx = std::max(mx, usage[k]);
            mean += (double)usage[k];
        }
        mean /= (double)usage.size();
        double var = 0.0;
        for (std::size_t k = 0; k < usage.size(); k++) {
            var += ((double)usage[k] - mean) * ((double)usage[k] - mean);
        }
        var /= (double)usage.size();
        std::printf("    %s 计数 = {", kCases[i].tag);
        for (std::size_t k = 0; k < usage.size(); k++) {
            std::printf("%lld%s", usage[k], (k + 1 < usage.size()) ? ", " : "}");
        }
        std::printf("  总计 %lld, 最大/均值 = %.2f, 变异系数 = %.2f\n",
                    tot, mean > 0 ? (double)mx / mean : 0.0,
                    mean > 0 ? std::sqrt(var) / mean : 0.0);
    }
    std::printf("    (负载均衡辅助损失系数 = %.3g; 它只在 learnBatch 里起作用, 所以\n"
                "     预训练 0 局时这里反映的是**随机初始门控**的偏好, 不是坍缩。\n"
                "     系数的影响见 --pretrain 下的\"训练期专家使用\"一行:\n"
                "     aux=0 时 8 个专家里有 3 个一次都没被选中, 见 docs/agents_design.md §11.4.2)\n",
                (double)g_cfg.aux);

    for (int i = 0; i < 4; i++) {
        delete agents[i];
    }
    return 0;
}
