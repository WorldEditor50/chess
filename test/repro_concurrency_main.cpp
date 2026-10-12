/*
 * repro_concurrency_main.cpp - "对弈 x 后台训练" 并发崩溃的最小复现 (调试用)
 *
 * 2026-09 用户报障: "对弈时自动保存权重的时候导致程序崩溃了"。
 * Windows 事件日志里的异常码是 **0xC0000374 (STATUS_HEAP_CORRUPTION, ntdll)** ——
 * 堆损坏, 也就是"有人写越界 / 用了已释放的内存", 而不是某个函数返回了错误。
 *
 * 缩小范围的过程 (见 docs/issues_review.md 零之二点二十五):
 *   1. 真界面里逐项排除: "对战AI(后台训练的目标) == 对弈的某一方" **必须成立**才会崩;
 *      训练目标与参赛者不同、或根本不训练时都不崩。**用 EVAB 也一样崩** ——
 *      所以这跟新加的 PPO+MCTS-MLP 无关, 是一个更早就存在的问题。
 *   2. 把同样的形状搬到无界面的测试里 (本文件): 对弈跑在**另一个线程**,
 *      主线程同时按"每手一次"的频率调 getAgentSelfCheck() (界面那个自检 worker 干的事),
 *      后台训练开着 —— 于是不需要界面就能稳定复现 (test_match 里那些"对弈在主线
 *      程上跑"的用例复现不出来, 这一点本身就是线索)。
 *
 * 这个文件**故意不是一个 ctest**: 它复现的是一类时序相关的堆损坏, 跑得快不快、
 * 崩不崩取决于调度; 它的用途是"改完并发相关的代码之后, 在这里压一下"。
 *
 * 用法:
 *   repro_concurrency.exe [轮数] [每局手数] [A 方 agent 序号] [B 方 agent 序号]
 *   :: ASan 版本能得到准确的越界/释放栈 (见 docs/issues_review.md 的复现命令)
 *
 * agent 序号 = ChessBoard::AgentType 的值 (0 Alpha-Beta, 1 MCTS, 2 PG, 3 DQN,
 * 4 PPO+MCTS, 5 PPO+MCTS-MLP, 6 DQN+MCTS, 7 EVAB, 8 SAC+AZ, 9 SAC+AZ-MoE, 10 DQN+AB,
 * …, 18 PPO+MCTS 卷积残差, 19 混合对手池)。
 *
 * ---- [2026-10 第四轮] 三个新增用法 (都是"B 方混合 agent"那一轮的验证路径) ----
 *   repro_concurrency 6 4 19      对局侧的混合对手池当 A 方: 报告里会多出
 *                                 "混合对手池: 出场 N 局 (逐对手): MCTS(800)/AB-L* 各行",
 *                                 并且相邻两轮用**不同**的 MCTS 占比 (0.5 / 0) 以证明
 *                                 池子是**每场对弈重算**的而不是 static 缓存。
 *   repro_concurrency 1 4 19 19   两边都是池子: 必须**各自独立抽一次** (逐对手那两行带
 *                                 "A 方: " / "B 方: " 前缀), 并在评估模式下印出"冻结口径"
 *                                 那一段 (池子抽出的 MCTS/AB 都没有可训练权重 ⇒ 不需要快照)。
 *   ⚠ 只抽一次而只替换 A 方的写法会让 B 方带着 AGENT_MIXED_POOL 走进决策 switch
 *     (那里没有这一档) ⇒ 退化成"每手走第一个合法走法"的**假对局**, 而报告看上去正常 ——
 *     所以这一条必须有"真对局"的路径可跑 (断言在 test_mixed_opponents [7] + 采样件本身)。
 */
#include <QApplication>
#include <QStringList>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include "chessboard.h"

int main(int argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    const int rounds = (argc > 1) ? std::atoi(argv[1]) : 3;
    const int plies = (argc > 2) ? std::atoi(argv[2]) : 12;
    const int agentIdx = (argc > 3) ? std::atoi(argv[3]) : (int)ChessBoard::AGENT_PPOMCTS_MLP;
    const ChessBoard::AgentType type = (ChessBoard::AgentType)agentIdx;
    /*
       [2026-10 第四轮] 可选的第 4 个参数 = B 方的 agent 序号 (默认 Alpha-Beta)。
       加它的唯一原因: 混合对手池的**两边都是池子**这一支必须有一条真对局的验证路径
       (`repro_concurrency 1 4 19 19`) —— 只抽一次而只替换 A 方的写法会让 B 方带着
       `AGENT_MIXED_POOL` 走进决策 switch (那里没有这一档), 退化成"每手走第一个合法走法"
       的假对局, 而报告看上去正常。
    */
    const int agentBIdx = (argc > 4) ? std::atoi(argv[4]) : (int)ChessBoard::AGENT_ALPHABETA;
    const ChessBoard::AgentType typeB = (ChessBoard::AgentType)agentBIdx;

    std::printf("=== 对弈 x 后台训练 并发复现 ===\n");
    std::printf("[build] %s %s | agent=%d 轮数=%d 每局手数=%d\n",
                __DATE__, __TIME__, agentIdx, rounds, plies);
    std::printf("[build] B 方 agent=%d\n", agentBIdx);

    ChessBoard board;
    /*
       [2026-10 第四轮] 混合对手池 (agent 序号 19 = AGENT_MIXED_POOL) 走的是**对局**路径,
       它的读数只在报告里 (st.poolStats), 所以这里把报告接出来打一遍 —— 这就是
       "对局侧的池子真的按对手分组记账了"的直接证据 (见 docs §7)。
       其它 agent 只打一行比分。
    */
    QObject::connect(&board, &ChessBoard::matchFinished,
                     [agentIdx, agentBIdx](const QString &summary, const QString &detail) {
        std::printf("  [报告] %s\n", summary.toUtf8().constData());
        if (agentIdx == (int)ChessBoard::AGENT_MIXED_POOL
            || agentBIdx == (int)ChessBoard::AGENT_MIXED_POOL) {
            /*
               打三类行: 标题 ("混合对手池: 出场 N 局")、逐对手那一行
               (形如 "MCTS(800): 3 局  2 胜 / 1 负 / 0 和"; 两边都是池子时名字带
               "A 方: " / "B 方: " 前缀, 否则同名对手会把两方的胜负混进同一行)、
               以及那句"总分是混合比分"的警告。
            */
            const QStringList lines = detail.split(QLatin1Char('\n'));
            for (const QString &ln : lines) {
                const QString s = ln.trimmed();
                if (ln.contains(QStringLiteral("混合对手池")) || s.contains(QStringLiteral(" 局"))
                    || s.contains(QStringLiteral("冻结")) || s.startsWith(QStringLiteral("MCTS"))
                    || s.startsWith(QStringLiteral("AB-L"))) {
                    std::printf("  [池子] %s\n", s.toUtf8().constData());
                }
            }
        }
    });
    /* 界面里的设置: 走子前探索 + 预训练 (默认 64 步) 也照旧打开 */
    std::printf("[ok] ChessBoard 构造完成\n");
    /*
       [2026-10 第四轮] 混合对手池没有权重/实例, 但它是下拉框里选得到的一档 —— 所以自检
       面板必须**说明它是什么**, 而不是显示"没有自检项"(那会被读成"还没接线")。这一段
       把那份报告打出来当证据 (数据源与 GUI 面板同一个: ChessBoard::getAgentSelfCheck)。
    */
    if (type == ChessBoard::AGENT_MIXED_POOL || typeB == ChessBoard::AGENT_MIXED_POOL) {
        std::printf("---- 混合对手池的自检报告 ----\n%s\n",
                    board.getAgentSelfCheck(ChessBoard::AGENT_MIXED_POOL).c_str());
    }

    /* ---- 关键: 后台训练的目标 == 对弈的 A 方 (同一个实例) ---- */
    board.setAgentType(type);
    /*
       [2026-10 第四轮] 两边都是混合对手池时把模式拨到**评估**: 池子抽出来的每一档都没有
       可训练权重, 所以评估模式下"权重冻结"这件事对它是自动成立的 —— 而这一段是
       `st.frozenNote` 里需要专门说清的一支 (不拨的话默认训练模式先命中, 那一条分支
       永远走不到)。
    */
    if (type == ChessBoard::AGENT_MIXED_POOL && typeB == ChessBoard::AGENT_MIXED_POOL) {
        board.setMatchMode(ChessBoard::MATCH_EVAL);
        std::printf("[ok] 双方都是池子 ⇒ 对弈模式设为评估 (查'冻结口径'那一段)\n");
    }
    board.setBackgroundTrainRound(1, 40);
    board.startBackgroundTraining();
    board.setMaxPliesPerGame(plies);
    std::printf("[ok] 后台训练已启动 (目标 = agent %d), 对弈每局 %d 手\n", agentIdx, plies);

    for (int g = 0; g < rounds; g++) {
        /*
           [2026-10 第四轮] 每一场对弈**重算**池子权重 (不是函数内 static 缓存一次)。
           奇数轮把 MCTS 占比压到 0 ⇒ 这一场只会抽到 AB-L1..L4 —— 报告里逐对手那一栏
           因此会证明"改配置之后下一场真的换了分布"。如果哪天有人把池子改成 static
           缓存, 这条输出会立刻变成"还是 MCTS 在出场", 而不是静默沿用旧权重。
        */
        if (agentIdx == (int)ChessBoard::AGENT_MIXED_POOL) {
            const float share = (g % 2 == 1) ? 0.0f : 0.5f;
            board.setMixedOpponentConfig(share, 800, 4);
            std::printf("  第 %d 轮: 池子设置 MCTS 占比 = %.2f\n", g + 1, (double)share);
        }
        /* ---- 对弈跑在独立线程 (MainWindow 用的是 m_selfPlayThread) ---- */
        std::atomic<bool> done{false};
        std::thread matchThread([&]() {
            board.matchAgents(type, typeB, 1);
            done = true;
        });

        /* ---- 主线程按"每手一次"的频率刷自检 (GUI 的自检 worker) ---- */
        long long checks = 0;
        while (!done.load()) {
            board.getAgentSelfCheck(type);
            checks++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        matchThread.join();

        /* ---- 对弈结束后立刻保存权重 (GUI 的"静默保存") ---- */
        const bool ok = board.saveCurrentAgentModel(type,
                                                    ChessBoard::defaultWeightPath(type));
        std::printf("  第 %d 轮: 完成 (并发自检 %lld 次, 保存=%d)\n", g + 1, checks, (int)ok);
    }

    board.stopBackgroundTraining();
    std::printf("=== 全部跑完, 没有崩溃 ===\n");
    return 0;
}
