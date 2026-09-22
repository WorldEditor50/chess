/*
 * test_match_main.cpp - Agent 对 Agent 对弈 (arena) 的验证
 *
 * 这个功能是"让两个 agent 打若干局, 用来比较强弱", 所以真正需要盯住的不是
 * "能不能下完", 而是几件统计上的事:
 *
 *   1. 每局**交换先后手**。中国象棋先手(红)优势很大, 固定谁执红的话, 结果
 *      只是在测"谁执红"而不是"谁更强"。日志里必须两局的执红方是相反的。
 *   2. 胜负按**参赛者 A/B** 归属, 不是按红黑。归错一边的话整个比分就没有意义。
 *   3. 达到手数上限判**和棋** (老代码在这里按静态评估判"红胜", 于是平局分支
 *      永远不可达)。
 *   4. 中止要在若干手之内生效, 且不能把半局算成完整一局。
 *
 * 做法上不去跑几百手的真对局: 把 setMaxPliesPerGame 调小 (这里是 4), 一局
 * 4 手必然到上限判和 —— 于是每局结果可预测, 断言就能做得很硬。
 */
#include <QApplication>
#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include "chessboard.h"
#include "abagent.h"      /* [2.10] "必输局面"下 ABAgent 必须仍返回合法走法 */
#include "dqnagent.h"
#include "sacazagent.h"   /* [2.19] 学习口径 == agent 自己的 computeReward/terminalReward */
#include "metricsview.h"
#include <cmath>
#include <algorithm>   /* std::count: 数自检报告有几行 */
#include <QVector>
#include <QMetaObject>

static int g_checks = 0;
static int g_failed = 0;
#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

static bool contains(const QString &hay, const QString &needle)
{
    return hay.contains(needle);
}

int main(int argc, char *argv[])
{
    /* 无缓冲: 崩溃时也能看到走到了哪一步 (printf 默认是行/块缓冲, 段错误会把它丢掉) */
    setvbuf(stdout, nullptr, _IONBF, 0);

    /* 无显示器也能跑: 强制用 offscreen 平台插件 */
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    std::printf("=== Agent 对 Agent 对弈 (arena) ===\n");
    std::printf("[build] %s %s\n", __DATE__, __TIME__);

    ChessBoard board;
    std::printf("[ok] ChessBoard 构造完成\n");
    /* 不探索: 这个测试关心的是对弈统计, 不是探索 (那是 test_pretrain 的事) */
    board.setPreTrainEnabled(false);
    board.setPreTrainSteps(0);

    /* ---------------------------------------------------------------- 1. 交换先后手 + 比分归属 */
    std::printf("\n[1] 2 局, 每局限 4 手 (必然判和), 检查先后手交换与统计\n");
    board.setMaxPliesPerGame(4);

    std::printf("[..] 调用 matchAgents(AlphaBeta, EVAB, 2)\n");
    ChessBoard::MatchStats st =
        board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_EVAB, 2);
    std::printf("[ok] matchAgents 返回\n");

    std::printf("    %s\n", st.summary().toUtf8().constData());
    std::printf("%s", st.detail().toUtf8().constData());
    std::printf("\n");

    CHECK(st.games == 2, "打完 2 局");
    CHECK(st.plies == 2 * 4, "总手数 = 局数 x 每局上限");
    CHECK(st.winA == 0 && st.winB == 0, "4 手之内不可能将杀, 不应有胜局");
    CHECK(st.draws == 2, "到上限判和 (不是判红胜)");
    CHECK(st.winA + st.winB + st.draws == st.games, "比分之和等于局数");
    CHECK(!st.aborted, "正常跑完不算被中止");
    CHECK(!board.isMatchRunning(), "跑完之后不再处于对弈中");

    /* 第 1 局 A 执红, 第 2 局 B 执红 —— 这一条是方法学的核心 */
    CHECK(contains(st.log, QStringLiteral("第 1 局: 红=Alpha-Beta 黑=EVAB")),
          "第 1 局 A(Alpha-Beta) 执红");
    CHECK(contains(st.log, QStringLiteral("第 2 局: 红=EVAB 黑=Alpha-Beta")),
          "第 2 局 B(EVAB) 执红 (先后手已交换)");

    /* ---------------------------------------------------------------- 2. 同一 agent 打两边 */
    std::printf("\n[2] 同一个 agent 打两边 (= 原来的 self-play)\n");
    ChessBoard::MatchStats self =
        board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_ALPHABETA, 1);
    std::printf("    %s\n", self.summary().toUtf8().constData());
    CHECK(self.games == 1, "自对弈也能作为对弈的一个特例跑完");
    CHECK(self.agentA == self.agentB, "两边是同一个 agent");

    /* ------------------------------------------------- 2.5 SAC+AZ 走一遍完整集成路径 */
    /* 这一步覆盖 ChessBoard::aiThinkForAgent 的 AGENT_SACAZ 分支 (含 preTrainThenDecide)
       与 SACAZAgent 的 MCTS 决策, 是"新 agent 真的能接入"的端到端证据。 */
    std::printf("\n[2.5] SAC+MCTS+AlphaZero 接入 arena (覆盖 GUI 同款代码路径)\n");
    ChessBoard::MatchStats saz = board.matchAgents(ChessBoard::AGENT_SACAZ,
                                                   ChessBoard::AGENT_ALPHABETA, 1);
    std::printf("    %s\n", saz.summary().toUtf8().constData());
    CHECK(saz.games == 1, "SAC+AZ 能打完一局");
    CHECK(saz.plies == 4, "手数正常 (没有被误判成无合法走法)");
    CHECK(saz.winA + saz.winB + saz.draws == 1, "比分归属正常");
    CHECK(saz.agentErrors == 0, "SAC+AZ 没有返回过无效走法");

    /* ------------------------------------------------- 2.6 即时奖励的符号约定 */
    /*
       这一节是一个**回归钉子**: 所有 agent 的 computeReward 原来都写成
       `(color == COLOR_BLACK) ? +v : -v` (黑方视角), 于是红方白吃一个黑子会拿到
       负奖励 —— 与终局奖励 (走子方视角的 ±1) 相反, 对红方等于在教它"吃子是坏事"。
       这里把"走子方视角"这个约定钉死: 谁吃子谁拿正奖励, 与颜色无关。
       (Chess::moveForward 的 totalReward 是另一套 (黑方视角) 记账, 一并钉住它的
       真实行为, 免得以后有人以为它是走子方视角。)
    */
    std::printf("\n[2.6] 即时奖励的符号约定 (走子方视角)\n");
    {
        Chess c;
        c.reset();
        Step cap;
        cap.valid = false;
        std::vector<Step *> legal;
        c.sample(Stone::COLOR_RED, legal);
        Steps::instance().put(legal);
        for (std::size_t i = 0; i < legal.size(); ++i) {
            if (legal[i]->nextId != Stone::ID_NONE) {
                cap = *legal[i];
                break;
            }
        }
        CHECK(cap.valid, "初始局面红方能找到一个吃子的走法");
        if (cap.valid) {
            Stone *victim = c.stones[cap.nextId];
            const double victimValue = victim->value;
            double totalReward = 0.0;
            c.moveForward(&cap, totalReward);
            std::printf("    红方吃 %s (value=%.2f): moveForward 记账 = %+.3f\n",
                        victim->name.c_str(), victimValue, totalReward);
            /*
               moveForward 的记账是**黑方视角**: 吃掉红子 +, 吃掉黑子 −。
               红方吃黑子 => totalReward 是负的 (红方视角的收益 = -totalReward > 0)。
            */
            CHECK(totalReward < 0.0, "Chess::moveForward 的记账是黑方视角 (吃黑子为负)");
            CHECK(-totalReward > 0.0, "换算到走子方 (红) 视角后是正收益");

            /* agent 侧: 谁吃子谁拿正奖励 */
            DQNAgent dqn(c, 64, 0.99f, 0.001f, 1.0f);
            const float rRed = dqn.computeReward(cap, Stone::COLOR_RED);
            std::printf("    computeReward(红) = %+.3f\n", (double)rRed);
            CHECK(rRed > 0.0f, "红方吃子 -> 红方拿**正**奖励 (走子方视角)");
        }

        /* 黑方吃红子: 也应该拿正奖励 */
        Chess c2;
        c2.reset();
        Step cap2;
        cap2.valid = false;
        std::vector<Step *> legal2;
        c2.sample(Stone::COLOR_BLACK, legal2);
        Steps::instance().put(legal2);
        for (std::size_t i = 0; i < legal2.size(); ++i) {
            if (legal2[i]->nextId != Stone::ID_NONE) {
                cap2 = *legal2[i];
                break;
            }
        }
        if (cap2.valid) {
            DQNAgent dqn2(c2, 64, 0.99f, 0.001f, 1.0f);
            const float rBlack = dqn2.computeReward(cap2, Stone::COLOR_BLACK);
            std::printf("    computeReward(黑) = %+.3f\n", (double)rBlack);
            CHECK(rBlack > 0.0f, "黑方吃子 -> 黑方拿**正**奖励 (与红方对称)");
        } else {
            std::printf("    (初始局面黑方没有可吃的子, 跳过对称性检查)\n");
        }
    }

    /* ------------------------------------------- 2.7 每个 agent 都要能上报训练损失 */
    /*
       用户的报障: "其他 agent 在对弈时无法显示训练损失"。损失曲线的数据源是
       ChessBoard::trainLossSample, 而它由 preTrainThenDecide() 在**每次在线训练之后**
       发出 —— 所以这里逐个 agent 跑一小局, 直接数信号, 不依赖界面。
       预期:
         上报: SAC+AZ / SAC+AZ-MoE / DQN / DQN+MCTS / PPO+MCTS / PG / EVAB
         不上报 (没有可训练参数, AgentBase::getLastTrainLoss 默认 NaN): Alpha-Beta / MCTS
    */
    std::printf("\n[2.7] 训练损失上报 (每个 agent 一小局, 数 trainLossSample 信号)\n");
    {
        struct Case {
            ChessBoard::AgentType type;
            const char *name;
            bool expectLoss;
        };
        const Case cases[] = {
            { ChessBoard::AGENT_ALPHABETA, "Alpha-Beta",   false },
            { ChessBoard::AGENT_MCTS,      "MCTS",         false },
            { ChessBoard::AGENT_PG,        "Policy Gradient", true },
            { ChessBoard::AGENT_DQN,       "DQN",          true },
            { ChessBoard::AGENT_PPOMCTS,   "PPO+MCTS",     true },
            { ChessBoard::AGENT_DQNMCTS,   "DQN+MCTS",     true },
            { ChessBoard::AGENT_EVAB,      "EVAB",         true },
            { ChessBoard::AGENT_SACAZ,     "SAC+AZ",       true },
            { ChessBoard::AGENT_SACAZ_MOE, "SAC+AZ-MoE",   true },
            /* 59e5233 行为还原版 (派生类 SACAZLegacyAgent): 显示名与"上报损失"都要接上 */
            { ChessBoard::AGENT_SACAZ_OLD, "SAC+AZ-59e5233", true },
            { ChessBoard::AGENT_PPOMCTS_MLP, "PPO+MCTS-MLP", true }
        };
        /*
           10 手 + 预训 32 步: 有几个 agent 的在线训练是**按回放池大小门控**的
           (SAC+AZ 的 learnBatch 在池 < batchSize 时故意不更新; DQN 的 learn()
           同样), 池没攒够之前"没训练" -> 不上报损失, 这是正确的行为而不是 bug。
           所以这里要给够步数, 让每个 agent 至少真的训练过一次。
        */
        board.setMaxPliesPerGame(10);
        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);

        for (const Case &cs : cases) {
            QVector<double> samples;
            QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&samples](double loss, const QString &agent, int step) {
                    (void)agent;
                    (void)step;
                    samples.append(loss);
                });
            const ChessBoard::MatchStats st =
                board.matchAgents(cs.type, ChessBoard::AGENT_ALPHABETA, 1);
            QObject::disconnect(conn);

            bool finite = true;
            for (double v : samples) {
                if (!std::isfinite(v)) {
                    finite = false;
                }
            }
            std::printf("    %-16s 信号 %d 次, 全部有限=%d, 局数=%d, 手数=%d\n",
                        cs.name, (int)samples.size(), (int)finite, st.games, st.plies);
            if (cs.expectLoss) {
                CHECK(!samples.isEmpty(), "该 agent 上报了训练损失 (曲线不再是空的)");
                CHECK(finite, "上报的损失都是有限值 (没有 NaN)");
            } else {
                CHECK(samples.isEmpty(), "没有可训练参数的 agent 不上报 (不画假线)");
            }
            CHECK(st.games == 1, "这一小局正常打完");
        }
        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
    }

    /* ------------------------------------------------- 2.8 对局过程中的奖励曲线 */
    /*
       用户反馈"对弈时奖励曲线没有更新"。原因不是信号断了, 而是**采样太稀**:
       gameRewardSample 一局只发一次, 而一局可能有几百手、跑十几分钟 (实测 276 手
       621 秒), 于是整局过程中曲线一动不动。现在 matchRewardProgress 每手发一次,
       带的是"本局到目前为止"累计的 A/B 环境奖励。

       这一节钉住三件事:
         (1) 每手都有一个进度点 (一局 12 手 -> 十几个点, 而不是 1 个);
         (2) 进度点按局分组, 组内手号从 1 开始连续;
         (3) 进度账与局末账是**同一本账**: 每局最后一个进度点 + 终局 ±1 == 局末那个点,
             且 A/B 的差恒为相反数 (谁赢谁 +1)。
    */
    std::printf("\n[2.8] 每手的奖励进度 (matchRewardProgress) 与局末奖励是同一本账\n");
    {
        board.setMaxPliesPerGame(12);
        struct GameTrace {
            QVector<int> plys;
            QVector<double> progA;
            QVector<double> progB;
            bool hasFinal = false;
            double finalA = 0.0;
            double finalB = 0.0;
        };
        QVector<GameTrace> games;
        const QMetaObject::Connection c1 = QObject::connect(
            &board, &ChessBoard::matchRewardProgress,
            [&games](int gameNo, int ply, double rewardA, double rewardB) {
                if (gameNo < 1) {
                    return;
                }
                while (games.size() < gameNo) {
                    games.append(GameTrace());
                }
                GameTrace &g = games[gameNo - 1];
                g.plys.append(ply);
                g.progA.append(rewardA);
                g.progB.append(rewardB);
            });
        const QMetaObject::Connection c2 = QObject::connect(
            &board, &ChessBoard::gameRewardSample,
            [&games](int gameNo, const QString &, const QString &,
                     double rewardA, double rewardB) {
                if (gameNo < 1 || gameNo > games.size()) {
                    return;
                }
                GameTrace &g = games[gameNo - 1];
                g.hasFinal = true;
                g.finalA = rewardA;
                g.finalB = rewardB;
            });
        const ChessBoard::MatchStats st2 =
            board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_ALPHABETA, 2);
        QObject::disconnect(c1);
        QObject::disconnect(c2);

        CHECK(games.size() == st2.games, "每局一组进度点");
        CHECK(st2.games == 2, "这一节跑完 2 局");
        int minPoints = -1;
        int deltaSum = 0;
        bool contiguous = true;
        bool sameBook = true;
        bool opposite = true;
        for (int i = 0; i < games.size(); ++i) {
            const GameTrace &g = games[i];
            std::printf("    第 %d 局: 进度点 %d 个, 最后一手累计 A=%+.2f B=%+.2f, "
                        "局末 A=%+.2f B=%+.2f\n",
                        i + 1, (int)g.progA.size(),
                        g.progA.isEmpty() ? 0.0 : g.progA.last(),
                        g.progB.isEmpty() ? 0.0 : g.progB.last(),
                        g.hasFinal ? g.finalA : 0.0, g.hasFinal ? g.finalB : 0.0);
            if (minPoints < 0 || g.progA.size() < minPoints) {
                minPoints = (int)g.progA.size();
            }
            /* 手号连续: 1,2,3,... (中途漏一手说明 emit 的位置漏了) */
            for (int k = 0; k < g.plys.size(); ++k) {
                if (g.plys[k] != k + 1) {
                    contiguous = false;
                }
            }
            CHECK(g.hasFinal, "这一局有局末奖励采样");
            if (g.hasFinal && !g.progA.isEmpty()) {
                const double dA = g.finalA - g.progA.last();
                const double dB = g.finalB - g.progB.last();
                /* 终局只加 ±1 (和棋 0), 所以差值只能是这三个值之一 */
                if (!(std::fabs(dA) < 1e-9 || std::fabs(dA - 1.0) < 1e-9
                      || std::fabs(dA + 1.0) < 1e-9)) {
                    sameBook = false;
                }
                if (!(std::fabs(dA + dB) < 1e-9)) {
                    opposite = false;
                }
                deltaSum += (int)std::lround(dA);
            }
        }
        /*
           一局 12 手 -> 进度点 11~12 个 (最后一手若是"被将死"那一次, 它不落子, 也就
           没有对应的进度点)。关键是**远大于 1**: 用户看到的"不更新"就是 1 个点。
        */
        std::printf("    每组最少进度点 = %d (局手数上限 12)\n", minPoints);
        CHECK(minPoints >= 11, "每局都有接近手数的进度点 (不是每局只 1 个点)");
        CHECK(contiguous, "进度点的手号是连续的");
        CHECK(sameBook, "最后一个进度点 + 终局 ±1 == 局末奖励 (同一本账)");
        CHECK(opposite, "A/B 的局末增量互为相反数 (一方 +1 另一方 -1)");
        CHECK(deltaSum == st2.winA - st2.winB,
              "两局的胜负增量之和 == 比分差 (A 胜局数 - B 胜局数)");
    }

    /* ------------------------------------------------- 2.9 奖励曲线的"换一批线" */
    /*
       用户在一次 100 局对弈里的读数标签实录:
         "奖励(局内累计) SAC+AZ-MoE: 暂无 | Alpha-Beta: 暂无 |
          SAC+AZ-MoE: 最新 0, ... 982 点 | Alpha-Beta: ... 982 点"
       四条线、两条永远"暂无" —— 因为"每场对弈重建两条奖励曲线"当时写成了
       `clearData()` (只清点、**不清线**) + `addSeries()` x2, 于是每跑一场就往图上
       多挂两条空线 (第三场 6 条、第十场 20 条), 导出的 CSV 也跟着多列。

       这一节把两个 API 的语义钉死, 并验证 MainWindow 现在用的那条路径 (removeAllSeries
       + addSeries x2) 真的只留两条线。CurveChart 是普通 QWidget, offscreen 下能直接构造。
    */
    std::printf("\n[2.9] 奖励曲线换一批线: clearData 只清点, removeAllSeries 才清线\n");
    {
        CurveChart chart;
        chart.addSeries(QStringLiteral("A"), QColor(Qt::red));
        chart.addSeries(QStringLiteral("B"), QColor(Qt::blue));
        chart.addPoint(0, 1.0);
        chart.addPoint(1, -1.0);
        CHECK(chart.seriesCount() == 2, "两条线");
        /* 注意 sampleCount() 是**各条线里最多的点数** (CSV 按它出行数), 不是总和 ——
           第一版这里写成 == 2, 于是它正确地失败了。 */
        CHECK(chart.sampleCount() == 1, "sampleCount 是'最多点数' = 1 (不是两条相加)");

        /* clearData: 点没了, 线还在 (放大窗口靠它同步, 语义要保持) */
        chart.clearData();
        CHECK(chart.seriesCount() == 2, "clearData 之后线还在 (它只清点)");
        CHECK(chart.sampleCount() == 0, "clearData 之后点没了");

        /* 这正是 bug 的形状: 只 clearData 再加两条 -> 四条 (两条空的) */
        chart.addSeries(QStringLiteral("A"), QColor(Qt::red));
        chart.addSeries(QStringLiteral("B"), QColor(Qt::blue));
        CHECK(chart.seriesCount() == 4, "旧的写法 (clearData+addSeries) 会挂成 4 条 -> 这就是那个 bug");

        /* 正确路径: 每场对弈开始 = removeAllSeries + 两条新线 */
        chart.removeAllSeries();
        CHECK(chart.seriesCount() == 0, "removeAllSeries 把线全清掉");
        chart.addSeries(QStringLiteral("C"), QColor(Qt::green));
        chart.addSeries(QStringLiteral("D"), QColor(Qt::darkYellow));
        chart.addPoint(0, 2.0);
        chart.addPoint(1, 3.0);
        CHECK(chart.seriesCount() == 2, "换一批线之后仍然只有两条 (不再每场多挂两条)");

        /* 读数里每个名字只应出现一次 (重复的线会带来重复的同名读数) */
        const QString readout = chart.readoutText(QStringLiteral("reward"));
        std::printf("    readout = %s\n", readout.toUtf8().constData());
        CHECK(readout.count(QStringLiteral("C:")) <= 1, "读数里同一个名字只出现一次");
        CHECK(readout.count(QStringLiteral("D:")) <= 1, "读数里同一个名字只出现一次 (第二条)");
        CHECK(!readout.contains(QStringLiteral("暂无")), "没有残留的空线 (不再出现'暂无')");

        /*
           ---- CSV 文本 (2026-09: 界面上多了"导出损失曲线"按钮) ----
           导出按钮本身要弹文件对话框 (自动化点不动), 所以把**格式**放在 CurveChart::toCsv()
           里, 在这里断言。要钉的三条:
             (1) 有注释行 + 表头列出所有曲线名 (否则导出文件读起来不知道哪列是谁);
             (2) 行数 = sampleCount (各条线里最多的点数);
             (3) **短的那条线后面的列留空, 不补 0** —— 补 0 会被读成"那个采样点上
                 损失是 0", 而真相是"它还没训练到那么多次"。这条最容易写错。
        */
        chart.clearData();
        chart.addPoint(0, 1.5);
        chart.addPoint(0, 2.5);
        chart.addPoint(0, 3.5);
        chart.addPoint(1, -0.5);
        const QString csv = chart.toCsv(QStringLiteral("测试注释"));
        const QStringList lines = csv.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        std::printf("    toCsv = %s\n", QString(csv).replace(QLatin1Char('\n'),
                                                              QLatin1String(" | "))
                                             .toUtf8().constData());
        CHECK(lines.size() == 5, "CSV = 1 注释行 + 1 表头 + 3 数据行 (按最多的点数)");
        CHECK(lines.size() > 0 && lines[0] == QStringLiteral("# 测试注释"),
              "第一行是注释行 (说明这一段是什么口径)");
        CHECK(lines.size() > 1 && lines[1] == QStringLiteral("sample,C,D"),
              "表头列出 sample 与所有曲线名");
        CHECK(lines.size() > 2 && lines[2] == QStringLiteral("1,1.5,-0.5"), "第 1 行数据");
        CHECK(lines.size() > 4 && lines[4] == QStringLiteral("3,3.5,"),
              "**短的那条线末列留空**(不补 0): 第 3 行只有 C 的值");

        /*
           ---- 窗口 vs 导出 (2026-09 用户踩到的坑) ----
           屏幕上的曲线受 setWindow 限制 (默认 2000 点, 为了画得动), 而**导出**必须是
           全部采样点: 用户拿导出的 CSV 当"整场 100 局"来分析, 而文件里只有最后 2000 个点
           —— 早期那个损失尖峰根本不在文件里, 结论就会错。这里用一个小窗口把这条规则钉住。
        */
        chart.setWindow(4);
        chart.clearData();
        for (int i = 0; i < 10; i++) { chart.addPoint(0, (double)i); }
        CHECK(chart.sampleCount() == 4, "屏幕窗口只保留最近 4 个点 (画图/读数口径)");
        CHECK(chart.historyCount() == 10, "**history 保留全部 10 个点** (导出/分析口径)");
        const QStringList csvLines2 = chart.toCsv(QStringLiteral("窗口测试"))
                                          .split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        CHECK(csvLines2.size() == 12, "导出 = 1 注释 + 1 表头 + **10** 行数据 (不受窗口影响)");
        CHECK(csvLines2.size() > 2 && csvLines2[2] == QStringLiteral("1,0,"),
              "导出从**第 1 个**采样点开始 (窗口已经挤掉的点仍在文件里; 末列空 = D 没有数据)");
        CHECK(csvLines2.size() > 11 && csvLines2[11] == QStringLiteral("10,9,"),
              "导出到最后一个采样点");
        chart.setWindow(2000);
    }

    /* ------------------------------------------- 2.10 "必输局面"必须仍返回合法走法 */
    /*
       用户报障的原文:
         [arena] agent(0) 返回无效走法 (第 6 局第 55 手): valid=0 id=0 pos=(0,0)->(0,0),
                 仍有 1 个合法走法, 已兜底
       agent(0) 就是 Alpha-Beta (按枚举顺序), 而返回的那个 Step 是**默认构造**的
       (id=0, pos=(0,0), valid=false)。根因在 ABAgent::findBestMove: 根节点的最优
       走法用严格不等号更新, 而它的初值是 ±value_infi —— 于是"每一步都必输"时
       (每个走法的子树都返回 ±value_infi) 一次都不会更新, best 保持 nullptr, 函数
       返回默认 Step, 调用方读成"这一步真无棋可走" -> 判负。

       这一节摆一个**必然触发**那个形状的局面:
         红: 帅 (9,4), 车 (7,1)
         黑: 将 (0,3), 车 (9,0), 车 (8,0)
       红正被 (9,0) 的车沿第 9 行将着; 帅的其它三个格子都被控制/占住
       ((8,4) 被 (8,0) 的车看着, (9,3)/(9,5) 都在第 9 行上), 所以唯一的合法走法
       是车(7,1)->(9,1) 垫将; 垫上之后黑方 车 x(9,1) 就是杀。
       也就是"红方怎么走都输" —— 正是那个"best 永远不被赋值"的条件。
       (下面同时把这个前提本身钉住: 只有一个合法走法 + 黑方确实有杀。)
    */
    std::printf("\n[2.10] 必输局面下 ABAgent 仍要返回合法走法 (arena 报障的回归钉)\n");
    {
        Chess c;
        c.reset();
        for (int i = 0; i < 32; i++) {
            c.m_children[i]->alive = false;
        }
        c.m_map.clear();
        auto place = [&c](int id, int x, int y) {
            Stone *s = c.m_children[id];
            s->alive = true;
            s->pos = Pos(x, y);
            c.m_map[Pos(x, y)] = s;
        };
        place(Stone::ID_RED_JIANG, 9, 4);
        place(Stone::ID_RED_CHE1, 7, 1);
        place(Stone::ID_BLACK_JIANG, 0, 3);
        place(Stone::ID_BLACK_CHE1, 9, 0);
        place(Stone::ID_BLACK_CHE2, 8, 0);
        c.sideToMove = Stone::COLOR_RED;
        c.halfMoveClock = 0;

        std::vector<Step *> legal;
        c.sample(Stone::COLOR_RED, legal);
        const std::size_t legalCount = legal.size();
        Step only;
        if (legalCount == 1) {
            only = *legal[0];
        }
        Steps::instance().put(legal);

        std::printf("    局面: 红方合法走法 %zu 个, getResult=%d (0=未终局)\n",
                    legalCount, c.getResult(Stone::COLOR_RED));
        CHECK(legalCount == 1, "局面构造正确: 红方只有一个合法走法 (只能垫将)");
        CHECK(c.getResult(Stone::COLOR_RED) == Chess::RESULT_ONGOING,
              "红方还没被判负 (所以这不是'真的无棋可走')");

        /* 前提之二: 垫将之后黑方立刻有杀 -> 这个局面确实"红方怎么走都输" */
        bool blackHasMate = false;
        if (legalCount == 1) {
            double r1 = 0;
            c.moveForward(&only, r1);
            std::vector<Step *> replies;
            c.sample(Stone::COLOR_BLACK, replies);
            for (Step *s : replies) {
                double r2 = 0;
                c.moveForward(s, r2);
                if (c.getResult(Stone::COLOR_RED) == Chess::RESULT_BLACK_WIN) {
                    blackHasMate = true;
                }
                c.moveBack(s, r2);
            }
            Steps::instance().put(replies);
            c.moveBack(&only, r1);
        }
        CHECK(blackHasMate, "这个局面确实是'红方怎么走都输' (垫将后黑方有杀)");

        /* 正题: 必输局面下必须返回**合法**走法 */
        ABAgent ab(c, 4);
        const Step mv = ab.getBestMove(Stone::COLOR_RED);
        std::printf("    ABAgent(depth=4) 返回: valid=%d id=%d pos=(%d,%d)->(%d,%d)\n",
                    (int)mv.valid, mv.id, mv.pos.x, mv.pos.y, mv.nextPos.x, mv.nextPos.y);
        CHECK(mv.valid, "必输局面下仍返回**合法**走法 (不再是 valid=false 的默认 Step)");
        if (mv.valid && legalCount == 1) {
            CHECK(mv.pos == only.pos && mv.nextPos == only.nextPos,
                  "返回的就是那个唯一的合法走法 (车垫将)");
        }
        /* 棋盘不能被动过: 这是"搜索在副本/回退上做"的契约 */
        std::vector<Step *> again;
        c.sample(Stone::COLOR_RED, again);
        CHECK(again.size() == legalCount, "搜索之后局面没被改动 (仍只有那一个合法走法)");
        Steps::instance().put(again);
    }

    /* ------------------------------------------------- 2.11 十个 agent 都有自检报告 */
    /*
       "给所有模型都配上自检方法"这条要求的回归钉:
         * 每个 agent 都要能给出非空报告;
         * 报告必须**可重复** (同一局面下两次调用逐字节相同 —— 自检的契约是只读、
          可重复调用, 它会在 GUI 线程被反复调用);
         * 报告里必须写明"这不是棋力"的口径 (否则 0 会被读成"没训练过")。
       Alpha-Beta / MCTS 没有常驻实例 (每一步现场构造一个), 所以它们能报出来本身
       就是"现场造一个 + 棋盘用副本"这条路径通了。
    */
    std::printf("\n[2.11] 模型自检 (十个 agent)\n");
    {
        const ChessBoard::AgentType all[] = {
            ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_MCTS,
            ChessBoard::AGENT_PG,        ChessBoard::AGENT_DQN,
            ChessBoard::AGENT_PPOMCTS,   ChessBoard::AGENT_DQNMCTS,
            ChessBoard::AGENT_EVAB,      ChessBoard::AGENT_SACAZ,
            ChessBoard::AGENT_SACAZ_MOE, ChessBoard::AGENT_DQNAB,
            ChessBoard::AGENT_PPOMCTS_MLP,
            /* 59e5233 行为还原版 (派生类 SACAZLegacyAgent), 2026-09 新增 */
            ChessBoard::AGENT_SACAZ_OLD
        };
        int reported = 0;
        for (ChessBoard::AgentType t : all) {
            const std::string a = board.getAgentSelfCheck(t);
            const std::string b = board.getAgentSelfCheck(t);
            const std::string w = board.getAgentWeightStatus(t);
            const int lines = (int)std::count(a.begin(), a.end(), '\n');
            std::printf("    type=%2d 报告 %2d 行 / %4d 字节, 权重状态 %2d 行, 可重复=%d\n",
                        (int)t, lines, (int)a.size(),
                        (int)std::count(w.begin(), w.end(), '\n'), (int)(a == b));
            if (a.empty()) {
                continue;   /* 该 agent 还没有实例 (权重文件没扫到) */
            }
            reported++;
            CHECK(a == b, "自检可重复调用 (两次结果逐字节相同)");
            CHECK(a.find("不是棋力") != std::string::npos,
                  "报告里写明'这不是棋力'的口径");
        }

        /*
           ---- 用户口径: **新旧 SAC 的权重文件必须用不同名字** (2026-09) ----
           判据取"真正会被写出的文件名" (getAgentWeightStatus 列的就是 weightFilesOf 的
           结果), 而不是前缀字符串 —— 名字漂移在本仓库真的发生过: PPO+MCTS 的扫描表探测
           裸文件名、而它写出的是 _actor/_critic, 于是那 279 MB x 2 的权重**从来没被载入
           过** (见 chessboard.cpp 的 weightFilesOf 注释)。
           这一条同时也是"两个 SAC 是分离的两支"在**权重层面**的证据: 它们的参数结构
           完全相同, 结构指纹挡不住串权重, 所以隔离只能靠文件名 + 类。
        */
        {
            const std::string oldW = board.getAgentWeightStatus(ChessBoard::AGENT_SACAZ_OLD);
            const std::string curW = board.getAgentWeightStatus(ChessBoard::AGENT_SACAZ);
            std::printf("    SAC+AZ (当前) 权重清单:\n%s", curW.c_str());
            std::printf("    SAC+AZ (59e5233) 权重清单:\n%s", oldW.c_str());
            CHECK(!oldW.empty() && !curW.empty(), "两个 SAC 的权重文件清单都报得出来");
            CHECK(oldW.find("sacaz_old_agent") != std::string::npos,
                  "59e5233 版的权重文件名带 sacaz_old_agent 前缀");
            CHECK(curW.find("sacaz_old_agent") == std::string::npos,
                  "当前 SAC 的权重文件名**不**带这个前缀 (没有共用)");
            CHECK(oldW != curW, "两支列出的文件名清单不同 (不会互相覆盖)");
        }
        /* 这两个是"现场造实例"那条路径: 它们**一定**能报 */
        CHECK(!board.getAgentSelfCheck(ChessBoard::AGENT_ALPHABETA).empty(),
              "Alpha-Beta 有自检报告 (无常驻实例, 现场构造)");
        CHECK(!board.getAgentSelfCheck(ChessBoard::AGENT_MCTS).empty(),
              "MCTS 有自检报告 (无常驻实例, 现场构造)");
        /*
           把 Alpha-Beta 的整份报告原样打出来: 断言只查"非空 / 可重复 / 口径",
           而这份报告是**回归指示器**(决策合法性自检)的载体 —— 人工读的时候要一眼看全,
           不必去翻源码里的格式串。它也是"必输局面"那个 bug 的验收读数 (见 [2.10])。
        */
        std::printf("    --- Alpha-Beta 自检报告 (原文) ---\n%s",
                    board.getAgentSelfCheck(ChessBoard::AGENT_ALPHABETA).c_str());
        std::printf("    --- EVAB 权重状态 (原文; '启动时未载入' 是因为本测试没跑 startupLoad) ---\n%s",
                    board.getAgentWeightStatus(ChessBoard::AGENT_EVAB).c_str());
        /* 前面几节已经跑过它们的一小局, 实例已存在 */
        CHECK(!board.getAgentSelfCheck(ChessBoard::AGENT_EVAB).empty(), "EVAB 有自检报告");
        CHECK(!board.getAgentSelfCheck(ChessBoard::AGENT_SACAZ).empty(), "SAC+AZ 有自检报告");
        std::printf("    能报自检的 agent: %d / %d (其余没有实例)\n",
                    reported, (int)(sizeof(all) / sizeof(all[0])));
        CHECK(reported >= 4, "至少 AB / MCTS / EVAB / SAC+AZ 这四个报了自检");
    }

    /* ------------------------------------------- 2.12 EVAB 的后台训练真的跑起来了 */
    /*
       用户报障的第二条:
         [train] 种子权重写入失败, 跳过本轮训练: agent "EVAB" 路径 weights/_temp_train.dat
       根因不是"写盘失败", 而是 `backgroundTrainLoop()` 里三个 switch (建实例 / 写种子 /
       训 clone / 同步回主 agent) **只有 PG、DQN、PPO+MCTS、DQN+MCTS 四个 case** —— 选
       EVAB 时 `seeded` 恒为 false, 于是那条为"写盘失败"写的消息被用来描述"这一支没写",
       用户看到的排查方向完全是错的, 而且 EVAB 的后台训练一直是空转。

       这一节直接盯**观测得到的后果** (不解析日志): 把当前 agent 切成 EVAB, 启动后台训练,
       等一轮训练上报损失。上报损失说明这一轮真的走完了
       "写种子 -> clone 载入 -> trainSelfPlay -> 写回 -> 同步回主 agent" 整条链
       (roundApplied 为 false 时根本不会 emit) —— 而这正是报障时不可能发生的事。
       一轮的规模是 BG_TRAIN_EPISODES=1 局 x 60 手 x (playDepth 3 + labelDepth 4),
       实测约 15~20 s, 所以这里的等待窗口给到 90 s。
    */
    std::printf("\n[2.12] EVAB 后台训练跑完一轮 (报障的回归钉)\n");
    {
        board.setAgentType(ChessBoard::AGENT_EVAB);
        QVector<double> evabLosses;
        const QMetaObject::Connection conn = QObject::connect(
            &board, &ChessBoard::trainLossSample,
            [&evabLosses](double loss, const QString &agent, int step) {
                (void)step;
                if (agent.contains(QStringLiteral("EVAB"))) {
                    evabLosses.append(loss);
                }
            });
        board.startBackgroundTraining();
        const auto t0 = std::chrono::steady_clock::now();
        while (evabLosses.isEmpty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const long long waited = std::chrono::duration_cast<std::chrono::seconds>(
                                         std::chrono::steady_clock::now() - t0).count();
            if (waited > 90) {
                break;
            }
            /* 线程在跑就说明它把主 agent 建起来了 —— 顺手确认一次 (这一支以前根本走不到) */
            if (!board.hasAgentInstance(ChessBoard::AGENT_EVAB)) {
                continue;
            }
        }
        board.stopBackgroundTraining();
        QObject::disconnect(conn);

        bool finite = true;
        for (double v : evabLosses) {
            if (!std::isfinite(v)) {
                finite = false;
            }
        }
        std::printf("    EVAB 后台训练上报损失 %d 次, 全部有限=%d\n",
                    (int)evabLosses.size(), (int)finite);
        CHECK(board.hasAgentInstance(ChessBoard::AGENT_EVAB),
              "后台训练把 EVAB 主 agent 建起来了 (以前这一支根本不存在)");
        CHECK(!evabLosses.isEmpty(),
              "EVAB 后台训练跑完了一轮并上报损失 (整条 写种子->训练->写回->同步 走通)");
        CHECK(finite, "上报的损失是有限值");
        /* 恢复: [3] 与界面默认都用 Alpha-Beta */
        board.setAgentType(ChessBoard::AGENT_ALPHABETA);
    }

    /* ------------------------------------------- 2.13 三个新接入的后台训练支路 */
    /*
       [2.12] 覆盖的是 EVAB。这一节把 **SAC+AZ / SAC+AZ-MoE / DQN+AB** 三个"2026-09
       才补齐"的支路也逐个走一遍 —— 理由就是 EVAB 那个 bug 的教训: 一条**没人走过的
       分支**等于没有接线, 而且它不会报错, 只会空转 (报障原文:
       "[train] 种子权重写入失败, 跳过本轮训练: agent \"EVAB\"" 其实是"这一支没写")。

       判据与 [2.12] 完全一样 (等一次 trainLossSample —— 只有 roundApplied 为真才
       emit, 也就是"写种子 -> clone 载入 -> 训练 -> 写回 -> 同步"整条链都成功),
       但这里用 setBackgroundTrainRound 把一轮缩到 **1 局 x 40 手**:
         * 默认的 1 局 x 60 手在这三个 agent 上是几十秒到几分钟
           (SAC+AZ-MoE 一次模拟 ~10.9 ms, DQN+AB 的主干是 37.5 M 参数的稀疏 MoE,
            临时文件的往返本身就是 ~276 MB x 2 次读 + 2 次写), 而这一节要验证的是
           "**接线**通不通", 不是"训得好不好";
         * 但**不能缩到几手**: 这三个 agent 的 learnBatch 都有"回放池 < batchSize(32)
           就直接返回、连 m_lastLoss 都不写"的门控 (SAC+AZ 见 sacazagent.cpp:857),
           于是损失曲线不上报 —— 第一版这里只给 6 手, 三个 agent 全部"上报 0 次",
           断言如实失败 (那不是接线坏了, 是根本没走到一次梯度更新)。
           40 手 > 32 且能被 learnEveryMoves=4 整除, 所以至少有一次真实更新。
       跑完把规模恢复成默认值 (后面的用例与界面都吃默认值)。
    */
    std::printf("\n[2.13] 三个新接入的后台训练支路 (SAC+AZ / SAC+AZ-MoE / DQN+AB)\n");
    {
        struct Case {
            ChessBoard::AgentType type;
            const char *name;
        };
        const Case cases[] = {
            { ChessBoard::AGENT_SACAZ,     "SAC+AZ" },
            { ChessBoard::AGENT_SACAZ_MOE, "SAC+AZ-MoE" },
            /*
               59e5233 行为还原版 (派生类 SACAZLegacyAgent)。这一条特别值得跑: 它的
               **训练 clone 是另一个类**, 而两支的参数结构完全相同 ⇒ 如果后台训练那一支
               忘了建派生类, save/load 一样成功、损失一样上报, 界面上**看不出任何异常**
               (只是按另一套口径在训)。这条断言挡的是"跑得通但训错了"。
            */
            { ChessBoard::AGENT_SACAZ_OLD, "SAC+AZ-59e5233" },
            { ChessBoard::AGENT_DQNAB,     "DQN+AB" },
            /*
               PPO+MCTS-MLP 也在这一组: 它的 learnFromReplay 同样有"池 ≥ replayBatchSize(64)
               才学"的门控, 而镜像是 2 倍增广 ⇒ 40 手的轮次给 80 条样本, 刚好过门槛。
            */
            { ChessBoard::AGENT_PPOMCTS_MLP, "PPO+MCTS-MLP" }
        };
        const int savedEpisodes = board.getBackgroundTrainEpisodes();
        const int savedMaxMoves = board.getBackgroundTrainMaxMoves();
        board.setBackgroundTrainRound(1, 40);
        for (const Case &cs : cases) {
            board.setAgentType(cs.type);
            QVector<double> losses;
            const QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&losses, &cs](double loss, const QString &agent, int step) {
                    (void)step;
                    if (agent.contains(QString::fromUtf8(cs.name))) {
                        losses.append(loss);
                    }
                });
            board.startBackgroundTraining();
            const auto t0 = std::chrono::steady_clock::now();
            while (losses.isEmpty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const long long waited = std::chrono::duration_cast<std::chrono::seconds>(
                                             std::chrono::steady_clock::now() - t0).count();
                if (waited > 240) {
                    break;
                }
            }
            board.stopBackgroundTraining();
            QObject::disconnect(conn);

            bool finite = true;
            for (double v : losses) {
                if (!std::isfinite(v)) {
                    finite = false;
                }
            }
            std::printf("    %-12s 上报损失 %d 次, 全部有限=%d\n",
                        cs.name, (int)losses.size(), (int)finite);
            CHECK(!losses.isEmpty(),
                  "这一支的后台训练往返已接上 (跑完一轮并做了一次真实更新 -> 上报损失)");
            CHECK(finite, "上报的损失是有限值");
        }
        board.setBackgroundTrainRound(savedEpisodes, savedMaxMoves);
        board.setAgentType(ChessBoard::AGENT_ALPHABETA);
    }

    /* ------------------------------------------------- 2.14 新 agent: PPO+MCTS (MLP 专家) */
    /*
       2026-09 新增: **同一套 PPO+MCTS+AlphaZero 实现**, 骨干换成稀疏 MoE + **MLP 专家**
       (E=8 top-2) —— 也就是 TB 专家那次改版之前的配置 (见 rl/ppo.h 顶部那张实测表:
       MlpExpert 前向 0.139 ms / 2.15 M 参数 vs TB<16,360> 3.59 ms / 38.0 M 参数)。

       做法与 SACAZAgent 支撑 AGENT_SACAZ / AGENT_SACAZ_MOE **完全相同** (一个类 +
       构造时选骨干), 所以要钉的不是"它能不能跑", 而是下面这四件**只有对照实验才需要**
       的性质 —— 少任何一件, "两种骨干"就会变成"两个会漂移的实现"或者"静默共用权重":
         (1) 结构确实不同 (专家数/topK/参数量);
         (2) 名字不同 (损失曲线按 agent 名分线, 同名会把两条线并成一条);
         (3) 权重路径不同;
         (4) **交叉载入必须失败** (结构指纹) —— 否则把 TB 的权重读进 MLP 骨干会
             静默串权重, 或者反过来把 MLP 的权重当 TB 用。
       最后跑一小局, 顺便量出这个 agent 的实际每手耗时 (PPO_MLP_SIMS=1600 的依据)。
    */
    std::printf("\n[2.14] 新 agent: PPO+MCTS (AlphaZero, 稀疏MoE+MLP专家)\n");
    {
        Chess c;
        c.reset();
        PPOMCTSAgent tb(c, 64, 0.99f, 0.001f, 1.414f);
        PPOMCTSAgent mlp(c, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true,
                         RL::PPO::Backbone::MlpExperts);

        std::printf("    TB  骨干: 专家 %d, topK %d, 参数量 %lld, 名字 \"%s\"\n",
                    tb.moeExpertCount(), tb.moeTopK(), tb.actorParamCount(),
                    tb.getName().c_str());
        std::printf("    MLP 骨干: 专家 %d, topK %d, 参数量 %lld, 名字 \"%s\"\n",
                    mlp.moeExpertCount(), mlp.moeTopK(), mlp.actorParamCount(),
                    mlp.getName().c_str());

        /* (1) 结构: 4/1 vs 8/2, 且 MLP 专家明显更小 */
        CHECK(tb.moeExpertCount() == RL::PPO::MOE_EXPERTS && tb.moeTopK() == RL::PPO::MOE_TOPK,
              "默认骨干 = TB 专家 (E=4 top-1), 与 PPO 的编译期常量一致");
        CHECK(mlp.moeExpertCount() == RL::PPO::MOE_MLP_EXPERTS
                  && mlp.moeTopK() == RL::PPO::MOE_MLP_TOPK,
              "新 agent 的骨干 = MLP 专家 (E=8 top-2)");
        CHECK(mlp.actorParamCount() < tb.actorParamCount(),
              "MLP 专家骨干的参数总量明显小于 TB 专家 (便宜 ~25x / 容量小 ~18x)");
        /* 两种骨干的**前向**都要能跑出有限值 (不是只把层搭起来) */
        {
            RL::Tensor st(PPOMCTSAgent::STATE_DIM, 1);
            st.zero();
            for (int i = 0; i < PPOMCTSAgent::STATE_DIM; i += 97) {
                st[i] = 1.0f;   /* 稀疏平面编码: 随便点几个 1, 只要不是全 0 */
            }
            std::vector<int> idx;
            for (int a = 0; a < 8; a++) {
                idx.push_back(a * 13);
            }
            std::vector<float> probsTb, probsMlp;
            const bool okTb = tb.ppo.actionMasked(st, idx, probsTb);
            const bool okMlp = mlp.ppo.actionMasked(st, idx, probsMlp);
            double sumTb = 0.0;
            double sumMlp = 0.0;
            for (float v : probsTb) { sumTb += v; }
            for (float v : probsMlp) { sumMlp += v; }
            std::printf("    前向 (同一局面): TB 和=%.4f, MLP 和=%.4f\n", sumTb, sumMlp);
            CHECK(okTb && okMlp, "两种骨干的策略前向都能跑 (actionMasked)");
            CHECK(std::fabs(sumTb - 1.0) < 1e-3 && std::fabs(sumMlp - 1.0) < 1e-3,
                  "两种骨干输出的都是归一化概率 (和 = 1)");
        }
        /* (2) 名字不同 */
        CHECK(tb.getName() != mlp.getName(),
              "两个骨干的 agent 名不同 (否则损失曲线会把两条线并成一条)");
        /* (3) 权重路径不同 */
        CHECK(ChessBoard::defaultWeightPath(ChessBoard::AGENT_PPOMCTS)
                  != ChessBoard::defaultWeightPath(ChessBoard::AGENT_PPOMCTS_MLP),
              "两个骨干的权重文件前缀不同 (不会互相覆盖)");
        /* (4) 交叉载入必须失败 (结构指纹) */
        {
            const std::string prefix = "weights/_test_cross_backbone";
            const bool saved = mlp.saveModel(prefix);
            std::printf("    MLP 权重落盘: %d\n", (int)saved);
            CHECK(saved, "MLP 骨干的权重能落盘 (前缀 _actor/_critic)");
            const bool crossTb = tb.loadModel(prefix);
            const bool crossMlp = mlp.loadModel(prefix);
            CHECK(!crossTb, "把 MLP 骨干的权重载入 TB 骨干**必须失败** (结构指纹)");
            CHECK(crossMlp, "同一份权重载回 MLP 骨干要成功 (指纹没把自家人挡住)");
            /* 用完删掉这两个临时文件: 留在 weights/ 里会干扰下次启动的权重扫描 */
            std::remove((prefix + "_actor").c_str());
            std::remove((prefix + "_critic").c_str());
        }

        /* 小局: 走 6 手, 顺便量每手耗时 (PPO_MLP_SIMS = 1600 的实测依据) */
        board.setMaxPliesPerGame(6);
        const ChessBoard::MatchStats st2 = board.matchAgents(ChessBoard::AGENT_PPOMCTS_MLP,
                                                             ChessBoard::AGENT_ALPHABETA, 1);
        board.setMaxPliesPerGame(300);
        std::printf("    %s\n", st2.summary().toUtf8().constData());
        if (st2.plies > 0) {
            /*
               这个数就是 chessboard.cpp 里 PPO_MLP_SIMS (=1600 次模拟) 的实测依据:
               TB 骨干在同样 400 次模拟下约 3.2 s/手, 而 MLP 骨干在**4 倍模拟次数**下
               仍然快一个数量级 —— 所以"便宜 25x"换成了"更准的访问分布", 而不是省时间。
            */
            std::printf("    PPO+MCTS-MLP 每手平均耗时: %.0f ms (预算 1600 次模拟, "
                        "见 chessboard.cpp 的 PPO_MLP_SIMS)\n",
                        (double)st2.totalThinkMs / (double)st2.plies);
        }
        CHECK(st2.games == 1, "新 agent 能打完一局 (走的是 GUI 同一条 aiThinkForAgent 路径)");
        CHECK(st2.plies == 6, "手数正常 (没有被误判成无效走法)");
        board.setMaxPliesPerGame(300);
    }

    /* ------------------------------------------------- 2.15 自动保存 × 后台训练并发 */
    /*
       用户报障: "对弈时自动保存权重的时候导致程序崩溃了"。

       根因是**缺锁**, 不是保存本身: `ChessBoard::saveCurrentAgentModel()` 原来直接
       从调用方线程去序列化主 agent 的网络, 而同时对同一张网动手的还有两处, 都规规矩矩
       拿着 `m_agentMutex`:
         * AI/对弈线程: aiThink* 的 RL 分支 (整段决策都在锁内);
         * 后台训练线程: 每轮开头的 `saveModel(_temp_train*)` 与结尾把新权重
           `loadModel(_temp_train*)` 同步回主 agent。
       于是"对局结束 → 静默保存"正好撞上"后台训练正在把新权重写进同一个网络":
       一个在遍历层逐个张量编码, 另一个在往那些张量里写 —— 数据竞争, 表现就是保存途中崩。

       这一节把那个场景**原样**跑起来: 后台训练开着 (PPO+MCTS-MLP, 一轮 40 手),
       同时另一个线程反复 `saveCurrentAgentModel()`、主线程反复 `getAgentSelfCheck()`。
       修复前这是崩溃/损坏的现场, 修复后三条路径在同一把锁上排队。
       (说明白它能证明什么: 竞态类 bug 没有"必定触发"的断言 —— 这一节是**应力测试**
        + 落盘有效性检查: 保存全部成功、最后一次落盘能被新 clone 载入、进程活着。
       串行化的**依据**在 saveCurrentAgentModel 的注释里, 不在这条断言里。)
    */
    std::printf("\n[2.15] 自动保存 × 后台训练 × 自检 并发 (报障场景)\n");
    {
        board.setAgentType(ChessBoard::AGENT_PPOMCTS_MLP);
        board.setBackgroundTrainRound(1, 40);
        board.startBackgroundTraining();

        const std::string savePath = "weights/_test_concurrent_save";
        std::atomic<int> saves{0};
        std::atomic<int> saveFails{0};
        std::atomic<bool> stopFlag{false};
        std::thread saver([&]() {
            while (!stopFlag.load()) {
                if (board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS_MLP, savePath)) {
                    saves.fetch_add(1);
                } else {
                    saveFails.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });

        int selfChecks = 0;
        int emptyChecks = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count() < 25) {
            const std::string r = board.getAgentSelfCheck(ChessBoard::AGENT_PPOMCTS_MLP);
            selfChecks++;
            if (r.empty()) {
                emptyChecks++;
            }
        }
        stopFlag = true;
        saver.join();
        board.stopBackgroundTraining();
        board.setBackgroundTrainRound(1, 60);
        board.setAgentType(ChessBoard::AGENT_ALPHABETA);

        std::printf("    25 s 内: 保存 %d 次 (失败 %d), 自检 %d 次 (空 %d)\n",
                    saves.load(), saveFails.load(), selfChecks, emptyChecks);
        CHECK(saves.load() > 0, "并发期间确实发生了保存 (场景真的被触发)");
        CHECK(saveFails.load() == 0, "每一次保存都成功 (没有被并发写坏/写失败)");
        CHECK(selfChecks > 0 && emptyChecks == 0, "并发期间自检一直有报告 (没读到空/崩)");

        /* 落盘的东西必须是真的: 新 clone 用结构指纹校验后能载入 */
        Chess probeChess;
        probeChess.reset();
        PPOMCTSAgent probe(probeChess, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true,
                           RL::PPO::Backbone::MlpExperts);
        const bool loaded = probe.loadModel(savePath);
        CHECK(loaded, "并发保存出来的权重文件能被新 clone 载入 (结构指纹通过)");
        std::remove((savePath + "_actor").c_str());
        std::remove((savePath + "_critic").c_str());
    }

    /* --------------------------------- 2.16 对弈 × 同一 agent 的后台训练 (崩溃复现) */
    /*
       2026-09 用户报障: "对弈时自动保存权重的时候导致程序崩溃了"。
       Windows 事件日志给的是 **0xC0000374 (堆损坏, ntdll 报的)** —— 那是"有人写越界/
       用了已释放的内存"的典型形状, 而不是"保存函数返回了错误"。
       下面这一段把现场按最小规模摆出来: 界面选中的 agent (= 后台训练的目标) 与对弈的
       A 方是**同一个实例**, 于是"对弈线程在搜 / 训练线程在载入新权重 / 结束后在保存"
       三件事会同时压在同一个对象上。
    */
    std::printf("\n[2.16] 对弈 × 同一 agent 的后台训练 (并发崩溃复现)\n");
    {
        board.setAgentType(ChessBoard::AGENT_PPOMCTS_MLP);
        board.setBackgroundTrainRound(1, 40);
        board.startBackgroundTraining();
        board.setMaxPliesPerGame(12);
        /*
           刻意与 GUI 同构: 对弈跑在**另一个线程** (MainWindow 用的是 m_selfPlayThread),
           主线程同时按"每手一次"的频率调 getAgentSelfCheck() —— 也就是界面那个
           自检 worker 干的事 (面板每手刷新一次)。
        */
        for (int g = 0; g < 3; g++) {
            ChessBoard::MatchStats st;
            std::thread matchThread([&]() {
                st = board.matchAgents(ChessBoard::AGENT_PPOMCTS_MLP,
                                       ChessBoard::AGENT_ALPHABETA, 1);
            });
            int checks = 0;
            while (matchThread.joinable()) {
                board.getAgentSelfCheck(ChessBoard::AGENT_PPOMCTS_MLP);
                checks++;
                if (checks > 3 && !board.isMatchRunning()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            matchThread.join();
            std::printf("    第 %d 次: %s (并发自检 %d 次)\n", g + 1,
                        st.summary().toUtf8().constData(), checks);
            CHECK(st.games == 1, "并发训练下这一局仍然正常打完");
            CHECK(st.agentErrors == 0, "没有 agent 返回无效走法");
        }
        board.stopBackgroundTraining();
        board.setBackgroundTrainRound(1, 60);
        board.setMaxPliesPerGame(300);
        board.setAgentType(ChessBoard::AGENT_ALPHABETA);
    }

    /* --------------------------------- 2.17 关掉 rollout 之后对弈仍然训练 (用户实测) */
    /*
       用户报的现象: **"只有勾选 rollout 才有 loss 曲线, 对弈过程才会进行训练"**。
       代码上确实如此: `preTrainThenDecide` 一开始就被 `m_preTrainEnabled` 短路, 而 SAC
       在整局里唯一的学习来源就是那条 rollout (它写进池的样本还是 hasSearch=false ——
       搜索算出来的 π_MCTS 被丢掉)。修法见 `SACAZAgent::learnFromSearch`: 每次真实决策
       之后, 把 (s, π_MCTS, a, r, s', done) 存进池并更新一次。

       这一节把那个场景原样跑起来: **关掉预训练**, 打一小局, 断言仍然收到 trainLossSample
       (损失曲线不再依赖那个勾选框), 并且比**改前**多出来的那几次更新确实发生了。
    */
    std::printf("\n[2.17] 关掉\"探索+预训练\"之后对弈仍然训练 (用户实测的回归钉)\n");
    {
        board.setPreTrainEnabled(false);      /* = 界面上不勾 rollout */
        board.setMaxPliesPerGame(12);
        QVector<double> samples;
        const QMetaObject::Connection conn = QObject::connect(
            &board, &ChessBoard::trainLossSample,
            [&samples](double loss, const QString &agent, int step) {
                (void)step;
                samples.append(loss);
                (void)agent;
            });
        const ChessBoard::MatchStats st =
            board.matchAgents(ChessBoard::AGENT_SACAZ, ChessBoard::AGENT_ALPHABETA, 1);
        QObject::disconnect(conn);
        board.setPreTrainEnabled(true);
        board.setMaxPliesPerGame(300);
        std::printf("    关掉 rollout: %s | 损失曲线点数 = %d\n",
                    st.summary().toUtf8().constData(), (int)samples.size());
        CHECK(st.games == 1, "这一小局正常打完");
        CHECK(!samples.isEmpty(),
              "**关掉 rollout 仍然上报了训练损失** (selectMove 里从自己的搜索学的那一次)");
    }

    /* --------------------------------- 2.19 奖励曲线改用"学习口径" (2026-09) */
    /*
       用户在界面导出的 CSV 上实测到 (docs/sac_learn_reward_2026_09.md §1.1): 界面那条
       奖励曲线用的是 `Chess::moveForward` 的 totalReward (**材质按原值 x1**), 而 agent
       在线学习用的是**自己的** `computeReward()` (**材质 x0.1** + 每步代价) 加终局值。
       两个口径差 **10 倍**, 于是"曲线上的比例"永远解释不了"学习信号的比例" —— 用户从
       CSV 里奖励最大值 4.5 读出"材质比赢棋重要 3.5 倍", 而 agent 学的是 0.35 : 1。

       这一节钉住四件事:
         (1) 口径表 (界面在开局前写标签要用它) 与**真实对象**一致;
         (2) 学习口径就是 agent 自己那两个函数 (逐位相同, 不是另抄一份公式);
         (3) 端到端: 曲线上发的值 == 学习口径账, 而学习口径账与引擎口径账之间**精确**
             满足   学习即时 = 0.1 x 引擎即时 − 0.001 x 该方手数
             (这条换算与棋局无关、逐手可验 —— 它就是"两本账都记"换来的可比对参照);
         (4) 纯搜索对手 (Alpha-Beta) 没有学习口径, 曲线仍旧走引擎口径并被标出来。
    */
    std::printf("\n[2.19] 奖励曲线改用学习口径 (agent 自己的 computeReward + 终局)\n");
    {
        /* ---- (1) 口径表与标签 ---- */
        CHECK(ChessBoard::agentHasLearningReward(ChessBoard::AGENT_SACAZ),
              "口径表: SAC+AZ 有学习口径");
        CHECK(ChessBoard::agentHasLearningReward(ChessBoard::AGENT_SACAZ_OLD),
              "口径表: 59e5233 还原版也有 (它是 SAC 的派生类, 同一份学习口径)");
        CHECK(!ChessBoard::agentHasLearningReward(ChessBoard::AGENT_MCTS),
              "口径表: MCTS 没有学习口径 (纯搜索)");
        CHECK(!ChessBoard::agentHasLearningReward(ChessBoard::AGENT_ALPHABETA),
              "口径表: Alpha-Beta 没有学习口径 (纯搜索)");
        CHECK(!ChessBoard::agentHasLearningReward(ChessBoard::AGENT_EVAB),
              "口径表: EVAB 没有 computeReward (它蒸馏给评估网络), 算引擎口径");
        const QString lLearn = ChessBoard::agentRewardCaliperLabel(ChessBoard::AGENT_SACAZ);
        const QString lEngine = ChessBoard::agentRewardCaliperLabel(ChessBoard::AGENT_MCTS);
        std::printf("    标签: SAC+AZ -> %s | MCTS -> %s\n",
                    lLearn.toUtf8().constData(), lEngine.toUtf8().constData());
        CHECK(contains(lLearn, QStringLiteral("学习口径")), "SAC+AZ 的标签写明学习口径");
        CHECK(contains(lEngine, QStringLiteral("引擎口径")), "MCTS 的标签写明引擎口径");

        /* ---- (2) 学习口径 == agent 自己那两个函数 (逐位) ---- */
        {
            Chess c;
            c.reset();
            Step probe;
            probe.valid = false;
            {
                std::vector<Step *> legal;
                c.sample(Stone::COLOR_RED, legal);
                /* 先挑着法再还池子: put 之后 legal 里的指针就不该再解引用了 */
                for (std::size_t i = 0; i < legal.size(); ++i) {
                    if (legal[i]->nextId != Stone::ID_NONE) {
                        probe = *legal[i];
                        break;
                    }
                }
                if (!probe.valid && !legal.empty()) {
                    probe = *legal[0];   /* 没找到吃子就用第一个合法着法 */
                }
                Steps::instance().put(legal);
            }
            CHECK(probe.valid, "探针棋盘上找得到一个合法着法");
            SACAZAgent sac(c, 64, 0.99f, 0.001f, 1.5f);
            CHECK(sac.hasLearningReward(), "SACAZAgent 自报有学习口径");
            const float viaCaliper = sac.learningStepReward(probe, Stone::COLOR_RED);
            const float viaAgent = sac.computeReward(probe, Stone::COLOR_RED);
            std::printf("    learningStepReward=%.6f vs computeReward=%.6f "
                        "(吃子=%s)\n", (double)viaCaliper, (double)viaAgent,
                        probe.nextId != Stone::ID_NONE ? "是" : "否");
            CHECK(viaCaliper == viaAgent,
                  "学习口径的即时奖励**逐位**等于 agent 的 computeReward");
            /*
               终局: 塑形开着 (rewardShape=2) 时它是 ±(1+败方材质/3.5), 与引擎口径的
               ±1 **不同** —— 这正是"必须问 agent 要"的理由 (抄一份公式迟早分叉)。
            */
            sac.rewardShape = 2;
            const float tCaliper =
                sac.learningTerminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);
            const float tAgent = sac.terminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED);
            std::printf("    rewardShape=2 的终局值: 学习口径 %.4f vs terminalReward %.4f "
                        "(引擎口径是 %.1f)\n", (double)tCaliper, (double)tAgent,
                        (double)outcomeForMover(Chess::RESULT_RED_WIN, Stone::COLOR_RED));
            CHECK(tCaliper == tAgent,
                  "学习口径的终局值**逐位**等于 agent 的 terminalReward");
            /*
               塑形开着时终局值 = ±(1 + 败方剩余材质/3.5) ∈ [1,2]:
               初始局面下败方一个子没少 (剩余 = 3.5) ⇒ 正好 **2.0** (实测值);
               磨到光将 ⇒ 1.0。这里断言"与引擎口径的 1.0 不同", 才是"真的走了 agent"。
            */
            CHECK(tCaliper >= 1.0f && tCaliper <= 2.0f,
                  "塑形开着时终局值落在 [1,2] (与引擎口径的常数 1.0 不同 -> 真的走了 agent)");
            sac.rewardShape = 0;
            CHECK(sac.learningTerminalReward(Chess::RESULT_RED_WIN, Stone::COLOR_RED) == 1.0f,
                  "rewardShape=0 时终局值就是 ±1 (与引擎口径一致)");
        }

        /* ---- (3) 端到端: 曲线值 == 学习口径账, 且两本账精确换算 ---- */
        board.setPreTrainEnabled(false);
        board.setMaxPliesPerGame(12);
        QVector<double> progA, progB;
        bool gotFinal = false;
        double finalA = 0.0, finalB = 0.0;
        const QMetaObject::Connection cp = QObject::connect(
            &board, &ChessBoard::matchRewardProgress,
            [&progA, &progB](int gameNo, int, double ra, double rb) {
                if (gameNo < 1) { return; }
                progA.append(ra);
                progB.append(rb);
            });
        const QMetaObject::Connection cf = QObject::connect(
            &board, &ChessBoard::gameRewardSample,
            [&gotFinal, &finalA, &finalB](int gameNo, const QString &, const QString &,
                                          double ra, double rb) {
                if (gameNo < 1) { return; }
                gotFinal = true;
                finalA = ra;
                finalB = rb;
            });
        const ChessBoard::MatchStats st19 =
            board.matchAgents(ChessBoard::AGENT_SACAZ, ChessBoard::AGENT_ALPHABETA, 1);
        QObject::disconnect(cp);
        QObject::disconnect(cf);
        board.setPreTrainEnabled(true);
        board.setMaxPliesPerGame(300);

        const ChessBoard::RewardAccounting &acct = board.lastRewardAccounting();
        std::printf("    一局: %s\n", st19.summary().toUtf8().constData());
        std::printf("    A(SAC+AZ, 学习口径=%d): 即时 引擎口径 %+.4f / 学习口径 %+.4f, "
                    "手数 %d, 终止 %+.3f -> 曲线 %+.4f\n",
                    (int)acct.learnCaliperA, acct.engineImmediateA, acct.learnImmediateA,
                    acct.movesA, acct.learnTerminalA, acct.curveFinalA());
        std::printf("    B(Alpha-Beta, 学习口径=%d): 即时 引擎口径 %+.4f, 手数 %d, "
                    "终止 %+.3f -> 曲线 %+.4f\n",
                    (int)acct.learnCaliperB, acct.engineImmediateB, acct.movesB,
                    acct.engineTerminalB, acct.curveFinalB());
        CHECK(st19.games == 1, "这一小局正常打完");
        CHECK(acct.learnCaliperA, "A(SAC+AZ) 的曲线用**学习口径**");
        CHECK(!acct.learnCaliperB, "B(Alpha-Beta) 没有学习口径 -> 曲线仍是引擎口径");
        CHECK(gotFinal, "收到局末奖励采样");
        CHECK(std::fabs(finalA - acct.curveFinalA()) < 1e-9,
              "局末曲线值 == 学习口径账 (A): 局末的点就是这条账算出来的");
        CHECK(std::fabs(finalB - acct.curveFinalB()) < 1e-9,
              "局末曲线值 == 引擎口径账 (B)");
        if (!progA.isEmpty()) {
            CHECK(std::fabs(progA.last() - acct.curveImmediateA()) < 1e-9,
                  "最后一个进度点 == 本局**即时**累计 (A, 不含终局值)");
        }
        /*
           两本账的精确换算 (容差 1e-6: 学习口径那一侧是用 float 的 0.1f / -0.001f 算的,
           与 double 的 0.1 / -0.001 有 ~1e-8 量级的表示误差; 逐手累积后仍远小于 1e-6)。
        */
        const double expectLearnA =
            0.1 * acct.engineImmediateA + (double)REWARD_STEP_COST * (double)acct.movesA;
        const double expectLearnB =
            0.1 * acct.engineImmediateB + (double)REWARD_STEP_COST * (double)acct.movesB;
        std::printf("    换算核对: A 学习 %+.6f vs 期望 %+.6f | B 学习 %+.6f vs 期望 %+.6f\n",
                    acct.learnImmediateA, expectLearnA, acct.learnImmediateB, expectLearnB);
        CHECK(std::fabs(acct.learnImmediateA - expectLearnA) < 1e-6,
              "学习口径即时 == 0.1x引擎即时 + 每步代价x手数 (A)");
        /*
           B 侧 (Alpha-Beta) 没有学习口径 ⇒ 它的学习账**故意**是空的 (0), 而不是
           "0.1x 引擎"。这一条把"没有口径就不记学习账"钉住 —— 否则 [2.19] 的换算关系
           会被误用到纯搜索 agent 身上 (那就是在半口径上做断言)。
        */
        if (acct.learnCaliperB) {
            CHECK(std::fabs(acct.learnImmediateB - expectLearnB) < 1e-6,
                  "学习口径即时 == 0.1x引擎即时 + 每步代价x手数 (B)");
        } else {
            CHECK(acct.learnImmediateB == 0.0,
                  "B 没有学习口径 ⇒ 学习账为空 (0), 曲线用的是它的引擎账");
            CHECK(std::fabs(acct.curveImmediateB() - acct.engineImmediateB) < 1e-12,
                  "B 的曲线值就是引擎口径账 (没有口径时不替换)");
        }
        CHECK(acct.movesA + acct.movesB > 0, "两方手数都记下来了 (每步代价的乘数)");
        /*
           手数之和 vs st.plies: st.plies 数是**决策次数**(含最后那次"无合法走法"),
           而被将死/困毙的那一手不会执行 moveForward, 所以账上的手数至多少 1。
        */
        CHECK(acct.movesA + acct.movesB <= st19.plies
                  && acct.movesA + acct.movesB + 1 >= st19.plies,
              "两方手数之和 == 总手数 (差至多 1: 被将死那一手没有落子)");
        if (std::fabs(acct.engineImmediateA) > 1e-9) {
            CHECK(std::fabs(acct.learnImmediateA) < std::fabs(acct.engineImmediateA),
                  "学习口径的量级小于引擎口径 (材质 x0.1 vs x1) —— 这正是两个口径差 10 倍");
        }
    }

    /* ---------------------------------------------------------------- 3. 中止 */    std::printf("\n[3] 中止: 请求 50 局, 跑一会儿后叫停\n");
    board.setMaxPliesPerGame(300);
    ChessBoard::MatchStats aborted;
    std::thread worker([&board, &aborted]() {
        aborted = board.matchAgents(ChessBoard::AGENT_ALPHABETA,
                                    ChessBoard::AGENT_EVAB, 50);
    });
    /* 等它真的开始动手, 再叫停 */
    for (int i = 0; i < 200 && !board.isMatchRunning(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(board.isMatchRunning(), "对弈已经在跑");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    board.abortMatch();
    worker.join();

    std::printf("    中止后: %s\n", aborted.summary().toUtf8().constData());
    CHECK(aborted.aborted, "标记为已中止");
    CHECK(aborted.games < 50, "没有打满请求的局数");
    CHECK(!board.isMatchRunning(), "中止后不再处于对弈中");
    /* 中止时那一局没打完, 不应被算进比分 */
    CHECK(aborted.winA + aborted.winB + aborted.draws == aborted.games,
          "半局没有被算成完整一局");

    /* ---------------------------------------------------------------- 4. 手数上限恢复 */
    std::printf("\n[4] 手数上限可恢复 (界面上默认 300)\n");
    board.setMaxPliesPerGame(300);
    CHECK(board.getMaxPliesPerGame() == 300, "上限设回 300");
    board.setMaxPliesPerGame(0);
    CHECK(board.getMaxPliesPerGame() == 1, "非法值 (0) 被夹到 1, 不会死循环");
    board.setMaxPliesPerGame(300);

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
