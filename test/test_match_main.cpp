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
#include "dqnagent.h"
#include <cmath>
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
            { ChessBoard::AGENT_SACAZ_MOE, "SAC+AZ-MoE",   true }
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
