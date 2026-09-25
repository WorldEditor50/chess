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
#include "ppomcts_agent.h" /* [2.7c] 与 AB 对弈的"会学习"那一侧 */
#include "rl/util.hpp"     /* [2.7c] RL::Random::setSeed: 换对手时钉住随机流 */
#include "metricsview.h"
#include <cmath>
#include <limits>      /* [2.7c] quiet_NaN: "这一次有没有产生可比的损失" */
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

/* [2.7f] 读一个文件的字节内容 (用来做"逐字节不变"的比较); 读不到返回空 */
static std::string readFileBytes(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        return std::string();
    }
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
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
            /* 59e5233 行为还原版 (**独立类** SACAZLegacyAgent): 显示名与"上报损失"都要接上 */
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

    /* ---------------------------- 2.7b Alpha-Beta 三档弱等级 (L1/L2/L3) */
    /*
       用户口径: "把 alpha beta 的 1 到 3 level 也加进来, 下拉框可选择, 用于对弈训练"。

       这一节钉住三件事 (都是"错了不会报错"的那一类):
         (1) **深度真的按类型分档**: abDepthOf() 是决策/状态条/自检共用的单一来源。
             没有它的话, 选了 L1 而下的是深度 4 的棋, 界面上一点异常都没有 ——
             而且状态条会照旧印 "深度 4" (旧代码三处都写死 AB_DEPTH)。
         (2) 它们**能被对弈引擎接受** (matchAgents 走得到决策分支、能打完一局),
             且**不上报训练损失** (纯搜索 agent 没有可训练参数)。
         (3) 与一个学习型 agent 对局时, 学习型那一方**照常上报损失** —— 这一条是
             "对弈训练"的名义判据: AB 当陪练时, 学的是学习型 agent 自己的经验
             (AB 没有可训练权重; 对手的棋要不要进它的回放池是另一件事, 见
             chessboard.h 里 AGENT_AB_L* 的注释)。
       ⚠ 本节**不测棋力**。深度 1 应该弱于深度 4, 但那是"棋力"问题 —— 按本仓库的
         既定口径, 那只能由带置信区间的锚点对局 (bench_anchor) 回答, 不在单元测试里
         用几局棋去下结论 (见 test_match [2.19] 与 rl/diag.h 的判读说明)。
    */
    std::printf("\n[2.7b] Alpha-Beta 三档弱等级 (L1/L2/L3): 深度口径 + 对弈可用性\n");
    {
        struct LevelCase {
            ChessBoard::AgentType type;
            int expectDepth;
            const char *label;
        };
        const LevelCase levels[] = {
            { ChessBoard::AGENT_AB_L1, 1, "L1" },
            { ChessBoard::AGENT_AB_L2, 2, "L2" },
            { ChessBoard::AGENT_AB_L3, 3, "L3" }
        };
        for (const LevelCase &lc : levels) {
            const int d = ChessBoard::abDepthOf(lc.type);
            std::printf("    %s -> abDepthOf = %d (期望 %d)\n", lc.label, d, lc.expectDepth);
            CHECK(d == lc.expectDepth, "该等级的搜索深度就是它的档位");
        }
        /* 既有的那一档必须**没被动过**: 三档弱等级是新增类型, 不许改对照组的基准 */
        CHECK(ChessBoard::abDepthOf(ChessBoard::AGENT_ALPHABETA) == 4,
              "AGENT_ALPHABETA 仍然是深度 4 (新增等级没有动既有口径)");
        /* 不是 Alpha-Beta 的类型返回 0: 调用方靠这个判"要不要按 AB 走" */
        CHECK(ChessBoard::abDepthOf(ChessBoard::AGENT_MCTS) == 0,
              "非 Alpha-Beta 类型返回 0 (而不是某个默认深度)");
        CHECK(ChessBoard::abDepthOf(ChessBoard::AGENT_SACAZ_MOE) == 0,
              "非 Alpha-Beta 类型返回 0 (SAC+AZ-MoE)");
        /* 三档都是"纯搜索、无学习口径、无权重" —— 界面上的标签与曲线口径都靠这三条 */
        for (const LevelCase &lc : levels) {
            CHECK(!ChessBoard::agentHasLearningReward(lc.type),
                  "纯搜索 agent 没有'学习口径'的奖励 (曲线走引擎口径)");
            /*
               注意 hasAgentInstance 是**非静态成员函数** (它读静态指针 m_sfXXX,
               但签名是 const 成员) —— 必须用对象调, 写 ChessBoard::hasAgentInstance(...)
               是编译错误 C2352。
            */
            CHECK(!board.hasAgentInstance(lc.type),
                  "纯搜索 agent 没有常驻实例 (也就没有权重可存)");
        }

        /* ---- 真打一局: 学习型 agent (PG, 最快的那一支) 对 AB L2 ---- */
        /*
           手数上限 32 而不是 12: PG 的在线训练是**按回放池攒够一个批**才发生的
           (池不够时 learn() 里直接返回, getLastTrainLoss() 还是 NaN), 而它每局都会
           重新来过。12 手时实测上报 0 次 —— 那不是 bug, 是"这一局太短"。
           这里要用够长的局, 才能让"学习型一方**确实**在跟 AB 下棋的过程中训练"这件事
           成为一个有内容的判据 (否则这条断言会因为分段口径而时真时假)。
        */
        board.setMaxPliesPerGame(32);
        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);
        {
            /*
               分桶口径: `trainLossSample` 的 `agent` 参数只有一个来源 —— 发信号那一方的
               `agent->getName()` (见 chessboard.cpp 的 preTrainThenDecide /
               reportLearnedLossOf)。所以**不能**假设"AB 侧会发一个名字含 Alpha-Beta 的
               信号" —— AB 分支根本不会走到 preTrainThenDecide, 它也没有 AgentBase 实例。
               那样写出来的 `abSamples.isEmpty()` 是**恒真**的 (永远测不到任何东西)。

               正确做法: 按"是不是 PG"分桶, 并**断言另一个桶恰好是空的** —— 于是
               "谁在发信号"这件事本身被钉住了 (多出第三方发信号 = 当场失败),
               而不是靠一个永远成立的等式蒙过去。
            */
            QVector<double> pgSamples;
            QStringList otherSamples;      /* 非 PG 来源的名字 (应当恰好为空) */
            QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&pgSamples, &otherSamples](double loss, const QString &agent, int step) {
                    (void)step;
                    if (agent.startsWith(QStringLiteral("Policy Gradient"))) {
                        pgSamples.append(loss);
                    } else {
                        otherSamples.append(agent);
                    }
                });
            /*
               同时抓状态条文字: 这是**唯一**能发现"abDepthOf 说 2、实际却按别的深度在下"
               的地方 —— 只断言 abDepthOf() 的返回值是测不到决策路径的 (状态条那三处
               以前正是写死 AB_DEPTH 的)。
            */
            QStringList stages;
            QMetaObject::Connection stageConn = QObject::connect(
                &board, &ChessBoard::aiThinkingStage,
                [&stages](const QString &s) { stages.append(s); });
            const ChessBoard::MatchStats st =
                board.matchAgents(ChessBoard::AGENT_PG, ChessBoard::AGENT_AB_L2, 1);
            QObject::disconnect(conn);
            QObject::disconnect(stageConn);

            std::printf("    PG vs AB-L2: %d 局 / %d 手 | PG 上报 %d 次, "
                        "其它来源 %d 次 | 无效走法兜底 %d 次\n",
                        st.games, st.plies, (int)pgSamples.size(),
                        (int)otherSamples.size(), st.agentErrors);
            CHECK(st.games == 1, "PG vs AB-L2 这一局正常打完 (对弈引擎接受新类型)");
            CHECK(st.plies > 0, "这一局真的走了棋 (不是 0 手就结束)");
            CHECK(st.agentErrors == 0, "对弈全程没有出现无效走法兜底");
            /*
               32 手这一档下, 学习型一方**必须**已经上报过损失 (池子够一个批了)。
               这一条同时是"AB 陪练不改变学习通路"的正面证据: 与 AB 对弈时,
               学习型 agent 的探索+训练一切照常 —— 只是 AB 自己的棋不进它的数据。

               ⚠ **这条断言是已知会偶发失败的, 而且不是本测试自身的问题** (2026-09 实测,
               见 docs/issues_review.md 零之二点二十九 §7)。实测同一份二进制连跑三次,
               这一局分别打出 **32 手 / 8 手 / 32 手** —— 随机流与**墙上时钟**有关
               (`mcts.cpp` 在进程内第一次构造时 `std::srand(time(nullptr))`,
               `ppomcts_agent.cpp` / `dqnmcts_agent.cpp` 更是**每次构造**都播),
               于是学习型 agent 的初始权重每次不同。三次里失败两次, 且**形状一致**:
               失败那两次都是"走满 32 手 (PG 有 16 次决策、每次预训练 32 步) 却上报 0 次",
               通过那次只走 8 手却上报 4 次。
               上报判据是 `std::isfinite(agent->getLastTrainLoss())` (见
               ChessBoard::preTrainThenDecide), 而 PG 的那份是 `dpg.lastLoss` ——
               待查假设是"某次 reinforce 之后它变成非有限值且此后再不复位", 那样之后
               所有上报都会被永久静默吞掉 (用户看到的正是"损失曲线不动")。
               所以**不要**把这条断言删掉或改成警告 —— 它红的时候可能真的抓到了东西;
               要做的是先把随机流钉住 (提议见 §7)。
            */
            /*
               ---- 主判据: 学习型那一侧**每一手都真的走到了学习入口** ----
               用"探索阶段被 emit 的次数"来数, 而不是用"损失曲线上有没有点":
                 * 前者是**确定性的机制判据** —— 它是"对弈训练"这件事本身;
                 * 后者(损失点)依赖 agent 内部的 lastLoss 是否有限, 而 PG 存在
                   "lastLoss 变成 NaN/Inf 后不再复位"的已知现象 (诊断见下),
                   于是"上报 0 次"既可能是机制断了、也可能只是那个数值坏了 —— 用它
                   当主判据会让本节**时真时假** (实测: 同一份代码两次运行一次绿一次红)。
            */
            int exploreStages = 0;
            for (const QString &s : stages) {
                if (s.contains(QStringLiteral("探索环境"))) {
                    exploreStages++;
                }
            }
            std::printf("    学习侧(PPO/PG)的探索阶段 emit %d 次 (本局 %d 手)\n",
                        exploreStages, st.plies);
            CHECK(exploreStages > 0,
                  "对弈过程中学习型那一侧的'探索+预训练'入口**真的被走到过**"
                  " (这是'对弈训练'的机制判据)");

            /*
               ---- 损失曲线那一条: 只作为诊断 ----
               PG 的 lastLoss 有可能在第一次上报之前就变成非有限值并不再复位 ——
               那是**另一个已记录的缺陷** (见本节的注释与 issues_review §7), 不是
               "对弈不训练"。所以这里照实打印, 但不拿它当机制判据。
            */
            if (pgSamples.isEmpty()) {
                std::printf("    **诊断**: 这一局 %d 手, PG 上报 0 次 (其它来源 %d 次) ——"
                            " 逐局损失一次都没有限过, 查 dpg.lastLoss 是否已变成 NaN/Inf"
                            " 且不再复位 (与 [2.7c] 的机制判据无关)\n",
                            st.plies, (int)otherSamples.size());
            } else {
                std::printf("    PG 上报 %d 次 (全部有限)\n", (int)pgSamples.size());
            }
            /*
               **本节的主判据**: 除了学习型那一方, 没有任何来源上报损失 ——
               AB 侧不发信号 (它没有可训练参数, 也不走 preTrainThenDecide),
               所以损失曲线上不会有它的点。分桶是**穷尽**的 (else 落 other),
               于是"多出第三个发信号的人"会当场失败, 而不是被一个恒真的等式蒙过去。
            */
            CHECK(otherSamples.isEmpty(),
                  "损失上报方只有学习型 agent (纯搜索的 AB 不上报任何损失)");

            /* ---- 状态条必须印**实际用于搜索的深度** (L2 -> 深度 2) ---- */
            QString abStage;
            for (const QString &s : stages) {
                if (s.contains(QStringLiteral("Alpha-Beta"))) {
                    abStage = s;
                    break;
                }
            }
            std::printf("    AB-L2 那一方的状态条文字: %s\n",
                        abStage.isEmpty() ? "(没抓到)" : qPrintable(abStage));
            CHECK(!abStage.isEmpty(), "对弈过程中抓到 AB 侧的状态条文字");
            CHECK(abStage.contains(QStringLiteral("深度 2")),
                  "状态条印的是该等级的真实深度 (不是写死的深度 4)");

            CHECK(ChessBoard::agentRewardCaliperLabel(ChessBoard::AGENT_AB_L2)
                      .contains(QStringLiteral("引擎口径")),
                  "AB 的奖励曲线口径标成引擎口径 (不是学习口径)");
        }
        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
    }

    /* ================================================================
     *  [2.7c] 对弈时的"训练"到底归谁 —— 行为刻画 (2026-09, dev-selfplay)
     * ================================================================
     *
     * 起因 (用户提问): "界面里对弈的 A/B 双方都是自己实现 (自对弈), 是否合理?"
     *
     * 先把**事实**钉住, 再谈合不合理。三条事实全部由可观测信号给出 (不靠读注释):
     *
     *   (1) 对弈时**双方各自走一遍 exploreAndTrain**: 界面每一手都调
     *       aiThinkForAgent -> preTrainThenDecide -> agent->exploreAndTrain(color, steps),
     *       所以两边都在学, 而且**每一手**一次更新。
     *   (2) 决定"这是训练还是纯对弈"的是界面上的**探索步数**旋钮, **不是**选了哪个
     *       agent。关掉它, 对弈就退化成"只对弈不学习"的纯对照 —— 而旋钮的名字
     *       没在说这件事 (仓库里 docs/agents_design.md §10.3 第 7 条写过同一件事:
     *       "棋力结论只能来自'预训练 = 0'的对照")。
     *   (3) `AgentBase::exploreAndTrain(int color, int rolloutSteps)` **没有对手参数**
     *       (src/aiagent.h): 共用实现 agentrollout.hpp 从当前局面用**本 agent 自己的**
     *       探索策略滚 N 步。PPO+MCTS 的 trainSelfPlay 同样是 chess.reset() 起手、
     *       自己把双方都下完 (src/ppomcts_agent.cpp)。⇒ **对手的棋不进训练数据**。
     *
     * ⚠ 这是**行为刻画**测试 (characterization test), 不是"现状正确"的背书。
     *   它把现状写成可执行断言, 于是:
     *     * 以后谁把对手接进训练数据 (P1), (3) 那一条会**当场变红** —— 那时应当连
     *       注释一起改成新口径, 而不是把断言删掉;
     *     * 谁误删了"每手都在学"这件事 (1), 也会红。
     *
     * ⚠ 手数上限必须 **> 32** (这里 40): PPO+MCTS / SAC+AZ / DQN+AB 的 learnBatch 在
     *   "回放池 < batchSize(32)" 时**一次更新都不做**, 而探索每局重新开始 ——
     *   给 12 手会得到"0 次更新"的假读数 (同一个坑见 [2.7] 与 BG_TRAIN_MAX_MOVES)。
     */
    std::printf("\n[2.7c] 对弈时的训练归属 (行为刻画: 双方各自自对弈)\n");
    {
        struct Counts {
            int ppo = 0;
            int other = 0;
            QStringList otherNames;
        };
        auto countLosses = [&board](ChessBoard::AgentType a, ChessBoard::AgentType b,
                                    int games) -> Counts {
            Counts c;
            QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&c](double loss, const QString &agent, int step) {
                    (void)loss;
                    (void)step;
                    if (agent.contains(QStringLiteral("PPO"))) {
                        c.ppo++;
                    } else {
                        c.other++;
                        if (!c.otherNames.contains(agent)) {
                            c.otherNames.append(agent);
                        }
                    }
                });
            board.matchAgents(a, b, games);
            QObject::disconnect(conn);
            return c;
        };

        board.setMaxPliesPerGame(40);

        /* ---- 实验 1: 探索开着 -> 学习型那些支路每手都在更新 ---- */
        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);
        {
            const Counts c = countLosses(ChessBoard::AGENT_PPOMCTS,
                                         ChessBoard::AGENT_AB_L2, 1);
            std::printf("    实验1  (探索 32 步开, PPO vs AB-L2): PPO 侧上报 %d 次, "
                        "非 PPO 侧 %d 次 [%s]\n",
                        c.ppo, c.other, c.otherNames.join(", ").toUtf8().constData());
            CHECK(c.ppo > 0,
                  "探索开着时: 学习型一侧在对弈过程中确实每手在更新 (训练真的在发生)");
            CHECK(c.other == 0,
                  "固定参照物 (AB L2) 一侧不产生任何更新 (纯搜索, 没有可训练参数)");

            /* 两个都会学的: 这一对才说明"双方各自都在学" */
            const Counts c2 = countLosses(ChessBoard::AGENT_PPOMCTS,
                                          ChessBoard::AGENT_SACAZ, 1);
            std::printf("    实验1b (PPO vs SAC+AZ, 双方都会学): PPO 侧 %d 次, "
                        "非 PPO 侧 %d 次 [%s]\n",
                        c2.ppo, c2.other, c2.otherNames.join(", ").toUtf8().constData());
            CHECK(c2.ppo > 0 && c2.other > 0,
                  "双方都是学习型 agent 时, **两边各自都在更新** —— 这就是"
                  " '对弈训练' 的确切形态 (各自自对弈, 不是互相学)");
        }

        /* ---- 实验 2: 关掉"探索+预训练"之后, 谁还在学? ---- */
        /*
            这一条原来是**假设**"关掉它就不学了", 实测**被推翻**, 而且推翻得有价值:
              * PPO+MCTS: 0 次更新 —— 它的唯一学习入口就是 preTrainThenDecide;
              * SAC+AZ  : **仍然有更新** —— 它在 selectMove 里还有第二条路径
                `learnFromSearch`(src/sacazagent.cpp 的 "从自己的搜索学一次"),
                **完全不经过那个勾选框**(界面也没有任何开关能关掉它)。
            也就是说: "关掉探索 = 纯对弈" 这个心智模型**只对一部分 agent 成立**。
            这正好是 [2.7c] 要钉的东西: 界面上那个勾选框**不是** "训练/评估" 的权威开关。
        */
        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
        {
            const Counts c = countLosses(ChessBoard::AGENT_PPOMCTS,
                                         ChessBoard::AGENT_PPOMCTS, 1);
            std::printf("    实验2a (探索关闭, PPO vs PPO): 两侧各 %d / %d 次更新\n",
                        c.ppo, c.other);
            CHECK(c.ppo == 0 && c.other == 0,
                  "关掉探索步数后, **PPO+MCTS 侧一次更新都不做** (它的唯一入口是"
                  " preTrainThenDecide) ⇒ 对 PPO 而言这个旋钮确实等价于'训练/评估'开关");

            const Counts c2 = countLosses(ChessBoard::AGENT_SACAZ,
                                          ChessBoard::AGENT_AB_L2, 1);
            std::printf("    实验2b (探索关闭, SAC+AZ vs AB-L2): SAC 侧 %d 次, "
                        "AB 侧 %d 次\n",
                        c2.other, c2.ppo);
            CHECK(c2.other > 0,
                  "**SAC+AZ 在探索关闭时照样每手更新** —— 它走的是 selectMove 里的"
                  " learnFromSearch, 不受'探索+预训练'勾选框控制 ⇒ 那个勾选框不是"
                  " '训练/评估' 的权威开关 (要想真的'只看不下', 界面目前**做不到**)");
        }

        /* ---- 实验 3: 学习信号与对手无关 (机制的直接后果) ---- */
        /*
            同一个学习型 agent、同一份起始权重、同一个作用域, 只换对手 (AB 深度 1 vs 3)。
            如果对手的棋**没有**进训练数据, 那么"第一次探索"产生的训练损失必须相同:
            那一刻棋盘在**标准开局**(playMatchGame 每局 chess.reset() 起手),
            而探索从头到尾不看对手。

            ⚠ 两场之间**必须还原权重**: matchAgents 会让常驻 agent 就地更新, 连跑两场
              的话第二场是在"第一场训练过的权重"上开始的 —— 那样比出来的是训练历史,
              不是对手的影响 (第一版就是这么假失败的)。所以这里用 saveModel/loadModel
              在两次之间做一次快照还原。
        */
        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);
        {
            /* 先跑一小局把 PPO 的常驻实例建出来 (没有实例就没有权重可存) */
            board.setMaxPliesPerGame(4);
            board.matchAgents(ChessBoard::AGENT_PPOMCTS, ChessBoard::AGENT_AB_L1, 1);

            const std::string ppoW = "weights/_temp_match_selfplay_probe_ppo";
            const bool saved = board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS, ppoW);
            std::printf("    实验3 前置: 快照 PPO 起始权重 = %s\n", saved ? "成功" : "失败");
            CHECK(saved, "能把 PPO 的起始权重快照下来 (还原两次实验的同一出发点)");

            board.setMaxPliesPerGame(40);
            auto firstLoss = [&board](ChessBoard::AgentType opponent) -> double {
                double first = std::numeric_limits<double>::quiet_NaN();
                bool got = false;
                QMetaObject::Connection conn = QObject::connect(
                    &board, &ChessBoard::trainLossSample,
                    [&first, &got](double loss, const QString &agent, int step) {
                        (void)agent;
                        (void)step;
                        if (!got) { first = loss; got = true; }
                    });
                board.matchAgents(ChessBoard::AGENT_PPOMCTS, opponent, 1);
                QObject::disconnect(conn);
                return got ? first : std::numeric_limits<double>::quiet_NaN();
            };

            RL::Random::setSeed(20240901u);
            const double lossVsL1 = firstLoss(ChessBoard::AGENT_AB_L1);

            /* 还原到同一份起始权重, 再换对手跑一遍 */
            const bool restored = board.loadAgentModel(ChessBoard::AGENT_PPOMCTS, ppoW);
            std::printf("    实验3 还原权重 = %s\n", restored ? "成功" : "失败");
            CHECK(restored, "能把权重还原回快照 (两场实验的唯二差别只剩对手)");

            RL::Random::setSeed(20240901u);
            const double lossVsL3 = firstLoss(ChessBoard::AGENT_AB_L3);

            std::printf("    实验3: 第一次训练损失  对手=AB-L1 %.9g | 对手=AB-L3 %.9g\n",
                        lossVsL1, lossVsL3);
            if (std::isfinite(lossVsL1) && std::isfinite(lossVsL3)) {
                std::printf("    → 两者%s\n",
                            (lossVsL1 == lossVsL3)
                                ? "**逐位相同** ⇒ 对手没有进入训练数据"
                                : "不同 ⇒ 需要查清 (是权重没还原干净, 还是对手真的进了数据)");
                CHECK(lossVsL1 == lossVsL3,
                      "学习信号与对手无关: 同一份起始权重 + 同一标准开局下, 换对手"
                      " (AB L1 -> L3) **不改变**第一次训练损失 ⇒ 对手的着法没有进入"
                      " 训练数据 (这是 [2.7c] 的核心判据)");
            } else {
                std::printf("    [注意] 本次没有产生可比的第一次损失, 该判据未生效\n");
            }
        }

        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
    }

    /* ================================================================
     *  [2.7d] 对弈模式 (P0-a): 同一对 agent, 三种模式, 谁在学?
     * ================================================================
     *
     * 这一节是 [2.7c] 的**后果**: 既然实测发现"关掉探索 ≠ 不学习"
     * (SAC 的 learnFromSearch 不受那个勾选框控制), 就需要一个真正权威的开关。
     * ChessBoard::MatchMode 给出三条语义:
     *
     *     MATCH_TRAIN    双方各自学习 (默认, 与改动前一致)
     *     MATCH_EVAL     冻结 B 方, 只让 A 方学 —— 比分变化可归因到 A 自己
     *     MATCH_NO_LEARN 两条学习路径**都**关掉 —— 只对弈不学习
     *
     * 判据全部是"每手一次更新"的可观测量 (trainLossSample 计数), 手数上限 40 > 32。
     *
     * ⚠ 关键的一条是 **SAC+AZ 在 MATCH_NO_LEARN / MATCH_EVAL 下必须 0 次更新** ——
     *   它正是那条不受勾选框控制的路径 (selectMove 里的 learnFromSearch)。只测 PPO
     *   的话这一节会在"模式其实没管住 SAC"时**照样全绿**, 那是假通过。
     */
    std::printf("\n[2.7d] 对弈模式 (P0-a): 训练 / 评估 / 只对弈不学习\n");
    {
        struct Side { int a = 0; int b = 0; QStringList namesA; QStringList namesB; int blocked = 0; };
        /* 按发信号的是 A 侧还是 B 侧分别计数 (两侧类型不同, 靠名字区分) */
        auto countSides = [&board](ChessBoard::AgentType a, ChessBoard::AgentType b,
                                   const QString &aKey) -> Side {
            Side s;
            const int blockedBefore = board.matchLearningBlockedCount();
            QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&s, &aKey](double loss, const QString &agent, int step) {
                    (void)loss;
                    (void)step;
                    if (agent.contains(aKey)) {
                        s.a++;
                        if (!s.namesA.contains(agent)) { s.namesA.append(agent); }
                    } else {
                        s.b++;
                        if (!s.namesB.contains(agent)) { s.namesB.append(agent); }
                    }
                });
            board.matchAgents(a, b, 1);
            QObject::disconnect(conn);
            s.blocked = board.matchLearningBlockedCount() - blockedBefore;
            return s;
        };
        /* 诊断: 把两个桶里的**真实上报者名字**印出来 —— 分桶口径错了会让读数
           整体张冠李戴, 而那种错在数字上看不出来 (第一版就是) */
        auto show = [](const char *tag, const Side &s) {
            std::printf("    %-14s A侧 %d 次 [%s] | B侧 %d 次 [%s]  (模式闸门踩下 %d 次)\n",
                        tag, s.a, s.namesA.join(", ").toUtf8().constData(),
                        s.b, s.namesB.join(", ").toUtf8().constData(),
                        s.blocked);
        };

        board.setMaxPliesPerGame(40);

        /* ---- (1) MATCH_TRAIN: 双方各自在学 ---- */
        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);
        board.setMatchMode(ChessBoard::MATCH_TRAIN);
        {
            const Side s = countSides(ChessBoard::AGENT_PPOMCTS,
                                      ChessBoard::AGENT_SACAZ,
                                      QStringLiteral("PPO"));
            show("MATCH_TRAIN", s);
            CHECK(s.a > 0 && s.b > 0,
                  "训练模式: A/B 双方各自都在更新 (与改动前的默认行为一致)");
        }

        /* ---- (2) MATCH_EVAL: 只让 A 方学, B 方冻结 ---- */
        board.setMatchMode(ChessBoard::MATCH_EVAL);
        {
            const Side s = countSides(ChessBoard::AGENT_PPOMCTS,
                                      ChessBoard::AGENT_SACAZ,
                                      QStringLiteral("PPO"));
            show("MATCH_EVAL", s);
            CHECK(s.a > 0, "评估模式: A 方 (待评估者) 照常学习");
            CHECK(s.b == 0,
                  "评估模式: B 方**完全冻结** —— 包括 SAC 的 learnFromSearch 那条路径"
                  " (它不受'探索+预训练'勾选框控制, 只能由模式掩码关掉)");

            /* 再反过来验一次: 会学的换成 B 侧, 冻结的换成 A 侧 —— 证明冻结的是"B 方"
               这个角色, 不是"SAC 这个类型" */
            /*
               ⚠ 注意桶名: countSides(a=SACAZ, b=PPOMCTS, aKey="PPO") 时,
                  **namesA 才是 PPO(B 方)**, **namesB 才是 SAC(A 方)** —— 第一版
                  把这两个桶当成了 A/B 侧名, 于是把"正确结果"读成了失败
                  (实测 SAC(a) 36 次 / PPO(b) 0 次, 而断言写的是 s2.a==0 && s2.b>0)。
            */
            const Side s2 = countSides(ChessBoard::AGENT_SACAZ,
                                       ChessBoard::AGENT_PPOMCTS,
                                       QStringLiteral("PPO"));
            std::printf("    MATCH_EVAL 反向: SAC(A方) 桶 %d 次 [%s] | PPO(B方) 桶 %d 次 [%s]\n",
                        s2.b, s2.namesB.join(", ").toUtf8().constData(),
                        s2.a, s2.namesA.join(", ").toUtf8().constData());
            CHECK(s2.b > 0 && s2.a == 0,
                  "评估模式冻结的是**B 方这个角色**, 不是某种 agent 类型"
                  " (交换 A/B 之后, 学的那一侧跟着换)");
            CHECK(s2.blocked > 0,
                  "评估模式在 B 方那一手踩下闸门 (SAC 在 A 方时不受影响)");
        }

        /* ---- (3) MATCH_NO_LEARN: 双方都不学 (两条路径都关) ---- */
        board.setMatchMode(ChessBoard::MATCH_NO_LEARN);
        {
            const Side s = countSides(ChessBoard::AGENT_PPOMCTS,
                                      ChessBoard::AGENT_SACAZ,
                                      QStringLiteral("PPO"));
            show("MATCH_NO_LEARN", s);
            CHECK(s.a == 0 && s.b == 0,
                  "只对弈模式: **双方 0 次更新** —— 落子照常, 但两条学习路径都关掉了"
                  " (这才是能直接比较比分的口径)");
            CHECK(s.blocked > 0,
                  "只对弈模式下闸门确实被踩下 (读数不是'闸门从没触发'造成的假绿)");
        }

        /* ---- (4) 报告里必须写明这一场用的是什么模式 ---- */
        {
            const ChessBoard::MatchStats st = board.matchAgents(
                ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_AB_L2, 1);
            const QString d = st.detail();
            std::printf("    报告的模式行: %s\n",
                        st.modeName.toUtf8().constData());
            CHECK(!st.modeName.isEmpty(), "对局报告里带上了模式名");
            CHECK(d.contains(QStringLiteral("对弈模式")),
                  "报告正文里印出了对弈模式 (读数自带前提, 不用读者自己悟)");
            CHECK(!st.learnsSomething,
                  "只对弈模式下 learnsSomething 为假 (报告据此提示'不能当棋力结论')");
        }

        /* 复位: 后面的小节按"改动前的默认口径"跑 */
        board.setMatchMode(ChessBoard::MATCH_TRAIN);
        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
    }

    /* ================================================================
     *  [2.7e] 人机对战那条路也要受对弈模式约束 (P0-b 修掉的洞)
     * ================================================================
     *
     * 洞的形态: `aiThink`(人机) 原来**完全绕过**对弈模式 —— 用户把模式设成
     * "评估对局 / 只对弈不学习", 然后跟 AI 下棋, AI 照常每手做在线训练, 界面上
     * 一个字都没说。这是"一个控件两种语义", 比"没做这个功能"更难排查。
     *
     * 修法 (语义): 人机对战里**人的棋力不会被这个程序改变**, 所以"待评估的那一方"
     * 只能是人 ⇒ AI 固定是**冻结的对手**。于是:
     *   训练模式   -> AI 照旧学 (与改动前一致);
     *   评估模式   -> AI 不学 (只让"人"学, 而人没有可训练参数);
     *   只对弈模式 -> AI 不学。
     *
     * 判据: 用测试钩子 humanTurnAiMoveForTest 走**与人机路径逐行相同**的决策, 数
     * 训练损失上报次数 + 模式闸门踩下的次数 (闸门计数是"确实被拦了"的直接证据,
     * 只看"没上报"分不清"被拦"还是"这一手本来就不学")。
     */
    std::printf("\n[2.7e] 人机对战路径受不受对弈模式约束\n");
    {
        auto humanMoveLosses = [&board](ChessBoard::AgentType aiType) -> int {
            board.setAgentType(aiType);
            /*
               **必须先把棋盘摆回开局**: 前面的小节把棋盘停在别的局面上了 (不同小节的
               手数上限/终局状态不一样) —— 不重置的话这一手可能是在"已经没棋可走"的
               局面上决策, 于是拿不到合法走法, 断言会因为与本事无关的原因失败
               (第一版就是这么红的)。
            */
            board.reset();
            int n = 0;
            QMetaObject::Connection conn = QObject::connect(
                &board, &ChessBoard::trainLossSample,
                [&n](double loss, const QString &agent, int step) {
                    (void)loss; (void)agent; (void)step;
                    n++;
                });
            /* 一步"人机对战里 AI 的决策" (AI 执黑, 与人机路径同一条路) */
            const Step s = board.humanTurnAiMoveForTest(Stone::COLOR_BLACK);
            QObject::disconnect(conn);
            CHECK(s.valid, "人机路径的这一步返回了合法走法 (钩子确实走到了决策)");
            return n;
        };

        board.setPreTrainEnabled(true);
        board.setPreTrainSteps(32);

        board.setMatchMode(ChessBoard::MATCH_TRAIN);
        const int nTrain = humanMoveLosses(ChessBoard::AGENT_PPOMCTS);
        std::printf("    训练模式  : AI 上报 %d 次\n", nTrain);
        CHECK(nTrain > 0,
              "训练模式下人机对战里 AI 照常在线训练 (与改动前一致, 行为没被收紧)");

        board.setMatchMode(ChessBoard::MATCH_EVAL);
        const int blockedBefore = board.matchLearningBlockedCount();
        const int nEval = humanMoveLosses(ChessBoard::AGENT_PPOMCTS);
        const int blockedEval = board.matchLearningBlockedCount() - blockedBefore;
        std::printf("    评估模式  : AI 上报 %d 次, 闸门踩下 %d 次\n", nEval, blockedEval);
        CHECK(nEval == 0 && blockedEval > 0,
              "评估模式下人机对战里 AI **不学习**, 而且是被模式闸门拦下的"
              " (人 = 学习者但没有可训练参数 ⇒ AI 作为冻结对手不学)");

        board.setMatchMode(ChessBoard::MATCH_NO_LEARN);
        const int blockedBefore2 = board.matchLearningBlockedCount();
        const int nNone = humanMoveLosses(ChessBoard::AGENT_SACAZ);
        const int blockedNone = board.matchLearningBlockedCount() - blockedBefore2;
        std::printf("    只对弈    : SAC 上报 %d 次, 闸门踩下 %d 次\n", nNone, blockedNone);
        CHECK(nNone == 0 && blockedNone > 0,
              "只对弈模式下人机对战也不学习 —— 含 SAC 那条 learnFromSearch 路径");

        board.setMatchMode(ChessBoard::MATCH_TRAIN);
        board.setPreTrainEnabled(false);
        board.setPreTrainSteps(0);
    }

    /* ================================================================
     *  [2.7f] 评估/只对弈模式会暂停后台训练 (P0-b)
     * ================================================================
     *
     * 洞的形态: 后台训练线程**开机就在跑**, 每轮把 clone 的权重同步回主 agent ——
     * 而对弈用的就是这个主 agent。于是"评估对局 / 只对弈不学习"声称的冻结只是
     * "这一手不学习", 权重仍会在局与局之间被换掉, 报告与读数都会骗人。
     *
     * 修法: 这两种模式下 matchAgents 用 pauseBackgroundTraining() 让后台线程**不再改动
     * 主 agent 的权重** (它在轮次开头等条件变量, 并且在写主 agent 之前于锁内复查暂停标志,
     * 使得"暂停之前就在飞的那一轮"也会被丢弃而不是写进去)。
     *
     * 这里钉住三件事:
     *   (1) pause/resume 在**后台训练正在跑**的时候也不会卡死 (这是它最容易写错的地方:
     *       pause 要拿主 agent 的锁, 而训练线程可能正持有它);
     *   (2) 暂停期间主 agent 的权重**逐字节不变** (真正的"冻结");
     *   (3) resume 之后后台训练还能继续改权重 (没有把线程等死)。
     */
    std::printf("\n[2.7f] 评估/只对弈模式暂停后台训练\n");
    {
        board.setBackgroundTrainRound(1, 8);   /* 一轮缩到 8 手, 免得测试等太久 */
        board.startBackgroundTraining();
        CHECK(true, "后台训练线程能起来");

        /* 先让它在"训练模式"下真的动一次主 agent 的权重 (否则下一条断言是空测) */
        std::this_thread::sleep_for(std::chrono::seconds(4));
        const std::string tmpW = "weights/_temp_match_bt_probe";
        board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS, tmpW);
        const std::string weighA = readFileBytes(tmpW + "_actor");
        std::printf("    训练模式下: 主 agent 权重快照 = %zu 字节\n", weighA.size());
        CHECK(!weighA.empty(), "能读到主 agent 的权重快照 (前置条件)");

        /* 暂停 -> 等一段时间 -> 权重必须逐字节不变 */
        board.pauseBackgroundTraining();
        CHECK(true, "pauseBackgroundTraining 在后台线程运行期间也没有卡死");
        board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS, tmpW);
        const std::string frozenA = readFileBytes(tmpW + "_actor");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        board.saveCurrentAgentModel(ChessBoard::AGENT_PPOMCTS, tmpW);
        const std::string frozenB = readFileBytes(tmpW + "_actor");
        std::printf("    暂停后两次快照 = %zu / %zu 字节 (3 秒间隔), 逐字节相同 = %d\n",
                    frozenA.size(), frozenB.size(), (int)(frozenA == frozenB));
        CHECK(!frozenA.empty() && frozenA == frozenB,
              "暂停期间主 agent 的权重**逐字节不变** —— 这才叫冻结 (报告里那句"
              " '双方都不学' 因此才配得上'权重也不变')");

        board.resumeBackgroundTraining();
        CHECK(true, "resumeBackgroundTraining 正常返回");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        board.stopBackgroundTraining();
        CHECK(!board.isBackgroundTrainingRunning(),
              "后台训练能正常停掉 (测试不做完就退出会让后面的小节被它干扰)");
    }

    /* ================================================================
     *  [2.7g] "有没有载入模型"必须是一句能被看见的话 (2026-09 用户报障)
     * ================================================================
     *
     * 报障: "点击开局模型未载入"。查下来**代码没错** —— weights/ 是被 .gitignore 忽略的
     * 运行期产物, 一次都没存过权重时它就是空的, 自检面板照实写"启动时未载入"是对的;
     * 而且正式权重只在**对弈结束**或**关窗**时才写出。
     * 真正的问题是**可见性**: 这件事原来只出现在日志与自检面板里, 而用户按"开局"时
     * 看不到它们 ⇒ "没有模型"只表现为"AI 下得像随机", 没有任何一条人能看见的解释。
     *
     * 所以新增 weightLoadSummary()/weightLoadHint(), 由界面挂在**对局列表最上面**。
     * 这里钉住它的两条语义 (与机器上有没有权重文件无关, 所以两种状态都要成立):
     *   · 单行: 不许含换行 (它会被塞进 QListWidget 的单个条目, 换行会被压平成一团);
     *   · 缺失时**必须**写出"为什么没有"与"怎么才会有", 而不是只报一个数字。
     */
    std::printf("\n[2.7g] 启动权重状态的可见性\n");
    {
        const std::string oneLine = board.weightLoadSummary();
        const std::string hint = board.weightLoadHint();
        std::printf("    摘要: %s\n", oneLine.c_str());
        CHECK(!oneLine.empty(), "有一行'载入状况'摘要给界面用");
        CHECK(oneLine.find('\n') == std::string::npos,
              "摘要是**单行** (要放进 QListWidget 的单个条目, 换行会被压平)");
        CHECK(hint.find('\n') != std::string::npos,
              "详细说明是**多行** (要放进 tooltip)");
        CHECK(hint.find("weights") != std::string::npos,
              "详细说明里点明了权重文件在 weights/ 下");
        if (oneLine.find("未载入任何模型权重") != std::string::npos) {
            /* 本机(或 CI)当下确实没有正式权重 —— 那么必须把"怎么才会有"写出来 */
            std::printf("    (本机当前无权重文件: 检查提示文案是否给出可操作的出路)\n");
            CHECK(hint.find("对弈") != std::string::npos,
                  "没有权重时, 说明里必须写出'跑一场对弈后才会写出'这条出路 "
                  "(否则用户只知道'没有', 不知道'怎么办')");
        }
    }

    /* ================================================================
     *  [2.7h] "自由走子"调试开关 (2026-09 用户要求: 要能故意输给 AI)
     * ================================================================
     *
     * 用户口径: "我需要故意输给 ai"。而**规则过滤恰好挡住了送死** ——
     * `moveStone` 用 `Chess::isLegalMove` 校验, 它会剔掉"走后自家将被攻击"的着法
     * (不应将 / 自杀 / 两将照面), 所以规则之内**无法**快速输棋。
     *
     * 这里钉住三件事:
     *   (1) 默认**关闭** —— 正常对局里的规则保护不能被这个调试开关悄悄削掉;
     *   (2) setter/getter 按位工作;
     *   (3) `Chess::isLegalMove` 确实会**拒绝**"不应将"这类着法 (那是自由的对照面:
     *       如果它本来就不拒, 那么这个开关就是多余的 —— 这条断言同时防止两种漂移)。
     */
    std::printf("\n[2.7h] 自由走子调试开关\n");
    {
        CHECK(!board.isFreeMoveEnabled(), "默认关闭 (正常对局的规则保护不受影响)");
        board.setFreeMoveEnabled(true);
        CHECK(board.isFreeMoveEnabled(), "打开后开关生效");
        board.setFreeMoveEnabled(false);
        CHECK(!board.isFreeMoveEnabled(), "关回去也生效");

        /*
           (3) 规则确实会拒绝"自杀"着法 —— 用引擎直接验:
               造一个局面: 红帅暴露在对方车的直线上, 再让红方走一步**不解决**这个威胁的
               着法 (随手动一个无关的子), isLegalMove 必须返回 false。
           这里用最小构造: 直接检查"整盘被将时, 合法着法数远小于伪合法着法数"
           —— 不需要摆特定子力, 也能证明过滤在起作用。
        */
        Chess fresh;
        fresh.reset();
        std::vector<Step *> legal;
        fresh.sample(Stone::COLOR_RED, legal);
        const int legalCount = (int)legal.size();
        Steps::instance().put(legal);
        std::printf("    开局红方合法着法数 = %d (44 = 无过滤时的全部形状着法)\n", legalCount);
        CHECK(legalCount > 0 && legalCount <= 44,
              "合法着法数不超过 44 (过滤在起作用, 且没有把开局封死)");
    }

    /* ================================================================
     *  [2.7i] 红兵过河后能横走 (2026-09 用户报障的规则面)
     * ================================================================
     *
     * 报障: "红方中央的兵走过河后, 吃掉黑方中间的卒后不能左右行走"。
     *
     * 先把**规则**钉住 (报障当时我读了一遍代码, 认为规则是对的; 这条断言把它变成事实):
     *   * 红兵未过河 (x >= 5): 只能前进 (x-1), 不能横走;
     *   * 红兵过河 (x <= 4)  : 可以横走 (y 方向 ±1), 也可以继续前进;
     *   * 无论何时都**不能后退** (x+1)。
     * 判据用引擎自己的 `Bing::tryMoveTo` (与走子路径同源, 不另写一套判定)。
     *
     * 如果这条绿而用户仍然走不动, 那就不是规则问题, 而是"点击没被处理/没选中/
     * 被状态拦掉" —— 那类问题由 chessboard.cpp 里的 [dbg] 日志回答 (见那里的一段说明)。
     */
    std::printf("\n[2.7i] 红兵过河后的走法规则\n");
    {
        Chess c;
        c.reset();
        /* 找红方中央那个兵 (x=6, y=4 是象棋初始的"中兵") */
        Stone *bing = nullptr;
        for (int i = Stone::ID_RED; i < Stone::ID_RED_END; i++) {
            Stone *s = c.m_children[i];
            if (s != nullptr && s->alive && s->type == Stone::TYPE_BING
                && s->pos.x == 6 && s->pos.y == 4) {
                bing = s;
                break;
            }
        }
        if (bing == nullptr) {
            std::printf("    [跳过] 没找到初始的中兵 (子力编号/初始摆法变了?)\n");
            CHECK(false, "能找到红方中兵 (x=6,y=4) —— 找不到说明初始摆法变了");
        } else {
            /* ① 未过河: 前进可以, 横走不行, 后退不行 */
            CHECK(bing->tryMoveTo(Pos(5, 4)), "未过河时前进 (x-1) 合法");
            CHECK(!bing->tryMoveTo(Pos(6, 3)), "未过河时**不能**横走");
            CHECK(!bing->tryMoveTo(Pos(7, 4)), "任何时候都**不能**后退 (x+1)");

            /*
               ② 把兵挪到过河位置 (x=4, 已过河) 再验 —— 这是报障的具体局面:
                  红兵吃掉黑卒之后停在河对岸。
               直接改 pos/m_map 是"摆局面"的最简方式; 本测试只读走法规则, 不做搜索。
            */
            c.m_map[bing->pos] = nullptr;
            bing->pos = Pos(4, 4);
            c.m_map[bing->pos] = bing;
            std::printf("    把中兵摆到过河位置 (4,4) 后:\n");
            CHECK(bing->tryMoveTo(Pos(3, 4)), "过河后仍可前进 (x-1)");
            CHECK(bing->tryMoveTo(Pos(4, 3)), "过河后**可以**横走 (y-1) —— 报障的核心");
            CHECK(bing->tryMoveTo(Pos(4, 5)), "过河后**可以**横走 (y+1) —— 报障的核心");
            CHECK(!bing->tryMoveTo(Pos(5, 4)), "过河后**不能**后退 (x+1)");

            /* ③ 走到对方底线 (x=0) 之后: 只能横走 —— 这是象棋规则, 不是 bug */
            c.m_map[bing->pos] = nullptr;
            bing->pos = Pos(0, 4);
            c.m_map[bing->pos] = bing;
            /*
               ⚠ 这里**不能**拿 `Pos(-1,4)` 当"再前进 (棋盘外)"来断言:
               `tryMoveTo` **不做棋盘边界校验** (那是 sample()/getPossibleSteps 那一层的事),
               而 x 从 0 变成 -1 并不是"后退"、delta 恰好是 1, 于是它返回 true ——
               第一版就是这么误报成失败的 (同一件事在 `test/probe_pawn_rule_main.cpp` 的
               [3] 节里有逐字说明)。要测"到底线后不能再过河", 正确形状是:
               只剩横走, 而**往回走 (x+1) 才是那个不合法方向**。
            */
            CHECK(!bing->tryMoveTo(Pos(1, 4)), "到底线后不能往回走 (x+1) —— 那是后退");
            CHECK(bing->tryMoveTo(Pos(0, 3)) && bing->tryMoveTo(Pos(0, 5)),
                  "到底线后只能横走 (这是象棋规则: 兵到底后只能左右)");
            /* 还原, 免得影响后面的小节 */
            c.m_map[bing->pos] = nullptr;
            bing->pos = Pos(6, 4);
            c.m_map[bing->pos] = bing;
        }
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
    std::printf("\n[2.11] 模型自检 (每个 agent 都要有一份, 且可重复)\n");
    {
        const ChessBoard::AgentType all[] = {
            ChessBoard::AGENT_ALPHABETA,
            /* Alpha-Beta 三档弱等级 (2026-09): 它们也必须有一份自检报告, 而且报告里
               印的深度必须是**该档位的**深度 —— 这一条专门抓"自检文字写着深度 4、
               实际按别的深度在下"那类假读数 (abagent.cpp 里原来那句手抄的
               "(界面 AB_DEPTH = 4)" 正是这么来的)。 */
            ChessBoard::AGENT_AB_L1, ChessBoard::AGENT_AB_L2, ChessBoard::AGENT_AB_L3,
            ChessBoard::AGENT_MCTS,
            ChessBoard::AGENT_PG,        ChessBoard::AGENT_DQN,
            ChessBoard::AGENT_PPOMCTS,   ChessBoard::AGENT_DQNMCTS,
            ChessBoard::AGENT_EVAB,      ChessBoard::AGENT_SACAZ,
            ChessBoard::AGENT_SACAZ_MOE, ChessBoard::AGENT_DQNAB,
            ChessBoard::AGENT_PPOMCTS_MLP,
            /* 59e5233 行为还原版 (**独立类** SACAZLegacyAgent), 2026-09 新增 */
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
            /*
               Alpha-Beta 各档: 报告必须体现**本档的深度**, 而且**不许**出现"界面
               AB_DEPTH = 4"这种手抄常量 —— 三档加进来之后, 那句话会在同一行里
               自相矛盾 (深度 2 与 AB_DEPTH = 4 并排)。纯搜索这一支的自检必然非空
               (它每次现场构造一个 ABAgent, 不依赖权重文件), 所以这里的 CHECK 是
               实打实会跑的, 不是"没有实例就跳过"。
            */
            const int d = ChessBoard::abDepthOf(t);
            if (d > 0) {
                char want[32];
                std::snprintf(want, sizeof(want), "深度 %d", d);
                CHECK(a.find(want) != std::string::npos,
                      "自检报告里印的是本档的实际搜索深度");
                CHECK(a.find("AB_DEPTH") == std::string::npos,
                      "自检报告里不再手抄 chessboard.cpp 的 AB_DEPTH 常量");
            }
        }
        /* Alpha-Beta 四档 (含对照组) 都必然有报告: 深度是构造参数, 不依赖权重 */
        for (ChessBoard::AgentType t : all) {
            if (ChessBoard::abDepthOf(t) > 0) {
                CHECK(!board.getAgentSelfCheck(t).empty(),
                      "Alpha-Beta 各档都有自检报告 (纯搜索, 不需要权重文件)");
            }
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
               59e5233 行为还原版 (**独立类** SACAZLegacyAgent)。这一条特别值得跑: 它的
               **训练 clone 是另一个类**, 而两支的参数结构完全相同 ⇒ 如果后台训练那一支
               忘了建**独立类**, save/load 一样成功、损失一样上报, 界面上**看不出任何异常**
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
              "口径表: 59e5233 还原版也有 (它是**独立的类** SACAZLegacyAgent, 但用同一套学习口径)");
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

    /*
       ================================================================
       [2.20] 吃子行为: "该吃的时候吃了吗" (O1, 2026-09)
       ================================================================
       为什么这条要进单测: 界面上"吃子无动于衷"这句话**只有这一个读数**能回答
       (奖励曲线不是合适的仪器 —— 单次吃一个車在学习口径下只占纵轴 1.3 px, 换引擎口径
       也只是 3.9 px, 而终局会从 26 px 缩到 8 px)。读数一旦数错, 用户看到的结论就会反过来,
       而它在界面上长得完全正常 —— 这正是本工程反复吃亏的那一类错误。

       分两段钉:
         (1) **格式化**(纯函数, 不跑棋局): 分母为 0 必须说"无机会"而不是印 0.0%,
             因为"一次机会都没遇到"与"有机会一次都没吃"是两件完全不同的事;
         (2) **端到端**: 真打一局, 断言计数不变量 (chosen <= avail) 与"每局明细里有这一项"。
    */
    std::printf("\n[2.20] 吃子行为: 该吃的时候吃了吗 (界面比分行/逐局明细的数据源)\n");
    {
        /* ---- (1) 格式化: 三个分支 (无机会 / 有机会吃到了 / 有机会没吃) ---- */
        ChessBoard::MatchStats ms;
        ms.agentA = QStringLiteral("A");
        ms.agentB = QStringLiteral("B");
        /* 一个样本都没有: 两边都必须是"无机会", 而且**不许**出现 0.0% */
        const QString noSample = ms.captureLine();
        std::printf("    无样本 : %s\n", noSample.toUtf8().constData());
        CHECK(noSample.contains(QStringLiteral("无机会")), "分母为 0 -> 说\"无机会\"");
        CHECK(!noSample.contains(QStringLiteral("0.0%")),
              "分母为 0 时**不印 0.0%** (\"没机会\"与\"有机会不吃\"不是同一件事)");
        /* A 4 次机会吃到 1 次; B 一次都没碰上 */
        ms.capAvailA = 4;
        ms.capChosenA = 1;
        const QString oneSide = ms.captureLine();
        std::printf("    A 1/4  : %s\n", oneSide.toUtf8().constData());
        CHECK(oneSide.contains(QStringLiteral("A 1/4=25.0%")), "分子/分母与百分比都印出来");
        CHECK(oneSide.contains(QStringLiteral("B 无机会")), "只有 B 无机会时只对 B 说无机会");
        /* 比分那一行在没有样本时**不附**吃子段 (免得每一场都挂一句无信息的话) */
        ChessBoard::MatchStats empty;
        empty.agentA = QStringLiteral("A");
        empty.agentB = QStringLiteral("B");
        CHECK(!empty.summary().contains(QStringLiteral("该吃时吃到")),
              "一个吃子样本都没有时, 比分行不附吃子段");
        CHECK(ms.summary().contains(QStringLiteral("该吃时吃到")),
              "有样本时比分行**带上**这一项 (用户在对弈过程中就盯着这一行)");

        /* ---- (2) 端到端: 真打一局, 计数必须自洽 ---- */
        board.setMaxPliesPerGame(60);
        const ChessBoard::MatchStats stc =
            board.matchAgents(ChessBoard::AGENT_ALPHABETA, ChessBoard::AGENT_ALPHABETA, 2);
        std::printf("    两局: %s\n", stc.summary().toUtf8().constData());
        std::printf("          吃到手 vs 对手: %.2f / %.2f  (材质原值, 不含将)\n",
                    stc.matGainedA, stc.matGainedB);
        CHECK(stc.capChosenA <= stc.capAvailA && stc.capChosenB <= stc.capAvailB,
              "分子不可能超过分母 (chosen <= avail, 两边都要成立)");
        CHECK(stc.capAvailA >= 0 && stc.capAvailB >= 0, "计数非负");
        CHECK(stc.matGainedA >= 0.0 && stc.matGainedB >= 0.0,
              "吃到的材质非负 (吃将不计入: value_jiang=1000 会把这个数顶爆)");
        /*
           两局 AB vs AB 必然出现吃子机会 (开局几步就有兑现交换), 所以这一条不是空断言;
           若真的一次都没出现, 那也是**值得当场知道**的事 (说明这个读数的分母口径写错了)。
        */
        CHECK(stc.capAvailA + stc.capAvailB > 0,
              "两局里至少出现过一次吃子机会 (否则这个读数等于没测到东西)");
        CHECK(stc.log.contains(QStringLiteral("该吃时吃到")),
              "每局明细那一行里带上本局的吃子数 (逐局才能看出它是不是均匀分布)");
    }

    /* ---------------------------------------------------------------- 3. 中止 */
    std::printf("\n[3] 中止: 请求 50 局, 跑一会儿后叫停\n");
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
