/*
 * test_dqnab_main.cpp - DQNABAgent (AB 当 DQN 的 planning head) 的验证
 *
 * 这个 agent 里真正容易错、而且错了不会报错的，是下面这几条，测试就盯这几条：
 *
 *   [1] 编码的镜像对称。规范视角是"一个网络服务红黑双方"的全部依据 —— 如果
 *       "把棋盘左右镜像 + 交换红黑"之后编码不变这条不成立，那 Q_red 与 Q_black
 *       就是两个不同的函数，零和对称（用户草案里那个必须满足的约束）直接失效。
 *   [2] 动作索引的双射与掩码：8100 无碰撞、镜像后下标也镜像、非法列永远不参与
 *       选点/Q 基线（DQN 那条"未掩码 argmax 取到未训练槽位"的缺陷在这里不能存在）。
 *   [3] **Dueling 双头的解析梯度 vs 有限差分**。Q = V + A − mean_legal(A) 对 V 与 A 的
 *       偏导（1 与 δ−1/n）是最容易写反的一处，而且写反了训练照样跑、只是学不到东西。
 *   [4] AB 规划的**战术能力**：给定"一步杀"的局面，搜索必须 100% 找到那一手，而
 *       纯 Q-argmax（只有网络、没有规划）只能是随机水平。这是"planning head 值不值"
 *       的直接证据，而且它不依赖训练（杀棋是终局判定，与 V 学得准不准无关）。
 *   [5] 学得动：固定经验 + 已知 TD 目标（done=true ⇒ y ≡ reward），Q 必须朝它走。
 *   [6] 自对弈链路 + 稀疏 MoE 路由健康 + 数值稳定，以及**棋盘必须逐字段复原**
 *       （search/探索 都在真棋盘上试走，这是本工程的硬性不变量）。
 *   [7] 代价：两种骨干的 ms/前向、ms/步、实际搜到的深度 —— "深度是预算的函数"这条
 *       设计要靠数字说话。
 *   [8] 存取往返（三个文件 trunk/v/a）。
 *   [9] 手工评估锚：探针构造对棋盘零副作用（训练入口也不许动棋局）、预训练真的把
 *       V 拉近手工锚（gap 下降）、门控在**蓄意毒化的更新**上回滚权重 —— 并且有
 *       "关掉门控就会被改坏"的对照，否则"回滚成功"可能只是"更新根本没生效"。
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "chess.h"
#include "chessstate.h"
#include "dqnabagent.h"
#include "rl/cpuinfo.hpp"
#include "rl/util.hpp"
#include "stone.h"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                               \
            ++g_failed;                                              \
            std::printf("  [FAIL] %s\n", (msg));                     \
        }                                                            \
    } while (0)

using Backbone = DQNABAgent::Backbone;

/* 棋盘摘要: 用来断言"搜索/探索对棋盘零副作用" */
static std::string digest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[i];
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x) + "," + std::to_string(st->pos.y) + ";";
        s += std::to_string((int)st->type) + std::to_string((int)st->color) + ";";
    }
    s += "side=" + std::to_string(c.sideToMove);
    s += "clock=" + std::to_string(c.halfMoveClock);
    s += "hist=" + std::to_string((int)c.history.size());
    s += "hash=" + std::to_string(c.computeHash());
    return s;
}

/*
   把棋盘左右镜像并交换红黑 (只交换 `color` 字段)。
   ⚠ **只对"纯几何/编码"检查有效**: `Chess::isInCheck` 是按 `&redJiang / &blackJiang`
   这两个**对象**取将帅的 (chess.cpp 里 `Stone *jiang = (color == RED) ? &redJiang : &blackJiang;`),
   而这里只改 color 字段、不搬对象 —— 于是镜像之后 `isInCheck(BLACK)` 看的还是原来那个
   帅对象。任何会走引擎规则的函数 (isInCheck / getResult / sample / isLegalMove)
   都**不能**建立在这样镜像出来的局面上。
   要造"真正的镜像局面", 用 setupSparse 按交换后的颜色重新摆 (它按 (type,color) 找棋子,
   于是 id 与 color 始终自洽), 再把 sideToMove 翻过来 —— 见 testEncoding 与 [10]。
*/
static void mirrorBoard(Chess &c)
{
    for (int i = 0; i < 32; i++) {
        Stone *s = c.m_children[i];
        if (s == nullptr) { continue; }
        s->pos.x = 9 - s->pos.x;
        s->color = (s->color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    /* m_map 重建 (用 StoneMap::clear + Pos 索引; operator[] 只接受 Pos) */
    c.m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = c.m_children[i];
        if (s != nullptr && s->alive) {
            c.m_map[Pos(s->pos.x, s->pos.y)] = s;
        }
    }
    c.sideToMove = (c.sideToMove == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                      : Stone::COLOR_RED;
}

static double nowMs()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e6;
}

/* 摆局面的工具 (实现在文件后面, 这里先声明 —— [1] 的镜像/将军平面也要用它) */
struct SparsePiece { int type; int color; int cell; };
static bool setupSparse(Chess &c, const std::vector<SparsePiece> &pieces);

/*
   把**同一个棋盘对象**重摆成"左右镜像 + 交换红黑"的局面, 并把走子方翻过来。
   为什么不直接用 mirrorBoard (只交换 color 字段): `Chess::isInCheck` 按
   `&redJiang / &blackJiang` 这两个**对象**取将帅, 只改 color 字段的话引擎会认错人
   (实测镜像后 isInCheck(BLACK) 看的还是原来那个帅 ⇒ 将军平面差 1.0)。
   用 setupSparse 按交换后的颜色重摆, id 与 color 始终自洽。
   这样"同一个 agent(同一套权重)在同一个棋盘对象上编码两次"才是干净的对照。
*/
static bool setupMirrored(Chess &c, const std::vector<SparsePiece> &src, int halfMoveClock)
{
    std::vector<SparsePiece> swapped;
    swapped.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); i++) {
        SparsePiece p;
        p.type = src[i].type;
        p.color = (src[i].color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
        p.cell = (9 - src[i].cell / 9) * 9 + (src[i].cell % 9);   /* x -> 9-x */
        swapped.push_back(p);
    }
    if (!setupSparse(c, swapped)) { return false; }
    c.sideToMove = Stone::COLOR_BLACK;
    c.halfMoveClock = halfMoveClock;
    return true;
}

/*
 * ============================================================
 *  [1] 编码: 棋子平面 + 规则/阶段平面
 * ============================================================ */
static void testEncoding()
{
    std::printf("\n[1] 编码: %d 维 = 14 棋子平面 + 5 规则/阶段平面 (规范视角)\n",
                DQNABAgent::STATE_DIM);

    Chess c;
    c.reset();
    DQNABAgent agent(c, 8, 0.99f, 0.001f, Backbone::Mlp);

    CHECK(DQNABAgent::STATE_DIM == 19 * 90, "STATE_DIM = 19 x 90 = 1710");
    CHECK(DQNABAgent::ACTION_DIM == 8100, "ACTION_DIM = 90 x 90 = 8100 (双射)");

    /*
       把**平面约定**钉住: 棋子平面下标 = type*2 + (是己方 ? 0 : 1) —— 与 PPO+MCTS、
       以及公共实现 src/chessstate.h 的 `encodePiecePlanes` 完全一致。
       这条以前没测过, 而"哪一类子落在哪个平面"正是最容易被两套约定悄悄分叉的地方
       (本 agent 原来是 type + 7*(对方), 与 PPO 的排列不同, 却没人看得出来)。
    */
    {
        Chess cc;
        cc.reset();
        DQNABAgent ag(cc, 8, 0.99f, 0.001f, Backbone::Mlp);
        RL::Tensor s(DQNABAgent::STATE_DIM, 1);
        ag.encodeStateFor(Stone::COLOR_RED, s);
        int checked = 0;
        for (int i = 0; i < 32; i++) {
            Stone *st = cc.m_children[i];
            if (st == nullptr || !st->alive || st->type < 0 || st->type >= 7) { continue; }
            const int isMine = (st->color == Stone::COLOR_RED) ? 0 : 1;
            const int plane  = st->type * 2 + isMine;
            const int cell   = ChessState::canonicalCell(st->pos.x, st->pos.y,
                                                         Stone::COLOR_RED);
            if (s[(std::size_t)(plane * DQNABAgent::CELLS + cell)] == 1.0f) { checked++; }
        }
        CHECK(checked == 32, "32 个棋子都落在 type*2+(是己方?...:...) 这个约定的平面上");
    }

    RL::Tensor a(DQNABAgent::STATE_DIM, 1);
    agent.encodeStateFor(Stone::COLOR_RED, a);

    /* 结构: 14 个棋子平面是 one-hot (32 个 1), 5 个规则/阶段平面是常数铺满 */
    int nonZeroPieces = 0;
    for (int p = 0; p < DQNABAgent::PIECE_PLANES; p++) {
        for (int cell = 0; cell < DQNABAgent::CELLS; cell++) {
            if (a[p * DQNABAgent::CELLS + cell] != 0.0f) { nonZeroPieces++; }
        }
    }
    CHECK(nonZeroPieces == 32, "14 个棋子平面上恰好 32 个 1 (32 个棋子)");

    const int constPlanes[5] = { DQNABAgent::PLANE_MATERIAL, DQNABAgent::PLANE_TEMPO,
                                 DQNABAgent::PLANE_HALFMOVE, DQNABAgent::PLANE_REPEAT,
                                 DQNABAgent::PLANE_CHECK };
    const float pv[5] = { a[constPlanes[0] * DQNABAgent::CELLS],
                          a[constPlanes[1] * DQNABAgent::CELLS],
                          a[constPlanes[2] * DQNABAgent::CELLS],
                          a[constPlanes[3] * DQNABAgent::CELLS],
                          a[constPlanes[4] * DQNABAgent::CELLS] };
    bool constPlane = true;
    for (int p = 0; p < 5; p++) {
        for (int cell = 1; cell < DQNABAgent::CELLS; cell++) {
            if (a[constPlanes[p] * DQNABAgent::CELLS + cell] != pv[p]) {
                constPlane = false;
            }
        }
    }
    std::printf("  5 个规则/阶段平面: 子力 %.4f, 总手数 %.4f, 无吃子 %.4f, "
                "重复 %.4f, 将军 %.4f\n",
                (double)pv[0], (double)pv[1], (double)pv[2], (double)pv[3], (double)pv[4]);
    CHECK(constPlane, "5 个规则/阶段平面都是全局常数 (该局面下 90 格同值)");
    CHECK(pv[0] > 0.9f && pv[0] <= 1.0f, "开局子力比例 ≈ 1");
    CHECK(pv[1] == 0.0f, "开局总手数比例 = 0");
    CHECK(pv[2] == 0.0f, "开局无吃子计数 = 0");
    CHECK(pv[3] == 0.0f, "开局未出现过重复 = 0");
    CHECK(pv[4] == 0.0f, "开局没人被将军 = 0");

    /*
       **镜像对称** (本 agent 最重要的一条): 把棋盘左右镜像 + 交换红黑之后,
       同一套"规范视角"编码必须**逐位不变** —— 这就是"一个网络服务红黑双方"的依据,
       所以 negamax 里 `val = −child` 的符号翻转与它自洽。

       注意: 这里必须用**没有历史**的局面 (直接摆子, 不落子) —— 重复次数平面是
       history 的函数, 而"只镜像棋盘"并不镜像 history (真实的对局里镜像的那一局
       连 history 一起镜像, 那个数自然相同)。用 setupSparse 摆局正好满足这一点,
       而且能摆出"被将军"的局面, 连将军平面一起测。
    */
    {
        /* 局面 A: 红车马 vs 黑车, 双方都不被将军 (黑将放 y=5, 避免"对脸将") */
        Chess cA;
        const std::vector<SparsePiece> piecesA = {
            { Stone::TYPE_JIANG, Stone::COLOR_RED,   9 * 9 + 4 },
            { Stone::TYPE_JIANG, Stone::COLOR_BLACK, 0 * 9 + 5 },
            { Stone::TYPE_CHE,   Stone::COLOR_RED,   7 * 9 + 1 },
            { Stone::TYPE_MA,    Stone::COLOR_RED,   5 * 9 + 6 },
            { Stone::TYPE_CHE,   Stone::COLOR_BLACK, 2 * 9 + 7 },
        };
        CHECK(setupSparse(cA, piecesA), "摆出局面 A (红车马 vs 黑车)");
        CHECK(!cA.isInCheck(Stone::COLOR_RED) && !cA.isInCheck(Stone::COLOR_BLACK),
              "局面 A 双方都不被将军");

        /* 局面 B: 红方被黑车照将 (测将军平面) —— 注意这也是一个合法可到的局面 */
        Chess cB;
        const std::vector<SparsePiece> piecesB = {
            { Stone::TYPE_JIANG, Stone::COLOR_RED,   9 * 9 + 4 },
            { Stone::TYPE_JIANG, Stone::COLOR_BLACK, 0 * 9 + 5 },
            { Stone::TYPE_CHE,   Stone::COLOR_BLACK, 9 * 9 + 8 },   /* 9 行上照将 */
        };
        CHECK(setupSparse(cB, piecesB), "摆出局面 B (黑车照将)");
        CHECK(cB.isInCheck(Stone::COLOR_RED), "局面 B 的红帅确实在被将军");

        Chess *cases[2] = { &cA, &cB };
        const std::vector<SparsePiece> *srcs[2] = { &piecesA, &piecesB };
        for (int k = 0; k < 2; k++) {
            Chess &cc = *cases[k];
            /* halfMoveClock 设成非零, 顺带测"无吃子计数"平面也镜像不变 */
            cc.halfMoveClock = 37;
            DQNABAgent ag(cc, 8, 0.99f, 0.001f, Backbone::Mlp);
            RL::Tensor orig(DQNABAgent::STATE_DIM, 1);
            ag.encodeStateFor(Stone::COLOR_RED, orig);
            const double hexOrig = orig[DQNABAgent::PLANE_HALFMOVE * DQNABAgent::CELLS];
            const double chkOrig = orig[DQNABAgent::PLANE_CHECK * DQNABAgent::CELLS];

            /* 同一个棋盘对象、同一个 agent (同一套权重) 上再编码一次镜像局面 */
            CHECK(setupMirrored(cc, *srcs[k], 37), "把同一个棋盘重摆成镜像局面");
            RL::Tensor mirrored(DQNABAgent::STATE_DIM, 1);
            ag.encodeStateFor(Stone::COLOR_BLACK, mirrored);

            double maxDiff = 0.0;
            for (std::size_t i = 0; i < orig.size(); i++) {
                maxDiff = std::max(maxDiff, std::fabs((double)orig[i] - (double)mirrored[i]));
            }
            std::printf("  局面 %c: 镜像后编码最大逐元素差 = %.3e "
                        "(无吃子平面 %.2f, 将军平面 %.1f -> %.1f)\n",
                        (k == 0) ? 'A' : 'B', maxDiff, hexOrig, chkOrig,
                        (double)mirrored[DQNABAgent::PLANE_CHECK * DQNABAgent::CELLS]);
            CHECK(maxDiff == 0.0, "镜像(左右翻转+交换红黑)后规范编码逐位不变");
            /* 将军平面必须是镜像不变的: B 里红方被将, 镜像后就是黑方被将 */
            CHECK(chkOrig == mirrored[DQNABAgent::PLANE_CHECK * DQNABAgent::CELLS],
                  "将军平面在镜像下不变 (被将的一方换成了另一方)");
        }
    }
}

/* ============================================================
 *  [2] 动作索引: 双射 / 镜像 / 掩码
 * ============================================================ */
static void testActionIndex()
{
    std::printf("\n[2] 动作索引: 双射无碰撞 + 镜像 + 掩码\n");

    Chess c;
    c.reset();
    DQNABAgent agent(c, 8, 0.99f, 0.001f, Backbone::Mlp);

    std::vector<Step*> legal;
    std::vector<int> idx;
    agent.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);

    CHECK(!idx.empty(), "开局红方有合法着法");
    std::vector<int> sorted = idx;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t uniq = (std::size_t)(std::unique(sorted.begin(), sorted.end())
                                           - sorted.begin());
    std::printf("  合法着法 %zu 个 -> 唯一索引 %zu 个\n", idx.size(), uniq);
    CHECK(uniq == idx.size(), "同一局面的合法着法**下标互不相同** (双射, 无哈希碰撞)");
    bool inRange = true;
    for (int v : idx) {
        if (v < 0 || v >= DQNABAgent::ACTION_DIM) { inRange = false; }
    }
    CHECK(inRange, "所有下标都落在 [0, 8100)");

    /*
       镜像等价的着法必须给出同一个下标。这条**不需要**真的去镜像棋盘:
       规范视角的含义就是 "红方着法 (a→b)" 与 "黑方着法 (9−a→9−b)" 在各自的坐标系里
       是同一格到同一格, 所以直接对 Step 做一次坐标翻转就够了 (纯函数, 不碰棋盘状态)。
    */
    {
        int compared = 0;
        int same = 0;
        for (Step *s : legal) {
            const int idxRed = DQNABAgent::actionIdxOf(*s, Stone::COLOR_RED);
            Step mirrored = *s;
            mirrored.pos.x = 9 - s->pos.x;
            mirrored.nextPos.x = 9 - s->nextPos.x;
            const int idxBlack = DQNABAgent::actionIdxOf(mirrored, Stone::COLOR_BLACK);
            compared++;
            if (idxRed == idxBlack) { same++; }
        }
        std::printf("  镜像着法下标一致: %d / %d\n", same, compared);
        CHECK(compared > 0 && same == compared,
              "红方着法与其镜像(黑方着法)给出同一个动作下标");
    }

    /* 掩码: 非法列不参与 */
    RL::Tensor mask(DQNABAgent::ACTION_DIM, 1);
    std::vector<Step*> ls;
    std::vector<int> is;
    agent.getLegalActions(Stone::COLOR_RED, ls, is, mask);
    Steps::instance().put(ls);
    int ones = 0;
    for (int i = 0; i < DQNABAgent::ACTION_DIM; i++) {
        if (mask[i] > 0.5f) { ones++; }
    }
    CHECK(ones == (int)is.size(), "掩码里的 1 的个数 = 合法着法数");

    /* 非法动作永远不该被选中 */
    int illegalPicks = 0;
    for (int k = 0; k < 6; k++) {
        std::printf("    selectMove #%d ...\n", k);
        const Step s = agent.selectMove(Stone::COLOR_RED, 0.0f);
        std::printf("    -> valid=%d\n", (int)s.valid);
        if (!s.valid || !c.isLegalMove(Stone::COLOR_RED, &s)) { illegalPicks++; }
    }
    CHECK(illegalPicks == 0, "selectMove 给出的都是合法走法");
}

/* ============================================================
 *  [3] Dueling 双头的解析梯度 vs 有限差分
 * ============================================================ */
static void testDuelingGradient()
{
    std::printf("\n[3] Dueling 头: 解析梯度 vs 中心差分\n");

    Chess c;
    c.reset();
    /* 小骨干 + 小隐层: 有限差分要跑很多次前向, 必须便宜 */
    DQNABAgent agent(c, 8, 0.99f, 0.001f, Backbone::Mlp);

    std::vector<Step*> legal;
    std::vector<int> idx;
    agent.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);

    const int taken = idx[idx.size() / 2];
    DQNABAgent::Sample s;
    /* Sample 里的张量默认是空的 —— 必须先按 STATE_DIM 分配再编码 (否则越界写) */
    s.state = RL::Tensor(DQNABAgent::STATE_DIM, 1);
    s.nextState = RL::Tensor(DQNABAgent::STATE_DIM, 1);
    agent.encodeStateFor(Stone::COLOR_RED, s.state);
    s.legalIdx = idx;
    s.action = taken;
    s.reward = 0.5f;
    s.done = true;                 /* done ⇒ Planned 标签就是 reward, 不触发搜索 */
    s.label = s.reward;

    /* 解析梯度 (只累积, 不调优化器) */
    std::printf("    accumulateGrad ...\n");
    agent.accumulateGrad(s);
    std::printf("    ok\n");

    /* 一个"算 Q_taken"的小工具: 走同一套前向路径 */
    RL::Tensor state(DQNABAgent::STATE_DIM, 1);
    agent.encodeStateFor(Stone::COLOR_RED, state);
    auto qTaken = [&]() {
        double v = 0.0;
        std::vector<double> q;
        agent.evaluateNode(state, idx, v, q, true);
        for (std::size_t i = 0; i < idx.size(); i++) {
            if (idx[i] == taken) { return q[i]; }
        }
        return 0.0;
    };
    const double y = 0.5;
    /* 损失口径必须与 accumulateGrad 里的一致: L = (Q - y)^2 (Loss::MSE::df = 2(out-target)) */
    auto loss = [&]() {
        const double d = qTaken() - y;
        return d * d;
    };

    RL::iFcLayer *vOut = dynamic_cast<RL::iFcLayer*>(agent.m_vHead[agent.m_vHead.size() - 1]);
    RL::iFcLayer *aOut = dynamic_cast<RL::iFcLayer*>(agent.m_aHead[agent.m_aHead.size() - 1]);
    CHECK(vOut != nullptr && aOut != nullptr, "两个头的输出层都是 iFcLayer");

    const double eps = 1e-3;
    double worstRel = 0.0;
    double worstAbs = 0.0;
    int compared = 0;

    /*
       对**某一个权重元素**做中心差分: (L(w+ε) − L(w−ε)) / 2ε 与解析梯度比。
       k = row * rowStride + col, rowStride = w.sizes[0] (= 输入维); 越界的组合直接跳过。
    */
    auto checkElem = [&](RL::iFcLayer *layer, int row, int col) {
        const std::size_t rowStride = (std::size_t)layer->w.sizes[0];
        if (row < 0 || (std::size_t)row >= (std::size_t)layer->w.sizes[1]) { return; }
        if (col < 0 || (std::size_t)col >= rowStride) { return; }
        const std::size_t k = (std::size_t)row * rowStride + (std::size_t)col;
        if (k >= layer->w.size() || k >= layer->g.w.size()) { return; }
        const float orig = layer->w.val[k];
        layer->w.val[k] = orig + (float)eps;
        const double lp = loss();
        layer->w.val[k] = orig - (float)eps;
        const double lm = loss();
        layer->w.val[k] = orig;
        const double numeric = (lp - lm) / (2.0 * eps);
        const double analytic = (double)layer->g.w.val[k];
        const double denom = std::max(1e-6, std::max(std::fabs(numeric), std::fabs(analytic)));
        worstRel = std::max(worstRel, std::fabs(numeric - analytic) / denom);
        worstAbs = std::max(worstAbs, std::fabs(numeric - analytic));
        compared++;
    };

    /* V 头输出层: 1 个输出 x 前 4 个输入权重 */
    for (int col = 0; col < 4; col++) { checkElem(vOut, 0, col); }
    /* A 头输出层: 被走那一行 / 另一个合法行 / 一个非法行, 各取前 3 个输入权重 */
    {
        const int illegalAction = (taken + 4001) % DQNABAgent::ACTION_DIM;
        bool legal0 = false;
        for (int v : idx) { if (v == illegalAction) { legal0 = true; } }
        const int rows[3] = {taken, idx[0], legal0 ? idx[1] : illegalAction};
        for (int r = 0; r < 3; r++) {
            for (int col = 0; col < 3; col++) { checkElem(aOut, rows[r], col); }
        }
    }

    std::printf("  比对 %d 个权重元素: 最大相对差 = %.3e, 最大绝对差 = %.3e\n",
                compared, worstRel, worstAbs);
    CHECK(worstRel < 5e-3 || worstAbs < 1e-5,
          "Dueling 双头的解析梯度 = 中心差分 (V 与 A 的偏导写对了)");

    /*
       非法列的梯度必须**恰好为 0**: Q 只在合法集上定义, 非法槽位不该被训练
       (这是 DQN 那条"未掩码 argmax 取未训练槽位"缺陷的对偶面)。
    */
    {
        const std::size_t rowStride = (std::size_t)aOut->w.sizes[0];
        const int illegalAction = (taken + 4001) % DQNABAgent::ACTION_DIM;
        bool isLegalAction = false;
        for (int v : idx) {
            if (v == illegalAction) { isLegalAction = true; }
        }
        if (!isLegalAction) {
            double mx = 0.0;
            for (std::size_t c = 0; c < rowStride; c++) {
                mx = std::max(mx, std::fabs((double)aOut->g.w.val[(std::size_t)illegalAction * rowStride + c]));
            }
            std::printf("  非法动作 %d 那一行的 |dL/dw|max = %.3e\n", illegalAction, mx);
            CHECK(mx == 0.0, "非法列的头权重梯度**恰好**为 0");
        }
    }
}

/* ============================================================
 *  [4] AB 规划的战术能力: 一步杀
 *
 *  随机自对弈几乎走不出"一步杀"的局面 (实测 8 次随机走到 60 手, 一个都没有),
 *  所以这里**直接摆局面**: 在空棋盘上放 红帅 + 两个红车 + 黑将, 穷举两个车的位置,
 *  用 getResult 验证"红方确实有一步杀"才收下 —— 判据来自引擎本身, 不是我的假设。
 * ============================================================ */
/* (SparsePiece / setupSparse 的声明在文件开头 —— [1] 也要用) */

/* 把棋盘清空, 只放 pieces 里指定的子 (每种 (type,color) 取第一个实例), 轮到红方 */
static bool setupSparse(Chess &c, const std::vector<SparsePiece> &pieces)
{
    c.reset();
    for (int i = 0; i < 32; i++) {
        if (c.m_children[i] != nullptr) { c.m_children[i]->alive = false; }
    }
    c.m_map.clear();
    std::vector<bool> used(32, false);
    for (std::size_t k = 0; k < pieces.size(); k++) {
        Stone *found = nullptr;
        for (int i = 0; i < 32; i++) {
            Stone *s = c.m_children[i];
            if (s == nullptr || used[(std::size_t)i]) { continue; }
            if (s->type == pieces[k].type && s->color == pieces[k].color) {
                found = s;
                used[(std::size_t)i] = true;
                break;
            }
        }
        if (found == nullptr) { return false; }
        found->alive = true;
        found->pos = Pos(pieces[k].cell / 9, pieces[k].cell % 9);
        c.m_map[found->pos] = found;
    }
    c.sideToMove = Stone::COLOR_RED;
    c.halfMoveClock = 0;
    c.history.clear();
    return true;
}

/* 轮到 color 走时, 有没有"走一步就把对方将死"的着法? 有就回填那一手 */
static bool hasMateInOne(Chess &c, int color)
{
    if (c.getResult(color) != Chess::RESULT_ONGOING) { return false; }
    std::vector<Step*> legal;
    c.sample(color, legal);
    bool found = false;
    for (Step *s : legal) {
        Step mv = *s;
        double dummy = 0.0;
        c.moveForward(&mv, dummy);
        const int after = c.getResult(c.sideToMove);
        c.moveBack(&mv, dummy);
        if (after != Chess::RESULT_ONGOING &&
            ((after == Chess::RESULT_RED_WIN) == (color == Stone::COLOR_RED))) {
            found = true;
            break;
        }
    }
    Steps::instance().put(legal);
    return found;
}

/* 走一手之后, 对方是否被将死 (用来判"这一步是不是杀着") */
static bool isMatingMove(Chess &c, int mover, const Step &s)
{
    if (!s.valid || !c.isLegalMove(mover, &s)) { return false; }
    Step mv = s;
    double dummy = 0.0;
    c.moveForward(&mv, dummy);
    const int after = c.getResult(c.sideToMove);
    c.moveBack(&mv, dummy);
    return after != Chess::RESULT_ONGOING &&
           ((after == Chess::RESULT_RED_WIN) == (mover == Stone::COLOR_RED));
}

static void testMateInOne()
{
    std::printf("\n[4] 规划头: 一步杀局面上的命中率 (AB 搜索 vs 纯 Q-argmax)\n");

    Chess c;
    c.reset();
    DQNABAgent agent(c, 32, 0.99f, 0.001f, Backbone::Mlp);
    agent.searchDepth = 3;
    agent.nodeBudget = 4000;

    const int want = 8;
    int positions = 0;
    int foundBySearch = 0;
    int foundByQ = 0;
    long long totalDepth = 0;
    long long totalNodes = 0;

    /* 红帅固定在 (9,4); 两个红车与黑将穷举 (宫格 3x3) */
    for (int r1 = 0; r1 < 90 && positions < want; r1++) {
        for (int r2 = r1 + 1; r2 < 90 && positions < want; r2++) {
            for (int kx = 0; kx < 3 && positions < want; kx++) {
                for (int ky = 3; ky < 6 && positions < want; ky++) {
                    const int kingCell = kx * 9 + ky;
                    if (r1 == kingCell || r2 == kingCell) { continue; }
                    if (r1 == 9 * 9 + 4 || r2 == 9 * 9 + 4) { continue; }

                    std::vector<SparsePiece> pieces = {
                        { Stone::TYPE_JIANG, Stone::COLOR_RED,   9 * 9 + 4 },
                        { Stone::TYPE_JIANG, Stone::COLOR_BLACK, kingCell },
                        { Stone::TYPE_CHE,   Stone::COLOR_RED,   r1 },
                        { Stone::TYPE_CHE,   Stone::COLOR_RED,   r2 },
                    };
                    if (!setupSparse(c, pieces)) { continue; }
                    if (c.isInCheck(Stone::COLOR_BLACK)) { continue; }  /* 已被将 = 不是"一步杀" */
                    if (!hasMateInOne(c, Stone::COLOR_RED)) { continue; }
                    positions++;

                    /* (a) AB 规划给不给那一手? */
                    const std::string before = digest(c);
                    const Step mv = agent.selectMove(Stone::COLOR_RED, 0.0f);
                    totalDepth += agent.m_lastDepth;
                    totalNodes += agent.m_lastNodes;
                    if (isMatingMove(c, Stone::COLOR_RED, mv)) { foundBySearch++; }
                    CHECK(digest(c) == before, "搜索之后棋盘逐字段复原");

                    /* (b) 纯 Q-argmax (没有规划) 给不给? —— 同一个网络, 只看 Q 的最大值 */
                    {
                        RL::Tensor st(DQNABAgent::STATE_DIM, 1);
                        agent.encodeStateFor(Stone::COLOR_RED, st);
                        std::vector<Step*> legal;
                        std::vector<int> idx;
                        agent.legalMoves(Stone::COLOR_RED, legal, idx);
                        double v = 0.0;
                        std::vector<double> q;
                        agent.evaluateNode(st, idx, v, q);
                        std::size_t bestI = 0;
                        for (std::size_t i = 1; i < q.size(); i++) {
                            if (q[i] > q[bestI]) { bestI = i; }
                        }
                        const Step qm = *legal[bestI];
                        Steps::instance().put(legal);
                        if (isMatingMove(c, Stone::COLOR_RED, qm)) { foundByQ++; }
                    }
                }
            }
        }
    }

    const double rateSearch = positions > 0 ? 100.0 * (double)foundBySearch / positions : 0.0;
    const double rateQ = positions > 0 ? 100.0 * (double)foundByQ / positions : 0.0;
    std::printf("  一步杀局面 %d 个:  AB 规划命中 %d (%.0f%%),  纯 Q-argmax 命中 %d (%.0f%%)\n",
                positions, foundBySearch, rateSearch, foundByQ, rateQ);
    if (positions > 0) {
        std::printf("  搜索平均深度 = %.1f, 平均节点 = %.0f\n",
                    (double)totalDepth / positions, (double)totalNodes / positions);
    }
    CHECK(positions >= 3, "构造出足够多的一步杀局面 (否则这条断言没有意义)");
    CHECK(foundBySearch == positions, "AB 规划在**每一个**一步杀局面都找到了杀棋");
    CHECK(foundByQ <= foundBySearch, "纯 Q-argmax 不可能比规划更好 (同一个网络)");
}

/* ============================================================
 *  [5] 学得动: 固定 TD 目标 (两臂配对 + **用默认学习率**)
 * ============================================================
 *  原来这一段的判据是"训练之后 Q 靠近 0.5", 而且用了 lr=0.02 (默认值的 20 倍)。
 *  实测 (`.r1build/dbg_learn_arm.cpp`, 同一个局面重复 64 遍的退化数据集):
 *
 *    lr=0.001 (默认): 后 50 次更新 Q 均值 +0.4877 / −0.4963, 波动 ±0.05, loss 0.003/0.009
 *    lr=0.005       : Q 均值 +0.4854 / −0.4879, 波动 ±0.13
 *    lr=0.02        : **周期 2 极限环** —— Q 在 −1.5 与 +0.99 之间摆, loss 在 0.006 与
 *                     1.31 之间跳 (逐次更新就翻一次), 均值 0.35 / −0.49
 *
 *  也就是说那个 20× 的学习率让"看某个时刻的 Q/loss"变成**赌它停在哪一相**:
 *  换一次编码排列 (随机特征随之改变) 就从"通过"变成"失败"。这不只是测试问题 ——
 *  clipGrad 让每步位移恒等于 lr (optimize.h:52), 于是大步长 + 退化数据必然过冲。
 *  所以这里改成: **默认 lr**, 两臂配对 (同 seed ⇒ 逐位相同的初始化), 一批标 +0.5、
 *  一批标 −0.5, 断言**两臂各自收敛到自己的目标** —— 这条判据不能靠运气通过
 *  (符号/目标接反都会红), 也不会因为换了编码就翻脸。
 * ============================================================ */
struct LearningArm {
    double qStart = 0.0;
    double qEnd = 0.0;
    double qMeanTail = 0.0;      /* 后 50 次更新的 Q 均值 (量"停在哪儿") */
    double lossMeanTail = 0.0;
    int updates = 0;
    int rolledBack = 0;
};

static LearningArm runLearningArm(unsigned seed, float label, float lr)
{
    Chess c;
    c.reset();
    RL::Random::setSeed(seed);          /* 两臂必须有**逐位相同**的初始权重 */
    DQNABAgent agent(c, 32, 0.99f, lr, Backbone::Mlp);
    agent.batchSize = 8;
    agent.targetSyncEvery = 8;

    std::vector<Step*> legal;
    std::vector<int> idx;
    agent.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);
    const int taken = idx[0];

    for (int i = 0; i < 64; i++) {
        DQNABAgent::Sample s;
        s.state = RL::Tensor(DQNABAgent::STATE_DIM, 1);
        agent.encodeStateFor(Stone::COLOR_RED, s.state);
        s.nextState = s.state;
        s.legalIdx = idx;
        s.nextLegalIdx = idx;
        s.action = taken;
        s.reward = label;
        s.done = true;           /* Planned 模式: 目标就是 label */
        s.label = label;
        agent.pushSample(std::move(s));
    }

    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    agent.encodeStateFor(Stone::COLOR_RED, st);
    auto q1 = [&]() {
        double v = 0.0;
        std::vector<double> q;
        agent.evaluateNode(st, idx, v, q);
        for (std::size_t i = 0; i < idx.size(); i++) {
            if (idx[i] == taken) { return q[i]; }
        }
        return 0.0;
    };

    LearningArm arm;
    arm.qStart = q1();
    double sumQ = 0.0, sumLoss = 0.0;
    int tail = 0;
    const int total = 200;
    for (int it = 0; it < total; it++) {
        if (agent.learnBatch(8, 1)) {
            arm.updates++;
            if (agent.lastUpdateRolledBack()) { arm.rolledBack++; }
        }
        if (it >= total - 50) {
            sumQ += q1();
            sumLoss += (double)agent.getLastTrainLoss();
            tail++;
        }
    }
    arm.qEnd = q1();
    arm.qMeanTail = tail > 0 ? sumQ / tail : 0.0;
    arm.lossMeanTail = tail > 0 ? sumLoss / tail : 0.0;
    return arm;
}

static void testLearning()
{
    std::printf("\n[5] 学习: 固定 TD 目标 (默认 lr=0.001, 两臂配对: +0.5 / −0.5)\n");

    const float lr = 0.001f;   /* 默认学习率: lr=0.02 会让这个实验进入周期 2 极限环 */
    const LearningArm plus  = runLearningArm(20240915u, +0.5f, lr);
    const LearningArm minus = runLearningArm(20240915u, -0.5f, lr);
    std::printf("  目标 +0.5: Q %.5f -> %.5f (后 50 次均值 %.5f, loss 均值 %.5f) "
                "更新 %d 回滚 %d\n",
                plus.qStart, plus.qEnd, plus.qMeanTail, plus.lossMeanTail,
                plus.updates, plus.rolledBack);
    std::printf("  目标 −0.5: Q %.5f -> %.5f (后 50 次均值 %.5f, loss 均值 %.5f) "
                "更新 %d 回滚 %d\n",
                minus.qStart, minus.qEnd, minus.qMeanTail, minus.lossMeanTail,
                minus.updates, minus.rolledBack);

    CHECK(std::fabs(plus.qStart - minus.qStart) < 1e-9,
          "两臂的初始权重逐位相同 (同 seed ⇒ 配对成立)");
    CHECK(plus.updates > 0 && minus.updates > 0, "两臂都真的更新了参数");
    CHECK(plus.rolledBack == 0 && minus.rolledBack == 0,
          "门控不误伤正常学习 (两臂各 200 次更新, 回滚 0 次)");
    CHECK(std::fabs(plus.qMeanTail - 0.5) < 0.1,
          "目标 +0.5 的那一臂收敛到 +0.5 (后 50 次均值误差 < 0.1)");
    CHECK(std::fabs(minus.qMeanTail + 0.5) < 0.1,
          "目标 −0.5 的那一臂收敛到 −0.5 (后 50 次均值误差 < 0.1)");
    CHECK(plus.lossMeanTail < 0.05 && minus.lossMeanTail < 0.05,
          "两臂的批平均 loss 都收敛到 0 附近");
}

/* ============================================================
 *  [6] 自对弈链路 + MoE 路由 + 数值 + 棋盘复原
 * ============================================================ */
static void testSelfPlayAndRouting()
{
    std::printf("\n[6] 自对弈链路 (MLP 骨干) + 交互式探索的零副作用\n");

    Chess c;
    c.reset();
    DQNABAgent agent(c, 32, 0.99f, 0.005f, Backbone::Mlp);
    agent.batchSize = 8;
    agent.searchDepth = 2;
    agent.nodeBudget = 400;
    agent.trainPlanDepth = 0;      /* 标签用单步 bootstrap: 快, 这条只验链路 */
    agent.replayCapacity = 512;

    /* 探索必须对棋盘零副作用 (硬性契约) */
    const std::string before = digest(c);
    const bool trained = agent.exploreAndTrain(Stone::COLOR_RED, 12);
    CHECK(digest(c) == before, "exploreAndTrain 之后棋盘逐字段复原 (含 sideToMove)");
    std::printf("  探索: %s\n", agent.getExploreInfo().c_str());
    CHECK(agent.replaySize() > 0, "探索把经验写进了回放池");
    (void)trained;

    const double loss = agent.trainSelfPlay(1, 24, false, 1.0f, 0.25f);
    std::printf("  自对弈 1 局(≤24 手): 池 = %zu, 学习步数 = %d, loss = %.4f\n",
                agent.replaySize(), agent.learnSteps(), loss);
    CHECK(agent.replaySize() > 8, "自对弈把经验写进了回放池");
    CHECK(agent.learnSteps() > 0, "自对弈过程中做过批更新");
    CHECK(std::isfinite(loss), "批平均损失是有限值");

    /* 数值稳定: Q / V 全有限 */
    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    agent.encodeState(st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    agent.legalMoves(c.sideToMove, legal, idx);
    Steps::instance().put(legal);
    double v = 0.0;
    std::vector<double> q;
    agent.evaluateNode(st, idx, v, q);
    bool finite = std::isfinite(v);
    for (double x : q) {
        if (!std::isfinite(x)) { finite = false; }
    }
    CHECK(finite, "训练之后 V 与 Q 全部是有限值 (无 NaN/inf)");
    CHECK(std::fabs(v) <= 1.0 + 1e-6, "V 头有界 (|V| <= 1, 与奖励量纲一致)");

    /* 稀疏 MoE 的路由: MLP 骨干没有 MoE, 这里应报 0 */
    std::vector<long long> usage;
    agent.moeUsage(usage);
    CHECK(agent.moeExpertCount() == 0, "MLP 骨干没有稀疏 MoE 层");
}

/* ============================================================
 *  [7] 稀疏 MoE + TB 专家骨干: 建/走/训/路由/代价
 * ============================================================ */
static void testTbBackbone()
{
    std::printf("\n[7] 稀疏 MoE + TB 专家骨干 (默认配置)\n");

    Chess c;
    c.reset();
    const double t0 = nowMs();
    DQNABAgent agent(c, 64, 0.99f, 0.001f, Backbone::SparseMoeTb);
    const double tBuild = nowMs() - t0;

    std::printf("  构建 %.1f s; 主干参数 = %lld, 头参数 = %lld (专家 %d, top-%d)\n",
                tBuild / 1000.0, agent.trunkParamCount(), agent.headParamCount(),
                agent.moeExpertCount(), agent.moeTopK());
    CHECK(agent.moeExpertCount() == DQNABAgent::MOE_EXPERTS, "专家数 = MOE_EXPERTS");
    CHECK(agent.moeTopK() == DQNABAgent::MOE_TOPK, "topK = MOE_TOPK");
    CHECK(agent.trunkParamCount() > 0, "主干参数量数得出来");

    /* 一次前向的代价 (含 V + A 头) */
    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    agent.encodeState(st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    agent.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);
    double v = 0.0;
    std::vector<double> q;
    const int fwdIters = 5;
    const double f0 = nowMs();
    for (int i = 0; i < fwdIters; i++) {
        agent.evaluateNode(st, idx, v, q);
    }
    const double msPerNode = (nowMs() - f0) / fwdIters;
    std::printf("  每个搜索节点 (主干 + V + 合法列 A) = %.2f ms\n", msPerNode);

    /* 走一步: 实测到达的深度与耗时 (深度是预算的函数) */
    agent.searchDepth = 6;
    agent.nodeBudget = 512;
    const std::string before = digest(c);
    const double m0 = nowMs();
    const Step mv = agent.selectMove(Stone::COLOR_RED, 0.0f);
    const double msMove = nowMs() - m0;
    std::printf("  selectMove (预算 512 节点): 深度 %d, 节点 %lld, %.1f ms/步; 合法 = %d\n",
                agent.m_lastDepth, agent.m_lastNodes, msMove,
                (int)(mv.valid && c.isLegalMove(Stone::COLOR_RED, &mv)));
    CHECK(digest(c) == before, "TB 骨干下搜索也不改动棋盘");
    CHECK(mv.valid && c.isLegalMove(Stone::COLOR_RED, &mv), "TB 骨干给出合法走法");
    /* 根节点本身与"软中止"会让节点数略超预算, 但必须是同一个量级 (而不是失控) */
    CHECK(agent.m_lastNodes <= 512 + 64, "节点数没有明显超出预算");
    CHECK(agent.m_lastDepth >= 2, "预算 512 下 TB 骨干至少能搜到 2 层");

    /* 训练一条 + 路由统计 */
    agent.replayCapacity = 128;
    agent.batchSize = 4;
    agent.trainPlanDepth = 0;
    agent.resetMoeUsage();
    agent.trainSelfPlay(1, 6, false, 1.0f, 0.25f);
    std::vector<long long> usage;
    agent.moeUsage(usage);
    long long total = 0;
    int unused = 0;
    for (std::size_t i = 0; i < usage.size(); i++) {
        total += usage[i];
        if (usage[i] == 0) { unused++; }
    }
    std::printf("  自对弈 6 手后: 池 = %zu, 学习步数 = %d, 专家使用 = [",
                agent.replaySize(), agent.learnSteps());
    for (std::size_t i = 0; i < usage.size(); i++) {
        std::printf("%lld%s", usage[i], (i + 1 < usage.size()) ? ", " : "]\n");
    }
    CHECK(total > 0, "TB 专家的稀疏 MoE 真的被前向过");
    CHECK(unused < (int)usage.size(), "至少有一个专家被选中 (路由没有全死)");
}

/* ============================================================
 *  [8] 存取往返
 * ============================================================ */
static void testSaveLoad()
{
    std::printf("\n[8] saveModel / loadModel 往返 (三个文件)\n");

    Chess c;
    c.reset();
    DQNABAgent a1(c, 16, 0.99f, 0.001f, Backbone::Mlp);

    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    a1.encodeStateFor(Stone::COLOR_RED, st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    a1.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);
    double v1 = 0.0;
    std::vector<double> q1;
    a1.evaluateNode(st, idx, v1, q1);

    const std::string prefix = "test_dqnab_tmp";
    CHECK(a1.saveModel(prefix), "saveModel 报告成功 (三个文件都写出来了)");

    Chess c2;
    c2.reset();
    DQNABAgent a2(c2, 16, 0.99f, 0.001f, Backbone::Mlp);
    CHECK(a2.loadModel(prefix), "loadModel 报告成功");
    double v2 = 0.0;
    std::vector<double> q2;
    a2.evaluateNode(st, idx, v2, q2);

    double dq = std::fabs(v1 - v2);
    for (std::size_t i = 0; i < q1.size() && i < q2.size(); i++) {
        dq = std::max(dq, std::fabs(q1[i] - q2[i]));
    }
    std::printf("  载入前后 V/Q 最大差 = %.3e\n", dq);
    CHECK(dq < 1e-5, "载入后的 V/Q 与保存前一致 (三个文件的读写顺序正确)");

    std::remove((prefix + "_trunk").c_str());
    std::remove((prefix + "_v").c_str());
    std::remove((prefix + "_a").c_str());
    CHECK(!a2.loadModel("no_such_prefix_xyz"), "载入不存在的权重返回 false");
}

/* ============================================================
 *  [9] 手工评估锚: 探针 / gap / 预训练 / 门控
 * ============================================================ */
/* 随机走若干手 (用来把棋盘从初始局面挪开 —— 初始局面测不出"history/clock 没还原") */
static void playRandomPlies(Chess &c, int plies)
{
    for (int i = 0; i < plies; i++) {
        const int turn = c.sideToMove;
        if (c.getResult(turn) != Chess::RESULT_ONGOING) { break; }
        std::vector<Step*> legal;
        c.sample(turn, legal);
        if (legal.empty()) { break; }
        Step *mv = legal[(std::size_t)(RL::Random::engine() % legal.size())];
        double d = 0.0;
        c.moveForward(mv, d);
        Steps::instance().put(legal);
    }
}

static void testHandAnchor()
{
    std::printf("\n[9] 手工评估锚: 探针零副作用 / 预训练降 gap / 门控回滚\n");

    Chess c;
    c.reset();
    playRandomPlies(c, 7);   /* 非初始局面: sideToMove/clock/history 都不是默认值 */
    /* MLP 骨干: 预训练要真跑几百个样本, TB 骨干只是慢 (算法完全相同) */
    DQNABAgent a(c, 32, 0.99f, 0.001f, Backbone::Mlp);
    a.moeAuxCoef = 0.1f;

    /* ---- 9.1 探针构造对棋盘零副作用 ---- */
    const std::string before = digest(c);
    const double handBefore = a.handEval(c.sideToMove);
    a.rebuildProbes(16, 60);
    CHECK(a.probeCount() == 16, "探针造出 16 个局面");
    CHECK(digest(c) == before, "造探针不改动棋盘 (棋子/走子方/clock/history/hash 全一致)");
    CHECK(std::fabs(a.handEval(c.sideToMove) - handBefore) < 1e-12,
          "造探针之后手工评估也不变 (棋盘真的回到了原局面)");

    /* 手工锚的口径: evaluate() 是黑方视角, 规范视角的锚必须红黑相反 */
    Chess cm;
    cm.reset();
    DQNABAgent am(cm, 32, 0.99f, 0.001f, Backbone::Mlp);
    const double hr = am.handEval(Stone::COLOR_RED);
    const double hb = am.handEval(Stone::COLOR_BLACK);
    std::printf("  初始局面手工锚: 红 %.4f / 黑 %.4f (evaluate() = %.3f, 黑方视角)\n",
                hr, hb, cm.evaluate());
    CHECK(std::fabs(hr + hb) < 1e-6, "手工锚红黑严格取负 (V 是走子方视角)");
    CHECK(std::fabs(hr) < 1e-6, "初始局面双方向对称 -> 锚为 0");

    /* ---- 9.2 预训练: 稠密标签必须真的把 V 拉近 ---- */
    /*
       用**预训练自己报出的** before/after: 它两次都在同一批探针上量
       (外部再量一次会换一批随机局面, 变成"两个不同尺子的数比大小")。
    */
    const double gPost = a.pretrainValueFromHand(1024, 60, 16, 3, true, true);
    const double gPre = a.lastPretrainGapBefore();
    CHECK(!a.lastPretrainRolledBack(), "正常预训练不该触发回滚");
    /*
       判据以 **corr 为主、gap 为辅**, 这是本轮实测逼出来的: gap 是"离手工锚多远",
       而锚的幅度会随局面集合变化, 于是"gap 至少降 10%"这种阈值会随维度/种子飘
       (实测过一次 0.2065 -> 0.1859, 差 0.00005 卡掉 —— 那是个假失败)。
       corr 是尺度无关的: 学了就是学了, 没学就是 0。
    */
    CHECK(a.pretrainStatsAfter().corr > a.pretrainStatsBefore().corr + 0.15,
          "预训练显著提高 V 与手工锚的相关性 (尺度无关判据, 至少 +0.2)");
    CHECK(gPost < gPre + (double)a.pretrainTolerance,
          "预训练没有把 gap 顶出容忍度 (否则它会被整段回滚)");
    CHECK(gPost >= 0.0 && gPost < 1.0, "gap 落在合理的 (0,1) 区间");

    /* ---- 9.3 预训练不动回放池 / 学习步数 (它不消费经验) ---- */
    const std::size_t poolBefore = a.replaySize();
    const int stepsBefore = a.learnSteps();
    a.pretrainValueFromHand(40, 60, 8, 1, false, false);   /* 冻结主干: 只训 V 头 */
    CHECK(a.replaySize() == poolBefore, "预训练不动回放池");
    CHECK(a.learnSteps() == stepsBefore, "预训练不计入 learnSteps (它不是一次 RL 更新)");

    /* 冻结主干的对照: 只训 V 头能不能降 gap (能降多少是另一回事, 见设计文档 §7.4) */
    const DQNABAgent::HandStats hsFrozen = a.netHandStats(0);
    std::printf("  冻结主干 (只训 V 头) 40 局面: gap %.4f -> %.4f, corr %.3f, %.0f ms\n",
                a.pretrainStatsBefore().gap, hsFrozen.gap, hsFrozen.corr,
                a.lastPretrainMs());

    /*
       ---- 尺子的盲区 (必须留下证据) ----
       把 V 头的输出层清零 ⇒ V ≡ 0。这是"什么都没学到"的网络, 但它的 gap **很小**
       (因为手工锚 tanh(evaluate()/3) 的典型幅度只有 ±0.2, 0 离它并不远) ——
       所以只看 gap 会把"V 塌成常数"读成"V 很准"。corr 在这种情况恒为 0, 这就是
       为什么两个数必须一起看 (bench 的 VERDICT 行也同时打 gap 与 corr)。
    */
    {
        RL::iFcLayer *vOut = dynamic_cast<RL::iFcLayer*>(a.m_vHead[a.m_vHead.size() - 1]);
        CHECK(vOut != nullptr, "V 头输出层是 iFcLayer (可以清零做对照)");
        if (vOut != nullptr) {
            for (std::size_t k = 0; k < vOut->w.val.size(); k++) { vOut->w.val[k] = 0.0f; }
            for (std::size_t k = 0; k < vOut->b.val.size(); k++) { vOut->b.val[k] = 0.0f; }
            const DQNABAgent::HandStats hsZero = a.netHandStats(0);
            std::printf("  常数 V (V 头清零): gap %.4f (看起来还行), corr %.3f (真话)\n",
                        hsZero.gap, hsZero.corr);
            CHECK(std::fabs(hsZero.corr) < 1e-9, "常数 V 的 corr 恒为 0 (尺度无关判据)");
            CHECK(hsZero.vStd < 1e-6, "常数 V 的 vStd = 0");
            CHECK(hsZero.gap < hsFrozen.gap + 0.1,
                  "而它的 gap 与训练过的网络同量级 —— gap 单独用会骗人");
        }
    }

    /* ---- 9.4 门控: 蓄意毒化的更新必须被回滚 ---- */
    Chess c2;
    c2.reset();
    /*
       lr=0.02 (默认的 20 倍): clipGrad 下每个张量每步位移 ≈ lr, 所以毒化更新会把 V
       一次打飞。阈值这里**显式**设成 0.001 (按 lr 缩放后 = 0.02), 因为这条测的是
       **机制** (测量→比较→回滚→逐元素还原), 而不是默认阈值该取多少。
    */
    DQNABAgent b(c2, 32, 0.99f, 0.02f, Backbone::Mlp);
    b.valueGateProbeCount = 16;
    b.valueGateEnabled = true;
    b.valueGateTolerance = 0.001f;
    b.targetSyncEvery = 0;
    b.rebuildProbes(16, 60);

    RL::Tensor st(DQNABAgent::STATE_DIM, 1);
    b.encodeStateFor(Stone::COLOR_RED, st);
    std::vector<Step*> legal;
    std::vector<int> idx;
    b.legalMoves(Stone::COLOR_RED, legal, idx);
    Steps::instance().put(legal);
    /*
       毒样本: 局面合法、动作合法, 但**标签是一个与局面无关的常数 +1**
       (Planned 模式直接用 label, 所以这是纯粹的"错标签"污染)。
       这正是门控要拦的东西: 回放池里只要有几条这种样本, TD 目标就会把 V 拖走。
    */
    for (int i = 0; i < 32; i++) {
        DQNABAgent::Sample s;
        s.state = st;
        s.nextState = st;
        s.legalIdx = idx;
        s.nextLegalIdx = idx;
        s.action = idx[0];
        s.reward = -1.0f;
        s.done = true;
        s.label = 1.0f;
        b.pushSample(std::move(s));
    }

    auto trunkW = [](DQNABAgent &ag) {
        std::vector<float> v;
        RL::iFcLayer *l0 = dynamic_cast<RL::iFcLayer*>(ag.m_trunk[0]);
        if (l0 != nullptr) { v.assign(l0->w.val.begin(), l0->w.val.end()); }
        return v;
    };
    const std::vector<float> wBefore = trunkW(b);
    const double g0 = b.netHandGap(16);
    CHECK(b.learnBatch(32, 1), "毒化批更新执行了 (池里 32 条)");
    const double g1 = b.netHandGap(16);
    std::printf("  毒化更新 (lr=0.02, 标签恒 +1, 阈值 %.3f x lr 缩放 = %.3f): "
                "gap %.4f -> %.4f, 回滚 = %d\n",
                (double)b.valueGateTolerance, b.gateToleranceNow(), g0, g1,
                (int)b.lastUpdateRolledBack());
    CHECK(b.lastUpdateRolledBack(), "门控告警并回滚了这次更新");
    CHECK(std::fabs(g1 - g0) < 1e-9, "回滚后 gap 与更新前一致");
    CHECK(trunkW(b) == wBefore, "回滚把**主干权重**逐元素还原 (不是只还原 V 头)");

    /* 对照: 关掉门控, 同一批毒样本必须真的改动权重并让 gap 变差 */
    b.valueGateEnabled = false;
    const std::vector<float> wCtrl = trunkW(b);
    const double gc0 = b.netHandGap(16);
    CHECK(b.learnBatch(32, 1), "对照: 同一批样本再更新一次");
    const double gc1 = b.netHandGap(16);
    std::printf("  对照 (门控关):          gap %.4f -> %.4f\n", gc0, gc1);
    CHECK(!b.lastUpdateRolledBack(), "关掉门控就没有回滚");
    CHECK(trunkW(b) != wCtrl, "对照: 权重确实变了 (上面那次回滚不是'更新根本没生效')");
    CHECK(gc1 > gc0, "对照: 毒标签让 gap 变差 —— 这正是门控拦下的东西");
}

/* ============================================================
 *  [10] Markov 性 / negamax 口径 (对着博弈论前提逐条断言)
 * ============================================================
 *  象棋是**双人零和、完全信息、交替行动**的 Markov Game (不是单 agent MDP)。
 *  这里把三条前提写成断言:
 *    (a) **状态要能决定终局** —— 同一局面的第 2 次与第 3 次出现, 棋盘逐位相同,
 *        但第 3 次直接判和。若编码不含"重复次数", 这两者就是同一个输入而价值不同,
 *        V(s) 就不是 s 的函数, Bellman 备份的前提直接破掉;
 *    (b) **V 必须"当前方视角"** —— 镜像 + 交换颜色后应当给出**同一个数**
 *        (同一个局面, 只是换了个说法), 于是 negamax 的 `val = −child` 自洽;
 *    (c) **TD 目标必须是 negamax 而不是单 agent 的 max** ——
 *        目标 = `r − γ·max_a' Q(s',a')`, 那个负号是"对手会挑对他最有利的"。
 *        这条用"把 A 头清零 + V 头钉成 +1"的办法做成**符号可判定**的实验:
 *        此时 qStar ≡ +1, 于是目标应当 ≈ `r − γ` (若写成 `r + γ` 就是单 agent 口径)。
 */
static void testMarkovAndNegamax()
{
    std::printf("\n[10] Markov 性 / negamax 口径 (零和 Markov Game 的三条前提)\n");

    /* ---- (a) 重复局面: 棋盘相同, 规则上下文不同 ---- */
    {
        Chess c;
        c.reset();
        DQNABAgent a(c, 16, 0.99f, 0.001f, Backbone::Mlp);

        RL::Tensor s0(DQNABAgent::STATE_DIM, 1);
        a.encodeStateFor(Stone::COLOR_RED, s0);
        const double rep0 = (double)s0[DQNABAgent::PLANE_REPEAT * DQNABAgent::CELLS];

        /* 双方各把一个马来回走两次: 走完 4 手回到起始局面 (第 2 次出现) */
        auto cycle = [&]() -> bool {
            /* 红马 (9,1) / 黑马 (0,1) 出去再回来 —— 4 手一个循环, 全是可逆的安静着法 */
            const int from[4] = { 9 * 9 + 1, 0 * 9 + 1, 7 * 9 + 2, 2 * 9 + 2 };
            const int to[4]   = { 7 * 9 + 2, 2 * 9 + 2, 9 * 9 + 1, 0 * 9 + 1 };
            for (int k = 0; k < 4; k++) {
                const int turn = c.sideToMove;
                std::vector<Step*> legal;
                c.sample(turn, legal);
                Step *mv = nullptr;
                for (std::size_t i = 0; i < legal.size(); i++) {
                    if (legal[i]->pos.x == from[k] / 9 && legal[i]->pos.y == from[k] % 9 &&
                        legal[i]->nextPos.x == to[k] / 9 && legal[i]->nextPos.y == to[k] % 9) {
                        mv = legal[i];
                        break;
                    }
                }
                if (mv == nullptr) { Steps::instance().put(legal); return false; }
                Step copy = *mv;
                Steps::instance().put(legal);
                double dummy = 0.0;
                c.moveForward(&copy, dummy);
            }
            return true;
        };

        CHECK(cycle(), "走一个可逆循环回到起始局面");
        RL::Tensor s1(DQNABAgent::STATE_DIM, 1);
        a.encodeStateFor(c.sideToMove, s1);
        const double rep1 = (double)s1[DQNABAgent::PLANE_REPEAT * DQNABAgent::CELLS];

        /* 棋子平面必须逐位相同 (确实是同一个局面) */
        double boardDiff = 0.0;
        for (int p = 0; p < DQNABAgent::PIECE_PLANES; p++) {
            for (int cell = 0; cell < DQNABAgent::CELLS; cell++) {
                const std::size_t k = (std::size_t)(p * DQNABAgent::CELLS + cell);
                boardDiff = std::max(boardDiff, std::fabs((double)s1[k] - (double)s0[k]));
            }
        }
        CHECK(boardDiff == 0.0, "循环之后 14 个棋子平面与起始局面逐位相同 (同一个局面)");
        CHECK(c.sideToMove == Stone::COLOR_RED, "循环是偶数手, 又轮到红方");
        std::printf("  回到起始局面: 棋平面差 %.1e, 重复平面 %.3f -> %.3f, "
                    "repetitionCount = %d (引擎 isRepetition = %d)\n",
                    boardDiff, rep0, rep1, a.repetitionCount(), (int)c.isRepetition());
        CHECK(boardDiff == 0.0 && rep1 > rep0,
              "**同一个局面, 重复平面却变了** —— 状态确实携带了规则上下文 (Markov 修正生效)");
        CHECK((a.repetitionCount() >= 3) == c.isRepetition(),
              "repetitionCount>=3 与引擎 isRepetition() 完全等价 (窗口/判据抄的是同一份)");

        /* 再走两轮: 第 3 次出现 = 判和。特征与引擎的判据必须一致 */
        const int rep2 = cycle() ? a.repetitionCount() : -1;
        const int rep3 = cycle() ? a.repetitionCount() : -1;
        const bool drew = (c.getResult(c.sideToMove) == Chess::RESULT_DRAW);
        std::printf("  第 2 轮之后: repetitionCount = %d, 第 3 轮之后: %d -> getResult = %s, "
                    "repetitionPhase = %.3f\n",
                    rep2, rep3, drew ? "DRAW" : "ONGOING", a.repetitionPhase());
        CHECK(drew, "同一局面出现 3 次 -> 引擎判和 (getResult = DRAW)");
        CHECK((a.repetitionCount() >= 3) == c.isRepetition(),
              "repetitionCount>=3 与引擎 isRepetition() 完全等价 (窗口/判据抄的是同一份)");
    }

    /* ---- (b) 无吃子计数平面: 不吃子就涨, 一吃子归零 ---- */
    {
        Chess c;
        /*
           红车 (5,0) 与黑车 (0,0) 同在 x=0 这一行且中间全空 —— 双方互相能吃到。
           红方先走一手**不吃子**的 (5,0)->(9,0): 仍然与黑车同行 ⇒ 黑方紧接着就有吃子,
           这样"计数涨"和"计子归零"两件事都能在同一局棋里量到。
        */
        const std::vector<SparsePiece> pieces = {
            { Stone::TYPE_JIANG, Stone::COLOR_RED,   9 * 9 + 4 },
            { Stone::TYPE_JIANG, Stone::COLOR_BLACK, 0 * 9 + 5 },   /* y=5: 不与红帅对脸 */
            { Stone::TYPE_CHE,   Stone::COLOR_RED,   5 * 9 + 0 },
            { Stone::TYPE_CHE,   Stone::COLOR_BLACK, 0 * 9 + 0 },
        };
        CHECK(setupSparse(c, pieces), "摆出局面 (红车 x=5 行0 / 黑车 x=0 行0, 同行互吃)");
        DQNABAgent a(c, 16, 0.99f, 0.001f, Backbone::Mlp);
        RL::Tensor st(DQNABAgent::STATE_DIM, 1);

        const double hm0 = a.halfmovePhase();
        std::vector<Step*> legal;
        c.sample(Stone::COLOR_RED, legal);
        Step *quiet = nullptr;
        for (std::size_t i = 0; i < legal.size(); i++) {
            if (legal[i]->pos.x == 5 && legal[i]->pos.y == 0 &&
                legal[i]->nextPos.x == 9 && legal[i]->nextPos.y == 0) {
                quiet = legal[i];
                break;
            }
        }
        CHECK(quiet != nullptr, "红车 (5,0)->(9,0) 是合法着法 (且不吃子)");
        if (quiet != nullptr) {
            Step mv1 = *quiet;
            Steps::instance().put(legal);
            double d = 0.0;
            c.moveForward(&mv1, d);
            const double hm1 = a.halfmovePhase();
            CHECK(c.halfMoveClock == 1, "不吃子的一步把 halfMoveClock 加到 1");
            CHECK(hm1 > hm0, "无吃子平面跟着涨 (60 回合判和的风险进了状态)");
            std::printf("  无吃子平面: %.4f -> %.4f (halfMoveClock = %d)\n",
                        hm0, hm1, c.halfMoveClock);
        } else {
            Steps::instance().put(legal);
        }

        /* 黑方此刻应当能吃红车: 走了这一手 halfMoveClock 与平面都必须归零 */
        std::vector<Step*> l2;
        std::vector<int> i2;
        a.legalMoves(c.sideToMove, l2, i2);
        Step *cap = nullptr;
        for (std::size_t i = 0; i < l2.size(); i++) {
            if (c.m_map[l2[i]->nextPos] != nullptr) { cap = l2[i]; break; }
        }
        CHECK(cap != nullptr, "黑方有吃子着法 (前面那手是故意留出这个吃子的)");
        if (cap != nullptr) {
            Step mv2 = *cap;
            Steps::instance().put(l2);
            double d = 0.0;
            c.moveForward(&mv2, d);
            CHECK(c.halfMoveClock == 0, "吃子把 halfMoveClock 归零");
            CHECK(a.halfmovePhase() == 0.0, "无吃子平面跟着归零");
            std::printf("  吃子之后: 无吃子平面 = %.4f (halfMoveClock = %d)\n",
                        a.halfmovePhase(), c.halfMoveClock);
        } else {
            Steps::instance().put(l2);
        }
    }

    /* ---- (c) negamax 的负号: 把 A 头清零 + V 头钉成 +1, 目标必须 ≈ r − γ ---- */
    {
        Chess c;
        c.reset();
        DQNABAgent a(c, 16, 0.99f, 0.001f, Backbone::Mlp);
        a.targetMode = DQNABAgent::TargetMode::OneStepDouble;
        a.targetSyncEvery = 1;      /* 一次更新就把在线网同步给目标网 */
        a.batchSize = 4;
        a.gamma = 0.99f;

        auto bake = [](RL::Net &net, bool valueOne) {
            RL::iFcLayer *out = dynamic_cast<RL::iFcLayer*>(net[net.size() - 1]);
            if (out == nullptr) { return false; }
            for (std::size_t k = 0; k < out->w.val.size(); k++) { out->w.val[k] = 0.0f; }
            for (std::size_t k = 0; k < out->b.val.size(); k++) {
                /* V 头是 Tanh 输出: tanh(20) = 1.0f (float 下精确饱和) ⇒ V ≡ +1 */
                out->b.val[k] = valueOne ? 20.0f : 0.0f;
            }
            return true;
        };
        CHECK(bake(a.m_vHead, true), "V 头输出层钉成 +1 (tanh(20) = 1.0f)");
        CHECK(bake(a.m_aHead, false), "A 头输出层清零 ⇒ A ≡ 0 ⇒ Q = V + A − mean_A ≡ +1");

        /* 用一次真实更新把在线网同步到目标网 (目标网是私有的, 只能这样"烘焙"它) */
        RL::Tensor st(DQNABAgent::STATE_DIM, 1);
        a.encodeStateFor(Stone::COLOR_RED, st);
        std::vector<Step*> legal;
        std::vector<int> idx;
        a.legalMoves(Stone::COLOR_RED, legal, idx);
        Steps::instance().put(legal);
        for (int i = 0; i < 4; i++) {
            DQNABAgent::Sample smp;
            smp.state = st;
            smp.nextState = st;
            smp.legalIdx = idx;
            smp.nextLegalIdx = idx;
            smp.action = idx[0];
            smp.reward = 0.0f;
            smp.done = false;
            smp.label = 0.0f;
            a.pushSample(std::move(smp));
        }
        CHECK(a.learnBatch(4, 1), "跑一次批更新 (它内部会把在线网硬拷贝到目标网)");

        /* 棋盘停在初始局面, s' 就是当前局面: 标签 = r − γ·qStar, qStar ≡ +1 */
        DQNABAgent::Sample s;
        s.state = st;
        s.nextState = st;
        s.legalIdx = idx;
        s.nextLegalIdx = idx;
        s.action = idx[0];
        s.reward = 0.0f;
        s.done = false;
        const double y = a.computeTarget(s);
        const double negamax = 0.0 - (double)a.gamma * 1.0;      /* r − γ·qStar */
        const double singleAgent = 0.0 + (double)a.gamma * 1.0;  /* r + γ·qStar (错的口径) */
        std::printf("  TD 目标 (Q≡+1, r=0): 实测 %.4f, negamax 口径 %.4f, "
                    "单 agent 口径 %.4f\n", y, negamax, singleAgent);
        CHECK(std::fabs(y - negamax) < 0.02, "目标是 r − γ·max Q (negamax, 不是单 agent 的 max)");
        CHECK(std::fabs(y - singleAgent) > 0.5, "与单 agent 口径 (r + γ·max Q) 明显不同");
    }

    /* ---- (b') V 是"当前方视角": 镜像 + 换色后必须给出同一个数 ---- */
    {
        Chess c;
        const std::vector<SparsePiece> pieces = {
            { Stone::TYPE_JIANG, Stone::COLOR_RED,   9 * 9 + 4 },
            { Stone::TYPE_JIANG, Stone::COLOR_BLACK, 0 * 9 + 5 },
            { Stone::TYPE_CHE,   Stone::COLOR_RED,   7 * 9 + 1 },
            { Stone::TYPE_PAO,   Stone::COLOR_BLACK, 2 * 9 + 5 },
            { Stone::TYPE_BING,  Stone::COLOR_RED,   6 * 9 + 3 },
        };
        CHECK(setupSparse(c, pieces), "摆出一个非对称局面 (红车兵 vs 黑炮)");
        DQNABAgent a(c, 16, 0.99f, 0.001f, Backbone::Mlp);

        RL::Tensor red(DQNABAgent::STATE_DIM, 1);
        a.encodeStateFor(Stone::COLOR_RED, red);
        std::vector<Step*> legal;
        std::vector<int> idx;
        a.legalMoves(Stone::COLOR_RED, legal, idx);
        Steps::instance().put(legal);
        double vRed = 0.0, vBlackView = 0.0;
        std::vector<double> qRed, qBlack;
        a.evaluateNode(red, idx, vRed, qRed);

        /*
           把**同一个棋盘对象**重摆成镜像局面, 用**同一个 agent**(同一套权重)再编码:
           这样 V 的比较才只反映"编码是否相同", 而不是两份随机初始化的网络。
        */
        CHECK(setupMirrored(c, pieces, 0), "摆出镜像局面 (颜色与 x 都换过)");
        RL::Tensor black(DQNABAgent::STATE_DIM, 1);
        a.encodeStateFor(Stone::COLOR_BLACK, black);
        std::vector<Step*> l2;
        std::vector<int> i2;
        a.legalMoves(Stone::COLOR_BLACK, l2, i2);
        Steps::instance().put(l2);
        CHECK(!i2.empty(), "镜像局面里黑方有合法着法 (否则这条断言毫无意义)");
        a.evaluateNode(black, i2, vBlackView, qBlack);

        std::printf("  V(取景红) = %.6f, V(镜像后取景黑) = %.6f (同一个数才说明"
                    "「一个网络服务双方、值是当前方视角」); 合法着法 %d / %d\n",
                    vRed, vBlackView, (int)qRed.size(), (int)qBlack.size());
        CHECK(std::fabs(vRed - vBlackView) < 1e-6,
              "V 是当前方视角: 镜像+换色后给出同一个数 (negamax 的符号翻转才自洽)");
        /*
           Q 必须**按下标对齐**比, 不能按列表位置比: 两次 legalMoves 的返回顺序不保证一致
           (一次是红方在原局面上的走法序, 一次是黑方在镜像局面上的), 而 Q 只取决于
           "哪个动作下标", 与它在列表里的位置无关。第一版就是这么错的 ——
           它报了 4 个"不一致", 其实只是顺序不同 (V 已经完全相同, 这本身就是线索)。
        */
        std::map<int, double> qmRed, qmBlack;
        for (std::size_t i = 0; i < qRed.size(); i++) { qmRed[idx[i]] = qRed[i]; }
        for (std::size_t i = 0; i < qBlack.size(); i++) { qmBlack[i2[i]] = qBlack[i]; }
        double maxQDiff = 0.0;
        for (std::map<int, double>::const_iterator it = qmRed.begin(); it != qmRed.end(); ++it) {
            std::map<int, double>::const_iterator f = qmBlack.find(it->first);
            if (f == qmBlack.end()) { maxQDiff = 1e9; break; }
            maxQDiff = std::max(maxQDiff, std::fabs(it->second - f->second));
        }
        std::printf("  Q 按下标对齐后最大差 = %.3e (合法集大小 %d vs %d)\n",
                    maxQDiff, (int)qmRed.size(), (int)qmBlack.size());
        CHECK(qmRed.size() == qmBlack.size(), "镜像后合法动作下标集合完全相同 (双射)");
        CHECK(maxQDiff < 1e-6, "Q 在镜像+换色后逐项一致 (同一个动作下标 -> 同一个 Q)");
    }
}

int main()
{
    /* 关掉 stdout 缓冲: 一旦崩了 (或者被 kill), 已经跑过的段落不会跟着缓冲区一起丢 */
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("=== DQNABAgent (Alpha-Beta 当 DQN 的 planning head) 测试 ===\n");
    std::printf("SIMD 内核: %s\n", RL::simdops::instructionSet());
    RL::Random::setSeed(20240915);

    testEncoding();
    testActionIndex();
    testDuelingGradient();
    testMateInOne();
    testLearning();
    testSelfPlayAndRouting();
    testTbBackbone();
    testSaveLoad();
    testHandAnchor();
    testMarkovAndNegamax();

    std::printf("\n=== %d 项断言, %d 项失败 ===\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
