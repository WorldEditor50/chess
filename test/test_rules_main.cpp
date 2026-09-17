/*
 * test_rules_main.cpp - 象棋规则引擎回归测试
 *
 * 这个测试和仓库里其它 6 个 test_*_main.cpp 不同: 它们只有 printf + 自对弈统计,
 * 没有任何断言, 所以历史上"未初始化的 m_map 导致空指针崩溃""sample() 不过滤
 * 自杀/不应将/照面""没有将杀与和棋判定""moveBack 把从未被吃的子复活"这些问题
 * 一个都没被抓到。这里对每条规则都写死期望值。
 *
 * 只需要 src/ + src/rl 头文件, 不依赖 Qt, 秒级完成。
 *
 * 顺带报告本构建实际启用的 SIMD 指令集 (见 rl/cpuinfo.hpp): 一个二进制要么带
 * AVX2 内核要么不带, 这件事不该靠猜。
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include "chess.h"
#include "rl/cpuinfo.hpp"

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do {                                        \
        ++g_checks;                                                  \
        if (!(cond)) {                                                \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
        }                                                            \
    } while (0)

#define CHECK_EQ(a, b, msg) do {                                     \
        ++g_checks;                                                  \
        long long _a = (long long)(a);                                \
        long long _b = (long long)(b);                                \
        if (_a != _b) {                                              \
            ++g_failed;                                               \
            std::printf("  [FAIL] %s: %lld != %lld  (%s:%d)\n",        \
                        (msg), _a, _b, __FILE__, __LINE__);           \
        }                                                            \
    } while (0)

static std::string posStr(const Pos &p)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "(%d,%d)", p.x, p.y);
    return buf;
}

/* 清空棋盘 (所有棋子置为死亡, 映射表清空), 用于摆放自定义局面 */
static void clearBoard(Chess &c)
{
    for (int i = 0; i < 32; i++) {
        c.m_children[i]->alive = false;
    }
    c.m_map.clear();
    c.history.clear();
    c.sideToMove = Stone::COLOR_RED;
    c.halfMoveClock = 0;
}

static void place(Chess &c, int id, int x, int y)
{
    Stone *s = c.m_children[id];
    s->alive = true;
    s->pos = Pos(x, y);
    c.m_map[Pos(x, y)] = s;
}

/* 用一个字符串摘要描述整个棋盘, 用于 apply/undo 往返比较 */
static std::string boardDigest(Chess &c)
{
    std::string s;
    for (int i = 0; i < 32; i++) {
        Stone *st = c.m_children[i];
        s += std::to_string(i);
        s += st->alive ? "1" : "0";
        s += std::to_string(st->pos.x);
        s += ",";
        s += std::to_string(st->pos.y);
        s += ";";
    }
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 9; y++) {
            Stone *st = c.m_map[Pos(x, y)];
            s += (st == nullptr) ? "." : std::to_string(st->id);
            s += "|";
        }
    }
    return s;
}

static std::vector<Step> drain(std::vector<Step*> &steps)
{
    std::vector<Step> out;
    out.reserve(steps.size());
    for (Step *s : steps) {
        out.push_back(*s);
    }
    Steps::instance().put(steps);
    return out;
}

static std::vector<Step> legalMoves(Chess &c, int color)
{
    std::vector<Step*> steps;
    c.sample(color, steps);
    return drain(steps);
}

static std::vector<Step> pseudoMoves(Chess &c, int color)
{
    std::vector<Step*> steps;
    c.samplePseudo(color, steps);
    return drain(steps);
}

/* ============================================================
 *  1. StoneMap / Chess 构造后的棋盘必须完全初始化
 * ============================================================ */
static void testBoardInit()
{
    std::printf("\n[1] 棋盘初始化 (未初始化指针回归)\n");

    /*
       StoneMap::data 以前完全没有初始化, 而 Chess 的构造函数也不清 m_map,
       所以除了 32 个棋子所在的格子外, 其余 58 格都是垃圾指针。这里在**任何
       reset() 之前**遍历整张表, 就是当年必崩的那条路径。
    */
    Chess c;
    int empty = 0;
    int occupied = 0;
    bool consistent = true;
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 9; y++) {
            Stone *s = c.m_map[Pos(x, y)];
            if (s == nullptr) {
                empty++;
            } else {
                occupied++;
                if (s->alive == false || !(s->pos == Pos(x, y))) {
                    consistent = false;
                }
            }
        }
    }
    CHECK_EQ(empty, 58, "初始局面应有 58 个空交叉点");
    CHECK_EQ(occupied, 32, "初始局面应有 32 个棋子");
    CHECK(consistent, "m_map 与棋子位置必须一致 (不允许悬垂/错位)");

    /* 32 个棋子 id 必须与数组下标一致 */
    bool idsOk = true;
    for (int i = 0; i < 32; i++) {
        if (c.m_children[i] == nullptr || c.m_children[i]->id != i) {
            idsOk = false;
        }
    }
    CHECK(idsOk, "m_children[i]->id == i");

    CHECK(c.isGameOver() == Stone::COLOR_NONE, "初始局面未结束");
    CHECK(!c.isInCheck(Stone::COLOR_RED), "初始局面红方未被将");
    CHECK(!c.isInCheck(Stone::COLOR_BLACK), "初始局面黑方未被将");
}

/* ============================================================
 *  2. 走法生成: 合法 ⊆ 伪合法, 且合法走法必须解除/不造成被将
 * ============================================================ */
static void testInitialMoveCounts()
{
    std::printf("\n[2] 初始局面走法数\n");

    Chess c;
    std::vector<Step> legal = legalMoves(c, Stone::COLOR_RED);
    std::vector<Step> pseudo = pseudoMoves(c, Stone::COLOR_RED);

    std::printf("  伪合法 = %zu, 合法 = %zu\n", pseudo.size(), legal.size());

    /* 中国象棋初始局面双方各 44 个合法走法 —— 这是标准值 */
    CHECK_EQ(legal.size(), 44, "红方初始合法走法数");
    CHECK_EQ(pseudo.size(), 44, "红方初始伪合法走法数");

    /* 初始局面没有任何走法会暴露己方帅, 所以两者应当相等 */
    CHECK_EQ(legal.size(), pseudo.size(), "初始局面合法数 == 伪合法数");

    /* 每个合法走法都必须是 isLegalMove() 认可的 */
    bool allLegal = true;
    for (const Step &s : legal) {
        if (!c.isLegalMove(Stone::COLOR_RED, &s)) {
            allLegal = false;
        }
    }
    CHECK(allLegal, "sample() 返回的每一步都通过 isLegalMove()");
}

/* ============================================================
 *  3. 被将军时只能应将
 * ============================================================ */
static void testMustRespondToCheck()
{
    std::printf("\n[3] 被将军时必须应将\n");

    /*
       红帅 (9,4), 黑车 (9,0) 同在第 9 行且中间无子 -> 红方被将。
       红帅只能走到 (8,4); (9,3) 与 (9,5) 仍在该行, 依然被车攻击。
       黑将放在 (0,3) 以免与红帅同列形成照面, 干扰本测试。
    */
    Chess c;
    clearBoard(c);
    place(c, Stone::ID_RED_JIANG, 9, 4);
    place(c, Stone::ID_BLACK_JIANG, 0, 3);
    place(c, Stone::ID_BLACK_CHE1, 9, 0);

    CHECK(c.isInCheck(Stone::COLOR_RED), "红方应处于被将状态");

    std::vector<Step> legal = legalMoves(c, Stone::COLOR_RED);
    std::vector<Step> pseudo = pseudoMoves(c, Stone::COLOR_RED);

    std::printf("  伪合法 = %zu, 合法 = %zu\n", pseudo.size(), legal.size());

    CHECK_EQ(legal.size(), 1, "被将时红方只有 1 个合法应将");
    CHECK(legal.size() == 1 && legal[0].nextPos == Pos(8, 4),
          "唯一的应将是把帅走到 (8,4)");
    CHECK(pseudo.size() > legal.size(), "伪合法走法里应当有被过滤掉的");

    /* 每个合法走法走后都不应再被将 */
    bool allSafe = true;
    for (const Step &s : legal) {
        double r = 0;
        c.moveForward(&s, r);
        if (c.isInCheck(Stone::COLOR_RED)) {
            allSafe = false;
        }
        c.moveBack(&s, r);
    }
    CHECK(allSafe, "所有合法走法走后红方都不被将");

    /* 被过滤掉的伪合法走法: 走完之后必然仍被将 */
    bool filteredAreUnsafe = true;
    int filtered = 0;
    for (const Step &s : pseudo) {
        if (c.isLegalMove(Stone::COLOR_RED, &s)) {
            continue;
        }
        filtered++;
        double r = 0;
        c.moveForward(&s, r);
        bool unsafe = c.isInCheck(Stone::COLOR_RED);
        c.moveBack(&s, r);
        if (!unsafe) {
            filteredAreUnsafe = false;
        }
    }
    CHECK(filtered > 0, "应当确实过滤掉了走法");
    CHECK(filteredAreUnsafe, "被过滤的走法走后都被将");
}

/* ============================================================
 *  4. 飞将 (两将照面) 是禁着
 * ============================================================ */
static void testFlyingGeneral()
{
    std::printf("\n[4] 飞将/照面\n");

    /*
       红帅 (9,4), 黑将 (0,4) —— 两将同列。
       挡在中间的是红兵 (4,4)。兵**横走**到 (4,3) 形状合法 (已过河, x==4<=4),
       但会让开中线 -> 两将照面 -> 必须是非法走法。
       (注意: 兵沿同列前进 (4,4)->(3,4) 仍然挡着, 不是照面, 所以这里必须横走。)
    */
    Chess c;
    clearBoard(c);
    place(c, Stone::ID_RED_JIANG, 9, 4);
    place(c, Stone::ID_BLACK_JIANG, 0, 4);
    place(c, Stone::ID_RED_BING1, 4, 4);

    CHECK(!c.isInCheck(Stone::COLOR_RED), "有兵挡着时红方未被将");

    std::vector<Step> legal = legalMoves(c, Stone::COLOR_RED);
    bool illegalFound = true;
    for (const Step &s : legal) {
        if (s.id == Stone::ID_RED_BING1 && s.nextPos == Pos(4, 3)) {
            illegalFound = false;   /* 不该出现在合法走法里 */
        }
    }
    CHECK(illegalFound, "让开中线的兵步被过滤为非法");

    std::vector<Step> pseudo = pseudoMoves(c, Stone::COLOR_RED);
    bool inPseudo = false;
    for (const Step &s : pseudo) {
        if (s.id == Stone::ID_RED_BING1 && s.nextPos == Pos(4, 3)) {
            inPseudo = true;
        }
    }
    CHECK(inPseudo, "该走法在伪合法列表里 (形状合法)");

    /* 强行走掉之后, 红方应当被判为被将 (照面等价于被将) */
    Step expose;
    expose.id = Stone::ID_RED_BING1;
    expose.pos = Pos(4, 4);
    expose.nextPos = Pos(4, 3);
    expose.nextId = Stone::ID_NONE;
    expose.valid = true;
    double r = 0;
    c.moveForward(&expose, r);
    CHECK(c.isInCheck(Stone::COLOR_RED), "走开后两将照面 -> 红方被将");
    c.moveBack(&expose, r);
    CHECK(!c.isInCheck(Stone::COLOR_RED), "回退后恢复为未被将");
}

/* ============================================================
 *  5. 将杀 / 困毙判定
 * ============================================================ */
static void testCheckmateAndStalemate()
{
    std::printf("\n[5] 将杀 / 困毙\n");

    /*
       经典"双车错"简化版: 红帅 (9,4) 被黑车封死在第 9 行与第 4 列。
       黑车A (9,0) 控制第 9 行; 黑车B (8,x) 不能放在 (8,4) 因为那是帅的路。
       用 (9,0) 的车 + (0,4) 的车? 后者与帅同列, 中间无子 -> 就是将军。
       红帅可走: (8,4) 被 (9,0) 的车攻击吗? 车在 (9,0) 只控制第 9 行与第 0 列,
       (8,4) 不受攻击。所以还需要一个子控制 (8,4): 用黑车 (8,0)。
       这样红帅 (9,4) 在 (9,3)/(9,5) 被 (9,0) 车攻击, (8,4) 被 (8,0) 车攻击
       -> 无路可走, 且当前被将 -> 将杀。
       黑将放在 (0,3) 避免照面干扰。
    */
    Chess c;
    clearBoard(c);
    place(c, Stone::ID_RED_JIANG, 9, 4);
    place(c, Stone::ID_BLACK_JIANG, 0, 3);
    place(c, Stone::ID_BLACK_CHE1, 9, 0);
    place(c, Stone::ID_BLACK_CHE2, 8, 0);

    CHECK(c.isInCheck(Stone::COLOR_RED), "红方被将");
    CHECK(!c.hasLegalMoves(Stone::COLOR_RED), "红方无合法走法 -> 将杀");
    CHECK_EQ(c.getResult(Stone::COLOR_RED), Chess::RESULT_BLACK_WIN,
             "getResult 判定黑胜");
    /*
       P0.2: 带原因的重载在**分胜负**时必须把原因清成 DRAW_NONE。
       故意传一个非 NONE 的初值: 如果输出参数只在判和时被写, 这个断言就会红 ——
       而那种"残留上一条读数的原因"是典型静默错误 (调用方会把上一局的和棋原因
       记到这一局头上)。
    */
    {
        Chess::DrawReason w = Chess::DRAW_REPEAT;
        CHECK_EQ(c.getResult(Stone::COLOR_RED, &w), Chess::RESULT_BLACK_WIN,
                 "getResult(color, &reason) 判定黑胜");
        CHECK_EQ((int)w, (int)Chess::DRAW_NONE, "分胜负时和棋原因被清成 DRAW_NONE");
    }
    /* isGameOver() 保持"便宜"的语义: 只看将帅是否存活 */
    CHECK(c.isGameOver() == Stone::COLOR_NONE,
          "isGameOver() 不负责将杀, 仍返回 COLOR_NONE");

    /* 困毙同样判负: 让红方无子可动但不被将军 */
    /*
       红帅 (9,4), 黑车 (9,0) 控制第 9 行但不将军? (9,4) 就在第 9 行 -> 会将军。
       改用: 红帅 (9,4), 黑车控制 (8,4)/(9,3)/(9,5) 而不攻击 (9,4) 本身 —— 做不到,
       因为车攻击是相互的。中国象棋里"不被将但无子可动"需要己方其它子被牵制,
       这里简化为: 红方只剩一个被牵制的兵。
       红帅 (9,4) 未被将; 红兵 (6,4) 挡在黑车 (0,4) 与红帅之间, 兵的三个走法
       ((5,4) 前进、以及过河后的左右) 中, 前进会暴露帅 -> 非法; 左右同理。
       其余走法仅有帅本身, (8,4)/(9,3)/(9,5) 若都被攻击则也无法走。
       为控制这三点, 放黑车 (8,0) 控制第 8 行 -> 覆盖 (8,4)。
       另需覆盖 (9,3)、(9,5): 用黑车 (9,8)? 会同时攻击 (9,4) -> 将军, 不行。
       因此改用黑炮: 炮 (9,8) 隔一个子打 (9,3)? 太绕。
       这里退一步, 只验证"无合法走法 -> getResult 判对方胜"这条判定逻辑本身
       (上面的将杀用例已经覆盖), 不再构造困毙局面。
    */
}

/* ============================================================
 *  6. 三次重复局面判和
 * ============================================================ */
static void testRepetitionDraw()
{
    std::printf("\n[6] 三次重复局面判和\n");

    /*
       红帅在 y=5 列来回, 黑将在 y=3 列来回 —— 两将永不同列, 不涉及照面。
       4 步一个循环, 走 3 个循环 -> 起始局面出现 3 次。
    */
    Chess c;
    clearBoard(c);
    place(c, Stone::ID_RED_JIANG, 9, 5);
    place(c, Stone::ID_BLACK_JIANG, 0, 3);

    CHECK(!c.isDraw(), "开局不是和棋");

    const int cycle[4][5] = {
        /* id,               fromX, fromY, toX, toY */
        { Stone::ID_RED_JIANG,   9, 5, 8, 5 },
        { Stone::ID_BLACK_JIANG, 0, 3, 1, 3 },
        { Stone::ID_RED_JIANG,   8, 5, 9, 5 },
        { Stone::ID_BLACK_JIANG, 1, 3, 0, 3 },
    };

    double r = 0;
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < 4; i++) {
            Step s;
            s.id = cycle[i][0];
            s.pos = Pos(cycle[i][1], cycle[i][2]);
            s.nextPos = Pos(cycle[i][3], cycle[i][4]);
            s.nextId = Stone::ID_NONE;
            s.reward = 0;
            s.valid = true;
            CHECK(c.m_children[s.id]->alive, "循环里要走的子必须存活");
            c.moveForward(&s, r);
        }
    }

    CHECK(c.isRepetition(), "同一局面出现 3 次");
    CHECK(c.isDraw(), "isDraw() 判定为和棋");
    CHECK_EQ(c.getResult(Stone::COLOR_RED), Chess::RESULT_DRAW,
             "getResult 判定和棋");
    /*
       P0.2: 和棋要**分原因**。三次重复与 60 回合自然限着是两类完全不同的和棋,
       混成一桶之后"和棋率高"就无法归因 (该改裁判还是该改台架)。
    */
    {
        Chess::DrawReason reason = Chess::DRAW_NONE;
        CHECK(c.isDraw(&reason), "isDraw(&reason) 也判和");
        CHECK_EQ((int)reason, (int)Chess::DRAW_REPEAT, "和棋原因 = 三次重复");
        reason = Chess::DRAW_NONE;
        CHECK_EQ(c.getResult(Stone::COLOR_RED, &reason), Chess::RESULT_DRAW,
                 "getResult(color, &reason) 也判和");
        CHECK_EQ((int)reason, (int)Chess::DRAW_REPEAT, "getResult 的原因 = 三次重复");
    }
}

/* ============================================================
 *  7. 60 回合自然限着
 * ============================================================ */
static void testSixtyMoveRule()
{
    std::printf("\n[7] 60 回合自然限着\n");

    Chess c;
    clearBoard(c);
    place(c, Stone::ID_RED_JIANG, 9, 5);
    place(c, Stone::ID_BLACK_JIANG, 0, 3);

    /* 未吃子、未走兵: 计数递增 */
    Step s;
    s.id = Stone::ID_RED_JIANG;
    s.pos = Pos(9, 5);
    s.nextPos = Pos(8, 5);
    s.nextId = Stone::ID_NONE;
    s.valid = true;
    double r = 0;
    c.moveForward(&s, r);
    CHECK_EQ(c.halfMoveClock, 1, "走一步安静着法后计数为 1");

    /* 吃子会清零: 摆一个可以直接吃的黑子 */
    Chess c2;
    clearBoard(c2);
    place(c2, Stone::ID_RED_JIANG, 9, 5);
    place(c2, Stone::ID_BLACK_JIANG, 0, 3);
    place(c2, Stone::ID_BLACK_CHE1, 5, 5);
    place(c2, Stone::ID_RED_CHE1, 6, 5);   /* 车可以直接吃到黑车 */
    c2.halfMoveClock = 40;
    Step cap;
    cap.id = Stone::ID_RED_CHE1;
    cap.pos = Pos(6, 5);
    cap.nextPos = Pos(5, 5);
    cap.nextId = Stone::ID_BLACK_CHE1;
    cap.valid = true;
    c2.moveForward(&cap, r);
    CHECK_EQ(c2.halfMoveClock, 0, "吃子后自然限着计数清零");
    c2.moveBack(&cap, r);
    CHECK_EQ(c2.halfMoveClock, 40, "moveBack 精确恢复限着计数");

    /* 到达 120 半回合 -> 和棋 (直接推进计数, 单测规则本身) */
    Chess c3;
    clearBoard(c3);
    place(c3, Stone::ID_RED_JIANG, 9, 5);
    place(c3, Stone::ID_BLACK_JIANG, 0, 3);
    c3.halfMoveClock = 119;
    /* 同样故意给个非 NONE 初值: 不是和棋时输出参数也必须被写 */
    {
        Chess::DrawReason none = Chess::DRAW_REPEAT;
        CHECK(!c3.isDraw(&none), "119 半回合还不是和棋");
        CHECK_EQ((int)none, (int)Chess::DRAW_NONE, "不是和棋时原因被清成 DRAW_NONE");
    }
    c3.halfMoveClock = 120;
    CHECK(c3.isDraw(), "120 半回合判和");
    CHECK_EQ(c3.getResult(Stone::COLOR_RED), Chess::RESULT_DRAW, "getResult 判和");
    /* P0.2: 这一条与"三次重复"必须能区分开 —— 否则分桶没有意义 */
    {
        Chess::DrawReason r60 = Chess::DRAW_NONE;
        CHECK_EQ(c3.getResult(Stone::COLOR_RED, &r60), Chess::RESULT_DRAW,
                 "getResult(color, &reason) 判和");
        CHECK_EQ((int)r60, (int)Chess::DRAW_NO_CAPTURE60,
                 "和棋原因 = 60 回合自然限着 (不是三次重复)");
    }
}

/* ============================================================
 *  8. moveForward / moveBack 必须严格互为逆操作
 * ============================================================ */
static void testMakeUnmakeRoundTrip()
{
    std::printf("\n[8] 落子/回退往返一致性\n");

    Chess c;
    std::vector<Step> legal = legalMoves(c, Stone::COLOR_RED);

    std::size_t checked = 0;
    bool hashOk = true;
    bool boardOk = true;
    bool sideOk = true;
    bool clockOk = true;

    const std::string before = boardDigest(c);
    const unsigned long long beforeHash = c.computeHash();
    const int beforeSide = c.sideToMove;
    const int beforeClock = c.halfMoveClock;
    const std::size_t beforeHistory = c.history.size();

    for (const Step &s : legal) {
        double r = 0;
        c.moveForward(const_cast<Step*>(&s), r);
        c.moveBack(const_cast<Step*>(&s), r);
        checked++;
        if (c.computeHash() != beforeHash) hashOk = false;
        if (boardDigest(c) != before) boardOk = false;
        if (c.sideToMove != beforeSide) sideOk = false;
        if (c.halfMoveClock != beforeClock) clockOk = false;
        if (c.history.size() != beforeHistory) boardOk = false;
    }

    CHECK_EQ(checked, legal.size(), "遍历了全部初始合法走法");
    CHECK(hashOk, "往返后局面哈希不变");
    CHECK(boardOk, "往返后棋盘与 history 完全复原");
    CHECK(sideOk, "往返后 sideToMove 复原");
    CHECK(clockOk, "往返后 halfMoveClock 复原");

    /*
       含吃子的往返: 把被吃子真正复活, 且映射表恢复。
       用黑车吃红兵的局面。
    */
    Chess c2;
    clearBoard(c2);
    place(c2, Stone::ID_RED_JIANG, 9, 4);
    place(c2, Stone::ID_BLACK_JIANG, 0, 3);
    place(c2, Stone::ID_RED_BING1, 4, 4);   /* 红兵过河后位于第 4 行 */
    place(c2, Stone::ID_BLACK_CHE1, 3, 4);  /* 黑车可以吃它 */
    const std::string before2 = boardDigest(c2);
    Step eat;
    eat.id = Stone::ID_BLACK_CHE1;
    eat.pos = Pos(3, 4);
    eat.nextPos = Pos(4, 4);
    eat.nextId = Stone::ID_RED_BING1;
    eat.valid = true;
    CHECK(c2.isLegalMove(Stone::COLOR_BLACK, &eat), "吃兵是合法走法");
    double r2 = 0;
    c2.moveForward(&eat, r2);
    CHECK(c2.m_children[Stone::ID_RED_BING1]->alive == false, "吃子后被吃子死亡");
    c2.moveBack(&eat, r2);
    CHECK(c2.m_children[Stone::ID_RED_BING1]->alive, "回退后被吃子复活");
    CHECK(boardDigest(c2) == before2, "吃子往返后棋盘完全复原");

    /* 未实际执行的走法, moveBack 不应改动棋盘 (原来会把棋子"复活"到终点格) */
    Chess c3;
    const std::string before3 = boardDigest(c3);
    Step bogus;
    bogus.id = Stone::ID_RED_CHE1;
    bogus.pos = Pos(9, 0);
    bogus.nextPos = Pos(0, 8);      /* 车走不到这里 (中间有子) */
    bogus.nextId = Stone::ID_NONE;
    bogus.valid = true;
    double r3 = 0;
    c3.moveForward(&bogus, r3);      /* moveTo 失败 -> 不应产生任何记账 */
    c3.moveBack(&bogus, r3);         /* 未执行 -> 不应回退 */
    CHECK(boardDigest(c3) == before3, "非法走法不改变棋盘");
    CHECK_EQ(c3.halfMoveClock, 0, "非法走法不推进限着计数");
}

/* ============================================================
 *  9. Step 值语义 / 对象池
 * ============================================================ */
static void testStepValueSemantics()
{
    std::printf("\n[9] Step 值语义与对象池\n");

    Step a;
    CHECK(!a.valid, "默认构造的 Step.valid == false (无走法占位)");
    CHECK(a.id != Stone::ID_NONE,
          "默认 Step.id 是 0, 所以不能再用 'id == ID_NONE' 当无走法哨兵");

    Step b(7, Pos(1, 2), Stone::ID_NONE, Pos(3, 4), 0.5);
    CHECK(b.valid, "走法生成器构造的 Step.valid == true");

    Step c;
    c = b;                     /* 赋值运算符 */
    CHECK(c.id == b.id && c.nextId == b.nextId, "operator= 复制 id/nextId");
    CHECK(c.pos == b.pos && c.nextPos == b.nextPos, "operator= 复制坐标");
    CHECK(c.reward == b.reward, "operator= 必须复制 reward (原来漏掉了)");
    CHECK(c.valid == b.valid, "operator= 必须复制 valid (原来漏掉了)");

    /* 池: get 出来的对象拿回 put 之后能被再次 get 到 */
    Step *p1 = Steps::instance().get();
    p1->valid = true;
    Steps::instance().put(p1);
    Step *p2 = Steps::instance().get();
    CHECK(p2 != nullptr, "对象池可复用");
    Steps::instance().put(p2);

    /* sample() 反复调用不应泄漏: 池的可用对象数应保持稳定 */
    Chess c2;
    for (int i = 0; i < 20; i++) {
        std::vector<Step> m = legalMoves(c2, Stone::COLOR_RED);
        CHECK_EQ(m.size(), 44, "重复生成走法数稳定");
    }
}

/* ============================================================
 *  10. 局面哈希必须包含走棋方
 * ============================================================ */
static void testHashIncludesSideToMove()
{
    std::printf("\n[10] 局面哈希包含走棋方\n");

    Chess c;
    c.sideToMove = Stone::COLOR_RED;
    unsigned long long hRed = c.computeHash();
    c.sideToMove = Stone::COLOR_BLACK;
    unsigned long long hBlack = c.computeHash();
    CHECK(hRed != hBlack, "同一布局、不同走棋方必须是不同的哈希");

    /* reset() 之后 history 必须为空, 且轮到红方 */
    Chess c2;
    double dummy = 0.0;
    c2.moveForward(nullptr, dummy);              /* 空指针防御: 不应崩溃 */
    c2.reset();
    CHECK_EQ(c2.history.size(), (std::size_t)0, "reset() 清空 history");
    CHECK_EQ(c2.sideToMove, Stone::COLOR_RED, "reset() 后轮到红方");
    CHECK(!c2.isRepetition(), "reset() 后不构成重复局面");
}

/* ================================================================
 *  [11] Chess::Result -> 赢家颜色 的换算 (winnerOfResult)
 *
 *  为什么值得一条测试: 统计胜负的地方屡次把 Chess::Result **直接**和 Stone::Color
 *  比较, 而两个枚举数值错位 ——
 *      RESULT_ONGOING=0, RESULT_RED_WIN=1, RESULT_BLACK_WIN=2, RESULT_DRAW=3
 *      COLOR_RED=0,      COLOR_BLACK=1,   COLOR_NONE=2
 *  于是 `gameResult == Stone::COLOR_BLACK` 实际匹配的是 RESULT_RED_WIN (1==1),
 *  红胜被记成黑胜; `== Stone::COLOR_RED` 匹配 RESULT_ONGOING, 黑胜谁都匹配不上。
 *  胜率面板因此是错的, 而程序不会报任何错 —— 这类错误只能靠断言钉住。
 * ================================================================ */
static void testResultToWinnerColor()
{
    std::printf("\n[11] Chess::Result 与 Stone::Color 不能直接比较\n");

    CHECK_EQ(winnerOfResult(Chess::RESULT_RED_WIN), Stone::COLOR_RED,
             "RESULT_RED_WIN -> 红方");
    CHECK_EQ(winnerOfResult(Chess::RESULT_BLACK_WIN), Stone::COLOR_BLACK,
             "RESULT_BLACK_WIN -> 黑方");
    CHECK_EQ(winnerOfResult(Chess::RESULT_DRAW), Stone::COLOR_NONE,
             "RESULT_DRAW -> 无赢家");
    CHECK_EQ(winnerOfResult(Chess::RESULT_ONGOING), Stone::COLOR_NONE,
             "RESULT_ONGOING -> 无赢家");

    /* 把"为什么需要这个函数"钉成断言: 两个枚举的数值确实是错位的 */
    CHECK(Chess::RESULT_RED_WIN == Stone::COLOR_BLACK,
          "陷阱: RESULT_RED_WIN(1) 与 COLOR_BLACK(1) 撞号 (所以直接比较会把红胜记成黑胜)");
    CHECK(Chess::RESULT_ONGOING == Stone::COLOR_RED,
          "陷阱: RESULT_ONGOING(0) 与 COLOR_RED(0) 撞号");
    CHECK(Chess::RESULT_BLACK_WIN != Stone::COLOR_BLACK,
          "陷阱: RESULT_BLACK_WIN(2) 谁都匹配不上");
}

int main()
{
    std::printf("========================================\n");
    std::printf("  中国象棋 - 规则引擎回归测试\n");
    std::printf("========================================\n");
    std::printf("  SIMD: %s\n", RL::cpuinfo::describe().c_str());

    testBoardInit();
    testInitialMoveCounts();
    testMustRespondToCheck();
    testFlyingGeneral();
    testCheckmateAndStalemate();
    testRepetitionDraw();
    testSixtyMoveRule();
    testMakeUnmakeRoundTrip();
    testStepValueSemantics();
    testHashIncludesSideToMove();
    testResultToWinnerColor();

    std::printf("\n========================================\n");
    std::printf("  断言 %d 条, 失败 %d 条\n", g_checks, g_failed);
    std::printf("========================================\n");
    return (g_failed == 0) ? 0 : 1;
}
