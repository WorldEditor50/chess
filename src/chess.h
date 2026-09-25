#ifndef CHESS_H
#define CHESS_H

#include "stone.h"
#include <algorithm>

class Chess
{
public:
    /* 终局判定结果 (Chess::getResult) */
    enum Result {
        RESULT_ONGOING = 0,
        RESULT_RED_WIN,
        RESULT_BLACK_WIN,
        RESULT_DRAW
    };

    /*
     * ====================================================================
     *  和棋原因 (P0.2, 2026-09) —— "判和"要能归因, 否则和棋率治不到点上
     * ====================================================================
     *
     *  为什么需要它: 在本工程当前的台架下, "和棋"至少是三类**性质完全不同**的事件:
     *
     *    * DRAW_REPEAT       —— 三次重复局面。规则结果, 也是唯一"白嫖和棋"的通道
     *                           (本工程没有长将/长捉判负, 见 docs/rl_plan_optimized.md §0.2)。
     *    * DRAW_NO_CAPTURE60 —— 60 回合 (120 半回合) 无吃子 (自然限着)。
     *                           **注意**: GUI 训练一局上限只有 60 ply, 所以这条在
     *                           训练路径上根本不可达, 只有在 --moves>=120 的 bench 里才会出现。
     *    * 截断              —— 走到手数上限。**这不是规则结果, 是台架产物**,
     *                           但它常常是训练里和棋的大头, 所以必须有办法把它单独数出来。
     *                           截断由调用方记录 (见 RL::Diag::END_TRUNCATED), 规则
     *                           引擎这里只负责前两类。
     *
     *  只把 RESULT_DRAW 折成一个数, 就会把上面三类混成一桶, "和棋率高"因此无法判断
     *  该改裁判、该改课程、还是该改手数上限。
     */
    enum DrawReason {
        DRAW_NONE = 0,       /* 不是和棋 (未终局 / 已分胜负) */
        DRAW_REPEAT,         /* 三次重复局面 */
        DRAW_NO_CAPTURE60    /* 60 回合 (120 半回合) 无吃子 — 自然限着 */
    };

    /* 历史记录: 用于三次重复局面判定; halfMoveClock 在同一记录里以便精确回退 */
    struct HistoryRecord {
        unsigned long long hash;
        int halfMoveClock;
    };

    /* 棋盘映射 & 棋子数组 (原 static Stone::map / Stone::children) */
    StoneMap<Stone> m_map;
    std::array<Stone*, 32> m_children;

    /* 红方 */
    Che redChe1;
    Ma redMa1;
    Xiang redXiang1;
    Shi redShi1;
    Jiang redJiang;
    Shi redShi2;
    Xiang redXiang2;
    Ma redMa2;
    Che redChe2;
    Pao redPao1;
    Pao redPao2;
    Bing redBing1;
    Bing redBing2;
    Bing redBing3;
    Bing redBing4;
    Bing redBing5;
    /* 黑方 */
    Che blackChe1;
    Ma blackMa1;
    Xiang blackXiang1;
    Shi blackShi1;
    Jiang blackJiang;
    Shi blackShi2;
    Xiang blackXiang2;
    Ma blackMa2;
    Che blackChe2;
    Pao blackPao1;
    Pao blackPao2;
    Bing blackBing1;
    Bing blackBing2;
    Bing blackBing3;
    Bing blackBing4;
    Bing blackBing5;
    /* 棋子集合引用 (保持与 m_children 同步) */
    std::array<Stone*, 32> &stones;
    /* 棋子-位置价值表 (Piece-Square Tables) */
    static const double chePST[10][9];
    static const double maPST[10][9];
    static const double paoPST[10][9];
    static const double bingPST[10][9];
    static const double jiangPST[10][9];
    static const double shiPST[10][9];
    static const double xiangPST[10][9];
    /* 局面价值项开关 (Phase 5): 默认开; 关掉用于 A/B 归因, 见 evaluate() 的说明 */
    static bool g_positionalEvalEnabled;
public:
    Chess();
    Chess(const Chess &other);
    Chess& operator=(const Chess &other);
    ~Chess();
    Stone *get(int id);
    void reset();
    void moveForward(const Step* s, double &totalReward);
    void moveBack(const Step *s, double &totalReward);
    /*
     * sample() 只产生**合法**走法 (伪合法走法再过滤掉"走后自己被将/照面"),
     * 所以所有 agent / GUI / 搜索都能直接用它, 不需要各自再校验一次。
     * samplePseudo() 保留伪合法版本, 仅用于实现 sample() 自身。
     */
    void sample(int color, std::vector<Step *> &steps);
    void samplePseudo(int color, std::vector<Step *> &steps);
    int isGameOver();
    bool isInCheck(int color);
    /* target 是否被 byColor 方的任何棋子攻击 (含飞将规则) */
    bool isAttacked(const Pos &target, int byColor);
    /* 把 s 落到棋盘上再撤销, 判断走后自己是否被将 */
    bool isLegalMove(int color, const Step *s);
    /*
     * ---- 自由走子 (调试, 2026-09 用户口径: "被将军时我希望能移动所有棋子") ----
     *
     * 为 true 时 `applyMove` 跳过棋子的形状校验, 于是任意棋子可以走到棋盘上任意一格
     * (仍不能吃自己的子)。用途: 摆局面 / 让玩家在残局里任意调整子力 —— 形状规则
     * (马走日、兵只前进、仕不出九宫…) 会挡住这类操作。
     *
     * ⚠ 这不是"合法的象棋": 打开之后局面可以走到规则上不存在的状态。
     *   默认 false, 而且**不参与任何搜索/评估路径** —— 只有 GUI 的
     *   "自由走子(调试)" 勾选框会打开它 (见 ChessBoard::setFreeMoveEnabled)。
     */
    bool freeMove = false;

    /*
     * isLegalMove() 的热路径版本: inCheck 由调用方预先算好 (它的值只取决于走之前
     * 的局面, 同一批候选走法共享)。搜索 / 走法生成里应当用这个重载, 避免每个
     * 走法都重算一次 isInCheck()。
     */
    bool isLegalMoveInternal(int color, const Step *s, bool inCheck);
    /* 该方是否还有合法走法 (将杀 / 困毙判定) */
    bool hasLegalMoves(int color);
    /*
     * 三次重复局面 或 60 回合(120 半回合)内无吃子 -> 和棋
     * reason (可选, P0.2): 非空时写回**是哪一类**和棋; 不是和棋时一律置 DRAW_NONE。
     */
    bool isDraw(DrawReason *reason = nullptr);
    /*
     * 完整终局判定 (含将杀/困毙/和棋); colorToMove = 轮到谁走
     * reason (可选, P0.2): 非空时写回和棋原因 (分胜负/未终局 -> DRAW_NONE)。
     */
    int getResult(int colorToMove, DrawReason *reason = nullptr);
    /* 优化: 增强评估函数 */
    double evaluate();
    /*
     * 完整局面价值 = evaluate() (材质+PST) + positionalScore() (将安全/空间/士象)。
     * **刻意与 evaluate() 分开**: evaluate() 是 ABAgent 的叶子评估 (一步 depth-4 要调
     * 约 170 万次, 叠加局面项会让它慢 6.5 倍、把等时间深度吃回去), 而这一路只给 RL 的
     * 势能塑形 Φ 用 (每手两次)。
     */
    double evaluatePositional();
    /*
     * 局面价值项 (Phase 5, 2026-09): 将安全 / 九宫受攻 / 攻击对方将 / 士象完整度。
     * 返回**黑方视角**的加权和 (与 evaluate() 同一口径), 由 evaluate() 在末尾叠加。
     * 它同时是势能 Φ 的原料 —— Φ 把"棋盘局面价值评估"接进 PPO 的训练信号。
     */
    double positionalScore();
    /*
     * 空间 / 机动性 (Phase 5, 黑方视角): 逐子几何地数"可达格数"与"伸进对方半场的
     * 格数" —— 它的意义是让**不吃子的着法也能改变局面价值** (材质要等吃子才动,
     * 而调子/占位/争空间的价值在未来的选择权里)。不用 isAttacked, 因为这里是 AB
     * 叶子热路径。被 positionalScore() 叠加。
     */
    double activityScore();
    /* 单个棋子是否攻击 target (positionalScore 内部用) */
    bool isAttackedOne(const Pos &target, const Stone *s);
    /*
     * 局面价值项的开关: 默认开。留它出来是为了做 A/B —— 改 evaluate() 会同时改掉
     * ABAgent 与 EVAB 的预训练标签, 必须能"关掉再量一遍"才能归因。
     */
    static void setPositionalEvalEnabled(bool on) { g_positionalEvalEnabled = on; }
    static bool positionalEvalEnabled() { return g_positionalEvalEnabled; }
    /* 长将/循环走法检测 */
    void pushHistory();
    bool isRepetition();
    std::vector<HistoryRecord> history;
    unsigned long long computeHash();
    /*
     * Zobrist 随机键表 (32 棋子 x 90 格 + 1 个"轮到黑方"键)。
     * computeHash() 用它算置换表键与重复局面键: 随机键之间的相关性远低于原来那种
     * (id | x<<8 | y<<12) 的结构化异或, 因此适合作为 TT 的键。
     */
    static const std::array<unsigned long long, 32*90 + 1> &zobrist();

    /* 当前走棋方 (由 moveForward/moveBack/reset 维护, 参与局面哈希) */
    int sideToMove;
    /* 距上一次吃子的半回合数 */
    int halfMoveClock;

private:
    /* 只改棋盘, 不做 history / sideToMove / 收益记账 (供 moveForward 与 isLegalMove 共用) */
    bool applyMove(const Step *s);
    void undoMove(const Step *s);
};

/* ====================================================================
 *  终局口径统一 (Phase 6, 2026-09)
 * ====================================================================
 *
 *  把 Chess::getResult() 的结果换算成"**某一步走子方**视角"的终局值。
 *
 *  为什么需要它: 工程里曾经并存**三套**终局定义 ——
 *    * isGameOver()  只看将帅是否存活 (便宜), 于是"被将死""判和"都不是终局;
 *    * getResult()   完整 (将杀/困毙/吃将/三次重复/60 回合无吃子判和);
 *    * trainSelfPlay 里还有一条"没有合法走法"的分支自己算赢家。
 *  各处按各自的口径取 ±1, 于是同一个终局事件在不同路径上含义不同, 而且
 *  **rollout 里的"被将死"根本不终止** —— 截断处一律给 0, 价值目标因此没有信号。
 *
 *  现在统一走 getResult() + 这个换算函数。调用点都在 moveForward **之后**,
 *  所以 getResult 的参数应当是 chess.sideToMove (刚被落子翻转成对手)。
 * ==================================================================== */
inline float outcomeForMover(int chessResult, int moverColor)
{
    if (chessResult == Chess::RESULT_DRAW) {
        return 0.0f;
    }
    if (chessResult == Chess::RESULT_ONGOING) {
        return 0.0f;   /* 未终局; 调用方应当先判 isOngoing */
    }
    const bool redWon = (chessResult == Chess::RESULT_RED_WIN);
    const bool moverIsRed = (moverColor == Stone::COLOR_RED);
    return (redWon == moverIsRed) ? REWARD_TERMINAL : -REWARD_TERMINAL;
}

/* ====================================================================
 *  winnerOfResult: Chess::Result -> 赢家颜色 (和棋 / 未终局 = COLOR_NONE)
 *
 *  为什么需要它: 统计胜负的地方屡次把 **Chess::Result 直接和 Stone::Color 比较**,
 *  而这两个枚举的数值是错位的 ——
 *      RESULT_ONGOING=0, RESULT_RED_WIN=1, RESULT_BLACK_WIN=2, RESULT_DRAW=3
 *      COLOR_RED=0,      COLOR_BLACK=1,   COLOR_NONE=2
 *  于是 `gameResult == Stone::COLOR_BLACK` 实际匹配的是 **RESULT_RED_WIN**
 *  (1 == 1), 红胜被记成黑胜; 而 `== Stone::COLOR_RED` 匹配 RESULT_ONGOING (0),
 *  黑胜(2) 谁都匹配不上 —— 胜率面板因此是错的, 而且不会报任何错。
 *  统计口径一律走这个函数, 不要再手写比较。
 * ==================================================================== */
inline int winnerOfResult(int chessResult)
{
    if (chessResult == Chess::RESULT_RED_WIN) {
        return Stone::COLOR_RED;
    }
    if (chessResult == Chess::RESULT_BLACK_WIN) {
        return Stone::COLOR_BLACK;
    }
    return Stone::COLOR_NONE;   /* 和棋 / 未终局 */
}

#endif // CHESS_H
