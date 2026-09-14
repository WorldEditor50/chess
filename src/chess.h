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
     * isLegalMove() 的热路径版本: inCheck 由调用方预先算好 (它的值只取决于走之前
     * 的局面, 同一批候选走法共享)。搜索 / 走法生成里应当用这个重载, 避免每个
     * 走法都重算一次 isInCheck()。
     */
    bool isLegalMoveInternal(int color, const Step *s, bool inCheck);
    /* 该方是否还有合法走法 (将杀 / 困毙判定) */
    bool hasLegalMoves(int color);
    /* 三次重复局面 或 60 回合(120 半回合)内无吃子 -> 和棋 */
    bool isDraw();
    /* 完整终局判定 (含将杀/困毙/和棋); colorToMove = 轮到谁走 */
    int getResult(int colorToMove);
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

#endif // CHESS_H
