#ifndef CHESS_H
#define CHESS_H

#include "stone.h"
#include <algorithm>

class Chess
{
public:
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
public:
    Chess();
    Chess(const Chess &other);
    Chess& operator=(const Chess &other);
    ~Chess();
    Stone *get(int id);
    void reset();
    void moveForward(const Step* s, double &totalReward);
    void moveBack(const Step *s, double &totalReward);
    void sample(int color, std::vector<Step *> &steps);
    int isGameOver();
    bool isInCheck(int color);
    /* 优化: 增强评估函数 */
    double evaluate();
    /* 长将/循环走法检测 */
    void pushHistory();
    bool isRepetition();
    std::vector<unsigned long long> history;
    unsigned long long computeHash();
};

#endif // CHESS_H
