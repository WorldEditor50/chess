#ifndef CHESS_H
#define CHESS_H

#include "stone.h"

class Chess
{
public:
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
    /* 棋子集合 */
    std::array<Stone*, 32> &stones;
public:
    Chess();
    ~Chess();
    Stone *get(int id);
    void reset();
    void moveForward(const Step* s, double &totalReward);
    void moveBack(const Step *s, double &totalReward);
    void sample(int color, std::vector<Step *> &steps);
    double minimizeAlpha(int color, int depth, double beta, double &totalReward);
    double maximizeBeta(int color, int depth, double alpha, double &totalReward);
    Step alphaBetaPruning(int color, int depth);
    int isGameOver();
};

#endif // CHESS_H
