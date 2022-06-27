#include "chess.h"

Chess::Chess():
    redChe1(Stone::ID_RED_CHE1, Stone::COLOR_RED, Pos(9, 0)),
    redMa1(Stone::ID_RED_MA1, Stone::COLOR_RED, Pos(9, 1)),
    redXiang1(Stone::ID_RED_XIANG1, Stone::COLOR_RED, Pos(9, 2)),
    redShi1(Stone::ID_RED_SHI1, Stone::COLOR_RED, Pos(9, 3)),
    redJiang(Stone::ID_RED_JIANG, Stone::COLOR_RED, Pos(9, 4)),
    redShi2(Stone::ID_RED_SHI2, Stone::COLOR_RED, Pos(9, 5)),
    redXiang2(Stone::ID_RED_XIANG2, Stone::COLOR_RED, Pos(9, 6)),
    redMa2(Stone::ID_RED_MA2, Stone::COLOR_RED, Pos(9, 7)),
    redChe2(Stone::ID_RED_CHE2, Stone::COLOR_RED, Pos(9, 8)),
    redPao1(Stone::ID_RED_PAO1, Stone::COLOR_RED, Pos(7, 1)),
    redPao2(Stone::ID_RED_PAO2, Stone::COLOR_RED, Pos(7, 7)),
    redBing1(Stone::ID_RED_BING1, Stone::COLOR_RED, Pos(6, 0)),
    redBing2(Stone::ID_RED_BING2, Stone::COLOR_RED, Pos(6, 2)),
    redBing3(Stone::ID_RED_BING3, Stone::COLOR_RED, Pos(6, 4)),
    redBing4(Stone::ID_RED_BING4, Stone::COLOR_RED, Pos(6, 6)),
    redBing5(Stone::ID_RED_BING5, Stone::COLOR_RED, Pos(6, 8)),
    blackChe1(Stone::ID_BLACK_CHE1, Stone::COLOR_BLACK, Pos(0, 0)),
    blackMa1(Stone::ID_BLACK_MA1, Stone::COLOR_BLACK, Pos(0, 1)),
    blackXiang1(Stone::ID_BLACK_XIANG1, Stone::COLOR_BLACK, Pos(0, 2)),
    blackShi1(Stone::ID_BLACK_SHI1, Stone::COLOR_BLACK, Pos(0, 3)),
    blackJiang(Stone::ID_BLACK_JIANG, Stone::COLOR_BLACK, Pos(0, 4)),
    blackShi2(Stone::ID_BLACK_SHI2, Stone::COLOR_BLACK, Pos(0, 5)),
    blackXiang2(Stone::ID_BLACK_XIANG2, Stone::COLOR_BLACK, Pos(0, 6)),
    blackMa2(Stone::ID_BLACK_MA2, Stone::COLOR_BLACK, Pos(0, 7)),
    blackChe2(Stone::ID_BLACK_CHE2, Stone::COLOR_BLACK, Pos(0, 8)),
    blackPao1(Stone::ID_BLACK_PAO1, Stone::COLOR_BLACK, Pos(2, 1)),
    blackPao2(Stone::ID_BLACK_PAO2, Stone::COLOR_BLACK, Pos(2, 7)),
    blackBing1(Stone::ID_BLACK_BING1, Stone::COLOR_BLACK, Pos(3, 0)),
    blackBing2(Stone::ID_BLACK_BING2, Stone::COLOR_BLACK, Pos(3, 2)),
    blackBing3(Stone::ID_BLACK_BING3, Stone::COLOR_BLACK, Pos(3, 4)),
    blackBing4(Stone::ID_BLACK_BING4, Stone::COLOR_BLACK, Pos(3, 6)),
    blackBing5(Stone::ID_BLACK_BING5, Stone::COLOR_BLACK, Pos(3, 8)),
    stones(Stone::children)
{

}

Chess::~Chess()
{

}

Stone *Chess::get(int id)
{
    return stones[id];
}

void Chess::reset()
{
    Stone::map.clear();
    /* 红方 */
    std::vector<Pos> redGroupPos = {{9, 0}, {9, 1}, {9, 2}, {9, 3},
                                    {9, 4}, {9, 5}, {9, 6}, {9, 7},
                                    {9, 8}, {7, 1}, {7, 7}, {6, 0},
                                    {6, 2}, {6, 4}, {6, 6}, {6, 8}};

    for (std::size_t i = 0; i < redGroupPos.size(); i++) {
        const Pos &pos = redGroupPos[i];
        Stone *stone = Stone::children[i + Stone::ID_RED];
        stone->pos = pos;
        stone->alive = 1;
        Stone::map[pos] = stone;
    }
    /* 黑方 */
    std::vector<Pos> blackGroupPos = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                      {0, 4}, {0, 5}, {0, 6}, {0, 7},
                                      {0, 8}, {2, 1}, {2, 7}, {3, 0},
                                      {3, 2}, {3, 4}, {3, 6}, {3, 8}};
    for (std::size_t i = 0; i < blackGroupPos.size(); i++) {
        Pos &pos = blackGroupPos[i];
        Stone *stone = Stone::children[i + Stone::ID_BLACK];
        stone->pos = pos;
        stone->alive = 1;
        Stone::map[pos] = stone;
    }
    return;
}

void Chess::moveForward(const Step *s, double &totalReward)
{
    Stone *stone = Stone::children[s->id];
    stone->moveTo(s->nextPos);
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = Stone::children[s->nextId];
        if (dst->color == Stone::COLOR_RED) {
            totalReward += dst->value;
        } else {
            totalReward -= dst->value;
        }
    }
    return;
}

void Chess::moveBack(const Step *s, double &totalReward)
{
    Stone *stone = Stone::children[s->id];
    Stone::map[s->pos] = stone;
    stone->pos = s->pos;
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = Stone::children[s->nextId];
        Stone::map[s->nextPos] = dst;
        dst->pos = s->nextPos;
        dst->alive = true;
        if (dst->color == Stone::COLOR_RED) {
            totalReward -= dst->value;
        } else {
            totalReward += dst->value;
        }
    } else {
        Stone::map[s->nextPos] = nullptr;
    }
    return;
}

void Chess::sample(int color, std::vector<Step *> &steps)
{
    int i0 = Stone::ID_RED;
    int in = Stone::ID_RED_END;
    if (color == Stone::COLOR_BLACK) {
        i0 = Stone::ID_BLACK;
        in = Stone::ID_BLACK_END;
    }
    for (int i = i0; i < in; i++) {
        if (stones[i]->alive == true) {
            stones[i]->getPossibleSteps(steps);
        }
    }
    return;
}

double Chess::minimizeAlpha(int color, int depth, double beta, double &totalReward)
{
    if (depth == 0) {
        return totalReward;
    }
    std::vector<Step *> steps;
    sample(color, steps);
    double alpha = Stone::value_infi;
    for (Step *s : steps) {
        moveForward(s, totalReward);
        int color_ = color == Stone::COLOR_RED ? Stone::COLOR_BLACK:Stone::COLOR_RED;
        double r = maximizeBeta(color_, depth - 1, alpha, totalReward);
        moveBack(s, totalReward);
        /* 剪枝 */
        if (r <= beta) {
            alpha = beta;
            break;
        }
        /* 最小化alpha */
        if (r < alpha) {
            alpha = r;
        }
    }
    std::cout<<"minimizeAlpha total reward:"<<totalReward<<std::endl;
    Steps::instance().put(steps);
    return alpha;
}

double Chess::maximizeBeta(int color, int depth, double apha, double &totalReward)
{
    if (depth == 0) {
        return totalReward;
    }
    std::vector<Step *> steps;
    sample(color, steps);
    double beta = -Stone::value_infi;
    for (Step *s : steps) {
        moveForward(s, totalReward);
        int color_ = color == Stone::COLOR_RED ? Stone::COLOR_BLACK:Stone::COLOR_RED;
        double r = minimizeAlpha(color_, depth - 1, beta, totalReward);
        moveBack(s, totalReward);
        /* 剪枝 */
        if (r >= apha) {
            beta = apha;
            break;
        }
        /* 最大化beta */
        if (r > beta) {
            beta = r;
        }
    }
    std::cout<<"maximizeBeta total reward:"<<totalReward<<std::endl;
    Steps::instance().put(steps);
    return beta;
}

Step Chess::alphaBetaPruning(int color, int depth)
{
    std::vector<Step*> steps;
    sample(color, steps);
    double beta = -Stone::value_infi;
    double totalReward = 0;
    Step* best = nullptr;
    for (Step *s : steps) {
        moveForward(s, totalReward);
        /* 计算对方收益 */
        int color_ = color == Stone::COLOR_RED ? Stone::COLOR_BLACK:Stone::COLOR_RED;
        double r = minimizeAlpha(color_, depth - 1, beta, totalReward);
        moveBack(s, totalReward);
        if (r > beta) {
            best = s;
            beta = r;
        }
    }
    Step step = *best;
    Steps::instance().put(steps);
    return step;
}

int Chess::isGameOver()
{
    if (redJiang.alive == false) {
        return Stone::COLOR_BLACK;
    }
    if (blackJiang.alive == false) {
        return Stone::COLOR_RED;
    }
    return Stone::COLOR_NONE;
}
