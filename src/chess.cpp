#include "chess.h"

/* ============================================================
 * 棋子-位置价值表 (Piece-Square Tables)
 * 红方视角 (黑方使用时应镜像翻转)
 * 数值越大表示该位置对该棋子越有利
 * 中心控制、进攻性位置给予更高权重
 * ============================================================ */

/* 车: 控制开放线, 占据中心 */
const double Chess::chePST[10][9] = {
    {0.00, 0.01, 0.02, 0.03, 0.05, 0.03, 0.02, 0.01, 0.00},
    {0.01, 0.05, 0.06, 0.07, 0.08, 0.07, 0.06, 0.05, 0.01},
    {0.02, 0.06, 0.08, 0.10, 0.13, 0.10, 0.08, 0.06, 0.02},
    {0.03, 0.07, 0.10, 0.12, 0.15, 0.12, 0.10, 0.07, 0.03},
    {0.03, 0.07, 0.10, 0.12, 0.18, 0.12, 0.10, 0.07, 0.03},
    {0.03, 0.07, 0.10, 0.12, 0.15, 0.12, 0.10, 0.07, 0.03},
    {0.02, 0.06, 0.08, 0.10, 0.13, 0.10, 0.08, 0.06, 0.02},
    {0.01, 0.05, 0.06, 0.07, 0.08, 0.07, 0.06, 0.05, 0.01},
    {0.01, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.01},
    {0.00, 0.01, 0.02, 0.03, 0.05, 0.03, 0.02, 0.01, 0.00}
};

/* 马: 中心区域价值高, 边角价值低 */
const double Chess::maPST[10][9] = {
    {0.00, 0.00, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00},
    {0.00, 0.02, 0.04, 0.05, 0.05, 0.05, 0.04, 0.02, 0.00},
    {0.01, 0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04, 0.01},
    {0.02, 0.05, 0.10, 0.14, 0.16, 0.14, 0.10, 0.05, 0.02},
    {0.02, 0.05, 0.10, 0.14, 0.18, 0.14, 0.10, 0.05, 0.02},
    {0.02, 0.05, 0.10, 0.14, 0.16, 0.14, 0.10, 0.05, 0.02},
    {0.01, 0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04, 0.01},
    {0.00, 0.02, 0.04, 0.05, 0.05, 0.05, 0.04, 0.02, 0.00},
    {0.00, 0.00, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 炮: 需要灵活位置, 中心及兵线价值高 */
const double Chess::paoPST[10][9] = {
    {0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.02, 0.00, 0.00},
    {0.01, 0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02, 0.01},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.03, 0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.05, 0.03},
    {0.03, 0.05, 0.08, 0.10, 0.14, 0.10, 0.08, 0.05, 0.03},
    {0.03, 0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.05, 0.03},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.01, 0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02, 0.01},
    {0.00, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.00},
    {0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.02, 0.00, 0.00}
};

/* 兵/卒: 过河后价值激增, 中路兵价值更高 */
const double Chess::bingPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.00},
    {0.01, 0.03, 0.06, 0.08, 0.10, 0.08, 0.06, 0.03, 0.01},
    {0.02, 0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.02},
    {0.03, 0.07, 0.10, 0.14, 0.20, 0.14, 0.10, 0.07, 0.03},
    {0.04, 0.08, 0.12, 0.16, 0.22, 0.16, 0.12, 0.08, 0.04},
    {0.03, 0.06, 0.09, 0.12, 0.15, 0.12, 0.09, 0.06, 0.03},
    {0.02, 0.04, 0.06, 0.08, 0.10, 0.08, 0.06, 0.04, 0.02},
    {0.01, 0.02, 0.03, 0.04, 0.06, 0.04, 0.03, 0.02, 0.01},
    {0.00, 0.00, 0.01, 0.02, 0.03, 0.02, 0.01, 0.00, 0.00}
};

/* 帅/将: 安全优先, 中路较安全 */
const double Chess::jiangPST[10][9] = {
    {0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.02, 0.03, 0.02, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.03, 0.05, 0.03, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.02, 0.03, 0.02, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.00, 0.00, 0.00}
};

/* 仕/士: 保护将/帅, 靠内线价值高 */
const double Chess::shiPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.05, 0.08, 0.05, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.05, 0.08, 0.05, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 相/象: 防守为主, 靠近将/帅区域价值高 */
const double Chess::xiangPST[10][9] = {
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00},
    {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
};

/* 位置价值函数 */
static double getPositionValue(int type, int color, int x, int y, const double pst[10][9])
{
    if (color == Stone::COLOR_RED) {
        /* 红方: 使用原表 */
        return pst[x][y];
    } else {
        /* 黑方: 上下镜像 (x -> 9-x) */
        return pst[9 - x][y];
    }
}

/* 增强评估函数: 材质 + 位置 */
double Chess::evaluate()
{
    double score = 0.0;
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        double matVal = s->value;
        double posVal = 0.0;
        switch (s->type) {
        case Stone::TYPE_CHE:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, chePST);   break;
        case Stone::TYPE_MA:    posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, maPST);    break;
        case Stone::TYPE_PAO:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, paoPST);   break;
        case Stone::TYPE_BING:  posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, bingPST);  break;
        case Stone::TYPE_JIANG: posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, jiangPST); break;
        case Stone::TYPE_SHI:   posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, shiPST);   break;
        case Stone::TYPE_XIANG: posVal = getPositionValue(s->type, s->color, s->pos.x, s->pos.y, xiangPST); break;
        }
        /* 返回值从AI(黑方)视角: 正=AI有利 */
        if (s->color == Stone::COLOR_BLACK) {
            score += matVal + posVal;
        } else {
            score -= matVal + posVal;
        }
    }
    return score;
}

Chess::Chess():
    stones(m_children),
    redChe1(Stone::ID_RED_CHE1, Stone::COLOR_RED, Pos(9, 0), &m_map, &m_children),
    redMa1(Stone::ID_RED_MA1, Stone::COLOR_RED, Pos(9, 1), &m_map, &m_children),
    redXiang1(Stone::ID_RED_XIANG1, Stone::COLOR_RED, Pos(9, 2), &m_map, &m_children),
    redShi1(Stone::ID_RED_SHI1, Stone::COLOR_RED, Pos(9, 3), &m_map, &m_children),
    redJiang(Stone::ID_RED_JIANG, Stone::COLOR_RED, Pos(9, 4), &m_map, &m_children),
    redShi2(Stone::ID_RED_SHI2, Stone::COLOR_RED, Pos(9, 5), &m_map, &m_children),
    redXiang2(Stone::ID_RED_XIANG2, Stone::COLOR_RED, Pos(9, 6), &m_map, &m_children),
    redMa2(Stone::ID_RED_MA2, Stone::COLOR_RED, Pos(9, 7), &m_map, &m_children),
    redChe2(Stone::ID_RED_CHE2, Stone::COLOR_RED, Pos(9, 8), &m_map, &m_children),
    redPao1(Stone::ID_RED_PAO1, Stone::COLOR_RED, Pos(7, 1), &m_map, &m_children),
    redPao2(Stone::ID_RED_PAO2, Stone::COLOR_RED, Pos(7, 7), &m_map, &m_children),
    redBing1(Stone::ID_RED_BING1, Stone::COLOR_RED, Pos(6, 0), &m_map, &m_children),
    redBing2(Stone::ID_RED_BING2, Stone::COLOR_RED, Pos(6, 2), &m_map, &m_children),
    redBing3(Stone::ID_RED_BING3, Stone::COLOR_RED, Pos(6, 4), &m_map, &m_children),
    redBing4(Stone::ID_RED_BING4, Stone::COLOR_RED, Pos(6, 6), &m_map, &m_children),
    redBing5(Stone::ID_RED_BING5, Stone::COLOR_RED, Pos(6, 8), &m_map, &m_children),
    blackChe1(Stone::ID_BLACK_CHE1, Stone::COLOR_BLACK, Pos(0, 0), &m_map, &m_children),
    blackMa1(Stone::ID_BLACK_MA1, Stone::COLOR_BLACK, Pos(0, 1), &m_map, &m_children),
    blackXiang1(Stone::ID_BLACK_XIANG1, Stone::COLOR_BLACK, Pos(0, 2), &m_map, &m_children),
    blackShi1(Stone::ID_BLACK_SHI1, Stone::COLOR_BLACK, Pos(0, 3), &m_map, &m_children),
    blackJiang(Stone::ID_BLACK_JIANG, Stone::COLOR_BLACK, Pos(0, 4), &m_map, &m_children),
    blackShi2(Stone::ID_BLACK_SHI2, Stone::COLOR_BLACK, Pos(0, 5), &m_map, &m_children),
    blackXiang2(Stone::ID_BLACK_XIANG2, Stone::COLOR_BLACK, Pos(0, 6), &m_map, &m_children),
    blackMa2(Stone::ID_BLACK_MA2, Stone::COLOR_BLACK, Pos(0, 7), &m_map, &m_children),
    blackChe2(Stone::ID_BLACK_CHE2, Stone::COLOR_BLACK, Pos(0, 8), &m_map, &m_children),
    blackPao1(Stone::ID_BLACK_PAO1, Stone::COLOR_BLACK, Pos(2, 1), &m_map, &m_children),
    blackPao2(Stone::ID_BLACK_PAO2, Stone::COLOR_BLACK, Pos(2, 7), &m_map, &m_children),
    blackBing1(Stone::ID_BLACK_BING1, Stone::COLOR_BLACK, Pos(3, 0), &m_map, &m_children),
    blackBing2(Stone::ID_BLACK_BING2, Stone::COLOR_BLACK, Pos(3, 2), &m_map, &m_children),
    blackBing3(Stone::ID_BLACK_BING3, Stone::COLOR_BLACK, Pos(3, 4), &m_map, &m_children),
    blackBing4(Stone::ID_BLACK_BING4, Stone::COLOR_BLACK, Pos(3, 6), &m_map, &m_children),
    blackBing5(Stone::ID_BLACK_BING5, Stone::COLOR_BLACK, Pos(3, 8), &m_map, &m_children)
{
}

/* 拷贝构造函数: 深度复制所有棋子的状态 */
Chess::Chess(const Chess &other):
    m_children(),  /* start empty, stones will be filled by stone constructors below */
    stones(m_children),
    redChe1(other.redChe1.id, other.redChe1.color, other.redChe1.pos, &m_map, &m_children),
    redMa1(other.redMa1.id, other.redMa1.color, other.redMa1.pos, &m_map, &m_children),
    redXiang1(other.redXiang1.id, other.redXiang1.color, other.redXiang1.pos, &m_map, &m_children),
    redShi1(other.redShi1.id, other.redShi1.color, other.redShi1.pos, &m_map, &m_children),
    redJiang(other.redJiang.id, other.redJiang.color, other.redJiang.pos, &m_map, &m_children),
    redShi2(other.redShi2.id, other.redShi2.color, other.redShi2.pos, &m_map, &m_children),
    redXiang2(other.redXiang2.id, other.redXiang2.color, other.redXiang2.pos, &m_map, &m_children),
    redMa2(other.redMa2.id, other.redMa2.color, other.redMa2.pos, &m_map, &m_children),
    redChe2(other.redChe2.id, other.redChe2.color, other.redChe2.pos, &m_map, &m_children),
    redPao1(other.redPao1.id, other.redPao1.color, other.redPao1.pos, &m_map, &m_children),
    redPao2(other.redPao2.id, other.redPao2.color, other.redPao2.pos, &m_map, &m_children),
    redBing1(other.redBing1.id, other.redBing1.color, other.redBing1.pos, &m_map, &m_children),
    redBing2(other.redBing2.id, other.redBing2.color, other.redBing2.pos, &m_map, &m_children),
    redBing3(other.redBing3.id, other.redBing3.color, other.redBing3.pos, &m_map, &m_children),
    redBing4(other.redBing4.id, other.redBing4.color, other.redBing4.pos, &m_map, &m_children),
    redBing5(other.redBing5.id, other.redBing5.color, other.redBing5.pos, &m_map, &m_children),
    blackChe1(other.blackChe1.id, other.blackChe1.color, other.blackChe1.pos, &m_map, &m_children),
    blackMa1(other.blackMa1.id, other.blackMa1.color, other.blackMa1.pos, &m_map, &m_children),
    blackXiang1(other.blackXiang1.id, other.blackXiang1.color, other.blackXiang1.pos, &m_map, &m_children),
    blackShi1(other.blackShi1.id, other.blackShi1.color, other.blackShi1.pos, &m_map, &m_children),
    blackJiang(other.blackJiang.id, other.blackJiang.color, other.blackJiang.pos, &m_map, &m_children),
    blackShi2(other.blackShi2.id, other.blackShi2.color, other.blackShi2.pos, &m_map, &m_children),
    blackXiang2(other.blackXiang2.id, other.blackXiang2.color, other.blackXiang2.pos, &m_map, &m_children),
    blackMa2(other.blackMa2.id, other.blackMa2.color, other.blackMa2.pos, &m_map, &m_children),
    blackChe2(other.blackChe2.id, other.blackChe2.color, other.blackChe2.pos, &m_map, &m_children),
    blackPao1(other.blackPao1.id, other.blackPao1.color, other.blackPao1.pos, &m_map, &m_children),
    blackPao2(other.blackPao2.id, other.blackPao2.color, other.blackPao2.pos, &m_map, &m_children),
    blackBing1(other.blackBing1.id, other.blackBing1.color, other.blackBing1.pos, &m_map, &m_children),
    blackBing2(other.blackBing2.id, other.blackBing2.color, other.blackBing2.pos, &m_map, &m_children),
    blackBing3(other.blackBing3.id, other.blackBing3.color, other.blackBing3.pos, &m_map, &m_children),
    blackBing4(other.blackBing4.id, other.blackBing4.color, other.blackBing4.pos, &m_map, &m_children),
    blackBing5(other.blackBing5.id, other.blackBing5.color, other.blackBing5.pos, &m_map, &m_children),
    history(other.history)
{
    /* Sync alive flags from source */
    for (int i = 0; i < 32; i++) {
        Stone *src = other.m_children[i];
        Stone *dst = m_children[i];
        if (src && dst) {
            dst->alive = src->alive;
        }
    }
    /* Rebuild m_map from only alive stones — dead stones may have left
     * stale entries during construction that overlap with alive stones. */
    m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s && s->alive) {
            m_map[s->pos] = s;
        }
    }
}

Chess& Chess::operator=(const Chess &other)
{
    if (this == &other) return *this;
    /* Copy state: positions, alive flags */
    for (int i = 0; i < 32; i++) {
        Stone *src = other.m_children[i];
        Stone *dst = m_children[i];
        if (src && dst) {
            dst->pos = src->pos;
            dst->alive = src->alive;
        }
    }
    /* Rebuild m_map */
    m_map.clear();
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s && s->alive) {
            m_map[s->pos] = s;
        }
    }
    history = other.history;
    return *this;
}

Chess::~Chess()
{
}

Stone *Chess::get(int id)
{
    return m_children[id];
}

void Chess::reset()
{
    m_map.clear();
    /* 红方 */
    std::vector<Pos> redGroupPos = {{9, 0}, {9, 1}, {9, 2}, {9, 3},
                                    {9, 4}, {9, 5}, {9, 6}, {9, 7},
                                    {9, 8}, {7, 1}, {7, 7}, {6, 0},
                                    {6, 2}, {6, 4}, {6, 6}, {6, 8}};

    for (std::size_t i = 0; i < redGroupPos.size(); i++) {
        const Pos &pos = redGroupPos[i];
        Stone *stone = m_children[i + Stone::ID_RED];
        stone->pos = pos;
        stone->alive = 1;
        m_map[pos] = stone;
    }
    /* 黑方 */
    std::vector<Pos> blackGroupPos = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                      {0, 4}, {0, 5}, {0, 6}, {0, 7},
                                      {0, 8}, {2, 1}, {2, 7}, {3, 0},
                                      {3, 2}, {3, 4}, {3, 6}, {3, 8}};
    for (std::size_t i = 0; i < blackGroupPos.size(); i++) {
        Pos &pos = blackGroupPos[i];
        Stone *stone = m_children[i + Stone::ID_BLACK];
        stone->pos = pos;
        stone->alive = 1;
        m_map[pos] = stone;
    }
    return;
}

void Chess::moveForward(const Step *s, double &totalReward)
{
    Stone *stone = m_children[s->id];
    stone->moveTo(s->nextPos);
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
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
    Stone *stone = m_children[s->id];
    m_map[s->pos] = stone;
    stone->pos = s->pos;
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
        m_map[s->nextPos] = dst;
        dst->pos = s->nextPos;
        dst->alive = true;
        if (dst->color == Stone::COLOR_RED) {
            totalReward -= dst->value;
        } else {
            totalReward += dst->value;
        }
    } else {
        m_map[s->nextPos] = nullptr;
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
        if (m_children[i]->alive) {
            m_children[i]->getPossibleSteps(steps);
        }
    }
    return;
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

bool Chess::isInCheck(int color)
{
    /* 找color方的将/帅位置 */
    Stone *jiang = (color == Stone::COLOR_RED) ? &redJiang : &blackJiang;
    if (jiang == nullptr || jiang->alive == false) {
        return false;
    }

    /* 检查对方所有棋子是否能攻击到将/帅 */
    int enemyColor = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    int i0 = (enemyColor == Stone::COLOR_RED) ? Stone::ID_RED : Stone::ID_BLACK;
    int in = (enemyColor == Stone::COLOR_RED) ? Stone::ID_RED_END : Stone::ID_BLACK_END;

    for (int i = i0; i < in; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        if (s->tryMoveTo(jiang->pos)) {
            Stone *dst = m_map[jiang->pos];
            /* 模拟: 如果目标位置有将/帅, 检查是否能吃 */
            if (dst == jiang) {
                return true;
            }
        }
    }
    /* 检查飞将: 两将/帅同列且中间无子 */
    if (redJiang.alive && blackJiang.alive) {
        if (redJiang.pos.y == blackJiang.pos.y) {
            int count = m_map.countStoneOnLine(redJiang.pos, blackJiang.pos);
            if (count == 0) {
                return true;
            }
        }
    }
    return false;
}

unsigned long long Chess::computeHash()
{
    unsigned long long h = 0;
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        unsigned long long v = (unsigned long long)i;
        v ^= (unsigned long long)(s->pos.x) << 8;
        v ^= (unsigned long long)(s->pos.y) << 12;
        h ^= v;
    }
    return h;
}

void Chess::pushHistory()
{
    history.push_back(computeHash());
}

bool Chess::isRepetition()
{
    unsigned long long h = computeHash();
    int count = 0;
    for (unsigned long long hh : history) {
        if (hh == h) {
            count++;
        }
    }
    /* 同一局面出现3次判和 */
    return count >= 3;
}
