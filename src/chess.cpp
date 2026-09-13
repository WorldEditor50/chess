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
    /*
       必须显式 reset(): StoneMap 的默认构造现在会把整张表清空, 而 32 个棋子
       构造函数只写了各自所在的 32 个格子 —— 不 reset 的话剩余 58 格虽是 nullptr
       (安全), 但棋盘状态没有保证。reset() 同时初始化 sideToMove / halfMoveClock /
       history。
    */
    reset();
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
    sideToMove = other.sideToMove;
    halfMoveClock = other.halfMoveClock;
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
    sideToMove = other.sideToMove;
    halfMoveClock = other.halfMoveClock;
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
    history.clear();
    sideToMove = Stone::COLOR_RED;
    halfMoveClock = 0;
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

/* ============================================================
 *  落子 / 回退
 * ============================================================ */

/*
 * applyMove: 只改棋盘 (m_map + 棋子位置 + 被吃子 alive), 不做任何记账。
 *   返回 false 表示这一步在棋盘上根本无法执行 (棋子已死 / 目标格有己方子),
 *   此时棋盘保持原样。moveForward 与 isLegalMove 共用它, 避免两份"落子"逻辑
 *   走偏。
 */
bool Chess::applyMove(const Step *s)
{
    /*
       先做边界校验, 再看棋子。原来直接 `m_children[s->id]` / `moveTo(s->nextPos)`:
       id 和坐标都是**外部传进来**的 (搜索返回的 Step、回放记录、数据库行), 越界时
       std::array 和 StoneMap::data[10][9] 都是不检查的 —— 一次越界写就落在 Chess
       对象的相邻成员上, 表现出来是"AI 明明有棋可走却说无合法走法"甚至段错误。
       这里把"这一步根本没法在棋盘上执行"变成返回 false (调用方本来就要处理 false)。
    */
    if (s == nullptr || !s->valid) {
        return false;
    }
    if (s->id < 0 || s->id >= (int)m_children.size()) {
        return false;
    }
    if (!m_map.isInner(s->pos) || !m_map.isInner(s->nextPos)) {
        return false;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr || stone->alive == false) {
        return false;
    }
    if (!(stone->pos == s->pos)) {
        /* Step 里的起点和棋子当前位置不一致 -> 这是一个过期的走法 */
        return false;
    }
    /* moveTo() 自带 tryMoveTo() 形状校验, 并会在目标格是己方子时返回 false */
    return stone->moveTo(s->nextPos);
}

void Chess::undoMove(const Step *s)
{
    if (s == nullptr || s->id < 0 || s->id >= (int)m_children.size()) {
        return;
    }
    if (!m_map.isInner(s->pos) || !m_map.isInner(s->nextPos)) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    m_map[s->pos] = stone;
    stone->pos = s->pos;
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
        if (dst == nullptr) {
            return;
        }
        m_map[s->nextPos] = dst;
        dst->pos = s->nextPos;
        dst->alive = true;
    } else {
        m_map[s->nextPos] = nullptr;
    }
}

void Chess::moveForward(const Step *s, double &totalReward)
{
    if (s == nullptr) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    const int mover = stone->color;
    /* 落子失败时不做任何记账: 以前的实现无视返回值, 结果 moveBack 会把
     * 一个从未被吃掉的棋子"复活"到终点格, 把棋盘写坏。 */
    if (applyMove(s) == false) {
        return;
    }
    bool captured = (s->nextId != Stone::ID_NONE);
    if (captured) {
        Stone *dst = m_children[s->nextId];
        if (dst->color == Stone::COLOR_RED) {
            totalReward += dst->value;
        } else {
            totalReward -= dst->value;
        }
    }
    sideToMove = (mover == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    /*
       先在"更新计数之前"记录, HistoryRecord.halfMoveClock 保存的就是走这一步
       **之前**的值, moveBack 才能精确恢复 (否则回退后计数会停在走完后的值上)。
    */
    pushHistory();
    /* 60 回合自然限着: 吃子或走兵/卒才重置计数 */
    if (captured || stone->type == Stone::TYPE_BING) {
        halfMoveClock = 0;
    } else {
        halfMoveClock++;
    }
    return;
}

void Chess::moveBack(const Step *s, double &totalReward)
{
    if (s == nullptr) {
        return;
    }
    Stone *stone = m_children[s->id];
    if (stone == nullptr) {
        return;
    }
    /* 该步没有被真正执行过 (applyMove 失败) -> 什么都不用回退 */
    if (!(stone->pos == s->nextPos)) {
        return;
    }
    undoMove(s);
    if (s->nextId != Stone::ID_NONE) {
        Stone *dst = m_children[s->nextId];
        if (dst->color == Stone::COLOR_RED) {
            totalReward -= dst->value;
        } else {
            totalReward += dst->value;
        }
    }
    sideToMove = stone->color;
    if (!history.empty()) {
        /* 恢复到走这一步之前的计数 */
        halfMoveClock = history.back().halfMoveClock;
        history.pop_back();
    }
    return;
}

/* ============================================================
 *  走法生成
 * ============================================================ */

void Chess::samplePseudo(int color, std::vector<Step *> &steps)
{
    int i0 = Stone::ID_RED;
    int in = Stone::ID_RED_END;
    if (color == Stone::COLOR_BLACK) {
        i0 = Stone::ID_BLACK;
        in = Stone::ID_BLACK_END;
    }
    for (int i = i0; i < in; i++) {
        if (m_children[i] != nullptr && m_children[i]->alive) {
            m_children[i]->getPossibleSteps(steps);
        }
    }
    return;
}

/*
 * sample: 伪合法走法 + 合法性过滤。
 *
 * 过滤掉的是"走后自家将/帅被攻击"的走法 —— 包括被将军时不应将、以及走成
 * 两将照面。这是中国象棋的核心规则, 以前完全缺失 (isInCheck() 写好了却从没
 * 被调用过), 于是 AI 和玩家都能自杀 / 不应将 / 主动走出照面。
 */
void Chess::sample(int color, std::vector<Step *> &steps)
{
    std::vector<Step *> pseudo;
    samplePseudo(color, pseudo);
    /* "当前是否被将" 与具体走法无关, 整批只算一次 */
    const bool inCheck = isInCheck(color);
    /* 被拒绝的走法攒起来一次性还回对象池 (逐个 put 会为每个走法分配一次临时
     * vector, 而这是搜索热路径) */
    std::vector<Step *> rejected;
    rejected.reserve(pseudo.size());
    for (std::size_t i = 0; i < pseudo.size(); i++) {
        Step *s = pseudo[i];
        if (isLegalMoveInternal(color, s, inCheck)) {
            steps.push_back(s);
        } else {
            rejected.push_back(s);
        }
    }
    if (!rejected.empty()) {
        Steps::instance().put(rejected);
    }
    return;
}

/* ============================================================
 *  攻击 / 将军 / 合法性
 * ============================================================ */

/*
 * isAttacked: target 是否被 byColor 方的棋子攻击。
 *
 * 按棋子类型做几何判定, 而不是对 16 个敌方棋子逐个调 tryMoveTo() ——
 * 后者每次都要重建走法形状, 而本函数处在一个会被搜索热点反复调用的位置。
 * 仕/士 与 相/象 永远到不了对方九宫, 直接跳过。
 */
bool Chess::isAttacked(const Pos &target, int byColor)
{
    int i0 = (byColor == Stone::COLOR_RED) ? Stone::ID_RED : Stone::ID_BLACK;
    int in = (byColor == Stone::COLOR_RED) ? Stone::ID_RED_END : Stone::ID_BLACK_END;

    for (int i = i0; i < in; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        switch (s->type) {
        case Stone::TYPE_CHE:
            /* 车: 同线且中间无子 */
            if ((s->pos.x == target.x || s->pos.y == target.y)
                    && m_map.countStoneOnLine(s->pos, target) == 0) {
                return true;
            }
            break;
        case Stone::TYPE_PAO:
            /* 炮: 同线且中间恰有一个炮架 */
            if ((s->pos.x == target.x || s->pos.y == target.y)
                    && m_map.countStoneOnLine(s->pos, target) == 1) {
                return true;
            }
            break;
        case Stone::TYPE_MA: {
            /* 马: 日字且马腿无子 */
            int dx = std::abs(target.x - s->pos.x);
            int dy = std::abs(target.y - s->pos.y);
            if ((dx == 2 && dy == 1) || (dx == 1 && dy == 2)) {
                Pos leg = s->pos;
                if (dx == 2) {
                    leg.x += (target.x > s->pos.x) ? 1 : -1;
                } else {
                    leg.y += (target.y > s->pos.y) ? 1 : -1;
                }
                if (m_map[leg] == nullptr) {
                    return true;
                }
            }
            break;
        }
        case Stone::TYPE_BING:
            /* 兵/卒: 只能攻击它自己走得到的那一格 */
            if (byColor == Stone::COLOR_RED) {
                if (target.x == s->pos.x - 1 && target.y == s->pos.y) {
                    return true;
                }
                if (s->pos.x <= 4 && target.x == s->pos.x
                        && std::abs(target.y - s->pos.y) == 1) {
                    return true;
                }
            } else {
                if (target.x == s->pos.x + 1 && target.y == s->pos.y) {
                    return true;
                }
                if (s->pos.x >= 5 && target.x == s->pos.x
                        && std::abs(target.y - s->pos.y) == 1) {
                    return true;
                }
            }
            break;
        case Stone::TYPE_JIANG: {
            /* 飞将: 只有当目标格上是对方将/帅时才是"攻击" */
            Stone *opp = m_map[target];
            if (opp != nullptr && opp->type == Stone::TYPE_JIANG
                    && opp->color != s->color
                    && opp->pos.y == s->pos.y
                    && m_map.countStoneOnLine(s->pos, target) == 0) {
                return true;
            }
            /* 将/帅: 对方九宫内相邻一格 */
            if (std::abs(target.x - s->pos.x) + std::abs(target.y - s->pos.y) == 1
                    && target.y >= 3 && target.y <= 5) {
                if (byColor == Stone::COLOR_RED) {
                    if (target.x >= 7 && target.x <= 9) {
                        return true;
                    }
                } else {
                    if (target.x >= 0 && target.x <= 2) {
                        return true;
                    }
                }
            }
            break;
        }
        default:
            /* 仕/士, 相/象: 活动范围限制在己方, 无法攻击对方将/帅 */
            break;
        }
    }
    return false;
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
    Stone *jiang = (color == Stone::COLOR_RED) ? &redJiang : &blackJiang;
    if (jiang->alive == false) {
        return false;
    }
    int enemyColor = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    return isAttacked(jiang->pos, enemyColor);
}

bool Chess::isLegalMove(int color, const Step *s)
{
    return isLegalMoveInternal(color, s, isInCheck(color));
}

/*
 * isLegalMoveInternal - isLegalMove() 的内部版本。
 *
 * inCheck 由调用方预先算好: 同一次 sample() 里几十个候选走法共享同一个"当前是否
 * 被将"的结论 (它只取决于走之前的局面), 于是热路径上每个走法的成本从"落子 +
 * isAttacked + 回退"降到几次坐标比较。
 *
 * 快速路径的判据: 当"移动的不是将/帅"且"当前没有被将"时, 这一步只可能通过下面
 * 两条途径改变己方将/帅的安全, 其余情况必然合法 ——
 *   (a) 改变某条经过己方将/帅的横线/竖线上的阻挡 (敌方车、炮、以及飞将规则);
 *   (b) 打开或封住某个正对己方将/帅的敌方马的蹩马腿。
 * 敌方兵与敌方将的攻击只取决于几何位置和将/帅自身所在格, 与其它子的阻挡无关,
 * 因此在"没动将/帅"的前提下不会因为这一步而改变。
 */
bool Chess::isLegalMoveInternal(int color, const Step *s, bool inCheck)
{
    if (s == nullptr || s->valid == false) {
        return false;
    }
    Stone *mover = m_children[s->id];
    if (mover == nullptr || mover->alive == false || mover->color != color) {
        return false;
    }
    /* 形状校验 (不改变棋盘) */
    if (mover->tryMoveTo(s->nextPos) == false) {
        return false;
    }
    Stone *dst = m_map[s->nextPos];
    if (dst != nullptr && dst->color == mover->color) {
        return false;
    }

    Stone *jiang = (color == Stone::COLOR_RED) ? &redJiang : &blackJiang;
    if (jiang->alive && mover->type != Stone::TYPE_JIANG && !inCheck) {
        const Pos g = jiang->pos;
        bool affects = (s->pos.x == g.x || s->pos.y == g.y
                        || s->nextPos.x == g.x || s->nextPos.y == g.y);
        if (!affects) {
            const int enemy = (color == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                          : Stone::COLOR_RED;
            const int i0 = (enemy == Stone::COLOR_RED) ? Stone::ID_RED : Stone::ID_BLACK;
            const int in = (enemy == Stone::COLOR_RED) ? Stone::ID_RED_END
                                                       : Stone::ID_BLACK_END;
            for (int i = i0; i < in; i++) {
                Stone *e = m_children[i];
                if (e == nullptr || e->alive == false || e->type != Stone::TYPE_MA) {
                    continue;
                }
                const int dx = std::abs(g.x - e->pos.x);
                const int dy = std::abs(g.y - e->pos.y);
                if (!((dx == 2 && dy == 1) || (dx == 1 && dy == 2))) {
                    continue;
                }
                Pos leg = e->pos;
                if (dx == 2) {
                    leg.x += (g.x > e->pos.x) ? 1 : -1;
                } else {
                    leg.y += (g.y > e->pos.y) ? 1 : -1;
                }
                if (leg == s->pos || leg == s->nextPos) {
                    affects = true;
                    break;
                }
            }
        }
        if (!affects) {
            return true;
        }
    }

    if (applyMove(s) == false) {
        return false;
    }
    bool bad = isInCheck(color);
    undoMove(s);
    return !bad;
}

bool Chess::hasLegalMoves(int color)
{
    std::vector<Step *> steps;
    sample(color, steps);
    bool any = !steps.empty();
    Steps::instance().put(steps);
    return any;
}

bool Chess::isRepetition()
{
    unsigned long long h = computeHash();
    int count = 0;
    /*
       只需要看最近 halfMoveClock 步: 局面只有在"这段没有吃子、没有走兵"的可逆
       区间内才可能重复。全量扫描在长对局里会把 O(history) 带进每个搜索节点。
    */
    int window = halfMoveClock;
    if (window > (int)history.size()) {
        window = (int)history.size();
    }
    int begin = (int)history.size() - window;
    for (int i = (int)history.size() - 1; i >= begin; i--) {
        if (history[i].hash == h) {
            count++;
        }
    }
    /* 同一局面出现3次判和 */
    return count >= 3;
}

bool Chess::isDraw()
{
    if (isGameOver() != Stone::COLOR_NONE) {
        return false;
    }
    /* 三次重复局面 */
    if (isRepetition()) {
        return true;
    }
    /* 60 回合 (120 半回合) 内双方都没有吃子 */
    if (halfMoveClock >= 120) {
        return true;
    }
    return false;
}

int Chess::getResult(int colorToMove)
{
    int win = isGameOver();
    if (win != Stone::COLOR_NONE) {
        return (win == Stone::COLOR_RED) ? RESULT_RED_WIN : RESULT_BLACK_WIN;
    }
    if (isDraw()) {
        return RESULT_DRAW;
    }
    /*
       将杀 / 困毙: 轮到走的一方没有任何合法走法。中国象棋里困毙同样判负,
       所以不需要区分两者。注意 isGameOver() 保持"便宜"的语义 (只看将帅是否
       存活), 这个较贵的判定只给 GUI / 训练循环的终局判断用。
    */
    if (!hasLegalMoves(colorToMove)) {
        return (colorToMove == Stone::COLOR_RED) ? RESULT_BLACK_WIN : RESULT_RED_WIN;
    }
    return RESULT_ONGOING;
}

/*
 * Zobrist 表: 32 个棋子 x 90 个格子 + 1 个"轮到黑方"键。
 *
 * 原来的 computeHash() 是把 (棋子id | x<<8 | y<<12) 直接异或起来, 结构化的键
 * 之间相关性很强 —— 用作重复局面判定勉强够, 但拿去当**置换表**的索引就会频繁
 * 碰撞: 碰撞后 TT 会返回别的局面的分数, 搜索直接出错。
 * 这里换成标准的 Zobrist 随机键 (splitmix64 + 固定种子, 保证每次运行一致)。
 */
const std::array<unsigned long long, 32*90 + 1> &Chess::zobrist()
{
    static const std::array<unsigned long long, 32*90 + 1> table = []() {
        std::array<unsigned long long, 32*90 + 1> t{};
        unsigned long long x = 0x9E3779B97F4A7C15ULL;
        for (std::size_t i = 0; i < t.size(); i++) {
            x += 0x9E3779B97F4A7C15ULL;
            unsigned long long z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            t[i] = z ^ (z >> 31);
        }
        return t;
    }();
    return table;
}

unsigned long long Chess::computeHash()
{
    const std::array<unsigned long long, 32*90 + 1> &z = zobrist();
    unsigned long long h = 0;
    for (int i = 0; i < 32; i++) {
        Stone *s = m_children[i];
        if (s == nullptr || s->alive == false) {
            continue;
        }
        h ^= z[(std::size_t)i*90 + (std::size_t)(s->pos.x*9 + s->pos.y)];
    }
    if (sideToMove == Stone::COLOR_BLACK) {
        h ^= z[32*90];
    }
    return h;
}

void Chess::pushHistory()
{
    HistoryRecord r;
    r.hash = computeHash();
    /* 记录当前 (即这一步**之前**) 的计数, 供 moveBack 精确恢复 */
    r.halfMoveClock = halfMoveClock;
    history.push_back(r);
}
