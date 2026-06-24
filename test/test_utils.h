#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "chess.h"
#include <chrono>
#include <cassert>
#include <string>
#include <map>
#include <iomanip>
#include <sstream>

/* ============================================================
 * 搜索统计结构
 * ============================================================ */
struct SearchStats
{
    long long totalNodesExamined = 0;
    long long totalTimeMs = 0;
    int gameCount = 0;
    int redWins = 0;
    int blackWins = 0;
    int draws = 0;
    int totalMoves = 0;

    void reset()
    {
        totalNodesExamined = 0;
        totalTimeMs = 0;
        gameCount = 0;
        redWins = 0;
        blackWins = 0;
        draws = 0;
        totalMoves = 0;
    }
};

/* ============================================================
 * 棋盘打印 (控制台输出)
 * ============================================================ */
static const char* stoneNameForPrint(int type, int color)
{
    if (color == Stone::COLOR_RED) {
        switch (type) {
            case Stone::TYPE_CHE:   return "车";
            case Stone::TYPE_MA:    return "马";
            case Stone::TYPE_XIANG: return "相";
            case Stone::TYPE_SHI:   return "仕";
            case Stone::TYPE_JIANG: return "帅";
            case Stone::TYPE_PAO:   return "炮";
            case Stone::TYPE_BING:  return "兵";
        }
    } else {
        switch (type) {
            case Stone::TYPE_CHE:   return "车";
            case Stone::TYPE_MA:    return "马";
            case Stone::TYPE_XIANG: return "象";
            case Stone::TYPE_SHI:   return "士";
            case Stone::TYPE_JIANG: return "将";
            case Stone::TYPE_PAO:   return "炮";
            case Stone::TYPE_BING:  return "卒";
        }
    }
    return "?";
}

/* 无颜色版本 (Windows兼容) */
static void printChessBoardPlain(Chess &chess)
{
    static const char* pieces[10][9];
    // 初始化
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 9; j++)
            pieces[i][j] = "  ";

    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s && s->alive) {
            const char *name = stoneNameForPrint(s->type, s->color);
            pieces[s->pos.x][s->pos.y] = name;
        }
    }

    printf("   0  1  2  3  4  5  6  7  8\n");
    printf("  -----------------------------\n");
    for (int x = 0; x < 10; x++) {
        printf("%d ", x);
        for (int y = 0; y < 9; y++) {
            printf("|%s", pieces[x][y]);
        }
        printf("|\n");
        printf("  -----------------------------\n");
    }
    printf("\n");
}

/* ============================================================
 * 计时器工具
 * ============================================================ */
class Timer
{
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    long long elapsedMs()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    long long elapsedUs()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

/* ============================================================
 * 走法描述
 * ============================================================ */
static std::string stepToString(const Step &step, Chess &chess)
{
    if (step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0) {
        return "(无合法走法)";
    }
    Stone *s = chess.stones[step.id];
    Stone *dst = (step.nextId != Stone::ID_NONE) ? chess.stones[step.nextId] : nullptr;
    std::stringstream ss;
    ss << stoneNameForPrint(s->type, s->color)
       << "(" << step.pos.x << "," << step.pos.y << ")"
       << " -> (" << step.nextPos.x << "," << step.nextPos.y << ")";
    if (dst) {
        ss << " 吃" << stoneNameForPrint(dst->type, dst->color);
    }
    return ss.str();
}

/* ============================================================
 * 局面评估分析
 * ============================================================ */
static void printEvaluation(Chess &chess)
{
    double score = chess.evaluate();
    int redCount = 0, blackCount = 0;
    double redMat = 0, blackMat = 0;

    for (int i = 0; i < 32; i++) {
        Stone *s = chess.stones[i];
        if (s == nullptr || s->alive == false) continue;
        if (s->color == Stone::COLOR_RED) {
            redCount++;
            redMat += s->value;
        } else {
            blackCount++;
            blackMat += s->value;
        }
    }

    printf("评估: score=%.4f\n", score);
    printf("  红方: %d子, 材质=%.4f\n", redCount, redMat);
    printf("  黑方: %d子, 材质=%.4f\n", blackCount, blackMat);
    printf("  材质差: %.4f\n", redMat - blackMat);

    int gameResult = chess.isGameOver();
    if (gameResult == Stone::COLOR_RED) {
        printf("  ==> 红方胜!\n");
    } else if (gameResult == Stone::COLOR_BLACK) {
        printf("  ==> 黑方胜!\n");
    }
}

#endif // TEST_UTILS_H
