#ifndef CHESS_TEST_OPENING_BOOK_H
#define CHESS_TEST_OPENING_BOOK_H
/*
 * opening_book.h - "固定开局集" (配对对局的起始局面)
 * ============================================================================
 *
 * 抽成头文件的理由 (本工程的惯例: **第二份拷贝出现才抽**): 这份逻辑原来只长在
 * `bench_anchor_main.cpp` 里, 2026-10 做"序列编码 A/B"时 `bench_agent_arena` 也需要
 * 同一套东西 —— 而**开局集是两次读数可不可比的前提**, 有两份拷贝迟早会漂移成
 * "两边都叫固定开局集, 但生成规则不一样"。
 *
 * 为什么需要它 (bench_anchor 那份注释里的三条, 原样保留在这里):
 *   1. 随机源是**局部的** `std::mt19937_64`, 只由 `(seed, index)` 决定 ——
 *      **不碰 `RL::Random`**。于是开局集与 agent 内部的随机流消耗顺序无关:
 *      换个权重、换个搜索预算, 跑的还是同一批局面 (否则"两次运行不可比")。
 *   2. 开局集印出 FNV 指纹, 两次运行指纹不同 = 开局集变了 (改过走法生成?), 此时
 *      两次结果**不可比** —— 宁可吵, 也不能静默地比两批不同的局面。
 *   3. 刻意**不做**从文件回读: `Step` 没有公开的"从坐标重建"入口, 手搓 `Step` 的
 *      id/nextId 不一致是静默错误的高发区。
 *
 * 用法:
 *   std::vector<Opening> book; unsigned long long fp = 0;
 *   if (!makeOpenings(scratch, count, plies, seed, book, fp)) { ... }
 *   const int color = applyOpening(chess, book[i % book.size()]);
 */
#include <cstdio>
#include <random>
#include <vector>

#include "chess.h"

typedef std::vector<Step> Opening;

inline unsigned long long fnv1a(unsigned long long h, unsigned long long v)
{
    h ^= v;
    return h * 1099511628211ull;
}

/*
 * 造 `count` 个开局, 每个 `plies` 手随机合法棋 (每个开局一条独立的 RNG 流)。
 * 指纹覆盖每个开局的每一步 (从/到坐标), 与开局集一一对应。
 */
inline bool makeOpenings(Chess &scratch, int count, int plies, unsigned seed,
                         std::vector<Opening> &out, unsigned long long &fingerprint)
{
    out.clear();
    fingerprint = 1469598103934665603ull;   /* FNV-1a 偏移基 */
    for (int i = 0; i < count; i++) {
        std::mt19937_64 rng((unsigned long long)seed * 7919ull
                            + (unsigned long long)i * 104729ull + 17ull);
        scratch.reset();
        Opening op;
        for (int p = 0; p < plies; p++) {
            if (scratch.getResult(scratch.sideToMove) != Chess::RESULT_ONGOING) {
                break;
            }
            std::vector<Step *> legal;
            scratch.sample(scratch.sideToMove, legal);
            if (legal.empty()) {
                Steps::instance().put(legal);
                break;
            }
            std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
            const Step s = *legal[(std::size_t)pick(rng)];
            Steps::instance().put(legal);
            op.push_back(s);
            double dummy = 0.0;
            scratch.moveForward(&s, dummy);
        }
        if (op.empty()) {
            std::printf("[错误] 第 %d 个开局造不出来 (开局的随机手数太多?)\n", i);
            return false;
        }
        for (std::size_t k = 0; k < op.size(); k++) {
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].pos.y);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.x);
            fingerprint = fnv1a(fingerprint, (unsigned long long)op[k].nextPos.y);
        }
        out.push_back(op);
    }
    return true;
}

/* 把开局摆上棋盘; 返回该谁走 */
inline int applyOpening(Chess &c, const Opening &op)
{
    c.reset();
    for (std::size_t i = 0; i < op.size(); i++) {
        double dummy = 0.0;
        c.moveForward(&op[i], dummy);
    }
    return c.sideToMove;
}

#endif /* CHESS_TEST_OPENING_BOOK_H */
