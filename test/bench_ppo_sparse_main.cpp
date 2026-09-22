/*
 * bench_ppo_sparse_main.cpp - PPO "只对合法动作动态打分" 的代价拆解与回归钉
 * ============================================================================
 *
 * 背景 (2026-09 用户要求): SAC 回退到 1263/128, PPO 保持 8100 双射, 但输出要走
 * **只对合法动作动态打分**的稀疏口径。PPO 的策略头其实已经有这条路径
 * (`PPOMCTSAgent::sparsePolicyHead = true` -> `RL::PPO::actionMasked` ->
 * `forwardTrunk` + `sparseLogits` + 合法集上 softmax), 这个程序用来:
 *
 *   1. **证明稀疏路径真的在生效** (不是悄悄回退到全量): 同一局面下
 *      稀疏与全量的合法列逐元素一致 + 头权重的读取字节数相差两个数量级;
 *   2. **把代价拆开**: 策略头 vs 骨干(trunk) vs 值头各占多少 —— 这决定"下一步该优化
 *      哪里" (R1 之后头已经不是什么了, 剩下的是骨干与值头);
 *   3. **接口层面对齐 SAC**: SAC 侧有三个入口 (policy/Q/softValue 的稀疏版),
 *      PPO 侧目前只有 `actionMasked` 一个 —— 这里量出"若值头也读合法集会不会更快",
 *      为"两边接口对齐"提供数据。
 *
 * 用法:
 *   bench_ppo_sparse.exe [--positions=25] [--reps=200] [--quiet]
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/ppo.h"
#include "rl/util.hpp"

namespace {

struct Timer {
    std::chrono::steady_clock::time_point t0;
    Timer() : t0(std::chrono::steady_clock::now()) {}
    void reset() { t0 = std::chrono::steady_clock::now(); }
    double us() const {
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - t0).count() / 1000.0;
    }
};

} // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int positions = 25;
    int reps = 200;
    bool quiet = false;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto val = [&](const char *k) -> const char* {
            const std::size_t n = std::strlen(k);
            if (std::strncmp(a, k, n) == 0 && a[n] == '=') { return a + n + 1; }
            return nullptr;
        };
        if (const char *v = val("--positions")) { positions = std::atoi(v); }
        else if (const char *v = val("--reps"))  { reps = std::atoi(v); }
        else if (std::strcmp(a, "--quiet") == 0) { quiet = true; }
    }
    if (reps < 1) { reps = 1; }
    RL::Random::setSeed(20240901);

    Chess board;
    board.reset();
    PPOMCTSAgent agent(board, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true,
                       RL::PPO::Backbone::MlpExperts);
    agent.replayBatchSize = 0;   /* 只推理 */

    std::printf("=== PPO 稀疏输出口径核对 (state=%d action=%d, 骨干=%s) ===\n",
                PPOMCTSAgent::STATE_DIM, PPOMCTSAgent::ACTION_DIM,
                RL::PPO::backboneName(RL::PPO::Backbone::MlpExperts));
    std::printf("sparsePolicyHead = %d (生产默认)\n\n", (int)agent.sparsePolicyHead);

    /* ---- 1) 等价性 + 头权重流量: 稀疏 vs 全量 (逐局面) ---- */
    double maxDev = 0.0;
    int checked = 0;
    double sparseBytes = 0.0, denseBytes = 0.0;
    {
        RL::iFcLayer *head = dynamic_cast<RL::iFcLayer *>(
            agent.ppo.actorP[agent.ppo.actorP.size() - 1]);
        if (head != nullptr) {
            denseBytes = (double)head->w.size() * (double)sizeof(float);
        }
        Chess c;
        c.reset();
        for (int p = 0; p < positions; p++) {
            /* 每个局面先随机走几手 (双方都不参与), 让样本覆盖开局到中局 */
            int turn = Stone::COLOR_RED;
            for (int k = 0; k < (p % 12); k++) {
                if (c.getResult(turn) != Chess::RESULT_ONGOING) { break; }
                std::vector<Step*> legal;
                c.sample(turn, legal);
                if (legal.empty()) { Steps::instance().put(legal); break; }
                std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
                const Step s = *legal[(std::size_t)pick(RL::Random::engine)];
                Steps::instance().put(legal);
                double d = 0.0;
                c.moveForward(&s, d);
                turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            }
            RL::Tensor st(PPOMCTSAgent::STATE_DIM, 1);
            RL::Tensor mk(PPOMCTSAgent::ACTION_DIM, 1);
            RL::Tensor full(PPOMCTSAgent::ACTION_DIM, 1);
            agent.encodeStateFor(turn, st);
            std::vector<Step*> legal;
            std::vector<int> li;
            agent.getLegalActions(turn, legal, li, mk);
            Steps::instance().put(legal);

            std::vector<float> sp;
            if (!agent.ppo.actionMasked(st, li, sp) || sp.size() != li.size()) {
                std::printf("  **稀疏路径没走通 (回退了)**\n");
                break;
            }
            full = agent.ppo.action(st);   /* 全量 (对照) */
            /* 全量 -> 合法集上归一, 与稀疏口径应逐元素一致 */
            double sum = 0.0;
            for (std::size_t k = 0; k < li.size(); k++) { sum += (double)full[(std::size_t)li[k]]; }
            for (std::size_t k = 0; k < li.size(); k++) {
                const double ref = (sum > 1e-12) ? (double)full[(std::size_t)li[k]] / sum : 0.0;
                maxDev = std::max(maxDev, std::fabs(ref - (double)sp[k]));
            }
            if (head != nullptr && sparseBytes == 0.0) {
                sparseBytes = (double)li.size() * (double)head->inputDim * (double)sizeof(float);
            }
            checked++;
        }
    }
    std::printf("[1] 稀疏 vs 全量 (合法集上归一): %d 个局面, 最大逐元素偏差 %.3e\n",
                checked, maxDev);
    std::printf("    策略头权重读取: 全量 %.2f MB/次 -> 稀疏 %.1f KB/次 (%.0fx 更少)\n",
                denseBytes / 1048576.0, sparseBytes / 1024.0,
                sparseBytes > 0.0 ? denseBytes / sparseBytes : 0.0);

    /* ---- 2) 代价拆解: 头 vs 骨干 vs 值头 ---- */
    board.reset();
    int turn0 = Stone::COLOR_RED;
    for (int k = 0; k < 4; k++) {
        std::vector<Step*> legal;
        board.sample(turn0, legal);
        if (legal.empty()) { Steps::instance().put(legal); break; }
        std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
        const Step s = *legal[(std::size_t)pick(RL::Random::engine)];
        Steps::instance().put(legal);
        double d = 0.0;
        board.moveForward(&s, d);
        turn0 = (turn0 == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
    }
    RL::Tensor st(PPOMCTSAgent::STATE_DIM, 1);
    RL::Tensor mk(PPOMCTSAgent::ACTION_DIM, 1);
    agent.encodeStateFor(board.sideToMove, st);
    std::vector<Step*> legal;
    std::vector<int> li;
    agent.getLegalActions(board.sideToMove, legal, li, mk);
    Steps::instance().put(legal);

    std::vector<float> probs;
    Timer t;
    for (int i = 0; i < reps; i++) { agent.ppo.actorP.forwardTrunk(st); }
    const double trunkUs = t.us() / reps;

    t.reset();
    for (int i = 0; i < reps; i++) { agent.ppo.actionMasked(st, li, probs); }
    const double sparseUs = t.us() / reps;

    t.reset();
    for (int i = 0; i < reps; i++) { agent.ppo.action(st); }
    const double denseUs = t.us() / reps;

    t.reset();
    for (int i = 0; i < reps; i++) { agent.ppo.value(st); }
    const double valueUs = t.us() / reps;

    std::printf("\n[2] 单次调用耗时 (reps=%d, 合法=%zu 个):\n", reps, li.size());
    std::printf("    forwardTrunk (骨干)          : %8.1f us\n", trunkUs);
    std::printf("    actionMasked (骨干+稀疏头)   : %8.1f us   <- 生产路径\n", sparseUs);
    std::printf("    action       (骨干+全量头)   : %8.1f us\n", denseUs);
    std::printf("    value        (骨干+值头)     : %8.1f us   <- 每次叶子估值都要\n", valueUs);
    std::printf("    稀疏头净开销 = %.1f us (%.1f%% of actionMasked); 全量头净开销 = %.1f us\n",
                sparseUs - trunkUs, 100.0 * (sparseUs - trunkUs) / std::max(1.0, sparseUs),
                denseUs - trunkUs);

    /* ---- 3) 一次真实决策里各部分的调用次数 ---- */
    t.reset();
    const Step mv = agent.selectMove(board.sideToMove, 400, 0.0f);
    const double moveUs = t.us();
    std::printf("\n[3] 一次 selectMove(400 模拟) = %.1f ms; 选点 (%d,%d)->(%d,%d)\n",
                moveUs / 1000.0, mv.pos.x, mv.pos.y, mv.nextPos.x, mv.nextPos.y);
    if (valueUs > 0.0) {
        std::printf("    按 value 单价换算: 400 次叶子 ≈ %.1f ms (占 %.0f%%)\n",
                    valueUs * 400.0 / 1000.0, 100.0 * valueUs * 400.0 / moveUs);
    }

    /* ---- 4) 端到端: 稀疏输出 vs 全量输出 (整局决策) ---- */
    {
        auto timeMatches = [&](bool sparse) {
            agent.sparsePolicyHead = sparse;
            board.reset();
            int turn = Stone::COLOR_RED;
            for (int k = 0; k < 4; k++) {   /* 同一个随机开局, 保证局面一致 */
                std::vector<Step*> legal;
                board.sample(turn, legal);
                if (legal.empty()) { Steps::instance().put(legal); break; }
                std::uniform_int_distribution<int> pick(0, (int)legal.size() - 1);
                const Step s = *legal[(std::size_t)pick(RL::Random::engine)];
                Steps::instance().put(legal);
                double d = 0.0;
                board.moveForward(&s, d);
                turn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK : Stone::COLOR_RED;
            }
            const double t0 = t.us();
            int calls = 0;
            for (int k = 0; k < 10; k++) {
                if (board.getResult(board.sideToMove) != Chess::RESULT_ONGOING) { break; }
                const Step s = agent.selectMove(board.sideToMove, 200, 0.0f);
                if (!s.valid) { break; }
                std::vector<Step*> lg;
                board.sample(board.sideToMove, lg);
                const bool okMove = !lg.empty() && board.isLegalMove(board.sideToMove, &s);
                Steps::instance().put(lg);
                if (!okMove) { break; }
                double d = 0.0;
                board.moveForward(&s, d);
                calls++;
            }
            const double us = t.us() - t0;
            agent.sparsePolicyHead = true;   /* 复原 */
            return calls > 0 ? us / calls : 0.0;
        };
        const double sparseMoveUs = timeMatches(true);
        const double denseMoveUs = timeMatches(false);
        std::printf("\n[4] 端到端 (200 模拟/步, 各 10 步): 稀疏 %.1f ms/步, 全量 %.1f ms/步 -> %.2fx\n",
                    sparseMoveUs / 1000.0, denseMoveUs / 1000.0,
                    sparseMoveUs > 0.0 ? denseMoveUs / sparseMoveUs : 0.0);
        if (sparseMoveUs > 0.0 && denseMoveUs > 0.0) {
            std::printf("    每次决策省下 %.1f ms\n", (denseMoveUs - sparseMoveUs) / 1000.0);
        }
    }

    const bool ok = (maxDev < 1e-5) && (checked > 0) && (sparseBytes > 0.0)
                    && (sparseBytes * 10.0 < denseBytes) && (mv.valid);
    std::printf("\nVERDICT: %s | 稀疏/全量偏差=%.1e 头流量比=%.0fx 合法动作数=%zu\n",
                ok ? "PASS" : "FAIL", maxDev,
                sparseBytes > 0.0 ? denseBytes / sparseBytes : 0.0, li.size());
    (void)quiet;
    return ok ? 0 : 1;
}
