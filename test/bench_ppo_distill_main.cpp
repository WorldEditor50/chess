/*
 * bench_ppo_distill_main.cpp - 价值头蒸馏 (Step 2, 不进 ctest)
 * ============================================================================
 *
 * 做什么: 用 **ABAgent 的搜索分**当监督目标, 单独预训练 PPO 的 critic (不动 actor)。
 *
 * 为什么是"AB 的搜索分"而不是手工局面评估:
 *   手工评估 = 深度 0 的判断; AB depth-3/4 的根分值 = **搜了三四层之后的结论**,
 *   里面已经含了战术 (吃子序列、被将死、兑子次序)。把它蒸馏进 critic, 等于把
 *   "AB 的搜索成果"搬进网络 —— 而 MCTS 正是拿 critic 当叶子评估用的, 所以这一步
 *   直接改善搜索质量, 不碰奖励、不依赖外部引擎、也没有许可证问题 (全仓无 FEN 接口,
 *   接 Pikafish/Stockfish 还得先写局面交换桥)。
 *
 * 口径 (最容易出静默错的地方): ABAgent 的根分值一律是**黑方视角** (黑 = MAX、红 = MIN),
 *   而 critic 是**规范视角** (轮到谁走就是谁的价值), 所以
 *       target = (sideToMove == BLACK ? +1 : -1) * tanh(score / SCALE)
 *   这里用与势能同一个 SCALE (stone.h 的 REWARD_POTENTIAL_SCALE = 2.0), 让 critic 的
 *   输出尺度与 Φ 一致 (Φ 就是"局面价值"的归一化形式)。
 *   为了不让符号搞错而不自知, 本程序会算 AB 分值与 evaluatePositional() 的**相关系数**
 *   (两者都是黑方视角) —— 若是负的, 说明符号错了; 这条比肉眼看公式可靠。
 *
 * 训练只动 critic: forward -> MSE -> backward -> RMSProp (iFcLayer::RMSProp 会在更新后
 *   清梯度, 所以是"累积一批后更新一次"的标准用法)。actor 一个字都不改 —— 于是可以用
 *   bench_policy_agreement 量"**只把价值头换好**"对选点的影响 (那是非循环的测量:
 *   蒸馏的是价值, 不是着法)。
 *
 * 用法:
 *   bench_ppo_distill.exe [--positions=1500] [--depth=3] [--opening=8]
 *                         [--epochs=6] [--batch=64] [--lr=0.002]
 *                         [--seed=N] [--save=PREFIX]
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

#include "abagent.h"
#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/loss.h"
#include "rl/util.hpp"

namespace {

int    g_positions = 1500;
int    g_depth = 3;
int    g_opening = 8;
int    g_epochs = 6;
int    g_batch = 64;
float  g_lr = 0.002f;
unsigned g_seed = 20240901u;
std::string g_savePrefix;
/*
 *  Step 3: **行为克隆预训练 actor**（--actor=1）。
 *
 *  为什么需要它: Step 2 实测出"价值头蒸馏成功（留出 MSE 降 4.2 倍）但选点几乎没动"，
 *  因为**先验（actor）还是随机的** —— 中局约 40 个合法着法、80~200 次模拟，搜索树主要
 *  靠（随机）先验铺开，好价值头无从发挥。佐证：未训练网络把模拟从 80 提到 200，探针数字
 *  一动不动。所以这一步先把 actor 从"随机噪声"拉到"像 AB 一样选点"，让搜索有东西可聚焦。
 *
 *  边界（必须记住）：这是**模仿**，上限就是被模仿者的水平（AB 深度 g_depth）；它是课程，
 *  不是终点。而且用它之后，"与 AB 的一致率"再上升就不再是独立的棋力证据了 —— 那时参照
 *  要换成更深的 AB。
 */
bool   g_actor = false;
int    g_actorEpochs = 6;
float  g_actorLr = 0.002f;

double nowSec()
{
    using clock = std::chrono::steady_clock;
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch()).count() / 1e9;
}

struct Sample {
    RL::Tensor state;
    float target;   /* critic 目标: 规范视角 (轮到走棋的一方), 已 tanh 归一 */
    int actionIdx;  /* actor 目标: AB 选的着法在该局面规范视角下的动作下标 */
};

/* 走 `openings` 手随机棋得到一个局面 (与权重无关, 可复现) */
void makePosition(int index, Chess &out)
{
    out.reset();
    out.sideToMove = Stone::COLOR_RED;
    std::mt19937 rng((unsigned)(g_seed + (unsigned)index * 7919u));
    for (int i = 0; i < g_opening; i++) {
        std::vector<Step *> legal;
        out.sample(out.sideToMove, legal);
        if (legal.empty()) break;
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        Step chosen(*legal[pick(rng)]);
        Steps::instance().put(legal);
        double dummy = 0.0;
        out.moveForward(&chosen, dummy);
    }
}

double mseOn(RL::PPO &ppo, const std::vector<Sample> &set)
{
    if (set.empty()) return 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < set.size(); i++) {
        const double v = ppo.critic.forward(set[i].state)[0];
        const double d = v - (double)set[i].target;
        sum += d * d;
    }
    return sum / (double)set.size();
}

/*
 *  [Step 3] actor 的三个信号层指标:
 *    * priorTop1: 策略头**单独**（不看搜索）的 argmax 是否等于 AB 选的着法 —— 这就是
 *      行为克隆的训练目标本身, 用来确认"克隆成功了";
 *    * probOnAb : 策略头在 AB 那个着法上的平均概率 (从 1/8100 涨到多少);
 *    * ce       : 交叉熵 (批平均), 与训练时同一口径。
 */
void actorMetrics(RL::PPO &ppo, const std::vector<Sample> &set,
                  double &top1Pct, double &probOnAb, double &ce)
{
    top1Pct = probOnAb = ce = 0.0;
    if (set.empty()) return;
    int hit = 0;
    for (std::size_t i = 0; i < set.size(); i++) {
        RL::Tensor &p = ppo.actorP.forward(set[i].state);
        const int a = set[i].actionIdx;
        if (a >= 0 && (std::size_t)a < p.size()) {
            if (p.argmax() == a) hit++;
            probOnAb += (double)p[(std::size_t)a];
            ce -= std::log((double)p[(std::size_t)a] + 1e-8);
        }
    }
    top1Pct = 100.0 * (double)hit / (double)set.size();
    probOnAb /= (double)set.size();
    ce /= (double)set.size();
}

} // namespace

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const std::size_t eq = a.find('=');
        const std::string k = (eq == std::string::npos) ? a : a.substr(0, eq);
        const std::string v = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);
        if (k == "--positions")     { g_positions = std::atoi(v.c_str()); }
        else if (k == "--depth")    { g_depth = std::atoi(v.c_str()); }
        else if (k == "--opening")  { g_opening = std::atoi(v.c_str()); }
        else if (k == "--epochs")   { g_epochs = std::atoi(v.c_str()); }
        else if (k == "--batch")    { g_batch = std::atoi(v.c_str()); }
        else if (k == "--lr")       { g_lr = (float)atof(v.c_str()); }
        else if (k == "--seed")     { g_seed = (unsigned)strtoul(v.c_str(), nullptr, 10); }
        else if (k == "--save")     { g_savePrefix = v; }
        else if (k == "--actor")    { g_actor = (std::atoi(v.c_str()) != 0); }
        else if (k == "--actor-epochs") { g_actorEpochs = std::atoi(v.c_str()); }
        else if (k == "--actor-lr") { g_actorLr = (float)atof(v.c_str()); }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }

    std::printf("=== 价值头蒸馏 (AB 搜索分 -> PPO critic) ===\n");
    std::printf("样本     : %d 个局面 (随机开局 %d 手, seed=%u)\n",
                g_positions, g_opening, g_seed);
    std::printf("监督目标 : ABAgent 深度 %d 的根分值 (黑方视角) -> 规范视角 tanh(s/%.1f)\n",
                g_depth, (double)REWARD_POTENTIAL_SCALE);
    std::printf("训练     : batch=%d, epochs=%d, lr=%.4f (只更新 critic, actor 不动)\n",
                g_batch, g_epochs, (double)g_lr);
    std::fflush(stdout);

    RL::Random::setSeed(g_seed);

    Chess env;
    env.reset();
    PPOMCTSAgent ppo(env, 64, 0.99f, 0.001f, 1.414f, 64, 0.1f, true);
    ppo.replayBatchSize = 0;
    /* 蒸馏不学策略: actor 保持构造时的状态, 这样"只换价值头"的效果才可归因 */

    Chess envAb;
    envAb.reset();
    ABAgent ab(envAb, g_depth);

    /* ---- 1. 造局面 + 打标签 ---- */
    std::vector<Sample> all;
    all.reserve((std::size_t)g_positions);
    double corrSum = 0.0, corrAbsA = 0.0, corrAbsB = 0.0, corrCross = 0.0;
    int skipped = 0;
    const double t0 = nowSec();
    for (int i = 0; i < g_positions; i++) {
        Chess pos;
        makePosition(i, pos);
        const int color = pos.sideToMove;

        envAb = pos;
        /* 一次搜索同时拿到"根分值"(critic 目标) 与"选点"(actor 目标) —— 不要搜两遍 */
        const Step abMove = ab.getBestMove(color, g_depth);
        if (!ab.getScoreValid()) { skipped++; continue; }

        const double score = ab.getLastScore();                 /* 黑方视角 */
        const double raw = std::tanh(score / (double)REWARD_POTENTIAL_SCALE);
        const double target = (color == Stone::COLOR_BLACK) ? raw : -raw;  /* 规范视角 */

        Sample s;
        s.state = RL::Tensor(PPOMCTSAgent::STATE_DIM, 1);
        s.state.zero();
        /* 用待蒸馏的网络自己的编码器 (与训练/推理一致) */
        ppo.chess = pos;
        ppo.encodeStateFor(color, s.state);
        s.target = (float)target;
        /*
           Step 3: actor 的监督目标 = AB 选的那一步, 换算成**同一规范视角**下的动作下标
           (stepToActionIdx 必须用与状态编码同一个 color, 否则先验会串帧)。
        */
        s.actionIdx = ppo.stepToActionIdx(abMove, color);
        all.push_back(s);

        /* 顺带算与手工局面评估的相关系数 (两者都是黑方视角), 用来验证符号没搞反 */
        ppo.chess = pos;
        const double hand = ppo.chess.evaluatePositional();
        corrSum += raw * hand;
        corrAbsA += raw * raw;
        corrAbsB += hand * hand;
    }
    const double labelSec = nowSec() - t0;
    if (all.empty()) {
        std::printf("**没有可用样本**\n");
        return 1;
    }
    const double corr = (corrAbsA > 0.0 && corrAbsB > 0.0)
                            ? corrSum / std::sqrt(corrAbsA * corrAbsB) : 0.0;
    std::printf("\n[1] 标签完成: %llu 个样本 (跳过 %d 个无合法走法的局面), %.1f s"
                " (%.1f ms/局面)\n",
                (unsigned long long)all.size(), skipped, labelSec,
                1000.0 * labelSec / (double)g_positions);
    std::printf("    符号校验: AB 分值与 evaluatePositional() 的相关系数 = %+.3f"
                "  (>0 说明两者视角一致)\n", corr);
    if (corr <= 0.0) {
        std::printf("    **相关系数非正 —— 标签的视角很可能搞反了, 后面的训练结果不可信**\n");
    }

    /* ---- 2. 划分训练/留出集 (按 index 奇偶, 避免顺序相关) ---- */
    std::vector<Sample> trainSet, holdout;
    for (std::size_t i = 0; i < all.size(); i++) {
        if (i % 5 == 0) holdout.push_back(all[i]);
        else            trainSet.push_back(all[i]);
    }
    std::printf("\n[2] 划分: 训练 %llu / 留出 %llu\n",
                (unsigned long long)trainSet.size(), (unsigned long long)holdout.size());
    std::printf("    蒸馏前 MSE: 训练 %.6f, 留出 %.6f\n",
                mseOn(ppo.ppo, trainSet), mseOn(ppo.ppo, holdout));

    /* ---- 3. 只训练 critic ---- */
    const double t1 = nowSec();
    std::mt19937 rng(g_seed ^ 0x5bd1e995u);
    RL::Tensor targetTensor(1, 1);
    for (int ep = 0; ep < g_epochs; ep++) {
        std::shuffle(trainSet.begin(), trainSet.end(), rng);
        int steps = 0;
        for (std::size_t i = 0; i + (std::size_t)g_batch <= trainSet.size();
             i += (std::size_t)g_batch) {
            for (int b = 0; b < g_batch; b++) {
                const Sample &s = trainSet[i + (std::size_t)b];
                RL::Tensor &v = ppo.ppo.critic.forward(s.state);
                targetTensor[0] = s.target;
                RL::Tensor mseLoss = RL::Loss::MSE::df(v, targetTensor);
                ppo.ppo.critic.backward(s.state, mseLoss);
            }
            ppo.ppo.critic.RMSProp(g_lr);
            steps++;
        }
        std::printf("    epoch %d/%d: 该轮 %d 次更新, 训练 MSE %.6f, 留出 MSE %.6f\n",
                    ep + 1, g_epochs, steps, mseOn(ppo.ppo, trainSet),
                    mseOn(ppo.ppo, holdout));
        std::fflush(stdout);
    }
    const double trainSec = nowSec() - t1;
    std::printf("    蒸馏后 MSE: 训练 %.6f, 留出 %.6f (耗时 %.1f s)\n",
                mseOn(ppo.ppo, trainSet), mseOn(ppo.ppo, holdout), trainSec);

    /* ---- 4. [Step 3] 行为克隆: 只训练 actor ---- */
    if (g_actor) {
        double t1p, p1, c1, t2p, p2, c2;
        actorMetrics(ppo.ppo, trainSet, t1p, p1, c1);
        actorMetrics(ppo.ppo, holdout, t2p, p2, c2);
        std::printf("\n[4] 行为克隆预训练 actor (目标 = AB 深度 %d 的选点)\n", g_depth);
        std::printf("    克隆前: prior top-1 训练 %.1f%% / 留出 %.1f%%;"
                    " P(AB 着法) 训练 %.5f / 留出 %.5f; CE 训练 %.3f / 留出 %.3f\n",
                    t1p, t2p, p1, p2, c1, c2);
        std::fflush(stdout);

        const double t2 = nowSec();
        RL::Tensor oneHot(PPOMCTSAgent::ACTION_DIM, 1);
        for (int ep = 0; ep < g_actorEpochs; ep++) {
            std::shuffle(trainSet.begin(), trainSet.end(), rng);
            int steps = 0;
            for (std::size_t i = 0; i + (std::size_t)g_batch <= trainSet.size();
                 i += (std::size_t)g_batch) {
                for (int b = 0; b < g_batch; b++) {
                    const Sample &s = trainSet[i + (std::size_t)b];
                    oneHot.zero();
                    if (s.actionIdx >= 0 && (std::size_t)s.actionIdx < oneHot.size()) {
                        oneHot[(std::size_t)s.actionIdx] = 1.0f;
                    }
                    RL::Tensor &p = ppo.ppo.actorP.forward(s.state);
                    RL::Tensor ceLoss = RL::Loss::CrossEntropy::df(p, oneHot);
                    ppo.ppo.actorP.backward(s.state, ceLoss);
                }
                ppo.ppo.actorP.RMSProp(g_actorLr);
                steps++;
            }
            actorMetrics(ppo.ppo, trainSet, t1p, p1, c1);
            actorMetrics(ppo.ppo, holdout, t2p, p2, c2);
            std::printf("    epoch %d/%d: %d 次更新, prior top-1 训练 %.1f%% / 留出 %.1f%%,"
                        " P(AB) 训练 %.4f / 留出 %.4f, CE 留出 %.3f\n",
                        ep + 1, g_actorEpochs, steps, t1p, t2p, p1, p2, c2);
            std::fflush(stdout);
        }
        std::printf("    克隆后: prior top-1 训练 %.1f%% / 留出 %.1f%%;"
                    " P(AB 着法) 训练 %.4f / 留出 %.4f; CE 训练 %.3f / 留出 %.3f"
                    " (耗时 %.1f s)\n", t1p, t2p, p1, p2, c1, c2, nowSec() - t2);
        std::printf("    注: prior top-1 是**克隆的训练目标本身**, 它高只说明「克隆成功」;\n"
                    "        这个数上升之后, 「与 AB 的一致率」就不再是独立的棋力证据了"
                    " (参照要换成更深的 AB)。\n");
    }

    /* ---- 5. 保存 ---- */
    if (!g_savePrefix.empty()) {
        const bool ok = ppo.saveModel(g_savePrefix);
        std::printf("\n[5] 权重已保存: %s_actor / %s_critic -> %s\n"
                    "    (用 bench_policy_agreement --load 量效果)\n",
                    g_savePrefix.c_str(), g_savePrefix.c_str(), ok ? "成功" : "**失败**");
        if (!ok) { return 1; }
    }
    return 0;
}
