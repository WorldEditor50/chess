#ifndef PPO_SELFPLAY_MT_HPP
#define PPO_SELFPLAY_MT_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "chess.h"
#include "ppomcts_agent.h"
#include "rl/ppo.h"
#include "rl/util.hpp"

/*
 * PpoSelfPlayMT —— PPO+MCTS 的多线程"分身"自对弈训练
 * ============================================================================
 *
 * 结构 (actor-learner):
 *
 *   ┌─ worker 0 ─┐  各自一份**只推理**的网络副本 + 自己的 Chess + 自己的搜索树
 *   ├─ worker 1 ─┤  循环: 自对弈一局 -> 本局样本塞进共享池 -> (有新版)拉权重
 *   ├─   ...     ┤
 *   └─ worker N ─┘
 *          │  样本 (一把互斥量保护的 deque)
 *          ▼
 *   ┌─ learner ──┐  循环: 把共享池搬进 master 的回放池 -> learnFromReplay
 *   └────────────┘  (梯度累积 + MoE 辅助损失) -> 定期把权重发布到 published 快照
 *
 * 为什么这么分
 * ------------
 * 单线程时"搜索"和"学习"是**串行**的 —— 实测各占一次决策-更新回合的一半左右
 * (见 test_ppomcts 的 compute budget 一节)。而搜索天然可并行 (每个 worker 一棵独立
 * 搜索树、一块独立棋盘), 学习只有一份参数、必须串行。把两者拆成两条流水线之后,
 * 学习就与搜索**重叠**了, 墙钟时间趋近于 max(生成/W, 学习)。
 *
 * 所以加速比的上限由"串行那一半"决定: 学习越便宜, 能拿到的并行收益越大。
 * 这也是为什么先做 P3(梯度累积, 每样本 3~5 倍便宜)与 P4(回放池)再来做这一步。
 *
 * 线程安全 (本文件最需要小心的部分)
 * --------------------------------
 *   * **每条线程一份随机引擎**: worker 入口调 Random::seedCurrentThread(i),
 *     learner 调 seedCurrentThread(0)。既消除数据竞争, 又让整轮跑**可复现**
 *     (i 由调用方给, 不依赖线程创建/调度顺序)。设计背景见 rl/util.hpp 里 Random
 *     的注释 —— 那里还记着"thread_local 的初始化式必须是纯函数"这条 MSVC 硬约束,
 *     踩过一次死锁。
 *   * `Steps` 对象池本来就是 thread_local ✓ (stone.h), 走法生成不需要额外同步。
 *   * **RL::PPO 本身不持锁** (刻意保持单线程语义): master 只被 learner 碰, 每个
 *     worker 的网络只被它自己碰。两条流水线之间靠 `published` 快照 + 一把互斥量
 *     交接 —— 学习期间 worker 读的是上一版快照, 谁都不用等谁。
 *   * worker 的网络是 withGrad=false 的 (只做搜索, 不需要 g/v/m 梯度缓冲, 内存与
 *     构造时间约 1/4), 并且 replayBatchSize=0 (只收集样本, 绝不触发学习)。
 */
class PpoSelfPlayMT
{
public:
    struct Config {
        int    workers       = 4;      /* worker 线程数 (learner 另算) */
        int    simulations   = 80;     /* 每次决策的模拟次数 */
        int    maxMoves      = 120;
        float  tempRoot      = 1.0f;
        float  tempFinal     = 0.1f;
        int    learnBatch    = 64;
        int    learnEpochs   = 2;
        float  lr            = 0.001f;
        /*
           每几轮学习把权重发布一次。太频繁 -> worker 频繁停下来重新同步;
           太稀疏 -> worker 用很旧的策略生成数据。默认每轮都发。
        */
        int    publishEveryLearnRounds = 1;
        unsigned seed        = 20240914;
        int    hiddenDim     = 64;
        int    expertHidden  = 64;
        float  moeAuxCoef    = 0.1f;
        /* 左右镜像数据增广 (P6, 见 ppomcts_agent.h)。worker 与 master 用同一个设置。 */
        bool   mirrorAugment = true;
    };

    struct Stats {
        long long games       = 0;    /* 完成的自我对局 */
        long long samples     = 0;    /* 产出的训练样本数 */
        long long learnRounds = 0;    /* learner 做的优化器更新轮数 */
        long long weightSyncs = 0;    /* worker 拉取权重的次数 */
        double    seconds     = 0.0;
        double gamesPerSec()   const { return seconds > 0.0 ? (double)games / seconds : 0.0; }
        double samplesPerSec() const { return seconds > 0.0 ? (double)samples / seconds : 0.0; }
    };

    PpoSelfPlayMT(Chess &env, const Config &cfg)
        : m_cfg(cfg),
          m_masterAgent(env, cfg.hiddenDim, 0.99f, cfg.lr, 1.414f,
                        cfg.expertHidden, cfg.moeAuxCoef, /*withGrad=*/true),
          m_published(PPOMCTSAgent::STATE_DIM, cfg.hiddenDim, PPOMCTSAgent::ACTION_DIM,
                      cfg.expertHidden, cfg.moeAuxCoef, /*withGrad=*/false)
    {
        /* master 的学习参数由 trainer 统一控制 */
        m_masterAgent.replayBatchSize = cfg.learnBatch;
        m_masterAgent.replayEpochs    = cfg.learnEpochs;
        m_masterAgent.mirrorAugment   = cfg.mirrorAugment;

        /* 主线程 (learner) 也要有确定的随机流 */
        RL::Random::seedCurrentThread(0u);
        m_masterAgent.ppo.clearReplay();
        /* 先把初始权重发布出去, worker 一开场就用同一份权重 */
        publish();
    }

    ~PpoSelfPlayMT()
    {
        stopAndJoin();
    }

    PpoSelfPlayMT(const PpoSelfPlayMT &) = delete;
    PpoSelfPlayMT &operator=(const PpoSelfPlayMT &) = delete;

    /* 训练后的主网络 (保存权重 / 交给 GUI 继续对弈) */
    PPOMCTSAgent &master() { return m_masterAgent; }

    /*
       跑 totalGames 局自对弈 (由 cfg.workers 条 worker 分摊), learner 全程并行更新。
       返回整轮的统计。

       注意: worker 是在**每局开始前**检查配额, 所以实际局数可能超出 totalGames
       最多 workers-1 局 (最后一批 worker 同时看到"还差一点")。统计按实际局数报。
    */
    Stats run(long long totalGames)
    {
        if (m_running) {
            return Stats{};   /* 已经在跑, 不做重入 */
        }
        m_running = true;
        m_targetGames = totalGames;
        m_gamesDone.store(0);
        m_samplesProduced.store(0);
        m_learnRounds.store(0);
        m_weightSyncs.store(0);
        m_stop.store(false);

        const int workers = (m_cfg.workers > 0) ? m_cfg.workers : 1;
        m_workers.clear();
        m_workers.reserve((std::size_t)workers);
        for (int i = 0; i < workers; i++) {
            std::unique_ptr<Worker> w(new Worker(m_cfg));
            m_workers.push_back(std::move(w));
        }

        const auto t0 = std::chrono::steady_clock::now();

        std::thread learner(&PpoSelfPlayMT::learnerLoop, this);
        std::vector<std::thread> pool;
        pool.reserve((std::size_t)workers);
        for (int i = 0; i < workers; i++) {
            pool.emplace_back(&PpoSelfPlayMT::workerLoop, this, i);
        }
        for (std::size_t i = 0; i < pool.size(); i++) {
            pool[i].join();
        }

        /* worker 都收工了: 让 learner 把共享池里剩下的样本学完再退出 */
        m_stop.store(true);
        learner.join();

        const auto t1 = std::chrono::steady_clock::now();

        Stats s;
        s.games       = m_gamesDone.load();
        s.samples     = m_samplesProduced.load();
        s.learnRounds = m_learnRounds.load();
        s.weightSyncs = m_weightSyncs.load();
        s.seconds = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

        m_workers.clear();
        m_running = false;
        return s;
    }

private:
    struct Worker {
        Chess env;
        PPOMCTSAgent agent;
        unsigned generation = 0;   /* 本副本对应的 published 版本号 */

        explicit Worker(const Config &cfg)
            : env(),
              agent(env, cfg.hiddenDim, 0.99f, cfg.lr, 1.414f,
                    cfg.expertHidden, cfg.moeAuxCoef, /*withGrad=*/false)
        {
            /*
               worker 只收集样本、绝不学习: commitEpisode 里
               `if (replayBatchSize > 0)` 是唯一的开关, 设 0 就退化成"只入池"。
               这也是为什么 worker 的网络可以不带梯度。
            */
            agent.replayBatchSize = 0;
            agent.mirrorAugment   = cfg.mirrorAugment;
            agent.ppo.clearReplay();
        }
    };

    void publish()
    {
        std::lock_guard<std::mutex> g(m_weightMutex);
        m_masterAgent.ppo.actorP.copyTo(m_published.actorP);
        m_masterAgent.ppo.critic.copyTo(m_published.critic);
        m_publishedGeneration.fetch_add(1, std::memory_order_release);
    }

    void syncWeights(Worker &w, bool force)
    {
        const unsigned gen = m_publishedGeneration.load(std::memory_order_acquire);
        if (!force && gen == w.generation) {
            return;   /* 已经是最新的 */
        }
        {
            std::lock_guard<std::mutex> g(m_weightMutex);
            m_published.actorP.copyTo(w.agent.ppo.actorP);
            m_published.critic.copyTo(w.agent.ppo.critic);
            w.generation = m_publishedGeneration.load(std::memory_order_relaxed);
        }
        m_weightSyncs.fetch_add(1);
    }

    void workerLoop(int index)
    {
        /*
           每条线程一份随机引擎。index+1 避开 learner 的 0 —— 两者不共用随机流。
           用调用方给的编号而不是"按创建顺序自动编号", 所以并行跑也完全可复现。
        */
        RL::Random::seedCurrentThread((unsigned)index + 1u);

        Worker &w = *m_workers[(std::size_t)index];
        syncWeights(w, /*force=*/true);

        while (!m_stop.load(std::memory_order_relaxed)) {
            if (m_gamesDone.load(std::memory_order_relaxed) >= m_targetGames) {
                break;
            }

            /* 自对弈一局 (replayBatchSize=0 -> 只收集, 不学习) */
            w.agent.trainSelfPlay(1, m_cfg.simulations, m_cfg.maxMoves, false,
                                  m_cfg.tempRoot, m_cfg.tempFinal);

            /* 把本局样本交给共享池 */
            std::deque<RL::PPO::ReplaySample> batch = w.agent.ppo.takeReplay();
            if (!batch.empty()) {
                std::lock_guard<std::mutex> g(m_queueMutex);
                for (std::size_t i = 0; i < batch.size(); i++) {
                    m_queue.push_back(std::move(batch[i]));
                }
                m_samplesProduced.fetch_add((long long)batch.size());
            }
            m_gamesDone.fetch_add(1);

            /* 有新版本就同步 (没有就什么都不做, 不碰锁) */
            syncWeights(w, /*force=*/false);
        }
    }

    /* 攒够一批就学一次, 顺手把权重发布出去 (返回本轮学习序号) */
    void learnOnceAndMaybePublish()
    {
        m_masterAgent.ppo.learnFromReplay((std::size_t)m_cfg.learnBatch,
                                          m_cfg.learnEpochs, m_cfg.lr);
        const long long n = m_learnRounds.fetch_add(1) + 1;
        const int every = (m_cfg.publishEveryLearnRounds > 0)
                              ? m_cfg.publishEveryLearnRounds : 1;
        if ((n % every) == 0) {
            publish();
        }
    }

    void learnerLoop()
    {
        RL::Random::seedCurrentThread(0u);

        while (true) {
            /*
               一把锁只做一次 swap: worker 在锁外 push, learner 在锁外处理。
               样本搬运本身是 O(1) (deque::swap), 不会让 worker 排队。
            */
            std::deque<RL::PPO::ReplaySample> batch;
            {
                std::lock_guard<std::mutex> g(m_queueMutex);
                batch.swap(m_queue);
            }
            for (std::size_t i = 0; i < batch.size(); i++) {
                RL::PPO::ReplaySample &s = batch[i];
                m_masterAgent.ppo.addReplay(s.state, s.actionIdx, s.actionProb, s.valueTarget);
            }

            /*
               收尾必须**先判停**, 不能写在 "没攒够一批" 那个 else 分支里 ——
               池子一旦涨到 learnBatch 以上就永远 "省得学", 那个分支再也进不去,
               learner 会在 worker 全部收工之后继续拿着同一批旧样本空转下去
               (这是个写错过一次的退出条件)。
               worker 已经 join 过了, 此刻队列里是最后一批样本; 把它学完就退出。
            */
            if (m_stop.load(std::memory_order_acquire)) {
                if (m_masterAgent.ppo.replaySize() >= (std::size_t)m_cfg.learnBatch) {
                    learnOnceAndMaybePublish();
                }
                break;
            }

            if (m_masterAgent.ppo.replaySize() >= (std::size_t)m_cfg.learnBatch) {
                learnOnceAndMaybePublish();
            } else {
                /* 没攒够一批: 让出 CPU, 别空转 */
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void stopAndJoin()
    {
        m_stop.store(true);
    }

private:
    Config m_cfg;

    PPOMCTSAgent m_masterAgent;   /* learner 独占 (带梯度) */
    RL::PPO      m_published;     /* 权重快照 (只推理), worker 从这里拉 */

    std::vector<std::unique_ptr<Worker> > m_workers;

    /* 共享样本池 */
    std::mutex m_queueMutex;
    std::deque<RL::PPO::ReplaySample> m_queue;

    /* 权重交接: 一把锁保护 m_published 的整份拷贝 */
    std::mutex m_weightMutex;
    std::atomic<unsigned> m_publishedGeneration{ 0u };

    std::atomic<bool>      m_stop{ false };
    std::atomic<long long> m_targetGames{ 0 };
    std::atomic<long long> m_gamesDone{ 0 };
    std::atomic<long long> m_samplesProduced{ 0 };
    std::atomic<long long> m_learnRounds{ 0 };
    std::atomic<long long> m_weightSyncs{ 0 };
    bool m_running = false;
};

#endif // PPO_SELFPLAY_MT_HPP
