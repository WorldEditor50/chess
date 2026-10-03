#ifndef RL_WEIGHTIO_HPP
#define RL_WEIGHTIO_HPP

/*
 * ================================================================
 *  WeightIO —— "对弈期间权重文件不落盘"的唯一通道 (2026-10 用户口径)
 * ================================================================
 *
 * 用户口径 (原话): **"对弈期间不更新保存模型权重"**。
 *
 * 收窄成可实现的判据 (与用户确认过两条):
 *   1. **范围**: 人机对弈 与 界面上的"开始对弈"整场, 两种都算;
 *   2. **程度**: **学习照常** (每手在线更新 + 后台训练), 但对弈期间
 *      **一个权重文件都不写** —— 退出时照旧统一保存。
 *
 * ---- 为什么需要这一层 (而不是"对弈期间把后台训练停掉") ----
 *
 * 因为"后台训练"这一套的落地方式**本来就是文件往返**:
 *
 *     主 agent --saveModel--> 临时权重文件 --loadModel--> 训练 clone
 *              --saveModel--> 临时权重文件 --loadModel--> 主 agent
 *
 * 这是对弈期间**唯一**会写权重文件的东西 (标准权重 `defaultWeightPath()` 只在退出时
 * 写一次, 见 MainWindow 析构)。所以"学习照常 + 一个权重文件都不写"这两条要同时成立,
 * 唯一做法就是**把这条往返改走内存** —— 也就是本模块。
 *
 * 为什么不做成"对弈期间暂停后台训练": 那当然也能做到"不落盘", 但它把用户明确要的那
 * 一半 (后台训练照常) 砍掉了 —— 对局期间后台线程会整体停摆, 一局几十分钟的训练量
 * 全丢。停摆那件事已经由 isBackgroundTrainingPaused() 在**评估/只对弈**模式上做了,
 * 那是"冻结"语义, 与本条不是一回事。
 *
 * ---- 判据: 只改道**临时**权重路径 (这是关键的安全边界) ----
 *
 * 只有以 `kTransientPrefix` (`weights/_temp`) 开头的路径会被改道。三张表是它的来源,
 * 而且它们**全**以这个前缀开头:
 *   * `tmpWeightsOf(type)`        —— 后台训练那一轮的临时文件 (TMP_WEIGHTS*)
 *   * `frozenSnapshotPathOf(type)`—— 评估模式的开场快照
 *   * `frozenAuditPathOf(type)`   —— 逐字节审计那一份 (默认关)
 * (chessboard.cpp 里有一个**运行期自检**: 每个 agent 的临时前缀都必须命中这个判据,
 *  否则打一条 qWarning —— 前缀规则一旦漂移, "不落盘"就会静默失效, 那正是本工程最怕的.)
 *
 * ⚠ **标准权重路径 (weights/xxx_agent.dat*) 永远走磁盘**, 不受改道影响。理由: 那些是
 *   用户真正要留下的东西, 把它们悄悄重定向到内存 = "点了保存, 进程一退全没了" 的
 *   静默失效。对弈期间本来也没有任何代码写标准路径 (唯一落盘点在退出), 所以这条边界
 *   既安全又不损失任何东西。
 *
 * ---- 内存成本 (说清楚, 别被当成免费) ----
 *
 * 一轮训练里, 该 agent 的临时权重会**在内存里多留一份**, 而且留的是**序列化后的文本**
 * (base64, 体积约等于磁盘上那些文件):
 *   * SAC+AZ-MoE / DQN+MCTS-MoE 一类: 3 x 146 MB ≈ 440 MB;
 *   * **PPO+MCTS 的 TB 骨干是最大的一支: 两份合计 ~1.06 GB**(README 里那份实测的落盘
 *     体积就是它) —— 这是本机制在内存上最贵的地方, 写在这里免得被读成"几百 MB 而已"。
 * 一轮结束后由后台线程 drop 掉 (见 backgroundTrainLoop), 所以峰值 ≈ +1 份, 不累积;
 * 评估模式的开场快照只在开了逐字节审计时才留到对局结束 (不开审计时"建冻结实例"读完就放)。
 *
 * 为什么不在"关掉改道"时统一 clear: 关的那一刻后台线程可能正好有一轮在飞, 清掉会让它
 * 读不到刚写下的种子 (那一轮会被丢弃并报一条警告 —— 不是损坏, 但是凭空白跑一轮)。
 * 所以释放的责任交给**知道生命周期的人**: 一轮结束 drop 自己的前缀, 快照用完 drop 自己的前缀。
 *
 * ---- 退出时必须关掉 (否则退出保存会写进内存然后随进程消失) ----
 *
 * `saveAllInstantiatedAgentsOnExit()` 第一件事就是把改道关掉 —— 这道防线不能只靠
 * "退出时对弈恰好结束了"。理由: 用户完全可以在**一局棋下到一半**时关窗, 那时改道
 * 还开着, 而退出保存写的是标准路径 (不受改道影响) —— 但把这条写死在这里, 是为了让
 * 将来任何"退出路径上写临时文件"的想法也不会静默丢数据。
 *
 * ---- 为什么不做成"静默": 计数 ----
 *
 * 改道次数与字节数都记着 (divertedWrites / divertedBytes), 由 ChessBoard 暴露出去,
 * 测试直接断言"对弈期间改道发生了 N 次, 而 weights/ 目录一个临时文件都没多出来"。
 * 本工程的惯例: 一个机制如果没有任何读数, 它与"没生效"在界面上完全一样。
 * ================================================================
 */

#include <atomic>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace RL {
namespace WeightIO {

/*
 * 所有"临时权重文件"的公共前缀 (单一来源)。
 * ⚠ 改这里等于改判据: chessboard.cpp 的运行期自检会核对 tmpWeightsOf / 快照 / 审计
 *   三个来源都以它开头。
 */
inline constexpr const char *kTransientPrefix = "weights/_temp";

struct State
{
    /* 改道开关: 对弈期间为真 (由 ChessBoard 置位, 每个进程一个) */
    std::atomic<bool> divert{false};
    /* 路径 -> 权重文件内容 (只在改道开着时有内容) */
    std::mutex mutex;
    std::map<std::string, std::string> blobs;
    /* 诊断计数: 被改道的写次数 / 字节数, 以及磁盘上真实发生的临时文件写 */
    std::atomic<long long> writes{0};
    std::atomic<long long> bytes{0};
};

inline State &state()
{
    static State s;      /* C++11: 函数内静态初始化是线程安全的 */
    return s;
}

/* 这个路径是不是"临时权重路径" (判据的唯一实现) */
inline bool isTransientPath(const std::string &path)
{
    const std::size_t n = std::strlen(kTransientPrefix);
    return path.size() >= n && path.compare(0, n, kTransientPrefix) == 0;
}

inline bool divertEnabled()
{
    return state().divert.load(std::memory_order_relaxed);
}

/*
 * 这一次写/读要不要走内存 (判据的唯一入口)。
 * 关着 -> 永远 false (行为与改动前逐字节相同, 这是默认状态)。
 */
inline bool shouldDivert(const std::string &path)
{
    return divertEnabled() && isTransientPath(path);
}

inline void setDivertEnabled(bool on)
{
    state().divert.store(on, std::memory_order_relaxed);
}

/*
 * 把一份权重内容存进内存。返回 true = 已接管 (**调用方不许再写磁盘**);
 * 返回 false = 不改道, 调用方照原来的方式写盘。
 */
inline bool divertWrite(const std::string &path, std::string data)
{
    if (!shouldDivert(path)) {
        return false;
    }
    State &s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    std::string &slot = s.blobs[path];
    s.bytes.fetch_add((long long)data.size() - (long long)slot.size());
    slot = std::move(data);
    s.writes.fetch_add(1);
    return true;
}

/* 内存里有没有这个路径的那一份 (判据同 shouldDivert: 关着就永远没有) */
inline bool divertLookup(const std::string &path, std::string &out)
{
    if (!shouldDivert(path)) {
        return false;
    }
    State &s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    auto it = s.blobs.find(path);
    if (it == s.blobs.end()) {
        return false;
    }
    out = it->second;
    return true;
}

/*
 * ---- 给"文件在不在 / 多大"这两个判据用 (weightFileWritten / weightFileReadable) ----
 *
 * 没有这两个, 改道会当场把功能弄坏: saveModel() 用"文件存在且非空"判成功, 而改道之后
 * 磁盘上**故意**没有那个文件 —— 于是后台训练每一轮都会报"种子权重写入失败"并跳过,
 * 表现是"开着对弈就完全不训练", 而原因与文件系统毫无关系。
 */
inline bool memorySize(const std::string &path, long long &bytes)
{
    if (!shouldDivert(path)) {
        return false;
    }
    State &s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    auto it = s.blobs.find(path);
    if (it == s.blobs.end()) {
        return false;
    }
    bytes = (long long)it->second.size();
    return true;
}

/*
 * 读一份权重内容: **内存优先, 磁盘回落**。
 * 返回 true = out 里有内容 (出自内存或磁盘)。
 * 回落是有意的: 改道关着的时候 (非对弈期间) 一切照旧, 而改道刚打开时内存里还没有
 * 任何东西 —— 这时读磁盘上的旧临时文件, 与改动前的行为完全一致。
 */
inline bool readBytes(const std::string &path, std::string &out)
{
    if (divertLookup(path, out)) {
        return true;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size <= 0) {
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((std::size_t)size);
    f.read(&out[0], size);
    if (!f.good() && !f.eof()) {
        return false;
    }
    return true;
}

/* 丢掉一份内存副本 (一轮训练跑完 / 快照用完时调用 —— 见 backgroundTrainLoop) */
inline void drop(const std::string &path)
{
    State &s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    auto it = s.blobs.find(path);
    if (it == s.blobs.end()) {
        return;
    }
    s.bytes.fetch_sub((long long)it->second.size());
    s.blobs.erase(it);
}

/*
 * "这一份权重此刻在不在" —— 内存里有, 或者磁盘上有。
 * ⚠ 不要用 readBytes 去探在不在 (那是把 146 MB 整个拷一份出来问"你在吗") ——
 *   逐字节审计那一类"对每个后缀先探存在性"的循环上, 差的就是几百 MB 的临时拷贝。
 */
inline bool exists(const std::string &path)
{
    long long bytes = 0;
    if (memorySize(path, bytes)) {
        return true;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

/* 全部丢掉 (改道关闭时调用) */
inline void clear()
{
    State &s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.blobs.clear();
    s.bytes.store(0);
}

inline long long divertedWrites() { return state().writes.load(); }
inline long long divertedBytes()  { return state().bytes.load(); }

/* 计数归零 (对局报告要的是**本场**的数, 与 m_frozenDecisions 同一条理由) */
inline void resetCounters()
{
    State &s = state();
    s.writes.store(0);
    s.bytes.store(0);
}

}   /* namespace WeightIO */
}   /* namespace RL */

#endif /* RL_WEIGHTIO_HPP */
