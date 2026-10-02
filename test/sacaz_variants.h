#ifndef SACAZ_VARIANTS_H
#define SACAZ_VARIANTS_H

/*
 * ================================================================
 *  sacaz_variants.h —— 工具/测试侧"按骨干选类"的唯一一处接线
 * ================================================================
 *
 *  [2026-09 独立类拆分] SAC+AZ+MCTS 这一族现在有**四个互不继承的类**:
 *
 *      SACAZAgent          纯 MLP (两个 Tanh 隐层)              AGENT_SACAZ
 *      SACAZMoEMlpAgent    稀疏 MoE(E=8, top-2) + MLP 专家      AGENT_SACAZ_MOE_MLP
 *      SACAZMoETbAgent     稀疏 MoE(E=8, top-2) + TB 专家       AGENT_SACAZ_MOE
 *      SACAZLegacyAgent    59e5233 行为还原版 (自带两支骨干)     AGENT_SACAZ_OLD*
 *
 *  拆分前它们是**一个类 + `Backbone` 枚举**: 工具里 `--backbone=tb` 只是给同一个
 *  构造函数换一个枚举值。现在骨干 = **类型**, 所以工具有两件事要做:
 *
 *    1. **按变体构造**: `withSacazAgent()` —— 一个 switch 里把四个类的构造参数各自
 *       写清楚, 调用方只拿到"一个已经建好的 agent 引用"。
 *    2. **把只属于某些骨干的读数抹平**: 表结构 `MoeInfo` / `TbHeadInfo` ——
 *       纯 MLP 没有路由、CoE-MLP 没有注意力头, 于是"读专家数"这类代码在四个变体上
 *       写法**同形** (默认返回 0/-1), 不需要每个工具里再写一遍 `if (是 TB 吗)`。
 *
 *  为什么不让工具直接 `dynamic_cast`: 那是**运行期**判断, 写错了要跑到那一步才知道;
 *  这里是模板特化, 写错了编译不过。而且 dynamic_cast 会把"这个类有没有这个成员"
 *  这件事从编译期搬到运行期 —— 正是本工程反复栽的那个跟头。
 *
 *  诚实记一笔**代价**: 四个类各自持有一份实现 (用户口径: 隔离必须是结构性的),
 *  所以"给所有变体加一个新读数"要改四处。这个文件的作用就是把**工具侧**的这类改动
 *  收敛到一处。
 * ================================================================
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <type_traits>   /* callAndReturn 判 f 的返回类型 */

#include "sacazagent.h"
#include "sacazmoemlpagent.h"
#include "sacazmoetbagent.h"

namespace sacazx {

/* ----------------------------------------------------------------
 *  1. 变体枚举 (命令行拼写 = 拆分前 `--backbone=` 的四个取值, 一个都没改)
 * ---------------------------------------------------------------- */
enum class Variant {
    Mlp = 0,
    MoeMlp,
    MoeTb,
    DenseMoeTb
};

/* 命令行 -> 变体。**拼写与拆分前逐字相同** (`mlp` / `moe-mlp` / `tb` / `dense-tb`),
   所以既有的脚本与文档不用改。 */
inline bool parseVariant(const std::string &s, Variant &out)
{
    if (s == "mlp")      { out = Variant::Mlp;       return true; }
    if (s == "moe-mlp")  { out = Variant::MoeMlp;    return true; }
    if (s == "tb")       { out = Variant::MoeTb;     return true; }
    if (s == "dense-tb") { out = Variant::DenseMoeTb; return true; }
    return false;
}

inline const char *variantKey(Variant v)
{
    switch (v) {
    case Variant::Mlp:       return "mlp";
    case Variant::MoeMlp:    return "moe-mlp";
    case Variant::MoeTb:     return "tb";
    case Variant::DenseMoeTb: return "dense-tb";
    }
    return "?";
}

/* 变体的中文名 —— 与各 `backboneName()` 的返回值一致 (报告里两处不能各说一套) */
inline const char *variantName(Variant v)
{
    switch (v) {
    case Variant::Mlp:       return "MLP";
    case Variant::MoeMlp:    return "稀疏MoE(MLP专家)";
    case Variant::MoeTb:     return "稀疏MoE(TB专家)";
    case Variant::DenseMoeTb: return "稠密MoE(TB专家,对照)";
    }
    return "?";
}

/* ----------------------------------------------------------------
 *  1b. 变体的**结构常量** —— 必须取自"这个变体真正建出来的那个类"
 *
 *  [2026-10 E=8/top-2 回归当场抓到的一个真实错误] 工具/测试里要断言"专家数 = ?"时,
 *  原来是写 `SACAZAgent::MOE_TB_EXPERTS` —— 但 `test_sacaz` 的骨干扫描建的是
 *  **独立类** `SACAZMoETbAgent`, 它有自己的同名常量 (4/1 -> 8/2 时只改了它自己那份)。
 *  于是: **测试的红灯看着像"代码错了", 其实是测试盯错了类** —— 这是四个类互不继承
 *  在工具侧的直接代价, 也是最容易犯的一类错 (同名常量在四个类里各有一份)。
 *
 *  所以这里给"按变体问结构"留**唯一一处**入口: 值来自那个变体实际建出来的类,
 *  改 E/top-k 时工具侧一个字都不用动。`Mlp` 没有 MoE 层 -> 0/0 (与 `MoeInfo` 把
 *  "只有某些骨干才有的读数"抹平是同一个口径)。
 *
 *  注意这**不是**自证: `moeExpertCount()` 读的是网里那个 `ISparseMoE` 层**建出来之后**
 *  的 `expertCount()` (见 sacazmoetbagent.cpp), 所以"类常量 == 层读数"这条断言查的是
 *  `buildNet` 有没有真的用这个常量建层 (写死成别的数字就会红)。
 * ---------------------------------------------------------------- */
struct VariantShape {
    int experts;   /* 稀疏 MoE 层建了几个专家 (0 = 这个变体没有 MoE 层) */
    int topK;      /* 每次前向激活几个 (dense 对照 = experts) */
};

inline VariantShape variantShape(Variant v)
{
    switch (v) {
    case Variant::Mlp:        return { 0, 0 };
    case Variant::MoeMlp:     return { SACAZMoEMlpAgent::MOE_MLP_EXPERTS,
                                       SACAZMoEMlpAgent::MOE_MLP_TOPK };
    case Variant::MoeTb:      return { SACAZMoETbAgent::MOE_TB_EXPERTS,
                                       SACAZMoETbAgent::MOE_TB_TOPK };
    /* "等参数不等算力"的对照: 专家数同上, 但**全算** —— topK = experts */
    case Variant::DenseMoeTb: return { SACAZMoETbAgent::MOE_TB_EXPERTS,
                                       SACAZMoETbAgent::MOE_TB_EXPERTS };
    }
    return { 0, 0 };
}

/* ----------------------------------------------------------------
 *  2. 构造参数 (四个类的 ctor 形状不同, 这里放公共的那几个)
 * ---------------------------------------------------------------- */
struct Opts {
    int   hidden = 64;          /* 隐层宽度 (四个类都有) */
    float gamma = 0.99f;
    float lr = 0.001f;
    float cpuct = 1.5f;
    int   expertHidden = 64;    /* 只用 MLP 专家时有效 (两个 MoE 类接受它) */
    float aux = 0.1f;           /* MoE 负载均衡辅助损失系数 */
    /*
       `TrunkMode` 是**每个类自己的嵌套枚举** (独立类各有一份, 类型互不兼容) ——
       所以这里存 `bool` 而不是枚举: 值 -> 枚举的映射写在下面每个分支里。
         false = TrunkMode::Separate (拆分前的默认) / true = TrunkMode::Shared
    */
    bool  shared = false;
    bool  tbHonorHeads = true;  /* 只用 TB 专家时有效; 默认 = 修好的口径 */
};

/* ----------------------------------------------------------------
 *  3. 按变体构造并跑一段泛型代码
 *
 *      return sacazx::withSacazAgent(chess, variant, opts, [&](auto &sac) {
 *          ... 对 sac 的写法对四个类同形 ...
 *          return 0;            // 也可以不返回 (void) -> 当作退出码 0
 *      });
 *
 *  `f` 的返回值就是本函数的返回值 (工具里通常返回 main 的退出码);
 *  **`f` 返回 void 也接受** (测试里很多段落只做断言, 没有退出码可言) —— 那时返回 0,
 *  于是同一套模板既能伺候工具也能伺候测试。
 *  变体非法时返回 `onBadVariant` (默认 2) —— 调用方应当已经校验过命令行,
 *  这里是兜底: 不静默跑另一个骨干。
 * ---------------------------------------------------------------- */
namespace detail {
template <class F, class A>
int callAndReturn(F &&f, A &a)
{
    if constexpr (std::is_void<typename std::invoke_result<F, A &>::type>::value) {
        f(a);
        return 0;
    } else {
        return (int)f(a);
    }
}
}   /* namespace detail */

template <class F>
int withSacazAgent(Chess &chess, Variant v, const Opts &o, F &&f, int onBadVariant = 2)
{
    switch (v) {
    case Variant::Mlp: {
        SACAZAgent sac(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                       o.shared ? SACAZAgent::TrunkMode::Shared
                                : SACAZAgent::TrunkMode::Separate);
        return detail::callAndReturn(f, sac);
    }
    case Variant::MoeMlp: {
        SACAZMoEMlpAgent sac(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                             o.expertHidden, o.aux,
                             o.shared ? SACAZMoEMlpAgent::TrunkMode::Shared
                                      : SACAZMoEMlpAgent::TrunkMode::Separate);
        return detail::callAndReturn(f, sac);
    }
    case Variant::MoeTb: {
        SACAZMoETbAgent sac(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                            o.expertHidden, o.aux,
                            o.shared ? SACAZMoETbAgent::TrunkMode::Shared
                                     : SACAZMoETbAgent::TrunkMode::Separate,
                            o.tbHonorHeads);
        return detail::callAndReturn(f, sac);
    }
    case Variant::DenseMoeTb: {
        /*
           "等参数不等算力"的对照组: 同一个类, 只是 4 个专家全算。
           **必须走构造参数 `denseMoe_`** —— 建网在构造函数里发生, 构造之后再赋值是
           静默空操作 (第一版就是这么写的, 拆分回归当场抓到: dense-tb 的指纹与拆分前
           不同, topK=1 / 3305 次前向 而不是 topK=4 / 13804 次)。
        */
        SACAZMoETbAgent sac(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                            o.expertHidden, o.aux,
                            o.shared ? SACAZMoETbAgent::TrunkMode::Shared
                                     : SACAZMoETbAgent::TrunkMode::Separate,
                            o.tbHonorHeads, true /* denseMoe */);
        return detail::callAndReturn(f, sac);
    }
    }
    return onBadVariant;
}

/* ----------------------------------------------------------------
 *  4. 把"只有某些骨干才有的读数"抹平 (默认值 = 这个骨干上它真的没有)
 * ---------------------------------------------------------------- */

/* 稀疏 MoE 的路由读数: 纯 MLP 没有 MoE 层 -> 专家 0 个 / topK 0 / 直方图空 */
template <class A> struct MoeInfo {
    static int expertCount(const A &) { return 0; }
    static int topK(const A &) { return 0; }
    static void usage(const A &, std::vector<long long> &out) { out.clear(); }
    static void resetUsage(A &) {}
    static void usageSplit(const A &, std::vector<long long> &tr, std::vector<long long> &inf)
    {
        tr.clear();
        inf.clear();
    }
};

template <> struct MoeInfo<SACAZMoEMlpAgent> {
    static int expertCount(const SACAZMoEMlpAgent &a) { return a.moeExpertCount(); }
    static int topK(const SACAZMoEMlpAgent &a) { return a.moeTopK(); }
    static void usage(const SACAZMoEMlpAgent &a, std::vector<long long> &out) { a.moeUsage(out); }
    static void resetUsage(SACAZMoEMlpAgent &a) { a.resetMoeUsage(); }
    static void usageSplit(const SACAZMoEMlpAgent &a, std::vector<long long> &tr,
                           std::vector<long long> &inf) { a.moeUsageSplit(tr, inf); }
};

template <> struct MoeInfo<SACAZMoETbAgent> {
    static int expertCount(const SACAZMoETbAgent &a) { return a.moeExpertCount(); }
    static int topK(const SACAZMoETbAgent &a) { return a.moeTopK(); }
    static void usage(const SACAZMoETbAgent &a, std::vector<long long> &out) { a.moeUsage(out); }
    static void resetUsage(SACAZMoETbAgent &a) { a.resetMoeUsage(); }
    static void usageSplit(const SACAZMoETbAgent &a, std::vector<long long> &tr,
                           std::vector<long long> &inf) { a.moeUsageSplit(tr, inf); }
};

/*
   ----------------------------------------------------------------
   [2026-10 门控实验] 无辅助损失的偏置均衡 —— 只有两个 MoE 骨干有它。
   纯 MLP / legacy 支没有稀疏 MoE 层 (或没有该开关), 这里是**空操作**,
   与 MoeInfo 把"只有某些骨干才有的读数"抹平是同一个做法。
   ----------------------------------------------------------------
*/
template <class A> struct MoeBias {
    static bool available() { return false; }
    static void set(A &, bool, float) {}
    static bool enabled(const A &) { return false; }
    static float rate(const A &) { return 0.0f; }
};

template <> struct MoeBias<SACAZMoEMlpAgent> {
    static bool available() { return true; }
    static void set(SACAZMoEMlpAgent &a, bool on, float r)
    {
        a.lossFreeBias = on;
        a.lossFreeBiasRate = r;
    }
    static bool enabled(const SACAZMoEMlpAgent &a) { return a.lossFreeBias; }
    static float rate(const SACAZMoEMlpAgent &a) { return a.lossFreeBiasRate; }
};

template <> struct MoeBias<SACAZMoETbAgent> {
    static bool available() { return true; }
    static void set(SACAZMoETbAgent &a, bool on, float r)
    {
        a.lossFreeBias = on;
        a.lossFreeBiasRate = r;
    }
    static bool enabled(const SACAZMoETbAgent &a) { return a.lossFreeBias; }
    static float rate(const SACAZMoETbAgent &a) { return a.lossFreeBiasRate; }
};

/* TB 专家的注意力头口径: 只有 TB 那一支有 -> 其余返回 -1 (与拆分前的返回值一致) */
template <class A> struct TbHeadInfo {
    static int requested(const A &) { return -1; }
    static int used(const A &) { return -1; }
    static int dim(const A &) { return -1; }
    static int allocated(const A &) { return -1; }
    static long long elements(const A &) { return -1; }
};

template <> struct TbHeadInfo<SACAZMoETbAgent> {
    static int requested(const SACAZMoETbAgent &a) { return a.tbHeadsRequested(); }
    static int used(const SACAZMoETbAgent &a) { return a.tbHeadsUsed(); }
    static int dim(const SACAZMoETbAgent &a) { return a.tbHeadDim(); }
    static int allocated(const SACAZMoETbAgent &a) { return a.tbHeadsAllocated(); }
    static long long elements(const SACAZMoETbAgent &a) { return a.tbAttentionElements(); }
};

/* ----------------------------------------------------------------
 *  4b. 训练诊断的统一入口 (TrainDiag 是每个类**各自**嵌套类型 —— 独立的代价)
 *      三个当前口径的类都有同名同字段的 `getTrainDiag()`, 所以一个模板就够;
 *      将来若某个类没有它, 这里编译不过 (而不是静默拿到空读数)。
 * ---------------------------------------------------------------- */
template <class A>
inline const typename A::TrainDiag &diagOf(const A &a)
{
    return a.getTrainDiag();
}

/* ----------------------------------------------------------------
 *  5. AnySac —— "任意骨干的 SAC+AZ 实例" (给要把 agent 存进结构体/数组的工具用)
 *
 *  为什么需要它: 拆分前 `--backbone=tb` 只是给**一个类**换枚举值, 工具里可以
 *  `SACAZAgent *p = new SACAZAgent(...)`; 现在骨干 = **类型**, 一个容器装不下三种。
 *
 *  设计: 三个 unique_ptr 只用一个非空 + 一个 `visit()` 逃生口。
 *    * **公共 API** 走 `base()` (AgentBase: selectMove / preTrainThenDecide / getName …);
 *    * **骨干相关的一切** (旋钮、训练、诊断读数) 走 `visit([&](auto &s){ ... })` ——
 *      lambda 里 `s` 是**真实类型**, 照旧 `s.rewardScale = ...` / `s.moeUsage(...)`,
 *      编译期派发 (不是 dynamic_cast), 写错了编译不过。
 *
 *  代价诚实记一笔: `visit` 的 lambda 会被实例化三份 (三个类各一份) —— 这正是
 *  "独立类"的隔离在工具侧的样子。
 * ---------------------------------------------------------------- */
class AnySac {
public:
    AnySac(Chess &chess, Variant v, const Opts &o) : v_(v)
    {
        switch (v) {
        case Variant::Mlp:
            mlp_.reset(new SACAZAgent(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                                      o.shared ? SACAZAgent::TrunkMode::Shared
                                               : SACAZAgent::TrunkMode::Separate));
            base_ = mlp_.get();
            break;
        case Variant::MoeMlp:
            moemlp_.reset(new SACAZMoEMlpAgent(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                                               o.expertHidden, o.aux,
                                               o.shared ? SACAZMoEMlpAgent::TrunkMode::Shared
                                                        : SACAZMoEMlpAgent::TrunkMode::Separate));
            base_ = moemlp_.get();
            break;
        case Variant::MoeTb:
            moetb_.reset(new SACAZMoETbAgent(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                                             o.expertHidden, o.aux,
                                             o.shared ? SACAZMoETbAgent::TrunkMode::Shared
                                                      : SACAZMoETbAgent::TrunkMode::Separate,
                                             o.tbHonorHeads, false));
            base_ = moetb_.get();
            break;
        case Variant::DenseMoeTb:
            moetb_.reset(new SACAZMoETbAgent(chess, o.hidden, o.gamma, o.lr, o.cpuct,
                                             o.expertHidden, o.aux,
                                             o.shared ? SACAZMoETbAgent::TrunkMode::Shared
                                                      : SACAZMoETbAgent::TrunkMode::Separate,
                                             o.tbHonorHeads, true /* denseMoe */));
            base_ = moetb_.get();
            break;
        }
    }

    Variant variant() const { return v_; }
    AgentBase *base() { return base_; }

    /* 骨干相关的一切: 真实类型在这里面 */
    template <class F>
    decltype(auto) visit(F &&f)
    {
        switch (v_) {
        case Variant::MoeMlp:     return f(*moemlp_);
        case Variant::MoeTb:      return f(*moetb_);
        case Variant::DenseMoeTb: return f(*moetb_);
        case Variant::Mlp:        return f(*mlp_);
        }
        return f(*mlp_);   /* 不可达 (变体已校验); 留着让各编译器都满意 */
    }
    template <class F>
    decltype(auto) visit(F &&f) const
    {
        switch (v_) {
        case Variant::MoeMlp:     return f(*moemlp_);
        case Variant::MoeTb:      return f(*moetb_);
        case Variant::DenseMoeTb: return f(*moetb_);
        case Variant::Mlp:        return f(*mlp_);
        }
        return f(*mlp_);
    }

    /* 几条最常用的转发 (三个类的签名完全相同) —— 免得每处都写一遍 visit */
    Step selectMove(int color, int simulations, float temperature)
    {
        return visit([&](auto &s) { return s.selectMove(color, simulations, temperature); });
    }
    void trainSelfPlay(int episodes, int simulations_, int maxMoves, bool verbose = false,
                       float tempRoot = 1.0f, float tempFinal = 0.25f, int learnEveryMoves = 4)
    {
        visit([&](auto &s) {
            s.trainSelfPlay(episodes, simulations_, maxMoves, verbose,
                            tempRoot, tempFinal, learnEveryMoves);
        });
    }
    bool loadModel(const std::string &p) { return visit([&](auto &s) { return s.loadModel(p); }); }
    bool saveModel(const std::string &p) { return visit([&](auto &s) { return s.saveModel(p); }); }
    int getLearnSteps() const { return visit([](const auto &s) { return s.getLearnSteps(); }); }
    float getAlpha() const { return visit([](const auto &s) { return s.getAlpha(); }); }
    std::size_t getMemorySize() const { return visit([](const auto &s) { return s.getMemorySize(); }); }
    long long getLeafEvals() const { return visit([](const auto &s) { return s.getLeafEvals(); }); }
    std::string selfCheckReport() const { return visit([](const auto &s) { return s.selfCheckReport(); }); }
    /* 训练诊断 (类型擦除版): 字段与三个类一致, 这里统一成"一份副本" */
    struct Diag {
        long long n = 0, clamped = 0, doneSamples = 0, decisiveSamples = 0;
        /* [2026-09 人机终局通道] 棋盘从外面补进来的终局反馈条数 (见 aiagent.h) */
        long long externalTerminals = 0;
        double yPreAbsSum = 0.0, qAbsMeanSum = 0.0, qSpreadSum = 0.0;
    };
    Diag diag() const
    {
        return visit([](const auto &s) {
            const auto &d = s.getTrainDiag();
            Diag o;
            o.n = d.n; o.clamped = d.clamped;
            o.doneSamples = d.doneSamples; o.decisiveSamples = d.decisiveSamples;
            o.externalTerminals = d.externalTerminals;
            o.yPreAbsSum = d.yPreAbsSum; o.qAbsMeanSum = d.qAbsMeanSum;
            o.qSpreadSum = d.qSpreadSum;
            return o;
        });
    }

    const char *backboneName() const
    {
        return visit([](const auto &s) { return s.backboneName(); });
    }
    int moeExpertCount() const
    {
        return visit([](const auto &s) {
            using A = typename std::decay<decltype(s)>::type;
            return MoeInfo<A>::expertCount(s);
        });
    }
    int moeTopK() const
    {
        return visit([](const auto &s) {
            using A = typename std::decay<decltype(s)>::type;
            return MoeInfo<A>::topK(s);
        });
    }
    void moeUsage(std::vector<long long> &out) const
    {
        visit([&](const auto &s) {
            using A = typename std::decay<decltype(s)>::type;
            MoeInfo<A>::usage(s, out);
        });
    }
    void resetMoeUsage()
    {
        visit([](auto &s) {
            using A = typename std::decay<decltype(s)>::type;
            MoeInfo<A>::resetUsage(s);
        });
    }

private:
    Variant v_;
    AgentBase *base_ = nullptr;
    std::unique_ptr<SACAZAgent> mlp_;
    std::unique_ptr<SACAZMoEMlpAgent> moemlp_;
    std::unique_ptr<SACAZMoETbAgent> moetb_;
};

}   /* namespace sacazx */

#endif   /* SACAZ_VARIANTS_H */
