#include "sacazlegacyagent.h"

#include "chessstate.h"   /* 完备 Markov 状态的公共实现 (规则上下文/规范格/动作双射) */
#include "agentrollout.hpp"
#include "rl/sparse_moe.hpp"
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
/*
   [2026-10 第五轮] 只有"专家换成序列 Transformer"的那一支才需要这个头 —— 与
   sacazmoetbagent.cpp 里那条同名 include 同一个纪律: 默认构建的编译面一个字节都没动
   (与它的权重/读数逐位不变相配套)。
   (头文件里已有一条同样的 include, 那是为了让类里的 `SeqExpert` typedef 成立; 这里再写
   一次是为了让"这一段依赖谁"在使用点可见 —— 有 include guard, 不会重复展开。)
*/
#include "rl/seq_transformer.hpp"
#endif

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>      /* selfCheckReport 的别名统计: 槽位 -> 该槽位上的互不相同走法 */
#include <set>

/* ============================================================
 *  构造: 三个网络
 *
 *  策略网: 1260 -> h -> h -> 128 **logits** (输出层不带激活)
 *    掩码 softmax 在 agent 内自己做 (见 maskedSoftmax), 所以这里输出 logits 而不是
 *    概率 —— 否则"掩码后归一化"的雅可比与 `Layer<Softmax>` 内部缓存的 o 对不上。
 *
 *  双 critic: 1260 -> h -> h -> 128 个 Q 值, 输出层同样是 `Layer<Linear>`。
 *    **刻意不用 Sigmoid 头**: 象棋奖励含负值 (输棋 −1, 丢子为负), 而
 *    `Layer<Sigmoid>` 值域是 (0,1), 结构上无法表示负 Q —— 这正是
 *    docs/issues_review.md 里 B18 记的问题 (`dqn.cpp` 至今仍用 Sigmoid 头)。
 *
 *  目标网: 只前向, withGrad=false。
 * ============================================================ */
namespace {

/*
   初始化缩放。`iFcLayer` 的构造函数把权重初始化成 U(-1,1); 对 1260 维输入来说
   pre-activation 的标准差约 sqrt(1260/3) ≈ 20 —— Tanh 一上来就饱和, 梯度接近 0,
   网络基本学不动。这里按 1/sqrt(fan_in) 再缩放一遍, 把 pre-activation 的标准差
   拉回 ~0.6。隐藏层是标准做法; 输出层顺带让初始 Q ≈ 0、初始策略接近均匀, 对探索有利。
*/
void scaleLayerInit(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::iFcLayer *fc = dynamic_cast<RL::iFcLayer*>(net[i]);
        if (fc == nullptr) {
            continue;
        }
        const float fanIn = (float)(fc->inputDim > 1 ? fc->inputDim : 1);
        const float s = 1.0f / std::sqrt(fanIn);
        for (std::size_t k = 0; k < fc->w.size(); k++) {
            fc->w[k] *= s;
        }
        for (std::size_t k = 0; k < fc->b.size(); k++) {
            fc->b[k] *= s;
        }
    }
}

/* 找出网络里第一个稀疏 MoE 层 (没有就返回 nullptr) */
RL::ISparseMoE *findSparseMoe(RL::Net &net)
{
    for (std::size_t i = 0; i < net.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(net[i]);
        if (m != nullptr) {
            return m;
        }
    }
    return nullptr;
}

#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
/*
   一个 SparseMoE 层 -> iLayer::sptr 的小工厂 (与 sacazmoetbagent.cpp 的同名帮手同源,
   理由也一样)。

   为什么要它: 稀疏/稠密两种 TopK 是两个**不同的模板实例** (TopK 是模板参数), 而它们的
   共同基类 ISparseMoE 不是 shared_ptr 能直接 covariant 转换的类型 —— 显式
   static_pointer_cast 一次, 两个骨干的调用点就不必各写一份几乎相同的 make_shared。

   ⚠ 只在序列专家那一支里存在: `Expert` 是 `SACAZLegacyAgent::SeqExpert`, 而那个 typedef
   本身只在开关打开时存在 (见头文件) —— 所以整个助手跟着守卫走, 默认构建里不该出现它。
*/
template<typename Expert, int E, int K>
RL::iLayer::sptr makeMoeLayer(int d, bool withGrad, int hidden)
{
    return std::static_pointer_cast<RL::iLayer>(
        std::make_shared<RL::SparseMoE<Expert, E, K> >(d, withGrad, hidden));
}

/*
   造"序列 Transformer 专家"那一层 (模板参数 K = topK: 稀疏骨干用 MOE_TB_TOPK,
   稠密对照骨干用 MOE_TB_EXPERTS ⇒ "等参数不等算力" 那条对照关系保持不变)。

   tokDim 的来源与参考实现 (sacazmoetbagent.cpp 的 makeTbExpertMoe) 逐字同口径:
   `expertHidden > 0 ? expertHidden : 64`。本类构造函数已经把 expertHidden 夹成
   ">0, 否则 64", 这里照抄同一个写法, 是为了两边的参数形状与读数能逐字对照。
   宽度必须能被 MOE_SEQ_HEADS 整除, 不满足时序列专家**自己**会 abort (它会把话说清楚),
   这里不重复判一次 —— 判两处就会有两份可能漂移的口径。
*/
template<int K>
RL::iLayer::sptr makeSeqExpertMoe(int dIn, bool withGrad, int expertHidden)
{
    typedef SACAZLegacyAgent::SeqExpert Expert;
    const int tokDim = (expertHidden > 0) ? expertHidden : 64;
    RL::iLayer::sptr layer =
        makeMoeLayer<Expert, SACAZLegacyAgent::MOE_TB_EXPERTS, K>(dIn, withGrad, tokDim);

    /*
       位置编码必须**显式**设上: 序列专家的默认是 `LEARN_1D` (老行为), 而界面 PPO 支选的
       是可学 2D —— 既然这一支的卖点就是"与 PPO 支同一个专家", 位置编码就不能各说一套。
       **设不上就响亮失败**: 返回 false 说明"外层开关说自己是序列专家, 但实际建出来的专家
       不认这个键", 也就是配置与实现不一致; 继续跑下去只会得到一份"看起来在训练、其实
       用的是老位置编码"的权重 (本工程记过多次"旋钮设了却没生效"的事故)。
       (SparseMoE 会把这个键转发给每个专家, 只有序列专家响应它。)
    */
    if (!layer->setPosModeByKey(SACAZLegacyAgent::moeSeqPosMode())) {
        std::fprintf(stderr,
                     "[SACAZLegacyAgent] 位置编码模式 \"%s\" 没有被专家接受 —— "
                     "外层开关 (SACAZ_LEGACY_MOE_EXPERT_SEQ) 与实际建出来的专家类型不一致。\n",
                     SACAZLegacyAgent::moeSeqPosMode());
        std::fflush(stderr);
        std::abort();
    }
    return layer;
}
#endif

} // namespace

const char *SACAZLegacyAgent::backboneName(Backbone b)
{
    /*
       [2026-10 第五轮] `SACAZ_LEGACY_MOE_EXPERT_SEQ=1` 的构建里, 两个 TB 骨干的专家换成了
       序列 Transformer ⇒ **名字必须跟着变**。理由是"读数不许说谎": 一个 TB 专家 12.9 M
       参数, 序列专家 91,731 (1/140), 前向的算力量级也不同 —— 印同一个名字会让
       "哪一支更强"这种结论直接被记到错的那本账上 (自检面板/测试都读这一处)。
       默认构建 (开关关) 的四个字符串**一字未改**。
    */
    switch (b) {
    case Backbone::Mlp:          return "MLP";
    case Backbone::SparseMoeMlp: return "稀疏MoE(MLP专家)";
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
    case Backbone::SparseMoeTb:  return "稀疏MoE(序列Transformer专家)";
    case Backbone::DenseMoeTb:   return "稠密MoE(序列Transformer专家,对照)";
#else
    case Backbone::SparseMoeTb:  return "稀疏MoE(TB专家)";
    case Backbone::DenseMoeTb:   return "稠密MoE(TB专家,对照)";
#endif
    default:                     return "?";
    }
}

/*
   按 backbone 造网络。
   输出层始终是 `Layer<Linear>` (不是 Sigmoid): 象棋奖励含负值, Q 必须能取负
   (docs/issues_review.md B18)。掩码 softmax 在 agent 里自己做, 所以策略头输出
   **logits**, 不是概率。
   两个结构上的注意点:
     * 稀疏 MoE 层是"同维进出"的 (专家的输入输出必须同维才能做门控加权和),
       所以后面必须再接一层 Tanh(d_model -> h) 把 STATE_DIM 维压到 h 维, 再进 Linear 头。
       (d_model = STATE_DIM: 改前 1263 / 对齐 1710; 专家的**内部**宽度与它无关 ——
       TB 专家用 MOE_TB_HEADS/MOE_TB_DFF, 序列专家用 expertHidden, MLP 专家也用 expertHidden。)
     * 专家权重由 SparseMoE 的构造函数调用 scaleExpertInit 缩放; 其余普通层由
       scaleLayerInit 缩放 (两者的依据都是 1/sqrt(fan_in))。
*/
RL::Net SACAZLegacyAgent::buildNet(bool withGrad) const
{
    const std::size_t h = (std::size_t)(hiddenDim > 0 ? hiddenDim : 64);
    RL::Net::Layers layers;

    /*
       隐层激活 = `Layer<RL::Tanh>` (与 59e5233 **逐字相同**的代码)。
       **不要**改用 `TanhNorm<Linear>` 且 r=1 去"复现"它: 那个层的偏置是加在 tanh
       外面的 (`tanh(r·Wx)+b`), 与这里的 `tanh(Wx+b)` 不是同一个函数 —— 实测同权重同
       局面下 max|ΔQ| = 8.9e-06 (test_sacaz [14])。完整来龙去脉见 sacazagent.h 里
       `hiddenActivationName()` 上面那段注释。
    */
    switch (backbone) {
    case Backbone::Mlp:
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        layers.push_back(RL::Layer<RL::Tanh>::_(h, h, true, withGrad));
        break;
    case Backbone::SparseMoeMlp:
        layers.push_back(std::make_shared<RL::SparseMoE<RL::MlpExpert,
                                                        MOE_MLP_EXPERTS,
                                                        MOE_MLP_TOPK> >(
            STATE_DIM, withGrad, expertHidden > 0 ? expertHidden : 64));
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    case Backbone::SparseMoeTb:
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
        /*
           [2026-10 第五轮] 用户口径: "把 TB 专家换成序列 transformer, 输入 19x90 特征,
           与 PPO agent 对齐"。这一支的专家 = `SeqExpert`
           (SeqTransformerExpert<90,4,2>, 与 PPO 支同一份实现), 稀疏路由不变 (E=4, top-1)。
           前提与后果逐条写在头文件 MOE_EXPERT_SEQ 那一段 (必须对齐表示 / 换权重前缀 /
           宽度改由 expertHidden 决定); 位置编码在助手函数里设上, 设不上当场 abort。
        */
        layers.push_back(makeSeqExpertMoe<MOE_TB_TOPK>(STATE_DIM, withGrad, expertHidden));
#else
        layers.push_back(std::make_shared<RL::SparseMoE<RL::TransformerBlock<MOE_TB_HEADS, MOE_TB_DFF>,
                                                        MOE_TB_EXPERTS,
                                                        MOE_TB_TOPK> >(STATE_DIM, withGrad, 0));
#endif
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    case Backbone::DenseMoeTb:
        /* TopK == NumExperts => 门控照旧, 但四个专家全算 (等参数不等算力的对照) */
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
        /* 同上, 只把 TopK 换成专家数 —— 对照关系 (同参数 / 不同算力) 在序列专家下同样成立 */
        layers.push_back(makeSeqExpertMoe<MOE_TB_EXPERTS>(STATE_DIM, withGrad, expertHidden));
#else
        layers.push_back(std::make_shared<RL::SparseMoE<RL::TransformerBlock<MOE_TB_HEADS, MOE_TB_DFF>,
                                                        MOE_TB_EXPERTS,
                                                        MOE_TB_EXPERTS> >(STATE_DIM, withGrad, 0));
#endif
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        break;
    default:
        layers.push_back(RL::Layer<RL::Tanh>::_(STATE_DIM, h, true, withGrad));
        layers.push_back(RL::Layer<RL::Tanh>::_(h, h, true, withGrad));
        break;
    }

    layers.push_back(RL::Layer<RL::Linear>::_(h, ACTION_DIM, true, withGrad));
    RL::Net net(layers);
    /*
       只缩放普通的 iFcLayer。稀疏 MoE 层本身不是 iFcLayer (dynamic_cast 会返回
       nullptr), 它的专家权重已经在 SparseMoE 的构造函数里用 scaleExpertInit 缩过了。
       (这里若重复缩放同一个层会把初始化标准差再压一次, 所以每层只走一次。)
       这一段与 59e5233 逐字相同 —— 谁能被缩、缩多少, 也是"逐位复现"的一部分
       (当年用 TanhNorm 去顶替那一层时, 它同样是 iFcLayer 的子类, 所以连这个缩放
       都看不出差别; 见上面 hiddenActivationName() 那段)。
    */
    scaleLayerInit(net);
    return net;
}

/*
 * 实际生效的隐层激活 —— **读建好的网络**, 不是回显某个开关。
 *
 * 为什么要有它: 上一轮的回归就是这一层被从 `Layer<Tanh>` 换成了 `TanhNorm<Sigmoid>`
 * (随机权重下棋力 -26 个点), 而**面板上一个字都看不出来** —— 参数量、权重指纹全都
 * 一样 (同形状层)。这里把 actor 第 2 层的**真实类型**报出来, 于是"激活被换了"这件事
 * 在自检面板上直接可见。它同时是一条回归断言: 正常情况下必须报 `tanh (Layer<Tanh>)`,
 * 正常的两个骨干 (Mlp / 稀疏 MoE 的专家骨干) 都是这一层。
 */
const char *SACAZLegacyAgent::hiddenActivationName() const
{
    RL::Net &self = const_cast<RL::Net &>(actor);   /* Net::operator[] 没有 const 重载 */
    if (self.size() < 2) {
        return "? (网络层数不足, buildNet 改坏了?)";
    }
    if (dynamic_cast<RL::Layer<RL::Tanh> *>(self[1]) != nullptr) {
        return "tanh (Layer<Tanh>) [59e5233 同款]";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Sigmoid> *>(self[1]) != nullptr) {
        return "**TanhNorm<Sigmoid> —— 这就是那个掉 26 个点的回归!**";
    }
    if (dynamic_cast<RL::TanhNorm<RL::Linear> *>(self[1]) != nullptr) {
        return "TanhNorm<Linear> —— 与 Layer<Tanh> **不等价** (偏置加在 tanh 外,"
               " 实测 max|dQ|=8.9e-06), 见 sacazagent.h";
    }
    return "其它 (既不是 Layer<Tanh> 也不是已知的 TanhNorm —— 检查 buildNet 第 2 层)";
}

/*
 * 界面上的哪一支 —— 本类服务**两个**界面类型, 区别只有骨干:
 *   AGENT_SACAZ_OLD     -> Mlp        (与 AGENT_SACAZ 同骨干)
 *   AGENT_SACAZ_OLD_MOE -> SparseMoeTb (与 AGENT_SACAZ_MOE 同骨干)
 * 所以标签按**骨干**分支, 而不是写死一句 (两支的参数量/耗时/自检读数差很多, 混在一本
 * 账上会让"哪一支更强"这种结论直接错)。
 * (2026-09 之前这里是 virtual 覆写: 那一版是 SACAZAgent 的派生类, 基类也要报自己的
 * 身份; 现在两个类没有继承关系, 各自的标签各写各的。)
 *
 * [2026-09 修正] 两条**不许退化的性质** (test_sacaz [14] 逐条钉住):
 *   1. **每个**骨干的标签都含 `59e5233` —— 第一版给其余骨干返回的是一句
 *      `bench/测试构造 (界面不为它建实例)`, 那句里没有 59e5233, 于是"还原版的身份"
 *      在非界面骨干上**丢了** (test_sacaz [14] 用 Mlp / SparseMoeMlp 两个骨干构造实例,
 *      这条断言在 SparseMoeMlp 上如实失败)。身份是**口径**的属性, 不是骨干的属性 ⇒
 *      59e5233 一律保留, 骨干只决定**后缀**。
 *   2. 四个骨干的标签**两两不同**, 且界面上的两个骨干各自带自己的**界面类型名**
 *      (AGENT_SACAZ_OLD / AGENT_SACAZ_OLD_MOE) —— 面板第一行要能同时回答
 *      "是不是 59e5233 口径"与"是哪一支"。
 * 非界面骨干在标签里**明说**"只在 test/bench 构造": 面板上选不到它, 不明说就会有人
 * 拿它的读数当界面读数用。
 * 判据只有这一处: 实例版本 guiAgentLabel() 只是转发到 guiAgentLabelFor(backbone),
 * 所以测试可以对**四个**骨干全查一遍, 而不用构造一个 TB 实例 (建网 + 拷贝权重很贵)。
 *
 * [2026-10 第五轮] 序列专家开关打开时, 骨干名与标签里的"专家是什么"跟着换
 * (SparseMoeTb / DenseMoeTb 两个分支各自 #if): 骨干**枚举**不变 (仍是这两个骨干, 界面
 * 类型也不变), 变的是专家 —— 而面板第一行要同时回答"是哪一支"与"它的专家是什么",
 * 所以后半句必须分支。
 */
const char *SACAZLegacyAgent::guiAgentLabel() const
{
    return guiAgentLabelFor(backbone);
}

const char *SACAZLegacyAgent::guiAgentLabelFor(Backbone b)
{
    switch (b) {
    case Backbone::Mlp:
        /* 界面类型 AGENT_SACAZ_OLD (与 AGENT_SACAZ 同骨干) */
        return "SAC+AZ-59e5233 (AGENT_SACAZ_OLD, 行为还原版)";
    case Backbone::SparseMoeTb:
        /* 界面类型 AGENT_SACAZ_OLD_MOE (与 AGENT_SACAZ_MOE 同骨干) */
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
        /* 口径一字不改, 变的只有专家 —— 标签里必须写出这一点, 否则与 TB 那一支混账 */
        return "SAC+AZ-59e5233-MoE (AGENT_SACAZ_OLD_MOE, 还原口径 + 稀疏MoE/序列Transformer专家)";
#else
        return "SAC+AZ-59e5233-MoE (AGENT_SACAZ_OLD_MOE, 还原口径 + 稀疏MoE/TB专家)";
#endif
    case Backbone::SparseMoeMlp:
        /*
           不是界面类型: 这一支只在 test/bench 里构造 (test_sacaz [14] 就是它 ——
           "TanhNorm 顶替那一层"的回归检验发生在这个骨干上)。
        */
        return "SAC+AZ-59e5233-MoE-MLP (非界面骨干: 稀疏MoE(MLP专家), 只在 test/bench 构造)";
    case Backbone::DenseMoeTb:
        /* 不是界面类型: "等参数不等算力"的对照组, 也只在 test/bench 构造 */
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
        return "SAC+AZ-59e5233-DenseTB (非界面骨干: 稠密MoE(序列Transformer专家)对照,"
               " 只在 test/bench 构造)";
#else
        return "SAC+AZ-59e5233-DenseTB (非界面骨干: 稠密MoE(TB专家)对照, 只在 test/bench 构造)";
#endif
    default:
        /* 兜底也保留 59e5233: 以后再加骨干时, "身份丢失"不会以这种方式静默发生 */
        return "SAC+AZ-59e5233-? (未知骨干; 口径仍是 59e5233 还原版)";
    }
}

/*
 * 权重前缀 —— 必须与 SACAZAgent 的 "weights/sacaz_agent" 不同 (用户口径: 新旧权重文件
 * 用不同名字区分开)。两者参数结构完全相同, 结构指纹挡不住串权重, 共用前缀会让
 * "后训练的那一支静默覆盖另一支"。
 *
 * [2026-09] 按**骨干**再分一次: 本类现在服务两个界面类型 (Mlp 与 SparseMoeTb), 它们的
 * 参数量不同、训练口径相同但权重不能互换 —— 一个骨干一个前缀, 免得"这份文件是哪一支的"
 * 只能靠猜。界面只用前两个; 另外两个是 bench 里的对照骨干。
 * [2026-10 第五轮] 专家换序列 Transformer 时再各换一个名字 (结构变了 ⇒ 名字必须变),
 * 逐条见下面函数体里的说明与头文件的 MOE_EXPERT_SEQ 那一段。
 */
const char *SACAZLegacyAgent::defaultWeightPrefix()
{
    return defaultWeightPrefix(Backbone::Mlp);
}

const char *SACAZLegacyAgent::defaultWeightPrefix(Backbone b)
{
    /*
       [2026-10 第五轮] 序列专家开关打开时, 两个"TB 骨干"的专家换成序列 Transformer ⇒
       **结构与参数量都变了** (一个专家 12.9 M -> 91,731) ⇒ 权重必须换前缀:
         * 结构指纹/参数量守卫只能拒绝**载入**, 拦不住"写出时把旧的盖掉" —— 共用一个前缀会
           让新的空网覆盖盘上那份 TB 专家的权重 (PPO 支换成 seq 骨干时踩过同一条, 所以
           那边的做法也是换名: weights/ppomcts_seq_agent.dat);
         * 反过来, 换了名字之后"载不到"是**看得见**的 (自检报告第一段就印本实例的前缀),
           不会是静默错读成 TB 专家的那一份。
       一个骨干一个前缀是本类的纪律 (见头文件), 所以两个 seq 变体各自再分一个。
       默认构建下这四个字符串**一字未改**。
    */
    switch (b) {
    case Backbone::Mlp:          return "weights/sacaz_old_agent";
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
    case Backbone::SparseMoeTb:  return "weights/sacaz_old_moe_seq_agent";
#else
    case Backbone::SparseMoeTb:  return "weights/sacaz_old_moe_agent";
#endif
    case Backbone::SparseMoeMlp: return "weights/sacaz_old_moemlp_agent";
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
    case Backbone::DenseMoeTb:   return "weights/sacaz_old_dense_seq_agent";
#else
    case Backbone::DenseMoeTb:   return "weights/sacaz_old_densetb_agent";
#endif
    default:                     return "weights/sacaz_old_agent";
    }
}

int SACAZLegacyAgent::moeExpertCount() const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->expertCount() : 0;
}

int SACAZLegacyAgent::moeTopK() const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    return (m != nullptr) ? m->topK() : 0;
}

void SACAZLegacyAgent::moeUsage(std::vector<long long> &out) const
{
    RL::Net &self = const_cast<RL::Net&>(actor);
    RL::ISparseMoE *m = findSparseMoe(self);
    if (m == nullptr) {
        out.clear();
        return;
    }
    m->usageSnapshot(out);
}

void SACAZLegacyAgent::resetMoeUsage()
{
    for (std::size_t i = 0; i < actor.size(); i++) {
        RL::ISparseMoE *m = dynamic_cast<RL::ISparseMoE*>(actor[i]);
        if (m != nullptr) {
            m->resetUsage();
        }
    }
}

SACAZLegacyAgent::SACAZLegacyAgent(Chess &chess_,
                       int hiddenDim_,
                       float gamma_,
                       float lr,
                       float cpuct,
                       Backbone backbone_,
                       int expertHidden_,
                       float auxLossCoef_)
    : chess(chess_),
      backbone(backbone_),
      expertHidden(expertHidden_ > 0 ? expertHidden_ : 64),
      auxLossCoef(auxLossCoef_),
      hiddenDim(hiddenDim_ > 0 ? hiddenDim_ : 64),
      gamma(gamma_),
      learningRateActor(lr),
      learningRateCritic(lr),
      learningRateAlpha(1e-3f),
      c_puct(cpuct),
      azWeight(1.0f),
      /*
         ---- α 自动调节的口径: 目标熵 0.98 / 学习率 1e-3 (2026-09 实测改回) ----
         上一轮把目标熵降到 0.5、把 alpha 学习率提到 5e-3, 理由是"实测 α 恒为初值 0.200
         ⇒ 自动调节名存实亡"(见 docs/arena_sac_vs_ppo_report.md §8.1)。**那个理由本身
         没错, 但结论是错的**: 在"边下边学 + 对 MCTS"的受控协议下 (bench_sac_learn, 20 局
         一档, MCTS 200 次模拟, 固定 mcts-srand, 同一 seed):

             目标熵 0.5  + αlr 5e-3 (改后)  -> 2 胜 7 负 11 和   37.5%
             目标熵 0.98 + αlr 1e-3 (改回)  -> 13 胜 0 负 7 和   **82.5%**
             只改目标熵 0.98 (αlr 仍 5e-3)  -> 10 胜 2 负 8 和   70.0%
             只改 αlr 1e-3 (目标熵仍 0.5)   -> 8 胜 11 负 1 和   42.5%
             四个 α 口径改回去的配置合池 80 局 -> 38 胜 6 负 36 和 = 71.9%
         也就是说:**改后那一支是对弈里"训练越久越弱"的直接原因** —— 把 α 口径改回去之后
         训练才开始有用 (未训练的对照只有 52.5% / 70.0%, 而改后 82.5% 已经超过它)。

         机制上也能对上 (同一工具训练后的读数): 改后那一支的 critic **尺度死掉了**
         (|Q| 均值 **0.035**, 与随机初始化 0.06 同量级), 而 αlr 大 5 倍 + 目标熵低一半
         会让 α 迅速缩小、策略被"没有信息的 Q"推着走; 改回 0.98/1e-3 之后 |Q| 落在
         **2.04** (那一档当时开着"目标钳位 2", 所以贴着钳位边界) —— 本类**不夹目标**,
         所以这里的 |Q| 是 5.6 那一档, 见下面"本类刻意不含 critic 值域约束"那段。
         这两组数是**两件事**: α 口径决定 critic 有没有信号, 值域约束决定它会不会发散。
         完整数据与复现命令见 docs/sac_learn_reward_2026_09.md §9。 */
      entropyRatio(0.98f),
      simulations(64),
      batchSize(32),
      /*
         [F1 2026-09] 目标网同步率在本类里**不是成员**, 而是两个编译期常数
         (POLYAK_TAU = 1e-3 / TARGET_SYNC_EVERY = 64, 见头文件里那张口径表)。

         背景: SACAZAgent 把这一对做成了可调成员, 于是 2026-09 的 F1 那一轮把**基类
         默认值**改成了"硬拷贝 / 每 256 步" —— 而当时"59e5233 行为还原版"是派生类,
         没有显式钉这两个量, 于是它的行为**跟着一起变了**(见
         docs/sac_critic_diagnosis_2026_09.md §13.5, 用户发现的就是这一条)。
         本类与 SACAZAgent **没有继承关系**(见头文件的说明), 而且把这一对写成常数:
         从外面没有任何办法把这一支调到别的同步率。
         数值上的依据 (为什么 1e-3/64 就是 59e5233 的行为): 一次 20 局的会话 ~2600 次
         learn 只把目标网从随机初始化挪动 2~4%, 所以自举项 V(s') 里的 E[min Q(s')]
         几乎恒为"随机网络的输出" —— 而 |Q_target| ≈ 0.08 正是这个现象的直接读数
         (bench_sac_learn --legacy 的实测)。这是**还原对象的一部分**, 不是待修的缺陷。
      */
      maxMemorySize(4096),
      totalEpisodes(0),
      learnSteps(0),
      m_leafEvals(0)
{
    totalWins[0] = 0;
    totalWins[1] = 0;

    actor = buildNet(true);
    q1 = buildNet(true);
    q2 = buildNet(true);
    q1Target = buildNet(false);
    q2Target = buildNet(false);

    /*
       目标网构建时 withGrad=false, 但参数仍然是随机初始化的 —— 必须从在线网拷一份
       过去, 否则目标网从"另一个随机点"开始, 自举项一开始就是纯噪声。
       (Net 的拷贝是**浅拷贝**: 共享层指针; 深拷贝必须走 copyTo。)
    */
    q1.copyTo(q1Target);
    q2.copyTo(q2Target);

    /* 温度 α: 标量, 自动调节 (SAC-Discrete 的做法) */
    alpha = RL::GradValue(1, 1);
    alpha[0] = 0.2f;

    m_stateBuf = RL::Tensor(STATE_DIM, 1);
    m_logits = RL::Tensor(ACTION_DIM, 1);
    m_q1 = RL::Tensor(ACTION_DIM, 1);
    m_q2 = RL::Tensor(ACTION_DIM, 1);
}

std::string SACAZLegacyAgent::getName() const
{
    /* 界面的对局日志/结果标签用这个名字; 与 AGENT_SACAZ 的名字明确区分开 */
    return "SAC+MCTS+AlphaZero (59e5233 行为还原版)";
}

/* ============================================================
 *  状态编码 (规范视角), **平面优先**布局: 下标 = plane * CELLS + cell
 *    对齐表示: 19 平面 (14 棋子 + 5 规则/阶段上下文) = STATE_DIM 1710,
 *              与 PPOMCTSAgent 逐位同口径 (CTX_* 顺序 / canonicalCell 镜像同一份);
 *    改前表示: 同一份 14 个棋子平面, 上下文只占尾部 3 个标量槽 = STATE_DIM 1263。
 *  两种表示共用下面这份**格子编码**, 差别只在"上下文放哪儿" —— 那一段在
 *  writeContext / readContext 里分支 (见那里的说明)。
 *  (备注: 这里原来还写着"下面有一条 static_assert 把两边维度一致钉成编译期事实" —— 那条
 *   断言现在不存在了, 理由见 contextOf 上面那段: CTX_COUNT 依赖表示开关, 而 static_assert
 *   在编译期无条件求值, 于是在改前表示下必然失败。一致性改由 test_sacaz 的运行期断言盯。)
 * ============================================================ */
void SACAZLegacyAgent::encodeSparse(int color, std::vector<std::uint16_t> &cells) const
{
    cells.clear();
    cells.reserve(32);
    for (int i = 0; i < 32; i++) {
        Stone *s = chess.m_children[i];
        if (s == nullptr || s->alive == false || s->type < 0 || s->type >= 7) {
            continue;
        }
        /* 轮到黑方时左右镜像 (x -> 9-x): "己方"永远在 x 大的那一侧。
           镜像只有一份实现 (src/chessstate.h), 这里调它。 */
        const int cell = ChessState::canonicalCell(s->pos.x, s->pos.y, color);
        /* 平面下标 = type*2 + (是己方?0:1), 与棋盘状态公共约定一致 */
        const int plane = s->type * 2 + ((s->color == color) ? 0 : 1);
        cells.push_back((std::uint16_t)(plane * CELLS + cell));
    }
}

/*
   5 个规则/阶段上下文 —— **顺序与数值口径就是 chessstate.h 的 CTX_* 那一份**,
   与 PPOMCTSAgent 写进 plane 14..18 的值逐个相同 (那边调 contextValues 再铺平面,
   这边取同样的 5 个数进 Transition::ctx)。
   这一致性是"两边状态同构"的全部内容: 只要有一边换了公式, 对照就不再受控。
*/
void SACAZLegacyAgent::contextOf(Chess &c, int color, float out[CTX_COUNT])
{
    /*
       两种表示下的**语义不同**, 这点很关键:
         * 对齐表示 (5 个): 与 PPOMCTS 的 plane 14..18 逐位相同, 顺序 = chessstate.h 的
           CTX_* (子力阶段 / 总手数 / 无吃子 / 重复 / 被将);
         * 改前表示 (3 个): **只有 3 个规则上下文**, 顺序 = [无吃子, 重复, 被将]
           —— 与 59e5233 的 ctx[3] 逐位一致 (那时没有"子力阶段/总手数"两项)。
       混用这两种顺序会静默错位 (值还是有限数, 只是喂错了通道), 所以写成显式分支。
    */
    /*
       注意: 这里**不能**用 static_assert(CTX_COUNT == ChessState::CTX_COUNT) 做门槛 ——
       CTX_COUNT 是依赖表示开关的常量, 断言在编译期无条件求值, 于是在**改前表示**
       (CTX_COUNT=3) 下必然失败, 哪怕那一支根本不会执行 (if constexpr 也一样, 因为它
       的丢弃规则不覆盖"模板外但依赖常量"的 static_assert)。两种口径的一致性改由
       test_sacaz 的运行期断言钉住 (它会在对齐表示下比较 CTX_COUNT 与 chessstate.h)。
    */
    if (ALIGNED_REPR) {
        double ctx[ChessState::CTX_COUNT];
        ChessState::contextValues(c, color, ctx);
        for (int i = 0; i < CTX_COUNT; i++) {
            out[i] = (float)ctx[i];
        }
    } else {
        out[0] = (float)ChessState::halfmovePhase(c);
        out[1] = (float)ChessState::repetitionPhase(c);
        out[2] = (float)ChessState::checkPhase(c, color);
    }
}

void SACAZLegacyAgent::writeContext(RL::Tensor &state, const float ctx[CTX_COUNT])
{
    if (state.size() < (std::size_t)STATE_DIM) {
        return;
    }
    if (ALIGNED_REPR) {
        /* 对齐: 上下文是**整平面** (每平面 90 个相同值), 与 PPOMCTS 的布局逐位相同 */
        for (int p = 0; p < CTX_COUNT; p++) {
            const float v = ctx[p];
            float *dst = &state[(std::size_t)(PIECE_PLANES + p) * CELLS];
            for (int i = 0; i < CELLS; i++) {
                dst[i] = v;
            }
        }
    } else {
        /* 改前: 上下文是棋子区之后的 CTX_COUNT 个标量槽 (稀疏回放友好) */
        for (int i = 0; i < CTX_COUNT; i++) {
            state[(std::size_t)(CTX_BASE + i)] = ctx[i];
        }
    }
}

void SACAZLegacyAgent::readContext(const RL::Tensor &state, float out[CTX_COUNT])
{
    for (int i = 0; i < CTX_COUNT; i++) {
        out[i] = 0.0f;
        if (state.size() < (std::size_t)STATE_DIM) {
            continue;
        }
        if (ALIGNED_REPR) {
            out[i] = state[(std::size_t)((PIECE_PLANES + i) * CELLS)];
        } else {
            out[i] = state[(std::size_t)(CTX_BASE + i)];
        }
    }
}

void SACAZLegacyAgent::expandSparse(const std::vector<std::uint16_t> &cells, RL::Tensor &state)
{
    state.zero();
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

void SACAZLegacyAgent::denseToSparse(const RL::Tensor &state, std::vector<std::uint16_t> &cells)
{
    cells.clear();
    cells.reserve(32);
    /* 只取**棋子平面**里非零的格: 上下文那 5 个槽不是"格子", 由 ctx[] 单独带。
       注意上界必须是 CTX_BASE 而不是 state.size() —— 否则 flat 布局下会把
       plane 14..18 的常数整平面 (90 个/平面) 全当成"非零格"塞进稀疏列表。 */
    const std::size_t boardLimit = (state.size() < (std::size_t)CTX_BASE)
                                       ? state.size() : (std::size_t)CTX_BASE;
    for (std::size_t i = 0; i < boardLimit; i++) {
        if (state[i] != 0.0f) {
            cells.push_back((std::uint16_t)i);
        }
    }
}

/* 稀疏格列表 -> 14 个棋子平面的 one-hot (plane*CELLS+cell 的下标直接就是扁平下标,
   所以与 flat 布局的棋子区完全对应)。 */
static void expandSparseGrids(const std::vector<std::uint16_t> &cells, RL::Tensor &state)
{
    for (std::size_t i = 0; i < cells.size(); i++) {
        const std::size_t idx = (std::size_t)cells[i];
        if (idx < state.size()) {
            state[idx] = 1.0f;
        }
    }
}

/* 5 个上下文标量铺成整平面 (每条 90 个相同值) —— 只在对齐表示下用。
   为什么值走标量而不是"再塞 450 个非零格": 稀疏回放只存棋子格; 稠密展开时按平面铺开
   是**确定性的函数**, 不增加任何回放内存。 */
static void fillContextPlanes(RL::Tensor &state, const float ctx[SACAZLegacyAgent::CTX_COUNT])
{
    for (int p = 0; p < SACAZLegacyAgent::CTX_COUNT; p++) {
        const float v = ctx[p];
        if (v == 0.0f) {
            continue;   /* 0 平面本来就是零, 跳过 (省 90 次写) */
        }
        float *dst = &state[(std::size_t)(SACAZLegacyAgent::PIECE_PLANES + p) * SACAZLegacyAgent::CELLS];
        for (int c = 0; c < SACAZLegacyAgent::CELLS; c++) {
            dst[c] = v;
        }
    }
}

void SACAZLegacyAgent::encodeStateFor(int color, RL::Tensor &state)
{
    if (state.size() != (std::size_t)STATE_DIM) {
        state = RL::Tensor(STATE_DIM, 1);
    }
    state.zero();
    std::vector<std::uint16_t> cells;
    encodeSparse(color, cells);
    expandSparseGrids(cells, state);
    float ctx[CTX_COUNT];
    contextOf(chess, color, ctx);
    /* 两种布局共用同一份上下文数值, 只是"铺平面"与"塞尾部槽"的区别 */
    writeContext(state, ctx);
}

void SACAZLegacyAgent::encodeState(RL::Tensor &state)
{
    /*
       视角由棋盘当前的 sideToMove 决定 —— MCTS 里走法是真正落在棋盘上的
       (`moveForward` 会翻转 sideToMove), 所以这里读到的就是该节点的走棋方。
       `rolloutFromCurrent` 进来时也会把 sideToMove 设成 color, 探索阶段同样正确。
    */
    encodeStateFor(chess.sideToMove, state);
}

/* ============================================================
 *  动作: **与 PPOMCTSAgent 同一个双射**
 *
 *      actionIdx = canonicalCell(from) * 90 + canonicalCell(to)      (8100)
 *
 *  两条要点 (与 PPOMCTS 的注释同源, 因为这就是同一个口径):
 *    * 与棋子 id 无关 —— id 会随吃子回收复用, 所以"同一格走到同一格"在不同局面下
 *      会落到不同槽位, 策略学不到稳定东西。用格子坐标没有这个问题。
 *    * 按 color 做规范镜像, 于是红黑双方的"同一步棋"共享同一个动作槽位,
 *      与状态编码的规范视角一致。
 *
 *  为什么这次必须换掉 128 槽哈希: 同局面平均挤掉 5.16 个着法 ⇒ 两个不同着法的
 *  策略目标与 Q 被强制共用一列, 那是**结构性上限** (训练多少次都消不掉), 而且
 *  它让 SAC 与 PPO 的动作空间不同构, "算法对照"就没法解释。见
 *  docs/agents_design.md §11 / §20.4 与 docs/arena_sac_vs_ppo_report.md §8.4。
 * ============================================================ */
int SACAZLegacyAgent::stepToActionIdx(const Step &s, int color) const
{
    /*
       两种表示的索引公式 (与头文件的开关一一对应):
         * 改前 (默认): `(id*37 + nextPos.x*13 + nextPos.y*7) % 128` —— 与 59e5233 逐字相同;
         * 对齐      : `canonicalCell(from)*90 + canonicalCell(to)` —— 与 PPOMCTS 同一公式。
       注意改前那一条**不用规范镜像**: 它是按"棋子 id + 目标格"哈希的, 而 id 与颜色相关
       (红黑棋子 id 不同段)。这是原实现的口径, 照抄才能复现它的棋力。
       `legacyHashAction` 只在对齐表示下生效 (用来单独隔离动作空间这个变量); 它的默认值
       跟着编译期开关走 —— 序列专家那一支 (SACAZ_LEGACY_MOE_EXPERT_SEQ=1) 默认 **true**
       (用户口径: 那一支保持 128 槽), 见头文件里该成员的说明。
       这个条件就是上面 buildNet 之外**动作编码的单一来源**: 自检报告判"实际用的是哪一套"
       时照抄它 (`hashAction`), 不许另写一套。
    */
    if (!ALIGNED_REPR || legacyHashAction) {
        const unsigned long long h = (unsigned long long)s.id * 37ULL
                                   + (unsigned long long)s.nextPos.x * 13ULL
                                   + (unsigned long long)s.nextPos.y * 7ULL;
        return (int)(h % (unsigned long long)LEGACY_ACTION_DIM);
    }
    const int from = ChessState::canonicalCell(s.pos.x, s.pos.y, color);
    const int to   = ChessState::canonicalCell(s.nextPos.x, s.nextPos.y, color);
    return ChessState::actionIndexOf(from, to);
}

void SACAZLegacyAgent::getLegalActions(int color,
                                 std::vector<Step*> &steps,
                                 std::vector<int> &actionIndices,
                                 RL::Tensor &actionMask)
{
    actionMask.zero();
    chess.sample(color, steps);
    actionIndices.clear();
    actionIndices.reserve(steps.size());
    for (Step *s : steps) {
        const int aidx = stepToActionIdx(*s, color);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;
    }
}

float SACAZLegacyAgent::computeReward(const Step &s, int color)
{
    (void)color;   /* 走子方视角, 与颜色无关 */

    if (s.nextId == Stone::ID_NONE) {
        return stepReward(false, false, 0.0);
    }
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr) {
        return stepReward(false, false, 0.0);
    }
    /*
       奖励尺度 (Phase 1 起与其他 agent 统一):
       本 agent 原来就用 `Stone::value` 原值 (车 0.5 / 马炮 0.3 / 兵 0.1), 方向是对的
       ——"单个吃子 < 终局 ±1"。但**一局累积**下来一方全材质 = 3.5 > 终局 1.0, 也就
       是说"吃光对方"在数值上仍然是"赢棋"的 3.5 倍 (诊断 [2] 的不变量当场抓住这一条,
       它对 SACAZ 同样成立)。现在统一到 stone.h 的 REWARD_MATERIAL_COEF = 0.1:
       一方全材质 0.35 < 终局 1.0, 终局真正主导。

       吃將仍然不给即时奖励 (这一点 SACAZ 原来就是对的, 现在推广到全部 agent):
       它必然是终局, 终局奖励会覆盖; 若按 value_jiang = 1000 给, Q/价值的尺度会被
       彻底压倒, 软备份与 MSE 都没法看。

       符号约定 (2026-09 修正, 见 docs/agents_design.md §17.2): 即时奖励是**走子方
       视角**的 —— 吃掉对方一个子永远是收益。原来的黑方视角写法会让红方白吃一个黑子
       拿到负奖励, 与终局 (resultValue 给的走子方视角 ±1) 相反。
       (Chess::moveForward 的 totalReward 仍是黑方视角, 那是它的记账约定; 界面上的
       奖励曲线在 ChessBoard::playMatchGame 里显式换算成走子方视角。)
       回归钉在 test_match 的 [2.6] 节。

       ---- [2026-09] 本类**刻意不含奖励塑形** ----
       本类没有"即时奖励整体缩放"与"去掉材质 / 终局放大"这两个旋钮 (SACAZAgent 那一支
       有, 它们是 2026-09 之后才加的实验开关, 59e5233 里没有)。
       用户口径是"这一支必须是 59e5233 的行为还原版", 所以这里**连成员都不留** ——
       不是"默认关掉", 而是任何 flag 都打不开 (加了就说明被污染了)。
       即时奖励就是 `stepReward(...)` 原式, 终局就是 engine 的真值 ±1/0。
    */
    return stepReward(true, victim->type == Stone::TYPE_JIANG, victim->value);
}

/*
 * 终局值 (走子方视角) —— 终局口径的**唯一出口**。三个产生点 (搜索叶子 terminalValue /
 * 自对弈 resultValue / rollout 的 outcomeForMover) 全部走这里, 理由是"搜索估的"与
 * "训练学的"必须是同一个游戏。
 *
 * ---- [2026-09] 本类**没有塑形分支** ----
 * SACAZAgent 里这个方法还承担"终局放大" (将死时按败方剩余材质放大到 [1,2), 即
 * 快杀 > 磨死)。那是 2026-09 用户提议的实验旋钮, 59e5233 里没有, 所以本类**只返回引擎
 * 真值** (±1 / 0) —— 连那个成员都不存在, 没有任何 flag 能打开它。
 * 函数名保留 (agentrollout.hpp 用 SFINAE 探测这个可选成员: 有就用它, 没有就退回共享的
 * outcomeForMover), 但因为返回值就是真值, 这条路径与回退路径**逐位等价**。
 *
 * 不再读 this->chess: 真值只取决于 (result, perspective), 与棋盘无关。读棋盘是塑形
 * 分支的需要 (它要数败方剩多少子) —— 那一支已经删掉, 于是这里也就成了纯函数。
 */
float SACAZLegacyAgent::terminalReward(int chessResult, int perspective) const
{
    return (float)outcomeForMover(chessResult, perspective);
}

/* ============================================================
 *  掩码 softmax 及其反向
 * ============================================================ */
void SACAZLegacyAgent::maskedSoftmax(const RL::Tensor &logits, const RL::Tensor &mask,
                               RL::Tensor &pi)
{
    pi.zero();
    float maxLogit = -std::numeric_limits<float>::max();
    bool any = false;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            any = true;
            if (logits[a] > maxLogit) {
                maxLogit = logits[a];
            }
        }
    }
    if (!any) {
        return;   /* 无合法动作: 保持全 0, 调用方必须先处理这种情况 */
    }
    float sum = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        if (mask[a] > 0.5f) {
            const float e = std::exp(logits[a] - maxLogit);
            pi[a] = e;
            sum += e;
        }
    }
    if (sum <= 0.0f) {
        pi.zero();
        return;
    }
    const float inv = 1.0f / sum;
    for (int a = 0; a < ACTION_DIM; a++) {
        pi[a] *= inv;
    }
}

void SACAZLegacyAgent::maskedSoftmaxBackward(const RL::Tensor &pi, const RL::Tensor &g,
                                       RL::Tensor &dz)
{
    /*
       dL/dz_c = π_c·( g_c − Σ_a g_a·π_a )
       非法动作 π_c = 0 -> 梯度为 0, 与掩码语义一致 (永远不会把非法动作抬起来)。
    */
    float dot = 0.0f;
    for (int a = 0; a < ACTION_DIM; a++) {
        dot += g[a] * pi[a];
    }
    for (int a = 0; a < ACTION_DIM; a++) {
        dz[a] = (pi[a] > 0.0f) ? (pi[a] * (g[a] - dot)) : 0.0f;
    }
}

void SACAZLegacyAgent::maskToBits(const RL::Tensor &mask, std::uint64_t bits[2])
{
    bits[0] = 0;
    bits[1] = 0;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] > 0.5f) {
            bits[(i < 64) ? 0 : 1] |= (std::uint64_t(1) << (i & 63));
        }
    }
}

void SACAZLegacyAgent::bitsToMask(const std::uint64_t bits[2], RL::Tensor &mask)
{
    mask.zero();
    for (int i = 0; i < ACTION_DIM; i++) {
        const std::uint64_t bit = (bits[(i < 64) ? 0 : 1] >> (i & 63)) & 1ULL;
        mask[i] = bit ? 1.0f : 0.0f;
    }
}

/* ============================================================
 *  前向 / 软价值
 * ============================================================ */
void SACAZLegacyAgent::policy(const RL::Tensor &state, const RL::Tensor &mask,
                        RL::Tensor &pi)
{
    /* actor 的输出缓冲会被下一次 forward 覆盖, 先拷出来再做掩码归一化 */
    m_logits = actor.forward(state);
    maskedSoftmax(m_logits, mask, pi);
}

void SACAZLegacyAgent::qValues(const RL::Tensor &state, RL::Tensor &q1Out, RL::Tensor &q2Out)
{
    q1Out = q1.forward(state);
    q2Out = q2.forward(state);
}

/*
 * ---- [2026-09] 本类**没有稀疏头推理** ----
 * SACAZAgent 那一支有三个"只算合法列"的函数 (稀疏策略 / 稀疏 Q / 稀疏软价值) 与它们的
 * 静态助手, 服务于"叶子估值走稀疏头"的快路径 (为 8100 动作空间才加的, 由那边的一个
 * 开关选择)。59e5233 的叶子估值是**全量** Q, 所以本类只需要 policy() / qValues() /
 * softValueFrom() 这一条路径 —— 那四个函数一起删掉了: 少一条与训练不同的代码路径,
 * 也少一个"两个口径差在哪"的变量。
 */

void SACAZLegacyAgent::qTargetValues(const RL::Tensor &state, RL::Tensor &q1Out,
                               RL::Tensor &q2Out)
{
    q1Out = q1Target.forward(state);
    q2Out = q2Target.forward(state);
}

float SACAZLegacyAgent::softValueFrom(const RL::Tensor &pi, const RL::Tensor &mask,
                                const RL::Tensor &q1In, const RL::Tensor &q2In) const
{
    const float a = alpha[0];
    float v = 0.0f;
    for (int i = 0; i < ACTION_DIM; i++) {
        if (mask[i] <= 0.5f || pi[i] <= 0.0f) {
            continue;
        }
        const float qmin = std::min(q1In[i], q2In[i]);
        /*
           ---- [2026-09] 本类没有"熵项去处"开关 ----
           SACAZAgent 那边可以把熵项从软价值里摘掉 (用来量"是熵项把 critic 顶走的")。
           本类**不含那个成员**: 熵项就是加上的 (`v = E[min Q] + α·H`), 与 59e5233 逐位
           一致的一行原式。要做那个消融就到 SACAZAgent 那一支去做 —— 在"行为还原版"上
           做消融等于把它变成另一支算法。
        */
        const float ent = a * std::log(pi[i]);
        v += pi[i] * (qmin - ent);
    }
    return v;
}

/* ============================================================
 *  MCTS
 * ============================================================ */
double SACAZLegacyAgent::getPUCT(int childID, int parentVisits) const
{
    const AZNode &child = nodes[childID];
    if (child.visitCount == 0) {
        /* 未访问过的子节点优先 (AlphaZero 的 PUCT 里 U 项在 N=0 时最大) */
        return std::numeric_limits<double>::max();
    }
    /*
       符号 (2026-09 修正): `totalValue` 按**当前走棋方视角**累计 (见 sacazagent.h 的
       字段注释与 backup 的逐层翻号), 而子节点的走棋方就是父节点的对手 ——
       所以父节点比较时必须取负号。漏掉它等于最大化对手的价值:
       搜索专挑对自己最差的着法, 且评估越准越糟 (症状是"loss 降、棋力不涨",
       以及不敢吃子 —— 吃子后对手少大子, 子节点 Q 对对手为负, 被算成亏着)。
       与 PPOMCTSAgent::getPUCT 的修正同一处口径, 两个 agent 必须一致。
    */
    const double q = -child.getQ();
    const double u = c_puct * child.prior
                     * std::sqrt((double)parentVisits)
                     / (1.0 + (double)child.visitCount);
    return q + u;
}

bool SACAZLegacyAgent::terminalValue(int color, double &value) const
{
    const int res = chess.getResult(color);
    if (res == Chess::RESULT_ONGOING) {
        return false;
    }
    /*
       走**同一家**的终局口径 (terminalReward): 搜索叶子估的值必须与训练目标同一个数,
       否则 PUCT 是在为一个与实际学的不同的游戏排序。
       (本类没有塑形, 所以这里与 outcomeForMover 逐位相同; 这条纪律是留给"万一以后
       终局口径要变"的 —— 三个产生点必须同时变。)
    */
    value = (double)terminalReward(res, color);
    return true;
}

bool SACAZLegacyAgent::resultValue(int result, int color, float &out)
{
    if (result == Chess::RESULT_ONGOING) {
        return false;
    }
    /* 同样走 terminalReward (这条老路径保留公开签名, 供测试/诊断单独调用) */
    out = terminalReward(result, color);
    return true;
}

void SACAZLegacyAgent::visitDistribution(int rootID, RL::Tensor &pi)
{
    pi.zero();
    int total = 0;
    for (int childID : nodes[rootID].childIDs) {
        total += nodes[childID].visitCount;
    }
    if (total <= 0) {
        return;
    }
    const float inv = 1.0f / (float)total;
    for (int childID : nodes[rootID].childIDs) {
        const AZNode &c = nodes[childID];
        pi[c.parentAction] += (float)c.visitCount * inv;
    }
}

/* ============================================================
 *  selectMove: AlphaZero 式 PUCT + SAC 软价值叶子
 *
 *  一次模拟 = 选择(按 PUCT 下潜) -> 展开(先验最高的未尝试动作) ->
 *             估值(终局用真实胜负, 否则用**在线**双 Q 的软价值) ->
 *             回传(negamax 翻转) -> 撤销试走
 *
 *  与 PPOMCTSAgent 的三点差别:
 *    1. 叶子估值是软价值 min_i Q_i − α·log π (最大熵), 而不是一个价值头;
 *    2. 终局节点用真实胜负, 不再自举 (PPOMCTS 在终局也用网络值);
 *    3. 展开时按**先验**挑动作, 而不是随机挑 (同样模拟次数下更有效)。
 * ============================================================ */

Step SACAZLegacyAgent::selectMove(int color, int simulations_, float temp, RL::Tensor *piOut)
{
    nodes.clear();
    if (simulations_ < 1) {
        simulations_ = 1;
    }
    nodes.reserve((std::size_t)simulations_ + 64);

    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor qa(ACTION_DIM, 1);
    RL::Tensor qb(ACTION_DIM, 1);

    /* ---- 根节点 ---- */
    std::vector<Step*> rootSteps;
    std::vector<int> rootIdx;
    getLegalActions(color, rootSteps, rootIdx, mask);

    AZNode root;
    root.currentColor = color;
    root.legalCount = (int)rootIdx.size();
    root.parentID = -1;
    root.parentAction = -1;

    if (root.legalCount > 0) {
        encodeStateFor(color, m_stateBuf);
        policy(m_stateBuf, mask, pi);
    }
    for (std::size_t i = 0; i < rootIdx.size(); i++) {
        root.untriedActionIndices.push_back(rootIdx[i]);
        root.untriedSteps.push_back(*rootSteps[i]);
        root.untriedPriors.push_back((double)pi[rootIdx[i]]);
    }
    Steps::instance().put(rootSteps);

    nodes.push_back(root);
    const int rootID = 0;

    if (root.legalCount == 0) {
        /* 无合法走法: 返回无效 Step, 由调用方按"真无棋可走"处理 (见 issues_review C2) */
        if (piOut != nullptr) {
            piOut->zero();
        }
        return Step();
    }

    /* ---- 主循环 ---- */
    for (int sim = 0; sim < simulations_; sim++) {
        std::vector<int> path;
        path.push_back(rootID);
        int nodeID = rootID;

        /* --- 1. SELECT --- */
        std::vector<Step> toApply;
        while (nodes[nodeID].untriedActionIndices.empty()
               && !nodes[nodeID].childIDs.empty()) {
            int bestChild = -1;
            double bestPuct = -std::numeric_limits<double>::max();
            for (int childID : nodes[nodeID].childIDs) {
                const double p = getPUCT(childID, nodes[nodeID].visitCount);
                if (p > bestPuct) {
                    bestPuct = p;
                    bestChild = childID;
                }
            }
            if (bestChild < 0) {
                break;
            }
            toApply.push_back(nodes[bestChild].step);
            nodeID = bestChild;
            path.push_back(nodeID);
        }
        double dummy = 0.0;
        for (const Step &s : toApply) {
            chess.moveForward(&s, dummy);
        }

        double leafValue = 0.0;
        bool haveLeafValue = false;

        /* 到达的节点本身可能已经终局 (走到这里时棋盘就是该节点的局面) */
        if (terminalValue(nodes[nodeID].currentColor, leafValue)) {
            haveLeafValue = true;
        }

        /* --- 2. EXPANSION --- */
        if (!haveLeafValue && !nodes[nodeID].untriedActionIndices.empty()) {
            std::size_t pickIdx = 0;
            for (std::size_t i = 1; i < nodes[nodeID].untriedPriors.size(); i++) {
                if (nodes[nodeID].untriedPriors[i] > nodes[nodeID].untriedPriors[pickIdx]) {
                    pickIdx = i;
                }
            }
            const int chosenAction = nodes[nodeID].untriedActionIndices[pickIdx];
            const Step chosenStep = nodes[nodeID].untriedSteps[pickIdx];
            const double chosenPrior = nodes[nodeID].untriedPriors[pickIdx];
            nodes[nodeID].untriedActionIndices.erase(
                nodes[nodeID].untriedActionIndices.begin() + (long)pickIdx);
            nodes[nodeID].untriedSteps.erase(
                nodes[nodeID].untriedSteps.begin() + (long)pickIdx);
            nodes[nodeID].untriedPriors.erase(
                nodes[nodeID].untriedPriors.begin() + (long)pickIdx);

            chess.moveForward(&chosenStep, dummy);

            const int nextColor = (nodes[nodeID].currentColor == Stone::COLOR_RED)
                                      ? Stone::COLOR_BLACK : Stone::COLOR_RED;

            std::vector<Step*> childSteps;
            std::vector<int> childIdx;
            getLegalActions(nextColor, childSteps, childIdx, mask);

            AZNode child(nodeID, chosenAction, chosenStep, chosenPrior, nextColor,
                         (int)childIdx.size());

            double v = 0.0;
            if (terminalValue(nextColor, v)) {
                /* 被将杀 / 困毙 / 和: 终局节点, 价值取自真实胜负 */
                child.isTerminal = true;
                leafValue = v;
            } else {
                encodeStateFor(nextColor, m_stateBuf);
                /*
                   ---- [2026-09] 叶子估值一律走**全量** (59e5233 的口径) ----
                   SACAZAgent 在这里还有一条"稀疏头"(只算合法列) 的快路径, 由那边的一个
                   开关选择。它是为 8100 动作空间 (对齐表示) 才加的, 而 59e5233 那时动作
                   空间是 128 槽, 全量 Q 只要算 128 列 —— 本类**没有那条路径**, 永远走
                   全量: 少一个变量, 也少一条与训练路径不同的代码 (两个口径之间任何一点
                   差异都会被放大成棋力差)。
                   (对齐表示 SACAZ_ALIGNED_REPR=1 —— 或本类的 SACAZ_LEGACY_MOE_EXPERT_SEQ=1,
                   它自带对齐表示 —— 下这条全量路径会慢, 那是刻意的: 本类的意义是"行为
                   还原", 不是"跑得快"。注意头张量宽度在那一档仍是 8100, 只是可达槽位被
                   哈希到 128。)
                */
                pi.zero();
                policy(m_stateBuf, mask, pi);
                qValues(m_stateBuf, qa, qb);
                for (std::size_t i = 0; i < childIdx.size(); i++) {
                    child.untriedActionIndices.push_back(childIdx[i]);
                    child.untriedSteps.push_back(*childSteps[i]);
                    child.untriedPriors.push_back((double)pi[childIdx[i]]);
                }
                leafValue = (double)softValueFrom(pi, mask, qa, qb);
                m_leafEvals++;
            }
            Steps::instance().put(childSteps);

            nodes.push_back(child);
            const int newID = (int)nodes.size() - 1;
            nodes[nodeID].childIDs.push_back(newID);
            nodeID = newID;
            path.push_back(newID);
            haveLeafValue = true;
        }

        /* --- 3. 已全展开的节点: 就地用软价值 (终局已在上面处理) --- */
        if (!haveLeafValue) {
            std::vector<Step*> ls;
            std::vector<int> li;
            getLegalActions(nodes[nodeID].currentColor, ls, li, mask);
            Steps::instance().put(ls);
            encodeStateFor(nodes[nodeID].currentColor, m_stateBuf);
            /* 全量叶子估值 (本类没有稀疏头开关, 见上面那一处的说明) */
            pi.zero();
            policy(m_stateBuf, mask, pi);
            qValues(m_stateBuf, qa, qb);
            leafValue = (double)softValueFrom(pi, mask, qa, qb);
            m_leafEvals++;
        }

        /* --- 4. BACKUP (negamax) --- */
        double v = leafValue;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            nodes[path[i]].visitCount++;
            nodes[path[i]].totalValue += v;
            v = -v;
        }

        /* --- 5. 撤销本轮试走 (path[0] 是根, 没有对应的 step) --- */
        for (std::size_t i = path.size(); i > 1; i--) {
            const Step &s = nodes[path[i - 1]].step;
            chess.moveBack(&s, dummy);
        }
    }

    if (piOut != nullptr) {
        visitDistribution(rootID, *piOut);
    }

    /* ---- 选择走法 ---- */
    int bestChildID = -1;
    if (temp <= 1e-6f) {
        int maxVisits = -1;
        for (int childID : nodes[rootID].childIDs) {
            if (nodes[childID].visitCount > maxVisits) {
                maxVisits = nodes[childID].visitCount;
                bestChildID = childID;
            }
        }
    } else {
        /* 按 N^(1/T) 采样 (自对弈用, 保证开局多样性) */
        RL::Tensor dist(ACTION_DIM, 1);
        dist.zero();
        double sum = 0.0;
        const double invT = 1.0 / (double)temp;
        for (int childID : nodes[rootID].childIDs) {
            const double w = std::pow((double)nodes[childID].visitCount, invT);
            dist[nodes[childID].parentAction] += (float)w;
            sum += w;
        }
        if (sum > 0.0) {
            const int a = RL::Random::categorical(dist);
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].parentAction == a) {
                    bestChildID = childID;
                    break;
                }
            }
        }
        if (bestChildID < 0) {
            int maxVisits = -1;
            for (int childID : nodes[rootID].childIDs) {
                if (nodes[childID].visitCount > maxVisits) {
                    maxVisits = nodes[childID].visitCount;
                    bestChildID = childID;
                }
            }
        }
    }

    if (bestChildID >= 0) {
        /*
           ---- [2026-09] 本类**没有**"从自己的搜索学一次"这条路径 ----
           SACAZAgent 那一支在这里还会用刚算出来的 π_MCTS 做一次 learnBatch (2026-09
           新加的 AlphaZero 在线监督信号, 由一个成员开关控制)。59e5233 **没有**这条路径:
           它对弈时只在 rollout (`exploreAndTrain`) 里学, 而那批样本 hasSearch=false。
           所以本类**连成员带函数**都没有 (那个开关、"把这一步存成样本"的函数都不存在)
           —— 不是"默认关掉", 是没有任何开关能打开。因此 `selectMove` 在本类里是
           **只读搜索**: 不训练、不往回放池写样本。
           (这也是 ops 上的差别: 工具/界面的"每手损失曲线"在本类里只由 rollout 与
           自对弈产生。)
        */
        return nodes[bestChildID].step;
    }
    /*
       根有合法走法却没选出一个孩子 (例如一次都没展开): 兜底取根节点未展开的第一手,
       不要返回 Step() —— 那会让调用方把"还有棋可下"读成"无棋可走"并判负
       (同 ABAgent 的"全负不返回走法", 见 abagent.cpp 的长注释)。
    */
    if (!nodes[rootID].untriedSteps.empty()) {
        if (piOut != nullptr) {
            /* π 目标交给**这一步**: 单点分布, 至少与真正走出的那一手一致 */
            piOut->zero();
            (*piOut)[nodes[rootID].untriedActionIndices.front()] = 1.0f;
        }
        return nodes[rootID].untriedSteps.front();
    }
    if (piOut != nullptr) {
        piOut->zero();
    }
    return Step();
}

void SACAZLegacyAgent::resetMoeBatchStats()
{
    if (auxLossCoef <= 0.0f) {
        return;
    }
    /*
       目标网 q1Target/q2Target 也要复位: 它们只前向、不训练, 门控统计永远用不到
       (辅助损失只注入在线网), 但 xSum/probSumBatch 是 float 累加器 —— 不复位的话
       会随一局的模拟次数一路涨上去, 精度慢慢烂掉。
    */
    RL::Net *nets[5] = {&actor, &q1, &q2, &q1Target, &q2Target};
    for (int ni = 0; ni < 5; ni++) {
        for (std::size_t li = 0; li < nets[ni]->size(); li++) {
            RL::ISparseMoE *moe = dynamic_cast<RL::ISparseMoE*>((*nets[ni])[li]);
            if (moe != nullptr) {
                moe->resetBatchStats();
            }
        }
    }
}

/* ============================================================
 *  learnBatch: 一次 mini-batch 的 SAC 更新 (critic / actor / α)
 * ============================================================ */
float SACAZLegacyAgent::learnBatch(int batchSize_, int epochs)
{
    if (batchSize_ < 1 || (int)memories.size() < batchSize_) {
        return 0.0f;
    }
    if (epochs <= 0) {
        epochs = replayEpochs > 0 ? replayEpochs : 1;
    }
    if (epochs < 1) {
        epochs = 1;
    }

    RL::Tensor state(STATE_DIM, 1);
    RL::Tensor nextState(STATE_DIM, 1);
    RL::Tensor mask(ACTION_DIM, 1);
    RL::Tensor nextMask(ACTION_DIM, 1);
    RL::Tensor pi(ACTION_DIM, 1);
    RL::Tensor piNext(ACTION_DIM, 1);
    RL::Tensor q1n(ACTION_DIM, 1);
    RL::Tensor q2n(ACTION_DIM, 1);
    RL::Tensor q1o(ACTION_DIM, 1);
    RL::Tensor q2o(ACTION_DIM, 1);
    RL::Tensor g(ACTION_DIM, 1);
    RL::Tensor dz(ACTION_DIM, 1);

    /*
       [P4] **每个 epoch 重新从池里抽** batchSize 条 (与 RL::PPO::learnFromReplay 同一
       做法), 累积梯度后优化器只在最后调一次 (P3)。

       为什么不是"把同一批复用 epochs 遍": 批内权重不变, 所以第 N 遍的梯度与第 1 遍
       **逐位相同**; 而 `RL::Net::RMSProp` 默认 clipGrad=true (`dw /= |dw|`) 会把这个
       纯倍数完全归一掉 —— 重复同一批对更新方向**毫无影响**, 只是白烧算力。
       每遍抽新样本才是真东西: 一次更新看到 batchSize×epochs 条经验 (等于把批放大
       epochs 倍, 但仍然只调一次优化器)。生成一条样本要走整棵 MCTS, 比抽一条贵得多,
       所以这个放大基本是白拿的。
    */
    std::uniform_int_distribution<int> pick(0, (int)memories.size() - 1);

    /*
       [MoE] 批统计的**边界**: 只反映本批的训练前向。
       搜索期间每次模拟都会跑一次策略/价值前向, 那些是"推理前向", 不该混进负载均衡
       辅助损失的批均值里 (见 resetMoeBatchStats 的说明)。
    */
    resetMoeBatchStats();

    /* 诊断读数按**本批**重置 (它们是"最近一次 learnBatch 的极值", 见头文件说明) */
    m_maxAbsTarget = 0.0;
    m_maxAbsTdErr = 0.0;

    float lossSum = 0.0f;
    int n = 0;
    float alphaGrad = 0.0f;
    const float a = alpha[0];

    for (int ep = 0; ep < epochs; ep++) {
    for (int it = 0; it < batchSize_; it++) {
        const Transition &tr = memories[(std::size_t)pick(RL::Random::engine)];
        expandSparse(tr.cells, state);
        writeContext(state, tr.ctx);
        expandSparse(tr.nextCells, nextState);
        writeContext(nextState, tr.nextCtx);
        bitsToMask(tr.curMask, mask);
        bitsToMask(tr.nextMask, nextMask);

        /* ---- 目标侧: 用目标网算下一局面的软价值 ---- */
        policy(nextState, nextMask, piNext);
        qTargetValues(nextState, q1n, q2n);
        const float vNext = softValueFrom(piNext, nextMask, q1n, q2n);

        /*
           符号: 所有价值都是"该局面走棋方视角"。s' 轮到**对手**走, 所以自举项要取负
           (negamax):   y = r − γ(1−done)·V(s')
        */
        const float y = tr.reward - gamma * (tr.done ? 0.0f : 1.0f) * vNext;
        /*
           ---- [2026-09] 目标**不夹**, 损失**纯 MSE** (59e5233 的口径) ----
           SACAZAgent 那一支在这里还有一对"值域约束": 把 y 夹到 ±2, 以及把 |err|>δ 的
           损失从平方改成线性 (Huber)。它们是 2026-09 为了压住实测到的 critic 发散
           (|Q| 0.063 -> 4.15 (40 局) -> 13.4 (150 局), 见
           docs/arena_sac_vs_ppo_report.md §5.2) 才加的, **59e5233 里没有**。
           本类两个成员都**不存在** (也没有第二条代码路径), 所以:
             * y 原样进目标 (发散是这一支的已知性质: 训练 20 局后 |Q| 均值 ~5.6,
               这正是"没有东西在抑制 critic"的可观测证据);
             * 损失是纯 MSE (`0.5·err²`), 梯度走 RL::Loss::MSE::df —— 两者逐位一致
               (那个 helper 给的就是 MSE 梯度)。
           用户口径: 这一支是"59e5233 行为还原版", 不许被后续的改进污染。要压发散请到
           SACAZAgent 那一支开它的两个约束, 不要在这里加。
        */
        m_maxAbsTarget = std::max(m_maxAbsTarget, std::fabs((double)y));

        /* ---- 当前局面 ---- */
        policy(state, mask, pi);
        qValues(state, q1o, q2o);

        /*
           critic 损失: 只对实际走的那一步回归 (SAC 的标准做法, 其余动作误差为 0),
           纯 MSE。
        */
        const float err = q1o[tr.action] - y;
        const double ae = std::fabs((double)err);
        m_maxAbsTdErr = std::max(m_maxAbsTdErr, ae);
        lossSum += (float)(0.5 * ae * ae);

        for (int ci = 0; ci < 2; ci++) {
            RL::Net &qnet = (ci == 0) ? q1 : q2;
            RL::Tensor target = (ci == 0) ? q1o : q2o;
            target[tr.action] = y;
            qnet.backward(state, RL::Loss::MSE::df((ci == 0) ? q1o : q2o, target));
        }

        /*
           策略损失 —— 两项都是"对 softmax 输出 π 的梯度", 相加后交给掩码 softmax
           的雅可比:
             (a) SAC 软 Q 项   dJ/dπ_a = α·(log π_a + 1) − min_i Q_i(s,a)
             (b) AlphaZero 监督项  CE(π_MCTS, π) 的梯度正好是 (π_a − π_MCTS,a);
                 用这个形式而不是 −π_MCTS/π, 因为后者在 π_a -> 0 时会爆掉。
        */
        for (int i = 0; i < ACTION_DIM; i++) {
            if (mask[i] <= 0.5f) {
                g[i] = 0.0f;   /* 非法动作不参与: π_i = 0, 绝不能把它抬起来 */
                continue;
            }
            float gi = a * (std::log(pi[i] + 1e-8f) + 1.0f)
                       - std::min(q1o[i], q2o[i]);
            if (tr.hasSearch) {
                gi += azWeight * (pi[i] - tr.pi[i]);
            }
            g[i] = gi;
        }
        maskedSoftmaxBackward(pi, g, dz);
        actor.backward(state, dz);

        /*
           α 自动调节:
             J(α) = α·(H − H̄)  =>  dJ/dα = H − H̄   (H = 策略熵, H̄ = 目标熵)
           Optimize::RMSProp 是梯度下降 (w -= lr·g/√v), 所以
             熵低于目标 -> g < 0 -> α 变大 -> 更探索;
             熵高于目标 -> g > 0 -> α 变小 -> 更利用。
        */
        float H = 0.0f;
        for (int i = 0; i < ACTION_DIM; i++) {
            if (pi[i] > 0.0f) {
                H -= pi[i] * std::log(pi[i]);
            }
        }
        const int lc0 = (tr.legalCount > 1) ? tr.legalCount : 2;
        /*
           ---- [2026-09] 目标熵的分母恒为**合法着法数** (59e5233 的口径) ----
           SACAZAgent 那一支还可以把分母换成"合法槽位数" (用来量 128 槽哈希碰撞对 α 的
           影响)。那是 2026-09 的实验开关, 59e5233 没有 —— 本类不含它, 所以这里只有一条
           路径, 原式 `H̄ = entropyRatio · log(合法着法数)`。
        */
        const int lc = lc0;
        const float Hbar = entropyRatio * std::log((float)lc);
        alphaGrad += (H - Hbar);
        n++;

        /*
           ---- [2026-09 ①] 训练中的 critic/α 诊断 (只累加, 不进任何梯度/更新路径) ----
           三个问题见 sacazlegacyagent.h 的 TrainDiag 说明。这里全部是**读**已经算好的量,
           不改变任何前向/反向/随机流 ⇒ 数值与改动前逐位一致。
           `D.clamped` (目标被值域约束夹住的条数) 在本类里**恒为 0**: 本类不夹目标。
           需要那个读数请到 SACAZAgent 那一支去量 —— 字段保留是为了两边的日志口径能直接
           对照。
        */
        {
            TrainDiag &D = trainDiag;
            D.n++;
            const double yAbs = std::fabs((double)y);
            D.yPreAbsSum += yAbs;
            D.yPreSum += (double)y;
            if (yAbs > D.yPreAbsMax) { D.yPreAbsMax = yAbs; }

            /* V(s') 的两项分解: E_π[min Q] 与熵项 α·H (y = r − γV) */
            double vQ = 0.0, vEnt = 0.0;
            for (int i = 0; i < ACTION_DIM; i++) {
                if (nextMask[i] <= 0.5f || piNext[i] <= 0.0f) { continue; }
                const double p = (double)piNext[i];
                vQ += p * (double)std::min(q1n[i], q2n[i]);
                vEnt += p * (-(double)a * std::log(p));
            }
            D.vNextSum += (double)vNext;
            D.vQSum += vQ;
            D.vEntSum += vEnt;

            /* 当前局面的合法槽位数 / 着法数, 以及 H 与两种 H̄ */
            double slotsCur = 0.0;
            double qm = 0.0, qs2 = 0.0;
            for (int i = 0; i < ACTION_DIM; i++) {
                if (mask[i] <= 0.5f) { continue; }
                slotsCur += 1.0;
                const double qq = (double)std::min(q1o[i], q2o[i]);
                qm += qq;
                qs2 += qq * qq;
            }
            double qSpread = 0.0;
            if (slotsCur > 0.0) {
                qm /= slotsCur;
                const double var = qs2 / slotsCur - qm * qm;
                qSpread = (var > 0.0) ? std::sqrt(var) : 0.0;
            }
            D.qSpreadSum += qSpread;
            D.qAbsMeanSum += std::fabs(qm);
            D.slotsSum += slotsCur;
            D.legalSum += (double)lc0;
            D.hSum += (double)H;
            D.hBarSum += (double)Hbar;
            const double hBarSlots = (double)entropyRatio
                                     * std::log((double)((slotsCur > 1.0) ? slotsCur : 2.0));
            D.hBarSlotsSum += hBarSlots;
            if ((double)H < (double)Hbar) { D.hBelowHbar++; }
            if ((double)H < hBarSlots) { D.hBelowHbarSlots++; }
            if (D.alphaFirst < 0.0) { D.alphaFirst = (double)a; }
            D.alphaLast = (double)a;
        }
    }
    }   /* ---- epochs 循环结束 (P4) ---- */

    if (n == 0) {
        return 0.0f;
    }

    /*
       本批的平均 critic 损失 (界面曲线用, 见 getLastTrainLoss)。
       取**批平均**而不是最后一条: 逐样本上报会让曲线变成低占空比的脉冲
       (重尾样本能差几个数量级), 而且与 DQN 报"平均平方 TD 误差"的口径对不上。
       P4 之后分母是 batchSize × epochs (每个 epoch 各抽 batchSize 条)。
    */
    m_lastLoss = lossSum / (float)n;
    m_lastBatchSamples = n;

    /*
       ---- 稀疏 MoE 的负载均衡辅助损失 ----
       放在优化器之前、主反向之后: 这个 mini-batch 里每个层的前向次数、被选中的
       专家次数、门控概率之和都已经累计好了, addAuxGradient 用这些统计算出
       "哪些专家被喂爆了", 把它们的 logit 压下去、把饿着的抬起来 (Switch
       Transformer 的 L_aux = E·Σ f_i·P_i 对 logits 的梯度)。
       没有这一项时, softmax 的反向会把没被选中的专家的概率继续压低, 路由会迅速
       坍缩到少数专家、其余永远不训练。细节与有限差分验证见 rl/sparse_moe.hpp
       和 test/test_sparse_moe_main.cpp [6][7]。
    */
    if (auxLossCoef > 0.0f) {
        RL::Net *nets[3] = {&actor, &q1, &q2};
        for (int ni = 0; ni < 3; ni++) {
            for (std::size_t li = 0; li < nets[ni]->size(); li++) {
                RL::ISparseMoE *moe = dynamic_cast<RL::ISparseMoE*>((*nets[ni])[li]);
                if (moe != nullptr) {
                    moe->addAuxGradient(auxLossCoef);
                }
            }
        }
    }

    /* ---- 应用梯度 ---- */
    actor.RMSProp(learningRateActor, 0.9f, 0.0f);
    q1.RMSProp(learningRateCritic, 0.9f, 0.0f);
    q2.RMSProp(learningRateCritic, 0.9f, 0.0f);

    /* α 的梯度是整批累加的, 取平均后再更新 */
    alpha.g[0] = alphaGrad / (float)n;
    alpha.RMSProp(learningRateAlpha, 0.9f, 0.0f);
    alpha.clamp(0.02f, 0.02f, 5.0f);

    /*
       ---- 目标网 Polyak 同步: 两个硬编码常数 (本类没有可调成员) ----
       59e5233 的值: Polyak 步长 tau = POLYAK_TAU = 1e-3, 每 TARGET_SYNC_EVERY = 64 次
       learn 同步一次 (一次会话 ~2600 步只把目标网挪动 2~4%, 所以 |Q_target| 一直停在
       随机初始化尺度 0.07~0.10 —— 这是**还原对象的一部分**, 不是待修的缺陷)。
       SACAZAgent 后来把这一对做成了两个可调成员 (F1 那一轮的实验), 本类刻意**不暴露**:
       从外面没有任何办法把这一支调到"硬拷贝 / 每 256 步" —— 那正是 2026-09 事故的形状
       (当时派生类没显式钉住这一对, 于是基类默认值一改, "行为还原版"跟着变了; 见
        docs/sac_critic_diagnosis_2026_09.md §13.5)。
    */
    learnSteps++;
    if (learnSteps % TARGET_SYNC_EVERY == 0) {
        q1.softUpdateTo(q1Target, POLYAK_TAU);
        q2.softUpdateTo(q2Target, POLYAK_TAU);
    }

    /* ---- 回放缓冲上限 ---- */
    while (memories.size() > maxMemorySize) {
        memories.pop_front();
    }

    return lossSum / (float)n;
}

/* ============================================================
 *  trainSelfPlay: 自对弈 (MCTS 访问分布做策略目标) + 回放训练
 * ============================================================ */
void SACAZLegacyAgent::trainSelfPlay(int episodes, int simulations_, int maxMoves,
                               bool verbose, float tempRoot, float tempFinal,
                               int learnEveryMoves)
{
    for (int ep = 0; ep < episodes; ep++) {
        chess.reset();
        int turn = Stone::COLOR_RED;
        int moves = 0;

        while (moves < maxMoves) {
            const int res = chess.getResult(turn);
            if (res != Chess::RESULT_ONGOING) {
                break;
            }

            /* 温度退火: 开局高 (多样性), 中后盘低 (质量) */
            const float frac = (float)moves / (float)(maxMoves > 1 ? maxMoves : 1);
            const float temp = tempRoot + (tempFinal - tempRoot) * frac;

            RL::Tensor piTarget(ACTION_DIM, 1);
            const Step s = selectMove(turn, simulations_, temp, &piTarget);
            if (!s.valid) {
                break;
            }

            Transition tr;
            encodeSparse(turn, tr.cells);
            contextOf(chess, turn, tr.ctx);
            for (int i = 0; i < ACTION_DIM; i++) {
                tr.pi[i] = piTarget[i];
            }
            tr.action = stepToActionIdx(s, turn);
            tr.hasSearch = true;
            tr.reward = computeReward(s, turn);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(turn, legal, idx, mask);
                tr.legalCount = (int)idx.size();
                maskToBits(mask, tr.curMask);
                Steps::instance().put(legal);
            }

            double dummy = 0.0;
            chess.moveForward(&s, dummy);

            const int nextTurn = (turn == Stone::COLOR_RED) ? Stone::COLOR_BLACK
                                                            : Stone::COLOR_RED;
            encodeSparse(nextTurn, tr.nextCells);
            {
                std::vector<Step*> legal;
                std::vector<int> idx;
                RL::Tensor mask(ACTION_DIM, 1);
                getLegalActions(nextTurn, legal, idx, mask);
                maskToBits(mask, tr.nextMask);
                Steps::instance().put(legal);
            }
            const int after = chess.getResult(nextTurn);
            tr.done = (after != Chess::RESULT_ONGOING);
            if (tr.done) {
                float rv = 0.0f;
                if (resultValue(after, turn, rv)) {
                    tr.reward = rv;   /* 终局奖励 ±1/0 覆盖即时奖励 */
                }
            }

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }

            turn = nextTurn;
            moves++;

            if (learnEveryMoves > 0 && (moves % learnEveryMoves) == 0) {
                learnBatch(batchSize);
            }
        }

        totalEpisodes++;
        const int res = chess.getResult(turn);
        if (res == Chess::RESULT_RED_WIN) {
            totalWins[0]++;
        } else if (res == Chess::RESULT_BLACK_WIN) {
            totalWins[1]++;
        }

        if (verbose) {
            std::printf("  [SAC+AZ] episode %d: %d 手, result=%d, 池=%zu, alpha=%.3f\n",
                        ep + 1, moves, res, memories.size(), (double)alpha[0]);
        }
    }
}

void SACAZLegacyAgent::warmupFromCurrent(int episodes, int simulations_, int maxMoves)
{
    /* 从当前局面继续 (不 reset) —— 调用方负责棋盘状态 */
    const int savedTurn = chess.sideToMove;
    trainSelfPlay(episodes, simulations_, maxMoves, false, 1.0f, 0.25f, 4);
    chess.sideToMove = savedTurn;
}

/* ============================================================
 *  exploreAndTrain: 走子前的"探索环境 + 在线训练一次"
 *
 *  与自对弈的区别: 这里**不做搜索**, 直接用掩码策略 π(·|s) 采样滚若干步 (最大熵
 *  策略本身就是探索策略), 把经验塞进回放缓冲后做一次 mini-batch 更新。
 *  这些样本的策略目标不是搜索出来的, 所以 hasSearch=false —— 只训练 critic 与
 *  SAC 的软 Q 项, 不用 AlphaZero 的监督项 (否则就是把策略往它自己身上拉)。
 * ============================================================ */
bool SACAZLegacyAgent::exploreAndTrain(int color, int rolloutSteps, const OpponentPolicy &opponent)
{
    if (rolloutSteps <= 0) {
        m_exploreInfo = "SAC+AZ: 探索步数为 0, 已跳过";
        return false;
    }
    /* 局部副本: rolloutFromCurrent 会把"真用了几手对手着法"回填到它里面 (P1) */
    OpponentPolicy opp = opponent;

    const int collected = rolloutFromCurrent(
        *this, chess, color, rolloutSteps,
        /* pick: 用掩码策略采样 */
        [this](const RL::Tensor &state, int turn) -> int {
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(turn, legal, idx, mask);
            if (legal.empty()) {
                Steps::instance().put(legal);
                return -1;
            }
            RL::Tensor pi(ACTION_DIM, 1);
            policy(state, mask, pi);
            /* 把这一步的合法数与掩码存下来, 给紧接着的 onTrans 复用 */
            m_pendingLegalCount = (int)idx.size();
            maskToBits(mask, m_pendingMask);
            Steps::instance().put(legal);
            return RL::Random::categorical(pi);
        },
        /* onTrans: 存一条 off-policy 经验 */
        [this](const Step &chosen, int actionIdx, const RL::Tensor &stateBefore,
               const RL::Tensor &nextState, float reward, bool done) {
            (void)chosen;
            Transition tr;
            denseToSparse(stateBefore, tr.cells);
            readContext(stateBefore, tr.ctx);
            denseToSparse(nextState, tr.nextCells);
            readContext(nextState, tr.nextCtx);
            tr.action = actionIdx;
            tr.legalCount = m_pendingLegalCount;
            tr.curMask[0] = m_pendingMask[0];
            tr.curMask[1] = m_pendingMask[1];
            tr.reward = reward;
            tr.done = done;
            tr.hasSearch = false;   /* 策略目标不来自搜索 */

            /* 下一局面的合法掩码: 此刻棋盘正好停在 nextState */
            std::vector<Step*> legal;
            std::vector<int> idx;
            RL::Tensor mask(ACTION_DIM, 1);
            getLegalActions(chess.sideToMove, legal, idx, mask);
            maskToBits(mask, tr.nextMask);
            Steps::instance().put(legal);

            memories.push_back(tr);
            while (memories.size() > maxMemorySize) {
                memories.pop_front();
            }
        },
        opp);

    /*
       在线训练一次。**批大小要按池里的实际条数夹一下**: learnBatch 在
       "池 < batchSize(默认 32)" 时是**故意**直接返回 0 的 (不拿半个 batch 去更新),
       于是刚开局时(池里只有探索刚收集的十几条)这一轮就白跑了 —— 界面上表现为
       "损失曲线一直是空的", 而且在线学习在开局阶段完全没发生。
    */
    const int onlineBatch = std::min(batchSize, (int)memories.size());
    const float loss = learnBatch(onlineBatch);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "SAC+AZ 探索 %d 步, 训练 1 次 (池 %zu, critic loss %.4f, alpha %.3f)%s",
                  collected, memories.size(), (double)loss, (double)alpha[0],
                  opponentRolloutInfo(opp).c_str());
    m_exploreInfo = buf;
    return collected > 0;
}

/* ============================================================
 *  存取: 一个路径前缀 -> 三个文件
 * ============================================================ */
bool SACAZLegacyAgent::saveModel(const std::string &filepath)
{
    actor.save(filepath + "_actor");
    q1.save(filepath + "_q1");
    q2.save(filepath + "_q2");
    return weightFileWritten(filepath + "_actor")
           && weightFileWritten(filepath + "_q1")
           && weightFileWritten(filepath + "_q2");
}

bool SACAZLegacyAgent::loadModel(const std::string &filepath)
{
    if (!weightFileReadable(filepath + "_actor")
        || !weightFileReadable(filepath + "_q1")
        || !weightFileReadable(filepath + "_q2")) {
        return false;
    }
    /*
       三个网络都**必须**检查载入结果 (见 ppomcts_agent.cpp 同一处修正的说明):
       原来只按"文件可读"就返回 true, 于是结构/CRC 不匹配的检查点会被静默忽略,
       训练循环表现成"每轮从随机权重重来却报告成功"。
       任何一个失败都直接返回 false —— 此时不碰 q1Target/q2Target, 避免把目标网
       拷成"载入失败后的混合状态"。
    */
    const int ra = actor.load(filepath + "_actor");
    const int r1 = q1.load(filepath + "_q1");
    const int r2 = q2.load(filepath + "_q2");
    if (ra != 0 || r1 != 0 || r2 != 0) {
        std::cerr << "[weights] SACAZLegacyAgent::loadModel 失败 (actor=" << ra
                  << ", q1=" << r1 << ", q2=" << r2 << "), 未同步目标网" << std::endl;
        return false;
    }
    q1.copyTo(q1Target);
    q2.copyTo(q2Target);
    return true;
}

/* ============================================================
 *  AgentBase
 * ============================================================ */
Step SACAZLegacyAgent::getBestMove(int color)
{
    return selectMove(color, simulations, 0.0f);
}

/* ============================================================
 *  selfCheckReport —— 界面"模型自检"面板的数据源
 *
 *  为什么这个函数必须存在, 而且必须长这样: 训练损失与自对弈胜率都**不能**回答
 *  "这个模型值不值得继续训" —— 后者里赢家和输家是同一份权重, 前者只说明网络与
 *  自己的目标一致 (critic 把自己估平了, MSE 也在降)。真正的前置判据落在表示层与
 *  终局口径上, 而这两类事实原来在界面里一个都看不到。本报告逐条给出它们, 判读写
 *  在同一行末尾 (面板的读者是看训练曲线的人, 不是读代码的人)。
 *
 *  具体到这个 agent, 有四条读数最值得看:
 *
 *   (1) **本实例是哪个骨干, 以及它的专家是什么**。这一个类同时被两个界面 agent 类型
 *       构造 — AGENT_SACAZ_OLD (Mlp) 与 AGENT_SACAZ_OLD_MOE (SparseMoeTb) —— 两块面板的
 *       数字会差很多 (参数量、有没有路由、每次前向 ms 级的专家开销)。所以第一行先报
 *       backboneName(backbone) 与它对应的界面类型, 否则"同名的两个 agent 读数不同"
 *       会被记到错的账上。
 *       [2026-10 第五轮] 专家本身也成了一个变量 (TB 12.9 M/个 vs 序列 Transformer
 *       91,731/个, 由 SACAZ_LEGACY_MOE_EXPERT_SEQ 选), 所以第 4 节还要多一行"专家类型"
 *       —— 那时"每次前向 ms 级的专家开销"是 TB 专家的读数, 序列专家下不适用。
 *
 *   (2) **规则上下文可观测 / 动作层有别名** —— 一好一坏, 都写在同一节里。
 *       好的一面: 14 个棋子平面之后跟着 3 个标量槽 (无吃子进度 / 重复
 *       次数 / 将军), 三次重复与自然限着因此进了 V(s); DQNMCTS 那种 90 维
 *       "每格一个子力值" 编码下这三个量逐字节不可观测 (probe_dqnmcts_aliasing [2]),
 *       于是同一局面的第 2、第 3 次出现会编码成同一个向量。坏的一面:
 *       `stepToActionIdx` 把 (棋子 id, 目标格) 哈希进 128 个槽位, 而真实走法空间是
 *       8100 个 (from,to) 对 —— 同一个局面里若干个互不相同的着法被迫共用同一个
 *       策略头/Q 槽位。
 *       代价是**梯度被平均**: 策略头在那一维上收到的是两个不同着法梯度的平均,
 *       而"着法 A 与着法 B 不同"这件事在 128 维里根本没有坐标 —— 所以这是**结构性
 *       上限, 训练多少次都消不掉**, 不是收敛慢。落子本身仍然正确 (走法按**节点
 *       访问数**取, 不按动作索引, 见 selectMove 的收尾), 受损的是策略目标的精度。
 *       要根治只能换无碰撞动作空间 (id*90 + x*9 + y = 2880), 见
 *       docs/agents_design.md §11。
 *       标准开局那一份是**确定性**的参照点 (与棋盘现状无关, 只要走法生成器不变就
 *       一直一样); 当前局面那一份是实时读数, 所以面板每手刷新时它会变。
 *
 *   (3) **终局用真实胜负, 不自举** —— 搜索走到终局节点时 `terminalValue` 直接给出
 *       ±1/0, 训练目标的自举项也被 (1-done) 截断。这是本 agent 与 PPOMCTS 的差别
 *       (那边终局也用网络值), 也是"一局结束时的价值目标值得信"的原因: 最后那几步
 *       回归的是真结果, 不是网络自己的估计。
 *
 *   (4) **MoE 使用直方图** —— 路由坍缩 (少数专家吃掉全部样本, 其余永远拿不到梯度)
 *       在损失曲线上完全看不出来, 只能读这个直方图: 只要有一个专家的计数是 0 而
 *       别的不是, 那一支就已经死了。
 *
 *  **只读 / 可重复 / 不动棋盘** (aiagent.h 的契约): 需要在棋盘上数别名的两处都用
 *  `Chess probe(chess)` 的**副本**并局部解码 —— 本函数会在对局中途由 GUI 线程调用,
 *  而那一刻搜索线程可能正拿着 `this->chess` 走子/回退, 碰它就是与搜索抢棋盘。
 *  不改任何成员 (函数是 const), 不跑搜索/前向/权重 IO, 也不复位 MoE 计数
 *  (那是写操作; 直方图按**全生命周期**累计正好是坍缩诊断要的口径)。
 *
 *  刻意**不**报棋力: 棋力只有带置信区间的锚点对局 (bench_anchor, Elo 差 + 95% 区间)
 *  能回答 —— 见最后一行。
 * ============================================================ */
namespace {

/*
 *  一个局面上的动作别名 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了几个
 *  互不相同的着法)。
 *
 *  为什么是**自由函数**而不是成员: 它只做统计, 不碰棋盘、不碰网络, 唯一要用的是
/*
 *  一个局面上的动作**别名**读数 (合法着法数 -> 用到的槽位数; worstSlot = 最挤槽位背了
 *  几个互不相同的着法)。
 *
 *  2026-09 换成 8100 双射之后这个函数的期望值是**恒等式**:
 *      slotCount == legalCount  且  worstSlot == 1
 *  它不再是"缺陷读数"而是一个**回归指示器**: 只要 slotCount < legalCount, 就说明
 *  动作索引又变成了会碰撞的写法 (或者规范镜像写错导致两个着法映到同一槽)。
 *  这正是本仓库的惯例 —— 把"本该成立的不变量"变成可见的断言, 而不是删掉读数。
 *
 *  为什么是**自由函数**: 它只做统计, 不碰棋盘、不碰网络, 唯一要用的是
 *  `stepToActionIdx` 那一份公式 —— 签名只要 `const SACAZLegacyAgent &` + color 就够。
 *  这样一来**面板数的一定是训练用的同一个公式** (DQNMCTS 那边只能把哈希再抄一份
 *  并靠注释提醒"改一处必须改两处"; 这里换成编译期耦合: 谁把那个 const 去掉, 这里
 *  当场编不过)。
 *
 *  去重: 每个槽位里存 (棋子 id, 目标格) 的集合, 统计的是**互不相同的走法**数,
 *  而不是合法着法列表的条数 —— 列表里若有重复项, 那也不该被算成"两个走法挤在一起"。
 *
 *  只读: 只看传进来的走法列表 (调用方负责 `Steps::instance().put()` 归还)。
 */
void aliasOfPosition(const SACAZLegacyAgent &ag, const std::vector<Step *> &legal, int color,
                     int &legalCount, int &slotCount, int &worstSlot)
{
    std::map<int, std::set<long long> > bucket;
    for (std::size_t i = 0; i < legal.size(); i++) {
        const Step &s = *legal[i];
        const long long key = ((long long)s.id << 16)
                            | ((long long)s.nextPos.x << 8)
                            | (long long)s.nextPos.y;
        bucket[ag.stepToActionIdx(s, color)].insert(key);
    }
    legalCount = (int)legal.size();
    slotCount = (int)bucket.size();
    worstSlot = 0;
    for (std::map<int, std::set<long long> >::const_iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        if ((int)it->second.size() > worstSlot) {
            worstSlot = (int)it->second.size();
        }
    }
}

} // namespace

std::string SACAZLegacyAgent::selfCheckReport() const
{
    char buf[512];
    std::string out;

    /*
       ---- 0. 本实例是**哪一支** (必须放最前面) ----
       本类服务**两个**界面类型 (AGENT_SACAZ_OLD = MLP 骨干 / AGENT_SACAZ_OLD_MOE =
       稀疏 MoE 骨干, 专家是 TB 或序列 Transformer —— 由 SACAZ_LEGACY_MOE_EXPERT_SEQ 选),
       而它们与 AGENT_SACAZ / AGENT_SACAZ_MOE 在面板上的读数长得很像 (同一表示、同样的口径、
       参数量也接近) —— 不把"界面类型 + 骨干"写在第一行, 读者会把四支的读数记到同一本账上。
       下一行这两句**都是开关感知的** (guiAgentLabel / backboneName 各自分支), 所以
       "专家是什么"和"是哪一支"一起印在第一行, 不需要读者自己去对宏。
       第二段是**本类最要紧的一句**: 它的口径全部硬编码, 而且**刻意保留**了 59e5233 的
       两个"看着像坏掉"的读数 (目标网移动率只有 2~4%、|Q| 会漂到 5.6)。不写出来,
       看面板的人第一反应会是"这个模型坏了", 然后去"修"它 —— 那正是这一支最不希望
       发生的改动。
    */
    std::snprintf(buf, sizeof(buf),
                  "界面 agent 类型 %s | 骨干 %s\n", guiAgentLabel(), backboneName(backbone));
    out += buf;
    out += "【59e5233 行为还原版 = 独立类 SACAZLegacyAgent】与 SACAZAgent 是**两份独立实现**"
           "(不继承), 所以那边的默认值/行为改动渗不进这里 (文件末尾有 static_assert 钉住);\n";
    out += "  本类口径全部硬编码: 即时奖励=材质x0.1+每步代价 (无塑形) / 终局=引擎真值 ±1/0"
           " (无塑形) / critic 目标不夹 + 纯 MSE / 叶子估值=全量 / 不从自己的搜索学 /\n";
    out += "  目标网 tau=1e-3 每 64 次 learn (编译期常数) —— 于是下面两条\"看起来像缺陷\"的读数"
           "是本类**刻意保留**的还原对象: 目标网移动率 ~4%、|Q| 会漂到 5.6;\n";
    out += "  网络结构与 59e5233 逐字相同 (隐层激活 Layer<Tanh>, 没有开关);"
           " \"用 TanhNorm<Linear> r=1 复现那一层\"已实测证伪 (偏置在 tanh 外, max|dQ| = 8.9e-06);\n";
    /* 前缀按**本实例的骨干**取 (本类现在有两个骨干, 两个前缀), 见 defaultWeightPrefix */
    out += "  权重文件独立: " + std::string(SACAZLegacyAgent::defaultWeightPrefix(backbone))
           + "_actor / _q1 / _q2 (与 AGENT_SACAZ 的 weights/sacaz_agent 不共用,"
             " 否则两边会互相覆盖/串权重)\n";
    out += "--------------------------------------------------------------\n";
    /*
       隐层激活报**实测值** (读 actor 第 2 层的类型), 不是回显开关 —— 理由见
       hiddenActivationName() 的注释 (开关曾因"建网之后才赋值"而静默失效)。
       权重文件名不在这里印: 面板顶端那两行由 weightFilesOf() 给出**真实文件名**,
       比在这里复述一遍前缀更可靠 (前缀与实际写出的文件名曾经漂移过一次)。
    */
    std::snprintf(buf, sizeof(buf), "隐层激活(实测) %s\n", hiddenActivationName());
    out += buf;

    /* ---- 1. 表示层: 状态编码 ----
       这一节的两行**必须随 SACAZ_ALIGNED_REPR 分支**: 默认构建是 1263 维 / 128 槽,
       而面板原来写死了"与 PPOMCTS 同口径 / 8100 双射" —— 那是一句在默认构建下
       不成立的话, 面板说谎比不说更坏 (用户口径: SAC 回退 1263/128, PPO 才用 8100)。
       [2026-10 第五轮] 而"动作层同口径"这半句还得再看**动作编码**选的是哪一套 (见下面
       hashAction): 对齐表示 + 128 槽哈希这一档 (序列专家开关的默认口径) 状态是对齐的、
       动作层**不是** —— 照旧念"8100 双射"就是让面板说谎。 */
    /*
       **动作编码的实际口径** = 与 `stepToActionIdx` **同一个条件**
       (`!ALIGNED_REPR || legacyHashAction`)。为什么在这里算一次而不是各处重写:
       第 1 节的"与 PPOMCTS 同口径"、第 2 节的动作公式与别名判据, 三处说的必须是同一件事。
       为什么不能只按 ALIGNED_REPR 判: 对齐表示 + 128 槽消融那一档会把**自己的正常行为**
       报成"**回归!**" (假警报), 而序列专家开关那一支的默认值正是 legacyHashAction=true
       ⇒ 那个假警报会直接变成默认读数。
       ⚠ 默认构建下 legacyHashAction 恒为 false, 所以 `hashAction == !ALIGNED_REPR` ——
       下面所有分支在开关关闭时的输出与改动前**逐字一致**。
    */
    const bool hashAction = (!ALIGNED_REPR || legacyHashAction);
    if (ALIGNED_REPR) {
        std::snprintf(buf, sizeof(buf),
                      "状态 %d 维 = %d 平面 x %d 格 = %d 棋子平面 (CTX_BASE=%d)"
                      " + 5 上下文平面 (%d/%d/%d/%d/%d)\n",
                      STATE_DIM, PLANES, CELLS, PIECE_PLANES, CTX_BASE,
                      PLANE_MATERIAL, PLANE_TEMPO, PLANE_HALFMOVE, PLANE_REPEAT,
                      PLANE_CHECK);
        out += buf;
        if (hashAction) {
            std::snprintf(buf, sizeof(buf),
                          "**状态与 PPOMCTSAgent 逐位同口径** (2026-09 对齐): 同一个"
                          " STATE_DIM=%d、同一份 chessstate.h 的 CTX_* 顺序、同一个规范"
                          " 镜像; **动作层不同** —— 本支按用户口径保留 %d 槽哈希"
                          " (legacyHashAction), 所以两个 agent 的差别是算法 + 动作编码\n",
                          STATE_DIM, LEGACY_ACTION_DIM);
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "**与 PPOMCTSAgent 逐位同口径** (2026-09 对齐): 同一个 STATE_DIM=%d、"
                          "同一份 chessstate.h 的 CTX_* 顺序、同一个规范镜像、同一个 8100 双射"
                          "动作空间 -> 两个 agent 的差别只剩算法本身\n",
                          STATE_DIM);
            out += buf;
        }
        std::snprintf(buf, sizeof(buf),
                      "规则/阶段上下文: 5 个 (子力阶段 / 总手数 / 无吃子进度 / 重复次数 / 被将)"
                      " -> 三次重复 / 自然限着 / 将军都是 V(s) 的输入 (可观测)\n");
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf),
                      "状态 %d 维 = %d 棋子平面 x %d 格 (%d) + **3 个规则上下文标量**"
                      " (无吃子进度 / 重复次数 / 被将)\n",
                      STATE_DIM, PIECE_PLANES, CELLS, CTX_BASE);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "**改前表示 (SACAZ_ALIGNED_REPR 未定义, 默认)**: 与 PPO+MCTS 的 "
                      "1710 维/8100 双射**不是同一口径** —— 用户口径 (2026-09) 是"
                      " SAC 回退 1263/128、PPO 保持 8100。跨 agent 状态对齐按口径撤销;"
                      " 对齐表示仍可编译进来 (-DSACAZ_ALIGNED_REPR=1)\n");
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "规则/阶段上下文: 3 个标量 (子力阶段与总手数在改前表示里**没有**"
                      "通道, 那是后来对齐时才加的)\n");
        out += buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "  对照: PGE / DQN / DQNMCTS 用的是 90 维每格一个子力值, 规则上下文"
                  "通道 0 个 (同一局面的第 2/第 3 次出现在那里逐字节不可分)\n");
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "视角: 规范视角 (轮到黑方时 x->9-x, 己方永远在 x 大的一侧) -> 红黑共用"
                  "一套权重, 价值函数不必从编码里反推该谁走\n");
    out += buf;

    /* ---- 2. 动作层: 双射 / 哈希 + **不变量**读数 (标准开局 / 当前局面) ----
       判据是**实际用的是哪一套** (hashAction, 与 stepToActionIdx 同条件), 不是"表示对没对齐":
       对齐 + 双射下"别名"应当恒为 0 (回归指示器); 用哈希时它是**结构性上限**
       (两个不同着法共用一槽 -> 策略头收到两者梯度的平均), 见 docs 的 13.83 vs 0.43。 */
    if (!hashAction) {
        std::snprintf(buf, sizeof(buf),
                      "动作 %d = 双射 canonicalCell(from)*%d + canonicalCell(to)"
                      " (与 PPOMCTS 同一公式, 无碰撞; 曾是 128 槽哈希)\n",
                      ACTION_DIM, CELLS);
        out += buf;
    } else {
        if (!ALIGNED_REPR) {
            /* 默认构建: 这一行与改动前**逐字相同** */
            std::snprintf(buf, sizeof(buf),
                          "动作 %d = **%d 槽哈希别名空间** (改动前表示; 真实着法空间是 %d 个"
                          " (from,to) 对 -> 别名是策略精度的结构性上限, 不是训练量的问题)\n",
                          ACTION_DIM, ACTION_DIM, ALIGNED_ACTION_DIM);
            out += buf;
        } else {
            /*
               对齐表示 + 保留哈希 (序列专家开关的默认口径)。这一档**必须**把"头张量宽度"与
               "可达槽位数"分开说: 两个数都印出来才对得上账, 而只印 ACTION_DIM = 8100 会让
               人以为动作空间是 8100 路 (那正是这一支刻意避开的东西)。
            */
            std::snprintf(buf, sizeof(buf),
                          "动作张量 %d 列 (对齐表示), 但**可达槽位只有 %d 个**"
                          " (legacyHashAction: 用户口径 —— 序列专家这一支保持 128 槽哈希)"
                          " | 真实着法空间是 %d 个 (from,to) 对 -> 别名是策略精度的"
                          "结构性上限, 不是训练量的问题\n",
                          ACTION_DIM, LEGACY_ACTION_DIM, ALIGNED_ACTION_DIM);
            out += buf;
        }
    }

    int initLegal = 0, initSlots = 0, initWorst = 0;
    int curLegal = 0, curSlots = 0, curWorst = 0;
    {
        /* 只看副本: 绝不能在 GUI 线程碰 this->chess (搜索可能正在用它) */
        Chess probe(chess);
        probe.reset();
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(*this, legal, probe.sideToMove, initLegal, initSlots, initWorst);
        Steps::instance().put(legal);
    }
    {
        /* 当前局面的同一份读数 (副本保留现状, 不做 reset) */
        Chess probe(chess);
        std::vector<Step *> legal;
        probe.sample(probe.sideToMove, legal);
        aliasOfPosition(*this, legal, probe.sideToMove, curLegal, curSlots, curWorst);
        Steps::instance().put(legal);
    }
    /*
       判据字符串随**动作编码**变 (hashAction): 用双射时 slotCount==legalCount 是**恒等式**,
       不成立就是索引公式/规范镜像写错 (回归); 用哈希时挤掉几个是**正常现象**, 只看数值。
    */
    const char *badInit = hashAction
                              ? "[结构性上限读数, 不是回归]"
                              : ((initLegal == initSlots && initWorst <= 1) ? "[不变量成立]"
                                                                           : "**回归!**");
    const char *badCur = hashAction
                             ? "[结构性上限读数, 不是回归]"
                             : ((curLegal == curSlots && curWorst <= 1) ? "[不变量成立]"
                                                                       : "**回归!**");
    std::snprintf(buf, sizeof(buf),
                  "动作别名(标准开局): %d 个合法着法 -> %d 个槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法) %s\n",
                  initLegal, initSlots, initLegal - initSlots, initWorst, badInit);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "动作别名(当前局面): %d 个合法着法 -> %d 个槽位, 挤掉 %d 个"
                  " (最挤槽位 %d 个着法) %s\n",
                  curLegal, curSlots, curLegal - curSlots, curWorst, badCur);
    out += buf;
    if (!hashAction) {
        out += "  为什么这项现在是回归指示器: 128 槽哈希时代它量的是**结构性上限**"
               "(两个不同着法共用一列 -> 策略头收到两者梯度的平均); 换成双射后"
               " slotCount==legalCount 是恒等式, 一旦不成立就是索引公式或规范镜像写错了\n";
    } else if (!ALIGNED_REPR) {
        out += "  改前表示下这一项**不是缺陷**: 槽位比合法着法多时可能无碰撞, 着法变多时"
               "必然开始挤 —— 它是策略精度的上限读数 (上面那两行就是当前局面的实际值)\n";
    } else {
        out += "  对齐表示 + 128 槽哈希 (用户口径: 序列专家这一支保持 128 槽): 挤掉槽位是"
               "**预期行为**, 不是回归; 换成双射之后这一项才会变成恒等式 (那时 slotCount"
               " < legalCount 才是写错了)\n";
    }

    /* ---- 3. 算法 / 口径 ---- */
    /*
       [F1] 这里打印的是**两个编译期常数** (POLYAK_TAU / TARGET_SYNC_EVERY), 不是回显
       某个开关: 本类没有"可调的目标网同步率"成员 (那是 SACAZAgent 那一支的实验旋钮),
       所以"面板显示的值"与"实际生效的值"在结构上就不可能不一致 —— 这直接堵住了
       2026-09 那次事故的形状 (回显开关的检查永远通过)。
       后面那行把"一次会话 (~2600 次 learn) 能移动多少"算出来: 低于 50% 就等于自举项里
       没有游戏信息 —— 老口径 (本类) 只有 2~4%, 而 |Q_target| ≈ 0.08 就是它的读数。
       这不是缺陷, 是**还原对象的一部分**。
    */
    const double kSessionLearnSteps = 2600.0;   /* 20 局 x ~130 步的典型 learn 次数 */
    const double kIter = (double)TARGET_SYNC_EVERY;
    const double perIter = 1.0 - std::pow(1.0 - (double)POLYAK_TAU, 1.0 / kIter);
    const double moved = 1.0 - std::pow(1.0 - perIter, kSessionLearnSteps);
    std::snprintf(buf, sizeof(buf),
                  "双 critic q1/q2 + 目标网 q1Target/q2Target | 每 %d 次 learn 做一次 Polyak"
                  " 同步 (tau=%.4f, **硬编码常数** 59e5233 口径) |"
                  " 叶子价值 = min_i Q_i - alpha*log pi (最大熵软价值)\n",
                  (int)TARGET_SYNC_EVERY, (double)POLYAK_TAU);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "目标网移动率: 2600 次 learn (~20 局) 后相对随机初始化 %.1f%% %s\n",
                  moved * 100.0,
                  moved < 0.5 ? "**< 50%: 自举项 V(s') 几乎还是随机网, TD 目标里没有游戏信息 ([F1])**"
                              : "(跟得上在线网)");
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "alpha=%.3f (自动调节, 界 [0.02, 5]) | 目标熵 %.2f x log(合法着法数) |"
                  " azWeight=%.2f | c_puct=%.2f | 模拟次数=%d | gamma=%.2f\n",
                  (double)getAlpha(), (double)entropyRatio, (double)azWeight,
                  (double)c_puct, simulations, (double)gamma);
    out += buf;
    /*
       **口径行的价值**: AGENT_SACAZ 与 AGENT_SACAZ_OLD 是**两个没有继承关系的类**
       (见头文件), 差别就是下面这一行。不印出来, "两个 SAC 谁强"就没法归因。
       本类的这一行全部是**常数**: 没有值域约束、没有塑形、没有稀疏头、不从搜索学。
    */
    /*
       [2026-10 第五轮] 这一处的三目原本是**反的** (`legacyHashAction ? "对齐双射 8100"
       : "128 槽哈希"`) —— 默认构建下歪打正着 (两个宏都不给时 legacyHashAction=false,
       印出来的恰好是"128 槽哈希"), 但对齐表示下就变成了"实际用双射、面板说哈希";
       序列专家开关那一支 (对齐 + 保留 128 槽) 更会印成"对齐双射 8100", 与真实动作空间相反。
       现在按**实际编码**判 (hashAction, 与 stepToActionIdx 同一个条件):
         * 默认构建 → 仍是 "128 槽哈希", 与改动前**一字不差**;
         * 对齐 + 双射 → "对齐双射 8100" (顺手把那条假读数修对);
         * 序列专家那一支 → "128 槽哈希 (对齐表示下按用户口径保留)"。
    */
    std::snprintf(buf, sizeof(buf),
                  "口径(全是硬编码, 本类没有对应成员): critic 目标**不夹** + 纯 MSE |"
                  " 叶子估值=全量(每片叶子算全量 Q) | 即时奖励=材质x0.1+每步代价 (无塑形) |"
                  " 终局=引擎真值 ±1/0 (无塑形) | 不从自己的搜索学 | 动作空间=%s\n",
                  hashAction ? (ALIGNED_REPR ? "128 槽哈希 (对齐表示下按用户口径保留)"
                                             : "128 槽哈希")
                             : "对齐双射 8100");
    out += buf;
    out += "  本类刻意不含奖励塑形、不含任何 critic 值域约束 (目标钳位 / Huber 分段 /"
           " 搜索叶子缩放 / 熵项开关也都没有) —— 59e5233 就是这样; 加了就说明被污染了\n";
    out += "  与 AGENT_SACAZ 的关系是**两份独立实现**(不继承): 那边的任何默认值/行为改动"
           "都渗不进来, 代价是共享算法上的修复要**刻意**决定要不要同步过来\n";
    std::snprintf(buf, sizeof(buf),
                  " (-log pi >= 0), 所以 alpha 越大, 选择多的局面估值越高\n",
                  (double)learningRateActor, (double)learningRateCritic,
                  (double)learningRateAlpha);
    out += buf;
    out += "终局: 搜索到终局节点取**真实胜负** (terminalValue 给 ±1/0), 不自举; 训练目标"
           "的自举项也被 (1-done) 截断 -> 一局最后几步回归的是真结果而不是网络自己的"
           "估计, 这是它局末价值目标可信的原因 (PPOMCTS 那边终局仍用网络值)\n";

    /* ---- 4. 骨干 / MoE ---- */
#if defined(SACAZ_LEGACY_MOE_EXPERT_SEQ) && SACAZ_LEGACY_MOE_EXPERT_SEQ
    /*
       [2026-10 第五轮] 关键词变了: `expertHidden` 在**这一支**下不是"只有 MLP 专家骨干用"
       的隐层宽度, 而是序列专家的 **tokDim** (token 宽度, 必须能被 MOE_SEQ_HEADS 整除)。
       所以这一行必须跟着开关分支 —— 照默认那一句印出来就是让面板说谎, 而"专家到底是什么"
       恰好是本支唯一那个实验变量。
       下面那行"专家类型"是**读实现** (编译期常量 + 构造时实际用的宽度 + 位置编码旋钮),
       不是回显某个 bool: 本工程记录过"开关设了却没生效"的事故, 参考实现
       (sacazmoetbagent.cpp) 把同一条读数当作护栏, 本类照抄同一个口径。
    */
    std::snprintf(buf, sizeof(buf),
                  "骨干结构: hiddenDim=%d | 序列专家 tokDim=%d (expertHidden; 必须能被"
                  " MOE_SEQ_HEADS=%d 整除) | auxLossCoef=%.3f\n",
                  hiddenDim, expertHidden, MOE_SEQ_HEADS, (double)auxLossCoef);
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "专家类型: **序列 Transformer** (SeqTransformerExpert<%d,%d,%d>, tokDim=%d)"
                  " | 位置编码: %s | 输入 %d 维 = %d 平面 x %d 格 (与 PPO 支同一份实现)"
                  " | 权重前缀: %s\n",
                  CELLS, MOE_SEQ_HEADS, MOE_SEQ_BLOCKS, expertHidden, moeSeqPosMode(),
                  STATE_DIM, PLANES, CELLS, defaultWeightPrefix(backbone));
    out += buf;
    out += "  判读: 专家从 TransformerBlock (12.9 M 参数/个) 换成序列专家 (91,731 参数/个)"
           " ⇒ MOE_TB_HEADS / MOE_TB_DFF 在这一支下**不生效**;\n";
    out += "  位置编码模式由 moeSeqPosMode() 给出 (建网时设不上会当场 abort), 所以这一行"
           "报的是**实际生效**的口径\n";
#else
    std::snprintf(buf, sizeof(buf),
                  "骨干结构: hiddenDim=%d | 专家隐层 expertHidden=%d (只有 MLP 专家骨干用)"
                  " | auxLossCoef=%.3f\n",
                  hiddenDim, expertHidden, (double)auxLossCoef);
    out += buf;
#endif
    const int experts = moeExpertCount();
    if (experts <= 0) {
        std::snprintf(buf, sizeof(buf),
                      "MoE: 无稀疏层 (本骨干是纯 MLP) -> 没有路由, 不存在专家坍缩\n");
        out += buf;
    } else {
        const int topK = moeTopK();
        std::vector<long long> usage;
        /* moeExpertCount / moeTopK / moeUsage 内部对 actor 做了 const_cast —— 只是因为
           `Net::operator[]` 没有 const 重载, 用到的 `usageSnapshot()` 本身是 const 且
           只读全生命周期计数, 所以这里没有写操作、也不复位 (复位是训练路径的事)。 */
        moeUsage(usage);
        std::snprintf(buf, sizeof(buf),
                      "MoE: 专家 %d 个, topK=%d (topK=1 是 Switch 式硬路由; 稠密对照骨干"
                      "取 topK=专家数, 同参数不同算力)\n",
                      experts, topK);
        out += buf;

        std::string hist;
        long long total = 0, mx = 0, mn = -1;
        for (std::size_t i = 0; i < usage.size(); i++) {
            hist += (i == 0) ? "" : "/";
            hist += std::to_string(usage[i]);
            total += usage[i];
            if (usage[i] > mx) {
                mx = usage[i];
            }
            if (mn < 0 || usage[i] < mn) {
                mn = usage[i];
            }
        }
        if (mn < 0) {
            mn = 0;
        }
        /* 直方图是全生命周期累计 (本函数不复位它, 只读): 一次前向都没发生过时全 0 */
        std::snprintf(buf, sizeof(buf),
                      "MoE 使用计数 (actor 第一个稀疏层, 全生命周期): %s\n", hist.c_str());
        out += buf;
        if (total <= 0) {
            out += "  全 0 = 还没有发生过前向 (没搜过也没训过), 现在读不出坍缩信息\n";
        } else {
            const double mean = (double)total / (double)experts;
            const double ratio = (mean > 0.0) ? (double)mx / mean : 0.0;
            std::snprintf(buf, sizeof(buf),
                          "  最挤 %lld (%.0f%%), 均值 %.1f, 最挤/均值 = %.2f",
                          mx, 100.0 * (double)mx / (double)total, mean, ratio);
            out += buf;
            if (mn == 0) {
                out += " -> **有专家一次都没被选中 = 路由已坍缩** (它拿不到任何梯度,"
                       " 负载均衡辅助损失没兜住)\n";
            } else if (ratio >= 2.0) {
                out += " -> 偏斜 (最挤是均值的 2 倍以上), 路由在收缩, 盯住下一轮\n";
            } else {
                out += " -> 均衡 (无专家为 0, 最挤/均值 < 2), 未观察到路由坍缩\n";
            }
        }
    }

    /* 参数量: Net::paramCount() 各层求和 (MoE 层与它的**专家** —— TransformerBlock 或
       序列 Transformer —— 都实现了它, 所以这里报的是真实总数, 不是"只算 Linear" 的旧
       口径)。目标网只前向, 单独列出来。
       读数本身随专家自动变: 序列专家那一支下一个专家 ~92 K, 而不是 TB 的 12.9 M。 */
    std::snprintf(buf, sizeof(buf),
                  "参数量: actor %lld | q1 %lld | q2 %lld | 目标网合计 %lld (只前向) |"
                  " alpha 1\n",
                  actor.paramCount(), q1.paramCount(), q2.paramCount(),
                  q1Target.paramCount() + q2Target.paramCount());
    out += buf;

    /* ---- 5. 训练进度 ---- */
    const int epochs = (replayEpochs > 0) ? replayEpochs : 1;
    std::snprintf(buf, sizeof(buf),
                  "训练进度: learnSteps=%d | 自对弈局数=%d | 回放池 %zu/%zu | 每次更新"
                  " %d 条 x %d epochs (上次实际攒 %d 条) | 叶子评估 %lld 次\n",
                  learnSteps, totalEpisodes, memories.size(), maxMemorySize,
                  batchSize, epochs, getLastBatchSamples(), getLeafEvals());
    out += buf;
    out += "  叶子评估 = 软价值前向次数 (每次模拟一次, 是**算力**读数, 不是搜索质量)"
           " —— 它随模拟次数线性长, 不代表搜得更准\n";
    if (getLastBatchSamples() > 0 && getLastBatchSamples() != batchSize * epochs) {
        std::snprintf(buf, sizeof(buf),
                      "  本批样本数 != batchSize x epochs -> 多 epoch 复用可能没生效"
                      " (每遍应当**重新抽** batchSize 条, 见 replayEpochs 的说明)\n");
        out += buf;
    }

    const float loss = getLastTrainLoss();
    if (std::isfinite(loss)) {
        std::snprintf(buf, sizeof(buf),
                      "最近一次 learnBatch 的 critic MSE: %.6g (批平均) —— 它衡量 critic"
                      " 与自己的目标差多少, **不是棋力**\n", (double)loss);
        out += buf;
    } else {
        std::snprintf(buf, sizeof(buf),
                      "最近一次 learnBatch 的 critic MSE: 未上报 (NaN, 还没学过或池 <"
                      " batchSize; 曲线控件会丢弃非有限值)\n");
        out += buf;
    }

    out += "以上是表示/口径事实, **不是棋力**; 棋力请用 bench_anchor 的锚点对局"
           " (带 95% 置信区间的 Elo 差) 回答\n";
    return out;
}
