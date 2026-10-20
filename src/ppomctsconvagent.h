#ifndef PPOMCTS_CONV_AGENT_H
#define PPOMCTS_CONV_AGENT_H

/*
 * ============================================================================
 *  ppomctsconvagent.h - PPO + MCTS + **ConvResNet 主干 (无 MoE, 加深)**
 * ============================================================================
 *
 * [2026-10-17 用户口径]
 *   用户: "使用单独类实现 PPO + MCTS + ConvResNet agent, 不使用 MOE, ConvResNet 加深网络,
 *          代替 chess 程序现有 ConvResNetExpert 对应的 agent"。
 *
 * 为什么要一个**单独的类** (而不是给 `PPOMCTSAgent` 加一个构造函数参数):
 *   1. **权重前缀必须分叉**: 无 MoE 的主干参数量/张量布局与 `SparseMoE<ConvResNetExpert,4,1>`
 *      完全不同 (少 3 份专家 + 门控), 两者共用一个前缀 = 让"载错权重"有一天静默发生。
 *      本工程的纪律是"结构变了就换前缀, 于是载不进来看得见"。
 *   2. **自检/报告要自称得对**: 这一支的骨干不是 MoE, 报告里不能出现"专家数 / top-k /
 *      MaxVio" 那些读数 (对它是恒等), 要报的是"块数 / 通道数 / 参数量 / 前向耗时"。
 *   3. **不动既有 agent**: `AGENT_PPOMCTS_CONV` 原来是 MoE 版, 一旦原地改掉,
 *      它已有的读数、权重与 bench 口径就全断了 —— 这正是本仓库为 `SACAZAgent`
 *      的分支拆出 `SACAZMoETbAgent` / `SACAZLegacyAgent` 时用过的同一条理由。
 *
 * 与 MoE 版的**结构差别** (一份网络 vs E 份专家):
 *               MoE 版 (`Backbone::ConvExperts`)        本类 (`Backbone::ConvDense`)
 *   计算         k 个专家 (top-1 ⇒ 1 份) + 门控          1 份主干 (无门控)
 *   参数         E 倍专家 + 门控                       1 份主干
 *   深度         4 个残差块 (编译期)                     **6 个**(`CONV_BLOCKS`), 可传参改
 *   负载均衡     需要 (辅助损失 / 偏置回路)              **不需要** (没有路由)
 *   坐标通道     由 `ConvResNetExpert` 默认开            由构造参数 `coordPlanes` 决定
 *
 * 为什么"加深"是 6 块: `docs/conv_resnet_moe_analysis.md` §2.2 记的 scaling 轴是**深度**
 * (3 层感知野 7x7、5 层 11x11 覆盖 9 列), 而 ConvExperts 用的是 4; 本轮按"加深"取 6,
 * 它是本类的**默认值**、可被构造参数覆盖, 于是 A/B 只差这一个变量。
 *
 * ⚠ 与 MoE 版的关系: 两者**并存**(不同 agent 类型/不同前缀), 所以可以直接在界面上
 *   或 bench 里对弈比较; 本类不删 MoE 版的任何代码。
 * ============================================================================ */

#include "ppomcts_agent.h"

class PPOMCTSConvAgent : public PPOMCTSAgent
{
public:
    /* 本类的默认结构 (界面/自检/权重前缀的唯一来源, 不要在别处另写一套) */
    static constexpr int CONV_CHANNELS = RL::PPO_CONV_CHANNELS;             /* 32 */
    static constexpr int CONV_BLOCKS = RL::PPO::PPO_CONV_DENSE_BLOCKS;      /* 6 (MoE 版是 4) */
    static constexpr bool CONV_COORD_PLANES = true;                          /* 23 通道 */

    /*
       构造: 与 `PPOMCTSAgent` 同形, 但**骨干固定为 ConvDense**这样调用方不可能
       误传一个不同的骨干 (`backbone_` 不在参数表里)。`moeAuxCoef` 传 0 ——
       没有门控就没有辅助损失, 传非 0 只会让人以为它生效了。
    */
    PPOMCTSConvAgent(Chess &chess_,
                     int hiddenDim = 64,
                     float gamma_ = 0.99f,
                     float lr = 0.001f,
                     float cpuct = 1.414f,
                     int channels = CONV_CHANNELS,
                     int blocks = CONV_BLOCKS,
                     bool withGrad_ = true,
                     bool coordPlanes = CONV_COORD_PLANES)
        : PPOMCTSAgent(chess_, hiddenDim, gamma_, lr, cpuct,
                       /* expertHidden = 通道数 (与 ConvExperts 同语义) */ channels,
                       0.0f,
                       withGrad_,
                       RL::PPO::Backbone::ConvDense,
                       blocks,
                       coordPlanes)
    {
    }

    ~PPOMCTSConvAgent() = default;

    /* ---- 结构读数 (自检面板/报告用; 每一项都是"这一支到底是什么"的可读证据) ---- */
    int convChannels() const { return ppo.expertHidden; }
    int convBlocks() const { return (ppo.convBlocks > 0) ? ppo.convBlocks
                                                        : RL::PPO::PPO_CONV_DENSE_BLOCKS; }
    bool convCoordPlanes() const { return ppo.convCoordPlanes; }
    /* 输入通道数 = 19 平面 (+4 个坐标/区域通道) */
    int convInChannels() const { return 19 + (convCoordPlanes() ? 4 : 0); }

    /*
       自检文本: **只报这一支真实存在的量**。
       为什么不复用父类的自检: 父类会印 MoE 的专家数/top-k/负载均衡读数, 而本类
       没有门控 —— 照抄父类就会在面板上印出"专家数 0 / topK 0 / MaxVio …"这类
       看起来像坏了的行 (与 chessboard.cpp 里对"移出下拉框的 agent 不再列进 kAll"
       是同一条理由: 面板只列真实存在的读数)。
    */
    std::string selfCheckReport() const override
    {
        std::string out;
        char buf[512];
        out += "PPO+MCTS+ConvResNet (独立类, **无 MoE**, 加深主干)\n";
        std::snprintf(buf, sizeof(buf),
                      "  主干: conv2d(23->%d, 3x3, pad=1) + Tanh -> %d x [conv(%d->%d)+Tanh+残差]"
                      " -> conv 1x1(%d->19) -> 展平回 %d\n",
                      convChannels(), convBlocks(), convChannels(), convChannels(),
                      convChannels(), PPOMCTSAgent::STATE_DIM);
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "  输入通道: %d (%s) | 通道数 %d | 残差块 %d (MoE 版是 4)\n",
                      convInChannels(),
                      convCoordPlanes() ? "19 平面 + 4 坐标/区域" : "19 平面 (坐标/区域已关)",
                      convChannels(), convBlocks());
        out += buf;
        std::snprintf(buf, sizeof(buf),
                      "  actor 参数量 = %lld | critic 参数量 = %lld\n",
                      (long long)ppo.actorParamCount(), (long long)ppo.criticParamCount());
        out += buf;
        out += "  与 MoE 版的差别: 一份主干 (不是 E 份专家) + 无门控 + 无辅助损失/偏置均衡;";
        out += " 空间归纳偏置只有一份, 不存在\"每个专家各学一套空间特征\"的分裂\n";
        out += "  权重前缀与本类结构绑定 (与 MoE 版的 prefix 不同), 两者不能互相载入\n";
        return out;
    }
};

#endif // PPOMCTS_CONV_AGENT_H
