/*
 * test/gui_seq_wiring_main.cpp - 钉住「界面那一支的专家**真的是**序列 Transformer」
 * ============================================================================
 *
 * 为什么需要这个测试 (它是 2026-10-16 一次**真实事故**的回归测试):
 *   `CMakeLists.txt` 用四个宏把界面里的四支 agent 的专家换成序列 Transformer:
 *     SACAZ_MOE_EXPERT_SEQ / SACAZ_LEGACY_MOE_EXPERT_SEQ /
 *     DQNMCTS_MOE_EXPERT_SEQ / DQNAB_MOE_EXPERT_SEQ
 *   但那些宏原先只加在 `chess` **自己的源文件**上 (APP_SOURCES = main / mainwindow / …),
 *   而真正**建 agent** 的 `chessboard.cpp` 与四个 agent 的 .cpp 全在 `CHESS_CORE` 里 ——
 *   它们不是 chess 的源文件 ⇒ 宏**从来没有到达建 agent 的那几个 TU**。
 *   后果 (静默、不报错、读数也看不出来): 下拉框上的字是"稀疏MoE+序列专家"
 *   (那几个字在 mainwindow.cpp 里, 它吃到了宏), 实际建出来的却是 **TB 专家**;
 *   `chessboard.cpp` 里 `a->legacyHashAction = true` 那一行同样被整段编掉。
 *   修法: 游戏层编两份 —— `CHESS_CORE`(不带宏, 给 test / bench 各目标) 与
 *   `CHESS_CORE_SEQ`(带宏, 只给界面 `chess`)。见 CMakeLists 里 CHESS_CORE_SEQ 那一段。
 *   ⚠ 四个宏在 `CHESS_CORE_SEQ` 上是 **PUBLIC**: 不只是库本身, 每一个 include
 *     这些 agent 头文件的 TU (界面自己的源文件、本测试) 都必须看到**同一套宏** ——
 *     否则 `STATE_DIM` 这样的编译期成员会在两个 TU 里取到不同的值, 那是 ODR 违反
 *     (本测试第一版就是这么炸的: 库带宏、测试不带宏 ⇒ static_assert 当场红)。
 *
 * 本测试**链 `CHESS_CORE_SEQ`**, 于是它断言的是"界面那一份"的结构。三条独立的证据:
 *
 *   [1] **编译期**: `stateDim()` 是开关的直接后果 (对齐表示 19x90 = 1710);
 *       不带宏时 DQNMCTS 是 1263、SACAZ_MOE 是 1263 ⇒ 静态断言会当场编译失败;
 *   [2] **权重前缀**: 序列专家那一支的前缀里必须带 `seq` —— 它与 TB 那一份
 *       **绝不能互相载入** (前缀是这条纪律唯一的执行点);
 *   [3] **结构 (参数量)**: TB 专家的主干是 28.8 M 级, 序列专家是 0.46 M 级 ——
 *       差 60 倍, 所以"到底是哪一种专家"可以用 `trunkParamCount()` 直接量出来,
 *       不依赖字符串、也不依赖报告里的措辞。
 *
 * 另外两条把**运行时旋钮**钉住 (超参搜索要用它们):
 *   [4] `expertHidden` (token 宽) 真的改变结构: 32 < 64 < 128 的参数严格递增;
 *   [5] 不传该参数时 == 传默认值 (MOE_SEQ_TOKDIM) ⇒ 既有调用点的结构一位不变。
 *
 * 用法:  cmake --build <build> --target test_gui_seq_wiring && <build>/test_gui_seq_wiring
 * 口径:  它**不测棋力**, 只测接线。界面上真正"跑起来是什么"还要求一次 UI 级运行
 *        (本测试所在的那一层没有 GUI)。
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "chess.h"
#include "dqnabagent.h"
#include "dqnmctsmoetbagent.h"
#include "sacazmoetbagent.h"

/* ============================================================================
 *  [1] 编译期: 表示维数 (开关的直接后果)
 * ============================================================================ */
static_assert(SACAZMoETbAgent::STATE_DIM == 1710,
              "CHESS_CORE_SEQ 下 SACAZMoETbAgent 必须是 19x90 = 1710 的对齐表示 "
              "(SACAZ_MOE_EXPERT_SEQ=1 隐含 SACAZ_ALIGNED_REPR=1)");
static_assert(DQNMCTSMOETbAgent::STATE_DIM == 1710,
              "CHESS_CORE_SEQ 下 DQNMCTSMOETbAgent 必须是 19x90 = 1710 的对齐表示 "
              "(DQNMCTS_MOE_EXPERT_SEQ=1 隐含对齐表示; 不带宏时是 1263)");
/* DQNAB 的表示本来就恒为 19x90 (开关只换专家) —— 这一条是"它没被顺手改坏"的对照 */
static_assert(DQNABAgent::STATE_DIM == 1710, "DQNAB 的 STATE_DIM 恒为 19x90");

namespace {

int g_total = 0;
int g_fail = 0;

void check(bool ok, const char *what)
{
    g_total++;
    if (ok) {
        std::printf("[ok]   %s\n", what);
    } else {
        g_fail++;
        std::printf("[FAIL] %s\n", what);
    }
}

bool contains(const char *hay, const char *needle)
{
    return hay != nullptr && needle != nullptr && std::strstr(hay, needle) != nullptr;
}

} // namespace

int main()
{
    std::printf("=== test_gui_seq_wiring: 界面那一支 (CHESS_CORE_SEQ) 的专家接线 ===\n\n");

    /* ========================================================================
     *  [2] 权重前缀: 序列专家那一支必须自带 seq 后缀
     * ======================================================================== */
    std::printf("--- [2] 权重前缀 (序列专家与 TB 专家不能互相载入) ---\n");
    const char *sacShared = SACAZMoETbAgent::sharedWeightPrefix();
    std::printf("     SACAZMoETbAgent::sharedWeightPrefix() = %s\n", sacShared);
    std::printf("     DQNABAgent::backboneName(SparseMoeTb) = %s\n",
                DQNABAgent::backboneName(DQNABAgent::Backbone::SparseMoeTb));
    check(contains(sacShared, "seq"),
          "SACAZMoETbAgent::sharedWeightPrefix() 带 seq (序列专家那一支)");
    check(contains(DQNABAgent::backboneName(DQNABAgent::Backbone::SparseMoeTb), "序列"),
          "DQNAB 的 SparseMoeTb 支自称**序列**专家 (不是 TB 专家)");

    /* ========================================================================
     *  [3] 结构: 主干参数量差 60 倍 ⇒ 一眼能分辨是哪种专家
     * ======================================================================== */
    std::printf("\n--- [3] 结构 (主干参数量): TB 约 28.8 M / 序列约 0.46 M ---\n");
    Chess env;
    env.reset();

    /* TB 专家的阈值: 一支 DQNAB 主干在 TB 口径下是 28.77 M 级; 序列口径 < 1 M。 */
    const long long kSeqMax = 2000000LL;

    DQNABAgent dqnab(env, 64, 0.99f, 0.001f, DQNABAgent::Backbone::SparseMoeTb);
    const long long dqnabTrunk = dqnab.trunkParamCount();
    std::printf("     DQNAB  主干参数 = %lld (头 %lld)\n", dqnabTrunk, dqnab.headParamCount());
    check(dqnabTrunk > 0 && dqnabTrunk < kSeqMax,
          "DQNAB 主干是**序列专家**那一档 (TB 口径会是 28.77 M 级)");

    DQNMCTSMOETbAgent dqnmcts(env, 64, 0.99f, 0.001f, 1.0f, 1.5f);
    const long long dqnmctsTrunk = dqnmcts.trunkParamCount();
    std::printf("     DQNMCTS 骨干名 = %s\n", dqnmcts.backboneName());
    check(contains(dqnmcts.backboneName(), "序列"),
          "DQNMCTSMOETbAgent 自称**序列**专家 (骨干名里带'序列')");
    std::printf("     DQNMCTS 主干参数 = %lld (头 %lld, STATE_DIM=%d)\n",
                dqnmctsTrunk, dqnmcts.headParamCount(), DQNMCTSMOETbAgent::STATE_DIM);
    check(dqnmctsTrunk > 0 && dqnmctsTrunk < kSeqMax,
          "DQNMCTS 主干是**序列专家**那一档 (TB 口径会是 28.77 M 级)");

    SACAZMoETbAgent sac(env, 64, 0.99f, 0.001f, 1.5f, 64, 0.1f,
                        SACAZMoETbAgent::TrunkMode::Shared);
    std::printf("     SACAZ_MOE 骨干名 = %s, 参数量 = %lld\n",
                sac.backboneName(), sac.uniqueParamCount());
    check(contains(sac.backboneName(), "序列"),
          "SACAZMoETbAgent 自称**序列**专家 (骨干名里带'序列')");

    /* ========================================================================
     *  [3b] 动作输出**与 PPO 同构**: 8100 双射 (2026-10-16 用户口径)
     * ========================================================================
     * 用户口径: "sac agent 对齐 ppo agent 输出"。表示 (19×90) 上一轮已经对齐了, 这一轮
     * 对齐的是**输出空间**: 界面那一支不再打开 `legacyHashAction`, 于是
     * `stepToActionIdx` 走 `canonicalCell(from)*90 + canonicalCell(to)` —— 与
     * `PPOMCTSAgent` 同一个 8100 双射 + 同一套合法集掩码 softmax。
     *
     * 为什么用**自检报告里的那句话**当判据: 它是本工程"实际用的是哪一套编码"的唯一
     * 可读证据 (与 `stepToActionIdx` 的判据逐字同源)。直接读私有成员做不到, 而另写一条
     * "猜"的逻辑正是当年那条三目印反的原因 (只看 legacyHashAction、漏了 ALIGNED_REPR)。
     */
    std::printf("\n--- [3b] SAC+AZ-MoE 的动作输出 = 与 PPO 同一个 8100 双射 ---\n");
    {
        const std::string rep = sac.selfCheckReport();
        const bool bijective = rep.find("对齐双射 8100") != std::string::npos;
        std::printf("     自检报告里的动作口径: %s\n",
                    bijective ? "对齐双射 8100" : "(不是双射 —— 检查 legacyHashAction)");
        check(bijective,
              "SAC+AZ-MoE 的动作口径 = 对齐双射 8100 (界面口径与 PPO 同构)");
        /*
           掩码的**逐位忠实性**同时被下面这条钉住: 掩码容器从 2 x uint64 (128 位, 会
           把 i 与 i+64k 折进同一位) 换成了 `std::bitset<ACTION_DIM>`。8100 位只能
           用 8100 位装 —— 这条断言是"没退回折叠版"的第一道门 (第二道在
           test_sacaz_aligned_seq 的学习动力学断言里)。
        */
        check(SACAZMoETbAgent::ACTION_DIM == 8100,
              "SAC+AZ-MoE 的动作空间大小 == 8100 (与 PPOMCTS 同一个动作空间)");
    }

    /* ========================================================================
     *  [4] 运行时旋钮 expertHidden 真的改结构 (32 < 64 < 128)
     * ======================================================================== */
    std::printf("\n--- [4] expertHidden (token 宽) 是**活的**运行时旋钮 ---\n");
    DQNABAgent dqnab32(env, 64, 0.99f, 0.001f, DQNABAgent::Backbone::SparseMoeTb, 32);
    DQNABAgent dqnab128(env, 64, 0.99f, 0.001f, DQNABAgent::Backbone::SparseMoeTb, 128);
    std::printf("     DQNAB 主干: tokDim 32 -> %lld | 64 -> %lld | 128 -> %lld\n",
                dqnab32.trunkParamCount(), dqnabTrunk, dqnab128.trunkParamCount());
    check(dqnab32.trunkParamCount() < dqnabTrunk
              && dqnabTrunk < dqnab128.trunkParamCount(),
          "DQNAB: tokDim 32 < 64 < 128 的参数严格递增");

    DQNMCTSMOETbAgent dqnmcts32(env, 64, 0.99f, 0.001f, 1.0f, 1.5f, false, 32);
    DQNMCTSMOETbAgent dqnmcts128(env, 64, 0.99f, 0.001f, 1.0f, 1.5f, false, 128);
    std::printf("     DQNMCTS 主干: tokDim 32 -> %lld | 64 -> %lld | 128 -> %lld\n",
                dqnmcts32.trunkParamCount(), dqnmctsTrunk, dqnmcts128.trunkParamCount());
    check(dqnmcts32.trunkParamCount() < dqnmctsTrunk
              && dqnmctsTrunk < dqnmcts128.trunkParamCount(),
          "DQNMCTS: tokDim 32 < 64 < 128 的参数严格递增");
    check(dqnmcts.expertHidden == DQNMCTSMOETbAgent::MOE_SEQ_TOKDIM
              && dqnmcts128.expertHidden == 128,
          "DQNMCTS: 不传 == 默认常量 MOE_SEQ_TOKDIM; 传了就回显真实取值");

    /* ========================================================================
     *  [5] 不传参数 == 传默认值 (既有调用点的结构一位不变)
     * ======================================================================== */
    std::printf("\n--- [5] 默认值口径 ---\n");
    DQNABAgent dqnabDefault(env, 64, 0.99f, 0.001f, DQNABAgent::Backbone::SparseMoeTb,
                            DQNABAgent::MOE_SEQ_TOKDIM);
    check(dqnabDefault.trunkParamCount() == dqnabTrunk,
          "DQNAB: 不传 expertHidden == 传 MOE_SEQ_TOKDIM (结构逐位相同)");

    std::printf("\n=== 断言 %d 条, 失败 %d 条 ===\n", g_total, g_fail);
    if (g_fail != 0) {
        std::printf("**接线错了**: 界面那一支的专家不是序列 Transformer —— "
                    "检查 CHESS_CORE_SEQ 是否真的带上了那四个宏 (见 CMakeLists)。\n");
    }
    return g_fail == 0 ? 0 : 1;
}
