#ifndef SACAZLEGACYAGENT_H
#define SACAZLEGACYAGENT_H

/*
 * ============================================================================
 *  SACAZLegacyAgent —— SAC+MCTS+AlphaZero 的**行为还原版** (提交 59e5233)
 * ============================================================================
 *
 * 用户口径 (2026-09): "另外实现 agent 还原回 59e5233 的 SAC agent", 并且要求
 * **新旧 SAC 用不同的 C++ 类区分开** (而不是同一个类里的一个运行时开关) ——
 * 于是这一支是**派生类**, 与 AGENT_SACAZ 在类型层面就是两个东西。
 *
 * ---------------------------------------------------------------------------
 * 为什么是派生类而不是"复制一份 59e5233 的实现"
 * ---------------------------------------------------------------------------
 * 本仓库对"同一算法两个变体"的既有做法是**一份实现**:
 *   * AGENT_PPOMCTS / AGENT_PPOMCTS_MLP  -> 同一个 PPOMCTSAgent 类, 只是骨干不同;
 *   * AGENT_SACAZ   / AGENT_SACAZ_MOE    -> 同一个 SACAZAgent 类, 只是 backbone 不同;
 *     (见 chessboard.h 的枚举注释与 mainwindow.cpp 的 kAgents 注释)
 * 理由当时写得很清楚: 两份实现会**漂移**, 而漂移出来的差别会被误读成"算法的差别"。
 * 复制近 2000 行 SAC 代码再让它冻结, 会立刻产生两个后果:
 *   ① 共享算法里以后修的 bug (本仓库刚修过好几个) 只修在新的一份上, 旧的一份带着
 *      已知缺陷长期留在界面上, 而没人会去看那份代码;
 *   ② "两个 SAC 谁强"的对局失去意义 —— 它们已经不是同一个算法了。
 * 所以这里只钉**真正不同的那几项**(下面那张表), 其余全部继承。钉住的手段有两个,
 * 都是"机器可查"的, 而不是靠注释:
 *   ① 这些值写在本类的构造函数体里 (唯一一处), 界面各处只构造这个类, 不手抄参数;
 *   ② test_sacaz 的 [14] 节断言"同权重同局面下, 本类的策略/双 Q 输出与 SACAZAgent
 *      **逐位相同**" —— 网络构造若被改动, 那条断言当场变红;
 *      外加 tools/verify_sac_golden.ps1 的走法序列基准钉住搜索行为。
 *
 * ---------------------------------------------------------------------------
 * 与 AGENT_SACAZ (当前口径) 的**全部**差异
 * (逐项依据: `git show 59e5233:src/sacazagent.{h,cpp}`)
 * ---------------------------------------------------------------------------
 *   项                      | 59e5233 (= 本类)   | 当前 AGENT_SACAZ | 影响面
 *   ------------------------|--------------------|------------------|----------
 *   目标熵 entropyRatio     | 0.98               | **0.98 (2026-09 改回)** | — (相同)
 *   alpha 学习率            | 1e-3               | **1e-3 (2026-09 改回)** | — (相同)
 *   critic 目标钳位         | 无 (纯 MSE)        | 夹到 ±2          | 损失 (训练)
 *   Huber δ                 | 无 (纯 MSE)        | 1.0              | 损失 (训练)
 *   搜索叶子估值            | 全量 Q             | 稀疏头 (只算合法列) | 搜索 (推理)
 *   "从搜索学一次"          | 无                 | 有 (learnFromSearch) | 训练
 *   目标网同步率            | tau=1e-3 / 每 64 步 | (2026-09 F1 期间曾被改成硬拷贝/256, 已改回) | 训练
 *   ------------------------|--------------------|------------------|----------
 *   网络结构 / 隐层激活     | **完全相同**       | 相同             | — (见下)
 *   动作/状态表示           | 128 槽 / 1263 维   | 相同 (默认构建)   | —
 *   其它新增开关            | —                  | rewardScale / valueScale = 1.0 (恒等)
 *
 * **α 那两行现在是相同的** —— 不是本类改了, 而是**当前实现改回了 59e5233 的值**:
 * 2026-09 的受控实验 (docs/sac_learn_reward_2026_09.md §9) 量到"目标熵 0.5 + alpha lr
 * 5e-3"那一套在对弈里会让训练**有害**(训练后 37.5% vs 未训练 52.5%), 而改回 0.98/1e-3
 * 之后是 82.5%。本类仍然显式写这两个值 (它要钉住 59e5233, 不该依赖基类的默认值)。
 *
 * **网络那一行值得单独说**: 59e5233 的 `buildNet` 所有分支用的都是
 * `RL::Layer<RL::Tanh>`, 当前代码**也是**(回归修好之后), 两边的 buildNet 是逐字相同的
 * 代码 —— 所以"复现 59e5233 的网络"靠的是"用同一行代码", 本类不需要、也没有任何
 * 激活开关。**不要**改回用 `RL::TanhNorm<RL::Linear>` 且 r=1 去"复现"那一层:
 * 它的偏置是加在 tanh **外面**的 (`tanh(r·Wx)+b`), 与 `Layer<Tanh>` 的 `tanh(Wx+b)`
 * 不是同一个函数 —— 实测 (test_sacaz [14], 同权重同局面, 稀疏 MLP 专家骨干)
 * max|Δπ| = 1.8e-07、**max|ΔQ| = 8.9e-06**。完整经过见 sacazagent.h 里
 * `hiddenActivationName()` 上面那段注释。
 * `SACAZAgent::hiddenActivationName()` 保留为**回归指示器**: 它读 actor 第 2 层的真实
 * 类型, 上一轮那个掉 26 个点的激活回归在面板上原本一个字都看不出来。
 *
 * 也就是说: 界面上"两个 SAC"的差别是**上表那 5 项口径**, 不是网络结构。
 * 这 5 项里有 4 项只影响**训练**, 1 项影响**搜索**(叶子估值口径) —— 所以两者用同一份
 * 初始权重起跑, 各自训练之后会分叉, 这正是要有独立权重文件的原因。
 *
 * ---------------------------------------------------------------------------
 * 权重文件必须独立 (用户口径: "新旧 sac agent 的权重文件用不同名字区分开")
 * ---------------------------------------------------------------------------
 *   AGENT_SACAZ      -> weights/sacaz_agent      (_actor / _q1 / _q2)
 *   AGENT_SACAZ_OLD  -> weights/sacaz_old_agent  (_actor / _q1 / _q2)   <- 本类
 *   AGENT_SACAZ_MOE  -> weights/sacaz_moe_agent  (_actor / _q1 / _q2)
 *
 * 为什么不能共用一个前缀 (两个后果都是**静默**的):
 *   * 两者的训练口径不同 (上表 5 项), 同一个局面会被训成两组不同的权重 ——
 *     共用前缀 = 后训练的那一支直接覆盖另一支, 界面上一切正常, 只有"棋力对不上
 *     训练量"这种无法归因的现象;
 *   * 载入也一样: 两者**参数结构完全相同** (都是 iFcLayer 的 w/b), 结构指纹挡不住,
 *     于是错的那一份会被当成对的那一份用 (本仓库 PPO+MCTS 就栽在"名字漂移 ⇒ 权重
 *     从来没被载入过"上, 见 chessboard.cpp 的 weightFilesOf 注释)。
 * 前缀跟着**类**走 (两个类各自的 defaultWeightPrefix()), 界面按 agent 类型取默认值;
 * 后台训练的临时前缀也独立 (weights/_temp_train_sacaz_old*)。
 *
 * ---------------------------------------------------------------------------
 * 为什么放在头文件里 (没有 .cpp)
 * ---------------------------------------------------------------------------
 * 本类只有"构造参数 + 5 个赋值 + 两段文案", 没有算法。写成 .cpp 就得把它加进
 * CMakeLists 里十几个目标 (每个自己列源文件, 见 CMakeLists 顶部"不用 GLOB"的说明),
 * 而其中大部分根本不构造它 —— 一个漏加就是链接错误。算法仍然全在 sacazagent.cpp 里,
 * 这里只是一层口径。
 */
class SACAZLegacyAgent : public SACAZAgent
{
public:
    /* 59e5233 的两个数 (原值见 `git show 59e5233:src/sacazagent.cpp` 的构造初始化表) */
    static constexpr float LEGACY_ENTROPY_RATIO = 0.98f;
    static constexpr float LEGACY_ALPHA_LR = 1e-3f;
    /*
       [2026-09 修正] 目标网同步率也必须钉住。
       为什么: F1 那一轮把**基类默认**改成了"硬拷贝每 256 步", 本类没有显式钉这两个成员
       ⇒ "59e5233 行为还原版"跟着一起变了 —— 而它的全部意义就是**行为还原**。
       这正是"基类默认值即口径"的坑 (与 docs/session_2026_09_sac.md §4 第 9 条同型):
       凡是被 59e5233 固定下来的量, 本类都要显式写一遍。基类默认已改回 61a974d 口径,
       但**这条钉住仍然保留** —— 下一个改基类默认的人不该再把还原版带跑。
    */
    static constexpr float LEGACY_TARGET_TAU = 1e-3f;   /* 59e5233: Polyak tau */
    static constexpr int LEGACY_TARGET_ITER = 64;       /* 59e5233: 每 64 步同步一次 */

    /*
       本类的权重前缀。**静态**: ChessBoard::defaultWeightPath() 是按 agent 类型查表的
       静态函数, 没有实例可用。调用处一律写**限定名** (SACAZLegacyAgent::...),
       因为它在派生类里遮蔽了基类同名静态函数 —— 非限定调用看着"能编过", 拿到的
       可能是另一支的前缀, 那正是本类最不希望发生的错误。
    */
    static const char *defaultWeightPrefix() { return "weights/sacaz_old_agent"; }

    /*
       构造参数与 SACAZAgent **逐字一致**, 不额外暴露任何口径开关:
       本类的口径是**固定**的, 不给"构造出一个半旧半新的东西"留口子。
         * 网络结构由基类 buildNet 决定 —— 与 59e5233 逐字相同, 没有开关;
         * 5 项口径差异在构造函数体里一次钉死。
    */
    SACAZLegacyAgent(Chess &chess_,
                     int hiddenDim_ = 64,
                     float gamma_ = 0.99f,
                     float lr = 0.001f,
                     float cpuct = 1.5f,
                     Backbone backbone_ = Backbone::Mlp,
                     int expertHidden_ = 64,
                     float auxLossCoef_ = 0.1f)
        : SACAZAgent(chess_, hiddenDim_, gamma_, lr, cpuct, backbone_, expertHidden_,
                     auxLossCoef_)
    {
        /* ---- 59e5233 口径 (上表那 5 项; 其余全部继承基类, 已经是同值) ---- */
        entropyRatio = LEGACY_ENTROPY_RATIO;
        learningRateAlpha = LEGACY_ALPHA_LR;
        /*
           下面两条是**回归排查这一轮新加的约束** (见 sacazagent.h 的 clampTarget 长注释),
           59e5233 里没有 —— "<=0" 就是原样关掉, 于是目标与损失都回到"纯 MSE"。
        */
        clampTarget = 0.0f;
        huberDelta = 0.0f;
        /*
           搜索叶子估值: 59e5233 每片叶子算**全量** Q (那时动作空间是 128 槽, 全量只要
           128 列; 稀疏头是为 8100 列才加的)。这里关掉稀疏路径 —— 不只是为了等价,
           也为了让"两个 SAC 的差别"里不含"估值走了另一条代码路径"这个变量。
        */
        sparseLeafEval = false;
        /*
           [新] "从自己的搜索学一次" 也钉成关 —— 59e5233 没有这条路径 (它对弈时只在
           rollout 里学, 而那批样本 hasSearch=false)。开着它会让这一支的行为偏离 59e5233,
           而本类的全部意义就是"钉住 59e5233 的行为"。
        */
        learnFromSearch = false;
        /*
           [2026-09 修正] 目标网同步率 (第 6/7 项口径): 基类默认值曾经被 F1 那一轮
           改成"硬拷贝 / 每 256 步", 而本类当时**没有**钉它 ⇒ 还原版被"优化"污染了。
           现在显式写死 59e5233 的值 (tau=1e-3 / 每 64 步 = 一次会话只移动 2~4%)。
           test_sacaz [14] 有一组断言是拿**对象里实际生效的值**对这个表, 所以以后
           基类默认再改也不会静默渗进来。
        */
        targetTau = LEGACY_TARGET_TAU;
        replaceTargetIter = LEGACY_TARGET_ITER;
    }

    ~SACAZLegacyAgent() override = default;

    /* 自检面板第一行要能一眼认出"这是哪一支" (基类按骨干区分, 本类按类区分) */
    const char *guiAgentLabel() const override
    {
        return "SAC+AZ-59e5233 (AGENT_SACAZ_OLD, 行为还原版)";
    }

    std::string getName() const override
    {
        return "SAC+MCTS+AlphaZero (59e5233 行为还原版)";
    }

    /*
       自检报告 = 本类的差异说明 + 基类的完整报告。
       为什么要**前置**一段: 基类那份报告是"表示/口径事实"的清单, 读者若不知道上面
       那张差异表, 会把两个 SAC 的读数当成同一个 agent 的读数 (两者的参数量、状态/动作
       维度、骨干全都一样, 面板上唯一能区分的就只有这一段和口径行)。
    */
    std::string selfCheckReport() const override
    {
        std::string out;
        out += "【59e5233 行为还原版 = 派生类 SACAZLegacyAgent】与 AGENT_SACAZ 是"
               "同一份算法 + 不同口径 (不是复制出来的第二份实现):\n";
        out += "  目标熵 0.98 (当前 0.5) | alpha 学习率 1e-3 (当前 5e-3) |"
               " critic 目标**不钳位**、纯 MSE (当前 夹 ±2 + Huber δ=1) |"
               " 叶子估值走**全量** (当前 稀疏头)\n";
        out += "  网络结构与 59e5233 逐字相同 (隐层激活 Layer<Tanh>, 没有开关);"
               " \"用 TanhNorm<Linear> r=1 复现那一层\"已实测证伪 (偏置在 tanh 外,\n";
        out += "  max|dQ| = 8.9e-06) —— 详见 sacazagent.h 里 hiddenActivationName()"
               " 上面那段\n";
        out += "  权重文件独立: " + std::string(SACAZLegacyAgent::defaultWeightPrefix())
               + "_actor / _q1 / _q2 (与 AGENT_SACAZ 的 weights/sacaz_agent 不共用,"
                 " 否则两边会互相覆盖/串权重)\n";
        out += "--------------------------------------------------------------\n";
        out += SACAZAgent::selfCheckReport();
        return out;
    }
};

#endif /* SACAZLEGACYAGENT_H */
