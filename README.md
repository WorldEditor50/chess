# 中国象棋 (Qt6 + C++17) — 带 AI Agent 与强化学习内核

一个用 Qt6 写的中国象棋程序：完整的棋规、可玩的界面、**9 个可选 AI Agent**（从
Alpha-Beta 到 SAC + MCTS + AlphaZero）、一个**纯 C++ 的强化学习内核**（SIMD 加速、
自带稀疏 MoE），以及一整套**可复现的验证手段**（13 个 ctest 套件 + 5 个界面自动化脚本 +
一批手动基准：`bench_moe` / `bench_ppo_vs_ab` / `bench_ppo_mt` / `bench_policy_agreement` /
`bench_ppo_sims` / **`bench_diag`（诊断仪表盘：搜索健康度 / 战术题库 / value 校准）** /
**`bench_anchor`（固定开局集 + 换先手成对计分 + Elo 置信区间）** / **`train_ppo`（无界面
常驻训练器）** / **`probe_dqnmcts_aliasing`（DQN+MCTS 的表示能力探针：动作别名 / 状态
不可分性 / 终局奖励通道 / 回放池新信息量）** 等）。

> 这个工程的写法偏"工程审计"风格：每个非显然的决定都写成注释，每个结论都有实测数字，
> 发现的问题（包括自己引入的）都记在 `docs/issues_review.md` 里，包括还没修好的。
> 想快速了解"为什么是这样"，直接看 `docs/` 比看代码快。

---

## 目录

- [快速开始](#快速开始)
- [功能](#功能)
- [AI Agent](#ai-agent)
- [强化学习内核 `RL_CORE`](#强化学习内核-rl_core)
- [测试与验证](#测试与验证)
- [项目结构](#项目结构)
- [文档](#文档)
- [已知限制](#已知限制)

---

## 快速开始

**环境**（这是本机实测通过的组合，其它版本没试过）：

| | 版本 |
|---|---|
| 编译器 | MSVC 2022 (14.44) |
| Qt | 6.9.2 `msvc2022_64`（Widgets + Sql） |
| CMake | 3.30 + Ninja |
| 系统 | Windows 10/11 x64（AVX2） |

**构建**（`build_main.bat` 就是把下面三行包起来，路径按需改）：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
C:\Qt\Tools\CMake_64\bin\cmake.exe -S . -B build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release ^
    -G Ninja -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 -DCMAKE_BUILD_TYPE=Release
C:\Qt\Tools\CMake_64\bin\cmake.exe --build build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release
```

或者直接跑 `build_main.bat`。

**运行 / 测试**：

```bat
build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe
cd build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release && ctest --output-on-failure
```

`ctest` 的 `test_match` 需要 Qt 的 DLL：脚本里用 `ENVIRONMENT_MODIFICATION` 把 Qt 的
`bin` 前置进 `PATH`。**从普通 shell 手动跑 `chess.exe` 时也要自己加**，否则进程以
`0xC0000135`（找不到 DLL）秒退 —— 看起来像"程序打不开"，其实是环境问题。

**CMake 选项**：

| 选项 | 默认 | 说明 |
|------|------|------|
| `CHESS_ENABLE_AVX2` | `ON` | 给 RL 内核加 `/arch:AVX2`；配置期用 `try_run` 做一次 CPUID 探测，不支持会自动关掉并警告 |
| `CHESS_BUILD_TESTS` | `ON` | 编译测试/基准可执行文件并注册进 ctest |
| `CHESS_USE_PCH` | `OFF` | 给 RL 库预编译 `rl/tensor.hpp` + `rl/layer.h` |

---

## 功能

### 棋

* 完整中国象棋规则：走法生成、**应将 / 自杀 / 照面**过滤、将杀与困毙、**三次重复**与
  60 回合自然限着
* 棋盘绘制与交互、悔棋、走法记录、SQLite 棋局库（保存/回放）
* 真界面上的"AI 正在思考"指示器：**沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时**，棋盘顶部另有
  一条状态条（分阶段显示"① 探索+预训练 / ② 搜索决策 / ③ 落子"）

### 对弈与训练（右侧面板）

* **Agent 对 Agent 对弈**：可设局数与每步预训练步数；每局**交换先后手**，胜负按参赛者
  A/B 记（中国象棋先手优势大，固定谁执红只是在测"谁执红"）
* **实时比分** + **逐局明细列表**（每局一行，含本局环境奖励**及其口径**与手数）
* **训练损失曲线**（每个 agent 一条线）与**环境奖励曲线**（A/B 两条，**每手一个点**：
  值是"本局累计"的走子方视角奖励，局末再补一个含终局值的点）：
  自绘控件，**双击可放大**到独立窗口（跟随源控件实时同步），支持**导出 CSV**；
  另有单独的**"导出损失曲线"**按钮（只导损失那一段）。导出用的是**全部采样点**，
  而屏幕曲线/读数按 2000 点滚动窗口 —— 两者口径不同是刻意的（见 `metricsview.h`
  的 `Series::history` 说明：拿窗口当整场分析会把结论带偏）
* **奖励曲线取的是"学习口径"**（2026-09）：即 agent **自己**的
  `computeReward()`（材质 ×0.1 + 每步代价）+ 终局值（SAC 开着塑形时是
  ±(1+败方材质/3.5)），曲线名与逐局明细里都标出是哪个口径。**没有学习口径的纯搜索
  agent（Alpha-Beta / MCTS / EVAB）仍是引擎口径（材质 ×1 + 终局 ±1）并标注
  `[引擎口径]`** —— 两个口径差 **10 倍**，同一张图上混着两种口径的线时不可直接比大小
  （见 `docs/sac_learn_reward_2026_09.md` §1.1 与 `chessboard.h` 的 `RewardAccounting`）。
* 对弈结束后**静默保存权重**到标准路径（不弹窗；几百 MB 的写盘会显示"请稍候"沙漏）
* **后台持续训练**（2026-09 补齐）：当前选中的 agent 在后台线程里持续"自对弈 → 在线更新 →
  写回主 agent"。**十一个有权重可训的 agent 全部接上**（PG / DQN / PPO+MCTS / PPO+MCTS-MLP /
  DQN+MCTS / EVAB / SAC+AZ / SAC+AZ-MoE / SAC+AZ-59e5233 / SAC+AZ-59e5233-MoE / DQN+AB ——
  PPO 的两种骨干、SAC 的两支与还原版的两种骨干各自都有独立权重；Alpha-Beta 与 MCTS
  没有可训练权重）。
  训练**在对弈期间照常进行**（每轮把权重同步进主 agent ⇒ 一局之内模型会变，这条是明确
  选择的行为，见 `docs/issues_review.md` 零之二点二十三 §2b）。
  一轮的时长 = 关窗等待时间，各 agent 差两三个数量级（PG/DQN 秒级、SAC+AZ-MoE 与
  DQN+AB 分钟级），可用 `setBackgroundTrainRound()` 调小 —— 但 **maxMoves 不要低于 32**：
  SAC+AZ / DQN+AB 的 `learnBatch` 在"回放池 < batchSize(32)"时直接返回，更短的轮次会
  **一次梯度更新都不做**（且损失曲线上报不出来）。
* **模型自检面板**（2026-09）：选中 agent 后自动显示该 agent 的**结构与口径**读数。
  存在的理由：面板上原来只有损失曲线与逐局明细，而**这两样都不能判断"模型值不值得继续训"**
  —— 损失只说明网络与自己的目标一致（PPO 的 0.003 与 DQN 的 22 量纲不同、都不可比，也都
  与棋力无关），自对弈的胜负里赢家和输家是同一份权重。面板报的是真正的前置判据：
  **权重文件状态**（启动扫描有没有命中、文件在不在、多大）、状态维度与**规则上下文通道数**、
  **动作别名**（同一局面里有多少互不相同的着法被迫共用一个 Q 槽位）、**终局通道**
  （`getResult()` 判出的终局里，旧口径 `isGameOver()` 漏掉了多少局）、搜索/训练口径
  （根搜索展开覆盖率、值门控的 gap 与回滚、MoE 路由直方图……）。
  **十六个 agent 全部实现了自检**（数据源 `AgentBase::selfCheckReport()`，由
  `ChessBoard::getAgentSelfCheck()` 转发），另外纯搜索的那几支（Alpha-Beta 四档
  / MCTS）还会在面板里跑一次**决策合法性自检**（"这一步返回值在合法集里吗"）—— 它正是
  "agent 返回默认 Step 被误读成无棋可走"那类 bug 的回归指示器。
  刷新时机是"选中 agent / 每手预训练之后 / 启动加载完成"；启动时每个已加载的模型也会
  各打一行 `[selfcheck] <名字>: <表示层摘要>`。面板明确写着"这是表示/口径事实，**不是棋力**"。
  按钮 **"全部模型自检"** 会把十六个 agent 排在一起 —— 编码/口径的差别只有横向对比才看得出来。
  自检面板第一段现在还会报**实测的隐层激活类型**（读 actor 第 2 层，不回显开关）与
  **实际生效的口径**（clampTarget / huberDelta / 叶子估值 / 动作空间）：上一轮那个
  "激活被换成 TanhNorm<Sigmoid>、棋力掉 26 个点"的回归，面板上原本一个字都看不出来。

### 工程

* 程序图标（窗口/任务栏 + exe 文件图标，由脚本生成）
* 载入/保存权重时的沙漏等待窗（复用思考指示器；**快的操作不闪窗**，延迟 300 ms 才显示）
* **启动时加载全部 7 组权重**（含稀疏 MoE 的 3×146 MB，约 10 s，沙漏全程给反馈）；
  每组各记一条 `[weights] <名字>: N ms`，保存也记一条（慢了能一眼看出是哪一组）
* 权重文件格式 v2：**逐比特无损** + 结构指纹 + 每张量 CRC32 + **原子写入**，
  并且**兼容旧的十进制文本格式**

---

## AI Agent

界面上可选的 16 个 agent（`主界面 → 对战AI / A方 / B方`）：

| Agent | 说明 | 每步预算（本机实测） |
|-------|------|---------------------|
| **Alpha-Beta Pruning** | 经典 α-β + 静态搜索 | 深度 4，约 90 ms |
| **Alpha-Beta L1 / L2 / L3** | **同一套搜索的三档弱等级**（只有深度不同：1 / 2 / 3）。**纯搜索、无权重**：既不训练也不上报训练损失。用途是当**陪练与标尺**——棋力可调、且完全不随训练漂移；与深度 4 那一档一起构成一条棋力阶梯 | 每步 0 / 3 / 34 ms |
| **MCTS** | UCB1 蒙特卡洛树搜索 | 800 次模拟 |
| **Policy Gradient** | REINFORCE + baseline，走子前在线训练 | 预训 64 步 |
| **Deep Q-Network** | 双网 + 经验回放 + ε-greedy | 预训 64 步 |
| **PPO+MCTS** | AlphaZero 风格：搜索访问分布监督 actor；骨干 = 稀疏 MoE + **TB 专家**（E=4 top-1） | 400 次模拟，约 3.2 s |
| **PPO+MCTS (稀疏MoE+MLP专家)** | **同一套实现**，骨干换成 `MlpExpert`（E=8 top-2）：便宜 ~25×、容量小 ~18×，于是同一时间预算下模拟次数给到 4 倍 | 1600 次模拟，约 0.23~0.39 s |
| **DQN+MCTS** | 用 Q 值做叶子估值 + 树搜索 | 200 次迭代 |
| **EVAB** | **学会评估的 Alpha-Beta**：置换表/迭代加深/排序 + 学习到的价值网按 `blend` 混合 | 深度 6 + 800 ms 上限 |
| **SAC+AZ** | **SAC + MCTS + AlphaZero**：最大熵 critic 给 PUCT 搜索估值，α 自动调节；每次真实决策还会**从自己的搜索学一次**（不依赖"探索+预训练"勾选框） | 256 次模拟，约 12 ms |
| **SAC+AZ (稀疏MoE+TB专家)** | 同上，骨干换成**稀疏路由 MoE + TransformerBlock 专家** | 16 次模拟，约 160 ms |
| **SAC+AZ (59e5233 行为还原版)** | 口径回到提交 `59e5233`：目标熵 0.98 / α 学习率 1e-3 / critic 不钳位且纯 MSE / 叶子全量估值。**独立的类** `SACAZLegacyAgent`（不继承 `SACAZAgent`），**权重文件独立**（`weights/sacaz_old_agent*`） | 256 次模拟，约 12 ms |
| **SAC+AZ (59e5233 还原版, 稀疏MoE+TB专家)** | 同一支还原版的**另一个骨干**：同一个类、同一套 59e5233 口径，骨干换成**稀疏路由 MoE + TransformerBlock 专家**（与 SAC+AZ (稀疏MoE+TB专家) 同骨干）—— 用来把"骨干"与"口径"两个变量分开比 | 16 次模拟，约 160 ms |

> **"只换骨干"的三个配对**（PPO+MCTS 的 MLP/TB 专家、SAC+AZ 的 MLP/TB 专家、
> SAC+AZ-59e5233 的 MLP/TB 专家）刻意共用同一个类与同一份搜索/训练/自检代码，构造时传
> 不同的骨干枚举 —— 界面上并列，就是为了能直接对弈比较，而不是维护两份会漂移的实现。
> 它们的权重文件、agent 名、参数量都不同；PPO 那对**交叉载入会被结构指纹当场拒绝**
> （`test_match` [2.14] 钉住）。
>
> **SAC 的两支与上面相反**：SAC+AZ 与 SAC+AZ (59e5233 行为还原版) 是**两份独立实现 +
> 两个互不继承的 C++ 类**（`SACAZAgent` 与 `SACAZLegacyAgent`，后者自带一份 SAC 实现，
> 目的是让前者的默认值/实现改动渗不进还原版），差别在口径。两者的**参数结构完全相同**
> ⇒ 结构指纹**挡不住**串权重，所以隔离只能靠 ① 不同的类、② 不同的文件名
> （`weights/sacaz_agent*` vs `weights/sacaz_old_agent*`；还原版的 TB 专家骨干那一支是
> `weights/sacaz_old_moe_agent*`），并由 `test_match` [2.11]（文件名清单不同）与
> `test_sacaz` [14]（口径、以及"那一批开关连成员都没有"的探针）钉住。口径差异的完整表在
> `src/sacazlegacyagent.h`，排查记录在 `docs/sac_regression_2026_09.md`。

所有可在线训练的 agent 都遵循同一条决策流程（仿 snakeAI）：**先探索环境 + 预训练一次，
再基于当前局面决策**（`AgentBase::exploreAndTrain()` + `src/agentrollout.hpp`）。探索全程
用 `moveForward/moveBack` 试走并原样回退，对真棋局零副作用（`test_pretrain` 钉住）。

---

## 强化学习内核 `RL_CORE`

`src/rl/` 是纯 C++（**不依赖 Qt**），编译成一个静态库 `RL_CORE`，界面与测试共用。

* **张量 + 自动拼装的反向传播**：`Tensor`、`Net`、`Layer<Fn>`、`LayerNorm`、`MHA`、
  `TransformerBlock`、经典 RL 算法（DQN / DPG / PPO / SAC / BCQ / DDPG / …）
* **SIMD 加速**（`simd_ops.hpp` 分派：标量 / SSE2 / AVX2，内核取自 N-spirits）：
  `Tensor::MM::ikkj` 13.274 → **0.053 ms**（250×）、GEMV 1.059 → **0.099 ns/MAC**（10.7×）、
  归约 8.2×。**反向的 GEMV 没有 SIMD**（0.991 ns/MAC，比前向慢 9.7×）—— 这是训练步的
  真正瓶颈，见 `docs/issues_review.md` C8
* **稀疏路由 MoE**（`sparse_moe.hpp`，本工程自己实现）：上游 `moe.hpp` 的 `forward` 是
  **稠密**的（16 个专家全算），在 d=1260 上不可用；稀疏版只算门控选中的 top-k 个。
  同样 28.7 M 参数下实测 **42.0 → 10.4 ms/模拟（4.1×）**，并配负载均衡辅助损失防坍缩
* **权重文件 v2**：无损（base64 + CRC32）、结构指纹、原子写入、旧格式兼容。
  实测往返**逐比特相同**、体积 **1.78×**、坏文件一律拒绝且不改动网络

---

## 测试与验证

### `ctest`（13 个套件，全过）

```bat
ctest --output-on-failure
```

| 套件 | 盯什么 |
|------|--------|
| `test_ab` | Alpha-Beta 搜索正确性与基准 |
| `test_mcts` | MCTS 四阶段 + 树规模（较慢，约 4 分钟） |
| `test_rules` | 棋规回归：走法数 / 应将 / 自杀 / 照面 / 将杀 / 重复 / 限着 + **`Chess::Result` 与 `Stone::Color` 的枚举换算**（两者数值撞号，手写比较会把红胜记成黑胜） |
| `test_diag` | **诊断指标的解析验证**（68 断言，秒级）：熵 / CE / KL / explained variance / 变异系数 / 校准分桶 / `qForParent` 的符号 / CSV 列数与表头一致 / `rootDiag` 的不变量（**根访问数之和 == 模拟次数**）/ **搜索能不能看见一步杀** |
| `test_pretrain` | **探索不得改动真棋局**（逐字段比对） |
| `test_match` | arena 统计（交换先后手 / 比分归属 / 判和 / 中止）+ 每个 agent 的**训练损失上报** + **即时奖励符号约定** + **每手奖励进度与局末奖励同账** + **曲线"换一批线"不残留空线** + **必输局面仍返回合法走法** + 十六个 agent 的自检与**两支 SAC 的权重文件名必须不同**（[2.11]）+ 五条后台训练支路（含 59e5233 版 SAC 的派生类 clone）+ **自动保存 × 后台训练并发**（[2.15]/[2.16]） |
| `test_grad` | **有限差分核对 SIMD 之后的解析梯度** + MM 内核"累加 vs 覆盖"语义探针 |
| `test_weights` | 权重格式：逐比特往返 / 坏文件拒绝 / 失败不改动网络 / 老格式兼容 |
| `test_sparse_moe` | 稀疏不变量 / 与上游 `MOE` 的等价性 / 反向有限差分 / 辅助损失 / `MOE` 的**专家模板参数**（默认 TB 保兼容、`MlpExpert`、`Layer<Fn>`）与 `copyTo` 是否真的复制专家 |
| `test_scaledconcat` | `ScaledConcat` 的结构不变量：**旧实现的门控上界 e¹ 与新实现的选择性**（有效路数）、门控与特征**逐位解耦**、参数与**输入梯度三条通路**的有限差分、三种专家模板参数、保维残差 / 存取往返 |
| `test_sacaz` | 掩码 softmax 雅可比 / 走法合法性 / 软价值 α 恒等式 / 四种骨干 |
| **`repro_concurrency`** | **"对弈 × 后台训练"并发的最小复现**（对弈跑在独立线程 + 每手刷自检 + 后台训练 + 结束后保存；**故意不进 ctest**，见 `docs/issues_review.md` C24 与零之二点二十五）：修前秒崩 `0xC0000374`（堆损坏），修后"全部跑完，没有崩溃"。用法 `repro_concurrency.exe [轮数] [每局手数] [agent枚举值]` |

`test_ppomcts` 也在这 13 个里（盯 PPO+MCTS 的策略目标 / 回放池 / 镜像增广 / 稀疏策略头 /
置换表 / **PUCT 的 Q 符号** / **`loadModel` 必须报告真实结果** / **根噪声与出招温度**）。

另外有 **`bench_moe`**（骨干 A/B/C/D 等时间对弈基准）**故意不进 ctest** —— 它跑真实对局、
依赖随机开局，放进去只是偶发失败：

```bat
build\...\bench_moe.exe --games=4 --plies=30 --budget=60 --pretrain=3
```

**`bench_ppo_vs_ab`** 是 PPO+MCTS 对 Alpha-Beta 的**静默**（无界面、不弹窗、不写权重）
对弈基准，同样是手动跑的：交换先后手、随机开局、可选**等时间**（`--budget` 先标定
ms/模拟再反推每步模拟次数）。**不进 ctest** 的理由同上，而且默认双方是随机初始权重，
几局之内翻转结论是常事。

```bat
build\...\bench_ppo_vs_ab.exe --games=20 --plies=120 --budget=106     # 等时间
build\...\bench_ppo_vs_ab.exe --games=6 --sims=80 --depth=4           # 固定预算
build\...\bench_ppo_vs_ab.exe --load=weights/ppomcts_agent.dat        # 用 GUI 存下的权重
```

实测（2026-09，未训练的随机权重，20 局等时间）：**PPO+MCTS 0 胜 / Alpha-Beta 20 胜，
全部被将死，平均 36 手**；双方耗时 139 vs 143 ms/步。加 20 局自对弈热身仍是 0 胜
（局均手数反而更短）。这与「已知限制」第 1 条一致 —— **它量的是链路的下限，不是棋力**。

**`bench_ppo_mt`** 量的是**多线程分身训练的吞吐**（不是棋力）：串行基线、worker 数扫描、
以及"每线程每模拟成本"。同样**不进 ctest**（跑真实自对弈、依赖线程调度，耗时以分钟计）。

```bat
build\...\bench_ppo_mt.exe --games=12 --sweep=1,2,3,4,6 --sims=80 --repeat=2
build\...\bench_ppo_mt.exe --games=6  --sweep=1,2,3,4,6,8 --batch=2000000000   # 只搜索
```

实测数字与**它暴露出的并行上限问题**（搜索受访存带宽限制，不是 learner 拖后腿）记在
[`docs/issues_review.md`](docs/issues_review.md) 的"零之二点十三"，
设计与成因分析见 [`docs/agents_design.md`](docs/agents_design.md) §17.6。

**2026-09 更新（R1，策略头只算合法列）**：那条带宽上限被推后了一截 —— 推理侧改成
`PPO::actionMasked`（`forwardTrunk` + 只算合法着法那几行的 logit + 在合法集上 softmax），
每模拟权重流量 5.81 MB → 3.75 MB。实测：单线程每步 18.3 → 11.0 ms（1.64×）、多线程每线程
ms/模拟 1.4–2.7×、最优配置吞吐 0.87 局/s ≈ 2.0× 串行基线（原来还不到 1×）。语义等价性在
`test_ppomcts` 的"测试11"里逐元素验到 3.7e-09；副作用是先验归一化口径变化带来约 2% 的
选点变化（`bench_policy_agreement --ab=1` 量出来的 97.5–98% 一致率）。全部数字、纠正过的
带宽拆账与 A/B 命令行见 [`docs/training_optimization.md`](docs/training_optimization.md) §9.4。

**同一轮的 R1.5（内核收尾，四项）**：训练步最慢的反向 GEMV（`ei = wᵀ·e`）补上了对称内核
（**1.060 → 0.070–0.098 ns/MAC ≈13×**，`learnFromReplay(64,2)` **490 → 304 ms**）；
SIMD 的 `ikjk`/`kijk` 从"覆盖"改成与标量一致的"累加"（**正确性**：多列输入曾会静默丢梯度）；
`MM::*` 入口加了 `NDEBUG` 下的形状契约断言（用 `/UNDEBUG` 重编测试跑通，并当场抓到 `lstm.cpp`
一处越约调用）；DQN 的 Q 头从 `Sigmoid` 改成 `Linear`（奖励含负项，旧头结构上装不下负 Q）。
细节与可复核命令见 [`docs/issues_review.md`](docs/issues_review.md) 的"零之二点十五"。

**再同一轮之后（B-5，置换表 + 子树复用）**：PPO+MCTS 不再每 ply `nodes.clear()` —— 子局面按
Zobrist 键登记进置换表，下一步落子走到同一局面就直接**换根复用**那棵子树（子树/先验/访问
计数/Q 全接着用）。跨局靠"≤2 步可达"判据与显式 `resetSearchTree()` 失效，`treeReuse=false`
可逐字复现旧行为（A/B 用）。**实测是"机制成立、决策中性"**：命中率 ~90%、每手白拿 ~10% 有效
模拟，但 80/400 模拟下复用开/关的选点 **100% 相同**（1200 模拟下 97.5%），对 AB 的一致率三档
预算完全相同 —— 因为 `getPUCT` 给未访问孩子 +∞，搜索先把 40 多个孩子全铺一遍，深挖余量只有
`sims − 分支数`。数字、方法与"为什么中性"见 [`docs/issues_review.md`](docs/issues_review.md)
的"零之二点十六"（`test_ppomcts` 测试12 + `bench_policy_agreement --reuse-ab=1`）。

**R2（训练侧也只算合法列）**：R1 只改了推理；R2 换的是**学习问题** —— 训练时的 softmax
分母也只覆盖该局面的合法着法（Z ≡ 1）。`PPO::accumulateGradSparse` + `Net::backwardFrom`
把"头只算合法列 + 只更新合法行 + 梯度只从合法行往下传"做实；解析梯度对着中心差分验到
1.1e-3（头）/3.5e-4（骨干），且**非法行的梯度恰好为 0**（全量口径下是 1.9e-04）。副作用确认：
R2 权重上"全量 softmax"已无意义（Z≈0.005，非法槽位从未被训练）—— 所以 `PPO::action()`、
`sparsePolicyHead=false`、`--ab=1` 这些不要在 R2 权重上用。**c_puct 重扫的结论是"扫不出
落点"**（80 模拟下 6 个取值选点逐位相同，400/1200 下落进 ±6% 噪声），保持 1.414。缺口：
BC 蒸馏路径仍是旧口径。见 [`docs/issues_review.md`](docs/issues_review.md) 的"零之二点十七"。

### 诊断仪表盘 `bench_diag`（2026-09）

棋力是**滞后指标** —— 本轮实测过：4 局对 AB 的得分率三次都压在 **0%** 地板上，这个协议
什么也答不了。`bench_diag` 量的是**中间量**，让"哪一层设计错了"当场可读：

```bat
:: 小网络冒烟（几十秒）
build\...\bench_diag.exe --games=2 --plies=40 --sims=40 --hidden=16 --expert=16 --tactics=20

:: 真实权重 + CSV（逐手宽表 + TensorBoard 长表）
build\...\bench_diag.exe --games=6 --plies=80 --sims=80 --tactics=20 --tactic-sims=80 ^
    --dirichlet-ab=10 --print-root=1 --csv=diag --load=<工作区>\weights\ppo_bc_d4
```

| 区块 | 量什么 |
|---|---|
| 搜索（老师） | 分支数 / 展开覆盖率 / top-1 访问份额 / 访问熵 / 先验熵 / **KL(访问‖先验)** / **吃子 vs 退让的 Q 分组** |
| 行为 | 每手材质变化 / 选到吃子的比例 / 叫将率 / 被将率 / **每手 V 增益**（附 `std(V)` 与 `corr(V_before,V_after)`，否则常数偏置会被读成"越走越差"） |
| value 校准 | **EV（explained variance）** + 分桶表（预测区间 → 实际胜率）。全和棋时显式提示"**EV 无定义**"而不是报 0 |
| 战术题库 | **自动生成**一步杀 / 白吃子题（随机造局面 + 扫描合法着法自校验，不需要手工摆局面）；命中率是硬指标 |
| Dirichlet 有效性 | 同一局面开/关根噪声，比较吃子着访问份额 —— 判断"噪声救不救得了" |
| 自对弈生态 | 红/黑/和比例 / 平均手数 / 开局种类数（第 8 手局面哈希去重） |

**它上线当天就抓到一个真缺陷**：一步杀命中 **0/20 = 0%** —— 根因是三个搜索入口都无条件
用 critic 估叶子、**从不判终局**（`SACAZAgent` 有 `terminalValue()`，PPO 这条没有）。
修 `evaluateLeaf()` 后 **20/20 = 100%**，而自对弈读数**一点没变**（外科式修复的回归证据）。

同一份仪表盘给出的**当前瓶颈定位**：KL(访问‖先验)=1.32（搜索不是白跑），但 **EV = −0.0293 < 0、
283 个样本里 281 个落在同一个校准桶** ⇒ V 是常数偏置 ⇒ **Q(吃子)−Q(退让) = −0.098** ⇒
白吃子命中率 10%。而噪声实测**救不了**（开/关差 −0.0037）。
**结论：下一轮的杠杆在 value，不在搜索参数。** 全部数字与三条"会让诊断说谎"的坑见
[`docs/issues_review.md`](docs/issues_review.md) 的"零之二点二十"。

### 无界面训练器 `train_ppo` 与锚点评测 `bench_anchor`（P0，2026-09）

这两个工具补的是同一件事的两半：**"这一轮自对弈到底在下什么样的棋"** 与 **"棋力到底涨没涨"**。
方案与优先级见 [`docs/rl_plan_optimized.md`](docs/rl_plan_optimized.md)（Part 6 是实施记录）。

```bat
:: 常驻训练（一个进程连跑 K 局, 每局零磁盘往返 —— GUI 路径是每局 3 趟 x ~555 MB 权重往返）
build\...\train_ppo.exe --games=20 --sims=400 --moves=60 --opening=8 --csv=run1

:: P1.1 的 PBRS 消融: 同一批参数跑两次, 比报告里的吃子与和棋构成
build\...\train_ppo.exe --games=40 --opening=8 --no-shaping            --csv=shaping_off
build\...\train_ppo.exe --games=40 --opening=8 --shaping-alpha=0.25    --csv=shaping_25

:: 锚点对局（固定开局集 + 换先手成对计分 + 95% 区间; 对 AB 可等时间）
build\...\bench_anchor.exe --a=weights/run_10k --b=weights/run_5k --openings=20 --plies=8
build\...\bench_anchor.exe --a=weights/run_10k --ab-depth=4 --budget=100 --openings=20
```

| 工具 | 量什么 |
|---|---|
| `train_ppo` | 和棋**分原因**（三次重复 / 60 回合自然限着 / **台架截断**）/ 手数 / 吃子 / 开局多样性 / 吞吐（局/小时）/ value MSE / MoE 专家负载；逐局 + 汇总写进 TB 长表 CSV |
| `bench_anchor` | 固定开局集（确定性生成 + FNV 指纹，**指纹不同即不可比**）→ 每个开局换先手下两局 → 得分率、**Elo 差与其 95% 区间**、和棋构成、双方 ms/步；`--budget=MS` 把 PPO 的模拟次数按 AB 实测耗时标定成等时间 |

**它们第一次运行就报出了最重要的一件事**：60 ply 手数上限下，训练里的"和棋"几乎全是
**台架截断**（`截断 1.000`），而规则里的 60 回合自然限着需要 120 半回合 —— 在 GUI 的训练
配置下**根本不可达**。也就是说"和棋率高"在这套台架里首先是**手数上限**问题，不是规则问题、
更不是"棋力到顶"；`Aggregates` 现在会直接打印这个判读。另外 `bench_anchor` 会明确说出
"6 局分辨不了 Elo，区间跨过 50% —— 没测出差别"，而旧的 4~6 局随机开局 bench 只会给一个
看起来像结论的数字。

### DQN+MCTS 表示能力探针 `probe_dqnmcts_aliasing`（2026-09）

**它回答的是"这个 agent 值不值得继续训"，而不是"它有多强"。** 起因是一个具体的困惑：
`DQN+MCTS` 报出"50 胜 0 负 50 和"、训练损失 22，而这些数字**无法**说明模型学没学会下棋。
探针把三类**与训练量无关**的结构事实量出来：

```bat
cmake --build <build> --target probe_dqnmcts_aliasing
build\...\probe_dqnmcts_aliasing.exe --positions=48 --plies=24
```

| 段 | 量什么 | 实测（本机） |
|---|---|---|
| [1] | **动作别名**：同一个局面内的合法着法 -> 128 个 Q 槽位 | 标准开局 44 个着法只落在 **38** 个槽位上（挤掉 6 个，最挤槽位背 3 个着法）；中局 96 个局面平均挤掉 **5.16** 个，最挤槽位 **4** 个着法。跨局面累计碰撞率只有 0.03（**这一条否掉了"128 槽位是灾难"的推测**：槽位本身够用） |
| [2] | **状态不可分性**：棋子分布相同、规则状态不同的局面，`encodeState` 是否区分 | 走子方 / 重复 1 次 vs 3 次 / 无吃子 0 手 vs 120 手 —— **三种情况编码逐字节相同**。90 维 = 一格一个子力值，规则上下文通道数为 **0**（PPO 是 19 平面） |
| [2d] | **终局检测口径**：`evaluateLeaf` 用 `isGameOver()` 判终局 | 已判和的局面上 `isGameOver()=COLOR_NONE` 而 `getResult()=RESULT_DRAW` ⇒ 判和与将杀**都不是终局**，叶子继续走 `max_a Q(s,a)` |
| [3] | **终局奖励通道**：自对弈里有多少局真走到了终局 | 6 局 x 200 手：`isGameOver` 看见 **0** 次终局，`getResult` 看见 1 次，**5 局撞手数上限** |
| [4] | **回放池里的"新信息"** | 1440 个 ply 里互不相同的局面 **1426** 个（去重率 **99.0%**）；`Transition` 约 1.23 KB/条 ⇒ 4096 条约 5.0 MB、20000 条约 24.6 MB |

> **[4] 段为什么存在**：它回答"把 DQN 的 replay 从 4096 提到 20000 能不能加速收敛"。
> 实测的 99% 去重率说明**容量装的确实是新信息**（否掉了"只是多存重复"这个猜测），但结论
> 仍然是**不会更快**，因为需求侧用不满：一局写 60 条、却抽走 15x32 = **480 条**，而 4096 条
> 要 **68 局**才淘汰一轮 —— 前 68 局里容量对训练轨迹的影响严格为零。真正的杠杆是
> `docs/agents_design.md` §10.4 那条**没做**的建议：「DQN 系"每步一次更新"破坏回放缓冲的
> i.i.d. 假设，建议改成**只写回放、攒批更新**」。

**这几条合起来是"前置闸门"**：过不了就不该拿这个 agent 的训练曲线（包括"50 胜"与损失 22）
去推断棋力。判据本身也在文档里：`docs/training_optimization.md` §10 与
`docs/agents_design.md` §20.4。棋力仍然只有 `bench_anchor` 能回答 —— 实测同一时期的
PPO 权重对 AB 深度 4 是 **0 胜 1 和 23 负**（Elo 差 −669），与"自对弈 50 胜"并不矛盾：
自对弈里的"胜"只是同一个网络赢了自己。

### 界面自动化（PowerShell + Windows UI Automation）

| 脚本 | 验证什么 |
|------|----------|
| `tools/verify_match_ui.ps1` | 真界面选 agent → 开局 → 断言比分/逐局明细/曲线有数据/**静默保存权重**/**双击放大窗与源控件逐字一致**；`-TwoMatches` 再打一场，断言奖励读数**仍然只有两条线**（每场换线不留空线） |
| `tools/verify_thinking_ui.ps1` | 采样像素：思考中状态条出现、空闲/结束后干净 |
| `tools/verify_busy_ui.ps1` | 启动载入权重时弹沙漏、载完收起（**不残留**） |
| `tools/verify_eager_load.ps1` | **启动时加载所有模型**的两个后果：启动沙漏出现并收起；首次使用稀疏 MoE 变体**不再**弹沙漏（回到懒加载就会 FAIL），且对局确实在推进 |
| `tools/verify_agent_combo.ps1` | **每个 agent 都能在界面上被选中**：把 `src/mainwindow.cpp` 的 `kAgents` 解析出来当期望值，展开三个下拉框，逐条断言它们**可见**（只"在模型里"不算 —— Qt 的 `maxVisibleItems` 默认 10，第 11 项曾被折叠在滚动区里，见 `issues_review.md` C23） |
| **`bench_sac_learn`** | **"SAC 边下边学"的受控复现**：每手 `exploreAndTrain()` + `selectMove()`（与界面 `preTrainThenDecide` 逐条同协议），对 MCTS 打 N 局，报**胜负和 / 和棋成因（手数上限·60 回合·重复）/ 分胜负的终局类型（吃将·将死·困毙）/ 两种口径的奖励累计 / 损失曲线点数与均值最大值 / 训练后 critic 的 \|Q\| 尺度与策略熵**。带 `--legacy`（59e5233 口径）、`--reward-shape=0\|1\|2`、`--no-search-learn`、`--entropy-ratio/--alpha-lr/--clamp/--huber/--no-sparse-leaf` 等消融旋钮。详见 `docs/sac_learn_reward_2026_09.md` |
| `tools/verify_app_icon.ps1` | ICO 结构 / 字形真的渲染 / 运行时加载 / exe 图标是"我们的" |
| `tools/make_app_icon.ps1` | 生成程序图标（改了能重跑，二进制资源可审） |

> 脚本有两个"血泪规则"写在注释里：**无 BOM 的 .ps1 会被 Windows PowerShell 按 ANSI 解码**，
> 所以要么纯 ASCII（窗口名用码位拼），要么带 BOM；两者混用会直接解析失败。

---

## 项目结构

```
chess/
├── CMakeLists.txt          # 构建 (Qt6 Widgets+Sql; src/rl 编成 RL_CORE 静态库)
├── build_main.bat          # vcvars64 + cmake 配置/构建
├── src/
│   ├── main.cpp            # 入口 (设置程序图标)
│   ├── mainwindow.*        # 主窗口: Android 风格右侧面板 (曲线/明细/对弈设置)
│   ├── chessboard.*        # 棋盘控件 + AI 线程 + Agent 对弈 (arena)
│   ├── chess.* stone.* pos.*   # 棋规 / 棋子 / 坐标
│   ├── thinkingindicator.* # "AI 正在思考" 指示器 (沙漏/粒子/呼吸灯)
│   ├── busydialog.*        # 载入/保存权重时的"请稍候"弹窗 (复用上面的沙漏)
│   ├── metricsview.*       # 自绘折线图 + 双击放大窗口
│   ├── *_agent.* / abagent.* / mcts.* / evagent.*   # 9 个 agent
│   ├── sacazlegacyagent.h                          # 第 10 个 agent: 59e5233 还原版 (派生于 sacazagent)
│   ├── app.rc app.ico app.png res.qrc              # 程序图标与资源
│   └── rl/                 # RL 内核 -> RL_CORE (纯 C++, 不含 Qt)
│       ├── tensor.hpp net.hpp layer.h ...          # 张量/网络/层
│       ├── simd_ops.hpp simd/ cpuinfo.hpp          # SIMD 分派与内核
│       ├── moe.hpp sparse_moe.hpp                  # 稠密 / 稀疏 MoE
│       ├── diag.h                                  # 诊断指标核心 (熵/CE/KL/EV/校准 + 和棋原因分桶 + CSV)
│       └── dqn.cpp dpg.cpp ppo.cpp sac.cpp ...     # RL 算法
├── test/                   # 13 个 ctest 套件 + 多个手动基准（含 bench_diag 诊断仪表盘）
│                           #   (bench_moe / bench_ppo_vs_ab / bench_ppo_mt /
│                           #    bench_policy_agreement / bench_ppo_sims / bench_diag /
│                           #    train_ppo / bench_anchor / probe_dqnmcts_aliasing)
│                           #   注意: test_dqnmcts 有断言 (16 条) 且按失败数返回退出码,
│                           #   但因为它要跑 ~340 s, 刻意不注册进 ctest
├── tools/                  # 界面验证脚本 + 图标生成
└── docs/                   # 设计/审查/同步/理论分析 (见下)
```

代码量：`src/` 约 **2.9 万行**，`test/` 约 **0.5 万行**。

---

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/agents_design.md`](docs/agents_design.md) | 各 agent 的设计与实测；§12 参数量理论分析；§13–16 界面可视化/EVAB 修复/沙漏等待/静默保存与图标；**§17 PPO 系列训练效率改造（P1–P7）与实测**（梯度累积/回放池/访问分布目标/镜像增广/多线程分身及其访存瓶颈）；**§20 完备 MDP 的公共化与推广**；**§21 DQN+MCTS 的表示闸门与自检面板**（探针四个读数 + 为什么损失与自对弈胜率都答不了"值不值得继续训"） |
| [`docs/issues_review.md`](docs/issues_review.md) | **问题清单与修复进度**（A/B/C 编号）、优化方法汇总（含实测数字）、当前待办。**零之二点十九**：PPO+MCTS 正确性审计（PUCT 的 Q 符号反了 / `loadModel` 静默成功 / 枚举混比 / 模拟数退化 / 搜索看不见将杀）；**零之二点二十**：诊断指标矩阵与当前瓶颈定位（EV<0 ⇒ 该修 value 而不是搜索参数） |
| [`docs/training_optimization.md`](docs/training_optimization.md) | **训练流程优化总结（Phase 0–5）**：奖励量纲摆正、自举、势能塑形（PBRS）、棋盘局面价值评估（将安全/空间/机动性）、搜索展开按先验选；每阶段的实测数字、两档评估的工程决策、明确列出的未做项。**§9.5** 是 2026-09 的诊断矩阵与负面结果 |
| [`docs/rl_sync.md`](docs/rl_sync.md) | 与上游 snakeAI `rl/` 的同步、chess 侧的差异、SIMD 之后梯度是否仍正确 |
| [`docs/xiangqi_capacity.md`](docs/xiangqi_capacity.md) | "多少参数量才能覆盖象棋求解空间"（~10⁴⁰ 参数 → 物理上不可能） |
| [`docs/analysis.md`](docs/analysis.md) | 文件树与模块分析 |
| [`docs/rl_plan_optimized.md`](docs/rl_plan_optimized.md) | **三轮讨论收敛出的可执行方案**（过强对手/信心崩塌、和棋率、更新器与回放容量）：逐条"原建议 → 优化后"、数字按本机吞吐重算、明确不做清单、验收标准；**Part 6 是 P0 实施记录**（改了什么 / 怎么跑 / 实测数字 / 下一步） |
| [`docs/agent_evab_design.md`](docs/agent_evab_design.md) | EVAB 专篇 |
| [`docs/sac_regression_2026_09.md`](docs/sac_regression_2026_09.md) | **SAC 与 59e5233 的回归排查**：根因（隐层激活被换成 `TanhNorm<Sigmoid>`，-26 点）、等价性证明（逐手相同 + 黄金基准）、§7 方法学（评测器不可复现等） |
| [`docs/sac_learn_reward_2026_09.md`](docs/sac_learn_reward_2026_09.md) | **"SAC 边下边学"的实测复核**：用户两份 CSV 的两个口径陷阱、奖励塑形实验（含"最终棋子数"为什么在即时奖励上不可实现）、**§9 训练为什么有害 + α 口径定位（37.5%→82.5%）** |
| [`docs/session_2026_09_sac.md`](docs/session_2026_09_sac.md) | **本轮会话交接件**：P1~P13 问题清单（现象/根因/证据/状态）、①~⑩ 优化清单（依据与效果）、**四项待验证清单（含命令与验收标准）**、方法学教训 |

---

## 已知限制

诚实列一下（细节与实测数据都在 `docs/`）：

1. **RL agent 棋力弱**。所有权重都是随机初始化开始训练的；界面上未训练的 SAC+AZ 会被
   Alpha-Beta 在十几手内将死。这个工程的价值在**链路完整、可训练、可测量**，不在棋力。
2. **稀疏 MoE 没有棋力结论**。等参数下稀疏路由确实快 3.85 倍（42.0 → 10.4 ms/模拟），
   但 A/B/C/D 等时对弈**全是和棋**（参赛者都是随机权重），所以只能说"机制通了、能赢棋了"。
3. **`bench_moe` 的对局样本太小**（每配置几局），单局偶然性足以翻转结论 —— 文档里标明了
   哪些结论能下、哪些不能下。
4. **反向 GEMV 没有 SIMD**（`kikj`），训练步的成本几乎没被优化到，见 issues_review C8。
5. **偶发的无效走法**：arena 里约每 200 次决策出现 1 次 agent 返回 `valid == false`
   （arena 会用第一个合法走法兜底并计数，不影响胜负归属），**根因未定位**，见
   `docs/agents_design.md` §14.9。
6. **MM::ikjk/kijk 的 SIMD 内核是"赋值"而标量是"累加"**：当前所有调用点 kdim=1
   走不到那条路径，但一旦改成多样本批量就是静默丢梯度的坑，见 issues_review C7。
7. 训练量不足以谈棋力：本机实测 `learnBatch(32)` 在 28.7 M 参数的骨干上要约 0.9 s，
   "每个参数喂一个样本"这种极度乐观的预算下也要 10⁹ 秒量级，见
   `docs/xiangqi_capacity.md`。

---

## 说明

* `src/rl/` 的内核与上游 snakeAI 工程的 `rl/` 同源（同步过程与差异记在
  `docs/rl_sync.md`）；本工程在其上做了 SIMD 分派、稀疏 MoE、权重格式 v2 与若干正确性修复。
* 权重文件默认放在运行目录的 `weights/` 下（`chess.exe` 的工作目录）。
