# 混合对手训练：MCTS 50% + AB-L1..L4 各 12.5%（均匀分布采样）（2026-10 第四轮）

> 用户口径（原文）：**"实现 mcts, abagent-1,2,3,4 混合对局, mcts 对局数量占 50%,
> 其余 abagent 平分, 按照均匀分布采样选择 agent 对弈训练"**。
>
> 一句话：给 PPO 的**自对弈训练**加一个**对手池** —— 每局开头按均匀分布抽一个对手
> （MCTS 50%，AB 深度 1/2/3/4 各 12.5%），对手的手**只推进局面、不进学习方的样本**，
> 并且**每局换先手**。默认关（纯自对弈，与改动前逐位相同）；界面后台训练已打开它。
>
> 代码：`src/ppomcts_agent.h/.cpp`（池子 + 训练循环里的对手分支）、
> `src/ppo_train_opponents.cpp`（池子的**构造**，单独一个 TU，理由见 §4）、
> `test/test_mixed_opponents_main.cpp`（**33 条断言**，秒级进 ctest）、
> `test/train_ppo_main.cpp`（`--opponents=mixed` 等四个参数 + 直方图读数）、
> `src/chessboard.cpp`（界面后台训练打开池子）。

---

## 1. 语义（三条，缺一条就不是这句话的意思）

| 口径 | 实现 | 为什么必须这样 |
|---|---|---|
| **"对局数量占 50%"** | 采样发生在**每局开头**，一局一个对手 | "数量"说的是**局**的比例；按手采样会变成"每手都可能换对手"，那不是"混合对局" |
| **"均匀分布采样"** | 一次 `U(0,1)` 落在**权重累积区间**的那一段：MCTS `[0,0.5)`、AB-L1 `[0.5,0.625)`、…、AB-L4 `[0.875,1)` | 权重与区间一一对应，且用 `RL::Random`（与 `setSeed` 同一颗引擎）⇒ **同一 seed 下对手序列可复现** —— 本工程所有随机性的统一口径 |
| **"其余 abagent 平分"** | `abWeight = (1 - mctsShare) / abMaxDepth`，默认 `0.5/4 = 0.125` | 参数化：`--mcts-share` / `--ab-max-depth` 想换成 3 档 AB 或 70/30 都只是参数 |

## 2. 两条边界（决定了这个功能的适用范围，不写清就会被误用）

### 2.1 对手的手**不进学习方的样本**（只推进局面）

自对弈里"对手那一半"是学习方自己按策略采样出来的，所以把它记成学习方的动作转移是
**on-policy** 的；换成真对手之后，同一份记录会变成 **"off-policy 却打着 on-policy 的标签"**
—— 对 PPO 这类需要行为策略的算法是实打实的偏差（那些动作从来没有以那个概率被采样过）。

所以对手的手只做一件事：**推进局面**。学习方自己的转移因此落在"被真对手应手之后的局面"上，
回报里带着对手的影响 —— 这正是"从对手身上学"的内容。
（与 `aiagent.h` 的 `OpponentPolicy` 逐字相同的纪律；那里还写了"把对手着法当样本收进来"
为什么需要另一套处理：要给网络对手特征、或按 off-policy 方法，否则价值目标不可辨识。）

### 2.2 每局换先手

奇数局对手执红（`trainSelfPlay` 的起手方是黑）。不换的话红黑两侧的样本量不对称，
而象棋的先手优势很大 —— 判据与 `bench_agent_arena` 的"交换先后手"同源。

## 3. 验证（`test_mixed_opponents`，**33 条断言 / 0 失败**，秒级进 ctest）

| 段 | 断言 | 实测读数 |
|---|---|---|
| [1] | 池子构成：MCTS 0.5 + AB-L1..L4 各 0.125、名字与深度对得上、权重之和 == 1 | `MCTS(800)=0.500  AB-L1..L4=0.125` |
| [2] | **采样直方图**：200 局，每个对手的局数落在**二项分布 ±5σ** 内（n=200 时 MCTS 的 σ≈7.07、AB 各 ≈4.68） | MCTS **100**（期望 100）、AB-L1..L4 = 17 / 28 / 29 / 26（期望各 25，全在区间内） |
| [3] | 每局换先手（第 0 局黑、第 1 局红、第 2 局又黑） | 三条都过 |
| [4] | **对手的手不进样本**：`maxMoves=1` 时，纯自对弈"学习方进轨迹 20 手 / 20 局"；混合对局"对手 10 手 + 学习方 10 手"，两者之和 == 总手数 | 20 / (10+10) ✓ |
| [5] | 默认关：池子空、`mixedOpponentGames()==0`、20 局全是学习方自己的手 | ✓ |
| [6] | 参数越界被**纠正**而不是静默换分布：`mctsShare=2` ⇒ 1.0（池里只剩 MCTS，**权重 0 的档位不建**）；`mctsShare=-1` ⇒ 0.0（只剩 AB，各 0.25）；`abMaxDepth=0` ⇒ 只剩 MCTS | ✓ |

> 这套测试跑的是**廉价骨干**（MLP 专家，每局 1 手 1 模拟），所以几百局只要几秒 ——
> 它量的是**采样分布**，不是棋力。放进 ctest 的理由：采样比例错一位（少一档 AB、
> 或 MCTS 抽成 100%）**不报错、损失曲线也看不出来**，而它正是用户口径的核心。

## 4. 一个实现上的决定：池子的构造单独一个 TU

构造对手池需要 `MCTS` 与 `ABAgent` 两个类（构造函数与 `findBestMove` 定义在
mcts.cpp / abagent.cpp）。把这段写进 `ppomcts_agent.cpp`，会让**15 个**已经链接它的目标
（`train_ppo` / `bench_ppo_*` / `test_diag` / `bench_ppo_sparse` / …）突然都需要再加两个
`.cpp` —— 而其中大多数根本不用这个功能。

所以：**池子的构造放在 `src/ppo_train_opponents.cpp`**，只有真正用到它的目标加那三个文件
（`CHESS_CORE`、`train_ppo`、`test_mixed_opponents`、`repro_concurrency`）。
配套还有一处实现细节：`trainSelfPlay` 里**只走虚函数** `AgentBase::getBestMove` 驱动对手，
MCTS 的模拟预算由 `BudgetMcts`（MCTS 的薄壳，把 `getBestMove` 覆写成
`findBestMove(color, budget)`）实现 —— 否则 `ppomcts_agent.cpp` 又会引用 `MCTS::findBestMove`
这个符号，把 15 个目标的链接问题带回来。

> ⚠ **CMake 上的一个坑（这个仓库自己记过、本轮又踩了一次）**：新增一个 `.cpp` 必须加到
> **每一个直接列源文件的目标**上，不能只加 `CHESS_CORE` —— 只在 `--target chess` 下构建
> 是查不出来的。本轮漏的是 `repro_concurrency`（它直接列源文件），表现是
> `LNK2019 指向 enableMixedOpponentTraining`。见 `CMakeLists.txt` 那一处的注释。
> 另一条同源的坑：**只 build `chess` 不会重链 `repro_concurrency.exe`** —— 用旧二进制跑
> 新功能会得到"看起来通过"的假读数（本轮实测：旧 harness 完全跑不出池子）。

## 5. 代价与读数（实测）

| 项 | 读数 |
|---|---|
| MCTS 对手（800 次迭代，与界面 MCTS agent 同预算） | ≈ **1.0 s/手**（实测 1.24 ms/迭代）；一局 60 手上限里它大约走一半 ⇒ **每轮多 ~30 s** |
| AB-L1..L4 | 0 / 3 / 34 / 90 ms 量级 ⇒ 可忽略 |
| 端到端（`train_ppo --backbone=conv --opponents=mixed --games=12 --sims=16 --moves=20 --opponent-mcts-sims=48`） | 12 局：MCTS 4 / AB-L1..L4 = 3 / 1 / 1 / 3；对手共走 **112 手**，学习方进轨迹 **112 手**（各占一半，符合"换先手"）；`unmatched = 0`（对手着法全部合法）；value MSE 0.618（训练在动） |
| 界面路径 | `repro_concurrency 1 4 4`（`ChessBoard` + 后台训练 + 自动保存）⇒ `全部跑完, 没有崩溃`；后台训练 clone 已带池子（`BG_TRAIN_MIXED_OPPONENTS = true`） |

**界面这一支的开销**（写进 `chessboard.cpp` 的常量注释）：MCTS 800 次迭代 ≈ 1.0 s/手，
一轮 60 手多 ~30 s；关掉它只需把 `BG_TRAIN_MIXED_OPPONENTS` 改成 `false`，立刻回到
纯自对弈（与改动前逐位相同）。

## 6. 复现命令

```bat
:: ---- 断言 (秒级) ----
cmake --build <build> --target test_mixed_opponents
<build>\test_mixed_opponents.exe        :: 33 条断言; [2] 是 200 局的采样直方图

:: ---- 训练器: 混合对手 (默认 50% MCTS + AB-L1..L4 各 12.5%) ----
<build>\train_ppo.exe --backbone=conv --opponents=mixed ^
                      --games=40 --sims=120 --moves=60 --opening=8 --seed=20240901
::   想换比例/对手集:  --mcts-share=0.7  --ab-max-depth=3  --opponent-mcts-sims=200
::   对照臂 (纯自对弈): 去掉 --opponents=mixed
::   直方图在报告末尾: "混合对手采样直方图" + "学习方进轨迹的手数"

:: ---- 界面路径 (后台训练已带池子) ----
cmake --build <build> --target chess repro_concurrency   :: ⚠ 两个都要 build (见 §4 的坑)
<build>\repro_concurrency.exe 1 4 4
```

---

## 7. 界面接线（用户口径："接入 chess 程序"）

不只是"后台训练默认打开"，而是**界面上一等的开关 + 可见读数**：

| 项 | 位置 / 内容 |
|---|---|
| **勾选框** | 训练面板那一行（"模式 / 对手入训 / **混合对手池** / 动态奖励"）：`mainwindow.cpp` 用代码插入（与旁边那个"对手入训"同一手法 —— 不写进 `mainwindow.ui`，免得那个文件的编码被一串新控件搞乱）。名字 `mixedOpponentCheck`，勾选状态启动时由 `ChessBoard::isMixedOpponentTrainingEnabled()` 给出 |
| 提示框 | 完整写清：**MCTS 50% + AB-L1..L4 各 12.5%**、每局开头抽一次、两条边界（对手的手不进样本 / 每局换先手）、代价（MCTS 800 次迭代 ≈ 1.0 s/手 ⇒ 一局 60 手多 ~30 s）、关掉即回纯自对弈 |
| 开关语义 | `ChessBoard::setMixedOpponentTrainingEnabled(bool)`（`std::atomic`，与 `setPreTrainEnabled` 同一套线程口径）；参数 `setMixedOpponentConfig(mctsShare, mctsSims, abMaxDepth)` 带越界纠正；**默认值来自 `chessboard.cpp` 的四个常量**（构造函数里取一次）⇒ 只有一处决定"出厂口径" |
| 生效点 | 两个 PPO 变体（`AGENT_PPOMCTS` / `AGENT_PPOMCTS_CONV`）的**后台训练 clone** 在 `trainSelfPlay` 之前按运行期开关打开池子；MLP 那一支与 SAC/DQN 那几支**不受影响**（口径是"PPO 这条线的对弈训练"） |
| **每轮读数** | 每轮训练结束打一行：`[train] 混合对手: 共 N 局 \| MCTS(800)(0.500) x局/y手 AB-L1(0.125) … \| 学习方进轨迹 M 手`。**为什么要它**：采样比例错了（少一档 AB / 全抽 MCTS）在损失曲线上**完全看不出来**，这一行是界面路径上唯一能看见它的地方（工具侧的对应读数是 `train_ppo` 报告末尾的直方图）。走 Qt 的 `qInfo`，与仓库其它诊断（`[P1] 对手入训 = …`、`[weights] …`）同一条通道，在 Qt Creator 的 Application Output 里看 |

**验证（实测）**：

| 项 | 读数 |
|---|---|
| `chess` + `repro_concurrency` 构建 | 0 error（56 步；⚠ 必须两个一起 build，只 build `chess` 不会重链 harness —— 见 §4 的坑） |
| **界面路径真的在跑池子** | `repro_concurrency 1 4 4`（界面路径：`ChessBoard` + 后台训练 + 自动保存）**耗时 17 s → 50 s** —— 这一轮 60 手上限的后台训练里，对手（MCTS 800 次迭代 ≈ 1.0 s/手）大约走一半 ⇒ 与"池子生效"的预期吻合；不崩、`保存=1`。**关掉勾选框就回到原来的 ~17 s** |
| 人机路径 | `probe_hvai_flow`（构造 `MainWindow` ⇒ 新的勾选框代码真的跑到了）`失败项: 0` |
| 单元 | `test_mixed_opponents` **33 断言 / 0 失败**（§3） |

> **为什么"每轮一行 qInfo"就够了**：这一支的读数归 Qt 的日志通道（仓库里 `[P1] 对手入训`、
> `[weights]`、`[selfcheck]` 都是这么报的），而不是画在某个面板上；面板上真要看"这一轮用了
> 哪些对手"，要么新加一个列表控件、要么把对手名混进损失曲线的线名（那会污染按 agent 名分线的
> 既有读数）。**本轮选了 qInfo + 工具侧直方图这两条**，并把"没做面板控件"写在这里。

## 8. 诚实边界

1. **本轮没有量棋力**：这份交付是"训练分布"的实现与验证（采样比例、两条边界、
   端到端能跑通）。"混合对手训练出来的权重比纯自对弈强多少"需要按
   `docs/conv_resnet_agent_2026_10.md` §4 的协议（100 局 / 50 个配对开局 / 等模拟预算）
   跑一轮 A/B —— **没做**。
2. **对手是"脚本型"的**（MCTS + AB 的固定等级），不是"冻结的历史权重"；
   本工程另有一条冻结快照的机制（`frozenOpponentOverrideFor`），本轮没接进这个池子。
3. **AB 的深度口径是"ABAgent 的搜索深度"**（1..4），与界面上的 `AGENT_AB_L1/L2/L3` +
   `AGENT_ALPHABETA`(深度 4) 同源；MCTS 的预算显式给 800（= 界面 MCTS agent），
   可用 `--opponent-mcts-sims` 改小来换墙钟。
4. **界面默认开、库默认关**：库侧 `trainSelfPlay` 在不打开池子时一行额外代码都不执行
   （不画随机数、不建实例），所以既有的自对弈读数与改动前**逐位可比**；界面侧默认勾上（`BG_TRAIN_MIXED_OPPONENTS = true`），不想用就取消勾选。
