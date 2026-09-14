# chess 程序问题审查报告

审查对象：`E:\home\code\chess`（`src/` + `test/` + `CMakeLists.txt`）。
方法：逐行静态审查 + 实际构建与运行验证（Release / MSVC 2022 14.44 / Qt 6.9.2 / CMake 3.30）
+ 运行时插桩（ASan、有限差分梯度检查、真实窗口的像素/无障碍树采样）。

> 行号对应当前工作区版本。`src/rl` 已同步到 snakeAI 的上游版本，
> 同步内容见 `docs/rl_sync.md`。

## 本文怎么读

| 章节 | 内容 |
|------|------|
| 零 | **修复进度总表**（A/B/C 三组编号，标"已修复/未修复"） |
| 零之二点五 | 修复后的实测结果（ctest、四个训练基准、搜索基准） |
| 零之二点七 | 决策前探索协议（仿 snakeAI）与 EVAB 接入 GUI |
| 零之二点八 | AI 思考过程可视化（长思考时"界面像死了"） |
| 零之二点九 | Agent 对弈 arena；**顺便挖出的堆损坏**（ASan 定位全过程） |
| 零之二点十 | 做奖励曲线时挖出的符号 bug（即时奖励一直是黑方视角） |
| 零之二点十一 | EVAB 的三个 bug（"与 AB 对弈无法取胜"的根因） |
| **零之二点十二** | **启动时加载所有模型（含为它做的加载提速）+ 奖励曲线改成每手一个点 + 验证方法学的五个坑** |
| 零之三 | 修复前的实测结果（作为对照基线） |
| **零之四** | **优化方法汇总（含实测数字）** —— 按手法组织，回答"哪个优化值得做、做完还剩什么瓶颈"<br>4.1 计算内核 / 4.2 构建与工具链 / 4.3 算法与交互 / 4.4 验证手法的沉淀 |
| 一 / 二 / 三 | 逐条问题的详细分析（A = 致命/高，B = 中/低）。**C 组的详细分析在"零之二点七 ~ 零之二点九"**，C7/C8（SIMD 相关）另见 `docs/rl_sync.md` |
| 四 | 与 `docs/analysis.md` §六 的核对（历史判断） |
| **五** | **当前仍未做的事 + 具体修法 + 优先级** |

编号约定：**A/B 是首轮审查**（A1–A15 / B1–B19），**C 是第三轮**
（Agent 对弈 / 思考可视化 / 探索预训练期间新发现的，C1–C8）。
已经在"修复进度"表里标"已修复"的条目，在下面各节里仍保留完整分析
（失效场景、证据、修法）—— 那是判断"真的修对了"的依据，不是待办。

---

## 零、修复进度总表

> A/B 为首轮审查的问题编号，C 为第三轮（Agent 对弈 / 思考可视化 / 探索预训练）。
> 每条后面括号里指向本文中给出完整分析的章节。

| 编号 | 问题 | 状态 |
|------|------|------|
| A1 | `StoneMap` 未初始化 + `Chess` 不清 `m_map`（野指针 / `test_ab` 段错误根因） | **已修复**：`StoneMap()` 调用 `clear()`；`Chess::Chess()` 末尾调用 `reset()` |
| A2 | 走法不做合法性过滤（自杀 / 不应将 / 照面），`isInCheck()` 零调用 | **已修复**：新增 `Chess::isAttacked/isLegalMove/hasLegalMoves`，`sample()` 只返回合法走法，GUI 落子也走 `isLegalMove()` |
| A3 | 没有将杀 / 困毙判定 | **已修复**：新增 `Chess::getResult(colorToMove)`（含将杀/困毙/和棋） |
| A4 | 重复局面 / 长将 / 60 回合全部未实现，`pushHistory()` 零调用 | **已修复**：`moveForward` 维护 `history`（含走棋方哈希）与 `halfMoveClock`；`isDraw()` 实现三次重复 + 120 半回合；搜索里重复局面返回 0 分 |
| A5 | 回放模式进入后无法退出 | **已修复**：`reset()` 清 `m_replayGameId` 并 `emit replayModeExited()` |
| A6 | 两个 detached 线程与对象生命周期脱钩 | **已修复**：改为成员 `std::thread`，`~MainWindow` 先 join 再 `delete ui` |
| A7 | `condit.wakeAll()` 未持锁；棋盘读写无锁 | **已修复**：`wakeAll()` 统一在 `QMutexLocker` 内；AI 落子与 `getResult` 在锁内执行；工作线程不再直接 `update()` |
| A8 | `Steps` 无锁全局对象池 | **已修复**：`get()/put()` 加 `std::mutex` |
| A9 | RL 即时奖励恒为 0（吃子信号丢失） | **已修复**：`computeReward` 不再检查 `victim->alive`（与调用时机解耦），4 个 agent 一致 |
| A10 | "无合法走法"哨兵用错常量 | **已修复**：`Step::valid` 取代 16 处 `id == ID_NONE` / `pos == (0,0)` 判断；`Step::operator=` 补上漏掉的 `reward`/`valid` |
| A11 | GUI 里 DQN+MCTS 实际随机走子 | **已修复**：`selectMove(color, 200, false)`（`training=false`） |
| A12 | 后台训练线程与 AI 线程竞争同一 agent | **已修复**：`aiThink`/`aiThinkForAgent` 的 RL 分支加 `m_agentMutex` |
| A13 | 关窗阻塞在训练线程 join 上 | **已修复**：单轮规模从 4 局×200 步×50 模拟降到 `BG_TRAIN_EPISODES/MAX_MOVES/SIMS` |
| A14 | SQLite 跨线程使用；写库接口未接线 | **未修复**（见下文，需要设计决定） |
| A15 | 在线训练接口 GUI 侧未调用 | **已修复**：每个 agent 实现 `exploreAndTrain()`，GUI 侧由 `ChessBoard::preTrainThenDecide()` 在走子前调用（见"零之二点七"） |
| B1 | `moveForward` 忽略 `moveTo()` 返回值导致棋盘写坏 | **已修复**：`applyMove/undoMove` 拆分；失败不记账；`moveBack` 只在确实执行过时回退 |
| B2/B3 | `reset()` 不清回放/历史状态、可与 AI 思考重叠 | **已修复**：`reset()` 加锁并清回放状态 |
| B4 | 只有 MAX 节点做静态搜索 | **已修复**：两处叶节点统一走 `quiescenceBlackView()` |
| B5 | 静态搜索丢父窗口导致不剪枝 | **已修复**：父边界换算到走棋方视角后传入 |
| B7 | 搜索里没有"重复局面 = 和棋 0 分" | **已修复** |
| B8 | 深度/迭代数注释与真实参数三处不一致 | **已修复**：集中为 `AB_DEPTH/MCTS_SIMS/PPO_SIMS/DQNMCTS_ITERATIONS` 常量 |
| B9 | 回放把列坐标当棋子 id | **已修复**：抽出 `applyDbStep()`，三处重复代码合一 |
| B10 | `getStonePos` 负数除法向零截断 | **已修复**：改 `std::floor` |
| B13 | `selfPlay` 从不返回平局 | **已修复**：返回 `Chess::RESULT_*` |
| B14 | `test_main.cpp` 的 static agent 悬垂引用 | **已修复**：改为每步新建 agent |
| B19 | `Tensor::MM` 只有 0.11 GFLOP/s | **已修复**：内层循环改为扁平指针 + 把 stride 提到循环外，实测 **190 倍**（见下） |

> **第三轮（Agent 对弈 / 思考可视化 / 探索预训练）新发现的问题编号为 C**，
> 详细定位过程与实测见"零之二点七 ~ 零之二点九"，优化手法汇总见"零之四"。

| 编号 | 问题 | 状态 |
|------|------|------|
| **C1** | **AI 工作线程被"虚假唤醒"**：条件变量写成 `if` 而不是 `while`，醒来后也不重新检查 `state` → `reset()` / `matchAgents()` 的 `wakeAll()` 把**空闲** worker 叫醒，它照跑 `aiThink()`，与另一个线程**同时改同一个 `env`** | **已修复**：`while (state != THINKING && state != TERMINATE)` 循环等待 + 进搜索前重新确认状态。ASan 定位到 `env.history` 的 double-free（见"零之二点九"） |
| **C2** | `!step.valid` 被直接当成"走棋方被将死" → agent 抽风返回无效走法时，对局在**一步都没落子**的情况下判胜负，之后棋盘再也点不动 | **已修复**：先分清"真无棋可走"与"agent 出错"；后者用第一个合法走法兜底 + stderr 报告 + 计入 `MatchStats::agentErrors`（`process()` 与 arena 两处） |
| **C3** | `applyMove/undoMove` 用外部传入的 `Step` 直接索引 `std::array` 与 `StoneMap::data[10][9]`，无任何边界校验 → 一个坏字段就是越界写，落在相邻的 `env` 上 | **已修复**：校验 `id ∈ [0,32)`、`pos/nextPos` 在棋盘内、棋子存活且与 `Step` 起点一致，否则返回 `false` |
| **C4** | 思考途中按"开局"：`reset()` 只把 `state` 改回 IDLE 就返回，工作线程算完**照样把子落到重置后的新棋盘**上 | **已修复**：`m_thinkGeneration`，`reset()` 在棋盘锁内 `+1`，`process()` 在同一个锁内比对，代数变了整步作废 |
| **C5** | `exploreAndTrain` 把调用方的 `color` 与棋盘 `sideToMove` 混为一谈，探索结束把 `sideToMove` 留在 rollout 末尾 | **已修复**：`rolloutFromCurrent` 与 `EVABAgent::exploreAndTrain` 入口保存/出口恢复；`test_pretrain` 用"连续 5 次探索后棋盘逐字段不变"守住 |
| **C6** | `shutdownSave()` 漏了 EVAB：界面上可选、能被在线训练的 agent 退出时从不落盘，每次启动都从随机价值网重来 | **已修复**：补 `weights/evab_agent.dat` |
| **C7** | `Tensor::MM::ikjk/kijk` 的 **SIMD 内核是赋值、标量是累加**（语义不一致）。当前调用点 kdim=1 走标量所以梯度没错，但多处注释明写 `+=`，`lstm.cpp:143-148` 更是一次累加到同一个 `delta.h` —— **一旦做批量训练（kdim ≥ 8）会静默只保留最后一次的贡献** | **未修复（潜在）**：内核改 `z[...] += dot(...)`。见"零之四"与 `rl_sync.md` |
| **C8** | `kikj`（反向 GEMV `ei = wᵀ·e`）**没有** SIMD 快速路径，掉回标量循环：实测 **0.991** vs 前向 `ikkj` 的 **0.102** ns/MAC（**9.7×**）—— 训练步卡在没被优化的那一半 | **未修复（性能）**：补一个与 `gemv_ikkj` 对称的 `gemv_kikj`。见"零之四" |
| **C9** | **奖励曲线在对局过程中一动不动**：`gameRewardSample` 一局只 emit 一次，而一局可能几百手、跑十几分钟（实测 `SAC+AZ-MoE vs Alpha-Beta` **276 手 / 620.8 秒**）→ 用户看到的就是"对弈时奖励曲线没有更新" | **已修复**：新增 `matchRewardProgress`，**每手**一个"本局累计"点（走子方视角），局末再由 `gameRewardSample` 补一个含终局 ±1 的点；读数前缀改成"奖励(局内累计)"。见"零之二点十二"；`test_match` [2.8] 9 条断言 + `verify_match_ui.ps1` 的"点数在涨"和"点数 ≥ 手数 − 局数"两条钉住 |
| **C10** | `playMatchGame` 的 `rewardA/rewardB` 出参是**死参数**（传进去从没被写过），A/B 换算在 `matchAgents` 里另算了一遍 → 两处规则一旦分叉就是"曲线上的点和逐局明细里的奖励对不上" | **已修复**：`playMatchGame` 增加 `bool aIsRed` 参数，红/黑 → A/B 的换算集中在 `syncAB()` 一处，出参与每手进度信号共用同一份结果 |
| **C11** | **验证方法学**：检查本身会给出错误结论（本轮真实发生 5 次）：后台保存没写完就断言 → 假失败；启动加载的日志冒充保存日志 → **对局没打完也报通过**；改脚本丢 BOM → 报一堆和原因无关的语法错；拿"本来就不该出现的东西"当失败（AB vs AB 时没有可训练 agent，本来就没有损失曲线和存权重行）；`Select-Object -First N` **掐断编译管道** → 拿旧 exe 验证并认真分析了假现象 | **已修复并记录**：见 `docs/agents_design.md` **16.3**（五个坑）；`verify_match_ui.ps1` 加 `$expectSave` 做适用性判断、按前缀剥壳再比、构建输出改走文件并核对 exe 时间戳 |

### B19 修复后的实测数据

同一台机器、同一个微基准（MSVC 2022 `/O2`，`ikkj`/`kikj` 各 300/200 次取平均）。
分三档：最初的标量实现 → 扁平指针的标量实现 → 启用 `rl/simd_ops.hpp` 的 AVX2 内核：

| 内核 | 最初（`posOf` 版） | 扁平指针标量 | **+AVX2 内核** | 总加速 |
|------|--------------------|--------------|----------------|--------|
| `ikkj (90x90)*(90x90)` | 13.274 ms / 0.11 GFLOP/s | 0.072 ms / 20.4 | **0.053 ms / 27.8** | **250x** |
| `ikkj (90x360)*(360x90)` | 53.016 ms / 0.11 | 0.277 ms / 21.1 | **0.157 ms / 37.2** | **338x** |
| `kikj (360x90)^T*(360x90)` | 53.088 ms / 0.11 | 0.274 ms / 21.3 | **0.272 ms / 21.4** | 195x |
| `ikjk (128x64)*(64x90)^T` | 18.11 ns/MAC | 0.242 ms / 6.0 | **0.254 ms / 5.8** | — |

归约（8100 元素）在启用 SIMD 后收益最大：

| 运算 | 标量 | AVX2 | 加速 |
|------|------|------|------|
| `sum` | 1.002 ns/elem | 0.122 ns/elem | **8.2x** |
| `max` | 1.005 ns/elem | 0.123 ns/elem | 8.2x |
| `norm2` | 1.007 ns/elem | 0.125 ns/elem | 8.1x |

逐元素算符（`a*b+c`）几乎没变（3.28 → 3.13 ns/elem），原因不是算术而是**临时量分配**：
每个算符构造一个新的 `Tensor`（一次 32KB 分配 + 一次零填充 + 一次拷贝），
`a*b+c` 因此要付 3 次分配。`tensorsi.hpp` 本身有同样的性质，它在注释里给出的解法是
`exprfunc.hpp` 的表达式模板（`expt::view(a)*expt::view(b) + expt::view(c)`，单趟融合、
一个临时量）—— 那需要改写调用点，不属于"替换 Tensor 类"的范围，见
`docs/rl_sync.md`。

数值正确性由独立交叉验证确认（朴素 `posOf` 三重循环 vs SIMD 内核，逐元素比较）：

```
ikkj (90x90)x(90x90)   rel = 1.8e-07      kikj (90x90)x(90x90)   rel = 1.6e-07
ikkj (90x360)x(360x90) rel = 2.3e-07      ikjk (out,in)+=(out,1)*(in,1)^T  完全相同 (rel = 0)
```

即 `ikkj`/`kikj` 只剩浮点重结合带来的 ~2e-7 相对误差，`ikjk` 在真实用法形状下**逐位相同**
（包括"连续累加两次"的语义）。

---

## 零之二点五、修复后的实测结果

以下数字**全部来自删掉 build/ 之后的干净构建**（见本节末尾关于"增量构建会留下陈旧
目标文件"的说明，那一度让本文的早期数字失真）。

`ctest`（注册了 `test_ab` / `test_mcts` / `test_rules` / `test_pretrain`）：

| 测试 | 最初 | 修复后 |
|------|------|--------|
| `test_ab` | **SEGFAULT** | **Passed，4.0 s** |
| `test_mcts` | Passed，110 s | Passed，220 s（走法生成现在会剔除非法走法，随机回放成本上升） |
| `test_rules`（新增） | — | Passed，0.06 s（92 条断言） |
| `test_pretrain`（新增） | — | Passed，3.9 s（23 条断言，见"零之二点七"） |

4 个训练基准（默认不注册 ctest，手工运行）：

| 测试 | 最初 | 扁平指针标量 | **+SIMD/GEMV** |
|------|------|--------------|----------------|
| `test_pg` | 346 s | 26 s | **18 s** |
| `test_dqn` | **>900 s 超时** | 536 s | **353 s** |
| `test_ppomcts` | **>900 s 超时** | 233 s | **78 s** |
| `test_dqnmcts` | **>900 s 超时** | 1299 s | **390 s** |

`test_ab` 的 Alpha-Beta 搜索基准（干净构建）：深度 3 ≈ 57 ms，深度 4 ≈ 148 ms，
深度 5 ≈ 3.1 s。GUI 的 `AB_DEPTH` 因此定为 4。

### 构建坑：增量构建会留下陈旧目标文件

修这个坑的过程中发现并修掉了一个**会让上面所有测量都失真**的构建问题：
本机 MSVC 输出的是本地化 `/showIncludes`（`注意: 包含文件:`），而 CMake 在编译器探测
阶段记录的前缀是另一串（乱码）字节，于是 ninja 解析不出任何头文件依赖 —— **改了
`src/chess.h` / `src/stone.h` 之后，目标文件不会重编**。snakeAI 的 CMakeLists 里对此
有一段说明并给 `RL_CORE` 加了 `OBJECT_DEPENDS`，我照做时只加了 `RL_CORE` 和 `chess`
两个目标，**漏掉了 6 个测试目标**。

症状很隐蔽：不报错、不失败，只是跑的是"新旧头文件混编"的二进制。实测同一份源码：

```
单独用 cl 编出来的 test_ab      : 5 s   (搜索基准 深度5 = 1910 ms)
CMake 增量构建出来的 test_ab    : 23 s  (搜索基准 深度5 = 3133 ms)
```

现在 `CMakeLists.txt` 里有一个 `chess_add_header_deps()` 函数，**每一个**目标
（`chess`、6 个 agent 测试、`test_rules`）都会调用它；清理后重建，`test_ab` 回到 4.6 s。

### 构建坑：SIMD 指令集必须向系统确认，而且判据要覆盖 MSVC

`CHESS_ENABLE_AVX2`（默认 ON）会给相关目标加 `/arch:AVX2`，从而解锁 `rl/simd/` 的
AVX2 内核。但这条开关有两个陷阱，都已处理：

1. **`/arch:AVX2` 会让二进制要求 CPU 支持 AVX2**，拿到不支持的机器上直接非法指令
   崩溃。现在配置阶段用 `try_run` 跑一次 CPUID 探测，系统不支持就自动不启用并给出
   警告；运行阶段用 `RL::cpuinfo::describe()` 把"编译进来的是哪套内核 + 这台机器支持
   什么"显示在窗口标题、测试横幅和启动日志里。
2. **MSVC 在 x64 下不定义 `__SSE2__`**（只定义 `_M_X64` / `_M_IX86_FP`），原来的
   `#elif defined(__SSE2__)` 判据让 MSVC 构建直接掉到标量路径、SSE2 内核永远选不上。
   现在判据为 `__SSE2__ || _M_X64 || _M_AMD64 || _M_IX86_FP >= 2`。

三条路径实测（GEMV `w*x`，i5-10400F）：标量 1.059 ns/MAC → SSE2 0.179（5.9x）→
AVX2 0.099（**10.7x**）。细节见 `docs/rl_sync.md`。

---

## 零之二点七、决策前探索（snakeAI 风格）与 EVAB 接入 GUI

### 协议：先"探索环境"训练，再落子

snakeAI 的决策流程是 `observe(state)` → 若 `trainFlag` 则**从当前状态出发**按探索策略
走 N 步、收集 transition、训练一次 → 最后才对**探索前**的状态取 `action(state0).argmax()`。
本工程原本完全没有这个环节（正是 A15：所有在线训练接口在 GUI 侧零调用），
所以 5 个 agent 在 GUI 里只会拿初始随机网络下棋。

现在 `AgentBase` 增加两个虚函数：

```cpp
virtual bool exploreAndTrain(int color, int rolloutSteps);  // 默认 no-op，返回 false
virtual std::string getExploreInfo() const;                 // 供 UI 显示"这步之前学了什么"
```

5 个可训练 agent 各自的实现在 `docs/agents_design.md` 里对应关系如下（`ABAgent` 无参数，
默认实现直接跳过）：

| Agent | 动作选择 | 训练调用 | 经验池 |
|-------|----------|----------|--------|
| `DQNAgent` / `DQNMCTSAgent` | `noiseAction` | `perceive` + `learn` | 复用 agent 自身的 replay buffer |
| `PGEagent` | `gumbelMax` | `reinforce` | 回合内累积的 log-prob/奖励 |
| `PPOMCTSAgent` | `ppo.action` + `Random::categorical` | `ppo.learnSelfPlay` | PPO 自己的轨迹缓冲 |
| `EVABAgent` | `exploreEps`-greedy 的搜索根 | `acceptNet` 门控的 TD-leaf 更新 | 树搜索叶节点样本 |

关键实现点在新增的 `src/agentrollout.hpp`（header-only 模板）：

```cpp
template <class Agent, class PickMove, class OnTransition>
int rolloutFromCurrent(Agent &agent, Chess &chess, int color, int steps,
                       PickMove pick, OnTransition onTrans);
```

它**保存/恢复 `chess.sideToMove`**，用 `moveForward` 前进、结束后逆序 `moveBack` 回退，
保证调用前后棋盘逐字节不变。这个"不变"是本轮新测试 `test_pretrain` 的第一个断言。

### 实测（`test_pretrain`，23 条断言全通过，3.9 s）

```
DQNAgent      训练 371 ms  rollout 32 步, 训练 1 次(池 32)
PGEagent      训练  69 ms  rollout 32 步, reinforce 1 次
PPOMCTSAgent  训练 471 ms  rollout 32 步, PPO 更新 1 次
DQNMCTSAgent  训练 373 ms  rollout 32 步, 训练 1 次(池 32)
EVABAgent     训练   3 ms  探索 8 步, 价值网络更新 1 次 (|net-hand| 1.009413)
ABAgent (对照) 跳过   0 ms
连续 5 次探索后棋盘仍然不变: OK
```

### 这次发现的真问题：`exploreAndTrain` 会篡改轮到谁走

第一版实现直接把 `chess.sideToMove` 当成 `color` 来展开 rollout。但调用方传进来的
`color` 与棋盘当前的 `sideToMove` **并不总是同一个值**（GUI 的"悔棋后重算"、
自对弈里给指定一方求招都会这样）。结果是探索结束后 `sideToMove` 被留在最后一层
rollout 的位置，调用方接着搜索就**从错误的走棋方**开始 —— 表现出来就是"AI 走出非法
招 / 瞬间认输"。

修复：`rolloutFromCurrent` 与 `EVABAgent::exploreAndTrain` 都在入口保存
`savedSideToMove`、出口恢复。`test_pretrain` 现在会连续调用 5 次探索并逐字段比对棋盘，
这个 bug 一旦回归会立刻被抓到。

### GUI 侧接线

- 模式下拉框新增 **"EVAB (学会评估的 Alpha-Beta)"**（`AGENT_EVAB`，权重
  `weights/evab_agent.dat`）。
- 控制区新增复选框 **"走子前先探索训练"**（默认勾选，`preTrainCheck`），
  连到 `ChessBoard::setPreTrainEnabled()`；步数由 `setPreTrainSteps()` 控制。
- 决策路径统一为 `ChessBoard::preTrainThenDecide(agent, color)`：
  先 `exploreAndTrain(color, steps)`，再用 `getExploreInfo()` 的内容填
  `getLastExploreInfo()` 供界面显示，**然后**才真正 `selectMove(...)`。
  勾选框关闭时该函数退化成纯粹的 `selectMove`，与旧行为完全一致。

---

## 零之二点八、AI 思考过程的可视化（长思考时"界面像死了"）

### 问题：思考期间界面零反馈

AI 在后台线程 (`ChessBoard::process`) 里思考，思考期间 `mousePressEvent` 开头就是
`if (state != STATE_IDEL) return;` —— **拒绝落子是对的**（轮到 AI 走），但界面上
没有任何东西说明"它还在算"：

- 棋子不动（AI 那一步还没算完）；
- 点自己的棋子也没有任何反应（点击被静默吞掉）；
- 唯一的反馈 `aiThinkFinished` 是**思考结束之后**才发的。

加入"走子前先探索+预训练"之后，单步思考从 ABAgent 的 ~150 ms 变成
PPOMCTS/DQNMCTS 的 2–3 秒（探索 ~0.5 s + 搜索 ~2 s），于是"到底还在思考还是已经
卡死"从一个理论问题变成了每一步都会遇到的实际问题。

### 修复

新增 `src/thinkingindicator.{h,cpp}` —— `ThinkingIndicator` 控件，三种动画同时给：

| 元素 | 含义 | 实现 |
|------|------|------|
| 呼吸灯 | 还活着、还在算 | 径向渐变的半径与透明度按 `sin(2π·phase)` 起伏 |
| 旋转粒子 | 正在推进 | 一圈粒子的亮度峰值（彗头）随时间绕环跑，带彗尾衰减 |
| 沙漏 | 这一轮有进展 | 上半部的沙按相位线性漏到下半部，漏完复位；**结束后沙子全在下半部**，一眼能看出"算完了" |

外加：实时耗时（百分秒，`Consolas`）、agent 名、当前阶段文字。动画由控件自己的
`QTimer`（33 ms ≈ 30 fps）驱动，不依赖 agent 上报进度 —— 所以哪怕某一步算很久，
画面也一直在动。每帧还会发 `elapsedChanged(ms)`，让主窗口的"AI思考时间"标签
在**思考过程中就跟着跳**，而不是等结束才出数字。

界面上落两处（一处是给"正在盯棋盘"的人看的，一处是给"想要细节"的人看的）：

1. **棋盘内嵌状态条**（`ChessBoard::drawThinkingOverlay`）：画在棋盘**上方那条
   24 px 空白带**里 —— 最上一排棋子的圆心在 `offsetY=50`、半径 24，所以
   `y < 26` 完全空着。放这里既不挡任何棋子，又正好在玩家视线中心：
   `[旋转粒子] AI 正在思考 1.42s · ② 搜索 / 决策`。
2. **右侧控制栏的 ThinkingIndicator**：完整的三动画 + 大号实时耗时 + 阶段。

只重绘这一小条（`update(thinkingOverlayRect())`），不是 25 fps 重画整个棋盘。

### 顺带修掉的两个交互问题

1. **思考期间点击不再静默吞掉**：改为在状态条上闪一句"请稍候, 现在还不能走子"
   （2.5 秒后自动消退）。"点不动"和"点了没反应"是两回事，后者才像死机。
2. **思考途中按"开局"会丢掉那一步**（真 bug）：`reset()` 原来只把 `state` 改回
   `STATE_IDEL` 就返回，而工作线程还在算，算完**照样把子落到重置后的新棋盘上** ——
   表现为"我刚开了新局，对方却已经走了一步"。现在有一个 `m_thinkGeneration`
   计数器：`reset()` 在棋盘锁内 `+1`，`process()` 在锁内比对，代数变了就整步作废
   （含它的计时），回到等待玩家状态。

### 阶段信号

`ChessBoard` 新增 4 个信号，都在工作线程 `emit`、由 Qt 队列投递到 GUI 线程
（接收者是 GUI 线程对象，`AutoConnection` 自动排队），所以主窗口的槽里可以安全
读写控件：

```
aiThinkingStarted(agentName, exploreSteps)   开始思考
aiThinkingStage(stage)                       ① 探索环境 + 预训练 / ② 搜索 / 决策 / ③ 落子
aiThinkingStopped()                          思考真正结束 (含被"开局"打断)
aiExploreInfo(info)                          本次"探索+预训练"做了什么
```

顺序上有一条硬约束：**必须先把 `state` 落成 `IDLE`，再发 `aiThinkingStopped`** ——
状态条的可见性看的就是 `state`，反过来的话 GUI 处理队列消息时可能多画一帧。

`aiExploreInfo` 的存在还顺手修掉一个数据竞争：`m_lastExploreInfo` 是
`std::string`，原来 GUI 线程可能直接去读工作线程正在写的那个对象（A7 那一类），
现在改成用信号送一份拷贝过去，getter 自己也加了锁。

### 验证（不是"看起来对"，是量出来的）

用一个 PS 脚本驱动真实窗口（点棋盘 → 抓棋盘顶部那一小条像素 → 数深色像素）。
脚本留在 `tools/verify_thinking_ui.ps1`，可以直接重跑：

```
powershell -ExecutionPolicy Bypass -File tools/verify_thinking_ui.ps1
```

两个坑都踩过：状态条是**混色**的（`QColor(28,38,54,230)` 叠在米黄棋盘上 ≈
`(47,52,62)`，按原色找会一个点都找不到）；默认 ABAgent 只算 ~150 ms，
**很容易在第一次采样之前就算完了**（脚本里 `ClickAt` 尾部那 200 ms 的 sleep
就足够错过它 —— 所以脚本会先用 Tab+Down 把 agent 切到 MCTS，再在点击后
**不 sleep** 立刻开抓）。

实测输出：

```
idle_strip_dark_pixels   = 0     空闲时状态条不存在
idle_ctrl_diff_pixels    = 0     空闲时控件不动 (没有空转动画)
thinking_strip_frames    = 4 / 90
thinking_strip_max_dark  = 677 / 728   思考期间状态条铺满采样窗 (93%)
thinking_ctrl_animating  = 2 / 29
thinking_ctrl_max_diff   = 8982        思考期间控件大幅变化 -> 动画确实在跑
after_strip_dark_pixels  = 0     思考结束状态条消失
RESULT: PASS
```

即"思考中 → 有状态条 + 控件在动；空闲/结束 → 两者都干净"。

---

## 零之二点九、写 Agent 对弈时挖出来的三个真问题（含一个堆损坏）

做 "Agent 对 Agent 对弈"（见 `docs/agents_design.md` §9）时新测试 `test_match`
在 ctest 下崩了：`0xc0000374`（HEAP_CORRUPTION）/ `0xc0000005`，
**约 75% 的运行会崩**，偶尔又能跑完但报出一个荒谬的比分 —— "第 1 局 (1 手): EVAB 胜"，
即**红方只走了一步就被判负**。单独运行同样代码却时常是好的，属于典型的内存问题。

### 定位过程（值得记下来，因为每一步都省掉了大量瞎猜）

1. **先把它变成可复现的**：写了一个临时诊断程序，把"arena 会做的一切"拆成四个场景
   （A 裸 Chess+AB / B 裸 Chess+AB、EVAB 交替 / C ChessBoard+AB、EVAB / D ChessBoard+AB、AB）。
   结果：**A、B 干净，D 干净，只有 C 崩** —— 也就是"ChessBoard + EVAB"这个组合。
   如果只盯着新写的 arena 代码看，永远找不到这里。
2. **Debug 构建没报错**：MSVC Debug 会给 `std::array` / `std::vector` 的 `operator[]`
   加越界检查，它一声不吭 —— 说明越界不在这些容器上。
3. **上 AddressSanitizer**（见 `CMakeLists.txt` 末尾记录的命令）。它直接给出了调用栈：

```
ERROR: AddressSanitizer: attempting double-free
  #7 Chess::pushHistory()            chess.cpp:834
  #8 Chess::moveForward()            chess.cpp:421
  #9 ABAgent::quiescenceSearch()     abagent.cpp:251
  #16 ABAgent::findBestMove()        abagent.cpp:100
  #18 ChessBoard::aiThink()          chessboard.cpp:874
  #19 ChessBoard::process()          chessboard.cpp:648   <-- AI 工作线程
```

### 根因：条件变量用 `if` 而不是 `while`，且醒来后不重新检查 state

`ChessBoard::process()`（AI 工作线程）原来是这样等的：

```cpp
QMutexLocker locker(&mutex);
if (state != STATE_THINKING) {   /* ← 应该是 while */
    condit.wait(&mutex);
}
if (state == STATE_TERMINATE) { break; }
/* 醒来就往下走, 不管 state 是什么 */
Step step = aiThink(Stone::COLOR_BLACK);
```

条件变量本来就会**虚假唤醒**；更要命的是 `reset()` 和（我新加的）`matchAgents()`
都会在 `state` 仍是 `IDLE` 的时候 `condit.wakeAll()` —— 它们只想叫醒"正在思考"的
worker 好让它作废在飞的那一步，但空闲的 worker 被叫醒后**照样往下跑 `aiThink()`**，
一边搜索一边往 `env.history` 里 `push_back`。如果此时另一个线程（arena）也在用同一个
`env`，两个线程同时改同一个 `std::vector` —— 这就是上面那个 double-free。
两个 `Chess` 对象 `chess` 与 `env` 是 `ChessBoard` 的成员，`aiThink` 只碰 `env`，
所以症状是"偶发"而不是"必崩"。

修复（`chessboard.cpp`）：`while (state != STATE_THINKING && state != STATE_TERMINATE)`
循环等待，并且在进入搜索前重新确认 `state == STATE_THINKING`。

**这个 bug 不只影响 arena**：`reset()`（界面上的"开局"按钮）也会 `wakeAll()`，
所以以前按一下"开局"，AI 工作线程就可能被凭空唤醒并自己走一步 —— 棋盘在没有轮到它的
情况下变化，正是那种"棋子自己动了 / 我点了没反应"的来源。

### 第二个问题：非法走法被当成"被将死"

`!step.valid` 以前直接判"走棋方被将死"：

```cpp
if (!step.valid) {
    emit sendResult(Chess::RESULT_RED_WIN);   /* AI 无合法走法 -> 红方胜 */
    state = STATE_TERMINATE;
}
```

于是**只要 agent 抽风返回一个无效 Step，对局就会在一步都没落子的情况下判红方胜**，
而且之后再也点不动棋盘 —— 这正好是玩家报告过的现象。现在改成先分清两种情况：
真的没有合法走法才判胜负；**如果还有合法走法，说明是 agent 出的错**，用第一个合法
走法兜底并在 stderr 上报一行，arena 还会把次数记进 `MatchStats::agentErrors`
（显示在比分行里）。修好根因之后这个分支已经不再触发（`test_match` 的 stderr 是空的）。

### 第三个问题：越界索引没有守卫

`Chess::applyMove/undoMove` 直接用外部传进来的 `Step` 索引棋盘：

```cpp
Stone *stone = m_children[s->id];    /* std::array, 不检查 */
... stone->moveTo(s->nextPos);       /* StoneMap::data[10][9], 也不检查 */
```

`Step` 来自搜索、回放记录、数据库行，任何一个字段坏了都会变成**越界写**，落在
`Chess` 对象的相邻成员上（`env` 就挨着 `chess`）。现在 `applyMove/undoMove` 先校验
`id ∈ [0,32)`、`pos`/`nextPos` 在棋盘内、棋子存活、且 `Step` 里的起点与棋子当前位置
一致，不满足就返回 `false`（`moveForward` 本来就要处理 `false`）。
把"内存被写坏"变成了"这一步不执行"。

### 顺带修掉的

`shutdownSave()` 漏了 EVAB —— 它是界面上可选、能被在线训练的 agent，但退出时从来不
落盘，于是每次启动都从随机价值网络重新开始，上一局学到的全丢。

### 验证

* ASan 下把诊断程序的两个关键场景各跑 3 遍：修之前 C/D 几乎必崩，修之后
  **6/6 全部退出码 0、0 次异常、无 ASan 报错**；另外两个场景 A/B 也各跑 1 遍通过。
* `test_match` 从"ctest 里必崩"变成 **Passed 1.9 s，18 条断言 0 失败**，
  比分也变回正确的 `共 2 局 / 8 手`（每局 4 手，判和）。
* `ctest` 5/5 通过。

---

## 零之二点十、做奖励曲线时挖出来的符号 bug（即时奖励一直是黑方视角）

用户要求"增加控件动态显示……模型每次对局所获的环境奖励曲线"。要画这条曲线，先得回答
"这一步到底给了多少奖励"，于是写了个 30 行探针（`build/reward_probe.cpp`，临时文件）
直接问棋盘和 agent：

```
红方: 炮(7,1) 吃 马(0,1), victim color=1 value=0.30
moveForward totalReward = -0.300
computeReward(红)      = -0.300
结论: 红方白吃一个黑子, 拿到的即时奖励是 负
```

**根因**：五个 agent（`dqnagent` / `dqnmcts_agent` / `pgagent` / `ppomcts_agent` /
`sacazagent`）各有一份 `computeReward` 拷贝，全都写成

```cpp
return (color == Stone::COLOR_BLACK) ? reward : -reward;   // 黑方视角
```

而 `Chess::moveForward` 的记账（`吃红子 += value`）也是黑方视角 —— 两处同源，所以看起来
"自洽"，很难被注意到。但**同一个 agent 的终局奖励是走子方视角的 ±1**
（`SACAZAgent::resultValue`），于是对红方来说塑形信号（吃子 −）与终局信号（赢棋 +）
方向相反：**红方被教成"吃子是坏事"**。
（`EVABAgent::evaluateLeaf` 的 `(color == BLACK) ? MATE : -MATE` 是正确的走子方视角，
说明原意就是走子方视角，`computeReward` 是写错了。`dqnmcts_agent.cpp` 里另有一处调用把
颜色硬编码成 `COLOR_BLACK`，是同一个错误的产物。）

**修复**：五份 `computeReward` 统一成走子方视角（吃子永远 `+value`），并在 `test_match`
第 [2.6] 节把**两套约定都钉住**（agent 侧走子方视角、棋盘侧黑方视角记账）：

```
红方吃 马 (value=0.30): moveForward 记账 = -0.300
computeReward(红) = +3.000
computeReward(黑) = +3.000
```


## 零之二点十一、EVAB 的三个 bug（"与 AB 对弈无法取胜"的根因）

用户报"evagent 与 abagent 多次对弈无法取胜，检查 evagent 在对弈过程是否累计梯度训练"。
查下来：**梯度确实在累计**（64 步探索 / 每 16 个样本一次 RMSProp），但训练出来的网络
**从来没有参与过决策**，而且它本身是坏的。三个独立的 bug：

| # | 问题 | 现象（探针实测） | 修复 |
|---|------|------------------|------|
| 1 | **价值网络没有按 fan_in 缩放初始化** | 输入是 1260 维 one-hot，第一层 pre-activation 标准差 ≈ √(32/3) ≈ 3.3 → tanh 一上来就饱和：任意局面输出都是 0.998，20 次更新只动了 **1.4e-4**，`|net-hand|` ≈ 0.95（几乎最大分歧） | 构造函数里加 `scaleLayerInit(valueNet)`（1/√fan_in）；修完 `|net-hand|` 0.95 → **0.02**，每轮真的在动 |
| 2 | **`blend`（学习评估与手工评估的混合比例）恒为 0** | 全工程只有测试改过它，agent 本体从没改过 → 界面上那个 "learned eval" 一次都没生效；EVAB(depth=4) 与 ABAgent 走法 **22/30 相同** | 在线更新成功后 `blend += 0.05`（上限 0.3），回滚时退两步（门控沿用原有的 `|net-hand|` 检查） |
| 3 | **搜索预算照抄了 AB 的深度** | `EVABAgent(env, 48, AB_DEPTH=4, 0)` —— 同深度下 EVAB 比 AB 快一个数量级（depth 5: EVAB 188 ms vs AB 1873 ms），等于把"更强的搜索"这个卖点也关掉了 | `EVAB_DEPTH = 6` + `EVAB_BUDGET_MS = 800`（迭代加深 + 时间上限自动适应评估成本） |

修完之后（`build/evab_arena.cpp` 探针，每配置 6 局、每局交换先后手、与界面同一条调用序列）：

| 配置 | 胜 | 和 | 负 | ms/步 |
|------|--:|--:|--:|------:|
| 同深度(4) + 学习评估 | 2 | 2 | 2 | 174 |
| 深一层(5) + 学习评估 | 1 | **5** | **0** | 796 |
| 深一层(5) + 纯手工评估（blend 钉 0） | **0** | 6 | 0 | 70 |

**能赢了**（修复前探针里 0 胜），而且"赢棋来自学习评估"这件事有对照：把 blend 钉回 0、
同样深度就是 6 局全和（两边同一个评估）。

**代价**：只要 blend > 0，每个叶子都要跑一次网络前向，实测贵 11~14 倍（depth 5: 70 → 796
ms/步）→ 评估变贵 = 搜索变浅。所以 blend 上限刻意压到 0.3、搜索用时间上限。
细节见 `docs/agents_design.md` §14。

**样本量说明**：每配置 6 局，单局偶然性足以翻转结论，所以结论只能到"机制通了、能赢棋了"，
不能到"EVAB 比 AB 强多少"。
**界面上也验过**（`tools/verify_match_ui.ps1 -AIndex 6 -BIndex 0 -Games 2 -Full`）：
`最终比分: EVAB 1 : 1 Alpha-Beta  共 2 局 / 190 手`，损失曲线 95 个样本、只有一条
（修了"名字里带 blend 导致同一 agent 分出一堆曲线"的问题），双击放大窗口与源控件逐字一致。

**还没定位的偶发现象**：arena 里偶尔出现 `[1 次无效走法已兜底]`（约每 200 次决策 1 次）。
兜底逻辑保证不影响胜负，但根因未知，见 `docs/agents_design.md` §14.9 —— 不假装已解决。

---

## 零之二点十二、启动时加载所有模型 + 奖励曲线改成每手一个点（含两条实测）

这一节记两件用户直接提的事，以及一个"我的验证方法本身在骗我"的教训。

### (1) "程序启动时加载所有模型"

上一轮为了启动速度，把稀疏 MoE 变体（3×146 MB）改成了**懒加载**（启动 1.7 s，首次用到它时
卡十几秒）。用户明确要求全量加载，于是改回预加载 —— 但**不能只是改回去**：一开始实测启动
**19.2 s**，那样的体验是在逼用户二选一。所以顺手把加载路径做快：

| 手法 | 做了什么 | 实测 |
|------|----------|------|
| 流式 base64 解码 | `Tensor::base64DecodeInto`：**一遍**同时做完合法性 + 长度 + 增量 CRC32 + 写目标（原来 getline 切出字符串、再解码、再单独校验一遍） | 解码 99 → **302 MB/s** |
| 256 项解码表 | `base64DecodeTable()`（无效 = −1，`=` = 64），取代逐字符分支 | 同上 |
| 零分配的载入前校验 | 第一版用 `getline` + `std::string` 逐行校验，**每次校验都在堆上造几百万个短字符串** | 校验从"启动多花十几秒"变成可忽略 |
| 每组一条计时日志 | `[weights] <名字>: N ms` | 慢的时候一眼看出是哪一组 |

结果（本机 Release，七组权重全部预加载）：

```
[weights] PG: 167 ms      DQN: 174 ms      EVAB: 22 ms     DQN+MCTS: 162 ms
[weights] SAC+AZ: 72 ms
[weights] SAC+AZ-MoE 建网(5 x 28.7M 参数): 1604 ms
[weights] SAC+AZ-MoE 读权重(3 x 146 MB): 6235 ms
→ 启动到界面可用 19.2 s → 10.1 s（沙漏全程覆盖，界面可用时它已收起）
```

**如实说明两件事**：**(a)** 那 6.2 s 里还有约 1 s/文件 是 `Net::load` 真正读那一遍走
`ifstream + getline`（把 `iLayer::read/write` 的签名换成 `std::istream&` 还能再省 2~3 s，
纯机械改动，没做）；**(b)** 懒加载那两个分支的 `loadModel()` 现在**不可达**（权重文件在，
`startupLoad` 就读了；不在，分支里也不会读），它们只剩"防空指针"的兜底价值 —— 所以原来那个
断言"首次使用会弹沙漏"的 `tools/verify_busy_lazy.ps1` **只能失败**，已替换成
`tools/verify_eager_load.ps1`（断言相反的事实）。

### (2) "对弈时奖励曲线没有更新"

信号没断、数据没错，是**采样太稀**：`gameRewardSample` 一局只发一次，而实测那局
`SAC+AZ-MoE vs Alpha-Beta` 是 **276 手 / 620.8 秒** —— 按下开始之后的十分钟里曲线只有 **1 个点**。

修法：新增 `matchRewardProgress(gameNo, ply, rewardA, rewardB)`，`playMatchGame` **每落一手**
emit 一次，值是"本局到目前为止"累计的走子方视角环境奖励；局末仍由 `gameRewardSample` 补一个
含终局 ±1 的点（所以每局最后一个点会比前一个多出胜负那一份，这是设计不是重复计数）。

| | 之前 | 现在 |
|---|---|---|
| 采样 | 每局 1 次 | **每手 1 次** + 局末 1 次 |
| 一局 276 手 | 1 个点 | **277 个点** |
| 实测（AB vs AB，2 局 256 手） | 2 点 | **258 点**（对局中取样 11 → 228 一直在涨） |

顺手修掉 C10：`playMatchGame` 多一个 `bool aIsRed`，红/黑 → A/B 的换算只做一次，出参与每手
进度共用同一份结果（原来那两个出参是死参数，调用方在外面重算了一遍）。

### (3) 验证方法学的五个坑（本轮最贵的教训）

这部分不是产品 bug，是**检查本身在骗人**，而且错误的检查比没有检查更坏 —— 它会给出一个
看起来很像结论的答案。逐条见 `docs/agents_design.md` **16.3**，这里只记最贵的那一条：

> 为了"只看前 20 行错误"，我把构建输出接进了 `Select-String ... | Select-Object -First 20`。
> PowerShell 的 `-First N` 一凑够 N 项就**终止整条管道**（包括正在编译的进程），于是编译被
> 拦腰杀掉、`chess.exe` 根本没重新链接，而我以为构建通过，接着拿**旧二进制**跑了一遍界面验证，
> 还认真分析了"奖励曲线为什么不动"—— 那本来就是修复前的现象。
> 现在：构建输出重定向到文件再读，并且**每次构建后核对 exe 的时间戳**。

其余四条：后台保存没写完就断言（假失败）、启动加载日志冒充保存日志（**对局没打完也报通过**）、
改脚本丢 BOM（报无关语法错）、拿"本来就不该出现的东西"当失败（AB vs AB 时没有可训练 agent，
没有损失曲线和存权重行都是正确的）。最后一条催生了 `verify_match_ui.ps1` 里的 `$expectSave`：
**先判断这条检查适不适用，不适用就打印"跳过 + 原因"，而不是判失败**。

---
## 零之三、修复前的实测结果（作为对照）

`ctest --test-dir build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release`：

| 测试 | 结果 | 耗时 |
|------|------|------|
| `test_ab` | **SEGFAULT** | 12.8 s |
| `test_mcts` | Passed | 110 s |
| `test_pg` | Passed | 346 s |
| `test_dqn` | **Timeout(900 s)** | >900 s |
| `test_ppomcts` | **Timeout(900 s)** | >900 s |
| `test_dqnmcts` | **Timeout(900 s)** | >900 s |

`test_ab` 的段错误与下面 **A1** 完全对应（第一次访问棋盘就解引用未初始化的指针）。
`test_pg` 能跑完说明 RL 内核本身可用。四个训练基准的耗时全部由 **B19** 解释：
`Tensor::MM::*` 实测只有 **0.11 GFLOP/s**（18.2 ns/MAC）。这一点做过对照实验——
把 DQN 主干换成小 8 倍的上游默认结构，`test_dqn` **仍然超过 1800 s**，
所以超时是矩阵乘实现问题，不是网络结构问题。

---

## 零之四、优化方法汇总（含实测数字）

前面几节是按"发现顺序"叙述的；这一节按**手法**把用过的优化集中起来，并给出
实测数字，方便判断"哪个优化值得做、做完还剩什么瓶颈"。

### 4.1 计算内核（RL 库）

| 优化 | 手法 | 实测 |
|------|------|------|
| MM 内层循环扁平化 | 不再每次访问都过 `posOf()` 构造索引；行指针 + 线性扫描，stride 提到循环外 | `ikkj (90×90)²` 13.274 → 0.072 ms（**184×**） |
| SIMD 内核分派层 | `rl/simd_ops.hpp`：把"能不能用 SIMD"的判据集中一处，内核取自 N-spirits 的 `sse2func/avx2func.hpp`；保持 `RL::Tensor_` **一个类型**不变（换类型要重新验证所有调用点语义，收益却很小） | 同上再降到 0.053 ms / **27.8 GFLOP/s**（总 **250×**） |
| GEMV 专用内核 | MatMul 内核沿 z 的**列**方向向量化，而 `o = w·x` 的 z 只有一列 → 向量循环长度为 0，用不上。改成"按行做点积"（`gemv_ikkj` → `Instruct::dot`） | 标量 1.059 → SSE2 0.179 → **AVX2 0.099 ns/MAC**（**10.7×**）；这是本库的绝对热路径 |
| 归约 SIMD | `sum`/`max`/`norm2` 走向量归约 | 1.00 → **0.12 ns/elem**（**8.2×**）；收益最大的一类 |
| 开放 AVX2 | `CHESS_ENABLE_AVX2`（默认 ON）给相关目标加 `/arch:AVX2`；配置期用 `try_run` 跑一次 CPUID 探测，系统不支持就自动关闭并警告；运行期把"编译进来的是哪套内核"显示在标题栏 | SSE2 → AVX2 再快约 1.8× |
| **未做**：反向 GEMV | `kikj` 没有 GEMV 内核（zCol=1 判据不成立） | 0.991 vs 0.102 ns/MAC → **C8，训练步的真正瓶颈** |
| **未做**：表达式模板 | `a*b+c` 这类逐元素算符的开销不是算术而是**临时量分配**（每个算符一次 32 KB 分配 + 零填充 + 拷贝）。上游 `tensorsi.hpp` 的解法是 `exprfunc.hpp` 的 `expt::view()` 单趟融合 | 3.28 → 3.13 ns/elem（几乎没变）；需要改写调用点，不在"替换 Tensor 类"范围内 |

### 4.2 构建与工具链

| 优化 | 手法 | 为什么必要 |
|------|------|-----------|
| 头文件依赖显式化 | `chess_add_header_deps()` 应用到**每一个**目标 | 本机 MSVC 输出本地化 `/showIncludes`（`注意: 包含文件:`），ninja 解析不出依赖 → **改了 `chess.h` 目标文件不重编**，跑的是新旧混编的二进制（实测 `test_ab` 5 s vs 23 s，搜索基准 1910 vs 3133 ms）。一度让本文早期所有数字失真 |
| `RL_CORE` 独立静态库 | 显式源文件列表、`AUTOMOC/UIC/RCC OFF`、不依赖 Qt | RL 内核与界面解耦；测试目标可以只链 `RL_CORE`（`test_grad` 完全不依赖 Qt） |
| `/bigobj` | RL 的模板实例化量很大 | 否则编译直接失败 |
| 测试铺开 | `ctest` 从 1 个崩的 `test_ab` 变成 **9 个全过**：`test_ab`/`test_mcts`/`test_rules`/`test_pretrain`/`test_match`/`test_grad`/`test_weights`/`test_sacaz`/`test_sparse_moe`（另外 `bench_moe` 是手动跑的基准，故意不进 ctest） | 见 4.4 |
| ASan 配方 | 单独构建目录 + `CMAKE_CXX_FLAGS=/fsanitize=address`（**整个**构建都要带，否则 LNK2038）；运行前把 `clang_rt.asan_dynamic-x86_64.dll` 目录加进 PATH，`ASAN_OPTIONS=detect_leaks=0` | 只给单个目标加 ASan 会链接失败。C1 那个堆损坏就是它定位的 |

### 4.3 算法与交互

| 优化 | 手法 | 实测 |
|------|------|------|
| EVAB 搜索 | 置换表改用标准 **Zobrist** 键（原来的结构化异或相关性强，碰撞会让 TT 返回别的局面的分数）；TT + killer + history 排序；修复静态搜索零宽窗口 | 深度 5：146 ms / 158 572 节点 / 31.1% TT 命中 vs `ABAgent` 3133 ms → **21×** |
| 小批量更新 | 原来把整轮几百个样本的梯度累积起来只调一次 `RMSProp`（三轮训练 = 3 步梯度，什么也学不到）→ 改成真正的小批量 | EVAB 模仿引导 |net−hand| 1.23 → 0.05 |
| 训练稳定性 | `acceptNet` 门控：`netHandGap` 不下降就整体回滚（TD-leaf 在样本量不足时会发散，实测误差 0.35→0.70） | 保证"不变坏" |
| 决策前探索统一化 | `AgentBase::exploreAndTrain()` + `src/agentrollout.hpp` 的 `rolloutFromCurrent()`（保存/恢复 `sideToMove`，逆序 `moveBack`） | 5 个 agent 共用一份实现；`test_pretrain` 23 条断言 |
| 对弈方法学 | 每局**交换先后手**、胜负按参赛者 A/B 记（中国象棋先手优势大，固定红黑只是在测"谁执红"） | `test_match` + `tools/verify_match_ui.ps1` |
| 长思考的可观测性 | `ThinkingIndicator`（沙漏+旋转粒子+呼吸灯+实时耗时）+ 棋盘顶部 24 px 空白带内的状态条；**只重绘这一小条**；思考期间点击给"请稍候"反馈 | `tools/verify_thinking_ui.ps1` 量出"思考中有状态条、空闲/结束都干净" |
| 对弈不阻塞界面 | 对弈在后台线程，进度用队列信号回 GUI；同一个按钮兼作"停止"（`abortMatch` 逐手检查） | 关窗先 `abortMatch()` 再 join |
| **新增 SAC+AZ agent** | 最大熵 critic（双 Q + 软备份 + 自动调节 α）给 AlphaZero 式 PUCT 搜索提供叶子估值；策略目标来自搜索访问分布，数据进回放反复利用 | 单步决策 64 模拟 **3 ms**（1260 维输入的小 MLP），GUI 里给到 256 模拟（约 12 ms）。设计见 `docs/agents_design.md` §11 |
| **稀疏路由 MoE** | 自己实现 `rl/sparse_moe.hpp`：门控选 top-k 个专家**只算这几个**（上游 `MOE::forward` 是 16 个全算的稠密混合）。`TopK==E` 退化成稠密，正好当等参数对照 | 同样 28.7 M 参数下 **42.0 → 10.9 ms/模拟（3.85×）**；专家从 16 个减到 4 个、head 取满真因数。实测与陷阱见 §零之四 4.5 |
| **对弈结果实时可见** | `matchScoreChanged` 每局更新实时比分；逐局明细进 `QListWidget`（一局一行，含本局环境奖励）；去掉模态结果框 | 长对弈中途也能看到"几比几"；脚本断言 `score shows a ratio = True` / `list has per-game lines = True` |
| **训练损失 / 环境奖励曲线** | 自绘 `CurveChart`（不引入 Qt Charts）+ `AgentBase::getLastTrainLoss()`；奖励**每手**采一个"本局累计"点（`matchRewardProgress`），局末再补一个含终局 ±1 的点 | 实测 AB vs AB 两局 256 手 → 258 个点（对局中一直在涨）；修复前是每局 1 个点（十分钟不动的观感）。见 §零之二点十二 与 `docs/agents_design.md` §13 |
| **即时奖励符号修正** | 五个 agent 的 `computeReward` 从"黑方视角"改成"走子方视角"（吃子永远 +value） | 红方吃子从 **−0.30 变成 +0.30**；`test_match` [2.6] 钉住约定。见 §零之二点十 |
| **权重文件格式 v2** | 无损编码 + 结构指纹 + 每张量 CRC32 + 原子写 + 载入前零分配校验 | 往返**逐比特相同**、体积 **1.78×**、坏文件一律拒绝且不改动网络。见 §零之四 4.6 |
| **EVAB 三修**（初始化缩放 / blend 阶梯 / 搜索预算） | 见 §零之二点十一 | 6 局对抗从"0 胜"变成"2 胜 2 和 2 负"（同深度）；深一层 1 胜 5 和 0 负 |
| **损失曲线上报补全** | `RL::DQN::lastLoss`（在 experienceReplay 累加）/ `RL::PPO::lastLoss` / `RL::DPG::reinforce+reinforce1`（PG agent 走的是 reinforce）；`EVABAgent::exploreAndTrain` 也写 `m_lastLoss`；SAC+AZ 的在线那次 `learnBatch` 改成按池大小夹批 | `test_match` [2.7]：7 个可训练 agent 都上报有限损失（修复前只有 PPO+MCTS / EVAB 两个上报），Alpha-Beta / MCTS 仍不上报 |
| **曲线读数公式修正** | 窗口淘汰点时必须把它从 `sum/mn/mx` 里去掉 | 均值和纵轴范围以前会随淘汰漂移（"显示表示也是错的"） |
| **双击放大曲线** | `CurveChartDialog`（跟随源控件、非模态、单实例） | 脚本断言"放大窗口的读数与源控件逐字一致" |
| **对弈结束静默存权重** | 去掉"另存为"与"成功"两个模态框；存到 `ChessBoard::defaultWeightPath()`（与启动加载/退出保存同一份）；后台线程写、结果写进逐局明细列表 | 顺带消灭了"另存为默认名 ≠ 启动读取名"的静默失效；`shutdownSave()` 也改成遍历同一张表（EVAB 当年就是这么漏存的）；每个 agent 再记一条 `[weights] 保存 <agent>: N ms`（实测稀疏 MoE 438 MB = 1983 ms，**不含 fsync**） |
| **程序图标** | `tools/make_app_icon.ps1` 生成 `src/app.png` + `src/app.ico`；运行时 `setWindowIcon`（走 res.qrc）+ exe 的 PE 图标（走 app.rc） | `tools/verify_app_icon.ps1` 33 项检查全过（ICO 结构 / 字形真的渲染出来 / 运行时加载 / ExtractAssociatedIcon 提取到我们的配色） |
| **载入/保存时的沙漏等待窗** | `src/busydialog.h/.cpp`：复用 `ThinkingIndicator`；`ChessBoard` 只发 `busyStarted/busyMessage/busyFinished`；保存放到工作线程（否则弹窗画面冻结）；加载中不可关闭 | `verify_busy_ui.ps1` + `verify_eager_load.ps1` 都 PASS。**最终取"启动时全量加载"**：7 组权重约 10 s，沙漏全程覆盖；靠"流式 base64 解码 / 256 项解码表 / 零分配校验"把启动从 19.2 s 压到 10.1 s。见 §零之二点十二 |
| 关窗不再冻结 | 后台训练单轮规模从 4 局×200 步×50 模拟降到 `BG_TRAIN_*` | 关窗 join 的等待从"几分钟"降到秒级 |

### 4.4 验证手法的沉淀（这部分是本轮最有复用价值的产出）

| 脚本 / 测试 | 作用 | 为什么不能用"看起来对"代替 |
|------|------|------|
| `test_pretrain`（23 断言） | 盯住"探索不能改动真棋局" | 象棋没法像贪吃蛇那样在局部坐标上模拟，试走只能落在真棋盘上，必须原样回退；C5 就是它抓到的 |
| `test_match`（**61 断言**） | arena 统计：交换先后手、比分归属、到上限判和、中止生效；[2.6] 钉住**即时奖励的符号约定**（走子方视角 vs 棋盘层的黑方视角记账）；[2.7] 每个可训练 agent 都要上报损失；[2.8] **每手的奖励进度必须和局末奖励同账**（一局 12 手要产生 ≥11 个进度点、手号连续、"最后一个进度点 + 终局增量 == 局末采样"、A/B 增量互为相反数） | 把每局压到 4~12 手，结果可预测，断言才做得硬。奖励符号那条是"两侧都自洽、但彼此相反"的典型：只有把物理含义写下来对照才发现。[2.8] 是"用户看得见的现象也要有断言"的例子：曲线不动这件事在代码里没有报错，只有把**采样密度**写成断言才守得住 |
| `test_grad`（6 断言） | 有限差分核对 SIMD 之后的解析梯度；直接探测 MM 内核"累加 vs 覆盖" | "前向对"推不出"梯度对"（前向/反向用不同的 GEMM 形式）；C7 就是它量出来的 |
| `test_weights`（**44 断言**） | 权重文件：逐比特往返、两次存出的文件逐字节相同、体积、**老格式仍可读**、截断/指纹/翻一位/结构不符/文件不存在都要失败、失败后网络逐比特不变、原子写不留 `.tmp` | "存下来再读回去一样"以前从来没验过 —— 一验就发现老格式是**有损**的（差 1.6e-06），而后台训练每轮都在存读 |
| `test_sacaz`（**94 断言**，第 [10] 节会遍历四种骨干） | 掩码 softmax 雅可比、走法合法性、软价值里 α 熵项的形式、critic 是否真的在学、**四种骨干都能建/能走/能训/能存取** | α 的符号我第一版就写反了（断言 `V(α=0.5)−V(α=0) = +0.5H` 才发现）；骨干开关这种"多分支构造"最容易只在某一个分支上写对；分层的 write/read **顺序**错位是静默的，所以必须用"存了再读、比对输出"来查 |
| `test_sparse_moe`（**67 断言**） | 稀疏路由的三个不变量：前向真的跳过未选中专家（改权重输出逐位不变）、`TopK==E` 时与上游 `MOE` 前向/反向逐位一致、未选中专家的梯度**恰好为 0**；加上辅助损失的有限差分与纠偏方向 | 假通过有两种：直接调 `layer->backward()` 时 `e` 全是 0，"两边都是 0"看起来也一致；拿**拷贝**出来的张量做差分时扰动改不到真权重，差分恒为 0。两次都发生在写这个测试的过程中 |
| `bench_moe`（**不进 ctest**） | A/B/C/D 等时间对弈 + 标定 ms/模拟 + 专家使用分布 | 等"模拟次数"比强弱等于比谁算得多；TB 骨干一步几十毫秒，放 ctest 会既慢又偶发失败 |
| `tools/verify_thinking_ui.ps1` | 驱动真实窗口 + 采样像素，量"思考中有状态条、空闲没有" | 两个坑：状态条是**混色**的（`QColor(28,38,54,230)` 叠在米黄上 ≈ `(47,52,62)`，按原色找不到）；默认 agent 只算 ~150 ms，点完再 sleep 就错过了 |
| `tools/verify_match_ui.ps1` | 走 Windows UI Automation 驱动按钮并**读回结果文字** | `SendKeys` 只在窗口拥有前台时有效，而 `AppActivate` 在控制台占前台时会静默失败；UIA 的 `InvokePattern` 与焦点无关，还能直接读控件矩形与文本 |
| `tools/verify_eager_load.ps1` | 断言"启动时全量加载"的两个后果：启动沙漏出现并**自己收起**；首次使用稀疏 MoE 变体**不再**弹沙漏（改回懒加载就 FAIL）；同时要求奖励点数在涨（防止"对局没跑起来所以没弹窗"的假通过） | 替换掉原来的 `verify_busy_lazy.ps1` —— 那个脚本断言的是**相反**的事实，行为一改它就永远红着。**一个只能在旧行为下通过的测试要重写，不能留着** |

### 4.5 稀疏路由 MoE，以及一个把功能卡住两轮的拷贝构造 bug

用户提的方案是"不降维、用另一套稀疏路由、减少专家数与 head 数，能不能提升"。
过程与全部实测数据在 `docs/agents_design.md` §11.4.2，这里只留两条最该记住的：

**(a) 收益是确定的、可复现的**：自己写的 `src/rl/sparse_moe.hpp` 只计算门控选中的
top-k 个专家。同一个模板、**同样 28.70 M 参数**，`TopK=1`（稀疏）与 `TopK=E=4`（全算）
在真实 agent 里是 **10.9 vs 42.0 ms/模拟（3.85×）**。这就是 MoE 唯一真正的卖点：
容量不按算力付费。但它换不来绝对算力 —— `d=1260` 上一个 `TransformerBlock` 专家
（O(d²) 注意力 + 8d² FFN）单价就是 3 ms 级，所以带 TB 专家的骨干只能配很低的
模拟次数（界面上那个变体用 16 次，约 160 ms/步）。

**(b) 真正花掉时间的是这个 bug**：`src/rl/layer.h` 里 `iFcLayer` 的**拷贝构造只复制了
维度**，没有复制 `w` / `b` / `o` / `e` / `g` / `v` / `m`：

```cpp
// 修复前
explicit iFcLayer(const iFcLayer &r)
    : iLayer(r), inputDim(r.inputDim), outputDim(r.outputDim), bias(r.bias) {}
```

`SparseMoE` 里 `experts[i] = ExpertFactory<Expert>::make(...)` 是**按值返回再赋值**，
于是只要 MSVC 没省略这次拷贝，专家就成了"维度对、权重空"的空壳（`w.size()==0`），
之后所有 MM 内核都在越界读写 —— 表现是稀疏 MoE 的反向出现 **15615 个 NaN/1e28 元素**，
而且**不同次构建结果不一样**（省略与不省略拷贝的差别），非常难查。

12 行的最小复现把它钉死了（`build/copytest.cpp`，临时文件）：

```
a (直接构造) w.size=10080  w[0]=-0.847383
b (拷贝构造) w.size=0      <-- 修复前
b (拷贝构造) w.size=10080  w[0]=-0.847383  <-- 修复后
```

修复后门控梯度从 2.6e33 变成 2.7e-2（有限），稀疏 MoE 的反向才通过有限差分。

**教训**：`Net` 的拷贝是浅拷贝（共享层指针），深拷贝必须走 `copyTo`；而"按值返回一个
层"这种写法会静默走拷贝构造 —— 在一个含 `std::vector` 成员的类里，"只复制标量成员"
的拷贝构造就是定时炸弹。同类隐患的检查办法：`test_grad` / `test_sparse_moe` 里的
有限差分能同时覆盖"前向用了正确权重"和"梯度填对了位置"，这类 bug 活不过一个 eps。

### 4.6 权重文件格式（有损 / 无校验 / 非原子）

用户要求"实现使用更好的方法保存模型权重"。老格式的问题不只是"土"，而是**会静默毁掉
训练成果**：

| 问题 | 后果 | 现在 |
|------|------|------|
| 十进制文本（`ostream << float`，6 位有效数字）**有损** | 存一次读回来权重漂移 ~1e-6 相对；而后台训练**每轮**都在 `save -> load`（`TMP_WEIGHTS`），等于每轮往网络里注入一次不该有的扰动 | `Tensor::toString` 改成无损编码（base64 + CRC32），实测往返**逐比特相同**、两次存出的文件逐字节相同 |
| 没有任何校验 | 文件被截断 / 翻一个字节 / 把别的 agent 的权重喂进来，都会静默载入一堆形状错乱或数值不对的张量，直到某次前向才崩（或者更糟：不崩但输出全是垃圾） | `CHWGT2 <层数> <结构指纹>` 头 + 每张量 CRC32 + 长度检查；结构不对/数据坏/被截断一律返回失败 |
| 直接写目标文件 | 写一半崩掉或被 kill，**上一次训练好的模型就没了**（关窗正好会打断后台训练） | 先写 `.tmp` 再 `std::filesystem::rename` 原子替换；失败时删掉临时文件、保留原文件 |
| 载入失败会把网络改成半成品 | `fromString` 失败返回空张量，赋给层成员后**下一次前向就是越界读** —— 一个坏文件能把"载入失败"升级成段错误 | 载入前先**零分配**地校验每一行（形状 + 数量 + CRC），全部通过才真正写进网络；`test_weights` 断言"失败后网络逐比特不变" |

实测（`test_weights`，44 条断言）：

```
v2 (base64+CRC32): 497143 字节, save 2.8 ms, load 23.2 ms   (5.34 字节/float)
v1 (十进制文本)   : 885386 字节                              -> 体积 1.78x
truncated / fingerprint / one-bit-corrupt / wrong-structure / missing file -> 全部返回 -1
载入失败后网络的变化 = 0.000e+00
```

另外**老格式的文件仍然能读**（不带 `CHWGT2` 头就走兼容分支），所以以前存下来的权重不用
转换；`test_weights` 第 [3] 节专门验证这条路径（并顺便把老格式的有损程度量了出来：
前向输出差 1.6e-06）。

顺带的一个坑：第一版预校验直接调用 `Tensor::fromString` 做检查，于是 16 MB 的老格式权重
文件要**解析两遍**、每行 `split()` 出几百万个 `std::string` —— GUI 启动从秒级变成 30 秒
以上（脚本等不到"开始对弈"按钮变 enabled 才发现）。改成"扫一遍字节 + 数一遍逗号"的
零分配校验后恢复。**教训**：校验逻辑要和解析逻辑一样在乎常数。

界面的奖励曲线在 `ChessBoard::playMatchGame` 里显式换算成走子方视角。

**影响**：这个 bug 从"给 RL 即时奖励"那一版代码起就存在，是"RL agent 学不动"的一个独立
成因（另一个见 `docs/agents_design.md` §10 的样本效率分析）。它不影响搜索类 agent
（Alpha-Beta / MCTS / EVAB 的搜索部分）。

---

## 一、致命 / 高严重度

### A1【致命】`StoneMap` 未初始化 + `Chess` 构造函数不清 `m_map` → 野指针

`src/stone.h:84-86`
```cpp
T* data[10][9];
public:
    StoneMap(){}
```
`src/chess.cpp:150-185`：`Chess::Chess()` 的初始化列表与构造体都**没有** `m_map.clear()`。
只有 `reset()`（`src/chess.cpp:277-306`）会清整张表。32 个棋子构造函数只写自己所在的
32 格（`src/stone.h:343`），其余 58 格是未初始化指针。

失败路径：`src/chessboard.cpp:151 return chess.m_map[pos];`、
`src/chessboard.cpp:161-162 if (stone != nullptr) { if (stone->color != color)`、
`src/chessboard.cpp:300-302`。Debug 构建（MSVC `0xCD` 填充）下点击任意空交叉点必崩；
Release 只是恰好新页为 0 才"能跑"。**`test_ab` 的段错误就是本项。**

修复：`StoneMap()` 里 `clear()`，或让 `Chess::Chess()` 结尾调用 `m_map.clear()`（更彻底是调 `reset()`）。

### A2【致命】走法合法性完全缺失（自杀 / 不应将 / 照面），`isInCheck()` 是死代码

- `src/chess.cpp:344-358` `sample()` 只做 `if (alive) getPossibleSteps(steps)`，无任何过滤。
- `src/chess.cpp:371-407` 定义了 `isInCheck()`（含飞将检测），但**全仓零调用**
  （`grep isInCheck` 只命中定义 `src/chess.cpp:371` 与声明 `src/chess.h:69`）。
- GUI 玩家同样不校验：`src/chessboard.cpp:168` / `:195` 只调 `tryMoveTo()` 就 `moveForward()`。
- `src/stone.h:589-610` 把"两将照面"实现成**合法的吃将走法**，却没有把"走后照面"设为禁着
  → 引擎会主动制造照面局面并"吃将"结束游戏。

修复：新增 `Chess::isLegalMove(Step)`（落子 → `isInCheck(自己)` → 回退），
在 `sample()` 与 `chessboard.cpp` 的落子处统一走它；被将时只保留应将走法。

### A3【高】没有将杀 / 困毙判定，对局只能以"吃掉将帅"结束

`src/chess.cpp:360-369`
```cpp
int Chess::isGameOver()
{
    if (redJiang.alive == false)   return Stone::COLOR_BLACK;
    if (blackJiang.alive == false) return Stone::COLOR_RED;
    return Stone::COLOR_NONE;
}
```
没有任何地方检查"当前走棋方还有没有合法走法"。

### A4【高】重复局面 / 长将 / 60 回合自然限着全部未实现

`src/chess.cpp:425-441`：`pushHistory()` **从未被调用**（只有定义处），
`history` 恒空 → `isRepetition()` 恒 `false`。
且 `computeHash()`（`src/chess.cpp:409-423`）只用棋子 id 与坐标异或，**不含走棋方**，
即使接上也存在误判。

### A5【高】回放模式进入后无法退出（棋盘永久失去响应）

`replayModeExited` 信号在 `src/chessboard.h:88` 声明、`src/mainwindow.cpp:57` 连接，
但**没有任何 `emit`**。`loadReplayGame()` 设 `m_replayGameId`（`src/chessboard.cpp:790-818`），
而 `reset()`（`src/chessboard.cpp:397-404`）不清它 → `mousePressEvent` 的
`if (isReplayMode()) return;`（`src/chessboard.cpp:296`）永久生效，只能重启程序。
`onReplayModeExited`（`src/mainwindow.cpp:176`）是死代码。

修复：`reset()` 里 `m_replayGameId = -1; emit replayModeExited();`（一行级）。

### A6【高】线程模型：两个 detached 线程与对象生命周期脱钩

- `src/mainwindow.cpp:83-87`：`std::thread loadThread([this]{ ui->gameWidget->startupLoad(); }); loadThread.detach();`
  捕获 `this`/`ui` 并访问成员；关窗即 UAF。
- `src/mainwindow.cpp:232-307`：`onSelfPlay` 同样 detached，线程里直接 `board->selfPlay()`。
- `src/chessboard.cpp:567-596` 的 `selfPlay()` 在后台线程 `chess.reset()` / `moveForward()`，
  而 `state` 仍是 `STATE_IDEL` → 玩家可同时点击落子（`src/chessboard.cpp:295` 的守卫不生效）。

### A7【高】共享数据无锁 + `wakeAll()` 未持锁

- worker 写棋盘 `src/chessboard.cpp:369-371`，GUI 读棋盘 `src/chessboard.cpp:236-242`（`paintEvent`）→ 数据竞争。
- `src/chessboard.cpp:330-334` 在**未持有** `mutex` 时 `condit.wakeAll()`（Qt 要求持锁，
  否则丢唤醒 → AI 可能永不落子）；同一文件 `src/chessboard.cpp:391` 是持锁唤醒的，前后不一致。
- `src/chessboard.cpp:365/378` 在工作线程里直接调 `update()`（Qt 不支持跨线程操作 QWidget），
  而 `src/chessboard.cpp:393` 用的是正确的 queued 调用。

### A8【高】`Steps` 无锁全局对象池被多线程并发使用

`src/stone.h:48-73`：`std::list<Step*>` 的 `get()/put()` 无同步，
而 `ChessBoard::process` 线程、`backgroundTrainLoop` 线程、`onSelfPlay` 的 detached 线程
都会经 `chess.sample()` 访问它 → 链表指针破坏 / 堆破坏（典型"随机崩"）。

### A9【高】RL 即时奖励恒为 0（吃子信号完全丢失）

`src/pgagent.cpp:95-109`
```cpp
float PGEagent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr || victim->alive == false) return 0.0f;   // ← 此时已 alive=false
    if (victim->type == Stone::TYPE_JIANG) return 100.0f;           // ← 不可达
```
调用点在落子**之后**：
- `src/pgagent.cpp:301` `moveForward()` → `src/pgagent.cpp:304` `computeReward()`
- `src/ppomcts_agent.cpp:616` → `:653`（以及 `:895` → `:918`）
- `src/dqnmcts_agent.cpp:678` → `:733`（`recordExperience` 内）

`Stone::moveTo()` 第一步就是 `dst->alive = false;`（`src/stone.h:286`），
故这些调用**永远返回 0**，吃将的 `+100` 分支不可达。三个 agent 只能靠终局 ±1 学习。
唯一顺序正确的是 `src/dqnagent.cpp:253-256`（先算奖励再落子）。

修复：在 `moveForward` **之前**求奖励，或把被吃子价值固化进 `Step`。

### A10【高】"无合法走法"哨兵用错常量 → 会拿默认 `Step` 去落子

`Step()` 默认 `id = 0`（`src/stone.h:19`，即 `ID_RED_CHE1` = 红车），
而调用方判断的是 `Stone::ID_NONE`(33)：
`src/pgagent.cpp:409`、`src/dqnmcts_agent.cpp:414/546/670/727`、
`test/test_pg_main.cpp:104`、`test/test_dqn_main.cpp:78`、
`test/test_ppomcts_main.cpp:131`、`test/test_dqnmcts_main.cpp:85`。
另有 6+ 处脆弱魔法判断：`step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0`
（`src/chessboard.cpp:361`、`:583`、`test/test_main.cpp:69`、`test/test_utils.h:123`、`test/test_mcts_main.cpp:59,198`）——
`pos == (0,0)` 是黑方底线起点，合法走法可能误命中。

修复：给 `Step` 加 `bool valid`，或返回 `std::optional<Step>`。

### A11【高】GUI 中 DQN+MCTS 实际在下随机棋

`src/chessboard.cpp:490` `return m_sfDQNMCTS->selectMove(color, 6, true);`
（注释仍写"400次迭代"）。`src/dqnmcts_agent.cpp:363-370`：
```cpp
if (training) {
    float r = (float)std::rand() / (float)RAND_MAX;
    if (r < dqn.exploringRate && !nodes[rootID].childIDs.empty()) {
        int idx = std::rand() % (int)nodes[rootID].childIDs.size();
        bestChildID = nodes[rootID].childIDs[idx];      // 纯随机子节点
```
构造时 `exploringRate = eps = 1.0f`（`src/chessboard.cpp:485`），
而衰减只发生在 `endOnlineEpisode`/`learn`（`src/dqnmcts_agent.cpp:764-778`）——GUI 从不调用
（见 A15），所以探索率恒为 1.0 → 每步都随机。

### A12【高】后台训练线程与 AI 线程竞争同一 agent；agent 懒创建无锁

- 写者持锁：`src/chessboard.cpp:743-761`（`loadModel`/`loadPolicy` 直接改写网络权重）
- 读者**不持锁**：`src/chessboard.cpp:452-461`（`selectMove` 做前向）
→ 一边重写权重一边推理。
- `aiThink`（`src/chessboard.cpp:454-459`）与 `backgroundTrainLoop`（`:663-684`）
  都可能 `new` 同一个 agent，前者不持锁 → 双检失效、指针被覆盖泄漏。

### A13【高】关窗会阻塞在训练线程 `join()` 上

`src/chessboard.cpp:640-646`：
```cpp
void ChessBoard::stopBackgroundTraining()
{
    m_bgTraining = false;
    if (m_bgTrainThread.joinable()) { m_bgTrainThread.join(); }
```
而一轮训练是"4 局 × 50 次 MCTS 模拟 × ≤200 步"（`src/chessboard.cpp:706-741`），
线程只在循环顶部检查标志 → GUI 线程可能被冻结几十秒到几分钟。

### A14【高】SQLite 连接跨线程使用；写库接口根本没接线

- 连接在加载线程创建：`src/chessboard.cpp:32`（由 `src/mainwindow.cpp:84-87` 的 detached 线程调用）
- 查询在 GUI 线程：`src/mainwindow.cpp:105-107`
- `src/gamedb.cpp:54` 用固定连接名 `"chess_connection"` 的单一 `QSqlDatabase` 成员
  → Qt 判定"connection does not belong to the calling thread"，`exec()` 失败，
  历史对局列表恒为空。
- `recordMove/startGame/endGame`（`src/gamedb.cpp:133/156/188`）**全仓无调用点**；
  `m_currentGameId/m_moveCount/m_dbEnabled`（`src/chessboard.h:126-128`）是死成员。
  docs/analysis.md §4 描述的"每步落库 → 可回放"实际未接线。

### A15【高】在线训练接口在 GUI 侧全部未调用

`recordExperience` / `endOnlineEpisode` / `trainAfterMove` / `recordOnline` / `beginOnline` /
`endOnline` 只在自己文件内部或 `warmupFromCurrent` 里出现，`chessboard.cpp` 一次都没调。
后果：`DQNAgent::trainAfterMove`（`src/dqnagent.cpp:643`）、
`PGEagent::beginOnline/recordOnline/endOnline`（`src/pgagent.cpp:474-513`）是死代码；
DQNMCTS 的 `m_trainingMode` 恒 `false`，但 `aiThink` 却传 `training=true`（见 A11）。

---

## 二、中严重度

| # | 问题 | 证据 |
|---|------|------|
| B1 | `moveForward` 丢弃 `moveTo()` 返回值，`moveBack` 无条件复活被吃子 → 把旧局面的 `Step` 应用到新局面时棋盘会被写坏 | `src/chess.cpp:311` vs `src/chess.cpp:328-332` |
| B2 | `Chess::reset()` 不清 `history`，也不复位 `m_moveCount`/`m_currentGameId`/`m_replayGameId` | `src/chess.cpp:277-306`、`src/chessboard.cpp:397-404` |
| B3 | `reset()` 直接把 `state` 打回 IDLE 而不等待/取消 worker → worker 随后把旧局面的走法落到新棋盘 | `src/chessboard.cpp:397-404` + `:369-371` |
| B4 | Alpha-Beta 叶节点处理不对称：只有 MAX（黑方）做静态搜索，MIN 节点纯静态评估 → 奇偶深度系统性偏差、红方吃子序列看不见 | `src/abagent.cpp:266-268` vs `:317-319` |
| B5 | 静态搜索用完全开放窗口 `(-value_infi, +value_infi)` → 丢掉父节点边界，`standPat >= beta` 永不成立，等于不剪枝 | `src/abagent.cpp:318` |
| B6 | 搜索无置换表 / 迭代加深 / 时间控制；`totalReward` 贯穿全搜索却从未被读取（死参数） | `src/abagent.cpp:87-118,256-349` |
| B7 | 搜索内没有"重复局面 = 和棋 0 分"的概念 → AI 会来回推磨 | `src/abagent.cpp:77-127` |
| B8 | 搜索深度三处说法不一致：注释"深度=8"、UI 标签"深度=4"、实际 `ABAgent abAI(env, 5)`；PPO+MCTS 注释 800 实际 80；DQN+MCTS 注释 400 实际 6 | `src/chessboard.cpp:428,433,443-444,480,490`、`src/mainwindow.cpp:187` |
| B9 | 回放构造 `Step` 时 `step->nextId = dbStep.toX`（把列坐标当棋子 id），而 `moveForward/moveBack` 会用它索引 `m_children` | `src/chessboard.cpp:808,844,866` |
| B10 | `getStonePos` 负数整数除法向零截断 → 棋盘上/左 20px 空白被当成第 0 行/列；`x<0 \|\| y<0` 分支永不可达 | `src/chessboard.cpp:127-135` |
| B11 | 硬编码几何、无 `resizeEvent`/`sizeHint`；`.ui` 的 `minimumSize` 600×600 < 绘制所需 614 高 → 压到最小会裁掉最后一排；窗口尺寸三处不一致（`setFixedSize(900,650)` / `.ui` 830×650 / docs 750×650） | `src/chessboard.cpp:113-116,227-233`、`src/mainwindow.ui:26-31`、`src/mainwindow.cpp:18` |
| B12 | `saveModel/loadModel` 无条件 `return true`（底层失败码被丢弃）→ UI 会报假的"保存成功" | `src/dqnagent.cpp:626-636`、`src/dqnmcts_agent.cpp:783-793`、`src/ppomcts_agent.cpp:966-988` |
| B13 | `selfPlay()` 从不返回平局（`score == 0` 也返回红胜）→ "平局"提示不可达 | `src/chessboard.cpp:567-596`、`src/mainwindow.cpp:251-253` |
| B14 | `static ABAgent abAI(chess, 8)` 绑定**第一个** `Chess`，而 `playOneGame` 每局新建局部 `Chess` → 第二局起对已析构对象搜索（UB）。`test_ab` 的段错误与此也有关系 | `test/test_main.cpp:35`、`:49` |
| B15 | `PGEagent::train` 里 `currentColor = COLOR_BLACK` 让黑先走；无走法分支后会二次 `reinforce1` + 二次计数 | `src/pgagent.cpp:209`、`:237-244` + `:345-353` |
| B16 | `GameDatabase::close()` 先 `removeDatabase()` 而 `m_db` 仍持有该连接 → Qt 警告；`deleteGame` 第一条 `exec()` 未检查返回；`recordMove` 每步一次自动提交（一局 200 次 fsync），无事务 | `src/gamedb.cpp:66-75`、`:337`、`:156-183` |
| B17 | QSS 主题完全未接线：`QssLoader::load()` 零调用、`main.cpp` 无 `setStyleSheet`、全仓无 `setStyleSheet`。（同步后 `res.qrc` 已被真正编入 `chess` 目标，但资源仍无人读取） | `src/qssloader.hpp:5-18`、`src/main.cpp:6-13` |
| B18 | `DQN` 的 Q 头是 `Layer<Sigmoid>`（值域 `(0,1)`），而 `DQNAgent` 的奖励含终局 ±1 与被吃红子产生的负值 → 结构上无法表示负 Q。上游只修了 `convdqn.cpp`，`dqn.cpp` 没修 | `src/rl/dqn.cpp`（`snakeAI` 默认分支与 chess 分支都是 `Layer<Sigmoid>`）、`src/dqnagent.cpp:119-132,264` |

### B19【高】`Tensor::MM::*` 是所有 RL 训练慢的根因：**0.11 GFLOP/s**

实测（Release / MSVC 14.44 / `/O2` / 本机，微基准 `MM::ikkj` 与 `MM::kikj`）：

```
MM::ikkj  (90x90)*(90x90)     :  13.274 ms/call   18.21 ns/MAC   0.11 GFLOP/s
MM::ikkj  (90x360)*(360x90)   :  53.016 ms/call   18.18 ns/MAC   0.11 GFLOP/s
MM::kikj  (360x90)^T*(360x90) :  53.088 ms/call   18.21 ns/MAC   0.11 GFLOP/s
```

原因在 `src/rl/tensor.hpp:941-984`：三重循环的内层每次乘加都要调两次
`operator()` → `posOf()`（`src/rl/tensor.hpp:535-544`）：

```cpp
inline std::size_t posOf(Index ...index) const
{
    int indexs[] = {index...};          // 每次访问都在栈上重建索引数组
    std::size_t pos = 0;
    std::size_t N = sizeof... (Index);
    for (std::size_t i = 0; i < N; i++) {
        pos += sizes[i]*indexs[i];      // sizes 是 std::vector，每次读都走一次间接
    }
    return pos;
}
...
x(i, j) += x1ik*x2(k, j);              // 两次 posOf + 两次 vector 取值
```

即每个 MAC 有两组"建数组 + 循环 + 读 vector"，编译器无法向量化
→ 比同规格的手写循环慢约 2~3 个数量级。

**这一项解释了零节里全部的超时**：我做过对照实验——把 DQN 主干从 chess 的
`MOE<16,16>`+`TransformerBlock<16>`（约 1.55M 参数）换成 snakeAI 审计过的默认
`MOE<8,4>`（约 8 倍小），`test_dqn` **仍然超过 1800 s**。所以超时不是网络结构问题，
而是矩阵乘的实现问题。（上游只对 `conv2d()` 做了扁平 stride 重写，
`Tensor::MM` 一直没动。）

修复方向：把 `MM::ikkj/kikj/ikjk` 内层改成先取行指针再线性扫描，例如

```cpp
float *xr = x.val.data() + i*x.sizes[0];
const float *x2r = x2.val.data() + k*x2.sizes[0];
for (std::size_t j = 0; j < x.shape[1]; j++) xr[j*x.sizes[1]] += x1ik*x2r[j*x2.sizes[1]];
```

（`sizes` 提到循环外，`posOf` 不再参与内层），预计可提升 1~2 个数量级。
这一步做完再看是否需要动网络结构。

---

## 三、低严重度 / 代码质量

- **重复代码**：`encodeState / getLegalActions / stepToActionIdx / computeReward / warmupFromCurrent`
  在 4 个 RL agent 里几乎逐字重复；`PIECE_VALUES` 有 4 份定义
  （`src/pgagent.h:41`、`src/dqnagent.cpp:7`、`src/ppomcts_agent.h:37`、`src/dqnmcts_agent.h:36`）；
  MVV-LVA 排序在 `src/abagent.cpp:132-153` 与 `:201-218` 抄了两遍。
- **死代码**：`Chess::isInCheck/pushHistory/isRepetition`、`Chess::get`（`src/chess.cpp:272`）、
  `StoneMap::show`/`Stone::show`（`src/stone.h:154,326`）、
  `chessboard.h:57-59` 三个回放访问器、`MCTSNode::parentID`/`AZNode::parentID`/`DQNMCTSNode::parentID`、
  `AgentBase::resetState`、4 个 `warmupFromCurrent`、`DQNAgent::trainAfterMove`、
  `PGEagent` 的 online 三件套、`gamedb.cpp:246/330`、`qssloader.hpp` 全文件、
  `src/rl/qlstm.*`（无任何 include）、`src/rl/rl.pri`（qmake 遗留，引用的 `bpnn.cpp`/`conv2d.cpp`/`lstmnet.h`/`util.h` 都不存在）。
- **`Step::operator=` 不复制 `reward`**（`src/stone.h:23-33`），而拷贝构造会复制 → 语义不一致。
- **`Chess::evaluate()` 的 `getPositionValue(int type, ...)` 形参 `type` 未使用**（`src/chess.cpp:109`）。
- **`abagent.h:26-27` 形参名误导**：`minimizeAlpha(..., double beta, ...)` 传的其实是父节点边界，
  不是本节点的 beta（docs §5 第 1 条反复出错就源于此）；建议改名 `parentBound`。
- **`STATE_IDEL` 拼写错误**（`src/chessboard.h:33`）。
- **`selectID` 的"空"值在 `-1` 与 `Stone::ID_NONE`(33) 之间混用**
  （`src/chessboard.cpp:90,298,331,390,400`）。
- **const 正确性**：`src/pos.h:22-27` 所有运算符非 const；`stone.h:105,138,154`；
  `src/chess.h:71 double evaluate();` 只读却非 const。
- **`alive` 用 `int` 却到处和 `bool` 比较**（`src/stone.h:244`）。
- **测试无断言**：6 个 `test_*_main.cpp` 全是 printf + 自对弈统计，
  `test/test_utils.h:6` 的 `<cassert>` 全仓没用过一次 → A1/A2/A3/A9/A5 这类缺陷一个都抓不到。
- **文档与代码脱节**：`docs/analysis.md` §一的文件树仍是根目录时代的旧布局；
  §5 第 3 条把"新增 isInCheck"列为已修复（实际零调用）；
  §六 第 3 条说"MCTS 未实现"（实际 `src/mcts.cpp:259-340` 已完整实现四阶段并接入 GUI，
  见 `src/chessboard.cpp:448-450`）；`docs/task_progress_mcts.md:74-75` 说 GUI 用 5000 次迭代（实际 800）。
- **`ChessBoard` 在 AI 线程里 `emit` 后由队列在主线程弹模态框**，
  模态事件循环期间 `reset()` 与 worker 可能重叠（与 B3 同源）。
- **`res.qrc` 之前从未参与构建**（虽然 `AUTORCC ON`，但 `.qrc` 未列进源文件）；
  随本次 RL_CORE 改造已接入（`qrc_res.cpp` 现在会生成）。

---

## 四、与 `docs/analysis.md` §六"仍存在的问题"的核对

> 下表是**首次审查时**的核对结论。其中 1、2 两条**后来都已修复**
> （见 A2/A3/A4/B7/B13），`docs/analysis.md` §六 也已同步更新为"已修复 + 当前真正的待办"。
> 保留此表是为了记录"当时的判断"，不再代表现状。

| docs 条目 | 结论（首次审查时） | 说明 |
|-----------|------|------|
| 六-1 "AI 缺乏应将/将军意识，`isInCheck()` 已实现但未集成" | **确认存在，且比文档更严重** → **现已修复** | `sample()` 无过滤（A2）；`isInCheck()` 零调用；玩家侧也不校验；没有将杀判定（A3）；飞将被实现成吃将走法而没有禁照面规则。**修复**：`Chess::isAttacked/isLegalMove/hasLegalMoves` + `getResult()` |
| 六-2 "游戏无法自然结束，缺长将检测" | **确认存在，表述需修正** → **现已修复** | `pushHistory()` 从未被调用（A4）；没有 60 回合限着；`selfPlay` 从不判和（B13）。修正：GUI 对局**会**结束，但结束条件是"将/帅被吃掉"而不是将杀。**修复**：`history` + `halfMoveClock` + `isDraw()`（三次重复 + 120 半回合），arena 到上限判和 |
| 六-3 "MCTS 未实现" | **已过期，应删除** | `src/mcts.cpp:78-181,194-239,259-340` 已实现 select/expand/simulate/backpropagate + UCB1，并接入 GUI `src/chessboard.cpp:448-450` |

同时，docs §5 第 3 条（"新增 `isInCheck()`"）应从"已修复"移到"仍存在"——
代码在，但从未被调用，等价于未实现。

---

## 零之二点十三、PPO 系列训练效率改造（P1–P7）暴露的问题

这一轮的目标是"同样的算力产出/消化更多训练数据"（改动与实测见
`docs/agents_design.md` §17）。下面是这一轮**发现的问题**，分三类：修掉的、判定为"设计使然"的、
以及**没修的**。

### (1) 修掉的

**① 奖励/价值目标的视角口径不一致（老一代 3 个 agent，9 处）—— 不报错、只让人学不动。**
PG / DQN / DQN+MCTS 的 `encodeState` 是"绝对坐标 + 黑子为正"，网络表达的是**对黑方的价值**；
而 `computeReward` / `rolloutFromCurrent` 给的是**走子方视角**（吃子者为正），
同一批经验里的终局常量 `(gameResult == COLOR_BLACK) ? 1 : -1` 又是黑方视角。
于是**一条轨迹里两种口径混用**，红方那一半样本的目标整体反号。
上一次只改了 `computeReward` 自己、没有跟着改消费端，这一轮补齐：新增
`stone.h` 的 `moverRewardToBlackFrame()` 两个重载，修 `pgagent.cpp`（`exploreAndTrain`，
这条路径**没有**终局覆盖，反号是实打实的）、`dqnagent.cpp`（4 处，含无调用方的 `trainAfterMove`）、
`dqnmcts_agent.cpp`（3 处），以及 `evagent.cpp` 的 `exploreAndTrain` 手工评估项
（它用 `next` 而状态与 `searchV` 都是 `turn`，两项按 0.5/0.5 相加，等于手工那一半一直在跟自己抵消）。
**判定为本来就对的**：`trainVsRandom` 各点（AI 只执黑，黑方视角 == 走子方视角）、
`sacazagent.cpp` 全部（规范视角，全程走子方口径）。逐点判定表见 `docs/agents_design.md` §17.2。

**② 损失曲线的两个读数恒为常数（既有 bug）。** `accumulateGrad`/`trainStep` 在
`backward()` **之后**才读 `v[0]` 与 `policy[i]`，而 `backward()` 会把层的输出清零 ——
`lastLoss` 恒等于 `target²`、`lastActorLoss` 恒等于 `-ln(1e-8)·Σtarget`，
17 轮里一个数都没变，而同一时间策略已经动了 3 个数量级。改成在 `backward()` 之前算标量损失。

**③ `PpoSelfPlayMT` 的 learner 退出条件写在"没攒够一批"的 else 分支里 → 永不退出。**
池子一旦涨到 `learnBatch` 以上就永远"够学"，那个分支再也进不去，worker 全部收工之后
learner 还会拿同一批旧样本空转。改成先判停、再判够不够。

### (2) 设计使然 / 记录备查

* **worker 的局数会超跑**：配额检查在"每局开始前"，W 条 worker 各自看到"还差一点"就会各下一局，
  最多超出 `W-1` 局（8 局的目标在 W=8 时能跑出 15 局）。不是 bug，但**任何拿"实际局数/秒"
  算加速比的写法都会被它抬高**，所以基准现在把实际局数单列并标注"超跑"。
* **PG 的物质收益被整条丢掉**：`PGEagent::train` / `warmupFromCurrent` 算出即时奖励之后，
  会被终局常量把整条轨迹的 reward 覆盖（三处出口全覆盖），所以 PG 实际只用"纯终局 ±1"学习。
  它是自洽的（都是黑方视角），所以**没改** —— 要接上得先把量纲调平
  （`computeReward` 给的是 `value*10`，与 ±1 差一个量级），列为待办。

### (3) 没修的（本轮最重要的结论）

**④ 多线程分身训练的并行收益上限只有 ~1.5×，因为搜索是访存带宽受限的。**
实测（`bench_ppo_mt`，i7-12650H = 10 核/16 逻辑，80 模拟/步、batch=64、epochs=2，
每配置 12 局 × 2 轮）：

| workers | 墙钟秒 | 相对串行基线 | 每线程 ms/模拟 |
|---|---|---|---|
| 串行基线 | 40.5 | 1.00× | 0.422 |
| 1 | 38.7 | 1.05× | 0.384 |
| 2 | 31.8 | 1.27× | 0.541 |
| **3** | **26.3** | **1.54×** | 0.698 |
| 4 | 31.3 | 1.29× | 1.001 |
| 6 | 34.0 | 1.19× | 1.364 |

把 learner 完全摘掉（`--batch` 给一个池子永远够不到的大数）再扫一遍，**曲线形状不变**
（一样在 2–3 条 worker 就到顶），所以 learner 不是限制；诊断输出也算得出来：
`学习轮数 / 理论上限` 只有 0.24–0.37，learner 一直是**喂不饱**的那一方。
而每线程的每模拟成本从 1 条 worker 的 0.294 ms 涨到 8 条的 1.929 ms（**6.6×**）——
这是带宽饱和的典型形状。折算总带宽：每次模拟要流过的权重约 3.0 MB，其中**策略头
`64×8100×4B = 2.07 MB` 占了大头**；实测聚合吞吐在 ~5100 次模拟/秒 到顶 → ≈15 GB/s，
正是笔记本双通道内存的量级。而 8100 维动作空间（`fromCell*90 + toCell`）注定了
每次模拟都要把这 2 MB 权重从头到尾流一遍。

**突破方向（未做，列为待办）**：让一次权重读取服务多条样本 ——
即多棵树的叶子**批量前向**（AlphaZero 那套 server + 批量推理 + virtual loss），
或把权重降到 fp16/int8 把字节数砍半。两者都是独立的大改动。
**结论：在当前硬件上靠堆线程最多 ~1.5×，同样的算力投到数据效率收益大得多**
（P3/P4/P6：同一批数据多次复用、"重放一条 7.7 ms vs 重生成一条 55 ms"、镜像白拿一倍数据）。

### (4) 验证方法学（这一轮最贵的教训）

**⑤ 吞吐指标用"均值之比" → 把加速比抬高了约 45%。** `bench_ppo_mt` 第一版每轮算
`局数/秒` 再对轮次取平均，而 `mean(1/t) ≠ 1/mean(t)`，重复轮次一抖动就系统性偏高：
那一版报出的是"W=4 时 2.22×"，改成**固定目标局数 / 平均墙钟秒**（与基线同口径）后是 1.54×。
→ **吞吐类指标要以"固定工作量的墙钟时间"为主口径，比值类的量不要先取比值再平均。**

**⑥ 用 shell 的文本 cmdlet 来回写源文件，一次性毁掉了 5 个文件里的全部中文注释。**
当时只是想把源码注释里的 `§12` 改成 `§17`，写成 `Get-Content -Raw` + `-replace` + `WriteAllText`。
而**这个环境里的 `pwsh` 实际是 Windows PowerShell 5.1**：`Get-Content -Raw` 按 **ANSI（GBK）**
解码 UTF-8 源文件、再按 UTF-8 写回，一次往返就把中文变成 `瑙嗚鎹㈢畻` 这类乱码；
更糟的是 **GBK 的双字节序列会吞掉紧跟其后的 ASCII**（`name = "车";` 的闭引号被吞成
`name = "?;`，直接是语法错）。反解不完全可逆（每个文件丢 1000–3000 字节），只能 `git checkout` 回滚；
回滚时确认了"工作区相对 HEAD 的差异全是注释"（代码一行没动），所以**代码层没有损失**，
丢的是这 5 个文件里上一轮补充的注释（`stone.h` / `pgagent.cpp` / `dqnagent.cpp` /
`dqnmcts_agent.cpp` / `evagent.cpp`，合计约 270 行注释）。两条规矩：
* **绝不**用 shell 的 `Get-Content`/`Set-Content`/`Out-File` 改仓库里的 UTF-8 文本文件；
  要么用带编码参数的 .NET API 显式指定 UTF-8，要么用专门的编辑工具。
* 判断"文件到底是什么内容"，**不要只信同一个 shell 里的 `Get-Item`/`ReadAllBytes`** ——
  这次它们返回了过期的大小（26273）和完全无关的二进制内容，而
  `git status` + `git diff --numstat` + `cmd /c dir` + `cmd /c type` + 行读取工具
  五路都说是正常的 771 行 C++ 文本。**跨进程交叉验证才能定案，否则会去"修"一个不存在的问题。**

---

## 五、当前待办与优先级（未修复项）

> **本节已按当前状态重写。**
> 原来的 P0/P1 清单（`StoneMap` 初始化、走法合法性、
> 将杀判定、`detach()`、`Steps` 加锁、SQLite、`MM` 扁平化……）**已经全部完成**，
> 见"零、修复进度"的 A1–A19 / B1–B19。下面只列**当前仍未做**的事。

### P1（正确性 / 会咬人的潜在问题）

1. **`MM::ikjk` / `kijk` 的 SIMD 内核改成累加**（现在是赋值，标量是累加）。
   当前所有调用点 kdim=1 走标量所以没有实际错误，但 `lstm.cpp:143-148` 那种
   "连写 5 次累加到同一个 `delta.h`" 的写法在多列输入下会静默丢梯度。
   **只要有人把训练改成多样本批量，这就是一个不会报错的正确性 bug。** → C7
   - 修法：`simd/avx2func.hpp` 与 `sse2func.hpp` 的 `MatMul::ikjk/kijk` 里
     `z[i*zCol + j] = dot(...)` 改成 `z[i*zCol + j] += dot(...)`；
     顺带在 `simd_ops.hpp` 的契约注释里把"四个内核都是累加"写死。
2. **补 `gemv_kikj`**（反向 GEMV）。`ei = wᵀ·e` 现在是最慢的一步：
   0.991 vs 前向 0.102 ns/MAC（9.7×）。训练步的成本几乎没有被 SIMD 优化到。 → C8
   - 修法：`Tensor::MM::kikj` 里加一个与 `ikkj` 对称的 GEMV 分支
     （形状判据：`x.shape[1]==1 && x2.shape[1]==1`），内核为
     `ei[i] += Σ_k w[k][i]·e[k]`（行主序下是跨步点积，可用
     `Instruct::dot` 对一个转置缓冲做，或按列 gather 后点积）；
     注意 `w` 是 `[out,in]`、结果长度是 `in`，与 `gemv_ikkj` 的方向相反。
3. **`MM::*` 入口加 debug 断言**（维度与连续性）。写微基准时用错维度会直接越界崩掉，
   现在只有 `Tensor::MM` 内部的 `contiguous2d` 判据，形状不匹配时是**静默算错**。
   → 附注
4. **SQLite 跨线程 / 写库未接线**：连接在加载线程建、查询在 GUI 线程用；`recordMove/
   startGame/endGame` 全仓零调用，"每步落库 → 可回放"实际未接线。需要设计决定
   （每线程连接 vs 全部集中在 GUI 线程）。 → A14

### P2（棋力 / 训练质量）

5. **训练数据管线**：预训练的利弊分析见 `docs/agents_design.md` §10。要点：
   DQN 系当前"每步一次更新"破坏回放缓冲的 i.i.d. 假设，建议改成
   **只写回放、攒批更新**；PPO 系的优势估计在稀疏终局奖励下需要**显式自举**
   （否则 64 步内没有终局 → 优势几乎全 0 → 梯度是噪声）。
6. **`DQN` 的 Q 头是 `Layer<Sigmoid>`**（值域 `(0,1)`），而奖励含终局 ±1 与被吃红子
   产生的负值 → 结构上无法表示负 Q。上游只修了 `convdqn.cpp`，`dqn.cpp` 没修。 → B18
7. **搜索的时间控制与迭代加深**：`AB_DEPTH`/`MCTS_SIMS`/`PPO_SIMS`/`DQNMCTS_ITERATIONS`
   都是固定值（已集中为常量，不再散落在注释里）；EVAB 有 `timeBudgetMs` 但其余没有。
   **稀疏 MoE + TB 专家那个变体现在也是固定 16 次模拟** —— 它最需要时间控制
   （一次模拟 10 ms，同一个预算下不同局面能跑的模拟数差很多）。
8. **QSS 主题仍未接线**：`QssLoader::load()` 零调用（`res.qrc` 已编入目标，资源无人读）。
   → B17
9. **没有一个骨干拿到棋力结论**：`bench_moe` 的 A/B/C/D 等时对弈**全是和棋**
   （连自我对照也是），因为参赛的都是随机初始权重，预训 3 局等于零。要谈"稀疏 MoE
   能否提升效果"必须投入真正的训练预算（自对弈几千局量级）+ 预训=0 的对照，
   这件事**还没做**，见 `docs/agents_design.md` §11.4.2 (6)。
10. **突破多线程搜索的访存墙**（零之二点十三 ④，本轮最重要的未修项）：靠堆线程最多 ~1.5×，
    要真正提速得让**一次权重读取服务多条样本** ——
    (a) **批量叶子前向**：一次选多个叶子（virtual loss 防重选）再合成一个 batch 前向，
        把 2 MB 策略头权重的流量摊到 batch 条样本上。这会改 MCTS 的选择-评估循环，
        是最有收益也最伤筋动骨的一项；
    (b) **权重低精度化**（fp16/int8）：字节数直接砍半/砍到 1/4，不动算法结构，
        但要重新验一遍梯度与数值稳定性；
    (c) 把 8100 维策略头换回更小的动作表示 —— **与已定的动作空间方案
        （`fromCell*90 + toCell`）冲突，不作为选项**，只在此记录取舍。
11. **PG 的物质收益接上**（零之二点十三 (2)）：先定标（`computeReward` 的 `value*10`
    与终局 ±1 的量纲差一个数量级），再把即时奖励与终局奖励按权重相加，
    并加一条"轨迹里必须同时含即时项与终局项"的断言。
12. **PPO 的在线路径（`exploreAndTrain`）还没走回放池**：P4 的回放池 + 梯度累积只接了
    `commitEpisode`（自对弈路径），GUI 里"走子前先探索再训练"那条路仍是 `learnSelfPlay`
    逐样本更新。要么统一，要么在文档里写清为什么两条路 deliberately 不同。
13. **4 个慢测试没进 ctest**（`test_dqn` 466 s、`test_dqnmcts` 410 s、`test_evab` 680 s、
    `test_ppomcts` 51 s，见本轮实测）：现在只能手动逐个跑。建议给它们注册成
    ctest 的 `-L slow` 标签（默认不跑，`ctest -L slow` 才跑），而不是完全不注册 ——
    否则"改完记得手跑那 4 个"只存在文档里。

### P3（工程化）

14. **`GameDatabase::recordMove` 每步一次自动提交**（一局 200 次 fsync），无事务。 → B16
15. `PGEagent::train` 里 `currentColor = COLOR_BLACK` 让黑先走、无走法分支后二次
    `reinforce1` + 二次计数。 → B15
16. 清理死代码、补 `const`、抽出 5 个 RL agent 的公共基类（`exploreAndTrain` 已经在
    `AgentBase` 上，其余仍各写各的）。`DQNAgent::trainAfterMove` 现在无调用方（本轮顺手修了
    它的视角口径，但更该做的是删掉或接线）。
17. 文档对齐：`docs/analysis.md` 的文件树与 §5/§6 条目、`task_progress_*.md` 的历史
    迭代数（那些是"当时的记录"，不必改，但要在顶部注明已过期）。
