# 中国象棋 (Qt6 + C++17) — 带 AI Agent 与强化学习内核

一个用 Qt6 写的中国象棋程序：完整的棋规、可玩的界面、**9 个可选 AI Agent**（从
Alpha-Beta 到 SAC + MCTS + AlphaZero）、一个**纯 C++ 的强化学习内核**（SIMD 加速、
自带稀疏 MoE），以及一整套**可复现的验证手段**（9 个 ctest 套件 + 5 个界面自动化脚本 + 1 个基准）。

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
* **实时比分** + **逐局明细列表**（每局一行，含本局环境奖励与手数）
* **训练损失曲线**（每个 agent 一条线）与**每局环境奖励曲线**（A/B 两条）：
  自绘控件，**双击可放大**到独立窗口（跟随源控件实时同步），支持**导出 CSV**
* 对弈结束后**静默保存权重**到标准路径（不弹窗；几百 MB 的写盘会显示"请稍候"沙漏）

### 工程

* 程序图标（窗口/任务栏 + exe 文件图标，由脚本生成）
* 载入/保存权重时的沙漏等待窗（复用思考指示器；**快的操作不闪窗**，延迟 300 ms 才显示）
* 权重文件格式 v2：**逐比特无损** + 结构指纹 + 每张量 CRC32 + **原子写入**，
  并且**兼容旧的十进制文本格式**

---

## AI Agent

界面上可选的 9 个 agent（`主界面 → 对战AI / A方 / B方`）：

| Agent | 说明 | 每步预算（本机实测） |
|-------|------|---------------------|
| **Alpha-Beta Pruning** | 经典 α-β + 静态搜索 | 深度 4，约 90 ms |
| **MCTS** | UCB1 蒙特卡洛树搜索 | 800 次模拟 |
| **Policy Gradient** | REINFORCE + baseline，走子前在线训练 | 预训 64 步 |
| **Deep Q-Network** | 双网 + 经验回放 + ε-greedy | 预训 64 步 |
| **PPO+MCTS** | AlphaZero 风格：搜索访问分布监督 actor | 80 次模拟 |
| **DQN+MCTS** | 用 Q 值做叶子估值 + 树搜索 | 200 次迭代 |
| **EVAB** | **学会评估的 Alpha-Beta**：置换表/迭代加深/排序 + 学习到的价值网按 `blend` 混合 | 深度 6 + 800 ms 上限 |
| **SAC+AZ** | **SAC + MCTS + AlphaZero**：最大熵 critic 给 PUCT 搜索估值，α 自动调节 | 256 次模拟，约 12 ms |
| **SAC+AZ (稀疏MoE+TB专家)** | 同上，骨干换成**稀疏路由 MoE + TransformerBlock 专家** | 16 次模拟，约 160 ms |

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

### `ctest`（9 个套件，全过）

```bat
ctest --output-on-failure
```

| 套件 | 盯什么 |
|------|--------|
| `test_ab` | Alpha-Beta 搜索正确性与基准 |
| `test_mcts` | MCTS 四阶段 + 树规模（较慢，约 4 分钟） |
| `test_rules` | 棋规回归：走法数 / 应将 / 自杀 / 照面 / 将杀 / 重复 / 限着 |
| `test_pretrain` | **探索不得改动真棋局**（逐字段比对） |
| `test_match` | arena 统计（交换先后手 / 比分归属 / 判和 / 中止）+ 每个 agent 的**训练损失上报** + **即时奖励符号约定** |
| `test_grad` | **有限差分核对 SIMD 之后的解析梯度** + MM 内核"累加 vs 覆盖"语义探针 |
| `test_weights` | 权重格式：逐比特往返 / 坏文件拒绝 / 失败不改动网络 / 老格式兼容 |
| `test_sparse_moe` | 稀疏不变量 / 与上游 `MOE` 的等价性 / 反向有限差分 / 辅助损失 |
| `test_sacaz` | 掩码 softmax 雅可比 / 走法合法性 / 软价值 α 恒等式 / 四种骨干 |

另外有 **`bench_moe`**（骨干 A/B/C/D 等时间对弈基准）**故意不进 ctest** —— 它跑真实对局、
依赖随机开局，放进去只是偶发失败：

```bat
build\...\bench_moe.exe --games=4 --plies=30 --budget=60 --pretrain=3
```

### 界面自动化（PowerShell + Windows UI Automation）

| 脚本 | 验证什么 |
|------|----------|
| `tools/verify_match_ui.ps1` | 真界面选 agent → 开局 → 断言比分/逐局明细/曲线有数据/**静默保存权重**/**双击放大窗与源控件逐字一致** |
| `tools/verify_thinking_ui.ps1` | 采样像素：思考中状态条出现、空闲/结束后干净 |
| `tools/verify_busy_ui.ps1` | 启动载入权重时弹沙漏、载完收起（**不残留**） |
| `tools/verify_busy_lazy.ps1` | 首次使用稀疏 MoE 变体（3×146 MB 懒加载）时弹沙漏 |
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
│   ├── app.rc app.ico app.png res.qrc              # 程序图标与资源
│   └── rl/                 # RL 内核 -> RL_CORE (纯 C++, 不含 Qt)
│       ├── tensor.hpp net.hpp layer.h ...          # 张量/网络/层
│       ├── simd_ops.hpp simd/ cpuinfo.hpp          # SIMD 分派与内核
│       ├── moe.hpp sparse_moe.hpp                  # 稠密 / 稀疏 MoE
│       └── dqn.cpp dpg.cpp ppo.cpp sac.cpp ...     # RL 算法
├── test/                   # ctest 套件 + bench_moe
├── tools/                  # 界面验证脚本 + 图标生成
└── docs/                   # 设计/审查/同步/理论分析 (见下)
```

代码量：`src/` 约 **2.9 万行**，`test/` 约 **0.5 万行**。

---

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/agents_design.md`](docs/agents_design.md) | 各 agent 的设计与实测；§12 参数量理论分析；§13–16 界面可视化/EVAB 修复/沙漏等待/静默保存与图标 |
| [`docs/issues_review.md`](docs/issues_review.md) | **问题清单与修复进度**（A/B/C 编号）、优化方法汇总（含实测数字）、当前待办 |
| [`docs/rl_sync.md`](docs/rl_sync.md) | 与上游 snakeAI `rl/` 的同步、chess 侧的差异、SIMD 之后梯度是否仍正确 |
| [`docs/xiangqi_capacity.md`](docs/xiangqi_capacity.md) | "多少参数量才能覆盖象棋求解空间"（~10⁴⁰ 参数 → 物理上不可能） |
| [`docs/analysis.md`](docs/analysis.md) | 文件树与模块分析 |
| [`docs/agent_evab_design.md`](docs/agent_evab_design.md) | EVAB 专篇 |

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
