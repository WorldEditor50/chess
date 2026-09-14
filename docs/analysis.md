# 中国象棋程序 - 全面功能分析

## 一、项目概览

这是一个基于 **Qt 6** 框架开发的中国象棋桌面游戏，集成了基于 **Alpha-Beta 剪枝** 的AI引擎。项目使用 C++17 编写，包含 GUI 界面（Qt Widgets）和 AI 算法两部分。

### 文件结构

```
chess/
├── CMakeLists.txt        # 构建配置 (Qt6 Widgets+Sql; src/rl 编译为 RL_CORE 静态库)
├── build_main.bat        # vcvars64 + cmake 配置/构建脚本
├── src/
│   ├── main.cpp              # 程序入口
│   ├── mainwindow.h/cpp/ui   # 主窗口 (Qt)
│   ├── chessboard.h/cpp      # 棋盘控件 (Qt 绘制 + 交互 + AI 线程 + Agent 对弈 arena)
│   ├── thinkingindicator.h/cpp # "AI 正在思考"指示器 (沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时)
│   ├── metricsview.h/cpp     # 自绘折线图控件 CurveChart (训练损失 / 环境奖励曲线) + 双击放大窗口
│   ├── busydialog.h/cpp      # 载入/保存权重时的"请稍候"弹窗 (里面复用思考指示器的沙漏)
│   ├── chess.h/cpp           # 棋局逻辑 + 评估函数 (+ Zobrist 键)
│   ├── stone.h/cpp           # 棋子基类与派生类 + 走法/对象池
│   ├── pos.h/cpp             # 坐标类
│   ├── aiagent.h             # AgentBase 接口 (+ exploreAndTrain / getExploreInfo)
│   ├── agentrollout.hpp      # 决策前探索的共用实现 (试走 + 原样回退)
│   ├── abagent.h/cpp         # Alpha-Beta 剪枝 agent (含静态搜索)
│   ├── mcts.h/cpp            # MCTS (四阶段已实现, 已接入 GUI)
│   ├── evagent.h/cpp         # EVAB: 学会评估的 Alpha-Beta (价值网 + TT/killer/history)
│   ├── sacazagent.h/cpp      # SAC + MCTS + AlphaZero (最大熵 critic 给 PUCT 搜索估值)
│   ├── pgagent.h/cpp         # 策略梯度 agent (RL::DPG)
│   ├── dqnagent.h/cpp        # DQN agent (RL::DQN)
│   ├── ppomcts_agent.h/cpp   # PPO + MCTS (AlphaZero 风格, 用 RL::PPO)
│   ├── dqnmcts_agent.h/cpp   # DQN + MCTS
│   ├── gamedb.h/cpp          # SQLite 棋局数据库
│   ├── qssloader.hpp         # QSS 样式加载工具 (目前未接线)
│   ├── appstyle.qss          # 样式文件
│   ├── res.qrc               # 资源文件 (内嵌 appstyle.qss + app.png 程序图标)
│   ├── app.rc / app.ico      # Windows exe 图标 (PE 资源; 由 tools/make_app_icon.ps1 生成)
│   └── rl/                   # RL 内核 -> 静态库 RL_CORE (纯 C++, 不含 Qt)
│       ├── simd_ops.hpp      # SIMD 分派层 (标量 / SSE2 / AVX2 三档)
│       ├── sparse_moe.hpp    # 稀疏路由 MoE (只算门控选中的 top-k 个专家; TopK==E 即稠密对照)
│       ├── simd/             # N-spirits 的 SSE2 / AVX2 张量内核
│       └── cpuinfo.hpp       # CPUID 探测 + "本构建启用了哪套内核"
├── test/
│   ├── test_utils.h           # 测试工具 (SearchStats/Timer/打印)
│   ├── test_main.cpp          # Alpha-Beta 沙盒 -> test_ab
│   ├── test_mcts_main.cpp     # MCTS -> test_mcts
│   ├── test_rules_main.cpp    # 棋规回归 (走法数/应将/将杀/重复) -> test_rules
│   ├── test_evab_main.cpp     # EVAB 搜索/对抗/训练 -> test_evab
│   ├── test_pretrain_main.cpp # 决策前探索不得改动真棋局 -> test_pretrain
│   ├── test_match_main.cpp    # Agent 对弈 arena 统计 -> test_match
│   ├── test_grad_main.cpp     # SIMD 之后的梯度传播 (有限差分) -> test_grad
│   ├── test_sacaz_main.cpp    # SAC+MCTS+AlphaZero -> test_sacaz
│   ├── test_sparse_moe_main.cpp # 稀疏路由 MoE (稀疏不变量/等价性/有限差分/辅助损失) -> test_sparse_moe
│   ├── test_weights_main.cpp  # 权重文件格式 (无损/校验/原子写/兼容老格式) -> test_weights
│   ├── bench_moe_main.cpp     # 骨干 A/B/C/D 等时对弈基准 -> bench_moe (只构建不注册)
│   ├── test_pg_main.cpp       # PG -> test_pg
│   ├── test_dqn_main.cpp      # DQN -> test_dqn
│   ├── test_ppomcts_main.cpp  # PPO+MCTS -> test_ppomcts
│   └── test_dqnmcts_main.cpp  # DQN+MCTS -> test_dqnmcts
├── tools/                # 界面验证脚本 (UI Automation / 像素采样 / 进程窗口枚举)
└── docs/                 # 本文档 + issues_review.md + rl_sync.md + agents_design.md
```

构建产物（`build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release`）：
`RL_CORE.lib`（17 个 TU 的静态库）+ `chess.exe` + **15 个测试/基准可执行文件**。
其中 `test_ab` / `test_mcts` / `test_rules` / `test_pretrain` / `test_match` / `test_grad`
/ `test_weights` / `test_sacaz` / `test_sparse_moe` 九个注册进了 `ctest`（`test_match` 需要
Qt 的 DLL：`ctest` 通过 `ENVIRONMENT_MODIFICATION` 把 Qt 的 `bin` 前置进 `PATH`）；
其余是分钟级的训练基准与对弈基准（`bench_moe` 会跑真实对局、依赖随机开局），
只构建不注册 —— 放进默认套件只会得到看起来像失败的超时。

文档：
* `issues_review.md` — 问题清单与修复进度（含"零之二点十"的即时奖励符号 bug）
* `agents_design.md` — 各 agent 的设计；§12 参数量理论分析、§13 界面可视化
* `xiangqi_capacity.md` — "多少参数量才能覆盖象棋求解空间"的完整推导
* `rl_sync.md` — 与上游 snakeAI `rl/` 的同步与差异（含权重文件格式 v2）
* `analysis.md` — 本文档

---

## 二、核心模块分析

### 2.1 基础数据结构 (`pos.h/cpp`)

- `Pos` 类：二维坐标，x 为行(0-9)，y 为列(0-8)
- 重载了 +、-、*、/ 运算符，支持坐标运算

### 2.2 棋子系统 (`stone.h/cpp`)

**类继承体系：**
```
Stone (基类)
├── Che   (车)   - 直线移动，中间无子
├── Ma    (马)   - 日字移动，蹩马腿
├── Xiang (相/象) - 田字移动，塞象眼，不过河
├── Shi   (仕/士) - 九宫内斜走一格
├── Jiang (帅/将) - 九宫直走一格 + 飞将
├── Pao   (炮)   - 直线移动，吃子需隔一子
└── Bing  (兵/卒) - 过河前只能前进，过河后可左右
```

**关键机制：**

1. **棋子标识**：使用 `Stone::ID` 枚举（0-31），红方 0-15，黑方 16-31
2. **走法生成**：`getPossibleSteps()` 虚函数生成该棋子的所有合法走法
3. **移动校验**：`tryMoveTo()` 虚函数校验是否能移动到目标位置
4. **对象池**：`Steps` 类作为 `Step` 对象池，避免频繁 new/delete
5. **棋盘映射**：`Stone::map` 类型为 `StoneMap<Stone>`，是 10×9 的二维数组，用于快速查找某个位置上的棋子
6. **走法结构**：`Step` 包含 `id`（己方棋子）、`nextId`（被吃棋子，无则为 NONE）、`pos`（起点）、`nextPos`（终点）

### 2.3 棋局逻辑 (`chess.h/cpp`)

**主要类：`Chess`**

- 包含 32 个棋子对象（红方 16 个、黑方 16 个）
- 引用 `Stone::children` 数组

**核心方法：**
| 方法 | 功能 |
|------|------|
| `reset()` | 重置棋盘到初始局面 |
| `sample(color, steps)` | 生成某方所有合法走法 |
| `moveForward(s, reward)` | 执行走法，更新累计收益 |
| `moveBack(s, reward)` | 回退走法 |
| `isGameOver()` | 检测游戏是否结束（将/帅被吃） |
| `evaluate()` | 局面评估函数 |
| `orderMoves(steps)` | MVV-LVA 走法排序 |

**评估函数：**
- 材质评估：各棋子基础价值（帅=1000, 车=0.5, 马=0.3, 炮=0.3, 仕/相=0.2, 兵=0.1）
- 位置评估：7 套 Piece-Square Table（PST），对每种棋子在不同位置有不同的加成

### 2.4 AI 搜索算法 (`chess.h/cpp`)

**主算法：Alpha-Beta 剪枝**

调用链：
```
alphaBetaPruning(color, depth)         # 顶层入口: 选最优走法
  -> maximizeBeta(color, depth-1, alpha, reward)  # MAX 节点: 最大化己方收益
    -> minimizeAlpha(color, depth-1, beta, reward) # MIN 节点: 最小化对方收益
      -> maximizeBeta(...)              # 递归交替
        -> quiescenceSearch(...)        # 叶节点静态搜索 (depth=0时调用)
```

**技术特点：**
1. **MVV-LVA 走法排序**：吃子走法按"最有价值受害者-最廉价攻击者"排序优先搜索，提升剪枝效率
2. **静态搜索**：在叶节点继续搜索吃子走法，缓解"水平线效应"
3. **Piece-Square Tables**：精细化的位置评估

**已知问题 (Bug)：**
- `minimizeAlpha` 中的剪枝条件 `if (r <= beta)` 在标准 Alpha-Beta 实现中是错误的，应该是 `if (r <= alpha)`，导致奇偶深度搜索行为异常
- `maximizeBeta` 中类似问题 `if (r >= alpha)` 应为 `if (r >= beta)`
- 叶节点评估只使用 `totalReward`（材质差），未结合 `evaluate()`（位置评估）
- AI 倾向于"推磨"（来回走子）而非积极将杀

### 2.5 GUI 界面 (`mainwindow.ui / chessboard.cpp`)

**棋盘控件 `ChessBoard`：**
- 使用 Qt `QPainter` 绘制棋盘（10×9 网格线、九宫线、棋子）
- 棋盘尺寸：600×620 像素，格子间距 60px，偏移 50px
- 鼠标交互：点击选择棋子，再次点击目标位置移动
- 红方为玩家，黑方为 AI
- AI 在后台线程运行，使用 `std::thread` + `QMutex`/`QWaitCondition` 同步

**主窗口 `MainWindow`：**
- 750×650 固定窗口，包含棋盘控件和重置按钮
- 使用 QSS 样式表美化

**交互流程：**
1. 玩家点击红方棋子 -> 选中（灰色高亮）
2. 玩家点击目标位置 -> 执行走法
3. 500ms 延迟后唤醒 AI 线程
4. AI 在后台线程中搜索，自动落黑方棋子
5. 检测游戏结果并弹出消息框

### 2.6 其它模块

- `mcts.h/cpp`：MCTS（蒙特卡洛树搜索）框架，仅有类声明，未实现具体逻辑
- `gamedb.h/cpp`：SQLite 棋局数据库，记录每步走法和对局结果
- `qssloader.hpp`：QSS 样式表加载工具
- `appstyle.qss`：UI 样式表
- `chess_games.db`：运行时自动生成的 SQLite 数据库文件

---

## 三、测试工具

> 这一节的早期版本只描述了 `test_utils.h` 与 `test_main.cpp`。现在测试已经扩到
> **11 个可执行文件**（目录树见 §一），其中 6 个注册进 `ctest`。下面保留对最老的
> 那套工具的说明作为背景，**新增测试的作用与验证手法的完整清单见
> `docs/issues_review.md` 的"零之四 4.4 验证手法的沉淀"**。

### test_utils.h
- `SearchStats` 结构体：搜索性能统计
- `Timer` 计时器类：微秒/毫秒精度
- `stepToString()`：走法转可读字符串
- `printChessBoardPlain()`：无颜色文本棋盘输出
- `printEvaluation()`：局面评估分析

### test_main.cpp（最早的一套，→ `test_ab`）
命令行测试程序，支持以下模式：

| 命令行参数 | 功能 |
|-----------|------|
| (无参数) | 运行全部测试套件 |
| `interactive` | 交互模式：手动下棋 |
| `play <depth>` | AI 自对弈（详细输出） |
| `benchmark` | 仅运行走法统计和性能测试 |

**测试套件：**
1. **testMoveCount()** —— 初始局面走法数统计
2. **testSearchPerformance()** —— 不同深度搜索性能基准
3. **testSpecificPositions()** —— 指定局面分析（含残局测试）
4. **testSelfPlay()** —— AI 自对弈统计
5. **testDepthComparison()** —— 搜索深度优势对比

### 后续新增（简要）
- `test_rules`：棋规回归（初始走法数、应将过滤、将杀/困毙、三次重复），92 条断言
- `test_pretrain`：决策前探索**不得改动真棋局**（`sideToMove` / `history` /
  `halfMoveClock` 逐字段比对），23 条断言
- `test_match`：Agent 对弈 arena 的统计正确性（交换先后手、比分归属、到上限判和、
  中止生效），18 条断言；用 `setMaxPliesPerGame(4)` 让结果可预测
- `test_grad`：**SIMD 之后的梯度传播**（有限差分核对解析梯度 + 直接探测 MM 内核
  "累加 vs 覆盖"），6 条断言；只链 `RL_CORE`，不依赖 Qt

## 四、SQLite 棋局记录功能

使用 Qt SQL 模块集成 SQLite 数据库，自动记录每局棋的完整走法和结果。

### 数据库设计

**games 表**（对局信息）：
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| start_time | TEXT | 对局开始时间 |
| end_time | TEXT | 对局结束时间 |
| red_player | TEXT | 红方玩家名 |
| black_player | TEXT | 黑方玩家名(默认AI) |
| result | INTEGER | 0=进行中, 1=红胜, 2=黑胜, 3=平局 |
| total_moves | INTEGER | 总步数 |

**moves 表**（走法记录）：
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| game_id | INTEGER FK | 关联对局 |
| move_number | INTEGER | 步序号 |
| color | INTEGER | 0=红, 1=黑 |
| stone_type | INTEGER | 棋子类型 |
| from_x, from_y | INTEGER | 起点坐标 |
| to_x, to_y | INTEGER | 终点坐标 |
| captured_type | INTEGER | 被吃子类型(-1=无) |
| notation | TEXT | 棋子名称 |

### 集成方式

- **`gamedb.h/cpp`**：`GameDatabase` 单例类，封装 SQLite 所有操作
- **`chessboard.cpp`**：
  - 构造函数中打开数据库并创建新对局
  - `mousePressEvent()` 记录玩家(红方)走法
  - `process()` 线程中记录 AI(黑方)走法
  - `checkGameOver()` 记录对局结果
  - `reset()` 结束旧对局并创建新对局
- **`main.cpp`**：程序退出时关闭数据库连接
- **`chess.pro`**：添加 `QT += sql`

数据库文件 `chess_games.db` 自动生成在程序工作目录。

### 查询方法

可使用任意 SQLite 工具查看，例如：
```bash
sqlite3 chess_games.db
# 查看最近对局
SELECT * FROM games ORDER BY id DESC;
# 查看某局所有走法
SELECT * FROM moves WHERE game_id = 1 ORDER BY move_number;
```

---

## 五、已修复问题

以下已知问题已修复：

| # | 问题 | 修复 |
|---|------|------|
| 1 | **Alpha-Beta 剪枝逻辑错误**：原代码中剪枝条件写反（`minimizeAlpha`中`r <= beta`应为`alpha <= beta`），且检查时机错误 | 修正为：先更新 MIN/MAX 最佳值，再与父节点的约束比较。`minimizeAlpha` 中 `alpha <= beta` 时剪枝；`maximizeBeta` 中 `beta >= alpha` 时剪枝 |
| 2 | **评估函数视角矛盾**：`evaluate()` 原从红方视角计算（正值=红方有利），但 AI（黑方）在`minimizeAlpha`中试图最小化该值，导致搜索方向完全颠倒 | 修改 `evaluate()` 从 AI（黑方）视角：黑子加分，红子减分。正分值 = AI有利 |
| 3 | **将/帅安全性检测缺失**：没有检测"将帅不可见面"| 新增了 `isInCheck(color)`（含飞将检测和对方棋子攻击检测）。**该条现已闭环**：`sample()` 只返回合法走法，`isInCheck` 在 `sample`/`isLegalMoveInternal` 里被真正调用（见 `docs/issues_review.md` A2） |
| 4 | **`isGameOver()` 误判**：`COLOR_NONE` 原本表示"将帅都在"（游戏未结束），但旧代码中没有正确处理 | `minimizeAlpha`/`maximizeBeta` 中，只在将帅被吃时（`COLOR_RED`/`COLOR_BLACK`）才返回极值；无走法时判负 |
| 5 | **兵(卒)走法**：分析发现原代码逻辑正确，无功能性 Bug，已更新注释消除误导 | 明确注释：未过河时禁止左右走；过河后允许水平移动，delta == 1 校验正确 |
| 6 | **叶节点评估粗糙**：`minimizeAlpha` 的叶节点原只返回 `totalReward`（纯材质差），未使用位置评估 | 改为调用 `evaluate()`（材质+位置 PST 综合评估） |
| 7 | **`alive == true` 类型混用**：`int` 与 `bool` 比较触发编译器警告 C4805 | 改为 `if (stones[i]->alive)` |
| 8 | **quiescenceSearch 零宽窗口误剪枝**：`maximizeBeta` depth==0 时调用 `quiescenceSearch(color, alpha, value_infi, 3)`，当从 MIN 节点传入 `alpha = value_infi` 时，MIN 分支的 `standPat <= alpha` 总是成立，所有奇数深度(1,3,5)返回 NULL | `quiescenceSearch` 重写为纯 negamax：`evaluate()` 统一转换到当前走棋方视角；统一使用 `standPat >= beta` 剪枝；递归用 `-quiescenceSearch(color_, -beta, -alpha, depth-1)`；depth==0 入口用 `(-value_infi, +value_infi)` 完全开放窗口 |


## 六、仍存在的问题

> 本节已按最新一轮复查更新。完整清单（含 file:line 证据、失效场景、实测结果、
> 优化手法汇总与修复优先级）见 **`docs/issues_review.md`**；RL 内核的同步、
> SIMD/AVX2 优化与**梯度传播正确性验证**见 **`docs/rl_sync.md`**；
> 各 agent 的设计与"探索预训练"的利弊分析见 **`docs/agents_design.md`**。
>
> **当前实测（与本文早先的记录已不同，早先那段数字已过期）**：
> `ctest` **6/6 通过** —— `test_ab` ~4.1 s（曾段错误）、`test_mcts` ~228 s、
> `test_rules` ~0.05 s、`test_pretrain` ~4.0 s、`test_match` ~1.8 s、`test_grad` ~19 s
> （耗时随机器负载浮动，这里给的是量级）。
> 四个训练基准也不再超时：`test_pg` ~18 s、`test_dqn` ~353 s、`test_ppomcts` ~78 s、
> `test_dqnmcts` ~390 s（曾全部 >900 s 超时），根因 B19 已修（`Tensor::MM` 0.11 →
> 27.8 GFLOP/s）。

下面 1–3 条是本文档早先记录的"仍存在的问题"，**均已修复**，保留在此仅为对照
（结论已更新）：

1. ~~**AI 缺乏应将/将军意识**~~ —— **已修复**：`sample()` 只返回合法走法
   （自杀 / 不应将 / 照面全部过滤），`Chess::isAttacked/isLegalMove/hasLegalMoves`
   落地，GUI 玩家侧也走 `isLegalMove()`，并有 `getResult()` 做将杀/困毙/和棋判定。
2. ~~**游戏无法自然结束**~~ —— **已修复**：`moveForward` 维护 `history`（含走棋方
   哈希）与 `halfMoveClock`，`isDraw()` 实现三次重复 + 120 半回合；搜索里重复局面
   返回 0 分；`matchAgents` 到达手数上限判和。
3. ~~**MCTS 未实现**~~ —— **该条早已过期**：`src/mcts.cpp` 已完整实现
   select / expand / simulate / backpropagate + UCB1 并接入 GUI。

### 当前真正仍未做的（按优先级，详见 `issues_review.md` §五）

4. **`Tensor::MM::ikjk/kijk` 的 SIMD 内核是赋值、标量是累加**（语义不一致）。
   当前调用点 kdim=1 走标量，所以梯度是对的（`test_grad` 实测确认）；但只要改成
   多样本批量（kdim ≥ 8）就会命中 SIMD 内核，**静默只保留最后一次的贡献**。
   修法一行：内核改 `z[...] += dot(...)`。
5. **反向 GEMV 没有 SIMD 内核**：`ei = wᵀ·e` 实测 0.991 ns/MAC，而前向 `o = w·x`
   是 0.102 —— **9.7×**。SIMD 只加速了前向那一半，训练步现在卡在反向。
   修法：补与 `gemv_ikkj` 对称的 `gemv_kikj`。
6. **训练数据管线（理论层面的取舍）**：见 `agents_design.md` §10。要点是
   DQN 系"每步一次更新"破坏回放缓冲的 i.i.d. 假设（建议改成只写回放、攒批更新），
   PPO 系在稀疏终局奖励下需要显式自举，否则优势几乎全 0、梯度等于噪声。
7. **SQLite 跨线程使用 + 写库接口未接线**（`recordMove/startGame/endGame` 全仓零调用），
   需要设计决定。
8. `DQN` 的 Q 头是 `Layer<Sigmoid>`（值域 `(0,1)`）而奖励含负值 → 结构上无法表示负 Q。
9. QSS 主题仍未接线（`QssLoader::load()` 零调用）。
