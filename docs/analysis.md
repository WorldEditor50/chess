# 中国象棋程序 - 全面功能分析

## 一、项目概览

这是一个基于 **Qt 6** 框架开发的中国象棋桌面游戏，集成了基于 **Alpha-Beta 剪枝** 的AI引擎。项目使用 C++17 编写，包含 GUI 界面（Qt Widgets）和 AI 算法两部分。

### 文件结构

```
chess/
├── main.cpp              # 程序入口
├── mainwindow.h/cpp/ui   # 主窗口 (Qt)
├── chessboard.h/cpp      # 棋盘控件 (Qt 绘制 + 交互)
├── chess.h/cpp           # 棋局逻辑 + AI搜索算法
├── stone.h/cpp           # 棋子基类与派生类 + 走法/对象池
├── pos.h/cpp             # 坐标类
├── mcts.h/cpp            # MCTS 骨架 (未实现)
├── qssloader.hpp         # QSS 样式加载工具
├── appstyle.qss          # 样式文件
├── res.qrc               # 资源文件
├── test_utils.h          # 测试工具类 (新增)
├── test_main.cpp         # 沙盒测试程序 (新增)
├── test_build.pro        # 测试项目配置 (新增)
└── chess.pro             # Qt 项目配置
```

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

## 三、新增测试工具

### test_utils.h
- `SearchStats` 结构体：搜索性能统计
- `Timer` 计时器类：微秒/毫秒精度
- `stepToString()`：走法转可读字符串
- `printChessBoardPlain()`：无颜色文本棋盘输出
- `printEvaluation()`：局面评估分析

### test_main.cpp
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

编译命令：
```bash
# 需先运行 vcvars64.bat 配置 MSVC 环境
cl /std:c++17 /utf-8 /O2 /EHsc -Fe:test_chess.exe chess.cpp pos.cpp stone.cpp test_main.cpp
```

---

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
| 3 | **将/帅安全性检测缺失**：没有检测"将帅不可见面"| 新增 `isInCheck(color)`，包含飞将检测和对方棋子攻击检测 |
| 4 | **`isGameOver()` 误判**：`COLOR_NONE` 原本表示"将帅都在"（游戏未结束），但旧代码中没有正确处理 | `minimizeAlpha`/`maximizeBeta` 中，只在将帅被吃时（`COLOR_RED`/`COLOR_BLACK`）才返回极值；无走法时判负 |
| 5 | **兵(卒)走法**：分析发现原代码逻辑正确，无功能性 Bug，已更新注释消除误导 | 明确注释：未过河时禁止左右走；过河后允许水平移动，delta == 1 校验正确 |
| 6 | **叶节点评估粗糙**：`minimizeAlpha` 的叶节点原只返回 `totalReward`（纯材质差），未使用位置评估 | 改为调用 `evaluate()`（材质+位置 PST 综合评估） |
| 7 | **`alive == true` 类型混用**：`int` 与 `bool` 比较触发编译器警告 C4805 | 改为 `if (stones[i]->alive)` |
| 8 | **quiescenceSearch 零宽窗口误剪枝**：`maximizeBeta` depth==0 时调用 `quiescenceSearch(color, alpha, value_infi, 3)`，当从 MIN 节点传入 `alpha = value_infi` 时，MIN 分支的 `standPat <= alpha` 总是成立，所有奇数深度(1,3,5)返回 NULL | `quiescenceSearch` 重写为纯 negamax：`evaluate()` 统一转换到当前走棋方视角；统一使用 `standPat >= beta` 剪枝；递归用 `-quiescenceSearch(color_, -beta, -alpha, depth-1)`；depth==0 入口用 `(-value_infi, +value_infi)` 完全开放窗口 |


## 六、仍存在的问题

1. **AI 缺乏应将/将军意识**：`sample()` 生成的走法未过滤"被将军时不移出"和"走后不可暴露将帅"的非法走法。`isInCheck()` 已实现但未集成到搜索中
2. **游戏无法自然结束**：AI 缺乏长将检测（`isRepetition()` 已实现但未集成），自对弈往往达到最大步数
3. **MCTS 未实现**：`mcts.h/cpp` 仅定义了框架结构，没有搜索实现
