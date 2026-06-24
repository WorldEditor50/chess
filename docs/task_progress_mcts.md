# MCTS Algorithm Implementation - 设计文档

## 概述

本项目在已有的Alpha-Beta剪枝搜索基础上，实现了基于蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）的AI走法生成器。MCTS是一种不依赖领域知识的启发式搜索算法，通过随机模拟（Rollout）来评估局面优劣，适用于搜索空间极大的博弈问题。

## MCTS 四阶段流程

每次迭代分为四个阶段：

### 1. Selection (选择)
从根节点开始，使用 **UCB1 (Upper Confidence Bound 1)** 公式选择最优子节点，直到到达一个尚未完全展开的节点。

```
UCB1 = W/N + C * sqrt(ln(N_parent) / N)
```
- `W`: 节点累积奖励
- `N`: 节点访问次数
- `N_parent`: 父节点访问次数
- `C`: 探索常数 (本实现使用 sqrt(2) ≈ 1.414)

**实现位置**: `mcts.cpp` 中的 `findBestMove()` 函数，while循环部分。

### 2. Expansion (扩展)
当选中节点还有未尝试的走法时，随机选取一个，执行该走法，创建新的子节点，并生成该子节点所有可能的合法走法列表供后续展开。

**实现位置**: `mcts.cpp` 中 `findBestMove()` 函数的 `if (!nodes[nodeID].untriedMoves.empty())` 部分。

### 3. Simulation / Rollout (模拟)
从当前叶子节点对应的局面开始，双方随机选择合法走法，直到游戏结束（将杀或无子可走）。

**奖励函数**:
- `+1.0`: 当前颜色获胜
- `-1.0`: 当前颜色失败
- 模拟过程中产生的所有走法在返回前都会撤销，恢复局面。

**实现位置**: `mcts.cpp` 中的 `simulateRandomPlay()` 函数。

### 4. Backpropagation (反向传播)
沿Selection和Expansion阶段记录的路径回溯，更新路径上每个节点的访问次数(`visitCount++`)和累积奖励(`totalReward += reward`)。每回溯一层，奖励符号翻转一次（因为轮到对方视角）。

**实现位置**: `mcts.cpp` 中 `findBestMove()` 函数的for循环backpropagation部分。

### 最优走法选择
迭代结束后，选择根节点下访问次数最高的子节点对应的走法作为最终结果。

## 类设计

### MCTSNode
| 成员 | 类型 | 说明 |
|------|------|------|
| `visitCount` | int | 节点访问次数 N(s) |
| `totalReward` | double | 累积奖励 W(s) |
| `parentID` | int | 父节点索引 (-1 表示根节点) |
| `step` | Step | 从父节点到达该节点的走法 |
| `childIDs` | vector\<int\> | 子节点索引列表 |
| `untriedMoves` | vector\<Step\> | 尚未展开的走法列表 |
| `currentColor` | int | 该节点对应的走棋方 |
| `getUCB1()` | double | 计算UCB1分数 |

### MCTS
| 成员/方法 | 说明 |
|-----------|------|
| `chess` | 引用棋盘对象，用于执行/撤销走法和生成走法 |
| `nodes` | 树的所有节点池 |
| `C` | UCB1探索常数 |
| `findBestMove(color, iterations)` | 主入口：运行MCTS搜索，返回最优走法 |
| `clearTree()` | 清空搜索树 |

## 集成方式

### GUI集成 (`src/chessboard.cpp`)
- 在 `process()` 中创建 `MCTS mctsAI(chess, 1.414)` 实例
- 调用 `mctsAI.findBestMove(Stone::COLOR_BLACK, 5000)` 代替原有的 `chess.alphaBetaPruning()`
- 每次AI走法执行5000次MCTS迭代

### 测试程序 (`test/test_mcts_main.cpp`)
独立于GUI的控制台测试程序，提供以下测试功能：
- **性能基准测试**: 100/500/1000/2000次迭代的搜索耗时
- **指定局面分析**: 初始局面MCTS搜索分析
- **自对弈测试**: MCTS AI双方对弈
- **MCTS vs Alpha-Beta**: 两种算法对战（MCTS黑方1000迭代 vs Alpha-Beta红方深度3）

## 性能特征

- **初始局面**: 2000次迭代约2秒
- **自对弈 (500次迭代)**: 每局约49秒（平均150步）
- **与Alpha-Beta对比**: 1000次MCTS迭代在初始局面下的棋力略弱于深度3的Alpha-Beta
  - 原因: 纯随机Rollout策略精度较低
  - 改进方向: 可引入启发式Rollout（如使用轻量级评估函数引导随机走法），或增大迭代次数

## 后续优化方向

1. **启发式Rollout**: 在Simulation阶段使用简单的评估函数 + epsilon-greedy策略选择走法
2. **Rave (Rapid Action Value Estimation)**: 加速相同走法在不同路径下的评估
3. **并行MCTS**: 使用多线程并行执行多次迭代
4. **渐进宽化 (Progressive Widening)**: 控制大分支因子下的展开速度
5. **迭代轮次自适应**: 根据剩余时间动态调整迭代次数
