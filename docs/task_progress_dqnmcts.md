# DQN + MCTS Agent - 任务进度

## 概述

DQN + MCTS 智能体将 Deep Q-Network (DQN) 与蒙特卡洛树搜索 (MCTS) 相结合。

### 设计思想

- **DQN**: 提供 Q(s,a) 值函数，评估每个状态下各动作的价值
- **MCTS**: 使用 UCB1 树搜索，DQN 替代传统随机模拟
- **训练**: 使用标准 DQN 经验回放机制

### 与 PPOMCTSAgent 的区别

| 特性 | PPOMCTSAgent | DQNMCTSAgent |
|------|-------------|-------------|
| 神经网络 | PPO Policy + Value heads | DQN Q-value network |
| MCTS 策略 | PUCT (先验策略) | UCB1 (无先验) |
| 叶节点评估 | Value head | max_a Q(s,a) |
| 训练算法 | PPO clip + GAEs | DQN experience replay |
| 网络深度 | MOE<8,4> + TanhNorm(256) + TanhNorm(128) + Output | MOE<8,4> + TanhNorm(256) + TanhNorm(128) + Output |

### 算法流程

1. **Selection**: UCB1 从根节点开始遍历树
2. **Expansion**: 选择未扩展的子节点
3. **Evaluation**: 调用 DQN 获取 max Q(s,a) 作为叶节点价值
4. **Backpropagation**: 反向传播更新节点统计

### 文件清单

- `src/dqnmcts_agent.h` - DQNMCTSAgent 类声明
- `src/dqnmcts_agent.cpp` - DQNMCTSAgent 实现
- `test/test_dqnmcts_main.cpp` - 测试程序
- `docs/task_progress_dqnmcts.md` - 本文件

### 测试方法

- `selectMove(color, iterations, training)` - 单步选择
- `trainVsRandom(episodes, iterations, maxMoves)` - 对随机弈训练
- `trainSelfPlay(episodes, iterations, maxMoves)` - 自对弈训练
- 推理测试: 200 次 MCTS 迭代进行单步决策
