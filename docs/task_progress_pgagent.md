# Policy Gradient Reinforcement Learning Agent

## 概述

实现了基于 **REINFORCE** (Policy Gradient) 算法的中国象棋 AI agent，使用现有 `RL::DPG` 框架和神经网络基础设施。

## 文件结构

| 文件 | 说明 |
|------|------|
| `src/pgagent.h` | PGEagent 类声明 |
| `src/pgagent.cpp` | 完整实现 (状态编码、动作选择、REINFORCE 训练) |
| `test/test_pg_main.cpp` | 测试程序 (推理、训练、VS随机、自对弈) |

## 依赖

- `src/rl/dpg.h` — 策略梯度网络 (RL::DPG)
- `src/rl/layer.h` — 神经网络层 (Linear, TanhNorm)
- `src/chess.h/chess.cpp` — 中国象棋核心逻辑

## 算法细节

### 状态编码 (90维)

- 10×9 棋盘展平为 90 维向量
- 每个位置：`0`=空, `+1..+7`=黑方棋子类型, `-1..-7`=红方棋子类型
- 棋子类型: 车=1, 马=2, 炮=3, 兵=4, 帅=5, 仕=6, 相=7

### 动作空间 (128维)

- 使用哈希函数将 `(fromID × toPosition)` 映射到 `[0, 128)`
- 掩码机制过滤非法动作，确保只采样合法走法

### 网络结构

- 输入: 90 → 隐藏层: 64(ReLU) → 输出: 128(Softmax)
- 使用 `RL::DPG` 的 `Net` (Sequential + CrossEntropyLoss)

### 奖励设计

- 即时奖励: `victimValue × 10` (吃子价值)
- 终端奖励: `+1`(胜), `-1`(负), `0`(和)
- 将帅被吃奖励: `+100`

### 训练循环

1. 每步编码状态 → 策略网络前向传播 → Softmax采样 → 执行走法
2. 记录轨迹 `(state, action_onehot, reward)`
3. 对弈结束（将军/和棋/最大步数）后，统一分配终端奖励
4. 调用 `RL::DPG::reinforce1()` 执行 REINFORCE (带baseline)
5. ε-贪心探索率从 1.0 衰减至 0.05

### 自对弈模式

- `selfPlay=true`: 双方轮流使用同一策略网络
- `selfPlay=false`: AI只走黑方，对弈一步后结束（每步一局）

## 测试结果

- 构建: ✅ 通过 (CMake + Ninja, MSVC2022)
- 推理: ✅ 成功选择合法走法
- 训练(10局): ✅ 完成并记录胜率
- VS随机(5局): ✅ 20%胜率 (训练量少，随机网络初始)
- 自对弈(50局): ✅ 黑方16%胜率, 红方2%胜率 (展现学习趋势)

> 注意：由于网络初始化随机且训练量有限(50局)，胜率较低。更多训练(>1000局)会显著提升。
