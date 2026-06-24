# Deep Q-Network (DQN) Chess Agent

## 概述

实现了基于 **Deep Q-Network** (DQN) 算法的中国象棋 AI agent，使用现有 `RL::DQN` 框架。

## 文件结构

| 文件 | 说明 |
|------|------|
| `src/dqnagent.h` | DQNAgent 类声明 |
| `src/dqnagent.cpp` | 完整实现 (ε-greedy, experience replay, target network) |
| `test/test_dqn_main.cpp` | 测试程序 (推理、VS随机训练+测试、自对弈) |

## 依赖

- `src/rl/dqn.h` / `dqn.cpp` — DQN 算法 (RL::DQN)
- `src/rl/layer.h` — 神经网络层 (MOE, TanhNorm)
- `src/chess.h/chess.cpp` — 中国象棋核心逻辑

## 算法细节

### 状态编码 (90维)
- 10×9 棋盘展平，正=黑方，负=红方，值按棋子类型排列

### 动作空间 (128维)
- 哈希映射：`(fromID * 37 + toPos.x * 13 + toPos.y * 7) % 128`
- 非法动作掩码为 `-inf` (Q-value)

### 网络结构
- MOE<8,4> → TanhNorm<Sigmoid>(stateDim, hiddenDim) → Layer<Sigmoid>(hiddenDim, actionDim)
- QMainNet + QTargetNet (软更新, τ=0.01)

### 训练流程
1. **Experience replay**: 存储 (s, a, r, s', done) 到 4096 大小缓冲区
2. **Experience replay learning**:
   - Q(s,a) = r + γ · max_a' Q_target(s', a') (非终止)
   - Q(s,a) = r (终止)
3. **Target network**: 每 256 步用 RMSProp 软更新
4. **ε-greedy 探索**: ε 从 1.0 以 0.9999 衰减至 0.1

### 训练模式
- `trainVsRandom()`: AI(黑) vs 随机对手(红)
- `trainSelfPlay()`: 双方使用同一 DQN 策略

## 测试结果

- 推理: ✅ 成功
- VS随机训练(10局): ✅ 70%胜率, target net 更新
- 无探索测试(5局): 1胜3负1平 (训练量少)
- 自对弈(20局): ✅ 黑方25%, 红方30% (双方学习均衡)

> 注意：DQN 需要数千至上万局训练才能达到强棋力。当前测试仅验证算法正确性。
