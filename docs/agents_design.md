# 中国象棋 AI Agent 设计文档

## 概述

本项目实现了五种 AI 象棋 Agent，均继承自统一的 `AgentBase` 接口。

```
AgentBase (抽象基类)
  ├── ABAgent        — Alpha-Beta 剪枝
  ├── MCTS           — 蒙特卡洛树搜索（随机模拟）
  ├── PGEagent       — 策略梯度 REINFORCE
  ├── DQNAgent       — 深度 Q 网络
  └── PPOMCTSAgent   — PPO + MCTS (AlphaZero 风格)
```

### AgentBase 接口 (`src/aiagent.h`)

```cpp
class AgentBase {
public:
    virtual Step getBestMove(int color) = 0;  // 核心：获取最佳走法
    virtual std::string getName() const = 0;  // 名称
    virtual void resetState() {}              // 可选重置

    // 走子前的"先探索环境、预训练一次"（仿 snakeAI 的 observe → train → act）
    // 默认实现是 no-op 并返回 false，无参数的 agent（ABAgent）直接跳过。
    virtual bool exploreAndTrain(int color, int rolloutSteps) {
        (void)color; (void)rolloutSteps;
        return false;
    }
    virtual std::string getExploreInfo() const { return m_exploreInfo; }
protected:
    std::string m_exploreInfo;
};
```

`exploreAndTrain` 由 `ChessBoard::preTrainThenDecide()` 在每次决策前调用，
**必须在返回时让棋盘逐字节复原**（含 `sideToMove`）。共用实现见
`src/agentrollout.hpp` 的 `rolloutFromCurrent()`，契约与实测见
`docs/issues_review.md` 的"零之二点七"。

### 公共约定

| 约定 | 说明 |
|------|------|
| 棋盘编码 | 90 维 (10×9 行主序)，红方负值，黑方正值 |
| 动作编码 | 128 维 one‑hot，用确定性哈希映射合法走法 |
| 棋子类型值 | CHE=1.0, MA=2.0, PAO=3.0, BING=4.0, JIANG=5.0, SHI=6.0, XIANG=7.0 |
| 哈希公式 | `(fromID * 37 + toX * 13 + toY * 7) % 128` |

---

## 动作编码原理详解

### 背景：为什么需要动作编码？

中国象棋的走法用 `Step` 结构体描述（见 `stone.h`）：

```cpp
struct Step {
    int id;        // 移动棋子的 ID (0~31)
    int nextId;    // 被吃掉的棋子 ID (ID_NONE=32 表示未吃子)
    Pos pos;       // 起点坐标 (x行 0~9, y列 0~8)
    Pos nextPos;   // 终点坐标
    double reward; // 奖励值
};
```

**问题：** 神经网络需要固定维度的输入/输出，但象棋每步的合法走法数量不固定——开局约 40 种，残局可能只有几种。所以需要一个确定性哈希函数将每个 `Step` 映射到固定大小的 one-hot 向量空间（128 维）。

### 哈希编码公式

所有使用神经网络的 Agent（PGEagent、DQNAgent、PPOMCTSAgent）使用完全相同的哈希函数：

```cpp
int stepToActionIdx(const Step &s) {
    unsigned long long h = (unsigned long long)s.id * 37ULL
                         + (unsigned long long)s.nextPos.x * 13ULL
                         + (unsigned long long)s.nextPos.y * 7ULL;
    return (int)(h % ACTION_DIM);  // ACTION_DIM = 128
}
```

代码位于：
- `src/pgagent.cpp` 第 82-89 行
- `src/dqnagent.cpp` 第 98-104 行  
- `src/ppomcts_agent.cpp` 第 94-100 行

#### 输入特征选择

| 特征 | 值域 | 作用 |
|------|------|------|
| `s.id` | 0~31 | 移动棋子的唯一 ID（标识"哪个棋子在走"） |
| `s.nextPos.x` | 0~9 | 目标位置的行坐标 |
| `s.nextPos.y` | 0~8 | 目标位置的列坐标 |

**为什么排除 `s.pos`（起点）和 `s.nextId`（被吃子）？**
- 起点可由 `id` 间接推导——每个棋子在给定时刻只有唯一位置
- 被吃子由目标位置间接确定——目标位置上有子时自然知道吃了什么

#### 质数系数的选择

- 37、13、7 均为质数，这是经典哈希技巧
- **目的：** 减少哈希冲突，使走法在 128 维空间中分布更均匀
- **反例：** 若用 `id*1 + x*1 + y*1`，相近走法会聚集到相邻索引，导致神经网络难以区分

#### 哈希冲突的影响

128 维空间远小于所有可能走法的总数（32 棋子 × 90 目标格 ≈ 2880 种可能），冲突必然存在。

- **同一局面内：** 两个不同合法走法映射到同一索引 → **无害**。它们共享同一个概率/Q 值/访问计数，相当于「合并统计」，不产生错误决策
- **跨局面：** 不同局面的走法映射到同一索引 → **无害**。Action Mask 确保只有当前局面的合法走法被激活

### 合法动作掩码 (Action Mask)

由于不是所有 128 个动作都合法，Agent 使用掩码机制屏蔽非法输出：

```
encodeState() → 90-dim board tensor
       │
       ▼
Neural Network → 128-dim raw logits
       │
       ▼
    [Action Mask]  ← 标记合法走法 (0/1 向量)
       │
       ▼
Softmax / Argmax (仅在合法索引中)
       │
       ▼
选择并执行走法
```

```cpp
void getLegalActions(int color,
                     std::vector<Step*> &steps,
                     std::vector<int> &actionIndices,
                     RL::Tensor &actionMask) {
    actionMask.zero();
    chess.sample(color, steps);          // 生成所有合法走法
    for (Step *s : steps) {
        int aidx = stepToActionIdx(*s);
        actionIndices.push_back(aidx);
        actionMask[aidx] = 1.0f;          // 标记为合法
    }
}
```

### 神经网络输出与动作编码的关系

```
棋盘状态 (10×9)
     │
     ▼
encodeState() → [0.5, -0.5, 0.3, ..., 0.0]  (90-dim)
     │
     ▼
神经网络
     │
     ▼
输出层 128 个神经元
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │... │ 45 │ 46 │ 47 │... │ 125│ 126│ 127│
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
     │    │                                              │
     │    └── 索引1 对应走法: 车(id=0)走到(0,1)           │
     │                                                    │
     └── 索引0 对应走法: 马(id=1)走到(2,0)               │
                                                          │
                                          索引47 对应走法: 炮(id=20)走到(5,2)
```

### 动作解码：从索引到 Step

**核心事实：哈希函数 `stepToActionIdx` 是不可逆的。** 没有 `actionIdxToStep(index) → Step` 这样的逆向函数——128 维空间远小于走法总数，同一索引可能对应多个不同的走法（哈希冲突）。

解码依靠的是**预计算的并行索引映射**——在 `getLegalActions()` 中同时维护 `steps` 和 `actionIndices` 两个平行列表，保证下标对齐：

```cpp
void getLegalActions(int color,
                     std::vector<Step*> &steps,       // output: 合法走法列表
                     std::vector<int> &actionIndices, // output: 每个走法对应的动作索引
                     RL::Tensor &actionMask) {        // output: 合法掩码
    chess.sample(color, steps);
    for (Step *s : steps) {
        int aidx = stepToActionIdx(*s);
        actionIndices.push_back(aidx);  // steps[i] ↔ actionIndices[i] (下标对齐)
        actionMask[aidx] = 1.0f;
    }
}
```

选择动作后，解码通过**线性扫描**匹配索引完成：

```
steps:        [ Step(车(0)→(0,1)), Step(马(1)→(2,0)), Step(炮(20)→(5,2)), ... ]
                    ↕                    ↕                    ↕
actionIndices: [ 47,                  13,                  81,                ... ]

神经网络输出: [0.0, 0.0, ..., 0.93, ..., 0.0, ...]
                                    ↑
                            选中最⼤索引 = 47
                                    ↕
                    线性扫描 actionIndices 找到第一个 47
                                    ↕
                    steps[0] = Step(车(0)→(0,1)) ← 实际走法
```

PGEagent 和 DQNAgent 的 `selectMove()` 中解码代码完全一致（来自源码 `dqnagent.cpp:173-183`，`pgagent.cpp:182-194`）：

```cpp
/* 从 actionIndices 中查找 selectedAction */
for (std::size_t i = 0; i < actionIndices.size(); i++) {
    if (actionIndices[i] == selectedAction) {   // 线性扫描匹配哈希值
        Step result = *steps[i];                // 通过下标取回 Step 对象
        Steps::instance().put(steps);            // 释放 Step 对象池
        return result;                           // 返回真实走法
    }
}

/* 保底：如果没找到（理论上不会发生），取第一个合法走法 */
Step result = *steps[0];
Steps::instance().put(steps);
return result;
```

**为什么用线性扫描而不是哈希表？**
1. 当前局面的合法走法通常只有 20~50 种，线性扫描代价极低
2. 省去哈希表维护开销和内存占用
3. 代码简单直接，易于理解

#### PGEagent 的完整解码流程

PGEagent 的 `selectMove()`（`pgagent.cpp:106-195`）的完整编码→解码流程：

```
encodeState() → state(90-dim)
    │
    ▼
getLegalActions() → steps[], actionIndices[], actionMask[]
    │
    ▼
policy network → policyOut(128-dim)
    │
    ▼
mask illegal actions (policyOut[i] = 0 where actionMask[i]==0)
    │
    ▼
renormalize softmax over legal actions only
    │
    ├── training:  categorical sampling from masked distribution
    │   或
    └── inference: argmax over masked logits
    │
    ▼
selectedAction = some index in [0, 127]
    │
    ▼
for (i = 0; i < actionIndices.size(); i++)
    if (actionIndices[i] == selectedAction)
        return steps[i];        // ← 解码完成
```

#### PPOMCTSAgent 的解码

PPOMCTSAgent 中的 MCTS 节点直接存储 `Step` 对象，无需通过索引反查。在展开节点时，`Step chosenStep = nodes[nodeID].untriedSteps[moveIdx]` 原地复制 `Step`；在选择最佳走法时直接返回 `nodes[bestChildID].step`。动作索引 `chosenAction` 仅用于记录先验概率 `prior = childPolicy[chosenAction]` 和回放 `oneHotAction[chosenAction] = 1.0f`。

### PGEagent/DQNAgent 中的使用细节

PGEagent 选择动作：

```cpp
// 1. 将非法动作的 logits 置零
for (int i = 0; i < ACTION_DIM; i++) {
    if (actionMask[i] < 0.5f) policyOut[i] = 0.0f;
}
// 2. 重归一化（只在合法动作上做 softmax）
float sum = sum(policyOut);
for (int i = 0; i < ACTION_DIM; i++) policyOut[i] /= sum;

// 3. 训练时采样，推理时 argmax（仅在合法索引上）
int selectedAction;
if (training) {
    selectedAction = RL::Random::categorical(policyOut);  // 概率采样
} else {
    selectedAction = argmax_masked(policyOut, actionMask); // 取最大概率
}

// 4. 解码：线性扫描 actionIndices → 匹配 → 返回 Step
for (std::size_t i = 0; i < actionIndices.size(); i++) {
    if (actionIndices[i] == selectedAction) {
        Step result = *steps[i];
        Steps::instance().put(steps);
        return result;
    }
}
```

DQNAgent 选择动作：

```cpp
// 1. 从 DQN 获取 Q 值
Tensor qValues = dqn.forward(state);
// 2. 非法动作 Q 值置 -∞
for (int i = 0; i < ACTION_DIM; i++) {
    if (actionMask[i] < 0.5f) qValues[i] = -1e9f;
}

// 3. ε-greedy 选择
int selectedAction;
if (training && rand() < epsilon) {
    int idx = std::rand() % steps.size();
    selectedAction = actionIndices[idx];   // 随机合法动作
} else {
    selectedAction = qValues.argmax();     // 最大 Q 值
}

// 4. 解码（同 PGEagent）
for (std::size_t i = 0; i < actionIndices.size(); i++) {
    if (actionIndices[i] == selectedAction) {
        Step result = *steps[i];
        Steps::instance().put(steps);
        return result;
    }
}
```

PPOMCTSAgent 中的使用：

PPO 网络输出作为 MCTS 的先验概率 P(s,a)，通过 PUCT 公式引导搜索：

```cpp
// PPO actor 输出策略先验
Tensor policyPrior = ppo.action(state);  // 128-dim softmax

// MCTS 展开新节点时记录该动作的先验
float prior = policyPrior[chosenAction];
newNode.prior = prior;

// PUCT 选择：Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
double puct = child.getQ() + c_puct * child.prior
              * sqrt(parentVisits) / (1 + child.visitCount);
```

### 非神经网络 Agent 的动作表示

- **ABAgent** 和 **MCTS** 不使用神经网络，因此不需要动作编码
- 它们直接在 `std::vector<Step*>` 上操作，节点之间直接用 `Step` 对象作为边的标签
- ABAgent 的走法排序使用 MVV-LVA 评分，完全基于棋子类型值，不涉及编码

---

## 1. ABAgent — Alpha-Beta 剪枝

**文件**: `abagent.h / abagent.cpp`

### 设计思路

传统极小极大搜索 + Alpha-Beta 剪枝，属于**确定性搜索**方法。没有神经网络，完全依靠局面评估函数。

### 核心流程

```
getBestMove(color)
  └── maximizeBeta(color, depth=4, -∞, +∞)  // 根节点为 MAX 层
        └── 遍历所有合法走法：
              ├── 执行走法 → 递归 minimizeAlpha(...)
              │     └── 遍历所有回应：
              │           └── 递归 maximizeBeta(...)
              └── Alpha-Beta 剪枝

若到达叶节点或深度=0 → evaluate() + quiescenceSearch()
```

### 关键技术

| 技术 | 说明 |
|------|------|
| **MVV-LVA 走法排序** | 先搜索最有希望的走法（受害者价值高/攻击者价值低的优先），提高剪枝效率 |
| **Quiescence Search** | 在叶节点继续搜索所有吃子走法，缓解水平效应（Horizon Effect） |
| **Piece-Square Tables (PST)** | 每个棋子类型有 10×9 位置价值表，红方直接查表，黑方垂直镜像 |
| **局面评估** | `Chess::evaluate()` — 双方材料值 + 位置值之和 |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 搜索确定性，无随机性 | 固定深度（4层），深层搜索极慢 |
| 不需要训练数据 | 评估函数依赖手工特征 |
| 在有限深度内可找到最优走法 | 对复杂局面可能视野不足 |

---

## 2. MCTS — 蒙特卡洛树搜索（随机模拟）

**文件**: `mcts.h / mcts.cpp`

### 设计思路

基于 UCB1 的纯 MCTS，使用**随机模拟（Rollout）**评估叶节点。不依赖任何神经网络，完全靠树搜索和随机采样。

### 核心流程

```
findBestMove(color, iterations)
  ├── 创建根节点，生成所有合法走法作为 untriedMoves
  │
  └── 主循环 iterations 次：
        ├── Phase 1: SELECTION — UCB1 选择
        │     └── UCB1 = W/N + C × √(ln(N_parent) / N)
        │     └── 未访问子节点返回 ∞（强制探索）
        │
        ├── Phase 2: EXPANSION — 随机选一个 untried 走法
        │     └── 创建子节点，生成其所有合法走法
        │
        ├── Phase 3: SIMULATION — 随机模拟至终局
        │     └── 双方随机走子直到将杀或无走法
        │     └── 返回 +1(胜) / -1(负) / 0(平)
        │
        ├── Phase 4: BACKPROPAGATION — 反向传播
        │     └── 路径上每个节点: visitCount++, totalReward += reward
        │     └── reward 在每层反转（交替玩家视角）
        │
        └── 每次迭代后撤销所有走法，恢复棋盘

返回访问次数最多的子节点对应的走法
```

### UCB1 公式

```cpp
double getUCB1(double totalParentVisits, double C) const {
    if (visitCount == 0) return ∞;
    double exploitation = totalReward / visitCount;           // Q(s,a)
    double exploration = C × sqrt(ln(N_parent) / N_child);    // U(s,a)
    return exploitation + exploration;
}
```

### 模拟 (Rollout) 细节

- 从当前叶节点开始，双方交替**均匀随机**选择合法走法
- 以将杀或无走法为准终止，返回模拟结果
- 所有模拟走法在退出前撤销，棋盘状态不变

### 参数

| 参数 | 值 | 说明 |
|------|----|------|
| C (explorationConstant) | 1.414 (√2) | UCB1 探索权重 |
| iterations (默认) | 800 | 主循环迭代次数 |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 不需要任何先验知识或训练 | 纯随机模拟精度低，需要大量迭代 |
| 可无限并行化（各次迭代独立） | 对深层战术组合不够敏锐 |
| 适合局面复杂、评估函数难设计的场景 | 比 Alpha-Beta 慢（同时间内） |

---

## 3. PGEagent — Policy Gradient REINFORCE

**文件**: `pgagent.h / pgagent.cpp`

### 设计思路

基于 REINFORCE 算法的策略梯度方法。使用神经网络 `RL::DPG` 直接学习策略 π(a|s)，通过**采样动作 → 收集轨迹 → 更新网络**的方式进行训练。

### 网络架构

```
RL::DPG (stochastic policy network)
  Input:  90-dim board state
  Hidden: Tanh(64)
  Output: Linear(128) → Softmax → 动作概率分布
```

### 训练流程

```
train(episodes, selfPlay=true)
  └── 每局：
        ├── 重置棋盘
        ├── 循环直到终局或 maxMoves：
        │     ├── encodeState() → 90-dim tensor
        │     ├── 采样动作: policy.forward(state) → 概率分布 → 随机采样
        │     ├── 执行走法，记录 (state, one-hot-action, reward)
        │     └── 切换玩家
        │
        └── 终局后：REINFORCE 更新
              └── dpg.reinforce1(trajectory, finalOutcome)
                    ├── 计算折扣回报 G_t = Σ γ^(k-t) × r_k
                    └── 策略梯度: ∇J(θ) ≈ Σ G_t × ∇log π(a_t|s_t)
```

### 选择策略

| 模式 | 行为 |
|------|------|
| **训练 (training=true)** | 从 softmax 概率分布采样（探索） |
| **推理 (training=false)** | argmax 选择最大概率动作（利用） |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 策略网络可学习复杂模式 | REINFORCE 高方差，收敛慢 |
| 可在线持续学习 | 需要大量对局才能学到有效策略 |
| 学到的是概率分布而非确定策略 | 容易陷入局部最优 |

---

## 4. DQNAgent — Deep Q-Network

**文件**: `dqnagent.h / dqnagent.cpp`

### 设计思路

基于 DQN 的值函数方法。使用神经网络 `RL::DQN` 学习 Q(s,a)，通过 **ε-greedy 探索 + 经验回放 + 目标网络** 实现稳定训练。

### 网络架构

```
RL::DQN (value network)
  ├── Online Network: state(90) → Tanh(64) → Linear(128) → Q(s,·)
  └── Target Network: 结构相同，参数软更新
```

### 训练流程

```
trainSelfPlay / trainVsRandom(episodes)
  └── 每步：
        ├── encodeState() → 90-dim state
        ├── ε-greedy: 以 ε 随机走 / 1-ε 选择 maxQ
        ├── 执行走法 → 获得 reward + nextState
        ├── 存入经验回放缓冲区 (max 4096)
        │
        └── 每步学习 (batch=32)：
              ├── 从缓冲区随机采样 batch
              ├── Q_target = reward + γ × max_a' Q_target(nextState, a')
              ├── 损失函数: MSE(Q_online(s,a), Q_target)
              └── RMSProp 更新 online network
        └── 每 256 步：online → target 复制
```

### 关键技术

| 技术 | 说明 |
|------|------|
| **ε-greedy 探索** | ε 初始 1.0，训练中按 `exploringRate *= 0.9999` 衰减 |
| **经验回放** | Replay Buffer 4096 条，随机采样 32 条打破时序相关性 |
| **目标网络** | 每 256 步将 online 网络全量复制到 target 网络，稳定训练 |
| **双训练模式** | trainSelfPlay（双方 DQN）、trainVsRandom（黑方 DQN vs 红方随机） |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 值函数方法，训练相对稳定 | 动作空间离散化受哈希冲突影响 |
| 经验回放提高样本效率 | 需要调节超参数（ε 衰减、学习率等） |
| 目标网络减少训练震荡 | 对局规模大时收敛慢 |

---

## 5. PPOMCTSAgent — PPO + MCTS（AlphaZero 风格）

**文件**: `ppomcts_agent.h / ppomcts_agent.cpp`

### 设计思路

融合 PPO 神经网络与 MCTS 搜索的 AlphaZero 范式。PPO 提供**先验概率 P(s,a)** 和**局面价值 V(s)**，MCTS 使用 PUCT 公式进行树搜索，进一步提高策略质量。

### 网络架构（2026-09 改版）

```
RL::PPO (two-headed network)
  ├── Actor : state(1440) → SparseMoE(E=8, top-2) → Tanh(64) → Softmax(8100)
  └── Critic: state(1440) → SparseMoE(E=8, top-2) → Tanh(64) → Linear(1) → V(s)
```

| | 改版前 | 改版后 |
|---|---|---|
| 状态编码 | 90 维（每格一个标量） | **1440 维**：16 平面 × 90 格（14 棋子平面 + 2 威胁平面），规范视角 |
| 动作编码 | 128 维**哈希** `(id*37+x*13+y*7)%128` | **8100 维**双射 `from*90 + to`，无碰撞 |
| MoE | **稠密** `MOE<8,4>`：8 个 TransformerBlock 专家**全算** | `SparseMoE<MlpExpert,8,2>`：只算门控选中的 **2 个** |
| 参数量 | 约 1.59 M（分析式：actor 0.80 M + critic 0.79 M） | **3 773 813**（实测 `Net::paramCount()`：actor 2 150 124 + critic 1 623 689） |
| 每次模拟的网络前向 | 2×actor + 2×critic | **1×actor + 1×critic** |

实测（`test_ppomcts`，同一台机器上两个二进制**交替**跑 3 轮，取各轮读数）：

| 搜索 | 改版前 | 改版后 | 加速 |
|---|---|---|---|
| 100 sims | 115 / 108 / 104 ms | 22 / 20 / 22 ms | ~5× |
| 500 sims | 591 / 508 / 537 ms | 111 / 103 / 105 ms | ~5× |
| 200 sims | 335 / 283 / 279 ms | 67 / 55 / 61 ms | ~4.8× |

同一张表里的对照列 `MCTS(random rollout)` 在两边几乎一致（78-83 / 147-162 / 299-339 ms），
说明这不是机器负载漂移造成的。**动作空间大 63 倍、状态大 16 倍、参数多 2.4 倍，反而快了
约 5 倍**，来源就是上表最后两行：稠密 8 个 TB 专家 → 稀疏 top-2 个 MLP 专家，以及每个模拟
少算两次前向。

三个**必须配套**的改动（缺任何一个都跑不起来）：

1. **`scaleLayerInit`**：输入维从 90 涨到 1440 之后，`iFcLayer` 默认的 `U(-1,1)` 初始化会让
   pre-activation 的标准差达到 ~22，Tanh 一上来就饱和、梯度接近 0，网络"能跑但不学"。
   普通层按 `1/sqrt(fan_in)` 重缩一遍（专家权重由 `SparseMoE` 构造函数自己缩）。
2. **负载均衡辅助损失**（`moeAuxCoef=0.1`，与 `SACAZAgent` 同值）：实测**初始化时路由已经
   坍缩** —— `test_ppomcts` 的诊断节里，推理 320 次前向只用到 8 个专家中的 4 个
   （`usage = [0,320,320,320,320,0,0,0]`）；训练阶段加上辅助损失之后 8 个专家全部在用
   （`[97,96,114,176,120,62,50,53]`，max/min = 3.5）。
3. **探索采样必须掩码**：`exploreAndTrain` 的 `pick` 原来直接从 softmax 分布采样。
   动作空间只有 128 且有哈希碰撞时，"采到的索引"有相当概率与某个合法走法撞上；
   换成 8100 无碰撞之后命中率掉到百分之几，`rolloutFromCurrent` 会一路退化成
   "走第一个合法走法"，策略再也得不到按自己意愿走子的机会。现在先按合法走法掩码再归一化。

### 动作空间：为什么是 `from*90 + to` 而不是哈希（2026-09）

旧的 `stepToActionIdx` 是 `(id*37 + x*13 + y*7) % 128`：2880 种可能落进 128 个槽位，
平均 22 个不同走法共用一个槽位。落子本身仍然正确（选走法是按**节点访问数**取的，不是按
动作索引），受害的是策略目标的精度 —— 网络学的是"桶"，不是"走法"。

新方案是双射，永不碰撞：

```
actionIdx = fromCell * 90 + toCell          // cell = x*9 + y
```

两个刻意的选择：

* **与棋子 id 无关**。旧哈希用的是 `(棋子 id, 终点)`，而 id 会随吃子被回收复用，于是
  "同一格走到同一格"在不同局面下可能落到不同槽位。用格子坐标就没有这个问题。
* **与状态一样按规范视角镜像**，所以红黑双方的"同一步棋"共享同一个动作槽位。

### 边先验来自父节点（2026-09 顺带修掉的一个既有缺陷）

边 (parent → child) 的先验定义是 `P(s_parent, a) = π(s_parent)[a]`。原来的代码拿的是
**子节点**的策略（`childPolicy[chosenAction]`）—— 既用错了网络（子节点的策略描述的是下一手
走棋方的选择），也用错了视角。在旧的"绝对坐标 + 哈希"编码下父/子恰好共用同一个坐标帧，
所以这个错误只表现为先验质量差；换成规范视角之后两个帧会直接错开（父按红方、子按黑方），
先验会变成完全无关的数。现在 `selectMove` / `trainSelfPlay` / `warmupFromCurrent` 三处
都在**落子之前**求父节点策略。

顺带去掉两次纯浪费的前向：原来每个模拟算 2×actor + 2×critic（"子节点策略"与"叶子估值"
其实是**同一个局面**算了两遍），现在每个模拟只算 1×actor + 1×critic。

> 已知残留：这段模拟循环在三个函数里各抄了一份，三份带着同一个先验错误。这次是逐份同步
> 修改的；正确的做法是抽成一个共用的 `runMctsSimulations()`，目前没做。

### MCTS 节点 (AZNode)

```cpp
struct AZNode {
    int   visitCount;       // N(s,a)
    double totalValue;       // W(s,a)
    double prior;            // P(s,a) — 来自 PPO actor 网络
    vector<int> childIDs;
    vector<int> untriedActionIndices;
    vector<Step> untriedSteps;
};
```

### PUCT 选择公式

```cpp
double getPUCT(int childID, int parentVisits) {
    if (visitCount == 0) return ∞;
    double q   = totalValue / visitCount;                      // Q(s,a)
    double puct = c_puct × prior × sqrt(N_parent) / (1+N_child); // U(s,a)
    return q + puct;
}
```

### MCTS 搜索流程

```
selectMove(color, simulations)
  ├── 对齐 chess.sideToMove = color（规范视角要用；函数结束时原样恢复）
  ├── 创建根节点（这里不下网络；根的策略在它第一次被展开时才算）
  │
  └── 循环 simulations 次：
        ├── SELECTION: PUCT 遍历到未展开节点
        ├── EXPANSION: 落子**之前**先 encodeState(父节点) → ppo.action()，
        │              边的先验 = π(父节点)[动作]
        ├── EVALUATION: PPO value head 给出 V(s)（无随机模拟）
        └── BACKPROP: 路径更新 visitCount/totalValue，每层翻转奖励

返回根节点访问次数最多的子走法
```

### 训练流程

```
trainSelfPlay(episodes, simulations)
  └── 每步：
        ├── 运行 MCTS（由 PPO 引导）
        ├── 从 MCTS 访问分布采样走法（温度退火）
        ├── 存储 (state, one-hot-action, reward)
        │
        └── 终局后：
              ├── 计算折扣回报
              └── trainStep(state, actionTarget, valueTarget)
                    ├── Actor: CrossEntropy(policy, mctsTarget)
                    └── Critic: MSE(value, gameOutcome)
```

### PPO 训练细节

```cpp
void trainStep(state, actionTarget, valueTarget) {
    // Actor loss: 交叉熵(策略输出 || MCTS 访问目标分布)
    policy = actorP.forward(state);
    ceLoss = CrossEntropy(policy, actionTarget);
    actorP.backward(state, ceLoss);

    // Critic loss: MSE(价值输出 || 终局结果 × γ^步数)
    v = critic.forward(state);
    mseLoss = MSE(v, valueTarget);
    critic.backward(state, mseLoss);

    // 参数更新
    actorP.RMSProp(lr);
    critic.RMSProp(lr);
}
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 结合搜索 + 神经网络，精度最高 | 实现复杂，计算开销大 |
| PPO 提供先验 + 价值，无需随机模拟 | 需要大量自对弈训练数据 |
| MCTS 搜索可弥补网络不足 | PPO 与 MCTS 之间接口需精细设计 |
| AlphaZero 范式已被验证有效 | 训练不稳定，超参数敏感 |

### 5.1 对 Alpha-Beta 的实测棋力（2026-09）

用 `test/bench_ppo_vs_ab_main.cpp`（目标 `bench_ppo_vs_ab`，**不进 ctest**）跑的无界面
对弈：交换先后手、每局随机开局 4 步、最多 120 手。这个基准的完整说明与用法见
`README.md` 的「测试与验证」一节。

| 配置 | 比分（PPO 视角） | 局均手数 | 耗时 |
|------|------------------|----------|------|
| 固定预算：PPO 80 次模拟 vs AB 深度 4 | **0 胜 / 6 负 / 0 和** | 49.2 | PPO 21.5、AB 106.0 ms/步 |
| 等时间：`--budget=106`（≈477 次模拟） | **0 胜 / 6 负 / 0 和** | 49.2 | PPO 150.5、AB 127.5 ms/步 |
| 等时间 + 20 局自对弈热身 | **0 胜 / 6 负 / 0 和** | 28.2 | PPO 139.7、AB 143.0 ms/步 |
| 等时间、20 局样本 | **0 胜 / 20 负 / 0 和** | 36.0 | PPO 139.1、AB 142.9 ms/步 |

**全部 38 局都是被将死，没有一局和棋。** 三个必须说清楚的前提：

1. **双方都没训练**。PPO 是随机初始化的网络（`--load` 没给），所以这测的是
   「这个搜索 + 随机策略网络」的**下限**，**不是棋力**。
2. **不是"等参数"的对比**。AB 是确定性搜索 + 手工评估（材质 + PST），PPO 是随机策略
   网络 + PUCT。前者的先验知识是人写进去的，后者需要从零学出来 —— 38 局全败并不意外。
3. **加 20 局热身没有帮助，反而更差**（局均 49.2 → 28.2 手）。这正是 `xiangqi_capacity.md`
   说的量级问题：本机实测 `learnBatch(32)` 在 28.7 M 参数的骨干上要约 0.9 s，
   20 局自对弈提供的样本量离「学会象棋」差着若干个数量级。n=6 也不足以谈统计显著性，
   只能说"看不出一条向好的趋势"。

结论与 README「已知限制」第 1 条一致：**这个工程的价值在链路完整、可训练、可测量，
不在棋力。** 要看训练后的样子，用 `--warmup-games=K`（确认链路不炸）或
`--load=<prefix>` 载入真实训练出来的权重。

---

## 6. Agent 对比总结

| 特性 | ABAgent | MCTS | PGEagent | DQNAgent | PPOMCTSAgent | EVABAgent |
|------|---------|------|----------|----------|--------------|-----------|
| **类型** | 确定性搜索 | 随机搜索 | 策略梯度 | 值函数 | 搜索+学习 | 搜索+学习 |
| **神经网络** | ❌ | ❌ | ✅ DPG | ✅ DQN | ✅ PPO | ✅ 价值网(1260维) |
| **需要训练** | ❌ | ❌ | ✅ 自对弈 | ✅ 自对弈 | ✅ 自对弈 | ✅ 手工评估引导 + TD-leaf |
| **搜索深度** | 固定4层 | 可变(迭代) | 1步 | 1步 | 可变(MCTS) | 固定/限时(可到5层) |
| **随机性** | 无 | 模拟随机 | 策略采样 | ε-greedy | MCTS+PUCT | 无（探索期 ε-greedy） |
| **评估方式** | 手工特征 | 随机模拟 | 策略网络 | Q值网络 | 价值网络 | 手工特征 ⊕ 价值网络 |
| **走法排序** | MVV-LVA | UCB1 | Softmax | argmax/ε | PUCT | TT + killer + history |
| **适合场景** | 有限深度最优 | 纯树搜索 | 学习策略 | 学习Q值 | 搜索+学习 | 搜索深度受限但要求强度 |

## 7. 各 Agent 耗时测试数据

| Agent | 每步耗时 | 强度 |
|-------|----------|------|
| ABAgent (depth=4) | ~50-200ms | 强（有限深度内） |
| MCTS (800 iter) | ~1000ms | 中等 |
| PGEagent (推理) | <5ms | 弱（未经大量训练） |
| DQNAgent (推理) | <5ms | 弱（未经大量训练） |
| PPOMCTSAgent (400 sim) | ~2000ms | 潜力最大（需训练） |
| EVABAgent (depth=5) | ~146ms | **强**（等时间下与 ABAgent 相当或更好） |

> *注：以上为基于测试程序的实测数据，实际耗时和强度取决于训练程度和参数调优。*
> EVAB 的搜索比 ABAgent 快约 21 倍（深度 5：146 ms vs 3133 ms），对抗与训练数据见
> `docs/agent_evab_design.md` §8。*

## 8. 决策前探索（5 个 agent 统一协议）

GUI 每次让 agent 走子前都会先走 `ChessBoard::preTrainThenDecide(agent, color)`，
对应 snakeAI 的 `observe → train → act`：

| Agent | 探索时的动作选择 | 训练调用 | 经验来源 |
|-------|------------------|----------|----------|
| `ABAgent` | —（无参数，默认跳过） | — | — |
| `PGEagent` | `gumbelMax` | `reinforce` | 回合内 log-prob / 奖励 |
| `DQNAgent` | `noiseAction` | `perceive` + `learn` | 自己的 replay buffer |
| `PPOMCTSAgent` | `ppo.action` + `Random::categorical` | `ppo.learnSelfPlay` | PPO 轨迹缓冲 |
| `DQNMCTSAgent` | `noiseAction` | `perceive` + `learn` | 自己的 replay buffer |
| `EVABAgent` | 根节点 ε-greedy 搜索 | 门控 TD-leaf 更新 | 树搜索叶节点样本 |

GUI 侧用复选框"走子前先探索训练"开关（默认开），步数由 `setPreTrainSteps()` 控制；
关闭时 `preTrainThenDecide` 退化为纯粹的 `selectMove`，与旧行为一致。

## 9. Agent 对 Agent 对弈（arena）

原来只有 `selfPlay(agentType)`：**同一个 agent** 自己跟自己下。它能说明"能不能收敛"，
但没法回答"哪个 agent 更强"。现在换成 `ChessBoard::matchAgents(typeA, typeB, games)`。

### 9.1 两条方法学要求

1. **每局交换先后手。** 中国象棋先手（红）优势很大，固定谁执红的话，结果只是在测
   "谁执红"而不是"谁更强"。所以胜负按**参赛者 A/B** 记，不按红黑记；
   偶数局 A 执红，奇数局 B 执红。日志里会把每局的执红方写清楚。
2. **局数要够，且可配置。** 单局的偶然性足以翻转结论，所以局数做成了界面参数，
   并且逐局列出明细（谁执红、谁胜、多少手）。

`matchAgents(A, A, games)` 就是原来的自对弈，所以旧功能没有丢。

### 9.2 界面

控制栏新增（都在 `mainwindow.ui`）：

| 控件 | 作用 |
|------|------|
| `matchAComboBox` / `matchBComboBox` | 两个参赛 agent（默认 A=Alpha-Beta, B=EVAB，都快，且正好是"纯搜索 vs 学会评估的搜索"） |
| `gamesSpin`（局数，1–100，默认 4） | 对弈局数 |
| `preTrainStepsSpin`（预训，0–2000，默认 64） | 每手决策前的探索步数上限，**0 = 不探索**（用来做对照） |
| `selfPlayBtn`（"开始对弈"） | 开始；对弈进行中同一个按钮变成"停止对弈" |
| `matchResultLabel` | 实时进度 → 最终比分，tooltip 里是逐局明细 |

对弈跑在后台线程，进度通过 `matchStarted` / `matchGameFinished` 信号回到 GUI 线程；
状态条与右侧指示器沿用第 8 节那套，阶段文字会带上"对弈 2/4 局 · 第 17 手"。

### 9.3 实测（`test_match`，18 条断言全过，约 2 s）

用 `setMaxPliesPerGame(4)` 把每局压到 4 手 —— 4 手之内不可能将杀，于是每局必然判和，
断言就能做得很硬：

```
Alpha-Beta 0 : 0 EVAB (和 2)  共 2 局 / 8 手
  第 1 局: 红=Alpha-Beta 黑=EVAB -> 和棋  (4 手)
  第 2 局: 红=EVAB 黑=Alpha-Beta -> 和棋  (4 手)
```

界面侧用 `tools/verify_match_ui.ps1`（走 Windows UI Automation，不依赖焦点、
不做像素识别，直接从无障碍树里读控件矩形和结果文字）：

```
start button rect = 1454,751,238,23
spinner count = 2 (expect 2: 局数 / 预训)
match_running = True  (button switched to 停止对弈)
result label = Alpha-Beta 1 : 0 EVAB (和 1)  共 2 局 / 368 手
   探索+预训练: 探索 64 步, 价值网络更新 1 次 (|net-hand| 0.814368)
RESULT: PASS
```

即"局数设置生效、预训练步数生效、对弈真的跑完、比分真的显示出来"。

### 9.4 这个方法顺带挖出来的三个真问题

写 arena 的过程中，ASan 抓到一个远比功能本身重要的 bug —— 详见
`docs/issues_review.md` 的"零之二点九"：AI 工作线程会被无谓地唤醒，和 arena 线程
**同时搜索同一张 `env`**，导致 `env.history` 双重释放。这也解释了为什么"agent 偶尔
返回一步非法走法"，而非法走法以前会被当成"走棋方被将死"直接判胜负。

---

## 10. "走子前先探索 + 预训练" 的利弊（理论分析）

这一节是设计取舍的记录，不是实测报告；相关的实测数字见
`docs/issues_review.md` 的"零之二点七"与 `agent_evab_design.md` §8.3。


### 10.1 它到底是什么算法

对每个 agent，`exploreAndTrain(color, N)` 做的是：从**当前局面**出发，用自己的
探索策略滚 ≤N 步，收集 transition，做**一次**参数更新，然后仍然用 argmax/搜索决策。
用 RL 的术语说：这是**单条轨迹片段 + 每步一次更新的在线（continual）学习**，
更新次数与"真实对局的手数"同阶。

对比三种标准做法：

| | 数据来源 | 每次更新用多少样本 | 数据是否 i.i.d. |
|---|---|---|---|
| 经验回放 (DQN) | 大 buffer 里均匀采样 | mini-batch（几十~几百） | 近似（回放的意义就在这） |
| 批量 on-policy (PPO) | 一整批 rollout | 几千~几万步 | 是（同一批策略） |
| **本工程的预训练** | **当前局面的 ≤N 步** | **N ≈ 32~64** | **不是，高度相关** |

### 10.2 好处（为什么值得做）

1. **把"GUI 里从来不训练"这个洞补上了。** 之前所有在线训练接口在界面侧零调用
   （A15），GUI 对局纯粹是拿初始随机网络在下棋，玩家下 100 局模型也毫无变化。
   预训练至少让每一步都产出一份梯度。
2. **对 EVAB 这类"学习评估函数"的 agent，蒸馏是有意义的。** 搜索本身是昂贵的
   教师信号：把搜索出来的叶节点价值回填给价值网络，是 AlphaZero 式的正路。
   实测 |net−hand| 从 1.23 降到 0.05，说明这条通路确实能学到东西。
3. **它给了"确定性决策"一条探索通道。** 决策走 argmax、搜索走确定性走法时，
   智能体永远不会访问未被选中的分支，也就永远没有新数据。探索 rollout 打破了
   这个闭环（这也是为什么实现里必须用 `noiseAction`/`gumbelMax`/采样，
   而不是在探索阶段也 argmax）。
4. **训练与对局的时间尺度一致。** 不需要额外的训练循环，玩家下棋的同时就在训练，
   对单机小工程是很实际的选择。

### 10.3 坏处（为什么不能指望它带来棋力提升）

1. **单条轨迹片段的梯度方差极大、且高度相关。** 一次更新只有 N 个样本，且它们
   来自同一条（自己生成的）轨迹，有效样本量远小于 N。用它做一步随机梯度下降，
   方向噪声远大于批量训练。
2. **目标在动。** 每步都更新 ⇒ 策略变 ⇒ 数据分布变 ⇒ 价值函数追着一个移动的
   目标。理论上的收敛条件（策略固定时的策略评估）在这里根本不成立，做的是
   "非平稳目标上的在线逼近"，没有收敛保证。
3. **致命三元组风险。** 函数逼近 + 自举（TD 目标用同一张网）+ 偏离 on-policy 的
   数据 —— 三者同时出现时值函数可能发散。EVAB 的实测正是这样：TD-leaf 精修在
   4 局/轮的样本量下误差从 0.35 涨到 0.70，最后只能靠 `acceptNet` 门控（不通过就
   整体回滚）兜住。**门控是补丁，不是解法**：它保证"不变坏"，不保证"变好"。
4. **对 DQN 系是**有害的**。DQN 依赖回放缓冲近似 i.i.d.；每步塞进 32 条**刚刚
   由当前网络自己生成**的、时序高度相关的样本，会系统性地把回放分布拉向当前
   对局的局部状态（近因偏置），并让 TD 误差被当前网络的自洽性压低 ——
   学到的信号反而变弱，同时冲掉早期经验（遗忘）。**对 DQN 来说，
   "每步训练一次"在数据分布上比"什么都不训练"更糟**（后者至少不会破坏回放假设）。
5. **对 PPO 系是"基本无害但要小心"。** PPO 的裁剪目标假设数据来自**最近的**策略，
   而"每步从当前局面重新采一批"恰好是最新鲜的 on-policy 数据，所以偏差最小。
   但优势估计依赖那 N 步的回放：**如果 N 步之内没有终局、又没有用价值网络自举
   做 GAE 截断**，优势几乎全是 0，梯度等于噪声。象棋的奖励是**稀疏终局奖励**
   （几乎全是 0，只有将杀 ±1），64 步内通常到不了终局 —— 这是本工程最关键的
   结构性问题。
6. **信息效率很低。** 探索是**真下在真棋盘上**的（`env` 副本上试走），
   单步成本实测 69~471 ms（32 步）。对 MCTS 类 agent，搜索本来就已经访问了大量
   状态，再从根节点重新滚一遍是**重复劳动**，边际信息接近 0，却付出成倍时间。
   在 arena 里这个代价尤其直接：**单步时间翻倍 = 同样时间里能打的局数减半 =
   统计功效减半**，"哪个 agent 更强"这个结论更难做出来。
7. **它改动了测评的对象。** 打开预训练之后，被测评的已经不是"这个 agent"，
   而是"这个 agent + 每步一次在线更新"这个复合系统；两边的结果不再可比。
   这也是为什么 arena 必须把"预训"步数做成显式参数并允许设 0。

### 10.4 结论与建议

- **保留，但按 agent 区别对待**：EVAB（学习评估）与 PPO（最新 on-policy 数据）
  受益；**DQN 系建议默认关闭**（它破坏回放假设），或改成"只写回放、不立即更新"
  （把更新攒到 buffer 满再做 mini-batch），这与"预训练"的直觉相反但符合 DQN 的理论。
- **优势/回报要么有终局信号，要么显式自举**：N 步内没到终局时不要用"未折扣的
  终局奖励"当目标，应当用价值网络在 N 步处自举（TD(λ)/GAE 截断）。
- **棋力结论只能来自"预训练 = 0"的对照**；arena 的对照组就是为此存在的。
- 可检验的预测（尚未做）：把预训练打开去打它自己冻结的副本，**应当不优于**
  且很可能劣于冻结副本 —— 如果结果是显著更好，说明当前的更新规模/门控确实
  在起作用，值得进一步投入；如果持平或更差，就该把预算转向离线批量训练。




---

## 11. SAC + MCTS + AlphaZero（`SACAZAgent`）

代码：`src/sacazagent.h/.cpp`，测试 `test/test_sacaz_main.cpp`（35 条断言）。
界面下拉框里叫 **"SAC+MCTS+AlphaZero (最大熵搜索)"**。

### 11.1 三个成分各负责什么

| 成分 | 在这个 agent 里承担的角色 |
|------|--------------------------|
| **AlphaZero (PUCT MCTS)** | **策略改进算子**：搜索得到访问分布 π_MCTS，当作策略的监督目标。低方差、质量高于当前策略。 |
| **SAC（最大熵）** | ① **叶子估值**：用软价值 `V(s)=Σ_a π(a)[min_i Q_i(s,a) − α·log π(a)] = E_π[min Q] + α·H(π)`，而不是单独的价值头；② **双 Q + 目标网 + 回放**（象棋奖励是稀疏终局奖励，off-policy 回放让一条终局经验被反复利用，这是 SAC 这一半真正的价值）；③ **α 自动调节**（熵低于目标就抬高 α）。 |
| 二者结合 | 搜索提供"往哪走"（π_MCTS），SAC 提供"局面有多好"（软价值）与"数据怎么反复用"（回放）。 |

### 11.2 为什么没有直接用 `RL::SAC`

`src/rl/sac.cpp` **已经是一个离散动作的 SAC**（one-hot 动作、softmax 策略、对全部动作求期望的软备份、可学习的 α），数学上完全够用。不用它而在这里重写，是三个象棋/搜索特有的原因：

1. **合法走法掩码**。128 维动作空间里绝大多数是非法走法，策略必须在掩码后的分布上定义（π(a)=0，a 非法）。`RL::SAC` 没有掩码，而且它的 critic 输入被拼成 `[state; prob]`，与掩码策略不自洽。
2. **与 MCTS 的整合**。策略目标来自访问分布、叶子估值要换成软价值，这要求 agent 掌握"前向/反向"的先后顺序（例如 `Layer::backward` 会清掉 `o`/`e`，必须在调用前把 π 拷出来）。库里的 SAC 是黑盒。
3. **掩码 softmax 的雅可比**。策略头输出 logits，掩码归一化必须自己实现它的反向，不能只对 `Layer<Softmax>` 的输出打补丁。

复用的仍然是库里的 `RL::Net` / `Layer<*>` / `Loss::MSE` / `Optimize::RMSProp` / `Random`。

### 11.3 关键实现点（容易写错的地方）

**状态编码用规范视角 14×90 = 1260 维**（与 EVAB 同一约定：7 类棋子 × {己方, 对方} 的 one-hot 平面，轮到黑方时 `x → 9-x` 镜像）。这样"轮到谁走"被编码进视角，同一个网络对红黑双方都是"我方的子在自己下方"。本项目其它 RL agent（PG/DQN/DQNMCTS）用的是 90 维**非**规范编码，价值函数的"谁走"是隐含的 —— 对 AlphaZero 这种"策略/价值都从走棋方视角定义"的算法，规范视角是必需的。（`PPOMCTSAgent` 原来也在"非规范"这一列，2026-09 已单独改成同一套规范视角，见 §5。）

**掩码 softmax 与它的反向**（全篇最容易错的一处）：
```
π_a = m_a·e^{z_a} / Σ_b m_b·e^{z_b}
dL/dz_c = π_c·(g_c − Σ_a g_a·π_a)        (g = dL/dπ)
```
测试里对这一条做了**有限差分核对**（对 L = Σ g_a π_a 求 dL/dz），相对误差 1.2e-4。

**策略损失是两项之和**（都对 softmax 输出 π 求导，再交给上面那个雅可比）：
```
dJ/dπ_a = α·(log π_a + 1) − min_i Q_i(s,a)      ← SAC 软 Q 项
        + azWeight·(π_a − π_MCTS,a)              ← AlphaZero 监督项 (交叉熵)
```
监督项刻意写成 `π − π_MCTS` 而不是 `−π_MCTS/π`：两者数学上等价，但后者在 π_a → 0 时会爆掉。

**符号（negamax）**：所有价值都是"该局面走棋方视角"，所以 TD 目标是
`y = r − γ(1−done)·V(s')`（s' 轮到**对手**走，自举项要取负）。这一条写反了整局就学不出来。

**叶子估值的分工**：搜索时用**在线**双 Q（估计更准），学习时的备份用**目标**双 Q（SAC 的常规做法）。

**Q 头用 `Layer<Linear>` 而不是 `Layer<Sigmoid>`**：象棋奖励含负值（输棋 −1、丢子为负），Sigmoid 的值域 (0,1) 结构上无法表示负 Q —— 正是 `issues_review.md` B18 记的问题（`dqn.cpp` 至今仍用 Sigmoid 头）。

**初始化缩放**：`iFcLayer` 构造函数把权重初始化成 U(−1,1)，对 1260 维输入来说 pre-activation 标准差约 20，Tanh 直接饱和、梯度接近 0。所以按 `1/sqrt(fan_in)` 再缩放一遍（`scaleLayerInit`）。

**比 PPOMCTSAgent 多做的三点**：
1. 叶子用软价值（最小熵正则化的 Q），不是一个价值头；
2. **终局节点用真实胜负**，不再自举（PPOMCTS 在将杀局面也用网络值）；
3. 展开时按**先验**挑未尝试动作，而不是随机挑。

### 11.4 为什么主干用 MLP 而不是本项目的 `MOE<16,16>` + `TransformerBlock<16>`

本项目的 `DQNAgent` / `RL::SAC` 用的是 `MOE<16,16>` + `TransformerBlock<16>` 主干，
本 agent 用的是普通的 `1260 -> 64 -> 64 -> 128` MLP。这是**量过之后**的选择，数据在
`test_sacaz` 的第 [8] 节（信息性，不做断言）：

| 主干 | 单次前向 | 说明 |
|------|---------|------|
| 本 agent 的 MLP（1260→64→64→128） | **0.017 ms** | 直接实测 |
| `TransformerBlock<16>` @1260（1 个专家） | 9.533 ms | 直接实测 |
| `MOE<16,16>`+`TB`+`TanhNorm` **@1260** | **152.5 ms** | 16 × 上面那行 + 门控（外推） |
| `MOE<16,16>`+`TB`+`TanhNorm` **@90**（DQN 的配置） | 0.733 ms | 直接实测 |

一次决策 = 256 次模拟 × 3 个网络（actor + 双 critic）：

- MLP 主干 ≈ **13 ms**（这是 GUI 现在的实际开销）
- MOE 主干 @1260 ≈ **117 秒**（一次走子）

所以 MOE 在本 agent 上是**不可用**的：慢 9152 倍，而且参数量 ~305M/网络 ≈ 1.2 GB
（本 agent 有 5 个网络：actor + 双 critic + 双目标网 ≈ 6 GB）。

三个原因值得记住：

1. **这里的 `MOE::forward` 是稠密混合**：它**循环调用全部 16 个专家**再做门控加权求和
   （`if (gi > 1e-8f)` 只跳过累加、不跳过计算）。MoE 本该靠"每个 token 只走少数专家"
   省算力，这里没有任何条件计算，代价就是干净的 16 倍。
2. **代价随 d_model² 增长，而本 agent 的输入恰好是 1260 维**。每个专家内部是一个
   `TransformerBlock`，其 attention 是 O(d²)、FFN 是 2·d·4d = O(d²)；DQN 的 d_model=90，
   所以它的 MOE 只要 0.733 ms，而 d 放大 14 倍后 d² 放大 196 倍、再乘 16 个专家 ——
   这就是 9152 倍的来源。**MOE 能不能用，取决于输入维度**，而不是"MOE 好不好"。
3. **架构收益在这里也拿不到**：`TransformerBlock` 收到的是一个 `(d_model, 1)` 的
   **单个向量**（没有 token 序列），所以 attention "关联序列位置" 的作用无从发挥，
   留下的是一个昂贵的特征混合块；单个输入向量也无法让专家按 token 特化。

另外这个主干在本工程里有历史包袱：`docs/rl_sync.md` §1.2 记录它**同步前是坏的**
（`attention.hpp` 曾把 d_model 90 向上取整到 96，而 `TransformerBlock` 的缓冲仍按 90
分配 → 越界读写）。它现在可用，但确实是这一类维度错配的高发区。

**想用 MOE 的正确姿势**（如果要，建议按这个顺序试，并用 arena 量）：

1. **稀疏路由 + 廉价专家**（唯一在 d=1260 上可行的路线）。把专家从
   `TransformerBlock` 换成普通 MLP（1260→64→64→1260，单个 **0.021 ms**），
   gate 只取 top-k 并只计算被选中的专家。实测：top-1 = 0.021 ms/前向 ⇒
   一次决策 ≈ **16 ms**；top-2 ≈ 21 ms；E=8 的总参数量 ≈ 1.29M ≈ 5 MB。
   注意"专家数 E"只影响**参数量/容量**，不影响前向计算量（top-k 只算 k 个）。
2. **先降维再用 MOE**：`Layer<Linear>(1260, 90)` 压到 90 维再接 `MOE<16,16>::_(90)`
   （0.733 ms/前向）。
3. **只在部分网络用**：例如只给 actor 或只给 critic 上 MoE。
4. **换回 90 维非规范编码**：那样 MOE 便宜，但会丢掉"轮到谁走"的规范视角
   （AlphaZero 的策略/价值都从走棋方视角定义，这个代价可能更大）。

### 11.4.1 两个反直觉的实测结论（`test_sacaz` 第 [9] 节）

**(a) 减少 head 数是变贵的，不是变便宜。** 本库的 `MultiHeadAttention` 从 `NumHeads`
往下找能整除 `d_model` 的 head 数，`d_k = d_model / head 数`，每个 head 的注意力矩阵是
`d_k × d_k`，所以总注意力代价 ≈ `Σ head数 × d_k² = d² / head数`。d_model=1260 时实测：

| NumHeads | 实际 head 数 | d_k | 单次前向 |
|---------:|-------------:|----:|---------:|
| 1 | 1 | 1260 | **49.58 ms** |
| 4 | 4 | 315 | 16.73 ms |
| 8 | 7 | 180 | 11.97 ms |
| 16 | 15 | 84 | **9.63 ms** |

即"1 head"比"15 heads"贵 5.1 倍。**要降注意力成本应该加 head 数（上限是 ≤NumHeads 的
最大真因数），而不是减。** d_model=1260 时 NumHeads=16 就已经取到 15（1260 = 2²·3²·5·7），
接近最优。

**(b) 光靠稀疏路由救不了 `TransformerBlock` 专家。** top-k 只把"16 倍"降成"k 倍"，
但**单个专家在 d=1260 上本身就要 9.63 ms**（O(d²) 的注意力 + 8d² 的 FFN）：

| 配置 | 一次决策 (256 模拟 × 3 网络) | 可行? |
|------|----------------------------:|:-----:|
| 普通 MLP（当前） | **13 ms** | ✅ |
| 稀疏 MoE（E=8, MLP 专家, top-1） | **16 ms** | ✅ |
| 稠密 MOE（TB 专家） | 118 s | ❌ |
| 稀疏 MoE（TB 专家, top-1） | 7.4 s | ❌（256 模拟下） |

所以结论是：**稀疏路由方向对，但必须同时把专家换成不含 MHA 的廉价结构**；
只做稀疏路由、保留 `TransformerBlock` 专家，在 256 次模拟的预算下仍然不可行。

> **后来实测修正（见 §11.4.2）**：上表最后一行是**外推**。真正接进 agent 之后量到的是
> **9.9 ms/模拟**（`bench_moe`，比外推的 `9.63 × 3` 还便宜，因为一次模拟并没有 3 次完整
> 主干前向），于是 256 模拟 = 2.5 s/步（仍不可行），但**16 模拟 = 160 ms/步是可行的** ——
> 这就是界面上那个 `AGENT_SACAZ_MOE` 用 16 次模拟的原因。结论要改成"**用很少的模拟
> 次数才可行**"，而不是"不可行"。

**这会提升效果吗？** 诚实地说：**机制上说得通，但本工程没有任何证据**，而且有三个已知风险：

- **专家坍缩**：top-k 的选择是离散的，梯度只流向被选中的专家（门控本身的梯度也只来自
  被选中的那几个分量），所以没有负载均衡辅助损失时 gate 会迅速偏向少数专家、其余永远
  不训练 —— 这是 MoE 的经典失败模式，在"自对弈几千个局面 + 回放"的数据量下风险不低。
- **路由抖动**：gate 随训练漂移，同一个局面可能换专家；对 SAC 的**目标网**来说这是新的
  非平稳来源，会放大 TD 目标的方差（§10 已经指出"每步一次更新"本身就偏不稳定）。
- **本工程的 MoE 是稠密的**（`MOE::forward` 16 个专家全算），那是昂贵集成，不是稀疏
  MoE，因此没有"专家确实特化了"的先例可循。

要做的话，建议按这个**对照实验**设计（只有实测能回答"能否提升"）：

| 组 | 配置 | 目的 |
|----|------|------|
| A | 当前 MLP（h=64） | 基线 |
| B | 稀疏 MoE：E=8，专家 h=64，top-1，+ 负载均衡辅助损失（系数 ~0.01） | 待验证 |
| C | 宽 MLP（h≈256） | **等参数量**对照（B 的 1.29M vs C 的 0.65M，可调到等参数） |
| D | 稠密 MoE：E=4，专家 h=64 | **等算力**对照（4×0.021 ≈ 0.084 ms/前向） |

评估用 §9 的 arena，**等时间**（不是等模拟次数）、预训=0、每组几十局（单局偶然性足以
翻转结论，见 §9.1），同时记录**专家使用分布**（判断是否坍缩）与每步耗时。

→ **这个方案已经执行，过程和实测数据在 §11.4.2。** 一句话预告结论：稀疏路由的
"容量不按算力付费"确实兑现了（同参数下快 3.8 倍），但 TB 专家的绝对单价让它在
这个状态空间上只能配很低的模拟次数；而**这一轮没有拿到任何棋力结论**（全是和棋）。

### 11.4.2 已实现：稀疏路由 MoE（`src/rl/sparse_moe.hpp`）+ 骨干开关 + TB 专家 agent

上面那张对照表已经按方案做完了。这一节记录**做了什么、验证了什么、量到了什么**，
以及一个把整个功能卡住两轮的隐蔽 bug。

#### (1) 代码

| 文件 | 内容 |
|------|------|
| `src/rl/sparse_moe.hpp`（新） | `MlpExpert`、`ISparseMoE`（非模板接口）、`ExpertFactory`、`SparseMoE<Expert, E, TopK>`、`scaleExpertInit` |
| `src/sacazagent.{h,cpp}` | `Backbone` 枚举 + `buildNet()`（四种骨干一个工厂）、辅助损失钩子、`moeUsage()` 坍缩诊断 |
| `src/chessboard.{h,cpp}` | 新 agent 类型 `AGENT_SACAZ_MOE`（稀疏 MoE + TB 专家，模拟 16 次）、权重前缀 `weights/sacaz_moe_agent_*` |
| `src/mainwindow.cpp` | 下拉框新条目 "SAC+MCTS+AlphaZero (稀疏MoE+TB专家)" |
| `test/test_sparse_moe_main.cpp`（新） | 67 条断言，见 (3) |
| `test/bench_moe_main.cpp`（新） | A/B/C/D 等时对弈基准（**不进 ctest**） |

关键点：**上游 `rl/moe.hpp` 一个字都没改**。`SparseMoE<Expert, E, E>`（TopK == 专家数）
就是"稠密 MoE"，所以对照实验的 D 组不需要第二份实现 —— 这也让 (3) 里的等价性检查
成为可能。

`SparseMoE` 的接口与 `MOE` 保持一致（`wg`/`gate`/`g.wg` 的语义、`forward/backward/SGD/
RMSProp/Adam/clamp/copyTo/softUpdateTo/write/read`），另外多了三样：

- `ISparseMoE::addAuxGradient(coef)`：负载均衡辅助损失 `L_aux = E·Σ f_i·P_i` 的梯度
  （只写门控参数）。`dL/dP_i = coef·E·f_i`，再过一次 softmax 的雅可比。
- `ISparseMoE::usageSnapshot()/resetUsage()`：每个专家被选中的次数（坍缩诊断）。
- `SparseMoE<..., TopK>=NumExperts` 退化成稠密对照。

#### (2) 一个把功能卡住两轮的 bug：`iFcLayer` 的拷贝构造丢掉了全部参数

现象：`SparseMoE` 的反向在 `E=4, top=2` 下报出 **15615 个非有限梯度元素**（NaN/1e28），
而 `E=1`/`E=2` 有时是干净的；同一个二进制不同次构建结果还不一样。

根因不在稀疏 MoE 里，在 `src/rl/layer.h`：

```cpp
// 修复前
explicit iFcLayer(const iFcLayer &r)
    : iLayer(r), inputDim(r.inputDim), outputDim(r.outputDim), bias(r.bias) {}
```

**它只复制了维度，没有复制 `w` / `b` / `o` / `e` / `g` / `v` / `m`。** 而上游 `Layer<T>`
的 `copyTo()` 是走 `pLayer->w = w` 的赋值（深拷贝），所以这个问题平时看不出来；
`SparseMoE` 里 `experts[i] = ExpertFactory<Expert>::make(...)` 是**按值返回再赋值**，
只要 MSVC 没有省略这次拷贝，专家就变成"维度对、权重空"的空壳 —— `w.size()==0`，
之后所有 `MM` 内核都在越界读写。

用 12 行的最小复现钉住了它（`build/copytest.cpp`，临时文件不入库）：

```
a (直接构造)   w.size=10080  w[0]=-0.847383
b (拷贝构造)   w.size=0      <-- 修复前
b (拷贝构造)   w.size=10080  w[0]=-0.847383  <-- 修复后
```

修好之后门控梯度从 `2.6e33` 变成 `2.7e-2`（有限），稀疏 MoE 的反向才通过了有限差分。
**这类 bug 的教训**：`Net` 的拷贝是浅拷贝（共享层指针），深拷贝必须走 `copyTo`；
而"按值返回一个层"这种写法会静默地走拷贝构造 —— 在含 `std::vector` 成员的类里，
"只复制标量成员"的拷贝构造就是一个定时炸弹。

#### (3) 反向到底对不对：`test_sparse_moe`（67 条断言，约 1.4 s）

比"跑起来不炸"更有意义的是这四类检查：

| 检查 | 方法 | 结果 |
|------|------|------|
| 前向真的跳过了未选中的专家 | `E=4, top=1` 时把未选中专家的权重改 **0.5**（不是 1e-6 的抖动），输出必须**逐位不变** | 变化 = 0.000e+00；改选中专家则 1.96e-02 |
| 门控约定与上游一致 | `SparseMoE<TransformerBlock<4>, 3, 3>` 与 `MOE<3,4>` 权重逐层 `copyTo`，比前向与门控梯度 | 前向差 **0.000e+00**，`dL/dwg` 差 **0.000e+00**（且 \|dL/dwg\|=130.6，不是"0 比 0"的假通过） |
| 解析梯度 vs 中心差分 | `E=4, top=2`，全部 `wg`(4×64) / `bg` / 两个选中专家的 w+b 逐个查 | 相对误差 **1.06e-03**(eps=1e-3)、**7.19e-05**(eps=1e-2) |
| 未被选中专家的梯度 | 同上 | 解析与数值**都恰好为 0**（它本来就不该被更新） |
| 辅助损失的梯度 | 单独构造 `L_aux = coef·E·Σ f_i P̄_i`，对 `wg`/`bg` 做中心差分 | `dwg` 7.18e-05、`dbg` 2.85e-05 |
| 辅助损失的方向 | 人为把路由压塌到专家 0，看 `dz` 的符号与多步后的熵 | 被喂爆的 `dz=+4.5e-2`（压下去）、饿着的 `dz=-1.5e-2`（抬起来）；熵 0.5291 → 1.0941 |

有限差分在 `eps=1e-4` 下误差反而变大（6.7e-3）——这是 float32 的舍入噪声底，不是数学错：
`L = Σ(o−t)²` 量级在 1e2~1e3，float32 分辨率约 `1e-4`，而 `(L(+ε)−L(−ε))/(2ε)` 在
`ε=1e-4` 时的噪声就是 `1e-4/2e-4 = 0.5` 的绝对量级，而梯度是 6.3。所以断言只放在
`eps ≥ 1e-3` 上，`eps=1e-4` 只作为"误差随 eps 减小而增大"的信息打印。

> 顺带修掉测试自己的一个假通过：第一版等价性检查直接调 `MOE::backward()`，而
> `Net::backward()` 才会先把 loss 写进最后一层的 `e` —— 结果两边的 `e` 都是 0，
> 梯度都是 0，差值当然也是 0，"等价性通过"完全是无意义的。现在改成两边都走
> `Net`，并额外断言 `|dL/dwg| > 1e-6`（防止再退回"0 比 0"）。

#### (4) 辅助损失系数到底该取多大（实测，不是抄论文）

论文里 Switch Transformer 用 0.01，但那里的"批"是几千个 token，而这里一个
mini-batch 只有 32 个样本、主干给门控的梯度比辅助项大两个数量级。实测（`B` 骨干
= 稀疏 MoE + 8 个 MLP 专家，预训练 3 局自对弈、20 次 `learnBatch`）：

| `auxLossCoef` | 一次都没被选中的专家 | 使用计数 最大/均值 |
|--------------:|:-------------------:|------------------:|
| 0（关掉） | **3 / 8** | 4.00 |
| 0.01 | 0 / 8 | 4.00 |
| **0.1（现在的默认值）** | **0 / 8** | **2.91** |
| 0.2 | 0 / 8 | 3.22 |
| 0.5 | 0 / 8 | 2.63 |
| 2.0 | 0 / 8 | 2.32 |

两件事由此确定：**(a) 不加辅助损失时路由确实会塌**（8 个专家饿死 3 个，且是最经典的
"少数专家吃满"形态）；**(b) 0.01 太小**，只能勉强不让专家彻底饿死，分布仍然极偏。
默认值因此取 **0.1**（`SACAZAgent::auxLossCoef`、`ChessBoard` 的 `SACAZ_MOE_AUX`）。
0.5 以上更均匀，但辅助项开始盖过真正的策略梯度，属于"为了均衡牺牲拟合"。

`C` 骨干（4 个 TB 专家、top-1）在同一系数下最大/均值只有 1.17–1.37 —— **专家越少、
top-k 越小，路由越不容易塌**，这与 MoE 的经验一致。

#### (5) 代价与"等参数不等算力"的对照（`bench_moe`，等时间预算 60 ms/步）

| 组 | 骨干 | 参数量 | ms/模拟 | 等时模拟次数 | 实际 ms/步 |
|----|------|-------:|--------:|------------:|-----------:|
| A | MLP 1260→64→64→128 | 0.09 M | 0.074 | 64（上限） | 4 |
| B | 稀疏 MoE（E=8, MLP 专家, top-2） | 1.43 M | 0.311 | 64（上限） | 20 |
| C | 稀疏 MoE（E=4, **TB 专家**, top-1） | 28.70 M | 9.918 | 6 | 60 |
| D | 稠密 MoE（E=4, TB 专家, **全算**） | **28.70 M** | 37.675 | 2 | 75 |

**C 与 D 参数量完全相同（28.70 M，同一个 `SparseMoE` 模板，只有 `TopK` 不同），
而稀疏路由让每次前向快 3.8 倍**（37.675 → 9.918 ms/模拟）。这是本次工作唯一一个
"确定性的、可复现的收益"：**容量不按算力付费这件事确实兑现了**。

但要诚实地说清楚它的**代价**：绝对算力上，`d=1260` 上的一个 `TransformerBlock` 专家
（O(d²) 注意力 + 8d² 的 FFN）就是要 3 ms 级，所以 C 组在 60 ms/步的预算下只能跑
**6 次模拟** —— 而 0.074 ms/模拟的 A 组能跑 64 次。**MCTS 的模拟次数本身是棋力的一部分
（§9 的 arena 与 §7 的数据都说明搜索深度/次数直接换算力），用 6 次模拟去换 300 倍参数量，
在这个状态空间上是亏的。** 稀疏路由解决的是"专家数不能乘在算力上"，解决不了
"单个专家本身就贵"。

#### (6) 对弈（`bench_moe`）：这一轮**没有得到棋力结论**，如实记录

按 §9.1 的方法学要求做了等时间、交换先后手、随机开局 6 步、预训 = 0（`--pretrain=3`
时两边同预算）、并记录了每步耗时与专家使用分布。结果是：

```
参赛者               胜    和    负 | case ms/步   A ms/步 | 平均手数
A  MLP                    0      4      0 |        4.1        4.1 |     16.8
B  稀疏MoE(MLP专家)      0      4      0 |       17.3        4.3 |     13.8
C  稀疏MoE(TB专家)       0      4      0 |       59.8        4.3 |     29.5
D  稠密MoE(TB专家)       0      4      0 |       84.0        4.2 |     17.8
```

**全是和棋**，连 A vs A 的自我对照也是 4 和。原因很直接：所有 agent 都是随机初始
权重（预训 3 局 = 20 次 `learnBatch`，对 28.7 M 参数的网络等于零），谁都不会杀棋，
于是要么拖到 30 手上限判和，要么走成重复局面判和。**因此这张表只能说明"四种骨干都
能在真实对局里走完整局、不崩、耗时符合预期"，不能说明任何一种更强。**
要得到棋力结论必须投入真正的训练预算（自对弈几千局量级），并与预训=0 的对照比较；
本工程目前没有这个预算，也没有做这个结论。

这一点也和 §10 的分析一致：**预训练/探索本身不是棋力来源，训练量才是。**

#### (7) 界面

下拉框里的新条目 "SAC+MCTS+AlphaZero (稀疏MoE+TB专家)" 对应 `AGENT_SACAZ_MOE`：
模拟次数 16（约 175 ms/步）、权重前缀 `weights/sacaz_moe_agent_{actor,q1,q2}`（与 MLP
骨干那套**不能共用**，层结构不同）。`test_sacaz` 的第 [10] 节会遍历四种骨干走一遍
真实 agent 路径（建网 → 决策 → 探索 → `learnBatch` → 诊断 → **权重存取往返**），
所以"界面里选得到、能走合法棋、训练链路不炸、权重存得下也读得回"这几件事有自动化覆盖。

**真界面端到端**（`tools/verify_match_ui.ps1 -AIndex 8 -BIndex 0 -Full`）：

```
Qt bin prepended = C:\Qt\6.9.2\msvc2022_64\bin
main window ready after ~0.5 s
A side agent = SAC+MCTS+AlphaZero (稀疏MoE+TB专家)
B side agent = Alpha-Beta Pruning (深度=4)
match_running = True  (button switched to the stop label)
result label = SAC+AZ-MoE 0 : 1 Alpha-Beta  共 1 局 / 60 手
   探索+预训练: SAC+AZ 探索 2 步, 训练 1 次 (池 1782, critic loss 0.1804, alpha 0.239)
A side in result = True (want 'SAC+AZ-MoE')
B side in result = True (want 'Alpha-Beta')
RESULT: PASS
```

> 这次跑之前先踩了两坑，都在脚本里修掉了：**(1)** 从普通 shell 启动时 `PATH` 里没有
> Qt 的 bin，`chess.exe` 以 `0xC0000135`（`STATUS_DLL_NOT_FOUND`）秒退 —— 表现出来
> 只是"窗口没起来"，很容易被当成应用启动崩了（`ctest` 那边的 `test_match` 也是同一个
> 坑，用 `ENVIRONMENT_MODIFICATION` 修的）。脚本现在会自己找 `C:\Qt\*\msvc*\bin`
> 并前置到 `PATH`，同时把固定的 6 s `sleep` 换成轮询 `MainWindowHandle`（最多 30 s）。
> **(2)** 校验选择是否生效原来用 `-like "*$want*"`，而 `"SAC+AZ"` 是 `"SAC+AZ-MoE"`
> 的**前缀** —— 想选 SAC+AZ、实际选中带 MoE 的那个时也会 PASS。已改成按完整词匹配
> （`(^|\s)…(\s|$)`）。

`探索 2 步` 不是 bug：这个标签是**逐手刷新**的，显示的是最后一手的信息。最后一手的前
一步棋已经把局面推到接近将杀，探索 rollout 从那里出发 2 步就终局了。


### 11.5 实测（`test_sacaz`，94 条断言全过，约 21 s）

```
[1] 掩码 softmax 雅可比 vs 有限差分: 最大相对误差 1.161e-04
[2] 初始局面合法走法 44; 访问分布 sum=1.0000, 非零 38 个
    64 次模拟的单步决策耗时 ≈ 3 ms          <- 网络小, 搜索很便宜
[3] H=1.3834, V(α=0)=-0.11005, V(α=0.5)=0.58165, 差=0.69170 = α·H  (恒等式精确成立)
[4] Q(s,10): -0.14365 -> 0.49225 (目标 0.5);  |误差| 0.64365 -> 0.00775
    alpha 0.2000 -> 0.2544 (自动调节), learnSteps=60
[5] 连续 5 次探索后棋盘逐字段不变; 回放池 0 -> 80
[6] save/load 往返: 策略最大差 2.235e-08
[7] 自对弈链路: 回放 24 条, learnSteps=5
[8] 主干代价: MLP 0.013 ms/次 vs MOE@1260 168.6 ms/次 (见 §11.4)
[9] head 数/稀疏路由/专家代价 (信息): 1 head 是 15 heads 的 5.4 倍
[10] 骨干扫描 (真实 agent 路径, 四种骨干各建网/决策/探索/学习/存取):
     MLP              专家数=0 topK=0  64 模拟 =   4.4 ms  (0.068 ms/模拟)
     稀疏MoE(MLP专家) 专家数=8 topK=2  32 模拟 =  11.3 ms  (0.354 ms/模拟)
     稀疏MoE(TB专家)  专家数=4 topK=1   4 模拟 =  41.5 ms  (10.376 ms/模拟)
     稠密MoE(TB专家)  专家数=4 topK=4   4 模拟 = 172.0 ms  (42.994 ms/模拟)
     -> 同样 28.7 M 参数, 稀疏路由比全算快 4.14 倍
        save/load 往返 (MLP / 稀疏MoE-MLP专家): 策略最大差 0.000e+00,
                                                  Q 值最大差 ~8e-07 (文本权重的精度)
```

> TB 专家那两种骨干**不在**这里做存取往返：它们各 28.7 M 参数，而权重是十进制文本存的，
> 存一次就是几百 MB，会让整个测试从 21 s 涨到 181 s。序列化顺序的风险改用**小
> d_model** 在 `test_sparse_moe` 里查（那里两种专家的往返都覆盖了，见 §11.4.2 (3)）。

另外还多了一个独立的测试目标 **`test_sparse_moe`**（67 条断言，约 1.4 s），专门盯
`rl/sparse_moe.hpp`：稀疏不变量（未选中的专家改权重输出逐位不变）、与上游 `MOE` 的
等价性（前向/门控梯度差都是 0.000e+00）、反向的有限差分、辅助损失的梯度与纠偏方向、
使用计数与权重往返。细节见 §11.4.2。

以及一个**不进 ctest** 的基准目标 `bench_moe`（等时对弈 A/B/C/D，用法见 §11.4.2 (5)(6)）：
跑真实对局、依赖随机开局、TB 骨干一步几十毫秒，放进 ctest 既慢又会偶发失败。

另外 `test_match` 里加了一条 **集成断言**（SAC+AZ vs Alpha-Beta 打 1 局 / 4 手），
它覆盖的是 `ChessBoard::aiThinkForAgent` 的 `AGENT_SACAZ` 分支 —— 也就是 GUI 走的
同一条代码路径。

**真界面端到端**（`tools/verify_match_ui.ps1`，走 Windows UI Automation）：

```
A side agent = SAC+MCTS+AlphaZero (最大熵搜索)
B side agent = Alpha-Beta Pruning (深度=4)
result label = SAC+AZ 0 : 1 Alpha-Beta  共 1 局 / 12 手
   AI思考时间: 169ms
   探索+预训练: SAC+AZ 探索 64 步, 训练 1 次 (池 384, critic loss 0.3979, alpha 0.212)
A side in result = True (want 'SAC+AZ')
B side in result = True (want 'Alpha-Beta')
RESULT: PASS
```

即"界面里选得到、走得出来、走子前真的在探索训练（池从 0 涨到 384）、结果归属正确"。

> 这个脚本里踩过一个坑值得记下来：Qt 的 `QComboBox` 在 UIA 里 **Name 是空的**，
> 弹窗项的 `SelectionItemPattern.Select()` **会静默失败**。第一版脚本因此"选中了
> SAC+AZ"却实际跑的是默认对手（Alpha-Beta vs EVAB），还报了 PASS —— 典型的假通过。
> 现在改成用鼠标点选项，并且**用对弈结果标签里的 agent 名字反查选择是否生效**，
> 不匹配就 FAIL。

**棋力说明**：网络是随机初始化的，未训练时棋力很弱（上面那局 12 手就被 Alpha-Beta
将死了）。所有 RL agent 都是这样（见 §7）。这个 agent 的价值在于**链路完整且可训练**：
搜索给策略目标、回放反复利用稀疏的终局奖励、α 自动调节探索。要提升棋力需要按 §9
的 arena 跑足够的自对弈或 `trainSelfPlay`，并用预训=0 的对照验证。

由于单步只要 3 ms（64 模拟），GUI 里给的是 **256 次模拟**（约 12 ms），成本线性。
---

## 12. "多少参数量才能覆盖象棋求解空间"（理论分析）

这个问题单独写了一份文档：**[`docs/xiangqi_capacity.md`](xiangqi_capacity.md)**。

结论提要（详细推导与数据出处见该文档）：

| 读法 | 需要多少 | 可行性 |
|------|---------:|--------|
| A. **记忆式覆盖**（每个局面都记住最优走法） | **~10⁴⁰ 参数 ≈ 7×10²⁷ TB** | ❌ 物理上不可能（是全球数据总量的 10¹⁶ 倍；按 Landauer 极限"写一遍"要全球年发电量的 1/4；以 10⁸ 局面/秒标注完要 10¹⁴ 倍宇宙年龄） |
| B. **泛化 + 搜索**（有限参数 + 尽量多模拟） | **10⁶~10⁷ 参数** | ✅ 象棋引擎（NNUE 类）的实际配置 |

关键的一句：**参数量每 ×10，每步能跑的模拟次数就 ÷10，而模拟次数本身就是棋力。**
本工程实测的成本曲线（§11.4.2）与世界最强象棋引擎的配置（小网络 + 深搜索）指向同一个
结论：本工程的优化方向是**样本效率与搜索质量**，不是继续加参数。

顺带回答"要不要在本工程里试 10⁹ 参数"：按实测比例外推，10⁹ 参数的骨干一次前向约
120 ms，`learnBatch(32)` 约 31 s（≈1 样本/秒）；即使假设"每个参数只喂一个样本"这种
极度乐观的预算，也要 10⁹ 秒 ≈ 31 年。参数量不是这个工程的瓶颈路线。

---

## 13. 界面上的训练/对弈可视化（本轮新增）

对弈与训练原来在界面上几乎是"黑盒"：只有一行结果文字，曲线什么都没有。这一节记录新增
的内容与它们的验证方式。

### 13.1 实时比分 + 逐局明细列表

- **实时比分**：`ChessBoard::matchScoreChanged(QString)` 每打完一局 emit 一次，界面右侧的
  `scoreLabel` 立刻更新成 `A 2 : 1 B   和 1   (3/10 局)`。原来只有整场结束时弹一个
  **模态**结果框 —— 长对弈（几十局）中途完全看不出比分。
- **逐局明细列表**：`QListWidget`（`gameListWidget`）一局一行地追加，赛后常驻可滚动查看，
  每行形如
  `第 2 局: 红=EVAB 黑=SAC+AZ-MoE -> EVAB 胜  (52 手)  奖励 A=-1.00 B=4.10`。
  模态结果框去掉了（它的内容现在都在列表里 + 顶部的 summary 标签里），只保留"存权重"
  的询问。
- 每局明细里的**奖励**字段来自 13.3 的同一份采样，所以"这一局谁赚了多少子力"在列表里
  也能直接对照曲线。

### 13.2 两条曲线（`src/metricsview.h/.cpp`，自绘，不引入 Qt Charts）

| 曲线 | 数据源 | 说明 |
|------|--------|------|
| **训练损失** | `ChessBoard::trainLossSample`（走子前的在线训练 + 后台训练每轮各一次） | 每个上报损失的 agent **单独一条**（不同 agent 的损失尺度不可比），图例里有"最新 / 均值" |
| **环境奖励** | `ChessBoard::matchRewardProgress`（**每手**）+ `gameRewardSample`（每局结束补一点） | 每场对弈两条（A / B），值是**本局累计**（走子方视角: 吃子 + 局末的 ±1），见 13.7 |

实现上的几个要点：

- 不依赖 Qt Charts（那要额外装一个 Qt 模块），`paintEvent` 里自绘：自动纵轴（1/2/5×10ⁿ
  的"好看"刻度）、0 线加粗、点多了自动变细、`n=` 样本数、图例带最新值与均值。
- **只保留最近 2000 个点**（`setWindow`）：后台训练会一直往里塞样本，不设上限就是内存泄漏。
- 横轴是**采样序号**而不是时间：训练是事件驱动的，时间轴没有意义。
- 曲线下面是"数字读数"标签（`损失 ... 最新 0.048, 均值 0.159, 51 点`）。这不只是为了
  好看：曲线是画出来的，UIA/无障碍工具读不到任何数字，有了这个标签**自动化脚本才能
  断言"曲线真的有数据"**（见 13.4）。
- 哪些 agent 上报损失：`AgentBase::getLastTrainLoss()`，默认返回 NaN（不上报）。目前
  **7 个可训练的 agent 全都上报**（PG / DQN / PPO+MCTS / DQN+MCTS / EVAB / SAC+AZ /
  SAC+AZ-MoE），只有 Alpha-Beta 与 MCTS 没有可训练参数、保持不上报（曲线里没有点，
  不画假线）。这条口径的来龙去脉（原来 PG/DQN 等漏报）见 14.5。
- 支持"清空曲线"与"导出 CSV"。

### 13.3 一个被这张曲线挖出来的真 bug：即时奖励的**符号反了**

做奖励曲线的时候顺手量了一下"这一步到底给了多少奖励"，于是发现：**所有 agent 的
`computeReward` 原来都写成黑方视角**。

```cpp
// 修复前 (dqn / pg / ppo+mcts / dqn+mcts / sacaz 五份拷贝都是这个写法)
return (color == Stone::COLOR_BLACK) ? reward : -reward;
```

配合 `Chess::moveForward` 的记账（`吃红子 += value`），实际效果是（探针实测）：

```
红方用炮白吃一个黑马  ->  moveForward 记账 = -0.30,  computeReward(红) = -0.30
黑方白吃一个红马      ->  computeReward(黑) = +0.30
```

也就是说：**红方吃子是"负奖励"**。而同一个 agent 的终局奖励（`SACAZAgent::resultValue`）
是走子方视角的 ±1，两者**方向相反** —— 对红方等于在教它"吃子是坏事"，只对黑方是对的。
（`EVABAgent::evaluateLeaf` 里 `(color == BLACK) ? MATE : -MATE` 是正确的走子方视角，
说明原本的意图就是走子方视角，`computeReward` 是写错了。另外 `dqnmcts_agent.cpp` 有一处
调用直接把颜色硬编码成 `COLOR_BLACK`，同样是这个错误约定的产物。）

修复：五份 `computeReward` 统一改成**走子方视角**（吃子永远是 +value），并把约定**钉**在
`test_match` 第 [2.6] 节：

```
红方吃 马 (value=0.30): moveForward 记账 = -0.300
computeReward(红) = +3.000
computeReward(黑) = +3.000     <- 两侧对称, 谁吃子谁拿正奖励
```

同时钉住了 `Chess::moveForward` 的"黑方视角记账"这个事实（它是棋盘层的记账约定，与 agent
的奖励不是一回事），界面上的奖励曲线在 `ChessBoard::playMatchGame` 里显式换算成走子方视角。

> 这个 bug 也解释了本项目 RL agent 一直"学不动"的一部分原因：红方拿到的是反的塑形信号。
> 修复之后红黑两侧的塑形信号与终局信号方向一致。

### 13.4 验证（`tools/verify_match_ui.ps1`，走 Windows UI Automation）

脚本新增了对这一节全部内容的断言，`-AIndex 8 -BIndex 6 -Games 2 -Full` 的实测输出：

```
score label = 最终比分: SAC+AZ-MoE 0 : 2 EVAB  共 2 局 / 103 手
   | 第 1 局: 红=SAC+AZ-MoE 黑=EVAB -> EVAB 胜  (51 手)  奖励 A=-1.00 B=4.40
   | 第 2 局: 红=EVAB 黑=SAC+AZ-MoE -> EVAB 胜  (52 手)  奖励 A=-1.00 B=4.10
loss readout   = 损失 SAC+MCTS+AlphaZero (最大熵搜索): 最新 0.047926, 均值 0.15906, 51 点
reward readout = 奖励 SAC+AZ-MoE: 最新 -1, 均值 -1, 2 点  |  EVAB: 最新 4.1, 均值 4.25, 2 点
list has per-game lines = True      score shows a ratio     = True
reward chart has points = True      loss readout present    = True
RESULT: PASS
```

> 注意：上面这段是**奖励改成"每手一个点"之前**的实录 —— 所以 `reward readout` 里只有
> "2 点"（一局一个点）。那正是 **13.7** 要修的现象：一局 51 手里曲线一直不动。同一条脚本
> 现在的输出里奖励点数是与手数同量级的（见 13.7 与 16.3 的实测）。

> 顺带记一个**看过取舍后被否掉**的方案：每手一个点带来"局内锯齿"（一局从 0 爬到终值、
> 下一局跳回 0），另一种视角是"只在每局结束时记一个终值"（传统的逐局学习曲线）。用户
> 明确选择保持现状（每手一个点），所以没做。真要做的话最省事的形式是给奖励曲线加一个
> **默认关闭**的复选框"只在局末记点"（`matchRewardProgress` 照旧发，界面按开关决定要不
> 要加点），不影响现有语义。

脚本这轮又踩了两个坑，都记在脚本注释里：

1. **窗口出现 ≠ 可以交互**：`startupLoad()` 完成前所有控件都是 disabled，这时给
   `gamesSpin` 设值会抛 `operation is not allowed on a nonenabled element`。改成轮询
   "开始对弈按钮 IsEnabled"。
2. **连续选两个下拉框**：第一个弹窗如果还没收起，第二次点击会打在仍然打开的弹窗上 ——
   实测把 A 方选成了弹窗里同一位置的另一个 agent（PPO+MCTS），**结果标签的校验把它抓
   住了**（这正是当初把"选择是否生效"改成看结果标签的意义）。现在收起弹窗后会轮询确认
   状态再继续。

### 13.5 沙漏的流动方向（`ThinkingIndicator`）

用户报的现象是"沙漏上半部沙子减少的方向是错的"。查下来的确如此，而且有两个独立的错：

1. **上半部的沙子是从下往上被吃掉的**。原实现把沙子区域画成"贴住腔体顶部、底边向上
   收"，于是随相位变化的其实是**底边**；真实沙漏的沙面是**从上往下**落的。
2. **下半部沙堆顶的半宽算错了**：用了 `hw·(1−drained)`，而按几何应当是 `hw·u/hh`。

现在两个腔体统一按"面积守恒"换算沙面高度：三角形腔体的截面积 `A(u) = hw·u²/hh`，所以
剩余比例 `f` 对应的高度是

```
u = hh·√f        (而不是线性 hh·f)
```

上半部剩 `f = 1−drained`、下半部积 `f = drained`，于是"漏下去的"和"堆起来的"在任何相位
面积都相等，沙面下降先快后慢 —— 这才是沙漏看起来该有的样子。

（这一条没有自动化断言"沙面往哪边走"：`tools/verify_thinking_ui.ps1` 只能采像素判断
"思考中有没有状态条"。改动本身是几何推导 + 人工核对。）

不过这个脚本这轮顺手修好了，它原来有三处脆弱点，都是被新界面和权重格式变化暴露出来的：

1. **没有 Qt 的 PATH 探测**：从普通 shell 启动时 `chess.exe` 以 `0xC0000135` 秒退，
   而脚本只会报 "chess board (tan background) not found on screen" —— 看起来像"界面没画
   棋盘"，其实是进程根本没起来。现在和 `verify_match_ui.ps1` 一样会自己把
   `C:\Qt\*\msvc*\bin` 前置到 `PATH`，并且在 `Start-Process` 之后检查 `HasExited`。
2. **固定 `sleep 6` 等启动**：现在启动要解析旧的十进制文本权重（16 MB 级），6 秒不一定
   够，而控件在 `startupLoad()` 完成前全是 disabled。改成轮询"第一个按钮 IsEnabled"。
3. **用 SendKeys（TAB + DOWN）选 agent**：这依赖焦点顺序，而新增曲线面板之后焦点顺序变了
   —— 于是它其实一直在跑默认的 Alpha-Beta（~150 ms），窗口太短就抓不到状态条。
   改成 UI Automation 点选（和另一个脚本同一套做法，与焦点无关，选不中会直接报错）。

修完之后的实测（`tools/verify_thinking_ui.ps1`，选 MCTS 800 次模拟）：

```
idle_strip_dark_pixels   = 0   (expect 0)
idle_ctrl_diff_pixels    = 0   (expect 0: no idle animation)
thinking_strip_frames    = 7 / 90   (expect > 0)
thinking_strip_max_dark  = 728 / 728
thinking_ctrl_animating  = 3 / 29   (expect > 0)
after_strip_dark_pixels  = 0  (expect 0)
RESULT: PASS
```

> 这个脚本**必须保持纯 ASCII 且不带 BOM**（Windows PowerShell 会把无 BOM 的 .ps1 按 ANSI
> 解码）。我试着在里面写中文注释和中文控件名，直接就解析失败了 —— 所以它的就绪判断改成
> "第一个 Button 是否 enabled"，不去匹配中文名字。`verify_match_ui.ps1` 正好相反：它里面
> 有中文字面量，**必须带 BOM**。

### 13.6 MoE-TB 骨干在界面上的实际体感
`SAC+AZ-MoE`（稀疏 MoE + TB 专家，16 次模拟）在默认"预训 64 步"下的实测耗时：
**平均 2.3 秒/手**（上面那两局共 103 手、累计 234.6 秒）。拆开看是 `16 × 10 ms`（搜索）
+ `64 × 3.5 ms`（探索时每步一次策略前向）+ 一次 `learnBatch(32)`（≈0.9 s）。
想要它更快就把"预训步数"调小（比如 16），或者调小 `chessboard.cpp` 里的 `SACAZ_MOE_SIMS`。

### 13.7 奖励曲线原来"每局一个点"，用户看到的是"不更新"

用户反馈：**"对弈时奖励曲线没有更新"**。查下来**信号没断、数据没错，是采样太稀**：

`gameRewardSample` 是**一局结束才发一次**。而一局有多长？实测 `SAC+AZ-MoE vs Alpha-Beta`
那局是 **276 手 / 620.8 秒**（10 分钟）。也就是说：按下"开始对弈"之后的十分钟里，奖励曲线
始终只有 **1 个点**——横轴不动、纵轴不动，看起来就是坏的。

修法（`ChessBoard::matchRewardProgress`，新增信号）：

| | 之前 | 现在 |
|---|---|---|
| 采样时机 | 每局结束 1 次 | **每手 1 次** + 局末再补 1 次 |
| 值 | 本局最终环境奖励（含 ±1） | 一手一个**本局累计**值；局末那一点含 ±1 |
| 一局 276 手时的点数 | 1 | 277 |
| 十分钟对局的观感 | 一直不动 | 每手都在走，吃子时台阶跳一下，局末再跳胜负 ±1 |

三条实现上的讲究：

- **A/B 的换算只做一次**。`playMatchGame` 多了一个 `bool aIsRed` 参数，把"红/黑两本账"
  换算成"A/B 两本账"后同时用于出参 `rewardA/rewardB` 和每手的进度信号；`matchAgents`
  不再自己算第二遍（以前那两个出参是**死参数**——传进去从来没被写过，调用方在外面
  重新算了一次，两处规则一旦不一致就是"曲线和逐局明细对不上"）。
- **锁外 emit**。奖励记账在 `QMutexLocker` 里，信号在锁外发（信号是队列投递到 GUI 线程
  的，持锁碰元对象系统没必要）。
- **局末那一点仍然单独发**：`gameRewardSample` 的值是"最后一个进度点 + 终局 ±1"，所以
  每一局的最后一个点会比它前面那个多出胜负那一份。曲线上看就是"局末跳一下"，这是**设计**，
  不是重复计数。

防回归钉在两处（都是"用户不盯着屏幕也能发现它又坏了"的形式）：

1. `test/test_match_main.cpp` 的 **[2.8]** 节：一局 12 手必须产生 ≥11 个进度点（"每局只 1 个
   点"就是 bug 复现）、进度点手号连续、`最后一个进度点 + 终局增量 == 局末采样`、A/B 增量
   互为相反数、两局增量之和 == 比分差。
2. `tools/verify_match_ui.ps1`：对局**进行中**每 700 ms 取一次奖励读数，要求点数在涨；
   并且赛后要求 `奖励点数 >= 总手数 - 局数`（一局 300 手却只有两三个点就会 FAIL）。

### 13.8 每跑一场就多挂两条线（奖励曲线"换一批线"没清干净）

13.7 之后，用户在一次 **100 局**对弈里读到的标签是这样的：

```
奖励(局内累计) SAC+AZ-MoE: 暂无  |  Alpha-Beta: 暂无  |
               SAC+AZ-MoE: 最新 0, 均值 0.049185, 最小 -1, 最大 0.8, 982 点  |
               Alpha-Beta: 最新 0.1, 均值 1.3437, 最小 0, 最大 4.5, 982 点
```

**四条线，前两条永远"暂无"。** 根因在"每场对弈开始重建两条奖励曲线"的写法：

```cpp
ui->rewardChart->clearData();                    // 只清**点**, 线还在
m_rewardSeriesA = ui->rewardChart->addSeries(agentA, ...);   // 于是每场往后**再挂两条**
m_rewardSeriesB = ui->rewardChart->addSeries(agentB, ...);
```

跑第 N 场时图上有 **2N 条**线（其中 2N−2 条是空的），读数标签里就是上面那串"暂无"，
导出的 CSV 也会多出几列同名空数据。

修法与钉子：

| | 之前 | 现在 |
|---|---|---|
| 换一批线 | `clearData()`（只清点）+ 追加两条 | **`removeAllSeries()`**（连名字/颜色/数据一起清）+ 两条 |
| "清空曲线"按钮 | 同上（`clearData` + `m_lossSeries.clear()`，于是下一次同名 agent 上报时会**再建一条同名线**） | 两个图都 `removeAllSeries()`，`m_lossSeries` 一并清空 |
| 面板上的静态说明 | "曲线: 训练损失 / 每局环境奖励" | "曲线: 训练损失 / 环境奖励(每手累计)" |

`CurveChart::clearData()` 的语义**保持不变**（只清点、保留线）—— 放大窗口的 `syncFromSource()`
就是靠"线还在、点被清空"来同步的。两个语义分开，别再拿一个当另一个用。

回归钉子在 `test/test_match_main.cpp` 的 **[2.9]**（`CurveChart` 是普通 QWidget，offscreen
下直接构造即可，为此把 `metricsview.cpp` 加进了 `test_match` 的源文件列表）：

```
[2.9] 奖励曲线换一批线: clearData 只清点, removeAllSeries 才清线
    readout = reward C: 最新 2, 均值 2, 最小 2, 最大 2, 1 点  |  D: 最新 3, ... 1 点
=== 71 项断言, 0 项失败 ===
```

它按顺序钉五件事：`clearData()` 后线还在、**只 clearData 再加两条会变成 4 条**（把 bug 的
形状写成断言）、`removeAllSeries()` 清到 0、换一批之后仍只有两条、读数里每个名字只出现一次
且没有"暂无"。写这个测试时还顺手暴露了自己一个错误假设：`sampleCount()` 返回的是
**各条线里最多的点数**（CSV 按它出行数）而不是总和，第一版断言写成 `== 2` 就正确地失败了。

单元测试守的是**控件语义**，"MainWindow 真的每场都换线"还得在真界面上守 ——
`tools/verify_match_ui.ps1` 因此加了 `-TwoMatches`（打完第一场后在同一进程里再打一场）：

```
reward series in readout = 2 条 (期望 2), 有空线 = False        <- 第一场
second match ran = True  (逐局明细 6 -> 11 行)                  <- 先证明第二场真的跑了
second match series = 2 条 (期望 2), 有空线 = False             <- C12 的端到端钉子
RESULT: PASS
```

第二场之后读数与第一场**逐字相同**（258 点、最新 2.4/2.6）不是巧合：AB 对 AB 是确定性的，
两场下的就是同样的两局。**"同一场读数没变"不能当成证据**，所以这里必须同时断言"逐局明细
行数涨了"——否则第二场根本没跑起来时，那条"只有两条线"也会假通过（读数还停在上一次）。

---

## 14. EVAB 为什么赢不了 AB（三个真 bug），以及损失曲线的补全

用户报了两个现象：**"evagent 与 abagent 多次对弈无法取胜"**、**"其他 agent 对弈时无法显示
训练损失，显示表示也是错的"**。查下来两边都有实锤，一共挖出三个 bug。

### 14.1 EVAB 的在线训练在跑，但训出来的网络**从来没被用过**

先回答"有没有累计梯度训练"：**有**。`exploreAndTrain` 每步滚 64 手、每 16 个样本累积一
次梯度再 `RMSProp`（4 步/手），探针里 `getExploreInfo` 也一直报"价值网络更新 1 次"。
但 `build/evab_probe2.cpp`（临时探针）量出来两件事：

```
网络在固定局面上的输出 = 0.998344        <- 饱和
20 次更新之后          = 0.998485  (Δ +1.4e-4)   <- 基本没动
|net-hand| 一直在 0.7~1.0 (几乎最大分歧)   blend 恒为 0.00
```

**根因 A：`iFcLayer` 的初始化没有按 fan_in 缩放。** 输入是 1260 维 one-hot（约 32 个 1），
第一层 pre-activation 的标准差 ≈ √(32/3) ≈ 3.3 → tanh 从一开始就饱和，网络输出恒为 ±1，
梯度 ≈ 0。这和 `sacazagent.cpp` 里 `scaleLayerInit` 解决的问题是同一个（那里早就修了，
EVAB 这条路径漏了）。

**根因 B：`blend` 的初值是 0，而且全工程没有任何地方改过它。** 于是 EVAB 在界面上永远只
用手工评估 —— `evagent.h` 里写着"blend 随训练爬到 1 时网络接管评估"，但那个"爬升"只存在
于 `test_evab_main.cpp` 的测试里，agent 本体没有实现。

修完之后（同一个探针）：

```
网络在固定局面上的输出 = -0.028916        <- 健康
|net-hand| 0.0503 -> 0.0190 (20 轮)      <- 真的在拟合手工评估了
网络输出漂移 = -1.5e-3 ~ -5e-3 每轮       <- 真的在学
blend 0.00 -> 0.50 (回滚 1/20)            <- 阶梯上升, 门控有效
```

### 14.2 `blend` 阶梯 + 回滚（把测试里的策略搬进 agent）

- `blendStep = 0.05`：每次**没有把网络带偏**（`|net-hand|` 不恶化，沿用原有的回滚门）的
  在线更新之后爬一步。
- `blendMax = 0.3`：上限刻意取小。理由是实测出来的：**只要 blend > 0，每个叶子评估都要跑
  一次网络前向**（1260→48→1 ≈ 60k MAC，而手工评估只有几十次运算）—— depth 5 时
  blend=0 是 60 ms/步，blend=0.5 是 724~858 ms/步（**11~14 倍**）。评估变贵 = 搜索变浅，
  所以"让网络接管评估"在当前实现下是亏的。
- 回滚时 `blend` 退两步（`-2·blendStep`），坏网络会自然被压回 0。
- `exploreBudgetMs = 600` + `labelDepth` 上限 4：探索每步都要跑一次 `labelDepth` 层的
  negamax，界面上默认 64 步 —— 深度 5 的标签就是 64×90 ms ≈ 6 s/步，界面直接没法用。
  现在步数是"上限"，实际滚了多少步写进 `getExploreInfo()`。

### 14.3 EVAB 的搜索预算：原来给的是 AB 的深度，等于自废武功

`ChessBoard` 原来构造的是 `EVABAgent(env, 48, AB_DEPTH, 0)` —— 和 ABAgent 同一个深度。
实测本机初始局面：

| 引擎 | depth 3 | depth 4 | depth 5 | depth 6 |
|------|--------:|--------:|--------:|--------:|
| EVAB | 24 ms | 90 ms | 188 ms | 890 ms |
| AB   | 33 ms | 89 ms | **1873 ms** | — |

**同一深度 EVAB 快一个数量级**（TT + 迭代加深 + 排序 + 静态搜索），也就是它用 AB 深度 4 的
时间可以搜到深度 5。改成 `EVAB_DEPTH = 6` + `EVAB_BUDGET_MS = 800`（用迭代加深 + 时间上限，
让"评估贵就搜浅一点"自动发生）。

### 14.4 修完之后到底能不能赢（6 局 × 3 配置，每局交换先后手）

`build/evab_arena.cpp`（临时探针，与界面同一条调用序列：每步先 `exploreAndTrain(64)` 再决策）：

| 配置 | 胜 | 和 | 负 | EVAB ms/步 |
|------|--:|--:|--:|----------:|
| A 同深度（4）+ 学习评估 | 2 | 2 | 2 | 174 |
| B 深一层（5）+ 学习评估 | **1** | **5** | **0** | 796 |
| C 深一层（5）+ **纯手工评估**（blend 钉 0） | **0** | **6** | 0 | 70 |

三个结论：

1. **EVAB 现在能赢了**（修复前：探针里 0 胜，界面上也是"无法取胜"）。
2. **赢棋来自"学习评估"，不只来自深度**：C 组（纯手工评估、同样深度）6 局全和 —— 因为
   两边用的是同一个评估，走法几乎一样。有学习评估参与时才会出现胜负。
3. **代价要如实说**：B 组的 796 ms/步 是 C 组的 11 倍，差距全在被拉去评估叶子的网络上。
   所以 blend 上限压到 0.3、搜索用时间上限；界面里 EVAB 大约 0.3~1.4 s/步。

> 样本量说明：每配置只有 6 局，单局偶然性足以翻转结论（§9.1 已经吃过一次亏）。
> 这里能下的结论是"**机制通了、能赢棋了**"，不是"EVAB 比 AB 强多少"。
> 要谈后者需要几十局量级 + 等时间预算的对照。

### 14.5 损失曲线：其他 agent 也能上报了 + 读数公式修对

**(a) 哪些 agent 上报什么**

| agent | 上报的量 | 出处 |
|-------|----------|------|
| SAC+AZ | critic 的 MSE（只对实际走过的一步回归） | `SACAZAgent::learnBatch` |
| SAC+AZ (MoE) | 同上 | 同上 |
| DQN / DQN+MCTS | **平均平方 TD 误差** | `RL::DQN::lastLoss`（在 `experienceReplay` 里累加，`learn()` 收尾平均） |
| PPO+MCTS | **critic 的价值 MSE** | `RL::PPO::lastLoss`（`trainStep` 里算；actor 的交叉熵放在 `lastActorLoss`） |
| Policy Gradient | **策略梯度损失** `-Σ A·log π(a|s)` 的批均值 | `RL::DPG::lastLoss`（`reinforce1` 里累加） |
| EVAB | 在线训练的**平均绝对误差** `|预测 − 标签|` | `EVABAgent::exploreAndTrain`（**这轮才补上**：以前只有离线 `trainBatch` 会写，而界面走的是在线路径，所以界面上 EVAB 的损失曲线一直是空的） |

**逐个 agent 的回归测试**（`test_match` 第 [2.7] 节：每个 agent 跑一小局并数 `trainLossSample`
信号 —— 以前是 7 个可训练 agent 里只有 2 个会上报）：

```
Alpha-Beta       信号 0 次   (预期: 没有可训练参数)
MCTS             信号 0 次   (预期)
Policy Gradient  信号 5 次     <- 修了: PG 走的是 DPG::reinforce(), 而 lastLoss 只加在 reinforce1() 里
DQN              信号 5 次     <- 修了: DQN 的 learn() 在"回放池 < batchSize"时不更新, 池没攒够就不上报
PPO+MCTS         信号 5 次     <- 修了: RL::PPO::trainStep 里补了 lastLoss
DQN+MCTS         信号 5 次     <- 同 DQN
EVAB             信号 5 次     <- 修了: 在线训练路径补了 m_lastLoss
SAC+AZ           信号 5 次     <- 修了: 在线那次 learnBatch 原来用固定 batchSize(32), 池不足时直接返回
SAC+AZ-MoE       信号 5 次     <- 同上
```

两个"看起来像 bug、其实是正确行为"的点，测试里写明了：损失是**按回放池大小门控**的
（池不够就不训练、也就没有损失），所以测试给的是"预训 32 步 + 10 手"而不是一小局。

不上报的（`AgentBase::getLastTrainLoss()` 默认 NaN，曲线直接丢弃这个点、不画假线）：
Alpha-Beta / MCTS（没有可训练参数）。

**(b) "显示表示也是错的"：窗口淘汰点时统计量没跟着走**

`CurveChart::addPoint` 在点数超过窗口（2000）时只 `remove` 了最老的**数据**，没有维护
`s.sum / s.mn / s.mx`。后果是：

* "均值" = 全部历史之和 ÷ 窗口内点数 → **越跑越大**，完全失真；
* "最小/最大" 还留着已经被淘汰的极值 → 纵轴范围也是错的。

现在淘汰时把被移除的值从 `sum` 里减掉，并在淘汰发生时重算 `mn/mx`（窗口只有 2000 点，
重算是微秒级）。读数同时把格式化统一到 `CurveChart::readoutText()` —— 以前界面标签、
放大窗口、无障碍描述三处各写一份，口径不一致。

### 14.6 双击曲线放大（用户要求的独立放大窗口）

- `CurveChart` 双击 → `doubleClicked()` 信号 → `MainWindow::openLargeChart()`。
- `CurveChartDialog`：900×560 的可缩放独立窗口（`Qt::Window`，可拖到别的屏幕），里面是同一
  个 `CurveChart` + 一行数字读数 + 关闭按钮。非模态（可以一边跑对弈一边看），
  `WA_DeleteOnClose` 自动析构。
- 数据同步用信号而不是拷贝：源控件每加一个点 `emit dataChanged()`，对话框整体同步一次
  （条数/名字/颜色/数据都同步 —— 每场对弈开始时 A/B 两条会重命名，不同步的话放大窗口会
  一直显示上一场的名字）。
- 同一个源只保留一个窗口：已经开着就抬到前面（重复双击不会开出一堆窗口）。
- 图右上角画出"n=51  双击放大"，让人知道这个操作存在。

**验证**（`tools/verify_match_ui.ps1`，PASS）：

```
double-click opens a large chart window = True
   large window readout = 读数 PPO+MCTS (AlphaZero): 最新 0, 均值 0, ... 45 点  |  Deep Q-Network (DQN): ... 46 点
   large window mirrors the source label = True
```

两个坑记在脚本注释里：**(1)** 放大窗口的父窗口是主窗口（`QDialog(parent)`），Windows 把它
当"被拥有的顶层窗口"，UIA 挂在**主窗口下面**而不是桌面根节点的 `Children` 里 —— 必须用
`Descendants` 找（实测 `Children=False, Descendants=True`）；**(2)** 对弈结束后
`offerSaveWeights()` 会弹一个**模态**文件对话框，它会吃掉之后所有鼠标事件，所以双击检查
之前要先按 ESC 把挂起的对话框关掉。
### 14.7 曲线分线键必须是"稳定的名字"（第六个 bug，也是"显示表示是错的"的一部分）

损失曲线是**按 agent 分线**的，键是 `agent->getName()`。而 EVAB 原来的名字里带着**当前的
blend**：

```
EVAB (learned eval, depth=6, blend=0.05)
EVAB (learned eval, depth=6, blend=0.10)
EVAB (learned eval, depth=6, blend=0.00)
...
```

blend 一变就多一条曲线 —— 界面实测一局下来同一张图上有 **6 条 "EVAB ..."**，图例挤成一团、
每条只有几个点，读数也会列出一长串。修法两条：

1. `getName()` 只留**结构信息**（`EVAB (learned eval, depth=6)`），会变的状态放
   `getExploreInfo()`（那里本来就有 `blend 0.00->0.05`）。
2. `MainWindow::lossSeriesFor()` 加一道防线：曲线超过 6 条就 `qWarning` 一次，
   提示"是不是又有 agent 把会变的状态写进名字了" —— 这类 bug 不该靠眼睛发现。

### 14.8 界面上最终的验证结果

`tools/verify_match_ui.ps1 -AIndex 6 -BIndex 0 -Games 2 -Full`（EVAB vs Alpha-Beta，真界面）：

```
result label = 最终比分: EVAB 1 : 1 Alpha-Beta  共 2 局 / 190 手
loss readout   = 损失 EVAB (learned eval, depth=6): 最新 0.59738, 均值 0.72552,
                        最小 0.42417, 最大 1.1335, 95 点     <- 一条曲线, 95 个样本
large window readout = 读数 EVAB (learned eval, depth=6): ... 95 点
large window mirrors the source label = True
RESULT: PASS
```

也就是：**EVAB 在真界面上能赢 AB 了**（1 胜 1 和），损失曲线有数据且只有一条，
双击放大窗口与源控件逐字一致。

### 14.9 一个还没定位的偶发现象（如实记录）

上面那局的比分里带着 `[1 次无效走法已兜底]` —— agent 偶尔会返回一个 `valid == false`
的走法。频率大约 **每 200 次 agent 决策 1 次**（几次长对局里各出现 1 次），arena 的兜底
逻辑（改用第一个合法走法并计数）保证了它不影响胜负归属，但根因还没查出来：

* EVAB 的 `search()` 只有在 `chess.sample()` 返回空（它认为无合法走法）时才会返回
  无效 Step，而 arena 在调用前刚用 `chess.getResult(turn)` 确认过对局未结束；
* 也不排除来自 ABAgent（arena 的 stderr 会打印是哪个 agent，但这几次没抓到）。

下一步的做法：给两个 agent 的"无合法走法"分支都加上"把 `getResult` 与局面的 Zobrist 键
打进 stderr"的诊断，等它下次出现。**在此之前不假装它已经解决。**
---

## 15. 载入/保存模型权重时的"请稍候"沙漏（`src/busydialog.h`）

用户要求：**加载模型时弹出沙漏控件等待**。做法与实测如下。

### 15.1 为什么值得做（先量了一下到底有多慢）

权重读写不是瞬间的事，实测（本机）：

| 场景 | 耗时 |
|------|------|
| 启动预加载 7 组权重（含 SAC+AZ 稀疏 MoE 那 3 个文件） | **~19 s** |
| 其中 SAC+AZ（稀疏 MoE + TB 专家） | 3 × **146 MB**（28.7 M 参数 × 3 网络），约 15 s |
| DQN+MCTS | 8.6 MB |
| EVAB | 0.3 MB |

原来这段时间界面只有一行 `正在加载...` + 所有控件变灰 —— 19 秒里看起来就是卡死。
（顺带发现：这 3 个文件是"每个 float 5.34 字节"的新格式；旧格式的同一个模型还要大 **1.8 倍**。）

### 15.2 实现

* **`src/busydialog.h/.cpp`**：一个 `QDialog`，里面**直接复用 `ThinkingIndicator`**
  （沙漏 + 旋转粒子 + 呼吸灯 + 实时耗时）—— 它本来就是"正在等"的统一视觉语言，
  用户不需要学第二套符号；再加一行"正在读写模型权重，完成后会自动关闭"。
* **`ChessBoard` 只报告状态**，不碰界面：`busyStarted(标题, 说明)` / `busyMessage(说明)` /
  `busyFinished()`。这些信号是从工作线程 emit 的，队列投递到 GUI 线程，所以
  `MainWindow` 里的处理是安全的。启动加载会逐个文件报到名字：

  ```
  正在载入 PPO+MCTS 权重… / 正在载入 EVAB 权重… / 正在载入 DQN+MCTS 权重… (文件较大，可能要几秒)
  ```
* **保存也走同一套**（`saveCurrentAgentModel` 里用 RAII 的 `BusyGuard`，保证**任何返回
  路径**都会发 `busyFinished`；以前那种"每个 return 前手工收尾"漏一个分支就是弹窗永远挂着）。
* **保存放在工作线程**：在 GUI 线程里同步写盘会把事件循环堵住，弹窗虽然显示出来但画面
  是**冻结**的（画不出下一帧、定时器也不跑）。`offerSaveWeights` 因此改成
  `m_saveThread` + 队列回调，析构函数里 join（和另外两个线程同一套生命周期规则）。
* **交互决定**：加载中**不可关闭**（`closeEvent`/`reject` 都忽略 —— 中途关掉会让人以为
  操作被取消了，其实后台线程还在写）、应用级模态、用 `show()` 而不是 `exec()`
  （加载在后台线程，GUI 事件循环要继续跑沙漏才会动）。

### 15.3 顺带修掉的一处不一致：两条分支行为不一样

查这个功能时发现 `aiThinkForAgent` 的 `AGENT_SACAZ_MOE` 分支**只创建对象、不载权重**，
而 `aiThink` 的分支会载 —— 两条路径行为不一致（对弈里用到它时跑的是随机初始化的网络）。
当时改成"两条都懒加载、都弹沙漏"；后来随着 15.4 改成启动时全量预加载，这两条分支里的
载入成了兜底路径（见 15.5 末段的说明），但仍然保持"建了对象就把权重载上"的一致行为。

### 15.4 启动时加载所有模型（用户要求），以及为它做的加载提速

用户后来要求"**程序启动时加载所有模型**"（这样选中任何一个 agent 都是立刻能下棋，
不会在第一次走子时卡十几秒）。于是稀疏 MoE 变体也回到启动时预加载 —— 启动从 1.7 s
回到 19 s。既然这个开销躲不掉，就把它压下去：

先量清楚再优化（`build/probe_load.cpp`，对真实的 146 MB actor 文件）：

| 环节 | 优化前 | 优化后 |
|------|-------:|-------:|
| 整块读文件 | 136 ms (1075 MB/s) | 136 ms |
| `ifstream` + `getline` 逐行 | 1056 ms (143 MB/s) | — |
| base64 解码 (纯算) | 1110 ms (99 MB/s) | **362 ms (302 MB/s)** |

三处改动（都在 `rl/tensor.hpp` / `rl/net.hpp`，行为不变，`test_weights` 逐比特往返照旧）：

1. **解码查表替代分支链**：每个 base64 字符原来要过 6 个比较，换成 256 项表
   （非法字符 = −1，`=` = 64），解码从 99 → **302 MB/s**。
2. **流式解码 + 增量 CRC**：`Net::load` 的"先校验一遍"过去是
   `std::string(ptr, n)` 拷一份 34 MB → `vector.push_back` 再分配 34 MB → 扫一遍算 CRC
   → 再 `memcpy` 进张量，而且**整套做两遍**。现在一次遍历里同时完成
   "合法性 + 长度 + 增量校验和 + 直接写进张量的存储"。
3. **预校验在内存里做**：文件先整块读进内存（1075 MB/s），再用 `memchr`/`find` 切行
   校验；省掉一次磁盘读和一整轮 143 MB/s 的逐行流式读取。

结果（同一台机器，启动时七组权重逐个计时）：

```
[weights] PG:            245 -> 171 ms
[weights] DQN:           339 -> 151 ms
[weights] EVAB:           12 ->   5 ms
[weights] DQN+MCTS:      343 -> 152 ms
[weights] SAC+AZ:         52 ->  23 ms
[weights] SAC+AZ-MoE 建网(5 x 28.7M 参数): 1537 ms   <- 分配 + 随机初始化, 与读盘无关
[weights] SAC+AZ-MoE 读权重(3 x 146 MB):  14873 -> 6003 ms
启动到界面可用:           19.2 s -> 10.1 s
```

顺带说一句这次改动的"自证"：第一版忘了在预校验里**跳过 v2 的文件头**，于是每个文件都在
"第 1 个张量校验失败"——而这个失败又恰好被"载入失败不改动网络"那条保证掩盖住了（网络是
干净的，只是什么都没载入）。`test_weights` 立刻把它抓了下来（44 条断言 5 条失败），修好
再测 44/44 全过。

剩下没做的：`Net::load` 的"真正载入"那一遍仍然走 `ifstream + getline`（146 MB 约 1 s/文件，
三个文件约 3 s）。要消掉它需要把 `iLayer::read/write` 的签名从 `std::ifstream&` 换成
`std::istream&`，然后用内存里的 `std::istringstream` 读 —— 纯机械改动（编译器会把所有
override 都找出来），预计还能省 2~3 s。

### 15.5 曾经试过又撤销的中间方案：懒加载

为了启动速度，这一版一度把稀疏 MoE 变体改成**懒加载**（第一次用到时才读，启动只要 1.7 s），
懒创建分支的代码还在（`aiThink` / `aiThinkForAgent` 里，仍然会载权重并弹沙漏）。用户明确
要求"启动时加载所有模型"之后，这条又改回预加载 —— 因为懒加载的代价是**第一次选中它时卡十几
秒**，而"所有模型都就绪"对使用体验更重要。

需要如实说明一点：改回预加载之后，那两个懒创建分支里的 `loadModel()` **实际上不可达**了 ——
权重文件存在时 `startupLoad()` 既登记路径也把它读进内存（`m_sfSACAZMoe != nullptr`），
文件不存在时 `s_weightPaths` 里根本没有这一项、分支里也不会去读。它们现在的价值只剩"兜底":
万一以后有人把预加载去掉，至少不会空指针崩。这也是 `tools/verify_busy_lazy.ps1`（断言"首次
使用**会**弹沙漏"）必须被替换掉的原因 —— 它的断言在新行为下**只能失败**，留着它就是留一个
永远红着的测试（见 15.6）。

两种取法的实测对比：

| 方案 | 启动到可用 | 第一次使用稀疏 MoE 变体 |
|------|-----------:|------------------------:|
| 预加载（当前） | 10.1 s | 0（已经就绪） |
| 懒加载（曾用） | 1.7 s | +15 s（弹沙漏） |

### 15.6 验证（三个脚本，都 PASS）

`tools/verify_busy_ui.ps1`（ASCII-only，窗口名用码位拼）：启动 → 每 250 ms 枚举本进程的
**可见**顶层窗口 → 断言"沙漏窗出现过"且"界面可用时它已经不在了"。

```
# 修懒加载之前（启动真的要 19 秒）
busy window appeared at 3.2s: '正在载入'
busy window closed at 18.9s
main UI became ready at 19s
busy window still open after ready = False      -> RESULT: PASS

# 改成懒加载之后
main UI became ready at 1.7s
busy window still open after ready = False      -> RESULT: PASS
```

`tools/verify_eager_load.ps1`（真界面：启动 → 选 A 方 = 稀疏 MoE 变体 → 开局 → 盯沙漏窗）
—— 它**替换**了原来的 `verify_busy_lazy.ps1`（那个断言"首次使用会弹沙漏"，在全量预加载下
只能失败）。新脚本一次验三件事：

```
startup hourglass appeared at 1.8s: '正在载入'
startup hourglass closed at 10.8s
ui ready at 10.9s = True                      <- [1] 启动沙漏覆盖了全部 438 MB 的加载
first-use hourglass appeared = False          <- [2] 首次使用**不再**加载 (全量预加载的钉子)
reward samples during the watch = 2 -> 20     <- [3] 对局确实在推进 (不是"没开始所以没弹窗")
match progressed while watching = True
RESULT: PASS
```

[2] 是这条要求的**核心断言**：把它改回懒加载（或者让预加载漏掉某一组），这里立刻会看到
沙漏重新出现。[3] 是防止 [2] 变成"假通过"——对局根本没跑起来时也不会有沙漏。

脚本自己踩的坑，记在里面：**(1)** 判断"界面可用"不能取"第一个 enabled 的 Button"
（曲线面板那两个按钮一开始就是 enabled 的，于是脚本会在权重还在加载时谎报 ready —— 必须
按名字找**开始**按钮）；**(2)** 只认**可见**顶层窗口，Qt 会创建名字非空但在屏幕外的辅助
窗口，把它们算进来"弹窗还在"就永远为真；**(3)** PowerShell 里调函数**不能带空括号**
（`RewardSamples()` 会报 `An expression was expected after '('`），这与 C/JS 的直觉相反，
第一版就是这么挂的。
---

## 16. 对弈结束后静默保存权重 + 程序图标

### 16.1 静默保存（用户要求：不再弹窗口）

原来是"对弈结束 → 弹一个**另存为**文件对话框问你存哪里 → 存完再弹一个**成功**消息框"。
两个窗口都是打断，而且它们的默认文件名（`evab_agent_weights.dat` 之类）与
`ChessBoard::shutdownSave()` 用的正式路径（`weights/evab_agent.dat`）**是两套名字** ——
用户随手一点就可能存到一个启动时不会读的地方（"存到 A、读 B"的静默失效）。

现在：

| | 以前 | 现在 |
|---|---|---|
| 存哪里 | 弹框问你 | **标准路径**：`ChessBoard::defaultWeightPath()`（与启动加载、退出保存同一份，只有这一个来源） |
| 什么时候 | 打完一场问你一次 | 每场结束后自动存**这场用过的**可训练 agent |
| 线程 | GUI 线程同步写（界面冻结） | `m_saveThread` 后台写，界面不卡 |
| 反馈 | 模态消息框 | 逐局明细列表里加一行：`—— 已静默保存权重: DQN -> weights/dqn_agent.dat ——`；失败也会写 `[失败]`，但**不挡操作** |
再加一道"延迟显示"：`busyStarted` 不会立刻弹沙漏，而是起一个 300 ms 的单发定时器，
只有到点还没结束才真的显示 —— 几 MB 的权重写几十毫秒就完了，用户**什么都看不到**；
几百 MB 的那种（稀疏 MoE 3×146 MB）才会看到沙漏。这样"静默保存"和"慢操作要有反馈"
两件事同时成立。（启动加载是例外：构造函数里已经直接亮起来了，因为它是秒级以上的等待，
用户需要立刻知道"程序在启动"。）

顺带清掉的：`MainWindow::agentWeightFilename()`（那套"另存为"的默认名）整段删掉，
只留 `defaultWeightPath()` 一个来源；`shutdownSave()` 也从"七个 if"改成遍历同一张表，
不会再出现"某个 agent 忘了存"（EVAB 当年就是这么漏的）。

**再补一行耗时日志**：界面列表里只能看到"存完了"，看不到花了多久。稀疏 MoE 变体一次存
3×146 MB，哪天保存路径被改慢了，光看列表是发现不了的。所以后台线程里每个 agent 前后各取
一次 `steady_clock`，打一行：

```
[weights] 保存 SAC+MCTS+AlphaZero (稀疏MoE+TB专家): 1983 ms -> weights/sacaz_moe_agent
```

实测（`tools/verify_match_ui.ps1 -AIndex 8 -BIndex 0 -Games 1 -Full -LogFile ...` 跑出来的
真数）：438 MB 花了 **1983 ms**，约 221 MB/s。这个数**不要**当成"落盘时间"读：`Net::save()`
只做到 `file.flush()` + `rename()`，**没有** `fsync`/`FlushFileBuffers`，所以量到的是
"写进页缓存 + 改名"的开销，真正刷到盘上是内核随后做的（进程被杀也不影响，页缓存还在）。
要量落盘得在保存后自己 `fsync`，那是另一件事。

（和启动加载的 `[weights] PG: 167 ms` 是同一套前缀，肉眼搜 `[weights]` 就能把"读了多久 /
写了多久"全捞出来。）

### 16.2 程序图标

图标**是脚本生成的**，不是塞进来的二进制（`tools/make_app_icon.ps1`）：一枚木色棋盘底
上的红"帅"棋子。理由：图标改了能重做，而且二进制资源在 diff 里什么也看不出来、没法审。

两个地方各需要一份，**缺一个就是"窗口有图标、exe 文件没有"的半吊子状态**：

* **运行时窗口/任务栏**：`res.qrc` 里加 `app.png`（256×256），`main.cpp` 里
  `QApplication::setWindowIcon()` —— 应用级的，所有窗口（主窗口 / 请稍候弹窗 / 放大曲线
  窗口 / 消息框）自动继承。启动时打一行日志 `[icon] window icon ok: 256x256`：
  图标丢了属于"看起来没坏但就是不对"的问题，有日志就不用猜。
* **exe 文件图标**（资源管理器、任务栏固定）：`src/app.rc` + `src/app.ico`
  （7 个尺寸 16/24/32/48/64/128/256，PNG 负载）由 rc.exe 编进 PE 资源的 RT_GROUP_ICON。

验证（`tools/verify_app_icon.ps1`，33 项检查全过）：

```
[1] ICO container:  22730 字节, 7 个条目, 每个都是 PNG 且 IHDR 宽高与目录一致
[2] glyph rendering: 字体 Microsoft YaHei; 真字墨量 988 px / 40 行,
                     "缺字方框"对照 353 px -> 字体确实有这个字 (不是豆腐块)
[3] runtime load:    [icon] window icon ok: 256x256
[4] exe file icon:   ExtractAssociatedIcon -> 32x32, 其中红色 107 px、木色 582 px
                     (默认的 exe 图标是蓝白窗口, 所以这能证明 exe 带的是**我们的**图标)
RESULT: PASS
```

生成脚本里踩的坑也记在注释里：PowerShell 的数组 `+=` 会把 `byte[]` **拆开**逐个追加
（第一版写出来的 `.ico` 每个条目长度都是 1，整个文件只有 125 字节）—— 要用
`ArrayList.Add()`；以及 `return ,$bytes` 的逗号不能省。

### 16.3 验证脚本/验证流程自己踩的五个坑（都真实发生过，记下来免得再犯）

这些都不是被测代码的问题，而是"检查写得不对"或者"我根本没验到新代码"，而且**错误的
检查比没有检查更坏**：它会给出一个看起来很像结论的答案。

**(1) 后台保存还没写完就去读列表 → 假失败。**
静默保存是后台线程，稀疏 MoE 那 438 MB 要一两秒（实测 1983 ms）。第一版脚本在下棋结束的
一瞬间就去读列表，于是 `silent weight save line = False` —— 代码是对的，脚本错了。
现在改成**轮询等它出现**（最多 60 s，`All-ListItems()` 每 500 ms 重读一次）。

**(2) 启动加载的计时行被当成了保存的计时行 → 假通过。**
加了"顺手核对 `[weights] 保存 ... N ms`"之后，第一版只匹配 `[weights]` 和 `ms`。
启动时那七行 `[weights] PG: 167 ms` 两条都满足 —— 于是**对局根本没打完**（`TimeoutSec`
到点，还停在 1/2 局、按钮还是"停止"）也报 `save timing logged = True`。现在必须同时
命中"保存"这个词。教训：用日志做断言时，匹配的**必须是那条日志独有**的特征。

**(3) 改完脚本丢了 BOM → 报的错和真正原因毫无关系。**
`tools/verify_match_ui.ps1` 里有中文字面量（要拿去和界面文本比对），所以文件必须是
**UTF-8 带 BOM**。这次用补丁工具改了两处之后 BOM 被丢掉，Windows PowerShell 5.1 就按
ANSI 解码，中文串变成 `鐐?`、`闈欓粯淇濆瓨`，报出来的是
`Missing expression after ','`、`Unexpected token '}'` 这种**语法错**——看上去像"脚本被
改坏了"，实际上是编码。判据很简单，改完看一眼前三个字节：

```powershell
[System.IO.File]::ReadAllBytes("tools\verify_match_ui.ps1")[0..2]   # 要 EF BB BF
```

同理，`verify_thinking_ui.ps1` / `verify_busy_ui.ps1` / `verify_eager_load.ps1` 是**纯 ASCII**
（它们只需要比对数字和英文标签，中文窗口名一律用码位拼），本来就不需要 BOM —— 只要不往里加
中文，就永远不会踩到坑 (3)。

**(4) 拿"根本不该出现的东西"当失败 → 假失败（而且会把配置问题报成产品 bug）。**
跑 `-AIndex 0 -BIndex 0`（两边都是 Alpha-Beta）时，脚本报了三处 FAIL：没有
`—— 已静默保存权重 ——` 那一行、`loss readout = 损失 -`、放大窗口"数据对不上"。
但 Alpha-Beta 与 MCTS **没有可训练参数**，`saveWeightsAfterMatch` 会跳过它们、损失曲线
本来就不该有点 —— 三条都是**正确行为**。现在脚本一开始就算一个 `$expectSave`
（`AIndex/BIndex > 1` 才算有可训练 agent），不适用就打印"跳过 + 原因"，不判失败。

同一处还有个更隐蔽的版本：跳过分支写成 `if ($lossText -eq "-")`，而 `$lossText` 是**带
前缀**的（实际值是 `"损失 -"`），于是这个分支**永远是死代码**，跳过逻辑一次都没生效过。
教训：拿界面上的字符串做判断，先把已知前缀剥掉再比（`-replace "^损失\s*", ""`）。

**(5) 用 `Select-Object -First N` 看构建输出，把编译掐断了 → 我拿旧 exe 验证了一遍。**
这条最值得记：为了"只看前几行错误"，我把构建输出接进了
`Select-String ... | Select-Object -First 20`。PowerShell 的 `-First N` 一凑够 N 项就
**终止整条管道**（包括上游那个正在编译的进程），于是编译在中途被杀掉、`chess.exe`
根本没重新链接，而我以为"构建通过"，接着拿**旧二进制**跑了一遍界面验证，还认真分析了
为什么曲线不动（真实原因见坑 4 之前的 13.7：这就是"奖励曲线不更新"的复现）。

现在的做法：构建输出重定向到文件再读文件；并且**构建后一定看二进制的时间戳**：

```powershell
(Get-Item build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe).LastWriteTime
```

顺带一个相关坑：**应用开着的时候链接会失败**（`LNK1104`，exe 被占用）。这次就撞上了 ——
用户手动开着的那个窗口一直锁着 exe，链接失败被上面的管道问题掩盖，直到查时间戳才发现。

---

## 17. PPO 系列训练效率改造（P1–P7）与实测

本轮的目标不是棋力，是**同样的算力能产出/消化多少训练数据**。上一轮量出来一个很难看的事实：
一局自对弈的算力里"搜索"和"学习"几乎对半（80 次模拟的一步 17.7–22.4 ms ≈ 0.22–0.28 ms/模拟，
一次 `trainStep` 17.0–21.2 ms），而**优化器占了单步的 66%、前向只占 1%**
（见 `test_ppomcts` 的 compute budget 一节）。也就是说：每条样本单独跑一次全参数 RMSProp，
绝大部分算力花在"把 377 万参数全部更新一遍"上，而不是花在读局面上。P1–P7 逐条拆掉这个结构。

### 17.1 七个阶段各自改了什么

| 阶段 | 改动 | 文件 | 实测 |
|---|---|---|---|
| P1 | `RL::Random` 改线程安全（每条线程一份引擎 + 调用方指定编号播种） | `src/rl/util.hpp/.cpp` | 消除数据竞争；并行跑可复现 |
| P2 | 奖励/价值目标的**视角口径**全量审查 + 修 9 处不一致 | `stone.h` + PG/DQN/DQN+MCTS/EVAB | 见 17.2 |
| P3 | 梯度累积：攒批反向、一次优化器更新 | `src/rl/ppo.h/.cpp`、`src/rl/net.hpp` | 每样本 22.5 → 7.7 ms（**2.9×**） |
| P4 | 回放池 + 多 epoch：样本可反复用，池里只存稀疏策略目标 | `src/rl/ppo.h/.cpp`、`ppomcts_agent.*` | 128 次样本更新 2355 → 617 ms（**3.8×**） |
| P5 | 策略目标从 one-hot 改成**根的访问分布** π ∝ N | `ppomcts_agent.*` | 见 17.4 |
| P6 | 左右镜像数据增广（y → 8−y），样本量翻倍 | `ppomcts_agent.*` | 见 17.5 |
| P7 | PPO+MCTS **多线程分身**自对弈（worker 自对弈 + learner 更新 + 权重快照） | `src/ppo_selfplay_mt.hpp`（新）、`test/bench_ppo_mt_main.cpp`（新） | 见 17.6 |

四个新测试/基准（都在 `test_`，`bench_*` 刻意不进 ctest）：
`test_ppomcts` 的 [7] 符号、[8] 访问分布、[9] 回放路径、[10] 镜像增广；
`bench_ppo_mt` 量吞吐；`bench_ppo_vs_ab` 量棋力（见 §5.1）。

### 17.2 两套价值口径 —— 一个不报错、只让人学不动的错误（P2）

本工程同时存在两套状态编码，于是也存在**两套价值口径**：

* **老一代** PG / DQN / DQN+MCTS：`encodeState` 是"棋盘绝对坐标 + 黑子为正"（红子取负），
  网络表达的是**对黑方的价值** → 奖励必须是**黑方视角**。
* **新一代** PPOMCTS / SACAZ / EVAB：`encodeState` 是**规范视角**（轮到黑方时整盘左右镜像），
  网络表达的是**走子方**的价值 → 奖励直接是**走子方视角**。

而 `Chess::moveForward` 的 `totalReward`、各 agent 的 `computeReward`、`rolloutFromCurrent`
产出的**都是走子方视角**（吃子者为正）。上一次把 `computeReward` 从"黑方视角"改成"走子方视角"
是对的（这条口径被 `test_match` 的 [2.6] 钉住了），但**只有 `computeReward` 自己改了，消费端没跟着改**：
老一代三个 agent 的即时奖励是走子方视角，而同一批经验里的终局常量
`(gameResult == COLOR_BLACK) ? 1 : -1` 是黑方视角 —— **一条轨迹里两种口径混用**，
红方那一半样本的目标整体反号。

这类错误**不崩溃、不报错、连测试都不一定挂**，症状只是"学不动"，所以做了一次逐点审查，
结论落在 `stone.h` 的 `moverRewardToBlackFrame()` 两个重载上（一个吃颜色，一个吃 `Step::id`，
后者给 `rolloutFromCurrent` 的回调用 —— 那个回调拿得到 `Step`、拿不到"当前走子方"这个变量）。

修了 9 处：

| 文件 | 位置 | 处理 |
|---|---|---|
| `pgagent.cpp` | `exploreAndTrain` 的 `onTrans` | 按 `chosen.id` 翻符号（这条路径**没有**终局覆盖，反号是实打实的） |
| `dqnagent.cpp` | `trainSelfPlay` / `warmupFromCurrent` / `trainAfterMove` / `exploreAndTrain` | 同上；`trainAfterMove` 目前无调用方，一并修掉只是不留反例 |
| `dqnmcts_agent.cpp` | `trainSelfPlay` / `recordExperience` / `exploreAndTrain` | 同上 |
| `evagent.cpp` | `exploreAndTrain` 的手工评估项 | 用了 `next`，而状态与 `searchV` 都是 `turn` → 两项按 0.5/0.5 相加，等于手工那一半一直在跟自己抵消 |

**判定为"本来就对、不要动"的**（写在这里免得以后有人"顺手统一"）：`trainVsRandom` 那几处
（AI 只执黑，黑方视角 == 走子方视角）、以及 `sacazagent.cpp` 全部（规范视角，全程走子方口径，
是新代码的参考实现）。

**顺带记一个不是符号错、但确实是浪费的地方**：`pgagent.cpp` 的 `train()` 与 `warmupFromCurrent`
算出即时奖励之后，会被终局常量把**整条轨迹**的 reward 覆盖掉（三处出口都覆盖）。所以 PG 实际
只用"纯终局 ±1"学习，物质收益完全没进学习信号。它是自洽的（都是黑方视角），所以**没改** ——
要接上得先把量纲调平（`computeReward` 给的是 `value*10`，与 ±1 差一个量级），那是独立一项。

### 17.3 梯度累积 + 回放池（P3 + P4）

`RL::DQN::learn()` 本来就是"累批再更新"，PPO 是唯一漏掉的那个。改动分两步，缺一不可：
先"攒批反向、一次优化器更新"（P3），再"同一批数据过很多遍"（P4）。**只有 P3 是净亏的** ——
它把优化器成本按 batch 摊薄的同时，也把优化器**步数**除以了 batch，等于用"更少但更便宜的更新"
换了原来那批更新。实测（`test_ppomcts`）：

```
per-sample cost:          22.50 ms  ->  7.68 ms with grad-accum B=64     (2.9x cheaper)
128 sample-updates:     2355.0 ms  ->   617.0 ms  learnFromReplay(64,2)  (3.8x cheaper)
```

回放池的样本刻意省内存：策略目标只存**非零项**（~40 项 / 320 B），而不是 8100 维稠密分布
（32 KB）。20000 条池子因此是 `20000 × 1440 × 4B ≈ 115 MB` 而不是 ~750 MB，而且**没丢 P5 那套软目标**。

回放路径真的在学（`test_ppomcts` [9]，17 轮，目标分布 0.80/0.15/0.05）：

```
P(target0) 0.000142 -> 0.513920      (排名与目标一致: 0.5139 > 0.0651 > 0.0403)
P(target1) 0.000118 -> 0.065082
P(target2) 0.000121 -> 0.040318
V(s)      -0.041864 -> 0.903878      (价值目标 0.75)
```

顺带查出一个**既有的假数据** bug：`accumulateGrad`/`trainStep` 在 `backward()` **之后**才去读
`v[0]` 与 `policy[i]`，而 `backward()` 会把层的输出清零 —— 于是 `lastLoss` 恒等于 `target²`
（0.5625）、`lastActorLoss` 恒等于 `-ln(1e-8)·Σtarget`（18.4207），**17 轮里一个数都没变**，
而同一时间策略已经动了 3 个数量级。改成在 `backward()` 之前算标量损失以后：
actorCE 8.896 → 5.224 → 3.052 → 1.664 单调下降。

### 17.4 策略目标 = 访问分布（P5）

一次 80 次模拟的搜索里，被选中的动作往往只比其它候选多访问一两次。只保留它的 one-hot，
等于把搜索退化成一个"随机挑一步"的采样器 —— AlphaZero 赖以工作的**改进算子**就没了。
改成 `π = N_root / ΣN`（稀疏存储），实测（`test_ppomcts` [8]）三处访问 5:3:2 时
得到 `pi[100]=0.5, pi[200]=0.3, pi[300]=0.2`，和为 1，**不是 one-hot**；空树时返回 false，
调用方退回 one-hot。

**P5 对 DQN+MCTS 不适用**（判定记在这里，免得以后有人去"补"）：`DQNMCTSAgent` 的学习目标是
**Q 值的 TD 目标**（`dqn.perceive` → `DQN::learn`），它**没有策略头**，也就没有"策略目标"这个
东西可以换。MCTS 的访问计数在它那里的作用是**选点**（`selectMove` 取 `visitCount` 最大的孩子，
训练时再叠一个 ε-greedy，见 `dqnmcts_agent.cpp:379-396`），这部分本来就是对的。
SACAZ 则是上一轮就已经用了访问分布（`sacazagent.cpp` 的 `visitDistribution`），所以 P5 实际
只需要改 PPO+MCTS 一家。

### 17.5 左右镜像数据增广（P6）

**y → 8−y 是象棋的规则对称**（九宫在 y=3..5 居中、河界横跨全部 9 列、各类棋子走法形状在 y 方向
都对称），所以 `(状态, 策略目标, 价值目标)` 三元组可以整体翻一倍使用，价值目标不变。
注意这跟 `encodeState` 里那个 **x → 9−x 的规范视角镜像**是两件事：那个是为了"红黑共用一套权重"
必须做的坐标变换；**x → 9−x 不能拿来增广** —— 它会把红方的子搬到黑方半场，翻出来的局面根本不存在。

这条性质不能靠读代码确认，所以 `test_ppomcts` [10] 造了两个**真实局面**来验：
一块棋盘走原走法序列、另一块把"动作下标恰好等于镜像下标"的走法走出来，每走一步同时验证

```
[a] mirrorCell 自反=1, 第4列不动=1, mirrorActionIdx 双射=1, 与 canonicalCell 交换=1
[b] 初始局面镜像不变: RED=1, BLACK=1
[c] 镜像走法序列: 走了 12 手, 同步=1, 走法集合对应=1, 镜像走法存在=1,
    平面不一致格数=0 (最大差 0.00e+00), 原局面自身非对称格数=42
[d] 增广: 池子 2 条, 状态镜像=1, 动作下标镜像=1, 下标互异=1, 价值目标相同=1
[d] 关闭增广: 池子 1 条
```

重点看两个数：**平面不一致格数 0、最大差 0.00e+00**（比的是 1440 个平面格，全等）；
以及**原局面自身非对称格数 42** —— 说明这条路真的翻动了东西，不是"因为局面本来对称所以碰巧过"。
[c] 里的"走法集合对应"比的是**全部合法走法**的下标集合：镜像后与原局面的下标集合完全相等，
这同时证明了"镜像走法既存在、下标又正好是镜像下标"。

第一版这个测试**写错了**，值得记下来：一开始是把走法 `Step` 的坐标翻一下就丢给另一块棋盘，
结果第一步就不同步。原因是 `Step` 带着**棋子 id**，而 id 是按列的左右顺序分配的
（`RED_CHE1` 在 y=0、`RED_CHE2` 在 y=8），左右翻转会把两侧的车对调，`id` 指向的棋子在镜像棋盘上
根本不在那个格子上，`moveForward` 直接失败。改成"在镜像局面上用**它自己的走法生成**去找
镜像下标那一步"以后就对了，而且验证强度还更高。

### 17.6 多线程分身自对弈（P7）与实测的**硬件墙**

结构（`src/ppo_selfplay_mt.hpp`）：W 条 worker 各自一份**只推理**的网络（`withGrad=false`，
内存与构造时间约 1/4）+ 自己的 `Chess` + 自己的搜索树，自对弈一局就把样本交给共享池；
learner 独占带梯度的 master，把池子搬进回放池后 `learnFromReplay`，定期把权重发布到
`published` 快照，worker 拉快照。`RL::PPO` 本身**不加锁**（保持单线程语义），跨线程只靠
"快照 + 一把互斥量"交接，所以学习期间 worker 读的是上一版权重、谁都不用等谁。
每条线程一份随机引擎，编号由调用方给，因此并行跑也可复现。

实测（`bench_ppo_mt`，i7-12650H = 10 核 / 16 逻辑；80 模拟/步、每局 ≤120 手、batch=64、epochs=2、
镜像增广开；每个配置 12 局 × 2 轮）：

| workers | 墙钟秒 | 相对串行基线 | 每线程 ms/模拟 |
|---|---|---|---|
| 串行基线 | 40.5 | 1.00× | 0.422 |
| 1 | 38.7 | 1.05× | 0.384 |
| 2 | 31.8 | 1.27× | 0.541 |
| **3** | **26.3** | **1.54×** | 0.698 |
| 4 | 31.3 | 1.29× | 1.001 |
| 6 | 34.0 | 1.19× | 1.364 |

把 learner 完全摘掉（`--batch` 给一个池子永远够不到的大数）再扫一遍，看**搜索自己**能并行到什么程度：

| workers | 每线程 ms/模拟 | 相对单 worker |
|---|---|---|
| 1 | 0.294 | 1.00× |
| 2 | 0.389 | 1.32× |
| 3 | 0.623 | 2.12× |
| 4 | 0.800 | 2.72× |
| 6 | 1.379 | 4.69× |
| 8 | 1.929 | **6.56×** |

**结论（与直觉相反，但两次实验互相印证）**：瓶颈**不是** learner，是**搜索本身的访存带宽**。

* 摘掉 learner 之后曲线形状不变（一样在 2–3 条 worker 就到顶），所以 learner 不是限制；
  诊断输出也算得出来：`学习轮数 / 理论上限` 只有 0.24–0.37，learner 一直是**喂不饱**的那一方。
* 每线程的每模拟成本从 1 条 worker 的 0.294 ms 一路涨到 8 条的 1.929 ms（**6.6×**）——
  线程越多、每条线程反而越慢。这是带宽饱和的典型形状。
* 折算总带宽：每次模拟要流过的权重约 3.0 MB，其中**策略头 64×8100×4B = 2.07 MB 占了大头**；
  实测聚合吞吐在 ~5100 次模拟/秒 就到顶 → ≈ 15 GB/s。笔记本双通道内存就在这个量级，
  而 8100 维动作空间（用户选定的方案 B）注定了每次模拟都要把这 2 MB 权重从头到尾流一遍。
* 反过来，**actor-learner 重叠这件事本身是按设计工作的**：1 条 worker 时整轮墙钟 38.7 s，
  而"只搜索"的同一批局是 37.8 s —— 学习被完全藏进了搜索里；只是因为学习在总算力里占比不大，
  所以"藏起来"只值 1.05×。

要真正突破这堵墙，得让**一次权重读取服务多条样本**，也就是把多棵树的叶子**批量前向**
（AlphaZero 那套 "server + 批量推理 + virtual loss"），或者把权重压到 fp16/int8 把字节数砍半 ——
两者都是独立的大改动，本轮没做。**在当前硬件上，靠堆线程最多拿到 ~1.5×**；
同样的算力投到数据效率（P3/P4/P6：同一批数据多次复用、"重放一条 7.7 ms vs 重生成一条 55 ms"、
镜像白拿一倍数据）收益大得多。这一条作为**未修项**连同三条取舍记在
`docs/issues_review.md`「零之二点十三」④ 与待办 P2 第 10 条。

### 17.7 这一轮发现的问题

**问题清单（含修掉的、设计使然的、以及没修的）统一记在 `docs/issues_review.md`
的「零之二点十三、PPO 系列训练效率改造（P1–P7）暴露的问题」**，包括：

* 修掉的 3 个：老一代 agent 的奖励视角口径不一致（9 处）、损失曲线两个读数恒为常数、
  `PpoSelfPlayMT` 的 learner 退出条件写错导致永不退出；
* 设计使然 2 条：worker 局数超跑（最多 `W-1` 局）、PG 的物质收益被终局常量整条覆盖；
* **没修的（本轮最重要）**：多线程搜索的访存带宽上限（17.6 的表就是它的证据），
  突破方向与取舍列在该节的待办里；
* 验证方法学 2 条：吞吐指标用了"均值之比"把加速比抬高 ~45%、
  以及"用 shell 文本 cmdlet 来回写 UTF-8 源文件"把 5 个文件的中文注释一次性毁掉
  （`git checkout` 回滚，代码层无损，丢的是注释）。

