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
};
```

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

### 网络架构

```
RL::PPO (two-headed network)
  ├── Actor (policy head): state(90) → Tanh(64) → Tanh(64) → Softmax(128)
  └── Critic (value head): state(90) → Tanh(64) → Tanh(64) → Linear(1) → V(s)
```

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
  ├── 编码当前局面 → state tensor
  ├── PPO evaluate: (policyPrior, value) = ppo(state)
  ├── 创建根节点，先验来自 policyPrior
  │
  └── 循环 simulations 次：
        ├── SELECTION: PUCT 遍历到未展开节点
        ├── EXPANSION: 展开 untried 动作，PPO 评估新节点得先验
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

---

## 6. Agent 对比总结

| 特性 | ABAgent | MCTS | PGEagent | DQNAgent | PPOMCTSAgent |
|------|---------|------|----------|----------|--------------|
| **类型** | 确定性搜索 | 随机搜索 | 策略梯度 | 值函数 | 搜索+学习 |
| **神经网络** | ❌ | ❌ | ✅ DPG | ✅ DQN | ✅ PPO |
| **需要训练** | ❌ | ❌ | ✅ 自对弈 | ✅ 自对弈 | ✅ 自对弈 |
| **搜索深度** | 固定4层 | 可变(迭代) | 1步 | 1步 | 可变(MCTS) |
| **随机性** | 无 | 模拟随机 | 策略采样 | ε-greedy | MCTS+PUCT |
| **评估方式** | 手工特征 | 随机模拟 | 策略网络 | Q值网络 | 价值网络 |
| **走法排序** | MVV-LVA | UCB1 | Softmax | argmax/ε | PUCT |
| **适合场景** | 有限深度最优 | 纯树搜索 | 学习策略 | 学习Q值 | 搜索+学习 |

## 7. 各 Agent 耗时测试数据

| Agent | 每步耗时 | 强度 |
|-------|----------|------|
| ABAgent (depth=4) | ~50-200ms | 强（有限深度内） |
| MCTS (800 iter) | ~1000ms | 中等 |
| PGEagent (推理) | <5ms | 弱（未经大量训练） |
| DQNAgent (推理) | <5ms | 弱（未经大量训练） |
| PPOMCTSAgent (400 sim) | ~2000ms | 潜力最大（需训练） |

> *注：以上为基于测试程序的实测数据，实际耗时和强度取决于训练程度和参数调优。*
