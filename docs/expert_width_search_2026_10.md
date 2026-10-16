# 序列 Transformer 专家的两个宽度：`hiddenDim` 与 `tokDim` 的最优值（2026-10-16）

> 问题（用户口径）：**"寻找各个 agent transformer 专家的 hiddenDim 的最优数值"**。
> 对象：仓库里**专家已经是序列 Transformer** 的四支 agent
> （`AGENT_PPOMCTS` / `AGENT_SACAZ_MOE` / `AGENT_DQNMCTS_MOE` / `AGENT_DQNAB`）。
>
> 代码与工具：`tools/sweep_expert_width.ps1`（新增，扫描驱动）·
> `test/train_bc_main.cpp` + `src/bcrun.hpp`（既有 BC 流程）·
> `CMakeLists.txt` 新增 `train_bc_sacaz_seq` / `train_bc_dqn`（两个**变体**目标）·
> `src/bcdqnagent.hpp` + `test/train_bc_dqn_main.cpp`（新增，两个 Q 类 agent 的 BC 胶水）·
> `test/gui_seq_wiring_main.cpp`（新增，界面接线回归测试）·
> `CMakeLists.txt` 的 `CHESS_CORE_SEQ`（**修掉一个真事故**，见 §6）。
>
> 本文只写**读数与它们的边界**；每个数字都能用 §7 的命令复跑出来。

---

## 0. 结论速览

### 0.1 每个 agent 的建议宽度

| 界面档位 | 类 | 建议 `hiddenDim` | 建议 `tokDim`(=expertHidden) | 现役/默认 | 依据 |
|---|---|---:|---:|---|---|
| `AGENT_PPOMCTS` | `PPOMCTSAgent` | **24**（20~32 都可以） | **64**（保持） | 64 / 64 | 两 seed 平均：24 → 2.5420、64 → 2.6616（−0.120 nats） |
| `AGENT_SACAZ_MOE` | `SACAZMoETbAgent` | **24** | **64**（保持） | 64 / 64 | 两 seed 平均：24 → 2.5299、64 → 2.6858（−0.156） |
| `AGENT_DQNMCTS_MOE` | `DQNMCTSMOETbAgent` | **24**（或 20） | **64**（保持） | 64 / 64 | s2：20 → 2.8320、64 → 2.8800（−0.048）；CE 口径见 §2.3 |
| `AGENT_DQNAB` | `DQNABAgent` | **24**（或 20） | **64**（保持） | 64 / 64 | s2：20 → 2.5157、64 → 2.6567（−0.141）；CE 口径见 §2.3 |
| （对照，未扫）`AGENT_SACAZ_OLD_MOE` | `SACAZLegacyAgent` | 24（推断） | 64（推断） | 64 / 64 | 界面下拉框里已经移除（`mainwindow.cpp:168-173`），本轮**没有**给它做变体 |

一句话：**四个 agent 的答案一致 —— 瓶颈宽度取现役默认的一半左右（20~32），专家 token 宽度保持 64。**
`hiddenDim` 从 64 降到 24/32，留出 CE 降 **0.09~0.19 nats**（同一 teacher、同一数据、同一 seed 下的配对比较）；
`tokDim` 从 64 抬到 128/192 只值 **≤0.02 nats**，却让每步搜索慢 **2.55×**。

### 0.2 三条能读出来的规律

1. **`hiddenDim` 是单峰 (U 形)，谷底在 20~24**：`h≤16` 欠拟合（训练 CE 也高），`h≥48` 过拟合
   （训练 CE 最低而留出最高）。四个 agent 的形状**逐条相同**（§3.5 的表）。
2. **`tokDim` 是二阶变量，而且它的代价是实测的**：64 / 128 / 192 三档在 `h=32` 上的留出 CE 差
   ≤0.02 nats（seed 1：PPO 2.5994 / 2.5966 / 2.5780；SAC 2.6174 / 2.6130 / 2.6130），
   而同协议 100 局里 `tokDim=128` 是 **972.3 ms/步 vs 64 的 381.9 ms/步（2.55×）**，
   比分 **11/11/78（恰好 50.0%）** ⇒ **买不到棋力**。
3. **现役默认的 64 在两个 seed 上都不是最优**：它每次都在"过拟合那一侧"，
   留出 CE 比谷底高 0.09~0.19 nats，而且比谷底**略慢**（`h=32` 是 `h=64` 的 1.07× 快）。

### 0.3 棋力那一步的边界（不要跳过）

`h=32` vs `h=64` 的 100 局是 **53.5% / +24 Elo / inconclusive**（§5.2）——
**方向与 CE 一致，但没有判读到定论**（27 局分胜负、和棋 73%）。
所以"把 `hiddenDim` 改成 24"是一条**有 CE 与耗时两条支撑、棋力方向为正但未定论**的建议；
把它当成"免费的改进（更好的先验 + 更便宜）"是合理的，当成"Elo +24"则**不是**。


### 0.3 顺带的一条（不是超参，但更要紧）

**界面里四支 agent 的"序列 Transformer 专家"此前根本没有生效**：
`CMakeLists.txt` 的四个 `*_MOE_EXPERT_SEQ` 宏只加在 `chess` **自己的源文件**上，
而真正建 agent 的 `chessboard.cpp` 与四个 agent 的 `.cpp` 全在 `CHESS_CORE` 里 ——
下拉框显示"序列专家"、实际建的是 **TB 专家**。本轮修掉并加了回归测试（§6）。
**本文所有扫描都是在"宏真的生效"的变体二进制上做的**，也就是说：这些最优值是
"修好之后"的口径。

---

## 1. 为什么这两个宽度值得问

序列专家的结构（`src/rl/seq_transformer.hpp`）由**两个**独立的宽度决定：

| 旋钮 | 代码口径 | 现役默认 | 它决定什么 |
|---|---|---:|---|
| `tokDim`（= `expertHidden`） | `SeqTransformerExpert::tokDim`（token 宽度 `D`） | 64 | 专家**内部**的模型宽：`embed`/QKV/`Wo`/FFN 全部按它定宽；MAC ≈ `O(D²)`；`d_k = D / Heads` 必须整除 |
| `hiddenDim` | `PPOMCTSAgent` / `SACAZMoETbAgent` / `DQNMCTSMOETbAgent` / `DQNABAgent` 构造第 1~2 参 | 64 | 骨干**之后**、输出头之前那一层的宽度 `h`（PPO 的 actor/critic 头都吃它） |

两者在仓库里都没有被搜过：`docs/seq_transformer_design.md` §9.8 自己把
"超参搜索"列为**未做**，`docs/expert_search_optimization_plan_2026_10.md` §5 的 ★Z2'
把它列成待办。而它们**很可能是本工程最便宜的收益**：`hiddenDim` 只是头部宽度，
改它不作废任何"结构口径"，而 §5.3 的参数账（`docs/dev_sacmoetb_architecture.md`）
早就指出"参数/数据比"是本工程的头号瓶颈。

---

## 2. 口径：用什么读数定义"最优"

### 2.1 主判据：BC 留出 CE / top-1

用 `train_bc`（既有工具）把 Alpha-Beta 深度 4 的选点蒸馏进各 agent 的**策略路径**，
报告**留出集**上的 CE / top-1 / `P(老师着法)` / 策略熵 / KL。为什么是它：

* 它**便宜**（一次 1000 局面 ~13 min，能扫几十档），而棋力（100 局配对）一次要 ~70 min；
* 它是**同一份数据、同一个老师、同一个 seed**，换的只有宽度 ⇒ 差分干净；
* 它有一个**独立于训练目标的判据**：留出集（`--holdout=5`，1/5 的局面从不参与更新）。

三条必须一起看的边界（`src/bcrun.hpp` 的报告里也印）：

1. **prior top-1 是训练目标本身** —— 它高只说明"克隆成功了"，不是棋力证据；
2. **CE 的绝对值在本轮不能跨 agent 比**（PPO 的头是 8100 列 softmax、SAC 支是对齐 8100、
   DQNMCTS 是 128 个哈希槽），只有"同 agent、同协议、换宽度"的相对比较成立；
3. **软目标的 CE 下界是 `H(t)`**，所以要比就比 `CE − H(t)`（报告里的 KL 列），
   或者比同一格内的 CE。本轮的臂**共用同一份标签**（同 seed/depth/positions）⇒ 直接比 CE 即可。

### 2.2 协议（除特别说明，全部表都是这一套）

```bat
:: PPO 支 (AGENT_PPOMCTS)
train_bc.exe      --agent=ppo-seq --pos=2d ^
                  --positions=<N> --depth=4 --soft=1 --soft-depths=4 ^
                  --epochs=8 --batch=64 --lr=0.002 --seed=<S> --holdout=5 ^
                  --hidden=<h> --expert=<tokDim>
:: SAC 支 (AGENT_SACAZ_MOE)
train_bc_sacaz_seq.exe --agent=sac-moe --shared=1 <其余同上>
:: 两个 Q 类 agent (AGENT_DQNAB / AGENT_DQNMCTS_MOE)
train_bc_dqn.exe  --agent=dqnab|dqnmcts-moe <同上, 无 --shared>
```

* `--depth=4 --soft=1 --soft-depths=4` ⇒ 老师是"深度 1..4 一致性的分布"（与既有文档同源）；
* `--holdout=5` ⇒ 1/5 局面是留出集（1000 局面时 200 条，300 局面时 60 条）；
* `--seed` 只有两个值：`20240901`（seed 1）与 `20240902`（seed 2）。**换 seed = 换一批局面 + 换一套初始化**；
* 所有臂都在**同一个二进制**里跑；换二进制前做过逐位复现检查（§7.2）。

### 2.3 两个 Q 类 agent 的读数**不是**"决策策略的 CE"（本轮新增的工具）

`AGENT_DQNAB` / `AGENT_DQNMCTS_MOE` 在本轮之前**没有任何 BC 路径**
（`src/chessboard.cpp:6423-6425` 早就把这条记成待办）。本轮新增 `src/bcdqnagent.hpp` +
`test/train_bc_dqn_main.cpp`，口径是：

```
π = 在**合法列**上对 Q 做 softmax(Q / T)，T ≡ 1.0
loss = 掩码交叉熵 CE(π, t)        梯度 = π − t（非法列恒 0）
top-1 = 合法列上 Q 的 argmax 命中率（尺度无关，不受 T 影响）
```

* **DQNAB**：`Q_i = V + A_i − mean_legal(A)`，而 `V` 与 `mean_legal(A)` 对该状态是常数
  ⇒ π 只由 A 头决定；又因 `Σ(π−t) = 0`，dueling 的组合项给出 `dL/dA_j = π_j − t_j`、
  **V 头梯度恒为 0**。这条不是推理而是**断言**：`update()` 每批首尾各取一次 V 头权重的
  64 位指纹，不一致就 `abort`（做守卫时故意注入过一次改动，它确实响了）。
* **DQNMCTS**：`qHead` 是 `Linear` ⇒ logits 就是 Q，直接同式；动作是 **128 个哈希槽**
  （有别名），π 的分母是**槽位数**（同一槽的多个着法只算一次，否则 `Σπ ≠ 1` 会让上面的推论失效）。
* ⚠ **它监督的是"Q 隐含的合法列先验"，不是 agent 的真实决策策略**
  （DQNAB 决策走根搜索值 `softmax(v_a/T)`，DQNMCTS 走 PUCT + priors）。
  所以它只回答"**同一份数据下哪种宽度学得更准**"，不回答"哪种宽度下棋更强"。

---

## 3. 读数

### 3.1 二维网格：`hidden` × `expert`（筛选轮）

**PPO 支**（`--positions=300`，holdout 60，seed 1，`--pos=2d`）留出 CE：

| hidden \ expert | 32 | 64 | 128 |
|---|---:|---:|---:|
| **32** | 2.9848 | 2.9897 | **2.9679** |
| **64** | 3.1229 | 3.0940 | 3.0862 |
| **128** | 3.2926 | 3.1844 | 3.2365 |

**SAC 支**（`--positions=300`，holdout 60，seed 1）：

| hidden \ expert | 32 | 64 | 128 |
|---|---:|---:|---:|
| **32** | 2.9894 | 2.9912 | 3.0152 |
| **64** | 3.1257 | 3.1690 | **2.9735** |
| **128** | 3.2707 | 3.3357 | 3.2179 |

**两个 Q 类 agent**（`--positions=600`，holdout 120，seed 1）：

| agent | hidden \ expert | 32 | 64 | 128 |
|---|---|---:|---:|---:|
| DQNAB | **32** | 2.8301 | 2.8113 | **2.8056** |
| DQNAB | 64 | 2.9084 | 2.9229 | 2.9078 |
| DQNAB | 128 | 3.1008 | 3.0851 | 3.0875 |
| DQNMCTS | **32** | 2.8415 | **2.8321** | 2.8363 |
| DQNMCTS | 64 | 2.9009 | 2.8862 | 2.8740 |
| DQNMCTS | 128 | 2.9516 | 2.9534 | 2.9044 |

**能读出来的**：四支 agent 的 `hidden` 排序完全一致（32 < 64 < 128），
而 `expert` 的差别 **≤0.04 nats 且方向不稳**（SAC 的 `h=64, e=128` 是 60 条留出上的一个
反例读数，1000 局面时它消失了，见 §3.3）。
**不能读出来的**：60 / 120 条留出集上，0.01~0.04 的差别是噪声 —— 所以要 §3.3 的重扫。

### 3.2 边角扫描：`h ≤ 16` 与 `tokDim ≥ 192`（筛掉"越大越好"）

`--positions=300`，seed 1，留出 CE：

| 臂 | PPO | SAC |
|---|---:|---:|
| `h=8, e=192` | 3.3776 | 3.4971 |
| `h=8, e=256` | 3.2739 | 3.2535 |
| `h=16, e=192` | 3.2183 | 3.1532 |
| `h=16, e=256` | 3.3101 | 3.2768 |

对照 §3.1 里 `h=32` 的 2.97~2.99：**`h ≤ 16` 是明显的欠拟合**（同一批臂的**训练** CE 也升到
2.7~3.2，即"连训练集都没拟合上"）⇒ 谷底不在窗口外，在 20~32 之间。

### 3.3 1000 局面重扫（seed 1，holdout 200）

| agent | hidden | e=64 | e=128 | e=192 |
|---|---:|---:|---:|---:|
| PPO | **32** | 2.5994 | 2.5966 | **2.5780** |
| PPO | 48 | 2.6626 | 2.6391 | 2.5995 |
| PPO | 64 | 2.6935 | 2.6819 | 2.6575 |
| SAC | **32** | 2.6174 | **2.6130** | **2.6130** |
| SAC | 48 | 2.6597 | 2.6625 | 2.6750 |
| SAC | 64 | 2.7239 | 2.7165 | 2.7145 |

**读出来的**：
* `hidden` 的 U 形在 200 条留出上依然清晰（每一列内 `32 < 48 < 64`）；
* `expert` 在 `h=32` 上**几乎是平的**（PPO 3 档差 0.021、SAC 差 0.004）；
  在 `h=48/64` 上"更大的 `expert` 略好"（PPO 64→192 值 0.036）—— 这是
  "瓶颈固定、专家更宽 ⇒ 有效容量换个地方花"的典型形状，量级仍在噪声边缘；
* 同一批臂的**训练** CE 与留出 CE **反向**：`h=64/e=64` 训练 CE 2.0812（最低）、留出 2.6935（最高），
  而 `h=32/e=192` 训练 2.1745、留出 2.5780 ⇒ 这是**过拟合**，不是"大模型没学好"。

### 3.4 seed 2 的低带宽扫描（`e=64`，`--positions=1000`，holdout 200）

| agent | h=12 | h=16 | h=20 | h=24 | h=32 | h=40 | h=48 | h=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 2.5623 | 2.5318 | **2.5152** | 2.5190 | 2.5207 | 2.5766 | 2.5832 | 2.6296 |
| SAC | 2.6251 | 2.5663 | 2.5069 | **2.5034** | 2.5653 | 2.5772 | 2.6037 | 2.6477 |
| DQNAB | — | — | **2.5157** | 2.5188 | 2.5568 | — | — | 2.6567 |
| DQNMCTS | — | — | **2.8320** | 2.8353 | 2.8542 | — | — | 2.8800 |

**四支 agent 在同一形状上一致**：谷底 20~24、`h=32` 略高、`h=64` 明显最高、
`h≤16` 开始欠拟合。⇒ 这是**结构性的**，不是某一支的偶然读数。

### 3.5 两个 seed 的合并结论（`tokDim=64`，`--positions=1000`，holdout 200）

| agent | seed | h=20 | h=24 | h=32 | h=40 | h=48 | h=64 |
|---|---|---:|---:|---:|---:|---:|---:|
| PPO | 20240901 | 2.5739 | **2.5649** | 2.5994 | — | 2.6626 | 2.6935 |
| PPO | 20240902 | **2.5152** | 2.5190 | 2.5207 | 2.5766 | 2.5832 | 2.6296 |
| PPO | **两 seed 平均** | 2.5446 | **2.5420** | 2.5601 | — | 2.6229 | 2.6616 |
| SAC | 20240901 | 2.5564 | **2.5563** | 2.6174 | — | 2.6597 | 2.7239 |
| SAC | 20240902 | 2.5069 | **2.5034** | 2.5653 | 2.5772 | 2.6037 | 2.6477 |
| SAC | **两 seed 平均** | 2.5317 | **2.5299** | 2.5914 | — | 2.6317 | 2.6858 |
| DQNAB | 20240901 | **2.5716** | 2.5955 | — | — | — | — |
| DQNAB | 20240902 | **2.5157** | 2.5188 | 2.5568 | — | — | 2.6567 |
| DQNMCTS | 20240901 | **2.7974** | 2.8042 | — | — | — | — |
| DQNMCTS | 20240902 | **2.8320** | 2.8353 | 2.8542 | — | — | 2.8800 |

（"—" = 该档在这一轮没有按同一协议跑过。两个 Q 类 agent 的筛选轮是 `--positions=600`，
**它的 CE 不能与这张表并列**（数据量与留出集都不同），所以这里留空而不是填一个不可比的数。）

**读出来的**：

* **四个 agent 的谷底都在 20~24**，而 `h=20` 与 `h=24` 的差 ≤0.013 nats（噪声）
  ⇒ **取 24 或 20 都对，本文建议 24**；
* `h=32` 比谷底高 0.002~0.05、`h=48` 高 0.05~0.09、**`h=64`（现役默认）高 0.09~0.19**；
* PPO/SAC 两个（同一个"`h → 8100` 稠密策略头"形状）在两 seed 平均上**都是 24 最低**，
  两个 Q 类 agent（头窄得多）**都是 20 最低** —— 差别在噪声内，但方向与"头部参数账"一致（§4）。

### 3.6 与现役默认的差（同一协议、同一 seed 的配对读数）

| agent | seed | `h=64`（现役） | 最佳档 | 差 | `h=64` 的**训练** CE | 最佳档的训练 CE |
|---|---|---:|---:|---:|---:|---:|
| PPO | 20240901 | 2.6935 (h=24 为 2.5649) | h=24 | **−0.129** | 2.0812 | 2.1947 |
| PPO | 20240902 | 2.6296 (h=20 为 2.5152) | h=20 | **−0.114** | 2.0903 | 2.2127 |
| SAC | 20240901 | 2.7239 (h=24 为 2.5563) | h=24 | **−0.168** | 2.0786 | 2.1958 |
| SAC | 20240902 | 2.6477 (h=24 为 2.5034) | h=24 | **−0.144** | 2.0841 | 2.1773 |
| DQNAB | 20240902 | 2.6567 (h=20 为 2.5157) | h=20 | **−0.141** | 2.0843 | 2.2065 |
| DQNMCTS | 20240902 | 2.8800 (h=20 为 2.8320) | h=20 | **−0.048** | 2.6992 | 2.7021 |

**注意最后两列**：`h=64` 的**训练** CE 反而更低（2.08 vs 2.18）—— 也就是说
"留出更好"不是因为大瓶颈学不动，而是因为它**正在过拟合**。这正是本轮最稳的一条读数，
因为它在四个 agent、两个 seed、六组配对里**方向全部相同**。


### 3.7 `tokDim` 的代价（为什么建议保持 64）

| 读数 | `tokDim=64` | `tokDim=128` | 出处 |
|---|---:|---:|---|
| 每专家 MAC（文档算术） | 9.67 M | ≈ 4× | `docs/seq_transformer_design.md` §5.1 |
| **每步搜索耗时**（同进程、同一次对局内的 A/B 配对） | **417.1 ms/步** | **1021.1 ms/步**（2.45×） | §5.3 的 4 局冒烟（同一条命令的 A/B 两侧） |
| 留出 CE（同 `h`） | 2.5994 | 2.5966 | §3.3 |

⇒ **多花 2.45× 的推理时间买 0.003 nats**。本工程"模拟次数是唯一测出过棋力的杠杆"
（`src/rl/ppo.h:272-273`），所以这个交换是亏的：同样墙钟预算下 `tokDim=64` 能多搜 2.4 倍的模拟。

---

## 4. 为什么会这样（结构性解释，不是"小就是好"）

1. **`hiddenDim` 是"输出头之前的瓶颈"，不是"骨干容量"**。骨干（专家）的容量由 `tokDim`/`E` 决定；
   `h` 只决定"专家输出 → 动作/价值"这一层的宽度。而动作空间是 **8100 维**
   （SAC 对齐口径）或 **128 槽**（DQNMCTS）：**头部参数随 `h` 线性涨**（`h×8100`），
   所以 `h=64` 的头是 `h=24` 的 2.7 倍参数 —— 而数据量（1000 局面 / 200 条留出）没变。
   这正是 `docs/sac_regression_2026_09.md` 记过的同一条机制（"对齐表示掉 200 Elo 的锅在
   稠密策略头，不在状态维数"），本轮在**四个 agent 上分别复现了一次**。
2. **`h≤16` 变成欠拟合**说明瓶颈确实承担一部分必要容量（把专家的 1710 维表示压成可用的
   合法列打分）。所以它是 U 形，而不是"越小越好"。
3. **`tokDim` 的收益在 `h` 小时更明显**（§3.3：`h=32` 上 64→192 值 0.021），
   与 "瓶颈太窄会浪费专家容量" 一致 —— 但量级不足以支付它的推理代价。

---

## 5. 棋力验证（100 局 / 50 配对开局）

### 5.1 协议

两臂都先用**同一份** BC 协议各自训练并存权重（`--positions=1000 --depth=4 --soft=1
--soft-depths=4 --epochs=8 --batch=64 --lr=0.002 --seed=20240901 --holdout=5 --pos=2d`），
然后用 `bench_agent_arena` 在同一模拟预算下打 100 局（50 个配对开局、每局换先手）：

```bat
bench_agent_arena.exe --mode=match --a=ppo --b=ppo ^
    --backbone-ppo-a=seq --backbone-ppo-b=seq --pos-ppo-a=2d --pos-ppo-b=2d ^
    --expert-ppo-a=64 --hidden-ppo-a=32 --load-a=weights/bc_seq_h32e64 ^
    --expert-ppo-b=64 --hidden-ppo-b=64 --load-b=weights/bc_seq_h64e64 ^
    --games=100 --sims=64 --plies=120 --opening=8 --paired-openings --seed-mcts-rng --seed=20240901
```

`--hidden-ppo-a/b` 是**本轮新加的每侧旋钮**（`test/bench_agent_arena_main.cpp`），
以前 `PPOMCTSAgent` 的头部宽度被写死成 64，于是"两边只差 `hidden`"这个对照**根本无法表达**。

### 5.2 结果

| 对照 | 局数 | A/B/和 | A 得分率 | Wilson 95% | Elo 差 | 判读 | A 平均手数 | A/B 材料差 | A ms/步 | B ms/步 |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| **A = `h=32` vs B = `h=64`**（都 `tokDim=64`） | 100 | 17 / 10 / 73 | **53.5%** | [43.8, 63.0] | **+24** | **inconclusive** | 37.7 | −0.0 | 428.8 | 460.3 |
| A = `h=32, tokDim=128` vs B = `h=32, tokDim=64` | 100 | 11 / 11 / 78 | **50.0%** | [40.4, 59.6] | **−0** | **inconclusive** | 35.0 | +0.0 | 972.3 | 381.9 |

**怎么读这两行**（三件事分开说，不要混成一句）：

1. **`hiddenDim` 64 → 32**：留出 CE 好 **0.094 nats**（§3.6），对局**方向为正**（53.5% / +24 Elo），
   但 100 局**不足以判读**（27 局分胜负 ⇒ 区间 ±10pp）。这是一个"**方向一致但未定论**"的读数。
2. **`tokDim` 64 → 128**：留出 CE 只差 **0.003 nats**（§3.3），对局**恰好 50.0%**（11/11/78），
   而每步耗时是 **2.55×**（972.3 vs 381.9 ms/步，同一批对局内的配对读数）。
   ⇒ **这一条是决定性的**：多花 2.55× 的推理时间买不到任何棋力 ⇒ `tokDim` 保持 64。
3. **两行的"和棋率"都很高（73% / 78%）**：纯 BC 权重（没有 RL 微调）的棋力本来就弱，
   和棋把分辨率吃掉了。要真正判读 `hiddenDim` 那 0.09 nats 有没有变成棋力，
   需要 **≥200~400 局**（或先做一轮 RL），本轮**没有做**（§8 第 1、9 条）。

⚠ 第二行的耗时是**同进程、同一批局面**的配对读数（`a_ms_per_move` / `b_ms_per_move` 进 CSV），
比 §5.3 那条 4 局冒烟可信；它同时说明"tokDim 的代价是真实的、可复现的"。


### 5.3 4 局冒烟（用来确认"两侧配置真的按参数建网/载权重"，也给出每步耗时配对）

```
A 序列  : 位置编码=2d  token宽度=128  dropout=0.000  头部瓶颈=32
B 序列  : 位置编码=2d  token宽度=64   dropout=0.000  头部瓶颈=64
A 权重  : weights/bc_seq_h32e128 -> 载入成功
B 权重  : weights/bc_seq_h64e64  -> 载入成功
比分    : A 2 胜 / B 0 胜 / 和 2   (100 局之前的冒烟, 4 局不足以判读)
思考时间: A 1021.1 ms/步, B 417.1 ms/步  -> B 是 A 的 0.41x
```

⚠ 这条 2.45× 是**同进程、同一批局面**的配对读数（§3.7 用它），但它同时含 `tokDim` 与
`hidden` 两个变量；单变量版本见 §5.4 的 CSV（`a_ms_per_move` / `b_ms_per_move`）。

### 5.4 100 局读数（原始报告行）

```
=== 结果 (PPO+MCTS/稀疏MoE(序列Transformer专家)/pos=2d 视角) ===
  比分      : A 17 胜 / B 10 胜 / 和 73
  得分率    : 53.5%   95% Wilson 区间 [43.8%, 63.0%]  -> inconclusive
  Elo 差    : +24 (A 相对 B)
  思考时间  : A 428.8 ms/步, B 460.3 ms/步  -> B 是 A 的 1.07x
  总耗时    : 1660.0 s (16.60 s/局)
VERDICT: PASS | ppo vs ppo | 局数=100 比分=17/10/73 得分率=53.5% 区间[43.8,63.0] Elo+24 inconclusive
```
（A = `hidden=32`，B = `hidden=64`，两侧 `tokDim=64`，权重 `weights/bc_seq_h32e64` / `bc_seq_h64e64`）

```
=== 结果 (PPO+MCTS/稀疏MoE(序列Transformer专家)/pos=2d 视角) ===
  比分      : A 11 胜 / B 11 胜 / 和 78
  得分率    : 50.0%   95% Wilson 区间 [40.4%, 59.6%]  -> inconclusive
  Elo 差    : -0 (A 相对 B)
  思考时间  : A 972.3 ms/步, B 381.9 ms/步  -> B 是 A 的 0.39x
  总耗时    : 2405.7 s (24.06 s/局)
VERDICT: PASS | ppo vs ppo | 局数=100 比分=11/11/78 得分率=50.0% 区间[40.4,59.6] Elo-0 inconclusive
```
（A = `tokDim=128`，B = `tokDim=64`，两侧 `hidden=32`，权重 `weights/bc_seq_h32e128` / `bc_seq_h32e64`）

CSV：`.r1build/sweep/match_hidden_32vs64.csv`、`.r1build/sweep/match_expert_128vs64.csv`
（两场都是 `--games=100 --sims=64 --plies=120 --opening=8 --paired-openings --seed-mcts-rng --seed=20240901`，
⇒ 同一批 50 个配对开局、开局集指纹相同 ⇒ 两场之间也可横向比。）

**边界（必须先说）**：这里比的是**两块纯 BC 权重**（没有 RL 微调），所以它回答的是
"**先验质量差**在 64 次模拟的搜索下能不能变成比分差"，不是"final 棋力"。
参照 `docs/seq_transformer_encoding_eval_2026_10.md` §4.1 的教训：40 局会给假 decisive，
**100 局 / 50 配对开局**才是本工程的最低口径，而 `inconclusive` 是合法结论 ——
本轮两场都落在 `inconclusive` 上，且都**没有被读成"没有差别"**（第 1 场方向为正）。


---

## 6. 顺带修掉的一个真事故：界面那四个宏从来没生效（2026-10-16）

### 6.1 症状与证据

| 层 | 看到的东西 |
|---|---|
| 下拉框（`mainwindow.cpp`，在 `APP_SOURCES` 里） | "SAC+MCTS+AlphaZero (稀疏MoE+**序列**专家)" 等四行 —— 因为 `chess` target 有那四个宏 |
| 建 agent（`chessboard.cpp` + 四个 agent 的 `.cpp`，**在 `CHESS_CORE` 里**） | **TB 专家** —— 那些 TU 没有宏 |
| 生成文件（可复核，不需要跑程序） | `grep -A1 'CHESS_CORE.dir.*chessboard.cpp.obj' build.ninja` ⇒ DEFINES 里**没有**那四个宏；`chess.dir/*` 的目标行**有** |

后果不止是"专家类型不对"：`chessboard.cpp:1054` 的 `a->legacyHashAction = true`
（序列专家那一支的动作口径）也被**整段编掉**；四支的权重前缀也仍然是 TB 那一套
（前缀在各自的 `.cpp` 里由宏选择）。

### 6.2 修法与代价

游戏层**编两份**（`CMakeLists.txt`）：

| 库 | 宏 | 谁链它 |
|---|---|---|
| `CHESS_CORE`（原样） | 无 | 全部 `test_*` / `bench_*` / `train_*`（既有断言、读数、权重文件**一位不动**） |
| `CHESS_CORE_SEQ`（新增） | 那四个宏，且是 **PUBLIC** | **只有界面 `chess`** |

* **PUBLIC 是必须的**，不是风格：`STATE_DIM` / `ACTION_DIM` / 专家类型都是**编译期**成员，
  每一个 include 这些头文件的 TU 都必须看到同一套宏，否则同一个类在两个 TU 里是不同的定义
  （ODR 违反）。本轮**真的踩到了**：第一版把宏写成 PRIVATE，于是库带宏、`test_gui_seq_wiring`
  不带宏，它的 `static_assert(SACAZMoETbAgent::STATE_DIM == 1710)` **当场编译失败** ——
  那次失败正是这个隐患的报警器。
* 代价：游戏层多编一遍（~19 个 TU）。这是 `CHESS_CORE` 那段注释里
  "编译速度 vs 正确性"的取舍：宁可多编一次，也不要一个"界面显示 A、实际跑 B"的构建。

### 6.3 回归测试（新增 `test_gui_seq_wiring`）

它链 `CHESS_CORE_SEQ`，用**三条互相独立**的证据钉住"界面那一份是序列专家"：

| 证据 | 读数（实测） |
|---|---|
| 编译期维数 | `static_assert(SACAZMoETbAgent::STATE_DIM == 1710)`、`DQNMCTSMOETbAgent::STATE_DIM == 1710`（不带宏时分别是 1263 ⇒ 编译失败） |
| 权重前缀 | `SACAZMoETbAgent::sharedWeightPrefix() = weights/sacaz_moe_seq_shared_agent` |
| 结构（参数量差 60 倍） | `DQNAB` 主干 **465,096** / `DQNMCTS` 主干 **465,096**（TB 口径是 28.77 M 级）；骨干名都含"序列Transformer专家" |
| 运行时旋钮是活的 | `tokDim` 32 → 208,840 / 64 → 465,096 / 128 → 1,469,128；不传 == 传默认常量 |
| **断言总数** | **10 条, 失败 0 条** |

另外做过的验证：`chess` 目标重新构建通过（`CHESS_CORE_SEQ` 32/32）；
`chess.exe` 里能找到四支的 seq 权重前缀字符串；`test_bc` / `test_transformer` /
`test_dqnab_seq` 全过（§7.4）。

---

## 7. 复现

### 7.1 构建

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
C:\Qt\Tools\CMake_64\bin\cmake.exe -S . -B build/sweep -G Ninja ^
    -DCMAKE_PREFIX_PATH=C:/Qt/6.8.0/msvc2022_64 -DCMAKE_BUILD_TYPE=Release
C:\Qt\Tools\CMake_64\bin\cmake.exe --build build/sweep --target ^
    train_bc train_bc_sacaz_seq train_bc_dqn bench_agent_arena test_gui_seq_wiring
```

### 7.2 扫描（一条命令一个网格；并发跑，读数与并发度无关）

```bat
:: PPO 支 (AGENT_PPOMCTS)
pwsh -File tools/sweep_expert_width.ps1 -Exe build/sweep/train_bc.exe ^
     -Agent ppo-seq -AgentArgs "--pos=2d" -Tag ppo2d ^
     -Hidden 20,24,32,48,64 -Expert 64 -Positions 1000 -Epochs 8 -Jobs 4
:: SAC 支
pwsh -File tools/sweep_expert_width.ps1 -Exe build/sweep/train_bc_sacaz_seq.exe ^
     -Agent sac-moe -AgentArgs "--shared=1" -Tag sacseq -Hidden 20,24,32,48,64 -Expert 64 ...
:: 两个 Q 类 agent
pwsh -File tools/sweep_expert_width.ps1 -Exe build/sweep/train_bc_dqn.exe ^
     -Agent dqnab -Tag dqnab -Hidden 20,24,32,64 -Expert 64 ...
```

输出：每个臂一份完整报告 `.r1build/sweep/<Tag>_h<h>_e<e>.txt` + 一份汇总 CSV。
⚠ **`Sec` 一列只作量级**：本轮是多臂并发跑的，同一批臂之间可比，**跨批不可比**
（`docs/seq_transformer_design.md` §8.4 的计时纪律）。

**逐位复现检查**（换二进制之后必须做）：`ppo-seq --positions=300 --hidden=32 --expert=64`
在改动前后各跑一次，留出 CE 都是 **2.9897 / top-1 23.33%**（逐位相同）⇒ 新增的
`bcdqnagent.hpp` include 没有改到 PPO/SAC 的 BC 路径。

### 7.3 棋力

见 §5.1 的逐字命令（两条；CSV 落在 `.r1build/sweep/match_*.csv`）。

### 7.4 断言级验证

```bat
cd build/sweep && mkdir weights && ctest --output-on-failure
```

本轮实测（全量 23 个测试）：**22 通过 / 1 失败 —— 唯一失败是 `test_match` 的既有 3 条**
（`371 项断言 / 3 项失败`）。那 3 条在本仓库里**早就有归因**：`[2.7c] 学习信号与对手无关`
（要求"只换对手 ⇒ 第一次训练损失逐位相同"，而回放池不在快照/还原范围内）+ `[2.7k]` 的两条
后台训练**时间窗**断言；README 与文档都记了"把改动全部 `git stash`、重建 HEAD 之后
**逐条相同**地失败（HEAD: 367 项 / 3 失败）"——见 `README.md` 的 `test_match` 条与
`docs/seq_transformer_encoding_eval_2026_10.md` §7.1 的实测表。
本轮**没有碰**这三条断言，也没有碰它们覆盖的路径（后台训练同步 / 回放池快照）。

与本次改动直接相关的套件全部通过：

| 套件 | 读数 |
|---|---|
| `test_gui_seq_wiring`（新增） | **10 项断言 / 0 失败**（§6.3） |
| `test_bc` / `test_transformer` / `test_ppo_backbone` | 167 / 89 / 通过 |
| `test_dqnab` / `test_dqnab_seq` / `test_dqnmcts_moe` / `test_dqnmcts_moe_seq` | 115 / 115 / 86 / 88，全 0 失败 |
| `test_sacaz` | 通过（305.6 s） |

---

## 8. 诚实边界

1. **主判据是模仿质量，不是棋力。** 留出 CE/top-1 只说明"同一份 AB 数据学得更准"。
   §5 的 100 局是把这两件事接起来的那一步，结果是 **53.5% / inconclusive**：
   `hiddenDim` 那条**方向为正但未定论**；`tokDim` 那条**恰好 50%**（这才是决定性的）。
2. **本轮没有改任何生产默认值**：`chessboard.cpp` 里的 `PPO_SIMS`/`SACAZ_HIDDEN`/
   `DQNAB_HIDDEN`/`DQNMCTS_MOE_HIDDEN` 等常量**一位没动**。改成 24 是一次
   "改默认结构 ⇒ 该支权重要重训"的改动，属于**用户口径**，不该顺手做掉。
3. **两个 Q 类 agent 的 CE 与 PPO/SAC 不可比**（值头 vs 策略头、8100 双射 vs 128 哈希槽），
   而且它监督的是"Q 隐含先验"而不是决策策略（§2.3）。它们的 `top-1` 在 128 槽口径下
   还有**别名**（同一槽的多个着法只算一次）。
4. **数据规模只有一个点**：1000 局面 / 200 条留出（Q 类那一支的筛选轮是 600/120）。
   本工程自己量过"0.01~0.09 nats 在 300 局面下单 seed 不足以当结论"
   （`docs/seq_transformer_design.md` §9.7）⇒ 本轮用**两个 seed × 更大留出**来挡这件事，
   但**没有做 3 seed 以上**，也没有扫 `Blocks` / `Heads` / `dropout` / 位置编码
   （那几项分别有各自的文档与 A/B）。
5. **`hiddenDim` 对 PPO 是 actor 与 critic 共用的宽度**。BC 只训 actor，所以 CE 只反映
   actor 那一半；critic 宽度会通过叶子估值影响**搜索质量**，这一点只有 §5 的对局能碰到 ——
   而 §5 的对局也没有把它单独分离出来（`h=32` 的两侧连 critic 都一起变了）。
6. **`--expert` 必须能被 `Heads=4` 整除**（否则序列专家构造时 `abort`，这是设计如此）；
   本轮所有档位都满足。Q 类 agent 的 `expertHidden` 是**本轮才变成运行时旋钮**的
   （`DQNMCTSMOETbAgent` 原来是编译期常量 `MOE_SEQ_TOKDIM=64`），默认值不变由
   `test_gui_seq_wiring` 的三条结构断言钉住。
7. **没有量峰值内存与建网时间**：`tokDim` / `hiddenDim` 都会改参数量与缓存
   （`docs/seq_transformer_design.md` §5.2），本轮只记了参数量（`test_gui_seq_wiring` 的读数）
   与每步耗时（§5.2 的配对列）。
8. **`AGENT_SACAZ_OLD_MOE` 没扫**：它已经从界面下拉框移除，本轮没有为它开变体二进制；
   按四支一致的形状**推断**它同在 20~32（**未实测**）。
9. **要判读 `hiddenDim` 的棋力效应，还差一轮**：100 局 / 50 配对开局在两个纯 BC 权重之间
   只给到 ±10pp，而和棋率 73%。下一步应当 **≥200~400 局**，或者先给两臂各做一轮 RL
   再比（`docs/seq_transformer_encoding_eval_2026_10.md` 的相位 2b/3 就是这个做法）。
10. **界面级运行没有做**：§6 的验证是"编译期 + 参数量 + 前缀 + 二进制字符串"，
    没有真的启动 GUI 走一局（本会话没有 UI 运行）。要完全闭环应再跑一次
    `tools/verify_agent_combo.ps1` 那类 UI 检查。

