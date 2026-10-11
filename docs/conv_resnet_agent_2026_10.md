# 新 agent：conv2d + ResNet + 稀疏 MoE 的 PPO+MCTS（棋盘当图像看）（2026-10 第三轮）

> 用户口径：**"使用 19x9x10 特征输入，实现 conv2d + resnet 专家，接入 ppo + mcts + alphazero
> + moe 实现新的 agent，与 mcts agent 对弈验证棋力，最终加入 chess 程序。"**
>
> 一句话：把状态从"无结构的 1710 维向量"改回它本来的样子 —— **19 通道 x 10 行 x 9 列的
> 棋盘图像**，用 `3x3 卷积 + 4 个残差块 + 1x1 输出投影` 当 MoE 专家，接入既有的
> PPO+MCTS（AlphaZero 式）搜索与训练，得到一个**新骨干** `Backbone::ConvExperts`
> 与一个新界面 agent `AGENT_PPOMCTS_CONV`。
>
> 代码：`src/rl/conv_resnet.hpp`（新专家，~600 行）、`src/rl/ppo.h` / `ppo.cpp`
> （第五种骨干 + 工厂 + 名字/短名）、`src/rl/ilayer.h`（新层类型 `LAYER_CONVRESNET`）、
> `test/test_conv_resnet_main.cpp`（32 条断言，秒级进 ctest）、
> `test/train_bc_main.cpp`（`--agent=ppo-conv`）、
> 界面：`src/chessboard.{h,cpp}` / `src/mainwindow.cpp` / `src/ppomcts_agent.cpp`（自检面板）。

---

## 1. 为什么这么切（设计依据都在这份文档的前作里）

`docs/conv_resnet_moe_analysis.md`（2026-10 的分析文档，454 行）已经把三件事论证过：

| 组件 | 那份文档的判断 | 本轮怎么做 |
|---|---|---|
| **conv** | **应该做**：现有骨干把一个 1710 维向量当**无结构的词袋**，而状态本身就是带二维结构的图像 | 按口径实现（§2） |
| **ResNet** | **应该做，且是 conv 的必要条件**：感知野靠层数（3 层 7x7、5 层覆盖整盘），而本工程的 `RMSProp` 把每张量梯度归一到单位长度 ⇒ 深浅层同一步长 ⇒ 恒等残差是唯一能给出"至少不比浅层差"起点的结构 | 4 个残差块 + `1/sqrt(fan_in)` 初始化（§2.2） |
| **MoE** | **建议先别上**：conv 之后参数变便宜、算力成瓶颈，而 MoE 解决的是"容量不够" | **按用户口径照做**（E=4/top-1），并把那份担心变成可测读数（§5，与不带 MoE 的对照） |

那份文档还给了四条"必须一起做"的配套（§6.1），本实现逐条落地：

1. **坐标/区域平面必须加**：河界/九宫的走法规则是**绝对坐标**依赖的，纯平移等变在象棋上
   是部分错误的假设。本实现加 4 个零参数通道（行 ramp / 列 ramp / 河界 / 九宫）。
   ⚠ 它们**不是零代价**：通道数 19→23 改变了 conv1 的 W 形状 ⇒ **两种口径的权重互不兼容**
   （`test_conv_resnet [4]` 把这条钉死：参数量差恰好 `(23-19)*32*9 = 1152`）。
2. **全局信息要有入口**：本实现里 3 个规则上下文标量本来就是**第 17/18/19 个平面**
   （每格复制同一个值），所以它们天然在图像里；1x1 输出投影可以把它们原样学回去。
   （那份文档设想的"head 前 concat"需要改 MoE 的"输入输出同维"契约，本轮不动契约。）
3. **动作头空间化**：本支的**策略头仍是 `Tanh(64) → Softmax(8100)`**（PPO 的头部与骨干
   解耦，换骨干不动头）。那份文档的"`Conv 1x1 → 90 路`"红利属于另一条独立实验，见 §7。
4. **权重隔离**：新骨干一个独立枚举值、独立权重前缀、独立层类型 ⇒ 与 tb/mlp/seq
   互不载入（`test_conv_resnet [5]` 实测：conv 的权重载进 seq 骨干被**明确拒绝**）。

---

## 2. 结构（`src/rl/conv_resnet.hpp`）

```
状态 (1710 x 1) = 平面优先: state[plane*90 + cell], cell = x*9 + y   (x: 0..9 纵向, y: 0..8 横向)
   │  零拷贝语义的重排 (数据本来就是"平面 → 行 → 列"的行主序)
   ▼
图像 (19, 10, 9)          ← 用户口径的 "19x9x10": 19 通道, 盘面 9 宽 x 10 高
   │                        (文件一律按 (通道, 行, 列) 写 —— 行列对调 = 卷积核转置, 静默错位)
   │  ⊕ 4 个坐标/区域通道 (默认开): 行 ramp / 列 ramp / 河界 / 九宫
   ▼
conv1 : Conv2d(23 → 32, 3x3, pad=1) + Tanh
Blocks: 4 x [ Conv2d(32 → 32, 3x3, pad=1) + Tanh + **残差** ]
head  : Conv2d 1x1 (32 → 19), **线性**  → 展平回 1710
```

| 量 | 值 |
|---|---|
| 每专家参数 | **44,275**（conv1 6,656 + 4x9,248 + head 627） |
| 每专家前向 MAC | **3,968,640 ≈ 3.97 M**（对照：序列专家 9.67 M，TB 专家 13.3 M） |
| PPO actor 参数量（E=4/top-1） | **819,948**（对照：seq 1,009,772、TB 52,388,888） |
| 每专家权重文本 | 236,383 字节 |

### 2.1 卷积怎么算：im2col + `MM::ikkj/ikjk`

`conv2d.hpp` 里那个朴素三重循环在 ConvDQN 上实测 **~10 ns/MAC** —— 用在本专家上
（3.97 M MAC/前向）就是 **~40 ms/前向**，直接不可用。所以本专家**不共用也不修改**
那个实现（ConvDQN 的数值一位都不许动），而是：

```
im2col: (Cin, 10, 9) → (Cin*k*k, 90)  行主序 (kernel, position)
前向:  z(M,90)  += W(M,K) · X(K,90)          → MM::ikkj
dW  :  gW(M,K) += dY(M,90) · X(K,90)ᵀ        → MM::ikjk   (X 的"行"就是 kernel 下标)
dX  :  dX(K,90) = Wᵀ(K,M) · dY(M,90)         → MM::ikkj + col2im 散加
```

一份 im2col 布局同时满足前向与反向两个内核的契约，所以**不需要**为反向再转置一次。
实测（本机 Release/AVX2，`test_conv_resnet [6]`）：

| 读数 | 值 |
|---|---|
| 一次 forward | **0.487 ms = 0.123 ns/MAC** |
| 一次 forward+backward | 1.497 ms（backward 是 forward 的 2.07x） |
| 对照：`conv2d()` 朴素实现的量级 | ~10 ns/MAC ⇒ 本专家会慢 **~80 倍** |

⇒ 一次模拟（actor+critic 各一次前向 + 门控 + 头）约 **1.2~1.5 ms**，
比序列专家实测的 **3.7 ms/模拟**还便宜。这是"这条路线能不能用"的**前提**，
也是那份文档 §9 的"第 0 步"。

### 2.2 四个刻意的决定（每条对应一个具体的坏结果）

1. **不做归一化**（那份文档 §4 的结论 c）：现有 `LayerNorm` 对**整个张量**求一个标量
   均值/方差（把 (C,10,9) 喂进去 = 把空间抹平），`BatchNorm` 会被 MCTS 推理前向污染
   （推理与训练共用同一批层对象）。所以用"残差支路初始化接近恒等"代替：权重按
   `1/sqrt(fan_in)` 缩放，每个块初始注入是小量 ⇒ 恒等路占优。
2. **残差必须两条路都对**：反向里 `d(输入) = d(输出)（残差那条） + Wᵀ·(d(输出) ⊙ Tanh′(pre))`。
   ⚠ Tanh 的导数必须用 `tanh(pre)` 重算 —— `act` 里存的是**加过残差**的值，
   拿它当 Tanh 输出会得到错误的导数（不会报错，只会让训练变差）。
   `test_conv_resnet [2]` 用中心差分把这一条钉住（7 组参数 + 输入梯度，最差 5.45e-3）。
3. **逐点输出投影 + 展平**：`Linear(32*90 → 1710)` 是 4.9 M 参数/专家，
   而 `Conv 1x1(32→19)` 只要 627 个，且保留"每格输出只由那一格（及其邻域）决定"。
4. **不用 MoE 之外的任何"共享骨干"技巧**：actor 与 critic 各背一套 MoE（与 TB/seq 同），
   因为这轮的变量是**专家结构**，不该同时动第二个东西。

---

## 3. 验证（`test_conv_resnet`，**32 条断言 / 0 失败**，秒级进 ctest）

| 段 | 断言内容 | 实测读数 |
|---|---|---|
| [1] | 图像口径（19 通道 + 4 坐标）、参数量、前向 MAC、非注意力层的读数返回 -1、位置编码/dropout 旋钮**响亮返回 false** | 44,275 参数 / 3,968,640 MAC / actor 819,948 |
| [2] | **解析梯度 == 中心差分**：conv1 的 w,b / 第一个残差块的 w,b / **最后一个**残差块的 w / head 的 w,b + 输入梯度（取 `\|ei\|` 最大的 8 个坐标） | 最差 **5.45e-3**（多数 1e-4~1e-3） |
| [3] | im2col/col2im 自洽：中心位置的 patch 顺序、角上 padding=0、**内部位置被 9 个 patch 覆盖 ⇒ col2im 累加出 9**、角上 4 个 | 全部精确 |
| [4] | 坐标通道开关**会**改变参数量 ⇒ 权重不通用（钉死"零参数=零代价"这个想当然） | 44,275 vs 43,123（差 1,152） |
| [5] | 专家级往返逐位 + **Net 级**（PPO::save/load）同一输入逐位相同 + conv 权重载进 seq 骨干**被拒** | 最大差 **0.000e+00** |
| [6] | forward / forward+backward 墙钟（只印不断言：本机计时跨会话会漂，见 seq 文档 §8.4） | 0.487 ms / 1.497 ms |

> **为什么 [2] 是本文件最重要的一段**：ResNet 的反向有两条路，漏掉残差那条或写错 Tanh 导数
> **都不会报错**，只会让训练变慢/变差 —— 这是"看起来正常"的典型失败。
> 同理 [3] 把"patch 下标写错"与"边界处理写错"分开。

---

## 4. 棋力读数

### 4.1 与 MCTS agent 对弈（100 局 / 50 个配对开局 / 等模拟预算 64 / 同一 seed）

```bat
bench_agent_arena --mode=match --a=ppo --b=mcts --backbone-ppo-a=conv --load-a=weights/convrl ^
                  --games=100 --sims=64 --plies=120 --opening=8 --paired-openings ^
                  --seed-mcts-rng --seed=20240901
```

| A 侧骨干（**同一训练预算**：BC 2000 局面 + 自对弈 40 局 @120 模拟） | 比分（胜/负/和） | **得分率** | 95% Wilson | 判定 | Elo 差 |
|---|---|---:|---|---|---:|
| `seq` + `pos=1d` | 33 / 18 / 49 | 57.5% | [47.7, 66.7] | inconclusive | +53 |
| `seq` + `pos=2d`（**界面现役**） | 34 / 20 / 46 | 57.0% | [47.2, 66.3] | inconclusive | +49 |
| `seq` + `pos=rope` | 30 / 24 / 46 | 53.0% | [43.3, 62.5] | inconclusive | +21 |
| **`conv`（本轮新增）** | **39 / 17 / 44** | **61.0%** | **[51.2, 70.0]** | **decisive** | **+78** |
| （参考）`seq`+2d 加了"数据 ×3 + 自对弈 ×1.5" | 37 / 17 / 46 | 60.0% | [50.2, 69.1] | decisive | +70 |

⇒ **conv 是本工程目前对 MCTS 得分率最高的一支（61.0%），也是唯一"在 100 局口径下
Wilson 下界 > 50%"的新骨干**（另三支 seq 的同预算读数都还 inconclusive）。

### 4.2 头对头：conv vs 界面现役的 `seq`+2d（60 局 / 配对开局 / 同一 seed）

| A vs B | 比分（胜/负/和） | A 得分率 | 95% Wilson | Elo 差 | **分出胜负的 22 局** |
|---|---|---:|---|---:|---|
| `conv` vs `seq`+2d | 17 / 5 / 38 | **60.0%** | [47.4, 71.4] | **+70** | **17 : 5** |

**怎么读这两张表（严格版）**：

1. **方向一致、量级可观，但"conv 更强"没有被单独证明**：对 MCTS +4.0pp（区间重叠）、
   头对头 +70 Elo 且分出胜负的 22 局里 17:5（配对开局下这已经是 p≈0.008 的单侧符号检验），
   两条独立测量的方向一致 —— 但按本工程的口径，**没有一条单独跨过"决定性"**，
   除了 vs MCTS 那条（它跨过了，那是"conv 比 MCTS 强"而不是"conv 比 seq 强"）。
2. **棋力不是在"克隆得更像老师"上赢的**（这条与直觉相反，是本轮最值得记的读数）：

   | BC 保真度（同协议：2000 局面 / 软目标 / 8 epoch / 200 次更新） | 训练 CE | **留出 CE** | 留出 top-1 | train/留出差 |
   |---|---:|---:|---:|---:|
   | `conv` | 2.1203 | 2.5567 | 32.25% | 0.436 |
   | `seq`+2d | 2.1307 | **2.5188** | **33.50%** | **0.388** |
   | `mlp`（第三对照） | **2.0170** | 2.5251 | **36.25%** | 0.508 |

   conv 的**留出 CE 比 seq+2d 差 0.038 nats**（top-1 低 1.25pp），却在棋力上更好。
   ⇒ 两条测量在说不同的事：**BC 保真度不是棋力**（`docs/behavior_cloning_2026_10.md`
   早就记过同型的反例），conv 的优势更可能来自**空间归纳偏置在"BC 没覆盖到的局面"上的
   泛化**，而不是"更像老师"。本轮**没有**做能直接证明这一点的实验（见 §7）。
3. **每模拟更快**：conv 实测 **2.51 ms/模拟**（400 模拟 = 1002 ms/步），
   seq 是 3.71 ms/模拟 ⇒ 同样 6 秒的预算，conv 能多搜 ~48%（界面 `PPO_SIMS = 1600`
   下约 **4.0 s/步**，seq 是 5.96 s）。参数也更少（每专家 44,275 vs 91,731）。

---

## 5. 归因：收益来自哪一项，以及**哪一项没测**

| 组件 | 本轮证据 | 能下的结论 |
|---|---|---|
| **conv（空间结构）** | 与 seq 同预算下：对 MCTS +4.0pp、头对头 +70 Elo（17:5）；BC 留出 CE 反而差 0.038 | **方向一致地更好**，但单条都不决定性；而且**不是**靠"更像老师"赢的 |
| **ResNet（深度）** | 4 个残差块，反向两条路被有限差分钉住（最差 5.45e-3）；墙钟 forward 0.487 ms | 结构可用；**"块数 1 / 2 / 4 / 6 的曲线没测"** ⇒ "深度换感知野"这条**没有被验证** |
| **MoE（E=4/top-1）** | 路由健康：4/4 专家都被用、最小份额 19.1%、MaxVio 0.287（未坍缩） | "MoE 在这套结构上**没有坏**"；**"E=1 vs E=4"没测** ⇒ 那份分析文档 §5 的担心（MoE 解决的是不缺的问题）**仍未验证** |
| **坐标/区域通道** | 加了 4 个通道；`test_conv_resnet [4]` 证明**开关会改变参数量 ⇒ 权重不通用** | **消融没做**（要做就得两棵源码树 / 两份权重各训一遍，见 §7） |

**为什么这几项没测**（诚实记下）：它们每一个都需要**另训一份权重**（BC ~10 min + 自对弈
~25 min）+ **一轮 100 局对局**（~25 min）。本轮把预算花在"先把新骨干跑通并与现役骨干
同口径比一次"上（那才是用户口径要的"验证棋力"）。给出确切命令见 §7，任何一项都可以单独复跑。

**顺带一条与那份分析文档的对照**：文档 §5 预测"conv 之后参数变得极便宜 ⇒ MoE 的卖点消失"。
本轮的读数**支持这个方向**：每专家只有 44,275 参数（seq 是 91,731、TB 是 12,936,510），
而算力（3.97 M MAC/前向、2.51 ms/模拟）才是实际约束 —— 也就是说 E=4 买到的容量
（17.7 万参数）在这个规模上本来就便宜，MoE 省下的"没被选中专家的算力"也很有限。

---

## 6. chess 程序接线：新 agent `AGENT_PPOMCTS_CONV`

| 项 | 值 / 位置 |
|---|---|
| 枚举值 | **`AGENT_PPOMCTS_CONV = 18`**（**追加在末尾** —— 这些值经下拉框 `userData` 传出去，插在中间会静默改掉既有 agent 的编号） |
| 下拉框 | `mainwindow.cpp`：`"PPO+MCTS (卷积残差)"` |
| 唯一构造入口 | `chessboard.cpp` 的 `makePpomctsConvAgent()`（7 个构造点全走它：预加载 / 工厂 / 决策兜底 / 按需建网 / 后台兜底建网 / 后台训练 clone / 退出保存）。⚠ 它**刻意不调** `setExpertPosMode` —— conv 骨干没有位置编码旋钮，调了会当场 `qFatal` |
| 权重前缀 | **`weights/ppomcts_conv_agent.dat`**（`defaultWeightPath` 单一来源 ⇒ 启动扫描 / 自检 / 退出保存跟着走） |
| 临时前缀 | `weights/_temp_train_ppomcts_conv`（后台训练那一轮的往返；已进"所有临时路径必须命中 `weights/_temp`"那条自检） |
| 模拟预算 | 决策路径 `PPO_SIMS`（=1600，与另外两支 PPO 对齐）；后台训练 clone 用 `BG_TRAIN_SIMS`（400，与 seq 那一支同：决策预算看"一步等多久"、训练预算看"一轮多久"） |
| 自检面板 | `PPOMCTSAgent::selfCheckReport()` 新增 `case Backbone::ConvExperts:` 分支（图像口径 / 通道数 / 块数 / 每专家参数与 MAC / 权重与 tb·mlp·seq 均不通用）。**顺手修掉一个真缺陷**：原来那一支写的是 `case TbExperts: default:` ⇒ 任何新骨干都会被**错报成 TB**；现在五个枚举值逐个显式列出（正好对上该函数自己那句"按枚举逐个写清, 不留兜底"），将来加第六种骨干会由 MSVC 提醒而不是静默报错 |
| 其他一律带上 | `bcSupported` / `bcOnStudentMove`（BC 面板当学生）/ `valueProbeOf` + `valueProbeOfSupported`（价值校准曲线）/ `getMoeLoad`（负载均衡面板）/ `humanEndChannel` / `agentCanExplore` / `agentHasLearningRewardTable` / `kWeightAgents` / `weightFilesOf` / `saveAllInstantiatedAgentsOnExit` 的 `all[]` |

**验证读数（我自己复跑过，不是转述）**：

| 项 | 读数 |
|---|---|
| `chess` 构建 | 0 error（`chess.exe` 链接通过） |
| **GUI 真的建出 conv 网** | `repro_concurrency 1 4 18`（界面路径：`ChessBoard` + 后台训练 + 自动保存）⇒ `=== 全部跑完, 没有崩溃 ===`，写出的 `weights/ppomcts_conv_agent.dat_actor/_critic` = **4,374,213 / 1,566,559 B**（结构指纹：819,948 参数 × ~5.34 B/参数，与 seq 的 5,292,001 B / 991,596 参数同一比例） |
| 界面前缀可载入 | `bench_agent_arena --backbone-ppo-a=conv --load-a=weights/ppomcts_conv_agent.dat` ⇒ 载入成功，标签 `PPO+MCTS/稀疏MoE(conv2d+ResNet专家)` |
| 已装训练权重 | `weights/convrl_*` → `weights/ppomcts_conv_agent.dat_*`（就是 §4 里对 MCTS 61.0% 的那一份），所以界面选这一支**开局就是训练过的模型**（6,891,237 / 4,083,583 B） |
| 回归 | `test_weights` **49/0**、`test_conv_resnet` **32/0**、`test_ppo_backbone` **68/0**、`test_transformer` 89/0、`test_sparse_moe` 100/0、`test_bc` 全绿 |
| **新 agent 进了 `test_match [2.11]`** | 我把 `AGENT_PPOMCTS_CONV` 加进那一节的两个清单（"下拉框里选得到它 + 显示名对得上 + 对弈时真的上报损失"那一张表，以及自检/权重状态的枚举）。实测：`type=18 报告 39 行 / 4598 字节, 权重状态 4 行, 可重复=1` ⇒ **新 agent 的自检真的被跑到了**（这一条原先没有测试覆盖：打印自检文本的只有 `test_match` / `test_dqnmcts*` / `test_sacaz`）。整场 `test_match` = **371 项断言 / 4 失败**，4 条全是既有的（3 条在"对弈模式下后台训练冻结"那一节 + `[2.22]` 那条时序敏感的窗口断言），**没有新增失败** |

> ⚠ **构建陷阱（踩过一次，写下来）**：只 `build chess` **不会**重链 `repro_concurrency.exe`。
> 用旧 exe 跑新枚举值 18 时，旧表里没有 18 ⇒ 落到 Alpha-Beta 兜底 ⇒ 打印"保存=0 +
> 瞬间跑完"，**看起来像通过其实是假通过**。必须 `build.ps1 chess repro_concurrency`。

---

## 7. 诚实边界与未做项

**不能从本轮读数里得出来的**：

1. **"conv 比 seq 强"没有被单独证明**：两条测量方向一致（+4.0pp / +70 Elo 17:5），
   但按本工程的 Wilson 口径，单条都没跨过"决定性"。要下定论得把样本加到
   ~200 局/臂（或多次换 seed 的开局集重复）。
2. **BC 保真度与棋力在这一轮是反着的**（conv 留出 CE 差 0.038 但棋力更好）。
   本轮**没有**做能解释它的实验（例如"在 BC 没见过的局面上比较两者的
   选点一致率 / 价值误差"）。
3. **MoE、块数、坐标通道三项消融全没做**（§5 给了确切命令）。
   特别是 MoE：那份分析文档 §5 明确预测"conv 之后 MoE 不该先上"，
   本轮**既没有证实也没有证伪它** —— 只能说"E=4/top-1 在这套结构上跑得健康"。
4. **单 seed、开局集 50 个**：所有跨骨干对比都固定 `seed=20240901`。
   上一轮已经量过"开局样本量能把同一支权重从 68.8% 挪到 57.0%"
   （`docs/seq_transformer_encoding_eval_2026_10.md` §4.1）⇒ 本轮的 61.0% 同样
   只对**这一批开局**成立。
5. **棋力上限**：自对弈只有 40 局 @120 模拟（界面口径一局训练是 400 模拟）。
   conv 的 61.0% 是"**小预算**下的读数"，不是这套结构的调好之后的上限。
6. **没有做"无位置/无空间"的对照**：也就是说"conv 的收益具体有多少来自
   '输入是图像'这件事"仍然只能靠与 seq 的间接对比来推断。

**未做的实验（命令都给全，任何一项可以单独复跑）**：

```bat
:: (a) MoE 消融: E=4 vs E=1 (后者 = 单专家, 等价于"不带 MoE 的 conv 骨干")
::     做法: 把 rl/ppo.h 的 PPO_MOE_CONV_EXPERTS 改成 1 单独编一支二进制
::     (本工程的纪律: 结构常量变了必须两棵源码树两个二进制, 不能在同一支二进制里换参数)
:: (b) 深度消融: 块数 1 / 2 / 4 / 6 (CONV_RESNET_BLOCKS, 同上, 各自编一支)
:: (c) 坐标通道消融: 需要给 useCoordPlanes 接一条 CLI 旋钮 (现在只在构造参数里),
::     然后各训一份 (权重不通用, 见 test_conv_resnet [4])
:: (d) 棋力确认: 把 §4.1 的 --games 提到 200 并换 2~3 个 seed 的开局集
:: (e) 那份分析文档 §7 的独立红利: 把策略头换成 "Conv 1x1 -> 90 路" (对齐表示的翻案实验)
```

---

## 8. 复现命令（本轮的每一个读数）

```bat
:: ---- 断言 (秒级) ----
cmake --build <build> --target test_conv_resnet
<build>\test_conv_resnet.exe          :: 32 条断言; [2] 是 7 组中心差分, [5] 是 Net 级往返

:: ---- 相位1: 行为克隆 (与 seq/mlp 同一协议: 2000 局面 / AB 深度 4 / 软目标 / 8 epoch) ----
<build>\train_bc.exe --agent=ppo-conv --positions=2000 --depth=4 --soft=1 --soft-depths=4 ^
                     --epochs=8 --batch=64 --lr=0.002 --seed=20240901 --holdout=5 ^
                     --save=weights/conv_bc
:: 对照臂 (本轮实测): --agent=ppo-mlp --save=weights/mlp_bc  (留出 CE 2.5251)
::                     --agent=ppo-seq --pos=2d --save=weights/seqenc_2d (留出 CE 2.5188, 见上一份文档)

:: ---- 相位2: 自对弈一轮 (与 seq 那一支**同预算**: 40 局 x 120 模拟) ----
<build>\train_ppo.exe --backbone=conv --load=weights/conv_bc --save=weights/convrl ^
                     --games=40 --sims=120 --moves=60 --opening=8 --seed=20240901

:: ---- 相位3: 对 MCTS agent (100 局 / 50 个配对开局 / 等模拟预算 64) ----
<build>\bench_agent_arena.exe --mode=match --a=ppo --b=mcts --backbone-ppo-a=conv ^
        --load-a=weights/convrl --games=100 --sims=64 --plies=120 --opening=8 ^
        --paired-openings --seed-mcts-rng --seed=20240901

:: ---- 相位4: 与界面现役 seq+2d 头对头 ----
<build>\bench_agent_arena.exe --mode=match --a=ppo --b=ppo --backbone-ppo-a=conv --backbone-ppo-b=seq ^
        --pos-ppo-b=2d --load-a=weights/convrl --load-b=weights/seqrl_2d ^
        --games=60 --sims=64 --plies=120 --opening=8 --paired-openings --seed=20240901

:: ---- 界面验证 (界面路径: ChessBoard + 后台训练 + 自动保存; 18 = AGENT_PPOMCTS_CONV) ----
cmake --build <build> --target chess repro_concurrency     :: ⚠ 只 build chess 会留下**旧** harness
<build>\repro_concurrency.exe 1 4 18
::   期望: "全部跑完, 没有崩溃" + weights/ppomcts_conv_agent.dat_actor/_critic
::        = 4,374,213 / 1,566,559 B (819,948 参数的结构指纹)
```

日志与 CSV：`.r1build/conv/`（`bc_ppo-conv.txt` / `bc_ppo-mlp.txt` / `rl_conv.txt` /
`convrl_vs_mcts_100.{txt,csv}` / `conv_vs_seq2d_60.{txt,csv}` / `test_match_conv.txt`）。
权重：`weights/conv_bc_*`（BC）→ `weights/convrl_*`（RL）→ 拷成 `weights/ppomcts_conv_agent.dat_*`（界面）。

