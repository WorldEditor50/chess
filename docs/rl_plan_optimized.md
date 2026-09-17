# 象棋自对弈训练：三轮建议的合并与优化（2026-09）

本文把三轮讨论（过强对手/信心崩塌、和棋率、更新器选型与回放容量）合并成一份可执行方案，
每条都标了**落点文件**、**验收指标**与**代价**，并显式列出**已实现**与**不做**的项。

所有结论都基于当前代码，引用格式为 `文件:行`。

---

## Part 0 · 三个前提（先纠正，再谈方案）

### 0.1 "被过强对手打崩信心"在当前仓库里没有通道

- PPO+MCTS 的学习入口只有一处：后台训练线程的 `clone.trainSelfPlay(...)`（`src/chessboard.cpp:1996`）。
  对局/评测路径里没有 `commitEpisode` / `learnFromReplay` / `beginOnline` / `endOnline` 的调用；
  `src/chessboard.cpp:1149` 注释自己写着在线路径"GUI 从不调用"。
- 与 AB 的对弈全在评测路径：`test/bench_ppo_vs_ab_main.cpp:5-11`（"不写权重文件，只在控制台输出每局结果"）。
- 全仓无对手池/league：`docs/agent_dqnab_design.md:478`、`docs/training_optimization.md:916` 均标为"未做"。

⇒ 观察到的"缩头、不吃子、只走一两手"必须先排除下面三个更近的嫌疑人，
而不是归因于"被强敌教怂了"。

### 0.2 和棋的大头不是"规则纵容"，是训练台架的 60 ply 截断

| 事实 | 证据 |
|---|---|
| 训练一局上限 60 **ply**（=30 回合），`for (moveNum = 0; moveNum < maxMoves; ...)` 一次迭代 = 1 ply | `src/chessboard.cpp:167,1421` |
| 自然限着要求 `halfMoveClock >= 120`（60 回合）—— **训练里不可达** | `src/chess.cpp:1056`、`src/stone.h:841` |
| 判和只有两类：三次重复、60 ply 截断（后者走 critic 自举） | `src/chess.cpp:1046-1060`、`src/ppomcts_agent.cpp:1741-1753` |
| 无长将/长捉判罚（`src/chess.h:138` 的"长将检测"注释名不副实，`isRepetition` 只数重复） | `src/chess.cpp:1024-1044` |

⇒ §"和棋分桶"里"50 回合无吃子"这一桶恒为 0；"三次重复"与"长将循环"在当前裁判下是**同一事件**。

### 0.3 更新器之争在本仓库不存在；回放窗口被吞吐量卡死

- `RL::PPO::accumulateGrad` = `CE(π_visit, π_net)` + `MSE(v, target)`，无 ratio / clip / advantage / entropy（`src/rl/ppo.cpp:212-264`）；
  `src/rl/diag.h:327-333` 自己写明 `clipFraction = -1` 且"对当前管线不适用"。
  ⇒ 本仓库的 "PPO+MCTS" **就是 AZ-CE 蒸馏**。
- 吞吐：400 sims × ~8 ms ≈ 3.2 s/ply → **192 s/局**（`src/chessboard.cpp:88-92`）。
- 存储：`STATE_DIM = 1710` 稠密 float ≈ **6.8 KB/条**（`src/ppomcts_agent.h:103`、`src/rl/ppo.h:219-243`）；
  每 ply 还额外写入一条镜像增广样本（`src/ppomcts_agent.cpp:909`），即 **120 条/局**。
- 现有池：`replayCapacity = 20000` 条 ≈ **167 局 ≈ 9 小时**自对弈（`src/rl/ppo.h:243`）。
- 每轮训练还有 **3 趟整模型磁盘往返**（各 ≈555 MB：actor 279 MB + critic 276 MB）
  —— 种子写盘（`src/chessboard.cpp:1915`）→ clone 读入（`:1991`）→ clone 写回（`:1999`）→ 主 agent 读入（`:2055`）。

---

## Part 1 · 逐条：原建议 → 优化后

### 1.1 过强对手 / 信心崩塌（第一轮）

| 原建议 | 结论 | 优化后 |
|---|---|---|
| 通道 A：`z=−1` 把 V 压到地平线 | **机制不符**：价值目标是 negamax 塑形折现回报（`src/rl/ppo.cpp:528-604`）+ PBRS 平移 `V'=V+Φ`（`src/stone.h:885`），不是裸 z；且 Φ 使**所有 V 读数带偏移** | 先做 Φ 归零口径的校准读数；真病根是"规范视角编码里没有对手信息"（identifiability），不是 target shaping |
| 通道 B：MCTS 发现"吃子=死更快" | 部分成立但归因错：低 sims 时目标已退化（`src/chessboard.cpp:150-164`），且 Q 排序被 Φ 的消项误差带偏 | 用固定局面集 + `qCaptureMax − qQuietMax` / `captureVisitShare` 量（`src/rl/diag.h:293-300`），并做 PBRS A/B |
| 通道 C：PPO clip 把 logp 猛推 → entropy 归零 | **不存在**：无 ratio/clip/advantage | 改写为 `priorKl→0 且 policyEntropy 降` = π 回声正反馈；解药是 PCR + 目标不确定性加权 |
| ① 对手池比例硬约束（数字） | 数字自相矛盾（池内 `ab_depth6`、`-20k` 与"25–75% 区间"互斥；`latest 0.35 < 旧快照 0.40`） | 限"单对手占 buffer 比例"而非强度；用**让子/算力拉平**保持对局势均（PFSP 的权重逻辑见 P2.1） |
| ② 暴打局按结果降权 | **删**：对价值目标做选择偏差，critic 变乐观 → 之后送子 | 按**冗余/陈旧**降权（死和局面的重复 ply、buffer age），且需先给 `ReplaySample` 加 `weight` 字段 |
| ③ `λz+(1−λ)rootQ` 软 target | **删**：root Q 来自正在被训的同一张网（自指），且压小目标方差 | 换成**独立来源的稠密监督**：AB 分值 `tanh(score/2)`（`test/bench_ppo_distill_main.cpp` 已验证只训 critic 可行） |
| ④ Dirichlet + 高温"永远开" | 方向对，但噪声只在前 30 ply（`src/ppomcts_agent.h:236`），温度已 1.0→0.1 线性退火（`src/ppomcts_agent.cpp:1421-1425`） | 中局缺的是**局面级**多样性（随机开局/残局种子），不是根节点噪声 |
| ⑤ Exploiter 最短杀目标 | 保留，但会**提高**暴打率，且其目标函数不得进入主模型的 value target | 与"拉平胜负"配对；exploiter 数据只用于训练它自己 |
| ⑥ 冻 MoE 专家防遗忘 | **改**：router 无对手/阶段输入，指望路由自学是空想；只训防守专家 = 把缩头策略局部化 | 要条件化就**显式喂对手特征**（或分池 adapter + value 头），别靠冻结专家 |

### 1.2 和棋率（第二轮）

| 原建议 | 结论 | 优化后 |
|---|---|---|
| 严格 WXF 判罚（长将/长捉/一将一捉） | 长将是**真通道**；长捉例外太多，手写 detector 误判不可观测 | 只做**长将判负**（P1.4），且必须同步加状态平面；长捉暂缓 |
| 搜索里"再走这手触发长将负"→ `V=−1`、`N 永远不给` | **错**：mask 会让该动作的 CE 目标恒 0，先验单调衰减 | 当作**终局**处理（`getResult` 判负 → `evaluateLeaf` 自然给 ∓1，`src/ppomcts_agent.cpp:583-596`） |
| 假 komi：`draw_value = clamp(mat_diff*0.05, ±0.25)` | **符号反了**："多子方和棋为正"= 鼓励兑现和棋；且材质是手写 eval | 若要做：优势方和棋必须是**负值**、幅度 ≤0.1、**搜索与目标两处同改**、并接进截断自举（P2.4） |
| PBRS 加"胜势紧迫感" | 这是在改 Φ 本身，会同时改 ABAgent/EVAB 预训练标签 | 先 A/B：`Chess::setPositionalEvalEnabled`（`src/chess.h:132-136`） |
| `r_step = −0.001` | **已实现**：`REWARD_STEP_COST`（`src/stone.h:840`），连"为何不能取 −0.005"都推过 | 不动 |
| MCTS 节点 Zobrist 去重 → 命中即标 0 | **回退**：搜索走真棋盘真历史 + `getResult`，路径内三次重复已经正确算终局 0 | 不动 |
| 温度前 50 步 τ=1.0 别早降 | **已满足**（1.0→0.1 线性） | 不动；加大 τ 只会让棋更乱、和更多 |
| 残局课程（3000~10000 种子） | **ROI 最高**，但随机摆子会混入白脸将/无胜法局面 | 用 ABAgent 深度 4~6 过滤"确有胜势"的合法非终局种子；指标 = **conversion rate**（P1.5） |
| 和棋/暴打局 buffer 降权 | **删**（同一选择偏差） | 见 1.1 ② |
| Draw-Hater Exploiter 进主 buffer | **有害**：目标函数不同 → 主模型价值目标条件在"对手在下另一盘棋" | 只当**评测对手**用 |

### 1.3 更新器与回放（第三轮）

| 原建议 | 结论 | 优化后 |
|---|---|---|
| "AZ-CE 最强，PPO 次之，DQN 不推荐" | AZ-CE 与 PPO 在本仓库是同一行；缺的第四行是 **SAC+AZ** | 真问题是"要不要在 CE 之上加 IS/clip"；判据 = `priorKl` 与 `policyEntropy` 的稳定性 |
| "PPO 才能控 entropy / 好接 reward" | entropy 对偶你已有（`convpg/dpg/drpg` 的 `alpha.g[k] += H − entropy0`；`src/sacazagent.h:219`）；reward 进的是 value target，与更新器正交 | 删掉这条理由 |
| DQN "符号容易写错" | 已是 negamax + `clampValue`（`src/dqnabagent.cpp:623,642,678`），BCQ 用 Clipped Double Q（`src/rl/bcq.cpp:65`） | 真实风险是**语义不一致**：max-Q 训的是最优博弈值，MCTS 叶子取的是期望 → 乐观被记两次 |
| DQN "样本效率低" | 排序有误：AZ 的效率来自"每个 ply 一条搜索监督样本"，不是回放 | 修正归因，避免推出"回放越大越好" |
| "和棋 z=0 压扁信号是 DQN 问题" | 不是：`test_reward_diag` 在 AZ 线实测过 `|target|>0.1` 占比 0%（`src/stone.h:870-873`） | 解药是 value target 本身（PBRS/稠密监督），已做 |
| 回放 50k~200k 局 / 5M~30M 条 / SQLite+weight | **吞吐与存储都不成立** | 见 Part 3 数字表；窗口改按**网络更新步数**定义 |
| "PPO 必须存 logp_old，最多重放 1~2 代" | 正确；但你的 `ReplaySample` 无 logp（`src/rl/ppo.h:224-238`），**已经在安全的那一半** | 加 clip 的代价 = 现有 2 万条池基本作废（P2.3） |
| "老数据只走 CE/value、新局走 clip" | 你现在**已经是**这一半 | 无需改；只有当 CE 的信任域不够时再加 |
| 三个 staleness scalar | `CE on replay vs fresh`、`corr(z_stored, V_net)` 有用；`clip fraction` 现在恒为 −1 | 前两个接进 P0.4 的 CSV 报告 |
| "评估 1600 sims" | 12.8 s/步 → 一局 ≈12.8 分钟；且与训练 400 口径不一致 | 训练/评估保持等 sims 或走 `--budget=MS` 等时间（`test/bench_ppo_vs_ab_main.cpp:17-20`） |
| "按 LibTorch / SQLite 接进去" | 不存在/不适用 | 无外部 DL 框架（手写 `RL::Tensor`/`Net`/`RMSProp`）；`src/gamedb.cpp` 是 GUI 棋局库，写接口全仓零调用且连接跨线程（`docs/issues_review.md:982`），诊断刻意选 CSV（`docs/training_optimization.md:918`） |

---

## Part 2 · 优化后的执行路线

### P0 · 把"能量出来 + 跑得动"建起来（1~2 天，零算法风险）

**P0.1 无界面常驻训练器**
- 动作：单进程连跑 K 局，删掉每局的 3 趟整模型磁盘往返（`weights/_temp_train.dat` 路径链）。
- 落点：新增 `test/train_ppo_main.cpp`（沿用 `bench_*` 的"不进 ctest"惯例），复用 `PPOMCTSAgent::trainSelfPlay`；
  GUI 的 clone 路径保留为展示用途。
- 验收：`games/hour` 提升（I/O 2.2 GB/局 → 0）、可中断续跑、日志含 Part 4 的全部读数。
- 代价：无（不动算法）。

**P0.2 和棋原因码 + 分桶**
- 动作：`getResult()` 区分 `DRAW_REPEAT / DRAW_NOMOVE60 / TRUNCATION / MATE`；`GameStat` 加 `drawReason`。
- 落点：`src/chess.h/.cpp`、`src/rl/diag.h:340-347`、`test/test_rules_main.cpp` 补判例。
- 验收：300 局自对弈的 **plies 直方图 + 原因占比**；决定 P1.4/P2.4 是否值得做。
- 代价：极小；但**这是所有后续判断的前提**（裁判改错在损失曲线上不可观测）。

**P0.3 固定锚点评测**
- 动作：`--anchors=<snap...>,ab2,ab4`；固定开局集（同一批 20~40 个局面、双方各下一次、成对计分）；
  输出 Elo 差 + 置信区间；区分"真怂 vs 真变强"。
- 落点：扩 `test/bench_ppo_vs_ab_main.cpp` 或新增 `bench_anchor_main.cpp`（复用 `--opening/--budget/--load`）。
- 验收：分辨 ~80 Elo（≈61.5% 得分率）的置信区间；**没有它，任何"回滚/降权"决策都不可证伪**。
- 代价：快照 ≈555 MB/个（actor+critic），锚点集控制 ≤20 个 ≈11 GB。

**P0.4 一次跑完的 CSV 报告**
- 动作：把已有诊断接成单一报告：`visitEntropy / priorEntropy / priorKl / topShare / coverage /
  captureVisitShare / qCaptureMax−qQuietMax / captureChosen / policyEntropy / valueEv / moeLoadCv`（`src/rl/diag.h`）
  + `CalibBucket` 校准（**先减 Φ**）。
- 落点：复用 `RL::Diag::scalarRow` 与 `bench_diag --csv=PREFIX` 的既有通路。
- 验收：**三读数**——① 固定局面集上的吃子率；② `π_visit` 选点 vs `π_prior` 一致率（priorKl）；
  ③ 减 Φ 后的价值校准 + EV。
  - 搜索找得到好着、prior 找不到 ⇒ 蒸馏问题（改 loss / 加 PCR）。
  - 搜索自己也找不到 ⇒ 价值/搜索问题。

### P1 · 直接对症（1~2 周）

**P1.1 PBRS A/B（最高优先，因为它可能是"不敢吃子"的真因）**
- 动作：`potentialShaping=false` / `shapingAlpha ∈ {0, 0.25, 0.5, 1}`，固定局面集上量 capture share 与 `qCap−qQuiet`。
- 依据：Φ 每步贡献最大 ≈(1+γ)|Φ| ≈ **1.99**，压过终局 ±1（`src/stone.h:819-821` 的"终局 > 一方全材质 2.9×"只在**未塑形**口径下成立）；
  `V'=V+Φ` 使所有 V 读数带偏移（`src/stone.h:885`）。
- 验收：若关掉塑形后吃子率显著回升 ⇒ 缩头来自塑形边界项，不是任何对手。

**P1.2 价值头的独立稠密监督**
- 动作：把 AB 根分值 `tanh(score/2)`（与 Φ 同尺度，`test/bench_ppo_distill_main.cpp:14-18`）作为**训练期辅助损失**接进 critic，
  权重小、只作用于中局样本。
- 落点：`RL::PPO::accumulateGrad/accumulateGradSparse` 加可选 aux 项。
- 理由：替代"软 target / root Q 混合"这类自指做法。

**P1.3 PCR + 目标不确定性加权（KataGo 的两条）**
- 动作：只让"深挖充分"的 ply 进 policy 目标（判据现成：`visitCount`、`topShare`、`coverage`）；
  目标按根值方差 / visit 熵加权。
- 依据：`src/chessboard.cpp:150-164` 已证明低 sims 的目标会退化成"把先验抹平"；
  KataGo 论文（arXiv:1902.10565）针对"playout 太低时 policy target 质量迅速崩坏"给的正是这两条。
- 验收：`priorKl` 上升（搜索真正纠正先验）、`policyEntropy` 不塌、锚点 Elo 上升。

**P1.4 长将判负（唯一的规则改动，必须与状态平面同批做）**
- 动作：`getResult()` 在"第三次重复且同一方在无吃子窗口内每步都将军"时判**将军方负**（绝对结果，`outcomeForMover` 换算链同步改）；
  **同时**加 1 个状态平面（我连续将军 N 手 / 对手连续将军我 N 手），否则规则部分可观测。
- 落点：`src/chess.cpp`、`src/chessstate.h`、`src/ppomcts_agent.h:99-103`、`test/test_rules_main.cpp`（判例）。
- 代价：`STATE_DIM` 1710 → 1800，**所有已存权重作废**（`Net::load` 会拒绝架构不匹配），需重跑 distill/BC 种子。
- 验收：P0.2 的原因码里 `DRAW_REPEAT` 占比下降多少（用数字回答"砍半"是否成立）。

**P1.5 残局课程 + conversion rate**
- 动作：AB 深度 4~6 过滤出的"确有胜势"残局种子（合法、非终局、双方均有合法着法），按 1/10 局混入；
  指标 = **优势残局 → 实际取胜的转化率**（当前全仓无此指标）。
- 落点：`PPOMCTSAgent::trainSelfPlay` 加"起始局面"参数（现在写死 `chess.reset()`，`src/ppomcts_agent.cpp:1406`），
  三个调用点（`src/chessboard.cpp:1965,1980,1996,2012`）同步改。
- 理由：唯一直接提高 z=±1 密度的措施；且"不会赢"才是和棋的底层原因。

### P2 · 有需要再做（2~4 周）

**P2.1 对手池 / league（若真要做）**
- PFSP 权重 `(1−winrate)^p`（不要按 Elo 限流）；用**让子 / 算力**把对局拉回势均，保留强敌的战术价值；
  单对手占 buffer 设上限；exploiter 的目标函数不进主 buffer 的 value target。
- 若要让主模型跟异质对手学，必须给网络**对手特征**（或分池 value 头），否则价值目标不可辨识。

**P2.2 回放扩展**
- 先按**网络更新步数**定义窗口（现状 2 万条 ≈167 局 ≈9 h 的尺度是对的）；
  存储改稀疏（`src/rl/ppo.h:240-242` 已注明这是后续项）之后再谈 10^5~10^6 量级。
- **不加按结果的权重**；要加优先级就按陈旧度 + 状态冗余。

**P2.3 更新器升级（可选，最后考虑）**
- 触发条件：P1.3 之后 `priorKl` 仍大、`policyEntropy` 仍不稳。
- 代价：`ReplaySample` 要加 `logpOld`（`src/rl/ppo.h:224-238`）；回放窗压到 1~2 代 ⇒ **现有 2 万条池基本作废**。

**P2.4 假 komi（最后考虑）**
- 触发条件：P1.5 之后，"优势方兑现和棋"仍占大头。
- 约束：优势方和棋值为**负**、幅度 ≤0.1、`evaluateLeaf`（`src/ppomcts_agent.cpp:588-589`）与 buffer 目标**同改**、
  并接进截断自举（`src/ppomcts_agent.cpp:1752`）。

---

## Part 3 · 数字表（按当前吞吐重算）

| 量 | 数值 | 出处 |
|---|---|---|
| 一步决策 | 400 sims × ~8 ms ≈ **3.2 s** | `src/chessboard.cpp:88-92` |
| 一局自对弈 | 60 ply ≈ **192 s（3.2 min）** | `src/chessboard.cpp:167` |
| 日产能（24/7） | ≈ **450 局/天** | 计算 |
| 一条样本存储 | 1710 维稠密 float ≈ **6.8 KB** | `src/ppomcts_agent.h:103` |
| 每局样本数 | 120 条（含镜像增广） | `src/ppomcts_agent.cpp:909` |
| 现有池 2 万条 | ≈ **167 局 ≈ 9 小时** | `src/rl/ppo.h:243` |
| 5,000 局 | ≈ 11 天 | 计算 |
| 50,000 局 | ≈ **111 天** | 计算 |
| 200,000 局 | ≈ **444 天** | 计算 |
| 5M 条（≈4.2 万局） | ≈ 92 天 / **34 GB** | 计算 |
| 30M 条（≈25 万局） | ≈ 555 天（1.5 年）/ **205 GB** | 计算 |
| 每轮训练额外 I/O | 3 趟 ×555 MB ≈ **2.2 GB/局** | `src/chessboard.cpp:1915,1991,1999,2055` |
| 快照成本 | ≈ **555 MB/个**（actor+critic） | `weights/ppo_bc_d4_actor` = 279 MB |
| 评估 1600 sims | 12.8 s/步 → 一局 ≈ 12.8 分钟 | 计算 |

**结论**：窗口只能按"网络更新步数 / 参数漂移"定义；任何以"局数"为单位的大数字（5 万~20 万局）
在本机上是数月到一年多，不具备可操作性。

---

## Part 4 · 明确不做

| 不做 | 理由 |
|---|---|
| 和棋 / 暴打局按结果降权 | 对价值目标的选择偏差（三轮里第三次出现）：critic 只在部分结果分布上校准 |
| MCTS Zobrist"历史命中即标 0" | 回退：真历史 + `getResult` 已正确处理搜索路径内三次重复 |
| 假 komi（按子力差给正和棋值） | 符号与目的相反；且是手写 eval 进终局目标，与 `evaluateLeaf` 口径分裂 |
| 长捉 / 一将一捉判负 | 规则例外多，误判不可观测，会教出一个错误的游戏 |
| DQN 的 max-Q TD 作主 target | 语义与 MCTS 平均叶值不一致，乐观被记两次（`DQNAB` 的 AB planning head 路线可用） |
| 把 RL 回放接进 SQLite | `src/gamedb.cpp` 是 GUI 棋局库：写接口零调用、连接跨线程（`docs/issues_review.md:982`），诊断刻意选 CSV |
| 按 LibTorch 写 loss 伪代码 | 无外部 DL 框架：手写 `RL::Tensor`/`Net`/`RMSProp`/`Loss::CrossEntropy::df` |
| 评估 sims 提到 1600+ | 12.8 s/步；且与训练 400 口径不一致，破坏可比性 |
| "冻结部分 MoE 专家防遗忘" | router 无对手/阶段输入；只训防守专家 = 把缩头策略局部化，trunk 仍漂 |

---

## Part 5 · 验收标准（"变好了"的定义）

1. **P0.2 原因码**：和棋里截断 / 重复 / 限着各占多少（先有数，再谈改）。
2. **固定局面集吃子率** + `qCaptureMax − qQuietMax`：该吃的大子还吃不吃。
3. **priorKl 与 π_visit/π_prior 一致率**：搜索还在纠正先验，还是只剩回声。
4. **减 Φ 后的价值校准 + explained variance**：领先局面不再被标成地狱。
5. **conversion rate**：优势残局 → 取胜的比例（"会不会赢"的直接读数）。
6. **锚点 Elo（带置信区间）**：对固定快照集 + 固定 AB 深度，分辨 ~80 Elo 的能力。

---

## Part 6 · 实施记录（P0 已落地，2026-09）

### 6.1 改了什么

| 项 | 落点 | 说明 |
|---|---|---|
| **P0.2 和棋原因码** | `src/chess.h` / `src/chess.cpp` | 新增 `Chess::DrawReason { DRAW_NONE, DRAW_REPEAT, DRAW_NO_CAPTURE60 }`；`isDraw(DrawReason*)` / `getResult(int, DrawReason*)` 用**默认参数**重载，旧调用点全部不变。**不是和棋时也写 DRAW_NONE**（否则会把上一局的原因残留到这一局） |
| **P0.4 结束方式分桶** | `src/rl/diag.h` | 新增 `GameEndKind { END_MATE, END_DRAW_REPEAT, END_DRAW_NO_CAPTURE60, END_TRUNCATED }`；`GameStat` 加 `endKind` / `drawReason`；`Aggregates` 加分桶计数 + `endMateRate()` / `drawRepeatRate()` / `drawNoCapture60Rate()` / `truncatedRate()` / `truncationShareOfDraws()` / `maxPlies()`，仪表盘新增一段"结束方式"并直接给出判读（截断占比 >0.5 就打印"先动课程/台架，不是动裁判"）。未填 `endKind` 的旧调用方不进任何一桶（`gamesClassified()` 会暴露"谁还没接上"） |
| **P0.1 常驻训练器** | `test/train_ppo_main.cpp`（新）+ `src/ppomcts_agent.h/.cpp` | 训练器一个进程连跑 K 局，**每局零磁盘往返**。为此在 agent 上加了两个最小钩子（默认值下行为逐位不变）：`std::vector<RL::Diag::GameStat> *gameLog`（每局追加一条统计，三个终局出口都接上了）与 `openingPlies` / `openingSeed`（每局起点随机化，**同时修掉了"随机开局后 currentColor 必须跟着棋盘走"** 这个错帧隐患） |
| **P0.3 锚点评测** | `test/bench_anchor_main.cpp`（新） | 固定开局集（局部 `mt19937_64` + FNV 指纹，**不碰 `RL::Random`**，所以换权重跑的是同一批局面）+ 换先手成对计分 + 得分率/Elo 的 95% 区间 + `--budget=MS` 等时间标定 + 和棋构成 |
| **P1.1 A/B 开关** | `test/train_ppo_main.cpp` | `--no-shaping` / `--shaping-alpha=F` / `--no-material-reward` / `--no-bootstrap` / `--no-root-noise`，外加减小骨干的 `--hidden=N --expert=N`（一次 A/B 要跑很多局面） |
| CMake | `CMakeLists.txt` | 新增 `train_ppo` / `bench_anchor` 两个目标（都**不进 ctest**，与 `bench_*` 同一惯例） |
| GUI 对局上限 | `src/mainwindow.cpp` | `gamesSpin` 的 `setRange(1, 100)` → **`setRange(1, 10000)`**（步进 10）。理由与 P0.3 同源：80 Elo ≈ 61.5% 得分率，用 4~100 局去分辨它是在噪声里读结论。`matchAgents` 本身只有下界钳制（`chessboard.cpp:1625`），没有上限假设，所以这条只是放开界面限制；对局期间按钮变"停止对弈"，可随时中止 |

### 6.2 怎么跑

```
cmake --build <build> --target train_ppo bench_anchor test_rules
<build>/train_ppo --games=20 --sims=400 --moves=60 --opening=8 --csv=run1
<build>/train_ppo --games=40 --sims=400 --opening=8 --no-shaping --csv=shaping_off   # P1.1 A/B
<build>/bench_anchor --a=weights/run_10k --b=weights/run_5k --openings=20 --plies=8
<build>/bench_anchor --a=weights/run_10k --ab-depth=4 --budget=100 --openings=20
```
（`<build>/` 是 Qt 的 bin 目录要在 `PATH` 里：`C:\Qt\6.8.0\msvc2022_64\bin`。）

### 6.3 实机验证（本机 Release / MSVC 2022 / Qt 6.8.0）

| 验证 | 结果 |
|---|---|
| `test_rules`（规则回归） | **108 断言 / 0 失败**（新增 7 条：三次重复→`DRAW_REPEAT`、120 半回合→`DRAW_NO_CAPTURE60`、分胜负与未判和时原因被清成 `DRAW_NONE`） |
| `train_ppo --games=2 --sims=12 --moves=60 --opening=6` | 跑通；两局都因 60 ply 上限截断，报告里 `截断 1.000`、并打印"和棋里属于台架截断的占比 1.000 → 大头是手数上限"。**这正是 P0.2/P0.4 要暴露的那件事**：此刻的"和棋多"是台架产物，不是规则问题 |
| 同一次运行的产物 | `p0_train_tb.csv`：逐局的 `game_end_*` 一行一个 plus `sum_*` 汇总行（`sum_truncation_share_of_draws = 1.0`） |
| `train_ppo --hidden=16 --expert=16 --no-shaping --shaping-alpha=0.25` | 跑通，配置行确认 `塑形: 关 (alpha=0.25)`；小骨干下 **1.3 s/局（≈2800 局/小时）**，P1.1 的 A/B 扫描因此可行 |
| `bench_anchor --openings=3 --plies=4 --sims=8 --ab-depth=2 --max-plies=16` | 跑通：指纹 `702FE5AB91BE6490`、成对计分、得分率 0.4167、Elo −58.5 **区间 [−174.9, +46.0] 跨过 50%** → 明确打印"没测出差别"（4~6 局就是分不出来，这正是旧 bench 会让人误判的地方） |
| 编译警告 | 新代码 **0 警告**（`/W3` 下仅剩 `rl/util.hpp:65`、`rl/tensor.hpp:558` 两条既有警告） |

### 6.4 还没做（下一步）

* **P0.3 的锚点集**：现在要手动传 `--a/--b`；把权重按固定命名（如 `weights/snap_<step>`）存一排、并把锚点清单写死进脚本，才能成为"阶梯"。
* **P1.2** critic 的独立稠密监督（AB 分值辅助损失）—— 落点 `RL::PPO::accumulateGrad*`。
* **P1.3** PCR + 目标不确定性加权（只让深挖充分的 ply 进 policy 目标）。
* **P1.4** 长将判负（必须与新增状态平面同批做；`STATE_DIM` 1710→1800，**所有已存权重作废**）。
* **P1.5** 残局课程 + conversion rate（`trainSelfPlay` 的起点参数已由 `openingPlies` 打开了一半，残局种子还需要"从给定局面起手"的入口）。
* **注意**：`train_ppo` 报告里"根/行为/训练"三行是 0 —— 那是 `bench_diag` 的职责（它自己驱动 `selectMove` 采 `RootDiag`/`MoveBehavior`）。训练路径没有"决策前的根"可供事后读取，这一层刻意不重复实现。

