# chess 程序问题审查报告

审查对象：`E:\home\code\chess`（`src/` + `test/` + `CMakeLists.txt`）。
方法：逐行静态审查 + 实际构建与运行验证（Release / MSVC 2022 14.44 / Qt 6.9.2 / CMake 3.30）
+ 运行时插桩（ASan、有限差分梯度检查、真实窗口的像素/无障碍树采样）。

> 行号对应当前工作区版本。`src/rl` 已同步到 snakeAI 的上游版本，
> 同步内容见 `docs/rl_sync.md`。

## 本文怎么读

| 章节 | 内容 |
|------|------|
| 零 | **修复进度总表**（A/B/C 三组编号，标"已修复/未修复"） |
| 零之二点五 | 修复后的实测结果（ctest、四个训练基准、搜索基准） |
| 零之二点七 | 决策前探索协议（仿 snakeAI）与 EVAB 接入 GUI |
| 零之二点八 | AI 思考过程可视化（长思考时"界面像死了"） |
| 零之二点九 | Agent 对弈 arena；**顺便挖出的堆损坏**（ASan 定位全过程） |
| 零之二点十 | 做奖励曲线时挖出的符号 bug（即时奖励一直是黑方视角） |
| 零之二点十一 | EVAB 的三个 bug（"与 AB 对弈无法取胜"的根因） |
| **零之二点十二** | **启动时加载所有模型（含为它做的加载提速）+ 奖励曲线改成每手一个点 + 验证方法学的五个坑 + 顺手抓到的"每场多挂两条空线"** |
| 零之二点十三 | PPO 系列训练效率改造（P1–P7）：稀疏 MoE、批平均损失、梯度累积、回放池、镜像增广、多线程分身；**暴露出的多线程访存墙（~1.5×）** |
| 零之二点十四 | 训练流程优化（Phase 0–5）暴露的问题：奖励量纲、自举、PBRS、两档评估、展开按先验；**方法论产出：造尺子 / 功效检验 / 先量噪声带** |
| 零之二点十五 | **R1.5**：内核语义（`ikjk`/`kijk` 累加）、反向 GEMV（**13×**）、`MM` 形状契约断言（抓到 `lstm.cpp` 一处越约调用）、DQN Q 头 |
| 零之二点十六 | **B-5**：置换表 + 子树复用（机制成立、**决策中性**） |
| 零之二点十七 | **R2**：训练侧也只算合法列（换学习问题，Z ≡ 1）+ **c_puct 重扫（扫不出落点）** |
| **零之二点十九** | **PPO+MCTS 正确性审计（2026-09）**：PUCT 的 Q 符号反了（选择器在最大化**对手**价值）、`loadModel` 静默成功（检查点全部失效却报"载入成功"，**连本轮自己的测量都被污染**）、`Result`/`Color` 枚举混比（胜率记账错位）、`BG_TRAIN_SIMS=20` 结构性退化（**零深挖**）、搜索从不判终局叶子（**一步杀 0/20**）；外加根 Dirichlet 噪声 / `selectMove` 温度 / 截断局自举三项补齐 |
| **零之二点二十** | **诊断指标矩阵（2026-09）**：`src/rl/diag.h`（无 Qt 依赖的指标核心）+ `rootDiag()` + `bench_diag`（自动战术题库 / Dirichlet 有效性 A/B）+ `test_diag`（68 断言，进 ctest）；**上线当天就抓出"搜索看不见将杀"**（0/20 → 20/20）；并量出三条会**让诊断说谎**的坑 |
| **零之二点二十一** | **P0 基础设施（2026-09）**：和棋**分原因**（三次重复 / 自然限着 / **台架截断**）+ 无界面**常驻**训练器 `train_ppo`（每局零磁盘往返）+ 固定开局集锚点评测 `bench_anchor`（成对计分 / Elo 置信区间 / `--budget` 等时间）；**第一次运行就报出"训练里的和棋几乎全是 60 ply 台架截断、自然限着根本不可达"**；并逐条核清两条易被引引的事实（本管线的 actor 是纯 CE 蒸馏、没有 clip；价值目标是 negamax 塑形回报、`V'=V+Φ` 使校准读数必须先减 Φ） |
| **零之二点二十二** | **DQN+MCTS 的表示闸门 + 自检进界面（2026-09）**：`probe_dqnmcts_aliasing` 把三个**与训练量无关**的结构事实量成数字（动作别名 44→38 槽位；走子方/重复/限着**逐字节不可观测**；6 局 × 200 手**终局信号 0 条**），并回答"回放 4096→20000 会不会更快"（**不会**，虽然去重率 99% 证明容量装的是新信息）；同时修掉 `trainVsRandom` 里"每一手都被写成 −1.0f"的错、把三处收尾统一到 `getResult()`、给 `test_dqnmcts` 补上它的**第一批断言（16 条）**，并把读数接进界面"模型自检"面板。详见本文"零之二点二十二"一节 |
| **零之二点十八** | **各会话（轮次）× 问题 × 优化方法 汇总 + 方法论沉淀**（本文的总索引） |
| 零之三 | 修复前的实测结果（作为对照基线） |
| **零之四** | **优化方法汇总（含实测数字）** —— 按手法组织，回答"哪个优化值得做、做完还剩什么瓶颈"<br>4.1 计算内核 / 4.2 构建与工具链 / 4.3 算法与交互 / 4.4 验证手法的沉淀 |
| 一 / 二 / 三 | 逐条问题的详细分析（A = 致命/高，B = 中/低）。**C 组的详细分析在"零之二点七 ~ 零之二点九"**，C7/C8（SIMD 相关）另见 `docs/rl_sync.md` |
| 四 | 与 `docs/analysis.md` §六 的核对（历史判断） |
| **五** | **当前仍未做的事 + 具体修法 + 优先级** |

编号约定：**A/B 是首轮审查**（A1–A15 / B1–B19），**C 是第三轮**
（Agent 对弈 / 思考可视化 / 探索预训练期间新发现的，C1–C8）。
已经在"修复进度"表里标"已修复"的条目，在下面各节里仍保留完整分析
（失效场景、证据、修法）—— 那是判断"真的修对了"的依据，不是待办。

---

## 零、修复进度总表

> A/B 为首轮审查的问题编号，C 为第三轮（Agent 对弈 / 思考可视化 / 探索预训练）。
> 每条后面括号里指向本文中给出完整分析的章节。

| 编号 | 问题 | 状态 |
|------|------|------|
| A1 | `StoneMap` 未初始化 + `Chess` 不清 `m_map`（野指针 / `test_ab` 段错误根因） | **已修复**：`StoneMap()` 调用 `clear()`；`Chess::Chess()` 末尾调用 `reset()` |
| A2 | 走法不做合法性过滤（自杀 / 不应将 / 照面），`isInCheck()` 零调用 | **已修复**：新增 `Chess::isAttacked/isLegalMove/hasLegalMoves`，`sample()` 只返回合法走法，GUI 落子也走 `isLegalMove()` |
| A3 | 没有将杀 / 困毙判定 | **已修复**：新增 `Chess::getResult(colorToMove)`（含将杀/困毙/和棋） |
| A4 | 重复局面 / 长将 / 60 回合全部未实现，`pushHistory()` 零调用 | **已修复**：`moveForward` 维护 `history`（含走棋方哈希）与 `halfMoveClock`；`isDraw()` 实现三次重复 + 120 半回合；搜索里重复局面返回 0 分 |
| A5 | 回放模式进入后无法退出 | **已修复**：`reset()` 清 `m_replayGameId` 并 `emit replayModeExited()` |
| A6 | 两个 detached 线程与对象生命周期脱钩 | **已修复**：改为成员 `std::thread`，`~MainWindow` 先 join 再 `delete ui` |
| A7 | `condit.wakeAll()` 未持锁；棋盘读写无锁 | **已修复**：`wakeAll()` 统一在 `QMutexLocker` 内；AI 落子与 `getResult` 在锁内执行；工作线程不再直接 `update()` |
| A8 | `Steps` 无锁全局对象池 | **已修复**：`get()/put()` 加 `std::mutex` |
| A9 | RL 即时奖励恒为 0（吃子信号丢失） | **已修复**：`computeReward` 不再检查 `victim->alive`（与调用时机解耦），4 个 agent 一致 |
| A10 | "无合法走法"哨兵用错常量 | **已修复**：`Step::valid` 取代 16 处 `id == ID_NONE` / `pos == (0,0)` 判断；`Step::operator=` 补上漏掉的 `reward`/`valid` |
| A11 | GUI 里 DQN+MCTS 实际随机走子 | **已修复**：`selectMove(color, 200, false)`（`training=false`） |
| A12 | 后台训练线程与 AI 线程竞争同一 agent | **已修复**：`aiThink`/`aiThinkForAgent` 的 RL 分支加 `m_agentMutex` |
| A13 | 关窗阻塞在训练线程 join 上 | **已修复**：单轮规模从 4 局×200 步×50 模拟降到 `BG_TRAIN_EPISODES/MAX_MOVES/SIMS` |
| A14 | SQLite 跨线程使用；写库接口未接线 | **未修复**（见下文，需要设计决定） |
| A15 | 在线训练接口 GUI 侧未调用 | **已修复**：每个 agent 实现 `exploreAndTrain()`，GUI 侧由 `ChessBoard::preTrainThenDecide()` 在走子前调用（见"零之二点七"） |
| B1 | `moveForward` 忽略 `moveTo()` 返回值导致棋盘写坏 | **已修复**：`applyMove/undoMove` 拆分；失败不记账；`moveBack` 只在确实执行过时回退 |
| B2/B3 | `reset()` 不清回放/历史状态、可与 AI 思考重叠 | **已修复**：`reset()` 加锁并清回放状态 |
| B4 | 只有 MAX 节点做静态搜索 | **已修复**：两处叶节点统一走 `quiescenceBlackView()` |
| B5 | 静态搜索丢父窗口导致不剪枝 | **已修复**：父边界换算到走棋方视角后传入 |
| B7 | 搜索里没有"重复局面 = 和棋 0 分" | **已修复** |
| B8 | 深度/迭代数注释与真实参数三处不一致 | **已修复**：集中为 `AB_DEPTH/MCTS_SIMS/PPO_SIMS/DQNMCTS_ITERATIONS` 常量 |
| B9 | 回放把列坐标当棋子 id | **已修复**：抽出 `applyDbStep()`，三处重复代码合一 |
| B10 | `getStonePos` 负数除法向零截断 | **已修复**：改 `std::floor` |
| B13 | `selfPlay` 从不返回平局 | **已修复**：返回 `Chess::RESULT_*` |
| B14 | `test_main.cpp` 的 static agent 悬垂引用 | **已修复**：改为每步新建 agent |
| B19 | `Tensor::MM` 只有 0.11 GFLOP/s | **已修复**：内层循环改为扁平指针 + 把 stride 提到循环外，实测 **190 倍**（见下） |

> **第三轮（Agent 对弈 / 思考可视化 / 探索预训练）新发现的问题编号为 C**，
> 详细定位过程与实测见"零之二点七 ~ 零之二点九"，优化手法汇总见"零之四"。

| 编号 | 问题 | 状态 |
|------|------|------|
| **C1** | **AI 工作线程被"虚假唤醒"**：条件变量写成 `if` 而不是 `while`，醒来后也不重新检查 `state` → `reset()` / `matchAgents()` 的 `wakeAll()` 把**空闲** worker 叫醒，它照跑 `aiThink()`，与另一个线程**同时改同一个 `env`** | **已修复**：`while (state != THINKING && state != TERMINATE)` 循环等待 + 进搜索前重新确认状态。ASan 定位到 `env.history` 的 double-free（见"零之二点九"） |
| **C2** | `!step.valid` 被直接当成"走棋方被将死" → agent 抽风返回无效走法时，对局在**一步都没落子**的情况下判胜负，之后棋盘再也点不动 | **已修复**：先分清"真无棋可走"与"agent 出错"；后者用第一个合法走法兜底 + stderr 报告 + 计入 `MatchStats::agentErrors`（`process()` 与 arena 两处） |
| **C3** | `applyMove/undoMove` 用外部传入的 `Step` 直接索引 `std::array` 与 `StoneMap::data[10][9]`，无任何边界校验 → 一个坏字段就是越界写，落在相邻的 `env` 上 | **已修复**：校验 `id ∈ [0,32)`、`pos/nextPos` 在棋盘内、棋子存活且与 `Step` 起点一致，否则返回 `false` |
| **C4** | 思考途中按"开局"：`reset()` 只把 `state` 改回 IDLE 就返回，工作线程算完**照样把子落到重置后的新棋盘**上 | **已修复**：`m_thinkGeneration`，`reset()` 在棋盘锁内 `+1`，`process()` 在同一个锁内比对，代数变了整步作废 |
| **C5** | `exploreAndTrain` 把调用方的 `color` 与棋盘 `sideToMove` 混为一谈，探索结束把 `sideToMove` 留在 rollout 末尾 | **已修复**：`rolloutFromCurrent` 与 `EVABAgent::exploreAndTrain` 入口保存/出口恢复；`test_pretrain` 用"连续 5 次探索后棋盘逐字段不变"守住 |
| **C6** | `shutdownSave()` 漏了 EVAB：界面上可选、能被在线训练的 agent 退出时从不落盘，每次启动都从随机价值网重来 | **已修复**：补 `weights/evab_agent.dat` |
| **C7** | `Tensor::MM::ikjk/kijk` 的 **SIMD 内核是赋值、标量是累加**（语义不一致）。当前调用点 kdim=1 走标量所以梯度没错，但多处注释明写 `+=`，`lstm.cpp:143-148` 更是一次累加到同一个 `delta.h` —— **一旦做批量训练（kdim ≥ 8）会静默只保留最后一次的贡献** | **未修复（潜在）**：内核改 `z[...] += dot(...)`。见"零之四"与 `rl_sync.md` |
| **C8** | `kikj`（反向 GEMV `ei = wᵀ·e`）**没有** SIMD 快速路径，掉回标量循环：实测 **0.991** vs 前向 `ikkj` 的 **0.102** ns/MAC（**9.7×**）—— 训练步卡在没被优化的那一半 | **未修复（性能）**：补一个与 `gemv_ikkj` 对称的 `gemv_kikj`。见"零之四" |
| **C9** | **奖励曲线在对局过程中一动不动**：`gameRewardSample` 一局只 emit 一次，而一局可能几百手、跑十几分钟（实测 `SAC+AZ-MoE vs Alpha-Beta` **276 手 / 620.8 秒**）→ 用户看到的就是"对弈时奖励曲线没有更新" | **已修复**：新增 `matchRewardProgress`，**每手**一个"本局累计"点（走子方视角），局末再由 `gameRewardSample` 补一个含终局 ±1 的点；读数前缀改成"奖励(局内累计)"。见"零之二点十二"；`test_match` [2.8] 9 条断言 + `verify_match_ui.ps1` 的"点数在涨"和"点数 ≥ 手数 − 局数"两条钉住 |
| **C10** | `playMatchGame` 的 `rewardA/rewardB` 出参是**死参数**（传进去从没被写过），A/B 换算在 `matchAgents` 里另算了一遍 → 两处规则一旦分叉就是"曲线上的点和逐局明细里的奖励对不上" | **已修复**：`playMatchGame` 增加 `bool aIsRed` 参数，红/黑 → A/B 的换算集中在 `syncAB()` 一处，出参与每手进度信号共用同一份结果 |
| **C11** | **验证方法学**：检查本身会给出错误结论（本轮真实发生 5 次）：后台保存没写完就断言 → 假失败；启动加载的日志冒充保存日志 → **对局没打完也报通过**；改脚本丢 BOM → 报一堆和原因无关的语法错；拿"本来就不该出现的东西"当失败（AB vs AB 时没有可训练 agent，本来就没有损失曲线和存权重行）；`Select-Object -First N` **掐断编译管道** → 拿旧 exe 验证并认真分析了假现象 | **已修复并记录**：见 `docs/agents_design.md` **16.3**（五个坑）；`verify_match_ui.ps1` 加 `$expectSave` 做适用性判断、按前缀剥壳再比、构建输出改走文件并核对 exe 时间戳 |
| **C12** | **每跑一场对弈就往奖励曲线上多挂两条线**：`resetMetricsForMatch` 用的是 `clearData()`（只清**点**、不清线）+ `addSeries()`×2 → 第 N 场图上有 2N 条线（2N−2 条空的），读数标签变成"SAC+AZ-MoE: 暂无 \| Alpha-Beta: 暂无 \| SAC+AZ-MoE: 最新 ... \| Alpha-Beta: ..."，导出的 CSV 也多出几列同名空数据。"清空曲线"按钮同一类问题（清空后同一个 agent 再上报会**新建一条同名线**） | **已修复**：新增 `CurveChart::removeAllSeries()`（连名字/颜色/数据一起清），换线处与"清空曲线"都改用它；`clearData()` 的语义保持不变（放大窗口同步要用）。`test_match` **[2.9]** 10 条断言钉住两个语义 + "只 clearData 再加两条会变成 4 条"（把 bug 形状写成断言）；`verify_match_ui.ps1 -TwoMatches` 在真界面里连打两场，断言第二场之后读数**仍然只有两条线且没有"暂无"**。见 §零之二点十二 与 `docs/agents_design.md` 13.8 |

> **第四轮（PPO+MCTS 正确性审计与诊断指标矩阵，2026-09）新发现的问题编号为 C13 起**，
> 详细定位过程、实测数字与修法见"零之二点十九 / 零之二点二十"。

| 编号 | 问题 | 状态 |
|------|------|------|
| **C13** | **PUCT 的 Q 项符号反了**：节点里存的是**该节点走棋方**视角的累计价值（backup 逐层翻号 + 价值头按走子方训练，`test_ppomcts` 的 discounted-return 一节把这条钉成断言），而子节点的走棋方**就是父节点的对手** —— 所以父节点比较时必须取负号。`getPUCT` 用的是 `q = child.getQ()` 直接相加取最大 ⇒ **搜索在最大化对手的价值**，专挑对自己最差的着法，而且**评估越准错得越狠**。旁证：`sacazagent.h:173` 的字段注释白纸黑字写着"累计价值 (当前走棋方视角)"，说明意图如此，只是消费端漏了。手算最小树：根为红方，A 走后红将死黑（A 的 Q = −1）、B 走后黑将死红（B 的 Q = +1），红方该选 A，漏负号会选 **B** | **已修复**：`PPOMCTSAgent::getPUCT` / `SACAZAgent::getPUCT` / `MCTSNode::getUCB1` 三处取负号；`test_ppomcts` 新增 "PUCT sign" 一节（两个方向都查，防"永远选第一个"的假通过）。**`dqnmcts_agent.cpp:221` 同类漏负号未修** —— 它的 `encodeState` 是固定红黑符号、无走棋方通道，叶子值是绝对视角却配交替 backup，**是真正的混帧**，在那里加一个负号是错的，需要单独设计 |
| **C14** | **`loadModel` 静默成功**：`RL::PPO::load` 原来是 `void`，把内核 `Net::load` 的 int 结果吞掉了；`PPOMCTSAgent::loadModel` 于是只按"文件可读"返回 `true`。实测现场：`bench_ppo_sims` 打印 **"A 成功 / B 成功"**，而 stderr 同时在喊 **"参数量不匹配 … 拒绝载入"**。对"每轮 保存→载入→训练"的训练循环这是最坏的失败形态：**静默退化成每次从随机权重重来**，一句报错都没有。**它连本轮的测量都污染了** —— 上一轮那次 A/B 两次都测出 2.5%，原因就是两次都载入失败、跑的其实是随机权重，结论完全无效 | **已修复**：`RL::PPO::save/load` 与 `RL::DQN::load` 改回真实返回值（**不短路**，两个网络都尝试；失败时明确警告可能处于 actor/critic 混态），`PPOMCTSAgent`/`DQNAgent`/`DQNMCTSAgent`/`SACAZAgent` 全部传播结果（`EVABAgent`/`DQNABAgent` 本来就是对的，作为参考写法）；`chessboard.cpp` 的后台训练往返改成**硬失败**（种子写不出去就跳过本轮、载入/写回失败就丢弃本轮、同步失败就 `qWarning`）；`test_ppomcts` 新增 "loadModel reports the REAL result" 一节（往返逐元素 0.000e+00 + 结构不匹配必须 false + 不存在路径必须 false）。**顺带确认：`weights/` 下所有检查点都与当前网络不兼容**（`bc8k_d4_actor` 2 150 124 元素 vs 当前 52 388 888）—— 即"续训"一直是静默地从随机权重重来 |
| **C15** | **`Result` 与 `Color` 枚举混比 → 胜率记账错位**：`Chess::Result` 是 `ONGOING=0, RED_WIN=1, BLACK_WIN=2, DRAW=3`，`Stone::Color` 是 `RED=0, BLACK=1, NONE=2`。`ppomcts_agent.cpp` 的两处统计写成 `if (gameResult == Stone::COLOR_BLACK) totalWins[1]++` ⇒ **红胜(1) 被记成黑胜**（1==1），黑胜(2) 谁都匹配不上，打印也把红胜显示成 "Black wins"。程序不会报任何错 | **已修复**：新增 `winnerOfResult(chessResult)`（`chess.h`，唯一的枚举换算入口），两处统计与打印改用它；`test_rules` 新增 **[11]** 段 7 条断言（四个映射 + 把"两个枚举数值撞号"本身写成断言，防后人再手写比较）。**注意**：审计报告起初把范围说大了 —— 使用局部变量 `winner`（本来就是 `Stone::Color`）的那几处是**正确**的，只有 `gameResult ==` 那两处是错的 |
| **C16** | **`BG_TRAIN_SIMS = 20` 是结构性退化（不是"少一点"）**：选择阶段只在"未展开列表为空"时才向下深挖，于是**前 #legal 次模拟只够把先验最高的那些孩子各展开一次**，`max(0, sims − #legal)` 次才是真正的深挖。象棋中局实测根节点平均 **38.7** 个合法着法，而训练只给 **20** 次 ⇒ **一次深挖都没有**。于是 π_target = "我自己先验前 20 名上各 1/20 的均匀分布"，出招再从这 20 个里采样 —— **搜索提供的信息量为零**，目标实际是个"把先验抹平"的算子。根节点诊断实测：20 次模拟下"根孩子数"恰好 **20.0**、深挖余量恰好 **0.0**（80 次 → 41.3，400 次 → 380.0）；20 vs 400 次模拟**有 85% 的决策不同**（即 20 次预算下的出招基本是任意的） | **已修复**：`BG_TRAIN_SIMS` 20 → **400**、`PPO_SIMS` 80 → **400**（保持训练/对局一致，符合"评估 sims ≥ 训练 sims"）。代价：当前 52.4 M 参数骨干下每步约 8 ms/模拟 ⇒ 单步 0.16 s → 约 3.2 s、一步决策 0.65 s → 3.2 s（关窗等待时间由单轮决定，`BG_TRAIN_EPISODES` 保持 1）。**注意 80 次时仍有约 48% 的访问落在"每个孩子一次"的地板里**（400 次降到约 10%） |
| **C17** | **搜索从不判终局叶子**：`selectMove` / `trainSelfPlay` / `warmupFromCurrent` 三处**无条件** `encodeState + ppo.value` 估叶子，从不调 `getResult`。于是"一步杀"那步落子后，叶子被交给一个只会**评估**的网络去猜 —— 它甚至不知道局面已经终局。诊断仪表盘实测：**一步杀命中 0/20 = 0%**，而非终局的"白吃子"有 16.7% —— 这个 0% vs 16.7% 的对比就是证据。`SACAZAgent` 一直有 `terminalValue()`（`sacazagent.cpp:663/698`），PPO 这条路径一直没有，两个 agent 的搜索口径本来就不一致 | **已修复**：新增 `PPOMCTSAgent::evaluateLeaf()` 作为三个入口的**唯一口径**（终局叶子按**叶子走棋方**给确定 ±1、判和给 0，非终局才走 critic）；`test_diag` **[7]** 段自动生成一步杀题并断言命中率 ≥ 50%。实测 **0/20 → 20/20**，战术总准确率 10.0% → **50.0%**；而**自对弈那几行读数一点没变**（Q(吃子)−Q(退让) 仍 −0.0998、吃子访问份额仍 0.136）⇒ 修复是**外科式**的，没有扰动自对弈分布 |
| **C18** | **根节点没有任何探索噪声；`selectMove` 的 `temp` 是死参数**：全仓 `Dirichlet` 零命中（内核里**连 Gamma/Dirichlet 采样器都没有**，这才是真正的原因 —— 不是不想要，是没工具）。而这套搜索的扩展开关是**确定性**的（按先验挑最大），于是低先验着法（典型是吃子）可能整局都不会被模拟一次。同时 `selectMove(color, sims, temp)` 的 `temp` **从头到尾没被读过**，无论传什么都走 argmax，与函数注释承诺的"temp > 0 时按访问分布采样"不符 | **已修复**：内核加 `RL::Random::gamma/dirichlet`（Marsaglia-Tsang + alpha<1 提升，零样本兜底成均匀分布，**绝不产生 NaN**）；`acquireRoot(color, withRootNoise)` 在 `P' = (1−eps)·P + eps·Dir(α)` 后接入，eps 按手数线性退火，**只在自对弈取数据时开**（评测/对局保持关，`evalRootNoise` 留作诊断 A/B）；出招逻辑抽成 `pickRootChildByVisits()` 供两处共用（采样数学与随机数调用顺序逐位不变）；截断局自举 `bootstrapOutcome()`（用 critic 估最后一步之后的局面、取负号换算到走子方视角，口径与在线路径 `endOnline` 一致）。`test_ppomcts` 新增两节：Dirichlet 采样器性质（和的偏差 ≤ 6.68e-08、alpha 语义）+ 根噪声接线（评测不开/自对弈开/合法集归一/30 手后自动关）+ 温度（`temp=1` 频率 0.721/0.185/0.094 对期望 0.727/0.182/0.091）。**但实测噪声救不了"怕吃子"**：吃子着访问份额 关 0.1263 vs 开 0.1225（差 **−0.0037**）—— 见"零之二点二十"的结论 |
| **C19** | **`ABAgent::findBestMove` 在"每一步都必输"时返回默认 `Step`**（`valid=false, id=0, pos=(0,0)`）：根节点窗口初值就是 `±Stone::value_infi`，而更新用严格不等号 ⇒ 所有根走法都等于 `±value_infi` 时一次都不更新，`best` 保持 `nullptr`。调用方把 `valid=false` 读成"真无棋可走" ⇒ **明明还有合法走法却判负**（用户报障 `[arena] agent(0) 返回无效走法 (第 6 局第 55 手): valid=0 id=0 pos=(0,0)->(0,0), 仍有 1 个合法走法, 已兜底`）。**同形状还有四处**：`MCTS` / `DQN+MCTS` / `PPO+MCTS` / `SAC+AZ` 的"根有合法走法但一个孩子都没展开"（`iterations/simulations ≤ 0`）也返回 `Step()` | **已修复**：AB 两处改 `best == nullptr \|\| ...`；四处搜索 agent 补"取根节点未展开列表的第一手"兜底；再加统一闸门 `ChessBoard::legalStepOrFallback()`（`aiThink`/`aiThinkForAgent` 出口，无效走法 + 还有合法走法 ⇒ 兜底 + 打印**带 agent 名字**的 `[gate]` 日志）。`test_match` **[2.10]** 用"红方怎么走都输"的构造局面钉住（修前必失败）。见"零之二点二十三" |
| **C20** | **`backgroundTrainLoop` 的三个 switch 都缺 EVAB/SAC+AZ/SAC+AZ-MoE/DQN+AB 的 case，却报成"种子权重写入失败"**：选这些 agent 时 `seeded` 恒为 false ⇒ 用户看到的 `[train] 种子权重写入失败, 跳过本轮训练: agent "EVAB" 路径 weights/_temp_train.dat` 把"这一支没接"说成了"写盘失败"，**排查方向完全错**（EVAB 是界面上可选、也确实有在线训练的 agent，等于它的后台训练一直是空转） | **已修复**：新增 `trainable` 判定把"接没接"与"写盘成不成功"分开报（未接的明说"后台训练尚未接入"，且每种 agent 只报一次）；**八个有权重可训的 agent 全部接上**（EVAB + SAC+AZ + SAC+AZ-MoE + DQN+AB），多文件家族各用独立临时前缀（PPO 的 `_actor` 与 SAC+AZ 的 `_actor` 不再同名）；新增 `setBackgroundTrainRound()` 让"一轮"可调；`test_match` **[2.12]/[2.13]** 逐个 agent 断言"跑完一轮并做了一次真实更新"（EVAB / SAC+AZ / SAC+AZ-MoE / DQN+AB 四条支路）。见"零之二点二十三" |
| **C22** | **后台训练的一轮可能"零更新"，而且不报错**：SAC+AZ / DQN+AB 的 `learnBatch` 在"回放池 < batchSize(32)"时**直接返回且不写 `m_lastLoss`**，而 clone 每轮都从空池开始 —— 所以一轮少于 32 手时它们一次梯度更新都不做，却照样把"没变过的权重"写回主 agent（损失曲线也不上报，因为还是 NaN）。是写[2.13]那条断言时踩出来的：给 6 手 → 三个 agent 全部"上报 0 次" | **已修复/已钉住**：`BG_TRAIN_MAX_MOVES` 的注释写明"必须 > 32"并给出两处门控的行号；`setBackgroundTrainRound()` 的文档同样写明"缩短轮次只适合验证接线"；断言把这条形状写进测试（不是改测试去迁就） |
| **C23** | **新 agent 加进了列表，界面上却"找不到"**：`QComboBox` 的 `maxVisibleItems` 默认是 **10**，而 agent 列表已经 11 项 —— 新加的 "PPO+MCTS (AlphaZero, 稀疏MoE+MLP专家)" 恰好排在第 11 位，于是它**在 model 里、能被测试与命令行创建、却折叠在下拉框的滚动区里看不到**（UIA 实测：展开只有 10 行可见）。这类"加了但看不见"最容易被误判成"没加"。顺带发现第二个隐患：`fillAgentCombo(ui->matchBComboBox, 6)` 用的是**写死的下标**，插入一项之后它会静默指到 DQN+MCTS | **已修复**：`fillAgentCombo()` 把 `maxVisibleItems` 放宽到列表长度 + 2（整份列表一次可见，不需要滚动）；默认项改成**按 agent 类型**查找（`findData(AGENT_EVAB)`），下标再也不会因为重排而指错；新 agent 移到 `PPO+MCTS (AlphaZero)` **后面**（两种骨干成对排列）。新增界面级验证 `tools/verify_agent_combo.ps1`：它把 `kAgents` 从源码里解析成期望值，展开三个下拉框逐条断言**可见性**（只"在模型里"不算），实测 `11 项 / missing = 0 / RESULT: PASS` |
| **C24** | **对弈时崩溃（`0xC0000374` 堆损坏）**：用户报"对弈时自动保存权重的时候导致程序崩溃"，Windows 事件日志三次记录都是 **ntdll + 0xC0000374（STATUS_HEAP_CORRUPTION）**。真因**不在保存**，而在 `ChessBoard::env` 的并发访问：`env = chess;`（决策入口那句"把局面复制到试走棋盘"）写在**锁外**，而 **Alpha-Beta / MCTS 两支在共用的 `env` 上搜索却完全不持锁**。于是"对弈线程在 `env.history` 上 push_back/moveBack"与"另一条线程读 `env`"（界面自检 worker 的 `Chess probe(agent->chess)`、AI 工作线程的决策、并发的另一场对弈）同时发生 —— **加锁的读者挡不住不加锁的写者**。ASan 给出精确位置：`Chess::Chess(const Chess&)`（chess.cpp:526 的 `history(other.history)`）**堆越界写**（按撕裂的 size 分配 96 字节、却按 112 字节搬元素）。触发条件（真界面逐项排除得到）：**后台训练的目标 == 对弈的某一方**，且对弈跑在独立线程、自检面板在刷；用 EVAB 也能复现 ⇒ **与新 agent 无关，是早就存在的问题**（事件日志里 04:55 / 11:29 两次崩溃早于本轮改动） | **已修复**：规则统一为"**凡碰 `env` 都拿 `m_agentMutex`**" —— `aiThinkRaw` / `aiThinkForAgentRaw` 的 `env = chess` 进锁；AB / MCTS 两个分支补上锁（它们同样在 env 上搜索）；`saveCurrentAgentModel` 与 `getAgentSelfCheck` 也进同一把锁（读者必须与写者同锁）。新增最小复现探针 **`repro_concurrency`**（对弈在独立线程 + 每手刷自检 + 后台训练 + 结束后保存；故意不进 ctest）与 `test_match` **[2.16]**；实测：修前 `agent=5/7/10` 全部 `0xC0000374` 秒崩，修后全部通过；ASan 版本修前报 `heap-buffer-overflow`、修后干净；真界面两种配置各跑一遍 **新增崩溃条目 = 0**，且"静默保存 + 保存计时日志"都正常出现。见"零之二点二十五" |
| **C21** | **PPO+MCTS 的权重从来没被启动加载过**：扫描表探测 `weights/ppomcts_agent.dat`，而 `PPOMCTSAgent::saveModel(prefix)` 写出的是 `<prefix>_actor` / `_critic` ⇒ 磁盘上 **279 MB × 2** 的文件一直没被读进来（`docs/agents_design.md` §18 已记过"RL::PPO 那一支栽在这里"，但那次只改对了 DQN+AB 那一行）。三份名字（扫描表 / `defaultWeightPath` / 各 `saveModel`）各自维护，漂移的表现就是**静默地从随机初始化开始跑** | **已修复**：新增 `weightFilesOf(type)`（"`saveModel(prefix)` 会写出哪些文件"的**唯一来源**，扫描与自检面板共用），`s_weightPaths` 统一存前缀；自检面板顶端直接打出"扫描有没有命中 + 每个文件在不在/多大"。见"零之二点二十三" |

### B19 修复后的实测数据

同一台机器、同一个微基准（MSVC 2022 `/O2`，`ikkj`/`kikj` 各 300/200 次取平均）。
分三档：最初的标量实现 → 扁平指针的标量实现 → 启用 `rl/simd_ops.hpp` 的 AVX2 内核：

| 内核 | 最初（`posOf` 版） | 扁平指针标量 | **+AVX2 内核** | 总加速 |
|------|--------------------|--------------|----------------|--------|
| `ikkj (90x90)*(90x90)` | 13.274 ms / 0.11 GFLOP/s | 0.072 ms / 20.4 | **0.053 ms / 27.8** | **250x** |
| `ikkj (90x360)*(360x90)` | 53.016 ms / 0.11 | 0.277 ms / 21.1 | **0.157 ms / 37.2** | **338x** |
| `kikj (360x90)^T*(360x90)` | 53.088 ms / 0.11 | 0.274 ms / 21.3 | **0.272 ms / 21.4** | 195x |
| `ikjk (128x64)*(64x90)^T` | 18.11 ns/MAC | 0.242 ms / 6.0 | **0.254 ms / 5.8** | — |

归约（8100 元素）在启用 SIMD 后收益最大：

| 运算 | 标量 | AVX2 | 加速 |
|------|------|------|------|
| `sum` | 1.002 ns/elem | 0.122 ns/elem | **8.2x** |
| `max` | 1.005 ns/elem | 0.123 ns/elem | 8.2x |
| `norm2` | 1.007 ns/elem | 0.125 ns/elem | 8.1x |

逐元素算符（`a*b+c`）几乎没变（3.28 → 3.13 ns/elem），原因不是算术而是**临时量分配**：
每个算符构造一个新的 `Tensor`（一次 32KB 分配 + 一次零填充 + 一次拷贝），
`a*b+c` 因此要付 3 次分配。`tensorsi.hpp` 本身有同样的性质，它在注释里给出的解法是
`exprfunc.hpp` 的表达式模板（`expt::view(a)*expt::view(b) + expt::view(c)`，单趟融合、
一个临时量）—— 那需要改写调用点，不属于"替换 Tensor 类"的范围，见
`docs/rl_sync.md`。

数值正确性由独立交叉验证确认（朴素 `posOf` 三重循环 vs SIMD 内核，逐元素比较）：

```
ikkj (90x90)x(90x90)   rel = 1.8e-07      kikj (90x90)x(90x90)   rel = 1.6e-07
ikkj (90x360)x(360x90) rel = 2.3e-07      ikjk (out,in)+=(out,1)*(in,1)^T  完全相同 (rel = 0)
```

即 `ikkj`/`kikj` 只剩浮点重结合带来的 ~2e-7 相对误差，`ikjk` 在真实用法形状下**逐位相同**
（包括"连续累加两次"的语义）。

---

## 零之二点五、修复后的实测结果

以下数字**全部来自删掉 build/ 之后的干净构建**（见本节末尾关于"增量构建会留下陈旧
目标文件"的说明，那一度让本文的早期数字失真）。

`ctest`（注册了 `test_ab` / `test_mcts` / `test_rules` / `test_pretrain`）：

| 测试 | 最初 | 修复后 |
|------|------|--------|
| `test_ab` | **SEGFAULT** | **Passed，4.0 s** |
| `test_mcts` | Passed，110 s | Passed，220 s（走法生成现在会剔除非法走法，随机回放成本上升） |
| `test_rules`（新增） | — | Passed，0.06 s（92 条断言） |
| `test_pretrain`（新增） | — | Passed，3.9 s（23 条断言，见"零之二点七"） |

4 个训练基准（默认不注册 ctest，手工运行）：

| 测试 | 最初 | 扁平指针标量 | **+SIMD/GEMV** |
|------|------|--------------|----------------|
| `test_pg` | 346 s | 26 s | **18 s** |
| `test_dqn` | **>900 s 超时** | 536 s | **353 s** |
| `test_ppomcts` | **>900 s 超时** | 233 s | **78 s** |
| `test_dqnmcts` | **>900 s 超时** | 1299 s | **390 s** |

`test_ab` 的 Alpha-Beta 搜索基准（干净构建）：深度 3 ≈ 57 ms，深度 4 ≈ 148 ms，
深度 5 ≈ 3.1 s。GUI 的 `AB_DEPTH` 因此定为 4。

### 构建坑：增量构建会留下陈旧目标文件

修这个坑的过程中发现并修掉了一个**会让上面所有测量都失真**的构建问题：
本机 MSVC 输出的是本地化 `/showIncludes`（`注意: 包含文件:`），而 CMake 在编译器探测
阶段记录的前缀是另一串（乱码）字节，于是 ninja 解析不出任何头文件依赖 —— **改了
`src/chess.h` / `src/stone.h` 之后，目标文件不会重编**。snakeAI 的 CMakeLists 里对此
有一段说明并给 `RL_CORE` 加了 `OBJECT_DEPENDS`，我照做时只加了 `RL_CORE` 和 `chess`
两个目标，**漏掉了 6 个测试目标**。

症状很隐蔽：不报错、不失败，只是跑的是"新旧头文件混编"的二进制。实测同一份源码：

```
单独用 cl 编出来的 test_ab      : 5 s   (搜索基准 深度5 = 1910 ms)
CMake 增量构建出来的 test_ab    : 23 s  (搜索基准 深度5 = 3133 ms)
```

现在 `CMakeLists.txt` 里有一个 `chess_add_header_deps()` 函数，**每一个**目标
（`chess`、6 个 agent 测试、`test_rules`）都会调用它；清理后重建，`test_ab` 回到 4.6 s。

### 构建坑：SIMD 指令集必须向系统确认，而且判据要覆盖 MSVC

`CHESS_ENABLE_AVX2`（默认 ON）会给相关目标加 `/arch:AVX2`，从而解锁 `rl/simd/` 的
AVX2 内核。但这条开关有两个陷阱，都已处理：

1. **`/arch:AVX2` 会让二进制要求 CPU 支持 AVX2**，拿到不支持的机器上直接非法指令
   崩溃。现在配置阶段用 `try_run` 跑一次 CPUID 探测，系统不支持就自动不启用并给出
   警告；运行阶段用 `RL::cpuinfo::describe()` 把"编译进来的是哪套内核 + 这台机器支持
   什么"显示在窗口标题、测试横幅和启动日志里。
2. **MSVC 在 x64 下不定义 `__SSE2__`**（只定义 `_M_X64` / `_M_IX86_FP`），原来的
   `#elif defined(__SSE2__)` 判据让 MSVC 构建直接掉到标量路径、SSE2 内核永远选不上。
   现在判据为 `__SSE2__ || _M_X64 || _M_AMD64 || _M_IX86_FP >= 2`。

三条路径实测（GEMV `w*x`，i5-10400F）：标量 1.059 ns/MAC → SSE2 0.179（5.9x）→
AVX2 0.099（**10.7x**）。细节见 `docs/rl_sync.md`。

---

## 零之二点七、决策前探索（snakeAI 风格）与 EVAB 接入 GUI

### 协议：先"探索环境"训练，再落子

snakeAI 的决策流程是 `observe(state)` → 若 `trainFlag` 则**从当前状态出发**按探索策略
走 N 步、收集 transition、训练一次 → 最后才对**探索前**的状态取 `action(state0).argmax()`。
本工程原本完全没有这个环节（正是 A15：所有在线训练接口在 GUI 侧零调用），
所以 5 个 agent 在 GUI 里只会拿初始随机网络下棋。

现在 `AgentBase` 增加两个虚函数：

```cpp
virtual bool exploreAndTrain(int color, int rolloutSteps);  // 默认 no-op，返回 false
virtual std::string getExploreInfo() const;                 // 供 UI 显示"这步之前学了什么"
```

5 个可训练 agent 各自的实现在 `docs/agents_design.md` 里对应关系如下（`ABAgent` 无参数，
默认实现直接跳过）：

| Agent | 动作选择 | 训练调用 | 经验池 |
|-------|----------|----------|--------|
| `DQNAgent` / `DQNMCTSAgent` | `noiseAction` | `perceive` + `learn` | 复用 agent 自身的 replay buffer |
| `PGEagent` | `gumbelMax` | `reinforce` | 回合内累积的 log-prob/奖励 |
| `PPOMCTSAgent` | `ppo.action` + `Random::categorical` | `ppo.learnSelfPlay` | PPO 自己的轨迹缓冲 |
| `EVABAgent` | `exploreEps`-greedy 的搜索根 | `acceptNet` 门控的 TD-leaf 更新 | 树搜索叶节点样本 |

关键实现点在新增的 `src/agentrollout.hpp`（header-only 模板）：

```cpp
template <class Agent, class PickMove, class OnTransition>
int rolloutFromCurrent(Agent &agent, Chess &chess, int color, int steps,
                       PickMove pick, OnTransition onTrans);
```

它**保存/恢复 `chess.sideToMove`**，用 `moveForward` 前进、结束后逆序 `moveBack` 回退，
保证调用前后棋盘逐字节不变。这个"不变"是本轮新测试 `test_pretrain` 的第一个断言。

### 实测（`test_pretrain`，23 条断言全通过，3.9 s）

```
DQNAgent      训练 371 ms  rollout 32 步, 训练 1 次(池 32)
PGEagent      训练  69 ms  rollout 32 步, reinforce 1 次
PPOMCTSAgent  训练 471 ms  rollout 32 步, PPO 更新 1 次
DQNMCTSAgent  训练 373 ms  rollout 32 步, 训练 1 次(池 32)
EVABAgent     训练   3 ms  探索 8 步, 价值网络更新 1 次 (|net-hand| 1.009413)
ABAgent (对照) 跳过   0 ms
连续 5 次探索后棋盘仍然不变: OK
```

### 这次发现的真问题：`exploreAndTrain` 会篡改轮到谁走

第一版实现直接把 `chess.sideToMove` 当成 `color` 来展开 rollout。但调用方传进来的
`color` 与棋盘当前的 `sideToMove` **并不总是同一个值**（GUI 的"悔棋后重算"、
自对弈里给指定一方求招都会这样）。结果是探索结束后 `sideToMove` 被留在最后一层
rollout 的位置，调用方接着搜索就**从错误的走棋方**开始 —— 表现出来就是"AI 走出非法
招 / 瞬间认输"。

修复：`rolloutFromCurrent` 与 `EVABAgent::exploreAndTrain` 都在入口保存
`savedSideToMove`、出口恢复。`test_pretrain` 现在会连续调用 5 次探索并逐字段比对棋盘，
这个 bug 一旦回归会立刻被抓到。

### GUI 侧接线

- 模式下拉框新增 **"EVAB (学会评估的 Alpha-Beta)"**（`AGENT_EVAB`，权重
  `weights/evab_agent.dat`）。
- 控制区新增复选框 **"走子前先探索训练"**（默认勾选，`preTrainCheck`），
  连到 `ChessBoard::setPreTrainEnabled()`；步数由 `setPreTrainSteps()` 控制。
- 决策路径统一为 `ChessBoard::preTrainThenDecide(agent, color)`：
  先 `exploreAndTrain(color, steps)`，再用 `getExploreInfo()` 的内容填
  `getLastExploreInfo()` 供界面显示，**然后**才真正 `selectMove(...)`。
  勾选框关闭时该函数退化成纯粹的 `selectMove`，与旧行为完全一致。

---

## 零之二点八、AI 思考过程的可视化（长思考时"界面像死了"）

### 问题：思考期间界面零反馈

AI 在后台线程 (`ChessBoard::process`) 里思考，思考期间 `mousePressEvent` 开头就是
`if (state != STATE_IDEL) return;` —— **拒绝落子是对的**（轮到 AI 走），但界面上
没有任何东西说明"它还在算"：

- 棋子不动（AI 那一步还没算完）；
- 点自己的棋子也没有任何反应（点击被静默吞掉）；
- 唯一的反馈 `aiThinkFinished` 是**思考结束之后**才发的。

加入"走子前先探索+预训练"之后，单步思考从 ABAgent 的 ~150 ms 变成
PPOMCTS/DQNMCTS 的 2–3 秒（探索 ~0.5 s + 搜索 ~2 s），于是"到底还在思考还是已经
卡死"从一个理论问题变成了每一步都会遇到的实际问题。

### 修复

新增 `src/thinkingindicator.{h,cpp}` —— `ThinkingIndicator` 控件，三种动画同时给：

| 元素 | 含义 | 实现 |
|------|------|------|
| 呼吸灯 | 还活着、还在算 | 径向渐变的半径与透明度按 `sin(2π·phase)` 起伏 |
| 旋转粒子 | 正在推进 | 一圈粒子的亮度峰值（彗头）随时间绕环跑，带彗尾衰减 |
| 沙漏 | 这一轮有进展 | 上半部的沙按相位线性漏到下半部，漏完复位；**结束后沙子全在下半部**，一眼能看出"算完了" |

外加：实时耗时（百分秒，`Consolas`）、agent 名、当前阶段文字。动画由控件自己的
`QTimer`（33 ms ≈ 30 fps）驱动，不依赖 agent 上报进度 —— 所以哪怕某一步算很久，
画面也一直在动。每帧还会发 `elapsedChanged(ms)`，让主窗口的"AI思考时间"标签
在**思考过程中就跟着跳**，而不是等结束才出数字。

界面上落两处（一处是给"正在盯棋盘"的人看的，一处是给"想要细节"的人看的）：

1. **棋盘内嵌状态条**（`ChessBoard::drawThinkingOverlay`）：画在棋盘**上方那条
   24 px 空白带**里 —— 最上一排棋子的圆心在 `offsetY=50`、半径 24，所以
   `y < 26` 完全空着。放这里既不挡任何棋子，又正好在玩家视线中心：
   `[旋转粒子] AI 正在思考 1.42s · ② 搜索 / 决策`。
2. **右侧控制栏的 ThinkingIndicator**：完整的三动画 + 大号实时耗时 + 阶段。

只重绘这一小条（`update(thinkingOverlayRect())`），不是 25 fps 重画整个棋盘。

### 顺带修掉的两个交互问题

1. **思考期间点击不再静默吞掉**：改为在状态条上闪一句"请稍候, 现在还不能走子"
   （2.5 秒后自动消退）。"点不动"和"点了没反应"是两回事，后者才像死机。
2. **思考途中按"开局"会丢掉那一步**（真 bug）：`reset()` 原来只把 `state` 改回
   `STATE_IDEL` 就返回，而工作线程还在算，算完**照样把子落到重置后的新棋盘上** ——
   表现为"我刚开了新局，对方却已经走了一步"。现在有一个 `m_thinkGeneration`
   计数器：`reset()` 在棋盘锁内 `+1`，`process()` 在锁内比对，代数变了就整步作废
   （含它的计时），回到等待玩家状态。

### 阶段信号

`ChessBoard` 新增 4 个信号，都在工作线程 `emit`、由 Qt 队列投递到 GUI 线程
（接收者是 GUI 线程对象，`AutoConnection` 自动排队），所以主窗口的槽里可以安全
读写控件：

```
aiThinkingStarted(agentName, exploreSteps)   开始思考
aiThinkingStage(stage)                       ① 探索环境 + 预训练 / ② 搜索 / 决策 / ③ 落子
aiThinkingStopped()                          思考真正结束 (含被"开局"打断)
aiExploreInfo(info)                          本次"探索+预训练"做了什么
```

顺序上有一条硬约束：**必须先把 `state` 落成 `IDLE`，再发 `aiThinkingStopped`** ——
状态条的可见性看的就是 `state`，反过来的话 GUI 处理队列消息时可能多画一帧。

`aiExploreInfo` 的存在还顺手修掉一个数据竞争：`m_lastExploreInfo` 是
`std::string`，原来 GUI 线程可能直接去读工作线程正在写的那个对象（A7 那一类），
现在改成用信号送一份拷贝过去，getter 自己也加了锁。

### 验证（不是"看起来对"，是量出来的）

用一个 PS 脚本驱动真实窗口（点棋盘 → 抓棋盘顶部那一小条像素 → 数深色像素）。
脚本留在 `tools/verify_thinking_ui.ps1`，可以直接重跑：

```
powershell -ExecutionPolicy Bypass -File tools/verify_thinking_ui.ps1
```

两个坑都踩过：状态条是**混色**的（`QColor(28,38,54,230)` 叠在米黄棋盘上 ≈
`(47,52,62)`，按原色找会一个点都找不到）；默认 ABAgent 只算 ~150 ms，
**很容易在第一次采样之前就算完了**（脚本里 `ClickAt` 尾部那 200 ms 的 sleep
就足够错过它 —— 所以脚本会先用 Tab+Down 把 agent 切到 MCTS，再在点击后
**不 sleep** 立刻开抓）。

实测输出：

```
idle_strip_dark_pixels   = 0     空闲时状态条不存在
idle_ctrl_diff_pixels    = 0     空闲时控件不动 (没有空转动画)
thinking_strip_frames    = 4 / 90
thinking_strip_max_dark  = 677 / 728   思考期间状态条铺满采样窗 (93%)
thinking_ctrl_animating  = 2 / 29
thinking_ctrl_max_diff   = 8982        思考期间控件大幅变化 -> 动画确实在跑
after_strip_dark_pixels  = 0     思考结束状态条消失
RESULT: PASS
```

即"思考中 → 有状态条 + 控件在动；空闲/结束 → 两者都干净"。

---

## 零之二点九、写 Agent 对弈时挖出来的三个真问题（含一个堆损坏）

做 "Agent 对 Agent 对弈"（见 `docs/agents_design.md` §9）时新测试 `test_match`
在 ctest 下崩了：`0xc0000374`（HEAP_CORRUPTION）/ `0xc0000005`，
**约 75% 的运行会崩**，偶尔又能跑完但报出一个荒谬的比分 —— "第 1 局 (1 手): EVAB 胜"，
即**红方只走了一步就被判负**。单独运行同样代码却时常是好的，属于典型的内存问题。

### 定位过程（值得记下来，因为每一步都省掉了大量瞎猜）

1. **先把它变成可复现的**：写了一个临时诊断程序，把"arena 会做的一切"拆成四个场景
   （A 裸 Chess+AB / B 裸 Chess+AB、EVAB 交替 / C ChessBoard+AB、EVAB / D ChessBoard+AB、AB）。
   结果：**A、B 干净，D 干净，只有 C 崩** —— 也就是"ChessBoard + EVAB"这个组合。
   如果只盯着新写的 arena 代码看，永远找不到这里。
2. **Debug 构建没报错**：MSVC Debug 会给 `std::array` / `std::vector` 的 `operator[]`
   加越界检查，它一声不吭 —— 说明越界不在这些容器上。
3. **上 AddressSanitizer**（见 `CMakeLists.txt` 末尾记录的命令）。它直接给出了调用栈：

```
ERROR: AddressSanitizer: attempting double-free
  #7 Chess::pushHistory()            chess.cpp:834
  #8 Chess::moveForward()            chess.cpp:421
  #9 ABAgent::quiescenceSearch()     abagent.cpp:251
  #16 ABAgent::findBestMove()        abagent.cpp:100
  #18 ChessBoard::aiThink()          chessboard.cpp:874
  #19 ChessBoard::process()          chessboard.cpp:648   <-- AI 工作线程
```

### 根因：条件变量用 `if` 而不是 `while`，且醒来后不重新检查 state

`ChessBoard::process()`（AI 工作线程）原来是这样等的：

```cpp
QMutexLocker locker(&mutex);
if (state != STATE_THINKING) {   /* ← 应该是 while */
    condit.wait(&mutex);
}
if (state == STATE_TERMINATE) { break; }
/* 醒来就往下走, 不管 state 是什么 */
Step step = aiThink(Stone::COLOR_BLACK);
```

条件变量本来就会**虚假唤醒**；更要命的是 `reset()` 和（我新加的）`matchAgents()`
都会在 `state` 仍是 `IDLE` 的时候 `condit.wakeAll()` —— 它们只想叫醒"正在思考"的
worker 好让它作废在飞的那一步，但空闲的 worker 被叫醒后**照样往下跑 `aiThink()`**，
一边搜索一边往 `env.history` 里 `push_back`。如果此时另一个线程（arena）也在用同一个
`env`，两个线程同时改同一个 `std::vector` —— 这就是上面那个 double-free。
两个 `Chess` 对象 `chess` 与 `env` 是 `ChessBoard` 的成员，`aiThink` 只碰 `env`，
所以症状是"偶发"而不是"必崩"。

修复（`chessboard.cpp`）：`while (state != STATE_THINKING && state != STATE_TERMINATE)`
循环等待，并且在进入搜索前重新确认 `state == STATE_THINKING`。

**这个 bug 不只影响 arena**：`reset()`（界面上的"开局"按钮）也会 `wakeAll()`，
所以以前按一下"开局"，AI 工作线程就可能被凭空唤醒并自己走一步 —— 棋盘在没有轮到它的
情况下变化，正是那种"棋子自己动了 / 我点了没反应"的来源。

### 第二个问题：非法走法被当成"被将死"

`!step.valid` 以前直接判"走棋方被将死"：

```cpp
if (!step.valid) {
    emit sendResult(Chess::RESULT_RED_WIN);   /* AI 无合法走法 -> 红方胜 */
    state = STATE_TERMINATE;
}
```

于是**只要 agent 抽风返回一个无效 Step，对局就会在一步都没落子的情况下判红方胜**，
而且之后再也点不动棋盘 —— 这正好是玩家报告过的现象。现在改成先分清两种情况：
真的没有合法走法才判胜负；**如果还有合法走法，说明是 agent 出的错**，用第一个合法
走法兜底并在 stderr 上报一行，arena 还会把次数记进 `MatchStats::agentErrors`
（显示在比分行里）。修好根因之后这个分支已经不再触发（`test_match` 的 stderr 是空的）。

### 第三个问题：越界索引没有守卫

`Chess::applyMove/undoMove` 直接用外部传进来的 `Step` 索引棋盘：

```cpp
Stone *stone = m_children[s->id];    /* std::array, 不检查 */
... stone->moveTo(s->nextPos);       /* StoneMap::data[10][9], 也不检查 */
```

`Step` 来自搜索、回放记录、数据库行，任何一个字段坏了都会变成**越界写**，落在
`Chess` 对象的相邻成员上（`env` 就挨着 `chess`）。现在 `applyMove/undoMove` 先校验
`id ∈ [0,32)`、`pos`/`nextPos` 在棋盘内、棋子存活、且 `Step` 里的起点与棋子当前位置
一致，不满足就返回 `false`（`moveForward` 本来就要处理 `false`）。
把"内存被写坏"变成了"这一步不执行"。

### 顺带修掉的

`shutdownSave()` 漏了 EVAB —— 它是界面上可选、能被在线训练的 agent，但退出时从来不
落盘，于是每次启动都从随机价值网络重新开始，上一局学到的全丢。

### 验证

* ASan 下把诊断程序的两个关键场景各跑 3 遍：修之前 C/D 几乎必崩，修之后
  **6/6 全部退出码 0、0 次异常、无 ASan 报错**；另外两个场景 A/B 也各跑 1 遍通过。
* `test_match` 从"ctest 里必崩"变成 **Passed 1.9 s，18 条断言 0 失败**，
  比分也变回正确的 `共 2 局 / 8 手`（每局 4 手，判和）。
* `ctest` 5/5 通过。

---

## 零之二点十、做奖励曲线时挖出来的符号 bug（即时奖励一直是黑方视角）

用户要求"增加控件动态显示……模型每次对局所获的环境奖励曲线"。要画这条曲线，先得回答
"这一步到底给了多少奖励"，于是写了个 30 行探针（`build/reward_probe.cpp`，临时文件）
直接问棋盘和 agent：

```
红方: 炮(7,1) 吃 马(0,1), victim color=1 value=0.30
moveForward totalReward = -0.300
computeReward(红)      = -0.300
结论: 红方白吃一个黑子, 拿到的即时奖励是 负
```

**根因**：五个 agent（`dqnagent` / `dqnmcts_agent` / `pgagent` / `ppomcts_agent` /
`sacazagent`）各有一份 `computeReward` 拷贝，全都写成

```cpp
return (color == Stone::COLOR_BLACK) ? reward : -reward;   // 黑方视角
```

而 `Chess::moveForward` 的记账（`吃红子 += value`）也是黑方视角 —— 两处同源，所以看起来
"自洽"，很难被注意到。但**同一个 agent 的终局奖励是走子方视角的 ±1**
（`SACAZAgent::resultValue`），于是对红方来说塑形信号（吃子 −）与终局信号（赢棋 +）
方向相反：**红方被教成"吃子是坏事"**。
（`EVABAgent::evaluateLeaf` 的 `(color == BLACK) ? MATE : -MATE` 是正确的走子方视角，
说明原意就是走子方视角，`computeReward` 是写错了。`dqnmcts_agent.cpp` 里另有一处调用把
颜色硬编码成 `COLOR_BLACK`，是同一个错误的产物。）

**修复**：五份 `computeReward` 统一成走子方视角（吃子永远 `+value`），并在 `test_match`
第 [2.6] 节把**两套约定都钉住**（agent 侧走子方视角、棋盘侧黑方视角记账）：

```
红方吃 马 (value=0.30): moveForward 记账 = -0.300
computeReward(红) = +3.000
computeReward(黑) = +3.000
```


## 零之二点十一、EVAB 的三个 bug（"与 AB 对弈无法取胜"的根因）

用户报"evagent 与 abagent 多次对弈无法取胜，检查 evagent 在对弈过程是否累计梯度训练"。
查下来：**梯度确实在累计**（64 步探索 / 每 16 个样本一次 RMSProp），但训练出来的网络
**从来没有参与过决策**，而且它本身是坏的。三个独立的 bug：

| # | 问题 | 现象（探针实测） | 修复 |
|---|------|------------------|------|
| 1 | **价值网络没有按 fan_in 缩放初始化** | 输入是 1260 维 one-hot，第一层 pre-activation 标准差 ≈ √(32/3) ≈ 3.3 → tanh 一上来就饱和：任意局面输出都是 0.998，20 次更新只动了 **1.4e-4**，`|net-hand|` ≈ 0.95（几乎最大分歧） | 构造函数里加 `scaleLayerInit(valueNet)`（1/√fan_in）；修完 `|net-hand|` 0.95 → **0.02**，每轮真的在动 |
| 2 | **`blend`（学习评估与手工评估的混合比例）恒为 0** | 全工程只有测试改过它，agent 本体从没改过 → 界面上那个 "learned eval" 一次都没生效；EVAB(depth=4) 与 ABAgent 走法 **22/30 相同** | 在线更新成功后 `blend += 0.05`（上限 0.3），回滚时退两步（门控沿用原有的 `|net-hand|` 检查） |
| 3 | **搜索预算照抄了 AB 的深度** | `EVABAgent(env, 48, AB_DEPTH=4, 0)` —— 同深度下 EVAB 比 AB 快一个数量级（depth 5: EVAB 188 ms vs AB 1873 ms），等于把"更强的搜索"这个卖点也关掉了 | `EVAB_DEPTH = 6` + `EVAB_BUDGET_MS = 800`（迭代加深 + 时间上限自动适应评估成本） |

修完之后（`build/evab_arena.cpp` 探针，每配置 6 局、每局交换先后手、与界面同一条调用序列）：

| 配置 | 胜 | 和 | 负 | ms/步 |
|------|--:|--:|--:|------:|
| 同深度(4) + 学习评估 | 2 | 2 | 2 | 174 |
| 深一层(5) + 学习评估 | 1 | **5** | **0** | 796 |
| 深一层(5) + 纯手工评估（blend 钉 0） | **0** | 6 | 0 | 70 |

**能赢了**（修复前探针里 0 胜），而且"赢棋来自学习评估"这件事有对照：把 blend 钉回 0、
同样深度就是 6 局全和（两边同一个评估）。

**代价**：只要 blend > 0，每个叶子都要跑一次网络前向，实测贵 11~14 倍（depth 5: 70 → 796
ms/步）→ 评估变贵 = 搜索变浅。所以 blend 上限刻意压到 0.3、搜索用时间上限。
细节见 `docs/agents_design.md` §14。

**样本量说明**：每配置 6 局，单局偶然性足以翻转结论，所以结论只能到"机制通了、能赢棋了"，
不能到"EVAB 比 AB 强多少"。
**界面上也验过**（`tools/verify_match_ui.ps1 -AIndex 6 -BIndex 0 -Games 2 -Full`）：
`最终比分: EVAB 1 : 1 Alpha-Beta  共 2 局 / 190 手`，损失曲线 95 个样本、只有一条
（修了"名字里带 blend 导致同一 agent 分出一堆曲线"的问题），双击放大窗口与源控件逐字一致。

**还没定位的偶发现象**：arena 里偶尔出现 `[1 次无效走法已兜底]`（约每 200 次决策 1 次）。
兜底逻辑保证不影响胜负，但根因未知，见 `docs/agents_design.md` §14.9 —— 不假装已解决。

---

## 零之二点十二、启动时加载所有模型 + 奖励曲线改成每手一个点（含两条实测）

这一节记两件用户直接提的事、一个顺手抓到的界面 bug，以及一个"我的验证方法本身在骗我"的教训。

### (1) "程序启动时加载所有模型"

上一轮为了启动速度，把稀疏 MoE 变体（3×146 MB）改成了**懒加载**（启动 1.7 s，首次用到它时
卡十几秒）。用户明确要求全量加载，于是改回预加载 —— 但**不能只是改回去**：一开始实测启动
**19.2 s**，那样的体验是在逼用户二选一。所以顺手把加载路径做快：

| 手法 | 做了什么 | 实测 |
|------|----------|------|
| 流式 base64 解码 | `Tensor::base64DecodeInto`：**一遍**同时做完合法性 + 长度 + 增量 CRC32 + 写目标（原来 getline 切出字符串、再解码、再单独校验一遍） | 解码 99 → **302 MB/s** |
| 256 项解码表 | `base64DecodeTable()`（无效 = −1，`=` = 64），取代逐字符分支 | 同上 |
| 零分配的载入前校验 | 第一版用 `getline` + `std::string` 逐行校验，**每次校验都在堆上造几百万个短字符串** | 校验从"启动多花十几秒"变成可忽略 |
| 每组一条计时日志 | `[weights] <名字>: N ms` | 慢的时候一眼看出是哪一组 |

结果（本机 Release，七组权重全部预加载）：

```
[weights] PG: 167 ms      DQN: 174 ms      EVAB: 22 ms     DQN+MCTS: 162 ms
[weights] SAC+AZ: 72 ms
[weights] SAC+AZ-MoE 建网(5 x 28.7M 参数): 1604 ms
[weights] SAC+AZ-MoE 读权重(3 x 146 MB): 6235 ms
→ 启动到界面可用 19.2 s → 10.1 s（沙漏全程覆盖，界面可用时它已收起）
```

**如实说明两件事**：**(a)** 那 6.2 s 里还有约 1 s/文件 是 `Net::load` 真正读那一遍走
`ifstream + getline`（把 `iLayer::read/write` 的签名换成 `std::istream&` 还能再省 2~3 s，
纯机械改动，没做）；**(b)** 懒加载那两个分支的 `loadModel()` 现在**不可达**（权重文件在，
`startupLoad` 就读了；不在，分支里也不会读），它们只剩"防空指针"的兜底价值 —— 所以原来那个
断言"首次使用会弹沙漏"的 `tools/verify_busy_lazy.ps1` **只能失败**，已替换成
`tools/verify_eager_load.ps1`（断言相反的事实）。

### (2) "对弈时奖励曲线没有更新"

信号没断、数据没错，是**采样太稀**：`gameRewardSample` 一局只发一次，而实测那局
`SAC+AZ-MoE vs Alpha-Beta` 是 **276 手 / 620.8 秒** —— 按下开始之后的十分钟里曲线只有 **1 个点**。

修法：新增 `matchRewardProgress(gameNo, ply, rewardA, rewardB)`，`playMatchGame` **每落一手**
emit 一次，值是"本局到目前为止"累计的走子方视角环境奖励；局末仍由 `gameRewardSample` 补一个
含终局 ±1 的点（所以每局最后一个点会比前一个多出胜负那一份，这是设计不是重复计数）。

| | 之前 | 现在 |
|---|---|---|
| 采样 | 每局 1 次 | **每手 1 次** + 局末 1 次 |
| 一局 276 手 | 1 个点 | **277 个点** |
| 实测（AB vs AB，2 局 256 手） | 2 点 | **258 点**（对局中取样 11 → 228 一直在涨） |

顺手修掉 C10：`playMatchGame` 多一个 `bool aIsRed`，红/黑 → A/B 的换算只做一次，出参与每手
进度共用同一份结果（原来那两个出参是死参数，调用方在外面重算了一遍）。

### (3) 验证方法学的五个坑（本轮最贵的教训）

这部分不是产品 bug，是**检查本身在骗人**，而且错误的检查比没有检查更坏 —— 它会给出一个
看起来很像结论的答案。逐条见 `docs/agents_design.md` **16.3**，这里只记最贵的那一条：

> 为了"只看前 20 行错误"，我把构建输出接进了 `Select-String ... | Select-Object -First 20`。
> PowerShell 的 `-First N` 一凑够 N 项就**终止整条管道**（包括正在编译的进程），于是编译被
> 拦腰杀掉、`chess.exe` 根本没重新链接，而我以为构建通过，接着拿**旧二进制**跑了一遍界面验证，
> 还认真分析了"奖励曲线为什么不动"—— 那本来就是修复前的现象。
> 现在：构建输出重定向到文件再读，并且**每次构建后核对 exe 的时间戳**。

其余四条：后台保存没写完就断言（假失败）、启动加载日志冒充保存日志（**对局没打完也报通过**）、
改脚本丢 BOM（报无关语法错）、拿"本来就不该出现的东西"当失败（AB vs AB 时没有可训练 agent，
没有损失曲线和存权重行都是正确的）。最后一条催生了 `verify_match_ui.ps1` 里的 `$expectSave`：
**先判断这条检查适不适用，不适用就打印"跳过 + 原因"，而不是判失败**。

### (4) 顺手抓到的 C12：每跑一场就多挂两条空线

在用户那次 **100 局**对弈的窗口里（只读 UIA 采样，没有干扰它）读到的奖励读数：

```
奖励(局内累计) SAC+AZ-MoE: 暂无 | Alpha-Beta: 暂无 |
               SAC+AZ-MoE: 最新 0, 均值 0.049185, ..., 982 点 |
               Alpha-Beta: 最新 0.1, 均值 1.3437, ..., 982 点
```

四条线、两条永远"暂无" —— `resetMetricsForMatch` 用的是 `clearData()`（只清**点**）+
`addSeries()`×2，于是每开一场就往图上多挂两条。修法（`removeAllSeries()`）、同类问题的
"清空曲线"按钮、以及 `test_match` **[2.9]** 的 10 条断言见 `docs/agents_design.md` **13.8**。
同一次采样还顺带确认了 C9 的修复真的在起作用：**982 个点**（每手一个），而不是修复前的个位数。

---
## 零之三、修复前的实测结果（作为对照）

`ctest --test-dir build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release`：

| 测试 | 结果 | 耗时 |
|------|------|------|
| `test_ab` | **SEGFAULT** | 12.8 s |
| `test_mcts` | Passed | 110 s |
| `test_pg` | Passed | 346 s |
| `test_dqn` | **Timeout(900 s)** | >900 s |
| `test_ppomcts` | **Timeout(900 s)** | >900 s |
| `test_dqnmcts` | **Timeout(900 s)** | >900 s |

`test_ab` 的段错误与下面 **A1** 完全对应（第一次访问棋盘就解引用未初始化的指针）。
`test_pg` 能跑完说明 RL 内核本身可用。四个训练基准的耗时全部由 **B19** 解释：
`Tensor::MM::*` 实测只有 **0.11 GFLOP/s**（18.2 ns/MAC）。这一点做过对照实验——
把 DQN 主干换成小 8 倍的上游默认结构，`test_dqn` **仍然超过 1800 s**，
所以超时是矩阵乘实现问题，不是网络结构问题。

---

## 零之四、优化方法汇总（含实测数字）

前面几节是按"发现顺序"叙述的；这一节按**手法**把用过的优化集中起来，并给出
实测数字，方便判断"哪个优化值得做、做完还剩什么瓶颈"。

### 4.1 计算内核（RL 库）

| 优化 | 手法 | 实测 |
|------|------|------|
| MM 内层循环扁平化 | 不再每次访问都过 `posOf()` 构造索引；行指针 + 线性扫描，stride 提到循环外 | `ikkj (90×90)²` 13.274 → 0.072 ms（**184×**） |
| SIMD 内核分派层 | `rl/simd_ops.hpp`：把"能不能用 SIMD"的判据集中一处，内核取自 N-spirits 的 `sse2func/avx2func.hpp`；保持 `RL::Tensor_` **一个类型**不变（换类型要重新验证所有调用点语义，收益却很小） | 同上再降到 0.053 ms / **27.8 GFLOP/s**（总 **250×**） |
| GEMV 专用内核 | MatMul 内核沿 z 的**列**方向向量化，而 `o = w·x` 的 z 只有一列 → 向量循环长度为 0，用不上。改成"按行做点积"（`gemv_ikkj` → `Instruct::dot`） | 标量 1.059 → SSE2 0.179 → **AVX2 0.099 ns/MAC**（**10.7×**）；这是本库的绝对热路径 |
| 归约 SIMD | `sum`/`max`/`norm2` 走向量归约 | 1.00 → **0.12 ns/elem**（**8.2×**）；收益最大的一类 |
| 开放 AVX2 | `CHESS_ENABLE_AVX2`（默认 ON）给相关目标加 `/arch:AVX2`；配置期用 `try_run` 跑一次 CPUID 探测，系统不支持就自动关闭并警告；运行期把"编译进来的是哪套内核"显示在标题栏 | SSE2 → AVX2 再快约 1.8× |
| **未做**：反向 GEMV | `kikj` 没有 GEMV 内核（zCol=1 判据不成立） | 0.991 vs 0.102 ns/MAC → **C8，训练步的真正瓶颈** |
| **未做**：表达式模板 | `a*b+c` 这类逐元素算符的开销不是算术而是**临时量分配**（每个算符一次 32 KB 分配 + 零填充 + 拷贝）。上游 `tensorsi.hpp` 的解法是 `exprfunc.hpp` 的 `expt::view()` 单趟融合 | 3.28 → 3.13 ns/elem（几乎没变）；需要改写调用点，不在"替换 Tensor 类"范围内 |

### 4.2 构建与工具链

| 优化 | 手法 | 为什么必要 |
|------|------|-----------|
| 头文件依赖显式化 | `chess_add_header_deps()` 应用到**每一个**目标 | 本机 MSVC 输出本地化 `/showIncludes`（`注意: 包含文件:`），ninja 解析不出依赖 → **改了 `chess.h` 目标文件不重编**，跑的是新旧混编的二进制（实测 `test_ab` 5 s vs 23 s，搜索基准 1910 vs 3133 ms）。一度让本文早期所有数字失真 |
| `RL_CORE` 独立静态库 | 显式源文件列表、`AUTOMOC/UIC/RCC OFF`、不依赖 Qt | RL 内核与界面解耦；测试目标可以只链 `RL_CORE`（`test_grad` 完全不依赖 Qt） |
| `/bigobj` | RL 的模板实例化量很大 | 否则编译直接失败 |
| 测试铺开 | `ctest` 从 1 个崩的 `test_ab` 变成 **11 个全过**：`test_ab`/`test_mcts`/`test_rules`/`test_pretrain`/`test_match`/`test_grad`/`test_weights`/`test_scaledconcat`/`test_sparse_moe`/`test_sac`/`test_sacaz`（另外 `bench_*` 与 4 个训练型 `test_pg`/`test_dqn`/`test_ppomcts`/`test_dqnmcts` 是手动跑的，故意不进 ctest） | 见 4.4 |
| ASan 配方 | 单独构建目录 + `CMAKE_CXX_FLAGS=/fsanitize=address`（**整个**构建都要带，否则 LNK2038）；运行前把 `clang_rt.asan_dynamic-x86_64.dll` 目录加进 PATH，`ASAN_OPTIONS=detect_leaks=0` | 只给单个目标加 ASan 会链接失败。C1 那个堆损坏就是它定位的 |

### 4.3 算法与交互

| 优化 | 手法 | 实测 |
|------|------|------|
| EVAB 搜索 | 置换表改用标准 **Zobrist** 键（原来的结构化异或相关性强，碰撞会让 TT 返回别的局面的分数）；TT + killer + history 排序；修复静态搜索零宽窗口 | 深度 5：146 ms / 158 572 节点 / 31.1% TT 命中 vs `ABAgent` 3133 ms → **21×** |
| 小批量更新 | 原来把整轮几百个样本的梯度累积起来只调一次 `RMSProp`（三轮训练 = 3 步梯度，什么也学不到）→ 改成真正的小批量 | EVAB 模仿引导 |net−hand| 1.23 → 0.05 |
| 训练稳定性 | `acceptNet` 门控：`netHandGap` 不下降就整体回滚（TD-leaf 在样本量不足时会发散，实测误差 0.35→0.70） | 保证"不变坏" |
| 决策前探索统一化 | `AgentBase::exploreAndTrain()` + `src/agentrollout.hpp` 的 `rolloutFromCurrent()`（保存/恢复 `sideToMove`，逆序 `moveBack`） | 5 个 agent 共用一份实现；`test_pretrain` 23 条断言 |
| 对弈方法学 | 每局**交换先后手**、胜负按参赛者 A/B 记（中国象棋先手优势大，固定红黑只是在测"谁执红"） | `test_match` + `tools/verify_match_ui.ps1` |
| 长思考的可观测性 | `ThinkingIndicator`（沙漏+旋转粒子+呼吸灯+实时耗时）+ 棋盘顶部 24 px 空白带内的状态条；**只重绘这一小条**；思考期间点击给"请稍候"反馈 | `tools/verify_thinking_ui.ps1` 量出"思考中有状态条、空闲/结束都干净" |
| 对弈不阻塞界面 | 对弈在后台线程，进度用队列信号回 GUI；同一个按钮兼作"停止"（`abortMatch` 逐手检查） | 关窗先 `abortMatch()` 再 join |
| **新增 SAC+AZ agent** | 最大熵 critic（双 Q + 软备份 + 自动调节 α）给 AlphaZero 式 PUCT 搜索提供叶子估值；策略目标来自搜索访问分布，数据进回放反复利用 | 单步决策 64 模拟 **3 ms**（1260 维输入的小 MLP），GUI 里给到 256 模拟（约 12 ms）。设计见 `docs/agents_design.md` §11 |
| **稀疏路由 MoE** | 自己实现 `rl/sparse_moe.hpp`：门控选 top-k 个专家**只算这几个**（上游 `MOE::forward` 是 16 个全算的稠密混合）。`TopK==E` 退化成稠密，正好当等参数对照 | 同样 28.7 M 参数下 **42.0 → 10.9 ms/模拟（3.85×）**；专家从 16 个减到 4 个、head 取满真因数。实测与陷阱见 §零之四 4.5 |
| **PPO 的专家换成 `TransformerBlock<16,360>`**（2026-09 第二轮） | 一行 typedef（`PPOExpert`）+ 两个常量；专家数/top-k 同时 8/2 → 4/1 | 参数量 2.15 M → **38.0 M**（17.7×），前向 0.139 → 3.59 ms（26×）；E=8/top-2 的 actor+critic ≈ **2.4 GB**（`w/g/v/m` 四份）⇒ 不可用。**这是拿 6–30× 的时间换 20× 的容量，不是净优化** —— 值得/不值得要靠棋力结论，这一轮没做。见 `agents_design.md` §18.1 |
| **PPO 的训练侧优化方法 → SAC**（两个 SAC 都改） | `RL::SAC` 重写（稀疏 MoE 骨干 + P3 拆分 + P4 多 epoch + 批统计复位/辅助损失 + 批平均损失 + R2 掩码口径）；`SACAZAgent` 补 `resetMoeBatchStats()`（划出批边界）与 `replayEpochs` | R2：非法列头权重**逐位不变**（0/5760）、合法列 2816/2816；P4：批 = batchSize×epochs、优化器只调一次；批统计：复位后推理前向零影响（A==B）、不复位则被污染（C≠B）。见 `agents_design.md` §18.2 |
| **对弈结果实时可见** | `matchScoreChanged` 每局更新实时比分；逐局明细进 `QListWidget`（一局一行，含本局环境奖励）；去掉模态结果框 | 长对弈中途也能看到"几比几"；脚本断言 `score shows a ratio = True` / `list has per-game lines = True` |
| **训练损失 / 环境奖励曲线** | 自绘 `CurveChart`（不引入 Qt Charts）+ `AgentBase::getLastTrainLoss()`；奖励**每手**采一个"本局累计"点（`matchRewardProgress`），局末再补一个含终局 ±1 的点 | 实测 AB vs AB 两局 256 手 → 258 个点（对局中一直在涨）；修复前是每局 1 个点（十分钟不动的观感）。见 §零之二点十二 与 `docs/agents_design.md` §13 |
| **即时奖励符号修正** | 五个 agent 的 `computeReward` 从"黑方视角"改成"走子方视角"（吃子永远 +value） | 红方吃子从 **−0.30 变成 +0.30**；`test_match` [2.6] 钉住约定。见 §零之二点十 |
| **权重文件格式 v2** | 无损编码 + 结构指纹 + 每张量 CRC32 + 原子写 + 载入前零分配校验 | 往返**逐比特相同**、体积 **1.78×**、坏文件一律拒绝且不改动网络。见 §零之四 4.6 |
| **EVAB 三修**（初始化缩放 / blend 阶梯 / 搜索预算） | 见 §零之二点十一 | 6 局对抗从"0 胜"变成"2 胜 2 和 2 负"（同深度）；深一层 1 胜 5 和 0 负 |
| **损失曲线上报补全** | `RL::DQN::lastLoss`（在 experienceReplay 累加）/ `RL::PPO::lastLoss` / `RL::DPG::reinforce+reinforce1`（PG agent 走的是 reinforce）；`EVABAgent::exploreAndTrain` 也写 `m_lastLoss`；SAC+AZ 的在线那次 `learnBatch` 改成按池大小夹批 | `test_match` [2.7]：7 个可训练 agent 都上报有限损失（修复前只有 PPO+MCTS / EVAB 两个上报），Alpha-Beta / MCTS 仍不上报 |
| **曲线读数公式修正** | 窗口淘汰点时必须把它从 `sum/mn/mx` 里去掉 | 均值和纵轴范围以前会随淘汰漂移（"显示表示也是错的"） |
| **双击放大曲线** | `CurveChartDialog`（跟随源控件、非模态、单实例） | 脚本断言"放大窗口的读数与源控件逐字一致" |
| **对弈结束静默存权重** | 去掉"另存为"与"成功"两个模态框；存到 `ChessBoard::defaultWeightPath()`（与启动加载/退出保存同一份）；后台线程写、结果写进逐局明细列表 | 顺带消灭了"另存为默认名 ≠ 启动读取名"的静默失效；`shutdownSave()` 也改成遍历同一张表（EVAB 当年就是这么漏存的）；每个 agent 再记一条 `[weights] 保存 <agent>: N ms`（实测稀疏 MoE 438 MB = 1983 ms，**不含 fsync**） |
| **程序图标** | `tools/make_app_icon.ps1` 生成 `src/app.png` + `src/app.ico`；运行时 `setWindowIcon`（走 res.qrc）+ exe 的 PE 图标（走 app.rc） | `tools/verify_app_icon.ps1` 33 项检查全过（ICO 结构 / 字形真的渲染出来 / 运行时加载 / ExtractAssociatedIcon 提取到我们的配色） |
| **载入/保存时的沙漏等待窗** | `src/busydialog.h/.cpp`：复用 `ThinkingIndicator`；`ChessBoard` 只发 `busyStarted/busyMessage/busyFinished`；保存放到工作线程（否则弹窗画面冻结）；加载中不可关闭 | `verify_busy_ui.ps1` + `verify_eager_load.ps1` 都 PASS。**最终取"启动时全量加载"**：7 组权重约 10 s，沙漏全程覆盖；靠"流式 base64 解码 / 256 项解码表 / 零分配校验"把启动从 19.2 s 压到 10.1 s。见 §零之二点十二 |
| 关窗不再冻结 | 后台训练单轮规模从 4 局×200 步×50 模拟降到 `BG_TRAIN_*` | 关窗 join 的等待从"几分钟"降到秒级 |

### 4.4 验证手法的沉淀（这部分是本轮最有复用价值的产出）

| 脚本 / 测试 | 作用 | 为什么不能用"看起来对"代替 |
|------|------|------|
| `test_pretrain`（23 断言） | 盯住"探索不能改动真棋局" | 象棋没法像贪吃蛇那样在局部坐标上模拟，试走只能落在真棋盘上，必须原样回退；C5 就是它抓到的 |
| `test_match`（**71 断言**） | arena 统计：交换先后手、比分归属、到上限判和、中止生效；[2.6] 钉住**即时奖励的符号约定**（走子方视角 vs 棋盘层的黑方视角记账）；[2.7] 每个可训练 agent 都要上报损失；[2.8] **每手的奖励进度必须和局末奖励同账**（一局 12 手要产生 ≥11 个进度点、手号连续、"最后一个进度点 + 终局增量 == 局末采样"、A/B 增量互为相反数）；[2.9] 曲线控件"换一批线"的两个语义（`clearData` 只清点 / `removeAllSeries` 才清线），并把"只 clearData 再加两条会变成 4 条"这个 bug 形状写成断言 | 把每局压到 4~12 手，结果可预测，断言才做得硬。奖励符号那条是"两侧都自洽、但彼此相反"的典型：只有把物理含义写下来对照才发现。[2.8]/[2.9] 是"用户看得见的现象也要有断言"的例子：曲线不动、或多出几条空线，在代码里都不会报错 |
| `test_grad`（6 断言） | 有限差分核对 SIMD 之后的解析梯度；直接探测 MM 内核"累加 vs 覆盖" | "前向对"推不出"梯度对"（前向/反向用不同的 GEMM 形式）；C7 就是它量出来的 |
| `test_weights`（**49 断言**） | 权重文件：逐比特往返、两次存出的文件逐字节相同、体积、**老格式仍可读**、截断/指纹/翻一位/结构不符/文件不存在都要失败、失败后网络逐比特不变、原子写不留 `.tmp`；**(g) 维度守卫**：同一套层结构、不同输入维度（1440 vs 1710）的文件**结构指纹相同**但必须被参数量守卫拒绝，并配"同维度仍能载入"的正对照 | "存下来再读回去一样"以前从来没验过 —— 一验就发现老格式是**有损**的（差 1.6e-06），而后台训练每轮都在存读。**(g) 是改状态编码的前置条件**：v2 的指纹只哈希层类型，不含维度，所以"同一套层、不同维度"的文件指纹一模一样 —— 载入会被放行，而 `Layer::read` 是整块替换，网络的张量被静默换成文件里的形状（不崩则输出全是垃圾） |
| `test_sacaz`（**135 断言**，第 [10] 节遍历四种骨干，[11] 是 PPO 移植那两条，[12] 是 TB 专家 SAC agent 专项；本轮 [3] 段加了 4 条"完备 Markov 状态"断言） | 掩码 softmax 雅可比、走法合法性、软价值里 α 熵项的形式、critic 是否真的在学、**四种骨干都能建/能走/能训/能存取**、**R2 的"非法列头部权重逐位不变"（配"合法列必须变"的正对照）**、**P4 的"批 = batchSize×epochs 而优化器只调一次"**、**TB 专家的 AZ 监督项（配对 A/B）/ critic 学习 / 自对弈路由健康 / 代价** | α 的符号我第一版就写反了（断言 `V(α=0.5)−V(α=0) = +0.5H` 才发现）；骨干开关这种"多分支构造"最容易只在某一个分支上写对；分层的 write/read **顺序**错位是静默的，所以必须用"存了再读、比对输出"来查。"没动"和"动不了"必须分开：一个根本没训练的实现也满足"权重不变"，所以每条"不变"都要配一条"必须变"。第 [12] 节的 AZ 项检查第一版写成"跑 12 步后 π 必须上升"——**假失败**，因为 clipGrad 让每步位移恒等于 lr、轨迹剧烈振荡（实测均值 0.316 / 峰值 0.778 / 终点 0.00039），改成配对 A/B + 轨迹统计量才对 |
| `test_sac`（**43 断言**，1.2 s，进 ctest） | `RL::SAC` 在这之前**没有任何调用方**。这个测试盯 R2 的四条口径（合法集 Σπ≡1 / 非法列 π 恰好为 0 / 非法列头权重梯度**恰好为 0** / s 与 s' 各用各的掩码）、P3 的两条（`accumulateGrad` 只攒不更新、梯度严格累加）、P4 的语义（批 = batchSize×epochs、优化器只调一次）、以及 MoE 的批统计**边界**（复位后推理前向零影响，配"不复位确实有污染"的正对照） | 写这个测试的过程本身就抓到两处静默缺陷（critic 的拼接被 `MM` 截断、`learn()` 的 `learningRate` 没接线）；另一条教训是配对实验里"同权重"必须**连 critic 一起拷**，否则"1 遍 vs 2 遍的梯度比值"会量出 2.59 而不是 2.00，看起来像累加坏了 |
| `test_sparse_moe`（**67 断言**） | 稀疏路由的三个不变量：前向真的跳过未选中专家（改权重输出逐位不变）、`TopK==E` 时与上游 `MOE` 前向/反向逐位一致、未选中专家的梯度**恰好为 0**；加上辅助损失的有限差分与纠偏方向 | 假通过有两种：直接调 `layer->backward()` 时 `e` 全是 0，"两边都是 0"看起来也一致；拿**拷贝**出来的张量做差分时扰动改不到真权重，差分恒为 0。两次都发生在写这个测试的过程中 |
| `bench_moe`（**不进 ctest**） | A/B/C/D 等时间对弈 + 标定 ms/模拟 + 专家使用分布 | 等"模拟次数"比强弱等于比谁算得多；TB 骨干一步几十毫秒，放 ctest 会既慢又偶发失败 |
| `bench_sacaz_vs_ab`（**不进 ctest**） | 带 TB 专家的 SAC+AZ（GUI 的 "SAC+AZ-MoE" 那一档）与 ABAgent 的**静默**对弈：逐手校验合法性、不许有无效走法、每局必须在手数上限内结束、对局后 Q/π 全有限、掩码口径 Σπ≡1；`--quiet` 只打一行 `VERDICT`，**退出码只看机制、比分不进退出码** | 随机权重输给 AB 是预期内的 —— 把比分当断言会让这个基准永远红着；反过来，把机制检查省掉又会让"走法非法/NaN"这类真正的问题悄悄过去（对局里 NaN 只表现为"走法没道理"，不会报错）。实测四种骨干全部 PASS，见 `agents_design.md` §18.7 |
| `test_dqnab`（**115 断言**，24.5 s，进 ctest） | 新 agent **DQN+AB**（`DQNABAgent`）：规范视角的**镜像对称**（编码逐位不变）、动作双射与镜像一致、**Dueling 双头解析梯度 vs 中心差分**、非法列梯度恰好为 0、**一步杀命中率（AB 规划 vs 纯 Q-argmax）**、固定 TD 目标收敛、自对弈/探索的**棋盘零副作用**、稀疏 MoE 路由、两种骨干的代价、存取往返；**[9] 段是第二轮优化**：手工锚口径、**造探针也零副作用**、预训练提高 corr、**常数 V 的 corr 恰好为 0（gap 的盲区）**、门控在毒化更新上回滚且**主干逐元素还原**（配"门控关就变坏"的对照）、**门控不误伤正常学习（回滚 0 次）**；**[10] 段是第三轮（Markov / negamax）**：同一局面不同重复历史 ⇒ **编码不同**、`repetitionCount()>=3 ⇔ 引擎 isRepetition()`、无吃子平面随 `halfMoveClock` 涨/吃子归零、**V 是当前方视角**（镜像+换色后同一个数）、**TD 目标是 negamax 而不是单 agent 的 max**（把 Q 钉成 +1 做符号实验：实测 −0.99 而不是 +0.99） | 这几条都是"错了不会报错"的：规范视角写错 → 红黑变成两个函数；Dueling 的 `∂Q/∂V=1`/`∂Q/∂A_j=δ−1/n` 写反 → 训练照跑、只是学不到东西（所以必须有限差分）；**TD 目标的负号写错 → 训练照跑、只是永远收敛到错的值**（只能靠"把值钉成常数"的符号实验查）；训练入口里 reset 棋盘 → 直接吃掉调用方的棋局。**门控那两条是被断言逼出来的**：阈值照抄 EVAB 的 0.05 时 `[5]` 的 60 次更新全被回滚，固定 0.25 时回滚 59/60，按 `lr` 缩放后才是 0 —— "回滚成功"必须配"不误伤"的对照。另外 `[5]` 与 `[1]` 各暴露过一次**假失败**：前者拿 clipGrad 振荡轨迹的**终点**比大小（改成"轨迹最小值 + loss 下降"），后者的 `mirrorBoard` 只交换 color 字段而 `isInCheck` 是按 `&redJiang/&blackJiang` 取将帅的（改用 `setupSparse` 重摆镜像局面） |
| `bench_dqnab_vs_ab`（**不进 ctest**） | DQN+AB 与 ABAgent 的静默对弈 + **planning head 的 A/B**：`--pure-q` 关掉搜索、只按 Q 取最大（同一个网络、同一个 seed），另有 `--backbone=mlp` 做"叶子便宜 → 搜得更深"的对照；`--pretrain=N` / `--pretrain-epochs` / `--pretrain-plies` / `--pretrain-head-only` / `--gate` 量手工锚预训练；报告节点数、到达深度、ms/步，以及**手工锚的 gap / corr / anchorStd / vStd** | "规划头有没有用"必须配对比较才说得清：实测关掉搜索 0 胜 4 负 0 和、开规划 0 胜 2 负 **2 和**；而**深搜反而更差**（同一 TB 骨干 256→1024 节点：12.5% → 0%）—— 叶子没训准之前堆深度是负收益。**尺子本身也骗过人**：第一版的 gap 降了 6.5 倍其实是把 V 压成常数（corr 没动），加 corr/vStd/锚标准差才看得见；即使 V 训到 corr 0.869 也仍 0 胜 4 负（蒸馏手工评估的上限就是手工评估），详见 `agent_dqnab_design.md` §7.3~§7.5 |
| `tools/verify_thinking_ui.ps1` | 驱动真实窗口 + 采样像素，量"思考中有状态条、空闲没有" | 两个坑：状态条是**混色**的（`QColor(28,38,54,230)` 叠在米黄上 ≈ `(47,52,62)`，按原色找不到）；默认 agent 只算 ~150 ms，点完再 sleep 就错过了 |
| `tools/verify_match_ui.ps1` | 走 Windows UI Automation 驱动按钮并**读回结果文字** | `SendKeys` 只在窗口拥有前台时有效，而 `AppActivate` 在控制台占前台时会静默失败；UIA 的 `InvokePattern` 与焦点无关，还能直接读控件矩形与文本 |
| `tools/verify_eager_load.ps1` | 断言"启动时全量加载"的两个后果：启动沙漏出现并**自己收起**；首次使用稀疏 MoE 变体**不再**弹沙漏（改回懒加载就 FAIL）；同时要求奖励点数在涨（防止"对局没跑起来所以没弹窗"的假通过） | 替换掉原来的 `verify_busy_lazy.ps1` —— 那个脚本断言的是**相反**的事实，行为一改它就永远红着。**一个只能在旧行为下通过的测试要重写，不能留着** |
| **"把值钉成常数"做符号实验**（`test_dqnab [10]`） | 检查 TD 目标是 negamax (`r − γ·max Q`) 还是单 agent 的 (`r + γ·max Q`)：把 A 头输出层清零（A ≡ 0 ⇒ Q ≡ V）、把 V 头偏置设成 20（`tanh(20) = 1.0f` 精确饱和 ⇒ V ≡ +1），于是 `qStar ≡ +1`，目标必须落在 `r − γ` 上 | **负号写错照样训练、照样不报错**，它只表现为"永远收敛到错的值"。把被检验的公式里的变量钉成常数，是唯一能让"一个符号"变成可判定事实的办法。实测 −0.9905 vs 单 agent 口径 +0.99 |
| **马尔可夫完整性：同一局面 + 不同历史 ⇒ 编码必须不同** | 走一个 4 手可逆循环回到原局面，断言 14 个棋子平面**逐位相同**而重复平面变化（0.000 → 0.333）；再断言 `repetitionCount()>=3 ⇔ Chess::isRepetition()` | "编码漏了历史依赖"不会报错，它只是让 V 学不到东西。**特征的判据必须和引擎的判终局判据抄同一份**（同一窗口、同一阈值），否则网络学的是另一个游戏 |
| **测"对称性"时不要引入第二套随机权重** | 镜像对称检查改成：**同一个棋盘对象**上按交换后的颜色重摆局面，用**同一个 agent** 编码两次再比 | 第一版用两个 agent（各自随机初始化）比 V，结果差 0.28 —— 那差的是权重，不是编码。同一时刻还发现 `mirrorBoard` 只交换 `color` 字段，而 `isInCheck` 按 `&redJiang/&blackJiang` **对象**取将帅 ⇒ 镜像局面在引擎眼里是错的（将军平面差 1.0） |
| **先检查"尺子有没有信号"再谈指标** | 手工锚的标准差 / V 的标准差 / 相关系数三个数一起打（bench 与测试都打） | 手工锚标准差 **0.0074** 时，"gap 降 6.5 倍"是把 V 压成常数（corr 没动）；corr **0.951** 而 vStd **0.0072** 时，叶子值近乎常数、搜索照样瞎。**任何单看一个指标的结论都可能是假的** |

### 4.5 稀疏路由 MoE，以及一个把功能卡住两轮的拷贝构造 bug

用户提的方案是"不降维、用另一套稀疏路由、减少专家数与 head 数，能不能提升"。
过程与全部实测数据在 `docs/agents_design.md` §11.4.2，这里只留两条最该记住的：

**(a) 收益是确定的、可复现的**：自己写的 `src/rl/sparse_moe.hpp` 只计算门控选中的
top-k 个专家。同一个模板、**同样 28.70 M 参数**，`TopK=1`（稀疏）与 `TopK=E=4`（全算）
在真实 agent 里是 **10.9 vs 42.0 ms/模拟（3.85×）**。这就是 MoE 唯一真正的卖点：
容量不按算力付费。但它换不来绝对算力 —— `d=1260` 上一个 `TransformerBlock` 专家
（O(d²) 注意力 + 8d² FFN）单价就是 3 ms 级，所以带 TB 专家的骨干只能配很低的
模拟次数（界面上那个变体用 16 次，约 160 ms/步）。

**(b) 真正花掉时间的是这个 bug**：`src/rl/layer.h` 里 `iFcLayer` 的**拷贝构造只复制了
维度**，没有复制 `w` / `b` / `o` / `e` / `g` / `v` / `m`：

```cpp
// 修复前
explicit iFcLayer(const iFcLayer &r)
    : iLayer(r), inputDim(r.inputDim), outputDim(r.outputDim), bias(r.bias) {}
```

`SparseMoE` 里 `experts[i] = ExpertFactory<Expert>::make(...)` 是**按值返回再赋值**，
于是只要 MSVC 没省略这次拷贝，专家就成了"维度对、权重空"的空壳（`w.size()==0`），
之后所有 MM 内核都在越界读写 —— 表现是稀疏 MoE 的反向出现 **15615 个 NaN/1e28 元素**，
而且**不同次构建结果不一样**（省略与不省略拷贝的差别），非常难查。

12 行的最小复现把它钉死了（`build/copytest.cpp`，临时文件）：

```
a (直接构造) w.size=10080  w[0]=-0.847383
b (拷贝构造) w.size=0      <-- 修复前
b (拷贝构造) w.size=10080  w[0]=-0.847383  <-- 修复后
```

修复后门控梯度从 2.6e33 变成 2.7e-2（有限），稀疏 MoE 的反向才通过有限差分。

**教训**：`Net` 的拷贝是浅拷贝（共享层指针），深拷贝必须走 `copyTo`；而"按值返回一个
层"这种写法会静默走拷贝构造 —— 在一个含 `std::vector` 成员的类里，"只复制标量成员"
的拷贝构造就是定时炸弹。同类隐患的检查办法：`test_grad` / `test_sparse_moe` 里的
有限差分能同时覆盖"前向用了正确权重"和"梯度填对了位置"，这类 bug 活不过一个 eps。

### 4.6 权重文件格式（有损 / 无校验 / 非原子）

用户要求"实现使用更好的方法保存模型权重"。老格式的问题不只是"土"，而是**会静默毁掉
训练成果**：

| 问题 | 后果 | 现在 |
|------|------|------|
| 十进制文本（`ostream << float`，6 位有效数字）**有损** | 存一次读回来权重漂移 ~1e-6 相对；而后台训练**每轮**都在 `save -> load`（`TMP_WEIGHTS`），等于每轮往网络里注入一次不该有的扰动 | `Tensor::toString` 改成无损编码（base64 + CRC32），实测往返**逐比特相同**、两次存出的文件逐字节相同 |
| 没有任何校验 | 文件被截断 / 翻一个字节 / 把别的 agent 的权重喂进来，都会静默载入一堆形状错乱或数值不对的张量，直到某次前向才崩（或者更糟：不崩但输出全是垃圾） | `CHWGT2 <层数> <结构指纹>` 头 + 每张量 CRC32 + 长度检查；结构不对/数据坏/被截断一律返回失败 |
| 直接写目标文件 | 写一半崩掉或被 kill，**上一次训练好的模型就没了**（关窗正好会打断后台训练） | 先写 `.tmp` 再 `std::filesystem::rename` 原子替换；失败时删掉临时文件、保留原文件 |
| 载入失败会把网络改成半成品 | `fromString` 失败返回空张量，赋给层成员后**下一次前向就是越界读** —— 一个坏文件能把"载入失败"升级成段错误 | 载入前先**零分配**地校验每一行（形状 + 数量 + CRC），全部通过才真正写进网络；`test_weights` 断言"失败后网络逐比特不变" |

实测（`test_weights`，44 条断言）：

```
v2 (base64+CRC32): 497143 字节, save 2.8 ms, load 23.2 ms   (5.34 字节/float)
v1 (十进制文本)   : 885386 字节                              -> 体积 1.78x
truncated / fingerprint / one-bit-corrupt / wrong-structure / missing file -> 全部返回 -1
载入失败后网络的变化 = 0.000e+00
```

另外**老格式的文件仍然能读**（不带 `CHWGT2` 头就走兼容分支），所以以前存下来的权重不用
转换；`test_weights` 第 [3] 节专门验证这条路径（并顺便把老格式的有损程度量了出来：
前向输出差 1.6e-06）。

顺带的一个坑：第一版预校验直接调用 `Tensor::fromString` 做检查，于是 16 MB 的老格式权重
文件要**解析两遍**、每行 `split()` 出几百万个 `std::string` —— GUI 启动从秒级变成 30 秒
以上（脚本等不到"开始对弈"按钮变 enabled 才发现）。改成"扫一遍字节 + 数一遍逗号"的
零分配校验后恢复。**教训**：校验逻辑要和解析逻辑一样在乎常数。

界面的奖励曲线在 `ChessBoard::playMatchGame` 里显式换算成走子方视角。

**影响**：这个 bug 从"给 RL 即时奖励"那一版代码起就存在，是"RL agent 学不动"的一个独立
成因（另一个见 `docs/agents_design.md` §10 的样本效率分析）。它不影响搜索类 agent
（Alpha-Beta / MCTS / EVAB 的搜索部分）。

---

## 一、致命 / 高严重度

### A1【致命】`StoneMap` 未初始化 + `Chess` 构造函数不清 `m_map` → 野指针

`src/stone.h:84-86`
```cpp
T* data[10][9];
public:
    StoneMap(){}
```
`src/chess.cpp:150-185`：`Chess::Chess()` 的初始化列表与构造体都**没有** `m_map.clear()`。
只有 `reset()`（`src/chess.cpp:277-306`）会清整张表。32 个棋子构造函数只写自己所在的
32 格（`src/stone.h:343`），其余 58 格是未初始化指针。

失败路径：`src/chessboard.cpp:151 return chess.m_map[pos];`、
`src/chessboard.cpp:161-162 if (stone != nullptr) { if (stone->color != color)`、
`src/chessboard.cpp:300-302`。Debug 构建（MSVC `0xCD` 填充）下点击任意空交叉点必崩；
Release 只是恰好新页为 0 才"能跑"。**`test_ab` 的段错误就是本项。**

修复：`StoneMap()` 里 `clear()`，或让 `Chess::Chess()` 结尾调用 `m_map.clear()`（更彻底是调 `reset()`）。

### A2【致命】走法合法性完全缺失（自杀 / 不应将 / 照面），`isInCheck()` 是死代码

- `src/chess.cpp:344-358` `sample()` 只做 `if (alive) getPossibleSteps(steps)`，无任何过滤。
- `src/chess.cpp:371-407` 定义了 `isInCheck()`（含飞将检测），但**全仓零调用**
  （`grep isInCheck` 只命中定义 `src/chess.cpp:371` 与声明 `src/chess.h:69`）。
- GUI 玩家同样不校验：`src/chessboard.cpp:168` / `:195` 只调 `tryMoveTo()` 就 `moveForward()`。
- `src/stone.h:589-610` 把"两将照面"实现成**合法的吃将走法**，却没有把"走后照面"设为禁着
  → 引擎会主动制造照面局面并"吃将"结束游戏。

修复：新增 `Chess::isLegalMove(Step)`（落子 → `isInCheck(自己)` → 回退），
在 `sample()` 与 `chessboard.cpp` 的落子处统一走它；被将时只保留应将走法。

### A3【高】没有将杀 / 困毙判定，对局只能以"吃掉将帅"结束

`src/chess.cpp:360-369`
```cpp
int Chess::isGameOver()
{
    if (redJiang.alive == false)   return Stone::COLOR_BLACK;
    if (blackJiang.alive == false) return Stone::COLOR_RED;
    return Stone::COLOR_NONE;
}
```
没有任何地方检查"当前走棋方还有没有合法走法"。

### A4【高】重复局面 / 长将 / 60 回合自然限着全部未实现

`src/chess.cpp:425-441`：`pushHistory()` **从未被调用**（只有定义处），
`history` 恒空 → `isRepetition()` 恒 `false`。
且 `computeHash()`（`src/chess.cpp:409-423`）只用棋子 id 与坐标异或，**不含走棋方**，
即使接上也存在误判。

### A5【高】回放模式进入后无法退出（棋盘永久失去响应）

`replayModeExited` 信号在 `src/chessboard.h:88` 声明、`src/mainwindow.cpp:57` 连接，
但**没有任何 `emit`**。`loadReplayGame()` 设 `m_replayGameId`（`src/chessboard.cpp:790-818`），
而 `reset()`（`src/chessboard.cpp:397-404`）不清它 → `mousePressEvent` 的
`if (isReplayMode()) return;`（`src/chessboard.cpp:296`）永久生效，只能重启程序。
`onReplayModeExited`（`src/mainwindow.cpp:176`）是死代码。

修复：`reset()` 里 `m_replayGameId = -1; emit replayModeExited();`（一行级）。

### A6【高】线程模型：两个 detached 线程与对象生命周期脱钩

- `src/mainwindow.cpp:83-87`：`std::thread loadThread([this]{ ui->gameWidget->startupLoad(); }); loadThread.detach();`
  捕获 `this`/`ui` 并访问成员；关窗即 UAF。
- `src/mainwindow.cpp:232-307`：`onSelfPlay` 同样 detached，线程里直接 `board->selfPlay()`。
- `src/chessboard.cpp:567-596` 的 `selfPlay()` 在后台线程 `chess.reset()` / `moveForward()`，
  而 `state` 仍是 `STATE_IDEL` → 玩家可同时点击落子（`src/chessboard.cpp:295` 的守卫不生效）。

### A7【高】共享数据无锁 + `wakeAll()` 未持锁

- worker 写棋盘 `src/chessboard.cpp:369-371`，GUI 读棋盘 `src/chessboard.cpp:236-242`（`paintEvent`）→ 数据竞争。
- `src/chessboard.cpp:330-334` 在**未持有** `mutex` 时 `condit.wakeAll()`（Qt 要求持锁，
  否则丢唤醒 → AI 可能永不落子）；同一文件 `src/chessboard.cpp:391` 是持锁唤醒的，前后不一致。
- `src/chessboard.cpp:365/378` 在工作线程里直接调 `update()`（Qt 不支持跨线程操作 QWidget），
  而 `src/chessboard.cpp:393` 用的是正确的 queued 调用。

### A8【高】`Steps` 无锁全局对象池被多线程并发使用

`src/stone.h:48-73`：`std::list<Step*>` 的 `get()/put()` 无同步，
而 `ChessBoard::process` 线程、`backgroundTrainLoop` 线程、`onSelfPlay` 的 detached 线程
都会经 `chess.sample()` 访问它 → 链表指针破坏 / 堆破坏（典型"随机崩"）。

### A9【高】RL 即时奖励恒为 0（吃子信号完全丢失）

`src/pgagent.cpp:95-109`
```cpp
float PGEagent::computeReward(const Step &s, int color)
{
    if (s.nextId == Stone::ID_NONE) return 0.0f;
    Stone *victim = chess.stones[s.nextId];
    if (victim == nullptr || victim->alive == false) return 0.0f;   // ← 此时已 alive=false
    if (victim->type == Stone::TYPE_JIANG) return 100.0f;           // ← 不可达
```
调用点在落子**之后**：
- `src/pgagent.cpp:301` `moveForward()` → `src/pgagent.cpp:304` `computeReward()`
- `src/ppomcts_agent.cpp:616` → `:653`（以及 `:895` → `:918`）
- `src/dqnmcts_agent.cpp:678` → `:733`（`recordExperience` 内）

`Stone::moveTo()` 第一步就是 `dst->alive = false;`（`src/stone.h:286`），
故这些调用**永远返回 0**，吃将的 `+100` 分支不可达。三个 agent 只能靠终局 ±1 学习。
唯一顺序正确的是 `src/dqnagent.cpp:253-256`（先算奖励再落子）。

修复：在 `moveForward` **之前**求奖励，或把被吃子价值固化进 `Step`。

### A10【高】"无合法走法"哨兵用错常量 → 会拿默认 `Step` 去落子

`Step()` 默认 `id = 0`（`src/stone.h:19`，即 `ID_RED_CHE1` = 红车），
而调用方判断的是 `Stone::ID_NONE`(33)：
`src/pgagent.cpp:409`、`src/dqnmcts_agent.cpp:414/546/670/727`、
`test/test_pg_main.cpp:104`、`test/test_dqn_main.cpp:78`、
`test/test_ppomcts_main.cpp:131`、`test/test_dqnmcts_main.cpp:85`。
另有 6+ 处脆弱魔法判断：`step.id == 0 && step.nextId == 0 && step.pos.x == 0 && step.pos.y == 0`
（`src/chessboard.cpp:361`、`:583`、`test/test_main.cpp:69`、`test/test_utils.h:123`、`test/test_mcts_main.cpp:59,198`）——
`pos == (0,0)` 是黑方底线起点，合法走法可能误命中。

修复：给 `Step` 加 `bool valid`，或返回 `std::optional<Step>`。

### A11【高】GUI 中 DQN+MCTS 实际在下随机棋

`src/chessboard.cpp:490` `return m_sfDQNMCTS->selectMove(color, 6, true);`
（注释仍写"400次迭代"）。`src/dqnmcts_agent.cpp:363-370`：
```cpp
if (training) {
    float r = (float)std::rand() / (float)RAND_MAX;
    if (r < dqn.exploringRate && !nodes[rootID].childIDs.empty()) {
        int idx = std::rand() % (int)nodes[rootID].childIDs.size();
        bestChildID = nodes[rootID].childIDs[idx];      // 纯随机子节点
```
构造时 `exploringRate = eps = 1.0f`（`src/chessboard.cpp:485`），
而衰减只发生在 `endOnlineEpisode`/`learn`（`src/dqnmcts_agent.cpp:764-778`）——GUI 从不调用
（见 A15），所以探索率恒为 1.0 → 每步都随机。

### A12【高】后台训练线程与 AI 线程竞争同一 agent；agent 懒创建无锁

- 写者持锁：`src/chessboard.cpp:743-761`（`loadModel`/`loadPolicy` 直接改写网络权重）
- 读者**不持锁**：`src/chessboard.cpp:452-461`（`selectMove` 做前向）
→ 一边重写权重一边推理。
- `aiThink`（`src/chessboard.cpp:454-459`）与 `backgroundTrainLoop`（`:663-684`）
  都可能 `new` 同一个 agent，前者不持锁 → 双检失效、指针被覆盖泄漏。

### A13【高】关窗会阻塞在训练线程 `join()` 上

`src/chessboard.cpp:640-646`：
```cpp
void ChessBoard::stopBackgroundTraining()
{
    m_bgTraining = false;
    if (m_bgTrainThread.joinable()) { m_bgTrainThread.join(); }
```
而一轮训练是"4 局 × 50 次 MCTS 模拟 × ≤200 步"（`src/chessboard.cpp:706-741`），
线程只在循环顶部检查标志 → GUI 线程可能被冻结几十秒到几分钟。

### A14【高】SQLite 连接跨线程使用；写库接口根本没接线

- 连接在加载线程创建：`src/chessboard.cpp:32`（由 `src/mainwindow.cpp:84-87` 的 detached 线程调用）
- 查询在 GUI 线程：`src/mainwindow.cpp:105-107`
- `src/gamedb.cpp:54` 用固定连接名 `"chess_connection"` 的单一 `QSqlDatabase` 成员
  → Qt 判定"connection does not belong to the calling thread"，`exec()` 失败，
  历史对局列表恒为空。
- `recordMove/startGame/endGame`（`src/gamedb.cpp:133/156/188`）**全仓无调用点**；
  `m_currentGameId/m_moveCount/m_dbEnabled`（`src/chessboard.h:126-128`）是死成员。
  docs/analysis.md §4 描述的"每步落库 → 可回放"实际未接线。

### A15【高】在线训练接口在 GUI 侧全部未调用

`recordExperience` / `endOnlineEpisode` / `trainAfterMove` / `recordOnline` / `beginOnline` /
`endOnline` 只在自己文件内部或 `warmupFromCurrent` 里出现，`chessboard.cpp` 一次都没调。
后果：`DQNAgent::trainAfterMove`（`src/dqnagent.cpp:643`）、
`PGEagent::beginOnline/recordOnline/endOnline`（`src/pgagent.cpp:474-513`）是死代码；
DQNMCTS 的 `m_trainingMode` 恒 `false`，但 `aiThink` 却传 `training=true`（见 A11）。

---

## 二、中严重度

| # | 问题 | 证据 |
|---|------|------|
| B1 | `moveForward` 丢弃 `moveTo()` 返回值，`moveBack` 无条件复活被吃子 → 把旧局面的 `Step` 应用到新局面时棋盘会被写坏 | `src/chess.cpp:311` vs `src/chess.cpp:328-332` |
| B2 | `Chess::reset()` 不清 `history`，也不复位 `m_moveCount`/`m_currentGameId`/`m_replayGameId` | `src/chess.cpp:277-306`、`src/chessboard.cpp:397-404` |
| B3 | `reset()` 直接把 `state` 打回 IDLE 而不等待/取消 worker → worker 随后把旧局面的走法落到新棋盘 | `src/chessboard.cpp:397-404` + `:369-371` |
| B4 | Alpha-Beta 叶节点处理不对称：只有 MAX（黑方）做静态搜索，MIN 节点纯静态评估 → 奇偶深度系统性偏差、红方吃子序列看不见 | `src/abagent.cpp:266-268` vs `:317-319` |
| B5 | 静态搜索用完全开放窗口 `(-value_infi, +value_infi)` → 丢掉父节点边界，`standPat >= beta` 永不成立，等于不剪枝 | `src/abagent.cpp:318` |
| B6 | 搜索无置换表 / 迭代加深 / 时间控制；`totalReward` 贯穿全搜索却从未被读取（死参数） | `src/abagent.cpp:87-118,256-349` |
| B7 | 搜索内没有"重复局面 = 和棋 0 分"的概念 → AI 会来回推磨 | `src/abagent.cpp:77-127` |
| B8 | 搜索深度三处说法不一致：注释"深度=8"、UI 标签"深度=4"、实际 `ABAgent abAI(env, 5)`；PPO+MCTS 注释 800 实际 80；DQN+MCTS 注释 400 实际 6 | `src/chessboard.cpp:428,433,443-444,480,490`、`src/mainwindow.cpp:187` |
| B9 | 回放构造 `Step` 时 `step->nextId = dbStep.toX`（把列坐标当棋子 id），而 `moveForward/moveBack` 会用它索引 `m_children` | `src/chessboard.cpp:808,844,866` |
| B10 | `getStonePos` 负数整数除法向零截断 → 棋盘上/左 20px 空白被当成第 0 行/列；`x<0 \|\| y<0` 分支永不可达 | `src/chessboard.cpp:127-135` |
| B11 | 硬编码几何、无 `resizeEvent`/`sizeHint`；`.ui` 的 `minimumSize` 600×600 < 绘制所需 614 高 → 压到最小会裁掉最后一排；窗口尺寸三处不一致（`setFixedSize(900,650)` / `.ui` 830×650 / docs 750×650） | `src/chessboard.cpp:113-116,227-233`、`src/mainwindow.ui:26-31`、`src/mainwindow.cpp:18` |
| B12 | `saveModel/loadModel` 无条件 `return true`（底层失败码被丢弃）→ UI 会报假的"保存成功" | `src/dqnagent.cpp:626-636`、`src/dqnmcts_agent.cpp:783-793`、`src/ppomcts_agent.cpp:966-988` |
| B13 | `selfPlay()` 从不返回平局（`score == 0` 也返回红胜）→ "平局"提示不可达 | `src/chessboard.cpp:567-596`、`src/mainwindow.cpp:251-253` |
| B14 | `static ABAgent abAI(chess, 8)` 绑定**第一个** `Chess`，而 `playOneGame` 每局新建局部 `Chess` → 第二局起对已析构对象搜索（UB）。`test_ab` 的段错误与此也有关系 | `test/test_main.cpp:35`、`:49` |
| B15 | `PGEagent::train` 里 `currentColor = COLOR_BLACK` 让黑先走；无走法分支后会二次 `reinforce1` + 二次计数 | `src/pgagent.cpp:209`、`:237-244` + `:345-353` |
| B16 | `GameDatabase::close()` 先 `removeDatabase()` 而 `m_db` 仍持有该连接 → Qt 警告；`deleteGame` 第一条 `exec()` 未检查返回；`recordMove` 每步一次自动提交（一局 200 次 fsync），无事务 | `src/gamedb.cpp:66-75`、`:337`、`:156-183` |
| B17 | QSS 主题完全未接线：`QssLoader::load()` 零调用、`main.cpp` 无 `setStyleSheet`、全仓无 `setStyleSheet`。（同步后 `res.qrc` 已被真正编入 `chess` 目标，但资源仍无人读取） | `src/qssloader.hpp:5-18`、`src/main.cpp:6-13` |
| B18 | `DQN` 的 Q 头是 `Layer<Sigmoid>`（值域 `(0,1)`），而 `DQNAgent` 的奖励含终局 ±1 与被吃红子产生的负值 → 结构上无法表示负 Q。上游只修了 `convdqn.cpp`，`dqn.cpp` 没修 | `src/rl/dqn.cpp`（`snakeAI` 默认分支与 chess 分支都是 `Layer<Sigmoid>`）、`src/dqnagent.cpp:119-132,264` |

### B19【高】`Tensor::MM::*` 是所有 RL 训练慢的根因：**0.11 GFLOP/s**

实测（Release / MSVC 14.44 / `/O2` / 本机，微基准 `MM::ikkj` 与 `MM::kikj`）：

```
MM::ikkj  (90x90)*(90x90)     :  13.274 ms/call   18.21 ns/MAC   0.11 GFLOP/s
MM::ikkj  (90x360)*(360x90)   :  53.016 ms/call   18.18 ns/MAC   0.11 GFLOP/s
MM::kikj  (360x90)^T*(360x90) :  53.088 ms/call   18.21 ns/MAC   0.11 GFLOP/s
```

原因在 `src/rl/tensor.hpp:941-984`：三重循环的内层每次乘加都要调两次
`operator()` → `posOf()`（`src/rl/tensor.hpp:535-544`）：

```cpp
inline std::size_t posOf(Index ...index) const
{
    int indexs[] = {index...};          // 每次访问都在栈上重建索引数组
    std::size_t pos = 0;
    std::size_t N = sizeof... (Index);
    for (std::size_t i = 0; i < N; i++) {
        pos += sizes[i]*indexs[i];      // sizes 是 std::vector，每次读都走一次间接
    }
    return pos;
}
...
x(i, j) += x1ik*x2(k, j);              // 两次 posOf + 两次 vector 取值
```

即每个 MAC 有两组"建数组 + 循环 + 读 vector"，编译器无法向量化
→ 比同规格的手写循环慢约 2~3 个数量级。

**这一项解释了零节里全部的超时**：我做过对照实验——把 DQN 主干从 chess 的
`MOE<16,16>`+`TransformerBlock<16>`（约 1.55M 参数）换成 snakeAI 审计过的默认
`MOE<8,4>`（约 8 倍小），`test_dqn` **仍然超过 1800 s**。所以超时不是网络结构问题，
而是矩阵乘的实现问题。（上游只对 `conv2d()` 做了扁平 stride 重写，
`Tensor::MM` 一直没动。）

修复方向：把 `MM::ikkj/kikj/ikjk` 内层改成先取行指针再线性扫描，例如

```cpp
float *xr = x.val.data() + i*x.sizes[0];
const float *x2r = x2.val.data() + k*x2.sizes[0];
for (std::size_t j = 0; j < x.shape[1]; j++) xr[j*x.sizes[1]] += x1ik*x2r[j*x2.sizes[1]];
```

（`sizes` 提到循环外，`posOf` 不再参与内层），预计可提升 1~2 个数量级。
这一步做完再看是否需要动网络结构。

---

## 三、低严重度 / 代码质量

- **重复代码**：`encodeState / getLegalActions / stepToActionIdx / computeReward / warmupFromCurrent`
  在 4 个 RL agent 里几乎逐字重复；`PIECE_VALUES` 有 4 份定义
  （`src/pgagent.h:41`、`src/dqnagent.cpp:7`、`src/ppomcts_agent.h:37`、`src/dqnmcts_agent.h:36`）；
  MVV-LVA 排序在 `src/abagent.cpp:132-153` 与 `:201-218` 抄了两遍。
- **死代码**：`Chess::isInCheck/pushHistory/isRepetition`、`Chess::get`（`src/chess.cpp:272`）、
  `StoneMap::show`/`Stone::show`（`src/stone.h:154,326`）、
  `chessboard.h:57-59` 三个回放访问器、`MCTSNode::parentID`/`AZNode::parentID`/`DQNMCTSNode::parentID`、
  `AgentBase::resetState`、4 个 `warmupFromCurrent`、`DQNAgent::trainAfterMove`、
  `PGEagent` 的 online 三件套、`gamedb.cpp:246/330`、`qssloader.hpp` 全文件、
  `src/rl/qlstm.*`（无任何 include）、`src/rl/rl.pri`（qmake 遗留，引用的 `bpnn.cpp`/`conv2d.cpp`/`lstmnet.h`/`util.h` 都不存在）。
- **`Step::operator=` 不复制 `reward`**（`src/stone.h:23-33`），而拷贝构造会复制 → 语义不一致。
- **`Chess::evaluate()` 的 `getPositionValue(int type, ...)` 形参 `type` 未使用**（`src/chess.cpp:109`）。
- **`abagent.h:26-27` 形参名误导**：`minimizeAlpha(..., double beta, ...)` 传的其实是父节点边界，
  不是本节点的 beta（docs §5 第 1 条反复出错就源于此）；建议改名 `parentBound`。
- **`STATE_IDEL` 拼写错误**（`src/chessboard.h:33`）。
- **`selectID` 的"空"值在 `-1` 与 `Stone::ID_NONE`(33) 之间混用**
  （`src/chessboard.cpp:90,298,331,390,400`）。
- **const 正确性**：`src/pos.h:22-27` 所有运算符非 const；`stone.h:105,138,154`；
  `src/chess.h:71 double evaluate();` 只读却非 const。
- **`alive` 用 `int` 却到处和 `bool` 比较**（`src/stone.h:244`）。
- **测试无断言**：6 个 `test_*_main.cpp` 全是 printf + 自对弈统计，
  `test/test_utils.h:6` 的 `<cassert>` 全仓没用过一次 → A1/A2/A3/A9/A5 这类缺陷一个都抓不到。
- **文档与代码脱节**：`docs/analysis.md` §一的文件树仍是根目录时代的旧布局；
  §5 第 3 条把"新增 isInCheck"列为已修复（实际零调用）；
  §六 第 3 条说"MCTS 未实现"（实际 `src/mcts.cpp:259-340` 已完整实现四阶段并接入 GUI，
  见 `src/chessboard.cpp:448-450`）；`docs/task_progress_mcts.md:74-75` 说 GUI 用 5000 次迭代（实际 800）。
- **`ChessBoard` 在 AI 线程里 `emit` 后由队列在主线程弹模态框**，
  模态事件循环期间 `reset()` 与 worker 可能重叠（与 B3 同源）。
- **`res.qrc` 之前从未参与构建**（虽然 `AUTORCC ON`，但 `.qrc` 未列进源文件）；
  随本次 RL_CORE 改造已接入（`qrc_res.cpp` 现在会生成）。

---

## 四、与 `docs/analysis.md` §六"仍存在的问题"的核对

> 下表是**首次审查时**的核对结论。其中 1、2 两条**后来都已修复**
> （见 A2/A3/A4/B7/B13），`docs/analysis.md` §六 也已同步更新为"已修复 + 当前真正的待办"。
> 保留此表是为了记录"当时的判断"，不再代表现状。

| docs 条目 | 结论（首次审查时） | 说明 |
|-----------|------|------|
| 六-1 "AI 缺乏应将/将军意识，`isInCheck()` 已实现但未集成" | **确认存在，且比文档更严重** → **现已修复** | `sample()` 无过滤（A2）；`isInCheck()` 零调用；玩家侧也不校验；没有将杀判定（A3）；飞将被实现成吃将走法而没有禁照面规则。**修复**：`Chess::isAttacked/isLegalMove/hasLegalMoves` + `getResult()` |
| 六-2 "游戏无法自然结束，缺长将检测" | **确认存在，表述需修正** → **现已修复** | `pushHistory()` 从未被调用（A4）；没有 60 回合限着；`selfPlay` 从不判和（B13）。修正：GUI 对局**会**结束，但结束条件是"将/帅被吃掉"而不是将杀。**修复**：`history` + `halfMoveClock` + `isDraw()`（三次重复 + 120 半回合），arena 到上限判和 |
| 六-3 "MCTS 未实现" | **已过期，应删除** | `src/mcts.cpp:78-181,194-239,259-340` 已实现 select/expand/simulate/backpropagate + UCB1，并接入 GUI `src/chessboard.cpp:448-450` |

同时，docs §5 第 3 条（"新增 `isInCheck()`"）应从"已修复"移到"仍存在"——
代码在，但从未被调用，等价于未实现。

---

## 零之二点十三、PPO 系列训练效率改造（P1–P7）暴露的问题

这一轮的目标是"同样的算力产出/消化更多训练数据"（改动与实测见
`docs/agents_design.md` §17）。下面是这一轮**发现的问题**，分三类：修掉的、判定为"设计使然"的、
以及**没修的**。

### (1) 修掉的

**① 奖励/价值目标的视角口径不一致（老一代 3 个 agent，9 处）—— 不报错、只让人学不动。**
PG / DQN / DQN+MCTS 的 `encodeState` 是"绝对坐标 + 黑子为正"，网络表达的是**对黑方的价值**；
而 `computeReward` / `rolloutFromCurrent` 给的是**走子方视角**（吃子者为正），
同一批经验里的终局常量 `(gameResult == COLOR_BLACK) ? 1 : -1` 又是黑方视角。
于是**一条轨迹里两种口径混用**，红方那一半样本的目标整体反号。
上一次只改了 `computeReward` 自己、没有跟着改消费端，这一轮补齐：新增
`stone.h` 的 `moverRewardToBlackFrame()` 两个重载，修 `pgagent.cpp`（`exploreAndTrain`，
这条路径**没有**终局覆盖，反号是实打实的）、`dqnagent.cpp`（4 处，含无调用方的 `trainAfterMove`）、
`dqnmcts_agent.cpp`（3 处），以及 `evagent.cpp` 的 `exploreAndTrain` 手工评估项
（它用 `next` 而状态与 `searchV` 都是 `turn`，两项按 0.5/0.5 相加，等于手工那一半一直在跟自己抵消）。
**判定为本来就对的**：`trainVsRandom` 各点（AI 只执黑，黑方视角 == 走子方视角）、
`sacazagent.cpp` 全部（规范视角，全程走子方口径）。逐点判定表见 `docs/agents_design.md` §17.2。

**② 损失曲线的两个读数恒为常数（既有 bug）。** `accumulateGrad`/`trainStep` 在
`backward()` **之后**才读 `v[0]` 与 `policy[i]`，而 `backward()` 会把层的输出清零 ——
`lastLoss` 恒等于 `target²`、`lastActorLoss` 恒等于 `-ln(1e-8)·Σtarget`，
17 轮里一个数都没变，而同一时间策略已经动了 3 个数量级。改成在 `backward()` 之前算标量损失。

**③ `PpoSelfPlayMT` 的 learner 退出条件写在"没攒够一批"的 else 分支里 → 永不退出。**
池子一旦涨到 `learnBatch` 以上就永远"够学"，那个分支再也进不去，worker 全部收工之后
learner 还会拿同一批旧样本空转。改成先判停、再判够不够。

### (2) 设计使然 / 记录备查

* **worker 的局数会超跑**：配额检查在"每局开始前"，W 条 worker 各自看到"还差一点"就会各下一局，
  最多超出 `W-1` 局（8 局的目标在 W=8 时能跑出 15 局）。不是 bug，但**任何拿"实际局数/秒"
  算加速比的写法都会被它抬高**，所以基准现在把实际局数单列并标注"超跑"。
* **PG 的物质收益被整条丢掉**：`PGEagent::train` / `warmupFromCurrent` 算出即时奖励之后，
  会被终局常量把整条轨迹的 reward 覆盖（三处出口全覆盖），所以 PG 实际只用"纯终局 ±1"学习。
  它是自洽的（都是黑方视角），所以**没改** —— 要接上得先把量纲调平
  （`computeReward` 给的是 `value*10`，与 ±1 差一个量级），列为待办。

### (3) 没修的（本轮最重要的结论）

**④ 多线程分身训练的并行收益上限只有 ~1.5×，因为搜索是访存带宽受限的。**
实测（`bench_ppo_mt`，i7-12650H = 10 核/16 逻辑，80 模拟/步、batch=64、epochs=2，
每配置 12 局 × 2 轮）：

| workers | 墙钟秒 | 相对串行基线 | 每线程 ms/模拟 |
|---|---|---|---|
| 串行基线 | 40.5 | 1.00× | 0.422 |
| 1 | 38.7 | 1.05× | 0.384 |
| 2 | 31.8 | 1.27× | 0.541 |
| **3** | **26.3** | **1.54×** | 0.698 |
| 4 | 31.3 | 1.29× | 1.001 |
| 6 | 34.0 | 1.19× | 1.364 |

把 learner 完全摘掉（`--batch` 给一个池子永远够不到的大数）再扫一遍，**曲线形状不变**
（一样在 2–3 条 worker 就到顶），所以 learner 不是限制；诊断输出也算得出来：
`学习轮数 / 理论上限` 只有 0.24–0.37，learner 一直是**喂不饱**的那一方。
而每线程的每模拟成本从 1 条 worker 的 0.294 ms 涨到 8 条的 1.929 ms（**6.6×**）——
这是带宽饱和的典型形状。折算总带宽：每次模拟要流过的权重约 3.0 MB，其中**策略头
`64×8100×4B = 2.07 MB` 占了大头**；实测聚合吞吐在 ~5100 次模拟/秒 到顶 → ≈15 GB/s，
正是笔记本双通道内存的量级。而 8100 维动作空间（`fromCell*90 + toCell`）注定了
每次模拟都要把这 2 MB 权重从头到尾流一遍。

**突破方向（未做，列为待办）**：让一次权重读取服务多条样本 ——
即多棵树的叶子**批量前向**（AlphaZero 那套 server + 批量推理 + virtual loss），
或把权重降到 fp16/int8 把字节数砍半。两者都是独立的大改动。
**结论：在当前硬件上靠堆线程最多 ~1.5×，同样的算力投到数据效率收益大得多**
（P3/P4/P6：同一批数据多次复用、"重放一条 7.7 ms vs 重生成一条 55 ms"、镜像白拿一倍数据）。

### (4) 验证方法学（这一轮最贵的教训）

**⑤ 吞吐指标用"均值之比" → 把加速比抬高了约 45%。** `bench_ppo_mt` 第一版每轮算
`局数/秒` 再对轮次取平均，而 `mean(1/t) ≠ 1/mean(t)`，重复轮次一抖动就系统性偏高：
那一版报出的是"W=4 时 2.22×"，改成**固定目标局数 / 平均墙钟秒**（与基线同口径）后是 1.54×。
→ **吞吐类指标要以"固定工作量的墙钟时间"为主口径，比值类的量不要先取比值再平均。**

**⑥ 用 shell 的文本 cmdlet 来回写源文件，一次性毁掉了 5 个文件里的全部中文注释。**
当时只是想把源码注释里的 `§12` 改成 `§17`，写成 `Get-Content -Raw` + `-replace` + `WriteAllText`。
而**这个环境里的 `pwsh` 实际是 Windows PowerShell 5.1**：`Get-Content -Raw` 按 **ANSI（GBK）**
解码 UTF-8 源文件、再按 UTF-8 写回，一次往返就把中文变成 `瑙嗚鎹㈢畻` 这类乱码；
更糟的是 **GBK 的双字节序列会吞掉紧跟其后的 ASCII**（`name = "车";` 的闭引号被吞成
`name = "?;`，直接是语法错）。反解不完全可逆（每个文件丢 1000–3000 字节），只能 `git checkout` 回滚；
回滚时确认了"工作区相对 HEAD 的差异全是注释"（代码一行没动），所以**代码层没有损失**，
丢的是这 5 个文件里上一轮补充的注释（`stone.h` / `pgagent.cpp` / `dqnagent.cpp` /
`dqnmcts_agent.cpp` / `evagent.cpp`，合计约 270 行注释）。两条规矩：
* **绝不**用 shell 的 `Get-Content`/`Set-Content`/`Out-File` 改仓库里的 UTF-8 文本文件；
  要么用带编码参数的 .NET API 显式指定 UTF-8，要么用专门的编辑工具。
* 判断"文件到底是什么内容"，**不要只信同一个 shell 里的 `Get-Item`/`ReadAllBytes`** ——
  这次它们返回了过期的大小（26273）和完全无关的二进制内容，而
  `git status` + `git diff --numstat` + `cmd /c dir` + `cmd /c type` + 行读取工具
  五路都说是正常的 771 行 C++ 文本。**跨进程交叉验证才能定案，否则会去"修"一个不存在的问题。**

### (5) 顺带做掉：行尾规范化（`.gitattributes` + 全仓 renormalize）

**症状**：提交完 P1–P7 之后体检，4472 行改动里有 **278 行（6.2%）是"仅行尾不同"的噪声**，
集中在 `pgagent.cpp`(+92) / `dqnmcts_agent.cpp`(+86) / `ppomcts_agent.cpp`(+78) /
`dqnagent.cpp`(+22)。评审时看不出"哪一行是真的改了"。

**根因不是那一次编辑，而是仓库长期没有 `.gitattributes`**：结存里同时存在两种行尾 ——
实测 70 个 `i/lf` + 51 个 `i/crlf` + 15 个 `i/lf`-`w/crlf` + **4 个 `i/mixed`**。
那几个文件**自身就是混行尾**（例如 `dqnagent.cpp` 755 行里 742 行 CRLF、13 行 LF），
于是任何一次工具编辑都会把它"规范化"一遍，顺手把那十几行也改掉。
`core.autocrlf=true` 只管 add 时的转换，**不会回头修正已提交的内容** —— 51 个 `i/crlf`
就是这么留下的。

**修法**：新增 `.gitattributes`（`* text=auto` 为主，`*.bat`/`*.ps1` 显式 `eol=crlf`，
二进制类型显式声明），然后 `git add --renormalize .`。约定是**仓库里一律存 LF**，
检出到工作区按平台转换（Windows + `core.autocrlf=true` → CRLF）。属性优先于
`core.autocrlf`，所以从此不再依赖每台机器上的 git 配置。

**安全复核**（这一步不能省，规范化会把 55 个文件都标成"改过"）：

* 逐文件比对 `git diff --cached --numstat` 与 `git diff --cached --ignore-cr-at-eol --numstat`
  —— 55 个文件忽略行尾后**全部为空**，唯一有真实内容变化的是新增的 `.gitattributes` 本身。
  合计 19477 行 CRLF → LF，**没有一行代码/文字内容改变**。
* 规范化后 index 行尾分布：**141 个 `i/lf` + 2 个 `i/-text`**（`app.ico` / `app.png`），
  不再有 `i/crlf` 与 `i/mixed`。
* **酸测试**：给一个 `w/crlf` 的文件用编辑工具加 1 行，`git diff --numstat` 返回
  **1/0**（以前是整文件重写）—— 证明"改一行=改半个文件"的病根去掉了。
* 这不是纯理论收益：仓库里 141 个文件的差异从此只反映真实改动。

**一个反直觉的坑（复核手法本身）**：`git diff --ignore-cr-at-eol --name-only` **仍然会
把纯行尾差异的文件列出来**（它影响的是 hunk 内容，不是"这个文件算不算不同"的判定），
所以拿它当"有没有内容改动"的判据会得到"55 个文件都有内容改动"的错误结论。
**可靠的判据是逐文件看 `--numstat`：忽略行尾后为空才算纯行尾。**

---

## 零之二点十四、训练流程优化（Phase 0–5）暴露的问题

改动与实测汇总见 `docs/training_optimization.md`。这里只记**问题**。

### (1) 修掉的

**① 奖励量纲把"赢"变成了次要目标（本轮的主问题）。** 实测：一方全部非将子力换算成奖励
是 **35.0**，而终局只有 **±1.0** —— 终局/全材质 = 0.029，吃一个車 (+5) 等于赢五盘棋。
**在这个奖励下最优策略是"吃子"而不是"赢"**，这直接解释了"将受威胁时下得对（将附近的
信号大到能压过一切）、整体却被将死"。同时吃將单设了 +100，让同一个胜负事件有两种量级
（而且自对弈路径会把这个 100 直接写进轨迹、经逐手翻号放大成 ±100 量级的目标）。
已修：奖励尺度集中到 `stone.h`（材质系数 0.1 / 终局 ±1 / 每步代价 −0.001），5 个学习型
agent 的 `computeReward` 全部走共享的 `stepReward()`；吃將不再单设奖励。三条不变量
（终局 > 一方全材质 2.86x、> 单次最大吃子 20x、> 最长一局步长累计 8.3x）由
`test_reward_diag` 断言。**顺带发现 SACAZ 的旧尺度也有同一个毛病**：单个吃子确实 < 终局，
但一局累积 3.5 仍是终局的 3.5 倍。

**② 价值目标几乎全是 0（"安静局面没有位置感"的真身）。** 截断的 rollout 一律传
`finalOutcome = 0`，材质又是唯一的稠密信号 ⇒ 诊断实测 `|value target| > 0.1` 的样本占比
**0.0%**，critic 只能学成常数。已修：在线路径**自举**（`finalOutcome = -V(s_end+1)`）
+ **势能塑形**，占比升到 **55–63%**。

**③ 势能塑形的边界项漏了平移。** PBRS 要求终局常量也一起平移
（`finalOutcome' = finalOutcome - Φ(落子后局面)`），漏掉时"塑形 − 不塑形"会差一个
`−Φ(s_n)`（实测 −0.02187，正确值 0.00000）。**这条是测试抓出来的，不是读代码看出来的** ——
值得记：这类"量纲/边界"错误不报错、不崩溃，只能靠"手算一个 2~3 步的小例子对数值"钉住。

**④ 损失曲线上报的是"一条样本"的损失。** `learnSelfPlay` 逐步 `trainStep`（每步 batch=1），
循环覆盖后只剩最后一条的 `(V−target)²`；实测"最后一条 / 整条批平均"差 1.5 倍以上，
遇到重尾样本差几个数量级 —— 这就是"损失曲线像低占空比脉冲"的直接原因。已修：整条轨迹
累积后一次 `applyGradients`，上报批平均（顺带每样本便宜 2.9x），并暴露出 actor 交叉熵。

**⑤ 搜索展开是随机挑未展开着法，先验完全没参与。** 三个搜索入口都是
`std::rand() % untried.size()`：中局约 40 个合法着法、一次决策 80 次模拟，于是**前 ~40 次
模拟全花在随机铺开 40 个孩子上**（每次还都要跑 actor + critic 前向）。已修：改成
`pickUntriedByPrior()`（按 `pi(s_parent)[a]` 最大者展开）。顺带把 `std::rand()` 从搜索里
清掉，恢复 `RL::Random::setSeed()` 的可复现性。

**⑥ 棋盘局面价值并进 `evaluate()` 会把 AB 的搜索深度吃回去。** 实测一次评估从
**0.100 us 涨到 0.650 us（6.5 倍）**，而 AB depth-4 一步要调约 170 万个叶子 ⇒ 单步
**107 ms → 1269 ms（12 倍）**。已修：拆成两档 —— `evaluate()`（AB/EVAB 的叶子评估，
保持不变）与 `evaluatePositional()`（只给 RL 的势能 Φ，每手 2 次）。

### (2) 未修（本轮明确留白，全部列在 training_optimization.md §8）

**0. 棋力层验证的结论：做了，但在 300 局预算下不可分辨。** 训练 300 局（塑形开 / 关各一份
权重）后与 ABAgent 等时间对局，两个种子各 20 局：

| 配置 | seed 20240901 | seed 20250101 | 平均手数 |
|---|---|---|---|
| 未训练（随机权重） | 0-18-2（5.0%） | 0-18-2（5.0%） | 48.8 / 40.8 |
| 塑形开，训练 300 局 | 0-20-0（0.0%） | 0-19-1（2.5%） | 28.1 / 43.1 |
| 塑形关，训练 300 局 | 0-19-1（2.5%） | 0-19-1（2.5%） | 43.2 / 34.5 |

**"5% vs 0%" 就是 1 局之差** ⇒ 20 局、每局 ±1 的协议噪声带约 ±5%，测不出 ≤5% 的差别。
所以只能说"没有可测提升"，**不能说"塑形有害"**。方向一致的现象是训练过的两组平均手数更短
（37 手 vs 45 手，更快被将死而不是拖到判和），与"塑形让 critic 的估值趋近 Φ（含将安全/空间，
偏进攻）而训练量只有要求的 0.03%"这个假设相容（未证实）。
**这次验证的真正收获是协议本身**：任何"改动是否提升棋力"的问题都必须先换协议
（选点一致率 / ≥100 局 × 3 种子 / 训练量-棋力曲线），否则得到的是噪声。详见
`docs/training_optimization.md` §7。

1. ~~最重要的一条：奖励/评估改动还没有棋力验证~~ → 见上面第 0 条：已做，结论是"该预算下
   不可分辨"；下一步是换更灵敏的协议与更大的训练预算。
2. **优势加权**：actor 损失仍是纯交叉熵。自对弈路径的目标是访问分布（AlphaZero 口径，
   不需要加权），但**在线路径的 one-hot 目标在模仿自己的探索噪声**，这条该补。
3. **目标归一化**（running mean/std）。
4. **Phase 3 的两项**：三条训练路径（`trainSelfPlay` / `exploreAndTrain` / MT worker）仍是
   三份近似重复实现；对弈时默认开着在线训练，会污染"谁更强"的结论。
5. **搜索侧**：~~策略头只算合法列（估算 3.0 MB → 0.9 MB/模拟）~~ **已做（R1，2026-09）**：
   推理侧只算合法列，权重流量 5.81 → 3.75 MB/模拟，实测单线程 1.64×、多线程每线程
   ms/模拟 1.4–2.7×（旧估算高估了：策略头只占流量约 1/3）；副作用是先验改成"合法集上
   归一化"后约 2% 的局面换手，见 `training_optimization.md` §9.4。
   剩下的：静止搜索、~~置换表 + 子树复用~~ **已做（B-5，2026-09）**：置换表 + 子树复用实现并
   验证（命中率 ~90%，每手白拿 ~10% 有效模拟），但实测**改变不了决策**（80/400 模拟下两侧
   选点 100% 相同），见零之二点十六；top-K/渐进放宽仍未做。
6. **特征**：编码里有"被对方攻击"却没有"我攻击的格"（会改 STATE_DIM；当前 `weights/` 不存在，
   代价只是重训）。
7. **自对弈路径的截断自举**：`commitEpisode(traj, 0.0f)`（走到手数上限）仍是截断。

---

## 零之二点十五、R1.5：内核语义 / 反向 GEMV / DQN Q 头（四项"不烧时间"的收尾）

R1（策略头只算合法列，见 `training_optimization.md` §9.4）之后，把待办里**不需要大预算、
不需要重训**的四项一次做掉。四项都是"小改动 + 能用现成尺子量"，所以都配了可复核的数字。

### 1. `MM::ikjk` / `kijk` 的 SIMD 内核改成累加（原 P1-1，正确性）

* **问题**：四个内核的语义是 `z += …`（梯度累加；`lstm.cpp` 连写五次累加到同一个 `delta.h`
  就靠它），但 SIMD 版 `ikjk`/`kijk` 写的是 `z(i,j) = dot(...)` —— **覆盖**。当时所有调用点
  的 `kdim=1` 都走标量路径，所以没暴露；一旦喂多列输入（批量化）就会**静默丢掉 z 里已有的
  值**，不报错。
* **修法**：`simd/avx2func.hpp` 与 `simd/sse2func.hpp` 的 `MatMul::ikjk/kijk` 改成 `+=`，
  并把"四个内核一律累加"写进 `Tensor::MM` 的契约注释。
* **验证**：`test_grad` 的 C 节原来只**打印**这个差异，现在改成硬断言（新增 kijk 探针）。
  修前 `k=32 (命中 SIMD)` 那两行是 `**覆盖 -> 只保留最后一次**`，修后四行全是
  `累加 -> 与标量语义一致`，11 项断言 0 失败。

### 2. 补反向 GEMV（原 P1-2，训练步最慢的一步）

* **问题**：`ei = wᵀ·e`（每个 `Layer::backward` 都跑）没有 GEMV 内核 —— `mm_kikj` 要求每一维
  ≥ 一个向量宽度，而 `ei` 是一列，判据不成立 → 掉回标量循环，且**循环方向选错**：
  按 `(i,k)` 遍历时 `w[k][i]` 的步长是 `in`（跨步 gather，向量化不了）。
* **修法**：`Tensor::MM::kikj` 里加反向 GEMV 分支，循环翻成"**外层 k 广播 e[k]、内层 i 整行
  累加**"（两条内存都是单位步长）。写出步长乘子的话 MSVC 认不出单位步长，仍不向量化
  （实测差 2 倍）；再加局部 `__restrict`（调用方本来就要求输出与输入不是同一块内存）。
  累加顺序仍是 k 升序，与标量分支只差 4 路展开的加法结合顺序（与 `ikkj` 同一约定）。
* **验证与实测**（`test_grad` D 节；这台机器噪声大，每侧取 3 轮最小值）：

  | 量 | 改动前 | 改动后 |
  |---|---|---|
  | `kikj` (360×90) | **1.060 ns/MAC** | **0.070–0.098 ns/MAC**（≈ **13×**） |
  | 反向 / 前向 的代价比 | **12.6×**（文档旧值 9.7×） | **0.8×**（反向现在比前向还快） |
  | `PPO::learnFromReplay(64,2)`（128 条样本的前向+反向 + 一次优化器） | **490 ms** | **304 ms**（**1.61×**） |
  | 逐样本 `trainStep` ×128（优化器占大头） | 1904 ms | 1649 ms（1.15×） |

  正确性：逐元素对着朴素实现 `ei[i] += Σ_k w[k][i]·e[k]` 比，最大偏差 1.1e-06；并特意用
  **非零初值**验证它是累加而不是覆盖。
* **顺带发现**：`lstm.cpp:143-148` 那五处 δh 累加写的是 `kijk`，形状契约
  （`x1.shape[0] == x2.shape[1]`）**不成立** —— 它靠"列向量下 `x2(j,k)` 与 `x2(k,j)` 都落到
  `x2[k]`"碰巧算对，多列输入会越界。已改成正确的 `kikj`（列向量下结果逐位相同）。**注意：
  仓里没有任何测试覆盖 LSTM/SSM/Mamba**，这条依据是下标代数 + 新的契约断言，不是回归测试。

### 3. `MM::*` 入口的形状契约断言（原 P1-3）

* **修法**：`requireShape2d()` + 每个内核三条形状关系，全部包在 `#ifndef NDEBUG` 里 ——
  **Release（NDEBUG）下一个字节都不变**。想验证就编译时加 `/UNDEBUG`。
* **验证方法**（可复核）：先确认 `/UNDEBUG` 真的把断言打开（用一个 `assert(1==2)` 的小程序，
  实测 abort + 打印 "Assertion failed"），再把 `test_grad` 与 `test_sparse_moe` 用
  `/UNDEBUG` 重编译跑一遍：**11 项 + 67 项断言全过**，说明这两个测试覆盖到的所有 MM 调用点
  （MLP / MOE / TransformerBlock / Attention / TanhNorm / ScaledConcat）都满足契约。
  这套断言**当场抓到了上面那处 `kijk` 越约调用**（第一次跑就 abort 在
  `x1.shape[0] == x2.shape[1]`）—— 这正是它存在的意义。

### 4. DQN 的 Q 头 `Layer<Sigmoid>` → `Layer<Linear>`（原 P2-6）

* **问题**：Q 头被压在 `(0,1)`，而象棋的奖励里有负项（输棋 -1、被吃子），TD 目标经常为负 ——
  网络**结构上**表示不出来。`convdqn.cpp` 早因为同一个理由改过并在那边留了实测
  （旧配置下 Q 全卡在 0.000、四选一等于瞎猜：负目标把 sigmoid 推进饱和区、导数≈0、梯度死掉），
  `dqn.cpp` 当时漏改。
* **修法**：四个候选骨干分支的 Q 头（`QMainNet`/`QTargetNet` 各 4 处共 8 处）一起改成
  `Layer<Linear>`，避免换分支时把 bug 换回来。
* **验证**：`test_dqn` 新增 `probeQRange()` —— 打印 Q 的 min/max 与负值个数（未训练、训练 20 局
  后各一次）。Sigmoid 头**结构上不可能**出现负值，所以"出现负 Q"就是这条修复的直接证据。
  实测：未训练 **min=−8.81 / max=+8.27、67/128 为负**；自对弈 20 局后 **min=−3.40 /
  max=+8.09、4/128 为负**（值在收敛，负值仍可表示）。另外 `test_dqn` / `test_dqnmcts` /
  `test_evab`（拿 DQN 当对手）全部重跑通过。

### 五项一起的回归（都在改动后重编译 + 重跑）

`test_rules` / `test_weights` / `test_sparse_moe` / `test_grad` / `test_ab` / `test_mcts` /
`test_match` / `test_pretrain` / `test_pg` / `test_ppomcts` / `test_dqn` / `test_dqnmcts` /
`test_sacaz` / `test_evab` **全部 exit=0**（慢的几个：test_mcts 310 s、test_dqn 225 s、
test_dqnmcts 321 s、test_evab 500 s）；`test_reward_diag --moves=60` 22 项全过；
`chess.exe`（Qt 界面）重新构建通过。

另外复核了 R1 的读数**没有被这轮内核改动带偏**（推理路径本来就与 MM 的改动无关）：
`bench_policy_agreement --ab=1` 两次跑的数字与 R1 那轮**逐位相同**
（bc8k_d4：一致率 30.5%、Z 均 0.5118、配对选点一致率 97.5%、每步 11.7 vs 19.4 ms）。

训练侧的端到端效果还体现在串行路径上（`bench_ppo_mt` 的串行基线 = 搜索与学习同线程，
学习在关键路径上）：**0.437 → 0.559 局/s（1.28×）**，每模拟 0.238 → 0.186 ms；
多线程那几档（learner 另占一条线程、本来就"喂不饱"）在噪声内没变（最优 0.87 → 0.89 局/s）。

---

## 零之二点十六、B-5：置换表 + 子树复用（**已实施**，但实测对决策是"中性"的）

### 做了什么

原来 PPO+MCTS **每 ply 都 `nodes.clear()`**：走一步就把整棵树扔掉，下一步从零重搜。
而搜索的结果本来大部分可以留给下一步 —— 落子到达的那个局面，正是上一棵树里的一个子节点。
现在做两件事（`src/ppomcts_agent.{h,cpp}`）：

1. **置换表**：`unordered_map<局面键, 节点下标>`，键是 `Chess::computeHash()`（Zobrist，
   已含走棋方）。子节点创建时登记一次（32 次 XOR），下次搜索同一局面直接命中。
2. **子树复用**：命中后把那个节点**当作新根**，它的子树、先验、访问计数、Q 估计全部继续用
   （PUCT 从继承来的统计上继续，而不是从 0 开始）。

三个搜索入口（`selectMove` / `trainSelfPlay` / `warmupFromCurrent`）统一走 `acquireRoot()`，
"没有合法走法"的分支也收进它（移动生成只做一次）。

### 哪些是刻意的取舍（写成断言/显式重置，不靠运气）

* **跨局必须失效**：命中只在"从上一棵树的根往下 ≤ 2 步可达"时才接受（`reachableWithin`）。
  自对弈里我们的着法 = 1 步、对手的应着 = 2 步；而**上一局的节点（包括每个新对局都会遇到的
  初始局面）必然不可达** ⇒ 自动新开一棵树。另外每局开头显式 `resetSearchTree()`，
  `loadModel()`（换权重）与 `beginOnline()`（新的人机局）也会重置。
  `treeReuse = false` 时 `acquireRoot` **逐字复现**旧行为（每 ply 清树），用于 A/B。
* **策略目标会变**（行为变化，不是等价优化）：自对弈的 π ∝ 根访问计数，复用后这些计数
  **跨 ply 累积**（AlphaZero 的标准做法）。等模拟数下目标分布更尖。
* **重复局面/60 回合**：Zobrist 键不含 halfMoveClock 与历史；树内本来不做终局判定
  （叶子用 critic），所以与既有口径一致，判和仍由 `getResult()` 在 rollout 那层负责。
* **内存**：复用让同一局的树越滚越大（这正是收益来源），`treeNodeCap = 50000` 兜底。

### 验证（test_ppomcts 测试12，4 项全过）

* **[a] 命中就是那个节点**（不是"差不多能用"）：手工走 agent 选中的那一步，第二次搜索的根
  **恰好**是上一棵树里对应那个孩子（下标相同），键/走棋方/（孩子 ∪ 未展开 = 合法着法集）
  逐项一致，原来的访问计数被继承且新模拟叠在上面 —— 实测 `7 → 71`（64 次模拟/步）。
* **[b] 独立局面下**（局面来自另一局 ⇒ 不可达）：复用开/关的选点 **8/8 逐位相同**，
  且复用命中 0 次。这是"改动不影响正常选点"的凭据。
* **[c] 自对弈**：一局 14 手命中 12 次；复用开的树大小 898（累加），关掉是 65（每 ply 扔）。
* 筑这道测试时踩了一个坑，值得记：第一版把"复用开/关"两个 agent 直接连着构造 —— 网络是
  随机初始化的，**两次构造就是从同一条随机流里拿了两份不同权重**，于是 8/8 全不同，
  看起来像"复用改变了选点"。两边各自播种构造后才逐位相同。配对实验里"同权重"必须显式保证。

### 实测收益：机制成立，但**在当前搜索预算下改变不了决策**（一个负面结果）

`bench_policy_agreement` 新增 `--reuse-ab=1`：让"复用开"的 agent 下一盘（对局由 AB 深度 4
推进，局面保持正常；因为 80 次模拟 > 开局 44 个合法着法，上一棵树的根**所有**孩子都展开过，
对手走的那一步必然在里面 ⇒ 复用照常命中），每个局面再问一次"复用关"的 agent（每 ply 从零），
两侧都与 AB 比较 —— **同一批局面、同一份权重、同样模拟数，只有"要不要复用"一个差别**。

| 权重 | 模拟/步 | 命中 | 继承访问数/步（白拿） | 对 AB 一致率 复用 / 从零 | 两侧选点相同率 | 每步耗时 复用 / 从零 |
|---|---|---|---|---|---|---|
| bc8k_d4 | 80 | 83/90 | 9.0（**11%**） | 20.0% / 20.0% | **100%** | 18.2 / 17.5 ms |
| bc8k_d4 | 400 | 36/40 | 40.4（**10%**） | 20.0% / 20.0% | **100%** | 72.8 / 73.1 ms |
| bc8k_d4 | 1200 | 36/40 | 148.1（**12%**） | 22.5% / 22.5% | 97.5% | 192.8 / 191.1 ms |
| bc20k | 80 | 83/90 | 7.7（10%） | 12.2% / 12.2% | 100% | 15.4 / 14.8 ms |
| bc20k | 400 | 36/40 | 32.1（8%） | 15.0% / 15.0% | 100% | 62.2 / 61.2 ms |
| bc20k | 1200 | 36/40 | 114.2（10%） | 15.0% / 15.0% | 97.5% | 187.7 / 183.3 ms |

读数（老实说）：

1. **机制完整可用**：命中率 ~90%，每一手**白拿约 10% 的有效模拟**（80 → ~89，1200 → ~1348），
   而且这个比例在 80/400/1200 三个预算上稳定（都是 ~10%，因为"第一个孩子能拿到的访问份额"
   与预算基本无关）。
2. **但它几乎不改变决策**：80 与 400 次模拟下两侧选点**逐位相同**（100%），1200 次下 97.5%
   （40 ply 里翻了 1 手）。对 AB 的一致率三档预算下**完全相同**。
   原因是这个搜索的形状：`getPUCT` 对未访问孩子返回 +∞ ⇒ 先把 40 多个孩子全铺一遍，
   真正"深挖"的余量只有 `sims − 分支数`；于是继承下来的那 10% 分摊到 40 多个孩子上，
   动不了 argmax。
3. 因此 §9.3 里对 B-5 的期待（"让搜索更强"）**在本工程的搜索形状下没有被证实**。
   要让复用真正值钱，得先改这两个前提之一：**(a) 每步模拟数远大于分支数**（现在 80 vs 44），
   或者 **(b) 选点不再只看访问数 argmax**（例如用 Q/访问的加权，让小的访问差异能体现出来）。
   这两条都不是"不烧时间"能做到的，所以留作后续。
4. 代价可忽略：每步耗时 +1%（1200 模拟档），内存 ~900 节点/局（上限 5 万）。
5. **标准探针逐位不变**（另一条中性证据）：`bench_policy_agreement`（200 个独立局面、
   `--ab=1`）在 B-5 之后的数字与之前**完全相同** —— bc8k_d4 一致率 30.5% / 起点 49.5% /
   Z 均 0.5118 / R1 配对 97.5% / 每步 11.4 vs 18.2 ms；bc20k 27.5% / 52.0% / 0.5903 /
   98.0% / 11.5 vs 17.4 ms。自对弈吞吐（`bench_ppo_mt`，复用默认开）0.588 局/s 串行基线、
   最优 workers=2 0.934 局/s（1.59× 基线），与 B-5 之前同量级。
6. **默认保持 `treeReuse = true`**（AlphaZero 的标准做法，且实测对决策无损害、无速度代价），
   但自对弈的**策略目标**因"计数跨 ply 累积"而变尖这件事**未做棋力验证** —— 开关与探针
   （`--reuse-ab=1`）都在，随时可以 A/B。

回归：`test_rules` / `test_weights` / `test_sparse_moe` / `test_grad` / `test_ab` /
`test_mcts` / `test_match` / `test_pretrain` / `test_pg` / `test_ppomcts` / `test_dqn` /
`test_dqnmcts` / `test_sacaz` / `test_evab` **全部 exit=0**（慢的：test_mcts 265 s、
test_dqn 202 s、test_dqnmcts 252 s、test_evab 418 s）；`test_reward_diag --moves=60` 22 项全过；
`chess.exe` 重建通过。

---

## 零之二点十七、R2：训练侧也只算合法列（**已实施**）+ c_puct 重扫

### R2 做了什么

R1 改的是**推理**（先验只在合法列上算）。R2 换的是**学习问题**：训练时的 softmax 分母
也只覆盖该局面的合法着法。

* **内核**：`Net::backwardFrom(startIndex, x)` —— 从第 `startIndex` 层往回传到第 0 层
  （`backward` 现在是它的薄封装，非稀疏路径一行未变）。
* **PPO**：`accumulateGradSparse(state, legalIdx, targetIdx, targetProb, valueTarget)`：
  骨干跑到 `h` → 头只算 `legalIdx` 那几行的 logit → **在合法集上 softmax** → CE 对合法
  logits 的解析梯度正好是 `p - t` → 头的权重/偏置只更新合法行、往下传的梯度只由合法行
  构成 → 交给 `backwardFrom` 走骨干。critic 一字未改。
  A/B 开关 `PPO::maskedTrainHead`（默认开）；缺前提（头不支持稀疏 / 没梯度缓冲 / 没合法集）
  自动回退全量口径。
* **合法集从哪来**：`ReplaySample` 新增 `legalIdx`；`PPOMCTSAgent::legalIndicesOf(nodeID)`
  从搜索树取节点的完整合法集（未展开 ∪ 已展开孩子的 `parentAction`，与 R1 那套同源），
  `trainSelfPlay` / `warmupFromCurrent` 每 ply 收集一份与轨迹同步传进 `commitEpisode`；
  镜像增广连合法集一起镜像；多线程分身训练的样本搬运也带过去。
  **注意 legalIdx ≠ actionIdx**：后者只是访问分布的支撑集，拿它当分母等于把"搜索没访问到的
  合法着法"从策略里抹掉 —— 那是另一种（更糟的）学习问题。

### 验证（两条独立证据）

1. **解析梯度 == 中心差分**（`test_grad` 新增 E 节，15 项断言全过）：
   头的权重 6 个抽样点最大相对误差 **1.1e-3**、头偏置 2.1e-6、**骨干（Tanh 层，即
   `backwardFrom` 那条路径）** 4 个抽样点 3.5e-4；不变量 `Σ_legal(p − t) = 0` 到 1e-17。
   踩坑记录：先用 `eps=1e-4` 时骨干的"相对误差"到 0.49 —— 那不是梯度错，是 **float32 差分
   噪声**（噪声地板 ≈ 5e-6，而骨干梯度只有 1e-5 量级）。现在判据是"相对误差 < 5e-3 **或**
   绝对误差到噪声地板"，并**特意挑梯度最大的抽样点**，免得判据变成空话。
2. **训练真的按合法集学**（`test_ppomcts` 新增测试13）：
   * [a] 稀疏口径：目标 0.8/0.2 落在两个合法着法上 → 4 轮后 P(target0)=**0.6047**，
     `actionMasked` 在合法集上和 = **1.000000**；
   * [c] 对照组（`maskedTrainHead=false`）也能学到 0.5183 —— A/B 的另一侧不是死的；
   * [b] **非法行的梯度恰好为 0**（稀疏口径 `0.000e+00`，全量口径 `1.891e-04`）——
     这是"只算合法列"最直接的证据。顺带纠正一个想当然：不能断言"非法行的权重逐位不变"，
     因为 RMSProp 带 weight decay（`w -= lr·decay·w` 对每个参数都生效，与梯度无关）；
     实测 5 轮后稀疏口径下只动 4.99e-03（纯 decay），全量口径下动了 1.17e-01。

### 训练一个 R2 权重：一个重要的副作用被确认

用 R2 跑 150 局自对弈（`bench_ppo_mt --games=150 --save=weights/r2_sp`，291.6 s、0.514 局/s、
35132 样本、795 轮更新）得到 `weights/r2_sp_{actor,critic}`。用选点一致率探针量它：

| 权重 | 对 AB(深度4) 一致率 | 合法集概率质量 Z（**全量** softmax 口径） |
|---|---|---|
| r2_sp（R2 训练） | 2.0–4.7%（≈ 随机水平 3.5%） | **均 0.0049**（最小 0.0031 / 最大 0.0069） |
| bc8k_d4（R1 时代权重） | 30.0% | 均 0.5186 |

两条读数：

1. **r2_sp 很弱**（150 局自对弈、随机初始化起步，命中率接近乱走）—— 这与 §7.x"未训练/随机
   起步的策略很弱"一致，不是 R2 的锅；要拿强策略还得走 BC 蒸馏（见下面的缺口）。
2. **R2 权重的"全量 softmax"已经没有意义**：Z ≈ 0.005 意味着 8100 个槽位里 99.5% 的质量落在
   非法槽位上 —— 那些行**从来没被训练过**（梯度恰为 0），是随机初值。所以：
   * `PPO::action()`（全量口径）、`sparsePolicyHead=false`、`bench_policy_agreement --ab=1`
     这些**在 R2 权重上一律不可用**（不是"略有偏差"，是无意义）；
   * 反过来，这正是 R2 想要的结果：策略被定义在合法集上，**R1 那条"先验被放大 1/Z"的
     副作用从根上消失了**（Z ≡ 1 由构造保证），不再需要靠探针去量它有多大。

### c_puct 重扫：**在当前预算下扫不出落点**（一个负面结果）

R1 让先验的实际尺度变成 `p_full/Z`（Z 均 0.51–0.59），等于把探索项强度放大 ~1.8×；R2 之后
先验是合法集上的概率（Z ≡ 1），`c_puct` 恢复了字面含义。所以"要不要重调 c_puct"必须量。
`bench_policy_agreement` 新增 `--cpuct=`，在 150 个固定局面上扫：

| 权重 | 模拟/步 | c_puct = 0.5 / 0.75 / 1.0 / 1.414 / 2.0 / 3.0 |
|---|---|---|
| bc8k_d4 | 80 | **30.0% / 30.0% / 30.0% / 30.0% / 30.0% / 30.0%**（逐位相同） |
| r2_sp | 80 | 2.7% / 2.7% / 2.7% / 3.3% / 2.0% / 4.7% |

再把预算提到 400/1200（60 局面，标准误 ±6%）：

| 模拟/步 | c_puct=0.5 | 1.414 | 3.0 |
|---|---|---|---|
| 400 | 33.3% | 31.7% | 33.3% |
| 1200 | 26.7% | 25.0% | 30.0% |

**结论（老实说）**：

1. **80 次模拟/步时选点对 c_puct 完全不敏感**（150 个局面 × 6 个值，一致率逐位相同）。
   机制与 B-5 那条中性结论同源：`getPUCT` 对未访问孩子返回 +∞ ⇒ 前 40 多次模拟先把
   ~44 个孩子全铺一遍，真正"深挖"的只剩三十几次；而这三十几次里探索项的量级
   （`c_puct·P·√N/(1+n)` ≈ 0.04–0.25，P≈1/44）**远小于 Q 的差异**，选择完全由 Q 决定。
2. **换成 400/1200 模拟后差异仍然落在噪声里**（±6% 的标准误，而且两个预算下"最好"的值
   还不一致：400 时两端最好、1200 时 c_puct=3 最好）—— 50 个局面的差异在 1–2 个百分点，
   这个尺子分辨不了。
3. 所以**没有可测的落点**，本轮**不动 `c_puct`（保持 1.414）**：R2 已经让它恢复字面含义，
   而"调到多少"这件事在本工程现有的尺子下回答不了（与 §7.4/§7.10 的结论同一性质）。
   要真回答它，得先有能分辨棋力的协议（几千局，或一个与棋力相关的代理指标）。

### 已知缺口（明确列出，不含糊）

* **BC 蒸馏路径还没走 R2**：`bench_ppo_distill` 直接调 `actorP.forward` +
  `CrossEntropy::df` + `actorP.backward` 训 actor（不走回放池），所以生产可用权重
  （`bc8k_*`/`bc20k`）仍是**旧口径**训练出来的。迁移它需要三件事：给每个样本存 legalIdx、
  在训练循环里改用稀疏口径（还要把 critic 那一路剥出去，免得顺手把 critic 也按 valueTarget=0
  训了）、以及把 `actorMetrics` 的 CE / P(AB) 换成合法集口径（旧口径在 R2 权重上无意义）。
  这三件都不难，但本轮没做 —— 所以 **R2 目前只在自对弈/回放这条生产路径上生效**。
* **R2 之后 `weights/` 下的旧权重全部作废**（不同学习问题训出来的先验不能混用）：
  现有 `verify_on/off`、`distill`、`bc8k_*`、`bc20k` 仍然能载入（格式未变），但**不要与
  R2 训练的权重混着比较先验尺度**。
* R2 的**棋力**影响未测（同 §7.4 的协议限制）。

## 零之二点十八、各会话（轮次）× 问题 × 优化方法：汇总与沉淀

> 这一节是**总索引**：把每一轮做过什么、发现了什么问题、用了什么优化方法、实测结论是什么、
> 详细记录在哪一节的哪一处，压成一张表。上面各节是"证据与推导"，这一节是"地图"。
> 写在最后（而不是最前）是因为它引用的东西都在上面。

### 18.1 逐轮汇总

| # | 轮次 / 会话 | 目标 | 发现的问题 | 用的优化方法 | 实测结论 | 详见 |
|---|---|---|---|---|---|---|
| 1 | 首轮审查与修复 | 把"能不能正确跑"钉住 | A1–A15（致命/高：`StoneMap` 初始化、走法合法性、将杀判定、`detach()`、`Steps` 加锁…）、B1–B19（中/低） | 逐行静态审查 + 实际构建运行 + **ASan** + 有限差分梯度检查 | 全部修复并回归 | 零、一、二、三 |
| 2 | 内核与工具链 | 让训练/搜索跑得动 | `MM` 内核逐元素走 `posOf()`（0.11 GFLOP/s） | 扁平指针 + 提升步长 + 4 路展开 + SIMD 分派 | **0.11 → 20 GFLOP/s**；`test_pg` 346 → 26 s | 零之二点五、`docs/rl_sync.md` |
| 3 | 交互与验证 | 决策前探索（仿 snakeAI）+ EVAB 接入 GUI | EVAB 在 GUI 里不可用 | 统一 `exploreAndTrain` 协议（`agentrollout.hpp`） | 接入并验证 | 零之二点七 |
| 4 | 可视化 | "界面像死了" | 长思考期间无反馈 | 棋盘内嵌状态条 + 右侧 ThinkingIndicator（三动画） | 像素/无障碍树采样验证 | 零之二点八 |
| 5 | 对弈 arena | 多 agent 对局 | **堆损坏**（用错维度越界） | 先复现 → 再上 **ASan** 拿调用栈 | 定位并修复 | 零之二点九 |
| 6 | 奖励曲线 | 曲线要有信息 | 即时奖励一直是**黑方视角**（红方白吃子是负奖励） | 统一到"走子方视角"，与终局项同源 | 回归钉在 `test_match` [2.6] | 零之二点十 |
| 7 | EVAB | 让"学会评估的 AB"真的能下 | 三个 bug（与 AB 对弈取胜不了的根因） | 逐个复现 + 修 + 对局验证 | 修复 | 零之二点十一 |
| 8 | 启动与曲线 | 启动/曲线可用 | 启动时懒加载导致首次使用卡顿；曲线口径 | 预加载全部模型 + 载入提速（内存整块读 + 内存预校验） | 启动 19 s → 10 s 量级 | 零之二点十二 |
| 9 | P1–P7 训练效率 | 吞吐 | 稠密 MoE 白算全部专家；逐步 `trainStep` 损失只反映最后一条；无回放池；单线程 | 稀疏路由 MoE（top-2）、批平均损失、梯度累积、回放池 + 稀疏目标、镜像增广、多线程分身（worker/learner 分离） | 稀疏 MoE **42.0 → 10.9 ms/模拟**；**多线程访存墙 ~1.5×**（不是 learner 拖后腿） | 零之二点十三 |
| 10 | Phase 0–5 训练流程 | 让目标有信号 | 奖励把"赢"变成次要目标（吃一个車 = 赢五盘）；截断目标恒 0（`\|target\|>0.1` 占比 0.0%）；局面价值并进 `evaluate()` 把 AB 深度吃回去；展开靠 `std::rand()` | 尺度集中到 `stone.h`、自举 + **PBRS 势能塑形**、**两档评估**（`evaluate` / `evaluatePositional`）、展开按先验；**造尺子**：选点一致率探针 + 功效检验 | 终局/全材质 0.029 → 2.86；`\|target\|>0.1` 占比 0 → 55–63%；AB depth-4 回到 101 ms | 零之二点十四、`training_optimization.md` §2–§6 |
| 11 | Phase 6 收尾 | 按外部建议补四项 | 判和/被将死不终止 rollout（三套终局定义并存）；α 无退火钩子；材质重复计账；威胁项被砍掉了 | 终局口径统一到 `outcomeForMover()`；`shapingAlpha`；`materialRewardEnabled`；威胁项只进 Φ | 22 项检查全过；`evaluate()` 单价与 AB 单步未变（0.100 µs / 101 ms） | `training_optimization.md` §8 |
| 12 | Step 1–3c + clipGrad | 棋力 | 20 局协议分辨力不足（±5%）；先验没参与展开；clipGrad 假设 | 选点一致率探针；按先验展开；梯度蒸馏 / BC 预训练 / 加数据 / 换老师；**受控 A/B 与假设否证** | 一致率 3.0% → 28% → 32%；**胜率一次都没动**（关键负面结果）；`clipGrad` 无害无益，保留默认 | `training_optimization.md` §7 |
| 13 | **R1**（本轮） | 搜索吞吐 | 策略头每次模拟都要整块读 2.07 MB | **推理侧只算合法列**：`forwardTrunk` + 稀疏 logits + 子集 softmax（`PPO::actionMasked`） | 权重流量 5.81 → 3.75 MB/模拟；**单线程 1.64×、多线程每线程 1.4–2.7×**；副作用：先验口径变化 ⇒ ~2% 局面换手（配对量出 97.5–98%） | §9.4 |
| 14 | **R1.5**（本轮） | 反掉几个"小但会咬人"的 | SIMD `ikjk`/`kijk` 是**覆盖**而标量是累加（多列输入会静默丢梯度）；反向 GEMV 无内核且循环方向错；`MM` 入口无契约断言；DQN Q 头 `Sigmoid` 装不下负 Q | SIMD 改累加 + 断言；`kikj` 加反向 GEMV 分支（外层 k 广播、内层 i 整行累加 + `__restrict`）；`#ifndef NDEBUG` 形状断言；Q 头改 `Linear` | 反向 GEMV **1.060 → 0.070–0.098 ns/MAC（≈13×）**，`learnFromReplay(64,2)` 490 → 304 ms；断言当场抓到 `lstm.cpp` 越约调用；未训练 Q ∈ [−8.81, +8.27] | 零之二点十五 |
| 15 | **B-5**（本轮） | 搜索更强 | 每 ply `nodes.clear()`：整棵树扔掉重搜 | **置换表 + 子树复用**（Zobrist 键 → 节点；命中即换根；"≤2 步可达"判据 + 每局重置保证跨局失效） | 命中率 ~90%、每手白拿 ~10% 有效模拟；**但选点 80/400 模拟下 100% 相同**（机制成立、决策中性） | 零之二点十六 |
| 16 | **R2 + c_puct**（本轮） | 从根上解决先验口径 | 训练侧 CE 的分母覆盖全部 8100 槽位（Z<1，先验尺度含糊）；R1 之后 `c_puct` 的"有效值"被放大 ~1.8× | **训练侧也只算合法列**（`accumulateGradSparse` + `Net::backwardFrom`；legalIdx 走回放池）；`--cpuct=` 扫参 | 解析梯度 vs 中心差分 1.1e-3/3.5e-4；**非法行梯度恰好为 0**；R2 权重 Z≈0.005 ⇒ 全量口径失效；**c_puct 6 个取值选点逐位相同 ⇒ 不动（保持 1.414）** | 零之二点十七 |
| 17 | **本轮**：PPO 换专家 + 把 PPO 的优化方法移植到 SAC | ① PPO 的容量 ② 让两个 SAC 拿到 P1–P7/R2 的训练侧方法 | `RL::SAC` **从没被跑过**（两个静默缺陷：critic 的 `[state;π]` 拼接被 `MM` 契约截断、`learn()` 的 `learningRate` 没接线）；SACAZ 的辅助损失批统计混着 MCTS 推理前向；P4 的文档口径不准（"每遍都产生新更新"其实只调一次优化器） | PPO 专家 `MlpExpert`→`TB<16,360>`（E/top-k 8/2→4/1，内存与算力两条硬约束）；`RL::SAC` 重写（稀疏 MoE 骨干 + P3 拆分 + P4 多 epoch + 批统计复位/辅助损失 + 批平均损失 + R2 掩码口径）；`SACAZAgent` + `resetMoeBatchStats()` + `replayEpochs`；新增 `test_sac`（43 断言，进 ctest）与 `test_sacaz` [11] | 专家：2.15 M→38.0 M 参数（17.7×），前向 0.139→3.59 ms（26×）；E=8/top-2 的 actor+critic ≈2.4 GB ⇒ 不可用；R2：非法列头权重**逐位不变**（0/5760）、合法列 2816/2816；P4：批 = batchSize×epochs、优化器只调一次；批统计复位 A==B 而不复位 C≠B | `agents_design.md` §18 |
| 18 | **本轮**：TB 专家 SAC 的对弈验证 + 新 agent **DQN+AB**（AB 当 DQN 的 planning head，文件 `src/dqnabagent.*`，原名 NeuralAB） | ① 验证带 TB 专家的 SAC agent ② 打穿"神经 agent 连和棋都难" | 五个神经 agent 与 AB 对局**全是 0 胜 0 和全败**（PPO 38 局全被将死；BC 2 万局面监督预训练后仍是 0 胜）；SAC+AZ-MoE 对 AB 8 局全 mate；纯 Q-argmax 在一步杀局面上命中 0/8 | `bench_sacaz_vs_ab`（TB 专家 SAC 的静默对弈验证，四种骨干全 PASS）；**新 agent `DQNABAgent`**：稀疏 MoE+TB 骨干 + Dueling 双头（V/A）+ AB 用 Q 排序、用 V 当叶子 + **AB 展开结果当 TD 目标**；规范视角编码 + 双射动作空间（8100）+ 阶段平面（动态子力价值自己学）；`test_dqnab`（**78 断言**，进 ctest）+ `bench_dqnab_vs_ab` | **一步杀 8/8 vs 纯 Q-argmax 0/8**；与 AB 深度 4 对局（随机权重、同 seed）：**关掉搜索 0 胜 4 负 0 和（平均 22 手）→ 开规划 0 胜 2 负 **2 和**（平均 44 手）**；但 MLP 骨干搜到 4.4 层反而 0 胜 4 负（深搜放大未训练 V 的误差）；Dueling 解析梯度 vs 中心差分 **1.7e-05**；镜像编码逐位不变（0.000e+00） | `agents_design.md` §18.7、§19 |
| 19 | **本轮**：DQN+AB 改名为 `dqnabagent.*` + **手工评估锚 / 门控优化**（按"先把 V 训准"的建议做） | V 没训准的时候深搜是负收益（`agents_design.md` §19.4） | 预训练照抄 EVAB 的造数据方式（从初始局面纯随机走 ≤12 手）时，**标签几乎没有信号**：手工锚标准差只有 **0.0074**，于是"gap 降低 6.5 倍"其实是把 V 压成常数（corr −0.504 → −0.547，什么都没学到）；门控阈值照抄 EVAB 的 0.05 会把正常学习**整体冻结**（60 次更新全回滚）；固定 0.25 仍回滚 59/60（clipGrad 下位移 ≈ lr）；12 个探针太少（锚标准差随 seed 差 6 倍） | `handEval` + `netHandStats`（gap/corr/anchorStd/vStd）+ `pretrainValueFromHand`（越训越差整段回滚）+ `learnBatch` 门控（按 lr 缩放 + 重同步目标网）；造数据加**吃子偏置** `probeCaptureBias`；探针 24（门控）/64（预训练）；bench 加 `--pretrain*` / `--gate` 与 gap+corr 输出；`test_dqnab` [9] 段 25 条断言 | 吃子偏置把锚标准差 **0.0074 → 0.2228**、corr **−0.305 → 0.938**（0.33 s）；MLP 8192 局面 × 3 遍才被接受（corr 0.433 → 0.869、vStd 0.007 → 0.091，6.4 s），512/2048 局面被回滚（被拒那次的失败模式是**幅度膨胀 13 倍**）；毒化更新被回滚且主干逐元素还原（对照：门控关则 gap 0.205 → 0.733）；**配对对局**：同一 TB 骨干 256→1024 节点 12.5% → **0%**（深搜更差），坏标签预训练让平均手数 54.2 → 31.5；**即使 V 训到 corr 0.869 仍 0 胜 4 负**（蒸馏手工评估的上限就是手工评估） | `agent_dqnab_design.md` §7.3~§7.5、`agents_design.md` §19.5 |
| 20 | **本轮**：DQN+AB 的**状态补全**（裸棋盘 ≠ Markov 状态）与零和口径审计 | 象棋是双人零和、完全信息、交替行动的 **Markov Game**；三次重复/60 回合/长将都依赖历史，只喂 14 个棋子平面时"同一局面第 2 次与第 3 次出现"编码成同一个向量 ⇒ V(s) 不是 s 的函数，Bellman 备份的前提破掉 | 编码只有 143 个信息位（14 棋子平面 + 2 阶段平面），`halfMoveClock` / 重复栈 / 将军状态都不在状态里；设计文档早期还把零和对称写成了 `Q_red(s,a)+Q_black(flip(s),a)=0`（规范编码下不适用） | 新增 3 个规则平面（无吃子 / 重复次数 / 将军），`STATE_DIM` **1440 → 1710**；`repetitionCount/repetitionPhase/halfmovePhase/checkPhase` 公开；`computeTarget` 公开以便做符号实验；`test_dqnab` [10] 段 12 条断言；测试工具的 `mirrorBoard` 修正为"按交换后的颜色重摆镜像局面" | 4 手可逆循环回到原局面：**棋平面差 0.0e+00 而重复平面 0.000 → 0.333**，第 3 次出现 `getResult = DRAW`；`repetitionCount()>=3 ⇔ isRepetition()`；无吃子平面 0.0000 → 0.0083 → 0.0000；**TD 目标符号实验：`Q≡+1, r=0` 时实测 −0.9905**（negamax `−0.99` vs 单 agent `+0.99`）；V(取景红) == V(镜像取景黑) = `-0.285557`、Q 按下标对齐最大差 0.000e+00 | `agent_dqnab_design.md` §3、§11、`agents_design.md` §19.6 |
| 21 | **本轮**：把"完备 MDP 设计"**推广到其它模型** | 同一套建模缺陷在 5 个 agent 上重复出现，而各 agent 各写一份镜像/规则口径 ⇒ 静默分叉 | ① v2 权重格式的"结构指纹"**只哈希层的类型序列、不含维度** ⇒ 改 `STATE_DIM` 之后旧文件会被**静默载入成另一个形状**（`Layer::read` 是整块替换）；② 第一版公共件 `encodeComplete` 无条件写 5 个上下文平面，而 EVAB 只留了 3 个 ⇒ 越界写 180 个 float，**堆损坏崩溃 0xC0000374**；③ `outcomeForMover` 在公共件里又抄了一份（`chess.h:177` 早有） | 新增 **`src/chessstate.h`**（规范格镜像 / 棋子平面 / 规则上下文 / 动作双射 / 终局口径，全部只有一份）；权重载入加**维度守卫**（元素总数 vs `paramCount()`，在网络被改动前拒绝）；PPO+MCTS **1440→1710**（+3 规则平面，保留自己的 2 个威胁平面）、SAC+AZ **1260→1263**（14 平面 + 3 规则**槽**，稀疏转移随身携带 `ctx[3]`）、EVAB **1260→1530**、DQN+AB 改为调用公共件；各 agent 测试加 Markov 完整性断言 | `test_weights (g)`：同类型不同维度**指纹相同**（96904 vs 114184）但被守卫拒绝，正对照同维度仍可载入（**49 断言**）；SACAZ `test_sacaz` **135 断言**（棋平面差 0.0e+00、规则上下文 0.000→0.333）；EVAB 1530 维循环断言（重复 0.000→0.333、无吃子 0→0.0333）；DQN+AB **115 断言**（`testLearning` 改成"默认 lr 下的两臂配对 + 收敛"，并量出 **lr=0.02 会出现周期 2 极限环**） | `agents_design.md` §20、`src/chessstate.h` |
| 22 | **本轮**：PPO+MCTS 正确性审计（按外部排查树逐条核对；四路并行审计 + 我逐条复核源码） | 把"多轮没效果 / 越训越臭 / 胆小不吃子"钉到真正的病灶上 | **C13 PUCT 的 Q 符号反了**（选择器在最大化**对手**价值；不报错、评估越准越糟、并直接解释"胆小不吃子"）；**C14 `loadModel` 静默成功**（`RL::PPO::load` 是 void ⇒ 只按"文件可读"返回 true ⇒ 拒载却报成功；**污染了本轮自己的测量**；顺带确认 `weights/` 全部检查点失效）；**C15 `Result`/`Color` 枚举混比**（红胜记成黑胜）；**C16 `BG_TRAIN_SIMS=20` 结构性退化**（模拟数 < 分支数 ⇒ **零深挖**）；**C17 搜索从不判终局叶子**；**C18 无根噪声 + `temp` 是死参数** | 三处 `getPUCT`/`getUCB1` 取负号；内核 `load` 传播真实结果（**不短路**）+ GUI 训练往返改硬失败；新增 `winnerOfResult()` 作唯一换算入口；模拟数 20/80 → **400**；新增 `evaluateLeaf()` 作叶子估值唯一口径；内核加 `Random::gamma/dirichlet` + `acquireRoot(..., withRootNoise)` + 出招抽成 `pickRootChildByVisits()` + 截断局自举 | 手算最小树证明符号（红该选 A、漏负号会选 B）；根节点诊断：**20 次模拟 ⇒ 根孩子数 20.0、深挖余量 0.0**（80→41.3、400→380.0）；20 vs 400 **85% 决策不同**；`test_ppomcts` 新增 3 节、`test_rules` +7 断言。**与 AB 的配对对局：0 胜 4 负、得分率三次都 0.0%，平均手数 24 → 28 → 34** —— 协议压在 0% 地板上，**分辨不出胜负**（一个必须如实说的负面结果；根因是检查点接近随机：critic 留出 MSE 0.0139 → 0.0144 没改善、actor top-1 一致率仅 3.0%） | 零之二点十九、C13–C18 |
| 23 | **本轮**：诊断指标矩阵（棋力是滞后指标，先把"信号健康度"接上） | 让"哪一层设计错了"当场可读 | 上一轮之所以只看到一个 0%，正是因为**缺中间观测量** | 新增 `src/rl/diag.h`（**无 Qt 依赖**的指标核心：熵/CE/KL/EV/CV/校准 + `RootDiag`/`MoveBehavior`/`TrainDiag`/`GameStat` + 流式 `Aggregates` + `CsvWriter`）；`PPOMCTSAgent::rootDiag()` / `printSortedRoot()` / `evalRootNoise`；`bench_diag`（**自动战术题库**：随机造局面 + 扫描合法着法自校验"一步杀/白吃子"，**不需要手工摆局面**）+ 逐手宽表 CSV + TensorBoard 长表；`test_diag`（**68 断言**，进 ctest，测试数 12 → 13） | **上线当天抓出 C17**：一步杀 **0/20 → 20/20**、战术总准确率 10% → **50%**，而**自对弈读数一点没变**（外科式修复的回归证据）；诊断结论：KL(访问‖先验)=**1.32**（搜索不是白跑）但 **EV = −0.0293 < 0、283 样本里 281 个落在同一个校准桶** ⇒ V 是常数偏置 ⇒ **下一轮杠杆在 value，不在搜索参数**；根噪声实测**救不了**（开/关吃子访问份额差 **−0.0037**）；并记下三条"会让诊断说谎"的坑（agent 绑死 `Chess&` 会静默量错局面、无随机开局让"开局多样性"量到自己、全和棋时 `EV=0` 是无定义不是"V 很差"） | 零之二点二十、§20.3~§20.6 |

### 18.2 方法论沉淀（可复用，按"下次还会用到"排序）

1. **先造尺子，再用尺子下结论**。所有"改动有没有用"的结论都必须先回答"这把尺子能分辨
   多大的差异"（功效检验）。反例：20 局等时间对局（±5%）回答不了 B-5/R2 这种 1–2% 的效应。
2. **配对比较 > 两次独立运行**。"改动前后各跑一次"会被机器状态漂移吃掉（本轮实测同一份
   代码单次微基准抖动有 2 倍）。同进程、同权重、同局面、只换一个开关才是有效对照
   （`sparsePolicyHead`、`treeReuse`、`maskedTrainHead`、`--ab`/`--reuse-ab` 都是这个用途）。
3. **区分"机制成立"与"决策改变"**。B-5 与 c_puct 两条中性结论都是同一个机制：搜索先把
   ~44 个孩子铺满，深挖只剩几次 ⇒ 继承的统计/探索项都动不了 argmax。**只报"机制数字"是
   自欺**，必须把"选点是否变"单独量出来。
4. **归因前先排除改动之外的原因**。`test_reward_diag` 默认参数下的"假回归"，用另一条开关
   跑出**逐位相同**的数字才排除掉（"看起来相关"不等于相关）。
5. **数值检验要知道自己的噪声地板**。float32 中心差分：损失量化噪声 ~1e-8、除以 `2eps` 后
   地板 ~5e-6；梯度只有 1e-5 量级时"相对误差 3%"是噪声。判据要写成"相对 < X **或** 绝对
   到地板"，并**特意挑信号最强的抽样点**。
6. **契约要用断言写进代码**（`#ifndef NDEBUG`，Release 零开销）。`MM` 形状断言第一次运行就
   抓到 `lstm.cpp` 一处"碰巧算对、多列就崩"的越约调用 —— 这类问题读代码看不出来。
7. **归一化口径必须显式**。R1 的"先验放大 1/Z（Z≈0.55）"、R2 的"Z ≡ 1"、B-5 的"策略目标
   计数跨 ply 累积"，都是同一类问题：**口径一变，下游所有比较都失去意义**（R2 之后
   `PPO::action()` / `--ab=1` 在 R2 权重上直接失效）。2026-09 又添一条同族：
   `clipGrad=true` 会把每个张量的梯度按范数归一，于是"梯度整体乘常数"**什么也不改** ——
   所以"把同一批数据重复过 N 遍"（权重不变的批内重复）对更新方向毫无影响，
   只有"每遍重新抽新样本"才是真的把批放大 N 倍（`agents_design.md` §18.2）。
8. **每一处"等价性主张"都要有对应的断言**。R1 的逐元素 3.7e-09、R1.5 的"列向量下
   `kijk`≡`kikj`（后者写不出来 —— 因为契约本来就不成立，于是改成 `kikj`）"、R2 的
   "非法行梯度恰为 0"，都是把"我以为"变成"测出来"。
9. **负面结果一样要写进文档**（胜率没动、B-5 中性、c_puct 扫不出落点、旧估算 6× 实际 1.55×）。
   它们决定了下一轮该往哪儿走，价值不比正面结果低。
10. **环境坑**（本轮踩到的，与算法无关但要记）：
    * **ninja 在本沙箱里一 spawn 子进程就挂死**（子进程输出管道被挡），只能用
      `ninja -t commands` 取命令行再自己跑 `cl.exe`/`lib.exe`/`link.exe`；
    * **cmd 的 `>` 重定向在本沙箱里会写出垃圾文件**（4096 B 未初始化块）；
    * **`$env:TEMP` 每次调用都不同**（沙箱按调用给子目录），脚本里一律用工作区绝对路径；
    * **无 BOM 的 `.ps1` 会被 Windows PowerShell 按 ANSI 解码**（中文注释直接解析失败），
      要么纯 ASCII 要么带 BOM；
    * **用 shell 文本 cmdlet 回写 UTF-8 源文件会毁掉中文注释**（`Get-Content` 默认按 ANSI
      解码）—— 本轮在 `test_reward_diag_main.cpp` 上复现了一次，`git checkout` 还原后用
      文件工具重做；
    * **两个 agent 连着构造 = 从同一条随机流里取两份不同权重**（配对实验里"同权重"必须
      显式保证：各自播种或 `copyTo`）。2026-09 又踩到同一坑的**第二个网络**：`RL::SAC`
      的策略头梯度依赖 critic，所以配对时只 `actor.copyTo` 是不够的（量出 2.59 而不是
      2.00，见 `agents_design.md` §18.5 第 1 条）；
    * **`ninja -t commands` 取命令行 + 直接跑 `cl.exe`** 在 2026-09 的沙箱里已经不必需：
      `cmake --build <build> --target <t>` 正常（`.r1build/mk.bat`）。
11. **"看起来成功"比"报错"更贵 —— 任何前后对比先确认前提成立。**（2026-09 本轮最贵的一条）
    `loadModel` 在拒载时返回 `true`，于是 `bench_ppo_sims` 打印"A 成功 / B 成功"、
    stderr 同时在喊"拒绝载入"。我据此做的 A/B 两次都测出 2.5% 并一度写成
    "这个修复没有可测量效果" —— 真相是**两次都载入失败、跑的都是随机权重**，
    结论完全无效。**规矩：跑配对实验前，先看载入/初始化那几行是不是真的成功了；
    报告里"权重"那一行必须出现在结论旁边。**
    同族：`evalRootNoise` / `treeReuse` / `sparsePolicyHead` 这类开关，也要先确认
    "打开它确实改变了输出"（`test_diag` 就用"同种子、同权重、只翻开关 ⇒ 访问分布必须不同"
    来钉住这条，否则一个没接上的开关会让你得出"这个机制无效"的错误结论）。
12. **诊断指标本身要有"自证"能力。** 三例：(a) 全和棋时 `EV = 0` 是**无定义**而不是
    "V 很差"，必须显式区分（否则读数一定被误读）；(b) "每手 V 增益"在 V 近似常数时
    等于 `−2·mean(V)`，必须同时给 `std(V)` 与 `corr(V_before, V_after)` 才不会被当成
    "agent 越走越差"；(c) 模拟数 < 分支数时**根访问分布被强制成均匀**，任何形如
    `Σ N×(下标)` 的哈希都会退化成常数（它只跟孩子个数有关）—— 这条顺带把"地板"的结论
    从"信息量低"锐化成"**信息量为零**"。**判据与阈值要写进仪表盘输出**，光有数字没有
    判读等于没接仪表盘。
13. **"量错了"与"模型差"长得一模一样。** 本轮三次把**工具/测试的缺陷**读成了模型的性质：
    `PPOMCTSAgent` 构造时绑死 `Chess&`（另建 `Chess probe = pos` 再 `selectMove` 只会搜
    `board` 上的别的局面，且**安静地给出 0%**）；自对弈没随机开局 + `temp=0`
    （"开局多样性 1/4"量的是自己）；测试用绝对阈值判"两个网络是否不同"
    （8100 维 softmax 上绝对差只有 1e-5 量级）。修掉后三个数字分别是 2.5% → 16.7%、
    1/4 → 6/6、以及稳定通过。**凡是"这个数字太差/太好"的地方，先问一遍"工具是不是量错了"**，
    并且**给每个新指标配一个反向对照**（构造一个已知答案的输入，看它是否报出预期值）。

---

## 零之二点十九、PPO+MCTS 正确性审计与六项修复（2026-09）

> 触发点：外部给出了一份 PPO+MCTS+AlphaZero 的排查树（"多轮没效果"通常不是单一原因，
> 而是"搜索没产出好目标 / 网络没学到目标 / 自对弈数据自循环退化"三件事至少坏了一件），
> 要求按中国象棋场景逐条核对。
>
> 做法：**四路并行审计**（MCTS 层 / 训练目标层 / 自对弈数据层 / 棋规与动作空间层），
> 每路都要求给出 `file:line` 证据并**在源码里核实，不信文档结论**。审计报告随后由我
> 逐条复核 —— 这个过程本身有价值：审计给的一条判断（`Result`/`Color` 混比的范围）被
> 复核修正了（C15 末尾）。

### 19.1 结论先行：真正的病灶是符号，不是超参

排查树里"损失降但棋力不涨 / 越训越臭 / 胆小不吃子"这三条症状，在这份代码里指向同一个
缺陷：**`getPUCT` 的 Q 项符号反了**（C13）。它有一个很讨厌的性质 ——
**不报错、而且评估越准错得越狠**：随机权重时 Q≈0、探索项盖住一切，所以 agent 看起来
"能下"；一旦 value 头开始有区分度，搜索就开始认真地挑对自己最差的着法。

它同时解释了外部第二条反馈里的"**胆小不吃子**"：吃子后轮到对手、对手少一个大子 ⇒
该子节点（对手视角）的 Q 为负 ⇒ 而在根上直接相加取最大，于是吃子被算成**亏着**，
而"平推不交换"的子节点 Q≈0 反而更大。**搜索于是系统性回避吃子**，与"看太远/看不够远"
无关。

### 19.2 六项修复与它们的证据

| # | 问题 | 修法 | 验证 |
|---|---|---|---|
| C13 | PUCT 的 Q 符号反了（最大化对手价值） | 三处取负号（PPOMCTS / SACAZ / 纯 MCTS） | `test_ppomcts` 新增 "PUCT sign"：手搭两孩子树、先验与访问数**完全相同**（U 项相等 ⇒ argmax 只可能由 Q 符号决定），两个方向都查 |
| C14 | `loadModel` 静默成功 | 内核 `load` 传播真实结果（不短路）；GUI 训练往返改硬失败 | `test_ppomcts` "loadModel reports the REAL result"：往返逐元素差 **0.000e+00**、结构不匹配必须 false、不存在路径必须 false；反向对照要求两个独立初始化的输出**相对差 > 10%**（绝对阈值会误判，见 19.4） |
| C15 | `Result`/`Color` 枚举混比 | 新增 `winnerOfResult()` 作为唯一换算入口 | `test_rules` **[11]** 7 条断言（含"两个枚举数值撞号"本身） |
| C16 | `BG_TRAIN_SIMS=20` ⇒ 零深挖 | 训练 20 → 400、对局 80 → 400 | 根节点诊断：20 次模拟下"根孩子数 = 20.0、深挖余量 = 0.0"；20 vs 400 有 **85%** 的决策不同 |
| C17 | 搜索从不判终局叶子 | 新增 `evaluateLeaf()` 作为唯一口径 | `test_diag` **[7]** 自动生成一步杀题；实测 **0/20 → 20/20**（详见 §20.3） |
| C18 | 无根噪声 + `temp` 是死参数 | 内核加 Gamma/Dirichlet；`acquireRoot(..., withRootNoise)`；出招抽成 `pickRootChildByVisits()`；截断局自举 | `test_ppomcts` 两节共 20 余条断言（采样器和的偏差 ≤ 6.68e-08、alpha 语义、评测不开噪声、30 手后自动关、`temp=1` 频率 0.721/0.185/0.094 对期望 0.727/0.182/0.091） |

### 19.3 与 AB 的配对对局：一个必须如实说的负面结果

按"同一检查点、同种子、同开局、交换先后手"跑 `bench_ppo_vs_ab`（AB 深度 4，4 局）：

| 配置 | 比分 | 得分率 | 平均手数 |
|---|---|---|---|
| 旧代码 sims=80 | 0 胜 4 负 | **0.0%** | 24.0 |
| 新代码 sims=80（**仅符号修复**） | 0 胜 4 负 | **0.0%** | 28.0 |
| 新代码 sims=400（符号 + 模拟数） | 0 胜 4 负 | **0.0%** | **34.0** |

**得分率三次都压在 0% 地板上，这个协议分辨不出胜负。** 唯一正向信号是平均存活手数
（24 → 28 → 34，两个改动各自贡献 +4 / +6）。灵敏探针（选点一致率，SE≈1.8%）给的是
**相反**的读数：7.5% → 7.0%（噪声内）→ 3.0%（400 模拟反而更低）。

后一个数字合理解释是"更深的搜索用**还不可用的 Q** 覆盖掉了还不错的先验" —— 这与工程里
已经记过的那条结论一致（`issues_review` P2 第 18 条：**叶子没训准之前，把搜索深度堆上去
是负收益**）。所以 §19.2 的 C16 改的是**训练侧的数据质量**（20 次模拟的目标是纯噪声），
而不是"多搜一定更强"。

**为什么这里的结论这么弱**：检查点本身接近随机。用 `bench_ppo_distill --positions=1500
--depth=4 --actor=1` 现造了一个与当前架构匹配的检查点，实测：

* critic：留出 MSE **0.0139 → 0.0144**（**没有改善**，它基本在预测均值）
* actor：CE 9.00 → 5.63、`P(AB着法)` 0.00012 → 0.0152、**top-1 一致率仅 3.0%**（机会水平 2.5%）

也就是说 0 胜 4 负是**检查点的锅，不是搜索的锅**。这个结论直接决定了下一轮该做什么
（见 §20.4 与 §五）。

### 19.4 方法论教训（本轮新增）

1. **指标算错比没有指标更贵，而"看起来成功"是最坏的一种输出。**
   C14 的现场是：`bench_ppo_sims` 打印"A 成功 / B 成功"、stderr 同时在喊"拒绝载入"。
   更值得记的是它**污染了本轮自己的测量** —— 那次 A/B 两次都测出 2.5%，我一度写成
   "符号修复没有可测量效果"，真相是两次都载入失败、跑的都是随机权重。
   **教训：任何"前后各跑一次"的对比，先确认两次都真的载入了预期的权重。**
2. **阈值的尺度要跟着量纲走。** `test_ppomcts` 的载入往返测试第一版用绝对阈值
   `1e-4` 判断"两个独立初始化的网络是否不同"，结果判成"相同"而失败。原因是策略头是
   8100 维 softmax，小网络的 logits 接近 0 ⇒ 每个概率只有 ~1.2e-4 量级，绝对差自然只有
   1e-5~1e-4。改成"相对均匀分布的差异"后稳定（实测 55.4%）。
3. **构造对象会消耗全局随机流。** 想验证"同种子 ⇒ 逐位可复现"时，我建了三个 agent
   再各自播种 —— 它们的**权重不同**（构造时各消耗了一段随机数），当然不可复现。
   正确做法是**同一个 agent 跑两次**。这正是 `bench_ppo_sims` 头注释里记过的同一个坑。
4. **审计结论也要复核范围。** 外部审计说"胜率记账把红胜记成黑胜"，方向是对的，但它把
   范围说大了（把使用局部变量 `winner` 的那几处也算进去了，而那几处是正确的）。
   逐条回到源码核对，才把结论收敛到真正的两处。

---

## 零之二点二十、诊断指标矩阵（2026-09）

> 动机：**棋力是滞后指标**。本轮实测就是例子 —— 4 局对局的得分率三次都压在 0% 地板上，
> 什么也答不了。能当场回答"哪一层设计错了"的是中间量：老师（搜索）产出的 π/Q 干不干净、
> 学生（网络）拟合得快不快、行为上敢不敢做正确交换、自对弈生态有没有在给压力。

### 20.1 落地结构

| 件 | 内容 |
|---|---|
| `src/rl/diag.h`（新） | **无 Qt 依赖**的指标核心：纯数学（`entropy` / `crossEntropy` / `klDivergence` / `variance` / `explainedVariance` / `coefficientOfVariation` / `calibration` / `qForParent`）+ 四个诊断结构（`RootDiag` / `MoveBehavior` / `TrainDiag` / `GameStat`）+ 流式 `Aggregates`（**只累加、不存序列**，百万手不涨内存）+ `CsvWriter`。每个公式都做了"分母为 0 / 全零分布 / 非有限值"的退化处理，**绝不产生 NaN**（NaN 进了曲线就是"曲线断了"，而人不会去查） |
| `PPOMCTSAgent::rootDiag()` | 从根节点整理 `RootDiag`：访问熵 / top-1 份额 / 先验熵 / **KL(访问‖先验)** / 展开覆盖率 / **吃子 vs 退让的 Q 分组**。这是唯一能回答"搜索是不是怕吃子"的地方 |
| `PPOMCTSAgent::printSortedRoot(k)` | 调试钩子：把根的已展开着法按访问数排序打印（P/Q/N + 是否吃子），人工核对"吃子着排第几" |
| `test/test_diag_main.cpp`（新） | **68 条断言**，已进 ctest（**Test #4**）。含解析验证（手算值钉公式）、`qForParent` 符号、CSV 列数与表头一致、端到端接线不变量（**根访问数之和 == 模拟次数**）、以及"搜索能不能看见一步杀" |
| `test/bench_diag_main.cpp`（新） | 自对弈逐手诊断 + **自动战术题库** + Dirichlet 有效性 A/B + value 校准 + 排序根打印 + CSV/TB 输出。**不进 ctest**（跑真实对局、依赖随机开局） |
| `CMakeLists.txt` | 注册两个新目标（`test_diag` / `bench_diag`），ctest 从 12 个测试涨到 13 个 |

**输出两种 CSV**：逐手宽表（22 列，pandas/Excel 直接看分布）+ TensorBoard 友好的
`tag,step,value` 长表（与 `add_scalar(tag, value, step)` 语义一一对应）。
**刻意不写 protobuf event 文件** —— 那要引入 protobuf 依赖，与本工程"只依赖 Widgets+Sql"
的取舍冲突（同 `metricsview.h` 里对 Qt Charts 的处理）。

### 20.2 自动战术题库（不需要手工摆局面）

本工程没有 FEN 接口，手工摆将杀局面很容易摆错，而且**摆错了测试会变成空转且不报错**。
做法是自动生成 + 自校验：随机走若干步得到一个局面，再扫描所有合法着法 ——

* **一步杀**：存在一个着法，落子后 `getResult` 判己方胜（将杀/困毙）
* **白吃子**：存在一个吃子着法，落子后该子**不被对方攻击**（没有反吃）

解的合法性由规则引擎自己保证，标准答案是搜索"找出来"的而不是外部引擎给的。
（"白吃子"用 `isAttacked` 做几何近似，不看反吃着法是否合法，因此对"有根子"这类复杂情形
会漏判 —— 这一点在代码里写明了，作为"敢不敢吃"的行为指标足够且完全可复现。）

### 20.3 上线当天就抓到一个真缺陷（这就是仪表盘的价值）

第一次跑 `bench_diag` 就报出 **一步杀命中 0/20 = 0%**，而**白吃子有 16.7%**。
这个 0% 与 16.7% 的对比直接定位到 **C17**：搜索从不判终局叶子 —— 一步杀那步落子后叶子
被交给 critic 去猜，而非终局的"白吃子"本来就该靠 V 比较。

修 `evaluateLeaf()` 之后，用**同一台仪器**复测：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 一步杀命中 | **0/20 = 0.0%** | **20/20 = 100.0%** |
| 战术总准确率 | 10.0% | **50.0%** |
| Q(吃子) − Q(退让) | −0.0998 | −0.0998（**未变**） |
| 吃子访问份额 | 0.136 | 0.136（**未变**） |

自对弈那几行**一点没动** —— 普通局面里没有即时杀棋，终局处理只在该生效的地方生效。
这个"只有该变的变了"本身就是回归证据，说明修复是**外科式**的。

### 20.4 现在的红线在哪（一组实测读数）

`bench_diag --games=6 --plies=80 --sims=80 --tactics=20 --load=weights/ppo_bc_d4`：

```
-- 搜索(老师) --
  分支数 39.4 | 展开覆盖率 1.000 | top1 访问份额 0.289
  访问熵 2.693 nats | 先验熵 1.821 nats | KL(访问||先验) 1.321   <- 搜索确实在纠正先验, 不是白跑
  吃子访问份额 0.132 | 吃子最优率 0.170
  Q(吃子)max 0.0274 | Q(退让)max 0.1258 | 差 -0.0983            <- 负: 搜索怕吃子
-- 行为 --
  每手材质变化 0.0406 | 选到吃子的比例 0.170 | 叫将率 0.018 | 被将率 0.014
  每手 V 增益 -0.2910 (V 均值 0.1457, 标准差 0.0407, corr -0.090)
[1b] value 校准 (283 样本): EV = -0.0293  <- <0: V 还不如"永远预测均值"
   [+0.00,+0.20) 预测 +0.148 -> 实际 +0.000  (n=281/283)
[3] Dirichlet: 吃子访问份额 关 0.1263 vs 开 0.1225 (差 -0.0037)
-- 自对弈生态 --  红胜 0.000 | 黑胜 0.167 | 和 0.833 | 平均 47.2 手 | 开局种类 6/6
```

**判读（这一段的结论比数字重要）**：

1. **搜索不是白跑**：KL(访问‖先验) = 1.32，说明访问分布确实偏离了网络先验，搜索在提供
   先验之外的信息。这一格排除掉"搜索没产出目标"。
2. **但叶子估值没有区分度**：EV = **−0.0293 < 0**，且 **283 个样本里 281 个落在同一个桶**
   （预测 +0.148、实际 +0.000）—— V 基本是一个"走子方恒为正"的常数偏置。
3. 于是因果链闭合：**V 无区分度 → 叶子上"吃子"和"退让"看起来差不多甚至更低 →
   Q(吃子)−Q(退让) = −0.098 → 白吃子命中率 10%**。
4. **噪声救不了**：开/关根噪声的吃子访问份额差 **−0.0037**，噪声甚至没把它推上去。
   这与外部反馈里的那句话完全一致（"Dirichlet 不能单独治好这个病…噪声白加"），
   现在有了实测数字。仪表盘自己的判读也是同一句：*几乎不变 = 叶子 Q 已把吃子压死，
   该回去修 value 而不是调噪声*。
5. **因此下一轮的杠杆在 value，不在搜索参数**（不要再去扫 `c_puct` / 温度 / 模拟数）。

### 20.5 三条"会让诊断说谎"的坑（比指标本身更值钱）

1. **`PPOMCTSAgent` 在构造时就把 `Chess&` 绑死了。** 我在题库 / 噪声 A/B / 打印根三处都写成
   `Chess probe = pos; ag.selectMove(...)` —— 那只会搜索 `board` 上当时的局面（自对弈结束时
   的残局），而且它**安静地给出 0%**，看起来像"模型很差"。修掉之后白吃子命中率
   2.5% → **16.7%**（即之前那个数字完全是假的）。
2. **自对弈没有随机开局 + `temp=0`** ⇒ 4 局必然走成**完全相同**的一盘棋，
   "开局种类 1/4"量的是我自己的缺陷而不是模型的问题。修掉后 6/6。
3. **全和棋时 `EV = 0` 是"无定义"而不是"V 很差"**（`Var(z) = 0` 时 EV 无解，函数返回 0）。
   那个 0 看起来像"V 和常数预测一样烂"。加了 z 方差护栏后直接打印
   "**这不是 value 的结论，而是自对弈生态的结论**"。
4. （测试里踩到的同类）**模拟数小于分支数时，根访问分布被强制成均匀**，于是任何形如
   `Σ N×(下标+1)` 的哈希都退化成常数（它只跟孩子**个数**有关）。这条顺带把 C16 的结论又
   锐化了一层：20 次模拟对 38.7 个分支时，搜索的信息量是**零**，不只是"低"。

### 20.6 尚未接线的部分（明确列出，不含糊）

| 项 | 状态 | 说明 |
|---|---|---|
| `TrainDiag` 的实时采集（policy CE / 策略熵 / MoE 负载 CV / clip fraction） | **部分已接（零之二点二十一）** | `train_ppo` 在训练路径上产出逐局生态（和棋原因/手数/吃子/开局多样性）+ 吞吐 + value MSE + MoE 专家负载，并写 TB 长表 CSV；**仍缺** policy CE / 策略熵 / EV（`Aggregates::addTrain` 要在 `commitEpisode` 里接一行）。`clipFraction` 对本管线不适用（见本表末行） |
| 快照 Elo 阶梯 / forgetting curve / buffer age 直方图 | **部分已做（零之二点二十一）** | `bench_anchor` 给出固定开局集（带 FNV 指纹）+ 换先手成对计分 + 得分率/Elo 的 95% 区间 + `--budget` 等时间 —— "棋力有没有涨"第一次可证伪。**仍缺**常驻快照池：现在每次 `bench_*` 仍是新进程，锚点要手动传 |
| Blunder delta | 未做 | 需要逐局面 AB 标注；AB agent 就在手边，接线成本不高 |
| SQLite 表 | 未做（**选择 CSV**） | 给了逐手宽表 + TB 长表两种 CSV；GUI 侧 `GameDatabase` 加诊断表没做 —— 这是刻意的取舍，先记在这里 |
| virtual loss contention | N/A | 各 worker 拥有**独立**搜索树，不存在多线程踩同一棵树（见"零之二点十三"） |
| 长将/长捉裁定 | 未做 | 规则层问题：重复局面目前**只判和**，WXF 下长将方应判负（C 组既有条目，仍开放） |
| `clip fraction / ratio` | **对本管线不适用** | actor 损失是纯交叉熵（AlphaZero 蒸馏口径），没有 importance ratio。`TrainDiag::clipFraction` 显式标成 **−1** 而不是填 0，免得看曲线的人以为"clip 永远 0 = 更新停滞" |

---

## 零之二点二十一、P0 基础设施：和棋归因 / 常驻训练器 / 锚点评测（2026-09）

> 背景：外部提出的三轮建议（过强对手打崩信心、和棋率、更新器选型与回放容量）先被逐条
> 对着代码核对了一遍，收敛成 [`rl_plan_optimized.md`](rl_plan_optimized.md)（含"不做清单"），
> 然后**只做 P0**：不动算法，先把"能量出来 + 跑得动"建起来。本节记事实与证据。

### 21.1 三条先被定性的判断（对着代码核对，不是推理）

1. **"被过强对手打崩信心"在当前仓库里没有通道。** PPO+MCTS 的学习入口只有
   `chessboard.cpp:1996` 的 `clone.trainSelfPlay(...)`；对局/评测路径里没有任何
   `commitEpisode` / `learnFromReplay` / `beginOnline` / `endOnline` 调用（`chessboard.cpp:1149`
   的注释自己写着在线路径"GUI 从不调用"），与 AB 的对弈全在评测路径且明确不写权重
   （`bench_ppo_vs_ab_main.cpp:5-11`）。全仓也无对手池/league（§20.6 一直列着）。
   ⇒ 观察到的"只走一两手、不敢吃子"必须先去查另外三个嫌疑人：**PBRS 的边界双计**、
   **低模拟数下 π 目标本身尖**、**截断自举**（见第 3 条）。
2. **和棋的大头是台架截断，不是规则纵容。** GUI 训练一局上限 `BG_TRAIN_MAX_MOVES = 60`
   （`chessboard.cpp:167`，循环一次迭代 = 1 ply），而"60 回合自然限着"要
   `halfMoveClock >= 120`（`chess.cpp:1056`）⇒ **这条规则在训练路径上不可达**。
   再加上判和只有"三次重复"与"截断"两类，于是"和棋率 70%"里混着性质完全不同的东西。
   长将/长捉**确实没有判罚**（`chess.h` 里"长将/循环走法检测"那行注释名不副实：
   `isRepetition()` 只数重复次数），这是唯一"白嫖和棋"的真通道 —— 仍未做（见 21.4）。
3. **本管线里不存在 PPO clip，且价值目标不是 z。** `RL::PPO::accumulateGrad`（`rl/ppo.cpp:212-264`）
   只有 `CE(π_visit, π_net) + MSE(v, target)`，没有 ratio / clip / advantage —— `TrainDiag::clipFraction = -1`
   的注释就是这件事（§20.6 最后一行）。价值目标走 negamax 折现回报（`rl/ppo.cpp:528-604`）
   再叠 PBRS 平移 `V' = V + Φ`（`stone.h:885`），于是任何"V 校准"读数**必须先减 Φ**，
   否则量到的是手写 `evaluate()`；而每步塑形项最大 ≈(1+γ)|Φ| ≈ **1.99**，压过终局 ±1。

### 21.2 落地件

| 件 | 作用 | 第一次运行报出来的事 |
|---|---|---|
| `Chess::DrawReason`（`chess.h`/`chess.cpp`） | 判和**分原因**：`DRAW_REPEAT` / `DRAW_NO_CAPTURE60`；用**默认参数**重载 `isDraw(DrawReason*)` / `getResult(int, DrawReason*)`，旧调用点零改动。不是和棋时也显式写 `DRAW_NONE`（防"上一局的原因残留到这一局"） | 判例进了 `test_rules`（108 断言），把"三次重复"与"自然限着"钉成两类 |
| `RL::Diag::GameEndKind` + `Aggregates` 分桶（`rl/diag.h`） | 结束方式四分：将杀 / 三次重复 / 自然限着 / **台架截断**；新增 `endMateRate()`/`drawRepeatRate()`/`drawNoCapture60Rate()`/`truncatedRate()`/`truncationShareOfDraws()`/`maxPlies()`，仪表盘直接打印判读 | 60 ply 配置下 `截断 1.000`、"和棋里属于台架截断的占比 1.000" ⇒ **"和棋率高"首先是手数上限问题**。未填 `endKind` 的旧调用方不进任何桶（`gamesClassified()` 会暴露"谁还没接上"） |
| `test/train_ppo_main.cpp`（目标 `train_ppo`） | 无界面**常驻**训练器：一个进程连跑 K 局，每局**零磁盘往返**（GUI 路径是每局 3 趟 x ~555 MB 权重往返，`chessboard.cpp:1915/1991/1999/2055`）。agent 上只加两个默认无害的钩子：`gameLog`（三个终局出口都接上）与 `openingPlies/openingSeed`（起点随机化；顺带修掉"随机开局后 `currentColor` 必须跟着棋盘走"的错帧隐患） | 小骨干 `--hidden=16 --expert=16` 下 **1.3 s/局（≈2800 局/小时）**，"一次 A/B 要跑很多局面"第一次变得可行 |
| `test/bench_anchor_main.cpp`（目标 `bench_anchor`） | 固定开局集（局部 `mt19937_64` + FNV 指纹，**不碰 `RL::Random`** ⇒ 换权重跑的是同一批局面）+ 换先手成对计分 + 得分率/Elo 的 **95% 区间** + `--budget=MS` 等时间标定 + 和棋构成 | 6 局时 Elo 区间 **[−174.9, +46.0] 跨过 50%**，程序自己打印"没测出差别"；旧的 4~6 局随机开局 bench 只会给一个看起来像结论的数字 |
| P1.1 的 A/B 开关（`train_ppo`） | `--no-shaping` / `--shaping-alpha=F` / `--no-material-reward` / `--no-bootstrap` / `--no-root-noise` + `--hidden/--expert` | 配置行确认 `塑形: 关 (alpha=0.25)` —— 第 1 条里"PBRS 边界双计"这个嫌疑人现在**可以量**了 |
| GUI 对局上限 | `mainwindow.cpp`：`gamesSpin` 100 → **10000**（步进 10） | `matchAgents` 只有下界钳制（`chessboard.cpp:1625`），放开上限只需改界面；80 Elo ≈ 61.5% 得分率，几十局的协议分辨不了 |

### 21.3 验证

* `test_rules`：**108 断言 / 0 失败**（新增 7 条：三次重复→`DRAW_REPEAT`、120 半回合→`DRAW_NO_CAPTURE60`、
  分胜负与未判和时原因被清成 `DRAW_NONE`）。
* `test_diag`：**68 断言 / 0 失败**（改过 `diag.h`，CSV 表头一致性那条断言仍绿）。两条都过 `ctest`。
* **全项目重建**（30+ 目标，含 `chess.exe`）退出码 0 —— `chess.h`/`diag.h` 是横切头文件，这一条是必需的。
* `train_ppo --games=2 --sims=12 --moves=60 --opening=6`：跑通并产出 `game_end_*` 逐局行 + `sum_*` 汇总行；
  `bench_anchor --openings=3 --plies=4 --sims=8 --ab-depth=2`：跑通并产出区间与和棋构成。
* 新代码 **0 编译警告**（仅剩 `rl/util.hpp:65`、`rl/tensor.hpp:558` 两条既有警告）。

### 21.4 仍未做

* **长将判负**（唯一规则改动）：必须与**新增状态平面**同批做（"是否被将军"这项特征告诉不了网络
  *是谁*在连续将军，而判负对象正是那一方），因此 `STATE_DIM` 1710 → 1800、**所有已存权重作废**。
* **P1.2** critic 的独立稠密监督（AB 根分值 `tanh(score/2)` 当辅助损失，`bench_ppo_distill` 已证明只训 critic 可行）；
  **P1.3** PCR + 目标不确定性加权（只让深挖充分的 ply 进 policy 目标）；**P1.5** 残局课程 + conversion rate
  （`trainSelfPlay` 的起点参数已由 `openingPlies` 打开一半，还缺"从给定局面起手"的入口）。
* **`TrainDiag` 的接线**：`train_ppo` 报告里"根/行为/训练"三行仍是 0（那是 `bench_diag` 的职责），
  但 `Aggregates::addTrain` 在训练路径上确实还没接。
* **快照池**：`bench_anchor` 只是"可证伪的那一半"，锚点还要手动 `--a/--b` 传；要成"阶梯/遗忘曲线"
  还得按固定命名存一排快照并脚本化。

---

## 零之二点二十二、DQN+MCTS 的表示闸门 + 自检进界面（2026-09）

**起因**：DQN+MCTS 报出"50 胜 0 负 50 和"、训练损失 22，而**这两个数字都无法说明模型学没学会
下棋** —— 前者里赢家和输家是同一份权重，后者的量纲（原始 Q 尺度的平方 TD 误差）与 PPO 的
`[−1,1]` 值域 MSE（0.003）根本不可比。真正要问的是：**这个 agent 值不值得继续训**。

### 1. 新探针 `probe_dqnmcts_aliasing`（秒级，不进 ctest）

量的是三类**与训练量无关**的结构事实：

| 段 | 事实 | 实测（本机 Release / MSVC 2022 / Qt 6.8.0） |
|---|---|---|
| [1] | **动作别名**：同一个局面内的合法着法 → 128 个 Q 槽位 | 标准开局 44 个着法只落在 **38** 个槽位上（挤掉 6 个，最挤槽位背 3 个着法）；中局 96 个局面平均挤掉 **5.16** 个，最挤槽位 **4** 个。**跨局面累计碰撞率仅 0.03** |
| [2] | **状态不可分性** | 走子方 / 重复 1 次 vs 3 次 / 无吃子 0 手 vs 120 手 —— `encodeState` 输出**逐字节相同**。90 维 = 一格一个子力值 ⇒ 规则上下文通道 **0 个** |
| [2d] | **终局检测口径** | 已判和的局面上 `isGameOver()=COLOR_NONE` vs `getResult()=RESULT_DRAW` ⇒ 判和与将杀**都不是终局**，`evaluateLeaf` 继续走 `max_a Q(s,a)` |
| [3] | **终局奖励通道** | 6 局 × 200 手：`isGameOver()` 看见 **0** 次终局、`getResult()` 看见 1 次、**5 局撞手数上限** ⇒ 终局 ±1 从没进过 Q 目标 |
| [4] | **回放池里的"新信息"** | 1440 个 ply → **1426** 个互不相同的局面（去重率 **99.0%**）；`Transition` ≈1.23 KB/条 ⇒ 4096 条 ≈5.0 MB、20000 条 ≈24.6 MB |

**两条自我否证（记下来，免得后人重复）：**

* **我原先猜"128 个动作槽位是灾难（真走法空间 8100，63 倍别名）"—— 被自己的探针否掉了。**
  跨局面碰撞率只有 0.03，槽位数本身够用；真正有代价的只是"同一局面内平均挤掉 5.16 个着法"。
* **我原先猜"回放池 4096 条里大量是重复局面，所以加大容量没用"—— 也被否掉了。** 去重率
  99.0%，容量装的确实是新信息。

### 2. 那为什么"回放 4096 → 20000"仍然不会更快

去重率 99% 只否掉了"多存重复"这一条理由，**需求侧仍然用不满**：

| 量 | 数值 | 依据 |
|---|---|---|
| 一局写入 | 60 条 | `BG_TRAIN_MAX_MOVES = 60` |
| 一局触发学习 | 每 4 手一次 → **15 次** | `learnCounter % 4` |
| 一局抽走 | 15 × 32 = **480 条** | `batchSize = 32` |
| 4096 条换一遍血 | **68 局 ≈ 9 小时** | `rl/dqn.cpp:213` 只丢最老的 32 条 |

⇒ **前 68 局里，容量这个参数对训练轨迹的影响严格为零**（池子还没满，淘汰根本没发生）。
而改成 20000 会把换血从 68 局拉到 **334 局**，即"最老样本与当前策略的偏离"拉长约 5 倍
（目标网硬拷贝间隔写死 256）。存储也不是理由（5 MB → 24.6 MB）。

**真正的杠杆**是 `docs/agents_design.md` §10.4 那条**已经写下但没做**的建议：

> DQN 系当前"每步一次更新"破坏回放缓冲的 i.i.d. 假设，建议改成**只写回放、攒批更新**

即"把 4096 改成 20000"是在调**下游**参数，而**上游**（每 4 手就更新一次、破坏回放假设）还没动。

### 3. 顺手修掉的两个真问题

* **`trainVsRandom` 的奖励标签是"每走一步 = 输一盘"**：它没有"非终局就用即时奖励"那一支，
  而 `terminalReward = (gameResult == COLOR_BLACK) ? 1.0f : -1.0f` 在非黑胜时恒为 **−1.0f**，
  于是每一手都按 done 写入 −1。**这是"损失下不去、最低 22"的直接原因**，比任何超参都重要。
* **三处收尾的终局口径不统一**：`trainVsRandom` / `trainSelfPlay` / `recordExperience` 都用
  `isGameOver()`（只认"将/帅还在不在场"）⇒ 将杀、困毙、三次重复、60 回合自然限着都不算终局。
  现统一走 `getResult()` + `outcomeForMover()`（与 Phase 6 的"终局口径统一"一致），
  统计改走 `winnerOfResult()`。
  **仍未统一的一处**：`evaluateLeaf`（搜索叶子）—— 它与"Q 是哪个帧"那个设计决定绑在一起，
  见待办 21。

### 4. 自检进界面 + `test_dqnmcts` 的第一批断言

* `AgentBase::selfCheckReport()`（默认空串；**只读、可重复调用、不动棋盘**的契约写在头文件里，
  因为面板会在对局中途被 GUI 线程调用）→ `ChessBoard::getAgentSelfCheck()`（按当前
  `m_agentType` 取对应实例；不用"遍历取第一个非空"——那会在切 agent 后继续显示上一个的结果）
  → 右侧"模型自检"面板（`QPlainTextEdit`），刷新时机 = 选中 agent / 每手预训练后 / 启动加载完成。
* `test_dqnmcts` 原来**一条断言都没有**（只打印），现在有 **16 条**：报告非空且含关键字段、
  **只读**（调用前后棋盘哈希 / 走子方 / halfMoveClock / 历史栈都不变）、可重复（两次调用逐字相同）、
  终局通道计数**一局恰好一条**。main() 按失败数返回退出码。
* **这批断言当场抓到一个真 bug**：自检计数第一版写在"每走一手"之后，于是 12 手的一局被记成
  **12 局**（断言报 `12 != 1`）。面板上的数字是要给人当决策依据的，所以它自己必须被钉住 ——
  这正是"每条读数都要有断言"的理由。

### 5. 复现

```bat
cmake --build <build> --target probe_dqnmcts_aliasing test_dqnmcts
<build>\probe_dqnmcts_aliasing.exe --positions=48 --plies=24   :: 秒级
<build>\test_dqnmcts.exe                                       :: 339.7 s, 16 断言, 退出码 0
```

界面侧：启动 `chess.exe` → 选 "DQN+MCTS" → 右侧"模型自检"面板；或跑一手（走子前探索）再回来看
"对局累计"两行。

---

## 零之二点二十三、用户报障的两条错误 + 九个 agent 全部自检（2026-09）

**起因**：用户贴了两行运行时输出，要求修掉它们，另外"给所有模型都配上自检方法"：

```
[arena] agent(0) 返回无效走法 (第 6 局第 55 手): valid=0 id=0 pos=(0,0)->(0,0), 仍有 1 个合法走法, 已兜底
[train] 种子权重写入失败, 跳过本轮训练: agent "EVAB" 路径 weights/_temp_train.dat
```

两条**都不是"偶发"**，而是各自的支路从来没有被走到过。修的过程中又顺手抓到第三个同类的
静默失效（PPO+MCTS 的权重从没被载入）。

### 1. `[arena] agent(0) 返回无效走法` —— 根因在 `ABAgent::findBestMove` 的根节点

`agent(0)` 就是 **Alpha-Beta**（枚举顺序），而 `valid=0 id=0 pos=(0,0)` 是**默认构造**的
`Step`。造成它的条件非常具体：

| | 根节点窗口初值 | 更新条件 | "每一步都必输"时 |
|---|---|---|---|
| 黑方（MAX 节点） | `beta = -Stone::value_infi` | `r > beta` | 每个 `r` 都是 `-value_infi` ⇒ 一次都不成立 |
| 红方（MIN 节点） | `alpha = +Stone::value_infi` | `r < alpha` | 每个 `r` 都是 `+value_infi` ⇒ 一次都不成立 |

于是 `best` 保持 `nullptr`，函数返回默认 `Step`（`valid=false`），调用方把它读成
**"这一步真无棋可走"** ⇒ 明明还有合法走法却被判负（界面上就是"棋子没动，我却输了"）。
第 55 手出现，正是因为那时一方已经进入"怎么走都输"的残局。

* **修**：`best == nullptr || r > beta`（黑）/ `best == nullptr || r < alpha`（红）——
  第一个走法无条件成为候选；"全负"时退化成"返回 MVV-LVA 排序后的第一手"（棋理上无差别，
  但契约上必须返回一步合法走法）。
* **同类形状还有四处**：`MCTS` / `DQN+MCTS` / `PPO+MCTS` / `SAC+AZ` 在"根有合法走法、但
  一个孩子都没展开"（`iterations`/`simulations <= 0`，或循环里没走到 EXPANSION）时同样
  返回 `Step()`。全部补上"取根节点未展开列表的第一手"兜底。
* **再加一道统一闸门**：`ChessBoard::legalStepOrFallback()`（放在 `aiThink` /
  `aiThinkForAgent` 的出口，大 switch 因此改名为 `aiThinkRaw` / `aiThinkForAgentRaw`）——
  无效走法 + 棋盘上还有合法走法 ⇒ 用第一个合法走法兜底，并打印**带 agent 名字**的
  `[gate]` 日志。原来那条 arena 日志只报 `agent(%d)` 编号，排查时要回去数枚举。
* **回归钉**（`test_match` [2.10]）：构造一个"红方怎么走都输"的局面——
  红 `帅(9,4)` + `车(7,1)`；黑 `将(0,3)` + `车(9,0)` + `车(8,0)`：红被沿第 9 行将着，
  帅的另外三个格子都被控制/占住，唯一的合法走法是 `车(7,1)->(9,1)` 垫将，而垫上之后
  `车x(9,1)` 就是杀。断言三件事：**只有一个合法走法**、**确实必输**（黑方有杀）、
  **`ABAgent(depth=4)` 返回的仍是那一步合法走法**。修之前这条断言会失败（返回 `valid=false`）。

### 2. `[train] 种子权重写入失败 … EVAB` —— 其实是"这一支根本没接"

`backgroundTrainLoop()` 里三个 switch（建实例 / 写种子 / 训 clone / 同步回主 agent）原来只有
`PG`、`DQN`、`PPO+MCTS`、`DQN+MCTS` 四个 `case`。选 EVAB 时 `seeded` **恒为 false**，
于是那条为"写盘失败"写的消息被用来描述"这一支没写"——**排查方向完全错**（用户看到
"写不出去"，实际上是"没人去写"）。

* **修 (a)**：把"接没接"与"写盘成不成功"分成两件事报。新增 `trainable` 判定；未接入的
  agent 明说 `[train] 该 agent 的后台训练尚未接入, 跳过本轮`，不再借用"种子权重写入失败"。
* **修 (b)**：真正把 EVAB 接上——建实例（与 `aiThink` 用同一组构造参数，否则结构对不上、`load` 必失败）
  → `saveModel` 写种子 → clone `loadModel` → `trainSelfPlay(...)` → `saveModel` 写回 →
  主 agent `loadModel` 同步。损失照旧走 `trainLossSample`。
* **修 (c)**：新增 `BG_TRAIN_EVAB_PLAY_DEPTH = 3` / `BG_TRAIN_EVAB_LABEL_DEPTH = 4`。
  理由是一轮 60 手里**每一手要搜两次**（`playDepth` 选步 + `labelDepth` 生成 TD-leaf 标签），
  而实测初始局面 `depth 4 = 90 ms`、`depth 5 = 188 ms`、`depth 6 = 890 ms`——照抄界面上的
  `EVAB_DEPTH = 6` 会让"关窗"要等分钟级（训练线程只在每轮开头看停止标志）。
* **仍未接的**：SAC+AZ / SAC+AZ-MoE / DQN+AB 的后台训练 —— 这一条只成立了半天，
  随后用户就要求补齐，见下面的 **§2b**（现在九个有权重可训的 agent 全部接上，
  再加上新增的 PPO+MCTS-MLP 就是十个里的九个；Alpha-Beta / MCTS 没有权重）。

### 2b. 把剩下三个 agent 也接上（同日追加）

用户随后要求"对弈时将所有模型 agent 接入后台训练"。三个支路（建实例 / 写种子 / 训 clone /
同步回主 agent）全部补齐，于是**八个有权重可训的 agent 一个不落**（AB / MCTS 没有权重）。

| Agent | 一轮的搜索预算 | 一轮的实测量级 | 说明 |
|---|---|---|---|
| PG / DQN | — | 秒级 | 原有 |
| EVAB | `playDepth 3` + `labelDepth 4` | ~15 s | 原有（C20 补的） |
| **SAC+AZ** | `BG_TRAIN_SACAZ_SIMS = 400` | ~1–2 s | MLP 骨干 ~0.05 ms/模拟，可以给足（400 ≫ 分支数 39 ⇒ π 目标是真搜出来的） |
| **SAC+AZ-MoE** | `BG_TRAIN_SACAZ_MOE_SIMS = 64` | ~1–2 min | 一次模拟 ~10.9 ms，给 400 就是小时级；给 16（界面决策预算）则**一次深挖都没有**（C16），64 刚过分支数地板 |
| **DQN+AB** | `nodeBudget = DQNAB_NODES(256)` | ~1–2 min | 主干 37.5 M 参数：临时文件一次往返就是 ~276 MB × 2 写 + 2 读（启动读一次实测 3.8 s） |

**三件必须一起做的事**（少任何一件都会"看起来接上了"）：

1. **多文件家族的临时前缀要分开**。`TMP_WEIGHTS` 是共用前缀，而 PPO 写 `<prefix>_actor`、
   SAC+AZ 也写 `<prefix>_actor` —— 同名。结构指纹会让误读当场失败（不会静默串权重），但
   那是"靠断言兜住的设计"，所以新增 `_temp_train_sacaz` / `_temp_train_sacaz_moe` /
   `_temp_train_dqnab` 三套前缀，由 `tmpWeightsOf(type)` 统一给出。
2. **构造参数必须与 `aiThinkRaw` 那一支逐字一致**（宽度 / 骨干 / c_puct / 专家宽度 / 辅助系数），
   否则主 agent 与训练 clone 的网络结构对不上，`save`/`load` 的结构指纹会让每一轮都载入失败。
3. **一轮的步数不能低于 32** —— 见下面的 C22。

**新增 `setBackgroundTrainRound(episodes, maxMoves)`**：一轮的时长 = 关窗等待时间，而不同 agent
的一轮成本差两三个数量级。把它做成可调的，测试就能在秒级验证"某条支路的往返通不通"，而不必等
一整轮（[2.13] 用 1 局 × 40 手跑完三个 agent）。默认值不变（1 局 × 60 手）。

**"对弈期间继续训练"是明确选择的行为**：后台训练线程不因对弈而暂停 —— 代价是每轮把权重同步进
主 agent，于是**一局之内模型会变**（同一份权重不再能解释完整一局的结果），而且训练与对弈抢
CPU。这一点写在 `backgroundTrainLoop` 的注释里，改主意时只需在对弈开始/结束处暂停/恢复训练线程。

### 3. 顺手抓到的第三个静默失效：PPO+MCTS 的权重从来没被载入过

启动扫描权重的那张表探测的是 `weights/ppomcts_agent.dat`，而 `PPOMCTSAgent::saveModel(prefix)`
写出的实际是 `<prefix>_actor` / `<prefix>_critic` —— 于是磁盘上那两个 **279 MB** 的文件一直
躺在 `weights/` 里没人读（`docs/agents_design.md` §18 的更正里已经记着"RL::PPO 那一支栽在这里"，
但那次只把 DQN+AB 那一行改对了，PPO 那一行留成了同样的形状）。

* **修**：新增 `weightFilesOf(type)` —— "`saveModel(prefix)` 会写出哪些文件"的**唯一来源**，
  启动扫描与自检面板共用；`s_weightPaths` 统一存**前缀**（下游 `load*()` 本来就是按前缀用的，
  SAC+AZ / DQN+AB 那两处 `_actor` / `_trunk` 后缀剥离因此变成无害的 no-op）。
  实测证据：修好之后界面启动日志里才出现 `[weights] PPO+MCTS: 9034 ms`（以前那一整段
  加载代码根本不会被执行）。
* 这正是自检面板第一段要报"权重文件在不在 + 启动扫描有没有命中"的理由：三份名字各自维护时，
  漂移的表现就是**静默地从随机初始化开始跑**。

### 4. 九个 agent 全部自检（`selfCheckReport()`）

面板的派发从"只接了 DQN+MCTS"扩到全部九个（`ChessBoard::getAgentSelfCheck(AgentType)`）。
Alpha-Beta / MCTS **没有常驻实例**（每一步现场构造），所以这两支由 `ChessBoard` 在棋盘**副本**上
现场造一个（唯一需要在自检路径上锁的地方）；其余七个转发各自的常驻实例。

| Agent | 表示层 | 动作层 | 该 agent 特有的读数 |
|---|---|---|---|
| Alpha-Beta | 手工 `evaluate()`（材质 + 位置表） | 搜索给出（无别名） | **决策合法性自检** + 开局合法走法数/评估值尺子 |
| MCTS | 随机走子终局（±1/0，高方差） | 搜索给出 | **决策合法性自检** + `MCTS_SIMS=800` 摊到分支因子的平均次数 |
| Policy Gradient / DQN | 90 维每格一值 ⇒ **规则上下文 0 个** | 128 槽哈希 ⇒ **别名（开局 44→N）** | 探索率/回放池/学习率 + 奖励口径（走子方视角） |
| PPO+MCTS | 1710 = 19 平面（含 3 规则上下文） | 8100 **双射（无别名）** | **根搜索展开覆盖率 / top1 份额 / 访问熵 / KL(访问‖先验)** + MoE 直方图 |
| DQN+MCTS | 90 维 ⇒ 规则上下文 0 个 | 128 槽哈希 ⇒ 别名 + 对局累计 | 终局通道四分桶（旧口径漏了多少局） |
| EVAB | 1530 = 17 平面（含 3 规则上下文） | alpha-beta 直接选（无动作头） | `blend`（手工/网络混合比例）+ 置换表命中率 + 节点/到达深度 |
| SAC+AZ / SAC+AZ-MoE | 1263 = 14×90 + **3 个规则上下文标量** | 128 槽哈希 ⇒ 别名 | 双 critic/软价值口径 + **MoE 路由直方图与坍缩判读** |
| DQN+AB | 1710 = 19 平面（材质/节奏阶段 + 规则上下文） | 8100 **双射** | **值门控**（容差 / gap 前后 / 是否回滚）+ 手工锚 gap/corr |

另外两处工程化：

* 面板顶端统一给出**权重文件状态**（`getAgentWeightStatus()`）：扫描有没有命中、每个文件在不在、多大
  （多文件模型逐个列出，如 SAC+AZ 的 `_actor/_q1/_q2`）。
* 新增按钮 **"全部模型自检"**：把九个 agent 排在一起 —— 编码/口径的差别只有横向对比才看得出来。
* 启动时对每个**已加载**的模型各跑一次自检并打一行 `[selfcheck] <名字>: <表示层摘要>`：
  自检被约定为只读、可重复、不动棋盘，所以这一步很便宜，而"权重没载进来"这类静默失效
  当场就会暴露。

### 5. 复现与实测

```bat
cmake --build <build> --target test_match test_evab test_pretrain
<build>\test_match.exe      :: 109 项断言 / 0 失败 (含 [2.10] / [2.11] / [2.12] / [2.13] 四节新断言)
<build>\ctest.exe           :: 13/13 通过; test_match 现在 ~250 s (原来 ~94 s), ctest 超时已 600 -> 900
```

* **[2.10]**（必输局面）：`红方合法走法 1 个` → `ABAgent(depth=4) 返回: valid=1 id=0
  pos=(7,1)->(9,1)` —— 就是那唯一的合法走法（车垫将）。修之前这里返回的是
  `valid=0 id=0 pos=(0,0)->(0,0)`，断言当场失败。
* **[2.11]**（九个 agent 自检）：`type=0..9` 逐个报告行数/字节数 + **两次调用逐字节相同**；
  能报的 9 个（DQN+AB 在本测试里没有实例 —— 它的实例要 `startupLoad()` 扫到权重才建）。
  顺带钉住"报告里必须写明**不是棋力**"。第一版 MCTS 的自检在这里被判为**不可重复**
  （它那一次模拟要 `std::rand()`），于是拆成"0 次模拟（兜底路径，可报坐标）/ 1 次模拟
  （正常路径，只报合法性）"两条 —— 这条断言当场抓到的第二个 bug。
* **[2.12]**（EVAB 后台训练）：切到 EVAB → `startBackgroundTraining()` → 等 `trainLossSample`
  上报。实测 `EVAB 后台训练上报损失 2 次, 全部有限=1` —— 整条
  "写种子 → clone 载入 → `trainSelfPlay` → 写回 → 同步回主 agent" 走通（`roundApplied`
  为假时根本不会 emit），而报障时这一支**根本不存在**。
* **[2.13]**（三个新接入的支路）：同一判据逐个跑 **SAC+AZ / SAC+AZ-MoE / DQN+AB**，
  用 `setBackgroundTrainRound(1, 40)` 把一轮缩到 40 手。实测三个都是
  `上报损失 1 次, 全部有限=1`。第一版给的是 6 手，三个**全部"上报 0 次"** —— 那不是接线
  坏了，而是它们 `learnBatch` 的"池 < batchSize(32) 就什么都不做"门控把一个 6 手的轮次
  变成了**零梯度更新**（于是也不写 `m_lastLoss`），这正是 C22；断言没有为了通过而放松，
  而是把"轮次必须 > 32"这件事写进注释与测试。
* **界面启动日志**（`QT_QPA_PLATFORM=offscreen` 跑 45 s，只看启动阶段）：

  ```
  [weights] PPO+MCTS: 9034 ms          <- 修好扫描名字之前, 这一行根本不会出现
  [weights] EVAB: 27 ms
  [selfcheck] Policy Gradient: 状态 90 维 (10x9 每格一个子力值) | 动作 128 槽位
  [selfcheck] EVAB: 状态 1530 维 = 17 平面 x 90 格 (14 棋子平面 + 3 规则上下文)
  [selfcheck] PPO+MCTS: 表示: 状态 1710 维 = 19 平面 x 90 格
  [selfcheck] SAC+AZ-MoE: 骨干 稀疏MoE(TB专家) | 界面 agent 类型 SAC+AZ-MoE (AGENT_SACAZ_MOE) ...
  [selfcheck] DQN+AB: 状态 1710 维 = 19 平面 x 90 格: 14 棋子平面(规范视角) + [14]剩余子力 ...
  ```

  八个已加载的模型都在启动时各打了一行表示层摘要；PPO+MCTS 那 9034 ms 正是那两个
  279 MB 的文件（以前它们躺在盘上没人读，所以启动更快 —— 那 9 秒是**修好之后**才出现的）。
  代价要说清楚：**启动从约 17 s 变成约 26 s**（那些权重本来就在盘上，只是从来没被读过）。
  要更快只有两条路：把这组权重删掉/换小骨干，或者明确选择"不加载 PPO+MCTS"
  （`tools/verify_eager_load.ps1` 里那句"启动约 8~10 s"因此也过期了）。

界面侧：启动 `chess.exe` → 右侧"模型自检"面板（顶端是权重文件状态）→ 点"全部模型自检"
看九个 agent 的横向对照；选 EVAB / SAC+AZ / SAC+AZ-MoE / DQN+AB 让后台训练跑一轮，
日志里不再出现"种子权重写入失败"或"尚未接入"。想看某条支路的往返通不通、又不想等一整轮，
在测试里用 `setBackgroundTrainRound(1, 40)`（**别低于 32 手**，见 C22）。

---

## 零之二点二十四、新 agent：PPO+MCTS+AlphaZero（稀疏 MoE + MLP 专家）（2026-09）

**需求**：新增一个"PPO + MCTS + AlphaZero，使用 MLP 专家"的 agent。

**关键决定：不复制任何一份实现。** `rl/ppo.h` 里原本只有一条编译期骨干
（`using PPOExpert = TransformerBlock<16,360>;`，E=4 top-1），而 2026-09 之前用的是
`MlpExpert`（E=8 top-2）。这次把"用哪套专家"变成 **`RL::PPO::Backbone` 这个构造参数**
（默认 = TB，现役行为逐位不变），两种骨干的**唯一差异点**收在一个工厂函数里：

```
rl/ppo.h        PPO::Backbone { TbExperts(默认) | MlpExperts } + PPO_MOE_MLP_EXPERTS/TOPK
rl/ppo.cpp      makeMoeLayer(backbone) —— 只在这里分支 (专家的模板参数不同)
ppomcts_agent   构造参数加一个 Backbone (默认 TB); getName()/自检报告带上骨干
chessboard      AGENT_PPOMCTS_MLP (追加在枚举末尾) + 各 switch 分支 + 独立临时/正式权重前缀
mainwindow      kAgents 多一行 -> 两个骨干能在界面上直接对弈
```

**为什么不复制 `ppo.cpp` 或 `ppomcts_agent.cpp`**：`expert.hpp` 顶部那条教训就是"三份拷贝
迟早漂移"。而且 PPO 的其余部分（损失 / 优化器 / 权重格式 / MoE 诊断 / 多线程分身）**完全
不知道专家是什么类型** —— 它只通过 `ISparseMoE` 接口用它（辅助损失、使用直方图、读写权重
都是虚函数分派），所以两种骨干真的只差那一个工厂。
**为什么不把专家改成运行时多态**：`SparseMoE<Expert, E, K>` 的三个参数都是模板参数，
虚化它要动整个 `RL::Net` 的层体系，代价远大于收益。

### 实测对照（`test_match` [2.14] 打印的真值）

| 骨干 | 专家/topK | actor 参数量 | 界面预算 | 每手实测 |
|---|---|---|---|---|
| TB（现役，默认） | 4 / 1 | **52,388,888** | 400 次模拟 | ~3.2 s |
| MLP（新 agent） | 8 / 2 | **2,448,204**（~21× 小） | 1600 次模拟 | **0.23 ~ 0.39 s** |

同一份"每手 400 次模拟"的预算下 MLP 骨干只花 ~80 ms，所以 `PPO_MLP_SIMS` 给到 **1600**
（"模拟次数 ≫ 分支数 39"是 π 目标有没有信息量的分水岭，见 C16）—— 结果是**参数少 21 倍、
模拟多 4 倍、每手仍快 8 倍**。这是"便宜换更准的访问分布"，不是省时间。

### 四条必须钉住的性质（少任何一条就会退化成"两个会漂移的实现"或"静默共用权重"）

全部在 `test_match` **[2.14]**（6 条断言 + 3 条结构断言）：

1. **结构确实不同**：4/1 vs 8/2；参数量 52.4 M vs 2.45 M；两种骨干的 `actionMasked`
   都给出归一化的概率（和 = 1）。
2. **agent 名不同**（`PPO+MCTS (AlphaZero)` vs `PPO+MCTS (AlphaZero, MLP专家)`）——
   损失曲线按名字分线，同名会把两条线并成一条。
3. **权重文件前缀不同**（`weights/ppomcts_agent.dat` vs `weights/ppomcts_mlp_agent.dat`）；
   后台训练的临时前缀也不同（多文件家族共用 `_temp_train.dat` 时 `_actor` 会**同名**）。
4. **交叉载入必须失败**：实测把 MLP 的权重载入 TB 骨干，内核打印
   `参数量不匹配 (文件 2448204 个元素, 当前网络 52388888 个) … 拒绝载入` 且 `PPO::load`
   返回失败；同一份权重载回 MLP 骨干则成功 —— 这是"权重格式的结构指纹"在跨骨干场景下
   的验收读数。

### 顺带确认（现役配置没有被这次改动碰到）

* TB 骨干的 actor 参数量 **52,388,888** 与 C14 里记的旧值**逐位相同**；
* 界面启动日志里 `[weights] PPO+MCTS: 8831 ms` 照旧（那 279 MB × 2 的旧检查点仍然能载入，
  没有出现"参数量不匹配"）；
* `test_ppomcts` / 全部 13 个 ctest 通过。

### 复现

```bat
cmake --build <build> --target test_match chess
<build>\test_match.exe     :: [2.14] 新 agent 的结构/命名/权重隔离 + 一小局实测耗时
```

界面侧：`chess.exe` → 任一 agent 下拉框里多出 "PPO+MCTS (AlphaZero, 稀疏MoE+MLP专家)"；
选它走一手（懒创建），退出时它的权重会落到 `weights/ppomcts_mlp_agent.dat_actor/_critic`，
之后启动日志里会多一行 `[selfcheck] PPO+MCTS-MLP: 骨干: 稀疏MoE(MLP专家) | 专家 8 个, topK=2 …`。

### 下拉框：加进去 ≠ 看得见（C23）

第一版把它**追加在列表末尾**（第 11 项），结果打开下拉框**看不到它** —— Qt 的
`QComboBox::maxVisibleItems` 默认 10，第 11 行落在滚动区里。UIA 实测（展开后逐行读）：

```
before: agent rows = 10   (PPO+MCTS ... DQN+AB, 新的那一项不可见)
after : agent rows = 11   (missing = 0)
```

两处一起改：`maxVisibleItems` 放宽到列表长度 + 2；新 agent 移到 `PPO+MCTS (AlphaZero)`
**后面**（两个骨干挨着，对照实验一眼能选中）。同时把 `fillAgentCombo` 的默认项从
**写死的下标**（`matchB` 原来是 `6`）改成**按 agent 类型查找** —— 否则插入一项之后
默认对手会静默变成别的 agent（正是这次差点发生的事）。

界面级验证脚本 `tools/verify_agent_combo.ps1`：把 `kAgents` 从 `src/mainwindow.cpp`
解析成期望值，启动真界面、展开三个下拉框，逐条断言**可见性**（"在 model 里"不算），
并显式打印缺了哪几条。实测输出：

```
combo #1: current = 'Alpha-Beta Pruning (深度=4)' | rows seen = 12 (agent rows 11) | missing = 0
combo #2: current = 'Alpha-Beta Pruning (深度=4)' | rows seen = 12 (agent rows 11) | missing = 0
combo #3: current = 'EVAB (学会评估的 Alpha-Beta)' | rows seen = 12 (agent rows 11) | missing = 0
agent selectors checked     = 3
every kAgents entry visible = True
RESULT: PASS
```

**这个脚本能证明什么、不能证明什么**（实测过再写下来的）：它能证明"每一行都看得见"，
但**不能**通过 UIA 驱动"点这一行把 agent 切过去" —— Qt 组合框弹出层的行接受
`SelectionItemPattern.Select()` 与 `InvokePattern.Invoke()` 却**什么都不做**（探针实测：
两个调用都返回成功，combo 的值不变、`onAgentSelected` 从不执行），而改用 `SendKeys`
需要前台焦点（本目录其它脚本刻意避开）。所以那一半由 `test_match` **[2.14]** 覆盖：
它用 `AGENT_PPOMCTS_MLP` 真的走了一局 6 手（走的就是 GUI 同一条
`ChessBoard::aiThinkForAgent` 路径），外加结构/命名/权重隔离四条断言。

---

## 零之二点二十五、对弈时崩溃：`env` 的并发访问把堆写坏了（2026-09）

**报障**：*"对弈时自动保存权重的时候导致程序崩溃了"*。

### 1. 先看清"崩溃"是什么形状（别顺着报障的措辞查）

Windows 事件日志里有三次 `chess.exe` 崩溃，异常码全是 **`0xC0000374`
= STATUS_HEAP_CORRUPTION（ntdll 报的堆损坏）**，出错偏移每次都一样：

```
出错应用程序名称： chess.exe
出错模块名称：     ntdll.dll        异常代码： 0xc0000374
```

`0xC0000374` 的含义是"**有人写越界 / 用了已释放的内存**"，而不是"某个函数返回了错误"。
所以"保存失败导致崩溃"这个方向一开始就不对 —— 保存只是**碰巧在旁边**。
另外：其中两次（04:55 / 11:29）**早于本轮所有改动**，说明这不是新引入的 bug。

### 2. 在真界面里逐项排除（UIA 驱动，见 `tools/verify_match_ui.ps1 -TrainIndex`）

| 配置 | 结果 |
|---|---|
| 对战AI(后台训练目标)=EVAB，A=EVAB | **崩** |
| 对战AI=PPO+MCTS-MLP，A=PPO+MCTS-MLP | **崩** |
| 对战AI=PPO+MCTS-MLP，A=Alpha-Beta（训练目标 ≠ 参赛者） | 不崩 |
| 对战AI=Alpha-Beta（不训练），A=PPO+MCTS-MLP | 不崩 |

两条结论：**(a)** 需要"后台训练的目标 == 对弈的某一方"；**(b)** 用 **EVAB 也能复现**
⇒ 与新加的 MLP agent 无关。

### 3. 搬进无界面的最小复现（`repro_concurrency`）

关键差别是**对弈必须跑在独立线程**（`test_match` 里"对弈在主线"的用例复现不出来，
这本身就是线索）。把形状照搬过去 —— 对弈在另一个线程、主线程按"每手一次"刷
`getAgentSelfCheck()`（界面自检 worker 干的事）、后台训练开着、结束后保存 —— 秒级稳定复现：

```
=== 对弈 x 后台训练 并发复现 ===
[ok] 后台训练已启动 (目标 = agent 5), 对弈每局 12 手
EXITCODE=-1073740940        <- 0xC0000374
```

### 4. ASan 给出精确位置（`/fsanitize=address`，独立 build 目录）

```
==ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 112
    #3 std::vector<Chess::HistoryRecord>::vector(...)        <- 拷贝构造
    #4 Chess::Chess                                         chess.cpp:526  (history(other.history))
    #5 DQNMCTSAgent::selfCheckReport                        dqnmcts_agent.cpp:1004   <- Chess probe(chess)
    #6 ChessBoard::getAgentSelfCheck                        chessboard.cpp
0x...6580 is located 0 bytes after 96-byte region      <- 按撕裂的 size 分配, 却按 112 字节搬元素
```

也就是：**拷贝一个正在被另一条线程修改的 `std::vector`**（新的 `env.history`）。

### 5. 真因：`env` 那把锁根本不对

`ChessBoard::env` 是**所有 agent 共用的试走棋盘**（每个 agent 构造时都拿它的引用），
而它有两个访问方从来没有同步：

* `aiThinkRaw` / `aiThinkForAgentRaw` 开头的 **`env = chess;`（写！）写在锁外** ——
  `=` 会把 `envelope.history` 整个替换掉；
* **Alpha-Beta / MCTS 两个分支在 `env` 上搜索（moveForward/moveBack → `env.history`
  push_back）却完全不持锁**（只有 RL 分支拿了 `m_agentMutex`）。

于是"加锁的读者"（界面自检 worker 的 `Chess probe(agent->chess)`、以及现在也加了锁的
`getAgentSelfCheck` / `saveCurrentAgentModel`）**挡不住不加锁的写者** —— 锁本身没错，
错的是"不是所有人都用同一把锁"。

### 6. 修法：一句话的规则 + 两处补锁

> **凡是碰 `env` 的地方，都拿 `m_agentMutex`。**

* `aiThinkRaw` / `aiThinkForAgentRaw`：`env = chess;` 进锁（一个单独的作用域）；
* `ABAGENT` / `MCTS` 分支补锁（它们也在 env 上搜索）；
* 顺带把两个"裸的"读者也接进同一把锁（它们各自都是独立的真问题）：
  `saveCurrentAgentModel()`（原来直接从保存线程序列化主 agent 的网络，与后台训练
  写同一张网 —— 这就是"对弈结束静默保存"最容易撞上的那一处）与
  `getAgentSelfCheck()`（读常驻 agent 的搜索树/网络）。

同时把界面那两条后台线程改成**常驻 + 请求标志**（`m_saveThread` / `m_selfCheckThread`），
因为它们在 GUI 线程上 `join()` 或同步等锁会把界面冻住几秒
（"上一场还在写 558 MB，这一场结束就要 join" 是原来就有的隐患）。

### 7. 验收

```bat
cmake --build <build> --target repro_concurrency test_match
<build>\repro_concurrency.exe 2 12 5      :: 修前 0xC0000374 秒崩; 修后 "全部跑完, 没有崩溃"
<build>\repro_concurrency.exe 2 12 7      :: EVAB 同样
<build>\repro_concurrency.exe 2 12 10     :: PPO+MCTS-MLP 同样
:: ASan 版本 (独立目录; 二进制已被 .gitignore 排除, 复跑不必重编):
::   cmake -S . -B build-asan -G Ninja -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 ^
::         -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi /Od"
::   cmake --build build-asan --target repro_concurrency
::   (把 clang_rt.asan_dynamic-x86_64.dll 放到 exe 旁边, 并把 Qt 的 bin 加进 PATH)
build-asan\repro_concurrency.exe 2 10 5   :: 修前 heap-buffer-overflow, 修后干净
:: 真界面 (需要真桌面):
powershell -File tools/verify_match_ui.ps1 -TrainIndex 7 -AIndex 7 -BIndex 0 -Games 1
powershell -File tools/verify_match_ui.ps1 -TrainIndex 5 -AIndex 5 -BIndex 0 -Games 1
```

实测：修前 agent 5/7/10 全部秒崩；修后全部通过；ASan 干净；真界面两种配置各跑一遍
**新增崩溃事件 = 0**，且"已静默保存权重"那一行与 `[weights] 保存 … ms` 计时日志都正常出现。
`test_match` 新增 **[2.16]**（对弈在独立线程 + 并发刷自检 + 同一 agent 的后台训练）
作为回归钉子；`repro_concurrency` **故意不进 ctest**：它压的是时序，调度一变可能一时不
复现，拿它当门禁只会得到随机红灯。

---

## 零之二点二十六、"开关写好了、也测过、其实一直没生效"：构造函数建网之后才赋值（2026-09）

**症状**：为了实现"把 agent 还原回 `59e5233`"，给 `SACAZAgent` 加了一个运行时开关
`bool legacyNet`（true = 用 `TanhNorm<Linear>` 且 `r=1` 表达 59e5233 的那层隐层激活）。
开关接线完毕、编译通过、`bench_sac_mcts_min --legacy-net` 也跑出了"与基线逐手等价"的
结果。**但那个开关从来没有生效过**：

```cpp
SACAZAgent sac(board, ...);
sac.legacyNet = true;      // <- 赋值在这里
// 而 SACAZAgent 的构造函数里:
//     actor = buildNet(true);  q1 = ...; q2 = ...; q1Target = ...; q2Target = ...;
// 5 张网**在建网时**就决定了用哪个层类型, 之后改成员没有任何作用。
```

**为什么没被发现（这一条比 bug 本身值钱）**：那个开关的两种取值**本来就几乎等价**
（见下一条），所以"两种设置跑出同一结果"被读成了"等价性验证通过"，而不是
"开关没生效"。**凡是靠"回显开关的值"做的检查，都必然通过。**

**修法（两层）**：
1. **让错误的写法编译不过**：`legacyNet` 改成 `const` 成员，只能由**构造参数**给出
   （`SACAZAgent(..., bool legacyNet_ = false)`）。事后赋值直接报错。
2. **让面板说真话**：`SACAZAgent::hiddenActivationName()` 不回声开关，而是
   `dynamic_cast` 读 **actor 第 2 层的真实类型**，印在自检面板第一段。
   上一轮那个"激活被换成 `TanhNorm<Sigmoid>`、随机权重棋力掉 26 个点"的回归，
   面板上原本**一个字都看不出来**（同形状层 ⇒ 参数量、权重指纹全都一样）。

**顺带被证伪的一件事**：那个开关想表达的等价关系**根本不成立**。
`TanhNorm::forward` 把偏置加在 tanh **外面**（`o1=Wx; o1*=r; o2=tanh(o1); o=Fn(o2+b)`），
而 `Layer<Tanh>::forward` 是 `tanh(W·x + b)` —— 取 `Fn=Linear`、`r=1` 只在 `b ≡ 0` 时相同。
实测（`test_sacaz` [14]，同权重同局面，稀疏 MLP 专家骨干）：max|Δπ| = 1.8e-07、
**max|ΔQ| = 8.9e-06**。所以开关**已删除**：需要 59e5233 的那一层，就用
`Layer<RL::Tanh>` 那一行代码本身（当前 `buildNet` 与 `git show 59e5233:src/sacazagent.cpp`
逐行相同，17 行全同）。"等价"这件事只能拿**同一输入比输出**来判定 —— 与
`docs/sac_regression_2026_09.md` §7 第 2 条同源。

**回归钉**：`test_sacaz` [14] 断言"同权重同局面下，59e5233 还原版（派生类
`SACAZLegacyAgent`）与 `SACAZAgent` 的策略/双 Q 输出**逐位相同**"，
`tools/verify_sac_golden.ps1` 钉住走法序列，`test_match` [2.11] 钉住两支的权重文件名不同。

---

## 零之二点二十七、"训练让 agent 变弱"：默认值也是结论，而且**方向可能反了**（2026-09）

**症状**（用户实测 + 本轮受控复现）：界面对弈里 SAC **不训练反而更强** ——
同一协议 20 局、固定对手随机流：完全不训练 **52.5%**，只 rollout 25.0%，
rollout + 从搜索学 37.5%。与 `docs/arena_sac_vs_ppo_report.md` §5.2 早先记过的
"训练 14 局后 38.3%、不如随机权重"是同一件事。

**根因**：α 自动调节的**口径**。上一轮以"实测 α 恒为初值 0.200 ⇒ 自动调节名存实亡"为由，
把目标熵从 **0.98 降到 0.5**、把 alpha 学习率从 **1e-3 提到 5e-3** —— 理由本身没错，
**结论是错的**：这一改让 α 迅速缩小、策略被"**没有信息的 Q**"推着走
（训练后 `|Q|` 均值 **0.034**，与随机初始化 0.06 同量级 ⇒ critic 等于没学到）。

**逐个消融（20 局/档，其余条件相同）**

| 配置 | 胜-负-和 | 得分率 | 训练后 \|Q\| |
|---|---|---|---|
| 熵比 0.5 / αlr 5e-3（改后默认） | 2-7-11 | 37.5% | 0.034 |
| 只改 αlr → 1e-3 | 8-11-1 | 42.5% | — |
| 只改熵比 → 0.98 | 10-2-8 | 70.0% | — |
| **熵比 0.98 + αlr 1e-3（= 59e5233 的值）** | **13-0-7** | **82.5%** | **2.01** |
| clampTarget→0 / huberDelta→0 / 两者都改 | 2-7-11 | 37.5% | — |

**就地配对验证（同 seed、同 mcts-srand）**：seed 20240901 **37.5% → 82.5%**，
seed 777 **35.0% → 72.5%**；合池 40 局 **36.25% → 77.5%**，分胜负局 24:2 vs 4:15，
**Fisher 双侧 p = 1.2e-06**，约 **+313 Elo**。

**修法**：`src/sacazagent.cpp` 的构造初始化表改回 `entropyRatio(0.98f)` /
`learningRateAlpha(1e-3f)`（只影响最新 SAC；派生类本来就显式写这两个值，行为不变）。
`clampTarget=2` / `huberDelta=1` **保留**（实测比关掉更好：82.5% vs 67.5%）。

**回归钉**：`test_sacaz` [14] 断言"两支的熵比与 alpha 学习率一致"（防止以后只改一边、
把差别误读成"新旧口径"）；`test_sacaz` [16] 与 `test_match` [2.17] 钉住"关掉 rollout 也训练"。

**教训（写进 `docs/session_2026_09_sac.md` §4）**：**默认值也是结论**。
"α 恒为初值"只是"自动调节没在动"这一现象，不构成"该把目标熵降到 0.5"的推论 ——
数据驱动的默认值必须写清测量协议，否则下一个人无法判断它是否还成立。

**完整清单与下一步 4 项验证**（① critic 尺度新状态、② 还原版开约束、③ 塑形 200 局配对、
④ 界面奖励曲线换学习口径）：见 [`docs/session_2026_09_sac.md`](session_2026_09_sac.md)。

---

## 五、当前待办与优先级（未修复项）

> **本节已按当前状态重写。**
> 原来的 P0/P1 清单（`StoneMap` 初始化、走法合法性、
> 将杀判定、`detach()`、`Steps` 加锁、SQLite、`MM` 扁平化……）**已经全部完成**，
> 见"零、修复进度"的 A1–A19 / B1–B19。下面只列**当前仍未做**的事。

### P1（正确性 / 会咬人的潜在问题）

1. ~~**`MM::ikjk` / `kijk` 的 SIMD 内核改成累加**（现在是赋值，标量是累加）~~ **已做（R1.5，
   2026-09）**：两条路径都是累加，`test_grad` C 节把四行都钉成断言；见零之二点十五 §1。
2. ~~**补 `gemv_kikj`**（反向 GEMV）~~ **已做（R1.5）**：`kikj` 从 1.060 → 0.070–0.098 ns/MAC
   （≈13×，反向/前向 12.6× → 0.8×），`learnFromReplay(64,2)` 490 → 304 ms（1.61×）；
   见零之二点十五 §2。
3. ~~**`MM::*` 入口加 debug 断言**（维度与连续性）~~ **已做（R1.5）**：`#ifndef NDEBUG` 下的
   形状契约断言，用 `/UNDEBUG` 重编 `test_grad`/`test_sparse_moe` 跑通（11+67 项断言），
   并当场抓到 `lstm.cpp` 一处越约调用；见零之二点十五 §3。
4. **SQLite 跨线程 / 写库未接线**：连接在加载线程建、查询在 GUI 线程用；`recordMove/
   startGame/endGame` 全仓零调用，"每步落库 → 可回放"实际未接线。需要设计决定
   （每线程连接 vs 全部集中在 GUI 线程）。 → A14

### P2（棋力 / 训练质量）

5. **训练数据管线**：预训练的利弊分析见 `docs/agents_design.md` §10。要点：
   DQN 系当前"每步一次更新"破坏回放缓冲的 i.i.d. 假设，建议改成
   **只写回放、攒批更新**；PPO 系的优势估计在稀疏终局奖励下需要**显式自举**
   （否则 64 步内没有终局 → 优势几乎全 0 → 梯度是噪声）。
6. ~~**`DQN` 的 Q 头是 `Layer<Sigmoid>`**（值域 `(0,1)`），而奖励含终局 ±1 与被吃红子
   产生的负值 → 结构上无法表示负 Q。上游只修了 `convdqn.cpp`，`dqn.cpp` 没修。~~ **已做
   （R1.5，2026-09）**：四个骨干分支的 Q 头都改成 `Layer<Linear>`，`test_dqn` 新增 Q 值域
   探针（负值出现即证据）；见零之二点十五 §4。 → B18
7. **搜索的时间控制与迭代加深**：`AB_DEPTH`/`MCTS_SIMS`/`PPO_SIMS`/`DQNMCTS_ITERATIONS`
   都是固定值（已集中为常量，不再散落在注释里）；EVAB 有 `timeBudgetMs` 但其余没有。
   **稀疏 MoE + TB 专家那个变体现在也是固定 16 次模拟** —— 它最需要时间控制
   （一次模拟 10 ms，同一个预算下不同局面能跑的模拟数差很多）。
8. **QSS 主题仍未接线**：`QssLoader::load()` 零调用（`res.qrc` 已编入目标，资源无人读）。
   → B17
9. **没有一个骨干拿到棋力结论**：`bench_moe` 的 A/B/C/D 等时对弈**全是和棋**
   （连自我对照也是），因为参赛的都是随机初始权重，预训 3 局等于零。要谈"稀疏 MoE
   能否提升效果"必须投入真正的训练预算（自对弈几千局量级）+ 预训=0 的对照，
   这件事**还没做**，见 `docs/agents_design.md` §11.4.2 (6)。
   **2026-09 把 PPO 的专家也换成 `TB<16,360>`（容量 17.7×、时间 6–30×，见 §18.1）
   之后，这件事变成更贵也更值得做的一项** —— 现在有四个骨干（MLP / 稀疏MoE(MLP专家) /
   稀疏MoE(TB专家) / 稠密MoE(TB,对照)）和两个算法（PPO / SAC+AZ）都还没有棋力数字。
10. **突破多线程搜索的访存墙**（零之二点十三 ④，本轮最重要的未修项）：靠堆线程只能拿到
    ~2.0×（2026-09 的 R1 之后；R1 之前连 1× 都不到 —— 策略头只算合法列把每模拟权重流量从
    5.81 MB 压到 3.75 MB，见 `training_optimization.md` §9.4）。
    要再提速得让**一次权重读取服务多条样本** ——
    (a) **批量叶子前向**：一次选多个叶子（virtual loss 防重选）再合成一个 batch 前向，
        把策略头（与剩下的 MoE 专家 / critic）权重的流量摊到 batch 条样本上。这会改 MCTS
        的选择-评估循环，是最有收益也最伤筋动骨的一项；
    (b) **权重低精度化**（fp16/int8）：字节数直接砍半/砍到 1/4，不动算法结构，
        但要重新验一遍梯度与数值稳定性；
    (c) 把 8100 维策略头换回更小的动作表示 —— **与已定的动作空间方案
        （`fromCell*90 + toCell`）冲突，不作为选项**，只在此记录取舍。
11. **PG 的物质收益接上**（零之二点十三 (2)）：先定标（`computeReward` 的 `value*10`
    与终局 ±1 的量纲差一个数量级），再把即时奖励与终局奖励按权重相加，
    并加一条"轨迹里必须同时含即时项与终局项"的断言。
12. **PPO 的在线路径（`exploreAndTrain`）还没走回放池**：P4 的回放池 + 梯度累积只接了
    `commitEpisode`（自对弈路径），GUI 里"走子前先探索再训练"那条路仍是 `learnSelfPlay`
    逐样本更新。要么统一，要么在文档里写清为什么两条路 deliberately 不同。
13. **4 个慢测试没进 ctest**（`test_dqn` 466 s、`test_dqnmcts` 410 s、`test_evab` 680 s、
    `test_ppomcts` 51 s，见本轮实测）：现在只能手动逐个跑。建议给它们注册成
    ctest 的 `-L slow` 标签（默认不跑，`ctest -L slow` 才跑），而不是完全不注册 ——
    否则"改完记得手跑那 4 个"只存在文档里。
    **2026-09 之后 `test_ppomcts` 更慢了**（PPO 换 TB 专家，见 §18.1），这条的优先级跟着涨。
14. **`RL::SAC` 现在能跑了，但仍然没有调用方**（2026-09）：它已经补齐 P3/P4/MoE/R2 与
    43 项断言（`test_sac` 进 ctest），但哪个 agent 都没用它 —— `SACAZAgent` 是另一套
    （掩码 + MCTS + AZ 目标，见 `sacazagent.h` 顶部"与 `RL::SAC` 的关系"）。要么把
    `SACAZAgent` 收敛到 `RL::SAC` 上，要么明确把 `RL::SAC` 标记成"参考实现/教学用"，
    别让它继续以第三种状态存在（能跑但没人用，下一个改它的人还得先读一遍才知道）。
15. **`SACAZAgent::replayEpochs` 默认 1**（= 与改版前逐位一致）：机制接好了但默认没开。
    要不要在自对弈训练路径上开到 2，需要一个"同权重、只改这一个开关"的对照实验
    （那边样本生成比这边贵得多，理论上划算；但在线路径里用户在等这一步，不该动）。
16. **`test_ppomcts` 的"优化器占比"是个不稳的测量**（2026-09 发现）：compute budget 一节
    把 `RMSProp` 单独拿出来反复调、不重新 backward —— 它是纯访存微基准，机器一忙就被挤。
    同一份二进制：空载 trainStep 305 ms / 单独计时的优化器 277 ms（占 **91%**，可信值）；
    有别的测试在跑时 335 ms / 439 ms（**超过 100%**，打出的每样本成本是负数）。
    另外梯度被清零后 `v = rho·v` 会衰减到非规格化数（R1.5 记过）。现在该行加了失效判据，
    打印"不可用"而不是负数；要拿可信占比得让每次调用前梯度都非零。
17. **SAC+AZ 的 AlphaZero 监督项会被"训练过的 critic"压住？**（2026-09 的一个**未定论**
    观察）：配对 A/B（同权重/同池/同步数，只改 `hasSearch`）在**全新** agent 上给出
    均值 0.316 vs 0.017（监督项明确有效，18×）；但先在同一个局面上用 30 批把 critic
    训到 |Q| ~ 0.4+ 之后，同一个协议只有 0.0139 vs 0.0100 —— 8 步内监督项拗不过软 Q 项。
    没隔离出是 **critic 的量级** 还是 **π 的饱和**（dz 正比于 π，π(目标) 被压到 0.014 时
    推动力也跟着变小）在起作用。要做的实验：把 critic 的输出人为缩放到同量级、
    再跑同一协议；以及把 30 批换成 5/10/20 批看转折点在哪。**这件事重要**因为
    AlphaZero 式训练里策略改进几乎全靠这一项。
18. **四个骨干都还没有棋力结论**（与第 9 条同一件事，这次多了一条证据）：
    `bench_sacaz_vs_ab` 在**随机初始化权重**下四种骨干都是 0 胜 6 负（全是 mate、
    平均 28.8 手），只有 TB 那条不是"随机走子"（它每步真的跑了 16 次 MCTS 搜索）。
    要谈棋力仍然需要训练预算 + 预训=0 的对照。
    **2026-09 补充（新 agent DQN+AB，原名 NeuralAB）**：它第一次让神经 agent 对 AB
    拿到**和棋**（0 胜 2 负 2 和，而关掉搜索是 0 胜 4 负 0 和），但**仍然是随机权重**，
    所以这依旧不是棋力结论。而且实测给出一条反直觉的方向性结论：**叶子没训准之前，
    把搜索深度堆上去是负收益**（同一 TB 骨干 256 → 1024 节点：12.5% → **0%**；
    MLP 骨干 4.4 层 0 胜 4 负）—— 所以下一轮的杠杆是把 V 训准，
    而不是先加深度或加模拟次数。

    **2026-09 再补充（手工锚优化之后的结论修正）**："像 EVAB 那样先从手工评估蒸馏"
    这条路**做到了**（与手工锚 corr 0.869、vStd 0.091，8192 局面 × 3 遍只要 6.4 s），
    但它**仍然 0 胜 4 负 0 和**，与随机权重时一模一样（都是 24.0 手）——
    因为**蒸馏手工评估的上限就是手工评估本身**，而 ABAgent 深度 4 用的是同一个
    `evaluate()` 而且是**全宽**搜索；一个"手工评估的有损模仿（corr 0.87）+
    选择性展开（每节点 2~5 个分支）"当然赢不过它。
    所以这一条的结论要修正为：**手工锚只能当起点（防止叶子是垃圾），棋力必须来自
    手工评估没有的东西**（终局胜负的真实概率）—— 即自对弈 + TD，
    以及 quiescence（AB 不在吃子序列中间停手）。
    另外这一轮还量出一条方法论教训：**"gap 降低"曾经是假进步**
    （把 V 压成常数就能降 gap），必须同时看 corr 与 vStd —— 详见
    `agent_dqnab_design.md` §7.3 第 4/5 条。

19. **【2026-09 新增，当前最高优先级】把 value 训出区分度。** 诊断矩阵已经把它指成
    唯一该修的层：`bench_diag` 实测 **EV = −0.0293 < 0**，且 **283 个样本里 281 个落在
    同一个校准桶**（预测 +0.148 / 实际 +0.000）—— V 基本是"走子方恒为正"的常数偏置。
    因果链是 **V 无区分度 → 叶子把吃子看得更低 → Q(吃子)−Q(退让)=−0.098 →
    白吃子命中率 10%**。而**噪声/温度/模拟数都救不了**（开/关根噪声的吃子访问份额差
    −0.0037）。具体做法按顺序：
    (a) 先解决**检查点**问题 —— `weights/` 下所有文件都与当前网络不兼容
        （C14：2 150 124 vs 52 388 888），任何"续训"都是静默地从随机权重重来；
    (b) 再解决**标签质量** —— 现在唯一现造出来的检查点（`bench_ppo_distill
        --positions=1500 --depth=4 --actor=1`）的 critic 留出 MSE **没有改善**
        （0.0139 → 0.0144），actor 的 top-1 一致率只有 3.0%（机会 2.5%）；
        数据量需要提到 8000+ 局面（`docs/training_optimization.md` §7.8 实测过
        "加数据有效"，而 1500 局面 × 108 次更新太少）；
    (c) 用 `bench_diag` 的 **EV / 校准分桶**当作"V 训准了没有"的判据，**EV 转正之后**
        再看白吃子命中率。**在此之前不要再去扫 `c_puct` / 温度 / 模拟数** ——
        §20.4 已经排除掉"搜索没产出目标"（KL(访问‖先验)=1.32）。
20. **【2026-09 新增】`weights/` 检查点集体失效，需要一个迁移或重训的决定。**
    C14 的副产品：`bc20k / bc40 / bc8k_d3 / bc8k_d4 / bc_full / distill / r2_sp /
    sweep_prior / verify_*` 全部是旧架构（约 2.15 M 参数），当前网络是 52.4 M。
    要么写一个"旧→新"的骨架迁移（把公共层搬过去、新层重新初始化），
    要么明确标记为"历史产物、不可用"并从零重训。
21. **【2026-09 新增】`dqnmcts_agent.cpp` 的混帧**（C13 的遗留）：它的 `encodeState`
    是固定红黑符号、**没有走棋方通道**，叶子值是绝对视角，却配一套交替翻号的 backup
    ⇒ 帧不一致。**不能像另外三个 agent 那样只加一个负号**（那会把错误换个方向），
    需要先决定 DQN 的 Q 到底是绝对视角还是走子方视角，再统一 backup 与选择器。
    **2026-09 进展**：已先做**终局口径统一**（三处收尾改走 `getResult()` +
    `outcomeForMover()`），并修掉 `trainVsRandom` 里那个"每一手都被写成 −1.0f
    （走一步 = 输一盘）"的错——它才是"损失下不去、最低 22"的原因。
    `evaluateLeaf` 那一处**仍未统一**，且"Q 是哪个帧"这个决定仍待定（见 24 号）。
24. **【2026-09 新增】DQN+MCTS 的表示层三个前置闸门（`probe_dqnmcts_aliasing`）。**
    这一条是把 21 号从"文档里记着"变成"可复跑的数字"：
    * **动作别名**：标准开局 44 个合法着法只落进 **38** 个 Q 槽位（挤掉 6 个，最挤槽位
      背 3 个着法）；中局 96 个局面平均挤掉 **5.16** 个，最挤槽位 **4** 个。
      **跨局面碰撞率仅 0.03 ⇒ 128 槽位本身够用** —— 这条同时**否掉了**"128 槽位是灾难"
      的推测（我原先也是这么猜的）。
    * **状态不可分**：走子方 / 重复 1 次 vs 3 次 / 无吃子 0 手 vs 120 手，`encodeState`
      输出**逐字节相同**；90 维 = 一格一个子力值，规则上下文通道 **0 个**（PPO 是 19 平面）。
    * **终局通道**：6 局 × 200 手，`isGameOver()` 看见 **0** 次终局、`getResult()` 看见 1 次、
      **5 局撞上限** ⇒ 终局 ±1 从没进过 Q 目标。而 `evaluateLeaf` 至今仍用
      `isGameOver()` 判终局（判和与将杀都不是终局）。
    * **回放池**：1440 ply → **1426** 个互不相同的局面（去重率 **99.0%**）⇒
      容量装的确实是新信息。但"4096 → 20000"**仍不会更快**：一局写 60 条、抽走 15×32=480 条，
      4096 条要 **68 局**才淘汰一轮 ⇒ 前 68 局里容量对训练轨迹的影响严格为零；而 20000 会把
      换血拉到 **334 局**、样本陈旧度拉长约 5 倍。真正的杠杆是 `docs/agents_design.md`
      §10.4 / `docs/training_optimization.md` §10 那条**未做**的建议：DQN 系改成
      "只写回放、攒批更新"，别每 4 手更新一次。
    **附带**：`test_dqnmcts` 原来**一条断言都没有**（只打印），现在有 16 条（报告非空/含关键
    字段、只读——调用前后棋盘哈希不变、可重复、终局通道一局恰好一条）并按失败数返回退出码；
    它仍不进 ctest（本机实测 339.7 s），但手动跑时 `%ERRORLEVEL%` 有意义了。
    **这批断言当场抓到了一个真 bug**：自检计数第一版写在"每走一手"之后，12 手的一局被记成
    **12 局**（断言报 `12 != 1`）—— 面板上的数字是要给人当决策依据的，所以它自己必须被钉住。
22. **【2026-09 新增】把 `TrainDiag` 接到训练路径上。** 接口（policy CE / KL /
    策略熵 / value MSE / EV / MoE 负载 CV）已经在 `src/rl/diag.h` 就位，
    `PPOMCTSAgent::moeUsage()` 也有，但**采集点还没接**：现在这几格只有
    `bench_diag`（它不训练）之外的空缺。接上之后才能看到"policy CE 是不是长期 >3"、
    "策略熵有没有秒归零"、"MoE 有没有死专家"这三类信号。
23. **【2026-09 新增】快照 Elo 阶梯与遗忘检测**（外部建议里唯一还没做的生态指标）。
    需要常驻的对手池 + 定期快照；现在每次 `bench_*` 都是新进程，没有这个基础设施。
    在它之前，"棋力有没有涨"只能靠 `bench_policy_agreement` 的代理指标，
    而那条路已经被证明会与胜率脱钩（`training_optimization.md` §7.10）。

### P3（工程化）

14. **`GameDatabase::recordMove` 每步一次自动提交**（一局 200 次 fsync），无事务。 → B16
15. `PGEagent::train` 里 `currentColor = COLOR_BLACK` 让黑先走、无走法分支后二次
    `reinforce1` + 二次计数。 → B15
16. 清理死代码、补 `const`、抽出 5 个 RL agent 的公共基类（`exploreAndTrain` 已经在
    `AgentBase` 上，其余仍各写各的）。`DQNAgent::trainAfterMove` 现在无调用方（本轮顺手修了
    它的视角口径，但更该做的是删掉或接线）。
17. 文档对齐：`docs/analysis.md` 的文件树与 §5/§6 条目、`task_progress_*.md` 的历史
    迭代数（那些是"当时的记录"，不必改，但要在顶部注明已过期）。
