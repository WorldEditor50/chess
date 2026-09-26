# SAC+AZ (稀疏 MoE + TB 专家) 的模型架构图

配套文档：[`dev_sacmoetb_2026_09.md`](dev_sacmoetb_2026_09.md)（改动理由与实测数字）。
这一份只画结构，数字都是本机实测（i5-10400F / MSVC 19.44 Release / AVX2，`d_model = 1263`）。

> **这份文档回答"改成什么样"；"为什么"在另一份的 §2.5**：
> (A) 独立骨干（5 张网各背一套）为什么难训练 —— 参数量/数据量比例、5 张里 2 张从不学习、
> 三份骨干各看 1/3 的梯度、双 critic 去相关被污染；
> (B) 共享骨干为什么有效 —— 并且**更正**了"骨干收到 3 倍梯度信号"这个直觉说法
> （本工程 `clipGrad` 会把每个张量的梯度归一到单位长度，所以变的是**方向**不是**大小**）；
> (C) 共享骨干的 8 条缺点与副作用（目标冲突、双 critic 去相关被削弱、学习率自由度、
> 视图共享层对象的**隐形契约**、路由统计语义变化、权重兼容性、以及"容量并没有真丢 3 倍"
> 这条其实不是缺点的点），每条都标了【实测】/【推理】/【未测】。

全文的尺寸约定（默认表示；`SACAZ_ALIGNED_REPR=1` 时见每张图下的括号）：

```
STATE_DIM  = 17 × 90 + 3 = 1263      状态: 14 棋子平面 × 90 格 + 3 个规则上下文标量
                                       (对齐表示: 19 × 90 = 1710)
hiddenDim  h              = 64         骨干之后、输出头之前的瓶颈宽度
ACTION_DIM                = 128        (对齐表示: 8100)
```

---

## 图 1 · 改动前：五张各自独立的网（`TrunkMode::Separate`）

```
                        SACAZAgent  (改动前唯一口径)
                                │
   ┌────────────┬───────────────┼───────────────┬────────────┐
   ▼            ▼               ▼               ▼            ▼
 actor         q1              q2          q1Target     q2Target      ← 5 个 RL::Net
   │            │               │               │            │
   │            │               │            (无梯度)     (无梯度)
   ▼            ▼               ▼               ▼            ▼
┌───────┐   ┌───────┐      ┌───────┐      ┌───────┐    ┌───────┐
│ 骨干  │   │ 骨干  │      │ 骨干  │      │ 骨干  │    │ 骨干  │      ← 5 份**互不相同**
│ 28.8M │   │ 28.8M │      │ 28.8M │      │ 28.8M │    │ 28.8M │        的 TB 骨干
└───┬───┘   └───┬───┘      └───┬───┘      └───┬───┘    └───┬───┘
    ▼           ▼              ▼              ▼            ▼
  128         128            128            128          128
  logits      Q 值           Q 值           Q 值         Q 值
```

* 合计参数量 **144,131,280**（≈ 576 MB 权重 + RMSProp 的 `v` 缓冲）
* 一次叶子估值要跑 **3 次骨干**（策略 1 + 双 Q 2）；
  一个训练样本要跑 **6 次骨干前向 + 3 次骨干反向**
* 实测：`learnBatch(4)` = 835.0 ms（208.74 ms/样本），`selectMove` = 20.05 ms/模拟

---

## 图 2 · 改动后：共享骨干 + 三个头（`TrunkMode::Shared`）

```
                        state (1263)
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
      ┌─────────────┐                 ┌─────────────┐
      │   trunk     │                 │ trunkTarget │        ← 只有 **2** 份骨干
      │ (在线/有梯度)│                 │ (目标/只前向)│           (改动前是 5 份)
      └──────┬──────┘                 └──────┬──────┘
             │ h (64)                        │ h (64)
      ┌──────┼──────┐                 ┌──────┴──────┐
      ▼      ▼      ▼                 ▼             ▼
 ┌────────┐┌──────┐┌──────┐     ┌───────────┐┌───────────┐
 │actorHead││q1Head││q2Head│     │q1TargetHead││q2TargetHead│
 │64→128  ││64→128││64→128│     │  64→128    ││  64→128    │
 └───┬────┘└──┬───┘└──┬───┘     └─────┬─────┘└─────┬─────┘
     ▼        ▼       ▼               ▼            ▼
   π(a|s)    Q1      Q2              Q1_t        Q2_t
   (128)    (128)   (128)           (128)       (128)
```

**"共享"指的是层对象共享，不是接口共享。** `actor` / `q1` / `q2` 这三个成员在
共享口径下仍然是**端到端可用的完整 `Net`** —— 它们与 `trunk` / 各 `Head`
**指向同一批 `shared_ptr<iLayer>`**（见 `SACAZAgent::buildSharedNets`）：

```
trunk       = Net[ L0(moe), L1(tanh) ]              ┐
actorHead   = Net[ L2a(linear) ]                    │  L0/L1 是**同一批对象**
q1Head      = Net[ L2b(linear) ]                    │  （shared_ptr 别名）
q2Head      = Net[ L2c(linear) ]                    │
                                                    │
actor  = Net[ L0, L1, L2a ]   ─┐                    │
q1     = Net[ L0, L1, L2b ]    ├─ 视图: 与上面同源 ──┘
q2     = Net[ L0, L1, L2c ]   ─┘
```

* 合计参数量 **57,586,536**（2.50x 少）
* `test_sacaz` [17B] 断言了"视图确实与 `trunk` 共享层对象"——
  否则 `actor/q1/q2` 会是一份**永不更新的陈旧副本**，而前向、稀疏路径、自检面板
  全都照常工作（这正是这类改法最容易静默失效的地方）。
* 热路径**不走视图**（走视图 = 骨干被跑三遍，那就退回改动前的开销）：
  搜索与训练都显式地"`trunk.forward` 一次 + 三个头各 `forward` 一次"。

---

## 图 3 · 骨干内部：`SparseMoE<TransformerBlock<15,315>, E=4, TopK=1>`

```
   x (1263)
    │
    ├──────────────► gate = softmax(Wg·x + bg)        Wg: 4 × 1263,  bg: 4
    │                        │
    │                        ▼
    │                 top-1 选一个专家  S = argmax_i gate_i      ← 只算 1 个, 其余不碰
    │                        │
    ▼                        ▼
  ┌─────────────────────────────────────────┐
  │  expert_S : TransformerBlock<15, 315>   │   ← 单价 **6.20 ms** (旧口径 19.06 ms)
  │  (1263 → 1263, 见 图 4)                  │
  └────────────────────┬────────────────────┘
                       │
                       ▼
        o = Σ_{i∈S} gate_i · expert_i(x)      (1263)
        + 负载均衡辅助损失 L_aux = E · Σ f_i · P_i
          (addAuxGradient, 系数 SACAZ_MOE_AUX = 0.1; 没有它会路由坍缩)
```

载荷统计（`test_sacaz` [10]/[12c] 会打印）：4 个专家的使用计数、`max/min` 比值。
实测未修复前曾出现 `{0, 29, 0, 0}`（3 个专家饿死），修复后自对弈一段是
`{51, 80, 72, 93}` / `{57, 84, 95, 60}`（均衡）。

---

## 图 4 · TB 专家内部，以及那个**静默降级**

`TransformerBlock<NumHeads=15, d_ff=315>` 建在 `d_model = STATE_DIM` 上：

```
  x (1263) ───────────────────────────────────────────────────────┐ 残差 1
   │                                                               │
   ▼                                                               │
 LayerNorm1  gamma1/beta1: 1263                                    │
   │  x_norm1 (1263)                                               │
   ▼                                                               │
 MultiHeadAttention<15>                                            │
   for i = 0 .. numHeads-1:                                        │
     q_i = Wq_i · x_norm1        Wq_i : d_k × 1263                 │
     k_i = Wk_i · x_norm1        Wk_i : d_k × 1263                 │
     v_i = Wv_i · x_norm1        Wv_i : d_k × 1263                 │
     z_i = softmax(q_i · k_iᵀ / √d_k)      ← d_k × d_k 矩阵        │
     head_i = z_i · v_i          (d_k)                             │
   a = concat(head_0 … head_{n-1})   (1263, 见下面的"没铺满")        │
   attn_out = Wo · a            Wo : 1263 × 1263                   │
   │                                                               │
   ▼                                                               │
  x_res1 = x + attn_out  ◄──────────────────────────────────────────┘
   │                                            ┌── 残差 2
   ▼                                            │
 LayerNorm2                                     │
   │  x_norm2 (1263)                            │
   ▼                                            │
 Layer<Gelu>(1263 → 315)      FFN 上行          │
   │  ffn_hidden (315)                          │
   ▼                                            │
 Layer<Linear>(315 → 1263)    FFN 下行          │
   │                                            │
   ▼                                            │
   o = x_res1 + ffn_out ◄────────────────────────┘
```

**单价从哪来**（`.r1build/tbcost.cpp`，单实例实测）：

| 子块 | 旧口径 | 新口径 | 占新口径的比例 |
|---|---|---|---|
| MHA（含 15 个 `d_k × d_k` 的 softmax） | 7.15 ms | **3.09 ms** | 50% |
| FFN `1263 → 315 → 1263` | 0.11 ms | 0.11 ms | 1.8% |
| 尾部 `Tanh(1263→64)` + `Linear(64→128)` | 0.012 ms | 0.012 ms | 0.2% |
| **整个 TB 专家前向** | **19.06 ms** | **6.20 ms** | 100% |
| 整个 TB 专家 前向 + 反向 | 71.86 ms | 33.44 ms | |

### 降级的机制

```
d_model = 1263 = 3 × 421          （421 是素数 ⇒ 因子只有 1 / 3 / 421 / 1263）

老口径: numHeads = max{ h ≤ 15 : h | 1263 } = **3**
        d_k = 1263 / 3 = 421
        注意力矩阵元素 = 3 × 421²  = 531,723      → 单专家前向 19.06 ms
        heads[15] 数组: [■][■][■][·][·][·][·][·][·][·][·][·]
                        └─用─┘  └──── 12 个是死的 (80%) ────┘

新口径: numHeads = 15          （HonorHeads: 按请求头数切, 允许没铺满）
        d_k = 1263 / 15 = 84  （整除截断, 余下 3 个坐标不进拼接, 仍走残差与 Wo）
        注意力矩阵元素 = 15 × 84² = 105,840       → 单专家前向 6.20 ms  (3.07x)
        heads: 全部在用, 一个死的都没有
```

`a` 的布局在新口径下（`numHeads·d_k = 1260 < 1263`）：

```
 a = [ head_0 | head_1 | … | head_14 | 0 0 0 ]
       └────────── 15 × 84 = 1260 ──────┘   └─ 尾部 3 个坐标
                                              没有 head 写它们 → forward 里显式清零
                                              (不清零会读到上一次前向的残留值)
```

`paramCount` 几乎不变（7,182,996 → 7,171,629，−0.158%）—— 这正是它当年能躲过
参数指纹/元素总数守卫的原因。变的只有"慢 3 倍"和"80% 的 head 张量是死的"。

### 为什么反向只快 2.15x

`ScaledDotProduct::backward` 的每一项：

```
  ∂L/∂v  = zᵀ·e              O(numHeads · d_k²)      ← 随头数变
  ∂L/∂ẑ  = e·vᵀ              O(numHeads · d_k²)      ← 随头数变
  ∂L/∂z  = Jᵀ·(∂L/∂ẑ)        O(numHeads · d_k²)      ← 随头数变
  ∂L/∂q  = (∂L/∂z)·k / d     O(numHeads · d_k²)      ← 随头数变
  ∂L/∂k  = (∂L/∂z)ᵀ·q / d    O(numHeads · d_k²)      ← 随头数变
  ∂L/∂x += Wqᵀ·dq + Wkᵀ·dk + Wvᵀ·dv     O(6 · d_model²)   ← **与头数无关**
  g.Wq += dq·xᵀ, g.Wk, g.Wv             O(3 · d_model²)   ← **与头数无关**
```

那两项 `~9·d_model²` 是**常数项**（1263² × 9 ≈ 14.4 M MAC），头数口径完全碰不到它 ——
所以单专家"前向"是 3.07x、"前向+反向"只有 2.15x。
`learnBatch` 一个样本 = 6 前向 + 3 反向，于是它被这一项拖住
（只修头数拿 1.28x，只上共享拿 2.76x，两个一起 3.59x）。
**这是下一轮最大的单点收益**，见 `dev_sacmoetb_2026_09.md` §7。

---

## 图 5 · 一次 MCTS 叶子估值（每个模拟一次）

```
改动前 (Separate)                          改动后 (Shared)
─────────────────────────────────          ─────────────────────────────────
 policySparse(state, legalIdx)              sparseLeaf(state, legalIdx, π, Q1, Q2)
   actor.forwardTrunk(state) ─┐ 骨干①         trunk.forward(state) ─┐ **骨干 ①**
   head.sparseLogits(...)     │                actorHead.sparseLogits(...) │
 qValuesSparse(state, ...)    │                q1Head.sparseLogits(...)    │
   q1.forwardTrunk(state) ────┤ 骨干②          q2Head.sparseLogits(...)    │
   q2.forwardTrunk(state) ────┘ 骨干③                                       │
                                                                           ▼
 骨干前向 **3** 次                            骨干前向 **1** 次 ────────────┘
 实测 20.05 ms/模拟                          实测 4.02 ms/模拟   (4.99x)
```

`sparseLeaf` 的语义与"`policySparse` + `qValuesSparse`"**逐元素相同**（同一个 `h`、
同一套合法集归一化）；`test_sacaz` [17B] 对它做了逐位断言（`max|Δπ| = max|ΔQ1| = 0`）。

---

## 图 6 · 一个训练样本（`learnBatch` 的一次迭代）

```
改动前: 6 次骨干前向 + 3 次骨干反向
────────────────────────────────────────────────────────────────────────────
  s' ─► actor      ─┐
  s' ─► q1Target   ─┤  前向 6 次
  s' ─► q2Target   ─┤  （每张网跑自己那份骨干）
  s  ─► actor      ─┤
  s  ─► q1         ─┤
  s  ─► q2         ─┘
  q1.backward(s) / q2.backward(s) / actor.backward(s)     ← 3 份骨干各反一次
  RMSProp: actor / q1 / q2                                ← 3 次（各更新自己的骨干）
  Polyak : q1→q1Target, q2→q2Target                       ← 2 份目标骨干


改动后: 3 次骨干前向 + 3 次"头"反向 + 1 次骨干反向
────────────────────────────────────────────────────────────────────────────
  s' ─► trunk        ─► actorHead                        → π(s')     ┐ 在线骨干
  s' ─► trunkTarget  ─► q1TargetHead, q2TargetHead       → Q1_t,Q2_t ┘ 目标骨干
  s  ─► trunk        ─► actorHead ─► π(s)                            ┐
                       q1Head    ─► Q1                              ├ 骨干只跑一次
                       q2Head    ─► Q2                              ┘
        │
        │  三个头各自 backward:
        │    actorHead.backward(h, dz)   → actorHead.inputGrad = ∂L_策略/∂h
        │    q1Head.backward(h, dQ1)     → q1Head.inputGrad    = ∂L_Q1/∂h
        │    q2Head.backward(h, dQ2)     → q2Head.inputGrad    = ∂L_Q2/∂h
        ▼
  gh = actorHead.inputGrad + q1Head.inputGrad + q2Head.inputGrad      ← **相加**
        │
        ▼
  trunk.backward(s, gh)                          ← 骨干**只反一次**
  RMSProp: trunk ×1 (learningRateTrunk) + 三个头各 ×1
  Polyak : trunk→trunkTarget, q1Head→q1TargetHead, q2Head→q2TargetHead
```

三处**必须按顺序/次数**的硬约束（写在 `src/sacazagent.cpp` 的注释里）：

1. 三个头的 `backward` 都要在 `trunk.backward` **之前** —— 它们读的是骨干的缓存输出 `h`，
   而 `Layer<Tanh>::backward` 结束时会把 `o` 清零；
2. 骨干的梯度是**相加**而不是"反向三次" —— 分三次反向 = 三次更新同一份权重 = 学习率乘 3；
3. `RMSProp` / Polyak 对骨干各只做**一次** —— 照旧遍历 `actor/q1/q2` 会把同一件事做三遍。

---

## 图 7 · 权重文件（两个口径**不能互相载入**）

```
Separate 口径  (AGENT_SACAZ, bench --trunk=separate)
  weights/sacaz_agent_actor     = 骨干 + 策略头      (3 个文件)
  weights/sacaz_agent_q1        = 骨干 + Q1头          目标网在载入时由在线网 copyTo 派生
  weights/sacaz_agent_q2        = 骨干 + Q2头

Shared 口径    (AGENT_SACAZ_MOE, bench --trunk=shared)   ← **改前缀了**
  weights/sacaz_shared_agent_trunk       = 骨干                 (4 个文件)
  weights/sacaz_shared_agent_actorhead   = 64 → 128
  weights/sacaz_shared_agent_q1head      = 64 → 128
  weights/sacaz_shared_agent_q2head      = 64 → 128
```

两个口径的第 2 个文件**语义完全不同**（一个是"骨干 + Q1"，一个是 `64→128` 的一层）。
共用前缀的后果不是报错，而是"有人为了兼容补一条回退"之后把独立口径的第一份骨干
当成共享骨干、另外两份丢掉 —— 静默、不报错、训练继续。

```
                       ┌──────────────────────────────┐
   weights/            │ sacaz_agent_*        (旧 MLP) │  AGENT_SACAZ
                       │ sacaz_old_agent_*    (还原版) │  AGENT_SACAZ_OLD
                       │ sacaz_old_moe_agent_*(还原版TB)│  AGENT_SACAZ_OLD_MOE
                       │ sacaz_shared_agent_* (本轮新) │  AGENT_SACAZ_MOE
                       └──────────────────────────────┘
```

---

## 图 8 · 一张总览：一次决策里谁被调用了

```
  ChessBoard::aiThink (GUI 线程)
      │
      ▼
  SACAZAgent::selectMove(color, simulations=40, temp, piOut)
      │
      ├─ 根节点: getLegalActions + policy()          ← 1 次 骨干前向 + 掩码 softmax
      │
      └─ for sim in 0..39:                            ← 每个模拟
           1. SELECT : 按 PUCT = -Q_child + c_puct·P·√N/(1+n) 下潜
           2. EXPAND : getLegalActions → **sparseLeaf**   ← 1 次骨干前向（图 5）
           3. VALUE  : π 上取 E_π[min Q] − α·log π          （最大熵软价值）
           4. BACKUP : negamax 翻号回传
           5. UNDO   : moveBack 逐手还原（对真棋局零副作用）
      │
      ├─ visitDistribution → learnFromSearchStep(...)  ← AGENT_SACAZ_MOE 里**关着**
      │                                                    (一次 learnBatch 对 175 ms/步
      │                                                     的界面预算还是太贵, 见源注释)
      ▼
  选访问数最大的孩子 → Step
```

训练侧（后台线程 / 自对弈 `trainSelfPlay`）走的是同一条 `learnBatch`（图 6），
只是 `simulations` 给到 64（`BG_TRAIN_SACAZ_MOE_SIMS`），并且 `hasSearch=true`
的样本会把 MCTS 的访问分布当作策略监督目标。

---

## 图 9 · 梯度流：独立骨干 vs 共享骨干（`dev_sacmoetb_2026_09.md` §2.5 的图）

```
独立口径 —— 三份骨干各吃自己那一条损失的梯度
───────────────────────────────────────────────────────────────────────────────
   state ──┬──► 骨干_A  ──► 策略头 ──► π ──► L_策略 ──► ∂L/∂骨干_A   (1 条路径)
           │
           ├──► 骨干_Q1 ──► Q1头  ──► Q1 ─► L_Q1  ──► ∂L/∂骨干_Q1  (1 条路径)
           │
           └──► 骨干_Q2 ──► Q2头  ──► Q2 ─► L_Q2  ──► ∂L/∂骨干_Q2  (1 条路径)

   三份骨干的输入分布**完全相同**, 却各自从零学一份表示; 每份只拿到 1/3 的更新次数。
   总计 144.1 M 参数, 而其中 q1Target / q2Target 那两份 (~40%) 几乎从不学习
   (Polyak tau=1e-3 / 每 64 步 => 20 局只挪 2~4%)。


共享口径 —— 一份骨干吃三条损失之和
───────────────────────────────────────────────────────────────────────────────
                          ┌──► 策略头 ──► π  ──► L_策略 ──┐
   state ──► 骨干 ──► h ──┼──► Q1头   ──► Q1 ──► L_Q1   ──┼──► gh = 三者之和
                          └──► Q2头   ──► Q2 ──► L_Q2   ──┘        │
                                                                    ▼
                                             骨干.backward(state, gh)   ← 只反一次

   注意 gh 是"三条损失的单位梯度之和", **不是三倍的步长**:
   `Optimize::RMSProp` 的 clipGrad 会把每个权重张量的梯度归一到单位长度
   (`dw /= dw.norm2()`), 所以大小信息被抹掉、有效步长恒等于 lr。
   共享改变的是**方向**(三条损失折中出来的方向), 不是**大小**。
   这个方向是好是坏 —— 本轮**没有测** (见 §2.5 (C1)/(C7))。
```

```
  A3 的量化对照: 每份骨干拿到的更新次数 (同一个训练样本)

    独立:  骨干_A  1 次    骨干_Q1  1 次    骨干_Q2  1 次
    共享:  骨干    1 次 (方向 = 三条损失之和) + 三个头各 1 次

  B1/B2 的量化对照: 同一个时间预算下能跑多少

    独立:  selectMove 20.05 ms/模拟  ->  175 ms 预算跑 ~19 次模拟
    共享:  selectMove  4.02 ms/模拟  ->  175 ms 预算跑 ~43 次模拟
```
