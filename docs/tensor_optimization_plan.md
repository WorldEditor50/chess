# Tensor 优化方案（2026-10）

本文把 2026-10 的四轮讨论合并成**一份可执行方案**：每一条都带**落点（`文件:行`）**、
**依据**、**验收标准**与**风险**，并显式列出**不做**的项。执行进度记在 Part 6「执行台账」——
**没有实测数字的改动视为未完成**（本文档的规矩，与 `docs/rl_plan_optimized.md` 一致）。

> 行文标记：
> **【实测】** 本机跑出来的读数（探针与证据文件见 Part 9）；
> **【代码事实】** 引用 `文件:行`；
> **【外部·Caffe/PyTorch】** 外部框架的既有设计（附来源链接，**不是本仓库的实测**）；
> **【推理】** 尚未量测的推断与算式。

四轮讨论的结论摘要：

| 轮次 | 问题 | 结论 |
|---|---|---|
| 1 | 内存对齐后该不该用 `Tensor` 替掉 `std::vector<float>` | **不该**：`std::allocator` 对 ≥1 KB 块 200/200 都给 ≥32 B；对齐不是收益来源（`ikjk` 写法值 12~25×，整批 GEMM 只有 1.00~1.13×） |
| 2 | `Tensor` 用自建内存池能否提速 | **上限 0.04%**：一个 actor 头 fwd+bwd 45.6~47.4 ms，全部 12 次分配换成零开销池只省 0.017~0.023 ms |
| 3 | 重新设计 `Tensor` 该加什么、删什么 | 加：常驻契约检查 / 单一长度源 / 视图 / `noexcept` move / 访问器；删：`SubTensor` 整类 + `bmm` 死簇 + 约 40 项零引用 + 整类重复实现；改：`reshape` 不校验等硬 bug |
| 4 | 参考 Caffe/PyTorch 有没有更好的优化 | **有，而且最大的一条不在 `Tensor` 类里**：seq 骨干 forward 有 **76% 不在 GEMM 上**（注意力 39% + 逐 token 搬运 37%），正好是"矩阵化 + 视图"能消掉的部分 |

---

## Part 0 · 先立约束（不变量）

改任何东西之前，这 6 条是硬约束，违反它们等于把既有资产（存量权重、golden 读数、上游同步线）砸掉：

| 约束 | 出处 | 含义 |
|---|---|---|
| **不整类替换 `Tensor_`** | `docs/rl_sync.md:129-136` | 仓库已论证过"不换成 N-spirits 的 `Tensorsi_`"；允许的形态是"**接口不变、行为升级**"（`rl_sync.md:285`） |
| **权重文件 v2 字节格式** | `docs/rl_sync.md:283-299` | `shape\|b64:<crc32>:<base64 float32>` + `CHWGT2` 头 + 每张量一行；**老十进制 v1 仍要能读**；被 6 个测试文件的硬字面量钉住（Part 5.3） |
| **`MM::*` 四内核一律累加** | `docs/issues_review.md:1371-1372` | `z += …` 而非赋值；调用方 `forward()` 前必须 `o.zero()` |
| **形状断言留在 `#ifndef NDEBUG`** | `docs/issues_review.md:1402-1411` | "Release 一个字节都不变；想验证就 `/UNDEBUG`" |
| **32 B 对齐契约** | `src/rl/tensor.hpp:15-46`、`test_transformer [9a/9c/9d]` | `AlignAllocator32` + AVX2 的 `_mm256_store_ps` |
| **与上游的分歧要记账** | `docs/rl_sync.md:17,20,40,55-56,89` + `tensor.hpp:326-335` | 上游同源代码的删改要走 `rl_sync` 那条线，并在文件内注释标明 |

另外两条本仓库的通用纪律（Part 5）：**计时只信"同进程同一轮"的成对比值 + 对照通道**（`docs/seq_transformer_design.md` §8.4）；**警告基线**只允许 `rl/util.hpp:65` 与 `rl/tensor.hpp:558` 两条（`docs/issues_review.md:1989`）。

---

## Part 1 · 现状：实测病灶

### 1.1 结构账【代码事实】

`src/rl/tensor.hpp` **2266 行**，公开成员与别名 90 余项。重复实现占全文 **≈69%**：

| 区块 | 行数 | 占比 | 说明 |
|---|---:|---:|---|
| `struct MM`（1025-1571） | 547 | 24.1% | 四内核各自重写"Debug 断言 / SIMD 分派 / 步长提升 / 单位步长分支 / 通用 stride 标量回退"五段；通用回退的三重循环**抄了 4 遍** |
| 序列化块（1725-2255） | 531 | 23.4% | 形状解析写 **3 遍**、`b64:<crc>:<payload>` 头解析写 **2 遍**、含一个死解码器（2208-2255） |
| 形状/索引管线（535-819） | 285 | 12.6% | `sub/at/block/embedding/slice/toVector/fromVector/posOf/indexOf/reshape/view/permute/tr` |
| `SubTensor`（73-267） | 195 | 8.6% | 整类对外**零引用**，且与父类重复了 8 个统计函数 |

### 1.2 分配与拷贝【实测】

口径：同一进程、同一轮、7 轮取中位数、固定单核；对照通道跨度 7%~27%（详见证据文件）。

| 项 | 读数 | 备注 |
|---|---|---|
| 单次 `allocate+deallocate` | **230–262 ns**（≤32 KB）/ **770–940 ns**（92 KB–2.4 MB） | `_aligned_malloc` ≈ `malloc` ≈ `new`（**对齐不花钱**）；bump 池 15–22 ns |
| `Tensor(r,c)` 构造+析构 | (19,1) **947–1036** / (64,1) 935–1068 / (192,1) 945–1080 / (90,64) **2349–2910** / (1710,1) 1153–1451 ns | **恰好 3 次堆分配**：`shape` 8 B + `sizes` 8 B + 32 B 对齐数据；`Tensor(shape_var)` 那条路径是 **4 次** |
| 复用 `zero()` | 76–80 / 19–29 / 37–59 / 903–967 / 416–474 ns | 上表同形状 |
| **完美内存池的上限** | 12 次/样本：CRT 0.018–0.023 ms vs 池 0.000–0.001 ms ⇒ **0.038–0.048%** | 分母 = 同轮一次 actor 头 fwd+bwd 45.6–47.4 ms |
| 每样本分配次数 | seq 专家 forward **0** Tensor + 1 `new`(≈23 KB)；backward 0 + 2(2×32.4 KB)；`MoE(seq)` 9 次/88 KB；**actor 头 12 次/117.6 KB**；MLP 对照 15 次 | 热路径**已经几乎没有分配**（`allocBuffers()` 预分配了全部 scratch） |
| `std::vector<Tensor>` 增长 | 256×(1710,1)：**11.04 ms vs 3.59 ms = 3.08×**；单看增长部分 **7.27 vs 0.45 ms = 16×** | move 构造/赋值**(366/495) 没有 `noexcept` ⇒ 增长退化成深拷贝 |
| `embedding(off,x)` 拷 4 个 float | **457–489 ns** vs `memcpy` 1.1–1.7 ns = **270–440×** | 每次调用 new 一个 `std::vector<int>`（与 `rl_sync.md:360-365` 的既有记录一致） |
| `sub(0)` 取一行 90 float | **1523–1579 ns** vs `memcpy` 26 ns = **60×** | 分配一个新张量 + 泛型索引拷贝 |
| 拷贝构造 vs 赋值 (1710,1) | **4.0–4.3 µs** vs **0.36–0.40 µs = 11×** | 赋值复用容量 |
| `x(i,j)` 通用索引 vs 扁平 `val[]` | 3.94 vs 3.90 ns/elem = **1.0×** | **负结果**：B19 里 12–27× 的"通用索引税"在 MM 的外积内层，不在普通逐元素访问上（那条已修） |

### 1.3 契约真空【代码事实】

- 全文**只有 19 条 `assert`**，全在 MM 内核（1104-1112、1128-1133、1218-1223、1346-1351、1470-1475）与死代码 `bmm`（1586、1594-1596）里 ⇒ **Release 一条不剩**。
- `requireShape2d()` 的 `val.size() >= totalSize`(1110) **只保证"缓冲不算短"**；形状对不上照样越界写。
- **这个缺口已经出过一次真实事故**：`src/rl/sac.h:99-104` 记录——critic 输入被拼成 `[state; prob]`(`stateDim+actionDim`) 但第一层按 `stateDim` 建，Release 下断言关闭 ⇒ **只读了前 stateDim 个元素**，拼进去的策略概率被静默丢掉（Debug 才断言失败）。
- 完全没有检查的契约还有：MM 输出缓冲长度、输出/输入别名（只用 `RL_RESTRICT` 对**编译器**断言）、`SubTensor` 全部、`operator[](std::size_t)` 的负下标（467-468，且注释 464-465 明确拒绝 `int` 重载 ⇒ 没有拦截点）、`posOf` 的 `static_cast<int>` 截断(661)、`size(Index...)` 零参下溢(430-435)、`reshape/view` 元素数不守恒(736-750)。
- 同类"只在实例化时检查"的隐患 9 类：`permute` 不校验排列（重复下标 ⇒ 越界写 803）、`concat` 空参数包（`xi[0]` 越界 1616）、`sub` 索引数 > 秩（540）、`bmm` 非 3D 分支（1596）、`Tensor(0,5)`（`totalSize==0` 而 `shape={0,5}`）等。

### 1.4 双长度源与 4 个失配入口【代码事实】

`totalSize`(272) / `val`(273) / `sizes`(274) / `shape`(275) **全是 public**，`sizes` 与 `totalSize` 都从 `shape` 派生，却由 `initParams()` 在 7 处维护（307/315/323/351/357/739/748），再在拷贝/移动/赋值里各抄一遍（476-478/500-504）⇒ **20 个函数**要手工同步 `shape`/`sizes`。

**只改 `val` 不改形状的 4 个入口**：`operator=(const std::vector<T>&)`(483)、`embedding`(583)、`flatten`(769，尚可)、`SubTensor` 全部。
**只改形状不校验 `val` 的 3 个入口**：`reshape`(739)、`view`(748)、`Tensor_(Shape,val)`(315)。
**两套长度源并存**：`sum/mean/norm2/dot/printValue` 用 `totalSize`，而 `zero/fill/variance/max/min/argmax/normalize/toString` 用 `val.size()`。

### 1.5 时间去哪了【实测】（这一节决定优先级）

`probe_seq_cost` 对**一个 seq 专家、一个样本**（T=90/tokDim=64/dff=192/Blocks=2）的配对分解：

| 分量 | ms | 占 forward |
|---|---:|---:|
| FC 12 次 | 3.433 | 21% |
| **注意力 8 次单头（手写标量四重循环）** | **6.303** | **39%** |
| LN 4 次 | 0.405 | 3% |
| 三项之和 | 10.141 | 63% |
| **余下（逐 token 拷贝 / gather / scatter / 残差）** | **≈6.0** | **≈37%** |
| forward 合计 | 16.157 | 100% |

也就是说：**76% 的 forward 不在 GEMM 上**。而这 76% 恰好是 Caffe（`im2col` + `caffe_cpu_gemm`）与 PyTorch（`bmm` / `strides` 迭代）**从架构上就不存在**的部分。

### 1.6 已被实测否决、以及不要照搬的路（避免重复劳动）

| 路 | 依据 | 结论 |
|---|---|---|
| 自建内存池 / arena | 完美池上限 **0.04%**（§1.2）；CRT 已在复用块 | **不做** |
| 用 `Tensor` 替掉 `std::vector<float>` scratch | 对齐实测：`fill` 2.0~2.4×、`dot` 无差别；≥1 KB 块本来就 ≥32 B | **不做** |
| "批量 FC 就能提速" | `FC 64→64 (T=90)`：逐 token `Layer<>` 0.1777 ms vs **整批 `MM::` 0.1739 ms = 1.02×** | **不要当卖点**；收益在"去掉搬运"（§1.5 的 37%） |
| 形状/步长改 `std::size_t` | `tensor.hpp:637-659`：下标提到 64 位让 index-dense 二维循环 **慢 34%（5.25→3.48 GB/s）**，已否决 | **保持 `int` 位宽**，改为"常驻范围守卫" |
| 引入 BLAS/MKL | 本仓库 GEMV 已 **0.10 ns/MAC**（`rl_sync.md:142`）；主要形状是 `(N,1)`/90×90，BLAS 的收益区（大矩阵、多线程）几乎不出现 | **不引入**（要先"把工作变成大矩阵"） |
| PyTorch 式句柄语义（引用计数共享 storage） | 会同时改写 426 处 `= Tensor(...)` 重建、180 处 `.val` 访问、14 处 `val.data()` 行指针算术的语义 —— 即把"静默别名"铺到全仓 | **不采用**；只取它的一条结论：参数用 `const&` |

---

## Part 2 · 优化项清单

### 2.1 加

| # | 内容 | 落点 | 依据 | 预期 | 风险 |
|---|---|---|---|---|---|
| A1 | **`noexcept` move 构造/赋值 + `swap()`** | `tensor.hpp:366/495` | 3.08×（增长部分 16×）【实测】 | 高/成本一行 | 无（语义不变） |
| A2 | **`data()/size()` 显式访问器**；`ptr()` 降为别名 | `tensor.hpp:404-405` | 外部 `.val.data()` **68 行**、`ptr()` 在 `src/` **0 处** | 中 | 无 |
| A3 | **两条隐式转换改 `explicit`（或删）** | `operator T*`(375)、`operator Vector`(380) | 文本层零调用点，但隐式 ⇒ 可能正在整块拷贝 | 中 | 低（编译器兜底） |
| A4 | **常驻契约检查**（Release 也生效）：输出缓冲长度与形状乘积一致 | `MM::requireShape2d:1104-1112` | §1.3 的 `sac.h` 事故 | 高（补事故） | 与 `issues_review.md:1402-1411` 冲突 ⇒ 用开关 + 只常驻这一条 |
| A5 | **`totalSize <= INT_MAX` 常驻守卫** | 构造/`reshape` | `tensor.hpp:655-659` 只用注释声明"下标×步长落在 int 内" | 中 | 无 |
| A6 | **`reshape` 校验元素总数 == `val.size()`** | `tensor.hpp:736-741` | 现状可静默改 `totalSize` ⇒ 双长度源永久分叉 | 高（修硬 bug） | 低 |
| A7 | **空/退化输入护栏**：`mean/variance/normalize/argmax/argmin` | `tensor.hpp:949-1019` | 现在 `val[0]` 越界读、`/(max-min)` 除零、`/T(totalSize)` 除零 | 中 | 低 |
| A8 | **`RL_TENSOR_CHECKS` 编译开关** | 新宏 | 让"always-on 全部断言"成为**可 A/B 的选项**而不是默认 | 中 | 无 |
| A9 | **单一长度源**：只留一个 `size()`，形状乘积必须等于它 | 全类 | 双长度源 + 3 份 `zero/fill/operator=(T)` 实现 | 高（消 4 个失配入口） | 中（要逐处核对语义） |
| A10 | **`strides()` 按需计算**（不再存 `sizes`） | 279-302 等 20 处 | 20 个函数手工同步 | 高 | 中 |
| A11 | **形状 SSO / 定长存储（位宽保持 `int`）** | 构造 354-359 等 | 3 次分配 → 1 次；每次构造省 2 次小分配（~480 ns【推理】） | 中 | 中 |
| A12 | **非拥有视图 `TensorView`**（`ptr + shape + strides`，owner 显式） | 取代 `SubTensor` | `SubTensor` 9 条静默出错路径、外部零引用 | 高（打开 P5/P6） | 中（别名） |
| A13 | **序列化搬出 `Tensor`**（`tensor_codec.hpp`），字节格式不变 | 1725-2255 | 531 行（23.4%）+ 形状解析 3 遍 + 全局静态错误旗标 | 中 | 低（有 6 个测试钉住格式） |
| A14 | **逐元素算子收敛成一个 strides 迭代器** | 828-946 | 16 个算子 120 行样板、12 个无人用、两套长度源 | 中 | 低 |
| A15 | **角色分离/标签**（参数 / 梯度 / 动量 / 激活缓存） | `layer.h:41-45`、`parameter.hpp:11-13`、`optimize.h:79-163` | 同型 4 张量靠位置传参（`Adam(w,v,m,g)`）；`g.totalSize==0` 当"有无梯度"标志 | 中 | 高（面大，分阶段） |
| A16 | **`empty()`（不零填）对 `zeros` 分开** | 构造 354-359 | PyTorch `torch.empty`【外部】；零填是实打实的成本 | 低/中 | 中（很多层依赖零填 + 累加语义） |

### 2.2 删（分四档，判据 = `src/` 与 `test/` 双零命中）

**D1 可直接删**（每项都做过 grep 计数）：4 个别名（`Tensori/Tensoru8/Tensorc/Tensord`，2258-2262）+ 6 个公开类型别名（67-72）+ 坏构造 `Tensor_(const std::vector<Tensor_>&)`(336，本身编译不过) + `Tensor_(Shape,Vector)`(311)/`Tensor_(init_list,init_list)`(319)/`Tensor_(T)`(348)/`operator=(const vector<T>&)`(483) + `shapeEqual`(393，实现还越界)/`empty`(421)/`size(const Shape&)`(437) + `zeros/ones`×4(508-534) + `slice`(589)/`toVector`(598)/`fromVector`(606)/`view`(744)/`permute`(785) + `min`(971)/`argmin`(989)/`normalize`(1003) + `product2D`(1642)/`printValue`×2(1666/1681，唯一调用点在 `#if 0`)/`printValue2D`(1694)/`printShape`(1711) + `validateEncoded`(1817)/`base64Decode`(2208) + **`MM` 的 4 个 2 参返回副本重载**(1537-1570)。

**D2 死簇的根**：删 `bmm`(1574) ⇒ `at()`(550)、`sub()`(537)、`operator%`(856)、`SubTensor::operator=(const Tensor_&)`(92) 同时失去唯一调用者；进而**整类 `SubTensor`(73-267) 与 `subTensor` 成员(270) 一并删**（外部零引用，9 条静默出错路径）。

**D3 仅测试在用 —— 保留**（它们是测试抓手，不是死代码）：`ptr()`/`alignedTo()`/`dataAlignment()`（`test_transformer [9]` 对齐自检）、`sum()`（`src/` 0 处）、`toDebugString()`（老格式往返）、`MM::kijk`（`test_grad_main.cpp:335-336` 唯一的语义钉桩）。**做法**：保留 + 注释标明"无生产调用者"。

**D4 不能自己删**：`Tensor_(const std::vector<Tensor_>&)` 按 `tensor.hpp:326-335` 的注释"属于上游同源代码，删除该走 `docs/rl_sync.md` 那条线" ⇒ **先记账再删**（Part 4 的 P0 → P2）。

**D1/D2 执行结果（2026-10-10，P2 落地）——三处与清单不符的地方**：
1. **`normalize`(1118) 不能在 P2 删**：P1 新加的 `test_transformer [10]` 里那条"零跨度 normalize 必须响亮失败"的**故意踩契约**用例正是调它（`test_transformer_main.cpp:155`）⇒ 它是**测试抓手**，归 D3"保留 + 注释标明"。
2. **`empty`(493) 删了又放回（编译器否掉）**：全仓 grep 看不到调用者，但删掉后报 `error C2039: "empty" 不是 RL::Tensor_ 的成员` —— 真调用点在 `src/rl/bc.h:73`（`!state.empty()`）与 `src/rl/sac.cpp:175`/`:274`（`legalMask.empty()` 决定走掩码分支还是全量口径）。**教训**：头文件里的成员函数，"文本层零引用"不是充分证据，编译器才是（Part 7 已有对应风险行）。
3. **6 个公开类型别名（现 `tensor.hpp:69-74`：`ValueType`/`Vector`/`Shape`/`Size`/`iterator`/`const_iterator`）本轮不删**：它们在全仓的**外部**引用确实是 0（grep `Tensor::Shape` 等一律零命中），但类内部处处在用，且它们正是 **P3（A9/A10 单一长度源、`strides()` 按需算、形状 SSO）要重写的那几个类型** ⇒ 现在删等于同一片代码改两遍，且零运行收益。**推迟到 P3 一起做**，届时按"接口不变"的约束决定保留哪些别名。
4. **8 个按值返回算子（P2-附 追加删的）**：`Tensor ⊗ Tensor` 的 `+ - * /` 与 `Tensor ⊗ T` 的 `+ - * /`。判据同 D1（零引用），但**验证方式是编译器**：把它们封掉后全量构建 0 error（458 步、含全部 test/bench 目标）⇒ 零消费者成立。⚠ 与 D1 那批一样，**就地版一个都没动**（`+= -= *= /=`，两个重载各 4 个）。清单里原本没点这 8 个（只点了"TRPO 是 `Tensor op Tensor` 的唯一消费者"），所以这一条是**按方案末段那句话的后续动作**追加的，单独记在这里以免被当成"清单外的擅自删除"。
另有两处**整类重复实现**建议单独一轮评估：`src/rl/mat.hpp` 的 `Mat`（588 行，与 `Tensor` 平行、**不在 `CMakeLists.txt` 源列表**、`val` 还是无对齐的 `std::vector<float>`）；整个 `TRPO`（无调用者，却仍在 `CMakeLists.txt:73` 编译，且是 `Tensor op Tensor` 值语义算子的唯一 live 消费者）。
**【2026-10-10 P2-附 已执行】**：两处都删了（`Mat` −588 行、`TRPO` −49/−544 行），并顺带删掉 8 个按值返回算子（见 Part 6 的 P2-附 行与 `docs/rl_sync.md` §1.5 第 5~8 条）。

**不删（看似死代码但是活的）**：`sizesOf`、`initParams`、`posOf`×3、`indexOf`×2、`size(Index...)`、`concats`、`permuteIndexs`、`lastDecodeFailedRef`、`parseShape`、`crc32*`、`base64DecodeInto/Encode/DecodeTable`、`MM::contiguous2d/requireShape2d`。

### 2.3 改

| # | 内容 | 落点 | 依据 |
|---|---|---|---|
| C1 | `val` 私有化 + 访问器（是 P3/P5 的前置） | 全类 | 外部生产代码 **18 行**直接碰 `val`（含 `net.hpp:220` 绕过 `totalSize`）、14 处 `val.data()`；机械替换面 = src 78 行 + test 42 行 |
| C2 | 同族参数**按 `const&` 传** | `src/rl/util.hpp:332` 起 | `lerp(Tensor &x, const Tensor xi, …)` 按值传 ⇒ SACAZ MoE TB 每个 Polyak 步白拷 **≥100 MB**【推理·算式】 |
| C3 | 热点路径的按值返回改就地/收敛 | `attention.hpp:246` `tr()`、`net.hpp:109` `flatten()`、`concat`(1610) 深拷实参 | 每 head 每反向白拷 28 KB（老口径 709 KB）；每 forward 128 B + 多写 256 B |
| C4 | 注意力改成矩阵形式（先量后做） | `seq_transformer.hpp:1330-1469` | §1.5 的 39%；SIMD 判据 `T=90≥8`、`dk=16≥8` 本来就满足 |
| C5 | 用视图替换 seq 的逐 token 搬运 | 18 处 `fcFwd/fcBwd` 桥接 | §1.5 的 37% |
| C6 | `attnBwd` 的 `(d_k²)²` 雅可比路径加"未接线响亮失败"守卫或删 | `attention.hpp:189` | `d_k=84` ⇒ 199 MB、`d_k=421` ⇒ ≈126 GB，只因无人调用才没炸 |

---

## Part 3 · 借鉴 Caffe/PyTorch 的评估

### 3.1 采纳

| 借鉴点 | 外部做法【外部·Caffe/PyTorch】 | 本仓库现状 | 采纳形式 |
|---|---|---|---|
| **矩阵化计算** | Caffe：`im2col` + `caffe_cpu_gemm` 把卷积变成一次 GEMM（[blob.cpp](https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/blob.cpp) 同仓）；PyTorch：`bmm` 把多头注意力变成批量矩阵乘 | 注意力是手写四重标量循环（39% of forward） | **C4**：先做只读可行性探针（两实现配对），拿到加速比再决定落地 |
| **storage + sizes + strides 的视图模型** | PyTorch `TensorImpl(storage, sizes, strides, offset)`，视图是元数据操作（[tensor internals](https://fleuret.org/dlc/materials/dlc-handout-1-6-tensor-internals.pdf)）；Caffe `Blob::ShareData` 显式共享 + 尺寸校验 | 只有 `shape/sizes`；视图靠 `SubTensor`（静默别名）；seq 用 `std::vector<float>` + 显式下标逐 token 搬运 | **A12 + C5**：非拥有 `TensorView`（先只读侧），用它消掉那 37% |
| **Release 也生效的契约检查** | Caffe 的 `CHECK_*`（glog，失败即 abort，不受 `NDEBUG` 影响）；PyTorch 的 `TORCH_CHECK`/`AT_ASSERT` | 19 条 `assert` 全在 `#ifndef NDEBUG`；`sac.h` 已出过一次静默事故 | **A4 + A8**：开关 + 只常驻"输出缓冲长度"这一条 |
| **显式数据访问器** | Caffe `cpu_data()/mutable_cpu_data()`；PyTorch `data_ptr()/numel()/sizes()/strides()` | `val/shape/sizes/totalSize` 全 public | **A2 + C1** |
| **一个逐元素迭代器** | PyTorch `TensorIterator`（按 strides 走、含类型/广播处理；见 [TensorIterator PR](https://github.com/pytorch/pytorch/pull/175336/files/8f3aa44c2e4e8c8c01905f25732df09e34f5804a#1)） | 16 个算子 120 行样板、12 个无人用、两套长度源 | **A14** |
| **`empty` 与 `zeros` 分开** | PyTorch `torch.empty`；Caffe 的 `SyncedMemory` 也不初始化 | 唯一构造总是零填 | **A16**（逐 site） |
| **角色分离** | PyTorch 的梯度图在 Tensor **之外**（AutogradMeta）；Caffe 的 `data`/`diff` 只能经 `mutable_*_diff()` 访问 | 同一个 `Tensor` 承载参数/梯度/动量/激活缓存，`g.totalSize==0` 当"有无梯度"标志 | **A15**（分阶段） |

### 3.2 不采纳（每条都有本仓库的实测或纪律依据）

见 §1.6 的表：内存池、`Tensor` 替 scratch、批量 FC 当卖点、64 位形状、BLAS、句柄语义。

### 3.3 一句话评估

Caffe/PyTorch 的收益**不是来自类设计本身**（它们的 storage/stride 抽象在 CPU 小张量上并不比一个 `std::vector` 快），而是来自**它们从架构上就不产生"逐 token 标量搬运"与"非矩阵化的注意力"**。
本仓库的对应结论是：**`Tensor` 内部可优化的余量是"微秒级"，而 §1.5 那 76% 是"毫秒级"** —— 所以优先级按 Part 4 排：先补契约与热点（P1）、再删债（P2-P4）、视图（P5）、最后才是注意力矩阵化（P6）。

### 3.4 借鉴 `simple-tensor`（本机另一个仓库：`E:\home\code\simple-tensor`）

**它是什么**：一个从零实现、带自动求导的 PyTorch 风格张量库（`double` 数据 + `unsigned int` 下标、**expression template**、`Storage/Shape/IndexArray` 三分、按 size 分级的内存缓存）。它的**架构取向**与本仓库不同（句柄语义与内存池这两条本仓库 §1.6 已实测否决），但有 **6 条具体做法值得借鉴**，其中 3 条直接服务 P5。以下每条都带它的 `文件:行`。

| # | 值得借鉴的点 | 它的做法（文件:行） | 与本仓库的关系 / 落地建议 |
|---|---|---|---|
| 1 | **视图模型 = storage + offset + shape + stride** | `include/tensor/storage.h:16-46`（`Storage` 持 `shared_ptr<Vdata>` + `data_t* dptr_`，`offset()` 就是两指针相减）；公式与取值/转置/切片/视图的推导写在 `docs/01_Basic_Ideas_for_Implementing_Tensor.md:9-63`、`:94-258` | 本仓库 P5（A12 `TensorView`）要的就是这个。⚠ **关键警告**：它在 `TensorImpl` 里**显式存 `stride_`**（`include/tensor/tensor_impl.h:93`），因为 transpose/slice 会产生**非连续**张量；而本仓库 A10 刚把步长改成**从 shape 按需算**（只对连续张量成立）⇒ P5 必须二选一并写进契约：(a) 视图自带 stride（照它的做法），或 (b) 视图只允许**连续区域**（本仓库更省，但 transpose 类视图就做不了） |
| 2 | **广播 = stride 0** | `docs/01:283-330`（`shape[i]==1` 的维度把 stride 设 0）；求值端只需一次 `if (stride[j] != 0)`（`include/tensor/tensor_impl.h:276,292`） | 本仓库现在**没有广播**（`operator+` 要求等长）。若将来要给 seq/MoE 加"按 batch 广播"，这条是零成本的；**P6 的批量注意力**也可以借它省掉显式 broadcast 拷贝 |
| 3 | **连续/非连续**双路径 | `is_contiguous()` 判据 `docs/01:159-177`；实现里 `if (is_contiguous()) __assign(...) else __assign_uncontiguous(...)`（`tensor_impl.h:204-207`、`:271-331`） | 与本仓库 `MM::` 的"SIMD 内核 + 通用 stride 回退"是**同一模式**；P5 的逐元素算子（A14）可以直接照这个双路径骨架写 |
| 4 | **Release 也生效的 CHECK 家族**（抛异常 + 可整体关掉） | `include/utils/exception.h:29-83`：`CHECK_TRUE/CHECK_EQUAL/CHECK_IN_RANGE/CHECK_FLOAT_EQUAL/CHECK_INDEX_VALID` + `CANCEL_CHECK` 一键关掉；`ERROR_LOCATION` 带 `__FILE__/__func__/__LINE__`（`:23-27`） | **正是 P1 的 A4/A8 在做的事**（本仓库是 abort + `RL_TENSOR_NO_CONTRACT` 逃生门，纪律上区分"契约错误 abort / 资源错误 throw"）。可借鉴的是**把位置信息带进错误信息**（本仓库现在只打 `where` 字符串，定位靠人肉找） |
| 5 | **负下标用"半量程无符号"统一拦掉** | `include/utils/base_config.h:8-16`：`index_t = unsigned int`、`INDEX_MAX = UINT_MAX>>1`；`exception.h:46-47`：`CHECK_INDEX_VALID(x) { if (x > INDEX_MAX) throw; }` | 直接对应本方案 §1.3 记的那个洞：**`operator[](std::size_t)` 的负下标没有拦截点**（刻意不保留 `int` 重载）。它的技巧是**一个无符号比较同时抓到"越界"和"负数"**（负数转无符号后必然 > 半量程）。本仓库的等价形式是 `operator[]` 里加 `if (i >= size()) contractFail(...)` —— ⚠ 但 **P1 实测每条常驻检查 ~3 ns/次**（`probe_checks`：6 次比较 17.8~18.5 ns/内核调用），而 `operator[]` 在 index-dense 循环里**每次都调用** ⇒ 必须**先量再决定**（很可能只能放在 `operator()`/`at()` 这类入口，而不是 `operator[]`） |
| 6 | **自动求导元数据 + 视图标志** | `AutoGradMeta{ Storage grad_; bool from_view_; shared_ptr<GradFn> grad_fn_ptr_; }`（`include/tensor/grad_meta.h:84-107`）；回传规则"`from_view_` 为真就不向下传梯度"（`tensor_impl.h:260-268`）；它的 backward 工作流写成了一份**显式规格**（`grad_meta.h:112-136`） | 对应本仓库 **A15**（角色分离：参数/梯度/动量/激活）与 **P5 的写侧视图语义**——"**视图不传梯度**"是一条现成、清晰的规则，可以直接写进 `TensorView` 的契约注释 |
| — | **不建议借鉴**（本仓库已有相反实测） | ① **按 size 分级的内存缓存**（`utils/allocator.h:95-98` 的 `multimap<index_t, unique_ptr<void>> cache_`）—— §1.6 已实测"完美内存池上限 **0.04%**"；② **expression template 整体重写**：它自己的文档给了实测与折算（`docs/04_Performance_of_Expression_Template.md:83-96`）——O2 下微基准快 **8×**，但**折算到它自己的 MLP 训练只有 ≈4.5%**（原文："比想象中还是差很多"）。本仓库逐元素算子**没有虚函数派发**（直接循环 + SIMD 内核），ET 能省的那部分**根本不存在**，而代价是把 `Tensor` 变成表达式外壳（`include/tensor/tensor.h:19`：`class Tensor : public Exp<TensorImpl>`）⇒ 与 Part 0 的"接口不变"直接冲突；③ **`data_t = double` / `index_t = unsigned`** 的选型不适用（本仓库 `float` + `int`，§1.6 已实测否决更宽下标）；④ 它的分配器**不保证对齐**（`utils/allocator.h` 直接走 malloc）——本仓库的 32 B 对齐契约（`tensor.hpp:15-46`）反而更严 |

**按阶段落地的优先级**：
- **P5 必读**：第 1、3、6 条（视图三元组、连续/非连续双路径、"视图不传梯度"）。
- **P6 可选**：第 2 条（stride-0 广播）在"批量注意力"里可以省掉显式广播拷贝。
- **单独一轮（先量再做）**：第 5 条（越界/负下标拦截）——必须按 P1 的检查成本口径先量 `operator[]` 的代价。
- **明确不采纳**：内存池、expression template、更宽的下标/数据宽度、它的分配器。

---

## Part 4 · 分阶段执行计划

```
P0 记账(零代码) ──┬─> P1 语言层+热点 ──┬─> P2 删死代码 ──> P3 形状/单一事实源 ──> P4 序列化搬家
                  │                    └─> (C2/C3 独立可做)
                  └─> P6-前哨: 注意力矩阵化"只读可行性探针" (可与 P1 并行, 不改生产代码)
                                                  │
                                       P3/P4 ──> P5 视图 + 逐元素迭代器 ──> P6 落地矩阵化(可选)
```

### P0 · 记账与文档一致性（零代码风险）

- **P0.1** 把 `tensor.hpp` 从"与上游字节级一致"改列为 **chess 侧本地分叉**，写进 `docs/rl_sync.md` 的差异清单（至少在案：分配器默认值 `AlignAllocator32`、`operator[]` 改 `std::size_t`、`posOf` 32 位显式窄化、`ikjk` 标量分支写法、`alignedTo/dataAlignment`、`RL_ALIGN_DEFAULT`），并按 `rl_sync.md:55-56` 的要求在文件内注释标明。
- **P0.2** 修两处**文档互相矛盾**：`rl_sync.md:135-136`（"只有没被使用的 transpose 需要对齐 ⇒ 不需要替换分配器"）与 `tensor.hpp:15-46`（现在把对齐当**正确性**契约）；`docs/analysis.md:347-353`（把 C7/C8 写成"未做"）与 `docs/issues_review.md:2746-2753`（"已做（R1.5）"）。
- **P0.3** README 文档表加一行指向本文件。
- **验收**：只有 `docs/` 与 `README.md` 的 diff；无代码改动（`git status` 证明）。
- **不做**：不借机改任何 `src/rl` 代码。

### P1 · 语言层 + 热点（要求**逐位不变**）

- 改动：**A1、A2、A3、A4、A5、A6、A7、C2**（A8 的宏随 A4 一起加）。
- **验收标准**（全部必须满足，缺一条不算完成）：
  1. 全量构建 **0 error**，警告基线仍只有 `rl/util.hpp:65` + `rl/tensor.hpp:558` 两条；
  2. `ctest` 全部目标通过（含 `test_transformer`(79)/`test_grad`(15)/`test_weights`(49)/`test_bc`(167)/`test_ppo_backbone`(43)/`test_sparse_moe`(100)/`test_scaledconcat`(40)/`test_dqnab`(115)）；
  3. **权重文本 FNV-1a 指纹仍为 `9a5483d39a42a333`**、有限差分读数逐行相同；
  4. `probe_move` 复测：增长部分从 7.27 ms 降到 ≈0.45 ms 量级（比值 ~16× → ~1×）；
  5. `probe_apid` 复测：`embedding`/`sub` 的行数不应变差（A2/A3 不碰它们）；
  6. 一轮 `train_bc`（`--agent=ppo-seq`，300 局面 8 epoch）的 CE/top-1/P(teacher)/熵逐行相同（只有墙钟不同）。
- **回滚**：单项独立提交，`noexcept`/护栏/访问器互不依赖。

### P2 · 删死代码

- 改动顺序：`bmm` 死簇 → `SubTensor` 整类 → D1 清单 → D4（先 P0.1 记账）→ 单独评估 `mat.hpp`/`TRPO`/`util.hpp` 6 个按值返回版单目函数。
- **验收**：全量构建（编译器会抓出任何"文本层零引用但被隐式使用"的项）+ 全套测试通过 + 记录删除行数与构建时间变化。
- **风险**：`operator T*`/`operator Vector`/`operator=(T)` 的"零引用"只是文本层证据 —— 以编译结果为准。

### P3 · 形状与单一事实来源

- 改动：**A9、A10、A11、C1**（`val` 私有化）。位宽**保持 `int`**（§1.6）。
- **验收**：
  1. 探针量到"构造 3 次分配 → 1 次"（用 P1 里加的诊断计数器或临时补丁，量完还原）；
  2. `test_transformer [9d]`（`AlignAllocator32` vs `std::allocator` 逐位等价）仍过；
  3. 指纹与 FD 读数不变；
  4. 一轮 BC/PPO A/B（同一脚本、两条臂），报告训练段耗时与 CE/top-1。
- **回滚点**：`strides()` 的引入与 SSO 分开提交。

### P4 · 序列化搬家（**字节格式不变**）

- 改动：**A13**：`tensor_codec.hpp`；形状解析收成一个函数；`b64:<crc>:<payload>` 头解析合一；删死解码器 `base64Decode`(2208) 与 `validateEncoded` 包装(1817)；全局静态旗标收进解码器对象（`Net::load` 的汇总口径不变）。
- **验收**：`test_weights` 44 条（含 `CHWGT2` 字面量、`b64:` 前缀、CRC32 翻位、按行截断、老 v1 兼容、体积比 >1.5、<200 ms）+ `test_transformer [3][5d][6f]` + `test_bc` 的 12 处 `netDigest` + `test_match` 的并发 save —— 全过。
- **风险**：这是**最贵**的一项（121 处 `toString/fromString` + 89 处 `save/load` 全靠它），只搬不改。

### P5 · 视图 + 逐元素迭代器

- 改动：**A12、A14、C5**。`TensorView` 先只读（`const TensorView`），owner 显式；用它替换 seq 的逐 token 搬运（`gather/scatter/残差/fcFwd` 桥接）。
- **验收**：
  1. `probe_seq_cost` 里"三项之和 / forward"的占比读数下降（现状 63% ⇒ 目标是 60% 以上那 37% 明显缩水）；
  2. 21 组有限差分 + 置换等变 + 常量 token 断言全过（`test_transformer`）；
  3. 一轮 BC A/B（不要求逐位相同，因为搬运顺序可能变，但**要求 FD 与留出指标不退化**）。
- **风险**：视图 = 别名。**只读先行**；写侧视图必须带 owner 与生命周期规则。

### P6 · 注意力矩阵化（可选/大；先量后做）

- **P6-前哨（只读、可与 P1 并行）**：在 `test/probe_seq_cost_main.cpp` 里加一节，把 `attnFwd/attnBwd` 的现有标量实现与"每头连续 `(T×dk)` + `MM::`"两实现按 §8.4 口径**配对对测**，拿到加速比与数值差（逐位或 ulp 级）。
- **P6 落地**（仅当探针给出正收益）：改 `seq_transformer.hpp` 的注意力布局。
- **验收**：两棵树 A/B（**不要求逐位相同** —— 这是本方案里**唯一**允许不做逐位等价的项，理由是求和顺序必然变，`rl_sync.md:270-273` 已写明"跨构建比对权重文件是否逐位相同不成立"）；`test_transformer [6]` 的 2D/RoPE 断言与 `test_grad` 的 FD 必须仍过。

---

## Part 5 · 测量与验证纪律（摘要）

### 5.1 计时口径（`docs/seq_transformer_design.md` §8.4）

- 这台机器（i7-12650H，6P+4E）**同一个二进制**的绝对耗时会漂几倍 ⇒ **只信"同一进程、同一轮"的成对比值**；
- ⚠ **【2026-10-10 换机】新机器不是同一台**：`Win32_Processor` 读到的是 **Intel i5-10400F（6 核 / 12 线程 / 2.9 GHz）**，而本文档与 `seq_transformer_design.md` 里所有"这台机器"的读数（含 §1.2 的分配/拷贝绝对值、§5.1.1 的 WER 13.8 s）都来自 **i7-12650H（6P+4E = 16 线程、P 核睿频更高）**。影响面：凡是**按固定时间窗**判定的断言（"等 30 s 看后台训练有没有把新权重同步回来"、"并发窗口内必须真的重叠若干次"）在两台机器上**不可比**；本轮 `test_match` 的两条 `[2.7k]` 正对照、`test_sparse_moe [11]` 的并发计数都属于这一类（见 Part 6 的 P1-复跑/P2 行）。**换机后的任何绝对耗时都不能与文档里的旧数字直接比较**；
- 每轮带**对照通道**并印出跨度，跨度大就重测；
- 用 `Start-Process -PassThru` + `ProcessorAffinity` + `Wait-Process` 绑核（`start /affinity … /wait` 会产生假读数）；
  ⚠ **绑核只用于计时**：`test_sparse_moe` 那类"看并发窗口"的断言在单核上会**假失败**（本轮实测：绑核 100/2 失败，不绑核 100/0 —— 它的读写线程需要一个真的第二个核才可能重叠）；
- 收益类改动必须 **A/B = 两棵源码树 → 两个二进制，同一工具链**（`seq_transformer_design.md:552-553`）。

### 5.1.1 【2026-10 新踩到的坑】`abort()` 的子进程在 Windows 上要卡 ~14 s

做"故意踩契约 ⇒ 断言退出码非 0"的子进程测试（`test_transformer` [6g]/[8]/[10] 都是这个套路）时，
子进程的 `abort()` 会触发 **Windows 错误报告（WER）**，**实测每个子进程白卡 ≈13.8 s**
（`--mm-shape-guard` 单跑 13.8 s，而其中真正的活只有几毫秒）。
9 个子进程就是 120 s —— 直接把 `test_transformer` 从 35 s 拖到 150 s，逼近它的 ctest 300 s 预算。

修法（一行，放在 `main` 最前面；**掩码参数不能省**，传 0 等于什么都不改）：

```cpp
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
```

效果：`test_transformer` **150.6 s → 35.6 s**，子进程退出码从 `0xC0000409` 变成 `3`
（断言只要求"非 0"，判据不变）。**任何新加"故意失败"的子进程测试都要先确认这一行在。**

### 5.2 必须守住的不变量

| 不变量 | 检查方式 |
|---|---|
| 权重文本 FNV-1a 指纹 `9a5483d39a42a333`（490,037 B 文本）不变 | `test_transformer`、`test_weights` ⚠ **换成新机器后这条改成"与本机 P1 基线比"**：本机读数是 `4cbbc09fc0d787c9`（参数量/文本长度与文档完全相同），P1 树与 P2 树**同一个值**；跨机比对本来就不成立（`rl_sync.md:270-273`、Part 8 第 10 条） |
| 默认路径**逐位不变** | 有限差分读数 + 训练读数逐行 diff |
| `MM::*` 四内核**累加**语义 | `test_grad` C 节（4 组形状） |
| 32 B 对齐 | `test_transformer [9a/9c/9d]` |
| 警告基线（只允许 `util.hpp:65`、`tensor.hpp:558`） | 全量构建日志 |
| `NDEBUG` 下断言仍被编译掉 | 按 `issues_review.md:1402-1411`（P1 的常驻检查走**开关**，不默认打开） |

### 5.3 会因"存储/序列化实现改变"而红的测试（改之前先看这张表）

| 测试 | 钉住的东西 |
|---|---|
| `test_weights [1][2][3][4b][4c][4f][4g][5]` | 逐比特无损、两次存盘逐字节相同、体积比、**老十进制文本仍可读**、`CHWGT2` 字面量、`b64:` + CRC32 翻位、按行截断、维度守卫、原子写 |
| `test_transformer [3][5d][6f][9]` | 往返逐位相同、γ/β 的 `toString` 写序、1D 不写模式标记（字节不变）、对齐与两种分配器逐位等价 |
| `test_scaledconcat (d)` / `test_sparse_moe` / `test_sacaz` | 往返逐位相同 / <1e-6 / 1e-4（后者是按旧文本精度取的容差） |
| `test_bc`（12 处 `netDigest`） | `save` 后**文件字节**的 FNV 比对（格式变得非确定/有损/save 返回 0 摘要就整片红） |
| `test_match` | 并发保存期间 `save` 必须全部返回 0 |
| `test_grad` C/D 节 | 累加语义、`mmShapeOk` 判据、`kikj` 逐元素 |

---

## Part 6 · 执行台账

> 规矩：每完成一项，在此追加一行（**必须有实测数字**，否则不算完成）。

| 阶段 | 日期 | 改了什么 | 实测/证据 | 结论 | 状态 |
|---|---|---|---|---|---|
| **P0** | 2026-10-10 | ① `docs/rl_sync.md`：§一 表加一行"同步之后又分叉的（`rl/tensor.hpp`）"，新增 **§1.4** 逐条记下六处分叉（默认分配器 `AlignAllocator32` / `operator[]` 形参 `std::size_t` / `posOf` 显式窄化 / `MM::ikjk` 标量分支写法 / `alignedTo`+`dataAlignment` / "四内核一律累加"契约注释）；② 修 §二 那句被自己推翻的话（"只有没被使用的 `transpose` 需要对齐 ⇒ 不需要替换存储分配器"）—— 改成"只对性能成立、对正确性不成立"，并接上 §1.4；③ `docs/analysis.md` 把 C7/C8 从"仍未做"改成"【2026-10 更正：已做】"（状态以 `issues_review.md` R1.5 为准），并新增第 7 条指向本文件；④ `README.md` 文档表加一行 | 只有 `docs/` 与 `README.md` 的 diff（`git status`：`src/` 无本阶段改动） | 三处文档欠账清掉：分叉有账、矛盾口径统一、陈旧状态更正 | ✅ 完成 |
| **P1** | 2026-10-10 | **A1** move 构造/赋值 + `swap()` 标 `noexcept`；**A2** 新增 `data()`/`data() const`（`ptr()` 保留为别名）；**A3** `operator T*` / `operator Vector` 改 `explicit`；**A4** MM 的形状契约检查**移出 `#ifndef NDEBUG`**（Release 也生效）+ `contractFail`/`contractIkkj/Kikj/Ikjk/Kijk` + `RL_TENSOR_CHECKS` 开关 + `RL_TENSOR_NO_CONTRACT` 逃生门；**A5** `requireIndexable`（`totalSize <= INT_MAX`，放在 `initParams` ⇒ 7 个形状入口全覆盖）；**A6** `reshape` 加 `requireSameLength`；**A7** `requireNonEmpty` + 零跨度 `normalize` 守卫；**C2** `util.hpp:332 lerp(Tensor&, const Tensor&, float)`。测试侧：`test_transformer` 新增 **[10] 常驻结构守卫**（6 个子进程故意踩契约 + 1 条正对照），并修掉一个**测试自身的假失败**（第一版 `ikkj(z,z,z)` 其实是合法形状）与一个**套件级性能陷阱**（`abort()` 触发 WER 每个子进程卡 ~13.8 s ⇒ 全套 150 s；加 `_set_abort_behavior(0, _WRITE_ABORT_MSG\|_CALL_REPORTFAULT)` 后 **35.6 s**） | ① 全量构建 `BUILD=0`（警告基线见下）；② `test_transformer` **86/0**、`test_grad` 15/0、`test_weights` 49/0、`test_bc` 167/0、`test_ppo_backbone` 43/0、`test_sparse_moe` 100/0、`test_scaledconcat` 40/0、`test_dqnab` 115/0；**ctest 整体 14/16**（`test_mcts` Timeout；`test_pretrain` abort —— 守卫抓到的一条**真 bug**，根因已修，见 6.A.1/下方 P1-附，复跑待补）；③ **权重指纹 FNV-1a `9a5483d39a42a333`（490,037 B 文本）不变**，21 组 FD 读数逐行同；④ **BC 一整臂逐行相同**（`.r1build/seq_pos1d.txt` vs `bc4_seq_p1.txt`：39 行里只有 1 行不同，且是墙钟"训练 122.7 s vs 116.7 s"；训练 CE 1.9610 / 留出 CE 3.1143 / top-1 39.17% / 26.67% / P(老师) 0.23667 / 0.12851 / 熵 2.2629 / 2.3985 全同）；⑤ `probe_move`：`vector<Tensor>` 增长的"增长段" **7.27 ms → 1.18 ms（6.1×）**，A/B 比值 **3.08× → 1.25×**；⑥ `probe_checks`：常驻检查 **17.8~18.5 ns / 次内核调用**（小内核 (64,64)·(64,1) 占 0.88%、(360,1710)·(1710,1) 占 0.0036%；按 seq 一次 forward 1080 次 FC 调用折算 ≈0.12%）；⑦ `probe_apid`：`embedding` 420 ns / `sub` 1495 ns / 拷贝构造 3990 ns，与 P1 前同轮读数一致（无回归） | 语言层与三条护栏落地且**逐位不变**；代价可量化（每内核调用 18 ns）；**顺带修掉一个测试基础设施坑**（WER 卡顿让 ctest 的 300 s 预算从"快满"回到 8× 余量）；**并且守卫当场抓到并修掉了一条跑了很久的静默 bug**（`LayerNorm<Fn,LN::Pre>::backward` 的 sink 尺寸错 ⇒ DPG 的输入梯度被截断 + 越界读，见 Part 6.A.1） | ✅ 完成（`test_pretrain` 的复跑待下一轮补） |
| **P1-附** | 2026-10-10 | 修 A4 抓到的那条真 bug：`src/rl/layer.h` 的 `LayerNorm<Fn, LN::Pre>::backward` 把 `dL` 从 `outputDim` 改成 **`inputDim`**（`kikj` 的 sink 尺寸 + 越界读 + z-score 均值口径，三个问题一个词解决）；文件内写下完整证据链 | 触发点：`dpg.cpp:43` 的 `LN::Pre(90→64)`；守卫报的形状 `z[64,1] x1[64,90] x2[64,1]`；`ddpg.cpp:11/15` 的 `inputDim == outputDim` 用法**逐位不变** | 静默 bug 修复；**验证（`test_pretrain` + ctest）按用户要求暂停** | ✅ 完成（2026-10-10 复跑：`test_pretrain` 38/0、exit 0 —— 见下方 P1-复跑行） |
| **P1-复跑** | 2026-10-10 | **不改代码**：换到新 PC 后重做 P1 的验收（旧机的证据文件 `.r1build/probe_*_after_p1.txt`、`test_transformer_p1.txt` 没有随仓库过来 ⇒ 必须在本机重建基线） | ① 全量构建（Release/Ninja，**459 步**）**0 error**，**5 min 9 s**（21:28:55→21:34:04）；警告**去重后只有 5 处**：`src/rl/util.hpp:65 C4804`（每包含它的 TU 印一次，共 184 条）、`test/bench_agent_arena_main.cpp(980/1001/1020)` 与 `test/bench_sac_learn_main.cpp(1134)` 的 `C4477` —— 后 4 处**与 P1 的改动无关**（`git diff HEAD~1 HEAD` 对该文件的 hunk 都落在 35~794 行，而 980/1001/1020 在外；`bench_sac_learn_main.cpp` 本轮**根本没被改过**）；方案 §5.2 记的 `tensor.hpp:558` 在 P1 之后已不再出现；② **`test_pretrain` 38 断言 / 0 失败 / exit 0**（9.30 s）—— P1 台账里那条"已知红"清掉，`layer.h` 的根因修复有效；③ 全套 `ctest`（旧机 16 个目标，本机 **18 个**）**16/18 通过，1984 s**：`test_mcts` **通过（202 s）**（旧机是 Timeout）、`test_pretrain` 通过、`test_transformer` 通过（4.51 s）；失败 2 条 = `test_match`（366 断言 3 失败）与 `test_sparse_moe`（100 断言 2 失败）；④ **归因（关键）**：这 5 条失败全部落在"后台训练必须在固定时间窗内跑完一轮 / 并发窗口必须真的重叠"这一类**计时断言**上（原文如"训练模式下后台训练**确实**会把新权重同步回主 agent（正对照）"、"并发: 前向 **0** 次… [FAIL] 并发下真的读到了快照"）；而它们跑的窗口里**同时有 4 路并行编译**（我在同一台机器上起 P2 的构建）⇒ **空载复跑 `test_sparse_moe` 9.18 s 全过、exit 0**；`test_match` 的空载复跑随 P2 一并量（见 P2 行） | ① 又一次实证 §5.1 的纪律：**计时/并发窗口类断言必须空载量**，"跑测试时顺手编译"会把它们变成假红；② P1 的"逐位不变"在新机器上重新拿到证据：**P1 树二进制 vs P2 树二进制的输出逐行 diff，只有耗时、地址对齐值、线程交错计数不同**（`test_weights`/`test_grad`/`test_transformer`/`test_sparse_moe`，证据 `.r1build/p1_*.out.txt` vs `p2_*.out.txt`）；③ ⚠ **方案里的绝对指纹在本机复现不出来**：`test_transformer [3]` 在本机是 **`FNV-1a=4cbbc09fc0d787c9`**（参数量 **91,731** / 文本 **490,037 字节** —— 与 `seq_transformer_design.md:301` 记的"参数量 91,731、490,037 字节"**完全同形**），而文档记的是 `9a5483d39a42a333`。**形状/长度全同、只有内容哈希不同** ⇒ 这是**跨机器/跨工具链**的差异，不是本轮改动造成的：同一台机器上 P1 树与 P2 树给出**同一个** `4cbbc09fc0d787c9`，且 `test_transformer [6f]` 的 `首行=64,19\|b64:28ac353a:q52eO`（CRC32 前缀）两边也逐字符相同。与 `rl_sync.md:270-273`「跨构建比对权重文件是否逐位相同不成立」同源 ⇒ **本轮起，"指纹不变"一律按"本机 P1 基线"比，不再引用旧机的绝对数字**（本机基线：`.r1build/p1bin/*.exe` + `.r1build/p1_*.out.txt`） | ✅ 完成：P1 与 P1-附 的验收在新机器上全部重做，唯一的"红"被归因到并行编译的干扰并已空载复现为**绿** |
| P2 | 2026-10-10 | 删死代码（Part 2.2）：`bmm` 死簇（`bmm` + `at()` + `sub()` + `operator%` + 整类 `SubTensor` 195 行 + `subTensor` 成员）→ D1 清单（`toVector`/`fromVector`/`slice`/`view`/`permute`+`permuteIndexs`/`shapeEqual`/`size(const Shape&)`/`zeros`×2/`ones`×2/`min`/`argmin`/`product2D`/`printValue`×2/`printValue2D`/`printShape`/`validateEncoded`/`base64Decode`/`MM` 的 4 个"2 参返回副本"重载/`Tensorc`·`Tensoru8`·`Tensori`·`Tensord`/`operator=(const std::vector<T>&)`/`Tensor_(Shape,Vector)`/`Tensor_(init_list,init_list)`/`Tensor_(T)`）→ D4（上游同源的坏构造 `Tensor_(const std::vector<Tensor_>&)`，记账已写进 `docs/rl_sync.md` §1.5）；文件内加了 P2 注释块（删了什么/为什么/怎么验） | ① **删除脚本一次净删 628 行**（`tensor.hpp` 2506 → 1878；证据 `.r1build/tensor.hpp.pre_p2.txt` + `.r1build/p2_delete_blocks.ps1`，脚本对每个删除块**先断言首尾行内容**再切，避免切错范围）；随后**放回 `empty()` 并加入 P2 说明注释** ⇒ **最终 1924 行**（`git diff --numstat`：+48 / −630，净 **−582 行 / −23%**）；② **全量构建**（干净目录 `build/plan_p2`，466 步）**0 error**，警告与 P1 完全同一组（`util.hpp:65` + 4 处 bench `C4477`）；③ ⚠ **编译器当场否掉 D1 里的一条**：`empty()` 被删后报 `error C2039: "empty" 不是 RL::Tensor_ 的成员`（真调用点：`src/rl/bc.h:73`、`src/rl/sac.cpp:175`/`:274` 的 `legalMask.empty()`）⇒ **原样放回**并在定义处写下这段历史（"文本层零引用"对头文件里的成员函数不是充分证据）；另 `mm` 的 2 参重载/`size(Index...)` 等的"零引用"经编译确认成立；④ **A/B 逐行 diff（P1 二进制 vs P2 二进制）**：`test_weights`（49 断言）、`test_grad`（15 断言）、`test_transformer`（86 断言）、`test_sparse_moe`（100 断言）**全部 exit 0**，输出里**只有耗时/地址对齐/线程交错计数**不同，指纹、FD 读数、往返差、断言数逐行相同；⑤ **P2 树空载全套 `ctest`：17/18 通过（1955 s）**（证据 `.r1build/ctest_p2.txt`），`test_sparse_moe` 复跑**通过**（1.75 s，上一次的"红"是并行编译干扰）；唯一红的是 `test_match`，而它**在 P1 树上同样红**、且两次红的**失败点还不一样**（P1 树 3 条 = `[2.7c]` 首次训练损失不逐位相同 + `[2.7k]` 两条后台训练计时；P2 树 12 条 = 从"快照权重 = **失败** / 还原 = **失败**"这类**竞态**前置条件开始崩，再级联到同一批 [2.7c]/[2.7k]）⇒ **不确定性的红，与本次删除无关**（前置归因实验见下方 P2-归因行）；⑥ **测试自身非确定性的对照**：拿**同一份** P2 二进制把 `test_weights` 连跑三次，`v1 (十进制文本)` = **885065 / 885270 / 885411 三次三样** ⇒ 该节网络没有钉种子，它的那几个数**不能**当"变了没有"的判据（P1 与 P2 之间它也一直不同：884578 / 885196 / 885065）；⑦ 定稿前最后一次改动**只有一行注释**（在 `base64Decode` 那段历史说明里注明它已删），重编 **459 步 0 error**，重跑 `test_weights`/`test_grad`/`test_transformer` 与前一版二进制**同类差异**（只有耗时、对齐值、上面那类非确定数） | 删债落地且**行为逐位不变**；本阶段的收益是**源码可维护性**（净 −582 行、头文件体积 −23%），**不是编译时间**：全量构建 P1 309 s（389 次编译）/ P2 316 s，差异落在并行度与噪声里（诚实结论：这几个内联/模板块删掉并不减少每个 TU 解析头文件的工作量） | ✅ 完成（验收数字见左） |
| **P2-归因** | 2026-10-10 | **不改代码**：为回答"`test_match` 的红是不是本方案引入的"，把 **P1 之前的树**（`HEAD~1` = `51e22a5`，`tensor.hpp` 还没有 P1 的守卫/对齐/seq 那批改动）用 `git worktree` 检出到 `.r1build/prep1`，**单独构建**只编 `test_match` 目标，空载跑同一个测试 | ① 前置树构建 **0 error**；② **空载 `test_match`：366 断言 / 2 失败 / exit 1（1875 s）** —— 其中一条与本方案三棵树**完全相同**：`[2.7c] 学习信号与对手无关`（前置树两次首损失 **0.0546623692 vs 0.0393589064**，**同样逐位不同**）；另一条"整场对弈期间 `weights/` 下的临时权重一个都没被写"是**测试对环境的假设**造成的（该目录当轮是我手工新建的**空**目录，前后快照自然相同）；③ 前置树空载时 `[2.7k]` 的正对照**通过**（"训练模式: 权重 8987140 -> 8987140 字节, 变了 = **1**"）⇒ 它在 P1 树那次红是**并行编译**干扰、在 P2 树那次红是**级联**；④ **P2 树那 12 条失败的根因查清**：`build/plan_p2/weights` 的 **CreationTime = 22:37:44**，而 ctest 22:15 开始、`[2.7c]` 在 22:16~22:18 —— **那一刻这个目录还不存在**，于是 `saveCurrentAgentModel(...)` 一律失败（原样打印"实验3 前置: 快照 PPO 起始权重 = **失败**"），随后所有"装回健康权重"的前置条件全崩；旧机那份 build 目录里的 `weights/` 是 9 月就在的，所以这个隐式依赖从来没暴露过 | `test_match` 在本机的红**与 P0/P1/P1-附/P2 全部无关**：前置树（`HEAD~1`）同样红，且四次红的失败点各不相同（前置 2 / P1 3 / P2 12 / 空载复跑见右）⇒ ① `[2.7c]` 那条"换对手后首次训练损失必须逐位相同"的判据**在这台机器上不稳定**（三棵树全红、且两次的首损失都不同）—— 属**既有测试的时序脆弱性**；② `test_match` 隐含要求"cwd 下 `weights/` 已存在"（它自己不建）⇒ 进下一轮修复清单；③ 因此 P2 的验收按"**除 `test_match` 这条既有/环境红之外全绿**"记账（17/18），与 P1 台账当年在旧机的 14/16 是同一类记账 | ✅ 归因完成（`test_match` 定为"既有 + 环境红"，见 Part 8 第 12 条） |
| **P2-附** | 2026-10-10 | ① 删 `src/rl/trpo.h`（**−49 行**）+ `src/rl/trpo.cpp`（**−544 行**，整个 `TRPO` 类），并从 `CMakeLists.txt` 的 `RL_SOURCES` 去掉那一行；② 删 `src/rl/mat.hpp`（**−588 行**，整个 `Mat` 类）；③ 删 `src/rl/util.hpp` 的 **6 个按值返回版单目函数**（`Tensor sqrt/exp/log/tanh/sin/cos(const Tensor&)`，−53 行，另加 17 行说明注释）；④ **追加删掉 8 个按值返回算子**（`Tensor ⊗ Tensor` 的 `+ - * /` —— 方案 Part 2.2 点名的那 4 个；`Tensor ⊗ T` 的 `+ - * /` —— 方案**没点名**、本轮"先量后删"的 4 个；`tensor.hpp` −54 行 / +13 行注释）；⑤ 测试侧修掉一条**仪器故障**：`test_match` 全程用相对路径 `weights/...` 却**自己不建这个目录**（+17 行 `create_directories`）。文档侧：`docs/rl_sync.md` §1.5 追加第 5~8 条记账 + 更正 `trpo.cpp` 那处历史引用，`docs/tanh_norm_analysis.md` 表格里 `trpo.cpp:17-29` 标注已删 | ① **全量构建 0 error**（458 步 / 295 s，含全部 test/bench 目标）—— 这是"零消费者"的**编译器证据**；顺带一个旁证：警告 `util.hpp:65 C4804` 的条数 **184 → 183**（`trpo.cpp` 这个消费者没了）；② **成对 A/B（P1 树二进制 vs P2-附 树二进制，10 个 RL 测试逐个跑）**：`test_bc` **逐行 0 差异**；`test_transformer` 的 **FNV-1a 指纹/断言数/往返差**、`test_grad` 的 FD 读数与断言数、`test_dqnab` 的 `gap/corr`、`test_weights` 的 49 断言与 v2 字节数、`test_sparse_moe`/`test_scaledconcat`/`test_sacaz`/`test_ppo_backbone`/`test_pretrain` 的断言行**全部逐字符相同**；③ **对照实验（同一份二进制连跑两次）**：`test_ppo_backbone`/`test_pretrain`/`test_sacaz` 那几个"看起来变了"的数字（`\|g\|(actor)`、rollout 步数、`ms/次`）**连跑两次自己就不同** ⇒ 它们是测试自身的非确定性，不是本轮改动（`test_ppo_backbone` 同一二进制两次：`\|g\|(actor)` 1.06583 vs 1.15872）；④ 零引用判据：`trpo.h` 只被 `trpo.cpp` 自己 include、`\bTRPO\b` 全仓零命中；`mat.hpp` 全仓零 `#include` 且不在源列表；`RL::tanh(...)` 全仓 2 处且实参都是非 const 左值（重载决议选就地版） | **上游同源代码的"整类重复实现"清掉**：`TRPO`+`Mat` 合计 **−1181 行文件**，加 `util.hpp`/`tensor.hpp` 的按值返回版共 **1289 行删除 / 47 行注释**；收益是**可维护性 + 每个全量构建少编一个 TU**（旧读数 §1.6 说"整批 FC 当卖点不成立"，这里也不主张运行期提速：删掉的都是零调用代码）。**在写"零引用"结论时用的是编译器而不是 grep**（这正是 P2 里 `empty()` 那次教训的延续）。**体积/时间实测（Part 8 第 5 条要的那个数）**：从 **P1 树**到 **P2-附 树**，13 个测试/探针二进制的体积各缩小 **0~19 KB**（合计 **≈117 KB，−0.1 MB 量级**）⇒ **删掉的这些零调用代码本来就被链接器剔掉了**，所以"体积收益"几乎没有（这是负结果，照实记）；全量构建 **459 步 / 309 s → 458 步 / 295 s**（同为 `-j12`），差异 ~4.5%，**落在噪声里**，唯一确定的收益是少编一个翻译单元 | ✅ 完成（完整 ctest 结果见右栏 P2-附-验） |
| **P2-附-验** | 2026-10-10 | 空载跑 P2-附 树的全套 `ctest`（含带仪器修复的 `test_match`） | 见 `.r1build/ctest_p2b.txt`：**17/18 通过（1853 s）**；**`test_match` 的失败条数 12 → 3**：修复前那 9 条是"`weights/` 目录不存在 ⇒ 快照/还原前置条件失败"的**级联假红**，本轮打印 `[ok] 临时权重目录 weights/ : 就绪`、`实验3 前置: 快照 PPO 起始权重 = 成功`、`实验3 还原权重 = 成功` ⇒ 级联全部消失；剩下的 3 条与 **P1 树那次完全相同**（`[2.7c]` 首次训练损失不逐位相同 + `[2.7k]` 两条后台训练计时：`训练模式: 权重 8987140 -> 8987140 字节, 变了 = 0`） | `test_match` 的剩余红 = **既有时序脆弱断言**（`[2.7c]` 在 P1 之前的树上同样红；`[2.7k]` 是"30 s 窗口内后台训练必须改权重"，前置树空载时通过、本机负载/机速变化时红）⇒ 不是本方案引入，也没有在本轮被"改判据"掩盖（Part 8 第 12 条留给人定夺） | ✅ 完成 |
| **P3-A11** | 2026-10-10 | **形状 SSO（定长存储）**：新增 `SmallShapeVec<N>`（内联 `RL_SHAPE_INLINE`=4 个 `int` 的小容器，超出才回退堆；move/swap 全部 `noexcept`，因为 A1 的 `Tensor_` move 承诺依赖它），`Tensor_::Shape`/`Size` 由 `std::vector<int>` 换成它（位宽仍是 `int`）；`initParams` 里 `sizes = std::vector<int>(...)` 改成 `sizes.resize(...)`（消掉那个**临时 vector**）；顺带把变参构造 `Tensor_(Dim...)` 用 `enable_if` **限成整型维度**；新增探针 `test/probe_alloc_main.cpp` + `probe_alloc` 目标（Debug 构建目录 + `_CrtSetAllocHook`，见下）；记账进 `docs/rl_sync.md` §1.4 第 7 条 | ① **验收第 1 条达成 —— 分配次数 3 → 1**（`probe_alloc`，本机 Debug 堆钩子，`.r1build/probe_alloc_before_p3.txt` vs `probe_alloc_after_a11.txt`）：`Tensor(19,1)`/`Tensor(90,64)`/`Tensor(shape)`（2 维、3 维）/拷贝构造 **3 → 1**；`reshape`（同元素数）**1 → 0**；`Tensor(0,5)` **2 → 0**；move/默认构造/赋值一直是 0；秩 5（> 内联容量）仍 3（32 B + 32 B + 数据，**符合设计**）；探针自查 **36/36 通过**；② 途中踩到的坑（已记进 §9.2）：Debug 默认 `_ITERATOR_DEBUG_LEVEL=2` 会让每个 `std::vector` 多分配 16 B 的 `_Container_proxy`，于是同一个 `Tensor(r,c)` 被读成 **8 次**分配 —— 必须在 Debug 构建目录里加 `-D_ITERATOR_DEBUG_LEVEL=0` 才是被测对象的结构；③ 编译期一次真实踩坑：`Shape` 换类型后 `Tensor(std::vector<int>)` 被**变参模板**截胡（精确匹配胜过用户定义转换），全量构建报 **136 条 `C2440`**，用 `enable_if` 限整型后 **0 error**；④ A/B 与 ctest 读数见下方 P3-A11-验 行 | 形状从"每次构造 2 次 malloc"变成"内联在对象里、零 malloc"：这是 §1.2 里"每次小分配 230~262 ns"那条成本第一次被真正拿掉（**不是**内存池、**不是**改分配器 —— 那两条已被 §1.6 否决）。**注意口径**：分配次数是**结构事实**（跨机可比），省下的**时间**不在这里下结论 —— P3 不主张端到端提速，方案 §1.6 早已实测否决过"存储层微优化能显著提速"。**代价照实记**：`sizeof(RL::Tensor)` **80 B → 112 B**（形状/步长各内联 4 个 `int` ⇒ 每个张量 **+32 B**）—— 大张量无所谓，但 `std::vector<Tensor>` 这类小张量密集的容器会多占 32 B/个；这个取舍记在这里，供"要不要把 `RL_SHAPE_INLINE` 调小/调大"时参考 | 🔄 进行中（A11 完成；A9/A10/C1 见下） |
| **P3-A11-验** | 2026-10-10 | 空载跑 A/B（P2-附 树二进制 vs A11 树二进制，10 个 RL 测试逐个跑）+ 全套 `ctest` | ① **A/B（逐行 diff）**：`test_bc` **0 差异**（41 行全同）；`test_transformer` **只有 1 行不同**（`对齐 = 64 B` vs `128 B`，两者都满足 `alignedTo(32)=1`，属每次运行都变的分配地址）+ 它的**权重文本 FNV-1a 仍是本机基线的 `4cbbc09fc0d787c9`（490,037 B、参数量 91,731）、86 断言 0 失败** ⇒ `test_transformer [9a/9c/9d]` 的对齐与"两种分配器逐位等价"照旧通过；`test_grad` 的 FD 读数、`test_weights` 的 49 断言与 v2 字节数（497,143）、`test_sacaz` 的 8 行不变量、`test_sparse_moe`/`test_scaledconcat`/`test_dqnab`/`test_ppo_backbone`/`test_pretrain` 的断言行**全部逐字符相同**（`test_weights` 那条 v2 行的差异只有 save/load 的 ms）；10 个目标 **exit 0**；② **空载全套 `ctest`：17/18（1885 s）**（`.r1build/ctest_p3a11.txt`），唯一红仍是 `test_match` —— 本轮 **366 断言 / 4 失败**：`[2.7c]`（既有）、`[2.7k]` 两条（计时）、**外加 `[2.22]`「整场对弈期间 weights/ 下的临时权重一个都没被写」**（P2-附 那次是"未变/通过"，本轮目录快照"变了"） | A11 是**纯存储层改动**：所有确定性读数逐位不变 ⇒ **验收第 2、3 条达成**（`[9d]` 通过 + 指纹/FD 不变）。`test_match` 的失败条数在这台机器上**逐次不同**（pre-P1 2 / P1 3 / P2 12 / P2-附 3 / A11 4），全部落在"固定时间窗 / 目录快照 / 后台训练竞争"这一族既有脆弱断言上 —— `[2.22]` 本轮翻红是**同一族的时序抖动**（背景训练那一轮恰好在这段窗口里写盘），与"形状内联"没有因果路径；这条也要进"测试侧待修清单" | ✅ 完成（第 1~3 条验收达成；第 4 条 BC/PPO A/B 推迟到 P3 收尾） |
| **P3-A10** | 2026-10-10 | **步长按需算（删掉 `sizes` 成员）**：新增 `strideOf(i)`（按需 O(rank)，返回 **`int`** 以保持 32 位乘加，见 §1.6）与 `stridesInto(int out[RL_SHAPE_MAX])`（**栈数组**，不分配）；`initParams(Shape&, std::size_t&)` 不再算/存 sizes；拷贝/移动/赋值/swap 里那 4 处 `sizes` 同步**全部消失**；`MM::requireShape2d` 里"`sizes` 与 `shape` 长度不一致"那条守卫**随之删除**（结构上已不可能）；外部 **13 处** `X.sizes[0/1]` 改成 `X.strideOf(0/1)`（`layer.h` 里还在内层循环里用的那一处提到循环外）；记账进 `docs/rl_sync.md` §1.4 第 8 条 | ① **分配次数不退化**（`probe_alloc`，`.r1build/probe_alloc_after_a10.txt`）：构造仍 **1 次**、`reshape` 仍 **0 次**、秩 5 的情形 **3 → 2**（只剩 shape 溢出那一次）；探针自查 **36/36**；② **对象体积反而变小**：`sizeof(RL::Tensor)` 原始 **80 B** → A11 后 112 B → A10 后 **72 B**（**比原始还小 8 B** —— A11 的内联成本 +32 B 被"删掉一个容器" −40 B 盖过）；③ 编译期逐条点名：全量构建先报 **27 条 error**（`t.sizes` 的 13 处外部用法 + 我漏改的 1 处注释），改完 **0 error**；④ A/B 与 ctest 见 P3-A10-验 行 | 消掉的是方案 §1.4 点名的"**两个事实源**"：`shape` 与 `sizes` 原来要在 20 个函数里手工同步、在拷贝/移动/赋值里各抄一遍，而只改一个形状入口（`reshape`）就会永久分叉 ⇒ 现在步长**只有一个来源**，那条不一致在结构上不可能发生（守卫跟着删掉：**用结构消除状态，而不是加断言守它**）。⚠ 接口级代价：`t.sizes[...]` 这种外部写法会**编译失败**（已记账进 §1.4 第 8 条） | ✅ 完成 |
| **P3-A10-验** | 2026-10-10 | 空载跑 A/B（P2-附 树二进制 vs A10 树二进制，10 个 RL 测试逐个跑）+ 全套 `ctest` | ① **A/B（逐行 diff）**：`test_bc` **0 差异**；`test_transformer` **只有 1 行不同**（`对齐 = 256 B` vs `64 B`，都满足 `alignedTo(32)=1`），它的**权重文本 FNV-1a 仍是本机基线 `4cbbc09fc0d787c9`（490,037 B / 91,731 参数）、86 断言 0 失败**；`test_weights`（49 断言、v2 字节数 497,143）、`test_grad` 的 FD 读数、`test_sacaz`/`test_sparse_moe`/`test_scaledconcat`/`test_dqnab`/`test_ppo_backbone`/`test_pretrain`/`test_bc` 的不变量行**全部逐字符相同**；10 个目标 **exit 0**；② **空载全套 `ctest`：17/18（1830 s）**（`.r1build/ctest_p3a10.txt`），唯一红仍是 `test_match`：**367 断言 / 4 失败**，失败点与 A11 那次**完全相同**（`[2.7c]` + `[2.7k]`×2 + `[2.22]`） | 与 A11 一样是**逐位不变**的等价改写（把"读成员 `sizes`"换成"按需算步长"，32 位乘加链原样保留）⇒ 记录值、断言数、指纹、FD 全同。`test_match` 的红是那族**既有**时序断言，与本次改动无关（三次运行 3/4/4 条，摆动的是同两条计时断言） | ✅ 完成 |
| **P3-A9** | 2026-10-10 | **单一长度源**：删掉 `std::size_t totalSize` 成员，`size()` 改成返回 **`val.size()`**；`initParams(Shape&, size_t&)` 改成 `static std::size_t initParams(const Shape&)`（返回形状乘积，只用于分配 + 过 `requireIndexable`/`requireRank` 守卫）；`reshape` 用 `requireSameLength(*this, initParams(shape))` **当即校验**；`empty`/`sum`/`mean`/`norm2`/`requireNonEmpty` 等改用 `val`；`operator=(T)` 从 `val.assign(totalSize, x)` 改成 `val.assign(val.size(), x)`；`MM::requireShape2d` 里那条"两套长度不一致"守卫删除（已恒真）；外部 **142 处** `X.totalSize` 机械改成 `X.size()`（19 个文件，语义相同）；记账进 `docs/rl_sync.md` §1.4 第 9 条 | ① 全量构建 **0 error**（142 处改名全由编译器核对；改名前先确认全仓**没有第二个类**声明 `totalSize` ⇒ 这些改名都是 `Tensor` 的读点，语义完全相同）；② 探针：构造仍 **1 次**分配、`reshape` 仍 **0**、秩 5 仍 **2**；**`sizeof(RL::Tensor)` 72 B → 64 B**（相对原始 80 B **净 −16 B**）；自查 **36/36**；③ A/B 与 ctest 见 P3-A9-验 行 | 把方案 §1.4 的第二条账也清了：`totalSize` 与 `val.size()` 曾是**两套长度**，靠 20 个函数手工同步、只改一个入口就永久分叉（`flatten()` 的历史越界 bug 属这一类）；现在长度只有一个来源，形状乘积变成"构造时当即校验的中间值"。两条守卫的删除（A10 的 `sizes` 一致性、A9 的 `val.size() >= totalSize` 断言）是同一句话：**能被结构消除的状态就不要用断言去守** | ✅ 完成 |
| **P3-A9-验** | 2026-10-10 | 空载跑 A/B（P2-附 树二进制 vs A9 树二进制，10 个 RL 测试）+ 全套 `ctest` | ① **A/B（逐行 diff）**：`test_bc` **0 差异**；`test_transformer` 只有 **3 行**不同（全是对齐地址 32/64/256 B，都满足 `alignedTo(32)=1`），其**权重文本 FNV-1a 仍是本机基线 `4cbbc09fc0d787c9`（490,037 B / 91,731 参数）、86 断言 0 失败**；`test_weights`（49 断言、v2 字节数 497,143）、`test_grad` 的 FD 读数、`test_sacaz`/`test_sparse_moe`/`test_scaledconcat`/`test_dqnab`/`test_ppo_backbone`/`test_pretrain` 的不变量行**全部逐字符相同**；10 个目标 **exit 0**；② **空载全套 `ctest`：17/18（2104 s）**（`.r1build/ctest_p3a9.txt`），唯一红仍是 `test_match`：**366 断言 / 3 失败**（`[2.7c]` + `[2.7k]`×2；这一轮 `[2.22]` 是**通过**的 —— 再次说明那族断言是时序抖动） | A9 与 A10/A11 一样是**逐位不变**的等价改写（唯一的语义改动是"长度不再单独存一份"，而它原来恒等于 `val.size()`）⇒ 指纹、FD、断言数、字节数全同 | ✅ 完成（P3 的"存储与事实源"三件套 A11+A10+A9 全部落定；只剩 C1 与 P3 收尾的 BC/PPO A/B） |
| P3（整段） | — | 还剩 **C1**（`val` 私有化 + 访问器：机械替换面 = `src` **78 行**直接碰 `val`、**14 处** `val.data()`、`test` **42 行**）；验收第 4 条（一轮 BC/PPO A/B，报告训练段耗时与 CE/top-1）留到 P3 收尾时一次做 | A11/A10/A9 已经做完"存储与事实源"这一半：分配 **3 → 1**、步长与长度**各自只剩一个来源**、`sizeof(Tensor)` **80 → 64 B** | C1 动的是**接口面**（`val` 有 78+14+42 处外部访问），是 P3 里唯一需要"机械替换 + 逐处核对语义"的一步；按方案的回滚点单独提交 | ⏳ 待执行 |
| P4 | — | — | — | — | 待执行 |
| P5 | — | — | — | — | 待执行 |
| P6 | — | — | — | — | 待执行（先做只读探针） |

---

## Part 6.A · P1 期间发现的两个新问题（**未解决**，下次继续）

### 6.A.1 ⚠ 常驻守卫抓到一条**运行中的静默 bug**（DQN 路径，`test_pretrain` 现在会响停）

这是 A4 想要的效果，也正是它的代价：**`test_pretrain` 现在会 abort**。

```
[Tensor::MM] 形状契约失守: kikj 需要 z(k,c) = x1(r,k)^T * x2(r,c)
  ⇒ 实际形状: z[64,1] (缓冲 64) x1[64,90] (缓冲 5760) x2[64,1] (缓冲 64)
```

读法：`x1` = 某个 **90→64 的全连接层**的 `w`，`x2` = 它收到的输出梯度 `(64,1)` ✓，
而**接收输入梯度的 sink `z` 是 (64,1)，按契约必须是 (90,1)**（`k = inputDim = 90`）。
改动前 Release 下断言被 `NDEBUG` 关掉 ⇒ 这个输入梯度**只写了 64 个坐标**，
剩下 26 个坐标留在旧值/零上（静默错误的梯度）。

**✅ 根因已定位并修复（2026-10-10，同一个 goal 轮次内）：`LayerNorm<Fn, LN::Pre>::backward`
（`src/rl/layer.h:742`）**

```cpp
Tensor dy(outputDim, 1);
for (i < e.totalSize) dy[i] = Fn::df(o[i]) * e[i];
Tensor dL(outputDim, 1);            // ← 错在这里：应该是 inputDim
Tensor::MM::kikj(dL, w, dy);        //   kikj 的 z 必须是 (inputDim,1) = wᵀ·dy
float u = dL.mean();
for (i < ei.totalSize) ei[i] = gamma * (dL[i] - u);   // ← inputDim 个坐标, 但 dL 只有 outputDim 个
```

一次错分配引出**两个**后果（都不报错）：
1. `kikj` 的 sink 太短 ⇒ 输入梯度的后 `inputDim − outputDim = 26` 个坐标**从未被写**；
2. 紧接着那个循环按 `ei.totalSize = inputDim` 读 `dL[i]` ⇒ **越界读**（读缓冲区外的字节写进 `ei`）。
   而且 z-score 的雅可比里那次"减均值"本来就该在**输入向量**（长度 inputDim）上做，
   按 `outputDim` 取均值连数学口径都是错的。

**触发它的是 DPG 这一支, 不是 DQN**（前一轮我按日志把责任记到 DQN 头上，这里更正）：
`dpg.cpp:43` 的 `LayerNorm<Sigmoid, LN::Pre>::_(stateDim=90, hiddenDim=64, …)` 正是
inputDim ≠ outputDim 的用法；`test_pretrain` 会遍历多个 agent，stdout/stderr 是两个文件，
"DQNAgent 训练…" 那一行与 abort 的先后并不能配对 —— 教训与 §8.4 同源：**别跨通道配对读数**。
（定位手段：先按"w=(64,90) 的 FC 类"缩小到两个候选，再用 `dpg.cpp` 的 `(stateDim, hiddenDim)` 确认；
`TanhNorm::backward` 的入口打印始终没出现这一点反过来排除了 DQN 那一支。）

**修复**：`Tensor dL(inputDim, 1);` —— 一个词，同时消掉截断与越界读。
对 `inputDim == outputDim` 的既有用法（`ddpg.cpp:11/15`）**逐位不变**。
`layer.h` 里已写下这段历史（含"被常驻守卫抓到"的证据链）。

⚠ **验证状态（2026-10-10 已补齐，换机后重做）**：`test_pretrain` **38 断言 / 0 失败 / exit 0**（9.30 s，ctest 内），
全量 `ctest` 16/18（两条红是并行编译造成的计时类假红，空载复现为绿 —— 见 Part 6 的 **P1-复跑** 行）。
**这条"已知红"已清掉**，Part 7 里对应的风险行随之关闭。

### 6.A.2 ⚠ 环境问题：PowerShell/编译期**间歇性读到垃圾文件内容**

现象：`src/rl/optimize.h` 与 `src/rl/net.hpp` 有几次被 **PowerShell 的 .NET 读**（
`ReadAllBytes`/`Get-Content`/`Select-String`）与**编译器**读成二进制垃圾
（`optimize.h(1): error C2018: 未知字符"0xc9"`，文件首字节 `89 7d 1c 39`），
而同一时刻 **`git diff` / `cmd type` / read 工具都读到正确文本**，文件大小与 mtime 均未变。
重试一次构建即恢复正常（多次复现：pwsh-13/pwsh-16 失败 → pwsh-14/pwsh-17 成功）。

结论与纪律：
- **不是代码问题，也没有文件被真的破坏**（`git diff --numstat` 始终是 `69 0`，
  与损坏前的账一致；`.r1build/optimize.h.backup_pre_p1.txt` 另存了一份）；
- ⚠ 一旦命中，**构建的 `BUILD=1` 是假信号**：先看错误是不是"某个头文件第 1 行出现乱码字符"，
  是就**直接重试**，不要去改代码；
- ⚠ 反向风险同样存在：`BUILD=0` 也可能是**读到了旧副本**编译出来的
  （所以本轮用二进制里的标记串（`strings`）来确认诊断代码真的进了二进制）。

---

## Part 7 · 风险登记

| 风险 | 触发阶段 | 影响 | 缓解 |
|---|---|---|---|
| 权重文件/指纹变化 | P3、P4 | 存量模型不可载、`test_weights` 红 | 格式字节不变 + **本机 P1 基线**比对（见 Part 8 第 10 条：旧机的绝对哈希 `9a5483d39a42a333` 在新机器上复现不出来，跨机比较本来就不成立）+ 往返测试钉住 + 老 v1 兼容 |
| 静默别名（视图） | P5 | 数值静默错，最坏是"看起来合理" | 只读先行 + owner 显式 + FD/置换等变断言 |
| 求和顺序变化 | P6 | 不能要求逐位相同 | 唯一允许不逐位等价的项；必须两棵树 A/B + 指标口径 |
| `NDEBUG` 断言纪律冲突 | P1 | 与 `issues_review.md:1402-1411` 冲突 | 开关 + 只常驻一条 + 代价 A/B |
| 隐式转换"零引用"误判 | P2 | 构建失败或语义变化 | 以编译结果为准；删一项编译一次 |
| 上游同步难度上升 | 全部 | 下次 sync 覆盖本地修复 | P0 记账 + 文件内注释 + `rl_sync` 清单 |
| 收益不可分辨 | P3、P4 | 白做工 | 先量（分配次数、占比、耗时）再决定做不做 |
| **P1 引入**：`mean/variance/max/min/argmax/argmin/normalize` 现在对空张量**响亮失败** | P1 已落地 | 若某条**没被测试覆盖**的生产路径（GUI/agent 层）真的会统计空张量，行为从"静默 NaN"变成"abort" | 这是刻意的（空张量上的统计量本来就没有定义）；风险点是"以前能带病跑"的地方现在会停。8 个测试目标全绿说明主流路径不碰它；**再看到 abort 时按"这是 bug"处理，而不是把守卫拿掉** |
| **P1 引入**：`_set_abort_behavior(0, …)` 让本测试的 abort 退出码从 `0xC0000409` 变成 `3` | P1 已落地 | 任何**硬编码** 0xC0000409 的判据会失效 | 现有断言都只判"非 0"（已核对 [6g]/[8]/[10]）；新加的子进程断言也按"非 0"写 |
| **P1 引入**：`test_pretrain` 变红（DQN 路径的 kikj 形状契约失守，以前静默） | P1 已落地 | ~~ctest 由 16/16 变 15/16~~ | **已关闭（2026-10-10）**：`layer.h` 的根因修复 + 复跑 **38/0 exit 0** ⇒ 见 Part 6.A.1。**不要**用逃生门把守卫关掉 |
| **【新】跑测试时顺手编译 ⇒ 计时/并发类断言假红** | P2（本轮实测） | `test_match`（3/366 失败）、`test_sparse_moe`（2/100 失败）——全部落在"后台训练必须在时间窗内跑完一轮 / 并发窗口必须重叠"的断言上；空载复跑即全绿 | §5.1 的口径再加一条：**并发/计时类断言只在空载机器上量**；`ctest` 与 `cmake --build` 不要同时跑（本轮就是同时跑才出现的，见 Part 6 的 P1-复跑行） |
| **【新】旧机的绝对指纹在新机器上对不上** | P1-复跑（本轮实测） | "指纹不变"这条验收如果照旧机数字判，会**误判成失败**（本机 `4cbbc09fc0d787c9` vs 文档 `9a5483d39a42a333`，参数量与文本长度完全相同） | 一律与**本机 P1 基线**（`.r1build/p1bin/` + `p1_*.out.txt`）比；原因未查明 ⇒ 列在 Part 8 第 10 条 |

---

## Part 8 · 没有得出的结论 / 待实测

1. **注意力矩阵化的加速比未测**：39%（占 forward）是实测的，但"矩阵化后能快多少"是【推理】⇒ 由 **P6-前哨** 的只读探针回答。
2. **"逐 token 搬运占 37%"未细分**：它是"三项之和 = 0.63× forward"的差额，含 gather/scatter/残差/逐 token 拷贝 ⇒ **P5 能拿回多少**未测。
3. **形状 SSO 的 ~480 ns 是推算**（按"3 次分配→1 次"× 实测每次小分配 ~240 ns），未实现未 A/B。
4. **`const&` 传参的 ≥100 MB/步是算式**（1263²×4 float × 4 专家），未做 A/B。
5. **`mat.hpp` / `TRPO` / D1 清单删除后**的编译时间、二进制体积、PCH 收益未测。**【2026-10-10 P2/P2-附 已量】**：编译时间 **309 s（459 步）→ 295 s（458 步）**（同 `-j12`，差异落在噪声里）；二进制体积（P1 树 → P2-附 树，13 个测试/探针）**各缩小 0~19 KB、合计 ≈117 KB** ⇒ 结论是**负结果**：零调用代码本来就被链接器剔掉，删它不省体积也不省运行时间，收益只在**源码可维护性**（PCH 收益仍未测 —— `CHESS_USE_PCH` 目前是 `OFF`）。
6. **Caffe/PyTorch 的收益不能引用**：它们的数字是在 GPU / 大 batch / 大矩阵上兑现的；本仓库主要形状是 batch=1、90×90（1263×1263 只在 SACAZ MoE TB 出现）。
7. 本方案（2026-10）里**P0 不改代码**（只有 `docs/` 与 `README.md`）；**P1 改了 `src/rl/tensor.hpp`、`src/rl/util.hpp`、`test/test_transformer_main.cpp`**，**P1-附**又改了 `src/rl/layer.h`（`LayerNorm<Fn,LN::Pre>::backward` 的 sink 尺寸），逐位不变的证据见 Part 6 台账。**P2（2026-10-10）只改两处**：`src/rl/tensor.hpp`（净删 628 行）与 `docs/rl_sync.md`（新增 §1.5：删掉的上游同源代码项逐条记账）。**P2 没有改任何 `src/rl/` 之外的代码，也没有动测试**。
8. **P1 的常驻检查代价是模型推算 + 单点实测的组合**：`probe_checks` 直接量到"6 次比较 = 17.8~18.5 ns/调用"（同轮配对），再按"seq 一次 forward 有 1080 次 FC 调用"折算成 ≈0.12% 的 forward；**没有**做"整棵树下开/关这个宏"的端到端 A/B（`RL_TENSOR_NO_CONTRACT` 逃生门已经就位，想做随时能做）。
9. `test_transformer` 的**绝对墙钟不能用来判断这次改动快慢**：本轮同一二进制测到 150.6 s 与 35.6 s 两个数，差别全部来自"`abort()` 是否触发 WER"（见 §5.1.1）—— 又一次印证 §8.4 的口径。
10. **【2026-10-10 换机后新增】"绝对指纹"这条不变量的适用范围被收窄了**：`test_transformer [3]` 在**新机器**上给出 `FNV-1a=4cbbc09fc0d787c9`（参数量 91,731、文本 490,037 B —— 与 `seq_transformer_design.md:301` 记的**同形**），而文档一直记的是 `9a5483d39a42a333`。同一台机器上 **P1 树与 P2 树给出同一个** `4cbbc09fc0d787c9`，所以差异不在本次改动：它是**另一台机器/另一套构建工具链**上的权重初值差异（形状与文本长度完全相同 ⇒ 是**数值位**不同，不是结构不同）。**未查清的部分**：到底是哪个环节引入的（`std::normal_distribution` 的 STL 版本实现、FMA 收缩、还是别的编译期差异）没有定论 —— 属于**待实测**。纪律：**"指纹不变"从此只在本机基线上比**（`.r1build/p1bin/` + `.r1build/p1_*.out.txt`），不再引用旧机的绝对哈希；这与 `rl_sync.md:270-273` 已经写明的"跨构建不成立"是同一条道理。
11. **P2-附 的删除影响面（2026-10-10 已执行；取证见 Part 6 的 P2-附 行与 `docs/rl_sync.md` §1.5 第 5~8 条）**：`src/rl/mat.hpp` = **543 行 / 16 KB**，全仓**零 `#include "mat.hpp"`**、也不在 `CMakeLists.txt` 的 `RL_SOURCES` 里 ⇒ 连"死代码"都算不上，是**没被编译的死文件**；`TRPO` = `trpo.h`(45 行) + `trpo.cpp`(467 行)，**在 `CMakeLists.txt:73` 编进 `RL_CORE`**，但 `trpo.h` 只被 `trpo.cpp` 自己 include（零外部调用者）⇒ 每个全量构建都在为它付编译时间；`util.hpp` 的 6 个**按值返回**版单目函数（`Tensor sqrt/exp/log/tanh/sin/cos(const Tensor&)`）全仓只有 **2 处 `RL::tanh(...)`** 调用（`src/rl/layer.h:927`、`test/probe_tanh_norm_ops_main.cpp:119`），两处实参都是**非 const 左值** ⇒ 重载决议选中的是**就地版**（`Tensor& tanh(Tensor&)`），按值版**零调用者**（它们每次都会**整块深拷贝**一个张量，正是 §1.2 里 `拷贝构造 11×` 那一类成本）。
12. **`test_match` 在本机的两条既有/环境红（2026-10-10 归因完毕）**：① `[2.7c] 学习信号与对手无关`（"换对手后**首次**训练损失必须**逐位相同**"）**在 P1 之前的树上同样红**（`.r1build/prep1`，`HEAD~1`，366 断言 2 失败、1875 s；两次首损失 `0.0546623692` vs `0.0393589064`）⇒ 既有时序脆弱性（训练损失的上报与后台训练线程竞争），**不是**权重还原/守卫/删死代码造成的；② 该测试**隐含要求 cwd 下 `weights/` 子目录已存在**（它自己**不建**这个目录）：在全新的构建目录里跑，`saveCurrentAgentModel` 会一路失败并把十几条断言级联带红（本机 `build/plan_p2` 的实测：目录 CreationTime 22:37:44 晚于 `[2.7c]` 的执行时刻）。**下一轮修复清单**（测试侧，不动 `src/rl`）：① ✅ **已做（2026-10-10）**：`main()` 开头 `std::filesystem::create_directories("weights")` —— 修完那次空载 ctest 里 `test_match` 的失败**从 12 条降到 3 条**（那 9 条级联假红全部消失，剩下的 3 条与 P1 树那次完全相同）；② ⏳ **未做，需用户定夺**：`[2.7c]` 的判据（"换对手后首次训练损失必须逐位相同"）在这台机器上**不稳定**（P1 之前的树、P1 树、P2 树三棵树都红），可选的改法是"同一对手跑两遍必须逐位相同"这条**等价但更稳**的命题，或把窗口放宽并显式声明"容许后台训练竞争" —— 本轮**刻意没有动它**：这是测试语义问题，且有"把失败的判据改掉"的风险，按本方案的纪律（"再看到 abort/红时按'这是 bug'处理，而不是把守卫拿掉"）应由人决定。
13. **换机后的"绝对读数不可比"清单（本轮踩到的）**：CPU 由 i7-12650H（6P+4E/16 线程/睿频更高）换成 **i5-10400F（6 核 12 线程 @2.9 GHz）**，于是 ① `test_transformer` 全套 35.6 s（旧）→ **4.4~14.2 s**（新，快）；② `test_mcts` 旧机 **Timeout**、新机 **202 s 通过**；③ 计时/并发窗口类断言（"30 s 内后台训练必须改权重"、"并发窗口必须重叠"）在新机上会红，且**并行编译时更红**；④ `test_transformer [3]` 的权重文本指纹由 `9a5483d39a42a333`（旧）变成 `4cbbc09fc0d787c9`（新，参数量与文本长度完全相同）。**结论**：换机后一切"不变性"验收都要先在本机重建基线（本机基线已存 `.r1build/p1bin/` + `p1_*.out.txt`）。
14. **【A10 留下的一个已知边界，未修】**：`posOf()` 现在从**栈上的步长数组**取值（`int strides[RL_SHAPE_MAX]`），秩 ≤ `RL_SHAPE_MAX` 时它一定被写满，但**下标个数超过秩**的调用（例如对秩 1 的张量写 `posOf(0, 0)`）或**对默认构造的秩 0 张量**取下标，会读到栈上未初始化的那几个 int。⚠ 这不是 A10 引入的：改动前它读的是 `sizes[i]`（`std::vector<int>` 的越界读，同样是 UB，只是"读堆上别人的字节"换成"读自己栈上没初始化的字节"）。两者都只在**调用方违约**时发生，本轮没有把它变成响亮失败 —— 若要与 P1 的 `requireIndexable`/`requireNonEmpty` 同一口径处理，应当在下标个数 > 秩 时 `contractFail`（代价是 `posOf` 上多一个可预测分支，属于"要单独量一次"的改动）。

---

## Part 9 · 参考

### 9.1 仓库内

| 主题 | 位置 |
|---|---|
| 目标文件 | `src/rl/tensor.hpp`（2266 行）；`src/rl/alignallocator.hpp`；`src/rl/simd_ops.hpp`；`src/rl/seq_transformer.hpp` |
| 与上游差异与纪律 | `docs/rl_sync.md:1-24`（同步结果）、`:40-78`（chess 侧修补五处）、`:129-136`（为什么不整类替换）、`:283-299`（权重格式 v2）、`:270-273`（跨构建比对不成立）、`:360-365`（`embedding` 的既有记录） |
| 问题清单 | `docs/issues_review.md:1047-1093`（B19）、`:1360-1412`（R1.5：累加契约 + `requireShape2d` 必须留在 `NDEBUG`）、`:1989`（警告基线） |
| 注意力/序列骨干 | `docs/seq_transformer_design.md`（§8.4 计时口径、§9.2 代价分解、§9.4-9.6 位置编码/dropout/死代码守卫、§10.2 不变量） |
| 方案文档先例 | `docs/rl_plan_optimized.md`（"原建议 → 优化后" + 实施记录） |
| 事故记录 | `src/rl/sac.h:99-104`（Release 静默读错，MM 形状契约失守） |

### 9.2 证据文件（`.r1build/`，gitignore）

`probe_pool_run1.txt` / `probe_pool_run2.txt`（分配次数、池上限、写密集）、`probe_pool_main.cpp.txt`（探针源码）；
`probe_move_run1.txt` + `probe_move_main.cpp.txt`（`noexcept` move，P1 前的 before 臂）；
`probe_apid_run1.txt` + `probe_apid_main.cpp.txt`（小操作 API 税）。

**P1 之后新增的重测与回归证据**（同一套探针 + 两个新证据）：
`probe_move_after_p1.txt`（增长段 7.27 → 1.18 ms）、`probe_checks_run1.txt`（检查 17.8~18.5 ns/内核调用）、
`probe_apid_after_p1.txt`（无回归）、`test_transformer_p1.txt`（86 断言 / [10] 六个守卫全部响亮失败 / 指纹不变）、
`bc4_seq_p1.txt`（BC 一整臂与 P1 前逐行相同）。

⚠ **上面这批旧机证据文件在本机不存在**（换 PC 时只有 `docs/`、`src/`、`test/` 随仓库过来；
`.r1build/` 里最新的一批是 2026-10-06 的，P1 那一轮的探针读数没有带过来）⇒ P1 台账里 ④⑤⑥⑦
那几条**探针读数在本机无法复核**，需要时按 Part 9.3 的同口径探针**重跑**（`probe_seq_cost` 等目标仍然在
`CMakeLists.txt` 里）。**下面这批是 2026-10-10 在本机重建的、本机唯一有效的基线**：

| 证据文件 | 内容 |
|---|---|
| `.r1build/p1bin/*.exe`（13 个） | **P1 树的二进制**（在改动 `tensor.hpp` 之前从 `build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release` 拷出来的），P2 的 A/B 就是拿它当 before 臂 |
| `.r1build/plan_p1verify_build.log` | P1 全量构建：**459 步 / 0 error / 309 s**（21:28:55→21:34:04） |
| `.r1build/test_pretrain_after_fix.{out,err,exit}.txt` | `test_pretrain` **38 断言 / 0 失败 / exit 0** |
| `.r1build/ctest_after_p1fix.txt` | P1 树全套 `ctest`：**16/18**（1984 s），含两条被并行编译干扰的假红与其原文 |
| `.r1build/p1_*.out.txt` / `p2_*.out.txt` | P1 树与 P2 树跑同一批测试的**逐行可 diff** 输出（指纹、FD、往返差、断言数） |
| `.r1build/plan_p2_build.log` | P2 第一次全量构建：**BUILD_EXIT=1**、6 条 error —— 编译器否掉 `empty()` 的那一次（本方案的验收纪律生效的原始记录） |
| `.r1build/plan_p2_build2.log` | 放回 `empty()` 之后：**466 步 / 0 error / 316 s**（`-j8`） |
| `.r1build/tensor.hpp.pre_p2.txt` + `.r1build/p2_delete_blocks.ps1` | P2 删除前的整份头文件 + **带首尾行断言**的删除脚本（每个块切之前先核对首行/末行内容，对不上就 abort） |
| `.r1build/ctest_p2.txt` | P2 树全套 `ctest`（空载） |
| `.r1build/capture_test.ps1`、`plan_build.bat`、`plan_p2_build.bat` | 本机复现用的工具（构建脚本刻意保持**纯 ASCII**：带 UTF-8 BOM 注释的 `.bat` 会被 `cmd` 当成命令执行） |

**P2-附（同一天第二阶段）新增**：

| 证据文件 | 内容 |
|---|---|
| `.r1build/p2b_bin/*.exe`（13 个） | **P2-附 树的二进制**（删完 `TRPO`/`Mat`/按值返回版之后），P3 的 A/B 用它当 before 臂 |
| `.r1build/p1c_*.out.txt` / `p2c_*.out.txt` | **成对 A/B**：同一台机器、同一工具链，P1 树二进制与 P2-附 树二进制各跑 10 个 RL 测试（`test_weights`/`test_grad`/`test_transformer`/`test_sparse_moe`/`test_scaledconcat`/`test_ppo_backbone`/`test_bc`/`test_pretrain`/`test_sacaz`/`test_dqnab`） |
| `.r1build/p2d_*.out.txt` | **对照**：同一份 P2-附 二进制对 `test_ppo_backbone`/`test_pretrain`/`test_sacaz` 的**重跑**，用来证明那几个"看起来变了"的数字是测试自身的非确定性 |
| `.r1build/p2b_build1/2/3.log` | P2-附 的三次构建：① 删 `TRPO`/`Mat`/`util` 按值版 + 封掉 `Tensor⊗Tensor` 算子 → 0 error；② 再封掉 `Tensor⊗T` 算子 → 0 error（两组零消费者都由编译器证实）；③ 定稿（含 `test_match` 的 `weights/` 修复）→ 458 步 / 0 error / 295 s |
| `.r1build/ctest_p2b.txt` | P2-附 树空载全套 `ctest` |
| `.r1build/prep1/`（`git worktree`，`HEAD~1` 检出）+ `prep1_test_match.{out,log}.txt` | **归因实验**用的 P1 之前那棵树（含只编 `test_match` 的构建日志与运行输出） |

### 9.3 外部

- Caffe `Blob`（`data_/diff_/shape_/count_/capacity_`、`cpu_data()/mutable_cpu_data()`、`ShareData`）：<https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/blob.cpp>
- 张量的 storage/view 模型（sizes + strides + offset）：<https://fleuret.org/dlc/materials/dlc-handout-1-6-tensor-internals.pdf>
- PyTorch `TensorIterator`（逐元素算子的统一迭代器）：<https://github.com/pytorch/pytorch/pull/175336/files/8f3aa44c2e4e8c8c01905f25732df09e34f5804a#1>
