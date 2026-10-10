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
另有两处**整类重复实现**建议单独一轮评估：`src/rl/mat.hpp` 的 `Mat`（588 行，与 `Tensor` 平行、**不在 `CMakeLists.txt` 源列表**、`val` 还是无对齐的 `std::vector<float>`）；整个 `TRPO`（无调用者，却仍在 `CMakeLists.txt:73` 编译，且是 `Tensor op Tensor` 值语义算子的唯一 live 消费者）。

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
| 权重文本 FNV-1a 指纹 `9a5483d39a42a333`（490,037 B 文本）不变 | `test_transformer`、`test_weights` |
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
| **P1-附** | 2026-10-10 | 修 A4 抓到的那条真 bug：`src/rl/layer.h` 的 `LayerNorm<Fn, LN::Pre>::backward` 把 `dL` 从 `outputDim` 改成 **`inputDim`**（`kikj` 的 sink 尺寸 + 越界读 + z-score 均值口径，三个问题一个词解决）；文件内写下完整证据链 | 触发点：`dpg.cpp:43` 的 `LN::Pre(90→64)`；守卫报的形状 `z[64,1] x1[64,90] x2[64,1]`；`ddpg.cpp:11/15` 的 `inputDim == outputDim` 用法**逐位不变** | 静默 bug 修复；**验证（`test_pretrain` + ctest）按用户要求暂停** | ⏳ 验证待补 |
| P2 | — | — | — | — | 待执行 |
| P3 | — | — | — | — | 待执行 |
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

⚠ **验证状态**：修复代码已入库，但 `test_pretrain` 的复跑按用户要求**暂停**（本轮只做到构建）。
下一轮第一件事就是跑它 + 全量 ctest，把 P1 的那一条从"已知红"清掉。

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
| 权重文件/指纹变化 | P3、P4 | 存量模型不可载、`test_weights` 红 | 格式字节不变 + 指纹与往返测试钉住 + 老 v1 兼容 |
| 静默别名（视图） | P5 | 数值静默错，最坏是"看起来合理" | 只读先行 + owner 显式 + FD/置换等变断言 |
| 求和顺序变化 | P6 | 不能要求逐位相同 | 唯一允许不逐位等价的项；必须两棵树 A/B + 指标口径 |
| `NDEBUG` 断言纪律冲突 | P1 | 与 `issues_review.md:1402-1411` 冲突 | 开关 + 只常驻一条 + 代价 A/B |
| 隐式转换"零引用"误判 | P2 | 构建失败或语义变化 | 以编译结果为准；删一项编译一次 |
| 上游同步难度上升 | 全部 | 下次 sync 覆盖本地修复 | P0 记账 + 文件内注释 + `rl_sync` 清单 |
| 收益不可分辨 | P3、P4 | 白做工 | 先量（分配次数、占比、耗时）再决定做不做 |
| **P1 引入**：`mean/variance/max/min/argmax/argmin/normalize` 现在对空张量**响亮失败** | P1 已落地 | 若某条**没被测试覆盖**的生产路径（GUI/agent 层）真的会统计空张量，行为从"静默 NaN"变成"abort" | 这是刻意的（空张量上的统计量本来就没有定义）；风险点是"以前能带病跑"的地方现在会停。8 个测试目标全绿说明主流路径不碰它；**再看到 abort 时按"这是 bug"处理，而不是把守卫拿掉** |
| **P1 引入**：`_set_abort_behavior(0, …)` 让本测试的 abort 退出码从 `0xC0000409` 变成 `3` | P1 已落地 | 任何**硬编码** 0xC0000409 的判据会失效 | 现有断言都只判"非 0"（已核对 [6g]/[8]/[10]）；新加的子进程断言也按"非 0"写 |
| **P1 引入**：`test_pretrain` 变红（DQN 路径的 kikj 形状契约失守，以前静默） | P1 已落地 | ctest 由 16/16 变 15/16（另一条 `test_mcts` 是 Timeout，与本次改动无关） | 见 Part 6.A.1：定位并修根因；**不要**用逃生门把守卫关掉 |

---

## Part 8 · 没有得出的结论 / 待实测

1. **注意力矩阵化的加速比未测**：39%（占 forward）是实测的，但"矩阵化后能快多少"是【推理】⇒ 由 **P6-前哨** 的只读探针回答。
2. **"逐 token 搬运占 37%"未细分**：它是"三项之和 = 0.63× forward"的差额，含 gather/scatter/残差/逐 token 拷贝 ⇒ **P5 能拿回多少**未测。
3. **形状 SSO 的 ~480 ns 是推算**（按"3 次分配→1 次"× 实测每次小分配 ~240 ns），未实现未 A/B。
4. **`const&` 传参的 ≥100 MB/步是算式**（1263²×4 float × 4 专家），未做 A/B。
5. **`mat.hpp` / `TRPO` / D1 清单删除后**的编译时间、二进制体积、PCH 收益未测。
6. **Caffe/PyTorch 的收益不能引用**：它们的数字是在 GPU / 大 batch / 大矩阵上兑现的；本仓库主要形状是 batch=1、90×90（1263×1263 只在 SACAZ MoE TB 出现）。
7. 本方案（2026-10）里**P0 不改代码**（只有 `docs/` 与 `README.md`）；**P1 改了 `src/rl/tensor.hpp`、`src/rl/util.hpp`、`test/test_transformer_main.cpp`**，**P1-附**又改了 `src/rl/layer.h`（`LayerNorm<Fn,LN::Pre>::backward` 的 sink 尺寸），逐位不变的证据见 Part 6 台账。
8. **P1 的常驻检查代价是模型推算 + 单点实测的组合**：`probe_checks` 直接量到"6 次比较 = 17.8~18.5 ns/调用"（同轮配对），再按"seq 一次 forward 有 1080 次 FC 调用"折算成 ≈0.12% 的 forward；**没有**做"整棵树下开/关这个宏"的端到端 A/B（`RL_TENSOR_NO_CONTRACT` 逃生门已经就位，想做随时能做）。
9. `test_transformer` 的**绝对墙钟不能用来判断这次改动快慢**：本轮同一二进制测到 150.6 s 与 35.6 s 两个数，差别全部来自"`abort()` 是否触发 WER"（见 §5.1.1）—— 又一次印证 §8.4 的口径。

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

### 9.3 外部

- Caffe `Blob`（`data_/diff_/shape_/count_/capacity_`、`cpu_data()/mutable_cpu_data()`、`ShareData`）：<https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/blob.cpp>
- 张量的 storage/view 模型（sizes + strides + offset）：<https://fleuret.org/dlc/materials/dlc-handout-1-6-tensor-internals.pdf>
- PyTorch `TensorIterator`（逐元素算子的统一迭代器）：<https://github.com/pytorch/pytorch/pull/175336/files/8f3aa44c2e4e8c8c01905f25732df09e34f5804a#1>
