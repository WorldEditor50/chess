# rl/ 代码同步 + RL_CORE 构建组织

本文记录两件事：

1. 把 `E:\home\lab\snakeAI\snakeAI\rl` 的代码同步到 `src/rl`（同步来源：snakeAI 仓库
   `5d3246e` "Fix reward plot after agent switch; audit fixes in rl/ and tests"）。
2. 参考 snakeAI 工程的构建组织，把 `src/rl` 编译为**独立的静态库 `RL_CORE`**。

同步前的完整备份：`E:\home\lab\backup\chess_rl_20260912_214647`（52 个文件）。

---

## 一、同步结果总览

| 类别 | 数量 | 说明 |
|------|------|------|
| 与 snakeAI 工作区完全一致（字节级） | 43 | 直接覆盖 |
| 保持 chess 版本，未覆盖 | 2 | `rl/ppo.h`、`rl/ppo.cpp` |
| chess 独有、snakeAI 没有 | 2 | `rl/qlstm.h`、`rl/qlstm.cpp` |
| 按 chess 侧需要改回/修补后与上游不同 | 5 | `rl/dqn.cpp`、`rl/dqn.h`、`rl/dpg.h`、`rl/layer.h`、`rl/convdqn.cpp` |

> 同步基线是 snakeAI 的 `5d3246e`（"Fix reward plot after agent switch; audit fixes
> in rl/ and tests"）。同步过程中发现 snakeAI **工作区**在同步之后又被改过两处
> （`rl/conv2d.hpp`、`rl/convdqn.cpp`，均为未提交改动），处理方式见 §1.3。

### 1.1 为什么 `ppo.h` / `ppo.cpp` 不能覆盖

chess 的 `ppo.h/cpp` 是**为 AlphaZero 风格改写过的简化 PPO**（`action()` / `value()` /
`trainStep()` / `learnSelfPlay()`，成员 `actorP` + `critic` 公开），而 snakeAI 的
`ppo.h/cpp` 是另一套算法（`learnWithKLpenalty` / `learnWithClipObjective` /
`eGreedyAction` / `gumbelMax` / `actorQ`，成员为 `protected`）。

`src/ppomcts_agent.cpp` 依赖 chess 版的 API：
`ppo.action()`（ppomcts_agent.cpp:137）、`ppo.value()`（:140）、
`ppo.learnSelfPlay()`（:432/632/668/689/759/906/928/942/1015）、
`ppo.exploringRate`（:33）、`ppo.save/load`（:969/976）。
直接覆盖会导致 7 个目标全部编译失败，因此保留 chess 版。
`ppo.h` 顶部已注明它与上游的分歧。

### 1.2 chess 侧保留/修补的五处（均在文件内以注释标明）

1. **`rl/dqn.cpp`** — chess 把 DQN 主干换成了 `MOE<16,16>` +
   `TransformerBlock<16>`（旧注释还写着 MOE<8,4>，属于注释未同步）。
   该架构在同步前**是坏的**：旧 `attention.hpp` 在 `d_model`(90) 无法被
   `NumHeads`(16) 整除时会把 `d_model` 向上取整为 96，而 `TransformerBlock`
   自己按 90 分配的缓冲不变 → 越界读写。同步后的 `attention.hpp` 改为把
   head 数降到 90 的最大公约数（15，`d_k=6`）且 `d_model` 永不变，
   chess 的这个主干因此**从"坏的"变成"可用的"**。
   处理方式：chess 架构保持为**当前生效分支**（`#if 1`），snakeAI 的三种
   主干作为 `#if 0` 分支保留在原位，改一个数字即可切换。
2. **`rl/dqn.h` / `rl/dpg.h`** — snakeAI 把成员区改成 `protected`，但 chess 的
   `src/dqnagent.cpp`(:39,40,163,219,…)、`src/dqnmcts_agent.cpp`(:40,41,365,…)、
   `src/dqnagent.h`(:121)、`src/dqnmcts_agent.h`(:169)、`src/pgagent.cpp`
   (:24,25,162,361,457,465,…) 直接读写 `gamma` / `exploringRate` /
   `learningRate` / `policyNet`。这两个头文件的成员区保持 `public:`，
   并在文件中注明"要收紧为 protected 时必须同时给调用方加访问器"。
3. **`rl/layer.h`** — snakeAI 的这份文件里有 33 处编码损坏（`·`→`路`、
   `→`→`鈫?`、`²`→`虏`，全部在注释里）。chess 原文件是干净的，因此覆盖后
   做了等价还原（`路`→`·`、`鈫?`→`→`、`虏`→`²`）。
4. **`rl/convdqn.cpp`** — chess 保留 snakeAI **已提交**的版本（Q 头 `Layer<Linear>`，
   与文件内那段"Q 头必须有界……不，必须无界"的审计注释一致）。
5. **`rl/layer.h` 的 `iFcLayer` 拷贝构造** — 同步进来的版本**只复制了维度**，
   `w` / `b` / `o` / `e` / `g` / `v` / `m` 全部丢失：

   ```cpp
   // 同步进来的版本 (坏了)
   explicit iFcLayer(const iFcLayer &r)
       : iLayer(r), inputDim(r.inputDim), outputDim(r.outputDim), bias(r.bias) {}
   ```

   平时看不出来（`Layer<T>` 的深拷贝走 `copyTo()`，是赋值而不是拷贝构造），但只要
   有人**按值**返回/传递一个层，拿到的就是"维度对、权重空"的空壳（`w.size()==0`），
   之后所有 MM 内核都在越界读写。chess 侧写稀疏 MoE 时正好踩上
   （`experts[i] = ExpertFactory<E>::make(...)`），表现是反向出现上万个 NaN/1e28，
   而且**换个构建就可能消失**（MSVC 是否省略那次拷贝）。已补全为逐成员复制，
   并保留 `copyTo()` 作为真正的深拷贝入口。细节见
   `docs/issues_review.md` §零之四 4.5 与 `docs/agents_design.md` §11.4.2 (2)，
   回归覆盖在 `test_sparse_moe`。

### 1.3 同步后发现的上游未提交改动

同步完成之后，snakeAI 工作区又出现了两处**未提交**改动，处理如下：

| 文件 | 上游改动 | 处理 |
|------|----------|------|
| `rl/conv2d.hpp` | `Conv2d::forward/backward` 的激活从硬编码 `Tanh::f`/`Tanh::df` 改为模板参数 `Fn::f`/`Fn::df` | **已拉取**。这是正确的（`Conv2d<Tanh>` 之外还有 `Conv2d<Sigmoid>` 等实例化，硬编码 Tanh 会让模板参数失效）；对 chess 是行为等价的，因为 chess 只用 `Conv2d<Tanh>` |
| `rl/convdqn.cpp` | 把 Q 头从 `Layer<Linear>` **改回** `Layer<Sigmoid>` | **未拉取**。这与紧随其上的注释直接矛盾（该注释解释了为什么必须用无界的 Linear），看起来是一次误改或未完成的实验；且 chess 从未实例化 `ConvDQN`，保持已提交的审计版本更安全 |

如果上游这两处后续提交了不同的最终版本，重新跑一次同步即可。

---

## 二、同步进来的主要修复（按主题）

### `Tensor::MM`：训练慢的根因（0.11 → 37 GFLOP/s）
`struct MM` 的所有内核（`ikkj`/`kikj`/`ikjk`/`kijk` 及其返回副本的版本）原来每次
乘加都要调两次 `operator()` → `posOf()`，而 `posOf()` 会在栈上重建索引数组、
循环读取 `std::vector sizes`，编译器完全无法向量化。实测 **18.2 ns/MAC**。

修复分两步：

1. **扁平指针 + 把 stride 提到循环外**（对标量路径，对所有形状都生效）。
2. **接入 N-spirits 的 SIMD 内核**（见下一节），在 x86 上再快一截。

| 内核 | 最初 | 扁平指针标量 | +AVX2 内核 |
|------|------|--------------|------------|
| `ikkj (90x90)*(90x90)` | 13.274 ms（0.11 GFLOP/s） | 0.072 ms（20.4） | **0.053 ms（27.8）** |
| `ikkj (90x360)*(360x90)` | 53.016 ms（0.11） | 0.277 ms（21.1） | **0.157 ms（37.2）** |
| `kikj (360x90)^T*(360x90)` | 53.088 ms（0.11） | 0.274 ms（21.3） | 0.272 ms（21.4） |

这也是 chess 侧 4 个 RL agent 训练基准此前全部超时的原因（详见
`docs/issues_review.md` 的 B19）；注意它**不是**网络结构问题：把 DQN 主干换成小
8 倍的结构，`test_dqn` 仍然超过 1800 s。

### SIMD：用 N-spirits 的 SIMD 张量内核加速 `RL::Tensor`

来源（原样复制，未改动内核本体）：

| 项目内路径 | 来源 | 作用 |
|-----------|------|------|
| `src/rl/simd/sse2func.hpp` | `N-spirits/basic/simd/sse2func.hpp` | `simd::SSE2` 内核 |
| `src/rl/simd/avx2func.hpp` | `N-spirits/basic/simd/avx2func.hpp` | `simd::AVX2` 内核 |
| `src/rl/simd_ops.hpp` | 参照 `N-spirits/basic/simd/tensorsi.hpp` 写的分派层 | 把 `RL::Tensor_` 的热点运算分派到上面的内核 |

两个内核头文件唯一的改动是把 `#include "../basic_def.h"` 换成文件内自带的
`FORCE_INLINE` / `VECTORCALL` 宏定义（那个头文件不属于本项目，而且它额外声明的全局
`pi` 与 `RL::pi` 重复）。

**为什么是"把内核接进 `Tensor_`"而不是"把 `Tensor` 换成 `Tensorsi_`"**：
`Tensorsi_` 继承的是 N-spirits 自己的 `Tensor_`（存储用 `Buffer<T, aligned>` +
`mempool`，并且要求 32 字节对齐分配器），而 `RL::Tensor_` 是另一个已经分叉的实现
（`std::vector` 存储）。整类替换意味着把整套 N-spirits 张量栈搬进来、再逐处重新验证
RL 库的语义（几十个文件，且差异会**编译通过但行为不同**）。而 `tensorsi.hpp` 里真正
带来加速的是它调用的那些指令集内核，把它们接到现有 `Tensor_` 上，收益一样、API 不变。
另外实测确认：AVX2 内核用的是 `_mm256_loadu_ps`/`_mm256_storeu_ps`（非对齐），
只有没被使用的 `transpose` 需要对齐 —— 所以**不需要**替换存储分配器。

实测收益：

| 运算 | 标量 | AVX2 | 加速 |
|------|------|------|------|
| **GEMV `w*x`（矩阵乘列向量）** | 1.06 ns/MAC | **0.10 ns/MAC** | **10.5x** |
| `sum` / `max` / `norm2`（8100 元素） | ~1.00 ns/elem | ~0.12 ns/elem | **8.2x** |
| `ikkj (90x90)*(90x90)` | 0.072 ms | 0.053 ms | 1.36x |
| `ikkj (90x360)*(360x90)`（分块内核） | 0.277 ms | 0.157 ms | 1.76x |
| `a*b+c`（8100 元素） | 3.28 ns/elem | 3.13 ns/elem | 1.0x（受临时量分配限制） |

其中 **GEMV 那一行是这次改动的关键**：`Layer<Fn>::forward` 的 `o = w*x`、attention 的
`q = wq*x` / `k = wk*x` / `v = wv*x` 全都是"矩阵乘列向量"，而
`tensorsi.hpp` 的 `MatMul` 内核是沿 **z 的列方向**向量化的 —— z 只有一列时它的向量
循环长度为 0，`longEnough()` 也正是因此要求每一维都 >= step。所以这些最热的乘法在
上游设计下**根本用不到 SIMD**。`rl/simd_ops.hpp` 里因此补了一条 `gemv_ikkj()`：
按行与 x2 做点积，直接复用 `Instruct::dot`（`_mm256_fmadd_ps`）。实测 10.5x，
并且对 `w(360x90)`、`w(90x360)`、`w(128x64)`、`w(64x90)` 这些本库真实形状都成立。

数值正确性：`gemv` 相对误差 1.5e-7 ~ 5.3e-7（浮点重结合），长度小于一个向量宽度的
形状逐位相同（rel = 0），累加语义保持。

端到端（干净构建，见 `docs/issues_review.md` 的对照表）：

| 训练基准 | 最初 | 扁平指针标量 | +SIMD/GEMV |
|----------|------|--------------|------------|
| `test_pg` | 346 s | 26 s | **18 s** |
| `test_dqn` | >900 s 超时 | 536 s | **353 s** |
| `test_ppomcts` | >900 s 超时 | 233 s | **78 s** |
| `test_dqnmcts` | >900 s 超时 | 1299 s | **390 s** |

### 打开"系统 AVX2 加速"

用 SIMD 内核就必须面对"用哪一套指令集"这件事，而它是**编译期**决定的
（`__AVX2__` / `__SSE2__`），MSVC 在 x64 下默认只定义后者（准确地说：**两个都不定义**，
见下）。所以:

**1. 编译期开关**：`CHESS_ENABLE_AVX2`（默认 ON）→ 给相关目标加 `/arch:AVX2`。
是 **PUBLIC**：`tensor.hpp` 是头文件，所有实例化它内联算符的目标（chess 程序、各 agent
测试）必须用同一套指令集编译，否则那些翻译单元会退回标量路径。

**2. 配置阶段向系统确认**（`CMakeLists.txt` 里的 `try_run` 探测）：`/arch:AVX2` 会让
二进制**要求** CPU 支持 AVX2，拿到不支持的机器上直接非法指令崩溃，所以不能盲目打开。
配置时跑一次 CPUID（顺序与 `RL::cpuinfo` 一致：先看 AVX2 位，再确认 OS 已通过
OSXSAVE+AVX+XCR0 打开 YMM 状态），系统不支持就**自动不启用**并给出警告：

```
-- SIMD: this system reports AVX2 -> enabling /arch:AVX2 (AVX2 kernels)
-- SIMD: this system does NOT report AVX2 (or the OS has not enabled YMM state).
      /arch:AVX2 is therefore NOT applied: the binary would fault with an
      illegal instruction on this machine. The SSE2 kernels are used instead.
```

**3. 运行阶段能报告、能自检**（`src/rl/cpuinfo.hpp`）：`RL::cpuinfo::describe()` 返回
一行说明，`avx2Supported()` 是 CPUID 探测结果。它被接到三处：

* `test_rules` / 4 个 agent 测试的横幅：`SIMD: AVX2 kernels active (CPU supports AVX2)`
* `chess.exe` 的窗口标题：`中国象棋 - Qt/AI  [AVX2 kernels active (CPU supports AVX2)]`
* 启动日志一行 `[SIMD] ...`

注意这行字描述的是**当前这个二进制**（按翻译单元决定），不是整台机器：`test_rules`
只测棋规、不碰 `Tensor`，它报告 SSE2 是正常的。

**顺带修掉一个真 bug**：原来判据写成 `#elif defined(__SSE2__)`，而 **MSVC 在 x64 下
不定义 `__SSE2__`**（它只定义 `_M_X64` / `_M_IX86_FP`），于是 MSVC 构建里 SSE2 内核
根本选不上、直接掉到标量路径 —— 表现就是 `test_rules` 打出 "scalar kernels active"。
现在判据是 `__SSE2__ || _M_X64 || _M_AMD64 || _M_IX86_FP >= 2`，与上游
`tensorsi.hpp`"没有 AVX2 就用 SSE2"的假设一致。

三条路径实测（GEMV `w*x`，本机 i5-10400F）：

| 路径 | ns/MAC | 相对标量 |
|------|--------|----------|
| 标量回退 | 1.059 | 1.0x |
| SSE2 内核 | 0.179 | 5.9x |
| AVX2 内核 | 0.099 | **10.7x** |

`-DCHESS_ENABLE_AVX2=OFF` 仍然可用：构出的二进制走 SSE2 内核，能在没有 AVX2 的机器上
跑（已验证能正常构建、92 条规则断言全过），只是比 AVX2 慢约 1.8x。

### SIMD 之后梯度传播是否仍然正确（实测定论）

"SIMD 只改实现、不改数学"这句话不能靠读代码确认，所以加了 `test_grad`
（`test/test_grad_main.cpp`，已注册进 ctest）：用**中心差分**核对 `Net::backward`
填出来的解析梯度，并直接探测 MM 内核的语义。

覆盖范围：普通 MLP 的**全部**参数逐个查；DQN 的真实骨干
（`MOE<16,16>` + `TransformerBlock<16>` + `TanhNorm<Sigmoid>` + `Sigmoid` 输出，
见 `dqn.cpp:29`）抽样查，包含门控 `wg`、专家 FFN、LayerNorm 的 `gamma/beta`
（它们的梯度要经过 `variance`/`sum` 归约，正是 SIMD 覆盖的部分）。
形状跨越 8 元素这个"走不走 SIMD"的边界（8/15/16/22/32/62/90/128/512/1408/2048/32400）。

结论：**正确**。

```
A. MLP (全部参数)          eps=1e-2: 2.28e-05   eps=1e-3: 2.36e-04
B. 生产配置 (抽样)          eps=1e-2: 5.58e-03   eps=1e-3: 1.62e-04
```

即误差在 float 中心差分的噪声水平（~1e-4 相对）。B 组在 eps 缩小 10 倍后误差
**降到 1/34**，是标准的 O(eps²) 截断行为 —— 说明解析值就是差分收敛到的极限，
而不是"两边都错得一样"。这一点值得强调：eps=1e-2 时 MOE 门控 / LayerNorm 上的
偏差能到 5%，**那是差分本身的截断误差**（这些层三阶导很大），不是梯度错；
`test_grad` 里的 eps sweep 把最差元素逐 eps 打出来就是为了避免误判。

同时确认两件与"正确性"相关的**实现事实**：

1. **`ikjk` / `kijk` 的 SIMD 内核是赋值，标量是累加** —— 语义不一致。
   `test_grad` 的 C 部分把它量了出来（同一对操作数连调两次）：
   ```
   ikjk k=1  (列向量, 真实用法): got/expect = 1.000000  累加 -> 一致
   ikjk k=32 (命中 SIMD):       got/expect = 0.500000  **覆盖 -> 只留最后一次**
   ```
   库里所有 `ikjk`/`kijk` 调用点的 kdim 都是 1（`MM::ikjk(g.w, e, x)`，`e` 是
   ∂L/∂o 列向量），`mmShapeOk(..., kdim=1) = 0`，所以**当前走不到**这条路径 ——
   梯度检查也印证了这点。但它是个埋着的坑：多个调用点的注释明写 `+=` 累加
   （`attention.hpp:257`、`moe.hpp:212`、`ssm.cpp:156/169`、`lstm.cpp:143-148`
   连写 5 次累加到同一个 `delta.h`），**一旦有人改成多样本批量（kdim ≥ 8），
   这些梯度会静默只保留最后一次的贡献**。修法：SIMD 的 `ikjk`/`kijk` 改成
   `z[...] += dot(...)`，与标量语义对齐。

2. **SIMD 只加速了前向那一半**。`ikkj`（`o = w·x`）有专门的 GEMV 内核
   （`gemv_ikkj`）；`kikj`（`ei = wᵀ·e`，反向求输入梯度）**没有** —— 它的矩阵内核
   要求每一维都 ≥ 一个向量宽度，而 `ei` 是一列（zCol=1），判据不成立，于是掉回
   标量循环。实测（360×90，`test_grad` 的 D 部分）：
   ```
   ikkj (o = w·x):   3313 ns/次 = 0.102 ns/MAC
   kikj (ei = wᵀ·e): 32116 ns/次 = 0.991 ns/MAC   -> 反向是前向的 9.7x
   ```
   也就是说**训练步现在卡在没被优化的那一半**。补一个 `gemv_kikj`
   （`ei[i] += Σ_k w[k][i]·e[k]`，按列点积、行主序下是跨步访问）即可，
   与 `gemv_ikkj` 对称。

3. 顺带记录一个无害但要知道的后果：SIMD 改了归约的**加法顺序**，所以同一份代码
   在 AVX2 / SSE2 / 标量三种构建下梯度只到 ~1e-7 相对一致，**不再逐位相同**。
   长时间训练下这点差异会被放大（混沌），因此"跨构建比对权重文件是否相同"
   这类测试是不成立的，只能比指标。

4. **写自己的 MoE 时又做了一遍同样的核对**（`test_sparse_moe`，61 条断言）：稀疏
   路由只计算门控选中的 top-k 个专家，破坏了两个原本显然的性质（"前向用了所有
   专家"和"所有专家都有梯度"），所以逐条查：把未选中专家的权重改 0.5 → 输出
   **逐位不变**（`0.000e+00`）；`TopK==E` 时与上游 `MOE<3,4>` 的前向与门控梯度差
   也是 `0.000e+00`（等价性不是"0 比 0"的假通过，`|dL/dwg|=130.6`）；未选中专家的
   梯度**恰好为 0**；门控/专家参数/辅助损失全部通过中心差分（1e-3 相对误差量级）。
   这一轮顺带挖出的 `iFcLayer` 拷贝构造 bug 见 1.2 第 5 条。

### 权重文件格式 v2（`tensor.hpp` 的 `toString/fromString` + `net.hpp` 的 `save/load`）

chess 侧对 RL 内核做的一处**接口不变、行为升级**的改动（上游还是老的十进制文本格式）：

| | v1（上游） | v2（现在） |
|---|---|---|
| 张量编码 | `shape\|v1,v2,...` 十进制文本（`ostream << float`，6 位有效数字） | `shape\|b64:<crc32>:<base64 原始 float32>` |
| 精度 | **有损**：存读一次漂移 ~1e-6 相对 | **逐比特无损**（`test_weights` 断言两次存出的文件逐字节相同） |
| 体积 | 一个 float 9~13 字节 | 5.34 字节/float（base64 的理论下限是 5.33） |
| 文件头 | 无 | `CHWGT2 <层数> <结构指纹(FNV-1a of layer types)>`，载入时校验 |
| 校验 | 无 | 每张量 CRC32 + 长度必须等于形状乘积 × sizeof(T) |
| 写入 | 直接写目标文件 | 写 `.tmp` 再 `std::filesystem::rename` 原子替换 |
| 载入失败 | 静默载入半个模型（空张量 → 之后越界读） | **先零分配校验每一行**，全部通过才写进网络；失败时网络逐比特不变 |
| 兼容 | — | 不带 `CHWGT2` 头的老文件走兼容分支（`fromString` 同时认两种编码） |

回归：`test_weights`（44 条断言，ctest 里 0.2 秒）。细节与实测见
`docs/issues_review.md` §零之四 4.6。

### 数值 / 反向传播正确性
- `activate.h`：`Selu` 原来是 clamp 到 [-1,1] 的 hard-tanh，与 SELU 无关；改为真正的 SELU。
- `layer.h`：`Optimize::SGD(w, g.w, lr, true)` 的第 4 个参数是 **gamma（weight decay）**
  而不是 `clipGrad`，传 `true` 等于让 `w = (1-1)·w - lr·dw`，每步把权重清零。
- `layer.h` / `concat.hpp` / `attention.hpp` / `moe.hpp` / `transformer.hpp`：
  `MM::ikkj` 是**累加**语义，`forward()` 前必须先 `o.zero()`，否则第二次 forward
  会叠加旧输出。
- `concat.hpp`：`ScaledConcat` 的 backward 推导错误——子层误差应当是"softmax 之后"
  的切片（不是 `e` 的切片），且缺少 `w2^T·dz` 的输入通路。已按解析推导重写。
- `attention.hpp`：`PositionalEncoder::pe` 从未分配 → 越界写；`backward()` 原为空实现。
- `attention.hpp`：`MultiHeadAttention` 的 head 数 / `d_model` 一致性（见 1.2 第 1 条）。
- `attention.hpp`：`MHA::backward` 里 `ei += da` 把"拼接头输出的梯度"误当输入梯度
  而双重计入，已移除（输入梯度只来自各 head 的 backward）。
- `tensor.hpp`：`fromVector` 用 `std::copy` 把子张量**数值**写进 `shape`（越界 + 形状错误）；
  `flatten()` 返回 1 维形状，导致 `Conv2d::backward` 读 `sizes[2]` 越界（ConvPG/ConvDQN
  第一次训练即崩）→ 改为 `(totalSize, 1)`。
- `conv2d.hpp`：`forward` 改为从**真实输入**推导空间几何（原来按构造参数，输入不一致时
  静默裁剪或越界读）；`backward` 先算 `tanh'(o)⊙e` 再传播（原来用裸 `e`，丢掉 tanh' 因子）；
  通道数改为取 `o.shape[0]`（`Loss::MSE::df` 返回扁平张量时会被误判成通道数）；
  `SGD` 的 gamma 误用同上；`conv2d()` 内核改为扁平 stride（这是热路径，原来 `operator()`
  每次访问都构造索引数组）。
- `gru.cpp`：`feedForward` 里 `g` 门需要**全部** `r` 算完后再算（改成两遍循环）；
  `backward` 从"EMA 版 BPTT"换回标准 BPTT（原版本会把 `Uz` 的梯度写进 `delta.h`、
  用错 `delta.g`、`delta.r` 多乘一次 `delta.h`）。
- `lstm.cpp`：`outputError` 维度应为 `outputDim`；`delta.c` 的 `Tanh::df` 自变量漏了
  `Tanh::f`；`read()` 读权重时 `u`/`b` 用错字符串、遗忘门/输出门参数串味。
- `ssm.cpp`：`feedForward` 里 `A·h` 被第二次 `ikkj` 覆盖；`backward` 的 `h(-1)` 取成
  `h(0)`、`A^T` 被应用了两次。
- `mamba.cpp`：`δh` 的 `Ā` 衰减被重复施加；`dΔ/dz_Δ` 误用 `Sigmoid::f`（1.702 缩放的
  fast sigmoid）而不是标准 logistic；`C`/`W_in` 初始化尺度过小，使该层近似"无状态"。
- `net.hpp`：`backward()` 内部的 `inputGrad` 原来按 `layers[0]->o.totalSize` 分配成扁平
  `(N,1)` → 维度不等时 `Layer<Fn>::backward` 越界，conv-first 网络读 `sizes[2]` 越界。
  改为按输入 `x.shape` 分配并暴露为 `inputGrad`。
- `vae.hpp`：`decoder[0]->backward()` 在缓存被清零之后调用，恒得到 0，
  整条重建路径的梯度丢失（只剩 KL 在训练 encoder）；`encoder.backward` 传了新的零张量，
  把两个头累积的 `e2` 抹掉。
- `parameter.hpp`：`GradValue::clamp(c0, cn)` 的第二个参数原来被当作"越界时赋的值"，
  但命名为上界 → 改为 `clamp(c0, ci, cn)`（越界钳到 `ci`）。
- `util.cpp`：`gaussian()` 的归一化常数写成了 `sqrt(2·π·σ)` 且除以 `σ` 而不是 `2σ²`，
  根本不是高斯分布；`clip()` 的形参名 `sup/inf` 与实际语义相反。
- `util.hpp`：`entropy()` 未加下限；`noise()` 除以可能为 0 的 `max()` 产生 inf/NaN；
  `gumbelSoftmax` 的 1e-8 下限提到 1e-7；新增 `onehot()`。

### 时序 / 循环状态

- `dpg.cpp` / `mpg.cpp` / `drpg.cpp`：`action()` 每次调用都把循环状态从快照恢复，
  使得状态无法在两次调用之间传递（时序任务退化成"给 s1 映射一个固定动作"）。
  已去掉该恢复，并在 `reinforce1()` 结束时把状态写回快照。
- `drpg.h` 新增 `resetState()`；`mpg.h` 的 `resetState()` 同时清 `mamba->h`。

### 策略梯度 / 熵项

- `dpg.cpp` / `mpg.cpp` / `drpg.cpp` / `convpg.cpp`：熵项口径与符号修正
  （SAC 对偶目标 `J(α)=α(H−H₀)`，故 `dJ/dα = H−H₀`）、`advantage` 做均值基线 +
  标准差归一（并说明在 `clipGrad` 归一化下是整形不变式）、`1/π` 加下限。
- `trpo.cpp`：改用 GAE(λ=0.95)、重新启用 advantage 归一化、KL 分母加下限、
  步长补上 N 修正。

### 其它

- `dqn.cpp`：去掉每次更新目标网络时打印的 `std::cout<<"update target net"`。
- `convdqn.cpp`：Q 头由 `Sigmoid` 改为 `Linear`（蛇的奖励多为负，Sigmoid 结构上无法表达）；
  探索率衰减 0.99999 → 0.9995（原来要 ~23 万次 `learn()` 才降到下限，等于全程随机）。
  `eGreedyAction()` 改为对副本操作，避免把网络自己的输出缓存写坏。
- `sac.cpp`：目标 Q 改为对**全部动作**求期望（原来只取本步动作 `k` 再乘 `π(k)`，
  那不是 `V(s')`）；`actor.forward()` 返回的是引用，后续 forward 会覆盖它，已改为拷贝；
  alpha 梯度原来读的是已被清零的缓冲（恒得 H=0）；Q 网络输入补上 one-hot 动作。

> 注意：`dqn.cpp` 的 `Layer<Sigmoid>` Q 头**上游没有改**（只改了 `convdqn.cpp`）。
> chess 的 `DQNAgent` 奖励里含终局 ±1 与被吃红子产生的负值，
> Sigmoid 头 `(0,1)` 结构上无法表示 —— 这是一个**仍然存在**的缺陷，见
> `docs/issues_review.md`。

---

## 三、RL_CORE 构建组织

`CMakeLists.txt` 按 snakeAI 的方式重写：

| 项目 | 之前 | 现在 |
|------|------|------|
| rl 源码 | `file(GLOB rl/*.h rl/*.hpp rl/*.cpp)` 追加进 **6 个目标** | `add_library(RL_CORE STATIC ...)`，**只编译一次** |
| 源文件列表 | GLOB（新增文件需重新 configure） | 显式 `RL_SOURCES` 列表 |
| Qt 代码生成 | `CMAKE_AUTOMOC/AUTOUIC/AUTORCC ON` 全局 | 只有 `chess`（GUI）目标开启，`RL_CORE` 与测试全部 OFF |
| Qt 依赖 | rl 与 Qt 混在同一目标 | `RL_CORE` 的编译行**不含任何 Qt include**（只有 `${SRC_DIR}`） |
| 头文件依赖 | 无（本机 MSVC 输出本地化的 `注意: 包含文件:`，CMake 记录的探测前缀对不上，ninja 抓不到任何头依赖） | `OBJECT_DEPENDS` 显式声明 rl 头文件依赖，改 `rl/layer.h` 会真的重编 |
| 大对象 | 无 | `RL_CORE` PUBLIC `/bigobj`（`tensor.hpp`/`layer.h`/`attention.hpp`/`moe.hpp` 模板实例化量很大） |
| 测试注册 | 无 | `enable_testing()` + `add_test()`，`ctest -N` 可见 6 个测试 |
| CMake 版本 | 3.5（已弃用告警） | 3.16 |
| `res.qrc` | 虽设了 AUTORCC 但 `.qrc` 未被列为源文件 → 从未编译 | 已列入 `chess` 源文件，`qrc_res.cpp` 真正生成 |
| `appstyle.qss` | 未被引用 | `res.qrc` 已接入构建（注意 `main.cpp` 仍未 `setStyleSheet`，见问题报告） |

编译单元数：**158 → 81**（rl 的 17 个 TU 由 5 个目标各编一遍变为只编一遍）。

构建与验证命令：

```bat
:: 全量构建（必须先 vcvars64，否则 cl 找不到标准库头）
build_main.bat

:: 只跑测试
"C:\Qt\Tools\CMake_64\bin\ctest.exe" --test-dir build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release --output-on-failure
```

可选项：

- `-DCHESS_BUILD_TESTS=OFF`：不构建 6 个测试目标。
- `-DCHESS_USE_PCH=ON`：对 `RL_CORE`（和 `chess`）预编译 `rl/tensor.hpp` + `rl/layer.h`，
  默认关闭，需先在本地工具链上复验。
