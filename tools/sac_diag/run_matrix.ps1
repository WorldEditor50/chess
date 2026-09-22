param(
    [string]$Phase = '1c',
    [int]$MaxParallel = 6,
    [string]$OutDir = ''
)

# ============================================================================
# run_matrix.ps1 - SAC critic/α 诊断与奖励塑形 A/B 的整批实验 (2026-09)
#
# 为什么要有它: 本轮结论全部来自"20 局/档 x 多个种子"的**配对**实验 (同 seed +
# 同 --mcts-srand), 手工一条条敲既容易漏参数, 也容易把两个不同的随机流放在一起比。
# 脚本把 phase / 并行度 / 输出目录都显式化, 并把**二进制的 sha256** 打进输出头部
# (本轮踩过"改了源码但二进制没重编"⇒ 得到"改了没差别"的假结论, 见
#  docs/session_2026_09_sac.md §4 第 5 条)。
#
# Phase 一览 (细节与结论见 docs/sac_critic_diagnosis_2026_09.md):
#   1  : ① 诊断三档 + clampTarget x huberDelta 扫描 + 两个干预 + ② 还原版开约束
#   1b : 第三种子上的机制复现 (带目标网探针的二进制), 6 个 run
#   1c : "得分率由搜索叶子值的常数决定, 与学习无关" 的直接检验 (--no-train x --value-scale)
#   1d : 1c 再加 rewardShape => "叶子值常数会不会把终局信号淹掉" 的交互检验
#   2  : ③ 奖励塑形配对复测 (4 种子 x 50 局 x {shape=0,2} x {新实现, 还原版})
#
# WARNING (2026-09, 一次会话里踩了两次): 下面这些中文注释**要求本文件存成 UTF-8 with BOM**。
# 没有 BOM 时 PowerShell 5.1 按 ANSI 读, 一个多字节注释会吃掉下一行 (或直接报
# "Unexpected token '}'"), 脚本还没跑到第一步就死了。注意**编辑/重写文件的工具可能把 BOM
# 丢掉** —— 每次改完都要验证:
#     [System.IO.File]::ReadAllBytes($f)[0..2]   ->  239 187 191
#     $e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$null,[ref]$e); $e.Count  ->  0
# ============================================================================

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$exe = "$root\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\bench_sac_learn.exe"
if ($OutDir -eq '') { $OutDir = "$root\build\exp\phase$Phase" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$out = $OutDir

$common = '--plies=200 --mcts-sims=200 --sims=256 --pre-train=64 --mcts-srand=12345'
$seeds2 = @(20240901, 777)
$seeds4 = @(20240901, 777, 31337, 424242)

$runs = New-Object System.Collections.ArrayList
function Add-Run([string]$label, [int]$games, [string]$extra) {
    [void]$runs.Add([pscustomobject]@{ label = $label; games = $games; extra = $extra })
}

if ($Phase -eq '1' -or $Phase -eq 'all') {
    foreach ($s in $seeds2) {
        # (1a) 诊断: 当前默认 / 改前的 α 口径 / 还原版 (每局一行诊断打在日志里)
        Add-Run "diag-new-s$s"     20 "--no-sparse-leaf --seed=$s --label=diag-new-s$s"
        Add-Run "diag-olda-s$s"    20 "--no-sparse-leaf --seed=$s --label=diag-olda-s$s --entropy-ratio=0.5 --alpha-lr=0.005"
        Add-Run "diag-legacy-s$s"  20 "--legacy --seed=$s --label=diag-legacy-s$s"
        # (1b) clampTarget x huberDelta 扫描 (固定 α 口径)
        foreach ($c in 0, 2, 3, 4) {
            foreach ($h in 0, 1) {
                Add-Run "sweep-c$c-h$h-s$s" 20 "--no-sparse-leaf --clamp=$c --huber=$h --seed=$s --label=sweep-c$c-h$h-s$s"
            }
        }
        # 稀疏叶子对照 (类默认口径), 只做参考那一档
        Add-Run "bridge-sparse-c2-h1-s$s" 20 "--seed=$s --label=bridge-sparse-c2-h1-s$s"
        # (1c) 两个干预: 熵项不进目标 / 目标熵分母换成槽位数
        Add-Run "inv-noent-s$s"  20 "--no-sparse-leaf --entropy-in-target=0 --seed=$s --label=inv-noent-s$s"
        Add-Run "inv-slots-s$s"  20 "--no-sparse-leaf --entropy-slots --seed=$s --label=inv-slots-s$s"
        Add-Run "inv-both-s$s"   20 "--no-sparse-leaf --entropy-in-target=0 --entropy-slots --seed=$s --label=inv-both-s$s"
        # (2) 还原版打开约束
        Add-Run "legacy-c2-h1-s$s" 20 "--legacy --clamp=2 --huber=1 --seed=$s --label=legacy-c2-h1-s$s"
        Add-Run "legacy-c4-h1-s$s" 20 "--legacy --clamp=4 --huber=1 --seed=$s --label=legacy-c4-h1-s$s"
    }
}

if ($Phase -eq '1b' -or $Phase -eq 'all') {
    # 第三个种子上的机制复现: (a) 目标网探针读数, (b) 给主结论加第三个种子。
    $s = 31337
    Add-Run "diag-new-s$s"    20 "--no-sparse-leaf --seed=$s --label=diag-new-s$s"
    Add-Run "diag-olda-s$s"   20 "--no-sparse-leaf --seed=$s --label=diag-olda-s$s --entropy-ratio=0.5 --alpha-lr=0.005"
    Add-Run "diag-legacy-s$s" 20 "--legacy --seed=$s --label=diag-legacy-s$s"
    Add-Run "inv-noent-s$s"   20 "--no-sparse-leaf --entropy-in-target=0 --seed=$s --label=inv-noent-s$s"
    Add-Run "inv-slots-s$s"   20 "--no-sparse-leaf --entropy-slots --seed=$s --label=inv-slots-s$s"
    Add-Run "legacy-c2-h1-s$s" 20 "--legacy --clamp=2 --huber=1 --seed=$s --label=legacy-c2-h1-s$s"
}

if ($Phase -eq '1c' -or $Phase -eq 'all') {
    # "得分率是被搜索叶子值的**常数**决定的, 与学习无关" 这条假设的直接检验。
    # --no-train = 一次都不学 (随机权重, 策略恒为均匀先验, critic 恒为随机网);
    # --value-scale 只乘**搜索**的叶子值 (不动学习侧目标) => 唯一变量就是叶子值常数:
    #   0 -> 叶子值 0 (只剩终局探测 ±1) = "教科书口径"
    #   1 -> 默认 (约 0.07 + 0.2*H ≈ 0.8)
    #   4 -> 常数约 3 (与"钳位饱和"那一档同量级)
    foreach ($s in $seeds2) {
        foreach ($v in 0, 1, 4) {
            Add-Run "vs$v-nolearn-s$s" 20 "--no-sparse-leaf --no-train --value-scale=$v --seed=$s --label=vs$v-nolearn-s$s"
        }
    }
}

if ($Phase -eq '1d' -or $Phase -eq 'all') {
    # 1d: "终局信号 vs 叶子值常数" —— ③ 的塑形效果为什么在两个 agent 上不一样?
    # 假设: 叶子值里的那个**常数**(αH, 由 value-scale 直接控制) 会把终局 ±1..2 淹掉,
    # 所以"同样的塑形"在常数小的时候有用、常数大的时候没用。
    # 做法: --no-train 固定策略与 critic (随机权重), 只扫 (valueScale x rewardShape)。
    # 预测: shape=2 相对 shape=0 的收益随 valueScale 变大而**缩小**。
    foreach ($s in $seeds2) {
        foreach ($v in 0, 1, 4) {
            foreach ($k in 0, 2) {
                Add-Run "shape-vs$v-k$k-s$s" 20 "--no-sparse-leaf --no-train --value-scale=$v --reward-shape=$k --seed=$s --label=shape-vs$v-k$k-s$s"
            }
        }
    }
}

if ($Phase -eq '3' -or $Phase -eq 'all') {
    # [F1] 目标网同步率 —— 本轮定位到的主缺陷, 也是唯一"只改一个数"的一档。
    # 基线臂**复用 phase 2 的 shape-new-k0**(同协议: --no-sparse-leaf, plies=200,
    # mcts-sims=200, sims=256, pre-train=64, 同 4 个种子, 200 局/臂) —— 默认路径的行为
    # 已验证与改动前逐位相同, 所以那批可以直接当配对基线, 不必重跑。
    # 每臂 4 种子 x 50 局 = 200 局:
    #   f1-t01-i08 : tau=0.01 每 8 步  (新默认候选: 2600 步移动 96%)
    #   f1-t05-i08 : tau=0.05 每 8 步  (更快)
    #   f1-hard-i256 : tau=1 每 256 步 (硬拷贝对照)
    #   f1-f2      : 新 tau + 熵项不进目标 (F2)
    #   f1-f2-f3   : 再加目标熵分母换槽位 (F3)
    foreach ($s in $seeds4) {
        Add-Run "f1-t01-i08-s$s"   50 "--no-sparse-leaf --target-tau=0.01 --target-iter=8 --seed=$s --label=f1-t01-i08-s$s"
        Add-Run "f1-t05-i08-s$s"   50 "--no-sparse-leaf --target-tau=0.05 --target-iter=8 --seed=$s --label=f1-t05-i08-s$s"
        Add-Run "f1-hard-i256-s$s" 50 "--no-sparse-leaf --target-tau=1 --target-iter=256 --seed=$s --label=f1-hard-i256-s$s"
        Add-Run "f1-f2-s$s"        50 "--no-sparse-leaf --target-tau=0.01 --target-iter=8 --entropy-in-target=0 --seed=$s --label=f1-f2-s$s"
        Add-Run "f1-f2-f3-s$s"     50 "--no-sparse-leaf --target-tau=0.01 --target-iter=8 --entropy-in-target=0 --entropy-slots --seed=$s --label=f1-f2-f3-s$s"
    }
}

if ($Phase -eq '2' -or $Phase -eq 'all') {
    # ③ 奖励塑形: 200 局/档 (4 种子 x 50 局), 两个 agent 都跑, 两臂同副牌。
    foreach ($s in $seeds4) {
        foreach ($k in 0, 2) {
            Add-Run "shape-new-k$k-s$s"    50 "--no-sparse-leaf --reward-shape=$k --seed=$s --label=shape-new-k$k-s$s"
            Add-Run "shape-legacy-k$k-s$s" 50 "--legacy --reward-shape=$k --seed=$s --label=shape-legacy-k$k-s$s"
        }
    }
}

$hash = (Get-FileHash $exe -Algorithm SHA256).Hash
Write-Output ("=== run_matrix phase={0} parallel={1} runs={2} ===" -f $Phase, $MaxParallel, $runs.Count)
Write-Output ("exe : {0}" -f $exe)
Write-Output ("      mtime={0} sha256={1}" -f (Get-Item $exe).LastWriteTime, $hash)
Write-Output ("out : {0}" -f $out)

$queue = New-Object System.Collections.Queue
foreach ($r in $runs) { $queue.Enqueue($r) }
$active = @{}
$t0 = Get-Date

while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($active.Count -lt $MaxParallel -and $queue.Count -gt 0) {
        $r = $queue.Dequeue()
        $log = "$out\$($r.label).log"
        $err = "$out\$($r.label).err"
        $csv = "$out\$($r.label).csv"
        $a = "--games=$($r.games) $common $($r.extra) --csv=$csv"
        # 走 cmd 的重定向: 直接字节落盘 (工具的 stdout 是 UTF-8, 用 PS 的管道会被
        # 按控制台代码页解码再编码, 中文日志就成乱码了)。
        $p = Start-Process -FilePath 'cmd.exe' -NoNewWindow -PassThru `
             -ArgumentList "/c `"`"$exe`" $a > `"$log`" 2> `"$err`"`""
        $active[$p.Id] = @{ proc = $p; label = $r.label; start = (Get-Date) }
        Write-Output ("[launch] {0} pid={1} running={2}" -f $r.label, $p.Id, $active.Count)
    }
    Start-Sleep -Seconds 5
    foreach ($id in @($active.Keys)) {
        if ($active[$id].proc.HasExited) {
            $e = $active[$id]
            $sec = ((Get-Date) - $e.start).TotalSeconds
            $ok = 'FAIL'
            if (Test-Path "$out\$($e.label).csv") { $ok = 'OK' }
            Write-Output ("[done]   {0} {1} {2:N0}s  (elapsed total {3:N0}s)" -f `
                          $e.label, $ok, $sec, ((Get-Date) - $t0).TotalSeconds)
            $active.Remove($id)
        }
    }
}

Write-Output ("=== all runs finished in {0:N0}s ===" -f ((Get-Date) - $t0).TotalSeconds)
Write-Output ("summary: powershell -NoProfile -File {0}\summary.ps1 -Dir {1}" -f $PSScriptRoot, (Split-Path $out -Leaf))
