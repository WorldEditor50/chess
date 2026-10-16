<#
  sweep_expert_width.ps1 - 扫「序列 Transformer 专家」的两个宽度 (2026-10)

  为什么要有这个脚本 (而不是手敲命令):
    * 一次搜索是 **几十个进程** (agent 数 x hiddenDim 档 x tokDim 档), 手敲必然出现
      "有两臂的协议不一样" 这种**读数不可比**的错 —— 而它从数字上看不出来;
    * 每个进程打出来的是**人读的报告**, 汇总表必须由**同一处解析**产生, 否则
      "哪一行是留出 CE" 这件事会有两种说法 (本工程反复记过这类事故)。

  口径 (与 `train_bc` 的报告行逐字对应, 见 src/bcrun.hpp 的 formatReport):
    * 留出 CE   = "[4] 克隆后" 块里 `留出 n=… CE …` 那一列 (越低越好);
    * 留出 top-1 = 同一行的 `prior top-1 …%`;
    * KL 口径   = `CE − H(t)`, 软目标下**唯一**能与 one-hot 口径比的那一列;
    * 总耗时    = 进程内墙钟 (`总耗时 … s`) —— ⚠ 只在**同一轮、同一并发度**下可比,
                  见 docs/seq_transformer_design.md §8.4 的计时纪律。

  用法示例 (PPO 支, 2D 网格 3x3):
    pwsh -File tools/sweep_expert_width.ps1 -Exe build/sweep/train_bc.exe `
         -Agent ppo-seq -AgentArgs '--pos=2d' -Tag ppo `
         -Hidden 32,64,128 -Expert 32,64,128 -Positions 300 -Epochs 8 -Jobs 5

  输出:
    <OutDir>/<Tag>_h<H>_e<E>.txt     每个臂的完整报告 (原样保存, 不许只留汇总)
    <OutDir>/<Tag>_summary.csv       汇总 (由日志解析出来, 可复核)
    stdout                           同一份汇总的人读表格
#>
[CmdletBinding()]
param(
    [string]   $Exe        = "build/sweep/train_bc.exe",
    [string]   $Agent      = "ppo-seq",
    [string]   $AgentArgs  = "",
    [string]   $Tag        = "sweep",
    [int[]]    $Hidden     = @(64),
    [int[]]    $Expert     = @(64),
    [int]      $Positions  = 300,
    [int]      $Depth      = 4,
    [int]      $SoftDepths = 4,
    [int]      $Epochs     = 8,
    [int]      $Batch      = 64,
    [double]   $Lr         = 0.002,
    [int]      $Seed       = 20240901,
    [int]      $Holdout    = 5,
    [int]      $Jobs       = 5,
    [string]   $OutDir     = ".r1build/sweep",
    [switch]   $Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$exePath  = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $repoRoot $Exe }
if (-not (Test-Path $exePath)) { throw "找不到可执行文件: $exePath" }
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repoRoot $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-ArmLog([string]$tag, [int]$h, [int]$e) { Join-Path $OutDir ("{0}_h{1}_e{2}.txt" -f $tag, $h, $e) }

# ---- 1) 造命令 ---------------------------------------------------------------
$arms = @()
foreach ($h in $Hidden) {
    foreach ($e in $Expert) {
        $log = Get-ArmLog $Tag $h $e
        if ((Test-Path $log) -and -not $Force) {
            $arms += [pscustomobject]@{ H = $h; E = $e; Log = $log; Skip = $true }
            continue
        }
        $args = @(
            "--agent=$Agent", "--hidden=$h", "--expert=$e",
            "--positions=$Positions", "--depth=$Depth", "--soft=1",
            "--soft-depths=$SoftDepths", "--epochs=$Epochs", "--batch=$Batch",
            "--lr=$Lr", "--seed=$Seed", "--holdout=$Holdout"
        )
        if ($AgentArgs -ne "") { $args += ($AgentArgs -split '\s+' | Where-Object { $_ -ne "" }) }
        $arms += [pscustomobject]@{ H = $h; E = $e; Log = $log; Skip = $false; Args = $args }
    }
}

# ---- 2) 跑 (滑动窗口并发; 读数与并发度无关, 但**墙钟有关**) -------------------
$todo = @($arms | Where-Object { -not $_.Skip })
Write-Host ("=== 扫 {0}: {1} 个臂 ({2} 个已有日志跳过), 并发 {3}, 协议 positions={4} epochs={5} batch={6} lr={7} seed={8} depth={9}(soft {10}) ===" -f `
    $Tag, $arms.Count, ($arms.Count - $todo.Count), $Jobs, $Positions, $Epochs, $Batch, $Lr, $Seed, $Depth, $SoftDepths)
$t0 = Get-Date
$running = @()
foreach ($arm in $todo) {
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Jobs) {
        Start-Sleep -Milliseconds 500
        $running = @($running | Where-Object { -not $_.HasExited })
    }
    Write-Host ("  -> h={0} e={1}: 起 (已跑 {2:N1} min)" -f $arm.H, $arm.E, ((Get-Date) - $t0).TotalMinutes)
    $p = Start-Process -FilePath $exePath -ArgumentList $arm.Args -WorkingDirectory $repoRoot `
                       -RedirectStandardOutput $arm.Log -RedirectStandardError ($arm.Log + ".err") `
                       -PassThru -NoNewWindow
    $running += $p
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) {
    Start-Sleep -Milliseconds 500
    $running = @($running | Where-Object { -not $_.HasExited })
}
Write-Host ("=== 全部结束: {0:N1} min ===" -f ((Get-Date) - $t0).TotalMinutes)

# ---- 3) 解析 (只认 [4] 克隆后 那一段的行; 这是报告里唯一的那一行) --------------
function Parse-Arm([string]$log) {
    $o = [ordered]@{ TrainTop1 = $null; TrainCE = $null; HoldTop1 = $null; HoldCE = $null;
                     PTeacher = $null; HoldKL = $null; Sec = $null; Label = $null; Pos = $null }
    if (-not (Test-Path $log)) { return [pscustomobject]$o }
    $lines = Get-Content -LiteralPath $log -Encoding UTF8
    $seenClone = $false
    foreach ($ln in $lines) {
        if ($ln -match '^\[4\] 克隆后') { $seenClone = $true; continue }
        if (-not $seenClone) { continue }
        if ($ln -match '^\s*训练\s+n=(\d+)\s+prior top-1\s+([\d.]+)%\s+P\(老师着法\)\s+([\d.]+)\s+CE\s+([\d.]+)') {
            $o.TrainTop1 = [double]$Matches[2]; $o.TrainCE = [double]$Matches[4]
        } elseif ($ln -match '^\s*留出\s+n=(\d+)\s+prior top-1\s+([\d.]+)%\s+P\(老师着法\)\s+([\d.]+)\s+CE\s+([\d.]+)') {
            $o.HoldTop1 = [double]$Matches[2]; $o.PTeacher = [double]$Matches[3]; $o.HoldCE = [double]$Matches[4]
        } elseif ($ln -match '留出\s+[\d.]+ -> ([\d.]+)\s+\(train/留出 KL 差') {
            $o.HoldKL = [double]$Matches[1]
        } elseif ($ln -match '^样本\s+:\s+.*') { $o.Pos = $ln.Trim() }
    }
    # 标签行与总耗时在 [4] 之外, 单独找
    foreach ($ln in $lines) {
        if ($ln -match '^标签\s+:\s+(.+)$' -and $o.Label -eq $null) { $o.Label = $Matches[1].Trim() }
        if ($ln -match '总耗时\s+([\d.]+)\s*s') { $o.Sec = [double]$Matches[1] }
    }
    return [pscustomobject]$o
}

$rows = @()
foreach ($arm in $arms) {
    $p = Parse-Arm $arm.Log
    $rows += [pscustomobject]@{
        Tag = $Tag; Agent = $Agent; Hidden = $arm.H; Expert = $arm.E
        HoldCE = $p.HoldCE; HoldTop1 = $p.HoldTop1; HoldKL = $p.HoldKL; PTeacher = $p.PTeacher
        TrainCE = $p.TrainCE; TrainTop1 = $p.TrainTop1; Sec = $p.Sec; Log = (Split-Path -Leaf $arm.Log)
    }
}
$csv = Join-Path $OutDir ("{0}_summary.csv" -f $Tag)
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8

# ---- 4) 人读表格 (按留出 CE 升序; CE 并列时按 top-1 降序) ---------------------
$sorted = $rows | Sort-Object -Property @{E={ if ($null -eq $_.HoldCE) { [double]::MaxValue } else { $_.HoldCE } }}, @{E={ -1 * $_.HoldTop1 }}
Write-Host ""
Write-Host ("== {0} 汇总 (按留出 CE 升序) ==" -f $Tag)
$sorted | Format-Table Hidden, Expert, HoldCE, HoldTop1, HoldKL, PTeacher, TrainCE, Sec, Log -AutoSize | Out-String -Width 200 | Write-Host
Write-Host ("CSV: {0}" -f $csv)
