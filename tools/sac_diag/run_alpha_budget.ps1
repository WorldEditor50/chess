param(
    [int]$MaxParallel = 6,
    [string]$OutDir = ''
)

# ============================================================================
# run_alpha_budget.ps1 - "alpha 无限制增长" 的判据实验 (2026-10)
#
# 要回答的两个问题:
#   1. alpha 为什么一直涨 -> 目标熵 Hbar 的分母 (合法着法数) 高于 H 的上界
#      (H <= log(合法槽位数) < log(合法着法数)), 梯度 (H - Hbar) 恒为负,
#      控制器没有不动点。把分母换成合法槽位数 (--entropy-slots) 之后呢?
#   2. 若把上界按 **alpha*H** 定 (--alpha-h-budget), alpha 会不会停在中间、
#      熵项在 TD 目标里占的比例会不会降下来, 而棋子力不塌?
#
# 判据 (不看 alpha 的绝对值, 看这三条):
#   * alpha 是否停在中间 (不是一路贴边界);
#   * 每局的 "alphaH = |y| 的 x%" 是否落到值域量级内 (而不是 99%);
#   * Qspread 与得分率有没有跟着塌 (压 alpha 不是免费的)。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File tools\sac_diag\run_alpha_budget.ps1
# 输出: build\exp\alpha_budget\<label>.log  (每局一行 [1] 诊断)
#
# 注意: 本文件必须保持 **ASCII only** —— 仓库在 PowerShell 5.1 上踩过
# "无 BOM 的中文注释吃掉下一行" 的坑 (见 tools\sac_diag\run_matrix.ps1 头部)。
# ============================================================================

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$exe = "$root\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\bench_sac_learn.exe"
if (-not (Test-Path $exe)) { throw "missing exe: $exe" }
if ($OutDir -eq '') { $OutDir = "$root\build\exp\alpha_budget" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$common = '--plies=200 --mcts-sims=200 --sims=256 --pre-train=64 --mcts-srand=12345 --no-sparse-leaf --games=20'
$seeds = @(20240901, 777)

# label -> extra flags.  IMPORTANT (2026-10, after the default change): `alphaHBudget`
# now defaults to 3.0, so the "old caliber" arm must pass `--alpha-h-budget=0`
# explicitly; the plain `default` arm is what the agent does today.
# `old-hb0` reproduces docs/sac_critic_diagnosis_2026_09.md 的 diag-new 那一档
# (alpha 0.20 -> 2.62, |y| 7.70, 被夹 99% 的 20 局口径下是 71%) --
# 它是"改动前口径仍然可复现"的回归检查。
$arms = @(
    @{ n = 'old-hb0';   x = '--alpha-h-budget=0' },
    @{ n = 'default';   x = '' },
    @{ n = 'ahb2-c1';   x = '--alpha-h-budget=2.0 --alpha-ceiling=1.0' },
    @{ n = 'slots098';  x = '--entropy-slots --alpha-h-budget=0' },
    @{ n = 'slots099';  x = '--entropy-slots --entropy-ratio=0.99 --alpha-h-budget=0' },
    @{ n = 'slots-bnd'; x = '--entropy-slots' },
    @{ n = 'slots100';  x = '--entropy-slots --entropy-ratio=1.0' }
)

$runs = New-Object System.Collections.ArrayList
foreach ($a in $arms) {
    foreach ($s in $seeds) {
        [void]$runs.Add([pscustomobject]@{
            label = "$($a.n)-s$s"
            args  = "$common --seed=$s --label=$($a.n)-s$s $($a.x)"
        })
    }
}

Write-Host "binary sha256: $((Get-FileHash $exe -Algorithm SHA256).Hash)"
Write-Host "runs: $($runs.Count), parallel: $MaxParallel, out: $OutDir"

$i = 0
while ($i -lt $runs.Count) {
    $batch = @()
    for ($k = 0; $k -lt $MaxParallel -and $i -lt $runs.Count; $k++, $i++) {
        $r = $runs[$i]
        $log = Join-Path $OutDir "$($r.label).log"
        $err = Join-Path $OutDir "$($r.label).err"
        Write-Host "  start $($r.label)"
        $batch += Start-Process -FilePath $exe -ArgumentList $r.args -NoNewWindow -PassThru `
                                -RedirectStandardOutput $log -RedirectStandardError $err
    }
    foreach ($p in $batch) { $p.WaitForExit() }
    Write-Host "  batch done ($($i)/$($runs.Count))"
}
Write-Host "all done -> $OutDir"
