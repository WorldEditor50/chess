<#
.SYNOPSIS
    SAC 走法序列的黄金基准回归检查 (不依赖 59e5233 worktree)。

.DESCRIPTION
    `tools/verify_sac_equiv_59e5233.ps1` 证明的是"当前 SAC 与改前 59e5233 逐手等价",
    但它需要那棵基线 worktree 和它自己的构建产物。本脚本把**首次等价验证时得到的**
    走法序列固化在 `test/golden/` 里, 于是以后任何改动只要动了 SAC 的行为, 这里当场变红
    —— 不需要基线树, 也能保证"等价"这件事不被后续改动悄悄破坏。

    基准是怎么来的 (2026-09): 在 `--mcts-srand=12345` (对手随机流固定) 下,
    当前树与 59e5233 各导一份走法序列, 逐字节 diff **完全相同** (258 / 300 / 61 行),
    于是把当前树那一份存成基准。它同时代表两版的行为。

    判据: 逐行 diff。失败时打印**第一处**差异及其行号 —— 因为一处走法不同会让后面
    整局都岔开, 只报"第 300 行不同"没有诊断价值。

    退出码 0 = 与基准逐手相同; 1 = 有差异; 2 = 前置条件缺失。
#>
param(
    [string]$Build = 'build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$exe  = Join-Path $root "$Build\bench_sac_mcts_min.exe"
$gold = Join-Path $root 'test\golden'
$out  = Join-Path $root 'build\golden_out'
New-Item -ItemType Directory -Force -Path $out | Out-Null

if (-not (Test-Path $exe)) { Write-Host "[错误] 找不到: $exe" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $gold)) { Write-Host "[错误] 找不到基准目录: $gold" -ForegroundColor Red; exit 2 }

# 用例: 名字 -> (骨干, 局数, 手数, 模拟, 种子)
$cases = @(
    @{ file='sac_moe_mlp_seed20240901.txt'; backbone='moe-mlp'; games=4; plies=80; sims=64; seed=20240901 },
    @{ file='sac_tb_seed20240901.txt';        backbone='tb';       games=2; plies=30; sims=16; seed=20240901 }
)

$failed = 0
foreach ($c in $cases) {
    $ref = Join-Path $gold $c.file
    $got = Join-Path $out  $c.file
    if (-not (Test-Path $ref)) { Write-Host "  [跳过] 基准缺失: $($c.file)" -ForegroundColor Yellow; continue }
    & $exe --games=$($c.games) --sims=$($c.sims) --plies=$($c.plies) --opening=4 `
           --seed=$($c.seed) --mcts-srand=12345 --backbone=$($c.backbone) --dump-moves=$got 2>&1 | Out-Null
    $a = Get-Content $ref; $b = Get-Content $got
    if ($a.Count -ne $b.Count) {
        Write-Host ("  [{0}] **行数不同**: 基准 {1} 行, 实测 {2} 行" -f $c.file, $a.Count, $b.Count) -ForegroundColor Red
        $failed++
        continue
    }
    $firstDiff = -1
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -ne $b[$i]) { $firstDiff = $i; break }
    }
    if ($firstDiff -lt 0) {
        Write-Host ("  [{0}] {1} 行  与基准逐手相同" -f $c.file, $a.Count) -ForegroundColor Green
    } else {
        Write-Host ("  [{0}] **第一处差异在第 {1} 行**" -f $c.file, ($firstDiff + 1)) -ForegroundColor Red
        Write-Host ("      基准: {0}" -f $a[$firstDiff])
        Write-Host ("      实测: {0}" -f $b[$firstDiff])
        $failed++
    }
}

Write-Host ""
if ($failed -eq 0) {
    Write-Host "VERDICT: PASS | SAC 走法序列与黄金基准一致 (等价行为未被破坏)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "VERDICT: FAIL | $failed 个用例偏离基准 —— 说明 SAC 的行为被改动了。" -ForegroundColor Red
    Write-Host "  若这是**有意的**行为改动, 请同时更新 test/golden/ 下的基准, 并在文档里写明理由。"
    exit 1
}
