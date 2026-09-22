<#
.SYNOPSIS
    判定"当前 SAC agent 是否与提交 59e5233 的 SAC agent **逐手等价**"。

.DESCRIPTION
    为什么不能只看比分: 比分只能给出**统计等价** (600 局配对下来两者区间重叠),
    给不了"同一局面下走同一步"。而这个 agent 的改动大多是"等价重写"
    (maskToBits 数组化、stepToActionIdx 带 color、contextOf 走量纲枚举、
     softValueFrom 加 valueScale、新增稀疏入口...), 这些改动的正确性判据就是
    **同一输入必须给出同一走法**。

    做法: 同一份 `bench_sac_mcts_min` 源码编到两棵树上 (当前树 + 59e5233 worktree),
    给两版**同一个 --mcts-srand** (MCTS 对手内部用 std::rand + std::srand(time()),
    不固定就等于每跑一次换一副牌), 各导出一份走法序列, 然后逐字节 diff。

.NOTES
    2026-09 实测结果 (本脚本跑出来的):
      moe-mlp 2 局 x 40 手  -> 81 行   完全相同
      moe-mlp 4 局 x 80 手  -> 258 行  完全相同
      moe-mlp 4 局 x 80 手  -> 300 行  完全相同 (换种子)
      tb      2 局 x 30 手  -> 61 行   完全相同
    退出码 0 = 全部逐手相同; 1 = 有差异 (或前置条件缺失)。
#>
param(
    [string]$Build   = 'build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release',
    [string]$BaseWorktree = 'E:\home\code\chess_base_59e5233',
    [string]$BaseBuild   = 'build-min',
    [int[]] $Seeds   = @(20240901, 777),
    [int]   $MctsSrand = 12345
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot          # 仓库根
$cur  = Join-Path $root "$Build\bench_sac_mcts_min.exe"
$base = Join-Path $BaseWorktree "$BaseBuild\bench_sac_mcts_min.exe"
$out  = Join-Path $root 'build\equiv_out'
New-Item -ItemType Directory -Force -Path $out | Out-Null

if (-not (Test-Path $cur))  { Write-Host "[错误] 找不到当前树的可执行: $cur"  -ForegroundColor Red;  exit 2 }
if (-not (Test-Path $base)) {
    Write-Host "[错误] 找不到基线树的可执行: $base" -ForegroundColor Red
    Write-Host "  先建基线 worktree 并编译 (一次性):"
    Write-Host "    git worktree add --detach $BaseWorktree 59e5233"
    Write-Host "    (把 test/bench_sac_mcts_min_main.cpp 拷进去, 并在它的 CMakeLists 里加同类目标)"
    Write-Host "    cmake -S $BaseWorktree -B $BaseWorktree\$BaseBuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64"
    Write-Host "    cmake --build $BaseWorktree\$BaseBuild --target bench_sac_mcts_min"
    exit 2
}

# 先自检: 固定 --mcts-srand 后, 同一命令两次必须给出相同走法序列
# (否则说明 std::srand 没生效, 后面所有比较都无意义)
$p1 = Join-Path $out 'selfcheck1.txt'; $p2 = Join-Path $out 'selfcheck2.txt'
foreach ($p in @($p1, $p2)) {
    & $cur --games=1 --sims=32 --plies=20 --opening=4 --seed=1 `
           --mcts-srand=$MctsSrand --backbone=moe-mlp --dump-moves=$p 2>&1 | Out-Null
}
if ($null -ne (Compare-Object (Get-Content $p1) (Get-Content $p2))) {
    Write-Host "[失败] 固定 --mcts-srand 后同一命令两次结果不同 => 对手随机流没被固定, 比较无效" -ForegroundColor Red
    exit 1
}
Write-Host "[前置] --mcts-srand 生效 (同一命令两次走法相同)" -ForegroundColor Green

$cases = @(
    @{ name = 'moe-mlp 4x80'; backbone = 'moe-mlp'; games = 4; plies = 80; sims = 64 },
    @{ name = 'tb       2x30'; backbone = 'tb';       games = 2; plies = 30; sims = 16 }
)
$failed = 0
foreach ($seed in $Seeds) {
    foreach ($c in $cases) {
        $fc = Join-Path $out ("cur_{0}_{1}.txt"  -f $c.backbone, $seed)
        $fb = Join-Path $out ("base_{0}_{1}.txt" -f $c.backbone, $seed)
        & $cur  --games=$($c.games) --sims=$($c.sims) --plies=$($c.plies) --opening=4 `
                 --seed=$seed --mcts-srand=$MctsSrand --backbone=$($c.backbone) --dump-moves=$fc 2>&1 | Out-Null
        & $base --games=$($c.games) --sims=$($c.sims) --plies=$($c.plies) --opening=4 `
                 --seed=$seed --mcts-srand=$MctsSrand --backbone=$($c.backbone) --dump-moves=$fb 2>&1 | Out-Null
        $a = Get-Content $fc; $b = Get-Content $fb
        $d = Compare-Object $a $b
        if ($null -eq $d) {
            Write-Host ("  [{0} seed={1}] {2} 行  逐手完全相同" -f $c.name, $seed, $a.Count) -ForegroundColor Green
        } else {
            Write-Host ("  [{0} seed={1}] **差异 {2} 处** (前 6 条如下)" -f $c.name, $seed, $d.Count) -ForegroundColor Red
            $d | Select-Object -First 6 | ForEach-Object { Write-Host ("      {0} {1}" -f $_.SideIndicator, $_.InputObject) }
            $failed++
        }
    }
}
Write-Host ""
if ($failed -eq 0) {
    Write-Host "VERDICT: PASS | 当前 SAC 与 59e5233 在以上用例上逐手等价" -ForegroundColor Green
    exit 0
} else {
    Write-Host "VERDICT: FAIL | $failed 个用例出现走法差异" -ForegroundColor Red
    exit 1
}
