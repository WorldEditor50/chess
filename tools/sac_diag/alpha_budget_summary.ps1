param([string]$Dir = '')
# ============================================================================
# alpha_budget_summary.ps1 - 汇总 run_alpha_budget.ps1 的输出 (2026-10)
#
# 判据不变: 不看 alpha 的绝对值, 看
#   (1) alpha 的轨迹 (是否贴边界)  (2) alphaH = |y| 的百分比   (3) Qspread / 得分率
# 本文件必须 ASCII only (PowerShell 5.1 + 无 BOM 的中文会吃掉下一行, 见 run_matrix.ps1)。
# ============================================================================
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
if ($Dir -eq '') { $Dir = "$root\build\exp\alpha_budget" }

# log is UTF-8 (the exe prints UTF-8); read it as such no matter the console codepage.
# NOTE: files still being written by a running arm must stay readable -> use
# Get-Content (shares the handle) instead of File::ReadAllText (exclusive open).
function Read-Utf8([string]$p) { return (Get-Content -LiteralPath $p -Raw -Encoding UTF8) }

$rows = @()
foreach ($f in Get-ChildItem "$Dir\*.log" | Sort-Object Name) {
    $t = Read-Utf8 $f.FullName
    $games = [regex]::Matches($t, '(?m)^\s+局\s+\d+/(\d+)')
    if ($games.Count -eq 0) { continue }
    $diag = [regex]::Matches($t, 'α (\d+\.\d+)→(\d+\.\d+) \(αH ([-+]\d+\.\d+) = \|y\| 的 (\d+)%\).*?被夹 (\d+)%.*?Qspread (\d+\.\d+)')
    $aFirst = @(); $aLast = @(); $aH = @(); $aHRatio = @(); $clamp = @(); $qs = @()
    foreach ($m in $diag) {
        $aFirst += [double]$m.Groups[1].Value
        $aLast  += [double]$m.Groups[2].Value
        $aH     += [double]$m.Groups[3].Value
        $aHRatio+= [double]$m.Groups[4].Value
        $clamp  += [double]$m.Groups[5].Value
        $qs     += [double]$m.Groups[6].Value
    }
    $score = [regex]::Match($t, '得分率\s+:\s+([\d\.]+)%')
    $qabs  = [regex]::Match($t, '\|Q\| 均值\s*=?\s*([\d\.]+)')
    $probe = [regex]::Match($t, '(?m)^\s{2}\|Q\|.*')
    if ($aLast.Count -eq 0) { continue }
    $rows += [pscustomobject]@{
        label   = $f.BaseName
        games   = $games.Count
        aStart  = $aFirst[0]
        aEnd    = $aLast[$aLast.Count - 1]
        aMax    = ($aLast | Measure-Object -Maximum).Maximum
        aMin    = ($aLast | Measure-Object -Minimum).Minimum
        aHmean  = [math]::Round((($aH  | Measure-Object -Average).Average), 2)
        aHpct   = [math]::Round((($aHRatio | Measure-Object -Average).Average), 0)
        clampPc = [math]::Round((($clamp | Measure-Object -Average).Average), 0)
        qspread = [math]::Round((($qs | Measure-Object -Average).Average), 3)
        score   = if ($score.Success) { [double]$score.Groups[1].Value } else { -1 }
    }
}
$rows | Format-Table -AutoSize
