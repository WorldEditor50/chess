param([string[]]$Dirs = @('phase1','phase1b','phase2'))
# Pool runs by label family and print pooled W-L-D / score / decisive win rate.
# ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
# NOTE: means are computed by hand -- Measure-Object throws (and then leaves the
# variable unset, which silently shifts every later {N} placeholder) when a column
# is missing from the CSV (phase1 CSVs predate the qTargetAbsMean column).
$rows = @()
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
foreach ($d in $Dirs) {
    $p = "$PSScriptRoot\$d"
    if (-not (Test-Path $p)) { $p = "$root\build\exp\$d" }
    if (-not (Test-Path $p)) { continue }
    foreach ($f in (Get-ChildItem "$p\*.csv" | Sort-Object Name)) {
        $rows += Import-Csv -Path $f.FullName
    }
}
function Prop($row, [string]$name) {
    $v = $row.PSObject.Properties[$name]
    if ($null -eq $v -or $null -eq $v.Value -or "$($v.Value)" -eq '') { return 0.0 }
    return [double]$v.Value
}
function MeanOf($g, [string]$name) {
    # 只对**真的有这一列**的那些 run 取平均 (phase1 的 CSV 早于 qTargetAbsMean 列;
    # 把缺失当成 0 会把均值稀释成"看起来很像 0.03"的假读数)。
    $s = 0.0; $k = 0
    foreach ($r in $g) {
        $v = $r.PSObject.Properties[$name]
        if ($null -eq $v -or $null -eq $v.Value -or "$($v.Value)" -eq '') { continue }
        $s += [double]$v.Value; $k++
    }
    if ($k -eq 0) { return [double]::NaN }
    return $s / $k
}
function Pool([string]$pattern, [string]$name) {
    $g = @($rows | Where-Object { $_.label -match $pattern })
    if ($g.Count -eq 0) { return }
    $w = 0; $l = 0; $dr = 0
    foreach ($r in $g) {
        $w += [int](Prop $r 'wins'); $l += [int](Prop $r 'losses'); $dr += [int](Prop $r 'draws')
    }
    $n = $w + $l + $dr
    $score = 0.0; $decisive = 0.0
    if ($n -gt 0) { $score = 100.0 * ($w + 0.5 * $dr) / $n }
    if ($w + $l -gt 0) { $decisive = 100.0 * $w / ($w + $l) }
    $line = "{0,-34} runs={1,-3} games={2,-4} W-L-D={3}-{4}-{5,-4} score={6,5:N1}%  decisive={7,5:N1}%" -f `
        $name, $g.Count, $n, $w, $l, $dr, $score, $decisive
    $line += "  |Q|={0,6:N3} |Qt|={1,5:N3} aEnd={2,5:N2} clamp={3,5:N1}% E[minQ]={4,6:N3} aH={5,5:N2} Qspread={6:N4}" -f `
        (MeanOf $g 'qAbsMean'), (MeanOf $g 'qTargetAbsMean'), (MeanOf $g 'alphaLast'), `
        (100.0 * (MeanOf $g 'clampFrac')), (MeanOf $g 'vQMean'), (MeanOf $g 'vEntMean'), (MeanOf $g 'qSpreadMean')
    Write-Output $line
}
Write-Output "=== pooled by family ($($rows.Count) runs) ==="
Pool '^diag-new'                    '1) current default (a=0.98/1e-3)'
Pool '^diag-olda'                   '1) pre-fix caliper (a=0.5/5e-3)'
Pool '^diag-legacy'                 '1) legacy 59e5233 (no clamp)'
Pool '^bridge-sparse'               '1) bridge: sparse leaf'
Pool '^sweep-c0-h0'                 '1) sweep clamp=0 huber=0'
Pool '^sweep-c0-h1'                 '1) sweep clamp=0 huber=1'
Pool '^sweep-c2-h0'                 '1) sweep clamp=2 huber=0'
Pool '^sweep-c2-h1'                 '1) sweep clamp=2 huber=1'
Pool '^sweep-c3-h'                  '1) sweep clamp=3 (h=0/1)'
Pool '^sweep-c4-h'                  '1) sweep clamp=4 (h=0/1)'
Pool '^inv-noent'                   '1) intervention: no entropy in target'
Pool '^inv-slots'                   '1) intervention: Hbar from slots'
Pool '^inv-both'                    '1) intervention: both'
Pool '^legacy-c2-h1'                '2) legacy + clamp=2 huber=1'
Pool '^legacy-c4-h1'                '2) legacy + clamp=4 huber=1'
Pool '^vs0-nolearn'                 '1c) untrained, valueScale=0'
Pool '^vs1-nolearn'                 '1c) untrained, valueScale=1'
Pool '^vs4-nolearn'                 '1c) untrained, valueScale=4'
Pool '^shape-new-k0'                '3) shaping: new, shape=0'
Pool '^shape-new-k2'                '3) shaping: new, shape=2'
Pool '^shape-legacy-k0'             '3) shaping: legacy, shape=0'
Pool '^shape-legacy-k2'             '3) shaping: legacy, shape=2'
