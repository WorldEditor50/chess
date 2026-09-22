param([string]$Dir = 'phase1')
# ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less files as ANSI, and a
# mangled multi-byte comment can swallow the following line (that happened here once:
# a Chinese comment ate the "$root = ..." line and every lookup became relative).
#
# One run per line, TSV (Format-Table drops trailing columns on an 80-column host).
# Dir: looks in <repo>/build/exp/<Dir> (where run_matrix.ps1 writes), or an absolute path.
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$out = $Dir
if (-not (Test-Path $out)) { $out = "$PSScriptRoot\$Dir" }
if (-not (Test-Path $out)) { $out = "$root\build\exp\$Dir" }
if (-not (Test-Path $out)) { Write-Output "[err] no such dir: $Dir"; exit 1 }
$rows = @()
foreach ($f in (Get-ChildItem "$out\*.csv" | Sort-Object Name)) {
    $rows += Import-Csv -Path $f.FullName
}
$sel = $rows | Select-Object label, games,
    @{n='W-L-D';e={"$($_.wins)-$($_.losses)-$($_.draws)"}},
    @{n='score%';e={[math]::Round(100*[double]$_.scoreRate,1)}},
    @{n='CI';e={"[" + [math]::Round(100*[double]$_.ciLo,0) + "," + [math]::Round(100*[double]$_.ciHi,0) + "]"}},
    @{n='mate';e={$_.checkmate}}, @{n='stale';e={$_.stalemate}}, @{n='drawCap';e={$_.drawCap}},
    @{n='plies';e={[math]::Round([double]$_.meanPlies,0)}},
    @{n='|Q|';e={[math]::Round([double]$_.qAbsMean,3)}},
    @{n='|Qt|';e={[math]::Round([double]$_.qTargetAbsMean,3)}},
    @{n='clamp%';e={[math]::Round(100*[double]$_.clampFrac,1)}},
    @{n='alpha';e={[math]::Round([double]$_.alphaFirst,2).ToString() + "->" + [math]::Round([double]$_.alphaLast,2).ToString()}},
    @{n='yPre|y|';e={[math]::Round([double]$_.yPreAbsMean,2)}},
    @{n='E[minQ]';e={[math]::Round([double]$_.vQMean,3)}},
    @{n='aH';e={[math]::Round([double]$_.vEntMean,2)}},
    @{n='H';e={[math]::Round([double]$_.hMean,2)}},
    @{n='Hbar';e={[math]::Round([double]$_.hBarMean,2)}},
    @{n='H<Hb%';e={[math]::Round(100*[double]$_.hBelowHbarFrac,0)}},
    @{n='H<HbS%';e={[math]::Round(100*[double]$_.hBelowHbarSlotsFrac,0)}},
    @{n='Qspread';e={[math]::Round([double]$_.qSpreadMean,4)}},
    @{n='leaf';e={ if ($_.sparseLeaf -eq '1') { 'sparse' } else { 'full' } }},
    @{n='shape';e={$_.rewardShape}}
$props = @('label','games','W-L-D','score%','CI','mate','stale','drawCap','plies','|Q|','|Qt|',
           'clamp%','alpha','yPre|y|','E[minQ]','aH','H','Hbar','H<Hb%','H<HbS%','Qspread','leaf','shape')
Write-Output ($props -join "`t")
foreach ($r in $sel) {
    $vals = foreach ($p in $props) { [string]$r.$p }
    Write-Output ($vals -join "`t")
}
