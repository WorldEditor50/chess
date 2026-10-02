# tools/measure_startup.ps1
#
# Measures two things about chess.exe startup, and they are different claims:
#   1. "startup -> interactive": wall-clock until the UI is actually usable
#      (selfPlayBtn becomes enabled, which happens in the startupComplete handler).
#      This is what the user feels behind the "正在载入" hourglass.
#   2. the app's own per-group [weights] timings (it logs one line per group), so a slow
#      startup can be attributed instead of guessed.
#
# Why this exists (2026-10): the four agents PG / DQN / SAC-59e5233 (both backbones) were
# removed from the drop-down AND from the startup preload list (user request: "减少加载
# 时间"). That change needs a number, not an opinion -- the measurement below is where the
# "40.8 s -> 29.5 s" figure in README.md comes from, and the per-group lines say WHY
# (the 59e5233 MoE arm alone was 4.5 s of network construction + 6.3 s of reading 440 MB).
#
# ASCII only, no BOM: Windows PowerShell decodes a BOM-less .ps1 as ANSI, so a Chinese
# comment here would corrupt the parse of the code below it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/measure_startup.ps1
#   powershell -ExecutionPolicy Bypass -File tools/measure_startup.ps1 -Tag after

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 240,
    # Tag for the two output files (build/startup_<tag>.log / .out) so several runs can be
    # compared side by side.
    [string]$Tag = "run"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

$err = Join-Path $exeDir ("startup_" + $Tag + ".log")
$out = Join-Path $exeDir ("startup_" + $Tag + ".out")
Remove-Item $err, $out -ErrorAction SilentlyContinue

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru `
    -RedirectStandardError $err -RedirectStandardOutput $out
$tStartup = -1.0
try {
    $root = $null
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 200
        if ($proc.HasExited) { break }
        if ($root -eq $null) {
            $cond = New-Object System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
            $root = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
                [System.Windows.Automation.TreeScope]::Children, $cond)
            continue
        }
        # selfPlayBtn is enabled only from the startupComplete handler -> "UI is usable".
        $btn = $null
        $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
        $stack = New-Object System.Collections.Stack
        $stack.Push($root)
        while ($stack.Count -gt 0) {
            $cur = $stack.Pop()
            try {
                $aid = [string]$cur.GetCurrentPropertyValue(
                    [System.Windows.Automation.AutomationElement]::AutomationIdProperty)
                if ($aid.EndsWith(".selfPlayBtn")) { $btn = $cur; break }
            } catch {}
            $child = $walker.GetFirstChild($cur)
            while ($child -ne $null) { $stack.Push($child); $child = $walker.GetNextSibling($child) }
        }
        if ($btn -ne $null -and $btn.Current.IsEnabled) {
            $tStartup = $sw.Elapsed.TotalSeconds
            break
        }
    }
} finally {
    # Let the app flush its log before the process dies. The redirected stream is
    # block-buffered, so the last few [weights]/[selfcheck] lines are otherwise LOST --
    # that is measured behaviour, not a guess (a first version killed the process right
    # after the UI became interactive and the tail of the log was missing).
    Start-Sleep -Seconds 3
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 800
}

Write-Output ("== measure_startup [" + $Tag + "] ==")
Write-Output ("exe: " + $Exe)
if ($tStartup -ge 0) {
    Write-Output ("startup -> interactive: {0:N1} s" -f $tStartup)
} else {
    Write-Output "startup -> interactive: TIMEOUT (the UI never became usable)"
}
Write-Output "---- the app's own per-group timings ----"
Get-Content $err -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "\[weights\]" } | ForEach-Object { Write-Output ("  " + $_) }
Write-Output "---- startup self-check lines (which models were actually loaded) ----"
Get-Content $err -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "\[selfcheck\]" } | ForEach-Object { Write-Output ("  " + $_) }

$sum = 0
$n = 0
Get-Content $err -ErrorAction SilentlyContinue | ForEach-Object {
    $m = [regex]::Match($_, "\[weights\].*:\s*(\d+)\s*ms")
    if ($m.Success) { $sum += [int]$m.Groups[1].Value; $n++ }
}
Write-Output ("sum of " + $n + " [weights] steps = " + $sum + " ms")
Write-Output "  (a lower bound on startup: the log tail can be missing if the process was killed too early)"
