# tools/measure_peak_working_set.ps1
#
# Runs a command and reports its PEAK working set (PeakWorkingSet64) plus a few sampled
# points, so "how much memory does this backbone actually cost" is a measured number
# instead of an estimate from parameter counts.
#
# Why external (and not inside the benchmark itself):
#   The natural way would be GetProcessMemoryInfo() inside the tool, but that needs
#   <windows.h>, and the Windows SDK's wingdi.h does `#define PLANES 14` while
#   ppomcts_agent.h has a member `PLANES` (19 planes). Those two collide and the error
#   surfaces *inside the agent header* ("error C2059: syntax error: constant") even
#   though the cause is the benchmark's include -- exactly the kind of misleading error
#   this repo keeps writing down. This script sidesteps it, and matches how the SAC
#   round measured its E=8 peak (1,138 MB).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/measure_peak_working_set.ps1 `
#       -Exe build\...\bench_ppo_backbone_tb.exe -Args "--quick"
#   ... -LogFile build\mem_ppo_tb.txt
#
# ASCII only, no BOM (Windows PowerShell decodes BOM-less .ps1 as ANSI).

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Args = "",
    [string]$LogFile = "",
    [int]$IntervalMs = 250,
    # Where the measured command's own stdout goes. Redirected at the OS level (not left
    # to the caller's pipeline) so that a long run can be tailed while it is still going,
    # and so output is not lost when the harness kills the call on its own timeout --
    # the benchmark's network construction can take minutes and a block-buffered pipe
    # shows nothing until it exits.
    [string]$StdoutFile = ""
)

$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

Write-Host "=== measure_peak_working_set ==="
Write-Host ("exe : " + $Exe)
Write-Host ("args: " + $Args)

if ($StdoutFile -eq "") {
    $StdoutFile = Join-Path $exeDir "measure_peak_working_set.out.txt"
}
$argList = @()
if ($Args -ne "") { $argList = $Args -split "\s+" }
$proc = Start-Process -FilePath $Exe -ArgumentList $argList -WorkingDirectory $exeDir `
                      -PassThru -NoNewWindow -RedirectStandardOutput $StdoutFile
Write-Host ("pid : " + $proc.Id)
Write-Host ("out : " + $StdoutFile)

$peak = 0L
$samples = New-Object System.Collections.ArrayList
while (-not $proc.HasExited) {
    try {
        $p = Get-Process -Id $proc.Id -ErrorAction Stop
        $ws = $p.WorkingSet64
        if ($ws -gt $peak) { $peak = $ws }
        [void]$samples.Add($ws)
    } catch { }
    Start-Sleep -Milliseconds $IntervalMs
}
$proc.WaitForExit()
$exit = $proc.ExitCode

$peakMB = [math]::Round($peak / 1MB, 0)
$lastMB = 0
if ($samples.Count -gt 0) { $lastMB = [math]::Round($samples[$samples.Count - 1] / 1MB, 0) }

$lines = New-Object System.Collections.ArrayList
[void]$lines.Add("cmd      : " + $Exe + " " + $Args)
[void]$lines.Add("exit     : " + $exit)
[void]$lines.Add("samples  : " + $samples.Count + " @ " + $IntervalMs + " ms")
[void]$lines.Add("peak MB  : " + $peakMB)
[void]$lines.Add("last MB  : " + $lastMB)
[void]$lines.Add("series   : " + (($samples | ForEach-Object { [math]::Round($_ / 1MB, 0) }) -join ","))

foreach ($l in $lines) { Write-Host $l }
if ($LogFile -ne "") {
    Set-Content -LiteralPath $LogFile -Value $lines -Encoding UTF8
    Write-Host ("log      : " + $LogFile)
}
if ($exit -ne 0) { exit $exit }
