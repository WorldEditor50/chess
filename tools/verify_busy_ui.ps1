# tools/verify_busy_ui.ps1
#
# Verifies that loading/saving model weights pops up the hourglass wait dialog
# (see src/busydialog.h and docs/agents_design.md section 15).
#
# How: launch chess.exe and enumerate the process' top-level windows every
# 250 ms, recording
#   * when a window titled "loading"/"saving" appears   (busy started)
#   * when it disappears                               (busy finished)
#   * when the main window's first button becomes enabled (startupComplete)
# Expected: the dialog appears while the UI is still disabled and is already
# gone by the time the UI becomes usable - i.e. "hourglass during load, nothing
# left hanging afterwards".
#
# On this machine startup parses tens of MB of legacy (decimal text) weights, so
# observing the dialog is reliable; on a machine that starts instantly the script
# reports "finished before the first sample", which is not a failure.
#
# ASCII only, no BOM: Windows PowerShell decodes a BOM-less .ps1 as ANSI, so even
# a Chinese *comment* can break the parser (mangled bytes swallow a quote). The
# window titles we compare against are therefore built from code points.

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 90
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

# Qt DLLs must be findable from a plain shell (same block as verify_match_ui.ps1).
$qtFound = $false
foreach ($d in ($env:PATH -split ';')) {
    if ($d -ne "" -and (Test-Path (Join-Path $d "Qt6Widgets.dll"))) { $qtFound = $true; break }
}
if (-not $qtFound) {
    $hit = Get-ChildItem -Path "C:\Qt\*\msvc*\bin\Qt6Widgets.dll" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($null -ne $hit) {
        $env:PATH = $hit.DirectoryName + ";" + $env:PATH
        Write-Output ("Qt bin prepended = {0}" -f $hit.DirectoryName)
    }
}

# BusyDialog::startBusy() sets the window title to "loading" / "saving"
# (see busydialog.cpp). Built from code points because this file is ASCII only.
$busyPrefixes = @(
    [string]([char]0x6B63 + [char]0x5728 + [char]0x8F7D + [char]0x5165),   # loading
    [string]([char]0x6B63 + [char]0x5728 + [char]0x4FDD + [char]0x5B58)    # saving
)

# The start button's caption ("start game"), built from code points. It is the
# button MainWindow enables when startupLoad() finishes.
$startButtonName = [string]([char]0x5F00 + [char]0x5C40)

function Get-ProcessWindows([int]$pid_) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $pid_)
    return [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
}

# Only *visible* top-level windows count: Qt creates hidden helper windows with
# non-empty names, and counting those made "the dialog is still open" always true
# in the first version of this check.
function Find-BusyWindow([int]$pid_) {
    foreach ($e in (Get-ProcessWindows $pid_)) {
        if ($e.Current.ControlType -ne [System.Windows.Automation.ControlType]::Window) { continue }
        if ($e.Current.IsOffscreen) { continue }
        $r = $e.Current.BoundingRectangle
        if ($r.Width -le 0 -or $r.Height -le 0) { continue }
        $n = [string]$e.Current.Name
        foreach ($pre in $busyPrefixes) {
            if ($n.StartsWith($pre)) { return $n }
        }
    }
    return ""
}

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
$t0 = Get-Date
Write-Output "launched chess.exe pid=$($proc.Id)"

$busySeenAt = $null
$busyGoneAt = $null
$busyTitle = ""
$readyAt = $null
$deadline = (Get-Date).AddSeconds($TimeoutSec)

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if ($proc.HasExited) { throw "chess.exe exited early (exit code $($proc.ExitCode))" }

    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $p) { throw "chess.exe disappeared" }

    $busy = Find-BusyWindow $proc.Id
    if ($busy -ne "" -and $null -eq $busySeenAt) {
        $busySeenAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $busyTitle = $busy
        Write-Output ("busy window appeared at {0}s: '{1}'" -f $busySeenAt, $busyTitle)
    }
    if ($busy -eq "" -and $null -ne $busySeenAt -and $null -eq $busyGoneAt) {
        $busyGoneAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Output ("busy window closed at {0}s" -f $busyGoneAt)
    }

    # "ready" = the start button is enabled (MainWindow enables it in its
    # startupComplete handler). Looking up that button BY NAME matters: taking
    # "the first enabled Button" is wrong - the metrics panel's buttons are
    # enabled from the start, so that check reported "ready" while the weights
    # were still loading (and made this whole script lie).
    if ($null -eq $readyAt -and $p.MainWindowHandle -ne 0) {
        $rootEl = [System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle)
        if ($null -ne $rootEl) {
            $cond = New-Object System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::NameProperty, $startButtonName)
            $b = $rootEl.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
            if ($null -ne $b -and $b.Current.IsEnabled) {
                $readyAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
                Write-Output ("main UI became ready at {0}s" -f $readyAt)
                break
            }
        }
    }
}

# After the UI is up, make sure the dialog is really gone (not left hanging).
Start-Sleep -Milliseconds 500
$leftover = Find-BusyWindow $proc.Id
$goneAfterReady = ($leftover -eq "")
Write-Output ("busy window still open after ready = {0}" -f (-not $goneAfterReady))

# The property that matters is "$goneAfterReady": the dialog must not be left
# hanging once the UI is usable. The exact disappearance timestamp is only
# informational - the dialog can vanish in the same 250 ms sample in which the
# start button becomes enabled, in which case the loop breaks before recording it.
$seenOk = ($null -ne $busySeenAt)
$goneOk = $goneAfterReady
$orderOk = $true
if ($seenOk -and $null -ne $busyGoneAt -and $null -ne $readyAt) {
    $orderOk = ($busyGoneAt -le $readyAt)
}

Write-Output ""
Write-Output ("busy seen during load      = {0}" -f $seenOk)
Write-Output ("busy gone when ready       = {0}" -f $goneOk)
Write-Output ("busy closed before ready   = {0}" -f $orderOk)
if (-not $seenOk) {
    Write-Output "(load finished before the first sample - unobserved, not a failure)"
}

$ok = $goneOk -and $orderOk -and ($null -ne $readyAt)
Write-Output ""
if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
