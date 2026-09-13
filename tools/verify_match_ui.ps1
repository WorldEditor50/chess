# tools/verify_match_ui.ps1
#
# Reproducible end-to-end check of the Agent-vs-Agent arena through the real GUI
# (see docs/agents_design.md section 9). Launches chess.exe, sets the number of
# games via the spinner, starts the match, and reads the result text back.
#
# It drives the app through Windows UI Automation instead of simulated
# keystrokes, for two reasons learned the hard way:
#
#   * SendKeys only works when the app owns the foreground, and
#     WScript.Shell.AppActivate fails silently when the console has it. UIA
#     InvokePattern works regardless of focus.
#   * UIA also hands back the widget rectangles and their text, so the script can
#     *assert on the result label* instead of trying to OCR pixels.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File tools/verify_match_ui.ps1
#   powershell ... -File tools/verify_match_ui.ps1 -Games 2 -Full
#
# ASCII only in code; Chinese string literals below are matched against the UI,
# so this file must be saved as UTF-8 **with BOM** for Windows PowerShell to
# decode it correctly.

param(
    [string]$Exe = "",
    [int]$Games = 1,
    [switch]$Full,           # wait for the whole match instead of aborting early
    [int]$TimeoutSec = 300,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
Write-Output "launched chess.exe pid=$($proc.Id)"
Start-Sleep -Seconds 6

$script:root = $null
try {
    $script:root = [System.Windows.Automation.AutomationElement]::FromHandle(
        (Get-Process -Id $proc.Id).MainWindowHandle)
    if ($null -eq $script:root) { throw "no main window handle" }

    function Find-ByName([string]$name) {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty, $name)
        return $script:root.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants, $cond)
    }

    function All-Texts {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Text)
        $all = $script:root.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants, $cond)
        $out = @()
        foreach ($t in $all) { $out += $t.Current.Name }
        return $out
    }

    function Set-Games([double]$v) {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Spinner)
        $spins = $script:root.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants, $cond)
        if ($spins.Count -lt 1) { throw "no spinner found" }
        $pat = $null
        if (-not $spins.Item(0).TryGetCurrentPattern(
                [System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$pat)) {
            throw "games spinner has no RangeValue pattern"
        }
        $pat.SetValue($v)
    }

    # --- the arena controls must exist ---
    $btn = Find-ByName "开始对弈"
    if ($null -eq $btn) { throw "'开始对弈' button not found" }
    Write-Output ("start button rect = {0}" -f $btn.Current.BoundingRectangle)

    $preSpin = $null
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Spinner)
    $preSpin = $script:root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $cond)
    Write-Output ("spinner count = {0} (expect 2: 局数 / 预训)" -f $preSpin.Count)

    Set-Games $Games
    Start-Sleep -Milliseconds 300
    Write-Output "games set to $Games"

    # --- start the match ---
    $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 1200

    $running = ($null -ne (Find-ByName "停止对弈"))
    Write-Output ("match_running = {0}  (button switched to 停止对弈)" -f $running)

    $final = ""
    if ($Full) {
        $deadline = (Get-Date).AddSeconds($TimeoutSec)
        while ((Get-Date) -lt $deadline) {
            foreach ($t in All-Texts) {
                if ($t -like "*共 *局*手*") { $final = $t }
            }
            if ($final -ne "" -and $final -notlike "*进行中*") { break }
            Start-Sleep -Milliseconds 700
        }
    } else {
        # just prove it is live, then ask it to stop
        Start-Sleep -Seconds 3
        $stop = Find-ByName "停止对弈"
        if ($null -ne $stop) {
            $stop.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            Write-Output "invoked 停止对弈"
        }
        Start-Sleep -Seconds 4
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            foreach ($t in All-Texts) {
                if ($t -like "*共 *局*手*") { $final = $t }
            }
            if ($final -ne "") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Output ("result label = " + $final)
    foreach ($t in All-Texts) {
        if ($t -like "AI思考时间:*" -or $t -like "探索+预训练:*") {
            Write-Output ("   " + $t)
        }
    }

    $back = ($null -ne (Find-ByName "开始对弈"))
    Write-Output ("button back to 开始对弈 = {0}" -f $back)

    # dismiss the result box and any weight-save dialog
    for ($i = 0; $i -lt 8; $i++) {
        [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
        Start-Sleep -Milliseconds 350
    }
    Start-Sleep -Seconds 1
    $alive = ($null -ne (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue))
    Write-Output ("app_still_alive = {0}" -f $alive)

    $ok = $running -and $alive -and ($final -ne "")
    Write-Output ""
    if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
}
finally {
    if (-not $KeepOpen) {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    }
}
