# tools/verify_busy_lazy.ps1
#
# Verifies that the FIRST use of SAC+AZ (sparse MoE) pops the hourglass wait
# dialog.
#
# Why a separate script: that agent's weights are 3 x 146 MB, so startupLoad()
# deliberately does NOT preload them (that pushed startup from ~2 s to ~19 s).
# They are loaded lazily on first use, emitting the same busyStarted/busyFinished
# signals. Only a real GUI run can check this:
#   pick A = SAC+AZ (sparse MoE) -> start a match -> the first move triggers the
#   lazy load -> assert the dialog appears during it and closes afterwards.
#
# ASCII only, no BOM (window/control captions are built from code points).

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 120
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(uint f,uint dx,uint dy,uint d,int e);' -Name MU -Namespace W32 -PassThru | Out-Null

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$exeDir = Split-Path $Exe -Parent

$qtFound = $false
foreach ($d in ($env:PATH -split ';')) {
    if ($d -ne "" -and (Test-Path (Join-Path $d "Qt6Widgets.dll"))) { $qtFound = $true; break }
}
if (-not $qtFound) {
    $hit = Get-ChildItem -Path "C:\Qt\*\msvc*\bin\Qt6Widgets.dll" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($null -ne $hit) { $env:PATH = $hit.DirectoryName + ";" + $env:PATH }
}

# code points: "loading" / "start game"
$busyPrefixes = @(
    [string]([char]0x6B63 + [char]0x5728 + [char]0x8F7D + [char]0x5165),
    [string]([char]0x6B63 + [char]0x5728 + [char]0x4FDD + [char]0x5B58)
)
$startButton = [string]([char]0x5F00 + [char]0x5C40)          # start game
$startMatch  = [string]([char]0x5F00 + [char]0x59CB + [char]0x5BF9 + [char]0x5F08)  # start match

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
$t0 = Get-Date
Write-Output "launched chess.exe pid=$($proc.Id)"

function Get-Root() {
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -eq $p -or $p.MainWindowHandle -eq 0) { return $null }
    return [System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle)
}
function Find-ByName([string]$name) {
    $r = Get-Root
    if ($null -eq $r) { return $null }
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    return $r.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $c)
}
function Find-BusyWindow() {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
    foreach ($e in [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants, $c)) {
        if ($e.Current.ControlType -ne [System.Windows.Automation.ControlType]::Window) { continue }
        if ($e.Current.IsOffscreen) { continue }
        $r = $e.Current.BoundingRectangle
        if ($r.Width -le 0 -or $r.Height -le 0) { continue }
        $n = [string]$e.Current.Name
        foreach ($pre in $busyPrefixes) { if ($n.StartsWith($pre)) { return $n } }
    }
    return ""
}
function Select-ComboItem([int]$comboIndex, [int]$itemIndex) {
    $r = Get-Root
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ComboBox)
    $combos = $r.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    $cb = $combos.Item($comboIndex)
    $ex = $null
    if (-not $cb.TryGetCurrentPattern(
            [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$ex)) {
        throw "combo $comboIndex cannot expand"
    }
    $ex.Expand()
    Start-Sleep -Milliseconds 600
    $lc = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $items = $cb.FindAll([System.Windows.Automation.TreeScope]::Descendants, $lc)
    if ($items.Count -le $itemIndex) { $ex.Collapse(); throw "no item $itemIndex" }
    $t = $items.Item($itemIndex)
    $br = $t.Current.BoundingRectangle
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(
        ([int]($br.X + $br.Width / 2)), ([int]($br.Y + $br.Height / 2)))
    Start-Sleep -Milliseconds 150
    [W32.MU]::mouse_event(0x0002, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 50
    [W32.MU]::mouse_event(0x0004, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 500
    if ($cb.Current.ExpandCollapseState -ne
        [System.Windows.Automation.ExpandCollapseState]::Collapsed) { $ex.Collapse() }
    Start-Sleep -Milliseconds 500
    return $t.Current.Name
}

# 1) wait until the UI is usable
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 300
    $b = Find-ByName $startButton
    if ($null -ne $b -and $b.Current.IsEnabled) { $ready = $true; break }
}
Write-Output ("ui ready = {0} ({1:N1} s)" -f $ready, ((Get-Date) - $t0).TotalSeconds)
if (-not $ready) { throw "UI never became ready" }

# 2) side A = agent index 8 (SAC+AZ with the sparse-MoE backbone), side B = 0 (Alpha-Beta)
Write-Output ("A side = " + (Select-ComboItem 1 8))
Start-Sleep -Milliseconds 400
Write-Output ("B side = " + (Select-ComboItem 2 0))
Start-Sleep -Milliseconds 400

# 3) start the match
$btn = Find-ByName $startMatch
if ($null -eq $btn) { throw "start-match button not found" }
$btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Write-Output ("match started at {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds)

# 4) watch the busy window: it must appear (3 x 146 MB lazy load) and then close
$seenAt = $null
$goneAt = $null
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    $busy = Find-BusyWindow
    if ($busy -ne "" -and $null -eq $seenAt) {
        $seenAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Output ("busy window appeared at {0}s: '{1}'" -f $seenAt, $busy)
    }
    if ($busy -eq "" -and $null -ne $seenAt -and $null -eq $goneAt) {
        $goneAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Output ("busy window closed at {0}s" -f $goneAt)
        break
    }
}

$ok = ($null -ne $seenAt) -and ($null -ne $goneAt)
Write-Output ""
Write-Output ("busy seen during lazy load = {0}" -f ($null -ne $seenAt))
Write-Output ("busy closed afterwards     = {0}" -f ($null -ne $goneAt))
Write-Output ""
if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
