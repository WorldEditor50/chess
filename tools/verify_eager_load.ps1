# tools/verify_eager_load.ps1
#
# Verifies eager loading ("load every model at startup": ChessBoard::startupLoad
# preloads ALL seven weight groups) through the real GUI.
#
# Three observable consequences, all asserted here:
#
#   [1] STARTUP pops the "loading" hourglass (title built from code points below)
#       and it closes by itself (no residue).
#       Startup is ~8-10 s because of those 438 MB, so the feedback has to be there.
#   [2] FIRST USE of the sparse-MoE agent does NOT pop any hourglass any more --
#       its weights were already read at startup. **This is the regression nail
#       for eager loading**: put the lazy load back and [2] fails.
#   [3] while [2] is being watched the match really is progressing (the reward
#       readout's sample count grows), so "no hourglass" cannot pass because the
#       match never started.
#
# This replaces tools/verify_busy_lazy.ps1, which asserted the OPPOSITE (that the
# first use DOES pop the hourglass) and therefore had to start failing the moment
# loading became eager. A test that can only pass under the old behaviour gets
# rewritten, not kept around.
#
# Why UIA instead of screenshots/keystrokes: the dialog is application-modal and
# owned by the main window, so only the accessibility tree reliably tells us
# "a window with this title exists and is visible right now" (SendKeys needs the
# foreground, and pixel checks can't tell the hourglass from the thinking widget).
#
# ASCII only, no BOM (window/control captions are built from code points below).

param(
    [string]$Exe = "",
    [int]$TimeoutSec = 120,   # how long to wait for the match's first plies
    [int]$QuietSec = 40       # how long to watch for a stray hourglass after start
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

# A shell without Qt's bin on PATH launches the exe into STATUS_DLL_NOT_FOUND
# (0xC0000135), which looks exactly like "the app crashed at startup".
$qtFound = $false
foreach ($d in ($env:PATH -split ';')) {
    if ($d -ne "" -and (Test-Path (Join-Path $d "Qt6Widgets.dll"))) { $qtFound = $true; break }
}
if (-not $qtFound) {
    $hit = Get-ChildItem -Path "C:\Qt\*\msvc*\bin\Qt6Widgets.dll" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($null -ne $hit) { $env:PATH = $hit.DirectoryName + ";" + $env:PATH }
}

# captions, built from code points so this file stays pure ASCII
$busyPrefixes = @(
    [string]([char]0x6B63 + [char]0x5728 + [char]0x8F7D + [char]0x5165),   # loading
    [string]([char]0x6B63 + [char]0x5728 + [char]0x4FDD + [char]0x5B58)    # saving
)
$startButton = [string]([char]0x5F00 + [char]0x5C40)                       # start game
$startMatch  = [string]([char]0x5F00 + [char]0x59CB + [char]0x5BF9 + [char]0x5F08)  # start match
$stopMatch   = [string]([char]0x505C + [char]0x6B62 + [char]0x5BF9 + [char]0x5F08)  # stop match
$rewardWord  = [string]([char]0x5956 + [char]0x52B1)                       # reward
$dotWord     = [string]([char]0x70B9)                                      # "N <samples>" suffix

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
# Any visible top-level window of OUR process whose title starts with a busy
# prefix. Findings are collected by window name, not by searching from the main
# window: the dialog is owned/modal, and UIA's root is the only reliable start.
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
function All-Texts() {
    $r = Get-Root
    if ($null -eq $r) { return @() }
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Text)
    $out = @()
    foreach ($e in $r.FindAll([System.Windows.Automation.TreeScope]::Descendants, $c)) {
        $out += [string]$e.Current.Name
    }
    return $out
}
function RewardSamples() {
    foreach ($t in All-Texts) {
        if ($t.StartsWith($rewardWord) -and $t -match ("(\d+)\s*" + $dotWord)) {
            return [int]$Matches[1]
        }
    }
    return -1
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

# ---------------------------------------------------------------- [1] startup
# Watch from the very first moment: the startup hourglass appears within ~1 s and
# has to disappear on its own once all seven weight groups are in memory.
$busySeenAt = $null
$busyTitle = ""
$busyGoneAt = $null
$startupDeadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $startupDeadline) {
    Start-Sleep -Milliseconds 200
    $busy = Find-BusyWindow
    if ($busy -ne "" -and $null -eq $busySeenAt) {
        $busySeenAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $busyTitle = $busy
        Write-Output ("startup hourglass appeared at {0}s: '{1}'" -f $busySeenAt, $busy)
    }
    if ($busy -eq "" -and $null -ne $busySeenAt) {
        $busyGoneAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Output ("startup hourglass closed at {0}s" -f $busyGoneAt)
        break
    }
    # If the UI is already usable we missed the dialog (it never showed) -> go on
    $b = Find-ByName $startButton
    if ($null -ne $b -and $b.Current.IsEnabled -and $null -eq $busySeenAt) { break }
}
# wait until the UI is usable (all models loaded)
$ready = $false
for ($i = 0; $i -lt 200; $i++) {
    $b = Find-ByName $startButton
    if ($null -ne $b -and $b.Current.IsEnabled) { $ready = $true; break }
    Start-Sleep -Milliseconds 300
}
$readyAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Write-Output ("ui ready at {0}s = {1}" -f $readyAt, $ready)
if (-not $ready) { throw "UI never became ready" }
$startupOk = ($null -ne $busySeenAt) -and ($null -ne $busyGoneAt)
Write-Output ("startup hourglass seen and closed = {0}" -f $startupOk)

# ---------------------------------------------------- [2]/[3] first use of MoE
# A = SAC+AZ with the sparse-MoE backbone, B = 0 (Alpha-Beta).
#
# [2026-09 修复] A 原来是**写死的下标 8**。给下拉框加了 Alpha-Beta 三档弱等级
# (kAgents 的第 1~3 位) 之后, 下标 8 变成了 PPO+MCTS (MLP 专家) —— 这个脚本会去
# "验证"另一个 agent 的首次使用不弹沙漏, 而这句话对那个 agent 根本不成立
# (它是本脚本最想抓的那类静默错位: 断言还在跑、指向的对象已经不是它说的那个)。
# 现在按**枚举名**从 src/mainwindow.cpp 的 kAgents 里解析出下标, 顺序变了下标自己会跟着走。
$kAgentsSrc = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "..\src\mainwindow.cpp")
$kAgentsBody = [regex]::Match($kAgentsSrc,
    'const AgentChoice kAgents\[\]\s*=\s*\{(?<body>.*?)\n\};',
    [System.Text.RegularExpressions.RegexOptions]::Singleline)
if (-not $kAgentsBody.Success) { throw "kAgents[] not found in src/mainwindow.cpp" }
$comboTypes = @()
# 字符类必须含数字: 枚举名里有 AGENT_AB_L1 这样的名字 ([A-Z_] 会静默漏掉它们)
foreach ($mm in [regex]::Matches($kAgentsBody.Groups['body'].Value,
        'ChessBoard::(?<type>AGENT_[A-Z0-9_]+)')) {
    $comboTypes += $mm.Groups['type'].Value
}
$idxMoe = [array]::IndexOf($comboTypes, "AGENT_SACAZ_MOE")
$idxAb  = [array]::IndexOf($comboTypes, "AGENT_ALPHABETA")
if ($idxMoe -lt 0) { throw "AGENT_SACAZ_MOE not found in kAgents" }
if ($idxAb -lt 0)  { throw "AGENT_ALPHABETA not found in kAgents" }
Write-Output ("combo index from source: SAC+AZ-MoE = {0}, Alpha-Beta = {1} (of {2} entries)" -f
              $idxMoe, $idxAb, $comboTypes.Count)
Write-Output ("A side = " + (Select-ComboItem 1 $idxMoe))
Start-Sleep -Milliseconds 400
Write-Output ("B side = " + (Select-ComboItem 2 $idxAb))
Start-Sleep -Milliseconds 400

$btn = Find-ByName $startMatch
if ($null -eq $btn) { throw "start-match button not found" }
$btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Start-Sleep -Milliseconds 1200
Write-Output ("match started at {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds)
$running = ($null -ne (Find-ByName $stopMatch))
Write-Output ("match_running = {0}" -f $running)

# NOTE: call these helpers WITHOUT parentheses -- PowerShell parses "Foo()" as a
# syntax error ("An expression was expected after '('"), unlike C/JS. That cost one
# failed run of this very script.
# The reward readout only exists once the first ply has been played (this agent
# burns ~10 s on its explore+pre-train phase before that), so wait for it to show
# up before using it as the "the match is really running" baseline.
$firstSamples = -1
for ($i = 0; $i -lt 40; $i++) {
    $firstSamples = RewardSamples
    if ($firstSamples -ge 1) { break }
    Start-Sleep -Milliseconds 500
}
Write-Output ("reward samples baseline = {0}" -f $firstSamples)
$lastSamples = $firstSamples
$strayAt = $null
$strayTitle = ""
$quietDeadline = (Get-Date).AddSeconds($QuietSec)
while ((Get-Date) -lt $quietDeadline) {
    Start-Sleep -Milliseconds 300
    $busy = Find-BusyWindow
    if ($busy -ne "" -and $null -eq $strayAt) {
        $strayAt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $strayTitle = $busy
    }
    $n = RewardSamples
    if ($n -ge 0) { $lastSamples = $n }
    # match over already -> stop watching early
    if ($null -eq (Find-ByName $stopMatch)) { break }
}

$noStray = ($null -eq $strayAt)
# "progressed" must not be a false pass: require a real baseline AND growth.
$progressed = ($firstSamples -ge 1) -and ($lastSamples -gt $firstSamples)
Write-Output ("first-use hourglass appeared = {0}" -f (-not $noStray))
if (-not $noStray) {
    Write-Output ("   at {0}s: '{1}'  <- lazy loading is back" -f $strayAt, $strayTitle)
}
Write-Output ("reward samples during the watch = {0} -> {1}" -f $firstSamples, $lastSamples)
Write-Output ("match progressed while watching = {0}" -f $progressed)

$ok = $startupOk -and $running -and $noStray -and $progressed
Write-Output ""
if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
