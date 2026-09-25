# tools/verify_chess_saves_weights.ps1
#
# 直接测 chess.exe: 一场【Agent 对弈】跑完之后, 权重是否真的写到标准路径。
#
# 为什么单独一个脚本: 这是用户报障的现象 ("再次开局/重开程序, agent 像没加载权重") 的
# 端到端判据 —— 单元测试测不到"界面这条链路到底有没有把权重存下来"。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File tools/verify_chess_saves_weights.ps1
#   powershell ... -File tools/verify_chess_saves_weights.ps1 -MaxWaitSec 420
#
# 判据 (三个都要成立):
#   1. 对弈真的开起来了 (按钮文字从"开始对弈"变成"停止对弈");
#   2. 对弈真的结束了 (按钮变回"开始对弈", 或逐局明细里出现"本场结束");
#   3. weights/ 下出现标准命名的权重文件 (非空)。
#
# 编码: 本文件含中文**字面量** (要跟界面文字比对), 必须存成 UTF-8 with BOM。
#       无 BOM 时 Windows PowerShell 按 ANSI 解码, 报的会是"字符串未终止"这类无关语法错。

param(
    [string]$Exe = "",
    [int]$MaxWaitSec = 420
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path

# 干净目录: 那里没有 weights/, 所以"出现权重文件"只可能来自这一场对弈
$sandbox = Join-Path $env:TEMP ("chess_save_probe_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $sandbox | Out-Null
Copy-Item $Exe (Join-Path $sandbox "chess.exe") -Force
Write-Output ("sandbox = {0}" -f $sandbox)

$root = [System.Windows.Automation.AutomationElement]::RootElement

function Find-ByName([string]$name) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
}
function All-ListItems {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $all = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    $out = @()
    foreach ($t in $all) { $out += $t.Current.Name }
    return $out
}
function Set-Games([double]$v) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Spinner)
    $spins = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($spins.Count -lt 1) { throw "no spinner found" }
    $pat = $null
    if (-not $spins.Item(0).TryGetCurrentPattern(
            [System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$pat)) {
        throw "games spinner has no RangeValue pattern"
    }
    $pat.SetValue($v)
}
# 选第 m 个下拉框的第 n 项 (Qt 的弹出项要用鼠标点, SelectionItemPattern 会静默无效)
function Select-ComboItem([int]$comboIndex, [int]$itemIndex) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ComboBox)
    $combos = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    if ($combos.Count -le $comboIndex) { throw "combo $comboIndex not found" }
    $cb = $combos.Item($comboIndex)
    $expand = $null
    if (-not $cb.TryGetCurrentPattern(
            [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$expand)) {
        throw "combo $comboIndex cannot be expanded"
    }
    $expand.Expand()
    Start-Sleep -Milliseconds 500
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $items = $cb.FindAll([System.Windows.Automation.TreeScope]::Descendants, $liCond)
    if ($items.Count -le $itemIndex) {
        $expand.Collapse()
        throw "combo $comboIndex has no item $itemIndex (count=$($items.Count))"
    }
    $target = $items.Item($itemIndex)
    $want = $target.Current.Name
    $r = $target.Current.BoundingRectangle
    if ($r.Width -le 0 -or $r.Height -le 0) {
        $expand.Collapse()
        throw "item $itemIndex of combo $comboIndex has no rectangle"
    }
    [System.Windows.Forms.Cursor]::Position =
        New-Object System.Drawing.Point([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
    Start-Sleep -Milliseconds 120
    # 点下去
    Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(int f,int x,int y,int d,int e);' -Name U -Namespace W -PassThru | Out-Null
    [W.U]::mouse_event(0x0002, 0, 0, 0, 0)
    [W.U]::mouse_event(0x0004, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 400
    $expand.Collapse()
    Start-Sleep -Milliseconds 200
    return $want
}

$proc = Start-Process -FilePath (Join-Path $sandbox "chess.exe") -WorkingDirectory $sandbox -PassThru
Write-Output ("launched pid={0}" -f $proc.Id)

# 等启动加载完成 (按钮变可用)
$deadline = (Get-Date).AddSeconds(120)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $b = Find-ByName "开始对弈"
    if ($null -ne $b -and $b.Current.IsEnabled) { $ready = $true; break }
    Start-Sleep -Milliseconds 400
}
Write-Output ("startup complete = {0}" -f $ready)
if (-not $ready) {
    try { $proc.Kill() } catch { }
    Write-Output "RESULT: FAIL (启动未完成)"
    exit 1
}

# A = combo#1 item 7 (PPO+MCTS, TB 专家), B = combo#2 item 7 -> 两个都会学
Write-Output ("A side = " + (Select-ComboItem 1 7))
Start-Sleep -Milliseconds 300
Write-Output ("B side = " + (Select-ComboItem 2 7))
Start-Sleep -Milliseconds 300
Set-Games 1
Start-Sleep -Milliseconds 300

$startBtn = Find-ByName "开始对弈"
if ($null -eq $startBtn) { try { $proc.Kill() } catch { }; throw "start button not found" }
$startBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Write-Output "match started"

# 判据 1: 按钮文字变成"停止对弈" = 对弈真的开起来了
Start-Sleep -Seconds 5
$running = ($null -ne (Find-ByName "停止对弈"))
Write-Output ("match running = {0}" -f $running)

# 判据 2: 等它结束 (按钮变回"开始对弈") 或超时后手动停止
$end = (Get-Date).AddSeconds($MaxWaitSec)
$finished = $false
while ((Get-Date) -lt $end) {
    if ($null -ne (Find-ByName "开始对弈")) { $finished = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $finished) {
    Write-Output "超时: 手动点'停止对弈' (中止也会走静默保存那条路)"
    $stopBtn = Find-ByName "停止对弈"
    if ($null -ne $stopBtn) {
        $stopBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    }
    Start-Sleep -Seconds 25
    $finished = ($null -ne (Find-ByName "开始对弈"))
}
Write-Output ("match finished = {0}" -f $finished)

# 等保存落盘 (权重 500 MB 量级, 写完要几秒)
$weightDir = Join-Path $sandbox "weights"
$saved = @()
$saveDeadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $saveDeadline) {
    $saved = @(Get-ChildItem $weightDir -File -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -notlike "_temp*" -and $_.Length -gt 0 })
    if ($saved.Count -gt 0) { break }
    Start-Sleep -Seconds 3
}

Write-Output ("weights/ 里的正式权重文件: {0} 个" -f $saved.Count)
foreach ($f in $saved) {
    Write-Output ("  {0}  ({1:N1} MB)" -f $f.Name, ($f.Length / 1MB))
}

# 逐局明细最后几行 (记录了保存结果)
$items = All-ListItems
Write-Output "逐局明细(尾部):"
$tail = if ($items.Count -gt 6) { $items[($items.Count - 6)..($items.Count - 1)] } else { $items }
foreach ($t in $tail) { Write-Output ("  " + $t) }

try { $proc.CloseMainWindow() | Out-Null } catch { }
Start-Sleep -Seconds 3
if (-not $proc.HasExited) { $proc.Kill() }

$ok = $running -and $finished -and ($saved.Count -gt 0)
Write-Output ("判据: 开起来了={0} 结束了={1} 权重已落盘={2}" -f $running, $finished, ($saved.Count -gt 0))
if ($ok) { Write-Output "RESULT: PASS" } else { Write-Output "RESULT: FAIL" }
Write-Output ("sandbox 保留供人工查看: {0}" -f $sandbox)
