# tools/verify_human_vs_ai.ps1
#
# 直接测 chess.exe 的**人机对弈**: 脚本自己执红走棋, 走到终局, 再点"开局"重来,
# 每走一步都把界面上的文字/弹窗抓出来 —— 用来定位用户报障:
#   "人机对弈, 黑方赢了之后, 再次开局沙漏显示模型未加载, 红方下棋子后黑方进入无限等待"
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File tools/verify_human_vs_ai.ps1
#   powershell ... -File tools/verify_human_vs_ai.ps1 -AgentIndex 7 -MaxMoves 60
#
# 编码: 含中文字面量 ⇒ 必须 UTF-8 with BOM。

param(
    [string]$Exe = "",
    [int]$AgentIndex = 7,      # 对战AI 下拉框的序号 (7 = PPO+MCTS)
    [int]$MaxMoves = 60,
    [int]$AiWaitSec = 90
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -MemberDefinition @'
[DllImport("user32.dll")] public static extern void mouse_event(int f,int x,int y,int d,int e);
[DllImport("user32.dll")] public static extern bool SetCursorPos(int x,int y);
'@ -Name M -Namespace W

if ($Exe -eq "") {
    $Exe = Join-Path $PSScriptRoot "..\build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\chess.exe"
}
$Exe = (Resolve-Path $Exe).Path
$sandbox = Join-Path $env:TEMP ("chess_hvai_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $sandbox | Out-Null
Copy-Item $Exe (Join-Path $sandbox "chess.exe") -Force
Write-Output ("sandbox = {0}" -f $sandbox)

$root = [System.Windows.Automation.AutomationElement]::RootElement
function Find-ByName([string]$name) {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $c)
}
function All-Texts {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Text)
    $all = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $c)
    $o = @(); foreach ($t in $all) { $o += $t.Current.Name }
    return $o
}
function All-ListItems {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $all = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $c)
    $o = @(); foreach ($t in $all) { $o += $t.Current.Name }
    return $o
}
# 点屏幕坐标
function Click-At([int]$x, [int]$y) {
    [W.M]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 80
    [W.M]::mouse_event(0x0002, 0, 0, 0, 0)
    [W.M]::mouse_event(0x0004, 0, 0, 0, 0)
}
# 棋盘几何: ChessBoard 的 offsetX/offsetY = 50, gridSize = 60 (见 chessboard.h)
function Board-Point([int]$col, [int]$row) {
    return @(($script:boardLeft + 50 + $col * 60), ($script:boardTop + 50 + $row * 60))
}

$proc = Start-Process -FilePath (Join-Path $sandbox "chess.exe") -WorkingDirectory $sandbox -PassThru
Write-Output ("launched pid={0}" -f $proc.Id)
Start-Sleep -Seconds 14

# 棋盘面板的位置。
# 注意: ChessBoard 是自定义控件 (native="true"), UIA 不一定给它 Name —— 按名字找不到时
# 先试 AutomationId=gameWidget, 再退到"主窗口矩形"(棋盘在窗口左上角的布局里)。
$boardEl = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants,
    (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, "gameWidget")))
if ($null -eq $boardEl) {
    Write-Output "[warn] 找不到 gameWidget 的 AutomationId, 退到主窗口矩形推算"
    $winEl = $root.FindFirst([System.Windows.Automation.TreeScope]::Children,
        (New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)))
    if ($null -eq $winEl) { throw "找不到主窗口" }
    $wr = $winEl.Current.BoundingRectangle
    Write-Output ("main window rect = {0}" -f $wr)
    $script:boardLeft = [int]$wr.X
    $script:boardTop  = [int]$wr.Y
} else {
    $br = $boardEl.Current.BoundingRectangle
    Write-Output ("board rect = {0}" -f $br)
    $script:boardLeft = [int]$br.X
    $script:boardTop  = [int]$br.Y
}
Write-Output ("board origin = ({0},{1})" -f $script:boardLeft, $script:boardTop)

# 选 AI (对战AI = combo#0)
$c = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::ComboBox)
$combos = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $c)
$cb = $combos.Item(0)
$expand = $null
$cb.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$expand) | Out-Null
$expand.Expand(); Start-Sleep -Milliseconds 500
$li = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::ListItem)
$items = $cb.FindAll([System.Windows.Automation.TreeScope]::Descendants, $li)
$target = $items.Item($AgentIndex)
$r = $target.Current.BoundingRectangle
Click-At ([int]($r.X + $r.Width / 2)) ([int]($r.Y + $r.Height / 2))
Start-Sleep -Milliseconds 500
$expand.Collapse(); Start-Sleep -Milliseconds 400
Write-Output ("AI agent = {0}" -f $cb.Current.Name)

# 红方走棋: 依次尝试所有 (from,to) 组合, 直到"AI 开始思考"或终局
$moves = 0
for ($attempt = 0; $attempt -lt $MaxMoves; $attempt++) {
    $played = $false
    for ($fx = 0; $fx -lt 9 -and -not $played; $fx++) {
        for ($fy = 0; $fy -lt 10 -and -not $played; $fy++) {
            $p = Board-Point $fx $fy
            Click-At $p[0] $p[1]
            Start-Sleep -Milliseconds 60
            # 试探一个落点 (先试正前方/横/斜几个常见方向之一: 用 (fx,fy-1) 与 (fx+1,fy))
            foreach ($d in @(@(0,-1), @(0,1), @(1,0), @(-1,0), @(1,1), @(-1,-1), @(1,-1), @(-1,1), @(0,-2), @(0,2), @(2,0), @(-2,0))) {
                $tx = $fx + $d[0]; $ty = $fy + $d[1]
                if ($tx -lt 0 -or $tx -gt 8 -or $ty -lt 0 -or $ty -gt 9) { continue }
                $q = Board-Point $tx $ty
                Click-At $q[0] $q[1]
                Start-Sleep -Milliseconds 120
                # 若 AI 开始思考 / 出现终局弹窗, 说明这一步走成了
                if ($null -ne (Find-ByName "黑方胜!") -or $null -ne (Find-ByName "红方胜!") -or
                    $null -ne (Find-ByName "平局!") -or ($null -ne (Find-ByName "思考中")) -or
                    (All-Texts | Where-Object { $_ -like "*正在思考*" }).Count -gt 0) {
                    $played = $true
                    break
                }
                # 没走成 -> 取消选择再试
                Click-At $p[0] $p[1]
                Start-Sleep -Milliseconds 60
            }
        }
    }
    if (-not $played) { Write-Output ("  第 {0} 步: 找不到可走的红方着法 (棋盘状态变了?)" -f ($attempt+1)); break }
    $moves++
    Write-Output ("  红方第 {0} 步已走" -f $moves)

    # 等 AI 走完 (最多 AiWaitSec)
    $end = (Get-Date).AddSeconds($AiWaitSec)
    $aiDone = $false
    while ((Get-Date) -lt $end) {
        $texts = All-Texts
        if (($texts | Where-Object { $_ -like "*AI思考时间*" }).Count -gt 0) { $aiDone = $true; break }
        $dlg = Find-ByName "黑方胜!"
        if ($null -ne $dlg) { Write-Output "  ** 终局: 黑方胜 **"; break }
        $dlg2 = Find-ByName "红方胜!"
        if ($null -ne $dlg2) { Write-Output "  ** 终局: 红方胜 **"; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $aiDone) { Write-Output ("  第 {0} 步后 AI 等待超时 (可能卡住)" -f $moves) }

    # 关掉终局弹窗 (回车/ESC)
    $dlg = Find-ByName "黑方胜!"
    if ($null -ne $dlg) {
        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        Start-Sleep -Seconds 1
        break
    }
}

Write-Output ("`n=== 走完 {0} 步后的界面状态 ===" -f $moves)
Write-Output "全部文字:"
foreach ($t in (All-Texts | Select-Object -Unique)) { Write-Output ("  | " + $t) }
Write-Output "`n逐局明细/列表:"
foreach ($t in (All-ListItems | Select-Object -Last 12)) { Write-Output ("  | " + $t) }

# 点"开局"再来一局: 看是否卡
$reset = Find-ByName "开局"
if ($null -ne $reset) {
    $rr = $reset.Current.BoundingRectangle
    Click-At ([int]($rr.X + $rr.Width/2)) ([int]($rr.Y + $rr.Height/2))
    Write-Output "`n已点击 [开局], 等 20 s 看状态..."
    Start-Sleep -Seconds 20
    Write-Output "开局后全部文字:"
    foreach ($t in (All-Texts | Select-Object -Unique)) { Write-Output ("  | " + $t) }
    # 再走一步红方, 看黑方会不会卡
    Write-Output "`n开局后再走一步红方..."
    for ($fx = 0; $fx -lt 9; $fx++) {
        for ($fy = 6; $fy -lt 10; $fy++) {
            $p = Board-Point $fx $fy
            Click-At $p[0] $p[1]; Start-Sleep -Milliseconds 80
            $q = Board-Point $fx ($fy-1)
            Click-At $q[0] $q[1]; Start-Sleep -Milliseconds 600
            $t = All-Texts
            if (($t | Where-Object { $_ -like "*思考*" }).Count -gt 0) { break }
            Click-At $p[0] $p[1]; Start-Sleep -Milliseconds 80
        }
    }
    Write-Output "等 90 s 看黑方有没有走子 (界面文字变化 = 它在动):"
    $before = (All-Texts) -join "|"
    Start-Sleep -Seconds 90
    $after = (All-Texts) -join "|"
    Write-Output ("  界面文字是否变化 = {0}" -f ($before -ne $after))
    foreach ($t in (All-Texts | Select-Object -Unique)) { Write-Output ("  | " + $t) }
}

Write-Output "`n=== weights/ ==="
Get-ChildItem (Join-Path $sandbox "weights") -File -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Output ("  {0} ({1:N1} MB)" -f $_.Name, ($_.Length/1MB)) }

try { $proc.Kill() } catch { }
Write-Output ("sandbox 保留: {0}" -f $sandbox)
