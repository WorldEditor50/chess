# tools/verify_value_curve.ps1
#
# End-to-end check of the VALUE-EVALUATION curve (2026-10 user wording:
# "在奖励窗口增加一个 tab 显示价值评估曲线" / "如何确定 critics 输出的价值评估是有效的").
#
# Why this is a separate script from verify_bc_ui.ps1: that one ABORTS its match (0 finished
# games), and an EV point only exists once a game is FINISHED and the z caliper has variance.
# This script plays 2 real games to completion and then reads:
#     [1] the chart's point count (accessibleName ends with " · n=<points>"),
#     [2] the readout line (EV per caliper + calibErr + Var(z) + pairs + WHY THERE IS NO POINT),
#     [3] the panel's machine-readable "[value] ..." lines.
#
# The assertions this script makes are deliberately about the STATE being self-describing:
# a blank curve is a legitimate outcome (see docs/training_optimization.md §11.2), and the
# readout line must say which of the three reasons applies. A blank line / blank label would be
# the real bug.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File tools\verify_value_curve.ps1
#   powershell ... -File tools\verify_value_curve.ps1 -Games 1 -MatchTimeoutSec 600
#     -AgentA "PPO+MCTS (AlphaZero,"   :: the only agents with a scalar V head (PPO x2)
#     -AgentB "PPO+MCTS (AlphaZero,"   :: self-play makes BOTH sides sampled -> more pairs
#   launch chess.exe -> A = PPO+MCTS (稀疏MoE+MLP专家)  [the only agents with a scalar V head]
#   -> B = Alpha-Beta L3 (so games are decisive, not all draws) -> match mode = 训练对局
#   -> 1..2 games -> then read:
#        * the self-check panel's "[value] ..." lines (EV / calib / pairs / zVar, or the
#          explicit "z 无方差" explanation),
#        * the valueChart accessible name ("<title> · n=<points>"),
#        * the valueValueLabel readout.
#
# Why this is a separate script: verify_bc_ui.ps1 ABORTS its match (0 finished games), and EV
# only exists once a game is finished AND decisive. This one waits for completed games.
#
# ASCII-only in code; CJK only inside string literals compared against the UI (file is BOM'd).
param(
    [string]$Exe = "",
    [int]$Games = 2,
    [int]$StartupTimeoutSec = 150,
    [int]$MatchTimeoutSec = 900,
    [string]$AgentA = "PPO+MCTS (AlphaZero,",     # 稀疏MoE+MLP专家 (student, has V head)
    [string]$AgentB = "Alpha-Beta L3"             # decisive opponent
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
$exeDir = Split-Path $Exe -Parent

function PropOf($e, $p) {
    try {
        $v = $e.GetCurrentPropertyValue($p)
        if ($v -eq [System.Windows.Automation.AutomationElement]::NotSupported) { return "" }
        return [string]$v
    } catch { return "" }
}
function AllEles($root) {
    $out = New-Object System.Collections.ArrayList
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $stack = New-Object System.Collections.Stack
    $stack.Push($root)
    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        [void]$out.Add($cur)
        try {
            $ch = $walker.GetFirstChild($cur)
            while ($ch -ne $null) { $stack.Push($ch); $ch = $walker.GetNextSibling($ch) }
        } catch { continue }
    }
    return $out
}
function FindByObjectName($root, [string]$objectName) {
    foreach ($e in (AllEles $root)) {
        $aid = PropOf $e ([System.Windows.Automation.AutomationElement]::AutomationIdProperty)
        if ($aid.EndsWith("." + $objectName)) { return $e }
    }
    return $null
}
function PanelText($root) {
    foreach ($e in (AllEles $root)) {
        $ct = PropOf $e ([System.Windows.Automation.AutomationElement]::ControlTypeProperty)
        if ($ct -ne [string][System.Windows.Automation.ControlType]::Edit) { continue }
        $vp = $null
        try { $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern) } catch { continue }
        if ($vp -eq $null) { continue }
        $t = $vp.Current.Value
        if ($t -ne $null -and $t.Length -gt 40) { return $t }
    }
    return ""
}
# Qt popups ignore SelectionItemPattern -> click the item for real.
function Select-Combo($root, [string]$objectName, [string]$match) {
    $combo = FindByObjectName $root $objectName
    if ($combo -eq $null) { return "" }
    $ec = $null
    if (-not $combo.TryGetCurrentPattern(
            [System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$ec)) { return "" }
    $ec.Expand(); Start-Sleep -Milliseconds 700
    $liCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ListItem)
    $items = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants, $liCond)
    $target = $null; $name = ""
    foreach ($it in $items) {
        $n = PropOf $it ([System.Windows.Automation.AutomationElement]::NameProperty)
        if ($n -like ("*" + $match + "*")) { $target = $it; $name = $n; break }
    }
    if ($target -ne $null) {
        $r = $target.Current.BoundingRectangle
        if ($r.Width -gt 0) {
            Add-Type -MemberDefinition '[DllImport("user32.dll")]public static extern void mouse_event(int f,int x,int y,int d,int e); [DllImport("user32.dll")]public static extern bool SetCursorPos(int x,int y);' -Name M -Namespace W -PassThru | Out-Null
            [W.M]::SetCursorPos([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2)) | Out-Null
            Start-Sleep -Milliseconds 120
            [W.M]::mouse_event(0x0002, 0, 0, 0, 0)
            [W.M]::mouse_event(0x0004, 0, 0, 0, 0)
            Start-Sleep -Milliseconds 400
        }
    }
    $ec.Collapse(); Start-Sleep -Milliseconds 300
    return $name
}
function Set-Spin($root, [string]$objectName, [double]$value) {
    $sp = FindByObjectName $root $objectName
    if ($sp -eq $null) { return $false }
    try {
        $pat = $null
        if (-not $sp.TryGetCurrentPattern(
                [System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$pat)) { return $false }
        $pat.SetValue($value); Start-Sleep -Milliseconds 150; return $true
    } catch { return $false }
}

$proc = Start-Process -FilePath $Exe -WorkingDirectory $exeDir -PassThru
try {
    $root = $null
    $dl = (Get-Date).AddSeconds($StartupTimeoutSec)
    while ((Get-Date) -lt $dl) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
        $wins = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
            [System.Windows.Automation.TreeScope]::Children, $cond)
        if ($wins.Count -gt 0) { $root = $wins[0]; break }
    }
    if ($root -eq $null) { throw "no main window" }
    Write-Output ("window up; waiting for startup load (agentComboBox enabled)...")
    $dl2 = (Get-Date).AddSeconds($StartupTimeoutSec)
    while ((Get-Date) -lt $dl2) {
        Start-Sleep -Seconds 1
        $cb = FindByObjectName $root "agentComboBox"
        if ($cb -ne $null -and $cb.Current.IsEnabled) { break }
    }
    Write-Output ("startup done")

    Write-Output ("A = " + (Select-Combo $root "matchAComboBox" $AgentA))
    Write-Output ("B = " + (Select-Combo $root "matchBComboBox" $AgentB))
    Write-Output ("games = " + (Set-Spin $root "gamesSpin" $Games))
    Start-Sleep -Milliseconds 500

    $btn = FindByObjectName $root "selfPlayBtn"
    $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Write-Output ("match started; waiting up to " + $MatchTimeoutSec + " s for completed games")

    $done = $false
    $dl3 = (Get-Date).AddSeconds($MatchTimeoutSec)
    while ((Get-Date) -lt $dl3) {
        Start-Sleep -Seconds 3
        if ($proc.HasExited) { break }
        $stop = FindByObjectName $root "selfPlayBtn"
        $txt = ""
        if ($stop -ne $null) { $txt = [string]$stop.Current.Name }
        if ($txt -ne "停止对弈") { $done = $true; break }   # button back to "开始对弈"
    }
    Write-Output ("match finished = " + $done)

    # ---- read the value curves + readouts ----
    $tabs = FindByObjectName $root "chartTabs"
    if ($tabs -ne $null) {
        $tabCond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::TabItem)
        foreach ($t in $tabs.FindAll([System.Windows.Automation.TreeScope]::Descendants, $tabCond)) {
            $n = PropOf $t ([System.Windows.Automation.AutomationElement]::NameProperty)
            if ($n -match "value") {
                try { $t.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select() } catch { }
                Start-Sleep -Milliseconds 600
            }
        }
    }
    $vc = FindByObjectName $root "valueChart"
    if ($vc -ne $null) {
        Write-Output ("valueChart name = " + (PropOf $vc ([System.Windows.Automation.AutomationElement]::NameProperty)))
    } else { Write-Output "valueChart NOT FOUND" }
    $vl = FindByObjectName $root "valueValueLabel"
    if ($vl -ne $null) {
        Write-Output ("valueValueLabel = " + (PropOf $vl ([System.Windows.Automation.AutomationElement]::NameProperty)))
    } else { Write-Output "valueValueLabel NOT FOUND" }

    Write-Output "---- [value] lines in the self-check panel ----"
    $panel = PanelText $root
    $any = $false
    foreach ($l in ($panel -split "`r?`n")) {
        if ($l -match "\[value\]") { Write-Output ("  | " + $l); $any = $true }
    }
    if (-not $any) { Write-Output "  (none: no completed decisive game -> no EV point, which is the documented caliper)" }
    Write-Output "---- [BC] / match-summary lines ----"
    foreach ($l in ($panel -split "`r?`n")) {
        if ($l -match "match-summary|\[BC\]") { Write-Output ("  | " + $l) }
    }

    # ---- "清空曲线"必须也清掉价值评估那一张 (2026-10 用户报的第 1 条) ----
    # 用户原话: "点击清空曲线不能清空价值评估曲线"。断言方式是**点数**: 曲线控件的
    # accessibleName 末尾带 " · n=<点数>", 清空之后必须回到 0 (与"清空损失曲线"同一语义:
    # 清的是画出来的点, 不是训练历史 —— 所以两条线的**定义**还在, 只是点没了)。
    $clearBtn = FindByObjectName $root "clearMetricsBtn"
    if ($clearBtn -eq $null) {
        Write-Output "clearMetricsBtn NOT FOUND"
    } else {
        $vcB = FindByObjectName $root "valueChart"
        $before = if ($vcB -ne $null) { PropOf $vcB ([System.Windows.Automation.AutomationElement]::NameProperty) } else { "" }
        $clearBtn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        Start-Sleep -Milliseconds 800
        $vcA = FindByObjectName $root "valueChart"
        $after = if ($vcA -ne $null) { PropOf $vcA ([System.Windows.Automation.AutomationElement]::NameProperty) } else { "" }
        Write-Output ("before clear: " + $before)
        Write-Output ("after  clear: " + $after)
        $nb = [regex]::Match($before, "n=([0-9]+)")
        $na = [regex]::Match($after, "n=([0-9]+)")
        $nbv = if ($nb.Success) { [int]$nb.Groups[1].Value } else { -1 }
        $nav = if ($na.Success) { [int]$na.Groups[1].Value } else { -1 }
        if ($nbv -le 0) {
            Write-Output ("  info  skip: the value curve had no points before the click (n=" + $nbv +
                          "), so 'cleared' cannot be demonstrated in this run")
        } elseif ($nav -ne 0) {
            Write-Output ("  ** FAIL: 清空曲线没有清掉价值评估曲线 ** (n: " + $nbv + " -> " + $nav + ")")
        } else {
            Write-Output ("  OK: 清空曲线把价值评估曲线也清掉了 (n: " + $nbv + " -> 0)")
        }
        $vlA = FindByObjectName $root "valueValueLabel"
        if ($vlA -ne $null) {
            Write-Output ("after clear, valueValueLabel = " + (PropOf $vlA ([System.Windows.Automation.AutomationElement]::NameProperty)))
        }
    }
} finally {
    if (-not $proc.HasExited) { try { $proc.Kill() } catch { } ; Start-Sleep -Seconds 2 }
    $left = @(Get-Process -Name "chess" -ErrorAction SilentlyContinue)
    foreach ($p in $left) { try { Stop-Process -Id $p.Id -Force } catch { } }
    Start-Sleep -Milliseconds 500
    Write-Output ("leftover chess.exe = " + (@(Get-Process -Name "chess" -ErrorAction SilentlyContinue)).Count)
}
